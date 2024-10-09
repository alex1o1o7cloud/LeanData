import Mathlib

namespace fibonacci_money_problem_l1734_173470

variable (x : ℕ)

theorem fibonacci_money_problem (h : 0 < x - 6) (eq_amounts : 90 / (x - 6) = 120 / x) : 
    90 / (x - 6) = 120 / x :=
sorry

end fibonacci_money_problem_l1734_173470


namespace person_dining_minutes_l1734_173446

theorem person_dining_minutes
  (initial_angle : ℕ)
  (final_angle : ℕ)
  (time_spent : ℕ)
  (minute_angle_per_minute : ℕ)
  (hour_angle_per_minute : ℕ)
  (h1 : initial_angle = 110)
  (h2 : final_angle = 110)
  (h3 : minute_angle_per_minute = 6)
  (h4 : hour_angle_per_minute = minute_angle_per_minute / 12)
  (h5 : time_spent = (final_angle - initial_angle) / (minute_angle_per_minute / (minute_angle_per_minute / 12) - hour_angle_per_minute)) :
  time_spent = 40 := sorry

end person_dining_minutes_l1734_173446


namespace orchestra_musicians_l1734_173441

theorem orchestra_musicians : ∃ (m n : ℕ), (m = n^2 + 11) ∧ (m = n * (n + 5)) ∧ m = 36 :=
by {
  sorry
}

end orchestra_musicians_l1734_173441


namespace abe_job_time_l1734_173462

theorem abe_job_time (A G C: ℕ) : G = 70 → C = 21 → (1 / G + 1 / A = 1 / C) → A = 30 := by
sorry

end abe_job_time_l1734_173462


namespace shaded_area_l1734_173401

-- Definition for the conditions provided in the problem
def side_length := 6
def area_square := side_length ^ 2
def area_square_unit := area_square * 4

-- The problem and proof statement
theorem shaded_area (sl : ℕ) (asq : ℕ) (nsq : ℕ):
    sl = 6 ∧
    asq = sl ^ 2 ∧
    nsq = asq * 4 →
    nsq - (4 * (sl^2 / 2)) = 72 :=
by
  sorry

end shaded_area_l1734_173401


namespace value_of_b_over_a_l1734_173490

def rectangle_ratio (a b : ℝ) : Prop :=
  let d := Real.sqrt (a^2 + b^2)
  let P := 2 * (a + b)
  (b / d) = (d / (a + b))

theorem value_of_b_over_a (a b : ℝ) (h : rectangle_ratio a b) : b / a = 1 :=
by sorry

end value_of_b_over_a_l1734_173490


namespace magnitude_range_l1734_173483

noncomputable def vector_a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def vector_b : ℝ × ℝ := (Real.sqrt 3, -1)

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem magnitude_range (θ : ℝ) : 
  0 ≤ (vector_magnitude (2 • vector_a θ - vector_b)) ∧ (vector_magnitude (2 • vector_a θ - vector_b)) ≤ 4 := 
sorry

end magnitude_range_l1734_173483


namespace minimum_value_l1734_173482

/-- The minimum value of the expression (x+2)^2 / (y-2) + (y+2)^2 / (x-2)
    for real numbers x > 2 and y > 2 is 50. -/
theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ z, z = (x + 2) ^ 2 / (y - 2) + (y + 2) ^ 2 / (x - 2) ∧ z = 50 :=
sorry

end minimum_value_l1734_173482


namespace henrys_distance_from_start_l1734_173461

noncomputable def meters_to_feet (x : ℝ) : ℝ := x * 3.281
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem henrys_distance_from_start :
  let west_walk_feet := meters_to_feet 15
  let north_walk_feet := 60
  let east_walk_feet := 156
  let south_walk_meter_backwards := 30
  let south_walk_feet_backwards := 12
  let total_south_feet := meters_to_feet south_walk_meter_backwards + south_walk_feet_backwards
  let net_south_feet := total_south_feet - north_walk_feet
  let net_east_feet := east_walk_feet - west_walk_feet
  distance 0 0 net_east_feet (-net_south_feet) = 118 := 
by
  sorry

end henrys_distance_from_start_l1734_173461


namespace area_of_triangle_is_sqrt3_l1734_173435

theorem area_of_triangle_is_sqrt3
  (a b c : ℝ)
  (B : ℝ)
  (h_geom_prog : b^2 = a * c)
  (h_b : b = 2)
  (h_B : B = Real.pi / 3) :
  (1/2) * a * c * Real.sin B = Real.sqrt 3 := 
by
  sorry

end area_of_triangle_is_sqrt3_l1734_173435


namespace total_population_is_700_l1734_173418

-- Definitions for the problem conditions
def L : ℕ := 200
def P : ℕ := L / 2
def E : ℕ := (L + P) / 2
def Z : ℕ := E + P

-- Proof statement (with sorry)
theorem total_population_is_700 : L + P + E + Z = 700 :=
by
  sorry

end total_population_is_700_l1734_173418


namespace sara_picked_peaches_l1734_173489

def peaches_original : ℕ := 24
def peaches_now : ℕ := 61
def peaches_picked (p_o p_n : ℕ) : ℕ := p_n - p_o

theorem sara_picked_peaches : peaches_picked peaches_original peaches_now = 37 :=
by
  sorry

end sara_picked_peaches_l1734_173489


namespace trig_identity_cosine_powers_l1734_173459

theorem trig_identity_cosine_powers :
  12 * (Real.cos (Real.pi / 8)) ^ 4 + 
  (Real.cos (3 * Real.pi / 8)) ^ 4 + 
  (Real.cos (5 * Real.pi / 8)) ^ 4 + 
  (Real.cos (7 * Real.pi / 8)) ^ 4 = 
  3 / 2 := 
  sorry

end trig_identity_cosine_powers_l1734_173459


namespace magic_square_solution_l1734_173497

theorem magic_square_solution (d e k f g h x y : ℤ)
  (h1 : x + 4 + f = 87 + d + f)
  (h2 : x + d + h = 87 + e + h)
  (h3 : x + y + 87 = 4 + d + e)
  (h4 : f + g + h = x + y + 87)
  (h5 : d = x - 83)
  (h6 : e = 2 * x - 170)
  (h7 : y = 3 * x - 274)
  (h8 : f = g)
  (h9 : g = h) :
  x = 62 ∧ y = -88 :=
by
  sorry

end magic_square_solution_l1734_173497


namespace initial_investment_proof_l1734_173400

noncomputable def initial_investment (A : ℝ) (r t : ℕ) : ℝ := 
  A / (1 + r / 100) ^ t

theorem initial_investment_proof : 
  initial_investment 1000 8 8 = 630.17 := sorry

end initial_investment_proof_l1734_173400


namespace lisa_flew_distance_l1734_173458

-- Define the given conditions
def speed := 32  -- speed in miles per hour
def time := 8    -- time in hours

-- Define the derived distance
def distance := speed * time  -- using the formula Distance = Speed × Time

-- Prove that the calculated distance is 256 miles
theorem lisa_flew_distance : distance = 256 :=
by
  sorry

end lisa_flew_distance_l1734_173458


namespace range_of_m_l1734_173414

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_m (h1 : ∀ x : ℝ, f (-x) = f x)
                   (h2 : ∀ a b : ℝ, a ≠ b → a ≤ 0 → b ≤ 0 → (f a - f b) / (a - b) < 0)
                   (h3 : f (m + 1) < f 2) : 
  ∃ m : ℝ, -3 < m ∧ m < 1 :=
sorry

end range_of_m_l1734_173414


namespace students_not_good_at_either_l1734_173448

theorem students_not_good_at_either (total good_at_english good_at_chinese both_good : ℕ) 
(h₁ : total = 45) 
(h₂ : good_at_english = 35) 
(h₃ : good_at_chinese = 31) 
(h₄ : both_good = 24) : total - (good_at_english + good_at_chinese - both_good) = 3 :=
by sorry

end students_not_good_at_either_l1734_173448


namespace log_base_change_l1734_173425

-- Define the conditions: 8192 = 2 ^ 13 and change of base formula
def x : ℕ := 8192
def a : ℕ := 2
def n : ℕ := 13
def b : ℕ := 5

theorem log_base_change (log : ℕ → ℕ → ℝ) 
  (h1 : x = a ^ n) 
  (h2 : ∀ (x b c: ℕ), c ≠ 1 → log x b = (log x c) / (log b c) ): 
  log x b = 13 / (log 5 2) :=
by
  sorry

end log_base_change_l1734_173425


namespace lcm_24_36_45_l1734_173426

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end lcm_24_36_45_l1734_173426


namespace euler_no_k_divisible_l1734_173411

theorem euler_no_k_divisible (n : ℕ) (k : ℕ) (h : k < 5^n - 5^(n-1)) : ¬ (5^n ∣ 2^k - 1) := 
sorry

end euler_no_k_divisible_l1734_173411


namespace composite_function_l1734_173451

def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x + 1

theorem composite_function : ∀ (x : ℝ), f (g x) = 2 * x + 1 :=
by
  intro x
  sorry

end composite_function_l1734_173451


namespace parabola_equation_l1734_173481

-- Define the conditions of the problem
def parabola_vertex := (0, 0)
def parabola_focus_x_axis := true
def line_eq (x y : ℝ) : Prop := x = y
def midpoint_of_AB (x1 y1 x2 y2 mx my: ℝ) : Prop := (mx, my) = ((x1 + x2) / 2, (y1 + y2) / 2)
def point_P := (1, 1)

theorem parabola_equation (A B : ℝ × ℝ) :
  (parabola_vertex = (0, 0)) →
  (parabola_focus_x_axis) →
  (line_eq A.1 A.2) →
  (line_eq B.1 B.2) →
  midpoint_of_AB A.1 A.2 B.1 B.2 point_P.1 point_P.2 →
  A = (0, 0) ∨ B = (0, 0) →
  B = A ∨ A = (0, 0) → B = (2, 2) →
  ∃ a, ∀ x y, y^2 = a * x → a = 2 :=
sorry

end parabola_equation_l1734_173481


namespace smallest_sum_of_20_consecutive_integers_twice_perfect_square_l1734_173452

theorem smallest_sum_of_20_consecutive_integers_twice_perfect_square :
  ∃ n : ℕ, ∃ k : ℕ, (∀ m : ℕ, m ≥ n → 0 < m) ∧ 10 * (2 * n + 19) = 2 * k^2 ∧ 10 * (2 * n + 19) = 450 :=
by
  sorry

end smallest_sum_of_20_consecutive_integers_twice_perfect_square_l1734_173452


namespace count_valid_numbers_is_31_l1734_173475

def is_valid_digit (n : Nat) : Prop := n = 0 ∨ n = 2 ∨ n = 6 ∨ n = 8

def count_valid_numbers : Nat :=
  let valid_digits := [0, 2, 6, 8]
  let one_digit := valid_digits.filter (λ n => n % 4 = 0)
  let two_digits := valid_digits.product valid_digits |>.filter (λ (a, b) => (10*a + b) % 4 = 0)
  let three_digits := valid_digits.product two_digits |>.filter (λ (a, (b, c)) => (100*a + 10*b + c) % 4 = 0)
  one_digit.length + two_digits.length + three_digits.length

theorem count_valid_numbers_is_31 : count_valid_numbers = 31 := by
  sorry

end count_valid_numbers_is_31_l1734_173475


namespace opposite_of_neg_quarter_l1734_173415

theorem opposite_of_neg_quarter : -(- (1/4 : ℝ)) = (1/4 : ℝ) :=
by
  sorry

end opposite_of_neg_quarter_l1734_173415


namespace first_representation_second_representation_third_representation_l1734_173429

theorem first_representation :
  1 + 2 + 3 + 4 + 5 + 6 + 7 + (8 * 9) = 100 := 
by 
  sorry

theorem second_representation:
  1 + 2 + 3 + 47 + (5 * 6) + 8 + 9 = 100 :=
by
  sorry

theorem third_representation:
  1 + 2 + 3 + 4 + 5 - 6 - 7 + 8 + 92 = 100 := 
by
  sorry

end first_representation_second_representation_third_representation_l1734_173429


namespace rhombus_perimeter_l1734_173439

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 72) (h2 : d2 = 30) 
  (h3 : ∀ {x y : ℝ}, (x = d1 / 2 ∧ y = d2 / 2) → (x^2 + y^2 = (d1 / 2)^2 + (d2 / 2)^2)) : 
  4 * (Real.sqrt ((d1/2)^2 + (d2/2)^2)) = 156 :=
by 
  rw [h1, h2]
  simp
  sorry

end rhombus_perimeter_l1734_173439


namespace find_blue_weights_l1734_173493

theorem find_blue_weights (B : ℕ) :
  (2 * B + 15 + 2 = 25) → B = 4 :=
by
  intro h
  sorry

end find_blue_weights_l1734_173493


namespace men_in_second_group_l1734_173465

theorem men_in_second_group (M : ℕ) (h1 : 16 * 30 = 480) (h2 : M * 24 = 480) : M = 20 :=
by
  sorry

end men_in_second_group_l1734_173465


namespace problem_a_lt_b_lt_0_implies_ab_gt_b_sq_l1734_173473

theorem problem_a_lt_b_lt_0_implies_ab_gt_b_sq (a b : ℝ) (h : a < b ∧ b < 0) : ab > b^2 := by
  sorry

end problem_a_lt_b_lt_0_implies_ab_gt_b_sq_l1734_173473


namespace alex_score_l1734_173407

theorem alex_score (n : ℕ) (avg19 avg20 alex : ℚ)
  (h1 : n = 20)
  (h2 : avg19 = 72)
  (h3 : avg20 = 74)
  (h_totalscore19 : 19 * avg19 = 1368)
  (h_totalscore20 : 20 * avg20 = 1480)
  (h_alexscore : alex = 112) :
  alex = (1480 - 1368 : ℚ) := 
sorry

end alex_score_l1734_173407


namespace product_of_D_l1734_173491

theorem product_of_D:
  ∀ (D : ℝ × ℝ), 
  (∃ M C : ℝ × ℝ, 
    M.1 = 4 ∧ M.2 = 3 ∧ 
    C.1 = 6 ∧ C.2 = -1 ∧ 
    M.1 = (C.1 + D.1) / 2 ∧ 
    M.2 = (C.2 + D.2) / 2) 
  → (D.1 * D.2 = 14) :=
sorry

end product_of_D_l1734_173491


namespace ab_cd_is_1_or_minus_1_l1734_173484

theorem ab_cd_is_1_or_minus_1 (a b c d : ℤ) (h1 : ∃ k₁ : ℤ, a = k₁ * (a * b - c * d))
  (h2 : ∃ k₂ : ℤ, b = k₂ * (a * b - c * d)) (h3 : ∃ k₃ : ℤ, c = k₃ * (a * b - c * d))
  (h4 : ∃ k₄ : ℤ, d = k₄ * (a * b - c * d)) :
  a * b - c * d = 1 ∨ a * b - c * d = -1 := 
sorry

end ab_cd_is_1_or_minus_1_l1734_173484


namespace no_solution_for_ab_ba_l1734_173486

theorem no_solution_for_ab_ba (a b x : ℕ)
  (ab ba : ℕ)
  (h_ab : ab = 10 * a + b)
  (h_ba : ba = 10 * b + a) :
  (ab^x - 2 = ba^x - 7) → false :=
by
  sorry

end no_solution_for_ab_ba_l1734_173486


namespace farmer_field_area_l1734_173488

variable (x : ℕ) (A : ℕ)

def planned_days : Type := {x : ℕ // 120 * x = 85 * (x + 2) + 40}

theorem farmer_field_area (h : {x : ℕ // 120 * x = 85 * (x + 2) + 40}) : A = 720 :=
by
  sorry

end farmer_field_area_l1734_173488


namespace correct_M_l1734_173413

-- Definition of the function M for calculating the position number
def M (k : ℕ) : ℕ :=
  if k % 2 = 1 then
    4 * k^2 - 4 * k + 2
  else
    4 * k^2 - 2 * k + 2

-- Theorem stating the correctness of the function M
theorem correct_M (k : ℕ) : M k = if k % 2 = 1 then 4 * k^2 - 4 * k + 2 else 4 * k^2 - 2 * k + 2 := 
by
  -- The proof is to be done later.
  -- sorry is used to indicate a placeholder.
  sorry

end correct_M_l1734_173413


namespace find_a_l1734_173423

-- Define given parameters and conditions
def parabola_eq (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

def shifted_parabola_eq (a : ℝ) (x : ℝ) : ℝ := parabola_eq a x - 3 * |a|

-- Define axis of symmetry function
def axis_of_symmetry (a : ℝ) : ℝ := 1

-- Conditions: a ≠ 0
variable (a : ℝ)
variable (h : a ≠ 0)

-- Define value for discriminant check
def discriminant (a : ℝ) (c : ℝ) : ℝ := (-2 * a)^2 - 4 * a * c

-- Problem statement
theorem find_a (ha : a ≠ 0) : 
  (axis_of_symmetry a = 1) ∧ (discriminant a (3 - 3 * |a|) = 0 → (a = 3 / 4 ∨ a = -3 / 2)) := 
by
  sorry -- proof to be filled in

end find_a_l1734_173423


namespace hyperbola_eccentricity_l1734_173466

-- Define the conditions and parameters for the problem
variables (m : ℝ) (c a e : ℝ)

-- Given conditions
def hyperbola_eq (m : ℝ) := ∀ x y : ℝ, (x^2 / m^2 - y^2 = 4)
def focal_distance : Prop := c = 4
def standard_hyperbola_form : Prop := a^2 = 4 * m^2 ∧ 4 = 4

-- Eccentricity definition
def eccentricity : Prop := e = c / a

-- Main theorem
theorem hyperbola_eccentricity (m : ℝ) (h_pos : 0 < m) (h_foc_dist : focal_distance c) (h_form : standard_hyperbola_form a m) :
  eccentricity e a c :=
by
  sorry

end hyperbola_eccentricity_l1734_173466


namespace find_person_10_number_l1734_173460

theorem find_person_10_number (n : ℕ) (a : ℕ → ℕ)
  (h1 : n = 15)
  (h2 : 2 * a 10 = a 9 + a 11)
  (h3 : 2 * a 3 = a 2 + a 4)
  (h4 : a 10 = 8)
  (h5 : a 3 = 7) :
  a 10 = 8 := 
by sorry

end find_person_10_number_l1734_173460


namespace compute_expression_l1734_173443

def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := 2 * x
noncomputable def f_inv (x : ℝ) : ℝ := x - 3
noncomputable def g_inv (x : ℝ) : ℝ := x / 2

theorem compute_expression : 
  f (g_inv (f_inv (f_inv (g (f 15))))) = 18 := by
  sorry

end compute_expression_l1734_173443


namespace brad_start_time_after_maxwell_l1734_173480

-- Assuming time is measured in hours, distance in kilometers, and speed in km/h
def meet_time (d : ℕ) (v_m : ℕ) (v_b : ℕ) (t_m : ℕ) : ℕ :=
  let d_m := t_m * v_m
  let t_b := t_m - 1
  let d_b := t_b * v_b
  d_m + d_b

theorem brad_start_time_after_maxwell (d : ℕ) (v_m : ℕ) (v_b : ℕ) (t_m : ℕ) :
  d = 54 → v_m = 4 → v_b = 6 → t_m = 6 → 
  meet_time d v_m v_b t_m = 54 :=
by
  intros hd hv_m hv_b ht_m
  have : meet_time d v_m v_b t_m = t_m * v_m + (t_m - 1) * v_b := rfl
  rw [hd, hv_m, hv_b, ht_m] at this
  sorry

end brad_start_time_after_maxwell_l1734_173480


namespace lilith_caps_collection_l1734_173408

noncomputable def monthlyCollectionYear1 := 3
noncomputable def monthlyCollectionAfterYear1 := 5
noncomputable def christmasCaps := 40
noncomputable def yearlyCapsLost := 15
noncomputable def totalYears := 5

noncomputable def totalCapsCollectedByLilith :=
  let firstYearCaps := monthlyCollectionYear1 * 12
  let remainingYearsCaps := monthlyCollectionAfterYear1 * 12 * (totalYears - 1)
  let christmasCapsTotal := christmasCaps * totalYears
  let totalCapsBeforeLosses := firstYearCaps + remainingYearsCaps + christmasCapsTotal
  let lostCapsTotal := yearlyCapsLost * totalYears
  let totalCapsAfterLosses := totalCapsBeforeLosses - lostCapsTotal
  totalCapsAfterLosses

theorem lilith_caps_collection : totalCapsCollectedByLilith = 401 := by
  sorry

end lilith_caps_collection_l1734_173408


namespace number_of_petri_dishes_l1734_173422

def germs_in_lab : ℕ := 3700
def germs_per_dish : ℕ := 25
def num_petri_dishes : ℕ := germs_in_lab / germs_per_dish

theorem number_of_petri_dishes : num_petri_dishes = 148 :=
by
  sorry

end number_of_petri_dishes_l1734_173422


namespace estimate_pi_l1734_173499

theorem estimate_pi :
  ∀ (r : ℝ) (side_length : ℝ) (total_beans : ℕ) (beans_in_circle : ℕ),
  r = 1 →
  side_length = 2 →
  total_beans = 80 →
  beans_in_circle = 64 →
  (π = 3.2) :=
by
  intros r side_length total_beans beans_in_circle hr hside htotal hin_circle
  sorry

end estimate_pi_l1734_173499


namespace max_sum_of_digits_l1734_173417

theorem max_sum_of_digits (X Y Z : ℕ) (hX : 1 ≤ X ∧ X ≤ 9) (hY : 1 ≤ Y ∧ Y ≤ 9) (hZ : 1 ≤ Z ∧ Z ≤ 9) (hXYZ : X > Y ∧ Y > Z) : 
  10 * X + 11 * Y + Z ≤ 185 :=
  sorry

end max_sum_of_digits_l1734_173417


namespace arc_length_of_circle_l1734_173432

theorem arc_length_of_circle (r θ : ℝ) (h_r : r = 2) (h_θ : θ = 120) : 
  (θ / 180 * r * Real.pi) = (4 / 3) * Real.pi := by
  sorry

end arc_length_of_circle_l1734_173432


namespace three_digit_numbers_with_properties_l1734_173464

noncomputable def valid_numbers_with_properties : List Nat :=
  [179, 239, 299, 359, 419, 479, 539, 599, 659, 719, 779, 839, 899, 959]

theorem three_digit_numbers_with_properties (N : ℕ) :
  N >= 100 ∧ N < 1000 ∧ 
  N ≡ 1 [MOD 2] ∧
  N ≡ 2 [MOD 3] ∧
  N ≡ 3 [MOD 4] ∧
  N ≡ 4 [MOD 5] ∧
  N ≡ 5 [MOD 6] ↔ N ∈ valid_numbers_with_properties :=
by
  sorry

end three_digit_numbers_with_properties_l1734_173464


namespace arithmetic_seq_and_general_formula_find_Tn_l1734_173421

-- Given definitions
def S : ℕ → ℕ := sorry
def a : ℕ → ℕ := sorry

-- Conditions
axiom a1 : a 1 = 1
axiom a2 : ∀ n : ℕ, n > 0 → n * S n.succ = (n+1) * S n + n^2 + n

-- Problem 1: Prove and derive general formula for Sₙ
theorem arithmetic_seq_and_general_formula (n : ℕ) (h : n > 0) :
  ∃ S : ℕ → ℕ, (∀ n : ℕ, n > 0 → (S (n+1)) / (n+1) - (S n) / n = 1) ∧ (S n = n^2) := sorry

-- Problem 2: Given bₙ and Tₙ, find Tₙ
def b (n : ℕ) : ℕ := 1 / (a n * a (n+1))
def T : ℕ → ℕ := sorry

axiom b1 : ∀ n : ℕ, n > 0 → b 1 = 1
axiom b2 : ∀ n : ℕ, n > 0 → T n = 1 / (2 * n + 1)

theorem find_Tn (n : ℕ) (h : n > 0) : T n = n / (2 * n + 1) := sorry

end arithmetic_seq_and_general_formula_find_Tn_l1734_173421


namespace never_return_to_start_l1734_173436

variable {City : Type} [MetricSpace City]

-- Conditions
variable (C : ℕ → City)  -- C is the sequence of cities
variable (dist : City → City → ℝ)  -- distance function
variable (furthest : City → City)  -- function that maps each city to the furthest city from it
variable (start : City)  -- initial city

-- Assuming C satisfies the properties in the problem statement
axiom initial_city : C 1 = start
axiom furthest_city_step : ∀ n, C (n + 1) = furthest (C n)
axiom no_ambiguity : ∀ c1 c2, (dist c1 (furthest c1) > dist c1 c2 ↔ c2 ≠ furthest c1)

-- Define the problem to prove that if C₁ ≠ C₃, then ∀ n ≥ 4, Cₙ ≠ C₁
theorem never_return_to_start (h : C 1 ≠ C 3) : ∀ n ≥ 4, C n ≠ start := sorry

end never_return_to_start_l1734_173436


namespace smallest_coprime_gt_one_l1734_173438

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l1734_173438


namespace sin_double_angle_plus_pi_over_4_l1734_173402

theorem sin_double_angle_plus_pi_over_4 (α : ℝ) 
  (h : Real.tan α = 3) : 
  Real.sin (2 * α + Real.pi / 4) = -Real.sqrt 2 / 10 := 
by 
  sorry

end sin_double_angle_plus_pi_over_4_l1734_173402


namespace range_of_a_l1734_173474

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a+3) / (5-a)) → -3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l1734_173474


namespace max_value_a_plus_b_plus_c_l1734_173409

-- Definitions used in the problem
def A_n (a n : ℕ) : ℕ := a * (10^n - 1) / 9
def B_n (b n : ℕ) : ℕ := b * (10^n - 1) / 9
def C_n (c n : ℕ) : ℕ := c * (10^(2 * n) - 1) / 9

-- Main statement of the problem
theorem max_value_a_plus_b_plus_c (n : ℕ) (a b c : ℕ) (h : n > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_eq : ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ C_n c n1 - B_n b n1 = 2 * (A_n a n1)^2 ∧ C_n c n2 - B_n b n2 = 2 * (A_n a n2)^2) :
  a + b + c ≤ 18 :=
sorry

end max_value_a_plus_b_plus_c_l1734_173409


namespace plants_remaining_l1734_173450

theorem plants_remaining (plants_initial plants_first_day plants_second_day_eaten plants_third_day_eaten : ℕ)
  (h1 : plants_initial = 30)
  (h2 : plants_first_day = 20)
  (h3 : plants_second_day_eaten = (plants_initial - plants_first_day) / 2)
  (h4 : plants_third_day_eaten = 1)
  : (plants_initial - plants_first_day - plants_second_day_eaten - plants_third_day_eaten) = 4 := 
by
  sorry

end plants_remaining_l1734_173450


namespace prism_volume_is_25_l1734_173463

noncomputable def triangle_area (a b : ℝ) : ℝ := (1 / 2) * a * b

noncomputable def prism_volume (base_area height : ℝ) : ℝ := base_area * height

theorem prism_volume_is_25 :
  let a := Real.sqrt 5
  let base_area := triangle_area a a
  let volume := prism_volume base_area 10
  volume = 25 :=
by
  intros
  sorry

end prism_volume_is_25_l1734_173463


namespace simplify_sqrt_sum_l1734_173467

noncomputable def sqrt_72 : ℝ := Real.sqrt 72
noncomputable def sqrt_32 : ℝ := Real.sqrt 32
noncomputable def sqrt_27 : ℝ := Real.sqrt 27
noncomputable def result : ℝ := 10 * Real.sqrt 2 + 3 * Real.sqrt 3

theorem simplify_sqrt_sum :
  sqrt_72 + sqrt_32 + sqrt_27 = result :=
by
  sorry

end simplify_sqrt_sum_l1734_173467


namespace total_samples_correct_l1734_173471

-- Define the conditions as constants
def samples_per_shelf : ℕ := 65
def number_of_shelves : ℕ := 7

-- Define the total number of samples and the expected result
def total_samples : ℕ := samples_per_shelf * number_of_shelves
def expected_samples : ℕ := 455

-- State the theorem to be proved
theorem total_samples_correct : total_samples = expected_samples := by
  -- Proof to be filled in
  sorry

end total_samples_correct_l1734_173471


namespace not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C_l1734_173469

theorem not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C (A B C : ℝ) (h1 : A = 2 * C) (h2 : B = 2 * C) (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := 
by 
  sorry

end not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C_l1734_173469


namespace seq_2011_l1734_173442

-- Definition of the sequence
def seq (a : ℕ → ℤ) := (a 1 = a 201) ∧ a 201 = 2 ∧ ∀ n : ℕ, a n + a (n + 1) = 0

-- The main theorem to prove that a_2011 = 2
theorem seq_2011 : ∀ a : ℕ → ℤ, seq a → a 2011 = 2 :=
by
  intros a h
  let seq := h
  sorry

end seq_2011_l1734_173442


namespace derivative_and_value_l1734_173457

-- Given conditions
def eqn (x y : ℝ) : Prop := 10 * x^3 + 4 * x^2 * y + y^2 = 0

-- The derivative y'
def y_prime (x y y' : ℝ) : Prop := y' = (-15 * x^2 - 4 * x * y) / (2 * x^2 + y)

-- Specific values derivatives
def y_prime_at_x_neg2_y_4 (y' : ℝ) : Prop := y' = -7 / 3

-- The main theorem
theorem derivative_and_value (x y y' : ℝ) 
  (h1 : eqn x y) (x_neg2 : x = -2) (y_4 : y = 4) : 
  y_prime x y y' ∧ y_prime_at_x_neg2_y_4 y' :=
sorry

end derivative_and_value_l1734_173457


namespace slower_train_speed_l1734_173424

theorem slower_train_speed (v : ℝ) (faster_train_speed : ℝ) (time_pass : ℝ) (train_length : ℝ) :
  (faster_train_speed = 46) →
  (time_pass = 36) →
  (train_length = 50) →
  (v = 36) :=
by
  intro h1 h2 h3
  -- Formal proof goes here
  sorry

end slower_train_speed_l1734_173424


namespace negative_product_implies_negatives_l1734_173437

theorem negative_product_implies_negatives (a b c : ℚ) (h : a * b * c < 0) :
  (∃ n : ℕ, n = 1 ∨ n = 3 ∧ (n = 1 ↔ (a < 0 ∧ b > 0 ∧ c > 0 ∨ a > 0 ∧ b < 0 ∧ c > 0 ∨ a > 0 ∧ b > 0 ∧ c < 0)) ∨ 
                                n = 3 ∧ (n = 3 ↔ (a < 0 ∧ b < 0 ∧ c < 0 ∨ a < 0 ∧ b < 0 ∧ c > 0 ∨ a < 0 ∧ b > 0 ∧ c < 0 ∨ a > 0 ∧ b < 0 ∧ c < 0))) :=
  sorry

end negative_product_implies_negatives_l1734_173437


namespace ratio_fifth_term_l1734_173487

-- Definitions of arithmetic sequences and sums
def arithmetic_seq_sum (a d : ℕ → ℕ) (n : ℕ) : ℕ := n * (2 * a 1 + (n - 1) * d 1) / 2

-- Conditions
variables (S_n S'_n : ℕ → ℕ) (n : ℕ)

-- Given conditions
axiom ratio_sum : ∀ (n : ℕ), S_n n / S'_n n = (5 * n + 3) / (2 * n + 7)
axiom sums_at_9 : S_n 9 = 9 * (S_n 1 + S_n 9) / 2
axiom sums'_at_9 : S'_n 9 = 9 * (S'_n 1 + S'_n 9) / 2

-- Theorem to prove
theorem ratio_fifth_term : (9 * (S_n 1 + S_n 9) / 2) / (9 * (S'_n 1 + S'_n 9) / 2) = 48 / 25 := sorry

end ratio_fifth_term_l1734_173487


namespace final_segment_distance_l1734_173447

theorem final_segment_distance :
  let north_distance := 2
  let east_distance := 1
  let south_distance := 1
  let net_north := north_distance - south_distance
  let net_east := east_distance
  let final_distance := Real.sqrt (net_north ^ 2 + net_east ^ 2)
  final_distance = Real.sqrt 2 :=
by
  sorry

end final_segment_distance_l1734_173447


namespace average_side_length_of_squares_l1734_173472

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l1734_173472


namespace equal_tuesdays_thursdays_l1734_173454

theorem equal_tuesdays_thursdays (days_in_month : ℕ) (tuesdays : ℕ) (thursdays : ℕ) : (days_in_month = 30) → (tuesdays = thursdays) → (∃ (start_days : Finset ℕ), start_days.card = 2) :=
by
  sorry

end equal_tuesdays_thursdays_l1734_173454


namespace solve_for_x_l1734_173403

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : x^3 - 2 * x^2 = 0 ↔ x = 2 :=
by sorry

end solve_for_x_l1734_173403


namespace customers_tipped_count_l1734_173406

variable (initial_customers : ℕ)
variable (added_customers : ℕ)
variable (customers_no_tip : ℕ)

def total_customers (initial_customers added_customers : ℕ) : ℕ :=
  initial_customers + added_customers

theorem customers_tipped_count 
  (h_init : initial_customers = 29)
  (h_added : added_customers = 20)
  (h_no_tip : customers_no_tip = 34) :
  (total_customers initial_customers added_customers - customers_no_tip) = 15 :=
by
  sorry

end customers_tipped_count_l1734_173406


namespace ratio_of_boys_to_girls_l1734_173445

def boys_girls_ratio (b g : ℕ) : ℚ := b / g

theorem ratio_of_boys_to_girls (b g : ℕ) (h1 : b = g + 6) (h2 : g + b = 40) :
  boys_girls_ratio b g = 23 / 17 :=
by
  sorry

end ratio_of_boys_to_girls_l1734_173445


namespace exactly_one_three_digit_perfect_cube_divisible_by_25_l1734_173476

theorem exactly_one_three_digit_perfect_cube_divisible_by_25 :
  ∃! (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ (∃ k : ℕ, n = k^3) ∧ n % 25 = 0 :=
sorry

end exactly_one_three_digit_perfect_cube_divisible_by_25_l1734_173476


namespace cakes_initially_made_l1734_173495

variables (sold bought total initial_cakes : ℕ)

theorem cakes_initially_made (h1 : sold = 105) (h2 : bought = 170) (h3 : total = 186) :
  initial_cakes = total - (sold - bought) :=
by
  rw [h1, h2, h3]
  sorry

end cakes_initially_made_l1734_173495


namespace r_exceeds_s_by_six_l1734_173440

theorem r_exceeds_s_by_six (x y : ℚ) (h1 : 3 * x + 2 * y = 16) (h2 : x + 3 * y = 26 / 5) :
  x - y = 6 := by
  sorry

end r_exceeds_s_by_six_l1734_173440


namespace solve_for_b_l1734_173456

def is_imaginary (z : ℂ) : Prop := z.re = 0

theorem solve_for_b (b : ℝ) (i_is_imag_unit : ∀ (z : ℂ), i * z = z * i):
  is_imaginary (i * (b * i + 1)) → b = 0 :=
by
  sorry

end solve_for_b_l1734_173456


namespace solve_quadratic_and_compute_l1734_173404

theorem solve_quadratic_and_compute (y : ℝ) (h : 4 * y^2 + 7 = 6 * y + 12) : (8 * y - 2)^2 = 248 := 
sorry

end solve_quadratic_and_compute_l1734_173404


namespace expression_in_terms_of_x_difference_between_x_l1734_173410

variable (E x : ℝ)

theorem expression_in_terms_of_x (h1 : E / (2 * x + 15) = 3) : E = 6 * x + 45 :=
by 
  sorry

variable (x1 x2 : ℝ)

theorem difference_between_x (h1 : E / (2 * x1 + 15) = 3) (h2: E / (2 * x2 + 15) = 3) (h3 : x2 - x1 = 12) : True :=
by 
  sorry

end expression_in_terms_of_x_difference_between_x_l1734_173410


namespace find_b_for_intersection_l1734_173478

theorem find_b_for_intersection (b : ℝ) :
  (∀ x : ℝ, bx^2 + 2 * x + 3 = 3 * x + 4 → bx^2 - x - 1 = 0) →
  (∀ x : ℝ, x^2 * b - x - 1 = 0 → (1 + 4 * b = 0) → b = -1/4) :=
by
  intros h_eq h_discriminant h_solution
  sorry

end find_b_for_intersection_l1734_173478


namespace number_of_possible_values_of_a_l1734_173434

theorem number_of_possible_values_of_a :
  ∃ (a_values : Finset ℕ), 
    (∀ a ∈ a_values, 3 ∣ a ∧ a ∣ 18 ∧ a ∣ 27 ∧ 0 < a) ∧
    a_values.card = 2 :=
by
  sorry

end number_of_possible_values_of_a_l1734_173434


namespace lucy_50_cent_items_l1734_173427

theorem lucy_50_cent_items :
  ∃ (a b c : ℕ), a + b + c = 30 ∧ 50 * a + 150 * b + 300 * c = 4500 ∧ a = 6 :=
by
  sorry

end lucy_50_cent_items_l1734_173427


namespace initial_hamburgers_count_is_nine_l1734_173492

-- Define the conditions
def hamburgers_initial (total_hamburgers : ℕ) (additional_hamburgers : ℕ) : ℕ :=
  total_hamburgers - additional_hamburgers

-- The statement to be proved
theorem initial_hamburgers_count_is_nine :
  hamburgers_initial 12 3 = 9 :=
by
  sorry

end initial_hamburgers_count_is_nine_l1734_173492


namespace exists_power_of_two_with_last_n_digits_ones_and_twos_l1734_173494

theorem exists_power_of_two_with_last_n_digits_ones_and_twos (N : ℕ) (hN : 0 < N) :
  ∃ k : ℕ, ∀ i < N, ∃ (d : ℕ), d = 1 ∨ d = 2 ∧ 
    (2^k % 10^N) / 10^i % 10 = d :=
sorry

end exists_power_of_two_with_last_n_digits_ones_and_twos_l1734_173494


namespace trig_identity_example_l1734_173449

open Real

noncomputable def tan_alpha_eq_two_tan_pi_fifths (α : ℝ) :=
  tan α = 2 * tan (π / 5)

theorem trig_identity_example (α : ℝ) (h : tan_alpha_eq_two_tan_pi_fifths α) :
  (cos (α - 3 * π / 10) / sin (α - π / 5)) = 3 :=
sorry

end trig_identity_example_l1734_173449


namespace value_of_a_l1734_173479

theorem value_of_a (a x y : ℤ) (h1 : x = 1) (h2 : y = -3) (h3 : a * x - y = 1) : a = -2 :=
by
  -- Placeholder for the proof
  sorry

end value_of_a_l1734_173479


namespace find_sides_of_triangle_ABC_find_angle_A_l1734_173433

variable (a b c A B C : ℝ)

-- Part (Ⅰ)
theorem find_sides_of_triangle_ABC
  (hC : C = Real.pi / 3)
  (hc : c = 2)
  (hArea : 1/2 * a * b * Real.sin (Real.pi / 3) = Real.sqrt 3) :
  a = 2 ∧ b = 2 := sorry

-- Part (Ⅱ)
theorem find_angle_A
  (hC : C = Real.pi / 3)
  (hc : c = 2)
  (hTrig : Real.sin C + Real.sin (B - A) = 2 * Real.sin (2 * A)) :
  A = Real.pi / 2 ∨ A = Real.pi / 6 := sorry

end find_sides_of_triangle_ABC_find_angle_A_l1734_173433


namespace average_age_at_marriage_l1734_173496

theorem average_age_at_marriage
  (A : ℕ)
  (combined_age_at_marriage : husband_age + wife_age = 2 * A)
  (combined_age_after_5_years : (A + 5) + (A + 5) + 1 = 57) :
  A = 23 := 
sorry

end average_age_at_marriage_l1734_173496


namespace area_of_table_l1734_173420

-- Definitions of the given conditions
def free_side_conditions (L W : ℝ) : Prop :=
  (L = 2 * W) ∧ (2 * W + L = 32)

-- Statement to prove the area of the rectangular table
theorem area_of_table {L W : ℝ} (h : free_side_conditions L W) : L * W = 128 := by
  sorry

end area_of_table_l1734_173420


namespace evaluate_g_at_2_l1734_173428

def g (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem evaluate_g_at_2 : g 2 = 5 :=
by
  sorry

end evaluate_g_at_2_l1734_173428


namespace comparison_of_A_and_B_l1734_173444

noncomputable def A (m : ℝ) : ℝ := Real.sqrt (m + 1) - Real.sqrt m
noncomputable def B (m : ℝ) : ℝ := Real.sqrt m - Real.sqrt (m - 1)

theorem comparison_of_A_and_B (m : ℝ) (h : m > 1) : A m < B m :=
by
  sorry

end comparison_of_A_and_B_l1734_173444


namespace lines_intersect_value_k_l1734_173405

theorem lines_intersect_value_k :
  ∀ (x y k : ℝ), (-3 * x + y = k) → (2 * x + y = 20) → (x = -10) → (k = 70) :=
by
  intros x y k h1 h2 h3
  sorry

end lines_intersect_value_k_l1734_173405


namespace average_distance_to_sides_l1734_173412

open Real

noncomputable def side_length : ℝ := 15
noncomputable def diagonal_distance : ℝ := 9.3
noncomputable def right_turn_distance : ℝ := 3

theorem average_distance_to_sides :
  let d1 := 9.58
  let d2 := 6.58
  let d3 := 5.42
  let d4 := 8.42
  (d1 + d2 + d3 + d4) / 4 = 7.5 :=
by
  let d1 := 9.58
  let d2 := 6.58
  let d3 := 5.42
  let d4 := 8.42
  have h : (d1 + d2 + d3 + d4) / 4 = 7.5
  { sorry }
  exact h

end average_distance_to_sides_l1734_173412


namespace speed_with_stream_l1734_173498

variable (V_as V_m V_ws : ℝ)

theorem speed_with_stream (h1 : V_as = 6) (h2 : V_m = 2) : V_ws = V_m + (V_as - V_m) :=
by
  sorry

end speed_with_stream_l1734_173498


namespace a_beats_b_by_10_seconds_l1734_173419

theorem a_beats_b_by_10_seconds :
  ∀ (T_A T_B D_A D_B : ℕ),
    T_A = 615 →
    D_A = 1000 →
    D_A - D_B = 16 →
    T_B = (D_A * T_A) / D_B →
    T_B - T_A = 10 :=
by
  -- Placeholder to ensure the theorem compiles
  intros T_A T_B D_A D_B h1 h2 h3 h4
  sorry

end a_beats_b_by_10_seconds_l1734_173419


namespace solve_for_m_l1734_173431

theorem solve_for_m (m : ℝ) (h : (4 * m + 6) * (2 * m - 5) = 159) : m = 5.3925 :=
sorry

end solve_for_m_l1734_173431


namespace tan_difference_identity_l1734_173455

theorem tan_difference_identity (a b : ℝ) (h1 : Real.tan a = 2) (h2 : Real.tan b = 3 / 4) :
  Real.tan (a - b) = 1 / 2 :=
sorry

end tan_difference_identity_l1734_173455


namespace price_of_turbans_l1734_173485

theorem price_of_turbans : 
  ∀ (salary_A salary_B salary_C : ℝ) (months_A months_B months_C : ℕ) (payment_A payment_B payment_C : ℝ)
    (prorated_salary_A prorated_salary_B prorated_salary_C : ℝ),
  salary_A = 120 → 
  salary_B = 150 → 
  salary_C = 180 → 
  months_A = 8 → 
  months_B = 7 → 
  months_C = 10 → 
  payment_A = 80 → 
  payment_B = 87.50 → 
  payment_C = 150 → 
  prorated_salary_A = (salary_A * (months_A / 12 : ℝ)) → 
  prorated_salary_B = (salary_B * (months_B / 12 : ℝ)) → 
  prorated_salary_C = (salary_C * (months_C / 12 : ℝ)) → 
  ∃ (price_A price_B price_C : ℝ),
  price_A = payment_A - prorated_salary_A ∧ 
  price_B = payment_B - prorated_salary_B ∧ 
  price_C = payment_C - prorated_salary_C ∧ 
  price_A = 0 ∧ price_B = 0 ∧ price_C = 0 := 
by
  sorry

end price_of_turbans_l1734_173485


namespace two_pow_ge_two_mul_l1734_173468

theorem two_pow_ge_two_mul (n : ℕ) : 2^n ≥ 2 * n :=
sorry

end two_pow_ge_two_mul_l1734_173468


namespace find_S_9_l1734_173430

variable (a : ℕ → ℝ)

def arithmetic_sum_9 (S_9 : ℝ) : Prop :=
  (a 1 + a 3 + a 5 = 39) ∧ (a 5 + a 7 + a 9 = 27) ∧ (S_9 = (9 * (a 3 + a 7)) / 2)

theorem find_S_9 
  (h1 : a 1 + a 3 + a 5 = 39)
  (h2 : a 5 + a 7 + a 9 = 27) :
  ∃ S_9, arithmetic_sum_9 a S_9 ∧ S_9 = 99 := 
by
  sorry

end find_S_9_l1734_173430


namespace total_original_grain_l1734_173477

-- Define initial conditions
variables (initial_warehouse1 : ℕ) (initial_warehouse2 : ℕ)
-- Define the amount of grain transported away from the first warehouse
def transported_away := 2500
-- Define the amount of grain in the second warehouse
def warehouse2_initial := 50200

-- Prove the total original amount of grain in the two warehouses
theorem total_original_grain 
  (h1 : transported_away = 2500)
  (h2 : warehouse2_initial = 50200)
  (h3 : initial_warehouse1 - transported_away = warehouse2_initial) : 
  initial_warehouse1 + warehouse2_initial = 102900 :=
sorry

end total_original_grain_l1734_173477


namespace sisterPassesMeInOppositeDirection_l1734_173416

noncomputable def numberOfPasses (laps_sister : ℕ) : ℕ :=
if laps_sister > 1 then 2 * laps_sister else 0

theorem sisterPassesMeInOppositeDirection
  (my_laps : ℕ) (laps_sister : ℕ) (passes_in_same_direction : ℕ) :
  my_laps = 1 ∧ passes_in_same_direction = 2 ∧ laps_sister > 1 →
  passes_in_same_direction * 2 = 4 :=
by intros; sorry

end sisterPassesMeInOppositeDirection_l1734_173416


namespace legally_drive_after_hours_l1734_173453

theorem legally_drive_after_hours (n : ℕ) :
  (∀ t ≥ n, 0.8 * (0.5 : ℝ) ^ t ≤ 0.2) ↔ n = 2 :=
by
  sorry

end legally_drive_after_hours_l1734_173453
