import Mathlib

namespace factor_expression_l1593_159337

theorem factor_expression (y : ℝ) : 5 * y * (y + 2) + 9 * (y + 2) + 2 * (y + 2) = (y + 2) * (5 * y + 11) := by
  sorry

end factor_expression_l1593_159337


namespace library_books_l1593_159348

theorem library_books (A : Prop) (B : Prop) (C : Prop) (D : Prop) :
  (¬A) → (B ∧ D) :=
by
  -- Assume the statement "All books in this library are available for lending." is represented by A.
  -- A is false.
  intro h_notA
  -- Show that statement II ("There is some book in this library not available for lending.")
  -- and statement IV ("Not all books in this library are available for lending.") are both true.
  -- These are represented as B and D, respectively.
  sorry

end library_books_l1593_159348


namespace ellipse_properties_l1593_159391

theorem ellipse_properties :
  ∀ {x y : ℝ}, 4 * x^2 + 2 * y^2 = 16 →
    (∃ a b e c, a = 2 * Real.sqrt 2 ∧ b = 2 ∧ e = Real.sqrt 2 / 2 ∧ c = 2) ∧
    (∃ f1 f2, f1 = (0, 2) ∧ f2 = (0, -2)) ∧
    (∃ v1 v2 v3 v4, v1 = (0, 2 * Real.sqrt 2) ∧ v2 = (0, -2 * Real.sqrt 2) ∧ v3 = (2, 0) ∧ v4 = (-2, 0)) :=
by
  sorry

end ellipse_properties_l1593_159391


namespace initial_matchsticks_l1593_159306

-- Define the problem conditions
def matchsticks_elvis := 4
def squares_elvis := 5
def matchsticks_ralph := 8
def squares_ralph := 3
def matchsticks_left := 6

-- Calculate the total matchsticks used by Elvis and Ralph
def total_used_elvis := matchsticks_elvis * squares_elvis
def total_used_ralph := matchsticks_ralph * squares_ralph
def total_used := total_used_elvis + total_used_ralph

-- The proof statement
theorem initial_matchsticks (matchsticks_elvis squares_elvis matchsticks_ralph squares_ralph matchsticks_left : ℕ) : total_used + matchsticks_left = 50 := 
by
  sorry

end initial_matchsticks_l1593_159306


namespace students_without_glasses_l1593_159349

theorem students_without_glasses (total_students : ℕ) (perc_glasses : ℕ) (students_with_glasses students_without_glasses : ℕ) 
  (h1 : total_students = 325) (h2 : perc_glasses = 40) (h3 : students_with_glasses = perc_glasses * total_students / 100)
  (h4 : students_without_glasses = total_students - students_with_glasses) : students_without_glasses = 195 := 
by
  sorry

end students_without_glasses_l1593_159349


namespace same_last_k_digits_pow_l1593_159300

theorem same_last_k_digits_pow (A B : ℤ) (k n : ℕ) 
  (h : A % 10^k = B % 10^k) : 
  (A^n % 10^k = B^n % 10^k) := 
by
  sorry

end same_last_k_digits_pow_l1593_159300


namespace find_f_of_three_l1593_159330

variable {f : ℝ → ℝ}

theorem find_f_of_three (h : ∀ x : ℝ, f (1 - 2 * x) = x^2 + x) : f 3 = 0 :=
by
  sorry

end find_f_of_three_l1593_159330


namespace quadratic_eq_a_val_l1593_159347

theorem quadratic_eq_a_val (a : ℝ) (h : a - 6 = 0) :
  a = 6 :=
by
  sorry

end quadratic_eq_a_val_l1593_159347


namespace inequality_solution_set_l1593_159314

theorem inequality_solution_set (a : ℝ) : 
    (a = 0 → (∃ x : ℝ, x > 1 ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a < 0 → (∃ x : ℝ, (x < 2/a ∨ x > 1) ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (0 < a ∧ a < 2 → (∃ x : ℝ, (1 < x ∧ x < 2/a) ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a = 2 → ¬(∃ x : ℝ, ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a > 2 → (∃ x : ℝ, (2/a < x ∧ x < 1) ∧ ax^2 - (a + 2) * x + 2 < 0)) :=
by sorry

end inequality_solution_set_l1593_159314


namespace find_daily_rate_of_first_company_l1593_159312

-- Define the daily rate of the first car rental company
def daily_rate_first_company (x : ℝ) : ℝ :=
  x + 0.18 * 48.0

-- Define the total cost for City Rentals
def total_cost_city_rentals : ℝ :=
  18.95 + 0.16 * 48.0

-- Prove the daily rate of the first car rental company
theorem find_daily_rate_of_first_company (x : ℝ) (h : daily_rate_first_company x = total_cost_city_rentals) : 
  x = 17.99 := 
by
  sorry

end find_daily_rate_of_first_company_l1593_159312


namespace triangle_inequality_product_l1593_159320

theorem triangle_inequality_product (x y z : ℝ) (h1 : x + y > z) (h2 : x + z > y) (h3 : y + z > x) :
  (x + y + z) * (x + y - z) * (x + z - y) * (z + y - x) ≤ 4 * x^2 * y^2 := 
by
  sorry

end triangle_inequality_product_l1593_159320


namespace odd_positive_int_divisible_by_24_l1593_159311

theorem odd_positive_int_divisible_by_24 (n : ℕ) (hn : n % 2 = 1 ∧ n > 0) : 24 ∣ (n ^ n - n) :=
sorry

end odd_positive_int_divisible_by_24_l1593_159311


namespace meet_time_correct_l1593_159352

variable (circumference : ℕ) (speed_yeonjeong speed_donghun : ℕ)

def meet_time (circumference speed_yeonjeong speed_donghun : ℕ) : ℕ :=
  circumference / (speed_yeonjeong + speed_donghun)

theorem meet_time_correct
  (h_circumference : circumference = 3000)
  (h_speed_yeonjeong : speed_yeonjeong = 100)
  (h_speed_donghun : speed_donghun = 150) :
  meet_time circumference speed_yeonjeong speed_donghun = 12 :=
by
  rw [h_circumference, h_speed_yeonjeong, h_speed_donghun]
  norm_num
  sorry

end meet_time_correct_l1593_159352


namespace trash_picked_outside_l1593_159322

theorem trash_picked_outside (T_tot : ℕ) (C1 C2 C3 C4 C5 C6 C7 C8 : ℕ)
  (hT_tot : T_tot = 1576)
  (hC1 : C1 = 124) (hC2 : C2 = 98) (hC3 : C3 = 176) (hC4 : C4 = 212)
  (hC5 : C5 = 89) (hC6 : C6 = 241) (hC7 : C7 = 121) (hC8 : C8 = 102) :
  T_tot - (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8) = 413 :=
by sorry

end trash_picked_outside_l1593_159322


namespace carnival_days_l1593_159395

-- Define the given conditions
def total_money := 3168
def daily_income := 144

-- Define the main theorem statement
theorem carnival_days : (total_money / daily_income) = 22 := by
  sorry

end carnival_days_l1593_159395


namespace jimin_class_students_l1593_159384

theorem jimin_class_students 
    (total_distance : ℝ)
    (interval_distance : ℝ)
    (h1 : total_distance = 242)
    (h2 : interval_distance = 5.5) :
    (total_distance / interval_distance) + 1 = 45 :=
by sorry

end jimin_class_students_l1593_159384


namespace annie_total_distance_traveled_l1593_159329

-- Definitions of conditions
def walk_distance : ℕ := 5
def bus_distance : ℕ := 7
def total_distance_one_way : ℕ := walk_distance + bus_distance
def total_distance_round_trip : ℕ := total_distance_one_way * 2

-- Theorem statement to prove the total number of blocks traveled
theorem annie_total_distance_traveled : total_distance_round_trip = 24 :=
by
  sorry

end annie_total_distance_traveled_l1593_159329


namespace product_terms_l1593_159387

variable (a_n : ℕ → ℝ)
variable (r : ℝ)

-- a1 = 1 and a10 = 3
axiom geom_seq  (h : ∀ n, a_n (n + 1) = r * a_n n) : a_n 1 = 1 → a_n 10 = 3

theorem product_terms :
  (∀ n, a_n (n + 1) = r * a_n n) → a_n 1 = 1 → a_n 10 = 3 → 
  a_n 2 * a_n 3 * a_n 4 * a_n 5 * a_n 6 * a_n 7 * a_n 8 * a_n 9 = 81 :=
by
  intros h1 h2 h3
  sorry

end product_terms_l1593_159387


namespace fraction_identity_l1593_159372

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := 2 * x - 3

theorem fraction_identity : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by
  sorry

end fraction_identity_l1593_159372


namespace minimum_spend_on_boxes_l1593_159364

def box_dimensions : ℕ × ℕ × ℕ := (20, 20, 12)
def cost_per_box : ℝ := 0.40
def total_volume : ℕ := 2400000
def volume_of_box (l w h : ℕ) : ℕ := l * w * h
def number_of_boxes (total_vol vol_per_box : ℕ) : ℕ := total_vol / vol_per_box
def total_cost (num_boxes : ℕ) (cost_box : ℝ) : ℝ := num_boxes * cost_box

theorem minimum_spend_on_boxes : total_cost (number_of_boxes total_volume (volume_of_box 20 20 12)) cost_per_box = 200 := by
  sorry

end minimum_spend_on_boxes_l1593_159364


namespace units_digit_1_to_99_is_5_l1593_159327

noncomputable def units_digit_of_product_of_odds : ℕ :=
  let seq := List.range' 1 99;
  (seq.filter (λ n => n % 2 = 1)).prod % 10

theorem units_digit_1_to_99_is_5 : units_digit_of_product_of_odds = 5 :=
by sorry

end units_digit_1_to_99_is_5_l1593_159327


namespace indeterminate_C_l1593_159342

variable (m n C : ℝ)

theorem indeterminate_C (h1 : m = 8 * n + C)
                      (h2 : m + 2 = 8 * (n + 0.25) + C) : 
                      False :=
by
  sorry

end indeterminate_C_l1593_159342


namespace missing_number_l1593_159339

theorem missing_number (n : ℝ) (h : (0.0088 * 4.5) / (0.05 * n * 0.008) = 990) : n = 0.1 :=
sorry

end missing_number_l1593_159339


namespace chef_earns_less_than_manager_l1593_159308

theorem chef_earns_less_than_manager :
  let manager_wage := 7.50
  let dishwasher_wage := manager_wage / 2
  let chef_wage := dishwasher_wage * 1.20
  (manager_wage - chef_wage) = 3.00 := by
    sorry

end chef_earns_less_than_manager_l1593_159308


namespace sum_of_ages_in_5_years_l1593_159304

noncomputable def age_will_three_years_ago := 4
noncomputable def years_elapsed := 3
noncomputable def age_will_now := age_will_three_years_ago + years_elapsed
noncomputable def age_diane_now := 2 * age_will_now
noncomputable def years_into_future := 5
noncomputable def age_will_in_future := age_will_now + years_into_future
noncomputable def age_diane_in_future := age_diane_now + years_into_future

theorem sum_of_ages_in_5_years :
  age_will_in_future + age_diane_in_future = 31 := by
  sorry

end sum_of_ages_in_5_years_l1593_159304


namespace no_integer_solution_for_system_l1593_159386

theorem no_integer_solution_for_system :
  ¬ ∃ (a b c d : ℤ), 
    (a * b * c * d - a = 1961) ∧ 
    (a * b * c * d - b = 961) ∧ 
    (a * b * c * d - c = 61) ∧ 
    (a * b * c * d - d = 1) :=
by {
  sorry
}

end no_integer_solution_for_system_l1593_159386


namespace rod_division_segments_l1593_159344

theorem rod_division_segments (L : ℕ) (K : ℕ) (hL : L = 72 * K) :
  let red_divisions := 7
  let blue_divisions := 11
  let black_divisions := 17
  let overlap_9_6 := 4
  let overlap_6_4 := 6
  let overlap_9_4 := 2
  let overlap_all := 2
  let total_segments := red_divisions + blue_divisions + black_divisions - overlap_9_6 - overlap_6_4 - overlap_9_4 + overlap_all
  (total_segments = 28) ∧ ((L / 72) = K)
:=
by
  sorry

end rod_division_segments_l1593_159344


namespace mean_value_of_quadrilateral_angles_l1593_159394

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l1593_159394


namespace quadratic_two_distinct_real_roots_l1593_159309

theorem quadratic_two_distinct_real_roots
  (a1 a2 a3 a4 : ℝ)
  (h : a1 > a2 ∧ a2 > a3 ∧ a3 > a4) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - (a1 + a2 + a3 + a4) * x1 + (a1 * a3 + a2 * a4) = 0)
  ∧ (x2^2 - (a1 + a2 + a3 + a4) * x2 + (a1 * a3 + a2 * a4) = 0) :=
by 
  sorry

end quadratic_two_distinct_real_roots_l1593_159309


namespace joan_seashells_initially_l1593_159318

variable (mikeGave joanTotal : ℕ)

theorem joan_seashells_initially (h : mikeGave = 63) (t : joanTotal = 142) : joanTotal - mikeGave = 79 := 
by
  sorry

end joan_seashells_initially_l1593_159318


namespace pool_filling_time_l1593_159378

noncomputable def fill_pool_time (hose_rate : ℕ) (cost_per_10_gallons : ℚ) (total_cost : ℚ) : ℚ :=
  let cost_per_gallon := cost_per_10_gallons / 10
  let total_gallons := total_cost / cost_per_gallon
  total_gallons / hose_rate

theorem pool_filling_time :
  fill_pool_time 100 (1 / 100) 5 = 50 := 
by
  sorry

end pool_filling_time_l1593_159378


namespace midpoint_in_polar_coordinates_l1593_159382

-- Define the problem as a theorem in Lean 4
theorem midpoint_in_polar_coordinates :
  let A := (10, Real.pi / 4)
  let B := (10, 3 * Real.pi / 4)
  ∃ r θ, (r = 5 * Real.sqrt 2) ∧ (θ = Real.pi / 2) ∧
         0 ≤ θ ∧ θ < 2 * Real.pi :=
by
  sorry

end midpoint_in_polar_coordinates_l1593_159382


namespace find_D_l1593_159358

noncomputable def point := (ℝ × ℝ)

def vector_add (u v : point) : point := (u.1 + v.1, u.2 + v.2)
def vector_sub (u v : point) : point := (u.1 - v.1, u.2 - v.2)
def scalar_multiplication (k : ℝ) (u : point) : point := (k * u.1, k * u.2)

namespace GeometryProblem

def A : point := (2, 3)
def B : point := (-1, 5)

def D : point := 
  let AB := vector_sub B A
  vector_add A (scalar_multiplication 3 AB)

theorem find_D : D = (-7, 9) := by
  sorry

end GeometryProblem

end find_D_l1593_159358


namespace find_f_neg_a_l1593_159397

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.tan x + 1

theorem find_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end find_f_neg_a_l1593_159397


namespace part_a_part_b_l1593_159335

noncomputable def same_start_digit (n x : ℕ) : Prop :=
  ∃ d : ℕ, ∀ k : ℕ, (k ≤ n) → (x * 10^(k-1) ≤ d * 10^(k-1) + 10^(k-1) - 1) ∧ ((d * 10^(k-1)) < x * 10^(k-1))

theorem part_a (x : ℕ) : 
  (same_start_digit 3 x) → ¬(∃ d : ℕ, d = 1) → false :=
  sorry

theorem part_b (x : ℕ) : 
  (same_start_digit 2015 x) → ¬(∃ d : ℕ, d = 1) → false :=
  sorry

end part_a_part_b_l1593_159335


namespace number_of_packs_of_cake_l1593_159333

-- Define the total number of packs of groceries
def total_packs : ℕ := 14

-- Define the number of packs of cookies
def packs_of_cookies : ℕ := 2

-- Define the number of packs of cake as total packs minus packs of cookies
def packs_of_cake : ℕ := total_packs - packs_of_cookies

theorem number_of_packs_of_cake :
  packs_of_cake = 12 := by
  -- Placeholder for the proof
  sorry

end number_of_packs_of_cake_l1593_159333


namespace order_a_c_b_l1593_159357

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 4 / Real.log 3
noncomputable def c : ℝ := Real.log 8 / Real.log 5

theorem order_a_c_b : a > c ∧ c > b := 
by {
  sorry
}

end order_a_c_b_l1593_159357


namespace gcd_8_10_l1593_159328

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_8_10 : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := 
by
  sorry

end gcd_8_10_l1593_159328


namespace rectangle_pentagon_ratio_l1593_159398

theorem rectangle_pentagon_ratio
  (l w p : ℝ)
  (h1 : l = 2 * w)
  (h2 : 2 * (l + w) = 30)
  (h3 : 5 * p = 30) :
  l / p = 5 / 3 :=
by
  sorry

end rectangle_pentagon_ratio_l1593_159398


namespace mod_neg_result_l1593_159323

-- Define the hypothesis as the residue equivalence and positive range constraint.
theorem mod_neg_result : 
  ∀ (a b : ℤ), (-1277 : ℤ) % 32 = 3 := by
  sorry

end mod_neg_result_l1593_159323


namespace tileable_if_and_only_if_l1593_159326

def is_tileable (n : ℕ) : Prop :=
  ∃ k : ℕ, 15 * n = 4 * k

theorem tileable_if_and_only_if (n : ℕ) :
  (n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 4 ∧ n ≠ 7) ↔ is_tileable n :=
sorry

end tileable_if_and_only_if_l1593_159326


namespace length_of_AD_l1593_159376

theorem length_of_AD (AB BC AC AD DC : ℝ)
    (h1 : AB = BC)
    (h2 : AD = 2 * DC)
    (h3 : AC = AD + DC)
    (h4 : AC = 27) : AD = 18 := 
by
  sorry

end length_of_AD_l1593_159376


namespace find_vector_c_l1593_159331

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (1, 2)
def c : ℝ × ℝ := (2, 1)

def perp (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, w = (k * v.1, k * v.2)

theorem find_vector_c : 
  perp (c.1 + b.1, c.2 + b.2) a ∧ parallel (c.1 - a.1, c.2 + a.2) b :=
by 
  sorry

end find_vector_c_l1593_159331


namespace inequality_not_always_hold_l1593_159355

theorem inequality_not_always_hold (a b : ℝ) (h : a > -b) : ¬ (∀ a b : ℝ, a > -b → (1 / a + 1 / b > 0)) :=
by
  intro h2
  have h3 := h2 a b h
  sorry

end inequality_not_always_hold_l1593_159355


namespace number_of_red_balls_l1593_159389

theorem number_of_red_balls (W R T : ℕ) (hW : W = 12) (h_freq : (R : ℝ) / (T : ℝ) = 0.25) (hT : T = W + R) : R = 4 :=
by
  sorry

end number_of_red_balls_l1593_159389


namespace find_m_l1593_159321

variable (m : ℝ)
def vector_a : ℝ × ℝ := (1, 3)
def vector_b : ℝ × ℝ := (m, -2)

theorem find_m (h : (1 + m) + 3 = 0) : m = -4 := by
  sorry

end find_m_l1593_159321


namespace sufficient_but_not_necessary_l1593_159340

theorem sufficient_but_not_necessary (a : ℝ) : ((a = 2) → ((a - 1) * (a - 2) = 0)) ∧ (¬(((a - 1) * (a - 2) = 0) → (a = 2))) := 
by 
sorry

end sufficient_but_not_necessary_l1593_159340


namespace smallest_integer_solution_l1593_159316

theorem smallest_integer_solution : ∃ x : ℤ, (x^2 = 3 * x + 78) ∧ x = -6 :=
by {
  sorry
}

end smallest_integer_solution_l1593_159316


namespace find_a_from_function_l1593_159341

theorem find_a_from_function (f : ℝ → ℝ) (h_f : ∀ x, f x = Real.sqrt (2 * x + 1)) (a : ℝ) (h_a : f a = 5) : a = 12 :=
by
  sorry

end find_a_from_function_l1593_159341


namespace octahedron_cut_area_l1593_159345

theorem octahedron_cut_area:
  let a := 9
  let b := 3
  let c := 8
  a + b + c = 20 :=
by
  sorry

end octahedron_cut_area_l1593_159345


namespace kenneth_fabric_amount_l1593_159396

theorem kenneth_fabric_amount :
  ∃ K : ℤ, (∃ N : ℤ, N = 6 * K ∧ (K * 40 + 140000 = N * 40) ∧ K > 0) ∧ K = 700 :=
by
  sorry

end kenneth_fabric_amount_l1593_159396


namespace johns_salary_percentage_increase_l1593_159369

theorem johns_salary_percentage_increase (initial_salary final_salary : ℕ) (h1 : initial_salary = 50) (h2 : final_salary = 90) :
  ((final_salary - initial_salary : ℕ) / initial_salary : ℚ) * 100 = 80 := by
  sorry

end johns_salary_percentage_increase_l1593_159369


namespace initial_value_exists_l1593_159315

theorem initial_value_exists (x : ℕ) (h : ∃ k : ℕ, x + 7 = k * 456) : x = 449 :=
sorry

end initial_value_exists_l1593_159315


namespace intersection_P_Q_l1593_159303

-- Define set P
def P : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

-- Define set Q (using real numbers, but we will be interested in natural number intersections)
def Q : Set ℝ := {x | x^2 + x - 6 ≤ 0}

-- The intersection of P with Q in the natural numbers should be {1, 2}
theorem intersection_P_Q :
  {x : ℕ | x ∈ P ∧ (x : ℝ) ∈ Q} = {1, 2} :=
by
  sorry

end intersection_P_Q_l1593_159303


namespace value_of_expression_l1593_159370

theorem value_of_expression (x y : ℝ) (h1 : 4 * x + y = 20) (h2 : x + 4 * y = 16) : 
  17 * x ^ 2 + 20 * x * y + 17 * y ^ 2 = 656 :=
sorry

end value_of_expression_l1593_159370


namespace expressions_divisible_by_17_l1593_159390

theorem expressions_divisible_by_17 (a b : ℤ) : 
  let x := 3 * b - 5 * a
  let y := 9 * a - 2 * b
  (∃ k : ℤ, (2 * x + 3 * y) = 17 * k) ∧ (∃ k : ℤ, (9 * x + 5 * y) = 17 * k) :=
by
  exact ⟨⟨a, by sorry⟩, ⟨b, by sorry⟩⟩

end expressions_divisible_by_17_l1593_159390


namespace max_at_zero_l1593_159365

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem max_at_zero : ∀ x : ℝ, f x ≤ f 0 :=
by
  sorry

end max_at_zero_l1593_159365


namespace remainder_when_divided_by_15_l1593_159388

theorem remainder_when_divided_by_15 (N : ℤ) (k : ℤ) 
  (h : N = 45 * k + 31) : (N % 15) = 1 := by
  sorry

end remainder_when_divided_by_15_l1593_159388


namespace FO_greater_DI_l1593_159334

-- The quadrilateral FIDO is assumed to be convex with specified properties
variables {F I D O E : Type*}

variables (length_FI length_DI length_DO length_FO : ℝ)
variables (angle_FIO angle_DIO : ℝ)
variables (E : I)

-- Given conditions
variables (convex_FIDO : Prop) -- FIDO is convex
variables (h1 : length_FI = length_DO)
variables (h2 : length_FI > length_DI)
variables (h3 : angle_FIO = angle_DIO)

-- Use given identity IE = ID
variables (length_IE : ℝ) (h4 : length_IE = length_DI)

theorem FO_greater_DI 
    (length_FI length_DI length_DO length_FO : ℝ)
    (angle_FIO angle_DIO : ℝ)
    (convex_FIDO : Prop)
    (h1 : length_FI = length_DO)
    (h2 : length_FI > length_DI)
    (h3 : angle_FIO = angle_DIO)
    (length_IE : ℝ)
    (h4 : length_IE = length_DI) : 
    length_FO > length_DI :=
sorry

end FO_greater_DI_l1593_159334


namespace abs_diff_simplification_l1593_159360

theorem abs_diff_simplification (a b : ℝ) (h1 : a < 0) (h2 : a * b < 0) : |b - a + 1| - |a - b - 5| = -4 :=
  sorry

end abs_diff_simplification_l1593_159360


namespace original_planned_length_l1593_159361

theorem original_planned_length (x : ℝ) (h1 : x > 0) (total_length : ℝ := 3600) (efficiency_ratio : ℝ := 1.8) (time_saved : ℝ := 20) 
  (h2 : total_length / x - total_length / (efficiency_ratio * x) = time_saved) :
  x = 80 :=
sorry

end original_planned_length_l1593_159361


namespace range_of_m_for_distance_l1593_159353

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (|x1 - x2|) + 2 * (|y1 - y2|)

theorem range_of_m_for_distance (m : ℝ) : 
  distance 2 1 (-1) m ≤ 5 ↔ 0 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_for_distance_l1593_159353


namespace domain_of_f_l1593_159356

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (Real.log x / Real.log 2 - 1))

theorem domain_of_f :
  {x : ℝ | x > 2} = {x : ℝ | x > 0 ∧ Real.log x / Real.log 2 - 1 > 0} := 
by
  sorry

end domain_of_f_l1593_159356


namespace lily_pad_half_lake_l1593_159375

theorem lily_pad_half_lake
  (P : ℕ → ℝ) -- Define a function P(n) which represents the size of the patch on day n.
  (h1 : ∀ n, P n = P (n - 1) * 2) -- Every day, the patch doubles in size.
  (h2 : P 58 = 1) -- It takes 58 days for the patch to cover the entire lake (normalized to 1).
  : P 57 = 1 / 2 :=
by
  sorry

end lily_pad_half_lake_l1593_159375


namespace total_rainfall_in_2011_l1593_159362

-- Define the given conditions
def avg_monthly_rainfall_2010 : ℝ := 36.8
def increase_2011 : ℝ := 3.5

-- Define the resulting average monthly rainfall in 2011
def avg_monthly_rainfall_2011 : ℝ := avg_monthly_rainfall_2010 + increase_2011

-- Calculate the total annual rainfall
def total_rainfall_2011 : ℝ := avg_monthly_rainfall_2011 * 12

-- State the proof problem
theorem total_rainfall_in_2011 :
  total_rainfall_2011 = 483.6 := by
  sorry

end total_rainfall_in_2011_l1593_159362


namespace total_training_hours_l1593_159332

-- Define Thomas's training conditions
def hours_per_day := 5
def days_initial := 30
def days_additional := 12
def total_days := days_initial + days_additional

-- State the theorem to be proved
theorem total_training_hours : total_days * hours_per_day = 210 :=
by
  sorry

end total_training_hours_l1593_159332


namespace group_weight_problem_l1593_159371

theorem group_weight_problem (n : ℕ) (avg_weight_increase : ℕ) (weight_diff : ℕ) (total_weight_increase : ℕ) 
  (h1 : avg_weight_increase = 3) (h2 : weight_diff = 75 - 45) (h3 : total_weight_increase = avg_weight_increase * n)
  (h4 : total_weight_increase = weight_diff) : n = 10 := by
  sorry

end group_weight_problem_l1593_159371


namespace trig_identity_proof_l1593_159399

noncomputable def check_trig_identities (α β : ℝ) : Prop :=
  3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2

theorem trig_identity_proof (α β : ℝ) (h : check_trig_identities α β) :
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 := by
  sorry

end trig_identity_proof_l1593_159399


namespace circle_equation_passing_through_points_symmetric_circle_equation_midpoint_trajectory_equation_l1593_159393

-- Prove the equation of the circle passing through points A and B with center on a specified line
theorem circle_equation_passing_through_points
  (A B : ℝ × ℝ) (line : ℝ → ℝ → Prop)
  (N : ℝ → ℝ → Prop) :
  A = (3, 1) →
  B = (-1, 3) →
  (∀ x y, line x y ↔ 3 * x - y - 2 = 0) →
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  sorry :=
sorry

-- Prove the symmetric circle equation regarding a specified line
theorem symmetric_circle_equation
  (N N' : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) :
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  (∀ x y, N' x y ↔ (x - 1)^2 + (y - 5)^2 = 10) →
  (∀ x y, line x y ↔ x - y + 3 = 0) →
  sorry :=
sorry

-- Prove the trajectory equation of the midpoint
theorem midpoint_trajectory_equation
  (C : ℝ × ℝ) (N : ℝ → ℝ → Prop) (M_trajectory : ℝ → ℝ → Prop) :
  C = (3, 0) →
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  (∀ x y, M_trajectory x y ↔ (x - 5 / 2)^2 + (y - 2)^2 = 5 / 2) →
  sorry :=
sorry

end circle_equation_passing_through_points_symmetric_circle_equation_midpoint_trajectory_equation_l1593_159393


namespace jogger_distance_ahead_l1593_159354

noncomputable def jogger_speed_kmph : ℤ := 9
noncomputable def train_speed_kmph : ℤ := 45
noncomputable def train_length_m : ℤ := 120
noncomputable def time_to_pass_seconds : ℤ := 38

theorem jogger_distance_ahead
  (jogger_speed_kmph : ℤ)
  (train_speed_kmph : ℤ)
  (train_length_m : ℤ)
  (time_to_pass_seconds : ℤ) :
  jogger_speed_kmph = 9 →
  train_speed_kmph = 45 →
  train_length_m = 120 →
  time_to_pass_seconds = 38 →
  ∃ distance_ahead : ℤ, distance_ahead = 260 :=
by 
  -- the proof would go here
  sorry  

end jogger_distance_ahead_l1593_159354


namespace implicit_derivative_l1593_159359

noncomputable section

open Real

section ImplicitDifferentiation

variable {x : ℝ} {y : ℝ → ℝ}

def f (x y : ℝ) : ℝ := y^2 + x^2 - 1

theorem implicit_derivative (h : f x (y x) = 0) :
  deriv y x = -x / y x :=
  sorry

end ImplicitDifferentiation

end implicit_derivative_l1593_159359


namespace find_mass_of_water_vapor_l1593_159363

noncomputable def heat_balance_problem : Prop :=
  ∃ (m_s : ℝ), m_s * 536 + m_s * 80 = 
  (50 * 80 + 50 * 20 + 300 * 20 + 100 * 0.5 * 20)
  ∧ m_s = 19.48

theorem find_mass_of_water_vapor : heat_balance_problem := by
  sorry

end find_mass_of_water_vapor_l1593_159363


namespace log_comparison_l1593_159383

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4
noncomputable def log6 (x : ℝ) : ℝ := Real.log x / Real.log 6

theorem log_comparison :
  let a := log2 6
  let b := log4 12
  let c := log6 18
  a > b ∧ b > c :=
by 
  sorry

end log_comparison_l1593_159383


namespace odd_lattice_points_on_BC_l1593_159380

theorem odd_lattice_points_on_BC
  (A B C : ℤ × ℤ)
  (odd_lattice_points_AB : Odd ((B.1 - A.1) * (B.2 - A.2)))
  (odd_lattice_points_AC : Odd ((C.1 - A.1) * (C.2 - A.2))) :
  Odd ((C.1 - B.1) * (C.2 - B.2)) :=
sorry

end odd_lattice_points_on_BC_l1593_159380


namespace probability_of_selecting_cooking_l1593_159302

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l1593_159302


namespace sachin_is_younger_than_rahul_by_18_years_l1593_159346

-- Definitions based on conditions
def sachin_age : ℕ := 63
def ratio_of_ages : ℚ := 7 / 9

-- Assertion that based on the given conditions, Sachin is 18 years younger than Rahul
theorem sachin_is_younger_than_rahul_by_18_years (R : ℕ) (h1 : (sachin_age : ℚ) / R = ratio_of_ages) : R - sachin_age = 18 :=
by
  sorry

end sachin_is_younger_than_rahul_by_18_years_l1593_159346


namespace power_mod_l1593_159338

theorem power_mod (h1: 5^2 % 17 = 8) (h2: 5^4 % 17 = 13) (h3: 5^8 % 17 = 16) (h4: 5^16 % 17 = 1):
  5^2024 % 17 = 16 :=
by
  sorry

end power_mod_l1593_159338


namespace symmetric_coordinates_l1593_159343

-- Define the point A as a tuple of its coordinates
def A : Prod ℤ ℤ := (-1, 2)

-- Define what it means for point A' to be symmetric to the origin
def symmetric_to_origin (p : Prod ℤ ℤ) : Prod ℤ ℤ :=
  (-p.1, -p.2)

-- The theorem we need to prove
theorem symmetric_coordinates :
  symmetric_to_origin A = (1, -2) :=
by
  sorry

end symmetric_coordinates_l1593_159343


namespace xyz_neg_l1593_159381

theorem xyz_neg {a b c x y z : ℝ} 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) 
  (h : |x - a| + |y - b| + |z - c| = 0) : 
  x * y * z < 0 :=
by 
  -- to be proven
  sorry

end xyz_neg_l1593_159381


namespace tangent_line_equation_l1593_159313

noncomputable def circle_eq1 (x y : ℝ) := x^2 + (y - 2)^2 - 4
noncomputable def circle_eq2 (x y : ℝ) := (x - 3)^2 + (y + 2)^2 - 21
noncomputable def line_eq (x y : ℝ) := 3*x - 4*y - 4

theorem tangent_line_equation :
  ∀ (x y : ℝ), (circle_eq1 x y = 0 ∧ circle_eq2 x y = 0) ↔ line_eq x y = 0 :=
sorry

end tangent_line_equation_l1593_159313


namespace trivia_team_missing_members_l1593_159301

theorem trivia_team_missing_members 
  (total_members : ℕ)
  (points_per_member : ℕ)
  (total_points : ℕ)
  (showed_up_members : ℕ)
  (missing_members : ℕ) 
  (h1 : total_members = 15) 
  (h2 : points_per_member = 3) 
  (h3 : total_points = 27) 
  (h4 : showed_up_members = total_points / points_per_member) 
  (h5 : missing_members = total_members - showed_up_members) : 
  missing_members = 6 :=
by
  sorry

end trivia_team_missing_members_l1593_159301


namespace floor_diff_l1593_159319

theorem floor_diff {x : ℝ} (h : x = 12.7) : 
  (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * (⌊x⌋ : ℤ) = 17 :=
by
  have h1 : x = 12.7 := h
  have hx2 : x^2 = 161.29 := by sorry
  have hfloor : ⌊x⌋ = 12 := by sorry
  have hfloor2 : ⌊x^2⌋ = 161 := by sorry
  sorry

end floor_diff_l1593_159319


namespace total_lunch_bill_l1593_159317

theorem total_lunch_bill (hotdog salad : ℝ) (h1 : hotdog = 5.36) (h2 : salad = 5.10) : hotdog + salad = 10.46 := 
by
  rw [h1, h2]
  norm_num
  

end total_lunch_bill_l1593_159317


namespace max_area_rectangle_min_area_rectangle_l1593_159325

theorem max_area_rectangle (n : ℕ) (x y : ℕ → ℕ)
  (S : ℕ → ℕ) (H1 : ∀ k, S k = 2^(2*k)) 
  (H2 : ∀ k, (1 ≤ k ∧ k ≤ n) → x k * y k = S k) 
  : (n - 1 + 2^(2*n)) * (4 * 2^(2*(n-1)) - 1/3) = 1/3 * (4^n - 1) * (4^n + n - 1) := sorry

theorem min_area_rectangle (n : ℕ) (x y : ℕ → ℕ)
  (S : ℕ → ℕ) (H1 : ∀ k, S k = 2^(2*k)) 
  (H2 : ∀ k, (1 ≤ k ∧ k ≤ n) → x k * y k = S k)
  : (2^n - 1)^2 = 4 * (2^n - 1)^2 := sorry

end max_area_rectangle_min_area_rectangle_l1593_159325


namespace oil_in_Tank_C_is_982_l1593_159310

-- Definitions of tank capacities and oil amounts
def capacity_A := 80
def capacity_B := 120
def capacity_C := 160
def capacity_D := 240

def total_oil_bought := 1387

def oil_in_A := 70
def oil_in_B := 95
def oil_in_D := capacity_D  -- Since Tank D is 100% full

-- Statement of the problem
theorem oil_in_Tank_C_is_982 :
  oil_in_A + oil_in_B + oil_in_D + (total_oil_bought - (oil_in_A + oil_in_B + oil_in_D)) = total_oil_bought :=
by
  sorry

end oil_in_Tank_C_is_982_l1593_159310


namespace arithmetic_geometric_seq_l1593_159374

variable {a_n : ℕ → ℝ}
variable {a_1 a_3 a_5 a_6 a_11 : ℝ}

theorem arithmetic_geometric_seq (h₁ : a_1 * a_5 + 2 * a_3 * a_6 + a_1 * a_11 = 16) 
                                  (h₂ : a_1 * a_5 = a_3^2) 
                                  (h₃ : a_1 * a_11 = a_6^2) 
                                  (h₄ : a_3 > 0)
                                  (h₅ : a_6 > 0) : 
    a_3 + a_6 = 4 := 
by {
    sorry
}

end arithmetic_geometric_seq_l1593_159374


namespace positive_rational_representation_l1593_159366

theorem positive_rational_representation (q : ℚ) (h_pos_q : 0 < q) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ q = (a^2021 + b^2023) / (c^2022 + d^2024) :=
by
  sorry

end positive_rational_representation_l1593_159366


namespace player_B_wins_l1593_159392

-- Here we define the scenario and properties from the problem statement.
def initial_pile1 := 100
def initial_pile2 := 252

-- Definition of a turn, conditions and the win condition based on the problem
structure Turn :=
  (pile1 : ℕ)
  (pile2 : ℕ)
  (player_A_turn : Bool)  -- True if it's player A's turn, False if it's player B's turn

-- The game conditions and strategy for determining the winner
def will_player_B_win (initial_pile1 initial_pile2 : ℕ) : Bool :=
  -- assuming the conditions are provided and correctly analyzed, 
  -- we directly state the known result according to the optimal strategies from the solution
  true  -- B wins as per the solution's analysis if both play optimally.

-- The final theorem stating Player B wins given the initial conditions with both playing optimally and A going first.
theorem player_B_wins : will_player_B_win initial_pile1 initial_pile2 = true :=
  sorry  -- Proof omitted.

end player_B_wins_l1593_159392


namespace required_speed_l1593_159368

theorem required_speed
  (D T : ℝ) (h1 : 30 = D / T) 
  (h2 : 2 * D / 3 = 30 * (T / 3)) :
  (D / 3) / (2 * T / 3) = 15 :=
by
  sorry

end required_speed_l1593_159368


namespace total_students_in_lunchroom_l1593_159379

theorem total_students_in_lunchroom (students_per_table : ℕ) (num_tables : ℕ) (total_students : ℕ) :
  students_per_table = 6 → 
  num_tables = 34 → 
  total_students = students_per_table * num_tables → 
  total_students = 204 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_students_in_lunchroom_l1593_159379


namespace value_of_z_l1593_159367

theorem value_of_z {x y z : ℤ} (h1 : x = 2) (h2 : y = x^2 - 5) (h3 : z = y^2 - 5) : z = -4 := by
  sorry

end value_of_z_l1593_159367


namespace find_k_l1593_159336

noncomputable def line1_slope : ℝ := -1
noncomputable def line2_slope (k : ℝ) : ℝ := -k / 3

theorem find_k (k : ℝ) : 
  (line2_slope k) * line1_slope = -1 → k = -3 := 
by
  sorry

end find_k_l1593_159336


namespace C_pays_228_for_cricket_bat_l1593_159305

def CostPriceA : ℝ := 152

def ProfitA (price : ℝ) : ℝ := 0.20 * price

def SellingPriceA (price : ℝ) : ℝ := price + ProfitA price

def ProfitB (price : ℝ) : ℝ := 0.25 * price

def SellingPriceB (price : ℝ) : ℝ := price + ProfitB price

theorem C_pays_228_for_cricket_bat :
  SellingPriceB (SellingPriceA CostPriceA) = 228 :=
by
  sorry

end C_pays_228_for_cricket_bat_l1593_159305


namespace john_new_bench_press_l1593_159373

theorem john_new_bench_press (initial_weight : ℕ) (decrease_percent : ℕ) (retain_percent : ℕ) (training_factor : ℕ) (final_weight : ℕ) 
  (h1 : initial_weight = 500)
  (h2 : decrease_percent = 80)
  (h3 : retain_percent = 20)
  (h4 : training_factor = 3)
  (h5 : final_weight = initial_weight * retain_percent / 100 * training_factor) : 
  final_weight = 300 := 
by sorry

end john_new_bench_press_l1593_159373


namespace minimum_value_of_sum_of_squares_l1593_159351

variable {x y : ℝ}

theorem minimum_value_of_sum_of_squares (h : x^2 + 2*x*y - y^2 = 7) : 
  x^2 + y^2 ≥ 7 * Real.sqrt 2 / 2 := by 
    sorry

end minimum_value_of_sum_of_squares_l1593_159351


namespace solution_set_of_f_x_gt_2_minimum_value_of_f_l1593_159385

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem solution_set_of_f_x_gt_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -7} ∪ {x : ℝ | x > 5 / 3} :=
by 
  sorry

theorem minimum_value_of_f : ∃ x : ℝ, f x = -9 / 2 :=
by 
  sorry

end solution_set_of_f_x_gt_2_minimum_value_of_f_l1593_159385


namespace ratio_steel_iron_is_5_to_2_l1593_159377

-- Definitions based on the given conditions
def amount_steel : ℕ := 35
def amount_iron : ℕ := 14

-- Main statement
theorem ratio_steel_iron_is_5_to_2 :
  (amount_steel / Nat.gcd amount_steel amount_iron) = 5 ∧
  (amount_iron / Nat.gcd amount_steel amount_iron) = 2 :=
by
  sorry

end ratio_steel_iron_is_5_to_2_l1593_159377


namespace factor_expression_l1593_159307

theorem factor_expression (x : ℝ) : 
  3 * x * (x - 5) + 4 * (x - 5) = (3 * x + 4) * (x - 5) :=
by
  sorry

end factor_expression_l1593_159307


namespace johns_number_l1593_159324

theorem johns_number (n : ℕ) :
  64 ∣ n ∧ 45 ∣ n ∧ 1000 < n ∧ n < 3000 -> n = 2880 :=
by
  sorry

end johns_number_l1593_159324


namespace number_of_small_companies_l1593_159350

theorem number_of_small_companies
  (large_companies : ℕ)
  (medium_companies : ℕ)
  (inspected_companies : ℕ)
  (inspected_medium_companies : ℕ)
  (total_inspected_companies : ℕ)
  (small_companies : ℕ)
  (inspection_fraction : ℕ → ℚ)
  (proportion : inspection_fraction 20 = 1 / 4)
  (H1 : large_companies = 4)
  (H2 : medium_companies = 20)
  (H3 : inspected_medium_companies = 5)
  (H4 : total_inspected_companies = 40)
  (H5 : inspected_companies = total_inspected_companies - large_companies - inspected_medium_companies)
  (H6 : small_companies = inspected_companies * 4)
  (correct_result : small_companies = 136) :
  small_companies = 136 :=
by sorry

end number_of_small_companies_l1593_159350
