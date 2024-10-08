import Mathlib

namespace polynomial_remainder_l136_136114

theorem polynomial_remainder (a : ℝ) (h : ∀ x : ℝ, x^3 + a * x^2 + 1 = (x^2 - 1) * (x + 2) + (x + 3)) : a = 2 :=
sorry

end polynomial_remainder_l136_136114


namespace store_earnings_correct_l136_136409

theorem store_earnings_correct :
  let graphics_cards_qty := 10
  let hard_drives_qty := 14
  let cpus_qty := 8
  let rams_qty := 4
  let psus_qty := 12
  let monitors_qty := 6
  let keyboards_qty := 18
  let mice_qty := 24

  let graphics_card_price := 600
  let hard_drive_price := 80
  let cpu_price := 200
  let ram_price := 60
  let psu_price := 90
  let monitor_price := 250
  let keyboard_price := 40
  let mouse_price := 20

  let total_earnings := graphics_cards_qty * graphics_card_price +
                        hard_drives_qty * hard_drive_price +
                        cpus_qty * cpu_price +
                        rams_qty * ram_price +
                        psus_qty * psu_price +
                        monitors_qty * monitor_price +
                        keyboards_qty * keyboard_price +
                        mice_qty * mouse_price
  total_earnings = 12740 :=
by
  -- definitions and calculations here
  sorry

end store_earnings_correct_l136_136409


namespace percentage_vets_recommend_puppy_kibble_l136_136038

theorem percentage_vets_recommend_puppy_kibble :
  ∀ (P : ℝ), (30 / 100 * 1000 = 300) → (1000 * P / 100 + 100 = 300) → P = 20 :=
by
  intros P h1 h2
  sorry

end percentage_vets_recommend_puppy_kibble_l136_136038


namespace probability_correct_l136_136984

namespace ProbabilitySongs

/-- Define the total number of ways to choose 2 out of 4 songs -/ 
def total_ways : ℕ := Nat.choose 4 2

/-- Define the number of ways to choose 2 songs such that neither A nor B is chosen (only C and D can be chosen) -/
def ways_without_AB : ℕ := Nat.choose 2 2

/-- The probability of playing at least one of A and B is calculated via the complementary rule -/
def probability_at_least_one_AB_played : ℚ := 1 - (ways_without_AB / total_ways)

theorem probability_correct : probability_at_least_one_AB_played = 5 / 6 := sorry
end ProbabilitySongs

end probability_correct_l136_136984


namespace tara_spent_more_on_icecream_l136_136457

def iceCreamCount : ℕ := 19
def yoghurtCount : ℕ := 4
def iceCreamCost : ℕ := 7
def yoghurtCost : ℕ := 1

theorem tara_spent_more_on_icecream :
  (iceCreamCount * iceCreamCost) - (yoghurtCount * yoghurtCost) = 129 := 
  sorry

end tara_spent_more_on_icecream_l136_136457


namespace train_departure_time_l136_136008

-- Conditions
def arrival_time : ℕ := 1000  -- Representing 10:00 as 1000 (in minutes since midnight)
def travel_time : ℕ := 15  -- 15 minutes

-- Definition of time subtraction
def time_sub (arrival : ℕ) (travel : ℕ) : ℕ :=
arrival - travel

-- Proof that the train left at 9:45
theorem train_departure_time : time_sub arrival_time travel_time = 945 := by
  sorry

end train_departure_time_l136_136008


namespace value_of_a_l136_136806

theorem value_of_a (a x : ℝ) (h1 : x = 2) (h2 : a * x = 4) : a = 2 :=
by
  sorry

end value_of_a_l136_136806


namespace initial_population_l136_136865

theorem initial_population (P : ℝ) (h : P * 1.21 = 12000) : P = 12000 / 1.21 :=
by sorry

end initial_population_l136_136865


namespace parallel_lines_slope_l136_136978

theorem parallel_lines_slope (m : ℝ) :
  (∀ x y : ℝ, (m + 3) * x + 4 * y + 3 * m - 5 = 0) ∧ (∀ x y : ℝ, 2 * x + (m + 5) * y - 8 = 0) →
  m = -7 :=
by
  intro H
  sorry

end parallel_lines_slope_l136_136978


namespace grasshopper_jump_distance_l136_136287

theorem grasshopper_jump_distance (g f m : ℕ)
    (h1 : f = g + 32)
    (h2 : m = f - 26)
    (h3 : m = 31) : g = 25 :=
by
  sorry

end grasshopper_jump_distance_l136_136287


namespace find_other_number_l136_136130

theorem find_other_number (a b : ℕ) (gcd_ab : Nat.gcd a b = 45) (lcm_ab : Nat.lcm a b = 1260) (a_eq : a = 180) : b = 315 :=
by
  -- proof goes here
  sorry

end find_other_number_l136_136130


namespace brick_piles_l136_136656

theorem brick_piles (x y z : ℤ) :
  2 * (x - 100) = y + 100 ∧
  x + z = 6 * (y - z) →
  x = 170 ∧ y = 40 :=
by
  sorry

end brick_piles_l136_136656


namespace car_distribution_l136_136546

theorem car_distribution :
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let fourth_supplier := (total_cars - (first_supplier + second_supplier + third_supplier)) / 2
  let fifth_supplier := fourth_supplier
  fourth_supplier = 325000 ∧ fifth_supplier = 325000 := 
by 
  sorry

end car_distribution_l136_136546


namespace tom_paid_450_l136_136789

-- Define the conditions
def hours_per_day : ℕ := 2
def number_of_days : ℕ := 3
def cost_per_hour : ℕ := 75

-- Calculated total number of hours Tom rented the helicopter
def total_hours_rented : ℕ := hours_per_day * number_of_days

-- Calculated total cost for renting the helicopter
def total_cost_rented : ℕ := total_hours_rented * cost_per_hour

-- Theorem stating that Tom paid $450 to rent the helicopter
theorem tom_paid_450 : total_cost_rented = 450 := by
  sorry

end tom_paid_450_l136_136789


namespace min_value_f_l136_136519

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 10)

theorem min_value_f (h : ∀ x > 10, f x ≥ 40) : ∀ x > 10, f x = 40 → x = 20 :=
by
  sorry

end min_value_f_l136_136519


namespace bounded_roots_l136_136124

open Polynomial

noncomputable def P : ℤ[X] := sorry -- Replace with actual polynomial if necessary

theorem bounded_roots (P : ℤ[X]) (n : ℕ) (hPdeg : P.degree = n) (hdec : 1 ≤ n) :
  ∀ k : ℤ, (P.eval k) ^ 2 = 1 → ∃ (r s : ℕ), r + s ≤ n + 2 := 
by 
  sorry

end bounded_roots_l136_136124


namespace counting_numbers_leave_remainder_6_divide_53_l136_136477

theorem counting_numbers_leave_remainder_6_divide_53 :
  ∃! n : ℕ, (∃ k : ℕ, 53 = n * k + 6) ∧ n > 6 :=
sorry

end counting_numbers_leave_remainder_6_divide_53_l136_136477


namespace john_eggs_per_week_l136_136119

theorem john_eggs_per_week
  (pens : ℕ)
  (emus_per_pen : ℕ)
  (female_ratio : ℚ)
  (eggs_per_female_per_day : ℕ)
  (days_in_week : ℕ) :
  pens = 4 →
  emus_per_pen = 6 →
  female_ratio = 1/2 →
  eggs_per_female_per_day = 1 →
  days_in_week = 7 →
  (pens * emus_per_pen * female_ratio * eggs_per_female_per_day * days_in_week = 84) :=
by
  intros h_pens h_emus h_ratio h_eggs h_days
  rw [h_pens, h_emus, h_ratio, h_eggs, h_days]
  norm_num

end john_eggs_per_week_l136_136119


namespace max_b_of_box_volume_l136_136186

theorem max_b_of_box_volume (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : Prime c) (h5 : a * b * c = 360) : b = 12 := 
sorry

end max_b_of_box_volume_l136_136186


namespace impossible_arrangement_l136_136041

theorem impossible_arrangement : 
  ∀ (a : Fin 111 → ℕ), (∀ i, a i ≤ 500) → (∀ i j, i ≠ j → a i ≠ a j) → 
  ¬ ∀ i : Fin 111, (a i % 10 = ((Finset.univ.sum (λ j => if j = i then 0 else a j)) % 10)) :=
by 
  sorry

end impossible_arrangement_l136_136041


namespace certain_number_l136_136853

theorem certain_number (x : ℝ) (h : x - 4 = 2) : x^2 - 3 * x = 18 :=
by
  -- Proof yet to be completed
  sorry

end certain_number_l136_136853


namespace determinant_inequality_l136_136318

open Real

def det (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_inequality (x : ℝ) :
  det 7 (x^2) 2 1 > det 3 (-2) 1 x ↔ -5/2 < x ∧ x < 1 :=
by
  sorry

end determinant_inequality_l136_136318


namespace correct_calculation_l136_136296

noncomputable def option_A : Prop := (Real.sqrt 3 + Real.sqrt 2) ≠ Real.sqrt 5
noncomputable def option_B : Prop := (Real.sqrt 3 * Real.sqrt 5) = Real.sqrt 15 ∧ Real.sqrt 15 ≠ 15
noncomputable def option_C : Prop := Real.sqrt (32 / 8) = 2 ∧ (Real.sqrt (32 / 8) ≠ -2)
noncomputable def option_D : Prop := (2 * Real.sqrt 3) - Real.sqrt 3 = Real.sqrt 3

theorem correct_calculation : option_D :=
by
  sorry

end correct_calculation_l136_136296


namespace polar_to_rectangular_l136_136270

theorem polar_to_rectangular (r θ : ℝ) (x y : ℝ) 
  (hr : r = 10) 
  (hθ : θ = (3 * Real.pi) / 4) 
  (hx : x = r * Real.cos θ) 
  (hy : y = r * Real.sin θ) 
  :
  x = -5 * Real.sqrt 2 ∧ y = 5 * Real.sqrt 2 := 
by
  -- We assume that the problem is properly stated
  -- Proof omitted here
  sorry

end polar_to_rectangular_l136_136270


namespace trains_pass_each_other_time_l136_136496

theorem trains_pass_each_other_time :
  ∃ t : ℝ, t = 240 / 191.171 := 
sorry

end trains_pass_each_other_time_l136_136496


namespace speed_of_man_in_still_water_l136_136778

-- Definition of the conditions
def effective_downstream_speed (v_m v_c : ℝ) : Prop := (v_m + v_c) = 10
def effective_upstream_speed (v_m v_c : ℝ) : Prop := (v_m - v_c) = 11.25

-- The proof problem statement
theorem speed_of_man_in_still_water (v_m v_c : ℝ) 
  (h1 : effective_downstream_speed v_m v_c)
  (h2 : effective_upstream_speed v_m v_c)
  : v_m = 10.625 :=
sorry

end speed_of_man_in_still_water_l136_136778


namespace pens_per_student_l136_136827

theorem pens_per_student (n : ℕ) (h1 : 0 < n) (h2 : n ≤ 50) (h3 : 100 % n = 0) (h4 : 50 % n = 0) : 100 / n = 2 :=
by
  -- proof goes here
  sorry

end pens_per_student_l136_136827


namespace base_of_number_l136_136218

theorem base_of_number (b : ℕ) : 
  (1 * b + 3)^2 = 2 * b^2 + 1 * b + 1 → b = 8 :=
by
  sorry

end base_of_number_l136_136218


namespace depth_of_canal_l136_136849

/-- The cross-section of a canal is a trapezium with a top width of 12 meters, 
a bottom width of 8 meters, and an area of 840 square meters. 
Prove that the depth of the canal is 84 meters.
-/
theorem depth_of_canal (top_width bottom_width area : ℝ) (h : ℝ) :
  top_width = 12 → bottom_width = 8 → area = 840 → 1 / 2 * (top_width + bottom_width) * h = area → h = 84 :=
by
  intros ht hb ha h_area
  sorry

end depth_of_canal_l136_136849


namespace days_c_worked_l136_136574

theorem days_c_worked (Da Db Dc : ℕ) (Wa Wb Wc : ℕ)
  (h1 : Da = 6) (h2 : Db = 9) (h3 : Wc = 100) (h4 : 3 * Wc = 5 * Wa)
  (h5 : 4 * Wc = 5 * Wb)
  (h6 : Wa * Da + Wb * Db + Wc * Dc = 1480) : Dc = 4 :=
by
  sorry

end days_c_worked_l136_136574


namespace seventeen_power_seven_mod_eleven_l136_136508

-- Define the conditions
def mod_condition : Prop := 17 % 11 = 6

-- Define the main goal (to prove the correct answer)
theorem seventeen_power_seven_mod_eleven (h : mod_condition) : (17^7) % 11 = 8 := by
  -- Proof goes here
  sorry

end seventeen_power_seven_mod_eleven_l136_136508


namespace find_angle_BDC_l136_136634

theorem find_angle_BDC
  (CAB CAD DBA DBC : ℝ)
  (h1 : CAB = 40)
  (h2 : CAD = 30)
  (h3 : DBA = 75)
  (h4 : DBC = 25) :
  ∃ BDC : ℝ, BDC = 45 :=
by
  sorry

end find_angle_BDC_l136_136634


namespace units_digit_of_n_l136_136801

theorem units_digit_of_n (n : ℕ) (h : n = 56^78 + 87^65) : (n % 10) = 3 :=
by
  sorry

end units_digit_of_n_l136_136801


namespace solve_for_x_l136_136717

theorem solve_for_x (x : ℕ) (h : 5 * (2 ^ x) = 320) : x = 6 :=
by
  sorry

end solve_for_x_l136_136717


namespace correct_calculated_value_l136_136559

theorem correct_calculated_value (n : ℕ) (h : n + 9 = 30) : n + 7 = 28 :=
by
  sorry

end correct_calculated_value_l136_136559


namespace part_I_part_II_l136_136014

noncomputable def f (x b c : ℝ) := x^2 + b*x + c

theorem part_I (x_1 x_2 b c : ℝ)
  (h1 : f x_1 b c = x_1) (h2 : f x_2 b c = x_2) (h3 : x_1 > 0) (h4 : x_2 - x_1 > 1) :
  b^2 > 2 * (b + 2 * c) :=
sorry

theorem part_II (x_1 x_2 b c t : ℝ)
  (h1 : f x_1 b c = x_1) (h2 : f x_2 b c = x_2) (h3 : x_1 > 0) (h4 : x_2 - x_1 > 1) (h5 : 0 < t ∧ t < x_1) :
  f t b c > x_1 :=
sorry

end part_I_part_II_l136_136014


namespace zoo_structure_l136_136413

theorem zoo_structure (P : ℕ) (h1 : ∃ (snakes monkeys elephants zebras : ℕ),
  snakes = 3 * P ∧
  monkeys = 6 * P ∧
  elephants = (P + snakes) / 2 ∧
  zebras = elephants - 3 ∧
  monkeys - zebras = 35) : P = 8 :=
sorry

end zoo_structure_l136_136413


namespace total_cats_l136_136810

variable (initialCats : ℝ)
variable (boughtCats : ℝ)

theorem total_cats (h1 : initialCats = 11.0) (h2 : boughtCats = 43.0) :
    initialCats + boughtCats = 54.0 :=
by
  sorry

end total_cats_l136_136810


namespace no_quadruples_solution_l136_136926

theorem no_quadruples_solution (a b c d : ℝ) :
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 →
    false :=
by 
  intros h
  sorry

end no_quadruples_solution_l136_136926


namespace smallest_N_l136_136651

theorem smallest_N (l m n : ℕ) (N : ℕ) (h_block : N = l * m * n)
  (h_invisible : (l - 1) * (m - 1) * (n - 1) = 120) :
  N = 216 :=
sorry

end smallest_N_l136_136651


namespace inequality1_inequality2_l136_136068

theorem inequality1 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) + a * b * c ≥ 2 * Real.sqrt 3 :=
by
  sorry

theorem inequality2 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  (1 / A) + (1 / B) + (1 / C) ≥ 9 / Real.pi :=
by
  sorry

end inequality1_inequality2_l136_136068


namespace intersecting_circles_l136_136243

theorem intersecting_circles (m n : ℝ) (h_intersect : ∃ c1 c2 : ℝ × ℝ, 
  (c1.1 - c1.2 - 2 = 0) ∧ (c2.1 - c2.2 - 2 = 0) ∧
  ∃ r1 r2 : ℝ, (c1.1 - 1)^2 + (c1.2 - 3)^2 = r1^2 ∧ (c2.1 - 1)^2 + (c2.2 - 3)^2 = r2^2 ∧
  (c1.1 - m)^2 + (c1.2 - n)^2 = r1^2 ∧ (c2.1 - m)^2 + (c2.2 - n)^2 = r2^2) :
  m + n = 4 :=
sorry

end intersecting_circles_l136_136243


namespace monotonically_increasing_range_a_l136_136407

theorem monotonically_increasing_range_a (a : ℝ) :
  (∀ x y : ℝ, x < y → (x^3 + a * x) ≤ (y^3 + a * y)) → a ≥ 0 := 
by
  sorry

end monotonically_increasing_range_a_l136_136407


namespace amount_of_water_formed_l136_136851

-- Define chemical compounds and reactions
def NaOH : Type := Unit
def HClO4 : Type := Unit
def NaClO4 : Type := Unit
def H2O : Type := Unit

-- Define the balanced chemical equation
def balanced_reaction (n_NaOH n_HClO4 : Int) : (n_NaOH = n_HClO4) → (n_NaOH = 1 → n_HClO4 = 1 → Int × Int × Int × Int) :=
  λ h_ratio h_NaOH h_HClO4 => 
    (n_NaOH, n_HClO4, 1, 1)  -- 1 mole of NaOH reacts with 1 mole of HClO4 to form 1 mole of NaClO4 and 1 mole of H2O

noncomputable def molar_mass_H2O : Float := 18.015 -- g/mol

theorem amount_of_water_formed :
  ∀ (n_NaOH n_HClO4 : Int), 
  (n_NaOH = 1 ∧ n_HClO4 = 1) →
  ((n_NaOH = n_HClO4) → molar_mass_H2O = 18.015) :=
by
  intros n_NaOH n_HClO4 h_condition h_ratio
  sorry

end amount_of_water_formed_l136_136851


namespace area_ratio_l136_136455

-- Definitions for the geometric entities
structure Point where
  x : ℚ
  y : ℚ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨0, 4⟩
def C : Point := ⟨2, 4⟩
def D : Point := ⟨2, 0⟩
def E : Point := ⟨1, 2⟩  -- Midpoint of BD
def F : Point := ⟨6 / 5, 0⟩  -- Given DF = 2/5 DA

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P Q R : Point) : ℚ :=
  (1 / 2) * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

-- Function to calculate the sum of the area of two triangles
def quadrilateral_area (P Q R S : Point) : ℚ :=
  triangle_area P Q R + triangle_area P R S

-- Prove the ratio of the areas
theorem area_ratio : 
  triangle_area D F E / quadrilateral_area A B E F = 4 / 13 := 
by {
  sorry
}

end area_ratio_l136_136455


namespace region_Z_probability_l136_136311

variable (P : Type) [Field P]
variable (P_X P_Y P_W P_Z : P)

theorem region_Z_probability :
  P_X = 1 / 3 → P_Y = 1 / 4 → P_W = 1 / 6 → P_X + P_Y + P_Z + P_W = 1 → P_Z = 1 / 4 := by
  sorry

end region_Z_probability_l136_136311


namespace midpoint_trace_quarter_circle_l136_136771

theorem midpoint_trace_quarter_circle (L : ℝ) (hL : 0 < L):
  ∃ (C : ℝ) (M : ℝ × ℝ → ℝ), 
    (∀ (x y : ℝ), x^2 + y^2 = L^2 → M (x, y) = C) ∧ 
    (C = (1/2) * L) ∧ 
    (∀ (x y : ℝ), M (x, y) = (x/2)^2 + (y/2)^2) → 
    ∀ (x y : ℝ), x^2 + y^2 = L^2 → (x/2)^2 + (y/2)^2 = (1/2 * L)^2 := 
by
  sorry

end midpoint_trace_quarter_circle_l136_136771


namespace james_total_earnings_l136_136027

def january_earnings : ℕ := 4000
def february_earnings : ℕ := january_earnings + (50 * january_earnings / 100)
def march_earnings : ℕ := february_earnings - (20 * february_earnings / 100)
def total_earnings : ℕ := january_earnings + february_earnings + march_earnings

theorem james_total_earnings :
  total_earnings = 14800 :=
by
  -- skip the proof
  sorry

end james_total_earnings_l136_136027


namespace exists_p_for_q_l136_136641

noncomputable def sqrt_56 : ℝ := Real.sqrt 56
noncomputable def sqrt_58 : ℝ := Real.sqrt 58

theorem exists_p_for_q (q : ℕ) (hq : q > 0) (hq_ne_1 : q ≠ 1) (hq_ne_3 : q ≠ 3) :
  ∃ p : ℤ, sqrt_56 < (p : ℝ) / q ∧ (p : ℝ) / q < sqrt_58 :=
by sorry

end exists_p_for_q_l136_136641


namespace opposite_of_7_l136_136536

-- Define the concept of an opposite number for real numbers
def is_opposite (x y : ℝ) : Prop := x = -y

-- Theorem statement
theorem opposite_of_7 :
  is_opposite 7 (-7) :=
by {
  sorry
}

end opposite_of_7_l136_136536


namespace f_one_equals_half_f_increasing_l136_136256

noncomputable def f : ℝ → ℝ := sorry

axiom f_add_half (x y : ℝ) : f (x + y) = f x + f y + 1/2

axiom f_half     : f (1/2) = 0

axiom f_positive (x : ℝ) (hx : x > 1/2) : f x > 0

theorem f_one_equals_half : f 1 = 1/2 := 
by 
  sorry

theorem f_increasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 > f x2 := 
by 
  sorry

end f_one_equals_half_f_increasing_l136_136256


namespace cats_left_l136_136924

theorem cats_left (siamese_cats : ℕ) (house_cats : ℕ) (cats_sold : ℕ) 
  (h1 : siamese_cats = 13) (h2 : house_cats = 5) (h3 : cats_sold = 10) : 
  siamese_cats + house_cats - cats_sold = 8 :=
by
  sorry

end cats_left_l136_136924


namespace most_probable_light_is_green_l136_136177

def duration_red := 30
def duration_yellow := 5
def duration_green := 40
def total_duration := duration_red + duration_yellow + duration_green

def prob_red := duration_red / total_duration
def prob_yellow := duration_yellow / total_duration
def prob_green := duration_green / total_duration

theorem most_probable_light_is_green : prob_green > prob_red ∧ prob_green > prob_yellow := 
  by
  sorry

end most_probable_light_is_green_l136_136177


namespace xy_system_solution_l136_136835

theorem xy_system_solution (x y : ℝ) (h₁ : x + 5 * y = 6) (h₂ : 3 * x - y = 2) : x + y = 2 := 
by 
  sorry

end xy_system_solution_l136_136835


namespace program_output_eq_l136_136947

theorem program_output_eq : ∀ (n : ℤ), n^2 + 3 * n - (2 * n^2 - n) = -n^2 + 4 * n := by
  intro n
  sorry

end program_output_eq_l136_136947


namespace necessary_but_not_sufficient_l136_136764

theorem necessary_but_not_sufficient (x : ℝ) : (x > -1) ↔ (∀ y : ℝ, (2 * y > 2) → (-1 < y)) :=
sorry

end necessary_but_not_sufficient_l136_136764


namespace least_possible_area_of_square_l136_136399

theorem least_possible_area_of_square :
  (∃ (side_length : ℝ), 3.5 ≤ side_length ∧ side_length < 4.5 ∧ 
    (∃ (area : ℝ), area = side_length * side_length ∧ 
    (∀ (side : ℝ), 3.5 ≤ side ∧ side < 4.5 → side * side ≥ 12.25))) :=
sorry

end least_possible_area_of_square_l136_136399


namespace find_pairs_l136_136538

theorem find_pairs :
  { (m, n) : ℕ × ℕ | (m > 0) ∧ (n > 0) ∧ (m^2 - n ∣ m + n^2)
      ∧ (n^2 - m ∣ n + m^2) } = { (2, 2), (3, 3), (1, 2), (2, 1), (3, 2), (2, 3) } :=
sorry

end find_pairs_l136_136538


namespace range_m_inequality_l136_136338

theorem range_m_inequality (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x^2 * Real.exp x < m) ↔ m > Real.exp 1 := 
  by
    sorry

end range_m_inequality_l136_136338


namespace sin_beta_l136_136713

variable (α β : ℝ)
variable (hα1 : 0 < α) (hα2 : α < Real.pi / 2)
variable (hβ1 : 0 < β) (hβ2: β < Real.pi / 2)
variable (h1 : Real.cos α = 5 / 13)
variable (h2 : Real.sin (α - β) = 4 / 5)

theorem sin_beta (α β : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
  (hβ1 : 0 < β) (hβ2 : β < Real.pi / 2) 
  (h1 : Real.cos α = 5 / 13) 
  (h2 : Real.sin (α - β) = 4 / 5) : 
  Real.sin β = 16 / 65 := 
by 
  sorry

end sin_beta_l136_136713


namespace f_increasing_f_t_range_l136_136381

noncomputable def f : Real → Real :=
  sorry

axiom f_prop1 : f 2 = 1
axiom f_prop2 : ∀ x, x > 1 → f x > 0
axiom f_prop3 : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y

theorem f_increasing (x1 x2 : Real) (hx1 : x1 > 0) (hx2 : x2 > 0) (h : x1 < x2) : f x1 < f x2 := by
  sorry

theorem f_t_range (t : Real) (ht : t > 0) (ht3 : t - 3 > 0) (hf : f t + f (t - 3) ≤ 2) : 3 < t ∧ t ≤ 4 := by
  sorry

end f_increasing_f_t_range_l136_136381


namespace solve_absolute_value_equation_l136_136993

theorem solve_absolute_value_equation (y : ℝ) : (|y - 4| + 3 * y = 15) ↔ (y = 19 / 4) := by
  sorry

end solve_absolute_value_equation_l136_136993


namespace find_a_l136_136138

theorem find_a (x a : ℝ) (h₁ : x^2 + x - 6 = 0) :
  (ax + 1 = 0 → (a = -1/2 ∨ a = -1/3) ∧ ax + 1 ≠ 0 ↔ false) := 
by
  sorry

end find_a_l136_136138


namespace A_work_days_l136_136389

theorem A_work_days (x : ℝ) :
  (1 / x + 1 / 6 + 1 / 12 = 7 / 24) → x = 24 :=
by
  intro h
  sorry

end A_work_days_l136_136389


namespace rectangles_perimeter_l136_136425

theorem rectangles_perimeter : 
  let base := 4
  let top := 4
  let left_side := 4
  let right_side := 6
  base + top + left_side + right_side = 18 := 
by {
  let base := 4
  let top := 4
  let left_side := 4
  let right_side := 6
  sorry
}

end rectangles_perimeter_l136_136425


namespace smallest_diff_PR_PQ_l136_136232

theorem smallest_diff_PR_PQ (PQ PR QR : ℤ) (h1 : PQ < PR) (h2 : PR ≤ QR) (h3 : PQ + PR + QR = 2021) : 
  ∃ PQ PR QR : ℤ, PQ < PR ∧ PR ≤ QR ∧ PQ + PR + QR = 2021 ∧ PR - PQ = 1 :=
by
  sorry

end smallest_diff_PR_PQ_l136_136232


namespace diana_shops_for_newborns_l136_136988

theorem diana_shops_for_newborns (total_children : ℕ) (num_toddlers : ℕ) (teenager_ratio : ℕ) (num_teens : ℕ) (num_newborns : ℕ)
    (h1 : total_children = 40) (h2 : num_toddlers = 6) (h3 : teenager_ratio = 5) (h4 : num_teens = teenager_ratio * num_toddlers) 
    (h5 : num_newborns = total_children - num_teens - num_toddlers) : 
    num_newborns = 4 := sorry

end diana_shops_for_newborns_l136_136988


namespace evaluate_division_l136_136486

theorem evaluate_division : 64 / 0.08 = 800 := by
  sorry

end evaluate_division_l136_136486


namespace ellipse_foci_x_axis_l136_136636

theorem ellipse_foci_x_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1) : 0 < a ∧ a < b :=
sorry

end ellipse_foci_x_axis_l136_136636


namespace shelves_filled_l136_136363

theorem shelves_filled (total_teddy_bears teddy_bears_per_shelf : ℕ) (h1 : total_teddy_bears = 98) (h2 : teddy_bears_per_shelf = 7) : 
  total_teddy_bears / teddy_bears_per_shelf = 14 := 
by 
  sorry

end shelves_filled_l136_136363


namespace smallest_possible_n_l136_136871

theorem smallest_possible_n : ∃ (n : ℕ), (∀ (r g b : ℕ), 24 * n = 18 * r ∧ 24 * n = 16 * g ∧ 24 * n = 20 * b) ∧ n = 30 :=
by
  -- Sorry, we're skipping the proof, as specified.
  sorry

end smallest_possible_n_l136_136871


namespace functional_relationship_l136_136237

-- Define the conditions
def directlyProportional (y x k : ℝ) : Prop :=
  y + 6 = k * (x + 1)

def specificCondition1 (x y : ℝ) : Prop :=
  x = 3 ∧ y = 2

-- State the theorem
theorem functional_relationship (k : ℝ) :
  (∀ x y, directlyProportional y x k) →
  specificCondition1 3 2 →
  ∀ x, ∃ y, y = 2 * x - 4 :=
by
  intro directProp
  intro specCond
  sorry

end functional_relationship_l136_136237


namespace nancy_carrots_l136_136586

-- Definitions based on the conditions
def initial_carrots := 12
def carrots_to_cook := 2
def new_carrot_seeds := 5
def growth_factor := 3
def kept_carrots := 10
def poor_quality_ratio := 3

-- Calculate new carrots grown from seeds
def new_carrots := new_carrot_seeds * growth_factor

-- Total carrots after new ones are added
def total_carrots := kept_carrots + new_carrots

-- Calculate poor quality carrots (integer part only)
def poor_quality_carrots := total_carrots / poor_quality_ratio

-- Calculate good quality carrots
def good_quality_carrots := total_carrots - poor_quality_carrots

-- Statement to prove
theorem nancy_carrots : good_quality_carrots = 17 :=
by
  sorry -- proof is not required

end nancy_carrots_l136_136586


namespace vets_recommend_yummy_dog_kibble_l136_136839

theorem vets_recommend_yummy_dog_kibble :
  (let total_vets := 1000
   let percentage_puppy_kibble := 20
   let vets_puppy_kibble := (percentage_puppy_kibble * total_vets) / 100
   let diff_yummy_puppy := 100
   let vets_yummy_kibble := vets_puppy_kibble + diff_yummy_puppy
   let percentage_yummy_kibble := (vets_yummy_kibble * 100) / total_vets
   percentage_yummy_kibble = 30) :=
by
  sorry

end vets_recommend_yummy_dog_kibble_l136_136839


namespace gcd_1729_867_l136_136999

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by
  sorry

end gcd_1729_867_l136_136999


namespace find_first_offset_l136_136092

theorem find_first_offset (d b : ℝ) (Area : ℝ) :
  d = 22 → b = 6 → Area = 165 → (first_offset : ℝ) → 22 * (first_offset + 6) / 2 = 165 → first_offset = 9 :=
by
  intros hd hb hArea first_offset heq
  sorry

end find_first_offset_l136_136092


namespace polynomial_roots_l136_136977

theorem polynomial_roots (k r : ℝ) (hk_pos : k > 0) 
(h_sum : r + 1 = 2 * k) (h_prod : r * 1 = k) : 
  r = 1 ∧ (∀ x, (x - 1) * (x - 1) = x^2 - 2 * x + 1) := 
by 
  sorry

end polynomial_roots_l136_136977


namespace find_expression_l136_136751

theorem find_expression 
  (E a : ℤ) 
  (h1 : (E + (3 * a - 8)) / 2 = 74) 
  (h2 : a = 28) : 
  E = 72 := 
by
  sorry

end find_expression_l136_136751


namespace sector_area_l136_136785

theorem sector_area (R : ℝ) (hR_pos : R > 0) (h_circumference : 4 * R = 2 * R + arc_length) :
  (1 / 2) * arc_length * R = R^2 :=
by sorry

end sector_area_l136_136785


namespace six_digit_number_representation_l136_136421

-- Defining that a is a two-digit number
def isTwoDigitNumber (a : ℕ) : Prop := a >= 10 ∧ a < 100

-- Defining that b is a four-digit number
def isFourDigitNumber (b : ℕ) : Prop := b >= 1000 ∧ b < 10000

-- The statement that placing a to the left of b forms the number 10000*a + b
theorem six_digit_number_representation (a b : ℕ) 
  (ha : isTwoDigitNumber a) 
  (hb : isFourDigitNumber b) : 
  (10000 * a + b) = (10^4 * a + b) :=
by
  sorry

end six_digit_number_representation_l136_136421


namespace repeating_decimal_as_fraction_l136_136156

theorem repeating_decimal_as_fraction :
  (0.58207 : ℝ) = 523864865 / 999900 := sorry

end repeating_decimal_as_fraction_l136_136156


namespace greatest_integer_b_l136_136773

theorem greatest_integer_b (b : ℤ) : (∀ x : ℝ, x^2 + (b : ℝ) * x + 7 ≠ 0) → b ≤ 5 :=
by sorry

end greatest_integer_b_l136_136773


namespace min_restoration_time_l136_136023

/-- Prove the minimum time required to complete the restoration work of three handicrafts. -/

def shaping_time_A : Nat := 9
def shaping_time_B : Nat := 16
def shaping_time_C : Nat := 10

def painting_time_A : Nat := 15
def painting_time_B : Nat := 8
def painting_time_C : Nat := 14

theorem min_restoration_time : 
  (shaping_time_A + painting_time_A + painting_time_C + painting_time_B) = 46 := by
  sorry

end min_restoration_time_l136_136023


namespace negation_is_false_l136_136958

-- Definitions corresponding to the conditions
def prop (x : ℝ) := x > 0 → x^2 > 0

-- Statement of the proof problem in Lean 4
theorem negation_is_false : ¬(∀ x : ℝ, ¬(x > 0 → x^2 > 0)) = false :=
by {
  sorry
}

end negation_is_false_l136_136958


namespace quadratic_inequality_solution_l136_136745

theorem quadratic_inequality_solution (x : ℝ) : (x^2 - 4 * x - 21 < 0) ↔ (-3 < x ∧ x < 7) :=
sorry

end quadratic_inequality_solution_l136_136745


namespace equation_solution_unique_l136_136253

theorem equation_solution_unique (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3) :
    (2 / (x - 3) = 3 / x ↔ x = 9) :=
by
  sorry

end equation_solution_unique_l136_136253


namespace inequality_x2_y2_l136_136082

theorem inequality_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  |x^2 + y^2| / (x + y) < |x^2 - y^2| / (x - y) :=
sorry

end inequality_x2_y2_l136_136082


namespace money_left_over_l136_136609

theorem money_left_over 
  (num_books : ℕ) 
  (price_per_book : ℝ) 
  (num_records : ℕ) 
  (price_per_record : ℝ) 
  (total_books : num_books = 200) 
  (book_price : price_per_book = 1.5) 
  (total_records : num_records = 75) 
  (record_price : price_per_record = 3) :
  (num_books * price_per_book - num_records * price_per_record) = 75 :=
by 
  -- calculation
  sorry

end money_left_over_l136_136609


namespace integral_eval_l136_136429

noncomputable def integral_problem : ℝ :=
  ∫ x in - (Real.pi / 2)..(Real.pi / 2), (x + Real.cos x)

theorem integral_eval : integral_problem = 2 :=
  by 
  sorry

end integral_eval_l136_136429


namespace sequence_divisibility_24_l136_136358

theorem sequence_divisibility_24 :
  ∀ (x : ℕ → ℕ), (x 0 = 2) → (x 1 = 3) →
    (∀ n : ℕ, x (n+2) = 7 * x (n+1) - x n + 280) →
    (∀ n : ℕ, (x n * x (n+1) + x (n+1) * x (n+2) + x (n+2) * x (n+3) + 2018) % 24 = 0) :=
by
  intro x h1 h2 h3
  sorry

end sequence_divisibility_24_l136_136358


namespace negation_of_all_men_are_tall_l136_136894

variable {α : Type}
variable (man : α → Prop) (tall : α → Prop)

theorem negation_of_all_men_are_tall :
  (¬ ∀ x, man x → tall x) ↔ ∃ x, man x ∧ ¬ tall x :=
sorry

end negation_of_all_men_are_tall_l136_136894


namespace cosine_midline_l136_136374

theorem cosine_midline (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_range : ∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) : 
  d = 3 := 
by 
  sorry

end cosine_midline_l136_136374


namespace gain_per_year_is_120_l136_136454

def principal := 6000
def rate_borrow := 4
def rate_lend := 6
def time := 2

def simple_interest (P R T : Nat) : Nat := P * R * T / 100

def interest_earned := simple_interest principal rate_lend time
def interest_paid := simple_interest principal rate_borrow time
def gain_in_2_years := interest_earned - interest_paid
def gain_per_year := gain_in_2_years / 2

theorem gain_per_year_is_120 : gain_per_year = 120 :=
by
  sorry

end gain_per_year_is_120_l136_136454


namespace trigonometric_identity_solution_l136_136267

theorem trigonometric_identity_solution (x : ℝ) (k : ℤ) :
  8.469 * (Real.sin x)^4 + 2 * (Real.cos x)^3 + 2 * (Real.sin x)^2 - Real.cos x + 1 = 0 ↔
  ∃ (k : ℤ), x = Real.pi + 2 * Real.pi * k := by
  sorry

end trigonometric_identity_solution_l136_136267


namespace ac_lt_bd_l136_136388

theorem ac_lt_bd (a b c d : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) : a * c < b * d :=
by
  sorry

end ac_lt_bd_l136_136388


namespace division_problem_l136_136691

theorem division_problem : (4 * 5) / 10 = 2 :=
by sorry

end division_problem_l136_136691


namespace factorize_expression_l136_136302

variable {R : Type} [CommRing R] (a x y : R)

theorem factorize_expression :
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l136_136302


namespace smallest_b_l136_136427

noncomputable def Q (b : ℤ) (x : ℤ) : ℤ := sorry -- Q is a polynomial, will be defined in proof

theorem smallest_b (b : ℤ) 
  (h1 : b > 0) 
  (h2 : ∀ x, x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 → Q b x = b) 
  (h3 : ∀ x, x = 1 ∨ x = 3 ∨ x = 5 ∨ x = 7 → Q b x = -b) 
  : b = 315 := sorry

end smallest_b_l136_136427


namespace tangent_line_at_1_l136_136251

-- Assume the curve and the point of tangency
noncomputable def curve (x : ℝ) : ℝ := x^3 - 2*x^2 + 4*x + 5

-- Define the point of tangency
def point_of_tangency : ℝ := 1

-- Define the expected tangent line equation in standard form Ax + By + C = 0
def tangent_line (x y : ℝ) : Prop := 3 * x - y + 5 = 0

theorem tangent_line_at_1 :
  tangent_line point_of_tangency (curve point_of_tangency) := 
sorry

end tangent_line_at_1_l136_136251


namespace distance_to_parabola_focus_l136_136230

theorem distance_to_parabola_focus :
  ∀ (x : ℝ), ((4 : ℝ) = (1 / 4) * x^2) → dist (0, 4) (0, 5) = 5 := 
by
  intro x
  intro hyp
  -- initial conditions indicate the distance is 5 and can be directly given
  sorry

end distance_to_parabola_focus_l136_136230


namespace three_hundred_percent_of_x_equals_seventy_five_percent_of_y_l136_136817

theorem three_hundred_percent_of_x_equals_seventy_five_percent_of_y
  (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 20) : y = 80 := by
  sorry

end three_hundred_percent_of_x_equals_seventy_five_percent_of_y_l136_136817


namespace number_of_rolls_in_case_l136_136796

-- Definitions: Cost of a case, cost per roll individually, percent savings per roll
def cost_of_case : ℝ := 9
def cost_per_roll_individual : ℝ := 1
def percent_savings_per_roll : ℝ := 0.25

-- Theorem: Proving the number of rolls in the case is 12
theorem number_of_rolls_in_case (n : ℕ) (h1 : cost_of_case = 9)
    (h2 : cost_per_roll_individual = 1)
    (h3 : percent_savings_per_roll = 0.25) : n = 12 := 
  sorry

end number_of_rolls_in_case_l136_136796


namespace find_ding_score_l136_136018

noncomputable def jia_yi_bing_avg_score : ℕ := 89
noncomputable def four_avg_score := jia_yi_bing_avg_score + 2
noncomputable def four_total_score := 4 * four_avg_score
noncomputable def jia_yi_bing_total_score := 3 * jia_yi_bing_avg_score
noncomputable def ding_score := four_total_score - jia_yi_bing_total_score

theorem find_ding_score : ding_score = 97 := 
by
  sorry

end find_ding_score_l136_136018


namespace find_y_l136_136767

theorem find_y (y : ℝ) (h : (17.28 / 12) / (3.6 * y) = 2) : y = 0.2 :=
by {
  sorry
}

end find_y_l136_136767


namespace quadratic_inequality_l136_136461

noncomputable def ax2_plus_bx_c (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |ax2_plus_bx_c a b c x| ≤ 1 / 2) →
  ∀ x : ℝ, |x| ≥ 1 → |ax2_plus_bx_c a b c x| ≤ x^2 - 1 / 2 :=
by
  sorry

end quadratic_inequality_l136_136461


namespace scheme_A_yield_percentage_l136_136950

-- Define the initial investments and yields
def initial_investment_A : ℝ := 300
def initial_investment_B : ℝ := 200
def yield_B : ℝ := 0.5 -- 50% yield

-- Define the equation given in the problem
def yield_A_equation (P : ℝ) : Prop :=
  initial_investment_A + (initial_investment_A * (P / 100)) = initial_investment_B + (initial_investment_B * yield_B) + 90

-- The proof statement we need to prove
theorem scheme_A_yield_percentage : yield_A_equation 30 :=
by
  sorry -- Proof is omitted

end scheme_A_yield_percentage_l136_136950


namespace algebraic_expression_eval_l136_136939

theorem algebraic_expression_eval (a b c : ℝ) (h : a * (-5:ℝ)^4 + b * (-5)^2 + c = 3): 
  a * (5:ℝ)^4 + b * (5)^2 + c = 3 :=
by
  sorry

end algebraic_expression_eval_l136_136939


namespace solve_for_y_l136_136852

variable (k y : ℝ)

-- Define the first equation for x
def eq1 (x : ℝ) : Prop := (1 / 2023) * x - 2 = 3 * x + k

-- Define the condition that x = -5 satisfies eq1
def condition1 : Prop := eq1 k (-5)

-- Define the second equation for y
def eq2 : Prop := (1 / 2023) * (2 * y + 1) - 5 = 6 * y + k

-- Prove that given condition1, y = -3 satisfies eq2
theorem solve_for_y : condition1 k → eq2 k (-3) :=
sorry

end solve_for_y_l136_136852


namespace condition_for_positive_expression_l136_136091

theorem condition_for_positive_expression (a b c : ℝ) :
  (∀ x y : ℝ, x^2 + x * y + y^2 + a * x + b * y + c > 0) ↔ a^2 - a * b + b^2 < 3 * c :=
by
  -- Proof should be provided here
  sorry

end condition_for_positive_expression_l136_136091


namespace weight_of_replaced_person_l136_136774

-- Define the conditions in Lean 4
variables {w_replaced : ℝ}   -- Weight of the person who was replaced
variables {w_new : ℝ}        -- Weight of the new person
variables {n : ℕ}            -- Number of persons
variables {avg_increase : ℝ} -- Increase in average weight

-- Set up the given conditions
axiom h1 : n = 8
axiom h2 : avg_increase = 2.5
axiom h3 : w_new = 40

-- Theorem that states the weight of the replaced person
theorem weight_of_replaced_person : w_replaced = 20 :=
by
  sorry

end weight_of_replaced_person_l136_136774


namespace solve_eq1_solve_eq2_l136_136715

-- For Equation (1)
theorem solve_eq1 (x : ℝ) : x^2 - 4*x - 6 = 0 → x = 2 + Real.sqrt 10 ∨ x = 2 - Real.sqrt 10 :=
sorry

-- For Equation (2)
theorem solve_eq2 (x : ℝ) : (x / (x - 1) - 1 = 3 / (x^2 - 1)) → x ≠ 1 ∧ x ≠ -1 → x = 2 :=
sorry

end solve_eq1_solve_eq2_l136_136715


namespace find_linear_odd_increasing_function_l136_136233

theorem find_linear_odd_increasing_function (f : ℝ → ℝ)
    (h1 : ∀ x, f (f x) = 4 * x)
    (h2 : ∀ x, f x = -f (-x))
    (h3 : ∀ x y, x < y → f x < f y)
    (h4 : ∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x) : 
    ∀ x, f x = 2 * x :=
by
  sorry

end find_linear_odd_increasing_function_l136_136233


namespace f_prime_neg_one_l136_136948

-- Given conditions and definitions
def f (x : ℝ) (a b c : ℝ) := a * x^4 + b * x^2 + c

def f_prime (x : ℝ) (a b : ℝ) := 4 * a * x^3 + 2 * b * x

-- The theorem we need to prove
theorem f_prime_neg_one (a b c : ℝ) (h : f_prime 1 a b = 2) : f_prime (-1) a b = -2 := by
  sorry

end f_prime_neg_one_l136_136948


namespace find_m_n_sum_l136_136542

noncomputable def point (x : ℝ) (y : ℝ) := (x, y)

def center_line (P : ℝ × ℝ) : Prop := P.1 - P.2 - 2 = 0

def on_circle (C : ℝ × ℝ) (P : ℝ × ℝ) (r : ℝ) : Prop := 
  (P.1 - C.1)^2 + (P.2 - C.2)^2 = r^2

def circles_intersect (A B C D : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  on_circle A C r₁ ∧ on_circle A D r₂ ∧ on_circle B C r₁ ∧ on_circle B D r₂

theorem find_m_n_sum 
  (A : ℝ × ℝ) (m n : ℝ)
  (C D : ℝ × ℝ)
  (r₁ r₂ : ℝ)
  (H1 : A = point 1 3)
  (H2 : circles_intersect A (point m n) C D r₁ r₂)
  (H3 : center_line C ∧ center_line D) :
  m + n = 4 :=
sorry

end find_m_n_sum_l136_136542


namespace train_length_eq_l136_136720

theorem train_length_eq 
  (speed_kmh : ℝ) (time_sec : ℝ) 
  (h_speed_kmh : speed_kmh = 126)
  (h_time_sec : time_sec = 6.856594329596489) : 
  ((speed_kmh * 1000 / 3600) * time_sec) = 239.9808045358781 :=
by
  -- We skip the proof with sorry, as per instructions
  sorry

end train_length_eq_l136_136720


namespace point_on_xoz_plane_l136_136618

def Point := ℝ × ℝ × ℝ

def lies_on_plane_xoz (p : Point) : Prop :=
  p.2 = 0

theorem point_on_xoz_plane :
  lies_on_plane_xoz (-2, 0, 3) :=
by
  sorry

end point_on_xoz_plane_l136_136618


namespace inequality_solution_l136_136432

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := |2 - 3 * x| ≥ 4

-- Define the solution set
def solution_set (x : ℝ) : Prop := x ≤ -2/3 ∨ x ≥ 2

-- The theorem that we need to prove
theorem inequality_solution : {x : ℝ | inequality_condition x} = {x : ℝ | solution_set x} :=
by sorry

end inequality_solution_l136_136432


namespace remaining_movie_time_l136_136353

def start_time := 200 -- represents 3:20 pm in total minutes from midnight
def end_time := 350 -- represents 5:44 pm in total minutes from midnight
def total_movie_duration := 180 -- 3 hours in minutes

theorem remaining_movie_time : total_movie_duration - (end_time - start_time) = 36 :=
by
  sorry

end remaining_movie_time_l136_136353


namespace find_x_l136_136741

noncomputable def arithmetic_sequence (x : ℝ) : Prop := 
  (x + 1) - (1/3) = 4 * x - (x + 1)

theorem find_x :
  ∃ x : ℝ, arithmetic_sequence x ∧ x = 5 / 6 :=
by
  use 5 / 6
  unfold arithmetic_sequence
  sorry

end find_x_l136_136741


namespace total_songs_performed_l136_136370

theorem total_songs_performed (lucy_songs : ℕ) (sarah_songs : ℕ) (beth_songs : ℕ) (jane_songs : ℕ) 
  (h1 : lucy_songs = 8)
  (h2 : sarah_songs = 5)
  (h3 : sarah_songs < beth_songs)
  (h4 : sarah_songs < jane_songs)
  (h5 : beth_songs < lucy_songs)
  (h6 : jane_songs < lucy_songs)
  (h7 : beth_songs = 6 ∨ beth_songs = 7)
  (h8 : jane_songs = 6 ∨ jane_songs = 7) :
  (lucy_songs + sarah_songs + beth_songs + jane_songs) / 3 = 9 :=
by
  sorry

end total_songs_performed_l136_136370


namespace saving_percentage_l136_136500

variable (I S : Real)

-- Conditions
def cond1 : Prop := S = 0.3 * I -- Man saves 30% of his income

def cond2 : Prop := let income_next_year := 1.3 * I
                    let savings_next_year := 2 * S
                    let expenditure_first_year := I - S
                    let expenditure_second_year := income_next_year - savings_next_year
                    expenditure_first_year + expenditure_second_year = 2 * expenditure_first_year

-- Question
theorem saving_percentage :
  cond1 I S →
  cond2 I S →
  S = 0.3 * I :=
by
  intros
  sorry

end saving_percentage_l136_136500


namespace proportionate_enlargement_l136_136475

theorem proportionate_enlargement 
  (original_width original_height new_width : ℕ)
  (h_orig_width : original_width = 3)
  (h_orig_height : original_height = 2)
  (h_new_width : new_width = 12) : 
  ∃ (new_height : ℕ), new_height = 8 :=
by
  -- sorry to skip proof
  sorry

end proportionate_enlargement_l136_136475


namespace find_integer_n_l136_136169

theorem find_integer_n (a b : ℕ) (n : ℕ)
  (h1 : n = 2^a * 3^b)
  (h2 : (2^(a+1) - 1) * ((3^(b+1) - 1) / (3 - 1)) = 1815) : n = 648 :=
  sorry

end find_integer_n_l136_136169


namespace ratio_out_of_state_to_in_state_l136_136719

/-
Given:
- total job applications Carly sent is 600
- job applications sent to companies in her state is 200

Prove:
- The ratio of job applications sent to companies in other states to the number sent to companies in her state is 2:1.
-/

def total_applications : ℕ := 600
def in_state_applications : ℕ := 200
def out_of_state_applications : ℕ := total_applications - in_state_applications

theorem ratio_out_of_state_to_in_state :
  (out_of_state_applications / in_state_applications) = 2 :=
by
  sorry

end ratio_out_of_state_to_in_state_l136_136719


namespace length_of_escalator_l136_136813

-- Define the conditions
def escalator_speed : ℝ := 15 -- ft/sec
def person_speed : ℝ := 5 -- ft/sec
def time_taken : ℝ := 10 -- sec

-- Define the length of the escalator
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ := 
  (escalator_speed + person_speed) * time

-- Theorem to prove
theorem length_of_escalator : escalator_length escalator_speed person_speed time_taken = 200 := by
  sorry

end length_of_escalator_l136_136813


namespace common_ratio_is_two_l136_136271

theorem common_ratio_is_two (a r : ℝ) (h_pos : a > 0) 
  (h_sum : a + a * r + a * r^2 + a * r^3 = 5 * (a + a * r)) : 
  r = 2 := 
by
  sorry

end common_ratio_is_two_l136_136271


namespace total_distance_covered_l136_136897

theorem total_distance_covered (h : ℝ) : (h > 0) → 
  ∑' n : ℕ, (h * (0.8 : ℝ) ^ n + h * (0.8 : ℝ) ^ (n + 1)) = 5 * h :=
  by
  sorry

end total_distance_covered_l136_136897


namespace cylinder_surface_area_l136_136093

theorem cylinder_surface_area (h r : ℝ) (h_height : h = 12) (r_radius : r = 4) : 
  2 * π * r * (r + h) = 128 * π :=
by
  -- providing the proof steps is beyond the scope of this task
  sorry

end cylinder_surface_area_l136_136093


namespace vector_dot_product_proof_l136_136821

def vector_dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_dot_product_proof : 
  let a := (-1, 2)
  let b := (2, 3)
  vector_dot_product a (a.1 - b.1, a.2 - b.2) = 1 :=
by {
  sorry
}

end vector_dot_product_proof_l136_136821


namespace least_number_of_trees_l136_136946

theorem least_number_of_trees :
  ∃ n : ℕ, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n % 7 = 0) ∧ n = 210 :=
by
  sorry

end least_number_of_trees_l136_136946


namespace cactus_species_minimum_l136_136307

theorem cactus_species_minimum :
  ∀ (collections : Fin 80 → Fin k → Prop),
  (∀ s : Fin k, ∃ (i : Fin 80), ¬ collections i s)
  → (∀ (c : Finset (Fin 80)), c.card = 15 → ∃ s : Fin k, ∀ (i : Fin 80), i ∈ c → collections i s)
  → 16 ≤ k := 
by 
  sorry

end cactus_species_minimum_l136_136307


namespace math_problem_l136_136030

noncomputable def f (x : ℝ) := |Real.exp x - 1|

theorem math_problem (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : x2 > 0)
  (h3 : - Real.exp x1 * Real.exp x2 = -1) :
  (x1 + x2 = 0) ∧
  (0 < (Real.exp x2 + Real.exp x1 - 2) / (x2 - x1)) ∧
  (0 < Real.exp x1 ∧ Real.exp x1 < 1) :=
by
  sorry

end math_problem_l136_136030


namespace simplify_expr1_simplify_expr2_l136_136009

variable (a b t : ℝ)

theorem simplify_expr1 : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l136_136009


namespace gasoline_price_increase_l136_136209

theorem gasoline_price_increase :
  let P_initial := 29.90
  let P_final := 149.70
  (P_final - P_initial) / P_initial * 100 = 400 :=
by
  let P_initial := 29.90
  let P_final := 149.70
  sorry

end gasoline_price_increase_l136_136209


namespace proof_problem_l136_136433

noncomputable def p : Prop := ∃ x : ℝ, x^2 + 1 / x^2 ≤ 2
def q : Prop := ¬ p

theorem proof_problem : q ∧ (p ∨ q) :=
by
  -- Insert proof here
  sorry

end proof_problem_l136_136433


namespace number_of_extreme_points_l136_136653

-- Define the function's derivative
def f_derivative (x : ℝ) : ℝ := (x + 1)^2 * (x - 1) * (x - 2)

-- State the theorem
theorem number_of_extreme_points : ∃ n : ℕ, n = 2 ∧ 
  (∀ x, (f_derivative x = 0 → ((f_derivative (x - ε) > 0 ∧ f_derivative (x + ε) < 0) ∨ 
                             (f_derivative (x - ε) < 0 ∧ f_derivative (x + ε) > 0))) → 
   (x = 1 ∨ x = 2)) :=
sorry

end number_of_extreme_points_l136_136653


namespace commencement_addresses_sum_l136_136598

noncomputable def addresses (S H L : ℕ) := 40

theorem commencement_addresses_sum
  (S H L : ℕ)
  (h1 : S = 12)
  (h2 : S = 2 * H)
  (h3 : L = S + 10) :
  S + H + L = addresses S H L :=
by
  sorry

end commencement_addresses_sum_l136_136598


namespace discount_percentage_is_10_l136_136430

-- Definitions of the conditions directly translated
def CP (MP : ℝ) : ℝ := 0.7 * MP
def GainPercent : ℝ := 0.2857142857142857
def SP (MP : ℝ) : ℝ := CP MP * (1 + GainPercent)

-- Using the alternative expression for selling price involving discount percentage
def DiscountSP (MP : ℝ) (D : ℝ) : ℝ := MP * (1 - D)

-- The theorem to prove the discount percentage is 10%
theorem discount_percentage_is_10 (MP : ℝ) : ∃ D : ℝ, DiscountSP MP D = SP MP ∧ D = 0.1 := 
by
  use 0.1
  sorry

end discount_percentage_is_10_l136_136430


namespace solve_system_of_equations_l136_136386

theorem solve_system_of_equations (a1 a2 a3 a4 : ℝ) (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (x1 x2 x3 x4 : ℝ)
  (h1 : |a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1)
  (h2 : |a2 - a1| * x1 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1)
  (h3 : |a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a4| * x4 = 1)
  (h4 : |a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 = 1) :
  x1 = 1 / (a4 - a1) ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 1 / (a4 - a1) := 
sorry

end solve_system_of_equations_l136_136386


namespace sequence_recurrence_l136_136206

noncomputable def a (n : ℕ) : ℤ := Int.floor ((1 + Real.sqrt 2) ^ n)

theorem sequence_recurrence (k : ℕ) (h : 2 ≤ k) : 
  ∀ n : ℕ, 
  (a 2 * k = 2 * a (2 * k - 1) + a (2 * k - 2)) ∧
  (a (2 * k + 1) = 2 * a (2 * k) + a (2 * k - 1) + 2) :=
sorry

end sequence_recurrence_l136_136206


namespace contrapositive_of_proposition_l136_136312

-- Proposition: If xy=0, then x=0
def proposition (x y : ℝ) : Prop := x * y = 0 → x = 0

-- Contrapositive: If x ≠ 0, then xy ≠ 0
def contrapositive (x y : ℝ) : Prop := x ≠ 0 → x * y ≠ 0

-- Proof that contrapositive of the given proposition holds
theorem contrapositive_of_proposition (x y : ℝ) : proposition x y ↔ contrapositive x y :=
by {
  sorry
}

end contrapositive_of_proposition_l136_136312


namespace find_coefficients_l136_136022

theorem find_coefficients (k b : ℝ) :
    (∀ x y : ℝ, (y = k * x) → ((x-2)^2 + y^2 = 1) → (2*x + y + b = 0)) →
    ((k = 1/2) ∧ (b = -4)) :=
by
  sorry

end find_coefficients_l136_136022


namespace count_terms_expansion_l136_136793

/-
This function verifies that the number of distinct terms in the expansion
of (a + b + c)(a + d + e + f + g) is equal to 15.
-/

theorem count_terms_expansion : 
    (a b c d e f g : ℕ) → 
    3 * 5 = 15 :=
by 
    intros a b c d e f g
    sorry

end count_terms_expansion_l136_136793


namespace asymptotes_of_hyperbola_l136_136578

variable (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
variable (h3 : (1 + b^2 / a^2) = (6 / 4))

theorem asymptotes_of_hyperbola :
  ∃ (m : ℝ), m = b / a ∧ (m = Real.sqrt 2 / 2) ∧ ∀ x : ℝ, (y = m*x) ∨ (y = -m*x) :=
by
  sorry

end asymptotes_of_hyperbola_l136_136578


namespace solve_for_x_minus_y_l136_136335

theorem solve_for_x_minus_y (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 24) : x - y = 4 := 
by
  sorry

end solve_for_x_minus_y_l136_136335


namespace find_f_600_l136_136698

variable (f : ℝ → ℝ)
variable (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y)
variable (h2 : f 500 = 3)

theorem find_f_600 : f 600 = 5 / 2 :=
by
  sorry

end find_f_600_l136_136698


namespace bear_problem_l136_136440

variables (w b br : ℕ)

theorem bear_problem 
    (h1 : b = 2 * w)
    (h2 : br = b + 40)
    (h3 : w + b + br = 190) :
    b = 60 :=
by
  sorry

end bear_problem_l136_136440


namespace k_gonal_number_proof_l136_136000

-- Definitions for specific k-gonal numbers based on given conditions.
def triangular_number (n : ℕ) := (1/2 : ℚ) * n^2 + (1/2 : ℚ) * n
def square_number (n : ℕ) := n^2
def pentagonal_number (n : ℕ) := (3/2 : ℚ) * n^2 - (1/2 : ℚ) * n
def hexagonal_number (n : ℕ) := 2 * n^2 - n

-- General definition for the k-gonal number
def k_gonal_number (n k : ℕ) : ℚ := ((k - 2) / 2) * n^2 + ((4 - k) / 2) * n

-- Corresponding Lean statement for the proof problem
theorem k_gonal_number_proof (n k : ℕ) (hk : k ≥ 3) :
    (k = 3 -> triangular_number n = k_gonal_number n k) ∧
    (k = 4 -> square_number n = k_gonal_number n k) ∧
    (k = 5 -> pentagonal_number n = k_gonal_number n k) ∧
    (k = 6 -> hexagonal_number n = k_gonal_number n k) ∧
    (n = 10 ∧ k = 24 -> k_gonal_number n k = 1000) :=
by
  intros
  sorry

end k_gonal_number_proof_l136_136000


namespace pot_filling_time_l136_136509

-- Define the given conditions
def drops_per_minute : ℕ := 3
def volume_per_drop : ℕ := 20 -- in ml
def pot_capacity : ℕ := 3000 -- in ml (3 liters * 1000 ml/liter)

-- Define the calculation for the drip rate
def drip_rate_per_minute : ℕ := drops_per_minute * volume_per_drop

-- Define the goal, i.e., how long it will take to fill the pot
def time_to_fill_pot (capacity : ℕ) (rate : ℕ) : ℕ := capacity / rate

-- Proof statement
theorem pot_filling_time :
  time_to_fill_pot pot_capacity drip_rate_per_minute = 50 := 
sorry

end pot_filling_time_l136_136509


namespace karl_present_salary_l136_136548

def original_salary : ℝ := 20000
def reduction_percentage : ℝ := 0.10
def increase_percentage : ℝ := 0.10

theorem karl_present_salary :
  let reduced_salary := original_salary * (1 - reduction_percentage)
  let present_salary := reduced_salary * (1 + increase_percentage)
  present_salary = 19800 :=
by
  sorry

end karl_present_salary_l136_136548


namespace exists_factor_between_10_and_20_l136_136165

theorem exists_factor_between_10_and_20 (n : ℕ) : ∃ k, (10 ≤ k ∧ k ≤ 20) ∧ k ∣ (2^n - 1) → k = 17 :=
by
  sorry

end exists_factor_between_10_and_20_l136_136165


namespace school_bought_50_cartons_of_markers_l136_136963

theorem school_bought_50_cartons_of_markers
  (n_puzzles : ℕ := 200)  -- the remaining amount after buying pencils
  (cost_per_carton_marker : ℕ := 4)  -- the cost per carton of markers
  :
  (n_puzzles / cost_per_carton_marker = 50) := -- the theorem to prove
by
  -- Provide skeleton proof strategy here
  sorry  -- details of the proof

end school_bought_50_cartons_of_markers_l136_136963


namespace cost_of_60_tulips_l136_136920

-- Definition of conditions
def cost_of_bouquet (n : ℕ) : ℝ :=
  if n ≤ 40 then n * 2
  else 40 * 2 + (n - 40) * 3

-- The main statement
theorem cost_of_60_tulips : cost_of_bouquet 60 = 140 := by
  sorry

end cost_of_60_tulips_l136_136920


namespace percentage_increase_of_sides_l136_136786

noncomputable def percentage_increase_in_area (L W : ℝ) (p : ℝ) : ℝ :=
  let A : ℝ := L * W
  let L' : ℝ := L * (1 + p / 100)
  let W' : ℝ := W * (1 + p / 100)
  let A' : ℝ := L' * W'
  ((A' - A) / A) * 100

theorem percentage_increase_of_sides (L W : ℝ) (hL : L > 0) (hW : W > 0) :
    percentage_increase_in_area L W 20 = 44 :=
by
  sorry

end percentage_increase_of_sides_l136_136786


namespace range_of_BC_in_triangle_l136_136330

theorem range_of_BC_in_triangle 
  (A B C : ℝ) 
  (a c BC : ℝ)
  (h1 : c = Real.sqrt 2)
  (h2 : a * Real.cos C = c * Real.sin A)
  (h3 : 0 < C ∧ C < Real.pi)
  (h4 : BC = 2 * Real.sin A)
  (h5 : ∃ A1 A2, 0 < A1 ∧ A1 < Real.pi / 2 ∧ Real.pi / 2 < A2 ∧ A2 < Real.pi ∧ Real.sin A = Real.sin A1 ∧ Real.sin A = Real.sin A2)
  : BC ∈ Set.Ioo (Real.sqrt 2) 2 :=
sorry

end range_of_BC_in_triangle_l136_136330


namespace prob_contact_l136_136742

variables (p : ℝ)
def prob_no_contact : ℝ := (1 - p) ^ 40

theorem prob_contact : 1 - prob_no_contact p = 1 - (1 - p) ^ 40 := by
  sorry

end prob_contact_l136_136742


namespace money_left_correct_l136_136483

-- Define the initial amount of money John had
def initial_money : ℝ := 10.50

-- Define the amount spent on sweets
def sweets_cost : ℝ := 2.25

-- Define the amount John gave to each friend
def gift_per_friend : ℝ := 2.20

-- Define the total number of friends
def number_of_friends : ℕ := 2

-- Calculate the total gifts given to friends
def total_gifts := gift_per_friend * (number_of_friends : ℝ)

-- Calculate the total amount spent
def total_spent := sweets_cost + total_gifts

-- Define the amount of money left
def money_left := initial_money - total_spent

-- The theorem statement
theorem money_left_correct : money_left = 3.85 := 
by 
  sorry

end money_left_correct_l136_136483


namespace product_of_terms_in_geometric_sequence_l136_136732

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m

noncomputable def roots_of_quadratic (a b c : ℝ) (r1 r2 : ℝ) : Prop :=
r1 * r2 = c

theorem product_of_terms_in_geometric_sequence
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : roots_of_quadratic 1 (-4) 3 (a 5) (a 7)) :
  a 2 * a 10 = 3 :=
sorry

end product_of_terms_in_geometric_sequence_l136_136732


namespace max_d_minus_r_l136_136665

theorem max_d_minus_r (d r : ℕ) (h1 : 2017 % d = r) (h2 : 1029 % d = r) (h3 : 725 % d = r) : 
  d - r = 35 :=
sorry

end max_d_minus_r_l136_136665


namespace horner_method_value_at_neg1_l136_136146

theorem horner_method_value_at_neg1 : 
  let f (x : ℤ) := 4 * x ^ 4 + 3 * x ^ 3 - 6 * x ^ 2 + x - 1
  let x := -1
  let v0 := 4
  let v1 := v0 * x + 3
  let v2 := v1 * x - 6
  v2 = -5 := by
  sorry

end horner_method_value_at_neg1_l136_136146


namespace total_floors_l136_136464

theorem total_floors (P Q R S T X F : ℕ) (h1 : 1 < X) (h2 : X < 50) :
  F = 1 + P - Q + R - S + T + X :=
sorry

end total_floors_l136_136464


namespace probability_correct_l136_136007

-- Definitions of given conditions
def P_AB := 2 / 3
def P_BC := 1 / 2

-- Probability that at least one road is at least 5 miles long
def probability_at_least_one_road_is_5_miles_long : ℚ :=
  1 - (1 - P_AB) * (1 - P_BC)

theorem probability_correct :
  probability_at_least_one_road_is_5_miles_long = 5 / 6 :=
by
  -- Proof goes here
  sorry

end probability_correct_l136_136007


namespace vector_addition_scalar_multiplication_l136_136164

def u : ℝ × ℝ × ℝ := (3, -2, 5)
def v : ℝ × ℝ × ℝ := (-1, 6, -3)
def result : ℝ × ℝ × ℝ := (4, 8, 4)

theorem vector_addition_scalar_multiplication :
  2 • (u + v) = result :=
by
  sorry

end vector_addition_scalar_multiplication_l136_136164


namespace sum_digits_of_3n_l136_136322

noncomputable def sum_digits (n : ℕ) : ℕ :=
sorry  -- Placeholder for a proper implementation of sum_digits

theorem sum_digits_of_3n (n : ℕ) 
  (h1 : sum_digits n = 100) 
  (h2 : sum_digits (44 * n) = 800) : 
  sum_digits (3 * n) = 300 := 
by
  sorry

end sum_digits_of_3n_l136_136322


namespace inequality_solution_l136_136097

theorem inequality_solution :
  {x : ℝ | |2 * x - 3| + |x + 1| < 7 ∧ x ≤ 4} = {x : ℝ | -5 / 3 < x ∧ x < 3} :=
by
  sorry

end inequality_solution_l136_136097


namespace square_area_is_8_point_0_l136_136572

theorem square_area_is_8_point_0 (A B C D E F : ℝ) 
    (h_square : E + F = 4)
    (h_diag : 1 + 2 + 1 = 4) : 
    ∃ (s : ℝ), s^2 = 8 :=
by
  sorry

end square_area_is_8_point_0_l136_136572


namespace perp_lines_a_value_l136_136244

theorem perp_lines_a_value :
  ∀ a : ℝ, ((a + 1) * 1 - 2 * (-a) = 0) → a = 1 :=
by
  intro a
  intro h
  -- We now state that a must satisfy the given condition and show that this leads to a = 1
  -- The proof is left as sorry
  sorry

end perp_lines_a_value_l136_136244


namespace find_a_given_even_l136_136760

def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem find_a_given_even (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 4 :=
by
  unfold f
  sorry

end find_a_given_even_l136_136760


namespace nonnegative_exists_l136_136219

theorem nonnegative_exists (a b c : ℝ) (h : a + b + c = 0) : a ≥ 0 ∨ b ≥ 0 ∨ c ≥ 0 :=
by
  sorry

end nonnegative_exists_l136_136219


namespace greenwood_school_l136_136962

theorem greenwood_school (f s : ℕ) (h : (3 / 4) * f = (1 / 3) * s) : s = 3 * f :=
by
  sorry

end greenwood_school_l136_136962


namespace minimum_a_plus_2c_l136_136650

theorem minimum_a_plus_2c (a c : ℝ) (h : (1 / a) + (1 / c) = 1) : a + 2 * c ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end minimum_a_plus_2c_l136_136650


namespace find_a_l136_136539

open Set

theorem find_a (A : Set ℝ) (B : Set ℝ) (f : ℝ → ℝ) (a : ℝ)
  (hA : A = Ici 0) 
  (hB : B = univ)
  (hf : ∀ x ∈ A, f x = 2^x - 1) 
  (ha_in_A : a ∈ A) 
  (ha_f_eq_3 : f a = 3) :
  a = 2 := 
by
  sorry

end find_a_l136_136539


namespace number_of_moles_of_HCl_l136_136922

-- Defining the chemical equation relationship
def reaction_relation (HCl NaHCO3 NaCl H2O CO2 : ℕ) : Prop :=
  H2O = HCl ∧ H2O = NaHCO3

-- Conditions
def conditions (HCl NaHCO3 H2O : ℕ) : Prop :=
  NaHCO3 = 3 ∧ H2O = 3

-- Theorem statement proving the number of moles of HCl given the conditions
theorem number_of_moles_of_HCl (HCl NaHCO3 NaCl H2O CO2 : ℕ) 
  (h1 : reaction_relation HCl NaHCO3 NaCl H2O CO2) 
  (h2 : conditions HCl NaHCO3 H2O) :
  HCl = 3 :=
sorry

end number_of_moles_of_HCl_l136_136922


namespace find_value_of_expression_l136_136820

theorem find_value_of_expression (a b : ℝ) (h : a + 2 * b - 1 = 0) : 3 * a + 6 * b = 3 :=
by
  sorry

end find_value_of_expression_l136_136820


namespace geometric_sequence_term_l136_136314

noncomputable def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_term {a : ℕ → ℤ} {q : ℤ}
  (h1 : geometric_sequence a q)
  (h2 : a 7 = 10)
  (h3 : q = -2) :
  a 10 = -80 :=
by
  sorry

end geometric_sequence_term_l136_136314


namespace cylinder_surface_area_l136_136976

theorem cylinder_surface_area (a b : ℝ) (h1 : a = 4 * Real.pi) (h2 : b = 8 * Real.pi) :
  (∃ S, S = 32 * Real.pi^2 + 8 * Real.pi ∨ S = 32 * Real.pi^2 + 32 * Real.pi) :=
by
  sorry

end cylinder_surface_area_l136_136976


namespace fraction_of_females_l136_136563

variable (participants_last_year males_last_year females_last_year males_this_year females_this_year participants_this_year : ℕ)

-- The conditions
def conditions :=
  males_last_year = 20 ∧
  participants_this_year = (110 * (participants_last_year/100)) ∧
  males_this_year = (105 * males_last_year / 100) ∧
  females_this_year = (120 * females_last_year / 100) ∧
  participants_last_year = males_last_year + females_last_year ∧
  participants_this_year = males_this_year + females_this_year

-- The proof statement
theorem fraction_of_females (h : conditions males_last_year females_last_year males_this_year females_this_year participants_last_year participants_this_year) :
  (females_this_year : ℚ) / (participants_this_year : ℚ) = 4 / 11 :=
  sorry

end fraction_of_females_l136_136563


namespace cost_price_of_watch_l136_136321

variable (CP SP1 SP2 : ℝ)

theorem cost_price_of_watch (h1 : SP1 = 0.9 * CP)
  (h2 : SP2 = 1.04 * CP)
  (h3 : SP2 = SP1 + 200) : CP = 10000 / 7 := 
by
  sorry

end cost_price_of_watch_l136_136321


namespace problem_domains_equal_l136_136061

/-- Proof problem:
    Prove that the domain of the function y = (x - 1)^(-1/2) is equal to the domain of the function y = ln(x - 1).
--/
theorem problem_domains_equal :
  {x : ℝ | x > 1} = {x : ℝ | x > 1} :=
by
  sorry

end problem_domains_equal_l136_136061


namespace not_possible_odd_sum_l136_136190

theorem not_possible_odd_sum (m n : ℤ) (h : (m ^ 2 + n ^ 2) % 2 = 0) : (m + n) % 2 ≠ 1 :=
sorry

end not_possible_odd_sum_l136_136190


namespace remainder_of_3024_l136_136077

theorem remainder_of_3024 (M : ℤ) (hM1 : M = 3024) (h_condition : ∃ k : ℤ, M = 24 * k + 13) :
  M % 1821 = 1203 :=
by
  sorry

end remainder_of_3024_l136_136077


namespace quadratic_root_condition_l136_136759

theorem quadratic_root_condition (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 1 ∧ x2 < 1 ∧ x1^2 + 2*a*x1 + 1 = 0 ∧ x2^2 + 2*a*x2 + 1 = 0) →
  a < -1 :=
by
  sorry

end quadratic_root_condition_l136_136759


namespace number_of_outfits_l136_136147

-- Definitions based on conditions
def trousers : ℕ := 4
def shirts : ℕ := 8
def jackets : ℕ := 3
def belts : ℕ := 2

-- The statement to prove
theorem number_of_outfits : trousers * shirts * jackets * belts = 192 := by
  sorry

end number_of_outfits_l136_136147


namespace find_num_biology_books_l136_136135

-- Given conditions
def num_chemistry_books : ℕ := 8
def total_ways_to_pick : ℕ := 2548

-- Function to calculate combinations
def combination (n k : ℕ) := n.choose k

-- Statement to be proved
theorem find_num_biology_books (B : ℕ) (h1 : combination num_chemistry_books 2 = 28) 
  (h2 : combination B 2 * 28 = total_ways_to_pick) : B = 14 :=
by 
  -- Proof goes here
  sorry

end find_num_biology_books_l136_136135


namespace units_digit_base8_of_sum_34_8_47_8_l136_136840

def is_units_digit (n m : ℕ) (u : ℕ) := (n + m) % 8 = u

theorem units_digit_base8_of_sum_34_8_47_8 :
  ∀ (n m : ℕ), n = 34 ∧ m = 47 → (is_units_digit (n % 8) (m % 8) 3) :=
by
  intros n m h
  rw [h.1, h.2]
  sorry

end units_digit_base8_of_sum_34_8_47_8_l136_136840


namespace exponent_equality_l136_136193

theorem exponent_equality (m : ℕ) (h : 9^4 = 3^m) : m = 8 := 
  sorry

end exponent_equality_l136_136193


namespace one_over_x_plus_one_over_y_eq_fifteen_l136_136709

theorem one_over_x_plus_one_over_y_eq_fifteen
  (x y : ℝ)
  (h1 : xy > 0)
  (h2 : 1 / xy = 5)
  (h3 : (x + y) / 5 = 0.6) : 
  (1 / x) + (1 / y) = 15 := 
by
  sorry

end one_over_x_plus_one_over_y_eq_fifteen_l136_136709


namespace geo_seq_a12_equal_96_l136_136772

def is_geometric (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geo_seq_a12_equal_96
  (a : ℕ → ℝ) (q : ℝ)
  (h0 : 1 < q)
  (h1 : is_geometric a q)
  (h2 : a 3 * a 7 = 72)
  (h3 : a 2 + a 8 = 27) :
  a 12 = 96 :=
sorry

end geo_seq_a12_equal_96_l136_136772


namespace cube_of_odd_sum_l136_136213

theorem cube_of_odd_sum (a : ℕ) (h1 : 1 < a) (h2 : ∃ (n : ℕ), (n = (a - 1) + 2 * (a - 1) + 1) ∧ n = 1979) : a = 44 :=
sorry

end cube_of_odd_sum_l136_136213


namespace inequality_solution_sum_of_squares_geq_sum_of_products_l136_136658

-- Problem 1
theorem inequality_solution (x : ℝ) : (0 < x ∧ x < 2/3) ↔ (x + 2) / (2 - 3 * x) > 1 :=
by
  sorry

-- Problem 2
theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
by
  sorry

end inequality_solution_sum_of_squares_geq_sum_of_products_l136_136658


namespace line_circle_no_intersection_l136_136903

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l136_136903


namespace molecular_weight_of_Y_l136_136597

def molecular_weight_X : ℝ := 136
def molecular_weight_C6H8O7 : ℝ := 192
def moles_C6H8O7 : ℝ := 5

def total_mass_reactants := molecular_weight_X + moles_C6H8O7 * molecular_weight_C6H8O7

theorem molecular_weight_of_Y :
  total_mass_reactants = 1096 := by
  sorry

end molecular_weight_of_Y_l136_136597


namespace angle_bisector_slope_l136_136608

-- Definitions of the conditions
def line1_slope := 2
def line2_slope := 4

-- The proof statement: Prove that the slope of the angle bisector is -12/7
theorem angle_bisector_slope : (line1_slope + line2_slope + Real.sqrt (line1_slope^2 + line2_slope^2 + 2 * line1_slope * line2_slope)) / 
                               (1 - line1_slope * line2_slope) = -12/7 :=
by
  sorry

end angle_bisector_slope_l136_136608


namespace travel_time_l136_136285

theorem travel_time (speed distance time : ℕ) (h_speed : speed = 60) (h_distance : distance = 180) : 
  time = distance / speed → time = 3 := by
  sorry

end travel_time_l136_136285


namespace jerry_remaining_debt_l136_136704

theorem jerry_remaining_debt :
  ∀ (paid_two_months_ago paid_last_month total_debt: ℕ),
  paid_two_months_ago = 12 →
  paid_last_month = paid_two_months_ago + 3 →
  total_debt = 50 →
  total_debt - (paid_two_months_ago + paid_last_month) = 23 :=
by
  intros paid_two_months_ago paid_last_month total_debt h1 h2 h3
  sorry

end jerry_remaining_debt_l136_136704


namespace distinct_values_for_T_l136_136246

-- Define the conditions given in the problem:
def distinct_digits (n : ℕ) : Prop :=
  n / 1000 ≠ (n / 100 % 10) ∧ n / 1000 ≠ (n / 10 % 10) ∧ n / 1000 ≠ (n % 10) ∧
  (n / 100 % 10) ≠ (n / 10 % 10) ∧ (n / 100 % 10) ≠ (n % 10) ∧
  (n / 10 % 10) ≠ (n % 10)

def Psum (P S T : ℕ) : Prop := P + S = T

-- Main theorem statement:
theorem distinct_values_for_T : ∀ (P S T : ℕ),
  distinct_digits P ∧ distinct_digits S ∧ distinct_digits T ∧
  Psum P S T → 
  (∃ (values : Finset ℕ), values.card = 7 ∧ ∀ val ∈ values, val = T) :=
by
  sorry

end distinct_values_for_T_l136_136246


namespace proof_problem_l136_136149

open Real

noncomputable def problem (c d : ℝ) : ℝ :=
  5^(c / d) + 2^(d / c)

theorem proof_problem :
  let c := log 8
  let d := log 25
  problem c d = 2 * sqrt 2 + 5^(2 / 3) :=
by
  intro c d
  have c_def : c = log 8 := rfl
  have d_def : d = log 25 := rfl
  rw [c_def, d_def]
  sorry

end proof_problem_l136_136149


namespace carpet_area_l136_136928

/-- A rectangular floor with a length of 15 feet and a width of 12 feet needs 20 square yards of carpet to cover it. -/
theorem carpet_area (length_feet : ℕ) (width_feet : ℕ) (feet_per_yard : ℕ) (length_yards : ℕ) (width_yards : ℕ) (area_sq_yards : ℕ) :
  length_feet = 15 ∧
  width_feet = 12 ∧
  feet_per_yard = 3 ∧
  length_yards = length_feet / feet_per_yard ∧
  width_yards = width_feet / feet_per_yard ∧
  area_sq_yards = length_yards * width_yards → 
  area_sq_yards = 20 :=
by
  sorry

end carpet_area_l136_136928


namespace coin_flip_probability_l136_136471

theorem coin_flip_probability : 
  ∀ (prob_tails : ℚ) (seq : List (Bool × ℚ)),
    prob_tails = 1/2 →
    seq = [(true, 1/2), (true, 1/2), (false, 1/2), (false, 1/2)] →
    (seq.map Prod.snd).prod = 0.0625 :=
by 
  intros prob_tails seq htails hseq 
  sorry

end coin_flip_probability_l136_136471


namespace not_parallel_to_a_l136_136954

noncomputable def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * u.1, k * u.2)

theorem not_parallel_to_a : ∀ k : ℝ, ¬ is_parallel (k^2 + 1, k^2 + 1) (1, -2) :=
sorry

end not_parallel_to_a_l136_136954


namespace find_number_l136_136873

theorem find_number (x : ℕ) (h : x - 18 = 3 * (86 - x)) : x = 69 :=
by
  sorry

end find_number_l136_136873


namespace largest_tile_side_length_l136_136657

theorem largest_tile_side_length (w l : ℕ) (hw : w = 120) (hl : l = 96) : 
  ∃ s, s = Nat.gcd w l ∧ s = 24 :=
by
  sorry

end largest_tile_side_length_l136_136657


namespace find_quadratic_function_l136_136929

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant condition
def discriminant_zero (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

-- Given derivative
def given_derivative (x : ℝ) : ℝ := 2 * x + 2

-- Prove that if these conditions hold, then f(x) = x^2 + 2x + 1
theorem find_quadratic_function :
  ∃ (a b c : ℝ), (∀ x, quadratic_function a b c x = 0 → discriminant_zero a b c) ∧
                (∀ x, (2 * a * x + b) = given_derivative x) ∧
                (quadratic_function a b c x = x^2 + 2 * x + 1) := 
by
  sorry

end find_quadratic_function_l136_136929


namespace smallest_prime_8_less_than_square_l136_136579

theorem smallest_prime_8_less_than_square :
  ∃ p : ℕ, (∃ n : ℤ, p = n^2 - 8) ∧ Nat.Prime p ∧ p > 0 ∧ (∀ q : ℕ, (∃ m : ℤ, q = m^2 - 8) ∧ Nat.Prime q → q ≥ p) :=
sorry

end smallest_prime_8_less_than_square_l136_136579


namespace sam_dimes_l136_136095

theorem sam_dimes (dimes_original dimes_given : ℕ) :
  dimes_original = 9 → dimes_given = 7 → dimes_original + dimes_given = 16 :=
by
  intros h1 h2
  sorry

end sam_dimes_l136_136095


namespace odd_function_condition_l136_136970

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.sin (ω * x + φ)

theorem odd_function_condition (A ω : ℝ) (hA : 0 < A) (hω : 0 < ω) (φ : ℝ) :
  (f A ω φ 0 = 0) ↔ (f A ω φ) = fun x => -f A ω φ (-x) := 
by
  sorry

end odd_function_condition_l136_136970


namespace triangle_shape_l136_136661

-- Defining the conditions:
variables (A B C a b c : ℝ)
variable (h1 : c - a * Real.cos B = (2 * a - b) * Real.cos A)

-- Defining the property to prove:
theorem triangle_shape : 
  (A = Real.pi / 2 ∨ A = B ∨ B = C ∨ C = A + B) :=
sorry

end triangle_shape_l136_136661


namespace reggie_marbles_bet_l136_136568

theorem reggie_marbles_bet 
  (initial_marbles : ℕ) (final_marbles : ℕ) (games_played : ℕ) (games_lost : ℕ) (bet_per_game : ℕ)
  (h_initial : initial_marbles = 100) 
  (h_final : final_marbles = 90) 
  (h_games : games_played = 9) 
  (h_losses : games_lost = 1) : 
  bet_per_game = 13 :=
by
  sorry

end reggie_marbles_bet_l136_136568


namespace liquor_and_beer_cost_l136_136866

-- Define the variables and conditions
variables (p_liquor p_beer : ℕ)

-- Main theorem to prove
theorem liquor_and_beer_cost (h1 : 2 * p_liquor + 12 * p_beer = 56)
                             (h2 : p_liquor = 8 * p_beer) :
  p_liquor + p_beer = 18 :=
sorry

end liquor_and_beer_cost_l136_136866


namespace smallest_int_x_l136_136776

theorem smallest_int_x (x : ℤ) (h : 2 * x + 5 < 3 * x - 10) : x = 16 :=
sorry

end smallest_int_x_l136_136776


namespace existence_of_function_implies_a_le_1_l136_136889

open Real

noncomputable def positive_reals := { x : ℝ // 0 < x }

theorem existence_of_function_implies_a_le_1 (a : ℝ) :
  (∃ f : positive_reals → positive_reals, ∀ x : positive_reals, 3 * (f x).val^2 = 2 * (f (f x)).val + a * x.val^4) → a ≤ 1 :=
by
  sorry

end existence_of_function_implies_a_le_1_l136_136889


namespace average_study_difference_is_6_l136_136468

def study_time_differences : List ℤ := [15, -5, 25, -10, 40, -30, 10]

def total_sum (lst : List ℤ) : ℤ := lst.foldr (· + ·) 0

def number_of_days : ℤ := 7

def average_difference : ℤ := total_sum study_time_differences / number_of_days

theorem average_study_difference_is_6 : average_difference = 6 :=
by
  unfold average_difference
  unfold total_sum 
  sorry

end average_study_difference_is_6_l136_136468


namespace pentagon_number_arrangement_l136_136258

def no_common_divisor_other_than_one (a b : ℕ) : Prop :=
  ∀ d : ℕ, d > 1 → (d ∣ a ∧ d ∣ b) → false

def has_common_divisor_greater_than_one (a b : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d ∣ a ∧ d ∣ b

theorem pentagon_number_arrangement :
  ∃ (A B C D E : ℕ),
    no_common_divisor_other_than_one A B ∧
    no_common_divisor_other_than_one B C ∧
    no_common_divisor_other_than_one C D ∧
    no_common_divisor_other_than_one D E ∧
    no_common_divisor_other_than_one E A ∧
    has_common_divisor_greater_than_one A C ∧
    has_common_divisor_greater_than_one A D ∧
    has_common_divisor_greater_than_one B D ∧
    has_common_divisor_greater_than_one B E ∧
    has_common_divisor_greater_than_one C E :=
sorry

end pentagon_number_arrangement_l136_136258


namespace ellipse_foci_k_value_l136_136688

theorem ellipse_foci_k_value 
    (k : ℝ) 
    (h1 : 5 * (0:ℝ)^2 + k * (2:ℝ)^2 = 5): 
    k = 1 := 
by 
  sorry

end ellipse_foci_k_value_l136_136688


namespace solve_for_x_l136_136033

theorem solve_for_x : 
  ∃ x : ℝ, (x^2 + 6 * x + 8 = -(x + 2) * (x + 6)) ∧ (x = -2 ∨ x = -5) :=
sorry

end solve_for_x_l136_136033


namespace original_monthly_bill_l136_136697

-- Define the necessary conditions
def increased_bill (original: ℝ): ℝ := original + 0.3 * original
def total_bill_after_increase : ℝ := 78

-- The proof we need to construct
theorem original_monthly_bill (X : ℝ) (H : increased_bill X = total_bill_after_increase) : X = 60 :=
by {
    sorry -- Proof is not required, only statement
}

end original_monthly_bill_l136_136697


namespace no_positive_integer_solutions_l136_136057

theorem no_positive_integer_solutions (x y : ℕ) (hx : x > 0) (hy : y > 0) : x^5 ≠ y^2 + 4 := 
by sorry

end no_positive_integer_solutions_l136_136057


namespace coefficient_a9_l136_136241

theorem coefficient_a9 (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℤ) :
  (x^2 + x^10 = a0 + a1 * (x + 1) + a2 * (x + 1)^2 + a3 * (x + 1)^3 +
   a4 * (x + 1)^4 + a5 * (x + 1)^5 + a6 * (x + 1)^6 + a7 * (x + 1)^7 +
   a8 * (x + 1)^8 + a9 * (x + 1)^9 + a10 * (x + 1)^10) →
  a10 = 1 →
  a9 = -10 :=
by
  sorry

end coefficient_a9_l136_136241


namespace team_savings_with_discount_l136_136317

def regular_shirt_cost : ℝ := 7.50
def regular_pants_cost : ℝ := 15.00
def regular_socks_cost : ℝ := 4.50
def discounted_shirt_cost : ℝ := 6.75
def discounted_pants_cost : ℝ := 13.50
def discounted_socks_cost : ℝ := 3.75
def team_size : ℕ := 12

theorem team_savings_with_discount :
  let regular_uniform_cost := regular_shirt_cost + regular_pants_cost + regular_socks_cost
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  total_savings = 36 := by
  sorry

end team_savings_with_discount_l136_136317


namespace find_number_l136_136495

theorem find_number
  (P : ℝ) (R : ℝ) (hP : P = 0.0002) (hR : R = 2.4712) :
  (12356 * P = R) := by
  sorry

end find_number_l136_136495


namespace binomial_12_3_equals_220_l136_136699

theorem binomial_12_3_equals_220 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_equals_220_l136_136699


namespace chess_pieces_missing_l136_136085

theorem chess_pieces_missing (total_pieces present_pieces missing_pieces : ℕ) 
  (h1 : total_pieces = 32)
  (h2 : present_pieces = 22)
  (h3 : missing_pieces = total_pieces - present_pieces) :
  missing_pieces = 10 :=
by
  sorry

end chess_pieces_missing_l136_136085


namespace find_abc_l136_136635

-- Definitions based on given conditions
variables (a b c : ℝ)
variable (h1 : a * b = 30 * (3 ^ (1/3)))
variable (h2 : a * c = 42 * (3 ^ (1/3)))
variable (h3 : b * c = 18 * (3 ^ (1/3)))

-- Formal statement of the proof problem
theorem find_abc : a * b * c = 90 * Real.sqrt 3 :=
by
  sorry

end find_abc_l136_136635


namespace mixture_weight_l136_136137

theorem mixture_weight (a b : ℝ) (h1 : a = 26.1) (h2 : a / (a + b) = 9 / 20) : a + b = 58 :=
sorry

end mixture_weight_l136_136137


namespace regular_12gon_symmetry_and_angle_l136_136035

theorem regular_12gon_symmetry_and_angle :
  ∀ (L R : ℕ), 
  (L = 12) ∧ (R = 30) → 
  (L + R = 42) :=
by
  -- placeholder for the actual proof
  sorry

end regular_12gon_symmetry_and_angle_l136_136035


namespace problem_solution_l136_136109

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 10
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 10
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 10
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 10

theorem problem_solution : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 0 :=
by
  sorry

end problem_solution_l136_136109


namespace amount_spent_on_raw_materials_l136_136234

-- Given conditions
def spending_on_machinery : ℝ := 125
def spending_as_cash (total_amount : ℝ) : ℝ := 0.10 * total_amount
def total_amount : ℝ := 250

-- Mathematically equivalent problem
theorem amount_spent_on_raw_materials :
  (X : ℝ) → X + spending_on_machinery + spending_as_cash total_amount = total_amount →
    X = 100 :=
by
  (intro X h)
  sorry

end amount_spent_on_raw_materials_l136_136234


namespace half_angle_quadrant_l136_136530

-- Define the given condition
def is_angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90

-- Define the result that needs to be proved
def is_angle_in_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, (k * 180 < α / 2 ∧ α / 2 < k * 180 + 45) ∨ (k * 180 + 180 < α / 2 ∧ α / 2 < k * 180 + 225)

-- The main theorem statement
theorem half_angle_quadrant (α : ℝ) (h : is_angle_in_first_quadrant α) : is_angle_in_first_or_third_quadrant α :=
sorry

end half_angle_quadrant_l136_136530


namespace E_72_eq_9_l136_136212

def E (n : ℕ) : ℕ :=
  -- Assume a function definition counting representations
  -- (this function body is a placeholder, as the exact implementation
  -- is not part of the problem statement)
  sorry

theorem E_72_eq_9 :
  E 72 = 9 :=
sorry

end E_72_eq_9_l136_136212


namespace arithmetic_sequence_common_difference_l136_136126

theorem arithmetic_sequence_common_difference (a : Nat → Int)
  (h1 : a 1 = 2) 
  (h3 : a 3 = 8)
  (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1))  -- General form for an arithmetic sequence given two terms
  : a 2 - a 1 = 3 :=
by
  -- The main steps of the proof will follow from the arithmetic progression properties
  sorry

end arithmetic_sequence_common_difference_l136_136126


namespace combined_PP_curve_l136_136036

-- Definitions based on the given conditions
def M1 (K : ℝ) : ℝ := 40 - 2 * K
def M2 (K : ℝ) : ℝ := 64 - K ^ 2
def combinedPPC (K1 K2 : ℝ) : ℝ := 128 - 0.5 * K1^2 + 40 - 2 * K2

theorem combined_PP_curve (K : ℝ) :
  (K ≤ 2 → combinedPPC K 0 = 168 - 0.5 * K^2) ∧
  (2 < K ∧ K ≤ 22 → combinedPPC 2 (K - 2) = 170 - 2 * K) ∧
  (22 < K ∧ K ≤ 36 → combinedPPC (K - 20) 20 = 20 * K - 0.5 * K^2 - 72) :=
by
  sorry

end combined_PP_curve_l136_136036


namespace juniors_score_l136_136637

/-- Mathematical proof problem stated in Lean 4 -/
theorem juniors_score 
  (total_students : ℕ) 
  (juniors seniors : ℕ)
  (junior_score senior_avg total_avg : ℝ)
  (h_total_students : total_students > 0)
  (h_juniors : juniors = total_students / 10)
  (h_seniors : seniors = (total_students * 9) / 10)
  (h_total_avg : total_avg = 84)
  (h_senior_avg : senior_avg = 83)
  (h_junior_score_same : ∀ j : ℕ, j < juniors → ∃ s : ℝ, s = junior_score)
  :
  junior_score = 93 :=
by
  sorry

end juniors_score_l136_136637


namespace b_plus_d_l136_136955

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem b_plus_d 
  (a b c d : ℝ) 
  (h1 : f a b c d 1 = 20) 
  (h2 : f a b c d (-1) = 16) 
: b + d = 18 :=
sorry

end b_plus_d_l136_136955


namespace best_in_district_round_l136_136387

-- Assume a structure that lets us refer to positions
inductive Position
| first
| second
| third
| last

open Position

-- Definitions of the statements
def Eva (p : Position → Prop) := ¬ (p first) ∧ ¬ (p last)
def Mojmir (p : Position → Prop) := ¬ (p last)
def Karel (p : Position → Prop) := p first
def Peter (p : Position → Prop) := p last

-- The main hypothesis
def exactly_one_lie (p : Position → Prop) :=
  (Eva p ∧ Mojmir p ∧ Karel p ∧ ¬ (Peter p)) ∨
  (Eva p ∧ Mojmir p ∧ ¬ (Karel p) ∧ Peter p) ∨
  (Eva p ∧ ¬ (Mojmir p) ∧ Karel p ∧ Peter p) ∨
  (¬ (Eva p) ∧ Mojmir p ∧ Karel p ∧ Peter p)

theorem best_in_district_round :
  ∃ (p : Position → Prop),
    (Eva p ∧ Mojmir p ∧ ¬ (Karel p) ∧ Peter p) ∧ exactly_one_lie p :=
by
  sorry

end best_in_district_round_l136_136387


namespace inequality_a_inequality_b_l136_136860

theorem inequality_a (R_A R_B R_C R_D d_A d_B d_C d_D : ℝ) :
  (R_A + R_B + R_C + R_D) * (1 / d_A + 1 / d_B + 1 / d_C + 1 / d_D) ≥ 48 :=
sorry

theorem inequality_b (R_A R_B R_C R_D d_A d_B d_C d_D : ℝ) :
  (R_A^2 + R_B^2 + R_C^2 + R_D^2) * (1 / d_A^2 + 1 / d_B^2 + 1 / d_C^2 + 1 / d_D^2) ≥ 144 :=
sorry

end inequality_a_inequality_b_l136_136860


namespace number_is_a_l136_136171

theorem number_is_a (x y z a : ℝ) (h1 : x + y + z = a) (h2 : (1 / x) + (1 / y) + (1 / z) = 1 / a) : 
  x = a ∨ y = a ∨ z = a :=
sorry

end number_is_a_l136_136171


namespace find_quotient_l136_136828

theorem find_quotient :
  ∃ q : ℕ, ∀ L S : ℕ, L = 1584 ∧ S = 249 ∧ (L - S = 1335) ∧ (L = S * q + 15) → q = 6 :=
by
  sorry

end find_quotient_l136_136828


namespace degree_sum_interior_angles_of_star_l136_136869

-- Definitions based on conditions provided.
def extended_polygon_star (n : Nat) (h : n ≥ 6) : Nat := 
  180 * (n - 2)

-- Theorem to prove the degree-sum of the interior angles.
theorem degree_sum_interior_angles_of_star (n : Nat) (h : n ≥ 6) : 
  extended_polygon_star n h = 180 * (n - 2) :=
by
  sorry

end degree_sum_interior_angles_of_star_l136_136869


namespace irreducible_polynomial_l136_136723

open Polynomial

theorem irreducible_polynomial (n : ℕ) : Irreducible ((X^2 + X)^(2^n) + 1 : ℤ[X]) := sorry

end irreducible_polynomial_l136_136723


namespace largest_z_l136_136588

theorem largest_z (x y z : ℝ) 
  (h1 : x + y + z = 5)  
  (h2 : x * y + y * z + x * z = 3) 
  : z ≤ 13 / 3 := sorry

end largest_z_l136_136588


namespace wage_difference_seven_l136_136410

-- Define the parameters and conditions
variables (P Q h : ℝ)

-- Given conditions
def condition1 : Prop := P = 1.5 * Q
def condition2 : Prop := P * h = 420
def condition3 : Prop := Q * (h + 10) = 420

-- Theorem to be proved
theorem wage_difference_seven (h : ℝ) (P Q : ℝ) 
  (h_condition1 : condition1 P Q)
  (h_condition2 : condition2 P h)
  (h_condition3 : condition3 Q h) :
  (P - Q) = 7 :=
  sorry

end wage_difference_seven_l136_136410


namespace radian_measure_of_minute_hand_rotation_l136_136964

theorem radian_measure_of_minute_hand_rotation :
  ∀ (t : ℝ), (t = 10) → (2 * π / 60 * t = -π/3) := by
  sorry

end radian_measure_of_minute_hand_rotation_l136_136964


namespace wholesale_cost_proof_l136_136290

-- Definitions based on conditions
def wholesale_cost (W : ℝ) := W
def retail_price (W : ℝ) := 1.20 * W
def employee_paid (R : ℝ) := 0.90 * R

-- Theorem statement: given the conditions, prove that the wholesale cost is $200.
theorem wholesale_cost_proof : 
  ∃ W : ℝ, (retail_price W = 1.20 * W) ∧ (employee_paid (retail_price W) = 216) ∧ W = 200 :=
by 
  let W := 200
  have hp : retail_price W = 1.20 * W := by sorry
  have ep : employee_paid (retail_price W) = 216 := by sorry
  exact ⟨W, hp, ep, rfl⟩

end wholesale_cost_proof_l136_136290


namespace coffee_serving_time_between_1_and_2_is_correct_l136_136980

theorem coffee_serving_time_between_1_and_2_is_correct
    (x : ℝ)
    (h_pos: 0 < x)
    (h_lt: x < 60) :
    30 + (x / 2) = 360 - (6 * x) → x = 660 / 13 :=
by
  sorry

end coffee_serving_time_between_1_and_2_is_correct_l136_136980


namespace fish_caught_by_dad_l136_136355

def total_fish_both : ℕ := 23
def fish_caught_morning : ℕ := 8
def fish_thrown_back : ℕ := 3
def fish_caught_afternoon : ℕ := 5
def fish_kept_brendan : ℕ := fish_caught_morning - fish_thrown_back + fish_caught_afternoon

theorem fish_caught_by_dad : total_fish_both - fish_kept_brendan = 13 := by
  sorry

end fish_caught_by_dad_l136_136355


namespace four_fours_expressions_l136_136511

theorem four_fours_expressions :
  (4 * 4 + 4) / 4 = 5 ∧
  4 + (4 + 4) / 2 = 6 ∧
  4 + 4 - 4 / 4 = 7 ∧
  4 + 4 + 4 - 4 = 8 ∧
  4 + 4 + 4 / 4 = 9 :=
by
  sorry

end four_fours_expressions_l136_136511


namespace eq_x_add_q_l136_136629

theorem eq_x_add_q (x q : ℝ) (h1 : abs (x - 5) = q) (h2 : x > 5) : x + q = 5 + 2*q :=
by {
  sorry
}

end eq_x_add_q_l136_136629


namespace greater_number_is_twenty_two_l136_136910

theorem greater_number_is_twenty_two (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * (x - y) = 12) : x = 22 :=
sorry

end greater_number_is_twenty_two_l136_136910


namespace cos_A_is_one_l136_136912

-- Definitions as per Lean's requirement
variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

-- Declaring the conditions are given
variables (α : ℝ) (cos_A : ℝ)
variables (AB CD AD BC : ℝ)
def is_convex_quadrilateral (A B C D : Type) : Prop := 
  sorry -- This would be a formal definition of convex quadrilateral

-- The conditions are specified in Lean terms
variables (h1 : is_convex_quadrilateral A B C D)
variables (h2 : α = 0) -- α = 0 implies cos(α) = 1
variables (h3 : AB = 240)
variables (h4 : CD = 240)
variables (h5 : AD ≠ BC)
variables (h6 : AB + CD + AD + BC = 960)

-- The proof statement to indicate that cos(α) = 1 under the given conditions
theorem cos_A_is_one : cos_A = 1 :=
by
  sorry -- Proof not included as per the instruction

end cos_A_is_one_l136_136912


namespace k_range_l136_136313

theorem k_range (k : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 0 ≤ 2 * x - 2 * k) → k ≤ 1 :=
by
  intro h
  have h1 := h 1 (by simp)
  have h3 := h 3 (by simp)
  sorry

end k_range_l136_136313


namespace triangle_angle_ratio_l136_136144

theorem triangle_angle_ratio (A B C D : Type*) 
  (α β γ δ : ℝ) -- α = ∠BAC, β = ∠ABC, γ = ∠BCA, δ = external angles
  (h1 : α + β + γ = 180)
  (h2 : δ = α + γ)
  (h3 : δ = β + γ) : (2 * 180 - (α + β)) / (α + β) = 2 :=
by
  sorry

end triangle_angle_ratio_l136_136144


namespace proof_problem_l136_136518

noncomputable def p : Prop := ∃ (α : ℝ), Real.cos (Real.pi - α) = Real.cos α
def q : Prop := ∀ (x : ℝ), x ^ 2 + 1 > 0

theorem proof_problem : p ∨ q := 
by
  sorry

end proof_problem_l136_136518


namespace heath_time_spent_l136_136909

variables (rows_per_carrot : ℕ) (plants_per_row : ℕ) (carrots_per_hour : ℕ) (total_hours : ℕ)

def total_carrots (rows_per_carrot plants_per_row : ℕ) : ℕ :=
  rows_per_carrot * plants_per_row

def time_spent (total_carrots carrots_per_hour : ℕ) : ℕ :=
  total_carrots / carrots_per_hour

theorem heath_time_spent
  (h1 : rows_per_carrot = 400)
  (h2 : plants_per_row = 300)
  (h3 : carrots_per_hour = 6000)
  (h4 : total_hours = 20) :
  time_spent (total_carrots rows_per_carrot plants_per_row) carrots_per_hour = total_hours :=
by
  sorry

end heath_time_spent_l136_136909


namespace range_of_m_l136_136204

theorem range_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : 0 < m) 
  (h4 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : m ≥ 4 :=
sorry

end range_of_m_l136_136204


namespace prove_angle_A_l136_136419

-- Definitions and conditions in triangle ABC
variables (A B C : ℝ) (a b c : ℝ) (h₁ : a^2 - b^2 = 3 * b * c) (h₂ : sin C = 2 * sin B)

-- Objective: Prove that angle A is 120 degrees
theorem prove_angle_A : A = 120 :=
sorry

end prove_angle_A_l136_136419


namespace smallest_n_value_l136_136989

theorem smallest_n_value :
  ∃ n, (∀ (sheets : Fin 2000 → Fin 4 → Fin 4),
        (∀ (n : Nat) (h : n ≤ 2000) (a b c d : Fin n) (h' : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d),
          ∃ (i j k : Fin 5), sheets a i = sheets b i ∧ sheets a j = sheets b j ∧ sheets a k = sheets b k → ¬ sheets a i = sheets c i ∧ ¬ sheets b j = sheets c j ∧ ¬ sheets a k = sheets c k)) ↔ n = 25 :=
sorry

end smallest_n_value_l136_136989


namespace range_of_a_l136_136777

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (1 ≤ x) ∧ (∀ a : ℝ, (1 + 1 / x) ^ (x + a) ≥ Real.exp 1 → a ≥ 1 / Real.log 2 - 1)

theorem range_of_a : problem_statement :=
sorry

end range_of_a_l136_136777


namespace maximize_profit_l136_136052

noncomputable section

-- Definitions of parameters
def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 200
def daily_cost : ℝ := 450
def price_min : ℝ := 30
def price_max : ℝ := 60

-- Function for daily profit
def daily_profit (x : ℝ) : ℝ := (x - 30) * daily_sales_volume x - daily_cost

-- Theorem statement
theorem maximize_profit :
  let max_profit_price := 60
  let max_profit_value := 1950
  30 ≤ max_profit_price ∧ max_profit_price ≤ 60 ∧
  daily_profit max_profit_price = max_profit_value :=
by
  sorry

end maximize_profit_l136_136052


namespace polynomial_problem_l136_136292

theorem polynomial_problem :
  ∀ P : Polynomial ℤ,
    (∃ R : Polynomial ℤ, (X^2 + 6*X + 10) * P^2 - 1 = R^2) → 
    P = 0 :=
by { sorry }

end polynomial_problem_l136_136292


namespace sum_of_remainders_is_six_l136_136223

def sum_of_remainders (n : ℕ) : ℕ :=
  n % 4 + (n + 1) % 4 + (n + 2) % 4 + (n + 3) % 4

theorem sum_of_remainders_is_six : ∀ n : ℕ, sum_of_remainders n = 6 :=
by
  intro n
  sorry

end sum_of_remainders_is_six_l136_136223


namespace half_abs_diff_squares_eq_40_l136_136547

theorem half_abs_diff_squares_eq_40 (x y : ℤ) (hx : x = 21) (hy : y = 19) :
  (|x^2 - y^2| / 2) = 40 :=
by
  sorry

end half_abs_diff_squares_eq_40_l136_136547


namespace tangent_y_intercept_range_l136_136526

theorem tangent_y_intercept_range :
  ∀ (x₀ : ℝ), (∃ y₀ : ℝ, y₀ = Real.exp x₀ ∧ (∃ m : ℝ, m = Real.exp x₀ ∧ ∃ b : ℝ, b = Real.exp x₀ * (1 - x₀) ∧ b < 0)) → x₀ > 1 := by
  sorry

end tangent_y_intercept_range_l136_136526


namespace find_divisor_l136_136833

theorem find_divisor (D N : ℕ) (k l : ℤ)
  (h1 : N % D = 255)
  (h2 : (2 * N) % D = 112) :
  D = 398 := by
  -- Proof here
  sorry

end find_divisor_l136_136833


namespace f_50_value_l136_136276

def f : ℝ → ℝ := sorry

axiom f_condition : ∀ x : ℝ, f (x^2 + x) + 2 * f (x^2 - 3*x + 2) = 9 * x^2 - 15 * x

theorem f_50_value : f 50 = 146 :=
by
  sorry

end f_50_value_l136_136276


namespace negation_of_universal_statement_l136_136021

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x > Real.sin x)) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end negation_of_universal_statement_l136_136021


namespace nitin_rank_last_l136_136134

theorem nitin_rank_last (total_students : ℕ) (rank_start : ℕ) (rank_last : ℕ) 
  (h1 : total_students = 58) 
  (h2 : rank_start = 24) 
  (h3 : rank_last = total_students - rank_start + 1) : 
  rank_last = 35 := 
by 
  -- proof can be filled in here
  sorry

end nitin_rank_last_l136_136134


namespace measure_of_angle_B_in_triangle_l136_136195

theorem measure_of_angle_B_in_triangle
  {a b c : ℝ} {A B C : ℝ} 
  (h1 : a * c = b^2 - a^2)
  (h2 : A = Real.pi / 6)
  (h3 : a / Real.sin A = b / Real.sin B) 
  (h4 : b / Real.sin B = c / Real.sin C)
  (h5 : A + B + C = Real.pi) :
  B = Real.pi / 3 :=
by sorry

end measure_of_angle_B_in_triangle_l136_136195


namespace uniqueFlavors_l136_136441

-- Definitions for the conditions
def numRedCandies : ℕ := 6
def numGreenCandies : ℕ := 4
def numBlueCandies : ℕ := 5

-- Condition stating each flavor must use at least two candies and no more than two colors
def validCombination (x y z : ℕ) : Prop :=
  (x = 0 ∨ y = 0 ∨ z = 0) ∧ (x + y ≥ 2 ∨ x + z ≥ 2 ∨ y + z ≥ 2)

-- The main theorem statement
theorem uniqueFlavors : 
  ∃ n : ℕ, n = 30 ∧ 
  (∀ x y z : ℕ, validCombination x y z → (x ≤ numRedCandies) ∧ (y ≤ numGreenCandies) ∧ (z ≤ numBlueCandies)) :=
sorry

end uniqueFlavors_l136_136441


namespace mother_daughter_age_relation_l136_136174

theorem mother_daughter_age_relation (x : ℕ) (hc1 : 43 - x = 5 * (11 - x)) : x = 3 := 
sorry

end mother_daughter_age_relation_l136_136174


namespace power_summation_l136_136263

theorem power_summation :
  (-1:ℤ)^(49) + (2:ℝ)^(3^3 + 5^2 - 48^2) = -1 + 1 / 2 ^ (2252 : ℝ) :=
by
  sorry

end power_summation_l136_136263


namespace taxi_ride_cost_l136_136489

def baseFare : ℝ := 1.50
def costPerMile : ℝ := 0.25
def milesTraveled : ℕ := 5
def totalCost := baseFare + (costPerMile * milesTraveled)

/-- The cost of a 5-mile taxi ride is $2.75. -/
theorem taxi_ride_cost : totalCost = 2.75 := by
  sorry

end taxi_ride_cost_l136_136489


namespace sum_of_angles_is_correct_l136_136401

noncomputable def hexagon_interior_angle : ℝ := 180 * (6 - 2) / 6
noncomputable def pentagon_interior_angle : ℝ := 180 * (5 - 2) / 5
noncomputable def sum_of_hexagon_and_pentagon_angles (A B C D : Type) 
  (hexagon_interior_angle : ℝ) 
  (pentagon_interior_angle : ℝ) : ℝ := 
  hexagon_interior_angle + pentagon_interior_angle

theorem sum_of_angles_is_correct (A B C D : Type) : 
  sum_of_hexagon_and_pentagon_angles A B C D hexagon_interior_angle pentagon_interior_angle = 228 := 
by
  simp [hexagon_interior_angle, pentagon_interior_angle]
  sorry

end sum_of_angles_is_correct_l136_136401


namespace square_of_105_l136_136940

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end square_of_105_l136_136940


namespace solve_equation_l136_136739

theorem solve_equation (x : ℚ) :
  (x^2 + 3 * x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 :=
by
  sorry

end solve_equation_l136_136739


namespace problem1_problem2_l136_136671

def box (n : ℕ) : ℕ := (10^n - 1) / 9

theorem problem1 (m : ℕ) :
  let b := box (3^m)
  b % (3^m) = 0 ∧ b % (3^(m+1)) ≠ 0 :=
  sorry

theorem problem2 (n : ℕ) :
  (n % 27 = 0) ↔ (box n % 27 = 0) :=
  sorry

end problem1_problem2_l136_136671


namespace fraction_subtraction_l136_136753

theorem fraction_subtraction (a b : ℕ) (h₁ : a = 18) (h₂ : b = 14) :
  (↑a / ↑b - ↑b / ↑a) = (32 / 63) := by
  sorry

end fraction_subtraction_l136_136753


namespace average_height_of_60_students_l136_136601

theorem average_height_of_60_students :
  (35 * 22 + 25 * 18) / 60 = 20.33 := 
sorry

end average_height_of_60_students_l136_136601


namespace abs_neg_three_halves_l136_136867

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := 
sorry

end abs_neg_three_halves_l136_136867


namespace value_of_a_l136_136690

/-- Given that 0.5% of a is 85 paise, prove that the value of a is 170 rupees. --/
theorem value_of_a (a : ℝ) (h : 0.005 * a = 85) : a = 170 := 
  sorry

end value_of_a_l136_136690


namespace logan_buys_15_pounds_of_corn_l136_136623

theorem logan_buys_15_pounds_of_corn (c b : ℝ) 
    (h1 : 1.20 * c + 0.60 * b = 27) 
    (h2 : b + c = 30) : 
    c = 15.0 :=
by
  sorry

end logan_buys_15_pounds_of_corn_l136_136623


namespace distinct_terms_in_expansion_l136_136458

theorem distinct_terms_in_expansion:
  (∀ (x y z u v w: ℝ), (x + y + z) * (u + v + w + x + y) = 0 → false) →
  3 * 5 = 15 := by sorry

end distinct_terms_in_expansion_l136_136458


namespace cauchy_schwarz_example_l136_136351

theorem cauchy_schwarz_example (a b c : ℝ) (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) : 
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
by
  sorry

end cauchy_schwarz_example_l136_136351


namespace eggs_sally_bought_is_correct_l136_136333

def dozen := 12

def eggs_sally_bought (dozens : Nat) : Nat :=
  dozens * dozen

theorem eggs_sally_bought_is_correct :
  eggs_sally_bought 4 = 48 :=
by
  sorry

end eggs_sally_bought_is_correct_l136_136333


namespace find_angle_A_range_area_of_triangle_l136_136660

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {S : ℝ}

theorem find_angle_A (h1 : b^2 + c^2 = a^2 - b * c) : A = (2 : ℝ) * Real.pi / 3 :=
by sorry

theorem range_area_of_triangle (h1 : b^2 + c^2 = a^2 - b * c)
(h2 : b * Real.sin A = 4 * Real.sin B) 
(h3 : Real.log b + Real.log c ≥ 1 - 2 * Real.cos (B + C)) 
(h4 : A = (2 : ℝ) * Real.pi / 3) :
(Real.sqrt 3 / 4 : ℝ) ≤ (1 / 2) * b * c * Real.sin A ∧
(1 / 2) * b * c * Real.sin A ≤ (4 * Real.sqrt 3 / 3 : ℝ) :=
by sorry

end find_angle_A_range_area_of_triangle_l136_136660


namespace min_value_expression_l136_136724

noncomputable section

variables {x y : ℝ}

theorem min_value_expression (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 1 ∧ 
    (∃ min_val : ℝ, min_val = (x^2 / (x + 2) + y^2 / (y + 1)) ∧ min_val = 1 / 4)) :=
  sorry

end min_value_expression_l136_136724


namespace greatest_nat_not_sum_of_two_composites_l136_136766

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

theorem greatest_nat_not_sum_of_two_composites :
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ 11 = a + b ∧
  (∀ n : ℕ, n > 11 → ¬ ∃ x y : ℕ, is_composite x ∧ is_composite y ∧ n = x + y) :=
sorry

end greatest_nat_not_sum_of_two_composites_l136_136766


namespace sum_is_272_l136_136205

-- Define the constant number x
def x : ℕ := 16

-- Define the sum of the number and its square
def sum_of_number_and_its_square (n : ℕ) : ℕ := n + n^2

-- State the theorem that the sum of the number and its square is 272 when the number is 16
theorem sum_is_272 : sum_of_number_and_its_square x = 272 :=
by
  sorry

end sum_is_272_l136_136205


namespace h_2023_eq_4052_l136_136721

theorem h_2023_eq_4052 (h : ℕ → ℕ) (h1 : h 1 = 2) (h2 : h 2 = 2) 
    (h3 : ∀ n ≥ 3, h n = h (n-1) - h (n-2) + 2 * n) : h 2023 = 4052 := 
by
  -- Use conditions as given
  sorry

end h_2023_eq_4052_l136_136721


namespace length_of_faster_train_l136_136740

theorem length_of_faster_train (speed_faster_train : ℝ) (speed_slower_train : ℝ) (elapsed_time : ℝ) (relative_speed : ℝ) (length_train : ℝ)
  (h1 : speed_faster_train = 50) 
  (h2 : speed_slower_train = 32) 
  (h3 : elapsed_time = 15) 
  (h4 : relative_speed = (speed_faster_train - speed_slower_train) * (1000 / 3600)) 
  (h5 : length_train = relative_speed * elapsed_time) :
  length_train = 75 :=
sorry

end length_of_faster_train_l136_136740


namespace larger_root_of_degree_11_l136_136901

theorem larger_root_of_degree_11 {x : ℝ} :
  (∃ x₁, x₁ > 0 ∧ (x₁ + x₁^2 + x₁^3 + x₁^4 + x₁^5 + x₁^6 + x₁^7 + x₁^8 = 8 - 10 * x₁^9)) ∧
  (∃ x₂, x₂ > 0 ∧ (x₂ + x₂^2 + x₂^3 + x₂^4 + x₂^5 + x₂^6 + x₂^7 + x₂^8 + x₂^9 + x₂^10 = 8 - 10 * x₂^11)) →
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧
    (x₁ + x₁^2 + x₁^3 + x₁^4 + x₁^5 + x₁^6 + x₁^7 + x₁^8 = 8 - 10 * x₁^9) ∧
    (x₂ + x₂^2 + x₂^3 + x₂^4 + x₂^5 + x₂^6 + x₂^7 + x₂^8 + x₂^9 + x₂^10 = 8 - 10 * x₂^11) ∧
    x₁ < x₂) :=
by
  sorry

end larger_root_of_degree_11_l136_136901


namespace find_third_number_l136_136945

theorem find_third_number (x y : ℕ) (h1 : x = 3)
  (h2 : (x + 1) / (x + 5) = (x + 5) / (x + y)) : y = 13 :=
by
  sorry

end find_third_number_l136_136945


namespace four_digit_palindrome_divisible_by_11_probability_zero_l136_136001

theorem four_digit_palindrome_divisible_by_11_probability_zero :
  (∃ a b : ℕ, 2 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (1001 * a + 110 * b) % 11 = 0) = false :=
by sorry

end four_digit_palindrome_divisible_by_11_probability_zero_l136_136001


namespace decimal_2_09_is_209_percent_l136_136932

-- Definition of the conversion from decimal to percentage
def decimal_to_percentage (x : ℝ) := x * 100

-- Theorem statement
theorem decimal_2_09_is_209_percent : decimal_to_percentage 2.09 = 209 :=
by sorry

end decimal_2_09_is_209_percent_l136_136932


namespace average_number_of_carnations_l136_136048

-- Define the conditions in Lean
def number_of_bouquet_1 : ℕ := 9
def number_of_bouquet_2 : ℕ := 14
def number_of_bouquet_3 : ℕ := 13
def total_bouquets : ℕ := 3

-- The main statement to be proved
theorem average_number_of_carnations : 
  (number_of_bouquet_1 + number_of_bouquet_2 + number_of_bouquet_3) / total_bouquets = 12 := 
by
  sorry

end average_number_of_carnations_l136_136048


namespace find_point_C_l136_136672

-- Definitions of the conditions
def line_eq (x y : ℝ) : Prop := x - 2 * y - 1 = 0
def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * x
def on_parabola (C : ℝ × ℝ) : Prop := parabola_eq C.1 C.2
def perpendicular_at_C (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Points A and B satisfy both the line and parabola equations
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_eq A.1 A.2 ∧ parabola_eq A.1 A.2 ∧
  line_eq B.1 B.2 ∧ parabola_eq B.1 B.2

-- Statement to be proven
theorem find_point_C (A B : ℝ × ℝ) (hA : intersection_points A B) :
  ∃ C : ℝ × ℝ, on_parabola C ∧ perpendicular_at_C A B C ∧
    (C = (1, -2) ∨ C = (9, -6)) :=
by
  sorry

end find_point_C_l136_136672


namespace decreasing_omega_range_l136_136064

open Real

theorem decreasing_omega_range {ω : ℝ} (h1 : 1 < ω) :
  (∀ x y : ℝ, π ≤ x ∧ x ≤ y ∧ y ≤ (5 * π) / 4 → 
    (|sin (ω * y + π / 3)| ≤ |sin (ω * x + π / 3)|)) → 
  (7 / 6 ≤ ω ∧ ω ≤ 4 / 3) :=
by
  sorry

end decreasing_omega_range_l136_136064


namespace sum_of_repeating_decimals_l136_136549

noncomputable def repeating_decimal_sum : ℚ :=
  let x := 1 / 3
  let y := 7 / 99
  let z := 8 / 999
  x + y + z

theorem sum_of_repeating_decimals :
  repeating_decimal_sum = 418 / 999 :=
by
  sorry

end sum_of_repeating_decimals_l136_136549


namespace minerals_found_today_l136_136552

noncomputable def yesterday_gemstones := 21
noncomputable def today_minerals := 48
noncomputable def today_gemstones := 21

theorem minerals_found_today :
  (today_minerals - (2 * yesterday_gemstones) = 6) :=
by
  sorry

end minerals_found_today_l136_136552


namespace percentage_increase_of_soda_l136_136470

variable (C S x : ℝ)

theorem percentage_increase_of_soda
  (h1 : 1.25 * C = 10)
  (h2 : S + x * S = 12)
  (h3 : C + S = 16) :
  x = 0.5 :=
sorry

end percentage_increase_of_soda_l136_136470


namespace g_at_4_l136_136673

noncomputable def f (x : ℝ) : ℝ := 5 / (3 - x)
noncomputable def f_inv (x : ℝ) : ℝ := 3 - 5 / x
noncomputable def g (x : ℝ) : ℝ := 2 / (f_inv x) + 7

theorem g_at_4 : g 4 = 8.142857 := by
  sorry

end g_at_4_l136_136673


namespace calculate_f_at_2_l136_136054

noncomputable def f (x : ℝ) : ℝ := sorry

theorem calculate_f_at_2 :
  (∀ x : ℝ, 25 * f (x / 1580) + (3 - Real.sqrt 34) * f (1580 / x) = 2017 * x) →
  f 2 = 265572 :=
by
  intro h
  sorry

end calculate_f_at_2_l136_136054


namespace digit_for_multiple_of_9_l136_136872

theorem digit_for_multiple_of_9 (d : ℕ) : (23450 + d) % 9 = 0 ↔ d = 4 := by
  sorry

end digit_for_multiple_of_9_l136_136872


namespace negation_equivalence_l136_136564

-- Definition of the original proposition
def proposition (x : ℝ) : Prop := x > 1 → Real.log x > 0

-- Definition of the negated proposition
def negation (x : ℝ) : Prop := ¬ (x > 1 → Real.log x > 0)

-- The mathematically equivalent proof problem as Lean statement
theorem negation_equivalence (x : ℝ) : 
  (¬ (x > 1 → Real.log x > 0)) ↔ (x ≤ 1 → Real.log x ≤ 0) := 
by 
  sorry

end negation_equivalence_l136_136564


namespace abs_sum_factors_l136_136580

theorem abs_sum_factors (a b c d : ℤ) : 
  (6 * x ^ 2 + x - 12 = (a * x + b) * (c * x + d)) →
  (|a| + |b| + |c| + |d| = 12) :=
by
  intros h
  sorry

end abs_sum_factors_l136_136580


namespace min_pictures_needed_l136_136795

theorem min_pictures_needed (n m : ℕ) (participants : Fin n → Fin m → Prop)
  (h1 : n = 60) (h2 : m ≤ 30)
  (h3 : ∀ (i j : Fin n), ∃ (k : Fin m), participants i k ∧ participants j k) :
  m = 6 :=
sorry

end min_pictures_needed_l136_136795


namespace mathieu_plot_area_l136_136456

def total_area (x y : ℕ) : ℕ := x * x

theorem mathieu_plot_area :
  ∃ (x y : ℕ), (x^2 - y^2 = 464) ∧ (x - y = 8) ∧ (total_area x y = 1089) :=
by sorry

end mathieu_plot_area_l136_136456


namespace average_speed_of_rocket_l136_136291

def distance_soared (speed_soaring : ℕ) (time_soaring : ℕ) : ℕ :=
  speed_soaring * time_soaring

def distance_plummeted : ℕ := 600

def total_distance (distance_soared : ℕ) (distance_plummeted : ℕ) : ℕ :=
  distance_soared + distance_plummeted

def total_time (time_soaring : ℕ) (time_plummeting : ℕ) : ℕ :=
  time_soaring + time_plummeting

def average_speed (total_distance : ℕ) (total_time : ℕ) : ℕ :=
  total_distance / total_time

theorem average_speed_of_rocket :
  let speed_soaring := 150
  let time_soaring := 12
  let time_plummeting := 3
  distance_soared speed_soaring time_soaring +
  distance_plummeted = 2400
  →
  total_time time_soaring time_plummeting = 15
  →
  average_speed (distance_soared speed_soaring time_soaring + distance_plummeted)
                (total_time time_soaring time_plummeting) = 160 :=
by
  sorry

end average_speed_of_rocket_l136_136291


namespace find_number_l136_136558

theorem find_number (a : ℕ) (h : a = 105) : 
  a^3 / (49 * 45 * 25) = 21 :=
by
  sorry

end find_number_l136_136558


namespace circle_radius_l136_136680

theorem circle_radius (P Q : ℝ) (h1 : P = π * r^2) (h2 : Q = 2 * π * r) (h3 : P / Q = 15) : r = 30 :=
by
  sorry

end circle_radius_l136_136680


namespace percentage_loss_l136_136384

theorem percentage_loss (SP_loss SP_profit CP : ℝ) 
  (h₁ : SP_loss = 9) 
  (h₂ : SP_profit = 11.8125) 
  (h₃ : SP_profit = CP * 1.05) : 
  (CP - SP_loss) / CP * 100 = 20 :=
by sorry

end percentage_loss_l136_136384


namespace teal_sold_pumpkin_pies_l136_136834

def pies_sold 
  (pumpkin_pie_slices : ℕ) (pumpkin_pie_price : ℕ) 
  (custard_pie_slices : ℕ) (custard_pie_price : ℕ) 
  (custard_pies_sold : ℕ) (total_revenue : ℕ) : ℕ :=
  total_revenue / (pumpkin_pie_slices * pumpkin_pie_price)

theorem teal_sold_pumpkin_pies : 
  pies_sold 8 5 6 6 5 340 = 4 := 
by 
  sorry

end teal_sold_pumpkin_pies_l136_136834


namespace smallest_palindrome_base2_base4_l136_136183

def is_palindrome_base (n : ℕ) (b : ℕ) : Prop :=
  let digits := (Nat.digits b n)
  digits = digits.reverse

theorem smallest_palindrome_base2_base4 : 
  ∃ (x : ℕ), x > 15 ∧ is_palindrome_base x 2 ∧ is_palindrome_base x 4 ∧ x = 17 :=
by
  sorry

end smallest_palindrome_base2_base4_l136_136183


namespace jimmys_speed_l136_136876

theorem jimmys_speed 
(Mary_speed : ℕ) (total_distance : ℕ) (t : ℕ)
(h1 : Mary_speed = 5)
(h2 : total_distance = 9)
(h3 : t = 1)
: ∃ (Jimmy_speed : ℕ), Jimmy_speed = 4 :=
by
  -- calculation steps skipped here
  sorry

end jimmys_speed_l136_136876


namespace probability_of_head_equal_half_l136_136231

def fair_coin_probability : Prop :=
  ∀ (H T : ℕ), (H = 1 ∧ T = 1 ∧ (H + T = 2)) → ((H / (H + T)) = 1 / 2)

theorem probability_of_head_equal_half : fair_coin_probability :=
sorry

end probability_of_head_equal_half_l136_136231


namespace remainder_when_divided_by_11_l136_136544

theorem remainder_when_divided_by_11 (n : ℕ) 
  (h1 : 10 ≤ n ∧ n < 100) 
  (h2 : n % 9 = 1) 
  (h3 : n % 10 = 3) : 
  n % 11 = 7 := 
sorry

end remainder_when_divided_by_11_l136_136544


namespace intersection_A_B_solution_inequalities_l136_136175

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}
def C : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = C :=
by
  sorry

theorem solution_inequalities (x : ℝ) :
  (2 * x^2 + x - 1 > 0) ↔ (x < -1 ∨ x > 1/2) :=
by
  sorry

end intersection_A_B_solution_inequalities_l136_136175


namespace inequality_abc_l136_136283

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := 
by 
  sorry

end inequality_abc_l136_136283


namespace sum_of_present_ages_l136_136450

def Jed_age_future (current_Jed: ℕ) (years: ℕ) : ℕ := 
  current_Jed + years

def Matt_age (current_Jed: ℕ) : ℕ := 
  current_Jed - 10

def sum_ages (jed_age: ℕ) (matt_age: ℕ) : ℕ := 
  jed_age + matt_age

theorem sum_of_present_ages :
  ∃ jed_curr_age matt_curr_age : ℕ, 
  (Jed_age_future jed_curr_age 10 = 25) ∧ 
  (jed_curr_age = matt_curr_age + 10) ∧ 
  (sum_ages jed_curr_age matt_curr_age = 20) :=
sorry

end sum_of_present_ages_l136_136450


namespace bus_A_speed_l136_136379

variable (v_A v_B : ℝ)
variable (h1 : v_A - v_B = 15)
variable (h2 : v_A + v_B = 75)

theorem bus_A_speed : v_A = 45 := sorry

end bus_A_speed_l136_136379


namespace k_range_l136_136972

def y_increasing (k : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ + 1 < k * x₂ + 1
def y_max_min (k : ℝ) : Prop := (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ (x^2 - 2 * x + 3 = 2)) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ (x^2 - 2 * x + 3 = 3))

theorem k_range (k : ℝ) (hk : (¬ (0 < k ∧ y_max_min k) ∧ (0 < k ∨ y_max_min k))) : 
  (0 < k ∧ k < 1) ∨ (k > 2) :=
sorry

end k_range_l136_136972


namespace battery_current_l136_136383

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l136_136383


namespace simple_interest_rate_l136_136790

/-- 
  Given conditions:
  1. Time period T is 10 years.
  2. Simple interest SI is 7/5 of the principal amount P.
  Prove that the rate percent per annum R for which the simple interest is 7/5 of the principal amount in 10 years is 14%.
-/
theorem simple_interest_rate (P : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ) (hT : T = 10) (hSI : SI = (7 / 5) * P) : 
  (SI = (P * R * T) / 100) → R = 14 := 
by 
  sorry

end simple_interest_rate_l136_136790


namespace find_hourly_charge_computer_B_l136_136107

noncomputable def hourly_charge_computer_B (B : ℝ) :=
  ∃ (A h : ℝ),
    A = 1.4 * B ∧
    B * (h + 20) = 550 ∧
    A * h = 550 ∧
    B = 7.86

theorem find_hourly_charge_computer_B : ∃ B : ℝ, hourly_charge_computer_B B :=
  sorry

end find_hourly_charge_computer_B_l136_136107


namespace plan_A_is_cost_effective_l136_136439

-- Definitions of the costs considering the problem's conditions
def cost_plan_A (days_A : ℕ) (rate_A : ℕ) : ℕ := days_A * rate_A
def cost_plan_C (days_AB : ℕ) (rate_A : ℕ) (rate_B : ℕ) (remaining_B : ℕ) : ℕ :=
  (days_AB * (rate_A + rate_B)) + (remaining_B * rate_B)

-- Specification of the days and rates from the conditions
def days_A := 12
def rate_A := 10000
def rate_B := 6000
def days_AB := 3
def remaining_B := 13

-- Costs for each plan
def A_cost := cost_plan_A days_A rate_A
def C_cost := cost_plan_C days_AB rate_A rate_B remaining_B

-- Theorem stating that Plan A is more cost-effective
theorem plan_A_is_cost_effective : A_cost < C_cost := by
  unfold A_cost
  unfold C_cost
  sorry

end plan_A_is_cost_effective_l136_136439


namespace solve_quadratic_for_q_l136_136707

-- Define the quadratic equation and the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main theorem statement
theorem solve_quadratic_for_q : ∃ q : ℝ, q ≠ 0 ∧ (discriminant q (-8) 2 = 0) → q = 8 :=
by
  -- Insert the assumptions and proof context here.
  -- However, since we were instructed not to consider the solution steps
  -- the proof is skipped with a "sorry".
  sorry

end solve_quadratic_for_q_l136_136707


namespace union_sets_l136_136113

def set_A : Set ℝ := {x | x^3 - 3 * x^2 - x + 3 < 0}
def set_B : Set ℝ := {x | |x + 1 / 2| ≥ 1}

theorem union_sets :
  set_A ∪ set_B = ( {x : ℝ | x < -1} ∪ {x : ℝ | x ≥ 1 / 2} ) :=
by
  sorry

end union_sets_l136_136113


namespace number_of_people_who_purchased_only_book_A_l136_136514

theorem number_of_people_who_purchased_only_book_A (x y v : ℕ) 
  (h1 : 2 * x = 500)
  (h2 : y = x + 500)
  (h3 : v = 2 * y) : 
  v = 1500 := 
sorry

end number_of_people_who_purchased_only_book_A_l136_136514


namespace number_of_boats_l136_136173

theorem number_of_boats (total_people : ℕ) (people_per_boat : ℕ)
  (h1 : total_people = 15) (h2 : people_per_boat = 3) : total_people / people_per_boat = 5 :=
by {
  -- proof steps here
  sorry
}

end number_of_boats_l136_136173


namespace range_of_k_l136_136042

theorem range_of_k (a b c d k : ℝ) (hA : b = k * a - 2 * a - 1) (hB : d = k * c - 2 * c - 1) (h_diff : a ≠ c) (h_lt : (c - a) * (d - b) < 0) : k < 2 := 
sorry

end range_of_k_l136_136042


namespace shaded_area_ratio_l136_136364

theorem shaded_area_ratio
  (large_square_area : ℕ := 25)
  (grid_dimension : ℕ := 5)
  (shaded_square_area : ℕ := 2)
  (num_squares : ℕ := 25)
  (ratio : ℚ := 2 / 25) :
  (shaded_square_area : ℚ) / large_square_area = ratio := 
by
  sorry

end shaded_area_ratio_l136_136364


namespace drinking_ratio_l136_136561

variable (t_mala t_usha : ℝ) (d_usha : ℝ)

theorem drinking_ratio :
  (t_mala = t_usha) → 
  (d_usha = 2 / 10) →
  (1 - d_usha = 8 / 10) →
  (4 * d_usha = 8) :=
by
  intros h1 h2 h3
  sorry

end drinking_ratio_l136_136561


namespace johnny_money_left_l136_136288

def total_saved (september october november : ℕ) : ℕ := september + october + november

def money_left (total amount_spent : ℕ) : ℕ := total - amount_spent

theorem johnny_money_left 
    (saved_september : ℕ)
    (saved_october : ℕ)
    (saved_november : ℕ)
    (spent_video_game : ℕ)
    (h1 : saved_september = 30)
    (h2 : saved_october = 49)
    (h3 : saved_november = 46)
    (h4 : spent_video_game = 58) :
    money_left (total_saved saved_september saved_october saved_november) spent_video_game = 67 := 
by sorry

end johnny_money_left_l136_136288


namespace maximum_mark_for_paper_i_l136_136448

noncomputable def maximum_mark (pass_percentage: ℝ) (secured_marks: ℝ) (failed_by: ℝ) : ℝ :=
  (secured_marks + failed_by) / pass_percentage

theorem maximum_mark_for_paper_i :
  maximum_mark 0.35 42 23 = 186 :=
by
  sorry

end maximum_mark_for_paper_i_l136_136448


namespace no_unique_day_in_august_l136_136350

def july_has_five_tuesdays (N : ℕ) : Prop :=
  ∃ (d : ℕ), ∀ k : ℕ, k < 5 → (d + k * 7) ≤ 30

def july_august_have_30_days (N : ℕ) : Prop :=
  true -- We're asserting this unconditionally since both months have exactly 30 days in the problem

theorem no_unique_day_in_august (N : ℕ) (h1 : july_has_five_tuesdays N) (h2 : july_august_have_30_days N) :
  ¬(∃ d : ℕ, ∀ k : ℕ, k < 5 → (d + k * 7) ≤ 30 ∧ ∃! wday : ℕ, (d + k * 7 + wday) % 7 = 0) :=
sorry

end no_unique_day_in_august_l136_136350


namespace find_y_intercept_l136_136490

theorem find_y_intercept (m : ℝ) 
  (h1 : ∀ x y : ℝ, y = 2 * x + m)
  (h2 : ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ y = 2 * x + m) : 
  m = -1 := 
sorry

end find_y_intercept_l136_136490


namespace songs_per_album_correct_l136_136811

-- Define the number of albums and total number of songs as conditions
def number_of_albums : ℕ := 8
def total_songs : ℕ := 16

-- Define the number of songs per album
def songs_per_album (albums : ℕ) (songs : ℕ) : ℕ := songs / albums

-- The main theorem stating that the number of songs per album is 2
theorem songs_per_album_correct :
  songs_per_album number_of_albums total_songs = 2 :=
by
  unfold songs_per_album
  sorry

end songs_per_album_correct_l136_136811


namespace fly_in_box_maximum_path_length_l136_136074

theorem fly_in_box_maximum_path_length :
  let side1 := 1
  let side2 := Real.sqrt 2
  let side3 := Real.sqrt 3
  let space_diagonal := Real.sqrt (side1^2 + side2^2 + side3^2)
  let face_diagonal1 := Real.sqrt (side1^2 + side2^2)
  let face_diagonal2 := Real.sqrt (side1^2 + side3^2)
  let face_diagonal3 := Real.sqrt (side2^2 + side3^2)
  (4 * space_diagonal + 2 * face_diagonal3) = 4 * Real.sqrt 6 + 2 * Real.sqrt 5 :=
by
  sorry

end fly_in_box_maximum_path_length_l136_136074


namespace complex_number_sum_l136_136515

variable (ω : ℂ)
variable (h1 : ω^9 = 1)
variable (h2 : ω ≠ 1)

theorem complex_number_sum :
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80 = ω^2 :=
by sorry

end complex_number_sum_l136_136515


namespace right_angled_triangle_hypotenuse_and_altitude_relation_l136_136692

variables (a b c m : ℝ)

theorem right_angled_triangle_hypotenuse_and_altitude_relation
  (h1 : b^2 + c^2 = a^2)
  (h2 : m^2 = (b - c)^2)
  (h3 : b * c = a * m) :
  m = (a * (Real.sqrt 5 - 1)) / 2 := 
sorry

end right_angled_triangle_hypotenuse_and_altitude_relation_l136_136692


namespace A_inter_B_eq_l136_136123

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 > 1}

theorem A_inter_B_eq : A ∩ B = {-2, 2} := 
by
  sorry

end A_inter_B_eq_l136_136123


namespace extreme_points_inequality_l136_136706

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^2 + a * Real.log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a - 4 * x + 2

theorem extreme_points_inequality (a : ℝ) (h_a : 0 < a ∧ a < 1) (x0 : ℝ)
  (h_ext : 4 * x0^2 - 4 * x0 + a = 0) (h_min : ∃ x1, x0 + x1 = 1 ∧ x0 < x1 ∧ x1 < 1) :
  g x0 a > 1 / 2 - Real.log 2 :=
sorry

end extreme_points_inequality_l136_136706


namespace least_positive_integer_l136_136499

theorem least_positive_integer (x : ℕ) :
  (∃ k : ℤ, (3 * x + 41) ^ 2 = 53 * k) ↔ x = 4 :=
by
  sorry

end least_positive_integer_l136_136499


namespace value_of_a_add_b_l136_136445

theorem value_of_a_add_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
sorry

end value_of_a_add_b_l136_136445


namespace arithmetic_sequence_length_l136_136141

theorem arithmetic_sequence_length :
  ∀ (a d a_n : ℕ), a = 6 → d = 4 → a_n = 154 → ∃ n: ℕ, a_n = a + (n-1) * d ∧ n = 38 :=
by
  intro a d a_n ha hd ha_n
  use 38
  rw [ha, hd, ha_n]
  -- Leaving the proof as an exercise
  sorry

end arithmetic_sequence_length_l136_136141


namespace find_natural_number_l136_136583

theorem find_natural_number (n : ℕ) (k : ℤ) (h : 2^n + 3 = k^2) : n = 0 :=
sorry

end find_natural_number_l136_136583


namespace find_original_number_l136_136465

theorem find_original_number (x : ℤ) : 4 * (3 * x + 29) = 212 → x = 8 :=
by
  intro h
  sorry

end find_original_number_l136_136465


namespace cos_thirteen_pi_over_three_l136_136060

theorem cos_thirteen_pi_over_three : Real.cos (13 * Real.pi / 3) = 1 / 2 := 
by
  sorry

end cos_thirteen_pi_over_three_l136_136060


namespace yolanda_walking_rate_l136_136071

-- Definitions for the conditions given in the problem
variables (X Y : ℝ) -- Points X and Y
def distance_X_to_Y := 52 -- Distance between X and Y in miles
def Bob_rate := 4 -- Bob's walking rate in miles per hour
def Bob_distance_walked := 28 -- The distance Bob walked in miles
def start_time_diff := 1 -- The time difference (in hours) between Yolanda and Bob starting

-- The statement to prove
theorem yolanda_walking_rate : 
  ∃ (y : ℝ), (distance_X_to_Y = y * (Bob_distance_walked / Bob_rate + start_time_diff) + Bob_distance_walked) ∧ y = 3 := by 
  sorry

end yolanda_walking_rate_l136_136071


namespace self_employed_tax_amount_l136_136332

-- Definitions for conditions
def gross_income : ℝ := 350000.0

def tax_rate_self_employed : ℝ := 0.06

-- Statement asserting the tax amount for self-employed individuals given the conditions
theorem self_employed_tax_amount :
  gross_income * tax_rate_self_employed = 21000.0 := by
  sorry

end self_employed_tax_amount_l136_136332


namespace intersection_point_l136_136222

theorem intersection_point (x y : ℚ) (h1 : 8 * x - 5 * y = 40) (h2 : 6 * x + 2 * y = 14) :
  x = 75 / 23 ∧ y = -64 / 23 :=
by
  -- Proof not needed, so we finish with sorry
  sorry

end intersection_point_l136_136222


namespace initial_investment_C_l136_136153

def total_investment : ℝ := 425
def increase_A (a : ℝ) : ℝ := 0.05 * a
def increase_B (b : ℝ) : ℝ := 0.08 * b
def increase_C (c : ℝ) : ℝ := 0.10 * c

theorem initial_investment_C (a b c : ℝ) (h1 : a + b + c = total_investment)
  (h2 : increase_A a = increase_B b) (h3 : increase_B b = increase_C c) : c = 100 := by
  sorry

end initial_investment_C_l136_136153


namespace problem1_l136_136718

theorem problem1 (m n : ℕ) (h1 : 3 ^ m = 4) (h2 : 3 ^ (m + 4 * n) = 324) : 2016 ^ n = 2016 := 
by 
  sorry

end problem1_l136_136718


namespace total_cost_of_repair_l136_136590

theorem total_cost_of_repair (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) (H1 : hours = 2) (H2 : hourly_rate = 75) (H3 : part_cost = 150) :
  hours * hourly_rate + part_cost = 300 := 
by
  sorry

end total_cost_of_repair_l136_136590


namespace poles_on_each_side_l136_136997

theorem poles_on_each_side (total_poles : ℕ) (sides_equal : ℕ)
  (h1 : total_poles = 104) (h2 : sides_equal = 4) : 
  (total_poles / sides_equal) = 26 :=
by
  sorry

end poles_on_each_side_l136_136997


namespace distinct_quadrilateral_areas_l136_136396

theorem distinct_quadrilateral_areas (A B C D E F : ℝ) 
  (h : A + B + C + D + E + F = 156) :
  ∃ (Q1 Q2 Q3 : ℝ), Q1 = 78 ∧ Q2 = 104 ∧ Q3 = 104 :=
sorry

end distinct_quadrilateral_areas_l136_136396


namespace parabolas_pass_through_origin_l136_136099

-- Definition of a family of parabolas
def parabola_family (p q : ℝ) (x : ℝ) : ℝ := -x^2 + p * x + q

-- Definition of vertices lying on y = x^2
def vertex_condition (p q : ℝ) : Prop :=
  ∃ a : ℝ, (a^2 = -a^2 + p * a + q)

-- Proving that all such parabolas pass through the point (0, 0)
theorem parabolas_pass_through_origin :
  ∀ (p q : ℝ), vertex_condition p q → parabola_family p q 0 = 0 :=
by
  sorry

end parabolas_pass_through_origin_l136_136099


namespace not_exists_k_eq_one_l136_136217

theorem not_exists_k_eq_one (k : ℝ) : (∃ x y : ℝ, y = k * x + 2 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 :=
by sorry

end not_exists_k_eq_one_l136_136217


namespace solution_set_of_f_gt_7_minimum_value_of_m_n_l136_136877

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem solution_set_of_f_gt_7 :
  { x : ℝ | f x > 7 } = { x | x > 4 ∨ x < -3 } :=
by
  ext x
  sorry

theorem minimum_value_of_m_n (m n : ℝ) (h : 0 < m ∧ 0 < n) (hfmin : ∀ x : ℝ, f x ≥ m + n) :
  m = n ∧ m = 3 / 2 ∧ m^2 + n^2 = 9 / 2 :=
by
  sorry

end solution_set_of_f_gt_7_minimum_value_of_m_n_l136_136877


namespace train_crosses_second_platform_l136_136933

theorem train_crosses_second_platform (
  length_train length_platform1 length_platform2 : ℝ) 
  (time_platform1 : ℝ) 
  (H1 : length_train = 100)
  (H2 : length_platform1 = 200)
  (H3 : length_platform2 = 300)
  (H4 : time_platform1 = 15) :
  ∃ t : ℝ, t = 20 := by
  sorry

end train_crosses_second_platform_l136_136933


namespace calculate_expression_l136_136520

theorem calculate_expression :
  (1.99^2 - 1.98 * 1.99 + 0.99^2 = 1) :=
by
  sorry

end calculate_expression_l136_136520


namespace sub_frac_pow_eq_l136_136569

theorem sub_frac_pow_eq :
  7 - (2 / 5)^3 = 867 / 125 := by
  sorry

end sub_frac_pow_eq_l136_136569


namespace average_weight_of_whole_class_l136_136121

/-- Section A has 30 students -/
def num_students_A : ℕ := 30

/-- Section B has 20 students -/
def num_students_B : ℕ := 20

/-- The average weight of Section A is 40 kg -/
def avg_weight_A : ℕ := 40

/-- The average weight of Section B is 35 kg -/
def avg_weight_B : ℕ := 35

/-- The average weight of the whole class is 38 kg -/
def avg_weight_whole_class : ℕ := 38

-- Proof that the average weight of the whole class is equal to 38 kg

theorem average_weight_of_whole_class : 
  ((num_students_A * avg_weight_A) + (num_students_B * avg_weight_B)) / (num_students_A + num_students_B) = avg_weight_whole_class :=
by
  -- Sorry indicates that the proof is omitted.
  sorry

end average_weight_of_whole_class_l136_136121


namespace find_angle_A_l136_136055

noncomputable def exists_angle_A (A B C : ℝ) (a b : ℝ) : Prop :=
  C = (A + B) / 2 ∧ 
  A + B + C = 180 ∧ 
  (a + b) / 2 = Real.sqrt 3 + 1 ∧ 
  C = 2 * Real.sqrt 2

theorem find_angle_A : ∃ A B C a b, 
  exists_angle_A A B C a b ∧ (A = 75 ∨ A = 45) :=
by
  -- This is where the detailed proof would go
  sorry

end find_angle_A_l136_136055


namespace distinct_triangle_areas_l136_136416

variables (A B C D E F G : ℝ) (h : ℝ)
variables (AB BC CD EF FG AC BD AD EG : ℝ)

def is_valid_points := AB = 2 ∧ BC = 1 ∧ CD = 3 ∧ EF = 1 ∧ FG = 2 ∧ AC = AB + BC ∧ BD = BC + CD ∧ AD = AB + BC + CD ∧ EG = EF + FG

theorem distinct_triangle_areas (h_pos : 0 < h) (valid : is_valid_points AB BC CD EF FG AC BD AD EG) : 
  ∃ n : ℕ, n = 5 := 
by
  sorry

end distinct_triangle_areas_l136_136416


namespace quadratic_has_two_distinct_real_roots_l136_136498

theorem quadratic_has_two_distinct_real_roots :
  let a := (1 : ℝ)
  let b := (-5 : ℝ)
  let c := (-1 : ℝ)
  b^2 - 4 * a * c > 0 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l136_136498


namespace cargo_arrival_in_days_l136_136797

-- Definitions for conditions
def days_navigate : ℕ := 21
def days_customs : ℕ := 4
def days_transport : ℕ := 7
def days_departed : ℕ := 30

-- Calculate the days since arrival in Vancouver
def days_arrival_vancouver : ℕ := days_departed - days_navigate

-- Calculate the days since customs processes finished
def days_since_customs_done : ℕ := days_arrival_vancouver - days_customs

-- Calculate the days for cargo to arrive at the warehouse from today
def days_until_arrival : ℕ := days_transport - days_since_customs_done

-- Expected number of days from today for the cargo to arrive at the warehouse
theorem cargo_arrival_in_days : days_until_arrival = 2 := by
  -- Insert the proof steps here
  sorry

end cargo_arrival_in_days_l136_136797


namespace billy_buys_bottle_l136_136466

-- Definitions of costs and volumes
def money : ℝ := 10
def cost1 : ℝ := 1
def volume1 : ℝ := 10
def cost2 : ℝ := 2
def volume2 : ℝ := 16
def cost3 : ℝ := 2.5
def volume3 : ℝ := 25
def cost4 : ℝ := 5
def volume4 : ℝ := 50
def cost5 : ℝ := 10
def volume5 : ℝ := 200

-- Statement of the proof problem
theorem billy_buys_bottle : ∃ b : ℕ, b = 1 ∧ cost5 = money := by 
  sorry

end billy_buys_bottle_l136_136466


namespace tomatoes_harvest_ratio_l136_136714

noncomputable def tomatoes_ratio (w t f : ℕ) (g r : ℕ) : ℕ × ℕ :=
  if (w = 400) ∧ ((w + t + f) = 2000) ∧ ((g = 700) ∧ (r = 700) ∧ ((g + r) = f)) ∧ (t = 200) then 
    (2, 1)
  else 
    sorry

theorem tomatoes_harvest_ratio : 
  ∀ (w t f : ℕ) (g r : ℕ), 
  (w = 400) → 
  (w + t + f = 2000) → 
  (g = 700) → 
  (r = 700) → 
  (g + r = f) → 
  (t = 200) →
  tomatoes_ratio w t f g r = (2, 1) :=
by {
  -- insert proof here
  sorry
}

end tomatoes_harvest_ratio_l136_136714


namespace race_length_l136_136198

variables (L : ℕ)

def distanceCondition1 := L - 70
def distanceCondition2 := L - 100
def distanceCondition3 := L - 163

theorem race_length (h1 : distanceCondition1 = L - 70) 
                    (h2 : distanceCondition2 = L - 100) 
                    (h3 : distanceCondition3 = L - 163)
                    (h4 : (L - 70) / (L - 163) = (L) / (L - 100)) : 
  L = 1000 :=
sorry

end race_length_l136_136198


namespace whale_population_ratio_l136_136695

theorem whale_population_ratio 
  (W_last : ℕ)
  (W_this : ℕ)
  (W_next : ℕ)
  (h1 : W_last = 4000)
  (h2 : W_next = W_this + 800)
  (h3 : W_next = 8800) :
  (W_this / W_last) = 2 := by
  sorry

end whale_population_ratio_l136_136695


namespace shape_of_constant_phi_l136_136898

-- Define the spherical coordinates structure
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the condition that φ is a constant c
def constant_phi (c : ℝ) (coords : SphericalCoordinates) : Prop :=
  coords.φ = c

-- Define the type for shapes
inductive Shape
  | Line : Shape
  | Circle : Shape
  | Plane : Shape
  | Sphere : Shape
  | Cylinder : Shape
  | Cone : Shape

-- The theorem statement
theorem shape_of_constant_phi (c : ℝ) (coords : SphericalCoordinates) 
  (h : constant_phi c coords) : Shape :=
  Shape.Cone

end shape_of_constant_phi_l136_136898


namespace simplify_expression_l136_136112

theorem simplify_expression (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a ≠ b) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b :=
by sorry

end simplify_expression_l136_136112


namespace percent_calculation_l136_136406

theorem percent_calculation (x : ℝ) (h : 0.40 * x = 160) : 0.30 * x = 120 :=
by
  sorry

end percent_calculation_l136_136406


namespace cricket_problem_l136_136155

theorem cricket_problem
  (x : ℕ)
  (run_rate_initial : ℝ := 3.8)
  (overs_remaining : ℕ := 40)
  (run_rate_remaining : ℝ := 6.1)
  (target_runs : ℕ := 282) :
  run_rate_initial * x + run_rate_remaining * overs_remaining = target_runs → x = 10 :=
by
  -- proof goes here
  sorry

end cricket_problem_l136_136155


namespace graveling_cost_l136_136089

def lawn_length : ℝ := 110
def lawn_breadth: ℝ := 60
def road_width : ℝ := 10
def cost_per_sq_meter : ℝ := 3

def road_1_area : ℝ := lawn_length * road_width
def intersecting_length : ℝ := lawn_breadth - road_width
def road_2_area : ℝ := intersecting_length * road_width
def total_area : ℝ := road_1_area + road_2_area
def total_cost : ℝ := total_area * cost_per_sq_meter

theorem graveling_cost :
  total_cost = 4800 := 
  by
    sorry

end graveling_cost_l136_136089


namespace length_of_train_is_correct_l136_136507

noncomputable def length_of_train (speed : ℕ) (time : ℕ) : ℕ :=
  (speed * (time / 3600) * 1000)

theorem length_of_train_is_correct : length_of_train 70 36 = 700 := by
  sorry

end length_of_train_is_correct_l136_136507


namespace value_of_m_l136_136020

-- Problem Statement
theorem value_of_m (m : ℝ) : (∃ x : ℝ, (m-2)*x^(|m|-1) + 16 = 0 ∧ |m| - 1 = 1) → m = -2 :=
by
  sorry

end value_of_m_l136_136020


namespace horner_example_l136_136203

def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldr (λ a acc => a + x * acc) 0

theorem horner_example : horner [12, 35, -8, 79, 6, 5, 3] (-4) = 220 := by
  sorry

end horner_example_l136_136203


namespace total_payment_leila_should_pay_l136_136304

-- Definitions of the conditions
def chocolateCakes := 3
def chocolatePrice := 12
def strawberryCakes := 6
def strawberryPrice := 22

-- Mathematical equivalent proof problem
theorem total_payment_leila_should_pay : 
  chocolateCakes * chocolatePrice + strawberryCakes * strawberryPrice = 168 := 
by 
  sorry

end total_payment_leila_should_pay_l136_136304


namespace part1_part2_l136_136327

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 6

theorem part1 (a b : ℝ) (h1 : f a b 1 = 8) : a + b = 2 := by
  rw [f] at h1
  sorry

theorem part2 (a b : ℝ) (h1 : f a b (-1) = f a b 3) : f a b 2 = 6 := by
  rw [f] at h1
  sorry

end part1_part2_l136_136327


namespace jose_birds_left_l136_136662

-- Define initial conditions
def chickens_initial : Nat := 28
def ducks : Nat := 18
def turkeys : Nat := 15
def chickens_sold : Nat := 12

-- Calculate remaining chickens
def chickens_left : Nat := chickens_initial - chickens_sold

-- Calculate total birds left
def total_birds_left : Nat := chickens_left + ducks + turkeys

-- Theorem statement to prove the number of birds left
theorem jose_birds_left : total_birds_left = 49 :=
by
  -- This is where the proof would typically go
  sorry

end jose_birds_left_l136_136662


namespace sum_of_areas_of_circles_l136_136207

-- Definitions of the conditions given in the problem
def triangle_side1 : ℝ := 6
def triangle_side2 : ℝ := 8
def triangle_side3 : ℝ := 10

-- Definitions of the radii r, s, t
variables (r s t : ℝ)

-- Conditions derived from the problem
axiom rs_eq : r + s = triangle_side1
axiom rt_eq : r + t = triangle_side2
axiom st_eq : s + t = triangle_side3

-- Main theorem to prove
theorem sum_of_areas_of_circles : (π * r^2) + (π * s^2) + (π * t^2) = 56 * π :=
by
  sorry

end sum_of_areas_of_circles_l136_136207


namespace conditional_prob_l136_136992

noncomputable def prob_A := 0.7
noncomputable def prob_AB := 0.4

theorem conditional_prob : prob_AB / prob_A = 4 / 7 :=
by
  sorry

end conditional_prob_l136_136992


namespace domain_of_f_l136_136011

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 12))

theorem domain_of_f :
  ∀ x : ℝ, f x ≠ 0 ↔ (x ≠ 15 / 2) :=
by
  sorry

end domain_of_f_l136_136011


namespace ratio_A_B_l136_136505

-- Given conditions as definitions
def P_both : ℕ := 500  -- Number of people who purchased both books A and B

def P_only_B : ℕ := P_both / 2  -- Number of people who purchased only book B

def P_only_A : ℕ := 1000  -- Number of people who purchased only book A

-- Total number of people who purchased books
def P_A : ℕ := P_only_A + P_both  -- Total number of people who purchased book A

def P_B : ℕ := P_only_B + P_both  -- Total number of people who purchased book B

-- The ratio of people who purchased book A to book B
theorem ratio_A_B : P_A / P_B = 2 :=
by
  sorry

end ratio_A_B_l136_136505


namespace function_eq_l136_136050

noncomputable def f (x : ℝ) : ℝ := x^4 - 2

theorem function_eq (f : ℝ → ℝ) (h1 : ∀ x : ℝ, deriv f x = 4 * x^3) (h2 : f 1 = -1) : 
  ∀ x : ℝ, f x = x^4 - 2 :=
by
  intro x
  -- Proof omitted
  sorry

end function_eq_l136_136050


namespace cos_inequality_m_range_l136_136696

theorem cos_inequality_m_range (m : ℝ) : 
  (-1 ≤ 1 - m ∧ 1 - m ≤ 1) ↔ (0 ≤ m ∧ m ≤ 2) :=
sorry

end cos_inequality_m_range_l136_136696


namespace problem_statement_l136_136200

-- Define the sides of the original triangle
def side_5 := 5
def side_12 := 12
def side_13 := 13

-- Define the perimeters of the isosceles triangles
def P := 3 * side_5
def Q := 3 * side_12
def R := 3 * side_13

-- Statement we want to prove
theorem problem_statement : P + R = (3 / 2) * Q := by
  sorry

end problem_statement_l136_136200


namespace binary_multiplication_l136_136140

theorem binary_multiplication :
  let a := 0b1101101
  let b := 0b1011
  let product := 0b10001001111
  a * b = product :=
sorry

end binary_multiplication_l136_136140


namespace total_wheels_in_parking_lot_l136_136891

-- Definitions (conditions)
def cars := 14
def wheels_per_car := 4
def missing_wheels_per_missing_car := 1
def missing_cars := 2

def bikes := 5
def wheels_per_bike := 2

def unicycles := 3
def wheels_per_unicycle := 1

def twelve_wheeler_trucks := 2
def wheels_per_twelve_wheeler_truck := 12
def damaged_wheels_per_twelve_wheeler_truck := 3
def damaged_twelve_wheeler_trucks := 1

def eighteen_wheeler_trucks := 1
def wheels_per_eighteen_wheeler_truck := 18

-- The total wheels calculation proof
theorem total_wheels_in_parking_lot :
  ((cars * wheels_per_car - missing_cars * missing_wheels_per_missing_car) +
   (bikes * wheels_per_bike) +
   (unicycles * wheels_per_unicycle) +
   (twelve_wheeler_trucks * wheels_per_twelve_wheeler_truck - damaged_twelve_wheeler_trucks * damaged_wheels_per_twelve_wheeler_truck) +
   (eighteen_wheeler_trucks * wheels_per_eighteen_wheeler_truck)) = 106 := by
  sorry

end total_wheels_in_parking_lot_l136_136891


namespace h_inch_approx_l136_136952

noncomputable def h_cm : ℝ := 14.5 - 2 * 1.7
noncomputable def cm_to_inch (cm : ℝ) : ℝ := cm / 2.54
noncomputable def h_inch : ℝ := cm_to_inch h_cm

theorem h_inch_approx : abs (h_inch - 4.37) < 1e-2 :=
by
  -- The proof is omitted
  sorry

end h_inch_approx_l136_136952


namespace centers_of_parallelograms_l136_136942

def is_skew_lines (l1 l2 l3 l4 : Line) : Prop :=
  -- A function that checks if 4 lines are pairwise skew and no three of them are parallel to the same plane.
  sorry

def count_centers_of_parallelograms (l1 l2 l3 l4 : Line) : ℕ :=
  -- A function that counts the number of lines through which the centers of parallelograms formed by the intersections of the lines pass.
  sorry

theorem centers_of_parallelograms (l1 l2 l3 l4 : Line) (h_skew: is_skew_lines l1 l2 l3 l4) : count_centers_of_parallelograms l1 l2 l3 l4 = 3 :=
  sorry

end centers_of_parallelograms_l136_136942


namespace weight_of_10_moles_approx_l136_136533

def atomic_mass_C : ℝ := 12.01
def atomic_mass_H : ℝ := 1.008
def atomic_mass_O : ℝ := 16.00

def molar_mass_C6H8O6 : ℝ := 
  (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)

def moles : ℝ := 10
def given_total_weight : ℝ := 1760

theorem weight_of_10_moles_approx (ε : ℝ) (hε : ε > 0) :
  abs ((moles * molar_mass_C6H8O6) - given_total_weight) < ε := by
  -- proof will go here.
  sorry

end weight_of_10_moles_approx_l136_136533


namespace florist_initial_roses_l136_136320

theorem florist_initial_roses : 
  ∀ (R : ℕ), (R - 16 + 19 = 40) → (R = 37) :=
by
  intro R
  intro h
  sorry

end florist_initial_roses_l136_136320


namespace sum_of_reciprocals_sum_of_square_reciprocals_sum_of_cubic_reciprocals_l136_136995

variable (p q : ℝ) (x1 x2 : ℝ)

-- Define the condition: Roots of the quadratic equation
def quadratic_equation_condition : Prop :=
  x1^2 + p * x1 + q = 0 ∧ x2^2 + p * x2 + q = 0

-- Define the identities for calculations based on properties of roots
def properties_of_roots : Prop :=
  x1 + x2 = -p ∧ x1 * x2 = q

-- First proof problem
theorem sum_of_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                           (h2 : properties_of_roots p q x1 x2) :
  1 / x1 + 1 / x2 = -p / q := 
by sorry

-- Second proof problem
theorem sum_of_square_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                                  (h2 : properties_of_roots p q x1 x2) :
  1 / (x1^2) + 1 / (x2^2) = (p^2 - 2*q) / (q^2) := 
by sorry

-- Third proof problem
theorem sum_of_cubic_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                                 (h2 : properties_of_roots p q x1 x2) :
  1 / (x1^3) + 1 / (x2^3) = p * (3*q - p^2) / (q^3) := 
by sorry

end sum_of_reciprocals_sum_of_square_reciprocals_sum_of_cubic_reciprocals_l136_136995


namespace find_radius_of_sphere_l136_136606

def radius_of_sphere_equal_to_cylinder_area (r : ℝ) (h : ℝ) (d : ℝ) : Prop :=
  (4 * Real.pi * r^2 = 2 * Real.pi * ((d / 2) * h))

theorem find_radius_of_sphere : ∃ r : ℝ, radius_of_sphere_equal_to_cylinder_area r 6 6 ∧ r = 3 :=
by
  sorry

end find_radius_of_sphere_l136_136606


namespace sam_correct_percent_l136_136682

variable (y : ℝ)
variable (h_pos : 0 < y)

theorem sam_correct_percent :
  ((8 * y - 3 * y) / (8 * y) * 100) = 62.5 := by
sorry

end sam_correct_percent_l136_136682


namespace largest_of_nine_consecutive_integers_l136_136694

theorem largest_of_nine_consecutive_integers (sum_eq_99: ∃ (n : ℕ), 99 = (n - 4) + (n - 3) + (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) : 
  ∃ n : ℕ, n = 15 :=
by
  sorry

end largest_of_nine_consecutive_integers_l136_136694


namespace min_value_5_l136_136414

theorem min_value_5 (x y : ℝ) : ∃ x y : ℝ, (xy - 2)^2 + (x + y + 1)^2 = 5 :=
sorry

end min_value_5_l136_136414


namespace problem1_l136_136858

noncomputable def sqrt7_minus_1_pow_0 : ℝ := (Real.sqrt 7 - 1)^0
noncomputable def minus_half_pow_neg_2 : ℝ := (-1 / 2)^(-2 : ℤ)
noncomputable def sqrt3_tan_30 : ℝ := Real.sqrt 3 * Real.tan (Real.pi / 6)

theorem problem1 : sqrt7_minus_1_pow_0 - minus_half_pow_neg_2 + sqrt3_tan_30 = -2 := by
  sorry

end problem1_l136_136858


namespace exists_city_reaching_all_l136_136469

variables {City : Type} (canReach : City → City → Prop)

-- Conditions from the problem
axiom reach_itself (A : City) : canReach A A
axiom reach_transitive {A B C : City} : canReach A B → canReach B C → canReach A C
axiom reach_any_two {P Q : City} : ∃ R : City, canReach R P ∧ canReach R Q

-- The proof problem
theorem exists_city_reaching_all (cities : City → Prop) :
  (∀ P Q, P ≠ Q → cities P → cities Q → ∃ R, cities R ∧ canReach R P ∧ canReach R Q) →
  ∃ C, ∀ A, cities A → canReach C A :=
by
  intros H
  sorry

end exists_city_reaching_all_l136_136469


namespace no_such_n_exists_l136_136611

-- Definition of the sum of the digits function s(n)
def s (n : ℕ) : ℕ := n.digits 10 |> List.sum

-- Statement of the proof problem
theorem no_such_n_exists : ¬ ∃ n : ℕ, n * s n = 20222022 :=
by
  -- argument based on divisibility rules as presented in the problem
  sorry

end no_such_n_exists_l136_136611


namespace cost_of_figurine_l136_136275

noncomputable def cost_per_tv : ℝ := 50
noncomputable def num_tvs : ℕ := 5
noncomputable def num_figurines : ℕ := 10
noncomputable def total_spent : ℝ := 260

theorem cost_of_figurine : 
  ((total_spent - (num_tvs * cost_per_tv)) / num_figurines) = 1 := 
by
  sorry

end cost_of_figurine_l136_136275


namespace symmetric_line_eq_l136_136919

theorem symmetric_line_eq (x y : ℝ) :
  (∀ x y, 2 * x - y + 1 = 0 → y = -x) → (∀ x y, x - 2 * y + 1 = 0) :=
by sorry

end symmetric_line_eq_l136_136919


namespace retailer_profit_percentage_l136_136434

theorem retailer_profit_percentage 
  (CP MP SP : ℝ)
  (hCP : CP = 100)
  (hMP : MP = CP + 0.65 * CP)
  (hSP : SP = MP - 0.25 * MP)
  : ((SP - CP) / CP) * 100 = 23.75 := 
sorry

end retailer_profit_percentage_l136_136434


namespace average_playtime_l136_136615

-- Definitions based on conditions
def h_w := 2 -- Hours played on Wednesday
def h_t := 2 -- Hours played on Thursday
def h_f := h_w + 3 -- Hours played on Friday (3 hours more than Wednesday)

-- Statement to prove
theorem average_playtime :
  (h_w + h_t + h_f) / 3 = 3 := by
  sorry

end average_playtime_l136_136615


namespace maximum_marks_l136_136664

-- Definitions based on the conditions
def passing_percentage : ℝ := 0.5
def student_marks : ℝ := 200
def marks_to_pass : ℝ := student_marks + 20

-- Lean 4 statement for the proof problem
theorem maximum_marks (M : ℝ) 
  (h1 : marks_to_pass = 220)
  (h2 : passing_percentage * M = marks_to_pass) :
  M = 440 :=
sorry

end maximum_marks_l136_136664


namespace three_pipes_time_l136_136277

variable (R : ℝ) (T : ℝ)

-- Condition: Two pipes fill the tank in 18 hours
def two_pipes_fill : Prop := 2 * R * 18 = 1

-- Question: How long does it take for three pipes to fill the tank?
def three_pipes_fill : Prop := 3 * R * T = 1

theorem three_pipes_time (h : two_pipes_fill R) : three_pipes_fill R 12 :=
by
  sorry

end three_pipes_time_l136_136277


namespace geometric_sequence_common_ratio_l136_136110

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q) 
  (h_inc : ∀ n, a (n + 1) > a n) (h2 : a 2 = 2) (h3 : a 4 - a 3 = 4) : q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l136_136110


namespace car_city_mileage_l136_136731

theorem car_city_mileage (h c t : ℝ) 
  (h_eq : h * t = 462)
  (c_eq : (h - 15) * t = 336) 
  (c_def : c = h - 15) : 
  c = 40 := 
by 
  sorry

end car_city_mileage_l136_136731


namespace fraction_result_l136_136752

theorem fraction_result (x : ℚ) (h₁ : x * (3/4) = (1/6)) : (x - (1/12)) = (5/36) := 
sorry

end fraction_result_l136_136752


namespace candidate_D_votes_l136_136226

theorem candidate_D_votes :
  let total_votes := 10000
  let invalid_votes_percentage := 0.25
  let valid_votes := (1 - invalid_votes_percentage) * total_votes
  let candidate_A_percentage := 0.40
  let candidate_B_percentage := 0.30
  let candidate_C_percentage := 0.20
  let candidate_D_percentage := 1.0 - (candidate_A_percentage + candidate_B_percentage + candidate_C_percentage)
  let candidate_D_votes := candidate_D_percentage * valid_votes
  candidate_D_votes = 750 :=
by
  sorry

end candidate_D_votes_l136_136226


namespace find_math_marks_l136_136734

theorem find_math_marks (subjects : ℕ)
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℝ)
  (math_marks : ℕ) :
  subjects = 5 →
  english_marks = 96 →
  physics_marks = 99 →
  chemistry_marks = 100 →
  biology_marks = 98 →
  average_marks = 98.2 →
  math_marks = 98 :=
by
  intros h_subjects h_english h_physics h_chemistry h_biology h_average
  sorry

end find_math_marks_l136_136734


namespace no_positive_integer_solution_l136_136117

theorem no_positive_integer_solution (m n : ℕ) (h : 0 < m) (h1 : 0 < n) : ¬ (5 * m^2 - 6 * m * n + 7 * n^2 = 2006) :=
sorry

end no_positive_integer_solution_l136_136117


namespace cost_of_drapes_l136_136685

theorem cost_of_drapes (D: ℝ) (h1 : 3 * 40 = 120) (h2 : D * 3 + 120 = 300) : D = 60 :=
  sorry

end cost_of_drapes_l136_136685


namespace find_larger_number_l136_136674

theorem find_larger_number (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end find_larger_number_l136_136674


namespace traveler_never_returns_home_l136_136228

variable (City : Type)
variable (Distance : City → City → ℝ)

variables (A B C : City)
variables (C_i C_i_plus_one C_i_minus_one : City)

-- Given conditions
axiom travel_far_from_A : ∀ (C : City), C ≠ B → Distance A B > Distance A C
axiom travel_far_from_B : ∀ (D : City), D ≠ C → Distance B C > Distance B D
axiom increasing_distance : ∀ i : ℕ, Distance C_i C_i_plus_one > Distance C_i_minus_one C_i

-- Given condition that C is not A
axiom C_not_eq_A : C ≠ A

-- Proof statement
theorem traveler_never_returns_home : ∀ i : ℕ, C_i ≠ A := sorry

end traveler_never_returns_home_l136_136228


namespace meet_floor_l136_136252

noncomputable def xiaoming_meets_xiaoying (x y meet_floor: ℕ) : Prop :=
  x = 4 → y = 3 → (meet_floor = 22)

theorem meet_floor (x y meet_floor: ℕ) (h1: x = 4) (h2: y = 3) :
  xiaoming_meets_xiaoying x y meet_floor :=
by
  sorry

end meet_floor_l136_136252


namespace exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l136_136128

-- Lean 4 statement for part (a)
theorem exists_palindromic_number_divisible_by_5 : 
  ∃ (n : ℕ), (n = 51715) ∧ (n % 5 = 0) := sorry

-- Lean 4 statement for part (b)
theorem count_palindromic_numbers_divisible_by_5 : 
  (∃ (count : ℕ), count = 100) := sorry

end exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l136_136128


namespace joan_lost_balloons_l136_136581

theorem joan_lost_balloons :
  let initial_balloons := 9
  let current_balloons := 7
  let balloons_lost := initial_balloons - current_balloons
  balloons_lost = 2 :=
by
  sorry

end joan_lost_balloons_l136_136581


namespace passengers_with_round_trip_tickets_l136_136702

theorem passengers_with_round_trip_tickets (P R : ℝ) : 
  (0.40 * R = 0.25 * P) → (R / P = 0.625) :=
by
  intro h
  sorry

end passengers_with_round_trip_tickets_l136_136702


namespace greatest_int_with_gcd_of_24_eq_2_l136_136935

theorem greatest_int_with_gcd_of_24_eq_2 (n : ℕ) (h1 : n < 200) (h2 : Int.gcd n 24 = 2) : n = 194 := 
sorry

end greatest_int_with_gcd_of_24_eq_2_l136_136935


namespace g_of_f_of_3_is_217_l136_136403

-- Definitions of f and g
def f (x : ℝ) : ℝ := x^2 - 4
def g (x : ℝ) : ℝ := x^3 + 3 * x^2 + 3 * x + 2

-- The theorem we need to prove
theorem g_of_f_of_3_is_217 : g (f 3) = 217 := by
  sorry

end g_of_f_of_3_is_217_l136_136403


namespace calc_expression_solve_equation_l136_136075

-- Problem 1: Calculation

theorem calc_expression : 
  |Real.sqrt 3 - 2| + Real.sqrt 12 - 6 * Real.sin (Real.pi / 6) + (-1/2 : Real)⁻¹ = Real.sqrt 3 - 3 := 
by {
  sorry
}

-- Problem 2: Solve the Equation

theorem solve_equation (x : Real) : 
  x * (x + 6) = -5 ↔ (x = -5 ∨ x = -1) := 
by {
  sorry
}

end calc_expression_solve_equation_l136_136075


namespace bruce_and_anne_clean_house_l136_136930

theorem bruce_and_anne_clean_house 
  (A : ℚ) (B : ℚ) (H1 : A = 1/12) 
  (H2 : 3 * (B + 2 * A) = 1) :
  1 / (B + A) = 4 := 
sorry

end bruce_and_anne_clean_house_l136_136930


namespace polynomial_sat_condition_l136_136104

theorem polynomial_sat_condition (P : Polynomial ℝ) (k : ℕ) (hk : 0 < k) :
  (P.comp P = P ^ k) →
  (P = 0 ∨ P = 1 ∨ (k % 2 = 1 ∧ P = -1) ∨ P = Polynomial.X ^ k) :=
sorry

end polynomial_sat_condition_l136_136104


namespace initial_peanuts_l136_136503

-- Definitions based on conditions
def peanuts_added := 8
def total_peanuts_now := 12

-- Statement to prove
theorem initial_peanuts (initial_peanuts : ℕ) (h : initial_peanuts + peanuts_added = total_peanuts_now) : initial_peanuts = 4 :=
sorry

end initial_peanuts_l136_136503


namespace time_after_4350_minutes_is_march_6_00_30_l136_136861

-- Define the start time as a date
def startDate := (2015, 3, 3, 0, 0) -- March 3, 2015 at midnight (00:00)

-- Define the total minutes to add
def totalMinutes := 4350

-- Function to convert minutes to a date and time given a start date
def addMinutes (date : (Nat × Nat × Nat × Nat × Nat)) (minutes : Nat) : (Nat × Nat × Nat × Nat × Nat) :=
  let hours := minutes / 60
  let remainMinutes := minutes % 60
  let days := hours / 24
  let remainHours := hours % 24
  let (year, month, day, hour, min) := date
  (year, month, day + days, remainHours, remainMinutes)

-- Expected result date and time
def expectedDate := (2015, 3, 6, 0, 30) -- March 6, 2015 at 00:30 AM

theorem time_after_4350_minutes_is_march_6_00_30 :
  addMinutes startDate totalMinutes = expectedDate :=
by
  sorry

end time_after_4350_minutes_is_march_6_00_30_l136_136861


namespace greatest_divisor_of_three_consecutive_odds_l136_136412

theorem greatest_divisor_of_three_consecutive_odds (n : ℕ) : 
  ∃ (d : ℕ), (∀ (k : ℕ), k = 2*n + 1 ∨ k = 2*n + 3 ∨ k = 2*n + 5 → d ∣ (2*n + 1) * (2*n + 3) * (2*n + 5)) ∧ d = 3 :=
by
  sorry

end greatest_divisor_of_three_consecutive_odds_l136_136412


namespace daily_salary_of_manager_l136_136294

theorem daily_salary_of_manager
  (M : ℕ)
  (salary_clerk : ℕ)
  (num_managers : ℕ)
  (num_clerks : ℕ)
  (total_salary : ℕ)
  (h1 : salary_clerk = 2)
  (h2 : num_managers = 2)
  (h3 : num_clerks = 3)
  (h4 : total_salary = 16)
  (h5 : 2 * M + 3 * salary_clerk = total_salary) :
  M = 5 := 
  sorry

end daily_salary_of_manager_l136_136294


namespace shepherd_flock_l136_136168

theorem shepherd_flock (x y : ℕ) (h1 : (x - 1) * 5 = 7 * y) (h2 : x * 3 = 5 * (y - 1)) :
  x + y = 25 :=
sorry

end shepherd_flock_l136_136168


namespace tangent_of_11pi_over_4_l136_136166

theorem tangent_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 :=
sorry

end tangent_of_11pi_over_4_l136_136166


namespace train_passes_man_in_approx_21_seconds_l136_136669

noncomputable def train_length : ℝ := 385
noncomputable def train_speed_kmph : ℝ := 60
noncomputable def man_speed_kmph : ℝ := 6

-- Convert speeds to m/s
noncomputable def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def man_speed_mps : ℝ := kmph_to_mps man_speed_kmph

-- Calculate relative speed
noncomputable def relative_speed_mps : ℝ := train_speed_mps + man_speed_mps

-- Calculate time
noncomputable def time_to_pass : ℝ := train_length / relative_speed_mps

theorem train_passes_man_in_approx_21_seconds : abs (time_to_pass - 21) < 1 :=
by
  sorry

end train_passes_man_in_approx_21_seconds_l136_136669


namespace length_of_first_train_is_correct_l136_136380

noncomputable def length_of_first_train (speed1_km_hr speed2_km_hr : ℝ) (time_cross_sec : ℝ) (length2_m : ℝ) : ℝ :=
  let speed1_m_s := speed1_km_hr * (5 / 18)
  let speed2_m_s := speed2_km_hr * (5 / 18)
  let relative_speed_m_s := speed1_m_s + speed2_m_s
  let total_distance_m := relative_speed_m_s * time_cross_sec
  total_distance_m - length2_m

theorem length_of_first_train_is_correct : 
  length_of_first_train 60 40 11.879049676025918 160 = 170 := by
  sorry

end length_of_first_train_is_correct_l136_136380


namespace simplify_expression_l136_136334

theorem simplify_expression (b : ℝ) (h : b ≠ -1) : 
  1 - (1 / (1 - (b / (1 + b)))) = -b :=
by {
  sorry
}

end simplify_expression_l136_136334


namespace value_of_neg_a_squared_sub_3a_l136_136959

variable (a : ℝ)
variable (h : a^2 + 3 * a - 5 = 0)

theorem value_of_neg_a_squared_sub_3a : -a^2 - 3*a = -5 :=
by
  sorry

end value_of_neg_a_squared_sub_3a_l136_136959


namespace card_selection_l136_136627

noncomputable def count_ways := 438400

theorem card_selection :
  let decks := 2
  let total_cards := 52 * decks
  let suits := 4
  let non_royal_count := 10 * decks
  let royal_count := 3 * decks
  let non_royal_options := non_royal_count * decks
  let royal_options := royal_count * decks
  1 * (non_royal_options)^4 + (suits.choose 1) * royal_options * (non_royal_options)^3 + (suits.choose 2) * (royal_options)^2 * (non_royal_options)^2 = count_ways :=
sorry

end card_selection_l136_136627


namespace optimal_pricing_l136_136573

-- Define the conditions given in the problem
def cost_price : ℕ := 40
def selling_price : ℕ := 60
def weekly_sales : ℕ := 300

def sales_volume (price : ℕ) : ℕ := weekly_sales - 10 * (price - selling_price)
def profit (price : ℕ) : ℕ := (price - cost_price) * sales_volume price

-- Statement to prove
theorem optimal_pricing : ∃ (price : ℕ), price = 65 ∧ profit price = 6250 :=
by {
  sorry
}

end optimal_pricing_l136_136573


namespace total_nails_l136_136431

-- Definitions based on the conditions
def Violet_nails : ℕ := 27
def Tickletoe_nails : ℕ := (27 - 3) / 2

-- Theorem to prove the total number of nails
theorem total_nails : Violet_nails + Tickletoe_nails = 39 := by
  sorry

end total_nails_l136_136431


namespace max_value_of_k_proof_l136_136990

noncomputable def maximum_value_of_k (x y k : ℝ) (h1: x > 0) (h2: y > 0) (h3: k > 0) 
(h4: 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) : Prop :=
  k = (-1 + Real.sqrt 17) / 2

-- This is the statement that needs to be proven:
theorem max_value_of_k_proof (x y k : ℝ) (h1: x > 0) (h2: y > 0) (h3: k > 0) 
(h4: 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) : maximum_value_of_k x y k h1 h2 h3 h4 :=
sorry

end max_value_of_k_proof_l136_136990


namespace henry_walks_distance_l136_136537

noncomputable def gym_distance : ℝ := 3

noncomputable def walk_factor : ℝ := 2 / 3

noncomputable def c_limit_position : ℝ := 1.5

noncomputable def d_limit_position : ℝ := 2.5

theorem henry_walks_distance :
  abs (c_limit_position - d_limit_position) = 1 := by
  sorry

end henry_walks_distance_l136_136537


namespace find_other_endpoint_l136_136453

theorem find_other_endpoint 
    (Mx My : ℝ) (x1 y1 : ℝ) 
    (hx_Mx : Mx = 3) (hy_My : My = 1)
    (hx1 : x1 = 7) (hy1 : y1 = -3) : 
    ∃ (x2 y2 : ℝ), Mx = (x1 + x2) / 2 ∧ My = (y1 + y2) / 2 ∧ x2 = -1 ∧ y2 = 5 :=
by
    sorry

end find_other_endpoint_l136_136453


namespace lena_glued_friends_pictures_l136_136622

-- Define the conditions
def clippings_per_friend : ℕ := 3
def glue_per_clipping : ℕ := 6
def total_glue : ℕ := 126

-- Define the proof problem statement
theorem lena_glued_friends_pictures : 
    ∃ (F : ℕ), F * (clippings_per_friend * glue_per_clipping) = total_glue ∧ F = 7 := 
by
  sorry

end lena_glued_friends_pictures_l136_136622


namespace negation_proposition_l136_136621

theorem negation_proposition (a b : ℝ) :
  (a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0) :=
by
  sorry

end negation_proposition_l136_136621


namespace no_solution_m_l136_136188

noncomputable def fractional_eq (x m : ℝ) : Prop :=
  2 / (x - 2) + m * x / (x^2 - 4) = 3 / (x + 2)

theorem no_solution_m (m : ℝ) : 
  (¬ ∃ x, fractional_eq x m) ↔ (m = -4 ∨ m = 6 ∨ m = 1) :=
sorry

end no_solution_m_l136_136188


namespace part1_part2_l136_136510

noncomputable def total_seating_arrangements : ℕ := 840
noncomputable def non_adjacent_4_people_arrangements : ℕ := 24
noncomputable def three_empty_adjacent_arrangements : ℕ := 120

theorem part1 : total_seating_arrangements - non_adjacent_4_people_arrangements = 816 := by
  sorry

theorem part2 : total_seating_arrangements - three_empty_adjacent_arrangements = 720 := by
  sorry

end part1_part2_l136_136510


namespace probability_of_non_defective_is_seven_ninetyninths_l136_136492

-- Define the number of total pencils, defective pencils, and the number of pencils selected
def total_pencils : ℕ := 12
def defective_pencils : ℕ := 4
def selected_pencils : ℕ := 5

-- Define the number of ways to choose k elements from n elements (the combination function)
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the total number of ways to choose 5 pencils out of 12
def total_ways : ℕ := combination total_pencils selected_pencils

-- Calculate the number of non-defective pencils
def non_defective_pencils : ℕ := total_pencils - defective_pencils

-- Calculate the number of ways to choose 5 non-defective pencils out of 8
def non_defective_ways : ℕ := combination non_defective_pencils selected_pencils

-- Calculate the probability that all 5 chosen pencils are non-defective
def probability_non_defective : ℚ :=
  non_defective_ways / total_ways

-- Prove that this probability equals 7/99
theorem probability_of_non_defective_is_seven_ninetyninths :
  probability_non_defective = 7 / 99 :=
by
  -- The proof is left as an exercise
  sorry

end probability_of_non_defective_is_seven_ninetyninths_l136_136492


namespace polynomial_expansion_l136_136442

theorem polynomial_expansion :
  let x := 1 
  let y := -1 
  let a_0 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_1 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  let a_2 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_3 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  let a_4 := (3 - 2 * x)^5 + (3 - 2 * y)^5 
  let a_5 := (3 - 2 * x)^5 - (3 - 2 * y)^5 
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3 + a_5)^2 = 3125 := by
sorry

end polynomial_expansion_l136_136442


namespace smallest_bdf_l136_136289

theorem smallest_bdf (a b c d e f : ℕ) (A : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : e > 0) (h6 : f > 0)
  (h7 : A = a * c * e / (b * d * f))
  (h8 : A = (a + 1) * c * e / (b * d * f) - 3)
  (h9 : A = a * (c + 1) * e / (b * d * f) - 4)
  (h10 : A = a * c * (e + 1) / (b * d * f) - 5) :
  b * d * f = 60 :=
by
  sorry

end smallest_bdf_l136_136289


namespace number_of_mango_trees_l136_136883

-- Define the conditions
variable (M : Nat) -- Number of mango trees
def num_papaya_trees := 2
def papayas_per_tree := 10
def mangos_per_tree := 20
def total_fruits := 80

-- Prove that the number of mango trees M is equal to 3
theorem number_of_mango_trees : 20 + (mangos_per_tree * M) = total_fruits -> M = 3 :=
by
  intro h
  sorry

end number_of_mango_trees_l136_136883


namespace annie_hamburgers_l136_136046

theorem annie_hamburgers (H : ℕ) (h₁ : 4 * H + 6 * 5 = 132 - 70) : H = 8 := by
  sorry

end annie_hamburgers_l136_136046


namespace factor_polynomial_l136_136326

theorem factor_polynomial : ∀ y : ℝ, 3 * y^2 - 27 = 3 * (y + 3) * (y - 3) :=
by
  intros y
  sorry

end factor_polynomial_l136_136326


namespace min_travel_time_l136_136809

/-- Two people, who have one bicycle, need to travel from point A to point B, which is 40 km away from point A. 
The first person walks at a speed of 4 km/h and rides the bicycle at 30 km/h, 
while the second person walks at a speed of 6 km/h and rides the bicycle at 20 km/h. 
Prove that the minimum time in which they can both get to point B is 25/9 hours. -/
theorem min_travel_time (d : ℕ) (v_w1 v_c1 v_w2 v_c2 : ℕ) (min_time : ℚ) 
  (h_d : d = 40)
  (h_v1_w : v_w1 = 4)
  (h_v1_c : v_c1 = 30)
  (h_v2_w : v_w2 = 6)
  (h_v2_c : v_c2 = 20)
  (h_min_time : min_time = 25 / 9) :
  ∃ y x : ℚ, 4*y + (2/3)*y*30 = 40 ∧ min_time = y + (2/3)*y :=
sorry

end min_travel_time_l136_136809


namespace hyperbola_condition_l136_136273

theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (m + 2)) + (y^2 / (m + 1)) = 1) ↔ (-2 < m ∧ m < -1) :=
by
  sorry

end hyperbola_condition_l136_136273


namespace dasha_rectangle_l136_136402

theorem dasha_rectangle:
  ∃ (a b c : ℤ), a * (2 * b + 2 * c - a) = 43 ∧ a = 1 ∧ b + c = 22 :=
by
  sorry

end dasha_rectangle_l136_136402


namespace product_of_y_coordinates_l136_136485

theorem product_of_y_coordinates (k : ℝ) (hk : k > 0) :
    let y1 := 2 + Real.sqrt (k^2 - 64)
    let y2 := 2 - Real.sqrt (k^2 - 64)
    y1 * y2 = 68 - k^2 :=
by 
  sorry

end product_of_y_coordinates_l136_136485


namespace area_of_picture_l136_136967

theorem area_of_picture {x y : ℕ} (hx : x > 1) (hy : y > 1) 
  (h : (2 * x + 3) * (y + 2) - x * y = 34) : x * y = 8 := 
by
  sorry

end area_of_picture_l136_136967


namespace find_x_l136_136591

noncomputable def h (x : ℚ) : ℚ :=
  (5 * ((x - 2) / 3) - 3)

theorem find_x : h (19/2) = 19/2 :=
by
  sorry

end find_x_l136_136591


namespace min_value_of_2a_plus_b_l136_136250

variable (a b : ℝ)

def condition := a > 0 ∧ b > 0 ∧ a - 2 * a * b + b = 0

-- Define what needs to be proved
theorem min_value_of_2a_plus_b (h : condition a b) : ∃ a b : ℝ, 2 * a + b = (3 / 2) + Real.sqrt 2 :=
sorry

end min_value_of_2a_plus_b_l136_136250


namespace work_completion_days_l136_136208

theorem work_completion_days (P R: ℕ) (hP: P = 80) (hR: R = 120) : P * R / (P + R) = 48 := by
  -- The proof is omitted as we are only writing the statement
  sorry

end work_completion_days_l136_136208


namespace solve_for_s_l136_136949

theorem solve_for_s : ∃ s, (∃ x, 4 * x^2 - 8 * x - 320 = 0) ∧ s = 81 :=
by {
  -- Sorry is used to skip the actual proof.
  sorry
}

end solve_for_s_l136_136949


namespace total_payment_is_correct_l136_136178

def length : ℕ := 30
def width : ℕ := 40
def construction_cost_per_sqft : ℕ := 3
def sealant_cost_per_sqft : ℕ := 1
def total_area : ℕ := length * width
def total_cost_per_sqft : ℕ := construction_cost_per_sqft + sealant_cost_per_sqft
def total_cost : ℕ := total_area * total_cost_per_sqft

theorem total_payment_is_correct : total_cost = 4800 := by
  sorry

end total_payment_is_correct_l136_136178


namespace total_weight_all_bags_sold_l136_136398

theorem total_weight_all_bags_sold (morning_potatoes afternoon_potatoes morning_onions afternoon_onions morning_carrots afternoon_carrots : ℕ)
  (weight_potatoes weight_onions weight_carrots total_weight : ℕ)
  (h_morning_potatoes : morning_potatoes = 29)
  (h_afternoon_potatoes : afternoon_potatoes = 17)
  (h_morning_onions : morning_onions = 15)
  (h_afternoon_onions : afternoon_onions = 22)
  (h_morning_carrots : morning_carrots = 12)
  (h_afternoon_carrots : afternoon_carrots = 9)
  (h_weight_potatoes : weight_potatoes = 7)
  (h_weight_onions : weight_onions = 5)
  (h_weight_carrots : weight_carrots = 4)
  (h_total_weight : total_weight = 591) :
  morning_potatoes + afternoon_potatoes * weight_potatoes +
  morning_onions + afternoon_onions * weight_onions +
  morning_carrots + afternoon_carrots * weight_carrots = total_weight :=
by {
  sorry
}

end total_weight_all_bags_sold_l136_136398


namespace smallest_positive_x_l136_136463

theorem smallest_positive_x (x : ℕ) (h : 42 * x + 9 ≡ 3 [MOD 15]) : x = 2 :=
sorry

end smallest_positive_x_l136_136463


namespace cost_of_55_lilies_l136_136800

-- Define the problem conditions
def price_per_dozen_lilies (p : ℝ) : Prop :=
  p * 24 = 30

def directly_proportional_price (p : ℝ) (n : ℕ) : ℝ :=
  p * n

-- State the problem to prove the cost of a 55 lily bouquet
theorem cost_of_55_lilies (p : ℝ) (c : ℝ) :
  price_per_dozen_lilies p →
  c = directly_proportional_price p 55 →
  c = 68.75 :=
by
  sorry

end cost_of_55_lilies_l136_136800


namespace Zhang_Laoshi_pens_l136_136185

theorem Zhang_Laoshi_pens (x : ℕ) (original_price new_price : ℝ)
  (discount : new_price = 0.75 * original_price)
  (more_pens : x * original_price = (x + 25) * new_price) :
  x = 75 :=
by
  sorry

end Zhang_Laoshi_pens_l136_136185


namespace intersection_in_fourth_quadrant_l136_136911

theorem intersection_in_fourth_quadrant (m : ℝ) :
  let x := (3 * m + 2) / 4
  let y := (-m - 2) / 8
  (x > 0) ∧ (y < 0) ↔ (m > -2 / 3) :=
by
  sorry

end intersection_in_fourth_quadrant_l136_136911


namespace record_loss_of_10_l136_136994

-- Definition of profit and loss recording
def record (x : Int) : Int :=
  if x ≥ 0 then x else -x

-- Condition: A profit of $20 should be recorded as +$20
axiom profit_recording : ∀ (p : Int), p ≥ 0 → record p = p

-- Condition: A loss should be recorded as a negative amount
axiom loss_recording : ∀ (l : Int), l < 0 → record l = l

-- Question: How should a loss of $10 be recorded?
-- Prove that if a small store lost $10, it should be recorded as -$10
theorem record_loss_of_10 : record (-10) = -10 :=
by sorry

end record_loss_of_10_l136_136994


namespace jerry_sister_increase_temp_l136_136479

theorem jerry_sister_increase_temp :
  let T0 := 40
  let T1 := 2 * T0
  let T2 := T1 - 30
  let T3 := T2 - 0.3 * T2
  let T4 := 59
  T4 - T3 = 24 := by
  sorry

end jerry_sister_increase_temp_l136_136479


namespace cost_of_previous_hay_l136_136712

theorem cost_of_previous_hay
    (x : ℤ)
    (previous_hay_bales : ℤ)
    (better_quality_hay_cost : ℤ)
    (additional_amount_needed : ℤ)
    (better_quality_hay_bales : ℤ)
    (new_total_cost : ℤ) :
    previous_hay_bales = 10 ∧ 
    better_quality_hay_cost = 18 ∧ 
    additional_amount_needed = 210 ∧ 
    better_quality_hay_bales = 2 * previous_hay_bales ∧ 
    new_total_cost = better_quality_hay_bales * better_quality_hay_cost ∧ 
    new_total_cost - additional_amount_needed = 10 * x → 
    x = 15 := by
  sorry

end cost_of_previous_hay_l136_136712


namespace find_abc_l136_136596

theorem find_abc (a b c : ℕ) (k : ℕ) 
  (h1 : a = 2 * k) 
  (h2 : b = 3 * k) 
  (h3 : c = 4 * k) 
  (h4 : k ≠ 0)
  (h5 : 2 * a - b + c = 10) : 
  a = 4 ∧ b = 6 ∧ c = 8 :=
sorry

end find_abc_l136_136596


namespace balloon_rearrangements_l136_136086

-- Define the letters involved: vowels and consonants
def vowels := ['A', 'O', 'O', 'O']
def consonants := ['B', 'L', 'L', 'N']

-- State the problem in Lean 4:
theorem balloon_rearrangements : 
  ∃ n : ℕ, 
  (∀ (vowels := ['A', 'O', 'O', 'O']) 
     (consonants := ['B', 'L', 'L', 'N']), 
     n = 32) := sorry  -- we state that the number of rearrangements is 32 but do not provide the proof itself.

end balloon_rearrangements_l136_136086


namespace seminar_duration_total_l136_136248

/-- The first part of the seminar lasted 4 hours and 45 minutes -/
def first_part_minutes := 4 * 60 + 45

/-- The second part of the seminar lasted 135 minutes -/
def second_part_minutes := 135

/-- The closing event lasted 500 seconds -/
def closing_event_minutes := 500 / 60

/-- The total duration of the seminar session in minutes, including the closing event, is 428 minutes -/
theorem seminar_duration_total :
  first_part_minutes + second_part_minutes + closing_event_minutes = 428 := by
  sorry

end seminar_duration_total_l136_136248


namespace inequality_proof_l136_136154

theorem inequality_proof
  {x1 x2 x3 x4 : ℝ}
  (h1 : x1 ≥ x2)
  (h2 : x2 ≥ x3)
  (h3 : x3 ≥ x4)
  (h4 : x4 ≥ 2)
  (h5 : x2 + x3 + x4 ≥ x1) :
  (x1 + x2 + x3 + x4)^2 ≤ 4 * x1 * x2 * x3 * x4 :=
by
  sorry

end inequality_proof_l136_136154


namespace sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees_l136_136098

theorem sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees :
  ∃ (n : ℕ), (n * (n - 3) / 2 = 14) → ((n - 2) * 180 = 900) :=
by
  sorry

end sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees_l136_136098


namespace find_other_factor_l136_136631

theorem find_other_factor (n : ℕ) (hn : n = 75) :
    ( ∃ k, k = 25 ∧ ∃ m, (k * 3^3 * m = 75 * 2^5 * 6^2 * 7^3) ) :=
by
  sorry

end find_other_factor_l136_136631


namespace fair_attendance_l136_136711

theorem fair_attendance :
  let this_year := 600
  let next_year := 2 * this_year
  let total_people := 2800
  let last_year := total_people - this_year - next_year
  (1200 - last_year = 200) ∧ (last_year = 1000) := by
  sorry

end fair_attendance_l136_136711


namespace polynomial_division_remainder_zero_l136_136340

theorem polynomial_division_remainder_zero (x : ℂ) (hx : x^5 + x^4 + x^3 + x^2 + x + 1 = 0)
  : (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end polynomial_division_remainder_zero_l136_136340


namespace rational_coordinates_l136_136136

theorem rational_coordinates (x : ℚ) : ∃ y : ℚ, 2 * x^3 + 2 * y^3 - 3 * x^2 - 3 * y^2 + 1 = 0 :=
by
  use (1 - x)
  sorry

end rational_coordinates_l136_136136


namespace probability_light_change_l136_136953

noncomputable def total_cycle_duration : ℕ := 45 + 5 + 50
def change_intervals : ℕ := 15

theorem probability_light_change :
  (15 : ℚ) / total_cycle_duration = 3 / 20 :=
by
  sorry

end probability_light_change_l136_136953


namespace money_lent_to_C_is_3000_l136_136896

def principal_B : ℕ := 5000
def time_B : ℕ := 2
def time_C : ℕ := 4
def rate_of_interest : ℕ := 12
def total_interest : ℕ := 2640
def interest_rate : ℚ := (rate_of_interest : ℚ) / 100
def interest_B : ℚ := principal_B * interest_rate * time_B
def interest_C (P_C : ℚ) : ℚ := P_C * interest_rate * time_C

theorem money_lent_to_C_is_3000 :
  ∃ P_C : ℚ, interest_B + interest_C P_C = total_interest ∧ P_C = 3000 :=
by
  use 3000
  unfold interest_B interest_C interest_rate principal_B time_B time_C rate_of_interest total_interest
  sorry

end money_lent_to_C_is_3000_l136_136896


namespace boxes_in_pantry_l136_136344

theorem boxes_in_pantry (b p c: ℕ) (h: p = 100) (hc: c = 50) (g: b = 225) (weeks: ℕ) (consumption: ℕ)
    (total_birdseed: ℕ) (new_boxes: ℕ) (initial_boxes: ℕ) : 
    weeks = 12 → consumption = (100 + 50) * weeks → total_birdseed = 1800 →
    new_boxes = 3 → total_birdseed = b * 8 → initial_boxes = 5 :=
by
  sorry

end boxes_in_pantry_l136_136344


namespace minimum_value_of_expression_l136_136892

theorem minimum_value_of_expression {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) : 
  (1 / (2 * a) + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 4 := 
sorry

end minimum_value_of_expression_l136_136892


namespace gcd_xyz_square_of_diff_l136_136904

theorem gcd_xyz_square_of_diff {x y z : ℕ} 
    (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) : 
    ∃ n : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = n ^ 2 :=
by
  sorry

end gcd_xyz_square_of_diff_l136_136904


namespace ace_first_king_second_prob_l136_136032

def cards : Type := { x : ℕ // x < 52 }

def ace (c : cards) : Prop := 
  c.1 = 0 ∨ c.1 = 1 ∨ c.1 = 2 ∨ c.1 = 3

def king (c : cards) : Prop := 
  c.1 = 4 ∨ c.1 = 5 ∨ c.1 = 6 ∨ c.1 = 7

def prob_ace_first_king_second : ℚ := 4 / 52 * 4 / 51

theorem ace_first_king_second_prob :
  prob_ace_first_king_second = 4 / 663 := by
  sorry

end ace_first_king_second_prob_l136_136032


namespace b_20_value_l136_136040

noncomputable def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => b (n+1) * b n

theorem b_20_value : b 19 = 2^4181 :=
sorry

end b_20_value_l136_136040


namespace length_segment_midpoints_diagonals_trapezoid_l136_136812

theorem length_segment_midpoints_diagonals_trapezoid
  (a b c d : ℝ)
  (h_side_lengths : (2 = a ∨ 2 = b ∨ 2 = c ∨ 2 = d) ∧ 
                    (10 = a ∨ 10 = b ∨ 10 = c ∨ 10 = d) ∧ 
                    (10 = a ∨ 10 = b ∨ 10 = c ∨ 10 = d) ∧ 
                    (20 = a ∨ 20 = b ∨ 20 = c ∨ 20 = d))
  (h_parallel_sides : (a = 20 ∧ b = 2) ∨ (a = 2 ∧ b = 20)) :
  (1/2) * |a - b| = 9 :=
by
  sorry

end length_segment_midpoints_diagonals_trapezoid_l136_136812


namespace max_x2_plus_4y_plus_3_l136_136325

theorem max_x2_plus_4y_plus_3 
  (x y : ℝ) 
  (h : x^2 + y^2 = 1) : 
  x^2 + 4*y + 3 ≤ 7 := sorry

end max_x2_plus_4y_plus_3_l136_136325


namespace eval_expr_l136_136610

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l136_136610


namespace bob_corn_stalks_per_row_l136_136722

noncomputable def corn_stalks_per_row
  (rows : ℕ)
  (bushels : ℕ)
  (stalks_per_bushel : ℕ) :
  ℕ :=
  (bushels * stalks_per_bushel) / rows

theorem bob_corn_stalks_per_row
  (rows : ℕ)
  (bushels : ℕ)
  (stalks_per_bushel : ℕ) :
  rows = 5 → bushels = 50 → stalks_per_bushel = 8 → corn_stalks_per_row rows bushels stalks_per_bushel = 80 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  unfold corn_stalks_per_row
  rfl

end bob_corn_stalks_per_row_l136_136722


namespace check_independence_and_expected_value_l136_136639

noncomputable def contingency_table (students: ℕ) (pct_75 : ℕ) (pct_less10 : ℕ) (num_75_10 : ℕ) : (ℕ × ℕ) × (ℕ × ℕ) :=
  let num_75 := students * pct_75 / 100
  let num_less10 := students * pct_less10 / 100
  let num_75_less10 := num_75 - num_75_10
  let num_not75 := students - num_75
  let num_not75_less10 := num_less10 - num_75_less10
  let num_not75_10 := num_not75 - num_not75_less10
  ((num_not75_less10, num_75_less10), (num_not75_10, num_75_10))

noncomputable def chi_square_statistic (a b c d : ℕ) (n: ℕ) : ℚ :=
  (n * ((a * d - b * c)^2)) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem check_independence_and_expected_value :
  let students := 500
  let pct_75 := 30
  let pct_less10 := 50
  let num_75_10 := 100
  let ((num_not75_less10, num_75_less10), (num_not75_10, num_75_10)) := contingency_table students pct_75 pct_less10 num_75_10
  let chi2 := chi_square_statistic num_not75_less10 num_75_less10 num_not75_10 num_75_10 students
  let critical_value := 10.828
  let p0 := 1 / 84
  let p1 := 3 / 14
  let p2 := 15 / 28
  let p3 := 5 / 21
  let expected_x := 0 * p0 + 1 * p1 + 2 * p2 + 3 * p3
  (chi2 > critical_value) ∧ (expected_x = 2) :=
by 
  sorry

end check_independence_and_expected_value_l136_136639


namespace average_sale_six_months_l136_136879

-- Define the sales for the first five months
def sale_month1 : ℕ := 6335
def sale_month2 : ℕ := 6927
def sale_month3 : ℕ := 6855
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562

-- Define the required sale for the sixth month
def sale_month6 : ℕ := 5091

-- Proof that the desired average sale for the six months is 6500
theorem average_sale_six_months : 
  (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / 6 = 6500 :=
by
  sorry

end average_sale_six_months_l136_136879


namespace solution_set_inequalities_l136_136264

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l136_136264


namespace product_pattern_l136_136566

theorem product_pattern (a b : ℕ) (h1 : b < 10) (h2 : 10 - b < 10) :
    (10 * a + b) * (10 * a + (10 - b)) = 100 * a * (a + 1) + b * (10 - b) :=
by
  sorry

end product_pattern_l136_136566


namespace radius_of_ball_l136_136063

theorem radius_of_ball (diameter depth : ℝ) (h₁ : diameter = 30) (h₂ : depth = 10) : 
  ∃ r : ℝ, r = 25 :=
by
  sorry

end radius_of_ball_l136_136063


namespace sqrt_x_minus_1_domain_l136_136626

theorem sqrt_x_minus_1_domain (x : ℝ) : (∃ y, y^2 = x - 1) → (x ≥ 1) := 
by 
  sorry

end sqrt_x_minus_1_domain_l136_136626


namespace repeating_decimal_fraction_l136_136127

noncomputable def repeating_decimal := 4.66666 -- Assuming repeating forever

theorem repeating_decimal_fraction : repeating_decimal = 14 / 3 :=
by 
  sorry

end repeating_decimal_fraction_l136_136127


namespace lilly_fish_count_l136_136239

-- Define the number of fish Rosy has
def rosy_fish : ℕ := 9

-- Define the total number of fish
def total_fish : ℕ := 19

-- Define the statement that Lilly has 10 fish given the conditions
theorem lilly_fish_count : rosy_fish + lilly_fish = total_fish → lilly_fish = 10 := by
  intro h
  sorry

end lilly_fish_count_l136_136239


namespace cos_arith_prog_impossible_l136_136390

noncomputable def sin_arith_prog (x y z : ℝ) : Prop :=
  (2 * Real.sin y = Real.sin x + Real.sin z) ∧ (Real.sin x < Real.sin y) ∧ (Real.sin y < Real.sin z)

theorem cos_arith_prog_impossible (x y z : ℝ) (h : sin_arith_prog x y z) : 
  ¬(2 * Real.cos y = Real.cos x + Real.cos z) := 
by 
  sorry

end cos_arith_prog_impossible_l136_136390


namespace probability_five_blue_marbles_is_correct_l136_136557

noncomputable def probability_of_five_blue_marbles : ℝ :=
let p_blue := (9 : ℝ) / 15
let p_red := (6 : ℝ) / 15
let specific_sequence_prob := p_blue ^ 5 * p_red ^ 3
let number_of_ways := (Nat.choose 8 5 : ℝ)
(number_of_ways * specific_sequence_prob)

theorem probability_five_blue_marbles_is_correct :
  probability_of_five_blue_marbles = 0.279 := by
sorry

end probability_five_blue_marbles_is_correct_l136_136557


namespace salt_solution_mixture_l136_136787

theorem salt_solution_mixture (x : ℝ) :  
  (0.80 * x + 0.35 * 150 = 0.55 * (150 + x)) → x = 120 :=
by 
  sorry

end salt_solution_mixture_l136_136787


namespace find_p_4_l136_136956

-- Define the polynomial p(x)
def p (x : ℕ) : ℚ := sorry

-- Given conditions
axiom h1 : p 1 = 1
axiom h2 : p 2 = 1 / 4
axiom h3 : p 3 = 1 / 9
axiom h4 : p 5 = 1 / 25

-- Prove that p(4) = -1/30
theorem find_p_4 : p 4 = -1 / 30 := 
  by sorry

end find_p_4_l136_136956


namespace fraction_equivalence_l136_136625

theorem fraction_equivalence : 
    (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 := by sorry

end fraction_equivalence_l136_136625


namespace line_equation_l136_136624

theorem line_equation (k : ℝ) (x1 y1 : ℝ) (P : x1 = 1 ∧ y1 = -1) (angle_slope : k = Real.tan (135 * Real.pi / 180)) : 
  ∃ (a b : ℝ), a = -1 ∧ b = -1 ∧ (y1 = k * x1 + b) ∧ (y1 = a * x1 + b) :=
by
  sorry

end line_equation_l136_136624


namespace emily_beads_l136_136045

theorem emily_beads (n : ℕ) (b : ℕ) (total_beads : ℕ) (h1 : n = 26) (h2 : b = 2) (h3 : total_beads = n * b) : total_beads = 52 :=
by
  sorry

end emily_beads_l136_136045


namespace exists_monomials_l136_136405

theorem exists_monomials (a b : ℕ) :
  ∃ x y : ℕ → ℕ → ℤ,
  (x 2 1 * y 2 1 = -12) ∧
  (∀ m n : ℕ, m ≠ 2 ∨ n ≠ 1 → x m n = 0 ∧ y m n = 0) ∧
  (∃ k l : ℤ, x 2 1 = k * (a ^ 2 * b ^ 1) ∧ y 2 1 = l * (a ^ 2 * b ^ 1) ∧ k + l = 1) :=
by
  sorry

end exists_monomials_l136_136405


namespace value_of_a_minus_b_l136_136474

theorem value_of_a_minus_b 
  (a b : ℤ)
  (h1 : 1010 * a + 1014 * b = 1018)
  (h2 : 1012 * a + 1016 * b = 1020) : 
  a - b = -3 :=
sorry

end value_of_a_minus_b_l136_136474


namespace solution_unique_for_alpha_neg_one_l136_136987

noncomputable def alpha : ℝ := sorry

axiom alpha_nonzero : alpha ≠ 0

def functional_eqn (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f (f (x + y)) = f (x + y) + f (x) * f (y) + alpha * x * y

theorem solution_unique_for_alpha_neg_one (f : ℝ → ℝ) :
  (alpha = -1 → (∀ x : ℝ, f x = x)) ∧ (alpha ≠ -1 → ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, functional_eqn f x y) :=
sorry

end solution_unique_for_alpha_neg_one_l136_136987


namespace max_x1_sq_plus_x2_sq_l136_136532

theorem max_x1_sq_plus_x2_sq (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = k - 2) 
  (h2 : x1 * x2 = k^2 + 3 * k + 5)
  (h3 : -4 ≤ k ∧ k ≤ -4 / 3) : 
  x1^2 + x2^2 ≤ 18 :=
by sorry

end max_x1_sq_plus_x2_sq_l136_136532


namespace prove_m_add_n_l136_136148

-- Definitions from conditions
variables (m n : ℕ)

def condition1 : Prop := m + 1 = 3
def condition2 : Prop := m = n - 1

-- Statement to prove
theorem prove_m_add_n (h1 : condition1 m) (h2 : condition2 m n) : m + n = 5 := 
sorry

end prove_m_add_n_l136_136148


namespace max_value_trig_expression_l136_136047

theorem max_value_trig_expression : ∀ x : ℝ, (3 * Real.cos x + 4 * Real.sin x) ≤ 5 := 
sorry

end max_value_trig_expression_l136_136047


namespace paul_reading_novel_l136_136971

theorem paul_reading_novel (x : ℕ) 
  (h1 : x - ((1 / 6) * x + 10) - ((1 / 5) * (x - ((1 / 6) * x + 10)) + 14) - ((1 / 4) * ((x - ((1 / 6) * x + 10) - ((1 / 5) * (x - ((1 / 6) * x + 10)) + 14)) + 16)) = 48) : 
  x = 161 :=
by sorry

end paul_reading_novel_l136_136971


namespace complex_number_simplification_l136_136584

theorem complex_number_simplification (i : ℂ) (hi : i^2 = -1) : i - (1 / i) = 2 * i :=
by
  sorry

end complex_number_simplification_l136_136584


namespace annual_percentage_increase_l136_136139

theorem annual_percentage_increase (present_value future_value : ℝ) (years: ℝ) (r : ℝ) 
  (h1 : present_value = 20000)
  (h2 : future_value = 24200)
  (h3 : years = 2) : 
  future_value = present_value * (1 + r)^years → r = 0.1 :=
sorry

end annual_percentage_increase_l136_136139


namespace intersection_A_B_l136_136678

open Set

-- Define sets A and B with given conditions
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | ∃ a ∈ A, x = 3 * a}

-- Prove the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {0, 3} := 
by
  sorry

end intersection_A_B_l136_136678


namespace emily_spending_l136_136981

theorem emily_spending (X Y : ℝ) 
  (h1 : (X + 2*X + 3*X + 12*X) = Y) : 
  X = Y / 18 := 
by
  sorry

end emily_spending_l136_136981


namespace age_sum_l136_136506

variable (b : ℕ)
variable (a : ℕ := b + 2)
variable (c : ℕ := b / 2)

theorem age_sum : b = 10 → a + b + c = 27 :=
by
  intros h
  rw [h]
  sorry

end age_sum_l136_136506


namespace find_b_l136_136452

theorem find_b (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 20)
  (h3 : (5 + 4 * 83 + 6 * 83^2 + 3 * 83^3 + 7 * 83^4 + 5 * 83^5 + 2 * 83^6 - b) % 17 = 0) :
  b = 8 :=
sorry

end find_b_l136_136452


namespace jerry_remaining_money_l136_136807

def cost_of_mustard_oil (price_per_liter : ℕ) (liters : ℕ) : ℕ := price_per_liter * liters
def cost_of_pasta (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds
def cost_of_sauce (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds

def total_cost (price_mustard_oil : ℕ) (liters_mustard : ℕ) (price_pasta : ℕ) (pounds_pasta : ℕ) (price_sauce : ℕ) (pounds_sauce : ℕ) : ℕ := 
  cost_of_mustard_oil price_mustard_oil liters_mustard + cost_of_pasta price_pasta pounds_pasta + cost_of_sauce price_sauce pounds_sauce

def remaining_money (initial_money : ℕ) (total_cost : ℕ) : ℕ := initial_money - total_cost

theorem jerry_remaining_money : 
  remaining_money 50 (total_cost 13 2 4 3 5 1) = 7 := by 
  sorry

end jerry_remaining_money_l136_136807


namespace youngest_brother_age_difference_l136_136923

def Rick_age : ℕ := 15
def Oldest_brother_age : ℕ := 2 * Rick_age
def Middle_brother_age : ℕ := Oldest_brother_age / 3
def Smallest_brother_age : ℕ := Middle_brother_age / 2
def Youngest_brother_age : ℕ := 3

theorem youngest_brother_age_difference :
  Smallest_brother_age - Youngest_brother_age = 2 :=
by
  -- sorry to skip the proof
  sorry

end youngest_brother_age_difference_l136_136923


namespace sin_6_cos_6_theta_proof_l136_136687

noncomputable def sin_6_cos_6_theta (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) : ℝ :=
  Real.sin θ ^ 6 + Real.cos θ ^ 6

theorem sin_6_cos_6_theta_proof (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) : 
  sin_6_cos_6_theta θ h = 19 / 64 :=
by
  sorry

end sin_6_cos_6_theta_proof_l136_136687


namespace solution_set_of_inequality_l136_136593

theorem solution_set_of_inequality (x : ℝ) : x^2 - 5 * |x| + 6 < 0 ↔ (-3 < x ∧ x < -2) ∨ (2 < x ∧ x < 3) :=
  sorry

end solution_set_of_inequality_l136_136593


namespace investment_years_l136_136385

def principal (P : ℝ) := P = 1200
def rate (r : ℝ) := r = 0.10
def interest_diff (P r : ℝ) (t : ℝ) :=
  let SI := P * r * t
  let CI := P * (1 + r)^t - P
  CI - SI = 12

theorem investment_years (P r : ℝ) (t : ℝ) 
  (h_principal : principal P) 
  (h_rate : rate r) 
  (h_diff : interest_diff P r t) : 
  t = 2 := 
sorry

end investment_years_l136_136385


namespace parallel_lines_condition_suff_not_nec_l136_136784

theorem parallel_lines_condition_suff_not_nec 
  (a : ℝ) : (a = -2) → 
  (∀ x y : ℝ, ax + 2 * y - 1 = 0) → 
  (∀ x y : ℝ, x + (a + 1) * y + 4 = 0) → 
  (∀ x1 y1 x2 y2 : ℝ, ((a = -2) → (2 * y1 - 2 * x1 = 1) → (y2 - x2 = -4) → (x1 = x2 → y1 = y2))) ∧ 
  (∃ b : ℝ, ¬ (b = -2) ∧ ((2 * y1 - b * x1 = 1) → (x2 - (b + 1) * y2 = -4) → ¬(x1 = x2 → y1 = y2)))
   :=
by
  sorry

end parallel_lines_condition_suff_not_nec_l136_136784


namespace symmetric_line_eq_x_axis_l136_136150

theorem symmetric_line_eq_x_axis (x y : ℝ) :
  (3 * x - 4 * y + 5 = 0) → (3 * x + 4 * y + 5 = 0) :=
sorry

end symmetric_line_eq_x_axis_l136_136150


namespace find_reals_abc_d_l136_136895

theorem find_reals_abc_d (a b c d : ℝ)
  (h1 : a * b * c + a * b + b * c + c * a + a + b + c = 1)
  (h2 : b * c * d + b * c + c * d + d * b + b + c + d = 9)
  (h3 : c * d * a + c * d + d * a + a * c + c + d + a = 9)
  (h4 : d * a * b + d * a + a * b + b * d + d + a + b = 9) :
  a = b ∧ b = c ∧ c = (2 : ℝ)^(1/3) - 1 ∧ d = 5 * (2 : ℝ)^(1/3) - 1 :=
sorry

end find_reals_abc_d_l136_136895


namespace number_of_teams_l136_136913

theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 * 10 = 1050) : n = 15 :=
by 
  sorry

end number_of_teams_l136_136913


namespace certain_event_positive_integers_sum_l136_136966

theorem certain_event_positive_integers_sum :
  ∀ (a b : ℕ), a > 0 → b > 0 → a + b > 1 :=
by
  intros a b ha hb
  sorry

end certain_event_positive_integers_sum_l136_136966


namespace no_integer_solutions_system_l136_136850

theorem no_integer_solutions_system :
  ¬(∃ x y z : ℤ, 
    x^6 + x^3 + x^3 * y + y = 147^157 ∧ 
    x^3 + x^3 * y + y^2 + y + z^9 = 157^147) :=
  sorry

end no_integer_solutions_system_l136_136850


namespace distance_Tim_covers_l136_136975

theorem distance_Tim_covers (initial_distance : ℕ) (tim_speed elan_speed : ℕ) (double_speed_time : ℕ)
  (h_initial_distance : initial_distance = 30)
  (h_tim_speed : tim_speed = 10)
  (h_elan_speed : elan_speed = 5)
  (h_double_speed_time : double_speed_time = 1) :
  ∃ t d : ℕ, d = 20 ∧ t ∈ {t | t = d / tim_speed + (initial_distance - d) / (tim_speed * 2)} :=
sorry

end distance_Tim_covers_l136_136975


namespace solve_equation_l136_136792

theorem solve_equation (x : ℝ) :
  (4 * x + 1) * (3 * x + 1) * (2 * x + 1) * (x + 1) = 3 * x ^ 4  →
  x = (-5 + Real.sqrt 13) / 6 ∨ x = (-5 - Real.sqrt 13) / 6 :=
by
  sorry

end solve_equation_l136_136792


namespace benny_spending_l136_136199

variable (S D V : ℝ)

theorem benny_spending :
  (200 - 45) = S + (D / 110) + (V / 0.75) :=
by
  sorry

end benny_spending_l136_136199


namespace num_divisors_m2_less_than_m_not_divide_m_l136_136016

namespace MathProof

def m : ℕ := 2^20 * 3^15 * 5^6

theorem num_divisors_m2_less_than_m_not_divide_m :
  let m2 := m ^ 2
  let total_divisors_m2 := 41 * 31 * 13
  let total_divisors_m := 21 * 16 * 7
  let divisors_m2_less_than_m := (total_divisors_m2 - 1) / 2
  divisors_m2_less_than_m - total_divisors_m = 5924 :=
by sorry

end MathProof

end num_divisors_m2_less_than_m_not_divide_m_l136_136016


namespace solve_for_x_l136_136746

theorem solve_for_x : ∃ x : ℝ, 4 * x + 6 * x = 360 - 9 * (x - 4) ∧ x = 396 / 19 :=
by
  sorry

end solve_for_x_l136_136746


namespace solution_set_l136_136841

-- Define the two conditions as hypotheses
variables (x : ℝ)

def condition1 : Prop := x + 6 ≤ 8
def condition2 : Prop := x - 7 < 2 * (x - 3)

-- The statement to prove
theorem solution_set (h1 : condition1 x) (h2 : condition2 x) : -1 < x ∧ x ≤ 2 :=
by
  sorry

end solution_set_l136_136841


namespace susan_coins_value_l136_136683

-- Define the conditions as Lean functions and statements.
def total_coins (n d : ℕ) := n + d = 30
def value_if_swapped (n : ℕ) := 10 * n + 5 * (30 - n)
def value_original (n : ℕ) := 5 * n + 10 * (30 - n)
def conditions (n : ℕ) := value_if_swapped n = value_original n + 90

-- The proof statement
theorem susan_coins_value (n d : ℕ) (h1 : total_coins n d) (h2 : conditions n) : 5 * n + 10 * d = 180 := by
  sorry

end susan_coins_value_l136_136683


namespace find_number_l136_136540

theorem find_number (x : ℝ) : (x / 2 = x - 5) → x = 10 :=
by
  intro h
  sorry

end find_number_l136_136540


namespace problem_statement_l136_136444

theorem problem_statement
  (x y : ℝ)
  (h1 : x * y = 1 / 9)
  (h2 : x * (y + 1) = 7 / 9)
  (h3 : y * (x + 1) = 5 / 18) :
  (x + 1) * (y + 1) = 35 / 18 :=
by
  sorry

end problem_statement_l136_136444


namespace algebraic_expression_correct_l136_136887

theorem algebraic_expression_correct (a b : ℝ) (h : a = 7 - 3 * b) : a^2 + 6 * a * b + 9 * b^2 = 49 := 
by sorry

end algebraic_expression_correct_l136_136887


namespace triangle_right_angle_l136_136025

theorem triangle_right_angle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A = B - C) : B = 90 :=
by sorry

end triangle_right_angle_l136_136025


namespace find_m_plus_n_l136_136675

variable (x n m : ℝ)

def condition : Prop := (x + 5) * (x + n) = x^2 + m * x - 5

theorem find_m_plus_n (hnm : condition x n m) : m + n = 3 := 
sorry

end find_m_plus_n_l136_136675


namespace find_f_2018_l136_136428

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_functional_eq : ∀ x : ℝ, f x = - (1 / f (x + 3))
axiom f_at_4 : f 4 = -2018

theorem find_f_2018 : f 2018 = -2018 :=
  sorry

end find_f_2018_l136_136428


namespace closest_point_on_ellipse_l136_136965

theorem closest_point_on_ellipse : 
  ∃ (x y : ℝ), (7 * x^2 + 4 * y^2 = 28 ∧ 3 * x - 2 * y - 16 = 0) ∧ 
  (∀ (x' y' : ℝ), 7 * x'^2 + 4 * y'^2 = 28 → dist (x, y) (0, 0) ≤ dist (x', y') (0, 0)) :=
sorry

end closest_point_on_ellipse_l136_136965


namespace original_height_l136_136592

theorem original_height (h : ℝ) (h_rebound : ∀ n : ℕ, h / (4/3)^(n+1) > 0) (total_distance : ∀ h : ℝ, h*(1 + 1.5 + 1.5*(0.75) + 1.5*(0.75)^2 + 1.5*(0.75)^3 + (0.75)^4) = 305) :
  h = 56.3 := 
sorry

end original_height_l136_136592


namespace hours_to_seconds_l136_136893

theorem hours_to_seconds : 
  (3.5 * 60 * 60) = 12600 := 
by 
  sorry

end hours_to_seconds_l136_136893


namespace product_of_abc_l136_136543

noncomputable def abc_product (a b c : ℝ) : ℝ :=
  a * b * c

theorem product_of_abc (a b c m : ℝ) 
    (h1 : a + b + c = 300)
    (h2 : m = 5 * a)
    (h3 : m = b + 14)
    (h4 : m = c - 14) : 
    abc_product a b c = 664500 :=
by sorry

end product_of_abc_l136_136543


namespace xy_extrema_l136_136422

noncomputable def xy_product (a : ℝ) : ℝ := a^2 - 1

theorem xy_extrema (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^2 + y^2 = -a^2 + 2) : 
  -1 ≤ xy_product a ∧ xy_product a ≤ 1/3 :=
by
  sorry

end xy_extrema_l136_136422


namespace percentage_increase_correct_l136_136051

def highest_price : ℕ := 24
def lowest_price : ℕ := 16

theorem percentage_increase_correct :
  ((highest_price - lowest_price) * 100 / lowest_price) = 50 :=
by
  sorry

end percentage_increase_correct_l136_136051


namespace spherical_to_rectangular_correct_l136_136725

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 3 (Real.pi / 2) (Real.pi / 3) = (0, (3 * Real.sqrt 3) / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_correct_l136_136725


namespace probability_A_given_B_l136_136677

def roll_outcomes : ℕ := 6^3 -- Total number of possible outcomes when rolling three dice

def P_AB : ℚ := 60 / 216 -- Probability of both events A and B happening

def P_B : ℚ := 91 / 216 -- Probability of event B happening

theorem probability_A_given_B : (P_AB / P_B) = (60 / 91) := by
  sorry

end probability_A_given_B_l136_136677


namespace solution_set_of_inequality_l136_136856

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (hA : f 0 = -2)
  (hB : f (-3) = 2) :
  { x : ℝ | |f (x - 2)| > 2 } = { x : ℝ | x < -1 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end solution_set_of_inequality_l136_136856


namespace triangle_properties_l136_136513

theorem triangle_properties (b c : ℝ) (C : ℝ)
  (hb : b = 10)
  (hc : c = 5 * Real.sqrt 6)
  (hC : C = Real.pi / 3) :
  let R := c / (2 * Real.sin C)
  let B := Real.arcsin (b * Real.sin C / c)
  R = 5 * Real.sqrt 2 ∧ B = Real.pi / 4 :=
by
  sorry

end triangle_properties_l136_136513


namespace max_length_OB_l136_136830

-- Define the problem conditions
def angle_AOB : ℝ := 45
def length_AB : ℝ := 2
def max_sin_angle_OAB : ℝ := 1

-- Claim to be proven
theorem max_length_OB : ∃ OB_max, OB_max = 2 * Real.sqrt 2 :=
by
  sorry

end max_length_OB_l136_136830


namespace theater_ticket_sales_l136_136710

theorem theater_ticket_sales (A K : ℕ) (h1 : A + K = 275) (h2 :  12 * A + 5 * K = 2150) : K = 164 := by
  sorry

end theater_ticket_sales_l136_136710


namespace y_coord_equidistant_l136_136534

theorem y_coord_equidistant (y : ℝ) :
  (dist (0, y) (-3, 0) = dist (0, y) (2, 5)) ↔ y = 2 := by
  sorry

end y_coord_equidistant_l136_136534


namespace remainder_13_pow_2031_mod_100_l136_136973

theorem remainder_13_pow_2031_mod_100 : (13^2031) % 100 = 17 :=
by sorry

end remainder_13_pow_2031_mod_100_l136_136973


namespace eighth_grade_students_l136_136915

def avg_books (total_books : ℕ) (num_students : ℕ) : ℚ :=
  total_books / num_students

theorem eighth_grade_students (x : ℕ) (y : ℕ)
  (h1 : x + y = 1800)
  (h2 : y = x - 150)
  (h3 : avg_books x 1800 = 1.5 * avg_books (x - 150) 1800) :
  y = 450 :=
by {
  sorry
}

end eighth_grade_students_l136_136915


namespace probability_range_l136_136163

noncomputable def probability_distribution (K : ℕ) : ℝ :=
  if K > 0 then 1 / (2^K) else 0

theorem probability_range (h2 : 2 < 3) (h3 : 3 ≤ 4) :
  probability_distribution 3 + probability_distribution 4 = 3 / 16 :=
by
  sorry

end probability_range_l136_136163


namespace ratio_of_installing_to_downloading_l136_136550

noncomputable def timeDownloading : ℕ := 10

noncomputable def ratioTimeSpent (installingTime : ℕ) : ℚ :=
  let tutorialTime := 3 * (timeDownloading + installingTime)
  let totalTime := timeDownloading + installingTime + tutorialTime
  if totalTime = 60 then
    (installingTime : ℚ) / (timeDownloading : ℚ)
  else 0

theorem ratio_of_installing_to_downloading : ratioTimeSpent 5 = 1 / 2 := by
  sorry

end ratio_of_installing_to_downloading_l136_136550


namespace percent_pelicans_non_swans_l136_136346

noncomputable def percent_geese := 0.20
noncomputable def percent_swans := 0.30
noncomputable def percent_herons := 0.10
noncomputable def percent_ducks := 0.25
noncomputable def percent_pelicans := 0.15

theorem percent_pelicans_non_swans :
  (percent_pelicans / (1 - percent_swans)) * 100 = 21.43 := 
by 
  sorry

end percent_pelicans_non_swans_l136_136346


namespace minimal_hair_loss_l136_136361

theorem minimal_hair_loss (cards : Fin 100 → ℕ)
    (sum_sage1 : ℕ)
    (communicate_card_numbers : List ℕ)
    (communicate_sum : ℕ) :
    (∀ i : Fin 100, (communicate_card_numbers.contains (cards i))) →
    communicate_sum = sum_sage1 →
    sum_sage1 = List.sum communicate_card_numbers →
    communicate_card_numbers.length = 100 →
    ∃ (minimal_loss : ℕ), minimal_loss = 101 := by
  sorry

end minimal_hair_loss_l136_136361


namespace x_y_sum_l136_136553

theorem x_y_sum (x y : ℝ) 
  (h1 : (x-1)^3 + 1997*(x-1) = -1)
  (h2 : (y-1)^3 + 1997*(y-1) = 1) :
  x + y = 2 :=
sorry

end x_y_sum_l136_136553


namespace distance_before_rest_l136_136316

theorem distance_before_rest (total_distance after_rest_distance : ℝ) (h1 : total_distance = 1) (h2 : after_rest_distance = 0.25) :
  total_distance - after_rest_distance = 0.75 :=
by sorry

end distance_before_rest_l136_136316


namespace problem1_problem2_l136_136632

-- Definitions related to the given problem
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 9 / (Real.cos θ^2 + 9 * Real.sin θ^2)

def standard_curve (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Proving the standard equation of the curve
theorem problem1 (ρ θ : ℝ) (h : polar_curve ρ θ) : ∃ x y, standard_curve x y :=
  sorry

-- Proving the perpendicular condition and its consequence
theorem problem2 (ρ1 ρ2 α : ℝ)
  (hA : polar_curve ρ1 α)
  (hB : polar_curve ρ2 (α + π/2))
  (perpendicular : ∀ (A B : (ℝ × ℝ)), A ≠ B → A.1 * B.1 + A.2 * B.2 = 0) :
  (1 / ρ1^2) + (1 / ρ2^2) = 10 / 9 :=
  sorry

end problem1_problem2_l136_136632


namespace r_at_6_l136_136848

-- Define the monic quintic polynomial r(x) with given conditions
def r (x : ℝ) : ℝ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) + x^2 + 2 

-- Given conditions
axiom r_1 : r 1 = 3
axiom r_2 : r 2 = 7
axiom r_3 : r 3 = 13
axiom r_4 : r 4 = 21
axiom r_5 : r 5 = 31

-- Proof goal
theorem r_at_6 : r 6 = 158 :=
by
  sorry

end r_at_6_l136_136848


namespace sequence_formula_l136_136366

theorem sequence_formula (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 5)
  (h3 : ∀ n > 1, a (n + 1) = 2 * a n - a (n - 1)) :
  ∀ n, a n = 4 * n - 3 :=
by
  sorry

end sequence_formula_l136_136366


namespace partI_partII_l136_136235

-- Define the absolute value function
def f (x : ℝ) := |x - 1|

-- Part I: Solve the inequality f(x) - f(x+2) < 1
theorem partI (x : ℝ) (h : f x - f (x + 2) < 1) : x > -1 / 2 := 
sorry

-- Part II: Find the range of values for a such that x - f(x + 1 - a) ≤ 1 for all x in [1,2]
theorem partII (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x - f (x + 1 - a) ≤ 1) : a ≤ 1 ∨ a ≥ 3 := 
sorry

end partI_partII_l136_136235


namespace inclination_angle_x_equals_3_is_90_l136_136659

-- Define the condition that line x = 3 is vertical
def is_vertical_line (x : ℝ) : Prop := x = 3

-- Define the inclination angle property for a vertical line
def inclination_angle_of_vertical_line_is_90 (x : ℝ) (h : is_vertical_line x) : ℝ :=
90   -- The angle is 90 degrees

-- Theorem statement to prove the inclination angle of the line x = 3 is 90 degrees
theorem inclination_angle_x_equals_3_is_90 :
  inclination_angle_of_vertical_line_is_90 3 (by simp [is_vertical_line]) = 90 :=
sorry  -- proof goes here


end inclination_angle_x_equals_3_is_90_l136_136659


namespace part1_part2_l136_136516

variable (a b c : ℝ)

-- Conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : a^2 + b^2 + 4*c^2 = 3

-- Part 1: Prove that a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 := sorry

-- Part 2: Given b = 2c, prove that 1/a + 1/c ≥ 3
axiom h5 : b = 2*c
theorem part2 : 1/a + 1/c ≥ 3 := sorry

end part1_part2_l136_136516


namespace total_cost_price_l136_136201

theorem total_cost_price (P_ct P_ch P_bs : ℝ) (h1 : 8091 = P_ct * 1.24)
    (h2 : 5346 = P_ch * 1.18 * 0.95) (h3 : 11700 = P_bs * 1.30) : 
    P_ct + P_ch + P_bs = 20295 := 
by 
    sorry

end total_cost_price_l136_136201


namespace max_value_of_expr_l136_136554

noncomputable def max_value (t : ℕ) : ℝ := (3^t - 2*t)*t / 9^t

theorem max_value_of_expr :
  ∃ t : ℕ, max_value t = 1 / 8 :=
sorry

end max_value_of_expr_l136_136554


namespace average_age_is_26_l136_136337

noncomputable def devin_age : ℕ := 12
noncomputable def eden_age : ℕ := 2 * devin_age
noncomputable def eden_mom_age : ℕ := 2 * eden_age
noncomputable def eden_grandfather_age : ℕ := (devin_age + eden_age + eden_mom_age) / 2
noncomputable def eden_aunt_age : ℕ := eden_mom_age / devin_age

theorem average_age_is_26 : 
  (devin_age + eden_age + eden_mom_age + eden_grandfather_age + eden_aunt_age) / 5 = 26 :=
by {
  sorry
}

end average_age_is_26_l136_136337


namespace quadratic_has_two_distinct_real_roots_iff_l136_136044

theorem quadratic_has_two_distinct_real_roots_iff (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6 * x - m = 0 ∧ y^2 - 6 * y - m = 0) ↔ m > -9 :=
by 
  sorry

end quadratic_has_two_distinct_real_roots_iff_l136_136044


namespace total_dolls_l136_136336

def sisters_dolls : ℝ := 8.5

def hannahs_dolls : ℝ := 5.5 * sisters_dolls

theorem total_dolls : hannahs_dolls + sisters_dolls = 55.25 :=
by
  -- Proof is omitted
  sorry

end total_dolls_l136_136336


namespace f_cos_eq_l136_136829

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Given condition
axiom f_sin_eq : f (Real.sin x) = 3 - Real.cos (2 * x)

-- The statement we want to prove
theorem f_cos_eq : f (Real.cos x) = 3 + Real.cos (2 * x) := 
by
  sorry

end f_cos_eq_l136_136829


namespace side_length_of_square_l136_136017

theorem side_length_of_square (m : ℕ) (a : ℕ) (hm : m = 100) (ha : a^2 = m) : a = 10 :=
by 
  sorry

end side_length_of_square_l136_136017


namespace one_minus_repeating_three_l136_136587

theorem one_minus_repeating_three : ∀ b : ℚ, b = 1 / 3 → 1 - b = 2 / 3 :=
by
  intro b hb
  rw [hb]
  norm_num

end one_minus_repeating_three_l136_136587


namespace machines_together_work_time_l136_136447

theorem machines_together_work_time :
  let rate_A := 1 / 4
  let rate_B := 1 / 12
  let rate_C := 1 / 6
  let rate_D := 1 / 8
  let rate_E := 1 / 18
  let total_rate := rate_A + rate_B + rate_C + rate_D + rate_E
  total_rate ≠ 0 → 
  let total_time := 1 / total_rate
  total_time = 72 / 49 :=
by
  sorry

end machines_together_work_time_l136_136447


namespace john_unanswered_questions_l136_136594

theorem john_unanswered_questions :
  ∃ (c w u : ℕ), (30 + 4 * c - w = 84) ∧ (5 * c + 2 * u = 93) ∧ (c + w + u = 30) ∧ (u = 9) :=
by
  sorry

end john_unanswered_questions_l136_136594


namespace stream_speed_fraction_l136_136957

theorem stream_speed_fraction (B S : ℝ) (h1 : B = 3 * S) 
  (h2 : (1 / (B - S)) = 2 * (1 / (B + S))) : (S / B) = 1 / 3 :=
sorry

end stream_speed_fraction_l136_136957


namespace non_isosceles_triangle_has_equidistant_incenter_midpoints_l136_136504

structure Triangle (α : Type*) :=
(a b c : α)
(incenter : α)
(midpoint_a_b : α)
(midpoint_b_c : α)
(midpoint_c_a : α)
(equidistant : Bool)
(non_isosceles : Bool)

-- Define the triangle with the specified properties.
noncomputable def counterexample_triangle : Triangle ℝ :=
{ a := 3,
  b := 4,
  c := 5, 
  incenter := 1, -- incenter length for the right triangle.
  midpoint_a_b := 2.5,
  midpoint_b_c := 2,
  midpoint_c_a := 1.5,
  equidistant := true,    -- midpoints of two sides are equidistant from incenter
  non_isosceles := true } -- the triangle is not isosceles

theorem non_isosceles_triangle_has_equidistant_incenter_midpoints :
  ∃ (T : Triangle ℝ), T.equidistant ∧ T.non_isosceles := by
  use counterexample_triangle
  sorry

end non_isosceles_triangle_has_equidistant_incenter_midpoints_l136_136504


namespace john_initial_clean_jerk_weight_l136_136049

def initial_snatch_weight : ℝ := 50
def increase_rate : ℝ := 1.8
def total_new_lifting_capacity : ℝ := 250

theorem john_initial_clean_jerk_weight :
  ∃ (C : ℝ), 2 * C + (increase_rate * initial_snatch_weight) = total_new_lifting_capacity ∧ C = 80 := by
  sorry

end john_initial_clean_jerk_weight_l136_136049


namespace larger_integer_l136_136906

theorem larger_integer (x y : ℕ) (h_diff : y - x = 8) (h_prod : x * y = 272) : y = 20 :=
by
  sorry

end larger_integer_l136_136906


namespace second_company_managers_percent_l136_136783

/-- A company's workforce consists of 10 percent managers and 90 percent software engineers.
    Another company's workforce consists of some percent managers, 10 percent software engineers, 
    and 60 percent support staff. The two companies merge, and the resulting company's 
    workforce consists of 25 percent managers. If 25 percent of the workforce originated from the 
    first company, what percent of the second company's workforce were managers? -/
theorem second_company_managers_percent
  (F S : ℝ)
  (h1 : 0.10 * F + m * S = 0.25 * (F + S))
  (h2 : F = 0.25 * (F + S)) :
  m = 0.225 :=
sorry

end second_company_managers_percent_l136_136783


namespace t_n_closed_form_t_2022_last_digit_l136_136084

noncomputable def t_n (n : ℕ) : ℕ :=
  (4^n - 3 * 3^n + 3 * 2^n - 1) / 6

theorem t_n_closed_form (n : ℕ) (hn : 0 < n) :
  t_n n = (4^n - 3 * 3^n + 3 * 2^n - 1) / 6 :=
by
  sorry

theorem t_2022_last_digit :
  (t_n 2022) % 10 = 1 :=
by
  sorry

end t_n_closed_form_t_2022_last_digit_l136_136084


namespace largest_of_three_consecutive_integers_l136_136308

theorem largest_of_three_consecutive_integers (N : ℤ) (h : N + (N + 1) + (N + 2) = 18) : N + 2 = 7 :=
sorry

end largest_of_three_consecutive_integers_l136_136308


namespace solve_equation1_solve_equation2_l136_136043

theorem solve_equation1 (x : ℝ) (h1 : 3 * x^3 - 15 = 9) : x = 2 :=
sorry

theorem solve_equation2 (x : ℝ) (h2 : 2 * (x - 1)^2 = 72) : x = 7 ∨ x = -5 :=
sorry

end solve_equation1_solve_equation2_l136_136043


namespace equation_one_solution_equation_two_solution_l136_136541

-- Define the conditions and prove the correctness of solutions to the equations
theorem equation_one_solution (x : ℝ) (h : 3 / (x - 2) = 9 / x) : x = 3 :=
by
  sorry

theorem equation_two_solution (x : ℝ) (h : x / (x + 1) = 2 * x / (3 * x + 3) - 1) : x = -3 / 4 :=
by
  sorry

end equation_one_solution_equation_two_solution_l136_136541


namespace min_value_2x_minus_y_l136_136802

theorem min_value_2x_minus_y :
  ∃ (x y : ℝ), (y = abs (x - 1) ∨ y = 2) ∧ (y ≤ 2) ∧ (2 * x - y = -4) :=
by
  sorry

end min_value_2x_minus_y_l136_136802


namespace blue_square_area_percentage_l136_136345

theorem blue_square_area_percentage (k : ℝ) (H1 : 0 < k) 
(Flag_area : ℝ := k^2) -- total area of the flag
(Cross_area : ℝ := 0.49 * Flag_area) -- total area of the cross and blue squares 
(one_blue_square_area : ℝ := Cross_area / 3) -- area of one blue square
(percentage : ℝ := one_blue_square_area / Flag_area * 100) :
percentage = 16.33 :=
by
  sorry

end blue_square_area_percentage_l136_136345


namespace find_distance_CD_l136_136941

noncomputable def distance_CD : ℝ :=
  let C : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (3, 6)
  Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)

theorem find_distance_CD :
  ∀ (C D : ℝ × ℝ), 
  (C = (0, 0) ∧ D = (3, 6)) ∧ 
  (∃ x y : ℝ, (y^2 = 12 * x ∧ (x^2 + y^2 - 4 * x - 6 * y = 0))) → 
  distance_CD = 3 * Real.sqrt 5 :=
by
  sorry

end find_distance_CD_l136_136941


namespace arctan_sum_pi_l136_136529

open Real

theorem arctan_sum_pi : arctan (1 / 3) + arctan (3 / 8) + arctan (8 / 3) = π := 
sorry

end arctan_sum_pi_l136_136529


namespace riverview_problem_l136_136648

theorem riverview_problem (h c : Nat) (p : Nat := 4 * h) (s : Nat := 5 * c) (d : Nat := 4 * p) :
  (p + h + s + c + d = 52 → false) :=
by {
  sorry
}

end riverview_problem_l136_136648


namespace solve_equation_l136_136799

theorem solve_equation (x : ℝ) (h₀ : x = 46) :
  ( (8 / (Real.sqrt (x - 10) - 10)) + 
    (2 / (Real.sqrt (x - 10) - 5)) + 
    (9 / (Real.sqrt (x - 10) + 5)) + 
    (15 / (Real.sqrt (x - 10) + 10)) = 0) := 
by 
  sorry

end solve_equation_l136_136799


namespace distribute_candies_l136_136487

theorem distribute_candies (n : ℕ) (h : ∃ m : ℕ, n = 2^m) : 
  ∀ k : ℕ, ∃ i : ℕ, (1 / 2) * i * (i + 1) % n = k :=
sorry

end distribute_candies_l136_136487


namespace problem_1_l136_136917

variable (x : ℝ) (a : ℝ)

theorem problem_1 (h1 : x - 1/x = 3) (h2 : a = x^2 + 1/x^2) : a = 11 := sorry

end problem_1_l136_136917


namespace add_fractions_result_l136_136352

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l136_136352


namespace trajectory_of_point_M_l136_136582

theorem trajectory_of_point_M (a x y : ℝ) (h: 0 < a) (A B M : ℝ × ℝ)
    (hA : A = (x, 0)) (hB : B = (0, y)) (hAB_length : Real.sqrt (x^2 + y^2) = 2 * a)
    (h_ratio : ∃ k, k ≠ 0 ∧ ∃ k', k' ≠ 0 ∧ A = k • M + k' • B ∧ (k + k' = 1) ∧ (k / k' = 1 / 2)) :
    (x / (4 / 3 * a))^2 + (y / (2 / 3 * a))^2 = 1 :=
sorry

end trajectory_of_point_M_l136_136582


namespace diagonal_plane_angle_l136_136755

theorem diagonal_plane_angle
  (α : Real)
  (a : Real)
  (plane_square_angle_with_plane : Real)
  (diagonal_plane_angle : Real) 
  (h1 : plane_square_angle_with_plane = α) :
  diagonal_plane_angle = Real.arcsin (Real.sin α / Real.sqrt 2) :=
sorry

end diagonal_plane_angle_l136_136755


namespace permutations_behind_Alice_l136_136331

theorem permutations_behind_Alice (n : ℕ) (h : n = 7) : 
  (Nat.factorial n) = 5040 :=
by
  rw [h]
  rw [Nat.factorial]
  sorry

end permutations_behind_Alice_l136_136331


namespace total_difference_in_cards_l136_136638

theorem total_difference_in_cards (cards_chris : ℕ) (cards_charlie : ℕ) (cards_diana : ℕ) (cards_ethan : ℕ)
  (h_chris : cards_chris = 18)
  (h_charlie : cards_charlie = 32)
  (h_diana : cards_diana = 25)
  (h_ethan : cards_ethan = 40) :
  (cards_charlie - cards_chris) + (cards_diana - cards_chris) + (cards_ethan - cards_chris) = 43 := by
  sorry

end total_difference_in_cards_l136_136638


namespace exit_condition_l136_136934

-- Define the loop structure in a way that is consistent with how the problem is described
noncomputable def program_loop (k : ℕ) : ℕ :=
  if k < 7 then 35 else sorry -- simulate the steps of the program

-- The proof goal is to show that the condition which stops the loop when s = 35 is k ≥ 7
theorem exit_condition (k : ℕ) (s : ℕ) : 
  (program_loop k = 35) → (k ≥ 7) :=
by {
  sorry
}

end exit_condition_l136_136934


namespace frac_1_7_correct_l136_136798

-- Define the fraction 1/7
def frac_1_7 : ℚ := 1 / 7

-- Define the decimal approximation 0.142857142857 as a rational number
def dec_approx : ℚ := 142857142857 / 10^12

-- Define the small fractional difference
def small_diff : ℚ := 1 / (7 * 10^12)

-- The theorem to be proven
theorem frac_1_7_correct :
  frac_1_7 = dec_approx + small_diff := 
sorry

end frac_1_7_correct_l136_136798


namespace river_width_after_30_seconds_l136_136211

noncomputable def width_of_river (initial_width : ℝ) (width_increase_rate : ℝ) (rowing_rate : ℝ) (time_taken : ℝ) : ℝ :=
  initial_width + (time_taken * rowing_rate * (width_increase_rate / 10))

theorem river_width_after_30_seconds :
  width_of_river 50 2 5 30 = 80 :=
by
  -- it suffices to check the calculations here
  sorry

end river_width_after_30_seconds_l136_136211


namespace helpers_cakes_l136_136728

theorem helpers_cakes (S : ℕ) (helpers large_cakes small_cakes : ℕ)
  (h1 : helpers = 10)
  (h2 : large_cakes = 2)
  (h3 : small_cakes = 700)
  (h4 : 1 * helpers * large_cakes = 20)
  (h5 : 2 * helpers * S = small_cakes) :
  S = 35 :=
by
  sorry

end helpers_cakes_l136_136728


namespace probability_two_red_balls_l136_136101

def total_balls : ℕ := 15
def red_balls_initial : ℕ := 7
def blue_balls_initial : ℕ := 8
def red_balls_after_first_draw : ℕ := 6
def remaining_balls_after_first_draw : ℕ := 14

theorem probability_two_red_balls :
  (red_balls_initial / total_balls) *
  (red_balls_after_first_draw / remaining_balls_after_first_draw) = 1 / 5 :=
by sorry

end probability_two_red_balls_l136_136101


namespace maximize_sum_l136_136073

def a_n (n : ℕ): ℤ := 11 - 2 * (n - 1)

theorem maximize_sum (n : ℕ) (S : ℕ → ℤ → Prop) :
  (∀ n, S n (a_n n)) → (a_n n ≥ 0) → n = 6 :=
by
  sorry

end maximize_sum_l136_136073


namespace mike_initial_cards_l136_136005

-- Define the conditions
def initial_cards (x : ℕ) := x + 13 = 100

-- Define the proof statement
theorem mike_initial_cards : initial_cards 87 :=
by
  sorry

end mike_initial_cards_l136_136005


namespace line_through_point_perpendicular_l136_136843

theorem line_through_point_perpendicular :
  ∃ (a b : ℝ), ∀ (x : ℝ), y = - (3 / 2) * x + 8 ∧ y - 2 = - (3 / 2) * (x - 4) ∧ 2*x - 3*y = 6 → y = - (3 / 2) * x + 8 :=
by 
  sorry

end line_through_point_perpendicular_l136_136843


namespace move_point_right_l136_136736

theorem move_point_right 
  (x y : ℤ)
  (h : (x, y) = (2, -1)) :
  (x + 3, y) = (5, -1) := 
by
  sorry

end move_point_right_l136_136736


namespace fish_remaining_l136_136628

def fish_caught_per_hour := 7
def hours_fished := 9
def fish_lost := 15

theorem fish_remaining : 
  (fish_caught_per_hour * hours_fished - fish_lost) = 48 :=
by
  sorry

end fish_remaining_l136_136628


namespace circle_reflection_l136_136079

-- Definition of the original center of the circle
def original_center : ℝ × ℝ := (8, -3)

-- Definition of the reflection transformation over the line y = x
def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Theorem stating that reflecting the original center over the line y = x results in a specific point
theorem circle_reflection : reflect original_center = (-3, 8) :=
  by
  -- skipping the proof part
  sorry

end circle_reflection_l136_136079


namespace evaluate_expression_l136_136269

theorem evaluate_expression :
  3 + 2*Real.sqrt 3 + 1/(3 + 2*Real.sqrt 3) + 1/(2*Real.sqrt 3 - 3) = 3 + (16 * Real.sqrt 3) / 3 :=
by
  sorry

end evaluate_expression_l136_136269


namespace max_profit_max_profit_price_l136_136062

-- Definitions based on the conditions
def purchase_price : ℝ := 80
def initial_selling_price : ℝ := 120
def initial_sales : ℕ := 20
def extra_sales_per_unit_decrease : ℕ := 2
def cost_price_constraint (x : ℝ) : Prop := 0 < x ∧ x ≤ 40

-- Expression for the profit function
def profit_function (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

-- Prove the maximum profit given the conditions
theorem max_profit : ∃ x : ℝ, cost_price_constraint x ∧ profit_function x = 1250 :=
by
  sorry

-- Proving that the selling price for max profit is 105 yuan
theorem max_profit_price : ∃ x : ℝ, cost_price_constraint x ∧ profit_function x = 1250 ∧ (initial_selling_price - x) = 105 :=
by
  sorry

end max_profit_max_profit_price_l136_136062


namespace B_necessary_not_sufficient_for_A_l136_136562

def A (x : ℝ) : Prop := 0 < x ∧ x < 5
def B (x : ℝ) : Prop := |x - 2| < 3

theorem B_necessary_not_sufficient_for_A (x : ℝ) :
  (A x → B x) ∧ (∃ x, B x ∧ ¬ A x) :=
by
  sorry

end B_necessary_not_sufficient_for_A_l136_136562


namespace original_group_men_l136_136645

-- Let's define the parameters of the problem
def original_days := 55
def absent_men := 15
def completed_days := 60

-- We need to show that the number of original men (x) is 180
theorem original_group_men (x : ℕ) (h : x * original_days = (x - absent_men) * completed_days) : x = 180 :=
by
  sorry

end original_group_men_l136_136645


namespace find_a_l136_136517

theorem find_a (a : ℝ) (h : a^2 + a^2 / 4 = 5) : a = 2 ∨ a = -2 := 
sorry

end find_a_l136_136517


namespace complex_modulus_l136_136763

open Complex

noncomputable def modulus_of_complex : ℂ :=
  (1 - 2 * Complex.I) * (1 - 2 * Complex.I) / Complex.I

theorem complex_modulus : Complex.abs modulus_of_complex = 5 :=
  sorry

end complex_modulus_l136_136763


namespace water_increase_factor_l136_136359

theorem water_increase_factor 
  (initial_koolaid : ℝ := 2) 
  (initial_water : ℝ := 16) 
  (evaporated_water : ℝ := 4) 
  (final_koolaid_percentage : ℝ := 4) : 
  (initial_water - evaporated_water) * (final_koolaid_percentage / 100) * initial_koolaid = 4 := 
by
  sorry

end water_increase_factor_l136_136359


namespace points_per_game_l136_136868

theorem points_per_game (total_points games : ℕ) (h1 : total_points = 91) (h2 : games = 13) :
  total_points / games = 7 :=
by
  sorry

end points_per_game_l136_136868


namespace break_room_capacity_l136_136788

theorem break_room_capacity :
  let people_per_table := 8
  let number_of_tables := 4
  people_per_table * number_of_tables = 32 :=
by
  let people_per_table := 8
  let number_of_tables := 4
  have h : people_per_table * number_of_tables = 32 := by sorry
  exact h

end break_room_capacity_l136_136788


namespace sum_of_numerator_and_denominator_of_decimal_0_345_l136_136435

def repeating_decimal_to_fraction_sum (x : ℚ) : ℕ :=
if h : x = 115 / 333 then 115 + 333 else 0

theorem sum_of_numerator_and_denominator_of_decimal_0_345 :
  repeating_decimal_to_fraction_sum 345 / 999 = 448 :=
by {
  -- Given: 0.\overline{345} = 345 / 999, simplified to 115 / 333
  -- hence the sum of numerator and denominator = 115 + 333
  -- We don't need the proof steps here, just conclude with the sum
  sorry }

end sum_of_numerator_and_denominator_of_decimal_0_345_l136_136435


namespace sqrt_solution_l136_136693

theorem sqrt_solution (x : ℝ) (h : x = Real.sqrt (1 + x)) : 1 < x ∧ x < 2 :=
by
  sorry

end sqrt_solution_l136_136693


namespace initial_pieces_of_fruit_l136_136857

-- Definitions for the given problem
def pieces_eaten_in_first_four_days : ℕ := 5
def pieces_kept_for_next_week : ℕ := 2
def pieces_brought_to_school : ℕ := 3

-- Problem statement
theorem initial_pieces_of_fruit 
  (pieces_eaten : ℕ)
  (pieces_kept : ℕ)
  (pieces_brought : ℕ)
  (h1 : pieces_eaten = pieces_eaten_in_first_four_days)
  (h2 : pieces_kept = pieces_kept_for_next_week)
  (h3 : pieces_brought = pieces_brought_to_school) :
  pieces_eaten + pieces_kept + pieces_brought = 10 := 
sorry

end initial_pieces_of_fruit_l136_136857


namespace average_sales_six_months_l136_136418

theorem average_sales_six_months :
  let sales1 := 4000
  let sales2 := 6524
  let sales3 := 5689
  let sales4 := 7230
  let sales5 := 6000
  let sales6 := 12557
  let total_sales_first_five := sales1 + sales2 + sales3 + sales4 + sales5
  let total_sales_six := total_sales_first_five + sales6
  let average_sales := total_sales_six / 6
  average_sales = 7000 :=
by
  let sales1 := 4000
  let sales2 := 6524
  let sales3 := 5689
  let sales4 := 7230
  let sales5 := 6000
  let sales6 := 12557
  let total_sales_first_five := sales1 + sales2 + sales3 + sales4 + sales5
  let total_sales_six := total_sales_first_five + sales6
  let average_sales := total_sales_six / 6
  have h : total_sales_first_five = 29443 := by sorry
  have h1 : total_sales_six = 42000 := by sorry
  have h2 : average_sales = 7000 := by sorry
  exact h2

end average_sales_six_months_l136_136418


namespace total_books_written_l136_136167

def books_written (Zig Flo : ℕ) : Prop :=
  (Zig = 60) ∧ (Zig = 4 * Flo) ∧ (Zig + Flo = 75)

theorem total_books_written (Zig Flo : ℕ) : books_written Zig Flo :=
  by
    sorry

end total_books_written_l136_136167


namespace ratio_of_green_to_blue_l136_136832

def balls (total blue red green yellow : ℕ) : Prop :=
  total = 36 ∧ blue = 6 ∧ red = 4 ∧ yellow = 2 * red ∧ green = total - (blue + red + yellow)

theorem ratio_of_green_to_blue (total blue red green yellow : ℕ) (h : balls total blue red green yellow) :
  (green / blue = 3) :=
by
  -- Unpack the conditions
  obtain ⟨total_eq, blue_eq, red_eq, yellow_eq, green_eq⟩ := h
  -- Simplify values based on the given conditions
  have blue_val := blue_eq
  have green_val := green_eq
  rw [blue_val, green_val]
  sorry

end ratio_of_green_to_blue_l136_136832


namespace slices_per_large_pizza_l136_136365

theorem slices_per_large_pizza (total_pizzas : ℕ) (slices_eaten : ℕ) (slices_remaining : ℕ) 
  (H1 : total_pizzas = 2) (H2 : slices_eaten = 7) (H3 : slices_remaining = 9) : 
  (slices_remaining + slices_eaten) / total_pizzas = 8 := 
by
  sorry

end slices_per_large_pizza_l136_136365


namespace max_triangles_convex_polygon_l136_136983

theorem max_triangles_convex_polygon (vertices : ℕ) (interior_points : ℕ) (total_points : ℕ) : 
  vertices = 13 ∧ interior_points = 200 ∧ total_points = 213 ∧ (∀ (x y z : ℕ), (x < total_points ∧ y < total_points ∧ z < total_points) → x ≠ y ∧ y ≠ z ∧ x ≠ z) →
  (∃ triangles : ℕ, triangles = 411) :=
by
  sorry

end max_triangles_convex_polygon_l136_136983


namespace train_cross_time_l136_136494

-- Define the conditions
def train_speed_kmhr := 52
def train_length_meters := 130

-- Conversion factor from km/hr to m/s
def kmhr_to_ms (speed_kmhr : ℕ) : ℕ := (speed_kmhr * 1000) / 3600

-- Speed of the train in m/s
def train_speed_ms := kmhr_to_ms train_speed_kmhr

-- Calculate time to cross the pole
def time_to_cross_pole (distance_m : ℕ) (speed_ms : ℕ) : ℕ := distance_m / speed_ms

-- The theorem to prove
theorem train_cross_time : time_to_cross_pole train_length_meters train_speed_ms = 9 := by sorry

end train_cross_time_l136_136494


namespace candies_per_person_l136_136716

theorem candies_per_person (a b people total_candies candies_per_person : ℕ)
  (h1: a = 17)
  (h2: b = 19)
  (h3: people = 9)
  (h4: total_candies = a + b)
  (h5: candies_per_person = total_candies / people) :
  candies_per_person = 4 :=
by sorry

end candies_per_person_l136_136716


namespace constant_term_in_expansion_l136_136754

theorem constant_term_in_expansion (x : ℂ) : 
  (2 - (3 / x)) * (x ^ 2 + 2 / x) ^ 5 = 0 := 
sorry

end constant_term_in_expansion_l136_136754


namespace inequality_proof_l136_136420

variable (a b c d : ℝ)

theorem inequality_proof (ha : 0 ≤ a ∧ a ≤ 1)
                       (hb : 0 ≤ b ∧ b ≤ 1)
                       (hc : 0 ≤ c ∧ c ≤ 1)
                       (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1) ^ 2 ≥ 4 * (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2) :=
sorry

end inequality_proof_l136_136420


namespace no_prize_for_A_l136_136301

variable (A B C D : Prop)

theorem no_prize_for_A 
  (hA : A → B) 
  (hB : B → C) 
  (hC : ¬D → ¬C) 
  (exactly_one_did_not_win : (¬A ∧ B ∧ C ∧ D) ∨ (A ∧ ¬B ∧ C ∧ D) ∨ (A ∧ B ∧ ¬C ∧ D) ∨ (A ∧ B ∧ C ∧ ¬D)) 
: ¬A := 
sorry

end no_prize_for_A_l136_136301


namespace scooped_water_amount_l136_136415

variables (x : ℝ)

def initial_water_amount : ℝ := 10
def total_amount : ℝ := initial_water_amount
def alcohol_concentration : ℝ := 0.75

theorem scooped_water_amount (h : x / total_amount = alcohol_concentration) : x = 7.5 :=
by sorry

end scooped_water_amount_l136_136415


namespace find_k_for_xy_solution_l136_136151

theorem find_k_for_xy_solution :
  ∀ (k : ℕ), (∃ (x y : ℕ), x * (x + k) = y * (y + 1))
  → k = 1 ∨ k ≥ 4 :=
by
  intros k h
  sorry -- proof goes here

end find_k_for_xy_solution_l136_136151


namespace solution_condition1_solution_condition2_solution_condition3_solution_condition4_l136_136010

-- Define the conditions
def Condition1 : Prop :=
  ∃ (total_population box1 box2 sampled : Nat),
  total_population = 30 ∧ box1 = 21 ∧ box2 = 9 ∧ sampled = 10

def Condition2 : Prop :=
  ∃ (total_population produced_by_A produced_by_B sampled : Nat),
  total_population = 30 ∧ produced_by_A = 21 ∧ produced_by_B = 9 ∧ sampled = 10

def Condition3 : Prop :=
  ∃ (total_population sampled : Nat),
  total_population = 300 ∧ sampled = 10

def Condition4 : Prop :=
  ∃ (total_population sampled : Nat),
  total_population = 300 ∧ sampled = 50

-- Define the appropriate methods
def LotteryMethod : Prop := ∃ method : String, method = "Lottery method"
def StratifiedSampling : Prop := ∃ method : String, method = "Stratified sampling"
def RandomNumberMethod : Prop := ∃ method : String, method = "Random number method"
def SystematicSampling : Prop := ∃ method : String, method = "Systematic sampling"

-- Statements to prove the appropriate methods for each condition
theorem solution_condition1 : Condition1 → LotteryMethod := by sorry
theorem solution_condition2 : Condition2 → StratifiedSampling := by sorry
theorem solution_condition3 : Condition3 → RandomNumberMethod := by sorry
theorem solution_condition4 : Condition4 → SystematicSampling := by sorry

end solution_condition1_solution_condition2_solution_condition3_solution_condition4_l136_136010


namespace ironed_clothing_count_l136_136884

theorem ironed_clothing_count : 
  (4 * 2 + 5 * 3) + (3 * 3 + 4 * 2) + (2 * 1 + 3 * 1) = 45 := by
  sorry

end ironed_clothing_count_l136_136884


namespace olivia_spent_amount_l136_136443

noncomputable def initial_amount : ℕ := 100
noncomputable def collected_amount : ℕ := 148
noncomputable def final_amount : ℕ := 159

theorem olivia_spent_amount :
  initial_amount + collected_amount - final_amount = 89 :=
by
  sorry

end olivia_spent_amount_l136_136443


namespace double_acute_angle_l136_136854

theorem double_acute_angle (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
by
  sorry

end double_acute_angle_l136_136854


namespace cost_per_pound_beef_is_correct_l136_136329

variable (budget initial_chicken_cost pounds_beef remaining_budget_after_purchase : ℝ)
variable (spending_on_beef cost_per_pound_beef : ℝ)

axiom h1 : budget = 80
axiom h2 : initial_chicken_cost = 12
axiom h3 : pounds_beef = 5
axiom h4 : remaining_budget_after_purchase = 53
axiom h5 : spending_on_beef = budget - initial_chicken_cost - remaining_budget_after_purchase
axiom h6 : cost_per_pound_beef = spending_on_beef / pounds_beef

theorem cost_per_pound_beef_is_correct : cost_per_pound_beef = 3 :=
by
  sorry

end cost_per_pound_beef_is_correct_l136_136329


namespace philip_farm_animal_count_l136_136279

def number_of_cows : ℕ := 20

def number_of_ducks : ℕ := number_of_cows * 3 / 2

def total_cows_and_ducks : ℕ := number_of_cows + number_of_ducks

def number_of_pigs : ℕ := total_cows_and_ducks / 5

def total_animals : ℕ := total_cows_and_ducks + number_of_pigs

theorem philip_farm_animal_count : total_animals = 60 := by
  sorry

end philip_farm_animal_count_l136_136279


namespace max_abs_value_l136_136925

open Complex Real

theorem max_abs_value (z : ℂ) (h : abs (z - 8) + abs (z + 6 * I) = 10) : abs z ≤ 8 :=
sorry

example : ∃ z : ℂ, abs (z - 8) + abs (z + 6 * I) = 10 ∧ abs z = 8 :=
sorry

end max_abs_value_l136_136925


namespace sum_of_coefficients_is_zero_l136_136305

noncomputable def expansion : Polynomial ℚ := (Polynomial.X^2 + Polynomial.X + 1) * (2*Polynomial.X - 2)^5

theorem sum_of_coefficients_is_zero :
  (expansion.coeff 0) + (expansion.coeff 1) + (expansion.coeff 2) + (expansion.coeff 3) + 
  (expansion.coeff 4) + (expansion.coeff 5) + (expansion.coeff 6) + (expansion.coeff 7) = 0 :=
by
  sorry

end sum_of_coefficients_is_zero_l136_136305


namespace positive_integer_divisibility_l136_136668

theorem positive_integer_divisibility (n : ℕ) (h_pos : n > 0) (h_div : (n^2 + 1) ∣ (n + 1)) : n = 1 := 
sorry

end positive_integer_divisibility_l136_136668


namespace angelfish_goldfish_difference_l136_136215

-- Given statements
variables {A G : ℕ}
def goldfish := 8
def total_fish := 44

-- Conditions
axiom twice_as_many_guppies : G = 2 * A
axiom total_fish_condition : A + G + goldfish = total_fish

-- Theorem
theorem angelfish_goldfish_difference : A - goldfish = 4 :=
by
  sorry

end angelfish_goldfish_difference_l136_136215


namespace gcd_of_repeated_three_digit_number_is_constant_l136_136225

theorem gcd_of_repeated_three_digit_number_is_constant (m : ℕ) (h1 : 100 ≤ m) (h2 : m < 1000) : 
  ∃ d, d = 1001001 ∧ ∀ n, n = 10010013 * m → (gcd 1001001 n) = 1001001 :=
by
  -- The proof would go here
  sorry

end gcd_of_repeated_three_digit_number_is_constant_l136_136225


namespace volume_of_right_prism_with_trapezoid_base_l136_136663

variable (S1 S2 H a b h: ℝ)

theorem volume_of_right_prism_with_trapezoid_base 
  (hS1 : S1 = a * H) 
  (hS2 : S2 = b * H) 
  (h_trapezoid : a ≠ b) : 
  1 / 2 * (S1 + S2) * h = (1 / 2 * (a + b) * h) * H :=
by 
  sorry

end volume_of_right_prism_with_trapezoid_base_l136_136663


namespace intersection_is_correct_l136_136002

noncomputable def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {x : ℝ | x < 4}

theorem intersection_is_correct : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 1} :=
sorry

end intersection_is_correct_l136_136002


namespace height_of_bottom_step_l136_136397

variable (h l w : ℝ)

theorem height_of_bottom_step
  (h l w : ℝ)
  (eq1 : l + h - w / 2 = 42)
  (eq2 : 2 * l + h = 38)
  (w_value : w = 4) : h = 34 := by
sorry

end height_of_bottom_step_l136_136397


namespace product_of_four_consecutive_even_numbers_divisible_by_240_l136_136242

theorem product_of_four_consecutive_even_numbers_divisible_by_240 :
  ∀ (n : ℤ), (n % 2 = 0) →
    (n + 2) % 2 = 0 →
    (n + 4) % 2 = 0 →
    (n + 6) % 2 = 0 →
    ((n * (n + 2) * (n + 4) * (n + 6)) % 240 = 0) :=
by
  intro n hn hnp2 hnp4 hnp6
  sorry

end product_of_four_consecutive_even_numbers_divisible_by_240_l136_136242


namespace david_pushups_more_than_zachary_l136_136039

theorem david_pushups_more_than_zachary :
  ∀ (Z D J : ℕ), Z = 51 → J = 69 → J = D - 4 → D = Z + 22 :=
by
  intros Z D J hZ hJ hJD
  sorry

end david_pushups_more_than_zachary_l136_136039


namespace no_solution_inequality_l136_136349

theorem no_solution_inequality (m : ℝ) : ¬(∃ x : ℝ, 2 * x - 1 > 1 ∧ x < m) → m ≤ 1 :=
by
  intro h
  sorry

end no_solution_inequality_l136_136349


namespace floor_neg_7_over_4_l136_136501

theorem floor_neg_7_over_4 : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_7_over_4_l136_136501


namespace oranges_to_put_back_l136_136133

theorem oranges_to_put_back
  (p_A p_O : ℕ)
  (A O : ℕ)
  (total_fruits : ℕ)
  (initial_avg_price new_avg_price : ℕ)
  (x : ℕ)
  (h1 : p_A = 40)
  (h2 : p_O = 60)
  (h3 : total_fruits = 15)
  (h4 : initial_avg_price = 48)
  (h5 : new_avg_price = 45)
  (h6 : A + O = total_fruits)
  (h7 : (p_A * A + p_O * O) / total_fruits = initial_avg_price)
  (h8 : (720 - 60 * x) / (15 - x) = 45) :
  x = 3 :=
by
  sorry

end oranges_to_put_back_l136_136133


namespace net_salary_change_l136_136378

variable (S : ℝ)

theorem net_salary_change (h1 : S > 0) : 
  (1.3 * S - 0.3 * (1.3 * S)) - S = -0.09 * S := by
  sorry

end net_salary_change_l136_136378


namespace best_representation_is_B_l136_136348

-- Define the conditions
structure Trip :=
  (home_to_diner : ℝ)
  (diner_stop : ℝ)
  (diner_to_highway : ℝ)
  (highway_to_mall : ℝ)
  (mall_stop : ℝ)
  (highway_return : ℝ)
  (construction_zone : ℝ)
  (return_city_traffic : ℝ)

-- Graph description
inductive Graph
| plateau : Graph
| increasing : Graph → Graph
| decreasing : Graph → Graph

-- Condition that describes the pattern of the graph
def correct_graph (trip : Trip) : Prop :=
  let d1 := trip.home_to_diner
  let d2 := trip.diner_stop
  let d3 := trip.diner_to_highway
  let d4 := trip.highway_to_mall
  let d5 := trip.mall_stop
  let d6 := trip.highway_return
  let d7 := trip.construction_zone
  let d8 := trip.return_city_traffic
  d1 > 0 ∧ d2 = 0 ∧ d3 > 0 ∧ d4 > 0 ∧ d5 = 0 ∧ d6 < 0 ∧ d7 < 0 ∧ d8 < 0

-- Theorem statement
theorem best_representation_is_B (trip : Trip) : correct_graph trip :=
by sorry

end best_representation_is_B_l136_136348


namespace painting_time_l136_136229

variable (a d e : ℕ)

theorem painting_time (h : a * e * d = a * d * e) : (d * x = a^2 * e) := 
by
   sorry

end painting_time_l136_136229


namespace correct_factorization_l136_136974

theorem correct_factorization (a b : ℝ) : a^2 - 4 * a * b + 4 * b^2 = (a - 2 * b)^2 :=
by sorry

end correct_factorization_l136_136974


namespace parabola_ordinate_l136_136845

theorem parabola_ordinate (x y : ℝ) (h : y = 2 * x^2) (d : dist (x, y) (0, 1 / 8) = 9 / 8) : y = 1 := 
sorry

end parabola_ordinate_l136_136845


namespace michael_weight_loss_in_may_l136_136394

-- Defining the conditions
def weight_loss_goal : ℕ := 10
def weight_loss_march : ℕ := 3
def weight_loss_april : ℕ := 4

-- Statement of the problem to prove
theorem michael_weight_loss_in_may (weight_loss_goal weight_loss_march weight_loss_april : ℕ) :
  weight_loss_goal - (weight_loss_march + weight_loss_april) = 3 :=
by
  sorry

end michael_weight_loss_in_may_l136_136394


namespace expand_simplify_correct_l136_136998

noncomputable def expand_and_simplify (x : ℕ) : ℕ :=
  (x + 4) * (x - 9)

theorem expand_simplify_correct (x : ℕ) : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by
  sorry

end expand_simplify_correct_l136_136998


namespace shortest_fence_length_l136_136115

open Real

noncomputable def area_of_garden (length width : ℝ) : ℝ := length * width

theorem shortest_fence_length (length width : ℝ) (h : area_of_garden length width = 64) :
  4 * sqrt 64 = 32 :=
by
  -- The statement sets up the condition that the area is 64 and asks to prove minimum perimeter (fence length = perimeter).
  sorry

end shortest_fence_length_l136_136115


namespace find_b_value_l136_136162

theorem find_b_value :
  ∃ b : ℕ, 70 = (2 * (b + 1)^2 + 3 * (b + 1) + 4) - (2 * (b - 1)^2 + 3 * (b - 1) + 4) ∧ b > 0 ∧ b < 1000 :=
by
  sorry

end find_b_value_l136_136162


namespace fraction_of_money_left_l136_136003

theorem fraction_of_money_left 
  (m c : ℝ) 
  (h1 : (1/4 : ℝ) * m = (1/2) * c) : 
  (m - c) / m = (1/2 : ℝ) :=
by
  -- the proof will be written here
  sorry

end fraction_of_money_left_l136_136003


namespace units_digit_17_pow_53_l136_136080

theorem units_digit_17_pow_53 : (17^53) % 10 = 7 := 
by sorry

end units_digit_17_pow_53_l136_136080


namespace part1_part2_l136_136535

def P : Set ℝ := {x | x ≥ 1 / 2 ∧ x ≤ 2}

def Q (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x + 2 > 0}

def R (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x + 2 = 0}

theorem part1 (a : ℝ) : (∃ x, x ∈ P ∧ x ∈ Q a) → a > -1 / 2 :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x, x ∈ P ∧ x ∈ R a) → a ≥ -1 / 2 ∧ a ≤ 1 / 2 :=
by
  sorry

end part1_part2_l136_136535


namespace age_of_15th_student_l136_136306

theorem age_of_15th_student (T : ℕ) (T8 : ℕ) (T6 : ℕ)
  (avg_15_students : T / 15 = 15)
  (avg_8_students : T8 / 8 = 14)
  (avg_6_students : T6 / 6 = 16) :
  (T - (T8 + T6)) = 17 := by
  sorry

end age_of_15th_student_l136_136306


namespace Li_age_is_12_l136_136158

-- Given conditions:
def Zhang_twice_Li (Li: ℕ) : ℕ := 2 * Li
def Jung_older_Zhang (Zhang: ℕ) : ℕ := Zhang + 2
def Jung_age := 26

-- Proof problem:
theorem Li_age_is_12 : ∃ Li: ℕ, Jung_older_Zhang (Zhang_twice_Li Li) = Jung_age ∧ Li = 12 :=
by
  sorry

end Li_age_is_12_l136_136158


namespace chloe_probability_l136_136015

theorem chloe_probability :
  let total_numbers := 60
  let multiples_of_4 := 15
  let non_multiples_of_4_prob := 3 / 4
  let neither_multiple_of_4_prob := (non_multiples_of_4_prob) ^ 2
  let at_least_one_multiple_of_4_prob := 1 - neither_multiple_of_4_prob
  at_least_one_multiple_of_4_prob = 7 / 16 := by
  sorry

end chloe_probability_l136_136015


namespace flatville_additional_plates_max_count_l136_136132

noncomputable def flatville_initial_plate_count : Nat :=
  6 * 4 * 5

noncomputable def flatville_max_plate_count : Nat :=
  6 * 6 * 6

theorem flatville_additional_plates_max_count : flatville_max_plate_count - flatville_initial_plate_count = 96 :=
by
  sorry

end flatville_additional_plates_max_count_l136_136132


namespace difference_in_square_sides_square_side_length_square_area_greater_than_rectangle_l136_136196

-- Exploration 1
theorem difference_in_square_sides (a b : ℝ) (h1 : a + b = 20) (h2 : a^2 - b^2 = 40) : a - b = 2 :=
by sorry

-- Exploration 2
theorem square_side_length (x y : ℝ) : (2 * x + 2 * y) / 4 = (x + y) / 2 :=
by sorry

theorem square_area_greater_than_rectangle (x y : ℝ) (h : x > y) : ( (x + y) / 2 ) ^ 2 > x * y :=
by sorry

end difference_in_square_sides_square_side_length_square_area_greater_than_rectangle_l136_136196


namespace total_marbles_left_is_correct_l136_136837

def marbles_left_after_removal : ℕ :=
  let red_initial := 80
  let blue_initial := 120
  let green_initial := 75
  let yellow_initial := 50
  let red_removed := red_initial / 4
  let blue_removed := 3 * (green_initial / 5)
  let green_removed := (green_initial * 3) / 10
  let yellow_removed := 25
  let red_left := red_initial - red_removed
  let blue_left := blue_initial - blue_removed
  let green_left := green_initial - green_removed
  let yellow_left := yellow_initial - yellow_removed
  red_left + blue_left + green_left + yellow_left

theorem total_marbles_left_is_correct :
  marbles_left_after_removal = 213 :=
  by
    sorry

end total_marbles_left_is_correct_l136_136837


namespace Kyle_age_l136_136260

-- Let's define the variables for each person's age.
variables (Shelley Kyle Julian Frederick Tyson Casey Sandra David Fiona : ℕ) 

-- Defining conditions based on given problem.
axiom condition1 : Shelley = Kyle - 3
axiom condition2 : Shelley = Julian + 4
axiom condition3 : Julian = Frederick - 20
axiom condition4 : Julian = Fiona + 5
axiom condition5 : Frederick = 2 * Tyson
axiom condition6 : Tyson = 2 * Casey
axiom condition7 : Casey = Fiona - 2
axiom condition8 : Casey = Sandra / 2
axiom condition9 : Sandra = David + 4
axiom condition10 : David = 16

-- The goal is to prove Kyle's age is 23 years old.
theorem Kyle_age : Kyle = 23 :=
by sorry

end Kyle_age_l136_136260


namespace collinear_values_k_l136_136255

/-- Define the vectors OA, OB, and OC using the given conditions. -/
def vectorOA (k : ℝ) : ℝ × ℝ := (k, 12)
def vectorOB : ℝ × ℝ := (4, 5)
def vectorOC (k : ℝ) : ℝ × ℝ := (10, k)

/-- Define vectors AB and BC using vector subtraction. -/
def vectorAB (k : ℝ) : ℝ × ℝ := (4 - k, -7)
def vectorBC (k : ℝ) : ℝ × ℝ := (6, k - 5)

/-- Collinearity condition for vectors AB and BC. -/
def collinear (k : ℝ) : Prop :=
  (4 - k) * (k - 5) + 42 = 0

/-- Prove that the value of k is 11 or -2 given the collinearity condition. -/
theorem collinear_values_k : ∀ k : ℝ, collinear k → (k = 11 ∨ k = -2) :=
by
  intros k h
  sorry

end collinear_values_k_l136_136255


namespace green_balls_in_bag_l136_136090

theorem green_balls_in_bag (b : ℕ) (P_blue : ℚ) (g : ℕ) (h1 : b = 8) (h2 : P_blue = 1 / 3) (h3 : P_blue = (b : ℚ) / (b + g)) :
  g = 16 :=
by
  sorry

end green_balls_in_bag_l136_136090


namespace union_complement_l136_136842

-- Definitions of the sets
def U : Set ℕ := {x | x > 0 ∧ x ≤ 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 2, 4}

-- Statement of the proof problem
theorem union_complement : A ∪ (U \ B) = {1, 3, 5} := by
  sorry

end union_complement_l136_136842


namespace bus_remaining_distance_l136_136880

noncomputable def final_distance (z x : ℝ) : ℝ :=
  z - (z * x / 5)

theorem bus_remaining_distance (z : ℝ) :
  (z / 2) / (z - 19.2) = x ∧ (z - 12) / (z / 2) = x → final_distance z x = 6.4 :=
by
  intro h
  sorry

end bus_remaining_distance_l136_136880


namespace calculate_ratio_l136_136170

variables (M Q P N R : ℝ)

-- Definitions of conditions
def M_def : M = 0.40 * Q := by sorry
def Q_def : Q = 0.30 * P := by sorry
def N_def : N = 0.60 * P := by sorry
def R_def : R = 0.20 * P := by sorry

-- Statement of the proof problem
theorem calculate_ratio (hM : M = 0.40 * Q) (hQ : Q = 0.30 * P)
  (hN : N = 0.60 * P) (hR : R = 0.20 * P) : 
  (M + R) / N = 8 / 15 := by
  sorry

end calculate_ratio_l136_136170


namespace calculate_purple_pants_l136_136735

def total_shirts : ℕ := 5
def total_pants : ℕ := 24
def plaid_shirts : ℕ := 3
def non_plaid_non_purple_items : ℕ := 21

theorem calculate_purple_pants : total_pants - (non_plaid_non_purple_items - (total_shirts - plaid_shirts)) = 5 :=
by 
  sorry

end calculate_purple_pants_l136_136735


namespace esther_walks_975_yards_l136_136360

def miles_to_feet (miles : ℕ) : ℕ := miles * 5280
def feet_to_yards (feet : ℕ) : ℕ := feet / 3

variable (lionel_miles : ℕ) (niklaus_feet : ℕ) (total_feet : ℕ) (esther_yards : ℕ)
variable (h_lionel : lionel_miles = 4)
variable (h_niklaus : niklaus_feet = 1287)
variable (h_total : total_feet = 25332)
variable (h_esther : esther_yards = 975)

theorem esther_walks_975_yards :
  let lionel_distance_in_feet := miles_to_feet lionel_miles
  let combined_distance := lionel_distance_in_feet + niklaus_feet
  let esther_distance_in_feet := total_feet - combined_distance
  feet_to_yards esther_distance_in_feet = esther_yards := by {
    sorry
  }

end esther_walks_975_yards_l136_136360


namespace fraction_decomposition_l136_136705

noncomputable def p (n : ℕ) : ℚ :=
  (n + 1) / 2

noncomputable def q (n : ℕ) : ℚ :=
  n * p n

theorem fraction_decomposition (n : ℕ) (h : ∃ k : ℕ, n = 5 + 2*k) :
  (2 / n : ℚ) = (1 / p n) + (1 / q n) :=
by
  sorry

end fraction_decomposition_l136_136705


namespace negation_of_existential_proposition_l136_136118

theorem negation_of_existential_proposition :
  (¬ (∃ x : ℝ, x > Real.sin x)) ↔ (∀ x : ℝ, x ≤ Real.sin x) :=
by 
  sorry

end negation_of_existential_proposition_l136_136118


namespace Steven_has_more_peaches_l136_136936

variable (Steven_peaches : Nat) (Jill_peaches : Nat)
variable (h1 : Steven_peaches = 19) (h2 : Jill_peaches = 6)

theorem Steven_has_more_peaches : Steven_peaches - Jill_peaches = 13 :=
by
  sorry

end Steven_has_more_peaches_l136_136936


namespace percentage_supports_policy_l136_136768

theorem percentage_supports_policy (men women : ℕ) (men_favor women_favor : ℝ) (total_population : ℕ) (total_supporters : ℕ) (percentage_supporters : ℝ)
  (h1 : men = 200) 
  (h2 : women = 800)
  (h3 : men_favor = 0.70)
  (h4 : women_favor = 0.75)
  (h5 : total_population = men + women)
  (h6 : total_supporters = (men_favor * men) + (women_favor * women))
  (h7 : percentage_supporters = (total_supporters / total_population) * 100) :
  percentage_supporters = 74 := 
by
  sorry

end percentage_supports_policy_l136_136768


namespace harly_dogs_final_count_l136_136482

theorem harly_dogs_final_count (initial_dogs : ℕ) (adopted_percentage : ℕ) (returned_dogs : ℕ) (adoption_rate : adopted_percentage = 40) (initial_count : initial_dogs = 80) (returned_count : returned_dogs = 5) :
  initial_dogs - (initial_dogs * adopted_percentage / 100) + returned_dogs = 53 :=
by
  sorry

end harly_dogs_final_count_l136_136482


namespace rotation_problem_l136_136122

theorem rotation_problem (y : ℝ) (hy : y < 360) :
  (450 % 360 == 90) ∧ (y == 360 - 90) ∧ (90 + (360 - y) % 360 == 0) → y == 270 :=
by {
  -- Proof steps go here
  sorry
}

end rotation_problem_l136_136122


namespace possible_even_and_odd_functions_l136_136249

def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem possible_even_and_odd_functions :
  ∃ p q : ℝ → ℝ, is_even_function p ∧ is_odd_function (p ∘ q) ∧ (¬(∀ x, p (q x) = 0)) :=
by
  sorry

end possible_even_and_odd_functions_l136_136249


namespace race_distance_l136_136315

variables (a b c d : ℝ)
variables (h1 : d / a = (d - 30) / b)
variables (h2 : d / b = (d - 15) / c)
variables (h3 : d / a = (d - 40) / c)

theorem race_distance : d = 90 :=
by 
  sorry

end race_distance_l136_136315


namespace intersection_of_sets_l136_136633

theorem intersection_of_sets :
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | x < 0 ∨ x > 3}
  ∀ x, (x ∈ A ∧ x ∈ B) ↔ (-2 < x ∧ x < 0) :=
by
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | x < 0 ∨ x > 3}
  intro x
  sorry

end intersection_of_sets_l136_136633


namespace circles_intersect_l136_136297

def circle1 := { x : ℝ × ℝ | (x.1 - 1)^2 + (x.2 + 2)^2 = 1 }
def circle2 := { x : ℝ × ℝ | (x.1 - 2)^2 + (x.2 + 1)^2 = 1 / 4 }

theorem circles_intersect :
  ∃ x : ℝ × ℝ, x ∈ circle1 ∧ x ∈ circle2 :=
sorry

end circles_intersect_l136_136297


namespace mangoes_rate_l136_136875

theorem mangoes_rate (grapes_weight mangoes_weight total_amount grapes_rate mango_rate : ℕ)
  (h1 : grapes_weight = 7)
  (h2 : grapes_rate = 68)
  (h3 : total_amount = 908)
  (h4 : mangoes_weight = 9)
  (h5 : total_amount - grapes_weight * grapes_rate = mangoes_weight * mango_rate) :
  mango_rate = 48 :=
by
  sorry

end mangoes_rate_l136_136875


namespace fraction_red_marbles_l136_136614

theorem fraction_red_marbles (x : ℕ) (h : x > 0) :
  let blue := (2/3 : ℚ) * x
  let red := (1/3 : ℚ) * x
  let new_red := 3 * red
  let new_total := blue + new_red
  new_red / new_total = (3/5 : ℚ) := by
  sorry

end fraction_red_marbles_l136_136614


namespace range_of_m_l136_136451

noncomputable def condition_p (x : ℝ) : Prop := -2 < x ∧ x < 10
noncomputable def condition_q (x m : ℝ) : Prop := (x - 1)^2 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) :
  (∀ x, condition_p x → condition_q x m) ∧ (∃ x, ¬ condition_p x ∧ condition_q x m) ↔ 9 ≤ m := sorry

end range_of_m_l136_136451


namespace B_subset_A_iff_l136_136679

namespace MathProofs

def A (x : ℝ) : Prop := -2 < x ∧ x < 5

def B (x : ℝ) (m : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem B_subset_A_iff (m : ℝ) :
  (∀ x : ℝ, B x m → A x) ↔ m < 3 :=
by
  sorry

end MathProofs

end B_subset_A_iff_l136_136679


namespace correct_statement_is_D_l136_136066

/-
Given the following statements and their conditions:
A: Conducting a comprehensive survey is not an accurate approach to understand the sleep situation of middle school students in Changsha.
B: The mode of the dataset \(-1\), \(2\), \(5\), \(5\), \(7\), \(7\), \(4\) is not \(7\) only, because both \(5\) and \(7\) are modes.
C: A probability of precipitation of \(90\%\) does not guarantee it will rain tomorrow.
D: If two datasets, A and B, have the same mean, and the variances \(s_{A}^{2} = 0.3\) and \(s_{B}^{2} = 0.02\), then set B with a lower variance \(s_{B}^{2}\) is more stable.

Prove that the correct statement based on these conditions is D.
-/
theorem correct_statement_is_D
  (dataset_A dataset_B : Type)
  (mean_A mean_B : ℝ)
  (sA2 sB2 : ℝ)
  (h_same_mean: mean_A = mean_B)
  (h_variances: sA2 = 0.3 ∧ sB2 = 0.02)
  (h_stability: sA2 > sB2) :
  (if sA2 = 0.3 ∧ sB2 = 0.02 ∧ sA2 > sB2 then "D" else "not D") = "D" := by
  sorry

end correct_statement_is_D_l136_136066


namespace IntersectionOfAandB_l136_136480

def setA : Set ℝ := {x | x < 5}
def setB : Set ℝ := {x | -1 < x}

theorem IntersectionOfAandB : setA ∩ setB = {x | -1 < x ∧ x < 5} :=
sorry

end IntersectionOfAandB_l136_136480


namespace larger_triangle_perimeter_is_126_l136_136391

noncomputable def smaller_triangle_side1 : ℝ := 12
noncomputable def smaller_triangle_side2 : ℝ := 12
noncomputable def smaller_triangle_base : ℝ := 18
noncomputable def larger_triangle_longest_side : ℝ := 54
noncomputable def similarity_ratio : ℝ := larger_triangle_longest_side / smaller_triangle_base
noncomputable def larger_triangle_side1 : ℝ := smaller_triangle_side1 * similarity_ratio
noncomputable def larger_triangle_side2 : ℝ := smaller_triangle_side2 * similarity_ratio
noncomputable def larger_triangle_perimeter : ℝ := larger_triangle_side1 + larger_triangle_side2 + larger_triangle_longest_side

theorem larger_triangle_perimeter_is_126 :
  larger_triangle_perimeter = 126 := by
  sorry

end larger_triangle_perimeter_is_126_l136_136391


namespace max_profit_l136_136585

noncomputable def revenue (x : ℝ) : ℝ := 
  if (0 < x ∧ x ≤ 10) then 13.5 - (1 / 30) * x^2 
  else if (x > 10) then (168 / x) - (2000 / (3 * x^2)) 
  else 0

noncomputable def cost (x : ℝ) : ℝ := 
  20 + 5.4 * x

noncomputable def profit (x : ℝ) : ℝ := revenue x * x - cost x

theorem max_profit : 
  ∃ (x : ℝ), 0 < x ∧ x ≤ 10 ∧ (profit x = 8.1 * x - (1 / 30) * x^3 - 20) ∧ 
    (∀ (y : ℝ), 0 < y ∧ y ≤ 10 → profit y ≤ profit 9) ∧ 
    ∀ (z : ℝ), z > 10 → profit z ≤ profit 9 :=
by
  sorry

end max_profit_l136_136585


namespace range_of_b_l136_136502

noncomputable def f (x b : ℝ) : ℝ := -1/2 * (x - 2)^2 + b * Real.log (x + 2)
noncomputable def derivative (x b : ℝ) := -(x - 2) + b / (x + 2)

-- Lean theorem statement
theorem range_of_b (b : ℝ) :
  (∀ x > 1, derivative x b ≤ 0) → b ≤ -3 :=
by
  sorry

end range_of_b_l136_136502


namespace maximum_value_product_cube_expression_l136_136986

theorem maximum_value_product_cube_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^3 - x * y^2 + y^3) * (x^3 - x^2 * z + z^3) * (y^3 - y^2 * z + z^3) ≤ 1 :=
sorry

end maximum_value_product_cube_expression_l136_136986


namespace natalie_bushes_to_zucchinis_l136_136056

/-- Each of Natalie's blueberry bushes yields ten containers of blueberries,
    and she trades six containers of blueberries for three zucchinis.
    Given this setup, prove that the number of bushes Natalie needs to pick
    in order to get sixty zucchinis is twelve. --/
theorem natalie_bushes_to_zucchinis :
  (∀ (bush_yield containers_needed : ℕ), bush_yield = 10 ∧ containers_needed = 60 * (6 / 3)) →
  (∀ (containers_total bushes_needed : ℕ), containers_total = 60 * (6 / 3) ∧ bushes_needed = containers_total * (1 / bush_yield)) →
  bushes_needed = 12 :=
by
  sorry

end natalie_bushes_to_zucchinis_l136_136056


namespace trapezoid_area_l136_136944

theorem trapezoid_area (l : ℝ) (r : ℝ) (a b : ℝ) (h : ℝ) (A : ℝ) :
  l = 9 →
  r = 4 →
  a + b = l + l →
  h = 2 * r →
  (a + b) / 2 * h = A →
  A = 72 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end trapezoid_area_l136_136944


namespace increasing_quadratic_l136_136247

noncomputable def f (a x : ℝ) : ℝ := 3 * x^2 - a * x + 4

theorem increasing_quadratic {a : ℝ} :
  (∀ x ≥ -5, 6 * x - a ≥ 0) ↔ a ≤ -30 :=
by
  sorry

end increasing_quadratic_l136_136247


namespace range_of_a_l136_136701

-- Define the function f
def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 2

-- Proposition P: f(x) has a root in the interval [-1, 1]
def P (a : ℝ) : Prop := ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f a x = 0

-- Proposition Q: There is only one real number x satisfying the inequality
def Q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

-- The theorem stating the range of a if either P or Q is false
theorem range_of_a (a : ℝ) : ¬(P a) ∨ ¬(Q a) → (a > -1 ∧ a < 0) ∨ (a > 0 ∧ a < 1) :=
sorry

end range_of_a_l136_136701


namespace father_has_4_chocolate_bars_left_l136_136825

noncomputable def chocolate_bars_given_to_father (initial_bars : ℕ) (num_people : ℕ) : ℕ :=
  let bars_per_person := initial_bars / num_people
  let bars_given := num_people * (bars_per_person / 2)
  bars_given

noncomputable def chocolate_bars_left_with_father (bars_given : ℕ) (bars_given_away : ℕ) : ℕ :=
  bars_given - bars_given_away

theorem father_has_4_chocolate_bars_left :
  ∀ (initial_bars num_people bars_given_away : ℕ), 
  initial_bars = 40 →
  num_people = 7 →
  bars_given_away = 10 →
  chocolate_bars_left_with_father (chocolate_bars_given_to_father initial_bars num_people) bars_given_away = 4 :=
by
  intros initial_bars num_people bars_given_away h_initial h_num h_given_away
  unfold chocolate_bars_given_to_father chocolate_bars_left_with_father
  rw [h_initial, h_num, h_given_away]
  exact sorry

end father_has_4_chocolate_bars_left_l136_136825


namespace parabola_c_value_l136_136616

theorem parabola_c_value (b c : ℝ) 
  (h1 : 6 = 2^2 + 2 * b + c) 
  (h2 : 20 = 4^2 + 4 * b + c) : 
  c = 0 :=
by {
  -- We state that we're skipping the proof
  sorry
}

end parabola_c_value_l136_136616


namespace gain_percent_l136_136357

variables (MP CP SP : ℝ)

-- problem conditions
axiom h1 : CP = 0.64 * MP
axiom h2 : SP = 0.84 * MP

-- To prove: Gain percent is 31.25%
theorem gain_percent (CP MP SP : ℝ) (h1 : CP = 0.64 * MP) (h2 : SP = 0.84 * MP) :
  ((SP - CP) / CP) * 100 = 31.25 :=
by sorry

end gain_percent_l136_136357


namespace sum_difference_l136_136652

noncomputable def sum_arith_seq (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_difference :
  let S_even := sum_arith_seq 2 2 1001
  let S_odd := sum_arith_seq 1 2 1002
  S_odd - S_even = 1002 :=
by
  sorry

end sum_difference_l136_136652


namespace sum_of_products_of_two_at_a_time_l136_136570

theorem sum_of_products_of_two_at_a_time (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : a + b + c = 21) : 
  a * b + b * c + a * c = 100 := 
  sorry

end sum_of_products_of_two_at_a_time_l136_136570


namespace train_speed_km_per_hr_l136_136375

-- Definitions for the conditions
def length_of_train_meters : ℕ := 250
def time_to_cross_pole_seconds : ℕ := 10

-- Conversion factors
def meters_to_kilometers (m : ℕ) : ℚ := m / 1000
def seconds_to_hours (s : ℕ) : ℚ := s / 3600

-- Theorem stating that the speed of the train is 90 km/hr
theorem train_speed_km_per_hr : 
  meters_to_kilometers length_of_train_meters / seconds_to_hours time_to_cross_pole_seconds = 90 := 
by 
  -- We skip the actual proof with sorry
  sorry

end train_speed_km_per_hr_l136_136375


namespace max_odd_integers_l136_136522

theorem max_odd_integers (a b c d e f : ℕ) 
  (hprod : a * b * c * d * e * f % 2 = 0) 
  (hpos_a : 0 < a) (hpos_b : 0 < b) 
  (hpos_c : 0 < c) (hpos_d : 0 < d) 
  (hpos_e : 0 < e) (hpos_f : 0 < f) : 
  ∃ x : ℕ, x ≤ 5 ∧ x = 5 :=
by sorry

end max_odd_integers_l136_136522


namespace james_ali_difference_l136_136472

theorem james_ali_difference (J A T : ℝ) (h1 : J = 145) (h2 : T = 250) (h3 : J + A = T) :
  J - A = 40 :=
by
  sorry

end james_ali_difference_l136_136472


namespace girls_ratio_correct_l136_136462

-- Define the number of total attendees
def total_attendees : ℕ := 100

-- Define the percentage of faculty and staff
def faculty_staff_percentage : ℕ := 10

-- Define the number of boys among the students
def number_of_boys : ℕ := 30

-- Define the function to calculate the number of faculty and staff
def faculty_staff (total_attendees faculty_staff_percentage: ℕ) : ℕ :=
  (faculty_staff_percentage * total_attendees) / 100

-- Define the function to calculate the number of students
def number_of_students (total_attendees faculty_staff_percentage: ℕ) : ℕ :=
  total_attendees - faculty_staff total_attendees faculty_staff_percentage

-- Define the function to calculate the number of girls
def number_of_girls (total_attendees faculty_staff_percentage number_of_boys: ℕ) : ℕ :=
  number_of_students total_attendees faculty_staff_percentage - number_of_boys

-- Define the function to calculate the ratio of girls to the remaining attendees
def ratio_girls_to_attendees (total_attendees faculty_staff_percentage number_of_boys: ℕ) : ℚ :=
  (number_of_girls total_attendees faculty_staff_percentage number_of_boys) / 
  (number_of_students total_attendees faculty_staff_percentage)

-- The theorem statement that needs to be proven (no proof required)
theorem girls_ratio_correct : ratio_girls_to_attendees total_attendees faculty_staff_percentage number_of_boys = 2 / 3 := 
by 
  -- The proof is skipped.
  sorry

end girls_ratio_correct_l136_136462


namespace percent_other_sales_l136_136620

-- Define the given conditions
def s_brushes : ℝ := 0.45
def s_paints : ℝ := 0.28

-- Define the proof goal in Lean
theorem percent_other_sales :
  1 - (s_brushes + s_paints) = 0.27 := by
-- Adding the conditions to the proof environment
  sorry

end percent_other_sales_l136_136620


namespace problem_statement_l136_136031

def p (x y : ℝ) : Prop :=
  (x^2 + y^2 ≠ 0) → ¬ (x = 0 ∧ y = 0)

def q (m : ℝ) : Prop :=
  (m > -2) → ∃ x : ℝ, x^2 + 2*x - m = 0

theorem problem_statement : ∀ (x y m : ℝ), p x y ∨ q m :=
sorry

end problem_statement_l136_136031


namespace rectangle_sides_l136_136024

theorem rectangle_sides (x y : ℝ) (h1 : 4 * x = 3 * y) (h2 : x * y = 2 * (x + y)) :
  (x = 7 / 2 ∧ y = 14 / 3) ∨ (x = 14 / 3 ∧ y = 7 / 2) :=
by {
  sorry
}

end rectangle_sides_l136_136024


namespace ascetic_height_l136_136565

theorem ascetic_height (h m : ℝ) (x : ℝ) (hx : h * (m + 1) = (x + h)^2 + (m * h)^2) : x = h * m / (m + 2) :=
sorry

end ascetic_height_l136_136565


namespace vladimir_can_invest_more_profitably_l136_136921

-- Conditions and parameters
def p_buckwheat_initial : ℝ := 70 -- initial price of buckwheat in RUB/kg
def p_buckwheat_2017 : ℝ := 85 -- price of buckwheat in early 2017 in RUB/kg
def rate_2015 : ℝ := 0.16 -- interest rate for annual deposit in 2015
def rate_2016 : ℝ := 0.10 -- interest rate for annual deposit in 2016
def rate_2yr : ℝ := 0.15 -- interest rate for two-year deposit per year

-- Amounts after investments
def amount_annual : ℝ := p_buckwheat_initial * (1 + rate_2015) * (1 + rate_2016)
def amount_2yr : ℝ := p_buckwheat_initial * (1 + rate_2yr)^2

-- Prove that the best investment amount is greater than the 2017 buckwheat price
theorem vladimir_can_invest_more_profitably : max amount_annual amount_2yr > p_buckwheat_2017 := by
  sorry

end vladimir_can_invest_more_profitably_l136_136921


namespace private_schools_in_district_B_l136_136649

theorem private_schools_in_district_B :
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_C_schools := total_schools - district_A_schools - district_B_schools
  let schools_per_kind_in_C := district_C_schools / 3
  let private_schools_in_C := schools_per_kind_in_C
  let remaining_private_schools := private_schools - private_schools_in_C
  remaining_private_schools = 4 :=
by
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_C_schools := total_schools - district_A_schools - district_B_schools
  let schools_per_kind_in_C := district_C_schools / 3
  let private_schools_in_C := schools_per_kind_in_C
  let remaining_private_schools := private_schools - private_schools_in_C
  sorry

end private_schools_in_district_B_l136_136649


namespace avg_salary_increases_by_150_l136_136882

def avg_salary_increase
  (emp_avg_salary : ℕ) (num_employees : ℕ) (mgr_salary : ℕ) : ℕ :=
  let total_salary_employees := emp_avg_salary * num_employees
  let total_salary_with_mgr := total_salary_employees + mgr_salary
  let new_avg_salary := total_salary_with_mgr / (num_employees + 1)
  new_avg_salary - emp_avg_salary

theorem avg_salary_increases_by_150 :
  avg_salary_increase 1800 15 4200 = 150 :=
by
  sorry

end avg_salary_increases_by_150_l136_136882


namespace sum_of_tangencies_l136_136274

noncomputable def f (x : ℝ) : ℝ := max (-7 * x - 23) (max (2 * x + 5) (5 * x + 17))

noncomputable def q (x : ℝ) : ℝ := sorry  -- since the exact form of q is not specified, we use sorry here

-- Define the tangency condition
def is_tangent (q f : ℝ → ℝ) (x : ℝ) : Prop := (q x = f x) ∧ (deriv q x = deriv f x)

-- Define the three points of tangency
variable {x₄ x₅ x₆ : ℝ}

-- q(x) is tangent to f(x) at points x₄, x₅, x₆
axiom tangent_x₄ : is_tangent q f x₄
axiom tangent_x₅ : is_tangent q f x₅
axiom tangent_x₆ : is_tangent q f x₆

-- Now state the theorem
theorem sum_of_tangencies : x₄ + x₅ + x₆ = -70 / 9 :=
sorry

end sum_of_tangencies_l136_136274


namespace minimize_distance_midpoint_Q5_Q6_l136_136393

theorem minimize_distance_midpoint_Q5_Q6 
  (Q : ℝ → ℝ)
  (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10 : ℝ)
  (h1 : Q2 = Q1 + 1)
  (h2 : Q3 = Q2 + 1)
  (h3 : Q4 = Q3 + 1)
  (h4 : Q5 = Q4 + 1)
  (h5 : Q6 = Q5 + 2)
  (h6 : Q7 = Q6 + 2)
  (h7 : Q8 = Q7 + 2)
  (h8 : Q9 = Q8 + 2)
  (h9 : Q10 = Q9 + 2) :
  Q ((Q5 + Q6) / 2) = (Q ((Q1 + Q2) / 2) + Q ((Q3 + Q4) / 2) + Q ((Q7 + Q8) / 2) + Q ((Q9 + Q10) / 2)) :=
sorry

end minimize_distance_midpoint_Q5_Q6_l136_136393


namespace remainder_6_pow_23_mod_5_l136_136094

theorem remainder_6_pow_23_mod_5 : (6 ^ 23) % 5 = 1 := 
by {
  sorry
}

end remainder_6_pow_23_mod_5_l136_136094


namespace fraction_one_two_three_sum_l136_136369

def fraction_one_bedroom : ℝ := 0.12
def fraction_two_bedroom : ℝ := 0.26
def fraction_three_bedroom : ℝ := 0.38
def fraction_four_bedroom : ℝ := 0.24

theorem fraction_one_two_three_sum :
  fraction_one_bedroom + fraction_two_bedroom + fraction_three_bedroom = 0.76 :=
by
  sorry

end fraction_one_two_three_sum_l136_136369


namespace teacher_problems_remaining_l136_136521

theorem teacher_problems_remaining (problems_per_worksheet : Nat) 
                                   (total_worksheets : Nat) 
                                   (graded_worksheets : Nat) 
                                   (remaining_problems : Nat)
  (h1 : problems_per_worksheet = 4)
  (h2 : total_worksheets = 9)
  (h3 : graded_worksheets = 5)
  (h4 : remaining_problems = total_worksheets * problems_per_worksheet - graded_worksheets * problems_per_worksheet) :
  remaining_problems = 16 :=
sorry

end teacher_problems_remaining_l136_136521


namespace find_a_value_l136_136551

theorem find_a_value (a x1 x2 : ℝ) (h1 : a > 0) (h2 : x1 + x2 = 15) 
  (h3 : ∀ x, x^2 - 2 * a * x - 8 * a^2 < 0) : a = 15 / 2 :=
  sorry

end find_a_value_l136_136551


namespace no_value_of_b_l136_136131

theorem no_value_of_b (b : ℤ) : ¬ ∃ (n : ℤ), 2 * b^2 + 3 * b + 2 = n^2 := 
sorry

end no_value_of_b_l136_136131


namespace CitadelSchoolEarnings_l136_136823

theorem CitadelSchoolEarnings :
  let apex_students : Nat := 9
  let apex_days : Nat := 5
  let beacon_students : Nat := 3
  let beacon_days : Nat := 4
  let citadel_students : Nat := 6
  let citadel_days : Nat := 7
  let total_payment : ℕ := 864
  let total_student_days : ℕ := (apex_students * apex_days) + (beacon_students * beacon_days) + (citadel_students * citadel_days)
  let daily_wage_per_student : ℚ := total_payment / total_student_days
  let citadel_student_days : ℕ := citadel_students * citadel_days
  let citadel_earnings : ℚ := daily_wage_per_student * citadel_student_days
  citadel_earnings = 366.55 := by
  sorry

end CitadelSchoolEarnings_l136_136823


namespace butterfat_mixture_l136_136028

/-
  Given:
  - 8 gallons of milk with 40% butterfat
  - x gallons of milk with 10% butterfat
  - Resulting mixture with 20% butterfat

  Prove:
  - x = 16 gallons
-/

theorem butterfat_mixture (x : ℝ) : 
  (0.40 * 8 + 0.10 * x) / (8 + x) = 0.20 → x = 16 := 
by
  sorry

end butterfat_mixture_l136_136028


namespace sum_of_digits_l136_136281

variable {w x y z : ℕ}

theorem sum_of_digits :
  (w + x + y + z = 20) ∧ w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  (y + w = 11) ∧ (x + y = 9) ∧ (w + z = 10) :=
by
  sorry

end sum_of_digits_l136_136281


namespace relationship_x_y_l136_136979

variable (a b x y : ℝ)

theorem relationship_x_y (h1: 0 < a) (h2: a < b)
  (hx : x = (Real.sqrt (a + b) - Real.sqrt b))
  (hy : y = (Real.sqrt b - Real.sqrt (b - a))) :
  x < y :=
  sorry

end relationship_x_y_l136_136979


namespace solve_abs_eqn_l136_136844

theorem solve_abs_eqn (y : ℝ) : (|y - 4| + 3 * y = 11) ↔ (y = 3.5) := by
  sorry

end solve_abs_eqn_l136_136844


namespace find_f_log_l136_136737

def even_function (f : ℝ → ℝ) :=
  ∀ (x : ℝ), f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ (x : ℝ), f (x + p) = f x

theorem find_f_log (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_periodic : periodic_function f 2)
  (h_condition : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 0 → f x = 3 * x + 4 / 9) :
  f (Real.log 5 / Real.log (1 / 3)) = -5 / 9 :=
by
  sorry

end find_f_log_l136_136737


namespace algebraic_expression_evaluation_l136_136299

theorem algebraic_expression_evaluation (a b : ℝ) (h : 1 / a + 1 / (2 * b) = 3) :
  (2 * a - 5 * a * b + 4 * b) / (4 * a * b - 3 * a - 6 * b) = -1 / 2 := 
by
  sorry

end algebraic_expression_evaluation_l136_136299


namespace tangent_line_through_P_is_correct_l136_136184

-- Define the circle and the point
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 3) ^ 2 = 25
def pointP : ℝ × ℝ := (-1, 7)

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y + 31 = 0

-- State the theorem
theorem tangent_line_through_P_is_correct :
  (circle_eq (-1) 7) → 
  (tangent_line (-1) 7) :=
sorry

end tangent_line_through_P_is_correct_l136_136184


namespace geometric_series_sum_squares_l136_136937

theorem geometric_series_sum_squares (a r : ℝ) (hr : -1 < r) (hr2 : r < 1) :
  (∑' n : ℕ, a^2 * r^(3 * n)) = a^2 / (1 - r^3) :=
by
  -- Note: Proof goes here
  sorry

end geometric_series_sum_squares_l136_136937


namespace telethon_total_revenue_l136_136459

noncomputable def telethon_revenue (first_period_hours : ℕ) (first_period_rate : ℕ) 
  (additional_percent_increase : ℕ) (second_period_hours : ℕ) : ℕ :=
  let first_revenue := first_period_hours * first_period_rate
  let second_period_rate := first_period_rate + (first_period_rate * additional_percent_increase / 100)
  let second_revenue := second_period_hours * second_period_rate
  first_revenue + second_revenue

theorem telethon_total_revenue : 
  telethon_revenue 12 5000 20 14 = 144000 :=
by 
  rfl -- replace 'rfl' with 'sorry' if the proof is non-trivial and longer

end telethon_total_revenue_l136_136459


namespace remy_water_usage_l136_136847

theorem remy_water_usage :
  ∃ R : ℕ, (Remy = 3 * R + 1) ∧ 
    (Riley = R + (3 * R + 1) - 2) ∧ 
    (R + (3 * R + 1) + (R + (3 * R + 1) - 2) = 48) ∧ 
    (Remy = 19) :=
sorry

end remy_water_usage_l136_136847


namespace initial_investment_l136_136644

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_investment (A : ℝ) (r : ℝ) (n t : ℕ) (P : ℝ) :
  A = 3630.0000000000005 → r = 0.10 → n = 1 → t = 2 → P = 3000 →
  A = compound_interest P r n t :=
by
  intros hA hr hn ht hP
  rw [compound_interest, hA, hr, hP]
  sorry

end initial_investment_l136_136644


namespace units_digit_sum_l136_136905

theorem units_digit_sum (n1 n2 : ℕ) (h1 : n1 % 10 = 1) (h2 : n2 % 10 = 3) : ((n1^3 + n2^3) % 10) = 8 := 
by
  sorry

end units_digit_sum_l136_136905


namespace find_number_l136_136187

theorem find_number (x : ℕ) (h : x + 3 * x = 20) : x = 5 :=
by
  sorry

end find_number_l136_136187


namespace age_ratio_rahul_deepak_l136_136938

/--
Prove that the ratio between Rahul and Deepak's current ages is 4:3 given the following conditions:
1. After 10 years, Rahul's age will be 26 years.
2. Deepak's current age is 12 years.
-/
theorem age_ratio_rahul_deepak (R D : ℕ) (h1 : R + 10 = 26) (h2 : D = 12) : R / D = 4 / 3 :=
by sorry

end age_ratio_rahul_deepak_l136_136938


namespace problem_l136_136961

theorem problem (q r : ℕ) (hq : 1259 = 23 * q + r) (hq_pos : 0 < q) (hr_pos : 0 < r) :
  q - r ≤ 37 :=
sorry

end problem_l136_136961


namespace length_of_smaller_cube_edge_is_5_l136_136159

-- Given conditions
def stacked_cube_composed_of_smaller_cubes (n: ℕ) (a: ℕ) : Prop := a * a * a = n

def volume_of_larger_cube (l: ℝ) (v: ℝ) : Prop := l ^ 3 = v

-- Problem statement: Prove that the length of one edge of the smaller cube is 5 cm
theorem length_of_smaller_cube_edge_is_5 :
  ∃ s: ℝ, stacked_cube_composed_of_smaller_cubes 8 2 ∧ volume_of_larger_cube (2*s) 1000 ∧ s = 5 :=
  sorry

end length_of_smaller_cube_edge_is_5_l136_136159


namespace solution_triples_l136_136373

noncomputable def find_triples (x y z : ℝ) : Prop :=
  x + y + z = 2008 ∧
  x^2 + y^2 + z^2 = 6024^2 ∧
  (1/x) + (1/y) + (1/z) = 1/2008

theorem solution_triples :
  ∃ (x y z : ℝ), find_triples x y z ∧ (x = 2008 ∧ y = 4016 ∧ z = -4016) :=
sorry

end solution_triples_l136_136373


namespace integer_with_exactly_12_integers_to_its_left_l136_136437

theorem integer_with_exactly_12_integers_to_its_left :
  let initial_list := List.range' 1 20
  let first_half := initial_list.take 10
  let second_half := initial_list.drop 10
  let new_list := second_half ++ first_half
  new_list.get! 12 = 3 :=
by
  let initial_list := List.range' 1 20
  let first_half := initial_list.take 10
  let second_half := initial_list.drop 10
  let new_list := second_half ++ first_half
  sorry

end integer_with_exactly_12_integers_to_its_left_l136_136437


namespace instantaneous_velocity_at_2_l136_136864

noncomputable def S (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

theorem instantaneous_velocity_at_2 :
  (deriv S 2) = 10 :=
by 
  sorry

end instantaneous_velocity_at_2_l136_136864


namespace needle_intersection_probability_l136_136726

noncomputable def needle_probability (a l : ℝ) (h : l < a) : ℝ :=
  (2 * l) / (a * Real.pi)

theorem needle_intersection_probability (a l : ℝ) (h : l < a) :
  needle_probability a l h = 2 * l / (a * Real.pi) :=
by
  -- This is the statement to be proved
  sorry

end needle_intersection_probability_l136_136726


namespace expected_number_of_adjacent_black_pairs_l136_136642

theorem expected_number_of_adjacent_black_pairs :
  let total_cards := 52
  let black_cards := 26
  let adjacent_probability := (black_cards - 1) / (total_cards - 1)
  let expected_per_black_card := black_cards * adjacent_probability / total_cards
  let expected_total := black_cards * adjacent_probability
  expected_total = 650 / 51 := 
by
  let total_cards := 52
  let black_cards := 26
  let adjacent_probability := (black_cards - 1) / (total_cards - 1)
  let expected_total := black_cards * adjacent_probability
  sorry

end expected_number_of_adjacent_black_pairs_l136_136642


namespace tetrahedron_volume_l136_136309

noncomputable def volume_of_tetrahedron (A B C O : Point) (r : ℝ) :=
  1 / 3 * (Real.sqrt (3) / 4 * 2^2 * Real.sqrt 11)

theorem tetrahedron_volume 
  (A B C O : Point)
  (side_length : ℝ)
  (surface_area : ℝ)
  (radius : ℝ)
  (h : ℝ)
  (radius_eq : radius = Real.sqrt (37 / 3))
  (side_length_eq : side_length = 2)
  (surface_area_eq : surface_area = (4 * Real.pi * radius^2))
  (sphere_surface_area_eq : surface_area = 148 * Real.pi / 3)
  (height_eq : h^2 = radius^2 - (2 / 3 * 2 * Real.sqrt 3 / 2)^2)
  (height_value_eq : h = Real.sqrt 11) :
  volume_of_tetrahedron A B C O radius = Real.sqrt 33 / 3 := sorry

end tetrahedron_volume_l136_136309


namespace number_of_pairs_l136_136794

theorem number_of_pairs : 
  (∀ n m : ℕ, (1 ≤ m ∧ m ≤ 2012) → (5^n < 2^m ∧ 2^m < 2^(m+2) ∧ 2^(m+2) < 5^(n+1))) → 
  (∃ c, c = 279) :=
by
  sorry

end number_of_pairs_l136_136794


namespace proof_problem_l136_136129

noncomputable def M : ℕ := 50
noncomputable def T : ℕ := M + Nat.div M 10
noncomputable def W : ℕ := 2 * (M + T)
noncomputable def Th : ℕ := W / 2
noncomputable def total_T_T_W_Th : ℕ := T + W + Th
noncomputable def total_M_T_W_Th : ℕ := M + total_T_T_W_Th
noncomputable def F_S_sun : ℕ := Nat.div (450 - total_M_T_W_Th) 3
noncomputable def car_tolls : ℕ := 150 * 2
noncomputable def bus_tolls : ℕ := 150 * 5
noncomputable def truck_tolls : ℕ := 150 * 10
noncomputable def total_toll : ℕ := car_tolls + bus_tolls + truck_tolls

theorem proof_problem :
  (total_T_T_W_Th = 370) ∧
  (F_S_sun = 10) ∧
  (total_toll = 2550) := by
  sorry

end proof_problem_l136_136129


namespace base_b_square_of_integer_l136_136481

theorem base_b_square_of_integer (b : ℕ) (h : b > 4) : ∃ n : ℕ, (n * n) = b^2 + 4 * b + 4 :=
by 
  sorry

end base_b_square_of_integer_l136_136481


namespace unique_solution_p_zero_l136_136556

theorem unique_solution_p_zero :
  ∃! (x y p : ℝ), 
    (x^2 - y^2 = 0) ∧ 
    (x * y + p * x - p * y = p^2) ↔ 
    p = 0 :=
by sorry

end unique_solution_p_zero_l136_136556


namespace smallest_n_l136_136254

theorem smallest_n :
∃ (n : ℕ), (0 < n) ∧ (∃ k1 : ℕ, 5 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 7 * n = k2 ^ 3) ∧ n = 1225 :=
sorry

end smallest_n_l136_136254


namespace algebraic_expression_value_l136_136108

theorem algebraic_expression_value (a x : ℝ) (h : 3 * a - x = x + 2) (hx : x = 2) : a^2 - 2 * a + 1 = 1 :=
by {
  sorry
}

end algebraic_expression_value_l136_136108


namespace part1_part2_part3_l136_136756

-- Part 1: Simplifying the Expression
theorem part1 (a b : ℝ) : 
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 :=
by sorry

-- Part 2: Finding the Value of an Expression
theorem part2 (x y : ℝ) (h : x^2 - 2 * y = 4) : 
  3 * x^2 - 6 * y - 21 = -9 :=
by sorry

-- Part 3: Evaluating a Compound Expression
theorem part3 (a b c d : ℝ) (h1 : a - 2 * b = 6) (h2 : 2 * b - c = -8) (h3 : c - d = 9) : 
  (a - c) + (2 * b - d) - (2 * b - c) = 7 :=
by sorry

end part1_part2_part3_l136_136756


namespace given_roots_find_coefficients_l136_136145

theorem given_roots_find_coefficients {a b c : ℝ} :
  (1:ℝ)^5 + 2*(1)^4 + a * (1:ℝ)^2 + b * (1:ℝ) = c →
  (-1:ℝ)^5 + 2*(-1:ℝ)^4 + a * (-1:ℝ)^2 + b * (-1:ℝ) = c →
  a = -6 ∧ b = -1 ∧ c = -4 :=
by
  intros h1 h2
  sorry

end given_roots_find_coefficients_l136_136145


namespace bricks_needed_l136_136328

theorem bricks_needed 
    (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ) 
    (wall_length_m : ℝ) (wall_height_m : ℝ) (wall_width_cm : ℝ)
    (H1 : brick_length = 25) (H2 : brick_width = 11.25) (H3 : brick_height = 6)
    (H4 : wall_length_m = 7) (H5 : wall_height_m = 6) (H6 : wall_width_cm = 22.5) :
    (wall_length_m * 100 * wall_height_m * 100 * wall_width_cm) / (brick_length * brick_width * brick_height) = 5600 :=
by
    sorry

end bricks_needed_l136_136328


namespace no_solutions_for_a3_plus_5b3_eq_2016_l136_136467

theorem no_solutions_for_a3_plus_5b3_eq_2016 (a b : ℤ) : a^3 + 5 * b^3 ≠ 2016 :=
by sorry

end no_solutions_for_a3_plus_5b3_eq_2016_l136_136467


namespace sufficient_but_not_necessary_condition_l136_136607

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : ∀ x : ℝ, x > a → x > 2 ∧ ¬(x > 2 → x > a)) : a > 2 :=
sorry

end sufficient_but_not_necessary_condition_l136_136607


namespace total_seats_in_theater_l136_136238

def theater_charges_adults : ℝ := 3.0
def theater_charges_children : ℝ := 1.5
def total_income : ℝ := 510
def number_of_children : ℕ := 60

theorem total_seats_in_theater :
  ∃ (A C : ℕ), C = number_of_children ∧ theater_charges_adults * A + theater_charges_children * C = total_income ∧ A + C = 200 :=
by
  sorry

end total_seats_in_theater_l136_136238


namespace used_car_percentage_l136_136152

-- Define the variables and conditions
variables (used_car_price original_car_price : ℕ) (h_used_car_price : used_car_price = 15000) (h_original_price : original_car_price = 37500)

-- Define the statement to prove the percentage
theorem used_car_percentage (h : used_car_price / original_car_price * 100 = 40) : true :=
sorry

end used_car_percentage_l136_136152


namespace jed_speed_l136_136282

theorem jed_speed
  (posted_speed_limit : ℕ := 50)
  (fine_per_mph_over_limit : ℕ := 16)
  (red_light_fine : ℕ := 75)
  (cellphone_fine : ℕ := 120)
  (parking_fine : ℕ := 50)
  (total_red_light_fines : ℕ := 2 * red_light_fine)
  (total_parking_fines : ℕ := 3 * parking_fine)
  (total_fine : ℕ := 1046)
  (non_speeding_fines : ℕ := total_red_light_fines + cellphone_fine + total_parking_fines)
  (speeding_fine : ℕ := total_fine - non_speeding_fines)
  (mph_over_limit : ℕ := speeding_fine / fine_per_mph_over_limit):
  (posted_speed_limit + mph_over_limit) = 89 :=
by
  sorry

end jed_speed_l136_136282


namespace min_knights_l136_136368

noncomputable def is_lying (n : ℕ) (T : ℕ → Prop) (p : ℕ → Prop) : Prop :=
    (T n → ∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m)) ∨ (¬T n → ¬∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m ∧ m < n))

open Nat

def islanders_condition (T : ℕ → Prop) (p : ℕ → Prop) :=
  ∀ n, n < 80 → (T n ∨ ¬T n) ∧ (T n → ∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m)) ∨ (¬T n → ¬∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m))

theorem min_knights : ∀ (T : ℕ → Prop) (p : ℕ → Prop), islanders_condition T p → ∃ k, k = 70 :=    
by
    sorry

end min_knights_l136_136368


namespace find_y_l136_136372

theorem find_y (x y : ℤ) (h1 : 2 * x - y = 11) (h2 : 4 * x + y ≠ 17) : y = -9 :=
by sorry

end find_y_l136_136372


namespace find_v4_l136_136775

noncomputable def horner_method (x : ℤ) : ℤ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 20
  let v4 := v3 * x - 8
  v4

theorem find_v4 : horner_method (-2) = -16 :=
  by {
    -- Proof goes here, but we are only required to write the statement.
    sorry
  }

end find_v4_l136_136775


namespace solve_for_x_l136_136780

noncomputable def valid_x (x : ℝ) : Prop :=
  let l := 4 * x
  let w := 2 * x + 6
  l * w = 2 * (l + w)

theorem solve_for_x : 
  ∃ (x : ℝ), valid_x x ↔ x = (-3 + Real.sqrt 33) / 4 :=
by
  sorry

end solve_for_x_l136_136780


namespace multiply_preserve_equiv_l136_136111

noncomputable def conditions_equiv_eqn (N D F : Polynomial ℝ) : Prop :=
  (D = F * (D / F)) ∧ (N.degree ≥ F.degree) ∧ (D ≠ 0)

theorem multiply_preserve_equiv (N D F : Polynomial ℝ) :
  conditions_equiv_eqn N D F →
  (N / D = 0 ↔ (N * F) / (D * F) = 0) :=
by
  sorry

end multiply_preserve_equiv_l136_136111


namespace find_angle_l136_136902

theorem find_angle (x : ℝ) (h : 90 - x = 2 * x + 15) : x = 25 :=
by
  sorry

end find_angle_l136_136902


namespace fractions_non_integer_l136_136377

theorem fractions_non_integer (a b c d : ℤ) : 
  ∃ (a b c d : ℤ), 
    ¬((a-b) % 2 = 0 ∧ 
      (b-c) % 2 = 0 ∧ 
      (c-d) % 2 = 0 ∧ 
      (d-a) % 2 = 0) :=
sorry

end fractions_non_integer_l136_136377


namespace triangle_balls_l136_136367

theorem triangle_balls (n : ℕ) (num_tri_balls : ℕ) (num_sq_balls : ℕ) :
  (∀ n : ℕ, num_tri_balls = n * (n + 1) / 2)
  ∧ (num_sq_balls = num_tri_balls + 424)
  ∧ (∀ s : ℕ, s = n - 8 → s * s = num_sq_balls)
  → num_tri_balls = 820 :=
by sorry

end triangle_balls_l136_136367


namespace unique_integer_solution_l136_136900

-- Define the problem statement and the conditions: integers x, y such that x^4 - 2y^2 = 1
theorem unique_integer_solution (x y: ℤ) (h: x^4 - 2 * y^2 = 1) : (x = 1 ∧ y = 0) :=
sorry

end unique_integer_solution_l136_136900


namespace zhou_catches_shuttle_probability_l136_136210

-- Condition 1: Shuttle arrival time and duration
def shuttle_arrival_start : ℕ := 420 -- 7:00 AM in minutes since midnight
def shuttle_duration : ℕ := 15

-- Condition 2: Zhou's random arrival time window
def zhou_arrival_start : ℕ := 410 -- 6:50 AM in minutes since midnight
def zhou_arrival_end : ℕ := 465 -- 7:45 AM in minutes since midnight

-- Total time available for Zhou to arrive (55 minutes) 
def total_time : ℕ := zhou_arrival_end - zhou_arrival_start

-- Time window when Zhou needs to arrive to catch the shuttle (15 minutes)
def successful_time : ℕ := shuttle_arrival_start + shuttle_duration - shuttle_arrival_start

-- Calculate the probability that Zhou catches the shuttle
theorem zhou_catches_shuttle_probability : 
  (successful_time : ℚ) / total_time = 3 / 11 := 
by 
  -- We don't need the actual proof steps, just the statement
  sorry

end zhou_catches_shuttle_probability_l136_136210


namespace appointment_on_tuesday_duration_l136_136769

theorem appointment_on_tuesday_duration :
  let rate := 20
  let monday_appointments := 5
  let monday_each_duration := 1.5
  let thursday_appointments := 2
  let thursday_each_duration := 2
  let saturday_duration := 6
  let weekly_earnings := 410
  let known_earnings := (monday_appointments * monday_each_duration * rate) + (thursday_appointments * thursday_each_duration * rate) + (saturday_duration * rate)
  let tuesday_earnings := weekly_earnings - known_earnings
  (tuesday_earnings / rate = 3) :=
by
  -- let rate := 20
  -- let monday_appointments := 5
  -- let monday_each_duration := 1.5
  -- let thursday_appointments := 2
  -- let thursday_each_duration := 2
  -- let saturday_duration := 6
  -- let weekly_earnings := 410
  -- let known_earnings := (monday_appointments * monday_each_duration * rate) + (thursday_appointments * thursday_each_duration * rate) + (saturday_duration * rate)
  -- let tuesday_earnings := weekly_earnings - known_earnings
  -- exact tuesday_earnings / rate = 3
  sorry

end appointment_on_tuesday_duration_l136_136769


namespace common_factor_of_polynomial_l136_136640

noncomputable def polynomial_common_factor (m : ℤ) : ℤ :=
  let polynomial := 2 * m^3 - 8 * m
  let common_factor := 2 * m
  common_factor  -- We're stating that the common factor is 2 * m

-- The theorem to verify that the common factor of each term in the polynomial is 2m
theorem common_factor_of_polynomial (m : ℤ) : 
  polynomial_common_factor m = 2 * m := by
  sorry

end common_factor_of_polynomial_l136_136640


namespace system_solutions_l136_136284

theorem system_solutions : {p : ℝ × ℝ | p.snd ^ 2 = p.fst ∧ p.snd = p.fst} = {⟨1, 1⟩, ⟨0, 0⟩} :=
by
  sorry

end system_solutions_l136_136284


namespace total_cost_of_refueling_l136_136744

theorem total_cost_of_refueling 
  (smaller_tank_capacity : ℤ)
  (larger_tank_capacity : ℤ)
  (num_smaller_planes : ℤ)
  (num_larger_planes : ℤ)
  (fuel_cost_per_liter : ℤ)
  (service_charge_per_plane : ℤ)
  (total_cost : ℤ) :
  smaller_tank_capacity = 60 →
  larger_tank_capacity = 90 →
  num_smaller_planes = 2 →
  num_larger_planes = 2 →
  fuel_cost_per_liter = 50 →
  service_charge_per_plane = 100 →
  total_cost = (num_smaller_planes * smaller_tank_capacity + num_larger_planes * larger_tank_capacity) * (fuel_cost_per_liter / 100) + (num_smaller_planes + num_larger_planes) * service_charge_per_plane →
  total_cost = 550 :=
by
  intros
  sorry

end total_cost_of_refueling_l136_136744


namespace arithmetic_sequence_a8_l136_136555

/-- In an arithmetic sequence with the given sum of terms, prove the value of a_8 is 14. -/
theorem arithmetic_sequence_a8 (a : ℕ → ℕ) (d : ℕ) (h1 : ∀ (n : ℕ), a (n+1) = a n + d)
    (h2 : a 2 + a 7 + a 8 + a 9 + a 14 = 70) : a 8 = 14 :=
  sorry

end arithmetic_sequence_a8_l136_136555


namespace domain_of_f_l136_136076

def domain_of_log_func := Set ℝ

def is_valid (x : ℝ) : Prop := x - 1 > 0

def func_domain (f : ℝ → ℝ) : domain_of_log_func := {x : ℝ | is_valid x}

theorem domain_of_f :
  func_domain (λ x => Real.log (x - 1)) = {x : ℝ | 1 < x} := by
  sorry

end domain_of_f_l136_136076


namespace union_of_sets_l136_136524

def setA := { x : ℝ | -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3 }
def setB := { x : ℝ | (x - 2) / x ≤ 0 }

theorem union_of_sets :
  { x : ℝ | -1 ≤ x ∧ x ≤ 2 } = setA ∪ setB :=
by
  sorry

end union_of_sets_l136_136524


namespace value_of_p_l136_136105

theorem value_of_p (p q r : ℕ) (h1 : p + q + r = 70) (h2 : p = 2*q) (h3 : q = 3*r) : p = 42 := 
by 
  sorry

end value_of_p_l136_136105


namespace tetrahedron_inequality_l136_136278

theorem tetrahedron_inequality (t1 t2 t3 t4 τ1 τ2 τ3 τ4 : ℝ) 
  (ht1 : t1 > 0) (ht2 : t2 > 0) (ht3 : t3 > 0) (ht4 : t4 > 0)
  (hτ1 : τ1 > 0) (hτ2 : τ2 > 0) (hτ3 : τ3 > 0) (hτ4 : τ4 > 0)
  (sphere_inscribed : ∀ {x y : ℝ}, x > 0 → y > 0 → x^2 / y^2 ≤ (x - 2 * y) ^ 2 / x ^ 2) :
  (τ1 / t1 + τ2 / t2 + τ3 / t3 + τ4 / t4) ≥ 1 
  ∧ (τ1 / t1 + τ2 / t2 + τ3 / t3 + τ4 / t4 = 1 ↔ t1 = t2 ∧ t2 = t3 ∧ t3 = t4) := by
  sorry

end tetrahedron_inequality_l136_136278


namespace maximum_value_of_f_l136_136125

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x - 2 * Real.sin (3 * x))

theorem maximum_value_of_f :
  ∃ x : ℝ, f x = (16 * Real.sqrt 3) / 9 :=
sorry

end maximum_value_of_f_l136_136125


namespace octahedron_parallel_edge_pairs_count_l136_136814

-- defining a regular octahedron structure
structure RegularOctahedron where
  vertices : Fin 8
  edges : Fin 12
  faces : Fin 8

noncomputable def numberOfStrictlyParallelEdgePairs (O : RegularOctahedron) : Nat :=
  12 -- Given the symmetry and structure.

theorem octahedron_parallel_edge_pairs_count (O : RegularOctahedron) : 
  numberOfStrictlyParallelEdgePairs O = 12 :=
by
  sorry

end octahedron_parallel_edge_pairs_count_l136_136814


namespace median_possible_values_l136_136392

variable {ι : Type} -- Representing the set S as a type
variable (S : Finset ℤ) -- S is a finite set of integers

def conditions (S: Finset ℤ) : Prop :=
  S.card = 9 ∧
  {5, 7, 10, 13, 17, 21} ⊆ S

theorem median_possible_values :
  ∀ S : Finset ℤ, conditions S → ∃ medians : Finset ℤ, medians.card = 7 :=
by
  sorry

end median_possible_values_l136_136392


namespace piggy_bank_balance_l136_136968

theorem piggy_bank_balance (original_amount : ℕ) (taken_out : ℕ) : original_amount = 5 ∧ taken_out = 2 → original_amount - taken_out = 3 :=
by sorry

end piggy_bank_balance_l136_136968


namespace arithmetic_mean_of_17_29_45_64_l136_136602

theorem arithmetic_mean_of_17_29_45_64 : (17 + 29 + 45 + 64) / 4 = 38.75 := by
  sorry

end arithmetic_mean_of_17_29_45_64_l136_136602


namespace find_complement_intersection_find_union_complement_subset_implies_a_range_l136_136575

-- Definitions for sets A and B
def A : Set ℝ := { x | 3 ≤ x ∧ x < 6 }
def B : Set ℝ := { x | 2 < x ∧ x < 9 }

-- Definitions for complements and subsets
def complement (S : Set ℝ) : Set ℝ := { x | x ∉ S }
def intersection (S T : Set ℝ) : Set ℝ := { x | x ∈ S ∧ x ∈ T }
def union (S T : Set ℝ) : Set ℝ := { x | x ∈ S ∨ x ∈ T }

-- Definition for set C as a parameterized set by a
def C (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

-- Proof statements
theorem find_complement_intersection :
  complement (intersection A B) = { x | x < 3 ∨ x ≥ 6 } :=
by sorry

theorem find_union_complement :
  union (complement B) A = { x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9 } :=
by sorry

theorem subset_implies_a_range (a : ℝ) :
  C a ⊆ B → a ∈ {x | 2 ≤ x ∧ x ≤ 8} :=
by sorry

end find_complement_intersection_find_union_complement_subset_implies_a_range_l136_136575


namespace no_positive_integer_solution_l136_136862

theorem no_positive_integer_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  ¬ (∃ (k : ℕ), (xy + 1) * (xy + x + 2) = k^2) :=
by {
  sorry
}

end no_positive_integer_solution_l136_136862


namespace initial_bottles_proof_l136_136072

-- Define the conditions as variables and statements
def initial_bottles (X : ℕ) : Prop :=
X - 8 + 45 = 51

-- Theorem stating the proof problem
theorem initial_bottles_proof : initial_bottles 14 :=
by
  -- We need to prove the following:
  -- 14 - 8 + 45 = 51
  sorry

end initial_bottles_proof_l136_136072


namespace fraction_of_value_l136_136013

def value_this_year : ℝ := 16000
def value_last_year : ℝ := 20000

theorem fraction_of_value : (value_this_year / value_last_year) = 4 / 5 := by
  sorry

end fraction_of_value_l136_136013


namespace range_of_p_l136_136356

-- Conditions: p is a prime number and the roots of the quadratic equation are integers 
def p_is_prime (p : ℕ) : Prop := Nat.Prime p

def roots_are_integers (p : ℕ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ x * y = -204 * p ∧ (x + y) = p

-- Main statement: Prove the range of p
theorem range_of_p (p : ℕ) (hp : p_is_prime p) (hr : roots_are_integers p) : 11 < p ∧ p ≤ 21 :=
  sorry

end range_of_p_l136_136356


namespace tips_fraction_l136_136729

-- Define the conditions
variables (S T : ℝ) (h : T = (2 / 4) * S)

-- The statement to be proved
theorem tips_fraction : (T / (S + T)) = 1 / 3 :=
by
  sorry

end tips_fraction_l136_136729


namespace matrix_sum_correct_l136_136295

def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 0],
  ![1, 2]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-5, -7],
  ![4, -9]
]

def C : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-2, -7],
  ![5, -7]
]

theorem matrix_sum_correct : A + B = C := by 
  sorry

end matrix_sum_correct_l136_136295


namespace joggers_meetings_l136_136259

theorem joggers_meetings (road_length : ℝ)
  (speed_A speed_B : ℝ)
  (start_time : ℝ)
  (meeting_time : ℝ) :
  road_length = 400 → 
  speed_A = 3 → 
  speed_B = 2.5 →
  start_time = 0 → 
  meeting_time = 1200 → 
  ∃ y : ℕ, y = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end joggers_meetings_l136_136259


namespace value_not_uniquely_determined_l136_136236

variables (v : Fin 9 → ℤ) (s : Fin 9 → ℤ)

-- Given conditions
axiom announced_sums : ∀ i, s i = v ((i - 1) % 9) + v ((i + 1) % 9)
axiom sums_sequence : s 0 = 3 ∧ s 1 = 7 ∧ s 2 = 12 ∧ s 3 = 18 ∧ s 4 = 24 ∧ s 5 = 31 ∧ s 6 = 40 ∧ s 7 = 48 ∧ s 8 = 53

-- Statement asserting the indeterminacy of v_5
theorem value_not_uniquely_determined (h: s 3 = 18) : 
  ∃ v : Fin 9 → ℤ, sorry :=
sorry

end value_not_uniquely_determined_l136_136236


namespace exponential_sequence_term_eq_l136_136985

-- Definitions for the conditions
variable {α : Type} [CommRing α] (q : α)
def a (n : ℕ) : α := q * (q ^ (n - 1))

-- Statement of the problem
theorem exponential_sequence_term_eq : a q 9 = a q 3 * a q 7 := by
  sorry

end exponential_sequence_term_eq_l136_136985


namespace charcoal_drawings_count_l136_136571

-- Defining the conditions
def total_drawings : Nat := 25
def colored_pencil_drawings : Nat := 14
def blending_marker_drawings : Nat := 7

-- Defining the target value for charcoal drawings
def charcoal_drawings : Nat := total_drawings - (colored_pencil_drawings + blending_marker_drawings)

-- The theorem we need to prove
theorem charcoal_drawings_count : charcoal_drawings = 4 :=
by
  -- Lean proof goes here, but since we skip the proof, we'll just use 'sorry'
  sorry

end charcoal_drawings_count_l136_136571


namespace sin_sum_leq_3div2_sqrt3_l136_136826

theorem sin_sum_leq_3div2_sqrt3 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 / 2) * Real.sqrt 3 :=
by
  sorry

end sin_sum_leq_3div2_sqrt3_l136_136826


namespace bianca_total_drawing_time_l136_136831

def total_drawing_time (a b : ℕ) : ℕ := a + b

theorem bianca_total_drawing_time :
  let a := 22
  let b := 19
  total_drawing_time a b = 41 :=
by
  sorry

end bianca_total_drawing_time_l136_136831


namespace virginia_sweettarts_l136_136012

theorem virginia_sweettarts (total_sweettarts : ℕ) (sweettarts_per_person : ℕ) (friends : ℕ) (sweettarts_left : ℕ) 
  (h1 : total_sweettarts = 13) 
  (h2 : sweettarts_per_person = 3) 
  (h3 : total_sweettarts = sweettarts_per_person * (friends + 1) + sweettarts_left) 
  (h4 : sweettarts_left < sweettarts_per_person) :
  friends = 3 :=
by
  sorry

end virginia_sweettarts_l136_136012


namespace maximum_value_of_expression_l136_136324

noncomputable def max_function_value (x y z : ℝ) : ℝ := 
  (x^3 - x * y^2 + y^3) * (x^3 - x * z^2 + z^3) * (y^3 - y * z^2 + z^3)

theorem maximum_value_of_expression : 
  ∃ x y z : ℝ, (x >= 0) ∧ (y >= 0) ∧ (z >= 0) ∧ (x + y + z = 3) 
  ∧ max_function_value x y z = 2916 / 2187 := 
sorry

end maximum_value_of_expression_l136_136324


namespace renu_work_rate_l136_136927

theorem renu_work_rate (R : ℝ) :
  (∀ (renu_rate suma_rate combined_rate : ℝ),
    renu_rate = 1 / R ∧
    suma_rate = 1 / 6 ∧
    combined_rate = 1 / 3 ∧    
    combined_rate = renu_rate + suma_rate) → 
    R = 6 :=
by
  sorry

end renu_work_rate_l136_136927


namespace avg_tickets_male_l136_136960

theorem avg_tickets_male (M F : ℕ) (w : ℕ) 
  (h1 : M / F = 1 / 2) 
  (h2 : (M + F) * 66 = M * w + F * 70) 
  : w = 58 := 
sorry

end avg_tickets_male_l136_136960


namespace minimum_sum_PE_PC_l136_136262

noncomputable def point := (ℝ × ℝ)
noncomputable def length (p1 p2 : point) : ℝ := Real.sqrt (((p1.1 - p2.1)^2) + ((p1.2 - p2.2)^2))

theorem minimum_sum_PE_PC :
  let A : point := (0, 3)
  let B : point := (3, 3)
  let C : point := (3, 0)
  let D : point := (0, 0)
  ∃ P E : point, E.1 = 3 ∧ E.2 = 1 ∧ (∃ t : ℝ, t ≥ 0 ∧ t ≤ 3 ∧ P.1 = 3 - t ∧ P.2 = t) ∧
    (length P E + length P C = Real.sqrt 13) :=
by
  sorry

end minimum_sum_PE_PC_l136_136262


namespace initial_number_of_men_l136_136750

variable (M : ℕ) (A : ℕ)
variable (change_in_age: ℕ := 16)
variable (age_increment: ℕ := 2)

theorem initial_number_of_men :
  ((A + age_increment) * M = A * M + change_in_age) → M = 8 :=
by
  intros h_1
  sorry

end initial_number_of_men_l136_136750


namespace rectangle_area_l136_136838

theorem rectangle_area (P : ℕ) (a : ℕ) (b : ℕ) (h₁ : P = 2 * (a + b)) (h₂ : P = 40) (h₃ : a = 5) : a * b = 75 :=
by
  sorry

end rectangle_area_l136_136838


namespace ratio_paislee_to_calvin_l136_136730

theorem ratio_paislee_to_calvin (calvin_points paislee_points : ℕ) (h1 : calvin_points = 500) (h2 : paislee_points = 125) : paislee_points / calvin_points = 1 / 4 := by
  sorry

end ratio_paislee_to_calvin_l136_136730


namespace percentage_unloaded_at_second_store_l136_136874

theorem percentage_unloaded_at_second_store
  (initial_weight : ℝ)
  (percent_unloaded_first : ℝ)
  (remaining_weight_after_deliveries : ℝ)
  (remaining_weight_after_first : ℝ)
  (weight_unloaded_second : ℝ)
  (percent_unloaded_second : ℝ) :
  initial_weight = 50000 →
  percent_unloaded_first = 0.10 →
  remaining_weight_after_deliveries = 36000 →
  remaining_weight_after_first = initial_weight * (1 - percent_unloaded_first) →
  weight_unloaded_second = remaining_weight_after_first - remaining_weight_after_deliveries →
  percent_unloaded_second = (weight_unloaded_second / remaining_weight_after_first) * 100 →
  percent_unloaded_second = 20 :=
by
  intros _
  sorry

end percentage_unloaded_at_second_store_l136_136874


namespace number_of_children_is_4_l136_136053

-- Define the conditions from the problem
def youngest_child_age : ℝ := 1.5
def sum_of_ages : ℝ := 12
def common_difference : ℝ := 1

-- Define the number of children
def n : ℕ := 4

-- Prove that the number of children is 4 given the conditions
theorem number_of_children_is_4 :
  (∃ n : ℕ, (n / 2) * (2 * youngest_child_age + (n - 1) * common_difference) = sum_of_ages) ↔ n = 4 :=
by sorry

end number_of_children_is_4_l136_136053


namespace common_ratio_of_geometric_series_l136_136343

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 12 / 7) : b / a = 3 := by
  sorry

end common_ratio_of_geometric_series_l136_136343


namespace distinct_cube_units_digits_l136_136700

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l136_136700


namespace original_price_of_cycle_l136_136157

theorem original_price_of_cycle (SP : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h_SP : SP = 1080)
  (h_gain_percent: gain_percent = 60)
  (h_relation : SP = 1.6 * P)
  : P = 675 :=
by {
  sorry
}

end original_price_of_cycle_l136_136157


namespace geo_seq_second_term_l136_136600

theorem geo_seq_second_term (b r : Real) 
  (h1 : 280 * r = b) 
  (h2 : b * r = 90 / 56) 
  (h3 : b > 0) 
  : b = 15 * Real.sqrt 2 := 
by 
  sorry

end geo_seq_second_term_l136_136600


namespace prime_product_sum_l136_136417

theorem prime_product_sum (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (h : (p * q * r = 101 * (p + q + r))) : 
  p = 101 ∧ q = 2 ∧ r = 103 :=
sorry

end prime_product_sum_l136_136417


namespace find_f_2015_plus_f_2016_l136_136576

def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom functional_equation (x : ℝ) : f (3/2 - x) = f x
axiom value_at_minus2 : f (-2) = -3

theorem find_f_2015_plus_f_2016 : f 2015 + f 2016 = 3 := 
by {
  sorry
}

end find_f_2015_plus_f_2016_l136_136576


namespace recurring_division_l136_136261

def recurring_to_fraction (recurring: ℝ) (part: ℝ): ℝ :=
  part * recurring

theorem recurring_division (recurring: ℝ) (part1 part2: ℝ):
  recurring_to_fraction recurring part1 = 0.63 →
  recurring_to_fraction recurring part2 = 0.18 →
  recurring ≠ 0 →
  (0.63:ℝ)/0.18 = (7:ℝ)/2 :=
by
  intros h1 h2 h3
  rw [recurring_to_fraction] at h1 h2
  sorry

end recurring_division_l136_136261


namespace cost_per_treat_l136_136182

def treats_per_day : ℕ := 2
def days_in_month : ℕ := 30
def total_spent : ℝ := 6.0

theorem cost_per_treat : (total_spent / (treats_per_day * days_in_month : ℕ)) = 0.10 :=
by 
  sorry

end cost_per_treat_l136_136182


namespace average_upstream_speed_l136_136376

/--
There are three boats moving down a river. Boat A moves downstream at a speed of 1 km in 4 minutes 
and upstream at a speed of 1 km in 8 minutes. Boat B moves downstream at a speed of 1 km in 
5 minutes and upstream at a speed of 1 km in 11 minutes. Boat C moves downstream at a speed of 
1 km in 6 minutes and upstream at a speed of 1 km in 10 minutes. Prove that the average speed 
of the boats against the current is 6.32 km/h.
-/
theorem average_upstream_speed :
  let speed_A_upstream := 1 / (8 / 60 : ℝ)
  let speed_B_upstream := 1 / (11 / 60 : ℝ)
  let speed_C_upstream := 1 / (10 / 60 : ℝ)
  let average_speed := (speed_A_upstream + speed_B_upstream + speed_C_upstream) / 3
  average_speed = 6.32 :=
by
  let speed_A_upstream := 1 / (8 / 60 : ℝ)
  let speed_B_upstream := 1 / (11 / 60 : ℝ)
  let speed_C_upstream := 1 / (10 / 60 : ℝ)
  let average_speed := (speed_A_upstream + speed_B_upstream + speed_C_upstream) / 3
  sorry

end average_upstream_speed_l136_136376


namespace discount_percentage_l136_136143

theorem discount_percentage (original_price sale_price : ℝ) (h1 : original_price = 150) (h2 : sale_price = 135) : 
  (original_price - sale_price) / original_price * 100 = 10 :=
by 
  sorry

end discount_percentage_l136_136143


namespace sophie_aunt_money_l136_136257

noncomputable def totalMoneyGiven (shirts: ℕ) (shirtCost: ℝ) (trousers: ℕ) (trouserCost: ℝ) (additionalItems: ℕ) (additionalItemCost: ℝ) : ℝ :=
  shirts * shirtCost + trousers * trouserCost + additionalItems * additionalItemCost

theorem sophie_aunt_money : totalMoneyGiven 2 18.50 1 63 4 40 = 260 := 
by
  sorry

end sophie_aunt_money_l136_136257


namespace num_of_consecutive_sets_sum_18_eq_2_l136_136816

theorem num_of_consecutive_sets_sum_18_eq_2 : 
  ∃ (sets : Finset (Finset ℕ)), 
    (∀ s ∈ sets, (∃ n a, n ≥ 3 ∧ (s = Finset.range (a + n - 1) \ Finset.range (a - 1)) ∧ 
    s.sum id = 18)) ∧ 
    sets.card = 2 := 
sorry

end num_of_consecutive_sets_sum_18_eq_2_l136_136816


namespace remaining_amount_needed_l136_136943

def goal := 150
def earnings_from_3_families := 3 * 10
def earnings_from_15_families := 15 * 5
def total_earnings := earnings_from_3_families + earnings_from_15_families
def remaining_amount := goal - total_earnings

theorem remaining_amount_needed : remaining_amount = 45 := by
  sorry

end remaining_amount_needed_l136_136943


namespace value_of_expression_l136_136371

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : a^2 - b^2 + 6 * b = 9 :=
by
  sorry

end value_of_expression_l136_136371


namespace second_term_of_geo_series_l136_136808

theorem second_term_of_geo_series
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h_r : r = -1 / 3)
  (h_S : S = 25)
  (h_sum : S = a / (1 - r)) :
  (a * r) = -100 / 9 :=
by
  -- Definitions and conditions here are provided
  have hr : r = -1 / 3 := by exact h_r
  have hS : S = 25 := by exact h_S
  have hsum : S = a / (1 - r) := by exact h_sum
  -- The proof of (a * r) = -100 / 9 goes here
  sorry

end second_term_of_geo_series_l136_136808


namespace area_half_l136_136733

theorem area_half (width height : ℝ) (h₁ : width = 25) (h₂ : height = 16) :
  (width * height) / 2 = 200 :=
by
  -- The formal proof is skipped here
  sorry

end area_half_l136_136733


namespace mul_mod_correct_l136_136227

theorem mul_mod_correct :
  (2984 * 3998) % 1000 = 32 :=
by
  sorry

end mul_mod_correct_l136_136227


namespace max_value_of_expression_l136_136791

theorem max_value_of_expression (x y : ℝ) (h1 : |x - y| ≤ 2) (h2 : |3 * x + y| ≤ 6) : x^2 + y^2 ≤ 10 :=
sorry

end max_value_of_expression_l136_136791


namespace percentage_increase_in_population_due_to_birth_is_55_l136_136298

/-- The initial population at the start of the period is 100,000 people. -/
def initial_population : ℕ := 100000

/-- The period of observation is 10 years. -/
def period : ℕ := 10

/-- The number of people leaving the area each year due to emigration is 2000. -/
def emigration_per_year : ℕ := 2000

/-- The number of people coming into the area each year due to immigration is 2500. -/
def immigration_per_year : ℕ := 2500

/-- The population at the end of the period is 165,000 people. -/
def final_population : ℕ := 165000

/-- The net migration per year is calculated by subtracting emigration from immigration. -/
def net_migration_per_year : ℕ := immigration_per_year - emigration_per_year

/-- The total net migration over the period is obtained by multiplying net migration per year by the number of years. -/
def net_migration_over_period : ℕ := net_migration_per_year * period

/-- The total population increase is the difference between the final and initial population. -/
def total_population_increase : ℕ := final_population - initial_population

/-- The increase in population due to birth is calculated by subtracting net migration over the period from the total population increase. -/
def increase_due_to_birth : ℕ := total_population_increase - net_migration_over_period

/-- The percentage increase in population due to birth is calculated by dividing the increase due to birth by the initial population, and then multiplying by 100 to convert to percentage. -/
def percentage_increase_due_to_birth : ℕ := (increase_due_to_birth * 100) / initial_population

/-- The final Lean statement to prove. -/
theorem percentage_increase_in_population_due_to_birth_is_55 :
  percentage_increase_due_to_birth = 55 := by
sorry

end percentage_increase_in_population_due_to_birth_is_55_l136_136298


namespace difference_is_1343_l136_136189

-- Define the larger number L and the relationship with the smaller number S.
def L : ℕ := 1608
def quotient : ℕ := 6
def remainder : ℕ := 15

-- Define the relationship: L = 6S + 15
def relationship (S : ℕ) : Prop := L = quotient * S + remainder

-- The theorem we want to prove: The difference between the larger and smaller number is 1343
theorem difference_is_1343 (S : ℕ) (h_rel : relationship S) : L - S = 1343 :=
by
  sorry

end difference_is_1343_l136_136189


namespace good_coloring_count_l136_136689

noncomputable def c_n (n : ℕ) : ℤ :=
  1 / 2 * (3^(n + 1) + (-1)^(n + 1))

theorem good_coloring_count (n : ℕ) : 
  ∃ c : ℕ → ℤ, c n = c_n n := sorry

end good_coloring_count_l136_136689


namespace x_quad_greater_l136_136362

theorem x_quad_greater (x : ℝ) : x^4 > x - 1/2 :=
sorry

end x_quad_greater_l136_136362


namespace find_range_of_a_l136_136266

noncomputable def value_range_for_a : Set ℝ := {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4}

theorem find_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0)  ∧
  (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0) ∧
  (¬ (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0) → ¬ (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0))
  → a ∈ value_range_for_a :=
sorry

end find_range_of_a_l136_136266


namespace odd_perfect_prime_form_n_is_seven_l136_136339

theorem odd_perfect_prime_form (n p s m : ℕ) (h₁ : n % 2 = 1) (h₂ : ∃ k : ℕ, p = 4 * k + 1) (h₃ : ∃ h : ℕ, s = 4 * h + 1) (h₄ : n = p^s * m^2) (h₅ : ¬ p ∣ m) :
  ∃ k h : ℕ, p = 4 * k + 1 ∧ s = 4 * h + 1 :=
sorry

theorem n_is_seven (n : ℕ) (h₁ : n > 1) (h₂ : ∃ k : ℕ, k * k = n -1) (h₃ : ∃ l : ℕ, l * l = (n * (n + 1)) / 2) :
  n = 7 :=
sorry

end odd_perfect_prime_form_n_is_seven_l136_136339


namespace cost_difference_l136_136342

-- Define the costs
def cost_chocolate : ℕ := 3
def cost_candy_bar : ℕ := 7

-- Define the difference to be proved
theorem cost_difference :
  cost_candy_bar - cost_chocolate = 4 :=
by
  -- trivial proof steps
  sorry

end cost_difference_l136_136342


namespace total_volume_of_barrel_l136_136681

-- Define the total volume of the barrel and relevant conditions.
variable (x : ℝ) -- total volume of the barrel

-- State the given condition about the barrel's honey content.
def condition := (0.7 * x - 0.3 * x = 30)

-- Goal to prove:
theorem total_volume_of_barrel : condition x → x = 75 :=
by
  sorry

end total_volume_of_barrel_l136_136681


namespace geoff_additional_votes_needed_l136_136202

-- Define the given conditions
def totalVotes : ℕ := 6000
def geoffPercentage : ℕ := 5 -- Represent 0.5% as 5 out of 1000 for better integer computation
def requiredPercentage : ℕ := 505 -- Represent 50.5% as 505 out of 1000 for better integer computation

-- Define the expressions for the number of votes received by Geoff and the votes required to win
def geoffVotes := (geoffPercentage * totalVotes) / 1000
def requiredVotes := (requiredPercentage * totalVotes) / 1000 + 1

-- The proposition to prove the additional number of votes needed for Geoff to win
theorem geoff_additional_votes_needed : requiredVotes - geoffVotes = 3001 := by sorry

end geoff_additional_votes_needed_l136_136202


namespace find_volume_of_pure_alcohol_l136_136081

variable (V1 Vf V2 : ℝ)
variable (P1 Pf : ℝ)

theorem find_volume_of_pure_alcohol
  (h : V2 = Vf * Pf / 100 - V1 * P1 / 100) : 
  V2 = Vf * (Pf / 100) - V1 * (P1 / 100) :=
by
  -- This is the theorem statement. The proof is omitted.
  sorry

end find_volume_of_pure_alcohol_l136_136081


namespace pure_imaginary_k_l136_136859

theorem pure_imaginary_k (k : ℝ) :
  (2 * k^2 - 3 * k - 2 = 0) → (k^2 - 2 * k ≠ 0) → k = -1 / 2 :=
by
  intro hr hi
  -- Proof will go here.
  sorry

end pure_imaginary_k_l136_136859


namespace smallest_special_number_gt_3429_l136_136240

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end smallest_special_number_gt_3429_l136_136240


namespace find_shortest_height_l136_136738

variable (T S P Q : ℝ)

theorem find_shortest_height (h1 : T = 77.75) (h2 : T = S + 9.5) (h3 : P = S + 5) (h4 : Q = P - 3) : S = 68.25 :=
  sorry

end find_shortest_height_l136_136738


namespace complement_of_P_union_Q_in_Z_is_M_l136_136058

-- Definitions of the sets M, P, Q
def M : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}
def P : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

-- Theorem statement
theorem complement_of_P_union_Q_in_Z_is_M : (Set.univ \ (P ∪ Q)) = M :=
by 
  sorry

end complement_of_P_union_Q_in_Z_is_M_l136_136058


namespace right_triangle_angle_l136_136951

open Real

theorem right_triangle_angle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h2 : c^2 = 2 * a * b) : 
  ∃ θ : ℝ, θ = 45 ∧ tan θ = a / b := 
by sorry

end right_triangle_angle_l136_136951


namespace largest_whole_number_l136_136303

theorem largest_whole_number (x : ℕ) : 9 * x < 150 → x ≤ 16 :=
by sorry

end largest_whole_number_l136_136303


namespace max_value_of_f_l136_136822

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x + Real.sin x

theorem max_value_of_f : ∃ M, ∀ x, f x ≤ M ∧ (∃ y, f y = M) := by
  use Real.sqrt 5
  sorry

end max_value_of_f_l136_136822


namespace annual_income_increase_l136_136805

variable (x y : ℝ)

-- Definitions of the conditions
def regression_line (x : ℝ) : ℝ := 0.254 * x + 0.321

-- The statement we want to prove
theorem annual_income_increase (x : ℝ) : regression_line (x + 1) - regression_line x = 0.254 := 
sorry

end annual_income_increase_l136_136805


namespace avg_weights_N_square_of_integer_l136_136102

theorem avg_weights_N_square_of_integer (N : ℕ) :
  (∃ S : ℕ, S > 0 ∧ ∃ k : ℕ, k * k = N + 1 ∧ S = (N * (N + 1)) / 2 / (N - k + 1) ∧ (N * (N + 1)) / 2 - S = (N - k) * S) ↔ (∃ k : ℕ, k * k = N + 1) := by
  sorry

end avg_weights_N_square_of_integer_l136_136102


namespace max_dot_product_on_circle_l136_136293

theorem max_dot_product_on_circle :
  (∃(x y : ℝ),
    x^2 + (y - 3)^2 = 1 ∧
    2 ≤ y ∧ y ≤ 4 ∧
    (∀(y : ℝ), (2 ≤ y ∧ y ≤ 4 →
      (x^2 + y^2 - 4) ≤ 12))) := by
  sorry

end max_dot_product_on_circle_l136_136293


namespace find_a7_l136_136914

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l136_136914


namespace triangle_equilateral_l136_136727

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
a = b ∧ b = c

theorem triangle_equilateral (A B C a b c : ℝ) (hB : B = 60) (hb : b^2 = a * c) (hcos : b^2 = a^2 + c^2 - a * c):
  is_equilateral a b c :=
by
  sorry

end triangle_equilateral_l136_136727


namespace train_length_l136_136191

theorem train_length (speed_kmph : ℤ) (time_sec : ℤ) (expected_length_m : ℤ) 
    (speed_kmph_eq : speed_kmph = 72)
    (time_sec_eq : time_sec = 7)
    (expected_length_eq : expected_length_m = 140) :
    expected_length_m = (speed_kmph * 1000 / 3600) * time_sec :=
by 
    sorry

end train_length_l136_136191


namespace committee_count_l136_136423

-- Definitions based on conditions
def num_males := 15
def num_females := 10

-- Define the binomial coefficient
def binomial (n k : ℕ) := Nat.choose n k

-- Define the total number of committees
def num_committees_with_at_least_two_females : ℕ :=
  binomial num_females 2 * binomial num_males 3 +
  binomial num_females 3 * binomial num_males 2 +
  binomial num_females 4 * binomial num_males 1 +
  binomial num_females 5 * binomial num_males 0

theorem committee_count : num_committees_with_at_least_two_females = 36477 :=
by {
  sorry
}

end committee_count_l136_136423


namespace problem_l136_136488

def Y (a b : ℤ) : ℤ := a^2 - 2 * a * b + b^2
def Z (a b : ℤ) : ℤ := a * b + a + b

theorem problem
  : Z (Y 5 3) (Y 2 1) = 9 := by
  sorry

end problem_l136_136488


namespace consecutive_even_numbers_divisible_by_384_l136_136142

theorem consecutive_even_numbers_divisible_by_384 (n : Nat) (h1 : n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) = 384) : n = 6 :=
sorry

end consecutive_even_numbers_divisible_by_384_l136_136142


namespace bricks_in_row_l136_136176

theorem bricks_in_row 
  (total_bricks : ℕ) 
  (rows_per_wall : ℕ) 
  (num_walls : ℕ)
  (total_rows : ℕ)
  (h1 : total_bricks = 3000)
  (h2 : rows_per_wall = 50)
  (h3 : num_walls = 2) 
  (h4 : total_rows = rows_per_wall * num_walls) :
  total_bricks / total_rows = 30 :=
by
  sorry

end bricks_in_row_l136_136176


namespace not_sixth_power_of_integer_l136_136034

theorem not_sixth_power_of_integer (n : ℕ) : ¬ ∃ k : ℤ, 6 * n^3 + 3 = k^6 :=
by
  sorry

end not_sixth_power_of_integer_l136_136034


namespace find_unknown_number_l136_136605

theorem find_unknown_number :
  ∃ (x : ℝ), (786 * x) / 30 = 1938.8 → x = 74 :=
by 
  sorry

end find_unknown_number_l136_136605


namespace inverse_ratio_l136_136916

theorem inverse_ratio (a b c d : ℝ) :
  (∀ x, x ≠ -6 → (3 * x - 2) / (x + 6) = (a * x + b) / (c * x + d)) →
  a/c = -6 :=
by
  sorry

end inverse_ratio_l136_136916


namespace circumradius_of_triangle_l136_136599

theorem circumradius_of_triangle (a b c : ℝ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 14) : 
  R = (35 * Real.sqrt 2) / 3 :=
by
  sorry

end circumradius_of_triangle_l136_136599


namespace drawing_red_ball_is_certain_l136_136059

def certain_event (balls : List String) : Prop :=
  ∀ ball ∈ balls, ball = "red"

theorem drawing_red_ball_is_certain:
  certain_event ["red", "red", "red", "red", "red"] :=
by
  sorry

end drawing_red_ball_is_certain_l136_136059


namespace Jerry_average_speed_l136_136531

variable (J : ℝ) -- Jerry's average speed in miles per hour
variable (C : ℝ) -- Carla's average speed in miles per hour
variable (T_J : ℝ) -- Time Jerry has been driving in hours
variable (T_C : ℝ) -- Time Carla has been driving in hours
variable (D : ℝ) -- Distance covered in miles

-- Given conditions
axiom Carla_speed : C = 35
axiom Carla_time : T_C = 3
axiom Jerry_time : T_J = T_C + 0.5

-- Distance covered by Carla in T_C hours at speed C
axiom Carla_distance : D = C * T_C

-- Distance covered by Jerry in T_J hours at speed J
axiom Jerry_distance : D = J * T_J

-- The goal to prove
theorem Jerry_average_speed : J = 30 :=
by
  sorry

end Jerry_average_speed_l136_136531


namespace find_abscissa_of_P_l136_136400

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem find_abscissa_of_P (x_P : ℝ) :
  (x + 2*y - 1 = 0 -> 
  (f' x_P = 2 -> 
  (f x_P - 2) * (x_P^2 - 1) = 0)) := by
  sorry

end find_abscissa_of_P_l136_136400


namespace trivia_team_original_members_l136_136078

theorem trivia_team_original_members (x : ℕ) (h1 : 6 * (x - 2) = 18) : x = 5 :=
by
  sorry

end trivia_team_original_members_l136_136078


namespace nina_jerome_age_ratio_l136_136103

variable (N J L : ℕ)

theorem nina_jerome_age_ratio (h1 : L = N - 4) (h2 : L + N + J = 36) (h3 : L = 6) : N / J = 1 / 2 := by
  sorry

end nina_jerome_age_ratio_l136_136103


namespace perimeter_of_square_is_64_l136_136493

noncomputable def side_length_of_square (s : ℝ) :=
  let rect_height := s
  let rect_width := s / 4
  let perimeter_of_rectangle := 2 * (rect_height + rect_width)
  perimeter_of_rectangle = 40

theorem perimeter_of_square_is_64 (s : ℝ) (h1 : side_length_of_square s) : 4 * s = 64 :=
by
  sorry

end perimeter_of_square_is_64_l136_136493


namespace complex_numbers_right_triangle_l136_136224

theorem complex_numbers_right_triangle (z : ℂ) (hz : z ≠ 0) :
  (∃ z₁ z₂ : ℂ, z₁ ≠ 0 ∧ z₂ ≠ 0 ∧ z₁^3 = z₂ ∧
                 (∃ θ₁ θ₂ : ℝ, z₁ = Complex.exp (Complex.I * θ₁) ∧
                               z₂ = Complex.exp (Complex.I * θ₂) ∧
                               (θ₂ - θ₁ = π/2 ∨ θ₂ - θ₁ = 3 * π/2))) →
  ∃ n : ℕ, n = 2 :=
by
  sorry

end complex_numbers_right_triangle_l136_136224


namespace terminating_decimal_l136_136116

theorem terminating_decimal : (45 / (2^2 * 5^3) : ℚ) = 0.090 :=
by
  sorry

end terminating_decimal_l136_136116


namespace smallest_area_of_right_triangle_l136_136449

noncomputable def right_triangle_area (a b : ℝ) : ℝ :=
  if a^2 + b^2 = 6^2 then (1/2) * a * b else 12

theorem smallest_area_of_right_triangle :
  min (right_triangle_area 4 (2 * Real.sqrt 5)) 12 = 4 * Real.sqrt 5 :=
by
  -- Proof goes here
  sorry

end smallest_area_of_right_triangle_l136_136449


namespace soybeans_to_oil_kg_l136_136408

-- Define initial data
def kgSoybeansToTofu : ℕ := 3
def kgSoybeansToOil : ℕ := 6
def kgTofuCostPerKg : ℕ := 3
def kgOilCostPerKg : ℕ := 15
def batchSoybeansKg : ℕ := 460
def totalRevenue : ℕ := 1800

-- Define problem statement
theorem soybeans_to_oil_kg (x y : ℕ) (h : x + y = batchSoybeansKg) 
  (hRevenue : 3 * kgTofuCostPerKg * x + (kgOilCostPerKg * y) / (kgSoybeansToOil) = totalRevenue) : 
  y = 360 :=
sorry

end soybeans_to_oil_kg_l136_136408


namespace min_total_bananas_l136_136836

noncomputable def total_bananas_condition (b1 b2 b3 : ℕ) : Prop :=
  let m1 := (5/8 : ℚ) * b1 + (5/16 : ℚ) * b2 + (23/48 : ℚ) * b3
  let m2 := (3/16 : ℚ) * b1 + (3/8 : ℚ) * b2 + (23/48 : ℚ) * b3
  let m3 := (3/16 : ℚ) * b1 + (5/16 : ℚ) * b2 + (1/24 : ℚ) * b3
  (((m1 : ℚ) * 4) = ((m2 : ℚ) * 3)) ∧ (((m1 : ℚ) * 4) = ((m3 : ℚ) * 2))

theorem min_total_bananas : ∃ (b1 b2 b3 : ℕ), b1 + b2 + b3 = 192 ∧ total_bananas_condition b1 b2 b3 :=
sorry

end min_total_bananas_l136_136836


namespace factorable_iff_m_eq_2_l136_136612

theorem factorable_iff_m_eq_2 (m : ℤ) :
  (∃ (A B C D : ℤ), (x y : ℤ) -> (x^2 + 2*x*y + 2*x + m*y + 2*m = (x + A*y + B) * (x + C*y + D))) ↔ m = 2 :=
sorry

end factorable_iff_m_eq_2_l136_136612


namespace flashes_in_fraction_of_hour_l136_136484

-- Definitions for the conditions
def flash_interval : ℕ := 6       -- The light flashes every 6 seconds
def hour_in_seconds : ℕ := 3600 -- There are 3600 seconds in an hour
def fraction_of_hour : ℚ := 3/4 -- ¾ of an hour

-- The translated proof problem statement in Lean
theorem flashes_in_fraction_of_hour (interval : ℕ) (sec_in_hour : ℕ) (fraction : ℚ) :
  interval = flash_interval →
  sec_in_hour = hour_in_seconds →
  fraction = fraction_of_hour →
  (fraction * sec_in_hour) / interval = 450 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end flashes_in_fraction_of_hour_l136_136484


namespace proof_f_f_f_3_l136_136404

def f (n : ℤ) : ℤ :=
  if n < 5
  then n^2 + 1
  else 2 * n - 3

theorem proof_f_f_f_3 :
  f (f (f 3)) = 31 :=
by 
  -- Here, we skip the proof as instructed
  sorry

end proof_f_f_f_3_l136_136404


namespace rectangle_area_l136_136643

theorem rectangle_area (a b c d : ℝ) 
  (ha : a = 4) 
  (hb : b = 4) 
  (hc : c = 4) 
  (hd : d = 1) :
  ∃ E F G H : ℝ,
    (E = 0 ∧ F = 3 ∧ G = 4 ∧ H = 0) →
    (a + b + c + d) = 10 :=
by
  intros
  sorry

end rectangle_area_l136_136643


namespace part1_correct_part2_correct_l136_136411

-- Definitions for conditions
def total_students := 200
def likes_employment := 140
def dislikes_employment := 60
def p_likes : ℚ := likes_employment / total_students

def male_likes := 60
def male_dislikes := 40
def female_likes := 80
def female_dislikes := 20
def n := total_students
def alpha := 0.005
def chi_squared_critical_value := 7.879

-- Part 1: Estimate the probability of selecting at least 2 students who like employment
def probability_at_least_2_of_3 : ℚ :=
  3 * ((7/10) ^ 2) * (3/10) + ((7/10) ^ 3)

-- Proof goal for Part 1
theorem part1_correct : probability_at_least_2_of_3 = 98 / 125 := by
  sorry

-- Part 2: Chi-squared test for independence between intention and gender
def a := male_likes
def b := male_dislikes
def c := female_likes
def d := female_dislikes
def chi_squared_statistic : ℚ :=
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Proof goal for Part 2
theorem part2_correct : chi_squared_statistic = 200 / 21 ∧ 200 / 21 > chi_squared_critical_value := by
  sorry

end part1_correct_part2_correct_l136_136411


namespace probability_calculation_l136_136216

noncomputable def probability_same_color (pairs_black pairs_brown pairs_gray : ℕ) : ℚ :=
  let total_shoes := 2 * (pairs_black + pairs_brown + pairs_gray)
  let prob_black := (2 * pairs_black : ℚ) / total_shoes * (pairs_black : ℚ) / (total_shoes - 1)
  let prob_brown := (2 * pairs_brown : ℚ) / total_shoes * (pairs_brown : ℚ) / (total_shoes - 1)
  let prob_gray := (2 * pairs_gray : ℚ) / total_shoes * (pairs_gray : ℚ) / (total_shoes - 1)
  prob_black + prob_brown + prob_gray

theorem probability_calculation :
  probability_same_color 7 4 3 = 37 / 189 :=
by
  sorry

end probability_calculation_l136_136216


namespace jessica_borrowed_amount_l136_136268

def payment_pattern (hour : ℕ) : ℕ :=
  match (hour % 6) with
  | 1 => 2
  | 2 => 4
  | 3 => 6
  | 4 => 8
  | 5 => 10
  | _ => 12

def total_payment (hours_worked : ℕ) : ℕ :=
  (hours_worked / 6) * 42 + (List.sum (List.map payment_pattern (List.range (hours_worked % 6))))

theorem jessica_borrowed_amount :
  total_payment 45 = 306 :=
by
  -- Proof omitted
  sorry

end jessica_borrowed_amount_l136_136268


namespace number_913n_divisible_by_18_l136_136528

theorem number_913n_divisible_by_18 (n : ℕ) (h1 : 9130 % 2 = 0) (h2 : (9 + 1 + 3 + n) % 9 = 0) : n = 8 :=
by
  sorry

end number_913n_divisible_by_18_l136_136528


namespace area_triangle_le_quarter_l136_136870

theorem area_triangle_le_quarter (S : ℝ) (S₁ S₂ S₃ S₄ S₅ S₆ S₇ : ℝ)
  (h₁ : S₃ + (S₂ + S₇) = S / 2)
  (h₂ : S₁ + S₆ + (S₂ + S₇) = S / 2) :
  S₁ ≤ S / 4 :=
by
  -- Proof skipped
  sorry

end area_triangle_le_quarter_l136_136870


namespace solution_set_inequality_l136_136436

theorem solution_set_inequality (x : ℝ) (h1 : 2 < 1 / (x - 1)) (h2 : 1 / (x - 1) < 3) (h3 : x - 1 > 0) :
  4 / 3 < x ∧ x < 3 / 2 :=
sorry

end solution_set_inequality_l136_136436


namespace henry_twice_jill_years_ago_l136_136245

def henry_age : ℕ := 23
def jill_age : ℕ := 17
def sum_of_ages (H J : ℕ) : Prop := H + J = 40

theorem henry_twice_jill_years_ago (H J : ℕ) (H1 : sum_of_ages H J) (H2 : H = 23) (H3 : J = 17) : ∃ x : ℕ, H - x = 2 * (J - x) ∧ x = 11 := 
by
  sorry

end henry_twice_jill_years_ago_l136_136245


namespace race_result_l136_136096

-- Definitions based on conditions
variable (hare_won : Bool)
variable (fox_second : Bool)
variable (hare_second : Bool)
variable (moose_first : Bool)

-- Condition that each squirrel had one error.
axiom owl_statement : xor hare_won fox_second ∧ xor hare_second moose_first

-- The final proof problem
theorem race_result : moose_first = true ∧ fox_second = true :=
by {
  -- Proving based on the owl's statement that each squirrel had one error
  sorry
}

end race_result_l136_136096


namespace ninety_one_square_friendly_unique_square_friendly_l136_136172

-- Given conditions
def square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, ∃ n : ℤ, m^2 + 18 * m + c = n^2

-- Part (a)
theorem ninety_one_square_friendly : square_friendly 81 :=
sorry

-- Part (b)
theorem unique_square_friendly (c c' : ℤ) (h_c : square_friendly c) (h_c' : square_friendly c') : c = c' :=
sorry

end ninety_one_square_friendly_unique_square_friendly_l136_136172


namespace maximum_possible_shortest_piece_length_l136_136885

theorem maximum_possible_shortest_piece_length :
  ∃ (A B C D E : ℝ), A ≤ B ∧ B ≤ C ∧ C ≤ D ∧ D ≤ E ∧ 
  C = 140 ∧ (A + B + C + D + E = 640) ∧ A = 80 :=
by
  sorry

end maximum_possible_shortest_piece_length_l136_136885


namespace mn_sum_l136_136757

theorem mn_sum {m n : ℤ} (h : ∀ x : ℤ, (x + 8) * (x - 1) = x^2 + m * x + n) : m + n = -1 :=
by
  sorry

end mn_sum_l136_136757


namespace complement_of_angle_l136_136748

def complement_angle (deg : ℕ) (min : ℕ) : ℕ × ℕ :=
  if deg < 90 then 
    let total_min := (90 * 60)
    let angle_min := (deg * 60) + min
    let comp_min := total_min - angle_min
    (comp_min / 60, comp_min % 60) -- degrees and remaining minutes
  else 
    (0, 0) -- this case handles if the angle is not less than complement allowable range

-- Definitions based on the problem
def given_angle_deg : ℕ := 57
def given_angle_min : ℕ := 13

-- Complement calculation
def comp (deg : ℕ) (min : ℕ) : ℕ × ℕ := complement_angle deg min

-- Expected result of the complement
def expected_comp : ℕ × ℕ := (32, 47)

-- Theorem to prove the complement of 57°13' is 32°47'
theorem complement_of_angle : comp given_angle_deg given_angle_min = expected_comp := by
  sorry

end complement_of_angle_l136_136748


namespace x_squared_minus_y_squared_l136_136087

theorem x_squared_minus_y_squared
    (x y : ℚ) 
    (h1 : x + y = 3 / 8) 
    (h2 : x - y = 1 / 4) : x^2 - y^2 = 3 / 32 := 
by 
    sorry

end x_squared_minus_y_squared_l136_136087


namespace problem_xy_squared_and_product_l136_136221

theorem problem_xy_squared_and_product (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) :
  x^2 - y^2 = 80 ∧ x * y = 96 :=
by
  sorry

end problem_xy_squared_and_product_l136_136221


namespace cube_less_than_three_times_square_l136_136888

theorem cube_less_than_three_times_square (x : ℤ) : x^3 < 3 * x^2 → x = 1 ∨ x = 2 :=
by
  sorry

end cube_less_than_three_times_square_l136_136888


namespace quadratic_real_roots_l136_136527

theorem quadratic_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 1 ∧ a ≠ 0) :=
by
  sorry

end quadratic_real_roots_l136_136527


namespace solve_for_x_l136_136019

theorem solve_for_x : ∃ x : ℚ, 5 * (2 * x - 3) = 3 * (3 - 4 * x) + 15 ∧ x = (39 : ℚ) / 22 :=
by
  use (39 : ℚ) / 22
  sorry

end solve_for_x_l136_136019


namespace solve_system_l136_136770

theorem solve_system (x y z : ℝ) :
  x^2 = y^2 + z^2 ∧
  x^2024 = y^2024 + z^2024 ∧
  x^2025 = y^2025 + z^2025 ↔
  (y = x ∧ z = 0) ∨
  (y = -x ∧ z = 0) ∨
  (y = 0 ∧ z = x) ∨
  (y = 0 ∧ z = -x) :=
by {
  sorry -- The detailed proof will be filled here.
}

end solve_system_l136_136770


namespace binom_8_3_eq_56_l136_136029

def binom (n k : ℕ) : ℕ :=
(n.factorial) / (k.factorial * (n - k).factorial)

theorem binom_8_3_eq_56 : binom 8 3 = 56 := by
  sorry

end binom_8_3_eq_56_l136_136029


namespace running_time_square_field_l136_136747

theorem running_time_square_field
  (side : ℕ)
  (running_speed_kmh : ℕ)
  (perimeter : ℕ := 4 * side)
  (running_speed_ms : ℕ := (running_speed_kmh * 1000) / 3600)
  (time : ℕ := perimeter / running_speed_ms) 
  (h_side : side = 35)
  (h_speed : running_speed_kmh = 9) :
  time = 56 := 
by
  sorry

end running_time_square_field_l136_136747


namespace no_positive_integers_exist_l136_136630

theorem no_positive_integers_exist 
  (a b c d : ℕ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (d_pos : 0 < d)
  (h₁ : a * b = c * d)
  (p : ℕ) 
  (hp : Nat.Prime p)
  (h₂ : a + b + c + d = p) : 
  False := 
by
  sorry

end no_positive_integers_exist_l136_136630


namespace range_of_a_l136_136491

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x - a
noncomputable def g (x : ℝ) : ℝ := 2*x + 2 * Real.log x
noncomputable def h (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x y, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ (1 / Real.exp 1) ≤ y ∧ y ≤ Real.exp 1 ∧ f x a = g x ∧ f y a = g y → x ≠ y) →
  1 < a ∧ a ≤ (1 / Real.exp 2) + 2 :=
sorry

end range_of_a_l136_136491


namespace find_d_l136_136595

theorem find_d (d : ℤ) :
  (∀ x : ℤ, 6 * x^3 + 19 * x^2 + d * x - 15 = 0) ->
  d = -32 :=
by
  sorry

end find_d_l136_136595


namespace find_pairs_l136_136969

noncomputable def f (k : ℤ) (x y : ℝ) : ℝ :=
  if k = 0 then 0 else (x^k + y^k + (-1)^k * (x + y)^k) / k

theorem find_pairs (x y : ℝ) (hxy : x ≠ 0 ∧ y ≠ 0 ∧ x + y ≠ 0) :
  ∃ (m n : ℤ), m ≠ 0 ∧ n ≠ 0 ∧ m ≤ n ∧ m + n ≠ 0 ∧ 
    (∀ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x + y ≠ 0 → f m x y * f n x y = f (m + n) x y) :=
  sorry

end find_pairs_l136_136969


namespace kelly_games_l136_136654

theorem kelly_games (initial_games give_away in_stock : ℕ) (h1 : initial_games = 50) (h2 : in_stock = 35) :
  give_away = initial_games - in_stock :=
by {
  -- initial_games = 50
  -- in_stock = 35
  -- Therefore, give_away = initial_games - in_stock
  sorry
}

end kelly_games_l136_136654


namespace mean_of_remaining_students_l136_136819

noncomputable def mean_remaining_students (k : ℕ) (h : k > 18) (mean_class : ℚ) (mean_18_students : ℚ) : ℚ :=
  (12 * k - 360) / (k - 18)

theorem mean_of_remaining_students (k : ℕ) (h : k > 18) (mean_class_eq : mean_class = 12) (mean_18_eq : mean_18_students = 20) :
  mean_remaining_students k h mean_class mean_18_students = (12 * k - 360) / (k - 18) :=
by sorry

end mean_of_remaining_students_l136_136819


namespace score_not_possible_l136_136214

theorem score_not_possible (c u i : ℕ) (score : ℤ) :
  c + u + i = 25 ∧ score = 79 → score ≠ 5 * c + 3 * u - 25 := by
  intro h
  sorry

end score_not_possible_l136_136214


namespace new_cost_after_decrease_l136_136708

theorem new_cost_after_decrease (C new_C : ℝ) (hC : C = 1100) (h_decrease : new_C = 0.76 * C) : new_C = 836 :=
-- To be proved based on the given conditions
sorry

end new_cost_after_decrease_l136_136708


namespace exp_gt_one_iff_a_gt_one_l136_136907

theorem exp_gt_one_iff_a_gt_one (a : ℝ) : 
  (∀ x : ℝ, 0 < x → a^x > 1) ↔ a > 1 :=
by
  sorry

end exp_gt_one_iff_a_gt_one_l136_136907


namespace intersection_eq_l136_136804

variable (A : Set ℤ) (B : Set ℤ)

def A_def := A = {-1, 0, 1, 2}
def B_def := B = {x | -1 < x ∧ x < 2}

theorem intersection_eq : A ∩ B = {0, 1} :=
by
  have A_def : A = {-1, 0, 1, 2} := sorry
  have B_def : B = {x | -1 < x ∧ x < 2} := sorry
  sorry

end intersection_eq_l136_136804


namespace total_money_l136_136497

theorem total_money (A B C : ℕ) (h1 : A + C = 400) (h2 : B + C = 750) (hC : C = 250) :
  A + B + C = 900 :=
sorry

end total_money_l136_136497


namespace triangle_cos_identity_l136_136382

variable {A B C : ℝ} -- Angle A, B, C are real numbers representing the angles of the triangle
variable {a b c : ℝ} -- Sides a, b, c are real numbers representing the lengths of the sides of the triangle

theorem triangle_cos_identity (h : 2 * b = a + c) : 5 * (Real.cos A) - 4 * (Real.cos A) * (Real.cos C) + 5 * (Real.cos C) = 4 :=
by
  sorry

end triangle_cos_identity_l136_136382


namespace find_k_l136_136666

def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (k : ℤ)
  (h1 : f a b c 1 = 0)
  (h2 : 60 < f a b c 9 ∧ f a b c 9 < 70)
  (h3 : 90 < f a b c 10 ∧ f a b c 10 < 100)
  (h4 : ∃ k : ℤ, 10000 * k < f a b c 100 ∧ f a b c 100 < 10000 * (k + 1))
  : k = 2 :=
sorry

end find_k_l136_136666


namespace find_a5_l136_136160

theorem find_a5 (a : ℕ → ℤ)
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = a n - 1) 
  (h2 : a 2 + a 4 + a 6 = 18) : 
  a 5 = 5 :=
sorry

end find_a5_l136_136160


namespace divisible_by_5_l136_136272

theorem divisible_by_5 (n : ℕ) : (∃ k : ℕ, 2^n - 1 = 5 * k) ∨ (∃ k : ℕ, 2^n + 1 = 5 * k) ∨ (∃ k : ℕ, 2^(2*n) + 1 = 5 * k) :=
sorry

end divisible_by_5_l136_136272


namespace divisible_by_1989_l136_136067

theorem divisible_by_1989 (n : ℕ) : 
  1989 ∣ (13 * (-50)^n + 17 * 40^n - 30) :=
by
  sorry

end divisible_by_1989_l136_136067


namespace ducks_in_smaller_pond_l136_136446

theorem ducks_in_smaller_pond (x : ℝ) (h1 : 50 > 0) 
  (h2 : 0.20 * x > 0) (h3 : 0.12 * 50 > 0) (h4 : 0.15 * (x + 50) = 0.20 * x + 0.12 * 50) 
  : x = 30 := 
sorry

end ducks_in_smaller_pond_l136_136446


namespace abs_add_lt_abs_sub_l136_136684

variable {a b : ℝ}

theorem abs_add_lt_abs_sub (h1 : a * b < 0) : |a + b| < |a - b| :=
sorry

end abs_add_lt_abs_sub_l136_136684


namespace increasing_function_range_l136_136818

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 4*a*x else (2*a + 3)*x - 4*a + 5

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (1/2 ≤ a ∧ a ≤ 3/2) :=
sorry

end increasing_function_range_l136_136818


namespace geometric_seq_sum_first_4_terms_l136_136347

theorem geometric_seq_sum_first_4_terms (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n * 2)
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  S 4 = 15 :=
by
  -- The actual proof would go here.
  sorry

end geometric_seq_sum_first_4_terms_l136_136347


namespace problem1_l136_136100

theorem problem1 (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) := by 
  sorry

end problem1_l136_136100


namespace altitudes_reciprocal_sum_eq_reciprocal_inradius_l136_136646

theorem altitudes_reciprocal_sum_eq_reciprocal_inradius
  (h1 h2 h3 r : ℝ)
  (h1_pos : h1 > 0) 
  (h2_pos : h2 > 0)
  (h3_pos : h3 > 0)
  (r_pos : r > 0)
  (triangle_area_eq : ∀ (a b c : ℝ),
    a * h1 = b * h2 ∧ b * h2 = c * h3 ∧ a + b + c > 0) :
  1 / h1 + 1 / h2 + 1 / h3 = 1 / r := 
by
  sorry

end altitudes_reciprocal_sum_eq_reciprocal_inradius_l136_136646


namespace last_three_digits_7_pow_80_l136_136782

theorem last_three_digits_7_pow_80 : (7^80) % 1000 = 961 := by
  sorry

end last_three_digits_7_pow_80_l136_136782


namespace exists_positive_integer_m_such_that_sqrt_8m_is_integer_l136_136886

theorem exists_positive_integer_m_such_that_sqrt_8m_is_integer :
  ∃ (m : ℕ), m > 0 ∧ ∃ (k : ℕ), 8 * m = k^2 :=
by
  sorry

end exists_positive_integer_m_such_that_sqrt_8m_is_integer_l136_136886


namespace point_c_third_quadrant_l136_136647

variable (a b : ℝ)

-- Definition of the conditions
def condition_1 : Prop := b = -1
def condition_2 : Prop := a = -3

-- Definition to check if a point is in the third quadrant
def is_third_quadrant (a b : ℝ) : Prop := a < 0 ∧ b < 0

-- The main statement to be proven
theorem point_c_third_quadrant (h1 : condition_1 b) (h2 : condition_2 a) :
  is_third_quadrant a b :=
by
  -- Proof of the theorem (to be completed)
  sorry

end point_c_third_quadrant_l136_136647


namespace basketball_game_points_l136_136319

variable (J T K : ℕ)

theorem basketball_game_points (h1 : T = J + 20) (h2 : J + T + K = 100) (h3 : T = 30) : 
  T / K = 1 / 2 :=
by sorry

end basketball_game_points_l136_136319


namespace tiling_implies_divisibility_l136_136758

def is_divisible_by (a b : Nat) : Prop := ∃ k : Nat, a = k * b

noncomputable def can_be_tiled (m n a b : Nat) : Prop :=
  a * b > 0 ∧ -- positivity condition for rectangle dimensions
  (∃ f_horiz : Fin (a * b) → Fin m, 
   ∃ g_vert : Fin (a * b) → Fin n, 
   True) -- A placeholder to denote tiling condition.

theorem tiling_implies_divisibility (m n a b : Nat)
  (hmn_pos : 0 < m ∧ 0 < n ∧ 0 < a ∧ 0 < b)
  (h_tiling : can_be_tiled m n a b) :
  is_divisible_by a m ∨ is_divisible_by b n :=
by
  sorry

end tiling_implies_divisibility_l136_136758


namespace min_value_l136_136083

open Real

noncomputable def func (x y z : ℝ) : ℝ := 1 / x + 1 / y + 1 / z

theorem min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) :
  func x y z ≥ 4.5 :=
by
  sorry

end min_value_l136_136083


namespace smallest_four_digit_multiple_of_18_l136_136815

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l136_136815


namespace find_b_l136_136779

theorem find_b (b : ℝ) :
  (∃ x1 x2 : ℝ, (x1 + x2 = -2) ∧
    ((x1 + 1)^3 + x1 / (x1 + 1) = -x1 + b) ∧
    ((x2 + 1)^3 + x2 / (x2 + 1) = -x2 + b)) →
  b = 0 :=
by
  sorry

end find_b_l136_136779


namespace quadratic_roots_sum_product_l136_136354

theorem quadratic_roots_sum_product {p q : ℝ} 
  (h1 : p / 3 = 10) 
  (h2 : q / 3 = 15) : 
  p + q = 75 := sorry

end quadratic_roots_sum_product_l136_136354


namespace adult_ticket_cost_is_16_l136_136006

-- Define the problem
def group_size := 6 + 10 -- Total number of people
def child_tickets := 6 -- Number of children
def adult_tickets := 10 -- Number of adults
def child_ticket_cost := 10 -- Cost per child ticket
def total_ticket_cost := 220 -- Total cost for all tickets

-- Prove the cost of an adult ticket
theorem adult_ticket_cost_is_16 : 
  (total_ticket_cost - (child_tickets * child_ticket_cost)) / adult_tickets = 16 := by
  sorry

end adult_ticket_cost_is_16_l136_136006


namespace perfect_square_difference_of_solutions_l136_136523

theorem perfect_square_difference_of_solutions
  (x y : ℤ) (hx : x > 0) (hy : y > 0) 
  (h : 3 * x^2 + x = 4 * y^2 + y) : ∃ k : ℤ, k^2 = x - y := 
sorry

end perfect_square_difference_of_solutions_l136_136523


namespace wooden_toy_price_l136_136426

noncomputable def price_of_hat : ℕ := 10
noncomputable def total_money : ℕ := 100
noncomputable def hats_bought : ℕ := 3
noncomputable def change_received : ℕ := 30
noncomputable def total_spent := total_money - change_received
noncomputable def cost_of_hats := hats_bought * price_of_hat

theorem wooden_toy_price :
  ∃ (W : ℕ), total_spent = 2 * W + cost_of_hats ∧ W = 20 := 
by 
  sorry

end wooden_toy_price_l136_136426


namespace inequality_solution_maximum_expression_l136_136686

-- Problem 1: Inequality for x
theorem inequality_solution (x : ℝ) : |x + 1| + 2 * |x - 1| < 3 * x + 5 ↔ x > -1/2 :=
by
  sorry

-- Problem 2: Maximum value for expression within [0, 1]
theorem maximum_expression (a b : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) : 
  ab + (1 - a - b) * (a + b) ≤ 1/3 :=
by
  sorry

end inequality_solution_maximum_expression_l136_136686


namespace quadratic_to_vertex_form_l136_136478

theorem quadratic_to_vertex_form :
  ∀ (x : ℝ), (1/2) * x^2 - 2 * x + 1 = (1/2) * (x - 2)^2 - 1 :=
by
  intro x
  -- full proof omitted
  sorry

end quadratic_to_vertex_form_l136_136478


namespace relationship_between_abc_l136_136617

theorem relationship_between_abc (u v a b c : ℝ)
  (h1 : u - v = a) 
  (h2 : u^2 - v^2 = b)
  (h3 : u^3 - v^3 = c) : 
  3 * b ^ 2 + a ^ 4 = 4 * a * c :=
sorry

end relationship_between_abc_l136_136617


namespace larger_number_l136_136676

variables (x y : ℕ)

theorem larger_number (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
sorry

end larger_number_l136_136676


namespace man_l136_136280

noncomputable def speed_in_still_water (current_speed_kmph : ℝ) (distance_m : ℝ) (time_seconds : ℝ) : ℝ :=
   let current_speed_mps := current_speed_kmph * 1000 / 3600
   let downstream_speed_mps := distance_m / time_seconds
   let still_water_speed_mps := downstream_speed_mps - current_speed_mps
   let still_water_speed_kmph := still_water_speed_mps * 3600 / 1000
   still_water_speed_kmph

theorem man's_speed_in_still_water :
  speed_in_still_water 6 100 14.998800095992323 = 18 := by
  sorry

end man_l136_136280


namespace prove_a_minus_c_l136_136194

-- Define the given conditions as hypotheses
def condition1 (a b d : ℝ) : Prop := (a + d + b + d) / 2 = 80
def condition2 (b c d : ℝ) : Prop := (b + d + c + d) / 2 = 180

-- State the theorem to be proven
theorem prove_a_minus_c (a b c d : ℝ) (h1 : condition1 a b d) (h2 : condition2 b c d) : a - c = -200 :=
by
  sorry

end prove_a_minus_c_l136_136194


namespace angle_quadrant_l136_136438

def same_terminal_side (θ α : ℝ) (k : ℤ) : Prop :=
  θ = α + 360 * k

def in_first_quadrant (α : ℝ) : Prop :=
  0 < α ∧ α < 90

theorem angle_quadrant (θ : ℝ) (k : ℤ) (h : same_terminal_side θ 12 k) : in_first_quadrant 12 :=
  by
    sorry

end angle_quadrant_l136_136438


namespace lena_candy_bars_l136_136855

/-- Lena has some candy bars. She needs 5 more candy bars to have 3 times as many as Kevin,
and Kevin has 4 candy bars less than Nicole. Lena has 5 more candy bars than Nicole.
How many candy bars does Lena have? -/
theorem lena_candy_bars (L K N : ℕ) 
  (h1 : L + 5 = 3 * K)
  (h2 : K = N - 4)
  (h3 : L = N + 5) : 
  L = 16 :=
sorry

end lena_candy_bars_l136_136855


namespace balloon_arrangements_correct_l136_136824

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the number of ways to arrange "BALLOON"
noncomputable def arrangements_balloon : ℕ := factorial 7 / (factorial 2 * factorial 2)

-- State the theorem
theorem balloon_arrangements_correct : arrangements_balloon = 1260 := by sorry

end balloon_arrangements_correct_l136_136824


namespace well_depth_and_rope_length_l136_136589

theorem well_depth_and_rope_length (h x : ℝ) : 
  (h / 3 = x + 4) ∧ (h / 4 = x + 1) :=
sorry

end well_depth_and_rope_length_l136_136589


namespace cookies_and_sugar_needed_l136_136476

-- Definitions derived from the conditions
def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def initial_sugar : ℝ := 1.5
def flour_needed : ℕ := 5

-- The proof statement
theorem cookies_and_sugar_needed :
  (initial_cookies / initial_flour) * flour_needed = 40 ∧ (initial_sugar / initial_flour) * flour_needed = 2.5 :=
by
  sorry

end cookies_and_sugar_needed_l136_136476


namespace milk_production_l136_136890

theorem milk_production (a b c x y z w : ℕ) : 
  ((b:ℝ) / c) * w + ((y:ℝ) / z) * w = (bw / c) + (yw / z) := sorry

end milk_production_l136_136890


namespace bill_project_days_l136_136004

theorem bill_project_days (naps: ℕ) (hours_per_nap: ℕ) (working_hours: ℕ) : 
  (naps = 6) → (hours_per_nap = 7) → (working_hours = 54) → 
  (naps * hours_per_nap + working_hours) / 24 = 4 := 
by
  intros h1 h2 h3
  sorry

end bill_project_days_l136_136004


namespace length_of_bridge_l136_136918

-- Definitions based on the conditions
def walking_speed_kmph : ℝ := 10 -- speed in km/hr
def time_minutes : ℝ := 24 -- crossing time in minutes
def conversion_factor_km_to_m : ℝ := 1000
def conversion_factor_hr_to_min : ℝ := 60

-- The main statement to prove
theorem length_of_bridge :
  let walking_speed_m_per_min := walking_speed_kmph * conversion_factor_km_to_m / conversion_factor_hr_to_min;
  walking_speed_m_per_min * time_minutes = 4000 := 
by
  let walking_speed_m_per_min := walking_speed_kmph * conversion_factor_km_to_m / conversion_factor_hr_to_min;
  sorry

end length_of_bridge_l136_136918


namespace aftershave_alcohol_concentration_l136_136460

def initial_volume : ℝ := 12
def initial_concentration : ℝ := 0.60
def desired_concentration : ℝ := 0.40
def water_added : ℝ := 6
def final_volume : ℝ := initial_volume + water_added

theorem aftershave_alcohol_concentration :
  initial_concentration * initial_volume = desired_concentration * final_volume :=
by
  sorry

end aftershave_alcohol_concentration_l136_136460


namespace sum_of_0_75_of_8_and_2_l136_136667

theorem sum_of_0_75_of_8_and_2 : 0.75 * 8 + 2 = 8 := by
  sorry

end sum_of_0_75_of_8_and_2_l136_136667


namespace runway_show_time_correct_l136_136065

def runwayShowTime (bathing_suit_sets evening_wear_sets formal_wear_sets models trip_time_in_minutes : ℕ) : ℕ :=
  let trips_per_model := bathing_suit_sets + evening_wear_sets + formal_wear_sets
  let total_trips := models * trips_per_model
  total_trips * trip_time_in_minutes

theorem runway_show_time_correct :
  runwayShowTime 3 4 2 10 3 = 270 :=
by
  sorry

end runway_show_time_correct_l136_136065


namespace row_trip_time_example_l136_136395

noncomputable def round_trip_time
    (rowing_speed : ℝ)
    (current_speed : ℝ)
    (total_distance : ℝ) : ℝ :=
  let downstream_speed := rowing_speed + current_speed
  let upstream_speed := rowing_speed - current_speed
  let one_way_distance := total_distance / 2
  let time_to_place := one_way_distance / downstream_speed
  let time_back := one_way_distance / upstream_speed
  time_to_place + time_back

theorem row_trip_time_example :
  round_trip_time 10 2 96 = 10 := by
  sorry

end row_trip_time_example_l136_136395


namespace evaluate_expression_l136_136286

noncomputable def f : ℝ → ℝ := sorry

lemma f_condition (a : ℝ) : f (a + 1) = f a * f 1 := sorry

lemma f_one : f 1 = 2 := sorry

theorem evaluate_expression :
  (f 2018 / f 2017) + (f 2019 / f 2018) + (f 2020 / f 2019) = 6 :=
sorry

end evaluate_expression_l136_136286


namespace green_paint_mixture_l136_136026

theorem green_paint_mixture :
  ∀ (x : ℝ), 
    let light_green_paint := 5
    let darker_green_paint := x
    let final_paint := light_green_paint + darker_green_paint
    1 + 0.4 * darker_green_paint = 0.25 * final_paint -> x = 5 / 3 := 
by 
  intros x
  let light_green_paint := 5
  let darker_green_paint := x
  let final_paint := light_green_paint + darker_green_paint
  sorry

end green_paint_mixture_l136_136026


namespace triangular_stack_log_count_l136_136192

theorem triangular_stack_log_count : 
  ∀ (a₁ aₙ d : ℤ) (n : ℤ), a₁ = 15 → aₙ = 1 → d = -2 → 
  (a₁ - aₙ) / (-d) + 1 = n → 
  (n * (a₁ + aₙ)) / 2 = 64 :=
by
  intros a₁ aₙ d n h₁ hₙ hd hn
  sorry

end triangular_stack_log_count_l136_136192


namespace find_amount_with_R_l136_136604

variable (P_amount Q_amount R_amount : ℝ)
variable (total_amount : ℝ) (r_has_twothirds : Prop)

noncomputable def amount_with_R (total_amount : ℝ) : ℝ :=
  let R_amount := 2 / 3 * (total_amount - R_amount)
  R_amount

theorem find_amount_with_R (P_amount Q_amount R_amount : ℝ) (total_amount : ℝ)
  (h_total : total_amount = 5000)
  (h_two_thirds : R_amount = 2 / 3 * (P_amount + Q_amount)) :
  R_amount = 2000 := by sorry

end find_amount_with_R_l136_136604


namespace jasonPears_l136_136180

-- Define the conditions
def keithPears : Nat := 47
def mikePears : Nat := 12
def totalPears : Nat := 105

-- Define the theorem stating the number of pears Jason picked
theorem jasonPears : (totalPears - keithPears - mikePears) = 46 :=
by 
  sorry

end jasonPears_l136_136180


namespace unique_integral_root_l136_136088

theorem unique_integral_root {x : ℤ} :
  x - 12 / (x - 3) = 5 - 12 / (x - 3) ↔ x = 5 :=
by
  sorry

end unique_integral_root_l136_136088


namespace karens_speed_l136_136577

noncomputable def average_speed_karen (k : ℝ) : Prop :=
  let late_start_in_hours := 4 / 60
  let total_distance_karen := 24 + 4
  let time_karen := total_distance_karen / k
  let distance_tom_start := 45 * late_start_in_hours
  let distance_tom_total := distance_tom_start + 45 * time_karen
  distance_tom_total = 24

theorem karens_speed : average_speed_karen 60 :=
by
  sorry

end karens_speed_l136_136577


namespace jill_arrives_15_minutes_before_jack_l136_136424

theorem jill_arrives_15_minutes_before_jack
  (distance : ℝ) (jill_speed : ℝ) (jack_speed : ℝ) (start_same_time : true)
  (h_distance : distance = 2) (h_jill_speed : jill_speed = 8) (h_jack_speed : jack_speed = 4) :
  (2 / 4 * 60) - (2 / 8 * 60) = 15 :=
by
  sorry

end jill_arrives_15_minutes_before_jack_l136_136424


namespace karsyn_total_payment_l136_136120

-- Define the initial price of the phone
def initial_price : ℝ := 600

-- Define the discounted rate for the phone
def discount_rate_phone : ℝ := 0.20

-- Define the prices for additional items
def phone_case_price : ℝ := 25
def screen_protector_price : ℝ := 15

-- Define the discount rates
def discount_rate_125 : ℝ := 0.05
def discount_rate_150 : ℝ := 0.10
def final_discount_rate : ℝ := 0.03

-- Define the tax rate and fee
def exchange_rate_fee : ℝ := 0.02

noncomputable def total_payment (initial_price : ℝ) (discount_rate_phone : ℝ) 
  (phone_case_price : ℝ) (screen_protector_price : ℝ) (discount_rate_125 : ℝ) 
  (discount_rate_150 : ℝ) (final_discount_rate : ℝ) (exchange_rate_fee : ℝ) : ℝ :=
  let discounted_phone_price := initial_price * discount_rate_phone
  let additional_items_price := phone_case_price + screen_protector_price
  let total_before_discounts := discounted_phone_price + additional_items_price
  let total_after_first_discount := total_before_discounts * (1 - discount_rate_125)
  let total_after_second_discount := total_after_first_discount * (1 - discount_rate_150)
  let total_after_all_discounts := total_after_second_discount * (1 - final_discount_rate)
  let total_with_exchange_fee := total_after_all_discounts * (1 + exchange_rate_fee)
  total_with_exchange_fee

theorem karsyn_total_payment :
  total_payment initial_price discount_rate_phone phone_case_price screen_protector_price 
    discount_rate_125 discount_rate_150 final_discount_rate exchange_rate_fee = 135.35 := 
  by 
  -- Specify proof steps here
  sorry

end karsyn_total_payment_l136_136120


namespace contest_score_order_l136_136567

variables (E F G H : ℕ) -- nonnegative scores of Emily, Fran, Gina, and Harry respectively

-- Conditions
axiom cond1 : E - F = G + H + 10
axiom cond2 : G + E > F + H + 5
axiom cond3 : H = F + 8

-- Statement to prove
theorem contest_score_order : (H > E) ∧ (E > F) ∧ (F > G) :=
sorry

end contest_score_order_l136_136567


namespace max_intersections_l136_136931

/-- Given two different circles and three different straight lines, the maximum number of
points of intersection on a plane is 17. -/
theorem max_intersections (c1 c2 : Circle) (l1 l2 l3 : Line) (h_distinct_cir : c1 ≠ c2) (h_distinct_lines : ∀ (l1 l2 : Line), l1 ≠ l2) :
  ∃ (n : ℕ), n = 17 :=
by
  sorry

end max_intersections_l136_136931


namespace temperature_difference_l136_136908

theorem temperature_difference (T_high T_low : ℝ) (h_high : T_high = 9) (h_low : T_low = -1) : 
  T_high - T_low = 10 :=
by
  rw [h_high, h_low]
  norm_num

end temperature_difference_l136_136908


namespace discount_problem_l136_136996

theorem discount_problem (n : ℕ) : 
  (∀ x : ℝ, 0 < x → (1 - n / 100 : ℝ) * x < min (0.72 * x) (min (0.6724 * x) (0.681472 * x))) ↔ n ≥ 33 :=
by
  sorry

end discount_problem_l136_136996


namespace slope_of_tangent_at_1_0_l136_136310

noncomputable def f (x : ℝ) : ℝ :=
2 * x^2 - 2 * x

def derivative_f (x : ℝ) : ℝ :=
4 * x - 2

theorem slope_of_tangent_at_1_0 : derivative_f 1 = 2 :=
by
  sorry

end slope_of_tangent_at_1_0_l136_136310


namespace dice_roll_probability_l136_136803

theorem dice_roll_probability : 
  ∃ (m n : ℕ), (1 ≤ m ∧ m ≤ 6) ∧ (1 ≤ n ∧ n ≤ 6) ∧ (m - n > 0) ∧ 
  ( (15 : ℚ) / 36 = (5 : ℚ) / 12 ) :=
by {
  sorry
}

end dice_roll_probability_l136_136803


namespace probability_alpha_in_interval_l136_136069

def vector_of_die_rolls_angle_probability : ℚ := 
  let total_outcomes := 36
  let favorable_pairs := 15
  favorable_pairs / total_outcomes

theorem probability_alpha_in_interval (m n : ℕ)
  (hm : 1 ≤ m ∧ m ≤ 6) (hn : 1 ≤ n ∧ n ≤ 6) :
  (vector_of_die_rolls_angle_probability = 5 / 12) := by
  sorry

end probability_alpha_in_interval_l136_136069


namespace polynomial_division_l136_136899

open Polynomial

-- Define the theorem statement
theorem polynomial_division (f g : ℤ[X])
  (h : ∀ n : ℤ, f.eval n ∣ g.eval n) :
  ∃ (h : ℤ[X]), g = f * h :=
sorry

end polynomial_division_l136_136899


namespace book_arrangement_count_l136_136106

theorem book_arrangement_count :
  let total_books := 7
  let identical_math_books := 3
  let identical_physics_books := 2
  (Nat.factorial 7) / ((Nat.factorial 3) * (Nat.factorial 2)) = 420 := 
by
  sorry

end book_arrangement_count_l136_136106


namespace coordinates_of_A_equidistant_BC_l136_136762

theorem coordinates_of_A_equidistant_BC :
  ∃ z : ℚ, (∀ A B C : ℚ × ℚ × ℚ, A = (0, 0, z) ∧ B = (7, 0, -15) ∧ C = (2, 10, -12) →
  (dist A B = dist A C)) ↔ z = -(13/3) :=
by sorry

end coordinates_of_A_equidistant_BC_l136_136762


namespace find_value_of_expression_l136_136670

theorem find_value_of_expression
  (x y : ℝ)
  (h : x^2 - 2*x + y^2 - 6*y + 10 = 0) :
  x^2 * y^2 + 2 * x * y + 1 = 16 :=
sorry

end find_value_of_expression_l136_136670


namespace total_distance_covered_l136_136846

-- Define the distances for each segment of Biker Bob's journey
def distance1 : ℕ := 45 -- 45 miles west
def distance2 : ℕ := 25 -- 25 miles northwest
def distance3 : ℕ := 35 -- 35 miles south
def distance4 : ℕ := 50 -- 50 miles east

-- Statement to prove that the total distance covered is 155 miles
theorem total_distance_covered : distance1 + distance2 + distance3 + distance4 = 155 :=
by
  -- This is where the proof would go
  sorry

end total_distance_covered_l136_136846


namespace find_xy_l136_136545

-- Defining the initial conditions
variable (x y : ℕ)

-- Defining the rectangular prism dimensions and the volume equation
def prism_volume_original : ℕ := 15 * 5 * 4 -- Volume = 300
def remaining_volume : ℕ := 120

-- The main theorem statement to prove the conditions and their solution
theorem find_xy (h1 : prism_volume_original - 5 * y * x = remaining_volume)
    (h2 : x < 4) 
    (h3 : y < 15) : 
    x = 3 ∧ y = 12 := sorry

end find_xy_l136_136545


namespace certain_event_l136_136300

-- Definitions for a line and plane
inductive Line
| mk : Line

inductive Plane
| mk : Plane

-- Definitions for parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def plane_parallel (p₁ p₂ : Plane) : Prop := sorry

-- Given conditions and the proof statement
theorem certain_event (l : Line) (α β : Plane) (h1 : perpendicular l α) (h2 : perpendicular l β) : plane_parallel α β :=
sorry

end certain_event_l136_136300


namespace find_salary_B_l136_136613

def salary_A : ℕ := 8000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000
def avg_salary : ℕ := 8000

theorem find_salary_B (S_B : ℕ) :
  (salary_A + S_B + salary_C + salary_D + salary_E) / 5 = avg_salary ↔ S_B = 5000 := by
  sorry

end find_salary_B_l136_136613


namespace total_weight_correct_l136_136525

-- Define the weights given in the problem
def dog_weight_kg := 2 -- weight in kilograms
def dog_weight_g := 600 -- additional grams
def cat_weight_g := 3700 -- weight in grams

-- Convert dog's weight to grams
def dog_weight_total_g : ℕ := dog_weight_kg * 1000 + dog_weight_g

-- Define the total weight of the animals (dog + cat)
def total_weight_animals_g : ℕ := dog_weight_total_g + cat_weight_g

-- Theorem stating that the total weight of the animals is 6300 grams
theorem total_weight_correct : total_weight_animals_g = 6300 := by
  sorry

end total_weight_correct_l136_136525


namespace pam_walked_1683_miles_l136_136765

noncomputable def pam_miles_walked 
    (pedometer_limit : ℕ)
    (initial_reading : ℕ)
    (flips : ℕ)
    (final_reading : ℕ)
    (steps_per_mile : ℕ)
    : ℕ :=
  (pedometer_limit + 1) * flips + final_reading / steps_per_mile

theorem pam_walked_1683_miles
    (pedometer_limit : ℕ := 49999)
    (initial_reading : ℕ := 0)
    (flips : ℕ := 50)
    (final_reading : ℕ := 25000)
    (steps_per_mile : ℕ := 1500) 
    : pam_miles_walked pedometer_limit initial_reading flips final_reading steps_per_mile = 1683 := 
  sorry

end pam_walked_1683_miles_l136_136765


namespace min_value_expression_min_value_expression_achieved_at_1_l136_136881

noncomputable def min_value_expr (a b : ℝ) (n : ℕ) : ℝ :=
  (1 / (1 + a^n)) + (1 / (1 + b^n))

theorem min_value_expression (a b : ℝ) (n : ℕ) (h1 : a + b = 2) (h2 : 0 < a) (h3 : 0 < b) : 
  (min_value_expr a b n) ≥ 1 :=
sorry

theorem min_value_expression_achieved_at_1 (n : ℕ) :
  (min_value_expr 1 1 n = 1) :=
sorry

end min_value_expression_min_value_expression_achieved_at_1_l136_136881


namespace truth_probability_of_A_l136_136037

theorem truth_probability_of_A (P_B : ℝ) (P_AB : ℝ) (h : P_AB = 0.45 ∧ P_B = 0.60 ∧ ∀ (P_A : ℝ), P_AB = P_A * P_B) : 
  ∃ (P_A : ℝ), P_A = 0.75 :=
by
  sorry

end truth_probability_of_A_l136_136037


namespace solve_inequality_range_of_a_l136_136655

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Define the set A
def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }

-- First part: Solve the inequality f(x) ≤ 3a^2 + 1 when a ≠ 0
-- Solution would be translated in a theorem
theorem solve_inequality (a : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, f x a ≤ 3 * a^2 + 1 → if a > 0 then -a ≤ x ∧ x ≤ 3 * a else -3 * a ≤ x ∧ x ≤ a :=
sorry

-- Second part: Find the range of a if there exists no x0 ∈ A such that f(x0) ≤ A is false
theorem range_of_a (a : ℝ) :
  (∀ x ∈ A, f x a > 0) ↔ a < 1 :=
sorry

end solve_inequality_range_of_a_l136_136655


namespace p_suff_but_not_nec_q_l136_136603

variable (p q : Prop)

-- Given conditions: ¬p is a necessary but not sufficient condition for ¬q.
def neg_p_nec_but_not_suff_neg_q : Prop :=
  (¬q → ¬p) ∧ ¬(¬p → ¬q)

-- Concluding statement: p is a sufficient but not necessary condition for q.
theorem p_suff_but_not_nec_q 
  (h : neg_p_nec_but_not_suff_neg_q p q) : (p → q) ∧ ¬(q → p) := 
sorry

end p_suff_but_not_nec_q_l136_136603


namespace inner_cube_surface_area_l136_136161

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l136_136161


namespace xiao_ming_math_score_l136_136749

noncomputable def math_score (C M E : ℕ) : ℕ :=
  let A := 94
  let N := 3
  let total_score := A * N
  let T_CE := (A - 1) * 2
  total_score - T_CE

theorem xiao_ming_math_score (C M E : ℕ)
    (h1 : (C + M + E) / 3 = 94)
    (h2 : (C + E) / 2 = 93) :
  math_score C M E = 96 := by
  sorry

end xiao_ming_math_score_l136_136749


namespace total_minutes_ironing_over_4_weeks_l136_136781

/-- Define the time spent ironing each day -/
def minutes_ironing_per_day : Nat := 5 + 3

/-- Define the number of days Hayden irons per week -/
def days_ironing_per_week : Nat := 5

/-- Define the number of weeks considered -/
def number_of_weeks : Nat := 4

/-- The main theorem we're proving is that Hayden spends 160 minutes ironing over 4 weeks -/
theorem total_minutes_ironing_over_4_weeks :
  (minutes_ironing_per_day * days_ironing_per_week * number_of_weeks) = 160 := by
  sorry

end total_minutes_ironing_over_4_weeks_l136_136781


namespace find_some_number_l136_136473

theorem find_some_number (some_number q x y : ℤ) 
  (h1 : x = some_number + 2 * q) 
  (h2 : y = 4 * q + 41) 
  (h3 : q = 7) 
  (h4 : x = y) : 
  some_number = 55 := 
by 
  sorry

end find_some_number_l136_136473


namespace car_speed_l136_136863

theorem car_speed (time : ℕ) (distance : ℕ) (h1 : time = 5) (h2 : distance = 300) : distance / time = 60 := by
  sorry

end car_speed_l136_136863


namespace ruda_received_clock_on_correct_date_l136_136991

/-- Ruda's clock problem -/
def ruda_clock_problem : Prop :=
  ∃ receive_date : ℕ → ℕ × ℕ × ℕ, -- A function mapping the number of presses to a date (Year, Month, Day)
  (∀ days_after_received, 
    receive_date days_after_received = 
    if days_after_received <= 45 then (2022, 10, 27 - (45 - days_after_received)) -- Calculating the receive date.
    else receive_date 45)
  ∧
  receive_date 45 = (2022, 12, 11) -- The day he checked the clock has to be December 11th

-- We want to prove that:
theorem ruda_received_clock_on_correct_date : ruda_clock_problem :=
by
  sorry

end ruda_received_clock_on_correct_date_l136_136991


namespace integer_solutions_l136_136743

theorem integer_solutions :
  { (x, y) : ℤ × ℤ |
       y^2 + y = x^4 + x^3 + x^2 + x } =
  { (-1, -1), (-1, 0), (0, -1), (0, 0), (2, 5), (2, -6) } :=
by
  sorry

end integer_solutions_l136_136743


namespace solution_l136_136703

open Real

variables (a b c A B C : ℝ)

-- Condition: In ΔABC, the sides opposite to angles A, B, and C are a, b, and c respectively
-- Condition: Given equation relating sides and angles in ΔABC
axiom eq1 : a * sin C / (1 - cos A) = sqrt 3 * c
-- Condition: b + c = 10
axiom eq2 : b + c = 10
-- Condition: Area of ΔABC
axiom eq3 : (1 / 2) * b * c * sin A = 4 * sqrt 3

-- The final statement to prove
theorem solution :
    (A = π / 3) ∧ (a = 2 * sqrt 13) :=
by
    sorry

end solution_l136_136703


namespace cars_in_first_section_l136_136323

noncomputable def first_section_rows : ℕ := 15
noncomputable def first_section_cars_per_row : ℕ := 10
noncomputable def total_cars_first_section : ℕ := first_section_rows * first_section_cars_per_row

theorem cars_in_first_section : total_cars_first_section = 150 :=
by
  sorry

end cars_in_first_section_l136_136323


namespace difference_seven_three_times_l136_136181

theorem difference_seven_three_times (n : ℝ) (h1 : n = 3) 
  (h2 : 7 * n = 3 * n + (21.0 - 9.0)) :
  7 * n - 3 * n = 12.0 := by
  sorry

end difference_seven_three_times_l136_136181


namespace part1_solution_part2_solution_l136_136070

-- Definitions of the lines
def l1 (a : ℝ) : ℝ × ℝ × ℝ :=
  (2 * a + 1, a + 2, 3)

def l2 (a : ℝ) : ℝ × ℝ × ℝ :=
  (a - 1, -2, 2)

-- Parallel lines condition
def parallel_lines (a : ℝ) : Prop :=
  let (A1, B1, C1) := l1 a
  let (A2, B2, C2) := l2 a
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ (A1 / B1) = (A2 / B2)

-- Perpendicular lines condition
def perpendicular_lines (a : ℝ) : Prop :=
  let (A1, B1, C1) := l1 a
  let (A2, B2, C2) := l2 a
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ (A1 * A2 + B1 * B2 = 0)

-- Statement for part 1
theorem part1_solution (a : ℝ) : parallel_lines a ↔ a = 0 :=
  sorry

-- Statement for part 2
theorem part2_solution (a : ℝ) : perpendicular_lines a ↔ (a = -1 ∨ a = 5 / 2) :=
  sorry


end part1_solution_part2_solution_l136_136070


namespace total_dogs_in_kennel_l136_136560

-- Definition of the given conditions
def T := 45       -- Number of dogs that wear tags
def C := 40       -- Number of dogs that wear flea collars
def B := 6        -- Number of dogs that wear both tags and collars
def D_neither := 1 -- Number of dogs that wear neither a collar nor tags

-- Theorem statement
theorem total_dogs_in_kennel : T + C - B + D_neither = 80 := 
by
  -- Proof omitted
  sorry

end total_dogs_in_kennel_l136_136560


namespace parallel_lines_a_l136_136179
-- Import necessary libraries

-- Define the given conditions and the main statement
theorem parallel_lines_a (a : ℝ) :
  (∀ x y : ℝ, a * x + y - 2 = 0 → 3 * x + (a + 2) * y + 1 = 0) →
  (a = -3 ∨ a = 1) :=
by
  -- Place the proof here
  sorry

end parallel_lines_a_l136_136179


namespace aero_flight_tees_per_package_l136_136197

theorem aero_flight_tees_per_package {A : ℕ} :
  (∀ (num_people : ℕ), num_people = 4 → 20 * num_people ≤ A * 28 + 2 * 12) →
  A * 28 ≥ 56 →
  A = 2 :=
by
  intros h1 h2
  sorry

end aero_flight_tees_per_package_l136_136197


namespace inequality_div_two_l136_136619

theorem inequality_div_two (a b : ℝ) (h : a > b) : (a / 2) > (b / 2) :=
sorry

end inequality_div_two_l136_136619


namespace combined_average_age_l136_136220

noncomputable def roomA : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
noncomputable def roomB : Set ℕ := {11, 12, 13, 14}
noncomputable def average_age_A := 55
noncomputable def average_age_B := 35
noncomputable def total_people := (10 + 4)
noncomputable def total_age_A := 10 * average_age_A
noncomputable def total_age_B := 4 * average_age_B
noncomputable def combined_total_age := total_age_A + total_age_B

theorem combined_average_age :
  (combined_total_age / total_people : ℚ) = 49.29 :=
by sorry

end combined_average_age_l136_136220


namespace area_of_triangle_with_sides_13_12_5_l136_136512

theorem area_of_triangle_with_sides_13_12_5 :
  let a := 13
  let b := 12
  let c := 5
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = 30 :=
by
  let a := 13
  let b := 12
  let c := 5
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  sorry

end area_of_triangle_with_sides_13_12_5_l136_136512


namespace volleyball_team_lineup_l136_136878

theorem volleyball_team_lineup : 
  let team_members := 10
  let lineup_positions := 6
  10 * 9 * 8 * 7 * 6 * 5 = 151200 := by sorry

end volleyball_team_lineup_l136_136878


namespace olivias_dad_total_spending_l136_136265

def people : ℕ := 5
def meal_cost : ℕ := 12
def drink_cost : ℕ := 3
def dessert_cost : ℕ := 5

theorem olivias_dad_total_spending : 
  (people * meal_cost) + (people * drink_cost) + (people * dessert_cost) = 100 := 
by
  sorry

end olivias_dad_total_spending_l136_136265


namespace scientific_notation_of_coronavirus_diameter_l136_136341

theorem scientific_notation_of_coronavirus_diameter:
  0.000000907 = 9.07 * 10^(-7) :=
sorry

end scientific_notation_of_coronavirus_diameter_l136_136341


namespace error_difference_l136_136761

noncomputable def total_income_without_error (T: ℝ) : ℝ :=
  T + 110000

noncomputable def total_income_with_error (T: ℝ) : ℝ :=
  T + 1100000

noncomputable def mean_without_error (T: ℝ) : ℝ :=
  (T + 110000) / 500

noncomputable def mean_with_error (T: ℝ) : ℝ :=
  (T + 1100000) / 500

theorem error_difference (T: ℝ) :
  mean_with_error T - mean_without_error T = 1980 :=
by
  sorry

end error_difference_l136_136761


namespace sequence_general_term_l136_136982

theorem sequence_general_term 
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 1 / 3)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n * a (n - 1) + a n * a (n + 1) = 2 * a (n - 1) * a (n + 1)) :
  ∀ n : ℕ, 1 ≤ n → a n = 1 / (2 * n - 1) := 
by
  sorry

end sequence_general_term_l136_136982
