import Mathlib

namespace Teresa_age_when_Michiko_born_l161_161138

theorem Teresa_age_when_Michiko_born 
  (Teresa_age : ℕ) (Morio_age : ℕ) (Michiko_born_age : ℕ) (Kenji_diff : ℕ)
  (Emiko_diff : ℕ) (Hideki_same_as_Kenji : Prop) (Ryuji_age_same_as_Morio : Prop)
  (h1 : Teresa_age = 59) 
  (h2 : Morio_age = 71) 
  (h3 : Morio_age = Michiko_born_age + 33)
  (h4 : Kenji_diff = 4)
  (h5 : Emiko_diff = 10)
  (h6 : Hideki_same_as_Kenji = True)
  (h7 : Ryuji_age_same_as_Morio = True) : 
  ∃ Michiko_age Hideki_age Michiko_Hideki_diff Teresa_birth_age,
    Michiko_age = 33 ∧ 
    Hideki_age = 29 ∧ 
    Michiko_Hideki_diff = 4 ∧ 
    Teresa_birth_age = 26 :=
sorry

end Teresa_age_when_Michiko_born_l161_161138


namespace measure_angle_BCA_l161_161826

theorem measure_angle_BCA 
  (BCD_angle : ℝ)
  (CBA_angle : ℝ)
  (sum_angles : BCD_angle + CBA_angle + BCA_angle = 190)
  (BCD_right : BCD_angle = 90)
  (CBA_given : CBA_angle = 70) :
  BCA_angle = 30 :=
by
  sorry

end measure_angle_BCA_l161_161826


namespace most_pieces_day_and_maximum_number_of_popular_days_l161_161806

-- Definitions for conditions:
def a_n (n : ℕ) : ℕ :=
if h : n ≤ 13 then 3 * n
else 65 - 2 * n

def S_n (n : ℕ) : ℕ :=
if h : n ≤ 13 then (3 + 3 * n) * n / 2
else 273 + (51 - n) * (n - 13)

-- Propositions to prove:
theorem most_pieces_day_and_maximum :
  ∃ k a_k, (1 ≤ k ∧ k ≤ 31) ∧
           (a_k = a_n k) ∧
           (∀ n, 1 ≤ n ∧ n ≤ 31 → a_n n ≤ a_k) ∧
           k = 13 ∧ a_k = 39 := 
sorry

theorem number_of_popular_days :
  ∃ days_popular,
    (∃ n1, 1 ≤ n1 ∧ n1 ≤ 13 ∧ S_n n1 > 200) ∧
    (∃ n2, 14 ≤ n2 ∧ n2 ≤ 31 ∧ a_n n2 < 20) ∧
    days_popular = (22 - 12 + 1) :=
sorry

end most_pieces_day_and_maximum_number_of_popular_days_l161_161806


namespace gcf_75_135_l161_161092

theorem gcf_75_135 : Nat.gcd 75 135 = 15 :=
  by sorry

end gcf_75_135_l161_161092


namespace maximum_rabbits_l161_161878

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l161_161878


namespace cos_sin_value_l161_161488

theorem cos_sin_value (α : ℝ) (h : Real.tan α = Real.sqrt 2) : Real.cos α * Real.sin α = Real.sqrt 2 / 3 :=
sorry

end cos_sin_value_l161_161488


namespace bus_speed_excluding_stoppages_l161_161441

theorem bus_speed_excluding_stoppages 
  (v : ℝ) 
  (speed_incl_stoppages : v * 54 / 60 = 45) : 
  v = 50 := 
  by 
    sorry

end bus_speed_excluding_stoppages_l161_161441


namespace min_large_buses_proof_l161_161184

def large_bus_capacity : ℕ := 45
def small_bus_capacity : ℕ := 30
def total_students : ℕ := 523
def min_small_buses : ℕ := 5

def min_large_buses_required (large_capacity small_capacity total small_buses : ℕ) : ℕ :=
  let remaining_students := total - (small_buses * small_capacity)
  let buses_needed := remaining_students / large_capacity
  if remaining_students % large_capacity = 0 then buses_needed else buses_needed + 1

theorem min_large_buses_proof :
  min_large_buses_required large_bus_capacity small_bus_capacity total_students min_small_buses = 9 :=
by
  sorry

end min_large_buses_proof_l161_161184


namespace find_x_for_equation_l161_161976

theorem find_x_for_equation 
  (x : ℝ)
  (h : (32 : ℝ)^(x-2) / (8 : ℝ)^(x-2) = (512 : ℝ)^(3 * x)) : 
  x = -4/25 :=
by
  sorry

end find_x_for_equation_l161_161976


namespace solve_for_A_l161_161408

def diamond (A B : ℝ) := 4 * A + 3 * B + 7

theorem solve_for_A : diamond A 5 = 71 → A = 12.25 := by
  intro h
  unfold diamond at h
  sorry

end solve_for_A_l161_161408


namespace solve_inequality_l161_161782

theorem solve_inequality (x : ℝ) :
  (x * (x + 2) / (x - 3) < 0) ↔ (x < -2 ∨ (0 < x ∧ x < 3)) :=
sorry

end solve_inequality_l161_161782


namespace Total_points_proof_l161_161846

noncomputable def Samanta_points (Mark_points : ℕ) : ℕ := Mark_points + 8
noncomputable def Mark_points (Eric_points : ℕ) : ℕ := Eric_points + (Eric_points / 2)
def Eric_points : ℕ := 6
noncomputable def Daisy_points (Total_points_Samanta_Mark_Eric : ℕ) : ℕ := Total_points_Samanta_Mark_Eric - (Total_points_Samanta_Mark_Eric / 4)

def Total_points_Samanta_Mark_Eric (Samanta_points Mark_points Eric_points : ℕ) : ℕ := Samanta_points + Mark_points + Eric_points

theorem Total_points_proof :
  let Mk_pts := Mark_points Eric_points
  let Sm_pts := Samanta_points Mk_pts
  let Tot_SME := Total_points_Samanta_Mark_Eric Sm_pts Mk_pts Eric_points
  let D_pts := Daisy_points Tot_SME
  Sm_pts + Mk_pts + Eric_points + D_pts = 56 := by
  sorry

end Total_points_proof_l161_161846


namespace correct_number_of_statements_l161_161431

theorem correct_number_of_statements (a b : ℤ) :
  (¬ (∃ h₁ : Even (a + 5 * b), ¬ Even (a - 7 * b)) ∧
   ∃ h₂ : a + b % 3 = 0, ¬ ((a % 3 = 0) ∧ (b % 3 = 0)) ∧
   ∃ h₃ : Prime (a + b), Prime (a - b)) →
   1 = 1 :=
by
  sorry

end correct_number_of_statements_l161_161431


namespace birch_tree_taller_than_pine_tree_l161_161402

theorem birch_tree_taller_than_pine_tree :
  let pine_tree_height := (49 : ℚ) / 4
  let birch_tree_height := (37 : ℚ) / 2
  birch_tree_height - pine_tree_height = 25 / 4 :=
by
  sorry

end birch_tree_taller_than_pine_tree_l161_161402


namespace externally_tangent_circles_m_l161_161279

def circle1_eqn (x y : ℝ) : Prop := x^2 + y^2 = 4

def circle2_eqn (x y m : ℝ) : Prop := x^2 + y^2 - 2 * m * x + m^2 - 1 = 0

theorem externally_tangent_circles_m (m : ℝ) :
  (∀ x y : ℝ, circle1_eqn x y) →
  (∀ x y : ℝ, circle2_eqn x y m) →
  m = 3 ∨ m = -3 :=
by sorry

end externally_tangent_circles_m_l161_161279


namespace time_to_groom_rottweiler_l161_161301

theorem time_to_groom_rottweiler
  (R : ℕ)  -- Time to groom a rottweiler
  (B : ℕ)  -- Time to groom a border collie
  (C : ℕ)  -- Time to groom a chihuahua
  (total_time_6R_9B_1C : 6 * R + 9 * B + C = 255)  -- Total time for grooming 6 rottweilers, 9 border collies, and 1 chihuahua
  (time_to_groom_border_collie : B = 10)  -- Time to groom a border collie is 10 minutes
  (time_to_groom_chihuahua : C = 45) :  -- Time to groom a chihuahua is 45 minutes
  R = 20 :=  -- Prove that it takes 20 minutes to groom a rottweiler
by
  sorry

end time_to_groom_rottweiler_l161_161301


namespace value_of_expression_l161_161916

theorem value_of_expression (a b : ℤ) (h : a - b = 1) : 3 * a - 3 * b - 4 = -1 :=
by {
  sorry
}

end value_of_expression_l161_161916


namespace find_unknown_rate_l161_161375

def cost_with_discount_and_tax (original_price : ℝ) (count : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let discounted_price := (original_price * count) * (1 - discount)
  discounted_price * (1 + tax)

theorem find_unknown_rate :
  let total_blankets := 10
  let average_price := 160
  let total_cost := total_blankets * average_price
  let cost_100_blankets := cost_with_discount_and_tax 100 3 0.05 0.12
  let cost_150_blankets := cost_with_discount_and_tax 150 5 0.10 0.15
  let cost_unknown_blankets := 2 * x
  total_cost = cost_100_blankets + cost_150_blankets + cost_unknown_blankets →
  x = 252.275 :=
by
  sorry

end find_unknown_rate_l161_161375


namespace number_of_boxes_initially_l161_161090

theorem number_of_boxes_initially (B : ℕ) (h1 : ∃ B, 8 * B - 17 = 15) : B = 4 :=
  by
  sorry

end number_of_boxes_initially_l161_161090


namespace teachers_can_sit_in_middle_l161_161332

-- Definitions for the conditions
def num_students : ℕ := 4
def num_teachers : ℕ := 3
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def permutations (n r : ℕ) : ℕ := factorial n / factorial (n - r)

-- Definition statements
def num_ways_teachers : ℕ := permutations num_teachers num_teachers
def num_ways_students : ℕ := permutations num_students num_students

-- Main theorem statement
theorem teachers_can_sit_in_middle : num_ways_teachers * num_ways_students = 144 := by
  -- Calculation goes here but is omitted for brevity
  sorry

end teachers_can_sit_in_middle_l161_161332


namespace wheat_flour_used_l161_161814

-- Conditions and definitions
def total_flour_used : ℝ := 0.3
def white_flour_used : ℝ := 0.1

-- Statement of the problem
theorem wheat_flour_used : 
  (total_flour_used - white_flour_used) = 0.2 :=
by
  sorry

end wheat_flour_used_l161_161814


namespace sum_of_squares_iff_double_l161_161119

theorem sum_of_squares_iff_double (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end sum_of_squares_iff_double_l161_161119


namespace frac_not_suff_nec_l161_161873

theorem frac_not_suff_nec {a b : ℝ} (hab : a / b > 1) : 
  ¬ ((∀ a b : ℝ, a / b > 1 → a > b) ∧ (∀ a b : ℝ, a > b → a / b > 1)) :=
sorry

end frac_not_suff_nec_l161_161873


namespace inequality_1_minimum_value_l161_161778

-- Definition for part (1)
theorem inequality_1 (a b m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (a^2 / m + b^2 / n) ≥ ((a + b)^2 / (m + n)) :=
sorry

-- Definition for part (2)
theorem minimum_value (x : ℝ) (hx : 0 < x) (hx' : x < 1) : 
  (∃ (y : ℝ), y = (1 / x + 4 / (1 - x)) ∧ y = 9) :=
sorry

end inequality_1_minimum_value_l161_161778


namespace lisa_pizza_l161_161701

theorem lisa_pizza (P H S : ℕ) 
  (h1 : H = 2 * P) 
  (h2 : S = P + 12) 
  (h3 : P + H + S = 132) : 
  P = 30 := 
by
  sorry

end lisa_pizza_l161_161701


namespace standard_deviation_is_one_l161_161258

noncomputable def standard_deviation (μ : ℝ) (σ : ℝ) : Prop :=
  ∀ x : ℝ, (0.68 * μ ≤ x ∧ x ≤ 1.32 * μ) → σ = 1

theorem standard_deviation_is_one (a : ℝ) (σ : ℝ) :
  (0.68 * a ≤ a + σ ∧ a + σ ≤ 1.32 * a) → σ = 1 :=
by
  -- Proof omitted.
  sorry

end standard_deviation_is_one_l161_161258


namespace arcsin_one_eq_pi_div_two_l161_161364

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := 
by
  sorry

end arcsin_one_eq_pi_div_two_l161_161364


namespace opposite_face_of_orange_is_blue_l161_161220

structure CubeOrientation :=
  (top : String)
  (front : String)
  (right : String)

def first_view : CubeOrientation := { top := "B", front := "Y", right := "S" }
def second_view : CubeOrientation := { top := "B", front := "V", right := "S" }
def third_view : CubeOrientation := { top := "B", front := "K", right := "S" }

theorem opposite_face_of_orange_is_blue
  (colors : List String)
  (c1 : CubeOrientation)
  (c2 : CubeOrientation)
  (c3 : CubeOrientation)
  (no_orange_in_views : "O" ∉ colors.erase c1.top ∧ "O" ∉ colors.erase c1.front ∧ "O" ∉ colors.erase c1.right ∧
                         "O" ∉ colors.erase c2.top ∧ "O" ∉ colors.erase c2.front ∧ "O" ∉ colors.erase c2.right ∧
                         "O" ∉ colors.erase c3.top ∧ "O" ∉ colors.erase c3.front ∧ "O" ∉ colors.erase c3.right) :
  (c1.top = "B" → c2.top = "B" → c3.top = "B" → c1.right = "S" → c2.right = "S" → c3.right = "S" → 
  ∃ opposite_color, opposite_color = "B") :=
sorry

end opposite_face_of_orange_is_blue_l161_161220


namespace total_seashells_found_intact_seashells_found_l161_161907

-- Define the constants for seashells found
def tom_seashells : ℕ := 15
def fred_seashells : ℕ := 43

-- Define total_intercept
def total_intercept : ℕ := 29

-- Statement that the total seashells found by Tom and Fred is 58
theorem total_seashells_found : tom_seashells + fred_seashells = 58 := by
  sorry

-- Statement that the intact seashells are obtained by subtracting cracked ones
theorem intact_seashells_found : tom_seashells + fred_seashells - total_intercept = 29 := by
  sorry

end total_seashells_found_intact_seashells_found_l161_161907


namespace bill_tossed_21_objects_l161_161436

-- Definitions based on the conditions from step a)
def ted_sticks := 10
def ted_rocks := 10
def bill_sticks := ted_sticks + 6
def bill_rocks := ted_rocks / 2

-- The condition of total objects tossed by Bill
def bill_total_objects := bill_sticks + bill_rocks

-- The theorem we want to prove
theorem bill_tossed_21_objects :
  bill_total_objects = 21 :=
  by
  sorry

end bill_tossed_21_objects_l161_161436


namespace bill_age_l161_161728

theorem bill_age (C : ℕ) (h1 : ∀ B : ℕ, B = 2 * C - 1) (h2 : C + (2 * C - 1) = 26) : 
  ∃ B : ℕ, B = 17 := 
by
  sorry

end bill_age_l161_161728


namespace smallest_positive_multiple_l161_161972

/-- Prove that the smallest positive multiple of 15 that is 7 more than a multiple of 65 is 255. -/
theorem smallest_positive_multiple : 
  ∃ n : ℕ, n > 0 ∧ n % 15 = 0 ∧ n % 65 = 7 ∧ n = 255 :=
sorry

end smallest_positive_multiple_l161_161972


namespace fraction_addition_l161_161260

theorem fraction_addition :
  (1 / 3 * 2 / 5) + 1 / 4 = 23 / 60 := 
  sorry

end fraction_addition_l161_161260


namespace income_calculation_l161_161312

theorem income_calculation
  (x : ℕ)
  (income : ℕ := 5 * x)
  (expenditure : ℕ := 4 * x)
  (savings : ℕ := income - expenditure)
  (savings_eq : savings = 3000) :
  income = 15000 :=
sorry

end income_calculation_l161_161312


namespace probability_part_not_scrap_l161_161722

noncomputable def probability_not_scrap : Prop :=
  let p_scrap_first := 0.01
  let p_scrap_second := 0.02
  let p_not_scrap_first := 1 - p_scrap_first
  let p_not_scrap_second := 1 - p_scrap_second
  let p_not_scrap := p_not_scrap_first * p_not_scrap_second
  p_not_scrap = 0.9702

theorem probability_part_not_scrap : probability_not_scrap :=
by simp [probability_not_scrap] ; sorry

end probability_part_not_scrap_l161_161722


namespace semicircle_area_l161_161151

theorem semicircle_area (x : ℝ) (y : ℝ) (r : ℝ) (h1 : x = 1) (h2 : y = 3) (h3 : x^2 + y^2 = (2*r)^2) :
  (1/2) * π * r^2 = (13 * π) / 8 :=
by
  sorry

end semicircle_area_l161_161151


namespace sons_ages_l161_161516

theorem sons_ages (x y : ℕ) (h1 : 2 * x = x + y + 18) (h2 : y = (x - y) - 6) : 
  x = 30 ∧ y = 12 := by
  sorry

end sons_ages_l161_161516


namespace trapezoid_area_l161_161017

theorem trapezoid_area (a b d1 d2 : ℝ) (ha : 0 < a) (hb : 0 < b) (hd1 : 0 < d1) (hd2 : 0 < d2)
  (hbase : a = 11) (hbase2 : b = 4) (hdiagonal1 : d1 = 9) (hdiagonal2 : d2 = 12) :
  (∃ area : ℝ, area = 54) :=
by
  sorry

end trapezoid_area_l161_161017


namespace sufficient_but_not_necessary_condition_l161_161602

variable {a : Type} {M : Type} (line : a → Prop) (plane : M → Prop)

-- Assume the definitions of perpendicularity
def perp_to_plane (a : a) (M : M) : Prop := sorry -- define perpendicular to plane
def perp_to_lines_in_plane (a : a) (M : M) : Prop := sorry -- define perpendicular to countless lines

-- Mathematical statement
theorem sufficient_but_not_necessary_condition (a : a) (M : M) :
  (perp_to_plane a M → perp_to_lines_in_plane a M) ∧ ¬(perp_to_lines_in_plane a M → perp_to_plane a M) :=
by
  sorry

end sufficient_but_not_necessary_condition_l161_161602


namespace greater_prime_of_lcm_and_sum_l161_161472

-- Define the problem conditions
def is_prime (n: ℕ) : Prop := Nat.Prime n
def is_lcm (a b l: ℕ) : Prop := Nat.lcm a b = l

-- Statement of the theorem to be proved
theorem greater_prime_of_lcm_and_sum (x y: ℕ) 
  (hx: is_prime x) 
  (hy: is_prime y) 
  (hlcm: is_lcm x y 10) 
  (h_sum: 2 * x + y = 12) : 
  x > y :=
sorry

end greater_prime_of_lcm_and_sum_l161_161472


namespace compute_cd_l161_161517

variable (c d : ℝ)

theorem compute_cd (h1 : c + d = 10) (h2 : c^3 + d^3 = 370) : c * d = 21 := by
  -- Proof would go here
  sorry

end compute_cd_l161_161517


namespace container_capacity_l161_161709

theorem container_capacity
  (C : ℝ)  -- Total capacity of the container in liters
  (h1 : C / 2 + 20 = 3 * C / 4)  -- Condition combining the water added and the fractional capacities
  : C = 80 := 
sorry

end container_capacity_l161_161709


namespace determine_b_l161_161847

variable (a b c : ℝ)

theorem determine_b
  (h1 : -a / 3 = -c)
  (h2 : 1 + a + b + c = -c)
  (h3 : c = 5) :
  b = -26 :=
by
  sorry

end determine_b_l161_161847


namespace units_digit_G_2000_l161_161720

-- Define the sequence G
def G (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 5 ^ (5 ^ n)

-- The main goal is to show that the units digit of G 2000 is 1
theorem units_digit_G_2000 : (G 2000) % 10 = 1 :=
by
  sorry

end units_digit_G_2000_l161_161720


namespace range_of_a_l161_161066

def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2) : a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by
  sorry

end range_of_a_l161_161066


namespace arctan_sum_l161_161392

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end arctan_sum_l161_161392


namespace rectangular_solid_volume_l161_161648

theorem rectangular_solid_volume 
  (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 15)
  (h3 : x * z = 12) :
  x * y * z = 60 :=
by
  sorry

end rectangular_solid_volume_l161_161648


namespace age_of_b_l161_161461

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 27) : b = 10 := by
  sorry

end age_of_b_l161_161461


namespace composite_of_n_gt_one_l161_161964

theorem composite_of_n_gt_one (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^4 + 4 = a * b :=
by
  sorry

end composite_of_n_gt_one_l161_161964


namespace alice_two_turns_probability_l161_161600

def alice_to_alice_first_turn : ℚ := 2 / 3
def alice_to_bob_first_turn : ℚ := 1 / 3
def bob_to_alice_second_turn : ℚ := 1 / 4
def bob_keeps_second_turn : ℚ := 3 / 4
def alice_keeps_second_turn : ℚ := 2 / 3

def probability_alice_keeps_twice : ℚ := alice_to_alice_first_turn * alice_keeps_second_turn
def probability_alice_bob_alice : ℚ := alice_to_bob_first_turn * bob_to_alice_second_turn

theorem alice_two_turns_probability : 
  probability_alice_keeps_twice + probability_alice_bob_alice = 37 / 108 := 
by
  sorry

end alice_two_turns_probability_l161_161600


namespace laundry_per_hour_l161_161710

-- Definitions based on the conditions
def total_laundry : ℕ := 80
def total_hours : ℕ := 4

-- Theorems to prove the number of pieces per hour
theorem laundry_per_hour : total_laundry / total_hours = 20 :=
by
  -- Placeholder for the proof
  sorry

end laundry_per_hour_l161_161710


namespace sequence_diff_n_l161_161395

theorem sequence_diff_n {a : ℕ → ℕ} (h1 : a 1 = 1) 
(h2 : ∀ n : ℕ, a (n + 1) ≤ 2 * n) (n : ℕ) :
  ∃ p q : ℕ, a p - a q = n :=
sorry

end sequence_diff_n_l161_161395


namespace benzene_carbon_mass_percentage_l161_161959

noncomputable def carbon_mass_percentage_in_benzene 
  (carbon_atomic_mass : ℝ) (hydrogen_atomic_mass : ℝ) 
  (benzene_formula_ratio : (ℕ × ℕ)) : ℝ := 
    let (num_carbon_atoms, num_hydrogen_atoms) := benzene_formula_ratio
    let total_carbon_mass := num_carbon_atoms * carbon_atomic_mass
    let total_hydrogen_mass := num_hydrogen_atoms * hydrogen_atomic_mass
    let total_mass := total_carbon_mass + total_hydrogen_mass
    100 * (total_carbon_mass / total_mass)

theorem benzene_carbon_mass_percentage 
  (carbon_atomic_mass : ℝ := 12.01) 
  (hydrogen_atomic_mass : ℝ := 1.008) 
  (benzene_formula_ratio : (ℕ × ℕ) := (6, 6)) : 
    carbon_mass_percentage_in_benzene carbon_atomic_mass hydrogen_atomic_mass benzene_formula_ratio = 92.23 :=
by 
  unfold carbon_mass_percentage_in_benzene
  sorry

end benzene_carbon_mass_percentage_l161_161959


namespace function_solution_l161_161921

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
if x = 0 then c else if x = 1 then 3 - 2 * c else (-x^3 + 3 * x^2 + 2) / (3 * x * (1 - x))

theorem function_solution (f : ℝ → ℝ) :
  (∀ x ≠ 0, f x + 2 * f ((x - 1) / x) = 3 * x) →
  (∃ c : ℝ, ∀ x : ℝ, f x = if x = 0 then c else if x = 1 then 3 - 2 * c else (-x^3 + 3 * x^2 + 2) / (3 * x * (1 - x))) :=
by
  intro h
  use (f 0)
  intro x
  split_ifs with h0 h1
  rotate_left -- to handle the cases x ≠ 0, 1 at first.
  sorry -- Additional proof steps required here.
  sorry -- Use the given conditions and functional equation to conclude f(0) = c.
  sorry -- Use the given conditions and functional equation to conclude f(1) = 3 - 2c.

end function_solution_l161_161921


namespace find_x_solutions_l161_161938

theorem find_x_solutions (x : ℝ) :
  let f (x : ℝ) := x^2 - 4*x + 1
  let f2 (x : ℝ) := (f x)^2
  f (f x) = f2 x ↔ x = 2 + (Real.sqrt 13) / 2 ∨ x = 2 - (Real.sqrt 13) / 2 := by
  sorry

end find_x_solutions_l161_161938


namespace x_intercept_is_correct_l161_161339

-- Define the original line equation
def original_line (x y : ℝ) : Prop := 4 * x + 5 * y = 10

-- Define the perpendicular line's y-intercept
def y_intercept (y : ℝ) : Prop := y = -3

-- Define the equation of the perpendicular line in slope-intercept form
def perpendicular_line (x y : ℝ) : Prop := y = (5 / 4) * x + -3

-- Prove that the x-intercept of the perpendicular line is 12/5
theorem x_intercept_is_correct : ∃ x : ℝ, x ≠ 0 ∧ (∃ y : ℝ, y = 0) ∧ (perpendicular_line x y) :=
sorry

end x_intercept_is_correct_l161_161339


namespace greatest_possible_value_of_a_l161_161479

theorem greatest_possible_value_of_a (a : ℤ) (h1 : ∃ x : ℤ, x^2 + a*x = -30) (h2 : 0 < a) :
  a ≤ 31 :=
sorry

end greatest_possible_value_of_a_l161_161479


namespace find_real_roots_of_PQ_l161_161164

noncomputable def P (x b : ℝ) : ℝ := x^2 + x / 2 + b
noncomputable def Q (x c d : ℝ) : ℝ := x^2 + c * x + d

theorem find_real_roots_of_PQ (b c d : ℝ)
  (h: ∀ x : ℝ, P x b * Q x c d = Q (P x b) c d)
  (h_d_zero: d = 0) :
  ∃ x : ℝ, P (Q x c d) b = 0 → x = (-c + Real.sqrt (c^2 + 2)) / 2 ∨ x = (-c - Real.sqrt (c^2 + 2)) / 2 :=
by
  sorry

end find_real_roots_of_PQ_l161_161164


namespace percentage_rent_this_year_l161_161506

variables (E : ℝ)

-- Define the conditions from the problem
def rent_last_year (E : ℝ) : ℝ := 0.20 * E
def earnings_this_year (E : ℝ) : ℝ := 1.15 * E
def rent_this_year (E : ℝ) : ℝ := 1.4375 * rent_last_year E

-- The main statement to prove
theorem percentage_rent_this_year : 
  0.2875 * E = (25 / 100) * (earnings_this_year E) :=
by sorry

end percentage_rent_this_year_l161_161506


namespace least_integer_value_satisfying_inequality_l161_161901

theorem least_integer_value_satisfying_inequality : ∃ x : ℤ, 3 * |x| + 6 < 24 ∧ (∀ y : ℤ, 3 * |y| + 6 < 24 → x ≤ y) :=
  sorry

end least_integer_value_satisfying_inequality_l161_161901


namespace pictures_left_l161_161988

def zoo_pics : ℕ := 802
def museum_pics : ℕ := 526
def beach_pics : ℕ := 391
def amusement_park_pics : ℕ := 868
def duplicates_deleted : ℕ := 1395

theorem pictures_left : 
  (zoo_pics + museum_pics + beach_pics + amusement_park_pics - duplicates_deleted) = 1192 := 
by
  sorry

end pictures_left_l161_161988


namespace find_a8_l161_161540

variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}

def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a_n n = a_n 1 + (n - 1) * (a_n 2 - a_n 1)

def sum_of_terms (a_n : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = n * (a_n 1 + a_n n) / 2

theorem find_a8
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_of_terms a_n S)
  (h_S15 : S 15 = 45) :
  a_n 8 = 3 :=
sorry

end find_a8_l161_161540


namespace plot_length_l161_161148

theorem plot_length (b : ℝ) (cost_per_meter cost_total : ℝ)
  (h1 : cost_per_meter = 26.5) 
  (h2 : cost_total = 5300) 
  (h3 : (2 * (b + (b + 20)) * cost_per_meter) = cost_total) : 
  b + 20 = 60 := 
by 
  -- Proof here
  sorry

end plot_length_l161_161148


namespace possible_number_of_students_l161_161328

theorem possible_number_of_students (n : ℕ) 
  (h1 : n ≥ 1) 
  (h2 : ∃ k : ℕ, 120 = 2 * n + 2 * k) :
  n = 58 ∨ n = 60 :=
sorry

end possible_number_of_students_l161_161328


namespace sticks_per_stool_is_two_l161_161679

-- Conditions
def sticks_from_chair := 6
def sticks_from_table := 9
def sticks_needed_per_hour := 5
def num_chairs := 18
def num_tables := 6
def num_stools := 4
def hours_to_keep_warm := 34

-- Question and Answer in Lean 4 statement
theorem sticks_per_stool_is_two : 
  (hours_to_keep_warm * sticks_needed_per_hour) - (num_chairs * sticks_from_chair + num_tables * sticks_from_table) = 2 * num_stools := 
  by
    sorry

end sticks_per_stool_is_two_l161_161679


namespace adjacent_angles_l161_161400

theorem adjacent_angles (α β : ℝ) (h1 : α = β + 30) (h2 : α + β = 180) : α = 105 ∧ β = 75 := by
  sorry

end adjacent_angles_l161_161400


namespace pencil_probability_l161_161872

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem pencil_probability : 
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := 6
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination non_defective_pencils selected_pencils
  let probability := non_defective_ways / total_ways
  probability = 5 / 14 :=
by
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := 6
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination non_defective_pencils selected_pencils
  let probability := non_defective_ways / total_ways
  have h : probability = 5 / 14 := sorry
  exact h

end pencil_probability_l161_161872


namespace sam_seashells_l161_161951

def seashells_problem := 
  let mary_seashells := 47
  let total_seashells := 65
  (total_seashells - mary_seashells) = 18

theorem sam_seashells :
  seashells_problem :=
by
  sorry

end sam_seashells_l161_161951


namespace box_volume_in_cubic_yards_l161_161626

theorem box_volume_in_cubic_yards (v_feet : ℕ) (conv_factor : ℕ) (v_yards : ℕ)
  (h1 : v_feet = 216) (h2 : conv_factor = 3) (h3 : 27 = conv_factor ^ 3) : 
  v_yards = 8 :=
by
  sorry

end box_volume_in_cubic_yards_l161_161626


namespace half_of_animals_get_sick_l161_161577

theorem half_of_animals_get_sick : 
  let chickens := 26
  let piglets := 40
  let goats := 34
  let total_animals := chickens + piglets + goats
  let sick_animals := total_animals / 2
  sick_animals = 50 :=
by
  sorry

end half_of_animals_get_sick_l161_161577


namespace find_x_l161_161411

theorem find_x (x : ℝ) (h : 0.75 * x = (1 / 3) * x + 110) : x = 264 :=
sorry

end find_x_l161_161411


namespace range_of_a_l161_161300

open Real

noncomputable def f (x a : ℝ) : ℝ := (exp x / 2) - (a / exp x)

def condition (x₁ x₂ a : ℝ) : Prop :=
  x₁ ≠ x₂ ∧ 1 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1 ≤ x₂ ∧ x₂ ≤ 2 ∧ ((abs (f x₁ a) - abs (f x₂ a)) * (x₁ - x₂) > 0)

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), condition x₁ x₂ a) ↔ (- (exp 2) / 2 ≤ a ∧ a ≤ (exp 2) / 2) :=
by
  sorry

end range_of_a_l161_161300


namespace ratio_of_sum_l161_161120

theorem ratio_of_sum (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : 
  (x + y) / 3 = 11 / 9 := 
by 
  sorry

end ratio_of_sum_l161_161120


namespace count_valid_N_l161_161579

theorem count_valid_N : ∃ (count : ℕ), count = 10 ∧ 
    (∀ N : ℕ, (10 ≤ N ∧ N < 100) → 
        (∃ a b c d : ℕ, 
            a < 3 ∧ b < 3 ∧ c < 3 ∧ d < 4 ∧
            N = 3 * a + b ∧ N = 4 * c + d ∧
            2 * N % 50 = ((9 * a + b) + (8 * c + d)) % 50)) :=
sorry

end count_valid_N_l161_161579


namespace trig_identity_example_l161_161518

open Real -- Using the Real namespace for trigonometric functions

theorem trig_identity_example :
  sin (135 * π / 180) * cos (-15 * π / 180) + cos (225 * π / 180) * sin (15 * π / 180) = 1 / 2 :=
by 
  -- sorry to skip the proof steps
  sorry

end trig_identity_example_l161_161518


namespace find_floor_abs_S_l161_161693

-- Conditions
-- For integers from 1 to 1500, x_1 + 2 = x_2 + 4 = x_3 + 6 = ... = x_1500 + 3000 = ∑(n=1 to 1500) x_n + 3001
def condition (x : ℕ → ℤ) (S : ℤ) : Prop :=
  ∀ a : ℕ, 1 ≤ a ∧ a ≤ 1500 →
    x a + 2 * a = S + 3001

-- Problem statement
theorem find_floor_abs_S (x : ℕ → ℤ) (S : ℤ)
  (h : condition x S) :
  (⌊|S|⌋ : ℤ) = 1500 :=
sorry

end find_floor_abs_S_l161_161693


namespace positive_value_of_A_l161_161749

theorem positive_value_of_A (A : ℝ) (h : A^2 + 3^2 = 130) : A = 11 :=
sorry

end positive_value_of_A_l161_161749


namespace arithmetic_sequence_sum_thirty_l161_161073

-- Definitions according to the conditions
def arithmetic_seq_sums (S : ℕ → ℤ) : Prop :=
  ∃ a d : ℤ, ∀ n : ℕ, S n = a + n * d

-- Main statement we need to prove
theorem arithmetic_sequence_sum_thirty (S : ℕ → ℤ)
  (h1 : S 10 = 10)
  (h2 : S 20 = 30)
  (h3 : arithmetic_seq_sums S) : 
  S 30 = 50 := 
sorry

end arithmetic_sequence_sum_thirty_l161_161073


namespace find_a_plus_b_l161_161568

theorem find_a_plus_b (a b : ℕ) (positive_a : 0 < a) (positive_b : 0 < b)
  (condition : ∀ (n : ℕ), (n > 0) → (∃ m n : ℕ, n = m * a + n * b) ∨ (∃ k l : ℕ, n = 2009 + k * a + l * b))
  (not_expressible : ∃ m n : ℕ, 1776 = m * a + n * b): a + b = 133 :=
sorry

end find_a_plus_b_l161_161568


namespace prism_surface_area_l161_161949

theorem prism_surface_area (a : ℝ) : 
  let surface_area_cubes := 6 * a^2 * 3
  let surface_area_shared_faces := 4 * a^2
  surface_area_cubes - surface_area_shared_faces = 14 * a^2 := 
by 
  let surface_area_cubes := 6 * a^2 * 3
  let surface_area_shared_faces := 4 * a^2
  have : surface_area_cubes - surface_area_shared_faces = 14 * a^2 := sorry
  exact this

end prism_surface_area_l161_161949


namespace remainder_of_power_mod_l161_161729

theorem remainder_of_power_mod :
  ∀ (x n m : ℕ), 
  x = 5 → n = 2021 → m = 17 →
  x^n % m = 11 := by
sorry

end remainder_of_power_mod_l161_161729


namespace area_of_polygon_ABHFGD_l161_161745

noncomputable def total_area_ABHFGD : ℝ :=
  let side_ABCD := 3
  let side_EFGD := 5
  let area_ABCD := side_ABCD * side_ABCD
  let area_EFGD := side_EFGD * side_EFGD
  let area_DBH := 0.5 * 3 * (3 / 2 : ℝ) -- Area of triangle DBH
  let area_DFH := 0.5 * 5 * (5 / 2 : ℝ) -- Area of triangle DFH
  area_ABCD + area_EFGD - (area_DBH + area_DFH)

theorem area_of_polygon_ABHFGD : total_area_ABHFGD = 25.5 := by
  sorry

end area_of_polygon_ABHFGD_l161_161745


namespace liars_are_C_and_D_l161_161512
open Classical 

-- We define inhabitants and their statements
inductive Inhabitant
| A | B | C | D

open Inhabitant

axiom is_liar : Inhabitant → Prop

-- Statements by the inhabitants:
-- A: "At least one of us is a liar."
-- B: "At least two of us are liars."
-- C: "At least three of us are liars."
-- D: "None of us are liars."

def statement_A : Prop := is_liar A ∨ is_liar B ∨ is_liar C ∨ is_liar D
def statement_B : Prop := (is_liar A ∧ is_liar B) ∨ (is_liar A ∧ is_liar C) ∨ (is_liar A ∧ is_liar D) ∨
                          (is_liar B ∧ is_liar C) ∨ (is_liar B ∧ is_liar D) ∨ (is_liar C ∧ is_liar D)
def statement_C : Prop := (is_liar A ∧ is_liar B ∧ is_liar C) ∨ (is_liar A ∧ is_liar B ∧ is_liar D) ∨
                          (is_liar A ∧ is_liar C ∧ is_liar D) ∨ (is_liar B ∧ is_liar C ∧ is_liar D)
def statement_D : Prop := ¬(is_liar A ∨ is_liar B ∨ is_liar C ∨ is_liar D)

-- Given that there are some liars
axiom some_liars_exist : ∃ x, is_liar x

-- Lean proof statement
theorem liars_are_C_and_D : is_liar C ∧ is_liar D ∧ ¬(is_liar A) ∧ ¬(is_liar B) :=
by
  sorry

end liars_are_C_and_D_l161_161512


namespace quadratic_value_l161_161013

theorem quadratic_value (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : 4 * a + 2 * b + c = 3) :
  a + 2 * b + 3 * c = 7 :=
by
  sorry

end quadratic_value_l161_161013


namespace lewis_found_20_items_l161_161250

noncomputable def tanya_items : ℕ := 4

noncomputable def samantha_items : ℕ := 4 * tanya_items

noncomputable def lewis_items : ℕ := samantha_items + 4

theorem lewis_found_20_items : lewis_items = 20 := by
  sorry

end lewis_found_20_items_l161_161250


namespace problem1_solution_l161_161655

theorem problem1_solution (x y : ℚ) (h1 : 3 * x + 2 * y = 10) (h2 : x / 2 - (y + 1) / 3 = 1) : 
  x = 3 ∧ y = 1 / 2 :=
by
  sorry

end problem1_solution_l161_161655


namespace find_constants_l161_161146

theorem find_constants (a b : ℚ) (h1 : 3 * a + b = 7) (h2 : a + 4 * b = 5) :
  a = 61 / 33 ∧ b = 8 / 11 :=
by
  sorry

end find_constants_l161_161146


namespace admission_fee_for_children_l161_161856

theorem admission_fee_for_children (x : ℝ) :
  (∀ (admission_fee_adult : ℝ) (total_people : ℝ) (total_fees_collected : ℝ) (children_admitted : ℝ) (adults_admitted : ℝ),
    admission_fee_adult = 4 ∧
    total_people = 315 ∧
    total_fees_collected = 810 ∧
    children_admitted = 180 ∧
    adults_admitted = total_people - children_admitted ∧
    total_fees_collected = children_admitted * x + adults_admitted * admission_fee_adult
  ) → x = 1.5 := sorry

end admission_fee_for_children_l161_161856


namespace coloring_possible_l161_161644

-- Define what it means for a graph to be planar and bipartite
def planar_graph (G : Type) : Prop := sorry
def bipartite_graph (G : Type) : Prop := sorry

-- The planar graph G results after subdivision without introducing new intersections
def subdivided_graph (G : Type) : Type := sorry

-- Main theorem to prove
theorem coloring_possible (G : Type) (h1 : planar_graph G) : 
  bipartite_graph (subdivided_graph G) :=
sorry

end coloring_possible_l161_161644


namespace six_people_paint_time_l161_161450

noncomputable def time_to_paint_house_with_six_people 
    (initial_people : ℕ) (initial_time : ℝ) (less_efficient_worker_factor : ℝ) 
    (new_people : ℕ) : ℝ :=
  let initial_total_efficiency := initial_people - 1 + less_efficient_worker_factor
  let total_work := initial_total_efficiency * initial_time
  let new_total_efficiency := (new_people - 1) + less_efficient_worker_factor
  total_work / new_total_efficiency

theorem six_people_paint_time (initial_people : ℕ) (initial_time : ℝ) 
    (less_efficient_worker_factor : ℝ) (new_people : ℕ) :
    initial_people = 5 → initial_time = 10 → less_efficient_worker_factor = 0.5 → new_people = 6 →
    time_to_paint_house_with_six_people initial_people initial_time less_efficient_worker_factor new_people = 8.18 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end six_people_paint_time_l161_161450


namespace complex_inverse_l161_161160

noncomputable def complex_expression (i : ℂ) (h_i : i ^ 2 = -1) : ℂ :=
  (3 * i - 3 * (1 / i))⁻¹

theorem complex_inverse (i : ℂ) (h_i : i^2 = -1) :
  complex_expression i h_i = -i / 6 :=
by
  -- the proof part is omitted
  sorry

end complex_inverse_l161_161160


namespace value_of_expression_l161_161935

theorem value_of_expression (x y : ℤ) (hx : x = 2) (hy : y = 1) : 2 * x - 3 * y = 1 :=
by
  -- Substitute the given values into the expression and calculate
  sorry

end value_of_expression_l161_161935


namespace smallest_n_divisible_by_125000_l161_161177

noncomputable def geometric_term_at (a r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

noncomputable def first_term : ℚ := 5 / 8
noncomputable def second_term : ℚ := 25
noncomputable def common_ratio : ℚ := second_term / first_term

theorem smallest_n_divisible_by_125000 :
  ∃ n : ℕ, n ≥ 7 ∧ geometric_term_at first_term common_ratio n % 125000 = 0 :=
by
  sorry

end smallest_n_divisible_by_125000_l161_161177


namespace value_of_q_when_p_is_smallest_l161_161420

-- Definitions of primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m > 1, m < n → ¬ (n % m = 0)

-- smallest prime number
def smallest_prime : ℕ := 2

-- Given conditions
def p : ℕ := 3
def q : ℕ := 2 + 13 * p

-- The theorem to prove
theorem value_of_q_when_p_is_smallest :
  is_prime smallest_prime →
  is_prime q →
  smallest_prime = 2 →
  p = 3 →
  q = 41 :=
by sorry

end value_of_q_when_p_is_smallest_l161_161420


namespace sufficient_not_necessary_condition_abs_eq_one_l161_161496

theorem sufficient_not_necessary_condition_abs_eq_one (a : ℝ) :
  (a = 1 → |a| = 1) ∧ (|a| = 1 → a = 1 ∨ a = -1) :=
by
  sorry

end sufficient_not_necessary_condition_abs_eq_one_l161_161496


namespace zoo_problem_l161_161223

theorem zoo_problem
  (num_zebras : ℕ)
  (num_camels : ℕ)
  (num_monkeys : ℕ)
  (num_giraffes : ℕ)
  (hz : num_zebras = 12)
  (hc : num_camels = num_zebras / 2)
  (hm : num_monkeys = 4 * num_camels)
  (hg : num_giraffes = 2) :
  num_monkeys - num_giraffes = 22 := by
  sorry

end zoo_problem_l161_161223


namespace intercepts_equal_l161_161368

theorem intercepts_equal (m : ℝ) :
  (∃ x y: ℝ, mx - y - 3 - m = 0 ∧ y ≠ 0 ∧ (x = 3 + m ∧ y = -(3 + m))) ↔ (m = -3 ∨ m = -1) :=
by 
  sorry

end intercepts_equal_l161_161368


namespace possible_to_position_guards_l161_161955

-- Define the conditions
def guard_sees (d : ℝ) : Prop := d = 100

-- Prove that it is possible to arrange guards around a point object so that neither the object nor the guards can be approached unnoticed
theorem possible_to_position_guards (num_guards : ℕ) (d : ℝ) (h : guard_sees d) : 
  (0 < num_guards) → 
  (∀ θ : ℕ, θ < num_guards → (θ * (360 / num_guards)) < 360) → 
  True :=
by 
  -- Details of the proof would go here
  sorry

end possible_to_position_guards_l161_161955


namespace determine_a_square_binomial_l161_161775

theorem determine_a_square_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, (4 * x^2 + 16 * x + a) = (2 * x + b)^2) → a = 16 := 
by
  sorry

end determine_a_square_binomial_l161_161775


namespace max_value_of_x_plus_y_l161_161835

variable (x y : ℝ)

-- Conditions
def conditions (x y : ℝ) : Prop := 
  x > 0 ∧ y > 0 ∧ x + y + (1/x) + (1/y) = 5

-- Theorem statement
theorem max_value_of_x_plus_y (x y : ℝ) (h : conditions x y) : x + y ≤ 4 := 
sorry

end max_value_of_x_plus_y_l161_161835


namespace intersection_eq_l161_161144

noncomputable def U : Set ℝ := Set.univ
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 3}

-- Complement of B in U
def complement_B : Set ℝ := {x | x < 2 ∨ x ≥ 3}

-- Intersection of A and complement of B
def intersection : Set ℕ := {x ∈ A | ↑x < 2 ∨ ↑x ≥ 3}

theorem intersection_eq : intersection = {1, 3, 4} :=
by
  sorry

end intersection_eq_l161_161144


namespace maria_needs_nuts_l161_161204

theorem maria_needs_nuts (total_cookies nuts_per_cookie : ℕ) 
  (nuts_fraction : ℚ) (chocolate_fraction : ℚ) 
  (H1 : nuts_fraction = 1 / 4) 
  (H2 : chocolate_fraction = 0.4) 
  (H3 : total_cookies = 60) 
  (H4 : nuts_per_cookie = 2) :
  (total_cookies * nuts_fraction + (total_cookies - total_cookies * nuts_fraction - total_cookies * chocolate_fraction) * nuts_per_cookie) = 72 := 
by
  sorry

end maria_needs_nuts_l161_161204


namespace max_apartment_size_l161_161929

theorem max_apartment_size (rental_price_per_sqft : ℝ) (budget : ℝ) (h1 : rental_price_per_sqft = 1.20) (h2 : budget = 720) : 
  budget / rental_price_per_sqft = 600 :=
by 
  sorry

end max_apartment_size_l161_161929


namespace find_n_l161_161094

theorem find_n 
  (molecular_weight : ℕ)
  (atomic_weight_Al : ℕ)
  (weight_OH : ℕ)
  (n : ℕ) 
  (h₀ : molecular_weight = 78)
  (h₁ : atomic_weight_Al = 27) 
  (h₂ : weight_OH = 17)
  (h₃ : molecular_weight = atomic_weight_Al + n * weight_OH) : 
  n = 3 := 
by 
  -- the proof is omitted
  sorry

end find_n_l161_161094


namespace race_victory_l161_161240

variable (distance : ℕ := 200)
variable (timeA : ℕ := 18)
variable (timeA_beats_B_by : ℕ := 7)

theorem race_victory : ∃ meters_beats_B : ℕ, meters_beats_B = 56 :=
by
  let speedA := distance / timeA
  let timeB := timeA + timeA_beats_B_by
  let speedB := distance / timeB
  let distanceB := speedB * timeA
  let meters_beats_B := distance - distanceB
  use meters_beats_B
  sorry

end race_victory_l161_161240


namespace distinct_roots_iff_l161_161961

def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 + |x1| = 2 * Real.sqrt (3 + 2*a*x1 - 4*a)) ∧
                       (x2 + |x2| = 2 * Real.sqrt (3 + 2*a*x2 - 4*a))

theorem distinct_roots_iff (a : ℝ) :
  has_two_distinct_roots a ↔ (a ∈ Set.Ioo 0 (3 / 4 : ℝ) ∨ 3 < a) :=
sorry

end distinct_roots_iff_l161_161961


namespace cos_A_eq_find_a_l161_161281

variable {A B C a b c : ℝ}

-- Proposition 1: If in triangle ABC, b^2 + c^2 - (sqrt 6) / 2 * b * c = a^2, then cos A = sqrt 6 / 4
theorem cos_A_eq (h : b ^ 2 + c ^ 2 - (Real.sqrt 6) / 2 * b * c = a ^ 2) : Real.cos A = Real.sqrt 6 / 4 :=
sorry

-- Proposition 2: Given b = sqrt 6, B = 2 * A, and b^2 + c^2 - (sqrt 6) / 2 * b * c = a^2, then a = 2
theorem find_a (h1 : b ^ 2 + c ^ 2 - (Real.sqrt 6) / 2 * b * c = a ^ 2) (h2 : B = 2 * A) (h3 : b = Real.sqrt 6) : a = 2 :=
sorry

end cos_A_eq_find_a_l161_161281


namespace parabola_transformation_l161_161542

theorem parabola_transformation :
  (∀ x : ℝ, y = 2 * x^2 → y = 2 * (x-3)^2 - 1) := by
  sorry

end parabola_transformation_l161_161542


namespace sum_digits_of_consecutive_numbers_l161_161894

-- Define the sum of digits function
def sum_digits (n : ℕ) : ℕ := sorry -- Placeholder, define the sum of digits function

-- Given conditions
variables (N : ℕ)
axiom h1 : sum_digits N + sum_digits (N + 1) = 200
axiom h2 : sum_digits (N + 2) + sum_digits (N + 3) = 105

-- Theorem statement to be proved
theorem sum_digits_of_consecutive_numbers : 
  sum_digits (N + 1) + sum_digits (N + 2) = 103 := 
sorry  -- Proof to be provided

end sum_digits_of_consecutive_numbers_l161_161894


namespace find_x_l161_161299

theorem find_x (x : ℤ) (h : (2008 + x)^2 = x^2) : x = -1004 :=
sorry

end find_x_l161_161299


namespace find_integer_solutions_l161_161352

theorem find_integer_solutions :
  ∀ (x y : ℕ), 0 < x → 0 < y → (2 * x^2 + 5 * x * y + 2 * y^2 = 2006 ↔ (x = 28 ∧ y = 3) ∨ (x = 3 ∧ y = 28)) :=
by
  sorry

end find_integer_solutions_l161_161352


namespace inequality_solution_l161_161062

theorem inequality_solution (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (x ∈ Set.Ioo (-2 : ℝ) (-1) ∨ x ∈ Set.Ioi 2) ↔ 
  (∃ x : ℝ, (x^2 + x - 2) / (x + 2) ≥ (3 / (x - 2)) + (3 / 2)) := by
  sorry

end inequality_solution_l161_161062


namespace quadratic_equation_solution_unique_l161_161257

theorem quadratic_equation_solution_unique (b : ℝ) (hb : b ≠ 0) (h1_sol : ∀ x1 x2 : ℝ, 2*b*x1^2 + 16*x1 + 5 = 0 → 2*b*x2^2 + 16*x2 + 5 = 0 → x1 = x2) :
  ∃ x : ℝ, x = -5/8 ∧ 2*b*x^2 + 16*x + 5 = 0 :=
by
  sorry

end quadratic_equation_solution_unique_l161_161257


namespace polynomial_multiplication_l161_161346

noncomputable def multiply_polynomials (a b : ℤ) :=
  let p1 := 3 * a ^ 4 - 7 * b ^ 3
  let p2 := 9 * a ^ 8 + 21 * a ^ 4 * b ^ 3 + 49 * b ^ 6 + 6 * a ^ 2 * b ^ 2
  let result := 27 * a ^ 12 + 18 * a ^ 6 * b ^ 2 - 42 * a ^ 2 * b ^ 5 - 343 * b ^ 9
  p1 * p2 = result

-- The main statement to prove
theorem polynomial_multiplication (a b : ℤ) : multiply_polynomials a b :=
by
  sorry

end polynomial_multiplication_l161_161346


namespace carson_clawed_39_times_l161_161725

def wombats_count := 9
def wombat_claws_per := 4
def rheas_count := 3
def rhea_claws_per := 1

def wombat_total_claws := wombats_count * wombat_claws_per
def rhea_total_claws := rheas_count * rhea_claws_per
def total_claws := wombat_total_claws + rhea_total_claws

theorem carson_clawed_39_times : total_claws = 39 :=
  by sorry

end carson_clawed_39_times_l161_161725


namespace cost_per_bag_l161_161323

-- Definitions and variables based on the conditions
def sandbox_length : ℝ := 3  -- Sandbox length in feet
def sandbox_width : ℝ := 3   -- Sandbox width in feet
def bag_area : ℝ := 3        -- Area of one bag of sand in square feet
def total_cost : ℝ := 12     -- Total cost to fill up the sandbox in dollars

-- Statement to prove
theorem cost_per_bag : (total_cost / (sandbox_length * sandbox_width / bag_area)) = 4 :=
by
  sorry

end cost_per_bag_l161_161323


namespace count_divisors_of_54_greater_than_7_l161_161318

theorem count_divisors_of_54_greater_than_7 : ∃ (S : Finset ℕ), S.card = 4 ∧ ∀ n ∈ S, n ∣ 54 ∧ n > 7 :=
by
  -- proof goes here
  sorry

end count_divisors_of_54_greater_than_7_l161_161318


namespace mary_cards_left_l161_161061

noncomputable def mary_initial_cards : ℝ := 18.0
noncomputable def cards_to_fred : ℝ := 26.0
noncomputable def cards_bought : ℝ := 40.0
noncomputable def mary_final_cards : ℝ := 32.0

theorem mary_cards_left :
  (mary_initial_cards + cards_bought) - cards_to_fred = mary_final_cards := 
by 
  sorry

end mary_cards_left_l161_161061


namespace grocery_store_distance_l161_161361

theorem grocery_store_distance 
    (park_house : ℕ) (park_store : ℕ) (total_distance : ℕ) (grocery_store_house: ℕ) :
    park_house = 5 ∧ park_store = 3 ∧ total_distance = 16 → grocery_store_house = 8 :=
by 
    sorry

end grocery_store_distance_l161_161361


namespace min_sum_of_factors_l161_161003

theorem min_sum_of_factors 
  (a b c: ℕ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0)
  (h4: a * b * c = 2450) :
  a + b + c ≥ 76 :=
sorry

end min_sum_of_factors_l161_161003


namespace smallest_positive_value_is_A_l161_161249

noncomputable def expr_A : ℝ := 12 - 4 * Real.sqrt 8
noncomputable def expr_B : ℝ := 4 * Real.sqrt 8 - 12
noncomputable def expr_C : ℝ := 20 - 6 * Real.sqrt 10
noncomputable def expr_D : ℝ := 60 - 15 * Real.sqrt 16
noncomputable def expr_E : ℝ := 15 * Real.sqrt 16 - 60

theorem smallest_positive_value_is_A :
  expr_A = 12 - 4 * Real.sqrt 8 ∧ 
  expr_B = 4 * Real.sqrt 8 - 12 ∧ 
  expr_C = 20 - 6 * Real.sqrt 10 ∧ 
  expr_D = 60 - 15 * Real.sqrt 16 ∧ 
  expr_E = 15 * Real.sqrt 16 - 60 ∧ 
  expr_A > 0 ∧ 
  expr_A < expr_C := 
sorry

end smallest_positive_value_is_A_l161_161249


namespace monotonic_increasing_interval_l161_161743

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 0 < x → (1 / 2 < x → (f (x + 0.1) > f x)) :=
by
  intro x hx h
  sorry

end monotonic_increasing_interval_l161_161743


namespace desired_depth_is_50_l161_161195

noncomputable def desired_depth_dig (d days : ℝ) : ℝ :=
  let initial_man_hours := 45 * 8 * d
  let additional_man_hours := 100 * 6 * d
  (initial_man_hours / additional_man_hours) * 30

theorem desired_depth_is_50 (d : ℝ) : desired_depth_dig d = 50 :=
  sorry

end desired_depth_is_50_l161_161195


namespace sum_of_consecutive_integers_largest_set_of_consecutive_positive_integers_l161_161107

theorem sum_of_consecutive_integers (n : ℕ) (a : ℕ) (h : n ≥ 1) (h_sum : n * (2 * a + n - 1) = 56) : n ≤ 7 := 
by
  sorry

theorem largest_set_of_consecutive_positive_integers : ∃ n a, n ≥ 1 ∧ n * (2 * a + n - 1) = 56 ∧ n = 7 := 
by
  use 7, 1
  repeat {split}
  sorry

end sum_of_consecutive_integers_largest_set_of_consecutive_positive_integers_l161_161107


namespace find_C_l161_161147

theorem find_C (A B C : ℕ) (h1 : A + B + C = 900) (h2 : A + C = 400) (h3 : B + C = 750) : C = 250 :=
by
  sorry

end find_C_l161_161147


namespace sum_of_even_factors_420_l161_161637

def sum_even_factors (n : ℕ) : ℕ :=
  if n ≠ 420 then 0
  else 
    let even_factors_sum :=
      (2 + 4) * (1 + 3) * (1 + 5) * (1 + 7)
    even_factors_sum

theorem sum_of_even_factors_420 : sum_even_factors 420 = 1152 :=
by {
  -- Proof skipped
  sorry
}

end sum_of_even_factors_420_l161_161637


namespace percent_decrease_area_square_l161_161754

/-- 
In a configuration, two figures, an equilateral triangle and a square, are initially given. 
The equilateral triangle has an area of 27√3 square inches, and the square has an area of 27 square inches.
If the side length of the square is decreased by 10%, prove that the percent decrease in the area of the square is 19%.
-/
theorem percent_decrease_area_square 
  (triangle_area : ℝ := 27 * Real.sqrt 3)
  (square_area : ℝ := 27)
  (percentage_decrease : ℝ := 0.10) : 
  let new_square_side := Real.sqrt square_area * (1 - percentage_decrease)
  let new_square_area := new_square_side ^ 2
  let area_decrease := square_area - new_square_area
  let percent_decrease := (area_decrease / square_area) * 100
  percent_decrease = 19 := 
by
  sorry

end percent_decrease_area_square_l161_161754


namespace problem_statement_l161_161802

noncomputable def a_b (a b : ℚ) : Prop :=
  a + b = 6 ∧ a / b = 6

theorem problem_statement (a b : ℚ) (h : a_b a b) : 
  (a * b - (a - b)) = 6 / 49 :=
by
  sorry

end problem_statement_l161_161802


namespace anita_apples_l161_161562

theorem anita_apples (num_students : ℕ) (apples_per_student : ℕ) (total_apples : ℕ) 
  (h1 : num_students = 60) 
  (h2 : apples_per_student = 6) 
  (h3 : total_apples = num_students * apples_per_student) : 
  total_apples = 360 := 
by
  sorry

end anita_apples_l161_161562


namespace geometric_series_sum_first_four_terms_l161_161213

theorem geometric_series_sum_first_four_terms :
  let a : ℝ := 1
  let r : ℝ := 1 / 3
  let n : ℕ := 4
  (a * (1 - r^n) / (1 - r)) = 40 / 27 := by
  let a : ℝ := 1
  let r : ℝ := 1 / 3
  let n : ℕ := 4
  sorry

end geometric_series_sum_first_four_terms_l161_161213


namespace max_area_right_triangle_in_semicircle_l161_161362

theorem max_area_right_triangle_in_semicircle :
  ∀ (r : ℝ), r = 1/2 → 
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ y > 0 ∧ 
  (∀ (x' y' : ℝ), x'^2 + y'^2 = r^2 ∧ y' > 0 → (1/2) * x * y ≥ (1/2) * x' * y') ∧ 
  (1/2) * x * y = 3 * Real.sqrt 3 / 32 := 
sorry

end max_area_right_triangle_in_semicircle_l161_161362


namespace problem_a_correct_answer_l161_161598

def initial_digit_eq_six (n : ℕ) : Prop :=
∃ k a : ℕ, n = 6 * 10^k + a ∧ a = n / 25

theorem problem_a_correct_answer :
  ∀ n : ℕ, initial_digit_eq_six n ↔ ∃ m : ℕ, n = 625 * 10^m :=
by
  sorry

end problem_a_correct_answer_l161_161598


namespace smaller_angle_measure_l161_161209

theorem smaller_angle_measure (x : ℝ) (a b : ℝ) (h_suppl : a + b = 180) (h_ratio : a = 4 * x ∧ b = x) :
  b = 36 :=
by
  sorry

end smaller_angle_measure_l161_161209


namespace medal_awarding_ways_l161_161414

def num_sprinters := 10
def num_americans := 4
def num_kenyans := 2
def medal_positions := 3 -- gold, silver, bronze

-- The main statement to be proven
theorem medal_awarding_ways :
  let ways_case1 := 2 * 3 * 5 * 4
  let ways_case2 := 4 * 3 * 2 * 2 * 5
  ways_case1 + ways_case2 = 360 :=
by
  sorry

end medal_awarding_ways_l161_161414


namespace focaccia_cost_l161_161058

theorem focaccia_cost :
  let almond_croissant := 4.50
  let salami_cheese_croissant := 4.50
  let plain_croissant := 3.00
  let latte := 2.50
  let total_spent := 21.00
  let known_costs := almond_croissant + salami_cheese_croissant + plain_croissant + 2 * latte
  let focaccia_cost := total_spent - known_costs
  focaccia_cost = 4.00 := 
by
  sorry

end focaccia_cost_l161_161058


namespace crayons_received_l161_161418

theorem crayons_received (crayons_left : ℕ) (crayons_lost_given_away : ℕ) (lost_twice_given : ∃ (G L : ℕ), L = 2 * G ∧ L + G = crayons_lost_given_away) :
  crayons_left = 2560 →
  crayons_lost_given_away = 9750 →
  ∃ (total_crayons_received : ℕ), total_crayons_received = 12310 :=
by
  intros h1 h2
  obtain ⟨G, L, hL, h_sum⟩ := lost_twice_given
  sorry -- Proof goes here

end crayons_received_l161_161418


namespace difference_students_rabbits_l161_161927

-- Define the number of students per classroom
def students_per_classroom := 22

-- Define the number of rabbits per classroom
def rabbits_per_classroom := 4

-- Define the number of classrooms
def classrooms := 6

-- Calculate the total number of students
def total_students := students_per_classroom * classrooms

-- Calculate the total number of rabbits
def total_rabbits := rabbits_per_classroom * classrooms

-- Prove the difference between the number of students and rabbits is 108
theorem difference_students_rabbits : total_students - total_rabbits = 108 := by
  sorry

end difference_students_rabbits_l161_161927


namespace find_all_triplets_l161_161868

theorem find_all_triplets (a b c : ℕ)
  (h₀_a : a > 0)
  (h₀_b : b > 0)
  (h₀_c : c > 0) :
  6^a = 1 + 2^b + 3^c ↔ 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 5 ∧ c = 1) :=
by
  sorry

end find_all_triplets_l161_161868


namespace Ramu_spent_on_repairs_l161_161009

theorem Ramu_spent_on_repairs (purchase_price : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (R : ℝ) 
  (h1 : purchase_price = 42000) 
  (h2 : selling_price = 61900) 
  (h3 : profit_percent = 12.545454545454545) 
  (h4 : selling_price = purchase_price + R + (profit_percent / 100) * (purchase_price + R)) : 
  R = 13000 :=
by
  sorry

end Ramu_spent_on_repairs_l161_161009


namespace base8_subtraction_l161_161627

theorem base8_subtraction : (7463 - 3154 = 4317) := by sorry

end base8_subtraction_l161_161627


namespace math_problem_l161_161597

theorem math_problem 
  (x y : ℝ) 
  (h : x^2 + y^2 - x * y = 1) 
  : (-2 ≤ x + y) ∧ (x^2 + y^2 ≤ 2) :=
by
  sorry

end math_problem_l161_161597


namespace investment_rate_l161_161466

theorem investment_rate
  (I_total I1 I2 : ℝ)
  (r1 r2 : ℝ) :
  I_total = 12000 →
  I1 = 5000 →
  I2 = 4500 →
  r1 = 0.035 →
  r2 = 0.045 →
  ∃ r3 : ℝ, (I1 * r1 + I2 * r2 + (I_total - I1 - I2) * r3) = 600 ∧ r3 = 0.089 :=
by
  intro hI_total hI1 hI2 hr1 hr2
  sorry

end investment_rate_l161_161466


namespace simplify_expression_l161_161082

theorem simplify_expression (x : ℝ) : 8 * x + 15 - 3 * x + 27 = 5 * x + 42 := 
by
  sorry

end simplify_expression_l161_161082


namespace range_of_x_l161_161877

theorem range_of_x : ∀ x : ℝ, (¬ (x + 3 = 0)) ∧ (4 - x ≥ 0) ↔ x ≤ 4 ∧ x ≠ -3 := by
  sorry

end range_of_x_l161_161877


namespace arithmetic_mean_of_first_40_consecutive_integers_l161_161055

-- Define the arithmetic sequence with the given conditions
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the given arithmetic sequence
def arithmetic_sum (a₁ d n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Define the arithmetic mean of the first n terms of the given arithmetic sequence
def arithmetic_mean (a₁ d n : ℕ) : ℚ :=
  (arithmetic_sum a₁ d n : ℚ) / n

-- The arithmetic sequence starts at 5, has a common difference of 1, and has 40 terms
theorem arithmetic_mean_of_first_40_consecutive_integers :
  arithmetic_mean 5 1 40 = 24.5 :=
by
  sorry

end arithmetic_mean_of_first_40_consecutive_integers_l161_161055


namespace area_triangle_DEF_l161_161053

theorem area_triangle_DEF 
  (DE EL EF : ℝ) (H1 : DE = 15) (H2 : EL = 12) (H3 : EF = 20) 
  (DL : ℝ) (H4 : DE^2 = EL^2 + DL^2) (H5 : DL * EF = DL * 20) :
  1/2 * EF * DL = 90 :=
by
  -- Use the assumptions and conditions to state the theorem.
  sorry

end area_triangle_DEF_l161_161053


namespace tileability_condition_l161_161580

theorem tileability_condition (a b k m n : ℕ) (h₁ : k ∣ a) (h₂ : k ∣ b) (h₃ : ∃ (t : Nat), t * (a * b) = m * n) : 
  2 * k ∣ m ∨ 2 * k ∣ n := 
sorry

end tileability_condition_l161_161580


namespace intersecting_chords_ratio_l161_161341

theorem intersecting_chords_ratio {XO YO WO ZO : ℝ} 
    (hXO : XO = 5) 
    (hWO : WO = 7) 
    (h_power_of_point : XO * YO = WO * ZO) : 
    ZO / YO = 5 / 7 :=
by
    rw [hXO, hWO] at h_power_of_point
    sorry

end intersecting_chords_ratio_l161_161341


namespace find_missing_number_l161_161761

-- Define the given numbers as a list
def given_numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14]

-- Define the arithmetic mean condition
def arithmetic_mean (xs : List ℕ) (mean : ℕ) : Prop :=
  (xs.sum + mean) / xs.length.succ = 12

-- Define the proof problem
theorem find_missing_number (x : ℕ) (h : arithmetic_mean given_numbers x) : x = 7 := 
sorry

end find_missing_number_l161_161761


namespace tangent_lines_to_circle_through_point_l161_161064

noncomputable def circle_center : ℝ × ℝ := (1, 2)
noncomputable def circle_radius : ℝ := 2
noncomputable def point_P : ℝ × ℝ := (-1, 5)

theorem tangent_lines_to_circle_through_point :
  ∃ m c : ℝ, (∀ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = 4 → (m * x + y + c = 0 → (y = -m * x - c))) ∧
  (m = 5/12 ∧ c = -55/12) ∨ (m = 0 ∧ ∀ x : ℝ, x = -1) :=
sorry

end tangent_lines_to_circle_through_point_l161_161064


namespace sum_infinite_series_l161_161682

theorem sum_infinite_series :
  (∑' n : ℕ, 1 / (n + 1) / (n + 4)) = 1 / 3 :=
sorry

end sum_infinite_series_l161_161682


namespace volume_of_prism_l161_161993

variables (a b c : ℝ)
variables (ab_prod : a * b = 36) (ac_prod : a * c = 48) (bc_prod : b * c = 72)

theorem volume_of_prism : a * b * c = 352.8 :=
by
  sorry

end volume_of_prism_l161_161993


namespace arithmetic_geometric_progression_l161_161537

theorem arithmetic_geometric_progression (a d : ℝ)
    (h1 : 2 * (a - d) * a * (a + d + 7) = 1000)
    (h2 : a^2 = 2 * (a - d) * (a + d + 7)) :
    d = 8 ∨ d = -8 := 
    sorry

end arithmetic_geometric_progression_l161_161537


namespace find_extrema_A_l161_161311

def eight_digit_number(n : ℕ) : Prop := n ≥ 10^7 ∧ n < 10^8

def coprime_with_thirtysix(n : ℕ) : Prop := Nat.gcd n 36 = 1

def transform_last_to_first(n : ℕ) : ℕ := 
  let last := n % 10
  let rest := n / 10
  last * 10^7 + rest

theorem find_extrema_A :
  ∃ (A_max A_min : ℕ), 
    (∃ B_max B_min : ℕ, 
      eight_digit_number B_max ∧ 
      eight_digit_number B_min ∧ 
      coprime_with_thirtysix B_max ∧ 
      coprime_with_thirtysix B_min ∧ 
      B_max > 77777777 ∧ 
      B_min > 77777777 ∧ 
      transform_last_to_first B_max = A_max ∧ 
      transform_last_to_first B_min = A_min) ∧ 
    A_max = 99999998 ∧ 
    A_min = 17777779 := 
  sorry

end find_extrema_A_l161_161311


namespace binom_18_4_l161_161358

theorem binom_18_4 : Nat.choose 18 4 = 3060 :=
by
  sorry

end binom_18_4_l161_161358


namespace fruit_basket_combinations_l161_161141

theorem fruit_basket_combinations (apples oranges : ℕ) (ha : apples = 6) (ho : oranges = 12) : 
  (∃ (baskets : ℕ), 
    (∀ a, 1 ≤ a ∧ a ≤ apples → ∃ b, 2 ≤ b ∧ b ≤ oranges ∧ baskets = a * b) ∧ baskets = 66) :=
by {
  sorry
}

end fruit_basket_combinations_l161_161141


namespace total_pictures_uploaded_is_65_l161_161432

-- Given conditions
def first_album_pics : ℕ := 17
def album_pics : ℕ := 8
def number_of_albums : ℕ := 6

-- The theorem to be proved
theorem total_pictures_uploaded_is_65 : first_album_pics + number_of_albums * album_pics = 65 :=
by
  sorry

end total_pictures_uploaded_is_65_l161_161432


namespace wang_hao_height_is_158_l161_161404

/-- Yao Ming's height in meters. -/
def yao_ming_height : ℝ := 2.29

/-- Wang Hao is 0.71 meters shorter than Yao Ming. -/
def height_difference : ℝ := 0.71

/-- Wang Hao's height in meters. -/
def wang_hao_height : ℝ := yao_ming_height - height_difference

theorem wang_hao_height_is_158 :
  wang_hao_height = 1.58 :=
by
  sorry

end wang_hao_height_is_158_l161_161404


namespace abs_neg_five_not_eq_five_l161_161813

theorem abs_neg_five_not_eq_five : -(abs (-5)) ≠ 5 := by
  sorry

end abs_neg_five_not_eq_five_l161_161813


namespace smallest_N_conditions_l161_161982

theorem smallest_N_conditions:
  ∃N : ℕ, N % 9 = 8 ∧
           N % 8 = 7 ∧
           N % 7 = 6 ∧
           N % 6 = 5 ∧
           N % 5 = 4 ∧
           N % 4 = 3 ∧
           N % 3 = 2 ∧
           N % 2 = 1 ∧
           N = 2519 :=
sorry

end smallest_N_conditions_l161_161982


namespace backpack_price_equation_l161_161881

-- Define the original price of the backpack
variable (x : ℝ)

-- Define the conditions
def discount1 (x : ℝ) : ℝ := 0.8 * x
def discount2 (d : ℝ) : ℝ := d - 10
def final_price (p : ℝ) : Prop := p = 90

-- Final statement to be proved
theorem backpack_price_equation : final_price (discount2 (discount1 x)) ↔ 0.8 * x - 10 = 90 := sorry

end backpack_price_equation_l161_161881


namespace chengdu_gdp_scientific_notation_l161_161355

theorem chengdu_gdp_scientific_notation :
  15000 = 1.5 * 10^4 :=
sorry

end chengdu_gdp_scientific_notation_l161_161355


namespace geometric_sequence_sum_l161_161902

theorem geometric_sequence_sum :
  ∀ (a : ℕ → ℝ) (q : ℝ),
    a 1 * q + a 1 * q ^ 3 = 20 →
    a 1 * q ^ 2 + a 1 * q ^ 4 = 40 →
    a 1 * q ^ 4 + a 1 * q ^ 6 = 160 :=
by
  sorry

end geometric_sequence_sum_l161_161902


namespace polynomial_factorization_l161_161205

-- Define the given polynomial expression
def given_poly (x : ℤ) : ℤ :=
  3 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 5 * x^2

-- Define the supposed factored form
def factored_poly (x : ℤ) : ℤ :=
  x * (3 * x^3 + 117 * x^2 + 1430 * x + 14895)

-- The theorem stating the equality of the two expressions
theorem polynomial_factorization (x : ℤ) : given_poly x = factored_poly x :=
  sorry

end polynomial_factorization_l161_161205


namespace triangle_area_l161_161714

theorem triangle_area
  (a b : ℝ)
  (C : ℝ)
  (h₁ : a = 2)
  (h₂ : b = 3)
  (h₃ : C = π / 3) :
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_area_l161_161714


namespace mathematical_proof_l161_161238

noncomputable def proof_problem (x y : ℝ) (hx_pos : y > 0) (hxy_gt2 : x + y > 2) : Prop :=
  (1 + x) / y < 2 ∨ (1 + y) / x < 2

theorem mathematical_proof (x y : ℝ) (hx_pos : y > 0) (hxy_gt2 : x + y > 2) : proof_problem x y hx_pos hxy_gt2 :=
by {
  sorry
}

end mathematical_proof_l161_161238


namespace rate_second_year_l161_161116

/-- Define the principal amount at the start. -/
def P : ℝ := 4000

/-- Define the rate of interest for the first year. -/
def rate_first_year : ℝ := 0.04

/-- Define the final amount after 2 years. -/
def A : ℝ := 4368

/-- Define the amount after the first year. -/
def P1 : ℝ := P + P * rate_first_year

/-- Define the interest for the second year. -/
def Interest2 : ℝ := A - P1

/-- Define the principal amount for the second year, which is the amount after the first year. -/
def P2 : ℝ := P1

/-- Prove that the rate of interest for the second year is 5%. -/
theorem rate_second_year : (Interest2 / P2) * 100 = 5 :=
by
  sorry

end rate_second_year_l161_161116


namespace at_least_one_casket_made_by_Cellini_son_l161_161642

-- Definitions for casket inscriptions
def golden_box := "The silver casket was made by Cellini"
def silver_box := "The golden casket was made by someone other than Cellini"

-- Predicate indicating whether a box was made by Cellini
def made_by_Cellini (box : String) : Prop :=
  box = "The golden casket was made by someone other than Cellini" ∨ box = "The silver casket was made by Cellini"

-- Our goal is to prove that at least one of the boxes was made by Cellini's son
theorem at_least_one_casket_made_by_Cellini_son :
  (¬ made_by_Cellini golden_box ∧ made_by_Cellini silver_box) ∨ (made_by_Cellini golden_box ∧ ¬ made_by_Cellini silver_box) → (¬ made_by_Cellini golden_box ∨ ¬ made_by_Cellini silver_box) :=
sorry

end at_least_one_casket_made_by_Cellini_son_l161_161642


namespace molecular_weight_constant_l161_161576

-- Given condition
def molecular_weight (compound : Type) : ℝ := 260

-- Proof problem statement (no proof yet)
theorem molecular_weight_constant (compound : Type) : molecular_weight compound = 260 :=
by
  sorry

end molecular_weight_constant_l161_161576


namespace contrapositive_of_zero_implication_l161_161685

theorem contrapositive_of_zero_implication (a b : ℝ) :
  (a = 0 ∨ b = 0 → a * b = 0) → (a * b ≠ 0 → (a ≠ 0 ∧ b ≠ 0)) :=
by
  intro h
  sorry

end contrapositive_of_zero_implication_l161_161685


namespace no_perfect_square_E_l161_161071

noncomputable def E (x : ℝ) : ℤ :=
  round x

theorem no_perfect_square_E (n : ℕ) (h : n > 0) : ¬ (∃ k : ℕ, E (n + Real.sqrt n) = k * k) :=
  sorry

end no_perfect_square_E_l161_161071


namespace solve_for_t_l161_161098

theorem solve_for_t : ∃ t : ℝ, 3 * 3^t + Real.sqrt (9 * 9^t) = 18 ∧ t = 1 :=
by
  sorry

end solve_for_t_l161_161098


namespace problem_statement_l161_161493

theorem problem_statement (x y z : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : z ≠ 0) (h₃ : x + y + z = 0) (h₄ : xy + xz + yz ≠ 0) : 
  (x^7 + y^7 + z^7) / (x * y * z * (x * y + x * z + y * z)) = -7 :=
by
  sorry

end problem_statement_l161_161493


namespace independence_test_categorical_l161_161638

-- Define what an independence test entails
def independence_test (X Y : Type) : Prop :=  
  ∃ (P : X → Y → Prop), ∀ x y1 y2, P x y1 → P x y2 → y1 = y2

-- Define the type of variables (categorical)
def is_categorical (V : Type) : Prop :=
  ∃ (f : V → ℕ), true

-- State the proposition that an independence test checks the relationship between categorical variables
theorem independence_test_categorical (X Y : Type) (hx : is_categorical X) (hy : is_categorical Y) :
  independence_test X Y := 
sorry

end independence_test_categorical_l161_161638


namespace johns_total_packs_l161_161630

-- Defining the conditions
def classes : ℕ := 6
def students_per_class : ℕ := 30
def packs_per_student : ℕ := 2

-- Theorem statement
theorem johns_total_packs : 
  (classes * students_per_class * packs_per_student) = 360 :=
by
  -- The proof would go here
  sorry

end johns_total_packs_l161_161630


namespace contractor_daily_wage_l161_161242

theorem contractor_daily_wage (total_days : ℕ) (absent_days : ℕ) (fine_per_absent_day total_amount : ℝ) (daily_wage : ℝ)
  (h_total_days : total_days = 30)
  (h_absent_days : absent_days = 8)
  (h_fine : fine_per_absent_day = 7.50)
  (h_total_amount : total_amount = 490) 
  (h_work_days : total_days - absent_days = 22)
  (h_total_fined : fine_per_absent_day * absent_days = 60)
  (h_total_earned : 22 * daily_wage - 60 = 490) :
  daily_wage = 25 := 
by 
  sorry

end contractor_daily_wage_l161_161242


namespace shortest_time_to_camp_l161_161991

/-- 
Given:
- The width of the river is 1 km.
- The camp is 1 km away from the point directly across the river.
- Swimming speed is 2 km/hr.
- Walking speed is 3 km/hr.

Prove the shortest time required to reach the camp is (2 + √5) / 6 hours.
--/
theorem shortest_time_to_camp :
  ∃ t : ℝ, t = (2 + Real.sqrt 5) / 6 := 
sorry

end shortest_time_to_camp_l161_161991


namespace minimum_value_of_x_is_4_l161_161629

-- Given conditions
variable {x : ℝ} (hx_pos : 0 < x) (h : log x ≥ log 2 + 1/2 * log x)

-- The minimum value of x is 4
theorem minimum_value_of_x_is_4 : x ≥ 4 :=
by
  sorry

end minimum_value_of_x_is_4_l161_161629


namespace number_of_terms_in_sequence_l161_161137

theorem number_of_terms_in_sequence : ∃ n : ℕ, 6 + (n-1) * 4 = 154 ∧ n = 38 :=
by
  sorry

end number_of_terms_in_sequence_l161_161137


namespace find_total_tennis_balls_l161_161760

noncomputable def original_white_balls : ℕ := sorry
noncomputable def original_yellow_balls : ℕ := sorry
noncomputable def dispatched_yellow_balls : ℕ := original_yellow_balls + 20

theorem find_total_tennis_balls
  (white_balls_eq : original_white_balls = original_yellow_balls)
  (ratio_eq : original_white_balls / dispatched_yellow_balls = 8 / 13) :
  original_white_balls + original_yellow_balls = 64 := sorry

end find_total_tennis_balls_l161_161760


namespace triangle_is_equilateral_l161_161464

   def sides_in_geometric_progression (a b c : ℝ) : Prop :=
     b^2 = a * c

   def angles_in_arithmetic_progression (A B C : ℝ) : Prop :=
     ∃ α δ : ℝ, A = α - δ ∧ B = α ∧ C = α + δ

   theorem triangle_is_equilateral {a b c A B C : ℝ} 
     (ha : a > 0) (hb : b > 0) (hc : c > 0)
     (hA : A > 0) (hB : B > 0) (hC : C > 0)
     (sum_angles : A + B + C = 180)
     (h1 : sides_in_geometric_progression a b c)
     (h2 : angles_in_arithmetic_progression A B C) : 
     a = b ∧ b = c ∧ A = 60 ∧ B = 60 ∧ C = 60 :=
   sorry
   
end triangle_is_equilateral_l161_161464


namespace mul_99_101_square_98_l161_161286

theorem mul_99_101 : 99 * 101 = 9999 := sorry

theorem square_98 : 98^2 = 9604 := sorry

end mul_99_101_square_98_l161_161286


namespace seven_y_minus_x_eq_three_l161_161788

-- Definitions for the conditions
variables (x y : ℤ)
variables (hx : x > 0)
variables (h1 : x = 11 * y + 4)
variables (h2 : 2 * x = 18 * y + 1)

-- The theorem we want to prove
theorem seven_y_minus_x_eq_three : 7 * y - x = 3 :=
by
  -- Placeholder for the proof.
  sorry

end seven_y_minus_x_eq_three_l161_161788


namespace total_cost_4kg_mangos_3kg_rice_5kg_flour_l161_161000

def cost_per_kg_mangos (M : ℝ) (R : ℝ) := (10 * M = 24 * R)
def cost_per_kg_flour_equals_rice (F : ℝ) (R : ℝ) := (6 * F = 2 * R)
def cost_of_flour (F : ℝ) := (F = 24)

theorem total_cost_4kg_mangos_3kg_rice_5kg_flour 
  (M R F : ℝ) 
  (h1 : cost_per_kg_mangos M R) 
  (h2 : cost_per_kg_flour_equals_rice F R) 
  (h3 : cost_of_flour F) : 
  4 * M + 3 * R + 5 * F = 1027.2 :=
by {
  sorry
}

end total_cost_4kg_mangos_3kg_rice_5kg_flour_l161_161000


namespace value_of_expression_l161_161960

theorem value_of_expression : (2112 - 2021) ^ 2 / 169 = 49 := by
  sorry

end value_of_expression_l161_161960


namespace sufficient_but_not_necessary_condition_l161_161356

-- Define set P
def P : Set ℝ := {1, 2, 3, 4}

-- Define set Q
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem statement
theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ∈ P → x ∈ Q) ∧ (¬(x ∈ Q → x ∈ P)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l161_161356


namespace average_of_three_numbers_is_165_l161_161942

variable (x y z : ℕ)
variable (hy : y = 90)
variable (h1 : z = 4 * y)
variable (h2 : y = 2 * x)

theorem average_of_three_numbers_is_165 : (x + y + z) / 3 = 165 := by
  sorry

end average_of_three_numbers_is_165_l161_161942


namespace largest_of_three_numbers_l161_161937

noncomputable def largest_root (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -8) (h3 : p * q * r = -20) : ℝ :=
  max p (max q r)

theorem largest_of_three_numbers (p q r : ℝ) 
  (h1 : p + q + r = 3) 
  (h2 : p * q + p * r + q * r = -8) 
  (h3 : p * q * r = -20) :
  largest_root p q r h1 h2 h3 = ( -1 + Real.sqrt 21 ) / 2 :=
by
  sorry

end largest_of_three_numbers_l161_161937


namespace find_A_find_b_and_c_l161_161233

open Real

variable {a b c A B C : ℝ}

-- Conditions for the problem
axiom triangle_sides : ∀ {A B C : ℝ}, a > 0
axiom sine_law_condition : b * sin B + c * sin C - sqrt 2 * b * sin C = a * sin A
axiom degrees_60 : B = π / 3
axiom side_a : a = 2

theorem find_A : A = π / 4 :=
by sorry

theorem find_b_and_c (h : A = π / 4) (hB : B = π / 3) (ha : a = 2) : b = sqrt 6 ∧ c = 1 + sqrt 3 :=
by sorry

end find_A_find_b_and_c_l161_161233


namespace arithmetic_mean_squares_l161_161034

theorem arithmetic_mean_squares (n : ℕ) (h : 0 < n) :
  let S_n2 := (n * (n + 1) * (2 * n + 1)) / 6 
  let A_n2 := S_n2 / n
  A_n2 = ((n + 1) * (2 * n + 1)) / 6 :=
by
  sorry

end arithmetic_mean_squares_l161_161034


namespace solve_p_value_l161_161672

noncomputable def solve_for_p (n m p : ℚ) : Prop :=
  (5 / 6 = n / 90) ∧ ((m + n) / 105 = (p - m) / 150) ∧ (p = 137.5)

theorem solve_p_value (n m p : ℚ) (h1 : 5 / 6 = n / 90) (h2 : (m + n) / 105 = (p - m) / 150) : 
  p = 137.5 :=
by
  sorry

end solve_p_value_l161_161672


namespace Sams_age_is_10_l161_161426

theorem Sams_age_is_10 (S M : ℕ) (h1 : M = S + 7) (h2 : S + M = 27) : S = 10 := 
by
  sorry

end Sams_age_is_10_l161_161426


namespace cos_value_l161_161089

theorem cos_value (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) : Real.cos (π / 3 - α) = 1 / 3 :=
  sorry

end cos_value_l161_161089


namespace smallest_prime_dividing_4_pow_11_plus_6_pow_13_l161_161345

-- Definition of the problem
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

theorem smallest_prime_dividing_4_pow_11_plus_6_pow_13 :
  ∃ p : ℕ, is_prime p ∧ p ∣ (4^11 + 6^13) ∧ ∀ q : ℕ, is_prime q ∧ q ∣ (4^11 + 6^13) → p ≤ q :=
by {
  sorry
}

end smallest_prime_dividing_4_pow_11_plus_6_pow_13_l161_161345


namespace range_of_a_l161_161689

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + a| > 2) ↔ a < -1 ∨ a > 3 :=
sorry

end range_of_a_l161_161689


namespace intersection_eq_singleton_l161_161792

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_eq_singleton :
  A ∩ B = {1} :=
sorry

end intersection_eq_singleton_l161_161792


namespace solve_for_c_l161_161564

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem solve_for_c (a b c d : ℝ) 
    (h : ∀ x : ℝ, quadratic_function a b c x ≥ d) : c = d + b^2 / (4 * a) :=
by
  sorry

end solve_for_c_l161_161564


namespace sum_of_three_numbers_l161_161152

theorem sum_of_three_numbers (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 12)
    (h4 : (a + b + c) / 3 = a + 8) (h5 : (a + b + c) / 3 = c - 18) : 
    a + b + c = 66 := 
sorry

end sum_of_three_numbers_l161_161152


namespace remove_two_fractions_sum_is_one_l161_161849

theorem remove_two_fractions_sum_is_one :
  let fractions := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let total_sum := (fractions.sum : ℚ)
  let remaining_sum := total_sum - (1/8 + 1/10)
  remaining_sum = 1 := by
    sorry

end remove_two_fractions_sum_is_one_l161_161849


namespace child_support_amount_l161_161297

-- Definitions
def base_salary_1_3 := 30000
def base_salary_4_7 := 36000
def bonus_1 := 2000
def bonus_2 := 3000
def bonus_3 := 4000
def bonus_4 := 5000
def bonus_5 := 6000
def bonus_6 := 7000
def bonus_7 := 8000
def child_support_1_5 := 30 / 100
def child_support_6_7 := 25 / 100
def paid_total := 1200

-- Total Income per year
def income_year_1 := base_salary_1_3 + bonus_1
def income_year_2 := base_salary_1_3 + bonus_2
def income_year_3 := base_salary_1_3 + bonus_3
def income_year_4 := base_salary_4_7 + bonus_4
def income_year_5 := base_salary_4_7 + bonus_5
def income_year_6 := base_salary_4_7 + bonus_6
def income_year_7 := base_salary_4_7 + bonus_7

-- Child Support per year
def support_year_1 := child_support_1_5 * income_year_1
def support_year_2 := child_support_1_5 * income_year_2
def support_year_3 := child_support_1_5 * income_year_3
def support_year_4 := child_support_1_5 * income_year_4
def support_year_5 := child_support_1_5 * income_year_5
def support_year_6 := child_support_6_7 * income_year_6
def support_year_7 := child_support_6_7 * income_year_7

-- Total Support calculation
def total_owed := support_year_1 + support_year_2 + support_year_3 + 
                  support_year_4 + support_year_5 +
                  support_year_6 + support_year_7

-- Final amount owed
def amount_owed := total_owed - paid_total

-- Theorem statement
theorem child_support_amount :
  amount_owed = 75150 :=
sorry

end child_support_amount_l161_161297


namespace problem_solution_l161_161035

noncomputable def problem (p q : ℝ) (α β γ δ : ℝ) : Prop :=
  (α^2 + p * α - 1 = 0) ∧
  (β^2 + p * β - 1 = 0) ∧
  (γ^2 + q * γ + 1 = 0) ∧
  (δ^2 + q * δ + 1 = 0) →
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = p^2 - q^2

theorem problem_solution (p q α β γ δ : ℝ) : 
  problem p q α β γ δ := 
by sorry

end problem_solution_l161_161035


namespace complementary_angle_problem_l161_161911

theorem complementary_angle_problem 
  (A B : ℝ) 
  (h1 : A + B = 90) 
  (h2 : A / B = 2 / 3) 
  (increase : A' = A * 1.20) 
  (new_sum : A' + B' = 90) 
  (B' : ℝ)
  (h3 : B' = B - B * 0.1333) :
  true := 
sorry

end complementary_angle_problem_l161_161911


namespace fraction_expression_simplifies_to_313_l161_161581

theorem fraction_expression_simplifies_to_313 :
  (12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) * (72^4 + 324) /
  (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) * (66^4 + 324) = 313 :=
by
  sorry

end fraction_expression_simplifies_to_313_l161_161581


namespace find_number_l161_161933

theorem find_number (x : ℝ) (h : 5020 - 502 / x = 5015) : x = 100.4 :=
by
  sorry

end find_number_l161_161933


namespace find_cost_10_pound_bag_l161_161252

def cost_5_pound_bag : ℝ := 13.82
def cost_25_pound_bag : ℝ := 32.25
def minimum_required_weight : ℝ := 65
def maximum_required_weight : ℝ := 80
def least_possible_cost : ℝ := 98.75
def cost_10_pound_bag (cost : ℝ) : Prop :=
  ∃ n m l, 
    (n * 5 + m * 10 + l * 25 ≥ minimum_required_weight) ∧
    (n * 5 + m * 10 + l * 25 ≤ maximum_required_weight) ∧
    (n * cost_5_pound_bag + m * cost + l * cost_25_pound_bag = least_possible_cost)

theorem find_cost_10_pound_bag : cost_10_pound_bag 2 := 
by
  sorry

end find_cost_10_pound_bag_l161_161252


namespace total_time_in_cocoons_l161_161359

theorem total_time_in_cocoons (CA CB CC: ℝ) 
    (h1: 4 * CA = 90)
    (h2: 4 * CB = 120)
    (h3: 4 * CC = 150) 
    : CA + CB + CC = 90 := 
by
  -- To be proved
  sorry

end total_time_in_cocoons_l161_161359


namespace total_water_filled_jars_l161_161405

theorem total_water_filled_jars (x : ℕ) (h : 4 * x + 2 * x + x = 14 * 4) : 3 * x = 24 :=
by
  sorry

end total_water_filled_jars_l161_161405


namespace minimum_value_l161_161674

theorem minimum_value (p q r s t u v w : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
    (ht : 0 < t) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (h₁ : p * q * r * s = 16) (h₂ : t * u * v * w = 25) :
    (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 40 := 
sorry

end minimum_value_l161_161674


namespace employee_payment_sum_l161_161840

theorem employee_payment_sum :
  ∀ (A B : ℕ), 
  (A = 3 * B / 2) → 
  (B = 180) → 
  (A + B = 450) :=
by
  intros A B hA hB
  sorry

end employee_payment_sum_l161_161840


namespace trapezium_distance_l161_161183

theorem trapezium_distance (a b h: ℝ) (area: ℝ) (h_area: area = 300) (h_sides: a = 22) (h_sides_2: b = 18)
  (h_formula: area = (1 / 2) * (a + b) * h): h = 15 :=
by
  sorry

end trapezium_distance_l161_161183


namespace base7_to_base10_l161_161634

theorem base7_to_base10 : (2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0) = 6068 := by
  sorry

end base7_to_base10_l161_161634


namespace absolute_value_condition_l161_161625

theorem absolute_value_condition (x : ℝ) : |x + 1| + |x - 2| ≤ 5 ↔ -2 ≤ x ∧ x ≤ 3 := sorry

end absolute_value_condition_l161_161625


namespace rectangular_prism_diagonal_inequality_l161_161702

variable (a b c l : ℝ)

theorem rectangular_prism_diagonal_inequality (h : l^2 = a^2 + b^2 + c^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := sorry

end rectangular_prism_diagonal_inequality_l161_161702


namespace sum_units_digits_3a_l161_161768

theorem sum_units_digits_3a (a : ℕ) (h_pos : 0 < a) (h_units : (2 * a) % 10 = 4) : 
  ((3 * (a % 10) = (6 : ℕ) ∨ (3 * (a % 10) = (21 : ℕ))) → 6 + 1 = 7) := 
by
  sorry

end sum_units_digits_3a_l161_161768


namespace rajas_salary_percentage_less_than_rams_l161_161218

-- Definitions from the problem conditions
def raja_salary : ℚ := sorry -- Placeholder, since Raja's salary doesn't need a fixed value
def ram_salary : ℚ := 1.25 * raja_salary

-- Theorem to be proved
theorem rajas_salary_percentage_less_than_rams :
  ∃ r : ℚ, (ram_salary - raja_salary) / ram_salary * 100 = 20 :=
by
  sorry

end rajas_salary_percentage_less_than_rams_l161_161218


namespace crease_length_l161_161925

noncomputable def length_of_crease (theta : ℝ) : ℝ :=
  8 * Real.sin theta

theorem crease_length (theta : ℝ) (hθ : 0 ≤ theta ∧ theta ≤ π / 2) : 
  length_of_crease theta = 8 * Real.sin theta :=
by sorry

end crease_length_l161_161925


namespace charlie_has_largest_final_answer_l161_161284

theorem charlie_has_largest_final_answer :
  let alice := (15 - 2)^2 + 3
  let bob := 15^2 - 2 + 3
  let charlie := (15 - 2 + 3)^2
  charlie > alice ∧ charlie > bob :=
by
  -- Definitions of intermediate variables
  let alice := (15 - 2)^2 + 3
  let bob := 15^2 - 2 + 3
  let charlie := (15 - 2 + 3)^2
  -- Comparison assertions
  sorry

end charlie_has_largest_final_answer_l161_161284


namespace michael_total_weight_loss_l161_161139

def weight_loss_march := 3
def weight_loss_april := 4
def weight_loss_may := 3

theorem michael_total_weight_loss : weight_loss_march + weight_loss_april + weight_loss_may = 10 := by
  sorry

end michael_total_weight_loss_l161_161139


namespace find_b_perpendicular_l161_161378

theorem find_b_perpendicular (a b : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 1 = 0 ∧ 3 * x + b * y + 5 = 0 → 
  - (a / 2) * - (3 / b) = -1) → b = -3 := 
sorry

end find_b_perpendicular_l161_161378


namespace circle_through_points_l161_161601

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l161_161601


namespace simplify_expression_l161_161211

variables {x p q r : ℝ}

theorem simplify_expression (h1 : p ≠ q) (h2 : p ≠ r) (h3 : q ≠ r) :
   ( (x + p)^4 / ((p - q) * (p - r)) + (x + q)^4 / ((q - p) * (q - r)) + (x + r)^4 / ((r - p) * (r - q)) 
   ) = p + q + r + 4 * x :=
sorry

end simplify_expression_l161_161211


namespace parallel_condition_perpendicular_condition_l161_161438

theorem parallel_condition (x : ℝ) (a b : ℝ × ℝ) :
  (a = (x, x + 2)) → (b = (1, 2)) → a.1 * b.2 = a.2 * b.1 → x = 2 := 
sorry

theorem perpendicular_condition (x : ℝ) (a b : ℝ × ℝ) :
  (a = (x, x + 2)) → (b = (1, 2)) → ((a.1 - b.1) * b.1 + (a.2 - b.2) * b.2) = 0 → x = 1 / 3 :=
sorry

end parallel_condition_perpendicular_condition_l161_161438


namespace power_24_eq_one_l161_161871

theorem power_24_eq_one (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^24 = 1 :=
by
  sorry

end power_24_eq_one_l161_161871


namespace g_extreme_value_f_ge_g_l161_161816

noncomputable def f (x : ℝ) : ℝ := Real.exp (x + 1) - 2 / x + 1
noncomputable def g (x : ℝ) : ℝ := Real.log x / x + 2

theorem g_extreme_value :
  ∃ (x : ℝ), x = Real.exp 1 ∧ g x = 1 / Real.exp 1 + 2 :=
by sorry

theorem f_ge_g (x : ℝ) (hx : 0 < x) : f x >= g x :=
by sorry

end g_extreme_value_f_ge_g_l161_161816


namespace a_plus_b_l161_161893

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x)

theorem a_plus_b (a b : ℝ) (h : f (a - 1) + f b = 0) : a + b = 1 :=
by
  sorry

end a_plus_b_l161_161893


namespace sum_of_center_coordinates_l161_161503

theorem sum_of_center_coordinates (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 7) (h2 : y1 = -6) (h3 : x2 = -5) (h4 : y2 = 4) :
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 0 := by
  -- Definitions and setup
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  sorry

end sum_of_center_coordinates_l161_161503


namespace min_value_of_x4_y3_z2_l161_161794

noncomputable def min_value_x4_y3_z2 (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem min_value_of_x4_y3_z2 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : 1/x + 1/y + 1/z = 9) : 
  min_value_x4_y3_z2 x y z = 1 / 3456 :=
by
  sorry

end min_value_of_x4_y3_z2_l161_161794


namespace length_of_chord_MN_l161_161172

theorem length_of_chord_MN 
  (m n : ℝ)
  (h1 : ∃ (M N : ℝ × ℝ), M ≠ N ∧ M.1 * M.1 + M.2 * M.2 + m * M.1 + n * M.2 - 4 = 0 ∧ N.1 * N.1 + N.2 * N.2 + m * N.1 + n * N.2 - 4 = 0 
    ∧ N.2 = M.1 ∧ N.1 = M.2) 
  (h2 : x + y = 0)
  : length_of_chord = 4 := sorry

end length_of_chord_MN_l161_161172


namespace hyperbola_min_sum_dist_l161_161109

open Real

theorem hyperbola_min_sum_dist (x y : ℝ) (F1 F2 A B : ℝ × ℝ) :
  -- Conditions for the hyperbola and the foci
  (∀ (x y : ℝ), x^2 / 9 - y^2 / 6 = 1) →
  F1 = (-c, 0) →
  F2 = (c, 0) →
  -- Minimum value of |AF2| + |BF2|
  ∃ (l : ℝ × ℝ → Prop), l F1 ∧ (∃ A B, l A ∧ l B ∧ A = (-3, y_A) ∧ B = (-3, y_B) ) →
  |dist A F2| + |dist B F2| = 16 :=
by
  sorry

end hyperbola_min_sum_dist_l161_161109


namespace one_third_percent_of_200_l161_161499

theorem one_third_percent_of_200 : ((1206 / 3) / 200) * 100 = 201 := by
  sorry

end one_third_percent_of_200_l161_161499


namespace cos_arithmetic_sequence_l161_161668

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem cos_arithmetic_sequence (a : ℕ → ℝ) (h_seq : arithmetic_sequence a) (h_sum : a 1 + a 5 + a 9 = 8 * Real.pi) :
  Real.cos (a 3 + a 7) = -1 / 2 :=
sorry

end cos_arithmetic_sequence_l161_161668


namespace find_divisor_l161_161733

theorem find_divisor (n x : ℕ) (hx : x ≠ 11) (hn : n = 386) 
  (h1 : ∃ k : ℤ, n = k * x + 1) (h2 : ∀ m : ℤ, n = 11 * m + 1 → n = 386) : x = 5 :=
  sorry

end find_divisor_l161_161733


namespace distance_to_gym_l161_161666

theorem distance_to_gym (v d : ℝ) (h_walked_200_m: 200 / v > 0) (h_double_speed: 2 * v = 2) (h_time_diff: 200 / v - d / (2 * v) = 50) : d = 300 :=
by sorry

end distance_to_gym_l161_161666


namespace inequality_system_solution_l161_161514

theorem inequality_system_solution:
  ∀ (x : ℝ),
  (1 - (2*x - 1) / 2 > (3*x - 1) / 4) ∧ (2 - 3*x ≤ 4 - x) →
  -1 ≤ x ∧ x < 1 :=
by
  intro x
  intro h
  sorry

end inequality_system_solution_l161_161514


namespace jamal_total_cost_l161_161595

-- Definitions based on conditions
def dozen := 12
def half_dozen := dozen / 2
def crayons_bought := 4 * half_dozen
def cost_per_crayon := 2
def total_cost := crayons_bought * cost_per_crayon

-- Proof statement (the question translated to a Lean theorem)
theorem jamal_total_cost : total_cost = 48 := by
  -- Proof skipped
  sorry

end jamal_total_cost_l161_161595


namespace three_digit_integers_count_l161_161923

theorem three_digit_integers_count : 
  ∃ (n : ℕ), n = 24 ∧
  (∃ (digits : Finset ℕ), digits = {2, 4, 7, 9} ∧
  (∀ a b c : ℕ, a ∈ digits → b ∈ digits → c ∈ digits → a ≠ b → b ≠ c → a ≠ c → 
  100 * a + 10 * b + c ∈ {y | 100 ≤ y ∧ y < 1000} → 4 * 3 * 2 = 24)) :=
by
  sorry

end three_digit_integers_count_l161_161923


namespace car_mpg_l161_161592

open Nat

theorem car_mpg (x : ℕ) (h1 : ∀ (m : ℕ), m = 4 * (3 * x) -> x = 27) 
                (h2 : ∀ (d1 d2 : ℕ), d2 = (4 * d1) / 3 - d1 -> d2 = 126) 
                (h3 : ∀ g : ℕ, g = 14)
                : x = 27 := 
by
  sorry

end car_mpg_l161_161592


namespace problem1_problem2_l161_161855

-- Define the conditions as noncomputable definitions
noncomputable def A : Real := sorry
noncomputable def tan_A : Real := 2
noncomputable def sin_A_plus_cos_A : Real := 1 / 5

-- Define the trigonometric identities
noncomputable def sin (x : Real) : Real := sorry
noncomputable def cos (x : Real) : Real := sorry
noncomputable def tan (x : Real) : Real := sin x / cos x

-- Ensure the conditions
axiom tan_A_condition : tan A = tan_A
axiom sin_A_plus_cos_A_condition : sin A + cos A = sin_A_plus_cos_A

-- Proof problem 1:
theorem problem1 : 
  (sin (π - A) + cos (-A)) / (sin A - sin (π / 2 + A)) = 3 := by
  sorry

-- Proof problem 2:
theorem problem2 : 
  sin A - cos A = 7 / 5 := by
  sorry

end problem1_problem2_l161_161855


namespace additional_oil_needed_l161_161819

def car_cylinders := 6
def car_oil_per_cylinder := 8
def truck_cylinders := 8
def truck_oil_per_cylinder := 10
def motorcycle_cylinders := 4
def motorcycle_oil_per_cylinder := 6

def initial_car_oil := 16
def initial_truck_oil := 20
def initial_motorcycle_oil := 8

theorem additional_oil_needed :
  let car_total_oil := car_cylinders * car_oil_per_cylinder
  let truck_total_oil := truck_cylinders * truck_oil_per_cylinder
  let motorcycle_total_oil := motorcycle_cylinders * motorcycle_oil_per_cylinder
  let car_additional_oil := car_total_oil - initial_car_oil
  let truck_additional_oil := truck_total_oil - initial_truck_oil
  let motorcycle_additional_oil := motorcycle_total_oil - initial_motorcycle_oil
  car_additional_oil = 32 ∧
  truck_additional_oil = 60 ∧
  motorcycle_additional_oil = 16 :=
by
  repeat (exact sorry)

end additional_oil_needed_l161_161819


namespace elements_map_to_4_l161_161134

def f (x : ℝ) : ℝ := x^2

theorem elements_map_to_4 :
  { x : ℝ | f x = 4 } = {2, -2} :=
by
  sorry

end elements_map_to_4_l161_161134


namespace sum_other_y_coordinates_l161_161315

-- Given points
structure Point where
  x : ℝ
  y : ℝ

def opposite_vertices (p1 p2 : Point) : Prop :=
  -- conditions defining opposite vertices of a rectangle
  (p1.x ≠ p2.x) ∧ (p1.y ≠ p2.y)

-- Function to sum y-coordinates of two points
def sum_y_coords (p1 p2 : Point) : ℝ :=
  p1.y + p2.y

-- Main theorem to prove
theorem sum_other_y_coordinates (p1 p2 : Point) (h : opposite_vertices p1 p2) :
  sum_y_coords p1 p2 = 11 ↔ 
  (p1 = {x := 1, y := 19} ∨ p1 = {x := 7, y := -8}) ∧ 
  (p2 = {x := 1, y := 19} ∨ p2 = {x := 7, y := -8}) :=
by {
  sorry
}

end sum_other_y_coordinates_l161_161315


namespace root_exists_in_interval_l161_161052

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem root_exists_in_interval :
  ∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0 := 
sorry

end root_exists_in_interval_l161_161052


namespace no_valid_m_n_l161_161004

theorem no_valid_m_n (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) : ¬ (m * n ∣ 3^m + 1 ∧ m * n ∣ 3^n + 1) :=
by
  sorry

end no_valid_m_n_l161_161004


namespace series_converges_to_one_l161_161545

noncomputable def infinite_series := ∑' n, (3^n) / (3^(2^n) + 2)

theorem series_converges_to_one :
  infinite_series = 1 := by
  sorry

end series_converges_to_one_l161_161545


namespace minimum_possible_sum_of_4x4x4_cube_l161_161740

theorem minimum_possible_sum_of_4x4x4_cube: 
  (∀ die: ℕ, (1 ≤ die) ∧ (die ≤ 6) ∧ (∃ opposite, die + opposite = 7)) → 
  (∃ sum, sum = 304) :=
by
  sorry

end minimum_possible_sum_of_4x4x4_cube_l161_161740


namespace two_digit_numbers_l161_161497

theorem two_digit_numbers :
  ∃ (x y : ℕ), 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ x < y ∧ 2000 + x + y = x * y := 
sorry

end two_digit_numbers_l161_161497


namespace at_least_two_equal_l161_161199

theorem at_least_two_equal (x y z : ℝ) (h : x / y + y / z + z / x = z / y + y / x + x / z) : 
  x = y ∨ y = z ∨ z = x := 
  sorry

end at_least_two_equal_l161_161199


namespace C_completes_work_in_4_days_l161_161781

theorem C_completes_work_in_4_days
  (A_days : ℕ)
  (B_efficiency : ℕ → ℕ)
  (C_efficiency : ℕ → ℕ)
  (hA : A_days = 12)
  (hB : ∀ {x}, B_efficiency x = x * 3 / 2)
  (hC : ∀ {x}, C_efficiency x = x * 2) :
  (1 / (1 / (C_efficiency (B_efficiency A_days)))) = 4 := by
  sorry

end C_completes_work_in_4_days_l161_161781


namespace necessary_but_not_sufficient_l161_161807

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a n = a 0 * q^n

def is_increasing_sequence (s : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, s n < s (n + 1)

def sum_first_n_terms (a : ℕ → ℝ) (q : ℝ) : ℕ → ℝ 
| 0 => a 0
| (n+1) => (sum_first_n_terms a q n) + (a 0 * q ^ n)

theorem necessary_but_not_sufficient (a : ℕ → ℝ) (q : ℝ) (h_geometric: is_geometric_sequence a q) :
  (q > 0) ∧ is_increasing_sequence (sum_first_n_terms a q) ↔ (q > 0)
:= sorry

end necessary_but_not_sufficient_l161_161807


namespace range_of_a_l161_161640

theorem range_of_a (a : ℝ) : (∃ x > 0, (2 * x - a) / (x + 1) = 1) ↔ a > -1 :=
by {
    sorry
}

end range_of_a_l161_161640


namespace number_of_red_balls_l161_161047

theorem number_of_red_balls (total_balls : ℕ) (prob_red : ℚ) (h : total_balls = 20 ∧ prob_red = 0.25) : ∃ x : ℕ, x = 5 :=
by
  sorry

end number_of_red_balls_l161_161047


namespace pencil_length_after_sharpening_l161_161536

def initial_length : ℕ := 50
def monday_sharpen : ℕ := 2
def tuesday_sharpen : ℕ := 3
def wednesday_sharpen : ℕ := 4
def thursday_sharpen : ℕ := 5

def total_sharpened : ℕ := monday_sharpen + tuesday_sharpen + wednesday_sharpen + thursday_sharpen

def final_length : ℕ := initial_length - total_sharpened

theorem pencil_length_after_sharpening : final_length = 36 := by
  -- Here would be the proof body
  sorry

end pencil_length_after_sharpening_l161_161536


namespace number_of_integers_l161_161624

theorem number_of_integers (n : ℤ) : 
  (20 < n^2) ∧ (n^2 < 150) → 
  (∃ m : ℕ, m = 16) :=
by
  sorry

end number_of_integers_l161_161624


namespace sqrt_square_sub_sqrt2_l161_161547

theorem sqrt_square_sub_sqrt2 (h : 1 < Real.sqrt 2) : Real.sqrt ((1 - Real.sqrt 2) ^ 2) = Real.sqrt 2 - 1 :=
by 
  sorry

end sqrt_square_sub_sqrt2_l161_161547


namespace decorations_total_l161_161212

def number_of_skulls : Nat := 12
def number_of_broomsticks : Nat := 4
def number_of_spiderwebs : Nat := 12
def number_of_pumpkins (spiderwebs : Nat) : Nat := 2 * spiderwebs
def number_of_cauldron : Nat := 1
def number_of_lanterns (trees : Nat) : Nat := 3 * trees
def number_of_scarecrows (trees : Nat) : Nat := 1 * (trees / 2)
def total_stickers : Nat := 30
def stickers_per_window (stickers : Nat) (windows : Nat) : Nat := (stickers / 2) / windows
def additional_decorations (bought : Nat) (used_percent : Nat) (leftover : Nat) : Nat := ((bought * used_percent) / 100) + leftover

def total_decorations : Nat :=
  number_of_skulls +
  number_of_broomsticks +
  number_of_spiderwebs +
  (number_of_pumpkins number_of_spiderwebs) +
  number_of_cauldron +
  (number_of_lanterns 5) +
  (number_of_scarecrows 4) +
  (additional_decorations 25 70 15)

theorem decorations_total : total_decorations = 102 := by
  sorry

end decorations_total_l161_161212


namespace other_number_l161_161857

theorem other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 2310) (h_hcf : Nat.gcd A B = 30) (h_A : A = 770) : B = 90 :=
by
  -- The proof is omitted here.
  sorry

end other_number_l161_161857


namespace value_of_x2_plus_y2_l161_161742

theorem value_of_x2_plus_y2 (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 :=
by
  sorry

end value_of_x2_plus_y2_l161_161742


namespace six_inch_cube_value_is_2700_l161_161113

noncomputable def value_of_six_inch_cube (value_four_inch_cube : ℕ) : ℕ :=
  let volume_four_inch_cube := 4^3
  let volume_six_inch_cube := 6^3
  let scaling_factor := volume_six_inch_cube / volume_four_inch_cube
  value_four_inch_cube * scaling_factor

theorem six_inch_cube_value_is_2700 : value_of_six_inch_cube 800 = 2700 := by
  sorry

end six_inch_cube_value_is_2700_l161_161113


namespace tax_paid_at_fifth_checkpoint_l161_161036

variable {x : ℚ}

theorem tax_paid_at_fifth_checkpoint (x : ℚ) (h : (x / 2) + (x / 2 * 1 / 3) + (x / 3 * 1 / 4) + (x / 4 * 1 / 5) + (x / 5 * 1 / 6) = 1) :
  (x / 5 * 1 / 6) = 1 / 25 :=
sorry

end tax_paid_at_fifth_checkpoint_l161_161036


namespace theater_earnings_l161_161561

theorem theater_earnings :
  let matinee_price := 5
  let evening_price := 7
  let opening_night_price := 10
  let popcorn_price := 10
  let matinee_customers := 32
  let evening_customers := 40
  let opening_night_customers := 58
  let half_of_customers_that_bought_popcorn := 
    (matinee_customers + evening_customers + opening_night_customers) / 2
  let total_earnings := 
    (matinee_price * matinee_customers) + 
    (evening_price * evening_customers) + 
    (opening_night_price * opening_night_customers) + 
    (popcorn_price * half_of_customers_that_bought_popcorn)
  total_earnings = 1670 :=
by
  sorry

end theater_earnings_l161_161561


namespace no_real_x_satisfying_quadratic_inequality_l161_161121

theorem no_real_x_satisfying_quadratic_inequality (a : ℝ) :
  ¬(∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end no_real_x_satisfying_quadratic_inequality_l161_161121


namespace contrapositive_proposition_contrapositive_equiv_l161_161716

theorem contrapositive_proposition (x : ℝ) (h : -1 < x ∧ x < 1) : (x^2 < 1) :=
sorry

theorem contrapositive_equiv (x : ℝ) (h : x^2 ≥ 1) : x ≥ 1 ∨ x ≤ -1 :=
sorry

end contrapositive_proposition_contrapositive_equiv_l161_161716


namespace correct_finance_specialization_l161_161591

-- Variables representing percentages of students specializing in different subjects
variables (students : Type) -- Type of students
           (is_specializing_finance : students → Prop) -- Predicate for finance specialization
           (is_specializing_marketing : students → Prop) -- Predicate for marketing specialization

-- Given conditions
def finance_specialization_percentage : ℝ := 0.88 -- 88% of students are taking finance specialization
def marketing_specialization_percentage : ℝ := 0.76 -- 76% of students are taking marketing specialization

-- The proof statement
theorem correct_finance_specialization (h_finance : finance_specialization_percentage = 0.88) :
  finance_specialization_percentage = 0.88 :=
by
  sorry

end correct_finance_specialization_l161_161591


namespace sid_spent_on_snacks_l161_161943

theorem sid_spent_on_snacks :
  let original_money := 48
  let money_spent_on_computer_accessories := 12
  let money_left_after_computer_accessories := original_money - money_spent_on_computer_accessories
  let remaining_money_after_purchases := 4 + original_money / 2
  ∃ snacks_cost, money_left_after_computer_accessories - snacks_cost = remaining_money_after_purchases ∧ snacks_cost = 8 :=
by
  sorry

end sid_spent_on_snacks_l161_161943


namespace coins_in_stack_l161_161785

-- Define the thickness of each coin type
def penny_thickness : ℝ := 1.55
def nickel_thickness : ℝ := 1.95
def dime_thickness : ℝ := 1.35
def quarter_thickness : ℝ := 1.75

-- Define the total stack height
def total_stack_height : ℝ := 15

-- The statement to prove
theorem coins_in_stack (pennies nickels dimes quarters : ℕ) :
  pennies * penny_thickness + nickels * nickel_thickness + 
  dimes * dime_thickness + quarters * quarter_thickness = total_stack_height →
  pennies + nickels + dimes + quarters = 9 :=
sorry

end coins_in_stack_l161_161785


namespace find_k_l161_161225

noncomputable def f (k : ℤ) (x : ℝ) := (k^2 + k - 1) * x^(k^2 - 3 * k)

-- The conditions in the problem
variables (k : ℤ) (x : ℝ)
axiom sym_y_axis : ∀ (x : ℝ), f k (-x) = f k x
axiom decreasing_on_positive : ∀ x1 x2, 0 < x1 → x1 < x2 → f k x1 > f k x2

-- The proof problem statement
theorem find_k : k = 1 :=
sorry

end find_k_l161_161225


namespace min_value_of_u_l161_161011

theorem min_value_of_u : ∀ (x y : ℝ), x ∈ Set.Ioo (-2) 2 → y ∈ Set.Ioo (-2) 2 → x * y = -1 → 
  (∀ u, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) → u ≥ 12 / 5) :=
by
  intros x y hx hy hxy u hu
  sorry

end min_value_of_u_l161_161011


namespace acute_angle_inequality_l161_161678

theorem acute_angle_inequality (a b : ℝ) (α β : ℝ) (γ : ℝ) (h : γ < π / 2) :
  (a^2 + b^2) * Real.cos (α - β) ≤ 2 * a * b :=
sorry

end acute_angle_inequality_l161_161678


namespace eggs_eaten_in_afternoon_l161_161843

theorem eggs_eaten_in_afternoon (initial : ℕ) (morning : ℕ) (final : ℕ) (afternoon : ℕ) :
  initial = 20 → morning = 4 → final = 13 → afternoon = initial - morning - final → afternoon = 3 :=
by
  intros h_initial h_morning h_final h_afternoon
  rw [h_initial, h_morning, h_final] at h_afternoon
  linarith

end eggs_eaten_in_afternoon_l161_161843


namespace max_roses_purchase_l161_161262

/--
Given three purchasing options for roses:
1. Individual roses cost $5.30 each.
2. One dozen (12) roses cost $36.
3. Two dozen (24) roses cost $50.
Given a total budget of $680, prove that the maximum number of roses that can be purchased is 317.
-/
noncomputable def max_roses : ℝ := 317

/--
Prove that given the purchasing options and the budget, the maximum number of roses that can be purchased is 317.
-/
theorem max_roses_purchase (individual_cost dozen_cost two_dozen_cost budget : ℝ) 
  (h1 : individual_cost = 5.30) 
  (h2 : dozen_cost = 36) 
  (h3 : two_dozen_cost = 50) 
  (h4 : budget = 680) : 
  max_roses = 317 := 
sorry

end max_roses_purchase_l161_161262


namespace shadow_of_tree_l161_161191

open Real

theorem shadow_of_tree (height_tree height_pole shadow_pole shadow_tree : ℝ) 
(h1 : height_tree = 12) (h2 : height_pole = 150) (h3 : shadow_pole = 100) 
(h4 : height_tree / shadow_tree = height_pole / shadow_pole) : shadow_tree = 8 := 
by 
  -- Proof will go here
  sorry

end shadow_of_tree_l161_161191


namespace variance_of_dataset_l161_161104

theorem variance_of_dataset (a : ℝ) 
  (h1 : (4 + a + 5 + 3 + 8) / 5 = a) :
  (1 / 5) * ((4 - a) ^ 2 + (a - a) ^ 2 + (5 - a) ^ 2 + (3 - a) ^ 2 + (8 - a) ^ 2) = 14 / 5 :=
by
  sorry

end variance_of_dataset_l161_161104


namespace solve_for_x_l161_161805

theorem solve_for_x (x : ℤ) (h : 15 * 2 = x - 3 + 5) : x = 28 :=
sorry

end solve_for_x_l161_161805


namespace arithmetic_sequence_term_l161_161115

theorem arithmetic_sequence_term :
  (∀ (a_n : ℕ → ℚ) (S : ℕ → ℚ),
    (∀ n, a_n n = a_n 1 + (n - 1) * 1) → -- Arithmetic sequence with common difference of 1
    (∀ n, S n = n * a_n 1 + (n * (n - 1)) / 2) →  -- Sum of first n terms of sequence
    S 8 = 4 * S 4 →
    a_n 10 = 19 / 2) :=
by
  intros a_n S ha_n hSn hS8_eq
  sorry

end arithmetic_sequence_term_l161_161115


namespace line_intercept_form_l161_161428

theorem line_intercept_form 
  (P : ℝ × ℝ) 
  (a : ℝ × ℝ) 
  (l_eq : ∃ m : ℝ, ∀ x y : ℝ, (x, y) = P → y - 3 = m * (x - 2))
  (P_coord : P = (2, 3)) 
  (a_vect : a = (2, -6)) 
  : ∀ x y : ℝ, y - 3 = (-3) * (x - 2) → 3 * x + y - 9 = 0 →  ∃ a' b' : ℝ, a' ≠ 0 ∧ b' ≠ 0 ∧ x / 3 + y / 9 = 1 :=
by
  sorry

end line_intercept_form_l161_161428


namespace less_than_reciprocal_l161_161750

theorem less_than_reciprocal (a b c d e : ℝ) (ha : a = -3) (hb : b = -1/2) (hc : c = 0.5) (hd : d = 1) (he : e = 3) :
  (a < 1 / a) ∧ (c < 1 / c) ∧ ¬(b < 1 / b) ∧ ¬(d < 1 / d) ∧ ¬(e < 1 / e) :=
by
  sorry

end less_than_reciprocal_l161_161750


namespace exists_divisible_diff_l161_161917

theorem exists_divisible_diff (l : List ℤ) (h_len : l.length = 2022) :
  ∃ i j, i ≠ j ∧ (l.nthLe i sorry - l.nthLe j sorry) % 2021 = 0 :=
by
  apply sorry -- Placeholder for proof

end exists_divisible_diff_l161_161917


namespace Rahul_batting_average_l161_161059

theorem Rahul_batting_average 
  (A : ℕ) (current_matches : ℕ := 12) (new_matches : ℕ := 13) (scored_today : ℕ := 78) (new_average : ℕ := 54)
  (h1 : (A * current_matches + scored_today) = new_average * new_matches) : A = 52 := 
by
  sorry

end Rahul_batting_average_l161_161059


namespace find_y_l161_161189

theorem find_y (x y : ℝ) 
  (h1 : 2 * x^2 + 6 * x + 4 * y + 2 = 0)
  (h2 : 3 * x + y + 4 = 0) :
  y^2 + 17 * y - 11 = 0 :=
by 
  sorry

end find_y_l161_161189


namespace find_g_3_l161_161409

-- Definitions and conditions
variable (g : ℝ → ℝ)
variable (h : ∀ x : ℝ, g (x - 1) = 2 * x + 6)

-- Theorem: Proof problem corresponding to the problem
theorem find_g_3 : g 3 = 14 :=
by
  -- Insert proof here
  sorry

end find_g_3_l161_161409


namespace solve_quadratic_difference_l161_161799

theorem solve_quadratic_difference :
  ∀ x : ℝ, (x^2 - 7*x - 48 = 0) → 
  let x1 := (7 + Real.sqrt 241) / 2
  let x2 := (7 - Real.sqrt 241) / 2
  abs (x1 - x2) = Real.sqrt 241 :=
by
  sorry

end solve_quadratic_difference_l161_161799


namespace integer_solutions_of_equation_l161_161447

def satisfies_equation (x y : ℤ) : Prop :=
  x * y - 2 * x - 2 * y + 7 = 0

theorem integer_solutions_of_equation :
  { (x, y) : ℤ × ℤ | satisfies_equation x y } = { (5, 1), (-1, 3), (3, -1), (1, 5) } :=
by sorry

end integer_solutions_of_equation_l161_161447


namespace problem1_problem2_l161_161735

-- Problem 1
theorem problem1 (a : ℝ) : 3 * a ^ 2 - 2 * a + 1 + (3 * a - a ^ 2 + 2) = 2 * a ^ 2 + a + 3 :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : x - 2 * (x - 3 / 2 * y) + 3 * (x - x * y) = 2 * x + 3 * y - 3 * x * y :=
by
  sorry

end problem1_problem2_l161_161735


namespace has_only_one_minimum_point_and_no_maximum_point_l161_161759

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3

theorem has_only_one_minimum_point_and_no_maximum_point :
  ∃! c : ℝ, (deriv f c = 0 ∧ ∀ x < c, deriv f x < 0 ∧ ∀ x > c, deriv f x > 0) ∧
  ∀ x, f x ≥ f c ∧ (∀ x, deriv f x > 0 ∨ deriv f x < 0) := sorry

end has_only_one_minimum_point_and_no_maximum_point_l161_161759


namespace remainder_p11_minus_3_div_p_minus_2_l161_161354

def f (p : ℕ) : ℕ := p^11 - 3

theorem remainder_p11_minus_3_div_p_minus_2 : f 2 = 2045 := 
by 
  sorry

end remainder_p11_minus_3_div_p_minus_2_l161_161354


namespace sum_of_other_endpoint_coordinates_l161_161928

theorem sum_of_other_endpoint_coordinates :
  ∃ (x y: ℤ), (8 + x) / 2 = 6 ∧ y / 2 = -10 ∧ x + y = -16 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l161_161928


namespace shaded_triangle_area_l161_161126

-- Definitions and conditions
def grid_width : ℕ := 15
def grid_height : ℕ := 5

def larger_triangle_base : ℕ := grid_width
def larger_triangle_height : ℕ := grid_height - 1

def smaller_triangle_base : ℕ := 12
def smaller_triangle_height : ℕ := 3

-- The proof problem stating that the area of the smaller shaded triangle is 18 units
theorem shaded_triangle_area :
  (smaller_triangle_base * smaller_triangle_height) / 2 = 18 :=
by
  sorry

end shaded_triangle_area_l161_161126


namespace snow_shoveling_l161_161738

noncomputable def volume_of_snow_shoveled (length1 length2 width depth1 depth2 : ℝ) : ℝ :=
  (length1 * width * depth1) + (length2 * width * depth2)

theorem snow_shoveling :
  volume_of_snow_shoveled 15 15 4 1 (1 / 2) = 90 :=
by
  sorry

end snow_shoveling_l161_161738


namespace value_of_fraction_l161_161371

-- Lean 4 statement
theorem value_of_fraction (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : (x + y) / 3 = 5 / 3 :=
by
  sorry

end value_of_fraction_l161_161371


namespace eval_expression_l161_161705

theorem eval_expression :
  let a := 3
  let b := 2
  (2 ^ a ∣ 200) ∧ ¬(2 ^ (a + 1) ∣ 200) ∧ (5 ^ b ∣ 200) ∧ ¬(5 ^ (b + 1) ∣ 200)
→ (1 / 3)^(b - a) = 3 :=
by sorry

end eval_expression_l161_161705


namespace quadratic_inequality_solution_l161_161723

theorem quadratic_inequality_solution (x : ℝ) : 
    (x^2 - 3*x - 4 > 0) ↔ (x < -1 ∨ x > 4) :=
sorry

end quadratic_inequality_solution_l161_161723


namespace estimate_white_balls_l161_161544

theorem estimate_white_balls :
  (∃ x : ℕ, (6 / (x + 6) : ℝ) = 0.2 ∧ x = 24) :=
by
  use 24
  sorry

end estimate_white_balls_l161_161544


namespace solution_set_of_inequality_l161_161573

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) - 3

theorem solution_set_of_inequality :
  { x : ℝ | f x < 0 } = { x : ℝ | x < Real.log 3 / Real.log 2 } :=
by
  sorry

end solution_set_of_inequality_l161_161573


namespace max_remainder_l161_161440

theorem max_remainder : ∃ n r, 2013 ≤ n ∧ n ≤ 2156 ∧ (n % 5 = r) ∧ (n % 11 = r) ∧ (n % 13 = r) ∧ (r ≤ 4) ∧ (∀ m, 2013 ≤ m ∧ m ≤ 2156 ∧ (m % 5 = r) ∧ (m % 11 = r) ∧ (m % 13 = r) ∧ (m ≤ n) ∧ (r ≤ 4) → r ≤ 4) := sorry

end max_remainder_l161_161440


namespace problem_statement_l161_161038

variable (P : ℕ → Prop)

theorem problem_statement
    (h1 : P 2)
    (h2 : ∀ k : ℕ, k > 0 → P k → P (k + 2)) :
    ∀ n : ℕ, n > 0 → 2 ∣ n → P n :=
by
  sorry

end problem_statement_l161_161038


namespace largest_five_digit_number_divisible_by_6_l161_161633

theorem largest_five_digit_number_divisible_by_6 : 
  ∃ n : ℕ, n < 100000 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 100000 ∧ m % 6 = 0 → m ≤ n :=
by
  sorry

end largest_five_digit_number_divisible_by_6_l161_161633


namespace percentage_of_150_l161_161995

theorem percentage_of_150 : (1 / 5 * (1 / 100) * 150 : ℝ) = 0.3 := by
  sorry

end percentage_of_150_l161_161995


namespace minimum_value_of_fraction_l161_161971

theorem minimum_value_of_fraction {x : ℝ} (hx : x ≥ 3/2) :
  ∃ y : ℝ, y = (2 * (x - 1) + (1 / (x - 1)) + 2) ∧ y = 2 * Real.sqrt 2 + 2 :=
sorry

end minimum_value_of_fraction_l161_161971


namespace pure_imaginary_solution_l161_161987

-- Defining the main problem as a theorem in Lean 4

theorem pure_imaginary_solution (m : ℝ) : 
  (∃ a b : ℝ, (m^2 - m = a ∧ a = 0) ∧ (m^2 - 3 * m + 2 = b ∧ b ≠ 0)) → 
  m = 0 :=
sorry -- Proof is omitted as per the instructions

end pure_imaginary_solution_l161_161987


namespace dimensions_multiple_of_three_l161_161888

theorem dimensions_multiple_of_three (a b c : ℤ) (h : a * b * c = (a + 1) * (b + 1) * (c - 2)) :
  (a % 3 = 0) ∨ (b % 3 = 0) ∨ (c % 3 = 0) :=
sorry

end dimensions_multiple_of_three_l161_161888


namespace fraction_of_speedsters_l161_161040

/-- Let S denote the total number of Speedsters and T denote the total inventory. 
    Given the following conditions:
    1. 54 Speedster convertibles constitute 3/5 of all Speedsters (S).
    2. There are 30 vehicles that are not Speedsters.

    Prove that the fraction of the current inventory that is Speedsters is 3/4.
-/
theorem fraction_of_speedsters (S T : ℕ)
  (h1 : 3 / 5 * S = 54)
  (h2 : T = S + 30) :
  (S : ℚ) / T = 3 / 4 :=
by
  sorry

end fraction_of_speedsters_l161_161040


namespace small_pos_int_n_l161_161998

theorem small_pos_int_n (a : ℕ → ℕ) (n : ℕ) (a1_val : a 1 = 7)
  (recurrence: ∀ n, a (n + 1) = a n * (a n + 2)) :
  ∃ n : ℕ, a n > 2 ^ 4036 ∧ ∀ m : ℕ, (m < n) → a m ≤ 2 ^ 4036 :=
by
  sorry

end small_pos_int_n_l161_161998


namespace part1_part2_l161_161731

def f (x a : ℝ) := |x - a| + x

theorem part1 (a : ℝ) (h_a : a = 1) : 
  {x : ℝ | f x a ≥ x + 2} = {x | x ≥ 3} ∪ {x | x ≤ -1} := 
by
  sorry

theorem part2 (a : ℝ) (h : {x : ℝ | f x a ≤ 3 * x} = {x | x ≥ 2}) : 
  a = 6 := 
by
  sorry

end part1_part2_l161_161731


namespace math_problem_l161_161353

open Real

theorem math_problem
  (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h : x^2 + y^2 + z^2 = 3) :
  sqrt (3 - ( (x + y) / 2) ^ 2) + sqrt (3 - ( (y + z) / 2) ^ 2) + sqrt (3 - ( (z + x) / 2) ^ 2) ≥ 3 * sqrt 2 :=
by 
  sorry

end math_problem_l161_161353


namespace sum_of_consecutive_evens_is_162_l161_161337

-- Define the smallest even number
def smallest_even : ℕ := 52

-- Define the next two consecutive even numbers
def second_even : ℕ := smallest_even + 2
def third_even : ℕ := smallest_even + 4

-- The sum of these three even numbers
def sum_of_consecutive_evens : ℕ := smallest_even + second_even + third_even

-- Assertion that the sum must be 162
theorem sum_of_consecutive_evens_is_162 : sum_of_consecutive_evens = 162 :=
by 
  -- To be proved
  sorry

end sum_of_consecutive_evens_is_162_l161_161337


namespace percentage_error_l161_161206

theorem percentage_error (x : ℚ) : 
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  percentage_error = 64 :=
by
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  sorry

end percentage_error_l161_161206


namespace problem_statement_l161_161018

theorem problem_statement :
  (¬ (∀ x : ℝ, 2 * x < 3 * x)) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := 
sorry

end problem_statement_l161_161018


namespace pool_capacity_l161_161267

theorem pool_capacity:
  (∃ (V1 V2 : ℝ) (t : ℝ), 
    (V1 = t / 120) ∧ 
    (V2 = V1 + 50) ∧ 
    (V1 + V2 = t / 48) ∧ 
    t = 12000) := 
by 
  sorry

end pool_capacity_l161_161267


namespace range_of_a_l161_161070

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 0 → a * 4^x - 2^x + 2 > 0) → a > -1 :=
by sorry

end range_of_a_l161_161070


namespace martha_blue_butterflies_l161_161574

variables (B Y : Nat)

theorem martha_blue_butterflies (h_total : B + Y + 5 = 11) (h_twice : B = 2 * Y) : B = 4 :=
by
  sorry

end martha_blue_butterflies_l161_161574


namespace a5_value_l161_161050

-- Definitions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n, a (n + 1) = q * a n

def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

def product_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n + 1) = 2^(2 * n + 1)

-- Theorem statement
theorem a5_value (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_pos : positive_terms a) (h_prod : product_condition a) : a 5 = 32 :=
sorry

end a5_value_l161_161050


namespace f_at_zero_f_positive_f_increasing_l161_161502

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined (x : ℝ) : true
axiom f_nonzero : f 0 ≠ 0
axiom f_pos_gt1 (x : ℝ) : x > 0 → f x > 1
axiom f_add (a b : ℝ) : f (a + b) = f a * f b

theorem f_at_zero : f 0 = 1 :=
sorry

theorem f_positive (x : ℝ) : f x > 0 :=
sorry

theorem f_increasing (x₁ x₂ : ℝ) : x₁ < x₂ → f x₁ < f x₂ :=
sorry

end f_at_zero_f_positive_f_increasing_l161_161502


namespace no_function_f_l161_161439

noncomputable def g (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_function_f (a b c : ℝ) (h : ∀ x, g a b c (g a b c x) = x) :
  ¬ ∃ f : ℝ → ℝ, ∀ x, f (f x) = g a b c x := 
sorry

end no_function_f_l161_161439


namespace cost_per_board_game_is_15_l161_161752

-- Definitions of the conditions
def number_of_board_games : ℕ := 6
def bill_paid : ℕ := 100
def bill_value : ℕ := 5
def bills_received : ℕ := 2

def total_change := bills_received * bill_value
def total_cost := bill_paid - total_change
def cost_per_board_game := total_cost / number_of_board_games

-- The theorem stating that the cost of each board game is $15
theorem cost_per_board_game_is_15 : cost_per_board_game = 15 := 
by
  -- Omitted proof steps
  sorry

end cost_per_board_game_is_15_l161_161752


namespace polygon_sides_l161_161193

theorem polygon_sides (sum_of_interior_angles : ℝ) (x : ℝ) (h : sum_of_interior_angles = 1080) : x = 8 :=
by
  sorry

end polygon_sides_l161_161193


namespace largest_angle_in_scalene_triangle_l161_161483

-- Define the conditions of the problem
def is_scalene (D E F : ℝ) : Prop :=
  D ≠ E ∧ D ≠ F ∧ E ≠ F

def angle_sum (D E F : ℝ) : Prop :=
  D + E + F = 180

def given_angles (D E : ℝ) : Prop :=
  D = 30 ∧ E = 50

-- Statement of the problem
theorem largest_angle_in_scalene_triangle :
  ∀ (D E F : ℝ), is_scalene D E F ∧ given_angles D E ∧ angle_sum D E F → F = 100 :=
by
  intros D E F h
  sorry

end largest_angle_in_scalene_triangle_l161_161483


namespace ralph_total_cost_correct_l161_161763

noncomputable def calculate_total_cost : ℝ :=
  let original_cart_cost := 54.00
  let small_issue_item_original := 20.00
  let additional_item_original := 15.00
  let small_issue_discount := 0.20
  let additional_item_discount := 0.25
  let coupon_discount := 0.10
  let sales_tax := 0.07

  -- Calculate the discounted prices
  let small_issue_discounted := small_issue_item_original * (1 - small_issue_discount)
  let additional_item_discounted := additional_item_original * (1 - additional_item_discount)

  -- Total cost before the coupon and tax
  let total_before_coupon := original_cart_cost + small_issue_discounted + additional_item_discounted

  -- Apply the coupon discount
  let total_after_coupon := total_before_coupon * (1 - coupon_discount)

  -- Apply the sales tax
  total_after_coupon * (1 + sales_tax)

-- Define the problem statement
theorem ralph_total_cost_correct : calculate_total_cost = 78.24 :=
by sorry

end ralph_total_cost_correct_l161_161763


namespace smallest_nonprime_in_range_l161_161934

def smallest_nonprime_with_no_prime_factors_less_than_20 (m : ℕ) : Prop :=
  ¬(Nat.Prime m) ∧ m > 10 ∧ ∀ p : ℕ, Nat.Prime p → p < 20 → ¬(p ∣ m)

theorem smallest_nonprime_in_range :
  smallest_nonprime_with_no_prime_factors_less_than_20 529 ∧ 520 < 529 ∧ 529 ≤ 540 := 
by 
  sorry

end smallest_nonprime_in_range_l161_161934


namespace exists_not_holds_l161_161357

variable (S : Type) [Nonempty S] [Inhabited S]
variable (op : S → S → S)
variable (h : ∀ a b : S, op a (op b a) = b)

theorem exists_not_holds : ∃ a b : S, (op (op a b) a) ≠ a := sorry

end exists_not_holds_l161_161357


namespace find_a_and_b_l161_161416

def star (a b : ℕ) : ℕ := a^b + a * b

theorem find_a_and_b (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 40) : a + b = 7 :=
by
  sorry

end find_a_and_b_l161_161416


namespace ratio_M_N_l161_161343

variable {R P M N : ℝ}

theorem ratio_M_N (h1 : P = 0.3 * R) (h2 : M = 0.35 * R) (h3 : N = 0.55 * R) : M / N = 7 / 11 := by
  sorry

end ratio_M_N_l161_161343


namespace complete_collection_prob_l161_161839

noncomputable def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem complete_collection_prob : 
  let n := 18
  let k := 10
  let needed := 6
  let uncollected_combinations := C 6 6
  let collected_combinations := C 12 4
  let total_combinations := C 18 10
  (uncollected_combinations * collected_combinations) / total_combinations = 5 / 442 :=
by 
  sorry

end complete_collection_prob_l161_161839


namespace inequality_solution_set_l161_161646

variable {a b x : ℝ}

theorem inequality_solution_set (h : ∀ x : ℝ, ax - b > 0 ↔ x < -1) : 
  ∀ x : ℝ, (x-2) * (ax + b) < 0 ↔ x < 1 ∨ x > 2 :=
by sorry

end inequality_solution_set_l161_161646


namespace conjunction_used_in_proposition_l161_161231

theorem conjunction_used_in_proposition (x : ℝ) (h : x^2 = 4) :
  (x = 2 ∨ x = -2) :=
sorry

end conjunction_used_in_proposition_l161_161231


namespace no_solution_for_a_l161_161739

theorem no_solution_for_a {a : ℝ} :
  (a ∈ Set.Iic (-32) ∪ Set.Ici 0) →
  ¬ ∃ x : ℝ,  9 * |x - 4 * a| + |x - a^2| + 8 * x - 4 * a = 0 :=
by
  intro h
  sorry

end no_solution_for_a_l161_161739


namespace apples_in_bowl_l161_161039

variable {A : ℕ}

theorem apples_in_bowl
  (initial_oranges : ℕ)
  (removed_oranges : ℕ)
  (final_oranges : ℕ)
  (total_fruit : ℕ)
  (fraction_apples : ℚ) :
  initial_oranges = 25 →
  removed_oranges = 19 →
  final_oranges = initial_oranges - removed_oranges →
  fraction_apples = (70 : ℚ) / (100 : ℚ) →
  final_oranges = total_fruit * (30 : ℚ) / (100 : ℚ) →
  A = total_fruit * fraction_apples →
  A = 14 :=
by
  sorry

end apples_in_bowl_l161_161039


namespace journey_duration_is_9_hours_l161_161983

noncomputable def journey_time : ℝ :=
  let d1 := 90 -- Distance traveled by Tom and Dick by car before Tom got off
  let d2 := 60 -- Distance Dick backtracked to pick up Harry
  let T := (d1 / 30) + ((120 - d1) / 5) -- Time taken for Tom's journey
  T

theorem journey_duration_is_9_hours : journey_time = 9 := 
by 
  sorry

end journey_duration_is_9_hours_l161_161983


namespace arithmetic_sequence_a5_l161_161161

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h1 : a 1 = 3) (h3 : a 3 = 5) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) : 
  a 5 = 7 :=
by
  -- proof to be filled later
  sorry

end arithmetic_sequence_a5_l161_161161


namespace original_flow_rate_l161_161867

theorem original_flow_rate :
  ∃ F : ℚ, 
  (F * 0.75 * 0.4 * 0.6 - 1 = 2) ∧
  (F = 50/3) :=
by
  sorry

end original_flow_rate_l161_161867


namespace intersection_correct_l161_161498

def set_A : Set ℤ := {-1, 1, 2, 4}
def set_B : Set ℤ := {x | |x - 1| ≤ 1}

theorem intersection_correct :
  set_A ∩ set_B = {1, 2} :=
  sorry

end intersection_correct_l161_161498


namespace larger_root_eq_5_over_8_l161_161664

noncomputable def find_larger_root : ℝ := 
    let x := ((5:ℝ) / 8)
    let y := ((23:ℝ) / 48)
    if x > y then x else y

theorem larger_root_eq_5_over_8 (x : ℝ) (y : ℝ) : 
  (x - ((5:ℝ) / 8)) * (x - ((5:ℝ) / 8)) + (x - ((5:ℝ) / 8)) * (x - ((1:ℝ) / 3)) = 0 → 
  find_larger_root = ((5:ℝ) / 8) :=
by
  intro h
  -- proof goes here
  sorry

end larger_root_eq_5_over_8_l161_161664


namespace no_prime_factor_congruent_to_7_mod_8_l161_161769

open Nat

theorem no_prime_factor_congruent_to_7_mod_8 (n : ℕ) (hn : 0 < n) : 
  ¬ (∃ p : ℕ, p.Prime ∧ p ∣ 2^n + 1 ∧ p % 8 = 7) :=
sorry

end no_prime_factor_congruent_to_7_mod_8_l161_161769


namespace james_main_game_time_l161_161675

-- Define the download time in minutes
def download_time := 10

-- Define the installation time as half the download time
def installation_time := download_time / 2

-- Define the combined time for download and installation
def combined_time := download_time + installation_time

-- Define the tutorial time as triple the combined time
def tutorial_time := combined_time * 3

-- Define the total time as the combined time plus the tutorial time
def total_time := combined_time + tutorial_time

-- Statement of the problem to prove
theorem james_main_game_time : total_time = 60 := by
  sorry

end james_main_game_time_l161_161675


namespace jars_of_peanut_butter_l161_161236

theorem jars_of_peanut_butter (x : Nat) : 
  (16 * x + 28 * x + 40 * x + 52 * x = 2032) → 
  (4 * x = 60) :=
by
  intro h
  sorry

end jars_of_peanut_butter_l161_161236


namespace area_of_BCD_l161_161168

theorem area_of_BCD (S_ABC : ℝ) (a_CD : ℝ) (h_ratio : ℝ) (h_ABC : ℝ) :
  S_ABC = 36 ∧ a_CD = 30 ∧ h_ratio = 0.5 ∧ h_ABC = 12 → 
  (1 / 2) * a_CD * (h_ratio * h_ABC) = 90 :=
by
  intros h
  sorry

end area_of_BCD_l161_161168


namespace total_paintable_wall_area_l161_161421

/-- 
  Conditions:
  - John's house has 4 bedrooms.
  - Each bedroom is 15 feet long, 12 feet wide, and 10 feet high.
  - Doorways, windows, and a fireplace occupy 85 square feet per bedroom.
  Question: Prove that the total paintable wall area is 1820 square feet.
--/
theorem total_paintable_wall_area 
  (num_bedrooms : ℕ)
  (length width height non_paintable_area : ℕ)
  (h_num_bedrooms : num_bedrooms = 4)
  (h_length : length = 15)
  (h_width : width = 12)
  (h_height : height = 10)
  (h_non_paintable_area : non_paintable_area = 85) :
  (num_bedrooms * ((2 * (length * height) + 2 * (width * height)) - non_paintable_area) = 1820) :=
by
  sorry

end total_paintable_wall_area_l161_161421


namespace Max_students_count_l161_161941

variables (M J : ℕ)

theorem Max_students_count :
  (M = 2 * J + 100) → 
  (M + J = 5400) → 
  M = 3632 := 
by 
  intros h1 h2
  sorry

end Max_students_count_l161_161941


namespace sum_of_digits_of_n_l161_161588

theorem sum_of_digits_of_n : 
  ∃ n : ℕ, n > 1500 ∧ 
    (Nat.gcd 40 (n + 105) = 10) ∧ 
    (Nat.gcd (n + 40) 105 = 35) ∧ 
    (Nat.digits 10 n).sum = 8 :=
by 
  sorry

end sum_of_digits_of_n_l161_161588


namespace max_trig_expression_l161_161348

theorem max_trig_expression (A : ℝ) : (2 * Real.sin (A / 2) + Real.cos (A / 2) ≤ Real.sqrt 3) :=
sorry

end max_trig_expression_l161_161348


namespace negated_proposition_false_l161_161105

theorem negated_proposition_false : ¬ ∀ x : ℝ, 2^x + x^2 > 1 :=
by 
sorry

end negated_proposition_false_l161_161105


namespace index_card_area_l161_161448

theorem index_card_area
  (L W : ℕ)
  (h1 : L = 4)
  (h2 : W = 6)
  (h3 : (L - 1) * W = 18) :
  (L * (W - 1) = 20) :=
by
  sorry

end index_card_area_l161_161448


namespace hyperbola_equation_l161_161667

-- Definitions based on the conditions
def parabola_focus : (ℝ × ℝ) := (2, 0)
def point_on_hyperbola : (ℝ × ℝ) := (1, 0)
def hyperbola_center : (ℝ × ℝ) := (0, 0)
def right_focus_of_hyperbola : (ℝ × ℝ) := parabola_focus

-- Given the above definitions, we should prove that the standard equation of hyperbola C is correct
theorem hyperbola_equation :
  ∃ (a b : ℝ), (a = 1) ∧ (2^2 = a^2 + b^2) ∧
  (hyperbola_center = (0, 0)) ∧ (point_on_hyperbola = (1, 0)) →
  (x^2 - (y^2 / 3) = 1) :=
by
  sorry

end hyperbola_equation_l161_161667


namespace sam_total_distance_l161_161273

-- Definitions based on conditions
def first_half_distance : ℕ := 120
def first_half_time : ℕ := 3
def second_half_distance : ℕ := 80
def second_half_time : ℕ := 2
def sam_time : ℚ := 5.5

-- Marguerite's overall average speed
def marguerite_average_speed : ℚ := (first_half_distance + second_half_distance) / (first_half_time + second_half_time)

-- Theorem statement: Sam's total distance driven
theorem sam_total_distance : ∀ (d : ℚ), d = (marguerite_average_speed * sam_time) ↔ d = 220 := by
  intro d
  sorry

end sam_total_distance_l161_161273


namespace count_multiples_5_or_7_but_not_both_l161_161747

-- Definitions based on the given problem conditions
def multiples_of_five (n : Nat) : Nat :=
  (n - 1) / 5

def multiples_of_seven (n : Nat) : Nat :=
  (n - 1) / 7

def multiples_of_thirty_five (n : Nat) : Nat :=
  (n - 1) / 35

def count_multiples (n : Nat) : Nat :=
  (multiples_of_five n) + (multiples_of_seven n) - 2 * (multiples_of_thirty_five n)

-- The main statement to be proved
theorem count_multiples_5_or_7_but_not_both : count_multiples 101 = 30 :=
by
  sorry

end count_multiples_5_or_7_but_not_both_l161_161747


namespace quotient_is_eight_l161_161234

theorem quotient_is_eight (d v r q : ℕ) (h₁ : d = 141) (h₂ : v = 17) (h₃ : r = 5) (h₄ : d = v * q + r) : q = 8 :=
by
  sorry

end quotient_is_eight_l161_161234


namespace arithmetic_sequence_integers_l161_161114

theorem arithmetic_sequence_integers (a3 a18 : ℝ) (d : ℝ) (n : ℕ)
  (h3 : a3 = 14) (h18 : a18 = 23) (hd : d = 0.6)
  (hn : n = 2010) : 
  (∃ (k : ℕ), n = 5 * (k + 1) - 2) ∧ (k ≤ 401) :=
by
  sorry

end arithmetic_sequence_integers_l161_161114


namespace pow_div_eq_l161_161219

theorem pow_div_eq : (8:ℕ) ^ 15 / (64:ℕ) ^ 6 = 512 := by
  have h1 : 64 = 8 ^ 2 := by sorry
  have h2 : (64:ℕ) ^ 6 = (8 ^ 2) ^ 6 := by sorry
  have h3 : (8 ^ 2) ^ 6 = 8 ^ 12 := by sorry
  have h4 : (8:ℕ) ^ 15 / (8 ^ 12) = 8 ^ (15 - 12) := by sorry
  have h5 : 8 ^ 3 = 512 := by sorry
  exact sorry

end pow_div_eq_l161_161219


namespace cone_lateral_surface_area_l161_161621

theorem cone_lateral_surface_area (r l : ℝ) (h_r : r = 2) (h_l : l = 3) : 
  (r * l * Real.pi = 6 * Real.pi) := by
  sorry

end cone_lateral_surface_area_l161_161621


namespace find_special_two_digit_numbers_l161_161756

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_special (A : ℕ) : Prop :=
  let sum_A := sum_digits A
  sum_A^2 = sum_digits (A^2)

theorem find_special_two_digit_numbers :
  {A : ℕ | 10 ≤ A ∧ A < 100 ∧ is_special A} = {10, 11, 12, 13, 20, 21, 22, 30, 31} :=
by 
  sorry

end find_special_two_digit_numbers_l161_161756


namespace industrial_lubricants_percentage_l161_161175

noncomputable def percentage_microphotonics : ℕ := 14
noncomputable def percentage_home_electronics : ℕ := 19
noncomputable def percentage_food_additives : ℕ := 10
noncomputable def percentage_gmo : ℕ := 24
noncomputable def total_percentage : ℕ := 100
noncomputable def percentage_basic_astrophysics : ℕ := 25

theorem industrial_lubricants_percentage :
  total_percentage - (percentage_microphotonics + percentage_home_electronics + 
  percentage_food_additives + percentage_gmo + percentage_basic_astrophysics) = 8 := 
sorry

end industrial_lubricants_percentage_l161_161175


namespace puzzle_solution_l161_161587

theorem puzzle_solution :
  (∀ n m k : ℕ, n + m + k = 111 → 9 * (n + m + k) / 3 = 9) ∧
  (∀ n m k : ℕ, n + m + k = 444 → 12 * (n + m + k) / 12 = 12) ∧
  (∀ n m k : ℕ, n + m + k = 777 → (7 * 3 ≠ 15 → (7 * 3 - 6 = 15)) ) →
  ∀ n m k : ℕ, n + m + k = 888 → 8 * (n + m + k / 3) - 6 = 18 :=
by
  intros h n m k h1
  sorry

end puzzle_solution_l161_161587


namespace remainder_div_38_l161_161670

theorem remainder_div_38 (n : ℕ) (h : n = 432 * 44) : n % 38 = 32 :=
sorry

end remainder_div_38_l161_161670


namespace Louisa_average_speed_l161_161779

theorem Louisa_average_speed : 
  ∀ (v : ℝ), (∀ v, (160 / v) + 3 = (280 / v)) → v = 40 :=
by
  intros v h
  sorry

end Louisa_average_speed_l161_161779


namespace combined_weight_of_two_new_students_l161_161940

theorem combined_weight_of_two_new_students (W : ℕ) (X : ℕ) 
  (cond1 : (W - 150 + X) / 8 = (W / 8) - 2) :
  X = 134 := 
sorry

end combined_weight_of_two_new_students_l161_161940


namespace students_in_zack_classroom_l161_161509

theorem students_in_zack_classroom 
(T M Z : ℕ)
(h1 : T = M)
(h2 : Z = (T + M) / 2)
(h3 : T + M + Z = 69) :
Z = 23 :=
by
  sorry

end students_in_zack_classroom_l161_161509


namespace only_real_solution_x_eq_6_l161_161007

theorem only_real_solution_x_eq_6 : ∀ x : ℝ, (x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3) → x = 6 :=
by
  sorry

end only_real_solution_x_eq_6_l161_161007


namespace complex_sum_l161_161890

open Complex

theorem complex_sum (w : ℂ) (h : w^2 - w + 1 = 0) :
  w^103 + w^104 + w^105 + w^106 + w^107 = -1 :=
sorry

end complex_sum_l161_161890


namespace probability_green_or_yellow_l161_161344

def green_faces : ℕ := 3
def yellow_faces : ℕ := 2
def blue_faces : ℕ := 1
def total_faces : ℕ := 6

theorem probability_green_or_yellow : 
  (green_faces + yellow_faces) / total_faces = 5 / 6 :=
by
  sorry

end probability_green_or_yellow_l161_161344


namespace multiple_of_8_and_12_l161_161969

theorem multiple_of_8_and_12 (x y : ℤ) (hx : ∃ k : ℤ, x = 8 * k) (hy : ∃ k : ℤ, y = 12 * k) :
  (∃ k : ℤ, y = 4 * k) ∧ (∃ k : ℤ, x - y = 4 * k) :=
by
  /- Proof goes here, based on the given conditions -/
  sorry

end multiple_of_8_and_12_l161_161969


namespace hyperbola_eccentricity_l161_161433

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > b) (hb : b > 0)
  (h_ellipse : (a^2 - b^2) / a^2 = 3 / 4) :
  (a^2 + b^2) / a^2 = 5 / 4 :=
by
  -- We start with the given conditions and need to show the result
  sorry  -- Proof omitted

end hyperbola_eccentricity_l161_161433


namespace tangent_line_eq_l161_161926

theorem tangent_line_eq (x y : ℝ) (h_curve : y = x^3 - x + 1) (h_point : (x, y) = (0, 1)) : x + y - 1 = 0 := 
sorry

end tangent_line_eq_l161_161926


namespace largest_integer_l161_161192

theorem largest_integer (a b c d : ℤ) 
  (h1 : a + b + c = 210) 
  (h2 : a + b + d = 230) 
  (h3 : a + c + d = 245) 
  (h4 : b + c + d = 260) : 
  d = 105 :=
by 
  sorry

end largest_integer_l161_161192


namespace bernardo_probability_is_correct_l161_161990

noncomputable def bernardo_larger_probability : ℚ :=
  let total_bernardo_combinations := (Nat.choose 10 3 : ℚ)
  let total_silvia_combinations := (Nat.choose 8 3 : ℚ)
  let bernardo_has_10 := (Nat.choose 8 2 : ℚ) / total_bernardo_combinations
  let bernardo_not_has_10 := ((total_silvia_combinations - 1) / total_silvia_combinations) / 2
  bernardo_has_10 * 1 + (1 - bernardo_has_10) * bernardo_not_has_10

theorem bernardo_probability_is_correct :
  bernardo_larger_probability = 19 / 28 := by
  sorry

end bernardo_probability_is_correct_l161_161990


namespace investment_initial_amount_l161_161811

theorem investment_initial_amount (P : ℝ) (h1 : ∀ (x : ℝ), 0 < x → (1 + 0.10) * x = 1.10 * x) (h2 : 1.21 * P = 363) : P = 300 :=
sorry

end investment_initial_amount_l161_161811


namespace problem_integer_solution_l161_161649

def satisfies_condition (n : ℤ) : Prop :=
  1 + ⌊(200 * n) / 201⌋ = ⌈(198 * n) / 200⌉

theorem problem_integer_solution :
  ∃! n : ℤ, 1 ≤ n ∧ n ≤ 20200 ∧ satisfies_condition n :=
sorry

end problem_integer_solution_l161_161649


namespace find_Q_l161_161307

-- We define the circles and their centers
def circle1 (x y r : ℝ) : Prop := (x + 1) ^ 2 + (y - 1) ^ 2 = r ^ 2
def circle2 (x y R : ℝ) : Prop := (x - 2) ^ 2 + (y + 2) ^ 2 = R ^ 2

-- Coordinates of point P
def P : ℝ × ℝ := (1, 2)

-- Defining the symmetry about the line y = -x
def symmetric_about (p q : ℝ × ℝ) : Prop := p.1 = -q.2 ∧ p.2 = -q.1

-- Theorem stating that if P is (1, 2), Q should be (-2, -1)
theorem find_Q {r R : ℝ} (h1 : circle1 1 2 r) (h2 : circle2 1 2 R) (hP : P = (1, 2)) :
  ∃ Q : ℝ × ℝ, symmetric_about P Q ∧ Q = (-2, -1) :=
by
  sorry

end find_Q_l161_161307


namespace fully_simplify_expression_l161_161085

theorem fully_simplify_expression :
  (3 + 4 + 5 + 6) / 2 + (3 * 6 + 9) / 3 = 18 :=
by
  sorry

end fully_simplify_expression_l161_161085


namespace remainder_squared_mod_five_l161_161609

theorem remainder_squared_mod_five (n k : ℤ) (h : n = 5 * k + 3) : ((n - 1) ^ 2) % 5 = 4 :=
by
  sorry

end remainder_squared_mod_five_l161_161609


namespace arithmetic_progression_sum_l161_161808

theorem arithmetic_progression_sum (a d : ℝ)
  (h1 : 10 * (2 * a + 19 * d) = 200)
  (h2 : 25 * (2 * a + 49 * d) = 0) :
  35 * (2 * a + 69 * d) = -466.67 :=
by
  sorry

end arithmetic_progression_sum_l161_161808


namespace correct_option_l161_161555

noncomputable def M : Set ℝ := {x | x > -2}

theorem correct_option : {0} ⊆ M := 
by 
  intros x hx
  simp at hx
  simp [M]
  show x > -2
  linarith

end correct_option_l161_161555


namespace sam_grew_3_carrots_l161_161978

-- Let Sandy's carrots and the total number of carrots be defined
def sandy_carrots : ℕ := 6
def total_carrots : ℕ := 9

-- Define the number of carrots grown by Sam
def sam_carrots : ℕ := total_carrots - sandy_carrots

-- The theorem to prove
theorem sam_grew_3_carrots : sam_carrots = 3 := by
  sorry

end sam_grew_3_carrots_l161_161978


namespace largest_common_divisor_476_330_l161_161135

theorem largest_common_divisor_476_330 :
  ∀ (S₁ S₂ : Finset ℕ), 
    S₁ = {1, 2, 4, 7, 14, 28, 17, 34, 68, 119, 238, 476} → 
    S₂ = {1, 2, 3, 5, 6, 10, 11, 15, 22, 30, 33, 55, 66, 110, 165, 330} → 
    ∃ D, D ∈ S₁ ∧ D ∈ S₂ ∧ ∀ x, x ∈ S₁ ∧ x ∈ S₂ → x ≤ D ∧ D = 2 :=
by
  intros S₁ S₂ hS₁ hS₂
  use 2
  sorry

end largest_common_divisor_476_330_l161_161135


namespace p_sufficient_not_necessary_for_q_l161_161227

-- Given conditions p and q
def p_geometric_sequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

def q_product_equality (a b c d : ℝ) : Prop :=
  a * d = b * c

-- Theorem statement: p implies q, but q does not imply p
theorem p_sufficient_not_necessary_for_q (a b c d : ℝ) :
  (p_geometric_sequence a b c d → q_product_equality a b c d) ∧
  (¬ (q_product_equality a b c d → p_geometric_sequence a b c d)) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l161_161227


namespace count_ways_to_choose_and_discard_l161_161097

theorem count_ways_to_choose_and_discard :
  let suits := 4 
  let cards_per_suit := 13
  let ways_to_choose_4_different_suits := Nat.choose 4 4
  let ways_to_choose_4_cards := cards_per_suit ^ 4
  let ways_to_discard_1_card := 4
  1 * ways_to_choose_4_cards * ways_to_discard_1_card = 114244 :=
by
  sorry

end count_ways_to_choose_and_discard_l161_161097


namespace trigonometric_identity_l161_161851

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 2 / 7 :=
by
  sorry

end trigonometric_identity_l161_161851


namespace rook_attack_expectation_correct_l161_161087

open ProbabilityTheory

noncomputable def rook_attack_expectation : ℝ := sorry

theorem rook_attack_expectation_correct :
  rook_attack_expectation = 35.33 := sorry

end rook_attack_expectation_correct_l161_161087


namespace problem1_problem2_l161_161608

def M := { x : ℝ | 0 < x ∧ x < 1 }

theorem problem1 :
  { x : ℝ | |2 * x - 1| < 1 } = M :=
by
  simp [M]
  sorry

theorem problem2 (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (a * b + 1) > (a + b) :=
by
  simp [M] at ha hb
  sorry

end problem1_problem2_l161_161608


namespace libby_quarters_left_l161_161110

theorem libby_quarters_left (initial_quarters : ℕ) (dress_cost_dollars : ℕ) (quarters_per_dollar : ℕ) 
  (h1 : initial_quarters = 160) (h2 : dress_cost_dollars = 35) (h3 : quarters_per_dollar = 4) : 
  initial_quarters - (dress_cost_dollars * quarters_per_dollar) = 20 := by
  sorry

end libby_quarters_left_l161_161110


namespace first_day_bacteria_exceeds_200_l161_161485

noncomputable def bacteria_growth (n : ℕ) : ℕ := 2 * 3^n

theorem first_day_bacteria_exceeds_200 : ∃ n : ℕ, 2 * 3^n > 200 ∧ ∀ m : ℕ, m < n → 2 * 3^m ≤ 200 :=
by
  -- sorry for skipping proof
  sorry

end first_day_bacteria_exceeds_200_l161_161485


namespace ab_is_zero_l161_161020

-- Define that a function is odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Define the given function f
def f (a b : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b - 2

-- The main theorem to prove
theorem ab_is_zero (a b : ℝ) (h_odd : is_odd (f a b)) : a * b = 0 := 
sorry

end ab_is_zero_l161_161020


namespace fraction_unchanged_when_multiplied_by_3_l161_161834

variable (x y : ℚ)

theorem fraction_unchanged_when_multiplied_by_3 (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x) / (3 * (3 * x + y)) = x / (3 * x + y) :=
by
  sorry

end fraction_unchanged_when_multiplied_by_3_l161_161834


namespace v3_at_2_is_15_l161_161863

-- Define the polynomial f(x)
def f (x : ℝ) := x^4 + 2*x^3 + x^2 - 3*x - 1

-- Define v3 using Horner's Rule at x
def v3 (x : ℝ) := ((x + 2) * x + 1) * x - 3

-- Prove that v3 at x = 2 equals 15
theorem v3_at_2_is_15 : v3 2 = 15 :=
by
  -- Skipping the proof with sorry
  sorry

end v3_at_2_is_15_l161_161863


namespace brenda_total_erasers_l161_161530

theorem brenda_total_erasers (number_of_groups : ℕ) (erasers_per_group : ℕ) (h1 : number_of_groups = 3) (h2 : erasers_per_group = 90) : number_of_groups * erasers_per_group = 270 := 
by
  sorry

end brenda_total_erasers_l161_161530


namespace min_log_value_l161_161437

theorem min_log_value (x y : ℝ) (h : 2 * x + 3 * y = 3) : ∃ (z : ℝ), z = Real.log (2^(4 * x) + 2^(3 * y)) / Real.log 2 ∧ z = 5 / 2 := 
by
  sorry

end min_log_value_l161_161437


namespace probability_of_selecting_one_marble_each_color_l161_161182

theorem probability_of_selecting_one_marble_each_color
  (total_red_marbles : ℕ) (total_blue_marbles : ℕ) (total_green_marbles : ℕ) (total_selected_marbles : ℕ) 
  (total_marble_count : ℕ) : 
  total_red_marbles = 3 → total_blue_marbles = 3 → total_green_marbles = 3 → total_selected_marbles = 3 → total_marble_count = 9 →
  (27 / 84) = 9 / 28 :=
by
  intros h_red h_blue h_green h_selected h_total
  sorry

end probability_of_selecting_one_marble_each_color_l161_161182


namespace multiplier_of_difference_l161_161838

variable (x y : ℕ)
variable (h : x + y = 49) (h1 : x > y)

theorem multiplier_of_difference (h2 : x^2 - y^2 = k * (x - y)) : k = 49 :=
by sorry

end multiplier_of_difference_l161_161838


namespace picked_balls_correct_l161_161703

-- Conditions
def initial_balls := 6
def final_balls := 24

-- The task is to find the number of picked balls
def picked_balls : Nat := final_balls - initial_balls

-- The proof goal
theorem picked_balls_correct : picked_balls = 18 :=
by
  -- We declare, but the proof is not required
  sorry

end picked_balls_correct_l161_161703


namespace internal_angles_triangle_ABC_l161_161583

theorem internal_angles_triangle_ABC (α β γ : ℕ) (h₁ : α + β + γ = 180)
  (h₂ : α + γ = 138) (h₃ : β + γ = 108) : (α = 72) ∧ (β = 42) ∧ (γ = 66) :=
by
  sorry

end internal_angles_triangle_ABC_l161_161583


namespace inequality_solution_range_l161_161773

theorem inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, 2 * x - 6 + m < 0 ∧ 4 * x - m > 0) → m < 4 :=
by
  intro h
  sorry

end inequality_solution_range_l161_161773


namespace number_of_unique_intersections_l161_161830

-- Definitions for the given lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 3
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 5 * x - 3 * y = 6

-- The problem is to show the number of unique intersection points is 2
theorem number_of_unique_intersections : ∃ p1 p2 : ℝ × ℝ,
  p1 ≠ p2 ∧
  (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
  (line1 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
  (p1 ≠ p2 → ∀ p : ℝ × ℝ, (line1 p.1 p.2 ∨ line2 p.1 p.2 ∨ line3 p.1 p.2) →
    (p = p1 ∨ p = p2)) :=
sorry

end number_of_unique_intersections_l161_161830


namespace driver_spending_increase_l161_161176

theorem driver_spending_increase (P Q : ℝ) (X : ℝ) (h1 : 1.20 * P = (1 + 20 / 100) * P) (h2 : 0.90 * Q = (1 - 10 / 100) * Q) :
  (1 + X / 100) * (P * Q) = 1.20 * P * 0.90 * Q → X = 8 := 
by
  sorry

end driver_spending_increase_l161_161176


namespace unique_solution_of_quadratics_l161_161293

theorem unique_solution_of_quadratics (y : ℚ) 
    (h1 : 9 * y^2 + 8 * y - 3 = 0) 
    (h2 : 27 * y^2 + 35 * y - 12 = 0) : 
    y = 1 / 3 :=
sorry

end unique_solution_of_quadratics_l161_161293


namespace min_value_inequality_l161_161117

open Real

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 3 * y = 4) :
  ∃ z, z = (2 / x + 3 / y) ∧ z = 25 / 4 :=
by
  sorry

end min_value_inequality_l161_161117


namespace max_min_values_l161_161821

def f (x : ℝ) : ℝ := 6 - 12 * x + x^3

theorem max_min_values :
  let a := (-1 / 3 : ℝ)
  let b := (1 : ℝ)
  max (f a) (f b) = f a ∧ f a = 269 / 27 ∧ min (f a) (f b) = f b ∧ f b = -5 :=
by
  let a := (-1 / 3 : ℝ)
  let b := (1 : ℝ)
  have ha : f a = 269 / 27 := sorry
  have hb : f b = -5 := sorry
  have max_eq : max (f a) (f b) = f a := by sorry
  have min_eq : min (f a) (f b) = f b := by sorry
  exact ⟨max_eq, ha, min_eq, hb⟩

end max_min_values_l161_161821


namespace absent_laborers_l161_161296

theorem absent_laborers (L : ℝ) (A : ℝ) (hL : L = 17.5) (h_work_done : (L - A) / 10 = L / 6) : A = 14 :=
by
  sorry

end absent_laborers_l161_161296


namespace exists_sphere_tangent_to_lines_l161_161324

variables
  (A B C D K L M N : Point)
  (AB BC CD DA : Line)
  (sphere : Sphere)

-- Given conditions
def AN_eq_AK : AN = AK := sorry
def BK_eq_BL : BK = BL := sorry
def CL_eq_CM : CL = CM := sorry
def DM_eq_DN : DM = DN := sorry
def sphere_tangent (s : Sphere) (l : Line) : Prop := sorry -- define tangency condition

-- Problem statement
theorem exists_sphere_tangent_to_lines :
  ∃ S : Sphere, 
    sphere_tangent S AB ∧
    sphere_tangent S BC ∧
    sphere_tangent S CD ∧
    sphere_tangent S DA := sorry

end exists_sphere_tangent_to_lines_l161_161324


namespace part_a_part_b_l161_161350

-- Problem (a)
theorem part_a :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f x ≠ 0) ∧ (∀ x, 2 * f (f x) = f x ∧ f x ≥ 0) ∧ Differentiable ℝ f :=
sorry

-- Problem (b)
theorem part_b :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f x ≠ 0) ∧ (∀ x, -1 ≤ 2 * f (f x) ∧ 2 * f (f x) = f x ∧ f x ≤ 1) ∧ Differentiable ℝ f :=
sorry

end part_a_part_b_l161_161350


namespace average_after_11th_inning_is_30_l161_161188

-- Define the conditions as Lean 4 definitions
def score_in_11th_inning : ℕ := 80
def increase_in_avg : ℕ := 5
def innings_before_11th : ℕ := 10

-- Define the average before 11th inning
def average_before (x : ℕ) : ℕ := x

-- Define the total runs before 11th inning
def total_runs_before (x : ℕ) : ℕ := innings_before_11th * (average_before x)

-- Define the total runs after 11th inning
def total_runs_after (x : ℕ) : ℕ := total_runs_before x + score_in_11th_inning

-- Define the new average after 11th inning
def new_average_after (x : ℕ) : ℕ := total_runs_after x / (innings_before_11th + 1)

-- Theorem statement
theorem average_after_11th_inning_is_30 : 
  ∃ (x : ℕ), new_average_after x = average_before x + increase_in_avg → new_average_after 25 = 30 :=
by
  sorry

end average_after_11th_inning_is_30_l161_161188


namespace sun_tzu_nests_count_l161_161434

theorem sun_tzu_nests_count :
  let embankments := 9
  let trees_per_embankment := 9
  let branches_per_tree := 9
  let nests_per_branch := 9
  nests_per_branch * branches_per_tree * trees_per_embankment * embankments = 6561 :=
by
  sorry

end sun_tzu_nests_count_l161_161434


namespace ratio_of_side_length_to_brush_width_l161_161909

theorem ratio_of_side_length_to_brush_width (s w : ℝ) (h : (w^2 + ((s - w)^2) / 2) = s^2 / 3) : s / w = 3 :=
by
  sorry

end ratio_of_side_length_to_brush_width_l161_161909


namespace factor_expression_l161_161507

theorem factor_expression (a b c : ℝ) :
  3*a^3*(b^2 - c^2) - 2*b^3*(c^2 - a^2) + c^3*(a^2 - b^2) =
  (a - b)*(b - c)*(c - a)*(3*a^2 - 2*b^2 - 3*a^3/c + c) :=
sorry

end factor_expression_l161_161507


namespace interest_difference_l161_161939

noncomputable def difference_between_interest (P : ℝ) (R : ℝ) (T : ℝ) (n : ℕ) : ℝ :=
  let SI := P * R * T / 100
  let CI := P * (1 + (R / (n*100)))^(n * T) - P
  CI - SI

theorem interest_difference (P : ℝ) (R : ℝ) (T : ℝ) (n : ℕ) (hP : P = 1200) (hR : R = 10) (hT : T = 1) (hn : n = 2) :
  difference_between_interest P R T n = -59.25 := by
  sorry

end interest_difference_l161_161939


namespace intersection_complement_eq_l161_161593

-- Definitions of the sets M and N
def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 > 0}

-- Complements with respect to the reals
def complement_R (A : Set ℝ) : Set ℝ := {x | x ∉ A}

-- Target goal to prove
theorem intersection_complement_eq :
  M ∩ (complement_R N) = {1, 2} :=
by
  sorry

end intersection_complement_eq_l161_161593


namespace hyperbola_equation_center_origin_asymptote_l161_161255

theorem hyperbola_equation_center_origin_asymptote
  (center_origin : ∀ x y : ℝ, x = 0 ∧ y = 0)
  (focus_parabola : ∃ x : ℝ, 4 * x^2 = 8 * x)
  (asymptote : ∀ x y : ℝ, x + y = 0):
  ∃ a b : ℝ, a^2 = 2 ∧ b^2 = 2 ∧ (x^2 / 2) - (y^2 / 2) = 1 := 
sorry

end hyperbola_equation_center_origin_asymptote_l161_161255


namespace one_cow_one_bag_in_46_days_l161_161953

-- Defining the conditions
def cows_eat_husk (n_cows n_bags n_days : ℕ) := n_cows = n_bags ∧ n_cows = n_days ∧ n_bags = n_days

-- The main theorem to be proved
theorem one_cow_one_bag_in_46_days (h : cows_eat_husk 46 46 46) : 46 = 46 := by
  sorry

end one_cow_one_bag_in_46_days_l161_161953


namespace math_problem_l161_161796

noncomputable def A (k : ℝ) : ℝ := k - 5
noncomputable def B (k : ℝ) : ℝ := k + 2
noncomputable def C (k : ℝ) : ℝ := k / 2
noncomputable def D (k : ℝ) : ℝ := 2 * k

theorem math_problem (k : ℝ) (h : A k + B k + C k + D k = 100) : 
  (A k) * (B k) * (C k) * (D k) =  (161 * 224 * 103 * 412) / 6561 :=
by
  sorry

end math_problem_l161_161796


namespace q_value_l161_161016

theorem q_value (p q : ℝ) (hpq1 : 1 < p) (hpql : p < q) (hq_condition : (1 / p) + (1 / q) = 1) (hpq2 : p * q = 8) : q = 4 + 2 * Real.sqrt 2 :=
  sorry

end q_value_l161_161016


namespace symmetry_center_of_tangent_l161_161607

noncomputable def tangentFunction (x : ℝ) : ℝ := Real.tan (2 * x - (Real.pi / 3))

theorem symmetry_center_of_tangent :
  (∃ k : ℤ, (Real.pi / 6) + (k * Real.pi / 4) = 5 * Real.pi / 12 ∧ tangentFunction ((5 * Real.pi) / 12) = 0 ) :=
sorry

end symmetry_center_of_tangent_l161_161607


namespace value_of_k_l161_161101

theorem value_of_k (x y : ℝ) (t : ℝ) (k : ℝ) : 
  (x + t * y + 8 = 0) ∧ (5 * x - t * y + 4 = 0) ∧ (3 * x - k * y + 1 = 0) → k = 5 :=
by
  sorry

end value_of_k_l161_161101


namespace point_on_imaginary_axis_point_in_fourth_quadrant_l161_161369

-- (I) For what value(s) of the real number m is the point A on the imaginary axis?
theorem point_on_imaginary_axis (m : ℝ) :
  m^2 - 8 * m + 15 = 0 ∧ m^2 + m - 12 ≠ 0 ↔ m = 5 := sorry

-- (II) For what value(s) of the real number m is the point A located in the fourth quadrant?
theorem point_in_fourth_quadrant (m : ℝ) :
  (m^2 - 8 * m + 15 > 0 ∧ m^2 + m - 12 < 0) ↔ -4 < m ∧ m < 3 := sorry

end point_on_imaginary_axis_point_in_fourth_quadrant_l161_161369


namespace smallest_possible_n_l161_161815

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def n_is_three_digit (n : ℕ) : Prop := 
  n ≥ 100 ∧ n < 1000

def prime_digits_less_than_10 (p : ℕ) : Prop :=
  p ∈ [2, 3, 5, 7]

def three_distinct_prime_factors (n a b : ℕ) : Prop :=
  a ≠ b ∧ is_prime a ∧ is_prime b ∧ is_prime (10 * a + b) ∧ n = a * b * (10 * a + b)

theorem smallest_possible_n :
  ∃ (n a b : ℕ), n_is_three_digit n ∧ prime_digits_less_than_10 a ∧ prime_digits_less_than_10 b ∧ three_distinct_prime_factors n a b ∧ n = 138 :=
by {
  sorry
}

end smallest_possible_n_l161_161815


namespace blue_tickets_per_red_ticket_l161_161543

-- Definitions based on conditions
def yellow_tickets_to_win_bible : Nat := 10
def red_tickets_per_yellow_ticket : Nat := 10
def blue_tickets_needed : Nat := 163
def additional_yellow_tickets_needed (current_yellow : Nat) : Nat := yellow_tickets_to_win_bible - current_yellow
def additional_red_tickets_needed (current_red : Nat) (needed_yellow : Nat) : Nat := needed_yellow * red_tickets_per_yellow_ticket - current_red

-- Given conditions
def current_yellow_tickets : Nat := 8
def current_red_tickets : Nat := 3
def current_blue_tickets : Nat := 7
def needed_yellow_tickets : Nat := additional_yellow_tickets_needed current_yellow_tickets
def needed_red_tickets : Nat := additional_red_tickets_needed current_red_tickets needed_yellow_tickets

-- Theorem to prove
theorem blue_tickets_per_red_ticket : blue_tickets_needed / needed_red_tickets = 10 :=
by
  sorry

end blue_tickets_per_red_ticket_l161_161543


namespace find_intended_number_l161_161989

theorem find_intended_number (x : ℕ) 
    (condition : 3 * x = (10 * 3 * x + 2) / 19 + 7) : 
    x = 5 :=
sorry

end find_intended_number_l161_161989


namespace pentagon_area_l161_161001

-- Define the lengths of the sides of the pentagon
def side1 := 18
def side2 := 25
def side3 := 30
def side4 := 28
def side5 := 25

-- Define the sides of the rectangle and triangle
def rectangle_length := side4
def rectangle_width := side2
def triangle_base := side1
def triangle_height := rectangle_width

-- Define areas of rectangle and right triangle
def area_rectangle := rectangle_length * rectangle_width
def area_triangle := (triangle_base * triangle_height) / 2

-- Define the total area of the pentagon
def total_area_pentagon := area_rectangle + area_triangle

theorem pentagon_area : total_area_pentagon = 925 := by
  sorry

end pentagon_area_l161_161001


namespace area_triangle_ABC_is_correct_l161_161150

noncomputable def radius : ℝ := 4

noncomputable def angleABDiameter : ℝ := 30

noncomputable def ratioAM_MB : ℝ := 2 / 3

theorem area_triangle_ABC_is_correct :
  ∃ (area : ℝ), area = (180 * Real.sqrt 3) / 19 :=
by sorry

end area_triangle_ABC_is_correct_l161_161150


namespace xy_over_y_plus_x_l161_161159

theorem xy_over_y_plus_x {x y z : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : 1/x + 1/y = 1/z) : z = xy/(y+x) :=
sorry

end xy_over_y_plus_x_l161_161159


namespace michael_peach_pies_l161_161100

/--
Michael ran a bakeshop and had to fill an order for some peach pies, 4 apple pies and 3 blueberry pies.
Each pie recipe called for 3 pounds of fruit each. At the market, produce was on sale for $1.00 per pound for both blueberries and apples.
The peaches each cost $2.00 per pound. Michael spent $51 at the market buying the fruit for his pie order.
Prove that Michael had to make 5 peach pies.
-/
theorem michael_peach_pies :
  let apple_pies := 4
  let blueberry_pies := 3
  let peach_pie_cost_per_pound := 2
  let apple_blueberry_cost_per_pound := 1
  let pounds_per_pie := 3
  let total_spent := 51
  (total_spent - ((apple_pies + blueberry_pies) * pounds_per_pie * apple_blueberry_cost_per_pound)) 
  / (pounds_per_pie * peach_pie_cost_per_pound) = 5 :=
by
  let apple_pies := 4
  let blueberry_pies := 3
  let peach_pie_cost_per_pound := 2
  let apple_blueberry_cost_per_pound := 1
  let pounds_per_pie := 3
  let total_spent := 51
  have H1 : (total_spent - ((apple_pies + blueberry_pies) * pounds_per_pie * apple_blueberry_cost_per_pound)) 
             / (pounds_per_pie * peach_pie_cost_per_pound) = 5 := sorry
  exact H1

end michael_peach_pies_l161_161100


namespace right_triangle_legs_sum_squares_area_l161_161489

theorem right_triangle_legs_sum_squares_area:
  ∀ (a b c : ℝ), 
  (0 < a) → (0 < b) → (0 < c) → 
  (a^2 + b^2 = c^2) → 
  (1 / 2 * a * b = 24) → 
  (a^2 + b^2 = 48) → 
  (a = 2 * Real.sqrt 6 ∧ b = 2 * Real.sqrt 6 ∧ c = 4 * Real.sqrt 3) := 
by
  sorry

end right_triangle_legs_sum_squares_area_l161_161489


namespace tangent_product_constant_l161_161965

variable (a x₁ x₂ y₁ y₂ : ℝ)

def point_on_parabola (x y : ℝ) := x^2 = 4 * y
def point_P := (a, -2)
def point_A := (x₁, y₁)
def point_B := (x₂, y₂)

theorem tangent_product_constant
  (h₁ : point_on_parabola x₁ y₁)
  (h₂ : point_on_parabola x₂ y₂)
  (h₃ : ∃ k₁ k₂ : ℝ, 
        (y₁ + 2 = k₁ * (x₁ - a) ∧ y₂ + 2 = k₂ * (x₂ - a)) 
        ∧ (k₁ * k₂ = -2)) :
  x₁ * x₂ + y₁ * y₂ = -4 :=
sorry

end tangent_product_constant_l161_161965


namespace fraction_multiplication_result_l161_161651

theorem fraction_multiplication_result :
  (5 * 7) / 8 = 4 + 3 / 8 :=
by
  sorry

end fraction_multiplication_result_l161_161651


namespace perimeter_after_adding_tiles_l161_161859

-- Initial perimeter given
def initial_perimeter : ℕ := 20

-- Number of initial tiles
def initial_tiles : ℕ := 10

-- Number of additional tiles to be added
def additional_tiles : ℕ := 2

-- New tile side must be adjacent to an existing tile
def adjacent_tile_side : Prop := true

-- Condition about the tiles being 1x1 squares
def sq_tile (n : ℕ) : Prop := n = 1

-- The perimeter should be calculated after adding the tiles
def new_perimeter_after_addition : ℕ := 19

theorem perimeter_after_adding_tiles :
  ∃ (new_perimeter : ℕ), 
    new_perimeter = 19 ∧ 
    initial_perimeter = 20 ∧ 
    initial_tiles = 10 ∧ 
    additional_tiles = 2 ∧ 
    adjacent_tile_side ∧ 
    sq_tile 1 :=
sorry

end perimeter_after_adding_tiles_l161_161859


namespace sequence_periodic_from_some_term_l161_161302

def is_bounded (s : ℕ → ℤ) (M : ℤ) : Prop :=
  ∀ n, |s n| ≤ M

def is_periodic_from (s : ℕ → ℤ) (N : ℕ) (p : ℕ) : Prop :=
  ∀ n, s (N + n) = s (N + n + p)

theorem sequence_periodic_from_some_term (s : ℕ → ℤ) (M : ℤ) (h_bounded : is_bounded s M)
    (h_recurrence : ∀ n, s (n + 5) = (5 * s (n + 4) ^ 3 + s (n + 3) - 3 * s (n + 2) + s n) / (2 * s (n + 2) + s (n + 1) ^ 2 + s (n + 1) * s n)) :
    ∃ N p, is_periodic_from s N p := by
  sorry

end sequence_periodic_from_some_term_l161_161302


namespace oranges_difference_l161_161360

-- Defining the number of sacks of ripe and unripe oranges
def sacks_ripe_oranges := 44
def sacks_unripe_oranges := 25

-- The statement to be proven
theorem oranges_difference : sacks_ripe_oranges - sacks_unripe_oranges = 19 :=
by
  -- Provide the exact calculation and result expected
  sorry

end oranges_difference_l161_161360


namespace path_traveled_by_A_l161_161708

-- Define the initial conditions
def RectangleABCD (A B C D : ℝ × ℝ) :=
  dist A B = 3 ∧ dist C D = 3 ∧ dist B C = 5 ∧ dist D A = 5

-- Define the transformations
def rotated90Clockwise (D : ℝ × ℝ) (A : ℝ × ℝ) (A' : ℝ × ℝ) : Prop :=
  -- 90-degree clockwise rotation moves point A to A'
  A' = (D.1 + D.2 - A.2, D.2 - D.1 + A.1)

def translated3AlongDC (D C A' : ℝ × ℝ) (A'' : ℝ × ℝ) : Prop :=
  -- Translation by 3 units along line DC moves point A' to A''
  A'' = (A'.1 - 3, A'.2)

-- Define the total path traveled
noncomputable def totalPathTraveled (rotatedPath translatedPath : ℝ) : ℝ :=
  rotatedPath + translatedPath

-- Prove the total path is 2.5*pi + 3
theorem path_traveled_by_A (A B C D A' A'' : ℝ × ℝ) (hRect : RectangleABCD A B C D) (hRotate : rotated90Clockwise D A A') (hTranslate : translated3AlongDC D C A' A'') :
  totalPathTraveled (2.5 * Real.pi) 3 = (2.5 * Real.pi + 3) := by
  sorry

end path_traveled_by_A_l161_161708


namespace greatest_possible_large_chips_l161_161093

theorem greatest_possible_large_chips (s l : ℕ) (even_prime : ℕ) (h1 : s + l = 100) (h2 : s = l + even_prime) (h3 : even_prime = 2) : l = 49 :=
by
  sorry

end greatest_possible_large_chips_l161_161093


namespace siblings_are_Emma_and_Olivia_l161_161590

structure Child where
  name : String
  eyeColor : String
  hairColor : String
  ageGroup : String

def Bella := Child.mk "Bella" "Green" "Red" "Older"
def Derek := Child.mk "Derek" "Gray" "Red" "Younger"
def Olivia := Child.mk "Olivia" "Green" "Brown" "Older"
def Lucas := Child.mk "Lucas" "Gray" "Brown" "Younger"
def Emma := Child.mk "Emma" "Green" "Red" "Older"
def Ryan := Child.mk "Ryan" "Gray" "Red" "Older"
def Sophia := Child.mk "Sophia" "Green" "Brown" "Younger"
def Ethan := Child.mk "Ethan" "Gray" "Brown" "Older"

def sharesCharacteristics (c1 c2 : Child) : Nat :=
  (if c1.eyeColor = c2.eyeColor then 1 else 0) +
  (if c1.hairColor = c2.hairColor then 1 else 0) +
  (if c1.ageGroup = c2.ageGroup then 1 else 0)

theorem siblings_are_Emma_and_Olivia :
  sharesCharacteristics Bella Emma ≥ 2 ∧
  sharesCharacteristics Bella Olivia ≥ 2 ∧
  (sharesCharacteristics Bella Derek < 2) ∧
  (sharesCharacteristics Bella Lucas < 2) ∧
  (sharesCharacteristics Bella Ryan < 2) ∧
  (sharesCharacteristics Bella Sophia < 2) ∧
  (sharesCharacteristics Bella Ethan < 2) :=
by
  sorry

end siblings_are_Emma_and_Olivia_l161_161590


namespace license_plate_difference_l161_161615

theorem license_plate_difference :
  let california_plates := 26^4 * 10^3
  let texas_plates := 26^3 * 10^4
  california_plates - texas_plates = 281216000 :=
by
  let california_plates := 26^4 * 10^3
  let texas_plates := 26^3 * 10^4
  have h1 : california_plates = 456976 * 1000 := by sorry
  have h2 : texas_plates = 17576 * 10000 := by sorry
  have h3 : 456976000 - 175760000 = 281216000 := by sorry
  exact h3

end license_plate_difference_l161_161615


namespace man_speed_against_current_eq_l161_161884

-- Definitions
def downstream_speed : ℝ := 22 -- Man's speed with the current in km/hr
def current_speed : ℝ := 5 -- Speed of the current in km/hr

-- Man's speed in still water
def man_speed_in_still_water : ℝ := downstream_speed - current_speed

-- Man's speed against the current
def speed_against_current : ℝ := man_speed_in_still_water - current_speed

-- Theorem: The man's speed against the current is 12 km/hr.
theorem man_speed_against_current_eq : speed_against_current = 12 := by
  sorry

end man_speed_against_current_eq_l161_161884


namespace apple_distribution_l161_161008

theorem apple_distribution : 
  (∀ (a b c d : ℕ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) → (a + b + c + d = 30) → 
  ∃ k : ℕ, k = (Nat.choose 29 3) ∧ k = 3276) :=
by
  intros a b c d h_pos h_sum
  use Nat.choose 29 3
  have h_eq : Nat.choose 29 3 = 3276 := by sorry
  exact ⟨rfl, h_eq⟩

end apple_distribution_l161_161008


namespace multiply_fractions_l161_161269

theorem multiply_fractions :
  (2 / 9) * (5 / 14) = 5 / 63 :=
by
  sorry

end multiply_fractions_l161_161269


namespace expected_value_of_monicas_winnings_l161_161827

def die_outcome (n : ℕ) : ℤ :=
  if n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 then n else if n = 1 ∨ n = 4 ∨ n = 6 ∨ n = 8 then 0 else -5

noncomputable def expected_winnings : ℚ :=
  (1/2 : ℚ) * 0 + (1/8 : ℚ) * 2 + (1/8 : ℚ) * 3 + (1/8 : ℚ) * 5 + (1/8 : ℚ) * 7 + (1/8 : ℚ) * (-5)

theorem expected_value_of_monicas_winnings : expected_winnings = 3/2 := by
  sorry

end expected_value_of_monicas_winnings_l161_161827


namespace circle_radius_equivalence_l161_161492

theorem circle_radius_equivalence (OP_radius : ℝ) (QR : ℝ) (a : ℝ) (P : ℝ × ℝ) (S : ℝ × ℝ)
  (h1 : P = (12, 5))
  (h2 : S = (a, 0))
  (h3 : QR = 5)
  (h4 : OP_radius = 13) :
  a = 8 := 
sorry

end circle_radius_equivalence_l161_161492


namespace area_percentage_of_smaller_square_l161_161947

theorem area_percentage_of_smaller_square 
  (radius : ℝ)
  (a A O B: ℝ)
  (side_length_larger_square side_length_smaller_square : ℝ) 
  (hyp1 : side_length_larger_square = 4)
  (hyp2 : radius = 2 * Real.sqrt 2)
  (hyp3 : a = 4) 
  (hyp4 : A = 2 + side_length_smaller_square / 4)
  (hyp5 : O = 2 * Real.sqrt 2)
  (hyp6 : side_length_smaller_square = 0.8) :
  (side_length_smaller_square^2 / side_length_larger_square^2) = 0.04 :=
by
  sorry

end area_percentage_of_smaller_square_l161_161947


namespace beam_equation_correctness_l161_161948

-- Define the conditions
def total_selling_price : ℕ := 6210
def freight_per_beam : ℕ := 3

-- Define the unknown quantity
variable (x : ℕ)

-- State the theorem
theorem beam_equation_correctness
  (h1 : total_selling_price = 6210)
  (h2 : freight_per_beam = 3) :
  freight_per_beam * (x - 1) = total_selling_price / x := 
sorry

end beam_equation_correctness_l161_161948


namespace weight_order_l161_161653

variables (A B C D : ℝ) -- Representing the weights of objects A, B, C, and D as real numbers.

-- Conditions given in the problem:
axiom eq1 : A + B = C + D
axiom ineq1 : D + A > B + C
axiom ineq2 : B > A + C

-- Proof stating that the weights in ascending order are C < A < B < D.
theorem weight_order (A B C D : ℝ) : C < A ∧ A < B ∧ B < D :=
by
  -- We are not providing the proof steps here.
  sorry

end weight_order_l161_161653


namespace father_twice_as_old_in_years_l161_161288

-- Conditions
def father_age : ℕ := 42
def son_age : ℕ := 14
def years : ℕ := 14

-- Proof statement
theorem father_twice_as_old_in_years : (father_age + years) = 2 * (son_age + years) :=
by
  -- Proof content is omitted as per the instruction.
  sorry

end father_twice_as_old_in_years_l161_161288


namespace sum_with_extra_five_l161_161140

theorem sum_with_extra_five 
  (a b c : ℕ)
  (h1 : a + b = 31)
  (h2 : b + c = 48)
  (h3 : c + a = 55) : 
  a + b + c + 5 = 72 :=
by
  sorry

end sum_with_extra_five_l161_161140


namespace smallest_constant_for_triangle_l161_161197

theorem smallest_constant_for_triangle 
  (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)  
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) < 2 := 
  sorry

end smallest_constant_for_triangle_l161_161197


namespace sum_of_digits_l161_161080

theorem sum_of_digits (a b : ℕ) (h1 : 4 * 100 + a * 10 + 3 + 984 = 1 * 1000 + 3 * 100 + b * 10 + 7)
  (h2 : (1 + b) - (3 + 7) % 11 = 0) : a + b = 10 := 
by
  sorry

end sum_of_digits_l161_161080


namespace unique_solution_pairs_l161_161870

theorem unique_solution_pairs :
  ∃! (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 = 4 * c) ∧ (c^2 = 4 * b) :=
sorry

end unique_solution_pairs_l161_161870


namespace simplify_expression_correct_l161_161370

def simplify_expression (y : ℝ) : ℝ :=
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + 5 * y ^ 9 + y ^ 8)

theorem simplify_expression_correct (y : ℝ) :
  simplify_expression y = 15 * y ^ 13 - y ^ 12 + 6 * y ^ 11 + 5 * y ^ 10 - 7 * y ^ 9 - 2 * y ^ 8 :=
by
  sorry

end simplify_expression_correct_l161_161370


namespace shooting_enthusiast_l161_161541

variables {P : ℝ} -- Declare P as a real number

-- Define the conditions where X follows a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) :=
  n * p * (1 - p)

-- State the theorem in Lean 4
theorem shooting_enthusiast (h : binomial_variance 3 P = 3 / 4) : 
  P = 1 / 2 :=
by
  sorry -- Proof goes here

end shooting_enthusiast_l161_161541


namespace cost_of_dozen_pens_l161_161019

-- Define the costs and conditions as given in the problem.
def cost_of_pen (x : ℝ) : ℝ := 5 * x
def cost_of_pencil (x : ℝ) : ℝ := x

-- The given conditions transformed into Lean definitions.
def condition1 (x : ℝ) : Prop := 3 * cost_of_pen x + 5 * cost_of_pencil x = 100
def condition2 (x : ℝ) : Prop := cost_of_pen x / cost_of_pencil x = 5

-- Prove that the cost of one dozen pens is Rs. 300.
theorem cost_of_dozen_pens : ∃ x : ℝ, condition1 x ∧ condition2 x ∧ 12 * cost_of_pen x = 300 := by
  sorry

end cost_of_dozen_pens_l161_161019


namespace length_of_AD_l161_161132

-- Define the segment AD and points B, C, and M as given conditions
variable (x : ℝ) -- Assuming x is the length of segments AB, BC, CD
variable (AD : ℝ)
variable (MC : ℝ)

-- Conditions given in the problem statement
def trisect (AD : ℝ) : Prop :=
  ∃ (x : ℝ), AD = 3 * x ∧ 0 < x

def one_third_way (M AD : ℝ) : Prop :=
  M = AD / 3

def distance_MC (M C : ℝ) : ℝ :=
  C - M

noncomputable def D : Prop := sorry

-- The main theorem statement
theorem length_of_AD (AD : ℝ) (M : ℝ) (MC : ℝ) : trisect AD → one_third_way M AD → MC = M / 3 → AD = 15 :=
by
  intro H1 H2 H3
  -- sorry is added to skip the actual proof
  sorry

end length_of_AD_l161_161132


namespace dutch_americans_with_window_seats_l161_161427

theorem dutch_americans_with_window_seats :
  let total_people := 90
  let dutch_fraction := 3 / 5
  let dutch_american_fraction := 1 / 2
  let window_seat_fraction := 1 / 3
  let dutch_people := total_people * dutch_fraction
  let dutch_americans := dutch_people * dutch_american_fraction
  let dutch_americans_window_seats := dutch_americans * window_seat_fraction
  dutch_americans_window_seats = 9 := by
sorry

end dutch_americans_with_window_seats_l161_161427


namespace log_sum_equality_l161_161930

noncomputable def evaluate_log_sum : ℝ :=
  3 / (Real.log 1000^4 / Real.log 8) + 4 / (Real.log 1000^4 / Real.log 10)

theorem log_sum_equality :
  evaluate_log_sum = (9 * Real.log 2 / Real.log 10 + 4) / 12 :=
by
  sorry

end log_sum_equality_l161_161930


namespace spherical_segment_equals_circle_area_l161_161186

noncomputable def spherical_segment_surface_area (R H : ℝ) : ℝ := 2 * Real.pi * R * H
noncomputable def circle_area (b : ℝ) : ℝ := Real.pi * (b * b)

theorem spherical_segment_equals_circle_area
  (R H b : ℝ) 
  (hb : b^2 = 2 * R * H) 
  : spherical_segment_surface_area R H = circle_area b :=
by
  sorry

end spherical_segment_equals_circle_area_l161_161186


namespace find_k_of_quadratic_polynomial_l161_161511

variable (k : ℝ)

theorem find_k_of_quadratic_polynomial (h1 : (k - 2) = 0) (h2 : k ≠ 0) : k = 2 :=
by
  -- proof omitted
  sorry

end find_k_of_quadratic_polynomial_l161_161511


namespace shaded_area_l161_161531

-- Definition of square side lengths
def side_lengths : List ℕ := [2, 4, 6, 8, 10]

-- Definition for the area of the largest square
def largest_square_area : ℕ := 10 * 10

-- Definition for the area of the smallest non-shaded square
def smallest_square_area : ℕ := 2 * 2

-- Total area of triangular regions
def triangular_area : ℕ := 2 * (2 * 4 + 2 * 6 + 2 * 8 + 2 * 10)

-- Question to prove
theorem shaded_area : largest_square_area - smallest_square_area - triangular_area = 40 := by
  sorry

end shaded_area_l161_161531


namespace circle_radius_l161_161145

theorem circle_radius (M N : ℝ) (h1 : M / N = 20) :
  ∃ r : ℝ, M = π * r^2 ∧ N = 2 * π * r ∧ r = 40 :=
by
  sorry

end circle_radius_l161_161145


namespace number_of_programs_correct_l161_161174

-- Conditions definition
def solo_segments := 5
def chorus_segments := 3

noncomputable def number_of_programs : ℕ :=
  let solo_permutations := Nat.factorial solo_segments
  let available_spaces := solo_segments + 1
  let chorus_placements := Nat.choose (available_spaces - 1) chorus_segments
  solo_permutations * chorus_placements

theorem number_of_programs_correct : number_of_programs = 7200 :=
  by
    -- The proof is omitted
    sorry

end number_of_programs_correct_l161_161174


namespace number_of_chain_links_l161_161292

noncomputable def length_of_chain (number_of_links : ℕ) : ℝ :=
  (number_of_links * (7 / 3)) + 1

theorem number_of_chain_links (n m : ℕ) (d : ℝ) (thickness : ℝ) (max_length min_length : ℕ) 
  (h1 : d = 2 + 1 / 3)
  (h2 : thickness = 0.5)
  (h3 : max_length = 36)
  (h4 : min_length = 22)
  (h5 : m = n + 6)
  : length_of_chain n = 22 ∧ length_of_chain m = 36 
  :=
  sorry

end number_of_chain_links_l161_161292


namespace sum_consecutive_integers_l161_161734

theorem sum_consecutive_integers (S : ℕ) (hS : S = 221) :
  ∃ (k : ℕ) (hk : k ≥ 2) (n : ℕ), 
    (S = k * n + (k * (k - 1)) / 2) → k = 2 := sorry

end sum_consecutive_integers_l161_161734


namespace eq_m_neg_one_l161_161684

theorem eq_m_neg_one (m : ℝ) (x : ℝ) (h1 : (m-1) * x^(m^2 + 1) + 2*x - 3 = 0) (h2 : m - 1 ≠ 0) (h3 : m^2 + 1 = 2) : 
  m = -1 :=
sorry

end eq_m_neg_one_l161_161684


namespace solve_for_y_l161_161832

theorem solve_for_y (x : ℝ) (y : ℝ) (h1 : x = 8) (h2 : x^(2*y) = 16) : y = 2/3 :=
by
  sorry

end solve_for_y_l161_161832


namespace ratio_of_ages_in_two_years_l161_161326

theorem ratio_of_ages_in_two_years (S M : ℕ) (h1: M = S + 28) (h2: M + 2 = (S + 2) * 2) (h3: S = 26) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_in_two_years_l161_161326


namespace simplest_quadratic_radical_l161_161966

theorem simplest_quadratic_radical :
  let a := Real.sqrt 12
  let b := Real.sqrt (2 / 3)
  let c := Real.sqrt 0.3
  let d := Real.sqrt 7
  d < a ∧ d < b ∧ d < c := 
by {
  -- the proof steps will go here, but we use sorry for now
  sorry
}

end simplest_quadratic_radical_l161_161966


namespace find_a_l161_161658

variable (a : ℝ) -- Declare a as a real number.

-- Define the given conditions.
def condition1 (a : ℝ) : Prop := a^2 - 2 * a = 0
def condition2 (a : ℝ) : Prop := a ≠ 2

-- Define the theorem stating that if conditions are true, then a must be 0.
theorem find_a (h1 : condition1 a) (h2 : condition2 a) : a = 0 :=
sorry -- Proof is not provided, it needs to be constructed.

end find_a_l161_161658


namespace total_distance_of_drive_l161_161388

theorem total_distance_of_drive :
  let christina_speed := 30
  let christina_time_minutes := 180
  let christina_time_hours := christina_time_minutes / 60
  let friend_speed := 40
  let friend_time := 3
  let distance_christina := christina_speed * christina_time_hours
  let distance_friend := friend_speed * friend_time
  let total_distance := distance_christina + distance_friend
  total_distance = 210 :=
by
  sorry

end total_distance_of_drive_l161_161388


namespace apples_given_to_Larry_l161_161958

-- Define the initial conditions
def initial_apples : ℕ := 75
def remaining_apples : ℕ := 23

-- The statement that we need to prove
theorem apples_given_to_Larry : initial_apples - remaining_apples = 52 :=
by
  -- skip the proof
  sorry

end apples_given_to_Larry_l161_161958


namespace distribute_coins_l161_161845

/-- The number of ways to distribute 25 identical coins among 4 schoolchildren -/
theorem distribute_coins :
  (Nat.choose 28 3) = 3276 :=
by
  sorry

end distribute_coins_l161_161845


namespace greatest_t_value_exists_l161_161458

theorem greatest_t_value_exists (t : ℝ) : (∃ t, (t^2 - t - 56) / (t - 8) = 3 / (t + 5)) → ∃ t, (t = -4) := 
by
  intro h
  -- Insert proof here
  sorry

end greatest_t_value_exists_l161_161458


namespace find_principal_amount_l161_161533

variable {P R T : ℝ} -- variables for principal, rate, and time
variable (H1: R = 25)
variable (H2: T = 2)
variable (H3: (P * (0.5625) - P * (0.5)) = 225)

theorem find_principal_amount
    (H1 : R = 25)
    (H2 : T = 2)
    (H3 : (P * 0.0625) = 225) : 
    P = 3600 := 
  sorry

end find_principal_amount_l161_161533


namespace smallest_positive_integer_divides_l161_161321

theorem smallest_positive_integer_divides (m : ℕ) : 
  (∀ z : ℂ, z ≠ 0 → (z^11 + z^10 + z^8 + z^7 + z^5 + z^4 + z^2 + 1) ∣ (z^m - 1)) →
  (m = 88) :=
sorry

end smallest_positive_integer_divides_l161_161321


namespace quadratic_roots_l161_161582

noncomputable def roots_quadratic : Prop :=
  ∀ (a b : ℝ), (a + b = 7) ∧ (a * b = 7) → (a^2 + b^2 = 35)

theorem quadratic_roots (a b : ℝ) (h : a + b = 7 ∧ a * b = 7) : a^2 + b^2 = 35 :=
by
  sorry

end quadratic_roots_l161_161582


namespace males_listen_l161_161786

theorem males_listen (total_listen : ℕ) (females_listen : ℕ) (known_total_listen : total_listen = 160)
  (known_females_listen : females_listen = 75) : (total_listen - females_listen) = 85 :=
by 
  sorry

end males_listen_l161_161786


namespace daily_sacks_per_section_l161_161180

theorem daily_sacks_per_section (harvests sections : ℕ) (h_harvests : harvests = 360) (h_sections : sections = 8) : harvests / sections = 45 := by
  sorry

end daily_sacks_per_section_l161_161180


namespace directrix_of_parabola_l161_161610

-- Define the given conditions
def parabola_eqn (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 5

-- The problem is to show that the directrix of this parabola has the equation y = 23/12
theorem directrix_of_parabola : 
  (∃ y : ℝ, ∀ x : ℝ, parabola_eqn x = y) →

  ∃ y : ℝ, y = 23 / 12 :=
sorry

end directrix_of_parabola_l161_161610


namespace second_concert_attendance_l161_161476

theorem second_concert_attendance (n1 : ℕ) (h1 : n1 = 65899) (h2 : n2 = n1 + 119) : n2 = 66018 :=
by
  -- proof goes here
  sorry

end second_concert_attendance_l161_161476


namespace average_hours_l161_161952

def hours_studied (week1 week2 week3 week4 week5 week6 week7 : ℕ) : ℕ :=
  week1 + week2 + week3 + week4 + week5 + week6 + week7

theorem average_hours (x : ℕ)
  (h1 : hours_studied 8 10 9 11 10 7 x / 7 = 9) :
  x = 8 :=
by
  sorry

end average_hours_l161_161952


namespace lottery_probability_correct_l161_161494

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_correct :
  let MegaBall_probability := 1 / 30
  let WinnerBalls_probability := 1 / (combination 50 6)
  MegaBall_probability * WinnerBalls_probability = 1 / 476721000 :=
by
  sorry

end lottery_probability_correct_l161_161494


namespace palindrome_clock_count_l161_161538

-- Definitions based on conditions from the problem statement.
def is_valid_hour (h : ℕ) : Prop := h < 24
def is_valid_minute (m : ℕ) : Prop := m < 60
def is_palindrome (h m : ℕ) : Prop :=
  (h < 10 ∧ m / 10 = h ∧ m % 10 = h) ∨
  (h >= 10 ∧ (h / 10) = (m % 10) ∧ (h % 10) = (m / 10 % 10))

-- Main theorem statement
theorem palindrome_clock_count : 
  (∃ n : ℕ, n = 66 ∧ ∀ (h m : ℕ), is_valid_hour h → is_valid_minute m → is_palindrome h m) := 
sorry

end palindrome_clock_count_l161_161538


namespace terms_of_sequence_are_equal_l161_161406

theorem terms_of_sequence_are_equal
    (n : ℤ)
    (h_n : n ≥ 2018)
    (a b : ℕ → ℕ)
    (h_a_distinct : ∀ i j, i ≠ j → a i ≠ a j)
    (h_b_distinct : ∀ i j, i ≠ j → b i ≠ b j)
    (h_a_bounds : ∀ i, a i ≤ 5 * n)
    (h_b_bounds : ∀ i, b i ≤ 5 * n)
    (h_arith_seq : ∀ i, (a (i + 1) * b i - a i * b (i + 1)) = (a 1 * b 0 - a 0 * b 1) * i) :
    ∀ i j, (a i * b j = a j * b i) := 
by 
  sorry

end terms_of_sequence_are_equal_l161_161406


namespace find_common_difference_l161_161882

variable {a : ℕ → ℝ} -- Define the sequence as a function from natural numbers to real numbers
variable {d : ℝ} -- Define the common difference as a real number

-- Sequence is arithmetic means there exists a common difference such that a_{n+1} = a_n + d
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Conditions from the problem
variable (h1 : a 3 = 5)
variable (h2 : a 15 = 41)
variable (h3 : is_arithmetic_sequence a d)

-- Theorem statement
theorem find_common_difference : d = 3 :=
by
  sorry

end find_common_difference_l161_161882


namespace worker_savings_multiple_l161_161167

variable (P : ℝ)

theorem worker_savings_multiple (h1 : P > 0) (h2 : 0.4 * P + 0.6 * P = P) : 
  (12 * 0.4 * P) / (0.6 * P) = 8 :=
by
  sorry

end worker_savings_multiple_l161_161167


namespace range_of_m_l161_161215

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), -2 ≤ x ∧ x ≤ 3 ∧ m * x + 6 = 0) ↔ (m ≤ -2 ∨ m ≥ 3) :=
by
  sorry

end range_of_m_l161_161215


namespace closest_multiple_of_18_l161_161291

def is_multiple_of_2 (n : ℤ) : Prop := n % 2 = 0
def is_multiple_of_9 (n : ℤ) : Prop := n % 9 = 0
def is_multiple_of_18 (n : ℤ) : Prop := is_multiple_of_2 n ∧ is_multiple_of_9 n

theorem closest_multiple_of_18 (n : ℤ) (h : n = 2509) : 
  ∃ k : ℤ, is_multiple_of_18 k ∧ (abs (2509 - k) = 7) :=
sorry

end closest_multiple_of_18_l161_161291


namespace batsman_average_l161_161430

theorem batsman_average 
  (inns : ℕ)
  (highest : ℕ)
  (diff : ℕ)
  (avg_excl : ℕ)
  (total_in_44 : ℕ)
  (total_in_46 : ℕ)
  (average_in_46 : ℕ)
  (H1 : inns = 46)
  (H2 : highest = 202)
  (H3 : diff = 150)
  (H4 : avg_excl = 58)
  (H5 : total_in_44 = avg_excl * (inns - 2))
  (H6 : total_in_46 = total_in_44 + highest + (highest - diff))
  (H7 : average_in_46 = total_in_46 / inns) :
  average_in_46 = 61 := 
sorry

end batsman_average_l161_161430


namespace find_value_of_a_l161_161228

theorem find_value_of_a (a : ℝ) (h: (1 + 3 + 2 + 5 + a) / 5 = 3) : a = 4 :=
by
  sorry

end find_value_of_a_l161_161228


namespace total_amount_invested_l161_161333

def annualIncome (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

def totalInvestment (T x y : ℝ) : Prop :=
  T - x = y

def condition (T : ℝ) : Prop :=
  let income_10_percent := annualIncome (T - 800) 0.10
  let income_8_percent := annualIncome 800 0.08
  income_10_percent - income_8_percent = 56

theorem total_amount_invested :
  ∃ (T : ℝ), condition T ∧ totalInvestment T 800 800 ∧ T = 2000 :=
by
  sorry

end total_amount_invested_l161_161333


namespace min_value_of_expression_l161_161163

theorem min_value_of_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : 4 * x + 3 * y = 1) :
  1 / (2 * x - y) + 2 / (x + 2 * y) = 9 :=
sorry

end min_value_of_expression_l161_161163


namespace alfred_gain_percent_l161_161335

theorem alfred_gain_percent (P : ℝ) (R : ℝ) (S : ℝ) (H1 : P = 4700) (H2 : R = 800) (H3 : S = 6000) : 
  (S - (P + R)) / (P + R) * 100 = 9.09 := 
by
  rw [H1, H2, H3]
  norm_num
  sorry

end alfred_gain_percent_l161_161335


namespace milan_total_minutes_l161_161309

-- Conditions
variables (x : ℝ) -- minutes on the second phone line
variables (minutes_first : ℝ := x + 20) -- minutes on the first phone line
def total_cost (x : ℝ) := 3 + 0.15 * (x + 20) + 4 + 0.10 * x

-- Statement to prove
theorem milan_total_minutes (x : ℝ) (h : total_cost x = 56) :
  x + (x + 20) = 252 :=
sorry

end milan_total_minutes_l161_161309


namespace min_value_proof_l161_161719

noncomputable def min_value (α γ : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin γ - 7)^2 + (3 * Real.sin α + 4 * Real.cos γ - 12)^2

theorem min_value_proof (α γ : ℝ) : ∃ α γ : ℝ, min_value α γ = 36 :=
by
  use (Real.arcsin 12/13), (Real.pi/2 - Real.arcsin 12/13)
  sorry

end min_value_proof_l161_161719


namespace find_positive_real_unique_solution_l161_161298

theorem find_positive_real_unique_solution (x : ℝ) (h : 0 < x ∧ (x - 6) / 16 = 6 / (x - 16)) : x = 22 :=
sorry

end find_positive_real_unique_solution_l161_161298


namespace maximum_side_length_l161_161594

theorem maximum_side_length 
    (D E F : ℝ) 
    (a b c : ℝ) 
    (h_cos : Real.cos (3 * D) + Real.cos (3 * E) + Real.cos (3 * F) = 1)
    (h_a : a = 12)
    (h_perimeter : a + b + c = 40) : 
    ∃ max_side : ℝ, max_side = 7 + Real.sqrt 23 / 2 :=
by
  sorry

end maximum_side_length_l161_161594


namespace total_earnings_per_week_correct_l161_161862

noncomputable def weekday_fee_kid : ℝ := 3
noncomputable def weekday_fee_adult : ℝ := 6
noncomputable def weekend_surcharge_ratio : ℝ := 0.5

noncomputable def num_kids_weekday : ℕ := 8
noncomputable def num_adults_weekday : ℕ := 10

noncomputable def num_kids_weekend : ℕ := 12
noncomputable def num_adults_weekend : ℕ := 15

noncomputable def weekday_earnings_kids : ℝ := (num_kids_weekday : ℝ) * weekday_fee_kid
noncomputable def weekday_earnings_adults : ℝ := (num_adults_weekday : ℝ) * weekday_fee_adult

noncomputable def weekday_earnings_total : ℝ := weekday_earnings_kids + weekday_earnings_adults

noncomputable def weekday_earning_per_week : ℝ := weekday_earnings_total * 5

noncomputable def weekend_fee_kid : ℝ := weekday_fee_kid * (1 + weekend_surcharge_ratio)
noncomputable def weekend_fee_adult : ℝ := weekday_fee_adult * (1 + weekend_surcharge_ratio)

noncomputable def weekend_earnings_kids : ℝ := (num_kids_weekend : ℝ) * weekend_fee_kid
noncomputable def weekend_earnings_adults : ℝ := (num_adults_weekend : ℝ) * weekend_fee_adult

noncomputable def weekend_earnings_total : ℝ := weekend_earnings_kids + weekend_earnings_adults

noncomputable def weekend_earning_per_week : ℝ := weekend_earnings_total * 2

noncomputable def total_weekly_earnings : ℝ := weekday_earning_per_week + weekend_earning_per_week

theorem total_earnings_per_week_correct : total_weekly_earnings = 798 := by
  sorry

end total_earnings_per_week_correct_l161_161862


namespace chord_bisect_angle_l161_161374

theorem chord_bisect_angle (AB AC : ℝ) (angle_CAB : ℝ) (h1 : AB = 2) (h2 : AC = 1) (h3 : angle_CAB = 120) : 
  ∃ x : ℝ, x = 3 := 
by
  -- Proof goes here
  sorry

end chord_bisect_angle_l161_161374


namespace find_sum_of_squares_of_roots_l161_161079

theorem find_sum_of_squares_of_roots:
  ∀ (a b c d : ℝ), (a^2 * b^2 * c^2 * d^2 - 15 * a * b * c * d + 56 = 0) → 
  a^2 + b^2 + c^2 + d^2 = 30 := by
  intros a b c d h
  sorry

end find_sum_of_squares_of_roots_l161_161079


namespace students_not_enrolled_in_either_l161_161316

-- Definitions based on conditions
def total_students : ℕ := 120
def french_students : ℕ := 65
def german_students : ℕ := 50
def both_courses_students : ℕ := 25

-- The proof statement
theorem students_not_enrolled_in_either : total_students - (french_students + german_students - both_courses_students) = 30 := by
  sorry

end students_not_enrolled_in_either_l161_161316


namespace weight_of_new_girl_l161_161822

theorem weight_of_new_girl (W N : ℝ) (h_weight_replacement: (20 * W / 20 + 40 - 40 + 40) / 20 = W / 20 + 2) :
  N = 80 :=
by
  sorry

end weight_of_new_girl_l161_161822


namespace find_a_l161_161490

-- Define the circle equation and the line equation as conditions
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 1
def line_eq (x y a : ℝ) : Prop := y = x + a
def chord_length (l : ℝ) : Prop := l = 2

-- State the main problem
theorem find_a (a : ℝ) (h1 : ∀ x y : ℝ, circle_eq x y → ∃ y', line_eq x y' a ∧ chord_length 2) :
  a = -2 :=
sorry

end find_a_l161_161490


namespace photocopy_distribution_l161_161612

-- Define the problem setting
variables {n k : ℕ}

-- Define the theorem stating the problem
theorem photocopy_distribution :
  ∀ n k : ℕ, (n > 0) → 
  (k + n).choose (n - 1) = (k + n - 1).choose (n - 1) :=
by sorry

end photocopy_distribution_l161_161612


namespace percentage_of_cobalt_is_15_l161_161853

-- Define the given percentages of lead and copper
def percent_lead : ℝ := 25
def percent_copper : ℝ := 60

-- Define the weights of lead and copper used in the mixture
def weight_lead : ℝ := 5
def weight_copper : ℝ := 12

-- Define the total weight of the mixture
def total_weight : ℝ := weight_lead + weight_copper

-- Prove that the percentage of cobalt is 15%
theorem percentage_of_cobalt_is_15 :
  (100 - (percent_lead + percent_copper) = 15) :=
by
  sorry

end percentage_of_cobalt_is_15_l161_161853


namespace trig_signs_l161_161410

-- The conditions formulated as hypotheses
theorem trig_signs (h1 : Real.pi / 2 < 2 ∧ 2 < 3 ∧ 3 < Real.pi ∧ Real.pi < 4 ∧ 4 < 3 * Real.pi / 2) : 
  Real.sin 2 * Real.cos 3 * Real.tan 4 < 0 := 
sorry

end trig_signs_l161_161410


namespace determine_b_l161_161757

theorem determine_b (b : ℤ) : (x - 5) ∣ (x^3 + 3 * x^2 + b * x + 5) → b = -41 :=
by
  sorry

end determine_b_l161_161757


namespace range_of_f_t_l161_161764

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (Real.exp x) + Real.log x - x

theorem range_of_f_t (a : ℝ) (t : ℝ) 
  (h_unique_critical : ∀ x, f a x = 0 → x = t) : 
  ∃ y : ℝ, y ≥ -2 ∧ ∀ z : ℝ, y = f a t :=
sorry

end range_of_f_t_l161_161764


namespace priya_speed_l161_161327

theorem priya_speed (Riya_speed Priya_speed : ℝ) (time_separation distance_separation : ℝ)
  (h1 : Riya_speed = 30) 
  (h2 : time_separation = 45 / 60) -- 45 minutes converted to hours
  (h3 : distance_separation = 60)
  : Priya_speed = 50 :=
sorry

end priya_speed_l161_161327


namespace change_received_is_zero_l161_161513

noncomputable def combined_money : ℝ := 10 + 8
noncomputable def cost_chicken_wings : ℝ := 6
noncomputable def cost_chicken_salad : ℝ := 4
noncomputable def cost_cheeseburgers : ℝ := 2 * 3.50
noncomputable def cost_fries : ℝ := 2
noncomputable def cost_sodas : ℝ := 2 * 1.00
noncomputable def total_cost_before_discount : ℝ := cost_chicken_wings + cost_chicken_salad + cost_cheeseburgers + cost_fries + cost_sodas
noncomputable def discount_rate : ℝ := 0.15
noncomputable def tax_rate : ℝ := 0.08
noncomputable def discounted_total : ℝ := total_cost_before_discount * (1 - discount_rate)
noncomputable def tax_amount : ℝ := discounted_total * tax_rate
noncomputable def total_cost_after_tax : ℝ := discounted_total + tax_amount

theorem change_received_is_zero : combined_money < total_cost_after_tax → 0 = combined_money - total_cost_after_tax + combined_money := by
  intros h
  sorry

end change_received_is_zero_l161_161513


namespace total_storage_l161_161463

variable (barrels largeCasks smallCasks : ℕ)
variable (cap_barrel cap_largeCask cap_smallCask : ℕ)

-- Given conditions
axiom h1 : barrels = 4
axiom h2 : largeCasks = 3
axiom h3 : smallCasks = 5
axiom h4 : cap_largeCask = 20
axiom h5 : cap_smallCask = cap_largeCask / 2
axiom h6 : cap_barrel = 2 * cap_largeCask + 3

-- Target statement
theorem total_storage : 4 * cap_barrel + 3 * cap_largeCask + 5 * cap_smallCask = 282 := 
by
  -- Proof is not required
  sorry

end total_storage_l161_161463


namespace divisible_by_27000_l161_161142

theorem divisible_by_27000 (k : ℕ) (h₁ : k = 30) : ∃ n : ℕ, k^3 = 27000 * n :=
by {
  sorry
}

end divisible_by_27000_l161_161142


namespace B_work_time_alone_l161_161012

theorem B_work_time_alone
  (A_rate : ℝ := 1 / 8)
  (together_rate : ℝ := 3 / 16) :
  ∃ (B_days : ℝ), B_days = 16 :=
by
  sorry

end B_work_time_alone_l161_161012


namespace total_items_left_in_store_l161_161604

noncomputable def items_ordered : ℕ := 4458
noncomputable def items_sold : ℕ := 1561
noncomputable def items_in_storeroom : ℕ := 575

theorem total_items_left_in_store : 
  (items_ordered - items_sold) + items_in_storeroom = 3472 := 
by 
  sorry

end total_items_left_in_store_l161_161604


namespace min_triangular_faces_l161_161695

theorem min_triangular_faces (l c e m n k : ℕ) (h1 : l > c) (h2 : l + c = e + 2) (h3 : l = c + k) (h4 : e ≥ (3 * m + 4 * n) / 2) :
  m ≥ 6 := sorry

end min_triangular_faces_l161_161695


namespace min_red_chips_l161_161820

theorem min_red_chips (w b r : ℕ) (h1 : b ≥ w / 3) (h2 : b ≤ r / 4) (h3 : w + b ≥ 75) : r ≥ 76 :=
sorry

end min_red_chips_l161_161820


namespace finish_fourth_task_l161_161283

noncomputable def time_task_starts : ℕ := 12 -- Time in hours (12:00 PM)
noncomputable def time_task_ends : ℕ := 15 -- Time in hours (3:00 PM)
noncomputable def total_tasks : ℕ := 4 -- Total number of tasks
noncomputable def tasks_time (tasks: ℕ) := (time_task_ends - time_task_starts) * 60 / (total_tasks - 1) -- Time in minutes for each task

theorem finish_fourth_task : tasks_time 1 + ((total_tasks - 1) * tasks_time 1) = 240 := -- 4:00 PM expressed as 240 minutes from 12:00 PM
by
  sorry

end finish_fourth_task_l161_161283


namespace pounds_of_apples_needed_l161_161508

-- Define the conditions
def n : ℕ := 8
def c_p : ℕ := 1
def a_p : ℝ := 2.00
def c_crust : ℝ := 2.00
def c_lemon : ℝ := 0.50
def c_butter : ℝ := 1.50

-- Define the theorem to be proven
theorem pounds_of_apples_needed : 
  (n * c_p - (c_crust + c_lemon + c_butter)) / a_p = 2 := 
by
  sorry

end pounds_of_apples_needed_l161_161508


namespace quadratic_roots_evaluation_l161_161027

theorem quadratic_roots_evaluation (x1 x2 : ℝ) (h1 : x1 + x2 = 1) (h2 : x1 * x2 = -2) :
  (1 + x1) + x2 * (1 - x1) = 4 :=
by
  sorry

end quadratic_roots_evaluation_l161_161027


namespace expand_expression_l161_161207

theorem expand_expression : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  intro x
  sorry

end expand_expression_l161_161207


namespace multiplication_in_P_l161_161614

-- Define the set P as described in the problem
def P := {x : ℕ | ∃ n : ℕ, x = n^2}

-- Prove that for all a, b in P, a * b is also in P
theorem multiplication_in_P {a b : ℕ} (ha : a ∈ P) (hb : b ∈ P) : a * b ∈ P :=
sorry

end multiplication_in_P_l161_161614


namespace max_rows_l161_161373

theorem max_rows (m : ℕ) : (∀ T : Matrix (Fin m) (Fin 8) (Fin 4), 
  ∀ i j : Fin m, ∀ k l : Fin 8, i ≠ j ∧ T i k = T j k ∧ T i l = T j l → k ≠ l) → m ≤ 28 :=
sorry

end max_rows_l161_161373


namespace ratio_divisor_to_remainder_l161_161578

theorem ratio_divisor_to_remainder (R D Q : ℕ) (hR : R = 46) (hD : D = 10 * Q) (hdvd : 5290 = D * Q + R) :
  D / R = 5 :=
by
  sorry

end ratio_divisor_to_remainder_l161_161578


namespace books_left_on_Fri_l161_161478

-- Define the conditions as constants or values
def books_at_beginning : ℕ := 98
def books_checked_out_Wed : ℕ := 43
def books_returned_Thu : ℕ := 23
def books_checked_out_Thu : ℕ := 5
def books_returned_Fri : ℕ := 7

-- The proof statement to verify the final number of books
theorem books_left_on_Fri (b : ℕ) :
  b = (books_at_beginning - books_checked_out_Wed) + books_returned_Thu - books_checked_out_Thu + books_returned_Fri := 
  sorry

end books_left_on_Fri_l161_161478


namespace natalie_height_l161_161149

variable (height_Natalie height_Harpreet height_Jiayin : ℝ)
variable (h1 : height_Natalie = height_Harpreet)
variable (h2 : height_Jiayin = 161)
variable (h3 : (height_Natalie + height_Harpreet + height_Jiayin) / 3 = 171)

theorem natalie_height : height_Natalie = 176 :=
by 
  sorry

end natalie_height_l161_161149


namespace factor_expression_l161_161915

theorem factor_expression (x : ℝ) : 45 * x^2 + 135 * x = 45 * x * (x + 3) := 
by
  sorry

end factor_expression_l161_161915


namespace sum_of_x_and_y_l161_161551

-- Define the given angles
def angle_A : ℝ := 34
def angle_B : ℝ := 74
def angle_C : ℝ := 32

-- State the theorem
theorem sum_of_x_and_y (x y : ℝ) :
  (680 - x - y) = 720 → (x + y = 40) :=
by
  intro h
  sorry

end sum_of_x_and_y_l161_161551


namespace quotient_correct_l161_161076

noncomputable def find_quotient (z : ℚ) : ℚ :=
  let dividend := (5 * z ^ 5 - 3 * z ^ 4 + 6 * z ^ 3 - 8 * z ^ 2 + 9 * z - 4)
  let divisor := (4 * z ^ 2 + 5 * z + 3)
  let quotient := ((5 / 4) * z ^ 3 - (47 / 16) * z ^ 2 + (257 / 64) * z - (1547 / 256))
  quotient

theorem quotient_correct (z : ℚ) :
  find_quotient z = ((5 / 4) * z ^ 3 - (47 / 16) * z ^ 2 + (257 / 64) * z - (1547 / 256)) :=
by
  sorry

end quotient_correct_l161_161076


namespace rides_ratio_l161_161376

theorem rides_ratio (total_money rides_spent dessert_spent money_left : ℕ) 
  (h1 : total_money = 30) 
  (h2 : dessert_spent = 5) 
  (h3 : money_left = 10) 
  (h4 : total_money - money_left = rides_spent + dessert_spent) : 
  (rides_spent : ℚ) / total_money = 1 / 2 := 
sorry

end rides_ratio_l161_161376


namespace circle_touch_externally_circle_one_inside_other_without_touching_circle_completely_outside_l161_161663

-- Definitions encapsulated in theorems with conditions and desired results
theorem circle_touch_externally {d R r : ℝ} (h1 : d = 10) (h2 : R = 8) (h3 : r = 2) : 
  d = R + r :=
by 
  rw [h1, h2, h3]
  sorry

theorem circle_one_inside_other_without_touching {d R r : ℝ} (h1 : d = 4) (h2 : R = 17) (h3 : r = 11) : 
  d < R - r :=
by 
  rw [h1, h2, h3]
  sorry

theorem circle_completely_outside {d R r : ℝ} (h1 : d = 12) (h2 : R = 5) (h3 : r = 3) : 
  d > R + r :=
by 
  rw [h1, h2, h3]
  sorry

end circle_touch_externally_circle_one_inside_other_without_touching_circle_completely_outside_l161_161663


namespace original_number_is_24_l161_161078

theorem original_number_is_24 (N : ℕ) 
  (h1 : (N + 1) % 25 = 0)
  (h2 : 1 = 1) : N = 24 := 
sorry

end original_number_is_24_l161_161078


namespace shaded_fraction_is_correct_l161_161201

-- Definitions based on the identified conditions
def initial_fraction_shaded : ℚ := 4 / 9
def geometric_series_sum (a r : ℚ) : ℚ := a / (1 - r)
def infinite_series_fraction_shaded : ℚ := 4 / 9 * (4 / 3)

-- The theorem stating the problem
theorem shaded_fraction_is_correct :
  infinite_series_fraction_shaded = 16 / 27 :=
by
  sorry -- proof to be provided

end shaded_fraction_is_correct_l161_161201


namespace ducks_remaining_after_three_nights_l161_161711

def initial_ducks : ℕ := 320
def first_night_ducks_eaten (ducks : ℕ) : ℕ := ducks * 1 / 4
def after_first_night (ducks : ℕ) : ℕ := ducks - first_night_ducks_eaten ducks
def second_night_ducks_fly_away (ducks : ℕ) : ℕ := ducks * 1 / 6
def after_second_night (ducks : ℕ) : ℕ := ducks - second_night_ducks_fly_away ducks
def third_night_ducks_stolen (ducks : ℕ) : ℕ := ducks * 30 / 100
def after_third_night (ducks : ℕ) : ℕ := ducks - third_night_ducks_stolen ducks

theorem ducks_remaining_after_three_nights : after_third_night (after_second_night (after_first_night initial_ducks)) = 140 :=
by 
  -- replace the following sorry with the actual proof steps
  sorry

end ducks_remaining_after_three_nights_l161_161711


namespace distances_from_median_l161_161118

theorem distances_from_median (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∃ (x y : ℝ), x = (b * c) / (a + b) ∧ y = (a * c) / (a + b) ∧ x + y = c :=
by
  sorry

end distances_from_median_l161_161118


namespace evaluate_cubic_difference_l161_161992

theorem evaluate_cubic_difference (x y : ℚ) (h1 : x + y = 10) (h2 : 2 * x - y = 16) :
  x^3 - y^3 = 17512 / 27 :=
by sorry

end evaluate_cubic_difference_l161_161992


namespace Jan_height_is_42_l161_161057

-- Given conditions
def Cary_height : ℕ := 72
def Bill_height : ℕ := Cary_height / 2
def Jan_height : ℕ := Bill_height + 6

-- Statement to prove
theorem Jan_height_is_42 : Jan_height = 42 := by
  sorry

end Jan_height_is_42_l161_161057


namespace max_students_l161_161063

-- Defining the problem's conditions
def cost_bus_rental : ℕ := 100
def max_capacity_students : ℕ := 25
def cost_per_student : ℕ := 10
def teacher_admission_cost : ℕ := 0
def total_budget : ℕ := 350

-- The Lean proof problem
theorem max_students (bus_cost : ℕ) (student_capacity : ℕ) (student_cost : ℕ) (teacher_cost : ℕ) (budget : ℕ) :
  bus_cost = cost_bus_rental → 
  student_capacity = max_capacity_students →
  student_cost = cost_per_student →
  teacher_cost = teacher_admission_cost →
  budget = total_budget →
  (student_capacity ≤ (budget - bus_cost) / student_cost) → 
  ∃ n : ℕ, n = student_capacity ∧ n ≤ (budget - bus_cost) / student_cost :=
by
  intros
  sorry

end max_students_l161_161063


namespace fencing_cost_l161_161892

noncomputable def pi_approx : ℝ := 3.14159

theorem fencing_cost 
  (d : ℝ) (r : ℝ)
  (h_d : d = 20) 
  (h_r : r = 1.50) :
  abs (r * pi_approx * d - 94.25) < 1 :=
by
  -- Proof omitted
  sorry

end fencing_cost_l161_161892


namespace complement_B_range_a_l161_161698

open Set

variable (A B : Set ℝ) (a : ℝ)

def mySetA : Set ℝ := {x | 2 * a - 2 < x ∧ x < a}
def mySetB : Set ℝ := {x | 3 / (x - 1) ≥ 1}

theorem complement_B_range_a (h : mySetA a ⊆ compl mySetB) : 
  compl mySetB = {x | x ≤ 1} ∪ {x | x > 4} ∧ (a ≤ 1 ∨ a ≥ 2) :=
by
  sorry

end complement_B_range_a_l161_161698


namespace min_value_pt_qu_rv_sw_l161_161029

theorem min_value_pt_qu_rv_sw (p q r s t u v w : ℝ) (h1 : p * q * r * s = 8) (h2 : t * u * v * w = 27) :
  (p * t) ^ 2 + (q * u) ^ 2 + (r * v) ^ 2 + (s * w) ^ 2 ≥ 96 :=
by
  sorry

end min_value_pt_qu_rv_sw_l161_161029


namespace yeast_population_correct_l161_161986

noncomputable def yeast_population_estimation 
    (count_per_small_square : ℕ)
    (dimension_large_square : ℝ)
    (dilution_factor : ℝ)
    (thickness : ℝ)
    (total_volume : ℝ) 
    : ℝ :=
    (count_per_small_square:ℝ) / ((dimension_large_square * dimension_large_square * thickness) / 400) * dilution_factor * total_volume

theorem yeast_population_correct:
    yeast_population_estimation 5 1 10 0.1 10 = 2 * 10^9 :=
by
    sorry

end yeast_population_correct_l161_161986


namespace find_integer_pairs_l161_161999

theorem find_integer_pairs (m n : ℤ) (h1 : m * n ≥ 0) (h2 : m^3 + n^3 + 99 * m * n = 33^3) :
  (m = -33 ∧ n = -33) ∨ ∃ k : ℕ, k ≤ 33 ∧ m = k ∧ n = 33 - k ∨ m = 33 - k ∧ n = k :=
by
  sorry

end find_integer_pairs_l161_161999


namespace find_m_n_l161_161677

theorem find_m_n (m n : ℤ) :
  (∀ x : ℤ, (x + 4) * (x - 2) = x^2 + m * x + n) → (m = 2 ∧ n = -8) :=
by
  intro h
  sorry

end find_m_n_l161_161677


namespace license_plate_count_l161_161274

-- Define the number of letters and digits
def num_letters := 26
def num_digits := 10
def num_odd_digits := 5  -- (1, 3, 5, 7, 9)
def num_even_digits := 5  -- (0, 2, 4, 6, 8)

-- Calculate the number of possible license plates
theorem license_plate_count : 
  (num_letters ^ 3) * ((num_even_digits * num_odd_digits * num_digits) * 3) = 13182000 :=
by sorry

end license_plate_count_l161_161274


namespace find_a_plus_b_l161_161922

noncomputable def f (a b x : ℝ) := a * x + b
noncomputable def g (x : ℝ) := 3 * x - 4

theorem find_a_plus_b (a b : ℝ) (h : ∀ (x : ℝ), g (f a b x) = 4 * x + 5) : a + b = 13 / 3 := 
  sorry

end find_a_plus_b_l161_161922


namespace no_positive_integer_solutions_l161_161002

def f (x : ℕ) : ℕ := x*x + x

theorem no_positive_integer_solutions :
  ∀ (a b : ℕ), a > 0 → b > 0 → 4 * (f a) ≠ (f b) :=
by
  intro a b a_pos b_pos
  sorry

end no_positive_integer_solutions_l161_161002


namespace age_difference_l161_161914

theorem age_difference
  (A B C : ℕ)
  (h1 : B = 12)
  (h2 : B = 2 * C)
  (h3 : A + B + C = 32) :
  A - B = 2 :=
sorry

end age_difference_l161_161914


namespace circle_center_and_radius_l161_161713

noncomputable def circle_eq : Prop :=
  ∀ x y : ℝ, (x^2 + y^2 + 2 * x - 2 * y - 2 = 0) ↔ (x + 1)^2 + (y - 1)^2 = 4

theorem circle_center_and_radius :
  ∃ center : ℝ × ℝ, ∃ r : ℝ, 
  center = (-1, 1) ∧ r = 2 ∧ circle_eq :=
by
  sorry

end circle_center_and_radius_l161_161713


namespace polynomial_not_33_l161_161548

theorem polynomial_not_33 (x y : ℤ) : x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end polynomial_not_33_l161_161548


namespace ball_of_yarn_costs_6_l161_161033

-- Define the conditions as variables and hypotheses
variable (num_sweaters : ℕ := 28)
variable (balls_per_sweater : ℕ := 4)
variable (price_per_sweater : ℕ := 35)
variable (gain_from_sales : ℕ := 308)

-- Define derived values
def total_revenue : ℕ := num_sweaters * price_per_sweater
def total_cost_of_yarn : ℕ := total_revenue - gain_from_sales
def total_balls_of_yarn : ℕ := num_sweaters * balls_per_sweater
def cost_per_ball_of_yarn : ℕ := total_cost_of_yarn / total_balls_of_yarn

-- The theorem to be proven
theorem ball_of_yarn_costs_6 :
  cost_per_ball_of_yarn = 6 :=
by sorry

end ball_of_yarn_costs_6_l161_161033


namespace find_principal_l161_161944

-- Defining the conditions
def A : ℝ := 5292
def r : ℝ := 0.05
def n : ℝ := 1
def t : ℝ := 2

-- The theorem statement
theorem find_principal :
  ∃ (P : ℝ), A = P * (1 + r / n) ^ (n * t) ∧ P = 4800 :=
by
  sorry

end find_principal_l161_161944


namespace bakery_combinations_l161_161715

theorem bakery_combinations (h : ∀ (a b c : ℕ), a + b + c = 8 ∧ a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ count : ℕ, count = 25 := 
sorry

end bakery_combinations_l161_161715


namespace validate_model_and_profit_range_l161_161996

noncomputable def is_exponential_model_valid (x y : ℝ) : Prop :=
  ∃ T a : ℝ, T > 0 ∧ a > 1 ∧ y = T * a^x

noncomputable def is_profitable_for_at_least_one_billion (x : ℝ) : Prop :=
  (∃ T a : ℝ, T > 0 ∧ a > 1 ∧ 1/5 * (Real.sqrt 2)^x ≥ 10 ∧ 0 < x ∧ x ≤ 12) ∨
  (-0.2 * (x - 12) * (x - 17) + 12.8 ≥ 10 ∧ x > 12)

theorem validate_model_and_profit_range :
  (is_exponential_model_valid 2 0.4) ∧
  (is_exponential_model_valid 4 0.8) ∧
  (is_exponential_model_valid 12 12.8) ∧
  is_profitable_for_at_least_one_billion 11.3 ∧
  is_profitable_for_at_least_one_billion 19 :=
by
  sorry

end validate_model_and_profit_range_l161_161996


namespace axis_of_symmetry_parabola_l161_161280

theorem axis_of_symmetry_parabola (x y : ℝ) : 
  (∃ k : ℝ, (y^2 = -8 * k) → (y^2 = -8 * x) → x = -1) :=
by
  sorry

end axis_of_symmetry_parabola_l161_161280


namespace simplify_expression_l161_161285

theorem simplify_expression (a b : ℤ) : 4 * a + 5 * b - a - 7 * b = 3 * a - 2 * b :=
by
  sorry

end simplify_expression_l161_161285


namespace train_speed_is_correct_l161_161081

-- Definitions of the problem
def length_of_train : ℕ := 360
def time_to_pass_bridge : ℕ := 25
def length_of_bridge : ℕ := 140
def conversion_factor : ℝ := 3.6

-- Distance covered by the train plus the length of the bridge
def total_distance : ℕ := length_of_train + length_of_bridge

-- Speed calculation in m/s
def speed_in_m_per_s := total_distance / time_to_pass_bridge

-- Conversion to km/h
def speed_in_km_per_h := speed_in_m_per_s * conversion_factor

-- The proof goal: the speed of the train is 72 km/h
theorem train_speed_is_correct : speed_in_km_per_h = 72 := by
  sorry

end train_speed_is_correct_l161_161081


namespace exists_k_l161_161068

-- Define P as a non-constant homogeneous polynomial with real coefficients
def homogeneous_polynomial (n : ℕ) (P : ℝ → ℝ → ℝ) :=
  ∀ (a b : ℝ), P (a * a) (b * b) = (a * a) ^ n * (b * b) ^ n

-- Define the main problem
theorem exists_k (P : ℝ → ℝ → ℝ) (hP : ∃ n : ℕ, homogeneous_polynomial n P)
  (h : ∀ t : ℝ, P (Real.sin t) (Real.cos t) = 1) :
  ∃ k : ℕ, ∀ x y : ℝ, P x y = (x^2 + y^2) ^ k :=
sorry

end exists_k_l161_161068


namespace temperature_on_April_15_and_19_l161_161789

/-
We define the daily temperatures as functions of the temperature on April 15 (T_15) with the given increment of 1.5 degrees each day. 
T_15 represents the temperature on April 15.
-/
theorem temperature_on_April_15_and_19 (T : ℕ → ℝ) (T_avg : ℝ) (inc : ℝ) 
  (h1 : inc = 1.5)
  (h2 : T_avg = 17.5)
  (h3 : ∀ n, T (15 + n) = T 15 + inc * n)
  (h4 : (T 15 + T 16 + T 17 + T 18 + T 19) / 5 = T_avg) :
  T 15 = 14.5 ∧ T 19 = 20.5 :=
by
  sorry

end temperature_on_April_15_and_19_l161_161789


namespace additional_machines_l161_161904

theorem additional_machines (r : ℝ) (M : ℝ) : 
  (5 * r * 20 = 1) ∧ (M * r * 10 = 1) → (M - 5 = 95) :=
by
  sorry

end additional_machines_l161_161904


namespace expected_rolls_to_2010_l161_161264

noncomputable def expected_rolls_for_sum (n : ℕ) : ℝ :=
  if n = 2010 then 574.5238095 else sorry

theorem expected_rolls_to_2010 : expected_rolls_for_sum 2010 = 574.5238095 := sorry

end expected_rolls_to_2010_l161_161264


namespace probability_of_odd_sum_l161_161480

def balls : List ℕ := [1, 1, 2, 3, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14]

noncomputable def num_combinations (n k : ℕ) : ℕ := sorry

noncomputable def probability_odd_sum_draw_7 : ℚ :=
  let total_combinations := num_combinations 15 7
  let favorable_combinations := 3200
  (favorable_combinations : ℚ) / (total_combinations : ℚ)

theorem probability_of_odd_sum:
  probability_odd_sum_draw_7 = 640 / 1287 := by
  sorry

end probability_of_odd_sum_l161_161480


namespace relationship_between_y1_y2_l161_161322

theorem relationship_between_y1_y2 (b y1 y2 : ℝ) 
  (h₁ : y1 = -(-1) + b) 
  (h₂ : y2 = -(2) + b) : 
  y1 > y2 := 
by 
  sorry

end relationship_between_y1_y2_l161_161322


namespace negation_of_proposition_l161_161271

open Real

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > sin x) ↔ (∃ x : ℝ, x ≤ sin x) :=
by
  sorry

end negation_of_proposition_l161_161271


namespace min_value_sin_cos_expr_l161_161975

open Real

theorem min_value_sin_cos_expr :
  (∀ x : ℝ, sin x ^ 4 + (3 / 2) * cos x ^ 4 ≥ 3 / 5) ∧ 
  (∃ x : ℝ, sin x ^ 4 + (3 / 2) * cos x ^ 4 = 3 / 5) :=
by
  sorry

end min_value_sin_cos_expr_l161_161975


namespace quadratic_equation_unique_solution_l161_161413

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  16 - 4 * a * c = 0 ∧ a + c = 5 ∧ a < c → (a, c) = (1, 4) :=
by
  sorry

end quadratic_equation_unique_solution_l161_161413


namespace abs_diff_x_plus_1_x_minus_2_l161_161351

theorem abs_diff_x_plus_1_x_minus_2 (a x : ℝ) (h1 : a < 0) (h2 : |a| * x ≤ a) : |x + 1| - |x - 2| = -3 :=
by
  sorry

end abs_diff_x_plus_1_x_minus_2_l161_161351


namespace area_of_rectangle_l161_161330

-- Define the conditions
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)
def area (length width : ℕ) : ℕ := length * width

-- Assumptions based on the problem conditions
variable (length : ℕ) (width : ℕ) (P : ℕ) (A : ℕ)
variable (h1 : width = 25)
variable (h2 : P = 110)

-- Goal: Prove the area is 750 square meters
theorem area_of_rectangle : 
  ∃ l : ℕ, perimeter l 25 = 110 → area l 25 = 750 :=
by
  sorry

end area_of_rectangle_l161_161330


namespace max_k_value_l161_161730

def A : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def B (i : ℕ) := {b : Finset ℕ // b ⊆ A ∧ b ≠ ∅ ∧ ∀ j ≠ i, ∃ k : Finset ℕ, k ⊆ A ∧ k ≠ ∅ ∧ (b ∩ k).card ≤ 2}

theorem max_k_value : ∃ k, k = 175 :=
  by
    sorry

end max_k_value_l161_161730


namespace find_number_l161_161534

theorem find_number {x : ℝ} (h : 0.5 * x - 10 = 25) : x = 70 :=
sorry

end find_number_l161_161534


namespace perfect_square_trinomial_l161_161423

theorem perfect_square_trinomial (k x y : ℝ) :
  (∃ a b : ℝ, 9 * x^2 - k * x * y + 4 * y^2 = (a * x + b * y)^2) ↔ (k = 12 ∨ k = -12) :=
by
  sorry

end perfect_square_trinomial_l161_161423


namespace sampling_methods_suitability_l161_161950

-- Define sample sizes and population sizes
def n1 := 2  -- Number of students to be selected in sample ①
def N1 := 10  -- Population size for sample ①
def n2 := 50  -- Number of students to be selected in sample ②
def N2 := 1000  -- Population size for sample ②

-- Define what it means for a sampling method to be suitable
def is_simple_random_sampling_suitable (n N : Nat) : Prop :=
  N <= 50 ∧ n < N

def is_systematic_sampling_suitable (n N : Nat) : Prop :=
  N > 50 ∧ n < N ∧ n ≥ 50 / 1000 * N  -- Ensuring suitable systematic sampling size

-- The proof statement
theorem sampling_methods_suitability :
  is_simple_random_sampling_suitable n1 N1 ∧ is_systematic_sampling_suitable n2 N2 :=
by
  -- Sorry blocks are used to skip the proofs
  sorry

end sampling_methods_suitability_l161_161950


namespace closest_to_fraction_is_2000_l161_161504

-- Define the original fractions and their approximations
def numerator : ℝ := 410
def denominator : ℝ := 0.21
def approximated_numerator : ℝ := 400
def approximated_denominator : ℝ := 0.2

-- Define the options to choose from
def options : List ℝ := [100, 500, 1900, 2000, 2500]

-- Statement to prove that the closest value to numerator / denominator is 2000
theorem closest_to_fraction_is_2000 : 
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 100) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 500) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 1900) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 2500) :=
sorry

end closest_to_fraction_is_2000_l161_161504


namespace balanced_apple_trees_l161_161185

theorem balanced_apple_trees: 
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    (x1 * y2 - x1 * y4 - x3 * y2 + x3 * y4 = 0) ∧
    (x2 * y1 - x2 * y3 - x4 * y1 + x4 * y3 = 0) ∧
    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧
    (y1 ≠ y2 ∧ y1 ≠ y3 ∧ y1 ≠ y4 ∧ y2 ≠ y3 ∧ y2 ≠ y4 ∧ y3 ≠ y4) :=
  sorry

end balanced_apple_trees_l161_161185


namespace sum_first_9_terms_l161_161549

variable (a b : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop := ∀ m n k l, m + n = k + l → a m * a n = a k * a l
def geometric_prop (a : ℕ → ℝ) : Prop := a 3 * a 7 = 2 * a 5
def arithmetic_b5_eq_a5 (a b : ℕ → ℝ) : Prop := b 5 = a 5

-- The Sum Sn of an arithmetic sequence up to the nth terms
def arithmetic_sum (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (n / 2) * (b 1 + b n)

-- Question statement: proving the required sum
theorem sum_first_9_terms (a b : ℕ → ℝ) (S : ℕ → ℝ) 
  (hg : is_geometric_sequence a) 
  (hp : geometric_prop a) 
  (hb : arithmetic_b5_eq_a5 a b) 
  (arith_sum: arithmetic_sum b S) :
  S 9 = 18 :=
  sorry

end sum_first_9_terms_l161_161549


namespace base4_addition_l161_161435

def base4_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n + 1 => (n % 10) + 4 * base4_to_base10 (n / 10)

def base10_to_base4 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n + 1 => (n % 4) + 10 * base10_to_base4 (n / 4)

theorem base4_addition :
  base10_to_base4 (base4_to_base10 234 + base4_to_base10 73) = 1203 := by
  sorry

end base4_addition_l161_161435


namespace sequence_general_formula_l161_161495

theorem sequence_general_formula :
  (∃ a : ℕ → ℕ, a 1 = 4 ∧ a 2 = 6 ∧ a 3 = 8 ∧ a 4 = 10 ∧ (∀ n : ℕ, a n = 2 * (n + 1))) :=
by
  sorry

end sequence_general_formula_l161_161495


namespace fraction_subtraction_l161_161836

theorem fraction_subtraction :
  (15 / 45) - (1 + (2 / 9)) = - (8 / 9) :=
by
  sorry

end fraction_subtraction_l161_161836


namespace necessary_but_not_sufficient_condition_l161_161899

variable (x y : ℤ)

def p : Prop := x ≠ 2 ∨ y ≠ 4
def q : Prop := x + y ≠ 6

theorem necessary_but_not_sufficient_condition :
  (p x y → q x y) ∧ (¬q x y → ¬p x y) :=
sorry

end necessary_but_not_sufficient_condition_l161_161899


namespace mean_proportional_l161_161798

theorem mean_proportional (x : ℝ) (h : (72.5:ℝ) = Real.sqrt (x * 81)): x = 64.9 := by
  sorry

end mean_proportional_l161_161798


namespace roots_cubic_reciprocal_l161_161460

theorem roots_cubic_reciprocal (a b c r s : ℝ) (h_eq : a ≠ 0) (h_r : a * r^2 + b * r + c = 0) (h_s : a * s^2 + b * s + c = 0) :
  1 / r^3 + 1 / s^3 = (-b^3 + 3 * a * b * c) / c^3 := 
by
  sorry

end roots_cubic_reciprocal_l161_161460


namespace quadratic_identity_l161_161885

theorem quadratic_identity (x : ℝ) : 
  (3*x + 1)^2 + 2*(3*x + 1)*(x - 3) + (x - 3)^2 = 16*x^2 - 16*x + 4 :=
by
  sorry

end quadratic_identity_l161_161885


namespace average_age_of_10_students_l161_161317

theorem average_age_of_10_students
  (avg_age_25_students : ℕ)
  (num_students_25 : ℕ)
  (avg_age_14_students : ℕ)
  (num_students_14 : ℕ)
  (age_25th_student : ℕ)
  (avg_age_10_students : ℕ)
  (h_avg_age_25 : avg_age_25_students = 25)
  (h_num_students_25 : num_students_25 = 25)
  (h_avg_age_14 : avg_age_14_students = 28)
  (h_num_students_14 : num_students_14 = 14)
  (h_age_25th : age_25th_student = 13)
  : avg_age_10_students = 22 :=
by
  sorry

end average_age_of_10_students_l161_161317


namespace divisible_by_55_l161_161567

theorem divisible_by_55 (n : ℤ) : 
  (55 ∣ (n^2 + 3 * n + 1)) ↔ (n % 55 = 46 ∨ n % 55 = 6) := 
by 
  sorry

end divisible_by_55_l161_161567


namespace sum_of_divisor_and_quotient_is_correct_l161_161398

theorem sum_of_divisor_and_quotient_is_correct (divisor quotient : ℕ)
  (h1 : 1000 ≤ divisor ∧ divisor < 10000) -- Divisor is a four-digit number.
  (h2 : quotient * divisor + remainder = original_number) -- Division condition (could be more specific)
  (h3 : remainder < divisor) -- Remainder condition
  (h4 : original_number = 82502) -- Given original number
  : divisor + quotient = 723 := 
sorry

end sum_of_divisor_and_quotient_is_correct_l161_161398


namespace part1_part2_l161_161099

-- Definition for f(x)
def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

-- The first proof problem: Solve the inequality f(x) > 0
theorem part1 {x : ℝ} : f x > 0 ↔ x > 1 ∨ x < -5 :=
sorry

-- The second proof problem: Finding the range of m
theorem part2 {m : ℝ} : (∀ x, f x + 3 * |x - 4| ≥ m) → m ≤ 9 :=
sorry

end part1_part2_l161_161099


namespace total_employees_l161_161010

def part_time_employees : ℕ := 2047
def full_time_employees : ℕ := 63109
def contractors : ℕ := 1500
def interns : ℕ := 333
def consultants : ℕ := 918

theorem total_employees : 
  part_time_employees + full_time_employees + contractors + interns + consultants = 66907 := 
by
  -- proof goes here
  sorry

end total_employees_l161_161010


namespace angle_T_in_pentagon_l161_161043

theorem angle_T_in_pentagon (P Q R S T : ℝ) 
  (h1 : P = R) (h2 : P = T) (h3 : Q + S = 180) 
  (h4 : P + Q + R + S + T = 540) : T = 120 :=
by
  sorry

end angle_T_in_pentagon_l161_161043


namespace evaluate_expression_l161_161295

variable {c d : ℝ}

theorem evaluate_expression (h : c ≠ d ∧ c ≠ -d) :
  (c^4 - d^4) / (2 * (c^2 - d^2)) = (c^2 + d^2) / 2 :=
by sorry

end evaluate_expression_l161_161295


namespace paint_needed_for_720_statues_l161_161412

noncomputable def paint_for_similar_statues (n : Nat) (h₁ h₂ : ℝ) (p₁ : ℝ) : ℝ :=
  let ratio := (h₂ / h₁) ^ 2
  n * (ratio * p₁)

theorem paint_needed_for_720_statues :
  paint_for_similar_statues 720 12 2 1 = 20 :=
by
  sorry

end paint_needed_for_720_statues_l161_161412


namespace value_of_a_8_l161_161128

-- Definitions of the sequence and sum of first n terms
def sum_first_terms (S : ℕ → ℕ) := ∀ n : ℕ, n > 0 → S n = n^2

-- Definition of the term a_n
def a_n (S : ℕ → ℕ) (n : ℕ) := S n - S (n - 1)

-- The theorem we want to prove: a_8 = 15
theorem value_of_a_8 (S : ℕ → ℕ) (h_sum : sum_first_terms S) : a_n S 8 = 15 :=
by
  sorry

end value_of_a_8_l161_161128


namespace avery_donation_clothes_l161_161340

theorem avery_donation_clothes :
  let shirts := 4
  let pants := 2 * shirts
  let shorts := pants / 2
  shirts + pants + shorts = 16 :=
by
  let shirts := 4
  let pants := 2 * shirts
  let shorts := pants / 2
  show shirts + pants + shorts = 16
  sorry

end avery_donation_clothes_l161_161340


namespace sum_of_squares_l161_161919

variables (x y z w : ℝ)

def condition1 := (x^2 / (2^2 - 1^2)) + (y^2 / (2^2 - 3^2)) + (z^2 / (2^2 - 5^2)) + (w^2 / (2^2 - 7^2)) = 1
def condition2 := (x^2 / (4^2 - 1^2)) + (y^2 / (4^2 - 3^2)) + (z^2 / (4^2 - 5^2)) + (w^2 / (4^2 - 7^2)) = 1
def condition3 := (x^2 / (6^2 - 1^2)) + (y^2 / (6^2 - 3^2)) + (z^2 / (6^2 - 5^2)) + (w^2 / (6^2 - 7^2)) = 1
def condition4 := (x^2 / (8^2 - 1^2)) + (y^2 / (8^2 - 3^2)) + (z^2 / (8^2 - 5^2)) + (w^2 / (8^2 - 7^2)) = 1

theorem sum_of_squares : condition1 x y z w → condition2 x y z w → 
                          condition3 x y z w → condition4 x y z w →
                          (x^2 + y^2 + z^2 + w^2 = 36) :=
by
  intros h1 h2 h3 h4
  sorry

end sum_of_squares_l161_161919


namespace min_performances_l161_161979

theorem min_performances (total_singers : ℕ) (m : ℕ) (n_pairs : ℕ := 28) (pairs_performance : ℕ := 6)
  (condition : total_singers = 108) 
  (const_pairs : ∀ (r : ℕ), (n_pairs * r = pairs_performance * m)) : m ≥ 14 :=
by
  sorry

end min_performances_l161_161979


namespace exists_solution_negation_correct_l161_161932

theorem exists_solution_negation_correct :
  (∃ x : ℝ, x^2 - x = 0) ↔ (∃ x : ℝ, True) ∧ (∀ x : ℝ, ¬ (x^2 - x = 0)) :=
by
  sorry

end exists_solution_negation_correct_l161_161932


namespace num_roots_l161_161737

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x - 2

theorem num_roots : ∃! x : ℝ, f x = 0 := 
sorry

end num_roots_l161_161737


namespace problem_statement_l161_161155

noncomputable def lhs: ℝ := 8^6 * 27^6 * 8^27 * 27^8
noncomputable def rhs: ℝ := 216^14 * 8^19

theorem problem_statement : lhs = rhs :=
by
  sorry

end problem_statement_l161_161155


namespace set_difference_equals_six_l161_161687

-- Set Operations definitions used
def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Define sets M and N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 3, 6}

-- Problem statement to prove
theorem set_difference_equals_six : set_difference N M = {6} :=
  sorry

end set_difference_equals_six_l161_161687


namespace sequence_value_l161_161367

theorem sequence_value (x : ℕ) : 
  (5 - 2 = 3) ∧ (11 - 5 = 6) ∧ (20 - 11 = 9) ∧ (x - 20 = 12) → x = 32 := 
by intros; sorry

end sequence_value_l161_161367


namespace square_triangle_same_area_l161_161956

theorem square_triangle_same_area (perimeter_square height_triangle : ℤ) (same_area : ℚ) 
  (h_perimeter_square : perimeter_square = 64) 
  (h_height_triangle : height_triangle = 64)
  (h_same_area : same_area = 256) :
  ∃ x : ℚ, x = 8 :=
by
  sorry

end square_triangle_same_area_l161_161956


namespace evaluate_expression_equals_128_l161_161313

-- Define the expression as a Lean function
def expression : ℕ := (8^6) / (4 * 8^3)

-- Theorem stating that the expression equals 128
theorem evaluate_expression_equals_128 : expression = 128 := 
sorry

end evaluate_expression_equals_128_l161_161313


namespace cone_volume_l161_161546

theorem cone_volume (diameter height : ℝ) (h_diam : diameter = 14) (h_height : height = 12) :
  (1 / 3 : ℝ) * Real.pi * ((diameter / 2) ^ 2) * height = 196 * Real.pi := by
  sorry

end cone_volume_l161_161546


namespace quadratic_single_solution_positive_n_l161_161746

variables (n : ℝ)

theorem quadratic_single_solution_positive_n :
  (∃ x : ℝ, 9 * x^2 + n * x + 36 = 0) ∧ (∀ x1 x2 : ℝ, 9 * x1^2 + n * x1 + 36 = 0 ∧ 9 * x2^2 + n * x2 + 36 = 0 → x1 = x2) →
  (n = 36) :=
sorry

end quadratic_single_solution_positive_n_l161_161746


namespace relationship_between_a_b_l161_161865

theorem relationship_between_a_b (a b x : ℝ) 
  (h₁ : x = (a + b) / 2)
  (h₂ : x^2 = (a^2 - b^2) / 2):
  a = -b ∨ a = 3 * b :=
sorry

end relationship_between_a_b_l161_161865


namespace hulk_jump_geometric_seq_l161_161521

theorem hulk_jump_geometric_seq :
  ∃ n : ℕ, (2 * 3^(n-1) > 2000) ∧ n = 8 :=
by
  sorry

end hulk_jump_geometric_seq_l161_161521


namespace ironing_pants_each_day_l161_161632

-- Given conditions:
def minutes_ironing_shirt := 5 -- minutes per day
def days_per_week := 5 -- days per week
def total_minutes_ironing_4_weeks := 160 -- minutes over 4 weeks

-- Target statement to prove:
theorem ironing_pants_each_day : 
  (total_minutes_ironing_4_weeks / 4 - minutes_ironing_shirt * days_per_week) /
  days_per_week = 3 :=
by 
sorry

end ironing_pants_each_day_l161_161632


namespace range_of_independent_variable_l161_161248

theorem range_of_independent_variable (x : ℝ) (hx : 1 - 2 * x ≥ 0) : x ≤ 0.5 :=
sorry

end range_of_independent_variable_l161_161248


namespace lily_pads_half_lake_l161_161553

theorem lily_pads_half_lake (n : ℕ) (h : n = 39) :
  (n - 1) = 38 :=
by
  sorry

end lily_pads_half_lake_l161_161553


namespace investment_of_c_l161_161661

-- Definitions of given conditions
def P_b: ℝ := 4000
def diff_Pa_Pc: ℝ := 1599.9999999999995
def Ca: ℝ := 8000
def Cb: ℝ := 10000

-- Goal to be proved
theorem investment_of_c (C_c: ℝ) : 
  (∃ P_a P_c, (P_a / Ca = P_b / Cb) ∧ (P_c / C_c = P_b / Cb) ∧ (P_a - P_c = diff_Pa_Pc)) → 
  C_c = 4000 :=
sorry

end investment_of_c_l161_161661


namespace find_period_l161_161456

-- Definitions based on conditions
def interest_rate_A : ℝ := 0.10
def interest_rate_C : ℝ := 0.115
def principal : ℝ := 4000
def total_gain : ℝ := 180

-- The question to prove
theorem find_period (n : ℝ) : 
  n = 3 :=
by 
  have interest_to_A := interest_rate_A * principal
  have interest_from_C := interest_rate_C * principal
  have annual_gain := interest_from_C - interest_to_A
  have equation := total_gain = annual_gain * n
  sorry

end find_period_l161_161456


namespace largest_integer_satisfying_condition_l161_161908

-- Definition of the conditions
def has_four_digits_in_base_10 (n : ℕ) : Prop :=
  10^3 ≤ n^2 ∧ n^2 < 10^4

-- Proof statement: N is the largest integer satisfying the condition
theorem largest_integer_satisfying_condition : ∃ (N : ℕ), 
  has_four_digits_in_base_10 N ∧ (∀ (m : ℕ), has_four_digits_in_base_10 m → m ≤ N) ∧ N = 99 := 
sorry

end largest_integer_satisfying_condition_l161_161908


namespace max_marks_equals_l161_161889

/-
  Pradeep has to obtain 45% of the total marks to pass.
  He got 250 marks and failed by 50 marks.
  Prove that the maximum marks is 667.
-/

-- Define the passing percentage
def passing_percentage : ℝ := 0.45

-- Define Pradeep's marks and the marks he failed by
def pradeep_marks : ℝ := 250
def failed_by : ℝ := 50

-- Passing marks is the sum of Pradeep's marks and the marks he failed by
def passing_marks : ℝ := pradeep_marks + failed_by

-- Prove that the maximum marks M is 667
theorem max_marks_equals : ∃ M : ℝ, passing_percentage * M = passing_marks ∧ M = 667 :=
sorry

end max_marks_equals_l161_161889


namespace ratio_of_girls_to_boys_l161_161946

theorem ratio_of_girls_to_boys (total_students girls boys : ℕ) (h_ratio : girls = 4 * (girls + boys) / 7) (h_total : total_students = 70) : 
  girls = 40 ∧ boys = 30 :=
by
  sorry

end ratio_of_girls_to_boys_l161_161946


namespace batsman_average_l161_161397

theorem batsman_average (avg_20 : ℕ) (avg_10 : ℕ) (total_matches_20 : ℕ) (total_matches_10 : ℕ) :
  avg_20 = 40 → avg_10 = 20 → total_matches_20 = 20 → total_matches_10 = 10 →
  (800 + 200) / 30 = 33.33 :=
by
  sorry

end batsman_average_l161_161397


namespace basketball_volleyball_problem_l161_161766

-- Define variables and conditions
variables (x y : ℕ) (m : ℕ)

-- Conditions
def price_conditions : Prop :=
  2 * x + 3 * y = 190 ∧ 3 * x = 5 * y

def price_solutions : Prop :=
  x = 50 ∧ y = 30

def purchase_conditions : Prop :=
  8 ≤ m ∧ m ≤ 10 ∧ 50 * m + 30 * (20 - m) ≤ 800

-- The most cost-effective plan
def cost_efficient_plan : Prop :=
  m = 8 ∧ (20 - m) = 12

-- Conjecture for the problem
theorem basketball_volleyball_problem :
  price_conditions x y ∧ purchase_conditions m →
  price_solutions x y ∧ cost_efficient_plan m :=
by {
  sorry
}

end basketball_volleyball_problem_l161_161766


namespace range_of_reciprocal_sum_l161_161181

theorem range_of_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) (h4 : a + b = 1) :
  ∃ c > 4, ∀ x, x = (1 / a + 1 / b) → c < x :=
sorry

end range_of_reciprocal_sum_l161_161181


namespace find_number_l161_161393

theorem find_number (n : ℝ) (h : n / 0.04 = 400.90000000000003) : n = 16.036 := 
by
  sorry

end find_number_l161_161393


namespace least_number_to_divisible_l161_161294

theorem least_number_to_divisible (x : ℕ) : 
  (∃ x, (1049 + x) % 25 = 0) ∧ (∀ y, y < x → (1049 + y) % 25 ≠ 0) ↔ x = 1 :=
by
  sorry

end least_number_to_divisible_l161_161294


namespace highest_and_lowest_score_average_score_l161_161676

def std_score : ℤ := 60
def scores : List ℤ := [36, 0, 12, -18, 20]

theorem highest_and_lowest_score 
  (highest_score : ℤ) (lowest_score : ℤ) : 
  highest_score = std_score + 36 ∧ lowest_score = std_score - 18 := 
sorry

theorem average_score (avg_score : ℤ) :
  avg_score = std_score + ((36 + 0 + 12 - 18 + 20) / 5) := 
sorry

end highest_and_lowest_score_average_score_l161_161676


namespace arithmetic_question_l161_161866

theorem arithmetic_question :
  ((3.25 - 1.57) * 2) = 3.36 :=
by 
  sorry

end arithmetic_question_l161_161866


namespace sqrt7_sub_m_div_n_gt_inv_mn_l161_161122

variables (m n : ℤ)
variables (h_m_nonneg : m ≥ 1) (h_n_nonneg : n ≥ 1)
variables (h_ineq : Real.sqrt 7 - (m : ℝ) / (n : ℝ) > 0)

theorem sqrt7_sub_m_div_n_gt_inv_mn : 
  Real.sqrt 7 - (m : ℝ) / (n : ℝ) > 1 / ((m : ℝ) * (n : ℝ)) :=
by
  sorry

end sqrt7_sub_m_div_n_gt_inv_mn_l161_161122


namespace error_estimate_alternating_series_l161_161585

theorem error_estimate_alternating_series :
  let S := (1:ℝ) - (1 / 2) + (1 / 3) - (1 / 4) + (-(1 / 5)) 
  let S₄ := (1:ℝ) - (1 / 2) + (1 / 3) - (1 / 4)
  ∃ ΔS : ℝ, ΔS = |-(1 / 5)| ∧ ΔS < 0.2 := by
  sorry

end error_estimate_alternating_series_l161_161585


namespace expression_equals_1390_l161_161522

theorem expression_equals_1390 :
  (25 + 15 + 8) ^ 2 - (25 ^ 2 + 15 ^ 2 + 8 ^ 2) = 1390 := 
by
  sorry

end expression_equals_1390_l161_161522


namespace Tim_bottle_quarts_l161_161673

theorem Tim_bottle_quarts (ounces_per_week : ℕ) (ounces_per_quart : ℕ) (days_per_week : ℕ) (additional_ounces_per_day : ℕ) (bottles_per_day : ℕ) : 
  ounces_per_week = 812 → ounces_per_quart = 32 → days_per_week = 7 → additional_ounces_per_day = 20 → bottles_per_day = 2 → 
  ∃ quarts_per_bottle : ℝ, quarts_per_bottle = 1.5 := 
by
  intros hw ho hd ha hb
  let total_quarts_per_week := (812 : ℝ) / 32 
  let total_quarts_per_day := total_quarts_per_week / 7 
  let additional_quarts_per_day := 20 / 32 
  let quarts_from_bottles := total_quarts_per_day - additional_quarts_per_day 
  let quarts_per_bottle := quarts_from_bottles / 2 
  use quarts_per_bottle 
  sorry

end Tim_bottle_quarts_l161_161673


namespace trigonometric_identity_l161_161528

theorem trigonometric_identity (x : ℝ) (h₁ : Real.sin x = 4 / 5) (h₂ : π / 2 ≤ x ∧ x ≤ π) :
  Real.cos x = -3 / 5 ∧ (Real.cos (-x) / (Real.sin (π / 2 - x) - Real.sin (2 * π - x)) = -3) := 
by
  sorry

end trigonometric_identity_l161_161528


namespace largest_n_employees_in_same_quarter_l161_161727

theorem largest_n_employees_in_same_quarter (n : ℕ) (h1 : 72 % 4 = 0) (h2 : 72 / 4 = 18) : 
  n = 18 :=
sorry

end largest_n_employees_in_same_quarter_l161_161727


namespace smallest_identical_digit_divisible_by_18_l161_161981

theorem smallest_identical_digit_divisible_by_18 :
  ∃ n : Nat, (∀ d : Nat, d < n → ∃ a : Nat, (n = a * (10 ^ d - 1) / 9 + 1 ∧ (∃ k : Nat, n = 18 * k))) ∧ n = 666 :=
by
  sorry

end smallest_identical_digit_divisible_by_18_l161_161981


namespace sum_of_decimals_l161_161169

theorem sum_of_decimals : 5.27 + 4.19 = 9.46 :=
by
  sorry

end sum_of_decimals_l161_161169


namespace quadratic_real_roots_iff_find_m_given_condition_l161_161924

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

noncomputable def roots (a b c : ℝ) : ℝ × ℝ :=
  let disc := quadratic_discriminant a b c
  if disc < 0 then (0, 0)
  else ((-b + disc.sqrt) / (2 * a), (-b - disc.sqrt) / (2 * a))

theorem quadratic_real_roots_iff (m : ℝ) :
  (quadratic_discriminant 1 (-2 * (m + 1)) (m ^ 2 + 5) ≥ 0) ↔ (m ≥ 2) :=
by sorry

theorem find_m_given_condition (x1 x2 m : ℝ) (h1 : x1 + x2 = 2 * (m + 1)) (h2 : x1 * x2 = m ^ 2 + 5) (h3 : (x1 - 1) * (x2 - 1) = 28) :
  m = 6 :=
by sorry

end quadratic_real_roots_iff_find_m_given_condition_l161_161924


namespace mayoral_election_votes_l161_161023

theorem mayoral_election_votes (Y Z : ℕ) 
  (h1 : 22500 = Y + Y / 2) 
  (h2 : 15000 = Z - Z / 5 * 2)
  : Z = 25000 := 
  sorry

end mayoral_election_votes_l161_161023


namespace binomial_expansion_five_l161_161336

open Finset

theorem binomial_expansion_five (a b : ℝ) : 
  (a + b)^5 = a^5 + 5 * a^4 * b + 10 * a^3 * b^2 + 10 * a^2 * b^3 + 5 * a * b^4 + b^5 := 
by sorry

end binomial_expansion_five_l161_161336


namespace toys_produced_each_day_l161_161465

-- Given conditions
def total_weekly_production := 5500
def days_worked_per_week := 4

-- Define daily production calculation
def daily_production := total_weekly_production / days_worked_per_week

-- Proof that daily production is 1375 toys
theorem toys_produced_each_day :
  daily_production = 1375 := by
  sorry

end toys_produced_each_day_l161_161465


namespace smallest_four_digit_multiple_of_13_l161_161772

theorem smallest_four_digit_multiple_of_13 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 13 = 0) ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 13 ≠ 0 :=
by
  sorry

end smallest_four_digit_multiple_of_13_l161_161772


namespace max_chord_length_line_eq_orthogonal_vectors_line_eq_l161_161041

-- Definitions
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def point_P (x y : ℝ) : Prop := x = 2 ∧ y = 1
def line_eq (slope intercept x y : ℝ) : Prop := y = slope * x + intercept

-- Problem 1: Prove the equation of line l that maximizes the length of chord AB
theorem max_chord_length_line_eq : 
  (∀ x y : ℝ, point_P x y → line_eq 1 (-1) x y)
  ∧ (∀ x y : ℝ, circle_eq x y → point_P x y) 
  → (∀ x y : ℝ, line_eq 1 (-1) x y) :=
by sorry

-- Problem 2: Prove the equation of line l given orthogonality condition of vectors
theorem orthogonal_vectors_line_eq : 
  (∀ x y : ℝ, point_P x y → line_eq (-1) 3 x y)
  ∧ (∀ x y : ℝ, circle_eq x y → point_P x y) 
  → (∀ x y : ℝ, line_eq (-1) 3 x y) :=
by sorry

end max_chord_length_line_eq_orthogonal_vectors_line_eq_l161_161041


namespace maria_correct_result_l161_161424

-- Definitions of the conditions
def maria_incorrect_divide_multiply (x : ℤ) : ℤ := x / 9 - 20
def maria_final_after_errors := 8

-- Definitions of the correct operations
def maria_correct_multiply_add (x : ℤ) : ℤ := x * 9 + 20

-- The final theorem to prove
theorem maria_correct_result (x : ℤ) (h : maria_incorrect_divide_multiply x = maria_final_after_errors) :
  maria_correct_multiply_add x = 2288 :=
sorry

end maria_correct_result_l161_161424


namespace vacation_cost_division_l161_161886

theorem vacation_cost_division (n : ℕ) (h1 : 360 = 4 * (120 - 30)) (h2 : 360 = n * 120) : n = 3 := 
sorry

end vacation_cost_division_l161_161886


namespace matches_between_withdrawn_players_l161_161500

theorem matches_between_withdrawn_players (n r : ℕ) (h : 50 = (n - 3).choose 2 + (6 - r) + r) : r = 1 :=
sorry

end matches_between_withdrawn_players_l161_161500


namespace wrongly_read_number_l161_161636

theorem wrongly_read_number (initial_avg correct_avg n wrong_correct_sum : ℝ) : 
  initial_avg = 23 ∧ correct_avg = 24 ∧ n = 10 ∧ wrong_correct_sum = 36
  → ∃ (X : ℝ), 36 - X = 10 ∧ X = 26 :=
by
  intro h
  sorry

end wrongly_read_number_l161_161636


namespace meaning_of_negative_angle_l161_161451

-- Condition: a counterclockwise rotation of 30 degrees is denoted as +30 degrees.
-- Here, we set up two simple functions to represent the meaning of positive and negative angles.

def counterclockwise (angle : ℝ) : Prop :=
  angle > 0

def clockwise (angle : ℝ) : Prop :=
  angle < 0

-- Question: What is the meaning of -45 degrees?
theorem meaning_of_negative_angle : clockwise 45 :=
by
  -- we know from the problem that a positive angle (like 30 degrees) indicates counterclockwise rotation,
  -- therefore a negative angle (like -45 degrees), by definition, implies clockwise rotation.
  sorry

end meaning_of_negative_angle_l161_161451


namespace sufficient_not_necessary_l161_161006

theorem sufficient_not_necessary (x : ℝ) : abs x < 2 → (x^2 - x - 6 < 0) ∧ (¬(x^2 - x - 6 < 0) → abs x ≥ 2) :=
by
  sorry

end sufficient_not_necessary_l161_161006


namespace part1_solution_set_k_3_part2_solution_set_k_lt_0_l161_161210

open Set

-- Definitions
def inequality (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 + (k - 2) * x - 2 < 0

-- Part 1: When k = 3
theorem part1_solution_set_k_3 : ∀ x : ℝ, inequality 3 x ↔ -1 < x ∧ x < (2 / 3) :=
by
  sorry

-- Part 2: When k < 0
theorem part2_solution_set_k_lt_0 :
  ∀ k : ℝ, k < 0 → 
    (k = -2 → ∀ x : ℝ, inequality k x ↔ x ≠ -1) ∧
    (k < -2 → ∀ x : ℝ, inequality k x ↔ x < -1 ∨ x > 2 / k) ∧
    (-2 < k → ∀ x : ℝ, inequality k x ↔ x > -1 ∨ x < 2 / k) :=
by
  sorry

end part1_solution_set_k_3_part2_solution_set_k_lt_0_l161_161210


namespace notebook_cost_l161_161380

theorem notebook_cost (s n c : ℕ) (h1 : s > 17) (h2 : n > 2 ∧ n % 2 = 0) (h3 : c > n) (h4 : s * c * n = 2013) : c = 61 :=
sorry

end notebook_cost_l161_161380


namespace no_two_champion_teams_l161_161643

theorem no_two_champion_teams
  (T : Type) 
  (M : T -> T -> Prop)
  (superior : T -> T -> Prop)
  (champion : T -> Prop)
  (h1 : ∀ A B, M A B ∨ (∃ C, M A C ∧ M C B) → superior A B)
  (h2 : ∀ A, champion A ↔ ∀ B, superior A B)
  (h3 : ∀ A B, M A B ∨ M B A)
  : ¬ ∃ A B, champion A ∧ champion B ∧ A ≠ B := 
sorry

end no_two_champion_teams_l161_161643


namespace smallest_five_digit_int_equiv_5_mod_9_l161_161459

theorem smallest_five_digit_int_equiv_5_mod_9 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 9 = 5 ∧ ∀ m : ℕ, (10000 ≤ m ∧ m < 100000 ∧ m % 9 = 5) → n ≤ m :=
by
  use 10000
  sorry

end smallest_five_digit_int_equiv_5_mod_9_l161_161459


namespace samuel_initial_speed_l161_161022

/-
Samuel is driving to San Francisco’s Comic-Con in his car and he needs to travel 600 miles to the hotel where he made a reservation. 
He drives at a certain speed for 3 hours straight, then he speeds up to 80 miles/hour for 4 hours. 
Now, he is 130 miles away from the hotel. What was his initial speed?
-/

theorem samuel_initial_speed : 
  ∃ v : ℝ, (3 * v + 320 = 470) ↔ (v = 50) :=
by
  use 50
  /- detailed proof goes here -/
  sorry

end samuel_initial_speed_l161_161022


namespace construct_quad_root_of_sums_l161_161379

theorem construct_quad_root_of_sums (a b : ℝ) : ∃ c : ℝ, c = (a^4 + b^4)^(1/4) := 
by
  sorry

end construct_quad_root_of_sums_l161_161379


namespace households_accommodated_l161_161520

theorem households_accommodated (floors_per_building : ℕ)
                                (households_per_floor : ℕ)
                                (number_of_buildings : ℕ)
                                (total_households : ℕ)
                                (h1 : floors_per_building = 16)
                                (h2 : households_per_floor = 12)
                                (h3 : number_of_buildings = 10)
                                : total_households = 1920 :=
by
  sorry

end households_accommodated_l161_161520


namespace ms_walker_drives_24_miles_each_way_l161_161724

theorem ms_walker_drives_24_miles_each_way
  (D : ℝ)
  (H1 : 1 / 60 * D + 1 / 40 * D = 1) :
  D = 24 := 
sorry

end ms_walker_drives_24_miles_each_way_l161_161724


namespace probability_range_l161_161319

theorem probability_range (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1)
  (h3 : (4 * p * (1 - p)^3) ≤ (6 * p^2 * (1 - p)^2)) : 
  2 / 5 ≤ p ∧ p ≤ 1 :=
by {
  sorry
}

end probability_range_l161_161319


namespace height_of_larger_box_l161_161903

/-- Define the dimensions of the larger box and smaller boxes, 
    and show that given the constraints, the height of the larger box must be 4 meters.-/
theorem height_of_larger_box 
  (L H : ℝ) (V_small : ℝ) (N_small : ℕ) (h : ℝ) 
  (dim_large : L = 6) (width_large : H = 5)
  (vol_small : V_small = 0.6 * 0.5 * 0.4) 
  (num_boxes : N_small = 1000) 
  (vol_large : 6 * 5 * h = N_small * V_small) : 
  h = 4 :=
by 
  sorry

end height_of_larger_box_l161_161903


namespace remaining_student_number_l161_161912

theorem remaining_student_number (s1 s2 s3 : ℕ) (h1 : s1 = 5) (h2 : s2 = 29) (h3 : s3 = 41) (N : ℕ) (hN : N = 48) :
  ∃ s4, s4 < N ∧ s4 ≠ s1 ∧ s4 ≠ s2 ∧ s4 ≠ s3 ∧ (s4 = 17) :=
by
  sorry

end remaining_student_number_l161_161912


namespace intersecting_lines_sum_c_d_l161_161793

theorem intersecting_lines_sum_c_d 
  (c d : ℚ)
  (h1 : 2 = 1 / 5 * (3 : ℚ) + c)
  (h2 : 3 = 1 / 5 * (2 : ℚ) + d) : 
  c + d = 4 :=
by sorry

end intersecting_lines_sum_c_d_l161_161793


namespace proof_problem_l161_161194

variables (p q : Prop)

-- Assuming p is true and q is false
axiom p_is_true : p
axiom q_is_false : ¬ q

-- Proving that (¬p) ∨ (¬q) is true
theorem proof_problem : (¬p) ∨ (¬q) :=
by {
  sorry
}

end proof_problem_l161_161194


namespace like_terms_sum_l161_161669

theorem like_terms_sum (m n : ℤ) (h_x : 1 = m - 2) (h_y : 2 = n + 3) : m + n = 2 :=
by
  sorry

end like_terms_sum_l161_161669


namespace breadth_of_plot_l161_161887

theorem breadth_of_plot (b l : ℝ) (h1 : l * b = 18 * b) (h2 : l - b = 10) : b = 8 :=
by
  sorry

end breadth_of_plot_l161_161887


namespace jackson_final_grade_l161_161075

def jackson_hours_playing_video_games : ℕ := 9

def ratio_study_to_play : ℚ := 1 / 3

def time_spent_studying (hours_playing : ℕ) (ratio : ℚ) : ℚ := hours_playing * ratio

def points_per_hour_studying : ℕ := 15

def jackson_grade (time_studied : ℚ) (points_per_hour : ℕ) : ℚ := time_studied * points_per_hour

theorem jackson_final_grade :
  jackson_grade
    (time_spent_studying jackson_hours_playing_video_games ratio_study_to_play)
    points_per_hour_studying = 45 :=
by
  sorry

end jackson_final_grade_l161_161075


namespace expression_equals_39_l161_161467

def expression : ℤ := (-2)^4 + (-2)^3 + (-2)^2 + (-2)^1 + 3 + 2^1 + 2^2 + 2^3 + 2^4

theorem expression_equals_39 : expression = 39 := by 
  sorry

end expression_equals_39_l161_161467


namespace trajectory_of_intersection_l161_161954

-- Define the conditions and question in Lean
structure Point where
  x : ℝ
  y : ℝ

def on_circle (C : Point) : Prop :=
  C.x^2 + C.y^2 = 1

def perp_to_x_axis (C D : Point) : Prop :=
  C.x = D.x ∧ C.y = -D.y

theorem trajectory_of_intersection (A B C D M : Point)
  (hA : A = {x := -1, y := 0})
  (hB : B = {x := 1, y := 0})
  (hC : on_circle C)
  (hD : on_circle D)
  (hCD : perp_to_x_axis C D)
  (hM : ∃ m n : ℝ, C = {x := m, y := n} ∧ M = {x := 1 / m, y := n / m}) :
  M.x^2 - M.y^2 = 1 ∧ M.y ≠ 0 :=
by
  sorry

end trajectory_of_intersection_l161_161954


namespace compute_expr1_factorize_expr2_l161_161386

-- Definition for Condition 1: None explicitly stated.

-- Theorem for Question 1
theorem compute_expr1 (y : ℝ) : (y - 1) * (y + 5) = y^2 + 4*y - 5 :=
by sorry

-- Definition for Condition 2: None explicitly stated.

-- Theorem for Question 2
theorem factorize_expr2 (x y : ℝ) : -x^2 + 4*x*y - 4*y^2 = -((x - 2*y)^2) :=
by sorry

end compute_expr1_factorize_expr2_l161_161386


namespace alpha_when_beta_neg4_l161_161103

theorem alpha_when_beta_neg4 :
  (∀ (α β : ℝ), (β ≠ 0) → α = 5 → β = 2 → α * β^2 = α * 4) →
   ∃ (α : ℝ), α = 5 → ∃ β, β = -4 → α = 5 / 4 :=
  by
    intros h
    use 5 / 4
    sorry

end alpha_when_beta_neg4_l161_161103


namespace major_axis_length_l161_161108

theorem major_axis_length (radius : ℝ) (k : ℝ) (minor_axis : ℝ) (major_axis : ℝ)
  (cyl_radius : radius = 2)
  (minor_eq_diameter : minor_axis = 2 * radius)
  (major_longer : major_axis = minor_axis * (1 + k))
  (k_value : k = 0.25) :
  major_axis = 5 :=
by
  -- Proof omitted, using sorry
  sorry

end major_axis_length_l161_161108


namespace trains_distance_l161_161208

theorem trains_distance (t x : ℝ) 
  (h1 : x = 20 * t)
  (h2 : x + 50 = 25 * t) : 
  x + (x + 50) = 450 := 
by 
  -- placeholder for the proof
  sorry

end trains_distance_l161_161208


namespace class_distances_l161_161381

theorem class_distances (x y z : ℕ) 
  (h1 : y = x + 8)
  (h2 : z = 3 * x)
  (h3 : x + y + z = 108) : 
  x = 20 ∧ y = 28 ∧ z = 60 := 
  by sorry

end class_distances_l161_161381


namespace rows_of_potatoes_l161_161639

theorem rows_of_potatoes (total_potatoes : ℕ) (seeds_per_row : ℕ) (h1 : total_potatoes = 54) (h2 : seeds_per_row = 9) : total_potatoes / seeds_per_row = 6 := 
by
  sorry

end rows_of_potatoes_l161_161639


namespace pipe_network_renovation_l161_161589

theorem pipe_network_renovation 
  (total_length : Real)
  (efficiency_increase : Real)
  (days_ahead_of_schedule : Nat)
  (days_completed : Nat)
  (total_period : Nat)
  (original_daily_renovation : Real)
  (additional_renovation : Real)
  (h1 : total_length = 3600)
  (h2 : efficiency_increase = 20 / 100)
  (h3 : days_ahead_of_schedule = 10)
  (h4 : days_completed = 20)
  (h5 : total_period = 40)
  (h6 : (3600 / original_daily_renovation) - (3600 / (1.2 * original_daily_renovation)) = 10)
  (h7 : 20 * (72 + additional_renovation) >= 3600 - 1440) :
  (1.2 * original_daily_renovation = 72) ∧ (additional_renovation >= 36) :=
by
  sorry

end pipe_network_renovation_l161_161589


namespace leah_coins_value_l161_161083

theorem leah_coins_value
  (p n : ℕ)
  (h₁ : n + p = 15)
  (h₂ : n + 2 = p) : p + 5 * n = 38 :=
by
  -- definitions used in converting conditions
  sorry

end leah_coins_value_l161_161083


namespace solve_fraction_eq_l161_161278

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 3) :
  (x = 0 ∧ (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0) ∨ 
  (x = 2 / 3 ∧ (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0) :=
sorry

end solve_fraction_eq_l161_161278


namespace intersecting_chords_l161_161042

noncomputable def length_of_other_chord (x : ℝ) : ℝ :=
  3 * x + 8 * x

theorem intersecting_chords
  (a b : ℝ) (h1 : a = 12) (h2 : b = 18) (r1 r2 : ℝ) (h3 : r1/r2 = 3/8) :
  length_of_other_chord 3 = 33 := by
  sorry

end intersecting_chords_l161_161042


namespace find_other_parallel_side_length_l161_161429

variable (a b h A : ℝ)

-- Conditions
def length_one_parallel_side := a = 18
def distance_between_sides := h = 12
def area_trapezium := A = 228
def trapezium_area_formula := A = 1 / 2 * (a + b) * h

-- Target statement to prove
theorem find_other_parallel_side_length
    (h1 : length_one_parallel_side a)
    (h2 : distance_between_sides h)
    (h3 : area_trapezium A)
    (h4 : trapezium_area_formula a b h A) :
    b = 20 :=
sorry

end find_other_parallel_side_length_l161_161429


namespace total_visitors_600_l161_161443

variable (Enjoyed Understood : Set ℕ)
variable (TotalVisitors : ℕ)
variable (E U : ℕ)

axiom no_enjoy_no_understand : ∀ v, v ∉ Enjoyed → v ∉ Understood
axiom equal_enjoy_understand : E = U
axiom enjoy_and_understand_fraction : E = 3 / 4 * TotalVisitors
axiom total_visitors_equation : TotalVisitors = E + 150

theorem total_visitors_600 : TotalVisitors = 600 := by
  sorry

end total_visitors_600_l161_161443


namespace find_length_AB_l161_161795

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line y = x - 1
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection length |AB|
noncomputable def length_AB (x1 x2 : ℝ) : ℝ := x1 + x2 + 2

-- Main theorem statement
theorem find_length_AB (x1 x2 : ℝ)
  (h₁ : parabola x1 (x1 - 1))
  (h₂ : parabola x2 (x2 - 1))
  (hx : x1 + x2 = 6) :
  length_AB x1 x2 = 8 := sorry

end find_length_AB_l161_161795


namespace range_g_l161_161748

noncomputable def g (x : ℝ) : ℝ := x / (x^2 - 2 * x + 2)

theorem range_g : Set.Icc (-(1:ℝ)/2) (1/2) = {y : ℝ | ∃ x : ℝ, g x = y} := 
by
  sorry

end range_g_l161_161748


namespace weaving_sum_first_seven_days_l161_161046

noncomputable def arithmetic_sequence (a_1 d : ℕ) (n : ℕ) : ℕ := a_1 + (n - 1) * d

theorem weaving_sum_first_seven_days
  (a_1 d : ℕ) :
  (arithmetic_sequence a_1 d 1) + (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 3) = 9 →
  (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 4) + (arithmetic_sequence a_1 d 6) = 15 →
  (arithmetic_sequence a_1 d 1) + (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 3) +
  (arithmetic_sequence a_1 d 4) + (arithmetic_sequence a_1 d 5) +
  (arithmetic_sequence a_1 d 6) + (arithmetic_sequence a_1 d 7) = 35 := by
  sorry

end weaving_sum_first_seven_days_l161_161046


namespace flute_cost_is_correct_l161_161861

-- Define the conditions
def total_spent : ℝ := 158.35
def stand_cost : ℝ := 8.89
def songbook_cost : ℝ := 7.0

-- Calculate the cost to be subtracted
def accessories_cost : ℝ := stand_cost + songbook_cost

-- Define the target cost of the flute
def flute_cost : ℝ := total_spent - accessories_cost

-- Prove that the flute cost is $142.46
theorem flute_cost_is_correct : flute_cost = 142.46 :=
by
  -- Here we would provide the proof
  sorry

end flute_cost_is_correct_l161_161861


namespace square_roots_sum_eq_zero_l161_161771

theorem square_roots_sum_eq_zero (x y : ℝ) (h1 : x^2 = 2011) (h2 : y^2 = 2011) : x + y = 0 :=
by sorry

end square_roots_sum_eq_zero_l161_161771


namespace total_elephants_l161_161475

-- Definitions and Hypotheses
def W : ℕ := 70
def G : ℕ := 3 * W

-- Proposition
theorem total_elephants : W + G = 280 := by
  sorry

end total_elephants_l161_161475


namespace participants_in_robbery_l161_161645

variables (A B V G : Prop)

theorem participants_in_robbery
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G :=
by
  sorry

end participants_in_robbery_l161_161645


namespace translation_correct_l161_161994

def parabola1 (x : ℝ) : ℝ := -2 * (x + 2)^2 + 3
def parabola2 (x : ℝ) : ℝ := -2 * (x - 1)^2 - 1

theorem translation_correct :
  ∀ x : ℝ, parabola2 (x - 3) = parabola1 x - 4 :=
by
  sorry

end translation_correct_l161_161994


namespace equation_of_parabola_max_slope_OQ_l161_161347

section parabola

variable (p : ℝ)
variable (y : ℝ) (x : ℝ)
variable (n : ℝ) (m : ℝ)

-- Condition: p > 0 and distance from focus F to directrix being 2
axiom positive_p : p > 0
axiom distance_focus_directrix : ∀ {F : ℝ}, F = 2 * p → 2 * p = 2

-- Prove these two statements
theorem equation_of_parabola : (y^2 = 4 * x) :=
  sorry

theorem max_slope_OQ : (∃ K : ℝ, K = 1 / 3) :=
  sorry

end parabola

end equation_of_parabola_max_slope_OQ_l161_161347


namespace messenger_speed_l161_161136

noncomputable def team_length : ℝ := 6

noncomputable def team_speed : ℝ := 5

noncomputable def total_time : ℝ := 0.5

theorem messenger_speed (x : ℝ) :
  (6 / (x + team_speed) + 6 / (x - team_speed) = total_time) →
  x = 25 := by
  sorry

end messenger_speed_l161_161136


namespace calc_1_calc_2_l161_161453

-- Question 1
theorem calc_1 : (5 / 17 * -4 - 5 / 17 * 15 + -5 / 17 * -2) = -5 :=
by sorry

-- Question 2
theorem calc_2 : (-1^2 + 36 / ((-3)^2) - ((-3 + 3 / 7) * (-7 / 24))) = 2 :=
by sorry

end calc_1_calc_2_l161_161453


namespace number_of_cars_repaired_l161_161519

theorem number_of_cars_repaired
  (oil_change_cost repair_cost car_wash_cost : ℕ)
  (oil_changes repairs car_washes total_earnings : ℕ)
  (h₁ : oil_change_cost = 20)
  (h₂ : repair_cost = 30)
  (h₃ : car_wash_cost = 5)
  (h₄ : oil_changes = 5)
  (h₅ : car_washes = 15)
  (h₆ : total_earnings = 475)
  (h₇ : 5 * oil_change_cost + 15 * car_wash_cost + repairs * repair_cost = total_earnings) :
  repairs = 10 :=
by sorry

end number_of_cars_repaired_l161_161519


namespace janet_initial_crayons_proof_l161_161391

-- Define the initial number of crayons Michelle has
def michelle_initial_crayons : ℕ := 2

-- Define the final number of crayons Michelle will have after receiving Janet's crayons
def michelle_final_crayons : ℕ := 4

-- Define the function that calculates Janet's initial crayons
def janet_initial_crayons (m_i m_f : ℕ) : ℕ := m_f - m_i

-- The Lean statement to prove Janet's initial number of crayons
theorem janet_initial_crayons_proof : janet_initial_crayons michelle_initial_crayons michelle_final_crayons = 2 :=
by
  -- Proof steps go here (we use sorry to skip the proof)
  sorry

end janet_initial_crayons_proof_l161_161391


namespace locust_population_doubling_time_l161_161266

theorem locust_population_doubling_time 
  (h: ℕ)
  (initial_population : ℕ := 1000)
  (time_past : ℕ := 4)
  (future_time: ℕ := 10)
  (population_limit: ℕ := 128000) :
  1000 * 2 ^ ((10 + 4) / h) > 128000 → h = 2 :=
by
  sorry

end locust_population_doubling_time_l161_161266


namespace exhaust_pipe_leak_time_l161_161586

theorem exhaust_pipe_leak_time : 
  (∃ T : Real, T > 0 ∧ 
                (1 / 10 - 1 / T) = 1 / 59.999999999999964 ∧ 
                T = 12) :=
by
  sorry

end exhaust_pipe_leak_time_l161_161586


namespace rational_root_contradiction_l161_161334

theorem rational_root_contradiction 
(a b c : ℤ) 
(h_odd_a : a % 2 ≠ 0) 
(h_odd_b : b % 2 ≠ 0)
(h_odd_c : c % 2 ≠ 0)
(rational_root_exists : ∃ (r : ℚ), a * r^2 + b * r + c = 0) :
false :=
sorry

end rational_root_contradiction_l161_161334


namespace falsity_of_proposition_implies_a_range_l161_161403

theorem falsity_of_proposition_implies_a_range (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, a * Real.sin x₀ + Real.cos x₀ ≥ 2) →
  a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) :=
by 
  sorry

end falsity_of_proposition_implies_a_range_l161_161403


namespace final_purchase_price_correct_l161_161732

-- Definitions
def initial_house_value : ℝ := 100000
def profit_percentage_Mr_Brown : ℝ := 0.10
def renovation_percentage : ℝ := 0.05
def profit_percentage_Mr_Green : ℝ := 0.07
def loss_percentage_Mr_Brown : ℝ := 0.10

-- Calculations
def purchase_price_mr_brown : ℝ := initial_house_value * (1 + profit_percentage_Mr_Brown)
def total_cost_mr_brown : ℝ := purchase_price_mr_brown * (1 + renovation_percentage)
def purchase_price_mr_green : ℝ := total_cost_mr_brown * (1 + profit_percentage_Mr_Green)
def final_purchase_price_mr_brown : ℝ := purchase_price_mr_green * (1 - loss_percentage_Mr_Brown)

-- Statement to prove
theorem final_purchase_price_correct : 
  final_purchase_price_mr_brown = 111226.50 :=
by
  sorry -- Proof is omitted

end final_purchase_price_correct_l161_161732


namespace winning_percentage_l161_161812

/-- A soccer team played 158 games and won 63.2 games. 
    Prove that the winning percentage of the team is 40%. --/
theorem winning_percentage (total_games : ℕ) (won_games : ℝ) (h1 : total_games = 158) (h2 : won_games = 63.2) :
  (won_games / total_games) * 100 = 40 :=
sorry

end winning_percentage_l161_161812


namespace initial_water_amount_l161_161245

theorem initial_water_amount (E D R F I : ℕ) 
  (hE : E = 2000) 
  (hD : D = 3500) 
  (hR : R = 350 * (30 / 10))
  (hF : F = 1550) 
  (h : I - (E + D) + R = F) : 
  I = 6000 :=
by
  sorry

end initial_water_amount_l161_161245


namespace system_of_equations_solution_exists_l161_161662

theorem system_of_equations_solution_exists :
  ∃ (x y : ℚ), (x * y^2 - 2 * y^2 + 3 * x = 18) ∧ (3 * x * y + 5 * x - 6 * y = 24) ∧ 
                ((x = 3 ∧ y = 3) ∨ (x = 75 / 13 ∧ y = -3 / 7)) :=
by
  sorry

end system_of_equations_solution_exists_l161_161662


namespace chris_money_before_birthday_l161_161382

/-- Chris's total money now is $279 -/
def money_now : ℕ := 279

/-- Money received from Chris's grandmother is $25 -/
def money_grandmother : ℕ := 25

/-- Money received from Chris's aunt and uncle is $20 -/
def money_aunt_uncle : ℕ := 20

/-- Money received from Chris's parents is $75 -/
def money_parents : ℕ := 75

/-- Total money received for his birthday -/
def money_received : ℕ := money_grandmother + money_aunt_uncle + money_parents

/-- Money Chris had before his birthday -/
def money_before_birthday : ℕ := money_now - money_received

theorem chris_money_before_birthday : money_before_birthday = 159 := by
  sorry

end chris_money_before_birthday_l161_161382


namespace other_discount_l161_161069

theorem other_discount (list_price : ℝ) (final_price : ℝ) (first_discount : ℝ) (other_discount : ℝ) :
  list_price = 70 → final_price = 61.74 → first_discount = 10 → (list_price * (1 - first_discount / 100) * (1 - other_discount / 100) = final_price) → other_discount = 2 := 
by
  intros h1 h2 h3 h4
  sorry

end other_discount_l161_161069


namespace ratio_of_areas_l161_161198

theorem ratio_of_areas
  (R_X R_Y : ℝ)
  (h : (60 / 360) * 2 * Real.pi * R_X = (40 / 360) * 2 * Real.pi * R_Y) :
  (Real.pi * R_X^2) / (Real.pi * R_Y^2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_l161_161198


namespace unique_solution_triple_l161_161657

theorem unique_solution_triple (x y z : ℝ) (h1 : x + y = 3) (h2 : x * y = z^3) : (x = 1.5 ∧ y = 1.5 ∧ z = 0) :=
by
  sorry

end unique_solution_triple_l161_161657


namespace tina_sales_ratio_l161_161968

theorem tina_sales_ratio (katya_sales ricky_sales t_sold_more : ℕ) 
  (h_katya : katya_sales = 8) 
  (h_ricky : ricky_sales = 9) 
  (h_tina_sold : t_sold_more = katya_sales + 26) 
  (h_tina_multiple : ∃ m : ℕ, t_sold_more = m * (katya_sales + ricky_sales)) :
  t_sold_more / (katya_sales + ricky_sales) = 2 := 
by 
  sorry

end tina_sales_ratio_l161_161968


namespace gcd_765432_654321_l161_161173

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 :=
by 
  sorry

end gcd_765432_654321_l161_161173


namespace temperature_at_midnight_is_minus4_l161_161697

-- Definitions of initial temperature and changes
def initial_temperature : ℤ := -2
def temperature_rise_noon : ℤ := 6
def temperature_drop_midnight : ℤ := 8

-- Temperature at midnight
def temperature_midnight : ℤ :=
  initial_temperature + temperature_rise_noon - temperature_drop_midnight

theorem temperature_at_midnight_is_minus4 :
  temperature_midnight = -4 := by
  sorry

end temperature_at_midnight_is_minus4_l161_161697


namespace part1_part2_l161_161396

open Real

def f (x : ℝ) (a : ℝ) : ℝ := |x - 2| + |3 * x + a|

theorem part1 (a : ℝ) (h : a = 1) :
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} := by
  sorry

theorem part2 (h : ∃ x_0 : ℝ, f x_0 (a := a) + 2 * |x_0 - 2| < 3) : -9 < a ∧ a < -3 := by
  sorry

end part1_part2_l161_161396


namespace lizard_eyes_l161_161303

theorem lizard_eyes (E W S : Nat) 
  (h1 : W = 3 * E) 
  (h2 : S = 7 * W) 
  (h3 : E = S + W - 69) : 
  E = 3 := 
by
  sorry

end lizard_eyes_l161_161303


namespace count_yellow_highlighters_l161_161962

-- Definitions of the conditions
def pink_highlighters : ℕ := 9
def blue_highlighters : ℕ := 5
def total_highlighters : ℕ := 22

-- Definition based on the question
def yellow_highlighters : ℕ := total_highlighters - (pink_highlighters + blue_highlighters)

-- The theorem to prove the number of yellow highlighters
theorem count_yellow_highlighters : yellow_highlighters = 8 :=
by
  -- Proof omitted as instructed
  sorry

end count_yellow_highlighters_l161_161962


namespace strawberries_weight_before_l161_161106

variables (M D E B : ℝ)

noncomputable def total_weight_before (M D E : ℝ) := M + D - E

theorem strawberries_weight_before :
  ∀ (M D E : ℝ), M = 36 ∧ D = 16 ∧ E = 30 → total_weight_before M D E = 22 :=
by
  intros M D E h
  simp [total_weight_before, h]
  sorry

end strawberries_weight_before_l161_161106


namespace principal_amount_l161_161462

variable (SI R T P : ℝ)

-- Given conditions
axiom SI_def : SI = 2500
axiom R_def : R = 10
axiom T_def : T = 5

-- Main theorem statement
theorem principal_amount : SI = (P * R * T) / 100 → P = 5000 :=
by
  sorry

end principal_amount_l161_161462


namespace elements_author_is_euclid_l161_161385

def author_of_elements := "Euclid"

theorem elements_author_is_euclid : author_of_elements = "Euclid" :=
by
  rfl -- Reflexivity of equality, since author_of_elements is defined to be "Euclid".

end elements_author_is_euclid_l161_161385


namespace simplify_sqrt_450_l161_161470

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l161_161470


namespace max_abs_sum_on_ellipse_l161_161051

theorem max_abs_sum_on_ellipse :
  ∀ (x y : ℝ), 4 * x^2 + y^2 = 4 -> |x| + |y| ≤ (3 * Real.sqrt 2) / Real.sqrt 5 :=
by
  intro x y h
  sorry

end max_abs_sum_on_ellipse_l161_161051


namespace exists_three_distinct_div_l161_161694

theorem exists_three_distinct_div (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  ∀ m : ℕ, ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ abc ∣ (x * y * z) ∧ m ≤ x ∧ x < m + 2*c ∧ m ≤ y ∧ y < m + 2*c ∧ m ≤ z ∧ z < m + 2*c :=
by
  sorry

end exists_three_distinct_div_l161_161694


namespace track_length_l161_161810

theorem track_length (y : ℝ) 
  (H1 : ∀ b s : ℝ, b + s = y ∧ b = y / 2 - 120 ∧ s = 120)
  (H2 : ∀ b s : ℝ, b + s = y + 180 ∧ b = y / 2 + 60 ∧ s = y / 2 - 60) :
  y = 600 :=
by 
  sorry

end track_length_l161_161810


namespace sasha_total_items_l161_161477

/-
  Sasha bought pencils at 13 rubles each and pens at 20 rubles each,
  paying a total of 350 rubles. 
  Prove that the total number of pencils and pens Sasha bought is 23.
-/
theorem sasha_total_items
  (x y : ℕ) -- Define x as the number of pencils and y as the number of pens
  (H: 13 * x + 20 * y = 350) -- Given total cost condition
  : x + y = 23 := 
sorry

end sasha_total_items_l161_161477


namespace max_value_l161_161290

open Real

theorem max_value (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (ha1 : a ≤ 1) (hb1 : b ≤ 1) (hc1 : c ≤ 1/2) :
  sqrt (a * b * c) + sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ (1 / sqrt 2) + (1 / 2) :=
sorry

end max_value_l161_161290


namespace arithmetic_seq_property_l161_161906

-- Define the arithmetic sequence {a_n}
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the conditions
variable (a d : ℤ)
variable (h1 : arithmetic_seq a d 3 + arithmetic_seq a d 9 + arithmetic_seq a d 15 = 30)

-- Define the statement to be proved
theorem arithmetic_seq_property : 
  arithmetic_seq a d 17 - 2 * arithmetic_seq a d 13 = -10 :=
by
  sorry

end arithmetic_seq_property_l161_161906


namespace values_of_d_l161_161606

theorem values_of_d (a b c d : ℕ) 
  (h : (ad - 1) / (a + 1) + (bd - 1) / (b + 1) + (cd - 1) / (c + 1) = d) : 
  d = 1 ∨ d = 2 ∨ d = 3 := 
sorry

end values_of_d_l161_161606


namespace smaller_angle_of_parallelogram_l161_161803

theorem smaller_angle_of_parallelogram (x : ℝ) (h : x + 3 * x = 180) : x = 45 :=
sorry

end smaller_angle_of_parallelogram_l161_161803


namespace white_area_correct_l161_161683

/-- The dimensions of the sign and the letter components -/
def sign_width : ℕ := 18
def sign_height : ℕ := 6
def vertical_bar_height : ℕ := 6
def vertical_bar_width : ℕ := 1
def horizontal_bar_length : ℕ := 4
def horizontal_bar_width : ℕ := 1

/-- The areas of the components of each letter -/
def area_C : ℕ := 2 * (vertical_bar_height * vertical_bar_width) + (horizontal_bar_length * horizontal_bar_width)
def area_O : ℕ := 2 * (vertical_bar_height * vertical_bar_width) + 2 * (horizontal_bar_length * horizontal_bar_width)
def area_L : ℕ := (vertical_bar_height * vertical_bar_width) + (horizontal_bar_length * horizontal_bar_width)

/-- The total area of the sign -/
def total_sign_area : ℕ := sign_height * sign_width

/-- The total black area covered by the letters "COOL" -/
def total_black_area : ℕ := area_C + 2 * area_O + area_L

/-- The area of the white portion of the sign -/
def white_area : ℕ := total_sign_area - total_black_area

/-- Proof that the area of the white portion of the sign is 42 square units -/
theorem white_area_correct : white_area = 42 := by
  -- Calculation steps (skipped, though the result is expected to be 42)
  sorry

end white_area_correct_l161_161683


namespace expression_evaluation_l161_161753

theorem expression_evaluation : 
  54 + (42 / 14) + (27 * 17) - 200 - (360 / 6) + 2^4 = 272 := by 
  sorry

end expression_evaluation_l161_161753


namespace find_a12_l161_161596

namespace ArithmeticSequence

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a n = a 0 + n * d

theorem find_a12 {a : ℕ → α} (h1 : a 4 = 1) (h2 : a 7 + a 9 = 16) :
  a 12 = 15 := 
sorry

end ArithmeticSequence

end find_a12_l161_161596


namespace solve_system_l161_161842

def F (t : ℝ) : ℝ := 32 * t ^ 5 + 48 * t ^ 3 + 17 * t - 15

def system_of_equations (x y z : ℝ) : Prop :=
  (1 / x = (32 / y ^ 5) + (48 / y ^ 3) + (17 / y) - 15) ∧
  (1 / y = (32 / z ^ 5) + (48 / z ^ 3) + (17 / z) - 15) ∧
  (1 / z = (32 / x ^ 5) + (48 / x ^ 3) + (17 / x) - 15)

theorem solve_system : ∃ (x y z : ℝ), system_of_equations x y z ∧ x = 2 ∧ y = 2 ∧ z = 2 :=
by
  sorry -- Proof not included

end solve_system_l161_161842


namespace average_of_remaining_two_numbers_l161_161575

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 6.40)
  (h2 : (a + b) / 2 = 6.2)
  (h3 : (c + d) / 2 = 6.1) :
  ((e + f) / 2) = 6.9 :=
by
  sorry

end average_of_remaining_two_numbers_l161_161575


namespace right_triangle_hypotenuse_len_l161_161481

theorem right_triangle_hypotenuse_len (a b : ℕ) (c : ℝ) (h₁ : a = 1) (h₂ : b = 3) 
  (h₃ : a^2 + b^2 = c^2) : c = Real.sqrt 10 := by
  sorry

end right_triangle_hypotenuse_len_l161_161481


namespace players_per_group_l161_161384

theorem players_per_group (new_players : ℕ) (returning_players : ℕ) (groups : ℕ) 
  (h1 : new_players = 48) 
  (h2 : returning_players = 6) 
  (h3 : groups = 9) : 
  (new_players + returning_players) / groups = 6 :=
by
  sorry

end players_per_group_l161_161384


namespace xy_value_l161_161452

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 10) : x * y = 10 :=
by
  sorry

end xy_value_l161_161452


namespace cake_flour_amount_l161_161686

theorem cake_flour_amount (sugar_cups : ℕ) (flour_already_in : ℕ) (extra_flour_needed : ℕ) (total_flour : ℕ) 
  (h1 : sugar_cups = 7) 
  (h2 : flour_already_in = 2)
  (h3 : extra_flour_needed = 2)
  (h4 : total_flour = sugar_cups + extra_flour_needed) : 
  total_flour = 9 := 
sorry

end cake_flour_amount_l161_161686


namespace triangle_expression_simplification_l161_161690

variable (a b c : ℝ)

theorem triangle_expression_simplification (h1 : a + b > c) 
                                           (h2 : a + c > b) 
                                           (h3 : b + c > a) :
  |a - b - c| + |b - a - c| - |c - a + b| = a - b + c :=
sorry

end triangle_expression_simplification_l161_161690


namespace hypotenuse_length_l161_161617

theorem hypotenuse_length (a b c : ℝ) (hC : (a^2 + b^2) * (a^2 + b^2 + 1) = 12) (right_triangle : a^2 + b^2 = c^2) : 
  c = Real.sqrt 3 := 
by
  sorry

end hypotenuse_length_l161_161617


namespace oven_capacity_correct_l161_161790

-- Definitions for the conditions
def dough_time := 30 -- minutes
def bake_time := 30 -- minutes
def pizzas_per_batch := 3
def total_time := 5 * 60 -- minutes (5 hours)
def total_pizzas := 12

-- Calculation of the number of batches
def batches_needed := total_pizzas / pizzas_per_batch

-- Calculation of the time for making dough
def dough_preparation_time := batches_needed * dough_time

-- Calculation of the remaining time for baking
def remaining_baking_time := total_time - dough_preparation_time

-- Calculation of the number of 30-minute baking intervals
def baking_intervals := remaining_baking_time / bake_time

-- Calculation of the capacity of the oven
def oven_capacity := total_pizzas / baking_intervals

theorem oven_capacity_correct : oven_capacity = 2 := by
  sorry

end oven_capacity_correct_l161_161790


namespace trapezoid_perimeter_l161_161165

theorem trapezoid_perimeter (AB CD AD BC h : ℝ)
  (AB_eq : AB = 40)
  (CD_eq : CD = 70)
  (AD_eq_BC : AD = BC)
  (h_eq : h = 24)
  : AB + BC + CD + AD = 110 + 2 * Real.sqrt 801 :=
by
  -- Proof goes here, you can replace this comment with actual proof.
  sorry

end trapezoid_perimeter_l161_161165


namespace gcd_of_462_and_330_l161_161178

theorem gcd_of_462_and_330 :
  Nat.gcd 462 330 = 66 :=
sorry

end gcd_of_462_and_330_l161_161178


namespace wilson_total_cost_l161_161102

noncomputable def total_cost_wilson_pays : ℝ :=
let hamburger_price : ℝ := 5
let cola_price : ℝ := 2
let fries_price : ℝ := 3
let sundae_price : ℝ := 4
let nugget_price : ℝ := 1.5
let salad_price : ℝ := 6.25
let hamburger_count : ℕ := 2
let cola_count : ℕ := 3
let nugget_count : ℕ := 4

let total_before_discounts := (hamburger_count * hamburger_price) +
                              (cola_count * cola_price) +
                              fries_price +
                              sundae_price +
                              (nugget_count * nugget_price) +
                              salad_price

let free_nugget_discount := 1 * nugget_price
let total_after_promotion := total_before_discounts - free_nugget_discount
let coupon_discount := 4
let total_after_coupon := total_after_promotion - coupon_discount
let loyalty_discount := 0.10 * total_after_coupon
let total_after_loyalty := total_after_coupon - loyalty_discount

total_after_loyalty

theorem wilson_total_cost : total_cost_wilson_pays = 26.77 := 
by
  sorry

end wilson_total_cost_l161_161102


namespace parabola_zero_sum_l161_161338

-- Define the original parabola equation and transformations
def original_parabola (x : ℝ) : ℝ := (x - 3) ^ 2 + 4

-- Define the resulting parabola after transformations
def transformed_parabola (x : ℝ) : ℝ := -(x - 7) ^ 2 + 1

-- Prove that the resulting parabola has zeros at p and q such that p + q = 14
theorem parabola_zero_sum : 
  ∃ (p q : ℝ), transformed_parabola p = 0 ∧ transformed_parabola q = 0 ∧ p + q = 14 :=
by
  sorry

end parabola_zero_sum_l161_161338


namespace repeating_decimal_sum_l161_161162

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l161_161162


namespace final_price_calculation_l161_161422

theorem final_price_calculation 
  (ticket_price : ℝ)
  (initial_discount : ℝ)
  (additional_discount : ℝ)
  (sales_tax : ℝ)
  (final_price : ℝ) 
  (h1 : ticket_price = 200) 
  (h2 : initial_discount = 0.25) 
  (h3 : additional_discount = 0.15) 
  (h4 : sales_tax = 0.07)
  (h5 : final_price = (ticket_price * (1 - initial_discount)) * (1 - additional_discount) * (1 + sales_tax)):
  final_price = 136.43 :=
by
  sorry

end final_price_calculation_l161_161422


namespace total_pencils_l161_161417

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (h1 : pencils_per_child = 2) (h2 : children = 9) : pencils_per_child * children = 18 :=
sorry

end total_pencils_l161_161417


namespace sqrt_a_minus_b_squared_eq_one_l161_161244

noncomputable def PointInThirdQuadrant (a b : ℝ) : Prop :=
  a < 0 ∧ b < 0

noncomputable def DistanceToYAxis (a : ℝ) : Prop :=
  abs a = 5

noncomputable def BCondition (b : ℝ) : Prop :=
  abs (b + 1) = 3

theorem sqrt_a_minus_b_squared_eq_one
    (a b : ℝ)
    (h1 : PointInThirdQuadrant a b)
    (h2 : DistanceToYAxis a)
    (h3 : BCondition b) :
    Real.sqrt ((a - b) ^ 2) = 1 := 
  sorry

end sqrt_a_minus_b_squared_eq_one_l161_161244


namespace range_of_m_l161_161539

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 1 > m

def proposition_q (m : ℝ) : Prop :=
  3 - m > 1

theorem range_of_m (m : ℝ) (p_false : ¬proposition_p m) (q_true : proposition_q m) (pq_false : ¬(proposition_p m ∧ proposition_q m)) (porq_true : proposition_p m ∨ proposition_q m) : 
  1 ≤ m ∧ m < 2 := 
sorry

end range_of_m_l161_161539


namespace smallest_5_digit_number_divisible_by_and_factor_of_l161_161791

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

def is_divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = y * k

def is_factor_of (x y : ℕ) : Prop := is_divisible_by y x

def is_5_digit_number (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem smallest_5_digit_number_divisible_by_and_factor_of :
  ∃ n : ℕ,
    is_5_digit_number n ∧
    is_divisible_by n 32 ∧
    is_divisible_by n 45 ∧
    is_divisible_by n 54 ∧
    is_factor_of n 30 ∧
    (∀ m : ℕ, is_5_digit_number m → is_divisible_by m 32 → is_divisible_by m 45 → is_divisible_by m 54 → is_factor_of m 30 → n ≤ m) :=
sorry

end smallest_5_digit_number_divisible_by_and_factor_of_l161_161791


namespace solve_f_sqrt_2009_l161_161831

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_never_zero : ∀ x : ℝ, f x ≠ 0
axiom functional_eq : ∀ x y : ℝ, f (x - y) = 2009 * f x * f y

theorem solve_f_sqrt_2009 :
  f (sqrt 2009) = 1 / 2009 := sorry

end solve_f_sqrt_2009_l161_161831


namespace find_p_if_geometric_exists_p_arithmetic_sequence_l161_161765

variable (a : ℕ → ℝ) (p : ℝ)

-- Condition 1: a_1 = 1
axiom a1_eq_1 : a 1 = 1

-- Condition 2: a_n + a_{n+1} = pn + 1
axiom a_recurrence : ∀ n : ℕ, a n + a (n + 1) = p * n + 1

-- Question 1: If a_1, a_2, a_4 form a geometric sequence, find p
theorem find_p_if_geometric (h_geometric : (a 2)^2 = (a 1) * (a 4)) : p = 2 := by
  -- Proof goes here
  sorry

-- Question 2: Does there exist a p such that the sequence {a_n} is an arithmetic sequence?
theorem exists_p_arithmetic_sequence : ∃ p : ℝ, (∀ n : ℕ, a n + a (n + 1) = p * n + 1) ∧ 
                                         (∀ m n : ℕ, a (m + n) - a n = m * p) := by
  -- Proof goes here
  exists 2
  sorry

end find_p_if_geometric_exists_p_arithmetic_sequence_l161_161765


namespace n_minus_one_divides_n_squared_plus_n_sub_two_l161_161455

theorem n_minus_one_divides_n_squared_plus_n_sub_two (n : ℕ) : (n - 1) ∣ (n ^ 2 + n - 2) :=
sorry

end n_minus_one_divides_n_squared_plus_n_sub_two_l161_161455


namespace probability_of_stopping_on_H_l161_161572

theorem probability_of_stopping_on_H (y : ℚ)
  (h1 : (1 / 5) + (1 / 4) + y + y + (1 / 10) = 1)
  : y = 9 / 40 :=
sorry

end probability_of_stopping_on_H_l161_161572


namespace sum_of_ages_l161_161230

theorem sum_of_ages (a b c d : ℕ) (h1 : a * b = 20) (h2 : c * d = 28) (distinct : ∀ (x y : ℕ), (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≠ y) : a + b + c + d = 19 :=
sorry

end sum_of_ages_l161_161230


namespace probability_3_closer_0_0_to_6_l161_161468

noncomputable def probability_closer_to_3_than_0 (a b c : ℝ) : ℝ :=
  if h₁ : a < b ∧ b < c then
    (c - ((a + b) / 2)) / (c - a)
  else 0

theorem probability_3_closer_0_0_to_6 : probability_closer_to_3_than_0 0 3 6 = 0.75 := by
  sorry

end probability_3_closer_0_0_to_6_l161_161468


namespace adam_first_half_correct_l161_161383

-- Define the conditions
def second_half_correct := 2
def points_per_question := 8
def final_score := 80

-- Define the number of questions Adam answered correctly in the first half
def first_half_correct :=
  (final_score - (second_half_correct * points_per_question)) / points_per_question

-- Statement to prove
theorem adam_first_half_correct : first_half_correct = 8 :=
by
  -- skipping the proof
  sorry

end adam_first_half_correct_l161_161383


namespace even_n_if_fraction_is_integer_l161_161387

theorem even_n_if_fraction_is_integer (n : ℕ) (h_pos : 0 < n) :
  (∃ a b : ℕ, 0 < b ∧ (a^2 + n^2) % (b^2 - n^2) = 0) → n % 2 = 0 := 
sorry

end even_n_if_fraction_is_integer_l161_161387


namespace c_divides_n_l161_161246

theorem c_divides_n (a b c n : ℤ) (h : a * n^2 + b * n + c = 0) : c ∣ n :=
sorry

end c_divides_n_l161_161246


namespace pieces_per_serving_l161_161045

-- Definitions based on conditions
def jaredPopcorn : Nat := 90
def friendPopcorn : Nat := 60
def numberOfFriends : Nat := 3
def totalServings : Nat := 9

-- Statement to verify
theorem pieces_per_serving : 
  ((jaredPopcorn + numberOfFriends * friendPopcorn) / totalServings) = 30 :=
by
  sorry

end pieces_per_serving_l161_161045


namespace purely_imaginary_z_l161_161800

open Complex

theorem purely_imaginary_z (b : ℝ) (h : z = (1 + b * I) / (2 + I) ∧ im z = 0) : z = -I :=
by
  sorry

end purely_imaginary_z_l161_161800


namespace number_told_to_sasha_l161_161445

-- Defining concepts
def two_digit_number (a b : ℕ) : Prop := a < 10 ∧ b < 10 ∧ a * b ≥ 1

def product_of_digits (a b : ℕ) (P : ℕ) : Prop := P = a * b

def sum_of_digits (a b : ℕ) (S : ℕ) : Prop := S = a + b

def petya_guesses_in_three_attempts (P : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ), P = a * b ∧ P = c * d ∧ P = e * f ∧ 
  (a * b) ≠ (c * d) ∧ (a * b) ≠ (e * f) ∧ (c * d) ≠ (e * f)

def sasha_guesses_in_four_attempts (S : ℕ) : Prop :=
  ∃ (a b c d e f g h i j : ℕ), 
  S = a + b ∧ S = c + d ∧ S = e + f ∧ S = g + h ∧ S = i + j ∧
  (a + b) ≠ (c + d) ∧ (a + b) ≠ (e + f) ∧ (a + b) ≠ (g + h) ∧ (a + b) ≠ (i + j) ∧ 
  (c + d) ≠ (e + f) ∧ (c + d) ≠ (g + h) ∧ (c + d) ≠ (i + j) ∧ 
  (e + f) ≠ (g + h) ∧ (e + f) ≠ (i + j) ∧ 
  (g + h) ≠ (i + j)

theorem number_told_to_sasha : ∃ (S : ℕ), 
  ∀ (a b : ℕ), two_digit_number a b → 
  (product_of_digits a b (a * b) → petya_guesses_in_three_attempts (a * b)) → 
  (sum_of_digits a b S → sasha_guesses_in_four_attempts S) → S = 10 :=
by
  sorry

end number_told_to_sasha_l161_161445


namespace find_transform_l161_161611

structure Vector3D (α : Type) := (x y z : α)

def T (u : Vector3D ℝ) : Vector3D ℝ := sorry

axiom linearity (a b : ℝ) (u v : Vector3D ℝ) : T (Vector3D.mk (a * u.x + b * v.x) (a * u.y + b * v.y) (a * u.z + b * v.z)) = 
                      Vector3D.mk (a * (T u).x + b * (T v).x) (a * (T u).y + b * (T v).y) (a * (T u).z + b * (T v).z)

axiom cross_product (u v : Vector3D ℝ) : T (Vector3D.mk (u.y * v.z - u.z * v.y) (u.z * v.x - u.x * v.z) (u.x * v.y - u.y * v.x)) = 
                    (Vector3D.mk ((T u).y * (T v).z - (T u).z * (T v).y) ((T u).z * (T v).x - (T u).x * (T v).z) ((T u).x * (T v).y - (T u).y * (T v).x))

axiom transform1 : T (Vector3D.mk 3 3 7) = Vector3D.mk 2 (-4) 5
axiom transform2 : T (Vector3D.mk (-2) 5 4) = Vector3D.mk 6 1 0

theorem find_transform : T (Vector3D.mk 5 15 11) = Vector3D.mk a b c := sorry

end find_transform_l161_161611


namespace tan_alpha_plus_pi_div_4_l161_161275

noncomputable def tan_plus_pi_div_4 (α : ℝ) : ℝ := Real.tan (α + Real.pi / 4)

theorem tan_alpha_plus_pi_div_4 (α : ℝ) 
  (h1 : α > Real.pi / 2) 
  (h2 : α < Real.pi) 
  (h3 : (Real.cos α, Real.sin α) • (Real.cos α ^ 2, Real.sin α - 1) = 1 / 5)
  : tan_plus_pi_div_4 α = -1 / 7 := sorry

end tan_alpha_plus_pi_div_4_l161_161275


namespace christmas_tree_problem_l161_161232

theorem christmas_tree_problem (b t : ℕ) (h1 : t = b + 1) (h2 : 2 * b = t - 1) : b = 3 ∧ t = 4 :=
by
  sorry

end christmas_tree_problem_l161_161232


namespace range_of_m_l161_161825

variable {x m : ℝ}

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) : (¬ p m ∨ ¬ q m) → m ≥ 2 := 
sorry

end range_of_m_l161_161825


namespace unique_toy_value_l161_161222

/-- Allie has 9 toys in total. The total worth of these toys is $52. 
One toy has a certain value "x" dollars and the remaining 8 toys each have a value of $5. 
Prove that the value of the unique toy is $12. -/
theorem unique_toy_value (x : ℕ) (h1 : 1 + 8 = 9) (h2 : x + 8 * 5 = 52) : x = 12 :=
by
  sorry

end unique_toy_value_l161_161222


namespace negation_exists_cube_positive_l161_161505

theorem negation_exists_cube_positive :
  ¬ (∃ x : ℝ, x^3 > 0) ↔ ∀ x : ℝ, x^3 ≤ 0 := by
  sorry

end negation_exists_cube_positive_l161_161505


namespace value_of_a_l161_161342

theorem value_of_a (P Q : Set ℝ) (a : ℝ) :
  (P = {x | x^2 = 1}) →
  (Q = {x | ax = 1}) →
  (Q ⊆ P) →
  (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  sorry

end value_of_a_l161_161342


namespace intersection_point_l161_161026

def line_eq (x y z : ℝ) : Prop :=
  (x - 1) / 1 = (y + 1) / 0 ∧ (y + 1) / 0 = (z - 1) / -1

def plane_eq (x y z : ℝ) : Prop :=
  3 * x - 2 * y - 4 * z - 8 = 0

theorem intersection_point : 
  ∃ (x y z : ℝ), line_eq x y z ∧ plane_eq x y z ∧ x = -6 ∧ y = -1 ∧ z = 8 :=
by 
  sorry

end intersection_point_l161_161026


namespace flour_per_special_crust_l161_161584

-- Definitions of daily pie crusts and flour usage for standard crusts
def daily_pie_crusts := 50
def flour_per_standard_crust := 1 / 10
def total_daily_flour := daily_pie_crusts * flour_per_standard_crust

-- Definitions for special pie crusts today
def special_pie_crusts := 25
def total_special_flour := total_daily_flour / special_pie_crusts

-- Problem statement in Lean
theorem flour_per_special_crust :
  total_special_flour = 1 / 5 := by
  sorry

end flour_per_special_crust_l161_161584


namespace geometric_arithmetic_sequence_ratio_l161_161526

-- Given a positive geometric sequence {a_n} with a_3, a_5, a_6 forming an arithmetic sequence,
-- we need to prove that (a_3 + a_5) / (a_4 + a_6) is among specific values {1, (sqrt 5 - 1) / 2}

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_pos: ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_arith : 2 * a 5 = a 3 + a 6) :
  (a 3 + a 5) / (a 4 + a 6) = 1 ∨ (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 :=
by
  -- The proof is omitted
  sorry

end geometric_arithmetic_sequence_ratio_l161_161526


namespace floor_pi_plus_four_l161_161848

theorem floor_pi_plus_four : Int.floor (Real.pi + 4) = 7 := by
  sorry

end floor_pi_plus_four_l161_161848


namespace price_percentage_combined_assets_l161_161854

variable (A B P : ℝ)

-- Conditions
axiom h1 : P = 1.20 * A
axiom h2 : P = 2 * B

-- Statement
theorem price_percentage_combined_assets : (P / (A + B)) * 100 = 75 := by
  sorry

end price_percentage_combined_assets_l161_161854


namespace number_properties_l161_161202

theorem number_properties : 
    ∃ (N : ℕ), 
    35 < N ∧ N < 70 ∧ N % 6 = 3 ∧ N % 8 = 1 ∧ N = 57 :=
by 
  sorry

end number_properties_l161_161202


namespace smallest_possible_input_l161_161692

def F (n : ℕ) := 9 * n + 120

theorem smallest_possible_input : ∃ n : ℕ, n > 0 ∧ F n = 129 :=
by {
  -- Here we would provide the proof steps, but we use sorry for now.
  sorry
}

end smallest_possible_input_l161_161692


namespace percentage_increase_after_decrease_l161_161111

variable (P : ℝ) (x : ℝ)

-- Conditions
def decreased_price : ℝ := 0.80 * P
def final_price_condition : Prop := 0.80 * P + (x / 100) * (0.80 * P) = 1.04 * P
def correct_answer : Prop := x = 30

-- The proof goal
theorem percentage_increase_after_decrease : final_price_condition P x → correct_answer x :=
by sorry

end percentage_increase_after_decrease_l161_161111


namespace bridge_length_l161_161277

variable (speed : ℝ) (time_minutes : ℝ)
variable (time_hours : ℝ := time_minutes / 60)

theorem bridge_length (h1 : speed = 5) (h2 : time_minutes = 15) : 
  speed * time_hours = 1.25 := by
  sorry

end bridge_length_l161_161277


namespace number_of_ways_two_girls_together_l161_161523

theorem number_of_ways_two_girls_together
  (boys girls : ℕ)
  (total_people : ℕ)
  (ways : ℕ) :
  boys = 3 →
  girls = 3 →
  total_people = boys + girls →
  ways = 432 :=
by
  intros
  sorry

end number_of_ways_two_girls_together_l161_161523


namespace xy_value_l161_161622

theorem xy_value : 
  ∀ (x y : ℝ),
  (∀ (A B C : ℝ × ℝ), A = (1, 8) ∧ B = (x, y) ∧ C = (6, 3) → 
  (C.1 = (A.1 + B.1) / 2) ∧ (C.2 = (A.2 + B.2) / 2)) → 
  x * y = -22 :=
sorry

end xy_value_l161_161622


namespace find_b_in_triangle_l161_161780

theorem find_b_in_triangle (a B C A b : ℝ)
  (ha : a = Real.sqrt 3)
  (hB : Real.sin B = 1 / 2)
  (hC : C = Real.pi / 6)
  (hA : A = 2 * Real.pi / 3) :
  b = 1 :=
by
  -- proof omitted
  sorry

end find_b_in_triangle_l161_161780


namespace range_of_m_l161_161613

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), ¬((x - m) < -3) ∧ (1 + 2*x)/3 ≥ x - 1) ∧ 
  (∀ (x1 x2 x3 : ℤ), 
    (¬((x1 - m) < -3) ∧ (1 + 2 * x1)/3 ≥ x1 - 1) ∧
    (¬((x2 - m) < -3) ∧ (1 + 2 * x2)/3 ≥ x2 - 1) ∧
    (¬((x3 - m) < -3) ∧ (1 + 2 * x3)/3 ≥ x3 - 1)) →
  (4 ≤ m ∧ m < 5) :=
by 
  sorry

end range_of_m_l161_161613


namespace car_speed_first_hour_l161_161707

theorem car_speed_first_hour (x : ℕ) (h1 : 60 > 0) (h2 : 40 > 0) (h3 : 2 > 0) (avg_speed : 40 = (x + 60) / 2) : x = 20 := 
by
  sorry

end car_speed_first_hour_l161_161707


namespace number_of_squares_l161_161454

def side_plywood : ℕ := 50
def side_square_1 : ℕ := 10
def side_square_2 : ℕ := 20
def total_cut_length : ℕ := 280

/-- Number of squares obtained given the side lengths of the plywood and the cut lengths -/
theorem number_of_squares (x y : ℕ) (h1 : 100 * x + 400 * y = side_plywood^2)
  (h2 : 40 * x + 80 * y = total_cut_length) : x + y = 16 :=
sorry

end number_of_squares_l161_161454


namespace determine_angles_l161_161331

theorem determine_angles 
  (small_angle1 : ℝ) 
  (small_angle2 : ℝ) 
  (large_angle1 : ℝ) 
  (large_angle2 : ℝ) 
  (triangle_sum_property : ∀ a b c : ℝ, a + b + c = 180) 
  (exterior_angle_property : ∀ a c : ℝ, a + c = 180) :
  (small_angle1 = 70) → 
  (small_angle2 = 180 - 130) → 
  (large_angle1 = 45) → 
  (large_angle2 = 50) → 
  ∃ α β : ℝ, α = 120 ∧ β = 85 :=
by
  intros h1 h2 h3 h4
  sorry

end determine_angles_l161_161331


namespace average_of_11_results_l161_161096

theorem average_of_11_results (a b c : ℝ) (avg_first_6 avg_last_6 sixth_result avg_all_11 : ℝ)
  (h1 : avg_first_6 = 58)
  (h2 : avg_last_6 = 63)
  (h3 : sixth_result = 66) :
  avg_all_11 = 60 :=
by
  sorry

end average_of_11_results_l161_161096


namespace intersection_eq_l161_161905

def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }
def B : Set ℝ := { x | Real.log (1 - x) > 0 }

theorem intersection_eq : A ∩ B = Set.Icc (-1) 0 :=
by
  sorry

end intersection_eq_l161_161905


namespace unique_increasing_seq_l161_161896

noncomputable def unique_seq (a : ℕ → ℕ) (r : ℝ) : Prop :=
∀ (b : ℕ → ℕ), (∀ n, b n = 3 * n - 2 → ∑' n, r ^ (b n) = 1 / 2 ) → (∀ n, a n = b n)

theorem unique_increasing_seq {r : ℝ} 
  (hr : 0.4 < r ∧ r < 0.5) 
  (hc : r^3 + 2*r = 1):
  ∃ a : ℕ → ℕ, (∀ n, a n = 3 * n - 2) ∧ (∑'(n), r^(a n) = 1/2) ∧ unique_seq a r :=
by
  sorry

end unique_increasing_seq_l161_161896


namespace jacob_peter_age_ratio_l161_161525

theorem jacob_peter_age_ratio
  (Drew Maya Peter John Jacob : ℕ)
  (h1: Drew = Maya + 5)
  (h2: Peter = Drew + 4)
  (h3: John = 2 * Maya)
  (h4: John = 30)
  (h5: Jacob = 11) :
  Jacob + 2 = 1 / 2 * (Peter + 2) := by
  sorry

end jacob_peter_age_ratio_l161_161525


namespace value_of_expression_l161_161129

theorem value_of_expression (a b : ℝ) (h1 : 3 * a^2 + 9 * a - 21 = 0) (h2 : 3 * b^2 + 9 * b - 21 = 0) :
  (3 * a - 4) * (5 * b - 6) = -27 :=
by
  -- The proof is omitted, place 'sorry' to indicate it.
  sorry

end value_of_expression_l161_161129


namespace vertex_of_f_C_l161_161967

def f_A (x : ℝ) : ℝ := (x + 4) ^ 2 - 3
def f_B (x : ℝ) : ℝ := (x + 4) ^ 2 + 3
def f_C (x : ℝ) : ℝ := (x - 4) ^ 2 - 3
def f_D (x : ℝ) : ℝ := (x - 4) ^ 2 + 3

theorem vertex_of_f_C : ∃ (h k : ℝ), h = 4 ∧ k = -3 ∧ ∀ x, f_C x = (x - h) ^ 2 + k :=
by
  sorry

end vertex_of_f_C_l161_161967


namespace partition_of_sum_l161_161196

-- Define the conditions
def is_positive_integer (n : ℕ) : Prop := n > 0
def is_bounded_integer (n : ℕ) : Prop := n ≤ 10
def can_be_partitioned (S : ℕ) (integers : List ℕ) : Prop :=
  ∃ (A B : List ℕ), 
    A.sum ≤ 70 ∧ 
    B.sum ≤ 70 ∧ 
    A ++ B = integers

-- Define the theorem statement
theorem partition_of_sum (S : ℕ) (integers : List ℕ)
  (h1 : ∀ x ∈ integers, is_positive_integer x ∧ is_bounded_integer x)
  (h2 : List.sum integers = S) :
  S ≤ 133 ↔ can_be_partitioned S integers :=
sorry

end partition_of_sum_l161_161196


namespace cos_2alpha_plus_pi_div_2_eq_neg_24_div_25_l161_161755

theorem cos_2alpha_plus_pi_div_2_eq_neg_24_div_25
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (h_tanα : Real.tan α = 4 / 3) :
  Real.cos (2 * α + π / 2) = - 24 / 25 :=
by sorry

end cos_2alpha_plus_pi_div_2_eq_neg_24_div_25_l161_161755


namespace ones_digit_largest_power_of_3_dividing_18_factorial_l161_161833

theorem ones_digit_largest_power_of_3_dividing_18_factorial :
  (3^8 % 10) = 1 :=
by sorry

end ones_digit_largest_power_of_3_dividing_18_factorial_l161_161833


namespace complex_in_third_quadrant_l161_161247

open Complex

noncomputable def quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else 4

theorem complex_in_third_quadrant (z : ℂ) (h : (2 + I) * z = -I) : quadrant z = 3 := by
  sorry

end complex_in_third_quadrant_l161_161247


namespace mean_of_remaining_l161_161419

variable (a b c : ℝ)
variable (mean_of_four : ℝ := 90)
variable (largest : ℝ := 105)

theorem mean_of_remaining (h1 : (a + b + c + largest) / 4 = mean_of_four) : (a + b + c) / 3 = 85 := by
  sorry

end mean_of_remaining_l161_161419


namespace cos_beta_value_l161_161880

noncomputable def cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) : Real :=
  Real.cos β

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 3 / 5) (h_cos_alpha_plus_beta : Real.cos (α + β) = 5 / 13) :
  Real.cos β = 56 / 65 :=
by
  sorry

end cos_beta_value_l161_161880


namespace total_handshakes_calculation_l161_161203

-- Define the conditions
def teams := 3
def players_per_team := 5
def total_players := teams * players_per_team
def referees := 2

def handshakes_among_players := (total_players * (players_per_team * (teams - 1))) / 2
def handshakes_with_referees := total_players * referees

def total_handshakes := handshakes_among_players + handshakes_with_referees

-- Define the theorem statement
theorem total_handshakes_calculation :
  total_handshakes = 105 :=
by
  sorry

end total_handshakes_calculation_l161_161203


namespace remainder_of_power_division_l161_161261

-- Define the main entities
def power : ℕ := 3
def exponent : ℕ := 19
def divisor : ℕ := 10

-- Define the proof problem
theorem remainder_of_power_division :
  (power ^ exponent) % divisor = 7 := 
  by 
    sorry

end remainder_of_power_division_l161_161261


namespace perpendicular_lines_slope_product_l161_161558

theorem perpendicular_lines_slope_product (a : ℝ) (x y : ℝ) :
  let l1 := ax + y + 2 = 0
  let l2 := x + y = 0
  ( -a * -1 = -1 ) -> a = -1 :=
sorry

end perpendicular_lines_slope_product_l161_161558


namespace mod_equiv_l161_161852

theorem mod_equiv (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (25 * m + 3 * n) % 83 = 0 ↔ (3 * m + 7 * n) % 83 = 0 :=
by
  sorry

end mod_equiv_l161_161852


namespace midpoint_AB_find_Q_find_H_l161_161818

-- Problem 1: Midpoint of AB
theorem midpoint_AB (x1 y1 x2 y2 : ℝ) : 
  let A := (x1, y1)
  let B := (x2, y2)
  let M := ( (x1 + x2) / 2, (y1 + y2) / 2 )
  M = ( (x1 + x2) / 2, (y1 + y2) / 2 )
:= 
  -- The lean statement that shows the midpoint formula is correct.
  sorry

-- Problem 2: Coordinates of Q given midpoint
theorem find_Q (px py mx my : ℝ) : 
  let P := (px, py)
  let M := (mx, my)
  let Q := (2 * mx - px, 2 * my - py)
  ( (px + Q.1) / 2 = mx ∧ (py + Q.2) / 2 = my )
:= 
  -- Lean statement to find Q
  sorry

-- Problem 3: Coordinates of H given midpoints coinciding
theorem find_H (xE yE xF yF xG yG : ℝ) :
  let E := (xE, yE)
  let F := (xF, yF)
  let G := (xG, yG)
  ∃ xH yH : ℝ, 
    ( (xE + xH) / 2 = (xF + xG) / 2 ∧ (yE + yH) / 2 = (yF + yG) / 2 ) ∨
    ( (xF + xH) / 2 = (xE + xG) / 2 ∧ (yF + yH) / 2 = (yE + yG) / 2 ) ∨
    ( (xG + xH) / 2 = (xE + xF) / 2 ∧ (yG + yH) / 2 = (yE + yF) / 2 )
:=
  -- Lean statement to find H
  sorry

end midpoint_AB_find_Q_find_H_l161_161818


namespace equation_graph_is_ellipse_l161_161030

theorem equation_graph_is_ellipse :
  ∃ a b c d : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b * (x^2 - 72 * y^2) + a * x + d = a * c * (y - 6)^2 :=
sorry

end equation_graph_is_ellipse_l161_161030


namespace ines_bought_3_pounds_l161_161524

-- Define initial and remaining money of Ines
def initial_money : ℕ := 20
def remaining_money : ℕ := 14

-- Define the cost per pound of peaches
def cost_per_pound : ℕ := 2

-- The total money spent on peaches
def money_spent := initial_money - remaining_money

-- The number of pounds of peaches bought
def pounds_of_peaches := money_spent / cost_per_pound

-- The proof problem
theorem ines_bought_3_pounds :
  pounds_of_peaches = 3 :=
by
  sorry

end ines_bought_3_pounds_l161_161524


namespace remainder_91_pow_91_mod_100_l161_161487

-- Definitions
def large_power_mod (a b n : ℕ) : ℕ :=
  (a^b) % n

-- Statement
theorem remainder_91_pow_91_mod_100 : large_power_mod 91 91 100 = 91 :=
by
  sorry

end remainder_91_pow_91_mod_100_l161_161487


namespace oranges_in_bin_l161_161550

theorem oranges_in_bin (initial : ℕ) (thrown_away : ℕ) (added : ℕ) (result : ℕ)
    (h_initial : initial = 40)
    (h_thrown_away : thrown_away = 25)
    (h_added : added = 21)
    (h_result : result = 36) : initial - thrown_away + added = result :=
by
  -- skipped proof steps
  exact sorry

end oranges_in_bin_l161_161550


namespace probability_empty_chair_on_sides_7_chairs_l161_161469

theorem probability_empty_chair_on_sides_7_chairs :
  (1 : ℚ) / (35 : ℚ) = 0.2 := by
  sorry

end probability_empty_chair_on_sides_7_chairs_l161_161469


namespace bail_rate_l161_161552

theorem bail_rate 
  (distance_to_shore : ℝ) 
  (shore_speed : ℝ) 
  (leak_rate : ℝ) 
  (boat_capacity : ℝ) 
  (time_to_shore_min : ℝ) 
  (net_water_intake : ℝ)
  (r : ℝ) :
  distance_to_shore = 2 →
  shore_speed = 3 →
  leak_rate = 12 →
  boat_capacity = 40 →
  time_to_shore_min = 40 →
  net_water_intake = leak_rate - r →
  net_water_intake * (time_to_shore_min) ≤ boat_capacity →
  r ≥ 11 :=
by
  intros h_dist h_speed h_leak h_cap h_time h_net h_ineq
  sorry

end bail_rate_l161_161552


namespace total_arrangements_l161_161605

theorem total_arrangements :
  let students := 6
  let venueA := 1
  let venueB := 2
  let venueC := 3
  (students.choose venueA) * ((students - venueA).choose venueB) = 60 :=
by
  -- placeholder for the proof
  sorry

end total_arrangements_l161_161605


namespace scientific_notation_40_9_billion_l161_161784

theorem scientific_notation_40_9_billion :
  (40.9 * 10^9) = 4.09 * 10^9 :=
by
  sorry

end scientific_notation_40_9_billion_l161_161784


namespace triangle_inequality_l161_161571

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  3 * (a * b + a * c + b * c) ≤ (a + b + c) ^ 2 ∧ (a + b + c) ^ 2 < 4 * (a * b + a * c + b * c) :=
sorry

end triangle_inequality_l161_161571


namespace white_pairs_coincide_l161_161190

theorem white_pairs_coincide :
  ∀ (red_triangles blue_triangles white_triangles : ℕ)
    (red_pairs blue_pairs red_blue_pairs : ℕ),
  red_triangles = 4 →
  blue_triangles = 4 →
  white_triangles = 6 →
  red_pairs = 3 →
  blue_pairs = 2 →
  red_blue_pairs = 1 →
  (2 * white_triangles - red_triangles - blue_triangles - red_blue_pairs) = white_triangles →
  6 = white_triangles :=
by
  intros red_triangles blue_triangles white_triangles
         red_pairs blue_pairs red_blue_pairs
         H_red H_blue H_white
         H_red_pairs H_blue_pairs H_red_blue_pairs
         H_pairs
  sorry

end white_pairs_coincide_l161_161190


namespace group_size_systematic_sampling_l161_161898

-- Define the total number of viewers
def total_viewers : ℕ := 10000

-- Define the number of viewers to be selected
def selected_viewers : ℕ := 10

-- Lean statement to prove the group size for systematic sampling
theorem group_size_systematic_sampling (n_total n_selected : ℕ) : n_total = total_viewers → n_selected = selected_viewers → (n_total / n_selected) = 1000 :=
by
  intros h_total h_selected
  rw [h_total, h_selected]
  sorry

end group_size_systematic_sampling_l161_161898


namespace focal_radii_l161_161272

theorem focal_radii (a e x y : ℝ) (h1 : x + y = 2 * a) (h2 : x - y = 2 * e) : x = a + e ∧ y = a - e :=
by
  -- We will add here the actual proof, but for now, we leave it as a placeholder.
  sorry

end focal_radii_l161_161272


namespace add_inequality_of_greater_l161_161484

theorem add_inequality_of_greater (a b c d : ℝ) (h₁ : a > b) (h₂ : c > d) : a + c > b + d := 
by sorry

end add_inequality_of_greater_l161_161484


namespace height_of_windows_l161_161804

theorem height_of_windows
  (L W H d_l d_w w_w : ℕ)
  (C T : ℕ)
  (hl : L = 25)
  (hw : W = 15)
  (hh : H = 12)
  (hdl : d_l = 6)
  (hdw : d_w = 3)
  (hww : w_w = 3)
  (hc : C = 3)
  (ht : T = 2718):
  ∃ h : ℕ, 960 - (18 + 9 * h) = 906 ∧ 
  (T = C * (960 - (18 + 9 * h))) ∧
  (960 = 2 * (L * H) + 2 * (W * H)) ∧ 
  (18 = d_l * d_w) ∧ 
  (9 * h = 3 * (h * w_w)) := 
sorry

end height_of_windows_l161_161804


namespace june_found_total_eggs_l161_161751

def eggs_in_tree_1 (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest
def eggs_in_tree_2 (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest
def eggs_in_yard (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest

def total_eggs (eggs_tree_1 : ℕ) (eggs_tree_2 : ℕ) (eggs_yard : ℕ) : ℕ :=
eggs_tree_1 + eggs_tree_2 + eggs_yard

theorem june_found_total_eggs :
  total_eggs (eggs_in_tree_1 2 5) (eggs_in_tree_2 1 3) (eggs_in_yard 1 4) = 17 :=
by
  sorry

end june_found_total_eggs_l161_161751


namespace inequality_proof_l161_161706

theorem inequality_proof
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (hxyz : x * y * z ≥ 1) :
  (x^3 - x^2) / (x^5 + y^2 + z^2) +
  (y^5 - y^2) / (y^5 + z^2 + x^2) +
  (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := 
sorry

end inequality_proof_l161_161706


namespace geometric_sequence_when_k_is_neg_one_l161_161314

noncomputable def S (n : ℕ) (k : ℝ) : ℝ := 3^n + k

noncomputable def a (n : ℕ) (k : ℝ) : ℝ :=
  if n = 1 then S 1 k else S n k - S (n-1) k

theorem geometric_sequence_when_k_is_neg_one :
  ∀ n : ℕ, n ≥ 1 → ∃ r : ℝ, ∀ m : ℕ, m ≥ 1 → a m (-1) = a 1 (-1) * r^(m-1) :=
by
  sorry

end geometric_sequence_when_k_is_neg_one_l161_161314


namespace total_cost_of_items_l161_161044

-- Definitions based on conditions in a)
def price_of_caramel : ℕ := 3
def price_of_candy_bar : ℕ := 2 * price_of_caramel
def price_of_cotton_candy : ℕ := (4 * price_of_candy_bar) / 2
def cost_of_6_candy_bars : ℕ := 6 * price_of_candy_bar
def cost_of_3_caramels : ℕ := 3 * price_of_caramel

-- Problem statement to be proved
theorem total_cost_of_items : cost_of_6_candy_bars + cost_of_3_caramels + price_of_cotton_candy = 57 :=
by
  sorry

end total_cost_of_items_l161_161044


namespace prove_equations_and_PA_PB_l161_161363

noncomputable def curve_C1_parametric (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, 1 + t * Real.sin α)

noncomputable def curve_C2_polar (ρ θ : ℝ) : Prop :=
  ρ + 7 / ρ = 4 * Real.cos θ + 4 * Real.sin θ

theorem prove_equations_and_PA_PB :
  (∀ (α : ℝ), 0 ≤ α ∧ α < π → 
    (∃ (C1_cart : ℝ → ℝ → Prop), ∀ x y, C1_cart x y ↔ x^2 = 4 * y) ∧
    (∃ (C1_polar : ℝ → ℝ → Prop), ∀ ρ θ, C1_polar ρ θ ↔ ρ^2 * Real.cos θ^2 = 4 * ρ * Real.sin θ) ∧
    (∃ (C2_cart : ℝ → ℝ → Prop), ∀ x y, C2_cart x y ↔ (x - 2)^2 + (y - 2)^2 = 1)) ∧
  (∃ (P A B : ℝ × ℝ), P = (0, 1) ∧ 
    curve_C1_parametric t (Real.pi / 2) = A ∧ 
    curve_C1_parametric t (Real.pi / 2) = B ∧ 
    |P - A| * |P - B| = 4) :=
sorry

end prove_equations_and_PA_PB_l161_161363


namespace manager_monthly_salary_l161_161389

theorem manager_monthly_salary (average_salary_20 : ℝ) (new_average_salary_21 : ℝ) (m : ℝ) 
  (h1 : average_salary_20 = 1300) 
  (h2 : new_average_salary_21 = 1400) 
  (h3 : 20 * average_salary_20 + m = 21 * new_average_salary_21) : 
  m = 3400 := 
by 
  -- Proof is omitted
  sorry

end manager_monthly_salary_l161_161389


namespace find_prime_p_l161_161407

theorem find_prime_p (p : ℕ) (hp : Nat.Prime p) (hp_plus_10 : Nat.Prime (p + 10)) (hp_plus_14 : Nat.Prime (p + 14)) : p = 3 := 
sorry

end find_prime_p_l161_161407


namespace miles_ridden_further_l161_161874

theorem miles_ridden_further (distance_ridden distance_walked : ℝ) (h1 : distance_ridden = 3.83) (h2 : distance_walked = 0.17) :
  distance_ridden - distance_walked = 3.66 := 
by sorry

end miles_ridden_further_l161_161874


namespace range_of_a_l161_161444

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 1/2 → x^2 + a*x + 1 ≥ 0) ↔ a ≥ -5/2 :=
by 
  sorry

end range_of_a_l161_161444


namespace range_of_ab_l161_161166

def circle_eq (x y : ℝ) := x^2 + y^2 + 2 * x - 4 * y + 1 = 0
def line_eq (a b x y : ℝ) := 2 * a * x - b * y + 2 = 0

theorem range_of_ab (a b : ℝ) :
  (∃ x y : ℝ, circle_eq x y ∧ line_eq a b x y) ∧ (∃ x y : ℝ, x = -1 ∧ y = 2) →
  ab <= 1/4 := 
by
  sorry

end range_of_ab_l161_161166


namespace boat_stream_ratio_l161_161157

-- Conditions: A man takes twice as long to row a distance against the stream as to row the same distance in favor of the stream.
theorem boat_stream_ratio (B S : ℝ) (h : ∀ (d : ℝ), d / (B - S) = 2 * (d / (B + S))) : B / S = 3 :=
by
  sorry

end boat_stream_ratio_l161_161157


namespace square_perimeter_l161_161659

-- First, declare the side length of the square (rectangle)
variable (s : ℝ)

-- State the conditions: the area is 484 cm^2 and it's a square
axiom area_condition : s^2 = 484
axiom is_square : ∀ (s : ℝ), s > 0

-- Define the perimeter of the square
def perimeter (s : ℝ) : ℝ := 4 * s

-- State the theorem: perimeter == 88 given the conditions
theorem square_perimeter : perimeter s = 88 :=
by 
  -- Prove the statement given the axiom 'area_condition'
  sorry

end square_perimeter_l161_161659


namespace single_bill_value_l161_161049

theorem single_bill_value 
  (total_amount : ℕ) 
  (num_5_dollar_bills : ℕ) 
  (amount_5_dollar_bills : ℕ) 
  (single_bill : ℕ) : 
  total_amount = 45 → 
  num_5_dollar_bills = 7 → 
  amount_5_dollar_bills = 5 → 
  total_amount = num_5_dollar_bills * amount_5_dollar_bills + single_bill → 
  single_bill = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end single_bill_value_l161_161049


namespace electric_poles_count_l161_161486

theorem electric_poles_count (dist interval: ℕ) (h_interval: interval = 25) (h_dist: dist = 1500):
  (dist / interval) + 1 = 61 := 
by
  -- Sorry to skip the proof steps
  sorry

end electric_poles_count_l161_161486


namespace valid_triplets_l161_161067

theorem valid_triplets (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_leq1 : a ≤ b) (h_leq2 : b ≤ c)
  (h_div1 : a ∣ (b + c)) (h_div2 : b ∣ (a + c)) (h_div3 : c ∣ (a + b)) :
  (a = b ∧ b = c) ∨ (a = b ∧ c = 2 * a) ∨ (b = 2 * a ∧ c = 3 * a) :=
sorry

end valid_triplets_l161_161067


namespace marcus_baseball_cards_l161_161560

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the number of additional cards Marcus has compared to Carter
def additional_cards : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + additional_cards

-- The proof statement asserting Marcus' total number of baseball cards
theorem marcus_baseball_cards : marcus_cards = 210 :=
by {
  -- This is where the proof steps would go, but we are skipping with sorry
  sorry
}

end marcus_baseball_cards_l161_161560


namespace sum_of_extreme_numbers_is_846_l161_161704

theorem sum_of_extreme_numbers_is_846 :
  let digits := [0, 2, 4, 6]
  let is_valid_hundreds_digit (d : Nat) := d ≠ 0
  let create_three_digit_number (h t u : Nat) := h * 100 + t * 10 + u
  let max_num := create_three_digit_number 6 4 2
  let min_num := create_three_digit_number 2 0 4
  max_num + min_num = 846 := by
  sorry

end sum_of_extreme_numbers_is_846_l161_161704


namespace hall_paving_l161_161158

theorem hall_paving :
  ∀ (hall_length hall_breadth stone_length stone_breadth : ℕ),
    hall_length = 72 →
    hall_breadth = 30 →
    stone_length = 8 →
    stone_breadth = 10 →
    let Area_hall := hall_length * hall_breadth
    let Length_stone := stone_length / 10
    let Breadth_stone := stone_breadth / 10
    let Area_stone := Length_stone * Breadth_stone 
    (Area_hall / Area_stone) = 2700 :=
by
  intros hall_length hall_breadth stone_length stone_breadth
  intro h1 h2 h3 h4
  let Area_hall := hall_length * hall_breadth
  let Length_stone := stone_length / 10
  let Breadth_stone := stone_breadth / 10
  let Area_stone := Length_stone * Breadth_stone 
  have h5 : Area_hall / Area_stone = 2700 := sorry
  exact h5

end hall_paving_l161_161158


namespace batsman_average_after_12th_inning_l161_161883

theorem batsman_average_after_12th_inning (average_initial : ℕ) (score_12th : ℕ) (average_increase : ℕ) (total_innings : ℕ) 
    (h_avg_init : average_initial = 29) (h_score_12th : score_12th = 65) (h_avg_inc : average_increase = 3) 
    (h_total_innings : total_innings = 12) : 
    (average_initial + average_increase = 32) := 
by
  sorry

end batsman_average_after_12th_inning_l161_161883


namespace find_n_equiv_l161_161767

theorem find_n_equiv :
  ∃ (n : ℕ), 3 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [MOD 6] ∧ (n = 3 ∨ n = 9) :=
by
  sorry

end find_n_equiv_l161_161767


namespace solve_for_x_l161_161787

theorem solve_for_x (x y z : ℕ) 
  (h1 : 3^x * 4^y / 2^z = 59049)
  (h2 : x - y + 2 * z = 10) : 
  x = 10 :=
sorry

end solve_for_x_l161_161787


namespace OHara_triple_example_l161_161265

def is_OHara_triple (a b x : ℕ) : Prop :=
  (Real.sqrt a + Real.sqrt b = x)

theorem OHara_triple_example : is_OHara_triple 36 25 11 :=
by {
  sorry
}

end OHara_triple_example_l161_161265


namespace find_k_l161_161237

-- Define the vectors and the condition that k · a + b is perpendicular to a
theorem find_k 
  (a : ℝ × ℝ) (b : ℝ × ℝ) (k : ℝ)
  (h_a : a = (1, 2))
  (h_b : b = (-2, 0))
  (h_perpendicular : ∀ (k : ℝ), (k * a.1 + b.1, k * a.2 + b.2) • a = 0 ) : k = 2 / 5 :=
sorry

end find_k_l161_161237


namespace max_marks_l161_161744

theorem max_marks (M : ℝ) (h1 : 0.40 * M = 200) : M = 500 := by
  sorry

end max_marks_l161_161744


namespace a7_b7_equals_29_l161_161869

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry

def cond1 := a + b = 1
def cond2 := a^2 + b^2 = 3
def cond3 := a^3 + b^3 = 4
def cond4 := a^4 + b^4 = 7
def cond5 := a^5 + b^5 = 11

theorem a7_b7_equals_29 : cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ cond5 → a^7 + b^7 = 29 :=
by
  sorry

end a7_b7_equals_29_l161_161869


namespace sin_double_angle_l161_161984

theorem sin_double_angle (α : ℝ) (h : Real.cos (Real.pi / 4 - α) = Real.sqrt 2 / 4) :
  Real.sin (2 * α) = -3 / 4 :=
sorry

end sin_double_angle_l161_161984


namespace tank_full_volume_l161_161130

theorem tank_full_volume (x : ℝ) (h1 : 5 / 6 * x > 0) (h2 : 5 / 6 * x - 15 = 1 / 3 * x) : x = 30 :=
by
  -- The proof is omitted as per the requirement.
  sorry

end tank_full_volume_l161_161130


namespace largest_root_is_sqrt6_l161_161216

theorem largest_root_is_sqrt6 (p q r : ℝ) 
  (h1 : p + q + r = 3) 
  (h2 : p * q + p * r + q * r = -6) 
  (h3 : p * q * r = -18) : 
  max p (max q r) = Real.sqrt 6 := 
sorry

end largest_root_is_sqrt6_l161_161216


namespace solve_inequality_l161_161829

theorem solve_inequality (x : ℝ) : (1 ≤ |x + 3| ∧ |x + 3| ≤ 4) ↔ (-7 ≤ x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_l161_161829


namespace cone_volume_with_same_radius_and_height_l161_161801

theorem cone_volume_with_same_radius_and_height (r h : ℝ) 
  (Vcylinder : ℝ) (Vcone : ℝ) (h1 : Vcylinder = 54 * Real.pi) 
  (h2 : Vcone = (1 / 3) * Vcylinder) : Vcone = 18 * Real.pi :=
by sorry

end cone_volume_with_same_radius_and_height_l161_161801


namespace decreasing_intervals_l161_161124

noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x + 1)

theorem decreasing_intervals : 
  (∀ x y : ℝ, x < y → ((y < -1 ∨ x > -1) → f y < f x)) ∧
  (∀ x y : ℝ, x < y → (y ≥ -1 ∧ x ≤ -1 → f y < f x)) :=
by 
  intros;
  sorry

end decreasing_intervals_l161_161124


namespace intersection_of_sets_l161_161224

def M : Set ℝ := { x | 3 * x - 6 ≥ 0 }
def N : Set ℝ := { x | x^2 < 16 }

theorem intersection_of_sets : M ∩ N = { x | 2 ≤ x ∧ x < 4 } :=
by {
  sorry
}

end intersection_of_sets_l161_161224


namespace coda_password_combinations_l161_161876

open BigOperators

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 
  ∨ n = 23 ∨ n = 29

def is_power_of_two (n : ℕ) : Prop :=
  n = 2 ∨ n = 4 ∨ n = 8 ∨ n = 16

def is_multiple_of_three (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n ≥ 1 ∧ n ≤ 30

def count_primes : ℕ :=
  10
def count_powers_of_two : ℕ :=
  4
def count_multiples_of_three : ℕ :=
  10

theorem coda_password_combinations : count_primes * count_powers_of_two * count_multiples_of_three = 400 := by
  sorry

end coda_password_combinations_l161_161876


namespace li_li_age_this_year_l161_161647

theorem li_li_age_this_year (A B : ℕ) (h1 : A + B = 30) (h2 : A = B + 6) : B = 12 := by
  sorry

end li_li_age_this_year_l161_161647


namespace largest_divisor_poly_l161_161217

-- Define the polynomial and the required properties
def poly (n : ℕ) : ℕ := (n+1) * (n+3) * (n+5) * (n+7) * (n+11)

-- Define the conditions and the proof statement
theorem largest_divisor_poly (n : ℕ) (h_even : n % 2 = 0) : ∃ d, d = 15 ∧ ∀ m, m ∣ poly n → m ≤ d :=
by
  sorry

end largest_divisor_poly_l161_161217


namespace monotonic_increasing_interval_l161_161570

def f (x : ℝ) : ℝ := x^2 - 2

theorem monotonic_increasing_interval :
  ∀ x y: ℝ, 0 <= x -> x <= y -> f x <= f y := 
by
  -- proof would be here
  sorry

end monotonic_increasing_interval_l161_161570


namespace percent_area_contained_l161_161777

-- Define the conditions as Lean definitions
def side_length_square (s : ℝ) : ℝ := s
def width_rectangle (s : ℝ) : ℝ := 2 * s
def length_rectangle (s : ℝ) : ℝ := 3 * (width_rectangle s)

-- Define areas based on definitions
def area_square (s : ℝ) : ℝ := (side_length_square s) ^ 2
def area_rectangle (s : ℝ) : ℝ := (length_rectangle s) * (width_rectangle s)

-- The main theorem stating the percentage of the rectangle's area contained within the square
theorem percent_area_contained (s : ℝ) (h : s ≠ 0) :
  (area_square s / area_rectangle s) * 100 = 8.33 := by
  sorry

end percent_area_contained_l161_161777


namespace total_students_correct_l161_161394

def third_grade_students := 203
def fourth_grade_students := third_grade_students + 125
def total_students := third_grade_students + fourth_grade_students

theorem total_students_correct :
  total_students = 531 :=
by
  -- We state that the total number of students is 531
  sorry

end total_students_correct_l161_161394


namespace max_area_angle_A_l161_161310

open Real

theorem max_area_angle_A (A B C : ℝ) (tan_A tan_B : ℝ) :
  tan A * tan B = 1 ∧ AB = sqrt 3 → 
  (∃ A, A = π / 4 ∧ area_maximized)
  :=
by sorry

end max_area_angle_A_l161_161310


namespace inequality_solution_l161_161014

theorem inequality_solution (x : ℝ) : (3 * x^2 - 4 * x - 4 < 0) ↔ (-2/3 < x ∧ x < 2) :=
sorry

end inequality_solution_l161_161014


namespace calculation_correct_l161_161031

theorem calculation_correct : (18 / (3 + 9 - 6)) * 4 = 12 :=
by
  sorry

end calculation_correct_l161_161031


namespace irreducible_fraction_l161_161365

theorem irreducible_fraction (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := 
by sorry

end irreducible_fraction_l161_161365


namespace false_statement_l161_161603

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x

def p : Prop := ∃ x0 : ℝ, f x0 = -1
def q : Prop := ∀ x : ℝ, f (2 * Real.pi + x) = f x

theorem false_statement : ¬ (p ∧ q) := sorry

end false_statement_l161_161603


namespace no_integer_pair_2006_l161_161554

theorem no_integer_pair_2006 : ∀ (x y : ℤ), x^2 - y^2 ≠ 2006 := by
  sorry

end no_integer_pair_2006_l161_161554


namespace problem_I_problem_II_l161_161471

-- Definitions of sets A and B
def A : Set ℝ := { x : ℝ | -1 < x ∧ x < 3 }
def B (a b : ℝ) : Set ℝ := { x : ℝ | x^2 - a * x + b < 0 }

-- Problem (I)
theorem problem_I (a b : ℝ) (h : A = B a b) : a = 2 ∧ b = -3 := 
sorry

-- Problem (II)
theorem problem_II (a : ℝ) (h₁ : ∀ x, (x ∈ A ∧ x ∈ B a 3) → x ∈ B a 3) : -2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 := 
sorry


end problem_I_problem_II_l161_161471


namespace B_speaks_truth_60_l161_161305

variable (P_A P_B P_A_and_B : ℝ)

-- Given conditions
def A_speaks_truth_85 : Prop := P_A = 0.85
def both_speak_truth_051 : Prop := P_A_and_B = 0.51

-- Solution condition
noncomputable def B_speaks_truth_percentage : ℝ := P_A_and_B / P_A

-- Statement to prove
theorem B_speaks_truth_60 (hA : A_speaks_truth_85 P_A) (hAB : both_speak_truth_051 P_A_and_B) : B_speaks_truth_percentage P_A_and_B P_A = 0.6 :=
by
  rw [A_speaks_truth_85] at hA
  rw [both_speak_truth_051] at hAB
  unfold B_speaks_truth_percentage
  sorry

end B_speaks_truth_60_l161_161305


namespace min_tiles_for_square_l161_161699

theorem min_tiles_for_square (a b : ℕ) (ha : a = 6) (hb : b = 4) (harea_tile : a * b = 24)
  (h_lcm : Nat.lcm a b = 12) : 
  let area_square := (Nat.lcm a b) * (Nat.lcm a b) 
  let num_tiles_required := area_square / (a * b)
  num_tiles_required = 6 :=
by
  sorry

end min_tiles_for_square_l161_161699


namespace largest_four_digit_perfect_cube_is_9261_l161_161072

-- Define the notion of a four-digit number and perfect cube
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

-- The main theorem statement
theorem largest_four_digit_perfect_cube_is_9261 :
  ∃ n, is_four_digit n ∧ is_perfect_cube n ∧ (∀ m, is_four_digit m ∧ is_perfect_cube m → m ≤ n) ∧ n = 9261 :=
sorry -- Proof is omitted

end largest_four_digit_perfect_cube_is_9261_l161_161072


namespace modulus_remainder_l161_161329

theorem modulus_remainder (n : ℕ) 
  (h1 : n^3 % 7 = 3) 
  (h2 : n^4 % 7 = 2) : 
  n % 7 = 6 :=
by
  sorry

end modulus_remainder_l161_161329


namespace Elberta_has_23_dollars_l161_161718

theorem Elberta_has_23_dollars :
  let granny_smith_amount := 63
  let anjou_amount := 1 / 3 * granny_smith_amount
  let elberta_amount := anjou_amount + 2
  elberta_amount = 23 := by
  sorry

end Elberta_has_23_dollars_l161_161718


namespace number_of_valid_triples_l161_161221

theorem number_of_valid_triples : 
  ∃ n, n = 7 ∧ ∀ (a b c : ℕ), b = 2023 → a ≤ b → b ≤ c → a * c = 2023^2 → (n = 7) :=
by 
  sorry

end number_of_valid_triples_l161_161221


namespace max_product_of_sum_300_l161_161077

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l161_161077


namespace ratio_of_radii_l161_161741

theorem ratio_of_radii (r R : ℝ) (hR : R > 0) (hr : r > 0)
  (h : π * R^2 - π * r^2 = 4 * (π * r^2)) : r / R = 1 / Real.sqrt 5 :=
by
  sorry

end ratio_of_radii_l161_161741


namespace susan_typing_time_l161_161229

theorem susan_typing_time :
  let Jonathan_rate := 1 -- page per minute
  let Jack_rate := 5 / 3 -- pages per minute
  let combined_rate := 4 -- pages per minute
  ∃ S : ℝ, (1 + 1/S + 5/3 = 4) → S = 30 :=
by
  sorry

end susan_typing_time_l161_161229


namespace base_7_units_digit_l161_161306

theorem base_7_units_digit (a : ℕ) (b : ℕ) (h₁ : a = 326) (h₂ : b = 57) : ((a * b) % 7) = 4 := by
  sorry

end base_7_units_digit_l161_161306


namespace sara_quarters_l161_161817

-- Conditions
def usd_to_eur (usd : ℝ) : ℝ := usd * 0.85
def eur_to_usd (eur : ℝ) : ℝ := eur * 1.15
def value_of_quarter_usd : ℝ := 0.25
def dozen : ℕ := 12

-- Theorem
theorem sara_quarters (sara_savings_usd : ℝ) (usd_to_eur_ratio : ℝ) (eur_to_usd_ratio : ℝ) (quarter_value_usd : ℝ) (doz : ℕ) : sara_savings_usd = 9 → usd_to_eur_ratio = 0.85 → eur_to_usd_ratio = 1.15 → quarter_value_usd = 0.25 → doz = 12 → 
  ∃ dozens : ℕ, dozens = 2 :=
by
  sorry

end sara_quarters_l161_161817


namespace complement_of_A_relative_to_U_l161_161179

-- Define the universal set U and set A
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 4, 5}

-- Define the proof statement for the complement of A with respect to U
theorem complement_of_A_relative_to_U : (U \ A) = {2} := by
  sorry

end complement_of_A_relative_to_U_l161_161179


namespace days_before_reinforcement_l161_161515

/-- A garrison of 2000 men originally has provisions for 62 days.
    After some days, a reinforcement of 2700 men arrives.
    The provisions are found to last for only 20 days more after the reinforcement arrives.
    Prove that the number of days passed before the reinforcement arrived is 15. -/
theorem days_before_reinforcement 
  (x : ℕ) 
  (num_men_orig : ℕ := 2000) 
  (num_men_reinf : ℕ := 2700) 
  (days_orig : ℕ := 62) 
  (days_after_reinf : ℕ := 20) 
  (total_provisions : ℕ := num_men_orig * days_orig)
  (remaining_provisions : ℕ := num_men_orig * (days_orig - x))
  (consumption_after_reinf : ℕ := (num_men_orig + num_men_reinf) * days_after_reinf) 
  (provisions_eq : remaining_provisions = consumption_after_reinf) : 
  x = 15 := 
by 
  sorry

end days_before_reinforcement_l161_161515


namespace train_length_l161_161170

theorem train_length (L : ℕ) (speed : ℕ) 
  (h1 : L + 1200 = speed * 45) 
  (h2 : L + 180 = speed * 15) : 
  L = 330 := 
sorry

end train_length_l161_161170


namespace opposite_of_two_is_negative_two_l161_161913

theorem opposite_of_two_is_negative_two : -2 = -2 :=
by
  sorry

end opposite_of_two_is_negative_two_l161_161913


namespace find_time_same_height_l161_161973

noncomputable def height_ball (t : ℝ) : ℝ := 60 - 9 * t - 8 * t^2
noncomputable def height_bird (t : ℝ) : ℝ := 3 * t^2 + 4 * t

theorem find_time_same_height : ∃ t : ℝ, t = 20 / 11 ∧ height_ball t = height_bird t := 
by
  use 20 / 11
  sorry

end find_time_same_height_l161_161973


namespace find_m_l161_161726

theorem find_m (m : ℕ) (h : 8 ^ 36 * 6 ^ 21 = 3 * 24 ^ m) : m = 43 :=
sorry

end find_m_l161_161726


namespace remainder_4015_div_32_l161_161131

theorem remainder_4015_div_32 : 4015 % 32 = 15 := by
  sorry

end remainder_4015_div_32_l161_161131


namespace steve_pencils_left_l161_161442

-- Conditions
def initial_pencils : Nat := 24
def pencils_given_to_Lauren : Nat := 6
def extra_pencils_given_to_Matt : Nat := 3

-- Question: How many pencils does Steve have left?
theorem steve_pencils_left :
  initial_pencils - (pencils_given_to_Lauren + (pencils_given_to_Lauren + extra_pencils_given_to_Matt)) = 9 := by
  -- You need to provide a proof here
  sorry

end steve_pencils_left_l161_161442


namespace find_interest_rate_l161_161532

-- Translating the identified conditions into Lean definitions
def initial_deposit (P : ℝ) : Prop := P > 0
def compounded_semiannually (n : ℕ) : Prop := n = 2
def growth_in_sum (A : ℝ) (P : ℝ) : Prop := A = 1.1592740743 * P
def time_period (t : ℝ) : Prop := t = 2.5

theorem find_interest_rate (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) (A : ℝ)
  (h_init : initial_deposit P)
  (h_n : compounded_semiannually n)
  (h_A : growth_in_sum A P)
  (h_t : time_period t) :
  r = 0.06 :=
by
  sorry

end find_interest_rate_l161_161532


namespace min_value_ab_sum_l161_161623

theorem min_value_ab_sum (a b : ℤ) (h : a * b = 100) : a + b ≥ -101 :=
  sorry

end min_value_ab_sum_l161_161623


namespace ab_product_l161_161758

theorem ab_product (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (2 * a + b) * (2 * b + a) = 4752) : a * b = 520 := 
by
  sorry

end ab_product_l161_161758


namespace correct_result_after_mistakes_l161_161974

theorem correct_result_after_mistakes (n : ℕ) (f : ℕ → ℕ → ℕ) (g : ℕ → ℕ → ℕ)
    (h1 : f n 4 * 4 + 18 = g 12 18) : 
    g (f n 4 * 4) 18 = 498 :=
by
  sorry

end correct_result_after_mistakes_l161_161974


namespace inequality_reciprocal_l161_161143

theorem inequality_reciprocal (a b : ℝ) (h₀ : a < b) (h₁ : b < 0) : (1 / a) > (1 / b) :=
sorry

end inequality_reciprocal_l161_161143


namespace number_is_7612_l161_161214

-- Definitions of the conditions
def digits_correct_wrong_positions (guess : Nat) (num_correct : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10, 
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → 
      (num_correct = 2 ∧
      (digits_placed 0 ≠ (guess / 1000) % 10 ∧ 
      digits_placed 1 ≠ (guess / 100) % 10 ∧ 
      digits_placed 2 ≠ (guess / 10) % 10 ∧ 
      digits_placed 3 ≠ guess % 10)))

def digits_correct_positions (guess : Nat) (num_correct : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10,
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → 
      (num_correct = 2 ∧
      (digits_placed 0 = (guess / 1000) % 10 ∨ 
      digits_placed 1 = (guess / 100) % 10 ∨ 
      digits_placed 2 = (guess / 10) % 10 ∨ 
      digits_placed 3 = guess % 10)))

def digits_not_correct (guess : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10,
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → False)

-- The main theorem to prove
theorem number_is_7612 :
  digits_correct_wrong_positions 8765 2 ∧
  digits_correct_wrong_positions 1023 2 ∧
  digits_correct_positions 8642 2 ∧
  digits_not_correct 5430 →
  ∃ (num : Nat), 
    (num / 1000) % 10 = 7 ∧
    (num / 100) % 10 = 6 ∧
    (num / 10) % 10 = 1 ∧
    num % 10 = 2 ∧
    num = 7612 :=
sorry

end number_is_7612_l161_161214


namespace inequality_change_l161_161875

theorem inequality_change (a b : ℝ) (h : a < b) : -2 * a > -2 * b :=
sorry

end inequality_change_l161_161875


namespace broken_perfect_spiral_shells_difference_l161_161304

theorem broken_perfect_spiral_shells_difference :
  let perfect_shells := 17
  let broken_shells := 52
  let broken_spiral_shells := broken_shells / 2
  let not_spiral_perfect_shells := 12
  let spiral_perfect_shells := perfect_shells - not_spiral_perfect_shells
  broken_spiral_shells - spiral_perfect_shells = 21 := by
  sorry

end broken_perfect_spiral_shells_difference_l161_161304


namespace area_per_car_l161_161681

/-- Given the length and width of the parking lot, 
and the percentage of usable area, 
and the number of cars that can be parked,
prove that the area per car is as expected. -/
theorem area_per_car 
  (length width : ℝ) 
  (usable_percentage : ℝ) 
  (number_of_cars : ℕ) 
  (h_length : length = 400) 
  (h_width : width = 500) 
  (h_usable_percentage : usable_percentage = 0.80) 
  (h_number_of_cars : number_of_cars = 16000) :
  (length * width * usable_percentage) / number_of_cars = 10 :=
by
  sorry

end area_per_car_l161_161681


namespace proof_problem_l161_161084

-- Definitions of arithmetic and geometric sequences
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b1 r : ℝ), ∀ n, b n = b1 * r^n

-- Lean statement of the problem
theorem proof_problem 
  (a b : ℕ → ℝ)
  (h_a_arithmetic : is_arithmetic_sequence a)
  (h_b_geometric : is_geometric_sequence b)
  (h_condition : a 1 - (a 7)^2 + a 13 = 0)
  (h_b7_a7 : b 7 = a 7) :
  b 3 * b 11 = 4 :=
sorry

end proof_problem_l161_161084


namespace spinner_points_east_l161_161774

-- Definitions for the conditions
def initial_direction := "north"

-- Clockwise and counterclockwise movements as improper fractions
def clockwise_move := (7 : ℚ) / 2
def counterclockwise_move := (17 : ℚ) / 4

-- Compute the net movement (negative means counterclockwise)
def net_movement := clockwise_move - counterclockwise_move

-- Translate net movement into a final direction (using modulo arithmetic with 1 revolution = 360 degrees equivalent)
def final_position : ℚ := (net_movement + 1) % 1

-- The goal is to prove that the final direction is east (which corresponds to 1/4 revolution)
theorem spinner_points_east :
  final_position = (1 / 4 : ℚ) :=
by
  sorry

end spinner_points_east_l161_161774


namespace find_triples_l161_161721

-- Definitions of the problem conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def satisfies_equation (a b p : ℕ) : Prop := a^p = factorial b + p

-- The main theorem statement based on the problem conditions
theorem find_triples :
  (satisfies_equation 2 2 2 ∧ is_prime 2) ∧
  (satisfies_equation 3 4 3 ∧ is_prime 3) ∧
  (∀ (a b p : ℕ), (satisfies_equation a b p ∧ is_prime p) → (a, b, p) = (2, 2, 2) ∨ (a, b, p) = (3, 4, 3)) :=
by
  -- Proof to be filled
  sorry

end find_triples_l161_161721


namespace intersection_x_coordinate_l161_161187

-- Definitions based on conditions
def line1 (x : ℝ) : ℝ := 3 * x + 5
def line2 (x : ℝ) : ℝ := 35 - 5 * x

-- Proof statement
theorem intersection_x_coordinate : ∃ x : ℝ, line1 x = line2 x ∧ x = 15 / 4 :=
by
  use 15 / 4
  sorry

end intersection_x_coordinate_l161_161187


namespace part1_part2_l161_161449

-- Definitions and conditions
def total_length : ℝ := 64
def ratio_larger_square_area : ℝ := 2.25
def total_area : ℝ := 160

-- Given problem parts
theorem part1 (x : ℝ) (h : (64 - 4 * x) / 4 * (64 - 4 * x) / 4 = 2.25 * x * x) : x = 6.4 :=
by
  -- Proof needs to be provided
  sorry

theorem part2 (y : ℝ) (h : (16 - y) * (16 - y) + y * y = 160) : y = 4 ∧ (64 - 4 * y) = 48 :=
by
  -- Proof needs to be provided
  sorry

end part1_part2_l161_161449


namespace pen_more_expensive_than_two_notebooks_l161_161028

variable (T R C : ℝ)

-- Conditions
axiom cond1 : T + R + C = 120
axiom cond2 : 5 * T + 2 * R + 3 * C = 350

-- Theorem statement
theorem pen_more_expensive_than_two_notebooks :
  R > 2 * T :=
by
  -- omit the actual proof, but check statement correctness
  sorry

end pen_more_expensive_than_two_notebooks_l161_161028


namespace geometric_sequence_ninth_term_l161_161446

-- Given conditions
variables (a r : ℝ)
axiom fifth_term_condition : a * r^4 = 80
axiom seventh_term_condition : a * r^6 = 320

-- Goal: Prove that the ninth term is 1280
theorem geometric_sequence_ninth_term : a * r^8 = 1280 :=
by
  sorry

end geometric_sequence_ninth_term_l161_161446


namespace max_four_color_rectangles_l161_161762

def color := Fin 4
def grid := Fin 100 × Fin 100
def colored_grid := grid → color

def count_four_color_rectangles (g : colored_grid) : ℕ := sorry

theorem max_four_color_rectangles (g : colored_grid) :
  count_four_color_rectangles g ≤ 9375000 := sorry

end max_four_color_rectangles_l161_161762


namespace sum_of_products_of_roots_eq_neg3_l161_161566

theorem sum_of_products_of_roots_eq_neg3 {p q r s : ℂ} 
  (h : ∀ {x : ℂ}, 4 * x^4 - 8 * x^3 + 12 * x^2 - 16 * x + 9 = 0 → (x = p ∨ x = q ∨ x = r ∨ x = s)) : 
  p * q + p * r + p * s + q * r + q * s + r * s = -3 := 
sorry

end sum_of_products_of_roots_eq_neg3_l161_161566


namespace range_of_a_l161_161970

-- Define the assumptions and target proof
theorem range_of_a {f : ℝ → ℝ}
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_monotonic : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h_condition : ∀ a : ℝ, f (2 - a) + f (4 - a) < 0)
  : ∀ a : ℝ, f (2 - a) + f (4 - a) < 0 → a < 3 :=
by
  intro a h
  sorry

end range_of_a_l161_161970


namespace number_line_move_l161_161977

theorem number_line_move (A B: ℤ):  A = -3 → B = A + 4 → B = 1 := by
  intros hA hB
  rw [hA] at hB
  rw [hB]
  sorry

end number_line_move_l161_161977


namespace unicorn_rope_length_l161_161997

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 1500
noncomputable def c : ℕ := 3

theorem unicorn_rope_length : a + b + c = 1593 :=
by
  -- The steps to prove the theorem should go here, but as stated, we skip this with "sorry".
  sorry

end unicorn_rope_length_l161_161997


namespace area_of_square_l161_161797

theorem area_of_square (r s l : ℕ) (h1 : l = (2 * r) / 5) (h2 : r = s) (h3 : l * 10 = 240) : s * s = 3600 :=
by
  sorry

end area_of_square_l161_161797


namespace geometric_series_sum_l161_161320

variable (a r : ℤ) (n : ℕ) 

theorem geometric_series_sum :
  a = -1 ∧ r = 2 ∧ n = 10 →
  (a * (r^n - 1) / (r - 1)) = -1023 := 
by
  intro h
  rcases h with ⟨ha, hr, hn⟩
  sorry

end geometric_series_sum_l161_161320


namespace probability_twice_correct_l161_161535

noncomputable def probability_at_least_twice (x y : ℝ) : ℝ :=
if (0 ≤ x ∧ x ≤ 1000) ∧ (0 ≤ y ∧ y ≤ 3000) then
  if y ≥ 2*x then (1/6 : ℝ) else 0
else 0

theorem probability_twice_correct : probability_at_least_twice 500 1000 = (1/6 : ℝ) :=
sorry

end probability_twice_correct_l161_161535


namespace power_sum_l161_161712

theorem power_sum (a b c : ℝ) (h1 : a + b + c = 1)
                  (h2 : a^2 + b^2 + c^2 = 3)
                  (h3 : a^3 + b^3 + c^3 = 4)
                  (h4 : a^4 + b^4 + c^4 = 5) :
  a^5 + b^5 + c^5 = 6 :=
  sorry

end power_sum_l161_161712


namespace solve_quadratic_l161_161635

theorem solve_quadratic :
    ∀ x : ℝ, x^2 - 6*x + 8 = 0 ↔ (x = 2 ∨ x = 4) :=
by
  intros x
  sorry

end solve_quadratic_l161_161635


namespace find_XY_in_triangle_l161_161032

-- Definitions
def Triangle := Type
def angle_measures (T : Triangle) : (ℕ × ℕ × ℕ) := sorry
def side_lengths (T : Triangle) : (ℕ × ℕ × ℕ) := sorry
def is_30_60_90_triangle (T : Triangle) : Prop := (angle_measures T = (30, 60, 90))

-- Given conditions and statement we want to prove
def triangle_XYZ : Triangle := sorry
def XY : ℕ := 6

-- Proof statement
theorem find_XY_in_triangle :
  is_30_60_90_triangle triangle_XYZ ∧ (side_lengths triangle_XYZ).1 = XY →
  XY = 6 :=
by
  intro h
  sorry

end find_XY_in_triangle_l161_161032


namespace distance_sum_identity_l161_161091

noncomputable def squared_distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem distance_sum_identity
  (a b c x y : ℝ)
  (A B C P G : ℝ × ℝ)
  (hA : A = (a, b))
  (hB : B = (-c, 0))
  (hC : C = (c, 0))
  (hG : G = (a / 3, b / 3))
  (hP : P = (x, y))
  (hG_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) :
  squared_distance A P + squared_distance B P + squared_distance C P =
  squared_distance A G + squared_distance B G + squared_distance C G + 3 * squared_distance G P :=
by sorry

end distance_sum_identity_l161_161091


namespace Tim_pays_correct_amount_l161_161619

def pays_in_a_week (hourly_rate : ℕ) (num_bodyguards : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hourly_rate * num_bodyguards * hours_per_day * days_per_week

theorem Tim_pays_correct_amount :
  pays_in_a_week 20 2 8 7 = 2240 := by
  sorry

end Tim_pays_correct_amount_l161_161619


namespace total_stars_correct_l161_161425

-- Define the number of gold stars Shelby earned each day
def monday_stars : ℕ := 4
def tuesday_stars : ℕ := 7
def wednesday_stars : ℕ := 3
def thursday_stars : ℕ := 8
def friday_stars : ℕ := 2

-- Define the total number of gold stars
def total_stars : ℕ := monday_stars + tuesday_stars + wednesday_stars + thursday_stars + friday_stars

-- Prove that the total number of gold stars Shelby earned throughout the week is 24
theorem total_stars_correct : total_stars = 24 :=
by
  -- The proof goes here, using sorry to skip the proof
  sorry

end total_stars_correct_l161_161425


namespace cost_prices_max_profit_l161_161015

theorem cost_prices (a b : ℝ) (x : ℝ) (y : ℝ)
    (h1 : a - b = 500)
    (h2 : 40000 / a = 30000 / b)
    (h3 : 0 ≤ x ∧ x ≤ 20)
    (h4 : 2000 * x + 1500 * (20 - x) ≤ 36000) :
    a = 2000 ∧ b = 1500 := sorry

theorem max_profit (x : ℝ) (y : ℝ)
    (h1 : 0 ≤ x ∧ x ≤ 12) :
    y = 200 * x + 6000 ∧ y ≤ 8400 := sorry

end cost_prices_max_profit_l161_161015


namespace range_of_a_l161_161809

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 3) : -2 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l161_161809


namespace train_speed_l161_161650

theorem train_speed (length_m : ℝ) (time_s : ℝ) 
  (h1 : length_m = 120) 
  (h2 : time_s = 3.569962336897346) 
  : (length_m / 1000) / (time_s / 3600) = 121.003 :=
by
  sorry

end train_speed_l161_161650


namespace prove_inequalities_l161_161037

variable {a b c R r_a r_b r_c : ℝ}

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def has_circumradius (a b c R : ℝ) : Prop :=
  ∃ S : ℝ, S = a * b * c / (4 * R)

def has_exradii (a b c r_a r_b r_c : ℝ) : Prop :=
  ∃ S : ℝ, 
    r_a = 2 * S / (b + c - a) ∧
    r_b = 2 * S / (a + c - b) ∧
    r_c = 2 * S / (a + b - c)

theorem prove_inequalities
  (h_triangle : is_triangle a b c)
  (h_circumradius : has_circumradius a b c R)
  (h_exradii : has_exradii a b c r_a r_b r_c)
  (h_two_R_le_r_a : 2 * R ≤ r_a) :
  a > b ∧ a > c ∧ 2 * R > r_b ∧ 2 * R > r_c := 
sorry

end prove_inequalities_l161_161037


namespace max_profit_l161_161474

noncomputable def initial_cost : ℝ := 10
noncomputable def cost_per_pot : ℝ := 0.0027
noncomputable def total_cost (x : ℝ) : ℝ := initial_cost + cost_per_pot * x

noncomputable def P (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 5.7 * x + 19
else 108 - 1000 / (3 * x)

noncomputable def r (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 3 * x + 9
else 98 - 1000 / (3 * x) - 27 * x / 10

theorem max_profit (x : ℝ) : r 10 = 39 :=
sorry

end max_profit_l161_161474


namespace Wilsons_number_l161_161963

theorem Wilsons_number (N : ℝ) (h : N - N / 3 = 16 / 3) : N = 8 := sorry

end Wilsons_number_l161_161963


namespace expand_product_l161_161652

theorem expand_product : ∀ (x : ℝ), (3 * x - 4) * (2 * x + 9) = 6 * x^2 + 19 * x - 36 :=
by
  intro x
  sorry

end expand_product_l161_161652


namespace table_capacity_l161_161858

def invited_people : Nat := 18
def no_show_people : Nat := 12
def number_of_tables : Nat := 2
def attendees := invited_people - no_show_people
def people_per_table : Nat := attendees / number_of_tables

theorem table_capacity : people_per_table = 3 :=
by
  sorry

end table_capacity_l161_161858


namespace find_x_from_conditions_l161_161556

theorem find_x_from_conditions (x y : ℝ)
  (h1 : (6 : ℝ) = (1 / 2 : ℝ) * x)
  (h2 : y = (1 / 2 :ℝ) * 10)
  (h3 : x * y = 60) : x = 12 := by
  sorry

end find_x_from_conditions_l161_161556


namespace enclosed_area_is_correct_l161_161200

noncomputable def area_between_curves : ℝ := 
  let circle (x y : ℝ) := x^2 + y^2 = 4
  let cubic_parabola (x : ℝ) := - 1 / 2 * x^3 + 2 * x
  let x1 : ℝ := -2
  let x2 : ℝ := Real.sqrt 2
  -- Properly calculate the area between the two curves
  sorry

theorem enclosed_area_is_correct :
  area_between_curves = 3 * ( Real.pi + 1 ) / 2 :=
sorry

end enclosed_area_is_correct_l161_161200


namespace product_base9_conversion_l161_161056

noncomputable def base_9_to_base_10 (n : ℕ) : ℕ :=
match n with
| 237 => 2 * 9^2 + 3 * 9^1 + 7
| 17 => 9 + 7
| _ => 0

noncomputable def base_10_to_base_9 (n : ℕ) : ℕ :=
match n with
-- Step of conversion from example: 3136 => 4*9^3 + 2*9^2 + 6*9^1 + 4*9^0
| 3136 => 4 * 1000 + 2 * 100 + 6 * 10 + 4 -- representing 4264 in base 9
| _ => 0

theorem product_base9_conversion :
  base_10_to_base_9 ((base_9_to_base_10 237) * (base_9_to_base_10 17)) = 4264 := by
  sorry

end product_base9_conversion_l161_161056


namespace negation_exists_l161_161074

theorem negation_exists (h : ∀ x : ℝ, 0 < x → Real.sin x < x) : ∃ x : ℝ, 0 < x ∧ Real.sin x ≥ x :=
by
  sorry

end negation_exists_l161_161074


namespace probability_three_even_l161_161559

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the probability of exactly three dice showing an even number
noncomputable def prob_exactly_three_even (n : ℕ) (k : ℕ) (p : ℚ) : ℚ := 
  (binomial n k : ℚ) * (p^k) * ((1 - p)^(n - k))

-- The main theorem stating the desired probability
theorem probability_three_even (n : ℕ) (p : ℚ) (k : ℕ) (h₁ : n = 6) (h₂ : p = 1/2) (h₃ : k = 3) :
  prob_exactly_three_even n k p = 5 / 16 := by
  sorry

-- Include required definitions and expected values for the theorem
#check binomial
#check prob_exactly_three_even
#check probability_three_even

end probability_three_even_l161_161559


namespace dutch_exam_problem_l161_161325

theorem dutch_exam_problem (a b c d : ℝ) : 
  (a * b + c + d = 3) ∧ 
  (b * c + d + a = 5) ∧ 
  (c * d + a + b = 2) ∧ 
  (d * a + b + c = 6) → 
  (a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3) := 
by
  sorry

end dutch_exam_problem_l161_161325


namespace area_of_region_l161_161025

-- Definitions drawn from conditions
def circle_radius := 36
def num_small_circles := 8

-- Main statement to be proven
theorem area_of_region :
  ∃ K : ℝ, 
    K = π * (circle_radius ^ 2) - num_small_circles * π * ((circle_radius * (Real.sqrt 2 - 1)) ^ 2) ∧
    ⌊ K ⌋ = ⌊ π * (circle_radius ^ 2) - num_small_circles * π * ((circle_radius * (Real.sqrt 2 - 1)) ^ 2) ⌋ :=
  sorry

end area_of_region_l161_161025


namespace shooting_prob_l161_161824

theorem shooting_prob (p q : ℚ) (h: p + q = 1) (n : ℕ) 
  (cond1: p = 2/3) 
  (cond2: q = 1 - p) 
  (cond3: n = 5) : 
  (q ^ (n-1)) = 1/81 := 
by 
  sorry

end shooting_prob_l161_161824


namespace average_income_proof_l161_161024

theorem average_income_proof:
  ∀ (A B C : ℝ),
    (A + B) / 2 = 5050 →
    (B + C) / 2 = 6250 →
    A = 4000 →
    (A + C) / 2 = 5200 := by
  sorry

end average_income_proof_l161_161024


namespace walking_speed_of_A_l161_161415

-- Given conditions
def B_speed := 20 -- kmph
def start_delay := 10 -- hours
def distance_covered := 200 -- km

-- Prove A's walking speed
theorem walking_speed_of_A (v : ℝ) (time_A : ℝ) (time_B : ℝ) :
  distance_covered = v * time_A ∧ distance_covered = B_speed * time_B ∧ time_B = time_A - start_delay → v = 10 :=
by
  intro h
  sorry

end walking_speed_of_A_l161_161415


namespace coins_dimes_count_l161_161243

theorem coins_dimes_count :
  ∃ (p n d q : ℕ), 
    p + n + d + q = 10 ∧ 
    p + 5 * n + 10 * d + 25 * q = 110 ∧ 
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 2 ∧ d = 5 :=
by {
    sorry
}

end coins_dimes_count_l161_161243


namespace a_range_of_proposition_l161_161270

theorem a_range_of_proposition (a : ℝ) : (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 + 5 <= a * x) ↔ a ∈ Set.Ici (2 * Real.sqrt 5) := by
  sorry

end a_range_of_proposition_l161_161270


namespace initial_people_count_25_l161_161127

-- Definition of the initial number of people (X) and the condition
def initial_people (X : ℕ) : Prop := X - 8 + 13 = 30

-- The theorem stating that the initial number of people is 25
theorem initial_people_count_25 : ∃ (X : ℕ), initial_people X ∧ X = 25 :=
by
  -- We add sorry here to skip the actual proof
  sorry

end initial_people_count_25_l161_161127


namespace prime_and_n_eq_m_minus_1_l161_161263

theorem prime_and_n_eq_m_minus_1 (n m : ℕ) (h1 : n ≥ 2) (h2 : m ≥ 2)
  (h3 : ∀ k : ℕ, k ∈ Finset.range n.succ → k^n % m = 1) : Nat.Prime m ∧ n = m - 1 := 
sorry

end prime_and_n_eq_m_minus_1_l161_161263


namespace latest_start_time_l161_161860

-- Define the times for each activity
def homework_time : ℕ := 30
def clean_room_time : ℕ := 30
def take_out_trash_time : ℕ := 5
def empty_dishwasher_time : ℕ := 10
def dinner_time : ℕ := 45

-- Define the total time required to finish everything in minutes
def total_time_needed : ℕ := homework_time + clean_room_time + take_out_trash_time + empty_dishwasher_time + dinner_time

-- Define the equivalent time in hours
def total_time_needed_hours : ℕ := total_time_needed / 60

-- Define movie start time and the time Justin gets home
def movie_start_time : ℕ := 20 -- (8 PM in 24-hour format)
def justin_home_time : ℕ := 17 -- (5 PM in 24-hour format)

-- Prove the latest time Justin can start his chores and homework
theorem latest_start_time : movie_start_time - total_time_needed_hours = 18 := by
  sorry

end latest_start_time_l161_161860


namespace square_area_from_inscribed_circle_l161_161891

theorem square_area_from_inscribed_circle (r : ℝ) (π_pos : 0 < Real.pi) (circle_area : Real.pi * r^2 = 9 * Real.pi) : 
  (2 * r)^2 = 36 :=
by
  -- Proof goes here
  sorry

end square_area_from_inscribed_circle_l161_161891


namespace trigonometric_identity_l161_161457

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 2 * Real.cos α) :
  Real.sin (π / 2 + 2 * α) = -3 / 5 :=
by
  sorry

end trigonometric_identity_l161_161457


namespace margaret_time_correct_l161_161268

def margaret_time : ℕ :=
  let n := 7
  let r := 15
  (Nat.factorial n) / r

theorem margaret_time_correct : margaret_time = 336 := by
  sorry

end margaret_time_correct_l161_161268


namespace power_of_two_divisor_l161_161841

theorem power_of_two_divisor {n : ℕ} (h_pos : n > 0) : 
  (∃ m : ℤ, (2^n - 1) ∣ (m^2 + 9)) → ∃ r : ℕ, n = 2^r :=
by
  sorry

end power_of_two_divisor_l161_161841


namespace number_of_students_l161_161696

-- Define John's total winnings
def john_total_winnings : ℤ := 155250

-- Define the proportion of winnings given to each student
def proportion_per_student : ℚ := 1 / 1000

-- Define the total amount received by students
def total_received_by_students : ℚ := 15525

-- Calculate the amount each student received
def amount_per_student : ℚ := john_total_winnings * proportion_per_student

-- Theorem to prove the number of students
theorem number_of_students : total_received_by_students / amount_per_student = 100 :=
by
  -- Lean will be expected to fill in this proof
  sorry

end number_of_students_l161_161696


namespace tire_price_l161_161253

theorem tire_price (x : ℕ) (h : 4 * x + 5 = 485) : x = 120 :=
by
  sorry

end tire_price_l161_161253


namespace two_digit_number_l161_161736

theorem two_digit_number (a : ℕ) (N M : ℕ) :
  (10 ≤ a) ∧ (a ≤ 99) ∧ (2 * a + 1 = N^2) ∧ (3 * a + 1 = M^2) → a = 40 :=
by
  sorry

end two_digit_number_l161_161736


namespace odd_function_property_l161_161366

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Lean 4 statement of the problem
theorem odd_function_property (f : ℝ → ℝ) (h : is_odd f) : ∀ x : ℝ, f x + f (-x) = 0 := 
  by sorry

end odd_function_property_l161_161366


namespace range_of_a_l161_161112

def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 2) * x + 5

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ 4 → f x a ≤ f (x+1) a) : a ≥ -2 := 
by
  sorry

end range_of_a_l161_161112


namespace problem_statement_l161_161491

theorem problem_statement (a b : ℝ) (h1 : a ≠ 0) (h2 : ({a, b / a, 1} : Set ℝ) = {a^2, a + b, 0}) :
  a^2017 + b^2017 = -1 := by
  sorry

end problem_statement_l161_161491


namespace shaded_area_is_correct_l161_161931

-- Conditions definition
def shaded_numbers : ℕ := 2015
def boundary_properties (segment : ℕ) : Prop := 
  segment = 1 ∨ segment = 2

theorem shaded_area_is_correct : ∀ n : ℕ, n = shaded_numbers → boundary_properties n → 
  (∃ area : ℚ, area = 47.5) :=
by
  sorry

end shaded_area_is_correct_l161_161931


namespace infinitely_many_pairs_l161_161945

theorem infinitely_many_pairs : ∀ b : ℕ, ∃ a : ℕ, 2019 < 2^a / 3^b ∧ 2^a / 3^b < 2020 := 
by
  sorry

end infinitely_many_pairs_l161_161945


namespace largest_angle_l161_161482

-- Definitions for our conditions
def right_angle : ℝ := 90
def sum_of_two_angles (a b : ℝ) : Prop := a + b = (4 / 3) * right_angle
def angle_difference (a b : ℝ) : Prop := b = a + 40

-- Statement of the problem to be proved
theorem largest_angle (a b c : ℝ) (h_sum : sum_of_two_angles a b) (h_diff : angle_difference a b) (h_triangle : a + b + c = 180) : c = 80 :=
by sorry

end largest_angle_l161_161482


namespace middle_school_soccer_league_l161_161399

theorem middle_school_soccer_league (n : ℕ) (h : (n * (n - 1)) / 2 = 36) : n = 9 := 
  sorry

end middle_school_soccer_league_l161_161399


namespace new_circle_equation_l161_161276

-- Define the initial conditions
def initial_circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0
def radius_of_new_circle : ℝ := 2

-- Define the target equation of the circle
def target_circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- The theorem statement
theorem new_circle_equation (x y : ℝ) :
  initial_circle_equation x y → target_circle_equation x y :=
sorry

end new_circle_equation_l161_161276


namespace range_of_m_l161_161254

theorem range_of_m (m : ℝ) (h : (8 - m) / (m - 5) > 1) : 5 < m ∧ m < 13 / 2 :=
by
  sorry

end range_of_m_l161_161254


namespace max_value_of_expression_l161_161005

theorem max_value_of_expression (x y : ℝ) 
  (h : (x - 4)^2 / 4 + y^2 / 9 = 1) : 
  (x^2 / 4 + y^2 / 9 ≤ 9) ∧ ∃ x y, (x - 4)^2 / 4 + y^2 / 9 = 1 ∧ x^2 / 4 + y^2 / 9 = 9 :=
by
  sorry

end max_value_of_expression_l161_161005


namespace toms_investment_l161_161770

theorem toms_investment 
  (P : ℝ)
  (rA : ℝ := 0.06)
  (nA : ℝ := 1)
  (tA : ℕ := 4)
  (rB : ℝ := 0.08)
  (nB : ℕ := 2)
  (tB : ℕ := 4)
  (delta : ℝ := 100)
  (A_A := P * (1 + rA / nA) ^ (nA * tA))
  (A_B := P * (1 + rB / nB) ^ (nB * tB))
  (h : A_B - A_A = delta) : 
  P = 942.59 := by
sorry

end toms_investment_l161_161770


namespace find_original_sales_tax_percentage_l161_161390

noncomputable def original_sales_tax_percentage (x : ℝ) : Prop :=
∃ (x : ℝ),
  let reduced_tax := 10 / 3 / 100;
  let market_price := 9000;
  let difference := 14.999999999999986;
  (x / 100 * market_price - reduced_tax * market_price = difference) ∧ x = 0.5

theorem find_original_sales_tax_percentage : original_sales_tax_percentage 0.5 :=
sorry

end find_original_sales_tax_percentage_l161_161390


namespace brody_calculator_battery_life_l161_161665

theorem brody_calculator_battery_life (h : ∃ t : ℕ, (3 / 4) * t + 2 + 13 = t) : ∃ t : ℕ, t = 60 :=
by
  -- Define the quarters used by Brody and the remaining battery life after the exam.
  obtain ⟨t, ht⟩ := h
  -- Simplify the equation (3/4) * t + 2 + 13 = t to get t = 60
  sorry

end brody_calculator_battery_life_l161_161665


namespace answer_l161_161349

-- Definitions of geometric entities in terms of vectors
structure Square :=
  (A B C D E : ℝ × ℝ)
  (side_length : ℝ)
  (hAB_eq : (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = side_length ^ 2)
  (hBC_eq : (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = side_length ^ 2)
  (hCD_eq : (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 = side_length ^ 2)
  (hDA_eq : (A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2 = side_length ^ 2)
  (hE_midpoint : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def EC_ED_dot_product (s : Square) : ℝ :=
  let EC := (s.C.1 - s.E.1, s.C.2 - s.E.2)
  let ED := (s.D.1 - s.E.1, s.D.2 - s.E.2)
  dot_product EC ED

theorem answer (s : Square) (h_side_length : s.side_length = 2) :
  EC_ED_dot_product s = 3 :=
sorry

end answer_l161_161349


namespace dean_taller_than_ron_l161_161864

theorem dean_taller_than_ron (d h r : ℕ) (h1 : d = 15 * h) (h2 : r = 13) (h3 : d = 255) : h - r = 4 := 
by 
  sorry

end dean_taller_than_ron_l161_161864


namespace range_of_a_plus_b_l161_161529

variable {a b : ℝ}

def has_two_real_roots (a b : ℝ) : Prop :=
  let discriminant := b^2 - 4 * a * (-4)
  discriminant ≥ 0

def has_root_in_interval (a b : ℝ) : Prop :=
  (a + b - 4) * (4 * a + 2 * b - 4) < 0

theorem range_of_a_plus_b 
  (h1 : has_two_real_roots a b) 
  (h2 : has_root_in_interval a b) 
  (h3 : a > 0) : 
  a + b < 4 :=
sorry

end range_of_a_plus_b_l161_161529


namespace john_profit_l161_161680

-- Definitions based on given conditions
def total_newspapers := 500
def selling_price_per_newspaper : ℝ := 2
def discount_percentage : ℝ := 0.75
def percentage_sold : ℝ := 0.80

-- Derived basic definitions
def cost_price_per_newspaper := selling_price_per_newspaper * (1 - discount_percentage)
def total_cost_price := cost_price_per_newspaper * total_newspapers
def newspapers_sold := total_newspapers * percentage_sold
def revenue := selling_price_per_newspaper * newspapers_sold
def profit := revenue - total_cost_price

-- Theorem stating the profit
theorem john_profit : profit = 550 := by
  sorry

#check john_profit

end john_profit_l161_161680


namespace geometric_sequence_correct_l161_161527

theorem geometric_sequence_correct (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 = 8)
  (h2 : a 2 * a 3 = -8)
  (h_geom : ∀ (n : ℕ), a (n + 1) = a n * r) :
  a 4 = -1 :=
by {
  sorry
}

end geometric_sequence_correct_l161_161527


namespace minimum_possible_area_l161_161377

theorem minimum_possible_area (l w l_min w_min : ℝ) (hl : l = 5) (hw : w = 7) 
  (hl_min : l_min = l - 0.5) (hw_min : w_min = w - 0.5) : 
  l_min * w_min = 29.25 :=
by
  sorry

end minimum_possible_area_l161_161377


namespace proof_problem_1_proof_problem_2_l161_161473

/-
  Problem statement and conditions:
  (1) $(2023-\sqrt{3})^0 + \left| \left( \frac{1}{5} \right)^{-1} - \sqrt{75} \right| - \frac{\sqrt{45}}{\sqrt{5}}$
  (2) $(\sqrt{3}-2)^2 - (\sqrt{2}+\sqrt{3})(\sqrt{3}-\sqrt{2})$
-/

noncomputable def problem_1 := 
  (2023 - Real.sqrt 3)^0 + abs ((1/5: ℝ)⁻¹ - Real.sqrt 75) - Real.sqrt 45 / Real.sqrt 5

noncomputable def problem_2 := 
  (Real.sqrt 3 - 2) ^ 2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 3 - Real.sqrt 2)

theorem proof_problem_1 : problem_1 = 5 * Real.sqrt 3 - 7 :=
  by
    sorry

theorem proof_problem_2 : problem_2 = 6 - 4 * Real.sqrt 3 :=
  by
    sorry


end proof_problem_1_proof_problem_2_l161_161473


namespace gino_popsicle_sticks_l161_161654

variable (my_sticks : ℕ) (total_sticks : ℕ) (gino_sticks : ℕ)

def popsicle_sticks_condition (my_sticks : ℕ) (total_sticks : ℕ) (gino_sticks : ℕ) : Prop :=
  my_sticks = 50 ∧ total_sticks = 113

theorem gino_popsicle_sticks
  (h : popsicle_sticks_condition my_sticks total_sticks gino_sticks) :
  gino_sticks = 63 :=
  sorry

end gino_popsicle_sticks_l161_161654


namespace negation_of_proposition_l161_161133

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1)) ↔
  (∃ x₀ : ℝ, x₀ ≤ 0 ∧ (x₀ + 1) * Real.exp x₀ ≤ 1) := 
sorry

end negation_of_proposition_l161_161133


namespace filling_tank_with_pipes_l161_161936

theorem filling_tank_with_pipes :
  let Ra := 1 / 70
  let Rb := 2 * Ra
  let Rc := 2 * Rb
  let Rtotal := Ra + Rb + Rc
  Rtotal = 1 / 10 →  -- Given the combined rate fills the tank in 10 hours
  3 = 3 :=  -- Number of pipes used to fill the tank
by
  intros Ra Rb Rc Rtotal h
  simp [Ra, Rb, Rc] at h
  sorry

end filling_tank_with_pipes_l161_161936


namespace total_sides_is_48_l161_161235

-- Definitions based on the conditions
def num_dice_tom : Nat := 4
def num_dice_tim : Nat := 4
def sides_per_die : Nat := 6

-- The proof problem statement
theorem total_sides_is_48 : (num_dice_tom + num_dice_tim) * sides_per_die = 48 := by
  sorry

end total_sides_is_48_l161_161235


namespace smallest_number_two_reps_l161_161171

theorem smallest_number_two_reps : 
  ∃ (n : ℕ), (∀ x1 y1 x2 y2 : ℕ, 3 * x1 + 4 * y1 = n ∧ 3 * x2 + 4 * y2 = n → (x1 = x2 ∧ y1 = y2 ∨ ¬(x1 = x2 ∧ y1 = y2))) ∧ 
  ∀ m < n, (∀ x y : ℕ, ¬(3 * x + 4 * y = m ∧ ¬∃ (x1 y1 : ℕ), 3 * x1 + 4 * y1 = m) ∧ 
            (∃ x1 y1 x2 y2 : ℕ, 3 * x1 + 4 * y1 = m ∧ 3 * x2 + 4 * y2 = m ∧ ¬(x1 = x2 ∧ y1 = y2))) :=
  sorry

end smallest_number_two_reps_l161_161171


namespace cards_selection_count_l161_161900

noncomputable def numberOfWaysToChooseCards : Nat :=
  (Nat.choose 4 3) * 3 * (Nat.choose 13 2) * (13 ^ 2)

theorem cards_selection_count :
  numberOfWaysToChooseCards = 158184 := by
  sorry

end cards_selection_count_l161_161900


namespace cost_of_each_pair_of_jeans_l161_161918

-- Conditions
def costWallet : ℕ := 50
def costSneakers : ℕ := 100
def pairsSneakers : ℕ := 2
def costBackpack : ℕ := 100
def totalSpent : ℕ := 450
def pairsJeans : ℕ := 2

-- Definitions
def totalSpentLeonard := costWallet + pairsSneakers * costSneakers
def totalSpentMichaelWithoutJeans := costBackpack

-- Goal: Prove the cost of each pair of jeans
theorem cost_of_each_pair_of_jeans :
  let totalCostJeans := totalSpent - (totalSpentLeonard + totalSpentMichaelWithoutJeans)
  let costPerPairJeans := totalCostJeans / pairsJeans
  costPerPairJeans = 50 :=
by
  intros
  let totalCostJeans := totalSpent - (totalSpentLeonard + totalSpentMichaelWithoutJeans)
  let costPerPairJeans := totalCostJeans / pairsJeans
  show costPerPairJeans = 50
  sorry

end cost_of_each_pair_of_jeans_l161_161918


namespace abc_prod_eq_l161_161631

-- Define a structure for points and triangles
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B C : Point)

-- Define the angles formed by points in a triangle
def angle (A B C : Point) : ℝ := sorry

-- Define the lengths between points
def length (A B : Point) : ℝ := sorry

-- Conditions of the problem
theorem abc_prod_eq (A B C D : Point) 
  (h1 : angle A D C = angle A B C + 60)
  (h2 : angle C D B = angle C A B + 60)
  (h3 : angle B D A = angle B C A + 60) : 
  length A B * length C D = length B C * length A D :=
sorry

end abc_prod_eq_l161_161631


namespace scientific_notation_560000_l161_161065

theorem scientific_notation_560000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 560000 = a * 10 ^ n ∧ a = 5.6 ∧ n = 5 :=
by 
  sorry

end scientific_notation_560000_l161_161065


namespace cos_sum_sin_sum_cos_diff_sin_diff_l161_161153

section

variables (A B : ℝ)

-- Definition of cos and sin of angles
def cos (θ : ℝ) : ℝ := sorry
def sin (θ : ℝ) : ℝ := sorry

-- Cosine of the sum of angles
theorem cos_sum : cos (A + B) = cos A * cos B - sin A * sin B := sorry

-- Sine of the sum of angles
theorem sin_sum : sin (A + B) = sin A * cos B + cos A * sin B := sorry

-- Cosine of the difference of angles
theorem cos_diff : cos (A - B) = cos A * cos B + sin A * sin B := sorry

-- Sine of the difference of angles
theorem sin_diff : sin (A - B) = sin A * cos B - cos A * sin B := sorry

end

end cos_sum_sin_sum_cos_diff_sin_diff_l161_161153


namespace crowdfunding_total_amount_l161_161226

theorem crowdfunding_total_amount
  (backers_highest_level : ℕ := 2)
  (backers_second_level : ℕ := 3)
  (backers_lowest_level : ℕ := 10)
  (amount_highest_level : ℝ := 5000) :
  ((backers_highest_level * amount_highest_level) + 
   (backers_second_level * (amount_highest_level / 10)) + 
   (backers_lowest_level * (amount_highest_level / 100))) = 12000 :=
by
  sorry

end crowdfunding_total_amount_l161_161226


namespace number_of_people_only_went_to_aquarium_is_5_l161_161895

-- Define the conditions
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group_size : ℕ := 10
def total_earnings : ℕ := 240

-- Define the problem in Lean
theorem number_of_people_only_went_to_aquarium_is_5 :
  ∃ x : ℕ, (total_earnings - (group_size * (admission_fee + tour_fee)) = x * admission_fee) → x = 5 :=
by
  sorry

end number_of_people_only_went_to_aquarium_is_5_l161_161895


namespace find_value_of_expression_l161_161879

theorem find_value_of_expression (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) :
  (a + b)^9 + a^6 = 2 :=
sorry

end find_value_of_expression_l161_161879


namespace work_problem_l161_161563

theorem work_problem (W : ℝ) (d : ℝ) :
  (1 / 40) * d * W + (28 / 35) * W = W → d = 8 :=
by
  intro h
  sorry

end work_problem_l161_161563


namespace minimum_phi_l161_161088

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 4))

theorem minimum_phi (φ : ℝ) (hφ : φ > 0) :
  (∃ k : ℤ, φ = (3/8) * Real.pi - (k * Real.pi / 2)) → φ = (3/8) * Real.pi :=
by
  sorry

end minimum_phi_l161_161088


namespace lines_are_skew_l161_161287

def line1 (a t : ℝ) : ℝ × ℝ × ℝ := 
  (2 + 3 * t, 1 + 4 * t, a + 5 * t)
  
def line2 (u : ℝ) : ℝ × ℝ × ℝ := 
  (5 + 6 * u, 3 + 3 * u, 1 + 2 * u)

theorem lines_are_skew (a : ℝ) : (∀ t u : ℝ, line1 a t ≠ line2 u) ↔ a ≠ -4/5 :=
sorry

end lines_are_skew_l161_161287


namespace sum_of_two_digit_factors_of_8060_l161_161620

theorem sum_of_two_digit_factors_of_8060 : ∃ (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (a * b = 8060) ∧ (a + b = 127) :=
by sorry

end sum_of_two_digit_factors_of_8060_l161_161620


namespace point_B_represents_2_or_neg6_l161_161501

def A : ℤ := -2

def B (move : ℤ) : ℤ := A + move

theorem point_B_represents_2_or_neg6 (move : ℤ) (h : move = 4 ∨ move = -4) : 
  B move = 2 ∨ B move = -6 :=
by
  cases h with
  | inl h1 => 
    rw [h1]
    unfold B
    unfold A
    simp
  | inr h1 => 
    rw [h1]
    unfold B
    unfold A
    simp

end point_B_represents_2_or_neg6_l161_161501


namespace not_odd_not_even_min_value_3_l161_161691

def f (x : ℝ) : ℝ := x^2 + abs (x - 2) - 1

-- Statement 1: Prove that the function is neither odd nor even.
theorem not_odd_not_even : 
  ¬(∀ x, f (-x) = -f x) ∧ ¬(∀ x, f (-x) = f x) :=
sorry

-- Statement 2: Prove that the minimum value of the function is 3.
theorem min_value_3 : ∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≥ 3 :=
sorry

end not_odd_not_even_min_value_3_l161_161691


namespace ashley_age_l161_161510

theorem ashley_age (A M : ℕ) (h1 : 4 * M = 7 * A) (h2 : A + M = 22) : A = 8 :=
sorry

end ashley_age_l161_161510


namespace orchid_bushes_total_l161_161850

def current_orchid_bushes : ℕ := 22
def new_orchid_bushes : ℕ := 13

theorem orchid_bushes_total : current_orchid_bushes + new_orchid_bushes = 35 := 
by 
  sorry

end orchid_bushes_total_l161_161850


namespace base_number_is_two_l161_161086

theorem base_number_is_two (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^22)
  (h2 : n = 21) : x = 2 :=
sorry

end base_number_is_two_l161_161086


namespace fraction_transform_l161_161156

theorem fraction_transform (x : ℕ) (h : 9 * (537 - x) = 463 + x) : x = 437 :=
by
  sorry

end fraction_transform_l161_161156


namespace solve_for_x_l161_161095

theorem solve_for_x (x y : ℝ) (h1 : 3 * x + y = 75) (h2 : 2 * (3 * x + y) - y = 138) : x = 21 :=
  sorry

end solve_for_x_l161_161095


namespace smallest_N_for_equal_adults_and_children_l161_161241

theorem smallest_N_for_equal_adults_and_children :
  ∃ (N : ℕ), N > 0 ∧ (∀ a b : ℕ, 8 * N = a ∧ 12 * N = b ∧ a = b) ∧ N = 3 :=
sorry

end smallest_N_for_equal_adults_and_children_l161_161241


namespace nonnegative_fraction_interval_l161_161618

theorem nonnegative_fraction_interval : 
  ∀ x : ℝ, (0 ≤ x ∧ x < 3) ↔ (0 ≤ (x - 15 * x^2 + 36 * x^3) / (9 - x^3)) := by
sorry

end nonnegative_fraction_interval_l161_161618


namespace multiply_exponents_l161_161060

theorem multiply_exponents (a : ℝ) : 2 * a^3 * 3 * a^2 = 6 * a^5 := by
  sorry

end multiply_exponents_l161_161060


namespace find_larger_number_l161_161565

theorem find_larger_number (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) (h3 : x * y = 375) (hx : x > y) : x = 25 :=
sorry

end find_larger_number_l161_161565


namespace marble_problem_l161_161656

theorem marble_problem
  (M : ℕ)
  (X : ℕ)
  (h1 : M = 18 * X)
  (h2 : M = 20 * (X - 1)) :
  M = 180 :=
by
  sorry

end marble_problem_l161_161656


namespace find_a_value_l161_161616

theorem find_a_value (a : ℝ) (m : ℝ) (f g : ℝ → ℝ)
  (f_def : ∀ x, f x = Real.log x / Real.log a)
  (g_def : ∀ x, g x = (2 + m) * Real.sqrt x)
  (a_pos : 0 < a) (a_neq_one : a ≠ 1)
  (max_f : ∀ x ∈ Set.Icc (1 / 2) 16, f x ≤ 4)
  (min_f : ∀ x ∈ Set.Icc (1 / 2) 16, m ≤ f x)
  (g_increasing : ∀ x y, 0 < x → x < y → g x < g y):
  a = 2 :=
sorry

end find_a_value_l161_161616


namespace minimum_value_of_z_l161_161985

theorem minimum_value_of_z 
  (x y : ℝ) 
  (h1 : x - 2 * y + 2 ≥ 0) 
  (h2 : 2 * x - y - 2 ≤ 0) 
  (h3 : y ≥ 0) :
  ∃ (z : ℝ), z = 3 * x + y ∧ z = -6 :=
sorry

end minimum_value_of_z_l161_161985


namespace k_greater_than_half_l161_161557

-- Definition of the problem conditions
variables {a b c k : ℝ}

-- Assume a, b, c are the sides of a triangle
axiom triangle_inequality : a + b > c

-- Given condition
axiom sides_condition : a^2 + b^2 = k * c^2

-- The theorem to prove k > 0.5
theorem k_greater_than_half (h1 : a + b > c) (h2 : a^2 + b^2 = k * c^2) : k > 0.5 :=
by
  sorry

end k_greater_than_half_l161_161557


namespace closed_broken_line_impossible_l161_161837

theorem closed_broken_line_impossible (n : ℕ) (h : n = 1989) : ¬ (∃ a b : ℕ, 2 * (a + b) = n) :=
by {
  sorry
}

end closed_broken_line_impossible_l161_161837


namespace permutation_6_2_eq_30_l161_161688

theorem permutation_6_2_eq_30 :
  (Nat.factorial 6) / (Nat.factorial (6 - 2)) = 30 :=
by
  sorry

end permutation_6_2_eq_30_l161_161688


namespace average_weight_proof_l161_161054

variables (W_A W_B W_C W_D W_E : ℝ)

noncomputable def final_average_weight (W_A W_B W_C W_D W_E : ℝ) : ℝ := (W_B + W_C + W_D + W_E) / 4

theorem average_weight_proof
  (h1 : (W_A + W_B + W_C) / 3 = 84)
  (h2 : W_A = 77)
  (h3 : (W_A + W_B + W_C + W_D) / 4 = 80)
  (h4 : W_E = W_D + 5) :
  final_average_weight W_A W_B W_C W_D W_E = 97.25 :=
by
  sorry

end average_weight_proof_l161_161054


namespace larger_integer_of_two_with_difference_8_and_product_168_l161_161660

theorem larger_integer_of_two_with_difference_8_and_product_168 :
  ∃ (x y : ℕ), x > y ∧ x - y = 8 ∧ x * y = 168 ∧ x = 14 :=
by
  sorry

end larger_integer_of_two_with_difference_8_and_product_168_l161_161660


namespace library_books_l161_161599

theorem library_books (N x y : ℕ) (h1 : x = N / 17) (h2 : y = x + 2000)
    (h3 : y = (N - 2 * 2000) / 15 + (14 * (N - 2000) / 17)): 
  N = 544000 := 
sorry

end library_books_l161_161599


namespace initial_bird_count_l161_161844

theorem initial_bird_count (B : ℕ) (h₁ : B + 7 = 12) : B = 5 :=
by
  sorry

end initial_bird_count_l161_161844


namespace exists_duplicate_parenthesizations_l161_161700

def expr : List Int := List.range' 1 (1991 + 1)

def num_parenthesizations : Nat := 2 ^ 995

def num_distinct_results : Nat := 3966067

theorem exists_duplicate_parenthesizations :
  num_parenthesizations > num_distinct_results :=
sorry

end exists_duplicate_parenthesizations_l161_161700


namespace range_of_x_plus_2y_l161_161289

theorem range_of_x_plus_2y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) : x + 2 * y ≥ 9 :=
sorry

end range_of_x_plus_2y_l161_161289


namespace total_cost_of_goods_l161_161828

theorem total_cost_of_goods :
  ∃ (M R F : ℝ),
    (10 * M = 24 * R) ∧
    (6 * F = 2 * R) ∧
    (F = 20.50) ∧
    (4 * M + 3 * R + 5 * F = 877.40) :=
by {
  sorry
}

end total_cost_of_goods_l161_161828


namespace power_function_k_values_l161_161125

theorem power_function_k_values (k : ℝ) : (∃ (a : ℝ), (k^2 - k - 5) = a) → (k = 3 ∨ k = -2) :=
by
  intro h
  have h1 : k^2 - k - 5 = 1 := sorry -- Using the condition that it is a power function
  have h2 : k^2 - k - 6 = 0 := by linarith -- Simplify the equation
  exact sorry -- Solve the quadratic equation

end power_function_k_values_l161_161125


namespace spring_work_done_l161_161259

theorem spring_work_done (F : ℝ) (l : ℝ) (stretched_length : ℝ) (k : ℝ) (W : ℝ) 
  (hF : F = 10) (hl : l = 0.1) (hk : k = F / l) (h_stretched_length : stretched_length = 0.06) : 
  W = 0.18 :=
by
  sorry

end spring_work_done_l161_161259


namespace equal_cost_l161_161401

theorem equal_cost (x : ℝ) : (2.75 * x + 125 = 1.50 * x + 140) ↔ (x = 12) := 
by sorry

end equal_cost_l161_161401


namespace prob_draw_correct_l161_161372

-- Given conditions
def prob_A_wins : ℝ := 0.40
def prob_A_not_lose : ℝ := 0.90

-- Definition to be proved
def prob_draw : ℝ := prob_A_not_lose - prob_A_wins

theorem prob_draw_correct : prob_draw = 0.50 := by
  sorry

end prob_draw_correct_l161_161372


namespace problem_solution_l161_161980

theorem problem_solution (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a^2 + b^2 + c^2 = 2) (h3 : a^3 + b^3 + c^3 = 3) :
  (a * b * c = 1 / 6) ∧ (a^4 + b^4 + c^4 = 25 / 6) :=
by {
  sorry
}

end problem_solution_l161_161980


namespace division_of_exponents_l161_161123

-- Define the conditions as constants and statements that we are concerned with
variables (x : ℝ)

-- The Lean 4 statement of the equivalent proof problem
theorem division_of_exponents (h₁ : x ≠ 0) : x^8 / x^2 = x^6 := 
sorry

end division_of_exponents_l161_161123


namespace product_of_legs_divisible_by_12_l161_161628

theorem product_of_legs_divisible_by_12 
  (a b c : ℕ) 
  (h_triangle : a^2 + b^2 = c^2) 
  (h_int : ∃ a b c : ℕ, a^2 + b^2 = c^2) :
  ∃ k : ℕ, a * b = 12 * k :=
sorry

end product_of_legs_divisible_by_12_l161_161628


namespace value_of_a_l161_161154

theorem value_of_a (a : ℝ) (h : 1 ∈ ({a, a ^ 2} : Set ℝ)) : a = -1 :=
sorry

end value_of_a_l161_161154


namespace arithmetic_sequence_sum_l161_161783

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + m) = a n + m * (a 1 - a 0)

theorem arithmetic_sequence_sum
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 5 + a 6 + a 7 = 15) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by
  sorry

end arithmetic_sequence_sum_l161_161783


namespace butter_needed_for_original_recipe_l161_161920

-- Define the conditions
def butter_to_flour_ratio : ℚ := 12 / 56

def flour_for_original_recipe : ℚ := 14

def butter_for_original_recipe (ratio : ℚ) (flour : ℚ) : ℚ :=
  ratio * flour

-- State the theorem
theorem butter_needed_for_original_recipe :
  butter_for_original_recipe butter_to_flour_ratio flour_for_original_recipe = 3 := 
sorry

end butter_needed_for_original_recipe_l161_161920


namespace square_of_chord_length_l161_161251

/--
Given two circles with radii 10 and 7, and centers 15 units apart, if they intersect at a point P such that the chords QP and PR are of equal lengths, then the square of the length of chord QP is 289.
-/
theorem square_of_chord_length :
  ∀ (r1 r2 d x : ℝ), r1 = 10 → r2 = 7 → d = 15 →
  let cos_theta1 := (x^2 + r1^2 - r2^2) / (2 * r1 * x)
  let cos_theta2 := (x^2 + r2^2 - r1^2) / (2 * r2 * x)
  cos_theta1 = cos_theta2 →
  x^2 = 289 := 
by
  intros r1 r2 d x h1 h2 h3
  let cos_theta1 := (x^2 + r1^2 - r2^2) / (2 * r1 * x)
  let cos_theta2 := (x^2 + r2^2 - r1^2) / (2 * r2 * x)
  intro h4
  sorry

end square_of_chord_length_l161_161251


namespace solve_for_x_l161_161717

theorem solve_for_x (A B C D: Type) 
(y z w x : ℝ) 
(h_triangle : ∃ a b c : Type, True) 
(h_D_on_extension : ∃ D_on_extension : Type, True)
(h_AD_GT_BD : ∃ s : Type, True) 
(h_x_at_D : ∃ t : Type, True) 
(h_y_at_A : ∃ u : Type, True) 
(h_z_at_B : ∃ v : Type, True) 
(h_w_at_C : ∃ w : Type, True)
(h_triangle_angle_sum : y + z + w = 180):
x = 180 - z - w := by
  sorry

end solve_for_x_l161_161717


namespace Seokjin_tangerines_per_day_l161_161641

theorem Seokjin_tangerines_per_day 
  (T_initial : ℕ) (D : ℕ) (T_remaining : ℕ) 
  (h1 : T_initial = 29) 
  (h2 : D = 8) 
  (h3 : T_remaining = 5) : 
  (T_initial - T_remaining) / D = 3 := 
by
  sorry

end Seokjin_tangerines_per_day_l161_161641


namespace monthly_pool_cost_is_correct_l161_161776

def cost_of_cleaning : ℕ := 150
def tip_percentage : ℕ := 10
def number_of_cleanings_in_a_month : ℕ := 30 / 3
def cost_of_chemicals_per_use : ℕ := 200
def number_of_chemical_uses_in_a_month : ℕ := 2

def monthly_cost_of_pool : ℕ :=
  let cost_per_cleaning := cost_of_cleaning + (cost_of_cleaning * tip_percentage / 100)
  let total_cleaning_cost := number_of_cleanings_in_a_month * cost_per_cleaning
  let total_chemical_cost := number_of_chemical_uses_in_a_month * cost_of_chemicals_per_use
  total_cleaning_cost + total_chemical_cost

theorem monthly_pool_cost_is_correct : monthly_cost_of_pool = 2050 :=
by
  sorry

end monthly_pool_cost_is_correct_l161_161776


namespace maximal_sum_of_xy_l161_161282

theorem maximal_sum_of_xy (x y : ℤ) (h : x^2 + y^2 = 100) : ∃ (s : ℤ), s = 14 ∧ ∀ (u v : ℤ), u^2 + v^2 = 100 → u + v ≤ s :=
by sorry

end maximal_sum_of_xy_l161_161282


namespace three_students_two_groups_l161_161239

theorem three_students_two_groups : 
  (2 : ℕ) ^ 3 = 8 := 
by
  sorry

end three_students_two_groups_l161_161239


namespace cone_volume_proof_l161_161256

noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_proof :
  (cone_volume 1 (Real.sqrt 3)) = (Real.sqrt 3 / 3) * Real.pi :=
by
  sorry

end cone_volume_proof_l161_161256


namespace find_v_l161_161910

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 1;
    3, 0]

noncomputable def v : Matrix (Fin 2) (Fin 1) ℝ :=
  !![0;
    1 / 30.333]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ :=
  1

theorem find_v : 
  (A ^ 10 + A ^ 8 + A ^ 6 + A ^ 4 + A ^ 2 + I) * v = !![0; 12] :=
  sorry

end find_v_l161_161910


namespace stock_price_drop_l161_161308

theorem stock_price_drop (P : ℝ) (h1 : P > 0) (x : ℝ)
  (h3 : (1.30 * (1 - x/100) * 1.20 * P) = 1.17 * P) :
  x = 25 :=
by
  sorry

end stock_price_drop_l161_161308


namespace power_sum_eq_l161_161569

theorem power_sum_eq : (-2)^2011 + (-2)^2012 = 2^2011 := by
  sorry

end power_sum_eq_l161_161569


namespace min_value_fraction_l161_161021

theorem min_value_fraction (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_gt_hy : x > y) (h_eq : x + 2 * y = 3) : 
  (∃ t, t = (1 / (x - y) + 9 / (x + 5 * y)) ∧ t = 8 / 3) :=
by 
  sorry

end min_value_fraction_l161_161021


namespace rhombus_diagonals_perpendicular_not_in_rectangle_l161_161671

-- Definitions for the rhombus
structure Rhombus :=
  (diagonals_perpendicular : Prop)

-- Definitions for the rectangle
structure Rectangle :=
  (diagonals_not_perpendicular : Prop)

-- The main proof statement
theorem rhombus_diagonals_perpendicular_not_in_rectangle 
  (R : Rhombus) 
  (Rec : Rectangle) : 
  R.diagonals_perpendicular ∧ Rec.diagonals_not_perpendicular :=
by sorry

end rhombus_diagonals_perpendicular_not_in_rectangle_l161_161671


namespace length_of_AB_l161_161823

theorem length_of_AB {A B P Q : ℝ} (h1 : P = 3 / 5 * B)
                    (h2 : Q = 2 / 5 * A + 3 / 5 * B)
                    (h3 : dist P Q = 5) :
  dist A B = 25 :=
by sorry

end length_of_AB_l161_161823


namespace necessary_but_not_sufficient_condition_l161_161957

noncomputable def condition (m : ℝ) : Prop := 1 < m ∧ m < 3

def represents_ellipse (m : ℝ) (x y : ℝ) : Prop :=
  (x ^ 2) / (m - 1) + (y ^ 2) / (3 - m) = 1

theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (∃ x y, represents_ellipse m x y) → condition m :=
sorry

end necessary_but_not_sufficient_condition_l161_161957


namespace sequence_noncongruent_modulo_l161_161897

theorem sequence_noncongruent_modulo 
  (a : ℕ → ℕ)
  (h0 : a 1 = 1)
  (h1 : ∀ n, a (n + 1) = a n + 2^(a n)) :
  ∀ (i j : ℕ), i ≠ j → i ≤ 32021 → j ≤ 32021 →
  (a i) % (3^2021) ≠ (a j) % (3^2021) := 
by
  sorry

end sequence_noncongruent_modulo_l161_161897


namespace partA_l161_161048

theorem partA (n : ℕ) : 
  1 < (n + 1 / 2) * Real.log (1 + 1 / n) ∧ (n + 1 / 2) * Real.log (1 + 1 / n) < 1 + 1 / (12 * n * (n + 1)) := 
sorry

end partA_l161_161048
