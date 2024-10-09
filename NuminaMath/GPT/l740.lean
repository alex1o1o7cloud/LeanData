import Mathlib

namespace max_cells_primitive_dinosaur_l740_74015

section Dinosaur

universe u

-- Define a dinosaur as a structure with at least 2007 cells
structure Dinosaur (α : Type u) :=
(cells : ℕ) (connected : α → α → Prop)
(h_cells : cells ≥ 2007)
(h_connected : ∀ (x y : α), connected x y → connected y x)

-- Define a primitive dinosaur where the cells cannot be partitioned into two or more dinosaurs
structure PrimitiveDinosaur (α : Type u) extends Dinosaur α :=
(h_partition : ∀ (x : α), ¬∃ (d1 d2 : Dinosaur α), (d1.cells + d2.cells = cells) ∧ 
  (d1 ≠ d2 ∧ d1.cells ≥ 2007 ∧ d2.cells ≥ 2007))

-- Prove that the maximum number of cells in a Primitive Dinosaur is 8025
theorem max_cells_primitive_dinosaur : ∀ (α : Type u), ∃ (d : PrimitiveDinosaur α), d.cells = 8025 :=
sorry

end Dinosaur

end max_cells_primitive_dinosaur_l740_74015


namespace calculate_value_l740_74081

theorem calculate_value : (24 + 12) / ((5 - 3) * 2) = 9 := by 
  sorry

end calculate_value_l740_74081


namespace inverse_of_g_at_1_over_32_l740_74028

noncomputable def g (x : ℝ) : ℝ := (x^5 + 2) / 4

theorem inverse_of_g_at_1_over_32 :
  g⁻¹ (1/32) = (-15 / 8)^(1/5) :=
sorry

end inverse_of_g_at_1_over_32_l740_74028


namespace find_p_l740_74010

def delta (a b : ℝ) : ℝ := a * b + a + b

theorem find_p (p : ℝ) (h : delta p 3 = 39) : p = 9 :=
by
  sorry

end find_p_l740_74010


namespace exists_y_equals_7_l740_74025

theorem exists_y_equals_7 : ∃ (x y z t : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ y = 7 ∧ x + y + z + t = 10 :=
by {
  sorry -- This is where the actual proof would go.
}

end exists_y_equals_7_l740_74025


namespace inequality_solution_ge_11_l740_74095

theorem inequality_solution_ge_11
  (m n : ℝ)
  (h1 : m > 0)
  (h2 : n > 1)
  (h3 : (1/m) + (2/(n-1)) = 1) :
  m + 2 * n ≥ 11 :=
sorry

end inequality_solution_ge_11_l740_74095


namespace baba_yaga_powder_problem_l740_74052

theorem baba_yaga_powder_problem (A B d : ℤ) 
  (h1 : A + B + d = 6) 
  (h2 : A + d = 3) 
  (h3 : B + d = 2) : 
  A = 4 ∧ B = 3 :=
by
  -- Proof omitted, as only the statement is required
  sorry

end baba_yaga_powder_problem_l740_74052


namespace part1_part2_l740_74017

-- Define Set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3/2}

-- Define Set B, parameterized by m
def B (m : ℝ) : Set ℝ := {x | 1-m < x ∧ x ≤ 3*m + 1}

-- Proof Problem (1): When m = 1, A ∩ B = {x | 0 < x ∧ x ≤ 3/2}
theorem part1 (x : ℝ) : (x ∈ A ∩ B 1) ↔ (0 < x ∧ x ≤ 3/2) := by
  sorry

-- Proof Problem (2): If ∀ x, x ∈ A → x ∈ B m, then m ∈ (-∞, 1/6]
theorem part2 (m : ℝ) : (∀ x, x ∈ A → x ∈ B m) → m ≤ 1/6 := by
  sorry

end part1_part2_l740_74017


namespace total_chocolate_bars_in_colossal_box_l740_74031

theorem total_chocolate_bars_in_colossal_box :
  let colossal_boxes := 350
  let sizable_boxes := 49
  let small_boxes := 75
  colossal_boxes * sizable_boxes * small_boxes = 1287750 :=
by
  sorry

end total_chocolate_bars_in_colossal_box_l740_74031


namespace shaded_areas_different_l740_74075

/-
Question: How do the shaded areas of three different large squares (I, II, and III) compare?
Conditions:
1. Square I has diagonals drawn, and small squares are shaded at each corner where diagonals meet the sides.
2. Square II has vertical and horizontal lines drawn through the midpoints, creating four smaller squares, with one centrally shaded.
3. Square III has one diagonal from one corner to the center and a straight line from the midpoint of the opposite side to the center, creating various triangles and trapezoids, with a trapezoid area around the center being shaded.
Proof:
Prove that the shaded areas of squares I, II, and III are all different given the conditions on how squares I, II, and III are partitioned and shaded.
-/
theorem shaded_areas_different :
  ∀ (a : ℝ) (A1 A2 A3 : ℝ), (A1 = 1/4 * a^2) ∧ (A2 = 1/4 * a^2) ∧ (A3 = 3/8 * a^2) → 
  A1 ≠ A3 ∧ A2 ≠ A3 :=
by
  sorry

end shaded_areas_different_l740_74075


namespace cost_of_three_pencils_and_two_pens_l740_74044

theorem cost_of_three_pencils_and_two_pens 
  (p q : ℝ) 
  (h1 : 3 * p + 2 * q = 4.15) 
  (h2 : 2 * p + 3 * q = 3.70) : 
  3 * p + 2 * q = 4.15 := 
by 
  exact h1

end cost_of_three_pencils_and_two_pens_l740_74044


namespace expansion_eq_l740_74047

variable (x y : ℝ) -- x and y are real numbers
def a := 5
def b := 3
def c := 15

theorem expansion_eq : (x + a) * (b * y + c) = 3 * x * y + 15 * x + 15 * y + 75 := by 
  sorry

end expansion_eq_l740_74047


namespace total_fuel_needed_l740_74069

/-- Given that Car B can travel 30 miles per gallon and needs to cover a distance of 750 miles,
    and Car C has a fuel consumption rate of 20 miles per gallon and will travel 900 miles,
    prove that the total combined fuel required for Cars B and C is 70 gallons. -/
theorem total_fuel_needed (miles_per_gallon_B : ℕ) (miles_per_gallon_C : ℕ)
  (distance_B : ℕ) (distance_C : ℕ)
  (hB : miles_per_gallon_B = 30) (hC : miles_per_gallon_C = 20)
  (dB : distance_B = 750) (dC : distance_C = 900) :
  (distance_B / miles_per_gallon_B) + (distance_C / miles_per_gallon_C) = 70 := by {
    sorry 
}

end total_fuel_needed_l740_74069


namespace a_1995_is_squared_l740_74037

variable (a : ℕ → ℕ)

-- Conditions on the sequence 
axiom seq_condition  {m n : ℕ} (h : m ≥ n) : 
  a (m + n) + a (m - n) = (a (2 * m) + a (2 * n)) / 2

axiom initial_value : a 1 = 1

-- Goal to prove
theorem a_1995_is_squared : a 1995 = 1995^2 :=
sorry

end a_1995_is_squared_l740_74037


namespace speed_of_man_in_still_water_l740_74009

variables (v_m v_s : ℝ)

theorem speed_of_man_in_still_water :
  (v_m + v_s) * 5 = 36 ∧ (v_m - v_s) * 7 = 22 → v_m = 5.17 :=
by 
  sorry

end speed_of_man_in_still_water_l740_74009


namespace pipe_filling_time_l740_74090

theorem pipe_filling_time (t : ℕ) (h : 2 * (1 / t + 1 / 15) + 10 * (1 / 15) = 1) : t = 10 := by
  sorry

end pipe_filling_time_l740_74090


namespace xy_in_B_l740_74082

def A : Set ℤ := 
  {z | ∃ a b k m : ℤ, z = m * a^2 + k * a * b + m * b^2}

def B : Set ℤ := 
  {z | ∃ a b k m : ℤ, z = a^2 + k * a * b + m^2 * b^2}

theorem xy_in_B (x y : ℤ) (h1 : x ∈ A) (h2 : y ∈ A) : x * y ∈ B := by
  sorry

end xy_in_B_l740_74082


namespace ellipse_range_m_l740_74012

theorem ellipse_range_m (m : ℝ) :
    (∀ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2 → 
    ∃ (c : ℝ), c = x^2 + (y + 1)^2 ∧ m > 5) :=
sorry

end ellipse_range_m_l740_74012


namespace find_m_for_root_l740_74055

-- Define the fractional equation to find m
def fractional_equation (x m : ℝ) : Prop :=
  (x + 2) / (x - 1) = m / (1 - x)

-- State the theorem that we need to prove
theorem find_m_for_root : ∃ m : ℝ, (∃ x : ℝ, fractional_equation x m) ∧ m = -3 :=
by
  sorry

end find_m_for_root_l740_74055


namespace people_after_five_years_l740_74074

noncomputable def population_in_year : ℕ → ℕ
| 0       => 20
| (k + 1) => 4 * population_in_year k - 18

theorem people_after_five_years : population_in_year 5 = 14382 := by
  sorry

end people_after_five_years_l740_74074


namespace inverse_of_square_l740_74038

variable (A : Matrix (Fin 2) (Fin 2) ℝ)

theorem inverse_of_square (h : A⁻¹ = ![
  ![3, -2],
  ![1, 1]
]) : 
  (A^2)⁻¹ = ![
  ![7, -8],
  ![4, -1]
] :=
sorry

end inverse_of_square_l740_74038


namespace collinear_points_solves_a_l740_74030

theorem collinear_points_solves_a : 
  ∀ (a : ℝ),
  let A := (1, 3)
  let B := (5, 8)
  let C := (29, a)
  (8 - 3) / (5 - 1) = (a - 8) / (29 - 5) → a = 38 :=
by 
  intro a
  let A := (1, 3)
  let B := (5, 8)
  let C := (29, a)
  intro h
  sorry

end collinear_points_solves_a_l740_74030


namespace geometric_series_ratio_l740_74051

theorem geometric_series_ratio (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ (n : ℕ), S n = a 1 * (1 - q^n) / (1 - q))
  (h2 : a 3 + 2 * a 6 = 0)
  (h3 : a 6 = a 3 * q^3)
  (h4 : q^3 = -1 / 2) :
  S 3 / S 6 = 2 := 
sorry

end geometric_series_ratio_l740_74051


namespace carlos_biked_more_than_daniel_l740_74008

-- Definitions modeled from conditions
def distance_carlos : ℕ := 108
def distance_daniel : ℕ := 90
def time_hours : ℕ := 6

-- Lean statement to prove the difference in distance
theorem carlos_biked_more_than_daniel : distance_carlos - distance_daniel = 18 := 
  by 
    sorry

end carlos_biked_more_than_daniel_l740_74008


namespace sequence_prime_bounded_l740_74068

theorem sequence_prime_bounded (c : ℕ) (h : c > 0) : 
  ∀ (p : ℕ → ℕ), (∀ k, Nat.Prime (p k)) → (p 0) = some_prime →
  (∀ k, ∃ q, Nat.Prime q ∧ q ∣ (p k + c) ∧ (∀ i < k, q ≠ p i)) → 
  (∃ N, ∀ m ≥ N, ∀ n ≥ N, p m = p n) :=
by
  sorry

end sequence_prime_bounded_l740_74068


namespace cook_stole_the_cookbook_l740_74092

-- Define the suspects
inductive Suspect
| CheshireCat
| Duchess
| Cook
deriving DecidableEq, Repr

-- Define the predicate for lying
def lied (s : Suspect) : Prop := sorry

-- Define the conditions
def conditions (thief : Suspect) : Prop :=
  lied thief ∧
  ((∀ s : Suspect, s ≠ thief → lied s) ∨ (∀ s : Suspect, s ≠ thief → ¬lied s))

-- Define the goal statement
theorem cook_stole_the_cookbook : conditions Suspect.Cook :=
sorry

end cook_stole_the_cookbook_l740_74092


namespace sum_of_first_9_terms_zero_l740_74026

variable (a_n : ℕ → ℝ) (d a₁ : ℝ)
def arithmetic_seq := ∀ n, a_n n = a₁ + (n - 1) * d

def condition (a_n : ℕ → ℝ) := (a_n 2 + a_n 9 = a_n 6)

theorem sum_of_first_9_terms_zero 
  (h_arith : arithmetic_seq a_n d a₁) 
  (h_cond : condition a_n) : 
  (9 * a₁ + (9 * 8 / 2) * d) = 0 :=
by
  sorry

end sum_of_first_9_terms_zero_l740_74026


namespace hyperbola_equation_l740_74046

variable (a b c : ℝ)

def system_eq1 := (4 / (-3 - c)) = (- a / b)
def system_eq2 := ((c - 3) / 2) * (b / a) = 2
def system_eq3 := a ^ 2 + b ^ 2 = c ^ 2

theorem hyperbola_equation (h1 : system_eq1 a b c) (h2 : system_eq2 a b c) (h3 : system_eq3 a b c) :
  ∃ a b : ℝ, c = 5 ∧ b^2 = 20 ∧ a^2 = 5 ∧ (∀ x y : ℝ, (x ^ 2 / 5) - (y ^ 2 / 20) = 1) :=
  sorry

end hyperbola_equation_l740_74046


namespace jerome_bought_last_month_l740_74050

-- Definitions representing the conditions in the problem
def total_toy_cars_now := 40
def original_toy_cars := 25
def bought_this_month (bought_last_month : ℕ) := 2 * bought_last_month

-- The main statement to prove
theorem jerome_bought_last_month : ∃ x : ℕ, original_toy_cars + x + bought_this_month x = total_toy_cars_now ∧ x = 5 :=
by
  sorry

end jerome_bought_last_month_l740_74050


namespace rectangular_frame_wire_and_paper_area_l740_74067

theorem rectangular_frame_wire_and_paper_area :
  let l1 := 3
  let l2 := 4
  let l3 := 5
  let wire_length := (l1 + l2 + l3) * 4
  let paper_area := ((l1 * l2) + (l1 * l3) + (l2 * l3)) * 2
  wire_length = 48 ∧ paper_area = 94 :=
by
  sorry

end rectangular_frame_wire_and_paper_area_l740_74067


namespace Danny_found_11_wrappers_l740_74076

theorem Danny_found_11_wrappers :
  ∃ wrappers_at_park : ℕ,
  (wrappers_at_park = 11) ∧
  (∃ bottle_caps : ℕ, bottle_caps = 12) ∧
  (∃ found_bottle_caps : ℕ, found_bottle_caps = 58) ∧
  (wrappers_at_park + 1 = bottle_caps) :=
by
  sorry

end Danny_found_11_wrappers_l740_74076


namespace simplify_and_evaluate_l740_74061

theorem simplify_and_evaluate (m : ℝ) (h : m = 1) : 
  (1 - 1 / (m - 2)) / ((m^2 - 6 * m + 9) / (m - 2)) = -1/2 :=
by
  sorry

end simplify_and_evaluate_l740_74061


namespace nine_point_five_minutes_in_seconds_l740_74056

-- Define the number of seconds in one minute
def seconds_per_minute : ℝ := 60

-- Define the function to convert minutes to seconds
def minutes_to_seconds (minutes : ℝ) : ℝ :=
  minutes * seconds_per_minute

-- Define the theorem to prove
theorem nine_point_five_minutes_in_seconds : minutes_to_seconds 9.5 = 570 :=
by
  sorry

end nine_point_five_minutes_in_seconds_l740_74056


namespace evaluate_expression_l740_74097

theorem evaluate_expression : (- (1 / 4))⁻¹ - (Real.pi - 3)^0 - |(-4 : ℝ)| + (-1)^(2021 : ℕ) = -10 := 
by
  sorry

end evaluate_expression_l740_74097


namespace number_of_street_trees_l740_74043

-- Definitions from conditions
def road_length : ℕ := 1500
def interval_distance : ℕ := 25

-- The statement to prove
theorem number_of_street_trees : (road_length / interval_distance) + 1 = 61 := 
by
  unfold road_length
  unfold interval_distance
  sorry

end number_of_street_trees_l740_74043


namespace batsman_highest_score_l740_74060

theorem batsman_highest_score (H L : ℕ) 
  (h₁ : (40 * 50 = 2000)) 
  (h₂ : (H = L + 172))
  (h₃ : (38 * 48 = 1824)) :
  (2000 = 1824 + H + L) → H = 174 :=
by 
  sorry

end batsman_highest_score_l740_74060


namespace sunscreen_cost_l740_74024

theorem sunscreen_cost (reapply_time : ℕ) (oz_per_application : ℕ) 
  (oz_per_bottle : ℕ) (cost_per_bottle : ℝ) (total_time : ℕ) (expected_cost : ℝ) :
  reapply_time = 2 →
  oz_per_application = 3 →
  oz_per_bottle = 12 →
  cost_per_bottle = 3.5 →
  total_time = 16 →
  expected_cost = 7 →
  (total_time / reapply_time) * (oz_per_application / oz_per_bottle) * cost_per_bottle = expected_cost :=
by
  intros
  sorry

end sunscreen_cost_l740_74024


namespace men_women_equal_after_city_Y_l740_74094

variable (M W M' W' : ℕ)

-- Initial conditions: total passengers, women to men ratio
variable (h1 : M + W = 72)
variable (h2 : W = M / 2)

-- Changes in city Y: men leave, women enter
variable (h3 : M' = M - 16)
variable (h4 : W' = W + 8)

theorem men_women_equal_after_city_Y (h1 : M + W = 72) (h2 : W = M / 2) (h3 : M' = M - 16) (h4 : W' = W + 8) : 
  M' = W' := 
by 
  sorry

end men_women_equal_after_city_Y_l740_74094


namespace compare_P_Q_l740_74033

noncomputable def P (n : ℕ) (x : ℝ) : ℝ := (1 - x)^(2*n - 1)
noncomputable def Q (n : ℕ) (x : ℝ) : ℝ := 1 - (2*n - 1)*x + (n - 1)*(2*n - 1)*x^2

theorem compare_P_Q :
  ∀ (n : ℕ) (x : ℝ), n > 0 →
  ((n = 1 → P n x = Q n x) ∧
   (n = 2 → ((x = 0 → P n x = Q n x) ∧ (x > 0 → P n x < Q n x) ∧ (x < 0 → P n x > Q n x))) ∧
   (n ≥ 3 → ((x > 0 → P n x < Q n x) ∧ (x < 0 → P n x > Q n x)))) :=
by
  intros
  sorry

end compare_P_Q_l740_74033


namespace quadrilateral_AB_length_l740_74096

/-- Let ABCD be a quadrilateral with BC = CD = DA = 1, ∠DAB = 135°, and ∠ABC = 75°. 
    Prove that AB = (√6 - √2) / 2.
-/
theorem quadrilateral_AB_length (BC CD DA : ℝ) (angle_DAB angle_ABC : ℝ) (h1 : BC = 1)
    (h2 : CD = 1) (h3 : DA = 1) (h4 : angle_DAB = 135) (h5 : angle_ABC = 75) :
    AB = (Real.sqrt 6 - Real.sqrt 2) / 2 := by
    sorry

end quadrilateral_AB_length_l740_74096


namespace soup_can_pyramid_rows_l740_74036

theorem soup_can_pyramid_rows (n : ℕ) :
  (∃ (n : ℕ), (2 * n^2 - n = 225)) → n = 11 :=
by
  sorry

end soup_can_pyramid_rows_l740_74036


namespace rachel_earnings_without_tips_l740_74093

theorem rachel_earnings_without_tips
  (num_people : ℕ) (tip_per_person : ℝ) (total_earnings : ℝ)
  (h1 : num_people = 20)
  (h2 : tip_per_person = 1.25)
  (h3 : total_earnings = 37) :
  total_earnings - (num_people * tip_per_person) = 12 :=
by
  sorry

end rachel_earnings_without_tips_l740_74093


namespace apples_remaining_l740_74048

variable (initial_apples : ℕ)
variable (picked_day1 : ℕ)
variable (picked_day2 : ℕ)
variable (picked_day3 : ℕ)

-- Given conditions
def condition1 : initial_apples = 200 := sorry
def condition2 : picked_day1 = initial_apples / 5 := sorry
def condition3 : picked_day2 = 2 * picked_day1 := sorry
def condition4 : picked_day3 = picked_day1 + 20 := sorry

-- Prove the total number of apples remaining is 20
theorem apples_remaining (H1 : initial_apples = 200) 
  (H2 : picked_day1 = initial_apples / 5) 
  (H3 : picked_day2 = 2 * picked_day1)
  (H4 : picked_day3 = picked_day1 + 20) : 
  initial_apples - (picked_day1 + picked_day2 + picked_day3) = 20 := 
by
  sorry

end apples_remaining_l740_74048


namespace find_purchase_price_minimum_number_of_speed_skating_shoes_l740_74042

/-
A certain school in Zhangjiakou City is preparing to purchase speed skating shoes and figure skating shoes to promote ice and snow activities on campus.

If they buy 30 pairs of speed skating shoes and 20 pairs of figure skating shoes, the total cost is $8500.
If they buy 40 pairs of speed skating shoes and 10 pairs of figure skating shoes, the total cost is $8000.
The school purchases a total of 50 pairs of both types of ice skates, and the total cost does not exceed $8900.
-/

def price_system (x y : ℝ) : Prop :=
  30 * x + 20 * y = 8500 ∧ 40 * x + 10 * y = 8000

def minimum_speed_skating_shoes (x y m : ℕ) : Prop :=
  150 * m + 200 * (50 - m) ≤ 8900

theorem find_purchase_price :
  ∃ x y : ℝ, price_system x y ∧ x = 150 ∧ y = 200 :=
by
  /- Proof goes here -/
  sorry

theorem minimum_number_of_speed_skating_shoes :
  ∃ m, minimum_speed_skating_shoes 150 200 m ∧ m = 22 :=
by
  /- Proof goes here -/
  sorry

end find_purchase_price_minimum_number_of_speed_skating_shoes_l740_74042


namespace watch_cost_price_l740_74091

theorem watch_cost_price (CP : ℝ) (h1 : (0.90 * CP) + 280 = 1.04 * CP) : CP = 2000 := 
by 
  sorry

end watch_cost_price_l740_74091


namespace angle_C_is_120_degrees_l740_74099

theorem angle_C_is_120_degrees (l m : ℝ) (A B C : ℝ) (hal : l = m) 
  (hA : A = 100) (hB : B = 140) : C = 120 := 
by 
  sorry

end angle_C_is_120_degrees_l740_74099


namespace cylinder_section_volume_l740_74035

theorem cylinder_section_volume (a : ℝ) :
  let volume := (π * a^3 / 4)
  let section1_volume := volume * (1 / 3)
  let section2_volume := volume * (1 / 4)
  let enclosed_volume := (section1_volume - section2_volume) / 2
  enclosed_volume = π * a^3 / 24 := by
  sorry

end cylinder_section_volume_l740_74035


namespace remainder_of_3_pow_17_mod_7_l740_74011

theorem remainder_of_3_pow_17_mod_7 :
  (3^17 % 7) = 5 :=
by 
  sorry

end remainder_of_3_pow_17_mod_7_l740_74011


namespace slope_of_AB_l740_74077

theorem slope_of_AB (k : ℝ) (y1 y2 x1 x2 : ℝ) 
  (hP : (1, Real.sqrt 2) ∈ {p : ℝ × ℝ | p.2^2 = 2*p.1})
  (hPA_eq : ∀ x, (x, y1) ∈ {p : ℝ × ℝ | p.2 = k * p.1 - k + Real.sqrt 2 ∧ p.2^2 = 2 * p.1}) 
  (hPB_eq : ∀ x, (x, y2) ∈ {p : ℝ × ℝ | p.2 = -k * p.1 + k + Real.sqrt 2 ∧ p.2^2 = 2 * p.1}) 
  (hx1 : y1 = k * x1 - k + Real.sqrt 2) 
  (hx2 : y2 = -k * x2 + k + Real.sqrt 2) :
  ((y2 - y1) / (x2 - x1)) = -2 - 2 * Real.sqrt 2 :=
by
  sorry

end slope_of_AB_l740_74077


namespace carol_maximizes_at_0_75_l740_74049

def winning_probability (a b c : ℝ) : Prop :=
(0 ≤ a ∧ a ≤ 1) ∧ (0.25 ≤ b ∧ b ≤ 0.75) ∧ (a < c ∧ c < b ∨ b < c ∧ c < a)

theorem carol_maximizes_at_0_75 :
  ∀ (a b : ℝ), (0 ≤ a ∧ a ≤ 1) → (0.25 ≤ b ∧ b ≤ 0.75) → (∃ c : ℝ, 0 ≤ c ∧ c ≤ 1 ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → winning_probability a b x ≤ winning_probability a b 0.75)) :=
sorry

end carol_maximizes_at_0_75_l740_74049


namespace isabel_initial_amount_l740_74080

theorem isabel_initial_amount (X : ℝ) (h : X / 2 - X / 4 = 51) : X = 204 :=
sorry

end isabel_initial_amount_l740_74080


namespace max_three_numbers_condition_l740_74070

theorem max_three_numbers_condition (n : ℕ) 
  (x : Fin n → ℝ) 
  (h : ∀ (i j k : Fin n), i ≠ j → i ≠ k → j ≠ k → (x i)^2 > (x j) * (x k)) : n ≤ 3 := 
sorry

end max_three_numbers_condition_l740_74070


namespace child_running_speed_on_still_sidewalk_l740_74013

theorem child_running_speed_on_still_sidewalk (c s : ℕ) 
  (h1 : c + s = 93) 
  (h2 : c - s = 55) : c = 74 :=
sorry

end child_running_speed_on_still_sidewalk_l740_74013


namespace pond_volume_l740_74004

theorem pond_volume {L W H : ℝ} (hL : L = 20) (hW : W = 12) (hH : H = 5) : L * W * H = 1200 := by
  sorry

end pond_volume_l740_74004


namespace coefficients_divisible_by_7_l740_74034

theorem coefficients_divisible_by_7 
  {a b c d e : ℤ}
  (h : ∀ x : ℤ, (a * x^4 + b * x^3 + c * x^2 + d * x + e) % 7 = 0) :
  ∃ k l m n o : ℤ, a = 7*k ∧ b = 7*l ∧ c = 7*m ∧ d = 7*n ∧ e = 7*o :=
by
  sorry

end coefficients_divisible_by_7_l740_74034


namespace problem_1_problem_2_problem_3_problem_4_l740_74040

theorem problem_1 : 3 * Real.sqrt 20 - Real.sqrt 45 - Real.sqrt (1/5) = 14 * Real.sqrt 5 / 5 :=
by sorry

theorem problem_2 : (Real.sqrt 6 * Real.sqrt 3) / Real.sqrt 2 - 1 = 2 :=
by sorry

theorem problem_3 : Real.sqrt 16 + 327 - 2 * Real.sqrt (1/4) = 330 :=
by sorry

theorem problem_4 : (Real.sqrt 3 - Real.sqrt 5) * (Real.sqrt 5 + Real.sqrt 3) - (Real.sqrt 5 - Real.sqrt 3) ^ 2 = 2 * Real.sqrt 15 - 6 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l740_74040


namespace average_letters_per_day_l740_74072

theorem average_letters_per_day:
  let letters_per_day := [7, 10, 3, 5, 12]
  (letters_per_day.sum / letters_per_day.length : ℝ) = 7.4 :=
by
  sorry

end average_letters_per_day_l740_74072


namespace max_value_of_x_squared_plus_xy_plus_y_squared_l740_74071

theorem max_value_of_x_squared_plus_xy_plus_y_squared
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 - x * y + y^2 = 9) : 
  (x^2 + x * y + y^2) ≤ 27 :=
sorry

end max_value_of_x_squared_plus_xy_plus_y_squared_l740_74071


namespace evaluate_powers_of_i_mod_4_l740_74027

theorem evaluate_powers_of_i_mod_4 :
  (Complex.I ^ 48 + Complex.I ^ 96 + Complex.I ^ 144) = 3 := by
  sorry

end evaluate_powers_of_i_mod_4_l740_74027


namespace nat_lemma_l740_74018

theorem nat_lemma (a b : ℕ) : (∃ k : ℕ, (a + b^2) * (b + a^2) = 2^k) → (a = 1 ∧ b = 1) := by
  sorry

end nat_lemma_l740_74018


namespace sin_neg_five_sixths_pi_l740_74021

theorem sin_neg_five_sixths_pi : Real.sin (- 5 / 6 * Real.pi) = -1 / 2 :=
sorry

end sin_neg_five_sixths_pi_l740_74021


namespace odd_function_f_value_l740_74041

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^3 + x + 1 else x^3 + x - 1

theorem odd_function_f_value : 
  f 2 = 9 := by
  sorry

end odd_function_f_value_l740_74041


namespace regular_polygon_sides_l740_74002

theorem regular_polygon_sides (θ : ℝ) (n : ℕ) (h1 : θ = 18) (h2 : 360 = θ * n) : n = 20 :=
by {
  sorry
}

end regular_polygon_sides_l740_74002


namespace parabola_points_relationship_l740_74059

theorem parabola_points_relationship :
  let y_1 := (-2)^2 + 2 * (-2) - 9
  let y_2 := 1^2 + 2 * 1 - 9
  let y_3 := 3^2 + 2 * 3 - 9
  y_3 > y_2 ∧ y_2 > y_1 :=
by
  sorry

end parabola_points_relationship_l740_74059


namespace largest_of_choices_l740_74098

theorem largest_of_choices :
  let A := 24680 + (1 / 13579)
  let B := 24680 - (1 / 13579)
  let C := 24680 * (1 / 13579)
  let D := 24680 / (1 / 13579)
  let E := 24680.13579
  A < D ∧ B < D ∧ C < D ∧ E < D :=
by
  let A := 24680 + (1 / 13579)
  let B := 24680 - (1 / 13579)
  let C := 24680 * (1 / 13579)
  let D := 24680 / (1 / 13579)
  let E := 24680.13579
  sorry

end largest_of_choices_l740_74098


namespace perpendicular_lines_condition_l740_74079

theorem perpendicular_lines_condition (a : ℝ) :
  (¬ a = 1/2 ∨ ¬ a = -1/2) ∧ a * (-4 * a) = -1 ↔ a = 1/2 :=
by
  sorry

end perpendicular_lines_condition_l740_74079


namespace proof_correct_option_C_l740_74063

def line := Type
def plane := Type
def perp (m : line) (α : plane) : Prop := sorry
def parallel (n : line) (α : plane) : Prop := sorry
def perpnal (m n: line): Prop := sorry 

variables (m n : line) (α β γ : plane)

theorem proof_correct_option_C : perp m α → parallel n α → perpnal m n := sorry

end proof_correct_option_C_l740_74063


namespace toms_profit_l740_74062

noncomputable def cost_of_flour : Int :=
  let flour_needed := 500
  let bag_size := 50
  let bag_cost := 20
  (flour_needed / bag_size) * bag_cost

noncomputable def cost_of_salt : Int :=
  let salt_needed := 10
  let salt_cost_per_pound := (2 / 10)  -- Represent $0.2 as a fraction to maintain precision with integers in Lean
  salt_needed * salt_cost_per_pound

noncomputable def total_expenses : Int :=
  let flour_cost := cost_of_flour
  let salt_cost := cost_of_salt
  let promotion_cost := 1000
  flour_cost + salt_cost + promotion_cost

noncomputable def revenue_from_tickets : Int :=
  let ticket_price := 20
  let tickets_sold := 500
  tickets_sold * ticket_price

noncomputable def profit : Int :=
  revenue_from_tickets - total_expenses

theorem toms_profit : profit = 8798 :=
  by
    sorry

end toms_profit_l740_74062


namespace part_1_part_2_l740_74086

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def B (m : ℝ) : Set ℝ := { x | x^2 - (2*m + 1)*x + 2*m < 0 }

theorem part_1 (m : ℝ) (h : m < 1/2) : 
  B m = { x | 2*m < x ∧ x < 1 } := 
sorry

theorem part_2 (m : ℝ) : 
  (A ∪ B m = A) ↔ -1/2 ≤ m ∧ m ≤ 1 := 
sorry

end part_1_part_2_l740_74086


namespace incorrect_proposition_example_l740_74022

theorem incorrect_proposition_example (p q : Prop) (h : ¬ (p ∧ q)) : ¬ (¬p ∧ ¬q) :=
by
  sorry

end incorrect_proposition_example_l740_74022


namespace A_sym_diff_B_l740_74078

-- Definitions of sets and operations
def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def sym_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M

def A : Set ℝ := {y | ∃ x : ℝ, y = 3^x}
def B : Set ℝ := {y | ∃ x : ℝ, y = -(x-1)^2 + 2}

-- The target equality to prove
theorem A_sym_diff_B : sym_diff A B = (({y | y ≤ 0}) ∪ ({y | y > 2})) :=
by
  sorry

end A_sym_diff_B_l740_74078


namespace number_of_values_l740_74003

/-- Given:
  - The mean of some values was 190.
  - One value 165 was wrongly copied as 130 for the computation of the mean.
  - The correct mean is 191.4.
  Prove: the total number of values is 25. --/
theorem number_of_values (n : ℕ) (h₁ : (190 : ℝ) = ((190 * n) - (165 - 130)) / n) (h₂ : (191.4 : ℝ) = ((190 * n + 35) / n)) : n = 25 :=
sorry

end number_of_values_l740_74003


namespace arithmetic_sequence_a5_l740_74054

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : a 1 + a 9 = 10) : a 5 = 5 :=
sorry

end arithmetic_sequence_a5_l740_74054


namespace day_after_1999_cubed_days_is_tuesday_l740_74029

theorem day_after_1999_cubed_days_is_tuesday : 
    let today := "Monday"
    let days_in_week := 7
    let target_days := 1999 ^ 3
    ∃ remaining_days, remaining_days = (target_days % days_in_week) ∧ today = "Monday" ∧ remaining_days = 1 → 
    "Tuesday" = "Tuesday" := 
by
  sorry

end day_after_1999_cubed_days_is_tuesday_l740_74029


namespace area_enclosed_by_absolute_value_linear_eq_l740_74073

theorem area_enclosed_by_absolute_value_linear_eq (x y : ℝ) :
  (|5 * x| + |3 * y| = 15) → ∃ (A : ℝ), A = 30 :=
by
  sorry

end area_enclosed_by_absolute_value_linear_eq_l740_74073


namespace total_cost_of_apples_l740_74089

def original_price_per_pound : ℝ := 1.6
def price_increase_percentage : ℝ := 0.25
def number_of_family_members : ℕ := 4
def pounds_per_person : ℝ := 2

theorem total_cost_of_apples : 
  let new_price_per_pound := original_price_per_pound * (1 + price_increase_percentage)
  let total_pounds := pounds_per_person * number_of_family_members
  let total_cost := total_pounds * new_price_per_pound
  total_cost = 16 := by
  sorry

end total_cost_of_apples_l740_74089


namespace chris_average_price_l740_74023

noncomputable def total_cost_dvd (price_per_dvd : ℝ) (num_dvds : ℕ) (discount : ℝ) : ℝ :=
  (price_per_dvd * (1 - discount)) * num_dvds

noncomputable def total_cost_bluray (price_per_bluray : ℝ) (num_blurays : ℕ) : ℝ :=
  price_per_bluray * num_blurays

noncomputable def total_cost_ultra_hd (price_per_ultra_hd : ℝ) (num_ultra_hds : ℕ) : ℝ :=
  price_per_ultra_hd * num_ultra_hds

noncomputable def total_cost (cost_dvd cost_bluray cost_ultra_hd : ℝ) : ℝ :=
  cost_dvd + cost_bluray + cost_ultra_hd

noncomputable def total_with_tax (total_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  total_cost * (1 + tax_rate)

noncomputable def average_price (total_with_tax : ℝ) (total_movies : ℕ) : ℝ :=
  total_with_tax / total_movies

theorem chris_average_price :
  let price_per_dvd := 15
  let num_dvds := 5
  let discount := 0.20
  let price_per_bluray := 20
  let num_blurays := 8
  let price_per_ultra_hd := 25
  let num_ultra_hds := 3
  let tax_rate := 0.10
  let total_movies := num_dvds + num_blurays + num_ultra_hds
  let cost_dvd := total_cost_dvd price_per_dvd num_dvds discount
  let cost_bluray := total_cost_bluray price_per_bluray num_blurays
  let cost_ultra_hd := total_cost_ultra_hd price_per_ultra_hd num_ultra_hds
  let pre_tax_total := total_cost cost_dvd cost_bluray cost_ultra_hd
  let total := total_with_tax pre_tax_total tax_rate
  average_price total total_movies = 20.28 :=
by
  -- substitute each definition one step at a time
  -- to show the average price exactly matches 20.28
  sorry

end chris_average_price_l740_74023


namespace mart_income_percentage_juan_l740_74032

-- Define the conditions
def TimIncomeLessJuan (J T : ℝ) : Prop := T = 0.40 * J
def MartIncomeMoreTim (T M : ℝ) : Prop := M = 1.60 * T

-- Define the proof problem
theorem mart_income_percentage_juan (J T M : ℝ) 
  (h1 : TimIncomeLessJuan J T) 
  (h2 : MartIncomeMoreTim T M) :
  M = 0.64 * J := 
  sorry

end mart_income_percentage_juan_l740_74032


namespace radius_increase_area_triple_l740_74065

theorem radius_increase_area_triple (r m : ℝ) (h : π * (r + m)^2 = 3 * π * r^2) : 
  r = (m * (Real.sqrt 3 - 1)) / 2 := 
sorry

end radius_increase_area_triple_l740_74065


namespace playground_ball_cost_l740_74058

-- Define the given conditions
def cost_jump_rope : ℕ := 7
def cost_board_game : ℕ := 12
def saved_by_dalton : ℕ := 6
def given_by_uncle : ℕ := 13
def additional_needed : ℕ := 4

-- Total money Dalton has
def total_money : ℕ := saved_by_dalton + given_by_uncle

-- Total cost needed to buy all three items
def total_cost_needed : ℕ := total_money + additional_needed

-- Combined cost of the jump rope and the board game
def combined_cost : ℕ := cost_jump_rope + cost_board_game

-- Prove the cost of the playground ball
theorem playground_ball_cost : ℕ := total_cost_needed - combined_cost

-- Expected result
example : playground_ball_cost = 4 := by
  sorry

end playground_ball_cost_l740_74058


namespace intersection_of_sets_is_closed_interval_l740_74088

noncomputable def A := {x : ℝ | x ≤ 0 ∨ x ≥ 2}
noncomputable def B := {x : ℝ | x < 1}

theorem intersection_of_sets_is_closed_interval :
  A ∩ B = {x : ℝ | x ≤ 0} :=
sorry

end intersection_of_sets_is_closed_interval_l740_74088


namespace find_speed_second_train_l740_74083

noncomputable def speed_second_train (length_train1 length_train2 : ℝ) (speed_train1_kmph : ℝ) (time_to_cross : ℝ) : ℝ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let total_distance := length_train1 + length_train2
  let relative_speed_mps := total_distance / time_to_cross
  let speed_train2_mps := speed_train1_mps - relative_speed_mps
  speed_train2_mps * 3600 / 1000

theorem find_speed_second_train :
  speed_second_train 380 540 72 91.9926405887529 = 36 := by
  sorry

end find_speed_second_train_l740_74083


namespace sum_of_squares_of_roots_l740_74001

theorem sum_of_squares_of_roots (x1 x2 : ℝ) 
    (h1 : 2 * x1^2 + 3 * x1 - 5 = 0) 
    (h2 : 2 * x2^2 + 3 * x2 - 5 = 0)
    (h3 : x1 + x2 = -3 / 2)
    (h4 : x1 * x2 = -5 / 2) : 
    x1^2 + x2^2 = 29 / 4 :=
by
  sorry

end sum_of_squares_of_roots_l740_74001


namespace lcm_12_18_24_l740_74019

theorem lcm_12_18_24 : Nat.lcm (Nat.lcm 12 18) 24 = 72 := by
  -- Given conditions (prime factorizations)
  have h1 : 12 = 2^2 * 3 := by norm_num
  have h2 : 18 = 2 * 3^2 := by norm_num
  have h3 : 24 = 2^3 * 3 := by norm_num
  -- Prove the LCM
  sorry

end lcm_12_18_24_l740_74019


namespace isabella_hair_length_l740_74000

theorem isabella_hair_length (final_length growth_length initial_length : ℕ) 
  (h1 : final_length = 24) 
  (h2 : growth_length = 6) 
  (h3 : final_length = initial_length + growth_length) : 
  initial_length = 18 :=
by
  sorry

end isabella_hair_length_l740_74000


namespace skate_time_correct_l740_74039

noncomputable def skate_time (path_length miles_length : ℝ) (skating_speed : ℝ) : ℝ :=
  let time_taken := (1.58 * Real.pi) / skating_speed
  time_taken

theorem skate_time_correct :
  skate_time 1 1 4 = 1.58 * Real.pi / 4 :=
by
  sorry

end skate_time_correct_l740_74039


namespace rachel_should_budget_940_l740_74064

-- Define the prices for Sara's shoes and dress
def sara_shoes : ℝ := 50
def sara_dress : ℝ := 200

-- Define the prices for Tina's shoes and dress
def tina_shoes : ℝ := 70
def tina_dress : ℝ := 150

-- Define the total spending for Sara and Tina, and Rachel's budget
def rachel_budget (sara_shoes sara_dress tina_shoes tina_dress : ℝ) : ℝ := 
  2 * (sara_shoes + sara_dress + tina_shoes + tina_dress)

theorem rachel_should_budget_940 : 
  rachel_budget sara_shoes sara_dress tina_shoes tina_dress = 940 := 
by
  -- skip the proof
  sorry 

end rachel_should_budget_940_l740_74064


namespace age_of_other_replaced_man_l740_74084

variable (A B C : ℕ)
variable (B_new1 B_new2 : ℕ)
variable (avg_old avg_new : ℕ)

theorem age_of_other_replaced_man (hB : B = 23) 
    (h_avg_new : (B_new1 + B_new2) / 2 = 25)
    (h_avg_inc : (A + B_new1 + B_new2) / 3 > (A + B + C) / 3) : 
    C = 26 := 
  sorry

end age_of_other_replaced_man_l740_74084


namespace sqrt_div_l740_74085

theorem sqrt_div (x: ℕ) (h1: Nat.sqrt 144 * Nat.sqrt 144 = 144) (h2: 144 = 12 * 12) (h3: 2 * x = 12) : x = 6 :=
sorry

end sqrt_div_l740_74085


namespace triangle_isosceles_or_right_angled_l740_74053

theorem triangle_isosceles_or_right_angled
  (β γ : ℝ)
  (h : Real.tan β * Real.sin γ ^ 2 = Real.tan γ * Real.sin β ^ 2) :
  (β = γ) ∨ (β + γ = π / 2) :=
sorry

end triangle_isosceles_or_right_angled_l740_74053


namespace simplify_division_l740_74087

noncomputable def simplify_expression (m : ℝ) : ℝ :=
  (m^2 - 3 * m + 1) / m + 1

noncomputable def divisor_expression (m : ℝ) : ℝ :=
  (m^2 - 1) / m

theorem simplify_division (m : ℝ) (hm1 : m ≠ 0) (hm2 : m ≠ 1) (hm3 : m ≠ -1) :
  (simplify_expression m) / (divisor_expression m) = (m - 1) / (m + 1) :=
by {
  sorry
}

end simplify_division_l740_74087


namespace axis_of_symmetry_parabola_l740_74066

theorem axis_of_symmetry_parabola : ∀ (x y : ℝ), y = 2 * x^2 → x = 0 :=
by
  sorry

end axis_of_symmetry_parabola_l740_74066


namespace retailer_initial_thought_profit_percentage_l740_74016

/-
  An uneducated retailer marks all his goods at 60% above the cost price and thinking that he will still make some profit, 
  offers a discount of 25% on the marked price. 
  His actual profit on the sales is 20.000000000000018%. 
  Prove that the profit percentage the retailer initially thought he would make is 60%.
-/

theorem retailer_initial_thought_profit_percentage
  (cost_price marked_price selling_price : ℝ)
  (h1 : marked_price = cost_price + 0.6 * cost_price)
  (h2 : selling_price = marked_price - 0.25 * marked_price)
  (h3 : selling_price - cost_price = 0.20000000000000018 * cost_price) :
  0.6 * 100 = 60 := by
  sorry

end retailer_initial_thought_profit_percentage_l740_74016


namespace number_of_poles_needed_l740_74005

def length := 90
def width := 40
def distance_between_poles := 5

noncomputable def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem number_of_poles_needed (l w d : ℕ) : perimeter l w / d = 52 :=
by
  rw [perimeter]
  sorry

end number_of_poles_needed_l740_74005


namespace cards_relationship_l740_74020

-- Definitions from the conditions given in the problem
variables (x y : ℕ)

-- Theorem statement proving the relationship
theorem cards_relationship (h : x + y = 8 * x) : y = 7 * x :=
sorry

end cards_relationship_l740_74020


namespace solve_inequality_x_squared_minus_6x_gt_15_l740_74045

theorem solve_inequality_x_squared_minus_6x_gt_15 :
  { x : ℝ | x^2 - 6 * x > 15 } = { x : ℝ | x < -1.5 } ∪ { x : ℝ | x > 7.5 } :=
by
  sorry

end solve_inequality_x_squared_minus_6x_gt_15_l740_74045


namespace part1_part2_l740_74014

noncomputable section
open Real

section
variables {x A a b c : ℝ}
variables {k : ℤ}

def f (x : ℝ) : ℝ := sin (2 * x - (π / 6)) + 2 * cos x ^ 2 - 1

theorem part1 (k : ℤ) : 
  ∀ x : ℝ, 
  k * π - (π / 3) ≤ x ∧ x ≤ k * π + (π / 6) → 
    ∀ x₁ x₂, 
      k * π - (π / 3) ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ k * π + (π / 6) → 
        f x₁ < f x₂ := sorry

theorem part2 {A a b c : ℝ} 
  (h_a_seq : 2 * a = b + c) 
  (h_dot : b * c * cos A = 9) 
  (h_A_fA : f A = 1 / 2) 
  : 
  a = 3 * sqrt 2 := sorry

end

end part1_part2_l740_74014


namespace factor_of_60n_l740_74057

theorem factor_of_60n
  (n : ℕ)
  (x : ℕ)
  (h_condition1 : ∃ k : ℕ, 60 * n = x * k)
  (h_condition2 : ∃ m : ℕ, 60 * n = 8 * m)
  (h_condition3 : n >= 8) :
  x = 60 :=
sorry

end factor_of_60n_l740_74057


namespace triangle_trig_identity_l740_74007

open Real

theorem triangle_trig_identity (A B C : ℝ) (h_triangle : A + B + C = 180) (h_A : A = 15) :
  sqrt 3 * sin A - cos (B + C) = sqrt 2 := by
  sorry

end triangle_trig_identity_l740_74007


namespace problem_statement_l740_74006

variable (x y : ℝ)
variable (h_cond1 : 1 / x + 1 / y = 4)
variable (h_cond2 : x * y - x - y = -7)

theorem problem_statement (h_cond1 : 1 / x + 1 / y = 4) (h_cond2 : x * y - x - y = -7) : 
  x^2 * y + x * y^2 = 196 / 9 := 
sorry

end problem_statement_l740_74006
