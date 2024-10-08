import Mathlib

namespace minimum_value_proof_l16_16906

noncomputable def minimum_value (x : ℝ) (h : x > 1) : ℝ :=
  (x^2 + x + 1) / (x - 1)

theorem minimum_value_proof : ∃ x : ℝ, x > 1 ∧ minimum_value x (by sorry) = 3 + 2*Real.sqrt 3 :=
sorry

end minimum_value_proof_l16_16906


namespace maximum_marks_l16_16231

theorem maximum_marks (passing_percentage : ℝ) (score : ℝ) (shortfall : ℝ) (total_marks : ℝ) : 
  passing_percentage = 30 → 
  score = 212 → 
  shortfall = 16 → 
  total_marks = (score + shortfall) * 100 / passing_percentage → 
  total_marks = 760 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  assumption

end maximum_marks_l16_16231


namespace bullet_trains_crossing_time_l16_16481

theorem bullet_trains_crossing_time
  (length_train1 : ℝ) (length_train2 : ℝ)
  (speed_train1_km_hr : ℝ) (speed_train2_km_hr : ℝ)
  (opposite_directions : Prop)
  (h_length1 : length_train1 = 140)
  (h_length2 : length_train2 = 170)
  (h_speed1 : speed_train1_km_hr = 60)
  (h_speed2 : speed_train2_km_hr = 40)
  (h_opposite : opposite_directions = true) :
  ∃ t : ℝ, t = 11.16 :=
by
  sorry

end bullet_trains_crossing_time_l16_16481


namespace solution_pairs_correct_l16_16531

theorem solution_pairs_correct:
  { (n, m) : ℕ × ℕ | m^2 + 2 * 3^n = m * (2^(n+1) - 1) }
  = {(3, 6), (3, 9), (6, 54), (6, 27)} :=
by
  sorry -- no proof is required as per the instruction

end solution_pairs_correct_l16_16531


namespace probability_larry_wins_l16_16656

noncomputable def P_larry_wins_game : ℝ :=
  let p_hit := (1 : ℝ) / 3
  let p_miss := (2 : ℝ) / 3
  let r := p_miss^3
  (p_hit / (1 - r))

theorem probability_larry_wins :
  P_larry_wins_game = 9 / 19 :=
by
  -- Proof is omitted, but the outline and logic are given in the problem statement
  sorry

end probability_larry_wins_l16_16656


namespace poly_division_l16_16507

noncomputable def A := 1
noncomputable def B := 3
noncomputable def C := 2
noncomputable def D := -1

theorem poly_division :
  (∀ x : ℝ, x ≠ -1 → (x^3 + 4*x^2 + 5*x + 2) / (x+1) = x^2 + 3*x + 2) ∧
  (A + B + C + D = 5) :=
by
  sorry

end poly_division_l16_16507


namespace time_to_write_all_rearrangements_l16_16889

-- Define the problem conditions
def sophie_name_length := 6
def rearrangements_per_minute := 18

-- Define the factorial function for calculating permutations
noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the total number of rearrangements of Sophie's name
noncomputable def total_rearrangements := factorial sophie_name_length

-- Define the time in minutes to write all rearrangements
noncomputable def time_in_minutes := total_rearrangements / rearrangements_per_minute

-- Convert the time to hours
noncomputable def minutes_to_hours (minutes : ℕ) : ℚ := minutes / 60

-- Prove the time in hours to write all the rearrangements
theorem time_to_write_all_rearrangements : minutes_to_hours time_in_minutes = (2 : ℚ) / 3 := 
  sorry

end time_to_write_all_rearrangements_l16_16889


namespace behavior_of_f_in_interval_l16_16440

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 3 * m * x + 3

-- Define the property of even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- The theorem statement
theorem behavior_of_f_in_interval (m : ℝ) (hf_even : is_even_function (f m)) :
  m = 0 → (∀ x : ℝ, -4 < x ∧ x < 0 → f 0 x < f 0 (-x)) ∧ (∀ x : ℝ, 0 < x ∧ x < 2 → f 0 (-x) > f 0 x) :=
by 
  sorry

end behavior_of_f_in_interval_l16_16440


namespace part1_part2_l16_16680

-- Define the parabola C as y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line l with slope k passing through point P(-2, 1)
def line (x y k : ℝ) : Prop := y - 1 = k * (x + 2)

-- Part 1: Prove the range of k for which line l intersects parabola C at two points is -1 < k < -1/2
theorem part1 (k : ℝ) : 
  (∃ x y, parabola x y ∧ line x y k) ∧ (∃ u v, parabola u v ∧ u ≠ x ∧ line u v k) ↔ -1 < k ∧ k < -1/2 := sorry

-- Part 2: Prove the equations of line l when it intersects parabola C at only one point are y = 1, y = -x - 1, and y = -1/2 x
theorem part2 (k : ℝ) : 
  (∃! x y, parabola x y ∧ line x y k) ↔ (k = 0 ∨ k = -1 ∨ k = -1/2) := sorry

end part1_part2_l16_16680


namespace nancy_carrots_next_day_l16_16626

-- Definitions based on conditions
def carrots_picked_on_first_day : Nat := 12
def carrots_thrown_out : Nat := 2
def total_carrots_after_two_days : Nat := 31

-- Problem statement
theorem nancy_carrots_next_day :
  let carrots_left_after_first_day := carrots_picked_on_first_day - carrots_thrown_out
  let carrots_picked_next_day := total_carrots_after_two_days - carrots_left_after_first_day
  carrots_picked_next_day = 21 :=
by
  sorry

end nancy_carrots_next_day_l16_16626


namespace certain_fraction_exists_l16_16663

theorem certain_fraction_exists (a b : ℚ) (h : a / b = 3 / 4) :
  (a / b) / (1 / 5) = (3 / 4) / (2 / 5) :=
by
  sorry

end certain_fraction_exists_l16_16663


namespace tony_fever_temperature_above_threshold_l16_16438

theorem tony_fever_temperature_above_threshold 
  (n : ℕ) (i : ℕ) (f : ℕ) 
  (h1 : n = 95) (h2 : i = 10) (h3 : f = 100) : 
  n + i - f = 5 :=
by
  sorry

end tony_fever_temperature_above_threshold_l16_16438


namespace max_possible_value_l16_16682

-- Define the expressions and the conditions
def expr1 := 10 * 10
def expr2 := 10 / 10
def expr3 := expr1 + 10
def expr4 := expr3 - expr2

-- Define our main statement that asserts the maximum value is 109
theorem max_possible_value: expr4 = 109 := by
  sorry

end max_possible_value_l16_16682


namespace find_k_l16_16117

theorem find_k (k : ℕ) (hk : k > 0) (h_coeff : 15 * k^4 < 120) : k = 1 := 
by 
  sorry

end find_k_l16_16117


namespace rational_squares_solution_l16_16129

theorem rational_squares_solution {x y u v : ℕ} (x_pos : 0 < x) (y_pos : 0 < y) (u_pos : 0 < u) (v_pos : 0 < v) 
  (h1 : ∃ q : ℚ, q = (Real.sqrt (x * y) + Real.sqrt (u * v))) 
  (h2 : |(x / 9 : ℚ) - (y / 4 : ℚ)| = |(u / 3 : ℚ) - (v / 12 : ℚ)| ∧ |(u / 3 : ℚ) - (v / 12 : ℚ)| = u * v - x * y) :
  ∃ k : ℕ, x = 9 * k ∧ y = 4 * k ∧ u = 3 * k ∧ v = 12 * k := by
  sorry

end rational_squares_solution_l16_16129


namespace easter_eggs_total_l16_16112

theorem easter_eggs_total (h he total : ℕ)
 (hannah_eggs : h = 42) 
 (twice_he : h = 2 * he) 
 (total_eggs : total = h + he) : 
 total = 63 := 
sorry

end easter_eggs_total_l16_16112


namespace exists_acute_triangle_l16_16391

theorem exists_acute_triangle (a b c d e : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (h_triangle_abc : a + b > c) (h_triangle_abd : a + b > d) (h_triangle_abe : a + b > e)
  (h_triangle_bcd : b + c > d) (h_triangle_bce : b + c > e) (h_triangle_cde : c + d > e)
  (h_triangle_abc2 : a + c > b) (h_triangle_abd2 : a + d > b) (h_triangle_abe2 : a + e > b)
  (h_triangle_bcd2 : b + d > c) (h_triangle_bce2 : b + e > c) (h_triangle_cde2 : c + e > d)
  (h_triangle_abc3 : b + c > a) (h_triangle_abd3 : b + d > a) (h_triangle_abe3 : b + e > a)
  (h_triangle_bcd3 : b + d > a) (h_triangle_bce3 : c + e > a) (h_triangle_cde3 : d + e > c) :
  ∃ x y z : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧ 
              (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧ 
              (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
              (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z) ∧
              x + y > z ∧ 
              ¬ (x^2 + y^2 ≤ z^2) :=
by
  sorry

end exists_acute_triangle_l16_16391


namespace union_of_sets_l16_16535

def A := { x : ℝ | 3 ≤ x ∧ x < 7 }
def B := { x : ℝ | 2 < x ∧ x < 10 }

theorem union_of_sets (x : ℝ) : (x ∈ A ∪ B) ↔ (x ∈ { x : ℝ | 2 < x ∧ x < 10 }) :=
by
  sorry

end union_of_sets_l16_16535


namespace correct_option_B_l16_16479

theorem correct_option_B (x y a b : ℝ) :
  (3 * x + 2 * x^2 ≠ 5 * x) →
  (-y^2 * x + x * y^2 = 0) →
  (-a * b - a * b ≠ 0) →
  (3 * a^3 * b^2 - 2 * a^3 * b^2 ≠ 1) →
  (-y^2 * x + x * y^2 = 0) :=
by
  intros hA hB hC hD
  exact hB

end correct_option_B_l16_16479


namespace linear_function_behavior_l16_16386

theorem linear_function_behavior (x y : ℝ) (h : y = -3 * x + 6) :
  ∀ x1 x2 : ℝ, x1 < x2 → (y = -3 * x1 + 6) → (y = -3 * x2 + 6) → -3 * (x1 - x2) > 0 :=
by
  sorry

end linear_function_behavior_l16_16386


namespace license_plate_count_l16_16861

theorem license_plate_count : 
  let consonants := 20
  let vowels := 6
  let digits := 10
  4 * consonants * vowels * consonants * digits = 24000 :=
by
  sorry

end license_plate_count_l16_16861


namespace inequality_proof_l16_16326

theorem inequality_proof (x y z : ℝ) (h : x * y * z + x + y + z = 4) : 
    (y * z + 6)^2 + (z * x + 6)^2 + (x * y + 6)^2 ≥ 8 * (x * y * z + 5) :=
by
  sorry

end inequality_proof_l16_16326


namespace delivery_driver_stops_l16_16983

theorem delivery_driver_stops (initial_stops more_stops total_stops : ℕ)
  (h_initial : initial_stops = 3)
  (h_more : more_stops = 4)
  (h_total : total_stops = initial_stops + more_stops) : total_stops = 7 := by
  sorry

end delivery_driver_stops_l16_16983


namespace initial_men_count_l16_16555

theorem initial_men_count (x : ℕ) (h : x * 25 = 15 * 60) : x = 36 :=
by
  sorry

end initial_men_count_l16_16555


namespace find_c_value_l16_16866

theorem find_c_value (b c : ℝ) 
  (h1 : 1 + b + c = 4) 
  (h2 : 25 + 5 * b + c = 4) : 
  c = 9 :=
by
  sorry

end find_c_value_l16_16866


namespace thomas_needs_more_money_l16_16632

-- Define the conditions in Lean
def weeklyAllowance : ℕ := 50
def hourlyWage : ℕ := 9
def hoursPerWeek : ℕ := 30
def weeklyExpenses : ℕ := 35
def weeksInYear : ℕ := 52
def carCost : ℕ := 15000

-- Define the total earnings for the first year
def firstYearEarnings : ℕ :=
  weeklyAllowance * weeksInYear

-- Define the weekly earnings from the second year job
def secondYearWeeklyEarnings : ℕ :=
  hourlyWage * hoursPerWeek

-- Define the total earnings for the second year
def secondYearEarnings : ℕ :=
  secondYearWeeklyEarnings * weeksInYear

-- Define the total earnings over two years
def totalEarnings : ℕ :=
  firstYearEarnings + secondYearEarnings

-- Define the total expenses over two years
def totalExpenses : ℕ :=
  weeklyExpenses * (2 * weeksInYear)

-- Define the net savings after two years
def netSavings : ℕ :=
  totalEarnings - totalExpenses

-- Define the amount more needed for the car
def amountMoreNeeded : ℕ :=
  carCost - netSavings

-- The theorem to prove
theorem thomas_needs_more_money : amountMoreNeeded = 2000 := by
  sorry

end thomas_needs_more_money_l16_16632


namespace prod_three_consec_cubemultiple_of_504_l16_16967

theorem prod_three_consec_cubemultiple_of_504 (a : ℤ) : (a^3 - 1) * a^3 * (a^3 + 1) % 504 = 0 := by
  sorry

end prod_three_consec_cubemultiple_of_504_l16_16967


namespace greatest_common_divisor_of_B_l16_16304

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l16_16304


namespace roger_steps_time_l16_16388

theorem roger_steps_time (steps_per_30_min : ℕ := 2000) (time_for_2000_steps : ℕ := 30) (goal_steps : ℕ := 10000) : 
  (goal_steps * time_for_2000_steps) / steps_per_30_min = 150 :=
by 
  -- This is the statement. Proof is omitted as per instruction.
  sorry

end roger_steps_time_l16_16388


namespace monotonic_decreasing_interval_l16_16655

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < (1 / 2) → (0 < x ∧ x < (1 / 2)) ∧ (f (1 / 2) - f x) > 0 :=
sorry

end monotonic_decreasing_interval_l16_16655


namespace sqrt_9_is_pm3_l16_16237

theorem sqrt_9_is_pm3 : {x : ℝ | x ^ 2 = 9} = {3, -3} := sorry

end sqrt_9_is_pm3_l16_16237


namespace feet_of_wood_required_l16_16437

def rung_length_in_inches : ℤ := 18
def spacing_between_rungs_in_inches : ℤ := 6
def height_to_climb_in_feet : ℤ := 50

def feet_per_rung := rung_length_in_inches / 12
def rungs_per_foot := 12 / spacing_between_rungs_in_inches
def total_rungs := height_to_climb_in_feet * rungs_per_foot
def total_feet_of_wood := total_rungs * feet_per_rung

theorem feet_of_wood_required :
  total_feet_of_wood = 150 :=
by
  sorry

end feet_of_wood_required_l16_16437


namespace percentage_shaded_in_square_l16_16446

theorem percentage_shaded_in_square
  (EFGH : Type)
  (square : EFGH → Prop)
  (side_length : EFGH → ℝ)
  (area : EFGH → ℝ)
  (shaded_area : EFGH → ℝ)
  (P : EFGH)
  (h_square : square P)
  (h_side_length : side_length P = 8)
  (h_area : area P = side_length P * side_length P)
  (h_small_shaded : shaded_area P = 4)
  (h_large_shaded : shaded_area P + 7 = 11) :
  (shaded_area P / area P) * 100 = 17.1875 :=
by
  sorry

end percentage_shaded_in_square_l16_16446


namespace v2_correct_at_2_l16_16593

def poly (x : ℕ) : ℕ := x^5 + x^4 + 2 * x^3 + 3 * x^2 + 4 * x + 1

def horner_v2 (x : ℕ) : ℕ :=
  let v0 := 1
  let v1 := v0 * x + 4
  let v2 := v1 * x + 3
  v2

theorem v2_correct_at_2 : horner_v2 2 = 15 := by
  sorry

end v2_correct_at_2_l16_16593


namespace no_integer_solutions_l16_16685

theorem no_integer_solutions (x y z : ℤ) : ¬ (x^2 + y^2 = 3 * z^2) :=
sorry

end no_integer_solutions_l16_16685


namespace time_first_tap_to_fill_cistern_l16_16007

-- Defining the conditions
axiom second_tap_empty_time : ℝ
axiom combined_tap_fill_time : ℝ
axiom second_tap_rate : ℝ
axiom combined_tap_rate : ℝ

-- Specifying the given conditions
def problem_conditions :=
  second_tap_empty_time = 8 ∧
  combined_tap_fill_time = 8 ∧
  second_tap_rate = 1 / 8 ∧
  combined_tap_rate = 1 / 8

-- Defining the problem statement
theorem time_first_tap_to_fill_cistern :
  problem_conditions →
  (∃ T : ℝ, (1 / T - 1 / 8 = 1 / 8) ∧ T = 4) :=
by
  intro h
  sorry

end time_first_tap_to_fill_cistern_l16_16007


namespace side_length_correct_l16_16991

noncomputable def find_side_length (b : ℝ) (angleB : ℝ) (sinA : ℝ) : ℝ :=
  let sinB := Real.sin angleB
  let a := b * sinA / sinB
  a

theorem side_length_correct (b : ℝ) (angleB : ℝ) (sinA : ℝ) (a : ℝ) 
  (hb : b = 4)
  (hangleB : angleB = Real.pi / 6)
  (hsinA : sinA = 1 / 3)
  (ha : a = 8 / 3) : 
  find_side_length b angleB sinA = a :=
by
  sorry

end side_length_correct_l16_16991


namespace no_such_function_exists_l16_16286

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = Real.sin x :=
by
  sorry

end no_such_function_exists_l16_16286


namespace possible_ways_to_choose_gates_l16_16662

theorem possible_ways_to_choose_gates : 
  ∃! (ways : ℕ), ways = 20 := 
by
  sorry

end possible_ways_to_choose_gates_l16_16662


namespace right_triangle_hypotenuse_l16_16495

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h1 : a + b + c = 60) 
  (h2 : 0.5 * a * b = 120) 
  (h3 : a^2 + b^2 = c^2) : 
  c = 26 :=
by {
  sorry
}

end right_triangle_hypotenuse_l16_16495


namespace sequence_property_l16_16769

theorem sequence_property (a : ℕ → ℝ)
    (h_rec : ∀ n ≥ 2, a n = a (n - 1) * a (n + 1))
    (h_a1 : a 1 = 1 + Real.sqrt 7)
    (h_1776 : a 1776 = 13 + Real.sqrt 7) :
    a 2009 = -1 + 2 * Real.sqrt 7 := 
    sorry

end sequence_property_l16_16769


namespace circumradius_inradius_perimeter_inequality_l16_16335

open Real

variables {R r P : ℝ} -- circumradius, inradius, perimeter
variable (triangle_type : String) -- acute, obtuse, right

def satisfies_inequality (R r P : ℝ) (triangle_type : String) : Prop :=
  if triangle_type = "right" then
    R ≥ (sqrt 2) / 2 * sqrt (P * r)
  else
    R ≥ (sqrt 3) / 3 * sqrt (P * r)

theorem circumradius_inradius_perimeter_inequality :
  ∀ (R r P : ℝ) (triangle_type : String), satisfies_inequality R r P triangle_type :=
by 
  intros R r P triangle_type
  sorry -- proof steps go here

end circumradius_inradius_perimeter_inequality_l16_16335


namespace necessary_but_not_sufficient_condition_l16_16021

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, x = 1 → x^2 - 3 * x + 2 = 0) ∧ (∃ x : ℝ, x^2 - 3 * x + 2 = 0 ∧ x ≠ 1) :=
by
  sorry

end necessary_but_not_sufficient_condition_l16_16021


namespace ratio_AR_AU_l16_16384

-- Define the conditions in the problem as variables and constraints
variables (A B C P Q U R : Type)
variables (AP PB AQ QC : ℝ)
variables (angle_bisector_AU : A -> U)
variables (intersect_AU_PQ_at_R : A -> U -> P -> Q -> R)

-- Assuming the given distances
def conditions (AP PB AQ QC : ℝ) : Prop :=
  AP = 2 ∧ PB = 6 ∧ AQ = 4 ∧ QC = 5

-- The statement to prove
theorem ratio_AR_AU (h : conditions AP PB AQ QC) : 
  (AR / AU) = 108 / 289 :=
sorry

end ratio_AR_AU_l16_16384


namespace Eva_is_16_l16_16295

def Clara_age : ℕ := 12
def Nora_age : ℕ := Clara_age + 3
def Liam_age : ℕ := Nora_age - 4
def Eva_age : ℕ := Liam_age + 5

theorem Eva_is_16 : Eva_age = 16 := by
  sorry

end Eva_is_16_l16_16295


namespace correct_statement_3_l16_16645

-- Definitions
def acute_angles (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def angles_less_than_90 (θ : ℝ) : Prop := θ < 90
def angles_in_first_quadrant (θ : ℝ) : Prop := ∃ k : ℤ, k * 360 < θ ∧ θ < k * 360 + 90

-- Sets
def M := {θ | acute_angles θ}
def N := {θ | angles_less_than_90 θ}
def P := {θ | angles_in_first_quadrant θ}

-- Proof statement
theorem correct_statement_3 : M ⊆ P := sorry

end correct_statement_3_l16_16645


namespace gcd_2197_2208_is_1_l16_16707

def gcd_2197_2208 : ℕ := Nat.gcd 2197 2208

theorem gcd_2197_2208_is_1 : gcd_2197_2208 = 1 :=
by
  sorry

end gcd_2197_2208_is_1_l16_16707


namespace cos_neg_13pi_div_4_l16_16068

theorem cos_neg_13pi_div_4 : (Real.cos (-13 * Real.pi / 4)) = -Real.sqrt 2 / 2 := 
by sorry

end cos_neg_13pi_div_4_l16_16068


namespace calculate_correctly_l16_16434

theorem calculate_correctly (n : ℕ) (h1 : n - 21 = 52) : n - 40 = 33 := 
by 
  sorry

end calculate_correctly_l16_16434


namespace min_value_at_2_l16_16964

noncomputable def min_value (x : ℝ) := x + 4 / x + 5

theorem min_value_at_2 (x : ℝ) (h : x > 0) : min_value x ≥ 9 :=
sorry

end min_value_at_2_l16_16964


namespace expression_value_l16_16725

theorem expression_value :
  (1 / (3 - (1 / (3 + (1 / (3 - (1 / 3))))))) = (27 / 73) :=
by 
  sorry

end expression_value_l16_16725


namespace sqrt_multiplication_l16_16163

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l16_16163


namespace range_of_a_if_solution_non_empty_l16_16014

variable (f : ℝ → ℝ) (a : ℝ)

/-- Given that the solution set of f(x) < | -1 | is non-empty,
    we need to prove that |a| ≥ 4. -/
theorem range_of_a_if_solution_non_empty (h : ∃ x, f x < 1) : |a| ≥ 4 :=
sorry

end range_of_a_if_solution_non_empty_l16_16014


namespace increasing_interval_f_l16_16145

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (Real.pi / 6))

theorem increasing_interval_f : ∃ a b : ℝ, a < b ∧ 
  (∀ x y : ℝ, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x < y → f x < f y) ∧
  (a = - (Real.pi / 6)) ∧ (b = (Real.pi / 3)) :=
by
  sorry

end increasing_interval_f_l16_16145


namespace balloons_left_after_distribution_l16_16181

theorem balloons_left_after_distribution :
  (22 + 40 + 70 + 90) % 10 = 2 := by
  sorry

end balloons_left_after_distribution_l16_16181


namespace quadratic_has_two_distinct_real_roots_l16_16587

-- Definitions of the conditions
def a : ℝ := 1
def b (k : ℝ) : ℝ := -3 * k
def c : ℝ := -2

-- Definition of the discriminant function
def discriminant (k : ℝ) : ℝ := (b k) ^ 2 - 4 * a * c

-- Logical statement to be proved
theorem quadratic_has_two_distinct_real_roots (k : ℝ) : discriminant k > 0 :=
by
  unfold discriminant
  unfold b a c
  simp
  sorry

end quadratic_has_two_distinct_real_roots_l16_16587


namespace sum_first_3000_terms_l16_16202

variable {α : Type*}

noncomputable def geometric_sum_1000 (a r : α) [Field α] : α := a * (r ^ 1000 - 1) / (r - 1)
noncomputable def geometric_sum_2000 (a r : α) [Field α] : α := a * (r ^ 2000 - 1) / (r - 1)
noncomputable def geometric_sum_3000 (a r : α) [Field α] : α := a * (r ^ 3000 - 1) / (r - 1)

theorem sum_first_3000_terms 
  {a r : ℝ}
  (h1 : geometric_sum_1000 a r = 1024)
  (h2 : geometric_sum_2000 a r = 2040) :
  geometric_sum_3000 a r = 3048 := 
  sorry

end sum_first_3000_terms_l16_16202


namespace find_root_D_l16_16239

/-- Given C and D are roots of the polynomial k x^2 + 2 x + 5 = 0, 
    and k = -1/4 and C = 10, then D must be -2. -/
theorem find_root_D 
  (k : ℚ) (C D : ℚ)
  (h1 : k = -1/4)
  (h2 : C = 10)
  (h3 : C^2 * k + 2 * C + 5 = 0)
  (h4 : D^2 * k + 2 * D + 5 = 0) : 
  D = -2 :=
by
  sorry

end find_root_D_l16_16239


namespace solve_trig_equation_proof_l16_16256

noncomputable def solve_trig_equation (θ : ℝ) : Prop :=
  2 * Real.cos θ ^ 2 - 5 * Real.cos θ + 2 = 0 ∧ (θ = 60 / 180 * Real.pi)

theorem solve_trig_equation_proof (θ : ℝ) :
  solve_trig_equation θ :=
sorry

end solve_trig_equation_proof_l16_16256


namespace length_of_second_train_l16_16739

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (cross_time : ℝ)
  (opposite_directions : Bool) :
  speed_first_train = 120 / 3.6 →
  speed_second_train = 80 / 3.6 →
  cross_time = 9 →
  length_first_train = 260 →
  opposite_directions = true →
  ∃ (length_second_train : ℝ), length_second_train = 240 :=
by
  sorry

end length_of_second_train_l16_16739


namespace total_worth_of_travelers_checks_l16_16534

theorem total_worth_of_travelers_checks (x y : ℕ) (h1 : x + y = 30) (h2 : 50 * (x - 18) + 100 * y = 900) : 
  50 * x + 100 * y = 1800 := 
by
  sorry

end total_worth_of_travelers_checks_l16_16534


namespace find_y_value_l16_16144

theorem find_y_value (y : ℕ) (h1 : y ≤ 150)
  (h2 : (45 + 76 + 123 + y + y + y) / 6 = 2 * y) :
  y = 27 :=
sorry

end find_y_value_l16_16144


namespace min_value_l16_16596

noncomputable def f (x : ℝ) : ℝ := 2017 * x + Real.sin (x / 2018) + (2019 ^ x - 1) / (2019 ^ x + 1)

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : f (2 * a) + f (b - 4) = 0) :
  2 * a + b = 4 → (1 / a + 2 / b) = 2 :=
by sorry

end min_value_l16_16596


namespace coffee_merchant_mixture_price_l16_16905

theorem coffee_merchant_mixture_price
  (c1 c2 : ℝ) (w1 w2 total_cost mixture_price : ℝ)
  (h_c1 : c1 = 9)
  (h_c2 : c2 = 12)
  (h_w1w2 : w1 = 25 ∧ w2 = 25)
  (h_total_weight : w1 + w2 = 100)
  (h_total_cost : total_cost = w1 * c1 + w2 * c2)
  (h_mixture_price : mixture_price = total_cost / (w1 + w2)) :
  mixture_price = 5.25 :=
by sorry

end coffee_merchant_mixture_price_l16_16905


namespace sufficient_condition_for_inequality_l16_16955

theorem sufficient_condition_for_inequality (m : ℝ) (h : m ≠ 0) : (m > 2) → (m + 4 / m > 4) :=
by
  sorry

end sufficient_condition_for_inequality_l16_16955


namespace number_of_yellow_parrots_l16_16103

theorem number_of_yellow_parrots (total_parrots : ℕ) (red_fraction : ℚ) 
  (h_total_parrots : total_parrots = 108) 
  (h_red_fraction : red_fraction = 5 / 6) : 
  ∃ (yellow_parrots : ℕ), yellow_parrots = total_parrots * (1 - red_fraction) ∧ yellow_parrots = 18 := 
by
  sorry

end number_of_yellow_parrots_l16_16103


namespace imaginary_part_of_complex_division_l16_16483

theorem imaginary_part_of_complex_division : 
  let i := Complex.I
  let z := (1 - 2 * i) / (2 - i)
  Complex.im z = -3 / 5 :=
by
  sorry

end imaginary_part_of_complex_division_l16_16483


namespace arithmetic_sequence_l16_16997

theorem arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n + 1) : 
  ∀ n, a (n + 1) - a n = 3 := by
  sorry

end arithmetic_sequence_l16_16997


namespace find_h_l16_16542

def f (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

theorem find_h : ∃ a h k, (h = -3 / 2) ∧ (f x = a * (x - h)^2 + k) :=
by
  -- Proof steps would go here
  sorry

end find_h_l16_16542


namespace total_tickets_l16_16536

theorem total_tickets (A C total_tickets total_cost : ℕ) 
  (adult_ticket_cost : ℕ := 8) (child_ticket_cost : ℕ := 5) 
  (total_cost_paid : ℕ := 201) (child_tickets_count : ℕ := 21) 
  (ticket_cost_eqn : 8 * A + 5 * 21 = 201) 
  (adult_tickets_count : A = total_cost_paid - (child_ticket_cost * child_tickets_count) / adult_ticket_cost) :
  total_tickets = A + child_tickets_count :=
sorry

end total_tickets_l16_16536


namespace larger_number_of_hcf_lcm_is_322_l16_16859

theorem larger_number_of_hcf_lcm_is_322
  (A B : ℕ)
  (hcf: ℕ := 23)
  (factor1 : ℕ := 13)
  (factor2 : ℕ := 14)
  (hcf_condition : ∀ d, d ∣ A → d ∣ B → d ≤ hcf)
  (lcm_condition : ∀ m n, m * n = A * B → m = factor1 * hcf ∨ m = factor2 * hcf) :
  max A B = 322 :=
by sorry

end larger_number_of_hcf_lcm_is_322_l16_16859


namespace vertical_asymptote_sum_l16_16881

theorem vertical_asymptote_sum :
  ∀ x y : ℝ, (4 * x^2 + 8 * x + 3 = 0) → (4 * y^2 + 8 * y + 3 = 0) → x ≠ y → x + y = -2 :=
by
  sorry

end vertical_asymptote_sum_l16_16881


namespace value_of_x_squared_plus_y_squared_l16_16134

theorem value_of_x_squared_plus_y_squared
  (x y : ℝ)
  (h1 : (x + y)^2 = 4)
  (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
sorry

end value_of_x_squared_plus_y_squared_l16_16134


namespace part_a_l16_16370

theorem part_a (x : ℝ) : 1 + (1 / (2 + 1 / ((4 * x + 1) / (2 * x + 1) - 1 / (2 + 1 / x)))) = 19 / 14 ↔ x = 1 / 2 := sorry

end part_a_l16_16370


namespace minimum_value_xy_minimum_value_x_plus_2y_l16_16229

-- (1) Prove that the minimum value of \(xy\) given \(x, y > 0\) and \(\frac{1}{x} + \frac{9}{y} = 1\) is \(36\).
theorem minimum_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 9 / y = 1) : x * y ≥ 36 := 
sorry

-- (2) Prove that the minimum value of \(x + 2y\) given \(x, y > 0\) and \(\frac{1}{x} + \frac{9}{y} = 1\) is \(19 + 6\sqrt{2}\).
theorem minimum_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 9 / y = 1) : x + 2 * y ≥ 19 + 6 * Real.sqrt 2 := 
sorry

end minimum_value_xy_minimum_value_x_plus_2y_l16_16229


namespace negq_sufficient_but_not_necessary_for_p_l16_16978

variable (p q : Prop)

theorem negq_sufficient_but_not_necessary_for_p
  (h1 : ¬p → q)
  (h2 : ¬(¬q → p)) :
  (¬q → p) ∧ ¬(p → ¬q) :=
sorry

end negq_sufficient_but_not_necessary_for_p_l16_16978


namespace fruit_bowl_apples_l16_16301

theorem fruit_bowl_apples (A : ℕ) (total_oranges initial_oranges remaining_oranges : ℕ) (percentage_apples : ℝ) :
  total_oranges = 20 →
  initial_oranges = total_oranges →
  remaining_oranges = initial_oranges - 14 →
  percentage_apples = 0.70 →
  percentage_apples * (A + remaining_oranges) = A →
  A = 14 :=
by 
  intro h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end fruit_bowl_apples_l16_16301


namespace problem_statement_l16_16760

noncomputable def a := 9
noncomputable def b := 729

theorem problem_statement (h1 : ∃ (terms : ℕ), terms = 430)
                          (h2 : ∃ (value : ℕ), value = 3) : a + b = 738 :=
by
  sorry

end problem_statement_l16_16760


namespace fraction_equivalence_l16_16896

theorem fraction_equivalence : (8 : ℝ) / (5 * 48) = 0.8 / (5 * 0.48) :=
  sorry

end fraction_equivalence_l16_16896


namespace parallel_lines_find_m_l16_16306

theorem parallel_lines_find_m (m : ℝ) :
  (((3 + m) / 2 = 4 / (5 + m)) ∧ ((3 + m) / 2 ≠ (5 - 3 * m) / 8)) → m = -7 :=
sorry

end parallel_lines_find_m_l16_16306


namespace abs_h_of_roots_sum_squares_eq_34_l16_16594

theorem abs_h_of_roots_sum_squares_eq_34 
  (h : ℝ)
  (h_eq : ∀ r s : ℝ, (2 * r^2 + 4 * h * r + 6 = 0) ∧ (2 * s^2 + 4 * h * s + 6 = 0)) 
  (sum_of_squares_eq : ∀ r s : ℝ, (2 * r^2 + 4 * h * r + 6 = 0) ∧ (2 * s^2 + 4 * h * s + 6 = 0) → r^2 + s^2 = 34) :
  |h| = Real.sqrt 10 :=
by
  sorry

end abs_h_of_roots_sum_squares_eq_34_l16_16594


namespace relationship_among_terms_l16_16334

theorem relationship_among_terms (a : ℝ) (h : a ^ 2 + a < 0) : 
  -a > a ^ 2 ∧ a ^ 2 > -a ^ 2 ∧ -a ^ 2 > a :=
sorry

end relationship_among_terms_l16_16334


namespace draw_points_value_l16_16824

theorem draw_points_value
  (D : ℕ) -- Let D be the number of points for a draw
  (victory_points : ℕ := 3) -- points for a victory
  (defeat_points : ℕ := 0) -- points for a defeat
  (total_matches : ℕ := 20) -- total matches
  (points_after_5_games : ℕ := 8) -- points scored in the first 5 games
  (minimum_wins_remaining : ℕ := 9) -- at least 9 matches should be won in the remaining matches
  (target_points : ℕ := 40) : -- target points by the end of the tournament
  D = 1 := 
by 
  sorry


end draw_points_value_l16_16824


namespace exponent_property_l16_16504

theorem exponent_property (a x y : ℝ) (h1 : 0 < a) (h2 : a ^ x = 2) (h3 : a ^ y = 3) : a ^ (x - y) = 2 / 3 := 
by
  sorry

end exponent_property_l16_16504


namespace sum_of_arithmetic_sequence_l16_16806

theorem sum_of_arithmetic_sequence (a d1 d2 : ℕ) 
  (h1 : d1 = d2 + 2) 
  (h2 : d1 + d2 = 24) 
  (a_pos : 0 < a) : 
  (a + (a + d1) + (a + d1) + (a + d1 + d2) = 54) := 
by 
  sorry

end sum_of_arithmetic_sequence_l16_16806


namespace max_value_x_plus_2y_max_of_x_plus_2y_l16_16805

def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 / 4 = 1

theorem max_value_x_plus_2y (x y : ℝ) (h : on_ellipse x y) :
  x + 2 * y ≤ Real.sqrt 22 :=
sorry

theorem max_of_x_plus_2y (x y : ℝ) (h : on_ellipse x y) :
  ∃θ ∈ Set.Icc 0 (2 * Real.pi), (x = Real.sqrt 6 * Real.cos θ) ∧ (y = 2 * Real.sin θ) :=
sorry

end max_value_x_plus_2y_max_of_x_plus_2y_l16_16805


namespace distinct_triple_identity_l16_16432

theorem distinct_triple_identity (p q r : ℝ) 
  (h1 : p ≠ q) 
  (h2 : q ≠ r) 
  (h3 : r ≠ p)
  (h : (p / (q - r)) + (q / (r - p)) + (r / (p - q)) = 3) : 
  (p^2 / (q - r)^2) + (q^2 / (r - p)^2) + (r^2 / (p - q)^2) = 3 :=
by 
  sorry

end distinct_triple_identity_l16_16432


namespace total_workers_in_workshop_l16_16151

theorem total_workers_in_workshop 
  (W : ℕ)
  (T : ℕ := 5)
  (avg_all : ℕ := 700)
  (avg_technicians : ℕ := 800)
  (avg_rest : ℕ := 650) 
  (total_salary_all : ℕ := W * avg_all)
  (total_salary_technicians : ℕ := T * avg_technicians)
  (total_salary_rest : ℕ := (W - T) * avg_rest) :
  total_salary_all = total_salary_technicians + total_salary_rest →
  W = 15 :=
by
  sorry

end total_workers_in_workshop_l16_16151


namespace marians_groceries_l16_16400

variables (G : ℝ)

theorem marians_groceries :
  let initial_balance := 126
  let returned_amount := 45
  let new_balance := 171
  let gas_expense := G / 2
  initial_balance + G + gas_expense - returned_amount = new_balance → G = 60 :=
sorry

end marians_groceries_l16_16400


namespace pipe_Q_fill_time_l16_16794

theorem pipe_Q_fill_time (x : ℝ) (h1 : 6 > 0)
    (h2 : 24 > 0)
    (h3 : 3.4285714285714284 > 0)
    (h4 : (1 / 6) + (1 / x) + (1 / 24) = 1 / 3.4285714285714284) :
    x = 8 := by
  sorry

end pipe_Q_fill_time_l16_16794


namespace triangles_from_pentadecagon_l16_16517

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon
    is 455, given that there are 15 vertices and none of them are collinear. -/

theorem triangles_from_pentadecagon : (Nat.choose 15 3) = 455 := 
by
  sorry

end triangles_from_pentadecagon_l16_16517


namespace initial_volume_mixture_l16_16345

theorem initial_volume_mixture (V : ℝ) (h1 : 0.84 * V = 0.6 * (V + 24)) : V = 60 :=
by
  sorry

end initial_volume_mixture_l16_16345


namespace digit_d_is_six_l16_16404

theorem digit_d_is_six (d : ℕ) (h_even : d % 2 = 0) (h_digits_sum : 7 + 4 + 8 + 2 + d % 9 = 0) : d = 6 :=
by 
  sorry

end digit_d_is_six_l16_16404


namespace rectangular_prism_edges_vertices_faces_sum_l16_16449

theorem rectangular_prism_edges_vertices_faces_sum (a b c : ℕ) (h1: a = 2) (h2: b = 3) (h3: c = 4) : 
  12 + 8 + 6 = 26 :=
by
  sorry

end rectangular_prism_edges_vertices_faces_sum_l16_16449


namespace original_players_count_l16_16823

theorem original_players_count (n : ℕ) (W : ℕ) :
  (W = n * 103) →
  ((W + 110 + 60) = (n + 2) * 99) →
  n = 7 :=
by sorry

end original_players_count_l16_16823


namespace total_length_of_visible_edges_l16_16036

theorem total_length_of_visible_edges (shortest_side : ℕ) (removed_side : ℕ) (longest_side : ℕ) (new_visible_sides_sum : ℕ) 
  (h1 : shortest_side = 4) 
  (h2 : removed_side = 2 * shortest_side) 
  (h3 : removed_side = longest_side / 2) 
  (h4 : longest_side = 16) 
  (h5 : new_visible_sides_sum = shortest_side + removed_side + removed_side) : 
  new_visible_sides_sum = 20 := by 
sorry

end total_length_of_visible_edges_l16_16036


namespace work_completion_time_l16_16503

theorem work_completion_time (A B C D : Type) 
  (work_rate_A : ℚ := 1 / 10) 
  (work_rate_AB : ℚ := 1 / 5)
  (work_rate_C : ℚ := 1 / 15) 
  (work_rate_D : ℚ := 1 / 20) 
  (combined_work_rate_AB : work_rate_A + (work_rate_AB - work_rate_A) = 1 / 10) : 
  (1 / (work_rate_A + (work_rate_AB - work_rate_A) + work_rate_C + work_rate_D)) = 60 / 19 := 
sorry

end work_completion_time_l16_16503


namespace find_pairs_l16_16735

-- Define a function that checks if a pair (n, d) satisfies the required conditions
def satisfies_conditions (n d : ℕ) : Prop :=
  ∀ S : ℤ, ∃! (a : ℕ → ℤ), 
    (∀ i : ℕ, i < n → a i ≤ a (i + 1)) ∧                -- Non-decreasing sequence condition
    ((Finset.range n).sum a = S) ∧                  -- Sum of the sequence equals S
    (a n.succ.pred - a 0 = d)                      -- The difference condition

-- The formal statement of the required proof
theorem find_pairs :
  {p : ℕ × ℕ | satisfies_conditions p.fst p.snd} = {(1, 0), (3, 2)} :=
by
  sorry

end find_pairs_l16_16735


namespace plane_equation_l16_16106

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + s + 2 * t, 4 - 2 * s, 1 - s + t)

def normal_vector : ℝ × ℝ × ℝ :=
  (-2, -3, 4)

def point_on_plane : ℝ × ℝ × ℝ :=
  (3, 4, 1)

theorem plane_equation : ∀ (x y z : ℝ),
  (∃ (s t : ℝ), (x, y, z) = parametric_plane s t) ↔
  2 * x + 3 * y - 4 * z - 14 = 0 :=
sorry

end plane_equation_l16_16106


namespace transportation_inverse_proportion_l16_16954

theorem transportation_inverse_proportion (V t : ℝ) (h: V * t = 10^5) : V = 10^5 / t :=
by
  sorry

end transportation_inverse_proportion_l16_16954


namespace initial_kittens_l16_16132

theorem initial_kittens (kittens_given : ℕ) (kittens_left : ℕ) (initial_kittens : ℕ) :
  kittens_given = 4 → kittens_left = 4 → initial_kittens = kittens_given + kittens_left → initial_kittens = 8 :=
by
  intros hg hl hi
  rw [hg, hl] at hi
  -- Skipping proof detail
  sorry

end initial_kittens_l16_16132


namespace value_this_year_l16_16981

def last_year_value : ℝ := 20000
def depreciation_factor : ℝ := 0.8

theorem value_this_year :
  last_year_value * depreciation_factor = 16000 :=
by
  sorry

end value_this_year_l16_16981


namespace geometric_seq_a7_l16_16961

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l16_16961


namespace geometric_sequence_sum_l16_16339

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (a_pos : ∀ n, 0 < a n)
  (h_a2 : a 2 = 1) (h_a3a7_a5 : a 3 * a 7 - a 5 = 56)
  (S_eq : ∀ n, S n = (a 1 * (1 - (2 : ℝ) ^ n)) / (1 - 2)) :
  S 5 = 31 / 2 := by
  sorry

end geometric_sequence_sum_l16_16339


namespace correct_expression_l16_16216

-- Definitions for the problem options.
def optionA (m n : ℕ) : ℕ := 2 * m + n
def optionB (m n : ℕ) : ℕ := m + 2 * n
def optionC (m n : ℕ) : ℕ := 2 * (m + n)
def optionD (m n : ℕ) : ℕ := (m + n) ^ 2

-- Statement for the proof problem.
theorem correct_expression (m n : ℕ) : optionB m n = m + 2 * n :=
by sorry

end correct_expression_l16_16216


namespace points_earned_l16_16679

-- Define the given conditions
def points_per_enemy := 5
def total_enemies := 8
def enemies_remaining := 6

-- Calculate the number of enemies defeated
def enemies_defeated := total_enemies - enemies_remaining

-- Calculate the points earned based on the enemies defeated
theorem points_earned : enemies_defeated * points_per_enemy = 10 := by
  -- Insert mathematical operations
  sorry

end points_earned_l16_16679


namespace macaroon_weight_l16_16281

theorem macaroon_weight (bakes : ℕ) (packs : ℕ) (bags_after_eat : ℕ) (remaining_weight : ℕ) (macaroons_per_bag : ℕ) (weight_per_bag : ℕ)
  (H1 : bakes = 12) 
  (H2 : packs = 4)
  (H3 : bags_after_eat = 3)
  (H4 : remaining_weight = 45)
  (H5 : macaroons_per_bag = bakes / packs) 
  (H6 : weight_per_bag = remaining_weight / bags_after_eat) :
  ∀ (weight_per_macaroon : ℕ), weight_per_macaroon = weight_per_bag / macaroons_per_bag → weight_per_macaroon = 5 :=
by
  sorry -- Proof will come here, not required as per instructions

end macaroon_weight_l16_16281


namespace total_money_made_l16_16860

-- Define the conditions
def dollars_per_day : Int := 144
def number_of_days : Int := 22

-- State the proof problem
theorem total_money_made : (dollars_per_day * number_of_days = 3168) :=
by
  sorry

end total_money_made_l16_16860


namespace integer_solutions_l16_16819

theorem integer_solutions (a b c : ℤ) :
  a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  intros h
  sorry

end integer_solutions_l16_16819


namespace gravitational_force_solution_l16_16110

noncomputable def gravitational_force_proportionality (d d' : ℕ) (f f' k : ℝ) : Prop :=
  (f * (d:ℝ)^2 = k) ∧
  d = 6000 ∧
  f = 800 ∧
  d' = 36000 ∧
  f' * (d':ℝ)^2 = k

theorem gravitational_force_solution : ∃ k, gravitational_force_proportionality 6000 36000 800 (1/45) k :=
by
  sorry

end gravitational_force_solution_l16_16110


namespace expression_equals_neg_eight_l16_16509

variable {a b : ℝ}

theorem expression_equals_neg_eight (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : |a| ≠ |b|) :
  ( (b^2 / a^2 + a^2 / b^2 - 2) * 
    ((a + b) / (b - a) + (b - a) / (a + b)) * 
    (((1 / a^2 + 1 / b^2) / (1 / b^2 - 1 / a^2)) - ((1 / b^2 - 1 / a^2) / (1 / a^2 + 1 / b^2)))
  ) = -8 :=
by
  sorry

end expression_equals_neg_eight_l16_16509


namespace batsman_average_increase_l16_16485

theorem batsman_average_increase (A : ℕ) 
    (h1 : 15 * A + 64 = 19 * 16) 
    (h2 : 19 - A = 3) : 
    19 - A = 3 := 
sorry

end batsman_average_increase_l16_16485


namespace no_x2_term_a_eq_1_l16_16292

theorem no_x2_term_a_eq_1 (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a * x + 1) * (x^2 - 3 * a + 2) = x^4 + bx^3 + cx + d) →
  c = 0 →
  a = 1 :=
sorry

end no_x2_term_a_eq_1_l16_16292


namespace double_pythagorean_triple_l16_16172

theorem double_pythagorean_triple (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  (2*a)^2 + (2*b)^2 = (2*c)^2 :=
by
  sorry

end double_pythagorean_triple_l16_16172


namespace number_of_chickens_l16_16871

def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12
def full_cartons : ℕ := 10

theorem number_of_chickens :
  (full_cartons * eggs_per_carton) / eggs_per_chicken = 20 :=
by
  sorry

end number_of_chickens_l16_16871


namespace find_k_l16_16749

noncomputable def curve (x k : ℝ) : ℝ := x + k * Real.log (1 + x)

theorem find_k (k : ℝ) :
  let y' := (fun x => 1 + k / (1 + x))
  (y' 1 = 2) ∧ ((1 + 2 * 1) = 0) → k = 2 :=
by
  sorry

end find_k_l16_16749


namespace complex_frac_eq_l16_16845

theorem complex_frac_eq (a b : ℝ) (i : ℂ) (h : i^2 = -1)
  (h1 : (1 - i) / (1 + i) = a + b * i) : a - b = 1 :=
by
  sorry

end complex_frac_eq_l16_16845


namespace sequence_term_l16_16365

noncomputable def geometric_sum (n : ℕ) : ℝ :=
  2 * (1 - (1 / 2) ^ n) / (1 - 1 / 2)

theorem sequence_term (m n : ℕ) (h : n < m) : 
  let Sn := geometric_sum n
  let Sn_plus_1 := geometric_sum (n + 1)
  Sn - Sn_plus_1 = -(1 / 2 ^ (n - 1)) := sorry

end sequence_term_l16_16365


namespace standard_spherical_coordinates_l16_16069

theorem standard_spherical_coordinates :
  ∀ (ρ θ φ : ℝ), 
  ρ = 5 → θ = 3 * Real.pi / 4 → φ = 9 * Real.pi / 5 →
  (ρ > 0) →
  (0 ≤ θ ∧ θ < 2 * Real.pi) →
  (0 ≤ φ ∧ φ ≤ Real.pi) →
  (ρ, θ, φ) = (5, 7 * Real.pi / 4, Real.pi / 5) :=
by sorry

end standard_spherical_coordinates_l16_16069


namespace books_read_l16_16868

theorem books_read (total_books remaining_books read_books : ℕ)
  (h_total : total_books = 14)
  (h_remaining : remaining_books = 6)
  (h_eq : read_books = total_books - remaining_books) : read_books = 8 := 
by 
  sorry

end books_read_l16_16868


namespace algebraic_expression_value_l16_16102

theorem algebraic_expression_value (a b : ℝ) (h : a + b = 3) : a^3 + b^3 + 9 * a * b = 27 :=
by
  sorry

end algebraic_expression_value_l16_16102


namespace problem1_problem2_l16_16778

-- Problem 1
theorem problem1 : (-1 : ℤ) ^ 2024 + (1 / 3 : ℝ) ^ (-2 : ℤ) - (3.14 - Real.pi) ^ 0 = 9 := 
sorry

-- Problem 2
theorem problem2 (x : ℤ) (y : ℤ) (hx : x = 2) (hy : y = 3) : 
  x * (x + 2 * y) - (x + 1) ^ 2 + 2 * x = 11 :=
sorry

end problem1_problem2_l16_16778


namespace gcd_72_120_168_l16_16171

theorem gcd_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  -- Each step would be proven individually here.
  sorry

end gcd_72_120_168_l16_16171


namespace no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square_l16_16166

theorem no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square :
  ∀ b : ℤ, ¬ ∃ k : ℤ, b^2 + 3*b + 1 = k^2 :=
by
  sorry

end no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square_l16_16166


namespace gary_chickens_l16_16491

theorem gary_chickens (initial_chickens : ℕ) (multiplication_factor : ℕ) 
  (weekly_eggs : ℕ) (days_in_week : ℕ)
  (h1 : initial_chickens = 4)
  (h2 : multiplication_factor = 8)
  (h3 : weekly_eggs = 1344)
  (h4 : days_in_week = 7) :
  (weekly_eggs / days_in_week) / (initial_chickens * multiplication_factor) = 6 :=
by
  sorry

end gary_chickens_l16_16491


namespace percentage_of_water_in_fresh_grapes_l16_16766

theorem percentage_of_water_in_fresh_grapes
  (P : ℝ)  -- Let P be the percentage of water in fresh grapes
  (fresh_grapes_weight : ℝ := 5)  -- weight of fresh grapes in kg
  (dried_grapes_weight : ℝ := 0.625)  -- weight of dried grapes in kg
  (dried_water_percentage : ℝ := 20)  -- percentage of water in dried grapes
  (h1 : (100 - P) / 100 * fresh_grapes_weight = (100 - dried_water_percentage) / 100 * dried_grapes_weight) :
  P = 90 := 
sorry

end percentage_of_water_in_fresh_grapes_l16_16766


namespace find_k_values_l16_16698

/-- 
Prove that the values of k such that the positive difference between the 
roots of 3x^2 + 5x + k = 0 equals the sum of the squares of the roots 
are exactly (70 + 10sqrt(33))/8 and (70 - 10sqrt(33))/8.
-/
theorem find_k_values (k : ℝ) :
  (∀ (a b : ℝ), (3 * a^2 + 5 * a + k = 0 ∧ 3 * b^2 + 5 * b + k = 0 ∧ |a - b| = a^2 + b^2))
  ↔ (k = (70 + 10 * Real.sqrt 33) / 8 ∨ k = (70 - 10 * Real.sqrt 33) / 8) :=
sorry

end find_k_values_l16_16698


namespace bobby_initial_candy_count_l16_16377

theorem bobby_initial_candy_count (x : ℕ) (h : x + 17 = 43) : x = 26 :=
by
  sorry

end bobby_initial_candy_count_l16_16377


namespace student_chose_number_l16_16381

theorem student_chose_number (x : ℤ) (h : 2 * x - 138 = 104) : x = 121 := by
  sorry

end student_chose_number_l16_16381


namespace minor_premise_of_syllogism_l16_16836

theorem minor_premise_of_syllogism (P Q : Prop)
  (h1 : ¬ (P ∧ ¬ Q))
  (h2 : Q) :
  Q :=
by
  sorry

end minor_premise_of_syllogism_l16_16836


namespace mark_sprint_distance_l16_16275

theorem mark_sprint_distance (t v : ℝ) (ht : t = 24.0) (hv : v = 6.0) : 
  t * v = 144.0 := 
by
  -- This theorem is formulated with the conditions that t = 24.0 and v = 6.0,
  -- we need to prove that the resulting distance is 144.0 miles.
  sorry

end mark_sprint_distance_l16_16275


namespace maya_lift_increase_l16_16409

def initial_lift_America : ℕ := 240
def peak_lift_America : ℕ := 300

def initial_lift_Maya (a_lift : ℕ) : ℕ := a_lift / 4
def peak_lift_Maya (p_lift : ℕ) : ℕ := p_lift / 2

def lift_difference (initial_lift : ℕ) (peak_lift : ℕ) : ℕ := peak_lift - initial_lift

theorem maya_lift_increase :
  lift_difference (initial_lift_Maya initial_lift_America) (peak_lift_Maya peak_lift_America) = 90 :=
by
  -- Proof is skipped with sorry
  sorry

end maya_lift_increase_l16_16409


namespace greatest_sum_of_consecutive_odd_integers_lt_500_l16_16003

-- Define the consecutive odd integers and their conditions
def consecutive_odd_integers (n : ℤ) : Prop :=
  n % 2 = 1 ∧ (n + 2) % 2 = 1

-- Define the condition that their product must be less than 500
def prod_less_500 (n : ℤ) : Prop :=
  n * (n + 2) < 500

-- The theorem statement
theorem greatest_sum_of_consecutive_odd_integers_lt_500 : 
  ∃ n : ℤ, consecutive_odd_integers n ∧ prod_less_500 n ∧ ∀ m : ℤ, consecutive_odd_integers m ∧ prod_less_500 m → n + (n + 2) ≥ m + (m + 2) :=
sorry

end greatest_sum_of_consecutive_odd_integers_lt_500_l16_16003


namespace cone_volume_ratio_l16_16998

theorem cone_volume_ratio (rC hC rD hD : ℝ) (h_rC : rC = 10) (h_hC : hC = 20) (h_rD : rD = 20) (h_hD : hD = 10) :
  ((1/3) * π * rC^2 * hC) / ((1/3) * π * rD^2 * hD) = 1/2 :=
by 
  sorry

end cone_volume_ratio_l16_16998


namespace buses_passed_on_highway_l16_16186

/-- Problem statement:
     Buses from Dallas to Austin leave every hour on the hour.
     Buses from Austin to Dallas leave every two hours, starting at 7:00 AM.
     The trip from one city to the other takes 6 hours.
     Assuming the buses travel on the same highway,
     how many Dallas-bound buses does an Austin-bound bus pass on the highway?
-/
theorem buses_passed_on_highway :
  ∀ (t_depart_A2D : ℕ) (trip_time : ℕ) (buses_departures_D2A : ℕ → ℕ),
  (∀ n, buses_departures_D2A n = n) →
  trip_time = 6 →
  ∃ n, t_depart_A2D = 7 ∧ 
    (∀ t, t_depart_A2D ≤ t ∧ t < t_depart_A2D + trip_time →
      ∃ m, m + 1 = t ∧ buses_departures_D2A (m - 6) ≤ t ∧ t < buses_departures_D2A (m - 6) + 6) ↔ n + 1 = 7 := 
sorry

end buses_passed_on_highway_l16_16186


namespace fish_speed_in_still_water_l16_16360

theorem fish_speed_in_still_water (u d : ℕ) (v : ℕ) : 
  u = 35 → d = 55 → 2 * v = u + d → v = 45 := 
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end fish_speed_in_still_water_l16_16360


namespace complex_division_l16_16605

-- Define the complex numbers in Lean
def i : ℂ := Complex.I

-- Claim to be proved
theorem complex_division :
  (1 + i) / (3 - i) = (1 + 2 * i) / 5 :=
by
  sorry

end complex_division_l16_16605


namespace book_price_net_change_l16_16072

theorem book_price_net_change (P : ℝ) :
  let decreased_price := P * 0.70
  let increased_price := decreased_price * 1.20
  let net_change := (increased_price - P) / P * 100
  net_change = -16 := 
by
  sorry

end book_price_net_change_l16_16072


namespace maximum_sum_of_diagonals_of_rhombus_l16_16730

noncomputable def rhombus_side_length : ℝ := 5
noncomputable def diagonal_bd_max_length : ℝ := 6
noncomputable def diagonal_ac_min_length : ℝ := 6
noncomputable def max_diagonal_sum : ℝ := 14

theorem maximum_sum_of_diagonals_of_rhombus :
  ∀ (s bd ac : ℝ), 
  s = rhombus_side_length → 
  bd ≤ diagonal_bd_max_length → 
  ac ≥ diagonal_ac_min_length → 
  bd + ac ≤ max_diagonal_sum → 
  max_diagonal_sum = 14 :=
by
  sorry

end maximum_sum_of_diagonals_of_rhombus_l16_16730


namespace complement_of_supplement_of_35_degree_l16_16921

def angle : ℝ := 35
def supplement (x : ℝ) : ℝ := 180 - x
def complement (x : ℝ) : ℝ := 90 - x

theorem complement_of_supplement_of_35_degree :
  complement (supplement angle) = -55 := by
  sorry

end complement_of_supplement_of_35_degree_l16_16921


namespace floor_sum_eq_126_l16_16734

-- Define the problem conditions
variable (a b c d : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
variable (h5 : a^2 + b^2 = 2008) (h6 : c^2 + d^2 = 2008)
variable (h7 : a * c = 1000) (h8 : b * d = 1000)

-- Prove the solution
theorem floor_sum_eq_126 : ⌊a + b + c + d⌋ = 126 :=
by
  sorry

end floor_sum_eq_126_l16_16734


namespace raghu_investment_l16_16697

theorem raghu_investment (R : ℝ) 
  (h1 : ∀ T : ℝ, T = 0.9 * R) 
  (h2 : ∀ V : ℝ, V = 0.99 * R) 
  (h3 : R + 0.9 * R + 0.99 * R = 6069) : 
  R = 2100 := 
by
  sorry

end raghu_investment_l16_16697


namespace intersection_in_fourth_quadrant_l16_16283

variable {a : ℝ} {x : ℝ}

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x / Real.log a
noncomputable def g (x : ℝ) (a : ℝ) := (1 - a) * x

theorem intersection_in_fourth_quadrant (h : a > 1) :
  ∃ x : ℝ, x > 0 ∧ f x a < 0 ∧ f x a = g x a :=
sorry

end intersection_in_fourth_quadrant_l16_16283


namespace cindy_added_pens_l16_16917

-- Define the initial number of pens
def initial_pens : ℕ := 5

-- Define the number of pens given by Mike
def pens_from_mike : ℕ := 20

-- Define the number of pens given to Sharon
def pens_given_to_sharon : ℕ := 10

-- Define the final number of pens
def final_pens : ℕ := 40

-- Formulate the theorem regarding the pens added by Cindy
theorem cindy_added_pens :
  final_pens = initial_pens + pens_from_mike - pens_given_to_sharon + 25 :=
by
  sorry

end cindy_added_pens_l16_16917


namespace solve_k_equality_l16_16221

noncomputable def collinear_vectors (e1 e2 : ℝ) (k : ℝ) (AB CB CD : ℝ) : Prop := 
  let BD := (2 * e1 - e2) - (e1 + 3 * e2)
  BD = e1 - 4 * e2 ∧ AB = 2 * e1 + k * e2 ∧ AB = k * BD
  
theorem solve_k_equality (e1 e2 k AB CB CD : ℝ) (h_non_collinear : (e1 ≠ 0 ∨ e2 ≠ 0)) :
  collinear_vectors e1 e2 k AB CB CD → k = -8 :=
by
  intro h_collinear
  sorry

end solve_k_equality_l16_16221


namespace smallest_sum_xy_min_45_l16_16613

theorem smallest_sum_xy_min_45 (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ y) (h4 : 1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 10) :
  x + y = 45 :=
by
  sorry

end smallest_sum_xy_min_45_l16_16613


namespace hairstylist_monthly_earnings_l16_16122

noncomputable def hairstylist_earnings_per_month : ℕ :=
  let monday_wednesday_friday_earnings : ℕ := (4 * 10) + (3 * 15) + (1 * 22);
  let tuesday_thursday_earnings : ℕ := (6 * 10) + (2 * 15) + (3 * 30);
  let weekend_earnings : ℕ := (10 * 22) + (5 * 30);
  let weekly_earnings : ℕ :=
    (monday_wednesday_friday_earnings * 3) +
    (tuesday_thursday_earnings * 2) +
    (weekend_earnings * 2);
  weekly_earnings * 4

theorem hairstylist_monthly_earnings : hairstylist_earnings_per_month = 5684 := by
  -- Assertion based on the provided problem conditions
  sorry

end hairstylist_monthly_earnings_l16_16122


namespace roots_mul_shift_eq_neg_2018_l16_16174

theorem roots_mul_shift_eq_neg_2018 {a b : ℝ}
  (h1 : a + b = -1)
  (h2 : a * b = -2020) :
  (a - 1) * (b - 1) = -2018 :=
sorry

end roots_mul_shift_eq_neg_2018_l16_16174


namespace difference_shares_l16_16203

-- Given conditions in the problem
variable (V : ℕ) (F R : ℕ)
variable (hV : V = 1500)
variable (hRatioF : F = 3 * (V / 5))
variable (hRatioR : R = 11 * (V / 5))

-- The statement we need to prove
theorem difference_shares : R - F = 2400 :=
by
  -- Using the conditions to derive the result.
  sorry

end difference_shares_l16_16203


namespace arithmetic_sequence_seventh_term_l16_16875

/-- In an arithmetic sequence, the sum of the first three terms is 9 and the third term is 8. 
    Prove that the seventh term is 28. -/
theorem arithmetic_sequence_seventh_term :
  ∃ (a d : ℤ), (a + (a + d) + (a + 2 * d) = 9) ∧ (a + 2 * d = 8) ∧ (a + 6 * d = 28) :=
by
  sorry

end arithmetic_sequence_seventh_term_l16_16875


namespace zoe_earnings_from_zachary_l16_16018

noncomputable def babysitting_earnings 
  (total_earnings : ℕ) (pool_cleaning_earnings : ℕ) (earnings_julie_ratio : ℕ) 
  (earnings_chloe_ratio : ℕ) 
  (earnings_zachary : ℕ) : Prop := 
total_earnings = 8000 ∧ 
pool_cleaning_earnings = 2600 ∧ 
earnings_julie_ratio = 3 ∧ 
earnings_chloe_ratio = 5 ∧ 
9 * earnings_zachary = 5400

theorem zoe_earnings_from_zachary : babysitting_earnings 8000 2600 3 5 600 :=
by 
  unfold babysitting_earnings
  sorry

end zoe_earnings_from_zachary_l16_16018


namespace positive_real_numbers_l16_16908

theorem positive_real_numbers
  (a b c : ℝ)
  (h1 : a + b + c > 0)
  (h2 : b * c + c * a + a * b > 0)
  (h3 : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 :=
by
  sorry

end positive_real_numbers_l16_16908


namespace inequality_proof_l16_16976

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬ (a + d > b + c) := sorry

end inequality_proof_l16_16976


namespace billy_points_l16_16717

theorem billy_points (B : ℤ) (h : B - 9 = 2) : B = 11 := 
by 
  sorry

end billy_points_l16_16717


namespace inv_prop_func_point_l16_16570

theorem inv_prop_func_point {k : ℝ} :
  (∃ y x : ℝ, y = k / x ∧ (x = 2 ∧ y = -1)) → k = -2 :=
by
  intro h
  -- Proof would go here
  sorry

end inv_prop_func_point_l16_16570


namespace larger_number_is_23_l16_16821

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l16_16821


namespace sector_area_l16_16089

theorem sector_area (r : ℝ) (θ : ℝ) (arc_area : ℝ) : 
  r = 24 ∧ θ = 110 ∧ arc_area = 176 * Real.pi → 
  arc_area = (θ / 360) * (Real.pi * r ^ 2) :=
by
  intros
  sorry

end sector_area_l16_16089


namespace possible_initial_triangles_l16_16629

-- Define the triangle types by their angles in degrees
inductive TriangleType
| T45T45T90
| T30T60T90
| T30T30T120
| T60T60T60

-- Define a Lean statement to express the problem
theorem possible_initial_triangles (T : TriangleType) :
  T = TriangleType.T45T45T90 ∨
  T = TriangleType.T30T60T90 ∨
  T = TriangleType.T30T30T120 ∨
  T = TriangleType.T60T60T60 :=
sorry

end possible_initial_triangles_l16_16629


namespace union_set_eq_l16_16497

open Set

def P := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
def Q := {x : ℝ | x^2 ≤ 4}

theorem union_set_eq : P ∪ Q = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by
  sorry

end union_set_eq_l16_16497


namespace number_of_outliers_l16_16618

def data_set : List ℕ := [10, 24, 36, 36, 42, 45, 45, 46, 58, 64]
def Q1 : ℕ := 36
def Q3 : ℕ := 46
def IQR : ℕ := Q3 - Q1
def low_threshold : ℕ := Q1 - 15
def high_threshold : ℕ := Q3 + 15
def outliers : List ℕ := data_set.filter (λ x => x < low_threshold ∨ x > high_threshold)

theorem number_of_outliers : outliers.length = 3 :=
  by
    -- Proof would go here
    sorry

end number_of_outliers_l16_16618


namespace orange_orchard_land_l16_16148

theorem orange_orchard_land (F H : ℕ) 
  (h1 : F + H = 120) 
  (h2 : ∃ x : ℕ, x + (2 * x + 1) = 10) 
  (h3 : ∃ x : ℕ, 2 * x + 1 = H)
  (h4 : ∃ x : ℕ, F = x) 
  (h5 : ∃ y : ℕ, H = 2 * y + 1) :
  F = 36 ∧ H = 84 :=
by
  sorry

end orange_orchard_land_l16_16148


namespace integer_roots_of_poly_l16_16421

-- Define the polynomial
def poly (x : ℤ) (b1 b2 : ℤ) : ℤ :=
  x^3 + b2 * x ^ 2 + b1 * x + 18

-- The list of possible integer roots
def possible_integer_roots := [-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18]

-- Statement of the theorem
theorem integer_roots_of_poly (b1 b2 : ℤ) :
  ∀ x : ℤ, poly x b1 b2 = 0 → x ∈ possible_integer_roots :=
sorry

end integer_roots_of_poly_l16_16421


namespace probability_of_2_out_of_5_accurate_probability_of_2_out_of_5_with_third_accurate_l16_16060

-- Defining the conditions
def p : ℚ := 4 / 5
def n : ℕ := 5
def k1 : ℕ := 2
def k2 : ℕ := 1

-- Binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Binomial probability function
def binom_prob (k n : ℕ) (p : ℚ) : ℚ :=
  binomial n k * p^k * (1 - p)^(n - k)

-- The first proof problem:
-- Prove that the probability of exactly 2 out of 5 forecasts being accurate is 0.05 given the accuracy rate
theorem probability_of_2_out_of_5_accurate :
  binom_prob k1 n p = 0.05 := by
  sorry

-- The second proof problem:
-- Prove that the probability of exactly 2 out of 5 forecasts being accurate, with the third forecast being one of the accurate ones, is 0.02 given the accuracy rate
theorem probability_of_2_out_of_5_with_third_accurate :
  binom_prob k2 (n - 1) p = 0.02 := by
  sorry

end probability_of_2_out_of_5_accurate_probability_of_2_out_of_5_with_third_accurate_l16_16060


namespace solve_for_x_l16_16691

theorem solve_for_x :
  ∀ x : ℝ, (1 / 6 + 7 / x = 15 / x + 1 / 15 + 2) → x = -80 / 19 :=
by
  intros x h
  sorry

end solve_for_x_l16_16691


namespace wedding_cost_l16_16419

theorem wedding_cost (venue_cost food_drink_cost guests_john : ℕ) 
  (guest_increment decorations_base decorations_per_guest transport_couple transport_per_guest entertainment_cost surchage_rate discount_thresh : ℕ) (discount_rate : ℕ) :
  let guests_wife := guests_john + (guests_john * guest_increment / 100)
  let venue_total := venue_cost + (venue_cost * surchage_rate / 100)
  let food_drink_total := if guests_wife > discount_thresh then (food_drink_cost * guests_wife) * (100 - discount_rate) / 100 else food_drink_cost * guests_wife
  let decorations_total := decorations_base + (decorations_per_guest * guests_wife)
  let transport_total := transport_couple + (transport_per_guest * guests_wife)
  (venue_total + food_drink_total + decorations_total + transport_total + entertainment_cost = 56200) :=
by {
  -- Constants given in the conditions
  let venue_cost := 10000
  let food_drink_cost := 500
  let guests_john := 50
  let guest_increment := 60
  let decorations_base := 2500
  let decorations_per_guest := 10
  let transport_couple := 200
  let transport_per_guest := 15
  let entertainment_cost := 4000
  let surchage_rate := 15
  let discount_thresh := 75
  let discount_rate := 10
  sorry
}

end wedding_cost_l16_16419


namespace wire_cut_problem_l16_16912

theorem wire_cut_problem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_eq_area : (a / 4) ^ 2 = π * (b / (2 * π)) ^ 2) : 
  a / b = 2 / Real.sqrt π :=
by
  sorry

end wire_cut_problem_l16_16912


namespace balls_in_boxes_l16_16900

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l16_16900


namespace average_speed_of_participant_l16_16352

noncomputable def average_speed (d : ℝ) : ℝ :=
  let total_distance := 4 * d
  let total_time := (d / 6) + (d / 12) + (d / 18) + (d / 24)
  total_distance / total_time

theorem average_speed_of_participant :
  ∀ (d : ℝ), d > 0 → average_speed d = 11.52 :=
by
  intros d hd
  unfold average_speed
  sorry

end average_speed_of_participant_l16_16352


namespace square_side_length_l16_16366

theorem square_side_length (x : ℝ) (h : 4 * x = x^2) : x = 4 := 
by
  sorry

end square_side_length_l16_16366


namespace intersection_A_B_complement_A_in_U_complement_B_in_U_l16_16020

-- Definitions and conditions
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {5, 6, 7, 8}
def B : Set ℕ := {2, 4, 6, 8}

-- Problems to prove
theorem intersection_A_B : A ∩ B = {6, 8} := by
  sorry

theorem complement_A_in_U : U \ A = {1, 2, 3, 4} := by
  sorry

theorem complement_B_in_U : U \ B = {1, 3, 5, 7} := by
  sorry

end intersection_A_B_complement_A_in_U_complement_B_in_U_l16_16020


namespace math_proof_problem_l16_16762

noncomputable def a_value := 1
noncomputable def b_value := 2

-- Defining the primary conditions
def condition1 (a b : ℝ) : Prop :=
  ∀ x : ℝ, (a * x^2 - 3 * x + 2 > 0) ↔ (x < 1 ∨ x > b)

def condition2 (a b : ℝ) : Prop :=
  ∀ x : ℝ, (a * x^2 - (2 * b - a) * x - 2 * b < 0) ↔ (-1 < x ∧ x < 4)

-- Defining the main goal
theorem math_proof_problem :
  ∃ a b : ℝ, a = a_value ∧ b = b_value ∧ condition1 a b ∧ condition2 a b := 
sorry

end math_proof_problem_l16_16762


namespace minimum_spending_l16_16922

noncomputable def box_volume (length width height : ℕ) : ℕ := length * width * height
noncomputable def total_boxes_needed (total_volume box_volume : ℕ) : ℕ := (total_volume + box_volume - 1) / box_volume
noncomputable def total_cost (num_boxes : ℕ) (price_per_box : ℝ) : ℝ := num_boxes * price_per_box

theorem minimum_spending
  (box_length box_width box_height : ℕ)
  (price_per_box : ℝ)
  (total_collection_volume : ℕ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 15)
  (h4 : price_per_box = 0.90)
  (h5 : total_collection_volume = 3060000) :
  total_cost (total_boxes_needed total_collection_volume (box_volume box_length box_width box_height)) price_per_box = 459 :=
by
  rw [h1, h2, h3, h4, h5]
  have box_vol : box_volume 20 20 15 = 6000 := by norm_num [box_volume]
  have boxes_needed : total_boxes_needed 3060000 6000 = 510 := by norm_num [total_boxes_needed, box_volume, *]
  have cost : total_cost 510 0.90 = 459 := by norm_num [total_cost]
  exact cost

end minimum_spending_l16_16922


namespace twice_perimeter_of_square_l16_16902

theorem twice_perimeter_of_square (s : ℝ) (h : s^2 = 625) : 2 * 4 * s = 200 :=
by sorry

end twice_perimeter_of_square_l16_16902


namespace trajectory_eqn_of_point_Q_l16_16575

theorem trajectory_eqn_of_point_Q 
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (A : ℝ × ℝ := (-2, 0))
  (B : ℝ × ℝ := (2, 0))
  (l : ℝ := 10 / 3) 
  (hP_on_l : P.1 = l)
  (hQ_on_AP : (Q.2 * -4) = Q.1 * (P.2 - 0) - (P.2 * -4))
  (hBP_perp_BQ : (Q.2 * 4) = -Q.1 * ((3 * P.2) / 4 - 2))
: (Q.1^2 / 4) + Q.2^2 = 1 :=
sorry

end trajectory_eqn_of_point_Q_l16_16575


namespace sam_age_two_years_ago_l16_16681

variables (S J : ℕ)
variables (h1 : J = 3 * S) (h2 : J + 9 = 2 * (S + 9))

theorem sam_age_two_years_ago : S - 2 = 7 := by
  sorry

end sam_age_two_years_ago_l16_16681


namespace penguin_fish_consumption_l16_16266

-- Definitions based on the conditions
def initial_penguins : ℕ := 158
def total_fish_per_day : ℕ := 237
def fish_per_penguin_per_day : ℚ := 1.5

-- Lean statement for the conditional problem
theorem penguin_fish_consumption
  (P : ℕ)
  (h_initial_penguins : P = initial_penguins)
  (h_total_fish_per_day : total_fish_per_day = 237)
  (h_current_penguins : P * 2 * 3 + 129 = 1077)
  : total_fish_per_day / P = fish_per_penguin_per_day := by
  sorry

end penguin_fish_consumption_l16_16266


namespace log_expression_value_l16_16768

theorem log_expression_value : (Real.log 8 / Real.log 10) + 3 * (Real.log 5 / Real.log 10) = 3 :=
by
  -- Assuming necessary properties and steps are already known and prove the theorem accordingly:
  sorry

end log_expression_value_l16_16768


namespace child_ticket_cost_l16_16395

noncomputable def cost_of_child_ticket : ℝ := 3.50

theorem child_ticket_cost
  (adult_ticket_price : ℝ)
  (total_tickets : ℕ)
  (total_cost : ℝ)
  (adult_tickets_bought : ℕ)
  (adult_ticket_price_eq : adult_ticket_price = 5.50)
  (total_tickets_bought_eq : total_tickets = 21)
  (total_cost_eq : total_cost = 83.50)
  (adult_tickets_count : adult_tickets_bought = 5) :
  cost_of_child_ticket = 3.50 :=
by
  sorry

end child_ticket_cost_l16_16395


namespace equation_of_l2_l16_16654

-- Define the initial line equation
def l1 (x : ℝ) : ℝ := -2 * x - 2

-- Define the transformed line equation after translation
def l2 (x : ℝ) : ℝ := l1 (x + 1) + 2

-- Statement to prove
theorem equation_of_l2 : ∀ x, l2 x = -2 * x - 2 := by
  sorry

end equation_of_l2_l16_16654


namespace number_of_ways_to_purchase_magazines_l16_16847

/-
Conditions:
1. The bookstore sells 11 different magazines.
2. 8 of these magazines are priced at 2 yuan each.
3. 3 of these magazines are priced at 1 yuan each.
4. Xiao Zhang has 10 yuan to buy magazines.
5. Xiao Zhang can buy at most one copy of each magazine.
6. Xiao Zhang wants to spend all 10 yuan.

Question:
The number of different ways Xiao Zhang can purchase magazines with 10 yuan.

Answer:
266
-/

theorem number_of_ways_to_purchase_magazines : ∀ (magazines_1_yuan magazines_2_yuan : ℕ),
  magazines_1_yuan = 3 →
  magazines_2_yuan = 8 →
  (∃ (ways : ℕ), ways = 266) :=
by
  intros
  sorry

end number_of_ways_to_purchase_magazines_l16_16847


namespace number_of_ways_to_place_balls_l16_16541

theorem number_of_ways_to_place_balls : 
  let balls := 3 
  let boxes := 4 
  (boxes^balls = 64) :=
by
  sorry

end number_of_ways_to_place_balls_l16_16541


namespace solution_set_eq_2m_add_2_gt_zero_l16_16941

theorem solution_set_eq_2m_add_2_gt_zero {m : ℝ} (h : ∀ x : ℝ, mx + 2 > 0 ↔ x < 2) : m = -1 :=
sorry

end solution_set_eq_2m_add_2_gt_zero_l16_16941


namespace lincoln_high_fraction_of_girls_l16_16788

noncomputable def fraction_of_girls_in_science_fair (total_girls total_boys : ℕ) (frac_girls_participated frac_boys_participated : ℚ) : ℚ :=
  let participating_girls := frac_girls_participated * total_girls
  let participating_boys := frac_boys_participated * total_boys
  participating_girls / (participating_girls + participating_boys)

theorem lincoln_high_fraction_of_girls 
  (total_girls : ℕ) (total_boys : ℕ)
  (frac_girls_participated : ℚ) (frac_boys_participated : ℚ)
  (h1 : total_girls = 150) (h2 : total_boys = 100)
  (h3 : frac_girls_participated = 4/5) (h4 : frac_boys_participated = 3/4) :
  fraction_of_girls_in_science_fair total_girls total_boys frac_girls_participated frac_boys_participated = 8/13 := 
by
  sorry

end lincoln_high_fraction_of_girls_l16_16788


namespace dave_paid_for_6_candy_bars_l16_16057

-- Given conditions
def number_of_candy_bars : ℕ := 20
def cost_per_candy_bar : ℝ := 1.50
def amount_paid_by_john : ℝ := 21

-- Correct answer
def number_of_candy_bars_paid_by_dave : ℝ := 6

-- The proof problem in Lean statement
theorem dave_paid_for_6_candy_bars (H : number_of_candy_bars * cost_per_candy_bar - amount_paid_by_john = 9) :
  number_of_candy_bars_paid_by_dave = 6 := by
sorry

end dave_paid_for_6_candy_bars_l16_16057


namespace four_distinct_real_roots_l16_16344

theorem four_distinct_real_roots (m : ℝ) : 
  (∀ x : ℝ, |(x-1)*(x-3)| = m*x → ∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ↔ 
  0 < m ∧ m < 4 - 2 * Real.sqrt 3 :=
by
  sorry

end four_distinct_real_roots_l16_16344


namespace shaded_area_percentage_is_correct_l16_16553

noncomputable def total_area_of_square : ℕ := 49

noncomputable def area_of_first_shaded_region : ℕ := 2^2

noncomputable def area_of_second_shaded_region : ℕ := 25 - 9

noncomputable def area_of_third_shaded_region : ℕ := 49 - 36

noncomputable def total_shaded_area : ℕ :=
  area_of_first_shaded_region + area_of_second_shaded_region + area_of_third_shaded_region

noncomputable def percent_shaded_area : ℚ :=
  (total_shaded_area : ℚ) / total_area_of_square * 100

theorem shaded_area_percentage_is_correct :
  percent_shaded_area = 67.35 := by
sorry

end shaded_area_percentage_is_correct_l16_16553


namespace intersection_eq_l16_16923

theorem intersection_eq {A : Set ℕ} {B : Set ℕ} 
  (hA : A = {0, 1, 2, 3, 4, 5, 6}) 
  (hB : B = {x | ∃ n ∈ A, x = 2 * n}) : 
  A ∩ B = {0, 2, 4, 6} := by
  sorry

end intersection_eq_l16_16923


namespace angle_B_of_isosceles_triangle_l16_16797

theorem angle_B_of_isosceles_triangle (A B C : ℝ) (h_iso : (A = B ∨ A = C) ∨ (B = C ∨ B = A) ∨ (C = A ∨ C = B)) (h_angle_A : A = 70) :
  B = 70 ∨ B = 55 :=
by
  sorry

end angle_B_of_isosceles_triangle_l16_16797


namespace youtube_dislikes_l16_16694

theorem youtube_dislikes (x y : ℕ) 
  (h1 : x = 3 * y) 
  (h2 : x = 100 + 2 * y) 
  (h_y_increased : ∃ y' : ℕ, y' = 3 * y) :
  y' = 300 := by
  sorry

end youtube_dislikes_l16_16694


namespace baseEight_conversion_l16_16353

-- Base-eight number is given as 1563
def baseEight : Nat := 1563

-- Function to convert a base-eight number to base-ten
noncomputable def baseEightToBaseTen (n : Nat) : Nat :=
  let digit3 := (n / 1000) % 10
  let digit2 := (n / 100) % 10
  let digit1 := (n / 10) % 10
  let digit0 := n % 10
  digit3 * 8^3 + digit2 * 8^2 + digit1 * 8^1 + digit0 * 8^0

theorem baseEight_conversion :
  baseEightToBaseTen baseEight = 883 := by
  sorry

end baseEight_conversion_l16_16353


namespace rohan_monthly_salary_expenses_l16_16733

theorem rohan_monthly_salary_expenses 
    (food_expense_pct : ℝ)
    (house_rent_expense_pct : ℝ)
    (entertainment_expense_pct : ℝ)
    (conveyance_expense_pct : ℝ)
    (utilities_expense_pct : ℝ)
    (misc_expense_pct : ℝ)
    (monthly_saved_amount : ℝ)
    (entertainment_expense_increase_after_6_months : ℝ)
    (conveyance_expense_decrease_after_6_months : ℝ)
    (monthly_salary : ℝ)
    (savings_pct : ℝ)
    (new_savings_pct : ℝ) : 
    (food_expense_pct + house_rent_expense_pct + entertainment_expense_pct + conveyance_expense_pct + utilities_expense_pct + misc_expense_pct = 90) → 
    (100 - (food_expense_pct + house_rent_expense_pct + entertainment_expense_pct + conveyance_expense_pct + utilities_expense_pct + misc_expense_pct) = savings_pct) → 
    (monthly_saved_amount = monthly_salary * savings_pct / 100) → 
    (entertainment_expense_pct + entertainment_expense_increase_after_6_months = 20) → 
    (conveyance_expense_pct - conveyance_expense_decrease_after_6_months = 7) → 
    (new_savings_pct = 100 - (30 + 25 + (entertainment_expense_pct + entertainment_expense_increase_after_6_months) + (conveyance_expense_pct - conveyance_expense_decrease_after_6_months) + 5 + 5)) → 
    monthly_salary = 15000 ∧ new_savings_pct = 8 := 
sorry

end rohan_monthly_salary_expenses_l16_16733


namespace find_bananas_l16_16758

theorem find_bananas 
  (bananas apples persimmons : ℕ) 
  (h1 : apples = 4 * bananas) 
  (h2 : persimmons = 3 * bananas) 
  (h3 : apples + persimmons = 210) : 
  bananas = 30 := 
  sorry

end find_bananas_l16_16758


namespace circle_radius_triple_area_l16_16558

/-- Given the area of a circle is tripled when its radius r is increased by n, prove that 
    r = n * (sqrt(3) - 1) / 2 -/
theorem circle_radius_triple_area (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 1) / 2 :=
sorry

end circle_radius_triple_area_l16_16558


namespace total_students_at_concert_l16_16710

-- Define the number of buses
def num_buses : ℕ := 8

-- Define the number of students per bus
def students_per_bus : ℕ := 45

-- State the theorem with the conditions and expected result
theorem total_students_at_concert : (num_buses * students_per_bus) = 360 := by
  -- Proof is not required as per the instructions; replace with 'sorry'
  sorry

end total_students_at_concert_l16_16710


namespace new_average_age_l16_16781

/--
The average age of 7 people in a room is 28 years.
A 22-year-old person leaves the room, and a 30-year-old person enters the room.
Prove that the new average age of the people in the room is \( 29 \frac{1}{7} \).
-/
theorem new_average_age (avg_age : ℕ) (num_people : ℕ) (leaving_age : ℕ) (entering_age : ℕ)
  (H1 : avg_age = 28)
  (H2 : num_people = 7)
  (H3 : leaving_age = 22)
  (H4 : entering_age = 30) :
  (avg_age * num_people - leaving_age + entering_age) / num_people = 29 + 1 / 7 := 
by
  sorry

end new_average_age_l16_16781


namespace circles_intersect_and_common_chord_l16_16075

theorem circles_intersect_and_common_chord :
  (∃ P : ℝ × ℝ, P.1 ^ 2 + P.2 ^ 2 - P.1 + P.2 - 2 = 0 ∧
                P.1 ^ 2 + P.2 ^ 2 = 5) ∧
  (∀ x y : ℝ, (x ^ 2 + y ^ 2 - x + y - 2 = 0 ∧ x ^ 2 + y ^ 2 = 5) →
              x - y - 3 = 0) ∧
  (∃ A B : ℝ × ℝ, A.1 ^ 2 + A.2 ^ 2 - A.1 + A.2 - 2 = 0 ∧
                   A.1 ^ 2 + A.2 ^ 2 = 5 ∧
                   B.1 ^ 2 + B.2 ^ 2 - B.1 + B.2 - 2 = 0 ∧
                   B.1 ^ 2 + B.2 ^ 2 = 5 ∧
                   (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 2) := sorry

end circles_intersect_and_common_chord_l16_16075


namespace avg_people_moving_per_hour_l16_16724

theorem avg_people_moving_per_hour (total_people : ℕ) (total_days : ℕ) (hours_per_day : ℕ) (h : total_people = 3000 ∧ total_days = 4 ∧ hours_per_day = 24) : 
  (total_people / (total_days * hours_per_day)).toFloat.round = 31 :=
by
  have h1 : total_people = 3000 := h.1;
  have h2 : total_days = 4 := h.2.1;
  have h3 : hours_per_day = 24 := h.2.2;
  rw [h1, h2, h3];
  sorry

end avg_people_moving_per_hour_l16_16724


namespace train_length_l16_16140

theorem train_length 
  (V : ℝ → ℝ) (L : ℝ) 
  (length_of_train : ∀ (t : ℝ), t = 8 → V t = L / 8) 
  (pass_platform : ∀ (d t : ℝ), d = L + 273 → t = 20 → V t = d / t) 
  : L = 182 := 
by
  sorry

end train_length_l16_16140


namespace banana_price_reduction_l16_16457

theorem banana_price_reduction (P_r : ℝ) (P : ℝ) (n : ℝ) (m : ℝ) (h1 : P_r = 3) (h2 : n = 40) (h3 : m = 64) 
  (h4 : 160 = (n / P_r) * 12) 
  (h5 : 96 = 160 - m) 
  (h6 : (40 / 8) = P) :
  (P - P_r) / P * 100 = 40 :=
by
  sorry

end banana_price_reduction_l16_16457


namespace checkerboard_sum_is_328_l16_16252

def checkerboard_sum : Nat :=
  1 + 2 + 9 + 8 + 73 + 74 + 81 + 80

theorem checkerboard_sum_is_328 : checkerboard_sum = 328 := by
  sorry

end checkerboard_sum_is_328_l16_16252


namespace box_prices_l16_16098

theorem box_prices (a b c : ℝ) 
  (h1 : a + b + c = 9) 
  (h2 : 3 * a + 2 * b + c = 16) : 
  c - a = 2 := 
by 
  sorry

end box_prices_l16_16098


namespace find_a_b_sum_l16_16556

theorem find_a_b_sum (a b : ℕ) (h : a^2 - b^4 = 2009) : a + b = 47 :=
sorry

end find_a_b_sum_l16_16556


namespace find_number_l16_16825

theorem find_number (x : ℝ) (h : 4 * (x - 220) = 320) : (5 * x) / 3 = 500 :=
by
  sorry

end find_number_l16_16825


namespace interest_is_less_by_1940_l16_16833

noncomputable def principal : ℕ := 2000
noncomputable def rate : ℕ := 3
noncomputable def time : ℕ := 3

noncomputable def simple_interest (P R T : ℕ) : ℕ :=
  (P * R * T) / 100

noncomputable def difference (sum_lent interest : ℕ) : ℕ :=
  sum_lent - interest

theorem interest_is_less_by_1940 :
  difference principal (simple_interest principal rate time) = 1940 :=
by
  sorry

end interest_is_less_by_1940_l16_16833


namespace distance_to_fourth_side_l16_16245

-- Let s be the side length of the square.
variable (s : ℝ) (d1 d2 d3 d4 : ℝ)

-- The given conditions:
axiom h1 : d1 = 4
axiom h2 : d2 = 7
axiom h3 : d3 = 13
axiom h4 : d1 + d2 + d3 + d4 = s
axiom h5 : 0 < d4

-- The statement to prove:
theorem distance_to_fourth_side : d4 = 10 ∨ d4 = 16 :=
by
  sorry

end distance_to_fourth_side_l16_16245


namespace inequality_proof_l16_16147

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) : 
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 := 
sorry

end inequality_proof_l16_16147


namespace salmon_trip_l16_16120

theorem salmon_trip (male_salmons : ℕ) (female_salmons : ℕ) : male_salmons = 712261 → female_salmons = 259378 → male_salmons + female_salmons = 971639 :=
  sorry

end salmon_trip_l16_16120


namespace algebraic_expression_for_A_l16_16559

variable {x y A : ℝ}

theorem algebraic_expression_for_A
  (h : (3 * x + 2 * y) ^ 2 = (3 * x - 2 * y) ^ 2 + A) :
  A = 24 * x * y :=
sorry

end algebraic_expression_for_A_l16_16559


namespace volumes_relation_l16_16290

-- Definitions and conditions based on the problem
variables {a b c : ℝ} (h_triangle : a > b) (h_triangle2 : b > c) (h_acute : 0 < θ ∧ θ < π)

-- The heights from vertices
variables (AD BE CF : ℝ)

-- Volumes of the tetrahedrons formed after folding
variables (V1 V2 V3 : ℝ)

-- The heights are given:
noncomputable def height_AD (BC : ℝ) (theta : ℝ) := AD
noncomputable def height_BE (CA : ℝ) (theta : ℝ) := BE
noncomputable def height_CF (AB : ℝ) (theta : ℝ) := CF

-- Using these heights and the acute nature of the triangle
noncomputable def volume_V1 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V1
noncomputable def volume_V2 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V2
noncomputable def volume_V3 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V3

-- The theorem stating the relationship between volumes
theorem volumes_relation
  (h_triangle: a > b)
  (h_triangle2: b > c)
  (h_acute: 0 < θ ∧ θ < π)
  (h_volumes: V1 > V2 ∧ V2 > V3):
  V1 > V2 ∧ V2 > V3 :=
sorry

end volumes_relation_l16_16290


namespace quadratic_eq_distinct_solutions_l16_16359

theorem quadratic_eq_distinct_solutions (b : ℤ) (k : ℤ) (h1 : 1 ≤ b ∧ b ≤ 100) :
  ∃ n : ℕ, n = 27 ∧ (x^2 + (2 * b + 3) * x + b^2 = 0 →
    12 * b + 9 = k^2 → 
    (∃ m n : ℤ, x = m ∧ x = n ∧ m ≠ n)) :=
sorry

end quadratic_eq_distinct_solutions_l16_16359


namespace actual_plot_area_in_acres_l16_16254

-- Condition Definitions
def base_cm : ℝ := 8
def height_cm : ℝ := 12
def scale_cm_to_miles : ℝ := 1  -- 1 cm = 1 mile
def miles_to_acres : ℝ := 320  -- 1 square mile = 320 acres

-- Theorem Statement
theorem actual_plot_area_in_acres (A : ℝ) :
  A = 15360 :=
by
  sorry

end actual_plot_area_in_acres_l16_16254


namespace cos_beta_eq_neg_16_over_65_l16_16975

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < π)
variable (h2 : 0 < β ∧ β < π)
variable (h3 : Real.sin β = 5 / 13)
variable (h4 : Real.tan (α / 2) = 1 / 2)

theorem cos_beta_eq_neg_16_over_65 : Real.cos β = -16 / 65 := by
  sorry

end cos_beta_eq_neg_16_over_65_l16_16975


namespace max_absolute_difference_l16_16601

theorem max_absolute_difference (a b c d e : ℤ) (p : ℤ) :
  0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e ∧ e ≤ 100 ∧ p = (a + b + c + d + e) / 5 →
  (|p - c| ≤ 40) :=
by
  sorry

end max_absolute_difference_l16_16601


namespace a_eq_zero_l16_16411

theorem a_eq_zero (a : ℝ) (h : ∀ x : ℝ, ax + 2 ≠ 0) : a = 0 := by
  sorry

end a_eq_zero_l16_16411


namespace avg_new_students_l16_16146

-- Definitions for conditions
def orig_strength : ℕ := 17
def orig_avg_age : ℕ := 40
def new_students_count : ℕ := 17
def decreased_avg_age : ℕ := 36 -- given that average decreases by 4 years, i.e., 40 - 4

-- Definition for the original total age
def total_age_orig : ℕ := orig_strength * orig_avg_age

-- Definition for the total number of students after new students join
def total_students : ℕ := orig_strength + new_students_count

-- Definition for the total age after new students join
def total_age_new : ℕ := total_students * decreased_avg_age

-- Definition for the total age of new students
def total_age_new_students : ℕ := total_age_new - total_age_orig

-- Definition for the average age of new students
def avg_age_new_students : ℕ := total_age_new_students / new_students_count

-- Lean theorem stating the proof problem
theorem avg_new_students : 
  avg_age_new_students = 32 := 
by sorry

end avg_new_students_l16_16146


namespace card_worth_l16_16561

theorem card_worth (value_per_card : ℕ) (num_cards_traded : ℕ) (profit : ℕ) (value_traded : ℕ) (worth_received : ℕ) :
  value_per_card = 8 →
  num_cards_traded = 2 →
  profit = 5 →
  value_traded = num_cards_traded * value_per_card →
  worth_received = value_traded + profit →
  worth_received = 21 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end card_worth_l16_16561


namespace natural_number_sets_solution_l16_16920

theorem natural_number_sets_solution (x y n : ℕ) (h : (x! + y!) / n! = 3^n) : (x = 0 ∧ y = 2 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1) :=
by
  sorry

end natural_number_sets_solution_l16_16920


namespace simplify_expression_l16_16872

theorem simplify_expression (x : ℤ) (h1 : 2 * (x - 1) < x + 1) (h2 : 5 * x + 3 ≥ 2 * x) :
  (x = 2) → (2 / (x^2 + x) / (1 - (x - 1) / (x^2 - 1)) = 1 / 2) :=
by
  sorry

end simplify_expression_l16_16872


namespace axis_of_symmetry_y_range_l16_16708

/-- 
The equation of the curve is given by |x| + y^2 - 3y = 0.
We aim to prove two properties:
1. The axis of symmetry of this curve is x = 0.
2. The range of possible values for y is [0, 3].
-/
noncomputable def curve (x y : ℝ) : ℝ := |x| + y^2 - 3*y

theorem axis_of_symmetry : ∀ x y : ℝ, curve x y = 0 → x = 0 :=
sorry

theorem y_range : ∀ y : ℝ, ∃ x : ℝ, curve x y = 0 → (0 ≤ y ∧ y ≤ 3) :=
sorry

end axis_of_symmetry_y_range_l16_16708


namespace evaluate_expression_l16_16865

def a : ℕ := 3^1
def b : ℕ := 3^2
def c : ℕ := 3^3
def d : ℕ := 3^4
def e : ℕ := 3^10
def S : ℕ := a + b + c + d

theorem evaluate_expression : e - S = 58929 := 
by
  sorry

end evaluate_expression_l16_16865


namespace find_x_minus_4y_l16_16133

theorem find_x_minus_4y (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - 3 * y = 10) : x - 4 * y = 5 :=
by 
  sorry

end find_x_minus_4y_l16_16133


namespace bar_weight_calc_l16_16393

variable (blue_weight green_weight num_blue_weights num_green_weights bar_weight total_weight : ℕ)

theorem bar_weight_calc
  (h1 : blue_weight = 2)
  (h2 : green_weight = 3)
  (h3 : num_blue_weights = 4)
  (h4 : num_green_weights = 5)
  (h5 : total_weight = 25)
  (weights_total := num_blue_weights * blue_weight + num_green_weights * green_weight)
  : bar_weight = total_weight - weights_total :=
by
  sorry

end bar_weight_calc_l16_16393


namespace cubic_solution_l16_16169

theorem cubic_solution (a b c : ℝ) (h_eq : ∀ x, x^3 - 4*x^2 + 7*x + 6 = 34 -> x = a ∨ x = b ∨ x = c)
(h_ge : a ≥ b ∧ b ≥ c) : 2 * a + b = 8 := 
sorry

end cubic_solution_l16_16169


namespace none_of_these_l16_16518

-- Problem Statement:
theorem none_of_these (r x y : ℝ) (h1 : r > 0) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : x^2 + y^2 > x^2 * y^2) :
  ¬ (-x > -y) ∧ ¬ (-x > y) ∧ ¬ (1 > -y / x) ∧ ¬ (1 < x / y) :=
by
  sorry

end none_of_these_l16_16518


namespace chicken_legs_baked_l16_16738

theorem chicken_legs_baked (L : ℕ) (H₁ : 144 / 16 = 9) (H₂ : 224 / 16 = 14) (H₃ : 16 * 9 = 144) :  L = 144 :=
by
  sorry

end chicken_legs_baked_l16_16738


namespace bob_cleaning_time_l16_16329

theorem bob_cleaning_time (alice_time : ℕ) (h1 : alice_time = 25) (bob_ratio : ℚ) (h2 : bob_ratio = 2 / 5) : 
  bob_time = 10 :=
by
  -- Definitions for conditions
  let bob_time := bob_ratio * alice_time
  -- Sorry to represent the skipped proof
  sorry

end bob_cleaning_time_l16_16329


namespace minimum_value_of_function_l16_16379

theorem minimum_value_of_function (x : ℝ) (h : x * Real.log 2 / Real.log 3 ≥ 1) : 
  ∃ t : ℝ, t = 2^x ∧ t ≥ 3 ∧ ∀ y : ℝ, y = t^2 - 2*t - 3 → y = (t-1)^2 - 4 := 
sorry

end minimum_value_of_function_l16_16379


namespace vector_computation_l16_16791

def v1 : ℤ × ℤ := (3, -5)
def v2 : ℤ × ℤ := (2, -10)
def s1 : ℤ := 4
def s2 : ℤ := 3

theorem vector_computation : s1 • v1 - s2 • v2 = (6, 10) :=
  sorry

end vector_computation_l16_16791


namespace count_letters_with_both_l16_16433

theorem count_letters_with_both (a b c x : ℕ) 
  (h₁ : a = 24) 
  (h₂ : b = 7) 
  (h₃ : c = 40) 
  (H : a + b + x = c) : 
  x = 9 :=
by {
  -- Proof here
  sorry
}

end count_letters_with_both_l16_16433


namespace find_g3_l16_16234

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x^2 + b * x^3 + c * x + d

theorem find_g3 (a b c d : ℝ) (h : g (-3) a b c d = 2) : g 3 a b c d = 0 := 
by 
  sorry

end find_g3_l16_16234


namespace inequality_a_neg_one_inequality_general_a_l16_16431

theorem inequality_a_neg_one : ∀ x : ℝ, (x^2 + x - 2 > 0) ↔ (x < -2 ∨ x > 1) :=
by { sorry }

theorem inequality_general_a : 
∀ (a x : ℝ), ax^2 - (a + 2)*x + 2 < 0 ↔ 
  if a = 0 then x > 1
  else if a < 0 then x < (2 / a) ∨ x > 1
  else if 0 < a ∧ a < 2 then 1 < x ∧ x < (2 / a)
  else if a = 2 then False
  else (2 / a) < x ∧ x < 1 :=
by { sorry }

end inequality_a_neg_one_inequality_general_a_l16_16431


namespace divide_subtract_result_l16_16544

theorem divide_subtract_result (x : ℕ) (h : (x - 26) / 2 = 37) : 48 - (x / 4) = 23 := 
by
  sorry

end divide_subtract_result_l16_16544


namespace base_eight_to_base_ten_l16_16123

theorem base_eight_to_base_ten : (5 * 8^1 + 2 * 8^0) = 42 := by
  sorry

end base_eight_to_base_ten_l16_16123


namespace min_value_fraction_l16_16597

theorem min_value_fraction (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : 
  ∀ x, (x = (1 / m + 8 / n)) → x ≥ 18 :=
by
  sorry

end min_value_fraction_l16_16597


namespace cost_of_fencing_per_meter_l16_16184

theorem cost_of_fencing_per_meter
  (length breadth : ℕ)
  (total_cost : ℝ)
  (h1 : length = breadth + 20)
  (h2 : length = 60)
  (h3 : total_cost = 5300) :
  (total_cost / (2 * length + 2 * breadth)) = 26.5 := 
by
  sorry

end cost_of_fencing_per_meter_l16_16184


namespace maximum_fraction_l16_16261

theorem maximum_fraction (a b h : ℝ) (d : ℝ) (h_d_def : d = Real.sqrt (a^2 + b^2 + h^2)) :
  (a + b + h) / d ≤ Real.sqrt 3 :=
sorry

end maximum_fraction_l16_16261


namespace no_pairs_satisfy_equation_l16_16424

theorem no_pairs_satisfy_equation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a^2 + 1 / b^2 = 1 / (a^2 + b^2)) → False :=
by
  sorry

end no_pairs_satisfy_equation_l16_16424


namespace geometric_series_ratio_half_l16_16644

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l16_16644


namespace line_through_P_with_equal_intercepts_line_through_A_with_inclination_90_l16_16661

-- Definitions for the first condition
def P : ℝ × ℝ := (3, 2)
def passes_through_P (l : ℝ → ℝ) := l P.1 = P.2
def equal_intercepts (l : ℝ → ℝ) := ∃ a : ℝ, l a = 0 ∧ l (-a) = 0

-- Equation 1: Line passing through P with equal intercepts
theorem line_through_P_with_equal_intercepts :
  (∃ l : ℝ → ℝ, passes_through_P l ∧ equal_intercepts l ∧ 
   (∀ x y : ℝ, l x = y ↔ (2 * x - 3 * y = 0) ∨ (x + y - 5 = 0))) :=
sorry

-- Definitions for the second condition
def A : ℝ × ℝ := (-1, -3)
def passes_through_A (l : ℝ → ℝ) := l A.1 = A.2
def inclination_90 (l : ℝ → ℝ) := ∀ x : ℝ, l x = l 0

-- Equation 2: Line passing through A with inclination 90°
theorem line_through_A_with_inclination_90 :
  (∃ l : ℝ → ℝ, passes_through_A l ∧ inclination_90 l ∧ 
   (∀ x y : ℝ, l x = y ↔ (x + 1 = 0))) :=
sorry

end line_through_P_with_equal_intercepts_line_through_A_with_inclination_90_l16_16661


namespace original_price_increased_by_total_percent_l16_16077

noncomputable def percent_increase_sequence (P : ℝ) : ℝ :=
  let step1 := P * 1.15
  let step2 := step1 * 1.40
  let step3 := step2 * 1.20
  let step4 := step3 * 0.90
  let step5 := step4 * 1.25
  (step5 - P) / P * 100

theorem original_price_increased_by_total_percent (P : ℝ) : percent_increase_sequence P = 117.35 :=
by
  -- Sorry is used here for simplicity, but the automated proof will involve calculating the exact percentage increase step-by-step.
  sorry

end original_price_increased_by_total_percent_l16_16077


namespace zeros_in_expansion_l16_16842

def num_zeros_expansion (n : ℕ) : ℕ :=
-- This function counts the number of trailing zeros in the decimal representation of n.
sorry

theorem zeros_in_expansion : num_zeros_expansion ((10^12 - 3)^2) = 11 :=
sorry

end zeros_in_expansion_l16_16842


namespace solve_x_l16_16212

theorem solve_x :
  (2 / 3 - 1 / 4) = 1 / (12 / 5) :=
by
  sorry

end solve_x_l16_16212


namespace add_to_both_num_and_denom_l16_16026

theorem add_to_both_num_and_denom (n : ℕ) : (4 + n) / (7 + n) = 7 / 8 ↔ n = 17 := by
  sorry

end add_to_both_num_and_denom_l16_16026


namespace revenue_95_percent_l16_16139

-- Definitions based on the conditions
variables (C : ℝ) (n : ℝ)
def revenue_full : ℝ := 1.20 * C
def tickets_sold_percentage : ℝ := 0.95

-- Statement of the theorem based on the problem translation
theorem revenue_95_percent (C : ℝ) :
  (tickets_sold_percentage * revenue_full C) = 1.14 * C :=
by
  sorry -- Proof to be provided

end revenue_95_percent_l16_16139


namespace bill_due_months_l16_16051

theorem bill_due_months {TD A: ℝ} (R: ℝ) : 
  TD = 189 → A = 1764 → R = 16 → 
  ∃ M: ℕ, A - TD * (1 + (R/100) * (M/12)) = 1764 - 189 * (1 + (16/100) * (10/12)) ∧ M = 10 :=
by
  intro hTD hA hR
  use 10
  sorry

end bill_due_months_l16_16051


namespace determine_y_l16_16249

theorem determine_y (y : ℝ) (y_nonzero : y ≠ 0) (h : (3 * y)^5 = (9 * y)^4) : y = 27 := 
sorry

end determine_y_l16_16249


namespace sally_took_out_5_onions_l16_16486

theorem sally_took_out_5_onions (X Y : ℕ) 
    (h1 : 4 + 9 - Y + X = X + 8) : Y = 5 := 
by
  sorry

end sally_took_out_5_onions_l16_16486


namespace hyperbola_foci_y_axis_condition_l16_16729

theorem hyperbola_foci_y_axis_condition (m n : ℝ) (h : m * n < 0) : 
  (mx^2 + ny^2 = 1) →
  (m < 0 ∧ n > 0) :=
sorry

end hyperbola_foci_y_axis_condition_l16_16729


namespace coefficient_a_for_factor_l16_16192

noncomputable def P (a : ℚ) (x : ℚ) : ℚ := x^3 + 2 * x^2 + a * x + 20

theorem coefficient_a_for_factor (a : ℚ) :
  (∀ x : ℚ, (x - 3) ∣ P a x) → a = -65/3 :=
by
  sorry

end coefficient_a_for_factor_l16_16192


namespace solve_inequality_l16_16557

theorem solve_inequality (x : ℝ) : 2 * x^2 + 8 * x ≤ -6 ↔ -3 ≤ x ∧ x ≤ -1 :=
by
  sorry

end solve_inequality_l16_16557


namespace area_ratio_XYZ_PQR_l16_16042

theorem area_ratio_XYZ_PQR 
  (PR PQ QR : ℝ)
  (p q r : ℝ) 
  (hPR : PR = 15) 
  (hPQ : PQ = 20) 
  (hQR : QR = 25)
  (hPX : p * PR = PR * p)
  (hQY : q * QR = QR * q) 
  (hPZ : r * PQ = PQ * r) 
  (hpq_sum : p + q + r = 3 / 4) 
  (hpq_sq_sum : p^2 + q^2 + r^2 = 9 / 16) : 
  (area_triangle_XYZ / area_triangle_PQR = 1 / 4) :=
sorry

end area_ratio_XYZ_PQR_l16_16042


namespace smallest_a_no_inverse_mod_72_90_l16_16619

theorem smallest_a_no_inverse_mod_72_90 :
  ∃ (a : ℕ), a > 0 ∧ ∀ b : ℕ, (b > 0 → gcd b 72 > 1 ∧ gcd b 90 > 1 → b ≥ a) ∧ gcd a 72 > 1 ∧ gcd a 90 > 1 ∧ a = 6 :=
by sorry

end smallest_a_no_inverse_mod_72_90_l16_16619


namespace solve_arithmetic_sequence_l16_16999

theorem solve_arithmetic_sequence (y : ℝ) (h : 0 < y) (h_arith : ∃ (d : ℝ), 4 + d = y^2 ∧ y^2 + d = 16 ∧ 16 + d = 36) :
  y = Real.sqrt 10 := by
  sorry

end solve_arithmetic_sequence_l16_16999


namespace negation_example_l16_16116

theorem negation_example : ¬ (∃ x : ℤ, x^2 + 2 * x + 1 ≤ 0) ↔ ∀ x : ℤ, x^2 + 2 * x + 1 > 0 := 
by 
  sorry

end negation_example_l16_16116


namespace larger_number_is_eight_l16_16135

theorem larger_number_is_eight (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 :=
by
  sorry

end larger_number_is_eight_l16_16135


namespace basket_weight_l16_16829

def weight_of_basket_alone (n_pears : ℕ) (weight_per_pear total_weight : ℚ) : ℚ :=
  total_weight - (n_pears * weight_per_pear)

theorem basket_weight :
  weight_of_basket_alone 30 0.36 11.26 = 0.46 := by
  sorry

end basket_weight_l16_16829


namespace distinguishable_triangles_count_l16_16565

def count_distinguishable_triangles (colors : ℕ) : ℕ :=
  let corner_cases := colors + (colors * (colors - 1)) + (colors * (colors - 1) * (colors - 2) / 6)
  let edge_cases := colors * colors
  let center_cases := colors
  corner_cases * edge_cases * center_cases

theorem distinguishable_triangles_count :
  count_distinguishable_triangles 8 = 61440 :=
by
  unfold count_distinguishable_triangles
  -- corner_cases = 8 + 8 * 7 + (8 * 7 * 6) / 6 = 120
  -- edge_cases = 8 * 8 = 64
  -- center_cases = 8
  -- Total = 120 * 64 * 8 = 61440
  sorry

end distinguishable_triangles_count_l16_16565


namespace division_problem_l16_16385

theorem division_problem (A : ℕ) (h : 23 = (A * 3) + 2) : A = 7 :=
sorry

end division_problem_l16_16385


namespace minimum_value_a_l16_16458

noncomputable def f (a b x : ℝ) := a * Real.log x - (1 / 2) * x^2 + b * x

theorem minimum_value_a (h : ∀ b x : ℝ, x > 0 → f a b x > 0) : a ≥ -Real.exp 3 := 
sorry

end minimum_value_a_l16_16458


namespace prime_divides_2_pow_n_minus_n_infinte_times_l16_16346

theorem prime_divides_2_pow_n_minus_n_infinte_times (p : ℕ) (hp : Nat.Prime p) : ∃ᶠ n in at_top, p ∣ 2^n - n :=
sorry

end prime_divides_2_pow_n_minus_n_infinte_times_l16_16346


namespace price_per_gaming_chair_l16_16994

theorem price_per_gaming_chair 
  (P : ℝ)
  (price_per_organizer : ℝ := 78)
  (num_organizers : ℕ := 3)
  (num_chairs : ℕ := 2)
  (total_paid : ℝ := 420)
  (delivery_fee_rate : ℝ := 0.05) 
  (cost_organizers : ℝ := num_organizers * price_per_organizer)
  (cost_gaming_chairs : ℝ := num_chairs * P)
  (total_sales : ℝ := cost_organizers + cost_gaming_chairs)
  (delivery_fee : ℝ := delivery_fee_rate * total_sales) :
  total_paid = total_sales + delivery_fee → P = 83 := 
sorry

end price_per_gaming_chair_l16_16994


namespace range_of_a_l16_16963

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → x^2 - 2*x + a ≥ 0) → a ≥ 1 :=
by
  sorry

end range_of_a_l16_16963


namespace vector_AB_to_vector_BA_l16_16937

theorem vector_AB_to_vector_BA (z : ℂ) (hz : z = -3 + 2 * Complex.I) : -z = 3 - 2 * Complex.I :=
by
  rw [hz]
  sorry

end vector_AB_to_vector_BA_l16_16937


namespace yoongi_class_combination_l16_16839

theorem yoongi_class_combination : (Nat.choose 10 3 = 120) := by
  sorry

end yoongi_class_combination_l16_16839


namespace rhombus_area_l16_16714

-- Declare the lengths of the diagonals
def diagonal1 := 6
def diagonal2 := 8

-- Define the area function for a rhombus
def area_of_rhombus (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

-- State the theorem
theorem rhombus_area : area_of_rhombus diagonal1 diagonal2 = 24 := by sorry

end rhombus_area_l16_16714


namespace bus_driver_limit_of_hours_l16_16657

theorem bus_driver_limit_of_hours (r o T H L : ℝ)
  (h_reg_rate : r = 16)
  (h_ot_rate : o = 1.75 * r)
  (h_total_comp : T = 752)
  (h_hours_worked : H = 44)
  (h_equation : r * L + o * (H - L) = T) :
  L = 40 :=
  sorry

end bus_driver_limit_of_hours_l16_16657


namespace net_gain_A_correct_l16_16489

-- Define initial values and transactions
def initial_cash_A : ℕ := 20000
def house_value : ℕ := 20000
def car_value : ℕ := 5000
def initial_cash_B : ℕ := 25000
def house_sale_price : ℕ := 21000
def car_sale_price : ℕ := 4500
def house_repurchase_price : ℕ := 19000
def car_depreciation : ℕ := 10
def car_repurchase_price : ℕ := 4050

-- Define the final cash calculations
def final_cash_A := initial_cash_A + house_sale_price + car_sale_price - house_repurchase_price - car_repurchase_price
def final_cash_B := initial_cash_B - house_sale_price - car_sale_price + house_repurchase_price + car_repurchase_price

-- Define the net gain calculations
def net_gain_A := final_cash_A - initial_cash_A
def net_gain_B := final_cash_B - initial_cash_B

-- Theorem to prove
theorem net_gain_A_correct : net_gain_A = 2000 :=
by 
  -- Definitions and calculations would go here
  sorry

end net_gain_A_correct_l16_16489


namespace total_spent_is_correct_l16_16713

def meal_prices : List ℕ := [12, 15, 10, 18, 20]
def ice_cream_prices : List ℕ := [2, 3, 3, 4, 4]
def tip_percentage : ℝ := 0.15
def tax_percentage : ℝ := 0.08

def total_meal_cost (prices : List ℕ) : ℝ :=
  prices.sum

def total_ice_cream_cost (prices : List ℕ) : ℝ :=
  prices.sum

def calculate_tip (total_meal_cost : ℝ) (tip_percentage : ℝ) : ℝ :=
  total_meal_cost * tip_percentage

def calculate_tax (total_meal_cost : ℝ) (tax_percentage : ℝ) : ℝ :=
  total_meal_cost * tax_percentage

def total_amount_spent (meal_prices : List ℕ) (ice_cream_prices : List ℕ) (tip_percentage : ℝ) (tax_percentage : ℝ) : ℝ :=
  let total_meal := total_meal_cost meal_prices
  let total_ice_cream := total_ice_cream_cost ice_cream_prices
  let tip := calculate_tip total_meal tip_percentage
  let tax := calculate_tax total_meal tax_percentage
  total_meal + total_ice_cream + tip + tax

theorem total_spent_is_correct :
  total_amount_spent meal_prices ice_cream_prices tip_percentage tax_percentage = 108.25 := 
by
  sorry

end total_spent_is_correct_l16_16713


namespace shoes_cost_l16_16448

theorem shoes_cost (S : ℝ) : 
  let suit := 430
  let discount := 100
  let total_paid := 520
  suit + S - discount = total_paid -> 
  S = 190 :=
by 
  intro h
  sorry

end shoes_cost_l16_16448


namespace remainder_proof_l16_16076

noncomputable def problem (n : ℤ) : Prop :=
  n % 9 = 4

noncomputable def solution (n : ℤ) : ℤ :=
  (4 * n - 11) % 9

theorem remainder_proof (n : ℤ) (h : problem n) : solution n = 5 := by
  sorry

end remainder_proof_l16_16076


namespace customer_survey_response_l16_16763

theorem customer_survey_response (N : ℕ)
  (avg_income : ℕ → ℕ)
  (avg_all : avg_income N = 45000)
  (avg_top10 : avg_income 10 = 55000)
  (avg_others : avg_income (N - 10) = 42500) :
  N = 50 := 
sorry

end customer_survey_response_l16_16763


namespace effect_of_dimension_changes_on_area_l16_16841

variable {L B : ℝ}  -- Original length and breadth

def original_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := 1.15 * L

def new_breadth (B : ℝ) : ℝ := 0.90 * B

def new_area (L B : ℝ) : ℝ := new_length L * new_breadth B

theorem effect_of_dimension_changes_on_area (L B : ℝ) :
  new_area L B = 1.035 * original_area L B :=
by
  sorry

end effect_of_dimension_changes_on_area_l16_16841


namespace largest_value_x_y_l16_16652

theorem largest_value_x_y (x y : ℝ) (h1 : 5 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) : x + y ≤ 11 / 4 :=
sorry

end largest_value_x_y_l16_16652


namespace negation_existence_l16_16341

-- The problem requires showing the equivalence between the negation of an existential
-- proposition and a universal proposition in the context of real numbers.

theorem negation_existence (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) → (∀ x : ℝ, x^2 - m * x - m ≥ 0) :=
by
  sorry

end negation_existence_l16_16341


namespace dodecahedron_interior_diagonals_l16_16846

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l16_16846


namespace decompose_number_4705_l16_16325

theorem decompose_number_4705 :
  4.705 = 4 * 1 + 7 * 0.1 + 0 * 0.01 + 5 * 0.001 := by
  sorry

end decompose_number_4705_l16_16325


namespace number_of_pebbles_l16_16198

theorem number_of_pebbles (P : ℕ) : 
  (P * (1/4 : ℝ) + 3 * (1/2 : ℝ) + 2 * 2 = 7) → P = 6 := by
  sorry

end number_of_pebbles_l16_16198


namespace min_sum_nonpos_l16_16267

theorem min_sum_nonpos (a b : ℤ) (h_nonpos_a : a ≤ 0) (h_nonpos_b : b ≤ 0) (h_prod : a * b = 144) : 
  a + b = -30 :=
sorry

end min_sum_nonpos_l16_16267


namespace simplify_expression1_simplify_expression2_l16_16637

/-- Proof Problem 1: Simplify the expression (a+2b)^2 - 4b(a+b) -/
theorem simplify_expression1 (a b : ℝ) : 
  (a + 2 * b)^2 - 4 * b * (a + b) = a^2 :=
sorry

/-- Proof Problem 2: Simplify the expression ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) ÷ (x - 1) / (x^2 - 4) -/
theorem simplify_expression2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) : 
  ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 :=
sorry

end simplify_expression1_simplify_expression2_l16_16637


namespace restock_quantities_correct_l16_16274

-- Definition for the quantities of cans required
def cans_peas : ℕ := 810
def cans_carrots : ℕ := 954
def cans_corn : ℕ := 675

-- Definition for the number of cans per box, pack, and case.
def cans_per_box_peas : ℕ := 4
def cans_per_pack_carrots : ℕ := 6
def cans_per_case_corn : ℕ := 5

-- Define the expected order quantities.
def order_boxes_peas : ℕ := 203
def order_packs_carrots : ℕ := 159
def order_cases_corn : ℕ := 135

-- Proof statement for the quantities required to restock exactly.
theorem restock_quantities_correct :
  (order_boxes_peas = Nat.ceil (cans_peas / cans_per_box_peas))
  ∧ (order_packs_carrots = cans_carrots / cans_per_pack_carrots)
  ∧ (order_cases_corn = cans_corn / cans_per_case_corn) :=
by
  sorry

end restock_quantities_correct_l16_16274


namespace total_pieces_of_paper_l16_16161

/-- Definitions according to the problem's conditions -/
def pieces_after_first_cut : Nat := 10

def pieces_after_second_cut (initial_pieces : Nat) : Nat := initial_pieces + 9

def pieces_after_third_cut (after_second_cut_pieces : Nat) : Nat := after_second_cut_pieces + 9

def pieces_after_fourth_cut (after_third_cut_pieces : Nat) : Nat := after_third_cut_pieces + 9

/-- The main theorem stating the desired result -/
theorem total_pieces_of_paper : 
  pieces_after_fourth_cut (pieces_after_third_cut (pieces_after_second_cut pieces_after_first_cut)) = 37 := 
by 
  -- The proof would go here, but it's omitted as per the instructions.
  sorry

end total_pieces_of_paper_l16_16161


namespace example_theorem_l16_16374

def not_a_term : Prop := ∀ n : ℕ, ¬ (24 - 2 * n = 3)

theorem example_theorem : not_a_term :=
  by sorry

end example_theorem_l16_16374


namespace factorize_expression_l16_16966

theorem factorize_expression (x : ℝ) : x * (x - 3) - x + 3 = (x - 1) * (x - 3) :=
by
  sorry

end factorize_expression_l16_16966


namespace marcus_has_210_cards_l16_16802

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the increment of baseball cards Marcus has over Carter
def increment : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + increment

-- Prove that Marcus has 210 baseball cards
theorem marcus_has_210_cards : marcus_cards = 210 :=
by simp [marcus_cards, carter_cards, increment]

end marcus_has_210_cards_l16_16802


namespace combinedAgeIn5Years_l16_16623

variable (Amy Mark Emily : ℕ)

-- Conditions
def amyAge : ℕ := 15
def markAge : ℕ := amyAge + 7
def emilyAge : ℕ := 2 * amyAge

-- Proposition to be proved
theorem combinedAgeIn5Years :
  Amy = amyAge →
  Mark = markAge →
  Emily = emilyAge →
  (Amy + 5) + (Mark + 5) + (Emily + 5) = 82 :=
by
  intros hAmy hMark hEmily
  sorry

end combinedAgeIn5Years_l16_16623


namespace daily_wage_male_worker_l16_16782

variables
  (num_male : ℕ) (num_female : ℕ) (num_child : ℕ)
  (wage_female : ℝ) (wage_child : ℝ) (avg_wage : ℝ)
  (total_workers : ℕ := num_male + num_female + num_child)
  (total_wage_all : ℝ := avg_wage * total_workers)
  (total_wage_female : ℝ := num_female * wage_female)
  (total_wage_child : ℝ := num_child * wage_child)
  (total_wage_male : ℝ := total_wage_all - (total_wage_female + total_wage_child))
  (wage_per_male : ℝ := total_wage_male / num_male)

theorem daily_wage_male_worker :
  num_male = 20 →
  num_female = 15 →
  num_child = 5 →
  wage_female = 20 →
  wage_child = 8 →
  avg_wage = 21 →
  wage_per_male = 25 :=
by
  intros
  sorry

end daily_wage_male_worker_l16_16782


namespace min_value_condition_l16_16354

theorem min_value_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  3 * m + n = 1 → (1 / m + 2 / n) ≥ 5 + 2 * Real.sqrt 6 :=
by
  sorry

end min_value_condition_l16_16354


namespace average_rate_of_change_l16_16709

noncomputable def f (x : ℝ) : ℝ := x^2 + 1

theorem average_rate_of_change (Δx : ℝ) : 
  (f (1 + Δx) - f 1) / Δx = 2 + Δx := 
by
  sorry

end average_rate_of_change_l16_16709


namespace probability_two_different_colors_l16_16744

noncomputable def probability_different_colors (total_balls red_balls black_balls : ℕ) : ℚ :=
  let total_ways := (Finset.range total_balls).card.choose 2
  let diff_color_ways := (Finset.range black_balls).card.choose 1 * (Finset.range red_balls).card.choose 1
  diff_color_ways / total_ways

theorem probability_two_different_colors (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ)
  (h_total : total_balls = 5) (h_red : red_balls = 2) (h_black : black_balls = 3) :
  probability_different_colors total_balls red_balls black_balls = 3 / 5 :=
by
  subst h_total
  subst h_red
  subst h_black
  -- Here the proof would follow using the above definitions and reasoning
  sorry

end probability_two_different_colors_l16_16744


namespace A_3_2_eq_29_l16_16004

-- Define the recursive function A(m, n).
def A : Nat → Nat → Nat
| 0, n => n + 1
| (m + 1), 0 => A m 1
| (m + 1), (n + 1) => A m (A (m + 1) n)

-- Prove that A(3, 2) = 29
theorem A_3_2_eq_29 : A 3 2 = 29 := by 
  sorry

end A_3_2_eq_29_l16_16004


namespace glass_heavier_than_plastic_l16_16000

-- Define the conditions
def condition1 (G : ℕ) : Prop := 3 * G = 600
def condition2 (G P : ℕ) : Prop := 4 * G + 5 * P = 1050

-- Define the theorem to prove
theorem glass_heavier_than_plastic (G P : ℕ) (h1 : condition1 G) (h2 : condition2 G P) : G - P = 150 :=
by
  sorry

end glass_heavier_than_plastic_l16_16000


namespace al_sandwiches_count_l16_16918

noncomputable def total_sandwiches (bread meat cheese : ℕ) : ℕ :=
  bread * meat * cheese

noncomputable def prohibited_combinations (bread_forbidden_combination cheese_forbidden_combination : ℕ) : ℕ := 
  bread_forbidden_combination + cheese_forbidden_combination

theorem al_sandwiches_count (bread meat cheese : ℕ) 
  (bread_forbidden_combination cheese_forbidden_combination : ℕ) 
  (h1 : bread = 5) 
  (h2 : meat = 7) 
  (h3 : cheese = 6) 
  (h4 : bread_forbidden_combination = 5) 
  (h5 : cheese_forbidden_combination = 6) : 
  total_sandwiches bread meat cheese - prohibited_combinations bread_forbidden_combination cheese_forbidden_combination = 199 :=
by
  sorry

end al_sandwiches_count_l16_16918


namespace calculator_press_count_l16_16944

theorem calculator_press_count : 
  ∃ n : ℕ, n ≥ 4 ∧ (2 ^ (2 ^ n)) > 500 := 
by
  sorry

end calculator_press_count_l16_16944


namespace article_cost_price_l16_16349

theorem article_cost_price (SP : ℝ) (CP : ℝ) (h1 : SP = 455) (h2 : SP = CP + 0.3 * CP) : CP = 350 :=
by sorry

end article_cost_price_l16_16349


namespace time_ratio_l16_16611

theorem time_ratio (A : ℝ) (B : ℝ) (h1 : B = 18) (h2 : 1 / A + 1 / B = 1 / 3) : A / B = 1 / 5 :=
by
  sorry

end time_ratio_l16_16611


namespace triangle_area_of_parabola_intersection_l16_16521

theorem triangle_area_of_parabola_intersection
  (line_passes_through : ∃ (p : ℝ × ℝ), p = (0, -2))
  (parabola_intersection : ∃ (x1 y1 x2 y2 : ℝ),
    (x1, y1) ∈ {p : ℝ × ℝ | p.snd ^ 2 = 16 * p.fst} ∧
    (x2, y2) ∈ {p : ℝ × ℝ | p.snd ^ 2 = 16 * p.fst})
  (y_cond : ∃ (y1 y2 : ℝ), y1 ^ 2 - y2 ^ 2 = 1) :
  ∃ (area : ℝ), area = 1 / 16 :=
by
  sorry

end triangle_area_of_parabola_intersection_l16_16521


namespace solve_for_q_l16_16062

variable (R t m q : ℝ)

def given_condition : Prop :=
  R = t / ((2 + m) ^ q)

theorem solve_for_q (h : given_condition R t m q) : 
  q = (Real.log (t / R)) / (Real.log (2 + m)) := 
sorry

end solve_for_q_l16_16062


namespace maximum_value_of_2x_plus_y_l16_16201

noncomputable def max_value_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) : ℝ :=
  (2 * x + y)

theorem maximum_value_of_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) :
  max_value_2x_plus_y x y h ≤ (2 * Real.sqrt 10) / 5 :=
sorry

end maximum_value_of_2x_plus_y_l16_16201


namespace john_saving_yearly_l16_16065

def old_monthly_cost : ℕ := 1200
def increase_percentage : ℕ := 40
def split_count : ℕ := 3

def old_annual_cost (monthly_cost : ℕ) := monthly_cost * 12
def new_monthly_cost (monthly_cost : ℕ) (percentage : ℕ) := monthly_cost * (100 + percentage) / 100
def new_monthly_share (new_cost : ℕ) (split : ℕ) := new_cost / split
def new_annual_cost (monthly_share : ℕ) := monthly_share * 12
def annual_savings (old_annual : ℕ) (new_annual : ℕ) := old_annual - new_annual

theorem john_saving_yearly 
  (old_cost : ℕ := old_monthly_cost)
  (increase : ℕ := increase_percentage)
  (split : ℕ := split_count) :
  annual_savings (old_annual_cost old_cost) 
                 (new_annual_cost (new_monthly_share (new_monthly_cost old_cost increase) split)) 
  = 7680 :=
by
  sorry

end john_saving_yearly_l16_16065


namespace problem_solution_l16_16684

theorem problem_solution :
  ∀ (x y z : ℤ),
  4 * x + y + z = 80 →
  3 * x + y - z = 20 →
  x = 20 →
  2 * x - y - z = 40 :=
by
  intros x y z h1 h2 hx
  rw [hx] at h1 h2
  -- Here you could continue solving but we'll use sorry to indicate the end as no proof is requested.
  sorry

end problem_solution_l16_16684


namespace quadratic_inequality_solution_l16_16547

theorem quadratic_inequality_solution (x : ℝ) :
    -15 * x^2 + 10 * x + 5 > 0 ↔ (-1 / 3 : ℝ) < x ∧ x < 1 :=
by
  sorry

end quadratic_inequality_solution_l16_16547


namespace smallest_integer_to_make_perfect_square_l16_16006

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_integer_to_make_perfect_square :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, (n * y) = k^2) ∧ n = 6 :=
by
  sorry

end smallest_integer_to_make_perfect_square_l16_16006


namespace sector_area_15deg_radius_6cm_l16_16378

noncomputable def sector_area (r : ℝ) (theta : ℝ) : ℝ :=
  0.5 * theta * r^2

theorem sector_area_15deg_radius_6cm :
  sector_area 6 (15 * Real.pi / 180) = 3 * Real.pi / 2 := by
  sorry

end sector_area_15deg_radius_6cm_l16_16378


namespace numMilkmen_rented_pasture_l16_16087

def cowMonths (cows: ℕ) (months: ℕ) : ℕ := cows * months

def totalCowMonths (a: ℕ) (b: ℕ) (c: ℕ) (d: ℕ) : ℕ := a + b + c + d

noncomputable def rentPerCowMonth (share: ℕ) (cowMonths: ℕ) : ℕ := 
  share / cowMonths

theorem numMilkmen_rented_pasture 
  (a_cows: ℕ) (a_months: ℕ) (b_cows: ℕ) (b_months: ℕ) (c_cows: ℕ) (c_months: ℕ) (d_cows: ℕ) (d_months: ℕ)
  (a_share: ℕ) (total_rent: ℕ) 
  (ha: a_cows = 24) (hma: a_months = 3) 
  (hb: b_cows = 10) (hmb: b_months = 5)
  (hc: c_cows = 35) (hmc: c_months = 4)
  (hd: d_cows = 21) (hmd: d_months = 3)
  (ha_share: a_share = 720) (htotal_rent: total_rent = 3250)
  : 4 = 4 := by
  sorry

end numMilkmen_rented_pasture_l16_16087


namespace maximize_cubic_quartic_l16_16031

theorem maximize_cubic_quartic (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + 2 * y = 35) : 
  (x, y) = (21, 7) ↔ x^3 * y^4 = (21:ℝ)^3 * (7:ℝ)^4 := 
by
  sorry

end maximize_cubic_quartic_l16_16031


namespace chromium_percentage_in_new_alloy_l16_16526

theorem chromium_percentage_in_new_alloy :
  ∀ (weight1 weight2 chromium1 chromium2: ℝ),
  weight1 = 15 → weight2 = 35 → chromium1 = 0.12 → chromium2 = 0.08 →
  (chromium1 * weight1 + chromium2 * weight2) / (weight1 + weight2) * 100 = 9.2 :=
by
  intros weight1 weight2 chromium1 chromium2 hweight1 hweight2 hchromium1 hchromium2
  sorry

end chromium_percentage_in_new_alloy_l16_16526


namespace total_girls_in_circle_l16_16162

theorem total_girls_in_circle (girls : Nat) 
  (h1 : (4 + 7) = girls + 2) : girls = 11 := 
by
  sorry

end total_girls_in_circle_l16_16162


namespace min_value_of_sequence_l16_16300

theorem min_value_of_sequence 
  (a : ℤ) 
  (a_sequence : ℕ → ℤ) 
  (h₀ : a_sequence 0 = a)
  (h_rec : ∀ n, a_sequence (n + 1) = 2 * a_sequence n - n ^ 2)
  (h_pos : ∀ n, a_sequence n > 0) :
  ∃ k, a_sequence k = 3 := 
sorry

end min_value_of_sequence_l16_16300


namespace percentage_of_smoking_teens_l16_16410

theorem percentage_of_smoking_teens (total_students : ℕ) (hospitalized_percentage : ℝ) (non_hospitalized_count : ℕ) 
  (h_total_students : total_students = 300)
  (h_hospitalized_percentage : hospitalized_percentage = 0.70)
  (h_non_hospitalized_count : non_hospitalized_count = 36) : 
  (non_hospitalized_count / (total_students * (1 - hospitalized_percentage))) * 100 = 40 := 
by 
  sorry

end percentage_of_smoking_teens_l16_16410


namespace square_area_ratio_l16_16343

theorem square_area_ratio (n : ℕ) (s₁ s₂: ℕ) (h1 : s₁ = 1) (h2 : s₂ = n^2) (h3 : 2 * s₂ - 1 = 17) :
  s₂ = 81 := 
sorry

end square_area_ratio_l16_16343


namespace arithmetic_mean_of_sequence_beginning_at_5_l16_16293

def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

def sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def arithmetic_mean (a d n : ℕ) : ℚ :=
  sequence_sum a d n / n

theorem arithmetic_mean_of_sequence_beginning_at_5 : 
  arithmetic_mean 5 1 60 = 34.5 :=
by
  sorry

end arithmetic_mean_of_sequence_beginning_at_5_l16_16293


namespace sqrt_product_eq_six_l16_16241

theorem sqrt_product_eq_six (sqrt24 sqrtThreeOverTwo: ℝ)
    (h1 : sqrt24 = Real.sqrt 24)
    (h2 : sqrtThreeOverTwo = Real.sqrt (3 / 2))
    : sqrt24 * sqrtThreeOverTwo = 6 := by
  sorry

end sqrt_product_eq_six_l16_16241


namespace banana_count_l16_16933

theorem banana_count : (2 + 7) = 9 := by
  rfl

end banana_count_l16_16933


namespace rent_percentage_l16_16372

variable (E : ℝ)
variable (last_year_rent : ℝ := 0.20 * E)
variable (this_year_earnings : ℝ := 1.20 * E)
variable (this_year_rent : ℝ := 0.30 * this_year_earnings)

theorem rent_percentage (E : ℝ) (h_last_year_rent : last_year_rent = 0.20 * E)
  (h_this_year_earnings : this_year_earnings = 1.20 * E)
  (h_this_year_rent : this_year_rent = 0.30 * this_year_earnings) : 
  this_year_rent / last_year_rent * 100 = 180 := by
  sorry

end rent_percentage_l16_16372


namespace train_length_l16_16938

-- Definitions based on conditions
def faster_train_speed := 46 -- speed in km/hr
def slower_train_speed := 36 -- speed in km/hr
def time_to_pass := 72 -- time in seconds
def relative_speed_kmph := faster_train_speed - slower_train_speed
def relative_speed_mps : ℚ := (relative_speed_kmph * 1000) / 3600

theorem train_length :
  ∃ L : ℚ, (2 * L = relative_speed_mps * time_to_pass / 1) ∧ L = 100 := 
by
  sorry

end train_length_l16_16938


namespace remaining_milk_and_coffee_l16_16321

/-- 
Given:
1. A cup initially contains 1 glass of coffee.
2. A quarter glass of milk is added to the cup.
3. The mixture is thoroughly stirred.
4. One glass of the mixture is poured back.

Prove:
The remaining content in the cup is 1/5 glass of milk and 4/5 glass of coffee. 
--/
theorem remaining_milk_and_coffee :
  let coffee_initial := 1  -- initial volume of coffee
  let milk_added := 1 / 4  -- volume of milk added
  let total_volume := coffee_initial + milk_added  -- total volume after mixing = 5/4 glasses
  let milk_fraction := milk_added / total_volume  -- fraction of milk in the mixture = 1/5
  let coffee_fraction := coffee_initial / total_volume  -- fraction of coffee in the mixture = 4/5
  let volume_poured := 1 / 4  -- volume of mixture poured out
  let milk_poured := (milk_fraction * volume_poured : ℝ)  -- volume of milk poured out = 1/20 glass
  let coffee_poured := (coffee_fraction * volume_poured : ℝ)  -- volume of coffee poured out = 1/5 glass
  let remaining_milk := milk_added - milk_poured  -- remaining volume of milk = 1/5 glass
  let remaining_coffee := coffee_initial - coffee_poured  -- remaining volume of coffee = 4/5 glass
  remaining_milk = 1 / 5 ∧ remaining_coffee = 4 / 5 :=
by
  sorry

end remaining_milk_and_coffee_l16_16321


namespace ratio_of_x_intercepts_l16_16506

theorem ratio_of_x_intercepts (b : ℝ) (hb: b ≠ 0) (u v: ℝ) (h₁: 8 * u + b = 0) (h₂: 4 * v + b = 0) : 
  u / v = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l16_16506


namespace price_per_unit_max_profit_l16_16862

-- Part 1: Finding the Prices

theorem price_per_unit (x y : ℕ) 
  (h1 : 2 * x + 3 * y = 690) 
  (h2 : x + 4 * y = 720) : 
  x = 120 ∧ y = 150 :=
by
  sorry

-- Part 2: Maximizing Profit

theorem max_profit (m : ℕ) 
  (h1 : m ≤ 3 * (40 - m)) 
  (h2 : 120 * m + 150 * (40 - m) ≤ 5400) : 
  (m = 20) ∧ (40 - m = 20) :=
by
  sorry

end price_per_unit_max_profit_l16_16862


namespace subset_a_eq_1_l16_16484

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_a_eq_1 (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_a_eq_1_l16_16484


namespace frank_eats_each_day_l16_16951

theorem frank_eats_each_day :
  ∀ (cookies_per_tray cookies_per_day days ted_eats remaining_cookies : ℕ),
  cookies_per_tray = 12 →
  cookies_per_day = 2 →
  days = 6 →
  ted_eats = 4 →
  remaining_cookies = 134 →
  (2 * cookies_per_tray * days) - (ted_eats + remaining_cookies) / days = 1 :=
  by
    intros cookies_per_tray cookies_per_day days ted_eats remaining_cookies ht hc hd hted hr
    sorry

end frank_eats_each_day_l16_16951


namespace next_leap_year_visible_after_2017_l16_16642

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0) ∧ ((y % 100 ≠ 0) ∨ (y % 400 = 0))

def stromquist_visible (start_year interval next_leap : ℕ) : Prop :=
  ∃ k : ℕ, next_leap = start_year + k * interval ∧ is_leap_year next_leap

theorem next_leap_year_visible_after_2017 :
  stromquist_visible 2017 61 2444 :=
  sorry

end next_leap_year_visible_after_2017_l16_16642


namespace monthly_salary_l16_16560

theorem monthly_salary (S : ℝ) (E : ℝ) 
  (h1 : S - 1.20 * E = 220)
  (h2 : E = 0.80 * S) :
  S = 5500 :=
by
  sorry

end monthly_salary_l16_16560


namespace combined_future_value_l16_16499

noncomputable def future_value (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem combined_future_value :
  let A1 := future_value 3000 0.05 3
  let A2 := future_value 5000 0.06 4
  let A3 := future_value 7000 0.07 5
  A1 + A2 + A3 = 19603.119 :=
by
  sorry

end combined_future_value_l16_16499


namespace solve_for_x_l16_16054

def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

theorem solve_for_x (x : ℝ) : star 6 x = 45 ↔ x = 19 / 3 := by
  sorry

end solve_for_x_l16_16054


namespace find_triples_l16_16870

-- Define the conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def power_of_p (p n : ℕ) : Prop := ∃ (k : ℕ), n = p^k

-- Given the conditions
variable (p x y : ℕ)
variable (h_prime : is_prime p)
variable (h_pos_x : x > 0)
variable (h_pos_y : y > 0)

-- The problem statement
theorem find_triples (h1 : power_of_p p (x^(p-1) + y)) (h2 : power_of_p p (x + y^(p-1))) : 
  (p = 3 ∧ x = 2 ∧ y = 5) ∨
  (p = 3 ∧ x = 5 ∧ y = 2) ∨
  (p = 2 ∧ ∃ (n i : ℕ), n > 0 ∧ i > 0 ∧ x = n ∧ y = 2^i - n ∧ 0 < n ∧ n < 2^i) := 
sorry

end find_triples_l16_16870


namespace vector_problem_solution_l16_16932

variables (a b c : ℤ × ℤ) (m n : ℤ)

def parallel (v1 v2 : ℤ × ℤ) : Prop := v1.1 * v2.2 = v1.2 * v2.1
def perpendicular (v1 v2 : ℤ × ℤ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem vector_problem_solution
  (a_eq : a = (1, -2))
  (b_eq : b = (2, m - 1))
  (c_eq : c = (4, n))
  (h1 : parallel a b)
  (h2 : perpendicular b c) :
  m + n = -1 := by
  sorry

end vector_problem_solution_l16_16932


namespace circle_intersects_cells_l16_16598

/-- On a grid with 1 cm x 1 cm cells, a circle with a radius of 100 cm is drawn.
    The circle does not pass through any vertices of the cells and does not touch the sides of the cells.
    Prove that the number of cells the circle can intersect is either 800 or 799. -/
theorem circle_intersects_cells (r : ℝ) (gsize : ℝ) (cells : ℕ) :
  r = 100 ∧ gsize = 1 ∧ cells = 800 ∨ cells = 799 :=
by
  sorry

end circle_intersects_cells_l16_16598


namespace xyz_value_l16_16109

variable {x y z : ℝ}

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 27)
                  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) :
                x * y * z = 6 := by
  sorry

end xyz_value_l16_16109


namespace lines_parallel_if_perpendicular_to_same_plane_l16_16399

-- Definitions and conditions
variables {Point : Type*} [MetricSpace Point]
variables {Line Plane : Type*}

def is_parallel (l₁ l₂ : Line) : Prop := sorry
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry

variables (m n : Line) (α : Plane)

-- Theorem statement
theorem lines_parallel_if_perpendicular_to_same_plane :
  is_perpendicular m α → is_perpendicular n α → is_parallel m n :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l16_16399


namespace hyperbola_eccentricity_l16_16336

noncomputable def calculate_eccentricity (a b c x0 y0 : ℝ) : ℝ :=
  c / a

theorem hyperbola_eccentricity :
  ∀ (a b c x0 y0 : ℝ),
    (c = 2) →
    (a^2 + b^2 = 4) →
    (x0 = 3) →
    (y0^2 = 24) →
    (5 = x0 + 2) →
    calculate_eccentricity a b c x0 y0 = 2 := 
by 
  intros a b c x0 y0 h1 h2 h3 h4 h5
  sorry

end hyperbola_eccentricity_l16_16336


namespace triangle_area_of_integral_sides_with_perimeter_8_l16_16259

theorem triangle_area_of_integral_sides_with_perimeter_8 :
  ∃ (a b c : ℕ), a + b + c = 8 ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ 
  ∃ (area : ℝ), area = 2 * Real.sqrt 2 := by
  sorry

end triangle_area_of_integral_sides_with_perimeter_8_l16_16259


namespace toothpick_problem_l16_16940

theorem toothpick_problem : 
  ∃ (N : ℕ), N > 5000 ∧ 
            N % 10 = 9 ∧ 
            N % 9 = 8 ∧ 
            N % 8 = 7 ∧ 
            N % 7 = 6 ∧ 
            N % 6 = 5 ∧ 
            N % 5 = 4 ∧ 
            N = 5039 :=
by
  sorry

end toothpick_problem_l16_16940


namespace AlyssaBottleCaps_l16_16209

def bottleCapsKatherine := 34
def bottleCapsGivenAway (bottleCaps: ℕ) := bottleCaps / 2
def bottleCapsLost (bottleCaps: ℕ) := bottleCaps - 8

theorem AlyssaBottleCaps : bottleCapsLost (bottleCapsGivenAway bottleCapsKatherine) = 9 := 
  by 
  sorry

end AlyssaBottleCaps_l16_16209


namespace paddington_more_goats_l16_16044

theorem paddington_more_goats (W P total : ℕ) (hW : W = 140) (hTotal : total = 320) (hTotalGoats : W + P = total) : P - W = 40 :=
by
  sorry

end paddington_more_goats_l16_16044


namespace min_x_div_y_l16_16703

theorem min_x_div_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + y = 2) : ∃c: ℝ, c = 1 ∧ ∀(a: ℝ), x = a → y = 1 → a/y ≥ c :=
by
  sorry

end min_x_div_y_l16_16703


namespace volume_diff_proof_l16_16987

def volume_difference (x y z x' y' z' : ℝ) : ℝ := x * y * z - x' * y' * z'

theorem volume_diff_proof : 
  (∃ (x y z x' y' z' : ℝ),
    2 * (x + y) = 12 ∧ 2 * (x + z) = 16 ∧ 2 * (y + z) = 24 ∧
    2 * (x' + y') = 12 ∧ 2 * (x' + z') = 16 ∧ 2 * (y' + z') = 20 ∧
    volume_difference x y z x' y' z' = -13) :=
by {
  sorry
}

end volume_diff_proof_l16_16987


namespace length_of_one_side_nonagon_l16_16152

def total_perimeter (n : ℕ) (side_length : ℝ) : ℝ := n * side_length

theorem length_of_one_side_nonagon (total_perimeter : ℝ) (n : ℕ) (side_length : ℝ) (h1 : n = 9) (h2 : total_perimeter = 171) : side_length = 19 :=
by
  sorry

end length_of_one_side_nonagon_l16_16152


namespace ln_of_gt_of_pos_l16_16013

variable {a b : ℝ}

theorem ln_of_gt_of_pos (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b :=
sorry

end ln_of_gt_of_pos_l16_16013


namespace total_handshakes_is_316_l16_16864

def number_of_couples : ℕ := 15
def number_of_people : ℕ := number_of_couples * 2

def handshakes_among_men (n : ℕ) : ℕ := n * (n - 1) / 2
def handshakes_men_women (n : ℕ) : ℕ := n * (n - 1)
def handshakes_between_women : ℕ := 1
def total_handshakes (n : ℕ) : ℕ := handshakes_among_men n + handshakes_men_women n + handshakes_between_women

theorem total_handshakes_is_316 : total_handshakes number_of_couples = 316 :=
by
  sorry

end total_handshakes_is_316_l16_16864


namespace divisor_of_930_l16_16272

theorem divisor_of_930 : ∃ d > 1, d ∣ 930 ∧ ∀ e, e ∣ 930 → e > 1 → d ≤ e :=
by
  sorry

end divisor_of_930_l16_16272


namespace rectangle_area_increase_l16_16105

variable {L W : ℝ} -- Define variables for length and width

theorem rectangle_area_increase (p : ℝ) (hW : W' = 0.4 * W) (hA : A' = 1.36 * (L * W)) :
  L' = L + (240 / 100) * L :=
by
  sorry

end rectangle_area_increase_l16_16105


namespace shaded_area_a_length_EF_b_length_EF_c_ratio_ab_d_l16_16273

-- (a) Prove that the area of the shaded region is 36 cm^2
theorem shaded_area_a (AB EF : ℕ) (h1 : AB = 10) (h2 : EF = 8) : (AB ^ 2) - (EF ^ 2) = 36 :=
by
  sorry

-- (b) Prove that the length of EF is 7 cm
theorem length_EF_b (AB : ℕ) (shaded_area : ℕ) (h1 : AB = 13) (h2 : shaded_area = 120)
  : ∃ EF, (AB ^ 2) - (EF ^ 2) = shaded_area ∧ EF = 7 :=
by
  sorry

-- (c) Prove that the length of EF is 9 cm
theorem length_EF_c (AB : ℕ) (h1 : AB = 18)
  : ∃ EF, (AB ^ 2) - ((1 / 4) * AB ^ 2) = (3 / 4) * AB ^ 2 ∧ EF = 9 :=
by
  sorry

-- (d) Prove that a / b = 5 / 3
theorem ratio_ab_d (a b : ℕ) (shaded_percent : ℚ) (h1 : shaded_percent = 0.64)
  : (a ^ 2) - ((0.36) * a ^ 2) = (a ^ 2) * shaded_percent ∧ (a / b) = (5 / 3) :=
by
  sorry

end shaded_area_a_length_EF_b_length_EF_c_ratio_ab_d_l16_16273


namespace solution_to_problem_l16_16124

theorem solution_to_problem (f : ℕ → ℕ) 
  (h1 : f 2 = 20)
  (h2 : ∀ n : ℕ, 0 < n → f (2 * n) + n * f 2 = f (2 * n + 2)) :
  f 10 = 220 :=
by
  sorry

end solution_to_problem_l16_16124


namespace parabola_focus_distance_l16_16318

theorem parabola_focus_distance (p : ℝ) (y₀ : ℝ) (h₀ : p > 0) 
  (h₁ : y₀^2 = 2 * p * 4) 
  (h₂ : dist (4, y₀) (p/2, 0) = 3/2 * p) : 
  p = 4 := 
sorry

end parabola_focus_distance_l16_16318


namespace range_of_a_l16_16067

noncomputable def f (a x : ℝ) : ℝ := x^2 - a * x + a + 3
noncomputable def g (a x : ℝ) : ℝ := a * x - 2 * a

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, f a x₀ < 0 ∧ g a x₀ < 0) → 7 < a :=
by
  intro h
  sorry

end range_of_a_l16_16067


namespace inequality_holds_for_all_x_iff_m_eq_1_l16_16121

theorem inequality_holds_for_all_x_iff_m_eq_1 (m : ℝ) (h_m : m ≠ 0) :
  (∀ x > 0, x^2 - 2 * m * Real.log x ≥ 1) ↔ m = 1 :=
by
  sorry

end inequality_holds_for_all_x_iff_m_eq_1_l16_16121


namespace gcd_f_50_51_l16_16465

-- Define f(x)
def f (x : ℤ) : ℤ := x^3 - x^2 + 2 * x + 2000

-- State the problem: Prove gcd(f(50), f(51)) = 8
theorem gcd_f_50_51 : Int.gcd (f 50) (f 51) = 8 := by
  sorry

end gcd_f_50_51_l16_16465


namespace marys_balloons_l16_16100

theorem marys_balloons (x y : ℝ) (h1 : x = 4 * y) (h2 : x = 7.0) : y = 1.75 := by
  sorry

end marys_balloons_l16_16100


namespace robie_initial_cards_l16_16043

def total_initial_boxes : Nat := 2 + 5
def cards_per_box : Nat := 10
def unboxed_cards : Nat := 5

theorem robie_initial_cards :
  (total_initial_boxes * cards_per_box + unboxed_cards) = 75 :=
by
  sorry

end robie_initial_cards_l16_16043


namespace find_B_share_l16_16880

theorem find_B_share (x : ℕ) (x_pos : 0 < x) (C_share_difference : 5 * x = 4 * x + 1000) (B_share_eq : 3 * x = B) : B = 3000 :=
by
  sorry

end find_B_share_l16_16880


namespace arithmetic_sequence_sum_l16_16747

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (k : ℕ) :
  a 1 = 1 →
  (∀ n, a (n + 1) = a n + 2) →
  (∀ n, S n = n * n) →
  S (k + 2) - S k = 24 →
  k = 5 :=
by
  intros a1 ha hS hSk
  sorry

end arithmetic_sequence_sum_l16_16747


namespace three_liters_to_gallons_l16_16939

theorem three_liters_to_gallons :
  (0.5 : ℝ) * 3 * 0.1319 = 0.7914 := by
  sorry

end three_liters_to_gallons_l16_16939


namespace smallest_positive_period_max_value_in_interval_l16_16811

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 - Real.cos (2 * x + Real.pi / 3)

theorem smallest_positive_period :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = Real.pi :=
sorry

theorem max_value_in_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 5 / 2 :=
sorry

end smallest_positive_period_max_value_in_interval_l16_16811


namespace red_pens_count_l16_16183

theorem red_pens_count (R : ℕ) : 
  (∃ (black_pens blue_pens : ℕ), 
  black_pens = R + 10 ∧ 
  blue_pens = R + 7 ∧ 
  R + black_pens + blue_pens = 41) → 
  R = 8 := by
  sorry

end red_pens_count_l16_16183


namespace arc_length_l16_16716

theorem arc_length (C : ℝ) (theta : ℝ) (hC : C = 100) (htheta : theta = 30) :
  (theta / 360) * C = 25 / 3 :=
by sorry

end arc_length_l16_16716


namespace expression_value_l16_16676

theorem expression_value :
  ( (120^2 - 13^2) / (90^2 - 19^2) * ((90 - 19) * (90 + 19)) / ((120 - 13) * (120 + 13)) ) = 1 := by
  sorry

end expression_value_l16_16676


namespace smallest_positive_integer_l16_16855

theorem smallest_positive_integer :
  ∃ (n a b m : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ n = 153846 ∧
  (n = 10^m * a + b) ∧
  (7 * n = 2 * (10 * b + a)) :=
by
  sorry

end smallest_positive_integer_l16_16855


namespace inequality_proof_l16_16935

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x / Real.sqrt y + y / Real.sqrt x) ≥ (Real.sqrt x + Real.sqrt y) := 
sorry

end inequality_proof_l16_16935


namespace kite_height_30_sqrt_43_l16_16907

theorem kite_height_30_sqrt_43
  (c d h : ℝ)
  (h1 : h^2 + c^2 = 170^2)
  (h2 : h^2 + d^2 = 150^2)
  (h3 : c^2 + d^2 = 160^2) :
  h = 30 * Real.sqrt 43 := by
  sorry

end kite_height_30_sqrt_43_l16_16907


namespace no_solution_integral_pairs_l16_16731

theorem no_solution_integral_pairs (a b : ℤ) : (1 / (a : ℚ) + 1 / (b : ℚ) = -1 / (a + b : ℚ)) → false :=
by
  sorry

end no_solution_integral_pairs_l16_16731


namespace fraction_to_decimal_l16_16586

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fraction_to_decimal_l16_16586


namespace exist_a_b_not_triangle_l16_16793

theorem exist_a_b_not_triangle (h₁ : ∀ a b : ℕ, (a > 1000) → (b > 1000) →
  ∃ c : ℕ, (∃ (k : ℕ), c = k * k) →
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  ∃ (a b : ℕ), (a > 1000 ∧ b > 1000) ∧ 
  ∀ c : ℕ, (∃ (k : ℕ), c = k * k) →
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
sorry

end exist_a_b_not_triangle_l16_16793


namespace rectangle_perimeter_l16_16025

variables (L B P : ℝ)

theorem rectangle_perimeter (h1 : B = 0.60 * L) (h2 : L * B = 37500) : P = 800 :=
by
  sorry

end rectangle_perimeter_l16_16025


namespace neg_three_lt_neg_sqrt_eight_l16_16333

theorem neg_three_lt_neg_sqrt_eight : -3 < -Real.sqrt 8 := 
sorry

end neg_three_lt_neg_sqrt_eight_l16_16333


namespace distance_inequality_solution_l16_16227

theorem distance_inequality_solution (x : ℝ) (h : |x| > |x + 1|) : x < -1 / 2 :=
sorry

end distance_inequality_solution_l16_16227


namespace inequality_bound_l16_16361

theorem inequality_bound 
  (a b c d : ℝ) 
  (ha : 0 ≤ a) (hb : a ≤ 1)
  (hb : 0 ≤ b) (hc : b ≤ 1)
  (hc : 0 ≤ c) (hd : c ≤ 1)
  (hd : 0 ≤ d) (ha2 : d ≤ 1) : 
  ab * (a - b) + bc * (b - c) + cd * (c - d) + da * (d - a) ≤ 8/27 := 
by
  sorry

end inequality_bound_l16_16361


namespace greatest_integer_y_l16_16357

-- Define the fraction and inequality condition
def inequality_condition (y : ℤ) : Prop := 8 * 17 > 11 * y

-- Prove the greatest integer y satisfying the condition is 12
theorem greatest_integer_y : ∃ y : ℤ, inequality_condition y ∧ (∀ z : ℤ, inequality_condition z → z ≤ y) ∧ y = 12 :=
by
  exists 12
  sorry

end greatest_integer_y_l16_16357


namespace probability_of_orange_face_l16_16799

theorem probability_of_orange_face :
  ∃ (G O P : ℕ) (total_faces : ℕ), total_faces = 10 ∧ G = 5 ∧ O = 3 ∧ P = 2 ∧
  (O / total_faces : ℚ) = 3 / 10 := by 
  sorry

end probability_of_orange_face_l16_16799


namespace part_I_part_II_l16_16178

def S_n (n : ℕ) : ℕ := sorry
def a_n (n : ℕ) : ℕ := sorry

theorem part_I (n : ℕ) (h1 : 2 * S_n n = 3^n + 3) :
  a_n n = if n = 1 then 3 else 3^(n-1) :=
sorry

theorem part_II (n : ℕ) (h1 : a_n 1 = 1) (h2 : ∀ n : ℕ, a_n (n + 1) - a_n n = 2^n) :
  S_n n = 2^(n + 1) - n - 2 :=
sorry

end part_I_part_II_l16_16178


namespace find_distance_city_A_B_l16_16088

-- Variables and givens
variable (D : ℝ)

-- Conditions from the problem
variable (JohnSpeed : ℝ := 40) (LewisSpeed : ℝ := 60)
variable (MeetDistance : ℝ := 160)
variable (TimeJohn : ℝ := (D - MeetDistance) / JohnSpeed)
variable (TimeLewis : ℝ := (D + MeetDistance) / LewisSpeed)

-- Lean 4 theorem statement for the proof
theorem find_distance_city_A_B :
  TimeJohn = TimeLewis → D = 800 :=
by
  sorry

end find_distance_city_A_B_l16_16088


namespace division_by_power_of_ten_l16_16562

theorem division_by_power_of_ten (a b : ℕ) (h_a : a = 10^7) (h_b : b = 5 * 10^4) : a / b = 200 := by
  sorry

end division_by_power_of_ten_l16_16562


namespace dice_probability_correct_l16_16711

noncomputable def probability_at_least_one_two_or_three : ℚ :=
  let total_outcomes := 64
  let favorable_outcomes := 64 - 36
  favorable_outcomes / total_outcomes

theorem dice_probability_correct :
  probability_at_least_one_two_or_three = 7 / 16 :=
by
  -- Proof will be provided here
  sorry

end dice_probability_correct_l16_16711


namespace square_area_from_diagonal_l16_16425

theorem square_area_from_diagonal :
  ∀ (d : ℝ), d = 10 * Real.sqrt 2 → (d / Real.sqrt 2) ^ 2 = 100 :=
by
  intros d hd
  sorry -- Skipping the proof

end square_area_from_diagonal_l16_16425


namespace triangle_A1B1C1_sides_l16_16008

theorem triangle_A1B1C1_sides
  (a b c x y z R : ℝ) 
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_positive_c : c > 0)
  (h_positive_x : x > 0)
  (h_positive_y : y > 0)
  (h_positive_z : z > 0)
  (h_positive_R : R > 0) :
  (↑a * ↑y / (2 * ↑R), ↑b * ↑z / (2 * ↑R), ↑c * ↑x / (2 * ↑R)) = (↑c * ↑x / (2 * ↑R), ↑a * ↑y / (2 * ↑R), ↑b * ↑z / (2 * ↑R)) :=
by sorry

end triangle_A1B1C1_sides_l16_16008


namespace difference_of_squares_example_l16_16835

theorem difference_of_squares_example (a b : ℕ) (h1 : a = 305) (h2 : b = 295) :
  (a^2 - b^2) / 10 = 600 :=
by
  sorry

end difference_of_squares_example_l16_16835


namespace B_age_l16_16430

-- Define the conditions
variables (x y : ℕ)
variable (current_year : ℕ)
axiom h1 : 10 * x + y + 4 = 43
axiom reference_year : current_year = 1955

-- Define the relationship between the digit equation and the year
def birth_year (x y : ℕ) : ℕ := 1900 + 10 * x + y

-- Birth year calculation
def age (current_year birth_year : ℕ) : ℕ := current_year - birth_year

-- Final theorem: Age of B
theorem B_age (x y : ℕ) (current_year : ℕ) (h1 : 10 * x  + y + 4 = 43) (reference_year : current_year = 1955) :
  age current_year (birth_year x y) = 16 :=
by
  sorry

end B_age_l16_16430


namespace part1_part2_l16_16784

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 2| + |x - a|

-- Prove part 1: For all x in ℝ, log(f(x, -8)) ≥ 1
theorem part1 : ∀ x : ℝ, Real.log (f x (-8)) ≥ 1 :=
by 
  sorry

-- Prove part 2: For all x in ℝ, if f(x,a) ≥ a, then a ≤ 1
theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ a) → a ≤ 1 :=
by
  sorry

end part1_part2_l16_16784


namespace calculate_expression_l16_16818

theorem calculate_expression : 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) = 3^16 - 1 :=
by 
  sorry

end calculate_expression_l16_16818


namespace cute_pairs_count_l16_16271

def is_cute_pair (a b : ℕ) : Prop :=
  a ≥ b / 2 + 7 ∧ b ≥ a / 2 + 7

def max_cute_pairs : Prop :=
  ∀ (ages : Finset ℕ), 
  (∀ x ∈ ages, 1 ≤ x ∧ x ≤ 100) →
  (∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ pair ∈ pairs, is_cute_pair pair.1 pair.2) ∧
    (∀ x ∈ pairs, ∀ y ∈ pairs, x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2) ∧
    pairs.card = 43)

theorem cute_pairs_count : max_cute_pairs := 
sorry

end cute_pairs_count_l16_16271


namespace correct_operation_is_d_l16_16720

theorem correct_operation_is_d (a b : ℝ) : 
  (∀ x y : ℝ, -x * y = -(x * y)) → 
  (∀ x : ℝ, x⁻¹ * (x ^ 2) = x) → 
  (∀ x : ℝ, x ^ 10 / x ^ 4 = x ^ 6) →
  ((a - b) * (-a - b) ≠ a ^ 2 - b ^ 2) ∧ 
  (2 * a ^ 2 * a ^ 3 ≠ 2 * a ^ 6) ∧ 
  ((-a) ^ 10 / (-a) ^ 4 = a ^ 6) :=
by
  intros h1 h2 h3
  sorry

end correct_operation_is_d_l16_16720


namespace white_ring_weight_l16_16566

def weight_of_orange_ring : ℝ := 0.08
def weight_of_purple_ring : ℝ := 0.33
def total_weight_of_rings : ℝ := 0.83

def weight_of_white_ring (total : ℝ) (orange : ℝ) (purple : ℝ) : ℝ :=
  total - (orange + purple)

theorem white_ring_weight :
  weight_of_white_ring total_weight_of_rings weight_of_orange_ring weight_of_purple_ring = 0.42 :=
by
  sorry

end white_ring_weight_l16_16566


namespace not_possible_coloring_l16_16364

def color : Nat → Option ℕ := sorry

def all_colors_used (f : Nat → Option ℕ) : Prop := 
  (∃ n, f n = some 0) ∧ (∃ n, f n = some 1) ∧ (∃ n, f n = some 2)

def valid_coloring (f : Nat → Option ℕ) : Prop :=
  ∀ (a b : Nat), 1 < a → 1 < b → f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b

theorem not_possible_coloring : ¬ (∃ f : Nat → Option ℕ, all_colors_used f ∧ valid_coloring f) := 
sorry

end not_possible_coloring_l16_16364


namespace person_b_worked_alone_days_l16_16233

theorem person_b_worked_alone_days :
  ∀ (x : ℕ), 
  (x / 10 + (12 - x) / 20 = 1) → x = 8 :=
by
  sorry

end person_b_worked_alone_days_l16_16233


namespace tangent_line_eq_extreme_values_range_of_a_l16_16356

noncomputable def f (x : ℝ) (a: ℝ) : ℝ := x^2 - a * Real.log x

-- (I) Proving the tangent line equation is y = x for a = 1 at x = 1.
theorem tangent_line_eq (h : ∀ x, f x 1 = x^2 - Real.log x) :
  ∃ y : (ℝ → ℝ), y = id ∧ y 1 = x :=
sorry

-- (II) Proving extreme values of the function f(x).
theorem extreme_values (a: ℝ) :
  (∃ x_min : ℝ, f x_min a = (a/2) - (a/2) * Real.log (a/2)) ∧ 
  (∀ x, ¬∃ x_max : ℝ, f x_max a > f x a) :=
sorry

-- (III) Proving the range of values for a.
theorem range_of_a :
  (∀ x, 2*x - (a/x) ≥ 0 → 2 < x) → a ≤ 8 :=
sorry

end tangent_line_eq_extreme_values_range_of_a_l16_16356


namespace overall_average_commission_rate_l16_16820

-- Define conditions for the commissions and transaction amounts
def C₁ := 0.25 / 100 * 100 + 0.25 / 100 * 105.25
def C₂ := 0.35 / 100 * 150 + 0.45 / 100 * 155.50
def C₃ := 0.30 / 100 * 80 + 0.40 / 100 * 83
def total_commission := C₁ + C₂ + C₃
def TA := 100 + 105.25 + 150 + 155.50 + 80 + 83

-- The proposition to prove
theorem overall_average_commission_rate : (total_commission / TA) * 100 = 0.3429 :=
  by
  sorry

end overall_average_commission_rate_l16_16820


namespace petya_can_write_divisible_by_2019_l16_16540

open Nat

theorem petya_can_write_divisible_by_2019 (M : ℕ) (h : ∃ k : ℕ, M = (10^k - 1) / 9) : ∃ N : ℕ, (N = (10^M - 1) / 9) ∧ 2019 ∣ N :=
by
  sorry

end petya_can_write_divisible_by_2019_l16_16540


namespace range_of_a_l16_16033

noncomputable def f (a : ℝ) (x : ℝ) := Real.sqrt (Real.exp x + (Real.exp 1 - 1) * x - a)
def exists_b_condition (a : ℝ) : Prop := ∃ b : ℝ, b ∈ Set.Icc 0 1 ∧ f a b = b

theorem range_of_a (a : ℝ) : exists_b_condition a → a ∈ Set.Icc 1 (2 * Real.exp 1 - 2) :=
sorry

end range_of_a_l16_16033


namespace min_k_valid_l16_16196

def S : Set ℕ := {1, 2, 3, 4}

def valid_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ b : Fin 4 → ℕ,
    (∀ i : Fin 4, b i ∈ S) ∧ b 3 ≠ 1 →
    ∃ i1 i2 i3 i4 : Fin (k + 1), i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧
      (a i1 = b 0 ∧ a i2 = b 1 ∧ a i3 = b 2 ∧ a i4 = b 3)

def min_k := 11

theorem min_k_valid : ∀ a : ℕ → ℕ,
  valid_sequence a min_k → 
  min_k = 11 :=
sorry

end min_k_valid_l16_16196


namespace segment_length_C_C_l16_16986

-- Define the points C and C''.
def C : ℝ × ℝ := (-3, 2)
def C'' : ℝ × ℝ := (-3, -2)

-- State the theorem that the length of the segment from C to C'' is 4.
theorem segment_length_C_C'' : dist C C'' = 4 := by
  sorry

end segment_length_C_C_l16_16986


namespace dosage_range_l16_16764

theorem dosage_range (d : ℝ) (h : 60 ≤ d ∧ d ≤ 120) : 15 ≤ (d / 4) ∧ (d / 4) ≤ 30 :=
by
  sorry

end dosage_range_l16_16764


namespace problem_statement_l16_16475

open Nat

theorem problem_statement (n a : ℕ) 
  (hn : n > 1) 
  (ha : a > n^2)
  (H : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ k, a + i = (n^2 + i) * k) :
  a > n^4 - n^3 := 
sorry

end problem_statement_l16_16475


namespace negation_proposition_l16_16078

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^3 + 5*x - 2 = 0) ↔ ∀ x : ℝ, x^3 + 5*x - 2 ≠ 0 :=
by sorry

end negation_proposition_l16_16078


namespace convenience_store_pure_milk_quantity_convenience_store_yogurt_discount_l16_16801

noncomputable def cost_per_pure_milk_box (x : ℕ) : ℝ := 2000 / x
noncomputable def cost_per_yogurt_box (x : ℕ) : ℝ := 4800 / (1.5 * x)

theorem convenience_store_pure_milk_quantity
  (x : ℕ)
  (hx : cost_per_yogurt_box x - cost_per_pure_milk_box x = 30) :
  x = 40 :=
by
  sorry

noncomputable def pure_milk_price := 80
noncomputable def yogurt_price (cost_per_yogurt_box : ℝ) : ℝ := cost_per_yogurt_box * 1.25

theorem convenience_store_yogurt_discount
  (x y : ℕ)
  (hx : cost_per_yogurt_box x - cost_per_pure_milk_box x = 30)
  (total_profit : ℕ)
  (profit_condition :
    pure_milk_price * x +
    yogurt_price (cost_per_yogurt_box x) * (1.5 * x - y) +
    yogurt_price (cost_per_yogurt_box x) * 0.9 * y - 2000 - 4800 = total_profit)
  (pure_milk_quantity : x = 40)
  (profit_value : total_profit = 2150) :
  y = 25 :=
by
  sorry

end convenience_store_pure_milk_quantity_convenience_store_yogurt_discount_l16_16801


namespace proof_G_eq_BC_eq_D_eq_AB_AC_l16_16413

-- Let's define the conditions of the problem first
variables (A B C O D G F E : Type) [Field A] [Field B] [Field C] [Field O] [Field D] [Field G] [Field F] [Field E]

-- Given triangle ABC with circumcenter O
variable {triangle_ABC: Prop}

-- Given point D on line segment BC
variable (D_on_BC : Prop)

-- Given circle Gamma with diameter OD
variable (circle_Gamma : Prop)

-- Given circles Gamma_1 and Gamma_2 are circumcircles of triangles ABD and ACD respectively
variable (circle_Gamma1 : Prop)
variable (circle_Gamma2 : Prop)

-- Given points F and E as intersection points
variable (intersect_F : Prop)
variable (intersect_E : Prop)

-- Given G as the second intersection point of the circumcircles of triangles BED and DFC
variable (second_intersect_G : Prop)

-- Prove that the condition for point G to be equidistant from points B and C is that point D is equidistant from lines AB and AC
theorem proof_G_eq_BC_eq_D_eq_AB_AC : 
  triangle_ABC ∧ D_on_BC ∧ circle_Gamma ∧ circle_Gamma1 ∧ circle_Gamma2 ∧ intersect_F ∧ intersect_E ∧ second_intersect_G → 
  G_dist_BC ↔ D_dist_AB_AC :=
by
  sorry

end proof_G_eq_BC_eq_D_eq_AB_AC_l16_16413


namespace discount_rate_l16_16722

theorem discount_rate (cost_shoes cost_socks cost_bag paid_price total_cost discount_amount amount_subject_to_discount discount_rate: ℝ)
  (h1 : cost_shoes = 74)
  (h2 : cost_socks = 2 * 2)
  (h3 : cost_bag = 42)
  (h4 : paid_price = 118)
  (h5 : total_cost = cost_shoes + cost_socks + cost_bag)
  (h6 : discount_amount = total_cost - paid_price)
  (h7 : amount_subject_to_discount = total_cost - 100)
  (h8 : discount_rate = (discount_amount / amount_subject_to_discount) * 100) :
  discount_rate = 10 := sorry

end discount_rate_l16_16722


namespace terry_total_miles_l16_16798

def total_gasoline_used := 9 + 17
def average_gas_mileage := 30

theorem terry_total_miles (M : ℕ) : 
  total_gasoline_used * average_gas_mileage = M → M = 780 :=
by
  intro h
  rw [←h]
  sorry

end terry_total_miles_l16_16798


namespace greatest_prime_factor_of_factorial_sum_l16_16422

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ p, Prime p ∧ p > 11 ∧ (∀ q, Prime q ∧ q > 11 → q ≤ 61) ∧ p = 61 :=
by
  sorry

end greatest_prime_factor_of_factorial_sum_l16_16422


namespace trigonometric_ratio_sum_l16_16406

open Real

theorem trigonometric_ratio_sum (x y : ℝ) 
  (h₁ : sin x / sin y = 2) 
  (h₂ : cos x / cos y = 1 / 3) :
  sin (2 * x) / sin (2 * y) + cos (2 * x) / cos (2 * y) = 41 / 57 := 
by
  sorry

end trigonometric_ratio_sum_l16_16406


namespace quadratic_inequality_solution_range_l16_16108

theorem quadratic_inequality_solution_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end quadratic_inequality_solution_range_l16_16108


namespace mul_three_point_six_and_zero_point_twenty_five_l16_16502

theorem mul_three_point_six_and_zero_point_twenty_five : 3.6 * 0.25 = 0.9 := by 
  sorry

end mul_three_point_six_and_zero_point_twenty_five_l16_16502


namespace saving_percentage_l16_16828

variable (S : ℝ) (saved_percent_last_year : ℝ) (made_more : ℝ) (saved_percent_this_year : ℝ)

-- Conditions from problem
def condition1 := saved_percent_last_year = 0.06
def condition2 := made_more = 1.20
def condition3 := saved_percent_this_year = 0.05 * made_more

-- The problem statement to prove
theorem saving_percentage (S : ℝ) (saved_percent_last_year : ℝ) (made_more : ℝ) (saved_percent_this_year : ℝ) :
  condition1 saved_percent_last_year →
  condition2 made_more →
  condition3 saved_percent_this_year made_more →
  (saved_percent_this_year * made_more = saved_percent_last_year * S * 1) :=
by 
  intros h1 h2 h3
  sorry

end saving_percentage_l16_16828


namespace remainder_196c_2008_mod_97_l16_16387

theorem remainder_196c_2008_mod_97 (c : ℤ) : ((196 * c) ^ 2008) % 97 = 44 := by
  sorry

end remainder_196c_2008_mod_97_l16_16387


namespace smallest_integer_divisibility_conditions_l16_16081

theorem smallest_integer_divisibility_conditions :
  ∃ n : ℕ, n > 0 ∧ (24 ∣ n^2) ∧ (900 ∣ n^3) ∧ (1024 ∣ n^4) ∧ n = 120 :=
by
  sorry

end smallest_integer_divisibility_conditions_l16_16081


namespace probability_vowel_probability_consonant_probability_ch_l16_16527

def word := "дифференцициал"
def total_letters := 12
def num_vowels := 5
def num_consonants := 7
def num_letter_ch := 0

theorem probability_vowel : (num_vowels : ℚ) / total_letters = 5 / 12 := by
  sorry

theorem probability_consonant : (num_consonants : ℚ) / total_letters = 7 / 12 := by
  sorry

theorem probability_ch : (num_letter_ch : ℚ) / total_letters = 0 := by
  sorry

end probability_vowel_probability_consonant_probability_ch_l16_16527


namespace conic_section_union_l16_16873

theorem conic_section_union : 
  ∀ (y x : ℝ), y^4 - 6*x^4 = 3*y^2 - 2 → 
  ( ( y^2 - 3*x^2 = 1 ∨ y^2 - 2*x^2 = 1 ) ∧ 
    ( y^2 - 2*x^2 = 2 ∨ y^2 - 3*x^2 = 2 ) ) :=
by
  sorry

end conic_section_union_l16_16873


namespace value_of_a_c_l16_16773

theorem value_of_a_c {a b c d : ℝ} :
  (∀ x y : ℝ, y = -|x - a| + b → (x = 1 ∧ y = 4) ∨ (x = 7 ∧ y = 2)) ∧
  (∀ x y : ℝ, y = |x - c| - d → (x = 1 ∧ y = 4) ∨ (x = 7 ∧ y = 2)) →
  a + c = 8 :=
by
  sorry

end value_of_a_c_l16_16773


namespace where_they_meet_l16_16447

/-- Define the conditions under which Petya and Vasya are walking. -/
structure WalkingCondition (n : ℕ) where
  lampposts : ℕ
  start_p : ℕ
  start_v : ℕ
  position_p : ℕ
  position_v : ℕ

/-- Initial conditions based on the problem statement. -/
def initialCondition : WalkingCondition 100 := {
  lampposts := 100,
  start_p := 1,
  start_v := 100,
  position_p := 22,
  position_v := 88
}

/-- Prove Petya and Vasya will meet at the 64th lamppost. -/
theorem where_they_meet (cond : WalkingCondition 100) : 64 ∈ { x | x = 64 } :=
  -- The formal proof would go here.
  sorry

end where_they_meet_l16_16447


namespace TV_height_l16_16584

theorem TV_height (area : ℝ) (width : ℝ) (height : ℝ) (h1 : area = 21) (h2 : width = 3) : height = 7 :=
  by
  sorry

end TV_height_l16_16584


namespace min_value_of_expression_l16_16037

theorem min_value_of_expression (a b : ℝ) (h1 : 1 < a) (h2 : 0 < b) (h3 : a + 2 * b = 2) : 
  4 * (1 + Real.sqrt 2) ≤ (2 / (a - 1) + a / b) :=
by
  sorry

end min_value_of_expression_l16_16037


namespace triple_square_side_area_l16_16634

theorem triple_square_side_area (s : ℝ) : (3 * s) ^ 2 ≠ 3 * (s ^ 2) :=
by {
  sorry
}

end triple_square_side_area_l16_16634


namespace isosceles_triangle_perimeter_l16_16696

def is_isosceles_triangle (A B C : ℝ) : Prop :=
  (A = B ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (A = C ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (B = C ∧ A + B > C ∧ A + C > B ∧ B + C > A)

theorem isosceles_triangle_perimeter {A B C : ℝ} 
  (h : is_isosceles_triangle A B C) 
  (h1 : A = 3 ∨ A = 7) 
  (h2 : B = 3 ∨ B = 7) 
  (h3 : C = 3 ∨ C = 7) : 
  A + B + C = 17 := 
sorry

end isosceles_triangle_perimeter_l16_16696


namespace shelly_total_money_l16_16296

-- Define the conditions
def num_of_ten_dollar_bills : ℕ := 10
def num_of_five_dollar_bills : ℕ := num_of_ten_dollar_bills - 4

-- Problem statement: How much money does Shelly have in all?
theorem shelly_total_money :
  (num_of_ten_dollar_bills * 10) + (num_of_five_dollar_bills * 5) = 130 :=
by
  sorry

end shelly_total_money_l16_16296


namespace surface_area_sphere_dihedral_l16_16251

open Real

theorem surface_area_sphere_dihedral (R a : ℝ) (hR : 0 < R) (haR : 0 < a ∧ a < R) (α : ℝ) :
  2 * R^2 * arccos ((R * cos α) / sqrt (R^2 - a^2 * sin α^2)) 
  - 2 * R * a * sin α * arccos ((a * cos α) / sqrt (R^2 - a^2 * sin α^2)) = sorry :=
sorry

end surface_area_sphere_dihedral_l16_16251


namespace jessica_initial_withdrawal_fraction_l16_16454

variable {B : ℝ} -- this is the initial balance

noncomputable def initial_withdrawal_fraction (B : ℝ) : Prop :=
  let remaining_balance := B - 400
  let deposit := (1 / 4) * remaining_balance
  let final_balance := remaining_balance + deposit
  final_balance = 750 → (400 / B) = 2 / 5

-- Our goal is to prove the statement given conditions.
theorem jessica_initial_withdrawal_fraction : 
  ∃ B : ℝ, initial_withdrawal_fraction B :=
sorry

end jessica_initial_withdrawal_fraction_l16_16454


namespace urn_contains_four_each_color_after_six_steps_l16_16777

noncomputable def probability_urn_four_each_color : ℚ := 2 / 7

def urn_problem (urn_initial : ℕ) (draws : ℕ) (final_urn : ℕ) (extra_balls : ℕ) : Prop :=
urn_initial = 2 ∧ draws = 6 ∧ final_urn = 8 ∧ extra_balls > 0

theorem urn_contains_four_each_color_after_six_steps :
  urn_problem 2 6 8 2 → probability_urn_four_each_color = 2 / 7 :=
by
  intro h
  cases h
  sorry

end urn_contains_four_each_color_after_six_steps_l16_16777


namespace fair_tickets_more_than_twice_baseball_tickets_l16_16213

theorem fair_tickets_more_than_twice_baseball_tickets :
  ∃ (fair_tickets baseball_tickets : ℕ), 
    fair_tickets = 25 ∧ baseball_tickets = 56 ∧ 
    fair_tickets + 87 = 2 * baseball_tickets := 
by
  sorry

end fair_tickets_more_than_twice_baseball_tickets_l16_16213


namespace charlie_golden_delicious_bags_l16_16482

theorem charlie_golden_delicious_bags :
  ∀ (total_bags fruit_bags macintosh_bags cortland_bags golden_delicious_bags : ℝ),
  total_bags = 0.67 →
  macintosh_bags = 0.17 →
  cortland_bags = 0.33 →
  total_bags = golden_delicious_bags + macintosh_bags + cortland_bags →
  golden_delicious_bags = 0.17 := by
  intros total_bags fruit_bags macintosh_bags cortland_bags golden_delicious_bags
  intros h_total h_macintosh h_cortland h_sum
  sorry

end charlie_golden_delicious_bags_l16_16482


namespace value_of_f_g_l16_16783

def f (x : ℝ) : ℝ := x^2 - 3*x + 7
def g (x : ℝ) : ℝ := x + 4

theorem value_of_f_g (h₁ : f (g 3) = 35) (h₂ : g (f 3) = 11) : f (g 3) - g (f 3) = 24 :=
by
  calc
    f (g 3) - g (f 3) = 35 - 11 := by rw [h₁, h₂]
                      _         = 24 := by norm_num

end value_of_f_g_l16_16783


namespace roots_of_quadratic_eq_l16_16630

theorem roots_of_quadratic_eq (h : ∀ x : ℝ, x^2 - 3 * x = 0 → x = 0 ∨ x = 3) :
  ∀ x : ℝ, x^2 - 3 * x = 0 → x = 0 ∨ x = 3 :=
by sorry

end roots_of_quadratic_eq_l16_16630


namespace not_sqrt2_rational_union_eq_intersection_implies_equal_intersection_eq_b_subset_a_element_in_both_implies_in_intersection_l16_16943

-- Definitions
def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ (x = a / b)
def union (A B : Set α) : Set α := {x | x ∈ A ∨ x ∈ B}
def intersection (A B : Set α) : Set α := {x | x ∈ A ∧ x ∈ B}
def subset (A B : Set α) : Prop := ∀ x, x ∈ A → x ∈ B

-- Statement A
theorem not_sqrt2_rational : ¬ is_rational (Real.sqrt 2) :=
sorry

-- Statement B
theorem union_eq_intersection_implies_equal {α : Type*} {A B : Set α}
  (h : union A B = intersection A B) : A = B :=
sorry

-- Statement C
theorem intersection_eq_b_subset_a {α : Type*} {A B : Set α}
  (h : intersection A B = B) : subset B A :=
sorry

-- Statement D
theorem element_in_both_implies_in_intersection {α : Type*} {A B : Set α} {a : α}
  (haA : a ∈ A) (haB : a ∈ B) : a ∈ intersection A B :=
sorry

end not_sqrt2_rational_union_eq_intersection_implies_equal_intersection_eq_b_subset_a_element_in_both_implies_in_intersection_l16_16943


namespace power_mod_l16_16830

theorem power_mod (n : ℕ) : 3^100 % 7 = 4 := by
  sorry

end power_mod_l16_16830


namespace inequality_solution_set_l16_16382

theorem inequality_solution_set :
  { x : ℝ | (3 * x + 1) / (x - 2) ≤ 0 } = { x : ℝ | -1/3 ≤ x ∧ x < 2 } :=
sorry

end inequality_solution_set_l16_16382


namespace solution_of_modified_system_l16_16934

theorem solution_of_modified_system
  (a b x y : ℝ)
  (h1 : 2*a*3 + 3*4 = 18)
  (h2 : -3 + 5*b*4 = 17)
  : (x + y = 7 ∧ x - y = -1) → (2*a*(x+y) + 3*(x-y) = 18 ∧ (x+y) - 5*b*(x-y) = -17) → (x = (7 / 2) ∧ y = (-1 / 2)) :=
by
sorry

end solution_of_modified_system_l16_16934


namespace exists_triplet_with_gcd_conditions_l16_16754

-- Given the conditions as definitions in Lean.
variables (S : Set ℕ)
variable [Infinite S] -- S is an infinite set of positive integers.
variables {a b c d x y z : ℕ}
variable (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S)
variable (hdistinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d) 
variable (hgcd_neq : gcd a b ≠ gcd c d)

-- The formal proof statement.
theorem exists_triplet_with_gcd_conditions :
  ∃ (x y z : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ gcd x y = gcd y z ∧ gcd y z ≠ gcd z x :=
sorry

end exists_triplet_with_gcd_conditions_l16_16754


namespace number_of_bonnies_l16_16950

theorem number_of_bonnies (B blueberries apples : ℝ) 
  (h1 : blueberries = 3 / 4 * B) 
  (h2 : apples = 3 * blueberries)
  (h3 : B + blueberries + apples = 240) : 
  B = 60 :=
by
  sorry

end number_of_bonnies_l16_16950


namespace simplify_expression_l16_16441

noncomputable def sqrt' (x : ℝ) : ℝ := Real.sqrt x

theorem simplify_expression :
  (3 * sqrt' 8 / (sqrt' 2 + sqrt' 3 + sqrt' 7)) = (sqrt' 2 + sqrt' 3 - sqrt' 7) := 
by
  sorry

end simplify_expression_l16_16441


namespace minyoung_division_l16_16297

theorem minyoung_division : 
  ∃ x : ℝ, 107.8 / x = 9.8 ∧ x = 11 :=
by
  use 11
  simp
  sorry

end minyoung_division_l16_16297


namespace range_of_2a_minus_b_l16_16567

variable (a b : ℝ)
variable (h1 : -2 < a ∧ a < 2)
variable (h2 : 2 < b ∧ b < 3)

theorem range_of_2a_minus_b (a b : ℝ) (h1 : -2 < a ∧ a < 2) (h2 : 2 < b ∧ b < 3) :
  -7 < 2 * a - b ∧ 2 * a - b < 2 := sorry

end range_of_2a_minus_b_l16_16567


namespace solve_for_P_l16_16471

theorem solve_for_P (P : Real) (h : (P ^ 4) ^ (1 / 3) = 9 * 81 ^ (1 / 9)) : P = 3 ^ (11 / 6) :=
by
  sorry

end solve_for_P_l16_16471


namespace solve_for_r_l16_16615

variable (n : ℝ) (r : ℝ)

theorem solve_for_r (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) : r = (n * (1 + Real.sqrt 3)) / 2 :=
by
  sorry

end solve_for_r_l16_16615


namespace divisibility_2_pow_a_plus_1_l16_16154

theorem divisibility_2_pow_a_plus_1 (a b : ℕ) (h_b_pos : 0 < b) (h_b_ge_2 : 2 ≤ b) 
  (h_div : (2^a + 1) % (2^b - 1) = 0) : b = 2 := by
  sorry

end divisibility_2_pow_a_plus_1_l16_16154


namespace area_triangle_QCA_l16_16602

noncomputable def area_of_triangle_QCA (p : ℝ) : ℝ :=
  let Q := (0, 12)
  let A := (3, 12)
  let C := (0, p)
  let QA := 3
  let QC := 12 - p
  (1/2) * QA * QC

theorem area_triangle_QCA (p : ℝ) : area_of_triangle_QCA p = (3/2) * (12 - p) :=
  sorry

end area_triangle_QCA_l16_16602


namespace T_10_mod_5_eq_3_l16_16515

def a_n (n : ℕ) : ℕ := -- Number of sequences of length n ending in A
sorry

def b_n (n : ℕ) : ℕ := -- Number of sequences of length n ending in B
sorry

def c_n (n : ℕ) : ℕ := -- Number of sequences of length n ending in C
sorry

def T (n : ℕ) : ℕ := -- Number of valid sequences of length n
  a_n n + b_n n

theorem T_10_mod_5_eq_3 :
  T 10 % 5 = 3 :=
sorry

end T_10_mod_5_eq_3_l16_16515


namespace total_birds_caught_l16_16347

theorem total_birds_caught 
  (day_birds : ℕ) 
  (night_birds : ℕ)
  (h1 : day_birds = 8) 
  (h2 : night_birds = 2 * day_birds) 
  : day_birds + night_birds = 24 := 
by 
  sorry

end total_birds_caught_l16_16347


namespace abs_expression_equals_l16_16427

theorem abs_expression_equals (h : Real.pi < 12) : 
  abs (Real.pi - abs (Real.pi - 12)) = 12 - 2 * Real.pi := 
by
  sorry

end abs_expression_equals_l16_16427


namespace initial_fee_is_correct_l16_16909

noncomputable def initial_fee (total_charge : ℝ) (charge_per_segment : ℝ) (segment_length : ℝ) (distance : ℝ) : ℝ :=
  total_charge - (⌊distance / segment_length⌋ * charge_per_segment)

theorem initial_fee_is_correct :
  initial_fee 4.5 0.25 (2/5) 3.6 = 2.25 :=
by 
  sorry

end initial_fee_is_correct_l16_16909


namespace number_of_triangles_2016_30_l16_16104

def f (m n : ℕ) : ℕ :=
  2 * m - n - 2

theorem number_of_triangles_2016_30 :
  f 2016 30 = 4000 := 
by
  sorry

end number_of_triangles_2016_30_l16_16104


namespace apples_to_pears_l16_16928

theorem apples_to_pears :
  (∀ (apples oranges pears : ℕ),
  12 * apples = 6 * oranges →
  3 * oranges = 5 * pears →
  24 * apples = 20 * pears) :=
by
  intros apples oranges pears h₁ h₂
  sorry

end apples_to_pears_l16_16928


namespace sum_infinite_series_l16_16893

theorem sum_infinite_series :
  ∑' n : ℕ, (3 * (n+1) + 2) / ((n+1) * (n+2) * (n+4)) = 29 / 36 :=
by
  sorry

end sum_infinite_series_l16_16893


namespace Sam_and_Tina_distance_l16_16989

theorem Sam_and_Tina_distance (marguerite_distance : ℕ) (marguerite_time : ℕ)
  (sam_time : ℕ) (tina_time : ℕ) (sam_distance : ℕ) (tina_distance : ℕ)
  (h1 : marguerite_distance = 150) (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) (h4 : tina_time = 2)
  (h5 : sam_distance = (marguerite_distance / marguerite_time) * sam_time)
  (h6 : tina_distance = (marguerite_distance / marguerite_time) * tina_time) :
  sam_distance = 200 ∧ tina_distance = 100 :=
by
  sorry

end Sam_and_Tina_distance_l16_16989


namespace given_trig_identity_l16_16751

variable {x : ℂ} {α : ℝ} {n : ℕ}

theorem given_trig_identity (h : x + 1/x = 2 * Real.cos α) : x^n + 1/x^n = 2 * Real.cos (n * α) :=
sorry

end given_trig_identity_l16_16751


namespace divides_difference_l16_16211

theorem divides_difference (n : ℕ) (h_composite : ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k) : 
  6 ∣ ((n^2)^3 - n^2) := 
sorry

end divides_difference_l16_16211


namespace impossible_to_place_numbers_l16_16775

noncomputable def divisible (a b : ℕ) : Prop := ∃ k : ℕ, a * k = b

def connected (G : Finset (ℕ × ℕ)) (u v : ℕ) : Prop := (u, v) ∈ G ∨ (v, u) ∈ G

def valid_assignment (G : Finset (ℕ × ℕ)) (f : ℕ → ℕ) : Prop :=
  ∀ ⦃i j⦄, connected G i j → divisible (f i) (f j) ∨ divisible (f j) (f i)

def invalid_assignment (G : Finset (ℕ × ℕ)) (f : ℕ → ℕ) : Prop :=
  ∀ ⦃i j⦄, ¬ connected G i j → ¬ divisible (f i) (f j) ∧ ¬ divisible (f j) (f i)

theorem impossible_to_place_numbers (G : Finset (ℕ × ℕ)) :
  (∃ f : ℕ → ℕ, valid_assignment G f ∧ invalid_assignment G f) → False :=
by
  sorry

end impossible_to_place_numbers_l16_16775


namespace parabola_latus_rectum_equation_l16_16030

theorem parabola_latus_rectum_equation :
  (∃ (y x : ℝ), y^2 = 4 * x) → (∀ x, x = -1) :=
by
  sorry

end parabola_latus_rectum_equation_l16_16030


namespace range_of_m_l16_16568

noncomputable def f (x m : ℝ) := (1/2) * x^2 + m * x + Real.log x

noncomputable def f_prime (x m : ℝ) := x + 1/x + m

theorem range_of_m (x0 m : ℝ) 
  (h1 : (1/2) ≤ x0 ∧ x0 ≤ 3) 
  (unique_x0 : ∀ y, f_prime y m = 0 → y = x0) 
  (cond1 : f_prime (1/2) m < 0) 
  (cond2 : f_prime 3 m ≥ 0) 
  : -10 / 3 ≤ m ∧ m < -5 / 2 :=
sorry

end range_of_m_l16_16568


namespace area_of_triangle_is_27_over_5_l16_16883

def area_of_triangle_bounded_by_y_axis_and_lines : ℚ :=
  let y_intercept_1 := -2
  let y_intercept_2 := 4
  let base := y_intercept_2 - y_intercept_1
  let x_intersection : ℚ := 9 / 5   -- Calculated using the system of equations
  1 / 2 * base * x_intersection

theorem area_of_triangle_is_27_over_5 :
  area_of_triangle_bounded_by_y_axis_and_lines = 27 / 5 := by
  sorry

end area_of_triangle_is_27_over_5_l16_16883


namespace inequality_problem_l16_16270

theorem inequality_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 / (b * (a + b)) + 2 / (c * (b + c)) + 2 / (a * (c + a))) ≥ (27 / (a + b + c)^2) :=
by
  sorry

end inequality_problem_l16_16270


namespace simplification_and_evaluation_l16_16371

theorem simplification_and_evaluation (a : ℚ) (h : a = -1 / 2) :
  (3 * a + 2) * (a - 1) - 4 * a * (a + 1) = 1 / 4 := 
by
  sorry

end simplification_and_evaluation_l16_16371


namespace abc_not_8_l16_16683

theorem abc_not_8 (a b c : ℕ) (h : 2^a * 3^b * 4^c = 192) : a + b + c ≠ 8 :=
sorry

end abc_not_8_l16_16683


namespace total_cost_l16_16980

-- Definitions:
def amount_beef : ℕ := 1000
def price_per_pound_beef : ℕ := 8
def amount_chicken := amount_beef * 2
def price_per_pound_chicken : ℕ := 3

-- Theorem: The total cost of beef and chicken is $14000.
theorem total_cost : (amount_beef * price_per_pound_beef) + (amount_chicken * price_per_pound_chicken) = 14000 :=
by
  sorry

end total_cost_l16_16980


namespace walking_time_l16_16771

theorem walking_time 
  (speed_km_hr : ℝ := 10) 
  (distance_km : ℝ := 6) 
  : (distance_km / (speed_km_hr / 60)) = 36 :=
by
  sorry

end walking_time_l16_16771


namespace tire_circumference_l16_16614

theorem tire_circumference 
  (rev_per_min : ℝ) -- revolutions per minute
  (car_speed_kmh : ℝ) -- car speed in km/h
  (conversion_factor : ℝ) -- conversion factor for speed from km/h to m/min
  (min_to_meter : ℝ) -- multiplier to convert minutes to meters
  (C : ℝ) -- circumference of the tire in meters
  : rev_per_min = 400 ∧ car_speed_kmh = 120 ∧ conversion_factor = 1000 / 60 ∧ min_to_meter = 1000 / 60 ∧ (C * rev_per_min = car_speed_kmh * min_to_meter) → C = 5 :=
by
  sorry

end tire_circumference_l16_16614


namespace incorrect_statement_implies_m_eq_zero_l16_16649

theorem incorrect_statement_implies_m_eq_zero
  (m : ℝ)
  (y : ℝ → ℝ)
  (h : ∀ x, y x = m * x + 4 * m - 2)
  (intersects_y_axis_at : y 0 = -2) :
  m = 0 :=
sorry

end incorrect_statement_implies_m_eq_zero_l16_16649


namespace mona_game_group_size_l16_16741

theorem mona_game_group_size 
  (x : ℕ)
  (h_conditions: 9 * (x - 1) - 3 = 33) : x = 5 := 
by 
  sorry

end mona_game_group_size_l16_16741


namespace qualified_flour_l16_16858

def is_qualified_flour (weight : ℝ) : Prop :=
  weight ≥ 24.75 ∧ weight ≤ 25.25

theorem qualified_flour :
  is_qualified_flour 24.80 ∧
  ¬is_qualified_flour 24.70 ∧
  ¬is_qualified_flour 25.30 ∧
  ¬is_qualified_flour 25.51 :=
by
  sorry

end qualified_flour_l16_16858


namespace james_hears_beats_per_week_l16_16394

theorem james_hears_beats_per_week
  (beats_per_minute : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (H1 : beats_per_minute = 200)
  (H2 : hours_per_day = 2)
  (H3 : days_per_week = 7) :
  beats_per_minute * hours_per_day * 60 * days_per_week = 168000 := 
by
  -- sorry proof step placeholder
  sorry

end james_hears_beats_per_week_l16_16394


namespace compare_exponents_product_of_roots_l16_16358

noncomputable def f (x : ℝ) (a : ℝ) := (Real.log x) / (x + a)

theorem compare_exponents : (2016 : ℝ) ^ 2017 > (2017 : ℝ) ^ 2016 :=
sorry

theorem product_of_roots (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f x1 0 = k) (h3 : f x2 0 = k) : 
  x1 * x2 > Real.exp 2 :=
sorry

end compare_exponents_product_of_roots_l16_16358


namespace base_of_parallelogram_l16_16639

theorem base_of_parallelogram (area height base : ℝ) 
  (h_area : area = 320)
  (h_height : height = 16) :
  base = area / height :=
by 
  rw [h_area, h_height]
  norm_num
  sorry

end base_of_parallelogram_l16_16639


namespace range_of_a_l16_16690

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- Lean statement for the problem
theorem range_of_a (a : ℝ) (h : A ∪ B a = B a) : -1 < a ∧ a ≤ 1 := 
by
  -- Proof is skipped
  sorry

end range_of_a_l16_16690


namespace total_crayons_l16_16982

noncomputable def original_crayons : ℝ := 479.0
noncomputable def additional_crayons : ℝ := 134.0

theorem total_crayons : original_crayons + additional_crayons = 613.0 := by
  sorry

end total_crayons_l16_16982


namespace largest_mersenne_prime_less_than_500_l16_16337

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Nat.Prime n ∧ p = 2^n - 1 ∧ Nat.Prime p

theorem largest_mersenne_prime_less_than_500 :
  ∃ p : ℕ, is_mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, is_mersenne_prime q ∧ q < 500 → q ≤ p → p = 127 :=
by
  sorry

end largest_mersenne_prime_less_than_500_l16_16337


namespace total_amount_silver_l16_16143

theorem total_amount_silver (x y : ℝ) (h₁ : y = 7 * x + 4) (h₂ : y = 9 * x - 8) : y = 46 :=
by {
  sorry
}

end total_amount_silver_l16_16143


namespace arithmetic_series_sum_l16_16856

theorem arithmetic_series_sum :
  let a := 2
  let d := 3
  let l := 50
  let n := (l - a) / d + 1
  let S := n * (2 * a + (n - 1) * d) / 2
  S = 442 := by
  sorry

end arithmetic_series_sum_l16_16856


namespace equilateral_triangle_ratio_correct_l16_16646

noncomputable def equilateral_triangle_ratio : ℝ :=
  let side_length := 12
  let perimeter := 3 * side_length
  let altitude := side_length * (Real.sqrt 3 / 2)
  let area := (side_length * altitude) / 2
  area / perimeter

theorem equilateral_triangle_ratio_correct :
  equilateral_triangle_ratio = Real.sqrt 3 :=
sorry

end equilateral_triangle_ratio_correct_l16_16646


namespace arctan_tan_equiv_l16_16197

theorem arctan_tan_equiv (h1 : Real.tan (Real.pi / 4 + Real.pi / 12) = 1 / Real.tan (Real.pi / 4 - Real.pi / 3))
  (h2 : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3):
  Real.arctan (Real.tan (5 * Real.pi / 12) - 2 * Real.tan (Real.pi / 6)) = 5 * Real.pi / 12 := 
sorry

end arctan_tan_equiv_l16_16197


namespace reginald_apples_sold_l16_16885

theorem reginald_apples_sold 
  (apple_price : ℝ) 
  (bike_cost : ℝ)
  (repair_percentage : ℝ)
  (remaining_fraction : ℝ)
  (discount_apples : ℕ)
  (free_apples : ℕ)
  (total_apples_sold : ℕ) : 
  apple_price = 1.25 → 
  bike_cost = 80 → 
  repair_percentage = 0.25 → 
  remaining_fraction = 0.2 → 
  discount_apples = 5 → 
  free_apples = 1 → 
  (∃ (E : ℝ), (125 = E ∧ total_apples_sold = 120)) → 
  total_apples_sold = 120 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end reginald_apples_sold_l16_16885


namespace FastFoodCost_l16_16530

theorem FastFoodCost :
  let sandwich_cost := 4
  let soda_cost := 1.5
  let fries_cost := 2.5
  let num_sandwiches := 4
  let num_sodas := 6
  let num_fries := 3
  let discount := 5
  let total_cost := (sandwich_cost * num_sandwiches) + (soda_cost * num_sodas) + (fries_cost * num_fries) - discount
  total_cost = 27.5 := 
by
  sorry

end FastFoodCost_l16_16530


namespace calc_g_g_neg3_l16_16179

def g (x : ℚ) : ℚ :=
x⁻¹ + x⁻¹ / (2 + x⁻¹)

theorem calc_g_g_neg3 : g (g (-3)) = -135 / 8 := 
by
  sorry

end calc_g_g_neg3_l16_16179


namespace large_square_pattern_l16_16436

theorem large_square_pattern :
  999999^2 = 1000000 * 999998 + 1 :=
by sorry

end large_square_pattern_l16_16436


namespace find_number_l16_16170

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 90) : x = 4000 :=
by
  sorry

end find_number_l16_16170


namespace tan_half_alpha_l16_16925

theorem tan_half_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 24 / 25) : Real.tan (α / 2) = 3 / 4 :=
by
  sorry

end tan_half_alpha_l16_16925


namespace large_bucket_capacity_l16_16299

variables (S L : ℝ)

theorem large_bucket_capacity (h1 : L = 2 * S + 3) (h2 : 2 * S + 5 * L = 63) : L = 11 :=
by sorry

end large_bucket_capacity_l16_16299


namespace octagon_area_l16_16721

noncomputable def area_of_octagon_concentric_squares : ℚ :=
  let m := 1
  let n := 8
  (m + n)

theorem octagon_area (O : ℝ × ℝ) (side_small side_large : ℚ) (AB : ℚ) 
  (h1 : side_small = 2) (h2 : side_large = 3) (h3 : AB = 1/4) : 
  area_of_octagon_concentric_squares = 9 := 
  by
  have h_area : 1/8 = 1/8 := rfl
  sorry

end octagon_area_l16_16721


namespace part1_part2_l16_16095

section PartOne

variables (x y : ℕ)
def condition1 := x + y = 360
def condition2 := x - y = 110

theorem part1 (h1 : condition1 x y) (h2 : condition2 x y) : x = 235 ∧ y = 125 := by {
  sorry
}

end PartOne

section PartTwo

variables (t W : ℕ)
def tents_capacity (t : ℕ) := 40 * t + 20 * (9 - t)
def food_capacity (t : ℕ) := 10 * t + 20 * (9 - t)
def transportation_cost (t : ℕ) := 4000 * t + 3600 * (9 - t)

theorem part2 
  (htents : tents_capacity t ≥ 235) 
  (hfood : food_capacity t ≥ 125) : 
  W = transportation_cost t → t = 3 ∧ W = 33600 := by {
  sorry
}

end PartTwo

end part1_part2_l16_16095


namespace monthly_salary_l16_16770

variables (S : ℕ) (h1 : S * 20 / 100 * 96 / 100 = 4 * 250)

theorem monthly_salary : S = 6250 :=
by sorry

end monthly_salary_l16_16770


namespace sum_of_distinct_roots_l16_16603

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l16_16603


namespace determine_p_in_terms_of_q_l16_16539

variable {p q : ℝ}

-- Given the condition in the problem
def log_condition (p q : ℝ) : Prop :=
  Real.log p + 2 * Real.log q = Real.log (2 * p + q)

-- The goal is to prove that under this condition, the following holds
theorem determine_p_in_terms_of_q (h : log_condition p q) :
  p = q / (q^2 - 2) :=
sorry

end determine_p_in_terms_of_q_l16_16539


namespace base_six_four_digit_odd_final_l16_16670

theorem base_six_four_digit_odd_final :
  ∃ b : ℕ, (b^4 > 285 ∧ 285 ≥ b^3 ∧ (285 % b) % 2 = 1) :=
by 
  use 6
  sorry

end base_six_four_digit_odd_final_l16_16670


namespace evaluate_expression_l16_16804

theorem evaluate_expression (x : ℕ) (h : x = 3) : (x^x)^(x^x) = 27^27 :=
by
  sorry

end evaluate_expression_l16_16804


namespace color_of_face_opposite_blue_l16_16936

/-- Assume we have a cube with each face painted in distinct colors. -/
structure Cube where
  top : String
  front : String
  right_side : String
  back : String
  left_side : String
  bottom : String

/-- Given three views of a colored cube, determine the color of the face opposite the blue face. -/
theorem color_of_face_opposite_blue (c : Cube)
  (h_top : c.top = "R")
  (h_right : c.right_side = "G")
  (h_view1 : c.front = "W")
  (h_view2 : c.front = "O")
  (h_view3 : c.front = "Y") :
  c.back = "Y" :=
sorry

end color_of_face_opposite_blue_l16_16936


namespace domain_of_composite_function_l16_16412

theorem domain_of_composite_function
    (f : ℝ → ℝ)
    (h : ∀ x, -2 ≤ x ∧ x ≤ 3 → f (x + 1) ∈ (Set.Icc (-2:ℝ) (3:ℝ))):
    ∃ s : Set ℝ, s = Set.Icc 0 (5/2) ∧ (∀ x, x ∈ s ↔ f (2 * x - 1) ∈ Set.Icc (-1) 4) :=
by
  sorry

end domain_of_composite_function_l16_16412


namespace one_third_of_five_times_seven_l16_16012

theorem one_third_of_five_times_seven:
  (1/3 : ℝ) * (5 * 7) = 35 / 3 := 
by
  -- Definitions and calculations go here
  sorry

end one_third_of_five_times_seven_l16_16012


namespace find_third_root_l16_16218

noncomputable def P (a b x : ℝ) : ℝ := a * x^3 + (a + 4 * b) * x^2 + (b - 5 * a) * x + (10 - a)

theorem find_third_root (a b : ℝ) (h1 : P a b (-1) = 0) (h2 : P a b 4 = 0) : 
 ∃ c : ℝ, c ≠ -1 ∧ c ≠ 4 ∧ P a b c = 0 ∧ c = 8 / 3 :=
 sorry

end find_third_root_l16_16218


namespace calculate_material_needed_l16_16428

theorem calculate_material_needed (area : ℝ) (pi_approx : ℝ) (extra_material : ℝ) (r : ℝ) (C : ℝ) : 
  area = 50.24 → pi_approx = 3.14 → extra_material = 4 → pi_approx * r ^ 2 = area → 
  C = 2 * pi_approx * r →
  C + extra_material = 29.12 :=
by
  intros h_area h_pi h_extra h_area_eq h_C_eq
  sorry

end calculate_material_needed_l16_16428


namespace yellow_block_heavier_than_green_l16_16342

theorem yellow_block_heavier_than_green :
  let yellow_block_weight := 0.6
  let green_block_weight := 0.4
  yellow_block_weight - green_block_weight = 0.2 := by
  let yellow_block_weight := 0.6
  let green_block_weight := 0.4
  show yellow_block_weight - green_block_weight = 0.2
  sorry

end yellow_block_heavier_than_green_l16_16342


namespace sum_of_positive_x_and_y_is_ten_l16_16284

theorem sum_of_positive_x_and_y_is_ten (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x^3 + y^3 + (x + y)^3 + 30 * x * y = 2000) : 
  x + y = 10 :=
sorry

end sum_of_positive_x_and_y_is_ten_l16_16284


namespace f_expression_f_odd_l16_16223

noncomputable def f (x : ℝ) (a b : ℝ) := (2^x + b) / (2^x + a)

theorem f_expression :
  ∃ a b, f 1 a b = 1 / 3 ∧ f 0 a b = 0 ∧ (∀ x, f x a b = (2^x - 1) / (2^x + 1)) :=
by
  sorry

theorem f_odd :
  ∀ x, f x 1 (-1) = (2^x - 1) / (2^x + 1) ∧ f (-x) 1 (-1) = -f x 1 (-1) :=
by
  sorry

end f_expression_f_odd_l16_16223


namespace bus_capacities_rental_plan_l16_16877

variable (x y : ℕ)
variable (m n : ℕ)

theorem bus_capacities :
  3 * x + 2 * y = 195 ∧ 2 * x + 4 * y = 210 → x = 45 ∧ y = 30 :=
by
  sorry

theorem rental_plan :
  7 * m + 3 * n = 20 ∧ m + n ≤ 7 ∧ 65 * m + 45 * n + 30 * (7 - m - n) = 310 →
  m = 2 ∧ n = 2 ∧ 7 - m - n = 3 :=
by
  sorry

end bus_capacities_rental_plan_l16_16877


namespace smallest_three_digit_multiple_of_17_l16_16737

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l16_16737


namespace solve_color_problem_l16_16160

variables (R B G C : Prop)

def color_problem (R B G C : Prop) : Prop :=
  (C → (R ∨ B)) ∧ (¬C → (¬R ∧ ¬G)) ∧ ((B ∨ G) → C) → C ∧ (R ∨ B)

theorem solve_color_problem (R B G C : Prop) (h : (C → (R ∨ B)) ∧ (¬C → (¬R ∧ ¬G)) ∧ ((B ∨ G) → C)) : C ∧ (R ∨ B) :=
  by {
    sorry
  }

end solve_color_problem_l16_16160


namespace find_C_coordinates_l16_16537

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 11, y := 9 }
def B : Point := { x := 2, y := -3 }
def D : Point := { x := -1, y := 3 }

-- Define the isosceles property
def is_isosceles (A B C : Point) : Prop :=
  Real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2) = Real.sqrt ((A.x - C.x) ^ 2 + (A.y - C.y) ^ 2)

-- Define the midpoint property
def is_midpoint (D B C : Point) : Prop :=
  D.x = (B.x + C.x) / 2 ∧ D.y = (B.y + C.y) / 2

theorem find_C_coordinates (C : Point)
  (h_iso : is_isosceles A B C)
  (h_mid : is_midpoint D B C) :
  C = { x := -4, y := 9 } := 
  sorry

end find_C_coordinates_l16_16537


namespace minimum_a_for_cube_in_tetrahedron_l16_16473

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  (Real.sqrt 6 / 12) * a

theorem minimum_a_for_cube_in_tetrahedron (a : ℝ) (r : ℝ) 
  (h_radius : r = radius_of_circumscribed_sphere a)
  (h_diag : Real.sqrt 3 = 2 * r) :
  a = 3 * Real.sqrt 2 :=
by
  sorry

end minimum_a_for_cube_in_tetrahedron_l16_16473


namespace solve_phi_l16_16664

theorem solve_phi (n : ℕ) : 
  (∃ (x y z : ℕ), 5 * x + 2 * y + z = 10 * n) → 
  (∃ (φ : ℕ), φ = 5 * n^2 + 4 * n + 1) :=
by 
  sorry

end solve_phi_l16_16664


namespace perfect_square_trinomial_l16_16080

theorem perfect_square_trinomial :
  120^2 - 40 * 120 + 20^2 = 10000 := sorry

end perfect_square_trinomial_l16_16080


namespace width_of_foil_covered_prism_l16_16723

theorem width_of_foil_covered_prism (L W H : ℝ) 
  (h1 : W = 2 * L)
  (h2 : W = 2 * H)
  (h3 : L * W * H = 128)
  (h4 : L = H) :
  W + 2 = 8 :=
sorry

end width_of_foil_covered_prism_l16_16723


namespace proof_a_minus_b_l16_16307

def S (a : ℕ) : Set ℕ := {1, 2, a}
def T (b : ℕ) : Set ℕ := {2, 3, 4, b}

theorem proof_a_minus_b (a b : ℕ)
  (hS : S a = {1, 2, a})
  (hT : T b = {2, 3, 4, b})
  (h_intersection : S a ∩ T b = {1, 2, 3}) :
  a - b = 2 := by
  sorry

end proof_a_minus_b_l16_16307


namespace greatest_sum_consecutive_integers_lt_500_l16_16289

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l16_16289


namespace waiter_net_earning_l16_16309

theorem waiter_net_earning (c1 c2 c3 m : ℤ) (h1 : c1 = 3) (h2 : c2 = 2) (h3 : c3 = 1) (t1 t2 t3 : ℤ) (h4 : t1 = 8) (h5 : t2 = 10) (h6 : t3 = 12) (hmeal : m = 5):
  c1 * t1 + c2 * t2 + c3 * t3 - m = 51 := 
by 
  sorry

end waiter_net_earning_l16_16309


namespace sum_of_coordinates_of_other_endpoint_of_segment_l16_16974

theorem sum_of_coordinates_of_other_endpoint_of_segment {x y : ℝ}
  (h1 : (6 + x) / 2 = 3)
  (h2 : (1 + y) / 2 = 7) :
  x + y = 13 := by
  sorry

end sum_of_coordinates_of_other_endpoint_of_segment_l16_16974


namespace solve_equation_l16_16009

theorem solve_equation (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 :=
by
  sorry

end solve_equation_l16_16009


namespace paco_ate_more_salty_than_sweet_l16_16204

-- Define the initial conditions
def sweet_start := 8
def salty_start := 6
def sweet_ate := 20
def salty_ate := 34

-- Define the statement to prove
theorem paco_ate_more_salty_than_sweet : (salty_ate - sweet_ate) = 14 := by
    sorry

end paco_ate_more_salty_than_sweet_l16_16204


namespace percent_time_in_meetings_l16_16671

theorem percent_time_in_meetings
  (work_day_minutes : ℕ := 8 * 60)
  (first_meeting_minutes : ℕ := 30)
  (second_meeting_minutes : ℕ := 3 * 30) :
  (first_meeting_minutes + second_meeting_minutes) / work_day_minutes * 100 = 25 :=
by
  -- sorry to skip the actual proof
  sorry

end percent_time_in_meetings_l16_16671


namespace first_term_is_5_over_2_l16_16331

-- Define the arithmetic sequence and the sum of the first n terms.
def arith_seq (a d : ℕ) (n : ℕ) := a + (n - 1) * d
def S (a d : ℕ) (n : ℕ) := (n * (2 * a + (n - 1) * d)) / 2

-- Define the constant ratio condition.
def const_ratio (a d : ℕ) (n : ℕ) (c : ℕ) :=
  (S a d (3 * n) * 2) = c * (S a d n * 2)

-- Prove the first term is 5/2 given the conditions.
theorem first_term_is_5_over_2 (c : ℕ) (n : ℕ) (h : const_ratio a 5 n 9) : 
  a = 5 / 2 :=
sorry

end first_term_is_5_over_2_l16_16331


namespace LindseyMinimumSavings_l16_16827
-- Import the library to bring in the necessary definitions and notations

-- Definitions from the problem conditions
def SeptemberSavings : ℕ := 50
def OctoberSavings : ℕ := 37
def NovemberSavings : ℕ := 11
def MomContribution : ℕ := 25
def VideoGameCost : ℕ := 87
def RemainingMoney : ℕ := 36

-- Problem statement as a Lean theorem
theorem LindseyMinimumSavings : 
  (SeptemberSavings + OctoberSavings + NovemberSavings) > 98 :=
  sorry

end LindseyMinimumSavings_l16_16827


namespace no_positive_integer_solutions_l16_16048

theorem no_positive_integer_solutions (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : x^4 * y^4 - 14 * x^2 * y^2 + 49 ≠ 0 := 
by sorry

end no_positive_integer_solutions_l16_16048


namespace sum_of_three_distinct_integers_product_625_l16_16689

theorem sum_of_three_distinct_integers_product_625 :
  ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 131 :=
by
  sorry

end sum_of_three_distinct_integers_product_625_l16_16689


namespace yuna_correct_multiplication_l16_16330

theorem yuna_correct_multiplication (x : ℕ) (h : 4 * x = 60) : 8 * x = 120 :=
by
  sorry

end yuna_correct_multiplication_l16_16330


namespace range_of_alpha_l16_16552

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 5 * x

theorem range_of_alpha (α : ℝ) (h₀ : -1 < α) (h₁ : α < 1) (h₂ : f (1 - α) + f (1 - α^2) < 0) : 1 < α ∧ α < Real.sqrt 2 := by
  sorry

end range_of_alpha_l16_16552


namespace arithmetic_sequence_properties_l16_16890

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d a_1, ∀ n, a n = a_1 + d * (n - 1)

def sum_n (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def sum_b (b : ℕ → ℕ) (T : ℕ → ℕ) : Prop :=
  ∀ n, T n = n^2 + n + (3^(n+1) - 3)/2

theorem arithmetic_sequence_properties :
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
    (arithmetic_seq a) →
    a 5 = 10 →
    S 7 = 56 →
    (∀ n, a n = 2 * n) ∧
    ∃ (b T : ℕ → ℕ), (∀ n, b n = a n + 3^n) ∧ sum_b b T :=
by
  intros a S ha h5 hS7
  sorry

end arithmetic_sequence_properties_l16_16890


namespace roots_of_quadratic_l16_16580

theorem roots_of_quadratic (b c : ℝ) (h1 : 1 + -2 = -b) (h2 : 1 * -2 = c) : b = 1 ∧ c = -2 :=
by
  sorry

end roots_of_quadratic_l16_16580


namespace find_k_l16_16225

theorem find_k (x y k : ℝ) (h1 : 2 * x + y = 4 * k) (h2 : x - y = k) (h3 : x + 2 * y = 12) : k = 4 :=
sorry

end find_k_l16_16225


namespace find_s_l16_16898

noncomputable def area_of_parallelogram (s : ℝ) : ℝ :=
  (3 * s) * (s * Real.sin (Real.pi / 3))

theorem find_s (s : ℝ) (h1 : area_of_parallelogram s = 27 * Real.sqrt 3) : s = 3 * Real.sqrt 2 := 
  sorry

end find_s_l16_16898


namespace M_minus_N_positive_l16_16405

variable (a b : ℝ)

def M : ℝ := 10 * a^2 + b^2 - 7 * a + 8
def N : ℝ := a^2 + b^2 + 5 * a + 1

theorem M_minus_N_positive : M a b - N a b ≥ 3 := by
  sorry

end M_minus_N_positive_l16_16405


namespace number_of_diet_soda_l16_16695

variable (d r : ℕ)

-- Define the conditions of the problem
def condition1 : Prop := r = d + 79
def condition2 : Prop := r = 83

-- State the theorem we want to prove
theorem number_of_diet_soda (h1 : condition1 d r) (h2 : condition2 r) : d = 4 :=
by
  sorry

end number_of_diet_soda_l16_16695


namespace sphere_volume_from_area_l16_16034

/-- Given the surface area of a sphere is 24π, prove that the volume of the sphere is 8√6π. -/ 
theorem sphere_volume_from_area :
  ∀ {R : ℝ},
    4 * Real.pi * R^2 = 24 * Real.pi →
    (4 / 3) * Real.pi * R^3 = 8 * Real.sqrt 6 * Real.pi :=
by
  intro R h
  sorry

end sphere_volume_from_area_l16_16034


namespace sam_initial_nickels_l16_16658

variable (n_now n_given n_initial : Nat)

theorem sam_initial_nickels (h_now : n_now = 63) (h_given : n_given = 39) (h_relation : n_now = n_initial + n_given) : n_initial = 24 :=
by
  sorry

end sam_initial_nickels_l16_16658


namespace sum_of_interior_angles_l16_16888

theorem sum_of_interior_angles (n : ℕ) 
  (h : 180 * (n - 2) = 3600) :
  180 * (n + 2 - 2) = 3960 ∧ 180 * (n - 2 - 2) = 3240 :=
by
  sorry

end sum_of_interior_angles_l16_16888


namespace opposite_of_fraction_l16_16750

def opposite_of (x : ℚ) : ℚ := -x

theorem opposite_of_fraction :
  opposite_of (1/2023) = - (1/2023) :=
by
  sorry

end opposite_of_fraction_l16_16750


namespace sequence_arithmetic_l16_16185

theorem sequence_arithmetic (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = 2 * n^2 - 2 * n) →
  (∀ n, a n = S n - S (n - 1)) →
  (∀ n, a n - a (n - 1) = 4) :=
by
  intros hS ha
  sorry

end sequence_arithmetic_l16_16185


namespace three_digit_numbers_eq_11_sum_squares_l16_16247

theorem three_digit_numbers_eq_11_sum_squares :
  ∃ (N : ℕ), 
    (N = 550 ∨ N = 803) ∧
    (∃ (a b c : ℕ), 
      N = 100 * a + 10 * b + c ∧ 
      100 * a + 10 * b + c = 11 * (a ^ 2 + b ^ 2 + c ^ 2) ∧
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧
      0 ≤ c ∧ c ≤ 9) :=
sorry

end three_digit_numbers_eq_11_sum_squares_l16_16247


namespace convert_to_dms_convert_to_decimal_degrees_l16_16692

-- Problem 1: Conversion of 24.29 degrees to degrees, minutes, and seconds 
theorem convert_to_dms (d : ℝ) (h : d = 24.29) : 
  (∃ deg min sec, d = deg + min / 60 + sec / 3600 ∧ deg = 24 ∧ min = 17 ∧ sec = 24) :=
by
  sorry

-- Problem 2: Conversion of 36 degrees 40 minutes 30 seconds to decimal degrees
theorem convert_to_decimal_degrees (deg min sec : ℝ) (h : deg = 36 ∧ min = 40 ∧ sec = 30) : 
  (deg + min / 60 + sec / 3600) = 36.66 :=
by
  sorry

end convert_to_dms_convert_to_decimal_degrees_l16_16692


namespace number_of_red_cars_l16_16910

theorem number_of_red_cars (B R : ℕ) (h1 : R / B = 3 / 8) (h2 : B = 70) : R = 26 :=
by
  sorry

end number_of_red_cars_l16_16910


namespace area_of_fifteen_sided_figure_l16_16291

noncomputable def figure_area : ℝ :=
  let full_squares : ℝ := 6
  let num_triangles : ℝ := 10
  let triangles_to_rectangles : ℝ := num_triangles / 2
  let triangles_area : ℝ := triangles_to_rectangles
  full_squares + triangles_area

theorem area_of_fifteen_sided_figure :
  figure_area = 11 := by
  sorry

end area_of_fifteen_sided_figure_l16_16291


namespace _l16_16701

noncomputable def angle_ACB_is_45_degrees (A B C D E F : Type) [LinearOrderedField A]
  (angle : A → A → A → A) (AB AC : A) (h1 : AB = 3 * AC)
  (BAE ACD : A) (h2 : BAE = ACD)
  (BCA : A) (h3 : BAE = 2 * BCA)
  (CF FE : A) (h4 : CF = FE)
  (is_isosceles : ∀ {X Y Z : Type} [LinearOrderedField X] (a b c : X), a = b → b = c → a = c)
  (triangle_sum : ∀ {X Y Z : Type} [LinearOrderedField X] (a b c : X), a + b + c = 180) :
  ∃ (angle_ACB : A), angle_ACB = 45 := 
by
  -- Here we assume we have the appropriate conditions from geometry
  -- Then you'd prove the theorem based on given hypotheses
  sorry

end _l16_16701


namespace tutors_meet_in_lab_l16_16985

theorem tutors_meet_in_lab (c a j t : ℕ)
  (hC : c = 5) (hA : a = 6) (hJ : j = 8) (hT : t = 9) :
  Nat.lcm (Nat.lcm (Nat.lcm c a) j) t = 360 :=
by
  rw [hC, hA, hJ, hT]
  rfl

end tutors_meet_in_lab_l16_16985


namespace y_intercept_of_line_l16_16790

theorem y_intercept_of_line : 
  ∀ (x y : ℝ), 3 * x - 5 * y = 7 → y = -7 / 5 :=
by
  intro x y h
  sorry

end y_intercept_of_line_l16_16790


namespace largest_non_representable_integer_l16_16641

theorem largest_non_representable_integer (n a b : ℕ) (h₁ : n = 42 * a + b)
  (h₂ : 0 ≤ b) (h₃ : b < 42) (h₄ : ¬ (b % 6 = 0)) :
  n ≤ 252 :=
sorry

end largest_non_representable_integer_l16_16641


namespace line_passes_through_point_l16_16224

theorem line_passes_through_point (k : ℝ) :
  (1 + 4 * k) * 2 - (2 - 3 * k) * 2 + 2 - 14 * k = 0 :=
by
  sorry

end line_passes_through_point_l16_16224


namespace cody_steps_away_from_goal_l16_16265

def steps_in_week (daily_steps : ℕ) : ℕ :=
  daily_steps * 7

def total_steps_in_4_weeks (initial_steps : ℕ) : ℕ :=
  steps_in_week initial_steps +
  steps_in_week (initial_steps + 1000) +
  steps_in_week (initial_steps + 2000) +
  steps_in_week (initial_steps + 3000)

theorem cody_steps_away_from_goal :
  let goal := 100000
  let initial_daily_steps := 1000
  let total_steps := total_steps_in_4_weeks initial_daily_steps
  goal - total_steps = 30000 :=
by
  sorry

end cody_steps_away_from_goal_l16_16265


namespace area_of_right_triangle_with_hypotenuse_and_angle_l16_16814

theorem area_of_right_triangle_with_hypotenuse_and_angle 
  (hypotenuse : ℝ) (angle : ℝ) (h_hypotenuse : hypotenuse = 9 * Real.sqrt 3) (h_angle : angle = 30) : 
  ∃ (area : ℝ), area = 364.5 := 
by
  sorry

end area_of_right_triangle_with_hypotenuse_and_angle_l16_16814


namespace domain_of_function_correct_l16_16019

noncomputable def domain_of_function (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (2 - x > 0) ∧ (Real.logb 10 (2 - x) ≠ 0)

theorem domain_of_function_correct :
  {x : ℝ | domain_of_function x} = {x : ℝ | x ∈ Set.Icc (-1 : ℝ) 1 \ {1}} ∪ {x : ℝ | x ∈ Set.Ioc 1 2} :=
by
  sorry

end domain_of_function_correct_l16_16019


namespace minimum_value_l16_16545

theorem minimum_value(a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1/2) :
  (2 / a + 3 / b) ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_l16_16545


namespace parabola_focus_standard_equation_l16_16302

theorem parabola_focus_standard_equation :
  ∃ (a b : ℝ), (a = 16 ∧ b = 0) ∨ (a = 0 ∧ b = -8) →
  (∃ (F : ℝ × ℝ), F = (4, 0) ∨ F = (0, -2) ∧ F ∈ {p : ℝ × ℝ | (p.1 - 2 * p.2 - 4 = 0)} →
  (∃ (x y : ℝ), (y^2 = a * x) ∨ (x^2 = b * y))) := sorry

end parabola_focus_standard_equation_l16_16302


namespace range_of_a_l16_16740

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x - 1 ≤ 0) → a ≤ -1 :=
sorry

end range_of_a_l16_16740


namespace weight_of_mixture_is_112_5_l16_16097

noncomputable def weight_of_mixture (W : ℝ) : Prop :=
  (5 / 14) * W + (3 / 10) * W + (2 / 9) * W + (1 / 7) * W + 2.5 = W

theorem weight_of_mixture_is_112_5 : ∃ W : ℝ, weight_of_mixture W ∧ W = 112.5 :=
by {
  use 112.5,
  sorry
}

end weight_of_mixture_is_112_5_l16_16097


namespace smallest_product_of_set_l16_16867

noncomputable def smallest_product_set : Set ℤ := { -10, -3, 0, 4, 6 }

theorem smallest_product_of_set :
  ∃ (a b : ℤ), a ∈ smallest_product_set ∧ b ∈ smallest_product_set ∧ a ≠ b ∧ a * b = -60 ∧
  ∀ (x y : ℤ), x ∈ smallest_product_set ∧ y ∈ smallest_product_set ∧ x ≠ y → x * y ≥ -60 := 
sorry

end smallest_product_of_set_l16_16867


namespace parabola_ratio_l16_16452

noncomputable def AF_over_BF (p : ℝ) (h_p : p > 0) : ℝ :=
  let AF := 4 * p
  let x := (4 / 7) * p -- derived from solving the equation in the solution
  AF / x

theorem parabola_ratio (p : ℝ) (h_p : p > 0) : AF_over_BF p h_p = 7 :=
  sorry

end parabola_ratio_l16_16452


namespace eval_expr_l16_16119

theorem eval_expr (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a > b) :
  (a^(b+1) * b^(a+1)) / (b^(b+1) * a^(a+1)) = (a / b)^(b - a) :=
sorry

end eval_expr_l16_16119


namespace minimum_value_analysis_l16_16474

theorem minimum_value_analysis
  (a : ℝ) (m n : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : 2 * m + n = 2)
  (h4 : m > 0)
  (h5 : n > 0) :
  (2 / m + 1 / n) ≥ 9 / 2 :=
sorry

end minimum_value_analysis_l16_16474


namespace pieces_on_third_day_impossibility_of_2014_pieces_l16_16066

-- Define the process of dividing and eating chocolate pieces.
def chocolate_pieces (n : ℕ) : ℕ :=
  9 + 8 * n

-- The number of pieces after the third day.
theorem pieces_on_third_day : chocolate_pieces 3 = 25 :=
sorry

-- It's impossible for Maria to have exactly 2014 pieces on any given day.
theorem impossibility_of_2014_pieces : ∀ n : ℕ, chocolate_pieces n ≠ 2014 :=
sorry

end pieces_on_third_day_impossibility_of_2014_pieces_l16_16066


namespace polygon_sides_l16_16895

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1440) : n = 10 := sorry

end polygon_sides_l16_16895


namespace S_2011_l16_16157

variable {α : Type*}

-- Define initial term and sum function for arithmetic sequence
def a1 : ℤ := -2011
noncomputable def S (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * 2

-- Given conditions
def condition1 : a1 = -2011 := rfl
def condition2 : (S 2010 / 2010) - (S 2008 / 2008) = 2 := by sorry

-- Proof statement
theorem S_2011 : S 2011 = -2011 := by 
  -- Use the given conditions to prove the statement
  sorry

end S_2011_l16_16157


namespace predict_sales_amount_l16_16114

theorem predict_sales_amount :
  let x_data := [2, 4, 5, 6, 8]
  let y_data := [30, 40, 50, 60, 70]
  let b := 7
  let x := 10 -- corresponding to 10,000 yuan investment
  let a := 15 -- \hat{a} calculated from the regression equation and data points
  let regression (x : ℝ) := b * x + a
  regression x = 85 :=
by
  -- Proof skipped
  sorry

end predict_sales_amount_l16_16114


namespace remainder_83_pow_89_times_5_mod_11_l16_16453

theorem remainder_83_pow_89_times_5_mod_11 : 
  (83^89 * 5) % 11 = 10 := 
by
  have h1 : 83 % 11 = 6 := by sorry
  have h2 : 6^10 % 11 = 1 := by sorry
  have h3 : 89 = 8 * 10 + 9 := by sorry
  sorry

end remainder_83_pow_89_times_5_mod_11_l16_16453


namespace possible_values_of_m_l16_16960

theorem possible_values_of_m (a b : ℤ) (h1 : a * b = -14) :
  ∃ m : ℤ, m = a + b ∧ (m = 5 ∨ m = -5 ∨ m = 13 ∨ m = -13) :=
by
  sorry

end possible_values_of_m_l16_16960


namespace laptop_price_l16_16153

theorem laptop_price (x : ℝ) : 
  (0.8 * x - 120) = 0.9 * x - 64 → x = 560 :=
by
  sorry

end laptop_price_l16_16153


namespace problem_statement_l16_16230

noncomputable def ellipse_equation (t : ℝ) (ht : t > 0) : String :=
  if h : t = 2 then "x^2/9 + y^2/2 = 1"
  else "invalid equation"

theorem problem_statement (m : ℝ) (t : ℝ) (ht : t > 0) (ha : t = 2) 
  (A E F B : ℝ × ℝ) (hA : A = (-3, 0)) (hB : B = (1, 0))
  (hl : ∀ x y, x = m * y + 1) (area : ℝ) (har : area = 16/3) :
  ((ellipse_equation t ht) = "x^2/9 + y^2/2 = 1") ∧
  (∃ M N : ℝ × ℝ, 
    (M.1 = 3 ∧ N.1 = 3) ∧
    ((M.1 - B.1) * (N.1 - B.1) + (M.2 - B.2) * (N.2 - B.2) = 0)) := 
sorry

end problem_statement_l16_16230


namespace total_rainfall_2004_l16_16322

def average_rainfall_2003 := 50 -- in mm
def extra_rainfall_2004 := 3 -- in mm
def average_rainfall_2004 := average_rainfall_2003 + extra_rainfall_2004 -- in mm
def days_february_2004 := 29
def days_other_months := 30
def months := 12
def months_without_february := months - 1

theorem total_rainfall_2004 : 
  (average_rainfall_2004 * days_february_2004) + (months_without_february * average_rainfall_2004 * days_other_months) = 19027 := 
by sorry

end total_rainfall_2004_l16_16322


namespace find_prime_c_l16_16279

-- Define the statement of the problem
theorem find_prime_c (c : ℕ) (hc : Nat.Prime c) (h : ∃ m : ℕ, (m > 0) ∧ (11 * c + 1 = m^2)) : c = 13 :=
by
  sorry

end find_prime_c_l16_16279


namespace point_movement_l16_16874

theorem point_movement (P : ℤ) (hP : P = -5) (k : ℤ) (hk : (k = 3 ∨ k = -3)) :
  P + k = -8 ∨ P + k = -2 :=
by {
  sorry
}

end point_movement_l16_16874


namespace sufficient_but_not_necessary_not_necessary_l16_16616

-- Conditions
def condition_1 (x : ℝ) : Prop := x > 3
def condition_2 (x : ℝ) : Prop := x^2 - 5 * x + 6 > 0

-- Theorem statement
theorem sufficient_but_not_necessary (x : ℝ) : condition_1 x → condition_2 x :=
sorry

theorem not_necessary (x : ℝ) : condition_2 x → ∃ y : ℝ, ¬ condition_1 y ∧ condition_2 y :=
sorry

end sufficient_but_not_necessary_not_necessary_l16_16616


namespace ways_to_sum_2022_l16_16816

theorem ways_to_sum_2022 : 
  ∃ n : ℕ, (∀ a b : ℕ, (2022 = 2 * a + 3 * b) ∧ n = (b - a) / 4 ∧ n = 338) := 
sorry

end ways_to_sum_2022_l16_16816


namespace problem_220_l16_16240

variables (x y : ℝ)

theorem problem_220 (h1 : x + y = 10) (h2 : (x * y) / (x^2) = -3 / 2) :
  x = -20 ∧ y = 30 :=
by
  sorry

end problem_220_l16_16240


namespace tea_set_costs_l16_16235
noncomputable section

-- Definition for the conditions of part 1
def cost_condition1 (x y : ℝ) : Prop := x + 2 * y = 250
def cost_condition2 (x y : ℝ) : Prop := 3 * x + 4 * y = 600

-- Definition for the conditions of part 2
def cost_condition3 (a : ℝ) : ℝ := 108 * a + 60 * (80 - a)

-- Definition for the conditions of part 3
def profit (a b : ℝ) : ℝ := 30 * a + 20 * b

theorem tea_set_costs (x y : ℝ) (a : ℕ) :
  cost_condition1 x y →
  cost_condition2 x y →
  x = 100 ∧ y = 75 ∧ a ≤ 30 ∧ profit 30 50 = 1900 := by
  sorry

end tea_set_costs_l16_16235


namespace total_missing_keys_l16_16002

theorem total_missing_keys :
  let total_vowels := 5
  let total_consonants := 21
  let missing_consonants := total_consonants / 7
  let missing_vowels := 2
  missing_consonants + missing_vowels = 5 :=
by {
  sorry
}

end total_missing_keys_l16_16002


namespace tens_digit_N_to_20_l16_16011

theorem tens_digit_N_to_20 (N : ℕ) (h1 : Even N) (h2 : ¬(∃ k : ℕ, N = 10 * k)) : 
  ((N ^ 20) / 10) % 10 = 7 := 
by 
  sorry

end tens_digit_N_to_20_l16_16011


namespace circle_k_range_l16_16039

def circle_equation (k x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

theorem circle_k_range (k : ℝ) (h : ∃ x y, circle_equation k x y) : k > 4 ∨ k < -1 :=
by
  sorry

end circle_k_range_l16_16039


namespace maximal_intersection_area_of_rectangles_l16_16785

theorem maximal_intersection_area_of_rectangles :
  ∀ (a b : ℕ), a * b = 2015 ∧ a < b →
  ∀ (c d : ℕ), c * d = 2016 ∧ c > d →
  ∃ (max_area : ℕ), max_area = 1302 ∧ ∀ intersection_area, intersection_area ≤ 1302 := 
by
  sorry

end maximal_intersection_area_of_rectangles_l16_16785


namespace conference_fraction_married_men_l16_16142

theorem conference_fraction_married_men 
  (total_women : ℕ) 
  (single_probability : ℚ) 
  (h_single_prob : single_probability = 3/7) 
  (h_total_women : total_women = 7) : 
  (4 : ℚ) / (11 : ℚ) = 4 / 11 := 
by
  sorry

end conference_fraction_married_men_l16_16142


namespace cost_price_decrease_proof_l16_16931

theorem cost_price_decrease_proof (x y : ℝ) (a : ℝ) (h1 : y - x = x * a / 100)
    (h2 : y = (1 + a / 100) * x)
    (h3 : y - 0.9 * x = (0.9 * x * a / 100) + 0.9 * x * 20 / 100) : a = 80 :=
  sorry

end cost_price_decrease_proof_l16_16931


namespace smallest_b_greater_than_1_l16_16368

def g (x : ℕ) : ℕ :=
  if x % 35 = 0 then x / 35
  else if x % 7 = 0 then 5 * x
  else if x % 5 = 0 then 7 * x
  else x + 5

def g_iter (n : ℕ) (x : ℕ) : ℕ := Nat.iterate g n x

theorem smallest_b_greater_than_1 (b : ℕ) :
  (b > 1) → 
  g_iter 1 3 = 8 ∧ g_iter b 3 = 8 →
  b = 21 := by
  sorry

end smallest_b_greater_than_1_l16_16368


namespace factor_expression_l16_16323

-- Define the variables
variables (x : ℝ)

-- State the theorem to prove
theorem factor_expression : 3 * x * (x + 1) + 7 * (x + 1) = (3 * x + 7) * (x + 1) :=
by
  sorry

end factor_expression_l16_16323


namespace soccer_balls_are_20_l16_16086

variable (S : ℕ)
variable (num_baseballs : ℕ) (num_volleyballs : ℕ)
variable (condition_baseballs : num_baseballs = 5 * S)
variable (condition_volleyballs : num_volleyballs = 3 * S)
variable (condition_total : num_baseballs + num_volleyballs = 160)

theorem soccer_balls_are_20 :
  S = 20 :=
by
  sorry

end soccer_balls_are_20_l16_16086


namespace number_of_eighth_graders_l16_16992

theorem number_of_eighth_graders (x y : ℕ) :
  (x > 0) ∧ (y > 0) ∧ (8 + x * y = (x * (x + 3) - 14) / 2) →
  x = 7 ∨ x = 14 :=
by
  sorry

end number_of_eighth_graders_l16_16992


namespace find_m_l16_16294

-- Let m be a real number such that m > 1 and
-- \sum_{n=1}^{\infty} \frac{3n+2}{m^n} = 2.
theorem find_m (m : ℝ) (h1 : m > 1) 
(h2 : ∑' n : ℕ, (3 * (n + 1) + 2) / m^(n + 1) = 2) : 
  m = 3 :=
sorry

end find_m_l16_16294


namespace total_value_l16_16269

/-- 
The total value of the item V can be determined based on the given conditions.
- The merchant paid an import tax of $109.90.
- The tax rate is 7%.
- The tax is only on the portion of the value above $1000.

Given these conditions, prove that the total value V is 2567.
-/
theorem total_value {V : ℝ} (h1 : 0.07 * (V - 1000) = 109.90) : V = 2567 :=
by
  sorry

end total_value_l16_16269


namespace factorize_expression_l16_16882

theorem factorize_expression (x : ℝ) : x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 :=
by sorry

end factorize_expression_l16_16882


namespace problem_statement_l16_16996

variables (u v w : ℝ)

theorem problem_statement (h₁: u + v + w = 3) : 
  (1 / (u^2 + 7) + 1 / (v^2 + 7) + 1 / (w^2 + 7) ≤ 3 / 8) :=
sorry

end problem_statement_l16_16996


namespace cross_section_area_correct_l16_16380

noncomputable def area_of_cross_section (a : ℝ) : ℝ :=
  (3 * a^2 * Real.sqrt 11) / 16

theorem cross_section_area_correct (a : ℝ) (h : 0 < a) :
  area_of_cross_section a = (3 * a^2 * Real.sqrt 11) / 16 := by
  sorry

end cross_section_area_correct_l16_16380


namespace arc_length_of_sector_l16_16071

noncomputable def central_angle := 36
noncomputable def radius := 15

theorem arc_length_of_sector : (central_angle * Real.pi * radius / 180 = 3 * Real.pi) :=
by
  sorry

end arc_length_of_sector_l16_16071


namespace votes_cast_l16_16338

theorem votes_cast (A F T : ℕ) (h1 : A = 40 * T / 100) (h2 : F = A + 58) (h3 : T = F + A) : 
  T = 290 := 
by
  sorry

end votes_cast_l16_16338


namespace sum_of_coefficients_eq_3125_l16_16736

theorem sum_of_coefficients_eq_3125 
  {b_5 b_4 b_3 b_2 b_1 b_0 : ℤ}
  (h : (2 * x + 3)^5 = b_5 * x^5 + b_4 * x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0) :
  b_5 + b_4 + b_3 + b_2 + b_1 + b_0 = 3125 := 
by 
  sorry

end sum_of_coefficients_eq_3125_l16_16736


namespace solve_for_m_l16_16608

-- Define the conditions for the lines being parallel
def condition_one (m : ℝ) : Prop :=
  ∃ x y : ℝ, x + m * y + 3 = 0

def condition_two (m : ℝ) : Prop :=
  ∃ x y : ℝ, (m - 1) * x + 2 * m * y + 2 * m = 0

def are_parallel (A B C D : ℝ) : Prop :=
  A * D = B * C

theorem solve_for_m :
  ∀ (m : ℝ),
    (condition_one m) → 
    (condition_two m) → 
    (are_parallel 1 m 3 (2 * m)) →
    (m = 0) :=
by
  intro m h1 h2 h_parallel
  sorry

end solve_for_m_l16_16608


namespace solve_cryptarithm_l16_16070

-- Definitions for digits mapped to letters
def C : ℕ := 9
def H : ℕ := 3
def U : ℕ := 5
def K : ℕ := 4
def T : ℕ := 1
def R : ℕ := 2
def I : ℕ := 0
def G : ℕ := 6
def N : ℕ := 8
def S : ℕ := 7

-- Function to evaluate the cryptarithm sum
def cryptarithm_sum : ℕ :=
  (C*10000 + H*1000 + U*100 + C*10 + K) +
  (T*10000 + R*1000 + I*100 + G*10 + G) +
  (T*10000 + U*1000 + R*100 + N*10 + S)

-- Equation checking the result
def cryptarithm_correct : Prop :=
  cryptarithm_sum = T*100000 + R*10000 + I*1000 + C*100 + K*10 + S

-- The theorem we want to prove
theorem solve_cryptarithm : cryptarithm_correct :=
by
  -- Proof steps would be filled here
  -- but for now, we just acknowledge it is a theorem
  sorry

end solve_cryptarithm_l16_16070


namespace slant_height_l16_16488

-- Define the variables and conditions
variables (r A : ℝ)
-- Assume the given conditions
def radius := r = 5
def area := A = 60 * Real.pi

-- Statement of the theorem to prove the slant height
theorem slant_height (r A l : ℝ) (h_r : r = 5) (h_A : A = 60 * Real.pi) : l = 12 :=
sorry

end slant_height_l16_16488


namespace cashier_correction_l16_16812

theorem cashier_correction (y : ℕ) :
  let quarter_value := 25
  let nickel_value := 5
  let penny_value := 1
  let dime_value := 10
  let quarters_as_nickels_value := y * (quarter_value - nickel_value)
  let pennies_as_dimes_value := y * (dime_value - penny_value)
  let total_correction := quarters_as_nickels_value - pennies_as_dimes_value
  total_correction = 11 * y := by
  sorry

end cashier_correction_l16_16812


namespace correct_equation_among_options_l16_16496

theorem correct_equation_among_options
  (a : ℝ) (x : ℝ) :
  (-- Option A
  ¬ ((-1)^3 = -3)) ∧
  (-- Option B
  ¬ (((-2)^2 * (-2)^3) = (-2)^6)) ∧
  (-- Option C
  ¬ ((2 * a - a) = 2)) ∧
  (-- Option D
  ((x - 2)^2 = x^2 - 4*x + 4)) :=
by
  sorry

end correct_equation_among_options_l16_16496


namespace cost_of_socks_l16_16643

theorem cost_of_socks (x : ℝ) : 
  let initial_amount := 20
  let hat_cost := 7 
  let final_amount := 5
  let socks_pairs := 4
  let remaining_amount := initial_amount - hat_cost
  remaining_amount - socks_pairs * x = final_amount 
  -> x = 2 := 
by 
  sorry

end cost_of_socks_l16_16643


namespace point_equidistant_x_axis_y_axis_line_l16_16969

theorem point_equidistant_x_axis_y_axis_line (x y : ℝ) (h1 : abs y = abs x) (h2 : abs (x + y - 2) / Real.sqrt 2 = abs x) :
  x = 1 :=
  sorry

end point_equidistant_x_axis_y_axis_line_l16_16969


namespace max_min_value_of_f_l16_16444

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem max_min_value_of_f :
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ f (Real.pi / 6)) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f (Real.pi / 2) ≤ f x) :=
by
  sorry

end max_min_value_of_f_l16_16444


namespace commercials_played_l16_16523

theorem commercials_played (M C : ℝ) (h1 : M / C = 9 / 5) (h2 : M + C = 112) : C = 40 :=
by
  sorry

end commercials_played_l16_16523


namespace triangle_area_correct_l16_16298

-- Define the vectors a, b, and c as given in the problem
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (6, 2)
def c : ℝ × ℝ := (1, -1)

-- Define the function to calculate the area of the triangle with the given vertices
def triangle_area (u v w : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v.1 - u.1) * (w.2 - u.2) - (w.1 - u.1) * (v.2 - u.2))

-- State the proof problem
theorem triangle_area_correct : triangle_area c (a.1 + c.1, a.2 + c.2) (b.1 + c.1, b.2 + c.2) = 8.5 :=
by
  -- Proof can go here
  sorry

end triangle_area_correct_l16_16298


namespace factor_expression_l16_16563

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end factor_expression_l16_16563


namespace rhombus_perimeter_52_l16_16528

-- Define the conditions of the rhombus
def isRhombus (a b c d : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = d

def rhombus_diagonals (p q : ℝ) : Prop :=
  p = 10 ∧ q = 24

-- Define the perimeter calculation
def rhombus_perimeter (s : ℝ) : ℝ :=
  4 * s

-- Main theorem statement
theorem rhombus_perimeter_52 (p q s : ℝ)
  (h_diagonals : rhombus_diagonals p q)
  (h_rhombus : isRhombus s s s s)
  (h_side_length : s = 13) :
  rhombus_perimeter s = 52 :=
by
  sorry

end rhombus_perimeter_52_l16_16528


namespace fraction_evaluation_l16_16911

theorem fraction_evaluation : (1 / (2 + 1 / (3 + 1 / 4))) = (13 / 30) := by
  sorry

end fraction_evaluation_l16_16911


namespace impossible_coins_l16_16195

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end impossible_coins_l16_16195


namespace divisibility_by_37_l16_16953

theorem divisibility_by_37 (a b c : ℕ) :
  (100 * a + 10 * b + c) % 37 = 0 → 
  (100 * b + 10 * c + a) % 37 = 0 ∧
  (100 * c + 10 * a + b) % 37 = 0 :=
by
  sorry

end divisibility_by_37_l16_16953


namespace balance_scale_equation_l16_16442

theorem balance_scale_equation 
  (G Y B W : ℝ)
  (h1 : 4 * G = 8 * B)
  (h2 : 3 * Y = 6 * B)
  (h3 : 2 * B = 3 * W) : 
  3 * G + 4 * Y + 3 * W = 16 * B :=
by
  sorry

end balance_scale_equation_l16_16442


namespace factor_polynomial_l16_16107

theorem factor_polynomial (x y : ℝ) : 
  (x^2 - 2*x*y + y^2 - 16) = (x - y + 4) * (x - y - 4) :=
sorry

end factor_polynomial_l16_16107


namespace smallest_positive_integer_form_3003_55555_l16_16032

theorem smallest_positive_integer_form_3003_55555 :
  ∃ (m n : ℤ), 3003 * m + 55555 * n = 57 :=
by {
  sorry
}

end smallest_positive_integer_form_3003_55555_l16_16032


namespace total_distance_run_l16_16092

-- Given conditions
def number_of_students : Nat := 18
def distance_per_student : Nat := 106

-- Prove that the total distance run by the students equals 1908 meters.
theorem total_distance_run : number_of_students * distance_per_student = 1908 := by
  sorry

end total_distance_run_l16_16092


namespace cleanup_drive_weight_per_mile_per_hour_l16_16470

theorem cleanup_drive_weight_per_mile_per_hour :
  let duration := 4
  let lizzie_group := 387
  let second_group := lizzie_group - 39
  let third_group := 560 / 16
  let total_distance := 8
  let total_garbage := lizzie_group + second_group + third_group
  total_garbage / total_distance / duration = 24.0625 := 
by {
  sorry
}

end cleanup_drive_weight_per_mile_per_hour_l16_16470


namespace geometric_progression_ratio_l16_16578

theorem geometric_progression_ratio (q : ℝ) (h : |q| < 1 ∧ ∀a : ℝ, a = 4 * (a * q / (1 - q) - a * q)) :
  q = 1 / 5 :=
by
  sorry

end geometric_progression_ratio_l16_16578


namespace find_digits_l16_16667

theorem find_digits (a b c d : ℕ) 
  (h₀ : 0 ≤ a ∧ a ≤ 9)
  (h₁ : 0 ≤ b ∧ b ≤ 9)
  (h₂ : 0 ≤ c ∧ c ≤ 9)
  (h₃ : 0 ≤ d ∧ d ≤ 9)
  (h₄ : (10 * a + c) / 99 + (1000 * a + 100 * b + 10 * c + d) / 9999 = 17 / 37) :
  1000 * a + 100 * b + 10 * c + d = 2315 :=
by
  sorry

end find_digits_l16_16667


namespace find_number_l16_16319

-- Defining the constants provided and the related condition
def eight_percent_of (x: ℝ) : ℝ := 0.08 * x
def ten_percent_of_40 : ℝ := 0.10 * 40
def is_solution (x: ℝ) : Prop := (eight_percent_of x) + ten_percent_of_40 = 5.92

-- Theorem statement
theorem find_number : ∃ x : ℝ, is_solution x ∧ x = 24 :=
by sorry

end find_number_l16_16319


namespace bags_of_cookies_l16_16852

theorem bags_of_cookies (bags : ℕ) (cookies_total candies_total : ℕ) 
    (h1 : bags = 14) (h2 : cookies_total = 28) (h3 : candies_total = 86) :
    bags = 14 :=
by
  exact h1

end bags_of_cookies_l16_16852


namespace no_factors_multiple_of_210_l16_16363

theorem no_factors_multiple_of_210 (n : ℕ) (h : n = 2^12 * 3^18 * 5^10) : ∀ d : ℕ, d ∣ n → ¬ (210 ∣ d) :=
by
  sorry

end no_factors_multiple_of_210_l16_16363


namespace one_and_two_thirds_eq_36_l16_16513

theorem one_and_two_thirds_eq_36 (x : ℝ) (h : (5 / 3) * x = 36) : x = 21.6 :=
sorry

end one_and_two_thirds_eq_36_l16_16513


namespace total_distance_is_75_l16_16079

def distance1 : ℕ := 30
def distance2 : ℕ := 20
def distance3 : ℕ := 25

def total_distance : ℕ := distance1 + distance2 + distance3

theorem total_distance_is_75 : total_distance = 75 := by
  sorry

end total_distance_is_75_l16_16079


namespace factorization_of_expression_l16_16687

theorem factorization_of_expression (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) := by
  sorry

end factorization_of_expression_l16_16687


namespace price_difference_VA_NC_l16_16745

/-- Define the initial conditions -/
def NC_price : ℝ := 2
def NC_gallons : ℕ := 10
def VA_gallons : ℕ := 10
def total_spent : ℝ := 50

/-- Define the problem to prove the difference in price per gallon between Virginia and North Carolina -/
theorem price_difference_VA_NC (NC_price VA_price total_spent : ℝ) (NC_gallons VA_gallons : ℕ) :
  total_spent = NC_price * NC_gallons + VA_price * VA_gallons →
  VA_price - NC_price = 1 := 
by
  sorry -- Proof to be filled in

end price_difference_VA_NC_l16_16745


namespace ratio_of_daily_wages_l16_16255

-- Definitions for daily wages and conditions
def daily_wage_man : ℝ := sorry
def daily_wage_woman : ℝ := sorry

axiom condition_for_men (M : ℝ) : 16 * M * 25 = 14400
axiom condition_for_women (W : ℝ) : 40 * W * 30 = 21600

-- Theorem statement for the ratio of daily wages
theorem ratio_of_daily_wages 
  (M : ℝ) (W : ℝ) 
  (hM : 16 * M * 25 = 14400) 
  (hW : 40 * W * 30 = 21600) :
  M / W = 2 := 
  sorry

end ratio_of_daily_wages_l16_16255


namespace num_common_points_l16_16913

-- Definitions of the given conditions:
def line1 (x y : ℝ) := x + 2 * y - 3 = 0
def line2 (x y : ℝ) := 4 * x - y + 1 = 0
def line3 (x y : ℝ) := 2 * x - y - 5 = 0
def line4 (x y : ℝ) := 3 * x + 4 * y - 8 = 0

-- The proof goal:
theorem num_common_points : 
  ∃! p : ℝ × ℝ, (line1 p.1 p.2 ∨ line2 p.1 p.2) ∧ (line3 p.1 p.2 ∨ line4 p.1 p.2) :=
sorry

end num_common_points_l16_16913


namespace total_books_l16_16055

def numberOfMysteryShelves := 6
def numberOfPictureShelves := 2
def booksPerShelf := 9

theorem total_books (hMystery : numberOfMysteryShelves = 6) 
                    (hPicture : numberOfPictureShelves = 2) 
                    (hBooksPerShelf : booksPerShelf = 9) :
  numberOfMysteryShelves * booksPerShelf + numberOfPictureShelves * booksPerShelf = 72 :=
  by 
  sorry

end total_books_l16_16055


namespace problem_l16_16472

variable (p q : Prop)

theorem problem (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
by
  sorry

end problem_l16_16472


namespace volume_of_fifth_section_l16_16480

theorem volume_of_fifth_section (a : ℕ → ℚ) (d : ℚ) :
  (a 1 + a 2 + a 3 + a 4) = 3 ∧ (a 9 + a 8 + a 7) = 4 ∧
  (∀ n, a n = a 1 + (n - 1) * d) →
  a 5 = 67 / 66 :=
by
  sorry

end volume_of_fifth_section_l16_16480


namespace reciprocal_expression_l16_16949

theorem reciprocal_expression :
  (1 / ((1 / 4 : ℚ) + (1 / 5 : ℚ)) / (1 / 3)) = (20 / 27 : ℚ) :=
by
  sorry

end reciprocal_expression_l16_16949


namespace quadratic_polynomial_divisible_by_3_l16_16620

theorem quadratic_polynomial_divisible_by_3
  (a b c : ℤ)
  (h : ∀ x : ℤ, 3 ∣ (a * x^2 + b * x + c)) :
  3 ∣ a ∧ 3 ∣ b ∧ 3 ∣ c :=
sorry

end quadratic_polynomial_divisible_by_3_l16_16620


namespace indefinite_integral_solution_l16_16704

open Real

theorem indefinite_integral_solution (c : ℝ) : 
  ∫ x, (1 - cos x) / (x - sin x) ^ 2 = - 1 / (x - sin x) + c := 
sorry

end indefinite_integral_solution_l16_16704


namespace enrolled_percentage_l16_16957

theorem enrolled_percentage (total_students : ℝ) (non_bio_students : ℝ)
    (h_total : total_students = 880)
    (h_non_bio : non_bio_students = 440.00000000000006) : 
    ((total_students - non_bio_students) / total_students) * 100 = 50 := 
by
  rw [h_total, h_non_bio]
  norm_num
  sorry

end enrolled_percentage_l16_16957


namespace least_number_of_stamps_is_11_l16_16258

theorem least_number_of_stamps_is_11 (s t : ℕ) (h : 5 * s + 6 * t = 60) : s + t = 11 := 
  sorry

end least_number_of_stamps_is_11_l16_16258


namespace greatest_line_segment_length_l16_16246

theorem greatest_line_segment_length (r : ℝ) (h : r = 4) : 
  ∃ d : ℝ, d = 2 * r ∧ d = 8 :=
by
  sorry

end greatest_line_segment_length_l16_16246


namespace fewest_keystrokes_to_256_l16_16803

def fewest_keystrokes (start target : Nat) : Nat :=
if start = 1 && target = 256 then 8 else sorry

theorem fewest_keystrokes_to_256 : fewest_keystrokes 1 256 = 8 :=
by
  sorry

end fewest_keystrokes_to_256_l16_16803


namespace cubic_function_value_l16_16238

noncomputable def g (x : ℝ) (p q r s : ℝ) : ℝ := p * x ^ 3 + q * x ^ 2 + r * x + s

theorem cubic_function_value (p q r s : ℝ) (h : g (-3) p q r s = -2) :
  12 * p - 6 * q + 3 * r - s = 2 :=
sorry

end cubic_function_value_l16_16238


namespace distance_inequality_l16_16468

theorem distance_inequality 
  (A B C D : Point)
  (dist : Point → Point → ℝ)
  (h_dist_pos : ∀ P Q : Point, dist P Q ≥ 0)
  (AC BD AD BC AB CD : ℝ)
  (hAC : AC = dist A C)
  (hBD : BD = dist B D)
  (hAD : AD = dist A D)
  (hBC : BC = dist B C)
  (hAB : AB = dist A B)
  (hCD : CD = dist C D) :
  AC^2 + BD^2 + AD^2 + BC^2 ≥ AB^2 + CD^2 := 
by
  sorry

end distance_inequality_l16_16468


namespace xy_ratio_l16_16853

variables (x y z t : ℝ)
variables (hx : x > y) (hz : z = (x + y) / 2) (ht : t = Real.sqrt (x * y)) (h : x - y = 3 * (z - t))

theorem xy_ratio (x y : ℝ) (hx : x > y) (hz : z = (x + y) / 2) (ht : t = Real.sqrt (x * y)) (h : x - y = 3 * (z - t)) :
  x / y = 25 :=
sorry

end xy_ratio_l16_16853


namespace socks_total_l16_16947

def socks_lisa (initial: Nat) := initial + 0

def socks_sandra := 20

def socks_cousin (sandra: Nat) := sandra / 5

def socks_mom (initial: Nat) := 8 + 3 * initial

theorem socks_total (initial: Nat) (sandra: Nat) :
  initial = 12 → sandra = 20 → 
  socks_lisa initial + socks_sandra + socks_cousin sandra + socks_mom initial = 80 :=
by
  intros h_initial h_sandra
  rw [h_initial, h_sandra]
  sorry

end socks_total_l16_16947


namespace compare_f_values_max_f_value_l16_16945

noncomputable def f (x : ℝ) : ℝ := -2 * Real.sin x - Real.cos (2 * x)

theorem compare_f_values :
  f (Real.pi / 4) > f (Real.pi / 6) :=
sorry

theorem max_f_value :
  ∃ x : ℝ, f x = 3 :=
sorry

end compare_f_values_max_f_value_l16_16945


namespace simplify_expression_l16_16546

theorem simplify_expression (x y : ℤ) :
  (2 * x + 20) + (150 * x + 30) + y = 152 * x + 50 + y :=
by sorry

end simplify_expression_l16_16546


namespace billy_tickets_l16_16743

theorem billy_tickets (ferris_wheel_rides bumper_car_rides rides_per_ride total_tickets : ℕ) 
  (h1 : ferris_wheel_rides = 7)
  (h2 : bumper_car_rides = 3)
  (h3 : rides_per_ride = 5)
  (h4 : total_tickets = (ferris_wheel_rides + bumper_car_rides) * rides_per_ride) :
  total_tickets = 50 := 
by 
  sorry

end billy_tickets_l16_16743


namespace solve_for_x_l16_16467

theorem solve_for_x (x : ℝ) (h : 4 * x + 45 ≠ 0) :
  (8 * x^2 + 80 * x + 4) / (4 * x + 45) = 2 * x + 3 → x = -131 / 22 := 
by 
  sorry

end solve_for_x_l16_16467


namespace problem_part_a_problem_part_b_problem_part_e_problem_part_c_disproof_problem_part_d_disproof_l16_16010

-- Define Lean goals for the true statements
theorem problem_part_a (x : ℝ) (h : x < 0) : x^3 < x := sorry
theorem problem_part_b (x : ℝ) (h : x^3 > 0) : x > 0 := sorry
theorem problem_part_e (x : ℝ) (h : x > 1) : x^3 > x := sorry

-- Disprove the false statements by showing the negation
theorem problem_part_c_disproof (x : ℝ) (h : x^3 < x) : ¬ (|x| > 1) := sorry
theorem problem_part_d_disproof (x : ℝ) (h : x^3 > x) : ¬ (x > 1) := sorry

end problem_part_a_problem_part_b_problem_part_e_problem_part_c_disproof_problem_part_d_disproof_l16_16010


namespace surface_area_increase_factor_l16_16757

theorem surface_area_increase_factor (n : ℕ) (h : n > 0) : 
  (6 * n^3) / (6 * n^2) = n :=
by {
  sorry -- Proof not required
}

end surface_area_increase_factor_l16_16757


namespace min_value_inequality_l16_16522

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 3^x + 9^y ≥ 2 * Real.sqrt 3 := 
by
  sorry

end min_value_inequality_l16_16522


namespace complement_union_eq_complement_l16_16494

open Set

variable (U : Set ℤ) 
variable (A : Set ℤ) 
variable (B : Set ℤ)

theorem complement_union_eq_complement : 
  U = {-2, -1, 0, 1, 2, 3} →
  A = {-1, 2} →
  B = {x | x^2 - 4*x + 3 = 0} →
  (U \ (A ∪ B)) = {-2, 0} :=
by
  intros hU hA hB
  -- sorry to skip the proof
  sorry

end complement_union_eq_complement_l16_16494


namespace value_of_f_at_3_l16_16815

def f (a c x : ℝ) : ℝ := a * x^3 + c * x + 5

theorem value_of_f_at_3 (a c : ℝ) (h : f a c (-3) = -3) : f a c 3 = 13 :=
by
  sorry

end value_of_f_at_3_l16_16815


namespace initial_percentage_of_gold_l16_16305

theorem initial_percentage_of_gold (x : ℝ) (h₁ : 48 * x / 100 + 12 = 40 * 60 / 100) : x = 25 :=
by
  sorry

end initial_percentage_of_gold_l16_16305


namespace number_of_subsets_of_M_l16_16848

def M : Set ℝ := { x | x^2 - 2 * x + 1 = 0 }

theorem number_of_subsets_of_M : M = {1} → ∃ n, n = 2 := by
  sorry

end number_of_subsets_of_M_l16_16848


namespace find_dividend_l16_16461

-- Define the conditions
def divisor : ℕ := 20
def quotient : ℕ := 8
def remainder : ℕ := 6

-- Lean 4 statement to prove the dividend
theorem find_dividend : (divisor * quotient + remainder) = 166 := by
  sorry

end find_dividend_l16_16461


namespace analytic_expression_of_f_l16_16780

noncomputable def f (x : ℝ) := Real.sin (x + Real.pi / 2)

noncomputable def g (α : ℝ) := Real.cos (α - Real.pi / 3)

theorem analytic_expression_of_f :
  (∀ x, f x = Real.cos x) ∧
  (∀ α, α ∈ Set.Icc 0 Real.pi → g α = 1/2 → (α = 0 ∨ α = 2 * Real.pi / 3)) :=
by
  sorry

end analytic_expression_of_f_l16_16780


namespace parabola_chord_solution_l16_16418

noncomputable def parabola_chord : Prop :=
  ∃ x_A x_B : ℝ, (140 = 5 * x_B^2 + 2 * x_A^2) ∧ 
  ((x_A = -5 * Real.sqrt 2 ∧ x_B = 2 * Real.sqrt 2) ∨ 
   (x_A = 5 * Real.sqrt 2 ∧ x_B = -2 * Real.sqrt 2))

theorem parabola_chord_solution : parabola_chord := 
sorry

end parabola_chord_solution_l16_16418


namespace sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq_l16_16276

open Real

theorem sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq :
  (∀ α : ℝ, sin α = cos α → ∃ k : ℤ, α = (k : ℝ) * π + π / 4) ∧
  (¬ ∀ k : ℤ, ∀ α : ℝ, α = (k : ℝ) * π + π / 4 → sin α = cos α) :=
by
  sorry

end sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq_l16_16276


namespace range_a_if_no_solution_l16_16099

def f (x : ℝ) : ℝ := abs (x - abs (2 * x - 4))

theorem range_a_if_no_solution (a : ℝ) :
  (∀ x : ℝ, f x > 0 → false) → a < 1 :=
by
  sorry

end range_a_if_no_solution_l16_16099


namespace son_father_age_sum_l16_16640

theorem son_father_age_sum
    (S F : ℕ)
    (h1 : F - 6 = 3 * (S - 6))
    (h2 : F = 2 * S) :
    S + F = 36 :=
sorry

end son_father_age_sum_l16_16640


namespace range_of_a_neg_p_true_l16_16466

theorem range_of_a_neg_p_true :
  (∀ x : ℝ, x ∈ Set.Ioo (-2:ℝ) 0 → x^2 + (2*a - 1)*x + a ≠ 0) →
  ∀ a : ℝ, a ∈ Set.Icc 0 ((2 + Real.sqrt 3) / 2) :=
sorry

end range_of_a_neg_p_true_l16_16466


namespace joe_lift_ratio_l16_16516

theorem joe_lift_ratio (F S : ℕ) 
  (h1 : F + S = 1800) 
  (h2 : F = 700) 
  (h3 : 2 * F = S + 300) : F / S = 7 / 11 :=
by
  sorry

end joe_lift_ratio_l16_16516


namespace sandy_goal_hours_l16_16278

def goal_liters := 3 -- The goal in liters
def liters_to_milliliters := 1000 -- Conversion rate from liters to milliliters
def goal_milliliters := goal_liters * liters_to_milliliters -- Total milliliters to drink
def drink_rate_milliliters := 500 -- Milliliters drunk every interval
def interval_hours := 2 -- Interval in hours

def sets_to_goal := goal_milliliters / drink_rate_milliliters -- The number of drink sets to reach the goal
def total_hours := sets_to_goal * interval_hours -- Total time in hours to reach the goal

theorem sandy_goal_hours : total_hours = 12 := by
  -- Proof steps would go here
  sorry

end sandy_goal_hours_l16_16278


namespace gcd_546_210_l16_16463

theorem gcd_546_210 : Nat.gcd 546 210 = 42 := by
  sorry -- Proof is required to solve

end gcd_546_210_l16_16463


namespace bridge_length_proof_l16_16242

open Real

def train_length : ℝ := 100
def train_speed_kmh : ℝ := 45
def crossing_time_s: ℝ := 30

noncomputable def bridge_length : ℝ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem bridge_length_proof : bridge_length = 275 := 
by
  sorry

end bridge_length_proof_l16_16242


namespace revenue_increase_l16_16956

theorem revenue_increase
  (P Q : ℝ)
  (h : 0 < P)
  (hQ : 0 < Q)
  (price_decrease : 0.90 = 0.90)
  (unit_increase : 2 = 2) :
  (0.90 * P) * (2 * Q) = 1.80 * (P * Q) :=
by
  sorry

end revenue_increase_l16_16956


namespace race_distance_l16_16612

/-
In a race, the ratio of the speeds of two contestants A and B is 3 : 4.
A has a start of 140 m.
A wins by 20 m.
Prove that the total distance of the race is 360 times the common speed factor.
-/
theorem race_distance (x D : ℕ)
  (ratio_A_B : ∀ (speed_A speed_B : ℕ), speed_A / speed_B = 3 / 4)
  (start_A : ∀ (start : ℕ), start = 140) 
  (win_A : ∀ (margin : ℕ), margin = 20) :
  D = 360 * x := 
sorry

end race_distance_l16_16612


namespace eccentricity_range_l16_16942

-- Definitions and conditions
variable (a b c e : ℝ) (A B: ℝ × ℝ)
variable (d1 d2 : ℝ)

variable (a_pos : a > 2)
variable (b_pos : b > 0)
variable (c_pos : c > 0)
variable (c_eq : c = Real.sqrt (a ^ 2 + b ^ 2))
variable (A_def : A = (a, 0))
variable (B_def : B = (0, b))
variable (d1_def : d1 = abs (b * 2 + a * 0 - a * b ) / Real.sqrt (a^2 + b^2))
variable (d2_def : d2 = abs (b * (-2) + a * 0 - a * b) / Real.sqrt (a^2 + b^2))
variable (d_ineq : d1 + d2 ≥ (4 / 5) * c)
variable (eccentricity : e = c / a)

-- Theorem statement
theorem eccentricity_range : (Real.sqrt 5 / 2 ≤ e) ∧ (e ≤ Real.sqrt 5) :=
by sorry

end eccentricity_range_l16_16942


namespace number_of_dress_designs_is_correct_l16_16125

-- Define the number of choices for colors, patterns, and fabric types as conditions
def num_colors : Nat := 4
def num_patterns : Nat := 5
def num_fabric_types : Nat := 2

-- Define the total number of dress designs
def total_dress_designs : Nat := num_colors * num_patterns * num_fabric_types

-- Prove that the total number of different dress designs is 40
theorem number_of_dress_designs_is_correct : total_dress_designs = 40 := by
  sorry

end number_of_dress_designs_is_correct_l16_16125


namespace integer_roots_polynomial_l16_16965

theorem integer_roots_polynomial (a : ℤ) :
  (∃ x : ℤ, x^3 + 3 * x^2 + a * x + 9 = 0) ↔ 
  (a = -109 ∨ a = -21 ∨ a = -13 ∨ a = 3 ∨ a = 11 ∨ a = 53) :=
by
  sorry

end integer_roots_polynomial_l16_16965


namespace evaluate_72_squared_minus_48_squared_l16_16096

theorem evaluate_72_squared_minus_48_squared :
  (72:ℤ)^2 - (48:ℤ)^2 = 2880 :=
by
  sorry

end evaluate_72_squared_minus_48_squared_l16_16096


namespace ratio_volumes_l16_16511

noncomputable def V_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def V_cone (r : ℝ) : ℝ := (1 / 3) * Real.pi * r^3

theorem ratio_volumes (r : ℝ) (hr : r > 0) : 
  (V_cone r) / (V_sphere r) = 1 / 4 :=
by
  sorry

end ratio_volumes_l16_16511


namespace min_abs_sum_l16_16886

-- Definitions based on given conditions for the problem
variable (p q r s : ℤ)
variable (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
variable (h : (matrix 2 2 ℤ ![(p, q), (r, s)]) ^ 2 = matrix 2 2 ℤ ![(9, 0), (0, 9)])

-- Statement of the proof problem
theorem min_abs_sum :
  |p| + |q| + |r| + |s| = 8 :=
by
  sorry

end min_abs_sum_l16_16886


namespace measure_angle_R_l16_16927

theorem measure_angle_R (P Q R : ℝ) (h1 : P + Q = 60) : R = 120 :=
by
  have sum_of_angles_in_triangle : P + Q + R = 180 := sorry
  rw [h1] at sum_of_angles_in_triangle
  linarith

end measure_angle_R_l16_16927


namespace ball_attendance_l16_16396

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end ball_attendance_l16_16396


namespace solve_quadratic_solution_l16_16742

theorem solve_quadratic_solution (x : ℝ) : (3 * x^2 - 6 * x = 0) ↔ (x = 0 ∨ x = 2) :=
sorry

end solve_quadratic_solution_l16_16742


namespace no_integer_b_satisfies_conditions_l16_16303

theorem no_integer_b_satisfies_conditions :
  ¬ ∃ b : ℕ, b^6 ≤ 196 ∧ 196 < b^7 :=
by
  sorry

end no_integer_b_satisfies_conditions_l16_16303


namespace incorrect_judgment_l16_16308

-- Define propositions p and q
def p : Prop := 2 + 2 = 5
def q : Prop := 3 > 2

-- The incorrect judgment in Lean statement
theorem incorrect_judgment : ¬((p ∧ q) ∧ ¬p) :=
by
  sorry

end incorrect_judgment_l16_16308


namespace bridge_length_is_correct_l16_16786

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_seconds : ℝ) : ℝ :=
  let speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * crossing_time_seconds
  total_distance - train_length

theorem bridge_length_is_correct :
  length_of_bridge 200 (60) 45 = 550.15 :=
by
  sorry

end bridge_length_is_correct_l16_16786


namespace sum_x_coordinates_intersection_mod_9_l16_16035

theorem sum_x_coordinates_intersection_mod_9 :
  ∃ x y : ℤ, (y ≡ 3 * x + 4 [ZMOD 9]) ∧ (y ≡ 7 * x + 2 [ZMOD 9]) ∧ x ≡ 5 [ZMOD 9] := sorry

end sum_x_coordinates_intersection_mod_9_l16_16035


namespace age_ratios_l16_16420

variable (A B : ℕ)

-- Given conditions
theorem age_ratios :
  (A / B = 2 / 1) → (A - 4 = B + 4) → ((A + 4) / (B - 4) = 5 / 1) :=
by
  intro h1 h2
  sorry

end age_ratios_l16_16420


namespace solve_for_x_l16_16350

noncomputable def proof (x : ℚ) : Prop :=
  (x + 6) / (x - 4) = (x - 7) / (x + 2)

theorem solve_for_x (x : ℚ) (h : proof x) : x = 16 / 19 :=
by
  sorry

end solve_for_x_l16_16350


namespace solution_interval_l16_16280

theorem solution_interval:
  ∃ x : ℝ, (x^3 = 2^(2-x)) ∧ 1 < x ∧ x < 2 :=
by
  sorry

end solution_interval_l16_16280


namespace baking_powder_difference_l16_16313

-- Define the known quantities
def baking_powder_yesterday : ℝ := 0.4
def baking_powder_now : ℝ := 0.3

-- Define the statement to prove, i.e., the difference in baking powder
theorem baking_powder_difference : baking_powder_yesterday - baking_powder_now = 0.1 :=
by
  -- Proof omitted
  sorry

end baking_powder_difference_l16_16313


namespace sufficiency_not_necessity_l16_16699

def l1 : Type := sorry
def l2 : Type := sorry

def skew_lines (l1 l2 : Type) : Prop := sorry
def do_not_intersect (l1 l2 : Type) : Prop := sorry

theorem sufficiency_not_necessity (p q : Prop) 
  (hp : p = skew_lines l1 l2)
  (hq : q = do_not_intersect l1 l2) :
  (p → q) ∧ ¬ (q → p) :=
by {
  sorry
}

end sufficiency_not_necessity_l16_16699


namespace percentage_of_stock_l16_16094

noncomputable def investment_amount : ℝ := 6000
noncomputable def income_derived : ℝ := 756
noncomputable def brokerage_percentage : ℝ := 0.25
noncomputable def brokerage_fee : ℝ := investment_amount * (brokerage_percentage / 100)
noncomputable def net_investment_amount : ℝ := investment_amount - brokerage_fee
noncomputable def dividend_yield : ℝ := (income_derived / net_investment_amount) * 100

theorem percentage_of_stock :
  ∃ (percentage_of_stock : ℝ), percentage_of_stock = dividend_yield := by
  sorry

end percentage_of_stock_l16_16094


namespace geometric_sequence_a4_a5_l16_16636

open BigOperators

theorem geometric_sequence_a4_a5 (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 + a 2 = 1)
  (h3 : a 3 + a 4 = 9) : 
  a 4 + a 5 = 27 ∨ a 4 + a 5 = -27 :=
sorry

end geometric_sequence_a4_a5_l16_16636


namespace sum_ratios_l16_16899

variable (a b d : ℕ)

def A_n (a b d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def arithmetic_sum (a n d : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem sum_ratios (k : ℕ) (h1 : 2 * (a + d) = 7 * k) (h2 : 4 * (a + 3 * d) = 6 * k) :
  arithmetic_sum a 7 d / arithmetic_sum a 3 d = 2 / 1 :=
by
  sorry

end sum_ratios_l16_16899


namespace train_speed_l16_16053

theorem train_speed
    (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ)
    (h_train : length_train = 250)
    (h_platform : length_platform = 250.04)
    (h_time : time_seconds = 25) :
    (length_train + length_platform) / time_seconds * 3.6 = 72.006 :=
by sorry

end train_speed_l16_16053


namespace find_b10_l16_16581

def seq (b : ℕ → ℕ) :=
  (b 1 = 2)
  ∧ (∀ m n, b (m + n) = b m + b n + 2 * m * n)

theorem find_b10 (b : ℕ → ℕ) (h : seq b) : b 10 = 110 :=
by 
  -- Proof omitted, as requested.
  sorry

end find_b10_l16_16581


namespace smaller_angle_at_7_15_l16_16226

theorem smaller_angle_at_7_15 (h_angle : ℝ) (m_angle : ℝ) : 
  h_angle = 210 + 0.5 * 15 →
  m_angle = 90 →
  min (abs (h_angle - m_angle)) (360 - abs (h_angle - m_angle)) = 127.5 :=
  by
    intros h_eq m_eq
    rw [h_eq, m_eq]
    sorry

end smaller_angle_at_7_15_l16_16226


namespace earnings_last_friday_l16_16149

theorem earnings_last_friday 
  (price_per_kg : ℕ := 2)
  (earnings_wednesday : ℕ := 30)
  (earnings_today : ℕ := 42)
  (total_kg_sold : ℕ := 48)
  (total_earnings : ℕ := total_kg_sold * price_per_kg) 
  (F : ℕ) :
  earnings_wednesday + F + earnings_today = total_earnings → F = 24 := by
  sorry

end earnings_last_friday_l16_16149


namespace curlers_count_l16_16572

theorem curlers_count (T P B G : ℕ) 
  (hT : T = 16)
  (hP : P = T / 4)
  (hB : B = 2 * P)
  (hG : G = T - (P + B)) : 
  G = 4 :=
by
  sorry

end curlers_count_l16_16572


namespace percentage_of_400_that_results_in_224_point_5_l16_16287

-- Let x be the unknown percentage of 400
variable (x : ℝ)

-- Condition: x% of 400 plus 45% of 250 equals 224.5
def condition (x : ℝ) : Prop := (400 * x / 100) + (250 * 45 / 100) = 224.5

theorem percentage_of_400_that_results_in_224_point_5 : condition 28 :=
by
  -- proof goes here
  sorry

end percentage_of_400_that_results_in_224_point_5_l16_16287


namespace given_statements_l16_16285

def addition_is_associative (x y z : ℝ) : Prop := (x + y) + z = x + (y + z)

def averaging_is_commutative (x y : ℝ) : Prop := (x + y) / 2 = (y + x) / 2

def addition_distributes_over_averaging (x y z : ℝ) : Prop := 
  x + (y + z) / 2 = (x + y + x + z) / 2

def averaging_distributes_over_addition (x y z : ℝ) : Prop := 
  (x + (y + z)) / 2 = ((x + y) / 2) + ((x + z) / 2)

def averaging_has_identity_element (x e : ℝ) : Prop := 
  (x + e) / 2 = x

theorem given_statements (x y z e : ℝ) :
  addition_is_associative x y z ∧ 
  averaging_is_commutative x y ∧ 
  addition_distributes_over_averaging x y z ∧ 
  ¬averaging_distributes_over_addition x y z ∧ 
  ¬∃ e, averaging_has_identity_element x e :=
by
  sorry

end given_statements_l16_16285


namespace original_deck_card_count_l16_16407

theorem original_deck_card_count (r b : ℕ) 
  (h1 : r / (r + b) = 1 / 4)
  (h2 : r / (r + b + 6) = 1 / 6) : r + b = 12 :=
sorry

end original_deck_card_count_l16_16407


namespace problem_statement_l16_16327

theorem problem_statement (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0) : (x - y - z) ^ 2002 = 0 :=
sorry

end problem_statement_l16_16327


namespace determinant_inequality_l16_16973

variable (x : ℝ)

def determinant (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem determinant_inequality (h : determinant 2 (3 - x) 1 x > 0) : x > 1 := by
  sorry

end determinant_inequality_l16_16973


namespace inequality_solution_l16_16533

theorem inequality_solution (m : ℝ) (h : m < -1) :
  (if m = -3 then
    {x : ℝ | x > 1} =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else if -3 < m ∧ m < -1 then
    ({x : ℝ | x < m / (m + 3)} ∪ {x : ℝ | x > 1}) =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else if m < -3 then
    {x : ℝ | 1 < x ∧ x < m / (m + 3)} =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else
    False) :=
by
  sorry

end inequality_solution_l16_16533


namespace basketball_free_throws_l16_16324

-- Define the given conditions as assumptions
variables {a b x : ℝ}
variables (h1 : 3 * b = 2 * a)
variables (h2 : x = 2 * a - 2)
variables (h3 : 2 * a + 3 * b + x = 78)

-- State the theorem to be proven
theorem basketball_free_throws : x = 74 / 3 :=
by {
  -- We will provide the proof later
  sorry
}

end basketball_free_throws_l16_16324


namespace binomial_12_10_eq_66_l16_16891

theorem binomial_12_10_eq_66 : (Nat.choose 12 10) = 66 :=
by
  sorry

end binomial_12_10_eq_66_l16_16891


namespace hcf_of_12_and_15_l16_16131

-- Definitions of LCM and HCF
def LCM (a b : ℕ) : ℕ := sorry  -- Placeholder for actual LCM definition
def HCF (a b : ℕ) : ℕ := sorry  -- Placeholder for actual HCF definition

theorem hcf_of_12_and_15 :
  LCM 12 15 = 60 → HCF 12 15 = 3 :=
by
  sorry

end hcf_of_12_and_15_l16_16131


namespace solve_system_of_equations_l16_16834

/-- Definition representing our system of linear equations. --/
def system_of_equations (x1 x2 : ℚ) : Prop :=
  (3 * x1 - 5 * x2 = 2) ∧ (2 * x1 + 4 * x2 = 5)

/-- The main theorem stating the solution to our system of equations. --/
theorem solve_system_of_equations : 
  ∃ x1 x2 : ℚ, system_of_equations x1 x2 ∧ x1 = 3/2 ∧ x2 = 1/2 :=
by
  sorry

end solve_system_of_equations_l16_16834


namespace find_f_of_4_l16_16248

def f (a b c x : ℝ) := a * x^2 + b * x + c

theorem find_f_of_4 {a b c : ℝ} (h1 : f a b c 1 = 3) (h2 : f a b c 2 = 12) (h3 : f a b c 3 = 27) :
  f a b c 4 = 48 := 
sorry

end find_f_of_4_l16_16248


namespace unique_triple_satisfying_conditions_l16_16508

theorem unique_triple_satisfying_conditions :
  ∃! (x y z : ℝ), x + y = 4 ∧ xy - z^2 = 4 :=
sorry

end unique_triple_satisfying_conditions_l16_16508


namespace find_x_squared_plus_inverse_squared_l16_16897

theorem find_x_squared_plus_inverse_squared (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^2 + (1 / x)^2 = 7 :=
by
  sorry

end find_x_squared_plus_inverse_squared_l16_16897


namespace kiwi_lemon_relationship_l16_16884

open Nat

-- Define the conditions
def total_fruits : ℕ := 58
def mangoes : ℕ := 18
def pears : ℕ := 10
def pawpaws : ℕ := 12
def lemons_in_last_two_baskets : ℕ := 9

-- Define the question and the proof goal
theorem kiwi_lemon_relationship :
  ∃ (kiwis lemons : ℕ), 
  kiwis = lemons_in_last_two_baskets ∧ 
  lemons = lemons_in_last_two_baskets ∧ 
  kiwis + lemons = total_fruits - (mangoes + pears + pawpaws) :=
sorry

end kiwi_lemon_relationship_l16_16884


namespace cannon_hit_probability_l16_16984

theorem cannon_hit_probability
  (P1 P2 P3 : ℝ)
  (h1 : P1 = 0.2)
  (h3 : P3 = 0.3)
  (h_none_hit : (1 - P1) * (1 - P2) * (1 - P3) = 0.27999999999999997) :
  P2 = 0.5 :=
by
  sorry

end cannon_hit_probability_l16_16984


namespace shonda_kids_calculation_l16_16348

def number_of_kids (B E P F A : Nat) : Nat :=
  let T := B * E
  let total_people := T / P
  total_people - (F + A + 1)

theorem shonda_kids_calculation :
  (number_of_kids 15 12 9 10 7) = 2 :=
by
  unfold number_of_kids
  exact rfl

end shonda_kids_calculation_l16_16348


namespace exist_infinite_a_l16_16277

theorem exist_infinite_a (n : ℕ) (a : ℕ) (h₁ : ∃ k : ℕ, k > 0 ∧ (n^6 + 3 * a = (n^2 + 3 * k)^3)) : 
  ∃ f : ℕ → ℕ, ∀ m : ℕ, (∃ k : ℕ, k > 0 ∧ f m = 9 * k^3 + 3 * n^2 * k * (n^2 + 3 * k)) :=
by 
  sorry

end exist_infinite_a_l16_16277


namespace nested_expression_value_l16_16194

theorem nested_expression_value : 
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))))))) = 87380 :=
by 
  sorry

end nested_expression_value_l16_16194


namespace total_interest_calculation_l16_16236

-- Define the total investment
def total_investment : ℝ := 20000

-- Define the fractional part of investment at 9 percent rate
def fraction_higher_rate : ℝ := 0.55

-- Define the investment amounts based on the fractional part
def investment_higher_rate : ℝ := fraction_higher_rate * total_investment
def investment_lower_rate : ℝ := total_investment - investment_higher_rate

-- Define interest rates
def rate_lower : ℝ := 0.06
def rate_higher : ℝ := 0.09

-- Define time period (in years)
def time_period : ℝ := 1

-- Define interest calculations
def interest_lower : ℝ := investment_lower_rate * rate_lower * time_period
def interest_higher : ℝ := investment_higher_rate * rate_higher * time_period

-- Define the total interest
def total_interest : ℝ := interest_lower + interest_higher

-- Theorem stating the total interest earned
theorem total_interest_calculation : total_interest = 1530 := by
  -- skip proof using sorry
  sorry

end total_interest_calculation_l16_16236


namespace initial_kittens_l16_16990

theorem initial_kittens (x : ℕ) (h : x + 3 = 9) : x = 6 :=
by {
  sorry
}

end initial_kittens_l16_16990


namespace cheat_buying_percentage_l16_16262

-- Definitions for the problem
def profit_margin := 0.5
def cheat_selling := 0.2

-- Prove that the cheating percentage while buying is 20%
theorem cheat_buying_percentage : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ x = 0.2 := by
  sorry

end cheat_buying_percentage_l16_16262


namespace fewer_pushups_l16_16548

theorem fewer_pushups (sets: ℕ) (pushups_per_set : ℕ) (total_pushups : ℕ) 
  (h1 : sets = 3) (h2 : pushups_per_set = 15) (h3 : total_pushups = 40) :
  sets * pushups_per_set - total_pushups = 5 :=
by
  sorry

end fewer_pushups_l16_16548


namespace compute_xy_l16_16878

theorem compute_xy (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 :=
sorry

end compute_xy_l16_16878


namespace transylvanian_sanity_l16_16244

theorem transylvanian_sanity (sane : Prop) (belief : Prop) (h1 : sane) (h2 : sane → belief) : belief :=
by
  sorry

end transylvanian_sanity_l16_16244


namespace ratio_of_bike_to_tractor_speed_l16_16085

theorem ratio_of_bike_to_tractor_speed (d_tr: ℝ) (t_tr: ℝ) (d_car: ℝ) (t_car: ℝ) (k: ℝ) (β: ℝ) 
  (h1: d_tr / t_tr = 25) 
  (h2: d_car / t_car = 90)
  (h3: 90 = 9 / 5 * β)
: β / (d_tr / t_tr) = 2 := 
by
  sorry

end ratio_of_bike_to_tractor_speed_l16_16085


namespace binomial_coeff_and_coeff_of_x8_l16_16624

theorem binomial_coeff_and_coeff_of_x8 (x : ℂ) :
  let expr := (x^2 + 4*x + 4)^5
  let expansion := (x + 2)^10
  ∃ (binom_coeff_x8 coeff_x8 : ℤ),
    binom_coeff_x8 = 45 ∧ coeff_x8 = 180 :=
by
  sorry

end binomial_coeff_and_coeff_of_x8_l16_16624


namespace shortest_chord_value_of_m_l16_16901

theorem shortest_chord_value_of_m :
  (∃ m : ℝ,
      (∀ x y : ℝ, mx + y - 2 * m - 1 = 0) ∧
      (∀ x y : ℝ, x ^ 2 + y ^ 2 - 2 * x - 4 * y = 0) ∧
      (mx + y - 2 * m - 1 = 0 → ∃ x y : ℝ, (x, y) = (2, 1))
  ) → m = -1 :=
by
  sorry

end shortest_chord_value_of_m_l16_16901


namespace angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees_l16_16633

-- Define the concept of a cube and the diagonals of its faces.
structure Cube :=
  (faces : Fin 6 → (Fin 4 → ℝ × ℝ × ℝ))    -- Representing each face as a set of four vertices in 3D space

def is_square_face (face : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  -- A function that checks if a given set of four vertices forms a square face.
  sorry

def are_adjacent_faces_perpendicular_diagonals 
  (face1 face2 : Fin 4 → ℝ × ℝ × ℝ) (c : Cube) : Prop :=
  -- A function that checks if the diagonals of two given adjacent square faces of a cube are perpendicular.
  sorry

-- The theorem stating the required proof:
theorem angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees
  (c : Cube)
  (h1 : is_square_face (c.faces 0))
  (h2 : is_square_face (c.faces 1))
  (h_adj: are_adjacent_faces_perpendicular_diagonals (c.faces 0) (c.faces 1) c) :
  ∃ q : ℝ, q = 90 :=
by
  sorry

end angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees_l16_16633


namespace num_second_grade_students_is_80_l16_16214

def ratio_fst : ℕ := 5
def ratio_snd : ℕ := 4
def ratio_trd : ℕ := 3
def total_students : ℕ := 240

def second_grade : ℕ := (ratio_snd * total_students) / (ratio_fst + ratio_snd + ratio_trd)

theorem num_second_grade_students_is_80 :
  second_grade = 80 := 
sorry

end num_second_grade_students_is_80_l16_16214


namespace inequality_solution_l16_16317

theorem inequality_solution (x : ℝ) :
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 2 ↔
  4 - Real.sqrt 21 < x ∧ x < 4 + Real.sqrt 21 :=
sorry

end inequality_solution_l16_16317


namespace flyers_total_l16_16712

theorem flyers_total (jack_flyers : ℕ) (rose_flyers : ℕ) (left_flyers : ℕ) 
  (hj : jack_flyers = 120) (hr : rose_flyers = 320) (hl : left_flyers = 796) :
  jack_flyers + rose_flyers + left_flyers = 1236 :=
by {
  sorry
}

end flyers_total_l16_16712


namespace negation_p_l16_16314

theorem negation_p (p : Prop) : 
  (∃ x : ℝ, x^2 ≥ x) ↔ ¬ (∀ x : ℝ, x^2 < x) :=
by 
  -- The proof is omitted
  sorry

end negation_p_l16_16314


namespace find_ratio_of_arithmetic_sequences_l16_16607

variable {a_n b_n : ℕ → ℕ}
variable {A_n B_n : ℕ → ℝ}

def arithmetic_sums (a_n b_n : ℕ → ℕ) (A_n B_n : ℕ → ℝ) : Prop :=
  ∀ n, A_n n = (n * (2 * a_n 1 + (n - 1) * (a_n 8 - a_n 7))) / 2 ∧
         B_n n = (n * (2 * b_n 1 + (n - 1) * (b_n 8 - b_n 7))) / 2

theorem find_ratio_of_arithmetic_sequences 
    (h : ∀ n, A_n n / B_n n = (5 * n - 3) / (n + 9)) :
    ∃ r : ℝ, r = 3 := by
  sorry

end find_ratio_of_arithmetic_sequences_l16_16607


namespace sum_of_integers_l16_16761

theorem sum_of_integers (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1)
  (h4 : a * b * c = 343000)
  (h5 : Nat.gcd a b = 1) (h6 : Nat.gcd b c = 1) (h7 : Nat.gcd a c = 1) :
  a + b + c = 476 :=
by
  sorry

end sum_of_integers_l16_16761


namespace sunset_time_correct_l16_16220

theorem sunset_time_correct : 
  let sunrise := (6 * 60 + 43)       -- Sunrise time in minutes (6:43 AM)
  let daylight := (11 * 60 + 56)     -- Length of daylight in minutes (11:56)
  let sunset := (sunrise + daylight) % (24 * 60) -- Calculate sunset time considering 24-hour cycle
  let sunset_hour := sunset / 60     -- Convert sunset time back into hours
  let sunset_minute := sunset % 60   -- Calculate remaining minutes
  (sunset_hour - 12, sunset_minute) = (6, 39)    -- Convert to 12-hour format and check against 6:39 PM
:= by
  sorry

end sunset_time_correct_l16_16220


namespace product_approximation_l16_16631

theorem product_approximation :
  (3.05 * 7.95 * (6.05 + 3.95)) = 240 := by
  sorry

end product_approximation_l16_16631


namespace greatest_x_l16_16182

theorem greatest_x (x : ℕ) : (x^6 / x^3 ≤ 27) → x ≤ 3 :=
by sorry

end greatest_x_l16_16182


namespace a_n_values_l16_16648

noncomputable def a : ℕ → ℕ := sorry
noncomputable def S : ℕ → ℕ := sorry

axiom Sn_property (n : ℕ) (hn : n > 0) : S n = 2 * (a n) - n

theorem a_n_values : a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 7 ∧ ∀ n : ℕ, n > 0 → a n = 2^n - 1 := 
by sorry

end a_n_values_l16_16648


namespace point_outside_circle_l16_16250

theorem point_outside_circle (a b : ℝ)
  (h_line_intersects_circle : ∃ (x1 y1 x2 y2 : ℝ), 
     x1^2 + y1^2 = 1 ∧ 
     x2^2 + y2^2 = 1 ∧ 
     a * x1 + b * y1 = 1 ∧ 
     a * x2 + b * y2 = 1 ∧ 
     (x1, y1) ≠ (x2, y2)) : 
  a^2 + b^2 > 1 :=
sorry

end point_outside_circle_l16_16250


namespace milk_packet_volume_l16_16609

theorem milk_packet_volume :
  ∃ (m : ℕ), (150 * m = 1250 * 30) ∧ m = 250 :=
by
  sorry

end milk_packet_volume_l16_16609


namespace minimum_value_l16_16551

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem minimum_value (a b c d : ℝ) (h1 : a < (2 / 3) * b) 
  (h2 : ∀ x, 3 * a * x^2 + 2 * b * x + c ≥ 0) : 
  ∃ (x : ℝ), ∀ c, 2 * b - 3 * a ≠ 0 → (c = (b^2 / 3 / a)) → (c / (2 * b - 3 * a) ≥ 1) :=
by
  sorry

end minimum_value_l16_16551


namespace average_rate_of_change_l16_16903

noncomputable def f (x : ℝ) := 2 * x + 1

theorem average_rate_of_change :
  (f 2 - f 1) / (2 - 1) = 2 :=
by
  sorry

end average_rate_of_change_l16_16903


namespace scientific_notation_of_0point0000025_l16_16038

theorem scientific_notation_of_0point0000025 : ∃ (a : ℝ) (n : ℤ), 0.0000025 = a * 10 ^ n ∧ a = 2.5 ∧ n = -6 :=
by {
  sorry
}

end scientific_notation_of_0point0000025_l16_16038


namespace smallest_positive_value_of_a_minus_b_l16_16550

theorem smallest_positive_value_of_a_minus_b :
  ∃ (a b : ℤ), 17 * a + 6 * b = 13 ∧ a - b = 17 :=
by
  sorry

end smallest_positive_value_of_a_minus_b_l16_16550


namespace ceil_minus_x_of_fractional_part_half_l16_16130

theorem ceil_minus_x_of_fractional_part_half (x : ℝ) (hx : x - ⌊x⌋ = 1 / 2) : ⌈x⌉ - x = 1 / 2 :=
by
 sorry

end ceil_minus_x_of_fractional_part_half_l16_16130


namespace find_roots_square_sum_and_min_y_l16_16746

-- Definitions from the conditions
def sum_roots (m : ℝ) :=
  -(m + 1)

def product_roots (m : ℝ) :=
  2 * m - 2

def roots_square_sum (m x₁ x₂ : ℝ) :=
  x₁^2 + x₂^2

def y (m : ℝ) :=
  (m - 1)^2 + 4

-- Proof statement
theorem find_roots_square_sum_and_min_y (m x₁ x₂ : ℝ) (h_sum : x₁ + x₂ = sum_roots m)
  (h_prod : x₁ * x₂ = product_roots m) :
  roots_square_sum m x₁ x₂ = (m - 1)^2 + 4 ∧ y m ≥ 4 :=
by
  sorry

end find_roots_square_sum_and_min_y_l16_16746


namespace area_of_triangle_FYG_l16_16732

theorem area_of_triangle_FYG (EF GH : ℝ) 
  (EF_len : EF = 15) 
  (GH_len : GH = 25) 
  (area_trapezoid : 0.5 * (EF + GH) * 10 = 200) 
  (intersection : true) -- Placeholder for intersection condition
  : 0.5 * GH * 3.75 = 46.875 := 
sorry

end area_of_triangle_FYG_l16_16732


namespace valid_range_of_x_l16_16401

theorem valid_range_of_x (x : ℝ) : 3 * x + 5 ≥ 0 → x ≥ -5 / 3 := 
by
  sorry

end valid_range_of_x_l16_16401


namespace people_in_each_van_l16_16621

theorem people_in_each_van
  (cars : ℕ) (taxis : ℕ) (vans : ℕ)
  (people_per_car : ℕ) (people_per_taxi : ℕ) (total_people : ℕ) 
  (people_per_van : ℕ) :
  cars = 3 → taxis = 6 → vans = 2 →
  people_per_car = 4 → people_per_taxi = 6 → total_people = 58 →
  3 * people_per_car + 6 * people_per_taxi + 2 * people_per_van = total_people →
  people_per_van = 5 :=
by sorry

end people_in_each_van_l16_16621


namespace find_number_l16_16156

theorem find_number (x : ℝ) : 3 * (2 * x + 9) = 57 → x = 5 :=
by
  sorry

end find_number_l16_16156


namespace fabian_cards_l16_16253

theorem fabian_cards : ∃ (g y b r : ℕ),
  (g > 0 ∧ g < 10) ∧ (y > 0 ∧ y < 10) ∧ (b > 0 ∧ b < 10) ∧ (r > 0 ∧ r < 10) ∧
  (g * y = g) ∧
  (b = r) ∧
  (b * r = 10 * g + y) ∧ 
  (g = 8) ∧
  (y = 1) ∧
  (b = 9) ∧
  (r = 9) :=
by
  sorry

end fabian_cards_l16_16253


namespace marbles_left_mrs_hilt_marbles_left_l16_16564

-- Define the initial number of marbles
def initial_marbles : ℕ := 38

-- Define the number of marbles lost
def marbles_lost : ℕ := 15

-- Define the number of marbles given away
def marbles_given_away : ℕ := 6

-- Define the number of marbles found
def marbles_found : ℕ := 8

-- Use these definitions to calculate the total number of marbles left
theorem marbles_left : ℕ :=
  initial_marbles - marbles_lost - marbles_given_away + marbles_found

-- Prove that total number of marbles left is 25
theorem mrs_hilt_marbles_left : marbles_left = 25 := by 
  sorry

end marbles_left_mrs_hilt_marbles_left_l16_16564


namespace radius_of_bicycle_wheel_is_13_l16_16167

-- Define the problem conditions
def diameter_cm : ℕ := 26

-- Define the function to calculate radius from diameter
def radius (d : ℕ) : ℕ := d / 2

-- Prove that the radius is 13 cm when diameter is 26 cm
theorem radius_of_bicycle_wheel_is_13 :
  radius diameter_cm = 13 := 
sorry

end radius_of_bicycle_wheel_is_13_l16_16167


namespace sqrt_200_eq_l16_16127

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l16_16127


namespace real_number_identity_l16_16459

theorem real_number_identity (a : ℝ) (h : a^2 - a - 1 = 0) : a^8 + 7 * a^(-(4:ℝ)) = 48 := by
  sorry

end real_number_identity_l16_16459


namespace max_marks_l16_16180

theorem max_marks (M : ℝ) (h_pass : 0.33 * M = 165) : M = 500 := 
by
  sorry

end max_marks_l16_16180


namespace two_digit_number_system_l16_16056

theorem two_digit_number_system (x y : ℕ) :
  (10 * x + y - 3 * (x + y) = 13) ∧ (10 * x + y - 6 = 4 * (x + y)) :=
by sorry

end two_digit_number_system_l16_16056


namespace sequence_2019_value_l16_16914

theorem sequence_2019_value :
  ∃ a : ℕ → ℤ, (∀ n ≥ 4, a n = a (n-1) * a (n-3)) ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ a 2019 = -1 :=
by
  sorry

end sequence_2019_value_l16_16914


namespace triangle_distance_bisectors_l16_16979

noncomputable def distance_between_bisectors {a b c : ℝ} (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) : ℝ :=
  (2 * a * b * c) / (b^2 - c^2)

theorem triangle_distance_bisectors 
  (a b c : ℝ) (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) :
  ∀ (DD₁ : ℝ), 
  DD₁ = distance_between_bisectors h₁ h₂ h₃ → 
  DD₁ = (2 * a * b * c) / (b^2 - c^2) := by 
  sorry

end triangle_distance_bisectors_l16_16979


namespace calculate_percentage_increase_l16_16164

variable (fish_first_round : ℕ) (fish_second_round : ℕ) (fish_total : ℕ) (fish_last_round : ℕ) (increase : ℚ) (percentage_increase : ℚ)

theorem calculate_percentage_increase
  (h1 : fish_first_round = 8)
  (h2 : fish_second_round = fish_first_round + 12)
  (h3 : fish_total = 60)
  (h4 : fish_last_round = fish_total - (fish_first_round + fish_second_round))
  (h5 : increase = fish_last_round - fish_second_round)
  (h6 : percentage_increase = (increase / fish_second_round) * 100) :
  percentage_increase = 60 := by
  sorry

end calculate_percentage_increase_l16_16164


namespace decimal_to_base_five_correct_l16_16810

theorem decimal_to_base_five_correct : 
  ∃ (d0 d1 d2 d3 : ℕ), 256 = d3 * 5^3 + d2 * 5^2 + d1 * 5^1 + d0 * 5^0 ∧ 
                          d3 = 2 ∧ d2 = 0 ∧ d1 = 1 ∧ d0 = 1 :=
by sorry

end decimal_to_base_five_correct_l16_16810


namespace gasoline_tank_capacity_l16_16850

theorem gasoline_tank_capacity :
  ∀ (x : ℕ), (5 / 6 * (x : ℚ) - 18 = 1 / 3 * (x : ℚ)) → x = 36 :=
by
  sorry

end gasoline_tank_capacity_l16_16850


namespace find_m_l16_16168

/-
Define the ellipse equation
-/
def ellipse_eqn (x y : ℝ) : Prop :=
  (x^2 / 9) + (y^2) = 1

/-
Define the region R
-/
def region_R (x y : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (2*y = x) ∧ ellipse_eqn x y

/-
Define the region R'
-/
def region_R' (x y m : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (y = m*x) ∧ ellipse_eqn x y

/-
The statement we want to prove
-/
theorem find_m (m : ℝ) : (∃ (x y : ℝ), region_R x y) ∧ (∃ (x y : ℝ), region_R' x y m) →
(m = (2 : ℝ) / 9) := 
sorry

end find_m_l16_16168


namespace volume_of_quadrilateral_pyramid_l16_16083

theorem volume_of_quadrilateral_pyramid (m α : ℝ) : 
  ∃ (V : ℝ), V = (2 / 3) * m^3 * (Real.cos α) * (Real.sin (2 * α)) :=
by
  sorry

end volume_of_quadrilateral_pyramid_l16_16083


namespace expand_expression_l16_16971

theorem expand_expression : ∀ (x : ℝ), (17 * x + 21) * 3 * x = 51 * x^2 + 63 * x :=
by
  intro x
  sorry

end expand_expression_l16_16971


namespace digits_in_8_20_3_30_base_12_l16_16666

def digits_in_base (n b : ℕ) : ℕ :=
  if n = 0 then 1 else 1 + Nat.log b n

theorem digits_in_8_20_3_30_base_12 : digits_in_base (8^20 * 3^30) 12 = 31 :=
by
  sorry

end digits_in_8_20_3_30_base_12_l16_16666


namespace angle_A_eq_pi_over_3_perimeter_eq_24_l16_16538

namespace TriangleProof

-- We introduce the basic setup for the triangle
variables {A B C : ℝ} {a b c : ℝ}

-- Given condition
axiom condition : 2 * b = 2 * a * Real.cos C + c

-- Part 1: Prove angle A is π/3
theorem angle_A_eq_pi_over_3 (h : 2 * b = 2 * a * Real.cos C + c) :
  A = Real.pi / 3 :=
sorry

-- Part 2: Given a = 10 and the area is 8√3, prove perimeter is 24
theorem perimeter_eq_24 (a_eq_10 : a = 10) (area_eq_8sqrt3 : 8 * Real.sqrt 3 = (1 / 2) * b * c * Real.sin A) :
  a + b + c = 24 :=
sorry

end TriangleProof

end angle_A_eq_pi_over_3_perimeter_eq_24_l16_16538


namespace walking_total_distance_l16_16650

theorem walking_total_distance :
  let t1 := 1    -- first hour on level ground
  let t2 := 0.5  -- next 0.5 hour on level ground
  let t3 := 0.75 -- 45 minutes uphill
  let t4 := 0.5  -- 30 minutes uphill
  let t5 := 0.5  -- 30 minutes downhill
  let t6 := 0.25 -- 15 minutes downhill
  let t7 := 1.5  -- 1.5 hours on level ground
  let t8 := 0.75 -- 45 minutes on level ground
  let s1 := 4    -- speed for t1 (4 km/hr)
  let s2 := 5    -- speed for t2 (5 km/hr)
  let s3 := 3    -- speed for t3 (3 km/hr)
  let s4 := 2    -- speed for t4 (2 km/hr)
  let s5 := 6    -- speed for t5 (6 km/hr)
  let s6 := 7    -- speed for t6 (7 km/hr)
  let s7 := 4    -- speed for t7 (4 km/hr)
  let s8 := 6    -- speed for t8 (6 km/hr)
  s1 * t1 + s2 * t2 + s3 * t3 + s4 * t4 + s5 * t5 + s6 * t6 + s7 * t7 + s8 * t8 = 25 :=
by sorry

end walking_total_distance_l16_16650


namespace solve_quartic_equation_l16_16532

theorem solve_quartic_equation (a b c : ℤ) (x : ℤ) : 
  x^4 + a * x^2 + b * x + c = 0 :=
sorry

end solve_quartic_equation_l16_16532


namespace simplify_expression_l16_16050

theorem simplify_expression (x : ℝ) : (3 * x)^5 + (5 * x) * (x^4) - 7 * x^5 = 241 * x^5 := 
by
  sorry

end simplify_expression_l16_16050


namespace function_defined_for_all_reals_l16_16177

theorem function_defined_for_all_reals (m : ℝ) :
  (∀ x : ℝ, 7 * x ^ 2 + m - 6 ≠ 0) → m > 6 :=
by
  sorry

end function_defined_for_all_reals_l16_16177


namespace profit_calculation_l16_16844

open Nat

-- Define the conditions 
def cost_of_actors : Nat := 1200 
def number_of_people : Nat := 50
def cost_per_person_food : Nat := 3
def sale_price : Nat := 10000

-- Define the derived costs
def total_food_cost : Nat := number_of_people * cost_per_person_food
def total_combined_cost : Nat := cost_of_actors + total_food_cost
def equipment_rental_cost : Nat := 2 * total_combined_cost
def total_cost : Nat := cost_of_actors + total_food_cost + equipment_rental_cost
def expected_profit : Nat := 5950 

-- Define the profit calculation
def profit : Nat := sale_price - total_cost 

-- The theorem to be proved
theorem profit_calculation : profit = expected_profit := by
  -- Proof is omitted
  sorry

end profit_calculation_l16_16844


namespace dance_team_recruits_l16_16752

theorem dance_team_recruits :
  ∃ (x : ℕ), x + 2 * x + (2 * x + 10) = 100 ∧ (2 * x + 10) = 46 :=
by
  sorry

end dance_team_recruits_l16_16752


namespace rectangle_dimensions_l16_16832

theorem rectangle_dimensions (a1 a2 : ℝ) (h1 : a1 * a2 = 216) (h2 : a1 + a2 = 30 - 6)
  (h3 : 6 * 6 = 36) : (a1 = 12 ∧ a2 = 18) ∨ (a1 = 18 ∧ a2 = 12) :=
by
  -- The conditions are set; now we need the proof, which we'll replace with sorry for now.
  sorry

end rectangle_dimensions_l16_16832


namespace value_of_a_plus_b_l16_16415

theorem value_of_a_plus_b :
  ∀ (a b x y : ℝ), x = 3 → y = -2 → 
  a * x + b * y = 2 → b * x + a * y = -3 → 
  a + b = -1 := 
by
  intros a b x y hx hy h1 h2
  subst hx
  subst hy
  sorry

end value_of_a_plus_b_l16_16415


namespace inequality_solution_set_minimum_value_expression_l16_16753

-- Definition of the function f
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Inequality solution set for f(x) ≤ 4
theorem inequality_solution_set :
  { x : ℝ | 0 ≤ x ∧ x ≤ 4 / 3 } = { x : ℝ | f x ≤ 4 } := 
sorry

-- Minimum value of the given expression given conditions on a and b
theorem minimum_value_expression (a b : ℝ) (h1 : a > 1) (h2 : b > 0)
  (h3 : a + 2 * b = 3) :
  (1 / (a - 1)) + (2 / b) = 9 / 2 := 
sorry

end inequality_solution_set_minimum_value_expression_l16_16753


namespace monic_polynomial_root_equivalence_l16_16719

noncomputable def roots (p : Polynomial ℝ) : List ℝ := sorry

theorem monic_polynomial_root_equivalence :
  let r1 := roots (Polynomial.C (8:ℝ) + Polynomial.X^3 - 3 * Polynomial.X^2)
  let p := Polynomial.C (216:ℝ) + Polynomial.X^3 - 9 * Polynomial.X^2
  r1.map (fun r => 3*r) = roots p :=
by
  sorry

end monic_polynomial_root_equivalence_l16_16719


namespace find_value_of_k_l16_16952

def line_equation_holds (m n : ℤ) : Prop := m = 2 * n + 5
def second_point_condition (m n k : ℤ) : Prop := m + 4 = 2 * (n + k) + 5

theorem find_value_of_k (m n k : ℤ) 
  (h1 : line_equation_holds m n) 
  (h2 : second_point_condition m n k) : 
  k = 2 :=
by sorry

end find_value_of_k_l16_16952


namespace inequality_proof_l16_16576

theorem inequality_proof (a b c d : ℝ) (h1 : b < a) (h2 : d < c) : a + c > b + d :=
  sorry

end inequality_proof_l16_16576


namespace three_exp_eq_l16_16332

theorem three_exp_eq (y : ℕ) (h : 3^y + 3^y + 3^y = 2187) : y = 6 :=
by
  sorry

end three_exp_eq_l16_16332


namespace sum_of_tens_and_ones_digits_pow_l16_16138

theorem sum_of_tens_and_ones_digits_pow : 
  let n := 7
  let exp := 12
  (n^exp % 100) / 10 + (n^exp % 10) = 1 :=
by
  sorry

end sum_of_tens_and_ones_digits_pow_l16_16138


namespace randy_final_amount_l16_16001

-- Conditions as definitions
def initial_dollars : ℝ := 30
def initial_euros : ℝ := 20
def lunch_cost : ℝ := 10
def ice_cream_percentage : ℝ := 0.25
def snack_percentage : ℝ := 0.10
def conversion_rate : ℝ := 0.85

-- Main proof statement without the proof body
theorem randy_final_amount :
  let euros_in_dollars := initial_euros / conversion_rate
  let total_dollars := initial_dollars + euros_in_dollars
  let dollars_after_lunch := total_dollars - lunch_cost
  let ice_cream_cost := dollars_after_lunch * ice_cream_percentage
  let dollars_after_ice_cream := dollars_after_lunch - ice_cream_cost
  let snack_euros := initial_euros * snack_percentage
  let snack_dollars := snack_euros / conversion_rate
  let final_dollars := dollars_after_ice_cream - snack_dollars
  final_dollars = 30.30 :=
by
  sorry

end randy_final_amount_l16_16001


namespace sum_of_first_six_terms_l16_16046

theorem sum_of_first_six_terms 
  (a₁ : ℝ) 
  (r : ℝ) 
  (h_ratio : r = 2) 
  (h_sum_three : a₁ + 2*a₁ + 4*a₁ = 3) 
  : a₁ * (r^6 - 1) / (r - 1) = 27 := 
by {
  sorry
}

end sum_of_first_six_terms_l16_16046


namespace square_side_length_l16_16451

theorem square_side_length (π : ℝ) (s : ℝ) :
  (∃ r : ℝ, 100 = π * r^2) ∧ (4 * s = 100) → s = 25 := by
  sorry

end square_side_length_l16_16451


namespace max_ski_trips_l16_16015

/--
The ski lift carries skiers from the bottom of the mountain to the top, taking 15 minutes each way, 
and it takes 5 minutes to ski back down the mountain. 
Given that the total available time is 2 hours, prove that the maximum number of trips 
down the mountain in that time is 6.
-/
theorem max_ski_trips (ride_up_time : ℕ) (ski_down_time : ℕ) (total_time : ℕ) :
  ride_up_time = 15 →
  ski_down_time = 5 →
  total_time = 120 →
  (total_time / (ride_up_time + ski_down_time) = 6) :=
by
  intros h1 h2 h3
  sorry

end max_ski_trips_l16_16015


namespace sequence_solution_l16_16498

theorem sequence_solution (a : ℕ → ℝ)
  (h₁ : a 1 = 0)
  (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 4 * (Real.sqrt (a n + 1)) + 4) :
  ∀ n ≥ 1, a n = 4 * n^2 - 4 * n :=
by
  sorry

end sequence_solution_l16_16498


namespace solve_for_a_l16_16919

theorem solve_for_a (a x : ℝ) (h : x = 3) (eqn : a * x - 5 = x + 1) : a = 3 :=
by
  -- proof omitted
  sorry

end solve_for_a_l16_16919


namespace find_n_l16_16316

def f (x n : ℝ) : ℝ := 2 * x^2 - 3 * x + n
def g (x n : ℝ) : ℝ := 2 * x^2 - 3 * x + 5 * n

theorem find_n (n : ℝ) (h : 3 * f 3 n = 2 * g 3 n) : n = 9 / 7 := by
  sorry

end find_n_l16_16316


namespace car_actual_speed_is_40_l16_16101

variable (v : ℝ) -- actual speed (we will prove it is 40 km/h)

-- Conditions
variable (hyp_speed : ℝ := v + 20) -- hypothetical speed
variable (distance : ℝ := 60) -- distance traveled
variable (time_difference : ℝ := 0.5) -- time difference in hours

-- Define the equation derived from the given conditions:
def speed_equation : Prop :=
  (distance / v) - (distance / hyp_speed) = time_difference

-- The theorem to prove:
theorem car_actual_speed_is_40 : speed_equation v → v = 40 :=
by
  sorry

end car_actual_speed_is_40_l16_16101


namespace total_number_of_people_l16_16392

-- Definitions corresponding to conditions
variables (A C : ℕ)
variables (cost_adult cost_child total_revenue : ℝ)
variables (ratio_child_adult : ℝ)

-- Assumptions given in the problem
axiom cost_adult_def : cost_adult = 7
axiom cost_child_def : cost_child = 3
axiom total_revenue_def : total_revenue = 6000
axiom ratio_def : C = 3 * A
axiom revenue_eq : total_revenue = cost_adult * A + cost_child * C

-- The main statement to prove
theorem total_number_of_people : A + C = 1500 :=
by
  sorry  -- Proof of the theorem

end total_number_of_people_l16_16392


namespace stacy_faster_than_heather_l16_16492

-- Definitions for the conditions
def distance : ℝ := 40
def heather_rate : ℝ := 5
def heather_distance : ℝ := 17.090909090909093
def heather_delay : ℝ := 0.4
def stacy_distance : ℝ := distance - heather_distance
def stacy_rate (S : ℝ) (T : ℝ) : Prop := S * T = stacy_distance
def heather_time (T : ℝ) : ℝ := T - heather_delay
def heather_walk_eq (T : ℝ) : Prop := heather_rate * heather_time T = heather_distance

-- The proof problem statement
theorem stacy_faster_than_heather :
  ∃ (S T : ℝ), stacy_rate S T ∧ heather_walk_eq T ∧ (S - heather_rate = 1) :=
by
  sorry

end stacy_faster_than_heather_l16_16492


namespace transport_cost_B_condition_l16_16779

-- Define the parameters for coal from Mine A
def calories_per_gram_A := 4
def price_per_ton_A := 20
def transport_cost_A := 8

-- Define the parameters for coal from Mine B
def calories_per_gram_B := 6
def price_per_ton_B := 24

-- Define the total cost for transporting one ton from Mine A to city N
def total_cost_A := price_per_ton_A + transport_cost_A

-- Define the question as a Lean theorem
theorem transport_cost_B_condition : 
  ∀ (transport_cost_B : ℝ), 
  (total_cost_A : ℝ) / (calories_per_gram_A : ℝ) = (price_per_ton_B + transport_cost_B) / (calories_per_gram_B : ℝ) → 
  transport_cost_B = 18 :=
by
  intros transport_cost_B h
  have h_eq : (total_cost_A : ℝ) / (calories_per_gram_A : ℝ) = (price_per_ton_B + transport_cost_B) / (calories_per_gram_B : ℝ) := h
  sorry

end transport_cost_B_condition_l16_16779


namespace distinct_pairs_count_l16_16045

theorem distinct_pairs_count :
  ∃ (n : ℕ), n = 2 ∧ 
  ∀ (x y : ℝ), (x = 3 * x^2 + y^2) ∧ (y = 3 * x * y) → 
    ((x = 0 ∧ y = 0) ∨ (x = 1 / 3 ∧ y = 0)) :=
by
  sorry

end distinct_pairs_count_l16_16045


namespace polynomial_coeffs_sum_l16_16968

theorem polynomial_coeffs_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) (x : ℝ) :
  (2*x - 3)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + 2*a_2 + 3*a_3 + 4*a_4 + 5*a_5 = 10 :=
by
  sorry

end polynomial_coeffs_sum_l16_16968


namespace interchanged_digits_subtraction_l16_16445

theorem interchanged_digits_subtraction (a b k : ℤ) (h1 : 10 * a + b = 2 * k * (a + b)) :
  10 * b + a - 3 * (a + b) = (9 - 4 * k) * (a + b) :=
by sorry

end interchanged_digits_subtraction_l16_16445


namespace greatest_distance_P_D_l16_16807

noncomputable def greatest_distance_from_D (P : ℝ × ℝ) (A B C : ℝ × ℝ) (D : ℝ × ℝ) : ℝ :=
  let u := (P.1 - A.1)^2 + (P.2 - A.2)^2
  let v := (P.1 - B.1)^2 + (P.2 - B.2)^2
  let w := (P.1 - C.1)^2 + (P.2 - C.2)^2
  if u + v = w + 1 then ((P.1 - D.1)^2 + (P.2 - D.2)^2).sqrt else 0

theorem greatest_distance_P_D (P : ℝ × ℝ) (u v w : ℝ)
  (h1 : u^2 + v^2 = w^2 + 1) :
  greatest_distance_from_D P (0,0) (2,0) (2,2) (0,2) = 5 :=
sorry

end greatest_distance_P_D_l16_16807


namespace households_selected_l16_16389

theorem households_selected (H : ℕ) (M L S n h : ℕ)
  (h1 : H = 480)
  (h2 : M = 200)
  (h3 : L = 160)
  (h4 : H = M + L + S)
  (h5 : h = 6)
  (h6 : (h : ℚ) / n = (S : ℚ) / H) : n = 24 :=
by
  sorry

end households_selected_l16_16389


namespace yellow_balls_count_l16_16403

theorem yellow_balls_count {R B Y G : ℕ} 
  (h1 : R + B + Y + G = 531)
  (h2 : R + B = Y + G + 31)
  (h3 : Y = G + 22) : 
  Y = 136 :=
by
  -- The proof is skipped, as requested.
  sorry

end yellow_balls_count_l16_16403


namespace number_of_proper_subsets_of_P_l16_16376

theorem number_of_proper_subsets_of_P (P : Set ℝ) (hP : P = {x | x^2 = 1}) : 
  (∃ n, n = 2 ∧ ∃ k, k = 2 ^ n - 1 ∧ k = 3) :=
by
  sorry

end number_of_proper_subsets_of_P_l16_16376


namespace first_number_in_a10_l16_16887

-- Define a function that captures the sequence of the first number in each sum 'a_n'.
def first_in_an (n : ℕ) : ℕ :=
  1 + 2 * (n * (n - 1)) / 2 

-- State the theorem we want to prove
theorem first_number_in_a10 : first_in_an 10 = 91 := 
  sorry

end first_number_in_a10_l16_16887


namespace remainder_98_mul_102_div_11_l16_16590

theorem remainder_98_mul_102_div_11 : (98 * 102) % 11 = 7 := by
  sorry

end remainder_98_mul_102_div_11_l16_16590


namespace possible_measure_of_angle_AOC_l16_16569

-- Given conditions
def angle_AOB : ℝ := 120
def OC_bisects_angle_AOB (x : ℝ) : Prop := x = 60
def OD_bisects_angle_AOB_and_OC_bisects_angle (x y : ℝ) : Prop :=
  (y = 60 ∧ (x = 30 ∨ x = 90))

-- Theorem statement
theorem possible_measure_of_angle_AOC (angle_AOC : ℝ) :
  (OC_bisects_angle_AOB angle_AOC ∨ 
  (OD_bisects_angle_AOB_and_OC_bisects_angle angle_AOC 60)) →
  (angle_AOC = 30 ∨ angle_AOC = 60 ∨ angle_AOC = 90) :=
by
  sorry

end possible_measure_of_angle_AOC_l16_16569


namespace solve_trigonometric_eqn_l16_16487

theorem solve_trigonometric_eqn (x : ℝ) : 
  (∃ k : ℤ, x = 3 * (π / 4 * (4 * k + 1))) ∨ (∃ n : ℤ, x = π * (3 * n + 1) ∨ x = π * (3 * n - 1)) :=
by 
  sorry

end solve_trigonometric_eqn_l16_16487


namespace zero_extreme_points_l16_16414

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x

theorem zero_extreme_points : ∀ x : ℝ, 
  ∃! (y : ℝ), deriv f y = 0 → y = x :=
by
  sorry

end zero_extreme_points_l16_16414


namespace hamsters_count_l16_16024

-- Define the conditions as parameters
variables (ratio_rabbit_hamster : ℕ × ℕ)
variables (rabbits : ℕ)
variables (hamsters : ℕ)

-- Given conditions
def ratio_condition : ratio_rabbit_hamster = (4, 5) := sorry
def rabbits_condition : rabbits = 20 := sorry

-- The theorem to be proven
theorem hamsters_count : ratio_rabbit_hamster = (4, 5) -> rabbits = 20 -> hamsters = 25 :=
by
  intro h1 h2
  sorry

end hamsters_count_l16_16024


namespace thomas_lost_pieces_l16_16058

theorem thomas_lost_pieces (audrey_lost : ℕ) (total_pieces_left : ℕ) (initial_pieces_each : ℕ) (total_pieces_initial : ℕ) (audrey_remaining_pieces : ℕ) (thomas_remaining_pieces : ℕ) : 
  audrey_lost = 6 → total_pieces_left = 21 → initial_pieces_each = 16 → total_pieces_initial = 32 → 
  audrey_remaining_pieces = initial_pieces_each - audrey_lost → 
  thomas_remaining_pieces = total_pieces_left - audrey_remaining_pieces → 
  initial_pieces_each - thomas_remaining_pieces = 5 :=
by
  sorry

end thomas_lost_pieces_l16_16058


namespace cos_double_angle_of_parallel_vectors_l16_16599

theorem cos_double_angle_of_parallel_vectors
  (α : ℝ)
  (a : ℝ × ℝ := (1/3, Real.tan α))
  (b : ℝ × ℝ := (Real.cos α, 2))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  Real.cos (2 * α) = 1 / 9 := 
sorry

end cos_double_angle_of_parallel_vectors_l16_16599


namespace matrix_power_A_2023_l16_16892

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![0, -1, 0],
    ![1, 0, 0],
    ![0, 0, 1]
  ]

theorem matrix_power_A_2023 :
  A ^ 2023 = ![
    ![0, 1, 0],
    ![-1, 0, 0],
    ![0, 0, 1]
  ] :=
sorry

end matrix_power_A_2023_l16_16892


namespace sugar_flour_ratio_10_l16_16583

noncomputable def sugar_to_flour_ratio (sugar flour : ℕ) : ℕ :=
  sugar / flour

theorem sugar_flour_ratio_10 (sugar flour : ℕ) (hs : sugar = 50) (hf : flour = 5) : sugar_to_flour_ratio sugar flour = 10 :=
by
  rw [hs, hf]
  unfold sugar_to_flour_ratio
  norm_num
  -- sorry

end sugar_flour_ratio_10_l16_16583


namespace tan_sum_identity_l16_16916

theorem tan_sum_identity :
  Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180) + 
  Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180) = 1 :=
by sorry

end tan_sum_identity_l16_16916


namespace find_base_b_l16_16199

theorem find_base_b : ∃ b : ℕ, b > 4 ∧ (b + 2)^2 = b^2 + 4 * b + 4 ∧ b = 5 := 
sorry

end find_base_b_l16_16199


namespace find_a_l16_16512

def A : Set ℤ := {-1, 1, 3}
def B (a : ℤ) : Set ℤ := {a + 1, a^2 + 4}
def intersection (a : ℤ) : Set ℤ := A ∩ B a

theorem find_a : ∃ a : ℤ, intersection a = {3} ∧ a = 2 :=
by
  sorry

end find_a_l16_16512


namespace no_such_integers_x_y_l16_16173

theorem no_such_integers_x_y (x y : ℤ) : x^2 + 1974 ≠ y^2 := by
  sorry

end no_such_integers_x_y_l16_16173


namespace tangent_eq_tangent_intersect_other_l16_16627

noncomputable def curve (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 - 9 * x^2 + 4

/-- Equation of the tangent line to curve C at x = 1 is y = -12x + 8 --/
theorem tangent_eq (tangent_line : ℝ → ℝ) (x : ℝ):
  tangent_line x = -12 * x + 8 :=
by
  sorry

/-- Apart from the tangent point (1, -4), the tangent line intersects the curve C at the points
    (-2, 32) and (2 / 3, 0) --/
theorem tangent_intersect_other (tangent_line : ℝ → ℝ) x:
  curve x = tangent_line x →
  (x = -2 ∧ curve (-2) = 32) ∨ (x = 2 / 3 ∧ curve (2 / 3) = 0) :=
by
  sorry

end tangent_eq_tangent_intersect_other_l16_16627


namespace arithmetic_seq_a1_l16_16610

theorem arithmetic_seq_a1 (a_1 d : ℝ) (h1 : a_1 + 4 * d = 9) (h2 : 2 * (a_1 + 2 * d) = (a_1 + d) + 6) : a_1 = -3 := by
  sorry

end arithmetic_seq_a1_l16_16610


namespace product_of_invertible_function_labels_l16_16826

noncomputable def Function6 (x : ℝ) : ℝ := x^3 - 3 * x
def points7 : List (ℝ × ℝ) := [(-6, 3), (-5, 1), (-4, 2), (-3, -1), (-2, 0), (-1, -2), (0, 4), (1, 5)]
noncomputable def Function8 (x : ℝ) : ℝ := Real.sin x
noncomputable def Function9 (x : ℝ) : ℝ := 3 / x

def is_invertible6 : Prop := ¬ ∃ (y : ℝ), ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ Function6 x1 = y ∧ Function6 x2 = y ∧ (-2 ≤ x1 ∧ x1 ≤ 2) ∧ (-2 ≤ x2 ∧ x2 ≤ 2)
def is_invertible7 : Prop := ∀ (y : ℝ), ∃! x : ℝ, (x, y) ∈ points7
def is_invertible8 : Prop := ∀ (x1 x2 : ℝ), Function8 x1 = Function8 x2 → x1 = x2 ∧ (-Real.pi/2 ≤ x1 ∧ x1 ≤ Real.pi/2) ∧ (-Real.pi/2 ≤ x2 ∧ x2 ≤ Real.pi/2)
def is_invertible9 : Prop := ¬ ∃ (y : ℝ), ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ Function9 x1 = y ∧ Function9 x2 = y ∧ (-4 ≤ x1 ∧ x1 ≤ 4 ∧ x1 ≠ 0) ∧ (-4 ≤ x2 ∧ x2 ≤ 4 ∧ x2 ≠ 0)

theorem product_of_invertible_function_labels :
  (is_invertible6 = false) →
  (is_invertible7 = true) →
  (is_invertible8 = true) →
  (is_invertible9 = true) →
  7 * 8 * 9 = 504
:= by
  intros h6 h7 h8 h9
  sorry

end product_of_invertible_function_labels_l16_16826


namespace team_average_typing_speed_l16_16115

-- Definitions of typing speeds of each team member
def typing_speed_rudy := 64
def typing_speed_joyce := 76
def typing_speed_gladys := 91
def typing_speed_lisa := 80
def typing_speed_mike := 89

-- Number of team members
def number_of_team_members := 5

-- Total typing speed calculation
def total_typing_speed := typing_speed_rudy + typing_speed_joyce + typing_speed_gladys + typing_speed_lisa + typing_speed_mike

-- Average typing speed calculation
def average_typing_speed := total_typing_speed / number_of_team_members

-- Theorem statement
theorem team_average_typing_speed : average_typing_speed = 80 := by
  sorry

end team_average_typing_speed_l16_16115


namespace abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0_l16_16367

theorem abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0 :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ (¬ ∀ x : ℝ, x^2 - x - 6 < 0 → |x| < 2) :=
by
  sorry

end abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0_l16_16367


namespace naomi_total_wheels_l16_16851

theorem naomi_total_wheels 
  (regular_bikes : ℕ) (children_bikes : ℕ) (tandem_bikes_4_wheels : ℕ) (tandem_bikes_6_wheels : ℕ)
  (wheels_per_regular_bike : ℕ) (wheels_per_children_bike : ℕ) (wheels_per_tandem_4wheel : ℕ) (wheels_per_tandem_6wheel : ℕ) :
  regular_bikes = 7 →
  children_bikes = 11 →
  tandem_bikes_4_wheels = 5 →
  tandem_bikes_6_wheels = 3 →
  wheels_per_regular_bike = 2 →
  wheels_per_children_bike = 4 →
  wheels_per_tandem_4wheel = 4 →
  wheels_per_tandem_6wheel = 6 →
  (regular_bikes * wheels_per_regular_bike) + 
  (children_bikes * wheels_per_children_bike) + 
  (tandem_bikes_4_wheels * wheels_per_tandem_4wheel) + 
  (tandem_bikes_6_wheels * wheels_per_tandem_6wheel) = 96 := 
by
  intros; sorry

end naomi_total_wheels_l16_16851


namespace juan_more_marbles_l16_16082

theorem juan_more_marbles (connie_marbles : ℕ) (juan_marbles : ℕ) (h1 : connie_marbles = 323) (h2 : juan_marbles = 498) :
  juan_marbles - connie_marbles = 175 :=
by
  -- Proof goes here
  sorry

end juan_more_marbles_l16_16082


namespace number_of_buses_proof_l16_16205

-- Define the conditions
def columns_per_bus : ℕ := 4
def rows_per_bus : ℕ := 10
def total_students : ℕ := 240
def seats_per_bus (c : ℕ) (r : ℕ) : ℕ := c * r
def number_of_buses (total : ℕ) (seats : ℕ) : ℕ := total / seats

-- State the theorem we want to prove
theorem number_of_buses_proof :
  number_of_buses total_students (seats_per_bus columns_per_bus rows_per_bus) = 6 := 
sorry

end number_of_buses_proof_l16_16205


namespace plane_triangle_coverage_l16_16450

noncomputable def percentage_triangles_covered (a : ℝ) : ℝ :=
  let total_area := (4 * a) ^ 2
  let triangle_area := 10 * (1 / 2 * a^2)
  (triangle_area / total_area) * 100

theorem plane_triangle_coverage (a : ℝ) :
  abs (percentage_triangles_covered a - 31.25) < 0.75 :=
  sorry

end plane_triangle_coverage_l16_16450


namespace probability_of_real_roots_is_correct_l16_16958

open Real

def has_real_roots (m : ℝ) : Prop :=
  2 * m^2 - 8 ≥ 0 

def favorable_set : Set ℝ := {m | has_real_roots m}

def interval_length (a b : ℝ) : ℝ := b - a

noncomputable def probability_of_real_roots : ℝ :=
  interval_length (-4) (-2) + interval_length 2 3 / interval_length (-4) 3

theorem probability_of_real_roots_is_correct : probability_of_real_roots = 3 / 7 :=
by
  sorry

end probability_of_real_roots_is_correct_l16_16958


namespace successive_numbers_product_2652_l16_16176

theorem successive_numbers_product_2652 (n : ℕ) (h : n * (n + 1) = 2652) : n = 51 :=
sorry

end successive_numbers_product_2652_l16_16176


namespace binomial_sum_equal_36_l16_16188

theorem binomial_sum_equal_36 (n : ℕ) (h : n > 0) :
  (n + n * (n - 1) / 2 = 36) → n = 8 :=
by
  sorry

end binomial_sum_equal_36_l16_16188


namespace fraction_of_l16_16688

theorem fraction_of (a b : ℚ) (h_a : a = 3/4) (h_b : b = 1/6) : b / a = 2/9 :=
by
  sorry

end fraction_of_l16_16688


namespace teresa_speed_l16_16118

def distance : ℝ := 25 -- kilometers
def time : ℝ := 5 -- hours

theorem teresa_speed :
  (distance / time) = 5 := by
  sorry

end teresa_speed_l16_16118


namespace y_star_definition_l16_16260

def y_star (y : Real) : Real := y - 1

theorem y_star_definition (y : Real) : (5 : Real) - y_star 5 = 1 :=
  by sorry

end y_star_definition_l16_16260


namespace quadratic_translation_l16_16282

theorem quadratic_translation (b c : ℝ) :
  (∀ x : ℝ, (x^2 + b * x + c = (x - 3)^2 - 2)) →
  b = 4 ∧ c = 6 :=
by
  sorry

end quadratic_translation_l16_16282


namespace reciprocal_of_fraction_diff_l16_16831

theorem reciprocal_of_fraction_diff : 
  (∃ (a b : ℚ), a = 1/4 ∧ b = 1/5 ∧ (1 / (a - b)) = 20) :=
sorry

end reciprocal_of_fraction_diff_l16_16831


namespace circle_ring_ratio_l16_16606

theorem circle_ring_ratio
  (r R c d : ℝ)
  (hr : 0 < r)
  (hR : 0 < R)
  (hc : 0 < c)
  (hd : 0 < d)
  (h_areas : π * R^2 = (c / d) * (π * R^2 - π * r^2)) :
  R / r = (Real.sqrt c) / (Real.sqrt (d - c)) := 
by 
  sorry

end circle_ring_ratio_l16_16606


namespace find_angle_MBA_l16_16505

-- Define the angles and the triangle
def triangle (A B C : Type) := true

-- Define the angles in degrees
def angle (deg : ℝ) := deg

-- Assume angles' degrees as given in the problem
variables {A B C M : Type}
variable {BAC ABC MAB MCA MBA : ℝ}

-- Given conditions
axiom angle_BAC : angle BAC = 30
axiom angle_ABC : angle ABC = 70
axiom angle_MAB : angle MAB = 20
axiom angle_MCA : angle MCA = 20

-- Prove that angle MBA is 30 degrees
theorem find_angle_MBA : angle MBA = 30 := 
by 
  sorry

end find_angle_MBA_l16_16505


namespace number_of_bricks_is_1800_l16_16706

-- Define the conditions
def rate_first_bricklayer (x : ℕ) : ℕ := x / 8
def rate_second_bricklayer (x : ℕ) : ℕ := x / 12
def combined_reduced_rate (x : ℕ) : ℕ := (rate_first_bricklayer x + rate_second_bricklayer x - 15)

-- Prove that the number of bricks in the wall is 1800
theorem number_of_bricks_is_1800 :
  ∃ x : ℕ, 5 * combined_reduced_rate x = x ∧ x = 1800 :=
by
  use 1800
  sorry

end number_of_bricks_is_1800_l16_16706


namespace evaluate_expression_l16_16137

theorem evaluate_expression : (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))) = (8 / 21) :=
by
  sorry

end evaluate_expression_l16_16137


namespace value_of_5a_l16_16635

variable (a : ℕ)

theorem value_of_5a (h : 5 * (a - 3) = 25) : 5 * a = 40 :=
sorry

end value_of_5a_l16_16635


namespace eggs_at_park_l16_16263

-- Define the number of eggs found at different locations
def eggs_at_club_house : Nat := 40
def eggs_at_town_hall : Nat := 15
def total_eggs_found : Nat := 80

-- Prove that the number of eggs found at the park is 25
theorem eggs_at_park :
  ∃ P : Nat, eggs_at_club_house + P + eggs_at_town_hall = total_eggs_found ∧ P = 25 := 
by
  sorry

end eggs_at_park_l16_16263


namespace find_greater_number_l16_16383

-- Define the two numbers x and y
variables (x y : ℕ)

-- Conditions
theorem find_greater_number (h1 : x + y = 36) (h2 : x - y = 12) : x = 24 := 
by
  sorry

end find_greater_number_l16_16383


namespace infinite_sqrt_eval_l16_16525

theorem infinite_sqrt_eval {x : ℝ} (h : x = Real.sqrt (3 - x)) : 
  x = (-1 + Real.sqrt 13) / 2 :=
by sorry

end infinite_sqrt_eval_l16_16525


namespace fraction_calculation_l16_16857

theorem fraction_calculation : 
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 :=
by sorry

end fraction_calculation_l16_16857


namespace three_digit_numbers_with_square_ending_in_them_l16_16822

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

theorem three_digit_numbers_with_square_ending_in_them (A : ℕ) :
  is_three_digit A → (A^2 % 1000 = A) → A = 376 ∨ A = 625 :=
by
  sorry

end three_digit_numbers_with_square_ending_in_them_l16_16822


namespace find_a_l16_16210

theorem find_a (a : ℝ) (A B : Set ℝ)
    (hA : A = {a^2, a + 1, -3})
    (hB : B = {a - 3, 2 * a - 1, a^2 + 1}) 
    (h : A ∩ B = {-3}) : a = -1 := by
  sorry

end find_a_l16_16210


namespace anya_initial_seat_l16_16310

theorem anya_initial_seat (V G D E A : ℕ) (A' : ℕ) 
  (h1 : V + G + D + E + A = 15)
  (h2 : V + 1 ≠ A')
  (h3 : G - 3 ≠ A')
  (h4 : (D = A' → E ≠ A') ∧ (E = A' → D ≠ A'))
  (h5 : A = 3 + 2)
  : A = 3 := by
  sorry

end anya_initial_seat_l16_16310


namespace find_x_l16_16604

def operation (x y : ℕ) : ℕ := 2 * x * y

theorem find_x : 
  (operation 4 5 = 40) ∧ (operation x 40 = 480) → x = 6 :=
by
  sorry

end find_x_l16_16604


namespace bowling_average_l16_16795

theorem bowling_average (gretchen_score mitzi_score beth_score : ℤ) (h1 : gretchen_score = 120) (h2 : mitzi_score = 113) (h3 : beth_score = 85) :
  (gretchen_score + mitzi_score + beth_score) / 3 = 106 :=
by
  sorry

end bowling_average_l16_16795


namespace cone_to_sphere_ratio_l16_16061

-- Prove the ratio of the cone's altitude to its base radius
theorem cone_to_sphere_ratio (r h : ℝ) (h_r_pos : 0 < r) 
  (vol_cone : ℝ) (vol_sphere : ℝ) 
  (hyp_vol_relation : vol_cone = (1 / 3) * vol_sphere)
  (vol_sphere_def : vol_sphere = (4 / 3) * π * r^3)
  (vol_cone_def : vol_cone = (1 / 3) * π * r^2 * h) :
  h / r = 4 / 3 :=
by
  sorry

end cone_to_sphere_ratio_l16_16061


namespace joelle_initial_deposit_l16_16529

-- Definitions for the conditions
def annualInterestRate : ℝ := 0.05
def initialTimePeriod : ℕ := 2 -- in years
def numberOfCompoundsPerYear : ℕ := 1
def finalAmount : ℝ := 6615

-- Compound interest formula: A = P(1 + r/n)^(nt)
noncomputable def initialDeposit : ℝ :=
  finalAmount / ((1 + annualInterestRate / numberOfCompoundsPerYear)^(numberOfCompoundsPerYear * initialTimePeriod))

-- Theorem statement to prove the initial deposit
theorem joelle_initial_deposit : initialDeposit = 6000 := 
  sorry

end joelle_initial_deposit_l16_16529


namespace find_m_l16_16462

def circle1 (x y m : ℝ) : Prop := (x + 2)^2 + (y - m)^2 = 9
def circle2 (x y m : ℝ) : Prop := (x - m)^2 + (y + 1)^2 = 4

theorem find_m (m : ℝ) : 
  ∃ x1 y1 x2 y2 : ℝ, 
    circle1 x1 y1 m ∧ 
    circle2 x2 y2 m ∧ 
    (m + 2)^2 + (-1 - m)^2 = 25 → 
    m = 2 :=
by
  sorry

end find_m_l16_16462


namespace percent_alcohol_in_new_solution_l16_16315

theorem percent_alcohol_in_new_solution (orig_vol : ℝ) (orig_percent : ℝ) (add_alc : ℝ) (add_water : ℝ) :
  orig_percent = 5 → orig_vol = 40 → add_alc = 5.5 → add_water = 4.5 →
  (((orig_vol * (orig_percent / 100) + add_alc) / (orig_vol + add_alc + add_water)) * 100) = 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end percent_alcohol_in_new_solution_l16_16315


namespace import_tax_percentage_l16_16869

theorem import_tax_percentage
  (total_value : ℝ)
  (non_taxable_portion : ℝ)
  (import_tax_paid : ℝ)
  (h_total_value : total_value = 2610)
  (h_non_taxable_portion : non_taxable_portion = 1000)
  (h_import_tax_paid : import_tax_paid = 112.70) :
  ((import_tax_paid / (total_value - non_taxable_portion)) * 100) = 7 :=
by
  sorry

end import_tax_percentage_l16_16869


namespace sqrt_4_eq_2_or_neg2_l16_16728

theorem sqrt_4_eq_2_or_neg2 (y : ℝ) (h : y^2 = 4) : y = 2 ∨ y = -2 :=
sorry

end sqrt_4_eq_2_or_neg2_l16_16728


namespace undefined_expression_values_l16_16493

theorem undefined_expression_values : 
    ∃ x : ℝ, x^2 - 9 = 0 ↔ (x = -3 ∨ x = 3) :=
by
  sorry

end undefined_expression_values_l16_16493


namespace expression_simplification_l16_16128

-- Definitions for P and Q based on x and y
def P (x y : ℝ) := x + y
def Q (x y : ℝ) := x - y

-- The mathematical property to prove
theorem expression_simplification (x y : ℝ) (h : x ≠ 0) (k : y ≠ 0) : 
  (P x y + Q x y) / (P x y - Q x y) - (P x y - Q x y) / (P x y + Q x y) = (x^2 - y^2) / (x * y) := 
by
  -- Sorry is used to skip the proof here
  sorry

end expression_simplification_l16_16128


namespace range_of_a_l16_16390

theorem range_of_a (a : ℝ) : |a - 1| + |a - 4| = 3 ↔ 1 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l16_16390


namespace angle_AOD_128_57_l16_16796

-- Define angles as real numbers
variables {α β : ℝ}

-- Define the conditions
def perp (v1 v2 : ℝ) := v1 = 90 - v2

theorem angle_AOD_128_57 
  (h1 : perp α 90)
  (h2 : perp β 90)
  (h3 : α = 2.5 * β) :
  α = 128.57 :=
by
  -- Proof would go here
  sorry

end angle_AOD_128_57_l16_16796


namespace least_number_of_marbles_divisible_l16_16073

theorem least_number_of_marbles_divisible (n : ℕ) : 
  (∀ k ∈ [2, 3, 4, 5, 6, 7, 8], n % k = 0) -> n >= 840 :=
by sorry

end least_number_of_marbles_divisible_l16_16073


namespace rowing_velocity_l16_16320

theorem rowing_velocity (v : ℝ) : 
  (∀ (d : ℝ) (s : ℝ) (total_time : ℝ), 
    s = 10 ∧ 
    total_time = 30 ∧ 
    d = 144 ∧ 
    (d / (s - v) + d / (s + v)) = total_time) → 
  v = 2 := 
by
  sorry

end rowing_velocity_l16_16320


namespace cheap_feed_amount_l16_16028

theorem cheap_feed_amount (x y : ℝ) (h1 : x + y = 27) (h2 : 0.17 * x + 0.36 * y = 7.02) : 
  x = 14.21 :=
sorry

end cheap_feed_amount_l16_16028


namespace ratio_of_areas_l16_16705

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  -- The problem is to prove the ratio of the areas is 4/9
  sorry

end ratio_of_areas_l16_16705


namespace smaller_angle_is_70_l16_16426

def measure_of_smaller_angle (x : ℕ) : Prop :=
  (x + (x + 40) = 180) ∧ (2 * x - 60 = 80)

theorem smaller_angle_is_70 {x : ℕ} : measure_of_smaller_angle x → x = 70 :=
by
  sorry

end smaller_angle_is_70_l16_16426


namespace range_of_a_l16_16113

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 4 * x + a - 3 > 0) ↔ (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l16_16113


namespace distinct_sequences_ten_flips_l16_16843

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l16_16843


namespace students_more_than_rabbits_l16_16668

-- Definitions of conditions
def classrooms : ℕ := 5
def students_per_classroom : ℕ := 22
def rabbits_per_classroom : ℕ := 2

-- Statement of the theorem
theorem students_more_than_rabbits :
  classrooms * students_per_classroom - classrooms * rabbits_per_classroom = 100 := 
  by
    sorry

end students_more_than_rabbits_l16_16668


namespace min_chips_to_color_all_cells_l16_16651

def min_chips_needed (n : ℕ) : ℕ := n

theorem min_chips_to_color_all_cells (n : ℕ) :
  min_chips_needed n = n :=
sorry

end min_chips_to_color_all_cells_l16_16651


namespace power_of_product_l16_16817

theorem power_of_product (x : ℝ) : (-x^4)^3 = -x^12 := 
by sorry

end power_of_product_l16_16817


namespace number_of_sheep_l16_16543

-- Define the conditions as given in the problem
variables (S H : ℕ)
axiom ratio_condition : S * 7 = H * 3
axiom food_condition : H * 230 = 12880

-- The theorem to prove
theorem number_of_sheep : S = 24 :=
by sorry

end number_of_sheep_l16_16543


namespace simplify_expression_l16_16257

theorem simplify_expression : 
  (((5 + 7 + 3) * 2 - 4) / 2 - (5 / 2) = 21 / 2) :=
by
  sorry

end simplify_expression_l16_16257


namespace factorize_expression_l16_16288

-- The problem is about factorizing the expression x^3y - xy
theorem factorize_expression (x y : ℝ) : x^3 * y - x * y = x * y * (x - 1) * (x + 1) := 
by sorry

end factorize_expression_l16_16288


namespace find_inverse_value_l16_16351

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic (x : ℝ) : f (x - 1) = f (x + 3)
axiom defined_interval (x : ℝ) (h : 4 ≤ x ∧ x ≤ 5) : f x = 2 ^ x + 1

noncomputable def f_inv : ℝ → ℝ := sorry
axiom inverse_defined (x : ℝ) (h : -2 ≤ x ∧ x ≤ 0) : f (f_inv x) = x

theorem find_inverse_value : f_inv 19 = 3 - 2 * (Real.log 3 / Real.log 2) := by
  sorry

end find_inverse_value_l16_16351


namespace solve_expression_l16_16876

noncomputable def given_expression : ℝ :=
  (10 * 1.8 - 2 * 1.5) / 0.3 + 3^(2 / 3) - Real.log 4 + Real.sin (Real.pi / 6) - Real.cos (Real.pi / 4) + Nat.factorial 4 / Nat.factorial 2

theorem solve_expression : given_expression = 59.6862 :=
by
  sorry

end solve_expression_l16_16876


namespace greatest_possible_avg_speed_l16_16840

theorem greatest_possible_avg_speed (initial_odometer : ℕ) (max_speed : ℕ) (time_hours : ℕ) (max_distance : ℕ) (target_palindrome : ℕ) :
  initial_odometer = 12321 →
  max_speed = 80 →
  time_hours = 4 →
  (target_palindrome = 12421 ∨ target_palindrome = 12521 ∨ target_palindrome = 12621 ∨ target_palindrome = 12721 ∨ target_palindrome = 12821 ∨ target_palindrome = 12921 ∨ target_palindrome = 13031) →
  target_palindrome - initial_odometer ≤ max_distance →
  max_distance = 300 →
  target_palindrome = 12621 →
  time_hours = 4 →
  target_palindrome - initial_odometer = 300 →
  (target_palindrome - initial_odometer) / time_hours = 75 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end greatest_possible_avg_speed_l16_16840


namespace investment_value_l16_16469

-- Define the compound interest calculation
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Given values
def P : ℝ := 8000
def r : ℝ := 0.05
def n : ℕ := 7

-- The theorem statement in Lean 4
theorem investment_value :
  round (compound_interest P r n) = 11257 :=
by
  sorry

end investment_value_l16_16469


namespace train_speed_incl_stoppages_l16_16443

theorem train_speed_incl_stoppages
  (speed_excl_stoppages : ℝ)
  (stoppage_time_minutes : ℝ)
  (h1 : speed_excl_stoppages = 42)
  (h2 : stoppage_time_minutes = 21.428571428571423)
  : ∃ speed_incl_stoppages, speed_incl_stoppages = 27 := 
sorry

end train_speed_incl_stoppages_l16_16443


namespace circumcircle_eqn_l16_16700

variables (D E F : ℝ)

def point_A := (4, 0)
def point_B := (0, 3)
def point_C := (0, 0)

-- Define the system of equations for the circumcircle
def system : Prop :=
  (16 + 4*D + F = 0) ∧
  (9 + 3*E + F = 0) ∧
  (F = 0)

theorem circumcircle_eqn : system D E F → (D = -4 ∧ E = -3 ∧ F = 0) :=
sorry -- Proof omitted

end circumcircle_eqn_l16_16700


namespace length_AB_l16_16514

theorem length_AB :
  ∀ (A B : ℝ × ℝ) (k : ℝ),
    (A.2 = k * A.1 - 2) ∧ (B.2 = k * B.1 - 2) ∧ (A.2^2 = 8 * A.1) ∧ (B.2^2 = 8 * B.1) ∧
    ((A.1 + B.1) / 2 = 2) →
  dist A B = 2 * Real.sqrt 15 :=
by
  sorry

end length_AB_l16_16514


namespace actual_height_of_boy_is_236_l16_16904

-- Define the problem conditions
def average_height (n : ℕ) (avg : ℕ) := n * avg
def incorrect_total_height := average_height 35 180
def correct_total_height := average_height 35 178
def wrong_height := 166
def height_difference := incorrect_total_height - correct_total_height

-- Proving the actual height of the boy whose height was wrongly written
theorem actual_height_of_boy_is_236 : 
  wrong_height + height_difference = 236 := sorry

end actual_height_of_boy_is_236_l16_16904


namespace domain_of_f_l16_16789

noncomputable def f (x : ℝ) := 1 / ((x - 3) + (x - 6))

theorem domain_of_f :
  (∀ x : ℝ, x ≠ 9/2 → ∃ y : ℝ, f x = y) ∧ (∀ x : ℝ, x = 9/2 → ¬ (∃ y : ℝ, f x = y)) :=
by
  sorry

end domain_of_f_l16_16789


namespace speed_of_boat_in_still_water_l16_16362

variable (b s : ℝ) -- Speed of the boat in still water and speed of the stream

-- Condition 1: The boat goes 9 km along the stream in 1 hour
def boat_along_stream := b + s = 9

-- Condition 2: The boat goes 5 km against the stream in 1 hour
def boat_against_stream := b - s = 5

-- Theorem to prove: The speed of the boat in still water is 7 km/hr
theorem speed_of_boat_in_still_water : boat_along_stream b s → boat_against_stream b s → b = 7 := 
by
  sorry

end speed_of_boat_in_still_water_l16_16362


namespace root_of_quadratic_gives_value_l16_16628

theorem root_of_quadratic_gives_value (a : ℝ) (h : a^2 + 3 * a - 5 = 0) : a^2 + 3 * a + 2021 = 2026 :=
by {
  -- We will skip the proof here.
  sorry
}

end root_of_quadratic_gives_value_l16_16628


namespace calculate_selling_price_l16_16477

-- Define the conditions
def purchase_price : ℝ := 900
def repair_cost : ℝ := 300
def gain_percentage : ℝ := 0.10

-- Define the total cost
def total_cost : ℝ := purchase_price + repair_cost

-- Define the gain
def gain : ℝ := gain_percentage * total_cost

-- Define the selling price
def selling_price : ℝ := total_cost + gain

-- The theorem to prove
theorem calculate_selling_price : selling_price = 1320 := by
  sorry

end calculate_selling_price_l16_16477


namespace base_number_is_2_l16_16435

open Real

noncomputable def valid_x (x : ℝ) (n : ℕ) := sqrt (x^n) = 64

theorem base_number_is_2 (x : ℝ) (n : ℕ) (h : valid_x x n) (hn : n = 12) : x = 2 := 
by 
  sorry

end base_number_is_2_l16_16435


namespace distance_from_edge_l16_16574

theorem distance_from_edge (wall_width picture_width x : ℕ) (h_wall : wall_width = 24) (h_picture : picture_width = 4) (h_centered : x + picture_width + x = wall_width) : x = 10 := by
  -- Proof is omitted
  sorry

end distance_from_edge_l16_16574


namespace multiple_of_k_l16_16328

theorem multiple_of_k (k : ℕ) (m : ℕ) (h₁ : 7 ^ k = 2) (h₂ : 7 ^ (m * k + 2) = 784) : m = 2 :=
sorry

end multiple_of_k_l16_16328


namespace remaining_movie_duration_l16_16995

/--
Given:
1. The laptop was fully charged at 3:20 pm.
2. Hannah started watching a 3-hour series.
3. The laptop turned off at 5:44 pm (fully discharged).

Prove:
The remaining duration of the movie Hannah needs to watch is 36 minutes.
-/
theorem remaining_movie_duration
    (start_full_charge : ℕ := 200)  -- representing 3:20 pm as 200 (20 minutes past 3:00)
    (end_discharge : ℕ := 344)  -- representing 5:44 pm as 344 (44 minutes past 5:00)
    (total_duration_minutes : ℕ := 180)  -- 3 hours in minutes
    (start_time_minutes : ℕ := 200)  -- convert 3:20 pm to minutes past noon
    (end_time_minutes : ℕ := 344)  -- convert 5:44 pm to minutes past noon
    : (total_duration_minutes - (end_time_minutes - start_time_minutes)) = 36 :=
by
  sorry

end remaining_movie_duration_l16_16995


namespace a_sub_b_eq_2_l16_16520

theorem a_sub_b_eq_2 (a b : ℝ)
  (h : (a - 5) ^ 2 + |b ^ 3 - 27| = 0) : a - b = 2 :=
by
  sorry

end a_sub_b_eq_2_l16_16520


namespace color_of_85th_bead_l16_16455

def bead_pattern : List String := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

def bead_color (n : ℕ) : String :=
  bead_pattern.get! (n % bead_pattern.length)

theorem color_of_85th_bead : bead_color 84 = "yellow" := 
by
  sorry

end color_of_85th_bead_l16_16455


namespace larger_solution_quadratic_l16_16863

theorem larger_solution_quadratic : 
  ∀ x1 x2 : ℝ, (x^2 - 13 * x - 48 = 0) → x1 ≠ x2 → (x1 = 16 ∨ x2 = 16) → max x1 x2 = 16 :=
by
  sorry

end larger_solution_quadratic_l16_16863


namespace vector_subtraction_l16_16206

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def vec_smul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

def vec_a : ℝ × ℝ := (3, 5)
def vec_b : ℝ × ℝ := (-2, 1)

theorem vector_subtraction : vec_sub vec_a (vec_smul 2 vec_b) = (7, 3) :=
by
  sorry

end vector_subtraction_l16_16206


namespace range_of_derivative_max_value_of_a_l16_16159

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.cos x - (x - Real.pi / 2) * Real.sin x

-- Define the derivative of f
noncomputable def f' (a x : ℝ) : ℝ :=
  -(1 + a) * Real.sin x - (x - Real.pi / 2) * Real.cos x

-- Part (1): Prove the range of the derivative when a = -1 is [0, π/2]
theorem range_of_derivative (x : ℝ) (h0 : 0 ≤ x) (hπ : x ≤ Real.pi / 2) :
  (0 ≤ f' (-1) x) ∧ (f' (-1) x ≤ Real.pi / 2) := 
sorry

-- Part (2): Prove the maximum value of 'a' when f(x) ≤ 0 always holds
theorem max_value_of_a (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f a x ≤ 0) :
  a ≤ -1 := 
sorry

end range_of_derivative_max_value_of_a_l16_16159


namespace simplify_fraction_l16_16091

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l16_16091


namespace antonieta_tickets_needed_l16_16767

-- Definitions based on conditions:
def ferris_wheel_tickets : ℕ := 6
def roller_coaster_tickets : ℕ := 5
def log_ride_tickets : ℕ := 7
def antonieta_initial_tickets : ℕ := 2

-- Theorem to prove the required number of tickets Antonieta should buy
theorem antonieta_tickets_needed : ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets - antonieta_initial_tickets = 16 :=
by
  sorry

end antonieta_tickets_needed_l16_16767


namespace sin_double_angle_l16_16718

theorem sin_double_angle (α : ℝ) (h : Real.tan α = -1/3) : Real.sin (2 * α) = -3/5 := by 
  sorry

end sin_double_angle_l16_16718


namespace dodecahedron_path_count_l16_16243

/-- A regular dodecahedron with constraints on movement between faces. -/
def num_ways_dodecahedron_move : Nat := 810

/-- Proving the number of different ways to move from the top face to the bottom face of a regular dodecahedron via a series of adjacent faces, such that each face is visited at most once, and movement from the lower ring to the upper ring is not allowed is 810. -/
theorem dodecahedron_path_count :
  num_ways_dodecahedron_move = 810 :=
by
  -- Proof goes here
  sorry

end dodecahedron_path_count_l16_16243


namespace monkey_slips_2_feet_each_hour_l16_16191

/-- 
  A monkey climbs a 17 ft tree, hopping 3 ft and slipping back a certain distance each hour.
  The monkey takes 15 hours to reach the top. Prove that the monkey slips back 2 feet each hour.
-/
def monkey_slips_back_distance (s : ℝ) : Prop :=
  ∃ s : ℝ, (14 * (3 - s) + 3 = 17) ∧ s = 2

theorem monkey_slips_2_feet_each_hour : monkey_slips_back_distance 2 := by
  -- Sorry, proof omitted
  sorry

end monkey_slips_2_feet_each_hour_l16_16191


namespace second_discount_percentage_l16_16141

theorem second_discount_percentage
    (original_price : ℝ)
    (first_discount : ℝ)
    (final_sale_price : ℝ)
    (second_discount : ℝ)
    (h1 : original_price = 390)
    (h2 : first_discount = 14)
    (h3 : final_sale_price = 285.09) :
    second_discount = 15 :=
by
  -- Since we are not providing the full proof, we assume the steps to be correct
  sorry

end second_discount_percentage_l16_16141


namespace integer_solutions_l16_16625

theorem integer_solutions (x y : ℤ) : 
  x^2 * y = 10000 * x + y ↔ 
  (x, y) = (-9, -1125) ∨ 
  (x, y) = (-3, -3750) ∨ 
  (x, y) = (0, 0) ∨ 
  (x, y) = (3, 3750) ∨ 
  (x, y) = (9, 1125) := 
by
  sorry

end integer_solutions_l16_16625


namespace part_a_part_b_l16_16524

-- Part (a)
theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (a b : ℝ), a ≠ b ∧ 0 < (a - b) / (1 + a * b) ∧ (a - b) / (1 + a * b) ≤ 1 := sorry

-- Part (b)
theorem part_b (x y z u : ℝ) :
  ∃ (a b : ℝ), a ≠ b ∧ 0 < (b - a) / (1 + a * b) ∧ (b - a) / (1 + a * b) ≤ 1 := sorry

end part_a_part_b_l16_16524


namespace inverse_proposition_l16_16090

-- Define the variables m, n, and a^2
variables (m n : ℝ) (a : ℝ)

-- State the proof problem
theorem inverse_proposition
  (h1 : m > n)
: m * a^2 > n * a^2 :=
sorry

end inverse_proposition_l16_16090


namespace equation_D_has_two_distinct_real_roots_l16_16673

def quadratic_has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem equation_D_has_two_distinct_real_roots : quadratic_has_two_distinct_real_roots 1 2 (-1) :=
by {
  sorry
}

end equation_D_has_two_distinct_real_roots_l16_16673


namespace intersection_x_value_l16_16175

theorem intersection_x_value:
  ∃ x y : ℝ, y = 4 * x - 29 ∧ 3 * x + y = 105 ∧ x = 134 / 7 :=
by
  sorry

end intersection_x_value_l16_16175


namespace complement_of_M_in_U_l16_16165

def universal_set : Set ℝ := {x | x > 0}
def set_M : Set ℝ := {x | x > 1}
def complement (U M : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ M}

theorem complement_of_M_in_U :
  complement universal_set set_M = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end complement_of_M_in_U_l16_16165


namespace range_of_m_l16_16622

noncomputable def inequality_has_solutions (x m : ℝ) :=
  |x + 2| - |x + 3| > m

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, inequality_has_solutions x m) → m < 1 :=
by
  sorry

end range_of_m_l16_16622


namespace factorize_expression_l16_16217

theorem factorize_expression (x : ℝ) : (x + 3) ^ 2 - (x + 3) = (x + 3) * (x + 2) :=
by
  sorry

end factorize_expression_l16_16217


namespace compare_magnitudes_l16_16924

noncomputable def log_base_3_of_2 : ℝ := Real.log 2 / Real.log 3   -- def a
noncomputable def ln_2 : ℝ := Real.log 2                          -- def b
noncomputable def five_minus_pi : ℝ := 5 - Real.pi                -- def c

theorem compare_magnitudes :
  let a := log_base_3_of_2
  let b := ln_2
  let c := five_minus_pi
  c < a ∧ a < b :=
by
  sorry

end compare_magnitudes_l16_16924


namespace positive_difference_enrollment_l16_16397

theorem positive_difference_enrollment 
  (highest_enrollment : ℕ)
  (lowest_enrollment : ℕ)
  (h_highest : highest_enrollment = 2150)
  (h_lowest : lowest_enrollment = 980) :
  highest_enrollment - lowest_enrollment = 1170 :=
by {
  -- Proof to be added here
  sorry
}

end positive_difference_enrollment_l16_16397


namespace greatest_possible_integer_l16_16772

theorem greatest_possible_integer 
  (n k l : ℕ) 
  (h1 : n < 150) 
  (h2 : n = 9 * k - 2) 
  (h3 : n = 6 * l - 4) : 
  n = 146 := 
sorry

end greatest_possible_integer_l16_16772


namespace abs_eq_cases_l16_16111

theorem abs_eq_cases (a b : ℝ) : (|a| = |b|) → (a = b ∨ a = -b) :=
sorry

end abs_eq_cases_l16_16111


namespace speed_conversion_l16_16589

theorem speed_conversion (speed_kmph : ℝ) (h : speed_kmph = 18) : speed_kmph * (1000 / 3600) = 5 := by
  sorry

end speed_conversion_l16_16589


namespace intersection_A_B_l16_16959

variable (A : Set ℤ) (B : Set ℤ)

-- Define the set A and B
def set_A : Set ℤ := {0, 1, 2}
def set_B : Set ℤ := {x | 1 < x ∧ x < 4}

theorem intersection_A_B :
  set_A ∩ set_B = {2} :=
by
  sorry

end intersection_A_B_l16_16959


namespace endomorphisms_of_Z2_are_linear_functions_l16_16340

namespace GroupEndomorphism

-- Definition of an endomorphism: a homomorphism from Z² to itself
def is_endomorphism (f : ℤ × ℤ → ℤ × ℤ) : Prop :=
  ∀ a b : ℤ × ℤ, f (a + b) = f a + f b

-- Definition of the specific form of endomorphisms for Z²
def specific_endomorphism_form (u v : ℤ × ℤ) (φ : ℤ × ℤ) : ℤ × ℤ :=
  (φ.1 * u.1 + φ.2 * v.1, φ.1 * u.2 + φ.2 * v.2)

-- Main theorem:
theorem endomorphisms_of_Z2_are_linear_functions :
  ∀ φ : ℤ × ℤ → ℤ × ℤ, is_endomorphism φ →
  ∃ u v : ℤ × ℤ, φ = specific_endomorphism_form u v := by
  sorry

end GroupEndomorphism

end endomorphisms_of_Z2_are_linear_functions_l16_16340


namespace university_diploma_percentage_l16_16660

-- Define the conditions
variables (P N JD ND : ℝ)
-- P: total population assumed as 100% for simplicity
-- N: percentage of people with university diploma
-- JD: percentage of people who have the job of their choice
-- ND: percentage of people who do not have a university diploma but have the job of their choice
variables (A : ℝ) -- A: University diploma percentage of those who do not have the job of their choice
variable (total_diploma : ℝ)
axiom country_Z_conditions : 
  (P = 100) ∧ (ND = 18) ∧ (JD = 40) ∧ (A = 25)

-- Define the proof problem
theorem university_diploma_percentage :
  (N = ND + (JD - ND) + (total_diploma * (P - JD * (P / JD) / P))) →
  N = 37 :=
by
  sorry

end university_diploma_percentage_l16_16660


namespace sum_common_divisors_60_18_l16_16638

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m => n % m = 0)

noncomputable def sum (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

theorem sum_common_divisors_60_18 :
  sum (List.filter (λ d => d ∈ divisors 18) (divisors 60)) = 12 :=
by
  sorry

end sum_common_divisors_60_18_l16_16638


namespace students_from_other_communities_eq_90_l16_16150

theorem students_from_other_communities_eq_90 {total_students : ℕ} 
  (muslims_percentage : ℕ)
  (hindus_percentage : ℕ)
  (sikhs_percentage : ℕ)
  (christians_percentage : ℕ)
  (buddhists_percentage : ℕ)
  : total_students = 1000 →
    muslims_percentage = 36 →
    hindus_percentage = 24 →
    sikhs_percentage = 15 →
    christians_percentage = 10 →
    buddhists_percentage = 6 →
    (total_students * (100 - (muslims_percentage + hindus_percentage + sikhs_percentage + christians_percentage + buddhists_percentage))) / 100 = 90 :=
by
  intros h_total h_muslims h_hindus h_sikhs h_christians h_buddhists
  -- Proof can be omitted as indicated
  sorry

end students_from_other_communities_eq_90_l16_16150


namespace decagon_diagonals_l16_16894

-- The condition for the number of diagonals in a polygon
def number_of_diagonals (n : Nat) : Nat :=
  n * (n - 3) / 2

-- The specific proof statement for a decagon
theorem decagon_diagonals : number_of_diagonals 10 = 35 := by
  -- The proof would go here
  sorry

end decagon_diagonals_l16_16894


namespace shirley_boxes_to_cases_l16_16759

theorem shirley_boxes_to_cases (boxes_sold : Nat) (boxes_per_case : Nat) (cases_needed : Nat) 
      (h1 : boxes_sold = 54) (h2 : boxes_per_case = 6) : cases_needed = 9 :=
by
  sorry

end shirley_boxes_to_cases_l16_16759


namespace base_square_eq_l16_16464

theorem base_square_eq (b : ℕ) (h : (3*b + 3)^2 = b^3 + 2*b^2 + 3*b) : b = 9 :=
sorry

end base_square_eq_l16_16464


namespace first_term_arithmetic_sequence_median_1010_last_2015_l16_16808

theorem first_term_arithmetic_sequence_median_1010_last_2015 (a₁ : ℕ) :
  let median := 1010
  let last_term := 2015
  (a₁ + last_term = 2 * median) → a₁ = 5 :=
by
  intros
  sorry

end first_term_arithmetic_sequence_median_1010_last_2015_l16_16808


namespace pencils_in_each_box_l16_16219

theorem pencils_in_each_box (n : ℕ) (h : 10 * n - 10 = 40) : n = 5 := by
  sorry

end pencils_in_each_box_l16_16219


namespace cube_property_l16_16970

theorem cube_property (x : ℝ) (s : ℝ) 
  (h1 : s^3 = 8 * x)
  (h2 : 6 * s^2 = 4 * x) :
  x = 5400 :=
by
  sorry

end cube_property_l16_16970


namespace complete_square_ratio_l16_16809

theorem complete_square_ratio (k : ℝ) :
  ∃ c p q : ℝ, 
    8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q ∧ 
    q / p = -142 / 3 :=
sorry

end complete_square_ratio_l16_16809


namespace _l16_16264

def triangle (A B C : Type) : Prop :=
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

def angles_not_equal_sides_not_equal (A B C : Type) (angleB angleC : ℝ) (sideAC sideAB : ℝ) : Prop :=
  triangle A B C →
  (angleB ≠ angleC → sideAC ≠ sideAB)
  
lemma xiaoming_theorem {A B C : Type} 
  (hTriangle : triangle A B C)
  (angleB angleC : ℝ)
  (sideAC sideAB : ℝ) :
  angleB ≠ angleC → sideAC ≠ sideAB := 
sorry

end _l16_16264


namespace symmetric_point_x_correct_l16_16126

-- Define the Cartesian coordinate system
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetry with respect to the x-axis
def symmetricPointX (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Given point (-2, 1, 4)
def givenPoint : Point3D := { x := -2, y := 1, z := 4 }

-- Define the expected symmetric point
def expectedSymmetricPoint : Point3D := { x := -2, y := -1, z := -4 }

-- State the theorem to prove the expected symmetric point
theorem symmetric_point_x_correct :
  symmetricPointX givenPoint = expectedSymmetricPoint := by
  -- here the proof would go, but we leave it as sorry
  sorry

end symmetric_point_x_correct_l16_16126


namespace count_valid_pairs_l16_16136

theorem count_valid_pairs : 
  ∃ n : ℕ, n = 5 ∧ 
  ∀ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 40 →
  (5^j - 2^i) % 1729 = 0 →
  i = 0 ∧ j = 36 ∨ 
  i = 1 ∧ j = 37 ∨ 
  i = 2 ∧ j = 38 ∨ 
  i = 3 ∧ j = 39 ∨ 
  i = 4 ∧ j = 40 :=
by
  sorry

end count_valid_pairs_l16_16136


namespace there_exists_l_l16_16200

theorem there_exists_l (m n : ℕ) (h1 : m ≠ 0) (h2 : n ≠ 0) 
  (h3 : ∀ k : ℕ, 0 < k → Nat.gcd (17 * k - 1) m = Nat.gcd (17 * k - 1) n) :
  ∃ l : ℤ, m = (17 : ℕ) ^ l.natAbs * n := 
sorry

end there_exists_l_l16_16200


namespace half_MN_correct_l16_16023

noncomputable def OM : ℝ × ℝ := (-2, 3)
noncomputable def ON : ℝ × ℝ := (-1, -5)
noncomputable def MN : ℝ × ℝ := (ON.1 - OM.1, ON.2 - OM.2)
noncomputable def half_MN : ℝ × ℝ := (MN.1 / 2, MN.2 / 2)

theorem half_MN_correct : half_MN = (1 / 2, -4) :=
by
  -- define the values of OM and ON
  let OM : ℝ × ℝ := (-2, 3)
  let ON : ℝ × ℝ := (-1, -5)
  -- calculate MN
  let MN : ℝ × ℝ := (ON.1 - OM.1, ON.2 - OM.2)
  -- calculate half of MN
  let half_MN : ℝ × ℝ := (MN.1 / 2, MN.2 / 2)
  -- assert the expected value
  exact sorry

end half_MN_correct_l16_16023


namespace base5_2004_to_decimal_is_254_l16_16787

def base5_to_decimal (n : Nat) : Nat :=
  match n with
  | 2004 => 2 * 5^3 + 0 * 5^2 + 0 * 5^1 + 4 * 5^0
  | _ => 0

theorem base5_2004_to_decimal_is_254 :
  base5_to_decimal 2004 = 254 :=
by
  -- Proof goes here
  sorry

end base5_2004_to_decimal_is_254_l16_16787


namespace negation_example_l16_16962

theorem negation_example :
  ¬(∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - x₀ > 0) :=
by
  sorry

end negation_example_l16_16962


namespace no_base_makes_131b_square_l16_16369

theorem no_base_makes_131b_square : ∀ (b : ℤ), b > 3 → ∀ (n : ℤ), n * n ≠ b^2 + 3 * b + 1 :=
by
  intros b h_gt_3 n
  sorry

end no_base_makes_131b_square_l16_16369


namespace exists_positive_real_u_l16_16595

theorem exists_positive_real_u (n : ℕ) (h_pos : n > 0) : 
  ∃ u : ℝ, u > 0 ∧ ∀ n : ℕ, n > 0 → (⌊u^n⌋ - n) % 2 = 0 :=
sorry

end exists_positive_real_u_l16_16595


namespace parabola_c_value_l16_16977

theorem parabola_c_value (a b c : ℝ) (h1 : 3 = a * (-1)^2 + b * (-1) + c)
  (h2 : 1 = a * (-2)^2 + b * (-2) + c) : c = 1 :=
sorry

end parabola_c_value_l16_16977


namespace radius_of_smaller_base_of_truncated_cone_l16_16672

theorem radius_of_smaller_base_of_truncated_cone 
  (r1 r2 r3 : ℕ) (touching : 2 * r1 = r2 ∧ r1 + r3 = r2 * 2):
  (∀ (R : ℕ), R = 6) :=
sorry

end radius_of_smaller_base_of_truncated_cone_l16_16672


namespace y_in_terms_of_x_l16_16005

theorem y_in_terms_of_x (p x y : ℝ) (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : y = x / (x - 1) := 
by 
  sorry

end y_in_terms_of_x_l16_16005


namespace opposite_neg_two_is_two_l16_16675

theorem opposite_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_neg_two_is_two_l16_16675


namespace find_numbers_with_lcm_gcd_l16_16228

theorem find_numbers_with_lcm_gcd :
  ∃ a b : ℕ, lcm a b = 90 ∧ gcd a b = 6 ∧ ((a = 18 ∧ b = 30) ∨ (a = 30 ∧ b = 18)) :=
by
  sorry

end find_numbers_with_lcm_gcd_l16_16228


namespace arithmetic_expression_equals_47_l16_16946

-- Define the arithmetic expression
def arithmetic_expression : ℕ :=
  2 + 5 * 3^2 - 4 + 6 * 2 / 3

-- The proof goal: arithmetic_expression equals 47
theorem arithmetic_expression_equals_47 : arithmetic_expression = 47 := 
by
  sorry

end arithmetic_expression_equals_47_l16_16946


namespace magic_king_total_episodes_l16_16554

theorem magic_king_total_episodes
  (total_seasons : ℕ)
  (first_half_seasons : ℕ)
  (second_half_seasons : ℕ)
  (episodes_first_half : ℕ)
  (episodes_second_half : ℕ)
  (h1 : total_seasons = 10)
  (h2 : first_half_seasons = total_seasons / 2)
  (h3 : second_half_seasons = total_seasons / 2)
  (h4 : episodes_first_half = 20)
  (h5 : episodes_second_half = 25)
  : (first_half_seasons * episodes_first_half + second_half_seasons * episodes_second_half) = 225 :=
by
  sorry

end magic_king_total_episodes_l16_16554


namespace number_of_organizations_in_foundation_l16_16549

def company_raised : ℕ := 2500
def donation_percentage : ℕ := 80
def each_organization_receives : ℕ := 250
def total_donated : ℕ := (donation_percentage * company_raised) / 100

theorem number_of_organizations_in_foundation : total_donated / each_organization_receives = 8 :=
by
  sorry

end number_of_organizations_in_foundation_l16_16549


namespace parabola_directrix_l16_16579

theorem parabola_directrix (y x : ℝ) : y^2 = -8 * x → x = -1 :=
by
  sorry

end parabola_directrix_l16_16579


namespace find_coordinates_of_P_l16_16577

-- Define the problem conditions
def P (m : ℤ) := (2 * m + 4, m - 1)
def A := (2, -4)
def line_l (y : ℤ) := y = -4
def P_on_line_l (m : ℤ) := line_l (m - 1)

theorem find_coordinates_of_P (m : ℤ) (h : P_on_line_l m) : P m = (-2, -4) := 
  by sorry

end find_coordinates_of_P_l16_16577


namespace point_in_second_quadrant_l16_16402

theorem point_in_second_quadrant (m n : ℝ)
  (h_translation : ∃ A' : ℝ × ℝ, A' = (m+2, n+3) ∧ (A'.1 < 0) ∧ (A'.2 > 0)) :
  m < -2 ∧ n > -3 :=
by
  sorry

end point_in_second_quadrant_l16_16402


namespace problem_statement_l16_16727

variables {Point Line Plane : Type}
variables (l : Line) (α β : Plane)

-- Conditions
def parallel (l : Line) (α : Plane) : Prop := sorry
def perpendicular (l : Line) (α : Plane) : Prop := sorry
def perpendicular_planes (α β : Plane) : Prop := sorry

-- The proof problem
theorem problem_statement (h1 : parallel l α) (h2 : perpendicular l β) : perpendicular_planes α β :=
sorry

end problem_statement_l16_16727


namespace total_amount_earned_l16_16022

-- Definitions of the conditions.
def work_done_per_day (days : ℕ) : ℚ := 1 / days

def total_work_done_per_day : ℚ :=
  work_done_per_day 6 + work_done_per_day 8 + work_done_per_day 12

def b_share : ℚ := work_done_per_day 8

def total_amount (b_earnings : ℚ) : ℚ := b_earnings * (total_work_done_per_day / b_share)

-- Main theorem stating that the total amount earned is $1170 if b's share is $390.
theorem total_amount_earned (h_b : b_share * 390 = 390) : total_amount 390 = 1170 := by sorry

end total_amount_earned_l16_16022


namespace distance_between_incenter_and_circumcenter_of_right_triangle_l16_16693

theorem distance_between_incenter_and_circumcenter_of_right_triangle (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) (right_triangle : a^2 + b^2 = c^2) :
    ∃ (IO : ℝ), IO = Real.sqrt 5 :=
by
  rw [h1, h2, h3] at right_triangle
  have h_sum : 6^2 + 8^2 = 10^2 := by sorry
  exact ⟨Real.sqrt 5, by sorry⟩

end distance_between_incenter_and_circumcenter_of_right_triangle_l16_16693


namespace remaining_area_l16_16500

theorem remaining_area (x : ℝ) :
  let A_large := (2 * x + 8) * (x + 6)
  let A_hole := (3 * x - 4) * (x + 1)
  A_large - A_hole = - x^2 + 22 * x + 52 := by
  let A_large := (2 * x + 8) * (x + 6)
  let A_hole := (3 * x - 4) * (x + 1)
  have hA_large : A_large = 2 * x^2 + 20 * x + 48 := by
    sorry
  have hA_hole : A_hole = 3 * x^2 - 2 * x - 4 := by
    sorry
  calc
    A_large - A_hole = (2 * x^2 + 20 * x + 48) - (3 * x^2 - 2 * x - 4) := by
      rw [hA_large, hA_hole]
    _ = -x^2 + 22 * x + 52 := by
      ring

end remaining_area_l16_16500


namespace water_remaining_l16_16686

variable (initial_amount : ℝ) (leaked_amount : ℝ)

theorem water_remaining (h1 : initial_amount = 0.75)
                       (h2 : leaked_amount = 0.25) :
  initial_amount - leaked_amount = 0.50 :=
by
  sorry

end water_remaining_l16_16686


namespace correct_judgement_l16_16765

noncomputable def f (x : ℝ) : ℝ :=
if -2 ≤ x ∧ x ≤ 2 then (1 / 2) * Real.sqrt (4 - x^2)
else - (1 / 2) * Real.sqrt (x^2 - 4)

noncomputable def F (x : ℝ) : ℝ := f x + x

theorem correct_judgement : (∀ y : ℝ, ∃ x : ℝ, (f x = y) ↔ (y ∈ Set.Iic 1)) ∧ (∃! x : ℝ, F x = 0) :=
by
  sorry

end correct_judgement_l16_16765


namespace line_equation_l16_16948

theorem line_equation :
  ∃ m b, m = 1 ∧ b = 5 ∧ (∀ x y, y = m * x + b ↔ x - y + 5 = 0) :=
by
  sorry

end line_equation_l16_16948


namespace perimeter_of_figure_l16_16677

variable (x y : ℝ)
variable (lengths : Set ℝ)
variable (perpendicular_adjacent : Prop)
variable (area : ℝ)

-- Conditions
def condition_1 : Prop := ∀ l ∈ lengths, l = x ∨ l = y
def condition_2 : Prop := perpendicular_adjacent
def condition_3 : Prop := area = 252
def condition_4 : Prop := x = 2 * y

-- Problem statement
theorem perimeter_of_figure
  (h1 : condition_1 x y lengths)
  (h2 : condition_2 perpendicular_adjacent)
  (h3 : condition_3 area)
  (h4 : condition_4 x y) :
  ∃ perimeter : ℝ, perimeter = 96 := by
  sorry

end perimeter_of_figure_l16_16677


namespace total_cookies_is_58_l16_16429

noncomputable def total_cookies : ℝ :=
  let M : ℝ := 5
  let T : ℝ := 2 * M
  let W : ℝ := T + 0.4 * T
  let Th : ℝ := W - 0.25 * W
  let F : ℝ := Th - 0.25 * Th
  let Sa : ℝ := F - 0.25 * F
  let Su : ℝ := Sa - 0.25 * Sa
  M + T + W + Th + F + Sa + Su

theorem total_cookies_is_58 : total_cookies = 58 :=
by
  sorry

end total_cookies_is_58_l16_16429


namespace remainder_of_sum_l16_16849

theorem remainder_of_sum (p q : ℤ) (c d : ℤ) 
  (hc : c = 100 * p + 78)
  (hd : d = 150 * q + 123) :
  (c + d) % 50 = 1 :=
sorry

end remainder_of_sum_l16_16849


namespace number_of_aluminum_atoms_l16_16647

def molecular_weight (n : ℕ) : ℝ :=
  n * 26.98 + 30.97 + 4 * 16.0

theorem number_of_aluminum_atoms (n : ℕ) (h : molecular_weight n = 122) : n = 1 :=
by
  sorry

end number_of_aluminum_atoms_l16_16647


namespace function_cannot_be_decreasing_if_f1_lt_f2_l16_16674

variable (f : ℝ → ℝ)

theorem function_cannot_be_decreasing_if_f1_lt_f2
  (h : f 1 < f 2) : ¬ (∀ x y, x < y → f y < f x) :=
by
  sorry

end function_cannot_be_decreasing_if_f1_lt_f2_l16_16674


namespace minimum_people_l16_16582

def num_photos : ℕ := 10
def num_center_men : ℕ := 10
def num_people_per_photo : ℕ := 3

theorem minimum_people (n : ℕ) (h : n = num_photos) :
  (∃ total_people, total_people = 16) :=
sorry

end minimum_people_l16_16582


namespace total_pieces_of_clothing_l16_16439

-- Define the conditions:
def boxes : ℕ := 4
def scarves_per_box : ℕ := 2
def mittens_per_box : ℕ := 6

-- Define the target statement:
theorem total_pieces_of_clothing : (boxes * (scarves_per_box + mittens_per_box)) = 32 :=
by
  sorry

end total_pieces_of_clothing_l16_16439


namespace sin2alpha_cos2beta_l16_16074

variable (α β : ℝ)

-- Conditions
def tan_add_eq : Prop := Real.tan (α + β) = -3
def tan_sub_eq : Prop := Real.tan (α - β) = 2

-- Question
theorem sin2alpha_cos2beta (h1 : tan_add_eq α β) (h2 : tan_sub_eq α β) : 
  (Real.sin (2 * α)) / (Real.cos (2 * β)) = -1 / 7 := 
  sorry

end sin2alpha_cos2beta_l16_16074


namespace random_events_l16_16800

-- Define what it means for an event to be random
def is_random_event (e : Prop) : Prop := ∃ (h : Prop), e ∨ ¬e

-- Define the events based on the problem statements
def event1 := ∃ (good_cups : ℕ), good_cups = 3
def event2 := ∃ (half_hit_targets : ℕ), half_hit_targets = 50
def event3 := ∃ (correct_digit : ℕ), correct_digit = 1
def event4 := true -- Opposite charges attract each other, which is always true
def event5 := ∃ (first_prize : ℕ), first_prize = 1

-- State the problem as a theorem
theorem random_events :
  is_random_event event1 ∧ is_random_event event2 ∧ is_random_event event3 ∧ is_random_event event5 :=
by
  sorry

end random_events_l16_16800


namespace speed_of_ship_with_two_sails_l16_16653

noncomputable def nautical_mile : ℝ := 1.15
noncomputable def land_miles_traveled : ℝ := 345
noncomputable def time_with_one_sail : ℝ := 4
noncomputable def time_with_two_sails : ℝ := 4
noncomputable def speed_with_one_sail : ℝ := 25

theorem speed_of_ship_with_two_sails :
  ∃ S : ℝ, 
    (S * time_with_two_sails + speed_with_one_sail * time_with_one_sail = land_miles_traveled / nautical_mile) → 
    S = 50  :=
by
  sorry

end speed_of_ship_with_two_sails_l16_16653


namespace claire_hours_cleaning_l16_16232

-- Definitions of given conditions
def total_hours_in_day : ℕ := 24
def hours_sleeping : ℕ := 8
def hours_cooking : ℕ := 2
def hours_crafting : ℕ := 5
def total_working_hours : ℕ := total_hours_in_day - hours_sleeping

-- Definition of the question
def hours_cleaning := total_working_hours - (hours_cooking + hours_crafting + hours_crafting)

-- The proof goal
theorem claire_hours_cleaning : hours_cleaning = 4 := by
  sorry

end claire_hours_cleaning_l16_16232


namespace find_a7_coefficient_l16_16659

theorem find_a7_coefficient (a_7 : ℤ) : 
    (∀ x : ℤ, (x+1)^5 * (2*x-1)^3 = a_8 * x^8 + a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) → a_7 = 28 :=
by
  sorry

end find_a7_coefficient_l16_16659


namespace smallest_value_a_plus_b_l16_16573

theorem smallest_value_a_plus_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 3^7 * 5^3 = a^b) : a + b = 3376 :=
sorry

end smallest_value_a_plus_b_l16_16573


namespace proof_of_arithmetic_sequence_l16_16591

theorem proof_of_arithmetic_sequence 
  (x y z : ℕ) 
  (h1 : x + y + z = 15) 
  (h2 : x < y) 
  (h3 : y < z)
  (h4 : (x + 1) * (z + 9) = (y + 3) ^ 2) : 
  (x, y, z) = (3, 5, 7) :=
sorry

end proof_of_arithmetic_sequence_l16_16591


namespace sequence_general_term_l16_16373

theorem sequence_general_term :
  ∀ n : ℕ, n > 0 → (∀ a: ℕ → ℝ,  a 1 = 4 ∧ (∀ n: ℕ, n > 0 → a (n + 1) = (3 * a n + 2) / (a n + 4))
  → a n = (2 ^ (n - 1) + 5 ^ (n - 1)) / (5 ^ (n - 1) - 2 ^ (n - 1))) :=
by
  sorry

end sequence_general_term_l16_16373


namespace sum_of_transformed_numbers_l16_16588

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  let a' := a + 4
  let b' := b + 4
  let a'' := 3 * a'
  let b'' := 3 * b'
  a'' + b'' = 3 * S + 24 := 
by
  let a' := a + 4
  let b' := b + 4
  let a'' := 3 * a'
  let b'' := 3 * b'
  sorry

end sum_of_transformed_numbers_l16_16588


namespace tan_alpha_value_l16_16190

open Real

variable (α : ℝ)

/- Conditions -/
def alpha_interval : Prop := (0 < α) ∧ (α < π)
def sine_cosine_sum : Prop := sin α + cos α = -7 / 13

/- Statement -/
theorem tan_alpha_value 
  (h1 : alpha_interval α)
  (h2 : sine_cosine_sum α) : 
  tan α = -5 / 12 :=
sorry

end tan_alpha_value_l16_16190


namespace red_card_value_l16_16585

theorem red_card_value (credits : ℕ) (total_cards : ℕ) (blue_card_value : ℕ) (red_cards : ℕ) (blue_cards : ℕ) 
    (condition1 : blue_card_value = 5)
    (condition2 : total_cards = 20)
    (condition3 : credits = 84)
    (condition4 : red_cards = 8)
    (condition5 : blue_cards = total_cards - red_cards) :
  (credits - blue_cards * blue_card_value) / red_cards = 3 :=
by
  sorry

end red_card_value_l16_16585


namespace probability_adjacent_difference_l16_16490

noncomputable def probability_no_adjacent_same_rolls : ℚ :=
  (7 / 8) ^ 6

theorem probability_adjacent_difference :
  let num_people := 6
  let sides_of_die := 8
  ( ∀ i : ℕ, 0 ≤ i ∧ i < num_people -> (∃ x : ℕ, 1 ≤ x ∧ x ≤ sides_of_die)) →
  probability_no_adjacent_same_rolls = 117649 / 262144 := 
by 
  sorry

end probability_adjacent_difference_l16_16490


namespace linear_function_does_not_pass_third_quadrant_l16_16193

/-
Given an inverse proportion function \( y = \frac{a^2 + 1}{x} \), where \( a \) is a constant, and given two points \( (x_1, y_1) \) and \( (x_2, y_2) \) on the same branch of this function, 
with \( b = (x_1 - x_2)(y_1 - y_2) \), prove that the graph of the linear function \( y = bx - b \) does not pass through the third quadrant.
-/

theorem linear_function_does_not_pass_third_quadrant 
  (a x1 x2 : ℝ) 
  (y1 y2 : ℝ)
  (h1 : y1 = (a^2 + 1) / x1) 
  (h2 : y2 = (a^2 + 1) / x2) 
  (h3 : b = (x1 - x2) * (y1 - y2)) : 
  ∃ b, ∀ x y : ℝ, (y = b * x - b) → (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0) :=
by 
  sorry

end linear_function_does_not_pass_third_quadrant_l16_16193


namespace rectangular_floor_paint_l16_16915

theorem rectangular_floor_paint (a b : ℕ) (ha : a > 0) (hb : b > a) (h1 : a * b = 2 * (a - 4) * (b - 4) + 32) : 
  ∃ pairs : Finset (ℕ × ℕ), pairs.card = 3 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → b > a :=
by 
  sorry

end rectangular_floor_paint_l16_16915


namespace ratio_of_members_l16_16600

theorem ratio_of_members (f m c : ℕ) 
  (h1 : (35 * f + 30 * m + 10 * c) / (f + m + c) = 25) :
  2 * f + m = 3 * c :=
by
  sorry

end ratio_of_members_l16_16600


namespace trig_expr_value_l16_16416

theorem trig_expr_value :
  (Real.cos (7 * Real.pi / 24)) ^ 4 +
  (Real.sin (11 * Real.pi / 24)) ^ 4 +
  (Real.sin (17 * Real.pi / 24)) ^ 4 +
  (Real.cos (13 * Real.pi / 24)) ^ 4 = 3 / 2 :=
by
  sorry

end trig_expr_value_l16_16416


namespace expected_score_two_free_throws_is_correct_l16_16041

noncomputable def expected_score_two_free_throws (p : ℝ) (n : ℕ) : ℝ :=
n * p

theorem expected_score_two_free_throws_is_correct : expected_score_two_free_throws 0.7 2 = 1.4 :=
by
  -- Proof will be written here.
  sorry

end expected_score_two_free_throws_is_correct_l16_16041


namespace lenny_remaining_amount_l16_16813

theorem lenny_remaining_amount :
  let initial_amount := 270
  let console_price := 149
  let console_discount := 0.15 * console_price
  let final_console_price := console_price - console_discount
  let groceries_price := 60
  let groceries_discount := 0.10 * groceries_price
  let final_groceries_price := groceries_price - groceries_discount
  let lunch_cost := 30
  let magazine_cost := 3.99
  let total_expenses := final_console_price + final_groceries_price + lunch_cost + magazine_cost
  initial_amount - total_expenses = 55.36 :=
by
  sorry

end lenny_remaining_amount_l16_16813


namespace find_m_of_ellipse_conditions_l16_16519

-- definition for isEllipseGivenFocus condition
def isEllipseGivenFocus (m : ℝ) : Prop :=
  ∃ (a : ℝ), a = 5 ∧ (-4)^2 = a^2 - m^2 ∧ 0 < m

-- statement to prove the described condition implies m = 3
theorem find_m_of_ellipse_conditions (m : ℝ) (h : isEllipseGivenFocus m) : m = 3 :=
sorry

end find_m_of_ellipse_conditions_l16_16519


namespace cafe_location_l16_16879

-- Definition of points and conditions
structure Point where
  x : ℤ
  y : ℚ

def mark : Point := { x := 1, y := 8 }
def sandy : Point := { x := -5, y := 0 }

-- The problem statement
theorem cafe_location :
  ∃ cafe : Point, cafe.x = -3 ∧ cafe.y = 8/3 := by
  sorry

end cafe_location_l16_16879


namespace shaded_region_area_l16_16155

def area_of_shaded_region (grid_height grid_width triangle_base triangle_height : ℝ) : ℝ :=
  let total_area := grid_height * grid_width
  let triangle_area := 0.5 * triangle_base * triangle_height
  total_area - triangle_area

theorem shaded_region_area :
  area_of_shaded_region 3 15 5 3 = 37.5 :=
by 
  sorry

end shaded_region_area_l16_16155


namespace find_m_l16_16930

theorem find_m (m : ℤ) (y : ℤ) : 
  (y^2 + m * y + 2) % (y - 1) = (m + 3) ∧ 
  (y^2 + m * y + 2) % (y + 1) = (3 - m) ∧
  (m + 3 = 3 - m) → m = 0 :=
sorry

end find_m_l16_16930


namespace cotton_needed_l16_16838

noncomputable def feet_of_cotton_per_teeshirt := 4
noncomputable def number_of_teeshirts := 15

theorem cotton_needed : feet_of_cotton_per_teeshirt * number_of_teeshirts = 60 := 
by 
  sorry

end cotton_needed_l16_16838


namespace mold_growth_problem_l16_16478

/-- Given the conditions:
    - Initial mold spores: 50 at 9:00 a.m.
    - Colony doubles in size every 10 minutes.
    - Time elapsed: 70 minutes from 9:00 a.m. to 10:10 a.m.,

    Prove that the number of mold spores at 10:10 a.m. is 6400 -/
theorem mold_growth_problem : 
  let initial_mold_spores := 50
  let doubling_period_minutes := 10
  let elapsed_minutes := 70
  let doublings := elapsed_minutes / doubling_period_minutes
  let final_population := initial_mold_spores * (2 ^ doublings)
  final_population = 6400 :=
by 
  let initial_mold_spores := 50
  let doubling_period_minutes := 10
  let elapsed_minutes := 70
  let doublings := elapsed_minutes / doubling_period_minutes
  let final_population := initial_mold_spores * (2 ^ doublings)
  sorry

end mold_growth_problem_l16_16478


namespace ellipse_major_minor_ratio_l16_16756

theorem ellipse_major_minor_ratio (m : ℝ) (x y : ℝ) (h1 : x^2 + y^2 / m = 1) (h2 : 2 * 1 = 4 * Real.sqrt m) 
  : m = 1 / 4 :=
sorry

end ellipse_major_minor_ratio_l16_16756


namespace flat_fee_is_65_l16_16592

-- Define the problem constants
def George_nights : ℕ := 3
def Noah_nights : ℕ := 6
def George_cost : ℤ := 155
def Noah_cost : ℤ := 290

-- Prove that the flat fee for the first night is 65, given the costs and number of nights stayed.
theorem flat_fee_is_65 
  (f n : ℤ)
  (h1 : f + (George_nights - 1) * n = George_cost)
  (h2 : f + (Noah_nights - 1) * n = Noah_cost) :
  f = 65 := 
sorry

end flat_fee_is_65_l16_16592


namespace toads_per_acre_l16_16774

theorem toads_per_acre (b g : ℕ) (h₁ : b = 25 * g)
  (h₂ : b / 4 = 50) : g = 8 :=
by
  -- Condition h₁: For every green toad, there are 25 brown toads.
  -- Condition h₂: One-quarter of the brown toads are spotted, and there are 50 spotted brown toads per acre.
  sorry

end toads_per_acre_l16_16774


namespace geometric_product_seven_terms_l16_16510

theorem geometric_product_seven_terms (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = 1) 
  (h2 : a 6 + a 4 = 2 * (a 3 + a 1)) 
  (h_geometric : ∀ n, a (n + 1) = q * a n) :
  (a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7) = 128 := 
by 
  -- Steps involving algebraic manipulation and properties of geometric sequences should be here
  sorry

end geometric_product_seven_terms_l16_16510


namespace transmission_time_calc_l16_16792

theorem transmission_time_calc
  (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) (time_in_minutes : ℕ)
  (h_blocks : blocks = 80)
  (h_chunks_per_block : chunks_per_block = 640)
  (h_transmission_rate : transmission_rate = 160) 
  (h_time_in_minutes : time_in_minutes = 5) : 
  (blocks * chunks_per_block / transmission_rate) / 60 = time_in_minutes := 
by
  sorry

end transmission_time_calc_l16_16792


namespace rectangle_max_area_l16_16215

theorem rectangle_max_area (w : ℝ) (h : ℝ) (hw : h = 2 * w) (perimeter : 2 * (w + h) = 40) :
  w * h = 800 / 9 := 
by
  -- Given: h = 2w and 2(w + h) = 40
  -- We need to prove that the area A = wh = 800/9
  sorry

end rectangle_max_area_l16_16215


namespace taxi_ride_cost_l16_16993

-- Definitions
def initial_fee : Real := 2
def distance_in_miles : Real := 4
def cost_per_mile : Real := 2.5

-- Theorem statement
theorem taxi_ride_cost (initial_fee : Real) (distance_in_miles : Real) (cost_per_mile : Real) : 
  initial_fee + distance_in_miles * cost_per_mile = 12 := 
by
  sorry

end taxi_ride_cost_l16_16993


namespace question_inequality_l16_16929

theorem question_inequality
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (cond : a + b ≤ 4) :
  (1 / a + 1 / b) ≥ 1 := 
sorry

end question_inequality_l16_16929


namespace min_red_chips_l16_16208

theorem min_red_chips (w b r : ℕ) 
  (h1 : b ≥ (1 / 3) * w)
  (h2 : b ≤ (1 / 4) * r)
  (h3 : w + b ≥ 70) : r ≥ 72 :=
by
  sorry

end min_red_chips_l16_16208


namespace sum_of_squares_l16_16702

theorem sum_of_squares (x y : ℤ) (h : ∃ k : ℤ, (x^2 + y^2) = 5 * k) : 
  ∃ a b : ℤ, (x^2 + y^2) / 5 = a^2 + b^2 :=
by sorry

end sum_of_squares_l16_16702


namespace sector_area_is_nine_l16_16222

-- Defining the given conditions
def arc_length (r θ : ℝ) : ℝ := r * θ
def sector_area (r θ : ℝ) : ℝ := 0.5 * r^2 * θ

-- Given conditions
variables (r : ℝ) (θ : ℝ)
variable (h1 : arc_length r θ = 6)
variable (h2 : θ = 2)

-- Goal: Prove that the area of the sector is 9
theorem sector_area_is_nine : sector_area r θ = 9 := by
  sorry

end sector_area_is_nine_l16_16222


namespace degree_of_k_l16_16017

open Polynomial

theorem degree_of_k (h k : Polynomial ℝ) 
  (h_def : h = -5 * X^5 + 4 * X^3 - 2 * X^2 + C 8)
  (deg_sum : (h + k).degree = 2) : k.degree = 5 :=
sorry

end degree_of_k_l16_16017


namespace minimum_perimeter_l16_16029

-- Define the area condition
def area_condition (l w : ℝ) : Prop := l * w = 64

-- Define the perimeter function
def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w

-- The theorem statement based on the conditions and the correct answer
theorem minimum_perimeter (l w : ℝ) (h : area_condition l w) : 
  perimeter l w ≥ 32 := by
sorry

end minimum_perimeter_l16_16029


namespace heather_total_oranges_l16_16748

--Definition of the problem conditions
def initial_oranges : ℝ := 60.0
def additional_oranges : ℝ := 35.0

--Statement of the theorem
theorem heather_total_oranges : initial_oranges + additional_oranges = 95.0 := by
  sorry

end heather_total_oranges_l16_16748


namespace MathContestMeanMedianDifference_l16_16047

theorem MathContestMeanMedianDifference :
  (15 / 100 * 65 + 20 / 100 * 85 + 40 / 100 * 95 + 25 / 100 * 110) - 95 = -3 := 
by
  sorry

end MathContestMeanMedianDifference_l16_16047


namespace lisa_total_miles_flown_l16_16375

-- Definitions based on given conditions
def distance_per_trip : ℝ := 256.0
def number_of_trips : ℝ := 32.0
def total_miles_flown : ℝ := 8192.0

-- Lean statement asserting the equivalence
theorem lisa_total_miles_flown : 
    (distance_per_trip * number_of_trips = total_miles_flown) :=
by 
    sorry

end lisa_total_miles_flown_l16_16375


namespace fraction_walk_is_three_twentieths_l16_16460

-- Define the various fractions given in the conditions
def fraction_bus : ℚ := 1 / 2
def fraction_auto : ℚ := 1 / 4
def fraction_bicycle : ℚ := 1 / 10

-- Defining the total fraction for students that do not walk
def total_not_walk : ℚ := fraction_bus + fraction_auto + fraction_bicycle

-- The remaining fraction after subtracting from 1
def fraction_walk : ℚ := 1 - total_not_walk

-- The theorem we want to prove that fraction_walk is 3/20
theorem fraction_walk_is_three_twentieths : fraction_walk = 3 / 20 := by
  sorry

end fraction_walk_is_three_twentieths_l16_16460


namespace units_digit_7_pow_l16_16027

theorem units_digit_7_pow (n : ℕ) : 
  ∃ k, 7^n % 10 = k ∧ ((7^1 % 10 = 7) ∧ (7^2 % 10 = 9) ∧ (7^3 % 10 = 3) ∧ (7^4 % 10 = 1) ∧ (7^5 % 10 = 7)) → 
  7^2010 % 10 = 9 :=
by
  sorry

end units_digit_7_pow_l16_16027


namespace cows_C_grazed_l16_16972

/-- Define the conditions for each milkman’s cow-months. -/
def A_cow_months := 24 * 3
def B_cow_months := 10 * 5
def D_cow_months := 21 * 3
def C_cow_months (x : ℕ) := x * 4

/-- Define the cost per cow-month based on A's share. -/
def cost_per_cow_month := 720 / A_cow_months

/-- Define the total rent. -/
def total_rent := 3250

/-- Define the total cow-months including C's cow-months as a variable. -/
def total_cow_months (x : ℕ) := A_cow_months + B_cow_months + C_cow_months x + D_cow_months

/-- Lean 4 statement to prove the number of cows C grazed. -/
theorem cows_C_grazed (x : ℕ) :
  total_rent = total_cow_months x * cost_per_cow_month → x = 35 := by {
  sorry
}

end cows_C_grazed_l16_16972


namespace trig_identity_example_l16_16417

theorem trig_identity_example : 4 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 := 
by
  -- The statement "π/12" is mathematically equivalent to 15 degrees.
  sorry

end trig_identity_example_l16_16417


namespace initial_mean_calculated_l16_16187

theorem initial_mean_calculated (M : ℝ) (h1 : 25 * M - 35 = 25 * 191.4 - 35) : M = 191.4 := 
  sorry

end initial_mean_calculated_l16_16187


namespace altitude_length_l16_16665

noncomputable def length_of_altitude (l w : ℝ) : ℝ :=
  2 * l * w / Real.sqrt (l ^ 2 + w ^ 2)

theorem altitude_length (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  ∃ h : ℝ, h = length_of_altitude l w := by
  sorry

end altitude_length_l16_16665


namespace factorize_quadratic_l16_16311

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l16_16311


namespace volume_of_prism_l16_16207

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 48) (h3 : b * c = 72) : a * b * c = 168 :=
by
  sorry

end volume_of_prism_l16_16207


namespace simplify_trig_expression_l16_16040

theorem simplify_trig_expression :
  (2 - Real.sin 21 * Real.sin 21 - Real.cos 21 * Real.cos 21 + 
  (Real.sin 17 * Real.sin 17) * (Real.sin 17 * Real.sin 17) + 
  (Real.sin 17 * Real.sin 17) * (Real.cos 17 * Real.cos 17) + 
  (Real.cos 17 * Real.cos 17)) = 2 :=
by
  sorry

end simplify_trig_expression_l16_16040


namespace part_I_part_II_l16_16456

variable (x : ℝ)

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Define complement of B in real numbers
def neg_RB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Part I: Statement for a = -2
theorem part_I (a : ℝ) (h : a = -2) : A a ∩ neg_RB = {x | -1 ≤ x ∧ x ≤ 1} := by
  sorry

-- Part II: Statement for A ∪ B = B
theorem part_II (a : ℝ) (h : ∀ x, A a x -> B x) : a < -4 ∨ a > 5 := by
  sorry

end part_I_part_II_l16_16456


namespace milk_price_per_liter_l16_16049

theorem milk_price_per_liter (M : ℝ) 
  (price_fruit_per_kg : ℝ) (price_each_fruit_kg_eq_2: price_fruit_per_kg = 2)
  (milk_liters_per_batch : ℝ) (milk_liters_per_batch_eq_10: milk_liters_per_batch = 10)
  (fruit_kg_per_batch : ℝ) (fruit_kg_per_batch_eq_3 : fruit_kg_per_batch = 3)
  (cost_three_batches : ℝ) (cost_three_batches_eq_63: cost_three_batches = 63) :
  M = 1.5 :=
by
  sorry

end milk_price_per_liter_l16_16049


namespace NaCl_yield_l16_16312

structure Reaction :=
  (reactant1 : ℕ)
  (reactant2 : ℕ)
  (product : ℕ)

def NaOH := 3
def HCl := 3

theorem NaCl_yield : ∀ (R : Reaction), R.reactant1 = NaOH → R.reactant2 = HCl → R.product = 3 :=
by
  sorry

end NaCl_yield_l16_16312


namespace arithmetic_sequence_general_term_l16_16854

theorem arithmetic_sequence_general_term:
  ∃ (a : ℕ → ℕ), 
    (∀ n, a n + 1 > a n) ∧
    (a 1 = 2) ∧ 
    ((a 2) ^ 2 = a 5 + 6) ∧ 
    (∀ n, a n = 2 * n) :=
by
  sorry

end arithmetic_sequence_general_term_l16_16854


namespace smallest_integer_with_20_divisors_l16_16617

theorem smallest_integer_with_20_divisors : ∃ n : ℕ, 
  (0 < n) ∧ 
  (∀ m : ℕ, (0 < m ∧ ∃ k : ℕ, m = n * k) ↔ (∃ d : ℕ, d.succ * (20 / d.succ) = 20)) ∧ 
  n = 240 := 
sorry

end smallest_integer_with_20_divisors_l16_16617


namespace jill_peaches_l16_16016

open Nat

theorem jill_peaches (Jake Steven Jill : ℕ)
  (h1 : Jake = Steven - 6)
  (h2 : Steven = Jill + 18)
  (h3 : Jake = 17) :
  Jill = 5 := 
by
  sorry

end jill_peaches_l16_16016


namespace mindmaster_code_count_l16_16837

theorem mindmaster_code_count :
  let colors := 7
  let slots := 5
  (colors ^ slots) = 16807 :=
by
  -- Define the given conditions
  let colors := 7
  let slots := 5
  -- Proof statement to be inserted here
  sorry

end mindmaster_code_count_l16_16837


namespace find_k_l16_16408

theorem find_k : ∃ k : ℚ, (k = (k + 4) / 4) ∧ k = 4 / 3 :=
by
  sorry

end find_k_l16_16408


namespace div_pow_sub_one_l16_16926

theorem div_pow_sub_one (n : ℕ) (h : n > 1) : (n - 1) ^ 2 ∣ n ^ (n - 1) - 1 :=
sorry

end div_pow_sub_one_l16_16926


namespace email_difference_l16_16063

def morning_emails_early : ℕ := 10
def morning_emails_late : ℕ := 15
def afternoon_emails_early : ℕ := 7
def afternoon_emails_late : ℕ := 12

theorem email_difference :
  (morning_emails_early + morning_emails_late) - (afternoon_emails_early + afternoon_emails_late) = 6 :=
by
  sorry

end email_difference_l16_16063


namespace num_ways_write_100_as_distinct_squares_l16_16093

theorem num_ways_write_100_as_distinct_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a^2 + b^2 + c^2 = 100 ∧
  (∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x^2 + y^2 + z^2 = 100 ∧ (x, y, z) ≠ (a, b, c) ∧ (x, y, z) ≠ (a, c, b) ∧ (x, y, z) ≠ (b, a, c) ∧ (x, y, z) ≠ (b, c, a) ∧ (x, y, z) ≠ (c, a, b) ∧ (x, y, z) ≠ (c, b, a)) ∧
  ∀ (p q r : ℕ), p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p^2 + q^2 + r^2 = 100 → (p, q, r) = (a, b, c) ∨ (p, q, r) = (a, c, b) ∨ (p, q, r) = (b, a, c) ∨ (p, q, r) = (b, c, a) ∨ (p, q, r) = (c, a, b) ∨ (p, q, r) = (c, b, a) ∨ (p, q, r) = (x, y, z) ∨ (p, q, r) = (x, z, y) ∨ (p, q, r) = (y, x, z) ∨ (p, q, r) = (y, z, x) ∨ (p, q, r) = (z, x, y) ∨ (p, q, r) = (z, y, x) :=
sorry

end num_ways_write_100_as_distinct_squares_l16_16093


namespace largest_number_in_systematic_sample_l16_16776

theorem largest_number_in_systematic_sample (n_products : ℕ) (start : ℕ) (interval : ℕ) (sample_size : ℕ) (largest_number : ℕ)
  (h1 : n_products = 500)
  (h2 : start = 7)
  (h3 : interval = 25)
  (h4 : sample_size = n_products / interval)
  (h5 : sample_size = 20)
  (h6 : largest_number = start + interval * (sample_size - 1))
  (h7 : largest_number = 482) :
  largest_number = 482 := 
  sorry

end largest_number_in_systematic_sample_l16_16776


namespace center_of_circle_l16_16501

theorem center_of_circle : ∀ (x y : ℝ), x^2 + y^2 = 4 * x - 6 * y + 9 → (x, y) = (2, -3) :=
by
sorry

end center_of_circle_l16_16501


namespace cost_price_of_a_ball_l16_16355

variables (C : ℝ) (selling_price : ℝ) (cost_price_20_balls : ℝ) (loss_on_20_balls : ℝ)

def cost_price_per_ball (C : ℝ) := (20 * C - 720 = 5 * C)

theorem cost_price_of_a_ball :
  (∃ C : ℝ, 20 * C - 720 = 5 * C) -> (C = 48) := 
by
  sorry

end cost_price_of_a_ball_l16_16355


namespace eleanor_distance_between_meetings_l16_16988

-- Conditions given in the problem
def track_length : ℕ := 720
def eric_time : ℕ := 4
def eleanor_time : ℕ := 5
def eric_speed : ℕ := track_length / eric_time
def eleanor_speed : ℕ := track_length / eleanor_time
def relative_speed : ℕ := eric_speed + eleanor_speed
def time_to_meet : ℚ := track_length / relative_speed

-- Proof task: prove that the distance Eleanor runs between consective meetings is 320 meters.
theorem eleanor_distance_between_meetings : eleanor_speed * time_to_meet = 320 := by
  sorry

end eleanor_distance_between_meetings_l16_16988


namespace range_of_x_l16_16052

noncomputable def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
noncomputable def specific_function (f : ℝ → ℝ) := ∀ x : ℝ, x ≥ 0 → f x = 2^x

theorem range_of_x (f : ℝ → ℝ)  
  (hf_even : even_function f) 
  (hf_specific : specific_function f) : {x : ℝ | f (1 - 2 * x) < f 3} = {x : ℝ | -1 < x ∧ x < 2} := 
by
  sorry

end range_of_x_l16_16052


namespace men_required_l16_16726

theorem men_required (W M : ℕ) (h1 : M * 20 * W = W) (h2 : (M - 4) * 25 * W = W) : M = 16 := by
  sorry

end men_required_l16_16726


namespace min_time_to_one_ball_l16_16571

-- Define the problem in Lean
theorem min_time_to_one_ball (n : ℕ) (h : n = 99) : 
  ∃ T : ℕ, T = 98 ∧ ∀ t < T, ∃ ball_count : ℕ, ball_count > 1 :=
by
  -- Since we are not providing the proof, we use "sorry"
  sorry

end min_time_to_one_ball_l16_16571


namespace solution_set_f_le_1_l16_16064

variable {f : ℝ → ℝ}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem solution_set_f_le_1 :
  is_even_function f →
  monotone_on_nonneg f →
  f (-2) = 1 →
  {x : ℝ | f x ≤ 1} = {x : ℝ | -2 ≤ x ∧ x ≤ 2} :=
by
  intros h_even h_mono h_f_neg_2
  sorry

end solution_set_f_le_1_l16_16064


namespace economical_speed_l16_16669

variable (a k : ℝ)
variable (ha : 0 < a) (hk : 0 < k)

theorem economical_speed (v : ℝ) : 
  v = (a / (2 * k))^(1/3) :=
sorry

end economical_speed_l16_16669


namespace simplify_fraction_l16_16268

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2 + 1) = 16250 / 601 :=
by sorry

end simplify_fraction_l16_16268


namespace sub_neg_seven_eq_neg_fourteen_l16_16398

theorem sub_neg_seven_eq_neg_fourteen : (-7) - 7 = -14 := 
  by
  sorry

end sub_neg_seven_eq_neg_fourteen_l16_16398


namespace ratio_of_volumes_of_tetrahedrons_l16_16423

theorem ratio_of_volumes_of_tetrahedrons (a b : ℝ) (h : a / b = 1 / 2) : (a^3) / (b^3) = 1 / 8 :=
by
-- proof goes here
sorry

end ratio_of_volumes_of_tetrahedrons_l16_16423


namespace bus_dispatch_interval_l16_16158

/--
Xiao Hua walks at a constant speed along the route of the "Chunlei Cup" bus.
He encounters a "Chunlei Cup" bus every 6 minutes head-on and is overtaken by a "Chunlei Cup" bus every 12 minutes.
Assume "Chunlei Cup" buses are dispatched at regular intervals, travel at a constant speed, and do not stop at any stations along the way.
Prove that the time interval between bus departures is 8 minutes.
-/
theorem bus_dispatch_interval
  (encounters_opposite_direction: ℕ)
  (overtakes_same_direction: ℕ)
  (constant_speed: Prop)
  (regular_intervals: Prop)
  (no_stops: Prop)
  (h1: encounters_opposite_direction = 6)
  (h2: overtakes_same_direction = 12)
  (h3: constant_speed)
  (h4: regular_intervals)
  (h5: no_stops) :
  True := 
sorry

end bus_dispatch_interval_l16_16158


namespace calculate_value_l16_16059

theorem calculate_value : 15 * (1 / 3) + 45 * (2 / 3) = 35 := 
by
simp -- We use simp to simplify the expression
sorry -- We put sorry as we are skipping the full proof

end calculate_value_l16_16059


namespace solve_system_of_equations_l16_16189

theorem solve_system_of_equations (x y : ℤ) (h1 : x + y = 8) (h2 : x - 3 * y = 4) : x = 7 ∧ y = 1 :=
by {
    -- Proof would go here
    sorry
}

end solve_system_of_equations_l16_16189


namespace find_y_l16_16678

theorem find_y (y : ℤ) (h : (15 + 26 + y) / 3 = 23) : y = 28 :=
by sorry

end find_y_l16_16678


namespace solve_for_a_l16_16084

theorem solve_for_a (x a : ℝ) (h : 3 * x + 2 * a = 3) (hx : x = 5) : a = -6 :=
by
  sorry

end solve_for_a_l16_16084


namespace chenny_friends_l16_16755

theorem chenny_friends (initial_candies : ℕ) (needed_candies : ℕ) (candies_per_friend : ℕ) (h1 : initial_candies = 10) (h2 : needed_candies = 4) (h3 : candies_per_friend = 2) :
  (initial_candies + needed_candies) / candies_per_friend = 7 :=
by
  sorry

end chenny_friends_l16_16755


namespace factor_polynomial_l16_16715

theorem factor_polynomial : 
  (x : ℝ) → x^4 - 4 * x^2 + 16 = (x^2 - 4 * x + 4) * (x^2 + 2 * x + 4) :=
by
sorry

end factor_polynomial_l16_16715


namespace sum_of_coefficients_eq_one_l16_16476

theorem sum_of_coefficients_eq_one :
  ∀ x y : ℤ, (x - 2 * y) ^ 18 = (1 - 2 * 1) ^ 18 → (x - 2 * y) ^ 18 = 1 :=
by
  intros x y h
  sorry

end sum_of_coefficients_eq_one_l16_16476
