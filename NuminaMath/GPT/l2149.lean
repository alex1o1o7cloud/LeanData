import Mathlib

namespace arrangment_ways_basil_tomato_l2149_214974

theorem arrangment_ways_basil_tomato (basil_plants tomato_plants : Finset ℕ) 
  (hb : basil_plants.card = 5) 
  (ht : tomato_plants.card = 3) 
  : (∃ total_ways : ℕ, total_ways = 4320) :=
by
  sorry

end arrangment_ways_basil_tomato_l2149_214974


namespace friends_for_picnic_only_l2149_214995

theorem friends_for_picnic_only (M MP MG G PG A P : ℕ) 
(h1 : M + MP + MG + A = 10)
(h2 : G + MG + A = 5)
(h3 : MP = 4)
(h4 : MG = 2)
(h5 : PG = 0)
(h6 : A = 2)
(h7 : M + P + G + MP + MG + PG + A = 31) : 
    P = 20 := by {
  sorry
}

end friends_for_picnic_only_l2149_214995


namespace contradiction_example_l2149_214924

theorem contradiction_example (a b c : ℕ) : (¬ (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0)) → (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) :=
by
  sorry

end contradiction_example_l2149_214924


namespace n_times_s_l2149_214981

noncomputable def f (x : ℝ) : ℝ := sorry

theorem n_times_s : (f 0 = 0 ∨ f 0 = 1) ∧
  (∀ (y : ℝ), f 0 = 0 → False) ∧
  (∀ (x y : ℝ), f x * f y - f (x * y) = x^2 + y^2) → 
  let n : ℕ := if f 0 = 0 then 1 else 1
  let s : ℝ := if f 0 = 0 then 0 else 1
  n * s = 1 :=
by
  sorry

end n_times_s_l2149_214981


namespace negation_of_universal_proposition_l2149_214918

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x_0 : ℝ, x_0^2 < 0) := sorry

end negation_of_universal_proposition_l2149_214918


namespace correct_calculation_l2149_214919

theorem correct_calculation :
  - (1 / 2) - (- (1 / 3)) = - (1 / 6) :=
by
  sorry

end correct_calculation_l2149_214919


namespace find_a_in_geometric_sequence_l2149_214961

theorem find_a_in_geometric_sequence (S : ℕ → ℝ) (a : ℝ) :
  (∀ n, S n = 3^(n+1) + a) →
  (∃ a, ∀ n, S n = 3^(n+1) + a ∧ (18 : ℝ) ^ 2 = (S 1 - (S 1 - S 2)) * (S 2 - S 3) → a = -3) := 
by
  sorry

end find_a_in_geometric_sequence_l2149_214961


namespace certain_number_l2149_214967

theorem certain_number (x : ℝ) (h : 4 * x = 200) : x = 50 :=
by
  sorry

end certain_number_l2149_214967


namespace rhombus_side_length_l2149_214989

variable {L S : ℝ}

theorem rhombus_side_length (hL : 0 ≤ L) (hS : 0 ≤ S) :
  (∃ m : ℝ, m = 1 / 2 * Real.sqrt (L^2 - 4 * S)) :=
sorry

end rhombus_side_length_l2149_214989


namespace abs_diff_of_two_numbers_l2149_214948

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 := 
by
  sorry

end abs_diff_of_two_numbers_l2149_214948


namespace tinas_extra_earnings_l2149_214993

def price_per_candy_bar : ℕ := 2
def marvins_candy_bars_sold : ℕ := 35
def tinas_candy_bars_sold : ℕ := 3 * marvins_candy_bars_sold

def marvins_earnings : ℕ := marvins_candy_bars_sold * price_per_candy_bar
def tinas_earnings : ℕ := tinas_candy_bars_sold * price_per_candy_bar

theorem tinas_extra_earnings : tinas_earnings - marvins_earnings = 140 := by
  sorry

end tinas_extra_earnings_l2149_214993


namespace lacy_correct_percentage_is_80_l2149_214976

-- Define the total number of problems
def total_problems (x : ℕ) : ℕ := 5 * x + 10

-- Define the number of problems Lacy missed
def problems_missed (x : ℕ) : ℕ := x + 2

-- Define the number of problems Lacy answered correctly
def problems_answered (x : ℕ) : ℕ := total_problems x - problems_missed x

-- Define the fraction of problems Lacy answered correctly
def fraction_answered_correctly (x : ℕ) : ℚ :=
  (problems_answered x : ℚ) / (total_problems x : ℚ)

-- The main theorem to prove the percentage of problems correctly answered is 80%
theorem lacy_correct_percentage_is_80 (x : ℕ) : 
  fraction_answered_correctly x = 4 / 5 := 
by 
  sorry

end lacy_correct_percentage_is_80_l2149_214976


namespace solve_quadratic_l2149_214952

theorem solve_quadratic :
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ (x = 2 ∨ x = -1) :=
by
  sorry

end solve_quadratic_l2149_214952


namespace bells_ring_together_l2149_214941

open Nat

theorem bells_ring_together :
  let library_interval := 18
  let fire_station_interval := 24
  let hospital_interval := 30
  let start_time := 0
  let next_ring_time := Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval
  let total_minutes_in_an_hour := 60
  next_ring_time / total_minutes_in_an_hour = 6 :=
by
  let library_interval := 18
  let fire_station_interval := 24
  let hospital_interval := 30
  let start_time := 0
  let next_ring_time := Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval
  let total_minutes_in_an_hour := 60
  have h_next_ring_time : next_ring_time = 360 := by
    sorry
  have h_hours : next_ring_time / total_minutes_in_an_hour = 6 := by
    sorry
  exact h_hours

end bells_ring_together_l2149_214941


namespace find_f_of_half_l2149_214944

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_half : (∀ x : ℝ, f (Real.logb 4 x) = x) → f (1 / 2) = 2 :=
by
  intros h
  have h1 := h (4 ^ (1 / 2))
  sorry

end find_f_of_half_l2149_214944


namespace math_problem_l2149_214912

open Real

theorem math_problem (α : ℝ) (h₁ : 0 < α) (h₂ : α < π / 2) (h₃ : cos (2 * π - α) - sin (π - α) = - sqrt 5 / 5) :
  (sin α + cos α = 3 * sqrt 5 / 5) ∧
  (cos (3 * π / 2 + α) ^ 2 + 2 * cos α * cos (π / 2 - α)) / (1 + sin (π / 2 - α) ^ 2) = 4 / 3 :=
by
  sorry

end math_problem_l2149_214912


namespace prime_squared_mod_six_l2149_214992

theorem prime_squared_mod_six (p : ℕ) (hp1 : p > 5) (hp2 : Nat.Prime p) : (p ^ 2) % 6 = 1 :=
sorry

end prime_squared_mod_six_l2149_214992


namespace range_of_m_l2149_214999

/-- The point (m^2, m) is within the planar region defined by x - 3y + 2 > 0. 
    Find the range of m. -/
theorem range_of_m {m : ℝ} : (m^2 - 3 * m + 2 > 0) ↔ (m < 1 ∨ m > 2) := 
by 
  sorry

end range_of_m_l2149_214999


namespace regular_polygon_sides_160_l2149_214968

theorem regular_polygon_sides_160 (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle : ℝ) = 160) : 
  n = 18 :=
by
  sorry

end regular_polygon_sides_160_l2149_214968


namespace adam_earnings_per_lawn_l2149_214953

theorem adam_earnings_per_lawn (total_lawns : ℕ) (forgot_lawns : ℕ) (total_earnings : ℕ) :
  total_lawns = 12 →
  forgot_lawns = 8 →
  total_earnings = 36 →
  (total_earnings / (total_lawns - forgot_lawns)) = 9 :=
by
  intros h1 h2 h3
  sorry

end adam_earnings_per_lawn_l2149_214953


namespace find_number_l2149_214946

variable (x : ℝ)

theorem find_number : ((x * 5) / 2.5 - 8 * 2.25 = 5.5) -> x = 11.75 :=
by
  intro h
  sorry

end find_number_l2149_214946


namespace cos_x_plus_2y_is_one_l2149_214937

theorem cos_x_plus_2y_is_one
    (x y : ℝ) (a : ℝ) 
    (hx : x ∈ Set.Icc (-Real.pi) Real.pi)
    (hy : y ∈ Set.Icc (-Real.pi) Real.pi)
    (h_eq : 2 * a = x ^ 3 + Real.sin x ∧ 2 * a = (-2 * y) ^ 3 - Real.sin (-2 * y)) :
    Real.cos (x + 2 * y) = 1 := 
sorry

end cos_x_plus_2y_is_one_l2149_214937


namespace sum_of_numbers_l2149_214942

theorem sum_of_numbers :
  145 + 35 + 25 + 5 = 210 :=
by
  sorry

end sum_of_numbers_l2149_214942


namespace gcd_example_l2149_214914

theorem gcd_example : Nat.gcd 8675309 7654321 = 36 := sorry

end gcd_example_l2149_214914


namespace florist_has_56_roses_l2149_214923

def initial_roses := 50
def roses_sold := 15
def roses_picked := 21

theorem florist_has_56_roses (r0 rs rp : ℕ) (h1 : r0 = initial_roses) (h2 : rs = roses_sold) (h3 : rp = roses_picked) : 
  r0 - rs + rp = 56 :=
by sorry

end florist_has_56_roses_l2149_214923


namespace min_value_proof_l2149_214906

noncomputable def min_value (a b : ℝ) : ℝ := (1 : ℝ)/a + (1 : ℝ)/b

theorem min_value_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + 2 * b = 2) :
  min_value a b = (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_proof_l2149_214906


namespace min_value_of_expr_min_value_achieved_final_statement_l2149_214935

theorem min_value_of_expr (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 3) :
  1 ≤ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by
  sorry

theorem min_value_achieved (x y z : ℝ) (h1 : x = 1) (h2 : y = 1) (h3 : z = 1) :
  1 = (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by
  sorry

theorem final_statement (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 3) :
  ∃ (x y z : ℝ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x + y + z = 3) ∧ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) = 1) :=
by
  sorry

end min_value_of_expr_min_value_achieved_final_statement_l2149_214935


namespace machine_a_production_rate_l2149_214921

/-
Given:
1. Machine p and machine q are each used to manufacture 440 sprockets.
2. Machine q produces 10% more sprockets per hour than machine a.
3. It takes machine p 10 hours longer to produce 440 sprockets than machine q.

Prove that machine a produces 4 sprockets per hour.
-/

theorem machine_a_production_rate (T A : ℝ) (hq : 440 = T * (1.1 * A)) (hp : 440 = (T + 10) * A) : A = 4 := 
by
  sorry

end machine_a_production_rate_l2149_214921


namespace f_positive_l2149_214905

variable (f : ℝ → ℝ)

-- f is a differentiable function on ℝ
variable (hf : differentiable ℝ f)

-- Condition: (x+1)f(x) + x f''(x) > 0
variable (H : ∀ x, (x + 1) * f x + x * (deriv^[2]) f x > 0)

-- Prove: ∀ x, f x > 0
theorem f_positive : ∀ x, f x > 0 := 
by
  sorry

end f_positive_l2149_214905


namespace diamond_evaluation_l2149_214990

def diamond (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem diamond_evaluation : diamond 2 (diamond 3 (diamond 1 4)) = -46652 :=
  by
  sorry

end diamond_evaluation_l2149_214990


namespace greatest_large_chips_l2149_214957

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 2 ≤ a ∧ 2 ≤ b ∧ n = a * b

theorem greatest_large_chips (s l : ℕ) (c : ℕ) (hc : is_composite c) (h : s + l = 60) (hs : s = l + c) :
  l ≤ 28 :=
sorry

end greatest_large_chips_l2149_214957


namespace handshakes_count_l2149_214979

-- Define the number of people
def num_people : ℕ := 10

-- Define a function to calculate the number of handshakes
noncomputable def num_handshakes (n : ℕ) : ℕ :=
  (n - 1) * n / 2

-- The main statement to be proved
theorem handshakes_count : num_handshakes num_people = 45 := by
  -- Proof will be filled in here
  sorry

end handshakes_count_l2149_214979


namespace length_of_AB_l2149_214972

noncomputable def parabola_intersection (x1 x2 : ℝ) (y1 y2 : ℝ) : ℝ :=
|x1 - x2|

theorem length_of_AB : 
  ∀ (x1 x2 y1 y2 : ℝ),
    (x1 + x2 = 6) →
    (A = (x1, y1)) →
    (B = (x2, y2)) →
    (y1^2 = 4 * x1) →
    (y2^2 = 4 * x2) →
    parabola_intersection x1 x2 y1 y2 = 8 :=
by
  sorry

end length_of_AB_l2149_214972


namespace union_M_N_eq_U_l2149_214933

def U : Set Nat := {2, 3, 4, 5, 6, 7}
def M : Set Nat := {3, 4, 5, 7}
def N : Set Nat := {2, 4, 5, 6}

theorem union_M_N_eq_U : M ∪ N = U := 
by {
  -- Proof would go here
  sorry
}

end union_M_N_eq_U_l2149_214933


namespace arithmetic_sequence_sum_nine_l2149_214915

variable {a : ℕ → ℤ} -- Define a_n sequence as a function from ℕ to ℤ

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ n m, a (n + m) = a n + m * d

def fifth_term_is_two (a : ℕ → ℤ) : Prop :=
  a 5 = 2

-- Lean statement to prove the sum of the first 9 terms
theorem arithmetic_sequence_sum_nine (a : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : fifth_term_is_two a) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 :=
sorry

end arithmetic_sequence_sum_nine_l2149_214915


namespace greatest_divisor_of_28_l2149_214907

theorem greatest_divisor_of_28 : ∀ d : ℕ, d ∣ 28 → d ≤ 28 :=
by
  sorry

end greatest_divisor_of_28_l2149_214907


namespace log8_512_is_3_l2149_214998

def log_base_8_of_512 : Prop :=
  ∀ (log8 : ℝ → ℝ),
    (log8 8 = 1 / 3 * log8 2) →
    (log8 512 = 9 * log8 2) →
    log8 8 = 3 → log8 512 = 3

theorem log8_512_is_3 : log_base_8_of_512 :=
by
  intros log8 H1 H2 H3
  -- here you would normally provide the detailed steps to solve this.
  -- however, we directly proclaim the result due to the proof being non-trivial.
  sorry

end log8_512_is_3_l2149_214998


namespace solve_eq_l2149_214978

-- Defining the condition
def eq_condition (x : ℝ) : Prop := (x - 3) ^ 2 = x ^ 2 - 9

-- The statement we need to prove
theorem solve_eq (x : ℝ) (h : eq_condition x) : x = 3 :=
by
  sorry

end solve_eq_l2149_214978


namespace directrix_of_parabola_l2149_214902

def parabola_directrix (x_y_eqn : ℝ → ℝ) : ℝ := by
  -- Assuming the parabola equation x = -(1/4) y^2
  sorry

theorem directrix_of_parabola : parabola_directrix (fun y => -(1/4) * y^2) = 1 := by
  sorry

end directrix_of_parabola_l2149_214902


namespace wrapping_third_roll_l2149_214934

theorem wrapping_third_roll (total_gifts first_roll_gifts second_roll_gifts third_roll_gifts : ℕ) 
  (h1 : total_gifts = 12) (h2 : first_roll_gifts = 3) (h3 : second_roll_gifts = 5) 
  (h4 : third_roll_gifts = total_gifts - (first_roll_gifts + second_roll_gifts)) :
  third_roll_gifts = 4 :=
sorry

end wrapping_third_roll_l2149_214934


namespace contracting_arrangements_1680_l2149_214994

def num_contracting_arrangements (n a b c d : ℕ) : ℕ :=
  Nat.choose n a * Nat.choose (n - a) b * Nat.choose (n - a - b) c

theorem contracting_arrangements_1680 : num_contracting_arrangements 8 3 1 2 2 = 1680 := by
  unfold num_contracting_arrangements
  simp
  sorry

end contracting_arrangements_1680_l2149_214994


namespace men_l2149_214958

namespace WagesProblem

def men_women_boys_equivalence (man woman boy : ℕ) : Prop :=
  9 * man = woman ∧ woman = 7 * boy

def total_earnings (man woman boy earnings : ℕ) : Prop :=
  (9 * man + woman + woman) = earnings ∧ earnings = 216

theorem men's_wages (man woman boy : ℕ) (h1 : men_women_boys_equivalence man woman boy) (h2 : total_earnings man woman 7 216) : 9 * man = 72 :=
sorry

end WagesProblem

end men_l2149_214958


namespace certain_event_l2149_214949

-- Definitions of the events
def event1 : Prop := ∀ (P : ℝ), P ≠ 20.0
def event2 : Prop := ∀ (x : ℤ), x ≠ 105 ∧ x ≤ 100
def event3 : Prop := ∃ (r : ℝ), 0 ≤ r ∧ r ≤ 1 ∧ ¬(r = 0 ∨ r = 1)
def event4 (a b : ℝ) : Prop := ∃ (area : ℝ), area = a * b

-- Statement to prove that event4 is the only certain event
theorem certain_event (a b : ℝ) : (event4 a b) := 
by
  sorry

end certain_event_l2149_214949


namespace find_missing_number_l2149_214997

theorem find_missing_number 
  (x : ℝ) (y : ℝ)
  (h1 : (12 + x + 42 + 78 + 104) / 5 = 62)
  (h2 : (128 + 255 + y + 1023 + x) / 5 = 398.2) :
  y = 511 := 
sorry

end find_missing_number_l2149_214997


namespace highest_water_level_changes_on_tuesday_l2149_214928

def water_levels : List (String × Float) :=
  [("Monday", 0.03), ("Tuesday", 0.41), ("Wednesday", 0.25), ("Thursday", 0.10),
   ("Friday", 0.0), ("Saturday", -0.13), ("Sunday", -0.2)]

theorem highest_water_level_changes_on_tuesday :
  ∃ d : String, d = "Tuesday" ∧ ∀ d' : String × Float, d' ∈ water_levels → d'.snd ≤ 0.41 := by
  sorry

end highest_water_level_changes_on_tuesday_l2149_214928


namespace shape_is_plane_l2149_214951

-- Define cylindrical coordinates
structure CylindricalCoord :=
  (r : ℝ) (theta : ℝ) (z : ℝ)

-- Define the condition
def condition (c : ℝ) (coord : CylindricalCoord) : Prop :=
  coord.z = c

-- The shape is described as a plane
def is_plane : Prop := ∀ (coord1 coord2 : CylindricalCoord), (coord1.z = coord2.z)

theorem shape_is_plane (c : ℝ) : 
  (∀ coord : CylindricalCoord, condition c coord) ↔ is_plane :=
by 
  sorry

end shape_is_plane_l2149_214951


namespace not_perfect_square_l2149_214925

theorem not_perfect_square (n : ℤ) : ¬ ∃ (m : ℤ), 4*n + 3 = m^2 := 
by 
  sorry

end not_perfect_square_l2149_214925


namespace xy_difference_l2149_214955

theorem xy_difference (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 12) : x - y = 2 := by
  sorry

end xy_difference_l2149_214955


namespace smallest_sum_a_b_l2149_214969

theorem smallest_sum_a_b :
  ∃ (a b : ℕ), (7 * b - 4 * a = 3) ∧ a > 7 ∧ b > 7 ∧ a + b = 24 :=
by
  sorry

end smallest_sum_a_b_l2149_214969


namespace bills_difference_l2149_214930

noncomputable def Mike_tip : ℝ := 5
noncomputable def Joe_tip : ℝ := 10
noncomputable def Mike_percentage : ℝ := 20
noncomputable def Joe_percentage : ℝ := 25

theorem bills_difference
  (m j : ℝ)
  (Mike_condition : (Mike_percentage / 100) * m = Mike_tip)
  (Joe_condition : (Joe_percentage / 100) * j = Joe_tip) :
  |m - j| = 15 :=
by
  sorry

end bills_difference_l2149_214930


namespace simplify_and_evaluate_l2149_214988

theorem simplify_and_evaluate (a : ℤ) (h : a = -2) : 
  (1 - (1 / (a + 1))) / ((a^2 - 2*a + 1) / (a^2 - 1)) = (2 / 3) :=
by
  sorry

end simplify_and_evaluate_l2149_214988


namespace max_abs_f_le_f0_f1_l2149_214956

noncomputable def f (a b x : ℝ) : ℝ := 3 * a * x^2 - 2 * (a + b) * x + b

theorem max_abs_f_le_f0_f1 (a b : ℝ) (h : 0 < a) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) :
  |f a b x| ≤ max (|f a b 0|) (|f a b 1|) :=
sorry

end max_abs_f_le_f0_f1_l2149_214956


namespace carol_twice_as_cathy_l2149_214984

-- Define variables for the number of cars each person owns
variables (C L S Ca x : ℕ)

-- Define conditions based on the problem statement
def lindsey_cars := L = C + 4
def susan_cars := S = Ca - 2
def carol_cars := Ca = 2 * x
def total_cars := C + L + S + Ca = 32
def cathy_cars := C = 5

-- State the theorem to prove
theorem carol_twice_as_cathy : 
  lindsey_cars C L ∧ 
  susan_cars S Ca ∧ 
  carol_cars Ca x ∧ 
  total_cars C L S Ca ∧ 
  cathy_cars C
  → x = 5 :=
by
  sorry

end carol_twice_as_cathy_l2149_214984


namespace isosceles_triangle_perimeter_l2149_214975

noncomputable def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem isosceles_triangle_perimeter {a b c : ℕ} (h1 : is_isosceles_triangle a b c) (h2 : a = 3 ∨ a = 6)
  (h3 : b = 3 ∨ b = 6) (h4 : c = 3 ∨ c = 6) (h5 : a + b + c = 15) : a + b + c = 15 :=
by
  sorry

end isosceles_triangle_perimeter_l2149_214975


namespace no_prime_roots_of_quadratic_l2149_214900

open Int Nat

theorem no_prime_roots_of_quadratic (k : ℤ) :
  ¬ (∃ p q : ℤ, Prime p ∧ Prime q ∧ p + q = 107 ∧ p * q = k) :=
by
  sorry

end no_prime_roots_of_quadratic_l2149_214900


namespace disjunction_of_p_and_q_l2149_214991

-- Define the propositions p and q
variable (p q : Prop)

-- Assume that p is true and q is false
theorem disjunction_of_p_and_q (h1 : p) (h2 : ¬q) : p ∨ q := 
sorry

end disjunction_of_p_and_q_l2149_214991


namespace union_M_N_is_R_l2149_214922

open Set

/-- Define the sets M and N -/
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x < 3}

/-- Main goal: prove M ∪ N = ℝ -/
theorem union_M_N_is_R : M ∪ N = univ :=
by
  sorry

end union_M_N_is_R_l2149_214922


namespace least_number_of_groups_l2149_214913

theorem least_number_of_groups (total_players : ℕ) (max_per_group : ℕ) (h1 : total_players = 30) (h2 : max_per_group = 12) : ∃ (groups : ℕ), groups = 3 := 
by {
  -- Mathematical conditions and solution to be formalized here
  sorry
}

end least_number_of_groups_l2149_214913


namespace number_of_buses_l2149_214916

theorem number_of_buses (x y : ℕ) (h1 : x + y = 40) (h2 : 6 * x + 4 * y = 210) : x = 25 :=
by
  sorry

end number_of_buses_l2149_214916


namespace jose_total_caps_l2149_214908

def initial_caps := 26
def additional_caps := 13
def total_caps := initial_caps + additional_caps

theorem jose_total_caps : total_caps = 39 :=
by
  sorry

end jose_total_caps_l2149_214908


namespace rhombus_area_l2149_214971

theorem rhombus_area (s : ℝ) (d1 d2 : ℝ) (h1 : s = Real.sqrt 145) (h2 : abs (d1 - d2) = 10) : 
  (1/2) * d1 * d2 = 100 :=
sorry

end rhombus_area_l2149_214971


namespace probability_of_blank_l2149_214917

-- Definitions based on conditions
def num_prizes : ℕ := 10
def num_blanks : ℕ := 25
def total_outcomes : ℕ := num_prizes + num_blanks

-- Statement of the proof problem
theorem probability_of_blank : (num_blanks / total_outcomes : ℚ) = 5 / 7 :=
by {
  sorry
}

end probability_of_blank_l2149_214917


namespace complement_intersection_l2149_214987

open Set

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {-1, 0, 1, 3}
def N : Set Int := {-2, 0, 2, 3}

theorem complement_intersection :
  ((U \ M) ∩ N = {-2, 2}) :=
by sorry

end complement_intersection_l2149_214987


namespace obtuse_triangle_l2149_214903

variable (A B C : ℝ)
variable (angle_sum : A + B + C = 180)
variable (cond1 : A + B = 141)
variable (cond2 : B + C = 165)

theorem obtuse_triangle : B > 90 :=
by
  sorry

end obtuse_triangle_l2149_214903


namespace decreasing_direct_proportion_l2149_214985

theorem decreasing_direct_proportion (k : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → k * x1 > k * x2) : k < 0 :=
by
  sorry

end decreasing_direct_proportion_l2149_214985


namespace mooncake_inspection_random_event_l2149_214939

-- Definition of event categories
inductive Event
| certain
| impossible
| random

-- Definition of the event in question
def mooncakeInspectionEvent (satisfactory: Bool) : Event :=
if satisfactory then Event.random else Event.random

-- Theorem statement to prove that the event is a random event
theorem mooncake_inspection_random_event (satisfactory: Bool) :
  mooncakeInspectionEvent satisfactory = Event.random :=
sorry

end mooncake_inspection_random_event_l2149_214939


namespace translation_symmetric_y_axis_phi_l2149_214986

theorem translation_symmetric_y_axis_phi :
  ∀ (f : ℝ → ℝ) (φ : ℝ),
    (∀ x : ℝ, f x = Real.sin (2 * x + π / 6)) →
    (0 < φ ∧ φ ≤ π / 2) →
    (∀ x, Real.sin (2 * (x + φ) + π / 6) = Real.sin (2 * (-x + φ) + π / 6)) →
    φ = π / 6 :=
by
  intros f φ f_def φ_bounds symmetry
  sorry

end translation_symmetric_y_axis_phi_l2149_214986


namespace carrie_bought_t_shirts_l2149_214966

theorem carrie_bought_t_shirts (total_spent : ℝ) (cost_each : ℝ) (n : ℕ) 
    (h_total : total_spent = 199) (h_cost : cost_each = 9.95) 
    (h_eq : n = total_spent / cost_each) : n = 20 := 
by
sorry

end carrie_bought_t_shirts_l2149_214966


namespace ratio_a_d_l2149_214927

variables (a b c d : ℕ)

-- Given conditions
def ratio_ab := 8 / 3
def ratio_bc := 1 / 5
def ratio_cd := 3 / 2
def b_value := 27

theorem ratio_a_d (h₁ : a / b = ratio_ab)
                  (h₂ : b / c = ratio_bc)
                  (h₃ : c / d = ratio_cd)
                  (h₄ : b = b_value) :
  a / d = 4 / 5 :=
sorry

end ratio_a_d_l2149_214927


namespace intersecting_lines_l2149_214932

theorem intersecting_lines (a b : ℚ) :
  (3 = (1 / 3 : ℚ) * 4 + a) → 
  (4 = (1 / 2 : ℚ) * 3 + b) → 
  a + b = 25 / 6 :=
by
  intros h1 h2
  sorry

end intersecting_lines_l2149_214932


namespace ball_bounce_l2149_214960

theorem ball_bounce :
  ∃ b : ℕ, 324 * (3 / 4) ^ b < 40 ∧ b = 8 :=
by
  have : (3 / 4 : ℝ) < 1 := by norm_num
  have h40_324 : (40 : ℝ) / 324 = 10 / 81 := by norm_num
  sorry

end ball_bounce_l2149_214960


namespace negation_proposition_equiv_l2149_214980

variable (m : ℤ)

theorem negation_proposition_equiv :
  (¬ ∃ x : ℤ, x^2 + x + m < 0) ↔ (∀ x : ℤ, x^2 + x + m ≥ 0) :=
by
  sorry

end negation_proposition_equiv_l2149_214980


namespace terminal_side_in_fourth_quadrant_l2149_214940

theorem terminal_side_in_fourth_quadrant 
  (h_sin_half : Real.sin (α / 2) = 3 / 5)
  (h_cos_half : Real.cos (α / 2) = -4 / 5) : 
  (Real.sin α < 0) ∧ (Real.cos α > 0) :=
by
  sorry

end terminal_side_in_fourth_quadrant_l2149_214940


namespace MichelangeloCeilingPainting_l2149_214977

theorem MichelangeloCeilingPainting (total_ceiling week1_ceiling next_week_fraction : ℕ) 
  (a1 : total_ceiling = 28) 
  (a2 : week1_ceiling = 12) 
  (a3 : total_ceiling - (week1_ceiling + next_week_fraction * week1_ceiling) = 13) : 
  next_week_fraction = 1 / 4 := 
by 
  sorry

end MichelangeloCeilingPainting_l2149_214977


namespace ab_multiple_of_7_2010_l2149_214931

theorem ab_multiple_of_7_2010 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 7 ^ 2009 ∣ a^2 + b^2) : 7 ^ 2010 ∣ a * b :=
by
  sorry

end ab_multiple_of_7_2010_l2149_214931


namespace sum_of_midpoints_eq_15_l2149_214962

theorem sum_of_midpoints_eq_15 (a b c d : ℝ) (h : a + b + c + d = 15) :
  (a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2 = 15 :=
by sorry

end sum_of_midpoints_eq_15_l2149_214962


namespace inverse_function_solution_l2149_214964

noncomputable def f (a b x : ℝ) := 2 / (a * x + b)

theorem inverse_function_solution (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : f a b 2 = 1 / 2) : b = 1 - 2 * a :=
by
  -- Assuming the inverse function condition means f(2) should be evaluated.
  sorry

end inverse_function_solution_l2149_214964


namespace max_median_soda_cans_l2149_214938

theorem max_median_soda_cans (total_customers total_cans : ℕ) 
    (h_customers : total_customers = 120)
    (h_cans : total_cans = 300) 
    (h_min_cans_per_customer : ∀ (n : ℕ), n < total_customers → 2 ≤ n) :
    ∃ (median : ℝ), median = 3.5 := 
sorry

end max_median_soda_cans_l2149_214938


namespace units_digit_fraction_l2149_214963

theorem units_digit_fraction :
  (30 * 31 * 32 * 33 * 34 * 35) % 10 = (2500) % 10 → 
  ((30 * 31 * 32 * 33 * 34 * 35) / 2500) % 10 = 1 := 
by 
  intro h
  sorry

end units_digit_fraction_l2149_214963


namespace distance_between_homes_l2149_214970

def speed (name : String) : ℝ :=
  if name = "Maxwell" then 4
  else if name = "Brad" then 6
  else 0

def meeting_time : ℝ := 4

def delay : ℝ := 1

def distance_covered (name : String) : ℝ :=
  if name = "Maxwell" then speed name * meeting_time
  else if name = "Brad" then speed name * (meeting_time - delay)
  else 0

def total_distance : ℝ :=
  distance_covered "Maxwell" + distance_covered "Brad"

theorem distance_between_homes : total_distance = 34 :=
by
  -- proof goes here
  sorry

end distance_between_homes_l2149_214970


namespace exists_consecutive_non_primes_l2149_214911

theorem exists_consecutive_non_primes (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℕ, ∀ i : ℕ, i < k → ¬Nat.Prime (n + i) := 
sorry

end exists_consecutive_non_primes_l2149_214911


namespace faulty_balance_inequality_l2149_214926

variable (m n a b G : ℝ)

theorem faulty_balance_inequality
  (h1 : m * a = n * G)
  (h2 : n * b = m * G) :
  (a + b) / 2 > G :=
sorry

end faulty_balance_inequality_l2149_214926


namespace shaded_area_eq_l2149_214947

theorem shaded_area_eq : 
  let side := 8 
  let radius := 3 
  let square_area := side * side
  let sector_area := (1 / 4) * Real.pi * (radius * radius)
  let four_sectors_area := 4 * sector_area
  let triangle_area := (1 / 2) * radius * radius
  let four_triangles_area := 4 * triangle_area
  let shaded_area := square_area - four_sectors_area - four_triangles_area
  shaded_area = 64 - 9 * Real.pi - 18 :=
by
  sorry

end shaded_area_eq_l2149_214947


namespace blocks_to_store_l2149_214910

theorem blocks_to_store
  (T : ℕ) (S : ℕ)
  (hT : T = 25)
  (h_total_walk : S + 6 + 8 = T) :
  S = 11 :=
by
  sorry

end blocks_to_store_l2149_214910


namespace janele_cats_average_weight_l2149_214929

noncomputable def average_weight_cats (w1 w2 w3 w4 : ℝ) : ℝ :=
  (w1 + w2 + w3 + w4) / 4

theorem janele_cats_average_weight :
  average_weight_cats 12 12 14.7 9.3 = 12 :=
by
  sorry

end janele_cats_average_weight_l2149_214929


namespace fewer_hours_l2149_214909

noncomputable def distance : ℝ := 300
noncomputable def speed_T : ℝ := 20
noncomputable def speed_A : ℝ := speed_T + 5

theorem fewer_hours (d : ℝ) (V_T : ℝ) (V_A : ℝ) :
    V_T = 20 ∧ V_A = V_T + 5 ∧ d = 300 → (d / V_T) - (d / V_A) = 3 := 
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end fewer_hours_l2149_214909


namespace find_19a_20b_21c_l2149_214920

theorem find_19a_20b_21c (a b c : ℕ) (h₁ : 29 * a + 30 * b + 31 * c = 366) 
  (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : 19 * a + 20 * b + 21 * c = 246 := 
sorry

end find_19a_20b_21c_l2149_214920


namespace ratio_of_men_to_women_l2149_214996

theorem ratio_of_men_to_women (C W M : ℕ) 
  (hC : C = 30) 
  (hW : W = 3 * C) 
  (hTotal : M + W + C = 300) : 
  M / W = 2 :=
by
  sorry

end ratio_of_men_to_women_l2149_214996


namespace circle_locus_l2149_214901

theorem circle_locus (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (5 - r)^2)) ↔ 
  13 * a^2 + 49 * b^2 - 12 * a - 1 = 0 := 
sorry

end circle_locus_l2149_214901


namespace simplify_expression_l2149_214983

theorem simplify_expression : 1 - (1 / (1 + Real.sqrt 2)) + (1 / (1 - Real.sqrt 2)) = 1 - 2 * Real.sqrt 2 := by
  sorry

end simplify_expression_l2149_214983


namespace root_difference_geom_prog_l2149_214945

theorem root_difference_geom_prog
  (x1 x2 x3 : ℝ)
  (h1 : 8 * x1^3 - 22 * x1^2 + 15 * x1 - 2 = 0)
  (h2 : 8 * x2^3 - 22 * x2^2 + 15 * x2 - 2 = 0)
  (h3 : 8 * x3^3 - 22 * x3^2 + 15 * x3 - 2 = 0)
  (geom_prog : ∃ (a r : ℝ), x1 = a / r ∧ x2 = a ∧ x3 = a * r) :
  |x3 - x1| = 33 / 14 :=
by
  sorry

end root_difference_geom_prog_l2149_214945


namespace find_missing_part_l2149_214965

variable (x y : ℚ) -- Using rationals as the base field for generality.

theorem find_missing_part :
  2 * x * (-3 * x^2 * y) = -6 * x^3 * y := 
by
  sorry

end find_missing_part_l2149_214965


namespace total_number_of_pages_l2149_214943

variable (x : ℕ)

-- Conditions
def first_day_remaining : ℕ := x - (x / 6 + 10)
def second_day_remaining : ℕ := first_day_remaining x - (first_day_remaining x / 5 + 20)
def third_day_remaining : ℕ := second_day_remaining x - (second_day_remaining x / 4 + 25)
def final_remaining : Prop := third_day_remaining x = 100

-- Theorem statement
theorem total_number_of_pages : final_remaining x → x = 298 :=
by
  intros h
  sorry

end total_number_of_pages_l2149_214943


namespace positiveDifferenceEquation_l2149_214959

noncomputable def positiveDifference (x y : ℝ) : ℝ := |y - x|

theorem positiveDifferenceEquation (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  positiveDifference x y = 60 / 7 :=
by
  sorry

end positiveDifferenceEquation_l2149_214959


namespace min_value_fraction_l2149_214950

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  (4 / x + 9 / y) ≥ 25 :=
sorry

end min_value_fraction_l2149_214950


namespace problem_number_eq_7_5_l2149_214904

noncomputable def number : ℝ := 7.5

theorem problem_number_eq_7_5 :
  ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = number :=
by
  sorry

end problem_number_eq_7_5_l2149_214904


namespace ellipse_condition_l2149_214973

theorem ellipse_condition (m : ℝ) :
  (m > 0) ∧ (2 * m - 1 > 0) ∧ (m ≠ 2 * m - 1) ↔ (m > 1/2) ∧ (m ≠ 1) :=
by
  sorry

end ellipse_condition_l2149_214973


namespace negation_proposition_l2149_214982

theorem negation_proposition:
  (¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) :=
by sorry

end negation_proposition_l2149_214982


namespace problem_l2149_214954

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem problem
  (ω : ℝ) 
  (hω : ω > 0)
  (hab : Real.sqrt (4 + (Real.pi ^ 2) / (ω ^ 2)) = 2 * Real.sqrt 2) :
  f ω 1 = Real.sqrt 3 / 2 := 
sorry

end problem_l2149_214954


namespace intersection_eq_zero_set_l2149_214936

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | x^2 ≤ 0}

theorem intersection_eq_zero_set : M ∩ N = {0} := by
  sorry

end intersection_eq_zero_set_l2149_214936
