import Mathlib

namespace max_value_of_expression_l278_27853

noncomputable def max_value_expr (a b c : ℝ) : ℝ :=
  a + b^2 + c^3

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  max_value_expr a b c ≤ 8 :=
  sorry

end max_value_of_expression_l278_27853


namespace mod_remainder_of_sum_of_primes_l278_27821

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def sum_of_odd_primes : ℕ := List.sum odd_primes_less_than_32

theorem mod_remainder_of_sum_of_primes : sum_of_odd_primes % 32 = 30 := by
  sorry

end mod_remainder_of_sum_of_primes_l278_27821


namespace obtuse_triangle_existence_l278_27841

theorem obtuse_triangle_existence :
  ∃ (a b c : ℝ), (a = 2 ∧ b = 6 ∧ c = 7 ∧ 
  (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2)) ∧
  ¬(6^2 + 7^2 < 8^2 ∨ 7^2 + 8^2 < 6^2 ∨ 8^2 + 6^2 < 7^2) ∧
  ¬(7^2 + 8^2 < 10^2 ∨ 8^2 + 10^2 < 7^2 ∨ 10^2 + 7^2 < 8^2) ∧
  ¬(5^2 + 12^2 < 13^2 ∨ 12^2 + 13^2 < 5^2 ∨ 13^2 + 5^2 < 12^2) :=
sorry

end obtuse_triangle_existence_l278_27841


namespace regular_polygon_properties_l278_27814

theorem regular_polygon_properties
  (n : ℕ)
  (h1 : (n - 2) * 180 = 3 * 360 + 180)
  (h2 : n > 2) :
  n = 9 ∧ (n - 2) * 180 / n = 140 := by
  sorry

end regular_polygon_properties_l278_27814


namespace determine_female_athletes_count_l278_27833

theorem determine_female_athletes_count (m : ℕ) (n : ℕ) (x y : ℕ) (probability : ℚ)
  (h_team : 56 + m = 56 + m) -- redundant, but setting up context
  (h_sample_size : n = 28)
  (h_probability : probability = 1 / 28)
  (h_sample_diff : x - y = 4)
  (h_sample_sum : x + y = n)
  (h_ratio : 56 * y = m * x) : m = 42 :=
by
  sorry

end determine_female_athletes_count_l278_27833


namespace dan_money_left_l278_27845

theorem dan_money_left (initial_amount spent_amount remaining_amount : ℤ) (h1 : initial_amount = 300) (h2 : spent_amount = 100) : remaining_amount = 200 :=
by 
  sorry

end dan_money_left_l278_27845


namespace find_a_l278_27880

theorem find_a (a x y : ℝ)
    (h1 : a * x - 5 * y = 5)
    (h2 : x / (x + y) = 5 / 7)
    (h3 : x - y = 3) :
    a = 3 := 
by 
  sorry

end find_a_l278_27880


namespace remainder_when_divided_by_44_l278_27847

theorem remainder_when_divided_by_44 (N : ℕ) (Q : ℕ) (R : ℕ)
  (h1 : N = 44 * 432 + R)
  (h2 : N = 31 * Q + 5) :
  R = 2 :=
sorry

end remainder_when_divided_by_44_l278_27847


namespace range_of_f_l278_27855

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 - (Real.sin x) ^ 2 - 2 * (Real.sin x) * (Real.cos x)

theorem range_of_f : 
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ -Real.sqrt 2 ∧ f x ≤ 1) :=
sorry

end range_of_f_l278_27855


namespace total_passengers_correct_l278_27868

-- Definition of the conditions
def passengers_on_time : ℕ := 14507
def passengers_late : ℕ := 213
def total_passengers : ℕ := passengers_on_time + passengers_late

-- Theorem statement
theorem total_passengers_correct : total_passengers = 14720 := by
  sorry

end total_passengers_correct_l278_27868


namespace correct_ignition_time_l278_27810

noncomputable def ignition_time_satisfying_condition (initial_length : ℝ) (l : ℝ) : ℕ :=
  let burn_rate1 := l / 240
  let burn_rate2 := l / 360
  let stub1 t := l - burn_rate1 * t
  let stub2 t := l - burn_rate2 * t
  let stub_length_condition t := stub2 t = 3 * stub1 t
  let time_difference_at_6AM := 360 -- 6 AM is 360 minutes after midnight
  360 - 180 -- time to ignite the candles

theorem correct_ignition_time : ignition_time_satisfying_condition l 6 = 180 := 
by sorry

end correct_ignition_time_l278_27810


namespace not_multiple_of_3_l278_27808

noncomputable def exists_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n*(n + 3) = m^2

theorem not_multiple_of_3 
  (n : ℕ) (h1 : 0 < n) (h2 : exists_perfect_square n) : ¬ ∃ k : ℕ, n = 3 * k := 
sorry

end not_multiple_of_3_l278_27808


namespace find_other_tax_l278_27890

/-- Jill's expenditure breakdown and total tax conditions. -/
def JillExpenditure 
  (total : ℝ)
  (clothingPercent : ℝ)
  (foodPercent : ℝ)
  (otherPercent : ℝ)
  (clothingTaxPercent : ℝ)
  (foodTaxPercent : ℝ)
  (otherTaxPercent : ℝ)
  (totalTaxPercent : ℝ) :=
  (clothingPercent + foodPercent + otherPercent = 100) ∧
  (clothingTaxPercent = 4) ∧
  (foodTaxPercent = 0) ∧
  (totalTaxPercent = 5.2) ∧
  (total > 0)

/-- The goal is to find the tax percentage on other items which Jill paid, given the constraints. -/
theorem find_other_tax
  {total clothingAmt foodAmt otherAmt clothingTax foodTax otherTaxPercent totalTax : ℝ}
  (h_exp : JillExpenditure total 50 10 40 clothingTax foodTax otherTaxPercent totalTax) :
  otherTaxPercent = 8 :=
by
  sorry

end find_other_tax_l278_27890


namespace three_digit_multiples_of_24_l278_27848

theorem three_digit_multiples_of_24 : 
  let lower_bound := 100
  let upper_bound := 999
  let div_by := 24
  let first := lower_bound + (div_by - lower_bound % div_by) % div_by
  let last := upper_bound - (upper_bound % div_by)
  ∃ n : ℕ, (n + 1) = (last - first) / div_by + 1 := 
sorry

end three_digit_multiples_of_24_l278_27848


namespace small_z_value_l278_27811

noncomputable def w (n : ℕ) := n
noncomputable def x (n : ℕ) := n + 1
noncomputable def y (n : ℕ) := n + 2
noncomputable def z (n : ℕ) := n + 4

theorem small_z_value (n : ℕ) 
  (h : w n ^ 3 + x n ^ 3 + y n ^ 3 = z n ^ 3)
  : z n = 9 :=
sorry

end small_z_value_l278_27811


namespace ratio_of_areas_l278_27803

theorem ratio_of_areas 
  (A B C D E F : Type)
  (AB AC AD : ℝ)
  (h1 : AB = 130)
  (h2 : AC = 130)
  (h3 : AD = 26)
  (CF : ℝ)
  (h4 : CF = 91)
  (BD : ℝ)
  (h5 : BD = 104)
  (AF : ℝ)
  (h6 : AF = 221)
  (EF DE BE CE : ℝ)
  (h7 : EF / DE = 91 / 104)
  (h8 : CE / BE = 3.5) :
  EF * CE = 318.5 * DE * BE :=
sorry

end ratio_of_areas_l278_27803


namespace find_equation_of_ellipse_find_range_OA_OB_find_area_quadrilateral_l278_27881

-- Define the ellipse and parameters
variables (a b c : ℝ) (x y : ℝ)
-- Conditions
def ellipse (a b : ℝ) : Prop := a > b ∧ b > 0 ∧ (∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)

-- Given conditions
def eccentricity (c a : ℝ) : Prop := c = a * (Real.sqrt 3 / 2)
def rhombus_area (a b : ℝ) : Prop := (1/2) * (2 * a) * (2 * b) = 4
def relation_a_b_c (a b c : ℝ) : Prop := a^2 = b^2 + c^2

-- Questions transformed into proof problems
def ellipse_equation (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def range_OA_OB (OA OB : ℝ) : Prop := OA * OB ∈ Set.union (Set.Icc (-(3/2)) 0) (Set.Ioo 0 (3/2))
def quadrilateral_area : ℝ := 4

-- Prove the results given the conditions
theorem find_equation_of_ellipse (a b c : ℝ) (h_ellipse : ellipse a b) (h_ecc : eccentricity c a) (h_area : rhombus_area a b) (h_rel : relation_a_b_c a b c) :
  ellipse_equation x y := by
  sorry

theorem find_range_OA_OB (OA OB : ℝ) (kAC kBD : ℝ) (h_mult : kAC * kBD = -(1/4)) :
  range_OA_OB OA OB := by
  sorry

theorem find_area_quadrilateral : quadrilateral_area = 4 := by
  sorry

end find_equation_of_ellipse_find_range_OA_OB_find_area_quadrilateral_l278_27881


namespace find_a_l278_27838

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom cond1 : a^2 / b = 5
axiom cond2 : b^2 / c = 3
axiom cond3 : c^2 / a = 7

theorem find_a : a = 15 := sorry

end find_a_l278_27838


namespace problem_solution_l278_27892

theorem problem_solution
  (y1 y2 y3 y4 y5 y6 y7 : ℝ)
  (h1 : y1 + 3*y2 + 5*y3 + 7*y4 + 9*y5 + 11*y6 + 13*y7 = 0)
  (h2 : 3*y1 + 5*y2 + 7*y3 + 9*y4 + 11*y5 + 13*y6 + 15*y7 = 10)
  (h3 : 5*y1 + 7*y2 + 9*y3 + 11*y4 + 13*y5 + 15*y6 + 17*y7 = 104) :
  7*y1 + 9*y2 + 11*y3 + 13*y4 + 15*y5 + 17*y6 + 19*y7 = 282 := by
  sorry

end problem_solution_l278_27892


namespace sum_of_edges_of_rectangular_solid_l278_27862

theorem sum_of_edges_of_rectangular_solid 
  (a r : ℝ) 
  (volume_eq : (a / r) * a * (a * r) = 343) 
  (surface_area_eq : 2 * ((a^2 / r) + (a^2 * r) + a^2) = 294) 
  (gp : a / r > 0 ∧ a > 0 ∧ a * r > 0) :
  4 * ((a / r) + a + (a * r)) = 84 :=
by
  sorry

end sum_of_edges_of_rectangular_solid_l278_27862


namespace red_blue_tile_difference_is_15_l278_27846

def num_blue_tiles : ℕ := 17
def num_red_tiles_initial : ℕ := 8
def additional_red_tiles : ℕ := 24
def num_red_tiles_new : ℕ := num_red_tiles_initial + additional_red_tiles
def tile_difference : ℕ := num_red_tiles_new - num_blue_tiles

theorem red_blue_tile_difference_is_15 : tile_difference = 15 :=
by
  sorry

end red_blue_tile_difference_is_15_l278_27846


namespace Jenny_older_than_Rommel_l278_27864

theorem Jenny_older_than_Rommel :
  ∃ t r j, t = 5 ∧ r = 3 * t ∧ j = t + 12 ∧ (j - r = 2) := 
by
  -- We insert the proof here using sorry to skip the actual proof part.
  sorry

end Jenny_older_than_Rommel_l278_27864


namespace latest_start_time_for_liz_l278_27835

def latest_start_time (weight : ℕ) (roast_time_per_pound : ℕ) (num_turkeys : ℕ) (dinner_time : ℕ) : ℕ :=
  dinner_time - (num_turkeys * weight * roast_time_per_pound) / 60

theorem latest_start_time_for_liz : 
  latest_start_time 16 15 2 18 = 10 := by
  sorry

end latest_start_time_for_liz_l278_27835


namespace num_sequences_of_student_helpers_l278_27888

-- Define the conditions
def num_students : ℕ := 15
def num_meetings : ℕ := 3

-- Define the statement to prove
theorem num_sequences_of_student_helpers : 
  (num_students ^ num_meetings) = 3375 :=
by sorry

end num_sequences_of_student_helpers_l278_27888


namespace minibus_seat_count_l278_27820

theorem minibus_seat_count 
  (total_children : ℕ) 
  (seats_with_3_children : ℕ) 
  (children_per_3_child_seat : ℕ) 
  (remaining_children : ℕ) 
  (children_per_2_child_seat : ℕ) 
  (total_seats : ℕ) :
  total_children = 19 →
  seats_with_3_children = 5 →
  children_per_3_child_seat = 3 →
  remaining_children = total_children - (seats_with_3_children * children_per_3_child_seat) →
  children_per_2_child_seat = 2 →
  total_seats = seats_with_3_children + (remaining_children / children_per_2_child_seat) →
  total_seats = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end minibus_seat_count_l278_27820


namespace solve_expression_l278_27884

theorem solve_expression : 3 ^ (1 ^ (0 ^ 2)) - ((3 ^ 1) ^ 0) ^ 2 = 2 := by
  sorry

end solve_expression_l278_27884


namespace sum_of_data_l278_27897

theorem sum_of_data (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a = 12) : a + b + c = 96 :=
by
  sorry

end sum_of_data_l278_27897


namespace find_f_lg_lg2_l278_27822

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x ^ 2) - x) + 4

theorem find_f_lg_lg2 :
  f (Real.logb 10 (2)) = 3 :=
sorry

end find_f_lg_lg2_l278_27822


namespace tangent_line_at_one_min_value_f_l278_27854

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + a * |Real.log x - 1|

theorem tangent_line_at_one (a : ℝ) (h1 : a = 1) : 
  ∃ (m b : ℝ), (∀ x : ℝ, f x a = m * x + b) ∧ m = 1 ∧ b = 1 ∧ (x - y + 1 = 0) := 
sorry

theorem min_value_f (a : ℝ) (h1 : 0 < a) : 
  (1 ≤ x ∧ x < e)  →  (x - f x a <= 0) ∨  (∀ (x : ℝ), 
  (f x a = if 0 < a ∧ a ≤ 2 then 1 + a 
          else if 2 < a ∧ a ≤ 2 * Real.exp (2) then 3 * (a / 2)^2 - (a / 2)^2 * Real.log (a / 2) else 
          Real.exp 2) 
   ) := 
sorry

end tangent_line_at_one_min_value_f_l278_27854


namespace ones_digit_11_pow_l278_27894

theorem ones_digit_11_pow (n : ℕ) (hn : n > 0) : (11^n % 10) = 1 := by
  sorry

end ones_digit_11_pow_l278_27894


namespace divisibility_of_product_l278_27851

theorem divisibility_of_product (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : (a * b) % 5 = 0) :
  a % 5 = 0 ∨ b % 5 = 0 :=
sorry

end divisibility_of_product_l278_27851


namespace fraction_changed_value_l278_27866

theorem fraction_changed_value:
  ∀ (num denom : ℝ), num / denom = 0.75 →
  (num + 0.15 * num) / (denom - 0.08 * denom) = 0.9375 :=
by
  intros num denom h_fraction
  sorry

end fraction_changed_value_l278_27866


namespace ratio_of_cube_dimensions_l278_27856

theorem ratio_of_cube_dimensions (V_original V_larger : ℝ) (hV_org : V_original = 64) (hV_lrg : V_larger = 512) :
  (∃ r : ℝ, r^3 = V_larger / V_original) ∧ r = 2 := 
sorry

end ratio_of_cube_dimensions_l278_27856


namespace cost_of_bananas_is_two_l278_27869

variable (B : ℝ)

theorem cost_of_bananas_is_two (h : 1.20 * (3 + B) = 6) : B = 2 :=
by
  sorry

end cost_of_bananas_is_two_l278_27869


namespace abc_proof_l278_27889

noncomputable def abc_value (a b c : ℝ) : ℝ :=
  a * b * c

theorem abc_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b = 24 * (3 ^ (1 / 3)))
  (h5 : a * c = 40 * (3 ^ (1 / 3)))
  (h6 : b * c = 16 * (3 ^ (1 / 3))) : 
  abc_value a b c = 96 * (15 ^ (1 / 2)) :=
sorry

end abc_proof_l278_27889


namespace elvin_fixed_monthly_charge_l278_27886

theorem elvin_fixed_monthly_charge
  (F C : ℝ) 
  (h1 : F + C = 50) 
  (h2 : F + 2 * C = 76) : 
  F = 24 := 
sorry

end elvin_fixed_monthly_charge_l278_27886


namespace martha_cakes_required_l278_27815

-- Conditions
def number_of_children : ℝ := 3.0
def cakes_per_child : ℝ := 18.0

-- The main statement to prove
theorem martha_cakes_required:
  (number_of_children * cakes_per_child) = 54.0 := 
by
  sorry

end martha_cakes_required_l278_27815


namespace div_240_of_prime_diff_l278_27876

-- Definitions
def is_prime (n : ℕ) : Prop := ∃ p : ℕ, p = n ∧ Prime p
def prime_with_two_digits (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ is_prime n

-- The theorem statement
theorem div_240_of_prime_diff (a b : ℕ) (ha : prime_with_two_digits a) (hb : prime_with_two_digits b) (h : a > b) :
  240 ∣ (a^4 - b^4) ∧ ∀ d : ℕ, (d ∣ (a^4 - b^4) → (∀ m n : ℕ, prime_with_two_digits m → prime_with_two_digits n → m > n → d ∣ (m^4 - n^4) ) → d ≤ 240) :=
by
  sorry

end div_240_of_prime_diff_l278_27876


namespace value_of_m_l278_27832

theorem value_of_m (m : ℕ) (h : 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 4934) : m = 4 :=
by
  sorry

end value_of_m_l278_27832


namespace num_clients_visited_garage_l278_27898

theorem num_clients_visited_garage :
  ∃ (num_clients : ℕ), num_clients = 24 ∧
    ∀ (num_cars selections_per_car selections_per_client : ℕ),
        num_cars = 16 → selections_per_car = 3 → selections_per_client = 2 →
        (num_cars * selections_per_car) / selections_per_client = num_clients :=
by
  sorry

end num_clients_visited_garage_l278_27898


namespace good_pair_bound_all_good_pairs_l278_27804

namespace good_pairs

-- Definition of a "good" pair
def is_good_pair (r s : ℕ) : Prop :=
  ∃ (P : ℤ → ℤ) (a : Fin r → ℤ) (b : Fin s → ℤ),
  (∀ i j : Fin r, i ≠ j → a i ≠ a j) ∧
  (∀ i j : Fin s, i ≠ j → b i ≠ b j) ∧
  (∀ i : Fin r, P (a i) = 2) ∧
  (∀ j : Fin s, P (b j) = 5)

-- (a) Show that for every good pair (r, s), r, s ≤ 3
theorem good_pair_bound (r s : ℕ) (h : is_good_pair r s) : r ≤ 3 ∧ s ≤ 3 :=
sorry

-- (b) Determine all good pairs
theorem all_good_pairs (r s : ℕ) : is_good_pair r s ↔ (r ≤ 3 ∧ s ≤ 3 ∧ (
  (r = 1 ∧ s = 1) ∨ (r = 1 ∧ s = 2) ∨ (r = 1 ∧ s = 3) ∨
  (r = 2 ∧ s = 1) ∨ (r = 2 ∧ s = 2) ∨ (r = 2 ∧ s = 3) ∨
  (r = 3 ∧ s = 1) ∨ (r = 3 ∧ s = 2))) :=
sorry

end good_pairs

end good_pair_bound_all_good_pairs_l278_27804


namespace original_amount_l278_27882

theorem original_amount {P : ℕ} {R : ℕ} {T : ℕ} (h1 : P = 1000) (h2 : T = 5) 
  (h3 : ∃ R, (1000 * (R + 5) * 5) / 100 + 1000 = 1750) : 
  1000 + (1000 * R * 5 / 100) = 1500 :=
by
  sorry

end original_amount_l278_27882


namespace team_E_not_played_against_team_B_l278_27895

-- Define the teams
inductive Team
| A | B | C | D | E | F
deriving DecidableEq

open Team

-- Define the matches played by each team
def matches_played : Team → Nat
| A => 5
| B => 4
| C => 3
| D => 2
| E => 1
| F => 0

-- Define the pairwise matches function
def paired : Team → Team → Prop
| A, B => true
| A, C => true
| A, D => true
| A, E => true
| A, F => true
| B, C => true
| B, D => true
| B, F  => true
| _, _ => false

-- Define the theorem based on the conditions and question
theorem team_E_not_played_against_team_B :
  ¬ paired E B :=
by
  sorry

end team_E_not_played_against_team_B_l278_27895


namespace virginia_taught_fewer_years_l278_27844

-- Definitions based on conditions
variable (V A D : ℕ)

-- Dennis has taught for 34 years
axiom h1 : D = 34

-- Virginia has taught for 9 more years than Adrienne
axiom h2 : V = A + 9

-- Combined total of years taught is 75
axiom h3 : V + A + D = 75

-- Proof statement: Virginia has taught for 9 fewer years than Dennis
theorem virginia_taught_fewer_years : D - V = 9 :=
  sorry

end virginia_taught_fewer_years_l278_27844


namespace simplify_expression_l278_27883

theorem simplify_expression (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_eq : x^3 + y^3 = 3 * (x + y)) :
  (x / y) + (y / x) - (3 / (x * y)) = 1 :=
by
  sorry

end simplify_expression_l278_27883


namespace scientific_notation_of_1653_billion_l278_27840

theorem scientific_notation_of_1653_billion :
  (1653 * (10 ^ 9) = 1.6553 * (10 ^ 12)) :=
sorry

end scientific_notation_of_1653_billion_l278_27840


namespace no_real_solution_l278_27809

-- Define the given equation as a function
def equation (x y : ℝ) : ℝ := 3 * x^2 + 5 * y^2 - 9 * x - 20 * y + 30 + 4 * x * y

-- State that the equation equals zero has no real solution.
theorem no_real_solution : ∀ x y : ℝ, equation x y ≠ 0 :=
by sorry

end no_real_solution_l278_27809


namespace sue_travel_time_correct_l278_27801

-- Define the flight and layover times as constants
def NO_to_ATL_flight_hours : ℕ := 2
def ATL_layover_hours : ℕ := 4
def ATL_to_CHI_flight_hours : ℕ := 5
def CHI_time_diff_hours : ℤ := -1
def CHI_layover_hours : ℕ := 3
def CHI_to_NY_flight_hours : ℕ := 3
def NY_time_diff_hours : ℤ := 1
def NY_layover_hours : ℕ := 16
def NY_to_DEN_flight_hours : ℕ := 6
def DEN_time_diff_hours : ℤ := -2
def DEN_layover_hours : ℕ := 5
def DEN_to_SF_flight_hours : ℕ := 4
def SF_time_diff_hours : ℤ := -1

-- Total time calculation including flights, layovers, and time zone changes
def total_travel_time_hours : ℕ :=
  NO_to_ATL_flight_hours +
  ATL_layover_hours +
  (ATL_to_CHI_flight_hours + CHI_time_diff_hours).toNat +  -- Handle time difference (ensure non-negative)
  CHI_layover_hours +
  (CHI_to_NY_flight_hours + NY_time_diff_hours).toNat +
  NY_layover_hours +
  (NY_to_DEN_flight_hours + DEN_time_diff_hours).toNat +
  DEN_layover_hours +
  (DEN_to_SF_flight_hours + SF_time_diff_hours).toNat

-- Statement to prove in Lean:
theorem sue_travel_time_correct : total_travel_time_hours = 45 :=
by {
  -- Skipping proof details since only the statement is required
  sorry
}

end sue_travel_time_correct_l278_27801


namespace find_r_l278_27865

theorem find_r (r : ℝ) (h1 : ∃ s : ℝ, 8 * x^3 - 4 * x^2 - 42 * x + 45 = 8 * (x - r)^2 * (x - s)) :
  r = 3 / 2 :=
by
  sorry

end find_r_l278_27865


namespace distinct_special_sums_l278_27817

def is_special_fraction (a b : ℕ) : Prop := a + b = 18

def is_special_sum (n : ℤ) : Prop :=
  ∃ (a1 b1 a2 b2 : ℕ), is_special_fraction a1 b1 ∧ is_special_fraction a2 b2 ∧ 
  n = (a1 : ℤ) * (b2 : ℤ) * b1 + (a2 : ℤ) * (b1 : ℤ) / a1

theorem distinct_special_sums : 
  (∃ (sums : Finset ℤ), 
    (∀ n, n ∈ sums ↔ is_special_sum n) ∧ 
    sums.card = 7) :=
sorry

end distinct_special_sums_l278_27817


namespace price_of_computer_and_desk_l278_27896

theorem price_of_computer_and_desk (x y : ℕ) 
  (h1 : 10 * x + 200 * y = 90000)
  (h2 : 12 * x + 120 * y = 90000) : 
  x = 6000 ∧ y = 150 :=
by
  sorry

end price_of_computer_and_desk_l278_27896


namespace part1_part2_l278_27806

def op (a b : ℤ) := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 := 
by
  -- Proof omitted
  sorry

theorem part2 (x : ℤ) : 
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- Proof omitted
  sorry

end part1_part2_l278_27806


namespace initial_ratio_l278_27800

-- Definitions of the initial state and conditions
variables (M W : ℕ)
def initial_men : ℕ := M
def initial_women : ℕ := W
def men_after_entry : ℕ := M + 2
def women_after_exit_and_doubling : ℕ := (W - 3) * 2
def current_men : ℕ := 14
def current_women : ℕ := 24

-- Theorem to prove the initial ratio
theorem initial_ratio (M W : ℕ) 
    (hm : men_after_entry M = current_men)
    (hw : women_after_exit_and_doubling W = current_women) :
  M / Nat.gcd M W = 4 ∧ W / Nat.gcd M W = 5 :=
by
  sorry

end initial_ratio_l278_27800


namespace eval_x_sq_minus_y_sq_l278_27878

theorem eval_x_sq_minus_y_sq (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 20) 
  (h2 : 4 * x + 3 * y = 29) : 
  x^2 - y^2 = -45 :=
sorry

end eval_x_sq_minus_y_sq_l278_27878


namespace ellipse_y_axis_intersection_l278_27802

open Real

/-- Defines an ellipse with given foci and a point on the ellipse,
    and establishes the coordinate of the other y-axis intersection. -/
theorem ellipse_y_axis_intersection :
  ∃ y : ℝ, (dist (0, y) (1, -1) + dist (0, y) (-2, 2) = 3 * sqrt 2) ∧ y = sqrt ((9 * sqrt 2 - 4) / 2) :=
sorry

end ellipse_y_axis_intersection_l278_27802


namespace calculate_expression_l278_27816

theorem calculate_expression 
  (a1 : 84 + 4 / 19 = 1600 / 19) 
  (a2 : 105 + 5 / 19 = 2000 / 19) 
  (a3 : 1.375 = 11 / 8) 
  (a4 : 0.8 = 4 / 5) :
  84 * (4 / 19) * (11 / 8) + 105 * (5 / 19) * (4 / 5) = 200 := 
sorry

end calculate_expression_l278_27816


namespace stickers_after_exchange_l278_27842

-- Given conditions
def Ryan_stickers : ℕ := 30
def Steven_stickers : ℕ := 3 * Ryan_stickers
def Terry_stickers : ℕ := Steven_stickers + 20
def Emily_stickers : ℕ := Steven_stickers / 2
def Jasmine_stickers : ℕ := Terry_stickers + Terry_stickers / 10

def total_stickers_before : ℕ := 
  Ryan_stickers + Steven_stickers + Terry_stickers + Emily_stickers + Jasmine_stickers

noncomputable def total_stickers_after : ℕ := 
  total_stickers_before - 2 * 5

-- The goal is to prove that the total stickers after the exchange event is 386
theorem stickers_after_exchange : total_stickers_after = 386 := 
  by sorry

end stickers_after_exchange_l278_27842


namespace product_of_three_equal_numbers_l278_27805

theorem product_of_three_equal_numbers
    (a b : ℕ) (x : ℕ)
    (h1 : a = 12)
    (h2 : b = 22)
    (h_mean : (a + b + 3 * x) / 5 = 20) :
    x * x * x = 10648 := by
  sorry

end product_of_three_equal_numbers_l278_27805


namespace temperature_on_Friday_l278_27852

-- Definitions of the temperatures on the days
variables {M T W Th F : ℝ}

-- Conditions given in the problem
def avg_temp_mon_thu (M T W Th : ℝ) : Prop := (M + T + W + Th) / 4 = 48
def avg_temp_tue_fri (T W Th F : ℝ) : Prop := (T + W + Th + F) / 4 = 46
def temp_mon (M : ℝ) : Prop := M = 44

-- Statement to prove
theorem temperature_on_Friday (h1 : avg_temp_mon_thu M T W Th)
                               (h2 : avg_temp_tue_fri T W Th F)
                               (h3 : temp_mon M) : F = 36 :=
sorry

end temperature_on_Friday_l278_27852


namespace star_sub_correctness_l278_27872

def star (x y : ℤ) : ℤ := x * y - 3 * x

theorem star_sub_correctness : (star 6 2) - (star 2 6) = -12 := by
  sorry

end star_sub_correctness_l278_27872


namespace larry_wins_probability_eq_l278_27859

-- Define the conditions
def larry_probability_knocks_off : ℚ := 1 / 3
def julius_probability_knocks_off : ℚ := 1 / 4
def larry_throws_first : Prop := True
def independent_events : Prop := True

-- Define the proof that Larry wins the game with probability 2/3
theorem larry_wins_probability_eq :
  larry_throws_first ∧ independent_events →
  larry_probability_knocks_off = 1/3 ∧ julius_probability_knocks_off = 1/4 →
  ∃ p : ℚ, p = 2 / 3 :=
by
  sorry

end larry_wins_probability_eq_l278_27859


namespace range_of_a_minus_b_l278_27829

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem range_of_a_minus_b (a b : ℝ) (h1 : ∃ α β : ℝ, α ≠ β ∧ f α a b = 0 ∧ f β a b = 0)
  (h2 : ∃ x1 x2 x3 x4 : ℝ, x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
                         (x2 - x1 = x3 - x2) ∧ (x3 - x2 = x4 - x3) ∧
                         f (x1^2 + 2 * x1 - 1) a b = 0 ∧
                         f (x2^2 + 2 * x2 - 1) a b = 0 ∧
                         f (x3^2 + 2 * x3 - 1) a b = 0 ∧
                         f (x4^2 + 2 * x4 - 1) a b = 0) :
  a - b ≤ 25 / 9 :=
sorry

end range_of_a_minus_b_l278_27829


namespace two_categorical_variables_l278_27873

-- Definitions based on the conditions
def smoking (x : String) : Prop := x = "Smoking" ∨ x = "Not smoking"
def sick (y : String) : Prop := y = "Sick" ∨ y = "Not sick"

def category1 (z : String) : Prop := z = "Whether smoking"
def category2 (w : String) : Prop := w = "Whether sick"

-- The main proof statement
theorem two_categorical_variables : 
  (category1 "Whether smoking" ∧ smoking "Smoking" ∧ smoking "Not smoking") ∧
  (category2 "Whether sick" ∧ sick "Sick" ∧ sick "Not sick") →
  "Whether smoking, Whether sick" = "Whether smoking, Whether sick" :=
by
  sorry

end two_categorical_variables_l278_27873


namespace unique_triple_solution_l278_27887

theorem unique_triple_solution (x y z : ℝ) :
  x = y^3 + y - 8 ∧ y = z^3 + z - 8 ∧ z = x^3 + x - 8 → (x, y, z) = (2, 2, 2) :=
by
  sorry

end unique_triple_solution_l278_27887


namespace unattainable_y_l278_27867

theorem unattainable_y (x : ℚ) (hx : x ≠ -4 / 3) : 
    ∀ y : ℚ, (y = (2 - x) / (3 * x + 4)) → y ≠ -1 / 3 :=
sorry

end unattainable_y_l278_27867


namespace find_prime_pairs_l278_27823

theorem find_prime_pairs (p q : ℕ) (p_prime : Nat.Prime p) (q_prime : Nat.Prime q) 
  (h1 : ∃ a : ℤ, a^2 = p - q)
  (h2 : ∃ b : ℤ, b^2 = p * q - q) : 
  (p, q) = (3, 2) :=
by {
    sorry
}

end find_prime_pairs_l278_27823


namespace Ivan_returns_alive_Ivan_takes_princesses_l278_27825

theorem Ivan_returns_alive (Tsarevnas Koscheis: Finset ℕ) (five_girls: Finset ℕ) 
  (cond1: Tsarevnas.card = 3) (cond2: Koscheis.card = 2) (cond3: five_girls.card = 5)
  (cond4: Tsarevnas ∪ Koscheis = five_girls)
  (cond5: ∀ g ∈ five_girls, ∃ t ∈ Tsarevnas, t ≠ g ∧ ∃ k ∈ Koscheis, k ≠ g)
  (cond6: ∀ girl : ℕ, girl ∈ five_girls → 
          ∃ truth_count : ℕ, 
          (truth_count = (if girl ∈ Tsarevnas then 2 else 3))): 
  ∃ princesses : Finset ℕ, princesses.card = 3 ∧ princesses ⊆ Tsarevnas ∧ ∀ k ∈ Koscheis, k ∉ princesses :=
sorry

theorem Ivan_takes_princesses (Tsarevnas Koscheis: Finset ℕ) (five_girls: Finset ℕ) 
  (cond1: Tsarevnas.card = 3) (cond2: Koscheis.card = 2) (cond3: five_girls.card = 5)
  (cond4: Tsarevnas ∪ Koscheis = five_girls)
  (cond5: ∀ g ∈ five_girls, ∃ t ∈ Tsarevnas, t ≠ g ∧ ∃ k ∈ Koscheis, k ≠ g)
  (cond6 and cond7: ∀ girl1 girl2 girl3 : ℕ, girl1 ≠ girl2 → girl2 ≠ girl3 → girl1 ∈ Tsarevnas → girl2 ∈ Tsarevnas → girl3 ∈ Tsarevnas → 
          ∃ (eldest middle youngest : ℕ), 
              (eldest ∈ Tsarevnas ∧ middle ∈ Tsarevnas ∧ youngest ∈ Tsarevnas) 
          ∧
              (eldest ≠ middle ∧ eldest ≠ youngest ∧ middle ≠ youngest)
          ∧
              (∀ k ∈ Koscheis, k ≠ eldest ∧ k ≠ middle ∧ k ≠ youngest)
  ):
  ∃ princesses : Finset ℕ, 
          princesses.card = 3 ∧ princesses ⊆ Tsarevnas ∧ 
          (∃ eldest ,∃ middle,∃ youngest : ℕ, eldest ∈ princesses ∧ middle ∈ princesses ∧ youngest ∈ princesses ∧ 
                 eldest ≠ middle ∧ eldest ≠ youngest ∧ middle ≠ youngest)
:=
sorry

end Ivan_returns_alive_Ivan_takes_princesses_l278_27825


namespace toothpicks_needed_for_cube_grid_l278_27831

-- Defining the conditions: a cube-shaped grid with dimensions 5x5x5.
def grid_length : ℕ := 5
def grid_width : ℕ := 5
def grid_height : ℕ := 5

-- The theorem to prove the number of toothpicks needed is 2340.
theorem toothpicks_needed_for_cube_grid (L W H : ℕ) (h1 : L = grid_length) (h2 : W = grid_width) (h3 : H = grid_height) :
  (L + 1) * (W + 1) * H + 2 * (L + 1) * W * (H + 1) = 2340 :=
  by
    -- Proof goes here
    sorry

end toothpicks_needed_for_cube_grid_l278_27831


namespace find_k_b_l278_27893

noncomputable def symmetric_line_circle_intersection : Prop :=
  ∃ (k b : ℝ), 
    (∀ (x y : ℝ),  (y = k * x) ∧ ((x-1)^2 + y^2 = 1)) ∧ 
    (∀ (x y : ℝ), (x - y + b = 0)) →
    (k = -1 ∧ b = -1)

theorem find_k_b :
  symmetric_line_circle_intersection :=
  by
    -- omitted proof
    sorry

end find_k_b_l278_27893


namespace monotonic_decreasing_range_l278_27891

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.cos x

theorem monotonic_decreasing_range (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) → a ≤ -1 :=
  sorry

end monotonic_decreasing_range_l278_27891


namespace cube_face_expression_l278_27870

theorem cube_face_expression (a b c : ℤ) (h1 : 3 * a + 2 = 17) (h2 : 7 * b - 4 = 10) (h3 : a + 3 * b - 2 * c = 11) : 
  a - b * c = 5 :=
by sorry

end cube_face_expression_l278_27870


namespace calculate_area_of_region_l278_27818

theorem calculate_area_of_region :
  let region := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 - 4 * p.2 = 12}
  ∃ area, area = 17 * Real.pi
:= by
  sorry

end calculate_area_of_region_l278_27818


namespace problem_1_problem_2_problem_3_problem_4_l278_27899

theorem problem_1 : 12 - (-18) + (-7) - 15 = 8 := sorry

theorem problem_2 : -0.5 + (- (3 + 1/4)) + (-2.75) + (7 + 1/2) = 1 := sorry

theorem problem_3 : -2^2 + 3 * (-1)^(2023) - abs (-4) * 5 = -27 := sorry

theorem problem_4 : -3 - (-5 + (1 - 2 * (3 / 5)) / (-2)) = 19 / 10 := sorry

end problem_1_problem_2_problem_3_problem_4_l278_27899


namespace initial_distance_between_A_and_B_l278_27858

theorem initial_distance_between_A_and_B
  (start_time : ℕ)        -- time in hours, 1 pm
  (meet_time : ℕ)         -- time in hours, 3 pm
  (speed_A : ℕ)           -- speed of A in km/hr
  (speed_B : ℕ)           -- speed of B in km/hr
  (time_walked : ℕ)       -- time walked in hours
  (distance_A : ℕ)        -- distance covered by A in km
  (distance_B : ℕ)        -- distance covered by B in km
  (initial_distance : ℕ)  -- initial distance between A and B

  (h1 : start_time = 1)
  (h2 : meet_time = 3)
  (h3 : speed_A = 5)
  (h4 : speed_B = 7)
  (h5 : time_walked = meet_time - start_time)
  (h6 : distance_A = speed_A * time_walked)
  (h7 : distance_B = speed_B * time_walked)
  (h8 : initial_distance = distance_A + distance_B) :

  initial_distance = 24 :=
by
  sorry

end initial_distance_between_A_and_B_l278_27858


namespace tea_drinking_proof_l278_27850

theorem tea_drinking_proof :
  ∃ (k : ℝ), 
    (∃ (c_sunday t_sunday c_wednesday t_wednesday : ℝ),
      c_sunday = 8.5 ∧ 
      t_sunday = 4 ∧ 
      c_wednesday = 5 ∧ 
      t_sunday * c_sunday = k ∧ 
      t_wednesday * c_wednesday = k ∧ 
      t_wednesday = 6.8) :=
sorry

end tea_drinking_proof_l278_27850


namespace travel_time_l278_27813

noncomputable def convert_kmh_to_mps (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

theorem travel_time
  (speed_kmh : ℝ)
  (distance_m : ℝ) :
  speed_kmh = 63 →
  distance_m = 437.535 →
  (distance_m / convert_kmh_to_mps speed_kmh) = 25 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end travel_time_l278_27813


namespace prob_and_relation_proof_l278_27849

-- Defining conditions
def total_buses : ℕ := 500

def A_on_time : ℕ := 240
def A_not_on_time : ℕ := 20
def B_on_time : ℕ := 210
def B_not_on_time : ℕ := 30

def A_total : ℕ := A_on_time + A_not_on_time
def B_total : ℕ := B_on_time + B_not_on_time

def prob_A_on_time : ℚ := A_on_time / A_total
def prob_B_on_time : ℚ := B_on_time / B_total

-- Defining K^2 calculation
def n : ℕ := total_buses
def a : ℕ := A_on_time
def b : ℕ := A_not_on_time
def c : ℕ := B_on_time
def d : ℕ := B_not_on_time

def K_squared : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def threshold_90_percent : ℚ := 2.706

-- Lean theorem statement
theorem prob_and_relation_proof :
  prob_A_on_time = 12 / 13 ∧
  prob_B_on_time = 7 / 8 ∧
  K_squared > threshold_90_percent :=
by {
   sorry
}

end prob_and_relation_proof_l278_27849


namespace find_m_l278_27828

theorem find_m 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a (n - 1) + a (n + 1) = 2 * a n)
  (h_cond1 : a (m - 1) + a (m + 1) - a m ^ 2 = 0)
  (h_cond2 : S (2 * m - 1) = 38) 
  : m = 10 :=
sorry

end find_m_l278_27828


namespace train_leave_tunnel_l278_27860

noncomputable def train_leave_time 
  (train_speed : ℝ) 
  (tunnel_length : ℝ) 
  (train_length : ℝ) 
  (enter_time : ℝ × ℝ) : ℝ × ℝ :=
  let speed_km_min := train_speed / 60
  let total_distance := train_length + tunnel_length
  let time_to_pass := total_distance / speed_km_min
  let enter_minutes := enter_time.1 * 60 + enter_time.2
  let leave_minutes := enter_minutes + time_to_pass
  let leave_hours := leave_minutes / 60
  let leave_remainder_minutes := leave_minutes % 60
  (leave_hours, leave_remainder_minutes)

theorem train_leave_tunnel : 
  train_leave_time 80 70 1 (5, 12) = (6, 5.25) := 
sorry

end train_leave_tunnel_l278_27860


namespace range_of_a_l278_27877

def P (a : ℝ) : Set ℝ := { x : ℝ | a - 4 < x ∧ x < a + 4 }
def Q : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 < 0 }

theorem range_of_a (a : ℝ) : (∀ x, Q x → P a x) → -1 < a ∧ a < 5 :=
by
  intro h
  sorry

end range_of_a_l278_27877


namespace find_x_when_y_is_72_l278_27827

theorem find_x_when_y_is_72 
  (x y : ℝ) (k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_const : ∀ x y, 0 < x → 0 < y → x^2 * y = k)
  (h_initial : 9 * 8 = k)
  (h_y_72 : y = 72)
  (h_x2_factor : x^2 = 4 * 9) :
  x = 1 :=
sorry

end find_x_when_y_is_72_l278_27827


namespace percent_increase_fifth_triangle_l278_27875

noncomputable def initial_side_length : ℝ := 3
noncomputable def growth_factor : ℝ := 1.2
noncomputable def num_triangles : ℕ := 5

noncomputable def side_length (n : ℕ) : ℝ :=
  initial_side_length * growth_factor ^ (n - 1)

noncomputable def perimeter_length (n : ℕ) : ℝ :=
  3 * side_length n

noncomputable def percent_increase (n : ℕ) : ℝ :=
  ((perimeter_length n / perimeter_length 1) - 1) * 100

theorem percent_increase_fifth_triangle :
  percent_increase 5 = 107.4 :=
by
  sorry

end percent_increase_fifth_triangle_l278_27875


namespace min_students_l278_27885

variable (L : ℕ) (H : ℕ) (M : ℕ) (e : ℕ)

def find_min_students : Prop :=
  H = 2 * L ∧ 
  M = L + H ∧ 
  e = L + M + H ∧ 
  e = 6 * L ∧ 
  L ≥ 1

theorem min_students (L : ℕ) (H : ℕ) (M : ℕ) (e : ℕ) : find_min_students L H M e → e = 6 := 
by 
  intro h 
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end min_students_l278_27885


namespace triangle_inequality_l278_27843

variable (a b c R : ℝ)

-- Assuming a, b, c as the sides of a triangle
-- and R as the circumradius.

theorem triangle_inequality:
  (1 / (a * b)) + (1 / (b * c)) + (1 / (c * a)) ≥ (1 / (R * R)) :=
by
  sorry

end triangle_inequality_l278_27843


namespace inequality_abc_l278_27874

variables {a b c : ℝ}

theorem inequality_abc 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2) ∧ 
    (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) := 
by
  sorry

end inequality_abc_l278_27874


namespace product_of_integers_is_eight_l278_27871

-- Define three different positive integers a, b, c such that they sum to 7
def sum_to_seven (a b c : ℕ) : Prop := a + b + c = 7 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Prove that the product of these integers is 8
theorem product_of_integers_is_eight (a b c : ℕ) (h : sum_to_seven a b c) : a * b * c = 8 := by sorry

end product_of_integers_is_eight_l278_27871


namespace find_integers_l278_27807

def isPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

theorem find_integers (x : ℤ) (h : isPerfectSquare (x^2 + 19 * x + 95)) : x = -14 ∨ x = -5 := by
  sorry

end find_integers_l278_27807


namespace cone_new_height_eq_sqrt_85_l278_27830

/-- A cone has a uniform circular base of radius 6 feet and a slant height of 13 feet.
    After the side breaks, the slant height reduces by 2 feet, making the new slant height 11 feet.
    We need to determine the new height from the base to the tip of the cone, and prove it is sqrt(85). -/
theorem cone_new_height_eq_sqrt_85 :
  let r : ℝ := 6
  let l : ℝ := 13
  let l' : ℝ := 11
  let h : ℝ := Real.sqrt (13^2 - 6^2)
  let H : ℝ := Real.sqrt (11^2 - 6^2)
  H = Real.sqrt 85 :=
by
  sorry


end cone_new_height_eq_sqrt_85_l278_27830


namespace expansion_no_x2_term_l278_27857

theorem expansion_no_x2_term (n : ℕ) (h1 : 5 ≤ n) (h2 : n ≤ 8) :
  ¬ ∃ (r : ℕ), 0 ≤ r ∧ r ≤ n ∧ n - 4 * r = 2 → n = 7 := by
  sorry

end expansion_no_x2_term_l278_27857


namespace log_equivalence_l278_27826

theorem log_equivalence :
  (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 :=
by
  sorry

end log_equivalence_l278_27826


namespace sally_weekend_reading_l278_27834

theorem sally_weekend_reading (pages_on_weekdays : ℕ) (total_pages : ℕ) (weeks : ℕ) (weekdays_per_week : ℕ) (total_days : ℕ) 
  (finishing_time : ℕ) (weekend_days : ℕ) (pages_weekdays_total : ℕ) :
  pages_on_weekdays = 10 →
  total_pages = 180 →
  weeks = 2 →
  weekdays_per_week = 5 →
  weekend_days = (total_days - weekdays_per_week * weeks) →
  total_days = 7 * weeks →
  finishing_time = weeks →
  pages_weekdays_total = pages_on_weekdays * weekdays_per_week * weeks →
  (total_pages - pages_weekdays_total) / weekend_days = 20 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end sally_weekend_reading_l278_27834


namespace sara_spent_on_rented_movie_l278_27879

def total_spent_on_movies : ℝ := 36.78
def spent_on_tickets : ℝ := 2 * 10.62
def spent_on_bought_movie : ℝ := 13.95

theorem sara_spent_on_rented_movie : 
  (total_spent_on_movies - spent_on_tickets - spent_on_bought_movie = 1.59) := 
by sorry

end sara_spent_on_rented_movie_l278_27879


namespace evaluate_expression_l278_27863

theorem evaluate_expression (a b c : ℤ)
  (h1 : c = b - 12)
  (h2 : b = a + 4)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3 : ℚ) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 7 / 3 := 
sorry

end evaluate_expression_l278_27863


namespace target_water_percentage_is_two_percent_l278_27837

variable (initial_milk_volume pure_milk_volume : ℕ)
variable (initial_water_percentage target_water_percentage : ℚ)

-- Conditions: Initial milk contains 5% water and we add 15 liters of pure milk
axiom initial_milk_condition : initial_milk_volume = 10
axiom pure_milk_condition : pure_milk_volume = 15
axiom initial_water_condition : initial_water_percentage = 5 / 100

-- Prove that target percentage of water in the milk is 2%
theorem target_water_percentage_is_two_percent :
  target_water_percentage = 2 / 100 := by
  sorry

end target_water_percentage_is_two_percent_l278_27837


namespace bowling_ball_surface_area_l278_27861

theorem bowling_ball_surface_area (d : ℝ) (hd : d = 9) : 
  4 * Real.pi * (d / 2)^2 = 81 * Real.pi :=
by
  -- proof goes here
  sorry

end bowling_ball_surface_area_l278_27861


namespace percent_counties_l278_27839

def p1 : ℕ := 21
def p2 : ℕ := 44
def p3 : ℕ := 18

theorem percent_counties (h1 : p1 = 21) (h2 : p2 = 44) (h3 : p3 = 18) : p1 + p2 + p3 = 83 :=
by sorry

end percent_counties_l278_27839


namespace saddle_value_l278_27819

theorem saddle_value (S : ℝ) (H : ℝ) (h1 : S + H = 100) (h2 : H = 7 * S) : S = 12.50 :=
by
  sorry

end saddle_value_l278_27819


namespace cory_fruit_eating_orders_l278_27812

open Nat

theorem cory_fruit_eating_orders : 
    let apples := 4
    let oranges := 3
    let bananas := 2
    let grape := 1
    let total_fruits := apples + oranges + bananas + grape
    apples + oranges + bananas + grape = 10 →
    total_fruits = 10 →
    apples ≥ 1 →
    factorial 9 / (factorial 3 * factorial 3 * factorial 2 * factorial 1) = 5040 :=
by
  intros apples oranges bananas grape total_fruits h_total h_sum h_apples
  sorry

end cory_fruit_eating_orders_l278_27812


namespace minimum_words_to_learn_for_90_percent_l278_27836

-- Define the conditions
def total_vocabulary_words : ℕ := 800
def minimum_percentage_required : ℚ := 0.90

-- Define the proof goal
theorem minimum_words_to_learn_for_90_percent (x : ℕ) (h1 : (x : ℚ) / total_vocabulary_words ≥ minimum_percentage_required) : x ≥ 720 :=
sorry

end minimum_words_to_learn_for_90_percent_l278_27836


namespace triangle_count_relationship_l278_27824

theorem triangle_count_relationship :
  let n_0 : ℕ := 20
  let n_1 : ℕ := 19
  let n_2 : ℕ := 18
  n_0 > n_1 ∧ n_1 > n_2 :=
by
  let n_0 := 20
  let n_1 := 19
  let n_2 := 18
  have h0 : n_0 > n_1 := by sorry
  have h1 : n_1 > n_2 := by sorry
  exact ⟨h0, h1⟩

end triangle_count_relationship_l278_27824
