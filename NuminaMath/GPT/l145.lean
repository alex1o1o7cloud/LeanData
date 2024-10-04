import Mathlib

namespace greatest_divisor_l145_145869

theorem greatest_divisor (d : ℕ) :
  (6215 % d = 23 ∧ 7373 % d = 29 ∧ 8927 % d = 35) → d = 36 :=
by
  sorry

end greatest_divisor_l145_145869


namespace joan_balloons_l145_145206

-- Defining the condition
def melanie_balloons : ℕ := 41
def total_balloons : ℕ := 81

-- Stating the theorem
theorem joan_balloons :
  ∃ (joan_balloons : ℕ), joan_balloons = total_balloons - melanie_balloons ∧ joan_balloons = 40 :=
by
  -- Placeholder for the proof
  sorry

end joan_balloons_l145_145206


namespace cos_relation_l145_145074

theorem cos_relation 
  (a b c A B C : ℝ)
  (h1 : a = b * Real.cos C + c * Real.cos B)
  (h2 : b = c * Real.cos A + a * Real.cos C)
  (h3 : c = a * Real.cos B + b * Real.cos A)
  (h_abc_nonzero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
  Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 + 2 * Real.cos A * Real.cos B * Real.cos C = 1 :=
sorry

end cos_relation_l145_145074


namespace investment_rate_l145_145964

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

end investment_rate_l145_145964


namespace train_length_l145_145146

theorem train_length
  (V L : ℝ)
  (h1 : L = V * 18)
  (h2 : L + 350 = V * 39) :
  L = 300 := 
by
  sorry

end train_length_l145_145146


namespace greatest_cars_with_ac_not_racing_stripes_l145_145264

def total_cars : ℕ := 100
def without_ac : ℕ := 49
def at_least_racing_stripes : ℕ := 51

theorem greatest_cars_with_ac_not_racing_stripes :
  (total_cars - without_ac) - (at_least_racing_stripes - without_ac) = 49 :=
by
  unfold total_cars without_ac at_least_racing_stripes
  sorry

end greatest_cars_with_ac_not_racing_stripes_l145_145264


namespace ella_age_l145_145821

theorem ella_age (s t e : ℕ) (h1 : s + t + e = 36) (h2 : e - 5 = s) (h3 : t + 4 = (3 * (s + 4)) / 4) : e = 15 := by
  sorry

end ella_age_l145_145821


namespace find_p_max_area_triangle_l145_145610

-- Define given conditions in lean
structure Parabola (p : ℝ) :=
(h_p : p > 0)
(eq : ∀ x y, x^2 = 2 * p * y)

structure Circle :=
(eq : ∀ x y, x^2 + (y + 4)^2 = 1)

-- Define the key distance condition
def min_distance (F P : ℝ × ℝ) (d : ℝ) : Prop :=
dist F P - 1 = d

-- Define problems to prove
theorem find_p (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle) (F : ℝ × ℝ) (d : ℝ)
  (hd : min_distance F (0, -4) 4) : p = 2 :=
sorry

theorem max_area_triangle (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle)
  (P A B : ℝ × ℝ) (PA PB : ℝ) : p = 2 → PA = P → PB = B → 
  ∃ A B : ℝ × ℝ, max_area (P A B) = 20 * sqrt 5 :=
sorry

end find_p_max_area_triangle_l145_145610


namespace min_value_of_a_plus_b_l145_145994

-- Definitions based on the conditions
variables (a b : ℝ)
def roots_real (a b : ℝ) : Prop := a^2 ≥ 8 * b ∧ b^2 ≥ a
def positive_vars (a b : ℝ) : Prop := a > 0 ∧ b > 0
def min_a_plus_b (a b : ℝ) : Prop := a + b = 6

-- Lean theorem statement
theorem min_value_of_a_plus_b (a b : ℝ) (hr : roots_real a b) (pv : positive_vars a b) : min_a_plus_b a b :=
sorry

end min_value_of_a_plus_b_l145_145994


namespace unique_hyperbolas_count_l145_145167

theorem unique_hyperbolas_count : 
  (Finset.card ((Finset.filter (fun b : ℕ => b > 1)
  (Finset.image (fun ⟨m, n⟩ => Nat.choose m n)
  ((Finset.Icc 1 5).product (Finset.Icc 1 5)))).toFinset)) = 6 := 
by
  sorry  

end unique_hyperbolas_count_l145_145167


namespace sequence_value_l145_145742

theorem sequence_value (r x y : ℝ) 
  (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : 
  x + y = 80 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end sequence_value_l145_145742


namespace circle_area_l145_145621

theorem circle_area (r : ℝ) (h : 6 / (2 * π * r) = r / 2) : π * r^2 = 3 :=
by
  sorry

end circle_area_l145_145621


namespace value_of_m_l145_145451

theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 2 ∧ x^2 - m * x + 8 = 0) → m = 6 := by
  sorry

end value_of_m_l145_145451


namespace smallest_steps_l145_145350

theorem smallest_steps (n : ℕ) :
  (n % 6 = 5) → (n % 7 = 1) → (n > 20) → n = 29 :=
by
  intros h1 h2 h3
  sorry

end smallest_steps_l145_145350


namespace people_going_to_movie_l145_145543

variable (people_per_car : ℕ) (number_of_cars : ℕ)

theorem people_going_to_movie (h1 : people_per_car = 6) (h2 : number_of_cars = 18) : 
    (people_per_car * number_of_cars) = 108 := 
by
  sorry

end people_going_to_movie_l145_145543


namespace average_loss_l145_145467

theorem average_loss (cost_per_lootbox : ℝ) (average_value_per_lootbox : ℝ) (total_spent : ℝ)
                      (h1 : cost_per_lootbox = 5)
                      (h2 : average_value_per_lootbox = 3.5)
                      (h3 : total_spent = 40) :
  (total_spent - (average_value_per_lootbox * (total_spent / cost_per_lootbox))) = 12 :=
by
  sorry

end average_loss_l145_145467


namespace Sandy_age_l145_145661

variable (S M : ℕ)

def condition1 (S M : ℕ) : Prop := M = S + 18
def condition2 (S M : ℕ) : Prop := S * 9 = M * 7

theorem Sandy_age (h1 : condition1 S M) (h2 : condition2 S M) : S = 63 := sorry

end Sandy_age_l145_145661


namespace bells_toll_together_l145_145980

theorem bells_toll_together : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 3 5) 8) 11) 15) 20 = 1320 := by
  sorry

end bells_toll_together_l145_145980


namespace relationship_abc_l145_145764

theorem relationship_abc (a b c : ℕ) (ha : a = 2^555) (hb : b = 3^444) (hc : c = 6^222) : a < c ∧ c < b := by
  sorry

end relationship_abc_l145_145764


namespace balls_in_box_l145_145561

def num_blue : Nat := 6
def num_red : Nat := 4
def num_green : Nat := 3 * num_blue
def num_yellow : Nat := 2 * num_red
def num_total : Nat := num_blue + num_red + num_green + num_yellow

theorem balls_in_box : num_total = 36 := by
  sorry

end balls_in_box_l145_145561


namespace g_of_3_equals_5_l145_145059

def g (x : ℝ) : ℝ := 2 * (x - 2) + 3

theorem g_of_3_equals_5 :
  g 3 = 5 :=
by
  sorry

end g_of_3_equals_5_l145_145059


namespace hulk_first_jump_more_than_500_l145_145500

def hulk_jumping_threshold : Prop :=
  ∃ n : ℕ, (3^n > 500) ∧ (∀ m < n, 3^m ≤ 500)

theorem hulk_first_jump_more_than_500 : ∃ n : ℕ, n = 6 ∧ hulk_jumping_threshold :=
  sorry

end hulk_first_jump_more_than_500_l145_145500


namespace molecular_weight_of_one_mole_l145_145379

theorem molecular_weight_of_one_mole 
  (total_weight : ℝ) (n_moles : ℝ) (mw_per_mole : ℝ)
  (h : total_weight = 792) (h2 : n_moles = 9) 
  (h3 : total_weight = n_moles * mw_per_mole) 
  : mw_per_mole = 88 :=
by
  sorry

end molecular_weight_of_one_mole_l145_145379


namespace min_value_x_add_one_div_y_l145_145300

theorem min_value_x_add_one_div_y (x y : ℝ) (h1 : x > 1) (h2 : x - y = 1) : 
x + 1 / y ≥ 3 :=
sorry

end min_value_x_add_one_div_y_l145_145300


namespace company_p_employees_december_l145_145976

theorem company_p_employees_december :
  let january_employees := 434.7826086956522
  let percent_more := 0.15
  let december_employees := january_employees + (percent_more * january_employees)
  december_employees = 500 :=
by
  sorry

end company_p_employees_december_l145_145976


namespace total_soaking_time_l145_145723

def stain_times (n_grass n_marinara n_coffee n_ink : Nat) (t_grass t_marinara t_coffee t_ink : Nat) : Nat :=
  n_grass * t_grass + n_marinara * t_marinara + n_coffee * t_coffee + n_ink * t_ink

theorem total_soaking_time :
  let shirt_grass_stains := 2
  let shirt_grass_time := 3
  let shirt_marinara_stains := 1
  let shirt_marinara_time := 7
  let pants_coffee_stains := 1
  let pants_coffee_time := 10
  let pants_ink_stains := 1
  let pants_ink_time := 5
  let socks_grass_stains := 1
  let socks_grass_time := 3
  let socks_marinara_stains := 2
  let socks_marinara_time := 7
  let socks_ink_stains := 1
  let socks_ink_time := 5
  let additional_ink_time := 2

  let shirt_time := stain_times shirt_grass_stains shirt_marinara_stains 0 0 shirt_grass_time shirt_marinara_time 0 0
  let pants_time := stain_times 0 0 pants_coffee_stains pants_ink_stains 0 0 pants_coffee_time pants_ink_time
  let socks_time := stain_times socks_grass_stains socks_marinara_stains 0 socks_ink_stains socks_grass_time socks_marinara_time 0 socks_ink_time
  let total_time := shirt_time + pants_time + socks_time
  let total_ink_stains := pants_ink_stains + socks_ink_stains
  let additional_ink_total_time := total_ink_stains * additional_ink_time
  let final_total_time := total_time + additional_ink_total_time

  final_total_time = 54 :=
by
  sorry

end total_soaking_time_l145_145723


namespace find_k_plus_m_l145_145361

def initial_sum := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
def initial_count := 9

def new_list_sum (m k : ℕ) := initial_sum + 8 * m + 9 * k
def new_list_count (m k : ℕ) := initial_count + m + k

def average_eq_73 (m k : ℕ) := (new_list_sum m k : ℝ) / (new_list_count m k : ℝ) = 7.3

theorem find_k_plus_m : ∃ (m k : ℕ), average_eq_73 m k ∧ (k + m = 21) :=
by
  sorry

end find_k_plus_m_l145_145361


namespace joan_balloons_l145_145205

-- Defining the condition
def melanie_balloons : ℕ := 41
def total_balloons : ℕ := 81

-- Stating the theorem
theorem joan_balloons :
  ∃ (joan_balloons : ℕ), joan_balloons = total_balloons - melanie_balloons ∧ joan_balloons = 40 :=
by
  -- Placeholder for the proof
  sorry

end joan_balloons_l145_145205


namespace exponent_multiplication_l145_145727

theorem exponent_multiplication (a : ℝ) : (a^3) * (a^2) = a^5 := 
by
  -- Using the property of exponents: a^m * a^n = a^(m + n)
  sorry

end exponent_multiplication_l145_145727


namespace rectangle_area_divisible_by_12_l145_145075

theorem rectangle_area_divisible_by_12 {a b c : ℕ} (h : a ^ 2 + b ^ 2 = c ^ 2) :
  12 ∣ (a * b) :=
sorry

end rectangle_area_divisible_by_12_l145_145075


namespace simplify_evaluate_l145_145495

theorem simplify_evaluate (x y : ℝ) (h : (x - 2)^2 + |1 + y| = 0) : 
  ((x - y) * (x + 2 * y) - (x + y)^2) / y = 1 :=
by
  sorry

end simplify_evaluate_l145_145495


namespace fill_grid_ways_l145_145293

theorem fill_grid_ways (n : ℕ) (h : n ≥ 3) :
  ∃ (ways : ℕ), ways = 2 ^ ((n - 1) ^ 2) * (n! ^ 3) :=
by
  use 2 ^ ((n - 1) ^ 2) * (n! ^ 3)
  sorry

end fill_grid_ways_l145_145293


namespace expression_value_l145_145317

theorem expression_value (x : ℕ) (h : x = 12) : (3 / 2 * x - 3 : ℚ) = 15 := by
  rw [h]
  norm_num
-- sorry to skip the proof if necessary
-- sorry 

end expression_value_l145_145317


namespace coed_softball_team_total_players_l145_145107

theorem coed_softball_team_total_players (M W : ℕ) 
  (h1 : W = M + 4) 
  (h2 : (M : ℚ) / W = 0.6363636363636364) :
  M + W = 18 := 
by sorry

end coed_softball_team_total_players_l145_145107


namespace necessary_but_not_sufficient_l145_145435

theorem necessary_but_not_sufficient (x y : ℝ) :
  (x = 0) → (x^2 + y^2 = 0) ↔ (x = 0 ∧ y = 0) :=
by sorry

end necessary_but_not_sufficient_l145_145435


namespace limit_of_sequence_l145_145940

theorem limit_of_sequence {ε : ℝ} (hε : ε > 0) : 
  ∃ (N : ℝ), ∀ (n : ℝ), n > N → |(2 * n^3) / (n^3 - 2) - 2| < ε :=
by
  sorry

end limit_of_sequence_l145_145940


namespace problem_1_problem_2_l145_145997

theorem problem_1 (n : ℕ) (h : n > 0) (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, (n > 0) → 
    (∃ α β, α + β = β * α + 1 ∧ 
            α * β = 1 / a n ∧ 
            a n * α^2 - a (n+1) * α + 1 = 0 ∧ 
            a n * β^2 - a (n+1) * β + 1 = 0)) :
  a (n + 1) = a n + 1 := sorry

theorem problem_2 (n : ℕ) (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n, (n > 0) → a (n+1) = a n + 1) :
  a n = n := sorry

end problem_1_problem_2_l145_145997


namespace range_of_a_l145_145993

/-- Given that the point (1, 1) is located inside the circle (x - a)^2 + (y + a)^2 = 4, 
    proving that the range of values for a is -1 < a < 1. -/
theorem range_of_a (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → 
  (-1 < a ∧ a < 1) :=
by
  intro h
  sorry

end range_of_a_l145_145993


namespace bus_stop_l145_145846

theorem bus_stop (M H : ℕ) 
  (h1 : H = 2 * (M - 15))
  (h2 : M - 15 = 5 * (H - 45)) :
  M = 40 ∧ H = 50 := 
sorry

end bus_stop_l145_145846


namespace system_has_integer_solution_l145_145493

theorem system_has_integer_solution (a b : ℤ) : 
  ∃ x y z t : ℤ, x + y + 2 * z + 2 * t = a ∧ 2 * x - 2 * y + z - t = b :=
by
  sorry

end system_has_integer_solution_l145_145493


namespace Joan_balloons_l145_145204

variable (J : ℕ) -- Joan's blue balloons

theorem Joan_balloons (h : J + 41 = 81) : J = 40 :=
by
  sorry

end Joan_balloons_l145_145204


namespace orange_sacks_after_95_days_l145_145617

-- Define the conditions as functions or constants
def harvest_per_day : ℕ := 150
def discard_per_day : ℕ := 135
def days_of_harvest : ℕ := 95

-- State the problem formally
theorem orange_sacks_after_95_days :
  (harvest_per_day - discard_per_day) * days_of_harvest = 1425 := 
by 
  sorry

end orange_sacks_after_95_days_l145_145617


namespace problem_C_l145_145701

theorem problem_C (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a * b :=
by sorry

end problem_C_l145_145701


namespace lines_are_perpendicular_l145_145878

noncomputable def line1 := {x : ℝ | ∃ y : ℝ, x + y - 1 = 0}
noncomputable def line2 := {x : ℝ | ∃ y : ℝ, x - y + 1 = 0}

theorem lines_are_perpendicular : 
  let slope1 := -1
  let slope2 := 1
  slope1 * slope2 = -1 := sorry

end lines_are_perpendicular_l145_145878


namespace speed_of_current_l145_145397

theorem speed_of_current (c r : ℝ) 
  (h1 : 12 = (c - r) * 6) 
  (h2 : 12 = (c + r) * 0.75) : 
  r = 7 := 
by
  sorry

end speed_of_current_l145_145397


namespace total_scoops_l145_145918

-- Define the conditions as variables
def flourCups := 3
def sugarCups := 2
def scoopSize := 1/3

-- Define what needs to be proved, i.e., the total amount of scoops needed
theorem total_scoops (flourCups sugarCups : ℚ) (scoopSize : ℚ) : 
  (flourCups / scoopSize) + (sugarCups / scoopSize) = 15 := 
by
  sorry

end total_scoops_l145_145918


namespace repetend_of_5_div_17_l145_145014

theorem repetend_of_5_div_17 : 
  ∃ repetend : ℕ, 
  decimal_repetend (5 / 17) = repetend ∧ 
  repetend = 2941176470588235 :=
by 
  skip

end repetend_of_5_div_17_l145_145014


namespace first_player_wins_l145_145880

-- Define the polynomial with placeholders
def P (X : ℤ) (a3 a2 a1 a0 : ℤ) : ℤ :=
  X^4 + a3 * X^3 + a2 * X^2 + a1 * X + a0

-- The statement that the first player can always win
theorem first_player_wins :
  ∀ (a3 a2 a1 a0 : ℤ),
    (a0 ≠ 0) → (a1 ≠ 0) → (a2 ≠ 0) → (a3 ≠ 0) →
    ∃ (strategy : ℕ → ℤ),
      (∀ n, strategy n ≠ 0) ∧
      ¬ ∃ (x y : ℤ), x ≠ y ∧ P x (strategy 0) (strategy 1) (strategy 2) (strategy 3) = 0 ∧ P y (strategy 0) (strategy 1) (strategy 2) (strategy 3) = 0 :=
by
  sorry

end first_player_wins_l145_145880


namespace cylinder_volume_scaling_l145_145700

theorem cylinder_volume_scaling (r h : ℝ) (V : ℝ) (V' : ℝ) 
  (h_original : V = π * r^2 * h) 
  (h_new : V' = π * (1.5 * r)^2 * (3 * h)) :
  V' = 6.75 * V := by
  sorry

end cylinder_volume_scaling_l145_145700


namespace evaluate_expression_equals_128_l145_145011

-- Define the expression as a Lean function
def expression : ℕ := (8^6) / (4 * 8^3)

-- Theorem stating that the expression equals 128
theorem evaluate_expression_equals_128 : expression = 128 := 
sorry

end evaluate_expression_equals_128_l145_145011


namespace geometric_progression_solution_l145_145935

-- Definitions and conditions as per the problem
def geometric_progression_first_term (b q : ℝ) : Prop :=
  b * (1 + q + q^2) = 21

def geometric_progression_sum_of_squares (b q : ℝ) : Prop :=
  b^2 * (1 + q^2 + q^4) = 189

-- The main theorem to be proven
theorem geometric_progression_solution (b q : ℝ) :
  (geometric_progression_first_term b q ∧ geometric_progression_sum_of_squares b q) →
  (b = 3 ∧ q = 2) ∨ (b = 12 ∧ q = 1 / 2) := 
by
  intros h
  sorry

end geometric_progression_solution_l145_145935


namespace time_to_walk_without_walkway_l145_145535

theorem time_to_walk_without_walkway 
  (vp vw : ℝ) 
  (h1 : (vp + vw) * 40 = 80) 
  (h2 : (vp - vw) * 120 = 80) : 
  80 / vp = 60 :=
by
  sorry

end time_to_walk_without_walkway_l145_145535


namespace problem_statement_l145_145873

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b < a) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = |Real.log x|) (h_eq : f a = f b) :
  a * b = 1 ∧ Real.exp a + Real.exp b > 2 * Real.exp 1 ∧ (1 / a)^2 - b + 5 / 4 ≥ 1 :=
by
  sorry

end problem_statement_l145_145873


namespace restocked_bags_correct_l145_145961

def initial_stock := 55
def sold_bags := 23
def final_stock := 164

theorem restocked_bags_correct :
  (final_stock - (initial_stock - sold_bags)) = 132 :=
by
  -- The proof would go here, but we use sorry to skip it.
  sorry

end restocked_bags_correct_l145_145961


namespace eq_sum_disjoint_subsets_of_10_twodigits_l145_145657

theorem eq_sum_disjoint_subsets_of_10_twodigits (E : Finset ℕ)
  (h_card : E.card = 10)
  (h_range : ∀ x ∈ E, 10 ≤ x ∧ x ≤ 99) :
  ∃ (A B : Finset ℕ), A ⊆ E ∧ B ⊆ E ∧ A ∩ B = ∅ ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ A.sum id = B.sum id := by
  sorry

end eq_sum_disjoint_subsets_of_10_twodigits_l145_145657


namespace set_inter_and_complement_l145_145314

def U : Set ℕ := {2, 3, 4, 5, 6, 7}
def A : Set ℕ := {4, 5, 7}
def B : Set ℕ := {4, 6}

theorem set_inter_and_complement :
  A ∩ (U \ B) = {5, 7} := by
  sorry

end set_inter_and_complement_l145_145314


namespace more_spent_on_keychains_bracelets_than_tshirts_l145_145856

-- Define the conditions as variables
variable (spent_keychains_bracelets spent_total_spent : ℝ)
variable (spent_keychains_bracelets_eq : spent_keychains_bracelets = 347.00)
variable (spent_total_spent_eq : spent_total_spent = 548.00)

-- Using these conditions, define the problem to prove the desired result
theorem more_spent_on_keychains_bracelets_than_tshirts :
  spent_keychains_bracelets - (spent_total_spent - spent_keychains_bracelets) = 146.00 :=
by
  rw [spent_keychains_bracelets_eq, spent_total_spent_eq]
  sorry

end more_spent_on_keychains_bracelets_than_tshirts_l145_145856


namespace walking_time_l145_145884

-- Define the conditions as Lean definitions
def minutes_in_hour : Nat := 60

def work_hours : Nat := 6
def work_minutes := work_hours * minutes_in_hour
def sitting_interval : Nat := 90
def walking_time_per_interval : Nat := 10

-- State the main theorem
theorem walking_time (h1 : 10 * 90 = 600) (h2 : 10 * (work_hours * 60) / 90 = 40) : 
  work_minutes / sitting_interval * walking_time_per_interval = 40 :=
  sorry

end walking_time_l145_145884


namespace banana_nn_together_count_l145_145864

open Finset

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def arrangements_banana_with_nn_together : ℕ :=
  (factorial 4) / (factorial 3)

theorem banana_nn_together_count : arrangements_banana_with_nn_together = 4 := by
  sorry

end banana_nn_together_count_l145_145864


namespace value_of_a_l145_145104

theorem value_of_a (a : ℝ) (h : (a, 0) ∈ {p : ℝ × ℝ | p.2 = p.1 + 8}) : a = -8 :=
sorry

end value_of_a_l145_145104


namespace capital_of_z_l145_145839

theorem capital_of_z (x y z : ℕ) (annual_profit z_share : ℕ) (months_x months_y months_z : ℕ) 
    (rx ry : ℕ) (r : ℚ) :
  x = 20000 →
  y = 25000 →
  z_share = 14000 →
  annual_profit = 50000 →
  rx = 240000 →
  ry = 300000 →
  months_x = 12 →
  months_y = 12 →
  months_z = 7 →
  r = 7 / 25 →
  z * months_z * r = z_share / (rx + ry + z * months_z) →
  z = 30000 := 
by intros; sorry

end capital_of_z_l145_145839


namespace sample_space_correct_events_A_and_B_not_independent_most_likely_sum_is_5_l145_145626

/-- Conditions: A bag contains 4 balls labeled 1, 2, 3, 4. Two balls are drawn without replacement. 
Event A: drawing the ball labeled 2 on the first draw. Event B: the sum of the numbers on the two balls drawn is 5. -/
def sampleSpace : Set (Nat × Nat) :=
  {(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3)}

def eventA : Set (Nat × Nat) :=
  {(2, 1), (2, 3), (2, 4)}

def eventB : Set (Nat × Nat) :=
  {(1, 4), (2, 3), (3, 2), (4, 1)}

theorem sample_space_correct :
  sampleSpace = {(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3)} :=
sorry

theorem events_A_and_B_not_independent :
  ¬ (ProbSpace.independence eventA eventB) :=
sorry

theorem most_likely_sum_is_5 :
  (∃ (subset : Set (Nat × Nat)), subset ⊆ sampleSpace ∧
  (∀ (sums : Nat), (sums ∈ (5 :: List.nil) → (ProbSpace.probability_of subset) > 
  (ProbSpace.probability_of {p | p.fst + p.snd = sums}) → sums = 5))) := sorry

end sample_space_correct_events_A_and_B_not_independent_most_likely_sum_is_5_l145_145626


namespace number_machine_output_l145_145957

def machine (x : ℕ) : ℕ := x + 15 - 6

theorem number_machine_output : machine 68 = 77 := by
  sorry

end number_machine_output_l145_145957


namespace winning_prob_correct_l145_145147

noncomputable def probability_winning_contest : ℚ :=
  let probability_one_correct := 1 / 4
  let probability_one_incorrect := 3 / 4
  let probability_all_four_correct := probability_one_correct ^ 4
  let combinations_three_correct := 4
  let probability_three_correct := combinations_three_correct * (probability_one_correct ^ 3 * probability_one_incorrect)
  probability_all_four_correct + probability_three_correct

theorem winning_prob_correct :
  probability_winning_contest = 13 / 256 :=
by
  sorry

end winning_prob_correct_l145_145147


namespace complement_U_A_l145_145613

def U : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def A : Set ℝ := {x | 3 ≤ 2 * x - 1 ∧ 2 * x - 1 < 5}

theorem complement_U_A : (U \ A) = {x | (0 ≤ x ∧ x < 2) ∨ (3 ≤ x)} := sorry

end complement_U_A_l145_145613


namespace sin_inverse_equation_l145_145004

noncomputable def a := Real.arcsin (4/5)
noncomputable def b := Real.arctan 1
noncomputable def c := Real.arccos (1/3)
noncomputable def sin_a_plus_b_minus_c := Real.sin (a + b - c)

theorem sin_inverse_equation : sin_a_plus_b_minus_c = 11 / 15 := sorry

end sin_inverse_equation_l145_145004


namespace joe_total_time_to_school_l145_145905

theorem joe_total_time_to_school:
  ∀ (d r_w: ℝ), (1 / 3) * d = r_w * 9 →
                  4 * r_w * (2 * (r_w * 9) / (3 * (4 * r_w))) = (2 / 3) * d →
                  (1 / 3) * d / r_w + (2 / 3) * d / (4 * r_w) = 13.5 :=
by
  intros d r_w h1 h2
  sorry

end joe_total_time_to_school_l145_145905


namespace Lou_receives_lollipops_l145_145809

theorem Lou_receives_lollipops
  (initial_lollipops : ℕ)
  (fraction_to_Emily : ℚ)
  (lollipops_kept : ℕ)
  (lollipops_given_to_Lou : ℕ) :
  initial_lollipops = 42 →
  fraction_to_Emily = 2 / 3 →
  lollipops_kept = 4 →
  lollipops_given_to_Lou = initial_lollipops - (initial_lollipops * fraction_to_Emily).natAbs - lollipops_kept →
  lollipops_given_to_Lou = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end Lou_receives_lollipops_l145_145809


namespace find_large_number_l145_145944

theorem find_large_number (L S : ℕ) (h1 : L - S = 1515) (h2 : L = 16 * S + 15) : L = 1615 :=
sorry

end find_large_number_l145_145944


namespace minyoung_yoojung_flowers_l145_145489

theorem minyoung_yoojung_flowers (m y : ℕ) 
(h1 : m = 4 * y) 
(h2 : m = 24) : 
m + y = 30 := 
by
  sorry

end minyoung_yoojung_flowers_l145_145489


namespace complex_expression_evaluation_l145_145559

theorem complex_expression_evaluation (i : ℂ) (h : i^2 = -1) : i^3 * (1 - i)^2 = -2 :=
by
  -- Placeholder for the actual proof which is skipped here
  sorry

end complex_expression_evaluation_l145_145559


namespace movie_production_l145_145469

theorem movie_production
  (LJ_annual_production : ℕ)
  (Johnny_additional_percent : ℕ)
  (LJ_annual_production_val : LJ_annual_production = 220)
  (Johnny_additional_percent_val : Johnny_additional_percent = 25) :
  (Johnny_additional_percent / 100 * LJ_annual_production + LJ_annual_production + LJ_annual_production) * 5 = 2475 :=
by
  have Johnny_additional_movies : ℕ := Johnny_additional_percent * LJ_annual_production / 100
  have Johnny_annual_production : ℕ := Johnny_additional_movies + LJ_annual_production
  have combined_annual_production : ℕ := Johnny_annual_production + LJ_annual_production
  have combined_five_years_production : ℕ := combined_annual_production * 5

  rw [LJ_annual_production_val, Johnny_additional_percent_val]
  have Johnny_additional_movies_calc : Johnny_additional_movies = 55 := by sorry
  have Johnny_annual_production_calc : Johnny_annual_production = 275 := by sorry
  have combined_annual_production_calc : combined_annual_production = 495 := by sorry
  have combined_five_years_production_calc : combined_five_years_production = 2475 := by sorry
  
  exact combined_five_years_production_calc.symm

end movie_production_l145_145469


namespace tangent_product_constant_l145_145115

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

end tangent_product_constant_l145_145115


namespace weight_of_hollow_golden_sphere_l145_145902

theorem weight_of_hollow_golden_sphere : 
  let diameter := 12
  let thickness := 0.3
  let pi := (3 : Real)
  let outer_radius := diameter / 2
  let inner_radius := (outer_radius - thickness)
  let outer_volume := (4 / 3) * pi * outer_radius^3
  let inner_volume := (4 / 3) * pi * inner_radius^3
  let gold_volume := outer_volume - inner_volume
  let weight_per_cubic_inch := 1
  let weight := gold_volume * weight_per_cubic_inch
  weight = 123.23 :=
by
  sorry

end weight_of_hollow_golden_sphere_l145_145902


namespace find_time_interval_l145_145627

-- Definitions for conditions
def birthRate : ℕ := 4
def deathRate : ℕ := 2
def netIncreaseInPopulationPerInterval (T : ℕ) : ℕ := birthRate - deathRate
def totalTimeInOneDay : ℕ := 86400
def netIncreaseInOneDay (T : ℕ) : ℕ := (totalTimeInOneDay / T) * (netIncreaseInPopulationPerInterval T)

-- Theorem statement
theorem find_time_interval (T : ℕ) (h1 : netIncreaseInPopulationPerInterval T = 2) (h2 : netIncreaseInOneDay T = 86400) : T = 2 :=
sorry

end find_time_interval_l145_145627


namespace area_of_quadrilateral_l145_145018

theorem area_of_quadrilateral (d o1 o2 : ℝ) (h1 : d = 24) (h2 : o1 = 9) (h3 : o2 = 6) :
  (1 / 2 * d * o1) + (1 / 2 * d * o2) = 180 :=
by {
  sorry
}

end area_of_quadrilateral_l145_145018


namespace intersection_M_N_l145_145779

-- Define sets M and N
def M := {x : ℝ | x^2 - 2*x ≤ 0}
def N := {x : ℝ | -2 < x ∧ x < 1}

-- The theorem stating the intersection of M and N equals [0, 1)
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := 
by
  sorry

end intersection_M_N_l145_145779


namespace factor_difference_of_squares_l145_145752

theorem factor_difference_of_squares (x : ℝ) : 36 - 9 * x^2 = 9 * (2 - x) * (2 + x) :=
by
  sorry

end factor_difference_of_squares_l145_145752


namespace find_tuesday_temp_l145_145283

variable (temps : List ℝ) (avg : ℝ) (len : ℕ) 

theorem find_tuesday_temp (h1 : temps = [99.1, 98.2, 99.3, 99.8, 99, 98.9, tuesday_temp])
                         (h2 : avg = 99)
                         (h3 : len = 7)
                         (h4 : (temps.sum / len) = avg) :
                         tuesday_temp = 98.7 := 
sorry

end find_tuesday_temp_l145_145283


namespace arithmetic_sequence_ratio_q_l145_145768

theorem arithmetic_sequence_ratio_q :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ), 
    (0 < q) →
    (S 2 = 3 * a 2 + 2) →
    (S 4 = 3 * a 4 + 2) →
    (q = 3 / 2) :=
by
  sorry

end arithmetic_sequence_ratio_q_l145_145768


namespace linda_savings_l145_145388

theorem linda_savings (S : ℕ) (h1 : (3 / 4) * S = x) (h2 : (1 / 4) * S = 240) : S = 960 :=
by
  sorry

end linda_savings_l145_145388


namespace part_a_l145_145386

theorem part_a (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : a + b = 1) : a * b = 0 := 
by 
  sorry

end part_a_l145_145386


namespace twigs_per_branch_l145_145634

/-- Definitions -/
def total_branches : ℕ := 30
def total_leaves : ℕ := 12690
def percentage_4_leaves : ℝ := 0.30
def leaves_per_twig_4_leaves : ℕ := 4
def percentage_5_leaves : ℝ := 0.70
def leaves_per_twig_5_leaves : ℕ := 5

/-- Given conditions translated to Lean -/
def hypothesis (T : ℕ) : Prop :=
  (percentage_4_leaves * T * leaves_per_twig_4_leaves) +
  (percentage_5_leaves * T * leaves_per_twig_5_leaves) = total_leaves

/-- The main theorem to prove -/
theorem twigs_per_branch
  (T : ℕ)
  (h : hypothesis T) :
  (T / total_branches) = 90 :=
sorry

end twigs_per_branch_l145_145634


namespace min_value_of_expression_l145_145211

theorem min_value_of_expression (x y : ℤ) (h : 4 * x + 5 * y = 7) : ∃ k : ℤ, 
  5 * Int.natAbs (3 + 5 * k) - 3 * Int.natAbs (-1 - 4 * k) = 1 :=
sorry

end min_value_of_expression_l145_145211


namespace exists_mod_inv_l145_145068

theorem exists_mod_inv (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (h : ¬ a ∣ p) : ∃ b : ℕ, a * b ≡ 1 [MOD p] :=
by
  sorry

end exists_mod_inv_l145_145068


namespace how_many_times_l145_145999

theorem how_many_times (a b : ℝ) (h1 : a = 0.5) (h2 : b = 0.01) : a / b = 50 := 
by 
  sorry

end how_many_times_l145_145999


namespace time_to_finish_work_l145_145320

theorem time_to_finish_work (a b c : ℕ) (h1 : 1/a + 1/9 + 1/18 = 1/4) : a = 12 :=
by
  sorry

end time_to_finish_work_l145_145320


namespace odd_function_max_to_min_l145_145897

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_max_to_min (a b : ℝ) (f : ℝ → ℝ)
  (hodd : is_odd_function f)
  (hmax : ∃ x : ℝ, x > 0 ∧ (a * f x + b * x + 1) = 2) :
  ∃ y : ℝ, y < 0 ∧ (a * f y + b * y + 1) = 0 :=
sorry

end odd_function_max_to_min_l145_145897


namespace drop_volume_l145_145504

theorem drop_volume :
  let leak_rate := 3 -- drops per minute
  let pot_volume := 3 * 1000 -- volume in milliliters
  let time := 50 -- minutes
  let total_drops := leak_rate * time -- total number of drops
  (pot_volume / total_drops) = 20 := 
by
  let leak_rate : ℕ := 3
  let pot_volume : ℕ := 3 * 1000
  let time : ℕ := 50
  let total_drops := leak_rate * time
  have h : (pot_volume / total_drops) = 20 := by sorry
  exact h

end drop_volume_l145_145504


namespace production_movie_count_l145_145468

theorem production_movie_count
  (LJ_annual : ℕ)
  (H1 : LJ_annual = 220)
  (H2 : ∀ n, n = 275 → n = LJ_annual + (LJ_annual * 25 / 100))
  (years : ℕ)
  (H3 : years = 5) :
  (LJ_annual + 275) * years = 2475 :=
by {
  sorry
}

end production_movie_count_l145_145468


namespace sandy_goal_hours_l145_145494

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

end sandy_goal_hours_l145_145494


namespace selling_price_l145_145332

def initial_cost : ℕ := 600
def food_cost_per_day : ℕ := 20
def number_of_days : ℕ := 40
def vaccination_and_deworming_cost : ℕ := 500
def profit : ℕ := 600

theorem selling_price (S : ℕ) :
  S = initial_cost + (food_cost_per_day * number_of_days) + vaccination_and_deworming_cost + profit :=
by
  sorry

end selling_price_l145_145332


namespace total_rainfall_in_2004_l145_145625

noncomputable def average_monthly_rainfall_2003 : ℝ := 35.0
noncomputable def average_monthly_rainfall_2004 : ℝ := average_monthly_rainfall_2003 + 4.0
noncomputable def total_rainfall_2004 : ℝ := 
  let regular_months := 11 * average_monthly_rainfall_2004
  let daily_rainfall_feb := average_monthly_rainfall_2004 / 30
  let feb_rain := daily_rainfall_feb * 29 
  regular_months + feb_rain

theorem total_rainfall_in_2004 : total_rainfall_2004 = 466.7 := by
  sorry

end total_rainfall_in_2004_l145_145625


namespace star_six_three_l145_145889

-- Definition of the operation
def star (a b : ℕ) : ℕ := 4 * a + 5 * b - 2 * a * b

-- Statement to prove
theorem star_six_three : star 6 3 = 3 := by
  sorry

end star_six_three_l145_145889


namespace find_a_l145_145325

theorem find_a (a : ℝ) : (∃ x y : ℝ, y = 4 - 3 * x ∧ y = 2 * x - 1 ∧ y = a * x + 7) → a = 6 := 
by
  sorry

end find_a_l145_145325


namespace remaining_amount_l145_145981

def initial_amount : ℕ := 18
def spent_amount : ℕ := 16

theorem remaining_amount : initial_amount - spent_amount = 2 := 
by sorry

end remaining_amount_l145_145981


namespace least_people_to_complete_job_on_time_l145_145005

theorem least_people_to_complete_job_on_time
  (total_duration : ℕ)
  (initial_days : ℕ)
  (initial_people : ℕ)
  (initial_work_done : ℚ)
  (efficiency_multiplier : ℚ)
  (remaining_work_fraction : ℚ)
  (remaining_days : ℕ)
  (resulting_people : ℕ)
  (work_rate_doubled : ℕ → ℚ → ℚ)
  (final_resulting_people : ℚ)
  : initial_work_done = 1/4 →
    efficiency_multiplier = 2 →
    remaining_work_fraction = 3/4 →
    total_duration = 40 →
    initial_days = 10 →
    initial_people = 12 →
    remaining_days = 20 →
    work_rate_doubled 12 2 = 24 →
    final_resulting_people = (1/2) →
    resulting_people = 6 :=
sorry

end least_people_to_complete_job_on_time_l145_145005


namespace total_amount_paid_l145_145280

theorem total_amount_paid (g_p g_q m_p m_q : ℝ) (g_d g_t m_d m_t : ℝ) : 
    g_p = 70 -> g_q = 8 -> g_d = 0.05 -> g_t = 0.08 -> 
    m_p = 55 -> m_q = 9 -> m_d = 0.07 -> m_t = 0.11 -> 
    (g_p * g_q * (1 - g_d) * (1 + g_t) + m_p * m_q * (1 - m_d) * (1 + m_t)) = 1085.55 := by 
    sorry

end total_amount_paid_l145_145280


namespace remainder_2468135792_mod_101_l145_145695

theorem remainder_2468135792_mod_101 : 
  2468135792 % 101 = 47 := 
sorry

end remainder_2468135792_mod_101_l145_145695


namespace max_gcd_coprime_l145_145581

theorem max_gcd_coprime (x y : ℤ) (h : Int.gcd x y = 1) : 
  Int.gcd (x + 2015 * y) (y + 2015 * x) ≤ 4060224 :=
sorry

end max_gcd_coprime_l145_145581


namespace max_plus_min_l145_145189

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 (x₁ x₂ : ℝ) : f (x₁ + x₂) = f x₁ + f x₂ - 2016
axiom condition2 (x : ℝ) : x > 0 → f x > 2016

theorem max_plus_min (M N : ℝ) (hM : M = f 2016) (hN : N = f (-2016)) : M + N = 4032 :=
by
  sorry

end max_plus_min_l145_145189


namespace degrees_for_combined_research_l145_145135

-- Define the conditions as constants.
def microphotonics_percentage : ℝ := 0.10
def home_electronics_percentage : ℝ := 0.24
def food_additives_percentage : ℝ := 0.15
def gmo_percentage : ℝ := 0.29
def industrial_lubricants_percentage : ℝ := 0.08
def nanotechnology_percentage : ℝ := 0.07

noncomputable def remaining_percentage : ℝ :=
  1 - (microphotonics_percentage + home_electronics_percentage + food_additives_percentage +
    gmo_percentage + industrial_lubricants_percentage + nanotechnology_percentage)

noncomputable def total_percentage : ℝ :=
  remaining_percentage + nanotechnology_percentage

noncomputable def degrees_in_circle : ℝ := 360

noncomputable def degrees_representing_combined_research : ℝ :=
  total_percentage * degrees_in_circle

-- State the theorem to prove the correct answer
theorem degrees_for_combined_research : degrees_representing_combined_research = 50.4 :=
by
  -- Proof will go here
  sorry

end degrees_for_combined_research_l145_145135


namespace max_sin_pow_two_l145_145378

theorem max_sin_pow_two (n : ℕ) (hn : n > 0) : ∃ M : ℝ, M = 0.8988 ∧ ∀ k, 0 ≤ k → 0 ≤ sin (2^k) ∧ sin (2^k) ≤ M := sorry

end max_sin_pow_two_l145_145378


namespace lines_perpendicular_slope_l145_145882

theorem lines_perpendicular_slope (k : ℝ) :
  (∀ (x : ℝ), k * 2 = -1) → k = (-1:ℝ)/2 :=
by
  sorry

end lines_perpendicular_slope_l145_145882


namespace real_solutions_x_inequality_l145_145009

theorem real_solutions_x_inequality (x : ℝ) :
  (∃ y : ℝ, y^2 + 6 * x * y + x + 8 = 0) ↔ (x ≤ -8 / 9 ∨ x ≥ 1) := 
sorry

end real_solutions_x_inequality_l145_145009


namespace parametric_to_ordinary_l145_145103

theorem parametric_to_ordinary (θ : ℝ) (x y : ℝ) : 
  x = Real.cos θ ^ 2 →
  y = 2 * Real.sin θ ^ 2 →
  (x ∈ Set.Icc 0 1) → 
  2 * x + y - 2 = 0 :=
by
  intros hx hy h_range
  sorry

end parametric_to_ordinary_l145_145103


namespace total_businesses_l145_145408

theorem total_businesses (B : ℕ) (h1 : B / 2 + B / 3 + 12 = B) : B = 72 :=
sorry

end total_businesses_l145_145408


namespace average_waiting_time_l145_145962

/-- 
A traffic light at a pedestrian crossing allows pedestrians to cross the street 
for one minute and prohibits crossing for two minutes. Prove that the average 
waiting time for a pedestrian who arrives at the intersection is 40 seconds.
-/ 
theorem average_waiting_time (pG : ℝ) (pR : ℝ) (eTG : ℝ) (eTR : ℝ) (cycle : ℝ) :
  pG = 1 / 3 ∧ pR = 2 / 3 ∧ eTG = 0 ∧ eTR = 1 ∧ cycle = 3 → 
  (eTG * pG + eTR * pR) * (60 / cycle) = 40 :=
by
  sorry

end average_waiting_time_l145_145962


namespace remaining_area_is_correct_l145_145546

-- Define the large rectangle's side lengths
def large_rectangle_length1 (x : ℝ) := x + 7
def large_rectangle_length2 (x : ℝ) := x + 5

-- Define the hole's side lengths
def hole_length1 (x : ℝ) := x + 1
def hole_length2 (x : ℝ) := x + 4

-- Calculate the areas
def large_rectangle_area (x : ℝ) := large_rectangle_length1 x * large_rectangle_length2 x
def hole_area (x : ℝ) := hole_length1 x * hole_length2 x

-- Define the remaining area after subtracting the hole area from the large rectangle area
def remaining_area (x : ℝ) := large_rectangle_area x - hole_area x

-- Problem statement: prove that the remaining area is 7x + 31
theorem remaining_area_is_correct (x : ℝ) : remaining_area x = 7 * x + 31 :=
by 
  -- The proof should be provided here, but for now we use 'sorry' to omit it
  sorry

end remaining_area_is_correct_l145_145546


namespace complex_expression_identity_l145_145804

open Complex

theorem complex_expression_identity
  (x y : ℂ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hxy : x^2 + x * y + y^2 = 0) :
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 :=
by
  sorry

end complex_expression_identity_l145_145804


namespace absolute_value_expression_l145_145339

theorem absolute_value_expression {x : ℤ} (h : x = 2024) :
  abs (abs (abs x - x) - abs x) = 0 :=
by
  sorry

end absolute_value_expression_l145_145339


namespace promotional_pricing_plan_l145_145715

theorem promotional_pricing_plan (n : ℕ) : 
  (8 * 100 = 800) ∧ 
  (∀ n > 100, 6 * n < 640) :=
by
  sorry

end promotional_pricing_plan_l145_145715


namespace b_gives_c_start_l145_145703

variable (Va Vb Vc : ℝ)

-- Conditions given in the problem
def condition1 : Prop := Va / Vb = 1000 / 930
def condition2 : Prop := Va / Vc = 1000 / 800
def race_distance : ℝ := 1000

-- Proposition to prove
theorem b_gives_c_start (h1 : condition1 Va Vb) (h2 : condition2 Va Vc) :
  ∃ x : ℝ, (1000 - x) / 1000 = (930 / 800) :=
sorry

end b_gives_c_start_l145_145703


namespace no_two_digit_prime_with_digit_sum_9_l145_145032

-- Define the concept of a two-digit number
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Define the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the concept of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem statement
theorem no_two_digit_prime_with_digit_sum_9 :
  ∀ n : ℕ, is_two_digit n ∧ digit_sum n = 9 → ¬is_prime n :=
by {
  -- proof omitted
  sorry
}  

end no_two_digit_prime_with_digit_sum_9_l145_145032


namespace four_letter_list_product_l145_145288

def letter_value (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then (c.toNat - 'A'.toNat + 1) else 0

def list_product (s : String) : Nat :=
  s.foldl (λ acc c => acc * letter_value c) 1

def target_product : Nat :=
  list_product "TUVW"

theorem four_letter_list_product : 
  ∀ (s1 s2 : String), s1.toList.all (λ c => 'A' ≤ c ∧ c ≤ 'Z') → s2.toList.all (λ c => 'A' ≤ c ∧ c ≤ 'Z') →
  s1.length = 4 → s2.length = 4 →
  list_product s1 = target_product → s2 = "BEHK" :=
by
  sorry

end four_letter_list_product_l145_145288


namespace minimum_value_expression_l145_145473

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a = 1 ∧ b = 1 ∧ c = 1) :
  (a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2) / (a * b * c) = 48 * Real.sqrt 6 := 
by
  sorry

end minimum_value_expression_l145_145473


namespace B_pow_2048_l145_145640

open Real Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![cos (π / 4), 0, -sin (π / 4)],
    ![0, 1, 0],
    ![sin (π / 4), 0, cos (π / 4)]]

theorem B_pow_2048 :
  B ^ 2048 = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by
  sorry

end B_pow_2048_l145_145640


namespace students_on_right_side_l145_145920

-- Define the total number of students and the number of students on the left side
def total_students : ℕ := 63
def left_students : ℕ := 36

-- Define the number of students on the right side using subtraction
def right_students (total_students left_students : ℕ) : ℕ := total_students - left_students

-- Theorem: Prove that the number of students on the right side is 27
theorem students_on_right_side : right_students total_students left_students = 27 := by
  sorry

end students_on_right_side_l145_145920


namespace sum_of_coordinates_B_l145_145656

theorem sum_of_coordinates_B :
  ∃ (x y : ℝ), (3, 5) = ((x + 6) / 2, (y + 8) / 2) ∧ x + y = 2 := by
  sorry

end sum_of_coordinates_B_l145_145656


namespace log2_sufficient_not_necessary_l145_145872

noncomputable def baseTwoLog (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem log2_sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (baseTwoLog a > baseTwoLog b) ↔ (a > b) :=
sorry

end log2_sufficient_not_necessary_l145_145872


namespace production_statistics_relation_l145_145235

noncomputable def a : ℚ := (10 + 12 + 14 + 14 + 15 + 15 + 16 + 17 + 17 + 17) / 10
noncomputable def b : ℚ := (15 + 15) / 2
noncomputable def c : ℤ := 17

theorem production_statistics_relation : c > a ∧ a > b :=
by
  sorry

end production_statistics_relation_l145_145235


namespace max_value_of_f_l145_145055

open Real

-- Given conditions
def m := ((1 : ℝ) / 2, 3)
def n := (π / 6, 0)
def vectorOp (a : ℝ × ℝ) (b1 : ℝ) : ℝ × ℝ := (a.1 * b1, a.2 * b1)
def P_path := λ x, (x, sin x)
def Q_path := λ (x y : ℝ), let (x0, y0) := P_path x in ((x0/2) + π/6, 3 * y0)
def f (x : ℝ) := 3 * sin (2 * x - π / 3)

-- Theorem statement: Maximum value of f(x)
theorem max_value_of_f : ∃ x, f x = 3 :=
sorry

end max_value_of_f_l145_145055


namespace collinear_vectors_l145_145614

-- Definitions
def a : ℝ × ℝ := (2, 4)
def b (x : ℝ) : ℝ × ℝ := (x, 6)

-- Proof statement
theorem collinear_vectors (x : ℝ) (h : ∃ k : ℝ, b x = k • a) : x = 3 :=
by sorry

end collinear_vectors_l145_145614


namespace sufficient_not_necessary_l145_145267

theorem sufficient_not_necessary (a : ℝ) : (a > 1 → a^2 > 1) ∧ ¬(a^2 > 1 → a > 1) :=
by {
  sorry
}

end sufficient_not_necessary_l145_145267


namespace tom_mileage_per_gallon_l145_145685

-- Definitions based on the given conditions
def daily_mileage : ℕ := 75
def cost_per_gallon : ℕ := 3
def amount_spent_in_10_days : ℕ := 45
def days : ℕ := 10

-- Main theorem to prove
theorem tom_mileage_per_gallon : 
  (amount_spent_in_10_days / cost_per_gallon) * 75 * days = 50 :=
by
  sorry

end tom_mileage_per_gallon_l145_145685


namespace ticket_cost_at_30_years_l145_145257

noncomputable def initial_cost : ℝ := 1000000
noncomputable def halving_period_years : ℕ := 10
noncomputable def halving_factor : ℝ := 0.5

def cost_after_n_years (initial_cost : ℝ) (halving_factor : ℝ) (years : ℕ) (period : ℕ) : ℝ :=
  initial_cost * halving_factor ^ (years / period)

theorem ticket_cost_at_30_years (initial_cost halving_factor : ℝ) (years period: ℕ) 
  (h_initial_cost : initial_cost = 1000000)
  (h_halving_factor : halving_factor = 0.5)
  (h_years : years = 30)
  (h_period : period = halving_period_years) : 
  cost_after_n_years initial_cost halving_factor years period = 125000 :=
by 
  sorry

end ticket_cost_at_30_years_l145_145257


namespace line_m_eq_line_n_eq_l145_145795
-- Definitions for conditions
def point_A : ℝ × ℝ := (-2, 1)
def line_l (x y : ℝ) := 2 * x - y - 3 = 0

-- Proof statement for part (1)
theorem line_m_eq :
  ∃ (m : ℝ → ℝ → Prop), (∀ x y, m x y ↔ (2 * x - y + 5 = 0)) ∧
    (∀ x y, line_l x y → m (-2) 1 → True) :=
sorry

-- Proof statement for part (2)
theorem line_n_eq :
  ∃ (n : ℝ → ℝ → Prop), (∀ x y, n x y ↔ (x + 2 * y = 0)) ∧
    (∀ x y, line_l x y → n (-2) 1 → True) :=
sorry

end line_m_eq_line_n_eq_l145_145795


namespace kira_travel_time_l145_145113

theorem kira_travel_time :
  let time_between_stations := 2 * 60 -- converting hours to minutes
  let break_time := 30 -- in minutes
  let total_time := 2 * time_between_stations + break_time
  total_time = 270 :=
by
  let time_between_stations := 2 * 60
  let break_time := 30
  let total_time := 2 * time_between_stations + break_time
  exact rfl

end kira_travel_time_l145_145113


namespace unique_non_zero_b_for_unique_x_solution_l145_145569

theorem unique_non_zero_b_for_unique_x_solution (c : ℝ) (hc : c ≠ 0) :
  c = 3 / 2 ↔ ∃! b : ℝ, b ≠ 0 ∧ ∃ x : ℝ, (x^2 + (b + 3 / b) * x + c = 0) ∧ 
  ∀ x1 x2 : ℝ, (x1^2 + (b + 3 / b) * x1 + c = 0) ∧ (x2^2 + (b + 3 / b) * x2 + c = 0) → x1 = x2 :=
sorry

end unique_non_zero_b_for_unique_x_solution_l145_145569


namespace minimum_value_expression_l145_145475

open Real

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2)) / (a * b * c) ≥ 216 :=
sorry

end minimum_value_expression_l145_145475


namespace min_det_is_neg_six_l145_145823

-- Define the set of possible values for a, b, c, d
def values : List ℤ := [-1, 1, 2]

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- State the theorem that the minimum value of the determinant is -6
theorem min_det_is_neg_six :
  ∃ (a b c d : ℤ), a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ 
  (∀ (a' b' c' d' : ℤ), a' ∈ values → b' ∈ values → c' ∈ values → d' ∈ values → det a b c d ≤ det a' b' c' d') ∧ det a b c d = -6 :=
by
  sorry

end min_det_is_neg_six_l145_145823


namespace interval_of_decrease_l145_145570

def quadratic (x : ℝ) := 3 * x^2 - 7 * x + 2

def decreasing_interval (y : ℝ) := y < 2 / 3

theorem interval_of_decrease :
  {x : ℝ | x < (1 / 3)} = {x : ℝ | x < (1 / 3)} :=
by sorry

end interval_of_decrease_l145_145570


namespace min_value_fraction_l145_145987

theorem min_value_fraction (a b : ℝ) (h1 : 2 * a + b = 3) (h2 : a > 0) (h3 : b > 0) (h4 : ∃ n : ℕ, b = n) : 
  (∃ a b : ℝ, 2 * a + b = 3 ∧ a > 0 ∧ b > 0 ∧ (∃ n : ℕ, b = n) ∧ ((1/(2*a) + 2/b) = 2)) := 
by
  sorry

end min_value_fraction_l145_145987


namespace number_of_white_balls_l145_145328

-- Definition of conditions
def red_balls : ℕ := 4
def frequency_of_red_balls : ℝ := 0.25
def total_balls (white_balls : ℕ) : ℕ := red_balls + white_balls

-- Proving the number of white balls given the conditions
theorem number_of_white_balls (x : ℕ) :
  (red_balls : ℝ) / total_balls x = frequency_of_red_balls → x = 12 :=
by
  sorry

end number_of_white_balls_l145_145328


namespace rectangle_area_l145_145718

theorem rectangle_area (x : ℝ) (h : (x - 3) * (2 * x + 3) = 4 * x - 9) : x = 7 / 2 :=
sorry

end rectangle_area_l145_145718


namespace sum_of_three_sqrt_139_l145_145898

theorem sum_of_three_sqrt_139 {x y z : ℝ} (h1 : x >= 0) (h2 : y >= 0) (h3 : z >= 0)
  (hx : x^2 + y^2 + z^2 = 75) (hy : x * y + y * z + z * x = 32) : x + y + z = Real.sqrt 139 := 
by
  sorry

end sum_of_three_sqrt_139_l145_145898


namespace sequence_sum_l145_145740

noncomputable def r : ℝ := 1 / 4
def t₀ : ℝ := 4096
def t₁ : ℝ := t₀ * r
def t₂ : ℝ := t₁ * r
def x : ℝ := t₂ * r
def y : ℝ := x * r

theorem sequence_sum : x + y = 80 := by
  -- The proof will go here
  sorry

end sequence_sum_l145_145740


namespace repeating_decimals_count_l145_145296

theorem repeating_decimals_count : 
  ∀ n : ℕ, 1 ≤ n ∧ n < 1000 → ¬(∃ k : ℕ, n + 1 = 2^k ∨ n + 1 = 5^k) :=
by
  sorry

end repeating_decimals_count_l145_145296


namespace point_outside_circle_l145_145788

theorem point_outside_circle (a b : ℝ)
  (h_line_intersects_circle : ∃ (x1 y1 x2 y2 : ℝ), 
     x1^2 + y1^2 = 1 ∧ 
     x2^2 + y2^2 = 1 ∧ 
     a * x1 + b * y1 = 1 ∧ 
     a * x2 + b * y2 = 1 ∧ 
     (x1, y1) ≠ (x2, y2)) : 
  a^2 + b^2 > 1 :=
sorry

end point_outside_circle_l145_145788


namespace radius_of_inscribed_circle_l145_145522

-- Define the triangle side lengths
def DE : ℝ := 8
def DF : ℝ := 8
def EF : ℝ := 10

-- Define the semiperimeter
def s : ℝ := (DE + DF + EF) / 2

-- Define the area using Heron's formula
def K : ℝ := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))

-- Define the radius of the inscribed circle
def r : ℝ := K / s

-- The statement to be proved
theorem radius_of_inscribed_circle : r = (5 * Real.sqrt 39) / 13 := by
  sorry

end radius_of_inscribed_circle_l145_145522


namespace broken_perfect_spiral_shells_difference_l145_145886

theorem broken_perfect_spiral_shells_difference :
  let perfect_shells := 17
  let broken_shells := 52
  let broken_spiral_shells := broken_shells / 2
  let not_spiral_perfect_shells := 12
  let spiral_perfect_shells := perfect_shells - not_spiral_perfect_shells
  broken_spiral_shells - spiral_perfect_shells = 21 := by
  sorry

end broken_perfect_spiral_shells_difference_l145_145886


namespace inv_matrix_eq_linear_comb_l145_145337

-- Define the matrix N
def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 1], ![0, -2]]

-- Define the constants a and b
def a := (1 : ℚ) / 6
def b := (1 : ℚ) / 6

-- prove that N⁻¹ = a * N + b * I
theorem inv_matrix_eq_linear_comb :
  N⁻¹ = (a : ℚ) • N + (b : ℚ) • (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
by
  -- proof to be provided
  sorry

end inv_matrix_eq_linear_comb_l145_145337


namespace boys_bought_balloons_l145_145405

def initial_balloons : ℕ := 3 * 12  -- Clown initially has 3 dozen balloons, i.e., 36 balloons
def girls_balloons : ℕ := 12        -- 12 girls buy a balloon each
def balloons_remaining : ℕ := 21     -- Clown is left with 21 balloons

def boys_balloons : ℕ :=
  initial_balloons - balloons_remaining - girls_balloons

theorem boys_bought_balloons :
  boys_balloons = 3 :=
by
  sorry

end boys_bought_balloons_l145_145405


namespace regular_polygon_sides_l145_145401

theorem regular_polygon_sides (n : ℕ) (h1 : ∃ a : ℝ, a = 120 ∧ ∀ i < n, 120 = a) : n = 6 :=
by
  sorry

end regular_polygon_sides_l145_145401


namespace shaded_cells_product_l145_145502

def product_eq (a b c : ℕ) (p : ℕ) : Prop := a * b * c = p

theorem shaded_cells_product :
  ∃ (a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃ a₃₁ a₃₂ a₃₃ : ℕ),
    product_eq a₁₁ a₁₂ a₁₃ 12 ∧
    product_eq a₂₁ a₂₂ a₂₃ 112 ∧
    product_eq a₃₁ a₃₂ a₃₃ 216 ∧
    product_eq a₁₁ a₂₁ a₃₁ 12 ∧
    product_eq a₁₂ a₂₂ a₃₂ 12 ∧
    (a₁₁ * a₂₂ * a₃₃ = 3 * 2 * 5) :=
sorry

end shaded_cells_product_l145_145502


namespace total_scoops_l145_145919

-- Define the conditions as variables
def flourCups := 3
def sugarCups := 2
def scoopSize := 1/3

-- Define what needs to be proved, i.e., the total amount of scoops needed
theorem total_scoops (flourCups sugarCups : ℚ) (scoopSize : ℚ) : 
  (flourCups / scoopSize) + (sugarCups / scoopSize) = 15 := 
by
  sorry

end total_scoops_l145_145919


namespace product_of_two_numbers_l145_145512

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x + y = 60) 
  (h2 : x - y = 16) : 
  x * y = 836 := 
by
  sorry

end product_of_two_numbers_l145_145512


namespace hyperbola_eccentricity_l145_145175

/-- Given a hyperbola with the equation x^2/a^2 - y^2/b^2 = 1, point B(0, b),
the line F1B intersects with the two asymptotes at points P and Q. 
We are given that vector QP = 4 * vector PF1. Prove that the eccentricity 
of the hyperbola is 3/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (F1 : ℝ × ℝ) (B : ℝ × ℝ) (P Q : ℝ × ℝ) 
  (h_F1 : F1 = (-c, 0)) (h_B : B = (0, b)) 
  (h_int_P : P = (-a * c / (c + a), b * c / (c + a)))
  (h_int_Q : Q = (a * c / (c - a), b * c / (c - a)))
  (h_vec : (Q.1 - P.1, Q.2 - P.2) = (4 * (P.1 - F1.1), 4 * (P.2 - F1.2))) :
  (eccentricity : ℝ) = 3 / 2 :=
sorry

end hyperbola_eccentricity_l145_145175


namespace work_earnings_t_l145_145492

theorem work_earnings_t (t : ℤ) (h1 : (t + 2) * (4 * t - 4) = (4 * t - 7) * (t + 3) + 3) : t = 10 := 
by
  sorry

end work_earnings_t_l145_145492


namespace no_two_digit_prime_sum_digits_nine_l145_145025

theorem no_two_digit_prime_sum_digits_nine :
  ¬ ∃ p : ℕ, prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 + p % 10 = 9) :=
sorry

end no_two_digit_prime_sum_digits_nine_l145_145025


namespace smallest_r_minus_p_l145_145683

theorem smallest_r_minus_p (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (prod_eq : p * q * r = nat.factorial 9) (h_lt : p < q ∧ q < r) :
  r - p = 219 := 
sorry

end smallest_r_minus_p_l145_145683


namespace domain_all_real_l145_145285

theorem domain_all_real (p : ℝ) : 
  (∀ x : ℝ, -3 * x ^ 2 + 3 * x + p ≠ 0) ↔ p < -3 / 4 := 
by
  sorry

end domain_all_real_l145_145285


namespace length_of_place_mat_l145_145544

noncomputable def radius : ℝ := 6
noncomputable def width : ℝ := 1.5
def inner_corner_touch (n : ℕ) : Prop := n = 6

theorem length_of_place_mat (y : ℝ) (h1 : radius = 6) (h2 : width = 1.5) (h3 : inner_corner_touch 6) :
  y = (Real.sqrt 141.75 + 1.5) / 2 :=
sorry

end length_of_place_mat_l145_145544


namespace smallest_number_is_21_5_l145_145851

-- Definitions of the numbers in their respective bases
def num1 := 3 * 4^0 + 3 * 4^1
def num2 := 0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3
def num3 := 2 * 3^0 + 2 * 3^1 + 1 * 3^2
def num4 := 1 * 5^0 + 2 * 5^1

-- Statement asserting that num4 is the smallest number
theorem smallest_number_is_21_5 : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 := by
  sorry

end smallest_number_is_21_5_l145_145851


namespace smallest_m_for_reflection_l145_145914

noncomputable def theta : Real := Real.arctan (1 / 3)
noncomputable def pi_8 : Real := Real.pi / 8
noncomputable def pi_12 : Real := Real.pi / 12
noncomputable def pi_4 : Real := Real.pi / 4
noncomputable def pi_6 : Real := Real.pi / 6

/-- The smallest positive integer m such that R^(m)(l) = l
where the transformation R(l) is described as:
l is reflected in l1 (angle pi/8), then the resulting line is
reflected in l2 (angle pi/12) -/
theorem smallest_m_for_reflection :
  ∃ (m : ℕ), m > 0 ∧ ∀ (k : ℤ), m = 12 * k + 12 := by
sorry

end smallest_m_for_reflection_l145_145914


namespace find_angle_A_find_area_l145_145775

noncomputable def triangle_area (a b c : ℝ) (A : ℝ) : ℝ :=
  0.5 * b * c * Real.sin A

theorem find_angle_A (a b c A : ℝ)
  (h1: ∀ x, 4 * Real.cos x * Real.sin (x - π/6) ≤ 4 * Real.cos A * Real.sin (A - π/6))
  (h2: a = b^2 + c^2 - 2 * b * c * Real.cos A) : 
  A = π / 3 := by
  sorry

theorem find_area (a b c : ℝ)
  (A : ℝ) (hA : A = π / 3)
  (ha : a = Real.sqrt 7) (hb : b = 2) 
  : triangle_area a b c A = (3 * Real.sqrt 3) / 2 := by
  sorry

end find_angle_A_find_area_l145_145775


namespace find_constants_l145_145642

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]

noncomputable def N_inv : Matrix (Fin 2) (Fin 2) ℚ := N⁻¹

theorem find_constants :
  ∃ c d : ℚ, N_inv = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

end find_constants_l145_145642


namespace max_value_l145_145589

theorem max_value (y : ℝ) (h : y ≠ 0) : 
  ∃ M, M = 1 / 25 ∧ 
       ∀ y ≠ 0,  ∀ value, value = y^2 / (y^4 + 4*y^3 + y^2 + 8*y + 16) 
       → value ≤ M :=
sorry

end max_value_l145_145589


namespace line_length_after_erasure_l145_145290

-- Defining the initial and erased lengths
def initial_length_cm : ℕ := 100
def erased_length_cm : ℕ := 33

-- The statement we need to prove
theorem line_length_after_erasure : initial_length_cm - erased_length_cm = 67 := by
  sorry

end line_length_after_erasure_l145_145290


namespace inequality_f_x_f_a_l145_145481

noncomputable def f (x : ℝ) : ℝ := x * x + x + 13

theorem inequality_f_x_f_a (a x : ℝ) (h : |x - a| < 1) : |f x * f a| < 2 * (|a| + 1) := 
sorry

end inequality_f_x_f_a_l145_145481


namespace sum_of_roots_l145_145528

theorem sum_of_roots (a b c : ℝ) (h_eq : a = 1) (h_b : b = -5) (h_c : c = 6) :
  (-b / a) = 5 := by
sorry

end sum_of_roots_l145_145528


namespace area_of_triangle_BXC_l145_145516

-- Define a trapezoid ABCD with given conditions
structure Trapezoid :=
  (A B C D X : Type)
  (AB CD : ℝ)
  (area_ABCD : ℝ)
  (intersect_at_X : Prop)

theorem area_of_triangle_BXC (t : Trapezoid) (h1 : t.AB = 24) (h2 : t.CD = 40)
  (h3 : t.area_ABCD = 480) (h4 : t.intersect_at_X) : 
  ∃ (area_BXC : ℝ), area_BXC = 120 :=
by {
  -- skip the proof here by using sorry
  sorry
}

end area_of_triangle_BXC_l145_145516


namespace minimum_value_expression_l145_145476

open Real

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2)) / (a * b * c) ≥ 216 :=
sorry

end minimum_value_expression_l145_145476


namespace find_g_8_l145_145273

def g (x : ℝ) : ℝ := x^2 + x + 1

theorem find_g_8 : (∀ x : ℝ, g (2*x - 4) = x^2 + x + 1) → g 8 = 43 := 
by sorry

end find_g_8_l145_145273


namespace percentage_calculation_l145_145127

theorem percentage_calculation :
  let total_amt := 1600
  let pct_25 := 0.25 * total_amt
  let pct_5 := 0.05 * pct_25
  pct_5 = 20 := by
sorry

end percentage_calculation_l145_145127


namespace geometric_progression_first_term_and_ratio_l145_145933

theorem geometric_progression_first_term_and_ratio (
  b_1 q : ℝ
) :
  b_1 * (1 + q + q^2) = 21 →
  b_1^2 * (1 + q^2 + q^4) = 189 →
  (b_1 = 12 ∧ q = 1/2) ∨ (b_1 = 3 ∧ q = 2) :=
by
  intros hsum hsumsq
  sorry

end geometric_progression_first_term_and_ratio_l145_145933


namespace two_digit_prime_sum_9_l145_145028

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- There are 0 two-digit prime numbers for which the sum of the digits equals 9 -/
theorem two_digit_prime_sum_9 : ∃! n : ℕ, (9 ≤ n ∧ n < 100) ∧ (n.digits 10).sum = 9 ∧ is_prime n :=
sorry

end two_digit_prime_sum_9_l145_145028


namespace abs_mult_example_l145_145002

theorem abs_mult_example : (|(-3)| * 2) = 6 := by
  have h1 : |(-3)| = 3 := by
    exact abs_of_neg (show -3 < 0 by norm_num)
  rw [h1]
  exact mul_eq_mul_left_iff.mpr (Or.inl rfl)

end abs_mult_example_l145_145002


namespace hens_egg_laying_l145_145207

theorem hens_egg_laying :
  ∀ (hens: ℕ) (price_per_dozen: ℝ) (total_revenue: ℝ) (weeks: ℕ) (total_hens: ℕ),
  hens = 10 →
  price_per_dozen = 3 →
  total_revenue = 120 →
  weeks = 4 →
  total_hens = hens →
  (total_revenue / price_per_dozen / 12) * 12 = 480 →
  (480 / weeks) = 120 →
  (120 / hens) = 12 :=
by sorry

end hens_egg_laying_l145_145207


namespace latus_rectum_of_parabola_l145_145020

theorem latus_rectum_of_parabola : 
  ∀ x y : ℝ, x^2 = -y → y = 1/4 :=
by
  -- Proof omitted
  sorry

end latus_rectum_of_parabola_l145_145020


namespace minimum_value_is_4_l145_145441

noncomputable def minimum_value (m n : ℝ) : ℝ :=
  if h : m > 0 ∧ n > 0 ∧ m + n = 1 then (1 / m) + (1 / n) else 0

theorem minimum_value_is_4 :
  (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m + n = 1) →
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m + n = 1 ∧ minimum_value m n = 4 :=
by
  sorry

end minimum_value_is_4_l145_145441


namespace minimum_value_expression_l145_145474

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a = 1 ∧ b = 1 ∧ c = 1) :
  (a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2) / (a * b * c) = 48 * Real.sqrt 6 := 
by
  sorry

end minimum_value_expression_l145_145474


namespace determine_n_l145_145417

-- Define the condition
def eq1 := (1 : ℚ) / (2 ^ 10) + (1 : ℚ) / (2 ^ 9) + (1 : ℚ) / (2 ^ 8)
def eq2 (n : ℚ) := n / (2 ^ 10)

-- The lean statement for the proof problem
theorem determine_n : ∃ (n : ℤ), eq1 = eq2 n ∧ n > 0 ∧ n = 7 := by
  sorry

end determine_n_l145_145417


namespace part_I_part_II_l145_145169

open Real

variable (a b : ℝ)

theorem part_I (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : (1 / a^2) + (1 / b^2) ≥ 8 := 
sorry

theorem part_II (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 := 
sorry

end part_I_part_II_l145_145169


namespace sum_of_coeffs_l145_145785

theorem sum_of_coeffs (a_5 a_4 a_3 a_2 a_1 a : ℤ) (h_eq : (x - 2)^5 = a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) (h_a : a = -32) :
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  sorry

end sum_of_coeffs_l145_145785


namespace number_of_fish_disappeared_l145_145346

-- First, define initial amounts of each type of fish
def goldfish_initial := 7
def catfish_initial := 12
def guppies_initial := 8
def angelfish_initial := 5

-- Define the total initial number of fish
def total_fish_initial := goldfish_initial + catfish_initial + guppies_initial + angelfish_initial

-- Define the current number of fish
def fish_current := 27

-- Define the number of fish disappeared
def fish_disappeared := total_fish_initial - fish_current

-- Proof statement
theorem number_of_fish_disappeared:
  fish_disappeared = 5 :=
by
  -- Sorry is a placeholder that indicates the proof is omitted.
  sorry

end number_of_fish_disappeared_l145_145346


namespace broken_more_than_perfect_spiral_l145_145885

theorem broken_more_than_perfect_spiral :
  let perfect_shells := 17
  let broken_shells := 52
  let broken_spiral_shells := broken_shells / 2
  let perfect_non_spiral_shells := 12
  let perfect_spiral_shells := perfect_shells - perfect_non_spiral_shells
  in broken_spiral_shells - perfect_spiral_shells = 21 :=
by
  sorry

end broken_more_than_perfect_spiral_l145_145885


namespace problem_part_1_problem_part_2_l145_145310

variable (θ : Real)
variable (m : Real)
variable (h_θ : θ ∈ Ioc 0 (2 * Real.pi))
variable (h_eq : ∀ x, 2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0 ↔ (x = Real.sin θ ∨ x = Real.cos θ))

theorem problem_part_1 : 
  (Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + (Real.cos θ)^2 / (Real.cos θ - Real.sin θ) = (Real.sqrt 3 + 1) / 2 := 
by
  sorry

theorem problem_part_2 : 
  m = Real.sqrt 3 / 2 := 
by 
  sorry

end problem_part_1_problem_part_2_l145_145310


namespace number_of_students_in_third_grade_l145_145270

theorem number_of_students_in_third_grade
    (total_students : ℕ)
    (sample_size : ℕ)
    (students_first_grade : ℕ)
    (students_second_grade : ℕ)
    (sample_first_and_second : ℕ)
    (students_in_third_grade : ℕ)
    (h1 : total_students = 2000)
    (h2 : sample_size = 100)
    (h3 : sample_first_and_second = students_first_grade + students_second_grade)
    (h4 : students_first_grade = 30)
    (h5 : students_second_grade = 30)
    (h6 : sample_first_and_second = 60)
    (h7 : sample_size - sample_first_and_second = students_in_third_grade)
    (h8 : students_in_third_grade * total_students = 40 * total_students / 100) :
  students_in_third_grade = 800 :=
sorry

end number_of_students_in_third_grade_l145_145270


namespace a_1995_eq_l145_145414

def a_3 : ℚ := (2 + 3) / (1 + 6)

def a (n : ℕ) : ℚ :=
  if n = 3 then a_3
  else if n ≥ 4 then
    let a_n_minus_1 := a (n - 1)
    (a_n_minus_1 + n) / (1 + n * a_n_minus_1)
  else
    0 -- We only care about n ≥ 3 in this problem

-- The problem itself
theorem a_1995_eq :
  a 1995 = 1991009 / 1991011 :=
by
  sorry

end a_1995_eq_l145_145414


namespace temperature_rise_l145_145681

variable (t : ℝ)

theorem temperature_rise (initial final : ℝ) (h : final = t) : final = 5 + t := by
  sorry

end temperature_rise_l145_145681


namespace find_p_max_area_triangle_l145_145611

-- Define the given conditions
def parabola (p : ℝ) := λ (x y : ℝ), x^2 = 2 * p * y
def circle (M : ℝ × ℝ) := λ (x y : ℝ), x^2 + (y + 4)^2 = 1

-- Focus and minimum distance condition
def focus (p : ℝ) := (0, p / 2)
def min_distance (F : ℝ × ℝ) (M : ℝ × ℝ) := dist F M = 4

-- The two main goals to prove
theorem find_p (p : ℝ) (x y : ℝ) :
  parabola p x y → circle (x, y) x y → min_distance (focus p) (x, y) → p = 2 :=
by
  sorry

theorem max_area_triangle (P A B : ℝ × ℝ) (k b : ℝ) :
  parabola 2 (A.1) (A.2) → parabola 2 (B.1) (B.2) →
  circle P.1 P.2 → P = (2 * k, -b) →
  (∃ y_P, y_P ∈ [-5, -3] ∧
    max_area k b = 20 * real.sqrt 5) :=
by
  sorry

end find_p_max_area_triangle_l145_145611


namespace production_line_B_units_l145_145953

theorem production_line_B_units {x y z : ℕ} (h1 : x + y + z = 24000) (h2 : 2 * y = x + z) : y = 8000 :=
sorry

end production_line_B_units_l145_145953


namespace trapezoid_inverse_sum_l145_145432

variables {A B C D O P : Point} 

theorem trapezoid_inverse_sum (h1: parallel A B C D) 
  (h2: intersection_of_diagonals A C B D O)
  (h3: ∃ P, on_line A D P ∧ angle A P B = angle C P D):
  1 / line_length A B + 1 / line_length C D = 1 / line_length O P := 
sorry

end trapezoid_inverse_sum_l145_145432


namespace solve_for_x_l145_145446

theorem solve_for_x (x : ℤ) (h : 3^(x - 2) = 9^3) : x = 8 :=
by
  sorry

end solve_for_x_l145_145446


namespace circles_intersect_on_AB_l145_145065

-- Define the structures and properties of the points and triangles involved
structure Point :=
(x : ℝ)
(y : ℝ)

def is_right_triangle (A B C : Point) : Prop :=
    C.x = A.x ∨ C.x = B.x ∧ C.y = A.y ∨ C.y = B.y

def midpoint (A B : Point) : Point :=
    {x := (A.x + B.x) / 2, y := (A.y + B.y) / 2}

def on_segment (G C : Point) : Prop := sorry -- Define G on the segment MC

def angle_eq (P A G : Point) (B C : Point) : Prop := sorry -- Define that corresponding angles are equal

noncomputable def proof_problem (A B C M G P Q : Point) : Prop :=
  is_right_triangle A B C ∧
  M = midpoint A B ∧
  on_segment G M ∧
  angle_eq P A G B C ∧
  angle_eq Q B G C A

theorem circles_intersect_on_AB
  (A B C M G P Q : Point) (h : proof_problem A B C M G P Q) :
  ∃ H : Point, sorry -- H is the intersection point on AB
    -- here, more conditions defining the circumscribed circles intersection
    -- this would typically involve circumcircle definitions and other geometric properties
    sorry :=
sorry -- Proof would go here

end circles_intersect_on_AB_l145_145065


namespace solve_for_a_l145_145890

open Complex

theorem solve_for_a (a : ℝ) (h : (2 + a * I) * (a - 2 * I) = -4 * I) : a = 0 :=
sorry

end solve_for_a_l145_145890


namespace parallel_lines_distance_l145_145991

theorem parallel_lines_distance (b c : ℝ) 
  (h1: b = 8) 
  (h2: (abs (10 - c) / (Real.sqrt (3^2 + 4^2))) = 3) :
  b + c = -12 ∨ b + c = 48 := by
 sorry

end parallel_lines_distance_l145_145991


namespace Joan_balloons_l145_145203

variable (J : ℕ) -- Joan's blue balloons

theorem Joan_balloons (h : J + 41 = 81) : J = 40 :=
by
  sorry

end Joan_balloons_l145_145203


namespace combined_angle_basic_astrophysics_nanotech_l145_145136

theorem combined_angle_basic_astrophysics_nanotech :
  let percentage_microphotonics : ℝ := 10
  let percentage_home_electronics : ℝ := 24
  let percentage_food_additives : ℝ := 15
  let percentage_gmo : ℝ := 29
  let percentage_industrial_lubricants : ℝ := 8
  let percentage_nanotechnology : ℝ := 7
  let total_percentage : ℝ := 100
  let percentage_basic_astrophysics := total_percentage - 
                                       (percentage_microphotonics + 
                                        percentage_home_electronics + 
                                        percentage_food_additives + 
                                        percentage_gmo + 
                                        percentage_industrial_lubricants + 
                                        percentage_nanotechnology)
  let combined_percentage := percentage_basic_astrophysics + 
                             percentage_nanotechnology
  let degrees_per_percentage : ℝ := 360 / total_percentage
  let combined_degrees := combined_percentage * degrees_per_percentage
  combined_degrees = 50.4 := by
begin
  sorry
end

end combined_angle_basic_astrophysics_nanotech_l145_145136


namespace product_of_three_numbers_l145_145680

theorem product_of_three_numbers 
  (a b c : ℕ) 
  (h1 : a + b + c = 300) 
  (h2 : 9 * a = b - 11) 
  (h3 : 9 * a = c + 15) : 
  a * b * c = 319760 := 
  sorry

end product_of_three_numbers_l145_145680


namespace students_remaining_after_fourth_stop_l145_145786

variable (n : ℕ)
variable (frac : ℚ)

def initial_students := (64 : ℚ)
def fraction_remaining := (2/3 : ℚ)

theorem students_remaining_after_fourth_stop : 
  let after_first_stop := initial_students * fraction_remaining
  let after_second_stop := after_first_stop * fraction_remaining
  let after_third_stop := after_second_stop * fraction_remaining
  let after_fourth_stop := after_third_stop * fraction_remaining
  after_fourth_stop = (1024 / 81) := 
by 
  sorry

end students_remaining_after_fourth_stop_l145_145786


namespace max_gcd_2015xy_l145_145584

theorem max_gcd_2015xy (x y : ℤ) (coprime : Int.gcd x y = 1) :
    ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
sorry

end max_gcd_2015xy_l145_145584


namespace good_games_count_l145_145225

-- Define the conditions
def games_from_friend : Nat := 50
def games_from_garage_sale : Nat := 27
def games_that_didnt_work : Nat := 74

-- Define the total games bought
def total_games_bought : Nat := games_from_friend + games_from_garage_sale

-- State the theorem to prove the number of good games
theorem good_games_count : total_games_bought - games_that_didnt_work = 3 :=
by
  sorry

end good_games_count_l145_145225


namespace find_starting_number_l145_145019

theorem find_starting_number (n : ℕ) (h : ((28 + n) / 2) = 18) : n = 8 :=
sorry

end find_starting_number_l145_145019


namespace candle_height_l145_145165

variable (h d a b x : ℝ)

theorem candle_height (h d a b : ℝ) : x = h * (1 + d / (a + b)) :=
by
  sorry

end candle_height_l145_145165


namespace correct_weights_swapped_l145_145513

theorem correct_weights_swapped 
  (W X Y Z : ℝ) 
  (h1 : Z > Y) 
  (h2 : X > W) 
  (h3 : Y + Z > W + X) :
  (W, Z) = (Z, W) :=
sorry

end correct_weights_swapped_l145_145513


namespace number_called_2009th_position_l145_145194

theorem number_called_2009th_position :
  let sequence := [1, 2, 3, 4, 3, 2]
  ∃ n, n = 2009 → sequence[(2009 % 6) - 1] = 3 := 
by
  -- let sequence := [1, 2, 3, 4, 3, 2]
  -- 2009 % 6 = 5
  -- sequence[4] = 3
  sorry

end number_called_2009th_position_l145_145194


namespace range_of_m_l145_145193

variable {x m : ℝ}

theorem range_of_m (h1 : x + 2 < 2 * m) (h2 : x - m < 0) (h3 : x < 2 * m - 2) : m ≤ 2 :=
sorry

end range_of_m_l145_145193


namespace money_spent_correct_l145_145906

-- Define the number of plays, acts per play, wigs per act, and the cost of each wig
def num_plays := 3
def acts_per_play := 5
def wigs_per_act := 2
def wig_cost := 5
def sell_price := 4

-- Given the total number of wigs he drops and sells from one play
def dropped_plays := 1
def total_wigs_dropped := dropped_plays * acts_per_play * wigs_per_act
def money_from_selling_dropped_wigs := total_wigs_dropped * sell_price

-- Calculate the initial cost
def total_wigs := num_plays * acts_per_play * wigs_per_act
def initial_cost := total_wigs * wig_cost

-- The final spent money should be calculated by subtracting money made from selling the wigs of the dropped play
def final_spent_money := initial_cost - money_from_selling_dropped_wigs

-- Specify the expected amount of money John spent
def expected_final_spent_money := 110

theorem money_spent_correct :
  final_spent_money = expected_final_spent_money := by
  sorry

end money_spent_correct_l145_145906


namespace jane_weekly_pages_l145_145636

-- Define the daily reading amounts
def monday_wednesday_morning_pages : ℕ := 5
def monday_wednesday_evening_pages : ℕ := 10
def tuesday_thursday_morning_pages : ℕ := 7
def tuesday_thursday_evening_pages : ℕ := 8
def friday_morning_pages : ℕ := 10
def friday_evening_pages : ℕ := 15
def weekend_morning_pages : ℕ := 12
def weekend_evening_pages : ℕ := 20

-- Define the number of days
def monday_wednesday_days : ℕ := 2
def tuesday_thursday_days : ℕ := 2
def friday_days : ℕ := 1
def weekend_days : ℕ := 2

-- Function to calculate weekly pages
def weekly_pages :=
  (monday_wednesday_days * (monday_wednesday_morning_pages + monday_wednesday_evening_pages)) +
  (tuesday_thursday_days * (tuesday_thursday_morning_pages + tuesday_thursday_evening_pages)) +
  (friday_days * (friday_morning_pages + friday_evening_pages)) +
  (weekend_days * (weekend_morning_pages + weekend_evening_pages))

-- Proof statement
theorem jane_weekly_pages : weekly_pages = 149 := by
  unfold weekly_pages
  norm_num
  sorry

end jane_weekly_pages_l145_145636


namespace no_quaint_two_digit_integers_l145_145963

theorem no_quaint_two_digit_integers :
  ∀ x : ℕ, 10 ≤ x ∧ x < 100 ∧ (∃ a b : ℕ, x = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) →  ¬(10 * x.div 10 + x % 10 = (x.div 10) + (x % 10)^3) :=
by
  sorry

end no_quaint_two_digit_integers_l145_145963


namespace functional_eq_solution_l145_145360

theorem functional_eq_solution (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2) :
  g 10 = -48 :=
sorry

end functional_eq_solution_l145_145360


namespace sugar_per_larger_cookie_l145_145907

theorem sugar_per_larger_cookie (c₁ c₂ : ℕ) (s₁ s₂ : ℝ) (h₁ : c₁ = 50) (h₂ : s₁ = 1 / 10) (h₃ : c₂ = 25) (h₄ : c₁ * s₁ = c₂ * s₂) : s₂ = 1 / 5 :=
by
  simp [h₁, h₂, h₃, h₄]
  sorry

end sugar_per_larger_cookie_l145_145907


namespace find_p_l145_145776

noncomputable def f (p : ℝ) : ℝ := 2 * p^2 + 20 * Real.sin p

theorem find_p : ∃ p : ℝ, f (f (f (f p))) = -4 :=
by
  sorry

end find_p_l145_145776


namespace octal_to_decimal_l145_145413

theorem octal_to_decimal (d0 d1 : ℕ) (n8 : ℕ) (n10 : ℕ) 
  (h1 : d0 = 3) (h2 : d1 = 5) (h3 : n8 = 53) (h4 : n10 = 43) : 
  (d1 * 8^1 + d0 * 8^0 = n10) :=
by
  sorry

end octal_to_decimal_l145_145413


namespace expression_evaluation_l145_145750

theorem expression_evaluation :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
sorry

end expression_evaluation_l145_145750


namespace geometric_progression_first_term_and_ratio_l145_145934

theorem geometric_progression_first_term_and_ratio (
  b_1 q : ℝ
) :
  b_1 * (1 + q + q^2) = 21 →
  b_1^2 * (1 + q^2 + q^4) = 189 →
  (b_1 = 12 ∧ q = 1/2) ∨ (b_1 = 3 ∧ q = 2) :=
by
  intros hsum hsumsq
  sorry

end geometric_progression_first_term_and_ratio_l145_145934


namespace find_cost_l145_145387

def cost_of_article (C : ℝ) (G : ℝ) : Prop :=
  (580 = C + G) ∧ (600 = C + G + 0.05 * G)

theorem find_cost (C : ℝ) (G : ℝ) (h : cost_of_article C G) : C = 180 :=
by
  sorry

end find_cost_l145_145387


namespace sum_of_roots_l145_145183

open Polynomial

noncomputable def f (a b : ℝ) : Polynomial ℝ := Polynomial.C b + Polynomial.C a * X + X^2
noncomputable def g (c d : ℝ) : Polynomial ℝ := Polynomial.C d + Polynomial.C c * X + X^2

theorem sum_of_roots (a b c d : ℝ)
  (h1 : eval 1 (f a b) = eval 2 (g c d))
  (h2 : eval 1 (g c d) = eval 2 (f a b))
  (hf_roots : ∃ r1 r2 : ℝ, (f a b).roots = {r1, r2})
  (hg_roots : ∃ s1 s2 : ℝ, (g c d).roots = {s1, s2}) :
  (-(a + c) = 6) :=
sorry

end sum_of_roots_l145_145183


namespace remainder_of_large_number_div_by_101_l145_145691

theorem remainder_of_large_number_div_by_101 :
  2468135792 % 101 = 52 :=
by
  sorry

end remainder_of_large_number_div_by_101_l145_145691


namespace percent_of_475_25_is_129_89_l145_145119

theorem percent_of_475_25_is_129_89 :
  (129.89 / 475.25) * 100 = 27.33 :=
by
  sorry

end percent_of_475_25_is_129_89_l145_145119


namespace packed_oranges_l145_145844

theorem packed_oranges (oranges_per_box : ℕ) (boxes_used : ℕ) (total_oranges : ℕ) 
  (h1 : oranges_per_box = 10) (h2 : boxes_used = 265) : 
  total_oranges = 2650 :=
by 
  sorry

end packed_oranges_l145_145844


namespace children_got_off_bus_l145_145392

theorem children_got_off_bus (initial : ℕ) (got_on : ℕ) (after : ℕ) : Prop :=
  initial = 22 ∧ got_on = 40 ∧ after = 2 → initial + got_on - 60 = after


end children_got_off_bus_l145_145392


namespace percentage_increase_in_area_l145_145536

-- Defining the lengths and widths in terms of real numbers
variables (L W : ℝ)

-- Defining the new lengths and widths
def new_length := 1.2 * L
def new_width := 1.2 * W

-- Original area of the rectangle
def original_area := L * W

-- New area of the rectangle
def new_area := new_length L * new_width W

-- Proof statement for the percentage increase
theorem percentage_increase_in_area : 
  ((new_area L W - original_area L W) / original_area L W) * 100 = 44 :=
by
  sorry

end percentage_increase_in_area_l145_145536


namespace divisibility_by_3_l145_145631

theorem divisibility_by_3 (x y z : ℤ) (h : x^3 + y^3 = z^3) : 3 ∣ x ∨ 3 ∣ y ∨ 3 ∣ z := 
sorry

end divisibility_by_3_l145_145631


namespace quadratic_y1_gt_y2_l145_145050

theorem quadratic_y1_gt_y2 (a b c y1 y2 : ℝ) (h_a_pos : a > 0) (h_sym : ∀ x, a * (x - 1)^2 + c = a * (1 - x)^2 + c) (h1 : y1 = a * (-1)^2 + b * (-1) + c) (h2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
sorry

end quadratic_y1_gt_y2_l145_145050


namespace remainder_2468135792_div_101_l145_145697

theorem remainder_2468135792_div_101 : (2468135792 % 101) = 52 := 
by 
  -- Conditions provided in the problem
  have decompose_num : 2468135792 = 24 * 10^8 + 68 * 10^6 + 13 * 10^4 + 57 * 10^2 + 92, 
  from sorry,
  
  -- Assert large powers of 10 modulo properties
  have ten_to_pow2 : (10^2 - 1) % 101 = 0, from sorry,
  have ten_to_pow4 : (10^4 - 1) % 101 = 0, from sorry,
  have ten_to_pow6 : (10^6 - 1) % 101 = 0, from sorry,
  have ten_to_pow8 : (10^8 - 1) % 101 = 0, from sorry,
  
  -- Summing coefficients
  have coefficients_sum : 24 + 68 + 13 + 57 + 92 = 254, from
  by linarith,
  
  -- Calculating modulus
  calc 
    2468135792 % 101
        = (24 * 10^8 + 68 * 10^6 + 13 * 10^4 + 57 * 10^2 + 92) % 101 : by rw decompose_num
    ... = (24 + 68 + 13 + 57 + 92) % 101 : by sorry
    ... = 254 % 101 : by rw coefficients_sum
    ... = 52 : by norm_num,

  sorry

end remainder_2468135792_div_101_l145_145697


namespace hyperbola_vertices_distance_l145_145577

theorem hyperbola_vertices_distance :
  ∀ (x y : ℝ), ((y^2 / 16) - (x^2 / 9) = 1) → 
    2 * real.sqrt 16 = 8 :=
by
  intro x y h
  have a2 : real.sqrt 16 = 4 := by norm_num
  have h2a : 2 * 4 = 8 := by norm_num
  rw [←a2, h2a]
  sorry

end hyperbola_vertices_distance_l145_145577


namespace two_digit_prime_sum_9_l145_145027

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- There are 0 two-digit prime numbers for which the sum of the digits equals 9 -/
theorem two_digit_prime_sum_9 : ∃! n : ℕ, (9 ≤ n ∧ n < 100) ∧ (n.digits 10).sum = 9 ∧ is_prime n :=
sorry

end two_digit_prime_sum_9_l145_145027


namespace least_possible_value_expression_l145_145158

theorem least_possible_value_expression :
  ∃ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2040 = 2039 :=
sorry

end least_possible_value_expression_l145_145158


namespace meeting_time_l145_145085

-- Define the conditions
def distance : ℕ := 600  -- distance between A and B
def speed_A_to_B : ℕ := 70  -- speed of the first person
def speed_B_to_A : ℕ := 80  -- speed of the second person
def start_time : ℕ := 10  -- start time in hours

-- State the problem formally in Lean 4
theorem meeting_time : (distance / (speed_A_to_B + speed_B_to_A)) + start_time = 14 := 
by
  sorry

end meeting_time_l145_145085


namespace inequality_inequality_only_if_k_is_one_half_l145_145162

theorem inequality_inequality_only_if_k_is_one_half :
  (∀ t : ℝ, -1 < t ∧ t < 1 → (1 + t) ^ k * (1 - t) ^ (1 - k) ≤ 1) ↔ k = 1 / 2 :=
by
  sorry

end inequality_inequality_only_if_k_is_one_half_l145_145162


namespace sequence_negation_l145_145170

theorem sequence_negation (x : ℕ → ℝ) (x1_pos : x 1 > 0) (x1_neq1 : x 1 ≠ 1)
  (rec_seq : ∀ n : ℕ, x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * x n ^ 2 + 1)) :
  ∃ n : ℕ, x n ≤ x (n + 1) :=
sorry

end sequence_negation_l145_145170


namespace odd_function_behavior_l145_145389

-- Define that f is odd
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x

-- Define f for x > 0
def f_pos (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → (f x = (Real.log x / Real.log 2) - 2 * x)

-- Prove that for x < 0, f(x) == -log₂(-x) - 2x
theorem odd_function_behavior (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_pos : f_pos f) :
  ∀ x, x < 0 → f x = -((Real.log (-x)) / (Real.log 2)) - 2 * x := 
by
  sorry -- proof goes here

end odd_function_behavior_l145_145389


namespace memorable_numbers_count_l145_145860

def is_memorable_number (d : Fin 10 → Fin 8 → ℕ) : Prop :=
  d 0 0 = d 1 0 ∧ d 0 1 = d 1 1 ∧ d 0 2 = d 1 2 ∧ d 0 3 = d 1 3

theorem memorable_numbers_count : 
  ∃ n : ℕ, n = 10000 ∧ ∀ (d : Fin 10 → Fin 8 → ℕ), is_memorable_number d → n = 10000 :=
sorry

end memorable_numbers_count_l145_145860


namespace direct_proportion_m_value_l145_145457

theorem direct_proportion_m_value (m : ℝ) : 
  (∀ x: ℝ, y = -7 * x + 2 + m -> y = k * x) -> m = -2 :=
by
  sorry

end direct_proportion_m_value_l145_145457


namespace fraction_value_sin_cos_value_l145_145305

open Real

-- Let alpha be an angle in radians satisfying the given condition
variable (α : ℝ)

-- Given condition
def condition  : Prop := sin α = 2 * cos α

-- First question
theorem fraction_value (h : condition α) : 
  (sin α - 4 * cos α) / (5 * sin α + 2 * cos α) = -1 / 6 :=
sorry

-- Second question
theorem sin_cos_value (h : condition α) : 
  sin α ^ 2 + 2 * sin α * cos α = 8 / 5 :=
sorry

end fraction_value_sin_cos_value_l145_145305


namespace strawberries_left_l145_145347

theorem strawberries_left (initial : ℝ) (eaten : ℝ) (remaining : ℝ) : initial = 78.0 → eaten = 42.0 → remaining = 36.0 → initial - eaten = remaining :=
by
  sorry

end strawberries_left_l145_145347


namespace price_of_shoes_on_tuesday_is_correct_l145_145649

theorem price_of_shoes_on_tuesday_is_correct :
  let price_thursday : ℝ := 30
  let price_friday : ℝ := price_thursday * 1.2
  let price_monday : ℝ := price_friday - price_friday * 0.15
  let price_tuesday : ℝ := price_monday - price_monday * 0.1
  price_tuesday = 27.54 := 
by
  sorry

end price_of_shoes_on_tuesday_is_correct_l145_145649


namespace gretchen_flavors_l145_145780

/-- 
Gretchen's local ice cream shop offers 100 different flavors. She tried a quarter of the flavors 2 years ago and double that amount last year. Prove how many more flavors she needs to try this year to have tried all 100 flavors.
-/
theorem gretchen_flavors (F T2 T1 T R : ℕ) (h1 : F = 100)
  (h2 : T2 = F / 4)
  (h3 : T1 = 2 * T2)
  (h4 : T = T2 + T1)
  (h5 : R = F - T) : R = 25 :=
sorry

end gretchen_flavors_l145_145780


namespace smallest_r_minus_p_l145_145684

theorem smallest_r_minus_p (p q r : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : r > 0)
  (h4 : p * q * r = Nat.factorial 9) (h5 : p < q) (h6 : q < r) :
  r - p = 396 :=
sorry

end smallest_r_minus_p_l145_145684


namespace gcd_max_possible_value_l145_145578

theorem gcd_max_possible_value (x y : ℤ) (h_coprime : Int.gcd x y = 1) : 
  ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
by
  sorry

end gcd_max_possible_value_l145_145578


namespace weight_gain_ratio_l145_145554

variable (J O F : ℝ)

theorem weight_gain_ratio :
  O = 5 ∧ F = (1/2) * J - 3 ∧ 5 + J + F = 20 → J / O = 12 / 5 :=
by
  intros h
  cases' h with hO h'
  cases' h' with hF hTotal
  sorry

end weight_gain_ratio_l145_145554


namespace find_x_in_plane_figure_l145_145118

theorem find_x_in_plane_figure (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 360) 
  (h3 : 2 * x + 160 = 360) : 
  x = 100 :=
by
  sorry

end find_x_in_plane_figure_l145_145118


namespace abc_eq_1_l145_145486

theorem abc_eq_1 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
(h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a)
(h7 : a + 1 / b^2 = b + 1 / c^2) (h8 : b + 1 / c^2 = c + 1 / a^2) :
  |a * b * c| = 1 :=
sorry

end abc_eq_1_l145_145486


namespace Foster_Farms_donated_45_chickens_l145_145758

def number_of_dressed_chickens_donated_by_Foster_Farms (C AS H BB D : ℕ) : Prop :=
  C + AS + H + BB + D = 375 ∧
  AS = 2 * C ∧
  H = 3 * C ∧
  BB = C ∧
  D = 2 * C - 30

theorem Foster_Farms_donated_45_chickens:
  ∃ C, number_of_dressed_chickens_donated_by_Foster_Farms C (2*C) (3*C) C (2*C - 30) ∧ C = 45 :=
by 
  sorry

end Foster_Farms_donated_45_chickens_l145_145758


namespace polynomial_evaluation_qin_jiushao_l145_145003

theorem polynomial_evaluation_qin_jiushao :
  let x := 3
  let V0 := 7
  let V1 := V0 * x + 6
  let V2 := V1 * x + 5
  let V3 := V2 * x + 4
  let V4 := V3 * x + 3
  V4 = 789 :=
by
  -- placeholder for proof
  sorry

end polynomial_evaluation_qin_jiushao_l145_145003


namespace forty_percent_of_number_l145_145945

theorem forty_percent_of_number (N : ℝ) 
  (h : (1/4) * (1/3) * (2/5) * N = 35) : 0.4 * N = 420 :=
by
  sorry

end forty_percent_of_number_l145_145945


namespace cricket_player_innings_l145_145398

theorem cricket_player_innings (n : ℕ) (h1 : 35 * n = 35 * n) (h2 : 35 * n + 79 = 39 * (n + 1)) : n = 10 := by
  sorry

end cricket_player_innings_l145_145398


namespace sum_of_first_n_terms_l145_145284

variable (a_n : ℕ → ℝ) -- Sequence term
variable (S_n : ℕ → ℝ) -- Sum of first n terms

-- Conditions given in the problem
axiom sum_first_term : a_n 1 = 2
axiom sum_first_two_terms : a_n 1 + a_n 2 = 7
axiom sum_first_three_terms : a_n 1 + a_n 2 + a_n 3 = 18

-- Expected result to prove
theorem sum_of_first_n_terms 
  (h1 : S_n 1 = 2)
  (h2 : S_n 2 = 7)
  (h3 : S_n 3 = 18) :
  S_n n = (3/2) * ((n * (n + 1) * (2 * n + 1) / 6) - (n * (n + 1) / 2) + 2 * n) :=
sorry

end sum_of_first_n_terms_l145_145284


namespace pencils_per_student_l145_145056

theorem pencils_per_student
  (boxes : ℝ) (pencils_per_box : ℝ) (students : ℝ)
  (h1 : boxes = 4.0)
  (h2 : pencils_per_box = 648.0)
  (h3 : students = 36.0) :
  (boxes * pencils_per_box) / students = 72.0 :=
by
  sorry

end pencils_per_student_l145_145056


namespace kids_on_soccer_field_l145_145108

def original_kids : ℕ := 14
def joined_kids : ℕ := 22
def total_kids : ℕ := 36

theorem kids_on_soccer_field : (original_kids + joined_kids) = total_kids :=
by 
  sorry

end kids_on_soccer_field_l145_145108


namespace probability_one_black_one_red_l145_145246

theorem probability_one_black_one_red (R B : Finset ℕ) (hR : R.card = 2) (hB : B.card = 3) :
  (2 : ℚ) / 5 = (6 + 6) / (5 * 4) := by
  sorry

end probability_one_black_one_red_l145_145246


namespace repetend_of_fraction_l145_145015

/-- The repeating sequence of the decimal representation of 5/17 is 294117 -/
theorem repetend_of_fraction : 
  let rep := list.take 6 (list.drop 1 (to_digits 10 (5 / 17) 8)) in
  rep = [2, 9, 4, 1, 1, 7] := 
by
  sorry

end repetend_of_fraction_l145_145015


namespace power_of_xy_l145_145593

-- Problem statement: Given a condition on x and y, find x^y.
theorem power_of_xy (x y : ℝ) (h : x^2 + y^2 + 4 * x - 6 * y + 13 = 0) : x^y = -8 :=
by {
  -- Proof will be added here
  sorry
}

end power_of_xy_l145_145593


namespace packed_oranges_l145_145845

theorem packed_oranges (oranges_per_box : ℕ) (boxes_used : ℕ) (total_oranges : ℕ) 
  (h1 : oranges_per_box = 10) (h2 : boxes_used = 265) : 
  total_oranges = 2650 :=
by 
  sorry

end packed_oranges_l145_145845


namespace novel_pages_l145_145702

theorem novel_pages (x : ℕ) (pages_per_day_in_reality : ℕ) (planned_days actual_days : ℕ)
  (h1 : planned_days = 20)
  (h2 : actual_days = 15)
  (h3 : pages_per_day_in_reality = x + 20)
  (h4 : pages_per_day_in_reality * actual_days = x * planned_days) :
  x * planned_days = 1200 :=
by
  sorry

end novel_pages_l145_145702


namespace find_decreased_value_l145_145958

theorem find_decreased_value (x v : ℝ) (hx : x = 7)
  (h : x - v = 21 * (1 / x)) : v = 4 :=
by
  sorry

end find_decreased_value_l145_145958


namespace percent_Asian_in_West_l145_145854

noncomputable def NE_Asian := 2
noncomputable def MW_Asian := 2
noncomputable def South_Asian := 2
noncomputable def West_Asian := 6

noncomputable def total_Asian := NE_Asian + MW_Asian + South_Asian + West_Asian

theorem percent_Asian_in_West (h1 : total_Asian = 12) : (West_Asian / total_Asian) * 100 = 50 := 
by sorry

end percent_Asian_in_West_l145_145854


namespace ones_digit_8_power_32_l145_145521

theorem ones_digit_8_power_32 : (8^32) % 10 = 6 :=
by sorry

end ones_digit_8_power_32_l145_145521


namespace total_number_of_students_l145_145123

theorem total_number_of_students 
    (T : ℕ)
    (h1 : ∃ a, a = T / 5) 
    (h2 : ∃ b, b = T / 4) 
    (h3 : ∃ c, c = T / 2) 
    (h4 : T - (T / 5 + T / 4 + T / 2) = 25) : 
  T = 500 := by 
  sorry

end total_number_of_students_l145_145123


namespace problem_statement_l145_145601

noncomputable def f (a x : ℝ) := a * (x ^ 2 + 1) + Real.log x

theorem problem_statement (a m : ℝ) (x : ℝ) 
  (h_a : -4 < a) (h_a' : a < -2) (h_x1 : 1 ≤ x) (h_x2 : x ≤ 3) :
  (m * a - f a x > a ^ 2) ↔ (m ≤ -2) :=
by
  sorry

end problem_statement_l145_145601


namespace find_number_of_each_coin_l145_145635

-- Define the number of coins
variables (n d q : ℕ)

-- Given conditions
axiom twice_as_many_nickels_as_quarters : n = 2 * q
axiom same_number_of_dimes_as_quarters : d = q
axiom total_value_of_coins : 5 * n + 10 * d + 25 * q = 1520

-- Statement to prove
theorem find_number_of_each_coin :
  q = 304 / 9 ∧
  n = 2 * (304 / 9) ∧
  d = 304 / 9 :=
sorry

end find_number_of_each_coin_l145_145635


namespace distance_between_trees_l145_145850

theorem distance_between_trees
  (yard_length : ℕ)
  (num_trees : ℕ)
  (h_yard_length : yard_length = 441)
  (h_num_trees : num_trees = 22) :
  (yard_length / (num_trees - 1)) = 21 :=
by
  sorry

end distance_between_trees_l145_145850


namespace units_digit_47_4_plus_28_4_l145_145255

theorem units_digit_47_4_plus_28_4 (units_digit_47 : Nat := 7) (units_digit_28 : Nat := 8) :
  (47^4 + 28^4) % 10 = 7 :=
by
  sorry

end units_digit_47_4_plus_28_4_l145_145255


namespace problem_solution_l145_145110

noncomputable def x : ℝ := 3 / 0.15
noncomputable def y : ℝ := 3 / 0.25
noncomputable def z : ℝ := 0.30 * y

theorem problem_solution : x - y + z = 11.6 := sorry

end problem_solution_l145_145110


namespace apple_juice_fraction_correct_l145_145687

def problem_statement : Prop :=
  let pitcher1_capacity := 800
  let pitcher2_capacity := 500
  let pitcher1_apple_fraction := 1 / 4
  let pitcher2_apple_fraction := 1 / 5
  let pitcher1_apple_volume := pitcher1_capacity * pitcher1_apple_fraction
  let pitcher2_apple_volume := pitcher2_capacity * pitcher2_apple_fraction
  let total_apple_volume := pitcher1_apple_volume + pitcher2_apple_volume
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_apple_volume / total_volume = 3 / 13

theorem apple_juice_fraction_correct : problem_statement := 
  sorry

end apple_juice_fraction_correct_l145_145687


namespace x_y_differ_by_one_l145_145215

theorem x_y_differ_by_one (x y : ℚ) (h : (1 + y) / (x - y) = x) : y = x - 1 :=
by
sorry

end x_y_differ_by_one_l145_145215


namespace min_value_inequality_l145_145911

noncomputable def min_value (x y z w : ℝ) : ℝ :=
  x^2 + 4 * x * y + 9 * y^2 + 6 * y * z + 8 * z^2 + 3 * x * w + 4 * w^2

theorem min_value_inequality 
  (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_prod : x * y * z * w = 3) : 
  min_value x y z w ≥ 81.25 := 
sorry

end min_value_inequality_l145_145911


namespace smallest_num_conditions_l145_145548

theorem smallest_num_conditions :
  ∃ n : ℕ, (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 4 = 3) ∧ n = 11 :=
by
  sorry

end smallest_num_conditions_l145_145548


namespace two_digit_prime_sum_9_l145_145029

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- There are 0 two-digit prime numbers for which the sum of the digits equals 9 -/
theorem two_digit_prime_sum_9 : ∃! n : ℕ, (9 ≤ n ∧ n < 100) ∧ (n.digits 10).sum = 9 ∧ is_prime n :=
sorry

end two_digit_prime_sum_9_l145_145029


namespace tony_age_in_6_years_l145_145633

theorem tony_age_in_6_years (jacob_age : ℕ) (tony_age : ℕ) (h : jacob_age = 24) (h_half : tony_age = jacob_age / 2) : (tony_age + 6) = 18 :=
by
  sorry

end tony_age_in_6_years_l145_145633


namespace find_interest_rate_l145_145519

noncomputable def interest_rate (total_investment remaining_investment interest_earned part_interest : ℝ) : ℝ :=
  (interest_earned - part_interest) / remaining_investment

theorem find_interest_rate :
  let total_investment := 9000
  let invested_at_8_percent := 4000
  let total_interest := 770
  let interest_at_8_percent := invested_at_8_percent * 0.08
  let remaining_investment := total_investment - invested_at_8_percent
  let interest_from_remaining := total_interest - interest_at_8_percent
  interest_rate total_investment remaining_investment total_interest interest_at_8_percent = 0.09 :=
by
  sorry

end find_interest_rate_l145_145519


namespace yanna_kept_apples_l145_145381

theorem yanna_kept_apples (total_apples : ℕ) (apples_to_Zenny : ℕ) (apples_to_Andrea : ℕ) 
  (h_total : total_apples = 60) (h_Zenny : apples_to_Zenny = 18) (h_Andrea : apples_to_Andrea = 6) : 
  (total_apples - apples_to_Zenny - apples_to_Andrea) = 36 := by
  -- Initial setup based on the problem conditions
  rw [h_total, h_Zenny, h_Andrea]
  -- Simplify the expression
  rfl

-- The theorem simplifies to proving 60 - 18 - 6 = 36

end yanna_kept_apples_l145_145381


namespace gcd_of_polynomial_l145_145990

theorem gcd_of_polynomial (a : ℤ) (h : 720 ∣ a) : Int.gcd (a^2 + 8*a + 18) (a + 6) = 6 := 
by 
  sorry

end gcd_of_polynomial_l145_145990


namespace Gretchen_walking_time_l145_145883

theorem Gretchen_walking_time
  (hours_worked : ℕ)
  (minutes_per_hour : ℕ)
  (sit_per_walk : ℕ)
  (walk_per_brake : ℕ)
  (h1 : hours_worked = 6)
  (h2 : minutes_per_hour = 60)
  (h3 : sit_per_walk = 90)
  (h4 : walk_per_brake = 10) :
  let total_minutes := hours_worked * minutes_per_hour,
      breaks := total_minutes / sit_per_walk,
      walking_time := breaks * walk_per_brake in
  walking_time = 40 := 
by
  sorry

end Gretchen_walking_time_l145_145883


namespace matrix_to_system_solution_l145_145599

theorem matrix_to_system_solution :
  ∀ (x y : ℝ),
  (2 * x + y = 5) ∧ (x - 2 * y = 0) →
  3 * x - y = 5 :=
by
  sorry

end matrix_to_system_solution_l145_145599


namespace average_headcount_spring_terms_l145_145117

def spring_headcount_02_03 := 10900
def spring_headcount_03_04 := 10500
def spring_headcount_04_05 := 10700

theorem average_headcount_spring_terms :
  (spring_headcount_02_03 + spring_headcount_03_04 + spring_headcount_04_05) / 3 = 10700 := by
  sorry

end average_headcount_spring_terms_l145_145117


namespace polynomial_expansion_sum_l145_145188

theorem polynomial_expansion_sum :
  ∀ P Q R S : ℕ, ∀ x : ℕ, 
  (P = 4 ∧ Q = 10 ∧ R = 1 ∧ S = 21) → 
  ((x + 3) * (4 * x ^ 2 - 2 * x + 7) = P * x ^ 3 + Q * x ^ 2 + R * x + S) → 
  P + Q + R + S = 36 :=
by
  intros P Q R S x h1 h2
  sorry

end polynomial_expansion_sum_l145_145188


namespace only_B_is_linear_system_l145_145533

def linear_equation (eq : String) : Prop := 
-- Placeholder for the actual definition
sorry 

def system_B_is_linear : Prop :=
  linear_equation "x + y = 2" ∧ linear_equation "x - y = 4"

theorem only_B_is_linear_system 
: (∀ (A B C D : Prop), 
       (A ↔ (linear_equation "3x + 4y = 6" ∧ linear_equation "5z - 6y = 4")) → 
       (B ↔ (linear_equation "x + y = 2" ∧ linear_equation "x - y = 4")) → 
       (C ↔ (linear_equation "x + y = 2" ∧ linear_equation "x^2 - y^2 = 8")) → 
       (D ↔ (linear_equation "x + y = 2" ∧ linear_equation "1/x - 1/y = 1/2")) → 
       (B ∧ ¬A ∧ ¬C ∧ ¬D))
:= 
sorry

end only_B_is_linear_system_l145_145533


namespace fenced_area_l145_145093

theorem fenced_area (w : ℕ) (h : ℕ) (cut_out : ℕ) (rectangle_area : ℕ) (cut_out_area : ℕ) (net_area : ℕ) :
  w = 20 → h = 18 → cut_out = 4 → rectangle_area = w * h → cut_out_area = cut_out * cut_out → net_area = rectangle_area - cut_out_area → net_area = 344 :=
by
  intros
  subst_vars
  sorry

end fenced_area_l145_145093


namespace total_treat_value_is_339100_l145_145862

def hotel_cost (cost_per_night : ℕ) (nights : ℕ) (discount : ℕ) : ℕ :=
  let total_cost := cost_per_night * nights
  total_cost - (total_cost * discount / 100)

def car_cost (base_price : ℕ) (tax : ℕ) : ℕ :=
  base_price + (base_price * tax / 100)

def house_cost (car_base_price : ℕ) (multiplier : ℕ) (property_tax : ℕ) : ℕ :=
  let house_value := car_base_price * multiplier
  house_value + (house_value * property_tax / 100)

def yacht_cost (hotel_value : ℕ) (car_value : ℕ) (multiplier : ℕ) (discount : ℕ) : ℕ :=
  let combined_value := hotel_value + car_value
  let yacht_value := combined_value * multiplier
  yacht_value - (yacht_value * discount / 100)

def gold_coins_cost (yacht_value : ℕ) (multiplier : ℕ) (tax : ℕ) : ℕ :=
  let gold_value := yacht_value * multiplier
  gold_value + (gold_value * tax / 100)

theorem total_treat_value_is_339100 :
  let hotel_value := hotel_cost 4000 2 5
  let car_value := car_cost 30000 10
  let house_value := house_cost 30000 4 2
  let yacht_value := yacht_cost 8000 30000 2 7
  let gold_coins_value := gold_coins_cost 76000 3 3
  hotel_value + car_value + house_value + yacht_value + gold_coins_value = 339100 :=
by sorry

end total_treat_value_is_339100_l145_145862


namespace gcd_max_possible_value_l145_145579

theorem gcd_max_possible_value (x y : ℤ) (h_coprime : Int.gcd x y = 1) : 
  ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
by
  sorry

end gcd_max_possible_value_l145_145579


namespace expected_value_of_painted_faces_of_random_small_cube_l145_145716

noncomputable def cubeExpectedValue : ℚ :=
  let P : ℕ → ℚ := λ n, match n with
    | 0 => 64 / 125
    | 1 => 48 / 125
    | 2 => 12 / 125
    | 3 => 1 / 125
    | _ => 0
  in (0 : ℚ) * P 0 + (1 : ℚ) * P 1 + (2 : ℚ) * P 2 + (3 : ℚ) * P 3

theorem expected_value_of_painted_faces_of_random_small_cube :
  cubeExpectedValue = 3 / 5 := by
sorry

end expected_value_of_painted_faces_of_random_small_cube_l145_145716


namespace paul_initial_savings_l145_145348

theorem paul_initial_savings (additional_allowance: ℕ) (cost_per_toy: ℕ) (number_of_toys: ℕ) (total_savings: ℕ) :
  additional_allowance = 7 →
  cost_per_toy = 5 →
  number_of_toys = 2 →
  total_savings + additional_allowance = cost_per_toy * number_of_toys →
  total_savings = 3 :=
by
  intros h_additional h_cost h_number h_total
  sorry

end paul_initial_savings_l145_145348


namespace find_x_when_y_neg_10_l145_145923

def inversely_proportional (x y : ℝ) (k : ℝ) := x * y = k

theorem find_x_when_y_neg_10 (k : ℝ) (h₁ : inversely_proportional 4 (-2) k) (yval : y = -10) 
: ∃ x, inversely_proportional x y k ∧ x = 4 / 5 := by
  sorry

end find_x_when_y_neg_10_l145_145923


namespace fenced_area_l145_145092

theorem fenced_area (w : ℕ) (h : ℕ) (cut_out : ℕ) (rectangle_area : ℕ) (cut_out_area : ℕ) (net_area : ℕ) :
  w = 20 → h = 18 → cut_out = 4 → rectangle_area = w * h → cut_out_area = cut_out * cut_out → net_area = rectangle_area - cut_out_area → net_area = 344 :=
by
  intros
  subst_vars
  sorry

end fenced_area_l145_145092


namespace count_triangles_in_hexagonal_grid_l145_145887

-- Define the number of smallest triangles in the figure.
def small_triangles : ℕ := 10

-- Define the number of medium triangles in the figure, composed of 4 small triangles each.
def medium_triangles : ℕ := 6

-- Define the number of large triangles in the figure, composed of 9 small triangles each.
def large_triangles : ℕ := 3

-- Define the number of extra-large triangle composed of 16 small triangles.
def extra_large_triangle : ℕ := 1

-- Define the total number of triangles in the figure.
def total_triangles : ℕ := small_triangles + medium_triangles + large_triangles + extra_large_triangle

-- The theorem we want to prove: the total number of triangles is 20.
theorem count_triangles_in_hexagonal_grid : total_triangles = 20 := by
  -- Placeholder for the proof.
  sorry

end count_triangles_in_hexagonal_grid_l145_145887


namespace modulus_of_z_l145_145301

open Complex

theorem modulus_of_z (z : ℂ) (h : (1 - I) * z = 2 + 2 * I) : abs z = 2 := 
sorry

end modulus_of_z_l145_145301


namespace second_share_interest_rate_is_11_l145_145552

noncomputable def calculate_interest_rate 
    (total_investment : ℝ)
    (amount_in_second_share : ℝ)
    (interest_rate_first : ℝ)
    (total_interest : ℝ) : ℝ := 
  let A := total_investment - amount_in_second_share
  let interest_first := (interest_rate_first / 100) * A
  let interest_second := total_interest - interest_first
  (100 * interest_second) / amount_in_second_share

theorem second_share_interest_rate_is_11 :
  calculate_interest_rate 100000 12499.999999999998 9 9250 = 11 := 
by
  sorry

end second_share_interest_rate_is_11_l145_145552


namespace product_of_roots_l145_145564

theorem product_of_roots (r1 r2 r3 : ℝ) : 
  (∀ x : ℝ, 2 * x^3 - 24 * x^2 + 96 * x + 56 = 0 → x = r1 ∨ x = r2 ∨ x = r3) →
  r1 * r2 * r3 = -28 :=
by
  sorry

end product_of_roots_l145_145564


namespace quadratic_roots_identity_l145_145238

theorem quadratic_roots_identity :
  ∀ (x1 x2 : ℝ), (x1^2 - 3 * x1 - 4 = 0) ∧ (x2^2 - 3 * x2 - 4 = 0) →
  (x1^2 - 2 * x1 * x2 + x2^2) = 25 :=
by
  intros x1 x2 h
  sorry

end quadratic_roots_identity_l145_145238


namespace remaining_oranges_l145_145372

/-- Define the conditions of the problem. -/
def oranges_needed_Michaela : ℕ := 20
def oranges_needed_Cassandra : ℕ := 2 * oranges_needed_Michaela
def total_oranges_picked : ℕ := 90

/-- State the proof problem. -/
theorem remaining_oranges : total_oranges_picked - (oranges_needed_Michaela + oranges_needed_Cassandra) = 30 := 
sorry

end remaining_oranges_l145_145372


namespace remainder_2468135792_div_101_l145_145696

theorem remainder_2468135792_div_101 : (2468135792 % 101) = 52 := 
by 
  -- Conditions provided in the problem
  have decompose_num : 2468135792 = 24 * 10^8 + 68 * 10^6 + 13 * 10^4 + 57 * 10^2 + 92, 
  from sorry,
  
  -- Assert large powers of 10 modulo properties
  have ten_to_pow2 : (10^2 - 1) % 101 = 0, from sorry,
  have ten_to_pow4 : (10^4 - 1) % 101 = 0, from sorry,
  have ten_to_pow6 : (10^6 - 1) % 101 = 0, from sorry,
  have ten_to_pow8 : (10^8 - 1) % 101 = 0, from sorry,
  
  -- Summing coefficients
  have coefficients_sum : 24 + 68 + 13 + 57 + 92 = 254, from
  by linarith,
  
  -- Calculating modulus
  calc 
    2468135792 % 101
        = (24 * 10^8 + 68 * 10^6 + 13 * 10^4 + 57 * 10^2 + 92) % 101 : by rw decompose_num
    ... = (24 + 68 + 13 + 57 + 92) % 101 : by sorry
    ... = 254 % 101 : by rw coefficients_sum
    ... = 52 : by norm_num,

  sorry

end remainder_2468135792_div_101_l145_145696


namespace problem_inequality_l145_145289

variable {a b c : ℝ}

theorem problem_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) :=
by sorry

end problem_inequality_l145_145289


namespace no_two_digit_prime_with_digit_sum_9_l145_145030

-- Define the concept of a two-digit number
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Define the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the concept of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem statement
theorem no_two_digit_prime_with_digit_sum_9 :
  ∀ n : ℕ, is_two_digit n ∧ digit_sum n = 9 → ¬is_prime n :=
by {
  -- proof omitted
  sorry
}  

end no_two_digit_prime_with_digit_sum_9_l145_145030


namespace solution_set_l145_145312

def f (x m : ℝ) : ℝ := |x - 1| - |x - m|

theorem solution_set (x : ℝ) :
  (f x 2) ≥ 1 ↔ x ≥ 2 :=
sorry

end solution_set_l145_145312


namespace baseball_card_problem_l145_145120

theorem baseball_card_problem:
  let initial_cards := 15
  let maria_takes := (initial_cards + 1) / 2
  let cards_after_maria := initial_cards - maria_takes
  let cards_after_peter := cards_after_maria - 1
  let final_cards := cards_after_peter * 3
  final_cards = 18 :=
by
  sorry

end baseball_card_problem_l145_145120


namespace range_of_m_l145_145322

theorem range_of_m (m : ℝ) : 
  ((0 - m)^2 + (0 + m)^2 < 4) → -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
by
  sorry

end range_of_m_l145_145322


namespace monotonic_increasing_interval_l145_145505

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 0 < x → (1 / 2 < x → (f (x + 0.1) > f x)) :=
by
  intro x hx h
  sorry

end monotonic_increasing_interval_l145_145505


namespace quadratic_y1_gt_y2_l145_145051

theorem quadratic_y1_gt_y2 (a b c y1 y2 : ℝ) (h_a_pos : a > 0) (h_sym : ∀ x, a * (x - 1)^2 + c = a * (1 - x)^2 + c) (h1 : y1 = a * (-1)^2 + b * (-1) + c) (h2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
sorry

end quadratic_y1_gt_y2_l145_145051


namespace maximize_cubic_quartic_l145_145069

theorem maximize_cubic_quartic (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + 2 * y = 35) : 
  (x, y) = (21, 7) ↔ x^3 * y^4 = (21:ℝ)^3 * (7:ℝ)^4 := 
by
  sorry

end maximize_cubic_quartic_l145_145069


namespace find_y_l145_145630

/-- 
  Given: The sum of angles around a point is 360 degrees, 
  and those angles are: 6y, 3y, 4y, and 2y.
  Prove: y = 24 
-/ 
theorem find_y (y : ℕ) (h : 6 * y + 3 * y + 4 * y + 2 * y = 360) : y = 24 :=
sorry

end find_y_l145_145630


namespace west_of_1km_l145_145501

def east_direction (d : Int) : Int :=
  d

def west_direction (d : Int) : Int :=
  -d

theorem west_of_1km :
  east_direction (2) = 2 →
  west_direction (1) = -1 := by
  sorry

end west_of_1km_l145_145501


namespace percentage_increase_from_March_to_January_l145_145152

variable {F J M : ℝ}

def JanuaryCondition (F J : ℝ) : Prop :=
  J = 0.90 * F

def MarchCondition (F M : ℝ) : Prop :=
  M = 0.75 * F

theorem percentage_increase_from_March_to_January (F J M : ℝ) (h1 : JanuaryCondition F J) (h2 : MarchCondition F M) :
  (J / M) = 1.20 := by 
  sorry

end percentage_increase_from_March_to_January_l145_145152


namespace extremum_point_is_three_l145_145191

noncomputable def f (x : ℝ) : ℝ := (x - 2) / Real.exp x

theorem extremum_point_is_three {x₀ : ℝ} (h : ∀ x, f x₀ ≤ f x) : x₀ = 3 :=
by
  -- proof goes here
  sorry

end extremum_point_is_three_l145_145191


namespace incorrect_eqn_x9_y9_neg1_l145_145892

theorem incorrect_eqn_x9_y9_neg1 (x y : ℂ) 
  (hx : x = (-1 + Complex.I * Real.sqrt 3) / 2) 
  (hy : y = (-1 - Complex.I * Real.sqrt 3) / 2) : 
  x^9 + y^9 ≠ -1 :=
sorry

end incorrect_eqn_x9_y9_neg1_l145_145892


namespace passing_marks_l145_145122

-- Define the conditions and prove P = 160 given these conditions
theorem passing_marks (T P : ℝ) (h1 : 0.40 * T = P - 40) (h2 : 0.60 * T = P + 20) : P = 160 :=
by
  sorry

end passing_marks_l145_145122


namespace determine_true_propositions_l145_145174

def p (x y : ℝ) := x > y → -x < -y
def q (x y : ℝ) := (1/x > 1/y) → x < y

theorem determine_true_propositions (x y : ℝ) :
  (p x y ∨ q x y) ∧ (p x y ∧ ¬ q x y) :=
by
  sorry

end determine_true_propositions_l145_145174


namespace binomial_coefficient_8_5_l145_145412

theorem binomial_coefficient_8_5 : Nat.choose 8 5 = 56 := by
  sorry

end binomial_coefficient_8_5_l145_145412


namespace increasing_function_range_l145_145043

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : ∀ x y : ℝ, x < y → f a x ≤ f a y) : 
  1.5 ≤ a ∧ a < 2 :=
sorry

end increasing_function_range_l145_145043


namespace remainder_2468135792_mod_101_l145_145694

theorem remainder_2468135792_mod_101 : 
  2468135792 % 101 = 47 := 
sorry

end remainder_2468135792_mod_101_l145_145694


namespace hotel_rolls_l145_145954

theorem hotel_rolls (m n : ℕ) (rel_prime : Nat.gcd m n = 1) : 
  let num_nut_rolls := 3
  let num_cheese_rolls := 3
  let num_fruit_rolls := 3
  let total_rolls := 9
  let num_guests := 3
  let rolls_per_guest := 3
  let probability_first_guest := (3 / 9) * (3 / 8) * (3 / 7)
  let probability_second_guest := (2 / 6) * (2 / 5) * (2 / 4)
  let probability_third_guest := 1
  let overall_probability := probability_first_guest * probability_second_guest * probability_third_guest
  overall_probability = (9 / 70) → m = 9 ∧ n = 70 → m + n = 79 :=
by
  intros
  sorry

end hotel_rolls_l145_145954


namespace remainder_poly_div_l145_145286

theorem remainder_poly_div 
    (x : ℤ) 
    (h1 : (x^2 + x + 1) ∣ (x^3 - 1)) 
    (h2 : x^5 - 1 = (x^3 - 1) * (x^2 + x + 1) - x * (x^2 + x + 1) + 1) : 
  ((x^5 - 1) * (x^3 - 1)) % (x^2 + x + 1) = 0 :=
by
  sorry

end remainder_poly_div_l145_145286


namespace find_x_for_vectors_l145_145998

theorem find_x_for_vectors
  (x : ℝ)
  (h1 : x ∈ Set.Icc 0 Real.pi)
  (a : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2)))
  (b : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2)))
  (h2 : (a.1 + b.1)^2 + (a.2 + b.2)^2 = 1) :
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 :=
by
  sorry

end find_x_for_vectors_l145_145998


namespace probability_neither_snow_nor_rain_in_5_days_l145_145827

def probability_no_snow (p_snow : ℚ) : ℚ := 1 - p_snow
def probability_no_rain (p_rain : ℚ) : ℚ := 1 - p_rain
def probability_no_snow_and_no_rain (p_no_snow p_no_rain : ℚ) : ℚ := p_no_snow * p_no_rain
def probability_no_snow_and_no_rain_5_days (p : ℚ) : ℚ := p ^ 5

theorem probability_neither_snow_nor_rain_in_5_days
    (p_snow : ℚ) (p_rain : ℚ)
    (h1 : p_snow = 2/3) (h2 : p_rain = 1/2) :
    probability_no_snow_and_no_rain_5_days (probability_no_snow_and_no_rain (probability_no_snow p_snow) (probability_no_rain p_rain)) = 1/7776 := by
  sorry

end probability_neither_snow_nor_rain_in_5_days_l145_145827


namespace contrapositive_example_l145_145359

theorem contrapositive_example (a b : ℕ) (h : a = 0 → ab = 0) : ab ≠ 0 → a ≠ 0 :=
by sorry

end contrapositive_example_l145_145359


namespace tangent_line_at_origin_tangent_line_passing_through_neg1_neg3_l145_145602

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x

noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 2

theorem tangent_line_at_origin :
  (∀ x y : ℝ, y = f x → x = 0 → y = 2 * x) := by
  sorry

theorem tangent_line_passing_through_neg1_neg3 :
  (∀ x y : ℝ, y = f x → (x, y) ≠ (-1, -3) → y = 5 * x + 2) := by
  sorry

end tangent_line_at_origin_tangent_line_passing_through_neg1_neg3_l145_145602


namespace Yoojung_total_vehicles_l145_145384

theorem Yoojung_total_vehicles : 
  let motorcycles := 2
  let bicycles := 5
  motorcycles + bicycles = 7 := 
by
  sorry

end Yoojung_total_vehicles_l145_145384


namespace emily_and_berengere_contribution_l145_145406

noncomputable def euro_to_usd : ℝ := 1.20
noncomputable def euro_to_gbp : ℝ := 0.85

noncomputable def cake_cost_euros : ℝ := 12
noncomputable def cookies_cost_euros : ℝ := 5
noncomputable def total_cost_euros : ℝ := cake_cost_euros + cookies_cost_euros

noncomputable def emily_usd : ℝ := 10
noncomputable def liam_gbp : ℝ := 10

noncomputable def emily_euros : ℝ := emily_usd / euro_to_usd
noncomputable def liam_euros : ℝ := liam_gbp / euro_to_gbp

noncomputable def total_available_euros : ℝ := emily_euros + liam_euros

theorem emily_and_berengere_contribution : total_available_euros >= total_cost_euros := by
  sorry

end emily_and_berengere_contribution_l145_145406


namespace relationship_of_y_values_l145_145303

noncomputable def quadratic_function (x : ℝ) (c : ℝ) := x^2 - 6*x + c

theorem relationship_of_y_values (c : ℝ) (y1 y2 y3 : ℝ) :
  quadratic_function 1 c = y1 →
  quadratic_function (2 * Real.sqrt 2) c = y2 →
  quadratic_function 4 c = y3 →
  y3 < y2 ∧ y2 < y1 :=
by
  intros hA hB hC
  sorry

end relationship_of_y_values_l145_145303


namespace ellipse_standard_equation_l145_145302

theorem ellipse_standard_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (-4)^2 / a^2 + 3^2 / b^2 = 1) 
    (h4 : a^2 = b^2 + 5^2) : 
    ∃ (a b : ℝ), a^2 = 40 ∧ b^2 = 15 ∧ 
    (∀ x y : ℝ, x^2 / 40 + y^2 / 15 = 1 → (∃ f1 f2 : ℝ, f1 = 5 ∧ f2 = -5)) :=
by {
    sorry
}

end ellipse_standard_equation_l145_145302


namespace mean_temperature_l145_145826

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem mean_temperature (temps : List ℝ) (length_temps_10 : temps.length = 10)
    (temps_vals : temps = [78, 80, 82, 85, 88, 90, 92, 95, 97, 95]) : 
    mean temps = 88.2 := by
  sorry

end mean_temperature_l145_145826


namespace seq_a6_l145_145778

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 3 * a n - 2

theorem seq_a6 (a : ℕ → ℕ) (h : seq a) : a 6 = 1 :=
by
  sorry

end seq_a6_l145_145778


namespace problem_two_probability_l145_145514

-- Definitions based on conditions
def physics_problems : List ℕ := [1, 2, 3, 4, 5]
def chemistry_problems : List ℕ := [6, 7, 8, 9]
def all_problems : List ℕ := physics_problems ++ chemistry_problems

def all_combinations : List (ℕ × ℕ) :=
  (all_problems.product all_problems).filter (λ (p : ℕ × ℕ), p.fst < p.snd)

def valid_combinations := all_combinations.filter (λ (p : ℕ × ℕ), 11 ≤ p.fst + p.snd ∧ p.fst + p.snd < 17)

-- Lean 4 statement
theorem problem_two_probability :
  (all_combinations.length = 36) ∧
  (valid_combinations.length = 15) →
  (valid_combinations.length.to_real / all_combinations.length.to_real = 5/12) := by
  sorry

end problem_two_probability_l145_145514


namespace find_digits_l145_145395

variable (M N : ℕ)
def x := 10 * N + M
def y := 10 * M + N

theorem find_digits (h₁ : x > y) (h₂ : x + y = 11 * (x - y)) : M = 4 ∧ N = 5 :=
sorry

end find_digits_l145_145395


namespace laran_weekly_profit_l145_145208

-- Definitions based on the problem conditions
def daily_posters_sold : ℕ := 5
def large_posters_sold_daily : ℕ := 2
def small_posters_sold_daily : ℕ := daily_posters_sold - large_posters_sold_daily

def price_large_poster : ℕ := 10
def cost_large_poster : ℕ := 5
def profit_large_poster : ℕ := price_large_poster - cost_large_poster

def price_small_poster : ℕ := 6
def cost_small_poster : ℕ := 3
def profit_small_poster : ℕ := price_small_poster - cost_small_poster

def daily_profit_large_posters : ℕ := large_posters_sold_daily * profit_large_poster
def daily_profit_small_posters : ℕ := small_posters_sold_daily * profit_small_poster
def total_daily_profit : ℕ := daily_profit_large_posters + daily_profit_small_posters

def school_days_week : ℕ := 5
def weekly_profit : ℕ := total_daily_profit * school_days_week

-- Statement to prove
theorem laran_weekly_profit : weekly_profit = 95 := sorry

end laran_weekly_profit_l145_145208


namespace simplify_and_evaluate_expression_l145_145231

theorem simplify_and_evaluate_expression (a b : ℤ) (h_a : a = 2) (h_b : b = -1) : 
  2 * (-a^2 + 2 * a * b) - 3 * (a * b - a^2) = 2 :=
by 
  sorry

end simplify_and_evaluate_expression_l145_145231


namespace max_marks_test_l145_145274

theorem max_marks_test (M : ℝ) : 
  (0.30 * M = 80 + 100) -> 
  M = 600 :=
by 
  sorry

end max_marks_test_l145_145274


namespace monthly_compounding_greater_than_yearly_l145_145708

open Nat Real

theorem monthly_compounding_greater_than_yearly : 
  1 + 3 / 100 < (1 + 3 / (12 * 100)) ^ 12 :=
by
  -- This is the proof we need to write.
  sorry

end monthly_compounding_greater_than_yearly_l145_145708


namespace dream_star_games_l145_145509

theorem dream_star_games (x y : ℕ) 
  (h1 : x + y + 2 = 9)
  (h2 : 3 * x + y = 17) : 
  x = 5 ∧ y = 2 := 
by 
  sorry

end dream_star_games_l145_145509


namespace max_gcd_coprime_l145_145582

theorem max_gcd_coprime (x y : ℤ) (h : Int.gcd x y = 1) : 
  Int.gcd (x + 2015 * y) (y + 2015 * x) ≤ 4060224 :=
sorry

end max_gcd_coprime_l145_145582


namespace inscribed_circle_radius_eq_l145_145527

noncomputable def radius_of_inscribed_circle (DE DF EF : ℝ) (h1 : DE = 8) (h2 : DF = 8) (h3 : EF = 10) : ℝ :=
  let s := (DE + DF + EF) / 2 in
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  area / s

theorem inscribed_circle_radius_eq :
  radius_of_inscribed_circle 8 8 10 8 8 10 = (15 * Real.sqrt 13) / 13 :=
  sorry

end inscribed_circle_radius_eq_l145_145527


namespace car_speed_second_hour_l145_145244

variable (x : ℝ)
variable (s1 : ℝ := 100)
variable (avg_speed : ℝ := 90)
variable (total_time : ℝ := 2)

-- The Lean statement equivalent to the problem
theorem car_speed_second_hour : (100 + x) / 2 = 90 → x = 80 := by 
  intro h
  have h₁ : 2 * 90 = 100 + x := by 
    linarith [h]
  linarith [h₁]

end car_speed_second_hour_l145_145244


namespace R_depends_on_d_and_n_l145_145910

-- Define the given properties of the arithmetic progression sums
def s1 (a d n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2
def s3 (a d n : ℕ) : ℕ := (3 * n * (2 * a + (3 * n - 1) * d)) / 2
def s5 (a d n : ℕ) : ℕ := (5 * n * (2 * a + (5 * n - 1) * d)) / 2

-- Define R in terms of s1, s3, and s5
def R (a d n : ℕ) : ℕ := s5 a d n - s3 a d n - s1 a d n

-- The main theorem to prove the statement about R's dependency
theorem R_depends_on_d_and_n (a d n : ℕ) : R a d n = 7 * d * n^2 := by 
  sorry

end R_depends_on_d_and_n_l145_145910


namespace binomial_expansion_terms_l145_145822

theorem binomial_expansion_terms (x n : ℝ) (hn : n = 8) : 
  ∃ t, t = 3 :=
  sorry

end binomial_expansion_terms_l145_145822


namespace find_freshmen_count_l145_145931

theorem find_freshmen_count
  (F S J R : ℕ)
  (h1 : F : S = 5 : 4)
  (h2 : S : J = 7 : 8)
  (h3 : J : R = 9 : 7)
  (total_students : F + S + J + R = 2158) :
  F = 630 :=
by 
  sorry

end find_freshmen_count_l145_145931


namespace perimeter_of_rectangle_l145_145959

theorem perimeter_of_rectangle (b l : ℝ) (h1 : l = 3 * b) (h2 : b * l = 75) : 2 * l + 2 * b = 40 := 
by 
  sorry

end perimeter_of_rectangle_l145_145959


namespace value_of_af_over_cd_l145_145454

variable (a b c d e f : ℝ)

theorem value_of_af_over_cd :
  a * b * c = 130 ∧
  b * c * d = 65 ∧
  c * d * e = 500 ∧
  d * e * f = 250 →
  (a * f) / (c * d) = 1 :=
by
  sorry

end value_of_af_over_cd_l145_145454


namespace least_number_with_remainder_l145_145520

variable (x : ℕ)

theorem least_number_with_remainder (x : ℕ) : 
  (x % 16 = 11) ∧ (x % 27 = 11) ∧ (x % 34 = 11) ∧ (x % 45 = 11) ∧ (x % 144 = 11) → x = 36731 := by
  sorry

end least_number_with_remainder_l145_145520


namespace scarves_sold_at_new_price_l145_145327

theorem scarves_sold_at_new_price :
  ∃ (p : ℕ), (∃ (c k : ℕ), (k = p * c) ∧ (p = 30) ∧ (c = 10)) ∧
  (∃ (new_c : ℕ), new_c = 165 / 10 ∧ k = new_p * new_c) ∧
  new_p = 18
:=
sorry

end scarves_sold_at_new_price_l145_145327


namespace satellite_orbit_time_approx_l145_145725

noncomputable def earth_radius_km : ℝ := 6371
noncomputable def satellite_speed_kmph : ℝ := 7000

theorem satellite_orbit_time_approx :
  let circumference := 2 * Real.pi * earth_radius_km 
  let time := circumference / satellite_speed_kmph 
  5.6 < time ∧ time < 5.8 :=
by
  sorry

end satellite_orbit_time_approx_l145_145725


namespace pascal_triangle_row51_sum_l145_145254

theorem pascal_triangle_row51_sum : (Nat.choose 51 4) + (Nat.choose 51 6) = 18249360 :=
by
  sorry

end pascal_triangle_row51_sum_l145_145254


namespace incorrect_equation_l145_145894

noncomputable def x : ℂ := (-1 + Real.sqrt 3 * Complex.I) / 2
noncomputable def y : ℂ := (-1 - Real.sqrt 3 * Complex.I) / 2

theorem incorrect_equation : x^9 + y^9 ≠ -1 := sorry

end incorrect_equation_l145_145894


namespace measure_of_angle_C_l145_145969

theorem measure_of_angle_C (A B : ℝ) (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := by
  sorry

end measure_of_angle_C_l145_145969


namespace compare_powers_l145_145763

theorem compare_powers (a b c : ℝ) (h1 : a = 2^555) (h2 : b = 3^444) (h3 : c = 6^222) : a < c ∧ c < b :=
by
  sorry

end compare_powers_l145_145763


namespace probability_of_winning_pair_l145_145831

-- Conditions: Define the deck composition and the winning pair.
inductive Color
| Red
| Green
| Blue

inductive Label
| A
| B
| C

structure Card :=
(color : Color)
(label : Label)

def deck : List Card :=
  [ {color := Color.Red, label := Label.A},
    {color := Color.Red, label := Label.B},
    {color := Color.Red, label := Label.C},
    {color := Color.Green, label := Label.A},
    {color := Color.Green, label := Label.B},
    {color := Color.Green, label := Label.C},
    {color := Color.Blue, label := Label.A},
    {color := Color.Blue, label := Label.B},
    {color := Color.Blue, label := Label.C} ]

def is_winning_pair (c1 c2 : Card) : Prop :=
  c1.color = c2.color ∨ c1.label = c2.label

-- Question: Prove the probability of drawing a winning pair.
theorem probability_of_winning_pair :
  (∃ (c1 c2 : Card), c1 ∈ deck ∧ c2 ∈ deck ∧ c1 ≠ c2 ∧ is_winning_pair c1 c2) →
  (∃ (c1 c2 : Card), c1 ∈ deck ∧ c2 ∈ deck ∧ c1 ≠ c2) →
  (9 + 9) / 36 = 1 / 2 :=
sorry

end probability_of_winning_pair_l145_145831


namespace sum_of_roots_of_quadratic_eqn_l145_145572

theorem sum_of_roots_of_quadratic_eqn (A B : ℝ) 
  (h₁ : 3 * A ^ 2 - 9 * A + 6 = 0)
  (h₂ : 3 * B ^ 2 - 9 * B + 6 = 0)
  (h_distinct : A ≠ B):
  A + B = 3 := by
  sorry

end sum_of_roots_of_quadratic_eqn_l145_145572


namespace probability_xi_leq_one_l145_145268

noncomputable def C (n k : ℕ) : ℕ := Nat.descFactorial n k / Nat.factorial k

theorem probability_xi_leq_one :
  let total_people := 12
  let excellent_students := 5
  let selected_people := 5
  let C_7_5 := C 7 5
  let C_5_1_C_7_4 := C 5 1 * C 7 4
  let C_12_5 := C total_people selected_people
  ∃ (xi : ℕ → ℕ), xi <= excellent_students →
    (C_7_5 + C_5_1_C_7_4) / C_12_5 = ∑ k in Finset.range 2, xi k :=
begin
  intro total_people,
  intro excellent_students,
  intro selected_people,
  intro C_7_5,
  intro C_5_1_C_7_4,
  intro C_12_5,
  use λ k, if k = 0 then 1 else if k = 1 then 1 else 0,
  intro hxi,
  sorry
end

end probability_xi_leq_one_l145_145268


namespace no_two_digit_prime_sum_digits_nine_l145_145024

theorem no_two_digit_prime_sum_digits_nine :
  ¬ ∃ p : ℕ, prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 + p % 10 = 9) :=
sorry

end no_two_digit_prime_sum_digits_nine_l145_145024


namespace second_car_distance_l145_145374

theorem second_car_distance (x : ℝ) : 
  let d_initial : ℝ := 150
  let d_first_car_initial : ℝ := 25
  let d_right_turn : ℝ := 15
  let d_left_turn : ℝ := 25
  let d_final_gap : ℝ := 65
  (d_initial - x = d_final_gap) → x = 85 := by
  sorry

end second_car_distance_l145_145374


namespace solve_for_y_l145_145416

theorem solve_for_y (y : ℝ) (h : y + 49 / y = 14) : y = 7 :=
sorry

end solve_for_y_l145_145416


namespace time_in_3467_hours_l145_145084

-- Define the current time, the number of hours, and the modulus
def current_time : ℕ := 2
def hours_from_now : ℕ := 3467
def clock_modulus : ℕ := 12

-- Define the function to calculate the future time on a 12-hour clock
def future_time (current_time : ℕ) (hours_from_now : ℕ) (modulus : ℕ) : ℕ := 
  (current_time + hours_from_now) % modulus

-- Theorem statement
theorem time_in_3467_hours :
  future_time current_time hours_from_now clock_modulus = 9 :=
by
  -- Proof would go here
  sorry

end time_in_3467_hours_l145_145084


namespace vector_eq_to_slope_intercept_form_l145_145847

theorem vector_eq_to_slope_intercept_form :
  ∀ (x y : ℝ), (2 * (x - 4) + 5 * (y - 1)) = 0 → y = -(2 / 5) * x + 13 / 5 := 
by 
  intros x y h
  sorry

end vector_eq_to_slope_intercept_form_l145_145847


namespace no_two_digit_prime_with_digit_sum_9_l145_145031

-- Define the concept of a two-digit number
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Define the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the concept of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem statement
theorem no_two_digit_prime_with_digit_sum_9 :
  ∀ n : ℕ, is_two_digit n ∧ digit_sum n = 9 → ¬is_prime n :=
by {
  -- proof omitted
  sorry
}  

end no_two_digit_prime_with_digit_sum_9_l145_145031


namespace find_n_l145_145271

theorem find_n (x : ℝ) (h1 : x = 4.0) (h2 : 3 * x + n = 48) : n = 36 := by
  sorry

end find_n_l145_145271


namespace hyperbola_eccentricity_l145_145673

-- Define the conditions given in the problem
def asymptote_equation_related (a b : ℝ) : Prop := a / b = 3 / 4
def hyperbola_eccentricity_relation (a c : ℝ) : Prop := c^2 / a^2 = 25 / 9

-- Define the proof problem
theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : asymptote_equation_related a b)
  (h2 : hyperbola_eccentricity_relation a c)
  (he : e = c / a) :
  e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l145_145673


namespace intersection_A_B_l145_145070

def setA : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem intersection_A_B :
  (setA ∩ setB = {x | 0 ≤ x ∧ x ≤ 2}) :=
by
  sorry

end intersection_A_B_l145_145070


namespace red_or_blue_probability_is_half_l145_145835

-- Define the number of each type of marble
def num_red_marbles : ℕ := 3
def num_blue_marbles : ℕ := 2
def num_yellow_marbles : ℕ := 5

-- Define the total number of marbles
def total_marbles : ℕ := num_red_marbles + num_blue_marbles + num_yellow_marbles

-- Define the number of marbles that are either red or blue
def num_red_or_blue_marbles : ℕ := num_red_marbles + num_blue_marbles

-- Define the probability of drawing a red or blue marble
def probability_red_or_blue : ℚ := num_red_or_blue_marbles / total_marbles

-- Theorem stating the probability is 0.5
theorem red_or_blue_probability_is_half : probability_red_or_blue = 0.5 := by
  sorry

end red_or_blue_probability_is_half_l145_145835


namespace mrs_hilt_rocks_proof_l145_145071

def num_rocks_already_placed : ℝ := 125.0
def total_num_rocks_planned : ℝ := 189
def num_more_rocks_needed : ℝ := 64

theorem mrs_hilt_rocks_proof : total_num_rocks_planned - num_rocks_already_placed = num_more_rocks_needed :=
by
  sorry

end mrs_hilt_rocks_proof_l145_145071


namespace sum_of_consecutive_page_numbers_l145_145678

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20412) : n + (n + 1) = 283 := 
sorry

end sum_of_consecutive_page_numbers_l145_145678


namespace geometric_series_sum_l145_145730

theorem geometric_series_sum :
  let a := 1
  let r := (1 / 4 : ℚ)
  (a / (1 - r)) = 4 / 3 :=
by
  sorry

end geometric_series_sum_l145_145730


namespace solve_equation_l145_145667

theorem solve_equation (x : ℂ) :
  (x^3 + 3 * x^2 * complex.sqrt 3 + 9 * x + 3 * complex.sqrt 3) + (x + complex.sqrt 3) = 0 ↔ 
    (x = -complex.sqrt 3 ∨ x = -complex.sqrt 3 + complex.I ∨ x = -complex.sqrt 3 - complex.I) :=
by sorry

end solve_equation_l145_145667


namespace remainder_of_2n_div_7_l145_145895

theorem remainder_of_2n_div_7 (n : ℤ) (k : ℤ) (h : n = 7 * k + 2) : (2 * n) % 7 = 4 :=
by
  sorry

end remainder_of_2n_div_7_l145_145895


namespace max_gcd_2015xy_l145_145586

theorem max_gcd_2015xy (x y : ℤ) (coprime : Int.gcd x y = 1) :
    ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
sorry

end max_gcd_2015xy_l145_145586


namespace correct_phone_call_sequence_l145_145442

-- Define the six steps as an enumerated type.
inductive Step
| Dial
| WaitDialTone
| PickUpHandset
| StartConversationOrHangUp
| WaitSignal
| EndCall

open Step

-- Define the problem as a theorem.
theorem correct_phone_call_sequence : 
  ∃ sequence : List Step, sequence = [PickUpHandset, WaitDialTone, Dial, WaitSignal, StartConversationOrHangUp, EndCall] :=
sorry

end correct_phone_call_sequence_l145_145442


namespace common_difference_of_common_terms_l145_145245

def sequence_a (n : ℕ) : ℕ := 4 * n - 3
def sequence_b (k : ℕ) : ℕ := 3 * k - 1

theorem common_difference_of_common_terms :
  ∃ (d : ℕ), (∀ (m : ℕ), 12 * m + 5 ∈ { x | ∃ (n k : ℕ), sequence_a n = x ∧ sequence_b k = x }) ∧ d = 12 := 
sorry

end common_difference_of_common_terms_l145_145245


namespace find_salary_l145_145704

variable (S : ℝ)
variable (house_rent_percentage : ℝ) (education_percentage : ℝ) (clothes_percentage : ℝ)
variable (remaining_amount : ℝ)

theorem find_salary (h1 : house_rent_percentage = 0.20)
                    (h2 : education_percentage = 0.10)
                    (h3 : clothes_percentage = 0.10)
                    (h4 : remaining_amount = 1377)
                    (h5 : (1 - clothes_percentage) * (1 - education_percentage) * (1 - house_rent_percentage) * S = remaining_amount) :
                    S = 2125 := 
sorry

end find_salary_l145_145704


namespace minimum_rounds_l145_145037

-- Given conditions based on the problem statement
variable (m : ℕ) (hm : m ≥ 17)
variable (players : Fin (2 * m)) -- Representing 2m players
variable (rounds : Fin (2 * m - 1)) -- Representing 2m - 1 rounds
variable (pairs : Fin m → Fin (2 * m) × Fin (2 * m)) -- Pairing for each of the m pairs in each round

-- Statement of the proof problem
theorem minimum_rounds (h1 : ∀ i j, i ≠ j → ∃! (k : Fin m), pairs k = (i, j) ∨ pairs k = (j, i))
(h2 : ∀ k : Fin m, (pairs k).fst ≠ (pairs k).snd)
(h3 : ∀ i j, i ≠ j → ∃ r : Fin (2 * m - 1), (∃ k : Fin m, pairs k = (i, j)) ∧ (∃ k : Fin m, pairs k = (j, i))) :
∃ (n : ℕ), n = m - 1 ∧ ∀ s : Fin 4 → Fin (2 * m), (∀ i j, i ≠ j → ¬ ∃ r : Fin n, ∃ k : Fin m, pairs k = (s i, s j)) ∨ (∃ r1 r2 : Fin n, ∃ i j, i ≠ j ∧ ∃ k1 k2 : Fin m, pairs k1 = (s i, s j) ∧ pairs k2 = (s j, s i)) :=
sorry

end minimum_rounds_l145_145037


namespace min_a2_b2_l145_145736

theorem min_a2_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) :
  a^2 + b^2 ≥ 4 / 5 :=
sorry

end min_a2_b2_l145_145736


namespace max_area_triangle_PAB_l145_145606

-- Define given parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the circle M
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus F of the parabola
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

-- Define the minimum distance from F to any point on the circle
def min_distance_from_focus_to_circle (p : ℝ) : Prop := 
  abs ((p / 2) + 4 - 1) = 4

-- Define point P on the circle
def point_on_circle (x y : ℝ) : Prop := circle x y

-- Define tangents PA and PB from point P to the parabola with points of tangency A and B
def tangents (x1 y1 x2 y2 p : ℝ) : Prop :=
  parabola p x1 y1 ∧ parabola p x2 y2

-- Define area of triangle PAB
def area_triangle_PAB (x1 y1 x2 y2 : ℝ) : ℝ := 
  let k := (x1 + x2) / 2
  let b := -y1
  4 * ((k^2 + b) ^ (3 / 2))

-- Prove that for p = 2, the maximum area of the triangle PAB is 20 sqrt 5
theorem max_area_triangle_PAB : ∃ (x1 y1 x2 y2 : ℝ), 
  parabola 2 x1 y1 ∧ parabola 2 x2 y2 ∧ point_on_circle 2 (-(x2 / 2)^2 / 4) ∧
  area_triangle_PAB x1 y1 x2 y2 = 20 * real.sqrt 5 :=
sorry

end max_area_triangle_PAB_l145_145606


namespace lashawn_three_times_kymbrea_l145_145334

-- Definitions based on the conditions
def kymbrea_collection (months : ℕ) : ℕ := 50 + 3 * months
def lashawn_collection (months : ℕ) : ℕ := 20 + 5 * months

-- Theorem stating the core of the problem
theorem lashawn_three_times_kymbrea (x : ℕ) 
  (h : lashawn_collection x = 3 * kymbrea_collection x) : x = 33 := 
sorry

end lashawn_three_times_kymbrea_l145_145334


namespace min_value_l145_145480

theorem min_value : ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (a = 1) → (b = 1) → (c = 1) →
  (∃ x, x = (a^2 + 4 * a + 2) / a ∧ x ≥ 6) ∧
  (∃ y, y = (b^2 + 4 * b + 2) / b ∧ y ≥ 6) ∧
  (∃ z, z = (c^2 + 4 * c + 2) / c ∧ z ≥ 6) →
  (∃ m, m = ((a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2)) / (a * b * c) ∧ m = 216) :=
by {
  sorry
}

end min_value_l145_145480


namespace how_many_peaches_l145_145201

-- Define the main problem statement and conditions.
theorem how_many_peaches (A P J_A J_P : ℕ) (h_person_apples: A = 16) (h_person_peaches: P = A + 1) (h_jake_apples: J_A = A + 8) (h_jake_peaches: J_P = P - 6) : P = 17 :=
by
  -- Since the proof is not required, we use sorry to skip it.
  sorry

end how_many_peaches_l145_145201


namespace jeans_cost_l145_145793

-- Definitions based on conditions
def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def total_cost : ℕ := 51
def n_shirts : ℕ := 3
def n_hats : ℕ := 4
def n_jeans : ℕ := 2

-- The goal is to prove that the cost of one pair of jeans (J) is 10
theorem jeans_cost (J : ℕ) (h : n_shirts * shirt_cost + n_jeans * J + n_hats * hat_cost = total_cost) : J = 10 :=
  sorry

end jeans_cost_l145_145793


namespace sin_cos_relationship_l145_145318

theorem sin_cos_relationship (α : ℝ) (h1 : Real.pi / 2 < α) (h2 : α < Real.pi) : 
  Real.sin α - Real.cos α > 1 :=
sorry

end sin_cos_relationship_l145_145318


namespace total_preparation_and_cooking_time_l145_145556

def time_to_chop_pepper : Nat := 3
def time_to_chop_onion : Nat := 4
def time_to_grate_cheese_per_omelet : Nat := 1
def time_to_cook_omelet : Nat := 5
def num_peppers : Nat := 4
def num_onions : Nat := 2
def num_omelets : Nat := 5

theorem total_preparation_and_cooking_time :
  num_peppers * time_to_chop_pepper +
  num_onions * time_to_chop_onion +
  num_omelets * (time_to_grate_cheese_per_omelet + time_to_cook_omelet) = 50 := 
by
  sorry

end total_preparation_and_cooking_time_l145_145556


namespace root_of_equation_l145_145365

open Nat

def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))
def permutation (n k : ℕ) : ℕ := (n.factorial) / ((n - k).factorial)

theorem root_of_equation : ∃ x : ℕ, 3 * combination (x - 3) 4 = 5 * permutation (x - 4) 2 ∧ 7 ≤ x ∧ x = 11 := 
by 
  sorry

end root_of_equation_l145_145365


namespace polynomial_sum_l145_145439

variable {R : Type*} [CommRing R] {x y : R}

/-- Given that the sum of a polynomial P and x^2 - y^2 is x^2 + y^2, we want to prove that P is 2y^2. -/
theorem polynomial_sum (P : R) (h : P + (x^2 - y^2) = x^2 + y^2) : P = 2 * y^2 :=
by
  sorry

end polynomial_sum_l145_145439


namespace solve_for_x_l145_145499

-- Define the equation as a predicate
def equation (x : ℝ) : Prop := (0.05 * x + 0.07 * (30 + x) = 15.4)

-- The proof statement
theorem solve_for_x :
  ∃ x : ℝ, equation x ∧ x = 110.8333 :=
by
  existsi (110.8333 : ℝ)
  split
  sorry -- Proof of the equation
  rfl -- Equality proof

end solve_for_x_l145_145499


namespace meeting_time_l145_145714

noncomputable def start_time : ℕ := 13 -- 1 pm in 24-hour format
noncomputable def speed_A : ℕ := 5 -- in kmph
noncomputable def speed_B : ℕ := 7 -- in kmph
noncomputable def initial_distance : ℕ := 24 -- in km

theorem meeting_time : start_time + (initial_distance / (speed_A + speed_B)) = 15 :=
by
  sorry

end meeting_time_l145_145714


namespace minimum_width_l145_145373

theorem minimum_width (A l w : ℝ) (hA : A >= 150) (hl : l = 2 * w) (hA_def : A = w * l) : 
  w >= 5 * Real.sqrt 3 := 
  by
    -- Using the given conditions, we can prove that w >= 5 * sqrt(3)
    sorry

end minimum_width_l145_145373


namespace minimum_value_expression_l145_145472

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a = 1 ∧ b = 1 ∧ c = 1) :
  (a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2) / (a * b * c) = 48 * Real.sqrt 6 := 
by
  sorry

end minimum_value_expression_l145_145472


namespace overall_average_score_l145_145083

-- Definitions used from conditions
def male_students : Nat := 8
def male_avg_score : Real := 83
def female_students : Nat := 28
def female_avg_score : Real := 92

-- Theorem to prove the overall average score is 90
theorem overall_average_score : 
  (male_students * male_avg_score + female_students * female_avg_score) / (male_students + female_students) = 90 := 
by 
  sorry

end overall_average_score_l145_145083


namespace cosine_identity_l145_145772

theorem cosine_identity (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = Real.sqrt 3 / 2) :
  Real.cos (Real.pi / 3 - α) = Real.sqrt 3 / 2 := 
by
  sorry

end cosine_identity_l145_145772


namespace quadratic_y1_gt_y2_l145_145048

theorem quadratic_y1_gt_y2 {a b c y1 y2 : ℝ} (ha : a > 0) (hy1 : y1 = a * (-1)^2 + b * (-1) + c) (hy2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
  sorry

end quadratic_y1_gt_y2_l145_145048


namespace isosceles_triangle_same_area_l145_145404

-- Given conditions of the original isosceles triangle
def original_base : ℝ := 10
def original_side : ℝ := 13

-- The problem states that an isosceles triangle has the base 10 cm and side lengths 13 cm, 
-- we need to show there's another isosceles triangle with a different base but the same area.
theorem isosceles_triangle_same_area : 
  ∃ (new_base : ℝ) (new_side : ℝ), 
    new_base ≠ original_base ∧ 
    (∃ (h1 h2: ℝ), 
      h1 = 12 ∧ 
      h2 = 5 ∧
      1/2 * original_base * h1 = 60 ∧ 
      1/2 * new_base * h2 = 60) := 
sorry

end isosceles_triangle_same_area_l145_145404


namespace plane_hover_central_time_l145_145852

theorem plane_hover_central_time (x : ℕ) (h1 : 3 + x + 2 + 5 + (x + 2) + 4 = 24) : x = 4 := by
  sorry

end plane_hover_central_time_l145_145852


namespace document_completion_time_l145_145686

-- Define the typing rates for different typists
def fast_typist_rate := 1 / 4
def slow_typist_rate := 1 / 9
def additional_typist_rate := 1 / 4

-- Define the number of typists
def num_fast_typists := 2
def num_slow_typists := 3
def num_additional_typists := 2

-- Define the distraction time loss per typist every 30 minutes
def distraction_loss := 1 / 6

-- Define the combined rate without distractions
def combined_rate : ℚ :=
  (num_fast_typists * fast_typist_rate) +
  (num_slow_typists * slow_typist_rate) +
  (num_additional_typists * additional_typist_rate)

-- Define the distraction rate loss per hour (two distractions per hour)
def distraction_rate_loss_per_hour := 2 * distraction_loss

-- Define the effective combined rate considering distractions
def effective_combined_rate : ℚ := combined_rate - distraction_rate_loss_per_hour

-- Prove that the document is completed in 1 hour with the effective rate
theorem document_completion_time :
  effective_combined_rate = 1 :=
sorry

end document_completion_time_l145_145686


namespace probability_X_lt_3_l145_145038

-- Given definitions
def X : ProbabilityTheory.RealDistribution := ProbabilityTheory.Normal 1 σ

-- Conditions
axiom P_0_lt_X_lt_3 : set.prob { x | 0 < x ∧ x < 3 } (X .val) = 0.5
axiom P_0_lt_X_lt_1 : set.prob { x | 0 < x ∧ x < 1 } (X .val) = 0.2

-- Theorem to prove the required probability
theorem probability_X_lt_3 (σ : ℝ) :
  set.prob { x | x < 3 } (X .val) = 0.8 :=
begin
  sorry
end

end probability_X_lt_3_l145_145038


namespace tan_alpha_calc_l145_145712

theorem tan_alpha_calc (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (2 * α) / Real.cos α ^ 2) = 6 :=
by sorry

end tan_alpha_calc_l145_145712


namespace arcsin_zero_l145_145733

theorem arcsin_zero : Real.arcsin 0 = 0 := by
  sorry

end arcsin_zero_l145_145733


namespace probability_of_odd_divisor_l145_145102

noncomputable def factorial_prime_factors : ℕ → List (ℕ × ℕ)
| 21 => [(2, 18), (3, 9), (5, 4), (7, 3), (11, 1), (13, 1), (17, 1), (19, 1)]
| _ => []

def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨_, exp⟩ => acc * (exp + 1)) 1

def number_of_odd_factors (factors : List (ℕ × ℕ)) : ℕ :=
  number_of_factors (factors.filter (λ ⟨p, _⟩ => p != 2))

theorem probability_of_odd_divisor : (number_of_odd_factors (factorial_prime_factors 21)) /
(number_of_factors (factorial_prime_factors 21)) = 1 / 19 := 
by
  sorry

end probability_of_odd_divisor_l145_145102


namespace count_three_letter_sets_l145_145443

-- Define the set of letters
def letters := Finset.range 10  -- representing letters A (0) to J (9)

-- Define the condition that J (represented by 9) cannot be the first initial
def valid_first_initials := letters.erase 9  -- remove 9 (J) from 0 to 9

-- Calculate the number of valid three-letter sets of initials
theorem count_three_letter_sets : 
  let first_initials := valid_first_initials
  let second_initials := letters
  let third_initials := letters
  first_initials.card * second_initials.card * third_initials.card = 900 := by
  sorry

end count_three_letter_sets_l145_145443


namespace greatest_possible_value_of_median_l145_145924

-- Given conditions as definitions
variables (k m r s t : ℕ)

-- condition 1: The average (arithmetic mean) of the 5 integers is 10
def avg_is_10 : Prop := k + m + r + s + t = 50

-- condition 2: The integers are in a strictly increasing order
def increasing_order : Prop := k < m ∧ m < r ∧ r < s ∧ s < t

-- condition 3: t is 20
def t_is_20 : Prop := t = 20

-- The main statement to prove
theorem greatest_possible_value_of_median : 
  avg_is_10 k m r s t → 
  increasing_order k m r s t → 
  t_is_20 t → 
  r = 13 :=
by
  intros
  sorry

end greatest_possible_value_of_median_l145_145924


namespace oranges_to_apples_equivalence_l145_145800

theorem oranges_to_apples_equivalence :
  (forall (o l a : ℝ), 4 * o = 3 * l ∧ 5 * l = 7 * a -> 20 * o = 21 * a) :=
by
  intro o l a
  intro h
  sorry

end oranges_to_apples_equivalence_l145_145800


namespace sophia_book_pages_l145_145668

theorem sophia_book_pages:
  ∃ (P : ℕ), (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 30 ∧ P = 90 :=
by
  sorry

end sophia_book_pages_l145_145668


namespace min_colors_to_distinguish_keys_l145_145551

def min_colors_needed (n : Nat) : Nat :=
  if n <= 2 then n
  else if n >= 6 then 2
  else 3

theorem min_colors_to_distinguish_keys (n : Nat) :
  (n ≤ 2 → min_colors_needed n = n) ∧
  (3 ≤ n ∧ n ≤ 5 → min_colors_needed n = 3) ∧
  (n ≥ 6 → min_colors_needed n = 2) :=
by
  sorry

end min_colors_to_distinguish_keys_l145_145551


namespace base7_addition_l145_145419

theorem base7_addition (X Y : ℕ) (h1 : Y + 2 = X) (h2 : X + 5 = 8) : X + Y = 4 :=
by
  sorry

end base7_addition_l145_145419


namespace exponent_multiplication_l145_145726

theorem exponent_multiplication (a : ℝ) : (a^3) * (a^2) = a^5 := 
by
  -- Using the property of exponents: a^m * a^n = a^(m + n)
  sorry

end exponent_multiplication_l145_145726


namespace evaluate_exponents_l145_145423

theorem evaluate_exponents :
  (5 ^ 0.4) * (5 ^ 0.6) * (5 ^ 0.2) * (5 ^ 0.3) * (5 ^ 0.5) = 25 := 
by
  sorry

end evaluate_exponents_l145_145423


namespace commutative_matrices_implies_fraction_l145_145335

-- Definitions
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 3], ![4, 5]]
def B (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]

-- Theorem Statement
theorem commutative_matrices_implies_fraction (a b c d : ℝ) 
    (h1 : A * B a b c d = B a b c d * A) 
    (h2 : 4 * b ≠ c) : 
    (a - d) / (c - 4 * b) = 3 / 8 :=
by
  sorry

end commutative_matrices_implies_fraction_l145_145335


namespace quadrilateral_is_parallelogram_l145_145679

theorem quadrilateral_is_parallelogram
  (AB BC CD DA : ℝ)
  (K L M N : ℝ)
  (H₁ : K = (AB + BC) / 2)
  (H₂ : L = (BC + CD) / 2)
  (H₃ : M = (CD + DA) / 2)
  (H₄ : N = (DA + AB) / 2)
  (H : K + M + L + N = (AB + BC + CD + DA) / 2)
  : ∃ P Q R S : ℝ, P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ S ≠ P ∧ 
    (P + R = AB) ∧ (Q + S = CD)  := 
sorry

end quadrilateral_is_parallelogram_l145_145679


namespace samantha_score_l145_145196

variables (correct_answers geometry_correct_answers incorrect_answers unanswered_questions : ℕ)
          (points_per_correct : ℝ := 1) (additional_geometry_points : ℝ := 0.5)

def total_score (correct_answers geometry_correct_answers : ℕ) : ℝ :=
  correct_answers * points_per_correct + geometry_correct_answers * additional_geometry_points

theorem samantha_score 
  (Samantha_correct : correct_answers = 15)
  (Samantha_geometry : geometry_correct_answers = 4)
  (Samantha_incorrect : incorrect_answers = 5)
  (Samantha_unanswered : unanswered_questions = 5) :
  total_score correct_answers geometry_correct_answers = 17 := 
by
  sorry

end samantha_score_l145_145196


namespace crayons_per_friend_l145_145977

theorem crayons_per_friend (total_crayons : ℕ) (num_friends : ℕ) (h1 : total_crayons = 210) (h2 : num_friends = 30) : total_crayons / num_friends = 7 :=
by
  sorry

end crayons_per_friend_l145_145977


namespace quadratic_roots_abs_eq_l145_145436

theorem quadratic_roots_abs_eq (x1 x2 m : ℝ) (h1 : x1 > 0) (h2 : x2 < 0) 
  (h_eq_roots : ∀ x, x^2 - (x1 + x2)*x + x1*x2 = 0) : 
  ∃ q : ℝ, q = x^2 - (1 - 4*m)/x + 2 := 
by
  sorry

end quadratic_roots_abs_eq_l145_145436


namespace find_x_l145_145444

theorem find_x (x : ℝ) (h : 3^(x - 2) = 9^3) : x = 8 := 
by 
  sorry

end find_x_l145_145444


namespace harkamal_total_amount_l145_145184

-- Define the conditions as constants
def quantity_grapes : ℕ := 10
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 55

-- Define the cost of grapes and mangoes based on the given conditions
def cost_grapes : ℕ := quantity_grapes * rate_grapes
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes

-- Define the total amount paid
def total_amount_paid : ℕ := cost_grapes + cost_mangoes

-- The theorem stating the problem and the solution
theorem harkamal_total_amount : total_amount_paid = 1195 := by
  -- Proof goes here (omitted)
  sorry

end harkamal_total_amount_l145_145184


namespace compare_fx_l145_145996

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
  x^2 - b * x + c

theorem compare_fx (b c : ℝ) (x : ℝ) (h1 : ∀ x : ℝ, f (1 - x) b c = f (1 + x) b c) (h2 : f 0 b c = 3) :
  f (2^x) b c ≤ f (3^x) b c :=
by
  sorry

end compare_fx_l145_145996


namespace choose_5_starters_including_twins_l145_145269

def number_of_ways_choose_starters (total_players : ℕ) (members_in_lineup : ℕ) (twins1 twins2 : (ℕ × ℕ)) : ℕ :=
1834

theorem choose_5_starters_including_twins :
  number_of_ways_choose_starters 18 5 (1, 2) (3, 4) = 1834 :=
sorry

end choose_5_starters_including_twins_l145_145269


namespace geometric_series_sum_l145_145728

-- Conditions
def is_geometric_series (a r : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a * r ^ n

-- The problem statement translated into Lean: Proving the sum of the series
theorem geometric_series_sum : ∃ S : ℕ → ℝ, is_geometric_series 1 (1/4) S ∧ ∑' n, S n = 4/3 :=
by
  sorry

end geometric_series_sum_l145_145728


namespace number_of_possible_values_l145_145676

-- Define the decimal number s and its representation
def s (e f g h : ℕ) : ℚ := e / 10 + f / 100 + g / 1000 + h / 10000

-- Define the condition that the closest fraction is 2/9
def closest_to_2_9 (s : ℚ) : Prop :=
  abs (s - 2 / 9) < min (abs (s - 1 / 5)) (abs (s - 1 / 6)) ∧
  abs (s - 2 / 9) < min (abs (s - 1 / 5)) (abs (s - 2 / 11))

-- The main theorem stating the number of possible values for s
theorem number_of_possible_values :
  (∃ e f g h : ℕ, 0 ≤ e ∧ e ≤ 9 ∧ 0 ≤ f ∧ f ≤ 9 ∧ 0 ≤ g ∧ g ≤ 9 ∧ 0 ≤ h ∧ h ≤ 9 ∧
    closest_to_2_9 (s e f g h)) → (∃ n : ℕ, n = 169) :=
by
  sorry

end number_of_possible_values_l145_145676


namespace simplify_expr1_simplify_expr2_l145_145665

theorem simplify_expr1 : 
  (1:ℝ) * (-3:ℝ) ^ 0 + (- (1/2:ℝ)) ^ (-2:ℝ) - (-3:ℝ) ^ (-1:ℝ) = 16 / 3 :=
by
  sorry

theorem simplify_expr2 (x : ℝ) : 
  ((-2 * x^3) ^ 2 * (-x^2)) / ((-x)^2) ^ 3 = -4 * x^2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l145_145665


namespace pqr_value_l145_145210

noncomputable def complex_numbers (p q r : ℂ) := p * q + 5 * q = -20 ∧ q * r + 5 * r = -20 ∧ r * p + 5 * p = -20

theorem pqr_value (p q r : ℂ) (h : complex_numbers p q r) : p * q * r = 80 := by
  sorry

end pqr_value_l145_145210


namespace geometric_arithmetic_sequence_sum_l145_145766

theorem geometric_arithmetic_sequence_sum {a b : ℕ → ℝ} (q : ℝ) (n : ℕ) 
(h1 : a 2 = 2)
(h2 : a 2 = 2)
(h3 : 2 * (a 3 + 1) = a 2 + a 4)
(h4 : ∀ (n : ℕ), (a (n + 1)) = a 0 * q ^ (n + 1))
(h5 : b n = n * (n + 1)) :
a 8 + (b 8 - b 7) = 144 :=
by { sorry }

end geometric_arithmetic_sequence_sum_l145_145766


namespace sum_of_non_visible_faces_l145_145986

theorem sum_of_non_visible_faces
    (d1 d2 d3 d4 : Fin 6 → Nat)
    (visible_faces : List Nat)
    (hv : visible_faces = [1, 2, 3, 4, 4, 5, 5, 6]) :
    let total_sum := 4 * 21
    let visible_sum := List.sum visible_faces
    total_sum - visible_sum = 54 := by
  sorry

end sum_of_non_visible_faces_l145_145986


namespace incorrect_equation_l145_145893

noncomputable def x : ℂ := (-1 + Real.sqrt 3 * Complex.I) / 2
noncomputable def y : ℂ := (-1 - Real.sqrt 3 * Complex.I) / 2

theorem incorrect_equation : x^9 + y^9 ≠ -1 := sorry

end incorrect_equation_l145_145893


namespace find_x_l145_145447

theorem find_x (x : ℤ) (h : 3^(x-2) = 9^3) : x = 8 :=
by sorry

end find_x_l145_145447


namespace negation_example_l145_145053

open Classical
variable (x : ℝ)

theorem negation_example :
  (¬ (∀ x : ℝ, 2 * x - 1 > 0)) ↔ (∃ x : ℝ, 2 * x - 1 ≤ 0) :=
by
  sorry

end negation_example_l145_145053


namespace inequality_of_distinct_positives_l145_145596

variable {a b c : ℝ}

theorem inequality_of_distinct_positives (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
(habc : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) :=
by
  sorry

end inequality_of_distinct_positives_l145_145596


namespace smallest_x_for_multiple_l145_145836

theorem smallest_x_for_multiple (x : ℕ) (h : x > 0) :
  (450 * x) % 500 = 0 ↔ x = 10 := by
  sorry

end smallest_x_for_multiple_l145_145836


namespace quadratic_inequality_solution_l145_145099

theorem quadratic_inequality_solution (a : ℝ) (h1 : ∀ x : ℝ, ax^2 + (a + 1) * x + 1 ≥ 0) : a = 1 := by
  sorry

end quadratic_inequality_solution_l145_145099


namespace find_p_max_area_triangle_l145_145609

-- Definitions and conditions for Part 1
def parabola (p : ℝ) := p > 0 ∧ ∀ x y : ℝ, y = x^2 / (2 * p)
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def minimum_distance (F : ℝ × ℝ) (d : ℝ) : Prop := ∃ x y, circle x y ∧ sqrt((x - F.1)^2 + (y - F.2)^2) = d
def distance_condition := minimum_distance (focus 2) 4

-- Statement for Part 1 proof
theorem find_p : ∃ p, parabola p ∧ distance_condition := by
  sorry

-- Definitions and conditions for Part 2
def tangent_to_parabola (P : ℝ × ℝ) (p : ℝ) := -- Assume suitable definition for tangents
def P_on_circle := ∃ x y, circle x y
def tangents_P (P : ℝ × ℝ) (tangents : list (ℝ × ℝ)) := ∀ p, tangent_to_parabola P p
def area_PAB (P : ℝ) (A B : ℝ × ℝ) : ℝ := -- Assume suitable definition for area

-- Statement for Part 2 proof
theorem max_area_triangle (P : ℝ × ℝ) (A B : ℝ × ℝ) : P_on_circle ∧ tangents_P P [A, B] → area_PAB P A B = 20 * sqrt 5 := by
  sorry

end find_p_max_area_triangle_l145_145609


namespace scoops_of_natural_seedless_raisins_l145_145185

theorem scoops_of_natural_seedless_raisins 
  (cost_natural : ℝ := 3.45) 
  (cost_golden : ℝ := 2.55) 
  (num_golden : ℝ := 20) 
  (cost_mixture : ℝ := 3) : 
  ∃ x : ℝ, (3.45 * x + 20 * 2.55 = 3 * (x + 20)) ∧ x = 20 :=
sorry

end scoops_of_natural_seedless_raisins_l145_145185


namespace discount_percentage_l145_145830

theorem discount_percentage
  (number_of_fandoms : ℕ)
  (tshirts_per_fandom : ℕ)
  (price_per_shirt : ℝ)
  (tax_rate : ℝ)
  (total_paid : ℝ)
  (total_expected_price_with_discount_without_tax : ℝ)
  (total_expected_price_without_discount : ℝ)
  (discount_amount : ℝ)
  (discount_percentage : ℝ) :

  number_of_fandoms = 4 ∧
  tshirts_per_fandom = 5 ∧
  price_per_shirt = 15 ∧
  tax_rate = 10 / 100 ∧
  total_paid = 264 ∧
  total_expected_price_with_discount_without_tax = total_paid / (1 + tax_rate) ∧
  total_expected_price_without_discount = number_of_fandoms * tshirts_per_fandom * price_per_shirt ∧
  discount_amount = total_expected_price_without_discount - total_expected_price_with_discount_without_tax ∧
  discount_percentage = (discount_amount / total_expected_price_without_discount) * 100 ->

  discount_percentage = 20 :=
sorry

end discount_percentage_l145_145830


namespace poly_comp_eq_l145_145213

variable {K : Type*} [Field K]

theorem poly_comp_eq {Q1 Q2 : Polynomial K} (P : Polynomial K) (hP : ¬P.degree = 0) :
  Q1.comp P = Q2.comp P → Q1 = Q2 :=
by
  intro h
  sorry

end poly_comp_eq_l145_145213


namespace people_got_off_train_l145_145109

theorem people_got_off_train (initial_people : ℕ) (people_left : ℕ) (people_got_off : ℕ) 
  (h1 : initial_people = 48) 
  (h2 : people_left = 31) 
  : people_got_off = 17 := by
  sorry

end people_got_off_train_l145_145109


namespace inequality_problem_l145_145594

open Real

theorem inequality_problem 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h1 : x + y^2016 ≥ 1) : 
  x^2016 + y > 1 - 1/100 :=
by
  sorry

end inequality_problem_l145_145594


namespace cone_volume_l145_145828

theorem cone_volume (V_cylinder : ℝ) (V_cone : ℝ) (h : V_cylinder = 81 * Real.pi) :
  V_cone = 27 * Real.pi :=
by
  sorry

end cone_volume_l145_145828


namespace catering_service_comparison_l145_145227

theorem catering_service_comparison :
  ∃ (x : ℕ), 150 + 18 * x > 250 + 15 * x ∧ (∀ y : ℕ, y < x -> (150 + 18 * y ≤ 250 + 15 * y)) ∧ x = 34 :=
sorry

end catering_service_comparison_l145_145227


namespace geometric_series_sum_l145_145729

-- Conditions
def is_geometric_series (a r : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a * r ^ n

-- The problem statement translated into Lean: Proving the sum of the series
theorem geometric_series_sum : ∃ S : ℕ → ℝ, is_geometric_series 1 (1/4) S ∧ ∑' n, S n = 4/3 :=
by
  sorry

end geometric_series_sum_l145_145729


namespace woman_work_rate_l145_145126

theorem woman_work_rate (M W : ℝ) (h1 : 10 * M + 15 * W = 1 / 8) (h2 : M = 1 / 100) : W = 1 / 600 :=
by 
  sorry

end woman_work_rate_l145_145126


namespace peaches_total_l145_145540

def peaches_in_basket (a b : Nat) : Nat :=
  a + b 

theorem peaches_total (a b : Nat) (h1 : a = 20) (h2 : b = 25) : peaches_in_basket a b = 45 := 
by
  sorry

end peaches_total_l145_145540


namespace monotonic_range_of_b_l145_145598

noncomputable def f (b x : ℝ) : ℝ := x^3 - b * x^2 + 3 * x - 5

theorem monotonic_range_of_b (b : ℝ) : (∀ x y: ℝ, (f b x) ≤ (f b y) → x ≤ y) ↔ -3 ≤ b ∧ b ≤ 3 :=
sorry

end monotonic_range_of_b_l145_145598


namespace transformed_equation_solutions_l145_145992

theorem transformed_equation_solutions :
  (∀ x : ℝ, x^2 + 2 * x - 3 = 0 → (x = 1 ∨ x = -3)) →
  (∀ x : ℝ, (x + 3)^2 + 2 * (x + 3) - 3 = 0 → (x = -2 ∨ x = -6)) :=
by
  intro h
  sorry

end transformed_equation_solutions_l145_145992


namespace min_value_l145_145479

theorem min_value : ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (a = 1) → (b = 1) → (c = 1) →
  (∃ x, x = (a^2 + 4 * a + 2) / a ∧ x ≥ 6) ∧
  (∃ y, y = (b^2 + 4 * b + 2) / b ∧ y ≥ 6) ∧
  (∃ z, z = (c^2 + 4 * c + 2) / c ∧ z ≥ 6) →
  (∃ m, m = ((a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2)) / (a * b * c) ∧ m = 216) :=
by {
  sorry
}

end min_value_l145_145479


namespace motorcyclist_initial_speed_l145_145399

theorem motorcyclist_initial_speed (x : ℝ) : 
  (120 = x * (120 / x)) ∧
  (120 = x + 6) → 
  (120 / x = 1 + 1/6 + (120 - x) / (x + 6)) →
  (x = 48) :=
by
  sorry

end motorcyclist_initial_speed_l145_145399


namespace cost_of_items_l145_145927

theorem cost_of_items (M R F : ℝ)
  (h1 : 10 * M = 24 * R) 
  (h2 : F = 2 * R) 
  (h3 : F = 20.50) : 
  4 * M + 3 * R + 5 * F = 231.65 := 
by
  sorry

end cost_of_items_l145_145927


namespace incorrect_eqn_x9_y9_neg1_l145_145891

theorem incorrect_eqn_x9_y9_neg1 (x y : ℂ) 
  (hx : x = (-1 + Complex.I * Real.sqrt 3) / 2) 
  (hy : y = (-1 - Complex.I * Real.sqrt 3) / 2) : 
  x^9 + y^9 ≠ -1 :=
sorry

end incorrect_eqn_x9_y9_neg1_l145_145891


namespace remainder_of_large_number_l145_145692

theorem remainder_of_large_number (n : ℕ) (r : ℕ) (h : n = 2468135792) :
  (n % 101) = 52 := 
by
  have h1 : (10 ^ 8 - 1) % 101 = 0 := sorry
  have h2 : (10 ^ 6 - 1) % 101 = 0 := sorry
  have h3 : (10 ^ 4 - 1) % 101 = 0 := sorry
  have h4 : (10 ^ 2 - 1) % 101 = 99 % 101 := sorry

  -- Using these properties to simplify n
  have n_decomposition : 2468135792 = 24 * 10 ^ 8 + 68 * 10 ^ 6 + 13 * 10 ^ 4 + 57 * 10 ^ 2 + 92 := sorry
  have div_property : 
    (24 * 10 ^ 8 + 68 * 10 ^ 6 + 13 * 10 ^ 4 + 57 * 10 ^ 2 + 92 - (24 + 68 + 13 + 57 + 92)) % 101 = 0 := sorry

  have simplified_sum : (24 + 68 + 13 + 57 + 92 = 254 := by norm_num) := sorry
  have resulting_mod : 254 % 101 = 52 := by norm_num

  -- Thus n % 101 = 52
  exact resulting_mod

end remainder_of_large_number_l145_145692


namespace root_exists_in_interval_l145_145532

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_exists_in_interval : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 := 
by
  sorry

end root_exists_in_interval_l145_145532


namespace matrix_sum_correct_l145_145000

def mat1 : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -1], ![3, 7]]
def mat2 : Matrix (Fin 2) (Fin 2) ℤ := ![![ -6, 8], ![5, -2]]
def mat_sum : Matrix (Fin 2) (Fin 2) ℤ := ![![-2, 7], ![8, 5]]

theorem matrix_sum_correct : mat1 + mat2 = mat_sum :=
by
  rw [mat1, mat2]
  sorry

end matrix_sum_correct_l145_145000


namespace sequence_sum_l145_145748

theorem sequence_sum :
  ∃ (r : ℝ) (x y : ℝ),
    (r = 1 / 4) ∧
    (x = 256 * r) ∧
    (y = x * r) ∧
    (x + y = 80) :=
by
  use 1 / 4, 64, 16
  split
  . exact rfl
  split
  . exact rfl
  split
  . exact rfl
  . norm_num


end sequence_sum_l145_145748


namespace f_inequality_solution_set_l145_145440

noncomputable
def f : ℝ → ℝ := sorry

axiom f_at_1 : f 1 = 1
axiom f_deriv : ∀ x : ℝ, deriv f x < 1/3

theorem f_inequality_solution_set :
  {x : ℝ | f (x^2) > (x^2 / 3) + 2 / 3} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end f_inequality_solution_set_l145_145440


namespace factorize_problem_1_factorize_problem_2_l145_145755

variables (x y : ℝ)

-- Problem 1: Prove that x^2 * y - 4 * x * y + 4 * y = y * (x - 2) ^ 2
theorem factorize_problem_1 : 
  x^2 * y - 4 * x * y + 4 * y = y * (x - 2) ^ 2 :=
sorry

-- Problem 2: Prove that x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y)
theorem factorize_problem_2 : 
  x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) :=
sorry

end factorize_problem_1_factorize_problem_2_l145_145755


namespace min_m_plus_n_l145_145819

theorem min_m_plus_n (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 32 * m = n^5) : m + n = 3 :=
  sorry

end min_m_plus_n_l145_145819


namespace kira_travel_time_l145_145111

def total_travel_time (hours_between_stations : ℕ) (break_minutes : ℕ) : ℕ :=
  let travel_time_hours := 2 * hours_between_stations
  let travel_time_minutes := travel_time_hours * 60
  travel_time_minutes + break_minutes

theorem kira_travel_time : total_travel_time 2 30 = 270 :=
  by sorry

end kira_travel_time_l145_145111


namespace traffic_accident_emergency_number_l145_145506

theorem traffic_accident_emergency_number (A B C D : ℕ) (h1 : A = 122) (h2 : B = 110) (h3 : C = 120) (h4 : D = 114) : 
  A = 122 := 
by
  exact h1

end traffic_accident_emergency_number_l145_145506


namespace tenly_more_stuffed_animals_than_kenley_l145_145223

def mckenna_stuffed_animals := 34
def kenley_stuffed_animals := 2 * mckenna_stuffed_animals
def total_stuffed_animals_all := 175
def total_stuffed_animals_mckenna_kenley := mckenna_stuffed_animals + kenley_stuffed_animals
def tenly_stuffed_animals := total_stuffed_animals_all - total_stuffed_animals_mckenna_kenley
def stuffed_animals_difference := tenly_stuffed_animals - kenley_stuffed_animals

theorem tenly_more_stuffed_animals_than_kenley :
  stuffed_animals_difference = 5 := by
  sorry

end tenly_more_stuffed_animals_than_kenley_l145_145223


namespace angles_equal_l145_145876

theorem angles_equal (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : Real.sin A = 2 * Real.cos B * Real.sin C) : B = C :=
by sorry

end angles_equal_l145_145876


namespace polynomial_p0_l145_145485

theorem polynomial_p0 :
  ∃ p : ℕ → ℚ, (∀ n : ℕ, n ≤ 6 → p (3^n) = 1 / (3^n)) ∧ (p 0 = 1093) :=
by
  sorry

end polynomial_p0_l145_145485


namespace no_nontrivial_solutions_in_integers_l145_145349

theorem no_nontrivial_solutions_in_integers (a b c n : ℤ) 
  (h : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
  by
    sorry

end no_nontrivial_solutions_in_integers_l145_145349


namespace product_eval_l145_145573

theorem product_eval (a : ℤ) (h : a = 3) : (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  sorry

end product_eval_l145_145573


namespace quadratic_y1_gt_y2_l145_145049

theorem quadratic_y1_gt_y2 {a b c y1 y2 : ℝ} (ha : a > 0) (hy1 : y1 = a * (-1)^2 + b * (-1) + c) (hy2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
  sorry

end quadratic_y1_gt_y2_l145_145049


namespace lock_and_key_requirements_l145_145128

/-- There are 7 scientists each with a key to an electronic lock which requires at least 4 scientists to open.
    - Prove that the minimum number of unique features (locks) the electronic lock must have is 35.
    - Prove that each scientist's key should have at least 20 features.
--/
theorem lock_and_key_requirements :
  ∃ (locks : ℕ) (features_per_key : ℕ), 
    locks = 35 ∧ features_per_key = 20 ∧
    (∀ (n_present : ℕ), n_present ≥ 4 → 7 - n_present ≤ 3) ∧
    (∀ (n_absent : ℕ), n_absent ≤ 3 → 7 - n_absent ≥ 4)
:= sorry

end lock_and_key_requirements_l145_145128


namespace middle_term_in_expansion_sum_of_odd_coefficients_weighted_sum_of_coefficients_l145_145041

noncomputable def term_in_expansion (n k : ℕ) : ℚ :=
  (Nat.choose n k) * ((-1/2) ^ k)

theorem middle_term_in_expansion :
  term_in_expansion 8 4 = 35 / 8 := by
  sorry

theorem sum_of_odd_coefficients :
  (term_in_expansion 8 1 + term_in_expansion 8 3 + term_in_expansion 8 5 + term_in_expansion 8 7) = -(205 / 16) := by
  sorry

theorem weighted_sum_of_coefficients :
  ((1 * term_in_expansion 8 1) + (2 * term_in_expansion 8 2) + (3 * term_in_expansion 8 3) + (4 * term_in_expansion 8 4) +
  (5 * term_in_expansion 8 5) + (6 * term_in_expansion 8 6) + (7 * term_in_expansion 8 7) + (8 * term_in_expansion 8 8)) =
  -(1 / 32) := by
  sorry

end middle_term_in_expansion_sum_of_odd_coefficients_weighted_sum_of_coefficients_l145_145041


namespace trigonometric_relationship_l145_145176

noncomputable def a : ℝ := Real.sin (46 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (46 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (46 * Real.pi / 180)

theorem trigonometric_relationship : c > a ∧ a > b :=
by
  -- This is the statement part; the proof will be handled here
  sorry

end trigonometric_relationship_l145_145176


namespace probability_of_male_selected_l145_145833

-- Define the total number of students
def num_students : ℕ := 100

-- Define the number of male students
def num_male_students : ℕ := 25

-- Define the number of students selected
def num_students_selected : ℕ := 20

theorem probability_of_male_selected :
  (num_students_selected : ℚ) / num_students = 1 / 5 :=
by
  sorry

end probability_of_male_selected_l145_145833


namespace tanA_over_tanB_l145_145199

noncomputable def tan_ratios (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A + 2 * c = 0

theorem tanA_over_tanB {A B C a b c : ℝ} (h : tan_ratios A B C a b c) : 
  Real.tan A / Real.tan B = -1 / 3 :=
by
  sorry

end tanA_over_tanB_l145_145199


namespace shirts_sewn_on_tuesday_l145_145079

theorem shirts_sewn_on_tuesday 
  (shirts_monday : ℕ) 
  (shirts_wednesday : ℕ) 
  (total_buttons : ℕ) 
  (buttons_per_shirt : ℕ) 
  (shirts_tuesday : ℕ) 
  (h1: shirts_monday = 4) 
  (h2: shirts_wednesday = 2) 
  (h3: total_buttons = 45) 
  (h4: buttons_per_shirt = 5) 
  (h5: shirts_tuesday * buttons_per_shirt + shirts_monday * buttons_per_shirt + shirts_wednesday * buttons_per_shirt = total_buttons) : 
  shirts_tuesday = 3 :=
by 
  sorry

end shirts_sewn_on_tuesday_l145_145079


namespace ray_has_4_nickels_left_l145_145076

variables {cents_per_nickel : ℕ := 5}

-- Conditions
def initial_cents := 95
def cents_given_to_peter := 25
def cents_given_to_randi := 2 * cents_given_to_peter
def total_cents_given := cents_given_to_peter + cents_given_to_randi
def remaining_cents := initial_cents - total_cents_given

-- Theorem statement
theorem ray_has_4_nickels_left :
  (remaining_cents / cents_per_nickel) = 4 :=
begin
  sorry
end

end ray_has_4_nickels_left_l145_145076


namespace length_of_metallic_sheet_l145_145956

variable (L : ℝ) (width side volume : ℝ)

theorem length_of_metallic_sheet (h1 : width = 36) (h2 : side = 8) (h3 : volume = 5120) :
  ((L - 2 * side) * (width - 2 * side) * side = volume) → L = 48 := 
by
  intros h_eq
  sorry

end length_of_metallic_sheet_l145_145956


namespace suff_but_not_nec_l145_145648

def M (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0
def N (a : ℝ) : Prop := ∃ x : ℝ, (a - 3) * x + 1 = 0

theorem suff_but_not_nec (a : ℝ) : M a → N a ∧ ¬(N a → M a) := by
  sorry

end suff_but_not_nec_l145_145648


namespace product_of_midpoint_coordinates_l145_145418

def x1 := 10
def y1 := -3
def x2 := 4
def y2 := 7

def midpoint_x := (x1 + x2) / 2
def midpoint_y := (y1 + y2) / 2

theorem product_of_midpoint_coordinates : 
  midpoint_x * midpoint_y = 14 :=
by
  sorry

end product_of_midpoint_coordinates_l145_145418


namespace correct_answer_l145_145641

def sum_squares_of_three_consecutive_even_integers (n : ℤ) : ℤ :=
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  a * a + b * b + c * c

def T : Set ℤ :=
  {t | ∃ n : ℤ, t = sum_squares_of_three_consecutive_even_integers n}

theorem correct_answer : (∀ t ∈ T, t % 4 = 0) ∧ (∀ t ∈ T, t % 7 ≠ 0) :=
sorry

end correct_answer_l145_145641


namespace range_of_f_find_a_l145_145390

-- Define the function f
def f (a x : ℝ) : ℝ := -a^2 * x - 2 * a * x + 1

-- Define the proposition for part (1)
theorem range_of_f (a : ℝ) (h : a > 1) : Set.range (f a) = Set.Iio 1 := sorry

-- Define the proposition for part (2)
theorem find_a (a : ℝ) (h : a > 1) (min_value : ∀ x, x ∈ Set.Icc (-2 : ℝ) 1 → f a x ≥ -7) : a = 2 :=
sorry

end range_of_f_find_a_l145_145390


namespace pirate_coins_l145_145655

theorem pirate_coins (x : ℕ) : 
  (x * (x + 1)) / 2 = 3 * x → 4 * x = 20 := by
  sorry

end pirate_coins_l145_145655


namespace geometric_sequence_common_ratio_l145_145171

theorem geometric_sequence_common_ratio (a₁ : ℕ) (S₃ : ℕ) (q : ℤ) 
  (h₁ : a₁ = 2) (h₂ : S₃ = 6) : 
  (q = 1 ∨ q = -2) :=
by
  sorry

end geometric_sequence_common_ratio_l145_145171


namespace find_x_l145_145542

theorem find_x (n x : ℚ) (h1 : 3 * n + x = 6 * n - 10) (h2 : n = 25 / 3) : x = 15 :=
by
  sorry

end find_x_l145_145542


namespace cars_left_in_parking_lot_l145_145937

-- Define constants representing the initial number of cars and cars that went out.
def initial_cars : ℕ := 24
def first_out : ℕ := 8
def second_out : ℕ := 6

-- State the theorem to prove the remaining cars in the parking lot.
theorem cars_left_in_parking_lot : 
  initial_cars - first_out - second_out = 10 := 
by {
  -- Here, 'sorry' is used to indicate the proof is omitted.
  sorry
}

end cars_left_in_parking_lot_l145_145937


namespace find_r_s_l145_145619

theorem find_r_s (r s : ℚ) :
  (-3)^5 - 2*(-3)^4 + 3*(-3)^3 - r*(-3)^2 + s*(-3) - 8 = 0 ∧
  2^5 - 2*(2^4) + 3*(2^3) - r*(2^2) + s*2 - 8 = 0 →
  (r, s) = (-482/15, -1024/15) :=
by
  sorry

end find_r_s_l145_145619


namespace no_integer_solutions_l145_145574

theorem no_integer_solutions (x y : ℤ) (hx : x ≠ 1) : (x^7 - 1) / (x - 1) ≠ y^5 - 1 :=
by
  sorry

end no_integer_solutions_l145_145574


namespace net_gain_is_88837_50_l145_145344

def initial_home_value : ℝ := 500000
def first_sale_price : ℝ := 1.15 * initial_home_value
def first_purchase_price : ℝ := 0.95 * first_sale_price
def second_sale_price : ℝ := 1.1 * first_purchase_price
def second_purchase_price : ℝ := 0.9 * second_sale_price

def total_sales : ℝ := first_sale_price + second_sale_price
def total_purchases : ℝ := first_purchase_price + second_purchase_price
def net_gain_for_A : ℝ := total_sales - total_purchases

theorem net_gain_is_88837_50 : net_gain_for_A = 88837.50 := by
  -- proof steps would go here, but they are omitted per instructions
  sorry

end net_gain_is_88837_50_l145_145344


namespace equation1_solutions_equation2_solutions_equation3_solutions_equation4_no_real_solutions_l145_145817

theorem equation1_solutions (x : ℝ) : x^2 - 4 * x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) :=
by sorry

theorem equation2_solutions (x : ℝ) : x * (3 * x + 1) = 2 * (3 * x + 1) ↔ (x = -1 / 3 ∨ x = 2) :=
by sorry

theorem equation3_solutions (x : ℝ) : 2 * x^2 + x - 4 = 0 ↔ (x = (-1 + Real.sqrt 33) / 4 ∨ x = (-1 - Real.sqrt 33) / 4) :=
by sorry

theorem equation4_no_real_solutions (x : ℝ) : ¬ ∃ x, 4 * x^2 - 3 * x + 1 = 0 :=
by sorry

end equation1_solutions_equation2_solutions_equation3_solutions_equation4_no_real_solutions_l145_145817


namespace equivalent_expression_ratio_l145_145813

theorem equivalent_expression_ratio :
  let c : ℚ := 8
  let p : ℚ := -3 / 8
  let q : ℚ := 604 / 32
  (c * ((j + p)^2) + q = 8 * j^2 - 6 * j + 20) →
  (q / p) = -151 / 3 :=
by
  intros c p q h
  rw [← h]
  sorry

end equivalent_expression_ratio_l145_145813


namespace total_buildings_proof_l145_145799

-- Given conditions
variables (stores_pittsburgh hospitals_pittsburgh schools_pittsburgh police_stations_pittsburgh : ℕ)
variables (stores_new hospitals_new schools_new police_stations_new buildings_new : ℕ)

-- Given values for Pittsburgh
def stores_pittsburgh := 2000
def hospitals_pittsburgh := 500
def schools_pittsburgh := 200
def police_stations_pittsburgh := 20

-- Definitions for the new city
def stores_new := stores_pittsburgh / 2
def hospitals_new := 2 * hospitals_pittsburgh
def schools_new := schools_pittsburgh - 50
def police_stations_new := police_stations_pittsburgh + 5
def buildings_new := stores_new + hospitals_new + schools_new + police_stations_new

-- Statement to prove
theorem total_buildings_proof : buildings_new = 2175 := by
  dsimp [buildings_new, stores_new, hospitals_new, schools_new, police_stations_new] 
  dsimp [stores_pittsburgh, hospitals_pittsburgh, schools_pittsburgh, police_stations_pittsburgh]
  rfl

end total_buildings_proof_l145_145799


namespace M_minus_N_l145_145909

theorem M_minus_N (a b c d : ℕ) (h1 : a + b = 20) (h2 : a + c = 24) (h3 : a + d = 22) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : 
  let M := 2 * b + 26
  let N := 2 * 1 + 26
  (M - N) = 36 :=
by
  sorry

end M_minus_N_l145_145909


namespace SeedMixtureWeights_l145_145662

theorem SeedMixtureWeights (x y z : ℝ) (h1 : x + y + z = 8) (h2 : x / 3 = y / 2) (h3 : x / 3 = z / 3) :
  x = 3 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end SeedMixtureWeights_l145_145662


namespace george_purchased_two_large_pizzas_l145_145720

noncomputable def small_slices := 4
noncomputable def large_slices := 8
noncomputable def small_pizzas_purchased := 3
noncomputable def george_slices := 3
noncomputable def bob_slices := george_slices + 1
noncomputable def susie_slices := bob_slices / 2
noncomputable def bill_slices := 3
noncomputable def fred_slices := 3
noncomputable def mark_slices := 3
noncomputable def leftover_slices := 10

noncomputable def total_slices_consumed := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

noncomputable def total_slices_before_eating := total_slices_consumed + leftover_slices

noncomputable def small_pizza_total_slices := small_pizzas_purchased * small_slices

noncomputable def large_pizza_total_slices := total_slices_before_eating - small_pizza_total_slices

noncomputable def large_pizzas_purchased := large_pizza_total_slices / large_slices

theorem george_purchased_two_large_pizzas : large_pizzas_purchased = 2 :=
sorry

end george_purchased_two_large_pizzas_l145_145720


namespace binomial_pmf_value_l145_145438

namespace BinomialDistributionProof

open ProbabilityTheory

noncomputable def binomial_pmf (n : ℕ) (p : ℚ) : ℕ → ℚ :=
  λ k, (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem binomial_pmf_value : binomial_pmf 6 (1 / 3) 2 = 80 / 243 := by
  sorry

end BinomialDistributionProof

end binomial_pmf_value_l145_145438


namespace tim_total_score_l145_145144

-- Definitions from conditions
def single_line_points : ℕ := 1000
def tetris_points : ℕ := 8 * single_line_points
def doubled_tetris_points : ℕ := 2 * tetris_points
def num_singles : ℕ := 6
def num_tetrises : ℕ := 4
def consecutive_tetrises : ℕ := 2
def regular_tetrises : ℕ := num_tetrises - consecutive_tetrises

-- Total score calculation
def total_score : ℕ :=
  num_singles * single_line_points +
  regular_tetrises * tetris_points +
  consecutive_tetrises * doubled_tetris_points

-- Prove that Tim's total score is 54000
theorem tim_total_score : total_score = 54000 :=
by 
  sorry

end tim_total_score_l145_145144


namespace simplify_radical_1_simplify_radical_2_find_value_of_a_l145_145125

-- Problem 1
theorem simplify_radical_1 : 7 + 2 * (Real.sqrt 10) = (Real.sqrt 2 + Real.sqrt 5) ^ 2 := 
by sorry

-- Problem 2
theorem simplify_radical_2 : (Real.sqrt (11 - 6 * (Real.sqrt 2))) = 3 - Real.sqrt 2 := 
by sorry

-- Problem 3
theorem find_value_of_a (a m n : ℕ) (h : a + 2 * Real.sqrt 21 = (Real.sqrt m + Real.sqrt n) ^ 2) : 
  a = 10 ∨ a = 22 := 
by sorry

end simplify_radical_1_simplify_radical_2_find_value_of_a_l145_145125


namespace number_of_correct_propositions_l145_145877

variable (Ω : Type) (R : Type) [Nonempty Ω] [Nonempty R]

-- Definitions of the conditions
def carsPassingIntersection (t : ℝ) : Ω → ℕ := sorry
def passengersInWaitingRoom (t : ℝ) : Ω → ℕ := sorry
def maximumFlowRiverEachYear : Ω → ℝ := sorry
def peopleExitingTheater (t : ℝ) : Ω → ℕ := sorry

-- Statement to prove the number of correct propositions
theorem number_of_correct_propositions : 4 = 4 := sorry

end number_of_correct_propositions_l145_145877


namespace fenced_area_with_cutout_l145_145087

def rectangle_area (length width : ℝ) : ℝ := length * width

def square_area (side : ℝ) : ℝ := side * side

theorem fenced_area_with_cutout :
  rectangle_area 20 18 - square_area 4 = 344 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end fenced_area_with_cutout_l145_145087


namespace total_pens_is_50_l145_145195

theorem total_pens_is_50
  (red : ℕ) (black : ℕ) (blue : ℕ) (green : ℕ) (purple : ℕ) (total : ℕ)
  (h1 : red = 8)
  (h2 : black = 3 / 2 * red)
  (h3 : blue = black + 5 ∧ blue = 1 / 5 * total)
  (h4 : green = blue / 2)
  (h5 : purple = 5)
  : total = red + black + blue + green + purple := sorry

end total_pens_is_50_l145_145195


namespace ascending_order_proof_l145_145281

noncomputable def frac1 : ℚ := 1 / 2
noncomputable def frac2 : ℚ := 3 / 4
noncomputable def frac3 : ℚ := 1 / 5
noncomputable def dec1 : ℚ := 0.25
noncomputable def dec2 : ℚ := 0.42

theorem ascending_order_proof :
  frac3 < dec1 ∧ dec1 < dec2 ∧ dec2 < frac1 ∧ frac1 < frac2 :=
by {
  -- The proof will show the conversions mentioned in solution steps
  sorry
}

end ascending_order_proof_l145_145281


namespace det_products_congruent_to_1_mod_101_l145_145482

theorem det_products_congruent_to_1_mod_101 :
  let A := matrix (fin 100) (fin 100) (λ m n : fin 100, (m.succ : ℕ) * (n.succ : ℕ)) in
  ∀ σ : equiv.perm (fin 100), 
    (∏ i, A (i, σ i) : ℕ) % 101 = 1 :=
by
  sorry

end det_products_congruent_to_1_mod_101_l145_145482


namespace find_m_plus_n_l145_145047

theorem find_m_plus_n (a m n : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : a^m = n) (h4 : a^0 = 1) : m + n = 1 :=
sorry

end find_m_plus_n_l145_145047


namespace least_number_to_add_l145_145707

theorem least_number_to_add (n : ℕ) (h₁ : n = 1054) :
  ∃ k : ℕ, (n + k) % 23 = 0 ∧ k = 4 :=
by
  use 4
  have h₂ : n % 23 = 19 := by sorry
  have h₃ : (n + 4) % 23 = 0 := by sorry
  exact ⟨h₃, rfl⟩

end least_number_to_add_l145_145707


namespace area_of_garden_l145_145221

theorem area_of_garden (L P : ℝ) (H1 : 1500 = 30 * L) (H2 : 1500 = 12 * P) (H3 : P = 2 * L + 2 * (P / 2 - L)) : 
  (L * (P/2 - L)) = 625 :=
by
  sorry

end area_of_garden_l145_145221


namespace chord_bisected_by_point_l145_145789

theorem chord_bisected_by_point (x y : ℝ) (h : (x - 2)^2 / 16 + (y - 1)^2 / 8 = 1) :
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -3 ∧ (∀ x y : ℝ, (a * x + b * y + c = 0 ↔ (x - 2)^2 / 16 + (y - 1)^2 / 8 = 1)) := by
  sorry

end chord_bisected_by_point_l145_145789


namespace geometric_sequence_product_l145_145330

-- Defining the geometric sequence and the equation
noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def satisfies_quadratic_roots (a : ℕ → ℝ) : Prop :=
  (a 2 = -1 ∧ a 18 = -16 / (-1 + 16 / -1) ∨
  a 18 = -1 ∧ a 2 = -16 / (-1 + 16 / -1))

-- Problem statement
theorem geometric_sequence_product (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_roots : satisfies_quadratic_roots a) : 
  a 3 * a 10 * a 17 = -64 :=
sorry

end geometric_sequence_product_l145_145330


namespace square_area_is_correct_l145_145376

-- Define the condition: the side length of the square field
def side_length : ℝ := 7

-- Define the theorem to prove the area of the square field with given side length
theorem square_area_is_correct : side_length * side_length = 49 := by
  -- Proof goes here
  sorry

end square_area_is_correct_l145_145376


namespace find_solutions_l145_145863

theorem find_solutions (n k : ℕ) (hn : n > 0) (hk : k > 0) : 
  n! + n = n^k → (n, k) = (2, 2) ∨ (n, k) = (3, 2) ∨ (n, k) = (5, 3) :=
sorry

end find_solutions_l145_145863


namespace bill_cooking_time_l145_145557

def total_time_spent 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ) : ℕ :=
num_peppers * chop_pepper_time + 
num_onions * chop_onion_time + 
num_omelets * grate_cheese_time + 
num_omelets * cook_omelet_time

theorem bill_cooking_time 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ)
  (chop_pepper_time_eq : chop_pepper_time = 3)
  (chop_onion_time_eq : chop_onion_time = 4)
  (grate_cheese_time_eq : grate_cheese_time = 1)
  (cook_omelet_time_eq : cook_omelet_time = 5)
  (num_peppers_eq : num_peppers = 4)
  (num_onions_eq : num_onions = 2)
  (num_omelets_eq : num_omelets = 5) :
  total_time_spent chop_pepper_time chop_onion_time grate_cheese_time cook_omelet_time num_peppers num_onions num_omelets = 50 :=
by {
  sorry
}

end bill_cooking_time_l145_145557


namespace rolls_in_package_l145_145133

theorem rolls_in_package (n : ℕ) :
  (9 : ℝ) = (n : ℝ) * (1 - 0.25) → n = 12 :=
by
  sorry

end rolls_in_package_l145_145133


namespace derivative_at_one_l145_145008

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x

theorem derivative_at_one : (deriv f 1) = 4 := by
  sorry

end derivative_at_one_l145_145008


namespace fanfan_home_distance_l145_145226

theorem fanfan_home_distance (x y z : ℝ) 
  (h1 : x / 3 = 10) 
  (h2 : x / 3 + y / 2 = 25) 
  (h3 : x / 3 + y / 2 + z = 85) :
  x + y + z = 120 :=
sorry

end fanfan_home_distance_l145_145226


namespace van_distance_l145_145276

theorem van_distance (D : ℝ) (t_initial t_new : ℝ) (speed_new : ℝ) 
  (h1 : t_initial = 6) 
  (h2 : t_new = (3 / 2) * t_initial) 
  (h3 : speed_new = 30) 
  (h4 : D = speed_new * t_new) : 
  D = 270 :=
by
  sorry

end van_distance_l145_145276


namespace solve_simultaneous_equations_l145_145354

theorem solve_simultaneous_equations (a b : ℚ) : 
  (a + b) * (a^2 - b^2) = 4 ∧ (a - b) * (a^2 + b^2) = 5 / 2 → 
  (a = 3 / 2 ∧ b = 1 / 2) ∨ (a = -1 / 2 ∧ b = -3 / 2) :=
by
  sorry

end solve_simultaneous_equations_l145_145354


namespace intersection_M_N_l145_145510

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def N : Set (ℝ × ℝ) := {p | p.1 = 1}

theorem intersection_M_N : M ∩ N = { (1, 0) } := by
  sorry

end intersection_M_N_l145_145510


namespace largest_angle_is_41_41_degrees_l145_145236

-- Define the triangle with sides such that the altitudes are 10, 12, and 15
noncomputable def triangle_sides (a b c : ℝ) : Prop :=
  let h1 := 10
  let h2 := 12
  let h3 := 15
  2 * (1 / (1/h1 + 1/h2 + 1/h3)) = a * b * c

-- Prove that the largest angle is approximately cos⁻¹(3/4)
theorem largest_angle_is_41_41_degrees :
  ∃ (a b c : ℝ), triangle_sides a b c ∧
  let largest_angle := Real.arccos (3 / 4) in specific_angle_approx_eq largest_angle (41.41 * (π / 180)) :=
sorry

end largest_angle_is_41_41_degrees_l145_145236


namespace only_solution_2_pow_eq_y_sq_plus_y_plus_1_l145_145575

theorem only_solution_2_pow_eq_y_sq_plus_y_plus_1 {x y : ℕ} (h1 : 2^x = y^2 + y + 1) : x = 0 ∧ y = 0 := 
by {
  sorry -- proof goes here
}

end only_solution_2_pow_eq_y_sq_plus_y_plus_1_l145_145575


namespace minimum_value_expression_l145_145477

open Real

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2)) / (a * b * c) ≥ 216 :=
sorry

end minimum_value_expression_l145_145477


namespace sequence_sum_l145_145745

theorem sequence_sum :
  (∀ r: ℝ, 
    (∀ x y: ℝ,
      r = 1 / 4 → 
      x = 256 * r → 
      y = x * r → 
      x + y = 80)) :=
by sorry

end sequence_sum_l145_145745


namespace find_y_in_similar_triangles_l145_145549

-- Define the variables and conditions of the problem
def is_similar (a1 b1 a2 b2 : ℚ) : Prop :=
  a1 / b1 = a2 / b2

-- Problem statement
theorem find_y_in_similar_triangles
  (a1 b1 a2 b2 : ℚ)
  (h1 : a1 = 15)
  (h2 : b1 = 12)
  (h3 : b2 = 10)
  (similarity_condition : is_similar a1 b1 a2 b2) :
  a2 = 25 / 2 :=
by
  rw [h1, h2, h3, is_similar] at similarity_condition
  sorry

end find_y_in_similar_triangles_l145_145549


namespace find_percentage_l145_145319

variable (P x : ℝ)

theorem find_percentage (h1 : x = 10)
    (h2 : (P / 100) * x = 0.05 * 500 - 20) : P = 50 := by
  sorry

end find_percentage_l145_145319


namespace circles_ordering_l145_145859

theorem circles_ordering :
  let rA := 3
  let AB := 12 * Real.pi
  let AC := 28 * Real.pi
  let rB := Real.sqrt 12
  let rC := Real.sqrt 28
  (rA < rB) ∧ (rB < rC) :=
by
  let rA := 3
  let AB := 12 * Real.pi
  let AC := 28 * Real.pi
  let rB := Real.sqrt 12
  let rC := Real.sqrt 28
  have rA_lt_rB: rA < rB := by sorry
  have rB_lt_rC: rB < rC := by sorry
  exact ⟨rA_lt_rB, rB_lt_rC⟩

end circles_ordering_l145_145859


namespace least_perimeter_of_triangle_l145_145362

theorem least_perimeter_of_triangle (c : ℕ) (h1 : 24 + 51 > c) (h2 : c > 27) : 24 + 51 + c = 103 :=
by
  sorry

end least_perimeter_of_triangle_l145_145362


namespace find_x_minus_y_l145_145471

theorem find_x_minus_y (x y n : ℤ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x > y) (h4 : n / 10 < 10 ∧ n / 10 ≥ 1) 
  (h5 : 2 * n = x + y) 
  (h6 : ∃ m : ℤ, m^2 = x * y ∧ m = (n % 10) * 10 + n / 10) 
  : x - y = 66 :=
sorry

end find_x_minus_y_l145_145471


namespace largest_divisor_product_of_consecutive_odds_l145_145377

theorem largest_divisor_product_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) : 
  15 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) :=
sorry

end largest_divisor_product_of_consecutive_odds_l145_145377


namespace instrument_costs_purchasing_plans_l145_145952

variable (x y : ℕ)
variable (a b : ℕ)

theorem instrument_costs : 
  (2 * x + 3 * y = 1700 ∧ 3 * x + y = 1500) →
  x = 400 ∧ y = 300 := 
by 
  intros h
  sorry

theorem purchasing_plans :
  (x = 400) → (y = 300) → (3 * a + 10 = b) →
  (400 * a + 300 * b ≤ 30000) →
  ((760 - 400) * a + (540 - 300) * b ≥ 21600) →
  (a = 18 ∧ b = 64 ∨ a = 19 ∧ b = 67 ∨ a = 20 ∧ b = 70) :=
by
  intros hx hy hab hcost hprofit
  sorry

end instrument_costs_purchasing_plans_l145_145952


namespace dvd_count_correct_l145_145342

def total_dvds (store_dvds online_dvds : Nat) : Nat :=
  store_dvds + online_dvds

theorem dvd_count_correct :
  total_dvds 8 2 = 10 :=
by
  sorry

end dvd_count_correct_l145_145342


namespace problem_proof_l145_145595

theorem problem_proof (x y : ℝ) (h_cond : (x + 3)^2 + |y - 2| = 0) : (x + y)^y = 1 :=
by
  sorry

end problem_proof_l145_145595


namespace can_form_set_l145_145260

-- Define each group of objects based on given conditions
def famous_movie_stars : Type := sorry
def small_rivers_in_our_country : Type := sorry
def students_2012_senior_class_Panzhihua : Type := sorry
def difficult_high_school_math_problems : Type := sorry

-- Define the property of having well-defined elements
def has_definite_elements (T : Type) : Prop := sorry

-- The groups in terms of propositions
def group_A : Prop := ¬ has_definite_elements famous_movie_stars
def group_B : Prop := ¬ has_definite_elements small_rivers_in_our_country
def group_C : Prop := has_definite_elements students_2012_senior_class_Panzhihua
def group_D : Prop := ¬ has_definite_elements difficult_high_school_math_problems

-- We need to prove that group C can form a set
theorem can_form_set : group_C :=
by
  sorry

end can_form_set_l145_145260


namespace cos_alpha_beta_half_l145_145434

open Real

theorem cos_alpha_beta_half (α β : ℝ)
  (h1 : cos (α - β / 2) = -1 / 3)
  (h2 : sin (α / 2 - β) = 1 / 4)
  (h3 : 3 * π / 2 < α ∧ α < 2 * π)
  (h4 : π / 2 < β ∧ β < π) :
  cos ((α + β) / 2) = -(2 * sqrt 2 + sqrt 15) / 12 :=
by
  sorry

end cos_alpha_beta_half_l145_145434


namespace function_pass_through_point_l145_145241

theorem function_pass_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), y = a^(x-2) - 1 ∧ (x, y) = (2, 0) := 
by
  use 2
  use 0
  sorry

end function_pass_through_point_l145_145241


namespace multiplication_addition_l145_145949

theorem multiplication_addition :
  108 * 108 + 92 * 92 = 20128 :=
by
  sorry

end multiplication_addition_l145_145949


namespace additional_oil_needed_l145_145036

variable (oil_per_cylinder : ℕ) (number_of_cylinders : ℕ) (oil_already_added : ℕ)

theorem additional_oil_needed (h1 : oil_per_cylinder = 8) (h2 : number_of_cylinders = 6) (h3 : oil_already_added = 16) :
  oil_per_cylinder * number_of_cylinders - oil_already_added = 32 :=
by
  -- proof here
  sorry

end additional_oil_needed_l145_145036


namespace cookie_contest_l145_145250

theorem cookie_contest (A B : ℚ) (hA : A = 5/6) (hB : B = 2/3) :
  A - B = 1/6 :=
by 
  sorry

end cookie_contest_l145_145250


namespace inequality_proof_l145_145912

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (hx1 : x ≤ 1) (hy1 : y ≤ 1) (hz1 : z ≤ 1) :
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 :=
by
  sorry

end inequality_proof_l145_145912


namespace isosceles_triangle_dot_product_l145_145331

theorem isosceles_triangle_dot_product :
  ∀ (A B C D : ℝ³), 
    ∃ (AB AC BC: ℝ), 
      AB = 2 ∧ AC = 2 ∧
      angle A B C = 2 * π / 3 ∧
      (∃ (ratios : ℝ), ratios = 3 ∧ area A C D = ratios * (area A B D)) →
      (vector.dot (A - B) (A - D) = 5 / 2) := 
sorry

end isosceles_triangle_dot_product_l145_145331


namespace circle_radius_6_l145_145590

theorem circle_radius_6 (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 10*x + y^2 + 6*y - k = 0 ↔ (x + 5)^2 + (y + 3)^2 = 36) → k = 2 :=
by
  sorry

end circle_radius_6_l145_145590


namespace trajectory_of_P_l145_145770

-- Definitions for points and distance
structure Point where
  x : ℝ
  y : ℝ

noncomputable def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Fixed points F1 and F2
variable (F1 F2 : Point)
-- Distance condition
axiom dist_F1F2 : dist F1 F2 = 8

-- Moving point P satisfying the condition
variable (P : Point)
axiom dist_PF1_PF2 : dist P F1 + dist P F2 = 8

-- Proof goal: P lies on the line segment F1F2
theorem trajectory_of_P : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = ⟨(1 - t) * F1.x + t * F2.x, (1 - t) * F1.y + t * F2.y⟩ :=
  sorry

end trajectory_of_P_l145_145770


namespace min_distance_squared_l145_145597

noncomputable def graph_function1 (x : ℝ) : ℝ := -x^2 + 3 * Real.log x

noncomputable def point_on_graph1 (a b : ℝ) : Prop := b = graph_function1 a

noncomputable def graph_function2 (x : ℝ) : ℝ := x + 2

noncomputable def point_on_graph2 (c d : ℝ) : Prop := d = graph_function2 c

theorem min_distance_squared (a b c d : ℝ) 
  (hP : point_on_graph1 a b)
  (hQ : point_on_graph2 c d) :
  (a - c)^2 + (b - d)^2 = 8 := 
sorry

end min_distance_squared_l145_145597


namespace andrew_paid_correct_amount_l145_145149

-- Definitions of the conditions
def cost_of_grapes : ℝ := 7 * 68
def cost_of_mangoes : ℝ := 9 * 48
def cost_of_apples : ℝ := 5 * 55
def cost_of_oranges : ℝ := 4 * 38

def total_cost_grapes_and_mangoes_before_discount : ℝ := cost_of_grapes + cost_of_mangoes
def discount_on_grapes_and_mangoes : ℝ := 0.10 * total_cost_grapes_and_mangoes_before_discount
def total_cost_grapes_and_mangoes_after_discount : ℝ := total_cost_grapes_and_mangoes_before_discount - discount_on_grapes_and_mangoes

def total_cost_all_fruits_before_tax : ℝ := total_cost_grapes_and_mangoes_after_discount + cost_of_apples + cost_of_oranges
def sales_tax : ℝ := 0.05 * total_cost_all_fruits_before_tax
def total_amount_to_pay : ℝ := total_cost_all_fruits_before_tax + sales_tax

-- Statement to be proved
theorem andrew_paid_correct_amount :
  total_amount_to_pay = 1306.41 :=
by
  sorry

end andrew_paid_correct_amount_l145_145149


namespace cubic_polynomial_Q_l145_145908

noncomputable def Q (x : ℝ) : ℝ := 27 * x^3 - 162 * x^2 + 297 * x - 156

theorem cubic_polynomial_Q {a b c : ℝ} 
  (h_roots : ∀ x, x^3 - 6 * x^2 + 11 * x - 6 = 0 → x = a ∨ x = b ∨ x = c)
  (h_vieta_sum : a + b + c = 6)
  (h_vieta_prod_sum : ab + bc + ca = 11)
  (h_vieta_prod : abc = 6)
  (hQ : Q a = b + c) 
  (hQb : Q b = a + c) 
  (hQc : Q c = a + b) 
  (hQ_sum : Q (a + b + c) = -27) :
  Q x = 27 * x^3 - 162 * x^2 + 297 * x - 156 :=
by { sorry }

end cubic_polynomial_Q_l145_145908


namespace k_lt_half_plus_sqrt_2n_l145_145067

noncomputable def set_equidistant_points (S : Set (ℝ × ℝ)) (P : ℝ × ℝ) (k : ℕ) : Prop :=
  ∃ T : Finset (ℝ × ℝ), T ⊆ S ∧ ∃ r : ℝ, r > 0 ∧ (T.card ≥ k ∧ ∀ Q ∈ T, dist P Q = r)

theorem k_lt_half_plus_sqrt_2n
  {n k : ℕ}
  (S : Set (ℝ × ℝ))
  (hS₁ : S.card = n)
  (hS₂ : ∀ (P Q R : ℝ × ℝ), P ∈ S → Q ∈ S → R ∈ S → P ≠ Q → Q ≠ R → P ≠ R → ¬Collinear ℝ {P, Q, R})
  (hS₃ : ∀ P ∈ S, set_equidistant_points S P k) :
  k < (1 / 2 : ℝ) + Real.sqrt (2 * n) :=
by
  sorry

end k_lt_half_plus_sqrt_2n_l145_145067


namespace find_pairs_l145_145756

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_pairs (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : 
  (digit_sum (a^(b+1)) = a^b) ↔ 
  ((a = 1) ∨ (a = 3 ∧ b = 2) ∨ (a = 9 ∧ b = 1)) :=
by
  sorry

end find_pairs_l145_145756


namespace treehouse_total_planks_l145_145562

theorem treehouse_total_planks (T : ℕ) 
    (h1 : T / 4 + T / 2 + 20 + 30 = T) : T = 200 :=
sorry

end treehouse_total_planks_l145_145562


namespace find_sum_of_squares_l145_145458

-- Definitions for the conditions: a, b, and c are different prime numbers,
-- and their product equals five times their sum.

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def condition (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  a * b * c = 5 * (a + b + c)

-- Statement of the proof problem.
theorem find_sum_of_squares (a b c : ℕ) (h : condition a b c) : a^2 + b^2 + c^2 = 78 :=
sorry

end find_sum_of_squares_l145_145458


namespace proof_part1_proof_part2_l145_145604

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

def min_distance_condition (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus p).snd + 4 - 1 = 4

def parabola (p : ℝ) : Prop := x^2 = 2 * p * y

axiom parabolic_eq : ∀ x y : ℝ, parabola 2

theorem proof_part1 : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus 2).snd + 4 - 1 = 4 :=
by sorry

noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1 / 2) * (abs (x1 * y2 - x2 * y1))

def max_triangle_area_condition (x y : ℝ) (A B : ℝ → ℝ) (tangent : ℝ → ℝ → ℝ) (P : (ℝ × ℝ)) : Prop :=
  ∀ P ∈ circle 1 (0, -4), P ∈ M_tangent_1 (A x) (B x) → triangle_area (A x) (B x) (P) = 20 * sqrt 5

theorem proof_part2 : ∀ P ∈ circle 1 (0, -4), 
  ∀ A B : (ℝ → ℝ),
  ∀ tangent : ℝ → ℝ → ℝ,
  max_triangle_area_condition P A B tangent P :=
by sorry

end proof_part1_proof_part2_l145_145604


namespace area_of_trapezium_l145_145251

/-- Two parallel sides of a trapezium are 4 cm and 5 cm respectively. 
    The perpendicular distance between the parallel sides is 6 cm.
    Prove that the area of the trapezium is 27 cm². -/
theorem area_of_trapezium (a b h : ℝ) (ha : a = 4) (hb : b = 5) (hh : h = 6) : 
  (1/2) * (a + b) * h = 27 := 
by 
  sorry

end area_of_trapezium_l145_145251


namespace area_within_fence_l145_145096

theorem area_within_fence : 
  let rectangle_area := 20 * 18
  let cutout_area := 4 * 4
  rectangle_area - cutout_area = 344 := by
    -- Definitions
    let rectangle_area := 20 * 18
    let cutout_area := 4 * 4
    
    -- Computation of areas
    show rectangle_area - cutout_area = 344
    sorry

end area_within_fence_l145_145096


namespace perimeter_ratio_l145_145230

variables (K T k R r : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (r = 2 * T / K)
def condition2 : Prop := (2 * T = R * k)

-- The statement we want to prove
theorem perimeter_ratio :
  condition1 K T r →
  condition2 T R k →
  K / k = R / r :=
by
  intros h1 h2
  sorry

end perimeter_ratio_l145_145230


namespace find_a_l145_145304

theorem find_a (a : ℝ) (M : Set ℝ) (N : Set ℝ) : 
  M = {1, 3} → N = {1 - a, 3} → (M ∪ N) = {1, 2, 3} → a = -1 :=
by
  intros hM hN hUnion
  sorry

end find_a_l145_145304


namespace sum_to_fraction_l145_145292

theorem sum_to_fraction :
  (2 / 10) + (3 / 100) + (4 / 1000) + (6 / 10000) + (7 / 100000) = 23467 / 100000 :=
by
  sorry

end sum_to_fraction_l145_145292


namespace sqrt_mult_minus_two_l145_145421

theorem sqrt_mult_minus_two (x y : ℝ) (hx : x = Real.sqrt 3) (hy : y = Real.sqrt 6) : 
  2 < x * y - 2 ∧ x * y - 2 < 3 := by
  sorry

end sqrt_mult_minus_two_l145_145421


namespace train_speed_l145_145262

theorem train_speed (length time_speed: ℝ) (h1 : length = 400) (h2 : time_speed = 16) : length / time_speed = 25 := 
by
    sorry

end train_speed_l145_145262


namespace range_of_alpha_minus_beta_l145_145622

open Real

theorem range_of_alpha_minus_beta (
    α β : ℝ) 
    (h1 : -π / 2 < α) 
    (h2 : α < 0)
    (h3 : 0 < β)
    (h4 : β < π / 3)
  : -5 * π / 6 < α - β ∧ α - β < 0 :=
by
  sorry

end range_of_alpha_minus_beta_l145_145622


namespace solve_for_x_l145_145496

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 15.4 → x = 110.8333333 :=
by
  intro h
  sorry

end solve_for_x_l145_145496


namespace four_digit_number_conditions_l145_145272

theorem four_digit_number_conditions :
  ∃ (a b c d : ℕ), 
    (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧ 
    (a * 1000 + b * 100 + c * 10 + d = 10 * 23) ∧ 
    (a + b + c + d = 26) ∧ 
    ((b * d / 10) % 10 = a + c) ∧ 
    ∃ (n : ℕ), (b * d - c^2 = 2^n) ∧ 
    (a * 1000 + b * 100 + c * 10 + d = 1979) :=
sorry

end four_digit_number_conditions_l145_145272


namespace find_SSE_l145_145061

theorem find_SSE (SST SSR : ℝ) (h1 : SST = 13) (h2 : SSR = 10) : SST - SSR = 3 :=
by
  sorry

end find_SSE_l145_145061


namespace sum_of_squares_of_diagonals_l145_145870

variable (OP R : ℝ)

theorem sum_of_squares_of_diagonals (AC BD : ℝ) :
  AC^2 + BD^2 = 8 * R^2 - 4 * OP^2 :=
sorry

end sum_of_squares_of_diagonals_l145_145870


namespace sum_of_a_for_repeated_root_l145_145865

theorem sum_of_a_for_repeated_root :
  ∀ a : ℝ, (∀ x : ℝ, 2 * x^2 + a * x + 10 * x + 16 = 0 → 
               (a + 10 = 8 * Real.sqrt 2 ∨ a + 10 = -8 * Real.sqrt 2)) → 
               (a = -10 + 8 * Real.sqrt 2 ∨ a = -10 - 8 * Real.sqrt 2) → 
               ((-10 + 8 * Real.sqrt 2) + (-10 - 8 * Real.sqrt 2) = -20) := by
sorry

end sum_of_a_for_repeated_root_l145_145865


namespace least_number_of_stamps_l145_145724

theorem least_number_of_stamps (p q : ℕ) (h : 5 * p + 4 * q = 50) : p + q = 11 :=
sorry

end least_number_of_stamps_l145_145724


namespace train_length_l145_145705

noncomputable def speed_kmph := 80
noncomputable def time_seconds := 5

 noncomputable def speed_mps := (speed_kmph * 1000) / 3600

 noncomputable def length_train : ℝ := speed_mps * time_seconds

theorem train_length : length_train = 111.1 := by
  sorry

end train_length_l145_145705


namespace find_larger_number_l145_145237

-- Define the conditions
variables (L S : ℕ)

theorem find_larger_number (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end find_larger_number_l145_145237


namespace gerald_bars_l145_145168

theorem gerald_bars (G : ℕ) 
  (H1 : ∀ G, ∀ teacher_bars : ℕ, teacher_bars = 2 * G → total_bars = G + teacher_bars) 
  (H2 : ∀ total_bars : ℕ, total_squares = total_bars * 8 → total_squares_needed = 24 * 7) 
  (H3 : ∀ total_squares : ℕ, total_squares_needed = 24 * 7) 
  : G = 7 :=
by
  sorry

end gerald_bars_l145_145168


namespace melissa_gave_x_books_l145_145915

-- Define the initial conditions as constants
def initial_melissa_books : ℝ := 123
def initial_jordan_books : ℝ := 27
def final_melissa_books (x : ℝ) : ℝ := initial_melissa_books - x
def final_jordan_books (x : ℝ) : ℝ := initial_jordan_books + x

-- The main theorem to prove how many books Melissa gave to Jordan
theorem melissa_gave_x_books : ∃ x : ℝ, final_melissa_books x = 3 * final_jordan_books x ∧ x = 10.5 :=
sorry

end melissa_gave_x_books_l145_145915


namespace perpendicular_os_bc_l145_145645

variable {A B C O S : Type}

noncomputable def acute_triangle (A B C : Type) := true -- Placeholder definition for acute triangle.

noncomputable def circumcenter (O : Type) (A B C : Type) := true -- Placeholder definition for circumcenter.

noncomputable def line_intersects_circumcircle_second_time (AC : Type) (circ : Type) (S : Type) := true -- Placeholder def.

-- Define the problem in Lean
theorem perpendicular_os_bc
  (ABC_is_acute : acute_triangle A B C)
  (O_is_circumcenter : circumcenter O A B C)
  (AC_intersects_AOB_circumcircle_at_S : line_intersects_circumcircle_second_time (A → C) (A → B → O) S) :
  true := -- Place for the proof that OS ⊥ BC
sorry

end perpendicular_os_bc_l145_145645


namespace cost_of_scissor_l145_145116

noncomputable def scissor_cost (initial_money: ℕ) (scissors: ℕ) (eraser_count: ℕ) (eraser_cost: ℕ) (remaining_money: ℕ) :=
  (initial_money - remaining_money - (eraser_count * eraser_cost)) / scissors

theorem cost_of_scissor : scissor_cost 100 8 10 4 20 = 5 := 
by 
  sorry 

end cost_of_scissor_l145_145116


namespace cubes_sum_correct_l145_145098

noncomputable def max_cubes : ℕ := 11
noncomputable def min_cubes : ℕ := 9

theorem cubes_sum_correct : max_cubes + min_cubes = 20 :=
by
  unfold max_cubes min_cubes
  sorry

end cubes_sum_correct_l145_145098


namespace not_divisor_60_l145_145820

variable (k : ℤ)
def n : ℤ := k * (k + 1) * (k + 2)

theorem not_divisor_60 
  (h₁ : ∃ k, n = k * (k + 1) * (k + 2) ∧ 5 ∣ n) : ¬(60 ∣ n) := 
sorry

end not_divisor_60_l145_145820


namespace original_cost_prices_l145_145721

variable (COST_A COST_B COST_C : ℝ)

theorem original_cost_prices :
  (COST_A * 0.8 + 100 = COST_A * 1.05) →
  (COST_B * 1.1 - 80 = COST_B * 0.92) →
  (COST_C * 0.85 + 120 = COST_C * 1.07) →
  COST_A = 400 ∧
  COST_B = 4000 / 9 ∧
  COST_C = 6000 / 11 := by
  intro h1 h2 h3
  sorry

end original_cost_prices_l145_145721


namespace good_carrots_total_l145_145975

-- Define the number of carrots picked by Carol and her mother
def carolCarrots := 29
def motherCarrots := 16

-- Define the number of bad carrots
def badCarrots := 7

-- Define the total number of carrots picked by Carol and her mother
def totalCarrots := carolCarrots + motherCarrots

-- Define the total number of good carrots
def goodCarrots := totalCarrots - badCarrots

-- The theorem to prove that the total number of good carrots is 38
theorem good_carrots_total : goodCarrots = 38 := by
  sorry

end good_carrots_total_l145_145975


namespace isosceles_trapezoid_side_length_l145_145669

theorem isosceles_trapezoid_side_length (A b1 b2 h half_diff s : ℝ) (h0 : A = 44) (h1 : b1 = 8) (h2 : b2 = 14) 
    (h3 : A = 0.5 * (b1 + b2) * h)
    (h4 : h = 4) 
    (h5 : half_diff = (b2 - b1) / 2) 
    (h6 : half_diff = 3)
    (h7 : s^2 = h^2 + half_diff^2)
    (h8 : s = 5) : 
    s = 5 :=
by 
    apply h8

end isosceles_trapezoid_side_length_l145_145669


namespace prism_volume_l145_145719

noncomputable def volume_of_prism (l w h : ℝ) : ℝ :=
l * w * h

theorem prism_volume (l w h : ℝ) (h1 : l = 2 * w) (h2 : l * w = 10) (h3 : w * h = 18) (h4 : l * h = 36) :
  volume_of_prism l w h = 36 * Real.sqrt 5 :=
by
  -- Proof goes here
  sorry

end prism_volume_l145_145719


namespace maddie_total_payment_l145_145220

def makeup_cost := 3 * 15
def lipstick_cost := 4 * 2.50
def hair_color_cost := 3 * 4

def total_cost := makeup_cost + lipstick_cost + hair_color_cost

theorem maddie_total_payment : total_cost = 67 := by
  sorry

end maddie_total_payment_l145_145220


namespace number_of_integer_terms_l145_145366

noncomputable def count_integer_terms_in_sequence (n : ℕ) (k : ℕ) (a : ℕ) : ℕ :=
  if h : a = k * 3 ^ n then n + 1 else 0

theorem number_of_integer_terms :
  count_integer_terms_in_sequence 5 (2^3 * 5) 9720 = 6 :=
by sorry

end number_of_integer_terms_l145_145366


namespace compute_expression_l145_145973

section
variable (a : ℝ)

theorem compute_expression :
  (-a^2)^3 * a^3 = -a^9 :=
sorry
end

end compute_expression_l145_145973


namespace radius_of_circle_l145_145137

theorem radius_of_circle (P Q : ℝ) (h : P / Q = 25) : ∃ r : ℝ, 2 * π * r = Q ∧ π * r^2 = P ∧ r = 50 := 
by
  -- Proof starts here
  sorry

end radius_of_circle_l145_145137


namespace seokgi_share_is_67_l145_145922

-- The total length of the wire
def length_of_wire := 150

-- Seokgi's share is 16 cm shorter than Yeseul's share
def is_shorter_by (Y S : ℕ) := S = Y - 16

-- The sum of Yeseul's and Seokgi's shares equals the total length
def total_share (Y S : ℕ) := Y + S = length_of_wire

-- Prove that Seokgi's share is 67 cm
theorem seokgi_share_is_67 (Y S : ℕ) (h1 : is_shorter_by Y S) (h2 : total_share Y S) : 
  S = 67 :=
sorry

end seokgi_share_is_67_l145_145922


namespace range_of_k_l145_145182

theorem range_of_k {x k : ℝ} :
  (∀ x, ((x - 2) * (x + 1) > 0) → ((2 * x + 7) * (x + k) < 0)) →
  (x = -3 ∨ x = -2) → 
  -3 ≤ k ∧ k < 2 :=
sorry

end range_of_k_l145_145182


namespace repetend_of_5_over_17_is_294117_l145_145016

theorem repetend_of_5_over_17_is_294117 :
  (∀ n : ℕ, (5 / 17 : ℚ) - (294117 : ℚ) / (10^6 : ℚ) ^ n = 0) :=
by
  sorry

end repetend_of_5_over_17_is_294117_l145_145016


namespace probability_neither_l145_145507

variable (P : Set ℕ → ℝ) -- Use ℕ as a placeholder for the event space
variables (A B : Set ℕ)
variables (hA : P A = 0.25) (hB : P B = 0.35) (hAB : P (A ∩ B) = 0.15)

theorem probability_neither :
  P (Aᶜ ∩ Bᶜ) = 0.55 :=
by
  sorry

end probability_neither_l145_145507


namespace matrix_commutation_l145_145336

open Matrix

theorem matrix_commutation 
  (a b c d : ℝ) (h₁ : 4 * b ≠ c) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5],
      B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d] in
  A * B = B * A → (a - d) / (c - 4 * b) = 2 :=
by 
  intro A B h₂ 
  sorry

end matrix_commutation_l145_145336


namespace package_contains_12_rolls_l145_145134

-- Define the problem conditions
def package_cost : ℝ := 9
def individual_roll_cost : ℝ := 1
def savings_percent : ℝ := 0.25

-- Define the price per roll in the package
def price_per_roll_in_package := individual_roll_cost * (1 - savings_percent)

-- Define the number of rolls in the package
def number_of_rolls_in_package := package_cost / price_per_roll_in_package

-- Question: How many rolls are in the package?
theorem package_contains_12_rolls : number_of_rolls_in_package = 12 := by
  sorry

end package_contains_12_rolls_l145_145134


namespace perfect_square_iff_divisibility_l145_145228

theorem perfect_square_iff_divisibility (A : ℕ) :
  (∃ d : ℕ, A = d^2) ↔ ∀ n : ℕ, n > 0 → ∃ j : ℕ, 1 ≤ j ∧ j ≤ n ∧ n ∣ (A + j)^2 - A :=
sorry

end perfect_square_iff_divisibility_l145_145228


namespace find_S3m_l145_145106
  
-- Arithmetic sequence with given properties
variable (m : ℕ)
variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

-- Define the conditions
axiom Sm : S m = 30
axiom S2m : S (2 * m) = 100

-- Problem statement to prove
theorem find_S3m : S (3 * m) = 170 :=
by
  sorry

end find_S3m_l145_145106


namespace arc_length_of_sector_l145_145177

theorem arc_length_of_sector (r A l : ℝ) (h_r : r = 2) (h_A : A = π / 3) (h_area : A = 1 / 2 * r * l) : l = π / 3 :=
by
  rw [h_r, h_A] at h_area
  sorry

end arc_length_of_sector_l145_145177


namespace original_price_eq_600_l145_145965

theorem original_price_eq_600 (P : ℝ) (h1 : 300 = P * 0.5) : 
  P = 600 :=
sorry

end original_price_eq_600_l145_145965


namespace each_spider_eats_seven_bugs_l145_145407

theorem each_spider_eats_seven_bugs (initial_bugs : ℕ) (reduction_rate : ℚ) (spiders_introduced : ℕ) (bugs_left : ℕ) (result : ℕ)
  (h1 : initial_bugs = 400)
  (h2 : reduction_rate = 0.80)
  (h3 : spiders_introduced = 12)
  (h4 : bugs_left = 236)
  (h5 : result = initial_bugs * (4 / 5) - bugs_left) :
  (result / spiders_introduced) = 7 :=
by
  sorry

end each_spider_eats_seven_bugs_l145_145407


namespace total_preparation_and_cooking_time_l145_145555

def time_to_chop_pepper : Nat := 3
def time_to_chop_onion : Nat := 4
def time_to_grate_cheese_per_omelet : Nat := 1
def time_to_cook_omelet : Nat := 5
def num_peppers : Nat := 4
def num_onions : Nat := 2
def num_omelets : Nat := 5

theorem total_preparation_and_cooking_time :
  num_peppers * time_to_chop_pepper +
  num_onions * time_to_chop_onion +
  num_omelets * (time_to_grate_cheese_per_omelet + time_to_cook_omelet) = 50 := 
by
  sorry

end total_preparation_and_cooking_time_l145_145555


namespace initial_yards_lost_l145_145717

theorem initial_yards_lost (x : ℤ) (h : -x + 7 = 2) : x = 5 := by
  sorry

end initial_yards_lost_l145_145717


namespace combined_age_l145_145888

theorem combined_age (H : ℕ) (Ryanne : ℕ) (Jamison : ℕ) 
  (h1 : Ryanne = H + 7) 
  (h2 : H + Ryanne = 15) 
  (h3 : Jamison = 2 * H) : 
  H + Ryanne + Jamison = 23 := 
by 
  sorry

end combined_age_l145_145888


namespace triangle_perimeter_l145_145364

theorem triangle_perimeter (x : ℕ) (a b c : ℕ) 
  (h1 : a = 3 * x) (h2 : b = 4 * x) (h3 : c = 5 * x)  
  (h4 : c - a = 6) : a + b + c = 36 := 
by
  sorry

end triangle_perimeter_l145_145364


namespace find_richards_score_l145_145660

variable (R B : ℕ)

theorem find_richards_score (h1 : B = R - 14) (h2 : B = 48) : R = 62 := by
  sorry

end find_richards_score_l145_145660


namespace problem1_problem2_l145_145974

-- Problem (1) proof statement
theorem problem1 (a : ℝ) (h : a ≠ 0) : 
  3 * a^2 * a^3 + a^7 / a^2 = 4 * a^5 :=
by
  sorry

-- Problem (2) proof statement
theorem problem2 (x : ℝ) : 
  (x - 1)^2 - x * (x + 1) + (-2023)^0 = -3 * x + 2 :=
by
  sorry

end problem1_problem2_l145_145974


namespace necessary_but_not_sufficient_condition_l145_145710

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a < 2) → (∃ x : ℂ, x^2 + (a : ℂ) * x + 1 = 0 ∧ x.im ≠ 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l145_145710


namespace division_remainder_3012_97_l145_145380

theorem division_remainder_3012_97 : 3012 % 97 = 5 := 
by 
  sorry

end division_remainder_3012_97_l145_145380


namespace gcd_sub_12_eq_36_l145_145295

theorem gcd_sub_12_eq_36 :
  Nat.gcd 7344 48 - 12 = 36 := 
by 
  sorry

end gcd_sub_12_eq_36_l145_145295


namespace circle_equation_line_equation_l145_145045

theorem circle_equation (a b r x y : ℝ) (h1 : a + b = 2 * x + y)
  (h2 : (a, 2*a - 2) = ((1, 2) : ℝ × ℝ))
  (h3 : (a, 2*a - 2) = ((2, 1) : ℝ × ℝ)) :
  (x - 2) ^ 2 + (y - 2) ^ 2 = 1 := sorry

theorem line_equation (x y m : ℝ) (h1 : y + 3 = (x - (-3)) * ((-3) - 0) / (m - (-3)))
  (h2 : (x, y) = (m, 0) ∨ (x, y) = (m, 0))
  (h3 : (m = 1 ∨ m = - 3 / 4)) :
  (3 * x + 4 * y - 3 = 0) ∨ (4 * x + 3 * y + 3 = 0) := sorry

end circle_equation_line_equation_l145_145045


namespace fill_missing_digits_l145_145983

noncomputable def first_number (a : ℕ) : ℕ := a * 1000 + 2 * 100 + 5 * 10 + 7
noncomputable def second_number (b c : ℕ) : ℕ := 2 * 1000 + b * 100 + 9 * 10 + c

theorem fill_missing_digits (a b c : ℕ) : a = 1 ∧ b = 5 ∧ c = 6 → first_number a + second_number b c = 5842 :=
by
  intros
  sorry

end fill_missing_digits_l145_145983


namespace last_digit_of_power_of_two_l145_145081

theorem last_digit_of_power_of_two (n : ℕ) (h : n ≥ 2) : (2 ^ (2 ^ n) + 1) % 10 = 7 :=
sorry

end last_digit_of_power_of_two_l145_145081


namespace factorize_expression_l145_145754

theorem factorize_expression (a : ℝ) : 
  (2 * a + 1) * a - 4 * a - 2 = (2 * a + 1) * (a - 2) :=
by 
  -- proof is skipped with sorry
  sorry

end factorize_expression_l145_145754


namespace compare_powers_l145_145762

theorem compare_powers (a b c : ℝ) (h1 : a = 2^555) (h2 : b = 3^444) (h3 : c = 6^222) : a < c ∧ c < b :=
by
  sorry

end compare_powers_l145_145762


namespace travel_time_proportion_l145_145132

theorem travel_time_proportion (D V : ℝ) (hV_pos : V > 0) :
  let Time1 := D / (16 * V)
  let Time2 := 3 * D / (4 * V)
  let TimeTotal := Time1 + Time2
  (Time1 / TimeTotal) = 1 / 13 :=
by
  sorry

end travel_time_proportion_l145_145132


namespace alice_weight_l145_145670

theorem alice_weight (a c : ℝ) (h1 : a + c = 200) (h2 : a - c = a / 3) : a = 120 :=
by
  sorry

end alice_weight_l145_145670


namespace cubic_roots_real_parts_neg_l145_145658

variable {a0 a1 a2 a3 : ℝ}

theorem cubic_roots_real_parts_neg (h_same_signs : (a0 > 0 ∧ a1 > 0 ∧ a2 > 0 ∧ a3 > 0) ∨ (a0 < 0 ∧ a1 < 0 ∧ a2 < 0 ∧ a3 < 0)) 
  (h_root_condition : a1 * a2 - a0 * a3 > 0) : 
    ∀ (x : ℝ), (a0 * x^3 + a1 * x^2 + a2 * x + a3 = 0 → x < 0 ∨ (∃ (z : ℂ), z.re < 0 ∧ z.im ≠ 0 ∧ z^2 = x)) :=
sorry

end cubic_roots_real_parts_neg_l145_145658


namespace sale_price_is_91_percent_of_original_price_l145_145938

variable (x : ℝ)
variable (h_increase : ∀ p : ℝ, p * 1.4)
variable (h_sale : ∀ p : ℝ, p * 0.65)

/--The sale price of an item is 91% of the original price.-/
theorem sale_price_is_91_percent_of_original_price {x : ℝ} 
  (h_increase : ∀ p, p * 1.4 = 1.40 * p)
  (h_sale : ∀ p, p * 0.65 = 0.65 * p): 
  (0.65 * 1.40 * x = 0.91 * x) := 
by 
  sorry

end sale_price_is_91_percent_of_original_price_l145_145938


namespace hallie_reads_121_pages_on_fifth_day_l145_145616

-- Definitions for the given conditions.
def book_length : ℕ := 480
def pages_day_one : ℕ := 63
def pages_day_two : ℕ := 95 -- Rounded from 94.5
def pages_day_three : ℕ := 115
def pages_day_four : ℕ := 86 -- Rounded from 86.25

-- Total pages read from day one to day four
def pages_read_first_four_days : ℕ :=
  pages_day_one + pages_day_two + pages_day_three + pages_day_four

-- Conclusion: the number of pages read on the fifth day.
def pages_day_five : ℕ := book_length - pages_read_first_four_days

-- Proof statement: Hallie reads 121 pages on the fifth day.
theorem hallie_reads_121_pages_on_fifth_day :
  pages_day_five = 121 :=
by
  -- Proof omitted
  sorry

end hallie_reads_121_pages_on_fifth_day_l145_145616


namespace find_biology_marks_l145_145567

theorem find_biology_marks (english math physics chemistry : ℕ) (avg_marks : ℕ) (biology : ℕ)
  (h_english : english = 86) (h_math : math = 89) (h_physics : physics = 82)
  (h_chemistry : chemistry = 87) (h_avg_marks : avg_marks = 85) :
  (english + math + physics + chemistry + biology) = avg_marks * 5 →
  biology = 81 :=
by
  sorry

end find_biology_marks_l145_145567


namespace trigonometric_identity_l145_145946

noncomputable def sin110cos40_minus_cos70sin40 : ℝ := 
  Real.sin (110 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) - 
  Real.cos (70 * Real.pi / 180) * Real.sin (40 * Real.pi / 180)

theorem trigonometric_identity : 
  sin110cos40_minus_cos70sin40 = 1 / 2 := 
by sorry

end trigonometric_identity_l145_145946


namespace linear_function_passes_through_point_l145_145345

theorem linear_function_passes_through_point :
  ∀ x y : ℝ, y = -2 * x - 6 → (x = -4 → y = 2) :=
by
  sorry

end linear_function_passes_through_point_l145_145345


namespace percent_round_trip_tickets_is_100_l145_145072

noncomputable def percent_round_trip_tickets (P : ℕ) (x : ℚ) : ℚ :=
  let R := x / 0.20
  R

theorem percent_round_trip_tickets_is_100
  (P : ℕ)
  (x : ℚ)
  (h : 20 * x = P) :
  percent_round_trip_tickets P (x / P) = 100 :=
by
  sorry

end percent_round_trip_tickets_is_100_l145_145072


namespace problem_l145_145209

theorem problem (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 7 := 
sorry

end problem_l145_145209


namespace prime_n_if_power_of_prime_l145_145913

theorem prime_n_if_power_of_prime (n : ℕ) (h1 : n ≥ 2) (b : ℕ) (h2 : b > 0) (p : ℕ) (k : ℕ) 
  (hk : k > 0) (hb : (b^n - 1) / (b - 1) = p^k) : Nat.Prime n :=
sorry

end prime_n_if_power_of_prime_l145_145913


namespace tangent_line_perpendicular_l145_145600

noncomputable def f (x k : ℝ) : ℝ := x^3 - (k^2 - 1) * x^2 - k^2 + 2

theorem tangent_line_perpendicular (k : ℝ) (b : ℝ) (a : ℝ)
  (h1 : ∀ (x : ℝ), f x k = x^3 - (k^2 - 1) * x^2 - k^2 + 2)
  (h2 : (3 - 2 * (k^2 - 1)) = -1) :
  a = -2 := sorry

end tangent_line_perpendicular_l145_145600


namespace probability_top_card_10_or_face_l145_145145

theorem probability_top_card_10_or_face :
  let ranks := ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
  let suits := ["spades", "hearts", "diamonds", "clubs"]
  let deck := list.product suits ranks
  let face_cards := ["J", "Q", "K"]
  let num_10s := 4
  let num_face_cards := faces_cards.length * suits.length
  let favorable_outcomes := num_10s + num_face_cards
  let total_cards := deck.length
in (favorable_outcomes.to_rat / total_cards.to_rat) = (4 / 13 : ℚ) := by
  sorry

end probability_top_card_10_or_face_l145_145145


namespace find_y_l145_145984

theorem find_y (y : ℚ) (h : ⌊y⌋ + y = 5) : y = 7 / 3 :=
sorry

end find_y_l145_145984


namespace polynomial_expansion_correct_l145_145982

def polynomial1 (x : ℝ) := 3 * x^2 - 4 * x + 3
def polynomial2 (x : ℝ) := -2 * x^2 + 3 * x - 4

theorem polynomial_expansion_correct {x : ℝ} :
  (polynomial1 x) * (polynomial2 x) = -6 * x^4 + 17 * x^3 - 30 * x^2 + 25 * x - 12 :=
by
  sorry

end polynomial_expansion_correct_l145_145982


namespace sum_first_9_terms_l145_145932

-- Definitions of the arithmetic sequence and sum.
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + a m

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Conditions
def a_n (n : ℕ) : ℤ := sorry -- we assume this function gives the n-th term of the arithmetic sequence
def S_n (n : ℕ) : ℤ := sorry -- sum of first n terms
axiom a_5_eq_2 : a_n 5 = 2
axiom arithmetic_sequence_proof : arithmetic_sequence a_n
axiom sum_first_n_proof : sum_first_n a_n S_n

-- Statement to prove
theorem sum_first_9_terms : S_n 9 = 18 :=
by
  sorry

end sum_first_9_terms_l145_145932


namespace repetend_of_five_over_seventeen_l145_145017

theorem repetend_of_five_over_seventeen : 
  let r := 5 / 17 in
  ∃ a b : ℕ, a * 10^b = 294117 ∧ (r * 10^b - a) = (r * 10^6 - r * (10^6 / 17))
   ∧ (r * 10^k = (r * 10^6).floor / 10^k ) where k = 6 := sorry

end repetend_of_five_over_seventeen_l145_145017


namespace find_p_max_area_of_triangle_l145_145605

def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {xy | xy.1^2 = 2 * p * xy.2 ∧ p > 0}

def circle : set (ℝ × ℝ) :=
  {xy | xy.1^2 + (xy.2 + 4)^2 = 1}

def focus (p : ℝ) : ℝ × ℝ :=
  (0, p/2)

def min_distance (p : ℝ) : ℝ :=
  let f := focus p in
  let distance_fn := λ (x y : ℝ), real.sqrt ((x - f.1)^2 + (y - f.2)^2) in
  inf (distance_fn <$> (circle.image prod.fst) <*> (circle.image prod.snd))

theorem find_p: 
  ∃ p > 0, min_distance p = 4 :=
sorry

theorem max_area_of_triangle (P A B: ℝ × ℝ) (P_on_circle : P ∈ circle)
  (A_on_parabola : ∃ p > 0, A ∈ parabola p)
  (B_on_parabola : ∃ p > 0, B ∈ parabola p)
  (PA_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P A p)
  (PB_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P B p) :
  ∃ max_area, max_area = 20 * real.sqrt 5 :=
sorry

end find_p_max_area_of_triangle_l145_145605


namespace find_m_value_l145_145448

theorem find_m_value (m : ℝ) (h₀ : m > 0) (h₁ : (4 - m) / (m - 2) = m) : m = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end find_m_value_l145_145448


namespace max_gcd_2015xy_l145_145585

theorem max_gcd_2015xy (x y : ℤ) (coprime : Int.gcd x y = 1) :
    ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
sorry

end max_gcd_2015xy_l145_145585


namespace mitch_total_scoops_l145_145916

theorem mitch_total_scoops :
  (3 : ℝ) / (1/3 : ℝ) + (2 : ℝ) / (1/3 : ℝ) = 15 :=
by
  sorry

end mitch_total_scoops_l145_145916


namespace sequence_sum_l145_145741

noncomputable def r : ℝ := 1 / 4
def t₀ : ℝ := 4096
def t₁ : ℝ := t₀ * r
def t₂ : ℝ := t₁ * r
def x : ℝ := t₂ * r
def y : ℝ := x * r

theorem sequence_sum : x + y = 80 := by
  -- The proof will go here
  sorry

end sequence_sum_l145_145741


namespace total_hatched_eggs_l145_145461

noncomputable def fertile_eggs (total_eggs : ℕ) (infertility_rate : ℝ) : ℝ :=
  total_eggs * (1 - infertility_rate)

noncomputable def hatching_eggs_after_calcification (fertile_eggs : ℝ) (calcification_rate : ℝ) : ℝ :=
  fertile_eggs * (1 - calcification_rate)

noncomputable def hatching_eggs_after_predator (hatching_eggs : ℝ) (predator_rate : ℝ) : ℝ :=
  hatching_eggs * (1 - predator_rate)

noncomputable def hatching_eggs_after_temperature (hatching_eggs : ℝ) (temperature_rate : ℝ) : ℝ :=
  hatching_eggs * (1 - temperature_rate)

open Nat

theorem total_hatched_eggs :
  let g1_total_eggs := 30
  let g2_total_eggs := 40
  let g1_infertility_rate := 0.20
  let g2_infertility_rate := 0.25
  let g1_calcification_rate := 1.0 / 3.0
  let g2_calcification_rate := 0.25
  let predator_rate := 0.10
  let temperature_rate := 0.05
  let g1_fertile := fertile_eggs g1_total_eggs g1_infertility_rate
  let g1_hatch_calcification := hatching_eggs_after_calcification g1_fertile g1_calcification_rate
  let g1_hatch_predator := hatching_eggs_after_predator g1_hatch_calcification predator_rate
  let g1_hatch_temp := hatching_eggs_after_temperature g1_hatch_predator temperature_rate
  let g2_fertile := fertile_eggs g2_total_eggs g2_infertility_rate
  let g2_hatch_calcification := hatching_eggs_after_calcification g2_fertile g2_calcification_rate
  let g2_hatch_predator := hatching_eggs_after_predator g2_hatch_calcification predator_rate
  let g2_hatch_temp := hatching_eggs_after_temperature g2_hatch_predator temperature_rate
  let total_hatched := g1_hatch_temp + g2_hatch_temp
  floor total_hatched = 32 :=
by
  sorry

end total_hatched_eggs_l145_145461


namespace angle_PDO_45_degrees_l145_145565

-- Define the square configuration
variables (A B C D L P Q M N O : Type)
variables (a : ℝ) -- side length of the square ABCD

-- Conditions as hypothesized in the problem
def is_square (v₁ v₂ v₃ v₄ : Type) := true -- Placeholder for the square property
def on_diagonal_AC (L : Type) := true -- Placeholder for L being on diagonal AC
def common_vertex_L (sq1_v1 sq1_v2 sq1_v3 sq1_v4 sq2_v1 sq2_v2 sq2_v3 sq2_v4 : Type) := true -- Placeholder for common vertex L
def point_on_side (P AB_side: Type) := true -- Placeholder for P on side AB of ABCD
def square_center (center sq_v1 sq_v2 sq_v3 sq_v4 : Type) := true -- Placeholder for square's center

-- Prove the angle PDO is 45 degrees
theorem angle_PDO_45_degrees 
  (h₁ : is_square A B C D)
  (h₂ : on_diagonal_AC L)
  (h₃ : is_square A P L Q)
  (h₄ : is_square C M L N)
  (h₅ : common_vertex_L A P L Q C M L N)
  (h₆ : point_on_side P B)
  (h₇ : square_center O C M L N)
  : ∃ θ : ℝ, θ = 45 := 
  sorry

end angle_PDO_45_degrees_l145_145565


namespace factorize_expr_l145_145753

theorem factorize_expr (a x y : ℝ) : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) :=
by
  sorry

end factorize_expr_l145_145753


namespace shaded_area_of_square_l145_145978

theorem shaded_area_of_square (side_square : ℝ) (leg_triangle : ℝ) (h1 : side_square = 40) (h2 : leg_triangle = 25) :
  let area_square := side_square ^ 2
  let area_triangle := (1 / 2) * leg_triangle * leg_triangle
  let total_area_triangles := 2 * area_triangle
  let shaded_area := area_square - total_area_triangles
  shaded_area = 975 :=
by
  sorry

end shaded_area_of_square_l145_145978


namespace solve_for_x_l145_145511

theorem solve_for_x (x : ℝ) (h : (3 * x + 15)^2 = 3 * (4 * x + 40)) :
  x = -5 / 3 ∨ x = -7 :=
sorry

end solve_for_x_l145_145511


namespace complex_i_power_l145_145682

theorem complex_i_power (i : ℂ) (h1 : i^2 = -1) (h2 : i^3 = -i) (h3 : i^4 = 1) : i^2015 = -i := 
by
  sorry

end complex_i_power_l145_145682


namespace constant_term_exists_l145_145537

theorem constant_term_exists (n : ℕ) (h : n = 6) : 
  (∃ r : ℕ, 2 * n - 3 * r = 0) ∧ 
  (∃ n' r' : ℕ, n' ≠ 6 ∧ 2 * n' - 3 * r' = 0) := by
  sorry

end constant_term_exists_l145_145537


namespace gretchen_flavors_l145_145781

/-- 
Gretchen's local ice cream shop offers 100 different flavors. She tried a quarter of the flavors 2 years ago and double that amount last year. Prove how many more flavors she needs to try this year to have tried all 100 flavors.
-/
theorem gretchen_flavors (F T2 T1 T R : ℕ) (h1 : F = 100)
  (h2 : T2 = F / 4)
  (h3 : T1 = 2 * T2)
  (h4 : T = T2 + T1)
  (h5 : R = F - T) : R = 25 :=
sorry

end gretchen_flavors_l145_145781


namespace area_of_shaded_trapezoid_l145_145034

-- Definitions of conditions:
def side_lengths : List ℕ := [1, 3, 5, 7]
def total_base : ℕ := side_lengths.sum
def height_largest_square : ℕ := 7
def ratio : ℚ := height_largest_square / total_base

def height_at_end (n : ℕ) : ℚ := ratio * n
def lower_base_height : ℚ := height_at_end 4
def upper_base_height : ℚ := height_at_end 9
def trapezoid_height : ℕ := 2

-- Main theorem:
theorem area_of_shaded_trapezoid :
  (1 / 2) * (lower_base_height + upper_base_height) * trapezoid_height = 91 / 8 :=
by
  sorry

end area_of_shaded_trapezoid_l145_145034


namespace verify_statements_l145_145879

def line1 (a x y : ℝ) : Prop := a * x - (a + 2) * y - 2 = 0
def line2 (a x y : ℝ) : Prop := (a - 2) * x + 3 * a * y + 2 = 0

theorem verify_statements (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ x = -1 ∧ y = -1) ∧
  (∀ x y : ℝ, (line1 a x y ∧ line2 a x y) → (a = 0 ∨ a = -4)) :=
by sorry

end verify_statements_l145_145879


namespace missing_dog_number_l145_145855

theorem missing_dog_number {S : Finset ℕ} (h₁ : S =  Finset.range 25 \ {24}) (h₂ : S.sum id = 276) :
  (∃ y ∈ S, y = (S.sum id - y) / (S.card - 1)) ↔ 24 ∉ S :=
by
  sorry

end missing_dog_number_l145_145855


namespace like_terms_exponents_equal_l145_145455

theorem like_terms_exponents_equal (a b : ℤ) :
  (∀ x y : ℝ, 2 * x^a * y^2 = -3 * x^3 * y^(b+3) → a = 3 ∧ b = -1) :=
by
  sorry

end like_terms_exponents_equal_l145_145455


namespace geometric_progression_solution_l145_145936

-- Definitions and conditions as per the problem
def geometric_progression_first_term (b q : ℝ) : Prop :=
  b * (1 + q + q^2) = 21

def geometric_progression_sum_of_squares (b q : ℝ) : Prop :=
  b^2 * (1 + q^2 + q^4) = 189

-- The main theorem to be proven
theorem geometric_progression_solution (b q : ℝ) :
  (geometric_progression_first_term b q ∧ geometric_progression_sum_of_squares b q) →
  (b = 3 ∧ q = 2) ∨ (b = 12 ∧ q = 1 / 2) := 
by
  intros h
  sorry

end geometric_progression_solution_l145_145936


namespace kira_travel_time_l145_145114

theorem kira_travel_time :
  let time_between_stations := 2 * 60 -- converting hours to minutes
  let break_time := 30 -- in minutes
  let total_time := 2 * time_between_stations + break_time
  total_time = 270 :=
by
  let time_between_stations := 2 * 60
  let break_time := 30
  let total_time := 2 * time_between_stations + break_time
  exact rfl

end kira_travel_time_l145_145114


namespace Wendy_age_l145_145375

theorem Wendy_age
  (years_as_accountant : ℕ)
  (years_as_manager : ℕ)
  (percent_accounting_related : ℝ)
  (total_accounting_related : ℕ)
  (total_lifespan : ℝ) :
  years_as_accountant = 25 →
  years_as_manager = 15 →
  percent_accounting_related = 0.50 →
  total_accounting_related = years_as_accountant + years_as_manager →
  (total_accounting_related : ℝ) = percent_accounting_related * total_lifespan →
  total_lifespan = 80 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Wendy_age_l145_145375


namespace cost_price_of_a_ball_l145_145654

variables (C : ℝ) (selling_price : ℝ) (cost_price_20_balls : ℝ) (loss_on_20_balls : ℝ)

def cost_price_per_ball (C : ℝ) := (20 * C - 720 = 5 * C)

theorem cost_price_of_a_ball :
  (∃ C : ℝ, 20 * C - 720 = 5 * C) -> (C = 48) := 
by
  sorry

end cost_price_of_a_ball_l145_145654


namespace Adam_bought_26_books_l145_145279

theorem Adam_bought_26_books (initial_books : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) (leftover_books : ℕ) :
  initial_books = 56 → shelves = 4 → books_per_shelf = 20 → leftover_books = 2 → 
  let total_capacity := shelves * books_per_shelf in
  let total_books_after := total_capacity + leftover_books in
  let books_bought := total_books_after - initial_books in
  books_bought = 26 :=
by
  intros h1 h2 h3 h4
  simp [total_capacity, total_books_after, books_bought]
  rw [h1, h2, h3, h4]
  sorry

end Adam_bought_26_books_l145_145279


namespace sticks_form_triangle_l145_145371

theorem sticks_form_triangle (a b c d e : ℝ) 
  (h1 : 2 < a) (h2 : a < 8)
  (h3 : 2 < b) (h4 : b < 8)
  (h5 : 2 < c) (h6 : c < 8)
  (h7 : 2 < d) (h8 : d < 8)
  (h9 : 2 < e) (h10 : e < 8) :
  ∃ x y z, 
    (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
    (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
    (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
    x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    x + y > z ∧ x + z > y ∧ y + z > x :=
by sorry

end sticks_form_triangle_l145_145371


namespace range_of_m_l145_145173

def p (m : ℝ) : Prop := m^2 - 4 > 0 ∧ m > 0
def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0
def condition1 (m : ℝ) : Prop := p m ∨ q m
def condition2 (m : ℝ) : Prop := ¬ (p m ∧ q m)

theorem range_of_m (m : ℝ) : condition1 m ∧ condition2 m → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  sorry

end range_of_m_l145_145173


namespace remainder_of_large_number_l145_145693

theorem remainder_of_large_number (n : ℕ) (r : ℕ) (h : n = 2468135792) :
  (n % 101) = 52 := 
by
  have h1 : (10 ^ 8 - 1) % 101 = 0 := sorry
  have h2 : (10 ^ 6 - 1) % 101 = 0 := sorry
  have h3 : (10 ^ 4 - 1) % 101 = 0 := sorry
  have h4 : (10 ^ 2 - 1) % 101 = 99 % 101 := sorry

  -- Using these properties to simplify n
  have n_decomposition : 2468135792 = 24 * 10 ^ 8 + 68 * 10 ^ 6 + 13 * 10 ^ 4 + 57 * 10 ^ 2 + 92 := sorry
  have div_property : 
    (24 * 10 ^ 8 + 68 * 10 ^ 6 + 13 * 10 ^ 4 + 57 * 10 ^ 2 + 92 - (24 + 68 + 13 + 57 + 92)) % 101 = 0 := sorry

  have simplified_sum : (24 + 68 + 13 + 57 + 92 = 254 := by norm_num) := sorry
  have resulting_mod : 254 % 101 = 52 := by norm_num

  -- Thus n % 101 = 52
  exact resulting_mod

end remainder_of_large_number_l145_145693


namespace matching_function_l145_145790

open Real

def table_data : List (ℝ × ℝ) := [(1, 4), (2, 2), (4, 1)]

theorem matching_function :
  ∃ a b c : ℝ, a > 0 ∧ 
               (∀ x y, (x, y) ∈ table_data → y = a * x^2 + b * x + c) := 
sorry

end matching_function_l145_145790


namespace school_students_count_l145_145900

def students_in_school (c n : ℕ) : ℕ := n * c

theorem school_students_count
  (c n : ℕ)
  (h1 : n * c = (n - 6) * (c + 5))
  (h2 : n * c = (n - 16) * (c + 20)) :
  students_in_school c n = 900 :=
by
  sorry

end school_students_count_l145_145900


namespace percentage_increase_l145_145275

theorem percentage_increase 
  (P : ℝ)
  (bought_price : ℝ := 0.80 * P) 
  (original_profit : ℝ := 0.3600000000000001 * P) :
  ∃ X : ℝ, X = 70.00000000000002 ∧ (1.3600000000000001 * P = bought_price * (1 + X / 100)) :=
sorry

end percentage_increase_l145_145275


namespace fenced_area_with_cutout_l145_145089

def rectangle_area (length width : ℝ) : ℝ := length * width

def square_area (side : ℝ) : ℝ := side * side

theorem fenced_area_with_cutout :
  rectangle_area 20 18 - square_area 4 = 344 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end fenced_area_with_cutout_l145_145089


namespace quadratic_inequality_l145_145299

theorem quadratic_inequality (m y1 y2 y3 : ℝ)
  (h1 : m < -2)
  (h2 : y1 = (m-1)^2 - 2*(m-1))
  (h3 : y2 = m^2 - 2*m)
  (h4 : y3 = (m+1)^2 - 2*(m+1)) :
  y3 < y2 ∧ y2 < y1 :=
by
  sorry

end quadratic_inequality_l145_145299


namespace voting_for_marty_l145_145464

/-- Conditions provided in the problem -/
def total_people : ℕ := 400
def percentage_biff : ℝ := 0.30
def percentage_clara : ℝ := 0.20
def percentage_doc : ℝ := 0.10
def percentage_ein : ℝ := 0.05
def percentage_undecided : ℝ := 0.15

/-- Statement to prove the number of people voting for Marty -/
theorem voting_for_marty : 
  (1 - percentage_biff - percentage_clara - percentage_doc - percentage_ein - percentage_undecided) * total_people = 80 :=
by
  sorry

end voting_for_marty_l145_145464


namespace pears_seed_avg_l145_145082

def apple_seed_avg : ℕ := 6
def grape_seed_avg : ℕ := 3
def total_seeds_required : ℕ := 60
def apples_count : ℕ := 4
def pears_count : ℕ := 3
def grapes_count : ℕ := 9
def seeds_short : ℕ := 3
def total_seeds_obtained : ℕ := total_seeds_required - seeds_short

theorem pears_seed_avg :
  (apples_count * apple_seed_avg) + (grapes_count * grape_seed_avg) + (pears_count * P) = total_seeds_obtained → 
  P = 2 :=
by
  sorry

end pears_seed_avg_l145_145082


namespace min_value_l145_145478

theorem min_value : ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (a = 1) → (b = 1) → (c = 1) →
  (∃ x, x = (a^2 + 4 * a + 2) / a ∧ x ≥ 6) ∧
  (∃ y, y = (b^2 + 4 * b + 2) / b ∧ y ≥ 6) ∧
  (∃ z, z = (c^2 + 4 * c + 2) / c ∧ z ≥ 6) →
  (∃ m, m = ((a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2)) / (a * b * c) ∧ m = 216) :=
by {
  sorry
}

end min_value_l145_145478


namespace stratified_sampling_number_l145_145143

noncomputable def students_in_grade_10 : ℕ := 150
noncomputable def students_in_grade_11 : ℕ := 180
noncomputable def students_in_grade_12 : ℕ := 210
noncomputable def total_students : ℕ := students_in_grade_10 + students_in_grade_11 + students_in_grade_12
noncomputable def sample_size : ℕ := 72
noncomputable def selection_probability : ℚ := sample_size / total_students
noncomputable def combined_students_grade_10_11 : ℕ := students_in_grade_10 + students_in_grade_11

theorem stratified_sampling_number :
  combined_students_grade_10_11 * selection_probability = 44 := 
by
  sorry

end stratified_sampling_number_l145_145143


namespace code_length_is_4_l145_145148

-- Definitions based on conditions provided
def code_length : ℕ := 4 -- Each code consists of 4 digits
def total_codes_with_leading_zeros : ℕ := 10^code_length -- Total possible codes allowing leading zeros
def total_codes_without_leading_zeros : ℕ := 9 * 10^(code_length - 1) -- Total possible codes disallowing leading zeros
def codes_lost_if_no_leading_zeros : ℕ := total_codes_with_leading_zeros - total_codes_without_leading_zeros -- Codes lost if leading zeros are disallowed
def manager_measured_codes_lost : ℕ := 10000 -- Manager's incorrect measurement

-- Theorem to be proved based on the problem
theorem code_length_is_4 : code_length = 4 :=
by
  sorry

end code_length_is_4_l145_145148


namespace lollipops_Lou_received_l145_145808

def initial_lollipops : ℕ := 42
def given_to_Emily : ℕ := 2 * initial_lollipops / 3
def kept_by_Marlon : ℕ := 4
def lollipops_left_after_Emily : ℕ := initial_lollipops - given_to_Emily
def lollipops_given_to_Lou : ℕ := lollipops_left_after_Emily - kept_by_Marlon

theorem lollipops_Lou_received : lollipops_given_to_Lou = 10 := by
  sorry

end lollipops_Lou_received_l145_145808


namespace standard_deviation_of_data_l145_145105

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  let m := mean data
  (data.map (fun x => (x - m)^2)).sum / data.length

noncomputable def std_dev (data : List ℝ) : ℝ :=
  Real.sqrt (variance data)

theorem standard_deviation_of_data :
  std_dev [5, 7, 7, 8, 10, 11] = 2 := 
sorry

end standard_deviation_of_data_l145_145105


namespace Lacy_correct_percentage_l145_145811

def problems_exam (y : ℕ) := 10 * y
def problems_section1 (y : ℕ) := 6 * y
def problems_section2 (y : ℕ) := 4 * y
def missed_section1 (y : ℕ) := 2 * y
def missed_section2 (y : ℕ) := y
def solved_section1 (y : ℕ) := problems_section1 y - missed_section1 y
def solved_section2 (y : ℕ) := problems_section2 y - missed_section2 y
def total_solved (y : ℕ) := solved_section1 y + solved_section2 y
def percent_correct (y : ℕ) := (total_solved y : ℚ) / (problems_exam y) * 100

theorem Lacy_correct_percentage (y : ℕ) : percent_correct y = 70 := by
  -- Proof would go here
  sorry

end Lacy_correct_percentage_l145_145811


namespace problem1_problem2_l145_145410

-- Problem 1
theorem problem1 : (1/4 / 1/5) - 1/4 = 1 := 
by 
  sorry

-- Problem 2
theorem problem2 : ∃ x : ℚ, x + 1/2 * x = 12/5 ∧ x = 4 :=
by
  sorry

end problem1_problem2_l145_145410


namespace scientific_notation_correct_l145_145653

theorem scientific_notation_correct :
  52000000 = 5.2 * 10^7 :=
sorry

end scientific_notation_correct_l145_145653


namespace housewife_spending_l145_145960

theorem housewife_spending (P R M : ℝ) (h1 : R = 65) (h2 : R = 0.75 * P) (h3 : M / R - M / P = 5) :
  M = 1300 :=
by
  -- Proof steps will be added here.
  sorry

end housewife_spending_l145_145960


namespace problem_statement_l145_145298

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem problem_statement :
  f (5 * Real.pi / 24) = Real.sqrt 2 ∧
  ∀ x, f x ≥ 1 ↔ ∃ k : ℤ, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 2 :=
by
  sorry

end problem_statement_l145_145298


namespace total_combinations_meals_l145_145234

-- Define the total number of menu items
def menu_items : ℕ := 12

-- Define the function for computing the number of combinations of meals ordered by three people
def combinations_of_meals (n : ℕ) : ℕ := n * n * n

-- Theorem stating the total number of different combinations of meals is 1728
theorem total_combinations_meals : combinations_of_meals menu_items = 1728 :=
by
  -- Placeholder for actual proof
  sorry

end total_combinations_meals_l145_145234


namespace part1_part2_l145_145040

-- Part (1)
theorem part1 (a : ℝ) (A B : Set ℝ) 
  (hA : A = { x : ℝ | x^2 - 3 * x + 2 = 0 }) 
  (hB : B = { x : ℝ | x^2 - a * x + a - 1 = 0 }) 
  (hUnion : A ∪ B = A) : 
  a = 2 ∨ a = 3 := 
sorry

-- Part (2)
theorem part2 (m : ℝ) (A C : Set ℝ) 
  (hA : A = { x : ℝ | x^2 - 3 * x + 2 = 0 }) 
  (hC : C = { x : ℝ | x^2 + 2 * (m + 1) * x + m^2 - 5 = 0 }) 
  (hInter : A ∩ C = C) : 
  m ∈ Set.Iic (-3) := 
sorry

end part1_part2_l145_145040


namespace angle_C_is_150_degrees_l145_145970

theorem angle_C_is_150_degrees
  (C D : ℝ)
  (h_supp : C + D = 180)
  (h_C_5D : C = 5 * D) :
  C = 150 :=
by
  sorry

end angle_C_is_150_degrees_l145_145970


namespace relationship_abc_l145_145765

theorem relationship_abc (a b c : ℕ) (ha : a = 2^555) (hb : b = 3^444) (hc : c = 6^222) : a < c ∧ c < b := by
  sorry

end relationship_abc_l145_145765


namespace radius_of_inscribed_circle_l145_145523

-- Define the triangle side lengths
def DE : ℝ := 8
def DF : ℝ := 8
def EF : ℝ := 10

-- Define the semiperimeter
def s : ℝ := (DE + DF + EF) / 2

-- Define the area using Heron's formula
def K : ℝ := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))

-- Define the radius of the inscribed circle
def r : ℝ := K / s

-- The statement to be proved
theorem radius_of_inscribed_circle : r = (5 * Real.sqrt 39) / 13 := by
  sorry

end radius_of_inscribed_circle_l145_145523


namespace greatest_four_digit_number_l145_145834

theorem greatest_four_digit_number (x : ℕ) :
  x ≡ 1 [MOD 7] ∧ x ≡ 5 [MOD 8] ∧ 1000 ≤ x ∧ x < 10000 → x = 9997 :=
by
  sorry

end greatest_four_digit_number_l145_145834


namespace sequence_sum_l145_145749

theorem sequence_sum :
  ∃ (r : ℝ) (x y : ℝ),
    (r = 1 / 4) ∧
    (x = 256 * r) ∧
    (y = x * r) ∧
    (x + y = 80) :=
by
  use 1 / 4, 64, 16
  split
  . exact rfl
  split
  . exact rfl
  split
  . exact rfl
  . norm_num


end sequence_sum_l145_145749


namespace read_time_proof_l145_145402

noncomputable def read_time_problem : Prop :=
  ∃ (x y : ℕ), 
    x > 0 ∧
    y = 480 / x ∧
    (y - 5) = 480 / (x + 16) ∧
    y = 15

theorem read_time_proof : read_time_problem := 
sorry

end read_time_proof_l145_145402


namespace triangle_problems_l145_145903

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {S : ℝ}

theorem triangle_problems
  (h1 : (2 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a = Real.sqrt 13)
  (h3 : b + c = 5) :
  (A = π / 3) ∧ (S = Real.sqrt 3) :=
by
  sorry

end triangle_problems_l145_145903


namespace minimum_value_is_correct_l145_145212

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  (x + 1/y) * (x + 1/y - 2024) + (y + 1/x) * (y + 1/x - 2024) + 2024

theorem minimum_value_is_correct (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (minimum_value x y) ≥ -2050208 := 
sorry

end minimum_value_is_correct_l145_145212


namespace min_value_inequality_l145_145646

theorem min_value_inequality (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 9)
  : (a^2 + b^2 + c^2)/(a + b + c) + (b^2 + c^2)/(b + c) + (c^2 + a^2)/(c + a) + (a^2 + b^2)/(a + b) ≥ 12 :=
by
  sorry

end min_value_inequality_l145_145646


namespace min_value_symmetry_l145_145995

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem min_value_symmetry (a b c : ℝ) (h_a : a > 0) (h_symm : ∀ x : ℝ, quadratic a b c (2 + x) = quadratic a b c (2 - x)) : 
  quadratic a b c 2 < quadratic a b c 1 ∧ quadratic a b c 1 < quadratic a b c 4 := 
sorry

end min_value_symmetry_l145_145995


namespace yanna_kept_36_apples_l145_145383

-- Define the initial number of apples Yanna has
def initial_apples : ℕ := 60

-- Define the number of apples given to Zenny
def apples_given_to_zenny : ℕ := 18

-- Define the number of apples given to Andrea
def apples_given_to_andrea : ℕ := 6

-- The proof statement that Yanna kept 36 apples
theorem yanna_kept_36_apples : initial_apples - apples_given_to_zenny - apples_given_to_andrea = 36 := by
  sorry

end yanna_kept_36_apples_l145_145383


namespace evaluate_expression_l145_145160

theorem evaluate_expression : (6^6) * (12^6) * (6^12) * (12^12) = 72^18 := 
by sorry

end evaluate_expression_l145_145160


namespace midpoint_sum_coordinates_l145_145929

theorem midpoint_sum_coordinates (x y : ℝ) 
  (midpoint_cond_x : (x + 10) / 2 = 4) 
  (midpoint_cond_y : (y + 4) / 2 = -8) : 
  x + y = -22 :=
by
  sorry

end midpoint_sum_coordinates_l145_145929


namespace domain_h_l145_145357

def domain_f : Set ℝ := Set.Icc (-12) 6
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-3*x)

theorem domain_h {f : ℝ → ℝ} (hf : ∀ x, x ∈ domain_f → f x ∈ Set.univ) {x : ℝ} :
  h f x ∈ Set.univ ↔ x ∈ Set.Icc (-2) 4 :=
by
  sorry

end domain_h_l145_145357


namespace ticket_cost_at_30_years_l145_145258

noncomputable def initial_cost : ℝ := 1000000
noncomputable def halving_period_years : ℕ := 10
noncomputable def halving_factor : ℝ := 0.5

def cost_after_n_years (initial_cost : ℝ) (halving_factor : ℝ) (years : ℕ) (period : ℕ) : ℝ :=
  initial_cost * halving_factor ^ (years / period)

theorem ticket_cost_at_30_years (initial_cost halving_factor : ℝ) (years period: ℕ) 
  (h_initial_cost : initial_cost = 1000000)
  (h_halving_factor : halving_factor = 0.5)
  (h_years : years = 30)
  (h_period : period = halving_period_years) : 
  cost_after_n_years initial_cost halving_factor years period = 125000 :=
by 
  sorry

end ticket_cost_at_30_years_l145_145258


namespace solve_system_equations_l145_145818

theorem solve_system_equations (x y z b : ℝ) :
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 + z^2 = b^2) → 
  b = 0 ∧ (∃ t, (x = 0 ∧ y = t ∧ z = -t) ∨ 
                (x = t ∧ y = 0 ∧ z = -t) ∨ 
                (x = -t ∧ y = t ∧ z = 0)) :=
by
  sorry -- Proof to be provided

end solve_system_equations_l145_145818


namespace inscribed_circle_radius_eq_l145_145526

noncomputable def radius_of_inscribed_circle (DE DF EF : ℝ) (h1 : DE = 8) (h2 : DF = 8) (h3 : EF = 10) : ℝ :=
  let s := (DE + DF + EF) / 2 in
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  area / s

theorem inscribed_circle_radius_eq :
  radius_of_inscribed_circle 8 8 10 8 8 10 = (15 * Real.sqrt 13) / 13 :=
  sorry

end inscribed_circle_radius_eq_l145_145526


namespace hyperbola_standard_equation_l145_145437

def a : ℕ := 5
def c : ℕ := 7
def b_squared : ℕ := c * c - a * a

theorem hyperbola_standard_equation (a_eq : a = 5) (c_eq : c = 7) :
    (b_squared = 24) →
    ( ∀ x y : ℝ, x^2 / (a^2 : ℝ) - y^2 / (b_squared : ℝ) = 1 ∨ 
                   y^2 / (a^2 : ℝ) - x^2 / (b_squared : ℝ) = 1) :=
by
  sorry

end hyperbola_standard_equation_l145_145437


namespace expression_simplification_l145_145539

theorem expression_simplification (a b : ℤ) : 
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by
  sorry

end expression_simplification_l145_145539


namespace oak_trees_initial_count_l145_145829

theorem oak_trees_initial_count (x : ℕ) (cut_down : ℕ) (remaining : ℕ) (h_cut : cut_down = 2) (h_remaining : remaining = 7)
  (h_equation : (x - cut_down) = remaining) : x = 9 := by
  -- We are given that cut_down = 2
  -- and remaining = 7
  -- and we need to show that the initial count x = 9
  sorry

end oak_trees_initial_count_l145_145829


namespace solve_system_of_inequalities_l145_145355

open Set

theorem solve_system_of_inequalities : ∀ x : ℕ, (2 * (x - 1) < x + 3) ∧ ((2 * x + 1) / 3 > x - 1) → x ∈ ({0, 1, 2, 3} : Set ℕ) :=
by
  intro x
  intro h
  sorry

end solve_system_of_inequalities_l145_145355


namespace marble_ratio_is_two_to_one_l145_145157

-- Conditions
def dan_blue_marbles : ℕ := 5
def mary_blue_marbles : ℕ := 10

-- Ratio definition
def marble_ratio : ℚ := mary_blue_marbles / dan_blue_marbles

-- Theorem statement
theorem marble_ratio_is_two_to_one : marble_ratio = 2 :=
by 
  -- Prove the statement here
  sorry

end marble_ratio_is_two_to_one_l145_145157


namespace problem1_solution_problem2_solution_l145_145233

-- Problem 1: Prove the solution set for the given inequality
theorem problem1_solution (x : ℝ) : (2 < x ∧ x ≤ (7 / 2)) ↔ ((x + 1) / (x - 2) ≥ 3) := 
sorry

-- Problem 2: Prove the solution set for the given inequality
theorem problem2_solution (x a : ℝ) : 
  (a = 0 ∧ x = 0) ∨ 
  (a > 0 ∧ -a ≤ x ∧ x ≤ 2 * a) ∨ 
  (a < 0 ∧ 2 * a ≤ x ∧ x ≤ -a) ↔ 
  x^2 - a * x - 2 * a^2 ≤ 0 := 
sorry

end problem1_solution_problem2_solution_l145_145233


namespace cylinder_volume_in_sphere_l145_145430

theorem cylinder_volume_in_sphere 
  (h_c : ℝ) (d_s : ℝ) : 
  (h_c = 1) → (d_s = 2) → 
  (π * (d_s / 2)^2 * (h_c / 2) = π / 2) :=
by 
  intros h_c_eq h_s_eq
  sorry

end cylinder_volume_in_sphere_l145_145430


namespace problem_statement_l145_145568

def S (a b : ℤ) : ℤ := 4 * a + 6 * b
def T (a b : ℤ) : ℤ := 2 * a - 3 * b

theorem problem_statement : T (S 8 3) 4 = 88 := by
  sorry

end problem_statement_l145_145568


namespace time_difference_correct_l145_145807

-- Definitions based on conditions
def malcolm_speed : ℝ := 5 -- Malcolm's speed in minutes per mile
def joshua_speed : ℝ := 7 -- Joshua's speed in minutes per mile
def race_length : ℝ := 12 -- Length of the race in miles

-- Calculate times based on speeds and race length
def malcolm_time : ℝ := malcolm_speed * race_length
def joshua_time : ℝ := joshua_speed * race_length

-- The statement that the difference in finish times is 24 minutes
theorem time_difference_correct : joshua_time - malcolm_time = 24 :=
by
  -- Proof goes here
  sorry

end time_difference_correct_l145_145807


namespace parabola_circle_distance_l145_145603

theorem parabola_circle_distance (p : ℝ) (hp : 0 < p) :
  let F := (0, p / 2)
  let M := EuclideanDistance
  let dist_F_M := ∀ (Q : ℝ × ℝ), Q ∈ {P : ℝ × ℝ | P.1^2 + (P.2 + 4)^2 = 1 } → ((0 - Q.1)^2 + (p / 2 - Q.2)^2).sqrt - 1 = 4 → p = 2
:= sorry

end parabola_circle_distance_l145_145603


namespace two_bishops_placement_l145_145628

theorem two_bishops_placement :
  let squares := 64
  let white_squares := 32
  let black_squares := 32
  let first_bishop_white_positions := 32
  let second_bishop_black_positions := 32 - 8
  first_bishop_white_positions * second_bishop_black_positions = 768 := by
  sorry

end two_bishops_placement_l145_145628


namespace floors_above_l145_145415

theorem floors_above (dennis_floor charlie_floor frank_floor : ℕ)
  (h1 : dennis_floor = 6)
  (h2 : frank_floor = 16)
  (h3 : charlie_floor = frank_floor / 4) :
  dennis_floor - charlie_floor = 2 :=
by
  sorry

end floors_above_l145_145415


namespace total_games_attended_l145_145637

theorem total_games_attended 
  (games_this_month : ℕ)
  (games_last_month : ℕ)
  (games_next_month : ℕ)
  (total_games : ℕ) 
  (h : games_this_month = 11)
  (h2 : games_last_month = 17)
  (h3 : games_next_month = 16) 
  (htotal : total_games = 44) :
  games_this_month + games_last_month + games_next_month = total_games :=
by sorry

end total_games_attended_l145_145637


namespace boatworks_total_canoes_l145_145151

theorem boatworks_total_canoes : 
  let jan := 5
  let feb := 3 * jan
  let mar := 3 * feb
  let apr := 3 * mar
  jan + feb + mar + apr = 200 := 
by 
  sorry

end boatworks_total_canoes_l145_145151


namespace arithmetic_base_conversion_l145_145153

-- We start with proving base conversions

def convert_base3_to_base10 (n : ℕ) : ℕ := 1 * (3^0) + 2 * (3^1) + 1 * (3^2)

def convert_base7_to_base10 (n : ℕ) : ℕ := 6 * (7^0) + 5 * (7^1) + 4 * (7^2) + 3 * (7^3)

def convert_base9_to_base10 (n : ℕ) : ℕ := 6 * (9^0) + 7 * (9^1) + 8 * (9^2) + 9 * (9^3)

-- Prove the main equality

theorem arithmetic_base_conversion:
  (2468 : ℝ) / convert_base3_to_base10 121 + convert_base7_to_base10 3456 - convert_base9_to_base10 9876 = -5857.75 :=
by
  have h₁ : convert_base3_to_base10 121 = 16 := by native_decide
  have h₂ : convert_base7_to_base10 3456 = 1266 := by native_decide
  have h₃ : convert_base9_to_base10 9876 = 7278 := by native_decide
  rw [h₁, h₂, h₃]
  sorry

end arithmetic_base_conversion_l145_145153


namespace no_two_digit_numbers_satisfy_condition_l145_145553

theorem no_two_digit_numbers_satisfy_condition :
  ¬ ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
  (10 * a + b) * (10 * c + d) = 1000 * a + 100 * b + 10 * c + d :=
by
  sorry

end no_two_digit_numbers_satisfy_condition_l145_145553


namespace angle_sum_equal_l145_145131

theorem angle_sum_equal 
  (AB AC DE DF : ℝ)
  (h_AB_AC : AB = AC)
  (h_DE_DF : DE = DF)
  (angle_BAC angle_EDF : ℝ)
  (h_angle_BAC : angle_BAC = 40)
  (h_angle_EDF : angle_EDF = 50)
  (angle_DAC angle_ADE : ℝ)
  (h_angle_DAC : angle_DAC = 70)
  (h_angle_ADE : angle_ADE = 65) :
  angle_DAC + angle_ADE = 135 := 
sorry

end angle_sum_equal_l145_145131


namespace grooming_time_l145_145632

theorem grooming_time (time_per_dog : ℕ) (num_dogs : ℕ) (days : ℕ) (minutes_per_hour : ℕ) :
  time_per_dog = 20 →
  num_dogs = 2 →
  days = 30 →
  minutes_per_hour = 60 →
  (time_per_dog * num_dogs * days) / minutes_per_hour = 20 := 
by
  intros
  exact sorry

end grooming_time_l145_145632


namespace part1_part2_1_part2_2_l145_145431

noncomputable def f (m : ℝ) (a x : ℝ) : ℝ :=
  m / x + Real.log (x / a)

-- Part (1)
theorem part1 (m a : ℝ) (h : m > 0) (ha : a > 0) (hmin : ∀ x, f m a x ≥ 2) : 
  m / a = Real.exp 1 :=
sorry

-- Part (2.1)
theorem part2_1 (a x₀ : ℝ) (ha : a > Real.exp 1) (hx₀ : x₀ > 1) (hzero : f 1 a x₀ = 0) : 
  1 / (2 * x₀) + x₀ < a - 1 :=
sorry

-- Part (2.2)
theorem part2_2 (a x₀ : ℝ) (ha : a > Real.exp 1) (hx₀ : x₀ > 1) (hzero : f 1 a x₀ = 0) : 
  x₀ + 1 / x₀ > 2 * Real.log a - Real.log (Real.log a) :=
sorry

end part1_part2_1_part2_2_l145_145431


namespace remaining_food_can_cater_children_l145_145138

theorem remaining_food_can_cater_children (A C : ℝ) 
  (h_food_adults : 70 * A = 90 * C) 
  (h_35_adults_ate : ∀ n: ℝ, (n = 35) → 35 * A = 35 * (9/7) * C) : 
  70 * A - 35 * A = 45 * C :=
by
  sorry

end remaining_food_can_cater_children_l145_145138


namespace parabola_focus_distance_max_area_triangle_l145_145608

-- Part 1: Prove that p = 2
theorem parabola_focus_distance (p : ℝ) (h₀ : 0 < p) 
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> dist (0, p/2) (x, y) ≥ 4 ) : 
  p = 2 := 
by {
  sorry -- Proof will be filled in
}

-- Part 2: Prove the maximum area of ∆ PAB is 20√5
theorem max_area_triangle (p : ℝ)
  (h₀ : p = 2)
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> 
           ∃ A B : ℝ × ℝ, A ∈ parabola_tangents C P ∧ B ∈ parabola_tangents C P ∧
                       ∃ P : ℝ × ℝ, point_on_circle M P)
  : ∃ P A B : ℝ × ℝ, area_of_triangle P A B = 20 * real.sqrt 5 :=
by {
  sorry
}

end parabola_focus_distance_max_area_triangle_l145_145608


namespace cafe_location_l145_145343

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

end cafe_location_l145_145343


namespace frosting_cupcakes_l145_145409

noncomputable def Cagney_rate := 1 / 20 -- cupcakes per second
noncomputable def Lacey_rate := 1 / 30 -- cupcakes per second
noncomputable def Hardy_rate := 1 / 40 -- cupcakes per second

noncomputable def combined_rate := Cagney_rate + Lacey_rate + Hardy_rate
noncomputable def total_time := 600 -- seconds (10 minutes)

theorem frosting_cupcakes :
  total_time * combined_rate = 65 := 
by 
  sorry

end frosting_cupcakes_l145_145409


namespace max_value_expression_l145_145735

theorem max_value_expression : ∃ s_max : ℝ, 
  (∀ s : ℝ, -3 * s^2 + 24 * s - 7 ≤ -3 * s_max^2 + 24 * s_max - 7) ∧
  (-3 * s_max^2 + 24 * s_max - 7 = 41) :=
sorry

end max_value_expression_l145_145735


namespace part1_part2_l145_145428

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - 3

theorem part1 {a : ℝ} :
  (∀ x > 0, 2 * f x ≥ g x a) → a ≤ 4 :=
sorry

theorem part2 :
  ∀ x > 0, Real.log x > (1 / Real.exp x) - (2 / (Real.exp 1) * x) :=
sorry

end part1_part2_l145_145428


namespace dot_product_is_4_l145_145615

-- Define the vectors a and b
def a (k : ℝ) : ℝ × ℝ := (1, k)
def b : ℝ × ℝ := (2, 2)

-- Define collinearity condition
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Define k based on the collinearity condition
def k_value : ℝ := 1 -- derived from solving the collinearity condition in the problem

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove that the dot product of a and b is 4 when k = 1
theorem dot_product_is_4 {k : ℝ} (h : k = k_value) : dot_product (a k) b = 4 :=
by
  rw [h]
  sorry

end dot_product_is_4_l145_145615


namespace hollis_student_loan_l145_145784

theorem hollis_student_loan
  (interest_loan1 : ℝ)
  (interest_loan2 : ℝ)
  (total_loan1 : ℝ)
  (total_loan2 : ℝ)
  (additional_amount : ℝ)
  (total_interest_paid : ℝ) :
  interest_loan1 = 0.07 →
  total_loan1 = total_loan2 + additional_amount →
  additional_amount = 1500 →
  total_interest_paid = 617 →
  total_loan2 = 4700 →
  total_loan1 * interest_loan1 + total_loan2 * interest_loan2 = total_interest_paid →
  total_loan2 = 4700 :=
by
  sorry

end hollis_student_loan_l145_145784


namespace median_of_data_set_l145_145759

def data_set := [2, 3, 3, 4, 6, 6, 8, 8]

def calculate_50th_percentile (l : List ℕ) : ℕ :=
  if H : l.length % 2 = 0 then
    (l.get ⟨l.length / 2 - 1, sorry⟩ + l.get ⟨l.length / 2, sorry⟩) / 2
  else
    l.get ⟨l.length / 2, sorry⟩

theorem median_of_data_set : calculate_50th_percentile data_set = 5 :=
by
  -- Insert the proof here
  sorry

end median_of_data_set_l145_145759


namespace repetend_of_5_div_17_l145_145012

theorem repetend_of_5_div_17 :
  let dec := 5 / 17 in
  decimal_repetend dec = "294117" := sorry

end repetend_of_5_div_17_l145_145012


namespace find_x_eq_eight_l145_145445

theorem find_x_eq_eight (x : ℕ) : 3^(x-2) = 9^3 → x = 8 := 
by
  sorry

end find_x_eq_eight_l145_145445


namespace ratio_sum_div_c_l145_145618

theorem ratio_sum_div_c (a b c : ℚ) (h : a / 3 = b / 4 ∧ b / 4 = c / 5) : (a + b + c) / c = 12 / 5 :=
by
  sorry

end ratio_sum_div_c_l145_145618


namespace solve_for_x_l145_145815

theorem solve_for_x (x : ℝ) (h : (3 + 2 / x)^(1 / 3) = 2) : x = 2 / 5 :=
by
  sorry

end solve_for_x_l145_145815


namespace area_covered_by_both_strips_is_correct_l145_145142

-- Definitions of lengths of the strips and areas
def length_total : ℝ := 16
def length_left : ℝ := 9
def length_right : ℝ := 7
def area_left_only : ℝ := 27
def area_right_only : ℝ := 18

noncomputable def width_strip : ℝ := sorry -- The width can be inferred from solution but is not the focus of the proof.

-- Definition of the area covered by both strips
def S : ℝ := 13.5

-- Proof statement
theorem area_covered_by_both_strips_is_correct :
  ∀ w : ℝ,
    length_left * w - S = area_left_only ∧ length_right * w - S = area_right_only →
    S = 13.5 := 
by
  sorry

end area_covered_by_both_strips_is_correct_l145_145142


namespace average_speed_30_l145_145541

theorem average_speed_30 (v : ℝ) (h₁ : 0 < v) (h₂ : 210 / v - 1 = 210 / (v + 5)) : v = 30 :=
sorry

end average_speed_30_l145_145541


namespace natasha_quarters_l145_145490

theorem natasha_quarters :
  ∃ n : ℕ, (4 < n) ∧ (n < 40) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) ∧ (n = 2) := sorry

end natasha_quarters_l145_145490


namespace negation_of_even_sum_l145_145391

variables (a b : Int)

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem negation_of_even_sum (h : ¬(is_even a ∧ is_even b)) : ¬is_even (a + b) :=
sorry

end negation_of_even_sum_l145_145391


namespace shaded_percentage_seven_by_seven_grid_l145_145530

theorem shaded_percentage_seven_by_seven_grid :
  let total_squares := 49
  let shaded_squares := 7
  let shaded_fraction := shaded_squares / total_squares
  let shaded_percentage := shaded_fraction * 100
  shaded_percentage = 14.29 := by
  sorry

end shaded_percentage_seven_by_seven_grid_l145_145530


namespace quadratic_solution_m_l145_145449

theorem quadratic_solution_m (m : ℝ) : (x = 2) → (x^2 - m*x + 8 = 0) → (m = 6) := 
by
  sorry

end quadratic_solution_m_l145_145449


namespace point_on_parabola_touching_x_axis_l145_145928

theorem point_on_parabola_touching_x_axis (a b c : ℤ) (h : ∃ r : ℤ, a * (r * r) + b * r + c = 0 ∧ (r * r) = 0) :
  ∃ (a' b' : ℤ), ∃ k : ℤ, (k * k) + a' * k + b' = 0 ∧ (k * k) = 0 :=
sorry

end point_on_parabola_touching_x_axis_l145_145928


namespace find_number_l145_145424

theorem find_number (x : ℚ) (h : 15 + 3 * x = 6 * x - 10) : x = 25 / 3 :=
by
  sorry

end find_number_l145_145424


namespace trapezoid_shorter_base_length_l145_145675

theorem trapezoid_shorter_base_length (longer_base : ℕ) (segment_length : ℕ) (shorter_base : ℕ) 
  (h1 : longer_base = 120) (h2 : segment_length = 7)
  (h3 : segment_length = (longer_base - shorter_base) / 2) : 
  shorter_base = 106 := by
  sorry

end trapezoid_shorter_base_length_l145_145675


namespace b_completes_work_alone_l145_145385

theorem b_completes_work_alone (A_twice_B : ∀ (B : ℕ), A = 2 * B)
  (together : ℕ := 7) : ∃ (B : ℕ), 21 = 3 * together :=
by
  sorry

end b_completes_work_alone_l145_145385


namespace mary_average_speed_l145_145650

noncomputable def average_speed (d1 d2 : ℝ) (t1 t2 : ℝ) : ℝ :=
  (d1 + d2) / ((t1 + t2) / 60)

theorem mary_average_speed :
  average_speed 1.5 1.5 45 15 = 3 := by
  sorry

end mary_average_speed_l145_145650


namespace length_AF_is_25_l145_145803

open Classical

noncomputable def length_AF : ℕ :=
  let AB := 5
  let AC := 11
  let DE := 8
  let EF := 4
  let BC := AC - AB
  let CD := BC / 3
  let AF := AB + BC + CD + DE + EF
  AF

theorem length_AF_is_25 :
  length_AF = 25 := by
  sorry

end length_AF_is_25_l145_145803


namespace part_a_part_b_l145_145713

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def valid_permutation (P : Fin 16 → ℕ) : Prop :=
  (∀ i : Fin 15, is_perfect_square (P i + P (i + 1))) ∧
  ∀ i, P i ∈ (Finset.range 16).image (λ x => x + 1)

def valid_cyclic_permutation (C : Fin 16 → ℕ) : Prop :=
  (∀ i : Fin 15, is_perfect_square (C i + C (i + 1))) ∧
  is_perfect_square (C 15 + C 0) ∧
  ∀ i, C i ∈ (Finset.range 16).image (λ x => x + 1)

theorem part_a :
  ∃ P : Fin 16 → ℕ, valid_permutation P := sorry

theorem part_b :
  ¬ ∃ C : Fin 16 → ℕ, valid_cyclic_permutation C := sorry

end part_a_part_b_l145_145713


namespace ball_arrangement_l145_145370

theorem ball_arrangement : ∃ (n : ℕ), n = 120 ∧
  (∀ (ball_count : ℕ), ball_count = 20 → ∃ (box1 box2 box3 : ℕ), 
    box1 ≥ 1 ∧ box2 ≥ 2 ∧ box3 ≥ 3 ∧ box1 + box2 + box3 = ball_count) :=
by
  sorry

end ball_arrangement_l145_145370


namespace sequence_x_y_sum_l145_145747

theorem sequence_x_y_sum (r : ℝ) (x y : ℝ)
  (h₁ : r = 1 / 4)
  (h₂ : x = 256 * r)
  (h₃ : y = x * r) :
  x + y = 80 :=
by
  sorry

end sequence_x_y_sum_l145_145747


namespace distance_upstream_l145_145140

/-- Proof that the distance a man swims upstream is 18 km given certain conditions. -/
theorem distance_upstream (c : ℝ) (h1 : 54 / (12 + c) = 3) (h2 : 12 - c = 6) : (12 - c) * 3 = 18 :=
by
  sorry

end distance_upstream_l145_145140


namespace bus_students_remain_l145_145787

theorem bus_students_remain (init_students : ℕ) 
  (third_got_off : ℕ → ℕ) 
  (first_stop_second_third_fourth : ℕ ≠ 0 ∧ init_students = 64 ∧ 
   ∀ s, third_got_off s = (s * 2) / 3 ∧ 
   third_got_off (init_students * 2 / 3) = ((init_students * 2 / 3) * 2) / 3 ∧ 
   third_got_off ((init_students * 2 / 3) * 2 / 3) = (((init_students * 2 / 3) * 2 / 3) * 2) / 3 ∧ 
   third_got_off ((((init_students * 2 / 3) * 2 / 3) * 2 / 3) * 2) / 3) : 
  (((((init_students * 2) / 3) * 2) / 3) * 2 / 3 * 2) / 3 * 2 = 1024 / 81 :=
by sorry

end bus_students_remain_l145_145787


namespace ellipse_standard_equation_chord_length_range_l145_145307

-- Conditions for question 1
def ellipse_center (O : ℝ × ℝ) : Prop := O = (0, 0)
def major_axis_x (major_axis : ℝ) : Prop := major_axis = 1
def eccentricity (e : ℝ) : Prop := e = (Real.sqrt 2) / 2
def perp_chord_length (AA' : ℝ) : Prop := AA' = Real.sqrt 2

-- Lean statement for question 1
theorem ellipse_standard_equation (O : ℝ × ℝ) (major_axis : ℝ) (e : ℝ) (AA' : ℝ) :
  ellipse_center O → major_axis_x major_axis → eccentricity e → perp_chord_length AA' →
  ∃ (a b : ℝ), a = Real.sqrt 2 ∧ b = 1 ∧ (∀ x y : ℝ, (x^2 / (a^2)) + y^2 / (b^2) = 1) := sorry

-- Conditions for question 2
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 2 + y^2 = 1
def max_area_triangle (S : ℝ) : Prop := S = 1 / 2

-- Lean statement for question 2
theorem chord_length_range (x y z w : ℝ) (E F G H : ℝ × ℝ) :
  circle_eq x y → ellipse_eq z w → max_area_triangle ((E.1 * F.1) * (Real.sin (E.2 * F.2))) →
  ( ∃ min_chord max_chord : ℝ, min_chord = Real.sqrt 3 ∧ max_chord = 2 ∧
    ∀ x1 y1 x2 y2 : ℝ, (G.1 = x1 ∧ H.1 = x2 ∧ G.2 = y1 ∧ H.2 = y2) →
    (min_chord ≤ (Real.sqrt ((1 + (x2 ^ 2)) * ((x1 ^ 2) - 4 * (x1 * x2)))) ∧
         Real.sqrt ((1 + (x2 ^ 2)) * ((x1 ^ 2) - 4 * (x1 * x2))) ≤ max_chord )) := sorry

end ellipse_standard_equation_chord_length_range_l145_145307


namespace bus_routes_setup_possible_l145_145064

noncomputable section

open Finset

-- Definitions based on conditions
def is_valid_configuration (lines : Finset ℤ) (intersections : Finset (ℤ × ℤ)) : Prop :=
  lines.card = 10 ∧ 
  intersections.card = 45 ∧ 
  ∀ (chosen8 : Finset ℤ), chosen8.card = 8 → ∃ (stop : ℤ × ℤ), stop ∉ intersections.filter (λ p, p.1 ∈ chosen8 ∧ p.2 ∈ chosen8) ∧
  ∀ (chosen9 : Finset ℤ), chosen9.card = 9 → 
  intersections ⊆ intersections.filter (λ p, p.1 ∈ chosen9 ∧ p.2 ∈ chosen9)

-- The final proof statement
theorem bus_routes_setup_possible : ∃ (lines : Finset ℤ) (intersections : Finset (ℤ × ℤ)), is_valid_configuration lines intersections :=
sorry

end bus_routes_setup_possible_l145_145064


namespace hall_length_l145_145825

variable (breadth length : ℝ)

def condition1 : Prop := length = breadth + 5
def condition2 : Prop := length * breadth = 750

theorem hall_length : condition1 breadth length ∧ condition2 breadth length → length = 30 :=
by
  intros
  sorry

end hall_length_l145_145825


namespace infinite_set_contains_all_positive_integers_l145_145967

open Set

theorem infinite_set_contains_all_positive_integers {B : Set ℕ} (h₁ : Infinite B)
  (h₂ : ∀ a b ∈ B, a > b → (a - b) / Nat.gcd a b ∈ B) :
  ∀ n : ℕ, n > 0 → n ∈ B :=
by
  intro n hn
  sorry

end infinite_set_contains_all_positive_integers_l145_145967


namespace smallest_angle_proof_l145_145792

noncomputable def smallest_angle_in_triangle : ℝ :=
  let angle1 := 135
  let angle_supplementary := 180 - angle1
  let triangle_angles := (60, angle_supplementary)
  180 - (triangle_angles.1 + triangle_angles.2)

theorem smallest_angle_proof : smallest_angle_in_triangle = 45 := 
by 
  let angle1 := 135
  let angle_supplementary := 180 - angle1
  let triangle_angles := (60, angle_supplementary)
  let x := 180 - (triangle_angles.1 + triangle_angles.2)
  have conclusion : x = 45 := by
    simp [triangle_angles]
    calc
            180 - (60 + 45) = 180 - 105 : by simp
            ... = 75 : by simp
  sorry

end smallest_angle_proof_l145_145792


namespace problem_statement_l145_145666

noncomputable def a : ℝ := Real.sqrt 3 - Real.sqrt 11
noncomputable def b : ℝ := Real.sqrt 3 + Real.sqrt 11

theorem problem_statement : (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = Real.sqrt 3 / 3 :=
by
  -- conditions
  let a := Real.sqrt 3 - Real.sqrt 11
  let b := Real.sqrt 3 + Real.sqrt 11
  have h1 : a = Real.sqrt 3 - Real.sqrt 11 := rfl
  have h2 : b = Real.sqrt 3 + Real.sqrt 11 := rfl
  -- question statement
  sorry

end problem_statement_l145_145666


namespace repetend_of_five_seventeenths_l145_145013

theorem repetend_of_five_seventeenths :
  (decimal_expansion (5 / 17)).repeat_called == "294117647" :=
sorry

end repetend_of_five_seventeenths_l145_145013


namespace sum_of_remainders_mod_15_l145_145531

theorem sum_of_remainders_mod_15 (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) :
  (a + b + c) % 15 = 8 :=
by
  sorry

end sum_of_remainders_mod_15_l145_145531


namespace translate_function_down_l145_145901

theorem translate_function_down 
  (f : ℝ → ℝ) 
  (a k : ℝ) 
  (h : ∀ x, f x = a * x) 
  : ∀ x, (f x - k) = a * x - k :=
by
  sorry

end translate_function_down_l145_145901


namespace area_of_square_l145_145100

theorem area_of_square (r s L B: ℕ) (h1 : r = s) (h2 : L = 5 * r) (h3 : B = 11) (h4 : 220 = L * B) : s^2 = 16 := by
  sorry

end area_of_square_l145_145100


namespace sin_35pi_over_6_l145_145947

theorem sin_35pi_over_6 : Real.sin (35 * Real.pi / 6) = -1 / 2 := by
  sorry

end sin_35pi_over_6_l145_145947


namespace ratio_of_a_b_l145_145060

variable (x y a b : ℝ)

theorem ratio_of_a_b (h₁ : 4 * x - 2 * y = a)
                     (h₂ : 6 * y - 12 * x = b)
                     (hb : b ≠ 0)
                     (ha_solution : ∃ x y, 4 * x - 2 * y = a ∧ 6 * y - 12 * x = b) :
                     a / b = 1 / 3 :=
by sorry

end ratio_of_a_b_l145_145060


namespace water_consumption_per_week_l145_145249

-- Definitions for the given conditions
def bottles_per_day := 2
def quarts_per_bottle := 1.5
def additional_ounces_per_day := 20
def days_per_week := 7
def ounces_per_quart := 32

-- Theorem to state the problem
theorem water_consumption_per_week :
  bottles_per_day * quarts_per_bottle * ounces_per_quart + additional_ounces_per_day 
  * days_per_week = 812 := 
by 
  sorry

end water_consumption_per_week_l145_145249


namespace find_largest_number_l145_145247

noncomputable def largest_of_three_numbers (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ x ≥ z then x
  else if y ≥ x ∧ y ≥ z then y
  else z

theorem find_largest_number (x y z : ℝ) (h1 : x + y + z = 3) (h2 : xy + xz + yz = -11) (h3 : xyz = 15) :
  largest_of_three_numbers x y z = Real.sqrt 5 := by
  sorry

end find_largest_number_l145_145247


namespace quadratic_greatest_value_and_real_roots_l145_145425

theorem quadratic_greatest_value_and_real_roots :
  (∀ x : ℝ, -x^2 + 9 * x - 20 ≥ 0 → x ≤ 5)
  ∧ (∃ x : ℝ, -x^2 + 9 * x - 20 = 0)
  :=
sorry

end quadratic_greatest_value_and_real_roots_l145_145425


namespace prove_inequality_l145_145774

variable (x y z : ℝ)
variable (h₁ : x > 0)
variable (h₂ : y > 0)
variable (h₃ : z > 0)
variable (h₄ : x + y + z = 1)

theorem prove_inequality :
  (3 * x^2 - x) / (1 + x^2) +
  (3 * y^2 - y) / (1 + y^2) +
  (3 * z^2 - z) / (1 + z^2) ≥ 0 :=
by
  sorry

end prove_inequality_l145_145774


namespace fly_distance_from_ceiling_l145_145517

/-- 
Assume a room where two walls and the ceiling meet at right angles at point P.
Let point P be the origin (0, 0, 0). 
Let the fly's position be (2, 7, z), where z is the distance from the ceiling.
Given the fly is 2 meters from one wall, 7 meters from the other wall, 
and 10 meters from point P, prove that the fly is at a distance sqrt(47) from the ceiling.
-/
theorem fly_distance_from_ceiling : 
  ∀ (z : ℝ), 
  (2^2 + 7^2 + z^2 = 10^2) → 
  z = Real.sqrt 47 :=
by 
  intro z h
  sorry

end fly_distance_from_ceiling_l145_145517


namespace evaluate_at_two_l145_145529

def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 1

theorem evaluate_at_two : f 2 = 15 :=
by
  sorry

end evaluate_at_two_l145_145529


namespace find_triangle_value_l145_145323

theorem find_triangle_value 
  (triangle : ℕ)
  (h_units : (triangle + 3) % 7 = 2)
  (h_tens : (1 + 4 + triangle) % 7 = 4)
  (h_hundreds : (2 + triangle + 1) % 7 = 2)
  (h_thousands : 3 + 0 + 1 = 4) :
  triangle = 6 :=
sorry

end find_triangle_value_l145_145323


namespace cubic_polynomial_root_sum_cube_value_l145_145566

noncomputable def α : ℝ := (17 : ℝ)^(1 / 3)
noncomputable def β : ℝ := (67 : ℝ)^(1 / 3)
noncomputable def γ : ℝ := (137 : ℝ)^(1 / 3)

theorem cubic_polynomial_root_sum_cube_value
    (p q r : ℝ)
    (h1 : (p - α) * (p - β) * (p - γ) = 1)
    (h2 : (q - α) * (q - β) * (q - γ) = 1)
    (h3 : (r - α) * (r - β) * (r - γ) = 1) :
    p^3 + q^3 + r^3 = 218 := 
by
  sorry

end cubic_polynomial_root_sum_cube_value_l145_145566


namespace election_winner_won_by_votes_l145_145198

theorem election_winner_won_by_votes (V : ℝ) (winner_votes : ℝ) (loser_votes : ℝ)
    (h1 : winner_votes = 0.62 * V)
    (h2 : winner_votes = 930)
    (h3 : loser_votes = 0.38 * V)
    : winner_votes - loser_votes = 360 := 
  sorry

end election_winner_won_by_votes_l145_145198


namespace completing_the_square_l145_145259

theorem completing_the_square :
  ∀ x : ℝ, x^2 - 4 * x - 2 = 0 ↔ (x - 2)^2 = 6 :=
by
  sorry

end completing_the_square_l145_145259


namespace matrices_inverse_sum_l145_145242

open Matrix

noncomputable def M1 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![x, 2, y], ![3, 3, 4], ![z, 6, w]]

noncomputable def M2 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![[-6, m, -12], ![n, -14, p], ![3, q, 5]]

theorem matrices_inverse_sum 
    (h_inv : M1 * M2 = (1 : Matrix (Fin 3) (Fin 3) ℤ)) : x + y + z + w + m + n + p + q = 49 := by
  sorry

end matrices_inverse_sum_l145_145242


namespace range_of_a_minus_abs_b_l145_145841

theorem range_of_a_minus_abs_b (a b : ℝ) (h1 : 1 < a ∧ a < 8) (h2 : -4 < b ∧ b < 2) : 
  -3 < a - |b| ∧ a - |b| < 8 :=
sorry

end range_of_a_minus_abs_b_l145_145841


namespace square_area_with_tangent_circles_l145_145761

theorem square_area_with_tangent_circles :
  let r := 3 -- radius of each circle in inches
  let d := 2 * r -- diameter of each circle in inches
  let side_length := 2 * d -- side length of the square in inches
  let area := side_length * side_length -- area of the square in square inches
  side_length = 12 ∧ area = 144 :=
by
  let r := 3
  let d := 2 * r
  let side_length := 2 * d
  let area := side_length * side_length
  sorry

end square_area_with_tangent_circles_l145_145761


namespace x_plus_y_l145_145739

noncomputable def sequence_constant : ℝ := 1 / 4

def x : ℝ := 256 * sequence_constant

def y : ℝ := x * sequence_constant

theorem x_plus_y : x + y = 80 := 
by
  -- Placeholder for the proof
  sorry

end x_plus_y_l145_145739


namespace simplify_expression_l145_145503

theorem simplify_expression :
  (2 * (Real.sqrt 2 + Real.sqrt 6)) / (3 * Real.sqrt (2 + Real.sqrt 3)) = 4 / 3 :=
by
  sorry

end simplify_expression_l145_145503


namespace true_inverse_propositions_count_l145_145674

-- Let P1, P2, P3, P4 denote the original propositions
def P1 := "Supplementary angles are congruent, and two lines are parallel."
def P2 := "If |a| = |b|, then a = b."
def P3 := "Right angles are congruent."
def P4 := "Congruent angles are vertical angles."

-- Let IP1, IP2, IP3, IP4 denote the inverse propositions
def IP1 := "Two lines are parallel, and supplementary angles are congruent."
def IP2 := "If a = b, then |a| = |b|."
def IP3 := "Congruent angles are right angles."
def IP4 := "Vertical angles are congruent angles."

-- Counting the number of true inverse propositions
def countTrueInversePropositions : ℕ :=
  let p1_inverse_true := true  -- IP1 is true
  let p2_inverse_true := true  -- IP2 is true
  let p3_inverse_true := false -- IP3 is false
  let p4_inverse_true := true  -- IP4 is true
  [p1_inverse_true, p2_inverse_true, p4_inverse_true].length

-- The statement to be proved
theorem true_inverse_propositions_count : countTrueInversePropositions = 3 := by
  sorry

end true_inverse_propositions_count_l145_145674


namespace sum_of_powers_eq_123_l145_145652

section

variables {a b : Real}

-- Conditions provided in the problem
axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7

-- Define the theorem to be proved
theorem sum_of_powers_eq_123 : a^10 + b^10 = 123 :=
sorry

end

end sum_of_powers_eq_123_l145_145652


namespace elvis_recording_time_l145_145159

theorem elvis_recording_time :
  ∀ (total_studio_time writing_time_per_song editing_time number_of_songs : ℕ),
  total_studio_time = 300 →
  writing_time_per_song = 15 →
  editing_time = 30 →
  number_of_songs = 10 →
  (total_studio_time - (number_of_songs * writing_time_per_song + editing_time)) / number_of_songs = 12 :=
by
  intros total_studio_time writing_time_per_song editing_time number_of_songs
  intros h1 h2 h3 h4
  sorry

end elvis_recording_time_l145_145159


namespace solution_set_of_bx2_minus_ax_minus_1_gt_0_l145_145459

theorem solution_set_of_bx2_minus_ax_minus_1_gt_0
  (a b : ℝ)
  (h1 : ∀ (x : ℝ), 2 < x ∧ x < 3 ↔ x^2 - a * x - b < 0) :
  ∀ (x : ℝ), -1 / 2 < x ∧ x < -1 / 3 ↔ b * x^2 - a * x - 1 > 0 :=
by
  sorry

end solution_set_of_bx2_minus_ax_minus_1_gt_0_l145_145459


namespace reflected_ray_equation_l145_145647

theorem reflected_ray_equation (x y : ℝ) (incident_ray : y = 2 * x + 1) (reflecting_line : y = x) :
  x - 2 * y - 1 = 0 :=
sorry

end reflected_ray_equation_l145_145647


namespace find_x_average_is_3_l145_145767

theorem find_x_average_is_3 (x : ℝ) (h : (2 + 4 + 1 + 3 + x) / 5 = 3) : x = 5 :=
sorry

end find_x_average_is_3_l145_145767


namespace sum_of_three_numbers_l145_145369

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : a * b + b * c + c * a = 100) : 
  a + b + c = 21 := 
by
  sorry

end sum_of_three_numbers_l145_145369


namespace marble_ratio_is_two_to_one_l145_145156

-- Conditions
def dan_blue_marbles : ℕ := 5
def mary_blue_marbles : ℕ := 10

-- Ratio definition
def marble_ratio : ℚ := mary_blue_marbles / dan_blue_marbles

-- Theorem statement
theorem marble_ratio_is_two_to_one : marble_ratio = 2 :=
by 
  -- Prove the statement here
  sorry

end marble_ratio_is_two_to_one_l145_145156


namespace adam_bought_26_books_l145_145278

-- Conditions
def initial_books : ℕ := 56
def shelves : ℕ := 4
def avg_books_per_shelf : ℕ := 20
def leftover_books : ℕ := 2

-- Definitions based on conditions
def capacity_books : ℕ := shelves * avg_books_per_shelf
def total_books_after_trip : ℕ := capacity_books + leftover_books

-- Question: How many books did Adam buy on his shopping trip?
def books_bought : ℕ := total_books_after_trip - initial_books

theorem adam_bought_26_books :
  books_bought = 26 :=
by
  sorry

end adam_bought_26_books_l145_145278


namespace solve_fraction_equation_l145_145353

theorem solve_fraction_equation (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ -3) :
  (2 / x + x / (x + 3) = 1) ↔ x = 6 := 
by
  sorry

end solve_fraction_equation_l145_145353


namespace aaron_earnings_l145_145403

def monday_hours : ℚ := 7 / 4
def tuesday_hours : ℚ := 1 + 10 / 60
def wednesday_hours : ℚ := 3 + 15 / 60
def friday_hours : ℚ := 45 / 60

def total_hours_worked : ℚ := monday_hours + tuesday_hours + wednesday_hours + friday_hours
def hourly_rate : ℚ := 4

def total_earnings : ℚ := total_hours_worked * hourly_rate

theorem aaron_earnings : total_earnings = 27 := by
  sorry

end aaron_earnings_l145_145403


namespace intersection_eq_l145_145881

def setA : Set ℝ := { y | ∃ x : ℝ, y = 2^x }
def setB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }
def expectedIntersection : Set ℝ := { y | 0 < y }

theorem intersection_eq :
  setA ∩ setB = expectedIntersection :=
sorry

end intersection_eq_l145_145881


namespace chosen_numbers_rel_prime_l145_145488

theorem chosen_numbers_rel_prime :
  ∀ (s : Finset ℕ), s ⊆ Finset.range 2003 → s.card = 1002 → ∃ (x y : ℕ), x ∈ s ∧ y ∈ s ∧ Nat.gcd x y = 1 :=
by
  sorry

end chosen_numbers_rel_prime_l145_145488


namespace union_A_B_intersection_complementA_B_range_of_a_l145_145313

-- Definition of the universal set U, sets A and B
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}

-- Complement of A in the universal set U
def complement_A : Set ℝ := {x | x < 1 ∨ x ≥ 5}

-- Definition of set C parametrized by a
def C (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

-- Prove that A ∪ B is {x | 1 ≤ x < 8}
theorem union_A_B : A ∪ B = {x | 1 ≤ x ∧ x < 8} :=
sorry

-- Prove that (complement_U A) ∩ B = {x | 5 ≤ x < 8}
theorem intersection_complementA_B : (complement_A ∩ B) = {x | 5 ≤ x ∧ x < 8} :=
sorry

-- Prove the range of values for a if C ∩ A = C
theorem range_of_a (a : ℝ) : (C a ∩ A = C a) → a ≤ -1 :=
sorry

end union_A_B_intersection_complementA_B_range_of_a_l145_145313


namespace total_bins_sum_l145_145866

def total_bins_soup : ℝ := 0.2
def total_bins_vegetables : ℝ := 0.35
def total_bins_fruits : ℝ := 0.15
def total_bins_pasta : ℝ := 0.55
def total_bins_canned_meats : ℝ := 0.275
def total_bins_beans : ℝ := 0.175

theorem total_bins_sum :
  total_bins_soup + total_bins_vegetables + total_bins_fruits + total_bins_pasta + total_bins_canned_meats + total_bins_beans = 1.7 :=
by
  sorry

end total_bins_sum_l145_145866


namespace length_of_bridge_is_correct_l145_145550

noncomputable def train_length : ℝ := 150
noncomputable def crossing_time : ℝ := 29.997600191984642
noncomputable def train_speed_kmph : ℝ := 36
noncomputable def kmph_to_mps (v : ℝ) : ℝ := (v * 1000) / 3600
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge_is_correct :
  bridge_length = 149.97600191984642 := by
  sorry

end length_of_bridge_is_correct_l145_145550


namespace intersection_count_sum_l145_145411

theorem intersection_count_sum : 
  let m := 252
  let n := 252
  m + n = 504 := 
by {
  let m := 252 
  let n := 252 
  exact Eq.refl 504
}

end intersection_count_sum_l145_145411


namespace problem_solution_l145_145773

theorem problem_solution
  (a b : ℝ)
  (h₁ : a > b)
  (h₂ : b > 0)
  (h3 : (sqrt a) ^ 2 - 5 * (sqrt a) + 2 = 0)
  (h4 : (sqrt b) ^ 2 - 5 * (sqrt b) + 2 = 0) :
  ( (a * sqrt a + b * sqrt b) / (a - b) * 
    (2 / sqrt a - 2 / sqrt b) / 
    (sqrt a - (a + b) / sqrt b) + 
    5 * (5 * sqrt a - a) / (b + 2)
  ) = 5 :=
by
  sorry

end problem_solution_l145_145773


namespace fraction_multiplication_l145_145394

noncomputable def a : ℚ := 5 / 8
noncomputable def b : ℚ := 7 / 12
noncomputable def c : ℚ := 3 / 7
noncomputable def n : ℚ := 1350

theorem fraction_multiplication : a * b * c * n = 210.9375 := by
  sorry

end fraction_multiplication_l145_145394


namespace value_of_m_l145_145452

theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 2 ∧ x^2 - m * x + 8 = 0) → m = 6 := by
  sorry

end value_of_m_l145_145452


namespace julian_comic_book_l145_145333

theorem julian_comic_book : 
  ∀ (total_frames frames_per_page : ℕ),
    total_frames = 143 →
    frames_per_page = 11 →
    total_frames / frames_per_page = 13 ∧ total_frames % frames_per_page = 0 :=
by
  intros total_frames frames_per_page
  intros h_total_frames h_frames_per_page
  sorry

end julian_comic_book_l145_145333


namespace max_product_of_roots_of_quadratic_l145_145180

theorem max_product_of_roots_of_quadratic :
  ∃ k : ℚ, 6 * k^2 - 8 * k + (4 / 3) = 0 ∧ (64 - 48 * k) ≥ 0 ∧ (∀ k' : ℚ, (64 - 48 * k') ≥ 0 → (k'/3) ≤ (4/9)) :=
by
  sorry

end max_product_of_roots_of_quadratic_l145_145180


namespace solve_x_l145_145816

theorem solve_x : ∃ (x : ℚ), (3*x - 17) / 4 = (x + 9) / 6 ∧ x = 69 / 7 :=
by
  sorry

end solve_x_l145_145816


namespace prosecutor_cases_knight_or_liar_l145_145832

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

end prosecutor_cases_knight_or_liar_l145_145832


namespace problem1_proof_l145_145538

-- Define the mathematical conditions and problems
def problem1_expression (x y : ℝ) : ℝ := y * (4 * x - 3 * y) + (x - 2 * y) ^ 2

-- State the theorem with the simplified form as the conclusion
theorem problem1_proof (x y : ℝ) : problem1_expression x y = x^2 + y^2 :=
by
  sorry

end problem1_proof_l145_145538


namespace johns_website_visits_l145_145639

theorem johns_website_visits (c: ℝ) (d: ℝ) (days: ℕ) (h1: c = 0.01) (h2: d = 10) (h3: days = 30) :
  d / c * days = 30000 :=
by
  sorry

end johns_website_visits_l145_145639


namespace area_within_fence_l145_145094

theorem area_within_fence : 
  let rectangle_area := 20 * 18
  let cutout_area := 4 * 4
  rectangle_area - cutout_area = 344 := by
    -- Definitions
    let rectangle_area := 20 * 18
    let cutout_area := 4 * 4
    
    -- Computation of areas
    show rectangle_area - cutout_area = 344
    sorry

end area_within_fence_l145_145094


namespace ratio_expression_l145_145620

variable (a b c : ℚ)
variable (h1 : a / b = 6 / 5)
variable (h2 : b / c = 8 / 7)

theorem ratio_expression (a b c : ℚ) (h1 : a / b = 6 / 5) (h2 : b / c = 8 / 7) :
  (7 * a + 6 * b + 5 * c) / (7 * a - 6 * b + 5 * c) = 751 / 271 := by
  sorry

end ratio_expression_l145_145620


namespace min_packs_126_l145_145232

-- Define the sizes of soda packs
def pack_sizes : List ℕ := [6, 12, 24, 48]

-- Define the total number of cans required
def total_cans : ℕ := 126

-- Define a function to calculate the minimum number of packs required
noncomputable def min_packs_to_reach_target (target : ℕ) (sizes : List ℕ) : ℕ :=
sorry -- Implementation will be complex dynamic programming or greedy algorithm

-- The main theorem statement to prove
theorem min_packs_126 (P : ℕ) (h1 : (min_packs_to_reach_target total_cans pack_sizes) = P) : P = 4 :=
sorry -- Proof not required

end min_packs_126_l145_145232


namespace unsuccessful_attempts_124_l145_145363

theorem unsuccessful_attempts_124 (num_digits: ℕ) (choices_per_digit: ℕ) (total_attempts: ℕ):
  num_digits = 3 → choices_per_digit = 5 → total_attempts = choices_per_digit ^ num_digits →
  total_attempts - 1 = 124 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact sorry

end unsuccessful_attempts_124_l145_145363


namespace unique_integral_solution_l145_145426

noncomputable def positiveInt (x : ℤ) : Prop := x > 0

theorem unique_integral_solution (m n : ℤ) (hm : positiveInt m) (hn : positiveInt n) (unique_sol : ∃! (x y : ℤ), x + y^2 = m ∧ x^2 + y = n) : 
  ∃ (k : ℕ), m - n = 2^k ∨ m - n = -2^k :=
sorry

end unique_integral_solution_l145_145426


namespace gasoline_price_decrease_l145_145200

theorem gasoline_price_decrease (a : ℝ) (h : 0 ≤ a) :
  8.1 * (1 - a / 100) ^ 2 = 7.8 :=
sorry

end gasoline_price_decrease_l145_145200


namespace calculate_final_number_l145_145689

theorem calculate_final_number (initial increment times : ℕ) (h₀ : initial = 540) (h₁ : increment = 10) (h₂ : times = 6) : initial + increment * times = 600 :=
by
  sorry

end calculate_final_number_l145_145689


namespace inscribed_circle_radius_of_DEF_l145_145525

theorem inscribed_circle_radius_of_DEF (DE DF EF : ℝ) (r : ℝ)
  (hDE : DE = 8) (hDF : DF = 8) (hEF : EF = 10) :
  r = 5 * Real.sqrt 39 / 13 :=
by
  sorry

end inscribed_circle_radius_of_DEF_l145_145525


namespace difference_max_min_students_l145_145463

-- Definitions for problem conditions
def total_students : ℕ := 50
def shanghai_university_min : ℕ := 40
def shanghai_university_max : ℕ := 45
def shanghai_normal_university_min : ℕ := 16
def shanghai_normal_university_max : ℕ := 20

-- Lean statement for the math proof problem
theorem difference_max_min_students :
  (∀ (a b : ℕ), shanghai_university_min ≤ a ∧ a ≤ shanghai_university_max →
                shanghai_normal_university_min ≤ b ∧ b ≤ shanghai_normal_university_max →
                15 ≤ a + b - total_students ∧ a + b - total_students ≤ 15) →
  (∀ (a b : ℕ), shanghai_university_min ≤ a ∧ a ≤ shanghai_university_max →
                shanghai_normal_university_min ≤ b ∧ b ≤ shanghai_normal_university_max →
                6 ≤ a + b - total_students ∧ a + b - total_students ≤ 6) →
  (∃ M m : ℕ, 
    (M = 15) ∧ 
    (m = 6) ∧ 
    (M - m = 9)) :=
by
  sorry

end difference_max_min_students_l145_145463


namespace values_of_a_and_b_range_of_c_isosceles_perimeter_l145_145316

def a : ℝ := 3
def b : ℝ := 4

axiom triangle_ABC (c : ℝ) : 0 < c

noncomputable def equation_condition (a b : ℝ) : Prop :=
  |a-3| + (b-4)^2 = 0

noncomputable def is_valid_c (c : ℝ) : Prop :=
  1 < c ∧ c < 7

theorem values_of_a_and_b (h : equation_condition a b) : a = 3 ∧ b = 4 := sorry

theorem range_of_c (h : equation_condition a b) : is_valid_c c := sorry

noncomputable def isosceles_triangle (c : ℝ) : Prop :=
  c = 4 ∨ c = 3

theorem isosceles_perimeter (h : equation_condition a b) (hc : isosceles_triangle c) : (3 + 3 + 4 = 10) ∨ (4 + 4 + 3 = 11) := sorry

end values_of_a_and_b_range_of_c_isosceles_perimeter_l145_145316


namespace find_a_value_l145_145571

theorem find_a_value :
  (∀ y : ℝ, y ∈ Set.Ioo (-3/2 : ℝ) 4 → y * (2 * y - 3) < (12 : ℝ)) ↔ (12 = 12) := 
by 
  sorry

end find_a_value_l145_145571


namespace sequence_x_y_sum_l145_145746

theorem sequence_x_y_sum (r : ℝ) (x y : ℝ)
  (h₁ : r = 1 / 4)
  (h₂ : x = 256 * r)
  (h₃ : y = x * r) :
  x + y = 80 :=
by
  sorry

end sequence_x_y_sum_l145_145746


namespace hyperbola_eccentricity_is_sqrt_3_l145_145487

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b^2 = 2 * a^2) : ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_eccentricity_is_sqrt_3 (a b : ℝ) (h1 : a > 0) (h2 : b^2 = 2 * a^2) :
  hyperbola_eccentricity a b h1 h2 = Real.sqrt 3 :=
by
  sorry

end hyperbola_eccentricity_is_sqrt_3_l145_145487


namespace main_inequality_equality_condition_l145_145644

variable {a b c : ℝ}
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem main_inequality 
  (hpos_a : 0 < a) 
  (hpos_b : 0 < b) 
  (hpos_c : 0 < c) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c)) :=
  sorry

theorem equality_condition 
  (hpos_a : 0 < a) 
  (hpos_b : 0 < b) 
  (hpos_c : 0 < c) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) = 3 / (1 + a * b * c) ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
  sorry

end main_inequality_equality_condition_l145_145644


namespace inequality_proof_l145_145814

theorem inequality_proof (a b : ℝ) : 
  a^2 + b^2 + 2 * (a - 1) * (b - 1) ≥ 1 := 
by 
  sorry

end inequality_proof_l145_145814


namespace integer_solutions_system_inequalities_l145_145022

theorem integer_solutions_system_inequalities:
  {x : ℤ} → (2 * x - 1 < x + 1) → (1 - 2 * (x - 1) ≤ 3) → x = 0 ∨ x = 1 := 
by
  intros x h1 h2
  sorry

end integer_solutions_system_inequalities_l145_145022


namespace solve_equation_l145_145352

theorem solve_equation : ∀ x : ℝ, x ≠ 0 ∧ x ≠ -3 → (2 / x + x / (x + 3) = 1) ↔ (x = 6) := by
  intro x h
  have h1 : x ≠ 0 := h.1
  have h2 : x ≠ -3 := h.2
  sorry

end solve_equation_l145_145352


namespace evaluate_expression_l145_145422

-- Definitions based on conditions
variables (b : ℤ) (x : ℤ)
def condition := x = 2 * b + 9

-- Statement of the problem
theorem evaluate_expression (b : ℤ) (x : ℤ) (h : condition b x) : x - 2 * b + 5 = 14 :=
by sorry

end evaluate_expression_l145_145422


namespace length_PT_30_l145_145794

noncomputable def length_PT (PQ QR : ℝ) (angle_QRT : ℝ) (T_on_RS : Prop) : ℝ := 
  if h : PQ = 30 ∧ QR = 15 ∧ angle_QRT = 75 then 30 else 0

theorem length_PT_30 (PQ QR : ℝ) (angle_QRT : ℝ) (T_on_RS : Prop) :
  PQ = 30 → QR = 15 → angle_QRT = 75 → length_PT PQ QR angle_QRT T_on_RS = 30 :=
sorry

end length_PT_30_l145_145794


namespace average_temperature_l145_145358

theorem average_temperature :
  ∀ (T : ℝ) (Tt : ℝ),
  -- Conditions
  (43 + T + T + T) / 4 = 48 → 
  Tt = 35 →
  -- Proof
  (T + T + T + Tt) / 4 = 46 :=
by
  intros T Tt H1 H2
  sorry

end average_temperature_l145_145358


namespace ball_travel_distance_l145_145966

theorem ball_travel_distance 
    (initial_height : ℕ)
    (half : ℕ → ℕ)
    (num_bounces : ℕ)
    (height_after_bounce : ℕ → ℕ)
    (total_distance : ℕ) :
    initial_height = 16 ∧ 
    (∀ n, half n = n / 2) ∧ 
    num_bounces = 4 ∧ 
    (height_after_bounce 0 = initial_height) ∧
    (∀ n, height_after_bounce (n + 1) = half (height_after_bounce n))
→ total_distance = 46 :=
by
  sorry

end ball_travel_distance_l145_145966


namespace simplify_fraction_l145_145663

theorem simplify_fraction : 
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1))) = 
  ((Real.sqrt 3) + 2 * (Real.sqrt 5) - 1) / (2 + 4 * Real.sqrt 5) := 
by 
  sorry

end simplify_fraction_l145_145663


namespace range_of_g_le_2_minus_x_l145_145592

noncomputable def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ :=
if x ≥ 0 then f x else -f (-x)

theorem range_of_g_le_2_minus_x : {x : ℝ | g x ≤ 2 - x} = {x : ℝ | x ≤ 1} :=
by sorry

end range_of_g_le_2_minus_x_l145_145592


namespace find_certain_number_l145_145393

theorem find_certain_number (x : ℝ) 
  (h : 3889 + x - 47.95000000000027 = 3854.002) : x = 12.95200000000054 :=
by
  sorry

end find_certain_number_l145_145393


namespace ratio_current_to_boat_l145_145129

def boat_sailing_condition (distance : ℕ) (time_upstream : ℕ) (time_downstream : ℕ) (b c : ℚ) : Prop :=
  (b - c = distance / time_upstream) ∧ (b + c = distance / time_downstream)

theorem ratio_current_to_boat (b c : ℚ) (h : boat_sailing_condition 15 5 3 b c) :
  c / b = 1 / 4 :=
  by sorry

end ratio_current_to_boat_l145_145129


namespace part1_part2_l145_145218

def A (x : ℝ) : Prop := x < -2 ∨ x > 3
def B (a : ℝ) (x : ℝ) : Prop := 1 - a < x ∧ x < a + 3

theorem part1 (x : ℝ) : (¬A x ∨ B 1 x) ↔ -2 ≤ x ∧ x < 4 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, ¬(A x ∧ B a x)) ↔ a ≤ 0 :=
by
  sorry

end part1_part2_l145_145218


namespace perfect_match_of_products_l145_145899

theorem perfect_match_of_products
  (x : ℕ)  -- number of workers assigned to produce nuts
  (h1 : 22 - x ≥ 0)  -- ensuring non-negative number of workers for screws
  (h2 : 1200 * (22 - x) = 2 * 2000 * x) :  -- the condition for perfect matching
  (2 * 1200 * (22 - x) = 2000 * x) :=  -- the correct equation
by sorry

end perfect_match_of_products_l145_145899


namespace number_times_half_squared_eq_eight_l145_145256

theorem number_times_half_squared_eq_eight : 
  ∃ n : ℝ, n * (1/2)^2 = 2^3 := 
sorry

end number_times_half_squared_eq_eight_l145_145256


namespace temperature_below_75_l145_145150

theorem temperature_below_75
  (T : ℝ)
  (H1 : ∀ T, T ≥ 75 → swimming_area_open)
  (H2 : ¬swimming_area_open) : 
  T < 75 :=
sorry

end temperature_below_75_l145_145150


namespace only_four_letter_list_with_same_product_as_TUVW_l145_145287

/-- Each letter of the alphabet is assigned a value $A=1, B=2, C=3, \ldots, Z=26$. 
    The product of a four-letter list is the product of the values of its four letters. 
    The product of the list $TUVW$ is $(20)(21)(22)(23)$. 
    Prove that the only other four-letter list with a product equal to the product of the list $TUVW$ 
    is the list $TUVW$ itself. 
 -/
theorem only_four_letter_list_with_same_product_as_TUVW : ∀ (l : list char), 
  (l.all (λ c, 'A' ≤ c ∧ c ≤ 'Z')) → 
  (l.prod (λ c, letter_value c) = (20 * 21 * 22 * 23)) →
  (l = ['T', 'U', 'V', 'W']) :=
by
  sorry

/-- Helper function to convert a letter to its value. -/
def letter_value (c : char) : ℕ :=
  c.to_nat - 'A'.to_nat + 1

#eval letter_value 'T' -- 20
#eval letter_value 'U' -- 21
#eval letter_value 'V' -- 22
#eval letter_value 'W' -- 23

end only_four_letter_list_with_same_product_as_TUVW_l145_145287


namespace probability_two_independent_events_l145_145420

def probability_first_die (n : ℕ) : ℚ := if n > 4 then 1/3 else 0
def probability_second_die (n : ℕ) : ℚ := if n > 2 then 2/3 else 0

theorem probability_two_independent_events :
  (probability_first_die 5) * (probability_second_die 3) = 2 / 9 := by
  sorry

end probability_two_independent_events_l145_145420


namespace set_A_is_listed_correctly_l145_145751

def A : Set ℤ := { x | -3 < x ∧ x < 1 }

theorem set_A_is_listed_correctly : A = {-2, -1, 0} := 
by
  sorry

end set_A_is_listed_correctly_l145_145751


namespace number_of_books_bought_l145_145277

def initial_books : ℕ := 56
def shelves : ℕ := 4
def books_per_shelf : ℕ := 20
def remaining_books : ℕ := 2

theorem number_of_books_bought : 
  let total_books_after_shopping := shelves * books_per_shelf + remaining_books in
  total_books_after_shopping - initial_books = 26 := 
by 
  sorry

end number_of_books_bought_l145_145277


namespace pineapple_total_cost_correct_l145_145534

-- Define the conditions
def pineapple_cost : ℝ := 1.25
def num_pineapples : ℕ := 12
def shipping_cost : ℝ := 21.00

-- Calculate total cost
noncomputable def total_pineapple_cost : ℝ := pineapple_cost * num_pineapples
noncomputable def total_cost : ℝ := total_pineapple_cost + shipping_cost
noncomputable def cost_per_pineapple : ℝ := total_cost / num_pineapples

-- The proof problem
theorem pineapple_total_cost_correct : cost_per_pineapple = 3 := by
  -- The proof will be filled in here
  sorry

end pineapple_total_cost_correct_l145_145534


namespace find_common_difference_l145_145805

def is_arithmetic_sequence (a : (ℕ → ℝ)) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def is_arithmetic_sequence_with_sum (a : (ℕ → ℝ)) (S : (ℕ → ℝ)) (d : ℝ) : Prop :=
  S 0 = a 0 ∧
  ∀ n, S (n + 1) = S n + a (n + 1) ∧
        ∀ n, (S (n + 1) / a (n + 1) - S n / a n) = d

theorem find_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a →
  is_arithmetic_sequence_with_sum a S d →
  (d = 1 ∨ d = 1 / 2) :=
sorry

end find_common_difference_l145_145805


namespace total_games_played_l145_145801

-- Define the conditions as parameters
def ratio_games_won_lost (W L : ℕ) : Prop := W / 2 = L / 3

-- Let's state the problem formally in Lean
theorem total_games_played (W L : ℕ) (h1 : ratio_games_won_lost W L) (h2 : W = 18) : W + L = 30 :=
by 
  sorry  -- The proof will be filled in


end total_games_played_l145_145801


namespace triangle_right_angle_l145_145465

theorem triangle_right_angle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : A + B = 90)
  (h2 : (a + b) * (a - b) = c ^ 2)
  (h3 : A / B = 1 / 2) :
  C = 90 :=
sorry

end triangle_right_angle_l145_145465


namespace jack_final_apples_l145_145904

-- Jack's transactions and initial count as conditions
def initial_count : ℕ := 150
def sold_to_jill : ℕ := initial_count * 30 / 100
def remaining_after_jill : ℕ := initial_count - sold_to_jill
def sold_to_june : ℕ := remaining_after_jill * 20 / 100
def remaining_after_june : ℕ := remaining_after_jill - sold_to_june
def donated_to_charity : ℕ := 5
def final_count : ℕ := remaining_after_june - donated_to_charity

-- Proof statement
theorem jack_final_apples : final_count = 79 := by
  sorry

end jack_final_apples_l145_145904


namespace triangle_perimeter_l145_145797

noncomputable theory

variables {A B C : ℝ}
variables {a b c : ℝ}

-- Define the perimeter of the triangle
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the conditions
axiom cos_eq_half : cos A = 1/2
axiom given_equation : c * cos B + b * cos C = 2 * a * cos A
axiom side_a : a = 2
axiom area_eq : (1 / 2) * b * c * sin A = sqrt 3

-- The theorem to prove
theorem triangle_perimeter : perimeter a b c = 6 :=
sorry

end triangle_perimeter_l145_145797


namespace largest_root_eq_l145_145757

theorem largest_root_eq : ∃ x, (∀ y, (abs (Real.cos (Real.pi * y) + y^3 - 3 * y^2 + 3 * y) = 3 - y^2 - 2 * y^3) → y ≤ x) ∧ x = 1 := sorry

end largest_root_eq_l145_145757


namespace females_advanced_degrees_under_40_l145_145460

-- Definitions derived from conditions
def total_employees : ℕ := 280
def female_employees : ℕ := 160
def male_employees : ℕ := 120
def advanced_degree_holders : ℕ := 120
def college_degree_holders : ℕ := 100
def high_school_diploma_holders : ℕ := 60
def male_advanced_degree_holders : ℕ := 50
def male_college_degree_holders : ℕ := 35
def male_high_school_diploma_holders : ℕ := 35
def percentage_females_under_40 : ℝ := 0.75

-- The mathematically equivalent proof problem
theorem females_advanced_degrees_under_40 : 
  (advanced_degree_holders - male_advanced_degree_holders) * percentage_females_under_40 = 52 :=
by
  sorry -- Proof to be provided

end females_advanced_degrees_under_40_l145_145460


namespace arithmetic_contains_geometric_l145_145266

theorem arithmetic_contains_geometric (a b : ℚ) (h : a^2 + b^2 ≠ 0) :
  ∃ (q : ℚ) (c : ℚ) (n₀ : ℕ) (n : ℕ → ℕ), (∀ k : ℕ, n (k+1) = n k + c * q^k) ∧
  ∀ k : ℕ, ∃ r : ℚ, a + b * n k = r * q^k :=
sorry

end arithmetic_contains_geometric_l145_145266


namespace length_of_bridge_l145_145849

theorem length_of_bridge
  (T : ℕ) (t : ℕ) (s : ℕ)
  (hT : T = 250)
  (ht : t = 20)
  (hs : s = 20) :
  ∃ L : ℕ, L = 150 :=
by
  sorry

end length_of_bridge_l145_145849


namespace carla_needs_30_leaves_l145_145858

-- Definitions of the conditions
def items_per_day : Nat := 5
def total_days : Nat := 10
def total_bugs : Nat := 20

-- Maths problem to be proved
theorem carla_needs_30_leaves :
  let total_items := items_per_day * total_days
  let required_leaves := total_items - total_bugs
  required_leaves = 30 :=
by
  sorry

end carla_needs_30_leaves_l145_145858


namespace marble_ratio_l145_145155

-- Definitions based on conditions
def dan_marbles : ℕ := 5
def mary_marbles : ℕ := 10

-- Statement of the theorem to prove the ratio is 2:1
theorem marble_ratio : mary_marbles / dan_marbles = 2 := by
  sorry

end marble_ratio_l145_145155


namespace fraction_subtraction_simplified_l145_145857

theorem fraction_subtraction_simplified :
  (8 / 19) - (5 / 57) = (1 / 3) :=
by
  sorry

end fraction_subtraction_simplified_l145_145857


namespace distance_between_homes_l145_145222

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

end distance_between_homes_l145_145222


namespace cory_prime_sum_l145_145861

def primes_between_30_and_60 : List ℕ := [31, 37, 41, 43, 47, 53, 59]

theorem cory_prime_sum :
  let smallest := 31
  let largest := 59
  let median := 43
  smallest ∈ primes_between_30_and_60 ∧
  largest ∈ primes_between_30_and_60 ∧
  median ∈ primes_between_30_and_60 ∧
  primes_between_30_and_60 = [31, 37, 41, 43, 47, 53, 59] → 
  smallest + largest + median = 133 := 
by
  intros; sorry

end cory_prime_sum_l145_145861


namespace cricket_initial_overs_l145_145791

theorem cricket_initial_overs
  (target_runs : ℚ) (initial_run_rate : ℚ) (remaining_run_rate : ℚ) (remaining_overs : ℕ)
  (total_runs_needed : target_runs = 282)
  (run_rate_initial : initial_run_rate = 3.4)
  (run_rate_remaining : remaining_run_rate = 6.2)
  (overs_remaining : remaining_overs = 40) :
  ∃ (initial_overs : ℕ), initial_overs = 10 :=
by
  sorry

end cricket_initial_overs_l145_145791


namespace newsletter_cost_l145_145035

theorem newsletter_cost (x : ℝ) (h1 : 14 * x < 16) (h2 : 19 * x > 21) : x = 1.11 :=
by
  sorry

end newsletter_cost_l145_145035


namespace calc_fraction_l145_145297

variable {x y : ℝ}

theorem calc_fraction (h : x + y = x * y - 1) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x) + (1 / y) = 1 - 1 / (x * y) := 
by 
  sorry

end calc_fraction_l145_145297


namespace complex_abs_of_sqrt_l145_145356

variable (z : ℂ)

theorem complex_abs_of_sqrt (h : z^2 = 16 - 30 * Complex.I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end complex_abs_of_sqrt_l145_145356


namespace total_buildings_proof_l145_145798

-- Given conditions
variables (stores_pittsburgh hospitals_pittsburgh schools_pittsburgh police_stations_pittsburgh : ℕ)
variables (stores_new hospitals_new schools_new police_stations_new buildings_new : ℕ)

-- Given values for Pittsburgh
def stores_pittsburgh := 2000
def hospitals_pittsburgh := 500
def schools_pittsburgh := 200
def police_stations_pittsburgh := 20

-- Definitions for the new city
def stores_new := stores_pittsburgh / 2
def hospitals_new := 2 * hospitals_pittsburgh
def schools_new := schools_pittsburgh - 50
def police_stations_new := police_stations_pittsburgh + 5
def buildings_new := stores_new + hospitals_new + schools_new + police_stations_new

-- Statement to prove
theorem total_buildings_proof : buildings_new = 2175 := by
  dsimp [buildings_new, stores_new, hospitals_new, schools_new, police_stations_new] 
  dsimp [stores_pittsburgh, hospitals_pittsburgh, schools_pittsburgh, police_stations_pittsburgh]
  rfl

end total_buildings_proof_l145_145798


namespace determine_m_l145_145321

-- Define a complex number structure in Lean
structure ComplexNumber where
  re : ℝ  -- real part
  im : ℝ  -- imaginary part

-- Define the condition where the complex number is purely imaginary
def is_purely_imaginary (z : ComplexNumber) : Prop :=
  z.re = 0

-- State the Lean theorem
theorem determine_m (m : ℝ) (h : is_purely_imaginary (ComplexNumber.mk (m^2 - m) m)) : m = 1 :=
by
  sorry

end determine_m_l145_145321


namespace problem_inequality_problem_equality_condition_l145_145214

theorem problem_inequality (a b c : ℕ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + a + b + c :=
sorry

theorem problem_equality_condition (a b c : ℕ) :
  (a^3 + b^3 + c^3) / 3 = a * b * c + a + b + c ↔ a + 1 = b ∧ b + 1 = c :=
sorry

end problem_inequality_problem_equality_condition_l145_145214


namespace geometric_Sn_over_n_sum_first_n_terms_l145_145216

-- The first problem statement translation to Lean 4
theorem geometric_Sn_over_n (a S : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n+1) = (n + 2) * S n) :
  ∃ r : ℕ, (r = 2 ∧ ∃ b : ℕ, b = 1 ∧ 
    ∀ n : ℕ, 0 < n → (S (n + 1)) / (n + 1) = r * (S n) / n) := 
sorry

-- The second problem statement translation to Lean 4
theorem sum_first_n_terms (a S : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n + 1) = (n + 2) * S n)
  (h3 : ∀ n : ℕ, S n = n * 2^(n - 1)) :
  ∀ n : ℕ, T n = (n - 1) * 2^n + 1 :=
sorry

end geometric_Sn_over_n_sum_first_n_terms_l145_145216


namespace distance_of_hyperbola_vertices_l145_145576

-- Define the hyperbola equation condition
def hyperbola : Prop := ∃ (y x : ℝ), (y^2 / 16) - (x^2 / 9) = 1

-- Define a variable for the distance between the vertices
def distance_between_vertices (a : ℝ) : ℝ := 2 * a

-- The main statement to be proved
theorem distance_of_hyperbola_vertices :
  hyperbola → distance_between_vertices 4 = 8 :=
by
  intro h
  sorry

end distance_of_hyperbola_vertices_l145_145576


namespace mass_of_man_is_correct_l145_145121

-- Definitions for conditions
def length_of_boat : ℝ := 3
def breadth_of_boat : ℝ := 2
def sinking_depth : ℝ := 0.012
def density_of_water : ℝ := 1000

-- Volume of water displaced
def volume_displaced := length_of_boat * breadth_of_boat * sinking_depth

-- Mass of the man
def mass_of_man := density_of_water * volume_displaced

-- Prove that the mass of the man is 72 kg
theorem mass_of_man_is_correct : mass_of_man = 72 := by
  sorry

end mass_of_man_is_correct_l145_145121


namespace alpha_beta_inequality_l145_145760

theorem alpha_beta_inequality (α β : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 0 < x → 0 < y → x^α * y^β < k * (x + y)) ↔ (0 ≤ α ∧ 0 ≤ β ∧ α + β = 1) :=
by
  sorry

end alpha_beta_inequality_l145_145760


namespace isosceles_triangle_congruent_side_length_l145_145925

theorem isosceles_triangle_congruent_side_length (BC : ℝ) (BM : ℝ) :
  BC = 4 * Real.sqrt 2 → BM = 5 → ∃ (AB : ℝ), AB = Real.sqrt 34 :=
by
  -- sorry is used here to indicate proof is not provided, but the statement is expected to build successfully.
  sorry

end isosceles_triangle_congruent_side_length_l145_145925


namespace solve_for_x_l145_145498

-- Define the equation as a predicate
def equation (x : ℝ) : Prop := (0.05 * x + 0.07 * (30 + x) = 15.4)

-- The proof statement
theorem solve_for_x :
  ∃ x : ℝ, equation x ∧ x = 110.8333 :=
by
  existsi (110.8333 : ℝ)
  split
  sorry -- Proof of the equation
  rfl -- Equality proof

end solve_for_x_l145_145498


namespace total_number_of_notes_l145_145547

theorem total_number_of_notes (x : ℕ) (h₁ : 37 * 50 + x * 500 = 10350) : 37 + x = 54 :=
by
  -- We state that the total value of 37 Rs. 50 notes plus x Rs. 500 notes equals Rs. 10350.
  -- According to this information, we prove that the total number of notes is 54.
  sorry

end total_number_of_notes_l145_145547


namespace problem_l145_145777

def f (x a b : ℝ) : ℝ := a * x ^ 3 - b * x + 1

theorem problem (a b : ℝ) (h : f 2 a b = -1) : f (-2) a b = 3 :=
by {
  sorry
}

end problem_l145_145777


namespace mushroom_drying_l145_145950

theorem mushroom_drying (M M' : ℝ) (m1 m2 : ℝ) :
  M = 100 ∧ m1 = 0.01 * M ∧ m2 = 0.02 * M' ∧ m1 = 1 → M' = 50 :=
by
  sorry

end mushroom_drying_l145_145950


namespace shaded_area_l145_145771

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

end shaded_area_l145_145771


namespace flavors_needed_this_year_l145_145782

def num_flavors_total : ℕ := 100

def num_flavors_two_years_ago : ℕ := num_flavors_total / 4

def num_flavors_last_year : ℕ := 2 * num_flavors_two_years_ago

def num_flavors_tried_so_far : ℕ := num_flavors_two_years_ago + num_flavors_last_year

theorem flavors_needed_this_year : 
  (num_flavors_total - num_flavors_tried_so_far) = 25 := by {
  sorry
}

end flavors_needed_this_year_l145_145782


namespace prob_multiples_of_3_or_4_l145_145243

open Set

def multiples_of_3_and_4 : Set ℕ := {n | n % 3 = 0 ∨ n % 4 = 0}

theorem prob_multiples_of_3_or_4 : 
  (1 / 2 : ℚ) = (Finite.toFinset (multiples_of_3_and_4 ∩ Icc 1 30)).card / 30 :=
begin
  sorry
end

end prob_multiples_of_3_or_4_l145_145243


namespace exponent_property_l145_145240

theorem exponent_property (a : ℝ) : a^7 = a^3 * a^4 :=
by
  -- The proof statement follows from the properties of exponents:
  -- a^m * a^n = a^(m + n)
  -- Therefore, a^3 * a^4 = a^(3 + 4) = a^7.
  sorry

end exponent_property_l145_145240


namespace power_identity_l145_145058

theorem power_identity :
  (3 ^ 12) * (3 ^ 8) = 243 ^ 4 :=
sorry

end power_identity_l145_145058


namespace no_two_digit_prime_sum_digits_nine_l145_145026

theorem no_two_digit_prime_sum_digits_nine :
  ¬ ∃ p : ℕ, prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 + p % 10 = 9) :=
sorry

end no_two_digit_prime_sum_digits_nine_l145_145026


namespace digit_in_tens_place_is_nine_l145_145400

/-
Given:
1. Two numbers represented as 6t5 and 5t6 (where t is a digit).
2. The result of subtracting these two numbers is 9?4, where '?' represents a single digit in the tens place.

Prove:
The digit represented by '?' in the tens place is 9.
-/

theorem digit_in_tens_place_is_nine (t : ℕ) (h1 : 0 ≤ t ∧ t ≤ 9) :
  let a := 600 + t * 10 + 5
  let b := 500 + t * 10 + 6
  let result := a - b
  (result % 100) / 10 = 9 :=
by {
  sorry
}

end digit_in_tens_place_is_nine_l145_145400


namespace unique_b_positive_solution_l145_145587

theorem unique_b_positive_solution (c : ℝ) (h : c ≠ 0) : 
  (∃ b : ℝ, b > 0 ∧ ∀ b : ℝ, b ≠ 0 → 
    ∀ x : ℝ, x^2 + (b + 1 / b) * x + c = 0 → x = - (b + 1 / b) / 2) 
  ↔ c = (5 + Real.sqrt 21) / 2 ∨ c = (5 - Real.sqrt 21) / 2 := 
by {
  sorry
}

end unique_b_positive_solution_l145_145587


namespace TimSpentTotal_l145_145187

variable (LunchCost : ℝ) (TipPercentage : ℝ)

def TotalAmountSpent (LunchCost : ℝ) (TipPercentage : ℝ) : ℝ := 
  LunchCost + (LunchCost * TipPercentage)

theorem TimSpentTotal (h1 : LunchCost = 50.50) (h2 : TipPercentage = 0.20) :
  TotalAmountSpent LunchCost TipPercentage = 60.60 := by
  sorry

end TimSpentTotal_l145_145187


namespace parabola_condition_max_area_triangle_l145_145612

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l145_145612


namespace q_can_do_work_in_10_days_l145_145124

theorem q_can_do_work_in_10_days (R_p R_q R_pq: ℝ)
  (h1 : R_p = 1 / 15)
  (h2 : R_pq = 1 / 6)
  (h3 : R_p + R_q = R_pq) :
  1 / R_q = 10 :=
by
  -- Proof steps go here.
  sorry

end q_can_do_work_in_10_days_l145_145124


namespace least_positive_integer_to_add_l145_145688

theorem least_positive_integer_to_add (n : ℕ) (h1 : n > 0) (h2 : (624 + n) % 5 = 0) : n = 1 := 
by
  sorry

end least_positive_integer_to_add_l145_145688


namespace union_of_sets_l145_145591

-- Definition for set M
def M : Set ℝ := {x | x^2 - 4 * x + 3 < 0}

-- Definition for set N
def N : Set ℝ := {x | 2 * x + 1 < 5}

-- The theorem linking M and N
theorem union_of_sets : M ∪ N = {x | x < 3} :=
by
  -- Proof goes here
  sorry

end union_of_sets_l145_145591


namespace max_gcd_coprime_l145_145583

theorem max_gcd_coprime (x y : ℤ) (h : Int.gcd x y = 1) : 
  Int.gcd (x + 2015 * y) (y + 2015 * x) ≤ 4060224 :=
sorry

end max_gcd_coprime_l145_145583


namespace num_balls_picked_l145_145842

-- Definitions based on the conditions
def numRedBalls : ℕ := 4
def numBlueBalls : ℕ := 3
def numGreenBalls : ℕ := 2
def totalBalls : ℕ := numRedBalls + numBlueBalls + numGreenBalls
def probFirstRed : ℚ := numRedBalls / totalBalls
def probSecondRed : ℚ := (numRedBalls - 1) / (totalBalls - 1)

-- Theorem stating the problem
theorem num_balls_picked :
  probFirstRed * probSecondRed = 1 / 6 → 
  (∃ (n : ℕ), n = 2) :=
by 
  sorry

end num_balls_picked_l145_145842


namespace least_area_of_triangle_DEF_l145_145367

theorem least_area_of_triangle_DEF :
  let z := c - 3 in
  let solutions := {w | ∃ k : ℕ, k < 10 ∧ w = 3 + (2 ^ (1/2 : ℝ) * Complex.exp (Complex.I * 2 * Real.pi * k / 10))} in
  let D := (Real.sqrt 2, 0) in
  let E := (Real.sqrt 2 * Real.cos (2 * Real.pi / 10), Real.sqrt 2 * Real.sin (2 * Real.pi / 10)) in
  let F := (Real.sqrt 2 * Real.cos (4 * Real.pi / 10), Real.sqrt 2 * Real.sin (4 * Real.pi / 10)) in
  (1 / 2 : ℝ) * 2 * Real.sqrt 2 * Real.sin (Real.pi / 10) * Real.sqrt 2 * Real.sin (2 * Real.pi / 10) = 2 * Real.sin (Real.pi / 10) * Real.sin (2 * Real.pi / 10).

end least_area_of_triangle_DEF_l145_145367


namespace total_potatoes_l145_145651

theorem total_potatoes (Nancy_potatoes : ℕ) (Sandy_potatoes : ℕ) (Andy_potatoes : ℕ) 
  (h1 : Nancy_potatoes = 6) (h2 : Sandy_potatoes = 7) (h3 : Andy_potatoes = 9) : 
  Nancy_potatoes + Sandy_potatoes + Andy_potatoes = 22 :=
by
  -- The proof can be written here
  sorry

end total_potatoes_l145_145651


namespace fraction_power_l145_145711

variables (a b c : ℝ)

theorem fraction_power :
  ( ( -2 * a^2 * b ) / (3 * c) )^2 = ( 4 * a^4 * b^2 ) / ( 9 * c^2 ) := 
by sorry

end fraction_power_l145_145711


namespace minimum_value_fraction_l145_145308

theorem minimum_value_fraction (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 2) : 
  (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 2 → 
    ((1 / (1 + x)) + (1 / (2 + 2 * y)) ≥ 4 / 5)) :=
by sorry

end minimum_value_fraction_l145_145308


namespace geom_seq_sum_2016_2017_l145_145796

noncomputable def geom_seq (n : ℕ) (a1 q : ℝ) : ℝ := a1 * q ^ (n - 1)

noncomputable def sum_geometric_seq (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then
  a1 * n
else
  a1 * (1 - q ^ n) / (1 - q)

theorem geom_seq_sum_2016_2017 :
  (a1 = 2) →
  (geom_seq 2 a1 q + geom_seq 5 a1 q = 0) →
  sum_geometric_seq a1 q 2016 + sum_geometric_seq a1 q 2017 = 2 :=
by
  sorry

end geom_seq_sum_2016_2017_l145_145796


namespace sean_div_julie_eq_two_l145_145921

def sum_n (n : ℕ) := n * (n + 1) / 2

def sean_sum := 2 * sum_n 500

def julie_sum := sum_n 500

theorem sean_div_julie_eq_two : sean_sum / julie_sum = 2 := 
by sorry

end sean_div_julie_eq_two_l145_145921


namespace geometric_progression_common_ratio_l145_145896

theorem geometric_progression_common_ratio (x y z w r : ℂ) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0)
  (h_geom : x * (y - w) = a ∧ y * (z - x) = a * r ∧ z * (w - y) = a * r^2 ∧ w * (x - z) = a * r^3) :
  1 + r + r^2 + r^3 = 0 :=
sorry

end geometric_progression_common_ratio_l145_145896


namespace flavors_needed_this_year_l145_145783

def num_flavors_total : ℕ := 100

def num_flavors_two_years_ago : ℕ := num_flavors_total / 4

def num_flavors_last_year : ℕ := 2 * num_flavors_two_years_ago

def num_flavors_tried_so_far : ℕ := num_flavors_two_years_ago + num_flavors_last_year

theorem flavors_needed_this_year : 
  (num_flavors_total - num_flavors_tried_so_far) = 25 := by {
  sorry
}

end flavors_needed_this_year_l145_145783


namespace sum_of_two_numbers_l145_145838

theorem sum_of_two_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 23 :=
by
  sorry

end sum_of_two_numbers_l145_145838


namespace new_computer_price_l145_145623

-- Define the initial conditions
def initial_price_condition (x : ℝ) : Prop := 2 * x = 540

-- Define the calculation for the new price after a 30% increase
def new_price (x : ℝ) : ℝ := x * 1.30

-- Define the final proof problem statement
theorem new_computer_price : ∃ x : ℝ, initial_price_condition x ∧ new_price x = 351 :=
by sorry

end new_computer_price_l145_145623


namespace area_of_region_l145_145941

theorem area_of_region (x y : ℝ) : |4 * x - 24| + |3 * y + 10| ≤ 6 → ∃ A : ℝ, A = 12 :=
by
  sorry

end area_of_region_l145_145941


namespace complex_multiplication_l145_145875

def i := Complex.I

theorem complex_multiplication (i := Complex.I) : (-1 + i) * (2 - i) = -1 + 3 * i := 
by 
    -- The actual proof steps would go here.
    sorry

end complex_multiplication_l145_145875


namespace lcm_of_two_numbers_l145_145939

-- Definitions based on the conditions
variable (a b l : ℕ)

-- The conditions from the problem
def hcf_ab : Nat := 9
def prod_ab : Nat := 1800

-- The main statement to prove
theorem lcm_of_two_numbers : Nat.lcm a b = 200 :=
by
  -- Skipping the proof implementation
  sorry

end lcm_of_two_numbers_l145_145939


namespace sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one_l145_145972

-- Defining the terms and the theorem
theorem sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one :
  (Real.sqrt 5 - 2) ^ 2023 * (Real.sqrt 5 + 2) ^ 2023 = 1 := 
by
  sorry

end sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one_l145_145972


namespace matrix_inverse_l145_145643

variable (N : Matrix (Fin 2) (Fin 2) ℚ) 
variable (I : Matrix (Fin 2) (Fin 2) ℚ)
variable (c d : ℚ)

def M1 : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]

def M2 : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]

theorem matrix_inverse (hN : N = M1) 
                       (hI : I = M2) 
                       (hc : c = 1/12) 
                       (hd : d = 1/12) :
                       N⁻¹ = c • N + d • I := by
  sorry

end matrix_inverse_l145_145643


namespace batsman_average_after_25th_innings_l145_145396

theorem batsman_average_after_25th_innings (A : ℝ) (runs_25th : ℝ) (increase : ℝ) (not_out_innings : ℕ) 
    (total_innings : ℕ) (average_increase_condition : 24 * A + runs_25th = 25 * (A + increase)) :       
    runs_25th = 150 ∧ increase = 3 ∧ not_out_innings = 3 ∧ total_innings = 25 → 
    ∃ avg : ℝ, avg = 88.64 := by 
  sorry

end batsman_average_after_25th_innings_l145_145396


namespace find_set_C_l145_145948

def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}
def C : Set ℝ := {a | B a ⊆ A}

theorem find_set_C : C = {0, 1, 2} :=
by
  sorry

end find_set_C_l145_145948


namespace remaining_cube_height_l145_145671

/-- Given a cube with side length 2 units, where a corner is chopped off such that the cut runs
    through points on the three edges adjacent to a selected vertex, each at 1 unit distance
    from that vertex, the height of the remaining portion of the cube when the freshly cut face 
    is placed on a table is equal to (5 * sqrt 3) / 3. -/
theorem remaining_cube_height (s : ℝ) (h : ℝ) : 
    s = 2 → h = 1 → 
    ∃ height : ℝ, height = (5 * Real.sqrt 3) / 3 := 
by
    sorry

end remaining_cube_height_l145_145671


namespace ray_has_4_nickels_left_l145_145077

theorem ray_has_4_nickels_left (initial_cents : ℕ) (given_to_peter : ℕ)
    (given_to_randi : ℕ) (value_of_nickel : ℕ) (remaining_cents : ℕ) 
    (remaining_nickels : ℕ) :
    initial_cents = 95 →
    given_to_peter = 25 →
    given_to_randi = 2 * given_to_peter →
    value_of_nickel = 5 →
    remaining_cents = initial_cents - given_to_peter - given_to_randi →
    remaining_nickels = remaining_cents / value_of_nickel →
    remaining_nickels = 4 :=
by
  intros
  sorry

end ray_has_4_nickels_left_l145_145077


namespace pq_sum_l145_145737

theorem pq_sum (p q : ℝ) 
  (h1 : p / 3 = 9) 
  (h2 : q / 3 = 15) : 
  p + q = 72 :=
sorry

end pq_sum_l145_145737


namespace rect_plot_length_more_than_breadth_l145_145101

theorem rect_plot_length_more_than_breadth (b x : ℕ) (cost_per_m : ℚ)
  (length_eq : b + x = 56)
  (fencing_cost : (4 * b + 2 * x) * cost_per_m = 5300)
  (cost_rate : cost_per_m = 26.50) : x = 12 :=
by
  sorry

end rect_plot_length_more_than_breadth_l145_145101


namespace Cody_money_final_l145_145732

-- Define the initial amount of money Cody had
def Cody_initial : ℝ := 45.0

-- Define the birthday gift amount
def birthday_gift : ℝ := 9.0

-- Define the amount spent on the game
def game_expense : ℝ := 19.0

-- Define the percentage of remaining money spent on clothes as a fraction
def clothes_spending_fraction : ℝ := 0.40

-- Define the late birthday gift received
def late_birthday_gift : ℝ := 4.5

-- Define the final amount of money Cody has
def Cody_final : ℝ :=
  let after_birthday := Cody_initial + birthday_gift
  let after_game := after_birthday - game_expense
  let spent_on_clothes := clothes_spending_fraction * after_game
  let after_clothes := after_game - spent_on_clothes
  after_clothes + late_birthday_gift

theorem Cody_money_final : Cody_final = 25.5 := by
  sorry

end Cody_money_final_l145_145732


namespace x_plus_y_l145_145738

noncomputable def sequence_constant : ℝ := 1 / 4

def x : ℝ := 256 * sequence_constant

def y : ℝ := x * sequence_constant

theorem x_plus_y : x + y = 80 := 
by
  -- Placeholder for the proof
  sorry

end x_plus_y_l145_145738


namespace min_value_of_expression_l145_145338

open Real

noncomputable def minValue (x y z : ℝ) : ℝ :=
  x + 3 * y + 5 * z

theorem min_value_of_expression : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 8 → minValue x y z = 14.796 :=
by
  intros x y z h
  sorry

end min_value_of_expression_l145_145338


namespace John_cycles_distance_l145_145638

-- Define the rate and time as per the conditions in the problem
def rate : ℝ := 8 -- miles per hour
def time : ℝ := 2.25 -- hours

-- The mathematical statement to prove: distance = rate * time
theorem John_cycles_distance : rate * time = 18 := by
  sorry

end John_cycles_distance_l145_145638


namespace max_sum_non_zero_nats_l145_145186

theorem max_sum_non_zero_nats (O square : ℕ) (hO : O ≠ 0) (hsquare : square ≠ 0) :
  (O / 11 < 7 / square) ∧ (7 / square < 4 / 5) → O + square = 77 :=
by 
  sorry -- Proof omitted as requested

end max_sum_non_zero_nats_l145_145186


namespace angle_A_equals_pi_div_3_min_AD_value_l145_145052

-- Define the triangle ABC with sides a, b, c, and angles A, B, C
variable (a b c : ℝ)
variable (A B C : ℝ)

-- Define the conditions provided in the problem
axiom cond1 : a * real.sin B = b * real.sin (A + π / 3)
axiom cond2 : S = (sqrt 3 / 2) * (∥BA∥ * ∥CA∥)
axiom cond3 : c * real.tan A = (2 * b - c) * real.tan C

-- Problem definition in Lean:
theorem angle_A_equals_pi_div_3 (h : cond1) : A = π / 3 := sorry

-- Additional geometric conditions for the second part of the problem
variable (S : ℝ)
axiom area_condition : S = 2 * sqrt 3

-- Given point D on BC such that BD = 2DC
variable (BD DC AD : ℝ)
axiom BD_double_DC : BD = 2 * DC

-- Prove the minimum value of AD
theorem min_AD_value (h1 : area_condition) (h2 : BD_double_DC) : AD = 4 * sqrt 3 / 3 := sorry

end angle_A_equals_pi_div_3_min_AD_value_l145_145052


namespace simplify_fraction_l145_145664

theorem simplify_fraction : 
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1))) = 
  ((Real.sqrt 3) + 2 * (Real.sqrt 5) - 1) / (2 + 4 * Real.sqrt 5) := 
by 
  sorry

end simplify_fraction_l145_145664


namespace angle_C_is_150_degrees_l145_145971

theorem angle_C_is_150_degrees
  (C D : ℝ)
  (h_supp : C + D = 180)
  (h_C_5D : C = 5 * D) :
  C = 150 :=
by
  sorry

end angle_C_is_150_degrees_l145_145971


namespace graph_avoid_third_quadrant_l145_145192

theorem graph_avoid_third_quadrant (k : ℝ) : 
  (∀ x y : ℝ, y = (2 * k - 1) * x + k → ¬ (x < 0 ∧ y < 0)) ↔ 0 ≤ k ∧ k < (1 / 2) :=
by sorry

end graph_avoid_third_quadrant_l145_145192


namespace units_produced_today_eq_90_l145_145033

-- Define the average production and number of past days
def average_past_production (n : ℕ) (past_avg : ℕ) : ℕ :=
  n * past_avg

def average_total_production (n : ℕ) (current_avg : ℕ) : ℕ :=
  (n + 1) * current_avg

def units_produced_today (n : ℕ) (past_avg : ℕ) (current_avg : ℕ) : ℕ :=
  average_total_production n current_avg - average_past_production n past_avg

-- Given conditions
def n := 5
def past_avg := 60
def current_avg := 65

-- Statement to prove
theorem units_produced_today_eq_90 : units_produced_today n past_avg current_avg = 90 :=
by
  -- Declare which parts need proving
  sorry

end units_produced_today_eq_90_l145_145033


namespace benedict_house_size_l145_145470

variable (K B : ℕ)

theorem benedict_house_size
    (h1 : K = 4 * B + 600)
    (h2 : K = 10000) : B = 2350 := by
sorry

end benedict_house_size_l145_145470


namespace p_q_work_l145_145837

theorem p_q_work (p_rate q_rate : ℝ) (h1: 1 / p_rate + 1 / q_rate = 1 / 6) (h2: p_rate = 15) : q_rate = 10 :=
by
  sorry

end p_q_work_l145_145837


namespace fenced_area_with_cutout_l145_145088

def rectangle_area (length width : ℝ) : ℝ := length * width

def square_area (side : ℝ) : ℝ := side * side

theorem fenced_area_with_cutout :
  rectangle_area 20 18 - square_area 4 = 344 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end fenced_area_with_cutout_l145_145088


namespace negate_proposition_l145_145172

theorem negate_proposition :
  (¬ ∃ (x₀ : ℝ), x₀^2 + 2 * x₀ + 3 ≤ 0) ↔ (∀ (x : ℝ), x^2 + 2 * x + 3 > 0) :=
by
  sorry

end negate_proposition_l145_145172


namespace max_possible_n_l145_145466

theorem max_possible_n :
  ∃ (n : ℕ), (n < 150) ∧ (∃ (k l : ℤ), n = 9 * k - 1 ∧ n = 6 * l - 5 ∧ n = 125) :=
by 
  sorry

end max_possible_n_l145_145466


namespace fenced_area_l145_145090

theorem fenced_area (w : ℕ) (h : ℕ) (cut_out : ℕ) (rectangle_area : ℕ) (cut_out_area : ℕ) (net_area : ℕ) :
  w = 20 → h = 18 → cut_out = 4 → rectangle_area = w * h → cut_out_area = cut_out * cut_out → net_area = rectangle_area - cut_out_area → net_area = 344 :=
by
  intros
  subst_vars
  sorry

end fenced_area_l145_145090


namespace bill_cooking_time_l145_145558

def total_time_spent 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ) : ℕ :=
num_peppers * chop_pepper_time + 
num_onions * chop_onion_time + 
num_omelets * grate_cheese_time + 
num_omelets * cook_omelet_time

theorem bill_cooking_time 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ)
  (chop_pepper_time_eq : chop_pepper_time = 3)
  (chop_onion_time_eq : chop_onion_time = 4)
  (grate_cheese_time_eq : grate_cheese_time = 1)
  (cook_omelet_time_eq : cook_omelet_time = 5)
  (num_peppers_eq : num_peppers = 4)
  (num_onions_eq : num_onions = 2)
  (num_omelets_eq : num_omelets = 5) :
  total_time_spent chop_pepper_time chop_onion_time grate_cheese_time cook_omelet_time num_peppers num_onions num_omelets = 50 :=
by {
  sorry
}

end bill_cooking_time_l145_145558


namespace cost_per_sqft_is_6_l145_145672

-- Define the dimensions of the room
def room_length : ℕ := 25
def room_width : ℕ := 15
def room_height : ℕ := 12

-- Define the dimensions of the door
def door_height : ℕ := 6
def door_width : ℕ := 3

-- Define the dimensions of the windows
def window_height : ℕ := 4
def window_width : ℕ := 3
def number_of_windows : ℕ := 3

-- Define the total cost of whitewashing
def total_cost : ℕ := 5436

-- Calculate areas
def area_one_pair_of_walls : ℕ :=
  (room_length * room_height) * 2

def area_other_pair_of_walls : ℕ :=
  (room_width * room_height) * 2

def total_wall_area : ℕ :=
  area_one_pair_of_walls + area_other_pair_of_walls

def door_area : ℕ :=
  door_height * door_width

def window_area : ℕ :=
  window_height * window_width

def total_window_area : ℕ :=
  window_area * number_of_windows

def area_to_be_whitewashed : ℕ :=
  total_wall_area - (door_area + total_window_area)

def cost_per_sqft : ℕ :=
  total_cost / area_to_be_whitewashed

-- The theorem statement proving the cost per square foot is 6
theorem cost_per_sqft_is_6 : cost_per_sqft = 6 := 
  by
  -- Proof goes here
  sorry

end cost_per_sqft_is_6_l145_145672


namespace remainder_of_large_number_div_by_101_l145_145690

theorem remainder_of_large_number_div_by_101 :
  2468135792 % 101 = 52 :=
by
  sorry

end remainder_of_large_number_div_by_101_l145_145690


namespace quadratic_solution_m_l145_145450

theorem quadratic_solution_m (m : ℝ) : (x = 2) → (x^2 - m*x + 8 = 0) → (m = 6) := 
by
  sorry

end quadratic_solution_m_l145_145450


namespace balloon_difference_l145_145840

theorem balloon_difference (your_balloons : ℕ) (friend_balloons : ℕ) (h1 : your_balloons = 7) (h2 : friend_balloons = 5) : your_balloons - friend_balloons = 2 :=
by
  sorry

end balloon_difference_l145_145840


namespace product_sum_125_l145_145023

theorem product_sum_125 :
  ∀ (m n : ℕ), m ≥ n ∧
              (∀ (k : ℕ), 0 < k → |Real.log m - Real.log k| < Real.log n → k ≠ 0)
              → (m * n = 125) :=
by sorry

end product_sum_125_l145_145023


namespace problem1_problem2_l145_145560

theorem problem1 : ( (2 / 3 - 1 / 4 - 5 / 6) * 12 = -5 ) :=
by sorry

theorem problem2 : ( (-3)^2 * 2 + 4 * (-3) - 28 / (7 / 4) = -10 ) :=
by sorry

end problem1_problem2_l145_145560


namespace freshman_count_630_l145_145930

-- Define the variables and conditions
variables (f o j s : ℕ)
variable (total_students : ℕ)

-- Define the ratios given in the problem
def freshmen_to_sophomore : Prop := f = (5 * o) / 4
def sophomore_to_junior : Prop := j = (8 * o) / 7
def junior_to_senior : Prop := s = (7 * j) / 9

-- Total number of students condition
def total_students_condition : Prop := f + o + j + s = total_students

theorem freshman_count_630
  (h1 : freshmen_to_sophomore f o)
  (h2 : sophomore_to_junior o j)
  (h3 : junior_to_senior j s)
  (h4 : total_students_condition f o j s 2158) :
  f = 630 :=
sorry

end freshman_count_630_l145_145930


namespace geometric_sequence_property_l145_145926

-- Define the sequence and the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the main property we are considering
def given_property (a: ℕ → ℝ) (n : ℕ) : Prop :=
  a (n + 1) * a (n - 1) = (a n) ^ 2

-- State the theorem
theorem geometric_sequence_property {a : ℕ → ℝ} (n : ℕ) (hn : n ≥ 2) :
  (is_geometric_sequence a → given_property a n ∧ ∀ a, given_property a n → ¬ is_geometric_sequence a) := sorry

end geometric_sequence_property_l145_145926


namespace number_of_children_is_five_l145_145368

/-- The sum of the ages of children born at intervals of 2 years each is 50 years, 
    and the age of the youngest child is 6 years.
    Prove that the number of children is 5. -/
theorem number_of_children_is_five (n : ℕ) (h1 : (0 < n ∧ n / 2 * (8 + 2 * n) = 50)): n = 5 :=
sorry

end number_of_children_is_five_l145_145368


namespace max_num_ones_l145_145491

theorem max_num_ones (S : Finset ℕ) (hS : S = (Finset.range 2015) \ {0}) (op : ℕ → ℕ → ℕ × ℕ) :
  op = (λ a b, (Nat.gcd a b, Nat.lcm a b)) →
  ∃ n, n = 1007 ∧
  ∀ S', (∀ a b ∈ S', (a, b) = S'.elems → op a b ∈ S') →
    Finset.card (S'.filter (λ x, x = 1)) = n := 
by sorry

end max_num_ones_l145_145491


namespace calculate_expression_l145_145001

theorem calculate_expression :
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) = 3^128 - 5^128 :=
by
  sorry

end calculate_expression_l145_145001


namespace maddie_total_cost_l145_145219

theorem maddie_total_cost :
  let price_palette := 15
  let price_lipstick := 2.5
  let price_hair_color := 4
  let num_palettes := 3
  let num_lipsticks := 4
  let num_hair_colors := 3
  let total_cost := (num_palettes * price_palette) + (num_lipsticks * price_lipstick) + (num_hair_colors * price_hair_color)
  total_cost = 67 := by
  sorry

end maddie_total_cost_l145_145219


namespace find_f_neg_19_div_3_l145_145046

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 1 then 
    8^x 
  else 
    sorry -- The full definition is complex and not needed for the statement

-- Define the properties of f
lemma f_periodic (x : ℝ) : f (x + 2) = f x := 
  sorry

lemma f_odd (x : ℝ) : f (-x) = -f x := 
  sorry

theorem find_f_neg_19_div_3 : f (-19/3) = -2 :=
  sorry

end find_f_neg_19_div_3_l145_145046


namespace total_stars_l145_145722

theorem total_stars (g s : ℕ) (hg : g = 10^11) (hs : s = 10^11) : g * s = 10^22 :=
by
  rw [hg, hs]
  sorry

end total_stars_l145_145722


namespace fenced_area_with_cutout_l145_145086

def rectangle_area (length width : ℝ) : ℝ := length * width

def square_area (side : ℝ) : ℝ := side * side

theorem fenced_area_with_cutout :
  rectangle_area 20 18 - square_area 4 = 344 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end fenced_area_with_cutout_l145_145086


namespace volume_original_cone_l145_145545

-- Given conditions
def V_cylinder : ℝ := 21
def V_truncated_cone : ℝ := 91

-- To prove: The volume of the original cone is 94.5
theorem volume_original_cone : 
    (∃ (H R h r : ℝ), (π * r^2 * h = V_cylinder) ∧ (1 / 3 * π * (R^2 + R * r + r^2) * (H - h) = V_truncated_cone)) →
    (1 / 3 * π * R^2 * H = 94.5) :=
by
  sorry

end volume_original_cone_l145_145545


namespace find_x_l145_145433

def f (x : ℝ) := 2 * x - 3

theorem find_x : ∃ x, 2 * (f x) - 11 = f (x - 2) ∧ x = 5 :=
by 
  unfold f
  exists 5
  sorry

end find_x_l145_145433


namespace curve_intersection_three_points_l145_145166

theorem curve_intersection_three_points (a : ℝ) :
  (∀ x y : ℝ, ((x^2 - y^2 = a^2) ∧ ((x-1)^2 + y^2 = 1)) → (a = 0)) :=
by
  sorry

end curve_intersection_three_points_l145_145166


namespace evaluate_fractions_l145_145291

-- Define the fractions
def frac1 := 7 / 12
def frac2 := 8 / 15
def frac3 := 2 / 5

-- Prove that the sum and difference is as specified
theorem evaluate_fractions :
  frac1 + frac2 - frac3 = 43 / 60 :=
by
  sorry

end evaluate_fractions_l145_145291


namespace new_person_weight_l145_145265

theorem new_person_weight
  (avg_increase : ℝ) (original_person_weight : ℝ) (num_people : ℝ) (new_weight : ℝ)
  (h1 : avg_increase = 2.5)
  (h2 : original_person_weight = 85)
  (h3 : num_people = 8)
  (h4 : num_people * avg_increase = new_weight - original_person_weight):
    new_weight = 105 :=
by
  sorry

end new_person_weight_l145_145265


namespace prob_B_independent_l145_145252

-- Definitions based on the problem's conditions
def prob_A := 0.7
def prob_A_union_B := 0.94

-- With these definitions established, we need to state the theorem.
-- The theorem should express that the probability of B solving the problem independently (prob_B) is 0.8.
theorem prob_B_independent : 
    (∃ (prob_B: ℝ), prob_A = 0.7 ∧ prob_A_union_B = 0.94 ∧ prob_B = 0.8) :=
by
    sorry

end prob_B_independent_l145_145252


namespace find_g_2_l145_145824

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_eq (x : ℝ) (hx : x ≠ 0) : 2 * g x - 3 * g (1 / x) = x ^ 2

theorem find_g_2 : g 2 = 8.25 :=
by {
  sorry
}

end find_g_2_l145_145824


namespace circle_equation_condition_l145_145190

theorem circle_equation_condition (m : ℝ) : 
  (∃ h k r : ℝ, (r > 0) ∧ ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 → x^2 + y^2 - 2*x - 4*y + m = 0) ↔ m < 5 :=
sorry

end circle_equation_condition_l145_145190


namespace lottery_ticket_random_event_l145_145239

-- Define the type of possible outcomes of buying a lottery ticket
inductive LotteryOutcome
| Win
| Lose

-- Define the random event condition
def is_random_event (outcome: LotteryOutcome) : Prop :=
  match outcome with
  | LotteryOutcome.Win => True
  | LotteryOutcome.Lose => True

-- The theorem to prove that buying 1 lottery ticket and winning is a random event
theorem lottery_ticket_random_event : is_random_event LotteryOutcome.Win :=
by
  sorry

end lottery_ticket_random_event_l145_145239


namespace series_value_is_correct_l145_145802

noncomputable def check_series_value : ℚ :=
  let p : ℚ := 1859 / 84
  let q : ℚ := -1024 / 63
  let r : ℚ := 512 / 63
  let m : ℕ := 3907
  let n : ℕ := 84
  100 * m + n

theorem series_value_is_correct : check_series_value = 390784 := 
by 
  sorry

end series_value_is_correct_l145_145802


namespace exists_a_satisfying_f_l145_145324

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 1 else x - 1

theorem exists_a_satisfying_f (a : ℝ) : 
  f (a + 1) = f a ↔ (a = -1/2 ∨ a = (-1 + Real.sqrt 5) / 2) :=
by
  sorry

end exists_a_satisfying_f_l145_145324


namespace speed_of_man_in_still_water_correct_l145_145261

def upstream_speed : ℝ := 25 -- Upstream speed in kmph
def downstream_speed : ℝ := 39 -- Downstream speed in kmph
def speed_in_still_water : ℝ := 32 -- The speed of the man in still water

theorem speed_of_man_in_still_water_correct :
  (upstream_speed + downstream_speed) / 2 = speed_in_still_water :=
by
  sorry

end speed_of_man_in_still_water_correct_l145_145261


namespace find_c_l145_145985

noncomputable def P (c : ℝ) : Polynomial ℝ := 3 * (Polynomial.X ^ 3) + c * (Polynomial.X ^ 2) - 8 * (Polynomial.X) + 50
def D : Polynomial ℝ := 3 * (Polynomial.X) + 5

theorem find_c (c : ℝ) :
  Polynomial.mod_by_monic (P c) (Polynomial.X - Polynomial.C (-5/3)) = Polynomial.C 7 → c = 18/25 :=
sorry

end find_c_l145_145985


namespace Yanna_kept_apples_l145_145382

theorem Yanna_kept_apples (initial_apples : ℕ) (apples_given_Zenny : ℕ) (apples_given_Andrea : ℕ) :
  initial_apples = 60 → apples_given_Zenny = 18 → apples_given_Andrea = 6 →
  (initial_apples - (apples_given_Zenny + apples_given_Andrea) = 36) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rw [Nat.add_sub_assoc]
  exact rfl
  apply Nat.succ_le_succ
  exact nat.succ_pos'


end Yanna_kept_apples_l145_145382


namespace sequence_formula_l145_145062

-- Define the sequence a_n using the recurrence relation
def a : ℕ → ℚ
| 0     := 0
| (n+1) := a n + (n+1)^3

-- The statement to be proved
theorem sequence_formula (n : ℕ) : 
  a n = (n^2 * (n+1)^2) / 4 := sorry

end sequence_formula_l145_145062


namespace num_of_factorizable_poly_l145_145007

theorem num_of_factorizable_poly : 
  ∃ (n : ℕ), (1 ≤ n ∧ n ≤ 2023) ∧ 
              (∃ (a : ℤ), n = a * (a + 1)) :=
sorry

end num_of_factorizable_poly_l145_145007


namespace f_is_odd_l145_145677

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2 * x

-- State the problem
theorem f_is_odd :
  ∀ x : ℝ, f (-x) = -f x := 
by
  sorry

end f_is_odd_l145_145677


namespace sales_tax_difference_l145_145853

theorem sales_tax_difference :
  let price : ℝ := 50
  let tax_rate1 : ℝ := 0.075
  let tax_rate2 : ℝ := 0.07
  (price * tax_rate1) - (price * tax_rate2) = 0.25 := by
  sorry

end sales_tax_difference_l145_145853


namespace measure_of_angle_C_l145_145968

theorem measure_of_angle_C (A B : ℝ) (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := by
  sorry

end measure_of_angle_C_l145_145968


namespace ellipse_area_proof_l145_145178

noncomputable def right_focus_of_ellipse (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c : ℝ) : Prop :=
  c = (2 : ℝ) * Real.sqrt 2

noncomputable def ellipse_eq (a b : ℝ) (pt : ℝ × ℝ) :=
  pt.1 ^ 2 / a ^ 2 + pt.2 ^ 2 / b ^ 2 = 1

theorem ellipse_area_proof 
  (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (h_focus : right_focus_of_ellipse a b (2 * Real.sqrt 2))
  (pt : ℝ × ℝ)
  (h_pass : ellipse_eq a b (3, 1))
  (slope : ℝ)
  (P : ℝ × ℝ)
  (vertex_P : P = (-3, 2)) :
  (∃ (a b : ℝ), ellipse_eq 12 4 (3, 1)) ∧ (area_triangle P A B = 9 / 2) :=
sorry

end ellipse_area_proof_l145_145178


namespace marble_ratio_l145_145154

-- Definitions based on conditions
def dan_marbles : ℕ := 5
def mary_marbles : ℕ := 10

-- Statement of the theorem to prove the ratio is 2:1
theorem marble_ratio : mary_marbles / dan_marbles = 2 := by
  sorry

end marble_ratio_l145_145154


namespace intersection_of_complements_l145_145217

-- Define the universal set U as a natural set with numbers <= 8
def U : Set ℕ := { x | x ≤ 8 }

-- Define the set A
def A : Set ℕ := { 1, 3, 7 }

-- Define the set B
def B : Set ℕ := { 2, 3, 8 }

-- Prove the statement for the intersection of the complements of A and B with respect to U
theorem intersection_of_complements : 
  ((U \ A) ∩ (U \ B)) = ({ 0, 4, 5, 6 } : Set ℕ) :=
by
  sorry

end intersection_of_complements_l145_145217


namespace tuples_satisfy_equation_l145_145734

theorem tuples_satisfy_equation (a b c : ℤ) :
  (a - b)^3 * (a + b)^2 = c^2 + 2 * (a - b) + 1 ↔ (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) :=
sorry

end tuples_satisfy_equation_l145_145734


namespace functional_equation_solution_l145_145006

noncomputable def f : ℕ → ℕ := sorry

theorem functional_equation_solution (f : ℕ → ℕ)
    (h : ∀ n : ℕ, f (f (f n)) + f (f n) + f n = 3 * n) :
    ∀ n : ℕ, f n = n := sorry

end functional_equation_solution_l145_145006


namespace A_rotated_l145_145073

-- Define initial coordinates of point A
def A_initial : ℝ × ℝ := (1, 2)

-- Define the transformation for a 180-degree clockwise rotation around the origin
def rotate_180_deg (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- The Lean statement to prove the coordinates after the rotation
theorem A_rotated : rotate_180_deg A_initial = (-1, -2) :=
by
  sorry

end A_rotated_l145_145073


namespace weight_of_B_l145_145706

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 :=
by sorry

end weight_of_B_l145_145706


namespace min_rounds_needed_l145_145326

-- Defining the number of players
def num_players : ℕ := 10

-- Defining the number of matches each player plays per round
def matches_per_round (n : ℕ) : ℕ := n / 2

-- Defining the scoring system
def win_points : ℝ := 1
def draw_points : ℝ := 0.5
def loss_points : ℝ := 0

-- Defining the total number of rounds needed for a clear winner to emerge
def min_rounds_for_winner : ℕ := 7

-- Theorem stating the minimum number of rounds required
theorem min_rounds_needed :
  ∀ (n : ℕ), n = num_players → (∃ r : ℕ, r = min_rounds_for_winner) :=
by
  intros n hn
  existsi min_rounds_for_winner
  sorry

end min_rounds_needed_l145_145326


namespace pythagorean_theorem_l145_145253

theorem pythagorean_theorem (a b c : ℝ) (h : a^2 + b^2 = c^2) : c^2 = a^2 + b^2 :=
sorry

end pythagorean_theorem_l145_145253


namespace Monica_tiles_count_l145_145224

noncomputable def total_tiles (length width : ℕ) := 
  let double_border_tiles := (2 * ((length - 4) + (width - 4)) + 8)
  let inner_area := (length - 4) * (width - 4)
  let three_foot_tiles := (inner_area + 8) / 9
  double_border_tiles + three_foot_tiles

theorem Monica_tiles_count : total_tiles 18 24 = 183 := 
by
  sorry

end Monica_tiles_count_l145_145224


namespace urea_formation_l145_145588

theorem urea_formation
  (CO2 NH3 Urea : ℕ) 
  (h_CO2 : CO2 = 1)
  (h_NH3 : NH3 = 2) :
  Urea = 1 := by
  sorry

end urea_formation_l145_145588


namespace find_k_for_circle_radius_l145_145871

theorem find_k_for_circle_radius (k : ℝ) :
  (∃ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ∧ (x + 7)^2 + (y + 4)^2 = 10^2) ↔ k = 35 :=
by
  sorry

end find_k_for_circle_radius_l145_145871


namespace machine_a_produces_18_sprockets_per_hour_l145_145341

theorem machine_a_produces_18_sprockets_per_hour :
  ∃ (A : ℝ), (∀ (B C : ℝ),
  B = 1.10 * A ∧
  B = 1.20 * C ∧
  990 / A = 990 / B + 10 ∧
  990 / C = 990 / A - 5) →
  A = 18 :=
by { sorry }

end machine_a_produces_18_sprockets_per_hour_l145_145341


namespace pentagon_product_condition_l145_145769

theorem pentagon_product_condition :
  ∃ (a b c d e : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ a + b + c + d + e = 1 ∧
  ∃ (a' b' c' d' e' : ℝ), 
    (a', b', c', d', e') ∈ {perm | perm = (a, b, c, d, e) ∨ perm = (b, c, d, e, a) ∨ perm = (c, d, e, a, b) ∨ perm = (d, e, a, b, c) ∨ perm = (e, a, b, c, d)} ∧
    (a'*b' ≤ 1/9 ∧ b'*c' ≤ 1/9 ∧ c'*d' ≤ 1/9 ∧ d'*e' ≤ 1/9 ∧ e'*a' ≤ 1/9) := sorry

end pentagon_product_condition_l145_145769


namespace hockey_team_helmets_l145_145139

theorem hockey_team_helmets (r b : ℕ) 
  (h1 : b = r - 6) 
  (h2 : r * 3 = b * 5) : 
  r + b = 24 :=
by
  sorry

end hockey_team_helmets_l145_145139


namespace intersection_of_M_and_N_l145_145181

def M : Set ℤ := { x | -3 < x ∧ x < 3 }
def N : Set ℤ := { x | x < 1 }

theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := by
  sorry

end intersection_of_M_and_N_l145_145181


namespace other_root_of_quadratic_l145_145456

theorem other_root_of_quadratic (a : ℝ) :
  (∀ x, (x^2 + 2*x - a) = 0 → x = -3) → (∃ z, z = 1 ∧ (z^2 + 2*z - a) = 0) :=
by
  sorry

end other_root_of_quadratic_l145_145456


namespace SandySpentTotal_l145_145080

theorem SandySpentTotal :
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 := by
  sorry

end SandySpentTotal_l145_145080


namespace smallest_m_for_probability_l145_145659

-- Define the conditions in Lean
def nonWithInTwoUnits (x y z : ℝ) : Prop :=
  abs (x - y) ≥ 2 ∧ abs (y - z) ≥ 2 ∧ abs (z - x) ≥ 2

def probabilityCondition (m : ℝ) : Prop :=
  (m - 4)^3 / m^3 > 2/3

-- The theorem statement
theorem smallest_m_for_probability : ∃ m : ℕ, 0 < m ∧ (∀ x y z : ℝ, 0 ≤ x ∧ x ≤ m ∧ 0 ≤ y ∧ y ≤ m ∧ 0 ≤ z ∧ z ≤ m → nonWithInTwoUnits x y z) → probabilityCondition m ∧ m = 14 :=
by sorry

end smallest_m_for_probability_l145_145659


namespace cats_to_dogs_l145_145508

theorem cats_to_dogs (c d : ℕ) (h1 : c = 24) (h2 : 4 * d = 5 * c) : d = 30 :=
by
  sorry

end cats_to_dogs_l145_145508


namespace fenced_area_l145_145091

theorem fenced_area (w : ℕ) (h : ℕ) (cut_out : ℕ) (rectangle_area : ℕ) (cut_out_area : ℕ) (net_area : ℕ) :
  w = 20 → h = 18 → cut_out = 4 → rectangle_area = w * h → cut_out_area = cut_out * cut_out → net_area = rectangle_area - cut_out_area → net_area = 344 :=
by
  intros
  subst_vars
  sorry

end fenced_area_l145_145091


namespace perpendicular_vectors_m_solution_l145_145874

theorem perpendicular_vectors_m_solution (m : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (m, 1)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : m = -2 := by
  sorry

end perpendicular_vectors_m_solution_l145_145874


namespace sufficient_not_necessary_condition_l145_145709

noncomputable def has_negative_root (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x < 0

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∃ x : ℝ, (a * x^2 + 2 * x + 1 = 0) ∧ x < 0) ↔ (a < 0) :=
sorry

end sufficient_not_necessary_condition_l145_145709


namespace correct_system_of_equations_l145_145629

theorem correct_system_of_equations (x y : ℕ) :
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔
  (∃ x y, (x / 3 = y - 2) ∧ ((x - 9) / 2 = y)) :=
by
  sorry

end correct_system_of_equations_l145_145629


namespace num_five_digit_integers_l145_145518

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

theorem num_five_digit_integers : 
  let num_ways := factorial 5 / (factorial 2 * factorial 3)
  num_ways = 10 :=
by 
  sorry

end num_five_digit_integers_l145_145518


namespace foci_distance_l145_145163

variable (x y : ℝ)

def ellipse_eq : Prop := (x^2 / 45) + (y^2 / 5) = 9

theorem foci_distance : ellipse_eq x y → (distance_between_foci : ℝ) = 12 * Real.sqrt 10 :=
by
  sorry

end foci_distance_l145_145163


namespace rectangle_area_l145_145848

theorem rectangle_area (x y : ℝ) (L W : ℝ) (h_diagonal : (L ^ 2 + W ^ 2) ^ (1 / 2) = x + y) (h_ratio : L / W = 3 / 2) : 
  L * W = (6 * (x + y) ^ 2) / 13 := 
sorry

end rectangle_area_l145_145848


namespace length_of_metallic_sheet_l145_145955

variable (L : ℝ) (width side volume : ℝ)

theorem length_of_metallic_sheet (h1 : width = 36) (h2 : side = 8) (h3 : volume = 5120) :
  ((L - 2 * side) * (width - 2 * side) * side = volume) → L = 48 := 
by
  intros h_eq
  sorry

end length_of_metallic_sheet_l145_145955


namespace product_of_tangents_l145_145563

theorem product_of_tangents : 
  (Real.tan (Real.pi / 8) * Real.tan (3 * Real.pi / 8) * 
   Real.tan (5 * Real.pi / 8) * Real.tan (7 * Real.pi / 8) = -2 * Real.sqrt 2) :=
sorry

end product_of_tangents_l145_145563


namespace boys_cannot_score_twice_as_girls_l145_145197

theorem boys_cannot_score_twice_as_girls :
  ∀ (participants : Finset ℕ) (boys girls : ℕ) (points : ℕ → ℝ),
    participants.card = 6 →
    boys = 2 →
    girls = 4 →
    (∀ p, p ∈ participants → points p = 1 ∨ points p = 0.5 ∨ points p = 0) →
    (∀ (p q : ℕ), p ∈ participants → q ∈ participants → p ≠ q → points p + points q = 1) →
    ¬ (∃ (boys_points girls_points : ℝ), 
      (∀ b ∈ (Finset.range 2), boys_points = points b) ∧
      (∀ g ∈ (Finset.range 4), girls_points = points g) ∧
      boys_points = 2 * girls_points) :=
by
  sorry

end boys_cannot_score_twice_as_girls_l145_145197


namespace sqrt_nat_or_irrational_l145_145484

theorem sqrt_nat_or_irrational {n : ℕ} : 
  (∃ m : ℕ, m^2 = n) ∨ (¬ ∃ q r : ℕ, r ≠ 0 ∧ (q^2 = n * r^2 ∧ r * r ≠ n * n)) :=
sorry

end sqrt_nat_or_irrational_l145_145484


namespace gcf_factorial_5_6_l145_145021

theorem gcf_factorial_5_6 : Nat.gcd (Nat.factorial 5) (Nat.factorial 6) = Nat.factorial 5 := by
  sorry

end gcf_factorial_5_6_l145_145021


namespace range_of_m_in_third_quadrant_l145_145329

theorem range_of_m_in_third_quadrant (m : ℝ) : (1 - (1/3) * m < 0) ∧ (m - 5 < 0) ↔ (3 < m ∧ m < 5) := 
by 
  intros
  sorry

end range_of_m_in_third_quadrant_l145_145329


namespace sequence_sum_l145_145744

theorem sequence_sum :
  (∀ r: ℝ, 
    (∀ x y: ℝ,
      r = 1 / 4 → 
      x = 256 * r → 
      y = x * r → 
      x + y = 80)) :=
by sorry

end sequence_sum_l145_145744


namespace minimum_xy_minimum_x_plus_y_l145_145429

theorem minimum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 :=
sorry

theorem minimum_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 :=
sorry

end minimum_xy_minimum_x_plus_y_l145_145429


namespace eval_expression_l145_145868

variable {x : ℝ}

theorem eval_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 8) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 8 * x + 2 :=
by
  sorry

end eval_expression_l145_145868


namespace evaluate_f_complex_l145_145066

noncomputable def f (z : ℂ) : ℂ :=
  if ¬((z.re : ℝ) = z) then z^2 + 1
  else if (z.re = z) && (z.im = 0) then -z^2 + 1
  else 2 * z

theorem evaluate_f_complex : f (f (f (f (1 + Complex.I)))) = 378 + 336 * Complex.I :=
  by
    sorry

end evaluate_f_complex_l145_145066


namespace area_within_fence_l145_145097

theorem area_within_fence : 
  let rectangle_area := 20 * 18
  let cutout_area := 4 * 4
  rectangle_area - cutout_area = 344 := by
    -- Definitions
    let rectangle_area := 20 * 18
    let cutout_area := 4 * 4
    
    -- Computation of areas
    show rectangle_area - cutout_area = 344
    sorry

end area_within_fence_l145_145097


namespace bookA_net_change_bookB_net_change_bookC_net_change_l145_145130

-- Define the price adjustments for Book A
def bookA_initial_price := 100.0
def bookA_after_first_adjustment := bookA_initial_price * (1 - 0.5)
def bookA_after_second_adjustment := bookA_after_first_adjustment * (1 + 0.6)
def bookA_final_price := bookA_after_second_adjustment * (1 + 0.1)
def bookA_net_percentage_change := (bookA_final_price - bookA_initial_price) / bookA_initial_price * 100

-- Define the price adjustments for Book B
def bookB_initial_price := 100.0
def bookB_after_first_adjustment := bookB_initial_price * (1 + 0.2)
def bookB_after_second_adjustment := bookB_after_first_adjustment * (1 - 0.3)
def bookB_final_price := bookB_after_second_adjustment * (1 + 0.25)
def bookB_net_percentage_change := (bookB_final_price - bookB_initial_price) / bookB_initial_price * 100

-- Define the price adjustments for Book C
def bookC_initial_price := 100.0
def bookC_after_first_adjustment := bookC_initial_price * (1 + 0.4)
def bookC_after_second_adjustment := bookC_after_first_adjustment * (1 - 0.1)
def bookC_final_price := bookC_after_second_adjustment * (1 - 0.05)
def bookC_net_percentage_change := (bookC_final_price - bookC_initial_price) / bookC_initial_price * 100

-- Statements to prove the net percentage changes
theorem bookA_net_change : bookA_net_percentage_change = -12 := by
  sorry

theorem bookB_net_change : bookB_net_percentage_change = 5 := by
  sorry

theorem bookC_net_change : bookC_net_percentage_change = 19.7 := by
  sorry

end bookA_net_change_bookB_net_change_bookC_net_change_l145_145130


namespace equivalent_proof_problem_l145_145263

theorem equivalent_proof_problem (x : ℤ) (h : (x - 5) / 7 = 7) : (x - 14) / 10 = 4 :=
by
  sorry

end equivalent_proof_problem_l145_145263


namespace playground_perimeter_km_l145_145515

def playground_length : ℕ := 360
def playground_width : ℕ := 480

def perimeter_in_meters (length width : ℕ) : ℕ := 2 * (length + width)

def perimeter_in_kilometers (perimeter_m : ℕ) : ℕ := perimeter_m / 1000

theorem playground_perimeter_km :
  perimeter_in_kilometers (perimeter_in_meters playground_length playground_width) = 168 :=
by
  sorry

end playground_perimeter_km_l145_145515


namespace solve_inequality_l145_145979

theorem solve_inequality (x : ℝ) : 3 * x^2 + 2 * x - 3 > 12 - 2 * x → x < -3 ∨ x > 5 / 3 :=
sorry

end solve_inequality_l145_145979


namespace fabric_length_l145_145161

-- Define the width and area as given in the problem
def width : ℝ := 3
def area : ℝ := 24

-- Prove that the length is 8 cm
theorem fabric_length : (area / width) = 8 :=
by
  sorry

end fabric_length_l145_145161


namespace sum_of_solutions_sum_of_possible_values_l145_145453

theorem sum_of_solutions (y : ℝ) (h : y^2 = 81) : y = 9 ∨ y = -9 :=
sorry

theorem sum_of_possible_values (y : ℝ) (h : y^2 = 81) : (∀ x, x = 9 ∨ x = -9 → x = 9 ∨ x = -9 → x = 9 + (-9)) :=
by
  have y_sol : y = 9 ∨ y = -9 := sum_of_solutions y h
  sorry

end sum_of_solutions_sum_of_possible_values_l145_145453


namespace max_sum_of_squares_eq_100_l145_145057

theorem max_sum_of_squares_eq_100 : 
  ∃ (x y : ℤ), x^2 + y^2 = 100 ∧ 
  (∀ (x y : ℤ), x^2 + y^2 = 100 → x + y ≤ 14) ∧ 
  (∃ (x y : ℕ), x^2 + y^2 = 100 ∧ x + y = 14) :=
by {
  sorry
}

end max_sum_of_squares_eq_100_l145_145057


namespace reflection_ray_equation_l145_145141

theorem reflection_ray_equation (x y : ℝ) : (y = 2 * x + 1) → (∃ (x' y' : ℝ), y' = x ∧ y = 2 * x' + 1 ∧ x - 2 * y - 1 = 0) :=
by
  intro h
  sorry

end reflection_ray_equation_l145_145141


namespace max_sum_of_arithmetic_sequence_l145_145042

theorem max_sum_of_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a3 : a 3 = 7)
  (h_a1_a7 : a 1 + a 7 = 10)
  (h_S : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 1 - a 0))) / 2) :
  ∃ n, S n = S 6 ∧ (∀ m, S m ≤ S 6) :=
sorry

end max_sum_of_arithmetic_sequence_l145_145042


namespace even_odd_product_zero_l145_145427

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem even_odd_product_zero (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : is_even f) (hg : is_odd g) : ∀ x, f (-x) * g (-x) + f x * g x = 0 :=
by
  intro x
  have h₁ := hf x
  have h₂ := hg x
  sorry

end even_odd_product_zero_l145_145427


namespace sequence_value_l145_145743

theorem sequence_value (r x y : ℝ) 
  (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : 
  x + y = 80 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end sequence_value_l145_145743


namespace problem_rewrite_expression_l145_145812

theorem problem_rewrite_expression (j : ℝ) : 
  ∃ (c p q : ℝ), (8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ (q / p = -77) :=
sorry

end problem_rewrite_expression_l145_145812


namespace solve_inequality_1_find_range_of_a_l145_145988

def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem solve_inequality_1 :
  {x : ℝ | f x ≥ 5} = {x : ℝ | x ≤ -3} ∪ {x : ℝ | x ≥ 2} :=
by
  sorry
  
theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x > a^2 - 2 * a - 5) ↔ -2 < a ∧ a < 4 :=
by
  sorry

end solve_inequality_1_find_range_of_a_l145_145988


namespace southern_northern_dynasties_congruence_l145_145867

theorem southern_northern_dynasties_congruence :
  let a := ∑ k in Finset.range 21, Nat.choose 20 k * 2^k
  ∃ b : ℤ, (a ≡ 2011 [MOD 10]) → b = 2011 :=
by { 
  sorry 
}

end southern_northern_dynasties_congruence_l145_145867


namespace problem_proof_l145_145311

def f (a x : ℝ) := |a - x|

theorem problem_proof (a x x0 : ℝ) (h_a : a = 3 / 2) (h_x0 : x0 < 0) : 
  f a (x0 * x) ≥ x0 * f a x + f a (a * x0) :=
sorry

end problem_proof_l145_145311


namespace conditional_probabilities_l145_145229

-- Define the events A and B
def three_dice := ℕ × ℕ × ℕ

def event_A (d : three_dice) : Prop :=
  d.1 ≠ d.2 ∧ d.2 ≠ d.3 ∧ d.3 ≠ d.1

def event_B (d : three_dice) : Prop :=
  d.1 = 6 ∨ d.2 = 6 ∨ d.3 = 6

-- Calculate conditional probabilities
noncomputable def P_A_given_B : ℚ :=
  60 / 91

noncomputable def P_B_given_A : ℚ :=
  1 / 2

-- Statement of the problem
theorem conditional_probabilities :
  ((P_A_given_B = 60 / 91) ∧ (P_B_given_A = 1 / 2)) :=
by
  sorry

end conditional_probabilities_l145_145229


namespace find_delta_l145_145989

theorem find_delta (p q Δ : ℕ) (h₁ : Δ + q = 73) (h₂ : 2 * (Δ + q) + p = 172) (h₃ : p = 26) : Δ = 12 :=
by
  sorry

end find_delta_l145_145989


namespace james_ali_difference_l145_145202

theorem james_ali_difference (J A T : ℝ) (h1 : J = 145) (h2 : T = 250) (h3 : J + A = T) :
  J - A = 40 :=
by
  sorry

end james_ali_difference_l145_145202


namespace distance_between_foci_of_ellipse_l145_145164

-- Definitions based on conditions in the problem
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 45 + (y^2) / 5 = 9

-- Proof statement
theorem distance_between_foci_of_ellipse : 
  ∀ (x y : ℝ), ellipse_eq x y → 2 * (√ (405 - 45)) = 12 * (√ 10) :=
by 
  sorry

end distance_between_foci_of_ellipse_l145_145164


namespace line_intersects_circle_l145_145309

theorem line_intersects_circle 
  (radius : ℝ) 
  (distance_center_line : ℝ) 
  (h_radius : radius = 4) 
  (h_distance : distance_center_line = 3) : 
  radius > distance_center_line := 
by 
  sorry

end line_intersects_circle_l145_145309


namespace mitch_total_scoops_l145_145917

theorem mitch_total_scoops :
  (3 : ℝ) / (1/3 : ℝ) + (2 : ℝ) / (1/3 : ℝ) = 15 :=
by
  sorry

end mitch_total_scoops_l145_145917


namespace area_within_fence_l145_145095

theorem area_within_fence : 
  let rectangle_area := 20 * 18
  let cutout_area := 4 * 4
  rectangle_area - cutout_area = 344 := by
    -- Definitions
    let rectangle_area := 20 * 18
    let cutout_area := 4 * 4
    
    -- Computation of areas
    show rectangle_area - cutout_area = 344
    sorry

end area_within_fence_l145_145095


namespace determine_min_k_l145_145340

open Nat

theorem determine_min_k (n : ℕ) (h : n ≥ 3) 
  (a : Fin n → ℕ) (b : Fin (choose n 2) → ℕ) : 
  ∃ k, k = (n - 1) * (n - 2) / 2 + 1 := 
sorry

end determine_min_k_l145_145340


namespace minimum_value_of_fraction_l145_145044

theorem minimum_value_of_fraction (x : ℝ) (h : x > 0) : 
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 - 1 ∧ ∀ y, y = (x^2 + x + 3) / (x + 1) -> y ≥ m :=
sorry

end minimum_value_of_fraction_l145_145044


namespace mike_initial_cards_l145_145810

-- Define the conditions
def initial_cards (x : ℕ) := x + 13 = 100

-- Define the proof statement
theorem mike_initial_cards : initial_cards 87 :=
by
  sorry

end mike_initial_cards_l145_145810


namespace geometric_series_sum_l145_145731

theorem geometric_series_sum :
  let a := 1
  let r := (1 / 4 : ℚ)
  (a / (1 - r)) = 4 / 3 :=
by
  sorry

end geometric_series_sum_l145_145731


namespace find_A_l145_145483

variable (A B x : ℝ)
variable (hB : B ≠ 0)
variable (h : f (g 2) = 0)
def f := λ x => A * x^3 - B
def g := λ x => B * x^2

theorem find_A (hB : B ≠ 0) (h : (λ x => A * x^3 - B) ((λ x => B * x^2) 2) = 0) : 
  A = 1 / (64 * B^2) :=
  sorry

end find_A_l145_145483


namespace trapezoid_not_isosceles_l145_145462

noncomputable def is_trapezoid (BC AD AC : ℝ) : Prop :=
BC = 3 ∧ AD = 4 ∧ AC = 6

def is_isosceles_trapezoid_not_possible (BC AD AC : ℝ) : Prop :=
is_trapezoid BC AD AC → ¬(BC = AD)

theorem trapezoid_not_isosceles (BC AD AC : ℝ) :
  is_isosceles_trapezoid_not_possible BC AD AC :=
sorry

end trapezoid_not_isosceles_l145_145462


namespace min_value_f_range_m_l145_145699

-- Part I: Prove that the minimum value of f(a) = a^2 + 2/a for a > 0 is 3
theorem min_value_f (a : ℝ) (h : a > 0) : a^2 + 2 / a ≥ 3 :=
sorry

-- Part II: Prove the range of m given the inequality for any positive real number a
theorem range_m (m : ℝ) : (∀ (a : ℝ), a > 0 → a^3 + 2 ≥ 3 * a * (|m - 1| - |2 * m + 3|)) → (m ≤ -3 ∨ m ≥ -1) :=
sorry

end min_value_f_range_m_l145_145699


namespace set_intersection_complement_l145_145806

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {1, 2, 3}

-- Define set B
def B : Set ℕ := {3, 4, 5}

-- State the theorem
theorem set_intersection_complement :
  (U \ A) ∩ B = {4, 5} := by
  sorry

end set_intersection_complement_l145_145806


namespace trajectory_midpoint_of_chord_l145_145306

theorem trajectory_midpoint_of_chord :
  ∀ (M: ℝ × ℝ), (∃ (C D : ℝ × ℝ), (C.1^2 + C.2^2 = 25 ∧ D.1^2 + D.2^2 = 25 ∧ dist C D = 8) ∧ M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
  → M.1^2 + M.2^2 = 9 :=
sorry

end trajectory_midpoint_of_chord_l145_145306


namespace sum_of_first_50_primes_is_5356_l145_145943

open Nat

-- Define the first 50 prime numbers
def first_50_primes : List Nat := 
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 
   83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 
   179, 181, 191, 193, 197, 199, 211, 223, 227, 229]

-- Calculate their sum
def sum_first_50_primes : Nat := List.foldr (Nat.add) 0 first_50_primes

-- Now we state the theorem we want to prove
theorem sum_of_first_50_primes_is_5356 : 
  sum_first_50_primes = 5356 := 
by
  -- Placeholder for proof
  sorry

end sum_of_first_50_primes_is_5356_l145_145943


namespace slope_acute_l145_145624

noncomputable def curve (a : ℤ) : ℝ → ℝ := λ x => x^3 - 2 * a * x^2 + 2 * a * x

noncomputable def tangent_slope (a : ℤ) : ℝ → ℝ := λ x => 3 * x^2 - 4 * a * x + 2 * a

theorem slope_acute (a : ℤ) : (∀ x : ℝ, (tangent_slope a x > 0)) ↔ (a = 1) := sorry

end slope_acute_l145_145624


namespace problem_value_l145_145698

theorem problem_value :
  4 * (8 - 3) / 2 - 7 = 3 := 
by
  sorry

end problem_value_l145_145698


namespace exactly_two_talents_l145_145010

open Nat

def total_students : Nat := 50
def cannot_sing_students : Nat := 20
def cannot_dance_students : Nat := 35
def cannot_act_students : Nat := 15

theorem exactly_two_talents : 
  (total_students - cannot_sing_students) + 
  (total_students - cannot_dance_students) + 
  (total_students - cannot_act_students) - total_students = 30 := by
  sorry

end exactly_two_talents_l145_145010


namespace kira_travel_time_l145_145112

def total_travel_time (hours_between_stations : ℕ) (break_minutes : ℕ) : ℕ :=
  let travel_time_hours := 2 * hours_between_stations
  let travel_time_minutes := travel_time_hours * 60
  travel_time_minutes + break_minutes

theorem kira_travel_time : total_travel_time 2 30 = 270 :=
  by sorry

end kira_travel_time_l145_145112


namespace solve_for_x_l145_145497

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 15.4 → x = 110.8333333 :=
by
  intro h
  sorry

end solve_for_x_l145_145497


namespace vector_perpendicular_sets_l145_145315

-- Define the problem in Lean
theorem vector_perpendicular_sets (x : ℝ) : 
  let a := (Real.sin x, Real.cos x)
  let b := (Real.sin x + Real.cos x, Real.sin x - Real.cos x)
  a.1 * b.1 + a.2 * b.2 = 0 ↔ ∃ (k : ℤ), x = k * (π / 2) + (π / 8) :=
sorry

end vector_perpendicular_sets_l145_145315


namespace total_copies_in_half_hour_l145_145843

-- Definitions of the machine rates and their time segments.
def machine1_rate := 35 -- copies per minute
def machine2_rate := 65 -- copies per minute
def machine3_rate1 := 50 -- copies per minute for the first 15 minutes
def machine3_rate2 := 80 -- copies per minute for the next 15 minutes
def machine4_rate1 := 90 -- copies per minute for the first 10 minutes
def machine4_rate2 := 60 -- copies per minute for the next 20 minutes

-- Time intervals for different machines
def machine3_time1 := 15 -- minutes
def machine3_time2 := 15 -- minutes
def machine4_time1 := 10 -- minutes
def machine4_time2 := 20 -- minutes

-- Proof statement
theorem total_copies_in_half_hour : 
  (machine1_rate * 30) + 
  (machine2_rate * 30) + 
  ((machine3_rate1 * machine3_time1) + (machine3_rate2 * machine3_time2)) + 
  ((machine4_rate1 * machine4_time1) + (machine4_rate2 * machine4_time2)) = 
  7050 :=
by 
  sorry

end total_copies_in_half_hour_l145_145843


namespace beer_drawing_time_l145_145951

theorem beer_drawing_time :
  let rate_A := 1 / 5
  let rate_C := 1 / 4
  let combined_rate := 9 / 20
  let extra_beer := 12
  let total_drawn := 48
  let t := total_drawn / combined_rate
  t = 48 * 20 / 9 :=
by {
  sorry -- proof not required
}

end beer_drawing_time_l145_145951


namespace water_intake_proof_l145_145248

variable {quarts_per_bottle : ℕ} {bottles_per_day : ℕ} {extra_ounces_per_day : ℕ} 
variable {days_per_week : ℕ} {ounces_per_quart : ℕ} 

def total_weekly_water_intake 
    (quarts_per_bottle : ℕ) 
    (bottles_per_day : ℕ) 
    (extra_ounces_per_day : ℕ) 
    (ounces_per_quart : ℕ) 
    (days_per_week : ℕ) 
    (correct_answer : ℕ) : Prop :=
    (quarts_per_bottle * ounces_per_quart * bottles_per_day + extra_ounces_per_day) * days_per_week = correct_answer

theorem water_intake_proof : 
    total_weekly_water_intake 3 2 20 32 7 812 := 
by
    sorry

end water_intake_proof_l145_145248


namespace quadratic_roots_p_l145_145054

noncomputable def equation : Type* := sorry

theorem quadratic_roots_p
  (α β : ℝ)
  (K : ℝ)
  (h1 : 3 * α ^ 2 + 7 * α + K = 0)
  (h2 : 3 * β ^ 2 + 7 * β + K = 0)
  (sum_roots : α + β = -7 / 3)
  (prod_roots : α * β = K / 3)
  : ∃ p : ℝ, p = -70 / 9 + 2 * K / 3 := 
sorry

end quadratic_roots_p_l145_145054


namespace min_dot_product_trajectory_l145_145039

-- Definitions of points and conditions
def point (x y : ℝ) : Prop := True

def trajectory (P : ℝ × ℝ) : Prop := 
  let x := P.1
  let y := P.2
  x * x - y * y = 2 ∧ x ≥ Real.sqrt 2

-- Definition of dot product over vectors from origin
def dot_product (A B : ℝ × ℝ) : ℝ :=
  A.1 * B.1 + A.2 * B.2

-- Stating the theorem for minimum value of dot product
theorem min_dot_product_trajectory (A B : ℝ × ℝ) (hA : trajectory A) (hB : trajectory B) : 
  dot_product A B ≥ 2 := 
sorry

end min_dot_product_trajectory_l145_145039


namespace gcd_max_possible_value_l145_145580

theorem gcd_max_possible_value (x y : ℤ) (h_coprime : Int.gcd x y = 1) : 
  ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
by
  sorry

end gcd_max_possible_value_l145_145580


namespace sequence_formula_l145_145063

theorem sequence_formula (a : ℕ → ℕ) (n : ℕ) (h : ∀ n ≥ 1, a n = a (n - 1) + n^3) : 
  a n = (n * (n + 1) / 2) ^ 2 := sorry

end sequence_formula_l145_145063


namespace largest_non_sum_of_multiple_of_30_and_composite_l145_145942

theorem largest_non_sum_of_multiple_of_30_and_composite :
  ∃ (n : ℕ), n = 211 ∧ ∀ a b : ℕ, (a > 0) → (b > 0) → (b < 30) → 
  n ≠ 30 * a + b ∧ ¬ ∃ k : ℕ, k > 1 ∧ k < b ∧ b % k = 0 :=
sorry

end largest_non_sum_of_multiple_of_30_and_composite_l145_145942


namespace barbara_current_savings_l145_145282

def wristwatch_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def initial_saving_duration : ℕ := 10
def further_saving_duration : ℕ := 16

theorem barbara_current_savings : 
  -- Given:
  -- wristwatch_cost: $100
  -- weekly_allowance: $5
  -- further_saving_duration: 16 weeks
  -- Prove:
  -- Barbara currently has $20
  wristwatch_cost - weekly_allowance * further_saving_duration = 20 :=
by
  sorry

end barbara_current_savings_l145_145282


namespace parabola_point_distance_eq_four_maximum_area_triangle_PAB_l145_145607

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := {p | p.1^2 = 2 * p * p.2}
noncomputable def circle : set (ℝ × ℝ) := {q | q.1^2 + (q.2 + 4)^2 = 1}
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
noncomputable def dist (a b : ℝ × ℝ) : ℝ := 
    real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

theorem parabola_point_distance_eq_four 
  (p : ℝ)
  (hp : p > 0) 
  (h : ∃ (q : ℝ × ℝ), q ∈ circle ∧ dist (focus p) q = 4 + 1) :
  p = 2 := 
sorry

theorem maximum_area_triangle_PAB
  (k b : ℝ)
  (hP : (2 * k, -b) ∈ circle)
  (p := 2) :
  4 * (k^2 + b)^(3/2) = 20 * real.sqrt 5 :=
sorry

end parabola_point_distance_eq_four_maximum_area_triangle_PAB_l145_145607


namespace problem_statement_l145_145179
open Real

noncomputable def l1 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (sin α) * x - (cos α) * y + 1 = 0
noncomputable def l2 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (sin α) * x + (cos α) * y + 1 = 0
noncomputable def l3 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (cos α) * x - (sin α) * y + 1 = 0
noncomputable def l4 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (cos α) * x + (sin α) * y + 1 = 0

theorem problem_statement:
  (∃ (α : ℝ), ∀ (x y : ℝ), l1 α x y → l2 α x y) ∧
  (∀ (α : ℝ), ∀ (x y : ℝ), l1 α x y → (sin α) * (cos α) + (-cos α) * (sin α) = 0) ∧
  (∃ (p : ℝ × ℝ), ∀ (α : ℝ), abs ((sin α) * p.1 - (cos α) * p.2 + 1) / sqrt ((sin α)^2 + (cos α)^2) = 1 ∧
                        abs ((sin α) * p.1 + (cos α) * p.2 + 1) / sqrt ((sin α)^2 + (cos α)^2) = 1 ∧
                        abs ((cos α) * p.1 - (sin α) * p.2 + 1) / sqrt ((cos α)^2 + (sin α)^2) = 1 ∧
                        abs ((cos α) * p.1 + (sin α) * p.2 + 1) / sqrt ((cos α)^2 + (sin α)^2) = 1) :=
sorry

end problem_statement_l145_145179


namespace ratio_of_age_difference_l145_145078

theorem ratio_of_age_difference (R J K : ℕ) 
  (h1 : R = J + 6) 
  (h2 : R + 4 = 2 * (J + 4)) 
  (h3 : (R + 4) * (K + 4) = 108) : 
  (R - J) / (R - K) = 2 :=
by 
  sorry

end ratio_of_age_difference_l145_145078


namespace inscribed_circle_radius_of_DEF_l145_145524

theorem inscribed_circle_radius_of_DEF (DE DF EF : ℝ) (r : ℝ)
  (hDE : DE = 8) (hDF : DF = 8) (hEF : EF = 10) :
  r = 5 * Real.sqrt 39 / 13 :=
by
  sorry

end inscribed_circle_radius_of_DEF_l145_145524


namespace percentage_increase_efficiency_l145_145351

-- Defining the times taken by Sakshi and Tanya
def sakshi_time : ℕ := 12
def tanya_time : ℕ := 10

-- Defining the efficiency in terms of work per day for Sakshi and Tanya
def sakshi_efficiency : ℚ := 1 / sakshi_time
def tanya_efficiency : ℚ := 1 / tanya_time

-- The statement of the proof: percentage increase
theorem percentage_increase_efficiency : 
  100 * ((tanya_efficiency - sakshi_efficiency) / sakshi_efficiency) = 20 := 
by
  -- The actual proof will go here
  sorry

end percentage_increase_efficiency_l145_145351


namespace solve_inequality_l145_145294

def satisfies_inequality (x : ℝ) : Prop :=
  (3 * x - 4) * (x + 1) / x ≥ 0

theorem solve_inequality :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | -1 ≤ x ∧ x < 0 ∨ x ≥ 4 / 3} :=
by
  sorry

end solve_inequality_l145_145294
