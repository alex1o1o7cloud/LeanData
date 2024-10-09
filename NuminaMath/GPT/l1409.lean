import Mathlib

namespace quadratic_roots_expression_l1409_140919

theorem quadratic_roots_expression :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 1 = 0) ∧ (x2^2 - 2 * x2 - 1 = 0) →
  (x1 + x2 - x1 * x2 = 3) :=
by
  intros x1 x2 h
  sorry

end quadratic_roots_expression_l1409_140919


namespace part_one_part_two_i_part_two_ii_l1409_140928

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem part_one (a b : ℝ) : 
  f (-a / 2 + 1) a b ≤ f (a^2 + 5 / 4) a b :=
sorry

theorem part_two_i (a b : ℝ) : 
  f 1 a b + f 3 a b - 2 * f 2 a b = 2 :=
sorry

theorem part_two_ii (a b : ℝ) : 
  ¬((|f 1 a b| < 1/2) ∧ (|f 2 a b| < 1/2) ∧ (|f 3 a b| < 1/2)) :=
sorry

end part_one_part_two_i_part_two_ii_l1409_140928


namespace normal_price_of_article_l1409_140965

theorem normal_price_of_article 
  (final_price : ℝ)
  (discount1 : ℝ) 
  (discount2 : ℝ) 
  (P : ℝ)
  (h : final_price = 108) 
  (h1 : discount1 = 0.10) 
  (h2 : discount2 = 0.20)
  (h_eq : (1 - discount1) * (1 - discount2) * P = final_price) :
  P = 150 := by
  sorry

end normal_price_of_article_l1409_140965


namespace seating_arrangements_l1409_140952

-- Definitions for conditions
def num_parents : ℕ := 2
def num_children : ℕ := 3
def num_front_seats : ℕ := 2
def num_back_seats : ℕ := 3
def num_family_members : ℕ := num_parents + num_children

-- The statement we need to prove
theorem seating_arrangements : 
  (num_parents * -- choices for driver
  (num_family_members - 1) * -- choices for the front passenger
  (num_back_seats.factorial)) = 48 := -- arrangements for the back seats
by
  sorry

end seating_arrangements_l1409_140952


namespace relationship_between_x_plus_one_and_ex_l1409_140957

theorem relationship_between_x_plus_one_and_ex (x : ℝ) : x + 1 ≤ Real.exp x :=
sorry

end relationship_between_x_plus_one_and_ex_l1409_140957


namespace math_players_count_l1409_140918

-- Define the conditions given in the problem.
def total_players : ℕ := 25
def physics_players : ℕ := 9
def both_subjects_players : ℕ := 5

-- Statement to be proven
theorem math_players_count :
  total_players = physics_players + both_subjects_players + (total_players - physics_players - both_subjects_players) → 
  total_players - physics_players + both_subjects_players = 21 := 
sorry

end math_players_count_l1409_140918


namespace algebraic_notation_correct_l1409_140921

def exprA : String := "a * 5"
def exprB : String := "a7"
def exprC : String := "3 1/2 x"
def exprD : String := "-7/8 x"

theorem algebraic_notation_correct :
  exprA ≠ "correct" ∧
  exprB ≠ "correct" ∧
  exprC ≠ "correct" ∧
  exprD = "correct" :=
by
  sorry

end algebraic_notation_correct_l1409_140921


namespace mary_travel_time_l1409_140953

noncomputable def ambulance_speed : ℝ := 60
noncomputable def don_speed : ℝ := 30
noncomputable def don_time : ℝ := 0.5

theorem mary_travel_time : (don_speed * don_time) / ambulance_speed * 60 = 15 := by
  sorry

end mary_travel_time_l1409_140953


namespace coordinates_of_AC_l1409_140973

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p1 p2 : Point3D) : Point3D :=
  { x := p1.x - p2.x,
    y := p1.y - p2.y,
    z := p1.z - p2.z }

def scalar_mult (k : ℝ) (v : Point3D) : Point3D :=
  { x := k * v.x,
    y := k * v.y,
    z := k * v.z }

noncomputable def A : Point3D := { x := 1, y := 2, z := 3 }
noncomputable def B : Point3D := { x := 4, y := 5, z := 9 }

theorem coordinates_of_AC : vector_sub B A = { x := 3, y := 3, z := 6 } →
  scalar_mult (1 / 3) (vector_sub B A) = { x := 1, y := 1, z := 2 } :=
by
  sorry

end coordinates_of_AC_l1409_140973


namespace find_expression_value_l1409_140968

theorem find_expression_value (m: ℝ) (h: m^2 - 2 * m - 1 = 0) : 
  (m - 1)^2 - (m - 3) * (m + 3) - (m - 1) * (m - 3) = 6 := 
by 
  sorry

end find_expression_value_l1409_140968


namespace opposite_of_negative_five_l1409_140984

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end opposite_of_negative_five_l1409_140984


namespace election_candidate_a_votes_l1409_140958

theorem election_candidate_a_votes :
  let total_votes : ℕ := 560000
  let invalid_percentage : ℚ := 15 / 100
  let candidate_a_percentage : ℚ := 70 / 100
  let total_valid_votes := total_votes * (1 - invalid_percentage)
  let candidate_a_votes := total_valid_votes * candidate_a_percentage
  candidate_a_votes = 333200 :=
by
  let total_votes : ℕ := 560000
  let invalid_percentage : ℚ := 15 / 100
  let candidate_a_percentage : ℚ := 70 / 100
  let total_valid_votes := total_votes * (1 - invalid_percentage)
  let candidate_a_votes := total_valid_votes * candidate_a_percentage
  show candidate_a_votes = 333200
  sorry

end election_candidate_a_votes_l1409_140958


namespace intersecting_lines_l1409_140983

theorem intersecting_lines (a b : ℝ) (h1 : 1 = 1 / 4 * 2 + a) (h2 : 2 = 1 / 4 * 1 + b) : 
  a + b = 9 / 4 := 
sorry

end intersecting_lines_l1409_140983


namespace todd_initial_gum_l1409_140929

-- Define the conditions and the final result
def initial_gum (final_gum: Nat) (given_gum: Nat) : Nat := final_gum - given_gum

theorem todd_initial_gum :
  initial_gum 54 16 = 38 :=
by
  -- Use the initial_gum definition to state the problem
  -- The proof is skipped with sorry
  sorry

end todd_initial_gum_l1409_140929


namespace inequality_l1409_140946
-- Import the necessary libraries from Mathlib

-- Define the theorem statement
theorem inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := 
by
  sorry

end inequality_l1409_140946


namespace add_in_base8_l1409_140998

def base8_add (a b : ℕ) (n : ℕ): ℕ :=
  a * (8 ^ n) + b

theorem add_in_base8 : base8_add 123 56 0 = 202 := by
  sorry

end add_in_base8_l1409_140998


namespace total_windows_l1409_140912

theorem total_windows (installed: ℕ) (hours_per_window: ℕ) (remaining_hours: ℕ) : installed = 8 → hours_per_window = 8 → remaining_hours = 48 → 
  (installed + remaining_hours / hours_per_window) = 14 := by 
  intros h1 h2 h3
  sorry

end total_windows_l1409_140912


namespace remainder_3_pow_2000_mod_17_l1409_140997

theorem remainder_3_pow_2000_mod_17 : (3^2000 % 17) = 1 := by
  sorry

end remainder_3_pow_2000_mod_17_l1409_140997


namespace jangshe_clothing_cost_l1409_140964

theorem jangshe_clothing_cost
  (total_spent : ℝ)
  (untaxed_piece1 : ℝ)
  (untaxed_piece2 : ℝ)
  (total_pieces : ℕ)
  (remaining_pieces : ℕ)
  (remaining_pieces_price : ℝ)
  (sales_tax : ℝ)
  (price_multiple_of_five : ℝ) :
  total_spent = 610 ∧
  untaxed_piece1 = 49 ∧
  untaxed_piece2 = 81 ∧
  total_pieces = 7 ∧
  remaining_pieces = 5 ∧
  sales_tax = 0.10 ∧
  (∃ k : ℕ, remaining_pieces_price = k * 5) →
  remaining_pieces_price / remaining_pieces = 87 :=
by
  sorry

end jangshe_clothing_cost_l1409_140964


namespace largest_marbles_l1409_140988

theorem largest_marbles {n : ℕ} (h1 : n < 400) (h2 : n % 3 = 1) (h3 : n % 7 = 2) (h4 : n % 5 = 0) : n = 310 :=
  sorry

end largest_marbles_l1409_140988


namespace hiker_speed_calculation_l1409_140901

theorem hiker_speed_calculation :
  ∃ (h_speed : ℝ),
    let c_speed := 10
    let c_time := 5.0 / 60.0
    let c_wait := 7.5 / 60.0
    let c_distance := c_speed * c_time
    let h_distance := c_distance
    h_distance = h_speed * c_wait ∧ h_speed = 10 * (5 / 7.5) := by
  sorry

end hiker_speed_calculation_l1409_140901


namespace multiplication_identity_l1409_140961

theorem multiplication_identity : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ 29 := by
  sorry

end multiplication_identity_l1409_140961


namespace inequality_proof_l1409_140936

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := 
by 
  sorry

end inequality_proof_l1409_140936


namespace decimal_representation_l1409_140989

theorem decimal_representation :
  (13 : ℝ) / (2 * 5^8) = 0.00001664 := 
  sorry

end decimal_representation_l1409_140989


namespace range_of_a_l1409_140920

-- Definition for set A
def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = -|x| - 2 }

-- Definition for set B
def B (a : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x - a)^2 + y^2 = a^2 }

-- Statement of the problem in Lean
theorem range_of_a (a : ℝ) : (∀ p, p ∈ A → p ∉ B a) → -2 * Real.sqrt 2 - 2 < a ∧ a < 2 * Real.sqrt 2 + 2 := by
  sorry

end range_of_a_l1409_140920


namespace card_selection_ways_l1409_140959

theorem card_selection_ways (deck_size : ℕ) (suits : ℕ) (cards_per_suit : ℕ) (total_cards_chosen : ℕ)
  (repeated_suit_count : ℕ) (distinct_suits_count : ℕ) (distinct_ranks_count : ℕ) 
  (correct_answer : ℕ) :
  deck_size = 52 ∧ suits = 4 ∧ cards_per_suit = 13 ∧ total_cards_chosen = 5 ∧ 
  repeated_suit_count = 2 ∧ distinct_suits_count = 3 ∧ distinct_ranks_count = 11 ∧ 
  correct_answer = 414384 :=
by 
  -- Sorry is used to skip actual proof steps, according to the instructions.
  sorry

end card_selection_ways_l1409_140959


namespace range_of_a_l1409_140978

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def is_monotone_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y : ℝ⦄, 0 ≤ x → 0 ≤ y → x < y → f x < f y

axiom even_f : is_even f
axiom monotone_f : is_monotone_on_nonneg f

theorem range_of_a (a : ℝ) (h : f a ≥ f 3) : a ≤ -3 ∨ a ≥ 3 :=
by
  sorry

end range_of_a_l1409_140978


namespace proof_l1409_140977

noncomputable def problem_statement (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∀ x : ℝ, |x + a| + |x - b| + c ≥ 4)

theorem proof (a b c : ℝ) (h : problem_statement a b c) :
  a + b + c = 4 ∧ (∀ x : ℝ, 1 / a + 4 / b + 9 / c ≥ 9) :=
by
  sorry

end proof_l1409_140977


namespace initial_average_daily_production_l1409_140994

variable (A : ℝ) -- Initial average daily production
variable (n : ℕ) -- Number of days

theorem initial_average_daily_production (n_eq_5 : n = 5) (new_production_eq_90 : 90 = 90) 
  (new_average_eq_65 : (5 * A + 90) / 6 = 65) : A = 60 :=
by
  sorry

end initial_average_daily_production_l1409_140994


namespace area_of_equilateral_triangle_with_inscribed_circle_l1409_140970

theorem area_of_equilateral_triangle_with_inscribed_circle 
  (r : ℝ) (A : ℝ) (area_circle_eq : A = 9 * Real.pi)
  (DEF_equilateral : ∀ {a b c : ℝ}, a = b ∧ b = c): 
  ∃ area_def : ℝ, area_def = 27 * Real.sqrt 3 :=
by
  -- proof omitted
  sorry

end area_of_equilateral_triangle_with_inscribed_circle_l1409_140970


namespace planA_equals_planB_at_3_l1409_140945

def planA_charge_for_first_9_minutes : ℝ := 0.24
def planA_charge (X: ℝ) (minutes: ℕ) : ℝ := if minutes <= 9 then X else X + 0.06 * (minutes - 9)
def planB_charge (minutes: ℕ) : ℝ := 0.08 * minutes

theorem planA_equals_planB_at_3 : planA_charge planA_charge_for_first_9_minutes 3 = planB_charge 3 :=
by sorry

end planA_equals_planB_at_3_l1409_140945


namespace find_values_of_a_b_l1409_140941

variable (a b : ℤ)

def A : Set ℤ := {1, b, a + b}
def B : Set ℤ := {a - b, a * b}
def common_set : Set ℤ := {-1, 0}

theorem find_values_of_a_b (h : A a b ∩ B a b = common_set) : (a, b) = (-1, 0) := by
  sorry

end find_values_of_a_b_l1409_140941


namespace t1_eq_t2_l1409_140913

variable (n : ℕ)
variable (s₁ s₂ s₃ : ℝ)
variable (t₁ t₂ : ℝ)
variable (S1 S2 S3 : ℝ)

-- Conditions
axiom h1 : S1 = s₁
axiom h2 : S2 = s₂
axiom h3 : S3 = s₃
axiom h4 : t₁ = s₂^2 - s₁ * s₃
axiom h5 : t₂ = ( (s₁ - s₃) / 2 )^2
axiom h6 : s₁ + s₃ = 2 * s₂

theorem t1_eq_t2 : t₁ = t₂ := by
  sorry

end t1_eq_t2_l1409_140913


namespace simplify_expression_l1409_140947

theorem simplify_expression :
  (1 * 2 * a * 3 * a^2 * 4 * a^3 * 5 * a^4 * 6 * a^5) = 720 * a^15 :=
by
  sorry

end simplify_expression_l1409_140947


namespace total_fencing_cost_l1409_140943

theorem total_fencing_cost
  (park_is_square : true)
  (cost_per_side : ℕ)
  (h1 : cost_per_side = 43) :
  4 * cost_per_side = 172 :=
by
  sorry

end total_fencing_cost_l1409_140943


namespace perimeter_change_l1409_140916

theorem perimeter_change (s h : ℝ) 
  (h1 : 2 * (1.3 * s + 0.8 * h) = 2 * (s + h)) :
  (2 * (0.8 * s + 1.3 * h) = 1.1 * (2 * (s + h))) :=
by
  sorry

end perimeter_change_l1409_140916


namespace sum_of_M_l1409_140938

theorem sum_of_M (x y z w M : ℕ) (hxw : w = x + y + z) (hM : M = x * y * z * w) (hM_cond : M = 12 * (x + y + z + w)) :
  ∃ sum_M, sum_M = 2208 :=
by 
  sorry

end sum_of_M_l1409_140938


namespace fraction_of_phone_numbers_begin_with_8_and_end_with_5_l1409_140917

theorem fraction_of_phone_numbers_begin_with_8_and_end_with_5 :
  let total_numbers := 7 * 10^7
  let specific_numbers := 10^6
  specific_numbers / total_numbers = 1 / 70 := by
  sorry

end fraction_of_phone_numbers_begin_with_8_and_end_with_5_l1409_140917


namespace find_k_l1409_140985

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (2, -3)

def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_k (k : ℝ) :
  is_perpendicular (k • vector_a - 2 • vector_b) vector_a ↔ k = -1 :=
sorry

end find_k_l1409_140985


namespace sum_primes_between_20_and_40_l1409_140906

theorem sum_primes_between_20_and_40 :
  23 + 29 + 31 + 37 = 120 :=
by
  sorry

end sum_primes_between_20_and_40_l1409_140906


namespace g_minus_1001_l1409_140900

def g (x : ℝ) : ℝ := sorry

theorem g_minus_1001 :
  (∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x) →
  g 1 = 3 →
  g (-1001) = 1005 :=
by
  intros h1 h2
  sorry

end g_minus_1001_l1409_140900


namespace part1_sales_increase_part2_price_reduction_l1409_140962

-- Part 1: If the price is reduced by 4 yuan, the new average daily sales will be 28 items.
theorem part1_sales_increase (initial_sales : ℕ) (increase_per_yuan : ℕ) (reduction : ℕ) :
  initial_sales = 20 → increase_per_yuan = 2 → reduction = 4 →
  initial_sales + increase_per_yuan * reduction = 28 :=
by sorry

-- Part 2: By how much should the price of each item be reduced for a daily profit of 1050 yuan.
theorem part2_price_reduction (initial_sales : ℕ) (increase_per_yuan : ℕ) (initial_profit : ℕ) 
  (target_profit : ℕ) (min_profit_per_item : ℕ) (x : ℕ) :
  initial_sales = 20 → increase_per_yuan = 2 → initial_profit = 40 → target_profit = 1050 
  → min_profit_per_item = 25 → (40 - x) * (20 + 2 * x) = 1050 → (40 - x) ≥ 25 → x = 5 :=
by sorry

end part1_sales_increase_part2_price_reduction_l1409_140962


namespace complex_modulus_proof_l1409_140927

noncomputable def complex_modulus_example : ℝ := 
  Complex.abs ⟨3/4, -3⟩

theorem complex_modulus_proof : complex_modulus_example = Real.sqrt 153 / 4 := 
by 
  unfold complex_modulus_example
  sorry

end complex_modulus_proof_l1409_140927


namespace solution_inequality_equivalence_l1409_140935

-- Define the inequality to be proved
def inequality (x : ℝ) : Prop :=
  (x + 1 / 2) * (3 / 2 - x) ≥ 0

-- Define the set of solutions such that -1/2 ≤ x ≤ 3/2
def solution_set (x : ℝ) : Prop :=
  -1 / 2 ≤ x ∧ x ≤ 3 / 2

-- The statement to be proved: the solution set of the inequality is {x | -1/2 ≤ x ≤ 3/2}
theorem solution_inequality_equivalence :
  {x : ℝ | inequality x} = {x : ℝ | solution_set x} :=
by 
  sorry

end solution_inequality_equivalence_l1409_140935


namespace number_of_students_l1409_140975

theorem number_of_students (n : ℕ) (h1 : n < 40) (h2 : n % 7 = 3) (h3 : n % 6 = 1) : n = 31 := 
by
  sorry

end number_of_students_l1409_140975


namespace percentage_of_total_is_sixty_l1409_140974

def num_boys := 600
def diff_boys_girls := 400
def num_girls := num_boys + diff_boys_girls
def total_people := num_boys + num_girls
def target_number := 960
def target_percentage := (target_number / total_people) * 100

theorem percentage_of_total_is_sixty :
  target_percentage = 60 := by
  sorry

end percentage_of_total_is_sixty_l1409_140974


namespace intersecting_lines_find_m_l1409_140909

theorem intersecting_lines_find_m : ∃ m : ℚ, 
  (∃ x y : ℚ, y = 4*x + 2 ∧ y = -3*x - 18 ∧ y = 2*x + m) ↔ m = -26/7 :=
by
  sorry

end intersecting_lines_find_m_l1409_140909


namespace math_problem_l1409_140991

def a : ℕ := 2013
def b : ℕ := 2014

theorem math_problem :
  (a^3 - 2 * a^2 * b + 3 * a * b^2 - b^3 + 1) / (a * b) = a := by
  sorry

end math_problem_l1409_140991


namespace find_operation_l1409_140922

theorem find_operation (a b : ℝ) (h_a : a = 0.137) (h_b : b = 0.098) :
  ((a + b) ^ 2 - (a - b) ^ 2) / (a * b) = 4 :=
by
  sorry

end find_operation_l1409_140922


namespace Mr_Kishore_Savings_l1409_140923

noncomputable def total_expenses := 
  5000 + 1500 + 4500 + 2500 + 2000 + 6100 + 3500 + 2700

noncomputable def monthly_salary (S : ℝ) := 
  total_expenses + 0.10 * S = S

noncomputable def savings (S : ℝ) := 
  0.10 * S

theorem Mr_Kishore_Savings : 
  ∃ S : ℝ, monthly_salary S ∧ savings S = 3422.22 :=
by
  sorry

end Mr_Kishore_Savings_l1409_140923


namespace symmetric_point_coordinates_l1409_140940

theorem symmetric_point_coordinates :
  ∀ (M N : ℝ × ℝ), M = (3, -4) ∧ M.fst = -N.fst ∧ M.snd = N.snd → N = (-3, -4) :=
by
  intro M N h
  sorry

end symmetric_point_coordinates_l1409_140940


namespace solution_set_of_inequality_system_l1409_140904

theorem solution_set_of_inequality_system (x : ℝ) :
  (4 * x + 1 > 2) ∧ (1 - 2 * x < 7) ↔ (x > 1 / 4) :=
by
  sorry

end solution_set_of_inequality_system_l1409_140904


namespace largest_x_value_l1409_140948

theorem largest_x_value : ∃ x : ℝ, (x / 7 + 3 / (7 * x) = 1) ∧ (∀ y : ℝ, (y / 7 + 3 / (7 * y) = 1) → y ≤ (7 + Real.sqrt 37) / 2) :=
by
  -- (Proof of the theorem is omitted for this task)
  sorry

end largest_x_value_l1409_140948


namespace train_travel_distance_l1409_140954

theorem train_travel_distance (m : ℝ) (h : 3 * 60 * 1 = m) : m = 180 :=
by
  sorry

end train_travel_distance_l1409_140954


namespace two_digit_numbers_l1409_140950

theorem two_digit_numbers (n m : ℕ) (Hn : 1 ≤ n ∧ n ≤ 9) (Hm : n < m ∧ m ≤ 9) :
  ∃ (count : ℕ), count = 36 :=
by
  sorry

end two_digit_numbers_l1409_140950


namespace min_shirts_to_save_money_l1409_140987

theorem min_shirts_to_save_money :
  let acme_cost (x : ℕ) := 75 + 12 * x
  let gamma_cost (x : ℕ) := 18 * x
  ∀ x : ℕ, acme_cost x < gamma_cost x → x ≥ 13 := 
by
  intros
  sorry

end min_shirts_to_save_money_l1409_140987


namespace coin_count_l1409_140972

theorem coin_count (x y : ℕ) 
  (h1 : x + y = 12) 
  (h2 : 5 * x + 10 * y = 90) :
  x = 6 ∧ y = 6 := 
sorry

end coin_count_l1409_140972


namespace problem_statement_l1409_140996

variable (x : ℝ) (x₀ : ℝ)

def p : Prop := ∀ x > 0, x + 4 / x ≥ 4

def q : Prop := ∃ x₀ ∈ Set.Ioi (0 : ℝ), 2 * x₀ = 1 / 2

theorem problem_statement : p ∧ ¬q :=
by
  sorry

end problem_statement_l1409_140996


namespace sufficient_but_not_necessary_condition_l1409_140966

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h₀ : b > a) (h₁ : a > 0) :
  (1 / (a ^ 2) > 1 / (b ^ 2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1409_140966


namespace point_on_xaxis_y_coord_zero_l1409_140976

theorem point_on_xaxis_y_coord_zero (m : ℝ) (h : (3, m).snd = 0) : m = 0 :=
by 
  -- proof goes here
  sorry

end point_on_xaxis_y_coord_zero_l1409_140976


namespace count_ball_distribution_l1409_140960

theorem count_ball_distribution (A B C D : ℕ) (balls : ℕ) :
  (A + B > C + D ∧ A + B + C + D = balls) → 
  (balls = 30) →
  (∃ n, n = 2600) :=
by
  intro h_ball_dist h_balls
  sorry

end count_ball_distribution_l1409_140960


namespace exists_good_pair_for_each_m_l1409_140995

def is_good_pair (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), m * n = a^2 ∧ (m + 1) * (n + 1) = b^2

theorem exists_good_pair_for_each_m : ∀ m : ℕ, ∃ n : ℕ, m < n ∧ is_good_pair m n := by
  intro m
  let n := m * (4 * m + 3)^2
  use n
  have h1 : m < n := sorry -- Proof that m < n
  have h2 : is_good_pair m n := sorry -- Proof that (m, n) is a good pair
  exact ⟨h1, h2⟩

end exists_good_pair_for_each_m_l1409_140995


namespace rounding_effect_l1409_140937

/-- Given positive integers x, y, and z, and rounding scenarios, the
  approximation of x/y - z is necessarily less than its exact value
  when z is rounded up and x and y are rounded down. -/
theorem rounding_effect (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
(RoundXDown RoundYDown RoundZUp : ℕ → ℕ) 
(HRoundXDown : ∀ a, RoundXDown a ≤ a)
(HRoundYDown : ∀ a, RoundYDown a ≤ a)
(HRoundZUp : ∀ a, a ≤ RoundZUp a) :
  (RoundXDown x) / (RoundYDown y) - (RoundZUp z) < x / y - z :=
sorry

end rounding_effect_l1409_140937


namespace zero_of_sum_of_squares_eq_zero_l1409_140932

theorem zero_of_sum_of_squares_eq_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end zero_of_sum_of_squares_eq_zero_l1409_140932


namespace total_profit_correct_l1409_140926

variables (x y : ℝ) -- B's investment and period
variables (B_profit : ℝ) -- profit received by B
variable (A_investment : ℝ) -- A's investment

-- Given conditions
def A_investment_cond := A_investment = 3 * x
def period_cond := 2 * y
def B_profit_given := B_profit = 4500
def total_profit := 7 * B_profit

theorem total_profit_correct :
  (A_investment = 3 * x)
  ∧ (B_profit = 4500)
  ∧ ((6 * x * 2 * y) / (x * y) = 6)
  → total_profit = 31500 :=
by sorry

end total_profit_correct_l1409_140926


namespace triangle_is_isosceles_l1409_140939

theorem triangle_is_isosceles 
  (A B C : ℝ)
  (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_triangle : A + B + C = π)
  (h_condition : (Real.sin B) * (Real.sin C) = (Real.cos (A / 2)) ^ 2) :
  (B = C) :=
sorry

end triangle_is_isosceles_l1409_140939


namespace collapsing_fraction_l1409_140942

-- Define the total number of homes on Gotham St as a variable.
variable (T : ℕ)

/-- Fraction of homes on Gotham Street that are termite-ridden. -/
def fraction_termite_ridden (T : ℕ) : ℚ := 1 / 3

/-- Fraction of homes on Gotham Street that are termite-ridden but not collapsing. -/
def fraction_termite_not_collapsing (T : ℕ) : ℚ := 1 / 10

/-- Fraction of termite-ridden homes that are collapsing. -/
theorem collapsing_fraction :
  (fraction_termite_ridden T - fraction_termite_not_collapsing T) = 7 / 30 :=
by
  sorry

end collapsing_fraction_l1409_140942


namespace people_own_pets_at_least_l1409_140930

-- Definitions based on given conditions
def people_owning_only_dogs : ℕ := 15
def people_owning_only_cats : ℕ := 10
def people_owning_only_cats_and_dogs : ℕ := 5
def people_owning_cats_dogs_snakes : ℕ := 3
def total_snakes : ℕ := 59

-- Theorem statement to prove the total number of people owning pets
theorem people_own_pets_at_least : 
  people_owning_only_dogs + people_owning_only_cats + people_owning_only_cats_and_dogs + people_owning_cats_dogs_snakes ≥ 33 :=
by {
  -- Proof steps will go here
  sorry
}

end people_own_pets_at_least_l1409_140930


namespace students_passed_l1409_140934

noncomputable def total_students : ℕ := 360
noncomputable def bombed : ℕ := (5 * total_students) / 12
noncomputable def not_bombed : ℕ := total_students - bombed
noncomputable def no_show : ℕ := (7 * not_bombed) / 15
noncomputable def remaining_after_no_show : ℕ := not_bombed - no_show
noncomputable def less_than_D : ℕ := 45
noncomputable def remaining_after_less_than_D : ℕ := remaining_after_no_show - less_than_D
noncomputable def technical_issues : ℕ := remaining_after_less_than_D / 8
noncomputable def passed_students : ℕ := remaining_after_less_than_D - technical_issues

theorem students_passed : passed_students = 59 := by
  sorry

end students_passed_l1409_140934


namespace marathons_yards_l1409_140914

theorem marathons_yards
  (miles_per_marathon : ℕ)
  (yards_per_marathon : ℕ)
  (miles_in_yard : ℕ)
  (marathons_run : ℕ)
  (total_miles : ℕ)
  (total_yards : ℕ)
  (y : ℕ) :
  miles_per_marathon = 30
  → yards_per_marathon = 520
  → miles_in_yard = 1760
  → marathons_run = 8
  → total_miles = (miles_per_marathon * marathons_run) + (yards_per_marathon * marathons_run) / miles_in_yard
  → total_yards = (yards_per_marathon * marathons_run) % miles_in_yard
  → y = 640 := 
by
  intros
  sorry

end marathons_yards_l1409_140914


namespace mark_card_sum_l1409_140971

/--
Mark has seven green cards numbered 1 through 7 and five red cards numbered 2 through 6.
He arranges the cards such that colors alternate and the sum of each pair of neighboring cards forms a prime.
Prove that the sum of the numbers on the last three cards in his stack is 16.
-/
theorem mark_card_sum {green_cards : Fin 7 → ℕ} {red_cards : Fin 5 → ℕ}
  (h_green_numbered : ∀ i, 1 ≤ green_cards i ∧ green_cards i ≤ 7)
  (h_red_numbered : ∀ i, 2 ≤ red_cards i ∧ red_cards i ≤ 6)
  (h_alternate : ∀ i, i < 6 → (∃ j k, green_cards j + red_cards k = prime) ∨ (red_cards j + green_cards k = prime)) :
  ∃ s, s = 16 := sorry

end mark_card_sum_l1409_140971


namespace flagpole_height_l1409_140986

theorem flagpole_height (h : ℕ)
  (shadow_flagpole : ℕ := 72)
  (height_pole : ℕ := 18)
  (shadow_pole : ℕ := 27)
  (ratio_shadow : shadow_flagpole / shadow_pole = 8 / 3) :
  h = 48 :=
by
  sorry

end flagpole_height_l1409_140986


namespace heptagon_angle_in_arithmetic_progression_l1409_140905

theorem heptagon_angle_in_arithmetic_progression (a d : ℝ) :
  a + 3 * d = 128.57 → 
  (7 * a + 21 * d = 900) → 
  ∃ angle : ℝ, angle = 128.57 :=
by
  sorry

end heptagon_angle_in_arithmetic_progression_l1409_140905


namespace unique_ordered_triple_lcm_l1409_140981

theorem unique_ordered_triple_lcm:
  ∃! (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c), 
    Nat.lcm a b = 2100 ∧ Nat.lcm b c = 3150 ∧ Nat.lcm c a = 4200 :=
by
  sorry

end unique_ordered_triple_lcm_l1409_140981


namespace string_length_correct_l1409_140908

noncomputable def cylinder_circumference : ℝ := 6
noncomputable def cylinder_height : ℝ := 18
noncomputable def number_of_loops : ℕ := 6

noncomputable def height_per_loop : ℝ := cylinder_height / number_of_loops
noncomputable def hypotenuse_per_loop : ℝ := Real.sqrt (cylinder_circumference ^ 2 + height_per_loop ^ 2)
noncomputable def total_string_length : ℝ := number_of_loops * hypotenuse_per_loop

theorem string_length_correct :
  total_string_length = 18 * Real.sqrt 5 := by
  sorry

end string_length_correct_l1409_140908


namespace three_digit_numbers_condition_l1409_140911

theorem three_digit_numbers_condition (a b c : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) :
  (100 * a + 10 * b + c = 2 * ((10 * a + b) + (10 * a + c) + (10 * b + a) + (10 * b + c) + (10 * c + a) + (10 * c + b)))
  ↔ (100 * a + 10 * b + c = 132 ∨ 100 * a + 10 * b + c = 264 ∨ 100 * a + 10 * b + c = 396) :=
by
  sorry

end three_digit_numbers_condition_l1409_140911


namespace find_volume_of_sphere_l1409_140967

noncomputable def volume_of_sphere (AB BC AA1 : ℝ) (hAB : AB = 2) (hBC : BC = 2) (hAA1 : AA1 = 2 * Real.sqrt 2) : ℝ :=
  let diagonal := Real.sqrt (AB^2 + BC^2 + AA1^2)
  let radius := diagonal / 2
  (4 * Real.pi * radius^3) / 3

theorem find_volume_of_sphere : volume_of_sphere 2 2 (2 * Real.sqrt 2) (by rfl) (by rfl) (by rfl) = (32 * Real.pi) / 3 :=
by
  sorry

end find_volume_of_sphere_l1409_140967


namespace value_of_a_l1409_140944

theorem value_of_a (a : ℝ) (h1 : a < 0) (h2 : |a| = 3) : a = -3 := 
by
  sorry

end value_of_a_l1409_140944


namespace intersection_eq_inter_l1409_140903

noncomputable def M : Set ℝ := { x | x^2 < 4 }
noncomputable def N : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
noncomputable def inter : Set ℝ := { x | -1 < x ∧ x < 2 }

theorem intersection_eq_inter : M ∩ N = inter :=
by sorry

end intersection_eq_inter_l1409_140903


namespace stamp_book_gcd_l1409_140925

theorem stamp_book_gcd (total1 total2 total3 : ℕ) 
    (h1 : total1 = 945) (h2 : total2 = 1260) (h3 : total3 = 630) : 
    ∃ d, d = Nat.gcd (Nat.gcd total1 total2) total3 ∧ d = 315 := 
by
  sorry

end stamp_book_gcd_l1409_140925


namespace age_of_B_l1409_140915

variables (A B C : ℝ)

theorem age_of_B :
  (A + B + C) / 3 = 26 →
  (A + C) / 2 = 29 →
  B = 20 :=
by
  intro h1 h2
  sorry

end age_of_B_l1409_140915


namespace mad_hatter_must_secure_at_least_70_percent_l1409_140949

theorem mad_hatter_must_secure_at_least_70_percent :
  ∀ (N : ℕ) (uM uH uD : ℝ) (α : ℝ),
    uM = 0.2 ∧ uH = 0.25 ∧ uD = 0.3 → 
    uM + α * 0.25 ≥ 0.25 + (1 - α) * 0.25 ∧
    uM + α * 0.25 ≥ 0.3 + (1 - α) * 0.25 →
    α ≥ 0.7 :=
by
  intros N uM uH uD α h hx
  sorry 

end mad_hatter_must_secure_at_least_70_percent_l1409_140949


namespace find_n_l1409_140955

theorem find_n (x n : ℤ) (k m : ℤ) (h1 : x = 82*k + 5) (h2 : x + n = 41*m + 22) : n = 5 := by
  sorry

end find_n_l1409_140955


namespace minimum_common_perimeter_l1409_140902

namespace IsoscelesTriangles

def integer_sided_isosceles_triangles (a b x : ℕ) :=
  2 * a + 10 * x = 2 * b + 8 * x ∧
  5 * Real.sqrt (a^2 - 25 * x^2) = 4 * Real.sqrt (b^2 - 16 * x^2) ∧
  5 * b = 4 * (b + x)

theorem minimum_common_perimeter : ∃ (a b x : ℕ), 
  integer_sided_isosceles_triangles a b x ∧
  2 * a + 10 * x = 192 :=
by
  sorry

end IsoscelesTriangles

end minimum_common_perimeter_l1409_140902


namespace probability_of_sum_leq_10_l1409_140969

open Nat

-- Define the three dice roll outcomes
def dice_outcomes := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define the total number of outcomes when rolling three dice
def total_outcomes : ℕ := 6 ^ 3

-- Count the number of valid outcomes where the sum of three dice is less than or equal to 10
def count_valid_outcomes : ℕ := 75  -- This is determined through combinatorial calculations or software

-- Define the desired probability
def desired_probability := (count_valid_outcomes : ℚ) / total_outcomes

-- Prove that the desired probability equals 25/72
theorem probability_of_sum_leq_10 :
  desired_probability = 25 / 72 :=
by sorry

end probability_of_sum_leq_10_l1409_140969


namespace solve_abs_inequality_l1409_140907

theorem solve_abs_inequality (x : ℝ) : x + |2 * x + 3| ≥ 2 ↔ (x ≤ -5 ∨ x ≥ -1/3) :=
by {
  sorry
}

end solve_abs_inequality_l1409_140907


namespace sum_of_solutions_l1409_140982

theorem sum_of_solutions (x : ℝ) (h : x + 16 / x = 12) : x = 8 ∨ x = 4 → 8 + 4 = 12 := by
  sorry

end sum_of_solutions_l1409_140982


namespace geometric_sequence_ratio_l1409_140956

theorem geometric_sequence_ratio (a : ℕ → ℤ) (q : ℤ) (n : ℕ) (i : ℕ → ℕ) (ε : ℕ → ℤ) :
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → a k = a 1 * q ^ (k - 1)) ∧
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n → ε k * a (i k) = 0) ∧
  (∀ m, 1 ≤ i m ∧ i m ≤ n) → q = -1 := 
sorry

end geometric_sequence_ratio_l1409_140956


namespace dima_age_l1409_140924

variable (x : ℕ)

-- Dima's age is x years
def age_of_dima := x

-- Dima's age is twice his brother's age
def age_of_brother := x / 2

-- Dima's age is three times his sister's age
def age_of_sister := x / 3

-- The average age of Dima, his sister, and his brother is 11 years
def average_age := (x + age_of_brother x + age_of_sister x) / 3 = 11

theorem dima_age (h1 : age_of_brother x = x / 2) 
                 (h2 : age_of_sister x = x / 3) 
                 (h3 : average_age x) : x = 18 := 
by sorry

end dima_age_l1409_140924


namespace initial_packs_l1409_140990

def num_invitations_per_pack := 3
def num_friends := 9
def extra_invitations := 3
def total_invitations := num_friends + extra_invitations

theorem initial_packs (h : total_invitations = 12) : (total_invitations / num_invitations_per_pack) = 4 :=
by
  have h1 : total_invitations = 12 := by exact h
  have h2 : num_invitations_per_pack = 3 := by exact rfl
  have H_pack : total_invitations / num_invitations_per_pack = 4 := by sorry
  exact H_pack

end initial_packs_l1409_140990


namespace magic_square_sum_l1409_140980

theorem magic_square_sum (S a b c d e : ℤ) (h1 : x + 15 + 100 = S)
                        (h2 : 23 + d + e = S)
                        (h3 : x + a + 23 = S)
                        (h4 : a = 92)
                        (h5 : 92 + b + d = x + 15 + 100)
                        (h6 : b = 0)
                        (h7 : d = 100) : x = 77 :=
by {
  sorry
}

end magic_square_sum_l1409_140980


namespace problem_lean_version_l1409_140931

theorem problem_lean_version (n : ℕ) : 
  (n > 0) ∧ (6^n - 1 ∣ 7^n - 1) ↔ ∃ k : ℕ, n = 4 * k :=
by
  sorry

end problem_lean_version_l1409_140931


namespace man_l1409_140963

-- Conditions
def speed_with_current : ℝ := 18
def speed_of_current : ℝ := 3.4

-- Problem statement
theorem man's_speed_against_current :
  (speed_with_current - speed_of_current - speed_of_current) = 11.2 := 
by
  sorry

end man_l1409_140963


namespace degree_measure_of_regular_hexagon_interior_angle_l1409_140993

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l1409_140993


namespace equivalent_region_l1409_140999

def satisfies_conditions (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 2 ∧ -1 ≤ x / (x + y) ∧ x / (x + y) ≤ 1

def region (x y : ℝ) : Prop :=
  y ≥ 0 ∧ y ≥ -2*x ∧ x^2 + y^2 ≤ 2

theorem equivalent_region (x y : ℝ) :
  satisfies_conditions x y = region x y := 
sorry

end equivalent_region_l1409_140999


namespace solve_equation_l1409_140992

theorem solve_equation (x : ℝ) : 2 * x - 4 = 0 ↔ x = 2 :=
by sorry

end solve_equation_l1409_140992


namespace width_of_wall_is_two_l1409_140910

noncomputable def volume_of_brick : ℝ := 20 * 10 * 7.5 / 10^6 -- Volume in cubic meters
def number_of_bricks : ℕ := 27000
noncomputable def volume_of_wall (width : ℝ) : ℝ := 27 * width * 0.75

theorem width_of_wall_is_two :
  ∃ (W : ℝ), volume_of_wall W = number_of_bricks * volume_of_brick ∧ W = 2 :=
by
  sorry

end width_of_wall_is_two_l1409_140910


namespace semicircle_inequality_l1409_140979

open Real

theorem semicircle_inequality {A B C D E : ℝ} (h : A^2 + B^2 + C^2 + D^2 + E^2 = 1):
  (A - B)^2 + (B - C)^2 + (C - D)^2 + (D - E)^2 + (A - B) * (B - C) * (C - D) + (B - C) * (C - D) * (D - E) < 4 :=
by
  -- proof omitted
  sorry

end semicircle_inequality_l1409_140979


namespace total_sheep_l1409_140933

theorem total_sheep (n : ℕ) 
  (h1 : 3 ∣ n)
  (h2 : 5 ∣ n)
  (h3 : 6 ∣ n)
  (h4 : 8 ∣ n)
  (h5 : n * 7 / 40 = 12) : 
  n = 68 :=
by
  sorry

end total_sheep_l1409_140933


namespace unique_arrangements_MOON_l1409_140951

theorem unique_arrangements_MOON : 
  let M := 1
  let O := 2
  let N := 1
  let total_letters := 4
  (Nat.factorial total_letters / (Nat.factorial O)) = 12 :=
by
  sorry

end unique_arrangements_MOON_l1409_140951
