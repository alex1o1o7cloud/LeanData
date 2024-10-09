import Mathlib

namespace first_platform_length_is_150_l2078_207877

-- Defining the conditions
def train_length : ℝ := 150
def first_platform_time : ℝ := 15
def second_platform_length : ℝ := 250
def second_platform_time : ℝ := 20

-- The distance covered when crossing the first platform is length of train + length of first platform
def distance_first_platform (L : ℝ) : ℝ := train_length + L

-- The distance covered when crossing the second platform is length of train + length of a known 250 m platform
def distance_second_platform : ℝ := train_length + second_platform_length

-- We are to prove that the length of the first platform, given the conditions, is 150 meters.
theorem first_platform_length_is_150 : ∃ L : ℝ, (distance_first_platform L / distance_second_platform) = (first_platform_time / second_platform_time) ∧ L = 150 :=
by
  let L := 150
  have h1 : distance_first_platform L = train_length + L := rfl
  have h2 : distance_second_platform = train_length + second_platform_length := rfl
  have h3 : distance_first_platform L / distance_second_platform = first_platform_time / second_platform_time :=
    by sorry
  use L
  exact ⟨h3, rfl⟩

end first_platform_length_is_150_l2078_207877


namespace shaded_percentage_l2078_207802

noncomputable def percent_shaded (side_len : ℕ) : ℝ :=
  let total_area := (side_len : ℝ) * side_len
  let shaded_area := (2 * 2) + (2 * 5) + (1 * 7)
  100 * (shaded_area / total_area)

theorem shaded_percentage (PQRS_side : ℕ) (hPQRS : PQRS_side = 7) :
  percent_shaded PQRS_side = 42.857 :=
  by
  rw [hPQRS]
  sorry

end shaded_percentage_l2078_207802


namespace pythagorean_theorem_l2078_207845

theorem pythagorean_theorem {a b c p q : ℝ} 
  (h₁ : p * c = a ^ 2) 
  (h₂ : q * c = b ^ 2)
  (h₃ : p + q = c) : 
  c ^ 2 = a ^ 2 + b ^ 2 := 
by 
  sorry

end pythagorean_theorem_l2078_207845


namespace percentage_of_40_eq_140_l2078_207854

theorem percentage_of_40_eq_140 (p : ℝ) (h : (p / 100) * 40 = 140) : p = 350 :=
sorry

end percentage_of_40_eq_140_l2078_207854


namespace problem_statement_l2078_207876

def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := x - 2

theorem problem_statement : f (g 5) - g (f 5) = -8 := by sorry

end problem_statement_l2078_207876


namespace simplify_fraction_l2078_207817

theorem simplify_fraction (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 
  (12 * x * y^3) / (9 * x^2 * y^2) = 16 / 9 :=
by
  sorry

end simplify_fraction_l2078_207817


namespace triangle_area_is_120_l2078_207819

-- Define the triangle sides
def a : ℕ := 10
def b : ℕ := 24
def c : ℕ := 26

-- Define a function to calculate the area of a right-angled triangle
noncomputable def right_triangle_area (a b : ℕ) : ℕ := (a * b) / 2

-- Statement to prove the area of the triangle
theorem triangle_area_is_120 : right_triangle_area 10 24 = 120 :=
by
  sorry

end triangle_area_is_120_l2078_207819


namespace sum_first_2009_terms_arith_seq_l2078_207856

variable {a : ℕ → ℝ}

-- Given condition a_1004 + a_1005 + a_1006 = 3
axiom H : a 1004 + a 1005 + a 1006 = 3

-- Arithmetic sequence definition
def is_arith_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_arith_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem sum_first_2009_terms_arith_seq
  (d : ℝ) (h_arith_seq : is_arith_seq a d)
  : sum_arith_seq a 2009 = 2009 := 
by
  sorry

end sum_first_2009_terms_arith_seq_l2078_207856


namespace calculate_expression_l2078_207842

theorem calculate_expression :
  ((650^2 - 350^2) * 3 = 900000) := by
  sorry

end calculate_expression_l2078_207842


namespace range_of_a_l2078_207801

noncomputable def f (x a : ℝ) := Real.exp (-x) - 2 * x - a

def curve (x : ℝ) := x ^ 3 + x

def y_in_range (x : ℝ) := x >= -2 ∧ x <= 2

theorem range_of_a : ∀ (a : ℝ), (∃ x, y_in_range (curve x) ∧ f (curve x) a = curve x) ↔ a ∈ Set.Icc (Real.exp (-2) - 6) (Real.exp 2 + 6) := by
  sorry

end range_of_a_l2078_207801


namespace perfect_square_factors_count_450_l2078_207879

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l2078_207879


namespace vector_computation_l2078_207816

def c : ℝ × ℝ × ℝ := (-3, 5, 2)
def d : ℝ × ℝ × ℝ := (5, -1, 3)

theorem vector_computation : 2 • c - 5 • d + c = (-34, 20, -9) := by
  sorry

end vector_computation_l2078_207816


namespace max_area_rectangle_l2078_207823

theorem max_area_rectangle (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 60) 
  (h2 : l - w = 10) : 
  l * w = 200 := 
by
  sorry

end max_area_rectangle_l2078_207823


namespace sandy_money_taken_l2078_207881

-- Condition: Let T be the total money Sandy took for shopping, and it is known that 70% * T = $224
variable (T : ℝ)
axiom h : 0.70 * T = 224

-- Theorem to prove: T is 320
theorem sandy_money_taken : T = 320 :=
by 
  sorry

end sandy_money_taken_l2078_207881


namespace imaginary_part_of_f_i_div_i_is_one_l2078_207862

def f (x : ℂ) : ℂ := x^3 - 1

theorem imaginary_part_of_f_i_div_i_is_one 
    (i : ℂ) (h : i^2 = -1) :
    ( (f i) / i ).im = 1 := 
sorry

end imaginary_part_of_f_i_div_i_is_one_l2078_207862


namespace tan_subtraction_formula_l2078_207885

theorem tan_subtraction_formula 
  (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 := 
by
  sorry

end tan_subtraction_formula_l2078_207885


namespace number_of_teams_l2078_207848

-- Define the conditions
def math_club_girls : ℕ := 4
def math_club_boys : ℕ := 7
def team_girls : ℕ := 3
def team_boys : ℕ := 3

-- Compute the number of ways to choose 3 girls from 4 girls
def choose_comb_girls : ℕ := Nat.choose math_club_girls team_girls

-- Compute the number of ways to choose 3 boys from 7 boys
def choose_comb_boys : ℕ := Nat.choose math_club_boys team_boys

-- Formulate the goal statement
theorem number_of_teams : choose_comb_girls * choose_comb_boys = 140 := by
  sorry

end number_of_teams_l2078_207848


namespace plains_total_square_miles_l2078_207891

theorem plains_total_square_miles (RegionB : ℝ) (h1 : RegionB = 200) (RegionA : ℝ) (h2 : RegionA = RegionB - 50) : 
  RegionA + RegionB = 350 := 
by 
  sorry

end plains_total_square_miles_l2078_207891


namespace verify_mass_percentage_l2078_207875

-- Define the elements in HBrO3
def hydrogen : String := "H"
def bromine : String := "Br"
def oxygen : String := "O"

-- Define the given molar masses
def molar_masses (e : String) : Float :=
  if e = hydrogen then 1.01
  else if e = bromine then 79.90
  else if e = oxygen then 16.00
  else 0.0

-- Define the molar mass of HBrO3
def molar_mass_HBrO3 : Float := 128.91

-- Function to calculate mass percentage of a given element in HBrO3
def mass_percentage (e : String) : Float :=
  if e = bromine then 79.90 / molar_mass_HBrO3 * 100
  else if e = hydrogen then 1.01 / molar_mass_HBrO3 * 100
  else if e = oxygen then 48.00 / molar_mass_HBrO3 * 100
  else 0.0

-- The proof problem statement
theorem verify_mass_percentage (e : String) (h : e ∈ [hydrogen, bromine, oxygen]) : mass_percentage e = 0.78 :=
sorry

end verify_mass_percentage_l2078_207875


namespace fraction_area_of_shaded_square_in_larger_square_is_one_eighth_l2078_207822

theorem fraction_area_of_shaded_square_in_larger_square_is_one_eighth :
  let side_larger_square := 4
  let area_larger_square := side_larger_square^2
  let side_shaded_square := Real.sqrt (1^2 + 1^2)
  let area_shaded_square := side_shaded_square^2
  area_shaded_square / area_larger_square = 1 / 8 := 
by 
  sorry

end fraction_area_of_shaded_square_in_larger_square_is_one_eighth_l2078_207822


namespace tan_add_pi_over_3_l2078_207837

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = Real.sqrt 3) :
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 := 
by 
  sorry

end tan_add_pi_over_3_l2078_207837


namespace erased_angle_is_97_l2078_207830

theorem erased_angle_is_97 (n : ℕ) (h1 : 3 ≤ n) (h2 : (n - 2) * 180 = 1703 + x) : 
  1800 - 1703 = 97 :=
by sorry

end erased_angle_is_97_l2078_207830


namespace exist_positive_int_for_arithmetic_mean_of_divisors_l2078_207803

theorem exist_positive_int_for_arithmetic_mean_of_divisors
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_distinct : p ≠ q) :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 
  (∃ k : ℕ, k * (a + 1) * (b + 1) = (p^(a+1) - 1) / (p - 1) * (q^(b+1) - 1) / (q - 1)) :=
sorry

end exist_positive_int_for_arithmetic_mean_of_divisors_l2078_207803


namespace find_a1_over_1_minus_q_l2078_207839

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem find_a1_over_1_minus_q 
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h1 : a 1 + a 2 + a 3 + a 4 = 3)
  (h2 : a 5 + a 6 + a 7 + a 8 = 48) :
  (a 1) / (1 - q) = -1 / 5 :=
sorry

end find_a1_over_1_minus_q_l2078_207839


namespace students_owning_both_pets_l2078_207824

theorem students_owning_both_pets:
  ∀ (students total students_dog students_cat : ℕ),
    total = 45 →
    students_dog = 28 →
    students_cat = 38 →
    -- Each student owning at least one pet means 
    -- total = students_dog ∪ students_cat
    total = students_dog + students_cat - students →
    students = 21 :=
by
  intros students total students_dog students_cat h_total h_dog h_cat h_union
  sorry

end students_owning_both_pets_l2078_207824


namespace evaluate_propositions_l2078_207871

variable (x y : ℝ)

def p : Prop := (x > y) → (-x < -y)
def q : Prop := (x < y) → (x^2 > y^2)

theorem evaluate_propositions : (p x y ∨ q x y) ∧ (p x y ∧ ¬q x y) := by
  -- Correct answer: \( \boxed{\text{C}} \)
  sorry

end evaluate_propositions_l2078_207871


namespace find_x_l2078_207851

theorem find_x (x : ℕ) (hv1 : x % 6 = 0) (hv2 : x^2 > 144) (hv3 : x < 30) : x = 18 ∨ x = 24 :=
  sorry

end find_x_l2078_207851


namespace magician_decks_l2078_207890

theorem magician_decks :
  ∀ (initial_decks price_per_deck earnings decks_sold decks_left_unsold : ℕ),
  initial_decks = 5 →
  price_per_deck = 2 →
  earnings = 4 →
  decks_sold = earnings / price_per_deck →
  decks_left_unsold = initial_decks - decks_sold →
  decks_left_unsold = 3 :=
by
  intros initial_decks price_per_deck earnings decks_sold decks_left_unsold
  intros h_initial h_price h_earnings h_sold h_left
  rw [h_initial, h_price, h_earnings] at *
  sorry

end magician_decks_l2078_207890


namespace shopkeeper_loss_percent_l2078_207867

theorem shopkeeper_loss_percent (cost_price goods_lost_percent profit_percent : ℝ)
    (h_cost_price : cost_price = 100)
    (h_goods_lost_percent : goods_lost_percent = 0.4)
    (h_profit_percent : profit_percent = 0.1) :
    let initial_revenue := cost_price * (1 + profit_percent)
    let goods_lost_value := cost_price * goods_lost_percent
    let remaining_goods_value := cost_price - goods_lost_value
    let remaining_revenue := remaining_goods_value * (1 + profit_percent)
    let loss_in_revenue := initial_revenue - remaining_revenue
    let loss_percent := (loss_in_revenue / initial_revenue) * 100
    loss_percent = 40 := sorry

end shopkeeper_loss_percent_l2078_207867


namespace reducedRatesFraction_l2078_207863

variable (total_hours_per_week : ℕ := 168)
variable (reduced_rate_hours_weekdays : ℕ := 12 * 5)
variable (reduced_rate_hours_weekends : ℕ := 24 * 2)

theorem reducedRatesFraction
  (h1 : total_hours_per_week = 7 * 24)
  (h2 : reduced_rate_hours_weekdays = 12 * 5)
  (h3 : reduced_rate_hours_weekends = 24 * 2) :
  (reduced_rate_hours_weekdays + reduced_rate_hours_weekends) / total_hours_per_week = 9 / 14 := 
  sorry

end reducedRatesFraction_l2078_207863


namespace exp_fn_max_min_diff_l2078_207878

theorem exp_fn_max_min_diff (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (max (a^1) (a^0) - min (a^1) (a^0)) = 1 / 2 → (a = 1 / 2 ∨ a = 3 / 2) :=
by
  sorry

end exp_fn_max_min_diff_l2078_207878


namespace number_of_educated_employees_l2078_207812

-- Define the context and input values
variable (T: ℕ) (I: ℕ := 20) (decrease_illiterate: ℕ := 15) (total_decrease_illiterate: ℕ := I * decrease_illiterate) (average_salary_decrease: ℕ := 10)

-- The theorem statement
theorem number_of_educated_employees (h1: total_decrease_illiterate / T = average_salary_decrease) (h2: T = I + 10): L = 10 := by
  sorry

end number_of_educated_employees_l2078_207812


namespace diagonal_AC_possibilities_l2078_207846

/-
In a quadrilateral with sides AB, BC, CD, and DA, the length of diagonal AC must 
satisfy the inequalities determined by the triangle inequalities for triangles 
ABC and CDA. Prove the number of different whole numbers that could be the 
length of diagonal AC is 13.
-/

def number_of_whole_numbers_AC (AB BC CD DA : ℕ) : ℕ :=
  if 6 < AB ∧ AB < 20 then 19 - 7 + 1 else sorry

theorem diagonal_AC_possibilities : number_of_whole_numbers_AC 7 13 15 10 = 13 :=
  by
    sorry

end diagonal_AC_possibilities_l2078_207846


namespace sqrt_of_1024_l2078_207861

theorem sqrt_of_1024 (x : ℝ) (h1 : x > 0) (h2 : x ^ 2 = 1024) : x = 32 :=
sorry

end sqrt_of_1024_l2078_207861


namespace num_distinct_ordered_pairs_l2078_207843

theorem num_distinct_ordered_pairs (a b c : ℕ) (h₀ : a + b + c = 50) (h₁ : c = 10) (h₂ : 0 < a ∧ 0 < b) :
  ∃ n : ℕ, n = 39 := 
sorry

end num_distinct_ordered_pairs_l2078_207843


namespace marbles_remaining_correct_l2078_207899

-- Define the number of marbles Chris has
def marbles_chris : ℕ := 12

-- Define the number of marbles Ryan has
def marbles_ryan : ℕ := 28

-- Define the total number of marbles in the pile
def total_marbles : ℕ := marbles_chris + marbles_ryan

-- Define the number of marbles each person takes away from the pile
def marbles_taken_each : ℕ := total_marbles / 4

-- Define the total number of marbles taken away
def total_marbles_taken : ℕ := 2 * marbles_taken_each

-- Define the number of marbles remaining in the pile
def marbles_remaining : ℕ := total_marbles - total_marbles_taken

theorem marbles_remaining_correct : marbles_remaining = 20 := by
  sorry

end marbles_remaining_correct_l2078_207899


namespace range_of_a_l2078_207827

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Ioo a (a + 1), ∃ f' : ℝ → ℝ, ∀ x, f' x = (x * Real.exp x) * (x + 2) ∧ f' x = 0) ↔ 
  a ∈ Set.Ioo (-3 : ℝ) (-2) ∪ Set.Ioo (-1) (0) := 
sorry

end range_of_a_l2078_207827


namespace length_other_diagonal_l2078_207895

variables (d1 d2 : ℝ) (Area : ℝ)

theorem length_other_diagonal 
  (h1 : Area = 432)
  (h2 : d1 = 36) :
  d2 = 24 :=
by
  -- Insert proof here
  sorry

end length_other_diagonal_l2078_207895


namespace arithmetic_sequence_second_term_l2078_207866

theorem arithmetic_sequence_second_term (a1 a5 : ℝ) (h1 : a1 = 2020) (h5 : a5 = 4040) : 
  ∃ d a2 : ℝ, a2 = a1 + d ∧ d = (a5 - a1) / 4 ∧ a2 = 2525 :=
by
  sorry

end arithmetic_sequence_second_term_l2078_207866


namespace initial_marbles_l2078_207804

theorem initial_marbles (M : ℝ) (h0 : 0.2 * M + 0.35 * (0.8 * M) + 130 = M) : M = 250 :=
by
  sorry

end initial_marbles_l2078_207804


namespace slope_parallel_to_line_l2078_207896

theorem slope_parallel_to_line (x y : ℝ) (h : 3 * x - 6 * y = 15) :
  (∃ m, (∀ b, y = m * x + b) ∧ (∀ k, k ≠ m → ¬ 3 * x - 6 * (k * x + b) = 15)) →
  ∃ p, p = 1/2 :=
sorry

end slope_parallel_to_line_l2078_207896


namespace no_square_number_divisible_by_six_in_range_l2078_207873

theorem no_square_number_divisible_by_six_in_range :
  ¬ ∃ x : ℕ, (∃ n : ℕ, x = n^2) ∧ (6 ∣ x) ∧ (50 < x) ∧ (x < 120) :=
by
  sorry

end no_square_number_divisible_by_six_in_range_l2078_207873


namespace solve_inequality_l2078_207855

theorem solve_inequality : {x : ℝ | (2 * x - 7) * (x - 3) / x ≥ 0} = {x | (0 < x ∧ x ≤ 3) ∨ (x ≥ 7 / 2)} :=
by
  sorry

end solve_inequality_l2078_207855


namespace difference_of_roots_l2078_207820

theorem difference_of_roots :
  ∀ (x : ℝ), (x^2 - 5*x + 6 = 0) → (∃ r1 r2 : ℝ, r1 > 2 ∧ r2 < r1 ∧ r1 - r2 = 1) :=
by
  sorry

end difference_of_roots_l2078_207820


namespace solve_equation_l2078_207886

theorem solve_equation (x : ℝ) (hx : (x + 1) ≠ 0) :
  (x = -3 / 4) ∨ (x = -1) ↔ (x^3 + x^2 + x + 1) / (x + 1) = x^2 + 4 * x + 4 :=
by
  sorry

end solve_equation_l2078_207886


namespace min_value_x_3y_6z_l2078_207880

theorem min_value_x_3y_6z (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0 ∧ xyz = 27) : x + 3 * y + 6 * z ≥ 27 :=
sorry

end min_value_x_3y_6z_l2078_207880


namespace line_passes_through_circle_center_l2078_207809

theorem line_passes_through_circle_center (a : ℝ) :
  (∃ x y : ℝ, (x^2 + y^2 + 2*x - 4*y = 0) ∧ (3*x + y + a = 0)) → a = 1 :=
by
  sorry

end line_passes_through_circle_center_l2078_207809


namespace polygon_interior_angle_sum_l2078_207887

theorem polygon_interior_angle_sum (n : ℕ) (h : (n - 2) * 180 = 1800) : n = 12 :=
by sorry

end polygon_interior_angle_sum_l2078_207887


namespace percent_owning_only_cats_l2078_207857

theorem percent_owning_only_cats (total_students dogs cats both : ℕ) (h1 : total_students = 500)
  (h2 : dogs = 150) (h3 : cats = 80) (h4 : both = 25) : (cats - both) / total_students * 100 = 11 :=
by
  sorry

end percent_owning_only_cats_l2078_207857


namespace carla_catches_up_in_three_hours_l2078_207808

-- Definitions as lean statements based on conditions
def john_speed : ℝ := 30
def carla_speed : ℝ := 35
def john_start_time : ℝ := 0
def carla_start_time : ℝ := 0.5

-- Lean problem statement to prove the catch-up time
theorem carla_catches_up_in_three_hours : 
  ∃ t : ℝ, 35 * t = 30 * (t + 0.5) ∧ t = 3 :=
by
  sorry

end carla_catches_up_in_three_hours_l2078_207808


namespace determine_constants_and_sum_l2078_207898

theorem determine_constants_and_sum (A B C x : ℝ) (h₁ : A = 3) (h₂ : B = 5) (h₃ : C = 40 / 3)
  (h₄ : (x + B) * (A * x + 40) / ((x + C) * (x + 5)) = 3) :
  ∀ x : ℝ, x ≠ -5 → x ≠ -40 / 3 → (-(5 : ℝ) + -40 / 3 = -55 / 3) :=
sorry

end determine_constants_and_sum_l2078_207898


namespace rob_has_12_pennies_l2078_207893

def total_value_in_dollars (quarters dimes nickels pennies : ℕ) : ℚ :=
  (quarters * 25 + dimes * 10 + nickels * 5 + pennies) / 100

theorem rob_has_12_pennies
  (quarters : ℕ) (dimes : ℕ) (nickels : ℕ) (pennies : ℕ)
  (h1 : quarters = 7) (h2 : dimes = 3) (h3 : nickels = 5) 
  (h4 : total_value_in_dollars quarters dimes nickels pennies = 2.42) :
  pennies = 12 :=
by
  sorry

end rob_has_12_pennies_l2078_207893


namespace angle_bisector_slope_l2078_207849

theorem angle_bisector_slope :
  ∀ m1 m2 : ℝ, m1 = 2 → m2 = 4 → (∃ k : ℝ, k = (6 - Real.sqrt 21) / (-7) → k = (-6 + Real.sqrt 21) / 7) :=
by
  sorry

end angle_bisector_slope_l2078_207849


namespace polygon_sides_eq_nine_l2078_207818

theorem polygon_sides_eq_nine (n : ℕ) 
  (interior_sum : ℕ := (n - 2) * 180)
  (exterior_sum : ℕ := 360)
  (condition : interior_sum = 4 * exterior_sum - 180) : 
  n = 9 :=
by {
  sorry
}

end polygon_sides_eq_nine_l2078_207818


namespace toys_in_row_l2078_207844

theorem toys_in_row (n_left n_right : ℕ) (hy : 10 = n_left + 1) (hy' : 7 = n_right + 1) :
  n_left + n_right + 1 = 16 :=
by
  -- Fill in the proof here
  sorry

end toys_in_row_l2078_207844


namespace root_sum_abs_gt_6_l2078_207834

variables (r1 r2 p : ℝ)

theorem root_sum_abs_gt_6 
  (h1 : r1 + r2 = -p)
  (h2 : r1 * r2 = 9)
  (h3 : p^2 > 36) :
  |r1 + r2| > 6 :=
by sorry

end root_sum_abs_gt_6_l2078_207834


namespace cubic_sum_l2078_207805

theorem cubic_sum (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 12) : x^3 + y^3 = 224 :=
by
  sorry

end cubic_sum_l2078_207805


namespace percentage_calculation_l2078_207882

theorem percentage_calculation (P : ℝ) : 
    (P / 100) * 24 + 0.10 * 40 = 5.92 ↔ P = 8 :=
by 
    sorry

end percentage_calculation_l2078_207882


namespace rita_bought_4_pounds_l2078_207847

-- Define the conditions
def card_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def amount_left : ℝ := 35.68

-- Define the theorem to prove the number of pounds of coffee bought is 4
theorem rita_bought_4_pounds :
  (card_amount - amount_left) / cost_per_pound = 4 := by sorry

end rita_bought_4_pounds_l2078_207847


namespace arithmetic_mean_of_fractions_l2078_207897

theorem arithmetic_mean_of_fractions :
  let a := (5 : ℚ) / 8
  let b := (9 : ℚ) / 16
  let c := (11 : ℚ) / 16
  a = (b + c) / 2 := by
  sorry

end arithmetic_mean_of_fractions_l2078_207897


namespace correct_operation_l2078_207815

theorem correct_operation (a b : ℝ) : ((-3 * a^2 * b)^2 = 9 * a^4 * b^2) := sorry

end correct_operation_l2078_207815


namespace ice_cream_orders_l2078_207807

variables (V C S M O T : ℕ)

theorem ice_cream_orders :
  (V = 56) ∧ (C = 28) ∧ (S = 70) ∧ (M = 42) ∧ (O = 84) ↔
  (V = 2 * C) ∧
  (S = 25 * T / 100) ∧
  (M = 15 * T / 100) ∧
  (T = 280) ∧
  (V = 20 * T / 100) ∧
  (V + C + S + M + O = T) :=
by
  sorry

end ice_cream_orders_l2078_207807


namespace exists_integers_x_l2078_207814

theorem exists_integers_x (a1 a2 a3 : ℤ) (h : 0 < a1 ∧ a1 < a2 ∧ a2 < a3) :
  ∃ (x1 x2 x3 : ℤ), (|x1| + |x2| + |x3| > 0) ∧ (a1 * x1 + a2 * x2 + a3 * x3 = 0) ∧ (max (max (|x1|) (|x2|)) (|x3|) < (2 / Real.sqrt 3 * Real.sqrt a3) + 1) := 
sorry

end exists_integers_x_l2078_207814


namespace price_of_basic_computer_l2078_207894

variable (C P : ℝ)

theorem price_of_basic_computer 
    (h1 : C + P = 2500)
    (h2 : P = (1 / 6) * (C + 500 + P)) : 
  C = 2000 :=
by
  sorry

end price_of_basic_computer_l2078_207894


namespace tangent_line_l2078_207821

variable (a b x₀ y₀ x y : ℝ)
variable (h_ab : a > b)
variable (h_b0 : b > 0)

def ellipse (a b x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem tangent_line (h_el : ellipse a b x₀ y₀) : 
  (x₀ * x / a^2) + (y₀ * y / b^2) = 1 :=
sorry

end tangent_line_l2078_207821


namespace cost_per_foot_of_fence_l2078_207884

theorem cost_per_foot_of_fence 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h_area : area = 289) 
  (h_total_cost : total_cost = 4080) 
  : total_cost / (4 * (Real.sqrt area)) = 60 := 
by
  sorry

end cost_per_foot_of_fence_l2078_207884


namespace find_M_coordinate_l2078_207883

-- Definitions of the given points
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := ⟨1, 0, 2⟩
def B : Point3D := ⟨1, -3, 1⟩
def M (y : ℝ) : Point3D := ⟨0, y, 0⟩

-- Definition for the squared distance between two points
def dist_sq (p1 p2 : Point3D) : ℝ :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2

-- Main theorem statement
theorem find_M_coordinate (y : ℝ) : 
  dist_sq (M y) A = dist_sq (M y) B → y = -1 :=
by
  simp [dist_sq, A, B, M]
  sorry

end find_M_coordinate_l2078_207883


namespace sandy_correct_sums_l2078_207833

theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 45) : c = 21 :=
  sorry

end sandy_correct_sums_l2078_207833


namespace length_of_room_l2078_207838

def area_of_room : ℝ := 10
def width_of_room : ℝ := 2

theorem length_of_room : width_of_room * 5 = area_of_room :=
by
  sorry

end length_of_room_l2078_207838


namespace poly_eq_zero_or_one_l2078_207868

noncomputable def k : ℝ := 2 -- You can replace 2 with any number greater than 1.

theorem poly_eq_zero_or_one (P : ℝ → ℝ) 
  (h1 : k > 1) 
  (h2 : ∀ x : ℝ, P (x ^ k) = (P x) ^ k) : 
  (∀ x, P x = 0) ∨ (∀ x, P x = 1) :=
sorry

end poly_eq_zero_or_one_l2078_207868


namespace washer_cost_difference_l2078_207869

theorem washer_cost_difference (W D : ℝ) 
  (h1 : W + D = 1200) (h2 : D = 490) : W - D = 220 :=
sorry

end washer_cost_difference_l2078_207869


namespace third_square_area_difference_l2078_207852

def side_length (p : ℕ) : ℕ :=
  p / 4

def area (s : ℕ) : ℕ :=
  s * s

theorem third_square_area_difference
  (p1 p2 p3 : ℕ)
  (h1 : p1 = 60)
  (h2 : p2 = 48)
  (h3 : p3 = 36)
  : area (side_length p3) = area (side_length p1) - area (side_length p2) :=
by
  sorry

end third_square_area_difference_l2078_207852


namespace triangle_area_l2078_207850

theorem triangle_area :
  ∃ (a b c : ℕ), a + b + c = 12 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ ∃ (A : ℝ), 
  A = Real.sqrt (6 * (6 - a) * (6 - b) * (6 - c)) ∧ A = 6 := by
  sorry

end triangle_area_l2078_207850


namespace friend_spent_more_l2078_207840

theorem friend_spent_more (total_spent friend_spent: ℝ) (h_total: total_spent = 15) (h_friend: friend_spent = 10) :
  friend_spent - (total_spent - friend_spent) = 5 :=
by
  sorry

end friend_spent_more_l2078_207840


namespace universal_proposition_l2078_207826

def is_multiple_of_two (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2 * k

def is_even (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2 * k

theorem universal_proposition : 
  (∀ x : ℕ, is_multiple_of_two x → is_even x) :=
by
  sorry

end universal_proposition_l2078_207826


namespace trigonometric_inequality_l2078_207836

theorem trigonometric_inequality (a b : ℝ) (ha : 0 < a ∧ a < Real.pi / 2) (hb : 0 < b ∧ b < Real.pi / 2) :
  5 / Real.cos a ^ 2 + 5 / (Real.sin a ^ 2 * Real.sin b ^ 2 * Real.cos b ^ 2) ≥ 27 * Real.cos a + 36 * Real.sin a :=
sorry

end trigonometric_inequality_l2078_207836


namespace simplify_sqrt_neg2_squared_l2078_207859

theorem simplify_sqrt_neg2_squared : 
  Real.sqrt ((-2 : ℝ)^2) = 2 := 
by
  sorry

end simplify_sqrt_neg2_squared_l2078_207859


namespace solve_fractional_equation_l2078_207813

theorem solve_fractional_equation (x : ℝ) (h : (3 / (x + 1) - 2 / (x - 1)) = 0) : x = 5 :=
sorry

end solve_fractional_equation_l2078_207813


namespace sum_of_remainders_l2078_207800

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : (n % 4) + (n % 5) = 4 := 
by {
  -- proof omitted
  sorry
}

end sum_of_remainders_l2078_207800


namespace negation_of_p_l2078_207835

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, x ≥ 2

-- State the proof problem as a Lean theorem
theorem negation_of_p : (∀ x : ℝ, x ≥ 2) → ∃ x₀ : ℝ, x₀ < 2 :=
by
  intro h
  -- Define how the proof would generally proceed
  -- as the negation of a universal statement is an existential statement.
  sorry

end negation_of_p_l2078_207835


namespace quadratic_function_has_specific_k_l2078_207864

theorem quadratic_function_has_specific_k (k : ℤ) :
  (∀ x : ℝ, ∃ y : ℝ, y = (k-1)*x^(k^2-k+2) + k*x - 1) ↔ k = 0 :=
by
  sorry

end quadratic_function_has_specific_k_l2078_207864


namespace coupons_used_l2078_207811

theorem coupons_used
  (initial_books : ℝ)
  (sold_books : ℝ)
  (coupons_per_book : ℝ)
  (remaining_books := initial_books - sold_books)
  (total_coupons := remaining_books * coupons_per_book) :
  initial_books = 40.0 →
  sold_books = 20.0 →
  coupons_per_book = 4.0 →
  total_coupons = 80.0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end coupons_used_l2078_207811


namespace solution_set_inequality_l2078_207872

theorem solution_set_inequality (x : ℝ) :
  (3 * x + 2 ≥ 1 ∧ (5 - x) / 2 < 0) ↔ (-1 / 3 ≤ x ∧ x < 5) :=
by
  sorry

end solution_set_inequality_l2078_207872


namespace class3_total_score_l2078_207828

theorem class3_total_score 
  (total_points : ℕ)
  (class1_score class2_score class3_score : ℕ)
  (class1_places class2_places class3_places : ℕ)
  (total_places : ℕ)
  (points_1st  points_2nd  points_3rd : ℕ)
  (h1 : total_points = 27)
  (h2 : class1_score = class2_score)
  (h3 : 2 * class1_places = class2_places)
  (h4 : class1_places + class2_places + class3_places = total_places)
  (h5 : 3 * points_1st + 3 * points_2nd + 3 * points_3rd = total_points)
  (h6 : total_places = 9)
  (h7 : points_1st = 5)
  (h8 : points_2nd = 3)
  (h9 : points_3rd = 1) :
  class3_score = 7 :=
sorry

end class3_total_score_l2078_207828


namespace tyler_age_l2078_207870

theorem tyler_age (T B : ℕ) (h1 : T = B - 3) (h2 : T + B = 11) : T = 4 :=
  sorry

end tyler_age_l2078_207870


namespace remaining_budget_for_public_spaces_l2078_207832

noncomputable def total_budget : ℝ := 32
noncomputable def policing_budget : ℝ := total_budget / 2
noncomputable def education_budget : ℝ := 12
noncomputable def remaining_budget : ℝ := total_budget - (policing_budget + education_budget)

theorem remaining_budget_for_public_spaces : remaining_budget = 4 :=
by
  -- Proof is skipped
  sorry

end remaining_budget_for_public_spaces_l2078_207832


namespace polygon_D_has_largest_area_l2078_207806

noncomputable def area_A := 4 * 1 + 2 * (1 / 2) -- 5
noncomputable def area_B := 2 * 1 + 2 * (1 / 2) + Real.pi / 4 -- ≈ 3.785
noncomputable def area_C := 3 * 1 + 3 * (1 / 2) -- 4.5
noncomputable def area_D := 3 * 1 + 1 * (1 / 2) + 2 * (Real.pi / 4) -- ≈ 5.07
noncomputable def area_E := 1 * 1 + 3 * (1 / 2) + 3 * (Real.pi / 4) -- ≈ 4.855

theorem polygon_D_has_largest_area :
  area_D > area_A ∧
  area_D > area_B ∧
  area_D > area_C ∧
  area_D > area_E :=
by
  sorry

end polygon_D_has_largest_area_l2078_207806


namespace expand_product_l2078_207888

theorem expand_product (x : ℝ) : (x + 2) * (x + 5) = x^2 + 7 * x + 10 := 
by 
  sorry

end expand_product_l2078_207888


namespace relationship_between_a_b_c_l2078_207831

-- Define the given parabola function
def parabola (x : ℝ) (k : ℝ) : ℝ := -(x - 2)^2 + k

-- Define the points A, B, C with their respective coordinates and expressions on the parabola
variables {a b c k : ℝ}

-- Conditions: Points lie on the parabola
theorem relationship_between_a_b_c (hA : a = parabola (-2) k)
                                  (hB : b = parabola (-1) k)
                                  (hC : c = parabola 3 k) :
  a < b ∧ b < c :=
by
  sorry

end relationship_between_a_b_c_l2078_207831


namespace second_reduction_percentage_l2078_207829

variable (P : ℝ) -- Original price
variable (x : ℝ) -- Second reduction percentage

-- Condition 1: After a 25% reduction
def first_reduction (P : ℝ) : ℝ := 0.75 * P

-- Condition 3: Combined reduction equivalent to 47.5%
def combined_reduction (P : ℝ) : ℝ := 0.525 * P

-- Question: Given the conditions, prove that the second reduction is 0.3
theorem second_reduction_percentage (P : ℝ) (x : ℝ) :
  (1 - x) * first_reduction P = combined_reduction P → x = 0.3 :=
by
  intro h
  sorry

end second_reduction_percentage_l2078_207829


namespace money_left_over_l2078_207853

def initial_amount : ℕ := 120
def sandwich_fraction : ℚ := 1 / 5
def museum_ticket_fraction : ℚ := 1 / 6
def book_fraction : ℚ := 1 / 2

theorem money_left_over :
  let sandwich_cost := initial_amount * sandwich_fraction
  let museum_ticket_cost := initial_amount * museum_ticket_fraction
  let book_cost := initial_amount * book_fraction
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  initial_amount - total_spent = 16 :=
by
  sorry

end money_left_over_l2078_207853


namespace balance_two_diamonds_three_bullets_l2078_207841

-- Define the variables
variables (a b c : ℝ)

-- Define the conditions as hypotheses
def condition1 : Prop := 3 * a + b = 9 * c
def condition2 : Prop := a = b + c

-- Goal is to prove two diamonds (2 * b) balance three bullets (3 * c)
theorem balance_two_diamonds_three_bullets (h1 : condition1 a b c) (h2 : condition2 a b c) : 
  2 * b = 3 * c := 
by 
  sorry

end balance_two_diamonds_three_bullets_l2078_207841


namespace scientific_notation_flu_virus_diameter_l2078_207889

theorem scientific_notation_flu_virus_diameter :
  0.000000823 = 8.23 * 10^(-7) :=
sorry

end scientific_notation_flu_virus_diameter_l2078_207889


namespace correct_bio_experiment_technique_l2078_207825

-- Let's define our conditions as hypotheses.
def yeast_count_method := "sampling_inspection"
def small_animal_group_method := "sampler_sampling"
def mitosis_rinsing_purpose := "wash_away_dissociation_solution"
def fat_identification_solution := "alcohol"

-- The question translated into a statement is to show that the method for counting yeast is the sampling inspection method.
theorem correct_bio_experiment_technique :
  yeast_count_method = "sampling_inspection" ∧
  small_animal_group_method ≠ "mark-recapture" ∧
  mitosis_rinsing_purpose ≠ "wash_away_dye" ∧
  fat_identification_solution ≠ "50%_hydrochloric_acid" :=
sorry

end correct_bio_experiment_technique_l2078_207825


namespace exists_same_color_points_one_meter_apart_l2078_207874

-- Declare the colors as an enumeration
inductive Color
| red : Color
| black : Color

-- Define the function that assigns a color to each point in the plane
def color (point : ℝ × ℝ) : Color := sorry

-- The theorem to be proven
theorem exists_same_color_points_one_meter_apart :
  ∃ x y : ℝ × ℝ, x ≠ y ∧ dist x y = 1 ∧ color x = color y :=
sorry

end exists_same_color_points_one_meter_apart_l2078_207874


namespace colby_mango_sales_l2078_207810

theorem colby_mango_sales
  (total_kg : ℕ)
  (mangoes_per_kg : ℕ)
  (remaining_mangoes : ℕ)
  (half_sold_to_market : ℕ) :
  total_kg = 60 →
  mangoes_per_kg = 8 →
  remaining_mangoes = 160 →
  half_sold_to_market = 20 := by
    sorry

end colby_mango_sales_l2078_207810


namespace new_average_income_l2078_207860

theorem new_average_income (old_avg_income : ℝ) (num_members : ℕ) (deceased_income : ℝ) 
  (old_avg_income_eq : old_avg_income = 735) (num_members_eq : num_members = 4) 
  (deceased_income_eq : deceased_income = 990) : 
  ((old_avg_income * num_members) - deceased_income) / (num_members - 1) = 650 := 
by sorry

end new_average_income_l2078_207860


namespace boats_equation_correct_l2078_207892

theorem boats_equation_correct (x : ℕ) (h1 : x ≤ 8) (h2 : 4 * x + 6 * (8 - x) = 38) : 
    4 * x + 6 * (8 - x) = 38 :=
by
  sorry

end boats_equation_correct_l2078_207892


namespace wade_total_spent_l2078_207865

def sandwich_cost : ℕ := 6
def drink_cost : ℕ := 4
def num_sandwiches : ℕ := 3
def num_drinks : ℕ := 2

def total_cost : ℕ :=
  (num_sandwiches * sandwich_cost) + (num_drinks * drink_cost)

theorem wade_total_spent : total_cost = 26 := by
  sorry

end wade_total_spent_l2078_207865


namespace max_value_f_on_interval_l2078_207858

open Real

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_f_on_interval :
  ∃ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), ∀ y ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f y ≤ f x ∧ f x = 23 := by
  sorry

end max_value_f_on_interval_l2078_207858
