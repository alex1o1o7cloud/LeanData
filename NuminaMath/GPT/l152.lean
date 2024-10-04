import Mathlib

namespace neg_p_l152_152429

variable {x : ℝ}

def p := ∀ x > 0, Real.sin x ≤ 1

theorem neg_p : ¬ p ↔ ∃ x > 0, Real.sin x > 1 :=
by
  sorry

end neg_p_l152_152429


namespace range_of_m_l152_152106

variables (m : ℝ)

def p : Prop := ∀ x : ℝ, 0 < x → (1/2 : ℝ)^x + m - 1 < 0
def q : Prop := ∃ x : ℝ, 0 < x ∧ m * x^2 + 4 * x - 1 = 0

theorem range_of_m (h : p m ∧ q m) : -4 ≤ m ∧ m ≤ 0 :=
sorry

end range_of_m_l152_152106


namespace trig_expression_value_l152_152437

theorem trig_expression_value (θ : ℝ) (h : Real.tan θ = 2) : 
  (2 * Real.cos θ) / (Real.sin (Real.pi / 2 + θ) + Real.sin (Real.pi + θ)) = -2 := 
by 
  sorry

end trig_expression_value_l152_152437


namespace supplementary_angle_proof_l152_152102

noncomputable def complementary_angle (α : ℝ) : ℝ := 125 + 12 / 60

noncomputable def calculate_angle (c : ℝ) := 180 - c

noncomputable def supplementary_angle (α : ℝ) := 90 - α

theorem supplementary_angle_proof :
    let α := calculate_angle (complementary_angle α)
    supplementary_angle α = 35 + 12 / 60 := 
by
  sorry

end supplementary_angle_proof_l152_152102


namespace marked_points_rational_l152_152647

theorem marked_points_rational {n : ℕ} (x : Fin (n + 2) → ℝ) :
  x 0 = 0 ∧ x (Fin.last (n + 1)) = 1 ∧
  (∀ i : Fin (n + 2), ∃ a b : Fin (n + 2),
    (x i = (x a + x b) / 2 ∧ x a < x i ∧ x i < x b) ∨
    (x i = (x a + 0) / 2 ∧ x a = x 0 ∨ (x i = (x b + 1) / 2 ∧ x b = 1))) →
  (∀ i : Fin (n + 2), ∃ q : ℚ, x i = q) :=
begin
  sorry
end

end marked_points_rational_l152_152647


namespace sum_of_common_ratios_l152_152730

variables {k p r : ℝ}
variables {a_2 a_3 b_2 b_3 : ℝ}

def geometric_seq1 (k p : ℝ) := a_2 = k * p ∧ a_3 = k * p^2
def geometric_seq2 (k r : ℝ) := b_2 = k * r ∧ b_3 = k * r^2

theorem sum_of_common_ratios (h1 : geometric_seq1 k p) (h2 : geometric_seq2 k r)
  (h3 : p ≠ r) (h4 : a_3 - b_3 = 4 * (a_2 - b_2)) : p + r = 4 :=
by sorry

end sum_of_common_ratios_l152_152730


namespace range_of_a_l152_152285

theorem range_of_a
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x + 2 * y + 4 = 4 * x * y)
  (h2 : ∀ a : ℝ, (x + 2 * y) * a ^ 2 + 2 * a + 2 * x * y - 34 ≥ 0) : 
  ∀ a : ℝ, a ≤ -3 ∨ a ≥ 5 / 2 :=
by
  sorry

end range_of_a_l152_152285


namespace no_three_real_numbers_satisfy_inequalities_l152_152744

theorem no_three_real_numbers_satisfy_inequalities (a b c : ℝ) :
  ¬ (|a| < |b - c| ∧ |b| < |c - a| ∧ |c| < |a - b| ) :=
by
  sorry

end no_three_real_numbers_satisfy_inequalities_l152_152744


namespace proof_problem_l152_152592

theorem proof_problem (k m : ℕ) (hk_pos : 0 < k) (hm_pos : 0 < m) (hkm : k > m)
  (hdiv : (k * m * (k ^ 2 - m ^ 2)) ∣ (k ^ 3 - m ^ 3)) :
  (k - m) ^ 3 > 3 * k * m :=
sorry

end proof_problem_l152_152592


namespace S_5_is_121_l152_152966

-- Definitions of the sequence and its terms
def S : ℕ → ℕ := sorry  -- Define S_n
def a : ℕ → ℕ := sorry  -- Define a_n

-- Conditions
axiom S_2 : S 2 = 4
axiom recurrence_relation : ∀ n : ℕ, S (n + 1) = 1 + 2 * S n

-- Proof that S_5 = 121 given the conditions
theorem S_5_is_121 : S 5 = 121 := by
  sorry

end S_5_is_121_l152_152966


namespace range_of_m_l152_152849

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (h_deriv : ∀ x, f' x < x)
variable (h_ineq : ∀ m, f (4 - m) - f m ≥ 8 - 4 * m)

theorem range_of_m (m : ℝ) : m ≥ 2 :=
sorry

end range_of_m_l152_152849


namespace imaginary_unit_multiplication_l152_152281

-- Statement of the problem   
theorem imaginary_unit_multiplication (i : ℂ) (hi : i ^ 2 = -1) : i * (1 + i) = -1 + i :=
by sorry

end imaginary_unit_multiplication_l152_152281


namespace perfect_square_factors_count_450_l152_152127

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l152_152127


namespace fraction_of_power_l152_152005

noncomputable def m : ℕ := 32^500

theorem fraction_of_power (h : m = 2^2500) : m / 8 = 2^2497 :=
by
  have hm : m = 2^2500 := h
  sorry

end fraction_of_power_l152_152005


namespace tenth_term_arith_seq_l152_152078

variable (a1 d : Int) -- Initial term and common difference
variable (n : Nat) -- nth term

-- Definition of the nth term in an arithmetic sequence
def arithmeticSeq (a1 d : Int) (n : Nat) : Int :=
  a1 + (n - 1) * d

-- Specific values for the problem
def a_10 : Int :=
  arithmeticSeq 10 (-3) 10

-- The theorem we want to prove
theorem tenth_term_arith_seq : a_10 = -17 := by
  sorry

end tenth_term_arith_seq_l152_152078


namespace factor_theorem_l152_152807

noncomputable def polynomial_to_factor : Prop :=
  ∀ x : ℝ, x^4 - 4 * x^2 + 4 = (x^2 - 2)^2

theorem factor_theorem : polynomial_to_factor :=
by
  sorry

end factor_theorem_l152_152807


namespace ratio_of_sides_l152_152197

theorem ratio_of_sides 
  (a b c d : ℝ) 
  (h1 : (a * b) / (c * d) = 0.16) 
  (h2 : b / d = 2 / 5) : 
  a / c = 0.4 := 
by 
  sorry

end ratio_of_sides_l152_152197


namespace points_total_l152_152985

/--
In a game, Samanta has 8 more points than Mark,
and Mark has 50% more points than Eric. Eric has 6 points.
How many points do Samanta, Mark, and Eric have in total?
-/
theorem points_total (Samanta Mark Eric : ℕ)
  (h1 : Samanta = Mark + 8)
  (h2 : Mark = Eric + Eric / 2)
  (h3 : Eric = 6) :
  Samanta + Mark + Eric = 32 := by
  sorry

end points_total_l152_152985


namespace find_radius_of_small_semicircle_l152_152172

noncomputable def radius_of_small_semicircle (R : ℝ) (r : ℝ) :=
  ∀ (x : ℝ),
    (12: ℝ = R) ∧ (6: ℝ = r) →
    (∃ (x: ℝ), R - x + r = sqrt((r + x)^2 - r^2)) →
    x = 4

theorem find_radius_of_small_semicircle : radius_of_small_semicircle 12 6 :=
begin
  unfold radius_of_small_semicircle,
  intro x,
  assume h1 h2,
  cases h2,
  sorry,
end

end find_radius_of_small_semicircle_l152_152172


namespace charlotte_overall_score_l152_152260

theorem charlotte_overall_score :
  (0.60 * 15 + 0.75 * 20 + 0.85 * 25).round / 60 = 0.75 :=
by
  sorry

end charlotte_overall_score_l152_152260


namespace equation_solution_l152_152016

def solve_equation (x : ℝ) : Prop :=
  ((3 * x + 6) / (x^2 + 5 * x - 6) = (4 - x) / (x - 2)) ↔ 
  x = -3 ∨ x = (1 + Real.sqrt 17) / 2 ∨ x = (1 - Real.sqrt 17) / 2

theorem equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) (h3 : x ≠ 2) : solve_equation x :=
by
  sorry

end equation_solution_l152_152016


namespace find_horizontal_length_l152_152030

variable (v h : ℝ)

-- Conditions
def is_horizontal_length_of_rectangle_perimeter_54_and_vertical_plus_3 (v h : ℝ) : Prop :=
  2 * h + 2 * v = 54 ∧ h = v + 3

-- The proof we aim to show
theorem find_horizontal_length (v h : ℝ) :
  is_horizontal_length_of_rectangle_perimeter_54_and_vertical_plus_3 v h → h = 15 :=
by
  sorry

end find_horizontal_length_l152_152030


namespace q_investment_l152_152774

theorem q_investment (p_investment : ℕ) (ratio_pq : ℕ × ℕ) (profit_ratio : ℕ × ℕ) (hp : p_investment = 12000) (hpr : ratio_pq = (3, 5)) : 
  (∃ q_investment, q_investment = 20000) :=
  sorry

end q_investment_l152_152774


namespace combination_10_5_l152_152577

theorem combination_10_5 :
  (Nat.choose 10 5) = 2520 :=
by
  sorry

end combination_10_5_l152_152577


namespace average_salary_increase_l152_152911

theorem average_salary_increase 
  (average_salary : ℕ) (manager_salary : ℕ)
  (n : ℕ) (initial_count : ℕ) (new_count : ℕ) (initial_average : ℕ)
  (total_salary : ℕ) (new_total_salary : ℕ) (new_average : ℕ)
  (salary_increase : ℕ) :
  initial_average = 1500 →
  manager_salary = 3600 →
  initial_count = 20 →
  new_count = initial_count + 1 →
  total_salary = initial_count * initial_average →
  new_total_salary = total_salary + manager_salary →
  new_average = new_total_salary / new_count →
  salary_increase = new_average - initial_average →
  salary_increase = 100 := by
  sorry

end average_salary_increase_l152_152911


namespace odd_function_property_l152_152883

theorem odd_function_property {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = - f x) (h2 : ∀ x, f (1 + x) = f (-x)) (h3 : f (-1 / 3) = 1 / 3) : f (5 / 3) = 1 / 3 := 
sorry

end odd_function_property_l152_152883


namespace ellipse_equation_constant_dot_product_fixed_point_exists_l152_152098

-- Definitions based on conditions
def a := 2
def b := sqrt 2
def ellipse_eq (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def C := (-2, 0)
def D := (2, 0)
def M (y₀ : ℝ) := (2, y₀)
def P (x₁ y₁ : ℝ) := (x₁, y₁)
def OP (x₁ y₁ : ℝ) := (x₁, y₁)
def OM (y₀ : ℝ) := (2, y₀)
def Q := (0, 0)

-- Proof statements
theorem ellipse_equation : ellipse_eq 4 2 := sorry

theorem constant_dot_product (y₀ : ℝ) (x₁ y₁ : ℝ) 
    (hP : ellipse_eq x₁ y₁) (hOP : OP x₁ y₁) (hOM : OM y₀) :
  OP x₁ y₁ • OM y₀ = 4 := sorry

theorem fixed_point_exists (y₀ : ℝ) (x₁ y₁ : ℝ)
    (hP : ellipse_eq x₁ y₁) (hM : M y₀) : 
  ∃ Qx Qy, Qx = 0 ∧ Qy = 0 ∧ circle_with_diameter (M y₀) (P x₁ y₁) Q :=
sorry

end ellipse_equation_constant_dot_product_fixed_point_exists_l152_152098


namespace new_volume_correct_l152_152789

-- Define the conditions
def original_volume : ℝ := 60
def length_factor : ℝ := 3
def width_factor : ℝ := 2
def height_factor : ℝ := 1.20

-- Define the new volume as a result of the above factors
def new_volume : ℝ := original_volume * length_factor * width_factor * height_factor

-- Proof statement for the new volume being 432 cubic feet
theorem new_volume_correct : new_volume = 432 :=
by 
    -- Directly state the desired equality
    sorry

end new_volume_correct_l152_152789


namespace polar_coordinates_of_point_l152_152663

theorem polar_coordinates_of_point {x y : ℝ} (hx : x = -3) (hy : y = 1) :
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.pi - Real.arctan (y / abs x)
  r = Real.sqrt 10 ∧ θ = Real.pi - Real.arctan (1 / 3) := 
by
  rw [hx, hy]
  sorry

end polar_coordinates_of_point_l152_152663


namespace total_chairs_needed_l152_152996

theorem total_chairs_needed (tables_4_seats tables_6_seats seats_per_table_4 seats_per_table_6 : ℕ) : 
  tables_4_seats = 6 → 
  seats_per_table_4 = 4 → 
  tables_6_seats = 12 → 
  seats_per_table_6 = 6 → 
  (tables_4_seats * seats_per_table_4 + tables_6_seats * seats_per_table_6) = 96 := 
by
  intros h1 h2 h3 h4
  -- sorry

end total_chairs_needed_l152_152996


namespace possible_measure_of_angle_AOC_l152_152174

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

end possible_measure_of_angle_AOC_l152_152174


namespace probability_vowel_probability_consonant_probability_ch_l152_152559

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

end probability_vowel_probability_consonant_probability_ch_l152_152559


namespace distinct_cyclic_quadrilaterals_perimeter_36_l152_152291

noncomputable def count_distinct_cyclic_quadrilaterals : Nat :=
  1026

theorem distinct_cyclic_quadrilaterals_perimeter_36 :
  (∃ (a b c d : ℕ), a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a + b + c + d = 36 ∧ a < b + c + d) → count_distinct_cyclic_quadrilaterals = 1026 :=
by
  rintro ⟨a, b, c, d, hab, hbc, hcd, hsum, hlut⟩
  sorry

end distinct_cyclic_quadrilaterals_perimeter_36_l152_152291


namespace birds_on_fence_total_l152_152386

variable (initial_birds : ℕ) (additional_birds : ℕ)

theorem birds_on_fence_total {initial_birds additional_birds : ℕ} (h1 : initial_birds = 4) (h2 : additional_birds = 6) :
    initial_birds + additional_birds = 10 :=
  by
  sorry

end birds_on_fence_total_l152_152386


namespace total_pages_read_is_785_l152_152465

-- Definitions based on the conditions in the problem
def pages_read_first_five_days : ℕ := 5 * 52
def pages_read_next_five_days : ℕ := 5 * 63
def pages_read_last_three_days : ℕ := 3 * 70

-- The main statement to prove
theorem total_pages_read_is_785 :
  pages_read_first_five_days + pages_read_next_five_days + pages_read_last_three_days = 785 :=
by
  sorry

end total_pages_read_is_785_l152_152465


namespace tree_age_difference_l152_152074

theorem tree_age_difference
  (groups_rings : ℕ)
  (rings_per_group : ℕ)
  (first_tree_groups : ℕ)
  (second_tree_groups : ℕ)
  (rings_per_year : ℕ)
  (h_rg : rings_per_group = 6)
  (h_ftg : first_tree_groups = 70)
  (h_stg : second_tree_groups = 40)
  (h_rpy : rings_per_year = 1) :
  ((first_tree_groups * rings_per_group) - (second_tree_groups * rings_per_group)) = 180 := 
by
  sorry

end tree_age_difference_l152_152074


namespace expression_not_equal_l152_152511

theorem expression_not_equal :
  let e1 := 250 * 12
  let e2 := 25 * 4 + 30
  let e3 := 25 * 40 * 3
  let product := 25 * 120
  e2 ≠ product :=
by
  let e1 := 250 * 12
  let e2 := 25 * 4 + 30
  let e3 := 25 * 40 * 3
  let product := 25 * 120
  sorry

end expression_not_equal_l152_152511


namespace find_integer_values_l152_152267

theorem find_integer_values (a : ℤ) (h : ∃ (n : ℤ), (a + 9) = n * (a + 6)) :
  a = -5 ∨ a = -7 ∨ a = -3 ∨ a = -9 :=
by
  sorry

end find_integer_values_l152_152267


namespace drivers_distance_difference_l152_152764

noncomputable def total_distance_driven (initial_distance : ℕ) (speed_A : ℕ) (speed_B : ℕ) (start_delay : ℕ) : ℕ := sorry

theorem drivers_distance_difference
  (initial_distance : ℕ)
  (speed_A : ℕ)
  (speed_B : ℕ)
  (start_delay : ℕ)
  (correct_difference : ℕ)
  (h_initial : initial_distance = 1025)
  (h_speed_A : speed_A = 90)
  (h_speed_B : speed_B = 80)
  (h_start_delay : start_delay = 1)
  (h_correct_difference : correct_difference = 145) :
  total_distance_driven initial_distance speed_A speed_B start_delay = correct_difference :=
sorry

end drivers_distance_difference_l152_152764


namespace abs_neg_five_l152_152612

theorem abs_neg_five : abs (-5) = 5 := 
by 
  sorry

end abs_neg_five_l152_152612


namespace sum_last_two_digits_is_correct_l152_152228

def fibs : List Nat := [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

def factorial_last_two_digits (n : Nat) : Nat :=
  (Nat.factorial n) % 100

def modified_fib_factorial_series : List Nat :=
  fibs.map (λ k => (factorial_last_two_digits k + 2) % 100)

def sum_last_two_digits : Nat :=
  (modified_fib_factorial_series.sum) % 100

theorem sum_last_two_digits_is_correct :
  sum_last_two_digits = 14 :=
sorry

end sum_last_two_digits_is_correct_l152_152228


namespace sum_of_nus_is_45_l152_152045

noncomputable def sum_of_valid_nu : ℕ :=
  ∑ ν in {ν | ν > 0 ∧ Nat.lcm ν 24 = 72}.toFinset, ν

theorem sum_of_nus_is_45 : sum_of_valid_nu = 45 := by
  sorry

end sum_of_nus_is_45_l152_152045


namespace remainder_of_6_power_700_mod_72_l152_152227

theorem remainder_of_6_power_700_mod_72 : (6^700) % 72 = 0 :=
by
  sorry

end remainder_of_6_power_700_mod_72_l152_152227


namespace sum_of_side_lengths_in_cm_l152_152695

-- Definitions for the given conditions
def side_length_meters : ℝ := 2.3
def meters_to_centimeters : ℝ := 100
def num_sides : ℕ := 8

-- The statement to prove
theorem sum_of_side_lengths_in_cm :
  let side_length_cm := side_length_meters * meters_to_centimeters in
  let total_length_cm := side_length_cm * (num_sides : ℝ) in
  total_length_cm = 1840 :=
by
  sorry

end sum_of_side_lengths_in_cm_l152_152695


namespace rohan_house_rent_percentage_l152_152199

noncomputable def house_rent_percentage (food_percentage entertainment_percentage conveyance_percentage salary savings: ℝ) : ℝ :=
  100 - (food_percentage + entertainment_percentage + conveyance_percentage + (savings / salary * 100))

-- Conditions
def food_percentage : ℝ := 40
def entertainment_percentage : ℝ := 10
def conveyance_percentage : ℝ := 10
def salary : ℝ := 10000
def savings : ℝ := 2000

-- Theorem
theorem rohan_house_rent_percentage :
  house_rent_percentage food_percentage entertainment_percentage conveyance_percentage salary savings = 20 := 
sorry

end rohan_house_rent_percentage_l152_152199


namespace numPerfectSquareFactorsOf450_l152_152145

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l152_152145


namespace arithmetic_expression_evaluation_l152_152950

theorem arithmetic_expression_evaluation : 
  (5 * 7 - (3 * 2 + 5 * 4) / 2) = 22 := 
by
  sorry

end arithmetic_expression_evaluation_l152_152950


namespace oranges_sold_l152_152494

def bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def rotten_oranges : ℕ := 50
def oranges_for_juice : ℕ := 30

theorem oranges_sold : (bags * oranges_per_bag) - rotten_oranges - oranges_for_juice = 220 := by
  sorry

end oranges_sold_l152_152494


namespace inequality_no_solution_l152_152690

-- Define the quadratic inequality.
def quadratic_ineq (m x : ℝ) : Prop :=
  (m + 1) * x^2 - m * x + (m - 1) > 0

-- Define the condition for m.
def range_of_m (m : ℝ) : Prop :=
  m ≤ - (2 * Real.sqrt 3) / 3

-- Theorem stating that if the inequality has no solution, m gets restricted.
theorem inequality_no_solution (m : ℝ) :
  (∀ x : ℝ, ¬ quadratic_ineq m x) ↔ range_of_m m :=
by sorry

end inequality_no_solution_l152_152690


namespace zoo_structure_l152_152531

theorem zoo_structure (P : ℕ) (h1 : ∃ (snakes monkeys elephants zebras : ℕ),
  snakes = 3 * P ∧
  monkeys = 6 * P ∧
  elephants = (P + snakes) / 2 ∧
  zebras = elephants - 3 ∧
  monkeys - zebras = 35) : P = 8 :=
sorry

end zoo_structure_l152_152531


namespace mismatching_socks_count_l152_152750

-- Define the conditions given in the problem
def total_socks : ℕ := 65
def pairs_matching_ankle_socks : ℕ := 13
def pairs_matching_crew_socks : ℕ := 10

-- Define the calculated counts as per the conditions
def matching_ankle_socks : ℕ := pairs_matching_ankle_socks * 2
def matching_crew_socks : ℕ := pairs_matching_crew_socks * 2
def total_matching_socks : ℕ := matching_ankle_socks + matching_crew_socks

-- The statement to prove
theorem mismatching_socks_count : total_socks - total_matching_socks = 19 := by
  sorry

end mismatching_socks_count_l152_152750


namespace range_of_a_l152_152838

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 + 2*x else -(x^2 + 2*(-x))

theorem range_of_a (a : ℝ) (h : f (3 - a^2) > f (2 * a)) : -3 < a ∧ a < 1 := sorry

end range_of_a_l152_152838


namespace a_leq_neg4_l152_152845

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0
def neg_p (a x : ℝ) : Prop := ¬(p a x)
def neg_q (x : ℝ) : Prop := ¬(q x)

theorem a_leq_neg4 (a : ℝ) (h_neg_p : ∀ x, neg_p a x → neg_q x) (h_a_neg : a < 0) :
  a ≤ -4 :=
sorry

end a_leq_neg4_l152_152845


namespace solution_set_l152_152373

theorem solution_set:
  (∃ x y : ℝ, x - y = 0 ∧ x^2 + y = 2) ↔ (∃ x y : ℝ, (x = 1 ∧ y = 1) ∨ (x = -2 ∧ y = -2)) :=
by
  sorry

end solution_set_l152_152373


namespace problem_expression_eval_l152_152023

theorem problem_expression_eval : (1 + 2 + 3) * (1 + 1/2 + 1/3) = 11 := by
  sorry

end problem_expression_eval_l152_152023


namespace acute_angle_LO_CA_l152_152569

noncomputable def triangle_CAT : ℝ := sorry
noncomputable def angle_C : ℝ := 36
noncomputable def angle_A : ℝ := 56
noncomputable def CA : ℝ := 12
noncomputable def CX : ℝ := 2
noncomputable def ZC : ℝ := 2
noncomputable def midpoint (a b : ℝ) : ℝ := (a + b) / 2
noncomputable def L := midpoint (CA : ℝ) 0
noncomputable def O := midpoint (CX : ℝ) (ZC : ℝ)
noncomputable def angle_LO_CA : ℝ := 88 -- The acute angle formed by lines LO and CA

theorem acute_angle_LO_CA 
  (h_triangle : triangle_CAT = ℝ)
  (h_angle_C : angle_C = 36)
  (h_angle_A : angle_A = 56)
  (h_CA : CA = 12)
  (h_CX : CX = 2)
  (h_ZC : ZC = 2)
  (h_midpoint_L : L = 6)
  (h_midpoint_O : O = (CX + ZC) / 2) :
  angle_LO_CA = 88 := 
sorry

end acute_angle_LO_CA_l152_152569


namespace james_out_of_pocket_cost_l152_152998

theorem james_out_of_pocket_cost (total_cost : ℝ) (coverage : ℝ) (out_of_pocket_cost : ℝ)
  (h1 : total_cost = 300) (h2 : coverage = 0.8) :
  out_of_pocket_cost = 60 :=
by
  sorry

end james_out_of_pocket_cost_l152_152998


namespace rational_powers_diff_is_integer_l152_152428

noncomputable def is_integer (r : ℚ) : Prop :=
  ∃ (n : ℤ), r = n

theorem rational_powers_diff_is_integer
    (a b : ℚ)
    (h_distinct : a ≠ b)
    (h_positive : 0 < a ∧ 0 < b)
    (h_infinite : ∀ n : ℕ, ∃ m > n, (a ^ m - b ^ m) ∈ ℤ) :
    is_integer a ∧ is_integer b :=
sorry -- proof required

end rational_powers_diff_is_integer_l152_152428


namespace initial_gasoline_percentage_calculation_l152_152653

variable (initial_volume : ℝ)
variable (initial_ethanol_percentage : ℝ)
variable (additional_ethanol : ℝ)
variable (final_ethanol_percentage : ℝ)

theorem initial_gasoline_percentage_calculation
  (h1: initial_ethanol_percentage = 5)
  (h2: initial_volume = 45)
  (h3: additional_ethanol = 2.5)
  (h4: final_ethanol_percentage = 10) :
  100 - initial_ethanol_percentage = 95 :=
by
  sorry

end initial_gasoline_percentage_calculation_l152_152653


namespace father_age_is_30_l152_152514

theorem father_age_is_30 {M F : ℝ} 
  (h1 : M = (2 / 5) * F) 
  (h2 : M + 6 = (1 / 2) * (F + 6)) :
  F = 30 :=
sorry

end father_age_is_30_l152_152514


namespace inequality_iff_positive_l152_152889

variable (x y : ℝ)

theorem inequality_iff_positive :
  x + y > abs (x - y) ↔ x > 0 ∧ y > 0 :=
sorry

end inequality_iff_positive_l152_152889


namespace largest_root_of_quadratic_l152_152536

theorem largest_root_of_quadratic :
  ∀ (x : ℝ), x^2 - 9*x - 22 = 0 → x ≤ 11 :=
by
  sorry

end largest_root_of_quadratic_l152_152536


namespace earnings_difference_is_200_l152_152262

noncomputable def difference_in_earnings : ℕ :=
  let asking_price := 5200
  let maintenance_cost := asking_price / 10
  let first_offer_earnings := asking_price - maintenance_cost
  let headlight_cost := 80
  let tire_cost := 3 * headlight_cost
  let total_repair_cost := headlight_cost + tire_cost
  let second_offer_earnings := asking_price - total_repair_cost
  second_offer_earnings - first_offer_earnings

theorem earnings_difference_is_200 : difference_in_earnings = 200 := by
  sorry

end earnings_difference_is_200_l152_152262


namespace speed_of_boat_in_still_water_l152_152990

-- Define a structure for the conditions
structure BoatConditions where
  V_b : ℝ    -- Speed of the boat in still water
  V_s : ℝ    -- Speed of the stream
  goes_along_stream : V_b + V_s = 11
  goes_against_stream : V_b - V_s = 5

-- Define the target theorem
theorem speed_of_boat_in_still_water (c : BoatConditions) : c.V_b = 8 :=
by
  sorry

end speed_of_boat_in_still_water_l152_152990


namespace guinea_pigs_food_difference_l152_152200

theorem guinea_pigs_food_difference :
  ∀ (first second third total : ℕ),
  first = 2 →
  second = first * 2 →
  total = 13 →
  first + second + third = total →
  third - second = 3 :=
by 
  intros first second third total h1 h2 h3 h4
  sorry

end guinea_pigs_food_difference_l152_152200


namespace manager_salary_l152_152645

theorem manager_salary (avg_salary_50 : ℕ) (num_employees : ℕ) (increment_new_avg : ℕ)
  (new_avg_salary : ℕ) (total_old_salary : ℕ) (total_new_salary : ℕ) (M : ℕ) :
  avg_salary_50 = 2000 →
  num_employees = 50 →
  increment_new_avg = 250 →
  new_avg_salary = avg_salary_50 + increment_new_avg →
  total_old_salary = num_employees * avg_salary_50 →
  total_new_salary = (num_employees + 1) * new_avg_salary →
  M = total_new_salary - total_old_salary →
  M = 14750 :=
by {
  sorry
}

end manager_salary_l152_152645


namespace final_population_correct_l152_152390

noncomputable def initialPopulation : ℕ := 300000
noncomputable def immigration : ℕ := 50000
noncomputable def emigration : ℕ := 30000

noncomputable def populationAfterImmigration : ℕ := initialPopulation + immigration
noncomputable def populationAfterEmigration : ℕ := populationAfterImmigration - emigration

noncomputable def pregnancies : ℕ := populationAfterEmigration / 8
noncomputable def twinPregnancies : ℕ := pregnancies / 4
noncomputable def singlePregnancies : ℕ := pregnancies - twinPregnancies

noncomputable def totalBirths : ℕ := twinPregnancies * 2 + singlePregnancies
noncomputable def finalPopulation : ℕ := populationAfterEmigration + totalBirths

theorem final_population_correct : finalPopulation = 370000 :=
by
  sorry

end final_population_correct_l152_152390


namespace a_and_b_together_work_days_l152_152643

-- Definitions for the conditions:
def a_work_rate : ℚ := 1 / 9
def b_work_rate : ℚ := 1 / 18

-- The theorem statement:
theorem a_and_b_together_work_days : (a_work_rate + b_work_rate)⁻¹ = 6 := by
  sorry

end a_and_b_together_work_days_l152_152643


namespace max_radius_of_additional_jar_l152_152247

open Real

noncomputable def max_jar_radius (pot_radius jar1_radius jar2_radius : ℝ) : ℝ :=
(pot_radius^2 - 5 * jar1_radius^2 - (jar2_radius - jar1_radius)^2) / (2 * (pot_radius - jar2_radius - jar1_radius))

theorem max_radius_of_additional_jar :
  let pot_diameter := 36.0
  let pot_radius := pot_diameter / 2
  let jar1_radius := 6.0
  let jar2_radius := 12.0
  let r := 36.0 / 7.0
  max_jar_radius pot_radius jar1_radius jar2_radius = r := by
  sorry

end max_radius_of_additional_jar_l152_152247


namespace events_per_coach_l152_152802

theorem events_per_coach {students events_per_student coaches events total_participations total_events : ℕ} 
  (h1 : students = 480) 
  (h2 : events_per_student = 4) 
  (h3 : (students * events_per_student) = total_participations) 
  (h4 : ¬ students * events_per_student ≠ total_participations)
  (h5 : total_participations = 1920) 
  (h6 : (total_participations / 20) = total_events) 
  (h7 : ¬ total_participations / 20 ≠ total_events)
  (h8 : total_events = 96)
  (h9 : coaches = 16) :
  (total_events / coaches) = 6 := sorry

end events_per_coach_l152_152802


namespace number_of_10_yuan_coins_is_1_l152_152761

theorem number_of_10_yuan_coins_is_1
  (n : ℕ) -- number of coins
  (v : ℕ) -- total value of coins
  (c1 c5 c10 c50 : ℕ) -- number of 1, 5, 10, and 50 yuan coins
  (h1 : n = 9) -- there are nine coins in total
  (h2 : v = 177) -- the total value of these coins is 177 yuan
  (h3 : c1 ≥ 1 ∧ c5 ≥ 1 ∧ c10 ≥ 1 ∧ c50 ≥ 1) -- at least one coin of each denomination
  (h4 : c1 + c5 + c10 + c50 = n) -- sum of all coins number is n
  (h5 : c1 * 1 + c5 * 5 + c10 * 10 + c50 * 50 = v) -- total value of all coins is v
  : c10 = 1 := 
sorry

end number_of_10_yuan_coins_is_1_l152_152761


namespace simplify_trig_expression_l152_152359

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem simplify_trig_expression :
  (tan (20 * Real.pi / 180) + tan (30 * Real.pi / 180) + tan (60 * Real.pi / 180) + tan (70 * Real.pi / 180)) / Real.sin (10 * Real.pi / 180) =
  1 / (2 * Real.sin (10 * Real.pi / 180) ^ 2 * Real.cos (20 * Real.pi / 180)) + 4 / (Real.sqrt 3 * Real.sin (10 * Real.pi / 180)) :=
by
  sorry

end simplify_trig_expression_l152_152359


namespace determine_omega_l152_152687

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

-- Conditions
variables (ω : ℝ) (ϕ : ℝ)
axiom omega_pos : ω > 0
axiom phi_bound : abs ϕ < Real.pi / 2
axiom symm_condition1 : ∀ x, f ω ϕ (Real.pi / 4 - x) = -f ω ϕ (Real.pi / 4 + x)
axiom symm_condition2 : ∀ x, f ω ϕ (-Real.pi / 2 - x) = f ω ϕ x
axiom monotonic_condition : ∀ x1 x2, 0 < x1 → x1 < x2 → x2 < Real.pi / 8 → f ω ϕ x1 < f ω ϕ x2

theorem determine_omega : ω = 1 ∨ ω = 5 :=
sorry

end determine_omega_l152_152687


namespace find_f_five_thrids_l152_152878

noncomputable def f : ℝ → ℝ := sorry

lemma odd_fun (x : ℝ) : f (-x) = -f(x) :=
sorry

lemma f_transformation (x : ℝ) : f (1 + x) = f (-x) :=
sorry

lemma f_neg_inv : f (-1 / 3) = 1 / 3 :=
sorry

theorem find_f_five_thrids : f (5 / 3) = 1 / 3 :=
by
  -- application of the conditions stated
  have h1 : f(1 + (5 / 3 - 1)) = f(-(5 / 3 - 1)),
    from f_transformation (5 / 3 - 1),
  have h2 : f(-(5 / 3 - 1)) = -f(5 / 3 - 1),
    from odd_fun (5 / 3 - 1),
  have h3 : 5 / 3 - 1 = 2 / 3,
    by norm_num,
  rw [h3] at h1,
  rw [h2] at h1,
  have h4 : f (-1 / 3) = 1 / 3,
    from f_neg_inv,
  rw [←h4] at h1,
  exact h1,
sorry

end find_f_five_thrids_l152_152878


namespace age_of_new_teacher_l152_152910

-- Definitions of conditions
def avg_age_20_teachers (sum_of_ages : ℕ) : Prop :=
  sum_of_ages = 49 * 20

def avg_age_21_teachers (sum_of_ages : ℕ) : Prop :=
  sum_of_ages = 48 * 21

-- The proof goal
theorem age_of_new_teacher (sum_age_20 : ℕ) (sum_age_21 : ℕ) (h1 : avg_age_20_teachers sum_age_20) (h2 : avg_age_21_teachers sum_age_21) : 
  sum_age_21 - sum_age_20 = 28 :=
sorry

end age_of_new_teacher_l152_152910


namespace selection_of_books_l152_152580

-- Define the problem context and the proof statement
theorem selection_of_books (n k : ℕ) (h_n : n = 10) (h_k : k = 5) : nat.choose n k = 252 := by
  -- Given: n = 10, k = 5
  -- Prove: (10 choose 5) = 252
  rw [h_n, h_k]
  norm_num
  sorry

end selection_of_books_l152_152580


namespace largest_minus_smallest_eq_13_l152_152947

theorem largest_minus_smallest_eq_13 :
  let a := (-1 : ℤ) ^ 3
  let b := (-1 : ℤ) ^ 2
  let c := -(2 : ℤ) ^ 2
  let d := (-3 : ℤ) ^ 2
  max (max a (max b c)) d - min (min a (min b c)) d = 13 := by
  sorry

end largest_minus_smallest_eq_13_l152_152947


namespace joel_age_when_dad_twice_l152_152344

theorem joel_age_when_dad_twice (x joel_age dad_age: ℕ) (h₁: joel_age = 12) (h₂: dad_age = 47) 
(h₃: dad_age + x = 2 * (joel_age + x)) : joel_age + x = 35 :=
by
  rw [h₁, h₂] at h₃ 
  sorry

end joel_age_when_dad_twice_l152_152344


namespace min_value_f_l152_152100

noncomputable def f (a b : ℝ) : ℝ := (1 / a^5 + a^5 - 2) * (1 / b^5 + b^5 - 2)

theorem min_value_f :
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → f a b ≥ (31^4 / 32^2) :=
by
  intros
  sorry

end min_value_f_l152_152100


namespace polar_to_cartesian_l152_152209

theorem polar_to_cartesian (ρ θ : ℝ) : (ρ * Real.cos θ = 0) → ρ = 0 ∨ θ = π/2 :=
by 
  sorry

end polar_to_cartesian_l152_152209


namespace average_speed_jeffrey_l152_152460
-- Import the necessary Lean library.

-- Initial conditions in the problem, restated as Lean definitions.
def distance_jog (d : ℝ) : Prop := d = 3
def speed_jog (s : ℝ) : Prop := s = 4
def distance_walk (d : ℝ) : Prop := d = 4
def speed_walk (s : ℝ) : Prop := s = 3

-- Target statement to prove using Lean.
theorem average_speed_jeffrey :
  ∀ (dj sj dw sw : ℝ), distance_jog dj → speed_jog sj → distance_walk dw → speed_walk sw →
    (dj + dw) / ((dj / sj) + (dw / sw)) = 3.36 := 
  by
    intros dj sj dw sw hj hs hw hw
    sorry

end average_speed_jeffrey_l152_152460


namespace find_ratio_of_constants_l152_152081

theorem find_ratio_of_constants (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h₁ : 8 * x - 6 * y = c) (h₂ : 12 * y - 18 * x = d) : c / d = -4 / 9 := 
sorry

end find_ratio_of_constants_l152_152081


namespace number_of_perfect_square_divisors_of_450_l152_152152

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l152_152152


namespace distance_l1_l2_l152_152434

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  |C2 - C1| / Real.sqrt (A^2 + B^2)

theorem distance_l1_l2 :
  distance_between_parallel_lines 3 4 (-3) 2 = 1 :=
by
  -- Add the conditions needed to assert the theorem
  let l1 := (3, 4, -3) -- definition of line l1
  let l2 := (3, 4, 2)  -- definition of line l2
  -- Calculate the distance using the given formula
  let d := distance_between_parallel_lines 3 4 (-3) 2
  -- Assert the result
  show d = 1
  sorry

end distance_l1_l2_l152_152434


namespace range_of_a_l152_152180

open Set

variable {x a : ℝ}

def p (x a : ℝ) := x^2 + 2 * a * x - 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) := x^2 + 2 * x - 8 < 0

theorem range_of_a (h : ∀ x, p x a → q x): 0 < a ∧ a ≤ 4 / 3 := 
  sorry

end range_of_a_l152_152180


namespace mary_max_earnings_l152_152892

def max_hours : ℕ := 40
def regular_rate : ℝ := 8
def first_hours : ℕ := 20
def overtime_rate : ℝ := regular_rate + 0.25 * regular_rate

def earnings : ℝ := 
  (first_hours * regular_rate) +
  ((max_hours - first_hours) * overtime_rate)

theorem mary_max_earnings : earnings = 360 := by
  sorry

end mary_max_earnings_l152_152892


namespace initial_slices_ham_l152_152254

def total_sandwiches : ℕ := 50
def slices_per_sandwich : ℕ := 3
def additional_slices_needed : ℕ := 119

-- Calculate the total number of slices needed to make 50 sandwiches.
def total_slices_needed : ℕ := total_sandwiches * slices_per_sandwich

-- Prove the initial number of slices of ham Anna has.
theorem initial_slices_ham : total_slices_needed - additional_slices_needed = 31 := by
  sorry

end initial_slices_ham_l152_152254


namespace convert_to_scientific_notation_l152_152340

theorem convert_to_scientific_notation :
  40.25 * 10^9 = 4.025 * 10^9 :=
by
  -- Sorry is used here to skip the proof
  sorry

end convert_to_scientific_notation_l152_152340


namespace shaded_area_triangle_l152_152865

theorem shaded_area_triangle (a b : ℝ) (h1 : a = 5) (h2 : b = 15) :
  let area_shaded : ℝ := (5^2) - (1/2 * ((15 / 4) * 5))
  area_shaded = 175 / 8 := 
by
  sorry

end shaded_area_triangle_l152_152865


namespace hyperbola_foci_coords_l152_152960

theorem hyperbola_foci_coords :
  ∀ x y, (x^2) / 8 - (y^2) / 17 = 1 → (x, y) = (5, 0) ∨ (x, y) = (-5, 0) :=
by
  sorry

end hyperbola_foci_coords_l152_152960


namespace initial_games_l152_152178

theorem initial_games (X : ℕ) (h1 : X + 31 - 105 = 6) : X = 80 :=
by
  sorry

end initial_games_l152_152178


namespace cubic_sum_identity_l152_152296

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l152_152296


namespace remainder_of_a55_l152_152003

def concatenate_integers (n : ℕ) : ℕ :=
  -- Function to concatenate integers from 1 to n into a single number.
  -- This is a placeholder, actual implementation may vary.
  sorry

theorem remainder_of_a55 (n : ℕ) (hn : n = 55) :
  concatenate_integers n % 55 = 0 := by
  -- Proof is omitted, provided as a guideline.
  sorry

end remainder_of_a55_l152_152003


namespace perfect_square_factors_450_l152_152132

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l152_152132


namespace num_even_divisors_of_210_l152_152213

open Set

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def divisors (n : ℕ) : Set ℕ := {d | d ∣ n}
def even_divisors (n : ℕ) : Set ℕ := {d | d ∣ n ∧ is_even d}

theorem num_even_divisors_of_210 : 
  let n := 210 in 
  let prime_factors_210 := [2, 3, 5, 7] in
  ∀ n = 2 * 3 * 5 * 7,
  (even_divisors n).card = 8 := 
by
  intro n h
  sorry

end num_even_divisors_of_210_l152_152213


namespace javier_total_time_spent_l152_152716

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2

theorem javier_total_time_spent : outlining_time + writing_time + practicing_time = 117 := by
  sorry

end javier_total_time_spent_l152_152716


namespace hexagram_shell_arrangements_l152_152590

/--
John places twelve different sea shells at the vertices of a regular six-pointed star (hexagram).
How many distinct ways can he place the shells, considering arrangements that differ by rotations or reflections as equivalent?
-/
theorem hexagram_shell_arrangements :
  (Nat.factorial 12) / 12 = 39916800 :=
by
  sorry

end hexagram_shell_arrangements_l152_152590


namespace number_of_perfect_square_factors_l152_152140

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l152_152140


namespace sum_of_positive_integers_lcm72_l152_152049

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_of_positive_integers_lcm72 : 
  ∑ ν in {ν | is_solution ν}.to_finset, ν = 180 :=
by 
  sorry

end sum_of_positive_integers_lcm72_l152_152049


namespace car_pedestrian_speed_ratio_l152_152251

theorem car_pedestrian_speed_ratio
  (L : ℝ) -- Length of the bridge
  (v_p v_c : ℝ) -- Speed of pedestrian and car
  (h1 : (4 / 9) * L / v_p = (5 / 9) * L / v_p + (5 / 9) * L / v_c) -- Initial meet at bridge start
  (h2 : (4 / 9) * L / v_p = (8 / 9) * L / v_c) -- If pedestrian continues to walk
  : v_c / v_p = 9 :=
sorry

end car_pedestrian_speed_ratio_l152_152251


namespace victor_score_l152_152504

-- Definitions based on the conditions
def max_marks : ℕ := 300
def percentage : ℕ := 80

-- Statement to be proved
theorem victor_score : (percentage * max_marks) / 100 = 240 := by
  sorry

end victor_score_l152_152504


namespace range_of_m_l152_152829

theorem range_of_m (m : ℝ) :
  (∃ (x : ℤ), x > -5 ∧ x ≤ m + 1) ∧ (∀ x, x > -5 → x ≤ m + 1 → x = -4 ∨ x = -3 ∨ x = -2) →
  (-3 ≤ m ∧ m < -2) :=
sorry

end range_of_m_l152_152829


namespace total_candy_l152_152406

/-- Bobby ate 26 pieces of candy initially. -/
def initial_candy : ℕ := 26

/-- Bobby ate 17 more pieces of candy thereafter. -/
def more_candy : ℕ := 17

/-- Prove that the total number of pieces of candy Bobby ate is 43. -/
theorem total_candy : initial_candy + more_candy = 43 := by
  -- The total number of candies should be 26 + 17 which is 43
  sorry

end total_candy_l152_152406


namespace find_number_of_pens_l152_152714

-- Definitions based on the conditions in the problem
def total_utensils (P L : ℕ) : Prop := P + L = 108
def pencils_formula (P L : ℕ) : Prop := L = 5 * P + 12

-- The theorem we need to prove
theorem find_number_of_pens (P L : ℕ) (h1 : total_utensils P L) (h2 : pencils_formula P L) : P = 16 :=
by sorry

end find_number_of_pens_l152_152714


namespace at_least_one_gt_one_l152_152743

theorem at_least_one_gt_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : x > 1 ∨ y > 1 :=
sorry

end at_least_one_gt_one_l152_152743


namespace number_in_parentheses_l152_152978

theorem number_in_parentheses (x : ℤ) (h : x - (-2) = 3) : x = 1 :=
by {
  sorry
}

end number_in_parentheses_l152_152978


namespace total_cost_is_correct_l152_152176

noncomputable def nights : ℕ := 3
noncomputable def cost_per_night : ℕ := 250
noncomputable def discount : ℕ := 100

theorem total_cost_is_correct :
  (nights * cost_per_night) - discount = 650 := by
sorry

end total_cost_is_correct_l152_152176


namespace gcd_subtraction_method_gcd_euclidean_algorithm_l152_152546

theorem gcd_subtraction_method (a b : ℕ) (h₁ : a = 72) (h₂ : b = 168) : Int.gcd a b = 24 := by
  sorry

theorem gcd_euclidean_algorithm (a b : ℕ) (h₁ : a = 98) (h₂ : b = 280) : Int.gcd a b = 14 := by
  sorry

end gcd_subtraction_method_gcd_euclidean_algorithm_l152_152546


namespace point_on_x_axis_m_eq_2_l152_152741

theorem point_on_x_axis_m_eq_2 (m : ℝ) (h : (m + 5, m - 2).2 = 0) : m = 2 :=
sorry

end point_on_x_axis_m_eq_2_l152_152741


namespace parabola_equation_standard_form_l152_152089

theorem parabola_equation_standard_form (p : ℝ) (x y : ℝ)
    (h₁ : y^2 = 2 * p * x)
    (h₂ : y = -4)
    (h₃ : x = -2) : y^2 = -8 * x := by
  sorry

end parabola_equation_standard_form_l152_152089


namespace functional_equation_solution_l152_152083

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) →
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
by
  sorry

end functional_equation_solution_l152_152083


namespace abs_difference_equality_l152_152535

theorem abs_difference_equality : (abs (3 - Real.sqrt 2) - abs (Real.sqrt 2 - 2) = 1) :=
  by
    -- Define our conditions as hypotheses
    have h1 : 3 > Real.sqrt 2 := sorry
    have h2 : Real.sqrt 2 < 2 := sorry
    -- The proof itself is skipped in this step
    sorry

end abs_difference_equality_l152_152535


namespace additional_oil_needed_l152_152965

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

end additional_oil_needed_l152_152965


namespace concentrate_to_water_ratio_l152_152532

theorem concentrate_to_water_ratio :
  ∀ (c w : ℕ), (∀ c, w = 3 * c) → (35 * 3 = 105) → (1 / 3 = (1 : ℝ) / (3 : ℝ)) :=
by
  intros c w h1 h2
  sorry

end concentrate_to_water_ratio_l152_152532


namespace oliver_cards_l152_152467

variable {MC AB BG : ℕ}

theorem oliver_cards : 
  (BG = 48) → 
  (BG = 3 * AB) → 
  (MC = 2 * AB) → 
  MC = 32 := 
by 
  intros h1 h2 h3
  sorry

end oliver_cards_l152_152467


namespace find_f_five_thirds_l152_152877

variable {R : Type*} [LinearOrderedField R]

-- Define the odd function and its properties
variable {f : R → R}
variable (oddf : ∀ x : R, f (-x) = -f x)
variable (propf : ∀ x : R, f (1 + x) = f (-x))
variable (val : f (- (1 / 3 : R)) = 1 / 3)

theorem find_f_five_thirds : f (5 / 3 : R) = 1 / 3 := by
  sorry

end find_f_five_thirds_l152_152877


namespace purchase_price_of_first_commodity_l152_152206

-- Define the conditions
variable (price_first price_second : ℝ)
variable (h1 : price_first - price_second = 127)
variable (h2 : price_first + price_second = 827)

-- Prove the purchase price of the first commodity is $477
theorem purchase_price_of_first_commodity : price_first = 477 :=
by
  sorry

end purchase_price_of_first_commodity_l152_152206


namespace parabola_directrix_l152_152286

theorem parabola_directrix (p : ℝ) (h_focus : ∃ x y : ℝ, y^2 = 2*p*x ∧ 2*x + 3*y - 4 = 0) : 
  ∀ x y : ℝ, y^2 = 2*p*x → x = -p/2 := 
sorry

end parabola_directrix_l152_152286


namespace sum_lcms_equals_l152_152050

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_lcms_equals :
  ( ∑ ν in (Finset.filter is_solution (Finset.range 100)), ν ) = 180 :=
sorry

end sum_lcms_equals_l152_152050


namespace polynomial_consecutive_integers_l152_152919

theorem polynomial_consecutive_integers (a : ℤ) (c : ℤ) (P : ℤ → ℤ)
  (hP : ∀ x : ℤ, P x = 2 * x ^ 3 - 30 * x ^ 2 + c * x)
  (h_consecutive : ∃ a : ℤ, P (a - 1) + 1 = P a ∧ P a = P (a + 1) - 1) :
  a = 5 ∧ c = 149 :=
by
  sorry

end polynomial_consecutive_integers_l152_152919


namespace sum_of_positive_integers_nu_lcm_72_l152_152042

theorem sum_of_positive_integers_nu_lcm_72:
  let ν_values := { ν | Nat.lcm ν 24 = 72 }
  ∑ ν in ν_values, ν = 180 := by
  sorry

end sum_of_positive_integers_nu_lcm_72_l152_152042


namespace maximal_value_fraction_l152_152982

noncomputable def maximum_value_ratio (a b c : ℝ) (S : ℝ) : ℝ :=
  if S = c^2 / 4 then 2 * Real.sqrt 2 else 0

theorem maximal_value_fraction (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (area_cond : 1/2 * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = c^2 / 4) :
  maximum_value_ratio a b c (c^2/4) = 2 * Real.sqrt 2 :=
sorry

end maximal_value_fraction_l152_152982


namespace cheryl_mms_eaten_l152_152805

variable (initial_mms : ℕ) (mms_after_dinner : ℕ) (mms_given_to_sister : ℕ) (total_mms_after_lunch : ℕ)

theorem cheryl_mms_eaten (h1 : initial_mms = 25)
                         (h2 : mms_after_dinner = 5)
                         (h3 : mms_given_to_sister = 13)
                         (h4 : total_mms_after_lunch = initial_mms - mms_after_dinner - mms_given_to_sister) :
                         total_mms_after_lunch = 7 :=
by sorry

end cheryl_mms_eaten_l152_152805


namespace jerry_claim_percentage_l152_152589

theorem jerry_claim_percentage
  (salary_years : ℕ)
  (annual_salary : ℕ)
  (medical_bills : ℕ)
  (punitive_multiplier : ℕ)
  (received_amount : ℕ)
  (total_claim : ℕ)
  (percentage_claim : ℕ) :
  salary_years = 30 →
  annual_salary = 50000 →
  medical_bills = 200000 →
  punitive_multiplier = 3 →
  received_amount = 5440000 →
  total_claim = (annual_salary * salary_years) + medical_bills + (punitive_multiplier * ((annual_salary * salary_years) + medical_bills)) →
  percentage_claim = (received_amount * 100) / total_claim →
  percentage_claim = 80 :=
by
  sorry

end jerry_claim_percentage_l152_152589


namespace common_internal_tangent_length_l152_152614

-- Definitions based on given conditions
def center_distance : ℝ := 50
def radius_small : ℝ := 7
def radius_large : ℝ := 10

-- Target theorem
theorem common_internal_tangent_length :
  let AB := center_distance
  let BE := radius_small + radius_large 
  let AE := Real.sqrt (AB^2 - BE^2)
  AE = Real.sqrt 2211 :=
by
  sorry

end common_internal_tangent_length_l152_152614


namespace find_base_b_l152_152961

theorem find_base_b :
  ∃ b : ℕ, (b > 7) ∧ (b > 10) ∧ (b > 8) ∧ (b > 12) ∧ 
    (4 + 3 = 7) ∧ ((2 + 7 + 1) % b = 3) ∧ ((3 + 4 + 1) % b = 5) ∧ 
    ((5 + 6 + 1) % b = 2) ∧ (1 + 1 = 2)
    ∧ b = 13 :=
by
  sorry

end find_base_b_l152_152961


namespace average_age_increase_l152_152613

theorem average_age_increase (n : ℕ) (m : ℕ) (a b : ℝ) (h1 : n = 19) (h2 : m = 20) (h3 : a = 20) (h4 : b = 40) :
  ((n * a + b) / (n + 1)) - a = 1 :=
by
  -- Proof omitted
  sorry

end average_age_increase_l152_152613


namespace product_is_zero_l152_152538

theorem product_is_zero (n : ℤ) (h : n = 3) :
  (n - 3) * (n - 2) * (n - 1) * n * (n + 1) * (n + 4) = 0 := 
by
  sorry

end product_is_zero_l152_152538


namespace odd_number_adjacent_product_diff_l152_152534

variable (x : ℤ)

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem odd_number_adjacent_product_diff (h : is_odd x)
  (adjacent_diff : x * (x + 2) - x * (x - 2) = 44) : x = 11 :=
by
  sorry

end odd_number_adjacent_product_diff_l152_152534


namespace initial_investment_l152_152270

theorem initial_investment (A r : ℝ) (n : ℕ) (P : ℝ) (hA : A = 630.25) (hr : r = 0.12) (hn : n = 5) :
  A = P * (1 + r) ^ n → P = 357.53 :=
by
  sorry

end initial_investment_l152_152270


namespace function_satisfy_f1_function_satisfy_f2_l152_152230

noncomputable def f1 (x : ℝ) : ℝ := 2
noncomputable def f2 (x : ℝ) : ℝ := x

theorem function_satisfy_f1 : 
  ∀ x y : ℝ, x > 0 → y > 0 → f1 (x + y) + f1 x * f1 y = f1 (x * y) + f1 x + f1 y :=
by 
  intros x y hx hy
  unfold f1
  sorry

theorem function_satisfy_f2 :
  ∀ x y : ℝ, x > 0 → y > 0 → f2 (x + y) + f2 x * f2 y = f2 (x * y) + f2 x + f2 y :=
by 
  intros x y hx hy
  unfold f2
  sorry

end function_satisfy_f1_function_satisfy_f2_l152_152230


namespace polynomial_divisibility_l152_152090

def P (a : ℤ) (x : ℤ) : ℤ := x^1000 + a*x^2 + 9

theorem polynomial_divisibility (a : ℤ) : (P a (-1) = 0) ↔ (a = -10) := by
  sorry

end polynomial_divisibility_l152_152090


namespace license_plate_increase_factor_l152_152338

def old_license_plates := 26^2 * 10^3
def new_license_plates := 26^3 * 10^4

theorem license_plate_increase_factor : (new_license_plates / old_license_plates) = 260 := by
  sorry

end license_plate_increase_factor_l152_152338


namespace perfect_square_factors_450_l152_152154

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l152_152154


namespace coordinates_of_point_P_l152_152334

noncomputable def tangent_slope_4 : Prop :=
  ∀ (x y : ℝ), y = 1 / x → (-1 / (x^2)) = -4 → (x = 1 / 2 ∧ y = 2) ∨ (x = -1 / 2 ∧ y = -2)

theorem coordinates_of_point_P : tangent_slope_4 :=
by sorry

end coordinates_of_point_P_l152_152334


namespace Jason_reroll_exactly_two_dice_probability_l152_152458

noncomputable def probability_reroll_two_dice : ℚ :=
  let favorable_outcomes := 5 * 3 + 1 * 3 + 5 * 3
  let total_possibilities := 6^3
  favorable_outcomes / total_possibilities

theorem Jason_reroll_exactly_two_dice_probability : probability_reroll_two_dice = 5 / 9 := 
  sorry

end Jason_reroll_exactly_two_dice_probability_l152_152458


namespace product_of_possible_values_l152_152754

theorem product_of_possible_values (b : ℝ) (side_length : ℝ) (square_condition : (b - 2) = side_length ∨ (2 - b) = side_length) : 
  (b = -3 ∨ b = 7) → (-3 * 7 = -21) :=
by
  intro h
  sorry

end product_of_possible_values_l152_152754


namespace greatest_possible_sum_of_squares_l152_152631

theorem greatest_possible_sum_of_squares (x y : ℤ) (h : x^2 + y^2 = 36) : x + y ≤ 8 :=
by sorry

end greatest_possible_sum_of_squares_l152_152631


namespace chelsea_cupcakes_time_l152_152261

theorem chelsea_cupcakes_time
  (batches : ℕ)
  (bake_time_per_batch : ℕ)
  (ice_time_per_batch : ℕ)
  (total_time : ℕ)
  (h1 : batches = 4)
  (h2 : bake_time_per_batch = 20)
  (h3 : ice_time_per_batch = 30)
  (h4 : total_time = (bake_time_per_batch + ice_time_per_batch) * batches) :
  total_time = 200 :=
  by
  -- The proof statement here
  -- The proof would go here, but we skip it for now
  sorry

end chelsea_cupcakes_time_l152_152261


namespace pure_imaginary_m_l152_152979

theorem pure_imaginary_m (m : ℝ) (h : (m^2 - m - 2 : ℝ) = 0) : (z : ℂ) = (m + 1) * complex.I :=
begin
  sorry
end

end pure_imaginary_m_l152_152979


namespace javier_total_time_spent_l152_152717

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2

theorem javier_total_time_spent : outlining_time + writing_time + practicing_time = 117 := by
  sorry

end javier_total_time_spent_l152_152717


namespace kenny_played_basketball_last_week_l152_152722

def time_practicing_trumpet : ℕ := 40
def time_running : ℕ := time_practicing_trumpet / 2
def time_playing_basketball : ℕ := time_running / 2
def answer : ℕ := 10

theorem kenny_played_basketball_last_week :
  time_playing_basketball = answer :=
by
  -- sorry to skip the proof
  sorry

end kenny_played_basketball_last_week_l152_152722


namespace product_of_roots_of_quadratic_equation_l152_152041

theorem product_of_roots_of_quadratic_equation :
  ∀ (x : ℝ), (x^2 + 14 * x + 48 = -4) → (-6) * (-8) = 48 :=
by
  sorry

end product_of_roots_of_quadratic_equation_l152_152041


namespace roots_greater_than_one_l152_152558

def quadratic_roots_greater_than_one (a : ℝ) : Prop :=
  ∀ x : ℝ, (1 + a) * x^2 - 3 * a * x + 4 * a = 0 → x > 1

theorem roots_greater_than_one (a : ℝ) :
  -16/7 < a ∧ a < -1 → quadratic_roots_greater_than_one a :=
sorry

end roots_greater_than_one_l152_152558


namespace average_points_per_player_l152_152723

theorem average_points_per_player (Lefty_points Righty_points OtherTeammate_points : ℕ)
  (hL : Lefty_points = 20)
  (hR : Righty_points = Lefty_points / 2)
  (hO : OtherTeammate_points = 6 * Righty_points) :
  (Lefty_points + Righty_points + OtherTeammate_points) / 3 = 30 :=
by
  sorry

end average_points_per_player_l152_152723


namespace janet_saves_minutes_l152_152812

theorem janet_saves_minutes 
  (key_search_time : ℕ := 8) 
  (complaining_time : ℕ := 3) 
  (days_in_week : ℕ := 7) : 
  (key_search_time + complaining_time) * days_in_week = 77 := 
by
  sorry

end janet_saves_minutes_l152_152812


namespace find_integer_K_l152_152698

-- Definitions based on the conditions
def is_valid_K (K Z : ℤ) : Prop :=
  Z = K^4 ∧ 3000 < Z ∧ Z < 4000 ∧ K > 1 ∧ ∃ (z : ℤ), K^4 = z^3

theorem find_integer_K :
  ∃ (K : ℤ), is_valid_K K 2401 :=
by
  sorry

end find_integer_K_l152_152698


namespace fill_tank_time_l152_152921

theorem fill_tank_time (hA : ∀ t : Real, t > 0 → (t / 10) = 1) 
                       (hB : ∀ t : Real, t > 0 → (t / 20) = 1) 
                       (hC : ∀ t : Real, t > 0 → (t / 30) = 1) : 
                       (60 / 7 : Real) = 60 / 7 :=
by
    sorry

end fill_tank_time_l152_152921


namespace reporters_not_covering_politics_l152_152539

-- Definitions of basic quantities
variables (R P : ℝ) (percentage_local : ℝ) (percentage_no_local : ℝ)

-- Conditions from the problem
def conditions : Prop :=
  R = 100 ∧
  percentage_local = 10 ∧
  percentage_no_local = 30 ∧
  percentage_local = 0.7 * P

-- Theorem statement for the problem
theorem reporters_not_covering_politics (h : conditions R P percentage_local percentage_no_local) :
  100 - P = 85.71 :=
by sorry

end reporters_not_covering_politics_l152_152539


namespace no_valid_a_exists_l152_152973

theorem no_valid_a_exists 
  (a : ℝ)
  (h1: ∀ x : ℝ, x^2 + 2*(a+1)*x - (a-1) = 0 → (1 < x ∨ x < 1)) :
  false := by
  sorry

end no_valid_a_exists_l152_152973


namespace calculate_expression_l152_152661

theorem calculate_expression (a : ℝ) : 3 * a * (2 * a^2 - 4 * a) - 2 * a^2 * (3 * a + 4) = -20 * a^2 :=
by
  sorry

end calculate_expression_l152_152661


namespace find_boxes_l152_152001

variable (John Jules Joseph Stan : ℕ)

-- Conditions
axiom h1 : John = 30
axiom h2 : John = 6 * Jules / 5 -- Equivalent to John having 20% more boxes than Jules
axiom h3 : Jules = Joseph + 5
axiom h4 : Joseph = Stan / 5 -- Equivalent to Joseph having 80% fewer boxes than Stan

-- Theorem to prove
theorem find_boxes (h1 : John = 30) (h2 : John = 6 * Jules / 5) (h3 : Jules = Joseph + 5) (h4 : Joseph = Stan / 5) : Stan = 100 :=
sorry

end find_boxes_l152_152001


namespace marbles_total_is_260_l152_152412

/-- Define the number of marbles in each jar. -/
def jar1 : ℕ := 80
def jar2 : ℕ := 2 * jar1
def jar3 : ℕ := jar1 / 4

/-- The total number of marbles Courtney has. -/
def total_marbles : ℕ := jar1 + jar2 + jar3

/-- Proving that the total number of marbles is 260. -/
theorem marbles_total_is_260 : total_marbles = 260 := 
by
  sorry

end marbles_total_is_260_l152_152412


namespace numPerfectSquareFactorsOf450_l152_152147

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l152_152147


namespace martha_black_butterflies_l152_152189

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies : ℕ)
  (h1 : total_butterflies = 11)
  (h2 : blue_butterflies = 4)
  (h3 : blue_butterflies = 2 * yellow_butterflies) :
  ∃ black_butterflies : ℕ, black_butterflies = total_butterflies - blue_butterflies - yellow_butterflies :=
sorry

end martha_black_butterflies_l152_152189


namespace find_f_five_thrids_l152_152880

noncomputable def f : ℝ → ℝ := sorry

lemma odd_fun (x : ℝ) : f (-x) = -f(x) :=
sorry

lemma f_transformation (x : ℝ) : f (1 + x) = f (-x) :=
sorry

lemma f_neg_inv : f (-1 / 3) = 1 / 3 :=
sorry

theorem find_f_five_thrids : f (5 / 3) = 1 / 3 :=
by
  -- application of the conditions stated
  have h1 : f(1 + (5 / 3 - 1)) = f(-(5 / 3 - 1)),
    from f_transformation (5 / 3 - 1),
  have h2 : f(-(5 / 3 - 1)) = -f(5 / 3 - 1),
    from odd_fun (5 / 3 - 1),
  have h3 : 5 / 3 - 1 = 2 / 3,
    by norm_num,
  rw [h3] at h1,
  rw [h2] at h1,
  have h4 : f (-1 / 3) = 1 / 3,
    from f_neg_inv,
  rw [←h4] at h1,
  exact h1,
sorry

end find_f_five_thrids_l152_152880


namespace route_down_distance_l152_152523

theorem route_down_distance :
  ∀ (rate_up rate_down time_up time_down distance_up distance_down : ℝ),
    -- Conditions
    rate_down = 1.5 * rate_up →
    time_up = time_down →
    rate_up = 6 →
    time_up = 2 →
    distance_up = rate_up * time_up →
    distance_down = rate_down * time_down →
    -- Question: Prove the correct answer
    distance_down = 18 :=
by
  intros rate_up rate_down time_up time_down distance_up distance_down h1 h2 h3 h4 h5 h6
  sorry

end route_down_distance_l152_152523


namespace cubic_sum_l152_152311

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l152_152311


namespace geometric_sequence_ratio_l152_152433

theorem geometric_sequence_ratio 
  (a_n b_n : ℕ → ℝ) 
  (S_n T_n : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, S_n n = a_n n * (1 - (1/2)^n)) 
  (h2 : ∀ n : ℕ, T_n n = b_n n * (1 - (1/3)^n))
  (h3 : ∀ n, n > 0 → (S_n n) / (T_n n) = (3^n + 1) / 4) : 
  (a_n 3) / (b_n 3) = 9 :=
by
  sorry

end geometric_sequence_ratio_l152_152433


namespace difference_in_surface_areas_l152_152068

-- Define the conditions: volumes and number of cubes
def V_large : ℕ := 343
def n : ℕ := 343
def V_small : ℕ := 1

-- Define the function to calculate the side length of a cube given its volume
def side_length (V : ℕ) : ℕ := V^(1/3 : ℕ)

-- Specify the side lengths of the larger and smaller cubes
def s_large : ℕ := side_length V_large
def s_small : ℕ := side_length V_small

-- Define the function to calculate the surface area of a cube given its side length
def surface_area (s : ℕ) : ℕ := 6 * s^2

-- Specify the surface areas of the larger cube and the total of the smaller cubes
def SA_large : ℕ := surface_area s_large
def SA_small_total : ℕ := n * surface_area s_small

-- State the theorem to prove
theorem difference_in_surface_areas : SA_small_total - SA_large = 1764 :=
by {
  -- Intentionally omit proof, as per instructions
  sorry
}

end difference_in_surface_areas_l152_152068


namespace fold_point_area_sum_l152_152561

noncomputable def fold_point_area (AB AC : ℝ) (angle_B : ℝ) : ℝ :=
  let BC := Real.sqrt (AB ^ 2 + AC ^ 2)
  -- Assuming the fold point area calculation as per the problem's solution
  let q := 270
  let r := 324
  let s := 3
  q * Real.pi - r * Real.sqrt s

theorem fold_point_area_sum (AB AC : ℝ) (angle_B : ℝ) (hAB : AB = 36) (hAC : AC = 72) (hangle_B : angle_B = π / 2) :
  let S := fold_point_area AB AC angle_B
  ∃ q r s : ℕ, S = q * Real.pi - r * Real.sqrt s ∧ q + r + s = 597 :=
by
  sorry

end fold_point_area_sum_l152_152561


namespace line_parallel_not_coincident_l152_152443

theorem line_parallel_not_coincident (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ k : ℝ, (∀ x y : ℝ, a * x + 2 * y + 6 = k * (x + (a - 1) * y + (a^2 - 1))) →
  a = -1 :=
by
  sorry

end line_parallel_not_coincident_l152_152443


namespace hall_length_l152_152785

theorem hall_length
  (width : ℝ)
  (stone_length : ℝ)
  (stone_width : ℝ)
  (num_stones : ℕ)
  (h₁ : width = 15)
  (h₂ : stone_length = 0.8)
  (h₃ : stone_width = 0.5)
  (h₄ : num_stones = 1350) :
  ∃ length : ℝ, length = 36 :=
by
  sorry

end hall_length_l152_152785


namespace derivative_at_one_is_four_l152_152363

-- Define the function y = x^2 + 2x + 1
def f (x : ℝ) := x^2 + 2*x + 1

-- State the theorem: The derivative of f at x = 1 is 4
theorem derivative_at_one_is_four : (deriv f 1) = 4 :=
by
  -- The proof is omitted here.
  sorry

end derivative_at_one_is_four_l152_152363


namespace salt_added_correctly_l152_152658

-- Define the problem's conditions and the correct answer in Lean
variable (x : ℝ) (y : ℝ)
variable (S : ℝ := 0.2 * x) -- original salt
variable (E : ℝ := (1 / 4) * x) -- evaporated water
variable (New_volume : ℝ := x - E + 10) -- new volume after adding water

theorem salt_added_correctly :
  x = 150 → y = (1 / 3) * New_volume - S :=
by
  sorry

end salt_added_correctly_l152_152658


namespace number_of_perfect_square_divisors_of_450_l152_152149

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l152_152149


namespace lengthDE_is_correct_l152_152912

noncomputable def triangleBase : ℝ := 12

noncomputable def triangleArea (h : ℝ) : ℝ := (1 / 2) * triangleBase * h

noncomputable def projectedArea (h : ℝ) : ℝ := 0.16 * triangleArea h

noncomputable def lengthDE (h : ℝ) : ℝ := 0.4 * triangleBase

theorem lengthDE_is_correct (h : ℝ) :
  lengthDE h = 4.8 :=
by
  simp [lengthDE, triangleBase, triangleArea, projectedArea]
  sorry

end lengthDE_is_correct_l152_152912


namespace number_of_true_propositions_is_one_l152_152964

-- Define propositions
def prop1 (a b c : ℝ) : Prop := a > b ∧ c ≠ 0 → a * c > b * c
def prop2 (a b c : ℝ) : Prop := a > b → a * c^2 > b * c^2
def prop3 (a b c : ℝ) : Prop := a * c^2 > b * c^2 → a > b
def prop4 (a b : ℝ) : Prop := a > b → (1 / a) < (1 / b)
def prop5 (a b c d : ℝ) : Prop := a > b ∧ b > 0 ∧ c > d → a * c > b * d

-- The main theorem stating the number of true propositions
theorem number_of_true_propositions_is_one (a b c d : ℝ) :
  (prop3 a b c) ∧ (¬ prop1 a b c) ∧ (¬ prop2 a b c) ∧ (¬ prop4 a b) ∧ (¬ prop5 a b c d) :=
by
  sorry

end number_of_true_propositions_is_one_l152_152964


namespace commuting_hours_l152_152587

theorem commuting_hours (walk_hours_per_trip bike_hours_per_trip : ℕ) 
  (walk_trips_per_week bike_trips_per_week : ℕ) 
  (walk_hours_per_trip = 2) 
  (bike_hours_per_trip = 1)
  (walk_trips_per_week = 3) 
  (bike_trips_per_week = 2) : 
  (2 * (walk_hours_per_trip * walk_trips_per_week) + 2 * (bike_hours_per_trip * bike_trips_per_week)) = 16 := 
  by
  sorry

end commuting_hours_l152_152587


namespace remainder_is_162_l152_152822

def polynomial (x : ℝ) : ℝ := 2 * x^4 - x^3 + 4 * x^2 - 5 * x + 6

theorem remainder_is_162 : polynomial 3 = 162 :=
by 
  sorry

end remainder_is_162_l152_152822


namespace initial_customers_l152_152943

theorem initial_customers (tables : ℕ) (people_per_table : ℕ) (customers_left : ℕ) (h1 : tables = 5) (h2 : people_per_table = 9) (h3 : customers_left = 17) :
  tables * people_per_table + customers_left = 62 :=
by
  sorry

end initial_customers_l152_152943


namespace geometric_sequence_third_term_l152_152024

-- Define the problem statement in Lean 4
theorem geometric_sequence_third_term :
  ∃ r : ℝ, (a = 1024) ∧ (a_5 = 128) ∧ (a_5 = a * r^4) ∧ 
  (a_3 = a * r^2) ∧ (a_3 = 256) :=
sorry

end geometric_sequence_third_term_l152_152024


namespace find_angle_A_find_area_ABC_l152_152866

-- Define the conditions
variables {A B C a b c : ℝ} (hTriangle : triangle A B C)
variables (hOpposites : opposite_sides_to_angles a A b B c C)
variables (f : ℝ → ℝ) (hf : ∀ x, f x = 2 * cos x * sin (x - A))
variables (xmin : ℝ) (hxmin : xmin = 11 * π / 12)  -- x = 11π / 12
variables (ha : ℝ) (ha_value : ha = 7)  -- a = 7
variables (sinB_plus_sinC : ℝ) (hsinB_plus_sinC : sinB_plus_sinC = sin B + sin C)
variables (hsinBplusC_value : sinB_plus_sinC = 13 * sqrt 3 / 14)

-- Define theorem for problem 1: finding angle A
theorem find_angle_A :
  A = π / 3 :=
sorry

-- Given that A = π / 3, define theorem for problem 2: finding the area of triangle ABC
theorem find_area_ABC (hA : A = π / 3) :
  area (triangle A B C) = 10 * sqrt 3 :=
sorry

end find_angle_A_find_area_ABC_l152_152866


namespace least_subtraction_l152_152821

theorem least_subtraction (n : ℕ) (d : ℕ) (r : ℕ) (h1 : n = 45678) (h2 : d = 47) (h3 : n % d = r) : r = 35 :=
by {
  sorry
}

end least_subtraction_l152_152821


namespace chord_intersects_inner_circle_l152_152036

noncomputable def probability_chord_intersects_inner_circle
  (r1 r2 : ℝ) (h1 : r1 = 2) (h2 : r2 = 5) : ℝ :=
0.098

theorem chord_intersects_inner_circle :
  probability_chord_intersects_inner_circle 2 5 rfl rfl = 0.098 :=
sorry

end chord_intersects_inner_circle_l152_152036


namespace proof_problem_l152_152593

open ProbabilityTheory
open Set

variables {Ω : Type*} [ProbabilitySpace Ω]
variables (A B : Set Ω)

-- conditions
theorem proof_problem (h_exclusive : Disjoint A B) (h_PA : 0 < P[A]) (h_PB : 0 < P[B]) : 
  P[A ∪ B] = P[A] + P[B] :=
sorry

end proof_problem_l152_152593


namespace solve_for_m_l152_152858

theorem solve_for_m (x m : ℝ) (h1 : 2 * 1 - m = -3) : m = 5 :=
by
  sorry

end solve_for_m_l152_152858


namespace max_area_trapezoid_l152_152762

theorem max_area_trapezoid :
  ∀ {AB CD : ℝ}, 
    AB = 6 → CD = 14 → 
    (∃ (r1 r2 : ℝ), r1 = AB / 2 ∧ r2 = CD / 2 ∧ r1 + r2 = 10) → 
    (1 / 2 * (AB + CD) * 10 = 100) :=
by
  intros AB CD hAB hCD hExist
  sorry

end max_area_trapezoid_l152_152762


namespace triangle_is_isosceles_right_l152_152336

theorem triangle_is_isosceles_right (A B C a b c : ℝ) 
  (h : a / (Real.cos A) = b / (Real.cos B) ∧ b / (Real.cos B) = c / (Real.sin C)) :
  A = π/4 ∧ B = π/4 ∧ C = π/2 := 
sorry

end triangle_is_isosceles_right_l152_152336


namespace gcd_lcm_product_l152_152552

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [h₁, h₂]
  unfold Nat.gcd Nat.lcm
  sorry

end gcd_lcm_product_l152_152552


namespace remainder_of_82460_div_8_l152_152506

theorem remainder_of_82460_div_8 :
  82460 % 8 = 4 :=
sorry

end remainder_of_82460_div_8_l152_152506


namespace problem_solution_set_l152_152369

open Nat

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)
def permutation (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem problem_solution_set :
  {n : ℕ | 2 * combination n 3 ≤ permutation n 2} = {n | n = 3 ∨ n = 4 ∨ n = 5} :=
by
  sorry

end problem_solution_set_l152_152369


namespace pos_int_satisfy_inequality_l152_152371

open Nat

def C (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))
def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

theorem pos_int_satisfy_inequality :
  {n : ℕ // 0 < n ∧ 2 * C n 3 ≤ A n 2} = {n // n = 3 ∨ n = 4 ∨ n = 5} :=
by
  sorry

end pos_int_satisfy_inequality_l152_152371


namespace count_distinct_ways_l152_152381

theorem count_distinct_ways (p : ℕ × ℕ → ℕ) (h_condition : ∃ j : ℕ × ℕ, j ∈ [(0, 0), (0, 1)] ∧ p j = 4)
  (h_grid_size : ∀ i : ℕ × ℕ, i ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → 1 ≤ p i ∧ p i ≤ 4)
  (h_distinct : ∀ i j : ℕ × ℕ, i ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → j ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → i ≠ j → p i ≠ p j) :
  ∃! l : Finset (ℕ × ℕ → ℕ), l.card = 12 :=
by
  sorry

end count_distinct_ways_l152_152381


namespace pow_fraction_eq_l152_152507

theorem pow_fraction_eq : (4:ℕ) = 2^2 ∧ (8:ℕ) = 2^3 → (4^800 / 8^400 = 2^400) :=
by
  -- proof steps should go here, but they are omitted as per the instruction
  sorry

end pow_fraction_eq_l152_152507


namespace coverage_is_20_l152_152913

noncomputable def cost_per_kg : ℝ := 60
noncomputable def total_cost : ℝ := 1800
noncomputable def side_length : ℝ := 10

-- Surface area of one side of the cube
noncomputable def area_side : ℝ := side_length * side_length

-- Total surface area of the cube
noncomputable def total_area : ℝ := 6 * area_side

-- Kilograms of paint used
noncomputable def kg_paint_used : ℝ := total_cost / cost_per_kg

-- Coverage per kilogram of paint
noncomputable def coverage_per_kg (total_area : ℝ) (kg_paint_used : ℝ) : ℝ := total_area / kg_paint_used

theorem coverage_is_20 : coverage_per_kg total_area kg_paint_used = 20 := by
  sorry

end coverage_is_20_l152_152913


namespace line_intersects_parabola_at_vertex_l152_152957

theorem line_intersects_parabola_at_vertex :
  ∃ (a : ℝ), (∀ x : ℝ, -x + a = x^2 + a^2) ↔ a = 0 ∨ a = 1 :=
by
  sorry

end line_intersects_parabola_at_vertex_l152_152957


namespace games_against_other_division_l152_152062

theorem games_against_other_division
  (N M : ℕ) (h1 : N > 2 * M) (h2 : M > 5)
  (total_games : N * 4 + 5 * M = 82) :
  5 * M = 30 :=
by
  sorry

end games_against_other_division_l152_152062


namespace satisfy_third_eq_l152_152745

theorem satisfy_third_eq 
  (x y : ℝ) 
  (h1 : x^2 - 3 * x * y + 2 * y^2 + x - y = 0)
  (h2 : x^2 - 2 * x * y + y^2 - 5 * x + 7 * y = 0) 
  : x * y - 12 * x + 15 * y = 0 :=
by
  sorry

end satisfy_third_eq_l152_152745


namespace expand_expression_l152_152540

variable (x y : ℝ)

theorem expand_expression :
  ((6 * x + 8 - 3 * y) * (4 * x - 5 * y)) = 
  (24 * x^2 - 42 * x * y + 32 * x - 40 * y + 15 * y^2) :=
by
  sorry

end expand_expression_l152_152540


namespace cubic_sum_l152_152320

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l152_152320


namespace smallest_m_plus_n_l152_152022

theorem smallest_m_plus_n : ∃ (m n : ℕ), m > 1 ∧ 
  (∃ (a b : ℝ), a = (1 : ℝ) / (m * n : ℝ) ∧ b = (m : ℝ) / (n : ℝ) ∧ b - a = (1 : ℝ) / 1007) ∧
  (∀ (k l : ℕ), k > 1 ∧ 
    (∃ (c d : ℝ), c = (1 : ℝ) / (k * l : ℝ) ∧ d = (k : ℝ) / (l : ℝ) ∧ d - c = (1 : ℝ) / 1007) → m + n ≤ k + l) ∧ 
  m + n = 19099 :=
sorry

end smallest_m_plus_n_l152_152022


namespace cubic_sum_identity_l152_152293

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l152_152293


namespace factorial_division_l152_152077

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem factorial_division :
  (factorial 10) / ((factorial 7) * (factorial 3)) = 120 := by sorry

end factorial_division_l152_152077


namespace min_a_plus_5b_l152_152683

theorem min_a_plus_5b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a * b + b^2 = b + 1) : 
  a + 5 * b ≥ 7 / 2 :=
by
  sorry

end min_a_plus_5b_l152_152683


namespace marbles_leftover_l152_152198

theorem marbles_leftover (r p : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) : (r + p) % 8 = 4 :=
by
  sorry

end marbles_leftover_l152_152198


namespace min_value_of_squares_find_p_l152_152273

open Real

theorem min_value_of_squares (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (eqn : a + sqrt 2 * b + sqrt 3 * c = 2 * sqrt 3) :
  a^2 + b^2 + c^2 = 2 :=
by sorry

theorem find_p (m : ℝ) (hm : m = 2) (p q : ℝ) :
  (∀ x, |x - 3| ≥ m ↔ x^2 + p * x + q ≥ 0) → p = -6 :=
by sorry

end min_value_of_squares_find_p_l152_152273


namespace average_vegetables_per_week_l152_152636

theorem average_vegetables_per_week (P Vp S W : ℕ) (h1 : P = 200) (h2 : Vp = 2) (h3 : S = 25) (h4 : W = 2) :
  (P / Vp) / S / W = 2 :=
by
  sorry

end average_vegetables_per_week_l152_152636


namespace find_d_l152_152350

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := c * x - 3

theorem find_d (c d : ℝ) (h : ∀ x, f (g x c) c = 15 * x + d) : d = -12 :=
by
  have h1 : ∀ x, f (g x c) c = 5 * (c * x - 3) + c := by intros; simp [f, g]
  have h2 : ∀ x, 5 * (c * x - 3) + c = 5 * c * x + c - 15 := by intros; ring
  specialize h 0
  rw [h1, h2] at h
  sorry

end find_d_l152_152350


namespace intersection_is_integer_for_m_l152_152165

noncomputable def intersects_at_integer_point (m : ℤ) : Prop :=
∃ x y : ℤ, y = x - 4 ∧ y = m * x + 2 * m

theorem intersection_is_integer_for_m :
  intersects_at_integer_point 8 :=
by
  -- The proof would go here
  sorry

end intersection_is_integer_for_m_l152_152165


namespace find_solutions_l152_152819

def binomial_coefficient (n m : ℕ) : ℕ :=
  n.factorial / (m.factorial * (n - m).factorial)

def is_solution (n m : ℕ) : Prop :=
  binomial_coefficient n (m - 1) = binomial_coefficient (n - 1) m

theorem find_solutions :
  ∀ k ∈ ℕ, is_solution (Fibonacci (2 * k) * Fibonacci (2 * k + 1)) (Fibonacci (2 * k) * Fibonacci (2 * k - 1)) :=
by 
  sorry

end find_solutions_l152_152819


namespace number_of_perfect_square_factors_l152_152138

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l152_152138


namespace circle_formed_by_PO_equals_3_l152_152599

variable (P : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ)
variable (h_O_fixed : True)
variable (h_PO_constant : dist P O = 3)

theorem circle_formed_by_PO_equals_3 : 
  {P | ∃ (x y : ℝ), dist (x, y) O = 3} = {P | (dist P O = r) ∧ (r = 3)} :=
by
  sorry

end circle_formed_by_PO_equals_3_l152_152599


namespace quadratic_inequality_solution_l152_152974

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, x^2 + m * x + 1 ≥ 0) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end quadratic_inequality_solution_l152_152974


namespace find_larger_number_l152_152931

theorem find_larger_number (L S : ℕ)
  (h1 : L - S = 1370)
  (h2 : L = 6 * S + 15) :
  L = 1641 := sorry

end find_larger_number_l152_152931


namespace max_f_l152_152855

open Real

variables (x m : ℝ)

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (sqrt 3 * sin x, m + cos x)
def b : ℝ × ℝ := (cos x, -m + cos x)

-- Definition of the function f
def f (x : ℝ) : ℝ := (sqrt 3 * sin x) * (cos x) + (m + cos x) * (-m + cos x)

-- Simplified expression for f(x)
lemma f_simplified (x m : ℝ) : f x = sin (2 * x + π / 6) + 1 / 2 - m ^ 2 := by
  sorry

-- Prove that the maximum value of f(x) in the given range is -5/2 at x = π/6
theorem max_f (x : ℝ) (h1 : -π/6 ≤ x) (h2 : x ≤ π/3) (h3 : ∀ m, f x ≥ -4) :
  ∃ x, f x = -5/2 ∧ x = π/6 :=
by
  sorry

end max_f_l152_152855


namespace sales_in_fourth_month_l152_152067

theorem sales_in_fourth_month 
  (s1 s2 s3 s5 s6 avg : ℝ) 
  (h_s1 : s1 = 8435) 
  (h_s2 : s2 = 8927) 
  (h_s3 : s3 = 8855) 
  (h_s5 : s5 = 8562) 
  (h_s6 : s6 = 6991)
  (h_avg : avg = 8500) :
  (6 * avg - (s1 + s2 + s3 + s5 + s6) = 9230) := 
by
  rw [h_avg, h_s1, h_s2, h_s3, h_s5, h_s6]
  simp
  rw [mul_comm 6 8500, ← add_assoc, mul_comm 8500 6]
  norm_num

end sales_in_fourth_month_l152_152067


namespace determine_number_of_shelves_l152_152492

-- Define the total distance Karen bikes round trip
def total_distance : ℕ := 3200

-- Define the number of books per shelf
def books_per_shelf : ℕ := 400

-- Calculate the one-way distance from Karen's home to the library
def one_way_distance (total_distance : ℕ) : ℕ := total_distance / 2

-- Define the total number of books, which is the same as the one-way distance
def total_books (one_way_distance : ℕ) : ℕ := one_way_distance

-- Calculate the number of shelves
def number_of_shelves (total_books : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books / books_per_shelf

theorem determine_number_of_shelves :
  number_of_shelves (total_books (one_way_distance total_distance)) books_per_shelf = 4 :=
by 
  -- the proof would go here
  sorry

end determine_number_of_shelves_l152_152492


namespace arrangement_count_l152_152759

def number_of_arrangements (n : ℕ) : ℕ :=
  if n = 6 then 5 * (Nat.factorial 5) else 0

theorem arrangement_count : number_of_arrangements 6 = 600 :=
by
  sorry

end arrangement_count_l152_152759


namespace cubic_sum_identity_l152_152298

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l152_152298


namespace directrix_of_parabola_l152_152086

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = (x^2 - 4 * x + 3) / 8 → y = -9 / 8 :=
by
  sorry

end directrix_of_parabola_l152_152086


namespace value_of_Y_is_669_l152_152565

theorem value_of_Y_is_669 :
  let A := 3009 / 3
  let B := A / 3
  let Y := A - B
  Y = 669 :=
by
  sorry

end value_of_Y_is_669_l152_152565


namespace pathway_area_ratio_l152_152250

theorem pathway_area_ratio (AB AD: ℝ) (r: ℝ) (A_rectangle A_circles: ℝ):
  AB = 24 → (AD / AB) = (4 / 3) → r = AB / 2 → 
  A_rectangle = AD * AB → A_circles = π * r^2 →
  (A_rectangle / A_circles) = 16 / (3 * π) :=
by
  sorry

end pathway_area_ratio_l152_152250


namespace functional_equation_solutions_l152_152596

theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)) : 
  (∀ x : ℝ, f x = 0) ∨
  (∀ x : ℝ, f x = x - 1) ∨
  (∀ x : ℝ, f x = 1 - x) :=
sorry

end functional_equation_solutions_l152_152596


namespace hyperbola_vertex_distance_l152_152667

open Real

/-- The distance between the vertices of the hyperbola represented by the equation
    (y-4)^2 / 32 - (x+3)^2 / 18 = 1 is 8√2. -/
theorem hyperbola_vertex_distance :
  let a := sqrt 32
  2 * a = 8 * sqrt 2 :=
by
  sorry

end hyperbola_vertex_distance_l152_152667


namespace number_of_cows_l152_152651

def each_cow_milk_per_day : ℕ := 1000
def total_milk_per_week : ℕ := 364000
def days_in_week : ℕ := 7

theorem number_of_cows : 
  (total_milk_per_week = 364000) →
  (each_cow_milk_per_day = 1000) →
  (days_in_week = 7) →
  (total_milk_per_week / (each_cow_milk_per_day * days_in_week)) = 52 :=
by
  sorry

end number_of_cows_l152_152651


namespace alice_average_speed_l152_152795

def average_speed (distance1 speed1 distance2 speed2 totalDistance totalTime : ℚ) :=
  totalDistance / totalTime

theorem alice_average_speed : 
  let d1 := 45
  let s1 := 15
  let d2 := 15
  let s2 := 45
  let totalDistance := d1 + d2
  let totalTime := (d1 / s1) + (d2 / s2)
  average_speed d1 s1 d2 s2 totalDistance totalTime = 18 :=
by
  sorry

end alice_average_speed_l152_152795


namespace systematic_sampling_id_fourth_student_l152_152781

theorem systematic_sampling_id_fourth_student (n : ℕ) (a b c d : ℕ) (h1 : n = 54) 
(h2 : a = 3) (h3 : b = 29) (h4 : c = 42) (h5 : d = a + 13) : d = 16 :=
by
  sorry

end systematic_sampling_id_fourth_student_l152_152781


namespace stratified_sampling_2nd_year_students_l152_152389

theorem stratified_sampling_2nd_year_students
  (students_1st_year : ℕ) (students_2nd_year : ℕ) (students_3rd_year : ℕ) (total_sample_size : ℕ) :
  students_1st_year = 1000 ∧ students_2nd_year = 800 ∧ students_3rd_year = 700 ∧ total_sample_size = 100 →
  (students_2nd_year * total_sample_size / (students_1st_year + students_2nd_year + students_3rd_year) = 32) :=
by
  intro h
  sorry

end stratified_sampling_2nd_year_students_l152_152389


namespace x_cubed_plus_y_cubed_l152_152328

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l152_152328


namespace product_telescope_l152_152926

theorem product_telescope : ((1 + (1 / 1)) * 
                             (1 + (1 / 2)) * 
                             (1 + (1 / 3)) * 
                             (1 + (1 / 4)) * 
                             (1 + (1 / 5)) * 
                             (1 + (1 / 6)) * 
                             (1 + (1 / 7)) * 
                             (1 + (1 / 8)) * 
                             (1 + (1 / 9)) * 
                             (1 + (1 / 10))) = 11 := 
by
  sorry

end product_telescope_l152_152926


namespace sushi_cost_l152_152533

variable (x : ℕ)

theorem sushi_cost (h1 : 9 * x = 180) : x + (9 * x) = 200 :=
by 
  sorry

end sushi_cost_l152_152533


namespace percent_increase_visual_range_l152_152779

theorem percent_increase_visual_range (original new : ℝ) (h_original : original = 60) (h_new : new = 150) : 
  ((new - original) / original) * 100 = 150 :=
by
  sorry

end percent_increase_visual_range_l152_152779


namespace jackson_chairs_l152_152997

theorem jackson_chairs (a b c d : ℕ) (h1 : a = 6) (h2 : b = 4) (h3 : c = 12) (h4 : d = 6) : a * b + c * d = 96 := 
by sorry

end jackson_chairs_l152_152997


namespace derivative_at_one_l152_152560

section

variable {f : ℝ → ℝ}

-- Define the condition
def limit_condition (f : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (1 + Δx) - f (1 - Δx)) / Δx + 6) < ε

-- State the main theorem
theorem derivative_at_one (h : limit_condition f) : deriv f 1 = -3 :=
by
  sorry

end

end derivative_at_one_l152_152560


namespace haley_cider_pints_l152_152110

noncomputable def apples_per_farmhand_per_hour := 240
noncomputable def working_hours := 5
noncomputable def total_farmhands := 6

noncomputable def golden_delicious_per_pint := 20
noncomputable def pink_lady_per_pint := 40
noncomputable def golden_delicious_ratio := 1
noncomputable def pink_lady_ratio := 2

noncomputable def total_apples := total_farmhands * apples_per_farmhand_per_hour * working_hours
noncomputable def total_parts := golden_delicious_ratio + pink_lady_ratio

noncomputable def golden_delicious_apples := total_apples / total_parts
noncomputable def pink_lady_apples := golden_delicious_apples * pink_lady_ratio

noncomputable def pints_golden_delicious := golden_delicious_apples / golden_delicious_per_pint
noncomputable def pints_pink_lady := pink_lady_apples / pink_lady_per_pint

theorem haley_cider_pints : 
  total_apples = 7200 → 
  golden_delicious_apples = 2400 → 
  pink_lady_apples = 4800 → 
  pints_golden_delicious = 120 → 
  pints_pink_lady = 120 → 
  pints_golden_delicious = pints_pink_lady →
  pints_golden_delicious = 120 :=
by
  sorry

end haley_cider_pints_l152_152110


namespace fraction_paint_left_after_third_day_l152_152339

noncomputable def original_paint : ℝ := 2
noncomputable def paint_after_first_day : ℝ := original_paint - (1 / 2 * original_paint)
noncomputable def paint_after_second_day : ℝ := paint_after_first_day - (1 / 4 * paint_after_first_day)
noncomputable def paint_after_third_day : ℝ := paint_after_second_day - (1 / 2 * paint_after_second_day)

theorem fraction_paint_left_after_third_day :
  paint_after_third_day / original_ppaint = 3 / 8 :=
sorry

end fraction_paint_left_after_third_day_l152_152339


namespace find_n_l152_152710

theorem find_n (n k : ℕ) (b : ℝ) (h_n2 : n ≥ 2) (h_ab : b ≠ 0 ∧ k > 0) (h_a_eq : ∀ (a : ℝ), a = k^2 * b) :
  (∀ (S : ℕ → ℝ → ℝ), S 1 b + S 2 b = 0) →
  n = 2 * k + 1 := 
sorry

end find_n_l152_152710


namespace contest_B_third_place_4_competitions_l152_152576

/-- Given conditions:
1. There are three contestants: A, B, and C.
2. Scores for the first three places in each knowledge competition are \(a\), \(b\), and \(c\) where \(a > b > c\) and \(a, b, c ∈ ℕ^*\).
3. The final score of A is 26 points.
4. The final scores of both B and C are 11 points.
5. Contestant B won first place in one of the competitions.
Prove that Contestant B won third place in four competitions.
-/
theorem contest_B_third_place_4_competitions
  (a b c : ℕ)
  (ha : a > b)
  (hb : b > c)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hc_pos : 0 < c)
  (hA_score : a + a + a + a + b + c = 26)
  (hB_score : a + c + c + c + c + b = 11)
  (hC_score : b + b + b + b + c + c = 11) :
  ∃ n1 n3 : ℕ,
    n1 = 1 ∧ n3 = 4 ∧
    ∃ k m l p1 p2 : ℕ,
      n1 * a + k * a + l * a + m * a + p1 * a + p2 * a + p1 * b + k * b + p2 * b + n3 * c = 11 := sorry

end contest_B_third_place_4_competitions_l152_152576


namespace perfect_square_factors_450_l152_152131

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l152_152131


namespace find_BC_line_eq_l152_152104

def line1_altitude : Prop := ∃ x y : ℝ, 2*x - 3*y + 1 = 0
def line2_altitude : Prop := ∃ x y : ℝ, x + y = 0
def vertex_A : Prop := ∃ a1 a2 : ℝ, a1 = 1 ∧ a2 = 2
def side_BC_equation : Prop := ∃ b c d : ℝ, b = 2 ∧ c = 3 ∧ d = 7

theorem find_BC_line_eq (H1 : line1_altitude) (H2 : line2_altitude) (H3 : vertex_A) : side_BC_equation :=
sorry

end find_BC_line_eq_l152_152104


namespace part1_part2_l152_152680

open Finset
open BigOperators

/-- Given a sequence {a_n} with initial conditions and relationships, prove that a_n = n for all natural numbers n. --/
theorem part1 (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, (a n) = (2 * S n) / (n + 1)) :
  ∀ n, a n = n := 
begin
  sorry
end

/-- Given that a_n equals n for all natural numbers n, prove the sum of the sequence {a_n / 2^n} is T_n = 2 - (n + 2) * (1 / 2)^n. --/
theorem part2 (a : ℕ → ℕ) (T : ℕ → ℝ)
  (h : ∀ n, a n = n) :
  ∀ n, T n = ↑2 - ((n + 2) * (1 / 2)^n) := 
begin
  sorry
end

end part1_part2_l152_152680


namespace alice_has_ball_after_three_turns_l152_152072

def alice_keeps_ball (prob_Alice_to_Bob: ℚ) (prob_Bob_to_Alice: ℚ): ℚ := 
  let prob_Alice_keeps := 1 - prob_Alice_to_Bob
  let prob_Bob_keeps := 1 - prob_Bob_to_Alice
  let path1 := prob_Alice_to_Bob * prob_Bob_to_Alice * prob_Alice_keeps
  let path2 := prob_Alice_keeps * prob_Alice_keeps * prob_Alice_keeps
  path1 + path2

theorem alice_has_ball_after_three_turns:
  alice_keeps_ball (1/2) (1/3) = 5/24 := 
by
  sorry

end alice_has_ball_after_three_turns_l152_152072


namespace emily_collected_total_eggs_l152_152809

def eggs_in_setA : ℕ := (200 * 36) + (250 * 24)
def eggs_in_setB : ℕ := (375 * 42) - 80
def eggs_in_setC : ℕ := (560 / 2 * 50) + (560 / 2 * 32)

def total_eggs_collected : ℕ := eggs_in_setA + eggs_in_setB + eggs_in_setC

theorem emily_collected_total_eggs : total_eggs_collected = 51830 := by
  -- proof goes here
  sorry

end emily_collected_total_eggs_l152_152809


namespace maggie_bouncy_balls_l152_152010

theorem maggie_bouncy_balls (yellow_packs green_pack_given green_pack_bought : ℝ)
    (balls_per_pack : ℝ)
    (hy : yellow_packs = 8.0)
    (hg_given : green_pack_given = 4.0)
    (hg_bought : green_pack_bought = 4.0)
    (hbp : balls_per_pack = 10.0) :
    (yellow_packs * balls_per_pack + green_pack_bought * balls_per_pack - green_pack_given * balls_per_pack = 80.0) :=
by
  sorry

end maggie_bouncy_balls_l152_152010


namespace trees_died_in_typhoon_l152_152109

-- Define the total number of trees, survived trees, and died trees
def total_trees : ℕ := 14

def survived_trees (S : ℕ) : ℕ := S

def died_trees (S : ℕ) : ℕ := S + 4

-- The Lean statement that formalizes the proof problem
theorem trees_died_in_typhoon : ∃ S : ℕ, survived_trees S + died_trees S = total_trees ∧ died_trees S = 9 :=
by
  -- Provide a placeholder for the proof
  sorry

end trees_died_in_typhoon_l152_152109


namespace number_of_knights_l152_152237

def traveler := Type
def is_knight (t : traveler) : Prop := sorry
def is_liar (t : traveler) : Prop := sorry

axiom total_travelers : Finset traveler
axiom vasily : traveler
axiom  h_total : total_travelers.card = 16

axiom kn_lie (t : traveler) : is_knight t ∨ is_liar t

axiom vasily_liar : is_liar vasily
axiom contradictory_statements_in_room (rooms: Finset (Finset traveler)):
  (∀ room ∈ rooms, ∃ t ∈ room, (is_liar t ∧ is_knight t))
  ∧
  (∀ room ∈ rooms, ∃ t ∈ room, (is_knight t ∧ is_liar t))

theorem number_of_knights : 
  ∃ k, k = 9 ∧ (∃ l, l = 7 ∧ ∀ t ∈ total_travelers, (is_knight t ∨ is_liar t)) :=
sorry

end number_of_knights_l152_152237


namespace sum_of_roots_l152_152274

theorem sum_of_roots (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - 2 * x₁ - 8 = 0) 
  (h₂ : x₂^2 - 2 * x₂ - 8 = 0)
  (h_distinct : x₁ ≠ x₂) : 
  x₁ + x₂ = 2 := 
sorry

end sum_of_roots_l152_152274


namespace prob_6_higher_than_3_after_10_shuffles_l152_152748

def p_k (k : Nat) : ℚ := (3^k - 2^k) / (2 * 3^k)

theorem prob_6_higher_than_3_after_10_shuffles :
  p_k 10 = (3^10 - 2^10) / (2 * 3^10) :=
by
  sorry

end prob_6_higher_than_3_after_10_shuffles_l152_152748


namespace f_sub_f_inv_eq_2022_l152_152918

def f (n : ℕ) : ℕ := 2 * n
def f_inv (n : ℕ) : ℕ := n

theorem f_sub_f_inv_eq_2022 : f 2022 - f_inv 2022 = 2022 := by
  -- Proof goes here
  sorry

end f_sub_f_inv_eq_2022_l152_152918


namespace solve_ode_with_initial_condition_l152_152420

-- Define the function y(x)
def y (x : ℝ) : ℝ := Real.sin x + 1

-- Define the differential equation condition
def diff_eq (x : ℝ) : Prop := deriv y x = Real.cos x

-- Define the initial condition
def initial_condition : Prop := y 0 = 1

-- The statement of the theorem
theorem solve_ode_with_initial_condition : (∀ x, diff_eq x) ∧ initial_condition := by
  sorry

end solve_ode_with_initial_condition_l152_152420


namespace operations_correctness_l152_152641

theorem operations_correctness (a b : ℝ) : 
  ((-ab)^2 ≠ -a^2 * b^2)
  ∧ (a^3 * a^2 ≠ a^6)
  ∧ ((a^3)^4 ≠ a^7)
  ∧ (b^2 + b^2 = 2 * b^2) :=
by
  sorry

end operations_correctness_l152_152641


namespace gcd_lcm_product_24_36_l152_152553

-- Definitions for gcd, lcm, and product for given numbers, skipping proof with sorry
theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  -- Sorry used to skip proof
  sorry

end gcd_lcm_product_24_36_l152_152553


namespace probability_no_rain_five_days_probability_drought_alert_approx_l152_152487

theorem probability_no_rain_five_days (p : ℚ) (h : p = 1/3) :
  (p ^ 5) = 1 / 243 :=
by
  -- Add assumptions and proceed
  sorry

theorem probability_drought_alert_approx (p : ℚ) (h : p = 1/3) :
  4 * (p ^ 2) = 4 / 9 :=
by
  -- Add assumptions and proceed
  sorry

end probability_no_rain_five_days_probability_drought_alert_approx_l152_152487


namespace birdhouse_flight_distance_l152_152628

variable (car_distance : ℕ)
variable (lawn_chair_distance : ℕ)
variable (birdhouse_distance : ℕ)

def problem_condition1 := car_distance = 200
def problem_condition2 := lawn_chair_distance = 2 * car_distance
def problem_condition3 := birdhouse_distance = 3 * lawn_chair_distance

theorem birdhouse_flight_distance
  (h1 : car_distance = 200)
  (h2 : lawn_chair_distance = 2 * car_distance)
  (h3 : birdhouse_distance = 3 * lawn_chair_distance) :
  birdhouse_distance = 1200 := by
  sorry

end birdhouse_flight_distance_l152_152628


namespace cubic_sum_l152_152316

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l152_152316


namespace more_oil_l152_152657

noncomputable def original_price (P : ℝ) :=
  P - 0.3 * P = 70

noncomputable def amount_of_oil_before (P : ℝ) :=
  700 / P

noncomputable def amount_of_oil_after :=
  700 / 70

theorem more_oil (P : ℝ) (h1 : original_price P) :
  (amount_of_oil_after - amount_of_oil_before P) = 3 :=
  sorry

end more_oil_l152_152657


namespace burger_cost_l152_152765

theorem burger_cost (b s : ℕ) (h1 : 3 * b + 2 * s = 385) (h2 : 2 * b + 3 * s = 360) : b = 87 :=
sorry

end burger_cost_l152_152765


namespace roque_commute_time_l152_152588

theorem roque_commute_time :
  let walk_time := 2
  let bike_time := 1
  let walks_per_week := 3
  let bike_rides_per_week := 2
  let total_walk_time := 2 * walks_per_week * walk_time
  let total_bike_time := 2 * bike_rides_per_week * bike_time
  total_walk_time + total_bike_time = 16 :=
by sorry

end roque_commute_time_l152_152588


namespace find_x_l152_152581

/-!
# Problem Statement
Given that the segment with endpoints (-8, 0) and (32, 0) is the diameter of a circle,
and the point (x, 20) lies on the circle, prove that x = 12.
-/

def point_on_circle (x y : ℝ) (center_x center_y radius : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = radius^2

theorem find_x : 
  let center_x := (32 + (-8)) / 2
  let center_y := (0 + 0) / 2
  let radius := (32 - (-8)) / 2
  ∃ x : ℝ, point_on_circle x 20 center_x center_y radius → x = 12 :=
by
  sorry

end find_x_l152_152581


namespace total_time_is_correct_l152_152746

-- Defining the number of items
def chairs : ℕ := 7
def tables : ℕ := 3
def bookshelves : ℕ := 2
def lamps : ℕ := 4

-- Defining the time spent on each type of furniture
def time_per_chair : ℕ := 4
def time_per_table : ℕ := 8
def time_per_bookshelf : ℕ := 12
def time_per_lamp : ℕ := 2

-- Defining the total time calculation
def total_time : ℕ :=
  (chairs * time_per_chair) + 
  (tables * time_per_table) +
  (bookshelves * time_per_bookshelf) +
  (lamps * time_per_lamp)

-- Theorem stating the total time
theorem total_time_is_correct : total_time = 84 :=
by
  -- Skipping the proof details
  sorry

end total_time_is_correct_l152_152746


namespace probability_at_least_one_white_ball_l152_152611

noncomputable def total_combinations : ℕ := (Nat.choose 5 3)
noncomputable def no_white_combinations : ℕ := (Nat.choose 3 3)
noncomputable def prob_no_white_balls : ℚ := no_white_combinations / total_combinations
noncomputable def prob_at_least_one_white_ball : ℚ := 1 - prob_no_white_balls

theorem probability_at_least_one_white_ball :
  prob_at_least_one_white_ball = 9 / 10 :=
by
  have h : total_combinations = 10 := by sorry
  have h1 : no_white_combinations = 1 := by sorry
  have h2 : prob_no_white_balls = 1 / 10 := by sorry
  have h3 : prob_at_least_one_white_ball = 1 - prob_no_white_balls := by sorry
  norm_num [prob_no_white_balls, prob_at_least_one_white_ball, h, h1, h2, h3]

end probability_at_least_one_white_ball_l152_152611


namespace us_more_than_canada_l152_152760

/-- Define the total number of supermarkets -/
def total_supermarkets : ℕ := 84

/-- Define the number of supermarkets in the US -/
def us_supermarkets : ℕ := 49

/-- Define the number of supermarkets in Canada -/
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

/-- The proof problem: Prove that there are 14 more supermarkets in the US than in Canada -/
theorem us_more_than_canada : us_supermarkets - canada_supermarkets = 14 := by
  sorry

end us_more_than_canada_l152_152760


namespace double_exceeds_one_fifth_by_nine_l152_152055

theorem double_exceeds_one_fifth_by_nine (x : ℝ) (h : 2 * x = (1 / 5) * x + 9) : x^2 = 25 :=
sorry

end double_exceeds_one_fifth_by_nine_l152_152055


namespace intersect_single_point_l152_152568

theorem intersect_single_point (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 4 * x + 2 = 0) ∧ ∀ x₁ x₂ : ℝ, 
  (m - 3) * x₁^2 - 4 * x₁ + 2 = 0 → (m - 3) * x₂^2 - 4 * x₂ + 2 = 0 → x₁ = x₂ ↔ m = 3 ∨ m = 5 := 
sorry

end intersect_single_point_l152_152568


namespace range_of_m_l152_152618

-- Define the function and its properties
variable {f : ℝ → ℝ}
variable (increasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 > f x2)

theorem range_of_m (h: ∀ m : ℝ, f (2 * m) > f (-m + 9)) : 
  ∀ m : ℝ, m > 3 ↔ f (2 * m) > f (-m + 9) :=
by
  intros
  sorry

end range_of_m_l152_152618


namespace total_weight_of_sections_l152_152033

theorem total_weight_of_sections :
  let doll_length := 5
  let doll_weight := 29 / 8
  let tree_length := 4
  let tree_weight := 2.8
  let section_length := 2
  let doll_weight_per_meter := doll_weight / doll_length
  let tree_weight_per_meter := tree_weight / tree_length
  let doll_section_weight := doll_weight_per_meter * section_length
  let tree_section_weight := tree_weight_per_meter * section_length
  doll_section_weight + tree_section_weight = 57 / 20 :=
sorry

end total_weight_of_sections_l152_152033


namespace triangle_inequality_l152_152712

variable {x y z : ℝ}
variable {A B C : ℝ}

theorem triangle_inequality (hA: A > 0) (hB : B > 0) (hC : C > 0) (h_sum : A + B + C = π):
  x^2 + y^2 + z^2 ≥ 2 * y * z * Real.sin A + 2 * z * x * Real.sin B - 2 * x * y * Real.cos C := by
  sorry

end triangle_inequality_l152_152712


namespace cannot_be_expressed_as_x_squared_plus_y_fifth_l152_152929

theorem cannot_be_expressed_as_x_squared_plus_y_fifth :
  ¬ ∃ x y : ℤ, 59121 = x^2 + y^5 :=
by sorry

end cannot_be_expressed_as_x_squared_plus_y_fifth_l152_152929


namespace dealer_overall_gain_l152_152652

noncomputable def dealer_gain_percentage (weight1 weight2 : ℕ) (cost_price : ℕ) : ℚ :=
  let actual_weight_sold := weight1 + weight2
  let supposed_weight_sold := 1000 + 1000
  let gain_item1 := cost_price - (weight1 / 1000) * cost_price
  let gain_item2 := cost_price - (weight2 / 1000) * cost_price
  let total_gain := gain_item1 + gain_item2
  let total_actual_cost := (actual_weight_sold / 1000) * cost_price
  (total_gain / total_actual_cost) * 100

theorem dealer_overall_gain :
  dealer_gain_percentage 900 850 100 = 14.29 := 
sorry

end dealer_overall_gain_l152_152652


namespace cubic_sum_l152_152312

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l152_152312


namespace smallest_number_mod_l152_152419

theorem smallest_number_mod (x : ℕ) :
  (x % 2 = 1) → (x % 3 = 2) → x = 5 :=
by
  sorry

end smallest_number_mod_l152_152419


namespace original_cost_of_meal_l152_152400

-- Definitions for conditions
def meal_cost (initial_cost : ℝ) : ℝ :=
  initial_cost + 0.085 * initial_cost + 0.18 * initial_cost

-- The theorem we aim to prove
theorem original_cost_of_meal (total_cost : ℝ) (h : total_cost = 35.70) :
  ∃ initial_cost : ℝ, initial_cost = 28.23 ∧ meal_cost initial_cost = total_cost :=
by
  use 28.23
  rw [meal_cost, h]
  sorry

end original_cost_of_meal_l152_152400


namespace solve_equation_l152_152360

-- Defining the equation and the conditions for positive integers
def equation (x y z v : ℕ) : Prop :=
  x + 1 / (y + 1 / (z + 1 / (v : ℚ))) = 101 / 91

-- Stating the problem in Lean
theorem solve_equation :
  ∃ x y z v : ℕ, equation x y z v ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ v > 0 ∧
  x = 1 ∧ y = 9 ∧ z = 9 ∧ v = 1 :=
by
  sorry

end solve_equation_l152_152360


namespace second_tap_emptying_time_l152_152780

theorem second_tap_emptying_time :
  ∀ (T : ℝ), (∀ (f e : ℝ),
  (f = 1 / 3) →
  (∀ (n : ℝ), (n = 1 / 4.5) →
  (n = f - e ↔ e = 1 / T))) →
  T = 9 :=
by
  sorry

end second_tap_emptying_time_l152_152780


namespace ron_spends_on_chocolate_bars_l152_152935

/-- Ron is hosting a camp for 15 scouts where each scout needs 2 s'mores.
    Each chocolate bar costs $1.50 and can be broken into 3 sections to make 3 s'mores.
    A discount of 15% applies if 10 or more chocolate bars are purchased.
    Calculate the total amount Ron will spend on chocolate bars after applying the discount if applicable. -/
theorem ron_spends_on_chocolate_bars :
  let cost_per_bar := 1.5
  let s'mores_per_bar := 3
  let scouts := 15
  let s'mores_per_scout := 2
  let total_s'mores := scouts * s'mores_per_scout
  let bars_needed := total_s'mores / s'mores_per_bar
  let discount := 0.15
  let total_cost := bars_needed * cost_per_bar
  let discount_amount := if bars_needed >= 10 then discount * total_cost else 0
  let final_cost := total_cost - discount_amount
  final_cost = 12.75 := by sorry

end ron_spends_on_chocolate_bars_l152_152935


namespace range_of_expression_l152_152852

noncomputable def range_expression (a b : ℝ) : ℝ := a^2 + b^2 - 6 * a - 8 * b

variables (a b : ℝ)

def circle1 (a : ℝ) : Prop := ∀ x y : ℝ, (x - a)^2 + y^2 = 1

def circle2 (b : ℝ) : Prop := ∀ x y : ℝ, x^2 + y^2 - 2 * b * y + b^2 - 4 = 0

theorem range_of_expression :
  (circle1 a ∧ circle2 b ∧ ∃ (a b : ℝ), sqrt (a^2 + b^2) = 3) → -21 ≤ range_expression a b ∧ range_expression a b ≤ 39 := 
sorry

end range_of_expression_l152_152852


namespace cone_generatrix_length_theorem_l152_152441

noncomputable def cone_generatrix_length 
  (diameter : ℝ) 
  (unfolded_side_area : ℝ) 
  (h_diameter : diameter = 6)
  (h_area : unfolded_side_area = 18 * Real.pi) : 
  ℝ :=
6

theorem cone_generatrix_length_theorem 
  (diameter : ℝ) 
  (unfolded_side_area : ℝ) 
  (h_diameter : diameter = 6)
  (h_area : unfolded_side_area = 18 * Real.pi) :
  cone_generatrix_length diameter unfolded_side_area h_diameter h_area = 6 :=
sorry

end cone_generatrix_length_theorem_l152_152441


namespace problem1_problem2_problem3_problem4_l152_152408

theorem problem1 : (-4.7 : ℝ) + 0.9 = -3.8 := by
  sorry

theorem problem2 : (- (1 / 2) : ℝ) - (-(1 / 3)) = -(1 / 6) := by
  sorry

theorem problem3 : (- (10 / 9) : ℝ) * (- (6 / 10)) = (2 / 3) := by
  sorry

theorem problem4 : (0 : ℝ) * (-5) = 0 := by
  sorry

end problem1_problem2_problem3_problem4_l152_152408


namespace max_correct_answers_l152_152395

theorem max_correct_answers (c w b : ℕ) (h1 : c + w + b = 30) (h2 : 4 * c - w = 85) : c ≤ 23 :=
  sorry

end max_correct_answers_l152_152395


namespace square_roots_of_four_ninths_cube_root_of_neg_sixty_four_l152_152214

theorem square_roots_of_four_ninths : {x : ℚ | x ^ 2 = 4 / 9} = {2 / 3, -2 / 3} :=
by
  sorry

theorem cube_root_of_neg_sixty_four : {y : ℚ | y ^ 3 = -64} = {-4} :=
by
  sorry

end square_roots_of_four_ninths_cube_root_of_neg_sixty_four_l152_152214


namespace relationship_between_y_values_l152_152366

-- Define the quadratic function given the constraints
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + abs b * x + c

-- Define the points (x1, y1), (x2, y2), (x3, y3)
def x1 := -14 / 3
def x2 := 5 / 2
def x3 := 3

def y1 (a b c : ℝ) : ℝ := quadratic_function a b c x1
def y2 (a b c : ℝ) : ℝ := quadratic_function a b c x2
def y3 (a b c : ℝ) : ℝ := quadratic_function a b c x3

theorem relationship_between_y_values 
  (a b c : ℝ) (h1 : - (abs b) / (2 * a) = -1) 
  (y1_value : ℝ := y1 a b c) 
  (y2_value : ℝ := y2 a b c) 
  (y3_value : ℝ := y3 a b c) : 
  y2_value < y1_value ∧ y1_value < y3_value := 
by 
  sorry

end relationship_between_y_values_l152_152366


namespace least_power_divisible_by_240_l152_152439

theorem least_power_divisible_by_240 (n : ℕ) (a : ℕ) (h_a : a = 60) (h : a^n % 240 = 0) : 
  n = 2 :=
by
  sorry

end least_power_divisible_by_240_l152_152439


namespace canteen_distance_l152_152777

-- Given definitions
def G_to_road : ℝ := 450
def G_to_B : ℝ := 700

-- Proof statement
theorem canteen_distance :
  ∃ x : ℝ, (x ≠ 0) ∧ 
           (G_to_road^2 + (x - G_to_road)^2 = x^2) ∧ 
           (x = 538) := 
by {
  sorry
}

end canteen_distance_l152_152777


namespace decreasing_interval_of_logarithm_derived_function_l152_152401

theorem decreasing_interval_of_logarithm_derived_function :
  ∀ (x : ℝ), 1 < x → ∃ (f : ℝ → ℝ), (f x = x / (x - 1)) ∧ (∀ (h : x ≠ 1), deriv f x < 0) :=
by
  sorry

end decreasing_interval_of_logarithm_derived_function_l152_152401


namespace closest_integer_to_10_minus_sqrt_12_l152_152946

theorem closest_integer_to_10_minus_sqrt_12 (a b c d : ℤ) (h_a : a = 4) (h_b : b = 5) (h_c : c = 6) (h_d : d = 7) :
  d = 7 :=
by
  sorry

end closest_integer_to_10_minus_sqrt_12_l152_152946


namespace least_of_10_consecutive_odd_integers_average_154_l152_152162

theorem least_of_10_consecutive_odd_integers_average_154 (x : ℤ)
  (h_avg : (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) + (x + 16) + (x + 18)) / 10 = 154) :
  x = 145 :=
by 
  sorry

end least_of_10_consecutive_odd_integers_average_154_l152_152162


namespace completing_square_solution_l152_152749

theorem completing_square_solution (x : ℝ) :
  2 * x^2 + 4 * x - 3 = 0 →
  (x + 1)^2 = 5 / 2 :=
by
  sorry

end completing_square_solution_l152_152749


namespace no_more_than_four_intersection_points_l152_152014

noncomputable def conic1 (a b c d e f : ℝ) (x y : ℝ) : Prop := 
  a * x^2 + 2 * b * x * y + c * y^2 + 2 * d * x + 2 * e * y = f

noncomputable def conic2_param (P Q A : ℝ → ℝ) (t : ℝ) : ℝ × ℝ :=
  (P t / A t, Q t / A t)

theorem no_more_than_four_intersection_points (a b c d e f : ℝ)
  (P Q A : ℝ → ℝ) :
  (∃ t1 t2 t3 t4 t5,
    conic1 a b c d e f (P t1 / A t1) (Q t1 / A t1) ∧
    conic1 a b c d e f (P t2 / A t2) (Q t2 / A t2) ∧
    conic1 a b c d e f (P t3 / A t3) (Q t3 / A t3) ∧
    conic1 a b c d e f (P t4 / A t4) (Q t4 / A t4) ∧
    conic1 a b c d e f (P t5 / A t5) (Q t5 / A t5)) → false :=
sorry

end no_more_than_four_intersection_points_l152_152014


namespace anya_wins_19_games_l152_152801

theorem anya_wins_19_games (total_rounds : ℕ)
                           (anya_rock anya_scissors anya_paper borya_rock borya_scissors borya_paper : ℕ)
                           (no_draws : total_rounds = 25)
                           (anya_choices : anya_rock = 12 ∧ anya_scissors = 6 ∧ anya_paper = 7)
                           (borya_choices : borya_rock = 13 ∧ borya_scissors = 9 ∧ borya_paper = 3) 
                           : ∃ (anya_wins : ℕ), anya_wins = 19 := 
by
  have anya_rock_wins := min anya_rock borya_scissors  -- Rock wins against Scissors
  have anya_scissors_wins := min anya_scissors borya_paper  -- Scissors win against Paper
  have anya_paper_wins := min anya_paper borya_rock  -- Paper wins against Rock
  let total_wins := anya_rock_wins + anya_scissors_wins + anya_paper_wins
  have : total_wins = 19 := by 
    rw [anya_choices, borya_choices]
    simp
    done sorry
  exact ⟨total_wins⟩


end anya_wins_19_games_l152_152801


namespace remaining_money_l152_152525

theorem remaining_money (m : ℝ) (c f t r : ℝ)
  (h_initial : m = 1500)
  (h_clothes : c = (1 / 3) * m)
  (h_food : f = (1 / 5) * (m - c))
  (h_travel : t = (1 / 4) * (m - c - f))
  (h_remaining : r = m - c - f - t) :
  r = 600 := 
by
  sorry

end remaining_money_l152_152525


namespace selection_of_books_l152_152579

-- Define the problem context and the proof statement
theorem selection_of_books (n k : ℕ) (h_n : n = 10) (h_k : k = 5) : nat.choose n k = 252 := by
  -- Given: n = 10, k = 5
  -- Prove: (10 choose 5) = 252
  rw [h_n, h_k]
  norm_num
  sorry

end selection_of_books_l152_152579


namespace complex_magnitude_l152_152678

theorem complex_magnitude (z : ℂ) (i_unit : ℂ := Complex.I) 
  (h : (z - i_unit) * i_unit = 2 + i_unit) : Complex.abs z = Real.sqrt 5 := 
by
  sorry

end complex_magnitude_l152_152678


namespace complement_of_A_l152_152851

open Set

-- Define the universal set U
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := { x | abs (x - 1) > 1 }

-- Define the problem statement
theorem complement_of_A :
  ∀ x : ℝ, x ∈ compl A ↔ x ∈ Icc 0 2 :=
by
  intro x
  rw [mem_compl_iff, mem_Icc]
  sorry

end complement_of_A_l152_152851


namespace root_in_interval_l152_152084

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.log x + x - (1 / x) - 2
noncomputable def f' (x : ℝ) : ℝ := (1 / (2 * x)) + 1 + (1 / (x ^ 2))

theorem root_in_interval : ∃ x ∈ Set.Ioo 2 Real.exp 1, f x = 0 :=
by {
  have mono : ∀ x > 0, f' x > 0,
  { intro x_pos, sorry },
  have f_at_2 : f 2 < 0, by { sorry },
  have f_at_e : f (Real.exp 1) > 0, by { sorry },
  rwa [← Set.mem_Ioo, ← exists_mem] at this,
  apply Intermediate_Value_Theorem,
  exact ⟨f_at_2, f_at_e, ⟨2_pos, Real.exp_pos⟩, mono⟩,
  apply_instance,
  use x,
  exact ⟨2_lt_e, e_lt_exp, h, hx⟩,
}

end root_in_interval_l152_152084


namespace maximum_M_value_l152_152600

noncomputable def max_value_of_M : ℝ :=
  Real.sqrt 2 + 1 

theorem maximum_M_value {x y z : ℝ} (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) :
  Real.sqrt (abs (x - y)) + Real.sqrt (abs (y - z)) + Real.sqrt (abs (z - x)) ≤ max_value_of_M :=
by
  sorry

end maximum_M_value_l152_152600


namespace time_to_pass_tree_l152_152397

-- Define the conditions given in the problem
def train_length : ℕ := 1200
def platform_length : ℕ := 700
def time_to_pass_platform : ℕ := 190

-- Calculate the total distance covered while passing the platform
def distance_passed_platform : ℕ := train_length + platform_length

-- The main theorem we need to prove
theorem time_to_pass_tree : (distance_passed_platform / time_to_pass_platform) * train_length = 120 := 
by
  sorry

end time_to_pass_tree_l152_152397


namespace total_marbles_is_260_l152_152414

-- Define the number of marbles in each jar based on the conditions.
def first_jar : Nat := 80
def second_jar : Nat := 2 * first_jar
def third_jar : Nat := (1 / 4 : ℚ) * first_jar

-- Prove that the total number of marbles is 260.
theorem total_marbles_is_260 : first_jar + second_jar + third_jar = 260 := by
  sorry

end total_marbles_is_260_l152_152414


namespace probability_of_interval_is_one_third_l152_152904

noncomputable def probability_in_interval (total_start total_end inner_start inner_end : ℝ) : ℝ :=
  (inner_end - inner_start) / (total_end - total_start)

theorem probability_of_interval_is_one_third :
  probability_in_interval 1 7 5 8 = 1 / 3 :=
by
  sorry

end probability_of_interval_is_one_third_l152_152904


namespace find_x_l152_152649

theorem find_x (x : ℝ) (h : 0.65 * x = 0.20 * 552.50) : x = 170 :=
by
  sorry

end find_x_l152_152649


namespace jenna_owes_amount_l152_152720

theorem jenna_owes_amount (initial_bill : ℝ) (rate : ℝ) (times : ℕ) : 
  initial_bill = 400 → rate = 0.02 → times = 3 → 
  owed_amount = (400 * (1 + 0.02)^3) := 
by
  intros
  sorry

end jenna_owes_amount_l152_152720


namespace sqrt_mult_pow_l152_152806

theorem sqrt_mult_pow (a : ℝ) (h_nonneg : 0 ≤ a) : (a^(2/3) * a^(1/5)) = a^(13/15) := by
  sorry

end sqrt_mult_pow_l152_152806


namespace total_time_spent_l152_152719

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2
def total_time : ℕ := outlining_time + writing_time + practicing_time

theorem total_time_spent : total_time = 117 := by
  sorry

end total_time_spent_l152_152719


namespace find_numbers_l152_152556

theorem find_numbers (x y : ℕ) (h1 : x / y = 3) (h2 : (x^2 + y^2) / (x + y) = 5) : 
  x = 6 ∧ y = 2 := 
by
  sorry

end find_numbers_l152_152556


namespace number_of_perfect_square_divisors_of_450_l152_152148

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l152_152148


namespace percentage_non_defective_l152_152449

theorem percentage_non_defective :
  let total_units : ℝ := 100
  let M1_percentage : ℝ := 0.20
  let M2_percentage : ℝ := 0.25
  let M3_percentage : ℝ := 0.30
  let M4_percentage : ℝ := 0.15
  let M5_percentage : ℝ := 0.10
  let M1_defective_percentage : ℝ := 0.02
  let M2_defective_percentage : ℝ := 0.04
  let M3_defective_percentage : ℝ := 0.05
  let M4_defective_percentage : ℝ := 0.07
  let M5_defective_percentage : ℝ := 0.08

  let M1_total := total_units * M1_percentage
  let M2_total := total_units * M2_percentage
  let M3_total := total_units * M3_percentage
  let M4_total := total_units * M4_percentage
  let M5_total := total_units * M5_percentage

  let M1_defective := M1_total * M1_defective_percentage
  let M2_defective := M2_total * M2_defective_percentage
  let M3_defective := M3_total * M3_defective_percentage
  let M4_defective := M4_total * M4_defective_percentage
  let M5_defective := M5_total * M5_defective_percentage

  let total_defective := M1_defective + M2_defective + M3_defective + M4_defective + M5_defective
  let total_non_defective := total_units - total_defective
  let percentage_non_defective := (total_non_defective / total_units) * 100

  percentage_non_defective = 95.25 := by
  sorry

end percentage_non_defective_l152_152449


namespace over_limit_weight_l152_152454

variable (hc_books : ℕ) (hc_weight : ℕ → ℝ)
variable (tex_books : ℕ) (tex_weight : ℕ → ℝ)
variable (knick_knacks : ℕ) (knick_weight : ℕ → ℝ)
variable (weight_limit : ℝ)

axiom hc_books_value : hc_books = 70
axiom hc_weight_value : hc_weight hc_books = 0.5
axiom tex_books_value : tex_books = 30
axiom tex_weight_value : tex_weight tex_books = 2
axiom knick_knacks_value : knick_knacks = 3
axiom knick_weight_value : knick_weight knick_knacks = 6
axiom weight_limit_value : weight_limit = 80

theorem over_limit_weight : 
  (hc_books * hc_weight hc_books + tex_books * tex_weight tex_books + knick_knacks * knick_weight knick_knacks) - weight_limit = 33 := by
  sorry

end over_limit_weight_l152_152454


namespace profit_after_discount_l152_152778

noncomputable def purchase_price : ℝ := 100
noncomputable def increase_rate : ℝ := 0.25
noncomputable def discount_rate : ℝ := 0.10

theorem profit_after_discount :
  let selling_price := purchase_price * (1 + increase_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let profit := discounted_price - purchase_price
  profit = 12.5 :=
by
  sorry 

end profit_after_discount_l152_152778


namespace positive_difference_of_R_coords_l152_152035

theorem positive_difference_of_R_coords :
    ∀ (xR yR : ℝ),
    ∃ (k : ℝ),
    (∀ (A B C R S : ℝ × ℝ), 
    A = (-1, 6) ∧ B = (1, 2) ∧ C = (7, 2) ∧ 
    R = (k, -0.5 * k + 5.5) ∧ S = (k, 2) ∧
    (0.5 * |7 - k| * |0.5 * k - 3.5| = 8)) → 
    |xR - yR| = 1 :=
by
  sorry

end positive_difference_of_R_coords_l152_152035


namespace range_of_m_l152_152103

open Real

noncomputable def x (y : ℝ) : ℝ := 2 / (1 - 1 / y)

theorem range_of_m (y : ℝ) (m : ℝ) (h1 : y > 0) (h2 : 1 - 1 / y > 0) (h3 : -4 < m) (h4 : m < 2) : 
  x y + 2 * y > m^2 + 2 * m := 
by 
  have hx_pos : x y > 0 := sorry
  have hxy_eq : 2 / x y + 1 / y = 1 := sorry
  have hxy_ge : x y + 2 * y ≥ 8 := sorry
  have h_m_le : 8 > m^2 + 2 * m := sorry
  exact sorry

end range_of_m_l152_152103


namespace handshakes_count_l152_152402

-- Define the parameters
def teams : ℕ := 3
def players_per_team : ℕ := 7
def referees : ℕ := 3

-- Calculate handshakes among team members
def handshakes_among_teams :=
  let unique_handshakes_per_team := players_per_team * 2 * players_per_team / 2
  unique_handshakes_per_team * teams

-- Calculate handshakes between players and referees
def players_shake_hands_with_referees :=
  teams * players_per_team * referees

-- Calculate total handshakes
def total_handshakes :=
  handshakes_among_teams + players_shake_hands_with_referees

-- Proof statement
theorem handshakes_count : total_handshakes = 210 := by
  sorry

end handshakes_count_l152_152402


namespace train_distance_proof_l152_152194

theorem train_distance_proof (c₁ c₂ c₃ : ℝ) : 
  (5 / c₁ + 5 / c₂ = 15) →
  (5 / c₂ + 5 / c₃ = 11) →
  ∀ (x : ℝ), (x / c₁ = 10 / c₂ + (10 + x) / c₃) →
  x = 27.5 := 
by
  sorry

end train_distance_proof_l152_152194


namespace perfect_square_factors_450_l152_152153

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l152_152153


namespace sum_of_diagonals_l152_152594

-- Definitions of the given lengths
def AB := 5
def CD := 5
def BC := 12
def DE := 12
def AE := 18

-- Variables for the diagonal lengths
variables (AC BD CE : ℚ)

-- The Lean 4 theorem statement
theorem sum_of_diagonals (hAC : AC = 723 / 44) (hBD : BD = 44 / 3) (hCE : CE = 351 / 22) :
  AC + BD + CE = 6211 / 132 :=
by
  sorry

end sum_of_diagonals_l152_152594


namespace min_value_of_alpha_beta_l152_152352

theorem min_value_of_alpha_beta 
  (k : ℝ)
  (h_k : k ≤ -4 ∨ k ≥ 5)
  (α β : ℝ)
  (h_αβ : α^2 - 2 * k * α + (k + 20) = 0 ∧ β^2 - 2 * k * β + (k + 20) = 0) :
  (α + 1) ^ 2 + (β + 1) ^ 2 = 18 → k = -4 :=
sorry

end min_value_of_alpha_beta_l152_152352


namespace kyler_wins_zero_l152_152448

-- Definitions based on conditions provided
def peter_games_won : ℕ := 5
def peter_games_lost : ℕ := 3
def emma_games_won : ℕ := 4
def emma_games_lost : ℕ := 4
def kyler_games_lost : ℕ := 4

-- Number of games each player played
def peter_total_games : ℕ := peter_games_won + peter_games_lost
def emma_total_games : ℕ := emma_games_won + emma_games_lost
def kyler_total_games (k : ℕ) : ℕ := k + kyler_games_lost

-- Step 1: total number of games in the tournament
def total_games (k : ℕ) : ℕ := (peter_total_games + emma_total_games + kyler_total_games k) / 2

-- Step 2: Total games equation
def games_equation (k : ℕ) : Prop := 
  (peter_games_won + emma_games_won + k = total_games k)

-- The proof problem, we need to prove Kyler's wins
theorem kyler_wins_zero : games_equation 0 := by
  -- proof omitted
  sorry

end kyler_wins_zero_l152_152448


namespace geom_seq_property_l152_152836

noncomputable def a_n : ℕ → ℝ := sorry  -- The definition of the geometric sequence

theorem geom_seq_property (a_n : ℕ → ℝ) (h : a_n 6 + a_n 8 = 4) :
  a_n 8 * (a_n 4 + 2 * a_n 6 + a_n 8) = 16 := by
sorry

end geom_seq_property_l152_152836


namespace garden_perimeter_is_64_l152_152234

-- Define the playground dimensions and its area 
def playground_length := 16
def playground_width := 12
def playground_area := playground_length * playground_width

-- Define the garden width and its area being the same as the playground's area
def garden_width := 8
def garden_area := playground_area

-- Calculate the garden's length
def garden_length := garden_area / garden_width

-- Calculate the perimeter of the garden
def garden_perimeter := 2 * (garden_length + garden_width)

theorem garden_perimeter_is_64 :
  garden_perimeter = 64 := 
sorry

end garden_perimeter_is_64_l152_152234


namespace total_weight_correct_l152_152602

def weight_male_clothes : ℝ := 2.6
def weight_female_clothes : ℝ := 5.98
def total_weight_clothes : ℝ := weight_male_clothes + weight_female_clothes

theorem total_weight_correct : total_weight_clothes = 8.58 := by
  sorry

end total_weight_correct_l152_152602


namespace rectangle_tileable_iff_divisible_l152_152604

def divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

def tileable_with_0b_tiles (m n b : ℕ) : Prop :=
  ∃ t : ℕ, t * (2 * b) = m * n  -- This comes from the total area divided by the area of one tile

theorem rectangle_tileable_iff_divisible (m n b : ℕ) :
  tileable_with_0b_tiles m n b ↔ divisible_by (2 * b) m ∨ divisible_by (2 * b) n := 
sorry

end rectangle_tileable_iff_divisible_l152_152604


namespace remainder_when_divided_by_DE_l152_152347

theorem remainder_when_divided_by_DE (P D E Q R M S C : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = E * M + S) :
  (∃ quotient : ℕ, P = quotient * (D * E) + (S * D + R + C)) :=
by {
  sorry
}

end remainder_when_divided_by_DE_l152_152347


namespace find_special_5_digit_number_l152_152112

theorem find_special_5_digit_number :
  ∃! (A : ℤ), (10000 ≤ A ∧ A < 100000) ∧ (A^2 % 100000 = A) ∧ A = 90625 :=
sorry

end find_special_5_digit_number_l152_152112


namespace commodity_price_l152_152208

-- Define the variables for the prices of the commodities.
variable (x y : ℝ)

-- Conditions given in the problem.
def total_cost (x y : ℝ) : Prop := x + y = 827
def price_difference (x y : ℝ) : Prop := x = y + 127

-- The main statement to prove.
theorem commodity_price (x y : ℝ) (h1 : total_cost x y) (h2 : price_difference x y) : x = 477 :=
by
  sorry

end commodity_price_l152_152208


namespace votes_cast_l152_152930

theorem votes_cast (V : ℝ) (h1 : 0.35 * V + 2250 = 0.65 * V) : V = 7500 := 
by
  sorry

end votes_cast_l152_152930


namespace number_of_committees_correct_l152_152392

noncomputable def number_of_committees (teams members host_selection non_host_selection : ℕ) : ℕ :=
  have ways_to_choose_host := teams
  have ways_to_choose_four_from_seven := Nat.choose members host_selection
  have ways_to_choose_two_from_seven := Nat.choose members non_host_selection
  have total_non_host_combinations := ways_to_choose_two_from_seven ^ (teams - 1)
  ways_to_choose_host * ways_to_choose_four_from_seven * total_non_host_combinations

theorem number_of_committees_correct :
  number_of_committees 5 7 4 2 = 34134175 := by
  sorry

end number_of_committees_correct_l152_152392


namespace number_of_perfect_square_factors_l152_152139

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l152_152139


namespace arithmetic_sequence_common_difference_l152_152280

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) (h_arith_seq: ∀ n, a n = a 1 + (n - 1) * d) 
  (h_cond1 : a 3 + a 9 = 4 * a 5) (h_cond2 : a 2 = -8) : 
  d = 4 :=
sorry

end arithmetic_sequence_common_difference_l152_152280


namespace prod_gcd_lcm_eq_864_l152_152549

theorem prod_gcd_lcm_eq_864 : 
  let a := 24
  let b := 36
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  gcd_ab * lcm_ab = 864 :=
by
  sorry

end prod_gcd_lcm_eq_864_l152_152549


namespace ab_sum_l152_152329

theorem ab_sum (a b : ℝ) (h₁ : ∀ x : ℝ, (x + a) * (x + 8) = x^2 + b * x + 24) (h₂ : 8 * a = 24) : a + b = 14 :=
by
  sorry

end ab_sum_l152_152329


namespace problem_solution_set_l152_152370

open Nat

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)
def permutation (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem problem_solution_set :
  {n : ℕ | 2 * combination n 3 ≤ permutation n 2} = {n | n = 3 ∨ n = 4 ∨ n = 5} :=
by
  sorry

end problem_solution_set_l152_152370


namespace relationship_between_y_values_l152_152368

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + |b| * x + c

theorem relationship_between_y_values
  (a b c y1 y2 y3 : ℝ)
  (h1 : quadratic_function a b c (-14 / 3) = y1)
  (h2 : quadratic_function a b c (5 / 2) = y2)
  (h3 : quadratic_function a b c 3 = y3)
  (axis_symmetry : -(|b| / (2 * a)) = -1)
  (h_pos : 0 < a) :
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_between_y_values_l152_152368


namespace cubic_sum_l152_152313

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l152_152313


namespace projection_of_point_onto_xOy_plane_l152_152584

def point := (ℝ × ℝ × ℝ)

def projection_onto_xOy_plane (P : point) : point :=
  let (x, y, z) := P
  (x, y, 0)

theorem projection_of_point_onto_xOy_plane : 
  projection_onto_xOy_plane (2, 3, 4) = (2, 3, 0) :=
by
  -- proof steps would go here
  sorry

end projection_of_point_onto_xOy_plane_l152_152584


namespace gold_hammer_weight_l152_152709

theorem gold_hammer_weight (a : ℕ → ℕ) 
  (h_arith_seq : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h_a1 : a 1 = 4) 
  (h_a5 : a 5 = 2) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := 
sorry

end gold_hammer_weight_l152_152709


namespace journeymen_percentage_after_layoff_l152_152466

noncomputable def total_employees : ℝ := 20210
noncomputable def fraction_journeymen : ℝ := 2 / 7
noncomputable def total_journeymen : ℝ := total_employees * fraction_journeymen
noncomputable def laid_off_journeymen : ℝ := total_journeymen / 2
noncomputable def remaining_journeymen : ℝ := total_journeymen / 2
noncomputable def remaining_employees : ℝ := total_employees - laid_off_journeymen
noncomputable def journeymen_percentage : ℝ := (remaining_journeymen / remaining_employees) * 100

theorem journeymen_percentage_after_layoff : journeymen_percentage = 16.62 := by
  sorry

end journeymen_percentage_after_layoff_l152_152466


namespace cube_identity_l152_152303

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l152_152303


namespace problem_proof_l152_152854

variables {m n : ℝ}

-- Line definitions
def l1 (m n x y : ℝ) : Prop := m * x + 8 * y + n = 0
def l2 (m x y : ℝ) : Prop := 2 * x + m * y - 1 = 0

-- Conditions
def intersects_at (m n : ℝ) : Prop :=
  l1 m n m (-1) ∧ l2 m m (-1)

def parallel (m n : ℝ) : Prop :=
  (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2)

def perpendicular (m n : ℝ) : Prop :=
  m = 0 ∧ n = 8

theorem problem_proof :
  intersects_at m n → (m = 1 ∧ n = 7) ∧
  parallel m n → (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2) ∧
  perpendicular m n → (m = 0 ∧ n = 8) :=
by
  sorry

end problem_proof_l152_152854


namespace simplify_fraction_l152_152257

theorem simplify_fraction (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c ≠ 0) :
  (a^2 + a * b - b^2 + a * c) / (b^2 + b * c - c^2 + b * a) = (a - b) / (b - c) :=
by
  sorry

end simplify_fraction_l152_152257


namespace cos_arcsin_l152_152955

theorem cos_arcsin (h : (7:ℝ) / 25 ≤ 1) : Real.cos (Real.arcsin ((7:ℝ) / 25)) = (24:ℝ) / 25 := by
  -- Proof to be provided
  sorry

end cos_arcsin_l152_152955


namespace gcd_lcm_product_l152_152551

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [h₁, h₂]
  unfold Nat.gcd Nat.lcm
  sorry

end gcd_lcm_product_l152_152551


namespace distance_to_cut_pyramid_l152_152630

theorem distance_to_cut_pyramid (V A V1 : ℝ) (h1 : V > 0) (h2 : A > 0) :
  ∃ d : ℝ, d = (3 / A) * (V - (V^2 * (V - V1))^(1 / 3)) :=
by
  sorry

end distance_to_cut_pyramid_l152_152630


namespace find_price_of_each_part_l152_152948

def original_price (total_cost : ℝ) (num_parts : ℕ) (price_per_part : ℝ) :=
  num_parts * price_per_part = total_cost

theorem find_price_of_each_part :
  original_price 439 7 62.71 :=
by
  sorry

end find_price_of_each_part_l152_152948


namespace solve_for_n_l152_152201

theorem solve_for_n (n : ℕ) (h : 2^n * 8^n = 64^(n - 30)) : n = 90 :=
by {
  sorry
}

end solve_for_n_l152_152201


namespace walking_time_difference_at_slower_speed_l152_152040

theorem walking_time_difference_at_slower_speed (T : ℕ) (v_s: ℚ) (h1: T = 32) (h2: v_s = 4/5) : 
  (T * (5/4) - T) = 8 :=
by
  sorry

end walking_time_difference_at_slower_speed_l152_152040


namespace problem_sufficient_necessary_condition_l152_152159

open Set

variable {x : ℝ}

def P (x : ℝ) : Prop := abs (x - 2) < 3
def Q (x : ℝ) : Prop := x^2 - 8 * x + 15 < 0

theorem problem_sufficient_necessary_condition :
    (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬ Q x) :=
by
  sorry

end problem_sufficient_necessary_condition_l152_152159


namespace proof_problem_l152_152422

-- Define the operation
def star (a b : ℝ) : ℝ := (a - b) ^ 2

-- The proof problem as a Lean statement
theorem proof_problem (x y : ℝ) : star ((x - y) ^ 2 + 1) ((y - x) ^ 2 + 1) = 0 := by
  sorry

end proof_problem_l152_152422


namespace manufacturing_section_degrees_l152_152202

variable (percentage_manufacturing : ℝ) (total_degrees : ℝ)

theorem manufacturing_section_degrees
  (h1 : percentage_manufacturing = 0.40)
  (h2 : total_degrees = 360) :
  percentage_manufacturing * total_degrees = 144 := 
by 
  sorry

end manufacturing_section_degrees_l152_152202


namespace roden_total_fish_l152_152606

def total_goldfish : Nat :=
  15 + 10 + 3 + 4

def total_blue_fish : Nat :=
  7 + 12 + 7 + 8

def total_green_fish : Nat :=
  5 + 9 + 6

def total_purple_fish : Nat :=
  2

def total_red_fish : Nat :=
  1

def total_fish : Nat :=
  total_goldfish + total_blue_fish + total_green_fish + total_purple_fish + total_red_fish

theorem roden_total_fish : total_fish = 89 :=
by
  unfold total_fish total_goldfish total_blue_fish total_green_fish total_purple_fish total_red_fish
  sorry

end roden_total_fish_l152_152606


namespace initial_value_exists_l152_152766

theorem initial_value_exists (x : ℕ) (h : ∃ k : ℕ, x + 7 = k * 456) : x = 449 :=
sorry

end initial_value_exists_l152_152766


namespace tan_alpha_add_pi_div_four_l152_152835

theorem tan_alpha_add_pi_div_four {α : ℝ} (h1 : α ∈ Set.Ioo 0 (Real.pi)) (h2 : Real.cos α = -4/5) :
  Real.tan (α + Real.pi / 4) = 1 / 7 :=
sorry

end tan_alpha_add_pi_div_four_l152_152835


namespace bat_pattern_area_l152_152751

-- Define the areas of the individual components
def area_large_square : ℕ := 8
def num_large_squares : ℕ := 2

def area_medium_square : ℕ := 4
def num_medium_squares : ℕ := 2

def area_triangle : ℕ := 1
def num_triangles : ℕ := 3

-- Define the total area calculation
def total_area : ℕ :=
  (num_large_squares * area_large_square) +
  (num_medium_squares * area_medium_square) +
  (num_triangles * area_triangle)

-- The theorem statement
theorem bat_pattern_area : total_area = 27 := by
  sorry

end bat_pattern_area_l152_152751


namespace min_three_beverages_overlap_l152_152193

variable (a b c d : ℝ)
variable (ha : a = 0.9)
variable (hb : b = 0.8)
variable (hc : c = 0.7)

theorem min_three_beverages_overlap : d = 0.7 :=
by
  sorry

end min_three_beverages_overlap_l152_152193


namespace value_of_f_l152_152826

variable {x t : ℝ}

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0 ∨ x = 1 then 0
  else (1 : ℝ) / x

theorem value_of_f (h1 : ∀ x, x ≠ 0 → x ≠ 1 → f (x / (x - 1)) = 1 / x)
                   (h2 : 0 ≤ t ∧ t ≤ Real.pi / 2) :
  f (Real.tan t ^ 2 + 1) = Real.sin (2 * t) ^ 2 / 4 :=
sorry

end value_of_f_l152_152826


namespace perfect_square_divisors_count_450_l152_152119

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l152_152119


namespace length_PF_l152_152922

-- Define the parabola function
def parabola (x : ℝ) : ℝ := sqrt (8 * (x + 2))

-- Define the line with inclination angle 60 degrees passing through the focus
noncomputable def line_through_focus (x : ℝ) : ℝ := (real.sqrt 3) * x

-- Define the points of intersection A and B
noncomputable def intersection_points : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.snd = parabola p.fst ∧ p.snd = line_through_focus p.fst }

-- Define the midpoint of segment AB
def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.fst + b.fst) / 2, (a.snd + b.snd) / 2)

-- Define the perpendicular bisector of AB
def perpendicular_bisector (mid : ℝ × ℝ) : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.fst = mid.fst }

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.fst - p2.fst)^2 + (p1.snd - p2.snd)^2)

-- Establish that PF is equal to 16/3
theorem length_PF (A B : ℝ × ℝ)
  (hA : (A ∈ intersection_points)) (hB : (B ∈ intersection_points)) :
  let P := (midpoint A B).fst in
  distance (P, 0) (0, 0) = 16 / 3 :=
  sorry

end length_PF_l152_152922


namespace jeans_price_increase_l152_152654

theorem jeans_price_increase 
  (C : ℝ) 
  (R : ℝ) 
  (F : ℝ) 
  (H1 : R = 1.40 * C)
  (H2 : F = 1.82 * C) 
  : (F - C) / C * 100 = 82 := 
sorry

end jeans_price_increase_l152_152654


namespace union_eq_interval_l152_152601

def A := { x : ℝ | 1 < x ∧ x < 4 }
def B := { x : ℝ | (x - 3) * (x + 1) ≤ 0 }

theorem union_eq_interval : (A ∪ B) = { x : ℝ | -1 ≤ x ∧ x < 4 } :=
by
  sorry

end union_eq_interval_l152_152601


namespace find_common_difference_l152_152562

variable {a : ℕ → ℝ} (d : ℝ) (a₁ : ℝ)

-- defining the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a₁ : ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a n = a₁ + n * d

-- condition for the sum of even indexed terms
def sum_even_terms (a : ℕ → ℝ) : ℝ := a 2 + a 4 + a 6 + a 8 + a 10

-- condition for the sum of odd indexed terms
def sum_odd_terms (a : ℕ → ℝ) : ℝ := a 1 + a 3 + a 5 + a 7 + a 9

-- main theorem to prove
theorem find_common_difference
  (a : ℕ → ℝ) (a₁ : ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a a₁ d)
  (h_even_sum : sum_even_terms a = 30)
  (h_odd_sum : sum_odd_terms a = 25) :
  d = 1 := by
  sorry

end find_common_difference_l152_152562


namespace triangle_angle_l152_152342

theorem triangle_angle (A B C : ℝ) (h1 : A - C = B) (h2 : A + B + C = 180) : A = 90 :=
by
  sorry

end triangle_angle_l152_152342


namespace ratio_john_to_total_cost_l152_152177

noncomputable def cost_first_8_years := 8 * 10000
noncomputable def cost_next_10_years := 10 * 20000
noncomputable def university_tuition := 250000
noncomputable def cost_john_paid := 265000
noncomputable def total_cost := cost_first_8_years + cost_next_10_years + university_tuition

theorem ratio_john_to_total_cost : (cost_john_paid / total_cost : ℚ) = 1 / 2 := by
  sorry

end ratio_john_to_total_cost_l152_152177


namespace club_co_presidents_l152_152782

def choose (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem club_co_presidents : choose 18 3 = 816 := by
  sorry

end club_co_presidents_l152_152782


namespace chess_club_team_selection_l152_152486

theorem chess_club_team_selection :
  (Nat.choose 8 5) * (Nat.choose 10 3) = 6720 :=
by
  sorry

end chess_club_team_selection_l152_152486


namespace percentage_increase_correct_l152_152026

-- Define the highest and lowest scores as given conditions.
def highest_score : ℕ := 92
def lowest_score : ℕ := 65

-- State that the percentage increase calculation will result in 41.54%
theorem percentage_increase_correct :
  ((highest_score - lowest_score) * 100) / lowest_score = 4154 / 100 :=
by sorry

end percentage_increase_correct_l152_152026


namespace root_expression_l152_152407

theorem root_expression {p q x1 x2 : ℝ}
  (h1 : x1^2 + p * x1 + q = 0)
  (h2 : x2^2 + p * x2 + q = 0) :
  (x1 / x2 + x2 / x1) = (p^2 - 2 * q) / q :=
by {
  sorry
}

end root_expression_l152_152407


namespace range_of_expression_l152_152853

theorem range_of_expression (a b : ℝ) (h : a^2 + b^2 = 9) :
  -21 ≤ a^2 + b^2 - 6*a - 8*b ∧ a^2 + b^2 - 6*a - 8*b ≤ 39 :=
by
  sorry

end range_of_expression_l152_152853


namespace roots_quadratic_eq_identity1_roots_quadratic_eq_identity2_l152_152196

variables {α : Type*} [Field α] (a b c x1 x2 : α)

theorem roots_quadratic_eq_identity1 (h_eq_roots: ∀ x, a * x^2 + b * x + c = 0 → (x = x1 ∨ x = x2)) 
(h_root1: a * x1^2 + b * x1 + c = 0) (h_root2: a * x2^2 + b * x2 + c = 0) :
  x1^2 + x2^2 = (b^2 - 2 * a * c) / a^2 :=
sorry

theorem roots_quadratic_eq_identity2 (h_eq_roots: ∀ x, a * x^2 + b * x + c = 0 → (x = x1 ∨ x = x2)) 
(h_root1: a * x1^2 + b * x1 + c = 0) (h_root2: a * x2^2 + b * x2 + c = 0) :
  x1^3 + x2^3 = (3 * a * b * c - b^3) / a^3 :=
sorry

end roots_quadratic_eq_identity1_roots_quadratic_eq_identity2_l152_152196


namespace make_up_set_money_needed_l152_152831

theorem make_up_set_money_needed (makeup_cost gabby_money mom_money: ℤ) (h1: makeup_cost = 65) (h2: gabby_money = 35) (h3: mom_money = 20) :
  (makeup_cost - (gabby_money + mom_money)) = 10 :=
by {
  sorry
}

end make_up_set_money_needed_l152_152831


namespace recommendation_plans_l152_152582

theorem recommendation_plans (students universities : ℕ) (max_students : universities -> ℕ) 
  (h_students : students = 4) (h_universities : universities = 3) (h_max_students : ∀ (u : ℕ), u < universities → max_students u = 2) :
  let plans := (4.choose 2 * (3.choose 3).permute) + (3.choose 2 * 4.choose 2) in
  plans = 54 :=
by
  sorry

end recommendation_plans_l152_152582


namespace kamal_marks_in_english_l152_152461

theorem kamal_marks_in_english :
  ∀ (E Math Physics Chemistry Biology Average : ℕ), 
    Math = 65 → 
    Physics = 82 → 
    Chemistry = 67 → 
    Biology = 85 → 
    Average = 79 → 
    (Math + Physics + Chemistry + Biology + E) / 5 = Average → 
    E = 96 :=
by
  intros E Math Physics Chemistry Biology Average
  intros hMath hPhysics hChemistry hBiology hAverage hTotal
  sorry

end kamal_marks_in_english_l152_152461


namespace number_is_square_plus_opposite_l152_152787

theorem number_is_square_plus_opposite (x : ℝ) (hx : x = x^2 + -x) : x = 0 ∨ x = 2 :=
by sorry

end number_is_square_plus_opposite_l152_152787


namespace circle_properties_l152_152846

theorem circle_properties (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * m * x - 4 * y + 5 * m = 0) →
  (m < 1 ∨ m > 4) ∧
  (m = -2 → ∃ d : ℝ, d = 2 * Real.sqrt (18 - 5)) :=
by
  sorry

end circle_properties_l152_152846


namespace area_of_regular_inscribed_polygon_f3_properties_of_f_l152_152485

noncomputable def f (n : ℕ) : ℝ :=
  if h : n ≥ 3 then (n / 2) * Real.sin (2 * Real.pi / n) else 0

theorem area_of_regular_inscribed_polygon_f3 :
  f 3 = (3 * Real.sqrt 3) / 4 :=
by
  sorry

theorem properties_of_f (n : ℕ) (hn : n ≥ 3) :
  (f n = (n / 2) * Real.sin (2 * Real.pi / n)) ∧
  (f n < f (n + 1)) ∧ 
  (f n < f (2 * n) ∧ f (2 * n) ≤ 2 * f n) :=
by
  sorry

end area_of_regular_inscribed_polygon_f3_properties_of_f_l152_152485


namespace average_vegetables_per_week_l152_152635

theorem average_vegetables_per_week (P Vp S W : ℕ) (h1 : P = 200) (h2 : Vp = 2) (h3 : S = 25) (h4 : W = 2) :
  (P / Vp) / S / W = 2 :=
by
  sorry

end average_vegetables_per_week_l152_152635


namespace regression_estimate_l152_152972

theorem regression_estimate :
  ∀ (x y : ℝ), (y = 0.50 * x - 0.81) → x = 25 → y = 11.69 :=
by
  intros x y h_eq h_x_val
  sorry

end regression_estimate_l152_152972


namespace describe_S_l152_152002

def S : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | (p.2 ≤ 11 ∧ p.1 = 2) ∨ (p.1 ≤ 2 ∧ p.2 = 11) ∨ (p.1 ≥ 2 ∧ p.2 = p.1 + 9) }

theorem describe_S :
  S = { p : ℝ × ℝ | (p.2 ≤ 11 ∧ p.1 = 2) ∨ (p.1 ≤ 2 ∧ p.2 = 11) ∨ (p.1 ≥ 2 ∧ p.2 = p.1 + 9) } := 
by
  -- proof is omitted
  sorry

end describe_S_l152_152002


namespace difference_divisible_by_10_l152_152474

theorem difference_divisible_by_10 : (43 ^ 43 - 17 ^ 17) % 10 = 0 := by
  sorry

end difference_divisible_by_10_l152_152474


namespace smallest_b_value_l152_152475

noncomputable def smallest_b (a b : ℝ) : ℝ :=
if a > 2 ∧ 2 < a ∧ a < b 
   ∧ (2 + a ≤ b) 
   ∧ ((1 / a) + (1 / b) ≤ 1 / 2) 
then b else 0

theorem smallest_b_value : ∀ (a b : ℝ), 
  (2 < a) → (a < b) → (2 + a ≤ b) → 
  ((1 / a) + (1 / b) ≤ 1 / 2) → 
  b = 3 + Real.sqrt 5 := sorry

end smallest_b_value_l152_152475


namespace perfect_square_factors_450_l152_152155

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l152_152155


namespace find_p_l152_152893

variable (f w : ℂ) (p : ℂ)
variable (h1 : f = 4)
variable (h2 : w = 10 + 200 * Complex.I)
variable (h3 : f * p - w = 20000)

theorem find_p : p = 5002.5 + 50 * Complex.I := by
  sorry

end find_p_l152_152893


namespace fraction_equality_l152_152195

theorem fraction_equality 
  (a b c d : ℝ)
  (h1 : a + c = 2 * b)
  (h2 : 2 * b * d = c * (b + d))
  (hb : b ≠ 0)
  (hd : d ≠ 0) :
  a / b = c / d :=
sorry

end fraction_equality_l152_152195


namespace cubic_sum_identity_l152_152294

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l152_152294


namespace rectangle_perimeter_ratio_l152_152791

theorem rectangle_perimeter_ratio (side_length : ℝ) (h : side_length = 4) :
  let small_rectangle_perimeter := 2 * (side_length + (side_length / 4))
  let large_rectangle_perimeter := 2 * (side_length + (side_length / 2))
  small_rectangle_perimeter / large_rectangle_perimeter = 5 / 6 :=
by
  sorry

end rectangle_perimeter_ratio_l152_152791


namespace find_f_five_thirds_l152_152885

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = - f x
def functional_equation (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (-x)
def specific_value (f : ℝ → ℝ) := f (-1/3) = 1/3

theorem find_f_five_thirds
  (hf_odd : odd_function f)
  (hf_fe : functional_equation f)
  (hf_value : specific_value f) :
  f (5 / 3) = 1 / 3 := by
  sorry

end find_f_five_thirds_l152_152885


namespace can_all_mushrooms_become_good_l152_152393

def is_bad (w : Nat) : Prop := w ≥ 10
def is_good (w : Nat) : Prop := w < 10

def mushrooms_initially_bad := 90
def mushrooms_initially_good := 10

def total_mushrooms := mushrooms_initially_bad + mushrooms_initially_good
def total_worms_initial := mushrooms_initially_bad * 10

theorem can_all_mushrooms_become_good :
  ∃ worms_distribution : Fin total_mushrooms → Nat,
  (∀ i : Fin total_mushrooms, is_good (worms_distribution i)) :=
sorry

end can_all_mushrooms_become_good_l152_152393


namespace james_out_of_pocket_cost_l152_152999

theorem james_out_of_pocket_cost (total_cost : ℝ) (coverage : ℝ) (out_of_pocket_cost : ℝ)
  (h1 : total_cost = 300) (h2 : coverage = 0.8) :
  out_of_pocket_cost = 60 :=
by
  sorry

end james_out_of_pocket_cost_l152_152999


namespace michael_truck_meet_once_l152_152192

/-- Michael walks at 6 feet per second -/
def michael_speed := 6

/-- Trash pails are located every 300 feet along the path -/
def pail_distance := 300

/-- A garbage truck travels at 15 feet per second -/
def truck_speed := 15

/-- The garbage truck stops for 45 seconds at each pail -/
def truck_stop_time := 45

/-- Michael passes a pail just as the truck leaves the next pail -/
def initial_distance := 300

/-- Prove that Michael and the truck meet exactly 1 time -/
theorem michael_truck_meet_once :
  ∀ (meeting_times : ℕ), meeting_times = 1 := by
  sorry

end michael_truck_meet_once_l152_152192


namespace flowers_given_to_mother_l152_152796

-- Definitions based on conditions:
def Alissa_flowers : Nat := 16
def Melissa_flowers : Nat := 16
def flowers_left : Nat := 14

-- The proof problem statement:
theorem flowers_given_to_mother :
  Alissa_flowers + Melissa_flowers - flowers_left = 18 := by
  sorry

end flowers_given_to_mother_l152_152796


namespace cube_identity_l152_152301

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l152_152301


namespace min_cubes_required_l152_152513

def volume_of_box (L W H : ℕ) : ℕ := L * W * H
def volume_of_cube (v_cube : ℕ) : ℕ := v_cube
def minimum_number_of_cubes (V_box V_cube : ℕ) : ℕ := V_box / V_cube

theorem min_cubes_required :
  minimum_number_of_cubes (volume_of_box 12 16 6) (volume_of_cube 3) = 384 :=
by sorry

end min_cubes_required_l152_152513


namespace perfect_square_factors_count_450_l152_152123

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l152_152123


namespace crease_length_l152_152994

theorem crease_length 
  (AB AC : ℝ) (BC : ℝ) (BA' : ℝ) (A'C : ℝ)
  (h1 : AB = 10) (h2 : AC = 10) (h3 : BC = 8) (h4 : BA' = 3) (h5 : A'C = 5) :
  ∃ PQ : ℝ, PQ = (Real.sqrt 7393) / 15 := by
  sorry

end crease_length_l152_152994


namespace inequality_example_equality_case_l152_152517

theorem inequality_example (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 2) :
  (1 / (1 + a * b)) + (1 / (1 + b * c)) + (1 / (1 + c * a)) ≥ 27 / 13 :=
by
  -- Proof skipped
  sorry

theorem equality_case (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 2) :
  (1 / (1 + a * b)) + (1 / (1 + b * c)) + (1 / (1 + c * a)) = 27 / 13 ↔ a = b ∧ b = c ∧ c = 2 / 3 :=
by
  -- Proof skipped
  sorry

end inequality_example_equality_case_l152_152517


namespace perfect_square_factors_450_l152_152117

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l152_152117


namespace trajectory_of_M_l152_152840

noncomputable def P : ℝ × ℝ := (2, 2)
noncomputable def circleC (x y : ℝ) : Prop := x^2 + y^2 - 8 * y = 0
noncomputable def isMidpoint (A B M : ℝ × ℝ) : Prop := M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def isIntersectionPoint (l : ℝ × ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, circleC x y ∧ l (x, y) ∧ ((A = (x, y)) ∨ (B = (x, y))) 

theorem trajectory_of_M (M : ℝ × ℝ) : 
  (∃ A B : ℝ × ℝ, isIntersectionPoint (fun p => ∃ k : ℝ, p = (k, k)) A B ∧ isMidpoint A B M) →
  (M.1 - 1)^2 + (M.2 - 3)^2 = 2 := 
sorry

end trajectory_of_M_l152_152840


namespace perfect_square_factors_450_l152_152156

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l152_152156


namespace part1_part2_part3_l152_152992

-- Definition of a companion point
structure Point where
  x : ℝ
  y : ℝ

def isCompanion (P Q : Point) : Prop :=
  Q.x = P.x + 2 ∧ Q.y = P.y - 4

-- Part (1) proof statement
theorem part1 (P Q : Point) (hPQ : isCompanion P Q) (hP : P = ⟨2, -1⟩) (hQ : Q.y = -20 / Q.x) : Q.x = 4 ∧ Q.y = -5 ∧ -20 / 4 = -5 :=
  sorry

-- Part (2) proof statement
theorem part2 (P Q : Point) (hPQ : isCompanion P Q) (hPLine : P.y = P.x - (-5)) (hQ : Q = ⟨-1, -2⟩) : P.x = -3 ∧ P.y = -3 - (-5) ∧ Q.x = -1 ∧ Q.y = -2 :=
  sorry

-- Part (3) proof statement
noncomputable def line2 (Q : Point) := 2*Q.x - 5

theorem part3 (P Q : Point) (hPQ : isCompanion P Q) (hP : P.y = 2*P.x + 3) (hQLine : Q.y = line2 Q) : line2 Q = 2*(P.x + 2) - 5 :=
  sorry

end part1_part2_part3_l152_152992


namespace equidistant_points_quadrants_l152_152563

theorem equidistant_points_quadrants (x y : ℝ)
  (h_line : 4 * x + 7 * y = 28)
  (h_equidistant : abs x = abs y) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end equidistant_points_quadrants_l152_152563


namespace max_result_l152_152398

-- Define the expressions as Lean definitions
def expr1 : Int := 2 + (-2)
def expr2 : Int := 2 - (-2)
def expr3 : Int := 2 * (-2)
def expr4 : Int := 2 / (-2)

-- State the theorem
theorem max_result : 
  (expr2 = 4) ∧ (expr2 > expr1) ∧ (expr2 > expr3) ∧ (expr2 > expr4) :=
by
  sorry

end max_result_l152_152398


namespace division_and_multiplication_l152_152752

theorem division_and_multiplication (dividend divisor quotient factor product : ℕ) 
  (h₁ : dividend = 24) 
  (h₂ : divisor = 3) 
  (h₃ : quotient = dividend / divisor) 
  (h₄ : factor = 5) 
  (h₅ : product = quotient * factor) : 
  quotient = 8 ∧ product = 40 := 
by 
  sorry

end division_and_multiplication_l152_152752


namespace numPerfectSquareFactorsOf450_l152_152143

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l152_152143


namespace correct_statement_about_residuals_l152_152770

-- Define the properties and characteristics of residuals as per the definition
axiom residuals_definition : Prop
axiom residuals_usefulness : residuals_definition → Prop

-- The theorem to prove that the correct statement about residuals is that they can be used to assess the effectiveness of model fitting
theorem correct_statement_about_residuals (h : residuals_definition) : residuals_usefulness h :=
sorry

end correct_statement_about_residuals_l152_152770


namespace perfect_square_factors_450_l152_152128

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l152_152128


namespace derivative_of_x_ln_x_l152_152616

noncomputable
def x_ln_x (x : ℝ) : ℝ := x * Real.log x

theorem derivative_of_x_ln_x (x : ℝ) (hx : x > 0) :
  deriv (x_ln_x) x = 1 + Real.log x :=
by
  -- Proof body, with necessary assumptions and justifications
  sorry

end derivative_of_x_ln_x_l152_152616


namespace fraction_before_simplification_is_24_56_l152_152066

-- Definitions of conditions
def fraction_before_simplification_simplifies_to_3_7 (a b : ℕ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) ∧ Int.gcd a b = 1 ∧ (a = 3 * Int.gcd a b ∧ b = 7 * Int.gcd a b)

def sum_of_numerator_and_denominator_is_80 (a b : ℕ) : Prop :=
  a + b = 80

-- Theorem to prove
theorem fraction_before_simplification_is_24_56 (a b : ℕ) :
  fraction_before_simplification_simplifies_to_3_7 a b →
  sum_of_numerator_and_denominator_is_80 a b →
  (a, b) = (24, 56) :=
sorry

end fraction_before_simplification_is_24_56_l152_152066


namespace part1_solution_part2_solution_l152_152841

def part1 (m : ℝ) (x1 : ℝ) (x2 : ℝ) : Prop :=
  (m * x1 - 2) * (m * x2 - 2) = 4

theorem part1_solution : part1 (1/3) 9 18 :=
by 
  sorry

def part2 (m x1 x2 : ℕ) : Prop :=
  ((m * x1 - 2) * (m * x2 - 2) = 4)

def count_pairs : ℕ := 7

theorem part2_solution 
  (m x1 x2 : ℕ) 
  (h_pos : m > 0 ∧ x1 > 0 ∧ x2 > 0) : 
  ∃ c, c = count_pairs ∧ 
  (part2 m x1 x2) :=
by 
  sorry

end part1_solution_part2_solution_l152_152841


namespace three_times_greater_than_two_l152_152815

theorem three_times_greater_than_two (x : ℝ) : 3 * x - 2 > 0 → 3 * x > 2 :=
by
  sorry

end three_times_greater_than_two_l152_152815


namespace proof_l_squared_l152_152521

noncomputable def longest_line_segment (diameter : ℝ) (sectors : ℕ) : ℝ :=
  let R := diameter / 2
  let theta := (2 * Real.pi) / sectors
  2 * R * (Real.sin (theta / 2))

theorem proof_l_squared :
  let diameter := 18
  let sectors := 4
  let l := longest_line_segment diameter sectors
  l^2 = 162 := by
  let diameter := 18
  let sectors := 4
  let l := longest_line_segment diameter sectors
  have h : l^2 = 162 := sorry
  exact h

end proof_l_squared_l152_152521


namespace complement_intersection_l152_152432

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def M : Set ℕ := {1, 4}
noncomputable def N : Set ℕ := {2, 3}

theorem complement_intersection :
  ((U \ M) ∩ N) = {2, 3} :=
by
  sorry

end complement_intersection_l152_152432


namespace marigolds_sold_second_day_l152_152011

theorem marigolds_sold_second_day (x : ℕ) (h1 : 14 ≤ x)
  (h2 : 2 * x + 14 + x = 89) : x = 25 :=
by
  sorry

end marigolds_sold_second_day_l152_152011


namespace cougar_sleep_hours_l152_152065

-- Definitions
def total_sleep_hours (C Z : Nat) : Prop :=
  C + Z = 70

def zebra_cougar_difference (C Z : Nat) : Prop :=
  Z = C + 2

-- Theorem statement
theorem cougar_sleep_hours :
  ∃ C : Nat, ∃ Z : Nat, zebra_cougar_difference C Z ∧ total_sleep_hours C Z ∧ C = 34 :=
sorry

end cougar_sleep_hours_l152_152065


namespace max_value_a_plus_b_plus_c_l152_152963

-- Definitions used in the problem
def A_n (a n : ℕ) : ℕ := a * (10^n - 1) / 9
def B_n (b n : ℕ) : ℕ := b * (10^n - 1) / 9
def C_n (c n : ℕ) : ℕ := c * (10^(2 * n) - 1) / 9

-- Main statement of the problem
theorem max_value_a_plus_b_plus_c (n : ℕ) (a b c : ℕ) (h : n > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_eq : ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ C_n c n1 - B_n b n1 = 2 * (A_n a n1)^2 ∧ C_n c n2 - B_n b n2 = 2 * (A_n a n2)^2) :
  a + b + c ≤ 18 :=
sorry

end max_value_a_plus_b_plus_c_l152_152963


namespace cubic_sum_l152_152319

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l152_152319


namespace students_average_vegetables_l152_152637

variable (points_needed : ℕ) (points_per_vegetable : ℕ) (students : ℕ) (school_days : ℕ) (school_weeks : ℕ)

def average_vegetables_per_student_per_week (points_needed points_per_vegetable students school_days school_weeks : ℕ) : ℕ :=
  let total_vegetables := points_needed / points_per_vegetable
  let vegetables_per_student := total_vegetables / students
  vegetables_per_student / school_weeks

theorem students_average_vegetables 
  (h1 : points_needed = 200) 
  (h2 : points_per_vegetable = 2) 
  (h3 : students = 25) 
  (h4 : school_days = 10) 
  (h5 : school_weeks = 2) : 
  average_vegetables_per_student_per_week points_needed points_per_vegetable students school_days school_weeks = 2 :=
by
  sorry

end students_average_vegetables_l152_152637


namespace longest_third_side_of_triangle_l152_152333

theorem longest_third_side_of_triangle {a b : ℕ} (ha : a = 8) (hb : b = 9) : 
  ∃ c : ℕ, 1 < c ∧ c < 17 ∧ ∀ (d : ℕ), (1 < d ∧ d < 17) → d ≤ c :=
by
  sorry

end longest_third_side_of_triangle_l152_152333


namespace sum_of_coefficients_l152_152354

noncomputable def integral_result : ℝ := 6 * (Real.sin (π / 2) - Real.sin 0)

noncomputable def f (x a : ℝ) : ℝ := (x - a) ^ integral_result

theorem sum_of_coefficients (a : ℝ) (h : f'.eval 0 / f.eval 0 = -3) :
  ∑ i in Finset.range (integral_result + 1), f.coeff i = 1 := by
  sorry

end sum_of_coefficients_l152_152354


namespace contractor_pays_male_worker_rs_35_l152_152776

theorem contractor_pays_male_worker_rs_35
  (num_male_workers : ℕ)
  (num_female_workers : ℕ)
  (num_child_workers : ℕ)
  (female_worker_wage : ℕ)
  (child_worker_wage : ℕ)
  (average_wage_per_day : ℕ)
  (total_workers : ℕ := num_male_workers + num_female_workers + num_child_workers)
  (total_wage : ℕ := average_wage_per_day * total_workers)
  (total_female_wage : ℕ := num_female_workers * female_worker_wage)
  (total_child_wage : ℕ := num_child_workers * child_worker_wage)
  (total_male_wage : ℕ := total_wage - total_female_wage - total_child_wage) :
  num_male_workers = 20 →
  num_female_workers = 15 →
  num_child_workers = 5 →
  female_worker_wage = 20 →
  child_worker_wage = 8 →
  average_wage_per_day = 26 →
  total_male_wage / num_male_workers = 35 :=
by
  intros h20 h15 h5 h20w h8w h26
  sorry

end contractor_pays_male_worker_rs_35_l152_152776


namespace xy_cubed_identity_l152_152306

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l152_152306


namespace total_gas_cost_l152_152662

def car_city_mpg : ℝ := 30
def car_highway_mpg : ℝ := 40
def city_miles : ℝ := 60 + 40 + 25
def highway_miles : ℝ := 200 + 150 + 180
def gas_price_per_gallon : ℝ := 3.00

theorem total_gas_cost : 
  (city_miles / car_city_mpg + highway_miles / car_highway_mpg) * gas_price_per_gallon = 52.25 := 
by
  sorry

end total_gas_cost_l152_152662


namespace perfect_square_factors_count_450_l152_152126

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l152_152126


namespace oranges_sold_l152_152495

def bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def rotten_oranges : ℕ := 50
def oranges_for_juice : ℕ := 30

theorem oranges_sold : (bags * oranges_per_bag) - rotten_oranges - oranges_for_juice = 220 := by
  sorry

end oranges_sold_l152_152495


namespace radius_of_small_semicircle_l152_152171

theorem radius_of_small_semicircle
  (radius_big_semicircle : ℝ)
  (radius_inner_circle : ℝ) 
  (pairwise_tangent : ∀ x : ℝ, x = radius_big_semicircle → x = 12 ∧ 
                                x = radius_inner_circle → x = 6 ∧ 
                                true) :
  ∃ (r : ℝ), r = 4 :=
by 
  sorry

end radius_of_small_semicircle_l152_152171


namespace anya_possible_wins_l152_152800

-- Define the total rounds played
def total_rounds := 25

-- Define Anya's choices
def anya_rock := 12
def anya_scissors := 6
def anya_paper := 7

-- Define Borya's choices
def borya_rock := 13
def borya_scissors := 9
def borya_paper := 3

-- Define the relationships in rock-paper-scissors game
def rock_beats_scissors := true
def scissors_beat_paper := true
def paper_beats_rock := true

-- Define no draws condition
def no_draws := total_rounds = anya_rock + anya_scissors + anya_paper ∧ total_rounds = borya_rock + borya_scissors + borya_paper

-- Proof problem statement
theorem anya_possible_wins : anya_rock + anya_scissors + anya_paper = total_rounds ∧
                             borya_rock + borya_scissors + borya_paper = total_rounds ∧
                             rock_beats_scissors ∧ scissors_beat_paper ∧ paper_beats_rock ∧
                             no_draws →
                             (9 + 3 + 7 = 19) := by
  sorry

end anya_possible_wins_l152_152800


namespace simplification_of_fractional_equation_l152_152928

theorem simplification_of_fractional_equation (x : ℝ) : 
  (x / (3 - x) - 4 = 6 / (x - 3)) -> (x - 4 * (3 - x) = -6) :=
by
  sorry

end simplification_of_fractional_equation_l152_152928


namespace number_of_articles_l152_152332

theorem number_of_articles (C S : ℝ) (N : ℝ) 
    (h1 : N * C = 40 * S) 
    (h2 : (S - C) / C * 100 = 49.999999999999986) : 
    N = 60 :=
sorry

end number_of_articles_l152_152332


namespace cosine_squared_identity_l152_152842

theorem cosine_squared_identity (α : ℝ) (h : Real.sin (2 * α) = 1 / 3) : Real.cos (α - (π / 4)) ^ 2 = 2 / 3 :=
sorry

end cosine_squared_identity_l152_152842


namespace sum_of_positive_integers_lcm72_l152_152048

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_of_positive_integers_lcm72 : 
  ∑ ν in {ν | is_solution ν}.to_finset, ν = 180 :=
by 
  sorry

end sum_of_positive_integers_lcm72_l152_152048


namespace possible_values_a1_l152_152887

theorem possible_values_a1 (m : ℕ) (h_m_pos : 0 < m)
    (a : ℕ → ℕ) (h_seq : ∀ n, a n.succ = if a n < 2^m then a n ^ 2 + 2^m else a n / 2)
    (h1 : ∀ n, a n > 0) :
    (∀ n, ∃ k : ℕ, a n = 2^k) ↔ (m = 2 ∧ ∃ ℓ : ℕ, a 0 = 2 ^ ℓ ∧ 0 < ℓ) :=
by sorry

end possible_values_a1_l152_152887


namespace michael_current_chickens_l152_152735

-- Defining variables and constants
variable (initial_chickens final_chickens annual_increase : ℕ)

-- Given conditions
def chicken_increase_condition : Prop :=
  final_chickens = initial_chickens + annual_increase * 9

-- Question to answer
def current_chickens (final_chickens annual_increase : ℕ) : ℕ :=
  final_chickens - annual_increase * 9

-- Proof problem
theorem michael_current_chickens
  (initial_chickens : ℕ)
  (final_chickens : ℕ)
  (annual_increase : ℕ)
  (h1 : chicken_increase_condition final_chickens initial_chickens annual_increase) :
  initial_chickens = 550 :=
by
  -- Formal proof would go here.
  sorry

end michael_current_chickens_l152_152735


namespace divisible_by_4_divisible_by_8_divisible_by_16_l152_152006

variable (A B C D : ℕ)
variable (hB : B % 2 = 0)

theorem divisible_by_4 (h1 : (A + 2 * B) % 4 = 0) : 
  (1000 * D + 100 * C + 10 * B + A) % 4 = 0 :=
sorry

theorem divisible_by_8 (h2 : (A + 2 * B + 4 * C) % 8 = 0) :
  (1000 * D + 100 * C + 10 * B + A) % 8 = 0 :=
sorry

theorem divisible_by_16 (h3 : (A + 2 * B + 4 * C + 8 * D) % 16 = 0) :
  (1000 * D + 100 * C + 10 * B + A) % 16 = 0 :=
sorry

end divisible_by_4_divisible_by_8_divisible_by_16_l152_152006


namespace compute_expression_l152_152954

theorem compute_expression :
  (75 * 1313 - 25 * 1313 + 50 * 1313 = 131300) :=
by
  sorry

end compute_expression_l152_152954


namespace find_b_l152_152282

-- Definitions
variable (k : ℤ) (b : ℤ)
def x := 3 * k
def y := 4 * k
def z := 7 * k

-- Conditions
axiom ratio : x / y = 3 / 4 ∧ y / z = 4 / 7
axiom equation : y = 15 * b - 5

-- Theorem statement
theorem find_b : ∃ b : ℤ, 4 * k = 15 * b - 5 ∧ b = 3 :=
by
  sorry

end find_b_l152_152282


namespace percentage_problem_l152_152603

theorem percentage_problem (P : ℝ) (h : (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 42) : P = 35 := 
by
  -- Proof goes here
  sorry

end percentage_problem_l152_152603


namespace sum_of_two_dice_is_9_probability_l152_152503

theorem sum_of_two_dice_is_9_probability : 
  let outcomes := [(1,8), (2,7), (3,6), (4,5), (5,4), (6,3), (7,2), (8,1)];
      total_outcomes := 8 * 8;
      favorable_outcomes := 2 * outcomes.length
  in (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 4 := by
  sorry

end sum_of_two_dice_is_9_probability_l152_152503


namespace intersection_A_B_l152_152850

def A := {x : ℝ | x^2 - x - 2 ≤ 0}
def B := {x : ℝ | ∃ y : ℝ, y = Real.log (1 - x)}

theorem intersection_A_B : (A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_A_B_l152_152850


namespace arithmetic_sequence_fifth_term_l152_152493

theorem arithmetic_sequence_fifth_term :
  ∀ (a d : ℤ), (a + 19 * d = 15) → (a + 20 * d = 18) → (a + 4 * d = -30) :=
by
  intros a d h1 h2
  sorry

end arithmetic_sequence_fifth_term_l152_152493


namespace probability_diff_geq_3_l152_152501

open Finset
open BigOperators

-- Define the set of numbers
def number_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the condition for pairs with positive difference ≥ 3
def valid_pair (a b : ℕ) : Prop := a ≠ b ∧ abs (a - b) ≥ 3

-- Calculate the probability as a fraction
theorem probability_diff_geq_3 : 
  let total_pairs := (number_set.product number_set).filter (λ ab, ab.1 < ab.2) in
  let valid_pairs := total_pairs.filter (λ ab, valid_pair ab.1 ab.2) in
  valid_pairs.card / total_pairs.card = 7 / 12 :=
by
  sorry

end probability_diff_geq_3_l152_152501


namespace determine_range_of_a_l152_152025

theorem determine_range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 ≤ sin x ∧ sin x ≤ 1 →
    (sin x = 1 → (sin x - a)^2 + 1 = 1) ∧
    (sin x = a → (\sin x - a)^2 + 1 = 1)) →
  -1 ≤ a ∧ a ≤ 0 :=
by
  sorry

end determine_range_of_a_l152_152025


namespace b_days_work_alone_l152_152644

theorem b_days_work_alone 
  (W_b : ℝ)  -- Work done by B in one day
  (W_a : ℝ)  -- Work done by A in one day
  (D_b : ℝ)  -- Number of days for B to complete the work alone
  (h1 : W_a = 2 * W_b)  -- A is twice as good a workman as B
  (h2 : 7 * (W_a + W_b) = D_b * W_b)  -- A and B took 7 days together to do the work
  : D_b = 21 :=
sorry

end b_days_work_alone_l152_152644


namespace inversely_proportional_x_y_l152_152907

noncomputable def k := 320

theorem inversely_proportional_x_y (x y : ℕ) (h1 : x * y = k) :
  (∀ x, y = 10 → x = 32) ↔ (x = 32) :=
by
  sorry

end inversely_proportional_x_y_l152_152907


namespace x_cubed_plus_y_cubed_l152_152323

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l152_152323


namespace sum_of_possible_N_l152_152916

theorem sum_of_possible_N 
  (a b c N : ℕ) (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = 2 * (a + b))
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∑ (ab_pairs : ℕ × ℕ) in ({(1, 16), (2, 8), (4, 4)} : Finset (ℕ × ℕ)), (16 * (ab_pairs.1 + ab_pairs.2)) = 560 := 
by
  sorry

end sum_of_possible_N_l152_152916


namespace craig_apples_total_l152_152415

-- Defining the conditions
def initial_apples_craig : ℝ := 20.0
def apples_from_eugene : ℝ := 7.0

-- Defining the total number of apples Craig will have
noncomputable def total_apples_craig : ℝ := initial_apples_craig + apples_from_eugene

-- The theorem stating that Craig will have 27.0 apples.
theorem craig_apples_total : total_apples_craig = 27.0 := by
  -- Proof here
  sorry

end craig_apples_total_l152_152415


namespace unique_triad_l152_152605

theorem unique_triad (x y z : ℕ) 
  (h_distinct: x ≠ y ∧ y ≠ z ∧ z ≠ x) 
  (h_gcd: Nat.gcd (Nat.gcd x y) z = 1)
  (h_div_properties: (z ∣ x + y) ∧ (x ∣ y + z) ∧ (y ∣ z + x)) :
  (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
sorry

end unique_triad_l152_152605


namespace Jim_weekly_savings_l152_152902

-- Define the given conditions
def Sara_initial_savings : ℕ := 4100
def Sara_weekly_savings : ℕ := 10
def weeks : ℕ := 820

-- Define the proof goal based on the conditions
theorem Jim_weekly_savings :
  let Sara_total_savings := Sara_initial_savings + (Sara_weekly_savings * weeks)
  let Jim_weekly_savings := Sara_total_savings / weeks
  Jim_weekly_savings = 15 := 
by 
  sorry

end Jim_weekly_savings_l152_152902


namespace angle_between_v1_v2_l152_152959

-- Define vectors
def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (4, 6)

-- Define the dot product function
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Define the magnitude function
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the cosine of the angle between two vectors
noncomputable def cos_theta (a b : ℝ × ℝ) : ℝ := (dot_product a b) / (magnitude a * magnitude b)

-- Define the angle in degrees between two vectors
noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := Real.arccos (cos_theta a b) * (180 / Real.pi)

-- The statement to prove
theorem angle_between_v1_v2 : angle_between_vectors v1 v2 = Real.arccos (-6 * Real.sqrt 13 / 65) * (180 / Real.pi) :=
sorry

end angle_between_v1_v2_l152_152959


namespace pineapple_rings_per_pineapple_l152_152345

def pineapples_purchased : Nat := 6
def cost_per_pineapple : Nat := 3
def rings_sold_per_set : Nat := 4
def price_per_set_of_4_rings : Nat := 5
def profit_made : Nat := 72

theorem pineapple_rings_per_pineapple : (90 / 5 * 4 / 6) = 12 := 
by 
  sorry

end pineapple_rings_per_pineapple_l152_152345


namespace number_of_children_riding_tricycles_l152_152949

-- Definitions
def bicycles_wheels := 2
def tricycles_wheels := 3

def adults := 6
def total_wheels := 57

-- Problem statement
theorem number_of_children_riding_tricycles (c : ℕ) (H : 12 + 3 * c = total_wheels) : c = 15 :=
by
  sorry

end number_of_children_riding_tricycles_l152_152949


namespace quadratic_roots_identity_l152_152427

theorem quadratic_roots_identity (α β : ℝ) (hαβ : α^2 - 3*α - 4 = 0 ∧ β^2 - 3*β - 4 = 0) : 
  α^2 + α*β - 3*α = 0 := 
by 
  sorry

end quadratic_roots_identity_l152_152427


namespace geometric_sequence_frac_l152_152583

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (h_geometric : ∃ q > 0, ∀ n, a (n+1) = a n * q)
variable (h_decreasing : ∀ n, a (n+1) < a n)
variable (h1 : a 2 * a 8 = 6)
variable (h2 : a 4 + a 6 = 5)

theorem geometric_sequence_frac (h_geometric : ∃ q > 0, ∀ n, a (n+1) = a n * q)
                                (h_decreasing : ∀ n, a (n+1) < a n)
                                (h1 : a 2 * a 8 = 6)
                                (h2 : a 4 + a 6 = 5) :
                                a 3 / a 7 = 9 / 4 :=
by sorry

end geometric_sequence_frac_l152_152583


namespace grape_juice_amount_l152_152783

theorem grape_juice_amount (total_juice : ℝ)
  (orange_juice_percent : ℝ) (watermelon_juice_percent : ℝ)
  (orange_juice_amount : ℝ) (watermelon_juice_amount : ℝ)
  (grape_juice_amount : ℝ) :
  orange_juice_percent = 0.25 →
  watermelon_juice_percent = 0.40 →
  total_juice = 200 →
  orange_juice_amount = total_juice * orange_juice_percent →
  watermelon_juice_amount = total_juice * watermelon_juice_percent →
  grape_juice_amount = total_juice - orange_juice_amount - watermelon_juice_amount →
  grape_juice_amount = 70 :=
by
  sorry

end grape_juice_amount_l152_152783


namespace geometric_sequence_n_terms_l152_152570

/-- In a geometric sequence with the first term a₁ and common ratio q,
the number of terms n for which the nth term aₙ has a given value -/
theorem geometric_sequence_n_terms (a₁ aₙ q : ℚ) (n : ℕ)
  (h1 : a₁ = 9/8)
  (h2 : aₙ = 1/3)
  (h3 : q = 2/3)
  (h_seq : aₙ = a₁ * q^(n-1)) :
  n = 4 := sorry

end geometric_sequence_n_terms_l152_152570


namespace movie_tickets_ratio_l152_152027

theorem movie_tickets_ratio (R H : ℕ) (hR : R = 25) (hH : H = 93) : 
  (H / R : ℚ) = 93 / 25 :=
by
  sorry

end movie_tickets_ratio_l152_152027


namespace find_a_b_l152_152679

def z := Complex.ofReal 3 + Complex.I * 4
def z_conj := Complex.ofReal 3 - Complex.I * 4

theorem find_a_b 
  (a b : ℝ) 
  (h : z + Complex.ofReal a * z_conj + Complex.I * b = Complex.ofReal 9) : 
  a = 2 ∧ b = 4 := 
by 
  sorry

end find_a_b_l152_152679


namespace prod_gcd_lcm_eq_864_l152_152548

theorem prod_gcd_lcm_eq_864 : 
  let a := 24
  let b := 36
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  gcd_ab * lcm_ab = 864 :=
by
  sorry

end prod_gcd_lcm_eq_864_l152_152548


namespace gabby_needs_more_money_l152_152833

theorem gabby_needs_more_money (cost_saved : ℕ) (initial_saved : ℕ) (additional_money : ℕ) (cost_remaining : ℕ) :
  cost_saved = 65 → initial_saved = 35 → additional_money = 20 → cost_remaining = (cost_saved - initial_saved) - additional_money → cost_remaining = 10 :=
by
  intros h_cost_saved h_initial_saved h_additional_money h_cost_remaining
  simp [h_cost_saved, h_initial_saved, h_additional_money] at h_cost_remaining
  exact h_cost_remaining

end gabby_needs_more_money_l152_152833


namespace max_distinct_counts_proof_l152_152239

-- Define the number of boys (B) and girls (G)
def B : ℕ := 29
def G : ℕ := 15

-- Define the maximum distinct dance counts achievable
def max_distinct_counts : ℕ := 29

-- The theorem to prove
theorem max_distinct_counts_proof:
  ∃ (distinct_counts : ℕ), distinct_counts = max_distinct_counts ∧ distinct_counts <= B + G := 
by
  sorry

end max_distinct_counts_proof_l152_152239


namespace percentage_of_items_sold_l152_152394

theorem percentage_of_items_sold (total_items price_per_item discount_rate debt creditors_balance remaining_balance : ℕ)
  (H1 : total_items = 2000)
  (H2 : price_per_item = 50)
  (H3 : discount_rate = 80)
  (H4 : debt = 15000)
  (H5 : remaining_balance = 3000) :
  (total_items * (price_per_item - (price_per_item * discount_rate / 100)) + remaining_balance = debt + remaining_balance) →
  (remaining_balance / (price_per_item - (price_per_item * discount_rate / 100)) / total_items * 100 = 90) :=
by
  sorry

end percentage_of_items_sold_l152_152394


namespace width_of_grassy_plot_l152_152790

-- Definitions
def length_plot : ℕ := 110
def width_path : ℝ := 2.5
def cost_per_sq_meter : ℝ := 0.50
def total_cost : ℝ := 425

-- Hypotheses and Target Proposition
theorem width_of_grassy_plot (w : ℝ) 
  (h1 : length_plot = 110)
  (h2 : width_path = 2.5)
  (h3 : cost_per_sq_meter = 0.50)
  (h4 : total_cost = 425)
  (h5 : (length_plot + 2 * width_path) * (w + 2 * width_path) = 115 * (w + 5))
  (h6 : 110 * w = 110 * w)
  (h7 : (115 * (w + 5) - (110 * w)) = total_cost / cost_per_sq_meter) :
  w = 55 := 
sorry

end width_of_grassy_plot_l152_152790


namespace problem_statement_l152_152699

-- Define the binary operation "*"
def custom_mul (a b : ℤ) : ℤ := a^2 + a * b - b^2

-- State the problem with the conditions
theorem problem_statement : custom_mul 5 (-3) = 1 := by
  sorry

end problem_statement_l152_152699


namespace medium_stores_in_sample_l152_152983

theorem medium_stores_in_sample :
  let total_stores := 300
  let large_stores := 30
  let medium_stores := 75
  let small_stores := 195
  let sample_size := 20
  sample_size * (medium_stores/total_stores) = 5 :=
by
  sorry

end medium_stores_in_sample_l152_152983


namespace Courtney_total_marbles_l152_152409

theorem Courtney_total_marbles (first_jar second_jar third_jar : ℕ) 
  (h1 : first_jar = 80)
  (h2 : second_jar = 2 * first_jar)
  (h3 : third_jar = first_jar / 4) :
  first_jar + second_jar + third_jar = 260 := 
by
  sorry

end Courtney_total_marbles_l152_152409


namespace Oliver_monster_club_cards_l152_152469

theorem Oliver_monster_club_cards (BG AB MC : ℕ) 
  (h1 : BG = 48) 
  (h2 : BG = 3 * AB) 
  (h3 : MC = 2 * AB) : 
  MC = 32 :=
by sorry

end Oliver_monster_club_cards_l152_152469


namespace three_digit_numbers_subtract_297_l152_152674

theorem three_digit_numbers_subtract_297:
  (∃ (p q r : ℕ), 1 ≤ p ∧ p ≤ 9 ∧ 0 ≤ q ∧ q ≤ 9 ∧ 0 ≤ r ∧ r ≤ 9 ∧ (100 * p + 10 * q + r - 297 = 100 * r + 10 * q + p)) →
  (num_valid_three_digit_numbers = 60) :=
by
  sorry

end three_digit_numbers_subtract_297_l152_152674


namespace find_radius_of_small_semicircle_l152_152173

noncomputable def radius_of_small_semicircle (R : ℝ) (r : ℝ) :=
  ∀ (x : ℝ),
    (12: ℝ = R) ∧ (6: ℝ = r) →
    (∃ (x: ℝ), R - x + r = sqrt((r + x)^2 - r^2)) →
    x = 4

theorem find_radius_of_small_semicircle : radius_of_small_semicircle 12 6 :=
begin
  unfold radius_of_small_semicircle,
  intro x,
  assume h1 h2,
  cases h2,
  sorry,
end

end find_radius_of_small_semicircle_l152_152173


namespace parallel_lines_not_coincident_l152_152445

theorem parallel_lines_not_coincident (a : ℝ) :
  (∀ x y : ℝ, ax + 2 * y + 6 = 0) ∧
  (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ (b : ℝ), ∀ x y : ℝ, ax + 2 * y + b = 0 ∧ x + (a - 1) * y + b = 0 →
  a = -1 :=
by
  sorry

end parallel_lines_not_coincident_l152_152445


namespace eggs_in_each_basket_l152_152721

theorem eggs_in_each_basket (n : ℕ) (h1 : 30 % n = 0) (h2 : 42 % n = 0) (h3 : n ≥ 5) :
  n = 6 :=
by sorry

end eggs_in_each_basket_l152_152721


namespace smallest_N_exists_l152_152450

theorem smallest_N_exists :
  ∃ (N : ℕ), (N ≥ 730) ∧
    (∀ (students : list (ℝ × ℝ × ℝ × ℝ)),
      students.length = N →
      (∃ (subset : list (ℝ × ℝ × ℝ × ℝ)),
        subset ⊆ students ∧ subset.length = 10 ∧
        (∃ (i j : Fin 4), i ≠ j ∧
          ∀ (seq : Fin 10 → ℕ),
            (∀ k : Fin 10, seq k < subset.length) →
            let subjects := λ (idx : Fin 10) => list.indexN idx {a,b,c,d} in
            (is_sorted (λ k, subjects (seq k))) ))) :=
sorry

end smallest_N_exists_l152_152450


namespace g_f_of_3_l152_152349

def f (x : ℝ) : ℝ := x^3 - 4
def g (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 2

theorem g_f_of_3 : g (f 3) = 1704 := by
  sorry

end g_f_of_3_l152_152349


namespace minimum_n_l152_152167

noncomputable def a (n : ℕ) : ℕ := 2 ^ (n - 2)

noncomputable def b (n : ℕ) : ℕ := n - 6 + a n

noncomputable def S (n : ℕ) : ℕ := (n * (n - 11)) / 2 + (2 ^ n - 1) / 2

theorem minimum_n (n : ℕ) (hn : n ≥ 5) : S 5 > 0 := by
  sorry

end minimum_n_l152_152167


namespace derivative_f_at_pi_l152_152676

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem derivative_f_at_pi : (deriv f π) = -1 := 
by
  sorry

end derivative_f_at_pi_l152_152676


namespace over_limit_weight_l152_152455

variable (hc_books : ℕ) (hc_weight : ℕ → ℝ)
variable (tex_books : ℕ) (tex_weight : ℕ → ℝ)
variable (knick_knacks : ℕ) (knick_weight : ℕ → ℝ)
variable (weight_limit : ℝ)

axiom hc_books_value : hc_books = 70
axiom hc_weight_value : hc_weight hc_books = 0.5
axiom tex_books_value : tex_books = 30
axiom tex_weight_value : tex_weight tex_books = 2
axiom knick_knacks_value : knick_knacks = 3
axiom knick_weight_value : knick_weight knick_knacks = 6
axiom weight_limit_value : weight_limit = 80

theorem over_limit_weight : 
  (hc_books * hc_weight hc_books + tex_books * tex_weight tex_books + knick_knacks * knick_weight knick_knacks) - weight_limit = 33 := by
  sorry

end over_limit_weight_l152_152455


namespace relationship_between_y_values_l152_152367

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + |b| * x + c

theorem relationship_between_y_values
  (a b c y1 y2 y3 : ℝ)
  (h1 : quadratic_function a b c (-14 / 3) = y1)
  (h2 : quadratic_function a b c (5 / 2) = y2)
  (h3 : quadratic_function a b c 3 = y3)
  (axis_symmetry : -(|b| / (2 * a)) = -1)
  (h_pos : 0 < a) :
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_between_y_values_l152_152367


namespace find_f_at_3_l152_152079

noncomputable def f (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * (x^32 + 1) - 1) / (x^(2^6 - 1) - 1)

theorem find_f_at_3 : f 3 = 3 :=
by
  sorry

end find_f_at_3_l152_152079


namespace tangent_line_eq_l152_152753

theorem tangent_line_eq (f : ℝ → ℝ) (f' : ℝ → ℝ) (x y : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, f' x = Real.exp x) →
  f 0 = 1 →
  f' 0 = 1 →
  x = 0 →
  y = 1 →
  x - y + 1 = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end tangent_line_eq_l152_152753


namespace janet_saves_minutes_l152_152811

theorem janet_saves_minutes 
  (key_search_time : ℕ := 8) 
  (complaining_time : ℕ := 3) 
  (days_in_week : ℕ := 7) : 
  (key_search_time + complaining_time) * days_in_week = 77 := 
by
  sorry

end janet_saves_minutes_l152_152811


namespace sum_a4_a5_a6_l152_152341

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Conditions
axiom h1 : a 1 = 2
axiom h2 : a 3 = -10

-- Definition of arithmetic sequence
axiom h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d

-- Proof problem statement
theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = -66 :=
by
  sorry

end sum_a4_a5_a6_l152_152341


namespace max_n_possible_l152_152220

theorem max_n_possible (k : ℕ) (h_k : k > 1) : ∃ n : ℕ, n = k - 1 :=
by
  sorry

end max_n_possible_l152_152220


namespace soul_inequality_phi_inequality_iff_t_one_l152_152004

noncomputable def e : ℝ := Real.exp 1

theorem soul_inequality (x : ℝ) : e^x ≥ x + 1 ↔ x = 0 :=
by sorry

theorem phi_inequality_iff_t_one (x t : ℝ) : (∀ x, e^x - t*x - 1 ≥ 0) ↔ t = 1 :=
by sorry

end soul_inequality_phi_inequality_iff_t_one_l152_152004


namespace supplements_of_congruent_angles_are_congruent_l152_152543

-- Define the concept of supplementary angles
def is_supplementary (α β : ℝ) : Prop := α + β = 180

-- Statement of the problem
theorem supplements_of_congruent_angles_are_congruent :
  ∀ {α β γ δ : ℝ},
  is_supplementary α β →
  is_supplementary γ δ →
  β = δ →
  α = γ :=
by
  intros α β γ δ h1 h2 h3
  sorry

end supplements_of_congruent_angles_are_congruent_l152_152543


namespace triangle_inequalities_l152_152681

open Real

-- Define a structure for a triangle with its properties
structure Triangle :=
(a b c R ra rb rc : ℝ)

-- Main statement to be proved
theorem triangle_inequalities (Δ : Triangle) (h : 2 * Δ.R ≤ Δ.ra) :
  Δ.a > Δ.b ∧ Δ.a > Δ.c ∧ 2 * Δ.R > Δ.rb ∧ 2 * Δ.R > Δ.rc :=
sorry

end triangle_inequalities_l152_152681


namespace solution_set_interval_l152_152264

theorem solution_set_interval (a : ℝ) : 
  {x : ℝ | x^2 - 2*a*x + a^2 - 1 < 0} = {x : ℝ | a - 1 < x ∧ x < a + 1} :=
sorry

end solution_set_interval_l152_152264


namespace quotient_equivalence_l152_152509

variable (N H J : ℝ)

theorem quotient_equivalence
  (h1 : N / H = 1.2)
  (h2 : H / J = 5 / 6) :
  N / J = 1 := by
  sorry

end quotient_equivalence_l152_152509


namespace trajectory_of_M_l152_152107

-- Define the two circles C1 and C2
def C1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the condition for the moving circle M being tangent to both circles
def isTangent (Mx My : ℝ) : Prop := 
  let distC1 := (Mx + 3)^2 + My^2
  let distC2 := (Mx - 3)^2 + My^2
  distC2 - distC1 = 4

-- The equation of the trajectory of M
theorem trajectory_of_M (Mx My : ℝ) (h : isTangent Mx My) : 
  Mx^2 - (My^2 / 8) = 1 ∧ Mx < 0 :=
sorry

end trajectory_of_M_l152_152107


namespace total_revenue_correct_l152_152212

noncomputable def revenue_calculation : ℕ :=
  let fair_tickets := 60
  let fair_price := 15
  let baseball_tickets := fair_tickets / 3
  let baseball_price := 10
  let play_tickets := 2 * fair_tickets
  let play_price := 12
  fair_tickets * fair_price
  + baseball_tickets * baseball_price
  + play_tickets * play_price

theorem total_revenue_correct : revenue_calculation = 2540 :=
  by
  sorry

end total_revenue_correct_l152_152212


namespace curve_crosses_itself_l152_152757

theorem curve_crosses_itself :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ (t1^2 - 3 = t2^2 - 3) ∧ (t1^3 - 6*t1 + 2 = t2^3 - 6*t2 + 2) ∧
  ((t1^2 - 3 = 3) ∧ (t1^3 - 6*t1 + 2 = 2)) :=
by
  sorry

end curve_crosses_itself_l152_152757


namespace sasha_stickers_l152_152477

variables (m n : ℕ) (t : ℝ)

-- Conditions
def conditions : Prop :=
  m < n ∧ -- Fewer coins than stickers
  m ≥ 1 ∧ -- At least one coin
  n ≥ 1 ∧ -- At least one sticker
  t > 1 ∧ -- t is greater than 1
  m * t + n = 100 ∧ -- Coin increase condition
  m + n * t = 101 -- Sticker increase condition

-- Theorem stating that the number of stickers must be 34 or 66
theorem sasha_stickers : conditions m n t → n = 34 ∨ n = 66 :=
sorry

end sasha_stickers_l152_152477


namespace simplify_fractions_l152_152479

theorem simplify_fractions : 5 * (21 / 6) * (18 / -63) = -5 := by
  sorry

end simplify_fractions_l152_152479


namespace sum_of_areas_is_correct_l152_152962

/-- Define the lengths of the rectangles -/
def lengths : List ℕ := [4, 16, 36, 64, 100]

/-- Define the common base width of the rectangles -/
def base_width : ℕ := 3

/-- Define the area of a rectangle given its length and a common base width -/
def area (length : ℕ) : ℕ := base_width * length

/-- Compute the total area of the given rectangles -/
def total_area : ℕ := (lengths.map area).sum

/-- Theorem stating that the total area of the five rectangles is 660 -/
theorem sum_of_areas_is_correct : total_area = 660 := by
  sorry

end sum_of_areas_is_correct_l152_152962


namespace integer_solutions_of_system_l152_152666

theorem integer_solutions_of_system (x y z : ℤ) :
  x + y + z = 2 ∧ x^3 + y^3 + z^3 = -10 ↔ 
  (x = 3 ∧ y = 3 ∧ z = -4) ∨ 
  (x = 3 ∧ y = -4 ∧ z = 3) ∨ 
  (x = -4 ∧ y = 3 ∧ z = 3) :=
sorry

end integer_solutions_of_system_l152_152666


namespace polynomial_negativity_l152_152358

theorem polynomial_negativity (a x : ℝ) (h₀ : 0 < x) (h₁ : x < a) (h₂ : 0 < a) : 
  (a - x)^6 - 3 * a * (a - x)^5 + (5 / 2) * a^2 * (a - x)^4 - (1 / 2) * a^4 * (a - x)^2 < 0 := 
by
  sorry

end polynomial_negativity_l152_152358


namespace mustard_found_at_third_table_l152_152403

variable (a b T : ℝ)
def found_mustard_at_first_table := (a = 0.25)
def found_mustard_at_second_table := (b = 0.25)
def total_mustard_found := (T = 0.88)

theorem mustard_found_at_third_table
  (h1 : found_mustard_at_first_table a)
  (h2 : found_mustard_at_second_table b)
  (h3 : total_mustard_found T) :
  T - (a + b) = 0.38 := by
  sorry

end mustard_found_at_third_table_l152_152403


namespace exists_tetrahedra_volume_and_face_area_conditions_l152_152898

noncomputable def volume (T : Tetrahedron) : ℝ := sorry
noncomputable def face_area (T : Tetrahedron) : List ℝ := sorry

-- The existence of two tetrahedra such that the volume of T1 > T2 
-- and the area of each face of T1 does not exceed any face of T2.
theorem exists_tetrahedra_volume_and_face_area_conditions :
  ∃ (T1 T2 : Tetrahedron), 
    (volume T1 > volume T2) ∧ 
    (∀ (a1 : ℝ), a1 ∈ face_area T1 → 
      ∃ (a2 : ℝ), a2 ∈ face_area T2 ∧ a2 ≥ a1) :=
sorry

end exists_tetrahedra_volume_and_face_area_conditions_l152_152898


namespace probability_odd_product_l152_152272

theorem probability_odd_product :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let pairs := { (a, b) | a ∈ S ∧ b ∈ S ∧ a < b }
  let odd_product_pairs := { (a, b) | a ∈ S ∧ b ∈ S ∧ a < b ∧ a % 2 = 1 ∧ b % 2 = 1 }
  (finset.card odd_product_pairs : ℚ) / (finset.card pairs : ℚ) = 5 / 18 :=
by
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let pairs := finset.filter (λ (p : ℕ × ℕ), p.1 < p.2) (finset.product (finset.filter (λ x, true) (finset.range 10)) (finset.filter (λ x, true) (finset.range 10)))
  let odd_product_pairs := finset.filter (λ (p : ℕ × ℕ), p.1 % 2 = 1 ∧ p.2 % 2 = 1) pairs
  have h_pairs : finset.card pairs = 36 := sorry
  have h_odd_product_pairs : finset.card odd_product_pairs = 10 := sorry
  exact (congr_arg (λ x, x : ℚ) h_odd_product_pairs) / (congr_arg (λ x, x : ℚ) h_pairs) ▸ sorry

end probability_odd_product_l152_152272


namespace Oliver_monster_club_cards_l152_152470

theorem Oliver_monster_club_cards (BG AB MC : ℕ) 
  (h1 : BG = 48) 
  (h2 : BG = 3 * AB) 
  (h3 : MC = 2 * AB) : 
  MC = 32 :=
by sorry

end Oliver_monster_club_cards_l152_152470


namespace toothpicks_at_20th_stage_l152_152498

theorem toothpicks_at_20th_stage (n : ℕ) (a d : ℕ) (h_a : a = 3) (h_d : d = 3) :
  T n = a + d * (n - 1) → T 20 = 60 :=
by
  intro formula
  rw [h_a, h_d]
  sorry

end toothpicks_at_20th_stage_l152_152498


namespace average_probable_weight_l152_152703

theorem average_probable_weight (weight : ℝ) (h1 : 61 < weight) (h2 : weight ≤ 64) : 
  (61 + 64) / 2 = 62.5 := 
by
  sorry

end average_probable_weight_l152_152703


namespace arithmetic_sequences_diff_l152_152684

theorem arithmetic_sequences_diff
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (d_a d_b : ℤ)
  (ha : ∀ n, a n = 3 + n * d_a)
  (hb : ∀ n, b n = -3 + n * d_b)
  (h : a 19 - b 19 = 16) :
  a 10 - b 10 = 11 := by
    sorry

end arithmetic_sequences_diff_l152_152684


namespace division_by_negative_divisor_l152_152951

theorem division_by_negative_divisor : 15 / (-3) = -5 :=
by sorry

end division_by_negative_divisor_l152_152951


namespace relationship_between_y_values_l152_152365

-- Define the quadratic function given the constraints
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + abs b * x + c

-- Define the points (x1, y1), (x2, y2), (x3, y3)
def x1 := -14 / 3
def x2 := 5 / 2
def x3 := 3

def y1 (a b c : ℝ) : ℝ := quadratic_function a b c x1
def y2 (a b c : ℝ) : ℝ := quadratic_function a b c x2
def y3 (a b c : ℝ) : ℝ := quadratic_function a b c x3

theorem relationship_between_y_values 
  (a b c : ℝ) (h1 : - (abs b) / (2 * a) = -1) 
  (y1_value : ℝ := y1 a b c) 
  (y2_value : ℝ := y2 a b c) 
  (y3_value : ℝ := y3 a b c) : 
  y2_value < y1_value ∧ y1_value < y3_value := 
by 
  sorry

end relationship_between_y_values_l152_152365


namespace perfect_squares_multiple_of_72_number_of_perfect_squares_multiple_of_72_l152_152975

theorem perfect_squares_multiple_of_72 (N : ℕ) : 
  (N^2 < 1000000) ∧ (N^2 % 72 = 0) ↔ N ≤ 996 :=
sorry

theorem number_of_perfect_squares_multiple_of_72 : 
  ∃ upper_bound : ℕ, upper_bound = 83 ∧ ∀ n : ℕ, (n < 1000000) → (n % 144 = 0) → n ≤ (12 * upper_bound) :=
sorry

end perfect_squares_multiple_of_72_number_of_perfect_squares_multiple_of_72_l152_152975


namespace value_of_larger_denom_eq_10_l152_152799

/-- Anna has 12 bills in her wallet, and the total value is $100. 
    She has 4 $5 bills and 8 bills of a larger denomination.
    Prove that the value of the larger denomination bill is $10. -/
theorem value_of_larger_denom_eq_10 (n : ℕ) (b : ℤ) (total_value : ℤ) (five_bills : ℕ) (larger_bills : ℕ):
    (total_value = 100) ∧ 
    (five_bills = 4) ∧ 
    (larger_bills = 8) ∧ 
    (n = five_bills + larger_bills) ∧ 
    (n = 12) → 
    (b = 10) :=
by
  sorry

end value_of_larger_denom_eq_10_l152_152799


namespace cards_choice_ways_l152_152158

theorem cards_choice_ways (S : List Char) (cards : Finset (Char × ℕ)) :
  (∀ c ∈ cards, c.1 ∈ S) ∧
  (∀ (c1 c2 : Char × ℕ), c1 ∈ cards → c2 ∈ cards → c1 ≠ c2 → c1.1 ≠ c2.1) ∧
  (∃ c ∈ cards, c.2 = 1 ∧ c.1 = 'H') →
  (∃ c ∈ cards, c.2 = 1) →
  ∃ (ways : ℕ), ways = 1014 := 
sorry

end cards_choice_ways_l152_152158


namespace product_of_solutions_l152_152088

theorem product_of_solutions : 
  ∀ y : ℝ, (|y| = 3 * (|y| - 2)) → ∃ a b : ℝ, (a = 3 ∧ b = -3) ∧ (a * b = -9) := 
by 
  sorry

end product_of_solutions_l152_152088


namespace find_number_l152_152018

-- Define the conditions
variables (y : ℝ) (Some_number : ℝ) (x : ℝ)

-- State the given equation
def equation := 19 * (x + y) + Some_number = 19 * (-x + y) - 21

-- State the proposition to prove
theorem find_number (h : equation 1 y Some_number) : Some_number = -59 :=
sorry

end find_number_l152_152018


namespace part_I_solution_set_part_II_range_a_l152_152688

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part_I_solution_set :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} :=
by
  sorry

theorem part_II_range_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 - a) ↔ (-1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end part_I_solution_set_part_II_range_a_l152_152688


namespace rectangle_length_reduction_l152_152364

theorem rectangle_length_reduction:
  ∀ (L W : ℝ) (X : ℝ),
  W > 0 →
  L > 0 →
  (L * (1 - X / 100) * (4 / 3)) * W = L * W →
  X = 25 :=
by
  intros L W X hW hL hEq
  sorry

end rectangle_length_reduction_l152_152364


namespace power_inequality_l152_152101

theorem power_inequality (n : ℕ) (x : ℝ) (h1 : 0 < n) (h2 : x > -1) : (1 + x)^n ≥ 1 + n * x :=
sorry

end power_inequality_l152_152101


namespace find_salary_l152_152233

theorem find_salary (S : ℤ) (food house_rent clothes left : ℤ) 
  (h_food : food = S / 5) 
  (h_house_rent : house_rent = S / 10) 
  (h_clothes : clothes = 3 * S / 5) 
  (h_left : left = 18000) 
  (h_spent : food + house_rent + clothes + left = S) : 
  S = 180000 :=
by {
  sorry
}

end find_salary_l152_152233


namespace initial_action_figures_correct_l152_152871

def initial_action_figures (x : ℕ) : Prop :=
  x + 11 - 10 = 8

theorem initial_action_figures_correct :
  ∃ x : ℕ, initial_action_figures x ∧ x = 7 :=
by
  sorry

end initial_action_figures_correct_l152_152871


namespace mutually_coprime_divisors_l152_152265

theorem mutually_coprime_divisors (a x y : ℕ) (h1 : a = 1944) 
  (h2 : ∃ d1 d2 d3, d1 * d2 * d3 = a ∧ gcd x y = 1 ∧ gcd x (x + y) = 1 ∧ gcd y (x + y) = 1) : 
  (x = 1 ∧ y = 2 ∧ x + y = 3) ∨ 
  (x = 1 ∧ y = 8 ∧ x + y = 9) ∨ 
  (x = 1 ∧ y = 3 ∧ x + y = 4) :=
sorry

end mutually_coprime_divisors_l152_152265


namespace plane_split_four_regions_l152_152080

theorem plane_split_four_regions :
  (∀ x y : ℝ, y = 3 * x ∨ x = 3 * y) → (exists regions : ℕ, regions = 4) :=
by
  sorry

end plane_split_four_regions_l152_152080


namespace inversely_proportional_x_y_l152_152908

noncomputable def k := 320

theorem inversely_proportional_x_y (x y : ℕ) (h1 : x * y = k) :
  (∀ x, y = 10 → x = 32) ↔ (x = 32) :=
by
  sorry

end inversely_proportional_x_y_l152_152908


namespace technical_class_average_age_l152_152988

noncomputable def average_age_in_technical_class : ℝ :=
  let average_age_arts := 21
  let num_arts_classes := 8
  let num_technical_classes := 5
  let overall_average_age := 19.846153846153847
  let total_classes := num_arts_classes + num_technical_classes
  let total_age_university := overall_average_age * total_classes
  ((total_age_university - (average_age_arts * num_arts_classes)) / num_technical_classes)

theorem technical_class_average_age :
  average_age_in_technical_class = 990.4 :=
by
  sorry  -- Proof to be provided

end technical_class_average_age_l152_152988


namespace total_marbles_l152_152571

theorem total_marbles (y b g : ℝ) (h1 : y = 1.4 * b) (h2 : g = 1.75 * y) :
  b + y + g = 3.4643 * y :=
sorry

end total_marbles_l152_152571


namespace day_of_week_150th_day_of_year_N_minus_1_l152_152713

/-- Given that the 250th day of year N is a Friday and year N is a leap year,
    prove that the 150th day of year N-1 is a Friday. -/
theorem day_of_week_150th_day_of_year_N_minus_1
  (N : ℕ) 
  (H1 : (250 % 7 = 5) → true)  -- Condition that 250th day is five days after Sunday (Friday).
  (H2 : 366 % 7 = 2)           -- Condition that year N is a leap year with 366 days.
  (H3 : (N - 1) % 7 = (N - 1) % 7) -- Used for year transition check.
  : 150 % 7 = 5 := sorry       -- Proving that the 150th of year N-1 is Friday.

end day_of_week_150th_day_of_year_N_minus_1_l152_152713


namespace integer_sequence_perfect_square_l152_152775

noncomputable def seq (a : ℕ → ℝ) : Prop :=
a 1 = 1 ∧ a 2 = 4 ∧ ∀ n ≥ 2, a n = (a (n - 1) * a (n + 1) + 1) ^ (1 / 2)

theorem integer_sequence {a : ℕ → ℝ} : 
  seq a → ∀ n, ∃ k : ℤ, a n = k := 
by sorry

theorem perfect_square {a : ℕ → ℝ} :
  seq a → ∀ n, ∃ k : ℤ, 2 * a n * a (n + 1) + 1 = k ^ 2 :=
by sorry

end integer_sequence_perfect_square_l152_152775


namespace original_number_is_842_l152_152384

theorem original_number_is_842 (x y z : ℕ) (h1 : x * z = y^2)
  (h2 : 100 * z + x = 100 * x + z - 594)
  (h3 : 10 * z + y = 10 * y + z - 18)
  (hx : x = 8) (hy : y = 4) (hz : z = 2) :
  100 * x + 10 * y + z = 842 :=
by
  sorry

end original_number_is_842_l152_152384


namespace minimize_area_of_quadrilateral_l152_152691

noncomputable def minimize_quad_area (AB BC CD DA A1 B1 C1 D1 : ℝ) (k : ℝ) : Prop :=
  -- Conditions
  AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0 ∧ k > 0 ∧
  A1 = k * AB ∧ B1 = k * BC ∧ C1 = k * CD ∧ D1 = k * DA →
  -- Conclusion
  k = 1 / 2

-- Statement without proof
theorem minimize_area_of_quadrilateral (AB BC CD DA : ℝ) : ∃ k : ℝ, minimize_quad_area AB BC CD DA (k * AB) (k * BC) (k * CD) (k * DA) k :=
sorry

end minimize_area_of_quadrilateral_l152_152691


namespace regular_octagon_side_length_sum_l152_152696

theorem regular_octagon_side_length_sum (s : ℝ) (h₁ : s = 2.3) (h₂ : 1 = 100) : 
  8 * (s * 100) = 1840 :=
by
  sorry

end regular_octagon_side_length_sum_l152_152696


namespace number_of_ways_to_write_528_as_sum_of_consecutive_integers_l152_152989

theorem number_of_ways_to_write_528_as_sum_of_consecutive_integers : 
  ∃ (n : ℕ), (2 ≤ n ∧ ∃ k : ℕ, n * (2 * k + n - 1) = 1056) ∧ n = 15 :=
by
  sorry

end number_of_ways_to_write_528_as_sum_of_consecutive_integers_l152_152989


namespace infinite_solutions_xyz_l152_152012

theorem infinite_solutions_xyz (k : ℤ) : 
  let x := k * (2 * k^2 + 1),
      y := 2 * k^2 + 1,
      z := -k * (2 * k^2 + 1)
  in x ^ 2 + y ^ 2 + z ^ 2 = x ^ 3 + y ^ 3 + z ^ 3 := 
by sorry

end infinite_solutions_xyz_l152_152012


namespace K_time_correct_l152_152385

open Real

noncomputable def K_speed : ℝ := sorry
noncomputable def M_speed : ℝ := K_speed - 1 / 2
noncomputable def K_time : ℝ := 45 / K_speed
noncomputable def M_time : ℝ := 45 / M_speed

theorem K_time_correct (K_speed_correct : 45 / K_speed - 45 / M_speed = 1 / 2) : K_time = 45 / K_speed :=
by
  sorry

end K_time_correct_l152_152385


namespace probability_two_cities_less_than_8000_l152_152625

-- Define the city names
inductive City
| Bangkok | CapeTown | Honolulu | London | NewYork
deriving DecidableEq, Inhabited

-- Define the distance between cities
def distance : City → City → ℕ
| City.Bangkok, City.CapeTown  => 6300
| City.Bangkok, City.Honolulu  => 6609
| City.Bangkok, City.London    => 5944
| City.Bangkok, City.NewYork   => 8650
| City.CapeTown, City.Bangkok  => 6300
| City.CapeTown, City.Honolulu => 11535
| City.CapeTown, City.London   => 5989
| City.CapeTown, City.NewYork  => 7800
| City.Honolulu, City.Bangkok  => 6609
| City.Honolulu, City.CapeTown => 11535
| City.Honolulu, City.London   => 7240
| City.Honolulu, City.NewYork  => 4980
| City.London, City.Bangkok    => 5944
| City.London, City.CapeTown   => 5989
| City.London, City.Honolulu   => 7240
| City.London, City.NewYork    => 3470
| City.NewYork, City.Bangkok   => 8650
| City.NewYork, City.CapeTown  => 7800
| City.NewYork, City.Honolulu  => 4980
| City.NewYork, City.London    => 3470
| _, _                         => 0

-- Prove the probability
theorem probability_two_cities_less_than_8000 :
  let pairs := [(City.Bangkok, City.CapeTown), (City.Bangkok, City.Honolulu), (City.Bangkok, City.London), (City.CapeTown, City.London), (City.CapeTown, City.NewYork), (City.Honolulu, City.London), (City.Honolulu, City.NewYork), (City.London, City.NewYork)]
  (pairs.length : ℚ) / 10 = 4 / 5 :=
sorry

end probability_two_cities_less_than_8000_l152_152625


namespace alice_study_time_for_average_75_l152_152073

variable (study_time : ℕ → ℚ)
variable (score : ℕ → ℚ)

def inverse_relation := ∀ n, study_time n * score n = 120

theorem alice_study_time_for_average_75
  (inverse_relation : inverse_relation study_time score)
  (study_time_1 : study_time 1 = 2)
  (score_1 : score 1 = 60)
  : study_time 2 = 4/3 := by
  sorry

end alice_study_time_for_average_75_l152_152073


namespace find_C_and_D_l152_152091

theorem find_C_and_D (C D : ℚ) (h1 : 5 * C + 3 * D - 4 = 47) (h2 : C = D + 2) : 
  C = 57 / 8 ∧ D = 41 / 8 :=
by 
  sorry

end find_C_and_D_l152_152091


namespace geometric_sequence_value_a3_l152_152424

-- Define the geometric sequence
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

-- Conditions given in the problem
variable (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 2)
variable (h₂ : (geometric_sequence a₁ q 4) * (geometric_sequence a₁ q 6) = 4 * (geometric_sequence a₁ q 7) ^ 2)

-- The goal is to prove that a₃ = 1
theorem geometric_sequence_value_a3 : geometric_sequence a₁ q 3 = 1 :=
by
  sorry

end geometric_sequence_value_a3_l152_152424


namespace xy_cubed_identity_l152_152307

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l152_152307


namespace number_of_sides_l152_152245

theorem number_of_sides (P l n : ℕ) (hP : P = 49) (hl : l = 7) (h : P = n * l) : n = 7 :=
by
  sorry

end number_of_sides_l152_152245


namespace expected_value_max_of_four_element_subset_l152_152591

open Nat

noncomputable def expected_value_max_element (S : finset ℕ) : ℚ := 
  ∑ i in S.filter (λ x, x = S.max ℕ), i.to_rat / S.card.to_rat

theorem expected_value_max_of_four_element_subset  :
  let S := (finset.range 9).powerset.filter (λ s, s.card = 4) in
  let expected_value := (S.sum (λ s, ↑(s.max ℕ) * s.card.to_rat)).sum / S.card.to_rat in
  m, n : ℕ,
  m = 36 ∧ n = 5 ∧ nat.coprime m n ∧ expected_value = (m / n : ℚ) → 
  m + n = 41 := by sorry

end expected_value_max_of_four_element_subset_l152_152591


namespace probability_of_both_selected_l152_152382

theorem probability_of_both_selected (pX pY : ℚ) (hX : pX = 1/7) (hY : pY = 2/5) : 
  pX * pY = 2 / 35 :=
by {
  sorry
}

end probability_of_both_selected_l152_152382


namespace product_of_divisors_sum_l152_152623

theorem product_of_divisors_sum :
  ∃ (a b c : ℕ), (a ∣ 11^3) ∧ (b ∣ 11^3) ∧ (c ∣ 11^3) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a * b * c = 11^3) ∧ (a + b + c = 133) :=
sorry

end product_of_divisors_sum_l152_152623


namespace p_work_alone_time_l152_152057

variable (Wp Wq : ℝ)
variable (x : ℝ)

-- Conditions
axiom h1 : Wp = 1.5 * Wq
axiom h2 : (1 / x) + (Wq / Wp) * (1 / x) = 1 / 15

-- Proof of the question (p alone can complete the work in x days)
theorem p_work_alone_time : x = 25 :=
by
  -- Add your proof here
  sorry

end p_work_alone_time_l152_152057


namespace minimum_value_expr_l152_152269

noncomputable def expr (x y z : ℝ) : ℝ := 
  3 * x^2 + 2 * x * y + 3 * y^2 + 2 * y * z + 3 * z^2 - 3 * x + 3 * y - 3 * z + 9

theorem minimum_value_expr : 
  ∃ (x y z : ℝ), ∀ (a b c : ℝ), expr a b c ≥ expr x y z ∧ expr x y z = 3/2 :=
sorry

end minimum_value_expr_l152_152269


namespace zero_function_l152_152462

noncomputable def f : ℝ → ℝ := sorry -- Let it be a placeholder for now.

theorem zero_function (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b))
  (h_int : ∀ n : ℕ, ∫ x in a..b, (x : ℝ)^n * f x = 0) :
  ∀ x ∈ Set.Icc a b, f x = 0 :=
by
  sorry -- placeholder for the proof

end zero_function_l152_152462


namespace x_cubed_plus_y_cubed_l152_152324

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l152_152324


namespace find_abc_l152_152185

theorem find_abc
  {a b c : ℤ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 30)
  (h2 : 1/a + 1/b + 1/c + 672/(a*b*c) = 1) :
  a * b * c = 2808 :=
sorry

end find_abc_l152_152185


namespace michael_num_dogs_l152_152191

variable (total_cost : ℕ)
variable (cost_per_animal : ℕ)
variable (num_cats : ℕ)
variable (num_dogs : ℕ)

-- Conditions
def michael_total_cost := total_cost = 65
def michael_num_cats := num_cats = 2
def michael_cost_per_animal := cost_per_animal = 13

-- Theorem to prove
theorem michael_num_dogs (h_total_cost : michael_total_cost total_cost)
                         (h_num_cats : michael_num_cats num_cats)
                         (h_cost_per_animal : michael_cost_per_animal cost_per_animal) :
  num_dogs = 3 :=
by
  sorry

end michael_num_dogs_l152_152191


namespace period_of_f_l152_152226

noncomputable def f (x : ℝ) : ℝ := (Real.tan (x/3)) + (Real.sin x)

theorem period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x :=
  sorry

end period_of_f_l152_152226


namespace probability_different_colors_l152_152447

-- Define the total number of blue and yellow chips
def blue_chips : ℕ := 5
def yellow_chips : ℕ := 7
def total_chips : ℕ := blue_chips + yellow_chips

-- Define the probability of drawing a blue chip and a yellow chip
def prob_blue : ℚ := blue_chips / total_chips
def prob_yellow : ℚ := yellow_chips / total_chips

-- Define the probability of drawing two chips of different colors
def prob_different_colors := 2 * (prob_blue * prob_yellow)

theorem probability_different_colors :
  prob_different_colors = (35 / 72) := by
  sorry

end probability_different_colors_l152_152447


namespace cubics_sum_div_abc_eq_three_l152_152597

theorem cubics_sum_div_abc_eq_three {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a + b + c = 0) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 :=
by
  sorry

end cubics_sum_div_abc_eq_three_l152_152597


namespace divisor_of_z_in_form_4n_minus_1_l152_152351

theorem divisor_of_z_in_form_4n_minus_1
  (x y : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (z : ℕ) 
  (hz : z = 4 * x * y / (x + y)) 
  (hz_odd : z % 2 = 1) :
  ∃ n : ℕ, n > 0 ∧ ∃ d : ℕ, d ∣ z ∧ d = 4 * n - 1 :=
sorry

end divisor_of_z_in_form_4n_minus_1_l152_152351


namespace count_convex_cyclic_quadrilaterals_is_1505_l152_152290

-- Define a quadrilateral using its sides
structure Quadrilateral where
  a b c d : ℕ
  deriving Repr, DecidableEq

-- Define what a convex cyclic quadrilateral is, given the integer sides and the perimeter condition
def isConvexCyclicQuadrilateral (q : Quadrilateral) : Prop :=
 q.a + q.b + q.c + q.d = 36 ∧
 q.a > 0 ∧ q.b > 0 ∧ q.c > 0 ∧ q.d > 0 ∧
 q.a + q.b > q.c + q.d ∧ q.c + q.d > q.a + q.b ∧
 q.b + q.c > q.d + q.a ∧ q.d + q.a > q.b + q.c

-- Noncomputable definition to count all convex cyclic quadrilaterals
noncomputable def countConvexCyclicQuadrilaterals : ℕ :=
  sorry

-- The theorem stating the count is equal to 1505
theorem count_convex_cyclic_quadrilaterals_is_1505 :
  countConvexCyclicQuadrilaterals = 1505 :=
  sorry

end count_convex_cyclic_quadrilaterals_is_1505_l152_152290


namespace parabola_focus_coordinates_l152_152287

open Real

theorem parabola_focus_coordinates (x y : ℝ) (h : y^2 = 6 * x) : (x, y) = (3 / 2, 0) :=
  sorry

end parabola_focus_coordinates_l152_152287


namespace determine_knights_l152_152238

noncomputable def number_of_knights (total_travelers: ℕ) (vasily_is_liar: Prop) (statement_by_vasily: ∀ room: ℕ, (more_liars room ∨ more_knights room) = false) : ℕ := 9

theorem determine_knights :
  ∀ (travelers: ℕ)
    (liar_iff_false: ∀ (P: Prop), liar P ↔ P = false)
    (vasily: traveler)
    (rooms: fin 3 → fin 16)
    (more_liars: Π (r: fin 3), Prop)
    (more_knights: Π (r: fin 3), Prop),
    travelers = 16 →
    liar (more_liars (rooms 0)) ∧ liar (more_knights (rooms 0)) ∧
    liar (more_liars (rooms 1)) ∧ liar (more_knights (rooms 1)) ∧
    liar (more_liars (rooms 2)) ∧ liar (more_knights (rooms 2)) →
    ∃ (k l: ℕ),
      k + l = 15 ∧ k - l = 1 ∧ k = 9 :=
begin
  sorry
end

end determine_knights_l152_152238


namespace prove_b_is_neg_two_l152_152702

-- Define the conditions
variables (b : ℝ)

-- Hypothesis: The real and imaginary parts of the complex number (2 - b * I) * I are opposites
def complex_opposite_parts (b : ℝ) : Prop :=
  b = -2

-- The theorem statement
theorem prove_b_is_neg_two : complex_opposite_parts b :=
sorry

end prove_b_is_neg_two_l152_152702


namespace total_time_spent_l152_152718

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2
def total_time : ℕ := outlining_time + writing_time + practicing_time

theorem total_time_spent : total_time = 117 := by
  sorry

end total_time_spent_l152_152718


namespace tom_seashells_left_l152_152223

def initial_seashells : ℕ := 5
def given_away_seashells : ℕ := 2

theorem tom_seashells_left : (initial_seashells - given_away_seashells) = 3 :=
by
  sorry

end tom_seashells_left_l152_152223


namespace maximize_profit_l152_152968

noncomputable def g (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then 13.5 - (1 / 30) * x^2
else if h : (x > 10) then 168 / x - 2000 / (3 * x^2)
else 0 -- default case included for totality

noncomputable def y (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then 8.1 * x - (1 / 30) * x^3 - 20
else if h : (x > 10) then 148 - 2 * (1000 / (3 * x) + 2.7 * x)
else 0 -- default case included for totality

theorem maximize_profit (x : ℝ) : 0 < x → y 9 = 28.6 :=
by sorry

end maximize_profit_l152_152968


namespace gcd_2023_1991_l152_152668

theorem gcd_2023_1991 : Nat.gcd 2023 1991 = 1 :=
by
  sorry

end gcd_2023_1991_l152_152668


namespace problem_1_problem_2_l152_152335

-- Define the given conditions
variables (a c : ℝ) (cosB : ℝ)
variables (b : ℝ) (S : ℝ)

-- Assuming the values for the variables
axiom h₁ : a = 4
axiom h₂ : c = 3
axiom h₃ : cosB = 1 / 8

-- Prove that b = sqrt(22)
theorem problem_1 : b = Real.sqrt 22 := by
  sorry

-- Prove that the area of triangle ABC is 9 * sqrt(7) / 4
theorem problem_2 : S = 9 * Real.sqrt 7 / 4 := by 
  sorry

end problem_1_problem_2_l152_152335


namespace perfect_square_divisors_count_450_l152_152118

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l152_152118


namespace find_f_five_thirds_l152_152875

variable {R : Type*} [LinearOrderedField R]

-- Define the odd function and its properties
variable {f : R → R}
variable (oddf : ∀ x : R, f (-x) = -f x)
variable (propf : ∀ x : R, f (1 + x) = f (-x))
variable (val : f (- (1 / 3 : R)) = 1 / 3)

theorem find_f_five_thirds : f (5 / 3 : R) = 1 / 3 := by
  sorry

end find_f_five_thirds_l152_152875


namespace probability_of_at_least_two_same_rank_approx_l152_152747

noncomputable def probability_at_least_two_same_rank (cards_drawn : ℕ) (total_cards : ℕ) : ℝ :=
  let ranks := 13
  let different_ranks_comb := Nat.choose ranks cards_drawn
  let rank_suit_combinations := different_ranks_comb * (4 ^ cards_drawn)
  let total_combinations := Nat.choose total_cards cards_drawn
  let p_complement := rank_suit_combinations / total_combinations
  1 - p_complement

theorem probability_of_at_least_two_same_rank_approx (h : 5 ≤ 52) : 
  abs (probability_at_least_two_same_rank 5 52 - 0.49) < 0.01 := 
by
  sorry

end probability_of_at_least_two_same_rank_approx_l152_152747


namespace num_integers_with_properties_l152_152671

theorem num_integers_with_properties : 
  ∃ (count : ℕ), count = 6 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 3400 ∧ (n % 34 = 0) ∧ 
             ((∃ (odd_divisors : ℕ), odd_divisors = (filter (λ d, d % 2 = 1) (n.divisors)) ∧ odd_divisors.length = 2) →
             (count = 6)) :=
begin
  sorry
end

end num_integers_with_properties_l152_152671


namespace isosceles_triangle_perimeter_l152_152574

theorem isosceles_triangle_perimeter (a b : ℕ) (c : ℕ) 
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 2) 
  (isosceles : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a) : 
  a + b + c = 12 :=
  sorry

end isosceles_triangle_perimeter_l152_152574


namespace chairs_per_row_l152_152675

theorem chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) 
  (h_total_chairs : total_chairs = 432) (h_num_rows : num_rows = 27) : 
  total_chairs / num_rows = 16 :=
by
  sorry

end chairs_per_row_l152_152675


namespace convex_polygon_interior_angle_l152_152374

theorem convex_polygon_interior_angle (n : ℕ) (h1 : 3 ≤ n)
  (h2 : (n - 2) * 180 = 2570 + x) : x = 130 :=
sorry

end convex_polygon_interior_angle_l152_152374


namespace division_theorem_l152_152184

theorem division_theorem (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y^(y^2) - 2 * y^(y + 1) + 1) := 
by
  sorry

end division_theorem_l152_152184


namespace sum_of_subsets_with_3_elements_l152_152724

open Finset

def P : Finset ℕ := {1, 3, 5, 7}

def subsets_of_P_with_3_elements : Finset (Finset ℕ) := P.powerset.filter (λ s, s.card = 3)

def sum_of_elements (s : Finset ℕ) : ℕ := s.sum id

def sum_of_sums_of_subsets (s : Finset (Finset ℕ)) : ℕ := s.sum sum_of_elements

theorem sum_of_subsets_with_3_elements :
  sum_of_sums_of_subsets subsets_of_P_with_3_elements = 48 :=
by
  sorry

end sum_of_subsets_with_3_elements_l152_152724


namespace original_population_l152_152236

theorem original_population (P : ℕ) (h1 : 0.1 * (P : ℝ) + 0.2 * (0.9 * P) = 4500) : P = 6250 :=
sorry

end original_population_l152_152236


namespace largest_tile_side_length_l152_152640

theorem largest_tile_side_length (w l : ℕ) (hw : w = 120) (hl : l = 96) : 
  ∃ s, s = Nat.gcd w l ∧ s = 24 :=
by
  sorry

end largest_tile_side_length_l152_152640


namespace prism_visibility_percentage_l152_152070

theorem prism_visibility_percentage
  (base_edge : ℝ)
  (height : ℝ)
  (cell_side : ℝ)
  (wraps : ℕ)
  (lateral_surface_area : ℝ)
  (transparent_area : ℝ) :
  base_edge = 3.2 →
  height = 5 →
  cell_side = 1 →
  wraps = 2 →
  lateral_surface_area = base_edge * height * 3 →
  transparent_area = 13.8 →
  (transparent_area / lateral_surface_area) * 100 = 28.75 :=
by
  intros h_base_edge h_height h_cell_side h_wraps h_lateral_surface_area h_transparent_area
  sorry

end prism_visibility_percentage_l152_152070


namespace arithmetic_sequence_common_difference_l152_152096

theorem arithmetic_sequence_common_difference 
  (a : Nat → Int)
  (a1 : a 1 = 5)
  (a6_a8_sum : a 6 + a 8 = 58) :
  ∃ d, ∀ n, a n = 5 + (n - 1) * d ∧ d = 4 := 
by 
  sorry

end arithmetic_sequence_common_difference_l152_152096


namespace train_length_and_speed_l152_152175

theorem train_length_and_speed (L_bridge : ℕ) (t_cross : ℕ) (t_on_bridge : ℕ) (L_train : ℕ) (v_train : ℕ)
  (h_bridge : L_bridge = 1000)
  (h_t_cross : t_cross = 60)
  (h_t_on_bridge : t_on_bridge = 40)
  (h_crossing_eq : (L_bridge + L_train) / t_cross = v_train)
  (h_on_bridge_eq : L_bridge / t_on_bridge = v_train) : 
  L_train = 200 ∧ v_train = 20 := 
  by
  sorry

end train_length_and_speed_l152_152175


namespace line_of_intersection_l152_152082

theorem line_of_intersection (x y z : ℝ) :
  (2 * x + 3 * y + 3 * z - 9 = 0) ∧ (4 * x + 2 * y + z - 8 = 0) →
  ((x / 4.5 + y / 3 + z / 3 = 1) ∧ (x / 2 + y / 4 + z / 8 = 1)) :=
by
  sorry

end line_of_intersection_l152_152082


namespace probability_of_desired_event_l152_152438

-- The problem condition: A fair six-sided die
def fair_six_sided_die : finset ℕ := {1, 2, 3, 4, 5, 6}

-- Roll the die five times
def roll_die_five_times : finset (fin 6 → ℕ) :=
  finset.pi_finset (fin 5) (λ _, fair_six_sided_die)

-- Event: Rolling a 1 at least twice
def at_least_two_ones (rolls : fin 5 → ℕ) : Prop :=
  (finset.filter (λ i, rolls i = 1) finset.univ).card ≥ 2

-- Event: Rolling a 2 at least twice
def at_least_two_twos (rolls : fin 5 → ℕ) : Prop :=
  (finset.filter (λ i, rolls i = 2) finset.univ).card ≥ 2

-- Event: Rolling one other number
def exactly_one_other_number (rolls : fin 5 → ℕ) : Prop :=
  (finset.filter (λ i, rolls i ≠ 1 ∧ rolls i ≠ 2) finset.univ).card = 1

-- Complete event: Rolling the number 1 at least twice and the number 2 at least twice
def desired_event (rolls : fin 5 → ℕ) : Prop :=
  at_least_two_ones rolls ∧ at_least_two_twos rolls ∧ exactly_one_other_number rolls

-- The measure for uniform probability space
noncomputable def uniform_measure : measure (fin 5 → ℕ) :=
  measure.pi 5 (λ _, measure_uniform (set.to_finite {1, 2, 3, 4, 5, 6}))

-- The statement of the probability
theorem probability_of_desired_event :
  (uniform_measure {rolls : fin 5 → ℕ | desired_event rolls} / uniform_measure {rolls : fin 5 → ℕ | true}) = 5 / 324 :=
by 
  sorry

end probability_of_desired_event_l152_152438


namespace num_students_left_l152_152942

variable (Joe_weight : ℝ := 45)
variable (original_avg_weight : ℝ := 30)
variable (new_avg_weight : ℝ := 31)
variable (final_avg_weight : ℝ := 30)
variable (diff_avg_weight : ℝ := 7.5)

theorem num_students_left (n : ℕ) (x : ℕ) (W : ℝ := n * original_avg_weight)
  (new_W : ℝ := W + Joe_weight) (A : ℝ := Joe_weight - diff_avg_weight) : 
  new_W = (n + 1) * new_avg_weight →
  W + Joe_weight - x * A = (n + 1 - x) * final_avg_weight →
  x = 2 :=
by
  sorry

end num_students_left_l152_152942


namespace gabby_needs_more_money_l152_152832

theorem gabby_needs_more_money (cost_saved : ℕ) (initial_saved : ℕ) (additional_money : ℕ) (cost_remaining : ℕ) :
  cost_saved = 65 → initial_saved = 35 → additional_money = 20 → cost_remaining = (cost_saved - initial_saved) - additional_money → cost_remaining = 10 :=
by
  intros h_cost_saved h_initial_saved h_additional_money h_cost_remaining
  simp [h_cost_saved, h_initial_saved, h_additional_money] at h_cost_remaining
  exact h_cost_remaining

end gabby_needs_more_money_l152_152832


namespace triangle_shape_l152_152446

open Float

theorem triangle_shape (A B C : ℝ) (a b c : ℝ) (h1 : sin A / sin B = 5 / 11) (h2 : sin B / sin C = 11 / 13) (h3 : ∀ (t : ℝ), t ≠ 0 → a = 5 * t ∧ b = 11 * t ∧ c = 13 * t) : A + B + C = π → cos C < 0 :=
sorry

end triangle_shape_l152_152446


namespace slant_height_of_cone_l152_152031

theorem slant_height_of_cone
  (r : ℝ) (CSA : ℝ) (l : ℝ)
  (hr : r = 14)
  (hCSA : CSA = 1539.3804002589986) :
  CSA = Real.pi * r * l → l = 35 := 
sorry

end slant_height_of_cone_l152_152031


namespace initial_apples_l152_152903

theorem initial_apples (Initially_Apples : ℕ) (Added_Apples : ℕ) (Total_Apples : ℕ)
  (h1 : Added_Apples = 8) (h2 : Total_Apples = 17) : Initially_Apples = 9 :=
by
  have h3 : Added_Apples + Initially_Apples = Total_Apples := by
    sorry
  linarith

end initial_apples_l152_152903


namespace estimated_value_of_y_l152_152971

theorem estimated_value_of_y (x : ℝ) (h : x = 25) : 
  let y := 0.50 * x - 0.81 in
  y = 11.69 :=
by
  rw [h]
  let y := 0.50 * 25 - 0.81
  sorry

end estimated_value_of_y_l152_152971


namespace total_pigs_indeterminate_l152_152020

noncomputable def average_weight := 15
def underweight_threshold := 16
def max_underweight_pigs := 4

theorem total_pigs_indeterminate :
  ∃ (P U : ℕ), U ≤ max_underweight_pigs ∧ (average_weight = 15) → P = P :=
sorry

end total_pigs_indeterminate_l152_152020


namespace investment_inequality_l152_152740

-- Defining the initial investment
def initial_investment : ℝ := 200

-- Year 1 changes
def alpha_year1 := initial_investment * 1.30
def beta_year1 := initial_investment * 0.80
def gamma_year1 := initial_investment * 1.10
def delta_year1 := initial_investment * 0.90

-- Year 2 changes
def alpha_final := alpha_year1 * 0.85
def beta_final := beta_year1 * 1.30
def gamma_final := gamma_year1 * 0.95
def delta_final := delta_year1 * 1.20

-- Prove the final inequality
theorem investment_inequality : beta_final < gamma_final ∧ gamma_final < delta_final ∧ delta_final < alpha_final :=
by {
  sorry
}

end investment_inequality_l152_152740


namespace egorov_theorem_l152_152897

open ProbabilityTheory

theorem egorov_theorem {Ω : Type*} {ℱ : MeasurableSpace Ω} (μ : MeasureTheory.Measure Ω) 
  {ξ : Ω → ℝ} {ξₙ : ℕ → (Ω → ℝ)} 
  (h_ξₙ_to_ξ_a.s : ∀ᵐ ω ∂μ, Filter.Tendsto (λ n, ξₙ n ω) Filter.atTop (𝓝 (ξ ω))) :
  ∀ ε > 0, ∃ A ∈ ℱ, μ Aᶜ ≤ ε ∧ ∀ ε' > 0, ∃ N : ℕ, ∀ n ≥ N, ∀ ω ∈ A, abs (ξₙ n ω - ξ ω) ≤ ε' :=
begin
  sorry
end

end egorov_theorem_l152_152897


namespace bryan_travel_ratio_l152_152000

theorem bryan_travel_ratio
  (walk_time : ℕ)
  (bus_time : ℕ)
  (evening_walk_time : ℕ)
  (total_travel_hours : ℕ)
  (days_per_year : ℕ)
  (minutes_per_hour : ℕ)
  (minutes_total : ℕ)
  (daily_travel_time : ℕ) :
  walk_time = 5 →
  bus_time = 20 →
  evening_walk_time = 5 →
  total_travel_hours = 365 →
  days_per_year = 365 →
  minutes_per_hour = 60 →
  minutes_total = total_travel_hours * minutes_per_hour →
  daily_travel_time = (walk_time + bus_time + evening_walk_time) * 2 →
  (minutes_total / daily_travel_time = days_per_year) →
  (walk_time + bus_time + evening_walk_time) = daily_travel_time / 2 →
  (walk_time + bus_time + evening_walk_time) = daily_travel_time / 2 :=
by
  intros
  sorry

end bryan_travel_ratio_l152_152000


namespace joelle_initial_deposit_l152_152872

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

end joelle_initial_deposit_l152_152872


namespace number_of_valid_integers_l152_152669

def has_two_odd_divisors (n : ℕ) : Prop :=
  (Nat.divisors n).filter (λ x, x % 2 = 1).length = 2

def is_multiple_of_34 (n : ℕ) : Prop :=
  n % 34 = 0

def count_valid_numbers : ℕ :=
  (Finset.range (3400 + 1)).filter (λ n, is_multiple_of_34 n ∧ has_two_odd_divisors n).card

theorem number_of_valid_integers : count_valid_numbers = 6 :=
by
  sorry

end number_of_valid_integers_l152_152669


namespace real_roots_of_f_l152_152672

noncomputable def f (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem real_roots_of_f :
  {x | f x = 0} = {-1, 1, 2, 3} :=
sorry

end real_roots_of_f_l152_152672


namespace john_weekly_earnings_increase_l152_152932

theorem john_weekly_earnings_increase (original_earnings new_earnings : ℕ) 
  (h₀ : original_earnings = 60) 
  (h₁ : new_earnings = 72) : 
  ((new_earnings - original_earnings) / original_earnings) * 100 = 20 :=
by
  sorry

end john_weekly_earnings_increase_l152_152932


namespace nn_gt_n1n1_l152_152609

theorem nn_gt_n1n1 (n : ℕ) (h : n > 1) : n^n > (n + 1)^(n - 1) := 
sorry

end nn_gt_n1n1_l152_152609


namespace find_f_five_thirds_l152_152886

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = - f x
def functional_equation (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (-x)
def specific_value (f : ℝ → ℝ) := f (-1/3) = 1/3

theorem find_f_five_thirds
  (hf_odd : odd_function f)
  (hf_fe : functional_equation f)
  (hf_value : specific_value f) :
  f (5 / 3) = 1 / 3 := by
  sorry

end find_f_five_thirds_l152_152886


namespace trig_identity_proof_l152_152418

theorem trig_identity_proof : 
  (\sin (20 * real.pi / 180) * \cos (10 * real.pi / 180) + \cos (160 * real.pi / 180) * \cos (110 * real.pi / 180)) / 
  (\sin (24 * real.pi / 180) * \cos (6 * real.pi / 180) + \cos (156 * real.pi / 180) * \cos (96 * real.pi / 180)) =
  (1 - \sin (40 * real.pi / 180)) / (1 - \sin (48 * real.pi / 180)) :=
by
  sorry

end trig_identity_proof_l152_152418


namespace number_of_girls_l152_152216

theorem number_of_girls (classes : ℕ) (students_per_class : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : classes = 4) 
  (h2 : students_per_class = 25) 
  (h3 : boys = 56) 
  (h4 : girls = (classes * students_per_class) - boys) : 
  girls = 44 :=
by
  sorry

end number_of_girls_l152_152216


namespace complex_expression_l152_152076

theorem complex_expression (i : ℂ) (h₁ : i^2 = -1) (h₂ : i^4 = 1) :
  (i + i^3)^100 + (i + i^2 + i^3 + i^4 + i^5)^120 = 1 := by
  sorry

end complex_expression_l152_152076


namespace combined_resistance_l152_152861

theorem combined_resistance (x y : ℝ) (r : ℝ) (hx : x = 4) (hy : y = 6) :
  (1 / r) = (1 / x) + (1 / y) → r = 12 / 5 :=
by
  sorry

end combined_resistance_l152_152861


namespace profit_percent_l152_152053

variable {P C : ℝ}

theorem profit_percent (h1: 2 / 3 * P = 0.82 * C) : ((P - C) / C) * 100 = 23 := by
  have h2 : C = (2 / 3 * P) / 0.82 := by sorry
  have h3 : (P - C) / C = (P - (2 / 3 * P) / 0.82) / ((2 / 3 * P) / 0.82) := by sorry
  have h4 : (P - (2 / 3 * P) / 0.82) / ((2 / 3 * P) / 0.82) = (0.82 * P - 2 / 3 * P) / (2 / 3 * P) := by sorry
  have h5 : (0.82 * P - 2 / 3 * P) / (2 / 3 * P) = 0.1533 := by sorry
  have h6 : 0.1533 * 100 = 23 := by sorry
  sorry

end profit_percent_l152_152053


namespace time_interval_for_7_students_l152_152357

-- Definitions from conditions
def students_per_ride : ℕ := 7
def total_students : ℕ := 21
def total_time : ℕ := 15

-- Statement of the problem
theorem time_interval_for_7_students : (total_time / (total_students / students_per_ride)) = 5 := 
by sorry

end time_interval_for_7_students_l152_152357


namespace first_two_cards_black_prob_l152_152792

noncomputable def probability_first_two_black : ℚ :=
  let total_cards := 52
  let black_cards := 26
  let first_draw_prob := black_cards / total_cards
  let second_draw_prob := (black_cards - 1) / (total_cards - 1)
  first_draw_prob * second_draw_prob

theorem first_two_cards_black_prob :
  probability_first_two_black = 25 / 102 :=
by
  sorry

end first_two_cards_black_prob_l152_152792


namespace sum_c_2017_l152_152095

def a (n : ℕ) : ℕ := 3 * n + 1

def b (n : ℕ) : ℕ := 4^(n-1)

def c (n : ℕ) : ℕ := if n = 1 then 7 else 3 * 4^(n-1)

theorem sum_c_2017 : (Finset.range 2017).sum c = 4^2017 + 3 :=
by
  -- definitions and required assumptions
  sorry

end sum_c_2017_l152_152095


namespace value_of_linear_combination_l152_152059

theorem value_of_linear_combination :
  ∀ (x1 x2 x3 x4 x5 : ℝ),
    2*x1 + x2 + x3 + x4 + x5 = 6 →
    x1 + 2*x2 + x3 + x4 + x5 = 12 →
    x1 + x2 + 2*x3 + x4 + x5 = 24 →
    x1 + x2 + x3 + 2*x4 + x5 = 48 →
    x1 + x2 + x3 + x4 + 2*x5 = 96 →
    3*x4 + 2*x5 = 181 :=
by
  intros x1 x2 x3 x4 x5 h1 h2 h3 h4 h5
  sorry

end value_of_linear_combination_l152_152059


namespace smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6_l152_152925

theorem smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6 : 
  ∃ n : ℕ, (∃ k : ℕ, n = 60 * k + 1) ∧ n % 9 = 0 ∧ ∀ m : ℕ, (∃ k' : ℕ, m = 60 * k' + 1) ∧ m % 9 = 0 → n ≤ m :=
by
  sorry

end smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6_l152_152925


namespace paco_ate_sweet_cookies_l152_152896

noncomputable def PacoCookies (sweet: Nat) (salty: Nat) (salty_eaten: Nat) (extra_sweet: Nat) : Nat :=
  let corrected_salty_eaten := if salty_eaten > salty then salty else salty_eaten
  corrected_salty_eaten + extra_sweet

theorem paco_ate_sweet_cookies : PacoCookies 39 6 23 9 = 15 := by
  sorry

end paco_ate_sweet_cookies_l152_152896


namespace altered_solution_ratio_l152_152984

variable (b d w : ℕ)
variable (b' d' w' : ℕ)
variable (ratio_orig_bd_ratio_orig_dw_ratio_orig_bw : Rat)
variable (ratio_new_bd_ratio_new_dw_ratio_new_bw : Rat)

noncomputable def orig_ratios (ratio_orig_bd ratio_orig_bw : Rat) (d w : ℕ) : Prop := 
    ratio_orig_bd = 2 / 40 ∧ ratio_orig_bw = 40 / 100

noncomputable def new_ratios (ratio_new_bd : Rat) (d' : ℕ) : Prop :=
    ratio_new_bd = 6 / 40 ∧ d' = 60

noncomputable def new_solution (w' : ℕ) : Prop :=
    w' = 300

theorem altered_solution_ratio : 
    ∀ (orig_ratios: Prop) (new_ratios: Prop) (new_solution: Prop),
    orig_ratios ∧ new_ratios ∧ new_solution →
    (d' / w = 2 / 5) :=
by
    sorry

end altered_solution_ratio_l152_152984


namespace remainder_mod_8_l152_152229

theorem remainder_mod_8 (x : ℤ) (h : x % 63 = 25) : x % 8 = 1 := 
sorry

end remainder_mod_8_l152_152229


namespace number_of_perfect_square_divisors_of_450_l152_152151

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l152_152151


namespace a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b_l152_152642

noncomputable def T_a : ℝ := 7.5
noncomputable def T_b : ℝ := 10
noncomputable def rounds_a (n : ℕ) : ℝ := n * T_a
noncomputable def rounds_b (n : ℕ) : ℝ := n * T_b

theorem a_beats_b_by_one_round_in_4_round_race :
  rounds_a 4 = rounds_b 3 := by
  sorry

theorem a_beats_b_by_T_a_minus_T_b :
  T_b - T_a = 2.5 := by
  sorry

end a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b_l152_152642


namespace sum_lcms_equals_l152_152051

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_lcms_equals :
  ( ∑ ν in (Finset.filter is_solution (Finset.range 100)), ν ) = 180 :=
sorry

end sum_lcms_equals_l152_152051


namespace sum_of_common_ratios_l152_152731

theorem sum_of_common_ratios (k p r : ℝ) (h₁ : k ≠ 0) (h₂ : p ≠ r) (h₃ : (k * (p ^ 2)) - (k * (r ^ 2)) = 4 * (k * p - k * r)) : 
  p + r = 4 :=
by
  -- Using the conditions provided, we can prove the sum of the common ratios is 4.
  sorry

end sum_of_common_ratios_l152_152731


namespace xy_cubed_identity_l152_152309

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l152_152309


namespace chocolates_sold_at_selling_price_l152_152700
noncomputable def chocolates_sold (C S : ℝ) (n : ℕ) : Prop :=
  (35 * C = n * S) ∧ ((S - C) / C * 100) = 66.67

theorem chocolates_sold_at_selling_price : ∃ n : ℕ, ∀ C S : ℝ,
  chocolates_sold C S n → n = 21 :=
by
  sorry

end chocolates_sold_at_selling_price_l152_152700


namespace problem_statement_l152_152685

theorem problem_statement (m : ℝ) (h : m^2 - m - 2 = 0) : m^2 - m + 2023 = 2025 :=
sorry

end problem_statement_l152_152685


namespace thirteen_pow_2011_mod_100_l152_152768

theorem thirteen_pow_2011_mod_100 : (13^2011) % 100 = 37 := by
  sorry

end thirteen_pow_2011_mod_100_l152_152768


namespace flags_left_l152_152377

theorem flags_left (interval circumference : ℕ) (total_flags : ℕ) (h1 : interval = 20) (h2 : circumference = 200) (h3 : total_flags = 12) : 
  total_flags - (circumference / interval) = 2 := 
by 
  -- Using the conditions h1, h2, h3
  sorry

end flags_left_l152_152377


namespace lucy_withdrawal_l152_152734

-- Given conditions
def initial_balance : ℕ := 65
def deposit : ℕ := 15
def final_balance : ℕ := 76

-- Define balance before withdrawal
def balance_before_withdrawal := initial_balance + deposit

-- Theorem to prove
theorem lucy_withdrawal : balance_before_withdrawal - final_balance = 4 :=
by sorry

end lucy_withdrawal_l152_152734


namespace correct_multiplier_l152_152784

theorem correct_multiplier (x : ℕ) 
  (h1 : 137 * 34 + 1233 = 137 * x) : 
  x = 43 := 
by 
  sorry

end correct_multiplier_l152_152784


namespace measure_of_one_exterior_angle_l152_152069

theorem measure_of_one_exterior_angle (n : ℕ) (h : n > 2) : 
  n > 2 → ∃ (angle : ℝ), angle = 360 / n :=
by 
  sorry

end measure_of_one_exterior_angle_l152_152069


namespace sum_of_reciprocals_of_roots_l152_152824

theorem sum_of_reciprocals_of_roots : 
  ∀ {r1 r2 : ℝ}, (r1 + r2 = 14) → (r1 * r2 = 6) → (1 / r1 + 1 / r2 = 7 / 3) :=
by
  intros r1 r2 h_sum h_product
  sorry

end sum_of_reciprocals_of_roots_l152_152824


namespace problem_solution_l152_152860

theorem problem_solution (a : ℚ) (h : 3 * a + 6 * a / 4 = 6) : a = 4 / 3 :=
by
  sorry

end problem_solution_l152_152860


namespace birdhouse_flight_distance_l152_152629

variable (car_distance : ℕ)
variable (lawn_chair_distance : ℕ)
variable (birdhouse_distance : ℕ)

def problem_condition1 := car_distance = 200
def problem_condition2 := lawn_chair_distance = 2 * car_distance
def problem_condition3 := birdhouse_distance = 3 * lawn_chair_distance

theorem birdhouse_flight_distance
  (h1 : car_distance = 200)
  (h2 : lawn_chair_distance = 2 * car_distance)
  (h3 : birdhouse_distance = 3 * lawn_chair_distance) :
  birdhouse_distance = 1200 := by
  sorry

end birdhouse_flight_distance_l152_152629


namespace caller_wins_both_at_35_l152_152232

theorem caller_wins_both_at_35 (n : ℕ) :
  ∀ n, (n % 5 = 0 ∧ n % 7 = 0) ↔ n = 35 :=
by
  sorry

end caller_wins_both_at_35_l152_152232


namespace base_6_to_base_10_exact_value_l152_152253

def base_6_to_base_10 (n : ℕ) : ℕ :=
  1 * 6^2 + 5 * 6^1 + 4 * 6^0

theorem base_6_to_base_10_exact_value : base_6_to_base_10 154 = 70 := by
  rfl

end base_6_to_base_10_exact_value_l152_152253


namespace symmetric_points_l152_152161

theorem symmetric_points (a b : ℤ) (h1 : (a, -2) = (1, -2)) (h2 : (-1, b) = (-1, -2)) :
  (a + b) ^ 2023 = -1 := by
  -- We know from the conditions:
  -- (a, -2) and (1, -2) implies a = 1
  -- (-1, b) and (-1, -2) implies b = -2
  -- Thus it follows that:
  sorry

end symmetric_points_l152_152161


namespace heather_lighter_than_combined_weights_l152_152111

noncomputable def heather_weight : ℝ := 87.5
noncomputable def emily_weight : ℝ := 45.3
noncomputable def elizabeth_weight : ℝ := 38.7
noncomputable def george_weight : ℝ := 56.9

theorem heather_lighter_than_combined_weights :
  heather_weight - (emily_weight + elizabeth_weight + george_weight) = -53.4 :=
by 
  sorry

end heather_lighter_than_combined_weights_l152_152111


namespace value_of_a_l152_152348

variable (a : ℤ)
def U : Set ℤ := {2, 4, 3 - a^2}
def P : Set ℤ := {2, a^2 + 2 - a}

theorem value_of_a (h : (U a) \ (P a) = {-1}) : a = 2 :=
sorry

end value_of_a_l152_152348


namespace xy_cubed_identity_l152_152310

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l152_152310


namespace roots_of_quadratic_l152_152279

theorem roots_of_quadratic (α β : ℝ) (h1 : α^2 - 4*α - 5 = 0) (h2 : β^2 - 4*β - 5 = 0) :
  3*α^4 + 10*β^3 = 2593 := 
by
  sorry

end roots_of_quadratic_l152_152279


namespace find_triples_solution_l152_152820

theorem find_triples_solution (x y z : ℕ) (h : x^5 + x^4 + 1 = 3^y * 7^z) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 2) :=
by
  sorry

end find_triples_solution_l152_152820


namespace eq_d_is_quadratic_l152_152252

def is_quadratic (eq : ℕ → ℤ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ eq 2 = a ∧ eq 1 = b ∧ eq 0 = c

def eq_cond_1 (n : ℕ) : ℤ :=
  match n with
  | 2 => 1  -- x^2 coefficient
  | 1 => 0  -- x coefficient
  | 0 => -1 -- constant term
  | _ => 0

theorem eq_d_is_quadratic : is_quadratic eq_cond_1 :=
  sorry

end eq_d_is_quadratic_l152_152252


namespace probability_incorrect_pairs_l152_152225

theorem probability_incorrect_pairs 
  (k : ℕ) (h_k : k < 6)
  : let m := 7
    let n := 72
    m + n = 79 :=
by
  sorry

end probability_incorrect_pairs_l152_152225


namespace range_of_a_l152_152827

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := 
by
  sorry

end range_of_a_l152_152827


namespace parallel_condition_l152_152934

theorem parallel_condition (a : ℝ) : (a = -2) ↔ (∀ x y : ℝ, ax + 2 * y = 0 → (-a / 2) = 1) :=
by
  sorry

end parallel_condition_l152_152934


namespace perpendicular_lines_parallel_l152_152423

noncomputable def line := Type
noncomputable def plane := Type

variables (m n : line) (α : plane)

def parallel (l1 l2 : line) : Prop := sorry -- Definition of parallel lines
def perpendicular (l : line) (α : plane) : Prop := sorry -- Definition of perpendicular line to a plane

theorem perpendicular_lines_parallel (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end perpendicular_lines_parallel_l152_152423


namespace automobile_travel_distance_l152_152399

theorem automobile_travel_distance
  (a r : ℝ) : 
  let feet_per_yard := 3
  let seconds_per_minute := 60
  let travel_feet := a / 4
  let travel_seconds := 2 * r
  let rate_yards_per_second := (travel_feet / travel_seconds) / feet_per_yard
  let total_seconds := 10 * seconds_per_minute
  let total_yards := rate_yards_per_second * total_seconds
  total_yards = 25 * a / r := by
  sorry

end automobile_travel_distance_l152_152399


namespace math_problem_l152_152859

variable (x y : ℚ)

theorem math_problem (h : 1.5 * x = 0.04 * y) : (y - x) / (y + x) = 73 / 77 := by
  sorry

end math_problem_l152_152859


namespace perfect_square_factors_450_l152_152116

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l152_152116


namespace abc_inequality_l152_152179

variable {a b c : ℝ}

theorem abc_inequality (h₀ : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h₁ : a + b + c = 6)
  (h₂ : a * b + b * c + c * a = 9) :
  0 < a * b * c ∧ a * b * c < 4 := by
  sorry

end abc_inequality_l152_152179


namespace count_4_digit_multiples_of_5_is_9_l152_152435

noncomputable def count_4_digit_multiples_of_5 : Nat :=
  let digits := [2, 7, 4, 5]
  let last_digit := 5
  let remaining_digits := [2, 7, 4]
  let case_1 := 3
  let case_2 := 3 * 2
  case_1 + case_2

theorem count_4_digit_multiples_of_5_is_9 : count_4_digit_multiples_of_5 = 9 :=
by
  sorry

end count_4_digit_multiples_of_5_is_9_l152_152435


namespace radius_of_small_semicircle_l152_152170

theorem radius_of_small_semicircle
  (radius_big_semicircle : ℝ)
  (radius_inner_circle : ℝ) 
  (pairwise_tangent : ∀ x : ℝ, x = radius_big_semicircle → x = 12 ∧ 
                                x = radius_inner_circle → x = 6 ∧ 
                                true) :
  ∃ (r : ℝ), r = 4 :=
by 
  sorry

end radius_of_small_semicircle_l152_152170


namespace three_pos_reals_inequality_l152_152186

open Real

theorem three_pos_reals_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_eq : a + b + c = a^2 + b^2 + c^2) :
  ((a^2) / (a^2 + b * c) + (b^2) / (b^2 + c * a) + (c^2) / (c^2 + a * b)) ≥ (a + b + c) / 2 :=
by
  sorry

end three_pos_reals_inequality_l152_152186


namespace carols_father_gave_5_peanuts_l152_152804

theorem carols_father_gave_5_peanuts : 
  ∀ (c: ℕ) (f: ℕ), c = 2 → c + f = 7 → f = 5 :=
by
  intros c f h1 h2
  sorry

end carols_father_gave_5_peanuts_l152_152804


namespace set_D_forms_triangle_l152_152231

theorem set_D_forms_triangle (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 6) : a + b > c ∧ a + c > b ∧ b + c > a := by
  rw [h1, h2, h3]
  show 4 + 5 > 6 ∧ 4 + 6 > 5 ∧ 5 + 6 > 4
  sorry

end set_D_forms_triangle_l152_152231


namespace purely_imaginary_z_value_l152_152567

theorem purely_imaginary_z_value (a : ℝ) (h : (a^2 - a - 2) = 0 ∧ (a + 1) ≠ 0) : a = 2 :=
sorry

end purely_imaginary_z_value_l152_152567


namespace simplify_polynomial_l152_152610

theorem simplify_polynomial (y : ℝ) :
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + y ^ 10 + 2 * y ^ 9) =
  15 * y ^ 13 - y ^ 12 - 3 * y ^ 11 + 4 * y ^ 10 - 4 * y ^ 9 := 
by
  sorry

end simplify_polynomial_l152_152610


namespace cubic_sum_l152_152318

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l152_152318


namespace roots_of_eq_l152_152489

theorem roots_of_eq (x : ℝ) : (x - 1) * (x - 2) = 0 ↔ (x = 1 ∨ x = 2) := by
  sorry

end roots_of_eq_l152_152489


namespace Sheila_attendance_probability_l152_152478

-- Definitions as per given conditions
def P_rain := 0.5
def P_sunny := 0.3
def P_cloudy := 0.2
def P_Sheila_goes_given_rain := 0.3
def P_Sheila_goes_given_sunny := 0.7
def P_Sheila_goes_given_cloudy := 0.5

-- Define the probability calculation
def P_Sheila_attends := 
  (P_rain * P_Sheila_goes_given_rain) + 
  (P_sunny * P_Sheila_goes_given_sunny) + 
  (P_cloudy * P_Sheila_goes_given_cloudy)

-- Final theorem statement
theorem Sheila_attendance_probability : P_Sheila_attends = 0.46 := by
  sorry

end Sheila_attendance_probability_l152_152478


namespace each_child_consumes_3_bottles_per_day_l152_152937

noncomputable def bottles_per_child_per_day : ℕ :=
  let first_group := 14
  let second_group := 16
  let third_group := 12
  let fourth_group := (first_group + second_group + third_group) / 2
  let total_children := first_group + second_group + third_group + fourth_group
  let cases_of_water := 13
  let bottles_per_case := 24
  let initial_bottles := cases_of_water * bottles_per_case
  let additional_bottles := 255
  let total_bottles := initial_bottles + additional_bottles
  let bottles_per_child := total_bottles / total_children
  let days := 3
  bottles_per_child / days

theorem each_child_consumes_3_bottles_per_day :
  bottles_per_child_per_day = 3 :=
by
  sorry

end each_child_consumes_3_bottles_per_day_l152_152937


namespace fourth_grade_students_l152_152452

theorem fourth_grade_students (initial_students left_students new_students final_students : ℕ) 
    (h1 : initial_students = 33) 
    (h2 : left_students = 18) 
    (h3 : new_students = 14) 
    (h4 : final_students = initial_students - left_students + new_students) :
    final_students = 29 := 
by 
    sorry

end fourth_grade_students_l152_152452


namespace find_a3_l152_152097

def sequence_sum (n : ℕ) : ℕ := n^2 + n

def a (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

theorem find_a3 : a 3 = 6 := by
  sorry

end find_a3_l152_152097


namespace sixth_grade_percentage_combined_l152_152491

def maplewood_percentages := [10, 20, 15, 15, 10, 15, 15]
def brookside_percentages := [16, 14, 13, 12, 12, 18, 15]

def maplewood_students := 150
def brookside_students := 180

def sixth_grade_maplewood := maplewood_students * (maplewood_percentages.get! 6) / 100
def sixth_grade_brookside := brookside_students * (brookside_percentages.get! 6) / 100

def total_students := maplewood_students + brookside_students
def total_sixth_graders := sixth_grade_maplewood + sixth_grade_brookside

def sixth_grade_percentage := total_sixth_graders / total_students * 100

theorem sixth_grade_percentage_combined : sixth_grade_percentage = 15 := by 
  sorry

end sixth_grade_percentage_combined_l152_152491


namespace Courtney_total_marbles_l152_152410

theorem Courtney_total_marbles (first_jar second_jar third_jar : ℕ) 
  (h1 : first_jar = 80)
  (h2 : second_jar = 2 * first_jar)
  (h3 : third_jar = first_jar / 4) :
  first_jar + second_jar + third_jar = 260 := 
by
  sorry

end Courtney_total_marbles_l152_152410


namespace box_marbles_l152_152639

variable {A B : Type}
variable [FinType A] [FinType B]
variable (a b : ℕ) (total_marbles : a + b = 24)
variable (x y : ℕ) (black_prob : (x / a) * (y / b) = 28 / 45)

theorem box_marbles :
  ∃ m n : ℕ, gcd m n = 1 ∧ (prob_white : (m / a) * (n / b) = 2 / 135) ∧ (m + n = 137) :=
by sorry

end box_marbles_l152_152639


namespace intersection_of_A_and_B_l152_152732

def set_A : Set ℝ := {x | x^2 ≤ 4 * x}
def set_B : Set ℝ := {x | |x| ≥ 2}

theorem intersection_of_A_and_B :
  {x | x ∈ set_A ∧ x ∈ set_B} = {x | 2 ≤ x ∧ x ≤ 4} :=
sorry

end intersection_of_A_and_B_l152_152732


namespace find_f_five_thirds_l152_152876

variable {R : Type*} [LinearOrderedField R]

-- Define the odd function and its properties
variable {f : R → R}
variable (oddf : ∀ x : R, f (-x) = -f x)
variable (propf : ∀ x : R, f (1 + x) = f (-x))
variable (val : f (- (1 / 3 : R)) = 1 / 3)

theorem find_f_five_thirds : f (5 / 3 : R) = 1 / 3 := by
  sorry

end find_f_five_thirds_l152_152876


namespace distance_covered_l152_152646

-- Definitions
def speed : ℕ := 150  -- Speed in km/h
def time : ℕ := 8  -- Time in hours

-- Proof statement
theorem distance_covered : speed * time = 1200 := 
by
  sorry

end distance_covered_l152_152646


namespace abs_neg_three_l152_152483

theorem abs_neg_three : abs (-3) = 3 :=
sorry

end abs_neg_three_l152_152483


namespace find_principal_amount_l152_152793

noncomputable def principal_amount (SI R T : ℝ) : ℝ :=
  SI / (R * T / 100)

theorem find_principal_amount :
  principal_amount 4052.25 9 5 = 9005 := by
sorry

end find_principal_amount_l152_152793


namespace perfect_square_factors_count_450_l152_152124

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l152_152124


namespace range_of_a_l152_152689

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- State the problem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 2) → (a ≤ -1 ∨ a ≥ 3) :=
by 
  sorry

end range_of_a_l152_152689


namespace right_triangle_third_side_square_l152_152704

theorem right_triangle_third_side_square (a b : ℕ) (c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2) :
  c^2 = 28 ∨ c^2 = 100 :=
by { sorry }

end right_triangle_third_side_square_l152_152704


namespace machine_present_value_l152_152939

/-- A machine depreciates at a certain rate annually.
    Given the future value after a certain number of years and the depreciation rate,
    prove the present value of the machine. -/
theorem machine_present_value
  (depreciation_rate : ℝ := 0.25)
  (future_value : ℝ := 54000)
  (years : ℕ := 3)
  (pv : ℝ := 128000) :
  (future_value = pv * (1 - depreciation_rate) ^ years) :=
sorry

end machine_present_value_l152_152939


namespace perfect_square_trinomial_l152_152292

theorem perfect_square_trinomial (m : ℝ) (h : ∃ a : ℝ, x^2 + 2 * x + m = (x + a)^2) : m = 1 := 
sorry

end perfect_square_trinomial_l152_152292


namespace find_coordinates_of_D_l152_152425

theorem find_coordinates_of_D
  (A B C D : ℝ × ℝ)
  (hA : A = (-1, 2))
  (hB : B = (0, 0))
  (hC : C = (1, 7))
  (hParallelogram : ∃ u v, u * (B - A) + v * (C - D) = (0, 0) ∧ u * (C - D) + v * (B - A) = (0, 0)) :
  D = (0, 9) :=
sorry

end find_coordinates_of_D_l152_152425


namespace janet_saves_time_l152_152814

theorem janet_saves_time (looking_time_per_day : ℕ := 8) (complaining_time_per_day : ℕ := 3) (days_per_week : ℕ := 7) :
  (looking_time_per_day + complaining_time_per_day) * days_per_week = 77 := 
sorry

end janet_saves_time_l152_152814


namespace student_total_marks_l152_152862

theorem student_total_marks (total_questions correct_answers incorrect_answer_score correct_answer_score : ℕ)
    (h1 : total_questions = 60)
    (h2 : correct_answers = 38)
    (h3 : correct_answer_score = 4)
    (h4 : incorrect_answer_score = 1)
    (incorrect_answers := total_questions - correct_answers) 
    : (correct_answers * correct_answer_score - incorrect_answers * incorrect_answer_score) = 130 :=
by
  -- proof to be provided here
  sorry

end student_total_marks_l152_152862


namespace probability_negative_product_l152_152037

theorem probability_negative_product :
  let s := {-5, -8, 7, 4, -2, 1, 9} in
  ∃ p : ℚ, p = 4 / 7 ∧ 
  (∑ (x : ℤ) in s, ∑ (y : ℤ) in s, if x ≠ y ∧ x * y < 0 then 1 else 0) / 
  (∑ (x : ℤ) in s, ∑ (y : ℤ) in s, if x ≠ y then 1 else 0) = p := 
by
  -- Variables
  let s := {-5, -8, 7, 4, -2, 1, 9}
  -- Calculate total number of pairs
  have total_pairs := 7 * 6 / 2
  -- Calculate favorable pairs
  have favorable_pairs := 3 * 4
  -- Calculate probability
  have prob := favorable_pairs / total_pairs
  existsi (4 / 7 : ℚ)
  split
  · refl
  · field_simp
    norm_num
    sorry

end probability_negative_product_l152_152037


namespace big_bea_bananas_l152_152803

theorem big_bea_bananas :
  ∃ (b : ℕ), (b + (b + 8) + (b + 16) + (b + 24) + (b + 32) + (b + 40) + (b + 48) = 196) ∧ (b + 48 = 52) := by
  sorry

end big_bea_bananas_l152_152803


namespace perfect_square_divisors_of_450_l152_152134

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l152_152134


namespace car_travel_distance_l152_152520

-- The rate of the car traveling
def rate_of_travel (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- Convert hours to minutes
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

-- The distance covered
def distance_covered (rate : ℝ) (time : ℝ) : ℝ := rate * time

-- Main theorem statement to prove
theorem car_travel_distance : distance_covered (rate_of_travel 3 4) (hours_to_minutes 2) = 90 := sorry

end car_travel_distance_l152_152520


namespace color_opposite_orange_is_indigo_l152_152211

-- Define the colors
inductive Color
| O | B | Y | S | V | I

-- Define a structure representing a view of the cube
structure CubeView where
  top : Color
  front : Color
  right : Color

-- Given views
def view1 := CubeView.mk Color.B Color.Y Color.S
def view2 := CubeView.mk Color.B Color.V Color.S
def view3 := CubeView.mk Color.B Color.I Color.Y

-- The statement to be proved: the color opposite to orange (O) is indigo (I), given the views
theorem color_opposite_orange_is_indigo (v1 v2 v3 : CubeView) :
  v1 = view1 →
  v2 = view2 →
  v3 = view3 →
  ∃ opposite_color : Color, opposite_color = Color.I :=
  by
    sorry

end color_opposite_orange_is_indigo_l152_152211


namespace find_g_values_l152_152105

variables (f g : ℝ → ℝ)

-- Conditions
axiom cond1 : ∀ x y, g (x - y) = g x * g y + f x * f y
axiom cond2 : f (-1) = -1
axiom cond3 : f 0 = 0
axiom cond4 : f 1 = 1

-- Goal
theorem find_g_values : g 0 = 1 ∧ g 1 = 0 ∧ g 2 = -1 :=
by
  sorry

end find_g_values_l152_152105


namespace birdhouse_flown_distance_l152_152626

-- Definition of the given conditions.
def car_distance : ℕ := 200
def lawn_chair_distance : ℕ := 2 * car_distance
def birdhouse_distance : ℕ := 3 * lawn_chair_distance

-- Statement of the proof problem.
theorem birdhouse_flown_distance : birdhouse_distance = 1200 := by
  sorry

end birdhouse_flown_distance_l152_152626


namespace base10_to_base7_of_804_l152_152505

def base7 (n : ℕ) : ℕ :=
  let d3 := n / 343
  let r3 := n % 343
  let d2 := r3 / 49
  let r2 := r3 % 49
  let d1 := r2 / 7
  let d0 := r2 % 7
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

theorem base10_to_base7_of_804 :
  base7 804 = 2226 :=
by
  -- Proof to be filled in.
  sorry

end base10_to_base7_of_804_l152_152505


namespace calculate_taxi_fare_l152_152071

theorem calculate_taxi_fare :
  ∀ (f_80 f_120: ℝ), f_80 = 160 ∧ f_80 = 20 + (80 * (140/80)) →
                      f_120 = 20 + (120 * (140/80)) →
                      f_120 = 230 :=
by
  intro f_80 f_120
  rintro ⟨h80, h_proportional⟩ h_120
  sorry

end calculate_taxi_fare_l152_152071


namespace probability_correct_l152_152706

open Finset

def boxA : Finset Nat := {1, 2}
def boxB : Finset Nat := {3, 4, 5, 6}

def favorable_pairs : Finset (Nat × Nat) := 
  {pair ∈ (boxA.product boxB) | (pair.fst + pair.snd) > 6}

def total_pairs : Finset (Nat × Nat) := boxA.product boxB

def probability_of_sum_greater_than_6 : ℚ := 
  (favorable_pairs.card : ℚ) / (total_pairs.card : ℚ)

theorem probability_correct : probability_of_sum_greater_than_6 = 3 / 8 := by
  sorry

end probability_correct_l152_152706


namespace numPerfectSquareFactorsOf450_l152_152144

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l152_152144


namespace birdhouse_flown_distance_l152_152627

-- Definition of the given conditions.
def car_distance : ℕ := 200
def lawn_chair_distance : ℕ := 2 * car_distance
def birdhouse_distance : ℕ := 3 * lawn_chair_distance

-- Statement of the proof problem.
theorem birdhouse_flown_distance : birdhouse_distance = 1200 := by
  sorry

end birdhouse_flown_distance_l152_152627


namespace min_people_in_group_l152_152711

theorem min_people_in_group (B G : ℕ) (h : B / (B + G : ℝ) > 0.94) : B + G ≥ 17 :=
sorry

end min_people_in_group_l152_152711


namespace Jaylen_total_vegetables_l152_152459

def Jaylen_vegetables (J_bell_peppers J_green_beans J_carrots J_cucumbers : Nat) : Nat :=
  J_bell_peppers + J_green_beans + J_carrots + J_cucumbers

theorem Jaylen_total_vegetables :
  let Kristin_bell_peppers := 2
  let Kristin_green_beans := 20
  let Jaylen_bell_peppers := 2 * Kristin_bell_peppers
  let Jaylen_green_beans := (Kristin_green_beans / 2) - 3
  let Jaylen_carrots := 5
  let Jaylen_cucumbers := 2
  Jaylen_vegetables Jaylen_bell_peppers Jaylen_green_beans Jaylen_carrots Jaylen_cucumbers = 18 := 
by
  sorry

end Jaylen_total_vegetables_l152_152459


namespace find_unit_prices_minimize_total_cost_l152_152388

def unit_prices_ (x y : ℕ) :=
  x + 2 * y = 40 ∧ 2 * x + 3 * y = 70
  
theorem find_unit_prices (x y: ℕ) (h: unit_prices_ x y): x = 20 ∧ y = 10 := 
  sorry

def total_cost (m: ℕ) := 20 * m + 10 * (60 - m)

theorem minimize_total_cost (m : ℕ) (h1 : 60 ≥ m) (h2 : m ≥ 20) : 
  total_cost m = 800 → m = 20 :=
  sorry

end find_unit_prices_minimize_total_cost_l152_152388


namespace train_length_l152_152794

theorem train_length
  (S : ℝ)
  (L : ℝ)
  (h1 : L + 140 = S * 15)
  (h2 : L + 250 = S * 20) :
  L = 190 :=
by
  -- Proof to be provided here
  sorry

end train_length_l152_152794


namespace original_price_of_cycle_l152_152524

/--
A man bought a cycle for some amount and sold it at a loss of 20%.
The selling price of the cycle is Rs. 1280.
What was the original price of the cycle?
-/
theorem original_price_of_cycle
    (loss_percent : ℝ)
    (selling_price : ℝ)
    (original_price : ℝ)
    (h_loss_percent : loss_percent = 0.20)
    (h_selling_price : selling_price = 1280)
    (h_selling_eqn : selling_price = (1 - loss_percent) * original_price) :
    original_price = 1600 :=
sorry

end original_price_of_cycle_l152_152524


namespace largest_possible_length_d_l152_152248

theorem largest_possible_length_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 2) 
  (h2 : a ≤ b)
  (h3 : b ≤ c)
  (h4 : c ≤ d) 
  (h5 : d < a + b + c) : 
  d < 1 :=
sorry

end largest_possible_length_d_l152_152248


namespace monthly_growth_rate_selling_price_april_l152_152241

-- First problem: Proving the monthly average growth rate
theorem monthly_growth_rate (sales_jan sales_mar : ℝ) (x : ℝ) 
    (h1 : sales_jan = 256)
    (h2 : sales_mar = 400)
    (h3 : sales_mar = sales_jan * (1 + x)^2) :
  x = 0.25 := 
sorry

-- Second problem: Proving the selling price in April
theorem selling_price_april (unit_profit desired_profit current_sales sales_increase_per_yuan_change current_price new_price : ℝ)
    (h1 : unit_profit = new_price - 25)
    (h2 : desired_profit = 4200)
    (h3 : current_sales = 400)
    (h4 : sales_increase_per_yuan_change = 4)
    (h5 : current_price = 40)
    (h6 : desired_profit = unit_profit * (current_sales + sales_increase_per_yuan_change * (current_price - new_price))) :
  new_price = 35 := 
sorry

end monthly_growth_rate_selling_price_april_l152_152241


namespace option_D_correct_l152_152510

theorem option_D_correct (x : ℝ) : 2 * x^2 * (3 * x)^2 = 18 * x^4 :=
by sorry

end option_D_correct_l152_152510


namespace bryan_initial_pushups_l152_152256

def bryan_pushups (x : ℕ) : Prop :=
  let totalPushups := x + x + (x - 5)
  totalPushups = 40

theorem bryan_initial_pushups (x : ℕ) (hx : bryan_pushups x) : x = 15 :=
by {
  sorry
}

end bryan_initial_pushups_l152_152256


namespace total_net_worth_after_2_years_l152_152914

def initial_value : ℝ := 40000
def depreciation_rate : ℝ := 0.05
def initial_maintenance_cost : ℝ := 2000
def inflation_rate : ℝ := 0.03
def years : ℕ := 2

def value_at_end_of_year (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  List.foldl (λ acc _ => acc * (1 - rate)) initial_value (List.range years)

def cumulative_maintenance_cost (initial_maintenance_cost : ℝ) (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  List.foldl (λ acc year => acc + initial_maintenance_cost * ((1 + inflation_rate) ^ year)) 0 (List.range years)

def total_net_worth (initial_value : ℝ) (depreciation_rate : ℝ) (initial_maintenance_cost : ℝ) (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  value_at_end_of_year initial_value depreciation_rate years - cumulative_maintenance_cost initial_maintenance_cost inflation_rate years

theorem total_net_worth_after_2_years : total_net_worth initial_value depreciation_rate initial_maintenance_cost inflation_rate years = 32040 :=
  by
    sorry

end total_net_worth_after_2_years_l152_152914


namespace find_top_row_number_l152_152028

theorem find_top_row_number (x z : ℕ) (h1 : 8 = x * 2) (h2 : 16 = 2 * z)
  (h3 : 56 = 8 * 7) (h4 : 112 = 16 * 7) : x = 4 :=
by sorry

end find_top_row_number_l152_152028


namespace cubic_sum_l152_152314

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l152_152314


namespace intersection_correct_l152_152967

def setA := {x : ℝ | (x - 2) * (2 * x + 1) ≤ 0}
def setB := {x : ℝ | x < 1}
def expectedIntersection := {x : ℝ | -1 / 2 ≤ x ∧ x < 1}

theorem intersection_correct : (setA ∩ setB) = expectedIntersection := by
  sorry

end intersection_correct_l152_152967


namespace bound_on_f_l152_152874

theorem bound_on_f 
  (f : ℝ → ℝ) 
  (h_domain : ∀ x, 0 ≤ x ∧ x ≤ 1) 
  (h_zeros : f 0 = 0 ∧ f 1 = 0)
  (h_condition : ∀ x1 x2, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ x1 ≠ x2 → |f x2 - f x1| < |x2 - x1|) 
  : ∀ x1 x2, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 → |f x2 - f x1| < 1/2 :=
by
  sorry

end bound_on_f_l152_152874


namespace decagonal_pyramid_volume_l152_152940

noncomputable def volume_of_decagonal_pyramid (m : ℝ) (apex_angle : ℝ) : ℝ :=
  let sin18 := Real.sin (18 * Real.pi / 180)
  let sin36 := Real.sin (36 * Real.pi / 180)
  let cos18 := Real.cos (18 * Real.pi / 180)
  (5 * m^3 * sin36) / (3 * (1 + 2 * cos18))

theorem decagonal_pyramid_volume : volume_of_decagonal_pyramid 39 (18 * Real.pi / 180) = 20023 :=
  sorry

end decagonal_pyramid_volume_l152_152940


namespace rounding_proof_l152_152650

def rounding_question : Prop :=
  let num := 9.996
  let rounded_value := ((num * 100).round / 100)
  rounded_value ≠ 10.00

theorem rounding_proof : rounding_question :=
by
  sorry

end rounding_proof_l152_152650


namespace fraction_ratio_l152_152268

theorem fraction_ratio :
  ∃ (x y : ℕ), y ≠ 0 ∧ (x:ℝ) / (y:ℝ) = 240 / 1547 ∧ ((x:ℝ) / (y:ℝ)) / (2 / 13) = (5 / 34) / (7 / 48) :=
sorry

end fraction_ratio_l152_152268


namespace function_domain_length_correct_l152_152417

noncomputable def function_domain_length : ℕ :=
  let p : ℕ := 240 
  let q : ℕ := 1
  p + q

theorem function_domain_length_correct : function_domain_length = 241 := by
  sorry

end function_domain_length_correct_l152_152417


namespace ratio_of_square_sides_l152_152527

theorem ratio_of_square_sides
  (a b : ℝ) 
  (h1 : ∃ square1 : ℝ, square1 = 2 * a)
  (h2 : ∃ square2 : ℝ, square2 = 2 * b)
  (h3 : a ^ 2 - 4 * a * b - 5 * b ^ 2 = 0) :
  2 * a / 2 * b = 5 :=
by
  sorry

end ratio_of_square_sides_l152_152527


namespace spadesuit_evaluation_l152_152956

def spadesuit (a b : ℝ) : ℝ := abs (a - b)

theorem spadesuit_evaluation : spadesuit 1.5 (spadesuit 2.5 (spadesuit 4.5 6)) = 0.5 :=
by
  sorry

end spadesuit_evaluation_l152_152956


namespace coordinates_reflect_y_axis_l152_152166

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem coordinates_reflect_y_axis (p : ℝ × ℝ) (h : p = (5, 2)) : reflect_y_axis p = (-5, 2) :=
by
  rw [h]
  rfl

end coordinates_reflect_y_axis_l152_152166


namespace solve_inequality_l152_152099

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def given_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x >= 0 → f x = x^3 - 8

theorem solve_inequality (f : ℝ → ℝ) (h_even : even_function f) (h_given : given_function f) :
  {x | f (x - 2) > 0} = {x | x < 0 ∨ x > 4} :=
by
  sorry

end solve_inequality_l152_152099


namespace find_coordinates_condition1_find_coordinates_condition2_find_coordinates_condition3_l152_152277

-- Define the coordinate functions for point P
def coord_x (m : ℚ) : ℚ := 3 * m + 6
def coord_y (m : ℚ) : ℚ := m - 3

-- Definitions for each condition
def condition1 (m : ℚ) : Prop := coord_x m = coord_y m
def condition2 (m : ℚ) : Prop := coord_y m = coord_x m + 5
def condition3 (m : ℚ) : Prop := coord_x m = 3

-- Proof statements for the coordinates based on each condition
theorem find_coordinates_condition1 : 
  ∃ m, condition1 m ∧ coord_x m = -7.5 ∧ coord_y m = -7.5 :=
by sorry

theorem find_coordinates_condition2 : 
  ∃ m, condition2 m ∧ coord_x m = -15 ∧ coord_y m = -10 :=
by sorry

theorem find_coordinates_condition3 : 
  ∃ m, condition3 m ∧ coord_x m = 3 ∧ coord_y m = -4 :=
by sorry

end find_coordinates_condition1_find_coordinates_condition2_find_coordinates_condition3_l152_152277


namespace correct_answer_l152_152648

-- Statement of the problem
theorem correct_answer :
  ∃ (answer : String),
    (answer = "long before" ∨ answer = "before long" ∨ answer = "soon after" ∨ answer = "shortly after") ∧
    answer = "long before" :=
by
  sorry

end correct_answer_l152_152648


namespace power_of_2_l152_152545

theorem power_of_2 (n : ℕ) (h1 : n ≥ 1) (h2 : ∃ m : ℕ, m ≥ 1 ∧ (2^n - 1) ∣ (m^2 + 9)) : ∃ k : ℕ, n = 2^k := 
sorry

end power_of_2_l152_152545


namespace determine_a_l152_152847

noncomputable def f (x : ℝ) : ℝ := Real.exp (abs (x - 1)) + 1

theorem determine_a (a : ℝ) (h : f a = 2) : a = 1 :=
by
  sorry

end determine_a_l152_152847


namespace youngest_brother_age_l152_152203

theorem youngest_brother_age (x : ℕ) (h : x + (x + 1) + (x + 2) = 96) : x = 31 :=
sorry

end youngest_brother_age_l152_152203


namespace max_S_value_max_S_value_achievable_l152_152463

theorem max_S_value (x y z w : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  (x^2 * y + y^2 * z + z^2 * w + w^2 * x - x * y^2 - y * z^2 - z * w^2 - w * x^2) ≤ 8 / 27 :=
sorry

theorem max_S_value_achievable :
  ∃ (x y z w : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) ∧ 
  (x^2 * y + y^2 * z + z^2 * w + w^2 * x - x * y^2 - y * z^2 - z * w^2 - w * x^2) = 8 / 27 :=
sorry

end max_S_value_max_S_value_achievable_l152_152463


namespace train_speed_l152_152224

theorem train_speed (v : ℝ) : (∃ t : ℝ, 2 * v + t * v = 285 ∧ t = 285 / 38) → v = 30 :=
by
  sorry

end train_speed_l152_152224


namespace evaluate_expression_l152_152873

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7

theorem evaluate_expression : ( (1 / a) + (1 / b) + (1 / c) + (1 / d) ) ^ 2 = (7 / 49) :=
by
  sorry

end evaluate_expression_l152_152873


namespace translated_line_eqn_l152_152708

theorem translated_line_eqn
  (c : ℝ) :
  ∀ (y_eqn : ℝ → ℝ), 
    (∀ x, y_eqn x = 2 * x + 1) →
    (∀ x, (y_eqn (x - 2) - 3) = (2 * x - 6)) :=
by
  sorry

end translated_line_eqn_l152_152708


namespace smallest_angle_half_largest_l152_152624

open Real

-- Statement of the problem
theorem smallest_angle_half_largest (a b c : ℝ) (α β γ : ℝ)
  (h_sides : a = 4 ∧ b = 5 ∧ c = 6)
  (h_angles : α < β ∧ β < γ)
  (h_cos_alpha : cos α = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_cos_gamma : cos γ = (a^2 + b^2 - c^2) / (2 * a * b)) :
  2 * α = γ := 
sorry

end smallest_angle_half_largest_l152_152624


namespace sum_invested_7000_l152_152064

-- Define the conditions
def interest_15 (P : ℝ) : ℝ := P * 0.15 * 2
def interest_12 (P : ℝ) : ℝ := P * 0.12 * 2

-- Main statement to prove
theorem sum_invested_7000 (P : ℝ) (h : interest_15 P - interest_12 P = 420) : P = 7000 := by
  sorry

end sum_invested_7000_l152_152064


namespace circumscribed_circle_radius_l152_152970

noncomputable def radius_of_circumscribed_circle (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 + b^2)) / 2

theorem circumscribed_circle_radius (a r l b R : ℝ)
  (h1 : r = 1)
  (h2 : a = 2 * Real.sqrt 3)
  (h3 : b = 3)
  (h4 : l = a)
  (h5 : R = radius_of_circumscribed_circle l b) :
  R = Real.sqrt 21 / 2 :=
by
  sorry

end circumscribed_circle_radius_l152_152970


namespace num_tables_l152_152060

theorem num_tables (T : ℕ) : 
  (6 * T = (17 / 3) * T) → 
  T = 6 :=
sorry

end num_tables_l152_152060


namespace expression_as_fraction_l152_152816

theorem expression_as_fraction :
  1 + (4 / (5 + (6 / 7))) = (69 : ℚ) / 41 := 
by
  sorry

end expression_as_fraction_l152_152816


namespace chocolate_factory_production_l152_152204

theorem chocolate_factory_production
  (candies_per_hour : ℕ)
  (total_candies : ℕ)
  (days : ℕ)
  (total_hours : ℕ := total_candies / candies_per_hour)
  (hours_per_day : ℕ := total_hours / days)
  (h1 : candies_per_hour = 50)
  (h2 : total_candies = 4000)
  (h3 : days = 8) :
  hours_per_day = 10 := by
  sorry

end chocolate_factory_production_l152_152204


namespace radius_of_smaller_semicircle_l152_152169

theorem radius_of_smaller_semicircle :
  ∃ x : ℝ, 0 < x ∧
    let AB := 6 in
    let AC := 12 - x in
    let BC := 6 + x in
    (AB = 6) ∧ 
    (AC = 12 - x) ∧ 
    (BC = 6 + x) ∧
    (AB^2 + AC^2 = BC^2) ∧
    x = 4 := 
by
  use 4
  split
  { exact zero_lt_four }
  split
  { reflexivity }
  split
  { reflexivity }
  split
  { reflexivity }
  { sorry }

end radius_of_smaller_semicircle_l152_152169


namespace samantha_erased_length_l152_152901

/--
Samantha drew a line that was originally 1 meter (100 cm) long, and then it was erased until the length was 90 cm.
This theorem proves that the amount erased was 10 cm.
-/
theorem samantha_erased_length : 
  let original_length := 100 -- original length in cm
  let final_length := 90 -- final length in cm
  original_length - final_length = 10 := 
by
  sorry

end samantha_erased_length_l152_152901


namespace find_2a_minus_b_l152_152481

-- Define conditions
def f (x : ℝ) (a b : ℝ) := a * x + b
def g (x : ℝ) := -5 * x + 7
def h (x : ℝ) (a b : ℝ) := f (g x) a b
def h_inv (x : ℝ) := x - 9

-- Statement to prove
theorem find_2a_minus_b (a b : ℝ) 
(h_eq : ∀ x, h x a b = a * (-5 * x + 7) + b)
(h_inv_eq : ∀ x, h_inv x = x - 9)
(h_hinv_eq : ∀ x, h (h_inv x) a b = x) :
  2 * a - b = -54 / 5 := sorry

end find_2a_minus_b_l152_152481


namespace fraction_of_jenny_bounce_distance_l152_152869

-- Definitions for the problem conditions
def jenny_initial_distance := 18
def jenny_bounce_fraction (f : ℚ) : ℚ := 18 * f
def jenny_total_distance (f : ℚ) : ℚ := jenny_initial_distance + jenny_bounce_fraction f

def mark_initial_distance := 15
def mark_bounce_distance := 2 * mark_initial_distance
def mark_total_distance : ℚ := mark_initial_distance + mark_bounce_distance

def distance_difference := 21

-- The theorem to prove
theorem fraction_of_jenny_bounce_distance (f : ℚ) :
  mark_total_distance = jenny_total_distance f + distance_difference →
  f = 1 / 3 :=
by
  sorry

end fraction_of_jenny_bounce_distance_l152_152869


namespace smallest_possible_value_l152_152857

open Nat

theorem smallest_possible_value (c d : ℕ) (hc : c > d) (hc_pos : 0 < c) (hd_pos : 0 < d) (odd_cd : ¬Even (c + d)) :
  (∃ (y : ℚ), y > 0 ∧ y = (c + d : ℚ) / (c - d) + (c - d : ℚ) / (c + d) ∧ y = 10 / 3) :=
by
  sorry

end smallest_possible_value_l152_152857


namespace total_value_proof_l152_152063

def total_bills : ℕ := 126
def five_dollar_bills : ℕ := 84
def ten_dollar_bills : ℕ := total_bills - five_dollar_bills
def value_five_dollar_bills : ℕ := five_dollar_bills * 5
def value_ten_dollar_bills : ℕ := ten_dollar_bills * 10
def total_value : ℕ := value_five_dollar_bills + value_ten_dollar_bills

theorem total_value_proof : total_value = 840 := by
  unfold total_value value_five_dollar_bills value_ten_dollar_bills
  unfold five_dollar_bills ten_dollar_bills total_bills
  -- Calculation steps to show that value_five_dollar_bills + value_ten_dollar_bills = 840
  sorry

end total_value_proof_l152_152063


namespace total_trip_duration_proof_l152_152499

-- Naming all components
def driving_time : ℝ := 5
def first_jam_duration (pre_first_jam_drive : ℝ) : ℝ := 1.5 * pre_first_jam_drive
def second_jam_duration (between_first_and_second_drive : ℝ) : ℝ := 2 * between_first_and_second_drive
def third_jam_duration (between_second_and_third_drive : ℝ) : ℝ := 3 * between_second_and_third_drive
def pit_stop_duration : ℝ := 0.5
def pit_stops : ℕ := 2
def initial_drive : ℝ := 1
def second_drive : ℝ := 1.5

-- Additional drive time calculation
def remaining_drive : ℝ := driving_time - initial_drive - second_drive

-- Total duration calculation
def total_duration (initial_drive : ℝ) (second_drive : ℝ) (remaining_drive : ℝ) (first_jam_duration : ℝ) 
(second_jam_duration : ℝ) (third_jam_duration : ℝ) (pit_stop_duration : ℝ) (pit_stops : ℕ) : ℝ :=
  driving_time + first_jam_duration + second_jam_duration + third_jam_duration + (pit_stop_duration * pit_stops)

theorem total_trip_duration_proof :
  total_duration initial_drive second_drive remaining_drive (first_jam_duration initial_drive)
                  (second_jam_duration second_drive) (third_jam_duration remaining_drive) pit_stop_duration pit_stops 
  = 18 :=
by
  -- Proof steps would go here
  sorry

end total_trip_duration_proof_l152_152499


namespace x_cubed_plus_y_cubed_l152_152327

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l152_152327


namespace frank_used_2_bags_l152_152092

theorem frank_used_2_bags (total_candy : ℕ) (candy_per_bag : ℕ) (h1 : total_candy = 16) (h2 : candy_per_bag = 8) : (total_candy / candy_per_bag) = 2 := 
by
  sorry

end frank_used_2_bags_l152_152092


namespace number_of_rocks_in_bucket_l152_152343

noncomputable def average_weight_rock : ℝ := 1.5
noncomputable def total_money_made : ℝ := 60
noncomputable def price_per_pound : ℝ := 4

theorem number_of_rocks_in_bucket : 
  let total_weight_rocks := total_money_made / price_per_pound
  let number_of_rocks := total_weight_rocks / average_weight_rock
  number_of_rocks = 10 :=
by
  sorry

end number_of_rocks_in_bucket_l152_152343


namespace casey_marathon_time_l152_152953

theorem casey_marathon_time (C : ℝ) (h : (C + (4 / 3) * C) / 2 = 7) : C = 10.5 :=
by
  sorry

end casey_marathon_time_l152_152953


namespace product_of_possible_values_l152_152755

theorem product_of_possible_values (b : ℝ) (side_length : ℝ) (square_condition : (b - 2) = side_length ∨ (2 - b) = side_length) : 
  (b = -3 ∨ b = 7) → (-3 * 7 = -21) :=
by
  intro h
  sorry

end product_of_possible_values_l152_152755


namespace second_player_wins_l152_152633

theorem second_player_wins 
  (pile1 : ℕ) (pile2 : ℕ) (pile3 : ℕ)
  (h1 : pile1 = 10) (h2 : pile2 = 15) (h3 : pile3 = 20) :
  (pile1 - 1) + (pile2 - 1) + (pile3 - 1) % 2 = 0 :=
by
  sorry

end second_player_wins_l152_152633


namespace num_div_divided_by_10_l152_152927

-- Given condition: the number divided by 10 equals 12
def number_divided_by_10_gives_12 (x : ℝ) : Prop :=
  x / 10 = 12

-- Lean statement for the mathematical problem
theorem num_div_divided_by_10 (x : ℝ) (h : number_divided_by_10_gives_12 x) : x = 120 :=
by
  sorry

end num_div_divided_by_10_l152_152927


namespace find_number_l152_152788

theorem find_number (x : ℕ) (h : x + 8 = 500) : x = 492 :=
by sorry

end find_number_l152_152788


namespace find_2x_plus_y_l152_152844

theorem find_2x_plus_y (x y : ℝ) 
  (h1 : (x + y) / 3 = 5 / 3) 
  (h2 : x + 2*y = 8) : 
  2*x + y = 7 :=
sorry

end find_2x_plus_y_l152_152844


namespace incorrect_statement_c_l152_152416

open Real

theorem incorrect_statement_c (p q: ℝ) : ¬(∀ x: ℝ, (x * abs x + p * x + q = 0 ↔ p^2 - 4 * q ≥ 0)) :=
sorry

end incorrect_statement_c_l152_152416


namespace cost_of_each_item_number_of_purchasing_plans_l152_152361

-- Question 1: Cost of each item
theorem cost_of_each_item : 
  ∃ (x y : ℕ), 
    (10 * x + 5 * y = 2000) ∧ 
    (5 * x + 3 * y = 1050) ∧ 
    (x = 150) ∧ 
    (y = 100) :=
by
    sorry

-- Question 2: Number of different purchasing plans
theorem number_of_purchasing_plans : 
  (∀ (a b : ℕ), 
    (150 * a + 100 * b = 4000) → 
    (a ≥ 12) → 
    (b ≥ 12) → 
    (4 = 4)) :=
by
    sorry

end cost_of_each_item_number_of_purchasing_plans_l152_152361


namespace number_of_perfect_square_factors_l152_152142

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l152_152142


namespace odd_function_property_l152_152881

theorem odd_function_property {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = - f x) (h2 : ∀ x, f (1 + x) = f (-x)) (h3 : f (-1 / 3) = 1 / 3) : f (5 / 3) = 1 / 3 := 
sorry

end odd_function_property_l152_152881


namespace monomial_combined_l152_152981

theorem monomial_combined (n m : ℕ) (h₁ : 2 = n) (h₂ : m = 4) : n^m = 16 := by
  sorry

end monomial_combined_l152_152981


namespace number_of_beakers_calculation_l152_152920

-- Conditions
def solution_per_test_tube : ℕ := 7
def number_of_test_tubes : ℕ := 6
def solution_per_beaker : ℕ := 14

-- Total amount of solution
def total_solution : ℕ := solution_per_test_tube * number_of_test_tubes

-- Number of beakers is the fraction of total solution and solution per beaker
def number_of_beakers : ℕ := total_solution / solution_per_beaker

-- Statement of the problem
theorem number_of_beakers_calculation : number_of_beakers = 3 :=
by 
  -- Proof goes here
  sorry

end number_of_beakers_calculation_l152_152920


namespace yellow_marbles_problem_l152_152496

variable (Y B R : ℕ)

theorem yellow_marbles_problem
  (h1 : Y + B + R = 19)
  (h2 : B = (3 * R) / 4)
  (h3 : R = Y + 3) :
  Y = 5 :=
by
  sorry

end yellow_marbles_problem_l152_152496


namespace canadian_math_olympiad_1992_l152_152283

theorem canadian_math_olympiad_1992
    (n : ℤ) (a : ℕ → ℤ) (k : ℕ)
    (h1 : n ≥ a 1) 
    (h2 : ∀ i, 1 ≤ i → i ≤ k → a i > 0)
    (h3 : ∀ i j, 1 ≤ i → i ≤ k → 1 ≤ j → j ≤ k → n ≥ Int.lcm (a i) (a j))
    (h4 : ∀ i, 1 ≤ i → i < k → a i > a (i + 1)) :
  ∀ i, 1 ≤ i → i ≤ k → i * a i ≤ n :=
sorry

end canadian_math_olympiad_1992_l152_152283


namespace radius_of_smaller_semicircle_l152_152168

theorem radius_of_smaller_semicircle :
  ∃ x : ℝ, 0 < x ∧
    let AB := 6 in
    let AC := 12 - x in
    let BC := 6 + x in
    (AB = 6) ∧ 
    (AC = 12 - x) ∧ 
    (BC = 6 + x) ∧
    (AB^2 + AC^2 = BC^2) ∧
    x = 4 := 
by
  use 4
  split
  { exact zero_lt_four }
  split
  { reflexivity }
  split
  { reflexivity }
  split
  { reflexivity }
  { sorry }

end radius_of_smaller_semicircle_l152_152168


namespace abs_neg_three_l152_152484

theorem abs_neg_three : abs (-3) = 3 :=
sorry

end abs_neg_three_l152_152484


namespace measure_of_angle_A_l152_152798

theorem measure_of_angle_A
    (A B : ℝ)
    (h1 : A + B = 90)
    (h2 : A = 3 * B) :
    A = 67.5 :=
by
  sorry

end measure_of_angle_A_l152_152798


namespace reciprocal_of_neg_two_thirds_l152_152488

-- Definition for finding the reciprocal
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The proof problem statement
theorem reciprocal_of_neg_two_thirds : reciprocal (-2 / 3) = -3 / 2 :=
sorry

end reciprocal_of_neg_two_thirds_l152_152488


namespace students_average_vegetables_l152_152638

variable (points_needed : ℕ) (points_per_vegetable : ℕ) (students : ℕ) (school_days : ℕ) (school_weeks : ℕ)

def average_vegetables_per_student_per_week (points_needed points_per_vegetable students school_days school_weeks : ℕ) : ℕ :=
  let total_vegetables := points_needed / points_per_vegetable
  let vegetables_per_student := total_vegetables / students
  vegetables_per_student / school_weeks

theorem students_average_vegetables 
  (h1 : points_needed = 200) 
  (h2 : points_per_vegetable = 2) 
  (h3 : students = 25) 
  (h4 : school_days = 10) 
  (h5 : school_weeks = 2) : 
  average_vegetables_per_student_per_week points_needed points_per_vegetable students school_days school_weeks = 2 :=
by
  sorry

end students_average_vegetables_l152_152638


namespace product_of_roots_cubic_l152_152659

theorem product_of_roots_cubic:
  (∀ x : ℝ, x^3 - 15 * x^2 + 60 * x - 45 = 0 → x = r_1 ∨ x = r_2 ∨ x = r_3) →
  r_1 * r_2 * r_3 = 45 :=
by
  intro h
  -- the proof should be filled in here
  sorry

end product_of_roots_cubic_l152_152659


namespace union_of_sets_l152_152009

def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {2, 3, 6}

theorem union_of_sets : A ∪ B = {1, 2, 3, 5, 6} :=
by sorry

end union_of_sets_l152_152009


namespace find_c_l152_152701

-- Define the function
def f (c x : ℝ) : ℝ := x^4 - 8 * x^2 + c

-- Condition: The function has a minimum value of -14 on the interval [-1, 3]
def condition (c : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-1 : ℝ) 3, ∀ y ∈ Set.Icc (-1 : ℝ) 3, f c x ≤ f c y ∧ f c x = -14

-- The theorem to be proved
theorem find_c : ∃ c : ℝ, condition c ∧ c = 2 :=
sorry

end find_c_l152_152701


namespace cubic_sum_l152_152315

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l152_152315


namespace number_of_lion_cubs_l152_152526

theorem number_of_lion_cubs 
    (initial_animal_count final_animal_count : ℕ)
    (gorillas_sent hippo_adopted rhinos_taken new_animals : ℕ)
    (lion_cubs meerkats : ℕ) :
    initial_animal_count = 68 ∧ 
    gorillas_sent = 6 ∧ 
    hippo_adopted = 1 ∧ 
    rhinos_taken = 3 ∧ 
    final_animal_count = 90 ∧ 
    meerkats = 2 * lion_cubs ∧
    new_animals = lion_cubs + meerkats ∧
    final_animal_count = initial_animal_count - gorillas_sent + hippo_adopted + rhinos_taken + new_animals
    → lion_cubs = 8 := sorry

end number_of_lion_cubs_l152_152526


namespace janet_earnings_eur_l152_152715

noncomputable def usd_to_eur (usd : ℚ) : ℚ :=
  usd * 0.85

def janet_earnings_usd : ℚ :=
  (130 * 0.25) + (90 * 0.30) + (30 * 0.40)

theorem janet_earnings_eur : usd_to_eur janet_earnings_usd = 60.78 :=
  by
    sorry

end janet_earnings_eur_l152_152715


namespace total_trip_length_l152_152476

theorem total_trip_length :
  ∀ (d : ℝ), 
    (∀ fuel_per_mile : ℝ, fuel_per_mile = 0.03 →
      ∀ battery_miles : ℝ, battery_miles = 50 →
      ∀ avg_miles_per_gallon : ℝ, avg_miles_per_gallon = 50 →
      (d / (fuel_per_mile * (d - battery_miles))) = avg_miles_per_gallon →
      d = 150) := 
by
  intros d fuel_per_mile fuel_per_mile_eq battery_miles battery_miles_eq avg_miles_per_gallon avg_miles_per_gallon_eq trip_condition
  sorry

end total_trip_length_l152_152476


namespace sum_lcm_eq_72_l152_152047

theorem sum_lcm_eq_72 (s : Finset ℕ) 
  (h1 : ∀ ν ∈ s, Nat.lcm ν 24 = 72) :
  s.sum id = 180 :=
by
  have h2 : ∀ ν, ν ∈ s → ∃ n, ν = 3 * n := 
    by sorry
  have h3 : ∀ n, ∃ ν, ν = 3 * n ∧ ν ∈ s := 
    by sorry
  sorry

end sum_lcm_eq_72_l152_152047


namespace most_cost_effective_years_l152_152615

noncomputable def total_cost (x : ℕ) : ℝ := 100000 + 15000 * x + 1000 + 2000 * ((x * (x - 1)) / 2)

noncomputable def average_annual_cost (x : ℕ) : ℝ := total_cost x / x

theorem most_cost_effective_years : ∃ (x : ℕ), x = 10 ∧
  (∀ y : ℕ, y ≠ 10 → average_annual_cost x ≤ average_annual_cost y) :=
by
  sorry

end most_cost_effective_years_l152_152615


namespace cubic_sum_l152_152321

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l152_152321


namespace area_of_square_l152_152993

theorem area_of_square (ABCD MN : ℝ) (h1 : 4 * (ABCD / 4) = ABCD) (h2 : MN = 3) : ABCD = 64 :=
by
  sorry

end area_of_square_l152_152993


namespace power_series_expansion_ln_l152_152541

theorem power_series_expansion_ln (x : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = ∑' n : ℕ, (λ n, if n = 0 then ln(2) else (-1)^(n+1) * (2^n + 1) / (2^n * n) * x^n) n) ∧ 
                (-1 < x ∧ x ≤ 1)) := 
  sorry

end power_series_expansion_ln_l152_152541


namespace find_y_l152_152421

theorem find_y (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : y = 5 :=
by
  sorry

end find_y_l152_152421


namespace sum_of_three_numbers_l152_152619

-- Definitions for the conditions
def mean_condition_1 (x y z : ℤ) := (x + y + z) / 3 = x + 20
def mean_condition_2 (x y z : ℤ) := (x + y + z) / 3 = z - 18
def median_condition (y : ℤ) := y = 9

-- The Lean 4 statement to prove the sum of x, y, and z is 21
theorem sum_of_three_numbers (x y z : ℤ) 
  (h1 : mean_condition_1 x y z) 
  (h2 : mean_condition_2 x y z) 
  (h3 : median_condition y) : 
  x + y + z = 21 := 
  by 
    sorry

end sum_of_three_numbers_l152_152619


namespace number_of_perfect_square_factors_l152_152141

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l152_152141


namespace inequality_inequality_l152_152808

theorem inequality_inequality (a b c d : ℝ) (h1 : a^2 + b^2 = 4) (h2 : c^2 + d^2 = 16) :
  ac + bd ≤ 8 :=
sorry

end inequality_inequality_l152_152808


namespace Alissa_presents_equal_9_l152_152537

def Ethan_presents : ℝ := 31.0
def difference : ℝ := 22.0
def Alissa_presents := Ethan_presents - difference

theorem Alissa_presents_equal_9 : Alissa_presents = 9.0 := 
by sorry

end Alissa_presents_equal_9_l152_152537


namespace problem_statement_l152_152729

variable {a b c x y z : ℝ}
variable (h1 : 17 * x + b * y + c * z = 0)
variable (h2 : a * x + 29 * y + c * z = 0)
variable (h3 : a * x + b * y + 53 * z = 0)
variable (ha : a ≠ 17)
variable (hx : x ≠ 0)

theorem problem_statement : 
  (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 :=
sorry

end problem_statement_l152_152729


namespace find_prices_max_sets_of_go_compare_options_l152_152108

theorem find_prices (x y : ℕ) (h1 : 2 * x + 3 * y = 140) (h2 : 4 * x + y = 130) :
  x = 25 ∧ y = 30 :=
by sorry

theorem max_sets_of_go (m : ℕ) (h3 : 25 * (80 - m) + 30 * m ≤ 2250) :
  m ≤ 50 :=
by sorry

theorem compare_options (a : ℕ) :
  (a < 10 → 27 * a < 21 * a + 60) ∧ (a = 10 → 27 * a = 21 * a + 60) ∧ (a > 10 → 27 * a > 21 * a + 60) :=
by sorry

end find_prices_max_sets_of_go_compare_options_l152_152108


namespace isosceles_triangle_perimeter_l152_152573

def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ b = c ∨ a = c) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 5) :
  ∃ c, is_isosceles_triangle a b c ∧ a + b + c = 12 :=
by {
  use 5,
  split,
  simp [is_isosceles_triangle, h1, h2],
  split,
  linarith,
  split,
  linarith,
  linarith,
  ring,
}

end isosceles_triangle_perimeter_l152_152573


namespace mary_visited_two_shops_l152_152190

-- Define the costs of items
def cost_shirt : ℝ := 13.04
def cost_jacket : ℝ := 12.27
def total_cost : ℝ := 25.31

-- Define the number of shops visited
def number_of_shops : ℕ := 2

-- Proof that Mary visited 2 shops given the conditions
theorem mary_visited_two_shops (h_shirt : cost_shirt = 13.04) (h_jacket : cost_jacket = 12.27) (h_total : cost_shirt + cost_jacket = total_cost) : number_of_shops = 2 :=
by
  sorry

end mary_visited_two_shops_l152_152190


namespace arithmetic_sequence_a9_l152_152218

noncomputable def S (n : ℕ) (a₁ aₙ : ℝ) : ℝ := (n * (a₁ + aₙ)) / 2

theorem arithmetic_sequence_a9 (a₁ a₁₇ : ℝ) (h1 : S 17 a₁ a₁₇ = 102) : (a₁ + a₁₇) / 2 = 6 :=
by
  sorry

end arithmetic_sequence_a9_l152_152218


namespace complex_expression_evaluation_l152_152021

-- Definition of the imaginary unit i with property i^2 = -1
def i : ℂ := Complex.I

-- Theorem stating that the given expression equals i
theorem complex_expression_evaluation : i * (1 - i) - 1 = i := by
  -- Proof omitted
  sorry

end complex_expression_evaluation_l152_152021


namespace sum_lcm_eq_72_l152_152046

theorem sum_lcm_eq_72 (s : Finset ℕ) 
  (h1 : ∀ ν ∈ s, Nat.lcm ν 24 = 72) :
  s.sum id = 180 :=
by
  have h2 : ∀ ν, ν ∈ s → ∃ n, ν = 3 * n := 
    by sorry
  have h3 : ∀ n, ∃ ν, ν = 3 * n ∧ ν ∈ s := 
    by sorry
  sorry

end sum_lcm_eq_72_l152_152046


namespace total_marbles_is_260_l152_152413

-- Define the number of marbles in each jar based on the conditions.
def first_jar : Nat := 80
def second_jar : Nat := 2 * first_jar
def third_jar : Nat := (1 / 4 : ℚ) * first_jar

-- Prove that the total number of marbles is 260.
theorem total_marbles_is_260 : first_jar + second_jar + third_jar = 260 := by
  sorry

end total_marbles_is_260_l152_152413


namespace value_added_to_075_of_number_l152_152490

theorem value_added_to_075_of_number (N V : ℝ) (h1 : 0.75 * N + V = 8) (h2 : N = 8) : V = 2 := by
  sorry

end value_added_to_075_of_number_l152_152490


namespace cubic_sum_identity_l152_152295

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l152_152295


namespace max_value_quadratic_function_l152_152453

open Real

theorem max_value_quadratic_function (r : ℝ) (x₀ y₀ : ℝ) (P_tangent : (2 / x₀) * x - y₀ = 0) 
  (circle_tangent : (x₀ - 3) * (x - 3) + y₀ * y = r^2) :
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ), f x = 1 / 2 * x * (3 - x)) ∧ 
  (∀ (x : ℝ), f x ≤ 9 / 8) :=
by
  sorry

end max_value_quadratic_function_l152_152453


namespace bobbo_minimum_speed_increase_l152_152405

theorem bobbo_minimum_speed_increase
  (initial_speed: ℝ)
  (river_width : ℝ)
  (current_speed : ℝ)
  (waterfall_distance : ℝ)
  (midpoint_distance : ℝ)
  (required_increase: ℝ) :
  initial_speed = 2 ∧ river_width = 100 ∧ current_speed = 5 ∧ waterfall_distance = 175 ∧ midpoint_distance = 50 ∧ required_increase = 3 → 
  (required_increase = (50 / (50 / current_speed)) - initial_speed) := 
by
  sorry

end bobbo_minimum_speed_increase_l152_152405


namespace pounds_over_weight_limit_l152_152457

-- Definitions based on conditions

def bookcase_max_weight : ℝ := 80

def weight_hardcover_book : ℝ := 0.5
def number_hardcover_books : ℕ := 70
def total_weight_hardcover_books : ℝ := number_hardcover_books * weight_hardcover_book

def weight_textbook : ℝ := 2
def number_textbooks : ℕ := 30
def total_weight_textbooks : ℝ := number_textbooks * weight_textbook

def weight_knick_knack : ℝ := 6
def number_knick_knacks : ℕ := 3
def total_weight_knick_knacks : ℝ := number_knick_knacks * weight_knick_knack

def total_weight_items : ℝ := total_weight_hardcover_books + total_weight_textbooks + total_weight_knick_knacks

theorem pounds_over_weight_limit : total_weight_items - bookcase_max_weight = 33 := by
  sorry

end pounds_over_weight_limit_l152_152457


namespace grocer_sales_l152_152655

theorem grocer_sales 
  (s1 s2 s3 s4 s5 s6 s7 s8 sales : ℝ)
  (h_sales_1 : s1 = 5420)
  (h_sales_2 : s2 = 5660)
  (h_sales_3 : s3 = 6200)
  (h_sales_4 : s4 = 6350)
  (h_sales_5 : s5 = 6500)
  (h_sales_6 : s6 = 6780)
  (h_sales_7 : s7 = 7000)
  (h_sales_8 : s8 = 7200)
  (h_avg : (5420 + 5660 + 6200 + 6350 + 6500 + 6780 + 7000 + 7200 + 2 * sales) / 10 = 6600) :
  sales = 9445 := 
  by 
  sorry

end grocer_sales_l152_152655


namespace perfect_square_factors_count_450_l152_152125

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l152_152125


namespace arithmetic_sequence_problem_l152_152682

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h1 : (a 1 - 3) ^ 3 + 3 * (a 1 - 3) = -3)
  (h12 : (a 12 - 3) ^ 3 + 3 * (a 12 - 3) = 3) :
  a 1 < a 12 ∧ (12 * (a 1 + a 12)) / 2 = 36 :=
by
  sorry

end arithmetic_sequence_problem_l152_152682


namespace probability_of_red_ball_is_correct_l152_152705

noncomputable def probability_of_drawing_red_ball (white_balls : ℕ) (red_balls : ℕ) :=
  let total_balls := white_balls + red_balls
  let favorable_outcomes := red_balls
  (favorable_outcomes : ℚ) / total_balls

theorem probability_of_red_ball_is_correct :
  probability_of_drawing_red_ball 5 2 = 2 / 7 :=
by
  sorry

end probability_of_red_ball_is_correct_l152_152705


namespace perfect_square_factors_450_l152_152130

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l152_152130


namespace cube_identity_l152_152304

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l152_152304


namespace floor_painting_cost_l152_152987

noncomputable def floor_painting_problem : Prop := 
  ∃ (B L₁ L₂ B₂ Area₁ Area₂ CombinedCost : ℝ),
  L₁ = 2 * B ∧
  Area₁ = L₁ * B ∧
  484 = Area₁ * 3 ∧
  L₂ = 0.8 * L₁ ∧
  B₂ = 1.3 * B ∧
  Area₂ = L₂ * B₂ ∧
  CombinedCost = 484 + (Area₂ * 5) ∧
  CombinedCost = 1320.8

theorem floor_painting_cost : floor_painting_problem :=
by
  sorry

end floor_painting_cost_l152_152987


namespace book_vs_necklace_price_difference_l152_152895

-- Problem-specific definitions and conditions
def necklace_price : ℕ := 34
def limit_price : ℕ := 70
def overspent : ℕ := 3
def total_spent : ℕ := limit_price + overspent
def book_price : ℕ := total_spent - necklace_price

-- Lean statement to prove the correct answer
theorem book_vs_necklace_price_difference :
  book_price - necklace_price = 5 := by
  sorry

end book_vs_necklace_price_difference_l152_152895


namespace each_person_received_5_l152_152522

theorem each_person_received_5 (S n : ℕ) (hn₁ : n > 5) (hn₂ : 5 * S = 2 * n * (n - 5)) (hn₃ : 4 * S = n * (n + 4)) :
  S / (n + 4) = 5 :=
by
  sorry

end each_person_received_5_l152_152522


namespace problem_l152_152346

variable {a b c x y z : ℝ}

theorem problem 
  (h1 : 5 * x + b * y + c * z = 0)
  (h2 : a * x + 7 * y + c * z = 0)
  (h3 : a * x + b * y + 9 * z = 0)
  (h4 : a ≠ 5)
  (h5 : x ≠ 0) :
  (a / (a - 5)) + (b / (b - 7)) + (c / (c - 9)) = 1 :=
by
  sorry

end problem_l152_152346


namespace sufficient_but_not_necessary_l152_152677

theorem sufficient_but_not_necessary (x : ℝ) :
  (|x - 3| - |x - 1| < 2) → x ≠ 1 ∧ ¬ (∀ x : ℝ, x ≠ 1 → |x - 3| - |x - 1| < 2) :=
by
  sorry

end sufficient_but_not_necessary_l152_152677


namespace perfect_square_divisors_of_450_l152_152135

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l152_152135


namespace divides_expression_l152_152181

theorem divides_expression (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y ^ (y ^ 2) - 2 * y ^ (y + 1) + 1) :=
sorry

end divides_expression_l152_152181


namespace solve_for_m_l152_152431

theorem solve_for_m (x y m : ℝ) 
  (h1 : 2 * x + y = 3 * m) 
  (h2 : x - 4 * y = -2 * m)
  (h3 : y + 2 * m = 1 + x) :
  m = 3 / 5 := 
by 
  sorry

end solve_for_m_l152_152431


namespace polar_equation_l152_152215

theorem polar_equation (y ρ θ : ℝ) (x : ℝ) 
  (h1 : y = ρ * Real.sin θ) 
  (h2 : x = ρ * Real.cos θ) 
  (h3 : y^2 = 12 * x) : 
  ρ * (Real.sin θ)^2 = 12 * Real.cos θ := 
by
  sorry

end polar_equation_l152_152215


namespace complement_M_eq_interval_l152_152288

-- Definition of the set M
def M : Set ℝ := { x | x * (x - 3) > 0 }

-- Universal set is ℝ
def U : Set ℝ := Set.univ

-- Theorem to prove the complement of M in ℝ is [0, 3]
theorem complement_M_eq_interval :
  U \ M = { x | 0 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end complement_M_eq_interval_l152_152288


namespace savings_increase_is_100_percent_l152_152246

variable (I : ℝ) -- Initial income
variable (S : ℝ) -- Initial savings
variable (I2 : ℝ) -- Income in the second year
variable (E1 : ℝ) -- Expenditure in the first year
variable (E2 : ℝ) -- Expenditure in the second year
variable (S2 : ℝ) -- Second year savings

-- Initial conditions
def initial_savings (I : ℝ) : ℝ := 0.25 * I
def first_year_expenditure (I : ℝ) (S : ℝ) : ℝ := I - S
def second_year_income (I : ℝ) : ℝ := 1.25 * I

-- Total expenditure condition
def total_expenditure_condition (E1 : ℝ) (E2 : ℝ) : Prop := E1 + E2 = 2 * E1

-- Prove that the savings increase in the second year is 100%
theorem savings_increase_is_100_percent :
   ∀ (I S E1 I2 E2 S2 : ℝ),
     S = initial_savings I →
     E1 = first_year_expenditure I S →
     I2 = second_year_income I →
     total_expenditure_condition E1 E2 →
     S2 = I2 - E2 →
     ((S2 - S) / S) * 100 = 100 := by
  sorry

end savings_increase_is_100_percent_l152_152246


namespace divides_expression_l152_152182

theorem divides_expression (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y ^ (y ^ 2) - 2 * y ^ (y + 1) + 1) :=
sorry

end divides_expression_l152_152182


namespace least_number_to_add_l152_152516

theorem least_number_to_add (n d : ℕ) (h : n = 1024) (h_d : d = 25) :
  ∃ x : ℕ, (n + x) % d = 0 ∧ x = 1 :=
by sorry

end least_number_to_add_l152_152516


namespace proof_problem_l152_152727

-- Given conditions
variables {a b : Type}  -- Two non-coincident lines
variables {α β : Type}  -- Two non-coincident planes

-- Definitions of the relationships
def is_parallel_to (x y : Type) : Prop := sorry  -- Parallel relationship
def is_perpendicular_to (x y : Type) : Prop := sorry  -- Perpendicular relationship

-- Statements to verify
def statement1 (a α b : Type) : Prop := 
  (is_parallel_to a α ∧ is_parallel_to b α) → is_parallel_to a b

def statement2 (a α β : Type) : Prop :=
  (is_perpendicular_to a α ∧ is_perpendicular_to a β) → is_parallel_to α β

def statement3 (α β : Type) : Prop :=
  is_perpendicular_to α β → ∃ l : Type, is_perpendicular_to l α ∧ is_parallel_to l β

def statement4 (α β : Type) : Prop :=
  is_perpendicular_to α β → ∃ γ : Type, is_perpendicular_to γ α ∧ is_perpendicular_to γ β

-- Proof problem: verifying which statements are true.
theorem proof_problem :
  ¬ (statement1 a α b) ∧ statement2 a α β ∧ statement3 α β ∧ statement4 α β :=
by
  sorry

end proof_problem_l152_152727


namespace counterexample_conjecture_l152_152034

theorem counterexample_conjecture 
    (odd_gt_5 : ℕ → Prop) 
    (is_prime : ℕ → Prop) 
    (conjecture : ∀ n, odd_gt_5 n → ∃ p1 p2 p3, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ n = p1 + p2 + p3) : 
    ∃ n, odd_gt_5 n ∧ ¬ (∃ p1 p2 p3, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ n = p1 + p2 + p3) :=
sorry

end counterexample_conjecture_l152_152034


namespace find_base_of_triangle_l152_152528

-- Given data
def perimeter : ℝ := 20 -- The perimeter of the triangle
def tangent_segment : ℝ := 2.4 -- The segment of the tangent to the inscribed circle contained between the sides

-- Define the problem and expected result
theorem find_base_of_triangle (a b c : ℝ) (P : a + b + c = perimeter)
  (tangent_parallel_base : ℝ := tangent_segment):
  a = 4 ∨ a = 6 :=
sorry

end find_base_of_triangle_l152_152528


namespace path_count_in_grid_l152_152664

theorem path_count_in_grid :
  let grid_width := 6
  let grid_height := 5
  let total_steps := 8
  let right_steps := 5
  let up_steps := 3
  ∃ (C : Nat), C = Nat.choose total_steps up_steps ∧ C = 56 :=
by
  sorry

end path_count_in_grid_l152_152664


namespace x_cubed_plus_y_cubed_l152_152325

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l152_152325


namespace convex_cyclic_quadrilaterals_count_l152_152289

theorem convex_cyclic_quadrilaterals_count :
  let num_quadrilaterals := ∑ i in (finset.range 36).powerset.filter(λ s, s.card = 4 
    ∧ let (a, b, c, d) := classical.some (vector.sorted_enum s)
    in a + b + c + d = 36 ∧ a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c 
  ),
  finset.count :=
  num_quadrilaterals = 819 :=
begin
  sorry
end

end convex_cyclic_quadrilaterals_count_l152_152289


namespace rectangle_length_l152_152331

theorem rectangle_length (s w : ℝ) (A : ℝ) (L : ℝ) (h1 : s = 9) (h2 : w = 3) (h3 : A = s * s) (h4 : A = w * L) : L = 27 :=
by
  sorry

end rectangle_length_l152_152331


namespace chris_car_offer_difference_l152_152263

theorem chris_car_offer_difference :
  ∀ (asking_price : ℕ) (maintenance_cost_factor : ℕ) (headlight_cost : ℕ) (tire_multiplier : ℕ),
  asking_price = 5200 →
  maintenance_cost_factor = 10 →
  headlight_cost = 80 →
  tire_multiplier = 3 →
  let first_earnings := asking_price - asking_price / maintenance_cost_factor,
      second_earnings := asking_price - (headlight_cost + headlight_cost * tire_multiplier) in
  second_earnings - first_earnings = 200 :=
by
  intros asking_price maintenance_cost_factor headlight_cost tire_multiplier h1 h2 h3 h4
  -- leave "sorry" as a placeholder for the proof
  sorry

end chris_car_offer_difference_l152_152263


namespace original_price_of_iWatch_l152_152958

theorem original_price_of_iWatch (P : ℝ) (h1 : 800 > 0) (h2 : P > 0)
    (h3 : 680 + 0.90 * P > 0) (h4 : 0.98 * (680 + 0.90 * P) = 931) :
    P = 300 := by
  sorry

end original_price_of_iWatch_l152_152958


namespace value_S3_S2_S5_S3_l152_152276

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
variable {d : ℝ}
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (d_ne_zero : d ≠ 0)
variable (h_geom_seq : (a 1 + 2 * d) ^ 2 = (a 1) * (a 1 + 3 * d))
variable (S_def : ∀ n, S n = n * a 1 + d * (n * (n - 1)) / 2)

theorem value_S3_S2_S5_S3 : (S 3 - S 2) / (S 5 - S 3) = 2 := by
  sorry

end value_S3_S2_S5_S3_l152_152276


namespace max_total_cut_length_l152_152519

theorem max_total_cut_length :
  let side_length := 30
  let num_pieces := 225
  let area_per_piece := (side_length ^ 2) / num_pieces
  let outer_perimeter := 4 * side_length
  let max_perimeter_per_piece := 10
  (num_pieces * max_perimeter_per_piece - outer_perimeter) / 2 = 1065 :=
by
  sorry

end max_total_cut_length_l152_152519


namespace geometric_sequence_a_div_n_sum_first_n_terms_l152_152275

variable {a : ℕ → ℝ} -- sequence a_n
variable {S : ℕ → ℝ} -- sum of first n terms S_n

axiom S_recurrence {n : ℕ} (hn : n > 0) : 
  S (n + 1) = S n + (n + 1) / (3 * n) * a n

axiom a_1 : a 1 = 1

theorem geometric_sequence_a_div_n :
  ∃ (r : ℝ), ∀ {n : ℕ} (hn : n > 0), (a n / n) = r^n := 
sorry

theorem sum_first_n_terms (n : ℕ) :
  S n = (9 / 4) - ((9 / 4) + (3 * n / 2)) * (1 / 3) ^ n :=
sorry

end geometric_sequence_a_div_n_sum_first_n_terms_l152_152275


namespace part_I_part_II_l152_152430

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

end part_I_part_II_l152_152430


namespace children_being_catered_l152_152512

-- Define the total meal units available
def meal_units_for_adults : ℕ := 70
def meal_units_for_children : ℕ := 90
def meals_eaten_by_adults : ℕ := 14
def remaining_meal_units : ℕ := meal_units_for_adults - meals_eaten_by_adults

theorem children_being_catered :
  (remaining_meal_units * meal_units_for_children) / meal_units_for_adults = 72 := by
{
  sorry
}

end children_being_catered_l152_152512


namespace find_a_l152_152693

noncomputable def circle1 (x y : ℝ) := x^2 + y^2 + 4 * y = 0

noncomputable def circle2 (x y a : ℝ) := x^2 + y^2 + 2 * (a - 1) * x + 2 * y + a^2 = 0

theorem find_a (a : ℝ) :
  (∀ x y, circle1 x y → circle2 x y a → false) → a = -2 :=
by sorry

end find_a_l152_152693


namespace correct_option_l152_152054

-- Defining the conditions for each option
def optionA (m n : ℝ) : Prop := (m / n)^7 = m^7 * n^(1/7)
def optionB : Prop := (4)^(4/12) = (-3)^(1/3)
def optionC (x y : ℝ) : Prop := ((x^3 + y^3)^(1/4)) = (x + y)^(3/4)
def optionD : Prop := (9)^(1/6) = 3^(1/3)

-- Asserting that option D is correct
theorem correct_option : optionD :=
by
  sorry

end correct_option_l152_152054


namespace sam_initial_investment_is_6000_l152_152608

variables (P : ℝ)
noncomputable def final_amount (P : ℝ) : ℝ :=
  P * (1 + 0.10 / 2) ^ (2 * 1)

theorem sam_initial_investment_is_6000 :
  final_amount 6000 = 6615 :=
by
  unfold final_amount
  sorry

end sam_initial_investment_is_6000_l152_152608


namespace Martha_reading_challenge_l152_152356

theorem Martha_reading_challenge :
  ∀ x : ℕ,
  (12 + 18 + 14 + 20 + 11 + 13 + 19 + 15 + 17 + x) / 10 = 15 ↔ x = 11 :=
by sorry

end Martha_reading_challenge_l152_152356


namespace cubic_inequality_l152_152834

theorem cubic_inequality (a b : ℝ) : (a > b) ↔ (a^3 > b^3) := sorry

end cubic_inequality_l152_152834


namespace combination_10_5_l152_152578

theorem combination_10_5 :
  (Nat.choose 10 5) = 2520 :=
by
  sorry

end combination_10_5_l152_152578


namespace find_f_five_thrids_l152_152879

noncomputable def f : ℝ → ℝ := sorry

lemma odd_fun (x : ℝ) : f (-x) = -f(x) :=
sorry

lemma f_transformation (x : ℝ) : f (1 + x) = f (-x) :=
sorry

lemma f_neg_inv : f (-1 / 3) = 1 / 3 :=
sorry

theorem find_f_five_thrids : f (5 / 3) = 1 / 3 :=
by
  -- application of the conditions stated
  have h1 : f(1 + (5 / 3 - 1)) = f(-(5 / 3 - 1)),
    from f_transformation (5 / 3 - 1),
  have h2 : f(-(5 / 3 - 1)) = -f(5 / 3 - 1),
    from odd_fun (5 / 3 - 1),
  have h3 : 5 / 3 - 1 = 2 / 3,
    by norm_num,
  rw [h3] at h1,
  rw [h2] at h1,
  have h4 : f (-1 / 3) = 1 / 3,
    from f_neg_inv,
  rw [←h4] at h1,
  exact h1,
sorry

end find_f_five_thrids_l152_152879


namespace students_taking_all_three_l152_152375

-- Definitions and Conditions
def total_students : ℕ := 25
def coding_students : ℕ := 12
def chess_students : ℕ := 15
def photography_students : ℕ := 10
def at_least_two_classes : ℕ := 10

-- Request to prove: Number of students taking all three classes
theorem students_taking_all_three (x y w z : ℕ) :
  (x + y + z + w = 10) →
  (coding_students - (10 - y) + chess_students - (10 - w) + (10 - x) = 21) →
  z = 4 :=
by
  intros
  -- Proof will go here
  sorry

end students_taking_all_three_l152_152375


namespace optimal_position_station_l152_152738

-- Definitions for the conditions
def num_buildings := 5
def building_workers (k : ℕ) : ℕ := if k ≤ 5 then k else 0
def distance_between_buildings := 50

-- Function to calculate the total walking distance
noncomputable def total_distance (x : ℝ) : ℝ :=
  |x| + 2 * |x - 50| + 3 * |x - 100| + 4 * |x - 150| + 5 * |x - 200|

-- Theorem statement
theorem optimal_position_station :
  ∃ x : ℝ, (∀ y : ℝ, total_distance x ≤ total_distance y) ∧ x = 150 :=
by
  sorry

end optimal_position_station_l152_152738


namespace red_stars_eq_35_l152_152863

-- Define the conditions
noncomputable def number_of_total_stars (x : ℕ) : ℕ := x + 20 + 15
noncomputable def red_star_frequency (x : ℕ) : ℚ := x / (number_of_total_stars x : ℚ)

-- Define the theorem statement
theorem red_stars_eq_35 : ∃ x : ℕ, red_star_frequency x = 0.5 ↔ x = 35 := sorry

end red_stars_eq_35_l152_152863


namespace move_line_down_eq_l152_152019

theorem move_line_down_eq (x y : ℝ) : (y = 2 * x) → (y - 3 = 2 * x - 3) :=
by
  sorry

end move_line_down_eq_l152_152019


namespace cube_identity_l152_152299

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l152_152299


namespace median_diff_expectation_le_variance_sqrt_l152_152899

open ProbabilityTheory

variables {Ω : Type*} {ℱ : measurable_space Ω}
  {P : MeasureTheory.ProbabilityMeasure ℱ}
  {ξ : Ω → ℝ}

def median (X : Ω → ℝ) := {m : ℝ | ∀ ε > 0, P({ω | X ω < m - ε}) ≤ 1/2 ∧ P({ω | X ω > m + ε}) ≤ 1/2}

theorem median_diff_expectation_le_variance_sqrt
  (μ : ℝ) (hμ : μ ∈ median ξ) :
  |μ - MeasureTheory.ProbabilityTheory.expectation P ξ| ≤ Real.sqrt (measure_variance P ξ) :=
by sorry

end median_diff_expectation_le_variance_sqrt_l152_152899


namespace janet_saves_time_l152_152813

theorem janet_saves_time (looking_time_per_day : ℕ := 8) (complaining_time_per_day : ℕ := 3) (days_per_week : ℕ := 7) :
  (looking_time_per_day + complaining_time_per_day) * days_per_week = 77 := 
sorry

end janet_saves_time_l152_152813


namespace probability_of_diff_ge_3_l152_152502

noncomputable def probability_diff_ge_3 : ℚ :=
  let elements := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let total_pairs := (elements.to_finset.card.choose 2 : ℕ)
  let valid_pairs := (total_pairs - 15 : ℕ) -- 15 pairs with a difference of less than 3
  valid_pairs / total_pairs

theorem probability_of_diff_ge_3 :
  probability_diff_ge_3 = 7 / 12 := by
  sorry

end probability_of_diff_ge_3_l152_152502


namespace sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three_l152_152258

theorem sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three : 
  ((Real.sqrt 7 + 2) * (Real.sqrt 7 - 2) = 3) := by
  sorry

end sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three_l152_152258


namespace pounds_over_weight_limit_l152_152456

-- Definitions based on conditions

def bookcase_max_weight : ℝ := 80

def weight_hardcover_book : ℝ := 0.5
def number_hardcover_books : ℕ := 70
def total_weight_hardcover_books : ℝ := number_hardcover_books * weight_hardcover_book

def weight_textbook : ℝ := 2
def number_textbooks : ℕ := 30
def total_weight_textbooks : ℝ := number_textbooks * weight_textbook

def weight_knick_knack : ℝ := 6
def number_knick_knacks : ℕ := 3
def total_weight_knick_knacks : ℝ := number_knick_knacks * weight_knick_knack

def total_weight_items : ℝ := total_weight_hardcover_books + total_weight_textbooks + total_weight_knick_knacks

theorem pounds_over_weight_limit : total_weight_items - bookcase_max_weight = 33 := by
  sorry

end pounds_over_weight_limit_l152_152456


namespace sum_of_nus_is_45_l152_152044

noncomputable def sum_of_valid_nu : ℕ :=
  ∑ ν in {ν | ν > 0 ∧ Nat.lcm ν 24 = 72}.toFinset, ν

theorem sum_of_nus_is_45 : sum_of_valid_nu = 45 := by
  sorry

end sum_of_nus_is_45_l152_152044


namespace perfect_square_divisors_of_450_l152_152136

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l152_152136


namespace gcd_lcm_product_24_36_l152_152555

-- Definitions for gcd, lcm, and product for given numbers, skipping proof with sorry
theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  -- Sorry used to skip proof
  sorry

end gcd_lcm_product_24_36_l152_152555


namespace rational_solution_system_l152_152266

theorem rational_solution_system (x y z t w : ℚ) :
  (t^2 - w^2 + z^2 = 2 * x * y) →
  (t^2 - y^2 + w^2 = 2 * x * z) →
  (t^2 - w^2 + x^2 = 2 * y * z) →
  x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intros h1 h2 h3
  sorry

end rational_solution_system_l152_152266


namespace find_x_l152_152856

theorem find_x (x : ℝ) (h : ∑' n : ℕ, (n + 1) * x ^ n = 9) : x = 2 / 3 :=
sorry

end find_x_l152_152856


namespace total_action_figures_l152_152870

def jerry_original_count : Nat := 4
def jerry_added_count : Nat := 6

theorem total_action_figures : jerry_original_count + jerry_added_count = 10 :=
by
  sorry

end total_action_figures_l152_152870


namespace marian_baked_cookies_l152_152355

theorem marian_baked_cookies :
  let cookies_per_tray := 12
  let trays_used := 23
  trays_used * cookies_per_tray = 276 :=
by
  sorry

end marian_baked_cookies_l152_152355


namespace silk_dyed_amount_l152_152923

-- Define the conditions
def yards_green : ℕ := 61921
def yards_pink : ℕ := 49500

-- Define the total calculation
def total_yards : ℕ := yards_green + yards_pink

-- State what needs to be proven: that the total yards is 111421
theorem silk_dyed_amount : total_yards = 111421 := by
  sorry

end silk_dyed_amount_l152_152923


namespace on_real_axis_in_first_quadrant_on_line_l152_152271

theorem on_real_axis (m : ℝ) : 
  (m = -3 ∨ m = 5) ↔ (m^2 - 2 * m - 15 = 0) := 
sorry

theorem in_first_quadrant (m : ℝ) : 
  (m < -3 ∨ m > 5) ↔ ((m^2 + 5 * m + 6 > 0) ∧ (m^2 - 2 * m - 15 > 0)) := 
sorry

theorem on_line (m : ℝ) : 
  (m = 1 ∨ m = -5 / 2) ↔ ((m^2 + 5 * m + 6) + (m^2 - 2 * m - 15) + 5 = 0) := 
sorry

end on_real_axis_in_first_quadrant_on_line_l152_152271


namespace num_ways_select_with_constraints_l152_152217

theorem num_ways_select_with_constraints (total_students : ℕ) (select_students : ℕ) (A_and_B : ℕ) (A_not_and_B : ℕ) :
  total_students = 10 → select_students = 6 → A_and_B = 70 → A_not_and_B = 140 → 
  (finset.card (finset.powerset_len select_students (finset.range total_students)) - A_and_B) = A_not_and_B :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end num_ways_select_with_constraints_l152_152217


namespace perfect_square_divisors_count_450_l152_152120

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l152_152120


namespace andrei_monthly_spending_l152_152797

noncomputable def original_price := 50
noncomputable def price_increase := 0.10
noncomputable def discount := 0.10
noncomputable def kg_per_month := 2

def new_price := original_price + original_price * price_increase
def discounted_price := new_price - new_price * discount
def monthly_spending := discounted_price * kg_per_month

theorem andrei_monthly_spending : monthly_spending = 99 := by
  sorry

end andrei_monthly_spending_l152_152797


namespace solve_equation_l152_152017

theorem solve_equation : ∀ (x : ℝ), x ≠ 2 → -2 * x^2 = (4 * x + 2) / (x - 2) → x = 1 :=
by
  intros x hx h_eq
  sorry

end solve_equation_l152_152017


namespace pos_int_satisfy_inequality_l152_152372

open Nat

def C (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))
def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

theorem pos_int_satisfy_inequality :
  {n : ℕ // 0 < n ∧ 2 * C n 3 ≤ A n 2} = {n // n = 3 ∨ n = 4 ∨ n = 5} :=
by
  sorry

end pos_int_satisfy_inequality_l152_152372


namespace asher_speed_l152_152255

theorem asher_speed :
  (5 * 60 ≠ 0) → (6600 / (5 * 60) = 22) :=
by
  intros h
  sorry

end asher_speed_l152_152255


namespace root_of_linear_equation_l152_152383

theorem root_of_linear_equation (b c : ℝ) (hb : b ≠ 0) :
  ∃ x : ℝ, 0 * x^2 + b * x + c = 0 → x = -c / b :=
by
  -- The proof steps would typically go here
  sorry

end root_of_linear_equation_l152_152383


namespace trains_cross_time_l152_152039

noncomputable def timeToCross (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let speed1_mps := speed1 * (5 / 18)
  let speed2_mps := speed2 * (5 / 18)
  let relative_speed := speed1_mps + speed2_mps
  let total_length := length1 + length2
  total_length / relative_speed

theorem trains_cross_time
  (length1 length2 : ℝ)
  (speed1 speed2 : ℝ)
  (h_length1 : length1 = 250)
  (h_length2 : length2 = 250)
  (h_speed1 : speed1 = 90)
  (h_speed2 : speed2 = 110) :
  timeToCross length1 length2 speed1 speed2 = 9 := 
by sorry

end trains_cross_time_l152_152039


namespace cube_identity_l152_152300

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l152_152300


namespace sequence_an_form_l152_152564

-- Definitions based on the given conditions
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := (n : ℝ)^2 * a n
def a_1 : ℝ := 1

-- The conjecture we need to prove
theorem sequence_an_form (a : ℕ → ℝ) (h₁ : ∀ n ≥ 2, sum_first_n_terms a n = (n : ℝ)^2 * a n)
  (h₂ : a 1 = a_1) :
  ∀ n ≥ 2, a n = 2 / (n * (n + 1)) :=
by
  sorry

end sequence_an_form_l152_152564


namespace amount_after_3_years_l152_152249

theorem amount_after_3_years (P t A' : ℝ) (R : ℝ) :
  P = 800 → t = 3 → A' = 992 →
  (800 * ((R + 3) / 100) * 3 = 192) →
  (A = P * (1 + (R / 100) * t)) →
  A = 1160 := by
  intros hP ht hA' hR hA
  sorry

end amount_after_3_years_l152_152249


namespace wall_building_time_l152_152995

theorem wall_building_time
  (m1 m2 : ℕ) 
  (d1 d2 : ℝ)
  (h1 : m1 = 20)
  (h2 : d1 = 3.0)
  (h3 : m2 = 30)
  (h4 : ∃ k, m1 * d1 = k ∧ m2 * d2 = k) :
  d2 = 2.0 :=
by
  sorry

end wall_building_time_l152_152995


namespace expression_evaluation_l152_152810

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) :=
by
  sorry

end expression_evaluation_l152_152810


namespace complement_intersection_U_l152_152235

-- Definitions of the sets based on the given conditions
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Definition of the complement of a set with respect to another set
def complement (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Statement asserting the equivalence
theorem complement_intersection_U :
  complement U (M ∩ N) = {1, 4} :=
by
  sorry

end complement_intersection_U_l152_152235


namespace candy_seller_initial_candies_l152_152665

-- Given conditions
def num_clowns : ℕ := 4
def num_children : ℕ := 30
def candies_per_person : ℕ := 20
def candies_left : ℕ := 20

-- Question: What was the initial number of candies?
def total_people : ℕ := num_clowns + num_children
def total_candies_given_out : ℕ := total_people * candies_per_person
def initial_candies : ℕ := total_candies_given_out + candies_left

theorem candy_seller_initial_candies : initial_candies = 700 :=
by
  sorry

end candy_seller_initial_candies_l152_152665


namespace sum_of_all_possible_radii_l152_152242

noncomputable def circle_center_and_tangent (r : ℝ) : Prop :=
  let C := (r, r) in
  let circleC_radius := r in
  let circleD_center := (5 : ℝ, 0 : ℝ) in
  let circleD_radius := (2 : ℝ) in
  (circleC_radius - 5)^2 + circleC_radius^2 = (circleC_radius + circleD_radius)^2

theorem sum_of_all_possible_radii : ∀ r : ℝ, circle_center_and_tangent r → (r = 7 + 2 * real.sqrt 7) ∨ (r = 7 - 2 * real.sqrt 7) → r + 7 - 2 * real.sqrt 7 = 14 :=
by
  intros r hcond hr;
  sorry

end sum_of_all_possible_radii_l152_152242


namespace marbles_total_is_260_l152_152411

/-- Define the number of marbles in each jar. -/
def jar1 : ℕ := 80
def jar2 : ℕ := 2 * jar1
def jar3 : ℕ := jar1 / 4

/-- The total number of marbles Courtney has. -/
def total_marbles : ℕ := jar1 + jar2 + jar3

/-- Proving that the total number of marbles is 260. -/
theorem marbles_total_is_260 : total_marbles = 260 := 
by
  sorry

end marbles_total_is_260_l152_152411


namespace sum_of_m_and_n_l152_152868

noncomputable section

variable {a b m n : ℕ}

theorem sum_of_m_and_n 
  (h1 : a = n * b)
  (h2 : (a + b) = m * (a - b)) :
  m + n = 5 :=
sorry

end sum_of_m_and_n_l152_152868


namespace avg_seven_consecutive_integers_l152_152905

variable (c d : ℕ)
variable (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7)

theorem avg_seven_consecutive_integers (c d : ℕ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 6 :=
sorry

end avg_seven_consecutive_integers_l152_152905


namespace midpoint_condition_l152_152029

theorem midpoint_condition (c : ℝ) :
  (∃ A B : ℝ × ℝ,
    A ∈ { p : ℝ × ℝ | p.2 = p.1^2 - 2 * p.1 - 3 } ∧
    B ∈ { p : ℝ × ℝ | p.2 = p.1^2 - 2 * p.1 - 3 } ∧
    A ∈ { p : ℝ × ℝ | p.2 = -p.1^2 + 4 * p.1 + c } ∧
    B ∈ { p : ℝ × ℝ | p.2 = -p.1^2 + 4 * p.1 + c } ∧
    ((A.1 + B.1) / 2 + (A.2 + B.2) / 2) = 2017
  ) ↔
  c = 4031 := sorry

end midpoint_condition_l152_152029


namespace correct_statement_l152_152976

theorem correct_statement (a b : ℝ) (ha : a < b) (hb : b < 0) : |a| / |b| > 1 :=
sorry

end correct_statement_l152_152976


namespace trigonometric_expression_l152_152093

theorem trigonometric_expression
  (α : ℝ)
  (h : (Real.tan α - 3) * (Real.sin α + Real.cos α + 3) = 0) :
  2 + (2 / 3) * Real.sin α ^ 2 + (1 / 4) * Real.cos α ^ 2 = 21 / 8 := 
by sorry

end trigonometric_expression_l152_152093


namespace prod_gcd_lcm_eq_864_l152_152547

theorem prod_gcd_lcm_eq_864 : 
  let a := 24
  let b := 36
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  gcd_ab * lcm_ab = 864 :=
by
  sorry

end prod_gcd_lcm_eq_864_l152_152547


namespace matrix_power_eigenvector_l152_152187

section MatrixEigen
variable (B : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ)

theorem matrix_power_eigenvector (h : B.mulVec ![3, -1] = ![-12, 4]) :
  (B ^ 5).mulVec ![3, -1] = ![-3072, 1024] := 
  sorry
end MatrixEigen

end matrix_power_eigenvector_l152_152187


namespace C_plus_D_l152_152221

theorem C_plus_D (D C : ℚ) (h : ∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 → (D * x - 17) / ((x - 3) * (x - 5)) = C / (x - 3) + 2 / (x - 5)) :
  C + D = 32 / 5 :=
by
  sorry

end C_plus_D_l152_152221


namespace number_of_perfect_square_divisors_of_450_l152_152150

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l152_152150


namespace book_length_l152_152480

theorem book_length (P : ℕ) (h1 : 2323 = (P - 2323) + 90) : P = 4556 :=
by
  sorry

end book_length_l152_152480


namespace find_A_and_B_l152_152817

theorem find_A_and_B (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ -6 → (5 * x - 3) / (x^2 + 3 * x - 18) = A / (x - 3) + B / (x + 6)) →
  A = 4 / 3 ∧ B = 11 / 3 :=
by
  intros h
  sorry

end find_A_and_B_l152_152817


namespace perfect_square_divisors_count_450_l152_152122

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l152_152122


namespace pipe_p_fills_cistern_in_12_minutes_l152_152038

theorem pipe_p_fills_cistern_in_12_minutes :
  (∃ (t : ℝ), 
    ∀ (q_fill_rate p_fill_rate : ℝ), 
      q_fill_rate = 1 / 15 ∧ 
      t > 0 ∧ 
      (4 * (1 / t + q_fill_rate) + 6 * q_fill_rate = 1) → t = 12) :=
sorry

end pipe_p_fills_cistern_in_12_minutes_l152_152038


namespace perfect_square_factors_450_l152_152114

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l152_152114


namespace num_real_roots_l152_152621

theorem num_real_roots (f : ℝ → ℝ)
  (h_eq : ∀ x, f x = 2 * x ^ 3 - 6 * x ^ 2 + 7)
  (h_interval : ∀ x, 0 < x ∧ x < 2 → f x < 0 ∧ f (2 - x) > 0) : 
  ∃! x, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end num_real_roots_l152_152621


namespace waiter_tables_l152_152530

theorem waiter_tables (total_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (tables : ℕ) :
  total_customers = 62 →
  left_customers = 17 →
  people_per_table = 9 →
  remaining_customers = total_customers - left_customers →
  tables = remaining_customers / people_per_table →
  tables = 5 := by
  sorry

end waiter_tables_l152_152530


namespace xn_plus_inv_xn_is_integer_l152_152598

theorem xn_plus_inv_xn_is_integer (x : ℝ) (hx : x ≠ 0) (k : ℤ) (h : x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by sorry

end xn_plus_inv_xn_is_integer_l152_152598


namespace solve_for_y_l152_152362

theorem solve_for_y (y : ℝ) (h_sum : (1 + 99) * 99 / 2 = 4950)
  (h_avg : (4950 + y) / 100 = 50 * y) : y = 4950 / 4999 :=
by
  sorry

end solve_for_y_l152_152362


namespace sum_of_possible_radii_l152_152243

theorem sum_of_possible_radii :
  ∃ r1 r2 : ℝ, 
    (∀ r, (r - 5)^2 + r^2 = (r + 2)^2 → r = r1 ∨ r = r2) ∧ 
    r1 + r2 = 14 :=
sorry

end sum_of_possible_radii_l152_152243


namespace inequality_system_has_three_integer_solutions_l152_152828

theorem inequality_system_has_three_integer_solutions (m : ℝ) :
  (∃ (s : finset ℤ), s.card = 3 ∧ ∀ x ∈ s, x + 5 > 0 ∧ x - m ≤ 1) ↔ -3 ≤ m ∧ m < -2 :=
by
  sorry

end inequality_system_has_three_integer_solutions_l152_152828


namespace cubic_sum_l152_152322

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l152_152322


namespace find_first_train_length_l152_152924

theorem find_first_train_length
  (length_second_train : ℝ)
  (initial_distance : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_minutes : ℝ) :
  length_second_train = 200 →
  initial_distance = 100 →
  speed_first_train_kmph = 54 →
  speed_second_train_kmph = 72 →
  time_minutes = 2.856914303998537 →
  ∃ (L : ℝ), L = 5699.52 :=
by
  sorry

end find_first_train_length_l152_152924


namespace tricycles_count_l152_152451

theorem tricycles_count (b t : ℕ) 
  (hyp1 : b + t = 10)
  (hyp2 : 2 * b + 3 * t = 26) : 
  t = 6 := 
by 
  sorry

end tricycles_count_l152_152451


namespace initial_ratio_milk_water_l152_152986

theorem initial_ratio_milk_water (M W : ℕ) 
  (h1 : M + W = 45) 
  (h2 : M = 3 * (W + 3)) 
  : M / W = 4 := 
sorry

end initial_ratio_milk_water_l152_152986


namespace news_spread_l152_152396

theorem news_spread {G : SimpleGraph (Fin 1000)} (connectedG : G.IsConnected) :
  ∃ (S : Finset (Fin 1000)), S.card = 90 ∧ ∀ v, ∃ u ∈ S, (G.path_length u v ≤ 10) :=
by
  sorry

end news_spread_l152_152396


namespace candy_bar_cost_l152_152085

theorem candy_bar_cost :
  ∀ (members : ℕ) (avg_candy_bars : ℕ) (total_earnings : ℝ), 
  members = 20 →
  avg_candy_bars = 8 →
  total_earnings = 80 →
  total_earnings / (members * avg_candy_bars) = 0.50 :=
by
  intros members avg_candy_bars total_earnings h_mem h_avg h_earn
  sorry

end candy_bar_cost_l152_152085


namespace distribute_money_equation_l152_152991

theorem distribute_money_equation (x : ℕ) (hx : x > 0) : 
  (10 : ℚ) / x = (40 : ℚ) / (x + 6) := 
sorry

end distribute_money_equation_l152_152991


namespace fish_too_small_l152_152404

theorem fish_too_small
    (ben_fish : ℕ) (judy_fish : ℕ) (billy_fish : ℕ) (jim_fish : ℕ) (susie_fish : ℕ)
    (total_filets : ℕ) (filets_per_fish : ℕ) :
    ben_fish = 4 →
    judy_fish = 1 →
    billy_fish = 3 →
    jim_fish = 2 →
    susie_fish = 5 →
    total_filets = 24 →
    filets_per_fish = 2 →
    (ben_fish + judy_fish + billy_fish + jim_fish + susie_fish) - (total_filets / filets_per_fish) = 3 := 
by 
  intros
  sorry

end fish_too_small_l152_152404


namespace perfect_square_factors_450_l152_152115

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l152_152115


namespace find_x_l152_152094

def vec := (ℝ × ℝ)

def a : vec := (1, 1)
def b (x : ℝ) : vec := (3, x)

def add_vec (v1 v2 : vec) : vec := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : vec) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_x (x : ℝ) (h : dot_product a (add_vec a (b x)) = 0) : x = -5 :=
by
  -- Proof steps (irrelevant for now)
  sorry

end find_x_l152_152094


namespace num_bijective_selfmaps_7fixedpoints_l152_152725

open Finset

-- Define the set M
def M : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the condition that a function is bijective
def bijective (f : ℕ → ℕ) : Prop := Function.Bijective f

-- Define the main theorem
theorem num_bijective_selfmaps_7fixedpoints : 
  ∃ f : M → M, bijective f ∧ (card (filter (λ x, f x = x) M) = 7) = 240 :=
by sorry

end num_bijective_selfmaps_7fixedpoints_l152_152725


namespace make_up_set_money_needed_l152_152830

theorem make_up_set_money_needed (makeup_cost gabby_money mom_money: ℤ) (h1: makeup_cost = 65) (h2: gabby_money = 35) (h3: mom_money = 20) :
  (makeup_cost - (gabby_money + mom_money)) = 10 :=
by {
  sorry
}

end make_up_set_money_needed_l152_152830


namespace max_product_of_three_numbers_l152_152906

theorem max_product_of_three_numbers (n : ℕ) (h_n_pos : 0 < n) :
  ∃ a b c : ℕ, (a + b + c = 3 * n + 1) ∧ (∀ a' b' c' : ℕ,
        (a' + b' + c' = 3 * n + 1) →
        a' * b' * c' ≤ a * b * c) ∧
    (a * b * c = n^3 + n^2) :=
by
  sorry

end max_product_of_three_numbers_l152_152906


namespace binom_np_p_div_p4_l152_152007

theorem binom_np_p_div_p4 (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (h3 : 3 < p) (hn : n % p = 1) : p^4 ∣ Nat.choose (n * p) p - n := 
sorry

end binom_np_p_div_p4_l152_152007


namespace A_is_11_years_older_than_B_l152_152337

-- Define the constant B as given in the problem
def B : ℕ := 41

-- Define the condition based on the problem statement
def condition (A : ℕ) := A + 10 = 2 * (B - 10)

-- Prove the main statement that A is 11 years older than B
theorem A_is_11_years_older_than_B (A : ℕ) (h : condition A) : A - B = 11 :=
by
  sorry

end A_is_11_years_older_than_B_l152_152337


namespace hair_growth_l152_152586

-- Define the length of Isabella's hair initially and the growth
def initial_length : ℕ := 18
def growth : ℕ := 4

-- Define the final length of the hair after growth
def final_length (initial_length : ℕ) (growth : ℕ) : ℕ := initial_length + growth

-- State the theorem that the final length is 22 inches
theorem hair_growth : final_length initial_length growth = 22 := 
by
  sorry

end hair_growth_l152_152586


namespace cube_surface_area_l152_152773

noncomputable def volume (x : ℝ) : ℝ := x ^ 3

noncomputable def surface_area (x : ℝ) : ℝ := 6 * x ^ 2

theorem cube_surface_area (x : ℝ) :
  surface_area x = 6 * x ^ 2 :=
by sorry

end cube_surface_area_l152_152773


namespace perfect_square_factors_450_l152_152129

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l152_152129


namespace blue_eyes_count_l152_152894

theorem blue_eyes_count (total_students students_both students_neither : ℕ)
  (ratio_blond_to_blue : ℕ → ℕ)
  (h_total : total_students = 40)
  (h_ratio : ratio_blond_to_blue 3 = 2)
  (h_both : students_both = 8)
  (h_neither : students_neither = 5) :
  ∃ y : ℕ, y = 18 :=
by
  sorry

end blue_eyes_count_l152_152894


namespace original_length_before_sharpening_l152_152867

/-- Define the current length of the pencil after sharpening -/
def current_length : ℕ := 14

/-- Define the length of the pencil that was sharpened off -/
def sharpened_off_length : ℕ := 17

/-- Prove that the original length of the pencil before sharpening was 31 inches -/
theorem original_length_before_sharpening : current_length + sharpened_off_length = 31 := by
  sorry

end original_length_before_sharpening_l152_152867


namespace xy_cubed_identity_l152_152305

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l152_152305


namespace cubic_sum_l152_152317

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l152_152317


namespace problem1_l152_152518

variable (α : ℝ)

theorem problem1 (h : Real.tan α = -3/4) : 
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (11 * π / 2 - α) * Real.sin (9 * π / 2 + α)) = -3/4 := 
sorry

end problem1_l152_152518


namespace geometric_sequence_first_term_l152_152219

theorem geometric_sequence_first_term (a r : ℚ) (third_term fourth_term : ℚ) 
  (h1 : third_term = a * r^2)
  (h2 : fourth_term = a * r^3)
  (h3 : third_term = 27)
  (h4 : fourth_term = 36) : 
  a = 243 / 16 :=
by
  sorry

end geometric_sequence_first_term_l152_152219


namespace find_ordered_pair_l152_152087

open Polynomial

theorem find_ordered_pair (a b : ℝ) :
  (∀ x : ℝ, (((x^3 + a * x^2 + 17 * x + 10 = 0) ∧ (x^3 + b * x^2 + 20 * x + 12 = 0)) → 
  (x = -6 ∧ y = -7))) :=
sorry

end find_ordered_pair_l152_152087


namespace x_cubed_plus_y_cubed_l152_152326

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l152_152326


namespace A_plus_2B_plus_4_is_perfect_square_l152_152888

theorem A_plus_2B_plus_4_is_perfect_square (n : ℕ) (hn : 0 < n) :
  let A := (4 / 9) * (10^(2*n) - 1)
  let B := (8 / 9) * (10^n - 1)
  ∃ k : ℚ, (A + 2 * B + 4) = k^2 :=
by
  let A := (4 / 9) * (10^(2*n) - 1)
  let B := (8 / 9) * (10^n - 1)
  use ((2/3) * (10^n + 2))
  sorry

end A_plus_2B_plus_4_is_perfect_square_l152_152888


namespace perfect_square_divisors_count_450_l152_152121

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l152_152121


namespace combination_sum_eq_l152_152952

theorem combination_sum_eq :
  ∀ (n : ℕ), (2 * n ≥ 10 - 2 * n) ∧ (3 + n ≥ 2 * n) →
  Nat.choose (2 * n) (10 - 2 * n) + Nat.choose (3 + n) (2 * n) = 16 :=
by
  intro n h
  cases' h with h1 h2
  sorry

end combination_sum_eq_l152_152952


namespace numPerfectSquareFactorsOf450_l152_152146

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l152_152146


namespace division_theorem_l152_152183

theorem division_theorem (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y^(y^2) - 2 * y^(y + 1) + 1) := 
by
  sorry

end division_theorem_l152_152183


namespace most_likely_maximum_people_in_room_l152_152936

theorem most_likely_maximum_people_in_room :
  ∃ k, 1 ≤ k ∧ k ≤ 3000 ∧
    (∃ p : ℕ → ℕ → ℕ → ℕ, (p 1000 1000 1000) = 1019) ∧
    (∀ a b c : ℕ, a + b + c = 3000 → a ≤ 1019 ∧ b ≤ 1019 ∧ c ≤ 1019 → max a (max b c) = 1019) :=
sorry

end most_likely_maximum_people_in_room_l152_152936


namespace min_value_48_l152_152278

noncomputable def min_value {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 1) : ℝ :=
  1 / a + 27 / b

theorem min_value_48 {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 1) : 
  min_value ha hb h = 48 := 
sorry

end min_value_48_l152_152278


namespace football_count_white_patches_count_l152_152634

theorem football_count (x : ℕ) (footballs : ℕ) (students : ℕ) (h1 : students - 9 = footballs + 9) (h2 : students = 2 * footballs + 9) : footballs = 27 :=
sorry

theorem white_patches_count (white_patches : ℕ) (h : 2 * 12 * 5 = 6 * white_patches) : white_patches = 20 :=
sorry

end football_count_white_patches_count_l152_152634


namespace sum_of_possible_N_values_l152_152915

theorem sum_of_possible_N_values (a b c N : ℕ) (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = 2 * (a + b)) :
  ∃ sum_N : ℕ, sum_N = 672 :=
by
  sorry

end sum_of_possible_N_values_l152_152915


namespace root_in_interval_l152_152890

noncomputable def f (x : ℝ) : ℝ := x + Real.log x - 3

theorem root_in_interval : ∃ m, f m = 0 ∧ 2 < m ∧ m < 3 :=
by
  sorry

end root_in_interval_l152_152890


namespace solve_quadratic_l152_152015

theorem solve_quadratic (x : ℝ) (h₁ : x > 0) (h₂ : 3 * x^2 + 7 * x - 20 = 0) : x = 5 / 3 :=
sorry

end solve_quadratic_l152_152015


namespace inequality_a2b3c_l152_152353

theorem inequality_a2b3c {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : 
  a + 2 * b + 3 * c ≥ 9 :=
sorry

end inequality_a2b3c_l152_152353


namespace value_is_85_over_3_l152_152436

theorem value_is_85_over_3 (a b : ℚ)  (h1 : 3 * a + 6 * b = 48) (h2 : 8 * a + 4 * b = 84) : 2 * a + 3 * b = 85 / 3 := 
by {
  -- Proof will go here
  sorry
}

end value_is_85_over_3_l152_152436


namespace perfect_square_divisors_of_450_l152_152133

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l152_152133


namespace inhabitable_land_fraction_l152_152440

theorem inhabitable_land_fraction (total_surface not_water_covered initially_inhabitable tech_advancement_viable : ℝ)
  (h1 : not_water_covered = 1 / 3 * total_surface)
  (h2 : initially_inhabitable = 1 / 3 * not_water_covered)
  (h3 : tech_advancement_viable = 1 / 2 * (not_water_covered - initially_inhabitable)) :
  (initially_inhabitable + tech_advancement_viable) / total_surface = 2 / 9 := 
sorry

end inhabitable_land_fraction_l152_152440


namespace line_parallel_not_coincident_l152_152442

theorem line_parallel_not_coincident (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ k : ℝ, (∀ x y : ℝ, a * x + 2 * y + 6 = k * (x + (a - 1) * y + (a^2 - 1))) →
  a = -1 :=
by
  sorry

end line_parallel_not_coincident_l152_152442


namespace unique_homomorphism_is_identity_l152_152707

-- Defining the graph structure
structure Graph :=
  (V : Type) -- Vertices
  (E : V → V → Prop) -- Edges (undirected)

-- Given graph with vertices A, B, C, and D
def G : Graph :=
  { V := { A, B, C, D },
    E := λ x y, (x = A ∧ y = B) ∨ (x = B ∧ y = A)
             ∨ (x = A ∧ y = C) ∨ (x = C ∧ y = A)
             ∨ (x = A ∧ y = D) ∨ (x = D ∧ y = A)
             ∨ (x = B ∧ y = C) ∨ (x = C ∧ y = B)
             ∨ (x = B ∧ y = D) ∨ (x = D ∧ y = B)
             ∨ (x = C ∧ y = D) ∨ (x = D ∧ y = C) }

-- Defining a graph homomorphism
def graph_homomorphism (G H : Graph) :=
  { f : G.V → H.V // ∀ (x y : G.V), G.E x y → H.E (f x) (f y) }

-- The identity map on a graph
def id_homomorphism (G : Graph) : graph_homomorphism G G :=
  ⟨id, by { intros x y hxy, exact hxy }⟩

-- Stating the theorem: The only graph homomorphism from G to itself is the identity map
theorem unique_homomorphism_is_identity : ∀ f : graph_homomorphism G G, f = id_homomorphism G :=
sorry

end unique_homomorphism_is_identity_l152_152707


namespace card_picking_l152_152656

/-
Statement of the problem:
- A modified deck of cards has 65 cards.
- The deck is divided into 5 suits, each of which has 13 cards.
- The cards are placed in random order.
- Prove that the number of ways to pick two different cards from this deck with the order of picking being significant is 4160.
-/
theorem card_picking : (65 * 64) = 4160 := by
  sorry

end card_picking_l152_152656


namespace perfect_square_factors_450_l152_152113

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l152_152113


namespace f_zero_eq_zero_l152_152839

-- Define the problem conditions
variable {f : ℝ → ℝ}
variables (h_odd : ∀ x : ℝ, f (-x) = -f (x))
variables (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
variables (h_eq : ∀ x : ℝ, f (1 - x) - f (1 + x) + 2 * x = 0)
variables (h_mono : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ 1 → f x₁ ≤ f x₂)

-- State the theorem
theorem f_zero_eq_zero : f 0 = 0 :=
by sorry

end f_zero_eq_zero_l152_152839


namespace probability_difference_3_or_greater_l152_152500

noncomputable def set_of_numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def total_combinations : ℕ := (finset.image (λ (p : finset ℕ), p) (finset.powerset (finset.univ ∩ set_of_numbers))).card / 2

def pairs_with_difference_less_than_3 (S : finset ℕ) : finset (finset ℕ) :=
finset.filter (λ (p : finset ℕ), p.card = 2 ∧ abs ((p.min' sorry) - (p.max' sorry)) < 3) S

def successful_pairs (S : finset ℕ) : finset (finset ℕ) :=
finset.filter (λ (p : finset ℕ), p.card = 2 ∧ abs ((p.min' sorry) - (p.max' sorry)) ≥ 3) S
   
def probability (S : finset ℕ) : ℚ :=
(successful_pairs S).card.to_rat / total_combinations

theorem probability_difference_3_or_greater :
  probability set_of_numbers = 7/12 :=
sorry

end probability_difference_3_or_greater_l152_152500


namespace odd_function_property_l152_152882

theorem odd_function_property {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = - f x) (h2 : ∀ x, f (1 + x) = f (-x)) (h3 : f (-1 / 3) = 1 / 3) : f (5 / 3) = 1 / 3 := 
sorry

end odd_function_property_l152_152882


namespace tenth_term_is_correct_l152_152837

-- Definitions corresponding to the problem conditions
def sequence_term (n : ℕ) : ℚ := (-1)^n * (2 * n + 1) / (n^2 + 1)

-- Theorem statement for the equivalent proof problem
theorem tenth_term_is_correct : sequence_term 10 = 21 / 101 := by sorry

end tenth_term_is_correct_l152_152837


namespace child_l152_152620

noncomputable def C (G : ℝ) := 60 - 46
noncomputable def G := 130 - 60
noncomputable def ratio := (C G) / G

theorem child's_weight_to_grandmother's_weight_is_1_5 :
  ratio = 1 / 5 :=
by
  sorry

end child_l152_152620


namespace bill_due_in_9_months_l152_152758

-- Define the conditions
def true_discount : ℝ := 240
def face_value : ℝ := 2240
def interest_rate : ℝ := 0.16

-- Define the present value calculated from the true discount and face value
def present_value := face_value - true_discount

-- Define the time in months required to match the conditions
noncomputable def time_in_months : ℝ := 12 * ((face_value / present_value - 1) / interest_rate)

-- State the theorem that the bill is due in 9 months
theorem bill_due_in_9_months : time_in_months = 9 :=
by
  sorry

end bill_due_in_9_months_l152_152758


namespace distance_probability_l152_152864

theorem distance_probability :
  let speed := 5
  let num_roads := 8
  let total_outcomes := num_roads * (num_roads - 1)
  let favorable_outcomes := num_roads * 3
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = 0.375 :=
by
  sorry

end distance_probability_l152_152864


namespace units_digit_of_3_pow_7_pow_6_l152_152825

theorem units_digit_of_3_pow_7_pow_6 :
  (3 ^ (7 ^ 6) % 10) = 3 := 
sorry

end units_digit_of_3_pow_7_pow_6_l152_152825


namespace circle_equation_k_range_l152_152980

theorem circle_equation_k_range (k : ℝ) :
  ∀ x y: ℝ, x^2 + y^2 + 4*k*x - 2*y + 4*k^2 - k = 0 →
  k > -1 := 
sorry

end circle_equation_k_range_l152_152980


namespace cube_identity_l152_152302

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l152_152302


namespace expr_for_pos_x_min_value_l152_152969

section
variable {f : ℝ → ℝ}
variable {a : ℝ}

def even_func (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def func_def (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, x ≤ 0 → f x = 4^(-x) - a * 2^(-x)

-- Assuming f is even and specified as in the problem for x ≤ 0
axiom ev_func : even_func f
axiom f_condition : 0 < a

theorem expr_for_pos_x (f : ℝ → ℝ) (a : ℝ) (h1 : even_func f) (h2 : func_def f a) : 
  ∀ x, 0 < x → f x = 4^x - a * 2^x :=
sorry -- this aims to prove the function's form for positive x.

theorem min_value (f : ℝ → ℝ) (a : ℝ) (h1 : even_func f) (h2 : func_def f a) :
  (0 < a ∧ a ≤ 2 → ∃ x, 0 < x ∧ f x = 1 - a) ∧
  (2 < a → ∃ x, 0 < x ∧ f x = -a^2 / 4) :=
sorry -- this aims to prove the minimum value on the interval (0, +∞).
end

end expr_for_pos_x_min_value_l152_152969


namespace range_of_function_l152_152756

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem range_of_function : Set.Icc (-2 : ℝ) 7 = Set.image f (Set.Icc (-3 : ℝ) 2) :=
by
  sorry

end range_of_function_l152_152756


namespace sum_of_positive_integers_nu_lcm_72_l152_152043

theorem sum_of_positive_integers_nu_lcm_72:
  let ν_values := { ν | Nat.lcm ν 24 = 72 }
  ∑ ν in ν_values, ν = 180 := by
  sorry

end sum_of_positive_integers_nu_lcm_72_l152_152043


namespace person_speed_in_kmph_l152_152772

noncomputable def speed_calculation (distance_meters : ℕ) (time_minutes : ℕ) : ℝ :=
  let distance_km := (distance_meters : ℝ) / 1000
  let time_hours := (time_minutes : ℝ) / 60
  distance_km / time_hours

theorem person_speed_in_kmph :
  speed_calculation 1080 12 = 5.4 :=
by
  sorry

end person_speed_in_kmph_l152_152772


namespace isosceles_triangle_perimeter_l152_152575

theorem isosceles_triangle_perimeter (a b : ℕ) (c : ℕ) 
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 2) 
  (isosceles : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a) : 
  a + b + c = 12 :=
  sorry

end isosceles_triangle_perimeter_l152_152575


namespace find_k_l152_152391

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n - 2

theorem find_k :
  ∃ k : ℤ, k % 2 = 1 ∧ f (f (f k)) = 35 ∧ k = 29 := 
sorry

end find_k_l152_152391


namespace gcd_factorial_8_10_l152_152378

theorem gcd_factorial_8_10 (n : ℕ) (hn : n = 10! - 8!): gcd 8! 10! = 8! := by
  sorry

end gcd_factorial_8_10_l152_152378


namespace van_distance_l152_152529

theorem van_distance
  (D : ℝ)  -- distance the van needs to cover
  (S : ℝ)  -- original speed
  (h1 : D = S * 5)  -- the van takes 5 hours to cover the distance D
  (h2 : D = 62 * 7.5)  -- the van should maintain a speed of 62 kph to cover the same distance in 7.5 hours
  : D = 465 :=         -- prove that the distance D is 465 kilometers
by
  sorry

end van_distance_l152_152529


namespace volume_calc_l152_152771

noncomputable
def volume_of_open_box {l w : ℕ} (sheet_length : l = 48) (sheet_width : w = 38) (cut_length : ℕ) (cut_length_eq : cut_length = 8) : ℕ :=
  let new_length := l - 2 * cut_length
  let new_width := w - 2 * cut_length
  let height := cut_length
  new_length * new_width * height

theorem volume_calc : volume_of_open_box (sheet_length := rfl) (sheet_width := rfl) (cut_length := 8) (cut_length_eq := rfl) = 5632 :=
sorry

end volume_calc_l152_152771


namespace divisible_by_five_l152_152284

theorem divisible_by_five (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  ∃ k : ℤ, (x-y)^5 + (y-z)^5 + (z-x)^5 = 5 * k * (y-z) * (z-x) * (x-y) :=
  sorry

end divisible_by_five_l152_152284


namespace range_of_f_l152_152686

-- Define the function f
def f (x : ℕ) : ℤ := 2 * (x : ℤ) - 3

-- Define the domain
def domain : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the expected range
def expected_range : Finset ℤ := {-1, 1, 3, 5, 7}

-- Prove the range of f given the domain
theorem range_of_f : domain.image f = expected_range :=
  sorry

end range_of_f_l152_152686


namespace calculation_correct_l152_152660

def calculation : ℝ := 1.23 * 67 + 8.2 * 12.3 - 90 * 0.123

theorem calculation_correct : calculation = 172.20 := by
  sorry

end calculation_correct_l152_152660


namespace fraction_scaled_l152_152160

theorem fraction_scaled (x y : ℝ) :
  ∃ (k : ℝ), (k = 3 * y) ∧ ((5 * x + 3 * y) / (x + 3 * y) = 5 * ((x + (3 * y)) / (x + (3 * y)))) := 
  sorry

end fraction_scaled_l152_152160


namespace number_of_convex_quadrilaterals_l152_152482

-- Each definition used in Lean 4 statement should directly appear in the conditions problem.

variable {n : ℕ} -- Definition of n in Lean

-- Conditions
def distinct_points_on_circle (n : ℕ) : Prop := n = 10

-- Question and correct answer
theorem number_of_convex_quadrilaterals (h : distinct_points_on_circle n) : 
    (n.choose 4) = 210 := by
  sorry

end number_of_convex_quadrilaterals_l152_152482


namespace gcd_lcm_product_l152_152550

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [h₁, h₂]
  unfold Nat.gcd Nat.lcm
  sorry

end gcd_lcm_product_l152_152550


namespace find_numbers_l152_152210

theorem find_numbers (A B C : ℝ) 
  (h1 : A - B = 1860) 
  (h2 : 0.075 * A = 0.125 * B) 
  (h3 : 0.15 * B = 0.05 * C) : 
  A = 4650 ∧ B = 2790 ∧ C = 8370 := 
by
  sorry

end find_numbers_l152_152210


namespace evaluate_f_5_minus_f_neg_5_l152_152977

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x^3

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 1250 :=
by 
  sorry

end evaluate_f_5_minus_f_neg_5_l152_152977


namespace gcd_lcm_product_24_36_l152_152554

-- Definitions for gcd, lcm, and product for given numbers, skipping proof with sorry
theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  -- Sorry used to skip proof
  sorry

end gcd_lcm_product_24_36_l152_152554


namespace geometric_sequence_ninth_term_l152_152617

-- Given conditions
variables (a r : ℝ)
axiom fifth_term_condition : a * r^4 = 80
axiom seventh_term_condition : a * r^6 = 320

-- Goal: Prove that the ninth term is 1280
theorem geometric_sequence_ninth_term : a * r^8 = 1280 :=
by
  sorry

end geometric_sequence_ninth_term_l152_152617


namespace walk_direction_east_l152_152163

theorem walk_direction_east (m : ℤ) (h : m = -2023) : m = -(-2023) :=
by
  sorry

end walk_direction_east_l152_152163


namespace triangle_angle_side_inequality_l152_152769

variable {A B C : Type} -- Variables for points in the triangle
variable {a b : ℝ} -- Variables for the lengths of sides opposite to angles A and B
variable {A_angle B_angle : ℝ} -- Variables for the angles at A and B in triangle ABC

-- Define that we are in a triangle setting
def is_triangle (A B C : Type) := True

-- Define the assumption for the proof by contradiction
def assumption (a b : ℝ) := a ≤ b

theorem triangle_angle_side_inequality (h_triangle : is_triangle A B C)
(h_angle : A_angle > B_angle) 
(h_assumption : assumption a b) : a > b := 
sorry

end triangle_angle_side_inequality_l152_152769


namespace commodity_price_l152_152207

-- Define the variables for the prices of the commodities.
variable (x y : ℝ)

-- Conditions given in the problem.
def total_cost (x y : ℝ) : Prop := x + y = 827
def price_difference (x y : ℝ) : Prop := x = y + 127

-- The main statement to prove.
theorem commodity_price (x y : ℝ) (h1 : total_cost x y) (h2 : price_difference x y) : x = 477 :=
by
  sorry

end commodity_price_l152_152207


namespace total_carrots_computation_l152_152933

-- Definitions
def initial_carrots : ℕ := 19
def thrown_out_carrots : ℕ := 4
def next_day_carrots : ℕ := 46

def total_carrots (c1 c2 t : ℕ) : ℕ := (c1 - t) + c2

-- The statement to prove
theorem total_carrots_computation :
  total_carrots initial_carrots next_day_carrots thrown_out_carrots = 61 :=
by sorry

end total_carrots_computation_l152_152933


namespace toothpicks_in_20th_stage_l152_152497

theorem toothpicks_in_20th_stage :
  (3 + 3 * (20 - 1) = 60) :=
by
  sorry

end toothpicks_in_20th_stage_l152_152497


namespace largest_fraction_l152_152944

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem largest_fraction {m n : ℕ} (hm : 1000 ≤ m ∧ m ≤ 9999) (hn : 1000 ≤ n ∧ n ≤ 9999) (h_sum : digit_sum m = digit_sum n) :
  (m = 9900 ∧ n = 1089) ∧ m / n = 9900 / 1089 :=
by
  sorry

end largest_fraction_l152_152944


namespace find_f_five_thirds_l152_152884

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = - f x
def functional_equation (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (-x)
def specific_value (f : ℝ → ℝ) := f (-1/3) = 1/3

theorem find_f_five_thirds
  (hf_odd : odd_function f)
  (hf_fe : functional_equation f)
  (hf_value : specific_value f) :
  f (5 / 3) = 1 / 3 := by
  sorry

end find_f_five_thirds_l152_152884


namespace expand_square_binomial_l152_152259

variable (m n : ℝ)

theorem expand_square_binomial : (3 * m - n) ^ 2 = 9 * m ^ 2 - 6 * m * n + n ^ 2 :=
by
  sorry

end expand_square_binomial_l152_152259


namespace lunas_phone_bill_percentage_l152_152464

variables (H F P : ℝ)

theorem lunas_phone_bill_percentage :
  F = 0.60 * H ∧ H + F = 240 ∧ H + F + P = 249 →
  (P / F) * 100 = 10 :=
by
  intros
  sorry

end lunas_phone_bill_percentage_l152_152464


namespace binomial_sum_zero_l152_152473

open BigOperators

theorem binomial_sum_zero {n m : ℕ} (h1 : 1 ≤ m) (h2 : m < n) :
  ∑ k in Finset.range (n + 1), (-1 : ℤ) ^ k * k ^ m * Nat.choose n k = 0 :=
by
  sorry

end binomial_sum_zero_l152_152473


namespace xy_cubed_identity_l152_152308

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l152_152308


namespace gcd_8_10_l152_152379

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_8_10 : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := 
by
  sorry

end gcd_8_10_l152_152379


namespace no_real_solution_l152_152818

theorem no_real_solution :
  ¬ ∃ x : ℝ, (1 / (x + 2) + 8 / (x + 6) ≥ 2) ∧ (5 / (x + 1) - 2 ≤ 1) :=
by
  sorry

end no_real_solution_l152_152818


namespace perfect_square_factors_450_l152_152157

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l152_152157


namespace travel_time_l152_152737

namespace NatashaSpeedProblem

def distance : ℝ := 60
def speed_limit : ℝ := 50
def speed_over_limit : ℝ := 10
def actual_speed : ℝ := speed_limit + speed_over_limit

theorem travel_time : (distance / actual_speed) = 1 := by
  sorry

end NatashaSpeedProblem

end travel_time_l152_152737


namespace intersection_of_solution_sets_solution_set_of_modified_inequality_l152_152692

open Set Real

theorem intersection_of_solution_sets :
  let A := {x | x ^ 2 - 2 * x - 3 < 0}
  let B := {x | x ^ 2 + x - 6 < 0}
  A ∩ B = {x | -1 < x ∧ x < 2} := by {
  sorry
}

theorem solution_set_of_modified_inequality :
  let A := {x | x ^ 2 + (-1) * x + (-2) < 0}
  A = {x | true} := by {
  sorry
}

end intersection_of_solution_sets_solution_set_of_modified_inequality_l152_152692


namespace oliver_cards_l152_152468

variable {MC AB BG : ℕ}

theorem oliver_cards : 
  (BG = 48) → 
  (BG = 3 * AB) → 
  (MC = 2 * AB) → 
  MC = 32 := 
by 
  intros h1 h2 h3
  sorry

end oliver_cards_l152_152468


namespace abscissa_range_of_point_P_l152_152742

-- Definitions based on the conditions from the problem
def y_function (x : ℝ) : ℝ := 4 - 3 * x
def point_P (x y : ℝ) : Prop := y = y_function x
def ordinate_greater_than_negative_five (y : ℝ) : Prop := y > -5

-- Theorem statement combining the above definitions
theorem abscissa_range_of_point_P (x y : ℝ) :
  point_P x y →
  ordinate_greater_than_negative_five y →
  x < 3 :=
sorry

end abscissa_range_of_point_P_l152_152742


namespace probability_sum_leq_12_l152_152763

theorem probability_sum_leq_12 (dice1 dice2 : ℕ) (h1 : 1 ≤ dice1 ∧ dice1 ≤ 8) (h2 : 1 ≤ dice2 ∧ dice2 ≤ 8) :
  (∃ outcomes : ℕ, (outcomes = 64) ∧ 
   (∃ favorable : ℕ, (favorable = 54) ∧ 
   (favorable / outcomes = 27 / 32))) :=
sorry

end probability_sum_leq_12_l152_152763


namespace johns_donation_is_correct_l152_152222

/-
Conditions:
1. Alice, Bob, and Carol donated different amounts.
2. The ratio of Alice's, Bob's, and Carol's donations is 3:2:5.
3. The sum of Alice's and Bob's donations is $120.
4. The average contribution increases by 50% and reaches $75 per person after John donates.

The statement to prove:
John's donation is $240.
-/

def donations_ratio : ℕ × ℕ × ℕ := (3, 2, 5)
def sum_Alice_Bob : ℕ := 120
def new_avg_after_john : ℕ := 75
def num_people_before_john : ℕ := 3
def avg_increase_factor : ℚ := 1.5

theorem johns_donation_is_correct (A B C J : ℕ) 
  (h1 : A * 2 = B * 3) 
  (h2 : B * 5 = C * 2) 
  (h3 : A + B = sum_Alice_Bob) 
  (h4 : (A + B + C) / num_people_before_john = 80) 
  (h5 : ((A + B + C + J) / (num_people_before_john + 1)) = new_avg_after_john) :
  J = 240 := 
sorry

end johns_donation_is_correct_l152_152222


namespace cat_food_finished_on_sunday_l152_152900

def cat_morning_consumption : ℚ := 1 / 2
def cat_evening_consumption : ℚ := 1 / 3
def total_food : ℚ := 10
def daily_consumption : ℚ := cat_morning_consumption + cat_evening_consumption
def days_to_finish_food (total_food daily_consumption : ℚ) : ℚ :=
  total_food / daily_consumption

theorem cat_food_finished_on_sunday :
  days_to_finish_food total_food daily_consumption = 7 := 
sorry

end cat_food_finished_on_sunday_l152_152900


namespace other_number_l152_152058

theorem other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 2310) (h_hcf : Nat.gcd A B = 30) (h_A : A = 770) : B = 90 :=
by
  -- The proof is omitted here.
  sorry

end other_number_l152_152058


namespace pen_sales_average_l152_152056

theorem pen_sales_average (d : ℕ) (h1 : 96 + 44 * d > 0) (h2 : (96 + 44 * d) / (d + 1) = 48) : d = 12 :=
by
  sorry

end pen_sales_average_l152_152056


namespace isosceles_triangle_perimeter_l152_152572

def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ b = c ∨ a = c) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 5) :
  ∃ c, is_isosceles_triangle a b c ∧ a + b + c = 12 :=
by {
  use 5,
  split,
  simp [is_isosceles_triangle, h1, h2],
  split,
  linarith,
  split,
  linarith,
  linarith,
  ring,
}

end isosceles_triangle_perimeter_l152_152572


namespace intersection_M_N_l152_152726

open Set Real

def M : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def N : Set ℝ := {x | log x / log 2 ≤ 1}

theorem intersection_M_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_M_N_l152_152726


namespace complete_square_l152_152075

theorem complete_square (x : ℝ) : (x^2 - 2 * x - 2 = 0) → ((x - 1)^2 = 3) :=
by
  intro h
  sorry

end complete_square_l152_152075


namespace infinite_solutions_x2_y2_z2_x3_y3_z3_l152_152013

-- Define the parametric forms
def param_x (k : ℤ) := k * (2 * k^2 + 1)
def param_y (k : ℤ) := 2 * k^2 + 1
def param_z (k : ℤ) := -k * (2 * k^2 + 1)

-- Prove the equation
theorem infinite_solutions_x2_y2_z2_x3_y3_z3 :
  ∀ k : ℤ, param_x k ^ 2 + param_y k ^ 2 + param_z k ^ 2 = param_x k ^ 3 + param_y k ^ 3 + param_z k ^ 3 :=
by
  intros k
  -- Calculation needs to be proved here, we place a placeholder for now
  sorry

end infinite_solutions_x2_y2_z2_x3_y3_z3_l152_152013


namespace solve_problem_l152_152472

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem solve_problem :
  is_prime 2017 :=
by
  have h1 : 2017 > 1 := by linarith
  have h2 : ∀ m : ℕ, m ∣ 2017 → m = 1 ∨ m = 2017 :=
    sorry
  exact ⟨h1, h2⟩

end solve_problem_l152_152472


namespace num_rectangles_on_grid_l152_152697

theorem num_rectangles_on_grid :
  let points := {(0,0), (0,5), (0,10), (0,15), (5,0), (5,5), (5,10), (5,15), (10,0), (10,5), (10,10), (10,15), (15,0), (15,5), (15,10), (15,15)} in
  let vertical_lines := {0, 5, 10, 15} in
  let horizontal_lines := {0, 5, 10, 15} in
  (vertical_lines.card.choose 2) * (horizontal_lines.card.choose 2) = 36 := 
by
  let points := {(0,0), (0,5), (0,10), (0,15), (5,0), (5,5), (5,10), (5,15), (10,0), (10,5), (10,10), (10,15), (15,0), (15,5), (15,10), (15,15)}
  let vertical_lines := {0, 5, 10, 15}
  let horizontal_lines := {0, 5, 10, 15}
  have h_vertical := vertical_lines.card_choose_2
  have h_horizontal := horizontal_lines.card_choose_2
  exact mul_eq_one

end num_rectangles_on_grid_l152_152697


namespace count_multiples_of_34_with_two_odd_divisors_l152_152670

-- Define a predicate to check if a number has exactly 2 odd natural divisors
def has_exactly_two_odd_divisors (n : ℕ) : Prop :=
  (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ d1 * d2 = n ∧ d1 % 2 = 1 ∧ d2 % 2 = 1) ∧
  (∀ d : ℕ, d ∣ n → (d % 2 = 0 ∨ d = d1 ∨ d = d2))

-- Main theorem to prove the number of integers that satisfy the given conditions
theorem count_multiples_of_34_with_two_odd_divisors : 
  let valid_numbers := {n : ℕ | n ≤ 3400 ∧ n % 34 = 0 ∧ has_exactly_two_odd_divisors n} in
  valid_numbers.to_finset.card = 6 :=
by
  sorry

end count_multiples_of_34_with_two_odd_divisors_l152_152670


namespace trees_died_more_than_survived_l152_152694

theorem trees_died_more_than_survived :
  ∀ (initial_trees survived_percent : ℕ),
    initial_trees = 25 →
    survived_percent = 40 →
    (initial_trees * survived_percent / 100) + (initial_trees - initial_trees * survived_percent / 100) -
    (initial_trees * survived_percent / 100) = 5 :=
by
  intro initial_trees survived_percent initial_trees_eq survived_percent_eq
  sorry

end trees_died_more_than_survived_l152_152694


namespace age_of_B_l152_152515

variable (a b : ℕ)

-- Conditions
def condition1 := a + 10 = 2 * (b - 10)
def condition2 := a = b + 5

-- The proof goal
theorem age_of_B (h1 : condition1 a b) (h2 : condition2 a b) : b = 35 := by
  sorry

end age_of_B_l152_152515


namespace parabola_vertex_at_origin_axis_x_passing_point_parabola_vertex_at_origin_axis_y_distance_focus_l152_152673

-- Define the first parabola proof problem
theorem parabola_vertex_at_origin_axis_x_passing_point :
  (∃ (m : ℝ), ∀ (x y : ℝ), y^2 = m * x ↔ (y, x) = (0, 0) ∨ (x = 6 ∧ y = -3)) → 
  ∃ m : ℝ, m = 1.5 ∧ (y^2 = m * x) :=
sorry

-- Define the second parabola proof problem
theorem parabola_vertex_at_origin_axis_y_distance_focus :
  (∃ (p : ℝ), ∀ (x y : ℝ), x^2 = 4 * p * y ↔ (y, x) = (0, 0) ∨ (p = 3)) → 
  ∃ q : ℝ, q = 12 ∧ (x^2 = q * y ∨ x^2 = -q * y) :=
sorry

end parabola_vertex_at_origin_axis_x_passing_point_parabola_vertex_at_origin_axis_y_distance_focus_l152_152673


namespace _l152_152945

open Nat

/-- Function to check the triangle inequality theorem -/
def canFormTriangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

example : canFormTriangle 6 4 5 := by
  /- Proof omitted -/
  sorry

end _l152_152945


namespace parallel_lines_not_coincident_l152_152444

theorem parallel_lines_not_coincident (a : ℝ) :
  (∀ x y : ℝ, ax + 2 * y + 6 = 0) ∧
  (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ (b : ℝ), ∀ x y : ℝ, ax + 2 * y + b = 0 ∧ x + (a - 1) * y + b = 0 →
  a = -1 :=
by
  sorry

end parallel_lines_not_coincident_l152_152444


namespace find_line_equation_l152_152938

-- Define point A
def point_A : ℝ × ℝ := (-3, 4)

-- Define the conditions
def passes_through_point_A (line_eq : ℝ → ℝ → ℝ) : Prop :=
  line_eq (-3) 4 = 0

def intercept_condition (line_eq : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ line_eq (2 * a) 0 = 0 ∧ line_eq 0 a = 0

-- Define the equations of the line
def line1 (x y : ℝ) : ℝ := 3 * y + 4 * x
def line2 (x y : ℝ) : ℝ := 2 * x - y - 5

-- Statement of the problem
theorem find_line_equation : 
  (passes_through_point_A line1 ∧ intercept_condition line1) ∨
  (passes_through_point_A line2 ∧ intercept_condition line2) :=
sorry

end find_line_equation_l152_152938


namespace gcd_fact8_fact10_l152_152380

-- Define the factorials
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- State the problem conditions
theorem gcd_fact8_fact10 : gcd (fact 8) (fact 10) = fact 8 := by
  sorry

end gcd_fact8_fact10_l152_152380


namespace dinner_customers_l152_152240

theorem dinner_customers 
    (breakfast : ℕ)
    (lunch : ℕ)
    (total_friday : ℕ)
    (H : breakfast = 73)
    (H1 : lunch = 127)
    (H2 : total_friday = 287) :
  (breakfast + lunch + D = total_friday) → D = 87 := by
  sorry

end dinner_customers_l152_152240


namespace set_equality_l152_152917

noncomputable def alpha_set : Set ℝ := {α | ∃ k : ℤ, α = k * Real.pi / 2 - Real.pi / 5 ∧ (-Real.pi < α ∧ α < Real.pi)}

theorem set_equality : alpha_set = {-Real.pi / 5, -7 * Real.pi / 10, 3 * Real.pi / 10, 4 * Real.pi / 5} :=
by
  -- proof omitted
  sorry

end set_equality_l152_152917


namespace distance_sum_identity_l152_152426

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

end distance_sum_identity_l152_152426


namespace gcd_of_powers_l152_152767

theorem gcd_of_powers (m n : ℕ) (h1 : m = 2^2016 - 1) (h2 : n = 2^2008 - 1) : 
  Nat.gcd m n = 255 :=
by
  -- (Definitions and steps are omitted as only the statement is required)
  sorry

end gcd_of_powers_l152_152767


namespace total_planks_needed_l152_152557

theorem total_planks_needed (large_planks small_planks : ℕ) (h1 : large_planks = 37) (h2 : small_planks = 42) : large_planks + small_planks = 79 :=
by
  sorry

end total_planks_needed_l152_152557


namespace amount_C_l152_152607

theorem amount_C (A_amt B_amt C_amt : ℚ)
  (h1 : A_amt + B_amt + C_amt = 527)
  (h2 : A_amt = (2 / 3) * B_amt)
  (h3 : B_amt = (1 / 4) * C_amt) :
  C_amt = 372 :=
sorry

end amount_C_l152_152607


namespace find_r_l152_152544

open Real

noncomputable def cond (r : ℝ) : Prop :=
  log 49 (3 * r - 2) = -1 / 2

theorem find_r (r : ℝ) (h : cond r) : r = 5 / 7 :=
  sorry

end find_r_l152_152544


namespace remainder_when_concat_numbers_1_to_54_div_55_l152_152595

def concat_numbers (n : ℕ) : ℕ :=
  let digits x := x.digits 10
  (List.range n).bind digits |> List.reverse |> List.foldl (λ acc x => acc * 10 + x) 0

theorem remainder_when_concat_numbers_1_to_54_div_55 :
  let M := concat_numbers 55
  M % 55 = 44 :=
by
  sorry

end remainder_when_concat_numbers_1_to_54_div_55_l152_152595


namespace number_of_paintings_l152_152061

def is_valid_painting (grid : Matrix (Fin 3) (Fin 3) Bool) : Prop :=
  ∀ i j, grid i j = true → 
    (∀ k, k.succ < 3 → grid k j = true → ¬ grid (k.succ) j = false) ∧
    (∀ l, l.succ < 3 → grid i l = true → ¬ grid i (l.succ) = false)

theorem number_of_paintings : 
  ∃ n, n = 50 ∧ 
       ∃ f : Finset (Matrix (Fin 3) (Fin 3) Bool), 
         (∀ grid ∈ f, is_valid_painting grid) ∧ 
         Finset.card f = n :=
sorry

end number_of_paintings_l152_152061


namespace problem_equation_l152_152909

def interest_rate : ℝ := 0.0306
def principal : ℝ := 5000
def interest_tax : ℝ := 0.20

theorem problem_equation (x : ℝ) :
  x + principal * interest_rate * interest_tax = principal * (1 + interest_rate) :=
sorry

end problem_equation_l152_152909


namespace supplements_of_congruent_angles_are_congruent_l152_152542

theorem supplements_of_congruent_angles_are_congruent (a b : ℝ) (h1 : a + \degree(180) = b + \degree(180)) : a = b :=
sorry

end supplements_of_congruent_angles_are_congruent_l152_152542


namespace sequence_an_formula_l152_152891

theorem sequence_an_formula (a : ℕ → ℝ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, a (n + 1) = a n^2 - n * a n + 1) :
  ∀ n : ℕ, a n = n + 1 :=
sorry

end sequence_an_formula_l152_152891


namespace one_over_a5_eq_30_l152_152566

noncomputable def S : ℕ → ℝ
| n => n / (n + 1)

noncomputable def a (n : ℕ) := if n = 0 then S 0 else S n - S (n - 1)

theorem one_over_a5_eq_30 :
  (1 / a 5) = 30 :=
by
  sorry

end one_over_a5_eq_30_l152_152566


namespace ghost_enter_exit_ways_l152_152786

theorem ghost_enter_exit_ways : 
  (∃ (enter_win : ℕ) (exit_win : ℕ), enter_win ≠ exit_win ∧ 1 ≤ enter_win ∧ enter_win ≤ 8 ∧ 1 ≤ exit_win ∧ exit_win ≤ 8) →
  ∃ (ways : ℕ), ways = 8 * 7 :=
by
  sorry

end ghost_enter_exit_ways_l152_152786


namespace total_hours_l152_152739

variable (K : ℕ) (P : ℕ) (M : ℕ)

-- Conditions:
axiom h1 : P = 2 * K
axiom h2 : P = (1 / 3 : ℝ) * M
axiom h3 : M = K + 105

-- Goal: Proving the total number of hours is 189
theorem total_hours : K + P + M = 189 := by
  sorry

end total_hours_l152_152739


namespace area_BEIH_l152_152164

noncomputable def point (x y : ℝ) : ℝ × ℝ := (x, y)

noncomputable def area_quad (B E I H : ℝ × ℝ) : ℝ :=
  (1/2) * ((B.1 * E.2 + E.1 * I.2 + I.1 * H.2 + H.1 * B.2) - (B.2 * E.1 + E.2 * I.1 + I.2 * H.1 + H.2 * B.1))

theorem area_BEIH :
  let A : ℝ × ℝ := point 0 3
  let B : ℝ × ℝ := point 0 0
  let C : ℝ × ℝ := point 3 0
  let D : ℝ × ℝ := point 3 3
  let E : ℝ × ℝ := point 0 2
  let F : ℝ × ℝ := point 1 0
  let I : ℝ × ℝ := point (3/10) 2.1
  let H : ℝ × ℝ := point (3/4) (3/4)
  area_quad B E I H = 1.0125 :=
by
  sorry

end area_BEIH_l152_152164


namespace purchase_price_of_first_commodity_l152_152205

-- Define the conditions
variable (price_first price_second : ℝ)
variable (h1 : price_first - price_second = 127)
variable (h2 : price_first + price_second = 827)

-- Prove the purchase price of the first commodity is $477
theorem purchase_price_of_first_commodity : price_first = 477 :=
by
  sorry

end purchase_price_of_first_commodity_l152_152205


namespace total_tickets_sold_is_336_l152_152941

-- Define the costs of the tickets
def cost_vip_ticket : ℕ := 45
def cost_ga_ticket : ℕ := 20

-- Define the total cost collected
def total_cost_collected : ℕ := 7500

-- Define the difference in the number of tickets sold
def vip_less_ga : ℕ := 276

-- Define the main theorem to be proved
theorem total_tickets_sold_is_336 (V G : ℕ) 
  (h1 : cost_vip_ticket * V + cost_ga_ticket * G = total_cost_collected)
  (h2 : V = G - vip_less_ga) : V + G = 336 :=
  sorry

end total_tickets_sold_is_336_l152_152941


namespace perfect_square_divisors_of_450_l152_152137

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l152_152137


namespace proof_problem_l152_152188

def sum_even_ints (n : ℕ) : ℕ := n * (n + 1)
def sum_odd_ints (n : ℕ) : ℕ := n^2
def sum_specific_primes : ℕ := [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97].sum

theorem proof_problem : (sum_even_ints 100 - sum_odd_ints 100) + sum_specific_primes = 1063 :=
by
  sorry

end proof_problem_l152_152188


namespace sum_of_three_integers_l152_152632

def three_positive_integers (x y z : ℕ) : Prop :=
  x + y = 2003 ∧ y - z = 1000

theorem sum_of_three_integers (x y z : ℕ) (h1 : x + y = 2003) (h2 : y - z = 1000) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) : 
  x + y + z = 2004 := 
by 
  sorry

end sum_of_three_integers_l152_152632


namespace scientific_notation_of_384000_l152_152471

theorem scientific_notation_of_384000 : 384000 = 3.84 * 10^5 :=
by
  sorry

end scientific_notation_of_384000_l152_152471


namespace inverse_proportion_points_l152_152330

theorem inverse_proportion_points (x1 x2 x3 : ℝ) :
  (10 / x1 = -5) →
  (10 / x2 = 2) →
  (10 / x3 = 5) →
  x1 < x3 ∧ x3 < x2 :=
by sorry

end inverse_proportion_points_l152_152330


namespace cubic_sum_identity_l152_152297

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l152_152297


namespace problem_solution_l152_152585

def triangle_with_parallel_vectors (a b c A B: ℝ) (R: ℝ) (angle_in_range : 0 < B ∧ B < 2 * Real.pi / 3) 
  (angle_eq : ∀ A, A = Real.pi / 3) : Prop :=
  a = 2 ∧ 
  (let m := (a, Real.sqrt 3 * b); n := (Real.cos A, Real.sin B) in
   ((a = 0 ∨ Real.sin B = 0) ∧ (Real.sqrt 3 * b = 0 ∨ Real.cos A = 0)) ∨
    (a * (Real.sin B) - (Real.sqrt 3) * b * (Real.cos A) = 0)) ∧ 
  2 * R = 4 * Real.sqrt 3 / 3 ∧ 
  ∀ (b c B : ℝ), b + c > 2 ∧ b + c ≤ 4

theorem problem_solution 
  (a b c B: ℝ) 
  (A : ℝ := Real.pi / 3) 
  (R: ℝ := 2 * a / Real.sin A)
  (angle_in_range : 0 < B ∧ B < 2 * Real.pi / 3) 
  (angle_eq : ∀ A, A = Real.pi / 3) 
  : 
  triangle_with_parallel_vectors 2 b c A B R angle_in_range angle_eq := 
  sorry

end problem_solution_l152_152585


namespace number_of_refuels_needed_l152_152387

noncomputable def fuelTankCapacity : ℕ := 50
noncomputable def distanceShanghaiHarbin : ℕ := 2560
noncomputable def fuelConsumptionRate : ℕ := 8
noncomputable def safetyFuel : ℕ := 6

theorem number_of_refuels_needed
  (fuelTankCapacity : ℕ)
  (distanceShanghaiHarbin : ℕ)
  (fuelConsumptionRate : ℕ)
  (safetyFuel : ℕ) :
  (fuelTankCapacity = 50) →
  (distanceShanghaiHarbin = 2560) →
  (fuelConsumptionRate = 8) →
  (safetyFuel = 6) →
  ∃ n : ℕ, n = 4 := by
  sorry

end number_of_refuels_needed_l152_152387


namespace exponential_fraction_l152_152508

theorem exponential_fraction :
  (2^2014 + 2^2012) / (2^2014 - 2^2012) = 5 / 3 := 
by
  sorry

end exponential_fraction_l152_152508


namespace acres_used_for_corn_l152_152244

noncomputable def total_acres : ℝ := 1634
noncomputable def beans_ratio : ℝ := 4.5
noncomputable def wheat_ratio : ℝ := 2.3
noncomputable def corn_ratio : ℝ := 3.8
noncomputable def barley_ratio : ℝ := 3.4

noncomputable def total_parts : ℝ := beans_ratio + wheat_ratio + corn_ratio + barley_ratio
noncomputable def acres_per_part : ℝ := total_acres / total_parts
noncomputable def corn_acres : ℝ := corn_ratio * acres_per_part

theorem acres_used_for_corn :
  corn_acres = 443.51 := by
  sorry

end acres_used_for_corn_l152_152244


namespace problem1_problem2_l152_152843

open Real

theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  sqrt a + sqrt b ≤ 2 :=
sorry

theorem problem2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (a + b^3) * (a^3 + b) ≥ 4 :=
sorry

end problem1_problem2_l152_152843


namespace slope_of_tangent_at_x1_l152_152848

noncomputable def f (x : ℝ) : ℝ := 2 - 1 / x

theorem slope_of_tangent_at_x1 :
  ∀ (f : ℝ → ℝ), (∀ x, f(x + 1) = (2 * x + 1) / (x + 1)) → 
  deriv f 1 = 1 :=
begin
  intros f h,
  have h_simp: ∀ (x : ℝ), f x = 2 - 1 / x := sorry,
  have h_deriv: ∀ (x : ℝ), deriv f x = 1 / x^2 := sorry,
  have h_eval: deriv f 1 = 1 := sorry,
  exact h_eval,
end

end slope_of_tangent_at_x1_l152_152848


namespace total_logs_in_both_stacks_l152_152376

-- Define the number of logs in the first stack
def first_stack_logs : Nat :=
  let bottom_row := 15
  let top_row := 4
  let number_of_terms := bottom_row - top_row + 1
  let average_logs := (bottom_row + top_row) / 2
  average_logs * number_of_terms

-- Define the number of logs in the second stack
def second_stack_logs : Nat :=
  let bottom_row := 5
  let top_row := 10
  let number_of_terms := top_row - bottom_row + 1
  let average_logs := (bottom_row + top_row) / 2
  average_logs * number_of_terms

-- Prove the total number of logs in both stacks
theorem total_logs_in_both_stacks : first_stack_logs + second_stack_logs = 159 := by
  sorry

end total_logs_in_both_stacks_l152_152376


namespace min_m_l152_152823

theorem min_m (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
    27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
by
  sorry

end min_m_l152_152823


namespace Linda_total_amount_at_21_years_l152_152733

theorem Linda_total_amount_at_21_years (P : ℝ) (r : ℝ) (n : ℕ) (initial_principal : P = 1500) (annual_rate : r = 0.03) (years : n = 21):
    P * (1 + r)^n = 2709.17 :=
by
  sorry

end Linda_total_amount_at_21_years_l152_152733


namespace boys_who_did_not_bring_laptops_l152_152736

-- Definitions based on the conditions.
def total_boys : ℕ := 20
def students_who_brought_laptops : ℕ := 25
def girls_who_brought_laptops : ℕ := 16

-- Main theorem statement.
theorem boys_who_did_not_bring_laptops : total_boys - (students_who_brought_laptops - girls_who_brought_laptops) = 11 := by
  sorry

end boys_who_did_not_bring_laptops_l152_152736


namespace hh_two_eq_902_l152_152728

def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem hh_two_eq_902 : h (h 2) = 902 := 
by
  sorry

end hh_two_eq_902_l152_152728


namespace wechat_payment_meaning_l152_152032

theorem wechat_payment_meaning (initial_balance after_receive_balance : ℝ)
  (recv_amount sent_amount : ℝ)
  (h1 : recv_amount = 200)
  (h2 : initial_balance + recv_amount = after_receive_balance)
  (h3 : after_receive_balance - sent_amount = initial_balance)
  : sent_amount = 200 :=
by
  -- starting the proof becomes irrelevant
  sorry

end wechat_payment_meaning_l152_152032


namespace binomial_expansion_terms_l152_152622

theorem binomial_expansion_terms (x y : ℝ) : 
  (x - y) ^ 10 = (11 : ℕ) :=
by
  sorry

end binomial_expansion_terms_l152_152622


namespace value_of_M_in_equation_l152_152052

theorem value_of_M_in_equation :
  ∀ {M : ℕ}, (32 = 2^5) ∧ (8 = 2^3) → (32^3 * 8^4 = 2^M) → M = 27 :=
by
  intros M h1 h2
  sorry

end value_of_M_in_equation_l152_152052


namespace sequence_non_positive_l152_152008

theorem sequence_non_positive
  (a : ℕ → ℝ) (n : ℕ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h : ∀ k, 1 ≤ k → k ≤ n - 1 → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) :
  ∀ k, k ≤ n → a k ≤ 0 := 
sorry

end sequence_non_positive_l152_152008
