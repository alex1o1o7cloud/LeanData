import Mathlib

namespace combined_future_value_l1228_122856

noncomputable def future_value (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem combined_future_value :
  let A1 := future_value 3000 0.05 3
  let A2 := future_value 5000 0.06 4
  let A3 := future_value 7000 0.07 5
  A1 + A2 + A3 = 19603.119 :=
by
  sorry

end combined_future_value_l1228_122856


namespace a_sub_b_eq_2_l1228_122808

theorem a_sub_b_eq_2 (a b : ℝ)
  (h : (a - 5) ^ 2 + |b ^ 3 - 27| = 0) : a - b = 2 :=
by
  sorry

end a_sub_b_eq_2_l1228_122808


namespace units_digit_of_k_squared_plus_2_to_k_l1228_122800

theorem units_digit_of_k_squared_plus_2_to_k (k : ℕ) (h : k = 2012 ^ 2 + 2 ^ 2014) : (k ^ 2 + 2 ^ k) % 10 = 5 := by
  sorry

end units_digit_of_k_squared_plus_2_to_k_l1228_122800


namespace imaginary_part_of_complex_division_l1228_122844

theorem imaginary_part_of_complex_division : 
  let i := Complex.I
  let z := (1 - 2 * i) / (2 - i)
  Complex.im z = -3 / 5 :=
by
  sorry

end imaginary_part_of_complex_division_l1228_122844


namespace find_rectangle_dimensions_area_30_no_rectangle_dimensions_area_32_l1228_122802

noncomputable def length_width_rectangle_area_30 : Prop :=
∃ (x y : ℝ), x * y = 30 ∧ 2 * (x + y) = 22 ∧ x = 6 ∧ y = 5

noncomputable def impossible_rectangle_area_32 : Prop :=
¬(∃ (x y : ℝ), x * y = 32 ∧ 2 * (x + y) = 22)

-- Proof statements (without proofs)
theorem find_rectangle_dimensions_area_30 : length_width_rectangle_area_30 :=
sorry

theorem no_rectangle_dimensions_area_32 : impossible_rectangle_area_32 :=
sorry

end find_rectangle_dimensions_area_30_no_rectangle_dimensions_area_32_l1228_122802


namespace system_of_equations_correct_l1228_122894

def weight_system (x y : ℝ) : Prop :=
  (5 * x + 6 * y = 1) ∧ (3 * x = y)

theorem system_of_equations_correct (x y : ℝ) :
  weight_system x y ↔ 
    (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 5 * x + 6 * y) :=
by sorry

end system_of_equations_correct_l1228_122894


namespace find_other_number_l1228_122862

theorem find_other_number (lcm_ab hcf_ab : ℕ) (A : ℕ) (h_lcm: Nat.lcm A (B) = lcm_ab)
  (h_hcf : Nat.gcd A (B) = hcf_ab) (h_a : A = 48) (h_lcm_value: lcm_ab = 192) (h_hcf_value: hcf_ab = 16) :
  B = 64 :=
by
  sorry

end find_other_number_l1228_122862


namespace triangles_from_pentadecagon_l1228_122809

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon
    is 455, given that there are 15 vertices and none of them are collinear. -/

theorem triangles_from_pentadecagon : (Nat.choose 15 3) = 455 := 
by
  sorry

end triangles_from_pentadecagon_l1228_122809


namespace union_set_eq_l1228_122821

open Set

def P := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
def Q := {x : ℝ | x^2 ≤ 4}

theorem union_set_eq : P ∪ Q = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by
  sorry

end union_set_eq_l1228_122821


namespace subset_a_eq_1_l1228_122845

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_a_eq_1 (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_a_eq_1_l1228_122845


namespace estate_problem_l1228_122873

def totalEstateValue (E a b : ℝ) : Prop :=
  (a + b = (3/5) * E) ∧ 
  (a = 2 * b) ∧ 
  (3 * b = (3/5) * E) ∧ 
  (E = a + b + (3 * b) + 4000)

theorem estate_problem (E : ℝ) (a b : ℝ) :
  totalEstateValue E a b → E = 20000 :=
by
  -- The proof will be filled here
  sorry

end estate_problem_l1228_122873


namespace poly_division_l1228_122826

noncomputable def A := 1
noncomputable def B := 3
noncomputable def C := 2
noncomputable def D := -1

theorem poly_division :
  (∀ x : ℝ, x ≠ -1 → (x^3 + 4*x^2 + 5*x + 2) / (x+1) = x^2 + 3*x + 2) ∧
  (A + B + C + D = 5) :=
by
  sorry

end poly_division_l1228_122826


namespace population_difference_is_16_l1228_122882

def total_birds : ℕ := 250

def pigeons_percent : ℕ := 30
def sparrows_percent : ℕ := 25
def crows_percent : ℕ := 20
def swans_percent : ℕ := 15
def parrots_percent : ℕ := 10

def black_pigeons_percent : ℕ := 60
def white_pigeons_percent : ℕ := 40
def black_male_pigeons_percent : ℕ := 20
def white_female_pigeons_percent : ℕ := 50

def female_sparrows_percent : ℕ := 60
def male_sparrows_percent : ℕ := 40

def female_crows_percent : ℕ := 30
def male_crows_percent : ℕ := 70

def male_parrots_percent : ℕ := 65
def female_parrots_percent : ℕ := 35

noncomputable
def black_male_pigeons : ℕ := (pigeons_percent * total_birds / 100) * (black_pigeons_percent * (black_male_pigeons_percent / 100)) / 100
noncomputable
def white_female_pigeons : ℕ := (pigeons_percent * total_birds / 100) * (white_pigeons_percent * (white_female_pigeons_percent / 100)) / 100
noncomputable
def male_sparrows : ℕ := (sparrows_percent * total_birds / 100) * (male_sparrows_percent / 100)
noncomputable
def female_crows : ℕ := (crows_percent * total_birds / 100) * (female_crows_percent / 100)
noncomputable
def male_parrots : ℕ := (parrots_percent * total_birds / 100) * (male_parrots_percent / 100)

noncomputable
def max_population : ℕ := max (max (max (max black_male_pigeons white_female_pigeons) male_sparrows) female_crows) male_parrots
noncomputable
def min_population : ℕ := min (min (min (min black_male_pigeons white_female_pigeons) male_sparrows) female_crows) male_parrots

noncomputable
def population_difference : ℕ := max_population - min_population

theorem population_difference_is_16 : population_difference = 16 :=
sorry

end population_difference_is_16_l1228_122882


namespace expression_equals_neg_eight_l1228_122822

variable {a b : ℝ}

theorem expression_equals_neg_eight (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : |a| ≠ |b|) :
  ( (b^2 / a^2 + a^2 / b^2 - 2) * 
    ((a + b) / (b - a) + (b - a) / (a + b)) * 
    (((1 / a^2 + 1 / b^2) / (1 / b^2 - 1 / a^2)) - ((1 / b^2 - 1 / a^2) / (1 / a^2 + 1 / b^2)))
  ) = -8 :=
by
  sorry

end expression_equals_neg_eight_l1228_122822


namespace eight_diamond_five_l1228_122865

def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem eight_diamond_five : diamond 8 5 = 160 :=
by sorry

end eight_diamond_five_l1228_122865


namespace expected_value_of_smallest_seven_selected_from_sixty_three_l1228_122898

noncomputable def expected_value_smallest_selected (n r : ℕ) : ℕ :=
  (n + 1) / (r + 1)

theorem expected_value_of_smallest_seven_selected_from_sixty_three :
  expected_value_smallest_selected 63 7 = 8 :=
by
  sorry -- Proof is omitted as per instructions

end expected_value_of_smallest_seven_selected_from_sixty_three_l1228_122898


namespace FastFoodCost_l1228_122837

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

end FastFoodCost_l1228_122837


namespace charlie_golden_delicious_bags_l1228_122835

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

end charlie_golden_delicious_bags_l1228_122835


namespace unique_triple_satisfying_conditions_l1228_122827

theorem unique_triple_satisfying_conditions :
  ∃! (x y z : ℝ), x + y = 4 ∧ xy - z^2 = 4 :=
sorry

end unique_triple_satisfying_conditions_l1228_122827


namespace car_gas_tank_capacity_l1228_122866

theorem car_gas_tank_capacity
  (initial_mileage : ℕ)
  (final_mileage : ℕ)
  (miles_per_gallon : ℕ)
  (tank_fills : ℕ)
  (usage : initial_mileage = 1728)
  (usage_final : final_mileage = 2928)
  (car_efficiency : miles_per_gallon = 30)
  (fills : tank_fills = 2):
  (final_mileage - initial_mileage) / miles_per_gallon / tank_fills = 20 :=
by
  sorry

end car_gas_tank_capacity_l1228_122866


namespace alicia_art_left_l1228_122888

-- Definition of the problem conditions.
def initial_pieces : ℕ := 70
def donated_pieces : ℕ := 46

-- The theorem to prove the number of art pieces left is 24.
theorem alicia_art_left : initial_pieces - donated_pieces = 24 := 
by
  sorry

end alicia_art_left_l1228_122888


namespace joe_lift_ratio_l1228_122830

theorem joe_lift_ratio (F S : ℕ) 
  (h1 : F + S = 1800) 
  (h2 : F = 700) 
  (h3 : 2 * F = S + 300) : F / S = 7 / 11 :=
by
  sorry

end joe_lift_ratio_l1228_122830


namespace T_10_mod_5_eq_3_l1228_122843

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

end T_10_mod_5_eq_3_l1228_122843


namespace probability_adjacent_difference_l1228_122838

noncomputable def probability_no_adjacent_same_rolls : ℚ :=
  (7 / 8) ^ 6

theorem probability_adjacent_difference :
  let num_people := 6
  let sides_of_die := 8
  ( ∀ i : ℕ, 0 ≤ i ∧ i < num_people -> (∃ x : ℕ, 1 ≤ x ∧ x ≤ sides_of_die)) →
  probability_no_adjacent_same_rolls = 117649 / 262144 := 
by 
  sorry

end probability_adjacent_difference_l1228_122838


namespace triangle_area_of_parabola_intersection_l1228_122847

theorem triangle_area_of_parabola_intersection
  (line_passes_through : ∃ (p : ℝ × ℝ), p = (0, -2))
  (parabola_intersection : ∃ (x1 y1 x2 y2 : ℝ),
    (x1, y1) ∈ {p : ℝ × ℝ | p.snd ^ 2 = 16 * p.fst} ∧
    (x2, y2) ∈ {p : ℝ × ℝ | p.snd ^ 2 = 16 * p.fst})
  (y_cond : ∃ (y1 y2 : ℝ), y1 ^ 2 - y2 ^ 2 = 1) :
  ∃ (area : ℝ), area = 1 / 16 :=
by
  sorry

end triangle_area_of_parabola_intersection_l1228_122847


namespace remaining_area_l1228_122814

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

end remaining_area_l1228_122814


namespace ratio_volumes_l1228_122842

noncomputable def V_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def V_cone (r : ℝ) : ℝ := (1 / 3) * Real.pi * r^3

theorem ratio_volumes (r : ℝ) (hr : r > 0) : 
  (V_cone r) / (V_sphere r) = 1 / 4 :=
by
  sorry

end ratio_volumes_l1228_122842


namespace slant_height_l1228_122819

-- Define the variables and conditions
variables (r A : ℝ)
-- Assume the given conditions
def radius := r = 5
def area := A = 60 * Real.pi

-- Statement of the theorem to prove the slant height
theorem slant_height (r A l : ℝ) (h_r : r = 5) (h_A : A = 60 * Real.pi) : l = 12 :=
sorry

end slant_height_l1228_122819


namespace net_gain_A_correct_l1228_122812

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

end net_gain_A_correct_l1228_122812


namespace multiple_of_first_number_is_eight_l1228_122801

theorem multiple_of_first_number_is_eight 
  (a b c k : ℤ)
  (h1 : a = 7) 
  (h2 : b = a + 2) 
  (h3 : c = b + 2) 
  (h4 : 7 * k = 3 * c + (2 * b + 5)) : 
  k = 8 :=
by
  sorry

end multiple_of_first_number_is_eight_l1228_122801


namespace relationship_y1_y2_y3_l1228_122881

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := -3 * x^2 + 2

-- Define the points and their coordinates
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-1, quadratic_function (-1)⟩
def B : Point := ⟨1, quadratic_function 1⟩
def C : Point := ⟨2, quadratic_function 2⟩

-- Prove the relationship between y1, y2, and y3
theorem relationship_y1_y2_y3 :
  A.y = B.y ∧ A.y > C.y :=
by
  sorry

end relationship_y1_y2_y3_l1228_122881


namespace aaron_earnings_l1228_122867

def monday_hours : ℚ := 7 / 4
def tuesday_hours : ℚ := 1 + 10 / 60
def wednesday_hours : ℚ := 3 + 15 / 60
def friday_hours : ℚ := 45 / 60

def total_hours_worked : ℚ := monday_hours + tuesday_hours + wednesday_hours + friday_hours
def hourly_rate : ℚ := 4

def total_earnings : ℚ := total_hours_worked * hourly_rate

theorem aaron_earnings : total_earnings = 27 := by
  sorry

end aaron_earnings_l1228_122867


namespace probability_vowel_probability_consonant_probability_ch_l1228_122857

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

end probability_vowel_probability_consonant_probability_ch_l1228_122857


namespace sequence_solution_l1228_122855

theorem sequence_solution (a : ℕ → ℝ)
  (h₁ : a 1 = 0)
  (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 4 * (Real.sqrt (a n + 1)) + 4) :
  ∀ n ≥ 1, a n = 4 * n^2 - 4 * n :=
by
  sorry

end sequence_solution_l1228_122855


namespace exists_coprime_positive_sum_le_m_l1228_122892

theorem exists_coprime_positive_sum_le_m (m : ℕ) (a b : ℤ) 
  (ha : 0 < a) (hb : 0 < b) (hcoprime : Int.gcd a b = 1)
  (h1 : a ∣ (m + b^2)) (h2 : b ∣ (m + a^2)) 
  : ∃ a' b', 0 < a' ∧ 0 < b' ∧ Int.gcd a' b' = 1 ∧ a' ∣ (m + b'^2) ∧ b' ∣ (m + a'^2) ∧ a' + b' ≤ m + 1 :=
by
  sorry

end exists_coprime_positive_sum_le_m_l1228_122892


namespace none_of_these_l1228_122810

-- Problem Statement:
theorem none_of_these (r x y : ℝ) (h1 : r > 0) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : x^2 + y^2 > x^2 * y^2) :
  ¬ (-x > -y) ∧ ¬ (-x > y) ∧ ¬ (1 > -y / x) ∧ ¬ (1 < x / y) :=
by
  sorry

end none_of_these_l1228_122810


namespace find_a_l1228_122834

def A : Set ℤ := {-1, 1, 3}
def B (a : ℤ) : Set ℤ := {a + 1, a^2 + 4}
def intersection (a : ℤ) : Set ℤ := A ∩ B a

theorem find_a : ∃ a : ℤ, intersection a = {3} ∧ a = 2 :=
by
  sorry

end find_a_l1228_122834


namespace functional_equation_solution_l1228_122864

theorem functional_equation_solution (f : ℕ+ → ℕ+) :
  (∀ n : ℕ+, f (f (f n)) + f (f n) + f n = 3 * n) →
  ∀ n : ℕ+, f n = n :=
by
  intro h
  sorry

end functional_equation_solution_l1228_122864


namespace matrix_B3_is_zero_unique_l1228_122805

theorem matrix_B3_is_zero_unique (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B^4 = 0) :
  ∃! (B3 : Matrix (Fin 2) (Fin 2) ℝ), B3 = B^3 ∧ B3 = 0 := sorry

end matrix_B3_is_zero_unique_l1228_122805


namespace p_arithmetic_fibonacci_term_correct_l1228_122859

noncomputable def p_arithmetic_fibonacci_term (p : ℕ) : ℝ :=
  5 ^ ((p - 1) / 2)

theorem p_arithmetic_fibonacci_term_correct (p : ℕ) : p_arithmetic_fibonacci_term p = 5 ^ ((p - 1) / 2) := 
by 
  rfl -- direct application of the definition

#check p_arithmetic_fibonacci_term_correct

end p_arithmetic_fibonacci_term_correct_l1228_122859


namespace fraction_unshaded_area_l1228_122874

theorem fraction_unshaded_area (s : ℝ) :
  let P := (s / 2, 0)
  let Q := (s, s / 2)
  let top_left := (0, s)
  let area_triangle : ℝ := 1 / 2 * (s / 2) * (s / 2)
  let area_square : ℝ := s * s
  let unshaded_area : ℝ := area_square - area_triangle
  let fraction_unshaded : ℝ := unshaded_area / area_square
  fraction_unshaded = 7 / 8 := 
by 
  sorry

end fraction_unshaded_area_l1228_122874


namespace complex_number_modulus_l1228_122803

open Complex

theorem complex_number_modulus :
  ∀ x : ℂ, x + I = (2 - I) / I → abs x = Real.sqrt 10 := by
  sorry

end complex_number_modulus_l1228_122803


namespace infinite_sqrt_eval_l1228_122854

theorem infinite_sqrt_eval {x : ℝ} (h : x = Real.sqrt (3 - x)) : 
  x = (-1 + Real.sqrt 13) / 2 :=
by sorry

end infinite_sqrt_eval_l1228_122854


namespace find_m_of_ellipse_conditions_l1228_122811

-- definition for isEllipseGivenFocus condition
def isEllipseGivenFocus (m : ℝ) : Prop :=
  ∃ (a : ℝ), a = 5 ∧ (-4)^2 = a^2 - m^2 ∧ 0 < m

-- statement to prove the described condition implies m = 3
theorem find_m_of_ellipse_conditions (m : ℝ) (h : isEllipseGivenFocus m) : m = 3 :=
sorry

end find_m_of_ellipse_conditions_l1228_122811


namespace min_value_inequality_l1228_122807

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 3^x + 9^y ≥ 2 * Real.sqrt 3 := 
by
  sorry

end min_value_inequality_l1228_122807


namespace undefined_expression_values_l1228_122823

theorem undefined_expression_values : 
    ∃ x : ℝ, x^2 - 9 = 0 ↔ (x = -3 ∨ x = 3) :=
by
  sorry

end undefined_expression_values_l1228_122823


namespace tomatoes_eaten_l1228_122861

theorem tomatoes_eaten 
  (initial_tomatoes : ℕ) 
  (final_tomatoes : ℕ) 
  (half_given : ℕ) 
  (B : ℕ) 
  (h_initial : initial_tomatoes = 127) 
  (h_final : final_tomatoes = 54) 
  (h_half : half_given = final_tomatoes * 2) 
  (h_remaining : initial_tomatoes - half_given = B)
  : B = 19 := 
by
  sorry

end tomatoes_eaten_l1228_122861


namespace solve_trigonometric_eqn_l1228_122806

theorem solve_trigonometric_eqn (x : ℝ) : 
  (∃ k : ℤ, x = 3 * (π / 4 * (4 * k + 1))) ∨ (∃ n : ℤ, x = π * (3 * n + 1) ∨ x = π * (3 * n - 1)) :=
by 
  sorry

end solve_trigonometric_eqn_l1228_122806


namespace stacy_faster_than_heather_l1228_122831

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

end stacy_faster_than_heather_l1228_122831


namespace prove_inequality_l1228_122880

noncomputable def inequality_problem :=
  ∀ (x y z : ℝ),
    0 < x ∧ 0 < y ∧ 0 < z ∧ x^2 + y^2 + z^2 = 3 → 
      (x ^ 2009 - 2008 * (x - 1)) / (y + z) + 
      (y ^ 2009 - 2008 * (y - 1)) / (x + z) + 
      (z ^ 2009 - 2008 * (z - 1)) / (x + y) ≥ 
      (x + y + z) / 2

theorem prove_inequality : inequality_problem := 
  by 
    sorry

end prove_inequality_l1228_122880


namespace side_length_of_square_l1228_122896

theorem side_length_of_square (d : ℝ) (s : ℝ) (h1 : d = 2 * Real.sqrt 2) (h2 : d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l1228_122896


namespace optimal_play_results_in_draw_l1228_122879

-- Define the concept of an optimal player, and a game state in Tic-Tac-Toe
structure Game :=
(board : Fin 3 × Fin 3 → Option Bool) -- Option Bool represents empty, O, or X
(turn : Bool) -- False for O's turn, True for X's turn

def draw (g : Game) : Bool :=
-- Implementation of checking for a draw will go here
sorry

noncomputable def optimal_move (g : Game) : Game :=
-- Implementation of finding the optimal move for the current player
sorry

theorem optimal_play_results_in_draw :
  ∀ (g : Game) (h : ∀ g, optimal_move g = g),
    draw (optimal_move g) = true :=
by
  -- The proof will be provided here
  sorry

end optimal_play_results_in_draw_l1228_122879


namespace gary_chickens_l1228_122851

theorem gary_chickens (initial_chickens : ℕ) (multiplication_factor : ℕ) 
  (weekly_eggs : ℕ) (days_in_week : ℕ)
  (h1 : initial_chickens = 4)
  (h2 : multiplication_factor = 8)
  (h3 : weekly_eggs = 1344)
  (h4 : days_in_week = 7) :
  (weekly_eggs / days_in_week) / (initial_chickens * multiplication_factor) = 6 :=
by
  sorry

end gary_chickens_l1228_122851


namespace bullet_trains_crossing_time_l1228_122817

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

end bullet_trains_crossing_time_l1228_122817


namespace clubs_popularity_order_l1228_122877

theorem clubs_popularity_order (chess drama art science : ℚ)
  (h_chess: chess = 14/35) (h_drama: drama = 9/28) (h_art: art = 11/21) (h_science: science = 8/15) :
  science > art ∧ art > chess ∧ chess > drama :=
by {
  -- Place proof steps here (optional)
  sorry
}

end clubs_popularity_order_l1228_122877


namespace work_completion_time_l1228_122839

theorem work_completion_time (A B C D : Type) 
  (work_rate_A : ℚ := 1 / 10) 
  (work_rate_AB : ℚ := 1 / 5)
  (work_rate_C : ℚ := 1 / 15) 
  (work_rate_D : ℚ := 1 / 20) 
  (combined_work_rate_AB : work_rate_A + (work_rate_AB - work_rate_A) = 1 / 10) : 
  (1 / (work_rate_A + (work_rate_AB - work_rate_A) + work_rate_C + work_rate_D)) = 60 / 19 := 
sorry

end work_completion_time_l1228_122839


namespace right_triangle_hypotenuse_l1228_122848

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h1 : a + b + c = 60) 
  (h2 : 0.5 * a * b = 120) 
  (h3 : a^2 + b^2 = c^2) : 
  c = 26 :=
by {
  sorry
}

end right_triangle_hypotenuse_l1228_122848


namespace total_number_of_people_l1228_122876

theorem total_number_of_people (L F LF N T : ℕ) (hL : L = 13) (hF : F = 15) (hLF : LF = 9) (hN : N = 6) : 
  T = (L + F - LF) + N → T = 25 :=
by
  intros h
  rw [hL, hF, hLF, hN] at h
  exact h

end total_number_of_people_l1228_122876


namespace directrix_of_parabola_l1228_122869

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = - (1 / 8) * x^2 → y = 2 :=
by
  sorry

end directrix_of_parabola_l1228_122869


namespace geometric_product_seven_terms_l1228_122849

theorem geometric_product_seven_terms (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = 1) 
  (h2 : a 6 + a 4 = 2 * (a 3 + a 1)) 
  (h_geometric : ∀ n, a (n + 1) = q * a n) :
  (a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7) = 128 := 
by 
  -- Steps involving algebraic manipulation and properties of geometric sequences should be here
  sorry

end geometric_product_seven_terms_l1228_122849


namespace chromium_percentage_in_new_alloy_l1228_122828

theorem chromium_percentage_in_new_alloy :
  ∀ (weight1 weight2 chromium1 chromium2: ℝ),
  weight1 = 15 → weight2 = 35 → chromium1 = 0.12 → chromium2 = 0.08 →
  (chromium1 * weight1 + chromium2 * weight2) / (weight1 + weight2) * 100 = 9.2 :=
by
  intros weight1 weight2 chromium1 chromium2 hweight1 hweight2 hchromium1 hchromium2
  sorry

end chromium_percentage_in_new_alloy_l1228_122828


namespace part_a_part_b_l1228_122850

-- Part (a)
theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (a b : ℝ), a ≠ b ∧ 0 < (a - b) / (1 + a * b) ∧ (a - b) / (1 + a * b) ≤ 1 := sorry

-- Part (b)
theorem part_b (x y z u : ℝ) :
  ∃ (a b : ℝ), a ≠ b ∧ 0 < (b - a) / (1 + a * b) ∧ (b - a) / (1 + a * b) ≤ 1 := sorry

end part_a_part_b_l1228_122850


namespace geometric_sequence_solution_l1228_122884

open Real

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n m, a (n + m) = a n * q ^ m

theorem geometric_sequence_solution :
  ∃ (a : ℕ → ℝ) (q : ℝ), geometric_sequence a q ∧
    (∀ n, 1 ≤ n ∧ n ≤ 5 → 10^8 ≤ a n ∧ a n < 10^9) ∧
    (∀ n, 6 ≤ n ∧ n ≤ 10 → 10^9 ≤ a n ∧ a n < 10^10) ∧
    (∀ n, 11 ≤ n ∧ n ≤ 14 → 10^10 ≤ a n ∧ a n < 10^11) ∧
    (∀ n, 15 ≤ n ∧ n ≤ 16 → 10^11 ≤ a n ∧ a n < 10^12) ∧
    (∀ i, a i = 7 * 3^(16-i) * 5^(i-1)) := sorry

end geometric_sequence_solution_l1228_122884


namespace solve_expression_l1228_122895

theorem solve_expression (a b c : ℝ) (ha : a^3 - 2020*a^2 + 1010 = 0) (hb : b^3 - 2020*b^2 + 1010 = 0) (hc : c^3 - 2020*c^2 + 1010 = 0) (habc_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
    (1 / (a * b) + 1 / (b * c) + 1 / (a * c) = -2) := 
sorry

end solve_expression_l1228_122895


namespace correct_operation_result_l1228_122893

-- Define the conditions
def original_number : ℤ := 231
def incorrect_result : ℤ := 13

-- Define the two incorrect operations and the intended corrections
def reverse_subtract : ℤ := incorrect_result + 20
def reverse_division : ℤ := reverse_subtract * 7

-- Define the intended operations
def intended_multiplication : ℤ := original_number * 7
def intended_addition : ℤ := intended_multiplication + 20

-- The theorem we need to prove
theorem correct_operation_result :
  original_number = reverse_division →
  intended_addition > 1100 :=
by
  intros h
  sorry

end correct_operation_result_l1228_122893


namespace ThreePowFifteenModFive_l1228_122868

def rem_div_3_pow_15_by_5 : ℕ :=
  let base := 3
  let mod := 5
  let exp := 15
  
  base^exp % mod

theorem ThreePowFifteenModFive (h1: 3^4 ≡ 1 [MOD 5]) : rem_div_3_pow_15_by_5 = 2 := by
  sorry

end ThreePowFifteenModFive_l1228_122868


namespace exponent_property_l1228_122840

theorem exponent_property (a x y : ℝ) (h1 : 0 < a) (h2 : a ^ x = 2) (h3 : a ^ y = 3) : a ^ (x - y) = 2 / 3 := 
by
  sorry

end exponent_property_l1228_122840


namespace proportion_of_triumphal_arch_photographs_l1228_122875

-- Define the constants
variables (x y z t : ℕ) -- x = castles, y = triumphal arches, z = waterfalls, t = cathedrals

-- The conditions
axiom half_photographed : t + x + y + z = (3*y + 2*x + 2*z + y) / 2
axiom three_times_cathedrals : ∃ (a : ℕ), t = 3 * a ∧ y = a
axiom same_castles_waterfalls : ∃ (b : ℕ), t + z = x + y
axiom quarter_photographs_castles : x = (t + x + y + z) / 4
axiom second_castle_frequency : t + z = 2 * x
axiom every_triumphal_arch_photographed : ∀ (c : ℕ), y = c ∧ y = c

theorem proportion_of_triumphal_arch_photographs : 
  ∃ (p : ℚ), p = 1 / 4 ∧ p = y / ((t + x + y + z) / 2) :=
sorry

end proportion_of_triumphal_arch_photographs_l1228_122875


namespace ratio_of_second_to_first_l1228_122863

theorem ratio_of_second_to_first:
  ∀ (x y z : ℕ), 
  (y = 90) → 
  (z = 4 * y) → 
  ((x + y + z) / 3 = 165) → 
  (y / x = 2) := 
by 
  intros x y z h1 h2 h3
  sorry

end ratio_of_second_to_first_l1228_122863


namespace length_AB_l1228_122813

theorem length_AB :
  ∀ (A B : ℝ × ℝ) (k : ℝ),
    (A.2 = k * A.1 - 2) ∧ (B.2 = k * B.1 - 2) ∧ (A.2^2 = 8 * A.1) ∧ (B.2^2 = 8 * B.1) ∧
    ((A.1 + B.1) / 2 = 2) →
  dist A B = 2 * Real.sqrt 15 :=
by
  sorry

end length_AB_l1228_122813


namespace m_is_perfect_square_l1228_122883

theorem m_is_perfect_square
  (m n k : ℕ) 
  (h1 : 0 < m) 
  (h2 : 0 < n) 
  (h3 : 0 < k) 
  (h4 : 1 + m + n * Real.sqrt 3 = (2 + Real.sqrt 3) ^ (2 * k + 1)) : 
  ∃ a : ℕ, m = a ^ 2 :=
by 
  sorry

end m_is_perfect_square_l1228_122883


namespace smallest_m_for_integral_solutions_l1228_122886

theorem smallest_m_for_integral_solutions (p q : ℤ) (h : p * q = 42) (h0 : p + q = m / 15) : 
  0 < m ∧ 15 * p * p - m * p + 630 = 0 ∧ 15 * q * q - m * q + 630 = 0 →
  m = 195 :=
by 
  sorry

end smallest_m_for_integral_solutions_l1228_122886


namespace simplify_complex_l1228_122872

open Complex

theorem simplify_complex : (5 : ℂ) / (I - 2) = -2 - I := by
  sorry

end simplify_complex_l1228_122872


namespace solution_pairs_correct_l1228_122846

theorem solution_pairs_correct:
  { (n, m) : ℕ × ℕ | m^2 + 2 * 3^n = m * (2^(n+1) - 1) }
  = {(3, 6), (3, 9), (6, 54), (6, 27)} :=
by
  sorry -- no proof is required as per the instruction

end solution_pairs_correct_l1228_122846


namespace contradiction_assumption_l1228_122860

theorem contradiction_assumption (a b : ℝ) (h : |a - 1| * |b - 1| = 0) : ¬ (a ≠ 1 ∧ b ≠ 1) :=
  sorry

end contradiction_assumption_l1228_122860


namespace not_equal_d_l1228_122890

def frac_14_over_6 : ℚ := 14 / 6
def mixed_2_and_1_3rd : ℚ := 2 + 1 / 3
def mixed_neg_2_and_1_3rd : ℚ := -(2 + 1 / 3)
def mixed_3_and_1_9th : ℚ := 3 + 1 / 9
def mixed_2_and_4_12ths : ℚ := 2 + 4 / 12
def target_fraction : ℚ := 7 / 3

theorem not_equal_d : mixed_3_and_1_9th ≠ target_fraction :=
by sorry

end not_equal_d_l1228_122890


namespace commercials_played_l1228_122825

theorem commercials_played (M C : ℝ) (h1 : M / C = 9 / 5) (h2 : M + C = 112) : C = 40 :=
by
  sorry

end commercials_played_l1228_122825


namespace volume_of_fifth_section_l1228_122832

theorem volume_of_fifth_section (a : ℕ → ℚ) (d : ℚ) :
  (a 1 + a 2 + a 3 + a 4) = 3 ∧ (a 9 + a 8 + a 7) = 4 ∧
  (∀ n, a n = a 1 + (n - 1) * d) →
  a 5 = 67 / 66 :=
by
  sorry

end volume_of_fifth_section_l1228_122832


namespace find_sp_l1228_122858

theorem find_sp (s p : ℝ) (t x y : ℝ) (h1 : x = 3 + 5 * t) (h2 : y = 3 + p * t) 
  (h3 : y = 4 * x - 9) : 
  s = 3 ∧ p = 20 := 
by
  -- Proof goes here
  sorry

end find_sp_l1228_122858


namespace center_of_circle_l1228_122815

theorem center_of_circle : ∀ (x y : ℝ), x^2 + y^2 = 4 * x - 6 * y + 9 → (x, y) = (2, -3) :=
by
sorry

end center_of_circle_l1228_122815


namespace age_condition_l1228_122804

theorem age_condition (x y z : ℕ) (h1 : x > y) : 
  (z > y) ↔ (y + z > 2 * x) ∧ (∀ x y z, y + z > 2 * x → z > y) := sorry

end age_condition_l1228_122804


namespace batsman_average_increase_l1228_122818

theorem batsman_average_increase (A : ℕ) 
    (h1 : 15 * A + 64 = 19 * 16) 
    (h2 : 19 - A = 3) : 
    19 - A = 3 := 
sorry

end batsman_average_increase_l1228_122818


namespace minimum_value_f_b_eq_neg_a_maximum_value_ab_inequality_for_f_and_f_l1228_122899

noncomputable def f (a b x : ℝ) := Real.exp x - a * x - b

theorem minimum_value_f_b_eq_neg_a (a : ℝ) (h : 0 < a) :
  ∃ m, m = 2 * a - a * Real.log a ∧ ∀ x : ℝ, f a (-a) x ≥ m :=
sorry

theorem maximum_value_ab (a b : ℝ) (h : ∀ x : ℝ, f a b x + a ≥ 0) :
  ab ≤ (1 / 2) * Real.exp 3 :=
sorry

theorem inequality_for_f_and_f' (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : b = -a) (h3 : f a b x1 = 0) (h4 : f a b x2 = 0) (h5 : x1 < x2)
  : f a (-a) (3 * Real.log a) > (Real.exp ((2 * x1 * x2) / (x1 + x2)) - a) :=
sorry

end minimum_value_f_b_eq_neg_a_maximum_value_ab_inequality_for_f_and_f_l1228_122899


namespace ratio_of_x_intercepts_l1228_122853

theorem ratio_of_x_intercepts (b : ℝ) (hb: b ≠ 0) (u v: ℝ) (h₁: 8 * u + b = 0) (h₂: 4 * v + b = 0) : 
  u / v = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l1228_122853


namespace mul_three_point_six_and_zero_point_twenty_five_l1228_122833

theorem mul_three_point_six_and_zero_point_twenty_five : 3.6 * 0.25 = 0.9 := by 
  sorry

end mul_three_point_six_and_zero_point_twenty_five_l1228_122833


namespace find_angle_MBA_l1228_122852

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

end find_angle_MBA_l1228_122852


namespace cricket_bat_cost_l1228_122885

noncomputable def CP_A_sol : ℝ := 444.96 / 1.95

theorem cricket_bat_cost (CP_A : ℝ) (SP_B : ℝ) (SP_C : ℝ) (SP_D : ℝ) :
  (SP_B = 1.20 * CP_A) →
  (SP_C = 1.25 * SP_B) →
  (SP_D = 1.30 * SP_C) →
  (SP_D = 444.96) →
  CP_A = CP_A_sol :=
by
  intros h1 h2 h3 h4
  sorry

end cricket_bat_cost_l1228_122885


namespace sally_took_out_5_onions_l1228_122841

theorem sally_took_out_5_onions (X Y : ℕ) 
    (h1 : 4 + 9 - Y + X = X + 8) : Y = 5 := 
by
  sorry

end sally_took_out_5_onions_l1228_122841


namespace find_2theta_plus_phi_l1228_122878

variable (θ φ : ℝ)
variable (hθ : 0 < θ ∧ θ < π / 2)
variable (hφ : 0 < φ ∧ φ < π / 2)
variable (tan_hθ : Real.tan θ = 2 / 5)
variable (cos_hφ : Real.cos φ = 1 / 2)

theorem find_2theta_plus_phi : 2 * θ + φ = π / 4 := by
  sorry

end find_2theta_plus_phi_l1228_122878


namespace find_integer_n_l1228_122870

theorem find_integer_n (n : ℤ) : (⌊(n^2 : ℤ) / 4⌋ - ⌊n / 2⌋ ^ 2 = 3) → n = 7 :=
by sorry

end find_integer_n_l1228_122870


namespace price_after_two_reductions_l1228_122891

-- Define the two reductions as given in the conditions
def first_day_reduction (P : ℝ) : ℝ := P * 0.88
def second_day_reduction (P : ℝ) : ℝ := first_day_reduction P * 0.9

-- Main theorem: Price on the second day is 79.2% of the original price
theorem price_after_two_reductions (P : ℝ) : second_day_reduction P = 0.792 * P :=
by
  sorry

end price_after_two_reductions_l1228_122891


namespace find_N_l1228_122897

theorem find_N (N : ℤ) :
  (10 + 11 + 12) / 3 = (2010 + 2011 + 2012 + N) / 4 → N = -5989 :=
by
  sorry

end find_N_l1228_122897


namespace logan_passengers_count_l1228_122887

noncomputable def passengers_used_Kennedy_Airport : ℝ := (1 / 3) * 38.3
noncomputable def passengers_used_Miami_Airport : ℝ := (1 / 2) * passengers_used_Kennedy_Airport
noncomputable def passengers_used_Logan_Airport : ℝ := passengers_used_Miami_Airport / 4

theorem logan_passengers_count : abs (passengers_used_Logan_Airport - 1.6) < 0.01 := by
  sorry

end logan_passengers_count_l1228_122887


namespace trisha_collects_4_dozen_less_l1228_122871

theorem trisha_collects_4_dozen_less (B C T : ℕ) 
  (h1 : B = 6) 
  (h2 : C = 3 * B) 
  (h3 : B + C + T = 26) : 
  B - T = 4 := 
by 
  sorry

end trisha_collects_4_dozen_less_l1228_122871


namespace complement_union_eq_complement_l1228_122829

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

end complement_union_eq_complement_l1228_122829


namespace one_and_two_thirds_eq_36_l1228_122824

theorem one_and_two_thirds_eq_36 (x : ℝ) (h : (5 / 3) * x = 36) : x = 21.6 :=
sorry

end one_and_two_thirds_eq_36_l1228_122824


namespace correct_equation_among_options_l1228_122820

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

end correct_equation_among_options_l1228_122820


namespace joelle_initial_deposit_l1228_122836

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

end joelle_initial_deposit_l1228_122836


namespace find_values_and_properties_l1228_122889

variable (f : ℝ → ℝ)

axiom f_neg1 : f (-1) = 2
axiom f_pos_x : ∀ x, x < 0 → f x > 1
axiom f_add : ∀ x y : ℝ, f (x + y) = f x * f y

theorem find_values_and_properties :
  f 0 = 1 ∧
  f (-4) = 16 ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, f (-4 * x^2) * f (10 * x) ≥ 1/16 ↔ x ≤ 1/2 ∨ x ≥ 2) :=
sorry

end find_values_and_properties_l1228_122889


namespace rhombus_perimeter_52_l1228_122816

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

end rhombus_perimeter_52_l1228_122816
