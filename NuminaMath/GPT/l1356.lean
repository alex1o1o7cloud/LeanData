import Mathlib

namespace NUMINAMATH_GPT_min_value_of_abc_l1356_135640

variables {a b c : ℝ}

noncomputable def satisfies_condition (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ (b + c) / a + (a + c) / b = (a + b) / c + 1

theorem min_value_of_abc (a b c : ℝ) (h : satisfies_condition a b c) : (a + b) / c ≥ 5 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_abc_l1356_135640


namespace NUMINAMATH_GPT_minimum_energy_H1_l1356_135651

-- Define the given conditions
def energyEfficiencyMin : ℝ := 0.1
def energyRequiredH6 : ℝ := 10 -- Energy in KJ
def energyLevels : Nat := 5 -- Number of energy levels from H1 to H6

-- Define the theorem to prove the minimum energy required from H1
theorem minimum_energy_H1 : (10 ^ energyLevels : ℝ) = 1000000 :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_minimum_energy_H1_l1356_135651


namespace NUMINAMATH_GPT_range_of_a_l1356_135684

theorem range_of_a (a : ℝ) : (∃ x : ℝ, a * x = 1) ↔ a ≠ 0 := by
sorry

end NUMINAMATH_GPT_range_of_a_l1356_135684


namespace NUMINAMATH_GPT_cost_of_candy_bar_l1356_135647

theorem cost_of_candy_bar (t c b : ℕ) (h1 : t = 13) (h2 : c = 6) (h3 : t = b + c) : b = 7 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_candy_bar_l1356_135647


namespace NUMINAMATH_GPT_mean_temperature_correct_l1356_135663

def temperatures : List ℤ := [-6, -3, -3, -4, 2, 4, 1]

def mean_temperature (temps : List ℤ) : ℚ :=
  (temps.sum : ℚ) / temps.length

theorem mean_temperature_correct :
  mean_temperature temperatures = -9 / 7 := 
by
  sorry

end NUMINAMATH_GPT_mean_temperature_correct_l1356_135663


namespace NUMINAMATH_GPT_rectangle_triangle_height_l1356_135612

theorem rectangle_triangle_height (l : ℝ) (h : ℝ) (w : ℝ) (d : ℝ) 
  (hw : w = Real.sqrt 2 * l)
  (hd : d = Real.sqrt (l^2 + w^2))
  (A_triangle : (1 / 2) * d * h = l * w) :
  h = (2 * l * Real.sqrt 6) / 3 := by
  sorry

end NUMINAMATH_GPT_rectangle_triangle_height_l1356_135612


namespace NUMINAMATH_GPT_time_spent_on_each_piece_l1356_135610

def chairs : Nat := 7
def tables : Nat := 3
def total_time : Nat := 40
def total_pieces := chairs + tables
def time_per_piece := total_time / total_pieces

theorem time_spent_on_each_piece : time_per_piece = 4 :=
by
  sorry

end NUMINAMATH_GPT_time_spent_on_each_piece_l1356_135610


namespace NUMINAMATH_GPT_trigonometric_identity_l1356_135652

theorem trigonometric_identity (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * (Real.cos (π / 6 + α / 2))^2 - 1 = 1 / 3 := 
by sorry

end NUMINAMATH_GPT_trigonometric_identity_l1356_135652


namespace NUMINAMATH_GPT_trapezoid_perimeter_area_sum_l1356_135622

noncomputable def distance (p1 p2 : Real × Real) : Real :=
  ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2).sqrt

noncomputable def perimeter (vertices : List (Real × Real)) : Real :=
  match vertices with
  | [a, b, c, d] => (distance a b) + (distance b c) + (distance c d) + (distance d a)
  | _ => 0

noncomputable def area_trapezoid (b1 b2 h : Real) : Real :=
  0.5 * (b1 + b2) * h

theorem trapezoid_perimeter_area_sum
  (A B C D : Real × Real)
  (h_AB : A = (2, 3))
  (h_BC : B = (7, 3))
  (h_CD : C = (9, 7))
  (h_DA : D = (0, 7)) :
  let perimeter := perimeter [A, B, C, D]
  let area := area_trapezoid (distance C D) (distance A B) (C.2 - B.2)
  perimeter + area = 42 + 4 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_perimeter_area_sum_l1356_135622


namespace NUMINAMATH_GPT_grover_total_profit_l1356_135634

-- Definitions based on conditions
def original_price : ℝ := 10
def discount_first_box : ℝ := 0.20
def discount_second_box : ℝ := 0.30
def discount_third_box : ℝ := 0.40
def packs_first_box : ℕ := 20
def packs_second_box : ℕ := 30
def packs_third_box : ℕ := 40
def masks_per_pack : ℕ := 5
def price_per_mask_first_box : ℝ := 0.75
def price_per_mask_second_box : ℝ := 0.85
def price_per_mask_third_box : ℝ := 0.95

-- Computations
def cost_first_box := original_price - (discount_first_box * original_price)
def cost_second_box := original_price - (discount_second_box * original_price)
def cost_third_box := original_price - (discount_third_box * original_price)

def total_cost := cost_first_box + cost_second_box + cost_third_box

def revenue_first_box := packs_first_box * masks_per_pack * price_per_mask_first_box
def revenue_second_box := packs_second_box * masks_per_pack * price_per_mask_second_box
def revenue_third_box := packs_third_box * masks_per_pack * price_per_mask_third_box

def total_revenue := revenue_first_box + revenue_second_box + revenue_third_box

def total_profit := total_revenue - total_cost

-- Proof statement
theorem grover_total_profit : total_profit = 371.5 := by
  sorry

end NUMINAMATH_GPT_grover_total_profit_l1356_135634


namespace NUMINAMATH_GPT_denny_followers_l1356_135699

theorem denny_followers (initial_followers: ℕ) (new_followers_per_day: ℕ) (unfollowers_in_year: ℕ) (days_in_year: ℕ)
  (h_initial: initial_followers = 100000)
  (h_new_per_day: new_followers_per_day = 1000)
  (h_unfollowers: unfollowers_in_year = 20000)
  (h_days: days_in_year = 365):
  initial_followers + (new_followers_per_day * days_in_year) - unfollowers_in_year = 445000 :=
by
  sorry

end NUMINAMATH_GPT_denny_followers_l1356_135699


namespace NUMINAMATH_GPT_area_difference_l1356_135687

theorem area_difference (T_area : ℝ) (omega_area : ℝ) (H1 : T_area = (25 * Real.sqrt 3) / 4) 
  (H2 : omega_area = 4 * Real.pi) (H3 : 3 * (X - Y) = T_area - omega_area) :
  X - Y = (25 * Real.sqrt 3) / 12 - (4 * Real.pi) / 3 :=
by 
  sorry

end NUMINAMATH_GPT_area_difference_l1356_135687


namespace NUMINAMATH_GPT_ria_number_is_2_l1356_135698

theorem ria_number_is_2 
  (R S : ℕ) 
  (consecutive : R = S + 1 ∨ S = R + 1) 
  (R_positive : R > 0) 
  (S_positive : S > 0) 
  (R_not_1 : R ≠ 1) 
  (Sylvie_does_not_know : S ≠ 1) 
  (Ria_knows_after_Sylvie : ∃ (R_known : ℕ), R_known = R) :
  R = 2 :=
sorry

end NUMINAMATH_GPT_ria_number_is_2_l1356_135698


namespace NUMINAMATH_GPT_final_building_height_l1356_135681

noncomputable def height_of_final_building 
    (Crane1_height : ℝ)
    (Building1_height : ℝ)
    (Crane2_height : ℝ)
    (Building2_height : ℝ)
    (Crane3_height : ℝ)
    (Average_difference : ℝ) : ℝ :=
    Crane3_height / (1 + Average_difference)

theorem final_building_height
    (Crane1_height : ℝ := 228)
    (Building1_height : ℝ := 200)
    (Crane2_height : ℝ := 120)
    (Building2_height : ℝ := 100)
    (Crane3_height : ℝ := 147)
    (Average_difference : ℝ := 0.13)
    (HCrane1 : 1 + (Crane1_height - Building1_height) / Building1_height = 1.14)
    (HCrane2 : 1 + (Crane2_height - Building2_height) / Building2_height = 1.20)
    (HAvg : (1.14 + 1.20) / 2 = 1.13) :
    height_of_final_building Crane1_height Building1_height Crane2_height Building2_height Crane3_height Average_difference = 130 := 
sorry

end NUMINAMATH_GPT_final_building_height_l1356_135681


namespace NUMINAMATH_GPT_soldiers_movement_l1356_135636

theorem soldiers_movement (n : ℕ) 
  (initial_positions : Fin (n+3) × Fin (n+1) → Prop) 
  (moves_to_adjacent : ∀ p : Fin (n+3) × Fin (n+1), initial_positions p → initial_positions (p.1 + 1, p.2) ∨ initial_positions (p.1 - 1, p.2) ∨ initial_positions (p.1, p.2 + 1) ∨ initial_positions (p.1, p.2 - 1))
  (final_positions : Fin (n+1) × Fin (n+3) → Prop) : Even n := 
sorry

end NUMINAMATH_GPT_soldiers_movement_l1356_135636


namespace NUMINAMATH_GPT_arithmetic_sequence_8th_term_l1356_135649

theorem arithmetic_sequence_8th_term (a d : ℤ) :
  (a + d = 25) ∧ (a + 5 * d = 49) → (a + 7 * d = 61) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_8th_term_l1356_135649


namespace NUMINAMATH_GPT_part1_part2_l1356_135632

-- Step 1: Define the problem for a triangle with specific side length conditions and perimeter
theorem part1 (x : ℝ) (h1 : 2 * x + 2 * (2 * x) = 18) : 
  x = 18 / 5 ∧ 2 * x = 36 / 5 :=
by
  sorry

-- Step 2: Verify if an isosceles triangle with a side length of 4 cm can be formed
theorem part2 (a b c : ℝ) (h2 : a = 4 ∨ b = 4 ∨ c = 4) (h3 : a + b + c = 18) : 
  (a = 4 ∧ b = 7 ∧ c = 7 ∨ b = 4 ∧ a = 7 ∧ c = 7 ∨ c = 4 ∧ a = 7 ∧ b = 7) ∨
  (¬(a = 4 ∧ b + c <= a ∨ b = 4 ∧ a + c <= b ∨ c = 4 ∧ a + b <= c)) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1356_135632


namespace NUMINAMATH_GPT_range_of_a_l1356_135677

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (3 / 2)^x = (2 + 3 * a) / (5 - a)) ↔ a ∈ Set.Ioo (-2 / 3) (3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1356_135677


namespace NUMINAMATH_GPT_chosen_number_is_155_l1356_135619

variable (x : ℤ)
variable (h₁ : 2 * x - 200 = 110)

theorem chosen_number_is_155 : x = 155 := by
  sorry

end NUMINAMATH_GPT_chosen_number_is_155_l1356_135619


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1356_135694

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 2 * x = 1) : x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1356_135694


namespace NUMINAMATH_GPT_problem_statement_l1356_135643

theorem problem_statement (x : ℝ) (hx : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1356_135643


namespace NUMINAMATH_GPT_total_cookies_is_390_l1356_135618

def abigail_boxes : ℕ := 2
def grayson_boxes : ℚ := 3 / 4
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48

def abigail_cookies : ℕ := abigail_boxes * cookies_per_box
def grayson_cookies : ℚ := grayson_boxes * cookies_per_box
def olivia_cookies : ℕ := olivia_boxes * cookies_per_box
def isabella_cookies : ℚ := (1 / 2) * grayson_cookies
def ethan_cookies : ℤ := (abigail_boxes * 2 * cookies_per_box) / 2

def total_cookies : ℚ := ↑abigail_cookies + grayson_cookies + ↑olivia_cookies + isabella_cookies + ↑ethan_cookies

theorem total_cookies_is_390 : total_cookies = 390 :=
by
  sorry

end NUMINAMATH_GPT_total_cookies_is_390_l1356_135618


namespace NUMINAMATH_GPT_matrix_A_pow_100_eq_l1356_135670

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![4, 1], ![-9, -2]]

theorem matrix_A_pow_100_eq : matrix_A ^ 100 = ![![301, 100], ![-900, -299]] :=
  sorry

end NUMINAMATH_GPT_matrix_A_pow_100_eq_l1356_135670


namespace NUMINAMATH_GPT_total_cost_of_books_and_pencils_l1356_135689

variable (a b : ℕ)

theorem total_cost_of_books_and_pencils (a b : ℕ) : 5 * a + 2 * b = 5 * a + 2 * b := by
  sorry

end NUMINAMATH_GPT_total_cost_of_books_and_pencils_l1356_135689


namespace NUMINAMATH_GPT_sum_of_k_values_l1356_135635

theorem sum_of_k_values 
  (h : ∀ (k : ℤ), (∀ x y : ℤ, x * y = 15 → x + y = k) → k > 0 → false) : 
  ∃ k_values : List ℤ, 
  (∀ (k : ℤ), k ∈ k_values → (∀ x y : ℤ, x * y = 15 → x + y = k) ∧ k > 0) ∧ 
  k_values.sum = 24 := sorry

end NUMINAMATH_GPT_sum_of_k_values_l1356_135635


namespace NUMINAMATH_GPT_problem1_problem2_l1356_135625

-- Problem (I)
theorem problem1 (α : ℝ) (h1 : Real.tan α = 3) :
  (4 * Real.sin (Real.pi - α) - 2 * Real.cos (-α)) / (3 * Real.cos (Real.pi / 2 - α) - 5 * Real.cos (Real.pi + α)) = 5 / 7 := by
sorry

-- Problem (II)
theorem problem2 (x : ℝ) (h2 : Real.sin x + Real.cos x = 1 / 5) (h3 : 0 < x ∧ x < Real.pi) :
  Real.sin x = 4 / 5 ∧ Real.cos x = -3 / 5 := by
sorry

end NUMINAMATH_GPT_problem1_problem2_l1356_135625


namespace NUMINAMATH_GPT_number_of_tables_l1356_135662

-- Define the total number of customers the waiter is serving
def total_customers := 90

-- Define the number of women per table
def women_per_table := 7

-- Define the number of men per table
def men_per_table := 3

-- Define the total number of people per table
def people_per_table : ℕ := women_per_table + men_per_table

-- Statement to prove the number of tables
theorem number_of_tables (T : ℕ) (h : T * people_per_table = total_customers) : T = 9 := by
  sorry

end NUMINAMATH_GPT_number_of_tables_l1356_135662


namespace NUMINAMATH_GPT_students_catching_up_on_homework_l1356_135633

theorem students_catching_up_on_homework :
  ∀ (total_students : ℕ) (half : ℕ) (third : ℕ),
  total_students = 24 → half = total_students / 2 → third = total_students / 3 →
  total_students - (half + third) = 4 :=
by
  intros total_students half third
  intros h_total h_half h_third
  sorry

end NUMINAMATH_GPT_students_catching_up_on_homework_l1356_135633


namespace NUMINAMATH_GPT_find_solutions_l1356_135648

theorem find_solutions (x y z : ℝ) :
    (x^2 + y^2 - z * (x + y) = 2 ∧ y^2 + z^2 - x * (y + z) = 4 ∧ z^2 + x^2 - y * (z + x) = 8) ↔
    (x = 1 ∧ y = -1 ∧ z = 2) ∨ (x = -1 ∧ y = 1 ∧ z = -2) := sorry

end NUMINAMATH_GPT_find_solutions_l1356_135648


namespace NUMINAMATH_GPT_night_crew_fraction_l1356_135638

theorem night_crew_fraction (D N : ℝ) (B : ℝ) 
  (h1 : ∀ d, d = D → ∀ n, n = N → ∀ b, b = B → (n * (3/4) * b) = (3/4) * (d * b) / 3)
  (h2 : ∀ t, t = (D * B + (N * (3/4) * B)) → (D * B) / t = 2 / 3) :
  N / D = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_night_crew_fraction_l1356_135638


namespace NUMINAMATH_GPT_compound_oxygen_atoms_l1356_135695

theorem compound_oxygen_atoms 
  (C_atoms : ℕ)
  (H_atoms : ℕ)
  (total_molecular_weight : ℝ)
  (atomic_weight_C : ℝ)
  (atomic_weight_H : ℝ)
  (atomic_weight_O : ℝ) :
  C_atoms = 4 →
  H_atoms = 8 →
  total_molecular_weight = 88 →
  atomic_weight_C = 12.01 →
  atomic_weight_H = 1.008 →
  atomic_weight_O = 16.00 →
  (total_molecular_weight - (C_atoms * atomic_weight_C + H_atoms * atomic_weight_H)) / atomic_weight_O = 2 := 
by 
  intros;
  sorry

end NUMINAMATH_GPT_compound_oxygen_atoms_l1356_135695


namespace NUMINAMATH_GPT_ratio_of_cream_l1356_135656

def initial_coffee := 12
def joe_drank := 2
def cream_added := 2
def joann_cream_added := 2
def joann_drank := 2

noncomputable def joe_coffee_after_drink_add := initial_coffee - joe_drank + cream_added
noncomputable def joe_cream := cream_added

noncomputable def joann_initial_mixture := initial_coffee + joann_cream_added
noncomputable def joann_portion_before_drink := joann_cream_added / joann_initial_mixture
noncomputable def joann_remaining_coffee := joann_initial_mixture - joann_drank
noncomputable def joann_cream_after_drink := joann_portion_before_drink * joann_remaining_coffee
noncomputable def joann_cream := joann_cream_after_drink

theorem ratio_of_cream : joe_cream / joann_cream = 7 / 6 :=
by sorry

end NUMINAMATH_GPT_ratio_of_cream_l1356_135656


namespace NUMINAMATH_GPT_triangle_area_l1356_135674

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * sqrt 3 * cos x^2 - sqrt 3

theorem triangle_area
  (A : ℝ) (b c : ℝ)
  (h1 : f A = 1)
  (h2 : b * c = 2) 
  (h3 : (b * cos A) * (c * cos A) = sqrt 2) : 
  (1 / 2 * b * c * sin A = sqrt 2 / 2) := 
sorry

end NUMINAMATH_GPT_triangle_area_l1356_135674


namespace NUMINAMATH_GPT_path_shorter_factor_l1356_135685

-- Declare variables
variables (x y z : ℝ)

-- Define conditions as hypotheses
def condition1 := x = 3 * (y + z)
def condition2 := 4 * y = z + x

-- State the proof statement
theorem path_shorter_factor (condition1 : x = 3 * (y + z)) (condition2 : 4 * y = z + x) :
  (4 * y) / z = 19 :=
sorry

end NUMINAMATH_GPT_path_shorter_factor_l1356_135685


namespace NUMINAMATH_GPT_probability_not_eat_pizza_l1356_135629

theorem probability_not_eat_pizza (P_eat_pizza : ℚ) (h : P_eat_pizza = 5 / 8) : 
  ∃ P_not_eat_pizza : ℚ, P_not_eat_pizza = 3 / 8 :=
by
  use 1 - P_eat_pizza
  sorry

end NUMINAMATH_GPT_probability_not_eat_pizza_l1356_135629


namespace NUMINAMATH_GPT_statements_evaluation_l1356_135693

-- Define the statements A, B, C, D, E as propositions
def A : Prop := ∀ (A B C D E : Prop), (A → ¬B ∧ ¬C ∧ ¬D ∧ ¬E)
def B : Prop := sorry  -- Assume we have some way to read the statement B under special conditions
def C : Prop := ∀ (A B C D E : Prop), (A ∧ B ∧ C ∧ D ∧ E)
def D : Prop := sorry  -- Assume we have some way to read the statement D under special conditions
def E : Prop := A

-- Prove the conditions
theorem statements_evaluation : ¬ A ∧ ¬ C ∧ ¬ E ∧ B ∧ D :=
by
  sorry

end NUMINAMATH_GPT_statements_evaluation_l1356_135693


namespace NUMINAMATH_GPT_find_k_l1356_135691

theorem find_k (k : ℝ) (h : (-3 : ℝ)^2 + (-3 : ℝ) - k = 0) : k = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1356_135691


namespace NUMINAMATH_GPT_certain_number_is_3_l1356_135650

theorem certain_number_is_3 (x : ℚ) (h : (x / 11) * ((121 : ℚ) / 3) = 11) : x = 3 := 
sorry

end NUMINAMATH_GPT_certain_number_is_3_l1356_135650


namespace NUMINAMATH_GPT_one_eq_one_of_ab_l1356_135678

variable {a b : ℝ}

theorem one_eq_one_of_ab (h : a * b = a^2 - a * b + b^2) : 1 = 1 := by
  sorry

end NUMINAMATH_GPT_one_eq_one_of_ab_l1356_135678


namespace NUMINAMATH_GPT_xz_squared_value_l1356_135609

theorem xz_squared_value (x y z : ℝ) (h₁ : 3 * x * 5 * z = (4 * y)^2) (h₂ : (y^2 : ℝ) = (x^2 + z^2) / 2) :
  x^2 + z^2 = 16 := 
sorry

end NUMINAMATH_GPT_xz_squared_value_l1356_135609


namespace NUMINAMATH_GPT_average_visitors_30_day_month_l1356_135667

def visitors_per_day (total_visitors : ℕ) (days : ℕ) : ℕ := total_visitors / days

theorem average_visitors_30_day_month (visitors_sunday : ℕ) (visitors_other_days : ℕ) 
  (total_days : ℕ) (sundays : ℕ) (other_days : ℕ) :
  visitors_sunday = 510 →
  visitors_other_days = 240 →
  total_days = 30 →
  sundays = 4 →
  other_days = 26 →
  visitors_per_day (sundays * visitors_sunday + other_days * visitors_other_days) total_days = 276 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_average_visitors_30_day_month_l1356_135667


namespace NUMINAMATH_GPT_math_problem_l1356_135607

noncomputable def proof_problem (a b c : ℝ) (h₀ : a < 0) (h₁ : b < 0) (h₂ : c < 0) : Prop :=
  let n1 := a + 1/b
  let n2 := b + 1/c
  let n3 := c + 1/a
  (n1 ≤ -2) ∨ (n2 ≤ -2) ∨ (n3 ≤ -2)

theorem math_problem (a b c : ℝ) (h₀ : a < 0) (h₁ : b < 0) (h₂ : c < 0) : proof_problem a b c h₀ h₁ h₂ :=
sorry

end NUMINAMATH_GPT_math_problem_l1356_135607


namespace NUMINAMATH_GPT_total_ingredients_cups_l1356_135657

theorem total_ingredients_cups (butter_ratio flour_ratio sugar_ratio sugar_cups : ℚ) 
  (h_ratio : butter_ratio / sugar_ratio = 1 / 4 ∧ flour_ratio / sugar_ratio = 6 / 4) 
  (h_sugar : sugar_cups = 10) : 
  butter_ratio * (sugar_cups / sugar_ratio) + flour_ratio * (sugar_cups / sugar_ratio) + sugar_cups = 27.5 :=
by
  sorry

end NUMINAMATH_GPT_total_ingredients_cups_l1356_135657


namespace NUMINAMATH_GPT_pavan_distance_travelled_l1356_135671

theorem pavan_distance_travelled (D : ℝ) (h1 : D / 60 + D / 50 = 11) : D = 300 :=
sorry

end NUMINAMATH_GPT_pavan_distance_travelled_l1356_135671


namespace NUMINAMATH_GPT_cos_2alpha_zero_l1356_135690

theorem cos_2alpha_zero (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
(h : Real.sin (2 * α) = Real.cos (Real.pi / 4 - α)) : 
  Real.cos (2 * α) = 0 :=
by
  sorry

end NUMINAMATH_GPT_cos_2alpha_zero_l1356_135690


namespace NUMINAMATH_GPT_max_distance_equals_2_sqrt_5_l1356_135628

noncomputable def max_distance_from_point_to_line : Real :=
  let P : Real × Real := (2, -1)
  let Q : Real × Real := (-2, 1)
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem max_distance_equals_2_sqrt_5 : max_distance_from_point_to_line = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_max_distance_equals_2_sqrt_5_l1356_135628


namespace NUMINAMATH_GPT_complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_U_A_intersection_B_l1356_135680

variable (U A B : Set ℝ)
variable (x : ℝ)

def universal_set := { x | x ≤ 4 }
def set_A := { x | -2 < x ∧ x < 3 }
def set_B := { x | -3 < x ∧ x ≤ 3 }

theorem complement_U_A : (U \ A) = { x | 3 ≤ x ∧ x ≤ 4 ∨ x ≤ -2 } := sorry

theorem intersection_A_B : (A ∩ B) = { x | -2 < x ∧ x < 3 } := sorry

theorem complement_U_intersection_A_B : (U \ (A ∩ B)) = { x | 3 ≤ x ∧ x ≤ 4 ∨ x ≤ -2 } := sorry

theorem complement_U_A_intersection_B : ((U \ A) ∩ B) = { x | -3 < x ∧ x ≤ -2 ∨ x = 3 } := sorry

end NUMINAMATH_GPT_complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_U_A_intersection_B_l1356_135680


namespace NUMINAMATH_GPT_set_intersection_example_l1356_135661

theorem set_intersection_example (A : Set ℝ) (B : Set ℝ):
  A = { -1, 1, 2, 4 } → 
  B = { x | |x - 1| ≤ 1 } → 
  A ∩ B = {1, 2} :=
by
  intros hA hB
  sorry

end NUMINAMATH_GPT_set_intersection_example_l1356_135661


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1356_135645

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2*x + 15 ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 5} := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1356_135645


namespace NUMINAMATH_GPT_interest_rate_is_10_percent_l1356_135654

theorem interest_rate_is_10_percent
  (principal : ℝ)
  (interest_rate_c : ℝ) 
  (time : ℝ)
  (gain_b : ℝ)
  (interest_c : ℝ := principal * interest_rate_c / 100 * time)
  (interest_a : ℝ := interest_c - gain_b)
  (expected_rate : ℝ := (interest_a / (principal * time)) * 100)
  (h1: principal = 3500)
  (h2: interest_rate_c = 12)
  (h3: time = 3)
  (h4: gain_b = 210)
  : expected_rate = 10 := 
  by 
  sorry

end NUMINAMATH_GPT_interest_rate_is_10_percent_l1356_135654


namespace NUMINAMATH_GPT_find_y_l1356_135624

theorem find_y (x y : Int) (h1 : x + y = 280) (h2 : x - y = 200) : y = 40 := 
by 
  sorry

end NUMINAMATH_GPT_find_y_l1356_135624


namespace NUMINAMATH_GPT_original_price_of_pants_l1356_135630

theorem original_price_of_pants (P : ℝ) 
  (sale_discount : ℝ := 0.50)
  (saturday_additional_discount : ℝ := 0.20)
  (savings : ℝ := 50.40)
  (saturday_effective_discount : ℝ := 0.40) :
  savings = 0.60 * P ↔ P = 84.00 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_pants_l1356_135630


namespace NUMINAMATH_GPT_ratio_sea_horses_penguins_l1356_135621

def sea_horses := 70
def penguins := sea_horses + 85

theorem ratio_sea_horses_penguins : (70 : ℚ) / (sea_horses + 85) = 14 / 31 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_ratio_sea_horses_penguins_l1356_135621


namespace NUMINAMATH_GPT_num_undef_values_l1356_135686

theorem num_undef_values : 
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℝ, (x^2 + 4 * x - 5) * (x - 4) = 0 → x = -5 ∨ x = 1 ∨ x = 4 :=
by
  -- We are stating that there exists a natural number n such that n = 3
  -- and for all real numbers x, if (x^2 + 4*x - 5)*(x - 4) = 0,
  -- then x must be one of -5, 1, or 4.
  sorry

end NUMINAMATH_GPT_num_undef_values_l1356_135686


namespace NUMINAMATH_GPT_cannot_have_2020_l1356_135626

theorem cannot_have_2020 (a b c : ℤ) : 
  ∀ (n : ℕ), n ≥ 4 → 
  ∀ (x y z : ℕ → ℤ), 
    (x 0 = a) → (y 0 = b) → (z 0 = c) → 
    (∀ (k : ℕ), x (k + 1) = y k - z k) →
    (∀ (k : ℕ), y (k + 1) = z k - x k) →
    (∀ (k : ℕ), z (k + 1) = x k - y k) → 
    (¬ (∃ k, k > 0 ∧ k ≤ n ∧ (x k = 2020 ∨ y k = 2020 ∨ z k = 2020))) := 
by
  intros
  sorry

end NUMINAMATH_GPT_cannot_have_2020_l1356_135626


namespace NUMINAMATH_GPT_john_runs_more_than_jane_l1356_135639

def street_width : ℝ := 25
def block_side : ℝ := 500
def jane_perimeter (side : ℝ) : ℝ := 4 * side
def john_perimeter (side : ℝ) (width : ℝ) : ℝ := 4 * (side + 2 * width)

theorem john_runs_more_than_jane :
  john_perimeter block_side street_width - jane_perimeter block_side = 200 :=
by
  -- Substituting values to verify the equality:
  -- Calculate: john_perimeter 500 25 = 4 * (500 + 2 * 25) = 4 * 550 = 2200
  -- Calculate: jane_perimeter 500 = 4 * 500 = 2000
  sorry

end NUMINAMATH_GPT_john_runs_more_than_jane_l1356_135639


namespace NUMINAMATH_GPT_cars_people_equation_l1356_135606

-- Define the first condition
def condition1 (x : ℕ) : ℕ := 4 * (x - 1)

-- Define the second condition
def condition2 (x : ℕ) : ℕ := 2 * x + 8

-- Main theorem which states that the conditions lead to the equation
theorem cars_people_equation (x : ℕ) : condition1 x = condition2 x :=
by
  sorry

end NUMINAMATH_GPT_cars_people_equation_l1356_135606


namespace NUMINAMATH_GPT_factory_produces_6400_toys_per_week_l1356_135682

-- Definition of worker productivity per day
def toys_per_day : ℝ := 2133.3333333333335

-- Definition of workdays per week
def workdays_per_week : ℕ := 3

-- Definition of total toys produced per week
def toys_per_week : ℝ := toys_per_day * workdays_per_week

-- Theorem stating the total number of toys produced per week
theorem factory_produces_6400_toys_per_week : toys_per_week = 6400 :=
by
  sorry

end NUMINAMATH_GPT_factory_produces_6400_toys_per_week_l1356_135682


namespace NUMINAMATH_GPT_lilies_per_centerpiece_l1356_135604

def centerpieces := 6
def roses_per_centerpiece := 8
def orchids_per_rose := 2
def total_flowers := 120
def ratio_roses_orchids_lilies_centerpiece := 1 / 2 / 3

theorem lilies_per_centerpiece :
  ∀ (c : ℕ) (r : ℕ) (o : ℕ) (l : ℕ),
  c = centerpieces → r = roses_per_centerpiece →
  o = orchids_per_rose * r →
  total_flowers = 6 * (r + o + l) →
  ratio_roses_orchids_lilies_centerpiece = r / o / l →
  l = 10 := by sorry

end NUMINAMATH_GPT_lilies_per_centerpiece_l1356_135604


namespace NUMINAMATH_GPT_Captain_Zarnin_staffing_scheme_l1356_135603

theorem Captain_Zarnin_staffing_scheme :
  let positions := 6
  let candidates := 15
  (Nat.choose candidates positions) * 
  (Nat.factorial positions) = 3276000 :=
by
  let positions := 6
  let candidates := 15
  let ways_to_choose := Nat.choose candidates positions
  let ways_to_permute := Nat.factorial positions
  have h : (ways_to_choose * ways_to_permute) = 3276000 := sorry
  exact h

end NUMINAMATH_GPT_Captain_Zarnin_staffing_scheme_l1356_135603


namespace NUMINAMATH_GPT_two_digit_number_determined_l1356_135665

theorem two_digit_number_determined
  (x y : ℕ)
  (hx : 0 ≤ x ∧ x ≤ 9)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (h : 2 * (5 * x - 3) + y = 21) :
  10 * y + x = 72 := 
sorry

end NUMINAMATH_GPT_two_digit_number_determined_l1356_135665


namespace NUMINAMATH_GPT_total_combinations_l1356_135679

/-- Tim's rearrangement choices for the week -/
def monday_choices : Nat := 1
def tuesday_choices : Nat := 2
def wednesday_choices : Nat := 3
def thursday_choices : Nat := 2
def friday_choices : Nat := 1

theorem total_combinations :
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_combinations_l1356_135679


namespace NUMINAMATH_GPT_average_speed_correct_l1356_135605

-- Define the conditions as constants
def distance (D : ℝ) := D
def first_segment_speed := 60 -- km/h
def second_segment_speed := 24 -- km/h
def third_segment_speed := 48 -- km/h

-- Define the function that calculates average speed
noncomputable def average_speed (D : ℝ) : ℝ :=
  let t1 := (D / 3) / first_segment_speed
  let t2 := (D / 3) / second_segment_speed
  let t3 := (D / 3) / third_segment_speed
  let total_time := t1 + t2 + t3
  let total_distance := D
  total_distance / total_time

-- Prove that the average speed is 720 / 19 km/h
theorem average_speed_correct (D : ℝ) (hD : D > 0) : 
  average_speed D = 720 / 19 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_correct_l1356_135605


namespace NUMINAMATH_GPT_partOneCorrectProbability_partTwoCorrectProbability_l1356_135697

noncomputable def teachers_same_gender_probability (mA fA mB fB : ℕ) : ℚ :=
  let total_outcomes := mA * mB + mA * fB + fA * mB + fA * fB
  let same_gender := mA * mB + fA * fB
  same_gender / total_outcomes

noncomputable def teachers_same_school_probability (SA SB : ℕ) : ℚ :=
  let total_teachers := SA + SB
  let total_outcomes := (total_teachers * (total_teachers - 1)) / 2
  let same_school := (SA * (SA - 1)) / 2 + (SB * (SB - 1)) / 2
  same_school / total_outcomes

theorem partOneCorrectProbability : teachers_same_gender_probability 2 1 1 2 = 4 / 9 := by
  sorry

theorem partTwoCorrectProbability : teachers_same_school_probability 3 3 = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_partOneCorrectProbability_partTwoCorrectProbability_l1356_135697


namespace NUMINAMATH_GPT_tom_gave_jessica_some_seashells_l1356_135637

theorem tom_gave_jessica_some_seashells
  (original_seashells : ℕ := 5)
  (current_seashells : ℕ := 3) :
  original_seashells - current_seashells = 2 :=
by
  sorry

end NUMINAMATH_GPT_tom_gave_jessica_some_seashells_l1356_135637


namespace NUMINAMATH_GPT_find_abc_l1356_135646

theorem find_abc :
  ∃ (N : ℕ), (N > 0 ∧ (N % 10000 = N^2 % 10000) ∧ (N % 1000 > 100)) ∧ (N % 1000 / 100 = 937) :=
sorry

end NUMINAMATH_GPT_find_abc_l1356_135646


namespace NUMINAMATH_GPT_right_triangle_area_l1356_135611

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1356_135611


namespace NUMINAMATH_GPT_M_union_N_eq_M_l1356_135653

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | abs (p.1 * p.2) = 1 ∧ p.1 > 0}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | Real.arctan p.1 + Real.arctan p.2 = Real.pi}

theorem M_union_N_eq_M : M ∪ N = M := by
  sorry

end NUMINAMATH_GPT_M_union_N_eq_M_l1356_135653


namespace NUMINAMATH_GPT_legos_set_cost_l1356_135623

-- Definitions for the conditions
def cars_sold : ℕ := 3
def price_per_car : ℕ := 5
def total_earned : ℕ := 45

-- The statement to prove
theorem legos_set_cost :
  total_earned - (cars_sold * price_per_car) = 30 := by
  sorry

end NUMINAMATH_GPT_legos_set_cost_l1356_135623


namespace NUMINAMATH_GPT_min_value_a2_plus_b2_l1356_135676

theorem min_value_a2_plus_b2 (a b : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 2 * b = 0 -> x = -2) : (∃ a b, a = 1 ∧ b = -1 ∧ ∀ a' b', a^2 + b^2 ≥ a'^2 + b'^2) := 
by {
  sorry
}

end NUMINAMATH_GPT_min_value_a2_plus_b2_l1356_135676


namespace NUMINAMATH_GPT_intersection_of_lines_l1356_135614

theorem intersection_of_lines : ∃ (x y : ℚ), 8 * x - 5 * y = 20 ∧ 6 * x + 2 * y = 18 ∧ x = 65 / 23 ∧ y = 1 / 2 :=
by {
  -- The solution to the theorem is left as an exercise
  sorry
}

end NUMINAMATH_GPT_intersection_of_lines_l1356_135614


namespace NUMINAMATH_GPT_min_value_of_reciprocals_l1356_135688

theorem min_value_of_reciprocals (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) : 
  ∃ x, x = (1 / a) + (1 / b) ∧ x ≥ 4 := 
sorry

end NUMINAMATH_GPT_min_value_of_reciprocals_l1356_135688


namespace NUMINAMATH_GPT_remainder_of_sum_l1356_135617

theorem remainder_of_sum (D k l : ℕ) (hk : 242 = k * D + 11) (hl : 698 = l * D + 18) :
  (242 + 698) % D = 29 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l1356_135617


namespace NUMINAMATH_GPT_circumscribed_steiner_ellipse_inscribed_steiner_ellipse_l1356_135666

variable {α β γ : ℝ}

/-- The equation of the circumscribed Steiner ellipse in barycentric coordinates -/
theorem circumscribed_steiner_ellipse (h : α + β + γ = 1) :
  β * γ + α * γ + α * β = 0 :=
sorry

/-- The equation of the inscribed Steiner ellipse in barycentric coordinates -/
theorem inscribed_steiner_ellipse (h : α + β + γ = 1) :
  2 * β * γ + 2 * α * γ + 2 * α * β = α^2 + β^2 + γ^2 :=
sorry

end NUMINAMATH_GPT_circumscribed_steiner_ellipse_inscribed_steiner_ellipse_l1356_135666


namespace NUMINAMATH_GPT_negation_of_universal_l1356_135608

theorem negation_of_universal (P : Prop) :
  (¬ (∀ x : ℝ, x > 0 → x^3 > 0)) ↔ (∃ x : ℝ, x > 0 ∧ x^3 ≤ 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_l1356_135608


namespace NUMINAMATH_GPT_solve_for_x_l1356_135615

theorem solve_for_x (x y z : ℤ) (h1 : x + y + z = 14) (h2 : x - y - z = 60) (h3 : x + z = 2 * y) : x = 37 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1356_135615


namespace NUMINAMATH_GPT_triangle_angle_B_l1356_135660

theorem triangle_angle_B (A B C : ℕ) (h₁ : B + C = 110) (h₂ : A + B + C = 180) (h₃ : A = 70) :
  B = 70 ∨ B = 55 ∨ B = 40 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_B_l1356_135660


namespace NUMINAMATH_GPT_average_weight_increase_l1356_135644

theorem average_weight_increase 
  (n : ℕ) (A : ℕ → ℝ)
  (h_total : n = 10)
  (h_replace : A 65 = 137) : 
  (137 - 65) / 10 = 7.2 := 
by 
  sorry

end NUMINAMATH_GPT_average_weight_increase_l1356_135644


namespace NUMINAMATH_GPT_bushels_given_away_l1356_135600

-- Definitions from the problem conditions
def initial_bushels : ℕ := 50
def ears_per_bushel : ℕ := 14
def remaining_ears : ℕ := 357

-- Theorem to prove the number of bushels given away
theorem bushels_given_away : 
  initial_bushels * ears_per_bushel - remaining_ears = 24 * ears_per_bushel :=
by
  sorry

end NUMINAMATH_GPT_bushels_given_away_l1356_135600


namespace NUMINAMATH_GPT_training_trip_duration_l1356_135655

-- Define the number of supervisors
def num_supervisors : ℕ := 15

-- Define the number of supervisors overseeing the pool each day
def supervisors_per_day : ℕ := 3

-- Define the number of pairs supervised per day
def pairs_per_day : ℕ := (supervisors_per_day * (supervisors_per_day - 1)) / 2

-- Define the total number of pairs from the given number of supervisors
def total_pairs : ℕ := (num_supervisors * (num_supervisors - 1)) / 2

-- Define the number of days required
def num_days : ℕ := total_pairs / pairs_per_day

-- The theorem we need to prove
theorem training_trip_duration : 
  (num_supervisors = 15) ∧
  (supervisors_per_day = 3) ∧
  (∀ (a b : ℕ), a * (a - 1) / 2 = b * (b - 1) / 2 → a = b) ∧ 
  (∀ (N : ℕ), total_pairs = N * pairs_per_day → N = 35) :=
by
  sorry

end NUMINAMATH_GPT_training_trip_duration_l1356_135655


namespace NUMINAMATH_GPT_train_speed_is_144_kmph_l1356_135675

noncomputable def length_of_train : ℝ := 130 -- in meters
noncomputable def time_to_cross_pole : ℝ := 3.249740020798336 -- in seconds
noncomputable def speed_m_per_s : ℝ := length_of_train / time_to_cross_pole -- in m/s
noncomputable def conversion_factor : ℝ := 3.6 -- 1 m/s = 3.6 km/hr

theorem train_speed_is_144_kmph : speed_m_per_s * conversion_factor = 144 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_144_kmph_l1356_135675


namespace NUMINAMATH_GPT_total_savings_l1356_135616

theorem total_savings (savings_sep savings_oct : ℕ) 
  (h1 : savings_sep = 260)
  (h2 : savings_oct = savings_sep + 30) :
  savings_sep + savings_oct = 550 := 
sorry

end NUMINAMATH_GPT_total_savings_l1356_135616


namespace NUMINAMATH_GPT_no_solution_in_natural_numbers_l1356_135696

theorem no_solution_in_natural_numbers (x y z : ℕ) : ¬((2 * x) ^ (2 * x) - 1 = y ^ (z + 1)) := 
  sorry

end NUMINAMATH_GPT_no_solution_in_natural_numbers_l1356_135696


namespace NUMINAMATH_GPT_adam_cat_food_vs_dog_food_l1356_135631

def cat_packages := 15
def dog_packages := 10
def cans_per_cat_package := 12
def cans_per_dog_package := 8

theorem adam_cat_food_vs_dog_food:
  cat_packages * cans_per_cat_package - dog_packages * cans_per_dog_package = 100 :=
by
  sorry

end NUMINAMATH_GPT_adam_cat_food_vs_dog_food_l1356_135631


namespace NUMINAMATH_GPT_debby_bought_bottles_l1356_135672

def bottles_per_day : ℕ := 109
def days_lasting : ℕ := 74

theorem debby_bought_bottles : bottles_per_day * days_lasting = 8066 := by
  sorry

end NUMINAMATH_GPT_debby_bought_bottles_l1356_135672


namespace NUMINAMATH_GPT_distance_from_P_to_AB_l1356_135658

-- Let \(ABC\) be an isosceles triangle where \(AB\) is the base. 
-- An altitude from vertex \(C\) to base \(AB\) measures 6 units.
-- A line drawn through a point \(P\) inside the triangle, parallel to base \(AB\), 
-- divides the triangle into two regions of equal area.
-- The vertex angle at \(C\) is a right angle.
-- Prove that the distance from \(P\) to \(AB\) is 3 units.

theorem distance_from_P_to_AB :
  ∀ (A B C P : Type)
    (distance_AB distance_AC distance_BC : ℝ)
    (is_isosceles : distance_AC = distance_BC)
    (right_angle_C : distance_AC^2 + distance_BC^2 = distance_AB^2)
    (altitude_C : distance_BC = 6)
    (line_through_P_parallel_to_AB : ∃ (P_x : ℝ), 0 < P_x ∧ P_x < distance_BC),
  ∃ (distance_P_to_AB : ℝ), distance_P_to_AB = 3 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_P_to_AB_l1356_135658


namespace NUMINAMATH_GPT_compare_numbers_l1356_135692

theorem compare_numbers :
  2^27 < 10^9 ∧ 10^9 < 5^13 :=
by {
  sorry
}

end NUMINAMATH_GPT_compare_numbers_l1356_135692


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1356_135683

theorem problem1 : 9 - 5 - (-4) + 2 = 10 := by
  sorry

theorem problem2 : (- (3 / 4) + 7 / 12 - 5 / 9) / (-(1 / 36)) = 26 := by
  sorry

theorem problem3 : -2^4 - ((-5) + 1 / 2) * (4 / 11) + (-2)^3 / (abs (-3^2 + 1)) = -15 := by
  sorry

theorem problem4 : (100 - 1 / 72) * (-36) = -(3600) + (1 / 2) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1356_135683


namespace NUMINAMATH_GPT_probability_of_pink_gumball_l1356_135668

theorem probability_of_pink_gumball (P_B P_P : ℝ)
    (h1 : P_B ^ 2 = 25 / 49)
    (h2 : P_B + P_P = 1) :
    P_P = 2 / 7 := 
    sorry

end NUMINAMATH_GPT_probability_of_pink_gumball_l1356_135668


namespace NUMINAMATH_GPT_toby_photo_shoot_l1356_135642

theorem toby_photo_shoot (initial_photos : ℕ) (deleted_bad_shots : ℕ) (cat_pictures : ℕ) (deleted_post_editing : ℕ) (final_photos : ℕ) (photo_shoot_photos : ℕ) :
  initial_photos = 63 →
  deleted_bad_shots = 7 →
  cat_pictures = 15 →
  deleted_post_editing = 3 →
  final_photos = 84 →
  final_photos = initial_photos - deleted_bad_shots + cat_pictures + photo_shoot_photos - deleted_post_editing →
  photo_shoot_photos = 16 :=
by
  intros
  sorry

end NUMINAMATH_GPT_toby_photo_shoot_l1356_135642


namespace NUMINAMATH_GPT_cos_pi_over_3_plus_double_alpha_l1356_135613

theorem cos_pi_over_3_plus_double_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 4) :
  Real.cos (π / 3 + 2 * α) = -7 / 8 :=
sorry

end NUMINAMATH_GPT_cos_pi_over_3_plus_double_alpha_l1356_135613


namespace NUMINAMATH_GPT_polygon_sides_eq_eight_l1356_135673

theorem polygon_sides_eq_eight (x : ℕ) (h : x ≥ 3) 
  (h1 : 2 * (x - 2) = 180 * (x - 2) / 90) 
  (h2 : ∀ x, x + 2 * (x - 2) = x * (x - 3) / 2) : 
  x = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_eight_l1356_135673


namespace NUMINAMATH_GPT_g_five_l1356_135664

variable (g : ℝ → ℝ)

-- Given conditions
axiom g_add : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_three : g 3 = 4

-- Prove g(5) = 16 * (1 / 4)^(1/3)
theorem g_five : g 5 = 16 * (1 / 4)^(1/3) := by
  sorry

end NUMINAMATH_GPT_g_five_l1356_135664


namespace NUMINAMATH_GPT_jerry_age_l1356_135601

theorem jerry_age
  (M J : ℕ)
  (h1 : M = 2 * J + 5)
  (h2 : M = 21) :
  J = 8 :=
by
  sorry

end NUMINAMATH_GPT_jerry_age_l1356_135601


namespace NUMINAMATH_GPT_corn_growth_ratio_l1356_135669

theorem corn_growth_ratio 
  (growth_first_week : ℕ := 2) 
  (growth_second_week : ℕ) 
  (growth_third_week : ℕ) 
  (total_height : ℕ := 22) 
  (r : ℕ) 
  (h1 : growth_second_week = 2 * r) 
  (h2 : growth_third_week = 4 * (2 * r)) 
  (h3 : growth_first_week + growth_second_week + growth_third_week = total_height) 
  : r = 2 := 
by 
  sorry

end NUMINAMATH_GPT_corn_growth_ratio_l1356_135669


namespace NUMINAMATH_GPT_interest_difference_l1356_135602

theorem interest_difference (P P_B : ℝ) (R_A R_B T : ℝ)
    (h₁ : P = 10000)
    (h₂ : P_B = 4000.0000000000005)
    (h₃ : R_A = 15)
    (h₄ : R_B = 18)
    (h₅ : T = 2) :
    let P_A := P - P_B
    let I_A := (P_A * R_A * T) / 100
    let I_B := (P_B * R_B * T) / 100
    I_A - I_B = 359.99999999999965 := 
by
  sorry

end NUMINAMATH_GPT_interest_difference_l1356_135602


namespace NUMINAMATH_GPT_problem_I_problem_II_l1356_135620

namespace MathProof

-- Define the function f(x) given m
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2 * |x + 1|

-- Problem (I)
theorem problem_I (x : ℝ) : (5 - |x - 1| - 2 * |x + 1| > 2) ↔ (-4/3 < x ∧ x < 0) := 
sorry

-- Define the quadratic function
def y (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Problem (II)
theorem problem_II (m : ℝ) : (∀ x : ℝ, ∃ t : ℝ, t = x^2 + 2*x + 3 ∧ t = f m x) ↔ (m ≥ 4) :=
sorry

end MathProof

end NUMINAMATH_GPT_problem_I_problem_II_l1356_135620


namespace NUMINAMATH_GPT_min_value_of_f_l1356_135659

noncomputable def f (x y : ℝ) : ℝ := (x^2 * y) / (x^3 + y^3)

theorem min_value_of_f :
  (∀ (x y : ℝ), (1/3 ≤ x ∧ x ≤ 2/3) ∧ (1/4 ≤ y ∧ y ≤ 1/2) → f x y ≥ 12 / 35) ∧
  ∃ (x y : ℝ), (1/3 ≤ x ∧ x ≤ 2/3) ∧ (1/4 ≤ y ∧ y ≤ 1/2) ∧ f x y = 12 / 35 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l1356_135659


namespace NUMINAMATH_GPT_jessica_quarters_l1356_135627

theorem jessica_quarters (original_borrowed : ℕ) (quarters_borrowed : ℕ) 
  (H1 : original_borrowed = 8)
  (H2 : quarters_borrowed = 3) : 
  original_borrowed - quarters_borrowed = 5 := sorry

end NUMINAMATH_GPT_jessica_quarters_l1356_135627


namespace NUMINAMATH_GPT_has_two_distinct_real_roots_parabola_equation_l1356_135641

open Real

-- Define the quadratic polynomial
def quad_poly (m : ℝ) (x : ℝ) : ℝ := x^2 - 2 * m * x + m^2 - 4

-- Question 1: Prove that the quadratic equation has two distinct real roots
theorem has_two_distinct_real_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (quad_poly m x₁ = 0) ∧ (quad_poly m x₂ = 0) := by
  sorry

-- Question 2: Prove the equation of the parabola given certain conditions
theorem parabola_equation (m : ℝ) (hx : quad_poly m 0 = 0) : 
  m = 0 ∧ ∀ x : ℝ, quad_poly m x = x^2 - 4 := by
  sorry

end NUMINAMATH_GPT_has_two_distinct_real_roots_parabola_equation_l1356_135641
