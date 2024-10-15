import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_first_term_l1959_195961

theorem arithmetic_sequence_first_term (a d : ℚ) 
  (h1 : 30 * (2 * a + 59 * d) = 500) 
  (h2 : 30 * (2 * a + 179 * d) = 2900) : 
  a = -34 / 3 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_first_term_l1959_195961


namespace NUMINAMATH_GPT_factorable_polynomial_with_integer_coeffs_l1959_195954

theorem factorable_polynomial_with_integer_coeffs (m : ℤ) : 
  ∃ A B C D E F : ℤ, 
  (A * D = 1) ∧ (B * E = 0) ∧ (A * E + B * D = 5) ∧ 
  (A * F + C * D = 1) ∧ (B * F + C * E = 2 * m) ∧ (C * F = -10) ↔ m = 5 := sorry

end NUMINAMATH_GPT_factorable_polynomial_with_integer_coeffs_l1959_195954


namespace NUMINAMATH_GPT_tiles_needed_l1959_195982

theorem tiles_needed (A_classroom : ℝ) (side_length_tile : ℝ) (H_classroom : A_classroom = 56) (H_side_length : side_length_tile = 0.4) :
  A_classroom / (side_length_tile * side_length_tile) = 350 :=
by
  sorry

end NUMINAMATH_GPT_tiles_needed_l1959_195982


namespace NUMINAMATH_GPT_range_a_l1959_195960

theorem range_a (a : ℝ) : (∀ x, x > 0 → x^2 - a * x + 1 > 0) → -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_GPT_range_a_l1959_195960


namespace NUMINAMATH_GPT_unique_function_solution_l1959_195941

theorem unique_function_solution (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, x ≥ 1 → f x ≥ 1)
  (h₂ : ∀ x : ℝ, x ≥ 1 → f x ≤ 2 * (x + 1))
  (h₃ : ∀ x : ℝ, x ≥ 1 → f (x + 1) = (f x)^2/x - 1/x) :
  ∀ x : ℝ, x ≥ 1 → f x = x + 1 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_unique_function_solution_l1959_195941


namespace NUMINAMATH_GPT_longest_sticks_triangle_shortest_sticks_not_triangle_l1959_195973

-- Define the lengths of the six sticks in descending order
variables {a1 a2 a3 a4 a5 a6 : ℝ}

-- Assuming the conditions
axiom h1 : a1 ≥ a2
axiom h2 : a2 ≥ a3
axiom h3 : a3 ≥ a4
axiom h4 : a4 ≥ a5
axiom h5 : a5 ≥ a6
axiom h6 : a1 + a2 > a3

-- Proof problem 1: It is always possible to form a triangle from the three longest sticks.
theorem longest_sticks_triangle : a1 < a2 + a3 := by sorry

-- Assuming an additional condition for proof problem 2
axiom two_triangles_formed : ∃ b1 b2 b3 b4 b5 b6: ℝ, 
  ((b1 + b2 > b3 ∧ b1 + b3 > b2 ∧ b2 + b3 > b1) ∧
   (b4 + b5 > b6 ∧ b4 + b6 > b5 ∧ b5 + b6 > b4 ∧ 
    a1 = b1 ∧ a2 = b2 ∧ a3 = b3 ∧ a4 = b4 ∧ a5 = b5 ∧ a6 = b6))

-- Proof problem 2: It is not always possible to form a triangle from the three shortest sticks.
theorem shortest_sticks_not_triangle : ¬(a4 < a5 + a6 ∧ a5 < a4 + a6 ∧ a6 < a4 + a5) := by sorry

end NUMINAMATH_GPT_longest_sticks_triangle_shortest_sticks_not_triangle_l1959_195973


namespace NUMINAMATH_GPT_root_range_m_l1959_195950

theorem root_range_m (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * m * x + 4 = 0 → (x > 1 ∧ ∃ y : ℝ, y < 1 ∧ y^2 - 2 * m * y + 4 = 0)
  ∨ (x < 1 ∧ ∃ y : ℝ, y > 1 ∧ y^2 - 2 * m * y + 4 = 0))
  → m > 5 / 2 := 
sorry

end NUMINAMATH_GPT_root_range_m_l1959_195950


namespace NUMINAMATH_GPT_lower_rent_amount_l1959_195966

-- Define the conditions and proof goal
variable (T R : ℕ)
variable (L : ℕ)

-- Condition 1: Total rent is $1000
def total_rent (T R : ℕ) (L : ℕ) := 60 * R + L * (T - R)

-- Condition 2: Reduction by 20% when 10 rooms are swapped
def reduced_rent (T R : ℕ) (L : ℕ) := 60 * (R - 10) + L * (T - R + 10)

-- Proof that the lower rent amount is $40 given the conditions
theorem lower_rent_amount (h1 : total_rent T R L = 1000)
                         (h2 : reduced_rent T R L = 800) : L = 40 :=
by
  sorry

end NUMINAMATH_GPT_lower_rent_amount_l1959_195966


namespace NUMINAMATH_GPT_Petya_wins_optimally_l1959_195933

-- Defining the game state and rules
inductive GameState
| PetyaWin
| VasyaWin

-- Rules of the game
def game_rule (n : ℕ) : Prop :=
  n > 0 ∧ (n % 3 = 0 ∨ n % 3 = 1 ∨ n % 3 = 2)

-- Determine the winner given the initial number of minuses
def determine_winner (n : ℕ) : GameState :=
  if n % 3 = 0 then GameState.PetyaWin else GameState.VasyaWin

-- Theorem: Petya will win the game if both play optimally
theorem Petya_wins_optimally (n : ℕ) (h1 : n = 2021) (h2 : game_rule n) : determine_winner n = GameState.PetyaWin :=
by {
  sorry
}

end NUMINAMATH_GPT_Petya_wins_optimally_l1959_195933


namespace NUMINAMATH_GPT_second_train_speed_l1959_195953

theorem second_train_speed (d : ℝ) (s₁ : ℝ) (t₁ : ℝ) (t₂ : ℝ) (meet_time : ℝ) (total_distance : ℝ) :
  d = 110 ∧ s₁ = 20 ∧ t₁ = 3 ∧ t₂ = 2 ∧ meet_time = 10 ∧ total_distance = d →
  60 + 2 * (total_distance - 60) / 2 = 110 →
  (total_distance - 60) / 2 = 25 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_second_train_speed_l1959_195953


namespace NUMINAMATH_GPT_arctan_tan_expression_l1959_195905

noncomputable def tan (x : ℝ) : ℝ := sorry
noncomputable def arctan (x : ℝ) : ℝ := sorry

theorem arctan_tan_expression :
  arctan (tan 65 - 2 * tan 40) = 25 := sorry

end NUMINAMATH_GPT_arctan_tan_expression_l1959_195905


namespace NUMINAMATH_GPT_intersecting_lines_ratio_l1959_195937

theorem intersecting_lines_ratio (k1 k2 a : ℝ) (h1 : k1 * a + 4 = 0) (h2 : k2 * a - 2 = 0) : k1 / k2 = -2 :=
by
    sorry

end NUMINAMATH_GPT_intersecting_lines_ratio_l1959_195937


namespace NUMINAMATH_GPT_ninety_eight_times_ninety_eight_l1959_195924

theorem ninety_eight_times_ninety_eight : 98 * 98 = 9604 := 
by
  sorry

end NUMINAMATH_GPT_ninety_eight_times_ninety_eight_l1959_195924


namespace NUMINAMATH_GPT_perpendicular_lines_l1959_195956

theorem perpendicular_lines (a : ℝ) 
  (h1 : (3 : ℝ) * y + (2 : ℝ) * x - 6 = 0) 
  (h2 : (4 : ℝ) * y + a * x - 5 = 0) : 
  a = -6 :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_l1959_195956


namespace NUMINAMATH_GPT_carpenter_needs_80_woodblocks_l1959_195974

-- Define the number of logs the carpenter currently has
def existing_logs : ℕ := 8

-- Define the number of woodblocks each log can produce
def woodblocks_per_log : ℕ := 5

-- Define the number of additional logs needed
def additional_logs : ℕ := 8

-- Calculate the total number of woodblocks needed
def total_woodblocks_needed : ℕ := 
  (existing_logs * woodblocks_per_log) + (additional_logs * woodblocks_per_log)

-- Prove that the total number of woodblocks needed is 80
theorem carpenter_needs_80_woodblocks : total_woodblocks_needed = 80 := by
  sorry

end NUMINAMATH_GPT_carpenter_needs_80_woodblocks_l1959_195974


namespace NUMINAMATH_GPT_divisible_by_77_l1959_195909

theorem divisible_by_77 (n : ℤ) : ∃ k : ℤ, n^18 - n^12 - n^8 + n^2 = 77 * k :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_77_l1959_195909


namespace NUMINAMATH_GPT_product_of_terms_form_l1959_195967

theorem product_of_terms_form 
  (a b c d : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) :
  ∃ p q : ℝ, 
    (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5) = p + q * Real.sqrt 5 
    ∧ 0 ≤ p 
    ∧ 0 ≤ q := 
by
  let p := a * c + 5 * b * d
  let q := a * d + b * c
  use p, q
  sorry

end NUMINAMATH_GPT_product_of_terms_form_l1959_195967


namespace NUMINAMATH_GPT_find_set_B_l1959_195994

open Set

variable (U : Finset ℕ) (A B : Finset ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7})
variable (h1 : (U \ (A ∪ B)) = {1, 3})
variable (h2 : A ∩ (U \ B) = {2, 5})

theorem find_set_B : B = {4, 6, 7} := by
  sorry

end NUMINAMATH_GPT_find_set_B_l1959_195994


namespace NUMINAMATH_GPT_fire_alarms_and_passengers_discrete_l1959_195904

-- Definitions of the random variables
def xi₁ : ℕ := sorry  -- number of fire alarms in a city within one day
def xi₂ : ℝ := sorry  -- temperature in a city within one day
def xi₃ : ℕ := sorry  -- number of passengers at a train station in a city within a month

-- Defining the concept of discrete random variable
def is_discrete (X : Type) : Prop := 
  ∃ f : X → ℕ, ∀ x : X, ∃ n : ℕ, f x = n

-- Statement of the proof problem
theorem fire_alarms_and_passengers_discrete :
  is_discrete ℕ ∧ is_discrete ℕ ∧ ¬ is_discrete ℝ :=
by
  have xi₁_discrete : is_discrete ℕ := sorry
  have xi₃_discrete : is_discrete ℕ := sorry
  have xi₂_not_discrete : ¬ is_discrete ℝ := sorry
  exact ⟨xi₁_discrete, xi₃_discrete, xi₂_not_discrete⟩

end NUMINAMATH_GPT_fire_alarms_and_passengers_discrete_l1959_195904


namespace NUMINAMATH_GPT_squares_of_roots_equation_l1959_195986

theorem squares_of_roots_equation (a b x : ℂ) 
  (h : ab * x^2 - (a + b) * x + 1 = 0) : 
  a^2 * b^2 * x^2 - (a^2 + b^2) * x + 1 = 0 :=
sorry

end NUMINAMATH_GPT_squares_of_roots_equation_l1959_195986


namespace NUMINAMATH_GPT_even_composite_sum_consecutive_odd_numbers_l1959_195918

theorem even_composite_sum_consecutive_odd_numbers (a k : ℤ) : ∃ (n m : ℤ), n = 2 * k ∧ m = n * (2 * a + n) ∧ m % 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_even_composite_sum_consecutive_odd_numbers_l1959_195918


namespace NUMINAMATH_GPT_find_unknown_number_l1959_195965

theorem find_unknown_number (y : ℝ) (h : 25 / y = 80 / 100) : y = 31.25 :=
sorry

end NUMINAMATH_GPT_find_unknown_number_l1959_195965


namespace NUMINAMATH_GPT_negation_of_exists_l1959_195995

theorem negation_of_exists (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_exists_l1959_195995


namespace NUMINAMATH_GPT_graph_t_intersects_x_axis_exists_integer_a_with_integer_points_on_x_axis_intersection_l1959_195987

open Real

def function_y (a x : ℝ) : ℝ := (4 * a + 2) * x^2 + (9 - 6 * a) * x - 4 * a + 4

theorem graph_t_intersects_x_axis (a : ℝ) : ∃ x : ℝ, function_y a x = 0 :=
by sorry

theorem exists_integer_a_with_integer_points_on_x_axis_intersection :
  ∃ (a : ℤ), 
  (∀ x : ℝ, (function_y a x = 0) → ∃ (x_int : ℤ), x = x_int) ∧ 
  (a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1) :=
by sorry

end NUMINAMATH_GPT_graph_t_intersects_x_axis_exists_integer_a_with_integer_points_on_x_axis_intersection_l1959_195987


namespace NUMINAMATH_GPT_audrey_lost_pieces_l1959_195902

theorem audrey_lost_pieces {total_pieces_on_board : ℕ} {thomas_lost : ℕ} {initial_pieces_each : ℕ} (h1 : total_pieces_on_board = 21) (h2 : thomas_lost = 5) (h3 : initial_pieces_each = 16) :
  (initial_pieces_each - (total_pieces_on_board - (initial_pieces_each - thomas_lost))) = 6 :=
by
  sorry

end NUMINAMATH_GPT_audrey_lost_pieces_l1959_195902


namespace NUMINAMATH_GPT_patsy_deviled_eggs_l1959_195991

-- Definitions based on given problem conditions
def guests : ℕ := 30
def appetizers_per_guest : ℕ := 6
def total_appetizers_needed : ℕ := appetizers_per_guest * guests
def pigs_in_blanket : ℕ := 2
def kebabs : ℕ := 2
def additional_appetizers_needed (already_planned : ℕ) : ℕ := 8 + already_planned
def already_planned_appetizers : ℕ := pigs_in_blanket + kebabs
def total_appetizers_planned : ℕ := additional_appetizers_needed already_planned_appetizers

-- The proof problem statement
theorem patsy_deviled_eggs : total_appetizers_needed = total_appetizers_planned * 12 → 
                            total_appetizers_planned = already_planned_appetizers + 8 →
                            (total_appetizers_planned - already_planned_appetizers) = 8 :=
by
  sorry

end NUMINAMATH_GPT_patsy_deviled_eggs_l1959_195991


namespace NUMINAMATH_GPT_three_digit_number_exists_l1959_195963

theorem three_digit_number_exists : 
  ∃ (x y z : ℕ), 
  (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧ 
  (100 * x + 10 * z + y + 1 = 2 * (100 * y + 10 * z + x)) ∧ 
  (100 * x + 10 * z + y = 793) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_exists_l1959_195963


namespace NUMINAMATH_GPT_find_C_l1959_195903

noncomputable def A : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x < a}
def isSolutionC (C : Set ℝ) : Prop := C = {2, 3}

theorem find_C : ∃ C : Set ℝ, isSolutionC C ∧ ∀ a, (A ∪ B a = A) ↔ a ∈ C :=
by
  sorry

end NUMINAMATH_GPT_find_C_l1959_195903


namespace NUMINAMATH_GPT_hypotenuse_length_l1959_195979

theorem hypotenuse_length (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a^2 + b^2 + c^2 = 1800) : 
  c = 30 :=
sorry

end NUMINAMATH_GPT_hypotenuse_length_l1959_195979


namespace NUMINAMATH_GPT_divisible_by_6_implies_divisible_by_2_not_divisible_by_2_implies_not_divisible_by_6_equivalence_of_propositions_l1959_195900

theorem divisible_by_6_implies_divisible_by_2 :
  ∀ (n : ℤ), (6 ∣ n) → (2 ∣ n) :=
by sorry

theorem not_divisible_by_2_implies_not_divisible_by_6 :
  ∀ (n : ℤ), ¬ (2 ∣ n) → ¬ (6 ∣ n) :=
by sorry

theorem equivalence_of_propositions :
  (∀ (n : ℤ), (6 ∣ n) → (2 ∣ n)) ↔ (∀ (n : ℤ), ¬ (2 ∣ n) → ¬ (6 ∣ n)) :=
by sorry


end NUMINAMATH_GPT_divisible_by_6_implies_divisible_by_2_not_divisible_by_2_implies_not_divisible_by_6_equivalence_of_propositions_l1959_195900


namespace NUMINAMATH_GPT_fraction_girls_at_meet_l1959_195998

-- Define the conditions of the problem
def numStudentsMaplewood : ℕ := 300
def ratioBoysGirlsMaplewood : ℕ × ℕ := (3, 2)
def numStudentsRiverview : ℕ := 240
def ratioBoysGirlsRiverview : ℕ × ℕ := (3, 5)

-- Define the combined number of students and number of girls
def totalStudentsMaplewood := numStudentsMaplewood
def totalStudentsRiverview := numStudentsRiverview

def numGirlsMaplewood : ℕ :=
  let (b, g) := ratioBoysGirlsMaplewood
  (totalStudentsMaplewood * g) / (b + g)

def numGirlsRiverview : ℕ :=
  let (b, g) := ratioBoysGirlsRiverview
  (totalStudentsRiverview * g) / (b + g)

def totalGirls := numGirlsMaplewood + numGirlsRiverview
def totalStudents := totalStudentsMaplewood + totalStudentsRiverview

-- Formalize the actual proof statement
theorem fraction_girls_at_meet : 
  (totalGirls : ℚ) / totalStudents = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_fraction_girls_at_meet_l1959_195998


namespace NUMINAMATH_GPT_min_value_3x_4y_l1959_195907

open Real

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y = 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_3x_4y_l1959_195907


namespace NUMINAMATH_GPT_percentage_pure_acid_l1959_195999

theorem percentage_pure_acid (volume_pure_acid total_volume: ℝ) (h1 : volume_pure_acid = 1.4) (h2 : total_volume = 4) : 
  (volume_pure_acid / total_volume) * 100 = 35 := 
by
  -- Given metric volumes of pure acid and total solution, we need to prove the percentage 
  -- Here, we assert the conditions and conclude the result
  sorry

end NUMINAMATH_GPT_percentage_pure_acid_l1959_195999


namespace NUMINAMATH_GPT_delta_gj_l1959_195938

def vj := 120
def total := 770
def gj := total - vj

theorem delta_gj : gj - 5 * vj = 50 := by
  sorry

end NUMINAMATH_GPT_delta_gj_l1959_195938


namespace NUMINAMATH_GPT_range_of_a_l1959_195976

theorem range_of_a (x y a : ℝ) (h1 : x < y) (h2 : (a - 3) * x > (a - 3) * y) : a < 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1959_195976


namespace NUMINAMATH_GPT_difference_between_two_numbers_l1959_195929

theorem difference_between_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 10) 
  (h3 : x^2 - y^2 = 200) : 
  x - y = 10 :=
by 
  sorry

end NUMINAMATH_GPT_difference_between_two_numbers_l1959_195929


namespace NUMINAMATH_GPT_stock_index_approximation_l1959_195917

noncomputable def stock_index_after_days (initial_index : ℝ) (daily_increase : ℝ) (days : ℕ) : ℝ :=
  initial_index * (1 + daily_increase / 100) ^ (days - 1)

theorem stock_index_approximation :
  let initial_index := 2
  let daily_increase := 0.02
  let days := 100
  abs (stock_index_after_days initial_index daily_increase days - 2.041) < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_stock_index_approximation_l1959_195917


namespace NUMINAMATH_GPT_length_of_each_train_l1959_195989

theorem length_of_each_train (L : ℝ) (s1 : ℝ) (s2 : ℝ) (t : ℝ)
    (h1 : s1 = 46) (h2 : s2 = 36) (h3 : t = 144) (h4 : 2 * L = ((s1 - s2) * (5 / 18)) * t) :
    L = 200 := 
sorry

end NUMINAMATH_GPT_length_of_each_train_l1959_195989


namespace NUMINAMATH_GPT_new_area_is_726_l1959_195934

variable (l w : ℝ)
variable (h_area : l * w = 576)
variable (l' : ℝ := 1.20 * l)
variable (w' : ℝ := 1.05 * w)

theorem new_area_is_726 : l' * w' = 726 := by
  sorry

end NUMINAMATH_GPT_new_area_is_726_l1959_195934


namespace NUMINAMATH_GPT_amit_work_days_l1959_195952

theorem amit_work_days (x : ℕ) (h : 2 * (1 / x : ℚ) + 16 * (1 / 20 : ℚ) = 1) : x = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_amit_work_days_l1959_195952


namespace NUMINAMATH_GPT_yogurt_amount_l1959_195985

namespace SmoothieProblem

def strawberries := 0.2 -- cups
def orange_juice := 0.2 -- cups
def total_ingredients := 0.5 -- cups

def yogurt_used := total_ingredients - (strawberries + orange_juice)

theorem yogurt_amount : yogurt_used = 0.1 :=
by
  unfold yogurt_used strawberries orange_juice total_ingredients
  norm_num
  sorry  -- Proof can be filled in as needed

end SmoothieProblem

end NUMINAMATH_GPT_yogurt_amount_l1959_195985


namespace NUMINAMATH_GPT_guo_can_pay_exactly_l1959_195930

theorem guo_can_pay_exactly (
  x y z : ℕ
) (h : 10 * x + 20 * y + 50 * z = 20000) : ∃ a b c : ℕ, a + 2 * b + 5 * c = 1000 :=
sorry

end NUMINAMATH_GPT_guo_can_pay_exactly_l1959_195930


namespace NUMINAMATH_GPT_condition_holds_iff_b_eq_10_l1959_195940

-- Define xn based on given conditions in the problem
def x_n (b : ℕ) (n : ℕ) : ℕ :=
  if b > 5 then
    b^(2*n) + b^n + 3*b - 5
  else
    0

-- State the main theorem to be proven in Lean
theorem condition_holds_iff_b_eq_10 :
  ∀ (b : ℕ), (b > 5) ↔ ∃ M : ℕ, ∀ n : ℕ, n > M → ∃ k : ℕ, x_n b n = k^2 := sorry

end NUMINAMATH_GPT_condition_holds_iff_b_eq_10_l1959_195940


namespace NUMINAMATH_GPT_sphere_surface_area_l1959_195928

theorem sphere_surface_area 
  (a b c : ℝ) 
  (h1 : a = 1)
  (h2 : b = 2)
  (h3 : c = 2)
  (h_spherical_condition : ∃ R : ℝ, ∀ (x y z : ℝ), x^2 + y^2 + z^2 = (2 * R)^2) :
  4 * Real.pi * ((3 / 2)^2) = 9 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l1959_195928


namespace NUMINAMATH_GPT_sum_of_angles_l1959_195925

-- Definitions of acute, right, and obtuse angles
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def is_right (θ : ℝ) : Prop := θ = 90
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- The main statement we want to prove
theorem sum_of_angles :
  (∀ (α β : ℝ), is_acute α ∧ is_acute β → is_acute (α + β) ∨ is_right (α + β) ∨ is_obtuse (α + β)) ∧
  (∀ (α β : ℝ), is_acute α ∧ is_right β → is_obtuse (α + β)) :=
by sorry

end NUMINAMATH_GPT_sum_of_angles_l1959_195925


namespace NUMINAMATH_GPT_solve_quadratic_equation_l1959_195980

theorem solve_quadratic_equation (x : ℝ) :
  (x^2 - 2 * x - 5 = 0) ↔ (x = 1 + Real.sqrt 6 ∨ x = 1 - Real.sqrt 6) := 
sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l1959_195980


namespace NUMINAMATH_GPT_factor_expression_l1959_195968

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1959_195968


namespace NUMINAMATH_GPT_minimum_workers_needed_l1959_195927

-- Definitions
def job_completion_time : ℕ := 45
def days_worked : ℕ := 9
def portion_job_done : ℚ := 1 / 5
def team_size : ℕ := 10
def job_remaining : ℚ := (1 - portion_job_done)
def days_remaining : ℕ := job_completion_time - days_worked
def daily_completion_rate_by_team : ℚ := portion_job_done / days_worked
def daily_completion_rate_per_person : ℚ := daily_completion_rate_by_team / team_size
def required_daily_rate : ℚ := job_remaining / days_remaining

-- Statement to be proven
theorem minimum_workers_needed :
  (required_daily_rate / daily_completion_rate_per_person) = 10 :=
sorry

end NUMINAMATH_GPT_minimum_workers_needed_l1959_195927


namespace NUMINAMATH_GPT_modulus_zero_l1959_195906

/-- Given positive integers k and α such that 10k - α is also a positive integer, 
prove that the remainder when 8^(10k + α) + 6^(10k - α) - 7^(10k - α) - 2^(10k + α) is divided by 11 is 0. -/
theorem modulus_zero {k α : ℕ} (h₁ : 0 < k) (h₂ : 0 < α) (h₃ : 0 < 10 * k - α) :
  (8 ^ (10 * k + α) + 6 ^ (10 * k - α) - 7 ^ (10 * k - α) - 2 ^ (10 * k + α)) % 11 = 0 :=
by
  sorry

end NUMINAMATH_GPT_modulus_zero_l1959_195906


namespace NUMINAMATH_GPT_solve_for_y_l1959_195947

def solution (y : ℝ) : Prop :=
  2 * Real.arctan (1/3) - Real.arctan (1/5) + Real.arctan (1/y) = Real.pi / 4

theorem solve_for_y (y : ℝ) : solution y → y = 31 / 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_y_l1959_195947


namespace NUMINAMATH_GPT_weight_of_B_l1959_195951

-- Definitions for the weights of A, B, and C
variable (A B C : ℝ)

-- Conditions given in the problem
def condition1 := (A + B + C) / 3 = 45
def condition2 := (A + B) / 2 = 40
def condition3 := (B + C) / 2 = 43

-- The theorem to prove that B = 31 under the given conditions
theorem weight_of_B : condition1 A B C → condition2 A B → condition3 B C → B = 31 := by
  intros
  sorry

end NUMINAMATH_GPT_weight_of_B_l1959_195951


namespace NUMINAMATH_GPT_distinct_arrangements_balloon_l1959_195957

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end NUMINAMATH_GPT_distinct_arrangements_balloon_l1959_195957


namespace NUMINAMATH_GPT_largest_angle_of_obtuse_isosceles_triangle_l1959_195970

variables (X Y Z : ℝ)

def is_triangle (X Y Z : ℝ) : Prop := X + Y + Z = 180
def is_isosceles_triangle (X Y : ℝ) : Prop := X = Y
def is_obtuse_triangle (X Y Z : ℝ) : Prop := X > 90 ∨ Y > 90 ∨ Z > 90

theorem largest_angle_of_obtuse_isosceles_triangle
  (X Y Z : ℝ)
  (h1 : is_triangle X Y Z)
  (h2 : is_isosceles_triangle X Y)
  (h3 : X = 30)
  (h4 : is_obtuse_triangle X Y Z) :
  Z = 120 :=
sorry

end NUMINAMATH_GPT_largest_angle_of_obtuse_isosceles_triangle_l1959_195970


namespace NUMINAMATH_GPT_find_X_d_minus_Y_d_l1959_195984

def digits_in_base_d (X Y d : ℕ) : Prop :=
  2 * d * X + X + Y = d^2 + 8 * d + 2 

theorem find_X_d_minus_Y_d (d X Y : ℕ) (h1 : digits_in_base_d X Y d) (h2 : d > 8) : X - Y = d - 8 :=
by 
  sorry

end NUMINAMATH_GPT_find_X_d_minus_Y_d_l1959_195984


namespace NUMINAMATH_GPT_constant_term_binomial_expansion_l1959_195944

theorem constant_term_binomial_expansion (n : ℕ) (hn : n = 6) :
  (2 : ℤ) * (x : ℝ) - (1 : ℤ) / (2 : ℝ) / (x : ℝ) ^ n == -20 := by
  sorry

end NUMINAMATH_GPT_constant_term_binomial_expansion_l1959_195944


namespace NUMINAMATH_GPT_bird_average_l1959_195972

theorem bird_average (a b c : ℤ) (h1 : a = 7) (h2 : b = 11) (h3 : c = 9) :
  (a + b + c) / 3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_bird_average_l1959_195972


namespace NUMINAMATH_GPT_find_f_neg_2017_l1959_195936

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic_function : ∀ x : ℝ, x ≥ 0 → f (x + 2) = f x
axiom log_function : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2

theorem find_f_neg_2017 : f (-2017) = 1 := by
  sorry

end NUMINAMATH_GPT_find_f_neg_2017_l1959_195936


namespace NUMINAMATH_GPT_cylinder_intersection_in_sphere_l1959_195910

theorem cylinder_intersection_in_sphere
  (a b c d e f : ℝ)
  (x y z : ℝ)
  (h1 : (x - a)^2 + (y - b)^2 < 1)
  (h2 : (y - c)^2 + (z - d)^2 < 1)
  (h3 : (z - e)^2 + (x - f)^2 < 1) :
  (x - (a + f) / 2)^2 + (y - (b + c) / 2)^2 + (z - (d + e) / 2)^2 < 3 / 2 := 
sorry

end NUMINAMATH_GPT_cylinder_intersection_in_sphere_l1959_195910


namespace NUMINAMATH_GPT_area_of_rectangle_with_diagonal_length_l1959_195911

variable (x : ℝ)

def rectangle_area_given_diagonal_length (x : ℝ) : Prop :=
  ∃ (w l : ℝ), l = 3 * w ∧ w^2 + l^2 = x^2 ∧ (w * l = (3 / 10) * x^2)

theorem area_of_rectangle_with_diagonal_length (x : ℝ) :
  rectangle_area_given_diagonal_length x :=
sorry

end NUMINAMATH_GPT_area_of_rectangle_with_diagonal_length_l1959_195911


namespace NUMINAMATH_GPT_total_squares_after_removals_l1959_195939

/-- 
Prove that the total number of squares of various sizes on a 5x5 grid,
after removing two 1x1 squares, is 55.
-/
theorem total_squares_after_removals (total_squares_in_5x5_grid: ℕ) (removed_squares: ℕ) : 
  (total_squares_in_5x5_grid = 25 + 16 + 9 + 4 + 1) →
  (removed_squares = 2) →
  (total_squares_in_5x5_grid - removed_squares = 55) :=
sorry

end NUMINAMATH_GPT_total_squares_after_removals_l1959_195939


namespace NUMINAMATH_GPT_arithmetic_sequence_l1959_195932

theorem arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 1 + a 2 + a 3 = 32) 
  (h2 : a 11 + a 12 + a 13 = 118) 
  (arith_seq : ∀ n, a (n + 1) = a n + d) : 
  a 4 + a 10 = 50 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_l1959_195932


namespace NUMINAMATH_GPT_a_eq_b_pow_n_l1959_195912

theorem a_eq_b_pow_n (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → (a - k^n) % (b - k) = 0) : a = b^n :=
sorry

end NUMINAMATH_GPT_a_eq_b_pow_n_l1959_195912


namespace NUMINAMATH_GPT_initial_decaf_percentage_l1959_195996

theorem initial_decaf_percentage 
  (x : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 100) 
  (h3 : (x / 100 * 400) + 60 = 220) :
  x = 40 :=
by sorry

end NUMINAMATH_GPT_initial_decaf_percentage_l1959_195996


namespace NUMINAMATH_GPT_adjusted_smallest_part_proof_l1959_195943

theorem adjusted_smallest_part_proof : 
  ∀ (x : ℝ), 14 * x = 100 → x + 12 = 19 + 1 / 7 := 
by
  sorry

end NUMINAMATH_GPT_adjusted_smallest_part_proof_l1959_195943


namespace NUMINAMATH_GPT_units_digit_of_subtraction_is_seven_l1959_195962

theorem units_digit_of_subtraction_is_seven (a b c: ℕ) (h1: a = c + 3) (h2: b = 2 * c) :
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let result := original_number - reversed_number
  result % 10 = 7 :=
by
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let result := original_number - reversed_number
  sorry

end NUMINAMATH_GPT_units_digit_of_subtraction_is_seven_l1959_195962


namespace NUMINAMATH_GPT_accelerations_l1959_195971

open Real

namespace Problem

variables (m M g : ℝ) (a1 a2 : ℝ)

theorem accelerations (mass_condition : 4 * m + M ≠ 0):
  (a1 = 2 * ((2 * m + M) * g) / (4 * m + M)) ∧
  (a2 = ((2 * m + M) * g) / (4 * m + M)) :=
sorry

end Problem

end NUMINAMATH_GPT_accelerations_l1959_195971


namespace NUMINAMATH_GPT_mixture_kerosene_l1959_195923

theorem mixture_kerosene (x : ℝ) (h₁ : 0.25 * x + 1.2 = 0.27 * (x + 4)) : x = 6 :=
sorry

end NUMINAMATH_GPT_mixture_kerosene_l1959_195923


namespace NUMINAMATH_GPT_subtraction_divisible_l1959_195935

theorem subtraction_divisible (n m d : ℕ) (h1 : n = 13603) (h2 : m = 31) (h3 : d = 13572) : 
  (n - m) % d = 0 := by
  sorry

end NUMINAMATH_GPT_subtraction_divisible_l1959_195935


namespace NUMINAMATH_GPT_second_person_percentage_of_Deshaun_l1959_195914

variable (days : ℕ) (books_read_by_Deshaun : ℕ) (pages_per_book : ℕ) (pages_per_day_by_second_person : ℕ)

theorem second_person_percentage_of_Deshaun :
  days = 80 →
  books_read_by_Deshaun = 60 →
  pages_per_book = 320 →
  pages_per_day_by_second_person = 180 →
  ((pages_per_day_by_second_person * days) / (books_read_by_Deshaun * pages_per_book) * 100) = 75 := 
by
  intros days_eq books_eq pages_eq second_pages_eq
  rw [days_eq, books_eq, pages_eq, second_pages_eq]
  simp
  sorry

end NUMINAMATH_GPT_second_person_percentage_of_Deshaun_l1959_195914


namespace NUMINAMATH_GPT_odd_function_at_zero_l1959_195901

theorem odd_function_at_zero
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  f 0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_at_zero_l1959_195901


namespace NUMINAMATH_GPT_trisect_chord_exists_l1959_195977

noncomputable def distance (O P : Point) : ℝ := sorry
def trisect (P : Point) (A B : Point) : Prop := 2 * (distance A P) = distance P B

-- Main theorem based on the given conditions and conclusions
theorem trisect_chord_exists (O P : Point) (r : ℝ) (hP_in_circle : distance O P < r) :
  (∃ A B : Point, trisect P A B) ↔ 
  (distance O P > r / 3 ∨ distance O P = r / 3) :=
by
  sorry

end NUMINAMATH_GPT_trisect_chord_exists_l1959_195977


namespace NUMINAMATH_GPT_correct_calculation_is_c_l1959_195908

theorem correct_calculation_is_c (a b : ℕ) :
  (2 * a ^ 2 * b) ^ 3 = 8 * a ^ 6 * b ^ 3 := 
sorry

end NUMINAMATH_GPT_correct_calculation_is_c_l1959_195908


namespace NUMINAMATH_GPT_locus_of_points_is_straight_line_l1959_195969

theorem locus_of_points_is_straight_line 
  (a R1 R2 : ℝ) 
  (h_nonzero_a : a ≠ 0)
  (h_positive_R1 : R1 > 0)
  (h_positive_R2 : R2 > 0) :
  ∃ x : ℝ, ∀ (y : ℝ),
  ((x + a)^2 + y^2 - R1^2 = (x - a)^2 + y^2 - R2^2) ↔ 
  x = (R1^2 - R2^2) / (4 * a) :=
by
  sorry

end NUMINAMATH_GPT_locus_of_points_is_straight_line_l1959_195969


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1959_195942

theorem sufficient_but_not_necessary_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x > 0 ∧ y > 0 → (x / y + y / x ≥ 2)) ∧ ¬((x / y + y / x ≥ 2) → (x > 0 ∧ y > 0)) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1959_195942


namespace NUMINAMATH_GPT_rosa_called_last_week_l1959_195949

noncomputable def total_pages_called : ℝ := 18.8
noncomputable def pages_called_this_week : ℝ := 8.6
noncomputable def pages_called_last_week : ℝ := total_pages_called - pages_called_this_week

theorem rosa_called_last_week :
  pages_called_last_week = 10.2 :=
by
  sorry

end NUMINAMATH_GPT_rosa_called_last_week_l1959_195949


namespace NUMINAMATH_GPT_range_of_a_l1959_195978

theorem range_of_a (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_mono : ∀ ⦃a b⦄, 0 ≤ a → a ≤ b → f a ≤ f b)
  (h_cond : ∀ a, f a < f (2 * a - 1) → a > 1) :
  ∀ a, f a < f (2 * a - 1) → 1 < a := 
sorry

end NUMINAMATH_GPT_range_of_a_l1959_195978


namespace NUMINAMATH_GPT_optimal_discount_savings_l1959_195992

theorem optimal_discount_savings : 
  let total_amount := 15000
  let discount1 := 0.30
  let discount2 := 0.15
  let single_discount := 0.40
  let two_successive_discounts := total_amount * (1 - discount1) * (1 - discount2)
  let one_single_discount := total_amount * (1 - single_discount)
  one_single_discount - two_successive_discounts = 75 :=
by
  sorry

end NUMINAMATH_GPT_optimal_discount_savings_l1959_195992


namespace NUMINAMATH_GPT_expand_product_l1959_195919

noncomputable def question_expression (x : ℝ) := -3 * (2 * x + 4) * (x - 7)
noncomputable def correct_answer (x : ℝ) := -6 * x^2 + 30 * x + 84

theorem expand_product (x : ℝ) : question_expression x = correct_answer x := 
by sorry

end NUMINAMATH_GPT_expand_product_l1959_195919


namespace NUMINAMATH_GPT_find_a9_l1959_195997

variable (a : ℕ → ℝ)  -- Define a sequence a_n.

-- Define the conditions for the arithmetic sequence.
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

variables (h_arith_seq : is_arithmetic_sequence a)
          (h_a3 : a 3 = 8)   -- Condition a_3 = 8
          (h_a6 : a 6 = 5)   -- Condition a_6 = 5 

-- State the theorem.
theorem find_a9 : a 9 = 2 := by
  sorry

end NUMINAMATH_GPT_find_a9_l1959_195997


namespace NUMINAMATH_GPT_solution_of_system_l1959_195915

variable (x y : ℝ) 

def equation1 (x y : ℝ) : Prop := 3 * |x| + 5 * y + 9 = 0
def equation2 (x y : ℝ) : Prop := 2 * x - |y| - 7 = 0

theorem solution_of_system : ∃ y : ℝ, equation1 0 y ∧ equation2 0 y := by
  sorry

end NUMINAMATH_GPT_solution_of_system_l1959_195915


namespace NUMINAMATH_GPT_find_p_value_l1959_195955

theorem find_p_value (D E F : ℚ) (α β : ℚ)
  (h₁: D ≠ 0) 
  (h₂: E^2 - 4*D*F ≥ 0) 
  (hαβ: D * (α^2 + β^2) + E * (α + β) + 2*F = 2*D^2 - E^2) :
  ∃ p : ℚ, (p = (2*D*F - E^2 - 2*D^2) / D^2) :=
sorry

end NUMINAMATH_GPT_find_p_value_l1959_195955


namespace NUMINAMATH_GPT_after_2_pow_2009_days_is_monday_l1959_195948

-- Define the current day as Thursday
def today := "Thursday"

-- Define the modulo operation for calculating days of the week
def day_of_week_after (days : ℕ) : ℕ :=
  days % 7

-- Define the exponent in question
def exponent := 2009

-- Since today is Thursday, which we can represent as 4 (considering Sunday as 0, Monday as 1, ..., Saturday as 6)
def today_as_num := 4

-- Calculate the day after 2^2009 days
def future_day := (today_as_num + day_of_week_after (2 ^ exponent)) % 7

-- Prove that the future_day is 1 (Monday)
theorem after_2_pow_2009_days_is_monday : future_day = 1 := by
  sorry

end NUMINAMATH_GPT_after_2_pow_2009_days_is_monday_l1959_195948


namespace NUMINAMATH_GPT_log_conditions_l1959_195975

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem log_conditions (m n : ℝ) (h₁ : log_base m 9 < log_base n 9)
  (h₂ : log_base n 9 < 0) : 0 < m ∧ m < n ∧ n < 1 :=
sorry

end NUMINAMATH_GPT_log_conditions_l1959_195975


namespace NUMINAMATH_GPT_total_cans_in_display_l1959_195958

-- Definitions and conditions
def first_term : ℕ := 30
def second_term : ℕ := 27
def nth_term : ℕ := 3
def common_difference : ℕ := second_term - first_term

-- Statement of the problem
theorem total_cans_in_display : 
  ∃ (n : ℕ), nth_term = first_term + (n - 1) * common_difference ∧
  (2 * 165 = n * (first_term + nth_term)) :=
by
  sorry

end NUMINAMATH_GPT_total_cans_in_display_l1959_195958


namespace NUMINAMATH_GPT_g_value_at_2002_l1959_195922

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Conditions given in the problem
axiom f_one : f 1 = 1
axiom f_inequality_5 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_inequality_1 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

-- Define the function g based on f
def g (x : ℝ) : ℝ := f x + 1 - x

-- The goal is to prove that g 2002 = 1
theorem g_value_at_2002 : g 2002 = 1 :=
sorry

end NUMINAMATH_GPT_g_value_at_2002_l1959_195922


namespace NUMINAMATH_GPT_common_point_graphs_l1959_195913

theorem common_point_graphs 
  (a b c d : ℝ)
  (h1 : ∃ x : ℝ, 2*a + (1 / (x - b)) = 2*c + (1 / (x - d))) :
  ∃ x : ℝ, 2*b + (1 / (x - a)) = 2*d + (1 / (x - c)) :=
by
  sorry

end NUMINAMATH_GPT_common_point_graphs_l1959_195913


namespace NUMINAMATH_GPT_triangle_area_l1959_195959

theorem triangle_area (f : ℝ → ℝ) (x1 x2 yIntercept base height area : ℝ)
  (h1 : ∀ x, f x = (x - 4)^2 * (x + 3))
  (h2 : f 0 = yIntercept)
  (h3 : x1 = -3)
  (h4 : x2 = 4)
  (h5 : base = x2 - x1)
  (h6 : height = yIntercept)
  (h7 : area = 1/2 * base * height) :
  area = 168 := sorry

end NUMINAMATH_GPT_triangle_area_l1959_195959


namespace NUMINAMATH_GPT_smallest_integer_value_of_m_l1959_195921

def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem smallest_integer_value_of_m :
  ∀ m : ℤ, (x^2 + 4 * x - m = 0) ∧ has_two_distinct_real_roots 1 4 (-m : ℝ) → m ≥ -3 :=
by
  intro m h
  sorry

end NUMINAMATH_GPT_smallest_integer_value_of_m_l1959_195921


namespace NUMINAMATH_GPT_smallest_d_l1959_195988

theorem smallest_d (d : ℕ) (h : 3150 * d = k ^ 2) : d = 14 :=
by
  -- assuming the condition: 3150 = 2 * 3 * 5^2 * 7
  have h_factorization : 3150 = 2 * 3 * 5^2 * 7 := by sorry
  -- based on the computation and verification, the smallest d that satisfies the condition is 14
  sorry

end NUMINAMATH_GPT_smallest_d_l1959_195988


namespace NUMINAMATH_GPT_baby_panda_daily_bamboo_intake_l1959_195931

theorem baby_panda_daily_bamboo_intake :
  ∀ (adult_bamboo_per_day baby_bamboo_per_day total_bamboo_per_week : ℕ),
    adult_bamboo_per_day = 138 →
    total_bamboo_per_week = 1316 →
    total_bamboo_per_week = 7 * adult_bamboo_per_day + 7 * baby_bamboo_per_day →
    baby_bamboo_per_day = 50 :=
by
  intros adult_bamboo_per_day baby_bamboo_per_day total_bamboo_per_week h1 h2 h3
  sorry

end NUMINAMATH_GPT_baby_panda_daily_bamboo_intake_l1959_195931


namespace NUMINAMATH_GPT_common_factor_is_n_plus_1_l1959_195981

def polynomial1 (n : ℕ) : ℕ := n^2 - 1
def polynomial2 (n : ℕ) : ℕ := n^2 + n

theorem common_factor_is_n_plus_1 (n : ℕ) : 
  ∃ (d : ℕ), d ∣ polynomial1 n ∧ d ∣ polynomial2 n ∧ d = n + 1 := by
  sorry

end NUMINAMATH_GPT_common_factor_is_n_plus_1_l1959_195981


namespace NUMINAMATH_GPT_bad_iff_prime_l1959_195990

def a_n (n : ℕ) : ℕ := (2 * n)^2 + 1

def is_bad (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a_n n = a^2 + b^2

theorem bad_iff_prime (n : ℕ) : is_bad n ↔ Nat.Prime (a_n n) :=
by
  sorry

end NUMINAMATH_GPT_bad_iff_prime_l1959_195990


namespace NUMINAMATH_GPT_division_and_multiply_l1959_195946

theorem division_and_multiply :
  (-128) / (-16) * 5 = 40 := 
by
  sorry

end NUMINAMATH_GPT_division_and_multiply_l1959_195946


namespace NUMINAMATH_GPT_find_number_l1959_195993

theorem find_number (x : ℤ) (N : ℤ) (h1 : 3 * x = (N - x) + 18) (hx : x = 11) : N = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1959_195993


namespace NUMINAMATH_GPT_Yoongi_stack_taller_than_Taehyung_l1959_195926

theorem Yoongi_stack_taller_than_Taehyung :
  let height_A := 3
  let height_B := 3.5
  let count_A := 16
  let count_B := 14
  let total_height_A := height_A * count_A
  let total_height_B := height_B * count_B
  total_height_B > total_height_A ∧ (total_height_B - total_height_A = 1) :=
by
  sorry

end NUMINAMATH_GPT_Yoongi_stack_taller_than_Taehyung_l1959_195926


namespace NUMINAMATH_GPT_find_cost_price_l1959_195964

variable (CP : ℝ)

def SP1 : ℝ := 0.80 * CP
def SP2 : ℝ := 1.06 * CP

axiom cond1 : SP2 - SP1 = 520

theorem find_cost_price : CP = 2000 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l1959_195964


namespace NUMINAMATH_GPT_smallest_m_l1959_195945

noncomputable def f (x : ℝ) : ℝ := sorry

theorem smallest_m (f : ℝ → ℝ) (x y : ℝ) (hx : 0 ≤ x) (hy : y ≤ 1) (h_eq : f 0 = f 1) 
(h_lt : forall x y : ℝ, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → |f x - f y| < |x - y|): 
|f x - f y| < 1 / 2 := 
sorry

end NUMINAMATH_GPT_smallest_m_l1959_195945


namespace NUMINAMATH_GPT_find_x_range_l1959_195916

-- Define the condition for the expression to be meaningful
def meaningful_expr (x : ℝ) : Prop := x - 3 ≥ 0

-- The range of values for x is equivalent to x being at least 3
theorem find_x_range (x : ℝ) : meaningful_expr x ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_GPT_find_x_range_l1959_195916


namespace NUMINAMATH_GPT_all_terms_perfect_squares_l1959_195983

def seq_x : ℕ → ℕ
| 0       => 1
| 1       => 1
| (n + 2) => 14 * seq_x (n + 1) - seq_x n - 4

theorem all_terms_perfect_squares : ∀ n, ∃ k, seq_x n = k^2 :=
by
  sorry

end NUMINAMATH_GPT_all_terms_perfect_squares_l1959_195983


namespace NUMINAMATH_GPT_reflected_ray_eqn_l1959_195920

theorem reflected_ray_eqn (P : ℝ × ℝ)
  (incident_ray : ∀ x : ℝ, P.2 = 2 * P.1 + 1)
  (reflection_line : P.2 = P.1) :
  P.1 - 2 * P.2 - 1 = 0 :=
sorry

end NUMINAMATH_GPT_reflected_ray_eqn_l1959_195920
