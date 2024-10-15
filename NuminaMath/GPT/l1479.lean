import Mathlib

namespace NUMINAMATH_GPT_rita_daily_minimum_payment_l1479_147969

theorem rita_daily_minimum_payment (total_cost down_payment balance daily_payment : ℝ) 
    (h1 : total_cost = 120)
    (h2 : down_payment = total_cost / 2)
    (h3 : balance = total_cost - down_payment)
    (h4 : daily_payment = balance / 10) : daily_payment = 6 :=
by
  sorry

end NUMINAMATH_GPT_rita_daily_minimum_payment_l1479_147969


namespace NUMINAMATH_GPT_candies_initial_count_l1479_147990

theorem candies_initial_count (x : ℕ) (h : (x - 29) / 13 = 15) : x = 224 :=
sorry

end NUMINAMATH_GPT_candies_initial_count_l1479_147990


namespace NUMINAMATH_GPT_product_of_roots_l1479_147961

variable {k m x1 x2 : ℝ}

theorem product_of_roots (h1 : 4 * x1 ^ 2 - k * x1 - m = 0) (h2 : 4 * x2 ^ 2 - k * x2 - m = 0) (h3 : x1 ≠ x2) :
  x1 * x2 = -m / 4 :=
sorry

end NUMINAMATH_GPT_product_of_roots_l1479_147961


namespace NUMINAMATH_GPT_total_population_of_cities_l1479_147992

theorem total_population_of_cities (n : ℕ) (avg_pop : ℕ) (pn : (n = 20)) (avg_factor: (avg_pop = (4500 + 5000) / 2)) : 
  n * avg_pop = 95000 := 
by 
  sorry

end NUMINAMATH_GPT_total_population_of_cities_l1479_147992


namespace NUMINAMATH_GPT_range_of_z_l1479_147967

theorem range_of_z (α β : ℝ) (z : ℝ) (h1 : -2 < α) (h2 : α ≤ 3) (h3 : 2 < β) (h4 : β ≤ 4) (h5 : z = 2 * α - (1 / 2) * β) :
  -6 < z ∧ z < 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_z_l1479_147967


namespace NUMINAMATH_GPT_candy_bar_cost_correct_l1479_147968

def quarters : ℕ := 4
def dimes : ℕ := 3
def nickel : ℕ := 1
def change_received : ℕ := 4

def total_paid : ℕ :=
  (quarters * 25) + (dimes * 10) + (nickel * 5)

def candy_bar_cost : ℕ :=
  total_paid - change_received

theorem candy_bar_cost_correct : candy_bar_cost = 131 := by
  sorry

end NUMINAMATH_GPT_candy_bar_cost_correct_l1479_147968


namespace NUMINAMATH_GPT_total_apples_collected_l1479_147944

variable (dailyPicks : ℕ) (days : ℕ) (remainingPicks : ℕ)

theorem total_apples_collected (h1 : dailyPicks = 4) (h2 : days = 30) (h3 : remainingPicks = 230) :
  dailyPicks * days + remainingPicks = 350 :=
by
  sorry

end NUMINAMATH_GPT_total_apples_collected_l1479_147944


namespace NUMINAMATH_GPT_find_sum_l1479_147937

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α)

-- Geometric sequence condition
def geometric_seq (a : ℕ → α) (r : α) := ∀ n : ℕ, a (n + 1) = a n * r

theorem find_sum (r : α)
  (h1 : geometric_seq a r)
  (h2 : a 4 + a 7 = 2)
  (h3 : a 5 * a 6 = -8) :
  a 1 + a 10 = -7 := 
sorry

end NUMINAMATH_GPT_find_sum_l1479_147937


namespace NUMINAMATH_GPT_fraction_value_l1479_147907

theorem fraction_value :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1 : ℚ) / (2 - 4 + 6 - 8 + 10 - 12 + 14) = 3/4 :=
sorry

end NUMINAMATH_GPT_fraction_value_l1479_147907


namespace NUMINAMATH_GPT_lillian_candies_l1479_147959

theorem lillian_candies (initial_candies : ℕ) (additional_candies : ℕ) (total_candies : ℕ) :
  initial_candies = 88 → additional_candies = 5 → total_candies = initial_candies + additional_candies → total_candies = 93 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_lillian_candies_l1479_147959


namespace NUMINAMATH_GPT_min_blocks_for_wall_l1479_147914

theorem min_blocks_for_wall (len height : ℕ) (blocks : ℕ → ℕ → ℕ)
  (block_1 : ℕ) (block_2 : ℕ) (block_3 : ℕ) :
  len = 120 → height = 9 →
  block_3 = 1 → block_2 = 2 → block_1 = 3 →
  blocks 5 41 + blocks 4 40 = 365 :=
by
  sorry

end NUMINAMATH_GPT_min_blocks_for_wall_l1479_147914


namespace NUMINAMATH_GPT_fraction_division_correct_l1479_147996

theorem fraction_division_correct :
  (5/6 : ℚ) / (7/9) / (11/13) = 195/154 := 
by {
  sorry
}

end NUMINAMATH_GPT_fraction_division_correct_l1479_147996


namespace NUMINAMATH_GPT_cafeteria_pies_l1479_147901

theorem cafeteria_pies (total_apples handed_out_per_student apples_per_pie : ℕ) (h1 : total_apples = 47) (h2 : handed_out_per_student = 27) (h3 : apples_per_pie = 4) :
  (total_apples - handed_out_per_student) / apples_per_pie = 5 := by
  sorry

end NUMINAMATH_GPT_cafeteria_pies_l1479_147901


namespace NUMINAMATH_GPT_smallest_d_l1479_147900

noncomputable def d := 53361

theorem smallest_d :
  ∃ (p q r : ℕ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧
    10000 * d = (p * q * r) ^ 2 ∧ d = 53361 :=
  by
    sorry

end NUMINAMATH_GPT_smallest_d_l1479_147900


namespace NUMINAMATH_GPT_min_value_of_frac_sum_l1479_147986

theorem min_value_of_frac_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 2) :
  (1 / a + 2 / b) = 9 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_frac_sum_l1479_147986


namespace NUMINAMATH_GPT_new_person_weight_l1479_147951

noncomputable def weight_of_new_person (W : ℝ) : ℝ :=
  W + 61 - 25

theorem new_person_weight {W : ℝ} : 
  ((W + 61 - 25) / 12 = W / 12 + 3) → 
  weight_of_new_person W = 61 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_new_person_weight_l1479_147951


namespace NUMINAMATH_GPT_chocolate_cost_is_correct_l1479_147938

def total_spent : ℕ := 13
def candy_bar_cost : ℕ := 7
def chocolate_cost : ℕ := total_spent - candy_bar_cost

theorem chocolate_cost_is_correct : chocolate_cost = 6 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_cost_is_correct_l1479_147938


namespace NUMINAMATH_GPT_tommy_saw_100_wheels_l1479_147953

-- Define the parameters
def trucks : ℕ := 12
def cars : ℕ := 13
def wheels_per_truck : ℕ := 4
def wheels_per_car : ℕ := 4

-- Define the statement to prove
theorem tommy_saw_100_wheels : (trucks * wheels_per_truck + cars * wheels_per_car) = 100 := by
  sorry 

end NUMINAMATH_GPT_tommy_saw_100_wheels_l1479_147953


namespace NUMINAMATH_GPT_age_difference_l1479_147995

variable (A B C D : ℕ)

theorem age_difference (h₁ : A + B > B + C) (h₂ : C = A - 15) : (A + B) - (B + C) = 15 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l1479_147995


namespace NUMINAMATH_GPT_sum_and_count_even_l1479_147913

-- Sum of integers from a to b (inclusive)
def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

-- Number of even integers from a to b (inclusive)
def count_even_integers (a b : ℕ) : ℕ :=
  ((b - if b % 2 == 0 then 0 else 1) - (a + if a % 2 == 0 then 0 else 1)) / 2 + 1

theorem sum_and_count_even (x y : ℕ) :
  x = sum_of_integers 20 40 →
  y = count_even_integers 20 40 →
  x + y = 641 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_and_count_even_l1479_147913


namespace NUMINAMATH_GPT_min_points_necessary_l1479_147945

noncomputable def min_points_on_circle (circumference : ℝ) (dist1 dist2 : ℝ) : ℕ :=
  1304

theorem min_points_necessary :
  ∀ (circumference : ℝ) (dist1 dist2 : ℝ),
  circumference = 1956 →
  dist1 = 1 →
  dist2 = 2 →
  (min_points_on_circle circumference dist1 dist2) = 1304 :=
sorry

end NUMINAMATH_GPT_min_points_necessary_l1479_147945


namespace NUMINAMATH_GPT_total_porridge_l1479_147912

variable {c1 c2 c3 c4 c5 c6 : ℝ}

theorem total_porridge (h1 : c3 = c1 + c2)
                      (h2 : c4 = c2 + c3)
                      (h3 : c5 = c3 + c4)
                      (h4 : c6 = c4 + c5)
                      (h5 : c5 = 10) :
                      c1 + c2 + c3 + c4 + c5 + c6 = 40 := 
by
  sorry

end NUMINAMATH_GPT_total_porridge_l1479_147912


namespace NUMINAMATH_GPT_base8_1724_to_base10_l1479_147922

/-- Define the base conversion function from base-eight to base-ten -/
def base8_to_base10 (d3 d2 d1 d0 : ℕ) : ℕ :=
  d3 * 8^3 + d2 * 8^2 + d1 * 8^1 + d0 * 8^0

/-- Base-eight representation conditions for the number 1724 -/
def base8_1724_digits := (1, 7, 2, 4)

/-- Prove the base-ten equivalent of the base-eight number 1724 is 980 -/
theorem base8_1724_to_base10 : base8_to_base10 1 7 2 4 = 980 :=
  by
    -- skipping the proof; just state that it is a theorem to be proved.
    sorry

end NUMINAMATH_GPT_base8_1724_to_base10_l1479_147922


namespace NUMINAMATH_GPT_infinite_solutions_of_linear_system_l1479_147999

theorem infinite_solutions_of_linear_system :
  ∀ (x y : ℝ), (2 * x - 3 * y = 5) ∧ (4 * x - 6 * y = 10) → ∃ (k : ℝ), x = (3 * k + 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_infinite_solutions_of_linear_system_l1479_147999


namespace NUMINAMATH_GPT_find_k_l1479_147988

-- Definitions
def a (n : ℕ) : ℤ := 1 + (n - 1) * 2
def S (n : ℕ) : ℤ := n / 2 * (2 * 1 + (n - 1) * 2)

-- Main theorem statement
theorem find_k (k : ℕ) (h : S (k + 2) - S k = 24) : k = 5 :=
by sorry

end NUMINAMATH_GPT_find_k_l1479_147988


namespace NUMINAMATH_GPT_add_neg3_and_2_mul_neg3_and_2_l1479_147949

theorem add_neg3_and_2 : -3 + 2 = -1 := 
by
  sorry

theorem mul_neg3_and_2 : (-3) * 2 = -6 := 
by
  sorry

end NUMINAMATH_GPT_add_neg3_and_2_mul_neg3_and_2_l1479_147949


namespace NUMINAMATH_GPT_angle_BAC_measure_l1479_147987

variable (A B C X Y : Type)
variables (angle_ABC angle_BAC : ℝ)
variables (len_AX len_XY len_YB len_BC : ℝ)

theorem angle_BAC_measure 
  (h1 : AX = XY) 
  (h2 : XY = YB) 
  (h3 : XY = 2 * AX) 
  (h4 : angle_ABC = 150) :
  angle_BAC = 26.25 :=
by
  -- The proof would be required here.
  -- Following the statement as per instructions.
  sorry

end NUMINAMATH_GPT_angle_BAC_measure_l1479_147987


namespace NUMINAMATH_GPT_train_length_is_120_l1479_147956

noncomputable def length_of_train (speed_kmh : ℝ) (time_s : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  let total_distance := speed_ms * time_s
  total_distance - bridge_length_m

theorem train_length_is_120 :
  length_of_train 70 13.884603517432893 150 = 120 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_120_l1479_147956


namespace NUMINAMATH_GPT_largest_xy_l1479_147919

-- Define the problem conditions
def conditions (x y : ℕ) : Prop := 27 * x + 35 * y ≤ 945 ∧ x > 0 ∧ y > 0

-- Define the largest value of xy
def largest_xy_value : ℕ := 234

-- Prove that the largest possible value of xy given conditions is 234
theorem largest_xy (x y : ℕ) (h : conditions x y) : x * y ≤ largest_xy_value :=
sorry

end NUMINAMATH_GPT_largest_xy_l1479_147919


namespace NUMINAMATH_GPT_no_preimage_iff_lt_one_l1479_147927

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem no_preimage_iff_lt_one (k : ℝ) :
  (∀ x : ℝ, f x ≠ k) ↔ k < 1 := 
by
  sorry

end NUMINAMATH_GPT_no_preimage_iff_lt_one_l1479_147927


namespace NUMINAMATH_GPT_find_x_l1479_147947

open Real

noncomputable def satisfies_equation (x : ℝ) : Prop :=
  log (x - 1) / log 3 + log (x^2 - 1) / log (sqrt 3) + log (x - 1) / log (1 / 3) = 3

theorem find_x : ∃ x : ℝ, 1 < x ∧ satisfies_equation x ∧ x = sqrt (1 + 3 * sqrt 3) := by
  sorry

end NUMINAMATH_GPT_find_x_l1479_147947


namespace NUMINAMATH_GPT_sector_area_l1479_147932

theorem sector_area (θ r arc_length : ℝ) (h_arc_length : arc_length = r * θ) (h_values : θ = 2 ∧ arc_length = 2) :
  1 / 2 * r^2 * θ = 1 := by
  sorry

end NUMINAMATH_GPT_sector_area_l1479_147932


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_x_equals_0_l1479_147994

theorem necessary_but_not_sufficient_condition_for_x_equals_0 (x : ℝ) :
  ((2 * x - 1) * x = 0 → x = 0 ∨ x = 1 / 2) ∧ (x = 0 → (2 * x - 1) * x = 0) ∧ ¬((2 * x - 1) * x = 0 → x = 0) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_x_equals_0_l1479_147994


namespace NUMINAMATH_GPT_even_numbers_count_l1479_147906

theorem even_numbers_count (a b : ℕ) (h1 : 150 < a) (h2 : a % 2 = 0) (h3 : b < 350) (h4 : b % 2 = 0) (h5 : 150 < b) (h6 : a < 350) (h7 : 154 ≤ b) (h8 : a ≤ 152) :
  ∃ n : ℕ, ∀ k : ℕ, k = 99 ↔ 2 * k + 150 = b - a + 2 :=
by
  sorry

end NUMINAMATH_GPT_even_numbers_count_l1479_147906


namespace NUMINAMATH_GPT_probability_sum_sixteen_l1479_147934

-- Define the probabilities involved
def probability_of_coin_fifteen := 1 / 2
def probability_of_die_one := 1 / 6

-- Define the combined probability
def combined_probability : ℚ := probability_of_coin_fifteen * probability_of_die_one

theorem probability_sum_sixteen : combined_probability = 1 / 12 := by
  sorry

end NUMINAMATH_GPT_probability_sum_sixteen_l1479_147934


namespace NUMINAMATH_GPT_height_drawn_to_hypotenuse_l1479_147908

-- Definitions for the given problem
variables {A B C D : Type}
variables {area : ℝ}
variables {angle_ratio : ℝ}
variables {h : ℝ}

-- Given conditions
def is_right_triangle (A B C : Type) : Prop := -- definition for the right triangle
sorry

def area_of_triangle (A B C : Type) (area: ℝ) : Prop := 
area = ↑(2 : ℝ) * Real.sqrt 3  -- area given as 2√3 cm²

def angle_bisector_ratios (A B C D : Type) (ratio: ℝ) : Prop :=
ratio = 1 / 2  -- given ratio 1:2

-- Question statement
theorem height_drawn_to_hypotenuse (A B C D : Type) 
  (right_triangle : is_right_triangle A B C)
  (area_cond : area_of_triangle A B C area)
  (angle_ratio_cond : angle_bisector_ratios A B C D angle_ratio):
  h = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_height_drawn_to_hypotenuse_l1479_147908


namespace NUMINAMATH_GPT_negation_proposition_l1479_147983

theorem negation_proposition (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) :=
sorry

end NUMINAMATH_GPT_negation_proposition_l1479_147983


namespace NUMINAMATH_GPT_commutative_matrices_implies_fraction_l1479_147950

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

end NUMINAMATH_GPT_commutative_matrices_implies_fraction_l1479_147950


namespace NUMINAMATH_GPT_polynomial_remainder_l1479_147981
-- Importing the broader library needed

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^5 + 2 * x^2 + 3

-- The statement of the theorem
theorem polynomial_remainder :
  p 2 = 43 :=
sorry

end NUMINAMATH_GPT_polynomial_remainder_l1479_147981


namespace NUMINAMATH_GPT_ab_greater_than_1_l1479_147991

noncomputable def log10_abs (x : ℝ) : ℝ :=
  abs (Real.logb 10 x)

theorem ab_greater_than_1
  {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a < b)
  (hf : log10_abs a < log10_abs b) : a * b > 1 := by
  sorry

end NUMINAMATH_GPT_ab_greater_than_1_l1479_147991


namespace NUMINAMATH_GPT_geometric_sequence_a7_l1479_147935

theorem geometric_sequence_a7
  (a : ℕ → ℤ)
  (is_geom_seq : ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h1 : a 1 = -16)
  (h4 : a 4 = 8) :
  a 7 = -4 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a7_l1479_147935


namespace NUMINAMATH_GPT_missed_the_bus_by_5_minutes_l1479_147982

theorem missed_the_bus_by_5_minutes 
    (usual_time : ℝ)
    (new_time : ℝ)
    (h1 : usual_time = 20)
    (h2 : new_time = usual_time * (5 / 4)) : 
    new_time - usual_time = 5 := 
by
  sorry

end NUMINAMATH_GPT_missed_the_bus_by_5_minutes_l1479_147982


namespace NUMINAMATH_GPT_graphs_differ_l1479_147973

theorem graphs_differ (x : ℝ) :
  (∀ (y : ℝ), y = x + 3 ↔ y ≠ (x^2 - 1) / (x - 1) ∧
              y ≠ (x^2 - 1) / (x - 1) ∧
              ∀ (y : ℝ), y = (x^2 - 1) / (x - 1) ↔ ∀ (z : ℝ), y ≠ x + 3 ∧ y ≠ x + 1) := sorry

end NUMINAMATH_GPT_graphs_differ_l1479_147973


namespace NUMINAMATH_GPT_probability_of_A_given_B_l1479_147941

-- Definitions of events
def tourist_attractions : List String := ["Pengyuan", "Jiuding Mountain", "Garden Expo Park", "Yunlong Lake", "Pan'an Lake"]

-- Probabilities for each scenario
noncomputable def P_AB : ℝ := 8 / 25
noncomputable def P_B : ℝ := 20 / 25
noncomputable def P_A_given_B : ℝ := 2 / 5

-- Proof statement
theorem probability_of_A_given_B : (P_AB / P_B) = P_A_given_B :=
by
  sorry

end NUMINAMATH_GPT_probability_of_A_given_B_l1479_147941


namespace NUMINAMATH_GPT_nominal_rate_of_interest_l1479_147974

noncomputable def nominal_rate (EAR : ℝ) (n : ℕ) : ℝ :=
  2 * (Real.sqrt (1 + EAR) - 1)

theorem nominal_rate_of_interest :
  nominal_rate 0.1025 2 = 0.100476 :=
by sorry

end NUMINAMATH_GPT_nominal_rate_of_interest_l1479_147974


namespace NUMINAMATH_GPT_piglet_straws_l1479_147943

theorem piglet_straws (total_straws : ℕ) (straws_adult_pigs_ratio : ℚ) (straws_piglets_ratio : ℚ) (number_piglets : ℕ) :
  total_straws = 300 →
  straws_adult_pigs_ratio = 3/5 →
  straws_piglets_ratio = 1/3 →
  number_piglets = 20 →
  (total_straws * straws_piglets_ratio) / number_piglets = 5 := 
by
  intros
  sorry

end NUMINAMATH_GPT_piglet_straws_l1479_147943


namespace NUMINAMATH_GPT_find_first_discount_percentage_l1479_147931

def first_discount_percentage 
  (price_initial : ℝ) 
  (price_final : ℝ) 
  (discount_x : ℝ) 
  : Prop := 
  price_initial * (1 - discount_x / 100) * 0.9 * 0.95 = price_final

theorem find_first_discount_percentage :
  first_discount_percentage 9941.52 6800 20.02 :=
by
  sorry

end NUMINAMATH_GPT_find_first_discount_percentage_l1479_147931


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1479_147940

theorem perfect_square_trinomial (a k : ℝ) : (∃ b : ℝ, (a^2 + 2*k*a + 9 = (a + b)^2)) ↔ (k = 3 ∨ k = -3) := 
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1479_147940


namespace NUMINAMATH_GPT_solve_for_t_l1479_147916

variable (A P0 r t : ℝ)

theorem solve_for_t (h : A = P0 * Real.exp (r * t)) : t = (Real.log (A / P0)) / r :=
  by
  sorry

end NUMINAMATH_GPT_solve_for_t_l1479_147916


namespace NUMINAMATH_GPT_find_sticker_price_l1479_147904

-- Defining the conditions:
def sticker_price (x : ℝ) : Prop := 
  let price_A := 0.85 * x - 90
  let price_B := 0.75 * x
  price_A + 15 = price_B

-- Proving the sticker price is $750 given the conditions
theorem find_sticker_price : ∃ x : ℝ, sticker_price x ∧ x = 750 := 
by
  use 750
  simp [sticker_price]
  sorry

end NUMINAMATH_GPT_find_sticker_price_l1479_147904


namespace NUMINAMATH_GPT_gcd_of_72_90_120_l1479_147972

theorem gcd_of_72_90_120 : Nat.gcd (Nat.gcd 72 90) 120 = 6 := 
by 
  have h1 : 72 = 2^3 * 3^2 := by norm_num
  have h2 : 90 = 2 * 3^2 * 5 := by norm_num
  have h3 : 120 = 2^3 * 3 * 5 := by norm_num
  sorry

end NUMINAMATH_GPT_gcd_of_72_90_120_l1479_147972


namespace NUMINAMATH_GPT_customer_total_payment_l1479_147962

structure PaymentData where
  rate : ℕ
  discount1 : ℕ
  lateFee1 : ℕ
  discount2 : ℕ
  lateFee2 : ℕ
  discount3 : ℕ
  lateFee3 : ℕ
  discount4 : ℕ
  lateFee4 : ℕ
  onTime1 : Bool
  onTime2 : Bool
  onTime3 : Bool
  onTime4 : Bool

noncomputable def monthlyPayment (rate discount late_fee : ℕ) (onTime : Bool) : ℕ :=
  if onTime then rate - (rate * discount / 100) else rate + (rate * late_fee / 100)

theorem customer_total_payment (data : PaymentData) : 
  monthlyPayment data.rate data.discount1 data.lateFee1 data.onTime1 +
  monthlyPayment data.rate data.discount2 data.lateFee2 data.onTime2 +
  monthlyPayment data.rate data.discount3 data.lateFee3 data.onTime3 +
  monthlyPayment data.rate data.discount4 data.lateFee4 data.onTime4 = 195 := by
  sorry

end NUMINAMATH_GPT_customer_total_payment_l1479_147962


namespace NUMINAMATH_GPT_lesser_of_two_numbers_l1479_147924

theorem lesser_of_two_numbers (x y : ℝ) (h1 : x + y = 70) (h2 : x * y = 1050) : min x y = 30 :=
sorry

end NUMINAMATH_GPT_lesser_of_two_numbers_l1479_147924


namespace NUMINAMATH_GPT_age_of_youngest_l1479_147942

theorem age_of_youngest
  (y : ℕ)
  (h1 : 4 * 25 = y + (y + 2) + (y + 7) + (y + 11)) : y = 20 :=
by
  sorry

end NUMINAMATH_GPT_age_of_youngest_l1479_147942


namespace NUMINAMATH_GPT_geometric_sequence_constant_l1479_147903

theorem geometric_sequence_constant (a : ℕ → ℝ) (q : ℝ) (h1 : q ≠ 1) (h2 : ∀ n, a (n + 1) = q * a n) (c : ℝ) :
  (∀ n, a (n + 1) + c = q * (a n + c)) → c = 0 := sorry

end NUMINAMATH_GPT_geometric_sequence_constant_l1479_147903


namespace NUMINAMATH_GPT_total_snakes_in_park_l1479_147998

theorem total_snakes_in_park :
  ∀ (pythons boa_constrictors rattlesnakes total_snakes : ℕ),
    boa_constrictors = 40 →
    pythons = 3 * boa_constrictors →
    rattlesnakes = 40 →
    total_snakes = boa_constrictors + pythons + rattlesnakes →
    total_snakes = 200 :=
by
  intros pythons boa_constrictors rattlesnakes total_snakes h1 h2 h3 h4
  rw [h1, h3] at h4
  rw [h2] at h4
  sorry

end NUMINAMATH_GPT_total_snakes_in_park_l1479_147998


namespace NUMINAMATH_GPT_condition_necessary_but_not_sufficient_l1479_147952

variable (a b : ℝ)

theorem condition_necessary_but_not_sufficient (h : a ≠ 1 ∨ b ≠ 2) : (a + b ≠ 3) ∧ ¬(a + b ≠ 3 → a ≠ 1 ∨ b ≠ 2) :=
by
  --Proof will go here
  sorry

end NUMINAMATH_GPT_condition_necessary_but_not_sufficient_l1479_147952


namespace NUMINAMATH_GPT_folded_quadrilateral_has_perpendicular_diagonals_l1479_147966

-- Define a quadrilateral and its properties
structure Quadrilateral :=
(A B C D : ℝ × ℝ)

structure Point :=
(x y : ℝ)

-- Define the diagonals within a quadrilateral
def diagonal1 (q : Quadrilateral) : ℝ × ℝ := (q.A.1 - q.C.1, q.A.2 - q.C.2)
def diagonal2 (q : Quadrilateral) : ℝ × ℝ := (q.B.1 - q.D.1, q.B.2 - q.D.2)

-- Define dot product to check perpendicularity
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Condition when folding quadrilateral vertices to a common point ensures no gaps or overlaps
def folding_condition (q : Quadrilateral) (P : Point) : Prop :=
sorry -- Detailed folding condition logic here if needed

-- The statement we need to prove
theorem folded_quadrilateral_has_perpendicular_diagonals (q : Quadrilateral) (P : Point)
    (h_folding : folding_condition q P)
    : dot_product (diagonal1 q) (diagonal2 q) = 0 :=
sorry

end NUMINAMATH_GPT_folded_quadrilateral_has_perpendicular_diagonals_l1479_147966


namespace NUMINAMATH_GPT_atomic_number_l1479_147985

theorem atomic_number (mass_number : ℕ) (neutrons : ℕ) (protons : ℕ) :
  mass_number = 288 →
  neutrons = 169 →
  (protons = mass_number - neutrons) →
  protons = 119 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_atomic_number_l1479_147985


namespace NUMINAMATH_GPT_algebraic_expression_value_l1479_147978

theorem algebraic_expression_value (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + x*y + y^2 = 0) :
  (x/(x + y))^2005 + (y/(x + y))^2005 = -1 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1479_147978


namespace NUMINAMATH_GPT_range_of_real_number_l1479_147976

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0}
def B (a : ℝ) : Set ℝ := {-1, -3, a}
def complement_A : Set ℝ := {x | x ≥ 0}

theorem range_of_real_number (a : ℝ) (h : (complement_A ∩ (B a)) ≠ ∅) : a ≥ 0 :=
sorry

end NUMINAMATH_GPT_range_of_real_number_l1479_147976


namespace NUMINAMATH_GPT_dishwasher_spending_l1479_147920

theorem dishwasher_spending (E : ℝ) (h1 : E > 0) 
    (rent : ℝ := 0.40 * E)
    (left_over : ℝ := 0.28 * E)
    (spent : ℝ := 0.72 * E)
    (dishwasher : ℝ := spent - rent)
    (difference : ℝ := rent - dishwasher) :
    ((difference / rent) * 100) = 20 := 
by
  sorry

end NUMINAMATH_GPT_dishwasher_spending_l1479_147920


namespace NUMINAMATH_GPT_MaireadRan40Miles_l1479_147964

def MaireadRanMiles (R : ℝ) (W : ℝ) (J : ℝ) : Prop :=
  W = (3 / 5) * R ∧ J = 3 * R ∧ R + W + J = 184

theorem MaireadRan40Miles : ∃ R W J, MaireadRanMiles R W J ∧ R = 40 :=
by sorry

end NUMINAMATH_GPT_MaireadRan40Miles_l1479_147964


namespace NUMINAMATH_GPT_range_of_a_l1479_147946

theorem range_of_a {a : ℝ} : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ - 3 / 5 < a ∧ a ≤ 1 := sorry

end NUMINAMATH_GPT_range_of_a_l1479_147946


namespace NUMINAMATH_GPT_lana_picked_37_roses_l1479_147929

def total_flowers_picked (used : ℕ) (extra : ℕ) := used + extra

def picked_roses (total : ℕ) (tulips : ℕ) := total - tulips

theorem lana_picked_37_roses :
    ∀ (tulips used extra : ℕ), tulips = 36 → used = 70 → extra = 3 → 
    picked_roses (total_flowers_picked used extra) tulips = 37 :=
by
  intros tulips used extra htulips husd hextra
  sorry

end NUMINAMATH_GPT_lana_picked_37_roses_l1479_147929


namespace NUMINAMATH_GPT_find_first_odd_number_l1479_147930

theorem find_first_odd_number (x : ℤ)
  (h : 8 * x = 3 * (x + 4) + 2 * (x + 2) + 5) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_first_odd_number_l1479_147930


namespace NUMINAMATH_GPT_directrix_of_parabola_l1479_147993

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = (x^2 - 4 * x + 3) / 8 → y = -9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1479_147993


namespace NUMINAMATH_GPT_waiter_tip_amount_l1479_147963

theorem waiter_tip_amount (n n_no_tip E : ℕ) (h_n : n = 10) (h_no_tip : n_no_tip = 5) (h_E : E = 15) :
  (E / (n - n_no_tip) = 3) :=
by
  -- Proof goes here (we are only writing the statement with sorry)
  sorry

end NUMINAMATH_GPT_waiter_tip_amount_l1479_147963


namespace NUMINAMATH_GPT_sine_cosine_fraction_l1479_147905

theorem sine_cosine_fraction (θ : ℝ) (h : Real.tan θ = 2) : 
    (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_sine_cosine_fraction_l1479_147905


namespace NUMINAMATH_GPT_max_min_values_l1479_147926

noncomputable def max_value (x y z w : ℝ) : ℝ :=
  if x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2 then
    max (x + y + z + w) 3
  else
    0

noncomputable def min_value (x y z w : ℝ) : ℝ :=
  if x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2 then
    min (x + y + z + w) (-2 + 5 / 2 * Real.sqrt 2)
  else
    0

theorem max_min_values (x y z w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z) (h_nonneg_w : 0 ≤ w)
  (h_eqn : x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2) :
  (x + y + z + w ≤ 3) ∧ (x + y + z + w ≥ -2 + 5 / 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_max_min_values_l1479_147926


namespace NUMINAMATH_GPT_work_rate_proof_l1479_147997

theorem work_rate_proof (A B C : ℝ) (h1 : A + B = 1 / 15) (h2 : C = 1 / 60) : 
  1 / (A + B + C) = 12 :=
by
  sorry

end NUMINAMATH_GPT_work_rate_proof_l1479_147997


namespace NUMINAMATH_GPT_units_digit_F500_is_7_l1479_147957

def F (n : ℕ) : ℕ := 2 ^ (2 ^ (2 * n)) + 1

theorem units_digit_F500_is_7 : (F 500) % 10 = 7 := 
  sorry

end NUMINAMATH_GPT_units_digit_F500_is_7_l1479_147957


namespace NUMINAMATH_GPT_rectangle_area_l1479_147954

theorem rectangle_area (w l: ℝ) (h1: l = 2 * w) (h2: 2 * l + 2 * w = 4) : l * w = 8 / 9 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1479_147954


namespace NUMINAMATH_GPT_sampling_interval_l1479_147960

theorem sampling_interval (total_students sample_size k : ℕ) (h1 : total_students = 1200) (h2 : sample_size = 40) (h3 : k = total_students / sample_size) : k = 30 :=
by
  sorry

end NUMINAMATH_GPT_sampling_interval_l1479_147960


namespace NUMINAMATH_GPT_selena_trip_length_l1479_147975

variable (y : ℚ)

def selena_trip (y : ℚ) : Prop :=
  y / 4 + 16 + y / 6 = y

theorem selena_trip_length : selena_trip y → y = 192 / 7 :=
by
  sorry

end NUMINAMATH_GPT_selena_trip_length_l1479_147975


namespace NUMINAMATH_GPT_jessica_minimal_withdrawal_l1479_147979

theorem jessica_minimal_withdrawal 
  (initial_withdrawal : ℝ)
  (initial_fraction : ℝ)
  (minimum_balance : ℝ)
  (deposit_fraction : ℝ)
  (after_withdrawal_balance : ℝ)
  (deposit_amount : ℝ)
  (current_balance : ℝ) :
  initial_withdrawal = 400 →
  initial_fraction = 2/5 →
  minimum_balance = 300 →
  deposit_fraction = 1/4 →
  after_withdrawal_balance = 1000 - initial_withdrawal →
  deposit_amount = deposit_fraction * after_withdrawal_balance →
  current_balance = after_withdrawal_balance + deposit_amount →
  current_balance - minimum_balance ≥ 0 →
  0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_jessica_minimal_withdrawal_l1479_147979


namespace NUMINAMATH_GPT_cost_of_one_book_l1479_147939

theorem cost_of_one_book (s b c : ℕ) (h1 : s > 18) (h2 : b > 1) (h3 : c > b) (h4 : s * b * c = 3203) (h5 : s ≤ 36) : c = 11 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_book_l1479_147939


namespace NUMINAMATH_GPT_calc_f_g_h_2_l1479_147984

def f (x : ℕ) : ℕ := x + 5
def g (x : ℕ) : ℕ := x^2 - 8
def h (x : ℕ) : ℕ := 2 * x + 1

theorem calc_f_g_h_2 : f (g (h 2)) = 22 := by
  sorry

end NUMINAMATH_GPT_calc_f_g_h_2_l1479_147984


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sum_l1479_147989

theorem min_value_of_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) :
  ∃ z, (z = 3 + 2 * Real.sqrt 2) ∧ (∀ z', (z' = 1 / x + 1 / y) → z ≤ z') :=
sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_sum_l1479_147989


namespace NUMINAMATH_GPT_no_intersection_pair_C_l1479_147909

theorem no_intersection_pair_C :
  let y1 := fun x : ℝ => x
  let y2 := fun x : ℝ => x - 3
  ∀ x : ℝ, y1 x ≠ y2 x :=
by
  sorry

end NUMINAMATH_GPT_no_intersection_pair_C_l1479_147909


namespace NUMINAMATH_GPT_profit_percentage_is_22_percent_l1479_147911

-- Define the given conditions
def scooter_cost (C : ℝ) := C
def repair_cost (C : ℝ) := 0.10 * C
def repair_cost_value := 500
def profit := 1100

-- Let's state the main theorem
theorem profit_percentage_is_22_percent (C : ℝ) 
  (h1 : repair_cost C = repair_cost_value)
  (h2 : profit = 1100) : 
  (profit / C) * 100 = 22 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_22_percent_l1479_147911


namespace NUMINAMATH_GPT_abc_inequality_l1479_147971

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (a / (1 + a * b))^2 + (b / (1 + b * c))^2 + (c / (1 + c * a))^2 ≥ 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_abc_inequality_l1479_147971


namespace NUMINAMATH_GPT_minValue_equality_l1479_147918

noncomputable def minValue (a b c : ℝ) : ℝ :=
  (a + 3 * b) * (b + 3 * c) * (a * c + 3)

theorem minValue_equality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 8) :
  minValue a b c = 48 :=
sorry

end NUMINAMATH_GPT_minValue_equality_l1479_147918


namespace NUMINAMATH_GPT_problem_1_problem_2_l1479_147958

noncomputable def problem_1_solution : Set ℝ := {6, -2}
noncomputable def problem_2_solution : Set ℝ := {2 + Real.sqrt 7, 2 - Real.sqrt 7}

theorem problem_1 :
  {x : ℝ | x^2 - 4 * x - 12 = 0} = problem_1_solution :=
by
  sorry

theorem problem_2 :
  {x : ℝ | x^2 - 4 * x - 3 = 0} = problem_2_solution :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1479_147958


namespace NUMINAMATH_GPT_sum_of_squared_residuals_l1479_147925

theorem sum_of_squared_residuals (S : ℝ) (r : ℝ) (hS : S = 100) (hr : r = 0.818) : 
    S * (1 - r^2) = 33.0876 :=
by
  rw [hS, hr]
  sorry

end NUMINAMATH_GPT_sum_of_squared_residuals_l1479_147925


namespace NUMINAMATH_GPT_true_q_if_not_p_and_p_or_q_l1479_147933

variables {p q : Prop}

theorem true_q_if_not_p_and_p_or_q (h1 : ¬p) (h2 : p ∨ q) : q :=
by 
  sorry

end NUMINAMATH_GPT_true_q_if_not_p_and_p_or_q_l1479_147933


namespace NUMINAMATH_GPT_circumscribed_circle_radius_l1479_147917

noncomputable def radius_of_circumscribed_circle (b c : ℝ) (A : ℝ) : ℝ :=
  let a := Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A)
  let R := a / (2 * Real.sin A)
  R

theorem circumscribed_circle_radius (b c : ℝ) (A : ℝ) (hb : b = 4) (hc : c = 2) (hA : A = Real.pi / 3) :
  radius_of_circumscribed_circle b c A = 2 := by
  sorry

end NUMINAMATH_GPT_circumscribed_circle_radius_l1479_147917


namespace NUMINAMATH_GPT_molecular_weight_H2O_correct_l1479_147923

-- Define the atomic weights of hydrogen and oxygen, and the molecular weight of H2O
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight calculation of H2O
def molecular_weight_H2O : ℝ := (2 * atomic_weight_H) + atomic_weight_O

-- Theorem to state the molecular weight of H2O is approximately 18.016 g/mol
theorem molecular_weight_H2O_correct : molecular_weight_H2O = 18.016 :=
by
  -- Putting the value and calculation
  sorry

end NUMINAMATH_GPT_molecular_weight_H2O_correct_l1479_147923


namespace NUMINAMATH_GPT_max_perimeter_right_triangle_l1479_147902

theorem max_perimeter_right_triangle (a b : ℝ) (h₁ : a^2 + b^2 = 25) :
  (a + b + 5) ≤ 5 + 5 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_max_perimeter_right_triangle_l1479_147902


namespace NUMINAMATH_GPT_points_on_line_initial_l1479_147948

theorem points_on_line_initial (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_initial_l1479_147948


namespace NUMINAMATH_GPT_polynomial_consecutive_integers_l1479_147910

theorem polynomial_consecutive_integers (a : ℤ) (c : ℤ) (P : ℤ → ℤ)
  (hP : ∀ x : ℤ, P x = 2 * x ^ 3 - 30 * x ^ 2 + c * x)
  (h_consecutive : ∃ a : ℤ, P (a - 1) + 1 = P a ∧ P a = P (a + 1) - 1) :
  a = 5 ∧ c = 149 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_consecutive_integers_l1479_147910


namespace NUMINAMATH_GPT_somu_present_age_l1479_147936

theorem somu_present_age (S F : ℕ) (h1 : S = (1 / 3) * F)
    (h2 : S - 5 = (1 / 5) * (F - 5)) : S = 10 := by
  sorry

end NUMINAMATH_GPT_somu_present_age_l1479_147936


namespace NUMINAMATH_GPT_caleb_spent_more_on_ice_cream_l1479_147970

theorem caleb_spent_more_on_ice_cream :
  ∀ (number_of_ic_cartons number_of_fy_cartons : ℕ)
    (cost_per_ic_carton cost_per_fy_carton : ℝ)
    (discount_rate sales_tax_rate : ℝ),
    number_of_ic_cartons = 10 →
    number_of_fy_cartons = 4 →
    cost_per_ic_carton = 4 →
    cost_per_fy_carton = 1 →
    discount_rate = 0.15 →
    sales_tax_rate = 0.05 →
    (number_of_ic_cartons * cost_per_ic_carton * (1 - discount_rate) + 
     (number_of_ic_cartons * cost_per_ic_carton * (1 - discount_rate) + 
      number_of_fy_cartons * cost_per_fy_carton) * sales_tax_rate) -
    (number_of_fy_cartons * cost_per_fy_carton) = 30 :=
by
  intros number_of_ic_cartons number_of_fy_cartons cost_per_ic_carton cost_per_fy_carton discount_rate sales_tax_rate
  sorry

end NUMINAMATH_GPT_caleb_spent_more_on_ice_cream_l1479_147970


namespace NUMINAMATH_GPT_eval_x_plus_one_eq_4_l1479_147980

theorem eval_x_plus_one_eq_4 (x : ℕ) (h : x = 3) : x + 1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_eval_x_plus_one_eq_4_l1479_147980


namespace NUMINAMATH_GPT_ball_reaches_height_l1479_147977

theorem ball_reaches_height (h₀ : ℝ) (ratio : ℝ) (target_height : ℝ) (bounces : ℕ) 
  (initial_height : h₀ = 16) 
  (bounce_ratio : ratio = 1/3) 
  (target : target_height = 2) 
  (bounce_count : bounces = 7) :
  h₀ * (ratio ^ bounces) < target_height := 
sorry

end NUMINAMATH_GPT_ball_reaches_height_l1479_147977


namespace NUMINAMATH_GPT_real_roots_of_quad_eq_l1479_147955

theorem real_roots_of_quad_eq (p q a : ℝ) (h : p^2 - 4 * q > 0) : 
  (2 * a - p)^2 + 3 * (p^2 - 4 * q) > 0 := 
by
  sorry

end NUMINAMATH_GPT_real_roots_of_quad_eq_l1479_147955


namespace NUMINAMATH_GPT_panda_bamboo_consumption_l1479_147965

theorem panda_bamboo_consumption (x : ℝ) (h : 0.40 * x = 16) : x = 40 :=
  sorry

end NUMINAMATH_GPT_panda_bamboo_consumption_l1479_147965


namespace NUMINAMATH_GPT_eraser_cost_l1479_147915

variable (P E : ℝ)
variable (h1 : E = P / 2)
variable (h2 : 20 * P = 80)

theorem eraser_cost : E = 2 := by 
  sorry

end NUMINAMATH_GPT_eraser_cost_l1479_147915


namespace NUMINAMATH_GPT_computer_price_problem_l1479_147928

theorem computer_price_problem (x : ℝ) (h : x + 0.30 * x = 351) : x + 351 = 621 :=
by
  sorry

end NUMINAMATH_GPT_computer_price_problem_l1479_147928


namespace NUMINAMATH_GPT_total_food_items_in_one_day_l1479_147921

-- Define the food consumption for each individual
def JorgeCroissants := 7
def JorgeCakes := 18
def JorgePizzas := 30

def GiulianaCroissants := 5
def GiulianaCakes := 14
def GiulianaPizzas := 25

def MatteoCroissants := 6
def MatteoCakes := 16
def MatteoPizzas := 28

-- Define the total number of each food type consumed
def totalCroissants := JorgeCroissants + GiulianaCroissants + MatteoCroissants
def totalCakes := JorgeCakes + GiulianaCakes + MatteoCakes
def totalPizzas := JorgePizzas + GiulianaPizzas + MatteoPizzas

-- The theorem statement
theorem total_food_items_in_one_day : 
  totalCroissants + totalCakes + totalPizzas = 149 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_food_items_in_one_day_l1479_147921
