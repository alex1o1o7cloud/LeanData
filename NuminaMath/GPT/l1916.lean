import Mathlib

namespace NUMINAMATH_GPT_remainder_n_squared_l1916_191685

theorem remainder_n_squared (n : ℤ) (h : n % 5 = 3) : (n^2) % 5 = 4 := 
    sorry

end NUMINAMATH_GPT_remainder_n_squared_l1916_191685


namespace NUMINAMATH_GPT_arrangement_count1_arrangement_count2_arrangement_count3_arrangement_count4_l1916_191638

-- Define the entities in the problem
inductive Participant
| Teacher
| Boy (id : Nat)
| Girl (id : Nat)

-- Define the conditions as properties or predicates
def girlsNextToEachOther (arrangement : List Participant) : Prop :=
  -- assuming the arrangement is a list of Participant
  sorry -- insert the actual condition as needed

def boysNotNextToEachOther (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

def boysInDecreasingOrder (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

def teacherNotInMiddle (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

def girlsNotAtEnds (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

-- Problem 1: Two girls must stand next to each other
theorem arrangement_count1 : ∃ arrangements, 1440 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, girlsNextToEachOther a := sorry

-- Problem 2: Boys must not stand next to each other
theorem arrangement_count2 : ∃ arrangements, 144 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, boysNotNextToEachOther a := sorry

-- Problem 3: Boys must stand in decreasing order of height
theorem arrangement_count3 : ∃ arrangements, 210 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, boysInDecreasingOrder a := sorry

-- Problem 4: Teacher not in middle, girls not at the ends
theorem arrangement_count4 : ∃ arrangements, 2112 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, teacherNotInMiddle a ∧ girlsNotAtEnds a := sorry

end NUMINAMATH_GPT_arrangement_count1_arrangement_count2_arrangement_count3_arrangement_count4_l1916_191638


namespace NUMINAMATH_GPT_pathway_width_l1916_191621

theorem pathway_width {r1 r2 : ℝ} 
  (h1 : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi)
  (h2 : r1 - r2 = 10) :
  r1 - r2 + 4 = 14 := 
by 
  sorry

end NUMINAMATH_GPT_pathway_width_l1916_191621


namespace NUMINAMATH_GPT_tree_initial_height_example_l1916_191602

-- The height of the tree at the time Tony planted it
def initial_tree_height (growth_rate final_height years : ℕ) : ℕ :=
  final_height - (growth_rate * years)

theorem tree_initial_height_example :
  initial_tree_height 5 29 5 = 4 :=
by
  -- This is where the proof would go, we use 'sorry' to indicate it's omitted.
  sorry

end NUMINAMATH_GPT_tree_initial_height_example_l1916_191602


namespace NUMINAMATH_GPT_janice_bought_30_fifty_cent_items_l1916_191609

theorem janice_bought_30_fifty_cent_items (x y z : ℕ) (h1 : x + y + z = 40) (h2 : 50 * x + 150 * y + 300 * z = 4500) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_janice_bought_30_fifty_cent_items_l1916_191609


namespace NUMINAMATH_GPT_number_of_participants_with_5_points_l1916_191639

-- Definitions for conditions
def num_participants : ℕ := 254

def points_for_victory : ℕ := 1

def additional_point_condition (winner_points loser_points : ℕ) : ℕ :=
  if winner_points < loser_points then 1 else 0

def points_for_loss : ℕ := 0

-- Theorem statement
theorem number_of_participants_with_5_points :
  ∃ num_students_with_5_points : ℕ, num_students_with_5_points = 56 := 
sorry

end NUMINAMATH_GPT_number_of_participants_with_5_points_l1916_191639


namespace NUMINAMATH_GPT_hyperbola_eq_l1916_191668

theorem hyperbola_eq (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : -b / a = -1/2) (h4 : a^2 + b^2 = 5^2) :
  ∃ (a b : ℝ), (a = 2 * Real.sqrt 5 ∧ b = Real.sqrt 5 ∧
  (∀ x y : ℝ, (x^2 / 20 - y^2 / 5 = 1) ↔ (x, y) ∈ {p : ℝ × ℝ | (x^2 / a^2 - y^2 / b^2 = 1)})) := sorry

end NUMINAMATH_GPT_hyperbola_eq_l1916_191668


namespace NUMINAMATH_GPT_determinant_of_matrix_A_l1916_191663

noncomputable def matrix_A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, -1, 4], ![3, x, -2], ![1, -3, 0]]

theorem determinant_of_matrix_A (x : ℝ) :
  Matrix.det (matrix_A x) = -46 - 4 * x :=
by
  sorry

end NUMINAMATH_GPT_determinant_of_matrix_A_l1916_191663


namespace NUMINAMATH_GPT_fraction_furniture_spent_l1916_191689

theorem fraction_furniture_spent (S T : ℕ) (hS : S = 600) (hT : T = 300) : (S - T) / S = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_furniture_spent_l1916_191689


namespace NUMINAMATH_GPT_interest_earned_is_correct_l1916_191622

-- Define the principal amount, interest rate, and duration
def principal : ℝ := 2000
def rate : ℝ := 0.02
def duration : ℕ := 3

-- The compound interest formula to calculate the future value
def future_value (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n

-- Calculate the interest earned
def interest (P : ℝ) (A : ℝ) : ℝ := A - P

-- Theorem statement: The interest Bart earns after 3 years is 122 dollars
theorem interest_earned_is_correct : interest principal (future_value principal rate duration) = 122 :=
by
  sorry

end NUMINAMATH_GPT_interest_earned_is_correct_l1916_191622


namespace NUMINAMATH_GPT_area_of_region_l1916_191672

theorem area_of_region : 
  (∀ x y : ℝ, x^2 + y^2 - 8*x + 6*y = 0 → 
     let a := (x - 4)^2 + (y + 3)^2 
     (a = 25) ∧ ∃ r : ℝ, r = 5 ∧ (π * r^2 = 25 * π)) := 
sorry

end NUMINAMATH_GPT_area_of_region_l1916_191672


namespace NUMINAMATH_GPT_student_ticket_count_l1916_191641

theorem student_ticket_count (S N : ℕ) (h1 : S + N = 821) (h2 : 2 * S + 3 * N = 1933) : S = 530 :=
sorry

end NUMINAMATH_GPT_student_ticket_count_l1916_191641


namespace NUMINAMATH_GPT_probability_four_red_four_blue_l1916_191696

noncomputable def urn_probability : ℚ :=
  let initial_red := 2
  let initial_blue := 1
  let operations := 5
  let final_red := 4
  let final_blue := 4
  -- calculate the probability using given conditions, this result is directly derived as 2/7
  2 / 7

theorem probability_four_red_four_blue :
  urn_probability = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_four_red_four_blue_l1916_191696


namespace NUMINAMATH_GPT_value_of_expression_l1916_191619

variables (a b c d m : ℝ)

theorem value_of_expression (h1: a + b = 0) (h2: c * d = 1) (h3: |m| = 3) :
  (a + b) / m + m^2 - c * d = 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1916_191619


namespace NUMINAMATH_GPT_percentage_B_of_C_l1916_191653

variable (A B C : ℝ)

theorem percentage_B_of_C (h1 : A = 0.08 * C) (h2 : A = 0.5 * B) : B = 0.16 * C :=
by
  sorry

end NUMINAMATH_GPT_percentage_B_of_C_l1916_191653


namespace NUMINAMATH_GPT_find_sides_of_isosceles_triangle_l1916_191687

noncomputable def isosceles_triangle_sides (b a : ℝ) : Prop :=
  ∃ (AI IL₁ : ℝ), AI = 5 ∧ IL₁ = 3 ∧
  b = 10 ∧ a = 12 ∧
  a = (6 / 5) * b ∧
  (b^2 = 8^2 + (3/5 * b)^2)

-- Proof problem statement
theorem find_sides_of_isosceles_triangle :
  ∀ (b a : ℝ), isosceles_triangle_sides b a → b = 10 ∧ a = 12 :=
by
  intros b a h
  sorry

end NUMINAMATH_GPT_find_sides_of_isosceles_triangle_l1916_191687


namespace NUMINAMATH_GPT_table_arrangement_division_l1916_191654

theorem table_arrangement_division (total_tables : ℕ) (rows : ℕ) (tables_per_row : ℕ) (tables_left_over : ℕ)
    (h1 : total_tables = 74) (h2 : rows = 8) (h3 : tables_per_row = total_tables / rows) (h4 : tables_left_over = total_tables % rows) :
    tables_per_row = 9 ∧ tables_left_over = 2 := by
  sorry

end NUMINAMATH_GPT_table_arrangement_division_l1916_191654


namespace NUMINAMATH_GPT_investment_time_l1916_191600

variable (P : ℝ) (R : ℝ) (SI : ℝ)

theorem investment_time (hP : P = 800) (hR : R = 0.04) (hSI : SI = 160) :
  SI / (P * R) = 5 := by
  sorry

end NUMINAMATH_GPT_investment_time_l1916_191600


namespace NUMINAMATH_GPT_greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l1916_191627

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l1916_191627


namespace NUMINAMATH_GPT_mean_median_modes_l1916_191626

theorem mean_median_modes (d μ M : ℝ)
  (dataset : Multiset ℕ)
  (h_dataset : dataset = Multiset.replicate 12 1 + Multiset.replicate 12 2 + Multiset.replicate 12 3 +
                         Multiset.replicate 12 4 + Multiset.replicate 12 5 + Multiset.replicate 12 6 +
                         Multiset.replicate 12 7 + Multiset.replicate 12 8 + Multiset.replicate 12 9 +
                         Multiset.replicate 12 10 + Multiset.replicate 12 11 + Multiset.replicate 12 12 +
                         Multiset.replicate 12 13 + Multiset.replicate 12 14 + Multiset.replicate 12 15 +
                         Multiset.replicate 12 16 + Multiset.replicate 12 17 + Multiset.replicate 12 18 +
                         Multiset.replicate 12 19 + Multiset.replicate 12 20 + Multiset.replicate 12 21 +
                         Multiset.replicate 12 22 + Multiset.replicate 12 23 + Multiset.replicate 12 24 +
                         Multiset.replicate 12 25 + Multiset.replicate 12 26 + Multiset.replicate 12 27 +
                         Multiset.replicate 12 28 + Multiset.replicate 12 29 + Multiset.replicate 12 30 +
                         Multiset.replicate 7 31)
  (h_M : M = 16)
  (h_μ : μ = 5797 / 366)
  (h_d : d = 15.5) :
  d < μ ∧ μ < M :=
sorry

end NUMINAMATH_GPT_mean_median_modes_l1916_191626


namespace NUMINAMATH_GPT_factorial_binomial_mod_l1916_191673

theorem factorial_binomial_mod (p : ℕ) (hp : Nat.Prime p) : 
  ((Nat.factorial (2 * p)) / (Nat.factorial p * Nat.factorial p)) - 2 ≡ 0 [MOD p] :=
by
  sorry

end NUMINAMATH_GPT_factorial_binomial_mod_l1916_191673


namespace NUMINAMATH_GPT_solution_set_inequality_l1916_191623

theorem solution_set_inequality (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ |2 * x - a| + a ≤ 6) → a = 2 :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1916_191623


namespace NUMINAMATH_GPT_laps_remaining_eq_five_l1916_191670

variable (total_distance : ℕ)
variable (distance_per_lap : ℕ)
variable (laps_already_run : ℕ)

theorem laps_remaining_eq_five 
  (h1 : total_distance = 99) 
  (h2 : distance_per_lap = 9) 
  (h3 : laps_already_run = 6) : 
  (total_distance / distance_per_lap - laps_already_run = 5) :=
by 
  sorry

end NUMINAMATH_GPT_laps_remaining_eq_five_l1916_191670


namespace NUMINAMATH_GPT_vacuum_pump_operations_l1916_191662

theorem vacuum_pump_operations (n : ℕ) (h : n ≥ 10) : 
  ∀ a : ℝ, 
  a > 0 → 
  (0.5 ^ n) * a < 0.001 * a :=
by
  intros a h_a
  sorry

end NUMINAMATH_GPT_vacuum_pump_operations_l1916_191662


namespace NUMINAMATH_GPT_can_capacity_is_30_l1916_191624

noncomputable def capacity_of_can (x: ℝ) : ℝ :=
  7 * x + 10

theorem can_capacity_is_30 :
  ∃ (x: ℝ), (4 * x + 10) / (3 * x) = 5 / 2 ∧ capacity_of_can x = 30 :=
by
  sorry

end NUMINAMATH_GPT_can_capacity_is_30_l1916_191624


namespace NUMINAMATH_GPT_yuan_representation_l1916_191671

-- Define the essential conditions and numeric values
def receiving (amount : Int) : Int := amount
def spending (amount : Int) : Int := -amount

-- The main theorem statement
theorem yuan_representation :
  receiving 80 = 80 ∧ spending 50 = -50 → receiving (-50) = spending 50 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_yuan_representation_l1916_191671


namespace NUMINAMATH_GPT_determine_m_l1916_191646

theorem determine_m (x y m : ℝ) :
  (3 * x - y = 4 * m + 1) ∧ (x + y = 2 * m - 5) ∧ (x - y = 4) → m = 1 :=
by sorry

end NUMINAMATH_GPT_determine_m_l1916_191646


namespace NUMINAMATH_GPT_squares_in_region_l1916_191637

theorem squares_in_region :
  let bounded_region (x y : ℤ) := y ≤ 2 * x ∧ y ≥ -1 ∧ x ≤ 6
  ∃ n : ℕ, ∀ (a b : ℤ), bounded_region a b → n = 118
:= 
  sorry

end NUMINAMATH_GPT_squares_in_region_l1916_191637


namespace NUMINAMATH_GPT_cistern_water_depth_l1916_191614

theorem cistern_water_depth 
  (l w a : ℝ)
  (hl : l = 8)
  (hw : w = 6)
  (ha : a = 83) :
  ∃ d : ℝ, 48 + 28 * d = 83 :=
by
  use 1.25
  sorry

end NUMINAMATH_GPT_cistern_water_depth_l1916_191614


namespace NUMINAMATH_GPT_sum_of_fifth_terms_arithmetic_sequences_l1916_191649

theorem sum_of_fifth_terms_arithmetic_sequences (a b : ℕ → ℝ) (d₁ d₂ : ℝ) 
  (h₁ : ∀ n, a (n + 1) = a n + d₁)
  (h₂ : ∀ n, b (n + 1) = b n + d₂)
  (h₃ : a 1 + b 1 = 7)
  (h₄ : a 3 + b 3 = 21) :
  a 5 + b 5 = 35 :=
sorry

end NUMINAMATH_GPT_sum_of_fifth_terms_arithmetic_sequences_l1916_191649


namespace NUMINAMATH_GPT_transfer_balls_l1916_191644

theorem transfer_balls (X Y q p b : ℕ) (h : p + b = q) :
  b = q - p :=
by
  sorry

end NUMINAMATH_GPT_transfer_balls_l1916_191644


namespace NUMINAMATH_GPT_purchase_options_l1916_191693

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end NUMINAMATH_GPT_purchase_options_l1916_191693


namespace NUMINAMATH_GPT_cd_percentage_cheaper_l1916_191625

theorem cd_percentage_cheaper (cost_cd cost_book cost_album difference percentage : ℝ) 
  (h1 : cost_book = cost_cd + 4)
  (h2 : cost_book = 18)
  (h3 : cost_album = 20)
  (h4 : difference = cost_album - cost_cd)
  (h5 : percentage = (difference / cost_album) * 100) : 
  percentage = 30 :=
sorry

end NUMINAMATH_GPT_cd_percentage_cheaper_l1916_191625


namespace NUMINAMATH_GPT_volume_of_water_displaced_l1916_191631

theorem volume_of_water_displaced (r h s : ℝ) (V : ℝ) 
  (r_eq : r = 5) (h_eq : h = 12) (s_eq : s = 6) :
  V = s^3 :=
by
  have cube_volume : V = s^3 := by sorry
  show V = s^3
  exact cube_volume

end NUMINAMATH_GPT_volume_of_water_displaced_l1916_191631


namespace NUMINAMATH_GPT_total_length_of_XYZ_l1916_191613

noncomputable def length_XYZ : ℝ :=
  let length_X := 2 + 2 + 2 * Real.sqrt 2
  let length_Y := 3 + 2 * Real.sqrt 2
  let length_Z := 3 + 3 + Real.sqrt 10
  length_X + length_Y + length_Z

theorem total_length_of_XYZ :
  length_XYZ = 13 + 4 * Real.sqrt 2 + Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_total_length_of_XYZ_l1916_191613


namespace NUMINAMATH_GPT_problem_l1916_191660

def g (x : ℤ) : ℤ := 3 * x^2 - x + 4

theorem problem : g (g 3) = 2328 := by
  sorry

end NUMINAMATH_GPT_problem_l1916_191660


namespace NUMINAMATH_GPT_smallest_value_is_A_l1916_191648

def A : ℤ := -(-3 - 2)^2
def B : ℤ := (-3) * (-2)
def C : ℚ := ((-3)^2 : ℚ) / (-2)^2
def D : ℚ := ((-3)^2 : ℚ) / (-2)

theorem smallest_value_is_A : A < B ∧ A < C ∧ A < D :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_is_A_l1916_191648


namespace NUMINAMATH_GPT_oak_trees_remaining_l1916_191632

-- Variables representing the initial number of oak trees and the number of cut down trees.
variables (initial_trees cut_down_trees remaining_trees : ℕ)

-- Conditions of the problem.
def initial_trees_condition : initial_trees = 9 := sorry
def cut_down_trees_condition : cut_down_trees = 2 := sorry

-- Theorem representing the proof problem.
theorem oak_trees_remaining (h1 : initial_trees = 9) (h2 : cut_down_trees = 2) :
  remaining_trees = initial_trees - cut_down_trees :=
sorry

end NUMINAMATH_GPT_oak_trees_remaining_l1916_191632


namespace NUMINAMATH_GPT_total_people_present_l1916_191643

def parents : ℕ := 105
def pupils : ℕ := 698
def total_people (parents pupils : ℕ) : ℕ := parents + pupils

theorem total_people_present : total_people parents pupils = 803 :=
by
  sorry

end NUMINAMATH_GPT_total_people_present_l1916_191643


namespace NUMINAMATH_GPT_gcd_repeated_five_digit_number_l1916_191647

theorem gcd_repeated_five_digit_number :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 →
  ∀ m : ℕ, 10000 ≤ m ∧ m < 100000 →
  (10000100001 : ℕ) ∣ ((10^10 + 10^5 + 1) * n) ∧
  (10000100001 : ℕ) ∣ ((10^10 + 10^5 + 1) * m) →
  gcd ((10^10 + 10^5 + 1) * n) ((10^10 + 10^5 + 1) * m) = 10000100001 :=
sorry

end NUMINAMATH_GPT_gcd_repeated_five_digit_number_l1916_191647


namespace NUMINAMATH_GPT_find_value_l1916_191651

theorem find_value (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 108) : a^2 * b + a * b^2 = 108 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l1916_191651


namespace NUMINAMATH_GPT_combined_error_percentage_l1916_191691

theorem combined_error_percentage 
  (S : ℝ) 
  (error_side : ℝ) 
  (error_area : ℝ) 
  (h1 : error_side = 0.20) 
  (h2 : error_area = 0.04) :
  (1.04 * ((1 + error_side) * S) ^ 2 - S ^ 2) / S ^ 2 * 100 = 49.76 := 
by
  sorry

end NUMINAMATH_GPT_combined_error_percentage_l1916_191691


namespace NUMINAMATH_GPT_negation_of_prop_equiv_l1916_191692

-- Define the proposition
def prop (x : ℝ) : Prop := x^2 + 1 > 0

-- State the theorem that negation of proposition forall x, prop x is equivalent to exists x, ¬ prop x
theorem negation_of_prop_equiv :
  ¬ (∀ x : ℝ, prop x) ↔ ∃ x : ℝ, ¬ prop x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_prop_equiv_l1916_191692


namespace NUMINAMATH_GPT_total_toys_l1916_191617

theorem total_toys 
  (jaxon_toys : ℕ)
  (gabriel_toys : ℕ)
  (jerry_toys : ℕ)
  (h1 : jaxon_toys = 15)
  (h2 : gabriel_toys = 2 * jaxon_toys)
  (h3 : jerry_toys = gabriel_toys + 8) : 
  jaxon_toys + gabriel_toys + jerry_toys = 83 :=
  by sorry

end NUMINAMATH_GPT_total_toys_l1916_191617


namespace NUMINAMATH_GPT_total_ladybugs_eq_11676_l1916_191635

def Number_of_leaves : ℕ := 84
def Ladybugs_per_leaf : ℕ := 139

theorem total_ladybugs_eq_11676 : Number_of_leaves * Ladybugs_per_leaf = 11676 := by
  sorry

end NUMINAMATH_GPT_total_ladybugs_eq_11676_l1916_191635


namespace NUMINAMATH_GPT_total_cost_is_9220_l1916_191695

-- Define the conditions
def hourly_rate := 60
def hours_per_day := 8
def total_days := 14
def cost_of_parts := 2500

-- Define the total cost the car's owner had to pay based on conditions
def total_hours := hours_per_day * total_days
def labor_cost := total_hours * hourly_rate
def total_cost := labor_cost + cost_of_parts

-- Theorem stating that the total cost is $9220
theorem total_cost_is_9220 : total_cost = 9220 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_9220_l1916_191695


namespace NUMINAMATH_GPT_geometric_sequence_sums_l1916_191601

open Real

theorem geometric_sequence_sums (S T R : ℝ)
  (h1 : ∃ a r, S = a * (1 + r))
  (h2 : ∃ a r, T = a * (1 + r + r^2 + r^3))
  (h3 : ∃ a r, R = a * (1 + r + r^2 + r^3 + r^4 + r^5)) :
  S^2 + T^2 = S * (T + R) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sums_l1916_191601


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1916_191636

-- Define the positive terms of the geometric sequence
variables {a_1 a_2 a_3 a_4 a_5 : ℝ}
-- Assume all terms are positive
variables (h1 : a_1 > 0) (h2 : a_2 > 0) (h3 : a_3 > 0) (h4 : a_4 > 0) (h5 : a_5 > 0)

-- Main condition given in the problem
variable (h_main : a_1 * a_3 + 2 * a_2 * a_4 + a_3 * a_5 = 16)

-- Goal: Prove that a_2 + a_4 = 4
theorem geometric_sequence_sum : a_2 + a_4 = 4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1916_191636


namespace NUMINAMATH_GPT_min_w_for_factors_l1916_191607

theorem min_w_for_factors (w : ℕ) (h_pos : w > 0)
  (h_product_factors : ∀ k, k > 0 → ∃ a b : ℕ, (1452 * w = k) → (a = 3^3) ∧ (b = 13^3) ∧ (k % a = 0) ∧ (k % b = 0)) : 
  w = 19773 :=
sorry

end NUMINAMATH_GPT_min_w_for_factors_l1916_191607


namespace NUMINAMATH_GPT_belfried_industries_tax_l1916_191655

noncomputable def payroll_tax (payroll : ℕ) : ℕ :=
  if payroll <= 200000 then
    0
  else
    ((payroll - 200000) * 2) / 1000

theorem belfried_industries_tax : payroll_tax 300000 = 200 :=
by
  sorry

end NUMINAMATH_GPT_belfried_industries_tax_l1916_191655


namespace NUMINAMATH_GPT_total_participants_l1916_191611

theorem total_participants (Petya Vasya total : ℕ) 
  (h1 : Petya = Vasya + 1) 
  (h2 : Petya = 10)
  (h3 : Vasya + 15 = total + 1) : 
  total = 23 :=
by
  sorry

end NUMINAMATH_GPT_total_participants_l1916_191611


namespace NUMINAMATH_GPT_triangles_with_two_white_vertices_l1916_191629

theorem triangles_with_two_white_vertices (p f z : ℕ) 
    (h1 : p * f + p * z + f * z = 213)
    (h2 : (p * (p - 1) / 2) + (f * (f - 1) / 2) + (z * (z - 1) / 2) = 112)
    (h3 : p * f * z = 540)
    (h4 : (p * (p - 1) / 2) * (f + z) = 612) :
    (f * (f - 1) / 2) * (p + z) = 210 ∨ (f * (f - 1) / 2) * (p + z) = 924 := 
  sorry

end NUMINAMATH_GPT_triangles_with_two_white_vertices_l1916_191629


namespace NUMINAMATH_GPT_floor_e_minus_3_eq_neg1_l1916_191606

noncomputable def e : ℝ := 2.718

theorem floor_e_minus_3_eq_neg1 : Int.floor (e - 3) = -1 := by
  sorry

end NUMINAMATH_GPT_floor_e_minus_3_eq_neg1_l1916_191606


namespace NUMINAMATH_GPT_smallest_scalene_prime_triangle_perimeter_l1916_191616

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a scalene triangle with distinct side lengths
def is_scalene (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Define the triangle inequality
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define a valid scalene triangle with prime side lengths
def valid_scalene_triangle (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_scalene a b c ∧ triangle_inequality a b c

-- Proof statement
theorem smallest_scalene_prime_triangle_perimeter : ∃ (a b c : ℕ), 
  valid_scalene_triangle a b c ∧ a + b + c = 15 := 
sorry

end NUMINAMATH_GPT_smallest_scalene_prime_triangle_perimeter_l1916_191616


namespace NUMINAMATH_GPT_prob_a_greater_than_b_l1916_191633

noncomputable def probability_of_team_a_finishing_with_more_points (n_teams : ℕ) (initial_win : Bool) : ℚ :=
  if initial_win ∧ n_teams = 9 then
    39203 / 65536
  else
    0 -- This is a placeholder and not accurate for other cases

theorem prob_a_greater_than_b (n_teams : ℕ) (initial_win : Bool) (hp : initial_win ∧ n_teams = 9) :
  probability_of_team_a_finishing_with_more_points n_teams initial_win = 39203 / 65536 :=
by
  sorry

end NUMINAMATH_GPT_prob_a_greater_than_b_l1916_191633


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l1916_191688

theorem average_of_remaining_two_numbers (A B C D E : ℝ) 
  (h1 : A + B + C + D + E = 50) 
  (h2 : A + B + C = 12) : 
  (D + E) / 2 = 19 :=
by
  sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l1916_191688


namespace NUMINAMATH_GPT_min_apples_l1916_191605

theorem min_apples :
  ∃ N : ℕ, 
  (N % 3 = 2) ∧ 
  (N % 4 = 2) ∧ 
  (N % 5 = 2) ∧ 
  (N = 62) :=
by
  sorry

end NUMINAMATH_GPT_min_apples_l1916_191605


namespace NUMINAMATH_GPT_find_k_for_minimum_value_l1916_191634

theorem find_k_for_minimum_value :
  ∃ (k : ℝ), (∀ (x y : ℝ), 9 * x^2 - 6 * k * x * y + (3 * k^2 + 1) * y^2 - 6 * x - 6 * y + 7 ≥ 1)
  ∧ (∃ (x y : ℝ), 9 * x^2 - 6 * k * x * y + (3 * k^2 + 1) * y^2 - 6 * x - 6 * y + 7 = 1)
  ∧ k = 3 :=
sorry

end NUMINAMATH_GPT_find_k_for_minimum_value_l1916_191634


namespace NUMINAMATH_GPT_find_abs_sum_roots_l1916_191669

noncomputable def polynomial_root_abs_sum (n p q r : ℤ) : Prop :=
(p + q + r = 0) ∧
(p * q + q * r + r * p = -2009) ∧
(p * q * r = -n) →
(|p| + |q| + |r| = 102)

theorem find_abs_sum_roots (n p q r : ℤ) :
  polynomial_root_abs_sum n p q r :=
sorry

end NUMINAMATH_GPT_find_abs_sum_roots_l1916_191669


namespace NUMINAMATH_GPT_total_interest_after_tenth_year_l1916_191659

variable {P R : ℕ}

theorem total_interest_after_tenth_year
  (h1 : (P * R * 10) / 100 = 900)
  (h2 : 5 * P * R / 100 = 450)
  (h3 : 5 * 3 * P * R / 100 = 1350) :
  (450 + 1350) = 1800 :=
by
  sorry

end NUMINAMATH_GPT_total_interest_after_tenth_year_l1916_191659


namespace NUMINAMATH_GPT_evaluate_expression_l1916_191679

theorem evaluate_expression : 1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 11 / 7 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1916_191679


namespace NUMINAMATH_GPT_retailer_should_focus_on_mode_l1916_191608

-- Define the conditions as options.
inductive ClothingModels
| Average
| Mode
| Median
| Smallest

-- Define a function to determine the best measure to focus on in the market share survey.
def bestMeasureForMarketShareSurvey (choice : ClothingModels) : Prop :=
  match choice with
  | ClothingModels.Average => False
  | ClothingModels.Mode => True
  | ClothingModels.Median => False
  | ClothingModels.Smallest => False

-- The theorem stating that the mode is the best measure to focus on.
theorem retailer_should_focus_on_mode : bestMeasureForMarketShareSurvey ClothingModels.Mode :=
by
  -- This proof is intentionally left blank.
  sorry

end NUMINAMATH_GPT_retailer_should_focus_on_mode_l1916_191608


namespace NUMINAMATH_GPT_contrapositive_statement_l1916_191684

theorem contrapositive_statement (x y : ℤ) : ¬ (x + y) % 2 = 1 → ¬ (x % 2 = 1 ∧ y % 2 = 1) :=
sorry

end NUMINAMATH_GPT_contrapositive_statement_l1916_191684


namespace NUMINAMATH_GPT_arithmetic_operators_correct_l1916_191664

theorem arithmetic_operators_correct :
  let op1 := (132: ℝ) - (7: ℝ) * (6: ℝ)
  let op2 := (12: ℝ) + (3: ℝ)
  (op1 / op2) = (6: ℝ) := by 
  sorry

end NUMINAMATH_GPT_arithmetic_operators_correct_l1916_191664


namespace NUMINAMATH_GPT_halfway_between_one_nine_and_one_eleven_l1916_191656

theorem halfway_between_one_nine_and_one_eleven : 
  (1/9 + 1/11) / 2 = 10/99 :=
by sorry

end NUMINAMATH_GPT_halfway_between_one_nine_and_one_eleven_l1916_191656


namespace NUMINAMATH_GPT_find_f3_l1916_191680

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_f3
  (a b : ℝ)
  (h1 : f a b 3 1 = 7)
  (h2 : f a b 3 2 = 12) :
  f a b 3 3 = 18 :=
sorry

end NUMINAMATH_GPT_find_f3_l1916_191680


namespace NUMINAMATH_GPT_value_of_expression_l1916_191682

theorem value_of_expression (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 4 / 3) :
  (1 / 3 * x^7 * y^6) * 4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1916_191682


namespace NUMINAMATH_GPT_interval_for_systematic_sampling_l1916_191666

-- Define the total number of students
def total_students : ℕ := 1200

-- Define the sample size
def sample_size : ℕ := 30

-- Define the interval for systematic sampling
def interval_k : ℕ := total_students / sample_size

-- The theorem to prove that the interval k should be 40
theorem interval_for_systematic_sampling :
  interval_k = 40 := sorry

end NUMINAMATH_GPT_interval_for_systematic_sampling_l1916_191666


namespace NUMINAMATH_GPT_ways_to_fifth_floor_l1916_191677

theorem ways_to_fifth_floor (floors : ℕ) (staircases : ℕ) (h_floors : floors = 5) (h_staircases : staircases = 2) :
  (staircases ^ (floors - 1)) = 16 :=
by
  rw [h_floors, h_staircases]
  sorry

end NUMINAMATH_GPT_ways_to_fifth_floor_l1916_191677


namespace NUMINAMATH_GPT_modulo_remainder_l1916_191657

theorem modulo_remainder :
  (7 * 10^24 + 2^24) % 13 = 8 := 
by
  sorry

end NUMINAMATH_GPT_modulo_remainder_l1916_191657


namespace NUMINAMATH_GPT_kite_AB_BC_ratio_l1916_191618

-- Define the kite properties and necessary elements to state the problem
def kite_problem (AB BC: ℝ) (angleB angleD : ℝ) (MN'_parallel_AC : Prop) : Prop :=
  angleB = 90 ∧ angleD = 90 ∧ MN'_parallel_AC ∧ AB / BC = (1 + Real.sqrt 5) / 2

-- Define the main theorem to be proven
theorem kite_AB_BC_ratio (AB BC : ℝ) (angleB angleD : ℝ) (MN'_parallel_AC : Prop) :
  kite_problem AB BC angleB angleD MN'_parallel_AC :=
by
  sorry

-- Statement of the condition that need to be satisfied
axiom MN'_parallel_AC : Prop

-- Example instantiation of the problem
example : kite_problem 1 1 90 90 MN'_parallel_AC :=
by
  sorry

end NUMINAMATH_GPT_kite_AB_BC_ratio_l1916_191618


namespace NUMINAMATH_GPT_cost_of_450_candies_l1916_191658

theorem cost_of_450_candies :
  let cost_per_box := 8
  let candies_per_box := 30
  let num_candies := 450
  cost_per_box * (num_candies / candies_per_box) = 120 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_450_candies_l1916_191658


namespace NUMINAMATH_GPT_inequality_proof_l1916_191675

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (n : ℕ) (hn : 0 < n) : 
  (x / (n * x + y + z) + y / (x + n * y + z) + z / (x + y + n * z)) ≤ 3 / (n + 2) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1916_191675


namespace NUMINAMATH_GPT_f_neg_a_l1916_191661

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end NUMINAMATH_GPT_f_neg_a_l1916_191661


namespace NUMINAMATH_GPT_count_distinct_product_divisors_l1916_191610

-- Define the properties of 8000 and its divisors
def isDivisor (n d : ℕ) := d > 0 ∧ n % d = 0

def T := {d : ℕ | isDivisor 8000 d}

-- The main statement to prove
theorem count_distinct_product_divisors : 
    (∃ n : ℕ, n ∈ { m | ∃ a b : ℕ, a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ m = a * b } ∧ n = 88) :=
by {
  sorry
}

end NUMINAMATH_GPT_count_distinct_product_divisors_l1916_191610


namespace NUMINAMATH_GPT_range_f_sum_l1916_191686

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x ^ 2)

theorem range_f_sum {a b : ℝ} (h₁ : Set.Ioo a b = (Set.Ioo (0 : ℝ) (3 : ℝ))) :
  a + b = 3 :=
sorry

end NUMINAMATH_GPT_range_f_sum_l1916_191686


namespace NUMINAMATH_GPT_inequality_represents_area_l1916_191697

theorem inequality_represents_area (a : ℝ) :
  (if a > 1 then ∀ (x y : ℝ), x + (a - 1) * y + 3 > 0 ↔ y < - (x + 3) / (a - 1)
  else ∀ (x y : ℝ), x + (a - 1) * y + 3 > 0 ↔ y > - (x + 3) / (a - 1)) :=
by sorry

end NUMINAMATH_GPT_inequality_represents_area_l1916_191697


namespace NUMINAMATH_GPT_max_chocolate_bars_l1916_191667

-- Definitions
def john_money := 2450
def chocolate_bar_cost := 220

-- Theorem statement
theorem max_chocolate_bars : ∃ (x : ℕ), x = 11 ∧ chocolate_bar_cost * x ≤ john_money ∧ (chocolate_bar_cost * (x + 1) > john_money) := 
by 
  -- This is to indicate we're acknowledging that the proof is left as an exercise
  sorry

end NUMINAMATH_GPT_max_chocolate_bars_l1916_191667


namespace NUMINAMATH_GPT_power_neg8_equality_l1916_191683

theorem power_neg8_equality :
  (1 / ((-8 : ℤ) ^ 2)^3) * (-8 : ℤ)^7 = 8 :=
by
  sorry

end NUMINAMATH_GPT_power_neg8_equality_l1916_191683


namespace NUMINAMATH_GPT_inequality_solution_l1916_191615

theorem inequality_solution (x : ℝ) :
  -1 < (x^2 - 12 * x + 35) / (x^2 - 4 * x + 8) ∧
  (x^2 - 12 * x + 35) / (x^2 - 4 * x + 8) < 1 ↔
  x > (27 / 8) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l1916_191615


namespace NUMINAMATH_GPT_problem1_problem2_l1916_191678

-- Problem 1
theorem problem1 (x : ℤ) : (x - 2) ^ 2 - (x - 3) * (x + 3) = -4 * x + 13 := by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (h₁ : x ≠ 1) : 
  (x^2 + 2 * x) / (x^2 - 1) / (x + 1 + (2 * x + 1) / (x - 1)) = 1 / (x + 1) := by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1916_191678


namespace NUMINAMATH_GPT_max_subsequences_2001_l1916_191650

theorem max_subsequences_2001 (seq : List ℕ) (h_len : seq.length = 2001) : 
  ∃ n : ℕ, n = 667^3 :=
sorry

end NUMINAMATH_GPT_max_subsequences_2001_l1916_191650


namespace NUMINAMATH_GPT_mary_spent_total_amount_l1916_191652

def shirt_cost : ℝ := 13.04
def jacket_cost : ℝ := 12.27
def total_cost : ℝ := 25.31

theorem mary_spent_total_amount :
  shirt_cost + jacket_cost = total_cost := sorry

end NUMINAMATH_GPT_mary_spent_total_amount_l1916_191652


namespace NUMINAMATH_GPT_total_money_raised_for_charity_l1916_191620

theorem total_money_raised_for_charity:
    let price_small := 2
    let price_medium := 3
    let price_large := 5
    let num_small := 150
    let num_medium := 221
    let num_large := 185
    num_small * price_small + num_medium * price_medium + num_large * price_large = 1888 := by
  sorry

end NUMINAMATH_GPT_total_money_raised_for_charity_l1916_191620


namespace NUMINAMATH_GPT_larger_number_is_2997_l1916_191698

theorem larger_number_is_2997 (L S : ℕ) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 := 
by
  sorry

end NUMINAMATH_GPT_larger_number_is_2997_l1916_191698


namespace NUMINAMATH_GPT_minimize_J_l1916_191674

noncomputable def H (p q : ℝ) : ℝ := -3 * p * q + 4 * p * (1 - q) + 5 * (1 - p) * q - 6 * (1 - p) * (1 - q) + 2 * p

noncomputable def J (p : ℝ) : ℝ := max (H p 0) (H p 1)

theorem minimize_J (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : p = 11 / 18 ↔ ∀ q, 0 ≤ q ∧ q ≤ 1 → J p = J (11 / 18) := 
by
  sorry

end NUMINAMATH_GPT_minimize_J_l1916_191674


namespace NUMINAMATH_GPT_find_angle_B_l1916_191630

def angle_A (B : ℝ) : ℝ := B + 21
def angle_C (B : ℝ) : ℝ := B + 36
def is_triangle_sum (A B C : ℝ) : Prop := A + B + C = 180

theorem find_angle_B (B : ℝ) 
  (hA : angle_A B = B + 21) 
  (hC : angle_C B = B + 36) 
  (h_sum : is_triangle_sum (angle_A B) B (angle_C B) ) : B = 41 :=
  sorry

end NUMINAMATH_GPT_find_angle_B_l1916_191630


namespace NUMINAMATH_GPT_ratio_of_populations_l1916_191640

theorem ratio_of_populations (ne_pop : ℕ) (combined_pop : ℕ) (ny_pop : ℕ) (h1 : ne_pop = 2100000) 
                            (h2 : combined_pop = 3500000) (h3 : ny_pop = combined_pop - ne_pop) :
                            (ny_pop * 3 = ne_pop * 2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_populations_l1916_191640


namespace NUMINAMATH_GPT_Taimour_painting_time_l1916_191676

theorem Taimour_painting_time (T : ℝ) 
  (h1 : ∀ (T : ℝ), Jamshid_time = 0.5 * T) 
  (h2 : (1 / T + 2 / T) * 7 = 1) : 
    T = 21 :=
by
  sorry

end NUMINAMATH_GPT_Taimour_painting_time_l1916_191676


namespace NUMINAMATH_GPT_number_of_n_l1916_191612

theorem number_of_n (n : ℕ) (h1 : n > 0) (h2 : n ≤ 1200) (h3 : ∃ k : ℕ, 12 * n = k^2) :
  ∃ m : ℕ, m = 10 :=
by { sorry }

end NUMINAMATH_GPT_number_of_n_l1916_191612


namespace NUMINAMATH_GPT_soccer_club_girls_count_l1916_191604

theorem soccer_club_girls_count
  (total_members : ℕ)
  (attended : ℕ)
  (B G : ℕ)
  (h1 : B + G = 30)
  (h2 : (1/3 : ℚ) * G + B = 18) : G = 18 := by
  sorry

end NUMINAMATH_GPT_soccer_club_girls_count_l1916_191604


namespace NUMINAMATH_GPT_factorize_a3_minus_4a_l1916_191694

theorem factorize_a3_minus_4a (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) := 
by
  sorry

end NUMINAMATH_GPT_factorize_a3_minus_4a_l1916_191694


namespace NUMINAMATH_GPT_omitted_decimal_sum_is_integer_l1916_191699

def numbers : List ℝ := [1.05, 1.15, 1.25, 1.4, 1.5, 1.6, 1.75, 1.85, 1.95]

theorem omitted_decimal_sum_is_integer :
  1.05 + 1.15 + 1.25 + 1.4 + (15 : ℝ) + 1.6 + 1.75 + 1.85 + 1.95 = 27 :=
by sorry

end NUMINAMATH_GPT_omitted_decimal_sum_is_integer_l1916_191699


namespace NUMINAMATH_GPT_find_a_n_l1916_191645

-- Definitions from the conditions
def seq (a : ℕ → ℤ) : Prop :=
  ∀ n, (3 - a (n + 1)) * (6 + a n) = 18

-- The Lean statement of the problem
theorem find_a_n (a : ℕ → ℤ) (h_a0 : a 0 ≠ 3) (h_seq : seq a) :
  ∀ n, a n = 2 ^ (n + 2) - n - 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_n_l1916_191645


namespace NUMINAMATH_GPT_find_car_costs_optimize_purchasing_plan_minimum_cost_l1916_191665

theorem find_car_costs (x y : ℝ) (h1 : 3 * x + y = 85) (h2 : 2 * x + 4 * y = 140) :
    x = 20 ∧ y = 25 :=
by
  sorry

theorem optimize_purchasing_plan (m : ℕ) (h_total : m + (15 - m) = 15) (h_constraint : m ≤ 2 * (15 - m)) :
    m = 10 :=
by
  sorry

theorem minimum_cost (w : ℝ) (h_cost_expr : ∀ (m : ℕ), w = 20 * m + 25 * (15 - m)) (m := 10) :
    w = 325 :=
by
  sorry

end NUMINAMATH_GPT_find_car_costs_optimize_purchasing_plan_minimum_cost_l1916_191665


namespace NUMINAMATH_GPT_equal_ivan_petrovich_and_peter_ivanovich_l1916_191603

theorem equal_ivan_petrovich_and_peter_ivanovich :
  (∀ n : ℕ, n % 10 = 0 → (n % 20 = 0) = (n % 200 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_equal_ivan_petrovich_and_peter_ivanovich_l1916_191603


namespace NUMINAMATH_GPT_survey_respondents_l1916_191681

theorem survey_respondents (X Y : ℕ) (hX : X = 150) (hRatio : X / Y = 5) : X + Y = 180 :=
by
  sorry

end NUMINAMATH_GPT_survey_respondents_l1916_191681


namespace NUMINAMATH_GPT_crayons_eaten_l1916_191628

def initial_crayons : ℕ := 87
def remaining_crayons : ℕ := 80

theorem crayons_eaten : initial_crayons - remaining_crayons = 7 := by
  sorry

end NUMINAMATH_GPT_crayons_eaten_l1916_191628


namespace NUMINAMATH_GPT_find_original_number_l1916_191690

-- Definitions based on the conditions of the problem
def tens_digit (x : ℕ) := 2 * x
def original_number (x : ℕ) := 10 * (tens_digit x) + x
def reversed_number (x : ℕ) := 10 * x + (tens_digit x)

-- Proof statement
theorem find_original_number (x : ℕ) (h1 : original_number x - reversed_number x = 27) : original_number x = 63 := by
  sorry

end NUMINAMATH_GPT_find_original_number_l1916_191690


namespace NUMINAMATH_GPT_common_divisor_greater_than_1_l1916_191642
open Nat

theorem common_divisor_greater_than_1 (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_ab : (a + b) ∣ (a * b)) (h_bc : (b + c) ∣ (b * c)) (h_ca : (c + a) ∣ (c * a)) :
    ∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ b ∧ k ∣ c := 
by
  sorry

end NUMINAMATH_GPT_common_divisor_greater_than_1_l1916_191642
