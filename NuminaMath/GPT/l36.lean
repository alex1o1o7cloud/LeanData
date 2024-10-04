import Mathlib

namespace total_amount_spent_correct_l36_36543

-- Definitions based on conditions
def price_of_food_before_tax_and_tip : ℝ := 140
def sales_tax_rate : ℝ := 0.10
def tip_rate : ℝ := 0.20

-- Definitions of intermediate steps
def sales_tax : ℝ := sales_tax_rate * price_of_food_before_tax_and_tip
def total_before_tip : ℝ := price_of_food_before_tax_and_tip + sales_tax
def tip : ℝ := tip_rate * total_before_tip
def total_amount_spent : ℝ := total_before_tip + tip

-- Theorem statement to be proved
theorem total_amount_spent_correct : total_amount_spent = 184.80 :=
by
  sorry -- Proof is skipped

end total_amount_spent_correct_l36_36543


namespace find_initial_marbles_l36_36880

-- Definitions based on conditions
def loses_to_street (initial_marbles : ℕ) : ℕ := initial_marbles - (initial_marbles * 60 / 100)
def loses_to_sewer (marbles_after_street : ℕ) : ℕ := marbles_after_street / 2

-- The given number of marbles left
def remaining_marbles : ℕ := 20

-- Proof statement
theorem find_initial_marbles (initial_marbles : ℕ) : 
  loses_to_sewer (loses_to_street initial_marbles) = remaining_marbles -> 
  initial_marbles = 100 :=
by
  sorry

end find_initial_marbles_l36_36880


namespace distinct_patterns_4x4_three_squares_l36_36159

noncomputable def count_distinct_patterns : ℕ :=
  sorry

theorem distinct_patterns_4x4_three_squares :
  count_distinct_patterns = 12 :=
by sorry

end distinct_patterns_4x4_three_squares_l36_36159


namespace min_value_expression_l36_36057

theorem min_value_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 8) : 
  (x + 3 * y) * (y + 3 * z) * (3 * x * z + 1) ≥ 72 :=
sorry

end min_value_expression_l36_36057


namespace M_inter_N_l36_36612

namespace ProofProblem

def M : Set ℝ := { x | 3 * x - x^2 > 0 }
def N : Set ℝ := { x | x^2 - 4 * x + 3 > 0 }

theorem M_inter_N : M ∩ N = { x | 0 < x ∧ x < 1 } :=
sorry

end ProofProblem

end M_inter_N_l36_36612


namespace angle_measure_l36_36879

theorem angle_measure (A B C : ℝ) (h1 : A = B) (h2 : A + B = 110 ∨ (A = 180 - 110)) :
  A = 70 ∨ A = 55 := by
  sorry

end angle_measure_l36_36879


namespace trisha_money_left_l36_36091

theorem trisha_money_left
    (meat cost: ℕ) (chicken_cost: ℕ) (veggies_cost: ℕ) (eggs_cost: ℕ) (dog_food_cost: ℕ) 
    (initial_money: ℕ) (total_spent: ℕ) (money_left: ℕ) :
    meat_cost = 17 →
    chicken_cost = 22 →
    veggies_cost = 43 →
    eggs_cost = 5 →
    dog_food_cost = 45 →
    initial_money = 167 →
    total_spent = meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost →
    money_left = initial_money - total_spent →
    money_left = 35 :=
by
    intros
    sorry

end trisha_money_left_l36_36091


namespace evaporation_amount_l36_36401

noncomputable def water_evaporated_per_day (total_water: ℝ) (percentage_evaporated: ℝ) (days: ℕ) : ℝ :=
  (percentage_evaporated / 100) * total_water / days

theorem evaporation_amount :
  water_evaporated_per_day 10 7 50 = 0.014 :=
by
  sorry

end evaporation_amount_l36_36401


namespace isosceles_trapezoid_with_inscribed_circle_area_is_20_l36_36138

def isosceles_trapezoid_area (a b c1 c2 h : ℕ) : ℕ :=
  (a + b) * h / 2

theorem isosceles_trapezoid_with_inscribed_circle_area_is_20
  (a b c h : ℕ)
  (ha : a = 2)
  (hb : b = 8)
  (hc : a + b = 2 * c)
  (hh : h ^ 2 = c ^ 2 - ((b - a) / 2) ^ 2) :
  isosceles_trapezoid_area a b c c h = 20 := 
by {
  sorry
}

end isosceles_trapezoid_with_inscribed_circle_area_is_20_l36_36138


namespace box_with_20_aluminium_80_plastic_weighs_494_l36_36238

def weight_of_box_with_100_aluminium_balls := 510 -- in grams
def weight_of_box_with_100_plastic_balls := 490 -- in grams
def number_of_aluminium_balls := 100
def number_of_plastic_balls := 100

-- Define the weights per ball type by subtracting the weight of the box
def weight_per_aluminium_ball := (weight_of_box_with_100_aluminium_balls - weight_of_box_with_100_plastic_balls) / number_of_aluminium_balls
def weight_per_plastic_ball := (weight_of_box_with_100_plastic_balls - weight_of_box_with_100_plastic_balls) / number_of_plastic_balls

-- Condition: The weight of the box alone (since it's present in both conditions)
def weight_of_empty_box := weight_of_box_with_100_plastic_balls - (weight_per_plastic_ball * number_of_plastic_balls)

-- Function to compute weight of the box with given number of aluminium and plastic balls
def total_weight (num_al : ℕ) (num_pl : ℕ) : ℕ :=
  weight_of_empty_box + (weight_per_aluminium_ball * num_al) + (weight_per_plastic_ball * num_pl)

-- The theorem to be proven
theorem box_with_20_aluminium_80_plastic_weighs_494 :
  total_weight 20 80 = 494 := sorry

end box_with_20_aluminium_80_plastic_weighs_494_l36_36238


namespace reciprocal_of_2022_l36_36670

theorem reciprocal_of_2022 : 1 / 2022 = (1 : ℝ) / 2022 :=
sorry

end reciprocal_of_2022_l36_36670


namespace sum_eighth_row_interior_numbers_l36_36631

-- Define the sum of the interior numbers in the nth row of Pascal's Triangle.
def sum_interior_numbers (n : ℕ) : ℕ := 2^(n-1) - 2

-- Problem statement: Prove the sum of the interior numbers of Pascal's Triangle in the eighth row is 126,
-- given the sums for the fifth and sixth rows.
theorem sum_eighth_row_interior_numbers :
  sum_interior_numbers 5 = 14 →
  sum_interior_numbers 6 = 30 →
  sum_interior_numbers 8 = 126 :=
by
  sorry

end sum_eighth_row_interior_numbers_l36_36631


namespace trigonometric_identity_l36_36594

theorem trigonometric_identity 
  (x : ℝ) 
  (h : Real.sin (x + (Real.pi / 6)) = 1 / 4) : 
  Real.sin (5 * Real.pi / 6 - x) + Real.cos (Real.pi / 3 - x) ^ 2 = 5 / 16 := 
sorry

end trigonometric_identity_l36_36594


namespace segment_length_R_R_l36_36973

theorem segment_length_R_R' :
  let R := (-4, 1)
  let R' := (-4, -1)
  let distance : ℝ := Real.sqrt ((R'.1 - R.1)^2 + (R'.2 - R.2)^2)
  distance = 2 :=
by
  sorry

end segment_length_R_R_l36_36973


namespace trapezium_other_parallel_side_l36_36897

theorem trapezium_other_parallel_side (a b h : ℝ) (area : ℝ) (h_area : area = (1 / 2) * (a + b) * h) (h_a : a = 18) (h_h : h = 20) (h_area_val : area = 380) :
  b = 20 :=
by 
  sorry

end trapezium_other_parallel_side_l36_36897


namespace spring_mass_relationship_l36_36627

theorem spring_mass_relationship (x y : ℕ) (h1 : y = 18 + 2 * x) : 
  y = 32 → x = 7 :=
by
  sorry

end spring_mass_relationship_l36_36627


namespace length_of_train_l36_36248

variables (L : ℝ) (t1 t2 : ℝ) (length_platform : ℝ)

-- Conditions
def condition1 := t1 = 39
def condition2 := t2 = 18
def condition3 := length_platform = 350

-- The goal is to prove the length of the train
theorem length_of_train : condition1 ∧ condition2 ∧ condition3 → L = 300 :=
by
  intros h
  sorry

end length_of_train_l36_36248


namespace problems_per_page_l36_36941

theorem problems_per_page (total_problems finished_problems remaining_pages problems_per_page : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : remaining_pages = 2)
  (h4 : total_problems - finished_problems = 14)
  (h5 : 14 = remaining_pages * problems_per_page) :
  problems_per_page = 7 := 
by
  sorry

end problems_per_page_l36_36941


namespace inscribed_circle_radius_integer_l36_36346

theorem inscribed_circle_radius_integer (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ (r : ℤ), r = (a + b - c) / 2 := by
  sorry

end inscribed_circle_radius_integer_l36_36346


namespace intersection_A_B_l36_36320

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {x | x ∈ A ∧ (x : ℝ) ∈ B}

theorem intersection_A_B : C = {0, 1, 2} := 
by
  sorry

end intersection_A_B_l36_36320


namespace total_flowers_purchased_l36_36636

-- Define the conditions
def sets : ℕ := 3
def pieces_per_set : ℕ := 90

-- State the proof problem
theorem total_flowers_purchased : sets * pieces_per_set = 270 :=
by
  sorry

end total_flowers_purchased_l36_36636


namespace modulo_7_example_l36_36450

def sum := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem modulo_7_example : (sum % 7) = 5 :=
by
  sorry

end modulo_7_example_l36_36450


namespace speed_of_rest_distance_l36_36721

theorem speed_of_rest_distance (D V : ℝ) (h1 : D = 26.67)
                                (h2 : (D / 2) / 5 + (D / 2) / V = 6) : 
  V = 20 :=
by
  sorry

end speed_of_rest_distance_l36_36721


namespace combined_length_in_scientific_notation_l36_36507

noncomputable def yards_to_inches (yards : ℝ) : ℝ := yards * 36
noncomputable def inches_to_cm (inches : ℝ) : ℝ := inches * 2.54
noncomputable def feet_to_inches (feet : ℝ) : ℝ := feet * 12

def sports_stadium_length_yards : ℝ := 61
def safety_margin_feet : ℝ := 2
def safety_margin_inches : ℝ := 9

theorem combined_length_in_scientific_notation :
  (inches_to_cm (yards_to_inches sports_stadium_length_yards) +
   (inches_to_cm (feet_to_inches safety_margin_feet + safety_margin_inches)) * 2) = 5.74268 * 10^3 :=
by
  sorry

end combined_length_in_scientific_notation_l36_36507


namespace cuboid_surface_area_l36_36762

/--
Given a cuboid with length 10 cm, breadth 8 cm, and height 6 cm, the surface area is 376 cm².
-/
theorem cuboid_surface_area 
  (length : ℝ) 
  (breadth : ℝ) 
  (height : ℝ) 
  (h_length : length = 10) 
  (h_breadth : breadth = 8) 
  (h_height : height = 6) : 
  2 * (length * height + length * breadth + breadth * height) = 376 := 
by 
  -- Replace these placeholders with the actual proof steps.
  sorry

end cuboid_surface_area_l36_36762


namespace gcd_largest_divisor_l36_36677

theorem gcd_largest_divisor (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 1008) : 
  ∃ d, nat.gcd a b = d ∧ d = 504 :=
begin
  sorry
end

end gcd_largest_divisor_l36_36677


namespace original_triangle_area_l36_36363

theorem original_triangle_area (area_of_new_triangle : ℝ) (side_length_ratio : ℝ) (quadrupled : side_length_ratio = 4) (new_area : area_of_new_triangle = 128) : 
  (area_of_new_triangle / side_length_ratio ^ 2) = 8 := by
  sorry

end original_triangle_area_l36_36363


namespace exist_common_divisor_l36_36332

theorem exist_common_divisor (a : ℕ → ℕ) (m : ℕ) (h_positive : ∀ i, 1 ≤ i ∧ i ≤ m → 0 < a i)
  (p : ℕ → ℤ) (h_poly : ∀ n : ℕ, ∃ i, 1 ≤ i ∧ i ≤ m ∧ (a i : ℤ) ∣ p n) :
  ∃ j, 1 ≤ j ∧ j ≤ m ∧ ∀ n, (a j : ℤ) ∣ p n :=
by
  sorry

end exist_common_divisor_l36_36332


namespace total_books_sold_amount_l36_36303

def num_fiction_books := 60
def num_non_fiction_books := 84
def num_children_books := 42

def fiction_books_sold := 3 / 4 * num_fiction_books
def non_fiction_books_sold := 5 / 6 * num_non_fiction_books
def children_books_sold := 2 / 3 * num_children_books

def price_fiction := 5
def price_non_fiction := 7
def price_children := 3

def total_amount_fiction := fiction_books_sold * price_fiction
def total_amount_non_fiction := non_fiction_books_sold * price_non_fiction
def total_amount_children := children_books_sold * price_children

def total_amount_received := total_amount_fiction + total_amount_non_fiction + total_amount_children

theorem total_books_sold_amount :
  total_amount_received = 799 :=
sorry

end total_books_sold_amount_l36_36303


namespace inequality_holds_if_and_only_if_c_lt_0_l36_36329

theorem inequality_holds_if_and_only_if_c_lt_0 (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (c < 0) :=
sorry

end inequality_holds_if_and_only_if_c_lt_0_l36_36329


namespace remainder_2_pow_19_div_7_l36_36523

theorem remainder_2_pow_19_div_7 :
  2^19 % 7 = 2 := by
  sorry

end remainder_2_pow_19_div_7_l36_36523


namespace servant_cash_received_l36_36289

theorem servant_cash_received (annual_cash : ℕ) (turban_price : ℕ) (served_months : ℕ) (total_months : ℕ) (cash_received : ℕ) :
  annual_cash = 90 → turban_price = 50 → served_months = 9 → total_months = 12 → 
  cash_received = (annual_cash + turban_price) * served_months / total_months - turban_price → 
  cash_received = 55 :=
by {
  intros;
  sorry
}

end servant_cash_received_l36_36289


namespace class_duration_l36_36720

theorem class_duration (h1 : 8 * 60 + 30 = 510) (h2 : 9 * 60 + 5 = 545) : (545 - 510 = 35) :=
by
  sorry

end class_duration_l36_36720


namespace reciprocal_of_2022_l36_36669

theorem reciprocal_of_2022 : 1 / 2022 = (1 : ℝ) / 2022 :=
sorry

end reciprocal_of_2022_l36_36669


namespace inscribed_circle_radius_integer_l36_36352

theorem inscribed_circle_radius_integer 
  (a b c : ℕ) (h : a^2 + b^2 = c^2) 
  (h₀ : 2 * (a + b - c) = k) 
  : ∃ (r : ℕ), r = (a + b - c) / 2 := 
begin
  sorry
end

end inscribed_circle_radius_integer_l36_36352


namespace find_M_N_sum_l36_36294

theorem find_M_N_sum
  (M N : ℕ)
  (h1 : 3 * 75 = 5 * M)
  (h2 : 3 * N = 5 * 90) :
  M + N = 195 := 
sorry

end find_M_N_sum_l36_36294


namespace vec_parallel_l36_36288

variable {R : Type*} [LinearOrderedField R]

def is_parallel (a b : R × R) : Prop :=
  ∃ k : R, a = (k * b.1, k * b.2)

theorem vec_parallel {x : R} : 
  is_parallel (1, x) (-3, 4) ↔ x = -4/3 := by
  sorry

end vec_parallel_l36_36288


namespace next_year_multiple_of_6_8_9_l36_36526

theorem next_year_multiple_of_6_8_9 (n : ℕ) (h₀ : n = 2016) (h₁ : n % 6 = 0) (h₂ : n % 8 = 0) (h₃ : n % 9 = 0) : ∃ m > n, m % 6 = 0 ∧ m % 8 = 0 ∧ m % 9 = 0 ∧ m = 2088 :=
by
  sorry

end next_year_multiple_of_6_8_9_l36_36526


namespace rain_at_least_once_prob_l36_36838

theorem rain_at_least_once_prob (p : ℚ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 :=
by {
  -- Implementation of Lean code is not required as per instructions.
  sorry
}

end rain_at_least_once_prob_l36_36838


namespace percentage_green_shirts_correct_l36_36467

variable (total_students blue_percentage red_percentage other_students : ℕ)

noncomputable def percentage_green_shirts (total_students blue_percentage red_percentage other_students : ℕ) : ℕ :=
  let total_blue_shirts := blue_percentage * total_students / 100
  let total_red_shirts := red_percentage * total_students / 100
  let total_blue_red_other_shirts := total_blue_shirts + total_red_shirts + other_students
  let green_shirts := total_students - total_blue_red_other_shirts
  (green_shirts * 100) / total_students

theorem percentage_green_shirts_correct
  (h1 : total_students = 800) 
  (h2 : blue_percentage = 45)
  (h3 : red_percentage = 23)
  (h4 : other_students = 136) : 
  percentage_green_shirts total_students blue_percentage red_percentage other_students = 15 :=
by
  sorry

end percentage_green_shirts_correct_l36_36467


namespace find_radius_of_third_circle_l36_36849

noncomputable def radius_of_third_circle_equals_shaded_region (r1 r2 r3 : ℝ) : Prop :=
  let area_large := Real.pi * (r2 ^ 2)
  let area_small := Real.pi * (r1 ^ 2)
  let area_shaded := area_large - area_small
  let area_third_circle := Real.pi * (r3 ^ 2)
  area_shaded = area_third_circle

theorem find_radius_of_third_circle (r1 r2 : ℝ) (r1_eq : r1 = 17) (r2_eq : r2 = 27) : ∃ r3 : ℝ, r3 = 10 * Real.sqrt 11 ∧ radius_of_third_circle_equals_shaded_region r1 r2 r3 := 
by
  sorry

end find_radius_of_third_circle_l36_36849


namespace largest_d_l36_36788

theorem largest_d (a b c d : ℤ) 
  (h₁ : a + 1 = b - 2) 
  (h₂ : a + 1 = c + 3) 
  (h₃ : a + 1 = d - 4) : 
  d > a ∧ d > b ∧ d > c := 
by 
  -- Here we would provide the proof, but for now we'll skip it
  sorry

end largest_d_l36_36788


namespace only_solution_is_2_3_7_l36_36433

theorem only_solution_is_2_3_7 (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h4 : c ∣ (a * b + 1)) (h5 : a ∣ (b * c + 1)) (h6 : b ∣ (c * a + 1)) :
  (a = 2 ∧ b = 3 ∧ c = 7) ∨ (a = 3 ∧ b = 7 ∧ c = 2) ∨ (a = 7 ∧ b = 2 ∧ c = 3) ∨
  (a = 2 ∧ b = 7 ∧ c = 3) ∨ (a = 7 ∧ b = 3 ∧ c = 2) ∨ (a = 3 ∧ b = 2 ∧ c = 7) :=
  sorry

end only_solution_is_2_3_7_l36_36433


namespace largest_integer_y_l36_36690

theorem largest_integer_y (y : ℤ) : 
  (∃ k : ℤ, (y^2 + 3*y + 10) = k * (y - 4)) → y ≤ 42 :=
sorry

end largest_integer_y_l36_36690


namespace find_b_l36_36282

theorem find_b (b : ℝ) (h : ∃ x : ℝ, x^2 + b*x - 35 = 0 ∧ x = -5) : b = -2 :=
by
  sorry

end find_b_l36_36282


namespace find_principal_amount_l36_36549

theorem find_principal_amount
  (P R T SI : ℝ) 
  (rate_condition : R = 12)
  (time_condition : T = 20)
  (interest_condition : SI = 2100) :
  SI = (P * R * T) / 100 → P = 875 :=
by
  sorry

end find_principal_amount_l36_36549


namespace range_a_l36_36056

open Set Real

-- Define the predicate p: real number x satisfies x^2 - 4ax + 3a^2 < 0, where a < 0
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0

-- Define the predicate q: real number x satisfies x^2 - x - 6 ≤ 0, or x^2 + 2x - 8 > 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- Define the complement sets
def not_p_set (a : ℝ) : Set ℝ := {x | ¬p x a}
def not_q_set : Set ℝ := {x | ¬q x}

-- Define p as necessary but not sufficient condition for q
def necessary_but_not_sufficient (a : ℝ) : Prop := 
  (not_q_set ⊆ not_p_set a) ∧ ¬(not_p_set a ⊆ not_q_set)

-- The main theorem to prove
theorem range_a : {a : ℝ | necessary_but_not_sufficient a} = {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4} :=
by
  sorry

end range_a_l36_36056


namespace average_tomatoes_per_day_l36_36689

theorem average_tomatoes_per_day :
  let t₁ := 120
  let t₂ := t₁ + 50
  let t₃ := 2 * t₂
  let t₄ := t₁ / 2
  (t₁ + t₂ + t₃ + t₄) / 4 = 172.5 := by
  sorry

end average_tomatoes_per_day_l36_36689


namespace find_other_number_l36_36396

/-- Given HCF(A, B), LCM(A, B), and a known A, proves the value of B. -/
theorem find_other_number (A B : ℕ) 
  (hcf : Nat.gcd A B = 16) 
  (lcm : Nat.lcm A B = 396) 
  (a_val : A = 36) : B = 176 :=
by
  sorry

end find_other_number_l36_36396


namespace A_is_guilty_l36_36716

-- Define the conditions
variables (A B C : Prop)  -- A, B, C are the propositions that represent the guilt of the individuals A, B, and C
variable  (car : Prop)    -- car represents the fact that the crime involved a car
variable  (C_never_alone : C → A)  -- C never commits a crime without A

-- Facts:
variables (crime_committed : A ∨ B ∨ C) -- the crime was committed by A, B, or C (or a combination)
variable  (B_knows_drive : B → car)     -- B knows how to drive

-- The proof goal: Show that A is guilty.
theorem A_is_guilty : A :=
sorry

end A_is_guilty_l36_36716


namespace compute_value_l36_36261

theorem compute_value : 12 - 4 * (5 - 10)^3 = 512 :=
by
  sorry

end compute_value_l36_36261


namespace sam_added_later_buckets_l36_36815

variable (initial_buckets : ℝ) (total_buckets : ℝ)

def buckets_added_later (initial_buckets total_buckets : ℝ) : ℝ :=
  total_buckets - initial_buckets

theorem sam_added_later_buckets :
  initial_buckets = 1 ∧ total_buckets = 9.8 → buckets_added_later initial_buckets total_buckets = 8.8 := by
  sorry

end sam_added_later_buckets_l36_36815


namespace find_k_values_l36_36825

noncomputable def problem (a b c d k : ℂ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  (a * k^3 + b * k^2 + c * k + d = 0) ∧
  (b * k^3 + c * k^2 + d * k + a = 0)

theorem find_k_values (a b c d k : ℂ) (h : problem a b c d k) : 
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end find_k_values_l36_36825


namespace matrix_B_cannot_be_obtained_from_matrix_A_l36_36863

def A : Matrix (Fin 5) (Fin 5) ℤ := ![
  ![1, 1, 1, 1, 1],
  ![1, 1, 1, -1, -1],
  ![1, -1, -1, 1, 1],
  ![1, -1, -1, -1, 1],
  ![1, 1, -1, 1, -1]
]

def B : Matrix (Fin 5) (Fin 5) ℤ := ![
  ![1, 1, 1, 1, 1],
  ![1, 1, 1, -1, -1],
  ![1, 1, -1, 1, -1],
  ![1, -1, -1, 1, 1],
  ![1, -1, 1, -1, 1]
]

theorem matrix_B_cannot_be_obtained_from_matrix_A :
  A.det ≠ B.det := by
  sorry

end matrix_B_cannot_be_obtained_from_matrix_A_l36_36863


namespace product_of_possible_values_of_N_l36_36120

theorem product_of_possible_values_of_N (N B D : ℤ) 
  (h1 : B = D - N) 
  (h2 : B + 10 - (D - 4) = 1 ∨ B + 10 - (D - 4) = -1) :
  N = 13 ∨ N = 15 → (13 * 15) = 195 :=
by sorry

end product_of_possible_values_of_N_l36_36120


namespace f_properties_l36_36642

noncomputable def f : ℕ → ℕ := sorry

theorem f_properties (f : ℕ → ℕ) :
  (∀ x y : ℕ, x > 0 → y > 0 → f (x * y) = f x + f y) →
  (f 10 = 16) →
  (f 40 = 24) →
  (f 3 = 5) →
  (f 800 = 44) :=
by
  intros h1 h2 h3 h4
  sorry

end f_properties_l36_36642


namespace number_of_lines_passing_through_four_points_l36_36024

-- Defining the three-dimensional points and conditions
structure Point3D where
  x : ℕ
  y : ℕ
  z : ℕ
  h1 : 1 ≤ x ∧ x ≤ 5
  h2 : 1 ≤ y ∧ y ≤ 5
  h3 : 1 ≤ z ∧ z ≤ 5

-- Define a valid line passing through four distinct points (Readonly accessors for the conditions)
def valid_line (p1 p2 p3 p4 : Point3D) : Prop := 
  sorry -- Define conditions for points to be collinear and distinct

-- Main theorem statement
theorem number_of_lines_passing_through_four_points : 
  ∃ (lines : ℕ), lines = 150 :=
sorry

end number_of_lines_passing_through_four_points_l36_36024


namespace find_a_l36_36439

theorem find_a (a : ℝ) (h : 3 ∈ ({1, a, a - 2} : Set ℝ)) : a = 5 :=
sorry

end find_a_l36_36439


namespace find_some_expression_l36_36454

noncomputable def problem_statement : Prop :=
  ∃ (some_expression : ℝ), 
    (5 + 7 / 12 = 6 - some_expression) ∧ 
    (some_expression = 0.4167)

theorem find_some_expression : problem_statement := 
  sorry

end find_some_expression_l36_36454


namespace number_of_six_digit_integers_l36_36782

-- Define the problem conditions
def digits := [1, 1, 3, 3, 7, 8]

-- State the theorem
theorem number_of_six_digit_integers : 
  (List.permutations digits).length = 180 := 
by sorry

end number_of_six_digit_integers_l36_36782


namespace sum_of_reciprocals_is_two_l36_36840

variable (x y : ℝ)
variable (h1 : x + y = 50)
variable (h2 : x * y = 25)

theorem sum_of_reciprocals_is_two (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1/x + 1/y) = 2 :=
by
  sorry

end sum_of_reciprocals_is_two_l36_36840


namespace necessary_but_not_sufficient_condition_l36_36909

variable {a b : ℝ}

theorem necessary_but_not_sufficient_condition
    (h1 : a ≠ 0)
    (h2 : b ≠ 0) :
    (a^2 + b^2 ≥ 2 * a * b) → 
    (¬(a^2 + b^2 ≥ 2 * a * b) → ¬(a / b + b / a ≥ 2)) ∧ 
    ((a / b + b / a ≥ 2) → (a^2 + b^2 ≥ 2 * a * b)) :=
sorry

end necessary_but_not_sufficient_condition_l36_36909


namespace solution_mn_l36_36595

theorem solution_mn (m n : ℤ) (h1 : |m| = 4) (h2 : |n| = 5) (h3 : n < 0) : m + n = -1 ∨ m + n = -9 := 
by
  sorry

end solution_mn_l36_36595


namespace required_amount_of_water_l36_36665

/-- 
Given:
- A solution of 12 ounces with 60% alcohol,
- A desired final concentration of 40% alcohol,

Prove:
- The required amount of water to add is 6 ounces.
-/
theorem required_amount_of_water 
    (original_volume : ℚ)
    (initial_concentration : ℚ)
    (desired_concentration : ℚ)
    (final_volume : ℚ)
    (amount_of_water : ℚ)
    (h1 : original_volume = 12)
    (h2 : initial_concentration = 0.6)
    (h3 : desired_concentration = 0.4)
    (h4 : final_volume = original_volume + amount_of_water)
    (h5 : amount_of_alcohol = original_volume * initial_concentration)
    (h6 : desired_amount_of_alcohol = final_volume * desired_concentration)
    (h7 : amount_of_alcohol = desired_amount_of_alcohol) : 
  amount_of_water = 6 := 
sorry

end required_amount_of_water_l36_36665


namespace equation_solution_l36_36575

theorem equation_solution (x : ℝ) (h : x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) :
  (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 4) ↔ 
  (x = 3 + 2 * Real.sqrt 5 ∨ x = 3 - 2 * Real.sqrt 5) := 
sorry

end equation_solution_l36_36575


namespace inscribed_circle_radius_integer_l36_36345

theorem inscribed_circle_radius_integer (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ (r : ℤ), r = (a + b - c) / 2 := by
  sorry

end inscribed_circle_radius_integer_l36_36345


namespace number_to_multiply_l36_36926

theorem number_to_multiply (a b x : ℝ) (h1 : x * a = 4 * b) (h2 : a * b ≠ 0) (h3 : a / 4 = b / 3) : x = 3 :=
sorry

end number_to_multiply_l36_36926


namespace train_length_l36_36982

theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 300)
  (h2 : time = 33) : (speed * 1000 / 3600) * time = 2750 := by
  sorry

end train_length_l36_36982


namespace total_fencing_cost_l36_36578

def side1 : ℕ := 34
def side2 : ℕ := 28
def side3 : ℕ := 45
def side4 : ℕ := 50
def side5 : ℕ := 55

def cost1_per_meter : ℕ := 2
def cost2_per_meter : ℕ := 2
def cost3_per_meter : ℕ := 3
def cost4_per_meter : ℕ := 3
def cost5_per_meter : ℕ := 4

def total_cost : ℕ :=
  side1 * cost1_per_meter +
  side2 * cost2_per_meter +
  side3 * cost3_per_meter +
  side4 * cost4_per_meter +
  side5 * cost5_per_meter

theorem total_fencing_cost : total_cost = 629 := by
  sorry

end total_fencing_cost_l36_36578


namespace find_square_number_divisible_by_six_l36_36135

theorem find_square_number_divisible_by_six :
  ∃ x : ℕ, (∃ k : ℕ, x = k^2) ∧ x % 6 = 0 ∧ 24 < x ∧ x < 150 ∧ (x = 36 ∨ x = 144) :=
by {
  sorry
}

end find_square_number_divisible_by_six_l36_36135


namespace arithmetic_mean_of_multiples_of_6_l36_36388

/-- The smallest three-digit multiple of 6 is 102. -/
def smallest_multiple_of_6 : ℕ := 102

/-- The largest three-digit multiple of 6 is 996. -/
def largest_multiple_of_6 : ℕ := 996

/-- The common difference in the arithmetic sequence of multiples of 6 is 6. -/
def common_difference_of_sequence : ℕ := 6

/-- The number of terms in the arithmetic sequence of three-digit multiples of 6. -/
def number_of_terms : ℕ := (largest_multiple_of_6 - smallest_multiple_of_6) / common_difference_of_sequence + 1

/-- The sum of the arithmetic sequence of three-digit multiples of 6. -/
def sum_of_sequence : ℕ := number_of_terms * (smallest_multiple_of_6 + largest_multiple_of_6) / 2

/-- The arithmetic mean of all positive three-digit multiples of 6 is 549. -/
theorem arithmetic_mean_of_multiples_of_6 : 
  let mean := sum_of_sequence / number_of_terms
  mean = 549 :=
by
  sorry

end arithmetic_mean_of_multiples_of_6_l36_36388


namespace angle_BAC_l36_36747

theorem angle_BAC
  (elevation_angle_B_from_A : ℝ)
  (depression_angle_C_from_A : ℝ)
  (h₁ : elevation_angle_B_from_A = 60)
  (h₂ : depression_angle_C_from_A = 70) :
  elevation_angle_B_from_A + depression_angle_C_from_A = 130 :=
by
  sorry

end angle_BAC_l36_36747


namespace line_divides_circle_1_3_l36_36905

noncomputable def circle_equidistant_from_origin : Prop := 
  ∃ l : ℝ → ℝ, ∀ x y : ℝ, ((x-1)^2 + (y-1)^2 = 2) → 
                     (l 0 = 0 ∧ (l x = l y) ∧ 
                     ((x = 0) ∨ (y = 0)))

theorem line_divides_circle_1_3 (x y : ℝ) : 
  (x - 1)^2 + (y - 1)^2 = 2 → 
  (x = 0 ∨ y = 0) :=
by
  sorry

end line_divides_circle_1_3_l36_36905


namespace probability_of_selecting_cooking_l36_36104

-- Define a type representing the courses.
inductive Course
| planting : Course
| cooking : Course
| pottery : Course
| carpentry : Course

-- Define the set of all courses
def all_courses : Finset Course := {Course.planting, Course.cooking, Course.pottery, Course.carpentry}

-- The condition that Xiao Ming randomly selects one of the four courses
def uniform_probability (s : Finset Course) (a : Course) : ℚ := 1 / s.card

-- Prove that the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking : uniform_probability all_courses Course.cooking = 1 / 4 :=
sorry

end probability_of_selecting_cooking_l36_36104


namespace face_card_then_number_card_prob_l36_36410

-- Definitions from conditions
def num_cards := 52
def num_face_cards := 12
def num_number_cards := 40
def total_ways_to_pick_two_cards := 52 * 51

-- Theorem statement
theorem face_card_then_number_card_prob : 
  (num_face_cards * num_number_cards) / total_ways_to_pick_two_cards = (40 : ℚ) / 221 :=
by
  sorry

end face_card_then_number_card_prob_l36_36410


namespace field_dimensions_l36_36293

theorem field_dimensions (W L : ℕ) (h1 : L = 2 * W) (h2 : 2 * L + 2 * W = 600) : W = 100 ∧ L = 200 :=
sorry

end field_dimensions_l36_36293


namespace find_minimum_value_l36_36768

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := x^2 + a * |x - 1| + 1

-- The statement of the proof problem
theorem find_minimum_value (a : ℝ) (h : a ≥ 0) :
  (a = 0 → ∀ x, f x a ≥ 1 ∧ ∃ x, f x a = 1) ∧
  ((0 < a ∧ a < 2) → ∀ x, f x a ≥ -a^2 / 4 + a + 1 ∧ ∃ x, f x a = -a^2 / 4 + a + 1) ∧
  (a ≥ 2 → ∀ x, f x a ≥ 2 ∧ ∃ x, f x a = 2) := 
by
  sorry

end find_minimum_value_l36_36768


namespace part1_part2_l36_36148

theorem part1 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (hA : A + B + C = π) 
  (ha : a = 2) 
  (hcosC : Real.cos C = -1 / 4) 
  (hsinA_sinB : Real.sin A = 2 * Real.sin B) : b = 1 ∧ c = Real.sqrt 6 := 
  sorry

theorem part2
  (A B C : ℝ) 
  (a b c : ℝ) 
  (hA : A + B + C = π) 
  (ha : a = 2) 
  (hcosC : Real.cos C = -1 / 4)
  (hcosA_minus_pi_div_4 : Real.cos (A - π / 4) = 4 / 5) : c = 5 * Real.sqrt 30 / 2 := 
  sorry

end part1_part2_l36_36148


namespace smallest_n_for_perfect_square_and_cube_l36_36706

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, 0 < n ∧ (∃ a1 b1 : ℕ, 4 * n = a1 ^ 2 ∧ 5 * n = b1 ^ 3 ∧ n = 50) :=
begin
  use 50,
  split,
  { norm_num, },
  { use [10, 5],
    split,
    { norm_num, },
    { split, 
      { norm_num, },
      { refl, }, },
  },
  sorry
end

end smallest_n_for_perfect_square_and_cube_l36_36706


namespace range_m_if_B_subset_A_range_m_if_A_inter_B_empty_l36_36287

variable (m : ℝ)

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def set_B : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Problem 1: Prove the range of m if B ⊆ A is (-∞, 3]
theorem range_m_if_B_subset_A : (set_B m ⊆ set_A) ↔ m ≤ 3 := sorry

-- Problem 2: Prove the range of m if A ∩ B = ∅ is m < 2 or m > 4
theorem range_m_if_A_inter_B_empty : (set_A ∩ set_B m = ∅) ↔ m < 2 ∨ m > 4 := sorry

end range_m_if_B_subset_A_range_m_if_A_inter_B_empty_l36_36287


namespace number_of_ways_to_divide_friends_l36_36784

theorem number_of_ways_to_divide_friends :
  let friends := 8
  let teams := 4
  (teams ^ friends) = 65536 := by
  sorry

end number_of_ways_to_divide_friends_l36_36784


namespace third_number_is_32_l36_36964

theorem third_number_is_32 (A B C : ℕ) 
  (hA : A = 24) (hB : B = 36) 
  (hHCF : Nat.gcd (Nat.gcd A B) C = 32) 
  (hLCM : Nat.lcm (Nat.lcm A B) C = 1248) : 
  C = 32 := 
sorry

end third_number_is_32_l36_36964


namespace find_f_ln2_l36_36602

variable (f : ℝ → ℝ)

-- Condition: f is an odd function
axiom odd_fn : ∀ x : ℝ, f (-x) = -f x

-- Condition: f(x) = e^(-x) - 2 for x < 0
axiom def_fn : ∀ x : ℝ, x < 0 → f x = Real.exp (-x) - 2

-- Problem: Find f(ln 2)
theorem find_f_ln2 : f (Real.log 2) = 0 := by
  sorry

end find_f_ln2_l36_36602


namespace max_tied_teams_for_most_wins_l36_36038

theorem max_tied_teams_for_most_wins 
  (n : ℕ) 
  (h₀ : n = 6)
  (total_games : ℕ := n * (n - 1) / 2)
  (game_result : Π (i j : ℕ), i ≠ j → (0 = 1 → false) ∨ (1 = 1))
  (rank_by_wins : ℕ → ℕ) : true := sorry

end max_tied_teams_for_most_wins_l36_36038


namespace remainder_div_197_l36_36226

theorem remainder_div_197 (x q : ℕ) (h_pos : 0 < x) (h_div : 100 = q * x + 3) : 197 % x = 3 :=
sorry

end remainder_div_197_l36_36226


namespace tear_paper_l36_36385

theorem tear_paper (n : ℕ) : 1 + 3 * n ≠ 2007 :=
by
  sorry

end tear_paper_l36_36385


namespace cos_pi_plus_alpha_l36_36153

-- Define the angle α and conditions given
variable (α : Real) (h1 : 0 < α) (h2 : α < π/2)

-- Given condition sine of α
variable (h3 : Real.sin α = 4/5)

-- Define the cosine identity to prove the assertion
theorem cos_pi_plus_alpha (h1 : 0 < α) (h2 : α < π/2) (h3 : Real.sin α = 4/5) :
  Real.cos (π + α) = -3/5 :=
sorry

end cos_pi_plus_alpha_l36_36153


namespace derivative_value_at_pi_over_2_l36_36448

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem derivative_value_at_pi_over_2 : deriv f (Real.pi / 2) = -1 :=
by
  sorry

end derivative_value_at_pi_over_2_l36_36448


namespace time_diff_is_6_l36_36426

-- Define the speeds for the different sails
def speed_of_large_sail : ℕ := 50
def speed_of_small_sail : ℕ := 20

-- Define the distance of the trip
def trip_distance : ℕ := 200

-- Calculate the time for each sail
def time_large_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed
def time_small_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Define the time difference
def time_difference (distance : ℕ) (speed_large : ℕ) (speed_small : ℕ) : ℕ := 
  (distance / speed_small) - (distance / speed_large)

-- Prove that the time difference between the large and small sails is 6 hours
theorem time_diff_is_6 : time_difference trip_distance speed_of_large_sail speed_of_small_sail = 6 := by
  -- useful := time_difference trip_distance speed_of_large_sail speed_of_small_sail,
  -- change useful with 6,
  sorry

end time_diff_is_6_l36_36426


namespace fraction_evaluation_l36_36514

theorem fraction_evaluation : (1 / 2) + (1 / 2 * 1 / 2) = 3 / 4 := by
  sorry

end fraction_evaluation_l36_36514


namespace absolute_value_simplify_l36_36789

variable (a : ℝ)

theorem absolute_value_simplify
  (h : a < 3) : |a - 3| = 3 - a := sorry

end absolute_value_simplify_l36_36789


namespace glycerin_percentage_l36_36990

theorem glycerin_percentage (x : ℝ) 
  (h1 : 100 * 0.75 = 75)
  (h2 : 75 + 75 = 100)
  (h3 : 75 * 0.30 + (x/100) * 75 = 75) : x = 70 :=
by
  sorry

end glycerin_percentage_l36_36990


namespace domain_of_sqrt_l36_36364

theorem domain_of_sqrt (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by sorry

end domain_of_sqrt_l36_36364


namespace sum_of_digits_18_to_21_l36_36086

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_18_to_21 : 
  (sum_digits 18 + sum_digits 19 + sum_digits 20 + sum_digits 21) = 24 := 
by 
  sorry

end sum_of_digits_18_to_21_l36_36086


namespace count_valid_pairs_l36_36618

theorem count_valid_pairs : 
  ∃ n : ℕ, n = 5 ∧ 
  ∀ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 40 →
  (5^j - 2^i) % 1729 = 0 →
  i = 0 ∧ j = 36 ∨ 
  i = 1 ∧ j = 37 ∨ 
  i = 2 ∧ j = 38 ∨ 
  i = 3 ∧ j = 39 ∨ 
  i = 4 ∧ j = 40 :=
by
  sorry

end count_valid_pairs_l36_36618


namespace books_after_donation_l36_36419

/-- 
  Total books Boris and Cameron have together after donating some books.
 -/
theorem books_after_donation :
  let B : ℕ := 24 in   -- Initial books Boris has
  let C : ℕ := 30 in   -- Initial books Cameron has
  let B_donated := B / 4 in  -- Boris donates a fourth of his books
  let C_donated := C / 3 in  -- Cameron donates a third of his books
  B - B_donated + (C - C_donated) = 38 :=  -- After donating, the total books
by
  sorry

end books_after_donation_l36_36419


namespace fraction_c_over_d_l36_36841

-- Assume that we have a polynomial equation ax^3 + bx^2 + cx + d = 0 with roots 1, 2, 3
def polynomial (a b c d x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

-- The roots of the polynomial are 1, 2, 3
def roots (a b c d : ℝ) : Prop := polynomial a b c d 1 ∧ polynomial a b c d 2 ∧ polynomial a b c d 3

-- Vieta's formulas give us the relation for c and d in terms of the roots
theorem fraction_c_over_d (a b c d : ℝ) (h : roots a b c d) : c / d = -11 / 6 :=
sorry

end fraction_c_over_d_l36_36841


namespace square_root_area_ratio_l36_36335

theorem square_root_area_ratio 
  (side_C : ℝ) (side_D : ℝ)
  (hC : side_C = 45) 
  (hD : side_D = 60) : 
  Real.sqrt ((side_C^2) / (side_D^2)) = 3 / 4 := by
  -- proof goes here
  sorry

end square_root_area_ratio_l36_36335


namespace find_middle_number_l36_36895

theorem find_middle_number (a b c d x e f g : ℝ) 
  (h1 : (a + b + c + d + x + e + f + g) / 8 = 7)
  (h2 : (a + b + c + d + x) / 5 = 6)
  (h3 : (x + e + f + g + d) / 5 = 9) :
  x = 9.5 := 
by 
  sorry

end find_middle_number_l36_36895


namespace juanita_spends_more_l36_36780

-- Define the expenditures
def grant_yearly_expenditure : ℝ := 200.00

def juanita_weekday_expenditure : ℝ := 0.50

def juanita_sunday_expenditure : ℝ := 2.00

def weeks_per_year : ℕ := 52

-- Given conditions translated to Lean
def juanita_weekly_expenditure : ℝ :=
  (juanita_weekday_expenditure * 6) + juanita_sunday_expenditure

def juanita_yearly_expenditure : ℝ :=
  juanita_weekly_expenditure * weeks_per_year

-- The statement we need to prove
theorem juanita_spends_more : (juanita_yearly_expenditure - grant_yearly_expenditure) = 60.00 :=
by
  sorry

end juanita_spends_more_l36_36780


namespace am_gm_example_l36_36645

variable {x y z : ℝ}

theorem am_gm_example (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x / y + y / z + z / x + y / x + z / y + x / z ≥ 6 :=
sorry

end am_gm_example_l36_36645


namespace polynomial_divisible_by_24_l36_36655

theorem polynomial_divisible_by_24 (n : ℤ) : 24 ∣ (n^4 + 6 * n^3 + 11 * n^2 + 6 * n) :=
sorry

end polynomial_divisible_by_24_l36_36655


namespace dispatch_plans_l36_36111

-- Define the problem conditions
def num_teachers : ℕ := 8
def num_selected : ℕ := 4

-- Teacher constraints
def constraint1 (A B : Prop) : Prop := ¬(A ∧ B)  -- A and B cannot go together
def constraint2 (A C : Prop) : Prop := (A ∧ C) ∨ (¬A ∧ ¬C)  -- A and C can only go together or not at all

theorem dispatch_plans (A B C: Prop) 
                      (h1: constraint1 A B)
                      (h2: constraint2 A C)
                      (h3: num_teachers = 8)  
                      (h4: num_selected = 4) 
                      : num_ways_to_dispatch = 600 :=
by 
  sorry

end dispatch_plans_l36_36111


namespace counterexample_exists_l36_36479

-- Define a function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- State the theorem equivalently in Lean
theorem counterexample_exists : (sum_of_digits 33 % 6 = 0) ∧ (33 % 6 ≠ 0) := by
  sorry

end counterexample_exists_l36_36479


namespace eq_x4_inv_x4_l36_36172

theorem eq_x4_inv_x4 (x : ℝ) (h : x^2 + (1 / x^2) = 2) : 
  x^4 + (1 / x^4) = 2 := 
by 
  sorry

end eq_x4_inv_x4_l36_36172


namespace choose_officers_ways_l36_36493

theorem choose_officers_ways :
  let members := 12
  let vp_candidates := 4
  let remaining_after_president := members - 1
  let remaining_after_vice_president := remaining_after_president - 1
  let remaining_after_secretary := remaining_after_vice_president - 1
  let remaining_after_treasurer := remaining_after_secretary - 1
  (members * vp_candidates * (remaining_after_vice_president) *
   (remaining_after_secretary) * (remaining_after_treasurer)) = 34560 := by
  -- Calculation here
  sorry

end choose_officers_ways_l36_36493


namespace probability_both_blue_buttons_l36_36935

theorem probability_both_blue_buttons :
  let initial_red_C := 6
  let initial_blue_C := 12
  let initial_total_C := initial_red_C + initial_blue_C
  let remaining_fraction_C := 2 / 3
  let remaining_total_C := initial_total_C * remaining_fraction_C
  let removed_buttons := initial_total_C - remaining_total_C
  let removed_red := removed_buttons / 2
  let removed_blue := removed_buttons / 2
  let remaining_blue_C := initial_blue_C - removed_blue
  let total_remaining_C := remaining_total_C
  let probability_blue_C := remaining_blue_C / total_remaining_C
  let probability_blue_D := removed_blue / removed_buttons
  probability_blue_C * probability_blue_D = 3 / 8 :=
by
  sorry

end probability_both_blue_buttons_l36_36935


namespace puppies_given_l36_36414

-- Definitions of the initial and left numbers of puppies
def initial_puppies : ℕ := 7
def left_puppies : ℕ := 2

-- Theorem stating that the number of puppies given to friends is the difference
theorem puppies_given : initial_puppies - left_puppies = 5 := by
  sorry -- Proof not required, so we use sorry

end puppies_given_l36_36414


namespace ratio_of_expenditures_l36_36963

-- Let us define the conditions and rewrite the proof problem statement.
theorem ratio_of_expenditures
  (income_P1 income_P2 expenditure_P1 expenditure_P2 : ℝ)
  (H1 : income_P1 / income_P2 = 5 / 4)
  (H2 : income_P1 = 5000)
  (H3 : income_P1 - expenditure_P1 = 2000)
  (H4 : income_P2 - expenditure_P2 = 2000) :
  expenditure_P1 / expenditure_P2 = 3 / 2 :=
sorry

end ratio_of_expenditures_l36_36963


namespace number_of_valid_permutations_l36_36726

open Finset

-- Define the finite set of permutations of the numbers 1 to 6
def permutations_6 : Finset (Fin 6 → Fin 6) :=
  univ.filter (λ σ : (Fin 6 → Fin 6), 
    σ 0 ≠ 1 ∧ σ 2 ≠ 3 ∧ σ 4 ≠ 5 ∧ σ 0 < σ 2 ∧ σ 2 < σ 4)

-- Prove that the number of such permutations is 30
theorem number_of_valid_permutations : 
  (permutations_6.card = 30) :=
by 
  sorry

end number_of_valid_permutations_l36_36726


namespace radius_of_inscribed_circle_is_integer_l36_36356

-- Define variables and conditions
variables (a b c : ℕ)
variables (h1 : c^2 = a^2 + b^2)

-- Define the radius r
noncomputable def r := (a + b - c) / 2

-- Proof statement
theorem radius_of_inscribed_circle_is_integer 
  (h2 : c^2 = a^2 + b^2)
  (h3 : (r : ℤ) = (a + b - c) / 2) : 
  ∃ r : ℤ, r = (a + b - c) / 2 :=
by {
   -- The proof will be provided here
   sorry
}

end radius_of_inscribed_circle_is_integer_l36_36356


namespace smaller_angle_of_parallelogram_l36_36033

theorem smaller_angle_of_parallelogram (x : ℝ) (h : x + 3 * x = 180) : x = 45 :=
sorry

end smaller_angle_of_parallelogram_l36_36033


namespace quadrilateral_midpoints_area_l36_36185

-- We set up the geometric context and define the problem in Lean 4.

noncomputable def area_of_midpoint_quadrilateral
  (AB CD : ℝ) (AD BC : ℝ)
  (h_AB_CD : AB = 15) (h_CD_AB : CD = 15)
  (h_AD_BC : AD = 10) (h_BC_AD : BC = 10)
  (mid_AB : Prop) (mid_BC : Prop) (mid_CD : Prop) (mid_DA : Prop) : ℝ :=
  37.5

-- The theorem statement validating the area of the quadrilateral.
theorem quadrilateral_midpoints_area (AB CD AD BC : ℝ) 
  (h_AB_CD : AB = 15) (h_CD_AB : CD = 15)
  (h_AD_BC : AD = 10) (h_BC_AD : BC = 10)
  (mid_AB : Prop) (mid_BC : Prop) (mid_CD : Prop) (mid_DA : Prop) :
  area_of_midpoint_quadrilateral AB CD AD BC h_AB_CD h_CD_AB h_AD_BC h_BC_AD mid_AB mid_BC mid_CD mid_DA = 37.5 :=
by 
  sorry  -- Proof is omitted.

end quadrilateral_midpoints_area_l36_36185


namespace simplify_expression_l36_36197

variable (m n : ℝ)

theorem simplify_expression : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end simplify_expression_l36_36197


namespace intersection_A_B_eq_C_l36_36317

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
noncomputable def C : Set ℝ := {0, 1, 2}

theorem intersection_A_B_eq_C : (A : Set ℝ) ∩ B = C :=
by {
  sorry
}

end intersection_A_B_eq_C_l36_36317


namespace verify_monomial_properties_l36_36979

def monomial : ℚ := -3/5 * (1:ℚ)^1 * (2:ℚ)^2

def coefficient (m : ℚ) : ℚ := -3/5  -- The coefficient of the monomial
def degree (m : ℚ) : ℕ := 3          -- The degree of the monomial

theorem verify_monomial_properties :
  coefficient monomial = -3/5 ∧ degree monomial = 3 :=
by
  sorry

end verify_monomial_properties_l36_36979


namespace black_friday_sales_projection_l36_36491

theorem black_friday_sales_projection (sold_now : ℕ) (increment : ℕ) (years : ℕ) 
  (h_now : sold_now = 327) (h_inc : increment = 50) (h_years : years = 3) : 
  let sold_three_years := sold_now + 3 * increment in
  sold_three_years = 477 := 
by
  -- Definitions according to the conditions
  have h1 : sold_now = 327 := h_now
  have h2 : increment = 50 := h_inc
  have h3 : years = 3 := h_years

  -- Calculation based on definitions
  have h_sold_next_year := sold_now + increment
  have h_sold_second_year := h_sold_next_year + increment
  have h_sold_third_year := h_sold_second_year + increment
  
  -- Haven't elaborated on proof steps as the problem requires the statement only
  sorry

end black_friday_sales_projection_l36_36491


namespace product_eq_sum_l36_36520

variables {x y : ℝ}

theorem product_eq_sum (h : x * y = x + y) (h_ne : y ≠ 1) : x = y / (y - 1) :=
sorry

end product_eq_sum_l36_36520


namespace remainder_by_19_l36_36873

theorem remainder_by_19 (N : ℤ) (k : ℤ) (h : N = 779 * k + 47) : N % 19 = 9 :=
by sorry

end remainder_by_19_l36_36873


namespace insects_in_lab_l36_36882

theorem insects_in_lab (total_legs number_of_legs_per_insect : ℕ) (h1 : total_legs = 36) (h2 : number_of_legs_per_insect = 6) : (total_legs / number_of_legs_per_insect) = 6 :=
by
  sorry

end insects_in_lab_l36_36882


namespace cone_surface_area_is_correct_l36_36550

noncomputable def cone_surface_area (central_angle_degrees : ℝ) (sector_area : ℝ) : ℝ :=
  if central_angle_degrees = 120 ∧ sector_area = 3 * Real.pi then 4 * Real.pi else 0

theorem cone_surface_area_is_correct :
  cone_surface_area 120 (3 * Real.pi) = 4 * Real.pi :=
by
  -- proof would go here
  sorry

end cone_surface_area_is_correct_l36_36550


namespace nat_representable_as_sequence_or_difference_l36_36337

theorem nat_representable_as_sequence_or_difference
  (a : ℕ → ℕ)
  (h1 : ∀ n, 0 < a n)
  (h2 : ∀ n, a n < 2 * n) :
  ∀ m : ℕ, ∃ k l : ℕ, k ≠ l ∧ (m = a k ∨ m = a k - a l) :=
by
  sorry

end nat_representable_as_sequence_or_difference_l36_36337


namespace team_E_not_played_against_team_B_l36_36555

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

end team_E_not_played_against_team_B_l36_36555


namespace no_real_roots_l36_36264

theorem no_real_roots (k : ℝ) (h : k ≠ 0) : ¬∃ x : ℝ, x^2 + k * x + 3 * k^2 = 0 :=
by
  sorry

end no_real_roots_l36_36264


namespace coconut_grove_average_yield_l36_36461

theorem coconut_grove_average_yield :
  ∀ (x : ℕ),
  40 * (x + 2) + 120 * x + 180 * (x - 2) = 100 * 3 * x →
  x = 7 :=
by
  intro x
  intro h
  /- sorry proof -/
  sorry

end coconut_grove_average_yield_l36_36461


namespace product_of_integers_l36_36099

theorem product_of_integers (x y : ℕ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 40) : x * y = 99 :=
by {
  sorry
}

end product_of_integers_l36_36099


namespace chris_leftover_money_l36_36422

def chris_will_have_leftover : Prop :=
  let video_game_cost := 60
  let candy_cost := 5
  let hourly_wage := 8
  let hours_worked := 9
  let total_earned := hourly_wage * hours_worked
  let total_cost := video_game_cost + candy_cost
  let leftover := total_earned - total_cost
  leftover = 7

theorem chris_leftover_money : chris_will_have_leftover := 
  by
    sorry

end chris_leftover_money_l36_36422


namespace tan_30_deg_l36_36734

theorem tan_30_deg : 
  let θ := (30 : ℝ) * (Real.pi / 180)
  in Real.sin θ = 1 / 2 ∧ Real.cos θ = Real.sqrt 3 / 2 → Real.tan θ = Real.sqrt 3 / 3 :=
by
  intro h
  let th := θ
  have h1 : Real.sin th = 1 / 2 := And.left h
  have h2 : Real.cos th = Real.sqrt 3 / 2 := And.right h
  sorry

end tan_30_deg_l36_36734


namespace prob_at_least_one_even_l36_36540

theorem prob_at_least_one_even :
  let events := [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)] in
  let success_events := [(1,2), (2,1), (2,2), (2,3), (3,2)] in
  (success_events.length / events.length : ℚ) = 5 / 9 :=
by
  sorry

end prob_at_least_one_even_l36_36540


namespace line_through_points_l36_36032

theorem line_through_points (a b : ℝ) (h1 : 3 = a * 2 + b) (h2 : 19 = a * 6 + b) :
  a - b = 9 :=
sorry

end line_through_points_l36_36032


namespace solve_for_x_l36_36955

theorem solve_for_x : ∃ x : ℚ, 7 * (4 * x + 3) - 3 = -3 * (2 - 5 * x) + 5 * x / 2 ∧ x = -16 / 7 := by
  sorry

end solve_for_x_l36_36955


namespace dart_lands_in_center_square_l36_36552

theorem dart_lands_in_center_square (s : ℝ) (h : 0 < s) :
  let center_square_area := (s / 2) ^ 2
  let triangle_area := 1 / 2 * (s / 2) ^ 2
  let total_triangle_area := 4 * triangle_area
  let total_board_area := center_square_area + total_triangle_area
  let probability := center_square_area / total_board_area
  probability = 1 / 3 :=
by
  sorry

end dart_lands_in_center_square_l36_36552


namespace total_weight_gain_l36_36728

def orlando_gained : ℕ := 5

def jose_gained (orlando : ℕ) : ℕ :=
  2 * orlando + 2

def fernando_gained (jose : ℕ) : ℕ :=
  jose / 2 - 3

theorem total_weight_gain (O J F : ℕ) 
  (ho : O = orlando_gained) 
  (hj : J = jose_gained O) 
  (hf : F = fernando_gained J) :
  O + J + F = 20 :=
by
  sorry

end total_weight_gain_l36_36728


namespace fraction_of_juniors_l36_36532

theorem fraction_of_juniors (J S : ℕ) (h1 : 0 < J) (h2 : 0 < S) (h3 : J = (4 / 3) * S) :
  (J : ℚ) / (J + S) = 4 / 7 :=
by
  sorry

end fraction_of_juniors_l36_36532


namespace exponent_power_rule_l36_36978

theorem exponent_power_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 :=
by sorry

end exponent_power_rule_l36_36978


namespace smallest_n_l36_36696

theorem smallest_n (n : ℕ) : (∃ (m1 m2 : ℕ), 4 * n = m1^2 ∧ 5 * n = m2^3) ↔ n = 500 := 
begin
  sorry
end

end smallest_n_l36_36696


namespace limit_proof_l36_36339

theorem limit_proof :
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → | (3 * x^2 - 5 * x - 2) / (x - 2) - 7 | < ε) :=
begin
  assume ε ε_pos,
  existsi ε / 3,
  use div_pos ε_pos (by norm_num),
  assume x hx,
  have h_denom : x ≠ 2, from λ h, hx.1 (by rwa h),
  calc
  |(3 * x^2 - 5 * x - 2) / (x - 2) - 7|
      = |((3 * x + 1) * (x - 2) / (x - 2)) - 7| : by {
        ring_exp,
        exact (by norm_num : (3 : ℝ) ≠ 0 ),
      }
  ... = |3 * x + 1 - 7| : by rw [(by_ring_exp * (x - 2)).symm], (ne_of_apply_ne (3 * x + 1) (by norm_num)).elim]
  ... = |3 * (x - 2)| : by ring_exp
  ... = 3 * |x - 2| : abs_mul,
  exact (mul_lt_iff_lt_one_left (by norm_num)).2 (calc
      |x - 2| < ε / 3 : hx.2
      ... = ε / 3
    )
end

end limit_proof_l36_36339


namespace Ki_tae_pencils_l36_36475

theorem Ki_tae_pencils (P B : ℤ) (h1 : P + B = 12) (h2 : 1000 * P + 1300 * B = 15000) : P = 2 :=
sorry

end Ki_tae_pencils_l36_36475


namespace raisin_cost_fraction_l36_36886

theorem raisin_cost_fraction
  (R : ℝ)                -- cost of a pound of raisins
  (cost_nuts : ℝ := 2 * R)  -- cost of a pound of nuts
  (cost_raisins : ℝ := 3 * R)  -- cost of 3 pounds of raisins
  (cost_nuts_total : ℝ := 4 * cost_nuts)  -- cost of 4 pounds of nuts
  (total_cost : ℝ := cost_raisins + cost_nuts_total)  -- total cost of the mixture
  (fraction_of_raisins : ℝ := cost_raisins / total_cost)  -- fraction of cost of raisins
  : fraction_of_raisins = 3 / 11 := 
by
  sorry

end raisin_cost_fraction_l36_36886


namespace find_number_l36_36063

theorem find_number (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
sorry

end find_number_l36_36063


namespace total_seating_arrangements_l36_36006

theorem total_seating_arrangements : 
  ∀ (n k : ℕ), n = 5 → k = 3 → 
  let comb := Nat.choose n k in
  let perm1 := Nat.factorial k in
  let perm2 := Nat.factorial (n - k) in
  2 * comb * perm1 * perm2 = 240 :=
by
  intros n k hn hk comb perm1 perm2
  have h1 : comb = Nat.choose n k := by rfl
  have h2 : perm1 = Nat.factorial k := by rfl
  have h3 : perm2 = Nat.factorial (n - k) := by rfl
  rw [hn, hk, Nat.choose, Nat.factorial]
  norm_num
  sorry

end total_seating_arrangements_l36_36006


namespace current_age_l36_36799

theorem current_age (A B S Y : ℕ) 
  (h1: Y = 4) 
  (h2: S = 2 * Y) 
  (h3: B = S + 3) 
  (h4: A + 10 = 2 * (B + 10))
  (h5: A + 10 = 3 * (S + 10))
  (h6: A + 10 = 4 * (Y + 10)) 
  (h7: (A + 10) + (B + 10) + (S + 10) + (Y + 10) = 88) : 
  A = 46 :=
sorry

end current_age_l36_36799


namespace find_range_of_a_l36_36608

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp (2 * x) - a * x

theorem find_range_of_a (a : ℝ) :
  (∀ x > 0, f x a > a * x^2 + 1) → a ≤ 2 :=
by
  sorry

end find_range_of_a_l36_36608


namespace theodore_total_monthly_earnings_l36_36088

-- Define the conditions
def stone_statues_per_month := 10
def wooden_statues_per_month := 20
def cost_per_stone_statue := 20
def cost_per_wooden_statue := 5
def tax_rate := 0.10

-- Calculate earnings from stone and wooden statues
def earnings_from_stone_statues := stone_statues_per_month * cost_per_stone_statue
def earnings_from_wooden_statues := wooden_statues_per_month * cost_per_wooden_statue

-- Total earnings before taxes
def total_earnings := earnings_from_stone_statues + earnings_from_wooden_statues

-- Taxes paid
def taxes_paid := total_earnings * tax_rate

-- Total earnings after taxes
def total_earnings_after_taxes := total_earnings - taxes_paid

-- Theorem stating the total earnings after taxes
theorem theodore_total_monthly_earnings : total_earnings_after_taxes = 270 := sorry

end theodore_total_monthly_earnings_l36_36088


namespace continuous_distribution_function_of_Y_l36_36537

noncomputable def problem (X : ℕ → ℝ) (α : ℝ) : Prop :=
  0 < α ∧ α < 1 ∧ 
  (∀ n m : ℕ, n ≠ m → Independent (X n) (X m)) ∧
  (∃ pdf : ℝ → ℝ, ∀ n, pdf = pdf ∧ pdf ≠ const 0 ∧ 
    let S := ∑ n, (α^n) * (X n)
    (λ x, Pr({y : ℝ | y = S }) < ⊤)) ∧
  (Continuous (λ x, Pr( {y : ℝ | y ≤ x} )))

-- The theorem we need to prove:
theorem continuous_distribution_function_of_Y (X : ℕ → ℝ) (α : ℝ) :
  problem X α → Continuous (λ x, Pr( { y : ℝ | y = ∑ i, (α^i) * X i})) :=
sorry

end continuous_distribution_function_of_Y_l36_36537


namespace daria_weeks_needed_l36_36570

-- Defining the parameters and conditions
def initial_amount : ℕ := 20
def weekly_savings : ℕ := 10
def cost_of_vacuum_cleaner : ℕ := 120

-- Defining the total money Daria needs to add to her initial amount
def additional_amount_needed : ℕ := cost_of_vacuum_cleaner - initial_amount

-- Defining the number of weeks needed to save the additional amount, given weekly savings
def weeks_needed : ℕ := additional_amount_needed / weekly_savings

-- The theorem stating that Daria needs exactly 10 weeks to cover the expense of the vacuum cleaner
theorem daria_weeks_needed : weeks_needed = 10 := by
  sorry

end daria_weeks_needed_l36_36570


namespace binom_15_3_eq_455_l36_36258

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem statement: Prove that binom 15 3 = 455
theorem binom_15_3_eq_455 : binom 15 3 = 455 := sorry

end binom_15_3_eq_455_l36_36258


namespace fraction_sum_proof_l36_36096

theorem fraction_sum_proof :
    (19 / ((2^3 - 1) * (3^3 - 1)) + 
     37 / ((3^3 - 1) * (4^3 - 1)) + 
     61 / ((4^3 - 1) * (5^3 - 1)) + 
     91 / ((5^3 - 1) * (6^3 - 1))) = (208 / 1505) :=
by
  -- Proof goes here
  sorry

end fraction_sum_proof_l36_36096


namespace tetrahedron_surface_area_l36_36876

theorem tetrahedron_surface_area (a : ℝ) (h : a = Real.sqrt 2) :
  let R := (a * Real.sqrt 6) / 4
  let S := 4 * Real.pi * R^2
  S = 3 * Real.pi := by
  /- Proof here -/
  sorry

end tetrahedron_surface_area_l36_36876


namespace evaporation_period_length_l36_36546

theorem evaporation_period_length
  (initial_water : ℕ) (daily_evaporation : ℝ) (evaporated_percentage : ℝ) : 
  evaporated_percentage * (initial_water : ℝ) / 100 / daily_evaporation = 22 :=
by
  -- Conditions of the problem
  let initial_water := 12
  let daily_evaporation := 0.03
  let evaporated_percentage := 5.5
  -- Sorry proof placeholder
  sorry

end evaporation_period_length_l36_36546


namespace find_x_in_inches_l36_36242

noncomputable def x_value (x : ℝ) : Prop :=
  let area_larger_square := (4 * x) ^ 2
  let area_smaller_square := (3 * x) ^ 2
  let area_triangle := (1 / 2) * (3 * x) * (4 * x)
  let total_area := area_larger_square + area_smaller_square + area_triangle
  total_area = 1100 ∧ x = Real.sqrt (1100 / 31)

theorem find_x_in_inches (x : ℝ) : x_value x :=
by sorry

end find_x_in_inches_l36_36242


namespace seating_arrangements_l36_36176

def count_arrangements (n k : ℕ) : ℕ :=
  (n.factorial) / (n - k).factorial

theorem seating_arrangements : count_arrangements 6 5 * 3 = 360 :=
  sorry

end seating_arrangements_l36_36176


namespace events_complementary_l36_36285

def defective_probability (total: ℕ) (defective: ℕ) (selection: ℕ) : Prop :=
  ∃ E F G : set (finset ℕ),
  -- Define event E: all 3 products are non-defective
  E = { s ∈ finset.powerset_len selection (finset.range total) | ∀ x ∈ s, x < total - defective } ∧
  -- Define event F: all 3 products are defective
  F = { s ∈ finset.powerset_len selection (finset.range total) | ∀ x ∈ s, x ≥ total - defective } ∧
  -- Define event G: at least one of the 3 products is defective
  G = { s ∈ finset.powerset_len selection (finset.range total) | ∃ x ∈ s, x ≥ total - defective } ∧
  -- E and G are complementary events
  E ∪ G = finset.powerset_len selection (finset.range total) ∧
  E ∩ G = ∅

theorem events_complementary (total defective selection : ℕ) (h1 : total = 100) (h2 : defective = 5) (h3 : selection = 3) :
  defective_probability total defective selection :=
by
  sorry

end events_complementary_l36_36285


namespace correct_choice_for_games_l36_36986
  
-- Define the problem context
def games_preferred (question : String) (answer : String) :=
  question = "Which of the two computer games did you prefer?" ∧
  answer = "Actually I didn’t like either of them."

-- Define the proof that the correct choice is 'either of them'
theorem correct_choice_for_games (question : String) (answer : String) :
  games_preferred question answer → answer = "either of them" :=
by
  -- Provided statement and proof assumptions
  intro h
  cases h
  exact sorry -- Proof steps will be here
  -- Here, the conclusion should be derived from given conditions

end correct_choice_for_games_l36_36986


namespace last_number_with_35_zeros_l36_36068

def count_zeros (n : Nat) : Nat :=
  if n = 0 then 1
  else if n < 10 then 0
  else count_zeros (n / 10) + count_zeros (n % 10)

def total_zeros_written (upto : Nat) : Nat :=
  (List.range (upto + 1)).foldl (λ acc n => acc + count_zeros n) 0

theorem last_number_with_35_zeros : ∃ n, total_zeros_written n = 35 ∧ ∀ m, m > n → total_zeros_written m ≠ 35 :=
by
  let x := 204
  have h1 : total_zeros_written x = 35 := sorry
  have h2 : ∀ m, m > x → total_zeros_written m ≠ 35 := sorry
  existsi x
  exact ⟨h1, h2⟩

end last_number_with_35_zeros_l36_36068


namespace returns_to_start_point_after_fourth_passenger_distance_after_last_passenger_total_earnings_l36_36199

noncomputable def driving_distances : List ℤ := [-5, 3, 6, -4, 7, -2]

def fare (distance : ℕ) : ℕ :=
  if distance ≤ 3 then 8 else 8 + 2 * (distance - 3)

theorem returns_to_start_point_after_fourth_passenger :
  List.sum (driving_distances.take 4) = 0 :=
by
  sorry

theorem distance_after_last_passenger :
  List.sum driving_distances = 5 :=
by
  sorry

theorem total_earnings :
  (fare 5 + fare 3 + fare 6 + fare 4 + fare 7 + fare 2) = 68 :=
by
  sorry

end returns_to_start_point_after_fourth_passenger_distance_after_last_passenger_total_earnings_l36_36199


namespace abc_area_l36_36969

def rectangle_area (length width : ℕ) : ℕ :=
  length * width

theorem abc_area :
  let smaller_side := 7
  let longer_side := 2 * smaller_side
  let length := 3 * longer_side -- since there are 3 identical rectangles placed side by side
  let width := smaller_side
  rectangle_area length width = 294 :=
by
  sorry

end abc_area_l36_36969


namespace find_constants_l36_36001

variables {A B C x : ℝ}

theorem find_constants (h : (A = 6) ∧ (B = -5) ∧ (C = 5)) :
  (x^2 + 5*x - 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1) :=
by sorry

end find_constants_l36_36001


namespace find_b_l36_36017

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 4) : b = 4 :=
sorry

end find_b_l36_36017


namespace heather_starts_24_minutes_after_stacy_l36_36076

theorem heather_starts_24_minutes_after_stacy :
  ∀ (distance_between : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) (heather_distance : ℝ),
    distance_between = 10 →
    heather_speed = 5 →
    stacy_speed = heather_speed + 1 →
    heather_distance = 3.4545454545454546 →
    60 * ((heather_distance / heather_speed) - ((distance_between - heather_distance) / stacy_speed)) = -24 :=
by
  sorry

end heather_starts_24_minutes_after_stacy_l36_36076


namespace sinA_mul_sinC_eq_three_fourths_l36_36300
open Real

-- Definitions based on conditions
def angles_form_arithmetic_sequence (A B C : ℝ) : Prop :=
  2 * B = A + C

def sides_form_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- The theorem to prove
theorem sinA_mul_sinC_eq_three_fourths
  (A B C a b c : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_angles : A + B + C = π)
  (h_angles_arithmetic : angles_form_arithmetic_sequence A B C)
  (h_sides_geometric : sides_form_geometric_sequence a b c) :
  sin A * sin C = 3 / 4 :=
sorry

end sinA_mul_sinC_eq_three_fourths_l36_36300


namespace same_cost_duration_l36_36859

-- Define the cost function for Plan A
def cost_plan_a (x : ℕ) : ℚ :=
 if x ≤ 8 then 0.60 else 0.60 + 0.06 * (x - 8)

-- Define the cost function for Plan B
def cost_plan_b (x : ℕ) : ℚ :=
 0.08 * x

-- The duration of a call for which the company charges the same under Plan A and Plan B is 14 minutes
theorem same_cost_duration (x : ℕ) : cost_plan_a x = cost_plan_b x ↔ x = 14 :=
by
  -- The proof is not required, using sorry to skip the proof steps
  sorry

end same_cost_duration_l36_36859


namespace intersection_A_B_l36_36313

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l36_36313


namespace edward_money_l36_36893

theorem edward_money (initial_amount spent1 spent2 : ℕ) (h_initial : initial_amount = 34) (h_spent1 : spent1 = 9) (h_spent2 : spent2 = 8) :
  initial_amount - (spent1 + spent2) = 17 :=
by
  sorry

end edward_money_l36_36893


namespace tan_30_l36_36739

theorem tan_30 : Real.tan (Real.pi / 6) = Real.sqrt 3 / 3 := 
by 
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  have h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2 := by sorry
  calc
    Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6) : Real.tan_eq_sin_div_cos _
    ... = (1 / 2) / (Real.sqrt 3 / 2) : by rw [h1, h2]
    ... = (1 / 2) * (2 / Real.sqrt 3) : by rw Div.div_eq_mul_inv
    ... = 1 / Real.sqrt 3 : by norm_num
    ... = Real.sqrt 3 / 3 : by rw [Div.inv_eq_inv, Mul.comm, Mul.assoc, Div.mul_inv_cancel (Real.sqrt_ne_zero _), one_div Real.sqrt 3, inv_mul_eq_div]

-- Additional necessary function apologies for the unproven theorems.
noncomputable def _root_.Real.sqrt (x:ℝ) : ℝ := sorry

noncomputable def _root_.Real.tan (x : ℝ) : ℝ :=
  (Real.sin x) / (Real.cos x)

#eval tan_30 -- check result

end tan_30_l36_36739


namespace power_vs_square_l36_36822

theorem power_vs_square (n : ℕ) (h : n ≥ 4) : 2^n ≥ n^2 := by
  sorry

end power_vs_square_l36_36822


namespace intersect_at_one_point_l36_36715

-- Definitions of points and circles
variable (Point : Type)
variable (Circle : Type)
variable (A : Point)
variable (C1 C2 C3 C4 : Circle)

-- Definition of intersection points
variable (B12 B13 B14 B23 B24 B34 : Point)

-- Note: Assumptions around the geometry structure axioms need to be defined
-- Assuming we have a function that checks if three points are collinear:
variable (are_collinear : Point → Point → Point → Prop)
-- Assuming we have a function that checks if a point is part of a circle:
variable (on_circle : Point → Circle → Prop)

-- Axioms related to the conditions
axiom collinear_B12_B34_B (hC1 : on_circle B12 C1) (hC2 : on_circle B12 C2) (hC3 : on_circle B34 C3) (hC4 : on_circle B34 C4) : 
  ∃ P : Point, are_collinear B12 P B34 

axiom collinear_B13_B24_B (hC1 : on_circle B13 C1) (hC2 : on_circle B13 C3) (hC3 : on_circle B24 C2) (hC4 : on_circle B24 C4) : 
  ∃ P : Point, are_collinear B13 P B24 

axiom collinear_B14_B23_B (hC1 : on_circle B14 C1) (hC2 : on_circle B14 C4) (hC3 : on_circle B23 C2) (hC4 : on_circle B23 C3) : 
  ∃ P : Point, are_collinear B14 P B23 

-- The theorem to be proved
theorem intersect_at_one_point :
  ∃ P : Point, 
    are_collinear B12 P B34 ∧ are_collinear B13 P B24 ∧ are_collinear B14 P B23 := 
sorry

end intersect_at_one_point_l36_36715


namespace expected_volunteers_by_2022_l36_36630

noncomputable def initial_volunteers : ℕ := 1200
noncomputable def increase_2021 : ℚ := 0.15
noncomputable def increase_2022 : ℚ := 0.30

theorem expected_volunteers_by_2022 :
  (initial_volunteers * (1 + increase_2021) * (1 + increase_2022)) = 1794 := 
by
  sorry

end expected_volunteers_by_2022_l36_36630


namespace smallest_number_of_set_s_l36_36534

theorem smallest_number_of_set_s : 
  ∀ (s : Set ℕ),
    (∃ n : ℕ, s = {k | ∃ m : ℕ, k = 5 * (m+n) ∧ m < 45}) ∧ 
    (275 ∈ s) → 
      (∃ min_elem : ℕ, min_elem ∈ s ∧ min_elem = 55) 
  :=
by
  sorry

end smallest_number_of_set_s_l36_36534


namespace area_of_rhombus_l36_36373

theorem area_of_rhombus (P D : ℕ) (area : ℝ) (hP : P = 48) (hD : D = 26) :
  area = 25 := by
  sorry

end area_of_rhombus_l36_36373


namespace total_tissues_l36_36827

-- define the number of students in each group
def g1 : Nat := 9
def g2 : Nat := 10
def g3 : Nat := 11

-- define the number of tissues per mini tissue box
def t : Nat := 40

-- state the main theorem
theorem total_tissues : (g1 + g2 + g3) * t = 1200 := by
  sorry

end total_tissues_l36_36827


namespace circumcircle_eqn_l36_36434

variables (D E F : ℝ)

def point_A := (4, 0)
def point_B := (0, 3)
def point_C := (0, 0)

-- Define the system of equations for the circumcircle
def system : Prop :=
  (16 + 4*D + F = 0) ∧
  (9 + 3*E + F = 0) ∧
  (F = 0)

theorem circumcircle_eqn : system D E F → (D = -4 ∧ E = -3 ∧ F = 0) :=
sorry -- Proof omitted

end circumcircle_eqn_l36_36434


namespace exam_rule_l36_36814

variable (P R Q : Prop)

theorem exam_rule (hp : P ∧ R → Q) : ¬ Q → ¬ P ∨ ¬ R :=
by
  sorry

end exam_rule_l36_36814


namespace longest_interval_between_friday_13ths_l36_36028

theorem longest_interval_between_friday_13ths
  (friday_the_13th : ℕ → ℕ → Prop)
  (at_least_once_per_year : ∀ year, ∃ month, friday_the_13th year month)
  (friday_occurs : ℕ) :
  ∃ (interval : ℕ), interval = 14 :=
by
  sorry

end longest_interval_between_friday_13ths_l36_36028


namespace rest_duration_per_kilometer_l36_36872

theorem rest_duration_per_kilometer
  (speed : ℕ)
  (total_distance : ℕ)
  (total_time : ℕ)
  (walking_time : ℕ := total_distance / speed * 60)  -- walking_time in minutes
  (rest_time : ℕ := total_time - walking_time)  -- total resting time in minutes
  (number_of_rests : ℕ := total_distance - 1)  -- number of rests after each kilometer
  (duration_per_rest : ℕ := rest_time / number_of_rests)
  (h1 : speed = 10)
  (h2 : total_distance = 5)
  (h3 : total_time = 50) : 
  (duration_per_rest = 5) := 
sorry

end rest_duration_per_kilometer_l36_36872


namespace smallest_n_for_perfect_square_and_cube_l36_36699

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, (∃ a : ℕ, 4 * n = a^2) ∧ (∃ b : ℕ, 5 * n = b^3) ∧ n = 125 :=
by
  sorry

end smallest_n_for_perfect_square_and_cube_l36_36699


namespace four_corresponds_to_364_l36_36465

noncomputable def number_pattern (n : ℕ) : ℕ :=
  match n with
  | 1 => 6
  | 2 => 36
  | 3 => 363
  | 5 => 365
  | 36 => 2
  | _ => 0 -- Assume 0 as the default case

theorem four_corresponds_to_364 : number_pattern 4 = 364 :=
sorry

end four_corresponds_to_364_l36_36465


namespace bus_speed_l36_36168

theorem bus_speed (distance time : ℝ) (h_distance : distance = 201) (h_time : time = 3) : 
  distance / time = 67 :=
by
  sorry

end bus_speed_l36_36168


namespace sum_of_interior_angles_octagon_l36_36212

theorem sum_of_interior_angles_octagon : (8 - 2) * 180 = 1080 :=
by
  sorry

end sum_of_interior_angles_octagon_l36_36212


namespace judy_shopping_total_l36_36131

noncomputable def carrot_price := 1
noncomputable def milk_price := 3
noncomputable def pineapple_price := 4 / 2 -- half price
noncomputable def flour_price := 5
noncomputable def ice_cream_price := 7

noncomputable def carrot_quantity := 5
noncomputable def milk_quantity := 3
noncomputable def pineapple_quantity := 2
noncomputable def flour_quantity := 2
noncomputable def ice_cream_quantity := 1

noncomputable def initial_cost : ℝ := 
  carrot_quantity * carrot_price 
  + milk_quantity * milk_price 
  + pineapple_quantity * pineapple_price 
  + flour_quantity * flour_price 
  + ice_cream_quantity * ice_cream_price

noncomputable def final_cost (initial_cost: ℝ) := if initial_cost ≥ 25 then initial_cost - 5 else initial_cost

theorem judy_shopping_total : final_cost initial_cost = 30 := by
  sorry

end judy_shopping_total_l36_36131


namespace total_revenue_is_correct_l36_36302

theorem total_revenue_is_correct :
  let fiction_books := 60
  let nonfiction_books := 84
  let children_books := 42

  let fiction_sold_frac := 3/4
  let nonfiction_sold_frac := 5/6
  let children_sold_frac := 2/3

  let fiction_price := 5
  let nonfiction_price := 7
  let children_price := 3

  let fiction_sold_qty := fiction_sold_frac * fiction_books
  let nonfiction_sold_qty := nonfiction_sold_frac * nonfiction_books
  let children_sold_qty := children_sold_frac * children_books

  let total_revenue := fiction_sold_qty * fiction_price
                       + nonfiction_sold_qty * nonfiction_price
                       + children_sold_qty * children_price
  in total_revenue = 799 :=
by
  sorry

end total_revenue_is_correct_l36_36302


namespace increasing_function_solution_l36_36269

noncomputable def solution (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x + y) * (f x + f y) = f x * f y

theorem increasing_function_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x + y) * (f x + f y) = f x * f y)
  ∧ (∀ x y : ℝ, x < y → f x < f y)
  → ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = 1 / (a * x) :=
by {
  sorry
}

end increasing_function_solution_l36_36269


namespace minimum_value_expression_l36_36919

-- Define the conditions in the problem
variable (m n : ℝ) (h1 : m > 0) (h2 : n > 0)
variable (h3 : 2 * m + 2 * n = 2)

-- State the theorem proving the minimum value of the given expression
theorem minimum_value_expression : (1 / m + 2 / n) = 3 + 2 * Real.sqrt 2 := by
  sorry

end minimum_value_expression_l36_36919


namespace gain_in_transaction_per_year_l36_36408

noncomputable def compounded_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def gain_per_year (P : ℝ) (t : ℝ) (r1 : ℝ) (n1 : ℕ) (r2 : ℝ) (n2 : ℕ) : ℝ :=
  let amount_repaid := compounded_interest P r1 n1 t
  let amount_received := compounded_interest P r2 n2 t
  (amount_received - amount_repaid) / t

theorem gain_in_transaction_per_year :
  let P := 8000
  let t := 3
  let r1 := 0.05
  let n1 := 2
  let r2 := 0.07
  let n2 := 4
  abs (gain_per_year P t r1 n1 r2 n2 - 191.96) < 0.01 :=
by
  sorry

end gain_in_transaction_per_year_l36_36408


namespace smallest_n_condition_l36_36705

theorem smallest_n_condition (n : ℕ) : (4 * n) ∣ (n^2) ∧ (5 * n) ∣ (u^3) → n = 100 :=
by
  sorry

end smallest_n_condition_l36_36705


namespace daria_weeks_needed_l36_36568

-- Defining the parameters and conditions
def initial_amount : ℕ := 20
def weekly_savings : ℕ := 10
def cost_of_vacuum_cleaner : ℕ := 120

-- Defining the total money Daria needs to add to her initial amount
def additional_amount_needed : ℕ := cost_of_vacuum_cleaner - initial_amount

-- Defining the number of weeks needed to save the additional amount, given weekly savings
def weeks_needed : ℕ := additional_amount_needed / weekly_savings

-- The theorem stating that Daria needs exactly 10 weeks to cover the expense of the vacuum cleaner
theorem daria_weeks_needed : weeks_needed = 10 := by
  sorry

end daria_weeks_needed_l36_36568


namespace smallest_positive_integer_solution_l36_36976

theorem smallest_positive_integer_solution (x : ℕ) (h : 5 * x ≡ 17 [MOD 29]) : x = 15 :=
sorry

end smallest_positive_integer_solution_l36_36976


namespace combinations_eight_choose_three_l36_36468

theorem combinations_eight_choose_three : Nat.choose 8 3 = 56 := by
  sorry

end combinations_eight_choose_three_l36_36468


namespace winning_post_distance_l36_36233

theorem winning_post_distance (v x : ℝ) (h₁ : x ≠ 0) (h₂ : v ≠ 0)
  (h₃ : 1.75 * v = v) 
  (h₄ : x = 1.75 * (x - 84)) : 
  x = 196 :=
by 
  sorry

end winning_post_distance_l36_36233


namespace find_X_l36_36478

def star (a b : ℤ) : ℤ := 5 * a - 3 * b

theorem find_X (X : ℤ) (h1 : star X (star 3 2) = 18) : X = 9 :=
by
  sorry

end find_X_l36_36478


namespace rationalize_denominator_to_find_constants_l36_36946

-- Definitions of the given conditions
def original_fraction := 3 / (4 * Real.sqrt 7 + 3 * Real.sqrt 13)
def simplified_fraction (A B C D E : ℤ) := (A * Real.sqrt B + C * Real.sqrt D) / E

-- Statement of the proof problem
theorem rationalize_denominator_to_find_constants :
  ∃ (A B C D E : ℤ),
    original_fraction = simplified_fraction A B C D E ∧
    B < D ∧
    (∀ p : ℕ, Real.sqrt (p * p) = p) ∧ -- Ensuring that all radicals are in simplest form
    A + B + C + D + E = 22 :=
sorry

end rationalize_denominator_to_find_constants_l36_36946


namespace reflection_curve_eq_l36_36648

theorem reflection_curve_eq (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, let y := ax^2 + bx + c in
   let C1 := ax^2 - bx + c in
   let C2 := -C1 in
   y = -ax^2 + bx - c) :=
sorry

end reflection_curve_eq_l36_36648


namespace set_intersection_l36_36324

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2.5}

theorem set_intersection : A ∩ B = {0, 1, 2} :=
by
  sorry

end set_intersection_l36_36324


namespace find_g_inv_f_neg7_l36_36415

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_def : ∀ x, f_inv (g x) = 5 * x + 3

theorem find_g_inv_f_neg7 : g_inv (f (-7)) = -2 :=
by
  sorry

end find_g_inv_f_neg7_l36_36415


namespace max_lambda_inequality_l36_36152

theorem max_lambda_inequality 
  (a b x y : ℝ) 
  (h1 : a ≥ 0) 
  (h2 : b ≥ 0)
  (h3 : x ≥ 0)
  (h4 : y ≥ 0)
  (h5 : a + b = 27) : 
  (a * x^2 + b * y^2 + 4 * x * y)^3 ≥ 4 * (a * x^2 * y + b * x * y^2)^2 :=
sorry

end max_lambda_inequality_l36_36152


namespace population_doubles_in_35_years_l36_36234

noncomputable def birth_rate : ℝ := 39.4 / 1000
noncomputable def death_rate : ℝ := 19.4 / 1000
noncomputable def natural_increase_rate : ℝ := birth_rate - death_rate
noncomputable def doubling_time (r: ℝ) : ℝ := 70 / (r * 100)

theorem population_doubles_in_35_years :
  doubling_time natural_increase_rate = 35 := by sorry

end population_doubles_in_35_years_l36_36234


namespace ellipse_equation_range_OQ_l36_36775

-- Conditions for the problem
variables {a b : ℝ} (h₀ : a > b) (h₁ : b > 1) 
variable (P : ℝ × ℝ)  -- Assuming P = (P.1, P.2) lies on the unit circle
variables {A F O : ℝ × ℝ}  -- Points A, F, O

-- Additional geometric conditions
variable h₂ : a = 2
variable h₃ : b = Real.sqrt 3

-- Given specific angle and distances
variable h4 : (angle P O A = Real.pi / 3)
variable h5 : dist P F = 1
variable h6 : dist P A = Real.sqrt 3 

-- We need to prove that the equation of C is as follows:
theorem ellipse_equation (h₀ : a > b) (h₁ : b > 1) (h₂ : a = 2) (h₃ : b = Real.sqrt 3) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) := 
by
  -- Equation of the ellipse using the given lengths
  sorry

-- We need to prove the range for |OQ| 
theorem range_OQ (P : ℝ × ℝ) (A F O : ℝ × ℝ) (h4: (angle P O A = Real.pi / 3)) (h5: dist P F = 1) (h6: dist P A = Real.sqrt 3) 
  (H : P.1^2 + P.2^2 = 1): (range_OQ (A F P O) = [3, 4]) :=
by
  -- Derivation based on given tangent conditions and distance formula
  sorry

end ellipse_equation_range_OQ_l36_36775


namespace sum_cube_eq_l36_36944

theorem sum_cube_eq (a b c : ℝ) (h : a + b + c = 0) : a^3 + b^3 + c^3 = 3 * a * b * c :=
by 
  sorry

end sum_cube_eq_l36_36944


namespace S_21_equals_4641_l36_36615

-- Define the first element of the nth set
def first_element_of_set (n : ℕ) : ℕ :=
  1 + (n * (n - 1)) / 2

-- Define the last element of the nth set
def last_element_of_set (n : ℕ) : ℕ :=
  (first_element_of_set n) + n - 1

-- Define the sum of the nth set
def S (n : ℕ) : ℕ :=
  n * ((first_element_of_set n) + (last_element_of_set n)) / 2

-- The goal statement we want to prove
theorem S_21_equals_4641 : S 21 = 4641 := by
  sorry

end S_21_equals_4641_l36_36615


namespace gcd_largest_divisor_l36_36676

theorem gcd_largest_divisor (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 1008) : 
  ∃ d, nat.gcd a b = d ∧ d = 504 :=
begin
  sorry
end

end gcd_largest_divisor_l36_36676


namespace calculate_train_length_l36_36874

noncomputable def train_length (speed_kmph : ℕ) (time_secs : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let speed_mps := (speed_kmph * 1000) / 3600
  let total_distance := speed_mps * time_secs
  total_distance - bridge_length_m

theorem calculate_train_length :
  train_length 60 14.998800095992321 140 = 110 :=
by
  sorry

end calculate_train_length_l36_36874


namespace solve_system_l36_36500

variable (x y z w : ℝ)
variable (s1 s2 s3 s4 : ℝ)
variable (p q m n b1 b2 b3 b4 : ℝ)

-- Defining the conditions from the problem
def conditions := 
  x - y + z - w = 2 ∧
  x^2 - y^2 + z^2 - w^2 = 6 ∧
  x^3 - y^3 + z^3 - w^3 = 20 ∧
  x^4 - y^4 + z^4 - w^4 = 66

-- Problem statement
theorem solve_system : 
  conditions x y z w →
  ∃ (x y z w : ℝ), conditions x y z w :=
by
  sorry

end solve_system_l36_36500


namespace find_p_if_parabola_axis_tangent_to_circle_l36_36610

theorem find_p_if_parabola_axis_tangent_to_circle :
  ∀ (p : ℝ), 0 < p →
    (∃ (C : ℝ × ℝ) (r : ℝ), 
      (C = (2, 0)) ∧ (r = 3) ∧ (dist (C.1 + p / 2, C.2) (C.1, C.2) = r) 
    ) → p = 2 :=
by
  intro p hp h
  rcases h with ⟨C, r, hC, hr, h_dist⟩ 
  have h_eq : C = (2, 0) := hC
  have hr_eq : r = 3 := hr
  rw [h_eq, hr_eq] at h_dist
  sorry

end find_p_if_parabola_axis_tangent_to_circle_l36_36610


namespace find_a_in_triangle_l36_36928

variable (a b c B : ℝ)

theorem find_a_in_triangle (h1 : b = Real.sqrt 3) (h2 : c = 3) (h3 : B = 30) :
    a = 2 * Real.sqrt 3 := by
  sorry

end find_a_in_triangle_l36_36928


namespace range_of_m_l36_36284

theorem range_of_m (m : ℝ) (h : (8 - m) / (m - 5) > 1) : 5 < m ∧ m < 13 / 2 :=
by
  sorry

end range_of_m_l36_36284


namespace tan_30_deg_l36_36735

theorem tan_30_deg : 
  let θ := (30 : ℝ) * (Real.pi / 180)
  in Real.sin θ = 1 / 2 ∧ Real.cos θ = Real.sqrt 3 / 2 → Real.tan θ = Real.sqrt 3 / 3 :=
by
  intro h
  let th := θ
  have h1 : Real.sin th = 1 / 2 := And.left h
  have h2 : Real.cos th = Real.sqrt 3 / 2 := And.right h
  sorry

end tan_30_deg_l36_36735


namespace normal_distribution_test_l36_36847

noncomputable def normal_distribution_at_least_90 : Prop :=
  let μ := 78
  let σ := 4
  -- Given reference data
  let p_within_3_sigma := 0.9974
  -- Calculate P(X >= 90)
  let p_at_least_90 := (1 - p_within_3_sigma) / 2
  -- The expected answer 0.13% ⇒ 0.0013
  p_at_least_90 = 0.0013

theorem normal_distribution_test :
  normal_distribution_at_least_90 :=
by
  sorry

end normal_distribution_test_l36_36847


namespace max_three_topping_pizzas_l36_36108

-- Define the combinations function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Assert the condition and the question with the expected answer
theorem max_three_topping_pizzas : combination 8 3 = 56 :=
by
  sorry

end max_three_topping_pizzas_l36_36108


namespace rationalize_denominator_l36_36950

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  A + B + C + D + E = 22 :=
by
  -- Proof goes here
  sorry

end rationalize_denominator_l36_36950


namespace tickets_difference_l36_36417

theorem tickets_difference :
  let tickets_won := 48.5
  let yoyo_cost := 11.7
  let keychain_cost := 6.3
  let plush_toy_cost := 16.2
  let total_cost := yoyo_cost + keychain_cost + plush_toy_cost
  let tickets_left := tickets_won - total_cost
  tickets_won - tickets_left = total_cost :=
by
  sorry

end tickets_difference_l36_36417


namespace smallest_n_condition_l36_36704

theorem smallest_n_condition (n : ℕ) : (4 * n) ∣ (n^2) ∧ (5 * n) ∣ (u^3) → n = 100 :=
by
  sorry

end smallest_n_condition_l36_36704


namespace arccos_neg_one_eq_pi_l36_36889

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := 
by
  sorry

end arccos_neg_one_eq_pi_l36_36889


namespace hamburger_combinations_l36_36023

def number_of_condiments := 8
def condiment_combinations := 2 ^ number_of_condiments
def number_of_meat_patties := 4
def total_hamburgers := number_of_meat_patties * condiment_combinations

theorem hamburger_combinations :
  total_hamburgers = 1024 :=
by
  sorry

end hamburger_combinations_l36_36023


namespace election_total_votes_l36_36039

theorem election_total_votes (V: ℝ) (valid_votes: ℝ) (candidate_votes: ℝ) (invalid_rate: ℝ) (candidate_rate: ℝ) :
  candidate_rate = 0.75 →
  invalid_rate = 0.15 →
  candidate_votes = 357000 →
  valid_votes = (1 - invalid_rate) * V →
  candidate_votes = candidate_rate * valid_votes →
  V = 560000 :=
by
  intros candidate_rate_eq invalid_rate_eq candidate_votes_eq valid_votes_eq equation
  sorry

end election_total_votes_l36_36039


namespace find_m_l36_36021

variable (m : ℝ)

-- Definitions of the vectors
def AB : ℝ × ℝ := (m + 3, 2 * m + 1)
def CD : ℝ × ℝ := (m + 3, -5)

-- Definition of perpendicular vectors, dot product is zero
def perp (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_m (h : perp (AB m) (CD m)) : m = 2 := by
  sorry

end find_m_l36_36021


namespace fraction_simplification_l36_36093

theorem fraction_simplification : (145^2 - 121^2) / 24 = 266 := by
  sorry

end fraction_simplification_l36_36093


namespace Alex_has_more_than_200_marbles_on_Monday_of_next_week_l36_36557

theorem Alex_has_more_than_200_marbles_on_Monday_of_next_week :
  ∃ k : ℕ, k > 0 ∧ 3 * 2^k > 200 ∧ k % 7 = 1 := by
  sorry

end Alex_has_more_than_200_marbles_on_Monday_of_next_week_l36_36557


namespace intersection_nonempty_implies_t_lt_1_l36_36613

def M (x : ℝ) := x ≤ 1
def P (t : ℝ) (x : ℝ) := x > t

theorem intersection_nonempty_implies_t_lt_1 {t : ℝ} (h : ∃ x, M x ∧ P t x) : t < 1 :=
by
  sorry

end intersection_nonempty_implies_t_lt_1_l36_36613


namespace correct_quotient_is_32_l36_36463

-- Definitions based on the conditions
def incorrect_divisor := 12
def correct_divisor := 21
def incorrect_quotient := 56
def dividend := incorrect_divisor * incorrect_quotient -- Given as 672

-- Statement of the theorem
theorem correct_quotient_is_32 :
  dividend / correct_divisor = 32 :=
by
  -- skip the proof
  sorry

end correct_quotient_is_32_l36_36463


namespace factor_theorem_l36_36136

theorem factor_theorem (t : ℝ) : (5 * t^2 + 15 * t - 20 = 0) ↔ (t = 1 ∨ t = -4) :=
by
  sorry

end factor_theorem_l36_36136


namespace sum_of_largest_and_smallest_is_correct_l36_36231

-- Define the set of digits
def digits : Finset ℕ := {2, 0, 4, 1, 5, 8}

-- Define the largest possible number using the digits
def largestNumber : ℕ := 854210

-- Define the smallest possible number using the digits
def smallestNumber : ℕ := 102458

-- Define the sum of largest and smallest possible numbers
def sumOfNumbers : ℕ := largestNumber + smallestNumber

-- Main theorem to prove
theorem sum_of_largest_and_smallest_is_correct : sumOfNumbers = 956668 := by
  sorry

end sum_of_largest_and_smallest_is_correct_l36_36231


namespace probability_no_shaded_square_l36_36866

theorem probability_no_shaded_square : 
  let n : ℕ := 502 * 1004
  let m : ℕ := 502^2
  let total_rectangles := 3 * n
  let rectangles_with_shaded := 3 * m
  let probability_includes_shaded := rectangles_with_shaded / total_rectangles
  1 - probability_includes_shaded = (1 : ℚ) / 2 := 
by 
  sorry

end probability_no_shaded_square_l36_36866


namespace find_function_l36_36137

/-- Any function f : ℝ → ℝ satisfying the two given conditions must be of the form f(x) = cx where |c| ≤ 1. -/
theorem find_function (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, x ≠ 0 → x * (f (x + 1) - f x) = f x)
  (h2 : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ c : ℝ, (∀ x : ℝ, f x = c * x) ∧ |c| ≤ 1 :=
by
  sorry

end find_function_l36_36137


namespace math_problem_l36_36558

theorem math_problem : 2 + 5 * 4 - 6 + 3 = 19 := by
  sorry

end math_problem_l36_36558


namespace fraction_raised_to_zero_l36_36222

theorem fraction_raised_to_zero:
  (↑(-4305835) / ↑1092370457 : ℚ)^0 = 1 := 
by
  sorry

end fraction_raised_to_zero_l36_36222


namespace geometric_sequence_ratio_l36_36150

variables {a b c q : ℝ}

theorem geometric_sequence_ratio (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sequence : ∃ q : ℝ, (a + b + c) * q = b + c - a ∧
                         (a + b + c) * q^2 = c + a - b ∧
                         (a + b + c) * q^3 = a + b - c) :
  q^3 + q^2 + q = 1 := 
sorry

end geometric_sequence_ratio_l36_36150


namespace sail_time_difference_l36_36428

theorem sail_time_difference (distance : ℕ) (v_big : ℕ) (v_small : ℕ) (t_big t_small : ℕ)
  (h_distance : distance = 200)
  (h_v_big : v_big = 50)
  (h_v_small : v_small = 20)
  (h_t_big : t_big = distance / v_big)
  (h_t_small : t_small = distance / v_small)
  : t_small - t_big = 6 := by
  sorry

end sail_time_difference_l36_36428


namespace smallest_h_l36_36535

theorem smallest_h (h : ℕ) : 
  (∀ k, h = k → (k + 5) % 8 = 0 ∧ 
        (k + 5) % 11 = 0 ∧ 
        (k + 5) % 24 = 0) ↔ h = 259 :=
by
  sorry

end smallest_h_l36_36535


namespace evaluate_64_pow_5_div_6_l36_36757

theorem evaluate_64_pow_5_div_6 : (64 : ℝ)^(5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ)^6 := by norm_num
  rw [← h1]
  have h2 : ((2 : ℝ)^6)^(5 / 6) = (2 : ℝ)^(6 * (5 / 6)) := by rw [Real.rpow_mul]
  rw [h2]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l36_36757


namespace find_t_l36_36600

variables {m n : ℝ}
variables (t : ℝ)
variables (mv nv : ℝ)
variables (dot_m_m dot_m_n dot_n_n : ℝ)
variables (cos_theta : ℝ)

-- Define the basic assumptions
axiom non_zero_vectors : m ≠ 0 ∧ n ≠ 0
axiom magnitude_condition : mv = 2 * nv
axiom cos_condition : cos_theta = 1 / 3
axiom perpendicular_condition : dot_m_n = (mv * nv * cos_theta) ∧ (t * dot_m_n + dot_m_m = 0)

-- Utilize the conditions and prove the target
theorem find_t : t = -6 :=
sorry

end find_t_l36_36600


namespace tangent_parallel_to_line_l36_36896

theorem tangent_parallel_to_line (x y : ℝ) :
  (y = x^3 + x - 1) ∧ (3 * x^2 + 1 = 4) → (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -3) := by
  sorry

end tangent_parallel_to_line_l36_36896


namespace students_not_enrolled_l36_36460

theorem students_not_enrolled (total_students : ℕ) (students_french : ℕ) (students_german : ℕ) (students_both : ℕ)
  (h1 : total_students = 94)
  (h2 : students_french = 41)
  (h3 : students_german = 22)
  (h4 : students_both = 9) : 
  ∃ (students_neither : ℕ), students_neither = 40 :=
by
  -- We would show the calculation here in a real proof 
  sorry

end students_not_enrolled_l36_36460


namespace line_equation_in_slope_intercept_form_l36_36997

variable {x y : ℝ}

theorem line_equation_in_slope_intercept_form :
  (3 * (x - 2) - 4 * (y - 8) = 0) → (y = (3 / 4) * x + 6.5) :=
by
  intro h
  sorry

end line_equation_in_slope_intercept_form_l36_36997


namespace sufficient_but_not_necessary_condition_l36_36601

theorem sufficient_but_not_necessary_condition (a : ℝ) (h₁ : a > 2) : a ≥ 1 ∧ ¬(∀ (a : ℝ), a ≥ 1 → a > 2) := 
by
  sorry

end sufficient_but_not_necessary_condition_l36_36601


namespace Juanita_spends_more_l36_36778

def Grant_annual_spend : ℝ := 200
def weekday_spend : ℝ := 0.50
def sunday_spend : ℝ := 2.00
def weeks_in_year : ℝ := 52

def Juanita_weekly_spend : ℝ := (6 * weekday_spend) + sunday_spend
def Juanita_annual_spend : ℝ := weeks_in_year * Juanita_weekly_spend
def spending_difference : ℝ := Juanita_annual_spend - Grant_annual_spend

theorem Juanita_spends_more : spending_difference = 60 := by
  sorry

end Juanita_spends_more_l36_36778


namespace find_x_l36_36451

theorem find_x (x : ℝ) (h : 0.60 / x = 6 / 2) : x = 0.2 :=
by {
  sorry
}

end find_x_l36_36451


namespace max_songs_played_l36_36902

theorem max_songs_played (n m t : ℕ) (h1 : n = 50) (h2 : m = 50) (h3 : t = 180) :
  3 * n + 5 * (m - ((t - 3 * n) / 5)) = 56 :=
by
  sorry

end max_songs_played_l36_36902


namespace combinations_eight_choose_three_l36_36469

theorem combinations_eight_choose_three : Nat.choose 8 3 = 56 := by
  sorry

end combinations_eight_choose_three_l36_36469


namespace farmer_plough_remaining_area_l36_36406

theorem farmer_plough_remaining_area :
  ∀ (x R : ℕ),
  (90 * x = 3780) →
  (85 * (x + 2) + R = 3780) →
  R = 40 :=
by
  intros x R h1 h2
  sorry

end farmer_plough_remaining_area_l36_36406


namespace find_four_digit_number_l36_36310

-- Definitions of the digit variables a, b, c, d, and their constraints.
def four_digit_expressions_meet_condition (abcd abc ab : ℕ) (a : ℕ) :=
  ∃ (b c d : ℕ), abcd = (1000 * a + 100 * b + 10 * c + d)
  ∧ abc = (100 * a + 10 * b + c)
  ∧ ab = (10 * a + b)
  ∧ abcd - abc - ab - a = 1787

-- Main statement to be proven.
theorem find_four_digit_number
: ∀ a b c d : ℕ, 
  four_digit_expressions_meet_condition (1000 * a + 100 * b + 10 * c + d) (100 * a + 10 * b + c) (10 * a + b) a
  → (a = 2 ∧ b = 0 ∧ ((c = 0 ∧ d = 9) ∨ (c = 1 ∧ d = 0))) :=
sorry

end find_four_digit_number_l36_36310


namespace topsoil_cost_correct_l36_36848

noncomputable def topsoilCost (price_per_cubic_foot : ℝ) (yard_to_foot : ℝ) (discount_threshold : ℝ) (discount_rate : ℝ) (volume_in_yards : ℝ) : ℝ :=
  let volume_in_feet := volume_in_yards * yard_to_foot
  let cost_without_discount := volume_in_feet * price_per_cubic_foot
  if volume_in_feet > discount_threshold then
    cost_without_discount * (1 - discount_rate)
  else
    cost_without_discount

theorem topsoil_cost_correct:
  topsoilCost 8 27 100 0.10 7 = 1360.8 :=
by
  sorry

end topsoil_cost_correct_l36_36848


namespace distance_between_points_l36_36974

def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (-5, 7)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance point1 point2 = 2 * real.sqrt 13 :=
by
  sorry

end distance_between_points_l36_36974


namespace length_of_train_is_250_02_l36_36981

noncomputable def train_speed_km_per_hr : ℝ := 100
noncomputable def time_to_cross_pole_sec : ℝ := 9

-- Convert speed from km/hr to m/s
noncomputable def speed_m_per_s : ℝ := train_speed_km_per_hr * (1000 / 3600)

-- Calculating the length of the train
noncomputable def length_of_train : ℝ := speed_m_per_s * time_to_cross_pole_sec

theorem length_of_train_is_250_02 :
  length_of_train = 250.02 := by
  -- Proof is omitted (replace 'sorry' with the actual proof)
  sorry

end length_of_train_is_250_02_l36_36981


namespace percentage_of_motorists_speeding_l36_36985

-- Definitions based on the conditions
def total_motorists : Nat := 100
def percent_motorists_receive_tickets : Real := 0.20
def percent_speeders_no_tickets : Real := 0.20

-- Define the variables for the number of speeders
variable (x : Real) -- the percentage of total motorists who speed 

-- Lean statement to formalize the problem
theorem percentage_of_motorists_speeding 
  (h1 : 20 = (0.80 * x) * (total_motorists / 100)) : 
  x = 25 :=
sorry

end percentage_of_motorists_speeding_l36_36985


namespace transformation_result_l36_36384

noncomputable def rotate_and_dilate (z : ℂ) : ℂ :=
  let rotation := (1/2 : ℝ) + ((Real.sqrt 3 / 2) : ℝ) * Complex.I
  let dilation := 2 : ℂ
  z * (rotation * dilation)

theorem transformation_result : rotate_and_dilate (1 + 3 * Complex.I) = 1 - 3 * Real.sqrt 3 + (3 + Real.sqrt 3) * Complex.I :=
by 
  sorry

end transformation_result_l36_36384


namespace Daria_vacuum_cleaner_problem_l36_36567

theorem Daria_vacuum_cleaner_problem (initial_savings weekly_savings target_savings weeks_needed : ℕ)
  (h1 : initial_savings = 20)
  (h2 : weekly_savings = 10)
  (h3 : target_savings = 120)
  (h4 : weeks_needed = (target_savings - initial_savings) / weekly_savings) : 
  weeks_needed = 10 :=
by
  sorry

end Daria_vacuum_cleaner_problem_l36_36567


namespace calculate_expression_l36_36884

-- Define the numerator and denominator
def numerator := 11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1
def denominator := 2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10

-- Prove the expression equals 1
theorem calculate_expression : (numerator / denominator) = 1 := by
  sorry

end calculate_expression_l36_36884


namespace part1_part2_l36_36920

-- Define the system of equations
def system_eq (x y k : ℝ) : Prop := 
  3 * x + y = k + 1 ∧ x + 3 * y = 3

-- Part (1): x and y are opposite in sign implies k = -4
theorem part1 (x y k : ℝ) (h_eq : system_eq x y k) (h_sign : x * y < 0) : k = -4 := by
  sorry

-- Part (2): range of values for k given extra inequalities
theorem part2 (x y k : ℝ) (h_eq : system_eq x y k) 
  (h_ineq1 : x + y < 3) (h_ineq2 : x - y > 1) : 4 < k ∧ k < 8 := by
  sorry

end part1_part2_l36_36920


namespace num_chickens_is_one_l36_36556

-- Define the number of dogs and the number of total legs
def num_dogs := 2
def total_legs := 10

-- Define the number of legs per dog and per chicken
def legs_per_dog := 4
def legs_per_chicken := 2

-- Define the number of chickens
def num_chickens := (total_legs - num_dogs * legs_per_dog) / legs_per_chicken

-- Prove that the number of chickens is 1
theorem num_chickens_is_one : num_chickens = 1 := by
  -- This is the proof placeholder
  sorry

end num_chickens_is_one_l36_36556


namespace jason_grass_cutting_time_l36_36632

def total_minutes (hours : ℕ) : ℕ := hours * 60
def minutes_per_yard : ℕ := 30
def total_yards_per_weekend : ℕ := 8 * 2
def total_minutes_per_weekend : ℕ := minutes_per_yard * total_yards_per_weekend
def convert_minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

theorem jason_grass_cutting_time : 
  convert_minutes_to_hours total_minutes_per_weekend = 8 := by
  sorry

end jason_grass_cutting_time_l36_36632


namespace find_positive_integers_l36_36270

theorem find_positive_integers (a b : ℕ) (h1 : a > 1) (h2 : b ∣ (a - 1)) (h3 : (2 * a + 1) ∣ (5 * b - 3)) : a = 10 ∧ b = 9 :=
sorry

end find_positive_integers_l36_36270


namespace sufficient_but_not_necessary_condition_l36_36278

variable (x : ℝ)

def p := x > 2
def q := x^2 > 4

theorem sufficient_but_not_necessary_condition : (p x) → (q x) ∧ ¬((q x) → (p x)) := 
by
  sorry

end sufficient_but_not_necessary_condition_l36_36278


namespace Mike_changed_64_tires_l36_36486

def tires_changed (motorcycles: ℕ) (cars: ℕ): ℕ := 
  (motorcycles * 2) + (cars * 4)

theorem Mike_changed_64_tires:
  (tires_changed 12 10) = 64 :=
by
  sorry

end Mike_changed_64_tires_l36_36486


namespace solve_for_x_l36_36499

theorem solve_for_x (x : ℝ) : 7 * (4 * x + 3) - 5 = -3 * (2 - 5 * x) ↔ x = -22 / 13 := 
by 
  sorry

end solve_for_x_l36_36499


namespace three_dice_prime_probability_l36_36924

noncomputable def rolling_three_dice_prime_probability : ℚ :=
  sorry

theorem three_dice_prime_probability : rolling_three_dice_prime_probability = 1 / 24 :=
  sorry

end three_dice_prime_probability_l36_36924


namespace Juanita_spends_more_l36_36779

def Grant_annual_spend : ℝ := 200
def weekday_spend : ℝ := 0.50
def sunday_spend : ℝ := 2.00
def weeks_in_year : ℝ := 52

def Juanita_weekly_spend : ℝ := (6 * weekday_spend) + sunday_spend
def Juanita_annual_spend : ℝ := weeks_in_year * Juanita_weekly_spend
def spending_difference : ℝ := Juanita_annual_spend - Grant_annual_spend

theorem Juanita_spends_more : spending_difference = 60 := by
  sorry

end Juanita_spends_more_l36_36779


namespace unique_positive_x_for_volume_l36_36980

variable (x : ℕ)

def prism_volume (x : ℕ) : ℕ :=
  (x + 5) * (x - 5) * (x ^ 2 + 25)

theorem unique_positive_x_for_volume {x : ℕ} (h : prism_volume x < 700) (h_pos : 0 < x) :
  ∃! x, (prism_volume x < 700) ∧ (x - 5 > 0) :=
by
  sorry

end unique_positive_x_for_volume_l36_36980


namespace quadratic_eq_roots_quadratic_eq_range_l36_36279

theorem quadratic_eq_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 2 * x1 + m + 1 = 0 ∧ x2^2 - 2 * x2 + m + 1 = 0 ∧ x1 + 3 * x2 = 2 * m + 8) →
  (m = -1 ∨ m = -2) :=
sorry

theorem quadratic_eq_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 2 * x1 + m + 1 = 0 ∧ x2^2 - 2 * x2 + m + 1 = 0) →
  m ≤ 0 :=
sorry

end quadratic_eq_roots_quadratic_eq_range_l36_36279


namespace jack_initial_yen_l36_36637

theorem jack_initial_yen 
  (pounds yen_per_pound euros pounds_per_euro total_yen : ℕ)
  (h₁ : pounds = 42)
  (h₂ : euros = 11)
  (h₃ : pounds_per_euro = 2)
  (h₄ : yen_per_pound = 100)
  (h₅ : total_yen = 9400) : 
  ∃ initial_yen : ℕ, initial_yen = 3000 :=
by
  sorry

end jack_initial_yen_l36_36637


namespace find_number_l36_36064

theorem find_number (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
sorry

end find_number_l36_36064


namespace gear_q_revolutions_per_minute_l36_36888

-- Define the constants and conditions
def revolutions_per_minute_p : ℕ := 10
def revolutions_per_minute_q : ℕ := sorry
def time_in_minutes : ℝ := 1.5
def extra_revolutions_q : ℕ := 45

-- Calculate the number of revolutions for gear p in 90 seconds
def revolutions_p_in_90_seconds := revolutions_per_minute_p * time_in_minutes

-- Condition that gear q makes exactly 45 more revolutions than gear p in 90 seconds
def revolutions_q_in_90_seconds := revolutions_p_in_90_seconds + extra_revolutions_q

-- Correct answer
def correct_answer : ℕ := 40

-- Prove that gear q makes 40 revolutions per minute
theorem gear_q_revolutions_per_minute : 
    revolutions_per_minute_q = correct_answer :=
sorry

end gear_q_revolutions_per_minute_l36_36888


namespace find_x_l36_36164

noncomputable def S (x : ℝ) : ℝ := 1 + 3 * x + 5 * x^2 + 7 * x^3 + ∑' n, (2 * n - 1) * x^n

theorem find_x (x : ℝ) (h : S x = 16) : x = 3/4 :=
sorry

end find_x_l36_36164


namespace total_tires_mike_changed_l36_36484

theorem total_tires_mike_changed (num_motorcycles : ℕ) (tires_per_motorcycle : ℕ)
                                (num_cars : ℕ) (tires_per_car : ℕ)
                                (total_tires : ℕ) :
  num_motorcycles = 12 →
  tires_per_motorcycle = 2 →
  num_cars = 10 →
  tires_per_car = 4 →
  total_tires = num_motorcycles * tires_per_motorcycle + num_cars * tires_per_car →
  total_tires = 64 := by
  intros h1 h2 h3 h4 h5
  sorry

end total_tires_mike_changed_l36_36484


namespace largest_gcd_l36_36678

theorem largest_gcd (a b : ℕ) (h : a + b = 1008) : ∃ d, d = gcd a b ∧ (∀ d', d' = gcd a b → d' ≤ d) ∧ d = 504 :=
by
  sorry

end largest_gcd_l36_36678


namespace complex_modulus_problem_l36_36916

open Complex

def modulus_of_z (z : ℂ) (h : (z - 2 * I) * (1 - I) = -2) : Prop :=
  abs z = Real.sqrt 2

theorem complex_modulus_problem (z : ℂ) (h : (z - 2 * I) * (1 - I) = -2) : 
  modulus_of_z z h :=
sorry

end complex_modulus_problem_l36_36916


namespace hourly_rate_for_carriage_l36_36638

theorem hourly_rate_for_carriage
  (d : ℕ) (s : ℕ) (f : ℕ) (c : ℕ)
  (h_d : d = 20)
  (h_s : s = 10)
  (h_f : f = 20)
  (h_c : c = 80) :
  (c - f) / (d / s) = 30 := by
  sorry

end hourly_rate_for_carriage_l36_36638


namespace define_interval_l36_36143

theorem define_interval (x : ℝ) : 
  (0 < x + 2) → (0 < 5 - x) → (-2 < x ∧ x < 5) :=
by
  intros h1 h2
  sorry

end define_interval_l36_36143


namespace variance_transformation_l36_36016

theorem variance_transformation (a_1 a_2 a_3 : ℝ) (h : (1 / 3) * ((a_1 - ((a_1 + a_2 + a_3) / 3))^2 + (a_2 - ((a_1 + a_2 + a_3) / 3))^2 + (a_3 - ((a_1 + a_2 + a_3) / 3))^2) = 1) :
  (1 / 3) * ((3 * a_1 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2 + (3 * a_2 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2 + (3 * a_3 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2) = 9 := by 
  sorry

end variance_transformation_l36_36016


namespace sufficient_but_not_necessary_l36_36167

theorem sufficient_but_not_necessary (a : ℝ) : (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  (∀ x : ℝ, (x - 1) * (x - 2) = 0 → x ≠ 2 → x = 1) ∧
  (a = 2 → (1 ≠ 2)) :=
by {
  sorry
}

end sufficient_but_not_necessary_l36_36167


namespace Daria_vacuum_cleaner_problem_l36_36565

theorem Daria_vacuum_cleaner_problem (initial_savings weekly_savings target_savings weeks_needed : ℕ)
  (h1 : initial_savings = 20)
  (h2 : weekly_savings = 10)
  (h3 : target_savings = 120)
  (h4 : weeks_needed = (target_savings - initial_savings) / weekly_savings) : 
  weeks_needed = 10 :=
by
  sorry

end Daria_vacuum_cleaner_problem_l36_36565


namespace unique_solution_pair_l36_36432

theorem unique_solution_pair (x p : ℕ) (hp : Nat.Prime p) (hx : x ≥ 0) (hp2 : p ≥ 2) :
  x * (x + 1) * (x + 2) * (x + 3) = 1679 ^ (p - 1) + 1680 ^ (p - 1) + 1681 ^ (p - 1) ↔ (x = 4 ∧ p = 2) := 
by
  sorry

end unique_solution_pair_l36_36432


namespace smallest_even_divisible_by_20_and_60_l36_36225

theorem smallest_even_divisible_by_20_and_60 : ∃ x, (Even x) ∧ (x % 20 = 0) ∧ (x % 60 = 0) ∧ (∀ y, (Even y) ∧ (y % 20 = 0) ∧ (y % 60 = 0) → x ≤ y) → x = 60 :=
by
  sorry

end smallest_even_divisible_by_20_and_60_l36_36225


namespace boss_total_amount_l36_36464

def number_of_staff : ℕ := 20
def rate_per_day : ℕ := 100
def number_of_days : ℕ := 30
def petty_cash_amount : ℕ := 1000

theorem boss_total_amount (number_of_staff : ℕ) (rate_per_day : ℕ) (number_of_days : ℕ) (petty_cash_amount : ℕ) :
  let total_allowance_one_staff := rate_per_day * number_of_days
  let total_allowance_all_staff := total_allowance_one_staff * number_of_staff
  total_allowance_all_staff + petty_cash_amount = 61000 := by
  sorry

end boss_total_amount_l36_36464


namespace evaluate_expression_l36_36129

theorem evaluate_expression : (36 + 12) / (6 - (2 + 1)) = 16 := by
  sorry

end evaluate_expression_l36_36129


namespace arithmetic_sequence_a4_a5_sum_l36_36154

theorem arithmetic_sequence_a4_a5_sum
  (a_n : ℕ → ℝ)
  (a1_a2_sum : a_n 1 + a_n 2 = -1)
  (a3_val : a_n 3 = 4)
  (h_arith : ∃ d : ℝ, ∀ (n : ℕ), a_n (n + 1) = a_n n + d) :
  a_n 4 + a_n 5 = 17 := 
by
  sorry

end arithmetic_sequence_a4_a5_sum_l36_36154


namespace ratio_of_ripe_mangoes_l36_36183

theorem ratio_of_ripe_mangoes (total_mangoes : ℕ) (unripe_two_thirds : ℚ)
  (kept_unripe_mangoes : ℕ) (mangoes_per_jar : ℕ) (jars_made : ℕ)
  (h1 : total_mangoes = 54)
  (h2 : unripe_two_thirds = 2 / 3)
  (h3 : kept_unripe_mangoes = 16)
  (h4 : mangoes_per_jar = 4)
  (h5 : jars_made = 5) :
  1 / 3 = 18 / 54 :=
sorry

end ratio_of_ripe_mangoes_l36_36183


namespace third_vertex_l36_36100

/-- Two vertices of a right triangle are located at (4, 3) and (0, 0).
The third vertex of the triangle lies on the positive branch of the x-axis.
Determine the coordinates of the third vertex if the area of the triangle is 24 square units. -/
theorem third_vertex (x : ℝ) (h : x > 0) : 
  (1 / 2 * |x| * 3 = 24) → (x, 0) = (16, 0) :=
by
  intro h_area
  sorry

end third_vertex_l36_36100


namespace gcd_max_value_l36_36674

theorem gcd_max_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) : 
  ∃ d, d = Nat.gcd a b ∧ d = 504 :=
by
  sorry

end gcd_max_value_l36_36674


namespace total_heads_of_cabbage_l36_36116

-- Problem definition for the first patch
def first_patch : ℕ := 12 * 15

-- Problem definition for the second patch
def second_patch : ℕ := 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24

-- Problem statement
theorem total_heads_of_cabbage : first_patch + second_patch = 316 := by
  sorry

end total_heads_of_cabbage_l36_36116


namespace limit_at_2_l36_36340

noncomputable def delta (ε : ℝ) : ℝ := ε / 3

theorem limit_at_2 (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ x : ℝ, (0 < |x - 2| ∧ |x - 2| < δ) → |(3 * x^2 - 5 * x - 2) / (x - 2) - 7| < ε :=
by
  let δ := delta ε
  have hδ : δ > 0 := by
    sorry
  use δ, hδ
  intros x hx
  sorry

end limit_at_2_l36_36340


namespace num_ways_first_to_fourth_floor_l36_36376

theorem num_ways_first_to_fourth_floor (floors : ℕ) (staircases_per_floor : ℕ) 
  (H_floors : floors = 4) (H_staircases : staircases_per_floor = 2) : 
  (staircases_per_floor) ^ (floors - 1) = 2^3 := 
by 
  sorry

end num_ways_first_to_fourth_floor_l36_36376


namespace fixed_point_of_function_l36_36368

theorem fixed_point_of_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ x y : ℝ, y = a^(x-1) + 1 ∧ (x, y) = (1, 2) :=
by 
  sorry

end fixed_point_of_function_l36_36368


namespace triangle_perimeters_sum_l36_36067

theorem triangle_perimeters_sum :
  ∃ (t : ℕ),
    (∀ (A B C D : Type) (x y : ℕ), 
      (AB = 7 ∧ BC = 17 ∧ AD = x ∧ CD = x ∧ BD = y ∧ x^2 - y^2 = 240) →
      t = 114) :=
sorry

end triangle_perimeters_sum_l36_36067


namespace fiona_reaches_pad_thirteen_without_predators_l36_36684

noncomputable def probability_reach_pad_thirteen : ℚ := sorry

theorem fiona_reaches_pad_thirteen_without_predators :
  probability_reach_pad_thirteen = 3 / 2048 :=
sorry

end fiona_reaches_pad_thirteen_without_predators_l36_36684


namespace highest_probability_white_ball_l36_36173

theorem highest_probability_white_ball :
  let red_balls := 2
  let black_balls := 3
  let white_balls := 4
  let total_balls := red_balls + black_balls + white_balls
  let prob_red := red_balls / total_balls
  let prob_black := black_balls / total_balls
  let prob_white := white_balls / total_balls
  prob_white > prob_black ∧ prob_black > prob_red :=
by
  sorry

end highest_probability_white_ball_l36_36173


namespace determine_abc_l36_36816

theorem determine_abc (a b c : ℕ) (h1 : a * b * c = 2^4 * 3^2 * 5^3) 
  (h2 : gcd a b = 15) (h3 : gcd a c = 5) (h4 : gcd b c = 20) : 
  a = 15 ∧ b = 60 ∧ c = 20 :=
by
  sorry

end determine_abc_l36_36816


namespace find_base_of_exponent_l36_36453

theorem find_base_of_exponent
  (x : ℝ)
  (h1 : 4 ^ (2 * x + 2) = (some_number : ℝ) ^ (3 * x - 1))
  (x_eq : x = 1) :
  some_number = 16 := 
by
  -- proof steps would go here
  sorry

end find_base_of_exponent_l36_36453


namespace edward_spent_money_l36_36894

-- Definitions based on the conditions
def books := 2
def cost_per_book := 3

-- Statement of the proof problem
theorem edward_spent_money : 
  (books * cost_per_book = 6) :=
by
  -- proof goes here
  sorry

end edward_spent_money_l36_36894


namespace translated_parabola_eq_l36_36519

-- Define the original parabola
def orig_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation function
def translate_upwards (f : ℝ → ℝ) (dy : ℝ) : (ℝ → ℝ) :=
  fun x => f x + dy

-- Define the translated parabola
def translated_parabola := translate_upwards orig_parabola 3

-- State the theorem
theorem translated_parabola_eq:
  translated_parabola = (fun x : ℝ => -2 * x^2 + 3) :=
by
  sorry

end translated_parabola_eq_l36_36519


namespace tan_30_deg_l36_36736

theorem tan_30_deg : 
  let θ := 30 * (Float.pi / 180) in  -- Conversion from degrees to radians
  Float.sin θ = 1 / 2 ∧ Float.cos θ = Float.sqrt 3 / 2 →
  Float.tan θ = Float.sqrt 3 / 3 := by
  intro h
  sorry

end tan_30_deg_l36_36736


namespace equation_solution_system_of_inequalities_solution_l36_36400

theorem equation_solution (x : ℝ) : (3 / (x - 1) = 1 / (2 * x + 3)) ↔ (x = -2) :=
by
  sorry

theorem system_of_inequalities_solution (x : ℝ) : ((3 * x - 1 ≥ x + 1) ∧ (x + 3 > 4 * x - 2)) ↔ (1 ≤ x ∧ x < 5 / 3) :=
by
  sorry

end equation_solution_system_of_inequalities_solution_l36_36400


namespace polynomial_root_divisibility_l36_36649

noncomputable def p (x : ℤ) (a b c : ℤ) : ℤ := x^3 + a * x^2 + b * x + c

theorem polynomial_root_divisibility (a b c : ℤ) (h : ∃ u v : ℤ, p 0 a b c = (u * v * u * v)) :
  2 * (p (-1) a b c) ∣ (p 1 a b c + p (-1) a b c - 2 * (1 + p 0 a b c)) :=
sorry

end polynomial_root_divisibility_l36_36649


namespace perfect_square_problem_l36_36910

-- Define the given conditions and question
theorem perfect_square_problem 
  (a b c : ℕ) 
  (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h_cond: 0 < a^2 + b^2 - a * b * c ∧ a^2 + b^2 - a * b * c ≤ c + 1) : 
  ∃ k : ℕ, k^2 = a^2 + b^2 - a * b * c := 
sorry

end perfect_square_problem_l36_36910


namespace problem_sum_of_relatively_prime_divisors_of_1000_l36_36477

open BigOperators
open Nat

/-- 
Let S be the sum of all numbers of the form a/b, where a and b are 
relatively prime positive divisors of 1000. Prove that the greatest integer 
that does not exceed S/10 is 29. 
-/
theorem problem_sum_of_relatively_prime_divisors_of_1000 :
  let S := ∑ (i j k l : ℕ) in
    finset.Icc 0 3 ×ᶠ finset.Icc 0 3 ×ᶠ finset.Icc 0 3 ×ᶠ finset.Icc 0 3,
    if gcd (2^i * 5^j) (2^k * 5^l) = 1 then (2^i * 5^j) / (2^k * 5^l) else 0
  in floor (S / 10) = 29 := sorry

end problem_sum_of_relatively_prime_divisors_of_1000_l36_36477


namespace number_of_true_statements_l36_36783

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m
def is_odd (n : ℕ) : Prop := ∃ m : ℕ, n = 2 * m + 1
def is_even (n : ℕ) : Prop := ∃ m : ℕ, n = 2 * m

theorem number_of_true_statements : 3 = (ite ((∀ p q : ℕ, is_prime p → is_prime q → is_prime (p * q)) = false) 0 1) +
                                     (ite ((∀ a b : ℕ, is_square a → is_square b → is_square (a * b)) = true) 1 0) +
                                     (ite ((∀ x y : ℕ, is_odd x → is_odd y → is_odd (x * y)) = true) 1 0) +
                                     (ite ((∀ u v : ℕ, is_even u → is_even v → is_even (u * v)) = true) 1 0) :=
by
  sorry

end number_of_true_statements_l36_36783


namespace find_hyperbola_equation_hyperbola_equation_l36_36002

-- Define the original hyperbola
def original_hyperbola (x y : ℝ) := (x^2 / 2) - y^2 = 1

-- Define the new hyperbola with unknown constant m
def new_hyperbola (x y m : ℝ) := (x^2 / (m * 2)) - (y^2 / m) = 1

variable (m : ℝ)

-- The point (2, 0)
def point_on_hyperbola (x y : ℝ) := x = 2 ∧ y = 0

theorem find_hyperbola_equation (h : ∀ (x y : ℝ), point_on_hyperbola x y → new_hyperbola x y m) :
  m = 2 :=
    sorry

theorem hyperbola_equation :
  ∀ (x y : ℝ), (x = 2 ∧ y = 0) → (x^2 / 4 - y^2 / 2 = 1) :=
    sorry

end find_hyperbola_equation_hyperbola_equation_l36_36002


namespace axis_of_symmetry_imp_cond_l36_36961

-- Necessary definitions
variables {p q r s x y : ℝ}

-- Given conditions
def curve_eq (x y p q r s : ℝ) : Prop := y = (2 * p * x + q) / (r * x + 2 * s)
def axis_of_symmetry (x y : ℝ) : Prop := y = x

-- Main statement
theorem axis_of_symmetry_imp_cond (h1 : curve_eq x y p q r s) (h2 : axis_of_symmetry x y) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) : p = -2 * s :=
sorry

end axis_of_symmetry_imp_cond_l36_36961


namespace find_x_l36_36165

noncomputable def S (x : ℝ) : ℝ := 1 + 3 * x + 5 * x^2 + 7 * x^3 + ∑' n, (2 * n - 1) * x^n

theorem find_x (x : ℝ) (h : S x = 16) : x = 3/4 :=
sorry

end find_x_l36_36165


namespace intersection_of_sets_l36_36316

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {0, 1, 2}

theorem intersection_of_sets :
  C = A ∩ B :=
sorry

end intersection_of_sets_l36_36316


namespace problem1_problem2_l36_36421

theorem problem1 : (1 * (-9)) - (-7) + (-6) - 5 = -13 := 
by 
  -- problem1 proof
  sorry

theorem problem2 : ((-5 / 12) + (2 / 3) - (3 / 4)) * (-12) = 6 := 
by 
  -- problem2 proof
  sorry

end problem1_problem2_l36_36421


namespace not_age_of_child_l36_36488

noncomputable def sum_from_1_to_n (n : ℕ) := n * (n + 1) / 2

theorem not_age_of_child (N : ℕ) (S : Finset ℕ) (a b : ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11} ∧
  N = 1100 * a + 11 * b ∧
  a ≠ b ∧
  N ≥ 1000 ∧ N < 10000 ∧
  ((S.sum id) = N) ∧
  (∀ age ∈ S, N % age = 0) →
  10 ∉ S := 
by
  sorry

end not_age_of_child_l36_36488


namespace count_valid_integer_area_triangles_l36_36263

def is_valid_point (x y : ℕ) : Prop :=
  41 * x + y = 2009

def triangle_area (x1 y1 x2 y2 : ℕ) : ℕ :=
  (x1 * y2 - x2 * y1).natAbs / 2

def is_distinct (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 ≠ x2) ∨ (y1 ≠ y2)

def is_integer_area (area : ℕ) : Prop :=
  area > 0

theorem count_valid_integer_area_triangles :
  let points := { p : ℕ × ℕ // is_valid_point p.1 p.2 }
  let valid_triangles :=
    { (p1, p2) : points × points //
      is_distinct p1.val.1 p1.val.2 p2.val.1 p2.val.2 ∧
      is_integer_area (triangle_area p1.val.1 p1.val.2 p2.val.1 p2.val.2) }
  in
  (finset.univ.card : valid_triangles) = 600 :=
by
  sorry

end count_valid_integer_area_triangles_l36_36263


namespace susan_probability_exactly_three_blue_marbles_l36_36078

open ProbabilityTheory

noncomputable def probability_blue_marbles (n_blue n_red : ℕ) (total_trials drawn_blue : ℕ) : ℚ :=
  let total_marbles := n_blue + n_red
  let prob_blue := (n_blue : ℚ) / total_marbles
  let prob_red := (n_red : ℚ) / total_marbles
  let n_comb := Nat.choose total_trials drawn_blue
  (n_comb : ℚ) * (prob_blue ^ drawn_blue) * (prob_red ^ (total_trials - drawn_blue))

theorem susan_probability_exactly_three_blue_marbles :
  probability_blue_marbles 8 7 7 3 = 35 * (1225621 / 171140625) :=
by
  sorry

end susan_probability_exactly_three_blue_marbles_l36_36078


namespace pyramid_height_eq_375_l36_36404

theorem pyramid_height_eq_375 :
  let a := 5 
  let b := 10
  let V_cube := a^3
  let V_pyramid := (b^2 * h) / 3
  V_cube = V_pyramid →
  h = 3.75 :=
by
  let a := 5
  let b := 10
  let V_cube := a^3
  let V_pyramid := (b^2 * h) / 3
  have : V_cube = 125 := by norm_num
  have : V_pyramid = (100 * h) / 3 := by norm_num
  sorry

end pyramid_height_eq_375_l36_36404


namespace min_value_of_f_l36_36205

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem min_value_of_f : 
  ∃ x ∈ Icc (-Real.pi / 2) 0, 
    ∀ y ∈ Icc (-Real.pi / 2) 0, f y ≥ f x ∧ x = -Real.pi / 2 := 
by
  sorry

end min_value_of_f_l36_36205


namespace first_quarter_days_2016_l36_36504

theorem first_quarter_days_2016 : 
  let leap_year := 2016
  let jan_days := 31
  let feb_days := if leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) then 29 else 28
  let mar_days := 31
  (jan_days + feb_days + mar_days) = 91 := 
by
  let leap_year := 2016
  let jan_days := 31
  let feb_days := if leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) then 29 else 28
  let mar_days := 31
  have h_leap_year : leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) := by sorry
  have h_feb_days : feb_days = 29 := by sorry
  have h_first_quarter : jan_days + feb_days + mar_days = 31 + 29 + 31 := by sorry
  have h_sum : 31 + 29 + 31 = 91 := by norm_num
  exact h_sum

end first_quarter_days_2016_l36_36504


namespace walt_total_interest_l36_36652

noncomputable def total_investment : ℝ := 12000
noncomputable def investment_at_7_percent : ℝ := 5500
noncomputable def investment_at_9_percent : ℝ := total_investment - investment_at_7_percent
noncomputable def rate_7_percent : ℝ := 0.07
noncomputable def rate_9_percent : ℝ := 0.09

theorem walt_total_interest :
  let interest_7 : ℝ := investment_at_7_percent * rate_7_percent
  let interest_9 : ℝ := investment_at_9_percent * rate_9_percent
  interest_7 + interest_9 = 970 := by
  sorry

end walt_total_interest_l36_36652


namespace pyramid_height_eq_375_l36_36403

theorem pyramid_height_eq_375 :
  let a := 5 
  let b := 10
  let V_cube := a^3
  let V_pyramid := (b^2 * h) / 3
  V_cube = V_pyramid →
  h = 3.75 :=
by
  let a := 5
  let b := 10
  let V_cube := a^3
  let V_pyramid := (b^2 * h) / 3
  have : V_cube = 125 := by norm_num
  have : V_pyramid = (100 * h) / 3 := by norm_num
  sorry

end pyramid_height_eq_375_l36_36403


namespace exist_pairs_sum_and_diff_l36_36496

theorem exist_pairs_sum_and_diff (N : ℕ) : ∃ a b c d : ℕ, 
  (a + b = c + d) ∧ (a * b + N = c * d ∨ a * b = c * d + N) := sorry

end exist_pairs_sum_and_diff_l36_36496


namespace set_intersection_eq_l36_36808

def A : Set ℝ := {x | |x - 1| ≤ 2}
def B : Set ℝ := {x | x^2 - 4 * x > 0}

theorem set_intersection_eq :
  A ∩ (Set.univ \ B) = {x | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end set_intersection_eq_l36_36808


namespace pet_store_cages_l36_36722

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 78) (h2 : sold_puppies = 30) (h3 : puppies_per_cage = 8) : 
  (initial_puppies - sold_puppies) / puppies_per_cage = 6 := 
by 
  sorry

end pet_store_cages_l36_36722


namespace num_pairs_satisfying_equation_and_sum_leq_50_l36_36160

theorem num_pairs_satisfying_equation_and_sum_leq_50 :
  {p : ℕ × ℕ | let a := p.1; let b := p.2 in a + b ≤ 50 ∧ (a ≠ 0 ∧ b ≠ 0) ∧
                                   (a : ℚ) + (b⁻¹ : ℚ) = 9 * ((a⁻¹ : ℚ) + (b : ℚ))}.card = 5 :=
by
  sorry

end num_pairs_satisfying_equation_and_sum_leq_50_l36_36160


namespace determine_range_of_k_l36_36773

noncomputable def inequality_holds_for_all_x (k : ℝ) : Prop :=
  ∀ (x : ℝ), x^4 + (k - 1) * x^2 + 1 ≥ 0

theorem determine_range_of_k (k : ℝ) : inequality_holds_for_all_x k ↔ k ≥ 1 := sorry

end determine_range_of_k_l36_36773


namespace cubic_poly_real_roots_l36_36241

theorem cubic_poly_real_roots (a b c d : ℝ) (h : a ≠ 0) : 
  ∃ (min_roots max_roots : ℕ), 1 ≤ min_roots ∧ max_roots ≤ 3 ∧ min_roots = 1 ∧ max_roots = 3 :=
by
  sorry

end cubic_poly_real_roots_l36_36241


namespace solve_diamond_l36_36787

theorem solve_diamond : 
  (∃ (Diamond : ℤ), Diamond * 5 + 3 = Diamond * 6 + 2) →
  (∃ (Diamond : ℤ), Diamond = 1) :=
by
  sorry

end solve_diamond_l36_36787


namespace abs_eq_of_sq_eq_l36_36525

theorem abs_eq_of_sq_eq (a b : ℝ) : a^2 = b^2 → |a| = |b| := by
  intro h
  sorry

end abs_eq_of_sq_eq_l36_36525


namespace total_tires_mike_changed_l36_36483

theorem total_tires_mike_changed (num_motorcycles : ℕ) (tires_per_motorcycle : ℕ)
                                (num_cars : ℕ) (tires_per_car : ℕ)
                                (total_tires : ℕ) :
  num_motorcycles = 12 →
  tires_per_motorcycle = 2 →
  num_cars = 10 →
  tires_per_car = 4 →
  total_tires = num_motorcycles * tires_per_motorcycle + num_cars * tires_per_car →
  total_tires = 64 := by
  intros h1 h2 h3 h4 h5
  sorry

end total_tires_mike_changed_l36_36483


namespace arithmetic_sequence_properties_l36_36626

/-- In an arithmetic sequence {a_n}, let S_n represent the sum of the first n terms, 
and it is given that S_6 < S_7 and S_7 > S_8. 
Prove that the correct statements among the given options are: 
1. The common difference d < 0 
2. S_9 < S_6 
3. S_7 is definitively the maximum value among all sums S_n. -/
theorem arithmetic_sequence_properties 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, S (n + 1) = S n + a (n + 1))
  (h_S6_lt_S7 : S 6 < S 7)
  (h_S7_gt_S8 : S 7 > S 8) :
  (a 7 > 0 ∧ a 8 < 0 ∧ ∃ d, ∀ n, a (n + 1) = a n + d ∧ d < 0 ∧ S 9 < S 6 ∧ ∀ n, S n ≤ S 7) :=
by
  -- Proof omitted
  sorry

end arithmetic_sequence_properties_l36_36626


namespace solve_abs_equation_l36_36211

theorem solve_abs_equation (x : ℝ) (h : |2001 * x - 2001| = 2001) : x = 0 ∨ x = 2 := by
  sorry

end solve_abs_equation_l36_36211


namespace jason_cutting_grass_time_l36_36634

-- Conditions
def time_to_cut_one_lawn : ℕ := 30 -- in minutes
def lawns_cut_each_day : ℕ := 8
def days : ℕ := 2
def minutes_in_an_hour : ℕ := 60

-- Proof that the number of hours Jason spends cutting grass over the weekend is 8
theorem jason_cutting_grass_time:
  ((lawns_cut_each_day * days) * time_to_cut_one_lawn) / minutes_in_an_hour = 8 :=
by
  sorry

end jason_cutting_grass_time_l36_36634


namespace trains_crossing_time_l36_36220

theorem trains_crossing_time :
  let length_first_train := 500
  let length_second_train := 800
  let speed_first_train := 80 * (5/18 : ℚ)  -- convert km/hr to m/s
  let speed_second_train := 100 * (5/18 : ℚ)  -- convert km/hr to m/s
  let relative_speed := speed_first_train + speed_second_train
  let total_distance := length_first_train + length_second_train
  let time_taken := total_distance / relative_speed
  time_taken = 26 :=
by
  sorry

end trains_crossing_time_l36_36220


namespace ruth_started_with_89_apples_l36_36195

theorem ruth_started_with_89_apples 
  (initial_apples : ℕ)
  (shared_apples : ℕ)
  (remaining_apples : ℕ)
  (h1 : shared_apples = 5)
  (h2 : remaining_apples = 84)
  (h3 : remaining_apples = initial_apples - shared_apples) : 
  initial_apples = 89 :=
by
  sorry

end ruth_started_with_89_apples_l36_36195


namespace not_invited_students_l36_36174

-- Definition of the problem conditions
def students := 15
def direct_friends_of_mia := 4
def unique_friends_of_each_friend := 2

-- Problem statement
theorem not_invited_students : (students - (1 + direct_friends_of_mia + direct_friends_of_mia * unique_friends_of_each_friend) = 2) :=
by
  sorry

end not_invited_students_l36_36174


namespace distance_from_point_to_focus_l36_36456

theorem distance_from_point_to_focus (P : ℝ × ℝ) (hP : P.2^2 = 8 * P.1) (hX : P.1 = 8) :
  dist P (2, 0) = 10 :=
sorry

end distance_from_point_to_focus_l36_36456


namespace k_sq_geq_25_over_4_l36_36749

theorem k_sq_geq_25_over_4
  (a1 a2 a3 a4 a5 k : ℝ)
  (h1 : |a1 - a2| ≥ 1 ∧ |a1 - a3| ≥ 1 ∧ |a1 - a4| ≥ 1 ∧ |a1 - a5| ≥ 1 ∧
       |a2 - a3| ≥ 1 ∧ |a2 - a4| ≥ 1 ∧ |a2 - a5| ≥ 1 ∧
       |a3 - a4| ≥ 1 ∧ |a3 - a5| ≥ 1 ∧
       |a4 - a5| ≥ 1)
  (h2 : a1 + a2 + a3 + a4 + a5 = 2 * k)
  (h3 : a1^2 + a2^2 + a3^2 + a4^2 + a5^2 = 2 * k^2) :
  k^2 ≥ 25 / 4 :=
sorry

end k_sq_geq_25_over_4_l36_36749


namespace intersection_A_B_eq_C_l36_36318

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
noncomputable def C : Set ℝ := {0, 1, 2}

theorem intersection_A_B_eq_C : (A : Set ℝ) ∩ B = C :=
by {
  sorry
}

end intersection_A_B_eq_C_l36_36318


namespace min_value_xy_expression_l36_36224

theorem min_value_xy_expression (x y : ℝ) : ∃ c : ℝ, (∀ x y : ℝ, (xy - 1)^2 + (x + y)^2 ≥ c) ∧ c = 1 :=
by {
  -- Placeholder for proof
  sorry
}

end min_value_xy_expression_l36_36224


namespace find_x_l36_36891

open Real

noncomputable def satisfies_equation (x : ℝ) : Prop :=
  log (x - 1) / log 3 + log (x^2 - 1) / log (sqrt 3) + log (x - 1) / log (1 / 3) = 3

theorem find_x : ∃ x : ℝ, 1 < x ∧ satisfies_equation x ∧ x = sqrt (1 + 3 * sqrt 3) := by
  sorry

end find_x_l36_36891


namespace abs_a_plus_2_always_positive_l36_36857

theorem abs_a_plus_2_always_positive (a : ℝ) : |a| + 2 > 0 := 
sorry

end abs_a_plus_2_always_positive_l36_36857


namespace inv_eq_self_l36_36937

noncomputable def g (m x : ℝ) : ℝ := (3 * x + 4) / (m * x - 3)

theorem inv_eq_self (m : ℝ) :
  (∀ x : ℝ, g m x = g m (g m x)) ↔ m ∈ Set.Iic (-9 / 4) ∪ Set.Ici (-9 / 4) :=
by
  sorry

end inv_eq_self_l36_36937


namespace beaver_hid_90_carrots_l36_36585

-- Defining the number of burrows and carrot condition homomorphic to the problem
def beaver_carrots (x : ℕ) := 5 * x
def rabbit_carrots (y : ℕ) := 7 * y

-- Stating the main theorem based on conditions derived from the problem
theorem beaver_hid_90_carrots (x y : ℕ) (h1 : beaver_carrots x = rabbit_carrots y) (h2 : y = x - 5) : 
  beaver_carrots x = 90 := 
by 
  sorry

end beaver_hid_90_carrots_l36_36585


namespace vector_sum_l36_36921

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, -2)

theorem vector_sum:
  2 • a + b = (-3, 4) :=
by 
  sorry

end vector_sum_l36_36921


namespace fixed_point_of_function_l36_36094

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : (a^(2-2) - 3) = -2 :=
by
  sorry

end fixed_point_of_function_l36_36094


namespace smaller_angle_of_parallelogram_l36_36034

theorem smaller_angle_of_parallelogram (x : ℝ) (h : x + 3 * x = 180) : x = 45 :=
sorry

end smaller_angle_of_parallelogram_l36_36034


namespace triangle_area_MEQF_l36_36934

theorem triangle_area_MEQF
  (radius_P : ℝ)
  (chord_EF : ℝ)
  (par_EF_MN : Prop)
  (MQ : ℝ)
  (collinear_MQPN : Prop)
  (P MEF : ℝ × ℝ)
  (segment_P_Q : ℝ)
  (EF_length : ℝ)
  (radius_value : radius_P = 10)
  (EF_value : chord_EF = 12)
  (MQ_value : MQ = 20)
  (MN_parallel : par_EF_MN)
  (collinear : collinear_MQPN) :
  ∃ (area : ℝ), area = 48 := 
sorry

end triangle_area_MEQF_l36_36934


namespace keats_library_percentage_increase_l36_36046

theorem keats_library_percentage_increase :
  let total_books_A := 8000
  let total_books_B := 10000
  let total_books_C := 12000
  let initial_bio_A := 0.20 * total_books_A
  let initial_bio_B := 0.25 * total_books_B
  let initial_bio_C := 0.28 * total_books_C
  let total_initial_bio := initial_bio_A + initial_bio_B + initial_bio_C
  let final_bio_A := 0.32 * total_books_A
  let final_bio_B := 0.35 * total_books_B
  let final_bio_C := 0.40 * total_books_C
  --
  let total_final_bio := final_bio_A + final_bio_B + final_bio_C
  let increase_in_bio := total_final_bio - total_initial_bio
  let percentage_increase := (increase_in_bio / total_initial_bio) * 100
  --
  percentage_increase = 45.58 := 
by
  sorry

end keats_library_percentage_increase_l36_36046


namespace smallest_n_for_4n_square_and_5n_cube_l36_36702

theorem smallest_n_for_4n_square_and_5n_cube :
  ∃ (n : ℕ), (n > 0 ∧ (∃ k : ℕ, 4 * n = k^2) ∧ (∃ m : ℕ, 5 * n = m^3)) ∧ n = 400 :=
by
  sorry

end smallest_n_for_4n_square_and_5n_cube_l36_36702


namespace people_joined_l36_36336

theorem people_joined (total_left : ℕ) (total_remaining : ℕ) (Molly_and_parents : ℕ)
  (h1 : total_left = 40) (h2 : total_remaining = 63) (h3 : Molly_and_parents = 3) :
  ∃ n, n = 100 := 
by
  sorry

end people_joined_l36_36336


namespace radius_of_inscribed_circle_is_integer_l36_36350

theorem radius_of_inscribed_circle_is_integer 
  (a b c : ℤ) 
  (h_pythagorean : c^2 = a^2 + b^2) 
  : ∃ r : ℤ, r = (a + b - c) / 2 :=
by
  sorry

end radius_of_inscribed_circle_is_integer_l36_36350


namespace coins_probability_l36_36717

theorem coins_probability :
  let pennies := 3
  let nickels := 5
  let dimes := 7
  let quarters := 4
  let total_coins := pennies + nickels + dimes + quarters
  let total_ways := Nat.choose total_coins 8
  let successful_ways := 2345
  let probability := (successful_ways : ℚ) / total_ways
  probability = 2345 / 75582 :=
by {
  let pennies := 3
  let nickels := 5
  let dimes := 7
  let quarters := 4
  let total_coins := pennies + nickels + dimes + quarters
  let total_ways := Nat.choose total_coins 8
  let successful_ways := 2345
  let probability := (successful_ways : ℚ) / total_ways
  show probability = 2345 / 75582, from sorry
}

end coins_probability_l36_36717


namespace sum_of_perimeters_l36_36965

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) :
  4 * (Real.sqrt 63 + Real.sqrt 22) = 4 * (Real.sqrt x^2 + Real.sqrt y^2) :=
by
  sorry

end sum_of_perimeters_l36_36965


namespace probability_one_solves_l36_36013

theorem probability_one_solves :
  let pA := 0.8
  let pB := 0.7
  (pA * (1 - pB) + pB * (1 - pA)) = 0.38 :=
by
  sorry

end probability_one_solves_l36_36013


namespace even_function_coeff_l36_36169

theorem even_function_coeff (a : ℝ) (h : ∀ x : ℝ, (a-2)*x^2 + (a-1)*x + 3 = (a-2)*(-x)^2 + (a-1)*(-x) + 3) : a = 1 :=
by {
  -- Proof here
  sorry
}

end even_function_coeff_l36_36169


namespace sam_has_75_dollars_l36_36257

variable (S B : ℕ)

def condition1 := B = 2 * S - 25
def condition2 := S + B = 200

theorem sam_has_75_dollars (h1 : condition1 S B) (h2 : condition2 S B) : S = 75 := by
  sorry

end sam_has_75_dollars_l36_36257


namespace simplify_evaluate_expression_l36_36074

theorem simplify_evaluate_expression (a b : ℚ) (h1 : a = -2) (h2 : b = 1/5) :
    2 * a * b^2 - (6 * a^3 * b + 2 * (a * b^2 - (1/2) * a^3 * b)) = 8 := 
by
  sorry

end simplify_evaluate_expression_l36_36074


namespace M_intersection_N_l36_36650

-- Definition of sets M and N
def M : Set ℝ := { x | x^2 + 2 * x - 8 < 0 }
def N : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

-- Goal: Prove that M ∩ N = (0, 2)
theorem M_intersection_N :
  M ∩ N = { y | 0 < y ∧ y < 2 } :=
sorry

end M_intersection_N_l36_36650


namespace quadratic_roots_ratio_l36_36508

theorem quadratic_roots_ratio (a b c : ℝ) (h1 : ∀ (s1 s2 : ℝ), s1 * s2 = a → s1 + s2 = -c → 3 * s1 + 3 * s2 = -a → 9 * s1 * s2 = b) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
  b / c = 27 := sorry

end quadratic_roots_ratio_l36_36508


namespace pythagorean_triangle_inscribed_circle_radius_is_integer_l36_36343

theorem pythagorean_triangle_inscribed_circle_radius_is_integer 
  (a b c : ℕ)
  (h1 : c^2 = a^2 + b^2) 
  (h2 : r = (a + b - c) / 2) :
  ∃ (r : ℕ), r = (a + b - c) / 2 :=
sorry

end pythagorean_triangle_inscribed_circle_radius_is_integer_l36_36343


namespace infinite_sequence_exists_l36_36357

noncomputable def has_k_distinct_positive_divisors (n k : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card ≥ k ∧ ∀ d ∈ S, d ∣ n

theorem infinite_sequence_exists :
    ∃ (a : ℕ → ℕ),
    (∀ k : ℕ, 0 < k → ∃ n : ℕ, (a n > 0) ∧ has_k_distinct_positive_divisors (a n ^ 2 + a n + 2023) k) :=
  sorry

end infinite_sequence_exists_l36_36357


namespace max_value_of_sequence_l36_36596

theorem max_value_of_sequence : 
  ∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → (∃ (a : ℝ), a = (m / (m^2 + 6 : ℝ)) ∧ a ≤ (n / (n^2 + 6 : ℝ))) :=
sorry

end max_value_of_sequence_l36_36596


namespace octopus_legs_l36_36377

-- Definitions of octopus behavior based on the number of legs
def tells_truth (legs: ℕ) : Prop := legs = 6 ∨ legs = 8
def lies (legs: ℕ) : Prop := legs = 7

-- Statements made by the octopuses
def blue_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 28
def green_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 27
def yellow_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 26
def red_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 25

noncomputable def legs_b := 7
noncomputable def legs_g := 6
noncomputable def legs_y := 7
noncomputable def legs_r := 7

-- Main theorem
theorem octopus_legs : 
  (tells_truth legs_g) ∧ 
  (lies legs_b) ∧ 
  (lies legs_y) ∧ 
  (lies legs_r) ∧ 
  blue_statement legs_b legs_g legs_y legs_r ∧ 
  green_statement legs_b legs_g legs_y legs_r ∧ 
  yellow_statement legs_b legs_g legs_y legs_r ∧ 
  red_statement legs_b legs_g legs_y legs_r := 
by 
  sorry

end octopus_legs_l36_36377


namespace amount_lent_by_A_to_B_l36_36547

theorem amount_lent_by_A_to_B
  (P : ℝ)
  (H1 : P * 0.115 * 3 - P * 0.10 * 3 = 1125) :
  P = 25000 :=
by
  sorry

end amount_lent_by_A_to_B_l36_36547


namespace other_cube_side_length_l36_36663

theorem other_cube_side_length (s_1 s_2 : ℝ) (h1 : s_1 = 1) (h2 : 6 * s_2^2 / 6 = 36) : s_2 = 6 :=
by
  sorry

end other_cube_side_length_l36_36663


namespace evaluate_root_l36_36756

theorem evaluate_root : 64 ^ (5 / 6 : ℝ) = 32 :=
by sorry

end evaluate_root_l36_36756


namespace shape_of_phi_eq_d_in_spherical_coordinates_l36_36274

theorem shape_of_phi_eq_d_in_spherical_coordinates (d : ℝ) : 
  (∃ (ρ θ : ℝ), ∀ (φ : ℝ), φ = d) ↔ ( ∃ cone_vertex : ℝ × ℝ × ℝ, ∃ opening_angle : ℝ, cone_vertex = (0, 0, 0) ∧ opening_angle = d) :=
sorry

end shape_of_phi_eq_d_in_spherical_coordinates_l36_36274


namespace number_of_beavers_l36_36126

-- Definitions of the problem conditions
def total_workers : Nat := 862
def number_of_spiders : Nat := 544

-- The statement we need to prove
theorem number_of_beavers : (total_workers - number_of_spiders) = 318 := 
by 
  sorry

end number_of_beavers_l36_36126


namespace samantha_more_posters_l36_36190

theorem samantha_more_posters :
  ∃ S : ℕ, S > 18 ∧ 18 + S = 51 ∧ S - 18 = 15 :=
by
  sorry

end samantha_more_posters_l36_36190


namespace crayons_lost_l36_36495

theorem crayons_lost (initial_crayons ending_crayons : ℕ) (h_initial : initial_crayons = 253) (h_ending : ending_crayons = 183) : (initial_crayons - ending_crayons) = 70 :=
by
  sorry

end crayons_lost_l36_36495


namespace leopards_arrangement_l36_36059

theorem leopards_arrangement :
  let total_leopards := 9
  let ends_leopards := 2
  let middle_leopard := 1
  let remaining_leopards := total_leopards - ends_leopards - middle_leopard
  (2 * 1 * (Nat.factorial remaining_leopards) = 1440) := by
  sorry

end leopards_arrangement_l36_36059


namespace breakfast_cost_l36_36591

theorem breakfast_cost :
  ∀ (muffin_cost fruit_cup_cost : ℕ) (francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups : ℕ),
  muffin_cost = 2 ∧ fruit_cup_cost = 3 ∧ francis_muffins = 2 ∧ francis_fruit_cups = 2 ∧ kiera_muffins = 2 ∧ kiera_fruit_cups = 1
  → (francis_muffins * muffin_cost + francis_fruit_cups * fruit_cup_cost + kiera_muffins * muffin_cost + kiera_fruit_cups * fruit_cup_cost = 17) :=
by
  intros muffin_cost fruit_cup_cost francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups
  intro cond
  cases cond with muffin_cost_eq rest
  cases rest with fruit_cup_cost_eq rest
  cases rest with francis_muffins_eq rest
  cases rest with francis_fruit_cups_eq rest
  cases rest with kiera_muffins_eq kiera_fruit_cups_eq

  rw [muffin_cost_eq, fruit_cup_cost_eq, francis_muffins_eq, francis_fruit_cups_eq, kiera_muffins_eq, kiera_fruit_cups_eq]
  norm_num
  sorry

end breakfast_cost_l36_36591


namespace fraction_students_walk_home_l36_36416

theorem fraction_students_walk_home :
  let bus := 1/3
  let auto := 1/5
  let bicycle := 1/8
  let total_students := 1
  let other_transportation := bus + auto + bicycle
  let walk_home := total_students - other_transportation
  walk_home = 41/120 :=
by 
  let bus := 1/3
  let auto := 1/5
  let bicycle := 1/8
  let total_students := 1
  let other_transportation := bus + auto + bicycle
  let walk_home := total_students - other_transportation
  have h_bus : bus = 40 / 120 := by sorry
  have h_auto : auto = 24 / 120 := by sorry
  have h_bicycle : bicycle = 15 / 120 := by sorry
  have h_total_transportation : other_transportation = 40 / 120 + 24 / 120 + 15 / 120 := by sorry
  have h_other_transportation_sum : other_transportation = 79 / 120 := by sorry
  have h_walk_home : walk_home = 1 - 79 / 120 := by sorry
  have h_walk_home_simplified : walk_home = 41 / 120 := by sorry
  exact h_walk_home_simplified

end fraction_students_walk_home_l36_36416


namespace range_of_a_l36_36809

noncomputable def f (x a b : ℝ) : ℝ := (2 * x^2 - a * x + b) * Real.log (x - 1)

theorem range_of_a (a b : ℝ) (h1 : ∀ x > 1, f x a b ≥ 0) : a ≤ 6 :=
by 
  let x := 2
  have hb_eq : b = 2 * a - 8 :=
    by sorry
  have ha_le_6 : a ≤ 6 :=
    by sorry
  exact ha_le_6

end range_of_a_l36_36809


namespace length_of_train_l36_36250

theorem length_of_train
  (T_platform : ℕ)
  (T_pole : ℕ)
  (L_platform : ℕ)
  (h1: T_platform = 39)
  (h2: T_pole = 18)
  (h3: L_platform = 350)
  (L : ℕ)
  (h4 : 39 * L = 18 * (L + 350)) :
  L = 300 :=
by
  sorry

end length_of_train_l36_36250


namespace total_students_l36_36553

theorem total_students (rank_right rank_left : ℕ) (h1 : rank_right = 16) (h2 : rank_left = 6) : rank_right + rank_left - 1 = 21 := by
  sorry

end total_students_l36_36553


namespace correctly_calculated_value_l36_36527

theorem correctly_calculated_value (x : ℝ) (hx : x + 0.42 = 0.9) : (x - 0.42) + 0.5 = 0.56 := by
  -- proof to be provided
  sorry

end correctly_calculated_value_l36_36527


namespace polynomial_remainder_l36_36522

theorem polynomial_remainder (x : ℤ) : 
  (2 * x + 3) ^ 504 % (x^2 - x + 1) = (16 * x + 5) :=
by
  sorry

end polynomial_remainder_l36_36522


namespace rate_is_15_l36_36719

variable (sum : ℝ) (interest12 : ℝ) (interest_r : ℝ) (r : ℝ)

-- Given conditions
def conditions : Prop :=
  sum = 7000 ∧
  interest12 = 7000 * 0.12 * 2 ∧
  interest_r = 7000 * (r / 100) * 2 ∧
  interest_r = interest12 + 420

-- The rate to prove
def rate_to_prove : Prop := r = 15

theorem rate_is_15 : conditions sum interest12 interest_r r → rate_to_prove r := 
by
  sorry

end rate_is_15_l36_36719


namespace susan_took_longer_l36_36101
variables (M S J T x : ℕ)
theorem susan_took_longer (h1 : M = 2 * S)
                         (h2 : S = J + x)
                         (h3 : J = 30)
                         (h4 : T = M - 7)
                         (h5 : M + S + J + T = 223) : x = 10 :=
sorry

end susan_took_longer_l36_36101


namespace find_x_l36_36853

theorem find_x (x : ℝ) : x - (502 / 100.4) = 5015 → x = 5020 :=
by
  sorry

end find_x_l36_36853


namespace total_pages_in_storybook_l36_36858

theorem total_pages_in_storybook
  (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) (Sₙ : ℕ) 
  (h₁ : a₁ = 12)
  (h₂ : d = 1)
  (h₃ : aₙ = 26)
  (h₄ : aₙ = a₁ + (n - 1) * d)
  (h₅ : Sₙ = n * (a₁ + aₙ) / 2) :
  Sₙ = 285 :=
by
  sorry

end total_pages_in_storybook_l36_36858


namespace tan_30_deg_l36_36737

theorem tan_30_deg : 
  let θ := 30 * (Float.pi / 180) in  -- Conversion from degrees to radians
  Float.sin θ = 1 / 2 ∧ Float.cos θ = Float.sqrt 3 / 2 →
  Float.tan θ = Float.sqrt 3 / 3 := by
  intro h
  sorry

end tan_30_deg_l36_36737


namespace min_x_div_y_l36_36009

theorem min_x_div_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + y = 2) : ∃c: ℝ, c = 1 ∧ ∀(a: ℝ), x = a → y = 1 → a/y ≥ c :=
by
  sorry

end min_x_div_y_l36_36009


namespace find_radius_of_large_circle_l36_36383

noncomputable def radius_of_large_circle (r : ℝ) : Prop :=
  let r_A := 3
  let r_B := 2
  let d := 6
  (r - r_A)^2 + (r - r_B)^2 + 2 * (r - r_A) * (r - r_B) = d^2 ∧
  r = (5 + Real.sqrt 33) / 2

theorem find_radius_of_large_circle : ∃ (r : ℝ), radius_of_large_circle r :=
by {
  sorry
}

end find_radius_of_large_circle_l36_36383


namespace blue_black_pen_ratio_l36_36936

theorem blue_black_pen_ratio (B K R : ℕ) 
  (h1 : B + K + R = 31) 
  (h2 : B = 18) 
  (h3 : K = R + 5) : 
  B / Nat.gcd B K = 2 ∧ K / Nat.gcd B K = 1 := 
by 
  sorry

end blue_black_pen_ratio_l36_36936


namespace smaller_of_x_and_y_l36_36850

theorem smaller_of_x_and_y 
  (x y a b c d : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < b + 1) 
  (h3 : x + y = c) 
  (h4 : x - y = d) 
  (h5 : x / y = a / (b + 1)) :
  min x y = (ac/(a + b + 1)) := 
by
  sorry

end smaller_of_x_and_y_l36_36850


namespace percent_of_d_is_e_l36_36646

variable (a b c d e : ℝ)
variable (h1 : d = 0.40 * a)
variable (h2 : d = 0.35 * b)
variable (h3 : e = 0.50 * b)
variable (h4 : e = 0.20 * c)
variable (h5 : c = 0.30 * a)
variable (h6 : c = 0.25 * b)

theorem percent_of_d_is_e : (e / d) * 100 = 15 :=
by sorry

end percent_of_d_is_e_l36_36646


namespace total_students_at_concert_l36_36844

-- Define the number of buses
def num_buses : ℕ := 8

-- Define the number of students per bus
def students_per_bus : ℕ := 45

-- State the theorem with the conditions and expected result
theorem total_students_at_concert : (num_buses * students_per_bus) = 360 := by
  -- Proof is not required as per the instructions; replace with 'sorry'
  sorry

end total_students_at_concert_l36_36844


namespace trapezium_area_l36_36576

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) :
  (1 / 2) * (a + b) * h = 247 :=
by
  sorry

end trapezium_area_l36_36576


namespace product_of_integers_eq_expected_result_l36_36435

theorem product_of_integers_eq_expected_result
  (E F G H I : ℚ) 
  (h1 : E + F + G + H + I = 80) 
  (h2 : E + 2 = F - 2) 
  (h3 : F - 2 = G * 2) 
  (h4 : G * 2 = H * 3) 
  (h5 : H * 3 = I / 2) :
  E * F * G * H * I = (5120000 / 81) := 
by 
  sorry

end product_of_integers_eq_expected_result_l36_36435


namespace negation_of_inverse_true_l36_36793

variables (P : Prop)

theorem negation_of_inverse_true (h : ¬P → false) : ¬P := by
  sorry

end negation_of_inverse_true_l36_36793


namespace rationalize_denominator_l36_36953

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5 in
  B < D ∧
  (12/5 * Real.sqrt 7) = ((1:ℚ) * Real.sqrt B / E * A) * (-1) ∧
  (9/5 * Real.sqrt 13) = (1:ℚ * Real.sqrt D / E * C) ∧
  (A + B + C + D + E = 22) :=
by
  sorry

end rationalize_denominator_l36_36953


namespace negation_of_exists_geq_prop_l36_36370

open Classical

variable (P : Prop) (Q : Prop)

-- Original proposition:
def exists_geq_prop : Prop := 
  ∃ x : ℝ, x^2 + x + 1 ≥ 0

-- Its negation:
def forall_lt_neg : Prop :=
  ∀ x : ℝ, x^2 + x + 1 < 0

-- The theorem to prove:
theorem negation_of_exists_geq_prop : ¬ exists_geq_prop ↔ forall_lt_neg := 
by 
  -- The proof steps will be filled in here
  sorry

end negation_of_exists_geq_prop_l36_36370


namespace equal_powers_equal_elements_l36_36904

theorem equal_powers_equal_elements
  (a : Fin 17 → ℕ)
  (h : ∀ i : Fin 17, a i ^ a (i + 1) % 17 = a ((i + 1) % 17) ^ a ((i + 2) % 17) % 17)
  : ∀ i j : Fin 17, a i = a j :=
by
  sorry

end equal_powers_equal_elements_l36_36904


namespace time_diff_is_6_l36_36425

-- Define the speeds for the different sails
def speed_of_large_sail : ℕ := 50
def speed_of_small_sail : ℕ := 20

-- Define the distance of the trip
def trip_distance : ℕ := 200

-- Calculate the time for each sail
def time_large_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed
def time_small_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Define the time difference
def time_difference (distance : ℕ) (speed_large : ℕ) (speed_small : ℕ) : ℕ := 
  (distance / speed_small) - (distance / speed_large)

-- Prove that the time difference between the large and small sails is 6 hours
theorem time_diff_is_6 : time_difference trip_distance speed_of_large_sail speed_of_small_sail = 6 := by
  -- useful := time_difference trip_distance speed_of_large_sail speed_of_small_sail,
  -- change useful with 6,
  sorry

end time_diff_is_6_l36_36425


namespace remainder_of_sum_mod_13_l36_36854

theorem remainder_of_sum_mod_13 {a b c d e : ℕ} 
  (h1 : a % 13 = 3) 
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7) 
  (h4 : d % 13 = 9) 
  (h5 : e % 13 = 11) : 
  (a + b + c + d + e) % 13 = 9 :=
by
  sorry

end remainder_of_sum_mod_13_l36_36854


namespace tan_30_eq_sqrt3_div_3_l36_36741

/-- Statement that proves the value of tang of 30 degrees, given the cosine
    and sine values. -/
theorem tan_30_eq_sqrt3_div_3 
  (cos_30 : Real) (sin_30 : Real) 
  (hcos : cos_30 = Real.sqrt 3 / 2) 
  (hsin : sin_30 = 1 / 2) : 
    Real.tan 30 = Real.sqrt 3 / 3 := 
by 
  sorry

end tan_30_eq_sqrt3_div_3_l36_36741


namespace rain_at_least_once_l36_36831

theorem rain_at_least_once (p : ℚ) (h : p = 3/4) : 
    (1 - (1 - p)^4) = 255/256 :=
by
  sorry

end rain_at_least_once_l36_36831


namespace total_volume_is_correct_l36_36885

theorem total_volume_is_correct :
  let carl_side := 3
  let carl_count := 3
  let kate_side := 1.5
  let kate_count := 4
  let carl_volume := carl_count * carl_side ^ 3
  let kate_volume := kate_count * kate_side ^ 3
  carl_volume + kate_volume = 94.5 :=
by
  sorry

end total_volume_is_correct_l36_36885


namespace distance_between_points_l36_36975

-- Define the points
def point1 := (1 : ℤ, 3 : ℤ)
def point2 := (-5 : ℤ, 7 : ℤ)

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℤ × ℤ) : ℤ := 
  Int.sqrt (((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2).toNat)

-- Define the problem statement
theorem distance_between_points : distance point1 point2 = 2 * (Int.sqrt 13) := by
  sorry

end distance_between_points_l36_36975


namespace problem_l36_36275

noncomputable def h (p x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 15

noncomputable def k (q r x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 150*x + r

theorem problem
  (p q r : ℝ)
  (h_has_distinct_roots: ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ h p a = 0 ∧ h p b = 0 ∧ h p c = 0)
  (h_roots_are_k_roots: ∀ x, h p x = 0 → k q r x = 0) :
  k q r 1 = -3322.25 :=
sorry

end problem_l36_36275


namespace strawberry_quality_meets_standard_l36_36210

def acceptable_weight_range (w : ℝ) : Prop :=
  4.97 ≤ w ∧ w ≤ 5.03

theorem strawberry_quality_meets_standard :
  acceptable_weight_range 4.98 :=
by
  sorry

end strawberry_quality_meets_standard_l36_36210


namespace first_term_correct_l36_36213

noncomputable def first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^3 / (1 - (r^3)) = 80) : ℝ :=
a

theorem first_term_correct (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^3 / (1 - (r^3)) = 80) :
  first_term a r h1 h2 = 3.42 :=
sorry

end first_term_correct_l36_36213


namespace no_such_sequence_exists_l36_36182

theorem no_such_sequence_exists :
  ¬ ∃ (a : ℕ → ℤ), (∀ n, a n ≠ 0) ∧ 
    (∀ n ≥ 2020, ∃ r : ℝ, 
      (∃ m, m ≥ n ∧ Polynomial.eval r (∑ i in Finset.range (m + 1), a i * X ^ i) = 0) ∧ 
      |r| > 2.001) :=
by {
    sorry
}

end no_such_sequence_exists_l36_36182


namespace one_point_one_billion_scientific_notation_l36_36079

theorem one_point_one_billion_scientific_notation :
  ∃ (n : ℝ), n = 1.1 * 10^9 ∧ scientific_notation 1.1e9 n :=
sorry

end one_point_one_billion_scientific_notation_l36_36079


namespace simplify_fraction_l36_36073

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l36_36073


namespace log_div_sqrt_defined_l36_36142

theorem log_div_sqrt_defined (x : ℝ) : -2 < x ∧ x < 5 ↔ ∃ y : ℝ, y = x ∧ ∃ z : ℝ, z = 5-x ∧ log(z) / sqrt(x+2) ∈ ℝ :=
by
  sorry

end log_div_sqrt_defined_l36_36142


namespace cost_price_of_book_l36_36983

theorem cost_price_of_book 
  (C : ℝ)
  (h1 : ∃ C, C > 0)
  (h2 : 1.10 * C = 1.15 * C - 120) :
  C = 2400 :=
sorry

end cost_price_of_book_l36_36983


namespace Janet_saves_154_minutes_per_week_l36_36130

-- Definitions for the time spent on each activity daily
def timeLookingForKeys := 8 -- minutes
def timeComplaining := 3 -- minutes
def timeSearchingForPhone := 5 -- minutes
def timeLookingForWallet := 4 -- minutes
def timeSearchingForSunglasses := 2 -- minutes

-- Total time spent daily on these activities
def totalDailyTime := timeLookingForKeys + timeComplaining + timeSearchingForPhone + timeLookingForWallet + timeSearchingForSunglasses
-- Time savings calculation for a week
def weeklySaving := totalDailyTime * 7

-- The proof statement that Janet will save 154 minutes every week
theorem Janet_saves_154_minutes_per_week : weeklySaving = 154 := by
  sorry

end Janet_saves_154_minutes_per_week_l36_36130


namespace greatest_integer_less_than_M_over_100_l36_36911

theorem greatest_integer_less_than_M_over_100
  (h : (1/(Nat.factorial 3 * Nat.factorial 18) + 1/(Nat.factorial 4 * Nat.factorial 17) + 
        1/(Nat.factorial 5 * Nat.factorial 16) + 1/(Nat.factorial 6 * Nat.factorial 15) + 
        1/(Nat.factorial 7 * Nat.factorial 14) + 1/(Nat.factorial 8 * Nat.factorial 13) + 
        1/(Nat.factorial 9 * Nat.factorial 12) + 1/(Nat.factorial 10 * Nat.factorial 11) = 
        1/(Nat.factorial 2 * Nat.factorial 19) * (M : ℚ))) :
  ⌊M / 100⌋ = 499 :=
by
  sorry

end greatest_integer_less_than_M_over_100_l36_36911


namespace hexagon_coloring_count_l36_36266

def num_possible_colorings : Nat :=
by
  /- There are 7 choices for first vertex A.
     Once A is chosen, there are 6 choices for the remaining vertices B, C, D, E, F considering the diagonal restrictions. -/
  let total_colorings := 7 * 6 ^ 5
  let restricted_colorings := 7 * 6 ^ 3
  let valid_colorings := total_colorings - restricted_colorings
  exact valid_colorings

theorem hexagon_coloring_count : num_possible_colorings = 52920 :=
  by
    /- Computation steps above show that the number of valid colorings is 52920 -/
    sorry   -- Proof computation already indicated

end hexagon_coloring_count_l36_36266


namespace find_k_range_l36_36286

theorem find_k_range (k : ℝ) : 
  (∃ x y : ℝ, y = -2 * x + 3 * k + 14 ∧ x - 4 * y = -3 * k - 2 ∧ x > 0 ∧ y < 0) ↔ -6 < k ∧ k < -2 :=
by
  sorry

end find_k_range_l36_36286


namespace find_x3_l36_36687

noncomputable def x3 : ℝ :=
  Real.log ((2 / 3) + (1 / 3) * Real.exp 2)

theorem find_x3 
  (x1 x2 : ℝ)
  (h1 : x1 = 0)
  (h2 : x2 = 2)
  (A : ℝ × ℝ := (x1, Real.exp x1))
  (B : ℝ × ℝ := (x2, Real.exp x2))
  (C : ℝ × ℝ := ((2 * A.1 + B.1) / 3, (2 * A.2 + B.2) / 3))
  (yC : ℝ := (2 / 3) * A.2 + (1 / 3) * B.2)
  (E : ℝ × ℝ := (x3, yC)) :
  E.1 = Real.log ((2 / 3) + (1 / 3) * Real.exp x2) := sorry

end find_x3_l36_36687


namespace rain_at_least_once_prob_l36_36837

theorem rain_at_least_once_prob (p : ℚ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 :=
by {
  -- Implementation of Lean code is not required as per instructions.
  sorry
}

end rain_at_least_once_prob_l36_36837


namespace age_difference_l36_36372

/-- 
The overall age of x and y is some years greater than the overall age of y and z. Z is 12 years younger than X.
Prove: The overall age of x and y is 12 years greater than the overall age of y and z.
-/
theorem age_difference {X Y Z : ℕ} (h1: X + Y > Y + Z) (h2: Z = X - 12) : 
  (X + Y) - (Y + Z) = 12 :=
by 
  -- proof goes here
  sorry

end age_difference_l36_36372


namespace find_age_of_b_l36_36958

variables (A B C : ℕ)

def average_abc (A B C : ℕ) : Prop := (A + B + C) / 3 = 28
def average_ac (A C : ℕ) : Prop := (A + C) / 2 = 29

theorem find_age_of_b (h1 : average_abc A B C) (h2 : average_ac A C) : B = 26 :=
by
  sorry

end find_age_of_b_l36_36958


namespace work_rate_b_l36_36528

theorem work_rate_b (W : ℝ) (A B C : ℝ) :
  (A = W / 11) → 
  (C = W / 55) →
  (8 * A + 4 * B + 4 * C = W) →
  B = W / (2420 / 341) :=
by
  intros hA hC hWork
  -- We start with the given assumptions and work towards showing B = W / (2420 / 341)
  sorry

end work_rate_b_l36_36528


namespace cos_C_correct_l36_36472

noncomputable def cos_C (B : ℝ) (AD BD : ℝ) : ℝ :=
  let sinB := Real.sin B
  let angleBAC := (2 : ℝ) * Real.arcsin ((Real.sqrt 3 / 3) * (sinB / 2)) -- derived from bisector property.
  let cosA := (2 : ℝ) * Real.cos angleBAC / 2 - 1
  let sinA := 2 * Real.sin angleBAC / 2 * Real.cos angleBAC / 2
  let cos2thirds := -1 / 2
  let sin2thirds := Real.sqrt 3 / 2
  cos2thirds * cosA + sin2thirds * sinA

theorem cos_C_correct : 
  ∀ (π : ℝ), 
  ∀ (A B C : ℝ),
  B = π / 3 →
  ∀ (AD : ℝ), AD = 3 →
  ∀ (BD : ℝ), BD = 2 →
  cos_C B AD BD = (2 * Real.sqrt 6 - 1) / 6 :=
by
  intros π A B C hB angleBisectorI hAD hBD
  sorry

end cos_C_correct_l36_36472


namespace visitors_correct_l36_36115

def visitors_that_day : ℕ := 92
def visitors_previous_day : ℕ := 419
def total_visitors_before_that_day : ℕ := 522
def visitors_two_days_before : ℕ := total_visitors_before_that_day - visitors_previous_day - visitors_that_day

theorem visitors_correct : visitors_two_days_before = 11 := by
  -- Sorry, proof to be filled in
  sorry

end visitors_correct_l36_36115


namespace parabola_hyperbola_coincide_directrix_l36_36042

noncomputable def parabola_directrix (p : ℝ) : ℝ := -p / 2
noncomputable def hyperbola_directrix : ℝ := -3 / 2

theorem parabola_hyperbola_coincide_directrix (p : ℝ) (hp : 0 < p) 
  (h_eq : parabola_directrix p = hyperbola_directrix) : p = 3 :=
by
  have hp_directrix : parabola_directrix p = -p / 2 := rfl
  have h_directrix : hyperbola_directrix = -3 / 2 := rfl
  rw [hp_directrix, h_directrix] at h_eq
  sorry

end parabola_hyperbola_coincide_directrix_l36_36042


namespace necessarily_positive_y_plus_xsq_l36_36358

theorem necessarily_positive_y_plus_xsq {x y z : ℝ} 
  (hx : 0 < x ∧ x < 2) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 1) : 
  y + x^2 > 0 :=
sorry

end necessarily_positive_y_plus_xsq_l36_36358


namespace cone_radius_l36_36606

theorem cone_radius (r l : ℝ)
  (h1 : 6 * Real.pi = Real.pi * r^2 + Real.pi * r * l)
  (h2 : 2 * Real.pi * r = Real.pi * l) :
  r = Real.sqrt 2 :=
by
  sorry

end cone_radius_l36_36606


namespace hyperbola_axes_asymptotes_ellipse_equation_intersect_condition_l36_36280

theorem hyperbola_axes_asymptotes :
  let a := sqrt 3
  let b := 1
  let c := 2
  in (2 * a = 2 * sqrt 3) ∧
     (2 * b = 2) ∧
     (∀ x, (y = sqrt 3 * x ∨ y = -sqrt 3 * x) ↔ (y^2 / 3 - x^2 = 0)) := by
  sorry

theorem ellipse_equation :
  (∀ (x y : ℝ), (x^2 + y^2 / 4 = 1) ↔
    ((0, - sqrt 3), (0, sqrt 3), 2) ∧ (sqrt 4 - 3 = 1)) := by sorry

theorem intersect_condition :
  (∀ (m : ℝ), (∃ x y : ℝ, y = x + m ∧ x^2 + y^2 / 4 = 1) ↔
    (-sqrt 5 ≤ m ∧ m ≤ sqrt 5)) := by
  sorry

end hyperbola_axes_asymptotes_ellipse_equation_intersect_condition_l36_36280


namespace number_of_boys_in_first_group_l36_36452

-- Define the daily work ratios
variables (M B : ℝ) (h_ratio : M = 2 * B)

-- Define the number of boys in the first group
variable (x : ℝ)

-- Define the conditions provided by the problem
variables (h1 : 5 * (12 * M + x * B) = 4 * (13 * M + 24 * B))

-- State the theorem and include the correct answer
theorem number_of_boys_in_first_group (M B : ℝ) (h_ratio : M = 2 * B) (x : ℝ)
    (h1 : 5 * (12 * M + x * B) = 4 * (13 * M + 24 * B)) 
    : x = 16 := 
by 
    sorry

end number_of_boys_in_first_group_l36_36452


namespace trapezium_area_l36_36577

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) :
  (1 / 2) * (a + b) * h = 247 :=
by
  rw [ha, hb, hh]
  norm_num
  sorry

end trapezium_area_l36_36577


namespace example_3_is_analogical_reasoning_l36_36255

-- Definitions based on the conditions of the problem:
def is_analogical_reasoning (reasoning: String): Prop :=
  reasoning = "from one specific case to another similar specific case"

-- Example of reasoning given in the problem.
def example_3 := "From the fact that the sum of the distances from a point inside an equilateral triangle to its three sides is a constant, it is concluded that the sum of the distances from a point inside a regular tetrahedron to its four faces is a constant."

-- Proof statement based on the conditions and correct answer.
theorem example_3_is_analogical_reasoning: is_analogical_reasoning example_3 :=
by 
  sorry

end example_3_is_analogical_reasoning_l36_36255


namespace sufficient_and_necessary_l36_36145

theorem sufficient_and_necessary (a b : ℝ) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end sufficient_and_necessary_l36_36145


namespace time_to_pass_telegraph_post_l36_36984

def conversion_factor_km_per_hour_to_m_per_sec := 1000 / 3600

noncomputable def train_length := 70
noncomputable def train_speed_kmph := 36

noncomputable def train_speed_m_per_sec := train_speed_kmph * conversion_factor_km_per_hour_to_m_per_sec

theorem time_to_pass_telegraph_post : (train_length / train_speed_m_per_sec) = 7 := by
  sorry

end time_to_pass_telegraph_post_l36_36984


namespace exists_clique_of_7_l36_36215

variables {Employee : Type} [fintype Employee]

-- Definitions and assumptions drawn from the conditions
def employees : finset Employee := finset.univ
def knows (a b : Employee) : Prop := sorry

axiom total_employees : finset.card employees = 2023
axiom knows_exactly_1686 (e : Employee) : (finset.filter (knows e) employees).card = 1686
axiom symmetric_knowing : ∀ {a b : Employee}, knows a b → knows b a

-- Proof statement
theorem exists_clique_of_7 :
  ∃ S : finset Employee, S.card = 7 ∧ ∀ (a b : Employee), a ∈ S → b ∈ S → a ≠ b → knows a b :=
sorry

end exists_clique_of_7_l36_36215


namespace cost_of_each_box_of_pencils_l36_36110

-- Definitions based on conditions
def cartons_of_pencils : ℕ := 20
def boxes_per_carton_of_pencils : ℕ := 10
def cartons_of_markers : ℕ := 10
def boxes_per_carton_of_markers : ℕ := 5
def cost_per_carton_of_markers : ℕ := 4
def total_spent : ℕ := 600

-- Variable to define cost per box of pencils
variable (P : ℝ)

-- Main theorem to prove
theorem cost_of_each_box_of_pencils :
  cartons_of_pencils * boxes_per_carton_of_pencils * P + 
  cartons_of_markers * cost_per_carton_of_markers = total_spent → 
  P = 2.80 :=
by
  sorry

end cost_of_each_box_of_pencils_l36_36110


namespace total_matches_in_chess_tournament_l36_36214

open Nat

theorem total_matches_in_chess_tournament:
  ∃ (n : ℕ), n = 150 ∧ (3 * (n.choose 2)) = 33750 :=
by
  use 150
  simp
  sorry

end total_matches_in_chess_tournament_l36_36214


namespace quadratic_equations_with_common_root_l36_36230

theorem quadratic_equations_with_common_root :
  ∃ (p1 q1 p2 q2 : ℝ),
    p1 ≠ p2 ∧ q1 ≠ q2 ∧
    ∀ x : ℝ,
      (x^2 + p1 * x + q1 = 0 ∧ x^2 + p2 * x + q2 = 0) →
      (x = 2 ∨ (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ ((x = r1 ∧ x == 2) ∨ (x = r2 ∧ x == 2)))) :=
sorry

end quadratic_equations_with_common_root_l36_36230


namespace smallest_n_for_perfect_square_and_cube_l36_36698

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, (∃ a : ℕ, 4 * n = a^2) ∧ (∃ b : ℕ, 5 * n = b^3) ∧ n = 125 :=
by
  sorry

end smallest_n_for_perfect_square_and_cube_l36_36698


namespace probability_at_least_half_of_six_children_are_girls_l36_36804

-- Define the probability space and binomial distribution
noncomputable def probability_at_least_half_are_girls : ℚ :=
  let n := 6
  let p := 1 / 2
  (∑ k in finset.range (n + 1), if 3 ≤ k then (nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k))) else 0 : ℚ)

theorem probability_at_least_half_of_six_children_are_girls :
  probability_at_least_half_are_girls = 21 / 32 :=
by
  sorry

end probability_at_least_half_of_six_children_are_girls_l36_36804


namespace radius_of_inscribed_circle_is_integer_l36_36355

-- Define variables and conditions
variables (a b c : ℕ)
variables (h1 : c^2 = a^2 + b^2)

-- Define the radius r
noncomputable def r := (a + b - c) / 2

-- Proof statement
theorem radius_of_inscribed_circle_is_integer 
  (h2 : c^2 = a^2 + b^2)
  (h3 : (r : ℤ) = (a + b - c) / 2) : 
  ∃ r : ℤ, r = (a + b - c) / 2 :=
by {
   -- The proof will be provided here
   sorry
}

end radius_of_inscribed_circle_is_integer_l36_36355


namespace crab_ratio_l36_36381

theorem crab_ratio 
  (oysters_day1 : ℕ) 
  (crabs_day1 : ℕ) 
  (total_days : ℕ) 
  (oysters_ratio : ℕ) 
  (oysters_day2 : ℕ) 
  (total_oysters_crabs : ℕ) 
  (crabs_day2 : ℕ) 
  (ratio : ℚ) :
  oysters_day1 = 50 →
  crabs_day1 = 72 →
  oysters_ratio = 2 →
  oysters_day2 = oysters_day1 / oysters_ratio →
  total_oysters_crabs = 195 →
  total_oysters_crabs = oysters_day1 + crabs_day1 + oysters_day2 + crabs_day2 →
  crabs_day2 = total_oysters_crabs - (oysters_day1 + crabs_day1 + oysters_day2) →
  ratio = (crabs_day2 : ℚ) / crabs_day1 →
  ratio = 2 / 3 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end crab_ratio_l36_36381


namespace smallest_n_condition_l36_36703

theorem smallest_n_condition (n : ℕ) : (4 * n) ∣ (n^2) ∧ (5 * n) ∣ (u^3) → n = 100 :=
by
  sorry

end smallest_n_condition_l36_36703


namespace geom_seq_308th_term_l36_36304

noncomputable def geometric_sequence (a r : ℤ) (n : ℕ) : ℤ := a * r ^ (n - 1)

theorem geom_seq_308th_term :
  geometric_sequence 12 (-2) 308 = -2 ^ 307 * 12 := by
  sorry

end geom_seq_308th_term_l36_36304


namespace order_abc_l36_36767

noncomputable def a : ℝ := Real.log 0.8 / Real.log 0.7
noncomputable def b : ℝ := Real.log 0.9 / Real.log 1.1
noncomputable def c : ℝ := Real.exp (0.9 * Real.log 1.1)

theorem order_abc : b < a ∧ a < c := by
  sorry

end order_abc_l36_36767


namespace sum_a_b_eq_5_l36_36147

theorem sum_a_b_eq_5 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * b = a - 2) (h4 : (-2)^2 = b * (2 * b + 2)) : a + b = 5 :=
sorry

end sum_a_b_eq_5_l36_36147


namespace red_window_exchange_l36_36510

-- Defining the total transaction amount for online and offline booths
variables (x y : ℝ)

-- Defining conditions
def offlineMoreThanOnline (y x : ℝ) : Prop := y - 7 * x = 1.8
def averageTransactionDifference (y x : ℝ) : Prop := (y / 71) - (x / 44) = 0.3

-- The proof problem
theorem red_window_exchange (x y : ℝ) :
  offlineMoreThanOnline y x ∧ averageTransactionDifference y x := 
sorry

end red_window_exchange_l36_36510


namespace find_a_value_l36_36188

theorem find_a_value 
  (A : Set ℤ := {-1, 0, 1})
  (a : ℤ) 
  (B : Set ℤ := {a, a^2}) 
  (h_union : A ∪ B = A) : 
  a = -1 :=
sorry

end find_a_value_l36_36188


namespace division_remainder_correct_l36_36692

def polynomial_div_remainder (x : ℝ) : ℝ :=
  3 * x^4 + 14 * x^3 - 50 * x^2 - 72 * x + 55

def divisor (x : ℝ) : ℝ :=
  x^2 + 8 * x - 4

theorem division_remainder_correct :
  ∀ x : ℝ, polynomial_div_remainder x % divisor x = 224 * x - 113 :=
by
  sorry

end division_remainder_correct_l36_36692


namespace hyperbola_sufficiency_l36_36746

open Real

theorem hyperbola_sufficiency (k : ℝ) : 
  (9 - k < 0 ∧ k - 4 > 0) → 
  (∃ x y : ℝ, (x^2) / (9 - k) + (y^2) / (k - 4) = 1) :=
by
  intro hk
  sorry

end hyperbola_sufficiency_l36_36746


namespace difference_of_squares_is_149_l36_36730

-- Definitions of the conditions
def are_consecutive (n m : ℤ) : Prop := m = n + 1
def sum_less_than_150 (n : ℤ) : Prop := (n + (n + 1)) < 150

-- The difference of their squares
def difference_of_squares (n m : ℤ) : ℤ := (m * m) - (n * n)

-- Stating the problem where the answer expected is 149
theorem difference_of_squares_is_149 :
  ∀ n : ℤ, 
  ∀ m : ℤ,
  are_consecutive n m →
  sum_less_than_150 n →
  difference_of_squares n m = 149 :=
by
  sorry

end difference_of_squares_is_149_l36_36730


namespace find_k_l36_36397

-- Definitions
variable (m n k : ℝ)

-- Given conditions
def on_line_1 : Prop := m = 2 * n + 5
def on_line_2 : Prop := (m + 5) = 2 * (n + k) + 5

-- Desired conclusion
theorem find_k (h1 : on_line_1 m n) (h2 : on_line_2 m n k) : k = 2.5 :=
sorry

end find_k_l36_36397


namespace empty_square_exists_in_4x4_l36_36301

theorem empty_square_exists_in_4x4  :
  ∀ (points: Finset (Fin 4 × Fin 4)), points.card = 15 → 
  ∃ (i j : Fin 4), (i, j) ∉ points :=
by
  sorry

end empty_square_exists_in_4x4_l36_36301


namespace arithmetic_mean_of_first_40_consecutive_integers_l36_36420

-- Define the arithmetic sequence with the given conditions
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the given arithmetic sequence
def arithmetic_sum (a₁ d n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Define the arithmetic mean of the first n terms of the given arithmetic sequence
def arithmetic_mean (a₁ d n : ℕ) : ℚ :=
  (arithmetic_sum a₁ d n : ℚ) / n

-- The arithmetic sequence starts at 5, has a common difference of 1, and has 40 terms
theorem arithmetic_mean_of_first_40_consecutive_integers :
  arithmetic_mean 5 1 40 = 24.5 :=
by
  sorry

end arithmetic_mean_of_first_40_consecutive_integers_l36_36420


namespace tan_30_eq_sqrt3_div3_l36_36743

theorem tan_30_eq_sqrt3_div3 (sin_30_cos_30 : ℝ → ℝ → Prop)
  (h1 : sin_30_cos_30 (1 / 2) (Real.sqrt 3 / 2)) :
  ∃ t, t = Real.tan (Real.pi / 6) ∧ t = Real.sqrt 3 / 3 :=
by
  existsi Real.tan (Real.pi / 6)
  sorry

end tan_30_eq_sqrt3_div3_l36_36743


namespace solve_for_k_l36_36106

def sameLine (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem solve_for_k :
  (sameLine (3, 10) (1, k) (-7, 2)) → k = 8.4 :=
by
  sorry

end solve_for_k_l36_36106


namespace select_7_jury_l36_36515

theorem select_7_jury (students : Finset ℕ) (jury : Finset ℕ)
  (likes : ℕ → Finset ℕ) (h_students : students.card = 100)
  (h_jury : jury.card = 25) (h_likes : ∀ s ∈ students, (likes s).card = 10) :
  ∃ (selected_jury : Finset ℕ), selected_jury.card = 7 ∧ ∀ s ∈ students, ∃ j ∈ selected_jury, j ∈ (likes s) :=
sorry

end select_7_jury_l36_36515


namespace grid_contains_unique_integers_l36_36942

theorem grid_contains_unique_integers : 
  ∀ (grid : Fin 101 → Fin 101 → Fin 102),
  (∀ n : Fin 102, (Finset.card (Finset.univ.filter (λ i j, grid i j = n )) = 101)) →
  ∃ i : Fin 101, (Finset.card (Finset.image (λ j, grid i j) Finset.univ) ≥ 11) ∨ 
                 ∃ j : Fin 101, (Finset.card (Finset.image (λ i, grid i j) Finset.univ) ≥ 11) :=
by sorry

end grid_contains_unique_integers_l36_36942


namespace problem_statement_l36_36277

theorem problem_statement (x : ℝ) (h : x = Real.sqrt 3 + 1) : x^2 - 2*x + 1 = 3 :=
sorry

end problem_statement_l36_36277


namespace exists_sequence_l36_36341

theorem exists_sequence (n : ℕ) : ∃ (a : ℕ → ℕ), 
  (∀ i, 1 ≤ i → i < n → (a i > a (i + 1))) ∧
  (∀ i, 1 ≤ i → i < n → (a i ∣ a (i + 1)^2)) ∧
  (∀ i j, 1 ≤ i → 1 ≤ j → i < n → j < n → (i ≠ j → ¬(a i ∣ a j))) :=
sorry

end exists_sequence_l36_36341


namespace intersection_eq_l36_36810

-- Definitions for M and N
def M : Set ℤ := Set.univ
def N : Set ℤ := {x : ℤ | x^2 - x - 2 < 0}

-- The theorem to be proved
theorem intersection_eq : M ∩ N = {0, 1} := 
  sorry

end intersection_eq_l36_36810


namespace find_number_l36_36097

theorem find_number (x : ℤ) 
  (h1 : 3 * (2 * x + 9) = 51) : x = 4 := 
by 
  sorry

end find_number_l36_36097


namespace ratio_prikya_ladonna_l36_36430

def total_cans : Nat := 85
def ladonna_cans : Nat := 25
def yoki_cans : Nat := 10
def prikya_cans : Nat := total_cans - ladonna_cans - yoki_cans

theorem ratio_prikya_ladonna : prikya_cans.toFloat / ladonna_cans.toFloat = 2 / 1 := 
by sorry

end ratio_prikya_ladonna_l36_36430


namespace corset_total_cost_l36_36259

def purple_bead_cost : ℝ := 50 * 20 * 0.12
def blue_bead_cost : ℝ := 40 * 18 * 0.10
def gold_bead_cost : ℝ := 80 * 0.08
def red_bead_cost : ℝ := 30 * 15 * 0.09
def silver_bead_cost : ℝ := 100 * 0.07

def total_cost : ℝ := purple_bead_cost + blue_bead_cost + gold_bead_cost + red_bead_cost + silver_bead_cost

theorem corset_total_cost : total_cost = 245.90 := by
  sorry

end corset_total_cost_l36_36259


namespace quadrilateral_sides_l36_36723

noncomputable def circle_radius : ℝ := 25
noncomputable def diagonal1_length : ℝ := 48
noncomputable def diagonal2_length : ℝ := 40

theorem quadrilateral_sides :
  ∃ (a b c d : ℝ),
    (a = 5 * Real.sqrt 10 ∧ 
    b = 9 * Real.sqrt 10 ∧ 
    c = 13 * Real.sqrt 10 ∧ 
    d = 15 * Real.sqrt 10) ∧ 
    (diagonal1_length = 48 ∧ 
    diagonal2_length = 40 ∧ 
    circle_radius = 25) :=
sorry

end quadrilateral_sides_l36_36723


namespace simplify_fraction_l36_36072

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l36_36072


namespace inscribed_circle_radius_integer_l36_36353

theorem inscribed_circle_radius_integer 
  (a b c : ℕ) (h : a^2 + b^2 = c^2) 
  (h₀ : 2 * (a + b - c) = k) 
  : ∃ (r : ℕ), r = (a + b - c) / 2 := 
begin
  sorry
end

end inscribed_circle_radius_integer_l36_36353


namespace family_reunion_weight_gain_l36_36727

def orlando_gained : ℕ := 5

def jose_gained (orlando: ℕ) : ℕ := 2 * orlando + 2

def fernando_gained (jose: ℕ) : ℕ := jose / 2 - 3

def total_weight_gained : ℕ := 
  let orlando := orlando_gained in
  let jose := jose_gained orlando in
  let fernando := fernando_gained jose in
  orlando + jose + fernando

theorem family_reunion_weight_gain : total_weight_gained = 20 := by
  sorry

end family_reunion_weight_gain_l36_36727


namespace maximum_candies_karlson_l36_36216

theorem maximum_candies_karlson (n : ℕ) (h_n : n = 40) :
  ∃ k, k = 780 :=
by
  sorry

end maximum_candies_karlson_l36_36216


namespace lime_bottom_means_magenta_top_l36_36823

-- Define the colors as an enumeration for clarity
inductive Color
| Purple : Color
| Cyan : Color
| Magenta : Color
| Lime : Color
| Silver : Color
| Black : Color

open Color

-- Define the function representing the question
def opposite_top_face_given_bottom (bottom : Color) : Color :=
  match bottom with
  | Lime => Magenta
  | _ => Lime  -- For simplicity, we're only handling the Lime case as specified

-- State the theorem
theorem lime_bottom_means_magenta_top : 
  opposite_top_face_given_bottom Lime = Magenta :=
by
  -- This theorem states exactly what we need: if Lime is the bottom face, then Magenta is the top face.
  sorry

end lime_bottom_means_magenta_top_l36_36823


namespace rationalize_denominator_l36_36949

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  (4 * Real.sqrt 7 + 3 * Real.sqrt 13) ≠ 0 →
  B < D →
  ∀ (x : ℝ), x = (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) →
    A + B + C + D + E = 22 := 
by
  intros
  -- Provide the actual theorem statement here
  sorry

end rationalize_denominator_l36_36949


namespace license_plate_count_l36_36930

def license_plate_combinations : Nat :=
  26 * Nat.choose 25 2 * Nat.choose 4 2 * 720

theorem license_plate_count :
  license_plate_combinations = 33696000 :=
by
  unfold license_plate_combinations
  sorry

end license_plate_count_l36_36930


namespace intersection_complement_A_B_l36_36776

open Set

variable (x : ℝ)

def U := ℝ
def A := {x | -2 ≤ x ∧ x ≤ 3}
def B := {x | x < -1 ∨ x > 4}

theorem intersection_complement_A_B :
  {x | -2 ≤ x ∧ x ≤ 3} ∩ compl {x | x < -1 ∨ x > 4} = {x | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end intersection_complement_A_B_l36_36776


namespace jimin_and_seokjin_total_l36_36095

def Jimin_coins := (5 * 100) + (1 * 50)
def Seokjin_coins := (2 * 100) + (7 * 10)
def total_coins := Jimin_coins + Seokjin_coins

theorem jimin_and_seokjin_total : total_coins = 820 :=
by
  sorry

end jimin_and_seokjin_total_l36_36095


namespace find_x_l36_36163

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n : ℕ, (2 * n + 1) * x^n

theorem find_x (x : ℝ) (H : series_sum x = 16) : 
  x = (33 - Real.sqrt 129) / 32 :=
by
  sorry

end find_x_l36_36163


namespace contractor_daily_wage_l36_36542

theorem contractor_daily_wage :
  let x := 25 in
  let total_days := 30 in
  let absent_days := 10 in
  let fine_per_absent_day := 7.50 in
  let total_earned := 425 in
  let worked_days := total_days - absent_days in
  (worked_days * x - absent_days * fine_per_absent_day = total_earned) → x = 25 :=
by
  intros
  sorry

end contractor_daily_wage_l36_36542


namespace rectangle_area_l36_36235

theorem rectangle_area (w : ℝ) (h : ℝ) (area : ℝ) 
  (h1 : w = 5)
  (h2 : h = 2 * w) :
  area = h * w := by
  sorry

end rectangle_area_l36_36235


namespace speed_of_man_rowing_upstream_l36_36548

theorem speed_of_man_rowing_upstream (Vm Vdownstream Vupstream : ℝ) (hVm : Vm = 40) (hVdownstream : Vdownstream = 45) : Vupstream = 35 :=
by
  sorry

end speed_of_man_rowing_upstream_l36_36548


namespace pool_houses_count_l36_36036

-- Definitions based on conditions
def total_houses : ℕ := 65
def num_garage : ℕ := 50
def num_both : ℕ := 35
def num_neither : ℕ := 10
def num_pool : ℕ := total_houses - num_garage - num_neither + num_both

theorem pool_houses_count :
  num_pool = 40 := by
  -- Simplified form of the problem expressed in Lean 4 theorem statement.
  sorry

end pool_houses_count_l36_36036


namespace father_ate_oranges_l36_36812

theorem father_ate_oranges (initial_oranges : ℝ) (remaining_oranges : ℝ) (eaten_oranges : ℝ) : 
  initial_oranges = 77.0 → remaining_oranges = 75 → eaten_oranges = initial_oranges - remaining_oranges → eaten_oranges = 2.0 :=
by
  intros h1 h2 h3
  sorry

end father_ate_oranges_l36_36812


namespace fraction_equation_solution_l36_36658

theorem fraction_equation_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) :
  (3 / (x - 3) = 4 / (x - 4)) → x = 0 :=
by
  sorry

end fraction_equation_solution_l36_36658


namespace six_digit_number_unique_solution_l36_36824

theorem six_digit_number_unique_solution
    (a b c d e f : ℕ)
    (hN : (N : ℕ) = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f)
    (hM : (M : ℕ) = 100000 * d + 10000 * e + 1000 * f + 100 * a + 10 * b + c)
    (h_eq : 7 * N = 6 * M) :
    N = 461538 :=
by
  sorry

end six_digit_number_unique_solution_l36_36824


namespace math_exam_problem_l36_36175

open ProbabilityTheory

noncomputable def number_of_students_between_100_and_110 (a : ℝ) (h : a > 0) : ℝ :=
  let n := 1000
  let μ := 100
  let σ := a
  let Z := NormalDist.mk μ σ
  let p := cdf Z 90
  if hp : p = 0.1 then
    n * (cdf Z 110 - cdf Z 100)
  else 0 -- this should never be triggered given the condition

theorem math_exam_problem : 
  ∀ a : ℝ, a > 0 →
  let n := 1000
  let μ := 100
  let Z := NormalDist.mk μ a
  (cdf Z 90 = 0.1) →
  number_of_students_between_100_and_110 a (by assumption) = 400 :=
by
  intros a ha n μ Z h_cdf
  rw [number_of_students_between_100_and_110]
  rw [h_cdf]
  sorry -- proof to be provided

end math_exam_problem_l36_36175


namespace eval_power_l36_36755

-- Given condition
def sixty_four : ℕ := 64

-- Given condition rewritten in Lean
def sixty_four_as_two_powersix : sixty_four = 2^6 := by
  sorry

-- Prove that 64^(5/6) = 32
theorem eval_power : real.exp (5/6 * real.log 64) = 32 := by
  have h1 : 64 = 2^6 := sixty_four_as_two_powersix
  sorry

end eval_power_l36_36755


namespace theodore_total_monthly_earning_l36_36087

def total_earnings (stone_statues: Nat) (wooden_statues: Nat) (cost_stone: Nat) (cost_wood: Nat) (tax_rate: Rat) : Rat :=
  let pre_tax_earnings := stone_statues * cost_stone + wooden_statues * cost_wood
  let tax := tax_rate * pre_tax_earnings
  pre_tax_earnings - tax

theorem theodore_total_monthly_earning : total_earnings 10 20 20 5 0.10 = 270 :=
by
  sorry

end theodore_total_monthly_earning_l36_36087


namespace triangles_xyz_l36_36563

theorem triangles_xyz (A B C D P Q R : Type) 
    (u v w x : ℝ)
    (angle_ADB angle_BDC angle_CDA : ℝ)
    (h1 : angle_ADB = 120) 
    (h2 : angle_BDC = 120) 
    (h3 : angle_CDA = 120) :
    x = u + v + w :=
sorry

end triangles_xyz_l36_36563


namespace minimize_y_l36_36018

noncomputable def y (x a b : ℝ) : ℝ := 2 * (x - a)^2 + 3 * (x - b)^2

theorem minimize_y (a b : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x a b ≤ y x' a b) ∧ x = (2 * a + 3 * b) / 5 :=
sorry

end minimize_y_l36_36018


namespace problem_equivalence_l36_36892

theorem problem_equivalence :
  (1 / Real.sin (Real.pi / 18) - Real.sqrt 3 / Real.sin (4 * Real.pi / 18)) = 4 := 
sorry

end problem_equivalence_l36_36892


namespace number_of_ways_to_select_cells_l36_36367

-- Define the selection problem
theorem number_of_ways_to_select_cells (n : ℕ) :
  let total_ways := factorial n ^ (n^2) * factorial (n^2)
  in total_ways == (n!)^(n^2) * (n^2)! :=
sorry

end number_of_ways_to_select_cells_l36_36367


namespace modulus_of_z_eq_sqrt2_l36_36446

noncomputable def complex_z : ℂ := (1 + 3 * Complex.I) / (2 - Complex.I)

theorem modulus_of_z_eq_sqrt2 : Complex.abs complex_z = Real.sqrt 2 := by
  sorry

end modulus_of_z_eq_sqrt2_l36_36446


namespace one_cow_one_bag_in_forty_days_l36_36796

theorem one_cow_one_bag_in_forty_days
    (total_cows : ℕ)
    (total_bags : ℕ)
    (total_days : ℕ)
    (husk_consumption : total_cows * total_bags = total_cows * total_days) :
  total_days = 40 :=
by sorry

end one_cow_one_bag_in_forty_days_l36_36796


namespace geometric_sequence_a9_l36_36180

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

variable (a : ℕ → ℝ)
variable (q : ℝ)

theorem geometric_sequence_a9
  (h_seq : geometric_sequence a q)
  (h2 : a 1 * a 4 = -32)
  (h3 : a 2 + a 3 = 4)
  (hq : ∃ n : ℤ, q = ↑n) :
  a 8 = -256 := 
sorry

end geometric_sequence_a9_l36_36180


namespace marble_probability_l36_36991

theorem marble_probability :
  let total_marbles := 13
  let blue_marbles := 4
  let red_marbles := 3
  let white_marbles := 6
  let first_blue_prob := (blue_marbles : ℚ) / total_marbles
  let second_red_prob := (red_marbles : ℚ) / (total_marbles - 1)
  let third_white_prob := (white_marbles : ℚ) / (total_marbles - 2)
  first_blue_prob * second_red_prob * third_white_prob = 6 / 143 :=
by
  sorry

end marble_probability_l36_36991


namespace copper_alloy_proof_l36_36089

variable (x p : ℝ)

theorem copper_alloy_proof
  (copper_content1 copper_content2 weight1 weight2 total_weight : ℝ)
  (h1 : weight1 = 3)
  (h2 : copper_content1 = 0.4)
  (h3 : weight2 = 7)
  (h4 : copper_content2 = 0.3)
  (h5 : total_weight = 8)
  (h6 : 1 ≤ x ∧ x ≤ 3)
  (h7 : p = 100 * (copper_content1 * x + copper_content2 * (total_weight - x)) / total_weight) :
  31.25 ≤ p ∧ p ≤ 33.75 := 
  sorry

end copper_alloy_proof_l36_36089


namespace probability_of_product_multiple_of_4_l36_36077

open Finset

noncomputable def count_pairs_with_product_multiple_of_4 : ℕ :=
let nums := Icc 1 20 in
let pairs := (nums.product nums).filter (λ p, p.1 < p.2) in
(pairs.filter (λ p, ∃ m n : ℕ, p.1 = 2 * m ∧ p.2 = 2 * n) ∨
              (∃ k : ℕ, p.1 = 4 * k ∨ p.2 = 4 * k)).card

noncomputable def total_pairs : ℕ :=
((Icc 1 20).product (Icc 1 20)).filter (λ p, p.1 < p.2).card

theorem probability_of_product_multiple_of_4 :
  (count_pairs_with_product_multiple_of_4.to_rat / total_pairs.to_rat) = 9 / 38 :=
sorry

end probability_of_product_multiple_of_4_l36_36077


namespace min_radius_of_circumcircle_l36_36725

theorem min_radius_of_circumcircle {a b : ℝ} (ha : a = 3) (hb : b = 4) : 
∃ R : ℝ, R = 2.5 ∧ (∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ a^2 + b^2 = c^2 ∧ 2 * R = c) :=
by 
  sorry

end min_radius_of_circumcircle_l36_36725


namespace arithmetic_mean_odd_primes_lt_30_l36_36712

theorem arithmetic_mean_odd_primes_lt_30 : 
  (3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29) / 9 = 14 :=
by
  sorry

end arithmetic_mean_odd_primes_lt_30_l36_36712


namespace clover_walk_distance_l36_36260

theorem clover_walk_distance (total_distance days walks_per_day : ℝ) (h1 : total_distance = 90) (h2 : days = 30) (h3 : walks_per_day = 2) :
  (total_distance / days / walks_per_day = 1.5) :=
by
  sorry

end clover_walk_distance_l36_36260


namespace value_of_neg2_neg4_l36_36745

def operation (a b x y : ℤ) : ℤ := a * x - b * y

theorem value_of_neg2_neg4 (a b : ℤ) (h : operation a b 1 2 = 8) : operation a b (-2) (-4) = -16 := by
  sorry

end value_of_neg2_neg4_l36_36745


namespace remainder_div_1234_567_89_1011_mod_12_l36_36389

theorem remainder_div_1234_567_89_1011_mod_12 :
  (1234^567 + 89^1011) % 12 = 9 := 
sorry

end remainder_div_1234_567_89_1011_mod_12_l36_36389


namespace dhoni_savings_percent_l36_36265

variable (E : ℝ) -- Assuming E is Dhoni's last month's earnings

-- Condition 1: Dhoni spent 25% of his earnings on rent
def spent_on_rent (E : ℝ) : ℝ := 0.25 * E

-- Condition 2: Dhoni spent 10% less than what he spent on rent on a new dishwasher
def spent_on_dishwasher (E : ℝ) : ℝ := 0.225 * E

-- Prove the percentage of last month's earnings Dhoni had left over
theorem dhoni_savings_percent (E : ℝ) : 
    52.5 / 100 * E = E - (spent_on_rent E + spent_on_dishwasher E) :=
by
  sorry

end dhoni_savings_percent_l36_36265


namespace find_MN_sum_l36_36786

noncomputable def M : ℝ := sorry -- Placeholder for the actual non-zero solution M
noncomputable def N : ℝ := M ^ 2

theorem find_MN_sum :
  (M^2 = N) ∧ (Real.log N / Real.log M = Real.log M / Real.log N) ∧ (M ≠ N) ∧ (M ≠ 1) ∧ (N ≠ 1) → (M + N = 6) :=
by
  intros h
  exact sorry -- Will be replaced by the actual proof


end find_MN_sum_l36_36786


namespace right_angled_triangle_other_angle_isosceles_triangle_base_angle_l36_36466

theorem right_angled_triangle_other_angle (a : ℝ) (h1 : 0 < a) (h2 : a < 90) (h3 : 40 = a) :
  50 = 90 - a :=
sorry

theorem isosceles_triangle_base_angle (v : ℝ) (h1 : 0 < v) (h2 : v < 180) (h3 : 80 = v) :
  50 = (180 - v) / 2 :=
sorry

end right_angled_triangle_other_angle_isosceles_triangle_base_angle_l36_36466


namespace plane_distance_l36_36662

theorem plane_distance (n : ℕ) : n % 45 = 0 ∧ (n / 10) % 100 = 39 ∧ n <= 5000 → n = 1395 := 
by
  sorry

end plane_distance_l36_36662


namespace remainder_of_expression_l36_36922

theorem remainder_of_expression (n : ℤ) (h : n % 100 = 99) : (n^2 + 2*n + 3 + n^3) % 100 = 1 :=
by
  sorry

end remainder_of_expression_l36_36922


namespace remainder_div_eq_4_l36_36228

theorem remainder_div_eq_4 {x y : ℕ} (h1 : y = 25) (h2 : (x / y : ℝ) = 96.16) : x % y = 4 := 
sorry

end remainder_div_eq_4_l36_36228


namespace gcd_lcm_product_l36_36208

theorem gcd_lcm_product (a b : ℤ) (h1 : Int.gcd a b = 8) (h2 : Int.lcm a b = 24) : a * b = 192 := by
  sorry

end gcd_lcm_product_l36_36208


namespace total_cost_of_breakfast_l36_36589

-- Definitions based on conditions
def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

-- The proof statement
theorem total_cost_of_breakfast : 
  muffin_cost * francis_muffins + 
  fruit_cup_cost * francis_fruit_cups + 
  muffin_cost * kiera_muffins + 
  fruit_cup_cost * kiera_fruit_cup = 17 := 
  by sorry

end total_cost_of_breakfast_l36_36589


namespace train_B_time_to_destination_l36_36219

theorem train_B_time_to_destination (speed_A : ℕ) (time_A : ℕ) (speed_B : ℕ) (dA : ℕ) :
  speed_A = 100 ∧ time_A = 9 ∧ speed_B = 150 ∧ dA = speed_A * time_A →
  dA / speed_B = 6 := 
by
  sorry

end train_B_time_to_destination_l36_36219


namespace min_possible_value_box_l36_36298

theorem min_possible_value_box (a b : ℤ) (h_ab : a * b = 35) : a^2 + b^2 ≥ 74 := sorry

end min_possible_value_box_l36_36298


namespace simplify_fraction_l36_36070

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l36_36070


namespace evaluate_pow_l36_36754

theorem evaluate_pow : 64^(5/6 : ℝ) = 32 := by
  sorry

end evaluate_pow_l36_36754


namespace total_apples_for_bobbing_l36_36813

theorem total_apples_for_bobbing (apples_per_bucket : ℕ) (buckets : ℕ) (total_apples : ℕ) : 
  apples_per_bucket = 9 → buckets = 7 → total_apples = apples_per_bucket * buckets → total_apples = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_apples_for_bobbing_l36_36813


namespace find_dividend_l36_36037

theorem find_dividend (R D Q V : ℤ) (hR : R = 5) (hD1 : D = 3 * Q) (hD2 : D = 3 * R + 3) : V = D * Q + R → V = 113 :=
by 
  sorry

end find_dividend_l36_36037


namespace tan_30_eq_sqrt3_div_3_l36_36740

/-- Statement that proves the value of tang of 30 degrees, given the cosine
    and sine values. -/
theorem tan_30_eq_sqrt3_div_3 
  (cos_30 : Real) (sin_30 : Real) 
  (hcos : cos_30 = Real.sqrt 3 / 2) 
  (hsin : sin_30 = 1 / 2) : 
    Real.tan 30 = Real.sqrt 3 / 3 := 
by 
  sorry

end tan_30_eq_sqrt3_div_3_l36_36740


namespace inequality_holds_l36_36970

theorem inequality_holds (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 12 → x^2 + 25 + |x^3 - 5 * x^2| ≥ a * x) ↔ a ≤ 2.5 := 
by
  sorry

end inequality_holds_l36_36970


namespace largest_beverage_amount_l36_36516

theorem largest_beverage_amount :
  let Milk := (3 / 8 : ℚ)
  let Cider := (7 / 10 : ℚ)
  let OrangeJuice := (11 / 15 : ℚ)
  OrangeJuice > Milk ∧ OrangeJuice > Cider :=
by
  have Milk := (3 / 8 : ℚ)
  have Cider := (7 / 10 : ℚ)
  have OrangeJuice := (11 / 15 : ℚ)
  sorry

end largest_beverage_amount_l36_36516


namespace rectangle_diagonal_length_l36_36828

theorem rectangle_diagonal_length (P : ℝ) (L W D : ℝ) 
  (hP : P = 72) 
  (h_ratio : 3 * W = 2 * L) 
  (h_perimeter : 2 * (L + W) = P) :
  D = Real.sqrt (L * L + W * W) :=
sorry

end rectangle_diagonal_length_l36_36828


namespace not_sum_three_nonzero_squares_l36_36443

-- To state that 8n - 1 is not the sum of three non-zero squares
theorem not_sum_three_nonzero_squares (n : ℕ) :
  ¬ (∃ a b c : ℕ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 8 * n - 1 = a^2 + b^2 + c^2) := by
  sorry

end not_sum_three_nonzero_squares_l36_36443


namespace find_fraction_result_l36_36333

open Complex

theorem find_fraction_result (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
    (h1 : x + y + z = 30)
    (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
    (x^3 + y^3 + z^3) / (x * y * z) = 33 := 
    sorry

end find_fraction_result_l36_36333


namespace isosceles_triangle_perimeter_l36_36932

theorem isosceles_triangle_perimeter {a b : ℕ} (h₁ : a = 4) (h₂ : b = 9) (h₃ : ∀ x y z : ℕ, 
  (x = a ∧ y = a ∧ z = b) ∨ (x = b ∧ y = b ∧ z = a) → 
  (x + y > z ∧ x + z > y ∧ y + z > x)) : 
  (a = 4 ∧ b = 9) → a + a + b = 22 :=
by sorry

end isosceles_triangle_perimeter_l36_36932


namespace min_value_product_expression_l36_36898

theorem min_value_product_expression (x : ℝ) : ∃ m, m = -2746.25 ∧ (∀ y : ℝ, (13 - y) * (8 - y) * (13 + y) * (8 + y) ≥ m) :=
sorry

end min_value_product_expression_l36_36898


namespace num_valid_n_l36_36141

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (Nat.succ n') => Nat.succ n' * factorial n'

def divisible (a b : ℕ) : Prop := b ∣ a

theorem num_valid_n (N : ℕ) :
  N ≤ 30 → 
  ¬ (∃ k, k + 1 ≤ 31 ∧ k + 1 > 1 ∧ (Prime (k + 1)) ∧ ¬ divisible (2 * factorial (k - 1)) (k + 1)) →
  ∃ m : ℕ, m = 20 :=
by
  sorry

end num_valid_n_l36_36141


namespace smallest_n_l36_36695

theorem smallest_n (n : ℕ) : (∃ (m1 m2 : ℕ), 4 * n = m1^2 ∧ 5 * n = m2^3) ↔ n = 500 := 
begin
  sorry
end

end smallest_n_l36_36695


namespace radius_of_inscribed_circle_is_integer_l36_36349

theorem radius_of_inscribed_circle_is_integer 
  (a b c : ℤ) 
  (h_pythagorean : c^2 = a^2 + b^2) 
  : ∃ r : ℤ, r = (a + b - c) / 2 :=
by
  sorry

end radius_of_inscribed_circle_is_integer_l36_36349


namespace english_speaking_students_l36_36459

theorem english_speaking_students (T H B E : ℕ) (hT : T = 40) (hH : H = 30) (hB : B = 10) (h_inclusion_exclusion : T = H + E - B) : E = 20 :=
by
  sorry

end english_speaking_students_l36_36459


namespace sahil_purchase_price_l36_36820

def purchase_price (P : ℝ) : Prop :=
  let repair_cost := 5000
  let transportation_charges := 1000
  let total_cost := repair_cost + transportation_charges
  let selling_price := 27000
  let profit_factor := 1.5
  profit_factor * (P + total_cost) = selling_price

theorem sahil_purchase_price : ∃ P : ℝ, purchase_price P ∧ P = 12000 :=
by
  use 12000
  unfold purchase_price
  simp
  sorry

end sahil_purchase_price_l36_36820


namespace arithmetic_geometric_sequence_fraction_l36_36605

theorem arithmetic_geometric_sequence_fraction 
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : a1 + a2 = 10)
  (h2 : 1 * b3 = 9)
  (h3 : b2 ^ 2 = 9) : 
  b2 / (a1 + a2) = 3 / 10 := 
by 
  sorry

end arithmetic_geometric_sequence_fraction_l36_36605


namespace smallest_circle_equation_l36_36603

theorem smallest_circle_equation :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ (x - 1)^2 + y^2 = 1 ∧ ((x - 1)^2 + y^2 = 1) = (x^2 + y^2 = 1) := 
sorry

end smallest_circle_equation_l36_36603


namespace win_lottery_amount_l36_36856

theorem win_lottery_amount (W : ℝ) (cond1 : W * 0.20 + 5 = 35) : W = 50 := by
  sorry

end win_lottery_amount_l36_36856


namespace minimal_dominoes_needed_l36_36521

-- Variables representing the number of dominoes and tetraminoes
variables (d t : ℕ)

-- Definitions related to the problem
def area_rectangle : ℕ := 2008 * 2010 -- Total area of the rectangle
def area_domino : ℕ := 1 * 2 -- Area of a single domino
def area_tetramino : ℕ := 2 * 3 - 2 -- Area of a single tetramino
def total_area_covered : ℕ := 2 * d + 4 * t -- Total area covered by dominoes and tetraminoes

-- The theorem we want to prove
theorem minimal_dominoes_needed :
  total_area_covered d t = area_rectangle → d = 0 :=
sorry

end minimal_dominoes_needed_l36_36521


namespace range_of_MF_plus_MN_l36_36019

open Real

noncomputable def point_on_parabola (x y : ℝ) : Prop := y^2 = 4 * x

theorem range_of_MF_plus_MN (M : ℝ × ℝ) (N : ℝ × ℝ) (F : ℝ × ℝ) (hM : point_on_parabola M.1 M.2) (hN : N = (2, 2)) (hF : F = (1, 0)) :
  ∃ y : ℝ, y ≥ 3 ∧ ∀ MF MN : ℝ, MF = abs (M.1 - F.1) + abs (M.2 - F.2) ∧ MN = abs (M.1 - N.1) + abs (M.2 - N.2) → MF + MN = y :=
sorry

end range_of_MF_plus_MN_l36_36019


namespace rooks_non_attacking_kings_non_attacking_bishops_non_attacking_knights_non_attacking_queens_non_attacking_l36_36041

-- Define the problem conditions: number of ways to place two same-color rooks that do not attack each other.
def num_ways_rooks : ℕ := 1568
theorem rooks_non_attacking : ∃ (n : ℕ), n = num_ways_rooks := by
  sorry

-- Define the problem conditions: number of ways to place two same-color kings that do not attack each other.
def num_ways_kings : ℕ := 1806
theorem kings_non_attacking : ∃ (n : ℕ), n = num_ways_kings := by
  sorry

-- Define the problem conditions: number of ways to place two same-color bishops that do not attack each other.
def num_ways_bishops : ℕ := 1736
theorem bishops_non_attacking : ∃ (n : ℕ), n = num_ways_bishops := by
  sorry

-- Define the problem conditions: number of ways to place two same-color knights that do not attack each other.
def num_ways_knights : ℕ := 1848
theorem knights_non_attacking : ∃ (n : ℕ), n = num_ways_knights := by
  sorry

-- Define the problem conditions: number of ways to place two same-color queens that do not attack each other.
def num_ways_queens : ℕ := 1288
theorem queens_non_attacking : ∃ (n : ℕ), n = num_ways_queens := by
  sorry

end rooks_non_attacking_kings_non_attacking_bishops_non_attacking_knights_non_attacking_queens_non_attacking_l36_36041


namespace weight_of_mixture_correct_l36_36871

-- Defining the fractions of each component in the mixture
def sand_fraction : ℚ := 2 / 9
def water_fraction : ℚ := 5 / 18
def gravel_fraction : ℚ := 1 / 6
def cement_fraction : ℚ := 7 / 36
def limestone_fraction : ℚ := 1 - sand_fraction - water_fraction - gravel_fraction - cement_fraction

-- Given weight of limestone
def limestone_weight : ℚ := 12

-- Total weight of the mixture that we need to prove
def total_mixture_weight : ℚ := 86.4

-- Proof problem statement
theorem weight_of_mixture_correct : (limestone_fraction * total_mixture_weight = limestone_weight) :=
by
  have h_sand := sand_fraction
  have h_water := water_fraction
  have h_gravel := gravel_fraction
  have h_cement := cement_fraction
  have h_limestone := limestone_fraction
  have h_limestone_weight := limestone_weight
  have h_total_weight := total_mixture_weight
  sorry

end weight_of_mixture_correct_l36_36871


namespace red_bushes_in_middle_probability_l36_36545

theorem red_bushes_in_middle_probability :
  let total_arrangements := (4.factorial / (2.factorial * 2.factorial))
  let favorable_arrangements := 1
  (favorable_arrangements.to_rat / total_arrangements.to_rat) = (1 / 6) := 
by
  sorry

end red_bushes_in_middle_probability_l36_36545


namespace rain_at_least_once_l36_36833

noncomputable def rain_probability (day_prob : ℚ) (days : ℕ) : ℚ :=
  1 - (1 - day_prob)^days

theorem rain_at_least_once :
  ∀ (day_prob : ℚ) (days : ℕ),
    day_prob = 3/4 → days = 4 →
    rain_probability day_prob days = 255/256 :=
by
  intros day_prob days h1 h2
  sorry

end rain_at_least_once_l36_36833


namespace radius_of_inscribed_circle_is_integer_l36_36348

theorem radius_of_inscribed_circle_is_integer 
  (a b c : ℤ) 
  (h_pythagorean : c^2 = a^2 + b^2) 
  : ∃ r : ℤ, r = (a + b - c) / 2 :=
by
  sorry

end radius_of_inscribed_circle_is_integer_l36_36348


namespace intersection_A_B_l36_36311

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l36_36311


namespace find_other_number_l36_36506

theorem find_other_number (a b : ℕ) (h₁ : Nat.lcm a b = 3780) (h₂ : Nat.gcd a b = 18) (h₃ : a = 180) : b = 378 := by
  sorry

end find_other_number_l36_36506


namespace inscribed_circle_radius_integer_l36_36347

theorem inscribed_circle_radius_integer (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ (r : ℤ), r = (a + b - c) / 2 := by
  sorry

end inscribed_circle_radius_integer_l36_36347


namespace halfway_between_3_4_and_5_7_l36_36386

-- Define the two fractions
def frac1 := 3/4
def frac2 := 5/7

-- Define the average function for two fractions
def halfway_fract (a b : ℚ) : ℚ := (a + b) / 2

-- Prove that the halfway fraction between 3/4 and 5/7 is 41/56
theorem halfway_between_3_4_and_5_7 : 
  halfway_fract frac1 frac2 = 41/56 := 
by 
  sorry

end halfway_between_3_4_and_5_7_l36_36386


namespace cube_skew_lines_l36_36544

theorem cube_skew_lines (cube : Prop) (diagonal : Prop) (edges : Prop) :
  ( ∃ n : ℕ, n = 6 ) :=
by
  sorry

end cube_skew_lines_l36_36544


namespace sets_equal_sufficient_condition_l36_36020

variable (a : ℝ)

-- Define sets A and B
def A (x : ℝ) : Prop := 0 < a * x + 1 ∧ a * x + 1 ≤ 5
def B (x : ℝ) : Prop := -1/2 < x ∧ x ≤ 2

-- Statement for Part 1: Sets A and B can be equal if and only if a = 2
theorem sets_equal (h : ∀ x, A a x ↔ B x) : a = 2 :=
sorry

-- Statement for Part 2: Proposition p ⇒ q holds if and only if a > 2 or a < -8
theorem sufficient_condition (h : ∀ x, A a x → B x) (h_neq : ∃ x, B x ∧ ¬A a x) : a > 2 ∨ a < -8 :=
sorry

end sets_equal_sufficient_condition_l36_36020


namespace value_proof_l36_36146

noncomputable def find_value (a b c : ℕ) (h : a + b + c = 240) (h_rat : ∃ (x : ℕ), a = 4 * x ∧ b = 5 * x ∧ c = 7 * x) : Prop :=
  2 * b - a + c = 195

theorem value_proof : ∃ (a b c : ℕ) (h : a + b + c = 240) (h_rat : ∃ (x : ℕ), a = 4 * x ∧ b = 5 * x ∧ c = 7 * x), find_value a b c h h_rat :=
  sorry

end value_proof_l36_36146


namespace set_intersection_l36_36325

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2.5}

theorem set_intersection : A ∩ B = {0, 1, 2} :=
by
  sorry

end set_intersection_l36_36325


namespace power_identity_l36_36660

theorem power_identity (a b : ℕ) (R S : ℕ) (hR : R = 2^a) (hS : S = 5^b) : 
    20^(a * b) = R^(2 * b) * S^a := 
by 
    -- Insert the proof here
    sorry

end power_identity_l36_36660


namespace number_solution_l36_36065

theorem number_solution (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
by
  -- The proof is omitted according to the instructions
  sorry

end number_solution_l36_36065


namespace rain_at_least_once_l36_36835

noncomputable def rain_probability (day_prob : ℚ) (days : ℕ) : ℚ :=
  1 - (1 - day_prob)^days

theorem rain_at_least_once :
  ∀ (day_prob : ℚ) (days : ℕ),
    day_prob = 3/4 → days = 4 →
    rain_probability day_prob days = 255/256 :=
by
  intros day_prob days h1 h2
  sorry

end rain_at_least_once_l36_36835


namespace simplify_fraction_l36_36071

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l36_36071


namespace largest_gcd_l36_36683

theorem largest_gcd (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 1008) : 
  ∃ d : ℕ, d = Int.gcd a b ∧ d = 504 :=
by
  sorry

end largest_gcd_l36_36683


namespace factorize_quadratic_trinomial_l36_36133

theorem factorize_quadratic_trinomial (t : ℝ) : t^2 - 10 * t + 25 = (t - 5)^2 :=
by
  sorry

end factorize_quadratic_trinomial_l36_36133


namespace pokemon_card_cost_l36_36276

theorem pokemon_card_cost 
  (football_cost : ℝ)
  (num_football_packs : ℕ) 
  (baseball_cost : ℝ) 
  (total_spent : ℝ) 
  (h_football : football_cost = 2.73)
  (h_num_football_packs : num_football_packs = 2)
  (h_baseball : baseball_cost = 8.95)
  (h_total : total_spent = 18.42) :
  (total_spent - (num_football_packs * football_cost + baseball_cost) = 4.01) :=
by
  -- Proof goes here
  sorry

end pokemon_card_cost_l36_36276


namespace intersection_A_B_l36_36312

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l36_36312


namespace constant_remainder_polynomial_division_l36_36268

theorem constant_remainder_polynomial_division (b : ℚ) :
  (∃ (r : ℚ), ∀ x : ℚ, r = (8 * x^3 - 9 * x^2 + b * x + 10) % (3 * x^2 - 2 * x + 5)) ↔ b = 118 / 9 :=
by
  sorry

end constant_remainder_polynomial_division_l36_36268


namespace expression_equality_l36_36826

theorem expression_equality : (2 + Real.sqrt 2 + 1 / (2 + Real.sqrt 2) + 1 / (Real.sqrt 2 - 2) = 2) :=
sorry

end expression_equality_l36_36826


namespace Angelina_speed_grocery_to_gym_l36_36530

-- Define parameters for distances and times
def distance_home_to_grocery : ℕ := 720
def distance_grocery_to_gym : ℕ := 480
def time_difference : ℕ := 40

-- Define speeds
variable (v : ℕ) -- speed in meters per second from home to grocery
def speed_home_to_grocery := v
def speed_grocery_to_gym := 2 * v

-- Define times using given speeds and distances
def time_home_to_grocery := distance_home_to_grocery / speed_home_to_grocery
def time_grocery_to_gym := distance_grocery_to_gym / speed_grocery_to_gym

-- Proof statement for the problem
theorem Angelina_speed_grocery_to_gym
  (v_pos : 0 < v)
  (condition : time_home_to_grocery - time_difference = time_grocery_to_gym) :
  speed_grocery_to_gym = 24 := by
  sorry

end Angelina_speed_grocery_to_gym_l36_36530


namespace vector_addition_and_scalar_multiplication_l36_36022

-- Specify the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 1)

-- Define the theorem we want to prove
theorem vector_addition_and_scalar_multiplication :
  a + 2 • b = (-3, 4) :=
sorry

end vector_addition_and_scalar_multiplication_l36_36022


namespace train_speed_in_kmph_l36_36411

variable (L V : ℝ) -- L is the length of the train in meters, and V is the speed of the train in m/s.

-- Conditions given in the problem
def crosses_platform_in_30_seconds : Prop := L + 200 = V * 30
def crosses_man_in_20_seconds : Prop := L = V * 20

-- Length of the platform
def platform_length : ℝ := 200

-- The proof problem: Prove the speed of the train is 72 km/h
theorem train_speed_in_kmph 
  (h1 : crosses_man_in_20_seconds L V) 
  (h2 : crosses_platform_in_30_seconds L V) : 
  V * 3.6 = 72 := 
by 
  sorry

end train_speed_in_kmph_l36_36411


namespace boys_at_reunion_l36_36541

theorem boys_at_reunion (n : ℕ) (H : n * (n - 1) / 2 = 45) : n = 10 :=
by sorry

end boys_at_reunion_l36_36541


namespace find_x_l36_36162

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n : ℕ, (2 * n + 1) * x^n

theorem find_x (x : ℝ) (H : series_sum x = 16) : 
  x = (33 - Real.sqrt 129) / 32 :=
by
  sorry

end find_x_l36_36162


namespace max_value_l36_36281

variable (a b c d : ℝ)

theorem max_value 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) 
  (h5 : b ≠ d) (h6 : c ≠ d)
  (cond1 : a / b + b / c + c / d + d / a = 4)
  (cond2 : a * c = b * d) :
  (a / c + b / d + c / a + d / b) ≤ -12 :=
sorry

end max_value_l36_36281


namespace product_of_real_values_r_l36_36004

theorem product_of_real_values_r {x r : ℝ} (h : x ≠ 0) (heq : (1 / (3 * x)) = ((r - x) / 8)) :
  (∃! x : ℝ, 24 * x^2 - 8 * r * x + 24 = 0) →
  r = 6 ∨ r = -6 ∧ (r * -r) = -36 :=
by
  sorry

end product_of_real_values_r_l36_36004


namespace number_of_small_gardens_l36_36062

def totalSeeds : ℕ := 85
def tomatoSeeds : ℕ := 42
def capsicumSeeds : ℕ := 26
def cucumberSeeds : ℕ := 17

def plantedTomatoSeeds : ℕ := 24
def plantedCucumberSeeds : ℕ := 17

def remainingTomatoSeeds : ℕ := tomatoSeeds - plantedTomatoSeeds
def remainingCapsicumSeeds : ℕ := capsicumSeeds
def remainingCucumberSeeds : ℕ := cucumberSeeds - plantedCucumberSeeds

def seedsInSmallGardenTomato : ℕ := 2
def seedsInSmallGardenCapsicum : ℕ := 1
def seedsInSmallGardenCucumber : ℕ := 1

theorem number_of_small_gardens : (remainingTomatoSeeds / seedsInSmallGardenTomato = 9) :=
by 
  sorry

end number_of_small_gardens_l36_36062


namespace parabola_standard_equation_l36_36085

theorem parabola_standard_equation (x y : ℝ) : 
  (3 * x - 4 * y - 12 = 0) →
  (y = 0 → x = 4 ∨ y = -3 → x = 0) →
  (y^2 = 16 * x ∨ x^2 = -12 * y) :=
by
  intros h_line h_intersect
  sorry

end parabola_standard_equation_l36_36085


namespace number_solution_l36_36066

theorem number_solution (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
by
  -- The proof is omitted according to the instructions
  sorry

end number_solution_l36_36066


namespace count_4_tuples_l36_36907

theorem count_4_tuples (p : ℕ) [hp : Fact (Nat.Prime p)] : 
  Nat.card {abcd : ℕ × ℕ × ℕ × ℕ // (0 < abcd.1 ∧ abcd.1 < p) ∧ 
                                     (0 < abcd.2.1 ∧ abcd.2.1 < p) ∧ 
                                     (0 < abcd.2.2.1 ∧ abcd.2.2.1 < p) ∧ 
                                     (0 < abcd.2.2.2 ∧ abcd.2.2.2 < p) ∧ 
                                     ((abcd.1 * abcd.2.2.2 - abcd.2.1 * abcd.2.2.1) % p = 0)} = (p - 1) * (p - 1) * (p - 1) :=
by
  sorry

end count_4_tuples_l36_36907


namespace train_speed_kmph_l36_36113

noncomputable def train_length : ℝ := 200
noncomputable def crossing_time : ℝ := 3.3330666879982935

theorem train_speed_kmph : (train_length / crossing_time) * 3.6 = 216.00072 := by
  sorry

end train_speed_kmph_l36_36113


namespace f_increasing_on_pos_real_l36_36890

noncomputable def f (x : ℝ) : ℝ := x^2 / (x^2 + 1)

theorem f_increasing_on_pos_real : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 < f x2 :=
by sorry

end f_increasing_on_pos_real_l36_36890


namespace determine_b_from_quadratic_l36_36290

theorem determine_b_from_quadratic (b n : ℝ) (h1 : b > 0) 
  (h2 : ∀ x, x^2 + b*x + 36 = (x + n)^2 + 20) : b = 8 := 
by 
  sorry

end determine_b_from_quadratic_l36_36290


namespace reciprocal_of_2022_l36_36668

noncomputable def reciprocal (x : ℝ) := 1 / x

theorem reciprocal_of_2022 : reciprocal 2022 = 1 / 2022 :=
by
  -- Define reciprocal
  sorry

end reciprocal_of_2022_l36_36668


namespace b_c_value_l36_36923

theorem b_c_value (a b c d : ℕ) 
  (h₁ : a + b = 12) 
  (h₂ : c + d = 3) 
  (h₃ : a + d = 6) : 
  b + c = 9 :=
sorry

end b_c_value_l36_36923


namespace second_store_earns_at_least_72000_more_l36_36993

-- Conditions as definitions in Lean.
def discount_price := 900000 -- 10% discount on 1 million yuan.
def full_price := 1000000 -- Full price for 1 million yuan without discount.

-- Prize calculation for the second department store.
def prize_first := 1000 * 5
def prize_second := 500 * 10
def prize_third := 200 * 20
def prize_fourth := 100 * 40
def prize_fifth := 10 * 1000

def total_prizes := prize_first + prize_second + prize_third + prize_fourth + prize_fifth

def second_store_net_income := full_price - total_prizes -- Net income after subtracting prizes.

-- The proof problem statement.
theorem second_store_earns_at_least_72000_more :
  second_store_net_income - discount_price >= 72000 := sorry

end second_store_earns_at_least_72000_more_l36_36993


namespace gcd_max_value_l36_36672

theorem gcd_max_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) : 
  ∃ d, d = Nat.gcd a b ∧ d = 504 :=
by
  sorry

end gcd_max_value_l36_36672


namespace count_sequences_l36_36124

theorem count_sequences : 
    let S := {s : Fin₅ × Fin₅ × Fin₅ × Fin₅ × Fin₅ // s.1 ≤ s.2 ∧ s.2 ≤ s.3 ∧ s.3 ≤ s.4 ∧ s.4 ≤ s.5 ∧ 
                                               s.1 ≤ 1 ∧ s.2 ≤ 2 ∧ s.3 ≤ 3 ∧ s.4 ≤ 4 ∧ s.5 ≤ 5}
    in S.card = 42 :=
by {
    sorry
}

end count_sequences_l36_36124


namespace minimum_value_of_f_l36_36092

def f (x : ℝ) : ℝ := |x - 4| + |x + 6| + |x - 5|

theorem minimum_value_of_f :
  ∃ x : ℝ, (x = -6 ∧ f (-6) = 1) ∧ ∀ y : ℝ, f y ≥ 1 :=
by
  sorry

end minimum_value_of_f_l36_36092


namespace initial_workers_l36_36957

theorem initial_workers (W : ℕ) (H1 : (8 * W) / 30 = W) (H2 : (6 * (2 * W - 45)) / 45 = 2 * W - 45) : W = 45 :=
sorry

end initial_workers_l36_36957


namespace radius_of_unique_circle_l36_36581

noncomputable def circle_radius (z : ℂ) (h k : ℝ) : ℝ :=
  if z = 2 then 1/4 else 0  -- function that determines the circle

def unique_circle_radius : Prop :=
  let x1 := 2
  let y1 := 0
  
  let x2 := 3 / 2
  let y2 := Real.sqrt 11 / 2

  let h := 7 / 4 -- x-coordinate of the circle's center
  let k := 0    -- y-coordinate of the circle's center

  let r := 1 / 4 -- Radius of the circle
  
  -- equation of the circle passing through (x1, y1) and (x2, y2) should satisfy
  -- the radius of the resulting circle is r

  (x1 - h)^2 + y1^2 = r^2 ∧ (x2 - h)^2 + y2^2 = r^2

theorem radius_of_unique_circle :
  unique_circle_radius :=
sorry

end radius_of_unique_circle_l36_36581


namespace breadth_of_rectangular_plot_l36_36236

theorem breadth_of_rectangular_plot (b : ℝ) (h1 : 3 * b * b = 972) : b = 18 :=
sorry

end breadth_of_rectangular_plot_l36_36236


namespace simplify_sqrt_expression_correct_l36_36299

noncomputable def simplify_sqrt_expression (m : ℝ) (h_triangle : (2 < m + 5) ∧ (m < 2 + 5) ∧ (5 < 2 + m)) : ℝ :=
  (Real.sqrt (9 - 6 * m + m^2)) - (Real.sqrt (m^2 - 14 * m + 49))

theorem simplify_sqrt_expression_correct (m : ℝ) (h_triangle : (2 < m + 5) ∧ (m < 2 + 5) ∧ (5 < 2 + m)) :
  simplify_sqrt_expression m h_triangle = 2 * m - 10 :=
sorry

end simplify_sqrt_expression_correct_l36_36299


namespace part1_part2_l36_36481

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x - 3|

theorem part1 (x : ℝ) (hx : f x ≤ 5) : x ∈ Set.Icc (-1/4 : ℝ) (9/4 : ℝ) := sorry

noncomputable def h (x a : ℝ) : ℝ := Real.log (f x + a)

theorem part2 (ha : ∀ x : ℝ, f x + a > 0) : a ∈ Set.Ioi (-2 : ℝ) := sorry

end part1_part2_l36_36481


namespace perfect_square_trinomial_l36_36171

theorem perfect_square_trinomial (a : ℝ) :
  (∃ m : ℝ, (x^2 + (a-1)*x + 9) = (x + m)^2) → (a = 7 ∨ a = -5) :=
by
  sorry

end perfect_square_trinomial_l36_36171


namespace deepak_current_age_l36_36399

theorem deepak_current_age (x : ℕ) (rahul_age deepak_age : ℕ) :
  (rahul_age = 4 * x) →
  (deepak_age = 3 * x) →
  (rahul_age + 10 = 26) →
  deepak_age = 12 :=
by
  intros h1 h2 h3
  -- You would write the proof here
  sorry

end deepak_current_age_l36_36399


namespace remainder_of_sum_of_integers_mod_15_l36_36710

theorem remainder_of_sum_of_integers_mod_15 (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end remainder_of_sum_of_integers_mod_15_l36_36710


namespace percentage_increase_in_ear_piercing_l36_36045

def cost_of_nose_piercing : ℕ := 20
def noses_pierced : ℕ := 6
def ears_pierced : ℕ := 9
def total_amount_made : ℕ := 390

def cost_of_ear_piercing : ℕ := (total_amount_made - (noses_pierced * cost_of_nose_piercing)) / ears_pierced

def percentage_increase (original new : ℕ) : ℚ := ((new - original : ℚ) / original) * 100

theorem percentage_increase_in_ear_piercing : 
  percentage_increase cost_of_nose_piercing cost_of_ear_piercing = 50 := 
by 
  sorry

end percentage_increase_in_ear_piercing_l36_36045


namespace largest_t_value_l36_36763

theorem largest_t_value : 
  ∃ t : ℝ, 
    (∃ s : ℝ, s > 0 ∧ t = 3 ∧
    ∀ u : ℝ, 
      (u = 3 →
        (15 * u^2 - 40 * u + 18) / (4 * u - 3) + 3 * u = 4 * u + 2 ∧
        u ≤ 3) ∧
      (u ≠ 3 → 
        (15 * u^2 - 40 * u + 18) / (4 * u - 3) + 3 * u = 4 * u + 2 → 
        u ≤ 3)) :=
sorry

end largest_t_value_l36_36763


namespace adam_room_shelves_l36_36998

def action_figures_per_shelf : ℕ := 15
def total_action_figures : ℕ := 120
def total_shelves (total_figures shelves_capacity : ℕ) : ℕ := total_figures / shelves_capacity

theorem adam_room_shelves :
  total_shelves total_action_figures action_figures_per_shelf = 8 :=
by
  sorry

end adam_room_shelves_l36_36998


namespace pythagorean_triangle_inscribed_circle_radius_is_integer_l36_36344

theorem pythagorean_triangle_inscribed_circle_radius_is_integer 
  (a b c : ℕ)
  (h1 : c^2 = a^2 + b^2) 
  (h2 : r = (a + b - c) / 2) :
  ∃ (r : ℕ), r = (a + b - c) / 2 :=
sorry

end pythagorean_triangle_inscribed_circle_radius_is_integer_l36_36344


namespace john_total_distance_l36_36803

def speed : ℕ := 45
def time1 : ℕ := 2
def time2 : ℕ := 3

theorem john_total_distance:
  speed * (time1 + time2) = 225 := by
  sorry

end john_total_distance_l36_36803


namespace white_tiles_in_square_l36_36098

theorem white_tiles_in_square :
  ∀ (n : ℕ), (n * n = 81) → (n ^ 2 - (2 * n - 1)) = 6480 :=
by
  intro n
  intro hn
  sorry

end white_tiles_in_square_l36_36098


namespace partitioning_staircase_l36_36811

def number_of_ways_to_partition_staircase (n : ℕ) : ℕ :=
  2^(n-1)

theorem partitioning_staircase (n : ℕ) : 
  number_of_ways_to_partition_staircase n = 2^(n-1) :=
by 
  sorry

end partitioning_staircase_l36_36811


namespace range_of_k_l36_36770

-- Definitions for the condition
def inequality_holds (k : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0

-- Theorem statement
theorem range_of_k (k : ℝ) : inequality_holds k → k ≥ 1 :=
sorry

end range_of_k_l36_36770


namespace gcd_largest_divisor_l36_36675

theorem gcd_largest_divisor (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 1008) : 
  ∃ d, nat.gcd a b = d ∧ d = 504 :=
begin
  sorry
end

end gcd_largest_divisor_l36_36675


namespace sufficient_not_necessary_a_equals_2_l36_36864

theorem sufficient_not_necessary_a_equals_2 {a : ℝ} :
  (∃ a : ℝ, (a = 2 ∧ 15 * a^2 = 60) → (15 * a^2 = 60) ∧ (15 * a^2 = 60 → a = 2)) → 
  (¬∀ a : ℝ, (15 * a^2 = 60) → a = 2) → 
  (a = 2 → 15 * a^2 = 60) ∧ ¬(15 * a^2 = 60 → a = 2) :=
by
  sorry

end sufficient_not_necessary_a_equals_2_l36_36864


namespace mean_of_set_is_16_6_l36_36664

theorem mean_of_set_is_16_6 (m : ℝ) (h : m + 7 = 16) :
  (9 + 11 + 16 + 20 + 27) / 5 = 16.6 :=
by
  -- Proof steps would go here, but we use sorry to skip the proof.
  sorry

end mean_of_set_is_16_6_l36_36664


namespace percentage_of_part_whole_l36_36539

theorem percentage_of_part_whole (part whole : ℝ) (h_part : part = 75) (h_whole : whole = 125) : 
  (part / whole) * 100 = 60 :=
by
  rw [h_part, h_whole]
  -- Simplification steps would follow, but we substitute in the placeholders
  sorry

end percentage_of_part_whole_l36_36539


namespace tan_30_eq_sqrt3_div_3_l36_36732

theorem tan_30_eq_sqrt3_div_3 :
  let opposite := 1
  let adjacent := sqrt (3 : ℝ) 
  tan (real.pi / 6) = opposite / adjacent := by 
    sorry

end tan_30_eq_sqrt3_div_3_l36_36732


namespace maximum_value_condition_l36_36644

open Real

theorem maximum_value_condition {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h1 : x + y = 16) (h2 : x = 2 * y) :
  (1 / x + 1 / y) = 9 / 32 :=
by
  sorry

end maximum_value_condition_l36_36644


namespace used_more_brown_sugar_l36_36393

-- Define the amounts of sugar used
def brown_sugar : ℝ := 0.62
def white_sugar : ℝ := 0.25

-- Define the statement to prove
theorem used_more_brown_sugar : brown_sugar - white_sugar = 0.37 :=
by
  sorry

end used_more_brown_sugar_l36_36393


namespace salami_pizza_fraction_l36_36620

theorem salami_pizza_fraction 
    (d_pizza : ℝ) 
    (n_salami_diameter : ℕ) 
    (n_salami_total : ℕ) 
    (h1 : d_pizza = 16)
    (h2 : n_salami_diameter = 8) 
    (h3 : n_salami_total = 32) 
    : 
    (32 * (Real.pi * (d_pizza / (2 * n_salami_diameter / 2)) ^ 2)) / (Real.pi * (d_pizza / 2) ^ 2) = 1 / 2 := 
by 
  sorry

end salami_pizza_fraction_l36_36620


namespace jason_grass_cutting_time_l36_36633

def total_minutes (hours : ℕ) : ℕ := hours * 60
def minutes_per_yard : ℕ := 30
def total_yards_per_weekend : ℕ := 8 * 2
def total_minutes_per_weekend : ℕ := minutes_per_yard * total_yards_per_weekend
def convert_minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

theorem jason_grass_cutting_time : 
  convert_minutes_to_hours total_minutes_per_weekend = 8 := by
  sorry

end jason_grass_cutting_time_l36_36633


namespace distinct_real_numbers_eq_l36_36291

theorem distinct_real_numbers_eq (x : ℝ) :
  (x^2 - 7)^2 + 2 * x^2 = 33 → 
  (∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                    {a, b, c, d} = {x | (x^2 - 7)^2 + 2 * x^2 = 33}) :=
sorry

end distinct_real_numbers_eq_l36_36291


namespace time_per_flash_l36_36996

def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60
def light_flashes_in_three_fourths_hour : ℕ := 180

-- Converting ¾ of an hour to minutes and then to seconds
def seconds_in_three_fourths_hour : ℕ := (3 * minutes_per_hour / 4) * seconds_per_minute

-- Proving that the time taken for one flash is 15 seconds
theorem time_per_flash : (seconds_in_three_fourths_hour / light_flashes_in_three_fourths_hour) = 15 :=
by
  sorry

end time_per_flash_l36_36996


namespace isosceles_triangle_angle_l36_36240

-- Definition of required angles and the given geometric context
variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]
variables (angleBAC : ℝ) (angleBCA : ℝ)

-- Given: shared vertex A, with angle BAC of pentagon
axiom angleBAC_def : angleBAC = 108

-- To Prove: determining the measure of angle BCA in the isosceles triangle
theorem isosceles_triangle_angle (h : 180 > 2 * angleBAC) : angleBCA = (180 - angleBAC) / 2 :=
  sorry

end isosceles_triangle_angle_l36_36240


namespace fill_boxes_l36_36574

theorem fill_boxes (a b c d e f g : ℤ) 
  (h1 : a + (-1) + 2 = 4)
  (h2 : 2 + 1 + b = 3)
  (h3 : c + (-4) + (-3) = -2)
  (h4 : b - 5 - 4 = -9)
  (h5 : f = d - 3)
  (h6 : g = d + 3)
  (h7 : -8 = 4 + 3 - 9 - 2 + (d - 3) + (d + 3)) : 
  a = 3 ∧ b = 0 ∧ c = 5 ∧ d = -2 ∧ e = -9 ∧ f = -5 ∧ g = 1 :=
by {
  sorry
}

end fill_boxes_l36_36574


namespace curve_C2_eq_l36_36647

def curve_C (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def reflect_y_axis (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-x)
def reflect_x_axis (f : ℝ → ℝ) (x : ℝ) : ℝ := - (f x)

theorem curve_C2_eq (a b c : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, reflect_x_axis (reflect_y_axis (curve_C a b c)) x = -a * x^2 + b * x - c := by
  sorry

end curve_C2_eq_l36_36647


namespace probability_red_side_l36_36870

theorem probability_red_side (total_cards : ℕ)
  (cards_black_black : ℕ) (cards_black_red : ℕ) (cards_red_red : ℕ)
  (h_total : total_cards = 9)
  (h_black_black : cards_black_black = 4)
  (h_black_red : cards_black_red = 2)
  (h_red_red : cards_red_red = 3) :
  let total_sides := (cards_black_black * 2) + (cards_black_red * 2) + (cards_red_red * 2)
  let red_sides := (cards_black_red * 1) + (cards_red_red * 2)
  (red_sides > 0) →
  ((cards_red_red * 2) / red_sides : ℚ) = 3 / 4 := 
by
  intros
  sorry

end probability_red_side_l36_36870


namespace cost_of_tax_free_items_l36_36639

-- Definitions based on the conditions.
def total_spending : ℝ := 20
def sales_tax_percentage : ℝ := 0.30
def tax_rate : ℝ := 0.06

-- Derived calculations for intermediate variables for clarity
def taxable_items_cost : ℝ := total_spending * (1 - sales_tax_percentage)
def sales_tax_paid : ℝ := taxable_items_cost * tax_rate
def tax_free_items_cost : ℝ := total_spending - taxable_items_cost

-- Lean 4 statement for the problem
theorem cost_of_tax_free_items :
  tax_free_items_cost = 6 := by
    -- The proof would go here, but we are skipping it.
    sorry

end cost_of_tax_free_items_l36_36639


namespace find_foci_l36_36561

def hyperbolaFoci : Prop :=
  let eq := ∀ x y, 2 * x^2 - 3 * y^2 + 8 * x - 12 * y - 23 = 0
  ∃ foci : ℝ × ℝ, foci = (-2 - Real.sqrt (5 / 6), -2) ∨ foci = (-2 + Real.sqrt (5 / 6), -2)

theorem find_foci : hyperbolaFoci :=
by
  sorry

end find_foci_l36_36561


namespace find_ratio_l36_36306

variables {EF GH EH EG EQ ER ES Q R S : ℝ}
variables (x : ℝ)
variables (E F G H : ℝ)

-- Conditions
def is_parallelogram : Prop := 
  -- Placeholder for parallelogram properties, not relevant for this example
  true

def point_on_segment (Q R : ℝ) (segment_length: ℝ) (ratio: ℝ): Prop := Q = segment_length * ratio ∧ R = segment_length * ratio

def intersect (EG QR : ℝ) (S : ℝ): Prop := 
  -- Placeholder for segment intersection properties, not relevant for this example
  true

-- Question
theorem find_ratio 
  (H_parallelogram: is_parallelogram)
  (H_pointQ: point_on_segment EQ ER EF (1/8))
  (H_pointR: point_on_segment ER ES EH (1/9))
  (H_intersection: intersect EG QR ES):
  (ES / EG) = (1/9) := 
by
  sorry

end find_ratio_l36_36306


namespace total_cost_of_breakfast_l36_36588

-- Definitions based on conditions
def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

-- The proof statement
theorem total_cost_of_breakfast : 
  muffin_cost * francis_muffins + 
  fruit_cup_cost * francis_fruit_cups + 
  muffin_cost * kiera_muffins + 
  fruit_cup_cost * kiera_fruit_cup = 17 := 
  by sorry

end total_cost_of_breakfast_l36_36588


namespace largest_gcd_l36_36679

theorem largest_gcd (a b : ℕ) (h : a + b = 1008) : ∃ d, d = gcd a b ∧ (∀ d', d' = gcd a b → d' ≤ d) ∧ d = 504 :=
by
  sorry

end largest_gcd_l36_36679


namespace moles_of_naoh_needed_l36_36580

-- Define the chemical reaction
def balanced_eqn (nh4no3 naoh nano3 nh4oh : ℕ) : Prop :=
  nh4no3 = naoh ∧ nh4no3 = nano3

-- Theorem stating the moles of NaOH required to form 2 moles of NaNO3 from 2 moles of NH4NO3
theorem moles_of_naoh_needed (nh4no3 naoh nano3 nh4oh : ℕ) (h_balanced_eqn : balanced_eqn nh4no3 naoh nano3 nh4oh) 
  (h_nano3: nano3 = 2) (h_nh4no3: nh4no3 = 2) : naoh = 2 :=
by
  unfold balanced_eqn at h_balanced_eqn
  sorry

end moles_of_naoh_needed_l36_36580


namespace shade_half_grid_additional_squares_l36_36179

/-- A 4x5 grid consists of 20 squares, of which 3 are already shaded. 
Prove that the number of additional 1x1 squares needed to shade half the grid is 7. -/
theorem shade_half_grid_additional_squares (total_squares shaded_squares remaining_squares: ℕ) 
  (h1 : total_squares = 4 * 5)
  (h2 : shaded_squares = 3)
  (h3 : remaining_squares = total_squares / 2 - shaded_squares) :
  remaining_squares = 7 :=
by
  -- Proof not required.
  sorry

end shade_half_grid_additional_squares_l36_36179


namespace intersection_of_sets_l36_36326

open Set

theorem intersection_of_sets : 
  let A := {-2, -1, 0, 1, 2}
  let B := {x : ℚ | 0 ≤ x ∧ x < 5/2}
  A ∩ B = {0, 1, 2} :=
by
  -- Lean's definition of finite sets uses List, need to convert List to Set for intersection
  let A : Set ℚ := {-2, -1, 0, 1, 2}
  let B : Set ℚ := {x | 0 ≤ x ∧ x < 5/2}
  let answer := {0, 1, 2}
  show A ∩ B = answer
  sorry

end intersection_of_sets_l36_36326


namespace find_last_three_digits_of_9_pow_107_l36_36003

theorem find_last_three_digits_of_9_pow_107 : (9 ^ 107) % 1000 = 969 := 
by 
  sorry

end find_last_three_digits_of_9_pow_107_l36_36003


namespace sum_of_squares_l36_36198

def satisfies_conditions (x y z : ℕ) : Prop :=
  x + y + z = 24 ∧
  Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10

theorem sum_of_squares (x y z : ℕ) (h : satisfies_conditions x y z) :
  ∀ (x y z : ℕ), x + y + z = 24 ∧ Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10 →
  x^2 + y^2 + z^2 = 216 :=
sorry

end sum_of_squares_l36_36198


namespace least_possible_value_minimum_at_zero_zero_l36_36223

theorem least_possible_value (x y : ℝ) : (xy - 1)^2 + (x + y)^2 ≥ 1 :=
begin
  sorry
end

theorem minimum_at_zero_zero : (xy - 1)^2 + (x + y)^2 = 1 ↔ x = 0 ∧ y = 0 :=
begin
  sorry
end

end least_possible_value_minimum_at_zero_zero_l36_36223


namespace measure_angle_BCQ_l36_36821

/-- Given:
  - Segment AB has a length of 12 units.
  - Segment AC is 9 units long.
  - Segment AC : CB = 3 : 1.
  - A semi-circle is constructed with diameter AB.
  - Another smaller semi-circle is constructed with diameter CB.
  - A line segment CQ divides the combined area of the two semi-circles into two equal areas.

  Prove: The degree measure of angle BCQ is 11.25°.
-/ 
theorem measure_angle_BCQ (AB AC CB : ℝ) (hAB : AB = 12) (hAC : AC = 9) (hRatio : AC / CB = 3) :
  ∃ θ : ℝ, θ = 11.25 :=
by
  sorry

end measure_angle_BCQ_l36_36821


namespace find_integers_l36_36518

theorem find_integers 
  (A k : ℕ) 
  (h_sum : A + A * k + A * k^2 = 93) 
  (h_product : A * (A * k) * (A * k^2) = 3375) : 
  (A, A * k, A * k^2) = (3, 15, 75) := 
by 
  sorry

end find_integers_l36_36518


namespace find_a_l36_36015

theorem find_a (a x : ℝ) (h1 : 3 * x + 2 * a = 2) (h2 : x = 1) : a = -1/2 :=
by
  sorry

end find_a_l36_36015


namespace radius_of_inscribed_circle_is_integer_l36_36354

-- Define variables and conditions
variables (a b c : ℕ)
variables (h1 : c^2 = a^2 + b^2)

-- Define the radius r
noncomputable def r := (a + b - c) / 2

-- Proof statement
theorem radius_of_inscribed_circle_is_integer 
  (h2 : c^2 = a^2 + b^2)
  (h3 : (r : ℤ) = (a + b - c) / 2) : 
  ∃ r : ℤ, r = (a + b - c) / 2 :=
by {
   -- The proof will be provided here
   sorry
}

end radius_of_inscribed_circle_is_integer_l36_36354


namespace correct_removal_of_parentheses_C_incorrect_removal_of_parentheses_A_incorrect_removal_of_parentheses_B_incorrect_removal_of_parentheses_D_l36_36855

theorem correct_removal_of_parentheses_C (a : ℝ) :
    -(2 * a - 1) = -2 * a + 1 :=
by sorry

theorem incorrect_removal_of_parentheses_A (a : ℝ) :
    -(7 * a - 5) ≠ -7 * a - 5 :=
by sorry

theorem incorrect_removal_of_parentheses_B (a : ℝ) :
    -(-1 / 2 * a + 2) ≠ -1 / 2 * a - 2 :=
by sorry

theorem incorrect_removal_of_parentheses_D (a : ℝ) :
    -(-3 * a + 2) ≠ 3 * a + 2 :=
by sorry

end correct_removal_of_parentheses_C_incorrect_removal_of_parentheses_A_incorrect_removal_of_parentheses_B_incorrect_removal_of_parentheses_D_l36_36855


namespace forgotten_angles_sum_l36_36653

theorem forgotten_angles_sum (n : ℕ) (h : (n-2) * 180 = 3240 + x) : x = 180 :=
by {
  sorry
}

end forgotten_angles_sum_l36_36653


namespace range_of_k_l36_36771

-- Definitions for the condition
def inequality_holds (k : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0

-- Theorem statement
theorem range_of_k (k : ℝ) : inequality_holds k → k ≥ 1 :=
sorry

end range_of_k_l36_36771


namespace gcd_of_three_numbers_l36_36140

theorem gcd_of_three_numbers (a b c : ℕ) (h1 : a = 15378) (h2 : b = 21333) (h3 : c = 48906) :
  Nat.gcd (Nat.gcd a b) c = 3 :=
by
  rw [h1, h2, h3]
  sorry

end gcd_of_three_numbers_l36_36140


namespace MariaTotalPaid_l36_36392

-- Define a structure to hold the conditions
structure DiscountProblem where
  discount_rate : ℝ
  discount_amount : ℝ

-- Define the given discount problem specific to Maria
def MariaDiscountProblem : DiscountProblem :=
  { discount_rate := 0.25, discount_amount := 40 }

-- Define our goal: proving the total amount paid by Maria
theorem MariaTotalPaid (p : DiscountProblem) (h₀ : p = MariaDiscountProblem) :
  let original_price := p.discount_amount / p.discount_rate
  let total_paid := original_price - p.discount_amount
  total_paid = 120 :=
by
  sorry

end MariaTotalPaid_l36_36392


namespace log_sqrt_defined_range_l36_36144

theorem log_sqrt_defined_range (x: ℝ) : 
  (∃ (y: ℝ), y = (log (5-x) / sqrt (x+2))) ↔ (-2 ≤ x ∧ x < 5) :=
by
  sorry

end log_sqrt_defined_range_l36_36144


namespace sum_of_coordinates_eq_69_l36_36622

theorem sum_of_coordinates_eq_69 {f k : ℝ → ℝ} (h₁ : f 4 = 8) (h₂ : ∀ x, k x = (f x)^2 + 1) : 4 + k 4 = 69 :=
by
  sorry

end sum_of_coordinates_eq_69_l36_36622


namespace stock_increase_l36_36117

theorem stock_increase (x : ℝ) (h₁ : x > 0) :
  (1.25 * (0.85 * x) - x) / x * 100 = 6.25 :=
by 
  -- {proof steps would go here}
  sorry

end stock_increase_l36_36117


namespace total_books_after_donations_l36_36418

variable (Boris_books : Nat := 24)
variable (Cameron_books : Nat := 30)

theorem total_books_after_donations :
  (Boris_books - Boris_books / 4) + (Cameron_books - Cameron_books / 3) = 38 := by
  sorry

end total_books_after_donations_l36_36418


namespace largest_gcd_l36_36682

theorem largest_gcd (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 1008) : 
  ∃ d : ℕ, d = Int.gcd a b ∧ d = 504 :=
by
  sorry

end largest_gcd_l36_36682


namespace smallest_n_for_perfect_square_and_cube_l36_36697

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, (∃ a : ℕ, 4 * n = a^2) ∧ (∃ b : ℕ, 5 * n = b^3) ∧ n = 125 :=
by
  sorry

end smallest_n_for_perfect_square_and_cube_l36_36697


namespace tangent_of_inclination_of_OP_l36_36915

noncomputable def point_P_x (φ : ℝ) : ℝ := 3 * Real.cos φ
noncomputable def point_P_y (φ : ℝ) : ℝ := 2 * Real.sin φ

theorem tangent_of_inclination_of_OP (φ : ℝ) (h: φ = Real.pi / 6) :
  (point_P_y φ / point_P_x φ) = 2 * Real.sqrt 3 / 9 :=
by
  have h1 : point_P_x φ = 3 * (Real.sqrt 3 / 2) := by sorry
  have h2 : point_P_y φ = 1 := by sorry
  sorry

end tangent_of_inclination_of_OP_l36_36915


namespace cube_identity_simplification_l36_36334

theorem cube_identity_simplification (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 0) :
  (x^3 + y^3 + z^3 + 3 * x * y * z) / (x * y * z) = 6 :=
by
  sorry

end cube_identity_simplification_l36_36334


namespace intersection_of_sets_l36_36328

open Set

theorem intersection_of_sets : 
  let A := {-2, -1, 0, 1, 2}
  let B := {x : ℚ | 0 ≤ x ∧ x < 5/2}
  A ∩ B = {0, 1, 2} :=
by
  -- Lean's definition of finite sets uses List, need to convert List to Set for intersection
  let A : Set ℚ := {-2, -1, 0, 1, 2}
  let B : Set ℚ := {x | 0 ≤ x ∧ x < 5/2}
  let answer := {0, 1, 2}
  show A ∩ B = answer
  sorry

end intersection_of_sets_l36_36328


namespace problems_left_to_grade_l36_36114

-- Definitions based on provided conditions
def problems_per_worksheet : ℕ := 4
def total_worksheets : ℕ := 16
def graded_worksheets : ℕ := 8

-- The statement for the required proof with the correct answer included
theorem problems_left_to_grade : 4 * (16 - 8) = 32 := by
  sorry

end problems_left_to_grade_l36_36114


namespace arithmetic_sequence_sum_l36_36027

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + m) = a n + m * (a 1 - a 0)

theorem arithmetic_sequence_sum
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 5 + a 6 + a 7 = 15) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by
  sorry

end arithmetic_sequence_sum_l36_36027


namespace race_length_l36_36624

theorem race_length (A_time : ℕ) (diff_distance diff_time : ℕ) (A_time_eq : A_time = 380)
  (diff_distance_eq : diff_distance = 50) (diff_time_eq : diff_time = 20) :
  let B_speed := diff_distance / diff_time
  let B_time := A_time + diff_time
  let race_length := B_speed * B_time
  race_length = 1000 := 
by
  sorry

end race_length_l36_36624


namespace product_of_real_values_r_l36_36005

noncomputable def product_of_real_r : ℚ :=
if (∃ r : ℚ, ∀ x : ℚ, x ≠ 0 → (1 / (3 * x) = (r - x) / 8) = true) then 
  (-32 / 3 : ℚ) 
else 
  0

theorem product_of_real_values_r : product_of_real_r = -32 / 3 :=
by sorry

end product_of_real_values_r_l36_36005


namespace tan_30_l36_36738

theorem tan_30 : Real.tan (Real.pi / 6) = Real.sqrt 3 / 3 := 
by 
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  have h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2 := by sorry
  calc
    Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6) : Real.tan_eq_sin_div_cos _
    ... = (1 / 2) / (Real.sqrt 3 / 2) : by rw [h1, h2]
    ... = (1 / 2) * (2 / Real.sqrt 3) : by rw Div.div_eq_mul_inv
    ... = 1 / Real.sqrt 3 : by norm_num
    ... = Real.sqrt 3 / 3 : by rw [Div.inv_eq_inv, Mul.comm, Mul.assoc, Div.mul_inv_cancel (Real.sqrt_ne_zero _), one_div Real.sqrt 3, inv_mul_eq_div]

-- Additional necessary function apologies for the unproven theorems.
noncomputable def _root_.Real.sqrt (x:ℝ) : ℝ := sorry

noncomputable def _root_.Real.tan (x : ℝ) : ℝ :=
  (Real.sin x) / (Real.cos x)

#eval tan_30 -- check result

end tan_30_l36_36738


namespace mrs_bil_earnings_percentage_in_may_l36_36458

theorem mrs_bil_earnings_percentage_in_may
  (M F : ℝ)
  (h₁ : 1.10 * M / (1.10 * M + F) = 0.7196) :
  M / (M + F) = 0.70 :=
sorry

end mrs_bil_earnings_percentage_in_may_l36_36458


namespace num_false_statements_is_three_l36_36992

-- Definitions of the statements on the card
def s1 : Prop := ∀ (false_statements : ℕ), false_statements = 1
def s2 : Prop := ∀ (false_statements_card1 false_statements_card2 : ℕ), false_statements_card1 + false_statements_card2 = 2
def s3 : Prop := ∀ (false_statements : ℕ), false_statements = 3
def s4 : Prop := ∀ (false_statements_card1 false_statements_card2 : ℕ), false_statements_card1 = false_statements_card2

-- Main proof problem: The number of false statements on this card is 3
theorem num_false_statements_is_three 
  (h_s1 : ¬ s1)
  (h_s2 : ¬ s2)
  (h_s3 : s3)
  (h_s4 : ¬ s4) :
  ∃ (n : ℕ), n = 3 :=
by
  sorry

end num_false_statements_is_three_l36_36992


namespace intersection_of_sets_l36_36327

open Set

theorem intersection_of_sets : 
  let A := {-2, -1, 0, 1, 2}
  let B := {x : ℚ | 0 ≤ x ∧ x < 5/2}
  A ∩ B = {0, 1, 2} :=
by
  -- Lean's definition of finite sets uses List, need to convert List to Set for intersection
  let A : Set ℚ := {-2, -1, 0, 1, 2}
  let B : Set ℚ := {x | 0 ≤ x ∧ x < 5/2}
  let answer := {0, 1, 2}
  show A ∩ B = answer
  sorry

end intersection_of_sets_l36_36327


namespace cone_to_cylinder_water_height_l36_36119

theorem cone_to_cylinder_water_height :
  let r_cone := 15 -- radius of the cone
  let h_cone := 24 -- height of the cone
  let r_cylinder := 18 -- radius of the cylinder
  let V_cone := (1 / 3: ℝ) * Real.pi * r_cone^2 * h_cone -- volume of the cone
  let h_cylinder := V_cone / (Real.pi * r_cylinder^2) -- height of the water in the cylinder
  h_cylinder = 8.33 := by
  sorry

end cone_to_cylinder_water_height_l36_36119


namespace rain_at_least_once_prob_l36_36836

theorem rain_at_least_once_prob (p : ℚ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 :=
by {
  -- Implementation of Lean code is not required as per instructions.
  sorry
}

end rain_at_least_once_prob_l36_36836


namespace box_interior_surface_area_l36_36075

-- Defining the conditions
def original_length := 30
def original_width := 20
def corner_length := 5
def num_corners := 4

-- Defining the area calculations based on given dimensions and removed corners
def original_area := original_length * original_width
def area_one_corner := corner_length * corner_length
def total_area_removed := num_corners * area_one_corner
def remaining_area := original_area - total_area_removed

-- Statement to prove
theorem box_interior_surface_area :
  remaining_area = 500 :=
by 
  sorry

end box_interior_surface_area_l36_36075


namespace find_greatest_integer_l36_36913

theorem find_greatest_integer :
  (M n : ℕ),
  ((∑ k in {3, 4, 5, 6, 7, 8, 9, 10}, 1 / (k! * (21 - k)!)) = (M / (2! * 19!))) →
  (⌊M / 100⌋ = 1048) :=
by
  sorry

end find_greatest_integer_l36_36913


namespace cone_lateral_area_l36_36200

noncomputable def lateral_area_of_cone (θ : ℝ) (r_base : ℝ) : ℝ :=
  if θ = 120 ∧ r_base = 2 then 
    12 * Real.pi 
  else 
    0 -- default case for the sake of definition, not used in our proof

theorem cone_lateral_area :
  lateral_area_of_cone 120 2 = 12 * Real.pi :=
by
  -- This is where the proof would go
  sorry

end cone_lateral_area_l36_36200


namespace tangent_lines_to_circle_l36_36512

theorem tangent_lines_to_circle 
  (x y : ℝ) 
  (circle : (x - 2) ^ 2 + (y + 1) ^ 2 = 1) 
  (point : x = 3 ∧ y = 3) : 
  (x = 3 ∨ 15 * x - 8 * y - 21 = 0) :=
sorry

end tangent_lines_to_circle_l36_36512


namespace Meadow_sells_each_diaper_for_5_l36_36060

-- Define the conditions as constants
def boxes_per_week := 30
def packs_per_box := 40
def diapers_per_pack := 160
def total_revenue := 960000

-- Calculate total packs and total diapers
def total_packs := boxes_per_week * packs_per_box
def total_diapers := total_packs * diapers_per_pack

-- The target price per diaper
def price_per_diaper := total_revenue / total_diapers

-- Statement of the proof theorem
theorem Meadow_sells_each_diaper_for_5 : price_per_diaper = 5 := by
  sorry

end Meadow_sells_each_diaper_for_5_l36_36060


namespace minimal_surface_area_l36_36256

-- Definitions based on the conditions in the problem.
def unit_cube (a b c : ℕ) : Prop := a * b * c = 25
def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + a * c + b * c)

-- The proof problem statement.
theorem minimal_surface_area : ∃ (a b c : ℕ), unit_cube a b c ∧ surface_area a b c = 54 := 
sorry

end minimal_surface_area_l36_36256


namespace absolute_value_of_sum_C_D_base5_l36_36761

theorem absolute_value_of_sum_C_D_base5 (C D : ℕ) (h1 : C - 3 = 1) (h2 : C = 4) (h3 : 2 - 2 = 3 . -. (7 - 2) = 3) (h4 : D - C = 2) (h5 : D = 0) : |C + D| = 4 := by
sorry

end absolute_value_of_sum_C_D_base5_l36_36761


namespace relationship_y1_y2_y3_l36_36785

variable (y1 y2 y3 : ℝ)

def quadratic_function (x : ℝ) : ℝ := -x^2 + 4 * x - 5

theorem relationship_y1_y2_y3
  (h1 : quadratic_function (-4) = y1)
  (h2 : quadratic_function (-3) = y2)
  (h3 : quadratic_function (1) = y3) :
  y1 < y2 ∧ y2 < y3 :=
sorry

end relationship_y1_y2_y3_l36_36785


namespace percentage_problem_l36_36790

theorem percentage_problem (x : ℝ)
  (h : 0.70 * 600 = 0.40 * x) : x = 1050 :=
sorry

end percentage_problem_l36_36790


namespace sequence_solution_l36_36157

-- Define the sequence x_n
def x (n : ℕ) : ℚ := n / (n + 2016)

-- Given condition: x_2016 = x_m * x_n
theorem sequence_solution (m n : ℕ) (h : x 2016 = x m * x n) : 
  m = 4032 ∧ n = 6048 := 
  by sorry

end sequence_solution_l36_36157


namespace solve_porters_transportation_l36_36688

variable (x : ℝ)

def porters_transportation_equation : Prop :=
  (5000 / x = 8000 / (x + 600))

theorem solve_porters_transportation (x : ℝ) (h₁ : 600 > 0) (h₂ : x > 0):
  porters_transportation_equation x :=
sorry

end solve_porters_transportation_l36_36688


namespace jason_cutting_grass_time_l36_36635

-- Conditions
def time_to_cut_one_lawn : ℕ := 30 -- in minutes
def lawns_cut_each_day : ℕ := 8
def days : ℕ := 2
def minutes_in_an_hour : ℕ := 60

-- Proof that the number of hours Jason spends cutting grass over the weekend is 8
theorem jason_cutting_grass_time:
  ((lawns_cut_each_day * days) * time_to_cut_one_lawn) / minutes_in_an_hour = 8 :=
by
  sorry

end jason_cutting_grass_time_l36_36635


namespace toys_profit_l36_36246

theorem toys_profit (sp cp : ℕ) (x : ℕ) (h1 : sp = 25200) (h2 : cp = 1200) (h3 : 18 * cp + x * cp = sp) :
  x = 3 :=
by
  sorry

end toys_profit_l36_36246


namespace math_problem_l36_36524

theorem math_problem (a b : ℕ) (ha : a = 45) (hb : b = 15) :
  (a + b)^2 - 3 * (a^2 + b^2 - 2 * a * b) = 900 :=
by
  sorry

end math_problem_l36_36524


namespace smallest_n_square_19_and_ends_89_l36_36271

theorem smallest_n_square_19_and_ends_89 : ∃ n : ℕ, (n^2 % 100 = 89) ∧ (n^2 / 10^(nat.log10 (n^2) - 1) = 19) ∧ n = 1383 :=
by
  sorry

end smallest_n_square_19_and_ends_89_l36_36271


namespace units_digit_difference_l36_36709

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_difference :
  units_digit (72^3) - units_digit (24^3) = 4 :=
by
  sorry

end units_digit_difference_l36_36709


namespace goods_train_length_l36_36245

-- Conditions
def train1_speed := 60 -- kmph
def train2_speed := 52 -- kmph
def passing_time := 9 -- seconds

-- Conversion factor from kmph to meters per second
def kmph_to_mps (speed_kmph : ℕ) : ℕ := speed_kmph * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps := kmph_to_mps (train1_speed + train2_speed)

-- Final theorem statement
theorem goods_train_length :
  relative_speed_mps * passing_time = 280 :=
sorry

end goods_train_length_l36_36245


namespace arithmetic_sequence_sum_is_right_l36_36505

noncomputable def arithmetic_sequence_sum : ℤ :=
  let a1 := 1
  let d := -2
  let a2 := a1 + d
  let a3 := a1 + 2 * d
  let a6 := a1 + 5 * d
  let S6 := 6 * a1 + (6 * (6-1)) / 2 * d
  S6

theorem arithmetic_sequence_sum_is_right {d : ℤ} (h₀ : d ≠ 0) 
(h₁ : (a1 + 2 * d) ^ 2 = (a1 + d) * (a1 + 5 * d)) :
  arithmetic_sequence_sum = -24 := by
  sorry

end arithmetic_sequence_sum_is_right_l36_36505


namespace ball_box_distribution_l36_36818

theorem ball_box_distribution:
  ∃ (C : ℕ → ℕ → ℕ) (A : ℕ → ℕ → ℕ),
  C 4 2 * A 3 3 = sorry := 
by sorry

end ball_box_distribution_l36_36818


namespace wheel_distance_3_revolutions_l36_36412

theorem wheel_distance_3_revolutions (r : ℝ) (n : ℝ) (circumference : ℝ) (total_distance : ℝ) :
  r = 2 →
  n = 3 →
  circumference = 2 * Real.pi * r →
  total_distance = n * circumference →
  total_distance = 12 * Real.pi := by
  intros
  sorry

end wheel_distance_3_revolutions_l36_36412


namespace james_hives_l36_36307

-- Define all conditions
def hive_honey : ℕ := 20  -- Each hive produces 20 liters of honey
def jar_capacity : ℕ := 1/2  -- Each jar holds 0.5 liters
def jars_needed : ℕ := 100  -- James needs 100 jars for half the honey

-- Translate to Lean statement
theorem james_hives (hive_honey jar_capacity jars_needed : ℕ) :
  (hive_honey = 20) → 
  (jar_capacity = 1 / 2) →
  (jars_needed = 100) →
  (∀ hives : ℕ, (hives * hive_honey = 200) → hives = 5) :=
by
  intros Hhoney Hjar Hjars
  intros hives Hprod
  sorry

end james_hives_l36_36307


namespace evaluate_pow_l36_36750

theorem evaluate_pow : (64 : ℝ) = (8 : ℝ) ^ 2 → (8 : ℝ) = (2 : ℝ) ^ 3 → (64 : ℝ) ^ (5 / 6) = 32 :=
by
  intros h1 h2
  rw h1
  rw h2
  have h3 : (2 : ℝ)^3 ^ 2 = (2 : ℝ) ^ 6 := by ring_exp
  rw h3
  sorry

end evaluate_pow_l36_36750


namespace number_of_true_statements_l36_36052

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem number_of_true_statements (n : ℕ) :
  let s1 := reciprocal 4 + reciprocal 8 ≠ reciprocal 12
  let s2 := reciprocal 9 - reciprocal 3 ≠ reciprocal 6
  let s3 := reciprocal 5 * reciprocal 10 = reciprocal 50
  let s4 := reciprocal 16 / reciprocal 4 = reciprocal 4
  (cond s1 1 0) + (cond s2 1 0) + (cond s3 1 0) + (cond s4 1 0) = 2 := by
  sorry

end number_of_true_statements_l36_36052


namespace set_union_intersection_example_l36_36614

open Set

theorem set_union_intersection_example :
  let A := {1, 3, 4, 5}
  let B := {2, 4, 6}
  let C := {0, 1, 2, 3, 4}
  (A ∪ B) ∩ C = ({1, 2, 3, 4} : Set ℕ) :=
by
  sorry

end set_union_intersection_example_l36_36614


namespace greatest_AB_CBA_div_by_11_l36_36247

noncomputable def AB_CBA_max_value (A B C : ℕ) : ℕ := 10001 * A + 1010 * B + 100 * C + 10 * B + A

theorem greatest_AB_CBA_div_by_11 :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
  2 * A - 2 * B + C % 11 = 0 ∧ 
  ∀ (A' B' C' : ℕ),
    A' ≠ B' ∧ B' ≠ C' ∧ C' ≠ A' ∧ 
    2 * A' - 2 * B' + C' % 11 = 0 → 
    AB_CBA_max_value A B C ≥ AB_CBA_max_value A' B' C' :=
  by sorry

end greatest_AB_CBA_div_by_11_l36_36247


namespace breakfast_cost_l36_36590

theorem breakfast_cost :
  ∀ (muffin_cost fruit_cup_cost : ℕ) (francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups : ℕ),
  muffin_cost = 2 ∧ fruit_cup_cost = 3 ∧ francis_muffins = 2 ∧ francis_fruit_cups = 2 ∧ kiera_muffins = 2 ∧ kiera_fruit_cups = 1
  → (francis_muffins * muffin_cost + francis_fruit_cups * fruit_cup_cost + kiera_muffins * muffin_cost + kiera_fruit_cups * fruit_cup_cost = 17) :=
by
  intros muffin_cost fruit_cup_cost francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups
  intro cond
  cases cond with muffin_cost_eq rest
  cases rest with fruit_cup_cost_eq rest
  cases rest with francis_muffins_eq rest
  cases rest with francis_fruit_cups_eq rest
  cases rest with kiera_muffins_eq kiera_fruit_cups_eq

  rw [muffin_cost_eq, fruit_cup_cost_eq, francis_muffins_eq, francis_fruit_cups_eq, kiera_muffins_eq, kiera_fruit_cups_eq]
  norm_num
  sorry

end breakfast_cost_l36_36590


namespace sail_time_difference_l36_36427

theorem sail_time_difference (distance : ℕ) (v_big : ℕ) (v_small : ℕ) (t_big t_small : ℕ)
  (h_distance : distance = 200)
  (h_v_big : v_big = 50)
  (h_v_small : v_small = 20)
  (h_t_big : t_big = distance / v_big)
  (h_t_small : t_small = distance / v_small)
  : t_small - t_big = 6 := by
  sorry

end sail_time_difference_l36_36427


namespace solve_toenail_problem_l36_36616

def toenail_problem (b_toenails r_toenails_already r_toenails_more : ℕ) : Prop :=
  (b_toenails = 20) ∧
  (r_toenails_already = 40) ∧
  (r_toenails_more = 20) →
  (r_toenails_already + r_toenails_more = 60)

theorem solve_toenail_problem : toenail_problem 20 40 20 :=
by {
  sorry
}

end solve_toenail_problem_l36_36616


namespace one_point_one_billion_in_scientific_notation_l36_36080

noncomputable def one_point_one_billion : ℝ := 1.1 * 10^9

theorem one_point_one_billion_in_scientific_notation :
  1.1 * 10^9 = 1100000000 :=
by
  sorry

end one_point_one_billion_in_scientific_notation_l36_36080


namespace slope_of_decreasing_linear_function_l36_36155

theorem slope_of_decreasing_linear_function (m b : ℝ) :
  (∀ x y : ℝ, x < y → mx + b > my + b) → m < 0 :=
by
  intro h
  sorry

end slope_of_decreasing_linear_function_l36_36155


namespace complex_number_solution_l36_36592

theorem complex_number_solution (a b : ℝ) (i : ℂ) (h₀ : Complex.I = i)
  (h₁ : (a - 2* (i^3)) / (b + i) = i) : a + b = 1 :=
by 
  sorry

end complex_number_solution_l36_36592


namespace frac_sum_diff_l36_36437

theorem frac_sum_diff (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1001) : (a + b) / (a - b) = -1001 :=
sorry

end frac_sum_diff_l36_36437


namespace find_g8_l36_36994

variable (g : ℝ → ℝ)

theorem find_g8 (h1 : ∀ x y : ℝ, g (x + y) = g x + g y) (h2 : g 7 = 8) : g 8 = 64 / 7 :=
sorry

end find_g8_l36_36994


namespace geometric_sequence_11th_term_l36_36204

theorem geometric_sequence_11th_term (a r : ℝ) (h₁ : a * r ^ 4 = 8) (h₂ : a * r ^ 7 = 64) : 
  a * r ^ 10 = 512 :=
by sorry

end geometric_sequence_11th_term_l36_36204


namespace complement_intersection_l36_36008

open Set

theorem complement_intersection {x : ℝ} :
  (x ∉ {x | -2 ≤ x ∧ x ≤ 2}) ∧ (x < 1) ↔ (x < -2) := 
by
  sorry

end complement_intersection_l36_36008


namespace area_of_triangle_ACD_l36_36513

theorem area_of_triangle_ACD :
  ∀ (AD AC height_AD height_AC : ℝ),
  AD = 6 → height_AD = 3 → AC = 3 → height_AC = 3 →
  (1 / 2 * AD * height_AD - 1 / 2 * AC * height_AC) = 4.5 :=
by
  intros AD AC height_AD height_AC hAD hheight_AD hAC hheight_AC
  sorry

end area_of_triangle_ACD_l36_36513


namespace circle_equation_passing_through_points_symmetric_circle_equation_midpoint_trajectory_equation_l36_36444

-- Prove the equation of the circle passing through points A and B with center on a specified line
theorem circle_equation_passing_through_points
  (A B : ℝ × ℝ) (line : ℝ → ℝ → Prop)
  (N : ℝ → ℝ → Prop) :
  A = (3, 1) →
  B = (-1, 3) →
  (∀ x y, line x y ↔ 3 * x - y - 2 = 0) →
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  sorry :=
sorry

-- Prove the symmetric circle equation regarding a specified line
theorem symmetric_circle_equation
  (N N' : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) :
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  (∀ x y, N' x y ↔ (x - 1)^2 + (y - 5)^2 = 10) →
  (∀ x y, line x y ↔ x - y + 3 = 0) →
  sorry :=
sorry

-- Prove the trajectory equation of the midpoint
theorem midpoint_trajectory_equation
  (C : ℝ × ℝ) (N : ℝ → ℝ → Prop) (M_trajectory : ℝ → ℝ → Prop) :
  C = (3, 0) →
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  (∀ x y, M_trajectory x y ↔ (x - 5 / 2)^2 + (y - 2)^2 = 5 / 2) →
  sorry :=
sorry

end circle_equation_passing_through_points_symmetric_circle_equation_midpoint_trajectory_equation_l36_36444


namespace problem_solution_l36_36917

noncomputable def f (x : ℝ) : ℝ := x / (Real.cos x)

variables (x1 x2 x3 : ℝ)

axiom a1 : |x1| < (Real.pi / 2)
axiom a2 : |x2| < (Real.pi / 2)
axiom a3 : |x3| < (Real.pi / 2)

axiom h1 : f x1 + f x2 ≥ 0
axiom h2 : f x2 + f x3 ≥ 0
axiom h3 : f x3 + f x1 ≥ 0

theorem problem_solution : f (x1 + x2 + x3) ≥ 0 := sorry

end problem_solution_l36_36917


namespace cos_alpha_in_second_quadrant_l36_36914

theorem cos_alpha_in_second_quadrant (α : ℝ) (hα : π / 2 < α ∧ α < π) (h_tan : Real.tan α = -1 / 2) :
  Real.cos α = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end cos_alpha_in_second_quadrant_l36_36914


namespace percentage_spent_on_household_items_l36_36497

def Raja_income : ℝ := 37500
def clothes_percentage : ℝ := 0.20
def medicines_percentage : ℝ := 0.05
def savings_amount : ℝ := 15000

theorem percentage_spent_on_household_items : 
  (Raja_income - (clothes_percentage * Raja_income + medicines_percentage * Raja_income + savings_amount)) / Raja_income * 100 = 35 :=
  sorry

end percentage_spent_on_household_items_l36_36497


namespace find_q_l36_36455

theorem find_q (q x : ℝ) (h1 : x = 2) (h2 : q * x - 3 = 11) : q = 7 :=
by
  sorry

end find_q_l36_36455


namespace profit_calculation_l36_36511

theorem profit_calculation (cost_price_per_card_yuan : ℚ) (total_sales_yuan : ℚ)
  (n : ℕ) (sales_price_per_card_yuan : ℚ)
  (h1 : cost_price_per_card_yuan = 0.21)
  (h2 : total_sales_yuan = 14.57)
  (h3 : total_sales_yuan = n * sales_price_per_card_yuan)
  (h4 : sales_price_per_card_yuan ≤ 2 * cost_price_per_card_yuan) :
  (total_sales_yuan - n * cost_price_per_card_yuan = 4.7) :=
by
  sorry

end profit_calculation_l36_36511


namespace units_digit_of_j_squared_plus_3_power_j_l36_36051

def j : ℕ := 19^2 + 3^10

theorem units_digit_of_j_squared_plus_3_power_j :
  ((j^2 + 3^j) % 10) = 3 :=
by
  sorry

end units_digit_of_j_squared_plus_3_power_j_l36_36051


namespace find_speed_of_stream_l36_36842

theorem find_speed_of_stream (x : ℝ) (h1 : ∃ x, 1 / (39 - x) = 2 * (1 / (39 + x))) : x = 13 :=
by
sorry

end find_speed_of_stream_l36_36842


namespace Alden_nephews_10_years_ago_l36_36253

noncomputable def nephews_Alden_now : ℕ := sorry
noncomputable def nephews_Alden_10_years_ago (N : ℕ) : ℕ := N / 2
noncomputable def nephews_Vihaan_now (N : ℕ) : ℕ := N + 60
noncomputable def total_nephews (N : ℕ) : ℕ := N + (nephews_Vihaan_now N)

theorem Alden_nephews_10_years_ago (N : ℕ) (h1 : total_nephews N = 260) : 
  nephews_Alden_10_years_ago N = 50 :=
by
  sorry

end Alden_nephews_10_years_ago_l36_36253


namespace complex_arithmetic_1_complex_arithmetic_2_l36_36865

-- Proof Problem 1
theorem complex_arithmetic_1 : 
  (1 : ℂ) * (-2 - 4 * I) - (7 - 5 * I) + (1 + 7 * I) = -8 + 8 * I := 
sorry

-- Proof Problem 2
theorem complex_arithmetic_2 : 
  (1 + I) * (2 + I) + (5 + I) / (1 - I) + (1 - I) ^ 2 = 3 + 4 * I := 
sorry

end complex_arithmetic_1_complex_arithmetic_2_l36_36865


namespace optionC_is_correct_l36_36640

def KalobsWindowLength : ℕ := 50
def KalobsWindowWidth : ℕ := 80
def KalobsWindowArea : ℕ := KalobsWindowLength * KalobsWindowWidth

def DoubleKalobsWindowArea : ℕ := 2 * KalobsWindowArea

def optionC_Length : ℕ := 50
def optionC_Width : ℕ := 160
def optionC_Area : ℕ := optionC_Length * optionC_Width

theorem optionC_is_correct : optionC_Area = DoubleKalobsWindowArea := by
  sorry

end optionC_is_correct_l36_36640


namespace smaller_angle_at_6_30_l36_36617
-- Import the Mathlib library

-- Define the conditions as a structure
structure ClockAngleConditions where
  hours_on_clock : ℕ
  degrees_per_hour : ℕ
  minute_hand_position : ℕ
  hour_hand_position : ℕ

-- Initialize the conditions for 6:30
def conditions : ClockAngleConditions := {
  hours_on_clock := 12,
  degrees_per_hour := 30,
  minute_hand_position := 180,
  hour_hand_position := 195
}

-- Define the theorem to be proven
theorem smaller_angle_at_6_30 (c : ClockAngleConditions) : 
  c.hour_hand_position - c.minute_hand_position = 15 :=
by
  -- Skip the proof
  sorry

end smaller_angle_at_6_30_l36_36617


namespace circle_representation_l36_36030

theorem circle_representation (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2 * m * x + 2 * m^2 + 2 * m - 3 = 0) ↔ m ∈ Set.Ioo (-3 : ℝ) (1 / 2 : ℝ) :=
by
  sorry

end circle_representation_l36_36030


namespace range_of_a_l36_36125

def tensor (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 2 → tensor (x - a) x ≤ a + 2) → a ≤ 7 :=
by
  sorry

end range_of_a_l36_36125


namespace range_of_a_l36_36989

def p (a : ℝ) := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0
def q (a : ℝ) := ∃ x₀ : ℝ, x₀^2 + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 :=
by
  sorry

end range_of_a_l36_36989


namespace population_net_increase_l36_36625

-- Definitions for birth and death rate, and the number of seconds in a day
def birth_rate : ℕ := 10
def death_rate : ℕ := 2
def seconds_in_day : ℕ := 86400

-- Calculate the population net increase in one day
theorem population_net_increase (birth_rate death_rate seconds_in_day : ℕ) :
  (seconds_in_day / 2) * birth_rate - (seconds_in_day / 2) * death_rate = 345600 :=
by
  sorry

end population_net_increase_l36_36625


namespace graph_passes_through_0_1_l36_36207

theorem graph_passes_through_0_1 (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (0, 1) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^x) } :=
sorry

end graph_passes_through_0_1_l36_36207


namespace minimum_value_768_l36_36330

noncomputable def min_value_expression (a b c : ℝ) := a^2 + 8 * a * b + 16 * b^2 + 2 * c^5

theorem minimum_value_768 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_condition : a * b^2 * c^3 = 256) : 
  min_value_expression a b c = 768 :=
sorry

end minimum_value_768_l36_36330


namespace speed_of_stream_l36_36407

theorem speed_of_stream (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 11) (h2 : upstream_speed = 8) : 
    (downstream_speed - upstream_speed) / 2 = 1.5 :=
by
  rw [h1, h2]
  simp
  norm_num

end speed_of_stream_l36_36407


namespace find_number_of_girls_l36_36462

noncomputable def B (G : ℕ) : ℕ := (8 * G) / 5

theorem find_number_of_girls (B G : ℕ) (h_ratio : B = (8 * G) / 5) (h_total : B + G = 312) : G = 120 :=
by
  -- the proof would be done here
  sorry

end find_number_of_girls_l36_36462


namespace value_of_k_l36_36641

open Nat

theorem value_of_k (k : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n : ℕ, 0 < n → S n = k * (n : ℝ) ^ 2 + (n : ℝ))
  (h_a : ∀ n : ℕ, 1 < n → a n = S n - S (n-1))
  (h_geom : ∀ m : ℕ, 0 < m → (a m) ≠ 0 → (a (2*m))^2 = a m * a (4*m)) :
  k = 0 ∨ k = 1 :=
sorry

end value_of_k_l36_36641


namespace octopus_legs_l36_36379

/-- Four octopuses made statements about their total number of legs.
    - Octopuses with 7 legs always lie.
    - Octopuses with 6 or 8 legs always tell the truth.
    - Blue: "Together we have 28 legs."
    - Green: "Together we have 27 legs."
    - Yellow: "Together we have 26 legs."
    - Red: "Together we have 25 legs."
   Prove that the Green octopus has 6 legs, and the Blue, Yellow, and Red octopuses each have 7 legs.
-/
theorem octopus_legs (L_B L_G L_Y L_R : ℕ) (H1 : (L_B + L_G + L_Y + L_R = 28 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 27 → L_B + L_G + L_Y + L_R = 27) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 26 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 25 → L_B ≠ 7)) : 
  (L_G = 6) ∧ (L_B = 7) ∧ (L_Y = 7) ∧ (L_R = 7) :=
sorry

end octopus_legs_l36_36379


namespace avg_nested_l36_36661

def avg (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem avg_nested {x y z : ℕ} :
  avg (avg 2 3 1) (avg 4 1 0) 5 = 26 / 9 :=
by
  sorry

end avg_nested_l36_36661


namespace problem_statement_l36_36538

-- Definitions
def div_remainder (a b : ℕ) : ℕ × ℕ :=
  (a / b, a % b)

-- Conditions and question as Lean structures
def condition := ∀ (a b k : ℕ), k ≠ 0 → div_remainder (a * k) (b * k) = (a / b, (a % b) * k)
def question := div_remainder 4900 600 = div_remainder 49 6

-- Theorem stating the problem's conclusion
theorem problem_statement (cond : condition) : ¬question :=
by
  sorry

end problem_statement_l36_36538


namespace min_f_eq_2_m_n_inequality_l36_36769

def f (x : ℝ) := abs (x + 1) + abs (x - 1)

theorem min_f_eq_2 : (∀ x, f x ≥ 2) ∧ (∃ x, f x = 2) :=
by
  sorry

theorem m_n_inequality (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m^3 + n^3 = 2) : m + n ≤ 2 :=
by
  sorry

end min_f_eq_2_m_n_inequality_l36_36769


namespace perpendicular_lines_l36_36987

theorem perpendicular_lines (a : ℝ) : 
  (a = -1 → (∀ x y : ℝ, 4 * x - (a + 1) * y + 9 = 0 → x ≠ 0 →  y ≠ 0 → 
  ∃ b : ℝ, (b^2 + 1) * x - b * y + 6 = 0)) ∧ 
  (∀ x y : ℝ, (4 * x - (a + 1) * y + 9 = 0) ∧ (∃ x y : ℝ, (a^2 - 1) * x - a * y + 6 = 0) → a ≠ -1) := 
sorry

end perpendicular_lines_l36_36987


namespace winston_initial_quarters_l36_36229

-- Defining the conditions
def spent_candy := 50 -- 50 cents spent on candy
def remaining_cents := 300 -- 300 cents left

-- Defining the value of a quarter in cents
def value_of_quarter := 25

-- Calculating the number of quarters Winston initially had
def initial_quarters := (spent_candy + remaining_cents) / value_of_quarter

-- Proof statement
theorem winston_initial_quarters : initial_quarters = 14 := 
by sorry

end winston_initial_quarters_l36_36229


namespace john_pays_percentage_of_srp_l36_36375

theorem john_pays_percentage_of_srp (P MP : ℝ) (h1 : P = 1.20 * MP) (h2 : MP > 0): 
  (0.60 * MP / P) * 100 = 50 :=
by
  sorry

end john_pays_percentage_of_srp_l36_36375


namespace exponential_inequality_l36_36394

theorem exponential_inequality (k l m : ℕ) : 2^(k+1) + 2^(k+m) + 2^(l+m) ≤ 2^(k+l+m+1) + 1 :=
by
  sorry

end exponential_inequality_l36_36394


namespace number_of_subsets_A_range_of_m_for_empty_intersection_l36_36903

def A := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) := {x | (m - 1) ≤ x ∧ x ≤ (2 * m + 1)}

theorem number_of_subsets_A (x : ℕ) (hx : 0 < x ∧ x ≤ 5) :
  fintype.card (settofinset { x | ∃ hx : ℕ, x ∈ A }) = 32 :=
by {
  -- We show that the number of subsets of {1, 2, 3, 4, 5}
  sorry
}

theorem range_of_m_for_empty_intersection (A : set ℝ) (B : ℝ → set ℝ) :
  (∀ x ∈ ℝ, (x ∈ A → x ∉ B m) ∧ (x ∈ B m → x ∉ A)) →
  (m < -3/2 ∨ m > 6) :=
by {
  -- We prove the range of m when A has no intersection with B
  sorry
}

end number_of_subsets_A_range_of_m_for_empty_intersection_l36_36903


namespace find_other_number_l36_36081

noncomputable def HCF : ℕ := 14
noncomputable def LCM : ℕ := 396
noncomputable def one_number : ℕ := 154
noncomputable def product_of_numbers : ℕ := HCF * LCM

theorem find_other_number (other_number : ℕ) :
  HCF * LCM = one_number * other_number → other_number = 36 :=
by
  sorry

end find_other_number_l36_36081


namespace man_speed_with_stream_is_4_l36_36107

noncomputable def man's_speed_with_stream (Vm Vs : ℝ) : ℝ := Vm + Vs

theorem man_speed_with_stream_is_4 (Vm : ℝ) (Vs : ℝ) 
  (h1 : Vm - Vs = 4) 
  (h2 : Vm = 4) : man's_speed_with_stream Vm Vs = 4 :=
by 
  -- The proof is omitted as per instructions
  sorry

end man_speed_with_stream_is_4_l36_36107


namespace combination_lock_code_l36_36202

theorem combination_lock_code :
  ∀ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ (x + y + x * y = 10 * x + y) →
  10 * x + y = 19 ∨ 10 * x + y = 29 ∨ 10 * x + y = 39 ∨ 10 * x + y = 49 ∨
  10 * x + y = 59 ∨ 10 * x + y = 69 ∨ 10 * x + y = 79 ∨ 10 * x + y = 89 ∨
  10 * x + y = 99 :=
by
  sorry

end combination_lock_code_l36_36202


namespace acid_base_mixture_ratio_l36_36382

theorem acid_base_mixture_ratio (r s t : ℝ) (hr : r ≥ 0) (hs : s ≥ 0) (ht : t ≥ 0) :
  (r ≠ -1) → (s ≠ -1) → (t ≠ -1) →
  let acid_volume := (r/(r+1) + s/(s+1) + t/(t+1))
  let base_volume := (1/(r+1) + 1/(s+1) + 1/(t+1))
  acid_volume / base_volume = (rst + rt + rs + st) / (rs + rt + st + r + s + t + 3) := 
by {
  sorry
}

end acid_base_mixture_ratio_l36_36382


namespace evaluate_expression_l36_36390

theorem evaluate_expression : 3^(1^(2^3)) + ((3^1)^2)^2 = 84 := 
by
  sorry

end evaluate_expression_l36_36390


namespace num_ways_to_select_five_crayons_including_red_l36_36843

noncomputable def num_ways_select_five_crayons (total_crayons : ℕ) (selected_crayons : ℕ) (fixed_red_crayon : ℕ) : ℕ :=
  Nat.choose (total_crayons - fixed_red_crayon) selected_crayons

theorem num_ways_to_select_five_crayons_including_red
  (total_crayons : ℕ) 
  (fixed_red_crayon : ℕ)
  (selected_crayons : ℕ)
  (h1 : total_crayons = 15)
  (h2 : fixed_red_crayon = 1)
  (h3 : selected_crayons = 4) : 
  num_ways_select_five_crayons total_crayons selected_crayons fixed_red_crayon = 1001 := by
  sorry

end num_ways_to_select_five_crayons_including_red_l36_36843


namespace find_m_l36_36296

theorem find_m (m : ℝ) (h : 2^2 + 2 * m + 2 = 0) : m = -3 :=
by {
  sorry
}

end find_m_l36_36296


namespace probability_red_side_given_observed_l36_36867

def total_cards : ℕ := 9
def black_black_cards : ℕ := 4
def black_red_cards : ℕ := 2
def red_red_cards : ℕ := 3

def red_sides : ℕ := red_red_cards * 2 + black_red_cards
def red_red_sides : ℕ := red_red_cards * 2
def probability_other_side_is_red (total_red_sides red_red_sides : ℕ) : ℚ :=
  red_red_sides / total_red_sides

theorem probability_red_side_given_observed :
  probability_other_side_is_red red_sides red_red_sides = 3 / 4 :=
by
  unfold red_sides
  unfold red_red_sides
  unfold probability_other_side_is_red
  sorry

end probability_red_side_given_observed_l36_36867


namespace juanita_spends_more_l36_36781

-- Define the expenditures
def grant_yearly_expenditure : ℝ := 200.00

def juanita_weekday_expenditure : ℝ := 0.50

def juanita_sunday_expenditure : ℝ := 2.00

def weeks_per_year : ℕ := 52

-- Given conditions translated to Lean
def juanita_weekly_expenditure : ℝ :=
  (juanita_weekday_expenditure * 6) + juanita_sunday_expenditure

def juanita_yearly_expenditure : ℝ :=
  juanita_weekly_expenditure * weeks_per_year

-- The statement we need to prove
theorem juanita_spends_more : (juanita_yearly_expenditure - grant_yearly_expenditure) = 60.00 :=
by
  sorry

end juanita_spends_more_l36_36781


namespace Jimin_weight_l36_36794

variable (T J : ℝ)

theorem Jimin_weight (h1 : T - J = 4) (h2 : T + J = 88) : J = 42 :=
sorry

end Jimin_weight_l36_36794


namespace ratio_area_rect_sq_l36_36209

/-- 
  Given:
  1. The longer side of rectangle R is 1.2 times the length of a side of square S.
  2. The shorter side of rectangle R is 0.85 times the length of a side of square S.
  Prove that the ratio of the area of rectangle R to the area of square S is 51/50.
-/
theorem ratio_area_rect_sq (s : ℝ) 
  (h1 : ∃ r1, r1 = 1.2 * s) 
  (h2 : ∃ r2, r2 = 0.85 * s) : 
  (1.2 * s * 0.85 * s) / (s * s) = 51 / 50 := 
by
  sorry

end ratio_area_rect_sq_l36_36209


namespace smallest_possible_sum_l36_36962

theorem smallest_possible_sum (E F G H : ℕ) (h1 : F > 0) (h2 : E + F + G = 3 * F) (h3 : F * G = 4 * F * F / 3) :
  E = 6 ∧ F = 9 ∧ G = 12 ∧ H = 16 ∧ E + F + G + H = 43 :=
by 
  sorry

end smallest_possible_sum_l36_36962


namespace bus_passengers_l36_36685

variable (P : ℕ) -- P represents the initial number of passengers

theorem bus_passengers (h1 : P + 16 - 17 = 49) : P = 50 :=
by
  sorry

end bus_passengers_l36_36685


namespace evaluate_expression_l36_36729

theorem evaluate_expression : (-1:ℤ)^2022 + |(-2:ℤ)| - (1/2 : ℚ)^0 - 2 * Real.tan (Real.pi / 4) = 0 := 
by
  sorry

end evaluate_expression_l36_36729


namespace average_of_numbers_l36_36503

noncomputable def x := (5050 : ℚ) / 5049

theorem average_of_numbers :
  let sum := (∑ i in Finset.range 101, (i + 1)) + x in
  let avg := sum / (101 + 1) in
  avg = 50 * x :=
by
  let sum := (∑ i in Finset.range 101, (i + 1)) + x
  let avg := sum / (101 + 1)
  have sum_formula : (∑ i in Finset.range 101, (i + 1)) = 5050 := sorry
  have avg_formula : avg = 50 * x := sorry
  exact avg_formula

end average_of_numbers_l36_36503


namespace translation_of_civilisation_l36_36531

def translation (word : String) (translation : String) : Prop :=
translation = "civilization"

theorem translation_of_civilisation (word : String) :
  word = "civilisation" → translation word "civilization" :=
by sorry

end translation_of_civilisation_l36_36531


namespace range_of_z_l36_36666

theorem range_of_z (x y : ℝ) (h : x^2 / 16 + y^2 / 9 = 1) : -5 ≤ x + y ∧ x + y ≤ 5 :=
sorry

end range_of_z_l36_36666


namespace total_journey_distance_l36_36713

theorem total_journey_distance : 
  ∃ D : ℝ, 
    (∀ (T : ℝ), T = 10) →
    ((D/2) / 21 + (D/2) / 24 = 10) →
    D = 224 := 
by
  sorry

end total_journey_distance_l36_36713


namespace geometric_sequence_a3_a5_l36_36629

-- Define the geometric sequence condition using a function
def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Define the given conditions
variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (h1 : is_geometric_seq a)
variable (h2 : a 1 > 0)
variable (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)

-- The main goal is to prove: a 3 + a 5 = 5
theorem geometric_sequence_a3_a5 : a 3 + a 5 = 5 :=
by
  simp [is_geometric_seq] at h1
  obtain ⟨q, ⟨hq_pos, hq⟩⟩ := h1
  sorry

end geometric_sequence_a3_a5_l36_36629


namespace plain_pancakes_l36_36571

/-- Define the given conditions -/
def total_pancakes : ℕ := 67
def blueberry_pancakes : ℕ := 20
def banana_pancakes : ℕ := 24

/-- Define a theorem stating the number of plain pancakes given the conditions -/
theorem plain_pancakes : total_pancakes - (blueberry_pancakes + banana_pancakes) = 23 := by
  -- Here we will provide a proof
  sorry

end plain_pancakes_l36_36571


namespace range_of_a_l36_36010

variable (a : ℝ)

def proposition_p : Prop :=
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x^2 - a ≥ 0)

def proposition_q : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + (2 - a) = 0

theorem range_of_a (hp : proposition_p a) (hq : proposition_q a) : a ≤ -2 ∨ a = 1 :=
  sorry

end range_of_a_l36_36010


namespace determine_S5_l36_36988

noncomputable def S (x : ℝ) (m : ℕ) : ℝ := x^m + 1 / x^m

theorem determine_S5 (x : ℝ) (h : x + 1 / x = 3) : S x 5 = 123 :=
by
  sorry

end determine_S5_l36_36988


namespace cost_of_pencils_and_notebooks_l36_36509

variable (p n : ℝ)

theorem cost_of_pencils_and_notebooks 
  (h1 : 9 * p + 10 * n = 5.06) 
  (h2 : 6 * p + 4 * n = 2.42) :
  20 * p + 14 * n = 8.31 :=
by
  sorry

end cost_of_pencils_and_notebooks_l36_36509


namespace Daria_vacuum_cleaner_problem_l36_36566

theorem Daria_vacuum_cleaner_problem (initial_savings weekly_savings target_savings weeks_needed : ℕ)
  (h1 : initial_savings = 20)
  (h2 : weekly_savings = 10)
  (h3 : target_savings = 120)
  (h4 : weeks_needed = (target_savings - initial_savings) / weekly_savings) : 
  weeks_needed = 10 :=
by
  sorry

end Daria_vacuum_cleaner_problem_l36_36566


namespace speed_of_man_l36_36724

/-
  Problem Statement:
  A train 100 meters long takes 6 seconds to cross a man walking at a certain speed in the direction opposite to that of the train. The speed of the train is 54.99520038396929 kmph. What is the speed of the man in kmph?
-/
 
theorem speed_of_man :
  ∀ (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_train_kmph : ℝ) (relative_speed_mps : ℝ),
    length_of_train = 100 →
    time_to_cross = 6 →
    speed_of_train_kmph = 54.99520038396929 →
    relative_speed_mps = length_of_train / time_to_cross →
    (relative_speed_mps - (speed_of_train_kmph * (1000 / 3600))) * (3600 / 1000) = 5.00479961403071 :=
by
  intros length_of_train time_to_cross speed_of_train_kmph relative_speed_mps
  intros h1 h2 h3 h4
  sorry

end speed_of_man_l36_36724


namespace quadratic_expression_value_l36_36765

theorem quadratic_expression_value (a : ℝ) :
  (∃ x : ℝ, (3 * a - 1) * x^2 - a * x + 1 / 4 = 0 ∧ 
  (3 * a - 1) * x^2 - a * x + 1 / 4 = 0 ∧ 
  a^2 - 3 * a + 1 = 0) → 
  a^2 - 2 * a + 2021 + 1 / a = 2023 := 
sorry

end quadratic_expression_value_l36_36765


namespace circle_line_intersection_points_l36_36604

noncomputable def radius : ℝ := 6
noncomputable def distance : ℝ := 5

theorem circle_line_intersection_points :
  radius > distance -> number_of_intersection_points = 2 := 
by
  sorry

end circle_line_intersection_points_l36_36604


namespace dosage_range_l36_36084

theorem dosage_range (d : ℝ) (h : 60 ≤ d ∧ d ≤ 120) : 15 ≤ (d / 4) ∧ (d / 4) ≤ 30 :=
by
  sorry

end dosage_range_l36_36084


namespace fly_least_distance_l36_36109

noncomputable def leastDistance (r : ℝ) (h : ℝ) (start_dist : ℝ) (end_dist : ℝ) : ℝ := 
  let C := 2 * Real.pi * r
  let R := Real.sqrt (r^2 + h^2)
  let θ := C / R
  let A := (start_dist, 0)
  let B := (Real.cos (θ / 2) * end_dist, Real.sin (θ / 2) * end_dist)
  Real.sqrt ((B.fst - A.fst)^2 + (B.snd - A.snd)^2)

theorem fly_least_distance : 
  leastDistance 600 (200 * Real.sqrt 7) 125 (375 * Real.sqrt 2) = 625 := 
sorry

end fly_least_distance_l36_36109


namespace octopus_legs_l36_36378

-- Definitions of octopus behavior based on the number of legs
def tells_truth (legs: ℕ) : Prop := legs = 6 ∨ legs = 8
def lies (legs: ℕ) : Prop := legs = 7

-- Statements made by the octopuses
def blue_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 28
def green_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 27
def yellow_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 26
def red_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 25

noncomputable def legs_b := 7
noncomputable def legs_g := 6
noncomputable def legs_y := 7
noncomputable def legs_r := 7

-- Main theorem
theorem octopus_legs : 
  (tells_truth legs_g) ∧ 
  (lies legs_b) ∧ 
  (lies legs_y) ∧ 
  (lies legs_r) ∧ 
  blue_statement legs_b legs_g legs_y legs_r ∧ 
  green_statement legs_b legs_g legs_y legs_r ∧ 
  yellow_statement legs_b legs_g legs_y legs_r ∧ 
  red_statement legs_b legs_g legs_y legs_r := 
by 
  sorry

end octopus_legs_l36_36378


namespace sequence_term_1000_l36_36035

theorem sequence_term_1000 :
  ∃ (a : ℕ → ℤ), a 1 = 2007 ∧ a 2 = 2008 ∧ (∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n) ∧ a 1000 = 2340 := 
by
  sorry

end sequence_term_1000_l36_36035


namespace inscribed_circle_radius_integer_l36_36351

theorem inscribed_circle_radius_integer 
  (a b c : ℕ) (h : a^2 + b^2 = c^2) 
  (h₀ : 2 * (a + b - c) = k) 
  : ∃ (r : ℕ), r = (a + b - c) / 2 := 
begin
  sorry
end

end inscribed_circle_radius_integer_l36_36351


namespace remainder_of_sum_of_integers_mod_15_l36_36711

theorem remainder_of_sum_of_integers_mod_15 (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end remainder_of_sum_of_integers_mod_15_l36_36711


namespace equilateral_triangle_side_length_l36_36817

noncomputable def side_length_of_triangle (PQ PR PS : ℕ) : ℝ := 
  let s := 8 * Real.sqrt 3
  s

theorem equilateral_triangle_side_length (PQ PR PS : ℕ) (P_inside_triangle : true) 
  (Q_foot : true) (R_foot : true) (S_foot : true)
  (hPQ : PQ = 2) (hPR : PR = 4) (hPS : PS = 6) : 
  side_length_of_triangle PQ PR PS = 8 * Real.sqrt 3 := 
sorry

end equilateral_triangle_side_length_l36_36817


namespace recreation_percentage_l36_36398

def wages_last_week (W : ℝ) : ℝ := W
def spent_on_recreation_last_week (W : ℝ) : ℝ := 0.15 * W
def wages_this_week (W : ℝ) : ℝ := 0.90 * W
def spent_on_recreation_this_week (W : ℝ) : ℝ := 0.30 * (wages_this_week W)

theorem recreation_percentage (W : ℝ) (hW: W > 0) :
  (spent_on_recreation_this_week W) / (spent_on_recreation_last_week W) * 100 = 180 := by
  sorry

end recreation_percentage_l36_36398


namespace black_friday_sales_l36_36492

variable (n : ℕ) (initial_sales increment : ℕ)

def yearly_sales (sales: ℕ) (inc: ℕ) (years: ℕ) : ℕ :=
  sales + years * inc

theorem black_friday_sales (h1 : initial_sales = 327) (h2 : increment = 50) :
  yearly_sales initial_sales increment 3 = 477 := by
  sorry

end black_friday_sales_l36_36492


namespace intersection_subset_l36_36611

def set_A : Set ℝ := {x | -4 < x ∧ x < 2}
def set_B : Set ℝ := {x | x > 1 ∨ x < -5}
def set_C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m}

theorem intersection_subset (m : ℝ) :
  (set_A ∩ set_B) ⊆ set_C m ↔ m = 2 :=
by
  sorry

end intersection_subset_l36_36611


namespace area_of_isosceles_trapezoid_with_inscribed_circle_l36_36139

-- Definitions
def is_isosceles_trapezoid_with_inscribed_circle (a b c : ℕ) : Prop :=
  a + b = 2 * c

def height_of_trapezoid (c : ℕ) (half_diff_bases : ℕ) : ℕ :=
  (c^2 - half_diff_bases^2).sqrt

noncomputable def area_of_trapezoid (a b h : ℕ) : ℕ :=
  (a + b) * h / 2

-- Given values
def base1 := 2
def base2 := 8
def leg := 5
def height := 4

-- Proof Statement
theorem area_of_isosceles_trapezoid_with_inscribed_circle :
  is_isosceles_trapezoid_with_inscribed_circle base1 base2 leg →
  height_of_trapezoid leg ((base2 - base1) / 2) = height →
  area_of_trapezoid base1 base2 height = 20 :=
by
  intro h1 h2
  sorry

end area_of_isosceles_trapezoid_with_inscribed_circle_l36_36139


namespace work_completion_days_l36_36494

variable (Paul_days Rose_days Sam_days : ℕ)

def Paul_rate := 1 / 80
def Rose_rate := 1 / 120
def Sam_rate := 1 / 150

def combined_rate := Paul_rate + Rose_rate + Sam_rate

noncomputable def days_to_complete_work := 1 / combined_rate

theorem work_completion_days :
  Paul_days = 80 →
  Rose_days = 120 →
  Sam_days = 150 →
  days_to_complete_work = 37 := 
by
  intros
  simp only [Paul_rate, Rose_rate, Sam_rate, combined_rate, days_to_complete_work]
  sorry

end work_completion_days_l36_36494


namespace maximum_acute_triangles_from_four_points_l36_36598

-- Define a point in a plane
structure Point (α : Type) := (x : α) (y : α)

-- Definition of an acute triangle is intrinsic to the problem
def is_acute_triangle {α : Type} [LinearOrderedField α] (A B C : Point α) : Prop :=
  sorry -- Assume implementation for determining if a triangle is acute angles based

def maximum_number_acute_triangles {α : Type} [LinearOrderedField α] (A B C D : Point α) : ℕ :=
  sorry -- Assume implementation for verifying maximum number of acute triangles from four points

theorem maximum_acute_triangles_from_four_points {α : Type} [LinearOrderedField α] (A B C D : Point α) :
  maximum_number_acute_triangles A B C D = 4 :=
  sorry

end maximum_acute_triangles_from_four_points_l36_36598


namespace smallest_n_for_perfect_square_and_cube_l36_36707

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, 0 < n ∧ (∃ a1 b1 : ℕ, 4 * n = a1 ^ 2 ∧ 5 * n = b1 ^ 3 ∧ n = 50) :=
begin
  use 50,
  split,
  { norm_num, },
  { use [10, 5],
    split,
    { norm_num, },
    { split, 
      { norm_num, },
      { refl, }, },
  },
  sorry
end

end smallest_n_for_perfect_square_and_cube_l36_36707


namespace split_cost_evenly_l36_36308

noncomputable def cupcake_cost : ℝ := 1.50
noncomputable def number_of_cupcakes : ℝ := 12
noncomputable def total_cost : ℝ := number_of_cupcakes * cupcake_cost
noncomputable def total_people : ℝ := 2

theorem split_cost_evenly : (total_cost / total_people) = 9 :=
by
  -- Skipping the proof for now
  sorry

end split_cost_evenly_l36_36308


namespace determine_range_of_k_l36_36772

noncomputable def inequality_holds_for_all_x (k : ℝ) : Prop :=
  ∀ (x : ℝ), x^4 + (k - 1) * x^2 + 1 ≥ 0

theorem determine_range_of_k (k : ℝ) : inequality_holds_for_all_x k ↔ k ≥ 1 := sorry

end determine_range_of_k_l36_36772


namespace second_game_score_count_l36_36218

-- Define the conditions and problem
def total_points (A1 A2 A3 B1 B2 B3 : ℕ) : Prop :=
  A1 + A2 + A3 + B1 + B2 + B3 = 31

def valid_game_1 (A1 B1 : ℕ) : Prop :=
  A1 ≥ 11 ∧ A1 - B1 ≥ 2

def valid_game_2 (A2 B2 : ℕ) : Prop :=
  B2 ≥ 11 ∧ B2 - A2 ≥ 2

def valid_game_3 (A3 B3 : ℕ) : Prop :=
  A3 ≥ 11 ∧ A3 - B3 ≥ 2

def game_sequence (A1 A2 A3 B1 B2 B3 : ℕ) : Prop :=
  valid_game_1 A1 B1 ∧ valid_game_2 A2 B2 ∧ valid_game_3 A3 B3

noncomputable def second_game_score_possibilities : ℕ := 
  8 -- This is derived from calculating the valid scores where B wins the second game.

theorem second_game_score_count (A1 A2 A3 B1 B2 B3 : ℕ) (h_total : total_points A1 A2 A3 B1 B2 B3) (h_sequence : game_sequence A1 A2 A3 B1 B2 B3) :
  second_game_score_possibilities = 8 := sorry

end second_game_score_count_l36_36218


namespace largest_gcd_l36_36680

theorem largest_gcd (a b : ℕ) (h : a + b = 1008) : ∃ d, d = gcd a b ∧ (∀ d', d' = gcd a b → d' ≤ d) ∧ d = 504 :=
by
  sorry

end largest_gcd_l36_36680


namespace pure_alcohol_addition_problem_l36_36232

-- Define the initial conditions
def initial_volume := 6
def initial_concentration := 0.30
def final_concentration := 0.50

-- Define the amount of pure alcohol to be added
def x := 2.4

-- Proof problem statement
theorem pure_alcohol_addition_problem (initial_volume initial_concentration final_concentration x : ℝ) :
  initial_volume * initial_concentration + x = final_concentration * (initial_volume + x) :=
by
  -- Initial condition values definition
  let initial_volume := 6
  let initial_concentration := 0.30
  let final_concentration := 0.50
  let x := 2.4
  -- Skip the proof
  sorry

end pure_alcohol_addition_problem_l36_36232


namespace napkin_ratio_l36_36490

theorem napkin_ratio (initial_napkins : ℕ) (napkins_after : ℕ) (olivia_napkins : ℕ) (amelia_napkins : ℕ)
  (h1 : initial_napkins = 15) (h2 : napkins_after = 45) (h3 : olivia_napkins = 10)
  (h4 : initial_napkins + olivia_napkins + amelia_napkins = napkins_after) :
  amelia_napkins / olivia_napkins = 2 := by
  sorry

end napkin_ratio_l36_36490


namespace simplify_expression_l36_36657

theorem simplify_expression (k : ℂ) : 
  ((1 / (3 * k)) ^ (-3) * ((-2) * k) ^ (4)) = 432 * (k ^ 7) := 
by sorry

end simplify_expression_l36_36657


namespace quotient_is_20_l36_36931

theorem quotient_is_20 (D d r Q : ℕ) (hD : D = 725) (hd : d = 36) (hr : r = 5) (h : D = d * Q + r) :
  Q = 20 :=
by sorry

end quotient_is_20_l36_36931


namespace time_to_cover_escalator_l36_36878

-- Definitions for the provided conditions.
def escalator_speed : ℝ := 7
def escalator_length : ℝ := 180
def person_speed : ℝ := 2

-- Goal to prove the time taken to cover the escalator length.
theorem time_to_cover_escalator : (escalator_length / (escalator_speed + person_speed)) = 20 := by
  sorry

end time_to_cover_escalator_l36_36878


namespace intersection_of_sets_l36_36314

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {0, 1, 2}

theorem intersection_of_sets :
  C = A ∩ B :=
sorry

end intersection_of_sets_l36_36314


namespace custom_op_two_neg_four_l36_36151

-- Define the binary operation *
def custom_op (x y : ℚ) : ℚ := (x * y) / (x + y)

-- Proposition stating 2 * (-4) = 4 using the custom operation
theorem custom_op_two_neg_four : custom_op 2 (-4) = 4 :=
by
  sorry

end custom_op_two_neg_four_l36_36151


namespace equal_probabilities_partitioned_nonpartitioned_conditions_for_equal_probabilities_l36_36791

variable (v1 v2 f1 f2 : ℝ)

theorem equal_probabilities_partitioned_nonpartitioned :
  (v1 * (v2 + f2) + v2 * (v1 + f1)) / (2 * (v1 + f1) * (v2 + f2)) =
  (v1 + v2) / ((v1 + f1) + (v2 + f2)) :=
by sorry

theorem conditions_for_equal_probabilities :
  (v1 * f2 = v2 * f1) ∨ (v1 + f1 = v2 + f2) :=
by sorry

end equal_probabilities_partitioned_nonpartitioned_conditions_for_equal_probabilities_l36_36791


namespace factorial_trailing_zeros_l36_36026

theorem factorial_trailing_zeros :
  ∃ (S : Finset ℕ), (∀ m ∈ S, 1 ≤ m ∧ m ≤ 30) ∧ (S.card = 24) ∧ (∀ m ∈ S, 
    ∃ n : ℕ, ∃ k : ℕ,  n ≥ k * 5 ∧ n ≤ (k + 1) * 5 - 1 ∧ 
      m = (n / 5) + (n / 25) + (n / 125) ∧ ((n / 5) % 5 = 0)) :=
sorry

end factorial_trailing_zeros_l36_36026


namespace tan_30_eq_sqrt3_div_3_l36_36733

theorem tan_30_eq_sqrt3_div_3 :
  let opposite := 1
  let adjacent := sqrt (3 : ℝ) 
  tan (real.pi / 6) = opposite / adjacent := by 
    sorry

end tan_30_eq_sqrt3_div_3_l36_36733


namespace binary_remainder_div_8_l36_36977

theorem binary_remainder_div_8 (n : ℕ) (h : n = 0b101100110011) : n % 8 = 3 :=
by sorry

end binary_remainder_div_8_l36_36977


namespace remainder_division_1614_254_eq_90_l36_36083

theorem remainder_division_1614_254_eq_90 :
  ∀ (x : ℕ) (R : ℕ),
    1614 - x = 1360 →
    x * 6 + R = 1614 →
    0 ≤ R →
    R < x →
    R = 90 := 
by
  intros x R h_diff h_div h_nonneg h_lt
  sorry

end remainder_division_1614_254_eq_90_l36_36083


namespace find_Pete_original_number_l36_36194

noncomputable def PeteOriginalNumber (x : ℝ) : Prop :=
  5 * (3 * x + 15) = 200

theorem find_Pete_original_number : ∃ x : ℝ, PeteOriginalNumber x ∧ x = 25 / 3 :=
by
  sorry

end find_Pete_original_number_l36_36194


namespace turtles_order_l36_36583

-- Define variables for each turtle as real numbers representing their positions
variables (O P S E R : ℝ)

-- Define the conditions given in the problem
def condition1 := S = O - 10
def condition2 := S = R + 25
def condition3 := R = E - 5
def condition4 := E = P - 25

-- Define the order of arrival
def order_of_arrival (O P S E R : ℝ) := 
     O = 0 ∧ 
     P = -5 ∧
     S = -10 ∧
     E = -30 ∧
     R = -35

-- Theorem to show the given conditions imply the order of arrival
theorem turtles_order (h1 : condition1 S O)
                     (h2 : condition2 S R)
                     (h3 : condition3 R E)
                     (h4 : condition4 E P) :
  order_of_arrival O P S E R :=
by sorry

end turtles_order_l36_36583


namespace necessarily_negative_b_ab_l36_36845

theorem necessarily_negative_b_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : -2 < b) (h4 : b < 0) : 
  b + a * b < 0 := by 
  sorry

end necessarily_negative_b_ab_l36_36845


namespace breakfast_cost_l36_36586

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3

def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2

def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

theorem breakfast_cost :
  muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  + muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cup = 17 :=
by
  -- skipping proof
  sorry

end breakfast_cost_l36_36586


namespace quadratic_has_distinct_real_roots_l36_36374

theorem quadratic_has_distinct_real_roots :
  ∀ (x : ℝ), x^2 - 2 * x - 1 = 0 → (∃ Δ > 0, Δ = ((-2)^2 - 4 * 1 * (-1))) := by
  sorry

end quadratic_has_distinct_real_roots_l36_36374


namespace minimum_revenue_maximum_marginal_cost_minimum_profit_l36_36798

noncomputable def R (x : ℕ) : ℝ := x^2 + 16 / x^2 + 40
noncomputable def C (x : ℕ) : ℝ := 10 * x + 40 / x
noncomputable def MC (x : ℕ) : ℝ := C (x + 1) - C x
noncomputable def z (x : ℕ) : ℝ := R x - C x

theorem minimum_revenue :
  ∀ x : ℕ, 1 ≤ x → x ≤ 10 → R x ≥ 72 :=
sorry

theorem maximum_marginal_cost :
  ∀ x : ℕ, 1 ≤ x → x ≤ 9 → MC x ≤ 86 / 9 :=
sorry

theorem minimum_profit :
  ∀ x : ℕ, 1 ≤ x → x ≤ 10 → (x = 1 ∨ x = 4) → z x ≥ 7 :=
sorry

end minimum_revenue_maximum_marginal_cost_minimum_profit_l36_36798


namespace calculate_expression_l36_36122

theorem calculate_expression : (Real.sqrt 8 + Real.sqrt (1 / 2)) * Real.sqrt 32 = 20 := by
  sorry

end calculate_expression_l36_36122


namespace maria_total_payment_l36_36391

theorem maria_total_payment
  (original_price discount : ℝ)
  (discount_percentage : ℝ := 0.25)
  (discount_amount : ℝ := 40) 
  (total_paid : ℝ := 120) :
  discount_amount = discount_percentage * original_price →
  total_paid = original_price - discount_amount → 
  total_paid = 120 :=
by
  intros h1 h2
  rw [h1] at h2
  exact h2

end maria_total_payment_l36_36391


namespace y_relationship_l36_36792

theorem y_relationship :
  ∀ (y1 y2 y3 : ℝ), 
  (y1 = (-2)^2 - 4*(-2) - 3) ∧ 
  (y2 = 1^2 - 4*1 - 3) ∧ 
  (y3 = 4^2 - 4*4 - 3) → 
  y1 > y3 ∧ y3 > y2 := 
by sorry

end y_relationship_l36_36792


namespace least_subtraction_for_divisibility_l36_36860

def original_number : ℕ := 5474827

def required_subtraction : ℕ := 7

theorem least_subtraction_for_divisibility :
  ∃ k : ℕ, (original_number - required_subtraction) = 12 * k :=
sorry

end least_subtraction_for_divisibility_l36_36860


namespace value_of_x_l36_36156

theorem value_of_x (x : ℝ) (m : ℕ) (h1 : m = 31) :
  ((x ^ m) / (5 ^ m)) * ((x ^ 16) / (4 ^ 16)) = 1 / (2 * 10 ^ 31) → x = 1 := by
  sorry

end value_of_x_l36_36156


namespace min_value_of_sum_l36_36014

theorem min_value_of_sum (x y : ℝ) (h1 : x + 4 * y = 2 * x * y) (h2 : 0 < x) (h3 : 0 < y) : 
  x + y ≥ 9 / 2 :=
sorry

end min_value_of_sum_l36_36014


namespace inequality_not_always_true_l36_36011

theorem inequality_not_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) : ¬ (∀ c, (a - b) / c > 0) := 
sorry

end inequality_not_always_true_l36_36011


namespace presidency_meeting_ways_l36_36105

theorem presidency_meeting_ways : 
  ∃ (ways : ℕ), ways = 4 * 6 * 3 * 225 := sorry

end presidency_meeting_ways_l36_36105


namespace sarah_house_units_digit_l36_36267

-- Sarah's house number has two digits
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- The four statements about Sarah's house number
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0
def has_digit_7 (n : ℕ) : Prop := n / 10 = 7 ∨ n % 10 = 7

-- Exactly three out of the four statements are true
def exactly_three_true (n : ℕ) : Prop :=
  (is_multiple_of_5 n ∧ is_odd n ∧ is_divisible_by_3 n ∧ ¬has_digit_7 n) ∨
  (is_multiple_of_5 n ∧ is_odd n ∧ ¬is_divisible_by_3 n ∧ has_digit_7 n) ∨
  (is_multiple_of_5 n ∧ ¬is_odd n ∧ is_divisible_by_3 n ∧ has_digit_7 n) ∨
  (¬is_multiple_of_5 n ∧ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_7 n)

-- Main statement
theorem sarah_house_units_digit : ∃ n : ℕ, is_two_digit n ∧ exactly_three_true n ∧ n % 10 = 5 :=
by
  sorry

end sarah_house_units_digit_l36_36267


namespace minimum_value_inequality_l36_36149

theorem minimum_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → (2 / x + 1 / y) ≥ 9)) :=
by
  -- skipping the proof
  sorry

end minimum_value_inequality_l36_36149


namespace coeff_x2_in_expansion_l36_36800

theorem coeff_x2_in_expansion : 
  binomial_expansion.coeff (x - (1 / (4 * x))) 6 x 2 = 15 / 16 :=
by
  sorry

end coeff_x2_in_expansion_l36_36800


namespace find_triplets_l36_36760

theorem find_triplets (x y z : ℕ) :
  (x^2 + y^2 = 3 * 2016^z + 77) →
  (x, y, z) = (77, 14, 1) ∨ (x, y, z) = (14, 77, 1) ∨ 
  (x, y, z) = (70, 35, 1) ∨ (x, y, z) = (35, 70, 1) ∨ 
  (x, y, z) = (8, 4, 0) ∨ (x, y, z) = (4, 8, 0) :=
by
  sorry

end find_triplets_l36_36760


namespace zinc_percentage_in_1_gram_antacid_l36_36802

theorem zinc_percentage_in_1_gram_antacid :
  ∀ (z1 z2 : ℕ → ℤ) (total_zinc : ℤ),
    z1 0 = 2 ∧ z2 0 = 2 ∧ z1 1 = 1 ∧ total_zinc = 650 ∧
    (z1 0) * 2 * 5 / 100 + (z2 1) * 3 = total_zinc / 100 →
    (z2 1) * 100 = 15 :=
by
  sorry

end zinc_percentage_in_1_gram_antacid_l36_36802


namespace pencils_to_sell_for_profit_l36_36112

theorem pencils_to_sell_for_profit 
    (total_pencils : ℕ) 
    (buy_price sell_price : ℝ) 
    (desired_profit : ℝ) 
    (h_total_pencils : total_pencils = 2000) 
    (h_buy_price : buy_price = 0.15) 
    (h_sell_price : sell_price = 0.30) 
    (h_desired_profit : desired_profit = 150) :
    total_pencils * buy_price + desired_profit = total_pencils * sell_price → total_pencils = 1500 :=
by
    sorry

end pencils_to_sell_for_profit_l36_36112


namespace total_respondents_l36_36489

theorem total_respondents (X Y : ℕ) (hX : X = 360) (h_ratio : 9 * Y = X) : X + Y = 400 := by
  sorry

end total_respondents_l36_36489


namespace three_hour_classes_per_week_l36_36429

theorem three_hour_classes_per_week (x : ℕ) : 
  (24 * (3 * x + 4 + 4) = 336) → x = 2 := by {
  sorry
}

end three_hour_classes_per_week_l36_36429


namespace divisible_by_42_l36_36654

theorem divisible_by_42 (a : ℤ) : ∃ k : ℤ, a^7 - a = 42 * k := 
sorry

end divisible_by_42_l36_36654


namespace find_n_l36_36000

noncomputable def satisfies_condition (n d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ) : Prop :=
  1 = d₁ ∧ d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ d₄ < d₅ ∧ d₅ < d₆ ∧ d₆ < d₇ ∧ d₇ < n ∧
  (∀ d, d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d = d₅ ∨ d = d₆ ∨ d = d₇ ∨ d = n → n % d = 0) ∧
  (∀ d, n % d = 0 → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d = d₅ ∨ d = d₆ ∨ d = d₇ ∨ d = n)

theorem find_n (n : ℕ) : (∃ d₁ d₂ d₃ d₄ d₅ d₆ d₇, satisfies_condition n d₁ d₂ d₃ d₄ d₅ d₆ d₇ ∧ n = d₆^2 + d₇^2 - 1) → (n = 144 ∨ n = 1984) :=
  by
  sorry

end find_n_l36_36000


namespace inscribed_angle_half_central_angle_l36_36943

theorem inscribed_angle_half_central_angle
  {O : Point} {A B C : Point}
  (hO_AO : distance O A = distance O B)
  (hO_AC : distance O A = distance O C)
  (hO_eq : ∃ O, Circle O B C A)
  (hBAC_eq : ∠ B A C = angle.inscribed O B C) 
  : ∠ B A C = (1 / 2) * ∠ B O C := 
sorry

end inscribed_angle_half_central_angle_l36_36943


namespace train_length_correct_l36_36252

noncomputable def length_of_train (train_speed_kmh : ℝ) (cross_time_s : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * cross_time_s
  total_distance - bridge_length_m

theorem train_length_correct :
  length_of_train 45 30 205 = 170 :=
by
  sorry

end train_length_correct_l36_36252


namespace positive_integers_of_m_n_l36_36273

theorem positive_integers_of_m_n (m n : ℕ) (p : ℕ) (a : ℕ) (k : ℕ) (h_m_ge_2 : m ≥ 2) (h_n_ge_2 : n ≥ 2) 
  (h_prime_q : Prime (m + 1)) (h_4k_1 : m + 1 = 4 * k - 1) 
  (h_eq : (m ^ (2 ^ n - 1) - 1) / (m - 1) = m ^ n + p ^ a) : 
  (m, n) = (p - 1, 2) ∧ Prime p ∧ ∃k, p = 4 * k - 1 := 
by {
  sorry
}

end positive_integers_of_m_n_l36_36273


namespace max_min_values_of_f_l36_36609

-- Define the function f(x) and the conditions about its coefficients
def f (x : ℝ) (p q : ℝ) : ℝ := x^3 - p * x^2 - q * x

def intersects_x_axis_at_1 (p q : ℝ) : Prop :=
  f 1 p q = 0

-- Define the maximum and minimum values on the interval [-1, 1]
theorem max_min_values_of_f (p q : ℝ) 
  (h1 : f 1 p q = 0) :
  (p = 2) ∧ (q = -1) ∧ (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x 2 (-1) ≤ f (1/3) 2 (-1)) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f (-1) 2 (-1) ≤ f x 2 (-1)) :=
sorry

end max_min_values_of_f_l36_36609


namespace find_a_b_max_profit_allocation_l36_36718

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (a * Real.log x) / x + 5 / x - b

theorem find_a_b :
  (∃ (a b : ℝ), f 1 a b = 5 ∧ f 10 a b = 16.515) :=
sorry

noncomputable def g (x : ℝ) := 2 * Real.sqrt x / x

noncomputable def profit (x : ℝ) := x * (5 * Real.log x / x + 5 / x) + (50 - x) * (2 * Real.sqrt (50 - x) / (50 - x))

theorem max_profit_allocation :
  (∃ (x : ℝ), 10 ≤ x ∧ x ≤ 40 ∧ ∀ y, (10 ≤ y ∧ y ≤ 40) → profit x ≥ profit y)
  ∧ profit 25 = 31.09 :=
sorry

end find_a_b_max_profit_allocation_l36_36718


namespace order_of_operations_example_l36_36387

theorem order_of_operations_example :
  3^2 * 4 + 5 * (6 + 3) - 15 / 3 = 76 := by
  sorry

end order_of_operations_example_l36_36387


namespace perimeter_of_square_l36_36409

/-- The perimeter of a square with side length 15 cm is 60 cm -/
theorem perimeter_of_square (side_length : ℝ) (area : ℝ) (h1 : side_length = 15) (h2 : area = 225) :
  (4 * side_length = 60) :=
by
  -- Proof steps would go here (omitted)
  sorry

end perimeter_of_square_l36_36409


namespace gcd_lcm_product_75_90_l36_36883

theorem gcd_lcm_product_75_90 :
  let a := 75
  let b := 90
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  gcd_ab * lcm_ab = 6750 :=
by
  let a := 75
  let b := 90
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  sorry

end gcd_lcm_product_75_90_l36_36883


namespace quadratic_functions_count_correct_even_functions_count_correct_l36_36901

def num_coefficients := 4
def valid_coefficients := [-1, 0, 1, 2]

def count_quadratic_functions : ℕ :=
  num_coefficients * num_coefficients * (num_coefficients - 1)

def count_even_functions : ℕ :=
  (num_coefficients - 1) * (num_coefficients - 2)

def total_quad_functions_correct : Prop := count_quadratic_functions = 18
def total_even_functions_correct : Prop := count_even_functions = 6

theorem quadratic_functions_count_correct : total_quad_functions_correct :=
by sorry

theorem even_functions_count_correct : total_even_functions_correct :=
by sorry

end quadratic_functions_count_correct_even_functions_count_correct_l36_36901


namespace point_in_fourth_quadrant_l36_36029

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a^2 + 1 > 0) (h2 : -1 - b^2 < 0) : 
  (a^2 + 1 > 0 ∧ -1 - b^2 < 0) ∧ (0 < a^2 + 1) ∧ (-1 - b^2 < 0) :=
by
  sorry

end point_in_fourth_quadrant_l36_36029


namespace min_segments_to_erase_l36_36338

noncomputable def nodes (m n : ℕ) : ℕ := (m - 2) * (n - 2)

noncomputable def segments_to_erase (m n : ℕ) : ℕ := (nodes m n + 1) / 2

theorem min_segments_to_erase (m n : ℕ) (hm : m = 11) (hn : n = 11) :
  segments_to_erase m n = 41 := by
  sorry

end min_segments_to_erase_l36_36338


namespace evaluate_expression_l36_36128

theorem evaluate_expression : 
  ∃ q : ℤ, ∀ (a : ℤ), a = 2022 → (2023 : ℚ) / 2022 - (2022 : ℚ) / 2023 = 4045 / q :=
by
  sorry

end evaluate_expression_l36_36128


namespace tank_capacity_l36_36405

theorem tank_capacity :
  (∃ c: ℝ, (∃ w: ℝ, w / c = 1/6 ∧ (w + 5) / c = 1/3) → c = 30) :=
by
  sorry

end tank_capacity_l36_36405


namespace factor_polynomial_l36_36431

theorem factor_polynomial (a : ℝ) : 74 * a^2 + 222 * a + 148 * a^3 = 74 * a * (2 * a^2 + a + 3) :=
by
  sorry

end factor_polynomial_l36_36431


namespace harmonic_sum_divisibility_l36_36807

-- Definitions based on conditions
def harmonic_sum (n : ℕ) : ℚ := (∑ i in (finset.range (n+1)).filter(λ k, k > 0), 1 / i)

def rel_prime_pos (a b : ℕ) : Prop := nat.coprime a b ∧ a > 0 ∧ b > 0

-- The theorem based on equivalent proof statement
theorem harmonic_sum_divisibility (n : ℕ) (p_n q_n : ℕ) (h : harmonic_sum n = p_n / q_n) (hpq : rel_prime_pos p_n q_n) : 
  ¬ (5 ∣ q_n) ↔ n ∈ finset.range 5 ∪ (finset.range (25) \ finset.range (20)) ∪ (finset.range (105) \ finset.range (100)) ∪ (finset.range (125) \ finset.range (120)) :=
sorry

end harmonic_sum_divisibility_l36_36807


namespace inscribed_quadrilateral_exists_l36_36564

theorem inscribed_quadrilateral_exists (a b c d : ℝ) (h1: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  ∃ (p q : ℝ),
    p = Real.sqrt ((a * c + b * d) * (a * d + b * c) / (a * b + c * d)) ∧
    q = Real.sqrt ((a * b + c * d) * (a * d + b * c) / (a * c + b * d)) ∧
    a * c + b * d = p * q :=
by
  sorry

end inscribed_quadrilateral_exists_l36_36564


namespace sandwiches_prepared_l36_36954

variable (S : ℕ)
variable (H1 : S > 0)
variable (H2 : ∃ r : ℕ, r = S / 4)
variable (H3 : ∃ b : ℕ, b = (3 * S / 4) / 6)
variable (H4 : ∃ c : ℕ, c = 2 * b)
variable (H5 : ∃ x : ℕ, 5 * x = 5)
variable (H6 : 3 * S / 8 - 5 = 4)

theorem sandwiches_prepared : S = 24 :=
by
  sorry

end sandwiches_prepared_l36_36954


namespace josh_bought_6_CDs_l36_36474

theorem josh_bought_6_CDs 
  (numFilms : ℕ)   (numBooks : ℕ) (numCDs : ℕ)
  (costFilm : ℕ)   (costBook : ℕ) (costCD : ℕ)
  (totalSpent : ℕ) :
  numFilms = 9 → 
  numBooks = 4 → 
  costFilm = 5 → 
  costBook = 4 → 
  costCD = 3 → 
  totalSpent = 79 → 
  numCDs = (totalSpent - numFilms * costFilm - numBooks * costBook) / costCD → 
  numCDs = 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  exact h7

end josh_bought_6_CDs_l36_36474


namespace difference_between_percent_and_value_is_five_l36_36968

def hogs : ℕ := 75
def ratio : ℕ := 3

def num_of_cats (hogs : ℕ) (ratio : ℕ) : ℕ := hogs / ratio

def cats : ℕ := num_of_cats hogs ratio

def percent_of_cats (cats : ℕ) : ℝ := 0.60 * cats
def value_to_subtract : ℕ := 10

def difference (percent : ℝ) (value : ℕ) : ℝ := percent - value

theorem difference_between_percent_and_value_is_five
    (hogs : ℕ)
    (ratio : ℕ)
    (cats : ℕ := num_of_cats hogs ratio)
    (percent : ℝ := percent_of_cats cats)
    (value : ℕ := value_to_subtract)
    :
    difference percent value = 5 :=
by {
    sorry
}

end difference_between_percent_and_value_is_five_l36_36968


namespace lucien_balls_count_l36_36939

theorem lucien_balls_count (lucca_balls : ℕ) (lucca_percent_basketballs : ℝ) (lucien_percent_basketballs : ℝ) (total_basketballs : ℕ)
  (h1 : lucca_balls = 100)
  (h2 : lucca_percent_basketballs = 0.10)
  (h3 : lucien_percent_basketballs = 0.20)
  (h4 : total_basketballs = 50) :
  ∃ lucien_balls : ℕ, lucien_balls = 200 :=
by
  sorry

end lucien_balls_count_l36_36939


namespace count_triangles_in_figure_l36_36161

-- Define the structure of the grid with the given properties.
def grid_structure : Prop :=
  ∃ (n1 n2 n3 n4 : ℕ), 
  n1 = 3 ∧  -- First row: 3 small triangles
  n2 = 2 ∧  -- Second row: 2 small triangles
  n3 = 1 ∧  -- Third row: 1 small triangle
  n4 = 1    -- 1 large inverted triangle

-- The problem statement
theorem count_triangles_in_figure (h : grid_structure) : 
  ∃ (total_triangles : ℕ), total_triangles = 9 :=
sorry

end count_triangles_in_figure_l36_36161


namespace red_car_speed_l36_36424

/-- Dale owns 4 sports cars where:
1. The red car can travel at twice the speed of the green car.
2. The green car can travel at 8 times the speed of the blue car.
3. The blue car can travel at a speed of 80 miles per hour.
We need to determine the speed of the red car. --/
theorem red_car_speed (r g b: ℕ) (h1: r = 2 * g) (h2: g = 8 * b) (h3: b = 80) : 
  r = 1280 :=
by
  sorry

end red_car_speed_l36_36424


namespace smallest_n_l36_36694

theorem smallest_n (n : ℕ) : (∃ (m1 m2 : ℕ), 4 * n = m1^2 ∧ 5 * n = m2^3) ↔ n = 500 := 
begin
  sorry
end

end smallest_n_l36_36694


namespace chord_intersection_eq_l36_36599

theorem chord_intersection_eq (x y : ℝ) (r : ℝ) : 
  (x + 1)^2 + y^2 = r^2 → 
  (x - 4)^2 + (y - 1)^2 = 4 → 
  (x = 4) → 
  (y = 1) → 
  (r^2 = 26) → (5 * x + y - 19 = 0) :=
by
  sorry

end chord_intersection_eq_l36_36599


namespace geometric_seq_l36_36766

def seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ (∀ n : ℕ, S (n + 1) + a n = S n + 5 * 4 ^ n)

theorem geometric_seq (a S : ℕ → ℝ) (h : seq a S) :
  ∃ r : ℝ, ∃ a1 : ℝ, (∀ n : ℕ, (a (n + 1) - 4 ^ (n + 1)) = r * (a n - 4 ^ n)) :=
by
  sorry

end geometric_seq_l36_36766


namespace range_of_a_l36_36206

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ)
  (h_cond : ∀ (n : ℕ), n > 0 → (a_seq n = if n ≤ 4 then 2^n - 1 else -n^2 + (a - 1) * n))
  (h_max_a5 : ∀ (n : ℕ), n > 0 → a_seq n ≤ a_seq 5) :
  9 ≤ a ∧ a ≤ 12 := 
by
  sorry

end range_of_a_l36_36206


namespace rectangle_diagonal_length_l36_36829

theorem rectangle_diagonal_length {k : ℝ} (h1 : 2 * (3 * k + 2 * k) = 72)
  (h2 : k = 7.2) : 
  let length := 3 * k in
  let width := 2 * k in
  let diagonal := real.sqrt ((length ^ 2) + (width ^ 2)) in
  diagonal = 25.96 :=
by
  sorry

end rectangle_diagonal_length_l36_36829


namespace sequence_general_term_l36_36449

theorem sequence_general_term (a : ℕ+ → ℤ) (h₁ : a 1 = 2) (h₂ : ∀ n : ℕ+, a (n + 1) = a n - 1) :
  ∀ n : ℕ+, a n = 3 - n := 
sorry

end sequence_general_term_l36_36449


namespace sequence_S15_is_211_l36_36043

theorem sequence_S15_is_211 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2)
  (h3 : ∀ n > 1, S (n + 1) + S (n - 1) = 2 * (S n + S 1)) :
  S 15 = 211 := 
sorry

end sequence_S15_is_211_l36_36043


namespace value_of_x_l36_36166

theorem value_of_x (x : ℝ) : (1 / 8) * (2 : ℝ) ^ 32 = (4 : ℝ) ^ x → x = 29 / 2 :=
by
  sorry

end value_of_x_l36_36166


namespace min_value_y_of_parabola_l36_36744

theorem min_value_y_of_parabola :
  ∃ y : ℝ, ∃ x : ℝ, (∀ y' x', (y' + x') = (y' - x')^2 + 3 * (y' - x') + 3 → y' ≥ y) ∧
            y = -1/2 :=
by
  sorry

end min_value_y_of_parabola_l36_36744


namespace interior_box_surface_area_l36_36659

-- Given conditions
def original_length : ℕ := 40
def original_width : ℕ := 60
def corner_side : ℕ := 8

-- Calculate the initial area
def area_original : ℕ := original_length * original_width

-- Calculate the area of one corner
def area_corner : ℕ := corner_side * corner_side

-- Calculate the total area removed by four corners
def total_area_removed : ℕ := 4 * area_corner

-- Theorem to state the final area remaining
theorem interior_box_surface_area : 
  area_original - total_area_removed = 2144 :=
by
  -- Place the proof here
  sorry

end interior_box_surface_area_l36_36659


namespace probability_red_side_given_observed_l36_36868

def total_cards : ℕ := 9
def black_black_cards : ℕ := 4
def black_red_cards : ℕ := 2
def red_red_cards : ℕ := 3

def red_sides : ℕ := red_red_cards * 2 + black_red_cards
def red_red_sides : ℕ := red_red_cards * 2
def probability_other_side_is_red (total_red_sides red_red_sides : ℕ) : ℚ :=
  red_red_sides / total_red_sides

theorem probability_red_side_given_observed :
  probability_other_side_is_red red_sides red_red_sides = 3 / 4 :=
by
  unfold red_sides
  unfold red_red_sides
  unfold probability_other_side_is_red
  sorry

end probability_red_side_given_observed_l36_36868


namespace simplify_and_evaluate_expression_l36_36498

theorem simplify_and_evaluate_expression :
  ∀ (x y : ℝ), 
  x = -1 / 3 → y = -2 → 
  (3 * x + 2 * y) * (3 * x - 2 * y) - 5 * x * (x - y) - (2 * x - y)^2 = -14 :=
by
  intros x y hx hy
  sorry

end simplify_and_evaluate_expression_l36_36498


namespace set_intersection_l36_36323

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2.5}

theorem set_intersection : A ∩ B = {0, 1, 2} :=
by
  sorry

end set_intersection_l36_36323


namespace fraction_of_n_is_80_l36_36619

-- Definitions from conditions
def n := (5 / 6) * 240

-- The theorem we want to prove
theorem fraction_of_n_is_80 : (2 / 5) * n = 80 :=
by
  -- This is just a placeholder to complete the statement, 
  -- actual proof logic is not included based on the prompt instructions
  sorry

end fraction_of_n_is_80_l36_36619


namespace slope_of_parallel_line_l36_36693

theorem slope_of_parallel_line (m : ℚ) (b : ℚ) :
  (∀ x y : ℚ, 5 * x - 3 * y = 21 → y = (5 / 3) * x + b) →
  m = 5 / 3 :=
by
  intros hyp
  sorry

end slope_of_parallel_line_l36_36693


namespace best_choice_for_square_formula_l36_36959

theorem best_choice_for_square_formula : 
  (89.8^2 = (90 - 0.2)^2) :=
by sorry

end best_choice_for_square_formula_l36_36959


namespace find_range_of_a_l36_36054

noncomputable def value_range_for_a : Set ℝ := {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4}

theorem find_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0)  ∧
  (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0) ∧
  (¬ (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0) → ¬ (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0))
  → a ∈ value_range_for_a :=
sorry

end find_range_of_a_l36_36054


namespace max_area_l36_36933

noncomputable def PA : ℝ := 3
noncomputable def PB : ℝ := 4
noncomputable def PC : ℝ := 5
noncomputable def BC : ℝ := 6

theorem max_area (PA PB PC BC : ℝ) (hPA : PA = 3) (hPB : PB = 4) (hPC : PC = 5) (hBC : BC = 6) : 
  ∃ (A B C : Type) (area_ABC : ℝ), area_ABC = 19 := 
by 
  sorry

end max_area_l36_36933


namespace allison_upload_ratio_l36_36254

theorem allison_upload_ratio :
  ∃ (x y : ℕ), (x + y = 30) ∧ (10 * x + 20 * y = 450) ∧ (x / 30 = 1 / 2) :=
by
  sorry

end allison_upload_ratio_l36_36254


namespace geography_book_price_l36_36999

open Real

-- Define the problem parameters
def num_english_books : ℕ := 35
def num_geography_books : ℕ := 35
def cost_english : ℝ := 7.50
def total_cost : ℝ := 630.00

-- Define the unknown we need to prove
def cost_geography : ℝ := 10.50

theorem geography_book_price :
  num_english_books * cost_english + num_geography_books * cost_geography = total_cost :=
by
  -- No need to include the proof steps
  sorry

end geography_book_price_l36_36999


namespace effective_rate_proof_l36_36862

noncomputable def nominal_rate : ℝ := 0.08
noncomputable def compounding_periods : ℕ := 2
noncomputable def effective_annual_rate (i : ℝ) (n : ℕ) : ℝ := (1 + i / n) ^ n - 1

theorem effective_rate_proof :
  effective_annual_rate nominal_rate compounding_periods = 0.0816 :=
by
  sorry

end effective_rate_proof_l36_36862


namespace probability_red_side_l36_36869

theorem probability_red_side (total_cards : ℕ)
  (cards_black_black : ℕ) (cards_black_red : ℕ) (cards_red_red : ℕ)
  (h_total : total_cards = 9)
  (h_black_black : cards_black_black = 4)
  (h_black_red : cards_black_red = 2)
  (h_red_red : cards_red_red = 3) :
  let total_sides := (cards_black_black * 2) + (cards_black_red * 2) + (cards_red_red * 2)
  let red_sides := (cards_black_red * 1) + (cards_red_red * 2)
  (red_sides > 0) →
  ((cards_red_red * 2) / red_sides : ℚ) = 3 / 4 := 
by
  intros
  sorry

end probability_red_side_l36_36869


namespace problem_sum_of_k_l36_36186

theorem problem_sum_of_k {a b c k : ℂ} (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_ratio : a / (1 - b) = k ∧ b / (1 - c) = k ∧ c / (1 - a) = k) :
  (if (k^2 - k + 1 = 0) then -(-1)/1 else 0) = 1 :=
sorry

end problem_sum_of_k_l36_36186


namespace valid_lineup_count_l36_36193

noncomputable def num_valid_lineups : ℕ :=
  let total_lineups := Nat.choose 18 8
  let unwanted_lineups := Nat.choose 14 4
  total_lineups - unwanted_lineups

theorem valid_lineup_count : num_valid_lineups = 42757 := by
  sorry

end valid_lineup_count_l36_36193


namespace calculate_expression_l36_36559

theorem calculate_expression : 5 * 12 + 6 * 11 - 2 * 15 + 7 * 9 = 159 := by
  sorry

end calculate_expression_l36_36559


namespace eval_64_pow_5_over_6_l36_36753

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l36_36753


namespace find_m_l36_36445

noncomputable def m_solution (m : ℝ) : ℂ := (m - 3 * Complex.I) / (2 + Complex.I)

theorem find_m :
  ∀ (m : ℝ), Complex.im (m_solution m) ≠ 0 → Complex.re (m_solution m) = 0 → m = 3 / 2 :=
by
  intro m h_im h_re
  sorry

end find_m_l36_36445


namespace part1_part2_l36_36918

-- Define the conditions and claims
theorem part1 (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |x - a|)
  (h_sol : ∀ x, f (2 * x) ≤ 4 ↔ 0 ≤ x ∧ x ≤ 4) : a = 4 :=
by
  sorry

theorem part2 (a : ℝ) (m : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |x - a|)
  (h_empty : ∀ x, ¬ (f x + f (x + m) < 2)) : m ≥ 2 ∨ m ≤ -2 :=
by
  sorry

end part1_part2_l36_36918


namespace union_A_B_l36_36189

def A : Set ℝ := {x | ∃ y : ℝ, y = Real.log x}
def B : Set ℝ := {x | x < 1}

theorem union_A_B : (A ∪ B) = Set.univ :=
by
  sorry

end union_A_B_l36_36189


namespace Archie_started_with_100_marbles_l36_36881

theorem Archie_started_with_100_marbles
  (M : ℕ) 
  (h1 : 0.60 * M + (0.50 * 0.40 * M) + 20 = M) 
  (h2 : 0.20 * M = 20) : 
  M = 100 :=
by
  sorry

end Archie_started_with_100_marbles_l36_36881


namespace gcd_max_value_l36_36673

theorem gcd_max_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) : 
  ∃ d, d = Nat.gcd a b ∧ d = 504 :=
by
  sorry

end gcd_max_value_l36_36673


namespace score_difference_l36_36805

-- Definitions of the given conditions
def Layla_points : ℕ := 70
def Total_points : ℕ := 112

-- The statement to be proven
theorem score_difference : (Layla_points - (Total_points - Layla_points)) = 28 :=
by sorry

end score_difference_l36_36805


namespace reciprocal_of_2022_l36_36667

noncomputable def reciprocal (x : ℝ) := 1 / x

theorem reciprocal_of_2022 : reciprocal 2022 = 1 / 2022 :=
by
  -- Define reciprocal
  sorry

end reciprocal_of_2022_l36_36667


namespace TinaTotalPens_l36_36971

variable (p g b : ℕ)
axiom H1 : p = 12
axiom H2 : g = p - 9
axiom H3 : b = g + 3

theorem TinaTotalPens : p + g + b = 21 := by
  sorry

end TinaTotalPens_l36_36971


namespace college_student_ticket_cost_l36_36305

theorem college_student_ticket_cost 
    (total_visitors : ℕ)
    (nyc_residents: ℕ)
    (college_students_nyc: ℕ)
    (total_money_received : ℕ) :
    total_visitors = 200 →
    nyc_residents = total_visitors / 2 →
    college_students_nyc = (nyc_residents * 30) / 100 →
    total_money_received = 120 →
    (total_money_received / college_students_nyc) = 4 := 
sorry

end college_student_ticket_cost_l36_36305


namespace octopus_legs_l36_36380

/-- Four octopuses made statements about their total number of legs.
    - Octopuses with 7 legs always lie.
    - Octopuses with 6 or 8 legs always tell the truth.
    - Blue: "Together we have 28 legs."
    - Green: "Together we have 27 legs."
    - Yellow: "Together we have 26 legs."
    - Red: "Together we have 25 legs."
   Prove that the Green octopus has 6 legs, and the Blue, Yellow, and Red octopuses each have 7 legs.
-/
theorem octopus_legs (L_B L_G L_Y L_R : ℕ) (H1 : (L_B + L_G + L_Y + L_R = 28 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 27 → L_B + L_G + L_Y + L_R = 27) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 26 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 25 → L_B ≠ 7)) : 
  (L_G = 6) ∧ (L_B = 7) ∧ (L_Y = 7) ∧ (L_R = 7) :=
sorry

end octopus_legs_l36_36380


namespace multiples_of_10_5_l36_36536

theorem multiples_of_10_5 (n : ℤ) (h1 : ∀ k : ℤ, k % 10 = 0 → k % 5 = 0) (h2 : n % 10 = 0) : n % 5 = 0 := 
by
  sorry

end multiples_of_10_5_l36_36536


namespace rain_at_least_once_l36_36834

noncomputable def rain_probability (day_prob : ℚ) (days : ℕ) : ℚ :=
  1 - (1 - day_prob)^days

theorem rain_at_least_once :
  ∀ (day_prob : ℚ) (days : ℕ),
    day_prob = 3/4 → days = 4 →
    rain_probability day_prob days = 255/256 :=
by
  intros day_prob days h1 h2
  sorry

end rain_at_least_once_l36_36834


namespace inverse_f_neg_3_l36_36925

def f (x : ℝ) : ℝ := 5 - 2 * x

theorem inverse_f_neg_3 : (∃ x : ℝ, f x = -3) ∧ (f 4 = -3) :=
by
  sorry

end inverse_f_neg_3_l36_36925


namespace other_ticket_price_l36_36362

theorem other_ticket_price (total_tickets : ℕ) (total_sales : ℝ) (cheap_tickets : ℕ) (cheap_price : ℝ) (expensive_tickets : ℕ) (expensive_price : ℝ) :
  total_tickets = 380 →
  total_sales = 1972.50 →
  cheap_tickets = 205 →
  cheap_price = 4.50 →
  expensive_tickets = 380 - 205 →
  205 * 4.50 + expensive_tickets * expensive_price = 1972.50 →
  expensive_price = 6.00 :=
by
  intros
  -- proof will be filled here
  sorry

end other_ticket_price_l36_36362


namespace sufficient_but_not_necessary_condition_l36_36441

theorem sufficient_but_not_necessary_condition (f : ℝ → ℝ) (h : ∀ x, f x = x⁻¹) :
  ∀ x, (x > 1 → f (x + 2) > f (2*x + 1)) ∧ (¬ (x > 1) → ¬ (f (x + 2) > f (2*x + 1))) :=
by
  sorry

end sufficient_but_not_necessary_condition_l36_36441


namespace value_of_q_l36_36050

theorem value_of_q (m p q a b : ℝ) 
  (h₁ : a * b = 6) 
  (h₂ : (a + 1 / b) * (b + 1 / a) = q): 
  q = 49 / 6 := 
sorry

end value_of_q_l36_36050


namespace triangle_even_number_in_each_row_from_third_l36_36181

/-- Each number in the (n+1)-th row of the triangle is the sum of three numbers 
  from the n-th row directly above this number and its immediate left and right neighbors.
  If such neighbors do not exist, they are considered as zeros.
  Prove that in each row of the triangle, starting from the third row,
  there is at least one even number. -/

theorem triangle_even_number_in_each_row_from_third (triangle : ℕ → ℕ → ℕ) :
  (∀ n i : ℕ, i > n → triangle n i = 0) →
  (∀ n i : ℕ, triangle (n+1) i = triangle n (i-1) + triangle n i + triangle n (i+1)) →
  ∀ n : ℕ, n ≥ 2 → ∃ i : ℕ, i ≤ n ∧ 2 ∣ triangle n i :=
by
  intros
  sorry

end triangle_even_number_in_each_row_from_third_l36_36181


namespace find_inverse_of_f_at_4_l36_36187

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2

-- Statement of the problem
theorem find_inverse_of_f_at_4 : ∃ t : ℝ, f t = 4 ∧ t ≤ 1 ∧ t = -1 := by
  sorry

end find_inverse_of_f_at_4_l36_36187


namespace ways_to_insert_plus_l36_36292

-- Definition of the problem conditions
def num_ones : ℕ := 15
def target_sum : ℕ := 0 

-- Binomial coefficient calculation
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to be proven
theorem ways_to_insert_plus :
  binomial 14 9 = 2002 :=
by
  sorry

end ways_to_insert_plus_l36_36292


namespace cost_effectiveness_l36_36875

-- Define the variables and conditions
def num_employees : ℕ := 30
def ticket_price : ℝ := 80
def group_discount_rate : ℝ := 0.8
def women_discount_rate : ℝ := 0.5

-- Define the costs for each scenario
def cost_with_group_discount : ℝ := num_employees * ticket_price * group_discount_rate

def cost_with_women_discount (x : ℕ) : ℝ :=
  ticket_price * women_discount_rate * x + ticket_price * (num_employees - x)

-- Formalize the equivalence of cost and comparison logic
theorem cost_effectiveness (x : ℕ) (h : 0 ≤ x ∧ x ≤ num_employees) :
  if x < 12 then cost_with_women_discount x > cost_with_group_discount
  else if x = 12 then cost_with_women_discount x = cost_with_group_discount
  else cost_with_women_discount x < cost_with_group_discount :=
by sorry

end cost_effectiveness_l36_36875


namespace find_pos_int_l36_36759

theorem find_pos_int (n p : ℕ) (h_prime : Nat.Prime p) (h_pos_n : 0 < n) (h_pos_p : 0 < p) : 
  n^8 - p^5 = n^2 + p^2 → (n = 2 ∧ p = 3) :=
by
  sorry

end find_pos_int_l36_36759


namespace expected_value_of_die_is_475_l36_36731

-- Define the given probabilities
def prob_1 : ℚ := 1 / 12
def prob_2 : ℚ := 1 / 12
def prob_3 : ℚ := 1 / 6
def prob_4 : ℚ := 1 / 12
def prob_5 : ℚ := 1 / 12
def prob_6 : ℚ := 7 / 12

-- Define the expected value calculation
def expected_value := 
  prob_1 * 1 + prob_2 * 2 + prob_3 * 3 +
  prob_4 * 4 + prob_5 * 5 + prob_6 * 6

-- The problem statement to prove
theorem expected_value_of_die_is_475 : expected_value = 4.75 := by
  sorry

end expected_value_of_die_is_475_l36_36731


namespace find_x_l36_36239

variable (x : ℝ)
variable (y : ℝ := x * 3.5)
variable (z : ℝ := y / 0.00002)

theorem find_x (h : z = 840) : x = 0.0048 :=
sorry

end find_x_l36_36239


namespace find_subtracted_number_l36_36839

theorem find_subtracted_number (x y : ℕ) (h1 : 6 * x - 5 * x = 5) (h2 : (30 - y) * 4 = (25 - y) * 5) : y = 5 :=
sorry

end find_subtracted_number_l36_36839


namespace sum_of_number_and_its_square_is_20_l36_36927

theorem sum_of_number_and_its_square_is_20 (n : ℕ) (h : n = 4) : n + n^2 = 20 :=
by
  sorry

end sum_of_number_and_its_square_is_20_l36_36927


namespace line_intersects_circle_l36_36244

theorem line_intersects_circle (m : ℝ) : 
  ∃ (x y : ℝ), y = m * x - 3 ∧ x^2 + (y - 1)^2 = 25 :=
sorry

end line_intersects_circle_l36_36244


namespace faye_total_crayons_l36_36134

-- Define the number of rows and the number of crayons per row as given conditions.
def num_rows : ℕ := 7
def crayons_per_row : ℕ := 30

-- State the theorem we need to prove.
theorem faye_total_crayons : (num_rows * crayons_per_row) = 210 :=
by
  sorry

end faye_total_crayons_l36_36134


namespace xiao_ming_selects_cooking_probability_l36_36103

theorem xiao_ming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let probability (event: String) := if event ∈ courses then 1 / (courses.length : ℝ) else 0
  probability "cooking" = 1 / 4 :=
by
  sorry

end xiao_ming_selects_cooking_probability_l36_36103


namespace tan_30_eq_sqrt3_div3_l36_36742

theorem tan_30_eq_sqrt3_div3 (sin_30_cos_30 : ℝ → ℝ → Prop)
  (h1 : sin_30_cos_30 (1 / 2) (Real.sqrt 3 / 2)) :
  ∃ t, t = Real.tan (Real.pi / 6) ∧ t = Real.sqrt 3 / 3 :=
by
  existsi Real.tan (Real.pi / 6)
  sorry

end tan_30_eq_sqrt3_div3_l36_36742


namespace probability_interval_l36_36440

noncomputable def Phi : ℝ → ℝ := sorry -- assuming Φ is a given function for CDF of a standard normal distribution

theorem probability_interval (h : Phi 1.98 = 0.9762) : 
  2 * Phi 1.98 - 1 = 0.9524 :=
by
  sorry

end probability_interval_l36_36440


namespace product_of_consecutive_integers_is_square_l36_36945

theorem product_of_consecutive_integers_is_square (x : ℤ) : 
  x * (x + 1) * (x + 2) * (x + 3) + 1 = (x^2 + 3 * x + 1) ^ 2 :=
by
  sorry

end product_of_consecutive_integers_is_square_l36_36945


namespace rationalize_denominator_l36_36948

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  (4 * Real.sqrt 7 + 3 * Real.sqrt 13) ≠ 0 →
  B < D →
  ∀ (x : ℝ), x = (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) →
    A + B + C + D + E = 22 := 
by
  intros
  -- Provide the actual theorem statement here
  sorry

end rationalize_denominator_l36_36948


namespace total_donations_l36_36361

-- Define the conditions
def started_donating_age : ℕ := 17
def current_age : ℕ := 71
def annual_donation : ℕ := 8000

-- Define the proof problem to show the total donation amount equals $432,000
theorem total_donations : (current_age - started_donating_age) * annual_donation = 432000 := 
by
  sorry

end total_donations_l36_36361


namespace relationship_among_m_n_k_l36_36031

theorem relationship_among_m_n_k :
  (¬ ∃ x : ℝ, |2 * x - 3| + m = 0) → 
  (∃! x: ℝ, |3 * x - 4| + n = 0) → 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |4 * x₁ - 5| + k = 0 ∧ |4 * x₂ - 5| + k = 0) →
  (m > n ∧ n > k) :=
by
  intros h1 h2 h3
  -- Proof part will be added here
  sorry

end relationship_among_m_n_k_l36_36031


namespace initial_bananas_per_child_l36_36651

theorem initial_bananas_per_child : 
  ∀ (B n m x : ℕ), 
  n = 740 → 
  m = 370 → 
  (B = n * x) → 
  (B = (n - m) * (x + 2)) → 
  x = 2 := 
by
  intros B n m x h1 h2 h3 h4
  sorry

end initial_bananas_per_child_l36_36651


namespace describe_set_T_l36_36560

-- Define the conditions for the set of points T
def satisfies_conditions (x y : ℝ) : Prop :=
  (x + 3 = 4 ∧ y < 7) ∨ (y - 3 = 4 ∧ x < 1)

-- Define the set T based on the conditions
def set_T := {p : ℝ × ℝ | satisfies_conditions p.1 p.2}

-- Statement to prove the geometric description of the set T
theorem describe_set_T :
  (∃ x y, satisfies_conditions x y) → ∃ p1 p2,
  (p1 = (1, t) ∧ t < 7 → satisfies_conditions 1 t) ∧
  (p2 = (t, 7) ∧ t < 1 → satisfies_conditions t 7) ∧
  (p1 ≠ p2) :=
sorry

end describe_set_T_l36_36560


namespace total_weight_of_containers_l36_36487

theorem total_weight_of_containers (x y z : ℕ) :
  x + y = 162 →
  y + z = 168 →
  z + x = 174 →
  x + y + z = 252 :=
by
  intros hxy hyz hzx
  -- proof skipped
  sorry

end total_weight_of_containers_l36_36487


namespace find_m_of_ellipse_conditions_l36_36447

-- definition for isEllipseGivenFocus condition
def isEllipseGivenFocus (m : ℝ) : Prop :=
  ∃ (a : ℝ), a = 5 ∧ (-4)^2 = a^2 - m^2 ∧ 0 < m

-- statement to prove the described condition implies m = 3
theorem find_m_of_ellipse_conditions (m : ℝ) (h : isEllipseGivenFocus m) : m = 3 :=
sorry

end find_m_of_ellipse_conditions_l36_36447


namespace geometric_sequence_sum_l36_36049

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geo : ∀ n, a (n + 1) = q * a n)
  (h1 : a 1 + a 2 + a 3 = 7)
  (h2 : a 2 + a 3 + a 4 = 14) :
  a 4 + a 5 + a 6 = 56 :=
sorry

end geometric_sequence_sum_l36_36049


namespace ages_correct_l36_36040

variables (Rehana_age Phoebe_age Jacob_age Xander_age : ℕ)

theorem ages_correct
  (h1 : Rehana_age = 25)
  (h2 : Rehana_age + 5 = 3 * (Phoebe_age + 5))
  (h3 : Jacob_age = 3 * Phoebe_age / 5)
  (h4 : Xander_age = Rehana_age + Jacob_age - 4) : 
  Rehana_age = 25 ∧ Phoebe_age = 5 ∧ Jacob_age = 3 ∧ Xander_age = 24 :=
by
  sorry

end ages_correct_l36_36040


namespace largest_gcd_l36_36681

theorem largest_gcd (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 1008) : 
  ∃ d : ℕ, d = Int.gcd a b ∧ d = 504 :=
by
  sorry

end largest_gcd_l36_36681


namespace exponential_monotone_l36_36283

theorem exponential_monotone {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a < b :=
sorry

end exponential_monotone_l36_36283


namespace daria_weeks_needed_l36_36569

-- Defining the parameters and conditions
def initial_amount : ℕ := 20
def weekly_savings : ℕ := 10
def cost_of_vacuum_cleaner : ℕ := 120

-- Defining the total money Daria needs to add to her initial amount
def additional_amount_needed : ℕ := cost_of_vacuum_cleaner - initial_amount

-- Defining the number of weeks needed to save the additional amount, given weekly savings
def weeks_needed : ℕ := additional_amount_needed / weekly_savings

-- The theorem stating that Daria needs exactly 10 weeks to cover the expense of the vacuum cleaner
theorem daria_weeks_needed : weeks_needed = 10 := by
  sorry

end daria_weeks_needed_l36_36569


namespace quadrilateral_BD_length_l36_36178

theorem quadrilateral_BD_length :
  ∃ (BD : ℕ), 
    (ABCD.exists
      ∧ AB = 5
      ∧ BC = 17
      ∧ CD = 5
      ∧ DA = 9
      ∧ BD = 13) :=
sorry

end quadrilateral_BD_length_l36_36178


namespace not_converge_to_a_l36_36436

theorem not_converge_to_a (x : ℕ → ℝ) (a : ℝ) :
  (∀ ε > 0, ∀ k : ℕ, ∃ n : ℕ, n > k ∧ |x n - a| ≥ ε) →
  ¬ (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |x n - a| < ε) :=
by sorry

end not_converge_to_a_l36_36436


namespace probability_of_B_given_A_l36_36929

noncomputable def balls_in_box : Prop :=
  let total_balls := 12
  let yellow_balls := 5
  let blue_balls := 4
  let green_balls := 3
  let event_A := (yellow_balls * green_balls + yellow_balls * blue_balls + green_balls * blue_balls) / (total_balls * (total_balls - 1) / 2)
  let event_B := (yellow_balls * blue_balls) / (total_balls * (total_balls - 1) / 2)
  (event_B / event_A) = 20 / 47

theorem probability_of_B_given_A : balls_in_box := sorry

end probability_of_B_given_A_l36_36929


namespace value_of_m_l36_36048

theorem value_of_m (a b m : ℚ) (h1 : 2 * a = m) (h2 : 5 * b = m) (h3 : a + b = 2) : m = 20 / 7 :=
by
  sorry

end value_of_m_l36_36048


namespace minimum_value_of_quadratic_l36_36562

theorem minimum_value_of_quadratic (p q : ℝ) (hp : 0 < p) (hq : 0 < q) : 
  ∃ x : ℝ, x = - (p + q) / 2 ∧ ∀ y : ℝ, (y^2 + p*y + q*y) ≥ ((- (p + q) / 2)^2 + p*(- (p + q) / 2) + q*(- (p + q) / 2)) := by
  sorry

end minimum_value_of_quadratic_l36_36562


namespace elevation_angle_second_ship_l36_36851

-- Assume h is the height of the lighthouse.
def h : ℝ := 100

-- Assume d_total is the distance between the two ships.
def d_total : ℝ := 273.2050807568877

-- Assume θ₁ is the angle of elevation from the first ship.
def θ₁ : ℝ := 30

-- Assume θ₂ is the angle of elevation from the second ship.
def θ₂ : ℝ := 45

-- Prove that angle of elevation from the second ship is 45 degrees.
theorem elevation_angle_second_ship : θ₂ = 45 := by
  sorry

end elevation_angle_second_ship_l36_36851


namespace domain_log_function_l36_36366

theorem domain_log_function :
  {x : ℝ | 1 < x ∧ x < 3 ∧ x ≠ 2} = {x : ℝ | (3 - x > 0) ∧ (x - 1 > 0) ∧ (x - 1 ≠ 1)} :=
sorry

end domain_log_function_l36_36366


namespace simplify_expression_l36_36360

noncomputable def a : ℝ := Real.sqrt 3 - 1

theorem simplify_expression : 
  ( (a - 1) / (a^2 - 2 * a + 1) / ( (a^2 + a) / (a^2 - 1) + 1 / (a - 1) ) = Real.sqrt 3 / 3 ) :=
by
  sorry

end simplify_expression_l36_36360


namespace rain_at_least_once_l36_36832

theorem rain_at_least_once (p : ℚ) (h : p = 3/4) : 
    (1 - (1 - p)^4) = 255/256 :=
by
  sorry

end rain_at_least_once_l36_36832


namespace value_of_expression_l36_36025

theorem value_of_expression (a : ℝ) (h : 10 * a^2 + 3 * a + 2 = 5) : 
  3 * a + 2 = (31 + 3 * Real.sqrt 129) / 20 :=
by sorry

end value_of_expression_l36_36025


namespace find_range_of_a_l36_36053

noncomputable def value_range_for_a : Set ℝ := {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4}

theorem find_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0)  ∧
  (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0) ∧
  (¬ (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0) → ¬ (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0))
  → a ∈ value_range_for_a :=
sorry

end find_range_of_a_l36_36053


namespace find_two_numbers_l36_36272

theorem find_two_numbers (x y : ℕ) : 
  (x + y = 20) ∧
  (x * y = 96) ↔ 
  ((x = 12 ∧ y = 8) ∨ (x = 8 ∧ y = 12)) := 
by
  sorry

end find_two_numbers_l36_36272


namespace foreman_can_establish_corr_foreman_cannot_with_less_l36_36797

-- Define the given conditions:
def num_rooms (n : ℕ) := 2^n
def num_checks (n : ℕ) := 2 * n

-- Part (a)
theorem foreman_can_establish_corr (n : ℕ) : 
  ∃ (c : ℕ), c = num_checks n ∧ (c ≥ 2 * n) :=
by
  sorry

-- Part (b)
theorem foreman_cannot_with_less (n : ℕ) : 
  ¬ (∃ (c : ℕ), c = 2 * n - 1 ∧ (c < 2 * n)) :=
by
  sorry

end foreman_can_establish_corr_foreman_cannot_with_less_l36_36797


namespace sum_of_a_for_one_solution_l36_36127

theorem sum_of_a_for_one_solution (a : ℝ) :
  (∀ x : ℝ, 3 * x^2 + (a + 15) * x + 18 = 0 ↔ (a + 15) ^ 2 - 4 * 3 * 18 = 0) →
  a = -15 + 6 * Real.sqrt 6 ∨ a = -15 - 6 * Real.sqrt 6 → a + (-15 + 6 * Real.sqrt 6) + (-15 - 6 * Real.sqrt 6) = -30 :=
by
  intros h1 h2
  have hsum : (-15 + 6 * Real.sqrt 6) + (-15 - 6 * Real.sqrt 6) = -30 := by linarith [Real.sqrt 6]
  sorry

end sum_of_a_for_one_solution_l36_36127


namespace ab_not_divisible_by_5_then_neither_divisible_l36_36221

theorem ab_not_divisible_by_5_then_neither_divisible (a b : ℕ) : ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) → ¬(5 ∣ (a * b)) :=
by
  -- Mathematical statement for proof by contradiction:
  have H1: ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) := sorry
  -- Rest of the proof would go here  
  sorry

end ab_not_divisible_by_5_then_neither_divisible_l36_36221


namespace packs_of_string_cheese_l36_36184

theorem packs_of_string_cheese (cost_per_piece: ℕ) (pieces_per_pack: ℕ) (total_cost_dollars: ℕ) 
                                (h1: cost_per_piece = 10) 
                                (h2: pieces_per_pack = 20) 
                                (h3: total_cost_dollars = 6) : 
  (total_cost_dollars * 100) / (cost_per_piece * pieces_per_pack) = 3 := 
by
  -- Insert proof here
  sorry

end packs_of_string_cheese_l36_36184


namespace hyperbola_problem_l36_36243

noncomputable def is_hyperbola (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) - ((y - 2)^2 / b^2) = 1

variables (s : ℝ)

theorem hyperbola_problem
  (h₁ : is_hyperbola 0 5 a b)
  (h₂ : is_hyperbola (-1) 6 a b)
  (h₃ : is_hyperbola s 3 a b)
  (hb : b^2 = 9)
  (ha : a^2 = 9 / 25) :
  s^2 = 2 / 5 :=
sorry

end hyperbola_problem_l36_36243


namespace point_in_second_quadrant_iff_l36_36628

theorem point_in_second_quadrant_iff (a : ℝ) : (a - 2 < 0) ↔ (a < 2) :=
by
  sorry

end point_in_second_quadrant_iff_l36_36628


namespace negation_of_p_l36_36774

variable (p : ∀ x : ℝ, x^2 + x - 6 ≤ 0)

theorem negation_of_p : (∃ x : ℝ, x^2 + x - 6 > 0) :=
sorry

end negation_of_p_l36_36774


namespace combinations_of_eight_choose_three_is_fifty_six_l36_36471

theorem combinations_of_eight_choose_three_is_fifty_six :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end combinations_of_eight_choose_three_is_fifty_six_l36_36471


namespace cats_left_l36_36237

theorem cats_left (siamese house sold : ℕ) (h1 : siamese = 12) (h2 : house = 20) (h3 : sold = 20) :  
  (siamese + house) - sold = 12 := 
by
  sorry

end cats_left_l36_36237


namespace find_initial_books_each_l36_36069

variable (x : ℝ)
variable (sandy_books : ℝ := x)
variable (tim_books : ℝ := 2 * x + 33)
variable (benny_books : ℝ := 3 * x - 24)
variable (total_books : ℝ := 100)

theorem find_initial_books_each :
  sandy_books + tim_books + benny_books = total_books → x = 91 / 6 := by
  sorry

end find_initial_books_each_l36_36069


namespace arithmetic_sequence_y_value_l36_36123

theorem arithmetic_sequence_y_value (y : ℚ) :
  ∃ y : ℚ, 
    (y - 2) - (2/3) = (4 * y - 1) - (y - 2) → 
    y = 11/6 := by
  sorry

end arithmetic_sequence_y_value_l36_36123


namespace pythagorean_triangle_inscribed_circle_radius_is_integer_l36_36342

theorem pythagorean_triangle_inscribed_circle_radius_is_integer 
  (a b c : ℕ)
  (h1 : c^2 = a^2 + b^2) 
  (h2 : r = (a + b - c) / 2) :
  ∃ (r : ℕ), r = (a + b - c) / 2 :=
sorry

end pythagorean_triangle_inscribed_circle_radius_is_integer_l36_36342


namespace parabola_vertex_x_coordinate_l36_36369

theorem parabola_vertex_x_coordinate (a b c : ℝ) (h1 : c = 0) (h2 : 16 * a + 4 * b = 0) (h3 : 9 * a + 3 * b = 9) : 
    -b / (2 * a) = 2 :=
by 
  -- You can start by adding a proof here
  sorry

end parabola_vertex_x_coordinate_l36_36369


namespace graph_symmetry_l36_36671

/-- Theorem:
The functions y = 2^x and y = 2^{-x} are symmetric about the y-axis.
-/
theorem graph_symmetry :
  ∀ (x : ℝ), (∃ (y : ℝ), y = 2^x) →
  (∃ (y' : ℝ), y' = 2^(-x)) →
  (∀ (y : ℝ), ∃ (x : ℝ), (y = 2^x ↔ y = 2^(-x)) → y = 2^x → y = 2^(-x)) :=
by
  intro x
  intro h1
  intro h2
  intro y
  exists x
  intro h3
  intro hy
  sorry

end graph_symmetry_l36_36671


namespace fraction_inequality_solution_l36_36262

theorem fraction_inequality_solution (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3) :
  3 * x + 2 < 2 * (5 * x - 4) → (10 / 7) < x ∧ x ≤ 3 :=
by
  sorry

end fraction_inequality_solution_l36_36262


namespace quadratic_positive_imp_ineq_l36_36480

theorem quadratic_positive_imp_ineq (b c : ℤ) :
  (∀ x : ℤ, x^2 + b * x + c > 0) → b^2 - 4 * c ≤ 0 :=
by 
  sorry

end quadratic_positive_imp_ineq_l36_36480


namespace margo_walks_total_distance_l36_36058

theorem margo_walks_total_distance :
  let time_to_house := 15
  let time_to_return := 25
  let total_time_minutes := time_to_house + time_to_return
  let total_time_hours := (total_time_minutes : ℝ) / 60
  let avg_rate := 3  -- units: miles per hour
  (avg_rate * total_time_hours = 2) := 
sorry

end margo_walks_total_distance_l36_36058


namespace smallest_n_for_4n_square_and_5n_cube_l36_36700

theorem smallest_n_for_4n_square_and_5n_cube :
  ∃ (n : ℕ), (n > 0 ∧ (∃ k : ℕ, 4 * n = k^2) ∧ (∃ m : ℕ, 5 * n = m^3)) ∧ n = 400 :=
by
  sorry

end smallest_n_for_4n_square_and_5n_cube_l36_36700


namespace max_points_on_poly_graph_l36_36597

theorem max_points_on_poly_graph (P : Polynomial ℤ) (h_deg : P.degree = 20):
  ∃ (S : Finset (ℤ × ℤ)), (∀ p ∈ S, 0 ≤ p.snd ∧ p.snd ≤ 10) ∧ S.card ≤ 20 ∧ 
  ∀ S' : Finset (ℤ × ℤ), (∀ p ∈ S', 0 ≤ p.snd ∧ p.snd ≤ 10) → S'.card ≤ 20 :=
by
  sorry

end max_points_on_poly_graph_l36_36597


namespace combined_height_of_trees_is_correct_l36_36047

noncomputable def original_height_of_trees 
  (h1_current : ℝ) (h1_growth_rate : ℝ)
  (h2_current : ℝ) (h2_growth_rate : ℝ)
  (h3_current : ℝ) (h3_growth_rate : ℝ)
  (conversion_rate : ℝ) : ℝ :=
  let h1 := h1_current / (1 + h1_growth_rate)
  let h2 := h2_current / (1 + h2_growth_rate)
  let h3 := h3_current / (1 + h3_growth_rate)
  (h1 + h2 + h3) / conversion_rate

theorem combined_height_of_trees_is_correct :
  original_height_of_trees 240 0.70 300 0.50 180 0.60 12 = 37.81 :=
by
  sorry

end combined_height_of_trees_is_correct_l36_36047


namespace moles_of_NaCl_formed_l36_36899

theorem moles_of_NaCl_formed (hcl moles : ℕ) (nahco3 moles : ℕ) (reaction : ℕ → ℕ → ℕ) :
  hcl = 3 → nahco3 = 3 → reaction 1 1 = 1 →
  reaction hcl nahco3 = 3 :=
by 
  intros h1 h2 h3
  -- Proof omitted
  sorry

end moles_of_NaCl_formed_l36_36899


namespace product_ends_in_36_l36_36686

theorem product_ends_in_36 (a b : ℕ) (ha : a < 10) (hb : b < 10) :
  ((10 * a + 6) * (10 * b + 6)) % 100 = 36 ↔ (a + b = 0 ∨ a + b = 5 ∨ a + b = 10 ∨ a + b = 15) :=
by
  sorry

end product_ends_in_36_l36_36686


namespace find_A_l36_36371

theorem find_A (
  A B C A' r : ℕ
) (hA : A = 312) (hB : B = 270) (hC : C = 211)
  (hremA : A % A' = 4 * r)
  (hremB : B % A' = 2 * r)
  (hremC : C % A' = r) :
  A' = 19 :=
by
  sorry

end find_A_l36_36371


namespace Mike_changed_64_tires_l36_36485

def tires_changed (motorcycles: ℕ) (cars: ℕ): ℕ := 
  (motorcycles * 2) + (cars * 4)

theorem Mike_changed_64_tires:
  (tires_changed 12 10) = 64 :=
by
  sorry

end Mike_changed_64_tires_l36_36485


namespace product_mod_25_l36_36573

def remainder_when_divided_by_25 (n : ℕ) : ℕ := n % 25

theorem product_mod_25 (a b c d : ℕ) 
  (h1 : a = 1523) (h2 : b = 1857) (h3 : c = 1919) (h4 : d = 2012) :
  remainder_when_divided_by_25 (a * b * c * d) = 8 :=
by
  sorry

end product_mod_25_l36_36573


namespace smallest_n_for_perfect_square_and_cube_l36_36708

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, 0 < n ∧ (∃ a1 b1 : ℕ, 4 * n = a1 ^ 2 ∧ 5 * n = b1 ^ 3 ∧ n = 50) :=
begin
  use 50,
  split,
  { norm_num, },
  { use [10, 5],
    split,
    { norm_num, },
    { split, 
      { norm_num, },
      { refl, }, },
  },
  sorry
end

end smallest_n_for_perfect_square_and_cube_l36_36708


namespace value_A_minus_B_l36_36995

-- Conditions definitions
def A : ℕ := (1 * 1000) + (16 * 100) + (28 * 10)
def B : ℕ := 355 + 245 * 3

-- Theorem statement
theorem value_A_minus_B : A - B = 1790 := by
  sorry

end value_A_minus_B_l36_36995


namespace find_integer_less_than_M_div_100_l36_36912

-- The problem and proof constants
theorem find_integer_less_than_M_div_100 :
  let M := 4992 in
  let result := ⌊M / 100⌋ in
  result = 49 :=
by
  -- The conditions given and the resultant M is defined.
  have h1 : 1 / (3! * 18!) + 1 / (4! * 17!) + 1 / (5! * 16!) + 1 / (6! * 15!) + 1 / (7! * 14!) + 
            1 / (8! * 13!) + 1 / (9! * 12!) + 1 / (10! * 11!) = M / (2! * 19!) := sorry,
  -- Hence, final result.
  have h2 : M = 4992 := sorry,
  have h3 : result = ⌊4992 / 100⌋ := by simp [M, result, int.floor_eq_iff, ←div_lt_iff, int.cast_49],
  exact h3

end find_integer_less_than_M_div_100_l36_36912


namespace helen_chocolate_chip_cookies_l36_36158

def number_of_raisin_cookies := 231
def difference := 25

theorem helen_chocolate_chip_cookies :
  ∃ C, C = number_of_raisin_cookies + difference ∧ C = 256 :=
by
  sorry -- Skipping the proof

end helen_chocolate_chip_cookies_l36_36158


namespace percentage_increase_l36_36533

theorem percentage_increase (L : ℝ) (h : L + 60 = 240) : ((60 / L) * 100 = 33 + (1 / 3) * 100) :=
by
  sorry

end percentage_increase_l36_36533


namespace train_length_l36_36249

theorem train_length (t_platform t_pole : ℕ) (platform_length : ℕ) (train_length : ℕ) :
  t_platform = 39 → t_pole = 18 → platform_length = 350 →
  (train_length + platform_length) / t_platform = train_length / t_pole →
  train_length = 300 :=
by
  intros ht_platform ht_pole hplatform_length hspeeds 
  have h1 : train_length / 18 = (train_length + 350) / 39, from hspeeds
  have h2 : 39 * (train_length / 18) = 39 * ((train_length + 350) / 39), from congrArg (λ x, 39 * x) h1
  sorry

end train_length_l36_36249


namespace blue_paint_amount_l36_36438

/-- 
Prove that if Giselle uses 15 quarts of white paint, then according to the ratio 4:3:5, she should use 12 quarts of blue paint.
-/
theorem blue_paint_amount (white_paint : ℚ) (h1 : white_paint = 15) : 
  let blue_ratio := 4;
  let white_ratio := 5;
  blue_ratio / white_ratio * white_paint = 12 :=
by
  sorry

end blue_paint_amount_l36_36438


namespace increasing_range_of_a_l36_36621

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 1 / x

theorem increasing_range_of_a (a : ℝ) : (∀ x > (1/2), (3 * x^2 + a - 1 / x^2) ≥ 0) ↔ a ≥ (13 / 4) :=
by sorry

end increasing_range_of_a_l36_36621


namespace least_number_to_add_l36_36691

theorem least_number_to_add (a b : ℤ) (d : ℤ) (h : a = 1054) (hb : b = 47) (hd : d = 27) :
  ∃ n : ℤ, (a + d) % b = 0 :=
by
  sorry

end least_number_to_add_l36_36691


namespace measure_of_angle_Z_l36_36801

theorem measure_of_angle_Z (X Y Z : ℝ) (h_sum : X + Y + Z = 180) (h_XY : X + Y = 80) : Z = 100 := 
by
  -- The proof is not required.
  sorry

end measure_of_angle_Z_l36_36801


namespace CindyHomework_l36_36887

theorem CindyHomework (x : ℤ) (h : (x - 7) * 4 = 48) : (4 * x - 7) = 69 := by
  sorry

end CindyHomework_l36_36887


namespace total_amount_is_70000_l36_36402

-- Definitions based on the given conditions
def total_amount_divided (amount_10: ℕ) (amount_20: ℕ) : ℕ :=
  amount_10 + amount_20

def interest_earned (amount_10: ℕ) (amount_20: ℕ) : ℕ :=
  (amount_10 * 10 / 100) + (amount_20 * 20 / 100)

-- Statement to be proved
theorem total_amount_is_70000 (amount_10: ℕ) (amount_20: ℕ) (total_interest: ℕ) :
  amount_10 = 60000 →
  total_interest = 8000 →
  interest_earned amount_10 amount_20 = total_interest →
  total_amount_divided amount_10 amount_20 = 70000 :=
by
  intros h1 h2 h3
  sorry

end total_amount_is_70000_l36_36402


namespace son_age_next_year_l36_36061

-- Definitions based on the given conditions
def my_current_age : ℕ := 35
def son_current_age : ℕ := my_current_age / 5

-- Theorem statement to prove the answer
theorem son_age_next_year : son_current_age + 1 = 8 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end son_age_next_year_l36_36061


namespace cost_of_paint_per_kg_l36_36082

/-- The cost of painting one square foot is Rs. 50. -/
theorem cost_of_paint_per_kg (side_length : ℝ) (cost_total : ℝ) (coverage_per_kg : ℝ) (total_surface_area : ℝ) (total_paint_needed : ℝ) (cost_per_kg : ℝ) 
  (h1 : side_length = 20)
  (h2 : cost_total = 6000)
  (h3 : coverage_per_kg = 20)
  (h4 : total_surface_area = 6 * side_length^2)
  (h5 : total_paint_needed = total_surface_area / coverage_per_kg)
  (h6 : cost_per_kg = cost_total / total_paint_needed) :
  cost_per_kg = 50 :=
sorry

end cost_of_paint_per_kg_l36_36082


namespace decreasing_function_condition_l36_36044

theorem decreasing_function_condition (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, x ≤ 3 → deriv f x ≤ 0) ↔ (m ≥ 1) :=
by 
  sorry

end decreasing_function_condition_l36_36044


namespace sum_geometric_arithmetic_progression_l36_36582

theorem sum_geometric_arithmetic_progression :
  ∃ (a b r d : ℝ), a = 1 * r ∧ b = 1 * r^2 ∧ b = a + d ∧ 16 = b + d ∧ (a + b = 12.64) :=
by
  sorry

end sum_geometric_arithmetic_progression_l36_36582


namespace side_length_estimate_l36_36502

theorem side_length_estimate (x : ℝ) (h : x^2 = 15) : 3 < x ∧ x < 4 :=
sorry

end side_length_estimate_l36_36502


namespace find_values_l36_36297

theorem find_values (a b : ℝ) 
  (h1 : a + b = 10)
  (h2 : a - b = 4) 
  (h3 : a^2 + b^2 = 58) : 
  a^2 - b^2 = 40 ∧ ab = 21 := 
by 
  sorry

end find_values_l36_36297


namespace find_subtracted_value_l36_36102

theorem find_subtracted_value (n x : ℕ) (h₁ : n = 36) (h₂ : ((n + 10) * 2 / 2 - x) = 44) : x = 2 :=
by
  sorry

end find_subtracted_value_l36_36102


namespace installation_cost_l36_36819

-- Definitions
variables (LP : ℝ) (P : ℝ := 16500) (D : ℝ := 0.2) (T : ℝ := 125) (SP : ℝ := 23100) (I : ℝ)

-- Conditions
def purchase_price := P = (1 - D) * LP
def selling_price := SP = 1.1 * LP
def total_cost := P + T + I = SP

-- Proof Statement
theorem installation_cost : I = 6350 :=
  by
    -- sorry is used to skip the proof
    sorry

end installation_cost_l36_36819


namespace neg_disj_imp_neg_conj_l36_36295

theorem neg_disj_imp_neg_conj (p q : Prop) (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
sorry

end neg_disj_imp_neg_conj_l36_36295


namespace simplify_expression_l36_36359

theorem simplify_expression (z : ℝ) : (3 - 5 * z^2) - (4 * z^2 + 2 * z - 5) = 8 - 9 * z^2 - 2 * z :=
by
  sorry

end simplify_expression_l36_36359


namespace evaluate_64_pow_5_div_6_l36_36752

theorem evaluate_64_pow_5_div_6 : (64 : ℝ) ^ (5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ) ^ 6 := by norm_num
  have h2 : (64 : ℝ) ^ (5 / 6) = ((2 : ℝ) ^ 6) ^ (5 / 6) := by rw h1
  have h3 : ((2 : ℝ) ^ 6) ^ (5 / 6) = (2 : ℝ) ^ (6 * (5 / 6)) := by rw [Real.rpow_mul]
  have h4 : (2 : ℝ) ^ (6 * (5 / 6)) = (2 : ℝ) ^ 5 := by norm_num
  rw [h2, h3, h4]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l36_36752


namespace PB_distance_eq_l36_36714

theorem PB_distance_eq {
  A B C D P : Type
} (PA PD PC : ℝ) (hPA: PA = 6) (hPD: PD = 8) (hPC: PC = 10)
  (h_equidistant: ∃ y : ℝ, PA^2 + y^2 = PB^2 ∧ PD^2 + y^2 = PC^2) :
  ∃ PB : ℝ, PB = 6 * Real.sqrt 2 := 
by
  sorry

end PB_distance_eq_l36_36714


namespace BD_value_l36_36177

def quadrilateral_ABCD_sides (AB BC CD DA : ℕ) (BD : ℕ) : Prop :=
  AB = 5 ∧ BC = 17 ∧ CD = 5 ∧ DA = 9 ∧ 12 < BD ∧ BD < 14 ∧ BD = 13

theorem BD_value (AB BC CD DA : ℕ) (BD : ℕ) : 
  quadrilateral_ABCD_sides AB BC CD DA BD → BD = 13 :=
by
  sorry

end BD_value_l36_36177


namespace parabola_translation_correct_l36_36217

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ := 8 * x^2

-- Define the transformation of translating 3 units to the left and 5 units down
def translate_parabola (f : ℝ → ℝ) (h k : ℝ) :=
  λ x, f (x + h) - k

-- Define the expected result after translation
def expected_result (x : ℝ) : ℝ := 8 * (x + 3)^2 - 5

-- The theorem statement proving the expected transformation
theorem parabola_translation_correct :
  ∀ x : ℝ, translate_parabola original_parabola 3 5 x = expected_result x :=
by
  intro x
  sorry

end parabola_translation_correct_l36_36217


namespace parallel_lines_implies_value_of_m_l36_36203

theorem parallel_lines_implies_value_of_m :
  ∀ (m : ℝ), (∀ (x y : ℝ), 3 * x + 2 * y - 2 = 0) ∧ (∀ (x y : ℝ), (2 * m - 1) * x + m * y + 1 = 0) → 
  m = 2 := 
by
  sorry

end parallel_lines_implies_value_of_m_l36_36203


namespace factor_expression_l36_36758

theorem factor_expression (b : ℝ) : 180 * b ^ 2 + 36 * b = 36 * b * (5 * b + 1) :=
by
  -- actual proof is omitted
  sorry

end factor_expression_l36_36758


namespace b_plus_one_prime_l36_36331

open Nat

def is_a_nimathur (a b : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ b / a → (a * n + 1) ∣ (Nat.choose (a * n) b - 1)

theorem b_plus_one_prime (a b : ℕ) 
  (ha : 1 ≤ a) 
  (hb : 1 ≤ b) 
  (h : is_a_nimathur a b) 
  (h_not : ¬is_a_nimathur a (b+2)) : 
  Prime (b + 1) := 
sorry

end b_plus_one_prime_l36_36331


namespace average_height_of_four_people_l36_36966

theorem average_height_of_four_people (
  h1 h2 h3 h4 : ℕ
) (diff12 : h2 = h1 + 2)
  (diff23 : h3 = h2 + 2)
  (diff34 : h4 = h3 + 6)
  (h4_eq : h4 = 83) :
  (h1 + h2 + h3 + h4) / 4 = 77 :=
by sorry

end average_height_of_four_people_l36_36966


namespace mapping_f_of_neg2_and_3_l36_36906

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x * y)

-- Define the given point
def p : ℝ × ℝ := (-2, 3)

-- Define the expected corresponding point
def expected_p : ℝ × ℝ := (1, -6)

-- The theorem stating the problem to be proved
theorem mapping_f_of_neg2_and_3 :
  f p.1 p.2 = expected_p := by
  sorry

end mapping_f_of_neg2_and_3_l36_36906


namespace solve_xy_l36_36956

theorem solve_xy (x y : ℕ) :
  (x^2 + (x + y)^2 = (x + 9)^2) ↔ (x = 0 ∧ y = 9) ∨ (x = 8 ∧ y = 7) ∨ (x = 20 ∧ y = 1) :=
by
  sorry

end solve_xy_l36_36956


namespace remainders_inequalities_l36_36777

theorem remainders_inequalities
  (X Y M A B s t u : ℕ)
  (h1 : X > Y)
  (h2 : X = Y + 8)
  (h3 : X % M = A)
  (h4 : Y % M = B)
  (h5 : s = (X^2) % M)
  (h6 : t = (Y^2) % M)
  (h7 : u = (A * B)^2 % M) :
  s ≠ t ∧ t ≠ u ∧ s ≠ u :=
sorry

end remainders_inequalities_l36_36777


namespace teacher_arrangement_l36_36846

theorem teacher_arrangement : 
  let total_teachers := 6
  let max_teachers_per_class := 4
  (∑ i in (finset.range max_teachers_per_class).filter (λ x, x ≤ total_teachers - max_teachers_per_class + 1), 
    nat.choose total_teachers (total_teachers - i) * (if i = total_teachers - i then 1 else 2)) = 31 :=
by sorry

end teacher_arrangement_l36_36846


namespace abs_sum_inequality_for_all_x_l36_36007

theorem abs_sum_inequality_for_all_x (m : ℝ) :
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) ↔ (m ≤ 3) :=
by
  sorry

end abs_sum_inequality_for_all_x_l36_36007


namespace worm_length_difference_is_correct_l36_36501

-- Define the lengths of the worms
def worm1_length : ℝ := 0.8
def worm2_length : ℝ := 0.1

-- Define the difference in length between the longer worm and the shorter worm
def length_difference (a b : ℝ) : ℝ := a - b

-- State the theorem that the length difference is 0.7 inches
theorem worm_length_difference_is_correct (h1 : worm1_length = 0.8) (h2 : worm2_length = 0.1) :
  length_difference worm1_length worm2_length = 0.7 :=
by
  sorry

end worm_length_difference_is_correct_l36_36501


namespace smallest_n_for_4n_square_and_5n_cube_l36_36701

theorem smallest_n_for_4n_square_and_5n_cube :
  ∃ (n : ℕ), (n > 0 ∧ (∃ k : ℕ, 4 * n = k^2) ∧ (∃ m : ℕ, 5 * n = m^3)) ∧ n = 400 :=
by
  sorry

end smallest_n_for_4n_square_and_5n_cube_l36_36701


namespace sum_four_digit_even_numbers_l36_36764

-- Define the digits set
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

-- Define the set of valid units digits for even numbers
def even_units : Finset ℕ := {0, 2, 4}

-- Define the set of all four-digit numbers using the provided digits
def four_digit_even_numbers : Finset ℕ :=
  (Finset.range (10000) \ Finset.range (1000)).filter (λ n =>
    n % 10 ∈ even_units ∧
    (n / 1000) ∈ digits ∧
    ((n / 100) % 10) ∈ digits ∧
    ((n / 10) % 10) ∈ digits)

theorem sum_four_digit_even_numbers :
  (four_digit_even_numbers.sum (λ x => x)) = 1769580 :=
  sorry

end sum_four_digit_even_numbers_l36_36764


namespace gold_coins_equality_l36_36584

theorem gold_coins_equality (pouches : List ℕ) 
  (h_pouches_length : pouches.length = 9)
  (h_pouches_sum : pouches.sum = 60)
  : (∃ s_2 : List (List ℕ), s_2.length = 2 ∧ ∀ l ∈ s_2, l.sum = 30) ∧
    (∃ s_3 : List (List ℕ), s_3.length = 3 ∧ ∀ l ∈ s_3, l.sum = 20) ∧
    (∃ s_4 : List (List ℕ), s_4.length = 4 ∧ ∀ l ∈ s_4, l.sum = 15) ∧
    (∃ s_5 : List (List ℕ), s_5.length = 5 ∧ ∀ l ∈ s_5, l.sum = 12) :=
sorry

end gold_coins_equality_l36_36584


namespace max_determinant_l36_36938

open Matrix

/-- Given vectors v and w as specified, and u being a unit vector orthogonal to v, 
    the largest possible determinant of the matrix formed by columns u, v, and w is 51 / √10. --/
theorem max_determinant (u v w : Vector3 ℝ) 
  (hv : v = ⟨3, 2, -2⟩) 
  (hw : w = ⟨2, -1, 4⟩) 
  (hu_unit : ‖u‖ = 1) 
  (hu_orthogonal : dot_product u v = 0) :
  let u_cross_vw := cross_product v w in
  determinant ![u, v, w] = 51 / Real.sqrt 10 :=
by
  /- The proof would involve showing the steps as per the solution. -/
  sorry

end max_determinant_l36_36938


namespace markov_inequality_l36_36012

variables {Ω : Type*} {X : Ω → ℝ} {n : ℕ} {p : ℝ}
  (G : MeasureTheory.Measure Ω)
  [MeasureTheory.ProbabilityMeasure G]
  [MeasureTheory.HasFiniteIntegral X]

theorem markov_inequality
  (X_nonneg : ∀ ω, 0 ≤ X ω)
  (a : ℝ)
  (a_pos : 0 < a) :
  MeasureTheory.measure (λ ω, X ω ≥ a) ≤ (MeasureTheory.expectation X) / a :=
by
  sorry

end markov_inequality_l36_36012


namespace chickens_and_rabbits_l36_36554

theorem chickens_and_rabbits (total_animals : ℕ) (total_legs : ℕ) (chickens : ℕ) (rabbits : ℕ) 
    (h1 : total_animals = 40) 
    (h2 : total_legs = 108) 
    (h3 : total_animals = chickens + rabbits) 
    (h4 : total_legs = 2 * chickens + 4 * rabbits) : 
    chickens = 26 ∧ rabbits = 14 :=
by
  sorry

end chickens_and_rabbits_l36_36554


namespace profit_sharing_l36_36529

theorem profit_sharing 
  (total_profit : ℝ) 
  (managing_share_percentage : ℝ) 
  (capital_a : ℝ) 
  (capital_b : ℝ) 
  (managing_partner_share : ℝ)
  (total_capital : ℝ) 
  (remaining_profit : ℝ) 
  (proportion_a : ℝ)
  (share_a_remaining : ℝ)
  (total_share_a : ℝ) : 
  total_profit = 8800 → 
  managing_share_percentage = 0.125 → 
  capital_a = 50000 → 
  capital_b = 60000 → 
  managing_partner_share = managing_share_percentage * total_profit → 
  total_capital = capital_a + capital_b → 
  remaining_profit = total_profit - managing_partner_share → 
  proportion_a = capital_a / total_capital → 
  share_a_remaining = proportion_a * remaining_profit → 
  total_share_a = managing_partner_share + share_a_remaining → 
  total_share_a = 4600 :=
by sorry

end profit_sharing_l36_36529


namespace log_function_domain_correct_l36_36365

def log_function_domain : Set ℝ :=
  {x | 1 < x ∧ x < 3 ∧ x ≠ 2}

theorem log_function_domain_correct :
  (∀ x : ℝ, y = log (x - 1) (3 - x) → x ∈ log_function_domain) :=
by
  sorry

end log_function_domain_correct_l36_36365


namespace trapezoid_base_length_l36_36090

-- Definitions from the conditions
def trapezoid_area (a b h : ℕ) : ℕ := (1 / 2) * (a + b) * h

theorem trapezoid_base_length (b : ℕ) (h : ℕ) (a : ℕ) (A : ℕ) (H_area : A = 222) (H_upper_side : a = 23) (H_height : h = 12) :
  A = trapezoid_area a b h ↔ b = 14 :=
by sorry

end trapezoid_base_length_l36_36090


namespace difference_of_bases_l36_36132

def base8_to_base10 (n : ℕ) : ℕ :=
  5 * (8^5) + 4 * (8^4) + 3 * (8^3) + 2 * (8^2) + 1 * (8^1) + 0 * (8^0)

def base5_to_base10 (n : ℕ) : ℕ :=
  4 * (5^4) + 3 * (5^3) + 2 * (5^2) + 1 * (5^1) + 0 * (5^0)

theorem difference_of_bases : 
  base8_to_base10 543210 - base5_to_base10 43210 = 177966 :=
by
  sorry

end difference_of_bases_l36_36132


namespace intersection_of_sets_l36_36315

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {0, 1, 2}

theorem intersection_of_sets :
  C = A ∩ B :=
sorry

end intersection_of_sets_l36_36315


namespace intersection_A_B_l36_36322

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {x | x ∈ A ∧ (x : ℝ) ∈ B}

theorem intersection_A_B : C = {0, 1, 2} := 
by
  sorry

end intersection_A_B_l36_36322


namespace intersection_A_B_l36_36321

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {x | x ∈ A ∧ (x : ℝ) ∈ B}

theorem intersection_A_B : C = {0, 1, 2} := 
by
  sorry

end intersection_A_B_l36_36321


namespace area_of_trapezoid_RSQT_l36_36877
-- Import the required library

-- Declare the geometrical setup and given areas
variables (PQ PR : ℝ)
variable (PQR_area : ℝ)
variable (small_triangle_area : ℝ)
variable (num_small_triangles : ℕ)
variable (inner_triangle_area : ℝ)
variable (trapezoid_RSQT_area : ℝ)

-- Define the conditions from part a)
def isosceles_triangle : Prop := PQ = PR
def triangle_PQR_area_given : Prop := PQR_area = 75
def small_triangle_area_given : Prop := small_triangle_area = 3
def num_small_triangles_given : Prop := num_small_triangles = 9
def inner_triangle_area_given : Prop := inner_triangle_area = 5 * small_triangle_area

-- Define the target statement (question == answer)
theorem area_of_trapezoid_RSQT :
  isosceles_triangle PQ PR ∧
  triangle_PQR_area_given PQR_area ∧
  small_triangle_area_given small_triangle_area ∧
  num_small_triangles_given num_small_triangles ∧
  inner_triangle_area_given small_triangle_area inner_triangle_area → 
  trapezoid_RSQT_area = 60 :=
sorry

end area_of_trapezoid_RSQT_l36_36877


namespace equation_of_perpendicular_line_l36_36579

theorem equation_of_perpendicular_line :
  ∃ c : ℝ, (∀ x y : ℝ, (2 * x + y + c = 0 ↔ (x = 1 ∧ y = 1))) → (c = -3) := 
by
  sorry

end equation_of_perpendicular_line_l36_36579


namespace combinations_of_eight_choose_three_is_fifty_six_l36_36470

theorem combinations_of_eight_choose_three_is_fifty_six :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end combinations_of_eight_choose_three_is_fifty_six_l36_36470


namespace probability_not_red_l36_36170

theorem probability_not_red (h : odds_red = 1 / 3) : probability_not_red_card = 3 / 4 :=
by
  sorry

end probability_not_red_l36_36170


namespace number_of_triangles_for_second_star_l36_36900

theorem number_of_triangles_for_second_star (a b : ℝ) (h₁ : a + b + 90 = 180) (h₂ : 5 * (360 / 5) = 360) :
  360 / (180 - 90 - (360 / 5)) = 20 :=
by
  sorry

end number_of_triangles_for_second_star_l36_36900


namespace percentage_difference_l36_36201

variables (G P R : ℝ)

-- Conditions
def condition1 : Prop := P = 0.9 * G
def condition2 : Prop := R = 3.0000000000000006 * G

-- Theorem to prove
theorem percentage_difference (h1 : condition1 P G) (h2 : condition2 R G) : 
  (R - P) / R * 100 = 70 :=
sorry

end percentage_difference_l36_36201


namespace train_passing_time_l36_36251

noncomputable def length_of_train : ℝ := 450
noncomputable def speed_kmh : ℝ := 80
noncomputable def length_of_station : ℝ := 300
noncomputable def speed_m_per_s : ℝ := speed_kmh * 1000 / 3600 -- Convert km/hour to m/second
noncomputable def total_distance : ℝ := length_of_train + length_of_station
noncomputable def passing_time : ℝ := total_distance / speed_m_per_s

theorem train_passing_time : abs (passing_time - 33.75) < 0.01 :=
by
  sorry

end train_passing_time_l36_36251


namespace mark_increase_reading_time_l36_36482

def initial_pages_per_day : ℕ := 100
def final_pages_per_week : ℕ := 1750
def days_in_week : ℕ := 7

def calculate_percentage_increase (initial_pages_per_day : ℕ) (final_pages_per_week : ℕ) (days_in_week : ℕ) : ℚ :=
  ((final_pages_per_week : ℚ) / ((initial_pages_per_day : ℚ) * (days_in_week : ℚ)) - 1) * 100

theorem mark_increase_reading_time :
  calculate_percentage_increase initial_pages_per_day final_pages_per_week days_in_week = 150 :=
by sorry

end mark_increase_reading_time_l36_36482


namespace overall_profit_percentage_is_30_l36_36551

noncomputable def overall_profit_percentage (n_A n_B : ℕ) (price_A price_B profit_A profit_B : ℝ) : ℝ :=
  (n_A * profit_A + n_B * profit_B) / (n_A * price_A + n_B * price_B) * 100

theorem overall_profit_percentage_is_30 :
  overall_profit_percentage 5 10 850 950 225 300 = 30 :=
by
  sorry

end overall_profit_percentage_is_30_l36_36551


namespace observable_sea_creatures_l36_36118

theorem observable_sea_creatures (P_shark : ℝ) (P_truth : ℝ) (n : ℕ)
  (h1 : P_shark = 0.027777777777777773)
  (h2 : P_truth = 1/6)
  (h3 : P_shark = P_truth * (1/n : ℝ)) : 
  n = 6 := 
  sorry

end observable_sea_creatures_l36_36118


namespace rationalize_denominator_l36_36951

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  A + B + C + D + E = 22 :=
by
  -- Proof goes here
  sorry

end rationalize_denominator_l36_36951


namespace ratio_of_areas_l36_36960

theorem ratio_of_areas (x y l : ℝ)
  (h1 : 2 * (x + 3 * y) = 2 * (l + y))
  (h2 : 2 * x + l = 3 * y) :
  (x * 3 * y) / (l * y) = 3 / 7 :=
by
  -- Proof will be provided here
  sorry

end ratio_of_areas_l36_36960


namespace number_of_candidates_l36_36395

theorem number_of_candidates (n : ℕ) (h : n * (n - 1) = 42) : n = 7 :=
sorry

end number_of_candidates_l36_36395


namespace log_400_cannot_be_computed_l36_36593

theorem log_400_cannot_be_computed :
  let log_8 : ℝ := 0.9031
  let log_9 : ℝ := 0.9542
  let log_7 : ℝ := 0.8451
  (∀ (log_2 log_3 log_5 : ℝ), log_2 = 1 / 3 * log_8 → log_3 = 1 / 2 * log_9 → log_5 = 1 → 
    (∀ (log_val : ℝ), 
      (log_val = log_21 → log_21 = log_3 + log_7 → log_val = (1 / 2) * log_9 + log_7)
      ∧ (log_val = log_9_over_8 → log_9_over_8 = log_9 - log_8)
      ∧ (log_val = log_126 → log_126 = log_2 + log_7 + log_9 → log_val = (1 / 3) * log_8 + log_7 + log_9)
      ∧ (log_val = log_0_875 → log_0_875 = log_7 - log_8)
      ∧ (log_val = log_400 → log_400 = log_8 + 1 + log_5) 
      → False))
:= 
sorry

end log_400_cannot_be_computed_l36_36593


namespace evaluate_64_pow_fifth_sixth_l36_36751

theorem evaluate_64_pow_fifth_sixth : 64 ^ (5 / 6) = 32 := by
  have h : 64 = 2 ^ 6 := by sorry
  calc 64 ^ (5 / 6) = (2 ^ 6) ^ (5 / 6) : by rw [h]
              ...   = 2 ^ (6 * (5 / 6))  : by sorry
              ...   = 2 ^ 5              : by sorry
              ...   = 32                 : by sorry

end evaluate_64_pow_fifth_sixth_l36_36751


namespace find_all_waldo_time_l36_36517

theorem find_all_waldo_time (b : ℕ) (p : ℕ) (t : ℕ) :
  b = 15 → p = 30 → t = 3 → b * p * t = 1350 := by
sorry

end find_all_waldo_time_l36_36517


namespace range_a_l36_36055

open Set Real

-- Define the predicate p: real number x satisfies x^2 - 4ax + 3a^2 < 0, where a < 0
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0

-- Define the predicate q: real number x satisfies x^2 - x - 6 ≤ 0, or x^2 + 2x - 8 > 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- Define the complement sets
def not_p_set (a : ℝ) : Set ℝ := {x | ¬p x a}
def not_q_set : Set ℝ := {x | ¬q x}

-- Define p as necessary but not sufficient condition for q
def necessary_but_not_sufficient (a : ℝ) : Prop := 
  (not_q_set ⊆ not_p_set a) ∧ ¬(not_p_set a ⊆ not_q_set)

-- The main theorem to prove
theorem range_a : {a : ℝ | necessary_but_not_sufficient a} = {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4} :=
by
  sorry

end range_a_l36_36055


namespace is_equilateral_l36_36309

open Complex

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry

-- Assume the conditions of the problem
axiom z1_distinct_z2 : z1 ≠ z2
axiom z2_distinct_z3 : z2 ≠ z3
axiom z3_distinct_z1 : z3 ≠ z1
axiom z1_unit_circle : abs z1 = 1
axiom z2_unit_circle : abs z2 = 1
axiom z3_unit_circle : abs z3 = 1
axiom condition : (1 / (2 + abs (z1 + z2)) + 1 / (2 + abs (z2 + z3)) + 1 / (2 + abs (z3 + z1))) = 1
axiom acute_angled_triangle : sorry

theorem is_equilateral (A B C : ℂ) (hA : A = z1) (hB : B = z2) (hC : C = z3) : 
  (sorry : Prop) := sorry

end is_equilateral_l36_36309


namespace rationalize_denominator_l36_36952

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5 in
  B < D ∧
  (12/5 * Real.sqrt 7) = ((1:ℚ) * Real.sqrt B / E * A) * (-1) ∧
  (9/5 * Real.sqrt 13) = (1:ℚ * Real.sqrt D / E * C) ∧
  (A + B + C + D + E = 22) :=
by
  sorry

end rationalize_denominator_l36_36952


namespace simplify_expression_l36_36196

variable (m n : ℝ)

theorem simplify_expression : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end simplify_expression_l36_36196


namespace jung_kook_blue_balls_l36_36748

def num_boxes := 2
def blue_balls_per_box := 5
def total_blue_balls := num_boxes * blue_balls_per_box

theorem jung_kook_blue_balls : total_blue_balls = 10 :=
by
  sorry

end jung_kook_blue_balls_l36_36748


namespace cubic_root_sum_l36_36806

noncomputable def poly : Polynomial ℝ := Polynomial.C (-4) + Polynomial.C 3 * X + Polynomial.C (-2) * X^2 + X^3

theorem cubic_root_sum (a b c : ℝ) (h1 : poly.eval a = 0) (h2 : poly.eval b = 0) (h3 : poly.eval c = 0)
  (h_sum : a + b + c = 2) (h_prod : a * b + b * c + c * a = 3) (h_triple_prod : a * b * c = 4) :
  a^3 + b^3 + c^3 = 2 := 
by {
  sorry
}

end cubic_root_sum_l36_36806


namespace complex_exp_neg_ipi_on_real_axis_l36_36476

theorem complex_exp_neg_ipi_on_real_axis :
  (Complex.exp (-Real.pi * Complex.I)).im = 0 :=
by 
  sorry

end complex_exp_neg_ipi_on_real_axis_l36_36476


namespace distance_between_trees_l36_36457

theorem distance_between_trees (L : ℝ) (n : ℕ) (hL : L = 375) (hn : n = 26) : 
  (L / (n - 1) = 15) :=
by
  sorry

end distance_between_trees_l36_36457


namespace tina_total_pens_l36_36972

theorem tina_total_pens : 
  let pink_pens := 12 in
  let green_pens := pink_pens - 9 in
  let blue_pens := green_pens + 3 in
  pink_pens + green_pens + blue_pens = 21 :=
by 
  let pink_pens := 12
  let green_pens := pink_pens - 9
  let blue_pens := green_pens + 3
  show pink_pens + green_pens + blue_pens = 21 from sorry

end tina_total_pens_l36_36972


namespace star_example_l36_36572

def star (x y : ℝ) : ℝ := 2 * x * y - 3 * x + y

theorem star_example : (star 6 4) - (star 4 6) = -8 := by
  sorry

end star_example_l36_36572


namespace rationalize_denominator_to_find_constants_l36_36947

-- Definitions of the given conditions
def original_fraction := 3 / (4 * Real.sqrt 7 + 3 * Real.sqrt 13)
def simplified_fraction (A B C D E : ℤ) := (A * Real.sqrt B + C * Real.sqrt D) / E

-- Statement of the proof problem
theorem rationalize_denominator_to_find_constants :
  ∃ (A B C D E : ℤ),
    original_fraction = simplified_fraction A B C D E ∧
    B < D ∧
    (∀ p : ℕ, Real.sqrt (p * p) = p) ∧ -- Ensuring that all radicals are in simplest form
    A + B + C + D + E = 22 :=
sorry

end rationalize_denominator_to_find_constants_l36_36947


namespace part1_part2_l36_36852

def is_sum_solution_equation (a b x : ℝ) : Prop :=
  x = b + a

def part1_statement := ¬ is_sum_solution_equation 3 4.5 (4.5 / 3)

def part2_statement (m : ℝ) : Prop :=
  is_sum_solution_equation 5 (m + 1) (m + 6) → m = (-29 / 4)

theorem part1 : part1_statement :=
by 
  -- Proof here
  sorry

theorem part2 (m : ℝ) : part2_statement m :=
by 
  -- Proof here
  sorry

end part1_part2_l36_36852


namespace wicket_keeper_older_than_captain_l36_36795

variables (captain_age : ℕ) (team_avg_age : ℕ) (num_players : ℕ) (remaining_avg_age : ℕ)

def x_older_than_captain (captain_age team_avg_age num_players remaining_avg_age : ℕ) : ℕ :=
  team_avg_age * num_players - remaining_avg_age * (num_players - 2) - 2 * captain_age

theorem wicket_keeper_older_than_captain 
  (captain_age : ℕ) (team_avg_age : ℕ) (num_players : ℕ) (remaining_avg_age : ℕ) 
  (h1 : captain_age = 25) (h2 : team_avg_age = 23) (h3 : num_players = 11) (h4 : remaining_avg_age = 22) :
  x_older_than_captain captain_age team_avg_age num_players remaining_avg_age = 5 :=
by sorry

end wicket_keeper_older_than_captain_l36_36795


namespace solve_for_x_l36_36861

theorem solve_for_x (x : ℝ) (h : 144 / 0.144 = 14.4 / x) : x = 0.0144 := 
by
  sorry

end solve_for_x_l36_36861


namespace find_e_l36_36643

theorem find_e (b e : ℝ) (f g : ℝ → ℝ)
    (h1 : ∀ x, f x = 5 * x + b)
    (h2 : ∀ x, g x = b * x + 3)
    (h3 : ∀ x, f (g x) = 15 * x + e) : e = 18 :=
by
  sorry

end find_e_l36_36643


namespace basin_more_than_tank2_l36_36413

/-- Define the water volumes in milliliters -/
def volume_bottle1 : ℕ := 1000 -- 1 liter = 1000 milliliters
def volume_bottle2 : ℕ := 400  -- 400 milliliters
def volume_tank : ℕ := 2800    -- 2800 milliliters
def volume_basin : ℕ := volume_bottle1 + volume_bottle2 + volume_tank -- total volume in basin
def volume_tank2 : ℕ := 4000 + 100 -- 4 liters 100 milliliters tank

/-- Theorem: The basin can hold 100 ml more water than the 4-liter 100-milliliter tank -/
theorem basin_more_than_tank2 : volume_basin = volume_tank2 + 100 :=
by
  -- This is where the proof would go, but it is not required for this exercise
  sorry

end basin_more_than_tank2_l36_36413


namespace nonneg_int_solutions_eq_binom_l36_36656

-- Definitions for the conditions
variables (k n : ℕ)

-- The statement to be proven
theorem nonneg_int_solutions_eq_binom (h_k_pos : k > 0) (h_n_pos : n > 0) :
  (∃ (x : Fin k → ℕ), (∑ i, x i) = n) ↔ Nat.choose (n+k-1) (k-1) := 
sorry

end nonneg_int_solutions_eq_binom_l36_36656


namespace rain_at_least_once_l36_36830

theorem rain_at_least_once (p : ℚ) (h : p = 3/4) : 
    (1 - (1 - p)^4) = 255/256 :=
by
  sorry

end rain_at_least_once_l36_36830


namespace find_k_l36_36908

theorem find_k (a : ℕ → ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h_nz : ∀ n, S n = n ^ 2 - a n) 
  (hSk : 1 < S k ∧ S k < 9) :
  k = 2 := 
sorry

end find_k_l36_36908


namespace find_symbols_l36_36227

theorem find_symbols (x y otimes oplus : ℝ) 
  (h1 : x + otimes * y = 3) 
  (h2 : 3 * x - otimes * y = 1) 
  (h3 : x = oplus) 
  (h4 : y = 1) : 
  otimes = 2 ∧ oplus = 1 := 
by
  sorry

end find_symbols_l36_36227


namespace correct_time_l36_36623

-- Define the observed times on the clocks
def time1 := 14 * 60 + 54  -- 14:54 in minutes
def time2 := 14 * 60 + 57  -- 14:57 in minutes
def time3 := 15 * 60 + 2   -- 15:02 in minutes
def time4 := 15 * 60 + 3   -- 15:03 in minutes

-- Define the inaccuracies of the clocks
def inaccuracy1 := 2  -- First clock off by 2 minutes
def inaccuracy2 := 3  -- Second clock off by 3 minutes
def inaccuracy3 := -4  -- Third clock off by 4 minutes
def inaccuracy4 := -5  -- Fourth clock off by 5 minutes

-- State that given these conditions, the correct time is 14:58
theorem correct_time : ∃ (T : Int), 
  (time1 + inaccuracy1 = T) ∧
  (time2 + inaccuracy2 = T) ∧
  (time3 + inaccuracy3 = T) ∧
  (time4 + inaccuracy4 = T) ∧
  (T = 14 * 60 + 58) :=
by
  sorry

end correct_time_l36_36623


namespace difference_max_min_y_l36_36121

theorem difference_max_min_y {total_students : ℕ} (initial_yes_pct initial_no_pct final_yes_pct final_no_pct : ℝ)
  (initial_conditions : initial_yes_pct = 0.4 ∧ initial_no_pct = 0.6)
  (final_conditions : final_yes_pct = 0.8 ∧ final_no_pct = 0.2) :
  ∃ (min_change max_change : ℝ), max_change - min_change = 0.2 := by
  sorry

end difference_max_min_y_l36_36121


namespace total_amount_shared_l36_36192

-- Define the amounts for Ken and Tony based on the conditions
def ken_amt : ℤ := 1750
def tony_amt : ℤ := 2 * ken_amt

-- The proof statement that the total amount shared is $5250
theorem total_amount_shared : ken_amt + tony_amt = 5250 :=
by 
  sorry

end total_amount_shared_l36_36192


namespace part1_part2_l36_36607

noncomputable def f (ω x : ℝ) : ℝ := 4 * ((Real.sin (ω * x - Real.pi / 4)) * (Real.cos (ω * x)))

noncomputable def g (α : ℝ) : ℝ := 2 * (Real.sin (α - Real.pi / 6)) - Real.sqrt 2

theorem part1 (ω : ℝ) (x : ℝ) (hω : 0 < ω ∧ ω < 2) (hx : f ω (Real.pi / 4) = Real.sqrt 2) : 
  ∃ T > 0, ∀ x, f ω (x + T) = f ω x :=
sorry

theorem part2 (α : ℝ) (hα: 0 < α ∧ α < Real.pi / 2) (h : g α = 4 / 3 - Real.sqrt 2) : 
  Real.cos α = (Real.sqrt 15 - 2) / 6 :=
sorry

end part1_part2_l36_36607


namespace comparison_of_neg_square_roots_l36_36423

noncomputable def compare_square_roots : Prop :=
  -2 * Real.sqrt 11 > -3 * Real.sqrt 5

theorem comparison_of_neg_square_roots : compare_square_roots :=
by
  -- Omitting the proof details
  sorry

end comparison_of_neg_square_roots_l36_36423


namespace sufficient_but_not_necessary_condition_not_neccessary_condition_l36_36442

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  ((x + 3)^2 + (y - 4)^2 = 0) → ((x + 3) * (y - 4) = 0) :=
by { sorry }

theorem not_neccessary_condition (x y : ℝ) :
  ((x + 3) * (y - 4) = 0) ↔ ((x + 3)^2 + (y - 4)^2 = 0) :=
by { sorry }

end sufficient_but_not_necessary_condition_not_neccessary_condition_l36_36442


namespace fgh_supermarkets_in_us_more_than_canada_l36_36967

theorem fgh_supermarkets_in_us_more_than_canada
  (total_supermarkets : ℕ)
  (us_supermarkets : ℕ)
  (canada_supermarkets : ℕ)
  (h1 : total_supermarkets = 70)
  (h2 : us_supermarkets = 42)
  (h3 : us_supermarkets + canada_supermarkets = total_supermarkets):
  us_supermarkets - canada_supermarkets = 14 :=
by
  sorry

end fgh_supermarkets_in_us_more_than_canada_l36_36967


namespace jacob_ate_five_pies_l36_36940

theorem jacob_ate_five_pies (weight_hot_dog weight_burger weight_pie noah_burgers mason_hotdogs_total_weight : ℕ)
    (H1 : weight_hot_dog = 2)
    (H2 : weight_burger = 5)
    (H3 : weight_pie = 10)
    (H4 : noah_burgers = 8)
    (H5 : mason_hotdogs_total_weight = 30)
    (H6 : ∀ x, 3 * x = (mason_hotdogs_total_weight / weight_hot_dog)) :
    (∃ y, y = (mason_hotdogs_total_weight / weight_hot_dog / 3) ∧ y = 5) :=
by
  sorry

end jacob_ate_five_pies_l36_36940


namespace intersection_A_B_eq_C_l36_36319

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
noncomputable def C : Set ℝ := {0, 1, 2}

theorem intersection_A_B_eq_C : (A : Set ℝ) ∩ B = C :=
by {
  sorry
}

end intersection_A_B_eq_C_l36_36319


namespace breakfast_cost_l36_36587

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3

def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2

def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

theorem breakfast_cost :
  muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  + muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cup = 17 :=
by
  -- skipping proof
  sorry

end breakfast_cost_l36_36587


namespace find_angle_A_range_area_of_triangle_l36_36473

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {S : ℝ}

theorem find_angle_A (h1 : b^2 + c^2 = a^2 - b * c) : A = (2 : ℝ) * Real.pi / 3 :=
by sorry

theorem range_area_of_triangle (h1 : b^2 + c^2 = a^2 - b * c)
(h2 : b * Real.sin A = 4 * Real.sin B) 
(h3 : Real.log b + Real.log c ≥ 1 - 2 * Real.cos (B + C)) 
(h4 : A = (2 : ℝ) * Real.pi / 3) :
(Real.sqrt 3 / 4 : ℝ) ≤ (1 / 2) * b * c * Real.sin A ∧
(1 / 2) * b * c * Real.sin A ≤ (4 * Real.sqrt 3 / 3 : ℝ) :=
by sorry

end find_angle_A_range_area_of_triangle_l36_36473


namespace correct_total_distance_l36_36191

theorem correct_total_distance (km_to_m : 3.5 * 1000 = 3500) (add_m : 3500 + 200 = 3700) : 
  3.5 * 1000 + 200 = 3700 :=
by
  -- The proof would be filled here.
  sorry

end correct_total_distance_l36_36191
