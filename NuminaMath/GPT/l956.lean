import Mathlib

namespace minimum_value_of_function_l956_95622

-- Define the function y = 2x + 1/(x - 1) with the constraint x > 1
noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / (x - 1)

-- Prove that the minimum value of the function for x > 1 is 2√2 + 2
theorem minimum_value_of_function : 
  ∃ x : ℝ, x > 1 ∧ ∀ y : ℝ, (y = f x) → y ≥ 2 * Real.sqrt 2 + 2 := 
  sorry

end minimum_value_of_function_l956_95622


namespace perimeter_of_specific_figure_l956_95692

-- Define the grid size and additional column properties as given in the problem
structure Figure :=
  (rows : ℕ)
  (cols : ℕ)
  (additionalCols : ℕ)
  (additionalRows : ℕ)

-- The specific figure properties from the problem statement
def specificFigure : Figure := {
  rows := 3,
  cols := 4,
  additionalCols := 1,
  additionalRows := 2
}

-- Define the perimeter computation
def computePerimeter (fig : Figure) : ℕ :=
  2 * (fig.rows + fig.cols + fig.additionalCols) + fig.additionalRows

theorem perimeter_of_specific_figure : computePerimeter specificFigure = 13 :=
by
  sorry

end perimeter_of_specific_figure_l956_95692


namespace Berry_read_pages_thursday_l956_95603

theorem Berry_read_pages_thursday :
  ∀ (pages_per_day : ℕ) (pages_sunday : ℕ) (pages_monday : ℕ) (pages_tuesday : ℕ) 
    (pages_wednesday : ℕ) (pages_friday : ℕ) (pages_saturday : ℕ),
    (pages_per_day = 50) →
    (pages_sunday = 43) →
    (pages_monday = 65) →
    (pages_tuesday = 28) →
    (pages_wednesday = 0) →
    (pages_friday = 56) →
    (pages_saturday = 88) →
    pages_sunday + pages_monday + pages_tuesday +
    pages_wednesday + pages_friday + pages_saturday + x = 350 →
    x = 70 := by
  sorry

end Berry_read_pages_thursday_l956_95603


namespace cost_apples_l956_95601

def total_cost := 42
def cost_bananas := 12
def cost_bread := 9
def cost_milk := 7

theorem cost_apples:
  total_cost - (cost_bananas + cost_bread + cost_milk) = 14 :=
by
  sorry

end cost_apples_l956_95601


namespace sum_of_multiples_is_even_l956_95600

theorem sum_of_multiples_is_even (a b : ℤ) (h1 : ∃ m : ℤ, a = 4 * m) (h2 : ∃ n : ℤ, b = 6 * n) : Even (a + b) :=
sorry

end sum_of_multiples_is_even_l956_95600


namespace base_length_of_prism_l956_95625

theorem base_length_of_prism (V : ℝ) (hV : V = 36 * Real.pi) : ∃ (AB : ℝ), AB = 3 * Real.sqrt 3 :=
by
  sorry

end base_length_of_prism_l956_95625


namespace remainder_is_6910_l956_95627

def polynomial (x : ℝ) : ℝ := 5 * x^7 - 3 * x^6 - 8 * x^5 + 3 * x^3 + 5 * x^2 - 20

def divisor (x : ℝ) : ℝ := 3 * x - 9

theorem remainder_is_6910 : polynomial 3 = 6910 := by
  sorry

end remainder_is_6910_l956_95627


namespace solution_set_16_sin_pi_x_cos_pi_x_l956_95649

theorem solution_set_16_sin_pi_x_cos_pi_x (x : ℝ) :
  (x = 1 / 4 ∨ x = -1 / 4) ↔ 16 * Real.sin (π * x) * Real.cos (π * x) = 16 * x + 1 / x :=
sorry

end solution_set_16_sin_pi_x_cos_pi_x_l956_95649


namespace mean_median_mode_relation_l956_95643

-- Defining the data set of the number of fish caught in twelve outings.
def fish_catches : List ℕ := [3, 0, 2, 2, 1, 5, 3, 0, 1, 4, 3, 3]

-- Proof statement to show the relationship among mean, median and mode.
theorem mean_median_mode_relation (hs : fish_catches = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5]) :
  let mean := (fish_catches.sum : ℚ) / fish_catches.length
  let median := (fish_catches.nthLe 5 sorry + fish_catches.nthLe 6 sorry : ℚ) / 2
  let mode := 3
  mean < median ∧ median < mode := by
  -- Placeholder for the proof. Details are skipped here.
  sorry

end mean_median_mode_relation_l956_95643


namespace max_min_of_f_l956_95677

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (2 * Real.pi + x) + 
  Real.sqrt 3 * Real.cos (2 * Real.pi - x) -
  Real.sin (2013 * Real.pi + Real.pi / 6)

theorem max_min_of_f : 
  - (Real.pi / 2) ≤ x ∧ x ≤ Real.pi / 2 →
  (-1 / 2) ≤ f x ∧ f x ≤ 5 / 2 :=
sorry

end max_min_of_f_l956_95677


namespace find_c_l956_95626

theorem find_c (x c : ℚ) (h1 : 3 * x + 5 = 1) (h2 : c * x + 8 = 6) : c = 3 / 2 := 
sorry

end find_c_l956_95626


namespace find_max_marks_l956_95658

variable (marks_scored : ℕ) -- 212
variable (shortfall : ℕ) -- 22
variable (pass_percentage : ℝ) -- 0.30

theorem find_max_marks (h_marks : marks_scored = 212) 
                       (h_short : shortfall = 22) 
                       (h_pass : pass_percentage = 0.30) : 
  ∃ M : ℝ, M = 780 :=
by {
  sorry
}

end find_max_marks_l956_95658


namespace tax_difference_is_correct_l956_95645

-- Define the original price and discount rate as constants
def original_price : ℝ := 50
def discount_rate : ℝ := 0.10

-- Define the state and local sales tax rates as constants
def state_sales_tax_rate : ℝ := 0.075
def local_sales_tax_rate : ℝ := 0.07

-- Calculate the discounted price
def discounted_price : ℝ := original_price * (1 - discount_rate)

-- Calculate state and local sales taxes after discount
def state_sales_tax : ℝ := discounted_price * state_sales_tax_rate
def local_sales_tax : ℝ := discounted_price * local_sales_tax_rate

-- Calculate the difference between state and local sales taxes
def tax_difference : ℝ := state_sales_tax - local_sales_tax

-- The proof to show that the difference is 0.225
theorem tax_difference_is_correct : tax_difference = 0.225 := by
  sorry

end tax_difference_is_correct_l956_95645


namespace student_C_has_sweetest_water_l956_95616

-- Define concentrations for each student
def concentration_A : ℚ := 35 / 175 * 100
def concentration_B : ℚ := 45 / 175 * 100
def concentration_C : ℚ := 65 / 225 * 100

-- Prove that Student C has the highest concentration
theorem student_C_has_sweetest_water :
  concentration_C > concentration_B ∧ concentration_C > concentration_A :=
by
  -- By direct calculation from the provided conditions
  sorry

end student_C_has_sweetest_water_l956_95616


namespace gain_percentage_is_30_l956_95641

def sellingPrice : ℕ := 195
def gain : ℕ := 45
def costPrice : ℕ := sellingPrice - gain

def gainPercentage : ℚ := (gain : ℚ) / (costPrice : ℚ) * 100

theorem gain_percentage_is_30 :
  gainPercentage = 30 := 
sorry

end gain_percentage_is_30_l956_95641


namespace jacket_final_price_l956_95624

theorem jacket_final_price :
    let initial_price := 150
    let first_discount := 0.30
    let second_discount := 0.10
    let coupon := 10
    let tax := 0.05
    let price_after_first_discount := initial_price * (1 - first_discount)
    let price_after_second_discount := price_after_first_discount * (1 - second_discount)
    let price_after_coupon := price_after_second_discount - coupon
    let final_price := price_after_coupon * (1 + tax)
    final_price = 88.725 :=
by
  sorry

end jacket_final_price_l956_95624


namespace side_length_irrational_l956_95639

theorem side_length_irrational (s : ℝ) (h : s^2 = 3) : ¬∃ (r : ℚ), s = r := by
  sorry

end side_length_irrational_l956_95639


namespace cost_per_bag_l956_95656

theorem cost_per_bag (C : ℝ)
  (total_bags : ℕ := 20)
  (price_per_bag_original : ℝ := 6)
  (sold_original : ℕ := 15)
  (price_per_bag_discounted : ℝ := 4)
  (sold_discounted : ℕ := 5)
  (net_profit : ℝ := 50) :
  sold_original * price_per_bag_original + sold_discounted * price_per_bag_discounted - net_profit = total_bags * C →
  C = 3 :=
by
  intros h
  sorry

end cost_per_bag_l956_95656


namespace num_students_in_research_study_group_prob_diff_classes_l956_95695

-- Define the number of students in each class and the number of students selected from class (2)
def num_students_class1 : ℕ := 18
def num_students_class2 : ℕ := 27
def selected_from_class2 : ℕ := 3

-- Prove the number of students in the research study group
theorem num_students_in_research_study_group : 
  (∃ (m : ℕ), (m / 18 = 3 / 27) ∧ (m + selected_from_class2 = 5)) := 
by
  sorry

-- Prove the probability that the students speaking in both activities come from different classes
theorem prob_diff_classes : 
  (12 / 25 = 12 / 25) :=
by
  sorry

end num_students_in_research_study_group_prob_diff_classes_l956_95695


namespace intersection_of_sets_l956_95632

def M (x : ℝ) : Prop := (x - 2) / (x - 3) < 0
def N (x : ℝ) : Prop := Real.log (x - 2) / Real.log (1 / 2) ≥ 1 

theorem intersection_of_sets : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 2 < x ∧ x ≤ 5 / 2} := by
  sorry

end intersection_of_sets_l956_95632


namespace average_buns_per_student_l956_95678

theorem average_buns_per_student (packages_class1 packages_class2 packages_class3 packages_class4 : ℕ)
    (buns_per_package students_per_class stale_buns uneaten_buns : ℕ)
    (h1 : packages_class1 = 20)
    (h2 : packages_class2 = 25)
    (h3 : packages_class3 = 30)
    (h4 : packages_class4 = 35)
    (h5 : buns_per_package = 8)
    (h6 : students_per_class = 30)
    (h7 : stale_buns = 16)
    (h8 : uneaten_buns = 20) :
  let total_buns_class1 := packages_class1 * buns_per_package
  let total_buns_class2 := packages_class2 * buns_per_package
  let total_buns_class3 := packages_class3 * buns_per_package
  let total_buns_class4 := packages_class4 * buns_per_package
  let total_uneaten_buns := stale_buns + uneaten_buns
  let uneaten_buns_per_class := total_uneaten_buns / 4
  let remaining_buns_class1 := total_buns_class1 - uneaten_buns_per_class
  let remaining_buns_class2 := total_buns_class2 - uneaten_buns_per_class
  let remaining_buns_class3 := total_buns_class3 - uneaten_buns_per_class
  let remaining_buns_class4 := total_buns_class4 - uneaten_buns_per_class
  let avg_buns_class1 := remaining_buns_class1 / students_per_class
  let avg_buns_class2 := remaining_buns_class2 / students_per_class
  let avg_buns_class3 := remaining_buns_class3 / students_per_class
  let avg_buns_class4 := remaining_buns_class4 / students_per_class
  avg_buns_class1 = 5 ∧ avg_buns_class2 = 6 ∧ avg_buns_class3 = 7 ∧ avg_buns_class4 = 9 :=
by
  sorry

end average_buns_per_student_l956_95678


namespace value_of_D_l956_95681

theorem value_of_D (E F D : ℕ) (cond1 : E + F + D = 15) (cond2 : F + E = 11) : D = 4 := 
by
  sorry

end value_of_D_l956_95681


namespace num_cats_l956_95610

-- Definitions based on conditions
variables (C S K Cap : ℕ)
variable (heads : ℕ) (legs : ℕ)

-- Conditions as equations
axiom heads_eq : C + S + K + Cap = 16
axiom legs_eq : 4 * C + 2 * S + 2 * K + 1 * Cap = 41

-- Given values from the problem
axiom K_val : K = 1
axiom Cap_val : Cap = 1

-- The proof goal in terms of satisfying the number of cats
theorem num_cats : C = 5 :=
by
  sorry

end num_cats_l956_95610


namespace polynomial_solutions_l956_95664

-- Define the type of the polynomials and statement of the problem
def P1 (x : ℝ) : ℝ := x
def P2 (x : ℝ) : ℝ := x^2 + 1
def P3 (x : ℝ) : ℝ := x^4 + 2*x^2 + 2

theorem polynomial_solutions :
  (∀ x : ℝ, P1 (x^2 + 1) = P1 x^2 + 1) ∧
  (∀ x : ℝ, P2 (x^2 + 1) = P2 x^2 + 1) ∧
  (∀ x : ℝ, P3 (x^2 + 1) = P3 x^2 + 1) :=
by
  -- Proof will go here
  sorry

end polynomial_solutions_l956_95664


namespace smallest_m_l956_95655

theorem smallest_m (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  ∃ m, (∀ (a b c : ℝ), a + b + c = 1 → 0 < a → 0 < b → 0 < c → m * (a ^ 3 + b ^ 3 + c ^ 3) ≥ 6 * (a ^ 2 + b ^ 2 + c ^ 2) + 1) ↔ m = 27 :=
by
  sorry

end smallest_m_l956_95655


namespace math_problem_l956_95682

noncomputable def x : ℝ := 24

theorem math_problem : ∀ (x : ℝ), x = 3/8 * x + 15 → x = 24 := 
by 
  intro x
  intro h
  sorry

end math_problem_l956_95682


namespace cost_of_gasoline_l956_95609

def odometer_initial : ℝ := 85120
def odometer_final : ℝ := 85150
def fuel_efficiency : ℝ := 30
def price_per_gallon : ℝ := 4.25

theorem cost_of_gasoline : 
  ((odometer_final - odometer_initial) / fuel_efficiency) * price_per_gallon = 4.25 := 
by 
  sorry

end cost_of_gasoline_l956_95609


namespace no_bounded_sequences_at_least_one_gt_20_l956_95618

variable (x y z : ℕ → ℝ)
variable (x1 y1 z1 : ℝ)
variable (h0 : x1 > 0) (h1 : y1 > 0) (h2 : z1 > 0)
variable (h3 : ∀ n, x (n + 1) = y n + (1 / z n))
variable (h4 : ∀ n, y (n + 1) = z n + (1 / x n))
variable (h5 : ∀ n, z (n + 1) = x n + (1 / y n))

-- Part (a)
theorem no_bounded_sequences : (∀ n, x n > 0) ∧ (∀ n, y n > 0) ∧ (∀ n, z n > 0) → ¬ (∃ M, ∀ n, x n < M ∧ y n < M ∧ z n < M) :=
sorry

-- Part (b)
theorem at_least_one_gt_20 : x 1 = x1 ∧ y 1 = y1 ∧ z 1 = z1 → x 200 > 20 ∨ y 200 > 20 ∨ z 200 > 20 :=
sorry

end no_bounded_sequences_at_least_one_gt_20_l956_95618


namespace maria_baggies_l956_95680

-- Definitions of the conditions
def total_cookies (chocolate_chip : Nat) (oatmeal : Nat) : Nat :=
  chocolate_chip + oatmeal

def cookies_per_baggie : Nat :=
  3

def number_of_baggies (total_cookies : Nat) (cookies_per_baggie : Nat) : Nat :=
  total_cookies / cookies_per_baggie

-- Proof statement
theorem maria_baggies :
  number_of_baggies (total_cookies 2 16) cookies_per_baggie = 6 := 
sorry

end maria_baggies_l956_95680


namespace verify_expressions_l956_95630

variable (x y : ℝ)
variable (h : x / y = 5 / 3)

theorem verify_expressions :
  (2 * x + y) / y = 13 / 3 ∧
  y / (y - 2 * x) = 3 / -7 ∧
  (x + y) / x = 8 / 5 ∧
  x / (3 * y) = 5 / 9 ∧
  (x - 2 * y) / y = -1 / 3 := by
sorry

end verify_expressions_l956_95630


namespace part1_part2_l956_95689

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |2 * x - 1|

theorem part1 : {x : ℝ | f x ≤ 5} = {x : ℝ | -7 / 4 ≤ x ∧ x ≤ 3 / 4} :=
sorry

theorem part2 (h : ∃ x : ℝ, f x < |m - 2|) : m > 6 ∨ m < -2 :=
sorry

end part1_part2_l956_95689


namespace parabola_find_m_l956_95642

theorem parabola_find_m
  (p m : ℝ) (h_p_pos : p > 0) (h_point_on_parabola : (2 * p * m) = 8)
  (h_chord_length : (m + (2 / m))^2 - m^2 = 7) : m = (2 * Real.sqrt 3) / 3 :=
by sorry

end parabola_find_m_l956_95642


namespace find_pairs_l956_95668

theorem find_pairs (n k : ℕ) (h_pos_n : 0 < n) (h_cond : n! + n = n ^ k) : 
  (n = 2 ∧ k = 2) ∨ (n = 3 ∧ k = 2) ∨ (n = 5 ∧ k = 3) := 
by 
  sorry

end find_pairs_l956_95668


namespace odd_pair_exists_k_l956_95631

theorem odd_pair_exists_k (a b : ℕ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) : 
  ∃ k : ℕ, (2^2018 ∣ b^k - a^2) ∨ (2^2018 ∣ a^k - b^2) := 
sorry

end odd_pair_exists_k_l956_95631


namespace rectangle_side_length_l956_95644

theorem rectangle_side_length (a c : ℝ) (h_ratio : a / c = 3 / 4) (hc : c = 4) : a = 3 :=
by
  sorry

end rectangle_side_length_l956_95644


namespace ratio_of_cats_to_dogs_sold_l956_95607

theorem ratio_of_cats_to_dogs_sold (cats dogs : ℕ) (h1 : cats = 16) (h2 : dogs = 8) :
  (cats : ℚ) / dogs = 2 / 1 :=
by
  sorry

end ratio_of_cats_to_dogs_sold_l956_95607


namespace value_of_x_l956_95648

theorem value_of_x (x : ℕ) : (1 / 16) * (2 ^ 20) = 4 ^ x → x = 8 := by
  sorry

end value_of_x_l956_95648


namespace negation_statement_contrapositive_statement_l956_95652

variable (x y : ℝ)

theorem negation_statement :
  (¬ ((x-1) * (y+2) ≠ 0 → x ≠ 1 ∧ y ≠ -2)) ↔ ((x-1) * (y+2) = 0 → x = 1 ∨ y = -2) :=
by sorry

theorem contrapositive_statement :
  (x = 1 ∨ y = -2) → ((x-1) * (y+2) = 0) :=
by sorry

end negation_statement_contrapositive_statement_l956_95652


namespace count_valid_three_digit_numbers_l956_95684

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, n = 720 ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 → 
    (∀ d ∈ [m / 100, (m / 10) % 10, m % 10], d ∉ [2, 5, 7, 9])) := 
sorry

end count_valid_three_digit_numbers_l956_95684


namespace original_expression_equals_l956_95690

noncomputable def evaluate_expression (a : ℝ) : ℝ :=
  ( (a / (a + 2) + 1 / (a^2 - 4)) / ( (a - 1) / (a + 2) + 1 / (a - 2) ))

theorem original_expression_equals (a : ℝ) (h : a = 2 + Real.sqrt 2) :
  evaluate_expression a = (Real.sqrt 2 + 1) :=
sorry

end original_expression_equals_l956_95690


namespace cubic_identity_l956_95673

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + ac + bc = 40) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 1575 := 
by
  sorry

end cubic_identity_l956_95673


namespace complement_of_M_l956_95691

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x^2 - 2*x > 0 }
def complement (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

theorem complement_of_M :
  complement U M = { x | 0 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end complement_of_M_l956_95691


namespace number_to_add_l956_95650

theorem number_to_add (a b n : ℕ) (h_a : a = 425897) (h_b : b = 456) (h_n : n = 47) : 
  (a + n) % b = 0 :=
by
  rw [h_a, h_b, h_n]
  sorry

end number_to_add_l956_95650


namespace not_divisible_by_1000_pow_m_minus_1_l956_95633

theorem not_divisible_by_1000_pow_m_minus_1 (m : ℕ) : ¬ (1000^m - 1 ∣ 1998^m - 1) :=
sorry

end not_divisible_by_1000_pow_m_minus_1_l956_95633


namespace processing_time_600_parts_l956_95612

theorem processing_time_600_parts :
  ∀ (x: ℕ), x = 600 → (∃ y : ℝ, y = 0.01 * x + 0.5 ∧ y = 6.5) :=
by
  sorry

end processing_time_600_parts_l956_95612


namespace log_one_plus_xsq_lt_xsq_over_one_plus_xsq_l956_95686

theorem log_one_plus_xsq_lt_xsq_over_one_plus_xsq (x : ℝ) (hx : 0 < x) : 
  Real.log (1 + x^2) < x^2 / (1 + x^2) :=
sorry

end log_one_plus_xsq_lt_xsq_over_one_plus_xsq_l956_95686


namespace exists_tiling_5x6_no_gaps_no_tiling_5x6_with_gaps_no_tiling_6x6_l956_95697

/-- There exists a way to completely tile a 5x6 board with dominos without leaving any gaps. -/
theorem exists_tiling_5x6_no_gaps :
  ∃ (tiling : List (Set (Fin 5 × Fin 6))), True := 
sorry

/-- It is not possible to tile a 5x6 board with dominos such that gaps are left. -/
theorem no_tiling_5x6_with_gaps :
  ¬ ∃ (tiling : List (Set (Fin 5 × Fin 6))), False := 
sorry

/-- It is impossible to tile a 6x6 board with dominos. -/
theorem no_tiling_6x6 :
  ¬ ∃ (tiling : List (Set (Fin 6 × Fin 6))), True := 
sorry

end exists_tiling_5x6_no_gaps_no_tiling_5x6_with_gaps_no_tiling_6x6_l956_95697


namespace sum_of_square_roots_l956_95653

theorem sum_of_square_roots :
  (Real.sqrt 1 + Real.sqrt (1 + 2) + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4)) = 
  (1 + Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 10) := 
sorry

end sum_of_square_roots_l956_95653


namespace radius_of_intersection_l956_95634

noncomputable def sphere_radius := 2 * Real.sqrt 17

theorem radius_of_intersection (s : ℝ) 
  (h1 : (3:ℝ)=(3:ℝ)) (h2 : (5:ℝ)=(5:ℝ)) (h3 : (0-3:ℝ)^2 + (5-5:ℝ)^2 + (s-(-8+8))^2 = sphere_radius^2) :
  s = Real.sqrt 59 :=
by
  sorry

end radius_of_intersection_l956_95634


namespace circumference_of_circle_l956_95623

def speed_cyclist1 : ℝ := 7
def speed_cyclist2 : ℝ := 8
def meeting_time : ℝ := 42
def circumference : ℝ := 630

theorem circumference_of_circle :
  (speed_cyclist1 * meeting_time + speed_cyclist2 * meeting_time = circumference) :=
by
  sorry

end circumference_of_circle_l956_95623


namespace decreasing_function_l956_95637

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem decreasing_function (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₂ ≤ 1) : 
  f x₁ > f x₂ :=
by
  -- Proof goes here
  sorry

end decreasing_function_l956_95637


namespace base_7_to_base_10_equiv_l956_95621

theorem base_7_to_base_10_equiv (digits : List ℕ) 
  (h : digits = [5, 4, 3, 2, 1]) : 
  (5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0) = 13539 := 
by 
  sorry

end base_7_to_base_10_equiv_l956_95621


namespace investment_amounts_proof_l956_95620

noncomputable def investment_proof_statement : Prop :=
  let p_investment_first_year := 52000
  let q_investment := (5/4) * p_investment_first_year
  let r_investment := (6/4) * p_investment_first_year;
  let p_investment_second_year := p_investment_first_year + (20/100) * p_investment_first_year;
  (q_investment = 65000) ∧ (r_investment = 78000) ∧ (q_investment = 65000) ∧ (r_investment = 78000)

theorem investment_amounts_proof : investment_proof_statement :=
  by
    sorry

end investment_amounts_proof_l956_95620


namespace Alyssa_next_year_games_l956_95602

theorem Alyssa_next_year_games 
  (games_this_year : ℕ) 
  (games_last_year : ℕ) 
  (total_games : ℕ) 
  (games_up_to_this_year : ℕ)
  (total_up_to_next_year : ℕ) 
  (H1 : games_this_year = 11)
  (H2 : games_last_year = 13)
  (H3 : total_up_to_next_year = 39)
  (H4 : games_up_to_this_year = games_this_year + games_last_year) :
  total_up_to_next_year - games_up_to_this_year = 15 :=
by
  sorry

end Alyssa_next_year_games_l956_95602


namespace number_of_paths_from_C_to_D_l956_95640

-- Define the grid and positions
def C := (0,0)  -- Bottom-left corner
def D := (7,3)  -- Top-right corner
def gridWidth : ℕ := 7
def gridHeight : ℕ := 3

-- Define the binomial coefficient function
-- Note: Lean already has binomial coefficient defined in Mathlib, use Nat.choose for that

-- The statement to prove
theorem number_of_paths_from_C_to_D : Nat.choose (gridWidth + gridHeight) gridHeight = 120 :=
by
  sorry

end number_of_paths_from_C_to_D_l956_95640


namespace dawn_lemonade_price_l956_95670

theorem dawn_lemonade_price (x : ℕ) : 
  (10 * 25) = (8 * x) + 26 → x = 28 :=
by 
  sorry

end dawn_lemonade_price_l956_95670


namespace five_fourths_of_x_over_3_l956_95651

theorem five_fourths_of_x_over_3 (x : ℚ) : (5/4) * (x/3) = 5 * x / 12 :=
by
  sorry

end five_fourths_of_x_over_3_l956_95651


namespace closest_fraction_l956_95679

theorem closest_fraction :
  let won_france := (23 : ℝ) / 120
  let fractions := [ (1 : ℝ) / 4, (1 : ℝ) / 5, (1 : ℝ) / 6, (1 : ℝ) / 7, (1 : ℝ) / 8 ]
  ∃ closest : ℝ, closest ∈ fractions ∧ ∀ f ∈ fractions, abs (won_france - closest) ≤ abs (won_france - f)  :=
  sorry

end closest_fraction_l956_95679


namespace original_price_eq_36_l956_95667

-- Definitions for the conditions
def first_cup_price (x : ℕ) : ℕ := x
def second_cup_price (x : ℕ) : ℕ := x / 2
def third_cup_price : ℕ := 3
def total_cost (x : ℕ) : ℕ := x + (x / 2) + third_cup_price
def average_price (total : ℕ) : ℕ := total / 3

-- The proof statement
theorem original_price_eq_36 (x : ℕ) (h : total_cost x = 57) : x = 36 :=
  sorry

end original_price_eq_36_l956_95667


namespace negation_universal_proposition_l956_95614

theorem negation_universal_proposition {x : ℝ} : 
  (¬ ∀ x : ℝ, x^2 ≥ 2) ↔ (∃ x : ℝ, x^2 < 2) := 
sorry

end negation_universal_proposition_l956_95614


namespace sum_of_tangents_l956_95606

theorem sum_of_tangents (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
    (h_tan_α : Real.tan α = 2) (h_tan_β : Real.tan β = 3) : α + β = 3 * π / 4 :=
by
  sorry

end sum_of_tangents_l956_95606


namespace initial_number_of_employees_l956_95629

variables (E : ℕ)
def hourly_rate : ℕ := 12
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 5
def weeks_per_month : ℕ := 4
def extra_employees : ℕ := 200
def total_payroll : ℕ := 1680000

-- Total hours worked by each employee per month
def monthly_hours_per_employee : ℕ := hours_per_day * days_per_week * weeks_per_month

-- Monthly salary per employee
def monthly_salary_per_employee : ℕ := monthly_hours_per_employee * hourly_rate

-- Condition expressing the constraint given in the problem
def payroll_equation : Prop :=
  (E + extra_employees) * monthly_salary_per_employee = total_payroll

-- The statement we are proving
theorem initial_number_of_employees :
  payroll_equation E → E = 500 :=
by
  -- Proof not required
  intros
  sorry

end initial_number_of_employees_l956_95629


namespace simplify_and_evaluate_l956_95696

-- Math proof problem
theorem simplify_and_evaluate :
  ∀ (a : ℤ), a = -1 →
  (2 - a)^2 - (1 + a) * (a - 1) - a * (a - 3) = 5 :=
by
  intros a ha
  sorry

end simplify_and_evaluate_l956_95696


namespace staff_discount_price_l956_95611

theorem staff_discount_price (d : ℝ) : (d - 0.15*d) * 0.90 = 0.765 * d :=
by
  have discount1 : d - 0.15 * d = d * 0.85 :=
    by ring
  have discount2 : (d * 0.85) * 0.90 = d * (0.85 * 0.90) :=
    by ring
  have final_price : d * (0.85 * 0.90) = d * 0.765 :=
    by norm_num
  rw [discount1, discount2, final_price]
  sorry

end staff_discount_price_l956_95611


namespace union_example_l956_95646

open Set

variable (A B : Set ℤ)
variable (AB : Set ℤ)

theorem union_example (hA : A = {-3, 1, 2})
                      (hB : B = {0, 1, 2, 3}) :
                      A ∪ B = {-3, 0, 1, 2, 3} :=
by
  rw [hA, hB]
  ext
  simp
  sorry

end union_example_l956_95646


namespace interest_rate_eq_ten_l956_95671

theorem interest_rate_eq_ten (R : ℝ) (P : ℝ) (SI CI : ℝ) :
  P = 1400 ∧
  SI = 14 * R ∧
  CI = 1400 * ((1 + R / 200) ^ 2 - 1) ∧
  CI - SI = 3.50 → 
  R = 10 :=
by
  sorry

end interest_rate_eq_ten_l956_95671


namespace cost_of_football_and_basketball_max_number_of_basketballs_l956_95699

-- Problem 1: Cost of one football and one basketball
theorem cost_of_football_and_basketball (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 310) 
  (h2 : 2 * x + 5 * y = 500) : 
  x = 50 ∧ y = 80 :=
sorry

-- Problem 2: Maximum number of basketballs
theorem max_number_of_basketballs (x : ℝ) 
  (h1 : 50 * (96 - x) + 80 * x ≤ 5800) 
  (h2 : x ≥ 0) 
  (h3 : x ≤ 96) : 
  x ≤ 33 :=
sorry

end cost_of_football_and_basketball_max_number_of_basketballs_l956_95699


namespace find_range_of_m_l956_95628

def equation1 (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 = 0 → x < 0

def equation2 (m : ℝ) : Prop :=
  ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 → false

theorem find_range_of_m (m : ℝ) (h1 : equation1 m → m > 2) (h2 : equation2 m → 1 < m ∧ m < 3) :
  (equation1 m ∨ equation2 m) ∧ ¬(equation1 m ∧ equation2 m) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  sorry

end find_range_of_m_l956_95628


namespace greatest_ratio_AB_CD_on_circle_l956_95659

/-- The statement proving the greatest possible value of the ratio AB/CD for points A, B, C, D lying on the 
circle x^2 + y^2 = 16 with integer coordinates and unequal distances AB and CD is sqrt 10 / 3. -/
theorem greatest_ratio_AB_CD_on_circle :
  ∀ (A B C D : ℤ × ℤ), A ≠ B → C ≠ D → 
  A.1^2 + A.2^2 = 16 → B.1^2 + B.2^2 = 16 → 
  C.1^2 + C.2^2 = 16 → D.1^2 + D.2^2 = 16 → 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let CD := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let ratio := AB / CD
  AB ≠ CD →
  ratio ≤ Real.sqrt 10 / 3 :=
sorry

end greatest_ratio_AB_CD_on_circle_l956_95659


namespace longest_chord_in_circle_l956_95693

theorem longest_chord_in_circle {O : Type} (radius : ℝ) (h_radius : radius = 3) :
  ∃ d, d = 6 ∧ ∀ chord, chord <= d :=
by sorry

end longest_chord_in_circle_l956_95693


namespace find_abc_sum_l956_95672

theorem find_abc_sum {U : Type} 
  (a b c : ℕ)
  (ha : a = 26)
  (hb : b = 1)
  (hc : c = 32)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) :
  a + b + c = 59 :=
by
  sorry

end find_abc_sum_l956_95672


namespace student_count_before_new_student_l956_95654

variable {W : ℝ} -- total weight of students before the new student joined
variable {n : ℕ} -- number of students before the new student joined
variable {W_new : ℝ} -- total weight including the new student
variable {n_new : ℕ} -- number of students including the new student

theorem student_count_before_new_student 
  (h1 : W = n * 28) 
  (h2 : W_new = W + 7) 
  (h3 : n_new = n + 1) 
  (h4 : W_new / n_new = 27.3) : n = 29 := 
by
  sorry

end student_count_before_new_student_l956_95654


namespace annual_interest_correct_l956_95666

-- Define the conditions
def Rs_total : ℝ := 3400
def P1 : ℝ := 1300
def P2 : ℝ := Rs_total - P1
def Rate1 : ℝ := 0.03
def Rate2 : ℝ := 0.05

-- Define the interests
def Interest1 : ℝ := P1 * Rate1
def Interest2 : ℝ := P2 * Rate2

-- The total interest
def Total_Interest : ℝ := Interest1 + Interest2

-- The theorem to prove
theorem annual_interest_correct :
  Total_Interest = 144 :=
by
  sorry

end annual_interest_correct_l956_95666


namespace find_points_PQ_l956_95698

-- Define the points A, B, M, and E in 3D space
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := ⟨0, 0, 0⟩
def B : Point := ⟨10, 0, 0⟩
def M : Point := ⟨5, 5, 0⟩
def E : Point := ⟨0, 0, 10⟩

-- Define the lines AB and EM
def line_AB (t : ℝ) : Point := ⟨10 * t, 0, 0⟩
def line_EM (s : ℝ) : Point := ⟨5 * s, 5 * s, 10 - 10 * s⟩

-- Define the points P and Q
def P (t : ℝ) : Point := line_AB t
def Q (s : ℝ) : Point := line_EM s

-- Define the distance function in 3D space
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

-- The main theorem
theorem find_points_PQ (t s : ℝ) (h1 : t = 0.4) (h2 : s = 0.8) :
  (P t = ⟨4, 0, 0⟩) ∧ (Q s = ⟨4, 4, 2⟩) ∧
  (distance (P t) (Q s) = distance (line_AB 0.4) (line_EM 0.8)) :=
by
  sorry

end find_points_PQ_l956_95698


namespace sum_nth_beginning_end_l956_95663

theorem sum_nth_beginning_end (n : ℕ) (F L : ℤ) (M : ℤ) 
  (consecutive : ℤ → ℤ) (median : M = 60) 
  (median_formula : M = (F + L) / 2) :
  n = n → F + L = 120 :=
by
  sorry

end sum_nth_beginning_end_l956_95663


namespace part1_part2_part3_l956_95688

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

end part1_part2_part3_l956_95688


namespace entire_hike_length_l956_95617

-- Definitions directly from the conditions in part a)
def tripp_backpack_weight : ℕ := 25
def charlotte_backpack_weight : ℕ := tripp_backpack_weight - 7
def miles_hiked_first_day : ℕ := 9
def miles_left_to_hike : ℕ := 27

-- Theorem proving the entire hike length
theorem entire_hike_length :
  miles_hiked_first_day + miles_left_to_hike = 36 :=
by
  sorry

end entire_hike_length_l956_95617


namespace crayons_end_of_school_year_l956_95669

-- Definitions based on conditions
def crayons_after_birthday : Float := 479.0
def total_crayons_now : Float := 613.0

-- The mathematically equivalent proof problem statement
theorem crayons_end_of_school_year : (total_crayons_now - crayons_after_birthday = 134.0) :=
by
  sorry

end crayons_end_of_school_year_l956_95669


namespace small_load_clothing_count_l956_95608

def initial_clothes : ℕ := 36
def first_load_clothes : ℕ := 18
def remaining_clothes := initial_clothes - first_load_clothes
def small_load_clothes := remaining_clothes / 2

theorem small_load_clothing_count : 
  small_load_clothes = 9 :=
by
  sorry

end small_load_clothing_count_l956_95608


namespace difference_of_interchanged_digits_l956_95613

theorem difference_of_interchanged_digits {x y : ℕ} (h : x - y = 4) :
  (10 * x + y) - (10 * y + x) = 36 :=
by sorry

end difference_of_interchanged_digits_l956_95613


namespace number_of_buses_in_month_l956_95660

-- Given conditions
def weekday_buses := 36
def saturday_buses := 24
def sunday_holiday_buses := 12
def num_weekdays := 18
def num_saturdays := 4
def num_sundays_holidays := 6

-- Statement to prove
theorem number_of_buses_in_month : 
  num_weekdays * weekday_buses + num_saturdays * saturday_buses + num_sundays_holidays * sunday_holiday_buses = 816 := 
by 
  sorry

end number_of_buses_in_month_l956_95660


namespace percentage_discount_total_amount_paid_l956_95683

variable (P Q : ℝ)

theorem percentage_discount (h₁ : P > Q) (h₂ : Q > 0) :
  100 * ((P - Q) / P) = 100 * (P - Q) / P :=
sorry

theorem total_amount_paid (h₁ : P > Q) (h₂ : Q > 0) :
  10 * Q = 10 * Q :=
sorry

end percentage_discount_total_amount_paid_l956_95683


namespace percentage_problem_l956_95605

theorem percentage_problem 
  (number : ℕ)
  (h1 : number = 6400)
  (h2 : 5 * number / 100 = 20 * 650 / 100 + 190) : 
  20 = 20 :=
by 
  sorry

end percentage_problem_l956_95605


namespace inradius_inequality_l956_95647

/-- Given a point P inside the triangle ABC, where da, db, and dc are the distances from P to the sides BC, CA, and AB respectively,
 and r is the inradius of the triangle ABC, prove the inequality -/
theorem inradius_inequality (a b c da db dc : ℝ) (r : ℝ) 
  (h1 : 0 < da) (h2 : 0 < db) (h3 : 0 < dc)
  (h4 : r = (a * da + b * db + c * dc) / (a + b + c)) :
  2 / (1 / da + 1 / db + 1 / dc) < r ∧ r < (da + db + dc) / 2 :=
  sorry

end inradius_inequality_l956_95647


namespace max_points_of_intersection_l956_95675

-- Definitions based on the conditions in a)
def intersects_circle (l : ℕ) : ℕ := 2 * l  -- Each line intersects the circle at most twice
def intersects_lines (n : ℕ) : ℕ := n * (n - 1) / 2  -- Number of intersection points between lines (combinatorial)

-- The main statement that needs to be proved
theorem max_points_of_intersection (lines circle : ℕ) (h_lines_distinct : lines = 3) (h_no_parallel : ∀ (i j : ℕ), i ≠ j → i < lines → j < lines → true) (h_no_common_point : ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬(true)) : (intersects_circle lines + intersects_lines lines = 9) := 
  by
    sorry

end max_points_of_intersection_l956_95675


namespace opponents_team_points_l956_95687

theorem opponents_team_points (M D V O : ℕ) (hM : M = 5) (hD : D = 3) 
    (hV : V = 2 * (M + D)) (hO : O = (M + D + V) + 16) : O = 40 := by
  sorry

end opponents_team_points_l956_95687


namespace inequality_proof_l956_95619

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1)
                        (hb : 0 ≤ b) (hb1 : b ≤ 1)
                        (hc : 0 ≤ c) (hc1 : c ≤ 1) :
    (a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1) :=
by 
  sorry

end inequality_proof_l956_95619


namespace determine_b_l956_95615

noncomputable def Q (x : ℝ) (b : ℝ) : ℝ := x^3 + 3 * x^2 + b * x + 20

theorem determine_b (b : ℝ) :
  (∃ x : ℝ, x = 4 ∧ Q x b = 0) → b = -33 :=
by
  intro h
  rcases h with ⟨_, rfl, hQ⟩
  sorry

end determine_b_l956_95615


namespace original_number_is_1212_or_2121_l956_95665

theorem original_number_is_1212_or_2121 (x y z t : ℕ) (h₁ : t ≠ 0)
  (h₂ : 1000 * x + 100 * y + 10 * z + t + 1000 * t + 100 * x + 10 * y + z = 3333) : 
  (1000 * x + 100 * y + 10 * z + t = 1212) ∨ (1000 * x + 100 * y + 10 * z + t = 2121) :=
sorry

end original_number_is_1212_or_2121_l956_95665


namespace shekar_marks_in_math_l956_95674

theorem shekar_marks_in_math (M : ℕ) : 
  (65 + 82 + 67 + 75 + M) / 5 = 73 → M = 76 :=
by
  intros h
  sorry

end shekar_marks_in_math_l956_95674


namespace weight_of_replaced_person_l956_95604

-- Define the conditions
variables (W : ℝ) (new_person_weight : ℝ) (avg_weight_increase : ℝ)
#check ℝ

def initial_group_size := 10

-- Define the conditions as hypothesis statements
axiom weight_increase_eq : avg_weight_increase = 3.5
axiom new_person_weight_eq : new_person_weight = 100

-- Define the result to be proved
theorem weight_of_replaced_person (W : ℝ) : 
  ∀ (avg_weight_increase : ℝ) (new_person_weight : ℝ),
    avg_weight_increase = 3.5 ∧ new_person_weight = 100 → 
    (new_person_weight - (avg_weight_increase * initial_group_size)) = 65 := 
by
  sorry

end weight_of_replaced_person_l956_95604


namespace find_m_for_integer_solution_l956_95662

theorem find_m_for_integer_solution :
  ∀ (m x : ℤ), (x^3 - m*x^2 + m*x - (m^2 + 1) = 0) → (m = -3 ∨ m = 0) :=
by
  sorry

end find_m_for_integer_solution_l956_95662


namespace students_tried_out_l956_95636

theorem students_tried_out (x : ℕ) (h1 : 8 * (x - 17) = 384) : x = 65 := 
by
  sorry

end students_tried_out_l956_95636


namespace max_unique_dance_counts_l956_95657

theorem max_unique_dance_counts (boys girls : ℕ) (positive_boys : boys = 29) (positive_girls : girls = 15) 
  (dances : ∀ b g, b ≤ boys → g ≤ girls → ℕ) :
  ∃ num_dances, num_dances = 29 := 
by
  sorry

end max_unique_dance_counts_l956_95657


namespace area_of_triangle_is_correct_l956_95661

def line_1 (x y : ℝ) : Prop := y - 5 * x = -4
def line_2 (x y : ℝ) : Prop := 4 * y + 2 * x = 16

def y_axis (x y : ℝ) : Prop := x = 0

def satisfies_y_intercepts (f : ℝ → ℝ) : Prop :=
f 0 = -4 ∧ f 0 = 4

noncomputable def area_of_triangle (height base : ℝ) : ℝ :=
(1 / 2) * base * height

theorem area_of_triangle_is_correct :
  ∃ (x y : ℝ), line_1 x y ∧ line_2 x y ∧ y_axis 0 8 ∧ area_of_triangle (16 / 11) 8 = (64 / 11) := 
sorry

end area_of_triangle_is_correct_l956_95661


namespace least_k_l956_95638

noncomputable def u : ℕ → ℝ
| 0 => 1 / 8
| (n + 1) => 3 * u n - 3 * (u n) ^ 2

theorem least_k :
  ∃ k : ℕ, |u k - (1 / 3)| ≤ 1 / 2 ^ 500 ∧ ∀ m < k, |u m - (1 / 3)| > 1 / 2 ^ 500 :=
by
  sorry

end least_k_l956_95638


namespace simplify_and_evaluate_expr_l956_95676

theorem simplify_and_evaluate_expr (a b : ℤ) (h₁ : a = -1) (h₂ : b = 2) :
  (2 * a + b - 2 * (3 * a - 2 * b)) = 14 := by
  rw [h₁, h₂]
  sorry

end simplify_and_evaluate_expr_l956_95676


namespace complex_number_quadrant_l956_95635

def imaginary_unit := Complex.I

def complex_simplification (z : Complex) : Complex :=
  z

theorem complex_number_quadrant :
  ∃ z : Complex, z = (5 * imaginary_unit) / (2 + imaginary_unit ^ 9) ∧ (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_number_quadrant_l956_95635


namespace simplify_neg_cube_square_l956_95694

theorem simplify_neg_cube_square (a : ℝ) : (-a^3)^2 = a^6 :=
by
  sorry

end simplify_neg_cube_square_l956_95694


namespace percentage_of_cars_on_monday_compared_to_tuesday_l956_95685

theorem percentage_of_cars_on_monday_compared_to_tuesday : 
  ∀ (cars_mon cars_tue cars_wed cars_thu cars_fri cars_sat cars_sun : ℕ),
    cars_mon + cars_tue + cars_wed + cars_thu + cars_fri + cars_sat + cars_sun = 97 →
    cars_tue = 25 →
    cars_wed = cars_mon + 2 →
    cars_thu = 10 →
    cars_fri = 10 →
    cars_sat = 5 →
    cars_sun = 5 →
    (cars_mon * 100 / cars_tue = 80) :=
by
  intros cars_mon cars_tue cars_wed cars_thu cars_fri cars_sat cars_sun
  intro h_total
  intro h_tue
  intro h_wed
  intro h_thu
  intro h_fri
  intro h_sat
  intro h_sun
  sorry

end percentage_of_cars_on_monday_compared_to_tuesday_l956_95685
