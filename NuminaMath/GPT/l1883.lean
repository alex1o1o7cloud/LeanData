import Mathlib

namespace greatest_length_of_pieces_l1883_188325

theorem greatest_length_of_pieces (a b c : ℕ) (ha : a = 48) (hb : b = 60) (hc : c = 72) :
  Nat.gcd (Nat.gcd a b) c = 12 := by
  sorry

end greatest_length_of_pieces_l1883_188325


namespace joe_initial_money_l1883_188390

theorem joe_initial_money (cost_notebook cost_book money_left : ℕ) 
                          (num_notebooks num_books : ℕ)
                          (h1 : cost_notebook = 4) 
                          (h2 : cost_book = 7)
                          (h3 : num_notebooks = 7) 
                          (h4 : num_books = 2) 
                          (h5 : money_left = 14) :
  (num_notebooks * cost_notebook + num_books * cost_book + money_left) = 56 := by
  sorry

end joe_initial_money_l1883_188390


namespace probability_fully_lit_l1883_188353

-- define the conditions of the problem
def characters : List String := ["K", "y", "o", "t", "o", " ", "G", "r", "a", "n", "d", " ", "H", "o", "t", "e", "l"]

-- define the length of the sequence
def length_sequence : ℕ := characters.length

-- theorem stating the probability of seeing the fully lit sign
theorem probability_fully_lit : (1 / length_sequence) = 1 / 5 :=
by
  -- The proof is omitted
  sorry

end probability_fully_lit_l1883_188353


namespace focus_of_parabola_y_eq_9x2_plus_6_l1883_188354

noncomputable def focus_of_parabola (a b : ℝ) : ℝ × ℝ :=
  (0, b + (1 / (4 * a)))

theorem focus_of_parabola_y_eq_9x2_plus_6 :
  focus_of_parabola 9 6 = (0, 217 / 36) :=
by
  sorry

end focus_of_parabola_y_eq_9x2_plus_6_l1883_188354


namespace x_coord_sum_l1883_188379

noncomputable def sum_x_coordinates (x : ℕ) : Prop :=
  (0 ≤ x ∧ x < 20) ∧ (∃ y, y ≡ 7 * x + 3 [MOD 20] ∧ y ≡ 13 * x + 18 [MOD 20])

theorem x_coord_sum : ∃ (x : ℕ), sum_x_coordinates x ∧ x = 15 := by 
  sorry

end x_coord_sum_l1883_188379


namespace exponentiation_addition_zero_l1883_188363

theorem exponentiation_addition_zero : (-2)^(3^2) + 2^(3^2) = 0 := 
by 
  -- proof goes here
  sorry

end exponentiation_addition_zero_l1883_188363


namespace ratio_of_weight_l1883_188319

theorem ratio_of_weight (B : ℝ) : 
    (2 * (4 + B) = 16) → ((B = 4) ∧ (4 + B) / 2 = 4) := by
  intro h
  have h₁ : B = 4 := by
    linarith
  have h₂ : (4 + B) / 2 = 4 := by
    rw [h₁]
    norm_num
  exact ⟨h₁, h₂⟩

end ratio_of_weight_l1883_188319


namespace company_x_total_employees_l1883_188395

-- Definitions for conditions
def initial_percentage : ℝ := 0.60
def Q2_hiring_males : ℕ := 30
def Q2_new_percentage : ℝ := 0.57
def Q3_hiring_females : ℕ := 50
def Q3_new_percentage : ℝ := 0.62
def Q4_hiring_males : ℕ := 40
def Q4_hiring_females : ℕ := 10
def Q4_new_percentage : ℝ := 0.58

-- Statement of the proof problem
theorem company_x_total_employees :
  ∃ (E : ℕ) (F : ℕ), 
    (F = initial_percentage * E ∧
     F = Q2_new_percentage * (E + Q2_hiring_males) ∧
     F + Q3_hiring_females = Q3_new_percentage * (E + Q2_hiring_males + Q3_hiring_females) ∧
     F + Q3_hiring_females + Q4_hiring_females = Q4_new_percentage * (E + Q2_hiring_males + Q3_hiring_females + Q4_hiring_males + Q4_hiring_females)) →
    E + Q2_hiring_males + Q3_hiring_females + Q4_hiring_males + Q4_hiring_females = 700 :=
sorry

end company_x_total_employees_l1883_188395


namespace lies_on_new_ellipse_lies_on_new_hyperbola_l1883_188310

variable (x y c d a : ℝ)

def new_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

-- Definition for new ellipse.
def is_new_ellipse (E : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (a : ℝ) : Prop :=
  new_distance E F1 + new_distance E F2 = 2 * a

-- Definition for new hyperbola.
def is_new_hyperbola (H : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (a : ℝ) : Prop :=
  |new_distance H F1 - new_distance H F2| = 2 * a

-- The point E lies on the new ellipse.
theorem lies_on_new_ellipse
  (E F1 F2 : ℝ × ℝ) (a : ℝ) :
  is_new_ellipse E F1 F2 a :=
by sorry

-- The point H lies on the new hyperbola.
theorem lies_on_new_hyperbola
  (H F1 F2 : ℝ × ℝ) (a : ℝ) :
  is_new_hyperbola H F1 F2 a :=
by sorry

end lies_on_new_ellipse_lies_on_new_hyperbola_l1883_188310


namespace exponentiation_and_multiplication_of_fractions_l1883_188396

-- Let's define the required fractions
def a : ℚ := 3 / 4
def b : ℚ := 1 / 5

-- Define the expected result
def expected_result : ℚ := 81 / 1280

-- State the theorem to prove
theorem exponentiation_and_multiplication_of_fractions : (a^4) * b = expected_result := by 
  sorry

end exponentiation_and_multiplication_of_fractions_l1883_188396


namespace arithmetic_geometric_sequence_min_sum_l1883_188308

theorem arithmetic_geometric_sequence_min_sum :
  ∃ (A B C D : ℕ), 
    (C - B = B - A) ∧ 
    (C * 4 = B * 7) ∧ 
    (D * 4 = C * 7) ∧ 
    (16 ∣ B) ∧ 
    (A + B + C + D = 97) :=
by sorry

end arithmetic_geometric_sequence_min_sum_l1883_188308


namespace polygon_sides_sum_720_l1883_188324

theorem polygon_sides_sum_720 (n : ℕ) (h1 : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end polygon_sides_sum_720_l1883_188324


namespace purely_imaginary_z_implies_m_zero_l1883_188342

theorem purely_imaginary_z_implies_m_zero (m : ℝ) :
  m * (m + 1) = 0 → m ≠ -1 := by sorry

end purely_imaginary_z_implies_m_zero_l1883_188342


namespace maximum_sine_sum_l1883_188302

open Real

theorem maximum_sine_sum (x y z : ℝ) (hx : 0 ≤ x) (hy : x ≤ π / 2) (hz : 0 ≤ y) (hw : y ≤ π / 2) (hv : 0 ≤ z) (hu : z ≤ π / 2) :
  ∃ M, M = sqrt 2 - 1 ∧ ∀ x y z : ℝ, 0 ≤ x → x ≤ π / 2 → 0 ≤ y → y ≤ π / 2 → 0 ≤ z → z ≤ π / 2 → 
  sin (x - y) + sin (y - z) + sin (z - x) ≤ M :=
by
  sorry

end maximum_sine_sum_l1883_188302


namespace probability_five_heads_in_six_tosses_is_09375_l1883_188374

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_exact_heads (n k : ℕ) (p : ℝ) : ℝ :=
  binomial n k * (p^k) * ((1-p)^(n-k))
  
theorem probability_five_heads_in_six_tosses_is_09375 :
  probability_exact_heads 6 5 0.5 = 0.09375 :=
by
  sorry

end probability_five_heads_in_six_tosses_is_09375_l1883_188374


namespace part1_max_value_part2_three_distinct_real_roots_l1883_188391

def f (x m : ℝ) : ℝ := x * (x - m)^2

theorem part1_max_value (m : ℝ) (h_max : ∀ x, f x m ≤ f 2 m) : m = 6 := by
  sorry

theorem part2_three_distinct_real_roots (a : ℝ) (h_m : (m = 6))
  (h_a : ∀ x₁ x₂ x₃ : ℝ, f x₁ m = a ∧ f x₂ m = a ∧ f x₃ m = a →
     x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) : 0 < a ∧ a < 32 := by
  sorry

end part1_max_value_part2_three_distinct_real_roots_l1883_188391


namespace hockey_team_helmets_l1883_188366

theorem hockey_team_helmets (r b : ℕ) 
  (h1 : b = r - 6) 
  (h2 : r * 3 = b * 5) : 
  r + b = 24 :=
by
  sorry

end hockey_team_helmets_l1883_188366


namespace jake_weight_l1883_188355

theorem jake_weight {J S : ℝ} (h1 : J - 20 = 2 * S) (h2 : J + S = 224) : J = 156 :=
by
  sorry

end jake_weight_l1883_188355


namespace num_values_divisible_by_120_l1883_188300

theorem num_values_divisible_by_120 (n : ℕ) (h_seq : ∀ n, ∃ k, n = k * (k + 1)) :
  ∃ k, k = 8 := sorry

end num_values_divisible_by_120_l1883_188300


namespace algebra_books_needed_l1883_188313

theorem algebra_books_needed (A' H' S' M' E' : ℕ) (x y : ℝ) (z : ℝ)
  (h1 : y > x)
  (h2 : A' ≠ H' ∧ A' ≠ S' ∧ A' ≠ M' ∧ A' ≠ E' ∧ H' ≠ S' ∧ H' ≠ M' ∧ H' ≠ E' ∧ S' ≠ M' ∧ S' ≠ E' ∧ M' ≠ E')
  (h3 : A' * x + H' * y = z)
  (h4 : S' * x + M' * y = z)
  (h5 : E' * x = 2 * z) :
  E' = (2 * A' * M' - 2 * S' * H') / (M' - H') :=
by
  sorry

end algebra_books_needed_l1883_188313


namespace inverse_relation_a1600_inverse_relation_a400_l1883_188345

variable (a b : ℝ)

def k := 400 

theorem inverse_relation_a1600 : (a * b = k) → (a = 1600) → (b = 0.25) :=
by
  sorry

theorem inverse_relation_a400 : (a * b = k) → (a = 400) → (b = 1) :=
by
  sorry

end inverse_relation_a1600_inverse_relation_a400_l1883_188345


namespace cookies_in_each_bag_l1883_188340

-- Definitions based on the conditions
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41
def baggies : ℕ := 6

-- Assertion of the correct answer
theorem cookies_in_each_bag : 
  (chocolate_chip_cookies + oatmeal_cookies) / baggies = 9 := by
  sorry

end cookies_in_each_bag_l1883_188340


namespace age_of_15th_student_l1883_188347

noncomputable def average_age_15_students := 15
noncomputable def average_age_7_students_1 := 14
noncomputable def average_age_7_students_2 := 16
noncomputable def total_students := 15
noncomputable def group_students := 7

theorem age_of_15th_student :
  let total_age_15_students := total_students * average_age_15_students
  let total_age_7_students_1 := group_students * average_age_7_students_1
  let total_age_7_students_2 := group_students * average_age_7_students_2
  let total_age_14_students := total_age_7_students_1 + total_age_7_students_2
  let age_15th_student := total_age_15_students - total_age_14_students
  age_15th_student = 15 :=
by
  sorry

end age_of_15th_student_l1883_188347


namespace number_of_carbons_l1883_188329

-- Definitions of given conditions
def molecular_weight (total_c total_h total_o c_weight h_weight o_weight : ℕ) := 
    total_c * c_weight + total_h * h_weight + total_o * o_weight

-- Given values
def num_hydrogen_atoms : ℕ := 8
def num_oxygen_atoms : ℕ := 2
def molecular_wt : ℕ := 88
def atomic_weight_c : ℕ := 12
def atomic_weight_h : ℕ := 1
def atomic_weight_o : ℕ := 16

-- The theorem to be proved
theorem number_of_carbons (num_carbons : ℕ) 
    (H_hydrogen : num_hydrogen_atoms = 8)
    (H_oxygen : num_oxygen_atoms = 2)
    (H_molecular_weight : molecular_wt = 88)
    (H_atomic_weight_c : atomic_weight_c = 12)
    (H_atomic_weight_h : atomic_weight_h = 1)
    (H_atomic_weight_o : atomic_weight_o = 16) :
    molecular_weight num_carbons num_hydrogen_atoms num_oxygen_atoms atomic_weight_c atomic_weight_h atomic_weight_o = molecular_wt → 
    num_carbons = 4 :=
by
  intros h
  sorry 

end number_of_carbons_l1883_188329


namespace polynomial_coefficients_even_or_odd_l1883_188311

-- Define the problem conditions as Lean definitions
variables {P Q : Polynomial ℤ}

-- Theorem: Given the conditions, prove the required statement
theorem polynomial_coefficients_even_or_odd
  (hP : ∀ n : ℕ, P.coeff n % 2 = 0)
  (hQ : ∀ n : ℕ, Q.coeff n % 2 = 0)
  (hProd : ¬ ∀ n : ℕ, (P * Q).coeff n % 4 = 0) :
  (∀ n : ℕ, P.coeff n % 2 = 0 ∧ ∃ k : ℕ, Q.coeff k % 2 ≠ 0) ∨
  (∀ n : ℕ, Q.coeff n % 2 = 0 ∧ ∃ k: ℕ, P.coeff k % 2 ≠ 0) :=
sorry

end polynomial_coefficients_even_or_odd_l1883_188311


namespace max_ounces_amber_can_get_l1883_188386

theorem max_ounces_amber_can_get :
  let money := 7
  let candy_cost := 1
  let candy_ounces := 12
  let chips_cost := 1.40
  let chips_ounces := 17
  let max_ounces := max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces)
  max_ounces = 85 := 
by
  sorry

end max_ounces_amber_can_get_l1883_188386


namespace desired_average_sale_l1883_188306

theorem desired_average_sale
  (sale1 sale2 sale3 sale4 sale5 sale6 : ℕ)
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h4 : sale4 = 7230)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 7991) :
  (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = 7000 :=
by
  sorry

end desired_average_sale_l1883_188306


namespace intersection_P_Q_l1883_188332

def P : Set ℝ := { x | x > 1 }
def Q : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_P_Q : P ∩ Q = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_P_Q_l1883_188332


namespace pollen_scientific_notation_correct_l1883_188385

def moss_flower_pollen_diameter := 0.0000084
def pollen_scientific_notation := 8.4 * 10^(-6)

theorem pollen_scientific_notation_correct :
  moss_flower_pollen_diameter = pollen_scientific_notation :=
by
  -- Proof skipped
  sorry

end pollen_scientific_notation_correct_l1883_188385


namespace percent_of_rectangle_area_inside_square_l1883_188303

theorem percent_of_rectangle_area_inside_square
  (s : ℝ)  -- Let the side length of the square be \( s \).
  (width : ℝ) (length: ℝ)
  (h1 : width = 3 * s)  -- The width of the rectangle is \( 3s \).
  (h2 : length = 2 * width) :  -- The length of the rectangle is \( 2 * width \).
  (s^2 / (length * width)) * 100 = 5.56 :=
by
  sorry

end percent_of_rectangle_area_inside_square_l1883_188303


namespace scientific_notation_correct_l1883_188380

theorem scientific_notation_correct :
  1200000000 = 1.2 * 10^9 := 
by
  sorry

end scientific_notation_correct_l1883_188380


namespace tangent_line_min_slope_equation_l1883_188343

def curve (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 1

theorem tangent_line_min_slope_equation :
  ∃ (k : ℝ) (b : ℝ), (∀ x y, y = curve x → y = k * x + b)
  ∧ (k = 3)
  ∧ (b = -2)
  ∧ (3 * x - y - 2 = 0) :=
by
  sorry

end tangent_line_min_slope_equation_l1883_188343


namespace sum_first_six_terms_geometric_seq_l1883_188397

theorem sum_first_six_terms_geometric_seq :
  let a := (1 : ℚ) / 4 
  let r := (1 : ℚ) / 4 
  let n := 6
  (a * (1 - r^n) / (1 - r)) = (1365 / 16384 : ℚ) :=
by
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  sorry

end sum_first_six_terms_geometric_seq_l1883_188397


namespace find_a_given_coefficient_l1883_188348

theorem find_a_given_coefficient (a : ℝ) (h : (a^3 * 10 = 80)) : a = 2 :=
by
  sorry

end find_a_given_coefficient_l1883_188348


namespace ordered_pairs_satisfying_condition_l1883_188394

theorem ordered_pairs_satisfying_condition : 
  ∃! (pairs : Finset (ℕ × ℕ)),
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ 
      m > 0 ∧ n > 0 ∧ m ≥ n ∧ m^2 - n^2 = 144) ∧ 
    pairs.card = 4 := sorry

end ordered_pairs_satisfying_condition_l1883_188394


namespace average_speed_correct_l1883_188362

-- Definitions of distances and speeds
def distance1 := 50 -- miles
def speed1 := 20 -- miles per hour
def distance2 := 20 -- miles
def speed2 := 40 -- miles per hour
def distance3 := 30 -- miles
def speed3 := 15 -- miles per hour

-- Definition of total distance and total time
def total_distance := distance1 + distance2 + distance3
def time1 := distance1 / speed1
def time2 := distance2 / speed2
def time3 := distance3 / speed3
def total_time := time1 + time2 + time3

-- Definition of average speed
def average_speed := total_distance / total_time

-- Statement to be proven
theorem average_speed_correct : average_speed = 20 := 
by 
  -- Proof will be provided here
  sorry

end average_speed_correct_l1883_188362


namespace length_increase_percentage_l1883_188301

theorem length_increase_percentage 
  (L B : ℝ)
  (x : ℝ)
  (h1 : B' = B * 0.8)
  (h2 : L' = L * (1 + x / 100))
  (h3 : A = L * B)
  (h4 : A' = L' * B')
  (h5 : A' = A * 1.04) 
  : x = 30 :=
sorry

end length_increase_percentage_l1883_188301


namespace messages_after_noon_l1883_188372

theorem messages_after_noon (t n : ℕ) (h1 : t = 39) (h2 : n = 21) : t - n = 18 := by
  sorry

end messages_after_noon_l1883_188372


namespace greatest_product_of_digits_l1883_188307

theorem greatest_product_of_digits :
  ∀ a b : ℕ, (10 * a + b) % 35 = 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  ∃ ab_max : ℕ, ab_max = a * b ∧ ab_max = 15 :=
by
  sorry

end greatest_product_of_digits_l1883_188307


namespace probability_XiaoCong_project_A_probability_same_project_not_C_l1883_188398

-- Definition of projects and conditions
inductive Project
| A | B | C

def XiaoCong : Project := sorry
def XiaoYing : Project := sorry

-- (1) Probability of Xiao Cong assigned to project A
theorem probability_XiaoCong_project_A : 
  (1 / 3 : ℝ) = 1 / 3 := 
by sorry

-- (2) Probability of Xiao Cong and Xiao Ying being assigned to the same project, given Xiao Ying not assigned to C
theorem probability_same_project_not_C : 
  (2 / 6 : ℝ) = 1 / 3 :=
by sorry

end probability_XiaoCong_project_A_probability_same_project_not_C_l1883_188398


namespace smaller_rectangle_area_l1883_188367

-- Define the lengths and widths of the rectangles
def bigRectangleLength : ℕ := 40
def bigRectangleWidth : ℕ := 20
def smallRectangleLength : ℕ := bigRectangleLength / 2
def smallRectangleWidth : ℕ := bigRectangleWidth / 2

-- Define the area of the rectangles
def area (length width : ℕ) : ℕ := length * width

-- Prove the area of the smaller rectangle
theorem smaller_rectangle_area : area smallRectangleLength smallRectangleWidth = 200 :=
by
  -- Skip the proof
  sorry

end smaller_rectangle_area_l1883_188367


namespace find_second_number_l1883_188378

-- The Lean statement for the given math problem:

theorem find_second_number
  (x y z : ℝ)  -- Represent the three numbers
  (h1 : x = 2 * y)  -- The first number is twice the second
  (h2 : z = (1/3) * x)  -- The third number is one-third of the first
  (h3 : x + y + z = 110)  -- The sum of the three numbers is 110
  : y = 30 :=  -- The second number is 30
sorry

end find_second_number_l1883_188378


namespace smallest_integer_is_840_l1883_188376

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def all_divide (N : ℕ) : Prop :=
  (2 ∣ N) ∧ (3 ∣ N) ∧ (5 ∣ N) ∧ (7 ∣ N)

def no_prime_digit (N : ℕ) : Prop :=
  ∀ d ∈ N.digits 10, ¬ is_prime_digit d

def smallest_satisfying_N (N : ℕ) : Prop :=
  no_prime_digit N ∧ all_divide N ∧ ∀ M, no_prime_digit M → all_divide M → N ≤ M

theorem smallest_integer_is_840 : smallest_satisfying_N 840 :=
by
  sorry

end smallest_integer_is_840_l1883_188376


namespace brother_raking_time_l1883_188341

theorem brother_raking_time (x : ℝ) (hx : x > 0)
  (h_combined : (1 / 30) + (1 / x) = 1 / 18) : x = 45 :=
by
  sorry

end brother_raking_time_l1883_188341


namespace minimum_value_l1883_188382

theorem minimum_value (x : ℝ) (h : x > 1) : 2 * x + 7 / (x - 1) ≥ 2 * Real.sqrt 14 + 2 := by
  sorry

end minimum_value_l1883_188382


namespace waiter_customers_l1883_188351

variable (initial_customers left_customers new_customers : ℕ)

theorem waiter_customers 
  (h1 : initial_customers = 33)
  (h2 : left_customers = 31)
  (h3 : new_customers = 26) :
  (initial_customers - left_customers + new_customers = 28) := 
by
  sorry

end waiter_customers_l1883_188351


namespace number_of_small_pizzas_ordered_l1883_188375

-- Define the problem conditions
def benBrothers : Nat := 2
def slicesPerPerson : Nat := 12
def largePizzaSlices : Nat := 14
def smallPizzaSlices : Nat := 8
def numLargePizzas : Nat := 2

-- Define the statement to prove
theorem number_of_small_pizzas_ordered : 
  ∃ (s : Nat), (benBrothers + 1) * slicesPerPerson - numLargePizzas * largePizzaSlices = s * smallPizzaSlices ∧ s = 1 :=
by
  sorry

end number_of_small_pizzas_ordered_l1883_188375


namespace min_value_cos_sin_l1883_188381

noncomputable def min_value_expression : ℝ :=
  -1 / 2

theorem min_value_cos_sin (θ : ℝ) (hθ1 : 0 ≤ θ) (hθ2 : θ ≤ 3 * Real.pi / 2) :
  ∃ (y : ℝ), y = Real.cos (θ / 3) * (1 - Real.sin θ) ∧ y = min_value_expression :=
sorry

end min_value_cos_sin_l1883_188381


namespace three_digit_numbers_divisible_by_11_are_550_or_803_l1883_188368

theorem three_digit_numbers_divisible_by_11_are_550_or_803 :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000 ∧ ∃ (a b c : ℕ), N = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 11 ∣ N ∧ (N / 11 = a^2 + b^2 + c^2)) → (N = 550 ∨ N = 803) :=
by
  sorry

end three_digit_numbers_divisible_by_11_are_550_or_803_l1883_188368


namespace area_of_triangle_bounded_by_line_and_axes_l1883_188328

theorem area_of_triangle_bounded_by_line_and_axes (x y : ℝ) (hx : 3 * x + 2 * y = 12) :
  ∃ (area : ℝ), area = 12 := by
sorry

end area_of_triangle_bounded_by_line_and_axes_l1883_188328


namespace find_goods_train_speed_l1883_188364

-- Definition of given conditions
def speed_of_man_train_kmph : ℝ := 120
def time_goods_train_seconds : ℝ := 9
def length_goods_train_meters : ℝ := 350

-- The proof statement
theorem find_goods_train_speed :
  let relative_speed_mps := (speed_of_man_train_kmph + goods_train_speed_kmph) * (5 / 18)
  ∃ (goods_train_speed_kmph : ℝ), relative_speed_mps = length_goods_train_meters / time_goods_train_seconds ∧ goods_train_speed_kmph = 20 :=
by {
  sorry
}

end find_goods_train_speed_l1883_188364


namespace correct_total_annual_salary_expression_l1883_188317

def initial_workers : ℕ := 8
def initial_salary : ℝ := 1.0 -- in ten thousand yuan
def new_workers : ℕ := 3
def new_worker_initial_salary : ℝ := 0.8 -- in ten thousand yuan
def salary_increase_rate : ℝ := 1.2 -- 20% increase each year

def total_annual_salary (n : ℕ) : ℝ :=
  (3 * n + 5) * salary_increase_rate^n + (new_workers * new_worker_initial_salary)

theorem correct_total_annual_salary_expression (n : ℕ) :
  total_annual_salary n = (3 * n + 5) * 1.2^n + 2.4 := 
by
  sorry

end correct_total_annual_salary_expression_l1883_188317


namespace tournament_cycle_exists_l1883_188326

theorem tournament_cycle_exists :
  ∃ (A B C : Fin 12), 
  (∃ M : Fin 12 → Fin 12 → Bool, 
    (∀ p : Fin 12, ∃ q : Fin 12, q ≠ p ∧ M p q) ∧
    M A B = true ∧ M B C = true ∧ M C A = true) :=
sorry

end tournament_cycle_exists_l1883_188326


namespace point_A_outside_circle_iff_l1883_188356

-- Define the conditions
def B : ℝ := 16
def radius : ℝ := 4
def A_position (t : ℝ) : ℝ := 2 * t

-- Define the theorem
theorem point_A_outside_circle_iff (t : ℝ) : (A_position t < B - radius) ∨ (A_position t > B + radius) ↔ (t < 6 ∨ t > 10) :=
by
  sorry

end point_A_outside_circle_iff_l1883_188356


namespace shaded_region_equality_l1883_188316

-- Define the necessary context and variables
variable {r : ℝ} -- radius of the circle
variable {θ : ℝ} -- angle measured in degrees

-- Define the relevant trigonometric functions
noncomputable def tan_degrees (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)
noncomputable def tan_half_degrees (x : ℝ) : ℝ := Real.tan ((x / 2) * Real.pi / 180)

-- State the theorem we need to prove given the conditions
theorem shaded_region_equality (hθ1 : θ / 2 = 90 - θ) :
  tan_degrees θ + (tan_degrees θ)^2 * tan_half_degrees θ = (θ * Real.pi) / 180 - (θ^2 * Real.pi) / 360 :=
  sorry

end shaded_region_equality_l1883_188316


namespace sum_of_squares_of_solutions_l1883_188388

theorem sum_of_squares_of_solutions :
  (∃ s₁ s₂ : ℝ, s₁ ≠ s₂ ∧ s₁ + s₂ = 17 ∧ s₁ * s₂ = 22) →
  ∃ s₁ s₂ : ℝ, s₁^2 + s₂^2 = 245 :=
by
  sorry

end sum_of_squares_of_solutions_l1883_188388


namespace homework_duration_equation_l1883_188344

-- Given conditions
def initial_duration : ℝ := 120
def final_duration : ℝ := 60
variable (x : ℝ)

-- The goal is to prove that the appropriate equation holds
theorem homework_duration_equation : initial_duration * (1 - x)^2 = final_duration := 
sorry

end homework_duration_equation_l1883_188344


namespace calculate_V3_at_2_l1883_188377

def polynomial (x : ℕ) : ℕ :=
  (((((2 * x + 5) * x + 6) * x + 23) * x - 8) * x + 10) * x - 3

theorem calculate_V3_at_2 : polynomial 2 = 71 := by
  sorry

end calculate_V3_at_2_l1883_188377


namespace John_bought_new_socks_l1883_188371

theorem John_bought_new_socks (initial_socks : ℕ) (thrown_away_socks : ℕ) (current_socks : ℕ) :
    initial_socks = 33 → thrown_away_socks = 19 → current_socks = 27 → 
    current_socks = (initial_socks - thrown_away_socks) + 13 :=
by
  sorry

end John_bought_new_socks_l1883_188371


namespace solve_for_m_l1883_188359

theorem solve_for_m :
  (∀ (m : ℕ), 
   ((1:ℚ)^(m+1) / 5^(m+1) * 1^18 / 4^18 = 1 / (2 * 10^35)) → m = 34) := 
by apply sorry

end solve_for_m_l1883_188359


namespace total_weight_of_balls_l1883_188361

theorem total_weight_of_balls :
  let weight_blue := 6
  let weight_brown := 3.12
  let weight_green := 4.5
  weight_blue + weight_brown + weight_green = 13.62 := by
  sorry

end total_weight_of_balls_l1883_188361


namespace subset_proof_l1883_188373

-- Define the set B
def B : Set ℝ := { x | x ≥ 0 }

-- Define the set A as the set {1, 2}
def A : Set ℝ := {1, 2}

-- The proof problem: Prove that A ⊆ B
theorem subset_proof : A ⊆ B := sorry

end subset_proof_l1883_188373


namespace parabola_equation_l1883_188320

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

-- Define the standard equation form of the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Define the right vertex of the hyperbola
def right_vertex (a : ℝ) : ℝ × ℝ :=
  (a, 0)

-- State the final proof problem
theorem parabola_equation :
  hyperbola 4 0 →
  parabola 8 x y →
  y^2 = 16 * x :=
by
  -- Skip the proof for now
  sorry

end parabola_equation_l1883_188320


namespace Billy_is_45_l1883_188357

variable (B J : ℕ)

-- Condition 1: Billy's age is three times Joe's age
def condition1 : Prop := B = 3 * J

-- Condition 2: The sum of their ages is 60
def condition2 : Prop := B + J = 60

-- The theorem we want to prove: Billy's age is 45
theorem Billy_is_45 (h1 : condition1 B J) (h2 : condition2 B J) : B = 45 := 
sorry

end Billy_is_45_l1883_188357


namespace negation_of_existence_l1883_188334

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) :=
sorry

end negation_of_existence_l1883_188334


namespace Ben_cards_left_l1883_188337

def BenInitialBasketballCards : ℕ := 4 * 10
def BenInitialBaseballCards : ℕ := 5 * 8
def BenTotalInitialCards : ℕ := BenInitialBasketballCards + BenInitialBaseballCards
def BenGivenCards : ℕ := 58
def BenRemainingCards : ℕ := BenTotalInitialCards - BenGivenCards

theorem Ben_cards_left : BenRemainingCards = 22 :=
by 
  -- The proof will be placed here.
  sorry

end Ben_cards_left_l1883_188337


namespace trig_expression_value_l1883_188349

theorem trig_expression_value (α : ℝ) (h₁ : Real.tan (α + π / 4) = -1/2) (h₂ : π / 2 < α ∧ α < π) :
  (Real.sin (2 * α) - 2 * (Real.cos α)^2) / Real.sin (α - π / 4) = - (2 * Real.sqrt 5) / 5 :=
by
  sorry

end trig_expression_value_l1883_188349


namespace problem_statement_l1883_188318

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x
noncomputable def g (x : ℝ) : ℝ := (2 ^ x) / 2 - 2 / (2 ^ x) - x + 1

theorem problem_statement (a : ℝ) (x₁ x₂ : ℝ) (h₀ : x₁ < x₂)
  (h₁ : f a x₁ = 0) (h₂ : f a x₂ = 0) : g x₁ + g x₂ > 0 :=
sorry

end problem_statement_l1883_188318


namespace min_value_of_expression_l1883_188358

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  (1 / x + 4 / y) ≥ 9 :=
by
  sorry

end min_value_of_expression_l1883_188358


namespace extreme_point_condition_l1883_188369

variable {R : Type*} [OrderedRing R]

def f (x a b : R) : R := x^3 - a*x - b

theorem extreme_point_condition (a b x0 x1 : R) (h₁ : ∀ x : R, f x a b = x^3 - a*x - b)
  (h₂ : f x0 a b = x0^3 - a*x0 - b)
  (h₃ : f x1 a b = x1^3 - a*x1 - b)
  (has_extreme : ∃ x0 : R, 3*x0^2 = a) 
  (hx1_extreme : f x1 a b = f x0 a b) 
  (hx1_x0_diff : x1 ≠ x0) :
  x1 + 2*x0 = 0 :=
by
  sorry

end extreme_point_condition_l1883_188369


namespace ratio_largest_middle_l1883_188331

-- Definitions based on given conditions
def A : ℕ := 24  -- smallest number
def B : ℕ := 40  -- middle number
def C : ℕ := 56  -- largest number

theorem ratio_largest_middle (h1 : C = 56) (h2 : A = C - 32) (h3 : A = 24) (h4 : B = 40) :
  C / B = 7 / 5 := by
  sorry

end ratio_largest_middle_l1883_188331


namespace waiter_tables_l1883_188333

theorem waiter_tables (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (tables : ℕ) :
  initial_customers = 62 → 
  customers_left = 17 → 
  people_per_table = 9 → 
  remaining_customers = initial_customers - customers_left →
  tables = remaining_customers / people_per_table →
  tables = 5 :=
by
  intros hinitial hleft hpeople hremaining htables
  rw [hinitial, hleft, hpeople] at *
  simp at *
  sorry

end waiter_tables_l1883_188333


namespace num_natural_numbers_divisible_by_7_l1883_188338

theorem num_natural_numbers_divisible_by_7 (a b : ℕ) (h₁ : 200 ≤ a) (h₂ : b ≤ 400) (h₃ : a = 203) (h₄ : b = 399) :
  (b - a) / 7 + 1 = 29 := 
by
  sorry

end num_natural_numbers_divisible_by_7_l1883_188338


namespace carrie_expected_strawberries_l1883_188321

noncomputable def calculate_strawberries (base height : ℝ) (plants_per_sq_ft strawberries_per_plant : ℝ) : ℝ :=
  let area := (1/2) * base * height
  let total_plants := plants_per_sq_ft * area
  total_plants * strawberries_per_plant

theorem carrie_expected_strawberries : calculate_strawberries 10 12 5 8 = 2400 :=
by
  /-
  Given: base = 10, height = 12, plants_per_sq_ft = 5, strawberries_per_plant = 8
  - calculate the area of the right triangle garden
  - calculate the total number of plants
  - calculate the total number of strawberries
  -/
  sorry

end carrie_expected_strawberries_l1883_188321


namespace men_in_first_group_l1883_188312

theorem men_in_first_group (M : ℕ) (h1 : M * 35 = 7 * 50) : M = 10 := by
  sorry

end men_in_first_group_l1883_188312


namespace geom_sequence_a1_value_l1883_188350

-- Define the conditions and the statement
theorem geom_sequence_a1_value (a_1 a_6 : ℚ) (a_3 a_4 : ℚ)
  (h1 : a_1 + a_6 = 11)
  (h2 : a_3 * a_4 = 32 / 9) :
  (a_1 = 32 / 3 ∨ a_1 = 1 / 3) :=
by 
-- We will prove the theorem here (skipped with sorry)
sorry

end geom_sequence_a1_value_l1883_188350


namespace smallest_sum_of_18_consecutive_integers_is_perfect_square_l1883_188352

theorem smallest_sum_of_18_consecutive_integers_is_perfect_square 
  (n : ℕ) 
  (S : ℕ) 
  (h1 : S = 9 * (2 * n + 17)) 
  (h2 : ∃ k : ℕ, 2 * n + 17 = k^2) 
  (h3 : ∀ m : ℕ, m < 5 → 2 * n + 17 ≠ m^2) : 
  S = 225 := 
by
  sorry

end smallest_sum_of_18_consecutive_integers_is_perfect_square_l1883_188352


namespace equilateral_triangle_perimeter_l1883_188365

theorem equilateral_triangle_perimeter (s : ℝ) 
  (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  -- Proof steps (omitted)
  sorry

end equilateral_triangle_perimeter_l1883_188365


namespace visits_365_days_l1883_188399

theorem visits_365_days : 
  let alice_visits := 3
  let beatrix_visits := 4
  let claire_visits := 5
  let total_days := 365
  ∃ days_with_exactly_two_visits, days_with_exactly_two_visits = 54 :=
by
  sorry

end visits_365_days_l1883_188399


namespace yellow_balls_count_l1883_188336

-- Definition of problem conditions
def initial_red_balls : ℕ := 16
def initial_blue_balls : ℕ := 2 * initial_red_balls
def red_balls_lost : ℕ := 6
def green_balls_given_away : ℕ := 7  -- This is not used in the calculations
def yellow_balls_bought : ℕ := 3 * red_balls_lost
def final_total_balls : ℕ := 74

-- Defining the total balls after all transactions
def remaining_red_balls : ℕ := initial_red_balls - red_balls_lost
def total_accounted_balls : ℕ := remaining_red_balls + initial_blue_balls + yellow_balls_bought

-- Lean statement to prove
theorem yellow_balls_count : yellow_balls_bought = 18 :=
by
  sorry

end yellow_balls_count_l1883_188336


namespace upgrade_days_to_sun_l1883_188335

/-- 
  Determine the minimum number of additional active days required for 
  a user currently at level 2 moons and 1 star to upgrade to 1 sun.
-/
theorem upgrade_days_to_sun (level_new_star : ℕ) (level_new_moon : ℕ) (active_days_initial : ℕ) : 
  active_days_initial =  9 * (9 + 4) → 
  level_new_star = 1 → 
  level_new_moon = 2 → 
  ∃ (days_required : ℕ), 
    (days_required + active_days_initial = 16 * (16 + 4)) ∧ (days_required = 203) :=
by
  sorry

end upgrade_days_to_sun_l1883_188335


namespace value_of_a_plus_b_l1883_188315

theorem value_of_a_plus_b (a b : ℕ) (h1 : Real.sqrt 44 = 2 * Real.sqrt a) (h2 : Real.sqrt 54 = 3 * Real.sqrt b) : a + b = 17 := 
sorry

end value_of_a_plus_b_l1883_188315


namespace factor_expression_l1883_188383

variable (x : ℝ)

theorem factor_expression :
  (4 * x ^ 3 + 100 * x ^ 2 - 28) - (-9 * x ^ 3 + 2 * x ^ 2 - 28) = 13 * x ^ 2 * (x + 7) :=
by
  sorry

end factor_expression_l1883_188383


namespace ratio_of_speeds_l1883_188330

theorem ratio_of_speeds (v1 v2 : ℝ) (h1 : v1 > v2) (h2 : 8 = (v1 + v2) * 2) (h3 : 8 = (v1 - v2) * 4) : v1 / v2 = 3 :=
by
  sorry

end ratio_of_speeds_l1883_188330


namespace rhombus_area_l1883_188370

theorem rhombus_area (R r : ℝ) : 
  ∃ A : ℝ, A = (8 * R^3 * r^3) / ((R^2 + r^2)^2) :=
by
  sorry

end rhombus_area_l1883_188370


namespace probability_of_two_black_balls_is_one_fifth_l1883_188387

noncomputable def probability_of_two_black_balls (W B : Nat) : ℚ :=
  let total_balls := W + B
  let prob_black1 := (B : ℚ) / total_balls
  let prob_black2_given_black1 := (B - 1 : ℚ) / (total_balls - 1)
  prob_black1 * prob_black2_given_black1

theorem probability_of_two_black_balls_is_one_fifth : 
  probability_of_two_black_balls 8 7 = 1 / 5 := 
by
  sorry

end probability_of_two_black_balls_is_one_fifth_l1883_188387


namespace third_smallest_four_digit_in_pascal_l1883_188322

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l1883_188322


namespace pythagorean_triple_example_l1883_188309

noncomputable def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_example :
  is_pythagorean_triple 5 12 13 :=
by
  sorry

end pythagorean_triple_example_l1883_188309


namespace heather_payment_per_weed_l1883_188392

noncomputable def seconds_in_hour : ℕ := 60 * 60

noncomputable def weeds_per_hour (seconds_per_weed : ℕ) : ℕ :=
  seconds_in_hour / seconds_per_weed

noncomputable def payment_per_weed (hourly_pay : ℕ) (weeds_per_hour : ℕ) : ℚ :=
  hourly_pay / weeds_per_hour

theorem heather_payment_per_weed (seconds_per_weed : ℕ) (hourly_pay : ℕ) :
  seconds_per_weed = 18 ∧ hourly_pay = 10 → payment_per_weed hourly_pay (weeds_per_hour seconds_per_weed) = 0.05 :=
by
  sorry

end heather_payment_per_weed_l1883_188392


namespace brownies_total_l1883_188360

theorem brownies_total :
  let initial_brownies := 2 * 12
  let after_father_ate := initial_brownies - 8
  let after_mooney_ate := after_father_ate - 4
  let additional_brownies := 2 * 12
  after_mooney_ate + additional_brownies = 36 :=
by
  let initial_brownies := 2 * 12
  let after_father_ate := initial_brownies - 8
  let after_mooney_ate := after_father_ate - 4
  let additional_brownies := 2 * 12
  show after_mooney_ate + additional_brownies = 36
  sorry

end brownies_total_l1883_188360


namespace range_of_a_l1883_188327

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x - a) * (1 - x - a) < 1) → -1/2 < a ∧ a < 3/2 :=
by
  sorry

end range_of_a_l1883_188327


namespace solve_for_x_l1883_188389

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 = -2 * x + 11) : x = 3 := 
sorry

end solve_for_x_l1883_188389


namespace determinant_matrix_zero_l1883_188314

theorem determinant_matrix_zero (θ φ : ℝ) : 
  Matrix.det ![
    ![0, Real.cos θ, -Real.sin θ],
    ![-Real.cos θ, 0, Real.cos φ],
    ![Real.sin θ, -Real.cos φ, 0]
  ] = 0 := by sorry

end determinant_matrix_zero_l1883_188314


namespace INPUT_is_input_statement_l1883_188346

-- Define what constitutes each type of statement
def isOutputStatement (stmt : String) : Prop :=
  stmt = "PRINT"

def isInputStatement (stmt : String) : Prop :=
  stmt = "INPUT"

def isConditionalStatement (stmt : String) : Prop :=
  stmt = "THEN"

def isEndStatement (stmt : String) : Prop :=
  stmt = "END"

-- The main theorem
theorem INPUT_is_input_statement : isInputStatement "INPUT" := by
  sorry

end INPUT_is_input_statement_l1883_188346


namespace area_of_triangle_XPQ_l1883_188384
open Real

/-- Given a triangle XYZ with area 15 square units and points P, Q, R on sides XY, YZ, and ZX respectively,
where XP = 3, PY = 6, and triangles XPQ and quadrilateral PYRQ have equal areas, 
prove that the area of triangle XPQ is 5/3 square units. -/
theorem area_of_triangle_XPQ 
  (Area_XYZ : ℝ) (h1 : Area_XYZ = 15)
  (XP PY : ℝ) (h2 : XP = 3) (h3 : PY = 6)
  (h4 : ∃ (Area_XPQ : ℝ) (Area_PYRQ : ℝ), Area_XPQ = Area_PYRQ) :
  ∃ (Area_XPQ : ℝ), Area_XPQ = 5/3 :=
sorry

end area_of_triangle_XPQ_l1883_188384


namespace solve_equation1_solve_equation2_l1883_188304

noncomputable def solutions_equation1 : Set ℝ := { x | x^2 - 2 * x - 8 = 0 }
noncomputable def solutions_equation2 : Set ℝ := { x | x^2 - 2 * x - 5 = 0 }

theorem solve_equation1 :
  solutions_equation1 = {4, -2} := 
by
  sorry

theorem solve_equation2 :
  solutions_equation2 = {1 + Real.sqrt 6, 1 - Real.sqrt 6} :=
by
  sorry

end solve_equation1_solve_equation2_l1883_188304


namespace domain_of_f_l1883_188323

-- The domain of the function is the set of all x such that the function is defined.
theorem domain_of_f:
  {x : ℝ | x > 3 ∧ x ≠ 4} = (Set.Ioo 3 4 ∪ Set.Ioi 4) := 
sorry

end domain_of_f_l1883_188323


namespace mod_equiv_pow_five_l1883_188393

theorem mod_equiv_pow_five (m : ℤ) (hm : 0 ≤ m ∧ m < 11) (h : 12^5 ≡ m [ZMOD 11]) : m = 1 :=
by
  sorry

end mod_equiv_pow_five_l1883_188393


namespace min_value_at_x_eq_2_l1883_188305

theorem min_value_at_x_eq_2 (x : ℝ) (h : x > 1) : 
  x + 1/(x-1) = 3 ↔ x = 2 :=
by sorry

end min_value_at_x_eq_2_l1883_188305


namespace find_m_root_zero_l1883_188339

theorem find_m_root_zero (m : ℝ) : (m - 1) * 0 ^ 2 + 0 + m ^ 2 - 1 = 0 → m = -1 :=
by
  intro h
  sorry

end find_m_root_zero_l1883_188339
