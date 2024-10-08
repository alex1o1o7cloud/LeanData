import Mathlib

namespace positive_rational_as_sum_of_cubes_l197_197016

theorem positive_rational_as_sum_of_cubes (q : ℚ) (h_q_pos : q > 0) : 
  ∃ (a b c d : ℤ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ q = ((a^3 + b^3) / (c^3 + d^3)) :=
sorry

end positive_rational_as_sum_of_cubes_l197_197016


namespace Ali_possible_scores_l197_197575

-- Defining the conditions
def categories := 5
def questions_per_category := 3
def correct_answers_points := 12
def total_questions := categories * questions_per_category
def incorrect_answers := total_questions - correct_answers_points

-- Defining the bonuses based on cases

-- All 3 incorrect answers in 1 category
def case_1_bonus := 4
def case_1_total := correct_answers_points + case_1_bonus

-- 3 incorrect answers split into 2 categories
def case_2_bonus := 3
def case_2_total := correct_answers_points + case_2_bonus

-- 3 incorrect answers split into 3 categories
def case_3_bonus := 2
def case_3_total := correct_answers_points + case_3_bonus

theorem Ali_possible_scores : 
  case_1_total = 16 ∧ case_2_total = 15 ∧ case_3_total = 14 :=
by
  -- Skipping the proof here
  sorry

end Ali_possible_scores_l197_197575


namespace gcd_360_128_is_8_l197_197548

def gcd_360_128 : ℕ :=
  gcd 360 128

theorem gcd_360_128_is_8 : gcd_360_128 = 8 :=
  by
    -- Proof goes here (use sorry for now)
    sorry

end gcd_360_128_is_8_l197_197548


namespace rancher_monetary_loss_l197_197993

def rancher_head_of_cattle := 500
def market_rate_per_head := 700
def sick_cattle := 350
def additional_cost_per_sick_animal := 80
def reduced_price_per_head := 450

def expected_revenue := rancher_head_of_cattle * market_rate_per_head
def loss_from_death := sick_cattle * market_rate_per_head
def additional_sick_cost := sick_cattle * additional_cost_per_sick_animal
def remaining_cattle := rancher_head_of_cattle - sick_cattle
def revenue_from_remaining_cattle := remaining_cattle * reduced_price_per_head

def total_loss := (expected_revenue - revenue_from_remaining_cattle) + additional_sick_cost

theorem rancher_monetary_loss : total_loss = 310500 := by
  sorry

end rancher_monetary_loss_l197_197993


namespace volume_rectangular_solid_l197_197538

theorem volume_rectangular_solid
  (a b c : ℝ) 
  (h1 : a * b = 12)
  (h2 : b * c = 8)
  (h3 : a * c = 6) :
  a * b * c = 24 :=
sorry

end volume_rectangular_solid_l197_197538


namespace JimSiblings_l197_197999

-- Define the students and their characteristics.
structure Student :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)
  (wearsGlasses : Bool)

def Benjamin : Student := ⟨"Benjamin", "Blue", "Blond", true⟩
def Jim : Student := ⟨"Jim", "Brown", "Blond", false⟩
def Nadeen : Student := ⟨"Nadeen", "Brown", "Black", true⟩
def Austin : Student := ⟨"Austin", "Blue", "Black", false⟩
def Tevyn : Student := ⟨"Tevyn", "Blue", "Blond", true⟩
def Sue : Student := ⟨"Sue", "Brown", "Blond", false⟩

-- Define the condition that students from the same family share at least one characteristic.
def shareCharacteristic (s1 s2 : Student) : Bool :=
  (s1.eyeColor = s2.eyeColor) ∨
  (s1.hairColor = s2.hairColor) ∨
  (s1.wearsGlasses = s2.wearsGlasses)

-- Define what it means to be siblings of a student.
def areSiblings (s1 s2 s3 : Student) : Bool :=
  shareCharacteristic s1 s2 ∧
  shareCharacteristic s1 s3 ∧
  shareCharacteristic s2 s3

-- The theorem we are trying to prove.
theorem JimSiblings : areSiblings Jim Sue Benjamin = true := 
  by sorry

end JimSiblings_l197_197999


namespace puppies_given_l197_197693

-- Definitions of the initial and left numbers of puppies
def initial_puppies : ℕ := 7
def left_puppies : ℕ := 2

-- Theorem stating that the number of puppies given to friends is the difference
theorem puppies_given : initial_puppies - left_puppies = 5 := by
  sorry -- Proof not required, so we use sorry

end puppies_given_l197_197693


namespace sum_of_reciprocals_eq_three_l197_197569

theorem sum_of_reciprocals_eq_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  (1/x + 1/y) = 3 := 
by
  sorry

end sum_of_reciprocals_eq_three_l197_197569


namespace perfect_squares_solutions_l197_197682

noncomputable def isPerfectSquare (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

theorem perfect_squares_solutions :
  ∀ (a b : ℕ),
    0 < a → 0 < b →
    (isPerfectSquare (↑a * ↑a - 4 * ↑b)) →
    (isPerfectSquare (↑b * ↑b - 4 * ↑a)) →
      (a = 4 ∧ b = 4) ∨
      (a = 5 ∧ b = 6) ∨
      (a = 6 ∧ b = 5) :=
by
  -- Proof omitted
  sorry

end perfect_squares_solutions_l197_197682


namespace train_passes_man_in_approx_18_seconds_l197_197137

noncomputable def train_length : ℝ := 300 -- meters
noncomputable def train_speed : ℝ := 68 -- km/h
noncomputable def man_speed : ℝ := 8 -- km/h
noncomputable def kmh_to_mps (v : ℝ) : ℝ := v * 1000 / 3600
noncomputable def relative_speed_mps : ℝ := kmh_to_mps (train_speed - man_speed)
noncomputable def time_to_pass_man : ℝ := train_length / relative_speed_mps

theorem train_passes_man_in_approx_18_seconds :
  abs (time_to_pass_man - 18) < 1 :=
by
  sorry

end train_passes_man_in_approx_18_seconds_l197_197137


namespace max_value_of_y_l197_197629

noncomputable def max_y (x y : ℝ) : ℝ :=
  if h : x^2 + y^2 = 20*x + 54*y then y else 0

theorem max_value_of_y (x y : ℝ) (h : x^2 + y^2 = 20*x + 54*y) :
  max_y x y ≤ 27 + Real.sqrt 829 :=
sorry

end max_value_of_y_l197_197629


namespace probability_of_blue_or_yellow_l197_197777

def num_red : ℕ := 6
def num_green : ℕ := 7
def num_yellow : ℕ := 8
def num_blue : ℕ := 9

def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue
def total_blue_or_yellow : ℕ := num_yellow + num_blue

theorem probability_of_blue_or_yellow (h : total_jelly_beans ≠ 0) : 
  (total_blue_or_yellow : ℚ) / (total_jelly_beans : ℚ) = 17 / 30 :=
by
  sorry

end probability_of_blue_or_yellow_l197_197777


namespace sequence_product_l197_197293

theorem sequence_product {n : ℕ} (h : 1 < n) (a : ℕ → ℕ) (h₀ : ∀ n, a n = 2^n) : 
  a (n-1) * a (n+1) = 4^n :=
by sorry

end sequence_product_l197_197293


namespace books_at_end_of_month_l197_197772

-- Definitions based on provided conditions
def initial_books : ℕ := 75
def loaned_books (x : ℕ) : ℕ := 40  -- Rounded from 39.99999999999999
def returned_books (x : ℕ) : ℕ := (loaned_books x * 70) / 100
def not_returned_books (x : ℕ) : ℕ := loaned_books x - returned_books x

-- The statement to be proved
theorem books_at_end_of_month (x : ℕ) : initial_books - not_returned_books x = 63 :=
by
  -- This will be filled in with the actual proof steps later
  sorry

end books_at_end_of_month_l197_197772


namespace largest_good_number_smallest_bad_number_l197_197252

def is_good_number (M : ℕ) : Prop :=
  ∃ a b c d : ℤ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

theorem largest_good_number :
  ∀ M : ℕ, is_good_number M ↔ M ≤ 576 :=
by sorry

theorem smallest_bad_number :
  ∀ M : ℕ, ¬ is_good_number M ↔ M ≥ 443 :=
by sorry

end largest_good_number_smallest_bad_number_l197_197252


namespace total_peaches_l197_197342

-- Definitions based on the given conditions
def initial_peaches : ℕ := 13
def picked_peaches : ℕ := 55

-- The proof goal stating the total number of peaches now
theorem total_peaches : initial_peaches + picked_peaches = 68 := by
  sorry

end total_peaches_l197_197342


namespace no_naturals_satisfy_divisibility_condition_l197_197626

theorem no_naturals_satisfy_divisibility_condition :
  ∀ (a b c : ℕ), ¬ (2013 * (a * b + b * c + c * a) ∣ a^2 + b^2 + c^2) :=
by
  sorry

end no_naturals_satisfy_divisibility_condition_l197_197626


namespace converse_of_proposition_inverse_of_proposition_contrapositive_of_proposition_l197_197297

variable (a b : ℝ)

theorem converse_of_proposition :
  (ab > 0 → a > 0 ∧ b > 0) = false := sorry

theorem inverse_of_proposition :
  (a ≤ 0 ∨ b ≤ 0 → ab ≤ 0) = false := sorry

theorem contrapositive_of_proposition :
  (ab ≤ 0 → a ≤ 0 ∨ b ≤ 0) = true := sorry

end converse_of_proposition_inverse_of_proposition_contrapositive_of_proposition_l197_197297


namespace ratio_of_middle_angle_l197_197410

theorem ratio_of_middle_angle (A B C : ℝ) 
  (h1 : A + B + C = 180)
  (h2 : C = 5 * A)
  (h3 : A = 20) :
  B / A = 3 :=
by
  sorry

end ratio_of_middle_angle_l197_197410


namespace sides_of_original_polygon_l197_197605

-- Define the sum of interior angles formula for a polygon with n sides
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the total sum of angles for the resulting polygon
def sum_of_new_polygon_angles : ℝ := 1980

-- The lean theorem statement to prove
theorem sides_of_original_polygon (n : ℕ) :
    sum_interior_angles n = sum_of_new_polygon_angles →
    n = 13 →
    12 ≤ n+1 ∧ n+1 ≤ 14 :=
by
  intro h1 h2
  sorry

end sides_of_original_polygon_l197_197605


namespace joshua_skittles_l197_197652

theorem joshua_skittles (eggs : ℝ) (skittles_per_friend : ℝ) (friends : ℝ) (h1 : eggs = 6.0) (h2 : skittles_per_friend = 40.0) (h3 : friends = 5.0) : skittles_per_friend * friends = 200.0 := 
by 
  sorry

end joshua_skittles_l197_197652


namespace arithmetic_sequence_and_sum_l197_197426

noncomputable def a_n (n : ℕ) : ℤ := 2 * n + 10

def S_n (n : ℕ) : ℤ := n * (12 + 2 * n + 10) / 2

theorem arithmetic_sequence_and_sum :
    (a_n 10 = 30) ∧ 
    (a_n 20 = 50) ∧ 
    (∀ n, S_n n = 11 * n + n^2) ∧ 
    (S_n 3 = 42) :=
by {
    -- a_n 10 = 2 * 10 + 10 = 30
    -- a_n 20 = 2 * 20 + 10 = 50
    -- S_n n = n * (2n + 22) / 2 = 11n + n^2
    -- S_n 3 = 3 * 14 = 42
    sorry
}

end arithmetic_sequence_and_sum_l197_197426


namespace find_a_10_l197_197364

theorem find_a_10 (a : ℕ → ℚ)
  (h0 : a 1 = 1)
  (h1 : ∀ n : ℕ, a (n + 1) = a n / (a n + 2)) :
  a 10 = 1 / 1023 :=
sorry

end find_a_10_l197_197364


namespace lisa_savings_l197_197049

-- Define the conditions
def originalPricePerNotebook : ℝ := 3
def numberOfNotebooks : ℕ := 8
def discountRate : ℝ := 0.30
def additionalDiscount : ℝ := 5

-- Define the total savings calculation
def calculateSavings (originalPricePerNotebook : ℝ) (numberOfNotebooks : ℕ) (discountRate : ℝ) (additionalDiscount : ℝ) : ℝ := 
  let totalPriceWithoutDiscount := originalPricePerNotebook * numberOfNotebooks
  let discountedPricePerNotebook := originalPricePerNotebook * (1 - discountRate)
  let totalPriceWith30PercentDiscount := discountedPricePerNotebook * numberOfNotebooks
  let totalPriceWithAllDiscounts := totalPriceWith30PercentDiscount - additionalDiscount
  totalPriceWithoutDiscount - totalPriceWithAllDiscounts

-- Theorem for the proof problem
theorem lisa_savings :
  calculateSavings originalPricePerNotebook numberOfNotebooks discountRate additionalDiscount = 12.20 :=
by
  -- Inserting the proof as sorry
  sorry

end lisa_savings_l197_197049


namespace volume_in_cubic_yards_l197_197972

-- Define the conditions given in the problem
def volume_in_cubic_feet : ℕ := 216
def cubic_feet_per_cubic_yard : ℕ := 27

-- Define the theorem that needs to be proven
theorem volume_in_cubic_yards :
  volume_in_cubic_feet / cubic_feet_per_cubic_yard = 8 :=
by
  sorry

end volume_in_cubic_yards_l197_197972


namespace james_muffins_baked_l197_197653

theorem james_muffins_baked (arthur_muffins : ℝ) (factor : ℝ) (h1 : arthur_muffins = 115.0) (h2 : factor = 12.0) :
  (arthur_muffins / factor) = 9.5833 :=
by 
  -- using the conditions given, we would proceed to prove the result:
  -- sorry is used to indicate that the proof is omitted here
  sorry

end james_muffins_baked_l197_197653


namespace derivative_odd_function_l197_197827

theorem derivative_odd_function (a b c : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x^3 + b * x^2 + c * x + 2) 
    (h_deriv_odd : ∀ x, deriv f (-x) = - deriv f x) : a^2 + c^2 ≠ 0 :=
by
  sorry

end derivative_odd_function_l197_197827


namespace divide_group_among_boats_l197_197784
noncomputable def number_of_ways_divide_group 
  (boatA_capacity : ℕ) 
  (boatB_capacity : ℕ) 
  (boatC_capacity : ℕ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (constraint : ∀ {boat : ℕ}, boat > 1 → num_children ≥ 1 → num_adults ≥ 1) : ℕ := 
    sorry

theorem divide_group_among_boats 
  (boatA_capacity : ℕ := 3) 
  (boatB_capacity : ℕ := 2) 
  (boatC_capacity : ℕ := 1) 
  (num_adults : ℕ := 2) 
  (num_children : ℕ := 2) 
  (constraint : ∀ {boat : ℕ}, boat > 1 → num_children ≥ 1 → num_adults ≥ 1) : 
  number_of_ways_divide_group boatA_capacity boatB_capacity boatC_capacity num_adults num_children constraint = 8 := 
sorry

end divide_group_among_boats_l197_197784


namespace total_bins_used_l197_197892

def bins_of_soup : ℝ := 0.12
def bins_of_vegetables : ℝ := 0.12
def bins_of_pasta : ℝ := 0.5

theorem total_bins_used : bins_of_soup + bins_of_vegetables + bins_of_pasta = 0.74 :=
by
  sorry

end total_bins_used_l197_197892


namespace day_of_week_proof_l197_197273

def day_of_week_17th_2003 := "Wednesday"
def day_of_week_305th_2003 := "Thursday"

theorem day_of_week_proof (d17 : day_of_week_17th_2003 = "Wednesday") : day_of_week_305th_2003 = "Thursday" := 
sorry

end day_of_week_proof_l197_197273


namespace convert_20202_3_l197_197102

def ternary_to_decimal (a4 a3 a2 a1 a0 : ℕ) : ℕ :=
  a4 * 3^4 + a3 * 3^3 + a2 * 3^2 + a1 * 3^1 + a0 * 3^0

theorem convert_20202_3 : ternary_to_decimal 2 0 2 0 2 = 182 :=
  sorry

end convert_20202_3_l197_197102


namespace people_at_table_l197_197182

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l197_197182


namespace smallest_value_of_expression_l197_197811

theorem smallest_value_of_expression :
  ∃ (k l : ℕ), 36^k - 5^l = 11 := 
sorry

end smallest_value_of_expression_l197_197811


namespace max_value_of_f_l197_197305

noncomputable def f (x : ℝ) : ℝ := 10 * x - 2 * x^2

theorem max_value_of_f : ∃ x : ℝ, f x = 12.5 :=
by
  sorry

end max_value_of_f_l197_197305


namespace total_profit_amount_l197_197638

-- Definitions representing the conditions:
def ratio_condition (P_X P_Y : ℝ) : Prop :=
  P_X / P_Y = (1 / 2) / (1 / 3)

def difference_condition (P_X P_Y : ℝ) : Prop :=
  P_X - P_Y = 160

-- The proof problem statement:
theorem total_profit_amount (P_X P_Y : ℝ) (h1 : ratio_condition P_X P_Y) (h2 : difference_condition P_X P_Y) :
  P_X + P_Y = 800 := by
  sorry

end total_profit_amount_l197_197638


namespace group_discount_l197_197940

theorem group_discount (P : ℝ) (D : ℝ) :
  4 * (P - (D / 100) * P) = 3 * P → D = 25 :=
by
  intro h
  sorry

end group_discount_l197_197940


namespace eliot_account_balance_l197_197121

theorem eliot_account_balance (A E : ℝ) 
  (h1 : A > E)
  (h2 : A - E = (1 / 12) * (A + E))
  (h3 : 1.10 * A - 1.15 * E = 22) : 
  E = 146.67 :=
by
  sorry

end eliot_account_balance_l197_197121


namespace parabola_coefficients_sum_l197_197852

theorem parabola_coefficients_sum (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ y = a * (x + 3)^2 + 2) ∧
  (-6 = a * (1 + 3)^2 + 2) →
  a + b + c = -11/2 :=
by
  sorry

end parabola_coefficients_sum_l197_197852


namespace sum_smallest_largest_even_integers_l197_197200

theorem sum_smallest_largest_even_integers (m b z : ℕ) (hm_even : m % 2 = 0)
  (h_mean : z = (b + (b + 2 * (m - 1))) / 2) :
  (b + (b + 2 * (m - 1))) = 2 * z :=
by
  sorry

end sum_smallest_largest_even_integers_l197_197200


namespace larry_wins_probability_l197_197502

noncomputable def probability_larry_wins (p_L : ℚ) (p_J : ℚ) : ℚ :=
  let q_L := 1 - p_L
  let q_J := 1 - p_J
  let r := q_L * q_J
  p_L / (1 - r)

theorem larry_wins_probability
  (p_L : ℚ) (p_J : ℚ) (h1 : p_L = 3 / 5) (h2 : p_J = 1 / 3) :
  probability_larry_wins p_L p_J = 9 / 11 :=
by 
  sorry

end larry_wins_probability_l197_197502


namespace find_S9_l197_197417

variables {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Condition: an arithmetic sequence with the sum of first n terms S_n.
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition: a_3 + a_4 + a_5 + a_6 + a_7 = 20.
def given_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 20

-- The sum of the first n terms.
def sum_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n / 2 : ℝ) * (a 1 + a n)

-- Prove that S_9 = 36.
theorem find_S9 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arithmetic_sequence : arithmetic_sequence a) 
  (h_given_condition : given_condition a)
  (h_sum_terms : sum_terms S a) : 
  S 9 = 36 :=
sorry

end find_S9_l197_197417


namespace sum_of_numbers_l197_197929

theorem sum_of_numbers {a b c : ℝ} (h1 : b = 7) (h2 : (a + b + c) / 3 = a + 8) (h3 : (a + b + c) / 3 = c - 20) : a + b + c = 57 :=
sorry

end sum_of_numbers_l197_197929


namespace right_triangle_perimeter_l197_197352

noncomputable def perimeter_right_triangle (a b : ℝ) (hypotenuse : ℝ) : ℝ :=
  a + b + hypotenuse

theorem right_triangle_perimeter (a b : ℝ) (ha : a^2 + b^2 = 25) (hab : a * b = 10) (hhypotenuse : hypotenuse = 5) :
  perimeter_right_triangle a b hypotenuse = 5 + 3 * Real.sqrt 5 :=
by
  sorry

end right_triangle_perimeter_l197_197352


namespace fabric_difference_fabric_total_l197_197910

noncomputable def fabric_used_coat : ℝ := 1.55
noncomputable def fabric_used_pants : ℝ := 1.05

theorem fabric_difference : fabric_used_coat - fabric_used_pants = 0.5 :=
by
  sorry

theorem fabric_total : fabric_used_coat + fabric_used_ppants = 2.6 :=
by
  sorry

end fabric_difference_fabric_total_l197_197910


namespace circle_area_radius_8_l197_197319

variable (r : ℝ) (π : ℝ)

theorem circle_area_radius_8 : r = 8 → (π * r^2) = 64 * π :=
by
  sorry

end circle_area_radius_8_l197_197319


namespace math_problem_l197_197329

theorem math_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4*x + y) / (x - 4*y) = -3) : 
  (x + 4*y) / (4*x - y) = 39 / 37 :=
by
  sorry

end math_problem_l197_197329


namespace sandy_fingernails_length_l197_197170

/-- 
Sandy, who just turned 12 this month, has a goal for tying the world record for longest fingernails, 
which is 26 inches. Her fingernails grow at a rate of one-tenth of an inch per month. 
She will be 32 when she achieves the world record. 
Prove that her fingernails are currently 2 inches long.
-/
theorem sandy_fingernails_length 
  (current_age : ℕ) (world_record_length : ℝ) (growth_rate : ℝ) (years_to_achieve : ℕ) : 
  current_age = 12 → 
  world_record_length = 26 → 
  growth_rate = 0.1 → 
  years_to_achieve = 20 →
  (world_record_length - growth_rate * 12 * years_to_achieve) = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_fingernails_length_l197_197170


namespace keiko_speed_l197_197544

theorem keiko_speed (a b : ℝ) (s : ℝ) (h1 : (2 * a + 2 * π * (b + 8)) / s = (2 * a + 2 * π * b) / s + 48) : s = π / 3 :=
by {
  sorry -- proof is not required
}

end keiko_speed_l197_197544


namespace locus_of_point_parabola_l197_197206

/-- If the distance from point P to the point F (4, 0) is one unit less than its distance to the line x + 5 = 0, then the equation of the locus of point P is y^2 = 16x. -/
theorem locus_of_point_parabola :
  ∀ P : ℝ × ℝ, dist P (4, 0) + 1 = abs (P.1 + 5) → P.2^2 = 16 * P.1 :=
by
  sorry

end locus_of_point_parabola_l197_197206


namespace correct_answer_l197_197243

-- Define the problem conditions and question
def equation (y : ℤ) : Prop := y + 2 = -3

-- Prove that the correct answer is y = -5
theorem correct_answer : ∀ y : ℤ, equation y → y = -5 :=
by
  intros y h
  unfold equation at h
  linarith

end correct_answer_l197_197243


namespace james_total_payment_is_correct_l197_197220

-- Define the constants based on the conditions
def numDirtBikes : Nat := 3
def costPerDirtBike : Nat := 150
def numOffRoadVehicles : Nat := 4
def costPerOffRoadVehicle : Nat := 300
def numTotalVehicles : Nat := numDirtBikes + numOffRoadVehicles
def registrationCostPerVehicle : Nat := 25

-- Define the total calculation using the given conditions
def totalPaidByJames : Nat :=
  (numDirtBikes * costPerDirtBike) +
  (numOffRoadVehicles * costPerOffRoadVehicle) +
  (numTotalVehicles * registrationCostPerVehicle)

-- State the proof problem
theorem james_total_payment_is_correct : totalPaidByJames = 1825 := by
  sorry

end james_total_payment_is_correct_l197_197220


namespace initial_cookies_count_l197_197057

def cookies_left : ℕ := 9
def cookies_eaten : ℕ := 9

theorem initial_cookies_count : cookies_left + cookies_eaten = 18 :=
by sorry

end initial_cookies_count_l197_197057


namespace sum_of_cubes_eq_three_l197_197040

theorem sum_of_cubes_eq_three (k : ℤ) : 
  (1 + 6 * k^3)^3 + (1 - 6 * k^3)^3 + (-6 * k^2)^3 + 1^3 = 3 :=
by 
  sorry

end sum_of_cubes_eq_three_l197_197040


namespace norma_initial_cards_l197_197568

theorem norma_initial_cards (x : ℝ) 
  (H1 : x + 70 = 158) : 
  x = 88 :=
by
  sorry

end norma_initial_cards_l197_197568


namespace cost_price_percentage_l197_197674

theorem cost_price_percentage (SP CP : ℝ) (hp : SP - CP = (1/3) * CP) : CP = 0.75 * SP :=
by
  sorry

end cost_price_percentage_l197_197674


namespace total_students_l197_197665

-- Define the problem statement in Lean 4
theorem total_students (n : ℕ) (h1 : n < 400)
  (h2 : n % 17 = 15) (h3 : n % 19 = 10) : n = 219 :=
sorry

end total_students_l197_197665


namespace find_x_when_fx_eq_3_l197_197013

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ -1 then x + 2 else
if x < 2 then x^2 else
2 * x

theorem find_x_when_fx_eq_3 : ∃ x : ℝ, f x = 3 ∧ x = Real.sqrt 3 := by
  sorry

end find_x_when_fx_eq_3_l197_197013


namespace minimal_bananas_l197_197968

noncomputable def total_min_bananas : ℕ :=
  let b1 := 72
  let b2 := 72
  let b3 := 216
  let b4 := 72
  b1 + b2 + b3 + b4

theorem minimal_bananas (total_bananas : ℕ) (ratio1 ratio2 ratio3 ratio4 : ℕ) 
  (b1 b2 b3 b4 : ℕ) 
  (h_ratio : ratio1 = 4 ∧ ratio2 = 3 ∧ ratio3 = 2 ∧ ratio4 = 1) 
  (h_div_constraints : ∀ n m : ℕ, (n % m = 0 ∨ m % n = 0) ∧ n ≥ ratio1 * ratio2 * ratio3 * ratio4) 
  (h_bananas : b1 = 72 ∧ b2 = 72 ∧ b3 = 216 ∧ b4 = 72 ∧ 
              4 * (b1 / 2 + b2 / 6 + b3 / 9 + 7 * b4 / 72) = 3 * (b1 / 6 + b2 / 3 + b3 / 9 + 7 * b4 / 72) ∧ 
              2 * (b1 / 6 + b2 / 6 + b3 / 6 + 7 * b4 / 72) = (b1 / 6 + b2 / 6 + b3 / 9 + b4 / 8)) : 
  total_bananas = 432 := by
  sorry

end minimal_bananas_l197_197968


namespace necessary_condition_abs_sq_necessary_and_sufficient_add_l197_197807

theorem necessary_condition_abs_sq (a b : ℝ) : a^2 > b^2 → |a| > |b| :=
sorry

theorem necessary_and_sufficient_add (a b c : ℝ) :
  (a > b) ↔ (a + c > b + c) :=
sorry

end necessary_condition_abs_sq_necessary_and_sufficient_add_l197_197807


namespace appointment_duration_l197_197018

-- Define the given conditions
def total_workday_hours : ℕ := 8
def permits_per_hour : ℕ := 50
def total_permits : ℕ := 100
def stamping_time : ℕ := total_permits / permits_per_hour
def appointment_time : ℕ := (total_workday_hours - stamping_time) / 2

-- State the theorem and ignore the proof part by adding sorry
theorem appointment_duration : appointment_time = 3 := by
  -- skipping the proof steps
  sorry

end appointment_duration_l197_197018


namespace curve_trajectory_a_eq_1_curve_fixed_point_a_ne_1_l197_197649

noncomputable def curve (x y a : ℝ) : ℝ :=
  x^2 + y^2 - 2 * a * x + 2 * (a - 2) * y + 2 

theorem curve_trajectory_a_eq_1 :
  ∃! (x y : ℝ), curve x y 1 = 0 ∧ x = 1 ∧ y = 1 := by
  sorry

theorem curve_fixed_point_a_ne_1 (a : ℝ) (ha : a ≠ 1) :
  curve 1 1 a = 0 := by
  sorry

end curve_trajectory_a_eq_1_curve_fixed_point_a_ne_1_l197_197649


namespace cows_in_group_l197_197797

theorem cows_in_group (D C : ℕ) 
  (h : 2 * D + 4 * C = 2 * (D + C) + 36) : 
  C = 18 :=
by
  sorry

end cows_in_group_l197_197797


namespace equal_sunday_tuesday_count_l197_197141

theorem equal_sunday_tuesday_count (h : ∀ (d : ℕ), d < 7 → d ≠ 0 → d ≠ 1 → d ≠ 2 → d ≠ 3) :
  ∃! d, d = 4 :=
by
  -- proof here
  sorry

end equal_sunday_tuesday_count_l197_197141


namespace temperature_difference_l197_197027

def highest_temperature : ℤ := 8
def lowest_temperature : ℤ := -2

theorem temperature_difference :
  highest_temperature - lowest_temperature = 10 := by
  sorry

end temperature_difference_l197_197027


namespace find_sum_of_vars_l197_197056

-- Definitions of the quadratic polynomials
def quadratic1 (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 11
def quadratic2 (y : ℝ) : ℝ := y^2 - 10 * y + 29
def quadratic3 (z : ℝ) : ℝ := 3 * z^2 - 18 * z + 32

-- Theorem statement
theorem find_sum_of_vars (x y z : ℝ) :
  quadratic1 x * quadratic2 y * quadratic3 z ≤ 60 → x + y - z = 0 :=
by 
-- here we would complete the proof steps
sorry

end find_sum_of_vars_l197_197056


namespace solve_triangle_problem_l197_197808
noncomputable def triangle_problem (A B C a b c : ℝ) (area : ℝ) : Prop :=
  (2 * c * Real.sin B * Real.cos A - b * Real.sin C = 0) ∧
  area = Real.sqrt 3 ∧ 
  b + c = 5 →
  (A = Real.pi / 3) ∧ (a = Real.sqrt 13)

-- Lean statement for the proof problem
theorem solve_triangle_problem 
  (A B C a b c : ℝ) 
  (h1 : 2 * c * Real.sin B * Real.cos A - b * Real.sin C = 0)
  (h2 : 1/2 * b * c * Real.sin A = Real.sqrt 3)
  (h3 : b + c = 5) :
  A = Real.pi / 3 ∧ a = Real.sqrt 13 :=
sorry

end solve_triangle_problem_l197_197808


namespace cost_price_of_article_l197_197201

theorem cost_price_of_article (x : ℝ) (h : 57 - x = x - 43) : x = 50 := 
by 
  sorry

end cost_price_of_article_l197_197201


namespace average_alligators_l197_197478

theorem average_alligators (t s n : ℕ) (h1 : t = 50) (h2 : s = 20) (h3 : n = 3) :
  (t - s) / n = 10 :=
by 
  sorry

end average_alligators_l197_197478


namespace like_terms_eq_l197_197382

theorem like_terms_eq : 
  ∀ (x y : ℕ), 
  (x + 2 * y = 3) → 
  (2 * x + y = 9) → 
  (x + y = 4) :=
by
  intros x y h1 h2
  sorry

end like_terms_eq_l197_197382


namespace max_values_of_f_smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l197_197889

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)

theorem max_values_of_f (k : ℤ) : 
  ∃ x, f x = 2 ∧ x = 4 * (k : ℝ) * Real.pi - (2 * Real.pi / 3) := 
sorry

theorem smallest_positive_period_of_f : 
  ∃ T, T = 4 * Real.pi := 
sorry

theorem intervals_where_f_is_monotonically_increasing (k : ℤ) : 
  ∀ x, (-(5 * Real.pi / 3) + 4 * (k : ℝ) * Real.pi ≤ x) ∧ (x ≤ Real.pi / 3 + 4 * (k : ℝ) * Real.pi) → 
  ∀ y, (-(5 * Real.pi / 3) + 4 * (k : ℝ) * Real.pi ≤ y) ∧ (y ≤ Real.pi / 3 + 4 * (k : ℝ) * Real.pi) → 
  (x ≤ y ↔ f x ≤ f y) :=
sorry

end max_values_of_f_smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l197_197889


namespace f_half_l197_197245

theorem f_half (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 - 2 * x) = 1 / (x ^ 2)) :
  f (1 / 2) = 16 :=
sorry

end f_half_l197_197245


namespace vertical_line_divides_triangle_equal_area_l197_197821

theorem vertical_line_divides_triangle_equal_area :
  let A : (ℝ × ℝ) := (1, 2)
  let B : (ℝ × ℝ) := (1, 1)
  let C : (ℝ × ℝ) := (10, 1)
  let area_ABC := (1 / 2 : ℝ) * (C.1 - A.1) * (A.2 - B.2)
  let a : ℝ := 5.5
  let area_left_triangle := (1 / 2 : ℝ) * (a - A.1) * (A.2 - B.2)
  let area_right_triangle := (1 / 2 : ℝ) * (C.1 - a) * (A.2 - B.2)
  area_left_triangle = area_right_triangle :=
by
  sorry

end vertical_line_divides_triangle_equal_area_l197_197821


namespace Kaylee_total_boxes_needed_l197_197158

-- Defining the conditions
def lemon_biscuits := 12
def chocolate_biscuits := 5
def oatmeal_biscuits := 4
def still_needed := 12

-- Defining the total boxes sold so far
def total_sold := lemon_biscuits + chocolate_biscuits + oatmeal_biscuits

-- Defining the total number of boxes that need to be sold in total
def total_needed := total_sold + still_needed

-- Lean statement to prove the required total number of boxes
theorem Kaylee_total_boxes_needed : total_needed = 33 :=
by
  sorry

end Kaylee_total_boxes_needed_l197_197158


namespace arithmetic_sequence_sum_l197_197337

theorem arithmetic_sequence_sum (x y z d : ℤ)
  (h₀ : d = 10 - 3)
  (h₁ : 10 = 3 + d)
  (h₂ : 17 = 10 + d)
  (h₃ : x = 17 + d)
  (h₄ : y = x + d)
  (h₅ : 31 = y + d)
  (h₆ : z = 31 + d) :
  x + y + z = 93 := by
sorry

end arithmetic_sequence_sum_l197_197337


namespace relay_race_time_l197_197023

-- Define the time it takes for each runner.
def Rhonda_time : ℕ := 24
def Sally_time : ℕ := Rhonda_time + 2
def Diane_time : ℕ := Rhonda_time - 3

-- Define the total time for the relay race.
def total_relay_time : ℕ := Rhonda_time + Sally_time + Diane_time

-- State the theorem we want to prove: the total relay time is 71 seconds.
theorem relay_race_time : total_relay_time = 71 := 
by 
  -- The following "sorry" indicates a step where the proof would be completed.
  sorry

end relay_race_time_l197_197023


namespace remainder_of_polynomial_l197_197214

theorem remainder_of_polynomial 
  (P : ℝ → ℝ) 
  (h₁ : P 15 = 16)
  (h₂ : P 10 = 4) :
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - 10) * (x - 15) * Q x + (12 / 5 * x - 20) :=
by
  sorry

end remainder_of_polynomial_l197_197214


namespace f_14_52_eq_364_l197_197034

def f : ℕ → ℕ → ℕ := sorry  -- Placeholder definition

axiom f_xx (x : ℕ) : f x x = x
axiom f_sym (x y : ℕ) : f x y = f y x
axiom f_rec (x y : ℕ) (h : x + y > 0) : (x + y) * f x y = y * f x (x + y)

theorem f_14_52_eq_364 : f 14 52 = 364 := 
by {
  sorry  -- Placeholder for the proof steps
}

end f_14_52_eq_364_l197_197034


namespace determine_angle_G_l197_197334

theorem determine_angle_G 
  (C D E F G : ℝ)
  (hC : C = 120) 
  (h_linear_pair : C + D = 180)
  (hE : E = 50) 
  (hF : F = D) 
  (h_triangle_sum : E + F + G = 180) :
  G = 70 := 
sorry

end determine_angle_G_l197_197334


namespace altitude_difference_l197_197390

theorem altitude_difference 
  (alt_A : ℤ) (alt_B : ℤ) (alt_C : ℤ)
  (hA : alt_A = -102) (hB : alt_B = -80) (hC : alt_C = -25) :
  (max (max alt_A alt_B) alt_C) - (min (min alt_A alt_B) alt_C) = 77 := 
by 
  sorry

end altitude_difference_l197_197390


namespace angle_bao_proof_l197_197697

noncomputable def angle_bao : ℝ := sorry -- angle BAO in degrees

theorem angle_bao_proof 
    (CD_is_diameter : true)
    (A_on_extension_DC_beyond_C : true)
    (E_on_semicircle : true)
    (B_is_intersection_AE_semicircle : B ≠ E)
    (AB_eq_OE : AB = OE)
    (angle_EOD_30_degrees : EOD = 30) : 
    angle_bao = 7.5 :=
sorry

end angle_bao_proof_l197_197697


namespace proposition_C_is_correct_l197_197310

theorem proposition_C_is_correct :
  ∃ a b : ℝ, (a > 2 ∧ b > 2) → (a * b > 4) :=
by
  sorry

end proposition_C_is_correct_l197_197310


namespace find_a_plus_b_l197_197871

theorem find_a_plus_b (a b : ℤ) (h1 : a^2 = 16) (h2 : b^3 = -27) (h3 : |a - b| = a - b) : a + b = 1 := by
  sorry

end find_a_plus_b_l197_197871


namespace interval_between_doses_l197_197532

noncomputable def dose_mg : ℕ := 2 * 375

noncomputable def total_mg_per_day : ℕ := 3000

noncomputable def hours_in_day : ℕ := 24

noncomputable def doses_per_day := total_mg_per_day / dose_mg

noncomputable def hours_between_doses := hours_in_day / doses_per_day

theorem interval_between_doses : hours_between_doses = 6 :=
by
  sorry

end interval_between_doses_l197_197532


namespace area_ratio_PQR_to_STU_l197_197457

-- Given Conditions
def triangle_PQR_sides (a b c : Nat) : Prop :=
  a = 9 ∧ b = 40 ∧ c = 41

def triangle_STU_sides (x y z : Nat) : Prop :=
  x = 7 ∧ y = 24 ∧ z = 25

-- Theorem Statement (math proof problem)
theorem area_ratio_PQR_to_STU :
  (∃ (a b c x y z : Nat), triangle_PQR_sides a b c ∧ triangle_STU_sides x y z) →
  9 * 40 / (7 * 24) = 15 / 7 :=
by
  intro h
  sorry

end area_ratio_PQR_to_STU_l197_197457


namespace correct_growth_equation_l197_197059

-- Define the parameters
def initial_income : ℝ := 2.36
def final_income : ℝ := 2.7
def growth_period : ℕ := 2

-- Define the growth rate x
variable (x : ℝ)

-- The theorem we want to prove
theorem correct_growth_equation : initial_income * (1 + x)^growth_period = final_income :=
sorry

end correct_growth_equation_l197_197059


namespace find_x_l197_197227

theorem find_x (n : ℕ) (h_odd : n % 2 = 1)
  (h_three_primes : ∃ (p1 p2 p3 : ℕ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ 
    11 = p1 ∧ (7 ^ n + 1) = p1 * p2 * p3) :
  (7 ^ n + 1) = 16808 :=
by
  sorry

end find_x_l197_197227


namespace plants_needed_correct_l197_197696

def total_plants_needed (ferns palms succulents total_desired : ℕ) : ℕ :=
 total_desired - (ferns + palms + succulents)

theorem plants_needed_correct : total_plants_needed 3 5 7 24 = 9 := by
  sorry

end plants_needed_correct_l197_197696


namespace range_of_expression_l197_197924

theorem range_of_expression (a b : ℝ) (h : a^2 + b^2 = 9) :
  -21 ≤ a^2 + b^2 - 6*a - 8*b ∧ a^2 + b^2 - 6*a - 8*b ≤ 39 :=
by
  sorry

end range_of_expression_l197_197924


namespace value_of_a_l197_197166

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → |a * x + 1| ≤ 3) ↔ a = 2 :=
by
  sorry

end value_of_a_l197_197166


namespace sum_of_radii_of_tangent_circles_l197_197372

theorem sum_of_radii_of_tangent_circles : 
  ∃ r1 r2 : ℝ, 
    r1 > 0 ∧
    r2 > 0 ∧
    ((r1 - 4)^2 + r1^2 = (r1 + 2)^2) ∧ 
    ((r2 - 4)^2 + r2^2 = (r2 + 2)^2) ∧
    r1 + r2 = 12 :=
by
  sorry

end sum_of_radii_of_tangent_circles_l197_197372


namespace greatest_number_of_problems_missed_l197_197659

theorem greatest_number_of_problems_missed 
    (total_problems : ℕ) (passing_percentage : ℝ) (max_missed : ℕ) :
    total_problems = 40 →
    passing_percentage = 0.85 →
    max_missed = total_problems - ⌈total_problems * passing_percentage⌉ →
    max_missed = 6 :=
by
  intros h1 h2 h3
  sorry

end greatest_number_of_problems_missed_l197_197659


namespace intersection_complement_eq_l197_197830

open Set

def U : Set Int := univ
def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 3}

theorem intersection_complement_eq :
  (U \ M) ∩ N = {3} :=
  by sorry

end intersection_complement_eq_l197_197830


namespace find_x_on_line_segment_l197_197721

theorem find_x_on_line_segment (x : ℚ) : 
    (∃ m : ℚ, m = (9 - (-1))/(1 - (-2)) ∧ (2 - 9 = m * (x - 1))) → x = -11/10 :=
by 
  sorry

end find_x_on_line_segment_l197_197721


namespace cistern_length_is_four_l197_197832

noncomputable def length_of_cistern (width depth total_area : ℝ) : ℝ :=
  let L := ((total_area - (2 * width * depth)) / (2 * (width + depth)))
  L

theorem cistern_length_is_four
  (width depth total_area : ℝ)
  (h_width : width = 2)
  (h_depth : depth = 1.25)
  (h_total_area : total_area = 23) :
  length_of_cistern width depth total_area = 4 :=
by 
  sorry

end cistern_length_is_four_l197_197832


namespace pentagon_coloring_l197_197792

theorem pentagon_coloring (convex : Prop) (unequal_sides : Prop)
  (colors : Prop) (adjacent_diff_color : Prop) :
  ∃ n : ℕ, n = 30 := by
  -- Definitions for conditions (in practical terms, these might need to be more elaborate)
  let convex := true           -- Simplified representation
  let unequal_sides := true    -- Simplified representation
  let colors := true           -- Simplified representation
  let adjacent_diff_color := true -- Simplified representation
  
  -- Proof that the number of coloring methods is 30
  existsi 30
  sorry

end pentagon_coloring_l197_197792


namespace cubic_sum_div_pqr_eq_three_l197_197512

theorem cubic_sum_div_pqr_eq_three (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p + q + r = 6) :
  (p^3 + q^3 + r^3) / (p * q * r) = 3 := 
by
  sorry

end cubic_sum_div_pqr_eq_three_l197_197512


namespace hyperbola_eqn_l197_197707

-- Definitions of given conditions
def a := 4
def b := 3
def c := 5

def hyperbola (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Hypotheses derived from conditions
axiom asymptotes : b / a = 3 / 4
axiom right_focus : a^2 + b^2 = c^2

-- Main theorem statement
theorem hyperbola_eqn : (forall x y, hyperbola x y ↔ x^2 / 16 - y^2 / 9 = 1) :=
by
  intros
  sorry

end hyperbola_eqn_l197_197707


namespace slightly_used_crayons_correct_l197_197939

def total_crayons : ℕ := 120
def new_crayons : ℕ := total_crayons / 3
def broken_crayons : ℕ := (total_crayons * 20) / 100
def slightly_used_crayons : ℕ := total_crayons - new_crayons - broken_crayons

theorem slightly_used_crayons_correct : slightly_used_crayons = 56 := sorry

end slightly_used_crayons_correct_l197_197939


namespace find_f_six_l197_197084

noncomputable def f : ℕ → ℤ := sorry

axiom f_one_eq_one : f 1 = 1
axiom f_add (x y : ℕ) : f (x + y) = f x + f y + 8 * x * y - 2
axiom f_seven_eq_163 : f 7 = 163

theorem find_f_six : f 6 = 116 := 
by {
  sorry
}

end find_f_six_l197_197084


namespace hamburgers_made_l197_197058

theorem hamburgers_made (initial_hamburgers additional_hamburgers total_hamburgers : ℝ)
    (h_initial : initial_hamburgers = 9.0)
    (h_additional : additional_hamburgers = 3.0)
    (h_total : total_hamburgers = initial_hamburgers + additional_hamburgers) :
    total_hamburgers = 12.0 :=
by
    sorry

end hamburgers_made_l197_197058


namespace alpha_in_third_quadrant_l197_197921

theorem alpha_in_third_quadrant (k : ℤ) (α : ℝ) :
  (4 * k + 1) * 180 < α ∧ α < (4 * k + 1) * 180 + 60 → 180 < α ∧ α < 240 :=
  sorry

end alpha_in_third_quadrant_l197_197921


namespace exists_integer_in_seq_l197_197789

noncomputable def x_seq (x : ℕ → ℚ) := ∀ n : ℕ, x (n + 1) = x n + 1 / ⌊x n⌋

theorem exists_integer_in_seq {x : ℕ → ℚ} (h1 : 1 < x 1) (h2 : x_seq x) : 
  ∃ n : ℕ, ∃ k : ℤ, x n = k :=
sorry

end exists_integer_in_seq_l197_197789


namespace classmates_ate_cake_l197_197008

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l197_197008


namespace range_of_p_l197_197163

def p (x : ℝ) : ℝ := (x^3 + 3)^2

theorem range_of_p :
  (∀ y, ∃ x ∈ Set.Ici (-1 : ℝ), p x = y) ↔ y ∈ Set.Ici (4 : ℝ) :=
by
  sorry

end range_of_p_l197_197163


namespace square_difference_l197_197104

variable (n : ℕ)

theorem square_difference (n : ℕ) : (n + 1)^2 - n^2 = 2 * n + 1 :=
sorry

end square_difference_l197_197104


namespace postcards_remainder_l197_197536

theorem postcards_remainder :
  let amelia := 45
  let ben := 55
  let charles := 23
  let total := amelia + ben + charles
  total % 15 = 3 :=
by
  let amelia := 45
  let ben := 55
  let charles := 23
  let total := amelia + ben + charles
  show total % 15 = 3
  sorry

end postcards_remainder_l197_197536


namespace stratified_sampling_l197_197632

-- Conditions
def total_students : ℕ := 1200
def freshmen : ℕ := 300
def sophomores : ℕ := 400
def juniors : ℕ := 500
def sample_size : ℕ := 60
def probability : ℚ := sample_size / total_students

-- Number of students to be sampled from each grade
def freshmen_sampled : ℚ := freshmen * probability
def sophomores_sampled : ℚ := sophomores * probability
def juniors_sampled : ℚ := juniors * probability

-- Theorem to prove
theorem stratified_sampling :
  freshmen_sampled = 15 ∧ sophomores_sampled = 20 ∧ juniors_sampled = 25 :=
by
  -- The actual proof would go here
  sorry

end stratified_sampling_l197_197632


namespace intersection_point_x_value_l197_197177

theorem intersection_point_x_value :
  ∃ x y : ℚ, (y = 3 * x - 22) ∧ (3 * x + y = 100) ∧ (x = 20 + 1 / 3) := by
  sorry

end intersection_point_x_value_l197_197177


namespace cube_loop_probability_l197_197010

-- Define the number of faces and alignments for a cube
def total_faces := 6
def stripe_orientations_per_face := 2

-- Define the total possible stripe combinations
def total_stripe_combinations := stripe_orientations_per_face ^ total_faces

-- Define the combinations for both vertical and horizontal loops
def vertical_and_horizontal_loop_combinations := 64

-- Define the probability space
def probability_at_least_one_each := vertical_and_horizontal_loop_combinations / total_stripe_combinations

-- The main theorem to state the probability of having at least one vertical and one horizontal loop
theorem cube_loop_probability : probability_at_least_one_each = 1 := by
  sorry

end cube_loop_probability_l197_197010


namespace rectangle_original_area_l197_197286

theorem rectangle_original_area (L L' A : ℝ) 
  (h1: A = L * 10)
  (h2: L' * 10 = (4 / 3) * A)
  (h3: 2 * L' + 2 * 10 = 60) : A = 150 :=
by 
  sorry

end rectangle_original_area_l197_197286


namespace maximize_area_CDFE_l197_197130

-- Given the side lengths of the rectangle
def AB : ℝ := 2
def AD : ℝ := 1

-- Definitions for points E and F
def AE (x : ℝ) : ℝ := x
def AF (x : ℝ) : ℝ := x

-- The formula for the area of quadrilateral CDFE
def area_CDFE (x : ℝ) : ℝ := 
  0.5 * x * (3 - 2 * x)

theorem maximize_area_CDFE : 
  ∃ x : ℝ, x = 3 / 4 ∧ area_CDFE x = 9 / 16 :=
by 
  sorry

end maximize_area_CDFE_l197_197130


namespace balcony_height_l197_197627

-- Definitions for conditions given in the problem

def final_position := 0 -- y, since the ball hits the ground
def initial_velocity := 5 -- v₀ in m/s
def time_elapsed := 3 -- t in seconds
def gravity := 10 -- g in m/s²

theorem balcony_height : 
  ∃ h₀ : ℝ, final_position = h₀ + initial_velocity * time_elapsed - (1/2) * gravity * time_elapsed^2 ∧ h₀ = 30 := 
by 
  sorry

end balcony_height_l197_197627


namespace alcohol_mixture_l197_197424

theorem alcohol_mixture:
  ∃ (x y z: ℝ), 
    0.10 * x + 0.30 * y + 0.50 * z = 157.5 ∧
    x + y + z = 450 ∧
    x = y ∧
    x = 112.5 ∧
    y = 112.5 ∧
    z = 225 :=
sorry

end alcohol_mixture_l197_197424


namespace find_fraction_l197_197581

theorem find_fraction (a b : ℝ) (h₁ : a ≠ b) (h₂ : a / b + (a + 6 * b) / (b + 6 * a) = 2) :
  a / b = 1 / 2 :=
sorry

end find_fraction_l197_197581


namespace total_strings_correct_l197_197872

-- Definitions based on conditions
def num_ukuleles : ℕ := 2
def num_guitars : ℕ := 4
def num_violins : ℕ := 2
def strings_per_ukulele : ℕ := 4
def strings_per_guitar : ℕ := 6
def strings_per_violin : ℕ := 4

-- Total number of strings
def total_strings : ℕ := num_ukuleles * strings_per_ukulele +
                         num_guitars * strings_per_guitar +
                         num_violins * strings_per_violin

-- The proof statement
theorem total_strings_correct : total_strings = 40 :=
by
  -- Proof omitted.
  sorry

end total_strings_correct_l197_197872


namespace bus_speed_excluding_stoppages_l197_197584

variable (v : ℝ) -- Speed of the bus excluding stoppages

-- Conditions
def bus_stops_per_hour := 45 / 60 -- 45 minutes converted to hours
def effective_driving_time := 1 - bus_stops_per_hour -- Effective time driving in an hour

-- Given Condition
def speed_including_stoppages := 12 -- Speed including stoppages in km/hr

theorem bus_speed_excluding_stoppages 
  (h : effective_driving_time * v = speed_including_stoppages) : 
  v = 48 :=
sorry

end bus_speed_excluding_stoppages_l197_197584


namespace intersection_and_complement_find_m_l197_197145

-- Define the sets A, B, C
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def C (m : ℝ) : Set ℝ := {x | m+1 ≤ x ∧ x ≤ 3*m}

-- State the first proof problem: intersection A ∩ B and complement of B
theorem intersection_and_complement (x : ℝ) : 
  (x ∈ (A ∩ B) ↔ (2 ≤ x ∧ x ≤ 3)) ∧ 
  (x ∈ (compl B) ↔ (x < 1 ∨ x > 4)) :=
by 
  sorry

-- State the second proof problem: find m satisfying A ∪ C(m) = A
theorem find_m (m : ℝ) (x : ℝ) : 
  (∀ x, (x ∈ A ∪ C m) ↔ (x ∈ A)) ↔ (m = 1) :=
by 
  sorry

end intersection_and_complement_find_m_l197_197145


namespace max_range_of_temps_l197_197062

noncomputable def max_temp_range (T1 T2 T3 T4 T5 : ℝ) : ℝ := 
  max (max (max (max T1 T2) T3) T4) T5 - min (min (min (min T1 T2) T3) T4) T5

theorem max_range_of_temps :
  ∀ (T1 T2 T3 T4 T5 : ℝ), 
  (T1 + T2 + T3 + T4 + T5) / 5 = 60 →
  T1 = 40 →
  (max_temp_range T1 T2 T3 T4 T5) = 100 :=
by
  intros T1 T2 T3 T4 T5 Havg Hlowest
  sorry

end max_range_of_temps_l197_197062


namespace initial_bottles_l197_197290

-- Define the conditions
def drank_bottles : ℕ := 144
def left_bottles : ℕ := 157

-- Define the total_bottles function
def total_bottles : ℕ := drank_bottles + left_bottles

-- State the theorem to be proven
theorem initial_bottles : total_bottles = 301 :=
by
  sorry

end initial_bottles_l197_197290


namespace number_of_added_groups_l197_197974

-- Define the total number of students in the class
def total_students : ℕ := 47

-- Define the number of students per table and the number of tables
def students_per_table : ℕ := 3
def number_of_tables : ℕ := 6

-- Define the number of girls in the bathroom and the multiplier for students in the canteen
def girls_in_bathroom : ℕ := 3
def canteen_multiplier : ℕ := 3

-- Define the number of foreign exchange students from each country
def foreign_exchange_germany : ℕ := 3
def foreign_exchange_france : ℕ := 3
def foreign_exchange_norway : ℕ := 3

-- Define the number of students per recently added group
def students_per_group : ℕ := 4

-- Calculate the number of students currently in the classroom
def students_in_classroom := number_of_tables * students_per_table

-- Calculate the number of students temporarily absent
def students_in_canteen := girls_in_bathroom * canteen_multiplier
def temporarily_absent := girls_in_bathroom + students_in_canteen

-- Calculate the number of foreign exchange students missing
def foreign_exchange_missing := foreign_exchange_germany + foreign_exchange_france + foreign_exchange_norway

-- Calculate the total number of students accounted for
def student_accounted_for := students_in_classroom + temporarily_absent + foreign_exchange_missing

-- The proof statement (main goal)
theorem number_of_added_groups : (total_students - student_accounted_for) / students_per_group = 2 :=
by
  sorry

end number_of_added_groups_l197_197974


namespace maria_dozen_flowers_l197_197035

theorem maria_dozen_flowers (x : ℕ) (h : 12 * x + 2 * x = 42) : x = 3 :=
by
  sorry

end maria_dozen_flowers_l197_197035


namespace contrapositive_false_1_negation_false_1_l197_197302

theorem contrapositive_false_1 (m : ℝ) : ¬ (¬ (∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

theorem negation_false_1 (m : ℝ) : ¬ ((m > 0) → ¬ (∃ x : ℝ, x^2 + x - m = 0)) :=
sorry

end contrapositive_false_1_negation_false_1_l197_197302


namespace angle_RPS_is_1_degree_l197_197718

-- Definitions of the given angles
def angle_QRS : ℝ := 150
def angle_PQS : ℝ := 60
def angle_PSQ : ℝ := 49
def angle_QPR : ℝ := 70

-- Definition for the calculated angle QPS
def angle_QPS : ℝ := 180 - angle_PQS - angle_PSQ

-- Definition for the target angle RPS
def angle_RPS : ℝ := angle_QPS - angle_QPR

-- The theorem we aim to prove
theorem angle_RPS_is_1_degree : angle_RPS = 1 := by
  sorry

end angle_RPS_is_1_degree_l197_197718


namespace volume_of_prism_l197_197783

-- Define the conditions
variables {a b c : ℝ}
-- Areas of the faces
def ab := 50
def ac := 72
def bc := 45

-- Theorem stating the volume of the prism
theorem volume_of_prism : a * b * c = 180 * Real.sqrt 5 :=
by
  sorry

end volume_of_prism_l197_197783


namespace sequence_sum_a1_a3_l197_197024

theorem sequence_sum_a1_a3 (S : ℕ → ℕ) (a : ℕ → ℤ) 
  (h1 : ∀ n, n ≥ 2 → S n + S (n - 1) = 2 * n - 1) 
  (h2 : S 2 = 3) : 
  a 1 + a 3 = -1 := by
  sorry

end sequence_sum_a1_a3_l197_197024


namespace parallelogram_area_l197_197745

noncomputable def angle_ABC : ℝ := 30
noncomputable def AX : ℝ := 20
noncomputable def CY : ℝ := 22

theorem parallelogram_area (angle_ABC_eq : angle_ABC = 30)
    (AX_eq : AX = 20)
    (CY_eq : CY = 22)
    : ∃ (BC : ℝ), (BC * AX = 880) := sorry

end parallelogram_area_l197_197745


namespace sum_of_valid_two_digit_numbers_l197_197222

theorem sum_of_valid_two_digit_numbers
  (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (a - b) ∣ (10 * a + b))
  (h4 : (a * b) ∣ (10 * a + b)) :
  (10 * a + b = 21) → (21 = 21) :=
sorry

end sum_of_valid_two_digit_numbers_l197_197222


namespace number_of_subsets_B_l197_197249

def A : Set ℕ := {1, 3}
def C : Set ℕ := {1, 3, 5}

theorem number_of_subsets_B : ∃ (n : ℕ), (∀ B : Set ℕ, A ∪ B = C → n = 4) :=
sorry

end number_of_subsets_B_l197_197249


namespace dinner_cost_l197_197953

theorem dinner_cost (tax_rate tip_rate total_cost : ℝ) (h_tax : tax_rate = 0.12) (h_tip : tip_rate = 0.20) (h_total : total_cost = 30.60) :
  let meal_cost := total_cost / (1 + tax_rate + tip_rate)
  meal_cost = 23.18 :=
by
  sorry

end dinner_cost_l197_197953


namespace remainder_of_1234567_div_123_l197_197735

theorem remainder_of_1234567_div_123 : 1234567 % 123 = 129 :=
by
  sorry

end remainder_of_1234567_div_123_l197_197735


namespace cube_diagonal_length_l197_197419

theorem cube_diagonal_length (s : ℝ) 
    (h₁ : 6 * s^2 = 54) 
    (h₂ : 12 * s = 36) :
    ∃ d : ℝ, d = 3 * Real.sqrt 3 ∧ d = Real.sqrt (3 * s^2) :=
by
  sorry

end cube_diagonal_length_l197_197419


namespace Julia_played_with_kids_on_Monday_l197_197604

theorem Julia_played_with_kids_on_Monday (kids_tuesday : ℕ) (more_kids_monday : ℕ) :
  kids_tuesday = 14 → more_kids_monday = 8 → (kids_tuesday + more_kids_monday = 22) :=
by
  sorry

end Julia_played_with_kids_on_Monday_l197_197604


namespace units_digit_M_M12_l197_197370

def modifiedLucas (n : ℕ) : ℕ :=
  match n with
  | 0     => 3
  | 1     => 2
  | n + 2 => modifiedLucas (n + 1) + modifiedLucas n

theorem units_digit_M_M12 (n : ℕ) (H : modifiedLucas 12 = 555) : 
  (modifiedLucas (modifiedLucas 12) % 10) = 1 := by
  sorry

end units_digit_M_M12_l197_197370


namespace sum_interior_angles_polygon_l197_197251

theorem sum_interior_angles_polygon (n : ℕ) (h : 180 * (n - 2) = 1440) :
  180 * ((n + 3) - 2) = 1980 := by
  sorry

end sum_interior_angles_polygon_l197_197251


namespace intersection_area_l197_197904

-- Define the square vertices
def vertex1 : (ℝ × ℝ) := (2, 8)
def vertex2 : (ℝ × ℝ) := (13, 8)
def vertex3 : (ℝ × ℝ) := (13, -3)
def vertex4 : (ℝ × ℝ) := (2, -3)  -- Derived from the conditions

-- Define the circle with center and radius
def circle_center : (ℝ × ℝ) := (2, -3)
def circle_radius : ℝ := 4

-- Define the square side length
def square_side_length : ℝ := 11  -- From vertex (2, 8) to vertex (2, -3)

-- Prove the intersection area
theorem intersection_area :
  let area := (1 / 4) * Real.pi * (circle_radius^2)
  area = 4 * Real.pi :=
by
  sorry

end intersection_area_l197_197904


namespace find_salary_l197_197836

-- Define the conditions
variables (S : ℝ) -- S is the man's monthly salary

def saves_25_percent (S : ℝ) : ℝ := 0.25 * S
def expenses (S : ℝ) : ℝ := 0.75 * S
def increased_expenses (S : ℝ) : ℝ := 0.75 * S + 0.10 * (0.75 * S)
def monthly_savings_after_increase (S : ℝ) : ℝ := S - increased_expenses S

-- Define the problem statement
theorem find_salary
  (h1 : saves_25_percent S = 0.25 * S)
  (h2 : increased_expenses S = 0.825 * S)
  (h3 : monthly_savings_after_increase S = 175) :
  S = 1000 :=
sorry

end find_salary_l197_197836


namespace disease_given_positive_l197_197625

-- Definitions and conditions extracted from the problem
def Pr_D : ℚ := 1 / 200
def Pr_Dc : ℚ := 1 - Pr_D
def Pr_T_D : ℚ := 1
def Pr_T_Dc : ℚ := 0.05

-- Derived probabilites from given conditions
def Pr_T : ℚ := Pr_T_D * Pr_D + Pr_T_Dc * Pr_Dc

-- Statement for the probability using Bayes' theorem
theorem disease_given_positive :
  (Pr_T_D * Pr_D) / Pr_T = 20 / 219 :=
sorry

end disease_given_positive_l197_197625


namespace solution_set_of_system_of_inequalities_l197_197186

theorem solution_set_of_system_of_inequalities :
  {x : ℝ | |x| - 1 < 0 ∧ x^2 - 3 * x < 0} = {x : ℝ | 0 < x ∧ x < 1} :=
sorry

end solution_set_of_system_of_inequalities_l197_197186


namespace digit_product_inequality_l197_197030

noncomputable def digit_count_in_n (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).count d

theorem digit_product_inequality (n : ℕ) (a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ)
  (h1 : a1 = digit_count_in_n n 1)
  (h2 : a2 = digit_count_in_n n 2)
  (h3 : a3 = digit_count_in_n n 3)
  (h4 : a4 = digit_count_in_n n 4)
  (h5 : a5 = digit_count_in_n n 5)
  (h6 : a6 = digit_count_in_n n 6)
  (h7 : a7 = digit_count_in_n n 7)
  (h8 : a8 = digit_count_in_n n 8)
  (h9 : a9 = digit_count_in_n n 9)
  : 2^a1 * 3^a2 * 4^a3 * 5^a4 * 6^a5 * 7^a6 * 8^a7 * 9^a8 * 10^a9 ≤ n + 1 :=
  sorry

end digit_product_inequality_l197_197030


namespace three_pow_2010_mod_eight_l197_197594

theorem three_pow_2010_mod_eight : (3^2010) % 8 = 1 :=
  sorry

end three_pow_2010_mod_eight_l197_197594


namespace percentage_increase_second_year_is_20_l197_197535

noncomputable def find_percentage_increase_second_year : ℕ :=
  let P₀ := 1000
  let P₁ := P₀ + (10 * P₀) / 100
  let Pf := 1320
  let P := (Pf - P₁) * 100 / P₁
  P

theorem percentage_increase_second_year_is_20 :
  find_percentage_increase_second_year = 20 :=
by
  sorry

end percentage_increase_second_year_is_20_l197_197535


namespace last_four_digits_of_5_pow_2011_l197_197150

theorem last_four_digits_of_5_pow_2011 : (5^2011 % 10000) = 8125 := by
  sorry

end last_four_digits_of_5_pow_2011_l197_197150


namespace probability_of_two_germinates_is_48_over_125_l197_197349

noncomputable def probability_of_exactly_two_germinates : ℚ :=
  let p := 4/5
  let n := 3
  let k := 2
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_of_two_germinates_is_48_over_125 :
  probability_of_exactly_two_germinates = 48/125 := by
    sorry

end probability_of_two_germinates_is_48_over_125_l197_197349


namespace problem1_problem2_l197_197728

-- Definitions
def total_questions := 5
def multiple_choice := 3
def true_false := 2
def total_outcomes := total_questions * (total_questions - 1)

-- (1) Probability of A drawing a true/false question and B drawing a multiple-choice question
def favorable_outcomes_1 := true_false * multiple_choice

-- (2) Probability of at least one of A or B drawing a multiple-choice question
def unfavorable_outcomes_2 := true_false * (true_false - 1)

-- Statements to be proved
theorem problem1 : favorable_outcomes_1 / total_outcomes = 3 / 10 := by sorry

theorem problem2 : 1 - (unfavorable_outcomes_2 / total_outcomes) = 9 / 10 := by sorry

end problem1_problem2_l197_197728


namespace problem_statement_l197_197051

theorem problem_statement (x y : ℝ) (h₁ : x + y = 5) (h₂ : x * y = 3) : 
  x + (x^2 / y) + (y^2 / x) + y = 95 / 3 := 
sorry

end problem_statement_l197_197051


namespace initial_interest_rate_l197_197281

theorem initial_interest_rate
    (P R : ℝ) 
    (h1 : P * R = 10120) 
    (h2 : P * (R + 6) = 12144) : 
    R = 30 :=
sorry

end initial_interest_rate_l197_197281


namespace total_cost_is_100_l197_197958

def shirts : ℕ := 10
def pants : ℕ := shirts / 2
def cost_shirt : ℕ := 6
def cost_pant : ℕ := 8

theorem total_cost_is_100 :
  shirts * cost_shirt + pants * cost_pant = 100 := by
  sorry

end total_cost_is_100_l197_197958


namespace triangle_third_side_l197_197635

theorem triangle_third_side (x : ℝ) (h1 : x > 2) (h2 : x < 6) : x = 5 :=
sorry

end triangle_third_side_l197_197635


namespace range_of_m_l197_197980

open Real

def f (x m: ℝ) : ℝ := x^2 - 2 * x + m^2 + 3 * m - 3

def p (m: ℝ) : Prop := ∃ x, f x m < 0

def q (m: ℝ) : Prop := (5 * m - 1 > 0) ∧ (m - 2 > 0)

theorem range_of_m (m : ℝ) : ¬ (p m ∨ q m) ∧ ¬ (p m ∧ q m) → (m ≤ -4 ∨ m ≥ 2) :=
by
  sorry

end range_of_m_l197_197980


namespace total_penalty_kicks_l197_197868

theorem total_penalty_kicks (total_players : ℕ) (goalies : ℕ) (hoop_challenges : ℕ)
  (h_total : total_players = 25) (h_goalies : goalies = 5) (h_hoop_challenges : hoop_challenges = 10) :
  (goalies * (total_players - 1)) = 120 :=
by
  sorry

end total_penalty_kicks_l197_197868


namespace angle_terminal_side_equiv_l197_197580

-- Define the function to check angle equivalence
def angle_equiv (θ₁ θ₂ : ℝ) : Prop := ∃ k : ℤ, θ₁ = θ₂ + k * 360

-- Theorem statement
theorem angle_terminal_side_equiv : angle_equiv 330 (-30) :=
  sorry

end angle_terminal_side_equiv_l197_197580


namespace possible_rectangle_configurations_l197_197607

-- Define the conditions as variables
variables (m n : ℕ)
-- Define the number of segments
def segments (m n : ℕ) : ℕ := 2 * m * n + m + n

theorem possible_rectangle_configurations : 
  (segments m n = 1997) → (m = 2 ∧ n = 399) ∨ (m = 8 ∧ n = 117) ∨ (m = 23 ∧ n = 42) :=
by
  sorry

end possible_rectangle_configurations_l197_197607


namespace find_difference_square_l197_197948

theorem find_difference_square (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 6) :
  (x - y)^2 = 25 :=
by
  sorry

end find_difference_square_l197_197948


namespace domain_of_function_l197_197328

noncomputable def domain_of_f : Set ℝ :=
  {x | x > -1/2 ∧ x ≠ 1}

theorem domain_of_function :
  (∀ x : ℝ, (2 * x + 1 ≥ 0) ∧ (2 * x^2 - x - 1 ≠ 0) ↔ (x > -1/2 ∧ x ≠ 1)) := by
  sorry

end domain_of_function_l197_197328


namespace smallest_sum_l197_197943

theorem smallest_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_neq : x ≠ y)
  (h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 18) : x + y = 75 :=
by
  sorry

end smallest_sum_l197_197943


namespace pentagon_area_greater_than_square_third_l197_197726

theorem pentagon_area_greater_than_square_third (a b : ℝ) :
  a^2 + (a * b) / 4 + (Real.sqrt 3 / 4) * b^2 > ((a + b)^2) / 3 :=
by
  sorry

end pentagon_area_greater_than_square_third_l197_197726


namespace find_years_invested_l197_197838

-- Defining the conditions and theorem
variables (P : ℕ) (r1 r2 D : ℝ) (n : ℝ)

-- Given conditions
def principal := (P : ℝ) = 7000
def rate_1 := r1 = 0.15
def rate_2 := r2 = 0.12
def interest_diff := D = 420

-- Theorem to be proven
theorem find_years_invested (h1 : principal P) (h2 : rate_1 r1) (h3 : rate_2 r2) (h4 : interest_diff D) :
  7000 * 0.15 * n - 7000 * 0.12 * n = 420 → n = 2 :=
by
  sorry

end find_years_invested_l197_197838


namespace length_of_d_in_proportion_l197_197132

variable (a b c d : ℝ)

theorem length_of_d_in_proportion
  (h1 : a = 3) 
  (h2 : b = 2)
  (h3 : c = 6)
  (h_prop : a / b = c / d) : 
  d = 4 :=
by
  sorry

end length_of_d_in_proportion_l197_197132


namespace red_marbles_count_l197_197380

noncomputable def total_marbles (R : ℕ) : ℕ := R + 16

noncomputable def P_blue (R : ℕ) : ℚ := 10 / (total_marbles R)

noncomputable def P_neither_blue (R : ℕ) : ℚ := (1 - P_blue R) * (1 - P_blue R)

noncomputable def P_either_blue (R : ℕ) : ℚ := 1 - P_neither_blue R

theorem red_marbles_count
  (R : ℕ) 
  (h1 : P_either_blue R = 0.75) :
  R = 4 :=
by
  sorry

end red_marbles_count_l197_197380


namespace johns_shirt_percentage_increase_l197_197494

variable (P S : ℕ)

theorem johns_shirt_percentage_increase :
  P = 50 →
  S + P = 130 →
  ((S - P) * 100 / P) = 60 := by
  sorry

end johns_shirt_percentage_increase_l197_197494


namespace plane_divides_space_into_two_parts_l197_197749

def divides_space : Prop :=
  ∀ (P : ℝ → ℝ → ℝ → Prop), (∀ x y z, P x y z → P x y z) →
  (∃ region1 region2 : ℝ → ℝ → ℝ → Prop,
    (∀ x y z, P x y z → (region1 x y z ∨ region2 x y z)) ∧
    (∀ x y z, region1 x y z → ¬region2 x y z) ∧
    (∃ x1 y1 z1 x2 y2 z2, region1 x1 y1 z1 ∧ region2 x2 y2 z2))

theorem plane_divides_space_into_two_parts (P : ℝ → ℝ → ℝ → Prop) (hP : ∀ x y z, P x y z → P x y z) : 
  divides_space :=
  sorry

end plane_divides_space_into_two_parts_l197_197749


namespace maria_min_score_fourth_quarter_l197_197254

theorem maria_min_score_fourth_quarter (x : ℝ) :
  (82 + 77 + 78 + x) / 4 ≥ 85 ↔ x ≥ 103 :=
by
  sorry

end maria_min_score_fourth_quarter_l197_197254


namespace triangle_area_is_24_l197_197244

-- Define the vertices
def vertex1 : ℝ × ℝ := (3, 2)
def vertex2 : ℝ × ℝ := (3, -4)
def vertex3 : ℝ × ℝ := (11, -4)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

-- Prove the area of the triangle with the given vertices is 24 square units
theorem triangle_area_is_24 : triangle_area vertex1 vertex2 vertex3 = 24 := by
  sorry

end triangle_area_is_24_l197_197244


namespace male_percentage_l197_197233

theorem male_percentage (total_employees : ℕ)
  (males_below_50 : ℕ)
  (percentage_males_at_least_50 : ℝ)
  (male_percentage : ℝ) :
  total_employees = 2200 →
  males_below_50 = 616 →
  percentage_males_at_least_50 = 0.3 → 
  male_percentage = 40 :=
by
  sorry

end male_percentage_l197_197233


namespace arithmetic_sequence_example_l197_197736

theorem arithmetic_sequence_example 
    (a : ℕ → ℤ) 
    (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) 
    (h2 : a 1 + a 4 + a 7 = 45) 
    (h3 : a 2 + a 5 + a 8 = 39) :
    a 3 + a 6 + a 9 = 33 :=
sorry

end arithmetic_sequence_example_l197_197736


namespace min_value_512_l197_197913

noncomputable def min_value (a b c d e f g h : ℝ) : ℝ :=
  (2 * a * e)^2 + (2 * b * f)^2 + (2 * c * g)^2 + (2 * d * h)^2

theorem min_value_512 
  (a b c d e f g h : ℝ)
  (H1 : a * b * c * d = 8)
  (H2 : e * f * g * h = 16) : 
  ∃ (min_val : ℝ), min_val = 512 ∧ min_value a b c d e f g h = min_val :=
sorry

end min_value_512_l197_197913


namespace equivalence_of_statements_l197_197540

variable (P Q : Prop)

theorem equivalence_of_statements (h : P → Q) :
  (P → Q) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) := by
  sorry

end equivalence_of_statements_l197_197540


namespace mono_intervals_range_of_a_l197_197815

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.exp (x - 1)

theorem mono_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x, f x a > 0) ∧ 
  (a > 0 → (∀ x, x < 1 - Real.log a → f x a > 0) ∧ (∀ x, x > 1 - Real.log a → f x a < 0)) :=
sorry

theorem range_of_a (h : ∀ x, f x a ≤ 0) : a ≥ 1 :=
sorry

end mono_intervals_range_of_a_l197_197815


namespace sign_of_b_l197_197298

variable (a b : ℝ)

theorem sign_of_b (h1 : (a + b > 0 ∨ a - b > 0) ∧ (a + b < 0 ∨ a - b < 0)) 
                  (h2 : (ab > 0 ∨ a / b > 0) ∧ (ab < 0 ∨ a / b < 0))
                  (h3 : (ab > 0 → a > 0 ∧ b > 0) ∨ (ab < 0 → (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0))) :
  b < 0 :=
sorry

end sign_of_b_l197_197298


namespace soda_costs_94_cents_l197_197976

theorem soda_costs_94_cents (b s: ℤ) (h1 : 4 * b + 3 * s = 500) (h2 : 3 * b + 4 * s = 540) : s = 94 := 
by
  sorry

end soda_costs_94_cents_l197_197976


namespace n_digit_numbers_modulo_3_l197_197518

def a (i : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then if i = 0 then 1 else 0 else 2 * a i (n - 1) + a ((i + 1) % 3) (n - 1) + a ((i + 2) % 3) (n - 1)

theorem n_digit_numbers_modulo_3 (n : ℕ) (h : 0 < n) : 
  (a 0 n) = (4^n + 2) / 3 :=
sorry

end n_digit_numbers_modulo_3_l197_197518


namespace main_theorem_l197_197743

open Nat

-- Define the conditions
def conditions (p q n : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Odd p ∧ Odd q ∧ n > 1 ∧
  (q^(n+2) % p^n = 3^(n+2) % p^n) ∧ (p^(n+2) % q^n = 3^(n+2) % q^n)

-- Define the conclusion
def conclusion (p q n : ℕ) : Prop :=
  (p = 3 ∧ q = 3)

-- Define the main problem
theorem main_theorem : ∀ p q n : ℕ, conditions p q n → conclusion p q n :=
  by
    intros p q n h
    sorry

end main_theorem_l197_197743


namespace set_intersection_l197_197453

def M := {x : ℝ | x^2 > 4}
def N := {x : ℝ | 1 < x ∧ x ≤ 3}
def complement_M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def intersection := N ∩ complement_M

theorem set_intersection : intersection = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

end set_intersection_l197_197453


namespace sandbox_length_l197_197845

theorem sandbox_length (width : ℕ) (area : ℕ) (h_width : width = 146) (h_area : area = 45552) : ∃ length : ℕ, length = 312 :=
by {
  sorry
}

end sandbox_length_l197_197845


namespace subtraction_of_twos_from_ones_l197_197428

theorem subtraction_of_twos_from_ones (n : ℕ) : 
  let ones := (10^n - 1) * 10^n + (10^n - 1)
  let twos := 2 * (10^n - 1)
  ones - twos = (10^n - 1) * (10^n - 1) :=
by
  sorry

end subtraction_of_twos_from_ones_l197_197428


namespace trapezoid_area_is_correct_l197_197650

noncomputable def isosceles_trapezoid_area : ℝ :=
  let a : ℝ := 12
  let b : ℝ := 24 - 12 * Real.sqrt 2
  let h : ℝ := 6 * Real.sqrt 2
  (24 + b) / 2 * h

theorem trapezoid_area_is_correct :
  let a := 12
  let b := 24 - 12 * Real.sqrt 2
  let h := 6 * Real.sqrt 2
  (24 + b) / 2 * h = 144 * Real.sqrt 2 - 72 :=
by
  sorry

end trapezoid_area_is_correct_l197_197650


namespace slope_of_line_m_equals_neg_2_l197_197326

theorem slope_of_line_m_equals_neg_2
  (m : ℝ)
  (h : (3 * m - 6) / (1 + m) = 12) :
  m = -2 :=
sorry

end slope_of_line_m_equals_neg_2_l197_197326


namespace min_value_reciprocal_sum_l197_197376

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (1 / a + 1 / b) ≥ 4 :=
sorry

end min_value_reciprocal_sum_l197_197376


namespace black_haired_girls_count_l197_197420

theorem black_haired_girls_count (initial_total_girls : ℕ) (initial_blonde_girls : ℕ) (added_blonde_girls : ℕ) (final_blonde_girls total_girls : ℕ) 
    (h1 : initial_total_girls = 80) 
    (h2 : initial_blonde_girls = 30) 
    (h3 : added_blonde_girls = 10) 
    (h4 : final_blonde_girls = initial_blonde_girls + added_blonde_girls) 
    (h5 : total_girls = initial_total_girls) : 
    total_girls - final_blonde_girls = 40 :=
by
  sorry

end black_haired_girls_count_l197_197420


namespace minimum_apples_collected_l197_197857

-- Anya, Vanya, Dania, Sanya, and Tanya each collected an integer percentage of the total number of apples,
-- with all these percentages distinct and greater than zero.
-- Prove that the minimum total number of apples is 20.

theorem minimum_apples_collected :
  ∃ (n : ℕ), (∀ (a v d s t : ℕ), 
    1 ≤ a ∧ 1 ≤ v ∧ 1 ≤ d ∧ 1 ≤ s ∧ 1 ≤ t ∧
    a ≠ v ∧ a ≠ d ∧ a ≠ s ∧ a ≠ t ∧ 
    v ≠ d ∧ v ≠ s ∧ v ≠ t ∧ 
    d ≠ s ∧ d ≠ t ∧ 
    s ≠ t ∧
    a + v + d + s + t = 100) →
  n ≥ 20 :=
by 
  sorry

end minimum_apples_collected_l197_197857


namespace num_arrangements_with_ab_together_l197_197616

theorem num_arrangements_with_ab_together (products : Fin 5 → Type) :
  (∃ A B : Fin 5 → Type, A ≠ B) →
  ∃ (n : ℕ), n = 48 :=
by
  sorry

end num_arrangements_with_ab_together_l197_197616


namespace quadratic_roots_range_l197_197331

theorem quadratic_roots_range (m : ℝ) : 
  (2 * x^2 - (m + 1) * x + m = 0) → 
  (m^2 - 6 * m + 1 > 0) → 
  (0 < m) → 
  (0 < m ∧ m < 3 - 2 * Real.sqrt 2 ∨ m > 3 + 2 * Real.sqrt 2) :=
by
  sorry

end quadratic_roots_range_l197_197331


namespace arithmetic_seq_a10_l197_197623

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

theorem arithmetic_seq_a10 (h_arith : arithmetic_sequence a) (h2 : a 3 = 5) (h5 : a 6 = 11) : a 10 = 19 := by
  sorry

end arithmetic_seq_a10_l197_197623


namespace alpha_value_l197_197661

theorem alpha_value (m : ℝ) (α : ℝ) (h : m * 8 ^ α = 1 / 4) : α = -2 / 3 :=
by
  sorry

end alpha_value_l197_197661


namespace price_white_stamp_l197_197007

variable (price_per_white_stamp : ℝ)

theorem price_white_stamp (simon_red_stamps : ℕ)
                          (peter_white_stamps : ℕ)
                          (price_per_red_stamp : ℝ)
                          (money_difference : ℝ)
                          (h1 : simon_red_stamps = 30)
                          (h2 : peter_white_stamps = 80)
                          (h3 : price_per_red_stamp = 0.50)
                          (h4 : money_difference = 1) :
    money_difference = peter_white_stamps * price_per_white_stamp - simon_red_stamps * price_per_red_stamp →
    price_per_white_stamp = 1 / 5 :=
by
  intros
  sorry

end price_white_stamp_l197_197007


namespace area_of_right_triangle_l197_197746

variable (a b : ℝ)

theorem area_of_right_triangle (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ (S : ℝ), S = a * b :=
sorry

end area_of_right_triangle_l197_197746


namespace range_of_m_l197_197828

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (-2 < x ∧ x ≤ 2) → x ≤ m) → m ≥ 2 :=
by
  intro h
  -- insert necessary proof steps here
  sorry

end range_of_m_l197_197828


namespace least_number_to_add_l197_197834

theorem least_number_to_add (n divisor : ℕ) (h₁ : n = 27306) (h₂ : divisor = 151) : 
  ∃ k : ℕ, k = 25 ∧ (n + k) % divisor = 0 := 
by
  sorry

end least_number_to_add_l197_197834


namespace rope_length_91_4_l197_197951

noncomputable def total_rope_length (n : ℕ) (d : ℕ) (pi_val : Real) : Real :=
  let linear_segments := 6 * d
  let arc_length := (d * pi_val / 3) * 6
  let total_length_per_tie := linear_segments + arc_length
  total_length_per_tie * 2

theorem rope_length_91_4 :
  total_rope_length 7 5 3.14 = 91.4 :=
by
  sorry

end rope_length_91_4_l197_197951


namespace inequality_int_part_l197_197395

theorem inequality_int_part (a : ℝ) (n : ℕ) (h1 : 1 ≤ a) (h2 : (0 : ℝ) ≤ n ∧ (n : ℝ) ≤ a) : 
  ⌊a⌋ > (n / (n + 1 : ℝ)) * a := 
by 
  sorry

end inequality_int_part_l197_197395


namespace trapezoid_shaded_fraction_l197_197515

theorem trapezoid_shaded_fraction (total_strips : ℕ) (shaded_strips : ℕ)
  (h_total : total_strips = 7) (h_shaded : shaded_strips = 4) :
  (shaded_strips : ℚ) / (total_strips : ℚ) = 4 / 7 := 
by
  sorry

end trapezoid_shaded_fraction_l197_197515


namespace jessica_exam_time_l197_197935

theorem jessica_exam_time (total_questions : ℕ) (answered_questions : ℕ) (used_minutes : ℕ)
    (total_time : ℕ) (remaining_time : ℕ) (rate : ℚ) :
    total_questions = 80 ∧ answered_questions = 16 ∧ used_minutes = 12 ∧ total_time = 60 ∧ rate = (answered_questions : ℚ) / used_minutes →
    remaining_time = total_time - used_minutes →
    remaining_time = 48 :=
by
  -- Proof will be filled in here
  sorry

end jessica_exam_time_l197_197935


namespace sum_of_elements_in_M_l197_197004

theorem sum_of_elements_in_M (m : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + m = 0) :
  (∀ x : ℝ, x ∈ {x | x^2 - 2 * x + m = 0} → x = 1) ∧ m = 1 ∨
  (∃ x1 x2 : ℝ, x1 ∈ {x | x^2 - 2 * x + m = 0} ∧ x2 ∈ {x | x^2 - 2 * x + m = 0} ∧ x1 ≠ x2 ∧
   x1 + x2 = 2 ∧ m < 1) :=
sorry

end sum_of_elements_in_M_l197_197004


namespace avg_age_family_now_l197_197257

namespace average_age_family

-- Define initial conditions
def avg_age_husband_wife_marriage := 23
def years_since_marriage := 5
def age_child := 1
def number_of_family_members := 3

-- Prove that the average age of the family now is 19 years
theorem avg_age_family_now :
  (2 * avg_age_husband_wife_marriage + 2 * years_since_marriage + age_child) / number_of_family_members = 19 := by
  sorry

end average_age_family

end avg_age_family_now_l197_197257


namespace original_stone_count_145_l197_197100

theorem original_stone_count_145 : 
  ∃ (n : ℕ), (n ≡ 1 [MOD 18]) ∧ (n = 145) :=
by
  sorry

end original_stone_count_145_l197_197100


namespace biscuit_dimensions_l197_197539

theorem biscuit_dimensions (sheet_length : ℝ) (sheet_width : ℝ) (num_biscuits : ℕ) 
  (h₁ : sheet_length = 12) (h₂ : sheet_width = 12) (h₃ : num_biscuits = 16) :
  ∃ biscuit_length : ℝ, biscuit_length = 3 :=
by
  sorry

end biscuit_dimensions_l197_197539


namespace calories_remaining_for_dinner_l197_197706

theorem calories_remaining_for_dinner :
  let daily_limit := 2200
  let breakfast := 353
  let lunch := 885
  let snack := 130
  let total_consumed := breakfast + lunch + snack
  let remaining_for_dinner := daily_limit - total_consumed
  remaining_for_dinner = 832 := by
  sorry

end calories_remaining_for_dinner_l197_197706


namespace probability_a_2b_3c_gt_5_l197_197715

def isInUnitCube (a b c : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1

theorem probability_a_2b_3c_gt_5 (a b c : ℝ) :
  isInUnitCube a b c → ¬(a + 2 * b + 3 * c > 5) :=
by
  intro h
  -- The proof goes here, currently using sorry as placeholder
  sorry

end probability_a_2b_3c_gt_5_l197_197715


namespace remaining_sweets_in_packet_l197_197085

theorem remaining_sweets_in_packet 
  (C : ℕ) (S : ℕ) (P : ℕ) (R : ℕ) (L : ℕ)
  (HC : C = 30) (HS : S = 100) (HP : P = 60) (HR : R = 25) (HL : L = 150) 
  : (C - (2 * C / 5) - ((C - P / 4) / 3)) 
  + (S - (S / 4)) 
  + (P - (3 * P / 5)) 
  + ((max 0 (R - (3 * R / 2))))
  + (L - (3 * (S / 4) / 2)) = 232 :=
by
  sorry

end remaining_sweets_in_packet_l197_197085


namespace initial_rows_l197_197509

theorem initial_rows (r T : ℕ) (h1 : T = 42 * r) (h2 : T = 28 * (r + 12)) : r = 24 :=
by
  sorry

end initial_rows_l197_197509


namespace find_point_N_l197_197867

theorem find_point_N 
  (M N : ℝ × ℝ) 
  (MN_length : Real.sqrt (((N.1 - M.1) ^ 2) + ((N.2 - M.2) ^ 2)) = 4)
  (MN_parallel_y_axis : N.1 = M.1)
  (M_coord : M = (-1, 2)) 
  : (N = (-1, 6)) ∨ (N = (-1, -2)) :=
sorry

end find_point_N_l197_197867


namespace intersection_point_l197_197280

theorem intersection_point (x y : ℚ) 
  (h1 : 3 * y = -2 * x + 6) 
  (h2 : 2 * y = 7 * x - 4) :
  x = 24 / 25 ∧ y = 34 / 25 :=
sorry

end intersection_point_l197_197280


namespace joe_eggs_around_park_l197_197354

variable (total_eggs club_house_eggs town_hall_garden_eggs park_eggs : ℕ)

def joe_eggs (total_eggs club_house_eggs town_hall_garden_eggs park_eggs : ℕ) : Prop :=
  total_eggs = club_house_eggs + town_hall_garden_eggs + park_eggs

theorem joe_eggs_around_park (h1 : total_eggs = 20) (h2 : club_house_eggs = 12) (h3 : town_hall_garden_eggs = 3) :
  ∃ park_eggs, joe_eggs total_eggs club_house_eggs town_hall_garden_eggs park_eggs ∧ park_eggs = 5 :=
by
  sorry

end joe_eggs_around_park_l197_197354


namespace find_angle_A_l197_197675

theorem find_angle_A (A B C : ℝ) (a b c : ℝ) 
  (h : 1 + (Real.tan A / Real.tan B) = 2 * c / b) : 
  A = Real.pi / 3 :=
sorry

end find_angle_A_l197_197675


namespace proof_problem_l197_197842

theorem proof_problem (a b : ℝ) (h : a^2 + b^2 + 2*a - 4*b + 5 = 0) : 2*a^2 + 4*b - 3 = 7 :=
sorry

end proof_problem_l197_197842


namespace ellie_total_distance_after_six_steps_l197_197439

-- Define the initial conditions and parameters
def initial_position : ℚ := 0
def target_distance : ℚ := 5
def step_fraction : ℚ := 1 / 4
def steps : ℕ := 6

-- Define the function that calculates the sum of the distances walked
def distance_walked (n : ℕ) : ℚ :=
  let first_term := target_distance * step_fraction
  let common_ratio := 3 / 4
  first_term * (1 - common_ratio^n) / (1 - common_ratio)

-- Define the theorem we want to prove
theorem ellie_total_distance_after_six_steps :
  distance_walked steps = 16835 / 4096 :=
by 
  sorry

end ellie_total_distance_after_six_steps_l197_197439


namespace find_d_l197_197599

variable {x1 x2 k d : ℝ}

axiom h₁ : x1 ≠ x2
axiom h₂ : 4 * x1^2 - k * x1 = d
axiom h₃ : 4 * x2^2 - k * x2 = d
axiom h₄ : x1 + x2 = 2

theorem find_d : d = -12 := by
  sorry

end find_d_l197_197599


namespace total_amount_spent_l197_197461

variable (your_spending : ℝ) (friend_spending : ℝ)
variable (h1 : friend_spending = your_spending + 3) (h2 : friend_spending = 10)

theorem total_amount_spent : your_spending + friend_spending = 17 :=
by sorry

end total_amount_spent_l197_197461


namespace mass_percentage_of_O_in_CaCO3_l197_197391

-- Assuming the given conditions as definitions
def molar_mass_Ca : ℝ := 40.08
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def formula_CaCO3 : (ℕ × ℝ) := (1, molar_mass_Ca) -- 1 atom of Calcium
def formula_CaCO3_C : (ℕ × ℝ) := (1, molar_mass_C) -- 1 atom of Carbon
def formula_CaCO3_O : (ℕ × ℝ) := (3, molar_mass_O) -- 3 atoms of Oxygen

-- Desired result
def mass_percentage_O_CaCO3 : ℝ := 47.95

-- The theorem statement to be proven
theorem mass_percentage_of_O_in_CaCO3 :
  let molar_mass_CaCO3 := formula_CaCO3.2 + formula_CaCO3_C.2 + (formula_CaCO3_O.1 * formula_CaCO3_O.2)
  let mass_percentage_O := (formula_CaCO3_O.1 * formula_CaCO3_O.2 / molar_mass_CaCO3) * 100
  mass_percentage_O = mass_percentage_O_CaCO3 :=
by
  sorry

end mass_percentage_of_O_in_CaCO3_l197_197391


namespace intersection_A_complement_B_range_of_a_l197_197667

-- Define sets A and B with their respective conditions
def U : Set ℝ := Set.univ
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Question 1: Prove the intersection when a = 2
theorem intersection_A_complement_B (a : ℝ) (h : a = 2) : 
  A a ∩ (U \ B a) = {x | 2 < x ∧ x ≤ 4} ∪ {x | 5 ≤ x ∧ x < 7} :=
by sorry

-- Question 2: Find the range of a such that A ∪ B = A given a ≠ 1
theorem range_of_a (a : ℝ) (h : a ≠ 1) : 
  (A a ∪ B a = A a) ↔ (1 < a ∧ a ≤ 3 ∨ a = -1) :=
by sorry

end intersection_A_complement_B_range_of_a_l197_197667


namespace percentage_increase_is_20_percent_l197_197975

noncomputable def SP : ℝ := 8600
noncomputable def CP : ℝ := 7166.67
noncomputable def percentageIncrease : ℝ := ((SP - CP) / CP) * 100

theorem percentage_increase_is_20_percent : percentageIncrease = 20 :=
by
  sorry

end percentage_increase_is_20_percent_l197_197975


namespace largest_base5_number_conversion_l197_197131

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l197_197131


namespace alpha_beta_square_l197_197211

-- Statement of the problem in Lean 4
theorem alpha_beta_square :
  ∀ (α β : ℝ), (α ≠ β ∧ ∀ x : ℝ, x^2 - 2 * x - 1 = 0 → (x = α ∨ x = β)) → (α - β)^2 = 8 := 
by
  intros α β h
  sorry

end alpha_beta_square_l197_197211


namespace carbon_copies_after_folding_l197_197152

-- Define the initial condition of sheets and carbon papers
def initial_sheets : ℕ := 3
def initial_carbons : ℕ := 2

-- Define the condition of folding the paper
def fold_paper (sheets carbons : ℕ) : ℕ := sheets * 2

-- Statement of the problem
theorem carbon_copies_after_folding : (fold_paper initial_sheets initial_carbons - initial_sheets + initial_carbons) = 4 :=
by
  sorry

end carbon_copies_after_folding_l197_197152


namespace center_of_circle_eq_l197_197758

theorem center_of_circle_eq {x y : ℝ} : (x - 2)^2 + (y - 3)^2 = 1 → (x, y) = (2, 3) :=
by
  intro h
  sorry

end center_of_circle_eq_l197_197758


namespace probability_at_least_two_defective_probability_at_most_one_defective_l197_197660

variable (P_no_defective : ℝ)
variable (P_one_defective : ℝ)
variable (P_two_defective : ℝ)
variable (P_all_defective : ℝ)

theorem probability_at_least_two_defective (hP_no_defective : P_no_defective = 0.18)
                                          (hP_one_defective : P_one_defective = 0.53)
                                          (hP_two_defective : P_two_defective = 0.27)
                                          (hP_all_defective : P_all_defective = 0.02) :
  P_two_defective + P_all_defective = 0.29 :=
  by sorry

theorem probability_at_most_one_defective (hP_no_defective : P_no_defective = 0.18)
                                          (hP_one_defective : P_one_defective = 0.53)
                                          (hP_two_defective : P_two_defective = 0.27)
                                          (hP_all_defective : P_all_defective = 0.02) :
  P_no_defective + P_one_defective = 0.71 :=
  by sorry

end probability_at_least_two_defective_probability_at_most_one_defective_l197_197660


namespace meal_service_count_l197_197681

/-- Define the number of people -/
def people_count : ℕ := 10

/-- Define the number of people that order pasta -/
def pasta_count : ℕ := 5

/-- Define the number of people that order salad -/
def salad_count : ℕ := 5

/-- Combination function to choose 2 people from 10 -/
def choose_2_from_10 : ℕ := Nat.choose 10 2

/-- Number of derangements of 8 people where exactly 2 people receive their correct meals -/
def derangement_8 : ℕ := 21

/-- Number of ways to correctly serve the meals where exactly 2 people receive the correct meal -/
theorem meal_service_count :
  choose_2_from_10 * derangement_8 = 945 :=
  by sorry

end meal_service_count_l197_197681


namespace car_travel_distance_l197_197153

theorem car_travel_distance :
  ∃ S : ℝ, 
    (S > 0) ∧ 
    (∃ v1 v2 t1 t2 t3 t4 : ℝ, 
      (S / 2 = v1 * t1) ∧ (26.25 = v2 * t2) ∧ 
      (S / 2 = v2 * t3) ∧ (31.2 = v1 * t4) ∧ 
      (∃ k : ℝ, k = (S - 31.2) / (v1 + v2) ∧ k > 0 ∧ 
        (S = 58))) := sorry

end car_travel_distance_l197_197153


namespace art_club_artworks_l197_197235

-- Define the conditions
def students := 25
def artworks_per_student_per_quarter := 3
def quarters_per_year := 4
def years := 3

-- Calculate total artworks
theorem art_club_artworks : 
  students * artworks_per_student_per_quarter * quarters_per_year * years = 900 :=
by
  sorry

end art_club_artworks_l197_197235


namespace problem_statement_l197_197527

variable (x : ℝ)

theorem problem_statement (h : x^2 - x - 1 = 0) : 1995 + 2 * x - x^3 = 1994 := by
  sorry

end problem_statement_l197_197527


namespace total_snow_volume_l197_197597

theorem total_snow_volume (length width initial_depth additional_depth: ℝ) 
  (h_length : length = 30) 
  (h_width : width = 3) 
  (h_initial_depth : initial_depth = 3 / 4) 
  (h_additional_depth : additional_depth = 1 / 4) : 
  (length * width * initial_depth) + (length * width * additional_depth) = 90 := 
by
  -- proof steps would go here
  sorry

end total_snow_volume_l197_197597


namespace subtraction_of_fractions_l197_197558

theorem subtraction_of_fractions :
  let S_1 := 3 + 5 + 7
  let S_2 := 2 + 4 + 6
  let S_3 := 2 + 4 + 6
  let S_4 := 3 + 5 + 7
  (S_1 / S_2 - S_3 / S_4) = 9 / 20 :=
by
  let S_1 := 3 + 5 + 7
  let S_2 := 2 + 4 + 6
  let S_3 := 2 + 4 + 6
  let S_4 := 3 + 5 + 7
  sorry

end subtraction_of_fractions_l197_197558


namespace diff_squares_example_l197_197361

theorem diff_squares_example :
  (311^2 - 297^2) / 14 = 608 :=
by
  -- The theorem statement directly follows from the conditions and question.
  sorry

end diff_squares_example_l197_197361


namespace fractional_product_l197_197406

theorem fractional_product :
  ((3/4) * (4/5) * (5/6) * (6/7) * (7/8)) = 3/8 :=
by
  sorry

end fractional_product_l197_197406


namespace find_certain_number_l197_197818

theorem find_certain_number (mystery_number certain_number : ℕ) (h1 : mystery_number = 47) 
(h2 : mystery_number + certain_number = 92) : certain_number = 45 :=
by
  sorry

end find_certain_number_l197_197818


namespace alice_walking_speed_l197_197076

theorem alice_walking_speed:
  ∃ v : ℝ, 
  (∀ t : ℝ, t = 1 → ∀ d_a d_b : ℝ, d_a = 25 → d_b = 41 - d_a → 
  ∀ s_b : ℝ, s_b = 4 → 
  d_b / s_b + t = d_a / v) ∧ v = 5 :=
by
  sorry

end alice_walking_speed_l197_197076


namespace percentage_of_number_l197_197405

theorem percentage_of_number (N : ℕ) (P : ℕ) (h1 : N = 120) (h2 : (3 * N) / 5 = 72) (h3 : (P * 72) / 100 = 36) : P = 50 :=
sorry

end percentage_of_number_l197_197405


namespace problem_l197_197961

def binom (n k : ℕ) : ℕ := n.choose k

def perm (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem problem : binom 10 3 * perm 8 2 = 6720 := by
  sorry

end problem_l197_197961


namespace unique_common_root_m_value_l197_197365

theorem unique_common_root_m_value (m : ℝ) (h : m > 5) :
  (∃ x : ℝ, x^2 - 5 * x + 6 = 0 ∧ x^2 + 2 * x - 2 * m + 1 = 0) →
  m = 8 :=
by
  sorry

end unique_common_root_m_value_l197_197365


namespace hoseok_needs_17_more_jumps_l197_197870

/-- Define the number of jumps by Hoseok and Minyoung -/
def hoseok_jumps : ℕ := 34
def minyoung_jumps : ℕ := 51

/-- Define the number of additional jumps Hoseok needs -/
def additional_jumps_hoseok : ℕ := minyoung_jumps - hoseok_jumps

/-- Prove that the additional jumps Hoseok needs is equal to 17 -/
theorem hoseok_needs_17_more_jumps (h_jumps : ℕ := hoseok_jumps) (m_jumps : ℕ := minyoung_jumps) :
  additional_jumps_hoseok = 17 := by
  -- Proof goes here
  sorry

end hoseok_needs_17_more_jumps_l197_197870


namespace maximum_k_value_l197_197795

noncomputable def max_value_k (a : ℝ) (b : ℝ) (k : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1 ∧ a^2 + b^2 ≥ k ∧ k = 1 / 2

theorem maximum_k_value (a b : ℝ) :
  (a > 0 ∧ b > 0 ∧ a + b = 1) → a^2 + b^2 ≥ 1 / 2 :=
by
  intro h
  obtain ⟨ha, hb, hab⟩ := h
  sorry

end maximum_k_value_l197_197795


namespace contrapositive_inverse_converse_negation_false_l197_197241

theorem contrapositive (a b : ℤ) : (a ≤ b) → (a - 2 ≤ b - 2) :=
sorry

theorem inverse (a b : ℤ) : (a - 2 ≤ b - 2) → (a ≤ b) :=
sorry

theorem converse (a b : ℤ) : (a - 2 > b - 2) → (a > b) :=
sorry

theorem negation_false (a b : ℤ) : ¬ ((a > b) → (a - 2 ≤ b - 2)) :=
sorry

end contrapositive_inverse_converse_negation_false_l197_197241


namespace union_A_B_inter_complB_A_l197_197333

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set A
def A : Set ℝ := {x | -3 < x ∧ x ≤ 6}

-- Define the set B
def B : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- Define the complement of B with respect to U
def compl_B : Set ℝ := {x | x ≤ -1 ∨ x ≥ 6}

-- Problem (1): Prove that A ∪ B = {x | -3 < x ∧ x ≤ 6}
theorem union_A_B : A ∪ B = {x | -3 < x ∧ x ≤ 6} := by
  sorry

-- Problem (2): Prove that (compl_B) ∩ A = {x | (-3 < x ∧ x ≤ -1) ∨ x = 6}
theorem inter_complB_A : compl_B ∩ A = {x | (-3 < x ∧ x ≤ -1) ∨ x = 6} := by 
  sorry

end union_A_B_inter_complB_A_l197_197333


namespace original_amount_water_l197_197113

theorem original_amount_water (O : ℝ) (h1 : (0.75 = 0.05 * O)) : O = 15 :=
by sorry

end original_amount_water_l197_197113


namespace solve_system_eqn_l197_197398

theorem solve_system_eqn (x y : ℚ) (h₁ : 3*y - 4*x = 8) (h₂ : 2*y + x = -1) :
  x = -19/11 ∧ y = 4/11 :=
by
  sorry

end solve_system_eqn_l197_197398


namespace find_integers_l197_197765

theorem find_integers (a b c : ℤ) (h1 : ∃ x : ℤ, a = 2 * x ∧ b = 5 * x ∧ c = 8 * x)
  (h2 : a + 6 = b / 3)
  (h3 : c - 10 = 5 * a / 4) :
  a = 36 ∧ b = 90 ∧ c = 144 :=
by
  sorry

end find_integers_l197_197765


namespace smallest_constant_N_l197_197409

theorem smallest_constant_N (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) :
  (a^2 + b^2 + c^2) / (a * b + b * c + c * a) > 1 :=
by
  sorry

end smallest_constant_N_l197_197409


namespace roots_sum_eq_product_l197_197157

theorem roots_sum_eq_product (m : ℝ) :
  (∀ x : ℝ, 2 * (x - 1) * (x - 3 * m) = x * (m - 4)) →
  (∀ a b : ℝ, 2 * a * b = 2 * (5 * m + 6) / -2 ∧ 2 * a * b = 6 * m / 2) →
  m = -2 / 3 :=
by
  sorry

end roots_sum_eq_product_l197_197157


namespace pounds_per_pie_l197_197708

-- Define the conditions
def total_weight : ℕ := 120
def applesauce_weight := total_weight / 2
def pies_weight := total_weight - applesauce_weight
def number_of_pies := 15

-- Define the required proof for pounds per pie
theorem pounds_per_pie :
  pies_weight / number_of_pies = 4 := by
  sorry

end pounds_per_pie_l197_197708


namespace degrees_to_radians_l197_197941

theorem degrees_to_radians (degrees : ℝ) (pi : ℝ) : 
  degrees * (pi / 180) = pi / 15 ↔ degrees = 12 :=
by 
  sorry

end degrees_to_radians_l197_197941


namespace hockey_league_games_l197_197670

def num_teams : ℕ := 18
def encounters_per_pair : ℕ := 10
def num_games (n : ℕ) (k : ℕ) : ℕ := (n * (n - 1)) / 2 * k

theorem hockey_league_games :
  num_games num_teams encounters_per_pair = 1530 :=
by
  sorry

end hockey_league_games_l197_197670


namespace existence_not_implied_by_validity_l197_197178

-- Let us formalize the theorem and then show that its validity does not imply the existence of such a function.

-- Definitions for condition (A) and the theorem statement
axiom condition_A (f : ℝ → ℝ) : Prop
axiom theorem_239 : ∀ f, condition_A f → ∃ T, ∀ x, f (x + T) = f x

-- Translation of the problem statement into Lean
theorem existence_not_implied_by_validity :
  (∀ f, condition_A f → ∃ T, ∀ x, f (x + T) = f x) → 
  ¬ (∃ f, condition_A f) :=
sorry

end existence_not_implied_by_validity_l197_197178


namespace john_spent_on_sweets_l197_197114

theorem john_spent_on_sweets (initial_amount : ℝ) (amount_given_per_friend : ℝ) (friends : ℕ) (amount_left : ℝ) (total_spent_on_sweets : ℝ) :
  initial_amount = 20.10 →
  amount_given_per_friend = 1.00 →
  friends = 2 →
  amount_left = 17.05 →
  total_spent_on_sweets = initial_amount - (amount_given_per_friend * friends) - amount_left →
  total_spent_on_sweets = 1.05 :=
by
  intros h_initial h_given h_friends h_left h_spent
  sorry

end john_spent_on_sweets_l197_197114


namespace problem_1_problem_2_l197_197851

theorem problem_1 (a : ℝ) : (∀ x1 x2 : ℝ, (a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) → x1 = x2) → (a = 0 ∨ a = 1) :=
by sorry

theorem problem_2 (a : ℝ) : (∀ x1 x2 : ℝ, (a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) → x1 = x2 ∨ ¬ ∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a ≥ 1 ∨ a = 0) :=
by sorry

end problem_1_problem_2_l197_197851


namespace add_fifteen_sub_fifteen_l197_197189

theorem add_fifteen (n : ℕ) (m : ℕ) : n + m = 195 :=
by {
  sorry  -- placeholder for the actual proof
}

theorem sub_fifteen (n : ℕ) (m : ℕ) : n - m = 165 :=
by {
  sorry  -- placeholder for the actual proof
}

-- Let's instantiate these theorems with the specific values from the problem:
noncomputable def verify_addition : 180 + 15 = 195 :=
by exact add_fifteen 180 15

noncomputable def verify_subtraction : 180 - 15 = 165 :=
by exact sub_fifteen 180 15

end add_fifteen_sub_fifteen_l197_197189


namespace percentage_increase_14point4_from_12_l197_197686

theorem percentage_increase_14point4_from_12 (x : ℝ) (h : x = 14.4) : 
  ((x - 12) / 12) * 100 = 20 := 
by
  sorry

end percentage_increase_14point4_from_12_l197_197686


namespace gcd_1080_920_is_40_l197_197223

theorem gcd_1080_920_is_40 : Nat.gcd 1080 920 = 40 :=
by
  sorry

end gcd_1080_920_is_40_l197_197223


namespace gcd_51457_37958_l197_197112

theorem gcd_51457_37958 : Nat.gcd 51457 37958 = 1 := 
  sorry

end gcd_51457_37958_l197_197112


namespace least_number_of_roots_l197_197909

variable {g : ℝ → ℝ}

-- Conditions
axiom g_defined (x : ℝ) : g x = g x
axiom g_symmetry_1 (x : ℝ) : g (3 + x) = g (3 - x)
axiom g_symmetry_2 (x : ℝ) : g (5 + x) = g (5 - x)
axiom g_at_1 : g 1 = 0

-- Root count in the interval
theorem least_number_of_roots : ∃ (n : ℕ), n >= 250 ∧ (∀ m, -1000 ≤ (1 + 8 * m:ℝ) ∧ (1 + 8 * m:ℝ) ≤ 1000 → g (1 + 8 * m) = 0) :=
sorry

end least_number_of_roots_l197_197909


namespace hexagon_coloring_l197_197399

def valid_coloring_hexagon : Prop :=
  ∃ (A B C D E F : Fin 8), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ A ≠ D ∧ B ≠ D ∧ C ≠ D ∧
    B ≠ E ∧ C ≠ E ∧ D ≠ E ∧ A ≠ F ∧ C ≠ F ∧ E ≠ F

theorem hexagon_coloring : ∃ (n : Nat), valid_coloring_hexagon ∧ n = 20160 := 
sorry

end hexagon_coloring_l197_197399


namespace middle_term_is_35_l197_197296

-- Define the arithmetic sequence condition
def arithmetic_sequence (a b c d e f : ℤ) : Prop :=
  b - a = c - b ∧ c - b = d - c ∧ d - c = e - d ∧ e - d = f - e

-- Given sequence values
def seq1 := 23
def seq6 := 47

-- Theorem stating that the middle term y in the sequence is 35
theorem middle_term_is_35 (x y z w : ℤ) :
  arithmetic_sequence seq1 x y z w seq6 → y = 35 :=
by
  sorry

end middle_term_is_35_l197_197296


namespace smallest_N_for_circular_table_l197_197212

/--
  Given a circular table with 60 chairs, prove that the smallest number of people, N,
  such that any additional person must sit next to someone already seated is 20.
-/
theorem smallest_N_for_circular_table (N : ℕ) (h : N = 20) : 
  ∀ (next_seated : ℕ), next_seated ≤ N → (∃ i : ℕ, i < N ∧ next_seated = i + 1 ∨ next_seated = i - 1) :=
by
  sorry

end smallest_N_for_circular_table_l197_197212


namespace necessary_but_not_sufficient_condition_l197_197516

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  ((a > 2) ∧ (b > 2) → (a + b > 4)) ∧ ¬((a + b > 4) → (a > 2) ∧ (b > 2)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l197_197516


namespace sum_of_digits_of_special_two_digit_number_l197_197451

theorem sum_of_digits_of_special_two_digit_number (x : ℕ) (h1 : 1 ≤ x ∧ x < 10) 
  (h2 : ∃ (n : ℕ), n = 11 * x + 30) 
  (h3 : ∃ (sum_digits : ℕ), sum_digits = (x + 3) + x) 
  (h4 : (11 * x + 30) % ((x + 3) + x) = 3)
  (h5 : (11 * x + 30) / ((x + 3) + x) = 7) :
  (x + 3) + x = 7 := 
by 
  sorry

end sum_of_digits_of_special_two_digit_number_l197_197451


namespace probability_of_usable_gas_pipe_l197_197397

theorem probability_of_usable_gas_pipe (x y : ℝ)
  (hx : 75 ≤ x) 
  (hy : 75 ≤ y)
  (hxy : x + y ≤ 225) :
  (∃ x y, 0 < x ∧ 0 < y ∧ x < 300 ∧ y < 300 ∧ x + y > 75 ∧ (300 - x - y) ≥ 75) → 
  ((150 * 150) / (300 * 300 / 2) = (1 / 4)) :=
by {
  sorry
}

end probability_of_usable_gas_pipe_l197_197397


namespace no_polynomial_deg_ge_3_satisfies_conditions_l197_197109

theorem no_polynomial_deg_ge_3_satisfies_conditions :
  ¬ ∃ f : Polynomial ℝ, f.degree ≥ 3 ∧ f.eval (x^2) = (f.eval x)^2 ∧ f.coeff 2 = 0 :=
sorry

end no_polynomial_deg_ge_3_satisfies_conditions_l197_197109


namespace problem_statement_l197_197039
noncomputable def a : ℕ := 10
noncomputable def b : ℕ := a^3

theorem problem_statement (a b : ℕ) (a_pos : 0 < a) (b_eq : b = a^3)
    (log_ab : Real.logb a (b : ℝ) = 3) (b_minus_a : b = a + 891) :
    a + b = 1010 :=
by
  sorry

end problem_statement_l197_197039


namespace cost_per_meter_of_fencing_l197_197936

/-- The sides of the rectangular field -/
def sides_ratio (length width : ℕ) : Prop := 3 * width = 4 * length

/-- The area of the rectangular field -/
def area (length width area : ℕ) : Prop := length * width = area

/-- The cost per meter of fencing -/
def cost_per_meter (total_cost perimeter : ℕ) : ℕ := total_cost * 100 / perimeter

/-- Prove that the cost per meter of fencing the field in paise is 25 given:
 1) The sides of a rectangular field are in the ratio 3:4.
 2) The area of the field is 8112 sq. m.
 3) The total cost of fencing the field is 91 rupees. -/
theorem cost_per_meter_of_fencing
  (length width perimeter : ℕ) 
  (h1 : sides_ratio length width)
  (h2 : area length width 8112)
  (h3 : perimeter = 2 * (length + width))
  (total_cost : ℕ)
  (h4 : total_cost = 91) :
  cost_per_meter total_cost perimeter = 25 :=
by
  sorry

end cost_per_meter_of_fencing_l197_197936


namespace area_ratio_l197_197183

theorem area_ratio (l b r : ℝ) (h1 : l = 2 * b) (h2 : 6 * b = 2 * π * r) :
  (l * b) / (π * r ^ 2) = 2 * π / 9 :=
by {
  sorry
}

end area_ratio_l197_197183


namespace sequence_a6_value_l197_197473

theorem sequence_a6_value 
  (a : ℕ → ℝ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 1)
  (h3 : ∀ n : ℕ, n ≥ 1 → (1 / a n) + (1 / a (n + 2)) = 2 / a (n + 1)) :
  a 6 = 1 / 3 :=
by
  sorry

end sequence_a6_value_l197_197473


namespace sum_of_coordinates_l197_197998

-- Define the conditions for m and n
def m : ℤ := -3
def n : ℤ := 2

-- State the proposition based on the conditions
theorem sum_of_coordinates : m + n = -1 := 
by 
  -- Provide an incomplete proof skeleton with "sorry" to skip the proof
  sorry

end sum_of_coordinates_l197_197998


namespace find_c_values_l197_197782

noncomputable def line_intercept_product (c : ℝ) : Prop :=
  let x_intercept := -c / 8
  let y_intercept := -c / 5
  x_intercept * y_intercept = 24

theorem find_c_values :
  ∃ c : ℝ, (line_intercept_product c) ∧ (c = 8 * Real.sqrt 15 ∨ c = -8 * Real.sqrt 15) :=
by
  sorry

end find_c_values_l197_197782


namespace determine_base_l197_197172

theorem determine_base (b : ℕ) (h : (3 * b + 1)^2 = b^3 + 2 * b + 1) : b = 10 :=
by
  sorry

end determine_base_l197_197172


namespace cosine_theorem_l197_197803

theorem cosine_theorem (a b c : ℝ) (A : ℝ) (hA : 0 < A) (hA_lt_pi : A < π) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A := sorry

end cosine_theorem_l197_197803


namespace right_triangle_and_inverse_l197_197666

theorem right_triangle_and_inverse :
  30 * 30 + 272 * 272 = 278 * 278 ∧ (∃ (n : ℕ), 0 ≤ n ∧ n < 4079 ∧ (550 * n) % 4079 = 1) :=
by
  sorry

end right_triangle_and_inverse_l197_197666


namespace bricks_needed_per_square_meter_l197_197054

theorem bricks_needed_per_square_meter 
  (num_rooms : ℕ) (room_length room_breadth : ℕ) (total_bricks : ℕ)
  (h1 : num_rooms = 5)
  (h2 : room_length = 4)
  (h3 : room_breadth = 5)
  (h4 : total_bricks = 340) : 
  (total_bricks / (room_length * room_breadth)) = 17 := 
by
  sorry

end bricks_needed_per_square_meter_l197_197054


namespace vertical_asymptote_l197_197690

theorem vertical_asymptote (x : ℝ) : 
  (∃ x, 4 * x + 5 = 0) → x = -5/4 :=
by 
  sorry

end vertical_asymptote_l197_197690


namespace remainder_55_pow_55_plus_10_mod_8_l197_197356

theorem remainder_55_pow_55_plus_10_mod_8 : (55 ^ 55 + 10) % 8 = 1 :=
by
  sorry

end remainder_55_pow_55_plus_10_mod_8_l197_197356


namespace calculate_square_difference_l197_197960

theorem calculate_square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 :=
by
  sorry

end calculate_square_difference_l197_197960


namespace remainder_of_9876543210_div_101_l197_197543

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 :=
  sorry

end remainder_of_9876543210_div_101_l197_197543


namespace perimeter_of_ghost_l197_197847
open Real

def radius := 2
def angle_degrees := 90
def full_circle_degrees := 360

noncomputable def missing_angle := angle_degrees
noncomputable def remaining_angle := full_circle_degrees - missing_angle
noncomputable def fraction_of_circle := remaining_angle / full_circle_degrees
noncomputable def full_circumference := 2 * π * radius
noncomputable def arc_length := fraction_of_circle * full_circumference
noncomputable def radii_length := 2 * radius

theorem perimeter_of_ghost : arc_length + radii_length = 3 * π + 4 :=
by
  sorry

end perimeter_of_ghost_l197_197847


namespace greatest_sum_of_int_pairs_squared_eq_64_l197_197264

theorem greatest_sum_of_int_pairs_squared_eq_64 :
  ∃ (x y : ℤ), x^2 + y^2 = 64 ∧ (∀ (a b : ℤ), a^2 + b^2 = 64 → a + b ≤ 8) ∧ x + y = 8 :=
by 
  sorry

end greatest_sum_of_int_pairs_squared_eq_64_l197_197264


namespace a_11_is_12_l197_197234

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
def a_2 (a : ℕ → ℝ) := a 2 = 3
def a_6 (a : ℕ → ℝ) := a 6 = 7

-- The statement to prove
theorem a_11_is_12 (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) (h_a2 : a_2 a) (h_a6 : a_6 a) : a 11 = 12 :=
  sorry

end a_11_is_12_l197_197234


namespace find_a6_l197_197712

noncomputable def a (n : ℕ) : ℝ := sorry

axiom geom_seq_inc :
  ∀ n : ℕ, a n < a (n + 1)

axiom root_eqn_a2_a4 :
  ∃ a2 a4 : ℝ, (a 2 = a2) ∧ (a 4 = a4) ∧ (a2^2 - 6 * a2 + 5 = 0) ∧ (a4^2 - 6 * a4 + 5 = 0)

theorem find_a6 : a 6 = 25 := 
sorry

end find_a6_l197_197712


namespace length_of_train_l197_197854

variable (L : ℝ) (S : ℝ)

-- Condition 1: The train crosses a 120 meters platform in 15 seconds
axiom condition1 : S = (L + 120) / 15

-- Condition 2: The train crosses a 250 meters platform in 20 seconds
axiom condition2 : S = (L + 250) / 20

-- The theorem to be proved
theorem length_of_train : L = 270 :=
by
  sorry

end length_of_train_l197_197854


namespace fourth_term_geometric_sequence_l197_197678

theorem fourth_term_geometric_sequence (x : ℝ) :
  ∃ r : ℝ, (r > 0) ∧ 
  x ≠ 0 ∧
  (3 * x + 3)^2 = x * (6 * x + 6) →
  x = -3 →
  6 * x + 6 ≠ 0 →
  4 * (6 * x + 6) * (3 * x + 3) = -24 :=
by
  -- Placeholder for the proof steps
  sorry

end fourth_term_geometric_sequence_l197_197678


namespace determinant_of_sine_matrix_is_zero_l197_197912

theorem determinant_of_sine_matrix_is_zero : 
  let M : Matrix (Fin 3) (Fin 3) ℝ :=
    ![![Real.sin 2, Real.sin 3, Real.sin 4],
      ![Real.sin 5, Real.sin 6, Real.sin 7],
      ![Real.sin 8, Real.sin 9, Real.sin 10]]
  Matrix.det M = 0 := 
by sorry

end determinant_of_sine_matrix_is_zero_l197_197912


namespace find_initial_red_balloons_l197_197651

-- Define the initial state of balloons and the assumption.
def initial_blue_balloons : ℕ := 4
def red_balloons_after_inflation (R : ℕ) : ℕ := R + 2
def blue_balloons_after_inflation : ℕ := initial_blue_balloons + 2
def total_balloons (R : ℕ) : ℕ := red_balloons_after_inflation R + blue_balloons_after_inflation

-- Define the likelihood condition.
def likelihood_red (R : ℕ) : Prop := (red_balloons_after_inflation R : ℚ) / (total_balloons R : ℚ) = 0.4

-- Statement of the problem.
theorem find_initial_red_balloons (R : ℕ) (h : likelihood_red R) : R = 2 := by
  sorry

end find_initial_red_balloons_l197_197651


namespace complement_union_l197_197938

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end complement_union_l197_197938


namespace num_teacher_volunteers_l197_197957

theorem num_teacher_volunteers (total_needed volunteers_from_classes extra_needed teacher_volunteers : ℕ)
  (h1 : teacher_volunteers + extra_needed + volunteers_from_classes = total_needed) 
  (h2 : total_needed = 50)
  (h3 : volunteers_from_classes = 6 * 5)
  (h4 : extra_needed = 7) :
  teacher_volunteers = 13 :=
by
  sorry

end num_teacher_volunteers_l197_197957


namespace route_down_distance_l197_197770

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

end route_down_distance_l197_197770


namespace Cindy_initial_marbles_l197_197997

theorem Cindy_initial_marbles (M : ℕ) 
  (h1 : 4 * (M - 320) = 720) : M = 500 :=
by
  sorry

end Cindy_initial_marbles_l197_197997


namespace cubic_function_decreasing_l197_197464

theorem cubic_function_decreasing (a : ℝ) :
  (∀ x : ℝ, 3 * a * x^2 - 1 ≤ 0) → (a ≤ 0) := 
by 
  sorry

end cubic_function_decreasing_l197_197464


namespace unique_m_value_l197_197890

theorem unique_m_value : ∀ m : ℝ,
  (m ^ 2 - 5 * m + 6 = 0 ∧ m ^ 2 - 3 * m + 2 = 0) →
  (m ^ 2 - 3 * m + 2 = 2 * (m ^ 2 - 5 * m + 6)) →
  ((m ^ 2 - 5 * m + 6) * (m ^ 2 - 3 * m + 2) > 0) →
  m = 2 :=
by
  sorry

end unique_m_value_l197_197890


namespace no_solution_for_steers_and_cows_purchase_l197_197087

theorem no_solution_for_steers_and_cows_purchase :
  ¬ ∃ (s c : ℕ), 30 * s + 32 * c = 1200 ∧ c > s :=
by
  sorry

end no_solution_for_steers_and_cows_purchase_l197_197087


namespace decimal_to_vulgar_fraction_l197_197962

theorem decimal_to_vulgar_fraction (d : ℚ) (h : d = 0.36) : d = 9 / 25 :=
by {
  sorry
}

end decimal_to_vulgar_fraction_l197_197962


namespace tangent_line_at_zero_l197_197154

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem tangent_line_at_zero : ∀ x : ℝ, x = 0 → Real.exp x * Real.sin x = 0 ∧ (Real.exp x * (Real.sin x + Real.cos x)) = 1 → (∀ y, y = x) :=
  by
    sorry

end tangent_line_at_zero_l197_197154


namespace find_m_l197_197128

variable (m : ℝ)

def vector_oa : ℝ × ℝ := (-1, 2)
def vector_ob : ℝ × ℝ := (3, m)

def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_m
  (h : orthogonal (vector_oa) (vector_ob m)) :
  m = 3 / 2 := by
  sorry

end find_m_l197_197128


namespace sample_capacity_l197_197029

theorem sample_capacity 
  (n : ℕ) 
  (model_A : ℕ) 
  (model_B model_C : ℕ) 
  (ratio_A ratio_B ratio_C : ℕ)
  (r_A : ratio_A = 2)
  (r_B : ratio_B = 3)
  (r_C : ratio_C = 5)
  (total_production_ratio : ratio_A + ratio_B + ratio_C = 10)
  (items_model_A : model_A = 15)
  (proportion : (model_A : ℚ) / (ratio_A : ℚ) = (n : ℚ) / 10) :
  n = 75 :=
by sorry

end sample_capacity_l197_197029


namespace find_P_l197_197773

variable (P : ℕ) 

-- Conditions
def cost_samosas : ℕ := 3 * 2
def cost_mango_lassi : ℕ := 2
def cost_per_pakora : ℕ := 3
def total_cost : ℕ := 25
def tip_rate : ℚ := 0.25

-- Total cost before tip
def total_cost_before_tip (P : ℕ) : ℕ := cost_samosas + cost_mango_lassi + cost_per_pakora * P

-- Total cost with tip
def total_cost_with_tip (P : ℕ) : ℚ := 
  (total_cost_before_tip P : ℚ) + (tip_rate * total_cost_before_tip P : ℚ)

-- Proof Goal
theorem find_P (h : total_cost_with_tip P = total_cost) : P = 4 :=
by
  sorry

end find_P_l197_197773


namespace martha_initial_apples_l197_197459

theorem martha_initial_apples :
  ∀ (jane_apples james_apples keep_apples more_to_give initial_apples : ℕ),
    jane_apples = 5 →
    james_apples = jane_apples + 2 →
    keep_apples = 4 →
    more_to_give = 4 →
    initial_apples = jane_apples + james_apples + keep_apples + more_to_give →
    initial_apples = 20 :=
by
  intros jane_apples james_apples keep_apples more_to_give initial_apples
  intro h_jane
  intro h_james
  intro h_keep
  intro h_more
  intro h_initial
  exact sorry

end martha_initial_apples_l197_197459


namespace smallest_integer_l197_197734

theorem smallest_integer (k : ℕ) : 
  (∀ (n : ℕ), n = 2^2 * 3^1 * 11^1 → 
  (∀ (f : ℕ), (f = 2^4 ∨ f = 3^3 ∨ f = 13^3) → f ∣ (n * k))) → 
  k = 79092 :=
  sorry

end smallest_integer_l197_197734


namespace range_of_a_not_empty_solution_set_l197_197633

theorem range_of_a_not_empty_solution_set :
  {a : ℝ | ∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0} =
  {a : ℝ | a ∈ {a : ℝ | a < -2} ∪ {a : ℝ | a ≥ 6 / 5}} :=
sorry

end range_of_a_not_empty_solution_set_l197_197633


namespace find_fff_l197_197015

def f (x : ℚ) : ℚ :=
  if x ≥ 2 then x + 2 else x * x

theorem find_fff : f (f (3/2)) = 17/4 := by
  sorry

end find_fff_l197_197015


namespace number_of_pen_refills_l197_197162

-- Conditions
variable (k : ℕ) (x : ℕ) (hk : k > 0) (hx : (4 + k) * x = 6)

-- Question and conclusion as a theorem statement
theorem number_of_pen_refills (hk : k > 0) (hx : (4 + k) * x = 6) : 2 * x = 2 :=
sorry

end number_of_pen_refills_l197_197162


namespace bicycle_cost_price_l197_197877

theorem bicycle_cost_price (CP_A : ℝ) (CP_B : ℝ) (SP_C : ℝ)
    (h1 : CP_B = 1.60 * CP_A)
    (h2 : SP_C = 1.25 * CP_B)
    (h3 : SP_C = 225) :
    CP_A = 225 / 2.00 :=
by
  sorry -- the proof steps will follow here

end bicycle_cost_price_l197_197877


namespace problem_statement_l197_197359

theorem problem_statement (a b c : ℝ) (h : a^6 + b^6 + c^6 = 3) : a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := 
  sorry

end problem_statement_l197_197359


namespace remainder_of_prime_powers_l197_197689

theorem remainder_of_prime_powers (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  (p^(q-1) + q^(p-1)) % (p * q) = 1 := 
sorry

end remainder_of_prime_powers_l197_197689


namespace price_of_one_shirt_l197_197308

variable (P : ℝ)

-- Conditions
def cost_two_shirts := 1.5 * P
def cost_three_shirts := 1.9 * P 
def full_price_three_shirts := 3 * P
def savings := full_price_three_shirts - cost_three_shirts

-- Correct answer
theorem price_of_one_shirt (hs : savings = 11) : P = 10 :=
by
  sorry

end price_of_one_shirt_l197_197308


namespace range_of_a_l197_197787

variable (f : ℝ → ℝ)

-- f is an odd function
def odd_function : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition 1: f is an odd function
axiom h_odd : odd_function f

-- Condition 2: f(x) + f(x + 3 / 2) = 0 for any real number x
axiom h_periodicity : ∀ x : ℝ, f x + f (x + 3 / 2) = 0

-- Condition 3: f(1) > 1
axiom h_f1 : f 1 > 1

-- Condition 4: f(2) = a for some real number a
variable (a : ℝ)
axiom h_f2 : f 2 = a

-- Goal: Prove that a < -1
theorem range_of_a : a < -1 :=
  sorry

end range_of_a_l197_197787


namespace ratio_of_milk_to_water_l197_197934

namespace MixtureProblem

def initial_milk (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) : ℕ :=
  (milk_ratio * total_volume) / (milk_ratio + water_ratio)

def initial_water (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) : ℕ :=
  (water_ratio * total_volume) / (milk_ratio + water_ratio)

theorem ratio_of_milk_to_water (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) (added_water : ℕ) :
  milk_ratio = 4 → water_ratio = 1 → total_volume = 45 → added_water = 21 → 
  (initial_milk total_volume milk_ratio water_ratio) = 36 →
  (initial_water total_volume milk_ratio water_ratio + added_water) = 30 →
  (36 / 30 : ℚ) = 6 / 5 :=
by
  intros
  sorry

end MixtureProblem

end ratio_of_milk_to_water_l197_197934


namespace perfect_apples_count_l197_197477

-- Definitions (conditions)
def total_apples := 30
def too_small_fraction := (1 : ℚ) / 6
def not_ripe_fraction := (1 : ℚ) / 3
def too_small_apples := (too_small_fraction * total_apples : ℚ)
def not_ripe_apples := (not_ripe_fraction * total_apples : ℚ)

-- Statement of the theorem (proof problem)
theorem perfect_apples_count : total_apples - too_small_apples - not_ripe_apples = 15 := by
  sorry

end perfect_apples_count_l197_197477


namespace arithmetic_sequence_sum_first_five_terms_l197_197771

theorem arithmetic_sequence_sum_first_five_terms:
  ∀ (a : ℕ → ℤ), a 2 = 1 → a 4 = 7 → (a 1 + a 5 = a 2 + a 4) → (5 * (a 1 + a 5) / 2 = 20) :=
by
  intros a h1 h2 h3
  sorry

end arithmetic_sequence_sum_first_five_terms_l197_197771


namespace cost_of_mozzarella_cheese_l197_197165

-- Define the problem conditions as Lean definitions
def blendCostPerKg : ℝ := 696.05
def romanoCostPerKg : ℝ := 887.75
def weightMozzarella : ℝ := 19
def weightRomano : ℝ := 18.999999999999986  -- Practically the same as 19 in context
def totalWeight : ℝ := weightMozzarella + weightRomano

-- Define the expected result for the cost per kilogram of mozzarella cheese
def expectedMozzarellaCostPerKg : ℝ := 504.40

-- Theorem statement to verify the cost of mozzarella cheese
theorem cost_of_mozzarella_cheese :
  weightMozzarella * (expectedMozzarellaCostPerKg : ℝ) + weightRomano * romanoCostPerKg = totalWeight * blendCostPerKg := by
  sorry

end cost_of_mozzarella_cheese_l197_197165


namespace inequality_cannot_hold_l197_197839

theorem inequality_cannot_hold (a b : ℝ) (h : a < b ∧ b < 0) : ¬ (1 / (a - b) > 1 / a) :=
by {
  sorry
}

end inequality_cannot_hold_l197_197839


namespace solve_fractional_eq_l197_197926

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) : (4 / (x - 2) = 2 / x) → (x = -2) :=
by 
  sorry

end solve_fractional_eq_l197_197926


namespace trains_time_distance_l197_197702

-- Define the speeds of the two trains
def speed1 : ℕ := 11
def speed2 : ℕ := 31

-- Define the distance between the two trains after time t
def distance_between_trains (t : ℕ) : ℕ :=
  speed2 * t - speed1 * t

-- Define the condition that this distance is 160 miles
def condition (t : ℕ) : Prop :=
  distance_between_trains t = 160

-- State the theorem to prove
theorem trains_time_distance : ∃ t : ℕ, condition t ∧ t = 8 :=
by
  use 8
  unfold condition
  unfold distance_between_trains
  -- Verifying the calculated distance
  sorry

end trains_time_distance_l197_197702


namespace triangle_angles_l197_197066

theorem triangle_angles (A B C : ℝ) 
  (h1 : B = 4 * A)
  (h2 : C - B = 27)
  (h3 : A + B + C = 180) : 
  A = 17 ∧ B = 68 ∧ C = 95 :=
by {
  -- Sorry will be replaced once the actual proof is provided
  sorry 
}

end triangle_angles_l197_197066


namespace enrollment_increase_1991_to_1992_l197_197193

theorem enrollment_increase_1991_to_1992 (E E_1992 E_1993 : ℝ)
    (h1 : E_1993 = 1.26 * E)
    (h2 : E_1993 = 1.05 * E_1992) :
    ((E_1992 - E) / E) * 100 = 20 :=
by
  sorry

end enrollment_increase_1991_to_1992_l197_197193


namespace arithmetic_geometric_seq_l197_197335

open Real

theorem arithmetic_geometric_seq (a d : ℝ) (h₀ : d ≠ 0) 
  (h₁ : (a + d) * (a + 5 * d) = (a + 2 * d) ^ 2) : 
  (a + 2 * d) / (a + d) = 3 :=
sorry

end arithmetic_geometric_seq_l197_197335


namespace sufficient_condition_not_necessary_condition_l197_197654

variable (a b : ℝ)

theorem sufficient_condition (hab : (a - b) * a^2 < 0) : a < b :=
by
  sorry

theorem not_necessary_condition (h : a < b) : (a - b) * a^2 < 0 ∨ (a - b) * a^2 = 0 :=
by
  sorry

end sufficient_condition_not_necessary_condition_l197_197654


namespace lines_intersect_at_point_l197_197423

theorem lines_intersect_at_point :
  ∃ (x y : ℝ), (3 * x + 4 * y + 7 = 0) ∧ (x - 2 * y - 1 = 0) ∧ (x = -1) ∧ (y = -1) :=
by
  sorry

end lines_intersect_at_point_l197_197423


namespace last_two_digits_of_quotient_l197_197270

noncomputable def greatest_integer_not_exceeding (x : ℝ) : ℤ := ⌊x⌋

theorem last_two_digits_of_quotient :
  let a : ℤ := 10 ^ 93
  let b : ℤ := 10 ^ 31 + 3
  let x : ℤ := greatest_integer_not_exceeding (a / b : ℝ)
  (x % 100) = 8 :=
by
  sorry

end last_two_digits_of_quotient_l197_197270


namespace proportion_option_B_true_l197_197636

theorem proportion_option_B_true {a b c d : ℚ} (h : a / b = c / d) : 
  (a + c) / c = (b + d) / d := 
by 
  sorry

end proportion_option_B_true_l197_197636


namespace hotel_made_correct_revenue_l197_197965

noncomputable def hotelRevenue : ℕ :=
  let totalRooms := 260
  let doubleRooms := 196
  let singleRoomCost := 35
  let doubleRoomCost := 60
  let singleRooms := totalRooms - doubleRooms
  let revenueSingleRooms := singleRooms * singleRoomCost
  let revenueDoubleRooms := doubleRooms * doubleRoomCost
  revenueSingleRooms + revenueDoubleRooms

theorem hotel_made_correct_revenue :
  hotelRevenue = 14000 := by
  sorry

end hotel_made_correct_revenue_l197_197965


namespace eve_spending_l197_197099

-- Definitions of the conditions
def cost_mitt : ℝ := 14.00
def cost_apron : ℝ := 16.00
def cost_utensils : ℝ := 10.00
def cost_knife : ℝ := 2 * cost_utensils -- Twice the amount of the utensils
def discount_rate : ℝ := 0.25
def num_nieces : ℝ := 3

-- Total cost before the discount for one kit
def total_cost_one_kit : ℝ :=
  cost_mitt + cost_apron + cost_utensils + cost_knife

-- Discount for one kit
def discount_one_kit : ℝ := 
  total_cost_one_kit * discount_rate

-- Discounted price for one kit
def discounted_cost_one_kit : ℝ :=
  total_cost_one_kit - discount_one_kit

-- Total cost for all kits
def total_cost_all_kits : ℝ :=
  num_nieces * discounted_cost_one_kit

-- The theorem statement
theorem eve_spending : total_cost_all_kits = 135.00 :=
by sorry

end eve_spending_l197_197099


namespace m_power_of_prime_no_m_a_k_l197_197003

-- Part (i)
theorem m_power_of_prime (m : ℕ) (p : ℕ) (k : ℕ) (h1 : m ≥ 1) (h2 : Prime p) (h3 : m * (m + 1) = p^k) : m = 1 :=
by sorry

-- Part (ii)
theorem no_m_a_k (m a k : ℕ) (h1 : m ≥ 1) (h2 : a ≥ 1) (h3 : k ≥ 2) (h4 : m * (m + 1) = a^k) : False :=
by sorry

end m_power_of_prime_no_m_a_k_l197_197003


namespace variance_of_data_set_l197_197187

theorem variance_of_data_set :
  let data_set := [2, 3, 4, 5, 6]
  let mean := (2 + 3 + 4 + 5 + 6) / 5
  let variance := (1 / 5 : Real) * ((2 - mean)^2 + (3 - mean)^2 + (4 - mean)^2 + (5 - mean)^2 + (6 - mean)^2)
  variance = 2 :=
by
  sorry

end variance_of_data_set_l197_197187


namespace car_speed_first_hour_l197_197723

theorem car_speed_first_hour (x : ℝ) (h : (79 = (x + 60) / 2)) : x = 98 :=
by {
  sorry
}

end car_speed_first_hour_l197_197723


namespace max_largest_integer_l197_197615

theorem max_largest_integer (A B C D E : ℕ) (h₀ : A ≤ B) (h₁ : B ≤ C) (h₂ : C ≤ D) (h₃ : D ≤ E) 
(h₄ : (A + B + C + D + E) = 225) (h₅ : E - A = 10) : E = 215 :=
sorry

end max_largest_integer_l197_197615


namespace sweet_treats_per_student_l197_197853

theorem sweet_treats_per_student :
  let cookies := 20
  let cupcakes := 25
  let brownies := 35
  let students := 20
  (cookies + cupcakes + brownies) / students = 4 :=
by 
  sorry

end sweet_treats_per_student_l197_197853


namespace ratio_of_volumes_l197_197801

noncomputable def inscribedSphereVolume (s : ℝ) : ℝ := (4 / 3) * Real.pi * (s / 2) ^ 3

noncomputable def cubeVolume (s : ℝ) : ℝ := s ^ 3

theorem ratio_of_volumes (s : ℝ) (h : s > 0) :
  inscribedSphereVolume s / cubeVolume s = Real.pi / 6 :=
by
  sorry

end ratio_of_volumes_l197_197801


namespace find_digit_property_l197_197292

theorem find_digit_property (a x : ℕ) (h : 10 * a + x = a + x + a * x) : x = 9 :=
sorry

end find_digit_property_l197_197292


namespace christmas_day_december_25_l197_197977

-- Define the conditions
def is_thursday (d: ℕ) : Prop := d % 7 = 4
def thanksgiving := 26
def december_christmas := 25

-- Define the problem as a proof problem
theorem christmas_day_december_25 :
  is_thursday (thanksgiving) → thanksgiving = 26 →
  december_christmas = 25 → 
  30 - 26 + 25 = 28 → 
  is_thursday (30 - 26 + 25) :=
by
  intro h_thursday h_thanksgiving h_christmas h_days
  -- skipped proof
  sorry

end christmas_day_december_25_l197_197977


namespace Vasechkin_result_l197_197824

-- Define the operations
def P (x : ℕ) : ℕ := (x / 2 * 7) - 1001
def V (x : ℕ) : ℕ := (x / 8) ^ 2 - 1001

-- Define the proposition
theorem Vasechkin_result (x : ℕ) (h_prime : P x = 7) : V x = 295 := 
by {
  -- Proof is omitted
  sorry
}

end Vasechkin_result_l197_197824


namespace exists_n_for_dvd_ka_pow_n_add_n_l197_197416

theorem exists_n_for_dvd_ka_pow_n_add_n 
  (a k : ℕ) (a_pos : 0 < a) (k_pos : 0 < k) (d : ℕ) (d_pos : 0 < d) :
  ∃ n : ℕ, 0 < n ∧ d ∣ k * (a ^ n) + n :=
by
  sorry

end exists_n_for_dvd_ka_pow_n_add_n_l197_197416


namespace smallest_n_l197_197701

theorem smallest_n (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : n % 9 = 2)
  (h3 : n % 6 = 4) : n = 146 :=
sorry

end smallest_n_l197_197701


namespace total_revenue_proof_l197_197415

-- Define constants for the problem
def original_price_per_case : ℝ := 25
def first_group_customers : ℕ := 8
def first_group_cases_per_customer : ℕ := 3
def first_group_discount_percentage : ℝ := 0.15
def second_group_customers : ℕ := 4
def second_group_cases_per_customer : ℕ := 2
def second_group_discount_percentage : ℝ := 0.10
def third_group_customers : ℕ := 8
def third_group_cases_per_customer : ℕ := 1

-- Calculate the prices after discount
def discounted_price_first_group : ℝ := original_price_per_case * (1 - first_group_discount_percentage)
def discounted_price_second_group : ℝ := original_price_per_case * (1 - second_group_discount_percentage)
def regular_price : ℝ := original_price_per_case

-- Calculate the total revenue
def total_revenue_first_group : ℝ := first_group_customers * first_group_cases_per_customer * discounted_price_first_group
def total_revenue_second_group : ℝ := second_group_customers * second_group_cases_per_customer * discounted_price_second_group
def total_revenue_third_group : ℝ := third_group_customers * third_group_cases_per_customer * regular_price

def total_revenue : ℝ := total_revenue_first_group + total_revenue_second_group + total_revenue_third_group

-- Prove that the total revenue is $890
theorem total_revenue_proof : total_revenue = 890 := by
  sorry

end total_revenue_proof_l197_197415


namespace min_value_reciprocal_sum_l197_197688

theorem min_value_reciprocal_sum (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) : 
  (∃ c, (∀ x y, x > 0 → y > 0 → x + y = 1 → (1/x + 1/y) ≥ c) ∧ (1/a + 1/b = c)) 
:= 
sorry

end min_value_reciprocal_sum_l197_197688


namespace residue_of_neg_2035_mod_47_l197_197303

theorem residue_of_neg_2035_mod_47 : (-2035 : ℤ) % 47 = 33 := 
by
  sorry

end residue_of_neg_2035_mod_47_l197_197303


namespace inverse_proportionality_l197_197431

theorem inverse_proportionality (a b c k a1 a2 b1 b2 c1 c2 : ℝ)
    (h1 : a * b * c = k)
    (h2 : a1 / a2 = 3 / 4)
    (h3 : b1 = 2 * b2)
    (h4 : c1 ≠ 0 ∧ c2 ≠ 0) :
    c1 / c2 = 2 / 3 :=
sorry

end inverse_proportionality_l197_197431


namespace john_running_speed_l197_197374

noncomputable def find_running_speed (x : ℝ) : Prop :=
  (12 / (3 * x + 2) + 8 / x = 2.2)

theorem john_running_speed : ∃ x : ℝ, find_running_speed x ∧ abs (x - 0.47) < 0.01 :=
by
  sorry

end john_running_speed_l197_197374


namespace product_of_roots_of_t_squared_equals_49_l197_197313

theorem product_of_roots_of_t_squared_equals_49 : 
  ∃ t : ℝ, (t^2 = 49) ∧ (t = 7 ∨ t = -7) ∧ (t * (7 + -7)) = -49 := 
by
  sorry

end product_of_roots_of_t_squared_equals_49_l197_197313


namespace avg_speed_in_mph_l197_197750

/-- 
Given conditions:
1. The man travels 10,000 feet due north.
2. He travels 6,000 feet due east in 1/4 less time than he took heading north, traveling at 3 miles per minute.
3. He returns to his starting point by traveling south at 1 mile per minute.
4. He travels back west at the same speed as he went east.
We aim to prove that the average speed for the entire trip is 22.71 miles per hour.
-/
theorem avg_speed_in_mph :
  let distance_north_feet := 10000
  let distance_east_feet := 6000
  let speed_east_miles_per_minute := 3
  let speed_south_miles_per_minute := 1
  let feet_per_mile := 5280
  let distance_north_mil := (distance_north_feet / feet_per_mile : ℝ)
  let distance_east_mil := (distance_east_feet / feet_per_mile : ℝ)
  let time_north_min := distance_north_mil / (1 / 3)
  let time_east_min := time_north_min * 0.75
  let time_south_min := distance_north_mil / speed_south_miles_per_minute
  let time_west_min := time_east_min
  let total_time_hr := (time_north_min + time_east_min + time_south_min + time_west_min) / 60
  let total_distance_miles := 2 * (distance_north_mil + distance_east_mil)
  let avg_speed_mph := total_distance_miles / total_time_hr
  avg_speed_mph = 22.71 := by
sorry

end avg_speed_in_mph_l197_197750


namespace expression_evaluation_l197_197002

variable (x y : ℤ)

theorem expression_evaluation (h₁ : x = -1) (h₂ : y = 1) : 
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 2 :=
by
  rw [h₁, h₂]
  have h₃ : (-1 + 1) * (-1 - 1) - (4 * (-1)^3 * 1 - 8 * (-1) * 1^3) / (2 * (-1) * 1) = (-2) - (-10 / -2) := by sorry
  have h₄ : (-2) - 5 = 2 := by sorry
  sorry

end expression_evaluation_l197_197002


namespace chemistry_marks_l197_197378

theorem chemistry_marks (marks_english : ℕ) (marks_math : ℕ) (marks_physics : ℕ) 
                        (marks_biology : ℕ) (average_marks : ℚ) (marks_chemistry : ℕ) 
                        (h_english : marks_english = 70) 
                        (h_math : marks_math = 60) 
                        (h_physics : marks_physics = 78) 
                        (h_biology : marks_biology = 65) 
                        (h_average : average_marks = 66.6) 
                        (h_total: average_marks * 5 = marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) : 
  marks_chemistry = 60 :=
by sorry

end chemistry_marks_l197_197378


namespace cost_of_tax_free_item_D_l197_197560

theorem cost_of_tax_free_item_D 
  (P_A P_B P_C : ℝ)
  (H1 : 0.945 * P_A + 1.064 * P_B + 1.18 * P_C = 225)
  (H2 : 0.045 * P_A + 0.12 * P_B + 0.18 * P_C = 30) :
  250 - (0.945 * P_A + 1.064 * P_B + 1.18 * P_C) = 25 := 
by
  -- The proof steps would go here.
  sorry

end cost_of_tax_free_item_D_l197_197560


namespace solution_set_of_inequality_l197_197255

theorem solution_set_of_inequality :
  {x : ℝ // (2 < x ∨ x < 2) ∧ x ≠ 3} =
  {x : ℝ // x < 2 ∨ 3 < x } :=
sorry

end solution_set_of_inequality_l197_197255


namespace correct_option_is_A_l197_197475

variable (a b : ℤ)

-- Option A condition
def optionA : Prop := 3 * a^2 * b / b = 3 * a^2

-- Option B condition
def optionB : Prop := a^12 / a^3 = a^4

-- Option C condition
def optionC : Prop := (a + b)^2 = a^2 + b^2

-- Option D condition
def optionD : Prop := (-2 * a^2)^3 = 8 * a^6

theorem correct_option_is_A : 
  optionA a b ∧ ¬optionB a ∧ ¬optionC a b ∧ ¬optionD a :=
by
  sorry

end correct_option_is_A_l197_197475


namespace max_sum_of_solutions_l197_197450

theorem max_sum_of_solutions (x y : ℤ) (h : 3 * x ^ 2 + 5 * y ^ 2 = 345) :
  x + y ≤ 13 :=
sorry

end max_sum_of_solutions_l197_197450


namespace problem_statement_l197_197724

def operation (x y z : ℕ) : ℕ := (2 * x * y + y^2) * z^3

theorem problem_statement : operation 7 (operation 4 5 3) 2 = 24844760 :=
by
  sorry

end problem_statement_l197_197724


namespace find_a_l197_197094

open Set

theorem find_a :
  ∀ (A B : Set ℕ) (a : ℕ),
    A = {1, 2, 3} →
    B = {2, a} →
    A ∪ B = {0, 1, 2, 3} →
    a = 0 :=
by
  intros A B a hA hB hUnion
  rw [hA, hB] at hUnion
  sorry

end find_a_l197_197094


namespace min_value_f_at_0_l197_197864

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem min_value_f_at_0 (a : ℝ) : (∀ x : ℝ, f a 0 ≤ f a x) ↔ 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end min_value_f_at_0_l197_197864


namespace cookies_remaining_in_jar_l197_197878

-- Definition of the conditions
variable (initial_cookies : Nat)

def cookies_taken_by_Lou_Senior := 3 + 1
def cookies_taken_by_Louie_Junior := 7
def total_cookies_taken := cookies_taken_by_Lou_Senior + cookies_taken_by_Louie_Junior

-- Debra's assumption and the proof goal
theorem cookies_remaining_in_jar (half_cookies_removed : total_cookies_taken = initial_cookies / 2) : 
  initial_cookies - total_cookies_taken = 11 := by
  sorry

end cookies_remaining_in_jar_l197_197878


namespace function_periodicity_l197_197959

variable {R : Type*} [Ring R]

def periodic_function (f : R → R) (k : R) : Prop :=
  ∀ x : R, f (x + 4*k) = f x

theorem function_periodicity {f : ℝ → ℝ} {k : ℝ} (h : ∀ x, f (x + k) * (1 - f x) = 1 + f x) (hk : k ≠ 0) : 
  periodic_function f k :=
sorry

end function_periodicity_l197_197959


namespace randy_brother_ate_l197_197383

-- Definitions
def initial_biscuits : ℕ := 32
def biscuits_from_father : ℕ := 13
def biscuits_from_mother : ℕ := 15
def remaining_biscuits : ℕ := 40

-- Theorem to prove
theorem randy_brother_ate : 
  initial_biscuits + biscuits_from_father + biscuits_from_mother - remaining_biscuits = 20 :=
by
  sorry

end randy_brother_ate_l197_197383


namespace vector_solution_l197_197756

theorem vector_solution
  (x y : ℝ)
  (h1 : (2*x - y = 0))
  (h2 : (x^2 + y^2 = 20)) :
  (x = 2 ∧ y = 4) ∨ (x = -2 ∧ y = -4) := 
by
  sorry

end vector_solution_l197_197756


namespace sum_of_squares_l197_197219

theorem sum_of_squares (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 512 * x ^ 3 + 125 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 6410 := 
sorry

end sum_of_squares_l197_197219


namespace remainder_n_l197_197658

-- Definitions for the conditions
/-- m is a positive integer leaving a remainder of 2 when divided by 6 -/
def m (m : ℕ) : Prop := m % 6 = 2

/-- The remainder when m - n is divided by 6 is 5 -/
def mn_remainder (m n : ℕ) : Prop := (m - n) % 6 = 5

-- Theorem statement
theorem remainder_n (m n : ℕ) (h1 : m % 6 = 2) (h2 : (m - n) % 6 = 5) (h3 : m > n) :
  n % 6 = 4 :=
by
  sorry

end remainder_n_l197_197658


namespace circle1_standard_form_circle2_standard_form_l197_197171

-- Define the first circle equation and its corresponding answer in standard form
theorem circle1_standard_form :
  ∀ x y : ℝ, (x^2 + y^2 + 2*x + 4*y - 4 = 0) ↔ ((x + 1)^2 + (y + 2)^2 = 9) :=
by
  intro x y
  sorry

-- Define the second circle equation and its corresponding answer in standard form
theorem circle2_standard_form :
  ∀ x y : ℝ, (3*x^2 + 3*y^2 + 6*x + 3*y - 15 = 0) ↔ ((x + 1)^2 + (y + 1/2)^2 = 25/4) :=
by
  intro x y
  sorry

end circle1_standard_form_circle2_standard_form_l197_197171


namespace correct_operation_l197_197769

variable (a b : ℝ)

theorem correct_operation : 2 * (a - 1) = 2 * a - 2 :=
sorry

end correct_operation_l197_197769


namespace polynomial_evaluation_l197_197019

theorem polynomial_evaluation :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 :=
by
  sorry

end polynomial_evaluation_l197_197019


namespace total_area_of_rug_l197_197657

theorem total_area_of_rug :
  let length_rect := 6
  let width_rect := 4
  let base_parallelogram := 3
  let height_parallelogram := 4
  let area_rect := length_rect * width_rect
  let area_parallelogram := base_parallelogram * height_parallelogram
  let total_area := area_rect + 2 * area_parallelogram
  total_area = 48 := by sorry

end total_area_of_rug_l197_197657


namespace quarterly_production_growth_l197_197436

theorem quarterly_production_growth (P_A P_Q2 : ℕ) (x : ℝ)
  (hA : P_A = 500000)
  (hQ2 : P_Q2 = 1820000) :
  50 + 50 * (1 + x) + 50 * (1 + x)^2 = 182 :=
by 
  sorry

end quarterly_production_growth_l197_197436


namespace tom_steps_l197_197301

theorem tom_steps (matt_rate : ℕ) (tom_extra_rate : ℕ) (matt_steps : ℕ) (tom_rate : ℕ := matt_rate + tom_extra_rate) (time : ℕ := matt_steps / matt_rate)
(H_matt_rate : matt_rate = 20)
(H_tom_extra_rate : tom_extra_rate = 5)
(H_matt_steps : matt_steps = 220) :
  tom_rate * time = 275 :=
by
  -- We start the proof here, but leave it as sorry.
  sorry

end tom_steps_l197_197301


namespace fountain_area_l197_197866

theorem fountain_area (A B D C : ℝ) (h₁ : B - A = 20) (h₂ : D = (A + B) / 2) (h₃ : C - D = 12) :
  ∃ R : ℝ, R^2 = 244 ∧ π * R^2 = 244 * π :=
by
  sorry

end fountain_area_l197_197866


namespace angle_XYZ_of_excircle_circumcircle_incircle_l197_197817

theorem angle_XYZ_of_excircle_circumcircle_incircle 
  (a b c x y z : ℝ) 
  (hA : a = 50)
  (hB : b = 70)
  (hC : c = 60) 
  (triangleABC : a + b + c = 180) 
  (excircle_Omega : Prop) 
  (incircle_Gamma : Prop) 
  (circumcircle_Omega_triangleXYZ : Prop) 
  (X_on_BC : Prop)
  (Y_on_AB : Prop) 
  (Z_on_CA : Prop): 
  x = 115 := 
by 
  sorry

end angle_XYZ_of_excircle_circumcircle_incircle_l197_197817


namespace polynomial_real_root_condition_l197_197067

theorem polynomial_real_root_condition (b : ℝ) :
    (∃ x : ℝ, x^4 + b * x^3 + x^2 + b * x - 1 = 0) ↔ (b ≥ 1 / 2) :=
by sorry

end polynomial_real_root_condition_l197_197067


namespace find_x_l197_197656

theorem find_x (h : 0.60 / x = 6 / 2) : x = 0.20 :=
by
  sorry

end find_x_l197_197656


namespace arithmetic_sequence_sum_l197_197526

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h_arith : ∀ k, S (k + 1) - S k = S 1 - S 0)
  (h_S5 : S 5 = 10) (h_S10 : S 10 = 18) : S 15 = 26 :=
by
  -- Rest of the proof goes here
  sorry

end arithmetic_sequence_sum_l197_197526


namespace simplify_expression_l197_197891

theorem simplify_expression (k : ℤ) : 
  let a := 1
  let b := 3
  (6 * k + 18) / 6 = k + 3 ∧ a / b = 1 / 3 :=
by
  sorry

end simplify_expression_l197_197891


namespace value_of_a_l197_197355

theorem value_of_a (x a : ℤ) (h1 : x = 2) (h2 : 3 * x - a = -x + 7) : a = 1 :=
by
  sorry

end value_of_a_l197_197355


namespace evaluate_fraction_l197_197371

theorem evaluate_fraction : 3 / (2 - 3 / 4) = 12 / 5 := by
  sorry

end evaluate_fraction_l197_197371


namespace max_possible_value_of_k_l197_197242

noncomputable def max_knights_saying_less : Nat :=
  let n := 2015
  let k := n - 2
  k

theorem max_possible_value_of_k : max_knights_saying_less = 2013 :=
by
  sorry

end max_possible_value_of_k_l197_197242


namespace standard_equation_of_circle_l197_197923

theorem standard_equation_of_circle
  (r : ℝ) (h_radius : r = 1)
  (h_center : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (x, y) = (a, b))
  (h_tangent_line : ∃ (a : ℝ), 1 = |4 * a - 3| / 5)
  (h_tangent_x_axis : ∃ (a : ℝ), a = 1) :
  (∃ (a b : ℝ), (x-2)^2 + (y-1)^2 = 1) :=
sorry

end standard_equation_of_circle_l197_197923


namespace sqrt_six_ineq_l197_197327

theorem sqrt_six_ineq : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end sqrt_six_ineq_l197_197327


namespace class3_qualifies_l197_197031

/-- Data structure representing a class's tardiness statistics. -/
structure ClassStats where
  mean : ℕ
  median : ℕ
  variance : ℕ
  mode : Option ℕ -- mode is optional because not all classes might have a unique mode.

def class1 : ClassStats := { mean := 3, median := 3, variance := 0, mode := none }
def class2 : ClassStats := { mean := 2, median := 0, variance := 1, mode := none }
def class3 : ClassStats := { mean := 2, median := 0, variance := 2, mode := none }
def class4 : ClassStats := { mean := 0, median := 2, variance := 0, mode := some 2 }

/-- Predicate to check if a class qualifies for the flag, meaning no more than 5 students tardy each day for 5 consecutive days. -/
def qualifies (cs : ClassStats) : Prop :=
  cs.mean = 2 ∧ cs.variance = 2

theorem class3_qualifies : qualifies class3 :=
by
  sorry

end class3_qualifies_l197_197031


namespace fraction_simplest_form_l197_197289

def fracA (a b : ℤ) : ℤ × ℤ := (|2 * a|, 5 * a^2 * b)
def fracB (a : ℤ) : ℤ × ℤ := (a, a^2 - 2 * a)
def fracC (a b : ℤ) : ℤ × ℤ := (3 * a + b, a + b)
def fracD (a b : ℤ) : ℤ × ℤ := (a^2 - a * b, a^2 - b^2)

theorem fraction_simplest_form (a b : ℤ) : (fracC a b).1 / (fracC a b).2 = (3 * a + b) / (a + b) :=
by sorry

end fraction_simplest_form_l197_197289


namespace discount_percentage_l197_197180

theorem discount_percentage (original_price new_price : ℕ) (h₁ : original_price = 120) (h₂ : new_price = 96) : 
  ((original_price - new_price) * 100 / original_price) = 20 := 
by
  -- sorry is used here to skip the proof
  sorry

end discount_percentage_l197_197180


namespace car_rental_cost_l197_197412

theorem car_rental_cost (D R M P C : ℝ) (hD : D = 5) (hR : R = 30) (hM : M = 500) (hP : P = 0.25) 
(hC : C = (R * D) + (P * M)) : C = 275 :=
by
  rw [hD, hR, hM, hP] at hC
  sorry

end car_rental_cost_l197_197412


namespace smallest_7_digit_number_divisible_by_all_l197_197534

def smallest_7_digit_number : ℕ := 7207200

theorem smallest_7_digit_number_divisible_by_all :
  smallest_7_digit_number >= 1000000 ∧ smallest_7_digit_number < 10000000 ∧
  smallest_7_digit_number % 35 = 0 ∧ 
  smallest_7_digit_number % 112 = 0 ∧ 
  smallest_7_digit_number % 175 = 0 ∧ 
  smallest_7_digit_number % 288 = 0 ∧ 
  smallest_7_digit_number % 429 = 0 ∧ 
  smallest_7_digit_number % 528 = 0 :=
by
  sorry

end smallest_7_digit_number_divisible_by_all_l197_197534


namespace geometric_progression_complex_l197_197886

theorem geometric_progression_complex (a b c m : ℂ) (r : ℂ) (hr : r ≠ 0) 
    (h1 : a = r) (h2 : b = r^2) (h3 : c = r^3) 
    (h4 : a / (1 - b) = m) (h5 : b / (1 - c) = m) (h6 : c / (1 - a) = m) : 
    ∃ m : ℂ, ∀ a b c : ℂ, ∃ r : ℂ, a = r ∧ b = r^2 ∧ c = r^3 
    ∧ r ≠ 0 
    ∧ (a / (1 - b) = m) 
    ∧ (b / (1 - c) = m) 
    ∧ (c / (1 - a) = m) := 
sorry

end geometric_progression_complex_l197_197886


namespace regular_polygon_angle_not_divisible_by_five_l197_197044

theorem regular_polygon_angle_not_divisible_by_five :
  ∃ (n_values : Finset ℕ), n_values.card = 5 ∧
    ∀ n ∈ n_values, 3 ≤ n ∧ n ≤ 15 ∧
      ¬ (∃ k : ℕ, (180 * (n - 2)) / n = 5 * k) := 
by
  sorry

end regular_polygon_angle_not_divisible_by_five_l197_197044


namespace sum_squares_inequality_l197_197684

theorem sum_squares_inequality (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
(h_sum : x + y + z = 3) : 
  1 / (x^2 + y + z) + 1 / (x + y^2 + z) + 1 / (x + y + z^2) ≤ 1 := 
by 
  sorry

end sum_squares_inequality_l197_197684


namespace flower_profit_equation_l197_197819

theorem flower_profit_equation
  (initial_plants : ℕ := 3)
  (initial_profit_per_plant : ℕ := 10)
  (decrease_in_profit_per_additional_plant : ℕ := 1)
  (target_profit_per_pot : ℕ := 40)
  (x : ℕ) :
  (initial_plants + x) * (initial_profit_per_plant - x) = target_profit_per_pot :=
sorry

end flower_profit_equation_l197_197819


namespace sum_of_solutions_l197_197119

theorem sum_of_solutions (x : ℝ) (h : x + (25 / x) = 10) : x = 5 :=
by
  sorry

end sum_of_solutions_l197_197119


namespace find_original_number_l197_197882

theorem find_original_number (x : ℝ) :
  (((x / 2.5) - 10.5) * 0.3 = 5.85) -> x = 75 :=
by
  sorry

end find_original_number_l197_197882


namespace pool_filling_water_amount_l197_197776

theorem pool_filling_water_amount (Tina_pail Tommy_pail Timmy_pail Trudy_pail : ℕ) 
  (h1 : Tina_pail = 4)
  (h2 : Tommy_pail = Tina_pail + 2)
  (h3 : Timmy_pail = 2 * Tommy_pail)
  (h4 : Trudy_pail = (3 * Timmy_pail) / 2)
  (Timmy_trips Trudy_trips Tommy_trips Tina_trips: ℕ)
  (h5 : Timmy_trips = 4)
  (h6 : Trudy_trips = 4)
  (h7 : Tommy_trips = 6)
  (h8 : Tina_trips = 6) :
  Timmy_trips * Timmy_pail + Trudy_trips * Trudy_pail + Tommy_trips * Tommy_pail + Tina_trips * Tina_pail = 180 := by
  sorry

end pool_filling_water_amount_l197_197776


namespace greatest_int_with_gcd_five_l197_197134

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l197_197134


namespace sum_of_min_value_and_input_l197_197563

def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem sum_of_min_value_and_input : 
  let a := -1
  let b := 3 * a - a ^ 3
  a + b = -3 := 
by
  let a := -1
  let b := 3 * a - a ^ 3
  sorry

end sum_of_min_value_and_input_l197_197563


namespace range_of_f_l197_197197

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^x else -Real.log x / Real.log 2

theorem range_of_f : Set.Iic 2 = Set.range f :=
  by sorry

end range_of_f_l197_197197


namespace pq_identity_l197_197754

theorem pq_identity (p q : ℝ) (h1 : p * q = 20) (h2 : p + q = 10) : p^2 + q^2 = 60 :=
sorry

end pq_identity_l197_197754


namespace tan_add_pi_div_three_l197_197585

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l197_197585


namespace necessary_and_sufficient_condition_for_geometric_sequence_l197_197500

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {c : ℝ}

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a_n (n+1) = r * a_n n

theorem necessary_and_sufficient_condition_for_geometric_sequence :
  (∀ n : ℕ, S_n n = 2^n + c) →
  (∀ n : ℕ, a_n n = S_n n - S_n (n-1)) →
  is_geometric_sequence a_n ↔ c = -1 :=
by
  sorry

end necessary_and_sufficient_condition_for_geometric_sequence_l197_197500


namespace similar_triangles_iff_l197_197903

variables {a b c a' b' c' : ℂ}

theorem similar_triangles_iff :
  (∃ (z w : ℂ), a' = a * z + w ∧ b' = b * z + w ∧ c' = c * z + w) ↔
  a' * (b - c) + b' * (c - a) + c' * (a - b) = 0 :=
sorry

end similar_triangles_iff_l197_197903


namespace diff_between_percent_and_fraction_l197_197348

theorem diff_between_percent_and_fraction :
  (0.75 * 800) - ((7 / 8) * 1200) = -450 :=
by
  sorry

end diff_between_percent_and_fraction_l197_197348


namespace seeking_the_cause_from_the_result_means_sufficient_condition_l197_197467

-- Define the necessary entities for the conditions
inductive Condition
| Necessary
| Sufficient
| NecessaryAndSufficient
| NecessaryOrSufficient

-- Define the statement of the proof problem
theorem seeking_the_cause_from_the_result_means_sufficient_condition :
  (seeking_the_cause_from_the_result : Condition) = Condition.Sufficient :=
sorry

end seeking_the_cause_from_the_result_means_sufficient_condition_l197_197467


namespace total_production_l197_197630

variable (x : ℕ) -- total units produced by 4 machines in 6 days
variable (R : ℕ) -- rate of production per machine per day

-- Condition 1: 4 machines can produce x units in 6 days
axiom rate_definition : 4 * R * 6 = x

-- Question: Prove the total amount of product produced by 16 machines in 3 days is 2x
theorem total_production : 16 * R * 3 = 2 * x :=
by 
  sorry

end total_production_l197_197630


namespace cubes_with_even_red_faces_count_l197_197732

def block_dimensions : ℕ × ℕ × ℕ := (6, 4, 2)
def is_painted_red : Prop := true
def total_cubes : ℕ := 48
def cubes_with_even_red_faces : ℕ := 24

theorem cubes_with_even_red_faces_count :
  ∀ (dimensions : ℕ × ℕ × ℕ) (painted_red : Prop) (cubes_count : ℕ), 
  dimensions = block_dimensions → painted_red = is_painted_red → cubes_count = total_cubes → 
  (cubes_with_even_red_faces = 24) :=
by intros dimensions painted_red cubes_count h1 h2 h3; exact sorry

end cubes_with_even_red_faces_count_l197_197732


namespace transfer_deck_l197_197557

-- Define the conditions
variables {k n : ℕ}

-- Assume conditions explicitly
axiom k_gt_1 : k > 1
axiom cards_deck : 2*n = 2*n -- Implicitly states that we have 2n cards

-- Define the problem statement
theorem transfer_deck (k_gt_1 : k > 1) (cards_deck : 2*n = 2*n) : n = k - 1 :=
sorry

end transfer_deck_l197_197557


namespace number_of_meters_sold_l197_197676

-- Define the given conditions
def price_per_meter : ℕ := 436 -- in kopecks
def total_revenue_end : ℕ := 728 -- in kopecks
def max_total_revenue : ℕ := 50000 -- in kopecks

-- State the problem formally in Lean 4
theorem number_of_meters_sold (x : ℕ) :
  price_per_meter * x ≡ total_revenue_end [MOD 1000] ∧
  price_per_meter * x ≤ max_total_revenue →
  x = 98 :=
sorry

end number_of_meters_sold_l197_197676


namespace total_files_deleted_l197_197106

theorem total_files_deleted 
  (initial_files : ℕ) (initial_apps : ℕ)
  (deleted_files1 : ℕ) (deleted_apps1 : ℕ)
  (added_files1 : ℕ) (added_apps1 : ℕ)
  (deleted_files2 : ℕ) (deleted_apps2 : ℕ)
  (added_files2 : ℕ) (added_apps2 : ℕ)
  (final_files : ℕ) (final_apps : ℕ)
  (h_initial_files : initial_files = 24)
  (h_initial_apps : initial_apps = 13)
  (h_deleted_files1 : deleted_files1 = 5)
  (h_deleted_apps1 : deleted_apps1 = 3)
  (h_added_files1 : added_files1 = 7)
  (h_added_apps1 : added_apps1 = 4)
  (h_deleted_files2 : deleted_files2 = 10)
  (h_deleted_apps2 : deleted_apps2 = 4)
  (h_added_files2 : added_files2 = 5)
  (h_added_apps2 : added_apps2 = 7)
  (h_final_files : final_files = 21)
  (h_final_apps : final_apps = 17) :
  (deleted_files1 + deleted_files2 = 15) := 
by
  sorry

end total_files_deleted_l197_197106


namespace union_complement_eq_l197_197430

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}

theorem union_complement_eq : M ∪ (U \ N) = {0, 1, 2} := by
  sorry

end union_complement_eq_l197_197430


namespace find_balanced_grid_pairs_l197_197440

-- Define a balanced grid condition
def is_balanced_grid (m n : ℕ) (grid : ℕ → ℕ → Prop) : Prop :=
  ∀ i j, i < m → j < n →
    (∀ k, k < m → grid i k = grid i j) ∧ (∀ l, l < n → grid l j = grid i j)

-- Main theorem statement
theorem find_balanced_grid_pairs (m n : ℕ) :
  (∃ grid, is_balanced_grid m n grid) ↔ (m = n ∨ m = n / 2 ∨ n = 2 * m) :=
by
  sorry

end find_balanced_grid_pairs_l197_197440


namespace perimeter_of_triangle_XYZ_l197_197691

/-- 
  Given the inscribed circle of triangle XYZ is tangent to XY at P,
  its radius is 15, XP = 30, and PY = 36, then the perimeter of 
  triangle XYZ is 83.4.
-/
theorem perimeter_of_triangle_XYZ :
  ∀ (XYZ : Type) (P : XYZ) (radius : ℝ) (XP PY perimeter : ℝ),
    radius = 15 → 
    XP = 30 → 
    PY = 36 →
    perimeter = 83.4 :=
by 
  intros XYZ P radius XP PY perimeter h_radius h_XP h_PY
  sorry

end perimeter_of_triangle_XYZ_l197_197691


namespace third_place_prize_is_120_l197_197831

noncomputable def prize_for_third_place (total_prize : ℕ) (first_place_prize : ℕ) (second_place_prize : ℕ) (prize_per_novel : ℕ) (num_novels_receiving_prize : ℕ) : ℕ :=
  let remaining_prize := total_prize - first_place_prize - second_place_prize
  let total_other_prizes := num_novels_receiving_prize * prize_per_novel
  remaining_prize - total_other_prizes

theorem third_place_prize_is_120 : prize_for_third_place 800 200 150 22 15 = 120 := by
  sorry

end third_place_prize_is_120_l197_197831


namespace total_lobster_pounds_l197_197822

variable (lobster_other_harbor1 : ℕ)
variable (lobster_other_harbor2 : ℕ)
variable (lobster_hooper_bay : ℕ)

-- Conditions
axiom h_eq : lobster_hooper_bay = 2 * (lobster_other_harbor1 + lobster_other_harbor2)
axiom other_harbors_eq : lobster_other_harbor1 = 80 ∧ lobster_other_harbor2 = 80

-- Proof statement
theorem total_lobster_pounds : 
  lobster_other_harbor1 + lobster_other_harbor2 + lobster_hooper_bay = 480 :=
by
  sorry

end total_lobster_pounds_l197_197822


namespace greatest_b_max_b_value_l197_197144

theorem greatest_b (b y : ℤ) (h : b > 0) (hy : y^2 + b*y = -21) : b ≤ 22 :=
sorry

theorem max_b_value : ∃ b : ℤ, (∀ y : ℤ, y^2 + b*y = -21 → b > 0) ∧ (b = 22) :=
sorry

end greatest_b_max_b_value_l197_197144


namespace marbles_problem_l197_197295

theorem marbles_problem (p : ℕ) (m n r : ℕ) 
(hp : Nat.Prime p) 
(h1 : p = 2017)
(h2 : N = p^m * n)
(h3 : ¬ p ∣ n)
(h4 : r = n % p) 
(h N : ∀ (N : ℕ), N = 3 * p * 632 - 1)
: p * m + r = 3913 := 
sorry

end marbles_problem_l197_197295


namespace coeff_x3_in_x_mul_1_add_x_pow_6_l197_197088

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem coeff_x3_in_x_mul_1_add_x_pow_6 :
  ∀ x : ℕ, (∃ c : ℕ, c * x^3 = x * (1 + x)^6 ∧ c = 15) :=
by
  sorry

end coeff_x3_in_x_mul_1_add_x_pow_6_l197_197088


namespace find_subtracted_number_l197_197922

theorem find_subtracted_number (x y : ℕ) (h1 : 6 * x - 5 * x = 5) (h2 : (30 - y) * 4 = (25 - y) * 5) : y = 5 :=
sorry

end find_subtracted_number_l197_197922


namespace circle_radius_increase_l197_197401

-- Defining the problem conditions and the resulting proof
theorem circle_radius_increase (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : r = n * (Real.sqrt 3 - 1) / 2 :=
sorry  -- Proof is left as an exercise

end circle_radius_increase_l197_197401


namespace smallest_n_modulo_l197_197092

theorem smallest_n_modulo (
  n : ℕ
) (h1 : 17 * n ≡ 5678 [MOD 11]) : n = 4 :=
by sorry

end smallest_n_modulo_l197_197092


namespace complement_intersection_l197_197844

open Set

namespace UniversalSetProof

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection :
  (U \ A) ∩ B = {4, 5} :=
by
  sorry

end UniversalSetProof

end complement_intersection_l197_197844


namespace yellow_balls_in_bag_l197_197517

open Classical

theorem yellow_balls_in_bag (Y : ℕ) (hY1 : (Y/(Y+2): ℝ) * ((Y-1)/(Y+1): ℝ) = 0.5) : Y = 5 := by
  sorry

end yellow_balls_in_bag_l197_197517


namespace age_ratio_l197_197218

theorem age_ratio (S M : ℕ) (h₁ : M = S + 35) (h₂ : S = 33) : 
  (M + 2) / (S + 2) = 2 :=
by
  -- proof goes here
  sorry

end age_ratio_l197_197218


namespace component_unqualified_l197_197469

/-- 
    The specified diameter range for a component is within [19.98, 20.02].
    The measured diameter of the component is 19.9.
    Prove that the component is unqualified.
-/
def is_unqualified (diameter_measured : ℝ) : Prop :=
    diameter_measured < 19.98 ∨ diameter_measured > 20.02

theorem component_unqualified : is_unqualified 19.9 :=
by
  -- Proof goes here
  sorry

end component_unqualified_l197_197469


namespace john_spent_at_candy_store_l197_197344

-- Definition of the conditions
def allowance : ℚ := 1.50
def arcade_spent : ℚ := (3 / 5) * allowance
def remaining_after_arcade : ℚ := allowance - arcade_spent
def toy_store_spent : ℚ := (1 / 3) * remaining_after_arcade

-- Statement and Proof of the Problem
theorem john_spent_at_candy_store : (remaining_after_arcade - toy_store_spent) = 0.40 :=
by
  -- Proof is left as an exercise
  sorry

end john_spent_at_candy_store_l197_197344


namespace city_G_has_highest_percentage_increase_l197_197611

-- Define the population data as constants.
def population_1990_F : ℕ := 50
def population_2000_F : ℕ := 60
def population_1990_G : ℕ := 60
def population_2000_G : ℕ := 80
def population_1990_H : ℕ := 90
def population_2000_H : ℕ := 110
def population_1990_I : ℕ := 120
def population_2000_I : ℕ := 150
def population_1990_J : ℕ := 150
def population_2000_J : ℕ := 190

-- Define the function that calculates the percentage increase.
def percentage_increase (pop_1990 pop_2000 : ℕ) : ℚ :=
  (pop_2000 : ℚ) / (pop_1990 : ℚ)

-- Calculate the percentage increases for each city.
def percentage_increase_F := percentage_increase population_1990_F population_2000_F
def percentage_increase_G := percentage_increase population_1990_G population_2000_G
def percentage_increase_H := percentage_increase population_1990_H population_2000_H
def percentage_increase_I := percentage_increase population_1990_I population_2000_I
def percentage_increase_J := percentage_increase population_1990_J population_2000_J

-- Prove that City G has the greatest percentage increase.
theorem city_G_has_highest_percentage_increase :
  percentage_increase_G > percentage_increase_F ∧ 
  percentage_increase_G > percentage_increase_H ∧
  percentage_increase_G > percentage_increase_I ∧
  percentage_increase_G > percentage_increase_J :=
by sorry

end city_G_has_highest_percentage_increase_l197_197611


namespace measure_of_angle_B_l197_197306

noncomputable def angle_opposite_side (a b c : ℝ) (h : (c^2)/(a+b) + (a^2)/(b+c) = b) : ℝ :=
  if h : (c^2)/(a+b) + (a^2)/(b+c) = b then 60 else 0

theorem measure_of_angle_B {a b c : ℝ} (h : (c^2)/(a+b) + (a^2)/(b+c) = b) : 
  angle_opposite_side a b c h = 60 :=
by
  sorry

end measure_of_angle_B_l197_197306


namespace wallpaper_job_completion_l197_197537

theorem wallpaper_job_completion (x : ℝ) (y : ℝ) 
  (h1 : ∀ a b : ℝ, (a = 1.5) → (7/x + (7-a)/(x-3) = 1)) 
  (h2 : y = x - 3) 
  (h3 : x - y = 3) : 
  (x = 14) ∧ (y = 11) :=
sorry

end wallpaper_job_completion_l197_197537


namespace pens_sales_consistency_books_left_indeterminate_l197_197714

-- The initial conditions
def initial_pens : ℕ := 42
def initial_books : ℕ := 143
def pens_left : ℕ := 19
def pens_sold : ℕ := 23

-- Prove the consistency of the number of pens sold
theorem pens_sales_consistency : initial_pens - pens_left = pens_sold := by
  sorry

-- Assert that the number of books left is indeterminate based on provided conditions
theorem books_left_indeterminate : ∃ b_left : ℕ, b_left ≤ initial_books ∧
    ∀ n_books_sold : ℕ, n_books_sold > 0 → b_left = initial_books - n_books_sold := by
  sorry

end pens_sales_consistency_books_left_indeterminate_l197_197714


namespace min_value_2a_3b_equality_case_l197_197224

theorem min_value_2a_3b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 3 / b = 1) : 
  2 * a + 3 * b ≥ 25 :=
sorry

theorem equality_case (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 3 / b = 1) :
  (a = 5) ∧ (b = 5) → 2 * a + 3 * b = 25 :=
sorry

end min_value_2a_3b_equality_case_l197_197224


namespace smallest_N_such_that_N_and_N_squared_end_in_same_three_digits_l197_197566

theorem smallest_N_such_that_N_and_N_squared_end_in_same_three_digits :
  ∃ N : ℕ, (N > 0) ∧ (N % 1000 = (N^2 % 1000)) ∧ (1 ≤ N / 100 % 10) ∧ (N = 376) :=
by
  sorry

end smallest_N_such_that_N_and_N_squared_end_in_same_three_digits_l197_197566


namespace unique_solution_of_equation_l197_197452

theorem unique_solution_of_equation :
  ∃! (x : Fin 8 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + 
                                  (x 2 - x 3)^2 + (x 3 - x 4)^2 + 
                                  (x 4 - x 5)^2 + (x 5 - x 6)^2 + 
                                  (x 6 - x 7)^2 + (x 7)^2 = 1 / 9 :=
sorry

end unique_solution_of_equation_l197_197452


namespace polynomial_has_roots_l197_197709

theorem polynomial_has_roots :
  ∃ x : ℝ, x ∈ [-4, -3, -1, 2] ∧ (x^4 + 6 * x^3 + 7 * x^2 - 14 * x - 12 = 0) :=
by
  sorry

end polynomial_has_roots_l197_197709


namespace product_of_square_roots_l197_197510
-- Importing the necessary Lean library

-- Declare the mathematical problem in Lean 4
theorem product_of_square_roots (x : ℝ) (hx : 0 ≤ x) :
  Real.sqrt (40 * x) * Real.sqrt (5 * x) * Real.sqrt (18 * x) = 60 * x * Real.sqrt (3 * x) :=
by
  sorry

end product_of_square_roots_l197_197510


namespace first_player_has_winning_strategy_l197_197606

-- Define the initial heap sizes and rules of the game.
def initial_heaps : List Nat := [38, 45, 61, 70]

-- Define a function that checks using the rules whether the first player has a winning strategy given the initial heap sizes.
def first_player_wins : Bool :=
  -- placeholder for the actual winning strategy check logic
  sorry

-- Theorem statement referring to the equivalency proof problem where player one is established to have the winning strategy.
theorem first_player_has_winning_strategy : first_player_wins = true :=
  sorry

end first_player_has_winning_strategy_l197_197606


namespace find_number_l197_197767

def problem (x : ℝ) : Prop :=
  0.25 * x = 130 + 190

theorem find_number (x : ℝ) (h : problem x) : x = 1280 :=
by 
  sorry

end find_number_l197_197767


namespace each_friend_received_12_candies_l197_197346

-- Define the number of friends and total candies given
def num_friends : ℕ := 35
def total_candies : ℕ := 420

-- Define the number of candies each friend received
def candies_per_friend : ℕ := total_candies / num_friends

theorem each_friend_received_12_candies :
  candies_per_friend = 12 :=
by
  -- Skip the proof
  sorry

end each_friend_received_12_candies_l197_197346


namespace relationship_among_abc_l197_197985

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := (2 / 3) ^ (2 / 5)
noncomputable def c : ℝ := Real.log (1 / 5) / Real.log (1 / 3)

theorem relationship_among_abc : c > b ∧ b > a :=
by
  have h1 : a = (1 / 3) ^ (2 / 5) := rfl
  have h2 : b = (2 / 3) ^ (2 / 5) := rfl
  have h3 : c = Real.log (1 / 5) / Real.log (1 / 3) := rfl
  sorry

end relationship_among_abc_l197_197985


namespace chess_tournament_games_l197_197964

theorem chess_tournament_games (n : ℕ) (h : 2 * 404 = n * (n - 4)) : False :=
by
  sorry

end chess_tournament_games_l197_197964


namespace hunter_3_proposal_l197_197185

theorem hunter_3_proposal {hunter1_coins hunter2_coins hunter3_coins : ℕ} :
  hunter3_coins = 99 ∧ hunter1_coins = 1 ∧ (hunter1_coins + hunter3_coins + hunter2_coins = 100) :=
  sorry

end hunter_3_proposal_l197_197185


namespace projectile_height_l197_197325

theorem projectile_height (t : ℝ) (h : (-16 * t^2 + 80 * t = 100)) : t = 2.5 :=
sorry

end projectile_height_l197_197325


namespace smallest_possible_recording_l197_197229

theorem smallest_possible_recording :
  ∃ (A B C : ℤ), 
      (0 ≤ A ∧ A ≤ 10) ∧ 
      (0 ≤ B ∧ B ≤ 10) ∧ 
      (0 ≤ C ∧ C ≤ 10) ∧ 
      (A + B + C = 12) ∧ 
      (A + B + C) % 5 = 0 ∧ 
      A = 0 :=
by
  sorry

end smallest_possible_recording_l197_197229


namespace max_a_satisfies_no_lattice_points_l197_197291

-- Define the conditions
def no_lattice_points (m : ℚ) (x_upper : ℕ) :=
  ∀ x : ℕ, 0 < x ∧ x ≤ x_upper → ¬∃ y : ℤ, y = m * x + 3

-- Final statement we need to prove
theorem max_a_satisfies_no_lattice_points :
  ∃ a : ℚ, a = 51 / 151 ∧ ∀ m : ℚ, 1 / 3 < m → m < a → no_lattice_points m 150 :=
sorry

end max_a_satisfies_no_lattice_points_l197_197291


namespace circle_m_range_l197_197005

theorem circle_m_range (m : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 - 2 * x + 6 * y + m = 0 → m < 10) :=
sorry

end circle_m_range_l197_197005


namespace min_absolute_sum_value_l197_197124

def absolute_sum (x : ℝ) : ℝ :=
  abs (x + 3) + abs (x + 6) + abs (x + 7)

theorem min_absolute_sum_value : ∃ x, absolute_sum x = 4 :=
sorry

end min_absolute_sum_value_l197_197124


namespace cindy_envelopes_l197_197433

theorem cindy_envelopes (h₁ : ℕ := 4) (h₂ : ℕ := 7) (h₃ : ℕ := 5) (h₄ : ℕ := 10) (h₅ : ℕ := 3) (initial : ℕ := 137) :
  initial - (h₁ + h₂ + h₃ + h₄ + h₅) = 108 :=
by
  sorry

end cindy_envelopes_l197_197433


namespace domain_of_g_l197_197850

def f : ℝ → ℝ := sorry

theorem domain_of_g 
  (hf_dom : ∀ x, -2 ≤ x ∧ x ≤ 4 → f x = f x) : 
  ∀ x, -2 ≤ x ∧ x ≤ 2 ↔ (∃ y, y = f x + f (-x)) := 
by {
  sorry
}

end domain_of_g_l197_197850


namespace function_inequality_l197_197628

noncomputable def f : ℝ → ℝ
| x => if x < 1 then (x + 1)^2 else 4 - Real.sqrt (x - 1)

theorem function_inequality : 
  {x : ℝ | f x ≥ x} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 10} :=
by
  sorry

end function_inequality_l197_197628


namespace root_proof_l197_197946

noncomputable def p : ℝ := (-5 + Real.sqrt 21) / 2
noncomputable def q : ℝ := (-5 - Real.sqrt 21) / 2

theorem root_proof :
  (∃ (p q : ℝ), (∀ x : ℝ, x^3 + 6 * x^2 + 6 * x + 1 = 0 → (x = p ∨ x = q ∨ x = -1)) ∧ 
                 ((p = (-5 + Real.sqrt 21) / 2) ∧ (q = (-5 - Real.sqrt 21) / 2))) →
  (p / q + q / p = 23) :=
by
  sorry

end root_proof_l197_197946


namespace range_of_a_l197_197835

variable {x a : ℝ}

theorem range_of_a (h1 : 2 * x - a < 0)
                   (h2 : 1 - 2 * x ≥ 7)
                   (h3 : ∀ x, x ≤ -3) : ∀ a, a > -6 :=
by
  sorry

end range_of_a_l197_197835


namespace mountain_bike_cost_l197_197631

theorem mountain_bike_cost (savings : ℕ) (lawns : ℕ) (lawn_rate : ℕ) (newspapers : ℕ) (paper_rate : ℕ) (dogs : ℕ) (dog_rate : ℕ) (remaining : ℕ) (total_earned : ℕ) (total_before_purchase : ℕ) (cost : ℕ) : 
  savings = 1500 ∧ lawns = 20 ∧ lawn_rate = 20 ∧ newspapers = 600 ∧ paper_rate = 40 ∧ dogs = 24 ∧ dog_rate = 15 ∧ remaining = 155 ∧ 
  total_earned = (lawns * lawn_rate) + (newspapers * paper_rate / 100) + (dogs * dog_rate) ∧
  total_before_purchase = savings + total_earned ∧
  cost = total_before_purchase - remaining →
  cost = 2345 := by
  sorry

end mountain_bike_cost_l197_197631


namespace maximize_fraction_l197_197118

theorem maximize_fraction (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 :=
sorry

end maximize_fraction_l197_197118


namespace ArithmeticSequenceSum_l197_197210

theorem ArithmeticSequenceSum (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 + a 2 = 10) 
  (h2 : a 4 = a 3 + 2)
  (h3 : ∀ n : ℕ, a n = a 1 + (n - 1) * d) :
  a 3 + a 4 = 18 :=
by
  sorry

end ArithmeticSequenceSum_l197_197210


namespace sample_size_calculation_l197_197414

/--
A factory produces three different models of products: A, B, and C. The ratio of their quantities is 2:3:5.
Using stratified sampling, a sample of size n is drawn, and it contains 16 units of model A.
We need to prove that the sample size n is 80.
-/
theorem sample_size_calculation
  (k : ℕ)
  (hk : 2 * k = 16)
  (n : ℕ)
  (hn : n = (2 + 3 + 5) * k) :
  n = 80 :=
by
  sorry

end sample_size_calculation_l197_197414


namespace sum_of_35_consecutive_squares_div_by_35_l197_197603

def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_35_consecutive_squares_div_by_35 (n : ℕ) :
  (sum_of_squares (n + 35) - sum_of_squares n) % 35 = 0 :=
by
  sorry

end sum_of_35_consecutive_squares_div_by_35_l197_197603


namespace fourth_root_eq_solution_l197_197804

theorem fourth_root_eq_solution (x : ℝ) (h : Real.sqrt (Real.sqrt x) = 16 / (8 - Real.sqrt (Real.sqrt x))) : x = 256 := by
  sorry

end fourth_root_eq_solution_l197_197804


namespace cylinder_height_l197_197209

theorem cylinder_height (h : ℝ)
  (circumference : ℝ)
  (rectangle_diagonal : ℝ)
  (C_eq : circumference = 12)
  (d_eq : rectangle_diagonal = 20) :
  h = 16 :=
by
  -- We derive the result based on the given conditions and calculations
  sorry -- Skipping the proof part

end cylinder_height_l197_197209


namespace find_m_l197_197271

theorem find_m {x : ℝ} (m : ℝ) (h : ∀ x, (0 < x ∧ x < 2) ↔ (-1/2 * x^2 + 2 * x > m * x)) : m = 1 :=
sorry

end find_m_l197_197271


namespace zeroes_y_minus_a_l197_197060

open Real

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then |2 ^ x - 1| else 3 / (x - 1)

theorem zeroes_y_minus_a (a : ℝ) : (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a) → (0 < a ∧ a < 1) :=
sorry

end zeroes_y_minus_a_l197_197060


namespace amount_per_friend_l197_197860

-- Definitions based on conditions
def cost_of_erasers : ℝ := 5 * 200
def cost_of_pencils : ℝ := 7 * 800
def total_cost : ℝ := cost_of_erasers + cost_of_pencils
def number_of_friends : ℝ := 4

-- The proof statement
theorem amount_per_friend : (total_cost / number_of_friends) = 1650 := by
  sorry

end amount_per_friend_l197_197860


namespace sum_of_values_of_z_l197_197837

def f (x : ℝ) := x^2 - 2*x + 3

theorem sum_of_values_of_z (z : ℝ) (h : f (5 * z) = 7) : z = 2 / 25 :=
sorry

end sum_of_values_of_z_l197_197837


namespace geometric_sequence_sum_twenty_terms_l197_197140

noncomputable def geom_seq_sum : ℕ → ℕ → ℕ := λ a r =>
  if r = 1 then a * (1 + 20 - 1) else a * ((1 - r^20) / (1 - r))

theorem geometric_sequence_sum_twenty_terms (a₁ q : ℕ) (h1 : a₁ * (q + 2) = 4) (h2 : (a₃:ℕ) * (q ^ 4) = (a₁ : ℕ) * (q ^ 4)) :
  geom_seq_sum a₁ q = 2^20 - 1 :=
sorry

end geometric_sequence_sum_twenty_terms_l197_197140


namespace find_n_l197_197275

noncomputable def parabola_focus : ℝ × ℝ :=
  (2, 0)

noncomputable def hyperbola_focus (n : ℝ) : ℝ × ℝ :=
  (Real.sqrt (3 + n), 0)

theorem find_n (n : ℝ) : hyperbola_focus n = parabola_focus → n = 1 :=
by
  sorry

end find_n_l197_197275


namespace apples_pie_calculation_l197_197360

-- Defining the conditions
def total_apples : ℕ := 34
def non_ripe_apples : ℕ := 6
def apples_per_pie : ℕ := 4 

-- Stating the problem
theorem apples_pie_calculation : (total_apples - non_ripe_apples) / apples_per_pie = 7 := by
  -- Proof would go here. For the structure of the task, we use sorry.
  sorry

end apples_pie_calculation_l197_197360


namespace simplify_and_multiply_l197_197111

theorem simplify_and_multiply :
  let a := 3
  let b := 17
  let d1 := 504
  let d2 := 72
  let m := 5
  let n := 7
  let fraction1 := a / d1
  let fraction2 := b / d2
  ((fraction1 - (b * n / (d2 * n))) * (m / n)) = (-145 / 882) :=
by
  sorry

end simplify_and_multiply_l197_197111


namespace race_result_l197_197598

theorem race_result
    (distance_race : ℕ)
    (distance_diff : ℕ)
    (distance_second_start_diff : ℕ)
    (speed_xm speed_xl : ℕ)
    (h1 : distance_race = 100)
    (h2 : distance_diff = 20)
    (h3 : distance_second_start_diff = 20)
    (xm_wins_first_race : speed_xm * distance_race >= speed_xl * (distance_race - distance_diff)) :
    speed_xm * (distance_race + distance_second_start_diff) >= speed_xl * (distance_race + distance_diff) :=
by
  sorry

end race_result_l197_197598


namespace blocks_added_l197_197856

theorem blocks_added (original_blocks new_blocks added_blocks : ℕ) 
  (h1 : original_blocks = 35) 
  (h2 : new_blocks = 65) 
  (h3 : new_blocks = original_blocks + added_blocks) : 
  added_blocks = 30 :=
by
  -- We use the given conditions to prove the statement
  sorry

end blocks_added_l197_197856


namespace calculate_savings_l197_197885

noncomputable def monthly_salary : ℕ := 10000
noncomputable def spent_on_food (S : ℕ) : ℕ := (40 * S) / 100
noncomputable def spent_on_rent (S : ℕ) : ℕ := (20 * S) / 100
noncomputable def spent_on_entertainment (S : ℕ) : ℕ := (10 * S) / 100
noncomputable def spent_on_conveyance (S : ℕ) : ℕ := (10 * S) / 100
noncomputable def total_spent (S : ℕ) : ℕ := spent_on_food S + spent_on_rent S + spent_on_entertainment S + spent_on_conveyance S
noncomputable def amount_saved (S : ℕ) : ℕ := S - total_spent S

theorem calculate_savings : amount_saved monthly_salary = 2000 :=
by
  sorry

end calculate_savings_l197_197885


namespace four_digit_numbers_sum_30_l197_197991

-- Definitions of the variables and constraints
def valid_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

-- The main statement we aim to prove
theorem four_digit_numbers_sum_30 : 
  ∃ (count : ℕ), 
  count = 20 ∧ 
  ∃ (a b c d : ℕ), 
  (1 ≤ a ∧ valid_digit a) ∧ 
  (valid_digit b) ∧ 
  (valid_digit c) ∧ 
  (valid_digit d) ∧ 
  a + b + c + d = 30 := sorry

end four_digit_numbers_sum_30_l197_197991


namespace speed_downstream_l197_197126

variables (V_m V_s V_u V_d : ℕ)
variables (h1 : V_u = 12)
variables (h2 : V_m = 25)
variables (h3 : V_u = V_m - V_s)

theorem speed_downstream (h1 : V_u = 12) (h2 : V_m = 25) (h3 : V_u = V_m - V_s) :
  V_d = V_m + V_s :=
by
  -- The proof goes here
  sorry

end speed_downstream_l197_197126


namespace liars_positions_l197_197549

structure Islander :=
  (position : Nat)
  (statement : String)

-- Define our islanders
def A : Islander := { position := 1, statement := "My closest tribesman in this line is 3 meters away from me." }
def D : Islander := { position := 4, statement := "My closest tribesman in this line is 1 meter away from me." }
def E : Islander := { position := 5, statement := "My closest tribesman in this line is 2 meters away from me." }

-- Define the other islanders with dummy statements
def B : Islander := { position := 2, statement := "" }
def C : Islander := { position := 3, statement := "" }
def F : Islander := { position := 6, statement := "" }

-- Define the main theorem
theorem liars_positions (knights_count : Nat) (liars_count : Nat) (is_knight : Islander → Bool)
  (is_lair : Islander → Bool) : 
  ( ∀ x, is_knight x ↔ ¬is_lair x ) → -- Knight and liar are mutually exclusive
  knights_count = 3 → 
  liars_count = 3 →
  is_knight A = false → 
  is_knight D = false → 
  is_knight E = false → 
  is_lair A = true ∧
  is_lair D = true ∧
  is_lair E = true := by
  sorry

end liars_positions_l197_197549


namespace proof_problem_l197_197895

theorem proof_problem
  (n : ℕ)
  (h : n = 16^3018) :
  n / 8 = 2^9032 := by
  sorry

end proof_problem_l197_197895


namespace quadratic_form_l197_197078

-- Define the constants b and c based on the problem conditions
def b : ℤ := 900
def c : ℤ := -807300

-- Create a statement that represents the proof goal
theorem quadratic_form (c_eq : c = -807300) (b_eq : b = 900) : c / b = -897 :=
by
  sorry

end quadratic_form_l197_197078


namespace janice_purchase_l197_197042

theorem janice_purchase (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 30 * a + 200 * b + 300 * c = 3000) : a = 20 :=
sorry

end janice_purchase_l197_197042


namespace largest_neg_integer_solution_l197_197595

theorem largest_neg_integer_solution 
  (x : ℤ) 
  (h : 34 * x + 6 ≡ 2 [ZMOD 20]) : 
  x = -6 := 
sorry

end largest_neg_integer_solution_l197_197595


namespace eleven_step_paths_l197_197053

def H : (ℕ × ℕ) := (0, 0)
def K : (ℕ × ℕ) := (4, 3)
def J : (ℕ × ℕ) := (6, 5)

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem eleven_step_paths (H K J : (ℕ × ℕ)) (H_coords : H = (0, 0)) (K_coords : K = (4, 3)) (J_coords : J = (6, 5)) : 
  (binomial 7 4) * (binomial 4 2) = 210 := by 
  sorry

end eleven_step_paths_l197_197053


namespace calc_difference_of_squares_l197_197080

theorem calc_difference_of_squares :
  625^2 - 375^2 = 250000 :=
by sorry

end calc_difference_of_squares_l197_197080


namespace sufficient_but_not_necessary_condition_l197_197551

theorem sufficient_but_not_necessary_condition (b : ℝ) :
  (∀ x : ℝ, b * x^2 - b * x + 1 > 0) ↔ (b = 0 ∨ (0 < b ∧ b < 4)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l197_197551


namespace height_of_tank_A_l197_197213

theorem height_of_tank_A (C_A C_B h_B : ℝ) (capacity_ratio : ℝ) :
  C_A = 8 → C_B = 10 → h_B = 8 → capacity_ratio = 0.4800000000000001 →
  ∃ h_A : ℝ, h_A = 6 := by
  intros hCA hCB hHB hCR
  sorry

end height_of_tank_A_l197_197213


namespace inequality_proof_l197_197703

variable (a b : ℝ)

theorem inequality_proof (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 :=
  by
    sorry

end inequality_proof_l197_197703


namespace unique_B_for_A47B_divisible_by_7_l197_197687

-- Define the conditions
def A : ℕ := 4

-- Define the main proof problem statement
theorem unique_B_for_A47B_divisible_by_7 : 
  ∃! B : ℕ, B ≤ 9 ∧ (100 * A + 70 + B) % 7 = 0 :=
        sorry

end unique_B_for_A47B_divisible_by_7_l197_197687


namespace dart_probability_l197_197009

noncomputable def area_hexagon (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

noncomputable def area_circle (s : ℝ) : ℝ := Real.pi * s^2

noncomputable def probability (s : ℝ) : ℝ := 
  (area_circle s) / (area_hexagon s)

theorem dart_probability (s : ℝ) (hs : s > 0) :
  probability s = (2 * Real.pi) / (3 * Real.sqrt 3) :=
by
  sorry

end dart_probability_l197_197009


namespace pos_int_divides_l197_197482

theorem pos_int_divides (n : ℕ) (h₀ : 0 < n) (h₁ : (n - 1) ∣ (n^3 + 4)) : n = 2 ∨ n = 6 :=
by sorry

end pos_int_divides_l197_197482


namespace proof_problem_l197_197967

theorem proof_problem (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x^2 + y^2 + z^2 = 1) :
  1 ≤ (x / (1 + y * z)) + (y / (1 + z * x)) + (z / (1 + x * y)) ∧ 
  (x / (1 + y * z)) + (y / (1 + z * x)) + (z / (1 + x * y)) ≤ Real.sqrt 2 :=
by
  sorry

end proof_problem_l197_197967


namespace smallest_positive_period_intervals_monotonic_increase_max_min_values_l197_197933

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2

theorem smallest_positive_period (x : ℝ) : (f (x + π)) = f x :=
sorry

theorem intervals_monotonic_increase (k : ℤ) (x : ℝ) : (k * π - π/3) ≤ x ∧ x ≤ (k * π + π/6) → ∃ a b : ℝ, a < b ∧ ∀ x : ℝ, (a ≤ x ∧ x ≤ b) →
  (f x < f (x + 1)) :=
sorry

theorem max_min_values (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ π/4) : (∃ y : ℝ, y = max (f 0) (f (π/6)) ∧ y = 1) ∧ (∃ z : ℝ, z = min (f 0) (f (π/6)) ∧ z = 0) :=
sorry

end smallest_positive_period_intervals_monotonic_increase_max_min_values_l197_197933


namespace subtraction_problem_solution_l197_197075

theorem subtraction_problem_solution :
  ∃ x : ℝ, (8 - x) / (9 - x) = 4 / 5 :=
by
  use 4
  sorry

end subtraction_problem_solution_l197_197075


namespace pencils_per_pack_l197_197256

def packs := 28
def rows := 42
def pencils_per_row := 16

theorem pencils_per_pack (total_pencils : ℕ) : total_pencils = rows * pencils_per_row → total_pencils / packs = 24 :=
by
  sorry

end pencils_per_pack_l197_197256


namespace range_half_diff_l197_197192

theorem range_half_diff (α β : ℝ) (h1 : -π/2 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) : 
    -π/2 ≤ (α - β) / 2 ∧ (α - β) / 2 < 0 := 
    sorry

end range_half_diff_l197_197192


namespace amy_remaining_money_l197_197947

-- Define initial amount and purchases
def initial_amount : ℝ := 15
def stuffed_toy_cost : ℝ := 2
def hot_dog_cost : ℝ := 3.5
def candy_apple_cost : ℝ := 1.5
def discount_rate : ℝ := 0.5

-- Define the discounted hot_dog_cost
def discounted_hot_dog_cost := hot_dog_cost * discount_rate

-- Define the total spent
def total_spent := stuffed_toy_cost + discounted_hot_dog_cost + candy_apple_cost

-- Define the remaining amount
def remaining_amount := initial_amount - total_spent

theorem amy_remaining_money : remaining_amount = 9.75 := by
  sorry

end amy_remaining_money_l197_197947


namespace find_value_of_expression_l197_197116

theorem find_value_of_expression (x : ℝ) (h : 5 * x - 3 = 7) : 3 * x^2 + 2 = 14 :=
by {
  sorry
}

end find_value_of_expression_l197_197116


namespace sequence_value_2009_l197_197198

theorem sequence_value_2009 
  (a : ℕ → ℝ)
  (h_recur : ∀ n ≥ 2, a n = a (n - 1) * a (n + 1))
  (h_a1 : a 1 = 1 + Real.sqrt 3)
  (h_a1776 : a 1776 = 4 + Real.sqrt 3) :
  a 2009 = (3 / 2) + (3 * Real.sqrt 3 / 2) := 
sorry

end sequence_value_2009_l197_197198


namespace area_of_triangle_PQR_l197_197385

-- Define the vertices P, Q, and R
def P : (Int × Int) := (-3, 2)
def Q : (Int × Int) := (1, 7)
def R : (Int × Int) := (3, -1)

-- Define the formula for the area of a triangle given vertices
def triangle_area (A B C : Int × Int) : Real :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Define the statement to prove
theorem area_of_triangle_PQR : triangle_area P Q R = 21 := 
  sorry

end area_of_triangle_PQR_l197_197385


namespace no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018_l197_197437

theorem no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018 (m n : ℕ) : ¬ (m^2 = n^2 + 2018) :=
sorry

end no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018_l197_197437


namespace TableCostEquals_l197_197133

-- Define the given conditions and final result
def total_spent : ℕ := 56
def num_chairs : ℕ := 2
def chair_cost : ℕ := 11
def table_cost : ℕ := 34

-- State the assertion to be proved
theorem TableCostEquals :
  table_cost = total_spent - (num_chairs * chair_cost) := 
by 
  sorry

end TableCostEquals_l197_197133


namespace james_choices_count_l197_197347

-- Define the conditions as Lean definitions
def isAscending (a b c d e : ℕ) : Prop := a < b ∧ b < c ∧ c < d ∧ d < e

def inRange (a b c d e : ℕ) : Prop := a ≤ 8 ∧ b ≤ 8 ∧ c ≤ 8 ∧ d ≤ 8 ∧ e ≤ 8

def meanEqualsMedian (a b c d e : ℕ) : Prop :=
  (a + b + c + d + e) / 5 = c

-- Define the problem statement
theorem james_choices_count :
  ∃ (s : Finset (ℕ × ℕ × ℕ × ℕ × ℕ)), 
    (∀ (a b c d e : ℕ), (a, b, c, d, e) ∈ s ↔ isAscending a b c d e ∧ inRange a b c d e ∧ meanEqualsMedian a b c d e) ∧
    s.card = 10 :=
sorry

end james_choices_count_l197_197347


namespace total_time_is_correct_l197_197683

-- Definitions based on conditions
def timeouts_for_running : ℕ := 5
def timeouts_for_throwing_food : ℕ := 5 * timeouts_for_running - 1
def timeouts_for_swearing : ℕ := timeouts_for_throwing_food / 3

-- Definition for total time-outs
def total_timeouts : ℕ := timeouts_for_running + timeouts_for_throwing_food + timeouts_for_swearing
-- Each time-out is 5 minutes
def timeout_duration : ℕ := 5

-- Total time in minutes
def total_time_in_minutes : ℕ := total_timeouts * timeout_duration

-- The proof statement
theorem total_time_is_correct : total_time_in_minutes = 185 := by
  sorry

end total_time_is_correct_l197_197683


namespace parallelogram_area_l197_197810

theorem parallelogram_area (base height : ℕ) (h_base : base = 5) (h_height : height = 3) :
  base * height = 15 :=
by
  -- Here would be the proof, but it is omitted per instructions
  sorry

end parallelogram_area_l197_197810


namespace spring_work_compression_l197_197444

theorem spring_work_compression :
  ∀ (k : ℝ) (F : ℝ) (x : ℝ), 
  (F = 10) → (x = 1 / 100) → (k = F / x) → (W = 5) :=
by
sorry

end spring_work_compression_l197_197444


namespace total_sequences_correct_l197_197367

/-- 
Given 6 blocks arranged such that:
1. Block 1 must be removed first.
2. Blocks 2 and 3 become accessible after Block 1 is removed.
3. Blocks 4, 5, and 6 become accessible after Blocks 2 and 3 are removed.
4. A block can only be removed if no other block is stacked on top of it. 

Prove that the total number of possible sequences to remove all the blocks is 10.
-/
def total_sequences_to_remove_blocks : ℕ := 10

theorem total_sequences_correct : 
  total_sequences_to_remove_blocks = 10 :=
sorry

end total_sequences_correct_l197_197367


namespace angles_between_plane_and_catheti_l197_197077

theorem angles_between_plane_and_catheti
  (α β : ℝ)
  (h_alpha : 0 < α ∧ α < π / 2)
  (h_beta : 0 < β ∧ β < π / 2) :
  ∃ γ θ : ℝ,
    γ = Real.arcsin (Real.sin β * Real.cos α) ∧
    θ = Real.arcsin (Real.sin β * Real.sin α) :=
by
  sorry

end angles_between_plane_and_catheti_l197_197077


namespace geometric_seq_arithmetic_condition_l197_197318

open Real

noncomputable def common_ratio (q : ℝ) := (q > 0) ∧ (q^2 - q - 1 = 0)

def arithmetic_seq_condition (a1 a2 a3 : ℝ) := (a2 = (a1 + a3) / 2)

theorem geometric_seq_arithmetic_condition (a1 a2 a3 a4 a5 : ℝ) (q : ℝ)
  (h1 : 0 < q)
  (h2 : q^2 - q - 1 = 0)
  (h3 : a2 = q * a1)
  (h4 : a3 = q * a2)
  (h5 : a4 = q * a3)
  (h6 : a5 = q * a4)
  (h7 : arithmetic_seq_condition a1 a2 a3) :
  (a4 + a5) / (a3 + a4) = (1 + sqrt 5) / 2 := 
sorry

end geometric_seq_arithmetic_condition_l197_197318


namespace direct_proportion_function_l197_197037

-- Definitions of the given functions
def fA (x : ℝ) : ℝ := 3 * x - 4
def fB (x : ℝ) : ℝ := -2 * x + 1
def fC (x : ℝ) : ℝ := 3 * x
def fD (x : ℝ) : ℝ := 4

-- Direct proportion function definition
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∀ x, f 0 = 0 ∧ (f x) / x = f 1 / 1

-- Prove that fC (x) is the only direct proportion function among the given options
theorem direct_proportion_function :
  is_direct_proportion fC ∧ ¬ is_direct_proportion fA ∧ ¬ is_direct_proportion fB ∧ ¬ is_direct_proportion fD :=
by
  sorry

end direct_proportion_function_l197_197037


namespace complement_union_l197_197760

def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B : Set ℝ := { x | x ≥ 1 }
def C (s : Set ℝ) : Set ℝ := { x | ¬ s x }

theorem complement_union :
  C (A ∪ B) = { x | x ≤ -1 } :=
by {
  sorry
}

end complement_union_l197_197760


namespace trigonometric_identity_proof_l197_197522

theorem trigonometric_identity_proof (α : ℝ) (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10)) / (Real.sin (α - Real.pi / 5)) = 3 :=
by
  sorry

end trigonometric_identity_proof_l197_197522


namespace baseball_cards_initial_count_unkn_l197_197805

-- Definitions based on the conditions
def cardValue : ℕ := 6
def tradedCards : ℕ := 2
def receivedCardsValue : ℕ := (3 * 2) + 9   -- 3 cards worth $2 each and 1 card worth $9
def profit : ℕ := receivedCardsValue - (tradedCards * cardValue)

-- Lean 4 statement to represent the proof problem
theorem baseball_cards_initial_count_unkn (h_trade : tradedCards * cardValue = 12)
    (h_receive : receivedCardsValue = 15)
    (h_profit : profit = 3) : ∃ n : ℕ, n >= 2 ∧ n = 2 + (n - 2) :=
sorry

end baseball_cards_initial_count_unkn_l197_197805


namespace candy_store_problem_l197_197181

variable (S : ℝ)
variable (not_caught_percentage : ℝ) (sample_percentage : ℝ)
variable (caught_percentage : ℝ := 1 - not_caught_percentage)

theorem candy_store_problem
  (h1 : not_caught_percentage = 0.15)
  (h2 : sample_percentage = 25.88235294117647) :
  caught_percentage * sample_percentage = 22 := by
  sorry

end candy_store_problem_l197_197181


namespace find_vertex_D_l197_197504

structure Point where
  x : ℤ
  y : ℤ

def vector_sub (a b : Point) : Point :=
  Point.mk (a.x - b.x) (a.y - b.y)

def vector_add (a b : Point) : Point :=
  Point.mk (a.x + b.x) (a.y + b.y)

def is_parallelogram (A B C D : Point) : Prop :=
  vector_sub B A = vector_sub D C

theorem find_vertex_D (A B C D : Point)
  (hA : A = Point.mk (-1) (-2))
  (hB : B = Point.mk 3 (-1))
  (hC : C = Point.mk 5 6)
  (hParallelogram: is_parallelogram A B C D) :
  D = Point.mk 1 5 :=
sorry

end find_vertex_D_l197_197504


namespace cumulative_distribution_X_maximized_expected_score_l197_197884

noncomputable def distribution_X (p_A : ℝ) (p_B : ℝ) : (ℝ × ℝ × ℝ) :=
(1 - p_A, p_A * (1 - p_B), p_A * p_B)

def expected_score (p_A : ℝ) (p_B : ℝ) (s_A : ℝ) (s_B : ℝ) : ℝ :=
0 * (1 - p_A) + s_A * (p_A * (1 - p_B)) + (s_A + s_B) * (p_A * p_B)

theorem cumulative_distribution_X :
  distribution_X 0.8 0.6 = (0.2, 0.32, 0.48) :=
sorry

theorem maximized_expected_score :
  expected_score 0.8 0.6 20 80 < expected_score 0.6 0.8 80 20 :=
sorry

end cumulative_distribution_X_maximized_expected_score_l197_197884


namespace girls_at_ends_no_girls_next_to_each_other_girl_A_right_of_girl_B_l197_197468

namespace PhotoArrangement

/-- There are 4 boys and 3 girls. -/
def boys : ℕ := 4
def girls : ℕ := 3

/-- Number of ways to arrange given conditions -/
def arrangementsWithGirlsAtEnds : ℕ := 720
def arrangementsWithNoGirlsNextToEachOther : ℕ := 1440
def arrangementsWithGirlAtoRightOfGirlB : ℕ := 2520

-- Problem 1: If there are girls at both ends
theorem girls_at_ends (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithGirlsAtEnds := by
  sorry

-- Problem 2: If no two girls are standing next to each other
theorem no_girls_next_to_each_other (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithNoGirlsNextToEachOther := by
  sorry

-- Problem 3: If girl A must be to the right of girl B
theorem girl_A_right_of_girl_B (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithGirlAtoRightOfGirlB := by
  sorry

end PhotoArrangement

end girls_at_ends_no_girls_next_to_each_other_girl_A_right_of_girl_B_l197_197468


namespace cheyenne_clay_pots_l197_197045

theorem cheyenne_clay_pots (P : ℕ) (cracked_ratio sold_ratio : ℝ) (total_revenue price_per_pot : ℝ) 
    (P_sold : ℕ) :
  cracked_ratio = (2 / 5) →
  sold_ratio = (3 / 5) →
  total_revenue = 1920 →
  price_per_pot = 40 →
  P_sold = 48 →
  (sold_ratio * P = P_sold) →
  P = 80 :=
by
  sorry

end cheyenne_clay_pots_l197_197045


namespace probability_neither_red_nor_purple_l197_197907

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 18
def yellow_balls : ℕ := 8
def red_balls : ℕ := 5
def purple_balls : ℕ := 7

theorem probability_neither_red_nor_purple : 
  (total_balls - (red_balls + purple_balls)) / total_balls = 4 / 5 :=
by sorry

end probability_neither_red_nor_purple_l197_197907


namespace perpendicular_lines_slope_l197_197593

theorem perpendicular_lines_slope {m : ℝ} : 
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0) → 
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0 → (m * (-1/2)) = -1) → 
  m = 2 :=
by 
  intros h_perpendicular h_slope
  sorry

end perpendicular_lines_slope_l197_197593


namespace Kelly_remaining_games_l197_197373

-- Definitions according to the conditions provided
def initial_games : ℝ := 121.0
def given_away : ℝ := 99.0
def remaining_games : ℝ := initial_games - given_away

-- The proof problem statement
theorem Kelly_remaining_games : remaining_games = 22.0 :=
by
  -- sorry is used here to skip the proof
  sorry

end Kelly_remaining_games_l197_197373


namespace smallest_integer_larger_than_expression_l197_197017

theorem smallest_integer_larger_than_expression :
  ∃ n : ℤ, n = 248 ∧ (↑n > ((Real.sqrt 5 + Real.sqrt 3) ^ 4 : ℝ)) :=
by
  sorry

end smallest_integer_larger_than_expression_l197_197017


namespace five_digit_number_is_40637_l197_197876

theorem five_digit_number_is_40637 
  (A B C D E F G : ℕ) 
  (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ 
        B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ 
        C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ 
        D ≠ E ∧ D ≠ F ∧ D ≠ G ∧
        E ≠ F ∧ E ≠ G ∧ 
        F ≠ G)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E ∧ 0 < F ∧ 0 < G)
  (h3 : A + 11 * A = 2 * (10 * B + A))
  (h4 : A + 10 * C + D = 2 * (10 * A + B))
  (h5 : 10 * C + D = 20 * A)
  (h6 : 20 + 62 = 2 * (10 * C + A)) -- for sequences formed by AB, CA, EF
  (h7 : 21 + 63 = 2 * (10 * G + A)) -- for sequences formed by BA, CA, GA
  : ∃ (C D E F G : ℕ), C * 10000 + D * 1000 + E * 100 + F * 10 + G = 40637 := 
sorry

end five_digit_number_is_40637_l197_197876


namespace sin_225_eq_neg_sqrt_two_div_two_l197_197774

theorem sin_225_eq_neg_sqrt_two_div_two :
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt_two_div_two_l197_197774


namespace not_perfect_square_l197_197888

theorem not_perfect_square (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h : ¬ (a^2 - b^2) % 4 = 0) : 
  ¬ ∃ k : ℤ, (a + 3*b) * (5*a + 7*b) = k^2 :=
sorry

end not_perfect_square_l197_197888


namespace expr_containing_x_to_y_l197_197671

theorem expr_containing_x_to_y (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  -- proof steps would be here
  sorry

end expr_containing_x_to_y_l197_197671


namespace cost_500_pencils_is_25_dollars_l197_197496

def cost_of_500_pencils (cost_per_pencil : ℕ) (pencils : ℕ) (cents_per_dollar : ℕ) : ℕ :=
  (cost_per_pencil * pencils) / cents_per_dollar

theorem cost_500_pencils_is_25_dollars : cost_of_500_pencils 5 500 100 = 25 := by
  sorry

end cost_500_pencils_is_25_dollars_l197_197496


namespace interval_length_correct_l197_197662

def sin_log_interval_sum : ℝ := sorry

theorem interval_length_correct :
  sin_log_interval_sum = 2^π / (1 + 2^π) :=
by
  -- Definitions
  let is_valid_x (x : ℝ) := x < 1 ∧ x > 0 ∧ (Real.sin (Real.log x / Real.log 2)) < 0
  
  -- Assertion
  sorry

end interval_length_correct_l197_197662


namespace problem_statement_l197_197930

-- Define the operation #
def op_hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

-- The main theorem statement
theorem problem_statement (a b : ℕ) (h1 : op_hash a b = 100) : (a + b) + 6 = 11 := 
sorry

end problem_statement_l197_197930


namespace monthly_savings_correct_l197_197668

-- Define the gross salaries before any deductions
def ivan_salary_gross : ℝ := 55000
def vasilisa_salary_gross : ℝ := 45000
def vasilisa_mother_salary_gross : ℝ := 18000
def vasilisa_father_salary_gross : ℝ := 20000
def son_scholarship_state : ℝ := 3000
def son_scholarship_non_state_gross : ℝ := 15000

-- Tax rate definition
def tax_rate : ℝ := 0.13

-- Net income calculations using the tax rate
def net_income (gross_income : ℝ) : ℝ := gross_income * (1 - tax_rate)

def ivan_salary_net : ℝ := net_income ivan_salary_gross
def vasilisa_salary_net : ℝ := net_income vasilisa_salary_gross
def vasilisa_mother_salary_net : ℝ := net_income vasilisa_mother_salary_gross
def vasilisa_father_salary_net : ℝ := net_income vasilisa_father_salary_gross
def son_scholarship_non_state_net : ℝ := net_income son_scholarship_non_state_gross

-- Monthly expenses total
def monthly_expenses : ℝ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

-- Net incomes for different periods
def total_net_income_before_01_05_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + vasilisa_mother_salary_net + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_01_05_2018_to_31_08_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_from_01_09_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + (son_scholarship_state + son_scholarship_non_state_net)

-- Savings calculations for different periods
def monthly_savings_before_01_05_2018 : ℝ :=
  total_net_income_before_01_05_2018 - monthly_expenses

def monthly_savings_01_05_2018_to_31_08_2018 : ℝ :=
  total_net_income_01_05_2018_to_31_08_2018 - monthly_expenses

def monthly_savings_from_01_09_2018 : ℝ :=
  total_net_income_from_01_09_2018 - monthly_expenses

-- Theorem to be proved
theorem monthly_savings_correct :
  monthly_savings_before_01_05_2018 = 49060 ∧
  monthly_savings_01_05_2018_to_31_08_2018 = 43400 ∧
  monthly_savings_from_01_09_2018 = 56450 :=
by
  sorry

end monthly_savings_correct_l197_197668


namespace f_expr_for_nonneg_l197_197052

-- Define the function f piecewise as per the given conditions
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then
    Real.exp (-x) + 2 * x - 1
  else
    -Real.exp x + 2 * x + 1

-- Prove that for x > 0, f(x) = -e^x + 2x + 1 given the conditions
theorem f_expr_for_nonneg (x : ℝ) (h : x ≥ 0) : f x = -Real.exp x + 2 * x + 1 := by
  sorry

end f_expr_for_nonneg_l197_197052


namespace product_of_fractions_l197_197160

-- Define the fractions as ratios.
def fraction1 : ℚ := 2 / 5
def fraction2 : ℚ := 7 / 10

-- State the theorem that proves the product of the fractions is equal to the simplified result.
theorem product_of_fractions : fraction1 * fraction2 = 7 / 25 :=
by
  -- Skip the proof.
  sorry

end product_of_fractions_l197_197160


namespace snail_distance_round_100_l197_197511

def snail_distance (n : ℕ) : ℕ :=
  if n = 0 then 100 else (100 * (n + 2)) / (n + 1)

theorem snail_distance_round_100 : snail_distance 100 = 5050 :=
  sorry

end snail_distance_round_100_l197_197511


namespace x_gt_1_implies_inv_x_lt_1_inv_x_lt_1_not_necessitates_x_gt_1_l197_197859

theorem x_gt_1_implies_inv_x_lt_1 (x : ℝ) (h : x > 1) : 1 / x < 1 :=
by
  sorry

theorem inv_x_lt_1_not_necessitates_x_gt_1 (x : ℝ) (h : 1 / x < 1) : ¬(x > 1) ∨ (x ≤ 1) :=
by
  sorry

end x_gt_1_implies_inv_x_lt_1_inv_x_lt_1_not_necessitates_x_gt_1_l197_197859


namespace total_weight_proof_l197_197449
-- Import the entire math library

-- Assume the conditions as given variables
variables (w r s : ℕ)
-- Assign values to the given conditions
def weight_per_rep := 15
def reps_per_set := 10
def number_of_sets := 3

-- Calculate total weight moved
def total_weight_moved := w * r * s

-- The theorem to prove the total weight moved
theorem total_weight_proof : total_weight_moved weight_per_rep reps_per_set number_of_sets = 450 :=
by
  -- Provide the expected result directly, proving the statement
  sorry

end total_weight_proof_l197_197449


namespace tim_investment_l197_197263

noncomputable def initial_investment_required 
  (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  A / ((1 + r / n) ^ (n * t))

theorem tim_investment :
  initial_investment_required 100000 0.10 2 3 = 74622 :=
by
  sorry

end tim_investment_l197_197263


namespace smallest_number_of_cookies_l197_197379

theorem smallest_number_of_cookies
  (n : ℕ) 
  (hn : 4 * n - 4 = (n^2) / 2) : n = 7 → n^2 = 49 := 
by
  sorry

end smallest_number_of_cookies_l197_197379


namespace circle_eq1_circle_eq2_l197_197025

-- Problem 1: Circle with center M(-5, 3) and passing through point A(-8, -1)
theorem circle_eq1 : ∀ (x y : ℝ), (x + 5) ^ 2 + (y - 3) ^ 2 = 25 :=
by
  sorry

-- Problem 2: Circle passing through three points A(-2, 4), B(-1, 3), C(2, 6)
theorem circle_eq2 : ∀ (x y : ℝ), x ^ 2 + (y - 5) ^ 2 = 5 :=
by
  sorry

end circle_eq1_circle_eq2_l197_197025


namespace donuts_per_student_l197_197083

theorem donuts_per_student 
    (dozens_of_donuts : ℕ)
    (students_in_class : ℕ)
    (percentage_likes_donuts : ℕ)
    (students_who_like_donuts : ℕ)
    (total_donuts : ℕ)
    (donuts_per_student : ℕ) :
    dozens_of_donuts = 4 →
    students_in_class = 30 →
    percentage_likes_donuts = 80 →
    students_who_like_donuts = (percentage_likes_donuts * students_in_class) / 100 →
    total_donuts = dozens_of_donuts * 12 →
    donuts_per_student = total_donuts / students_who_like_donuts →
    donuts_per_student = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end donuts_per_student_l197_197083


namespace combined_selling_price_correct_l197_197705

noncomputable def cost_A : ℝ := 500
noncomputable def cost_B : ℝ := 800
noncomputable def profit_A_perc : ℝ := 0.10
noncomputable def profit_B_perc : ℝ := 0.15
noncomputable def tax_perc : ℝ := 0.05
noncomputable def packaging_fee : ℝ := 50

-- Calculating selling prices before tax and fees
noncomputable def selling_price_A_before_tax_fees : ℝ := cost_A * (1 + profit_A_perc)
noncomputable def selling_price_B_before_tax_fees : ℝ := cost_B * (1 + profit_B_perc)

-- Calculating taxes
noncomputable def tax_A : ℝ := selling_price_A_before_tax_fees * tax_perc
noncomputable def tax_B : ℝ := selling_price_B_before_tax_fees * tax_perc

-- Adding tax to selling prices
noncomputable def selling_price_A_incl_tax : ℝ := selling_price_A_before_tax_fees + tax_A
noncomputable def selling_price_B_incl_tax : ℝ := selling_price_B_before_tax_fees + tax_B

-- Adding packaging and shipping fees
noncomputable def final_selling_price_A : ℝ := selling_price_A_incl_tax + packaging_fee
noncomputable def final_selling_price_B : ℝ := selling_price_B_incl_tax + packaging_fee

-- Combined selling price
noncomputable def combined_selling_price : ℝ := final_selling_price_A + final_selling_price_B

theorem combined_selling_price_correct : 
  combined_selling_price = 1643.5 := by
  sorry

end combined_selling_price_correct_l197_197705


namespace cost_of_first_shipment_1100_l197_197001

variables (S J : ℝ)
-- conditions
def second_shipment (S J : ℝ) := 5 * S + 15 * J = 550
def first_shipment (S J : ℝ) := 10 * S + 20 * J

-- goal
theorem cost_of_first_shipment_1100 (S J : ℝ) (h : second_shipment S J) : first_shipment S J = 1100 :=
sorry

end cost_of_first_shipment_1100_l197_197001


namespace symmetric_points_l197_197065

theorem symmetric_points (a b : ℝ) (h1 : 2 * a + 1 = -1) (h2 : 4 = -(3 * b - 1)) :
  2 * a + b = -3 := 
sorry

end symmetric_points_l197_197065


namespace train_length_l197_197196

/-- Proof problem: 
  Given the speed of a train is 52 km/hr and it crosses a 280-meter long platform in 18 seconds,
  prove that the length of the train is 259.92 meters.
-/
theorem train_length (speed_kmh : ℕ) (platform_length : ℕ) (time_sec : ℕ) (speed_mps : ℝ) 
  (distance_covered : ℝ) (train_length : ℝ) :
  speed_kmh = 52 → platform_length = 280 → time_sec = 18 → 
  speed_mps = (speed_kmh * 1000) / 3600 → distance_covered = speed_mps * time_sec →
  train_length = distance_covered - platform_length →
  train_length = 259.92 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end train_length_l197_197196


namespace polar_to_cartesian_l197_197897

theorem polar_to_cartesian (ρ θ : ℝ) : (ρ * Real.cos θ = 0) → ρ = 0 ∨ θ = π/2 :=
by 
  sorry

end polar_to_cartesian_l197_197897


namespace committee_combinations_l197_197020

-- We use a broader import to ensure all necessary libraries are included.
-- Definitions and theorem

def club_member_count : ℕ := 20
def committee_member_count : ℕ := 3

theorem committee_combinations : 
  (Nat.choose club_member_count committee_member_count) = 1140 := by
sorry

end committee_combinations_l197_197020


namespace smallest_n_satisfying_equation_l197_197484

theorem smallest_n_satisfying_equation : ∃ (k : ℤ), (∃ (n : ℤ), n > 0 ∧ n % 2 = 1 ∧ (n ^ 3 + 2 * n ^ 2 = k ^ 2) ∧ ∀ m : ℤ, (m > 0 ∧ m < n ∧ m % 2 = 1) → ¬ (∃ j : ℤ, m ^ 3 + 2 * m ^ 2 = j ^ 2)) ∧ k % 2 = 1 :=
sorry

end smallest_n_satisfying_equation_l197_197484


namespace A_times_B_is_correct_l197_197429

noncomputable def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {x : ℝ | x ≥ 0}

noncomputable def A_union_B : Set ℝ := {x : ℝ | x ≥ 0}
noncomputable def A_inter_B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

noncomputable def A_times_B : Set ℝ := {x : ℝ | x ∈ A_union_B ∧ x ∉ A_inter_B}

theorem A_times_B_is_correct :
  A_times_B = {x : ℝ | x > 2} := sorry

end A_times_B_is_correct_l197_197429


namespace average_speed_car_y_l197_197247

-- Defining the constants based on the problem conditions
def speedX : ℝ := 35
def timeDifference : ℝ := 1.2  -- This is 72 minutes converted to hours
def distanceFromStartOfY : ℝ := 294

-- Defining the main statement
theorem average_speed_car_y : 
  ( ∀ timeX timeY distanceX distanceY : ℝ, 
      timeX = timeY + timeDifference ∧
      distanceX = speedX * timeX ∧
      distanceY = distanceFromStartOfY ∧
      distanceX = distanceFromStartOfY + speedX * timeDifference
  → distanceY / timeX = 30.625) :=
sorry

end average_speed_car_y_l197_197247


namespace second_sweet_red_probability_l197_197574

theorem second_sweet_red_probability (x y : ℕ) : 
  (y / (x + y : ℝ)) = y / (x + y + 10) * x / (x + y) + (y + 10) / (x + y + 10) * y / (x + y) :=
by
  sorry

end second_sweet_red_probability_l197_197574


namespace no_valid_x_for_given_circle_conditions_l197_197458

theorem no_valid_x_for_given_circle_conditions :
  ∀ x : ℝ,
    ¬ ((x - 15)^2 + 18^2 = 225 ∧ (x - 15)^2 + (-18)^2 = 225) :=
by
  sorry

end no_valid_x_for_given_circle_conditions_l197_197458


namespace gcd_of_35_and_number_between_70_and_90_is_7_l197_197564

def number_between_70_and_90 (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 90

def gcd_is_7 (a b : ℕ) : Prop :=
  Nat.gcd a b = 7

theorem gcd_of_35_and_number_between_70_and_90_is_7 : 
  ∃ (n : ℕ), number_between_70_and_90 n ∧ gcd_is_7 35 n ∧ (n = 77 ∨ n = 84) :=
by
  sorry

end gcd_of_35_and_number_between_70_and_90_is_7_l197_197564


namespace zumish_12_words_remainder_l197_197411

def zumishWords n :=
  if n < 2 then (0, 0, 0)
  else if n == 2 then (4, 4, 4)
  else let (a, b, c) := zumishWords (n - 1)
       (2 * (a + c) % 1000, 2 * a % 1000, 2 * b % 1000)

def countZumishWords (n : Nat) :=
  let (a, b, c) := zumishWords n
  (a + b + c) % 1000

theorem zumish_12_words_remainder :
  countZumishWords 12 = 322 :=
by
  intros
  sorry

end zumish_12_words_remainder_l197_197411


namespace difference_of_extreme_valid_numbers_l197_197138

theorem difference_of_extreme_valid_numbers :
  ∃ (largest smallest : ℕ),
    (largest = 222210 ∧ smallest = 100002) ∧ 
    (largest % 3 = 0 ∧ smallest % 3 = 0) ∧ 
    (largest ≥ 100000 ∧ largest < 1000000) ∧
    (smallest ≥ 100000 ∧ smallest < 1000000) ∧
    (∀ d, d ∈ [0, 1, 2] → (d ∈ [largest / 100000 % 10, largest / 10000 % 10, largest / 1000 % 10, largest / 100 % 10, largest / 10 % 10, largest % 10])) ∧
    (∀ d, d ∈ [0, 1, 2] → (d ∈ [smallest / 100000 % 10, smallest / 10000 % 10, smallest / 1000 % 10, smallest / 100 % 10, smallest / 10 % 10, smallest % 10])) ∧ 
    (∀ d ∈ [largest / 100000 % 10, largest / 10000 % 10, largest / 1000 % 10, largest / 100 % 10, largest / 10 % 10, largest % 10], d ∈ [0, 1, 2]) ∧
    (∀ d ∈ [smallest / 100000 % 10, smallest / 10000 % 10, smallest / 1000 % 10, smallest / 100 % 10, smallest / 10 % 10, smallest % 10], d ∈ [0, 1, 2]) ∧
    (largest - smallest = 122208) :=
by
  sorry

end difference_of_extreme_valid_numbers_l197_197138


namespace ratio_of_shorts_to_pants_is_half_l197_197978

-- Define the parameters
def shirts := 4
def pants := 2 * shirts
def total_clothes := 16

-- Define the number of shorts
def shorts := total_clothes - (shirts + pants)

-- Define the ratio
def ratio := shorts / pants

-- Prove the ratio is 1/2
theorem ratio_of_shorts_to_pants_is_half : ratio = 1 / 2 :=
by
  -- Start the proof, but leave it as sorry
  sorry

end ratio_of_shorts_to_pants_is_half_l197_197978


namespace bird_count_l197_197432

def initial_birds : ℕ := 12
def new_birds : ℕ := 8
def total_birds : ℕ := initial_birds + new_birds

theorem bird_count : total_birds = 20 := by
  sorry

end bird_count_l197_197432


namespace remainder_when_divided_by_x_minus_2_l197_197340

noncomputable def f (x : ℝ) : ℝ :=
  x^4 - 8 * x^3 + 12 * x^2 + 20 * x - 18

theorem remainder_when_divided_by_x_minus_2 :
  f 2 = 22 := 
sorry

end remainder_when_divided_by_x_minus_2_l197_197340


namespace line_through_points_l197_197156

/-- The line passing through points A(1, 1) and B(2, 3) satisfies the equation 2x - y - 1 = 0. -/
theorem line_through_points (x y : ℝ) :
  (∃ k : ℝ, k * (y - 1) = 2 * (x - 1)) → 2 * x - y - 1 = 0 :=
by
  sorry

end line_through_points_l197_197156


namespace sum_of_roots_of_quadratic_eqn_l197_197488

theorem sum_of_roots_of_quadratic_eqn (A B : ℝ) 
  (h₁ : 3 * A ^ 2 - 9 * A + 6 = 0)
  (h₂ : 3 * B ^ 2 - 9 * B + 6 = 0)
  (h_distinct : A ≠ B):
  A + B = 3 := by
  sorry

end sum_of_roots_of_quadratic_eqn_l197_197488


namespace alex_walking_distance_l197_197855

theorem alex_walking_distance
  (distance : ℝ)
  (time_45 : ℝ)
  (walking_rate : distance = 1.5 ∧ time_45 = 45):
  ∃ distance_90, distance_90 = 3 :=
by 
  sorry

end alex_walking_distance_l197_197855


namespace smallest_n_satisfying_conditions_l197_197612

theorem smallest_n_satisfying_conditions : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ x : ℕ, 3 * n = x^4) ∧ (∃ y : ℕ, 2 * n = y^5) ∧ n = 432 :=
by
  sorry

end smallest_n_satisfying_conditions_l197_197612


namespace distance_from_P_to_AD_l197_197403

-- Definitions of points and circles
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 0, y := 4}
def D : Point := {x := 0, y := 0}
def C : Point := {x := 4, y := 0}
def M : Point := {x := 2, y := 0}
def radiusM : ℝ := 2
def radiusA : ℝ := 4

-- Definition of the circles
def circleM (P : Point) : Prop := (P.x - M.x)^2 + P.y^2 = radiusM^2
def circleA (P : Point) : Prop := P.x^2 + (P.y - A.y)^2 = radiusA^2

-- Definition of intersection point \(P\) of the two circles
def is_intersection (P : Point) : Prop := circleM P ∧ circleA P

-- Distance from point \(P\) to line \(\overline{AD}\) computed as the x-coordinate
def distance_to_line_AD (P : Point) : ℝ := P.x

-- The theorem to prove
theorem distance_from_P_to_AD :
  ∃ P : Point, is_intersection P ∧ distance_to_line_AD P = 16/5 :=
by {
  -- Use "sorry" as the proof placeholder
  sorry
}

end distance_from_P_to_AD_l197_197403


namespace total_rainfall_l197_197672

-- Given conditions
def sunday_rainfall : ℕ := 4
def monday_rainfall : ℕ := sunday_rainfall + 3
def tuesday_rainfall : ℕ := 2 * monday_rainfall

-- Question: Total rainfall over the 3 days
theorem total_rainfall : sunday_rainfall + monday_rainfall + tuesday_rainfall = 25 := by
  sorry

end total_rainfall_l197_197672


namespace tan_subtraction_example_l197_197618

noncomputable def tan_subtraction_identity (alpha beta : ℝ) : ℝ :=
  (Real.tan alpha - Real.tan beta) / (1 + Real.tan alpha * Real.tan beta)

theorem tan_subtraction_example (theta : ℝ) (h : Real.tan theta = 1 / 2) :
  Real.tan (π / 4 - theta) = 1 / 3 := 
by
  sorry

end tan_subtraction_example_l197_197618


namespace arithmetic_sequence_product_l197_197642

theorem arithmetic_sequence_product 
  (b : ℕ → ℤ) 
  (h_arith : ∀ n, b n = b 0 + (n : ℤ) * (b 1 - b 0))
  (h_inc : ∀ n, b n ≤ b (n + 1))
  (h4_5 : b 4 * b 5 = 21) : 
  b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 := 
by 
  sorry

end arithmetic_sequence_product_l197_197642


namespace conditional_probability_l197_197514

variable (pA pB pAB : ℝ)
variable (h1 : pA = 0.2)
variable (h2 : pB = 0.18)
variable (h3 : pAB = 0.12)

theorem conditional_probability : (pAB / pB = 2 / 3) :=
by
  -- sorry is used to skip the proof
  sorry

end conditional_probability_l197_197514


namespace workdays_ride_l197_197260

-- Define the conditions
def work_distance : ℕ := 20
def weekend_ride : ℕ := 200
def speed : ℕ := 25
def hours_per_week : ℕ := 16

-- Define the question
def total_distance : ℕ := speed * hours_per_week
def distance_during_workdays : ℕ := total_distance - weekend_ride
def round_trip_distance : ℕ := 2 * work_distance

theorem workdays_ride : 
  (distance_during_workdays / round_trip_distance) = 5 :=
by
  sorry

end workdays_ride_l197_197260


namespace max_regions_with_6_chords_l197_197155

-- Definition stating the number of regions created by k chords
def regions_by_chords (k : ℕ) : ℕ :=
  1 + (k * (k + 1)) / 2

-- Lean statement for the proof problem
theorem max_regions_with_6_chords : regions_by_chords 6 = 22 :=
  by sorry

end max_regions_with_6_chords_l197_197155


namespace eq_perp_bisector_BC_area_triangle_ABC_l197_197506

section Triangle_ABC

open Real

-- Define the vertices A, B, and C
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (3, 3)

-- Define the equation of the perpendicular bisector
theorem eq_perp_bisector_BC : ∀ x y : ℝ, 2 * x + y - 4 = 0 :=
sorry

-- Define the area of the triangle ABC
noncomputable def triangle_area : ℝ :=
1 / 2 * (abs ((-1 * 3 + 3 * (-2) + 3 * 4) - (3 * 4 + 1 * (-2) + 3*(-1))))

theorem area_triangle_ABC : triangle_area = 7 :=
sorry

end Triangle_ABC

end eq_perp_bisector_BC_area_triangle_ABC_l197_197506


namespace base_height_ratio_l197_197205

-- Define the conditions
def cultivation_cost : ℝ := 333.18
def rate_per_hectare : ℝ := 24.68
def base_of_field : ℝ := 300
def height_of_field : ℝ := 300

-- Prove the ratio of base to height is 1
theorem base_height_ratio (b h : ℝ) (cost rate : ℝ)
  (h1 : cost = 333.18) (h2 : rate = 24.68) 
  (h3 : b = 300) (h4 : h = 300) : b / h = 1 :=
by
  sorry

end base_height_ratio_l197_197205


namespace exists_x_for_ax2_plus_2x_plus_a_lt_0_l197_197880

theorem exists_x_for_ax2_plus_2x_plus_a_lt_0 (a : ℝ) : (∃ x : ℝ, a * x^2 + 2 * x + a < 0) ↔ a < 1 :=
by
  sorry

end exists_x_for_ax2_plus_2x_plus_a_lt_0_l197_197880


namespace alice_commute_distance_l197_197695

noncomputable def office_distance_commute (commute_time_regular commute_time_holiday : ℝ) (speed_increase : ℝ) : ℝ := 
  let v := commute_time_regular * ((commute_time_regular + speed_increase) / commute_time_holiday - speed_increase)
  commute_time_regular * v

theorem alice_commute_distance : 
  office_distance_commute 0.5 0.3 12 = 9 := 
sorry

end alice_commute_distance_l197_197695


namespace batsman_inning_problem_l197_197093

-- Define the problem in Lean 4
theorem batsman_inning_problem (n R : ℕ) (h1 : R = 55 * n) (h2 : R + 110 = 60 * (n + 1)) : n + 1 = 11 := 
  sorry

end batsman_inning_problem_l197_197093


namespace possible_triangle_perimeters_l197_197167

theorem possible_triangle_perimeters :
  {p | ∃ (a b c : ℝ), ((a = 3 ∨ a = 6) ∧ (b = 3 ∨ b = 6) ∧ (c = 3 ∨ c = 6)) ∧
                        (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
                        p = a + b + c} = {9, 15, 18} :=
by
  sorry

end possible_triangle_perimeters_l197_197167


namespace examination_duration_in_hours_l197_197159

theorem examination_duration_in_hours 
  (total_questions : ℕ)
  (type_A_questions : ℕ)
  (time_for_A_problems : ℝ) 
  (time_ratio_A_to_B : ℝ)
  (total_time_for_A : ℝ) 
  (total_time : ℝ) :
  total_questions = 200 → 
  type_A_questions = 15 → 
  time_ratio_A_to_B = 2 → 
  total_time_for_A = 25.116279069767444 →
  total_time = (total_time_for_A + 185 * (25.116279069767444 / 15 / 2)) → 
  total_time / 60 = 3 :=
by sorry

end examination_duration_in_hours_l197_197159


namespace focal_chord_length_perpendicular_l197_197483

theorem focal_chord_length_perpendicular (x1 y1 x2 y2 : ℝ)
  (h_parabola : y1^2 = 4 * x1 ∧ y2^2 = 4 * x2)
  (h_perpendicular : x1 = x2) :
  abs (y1 - y2) = 4 :=
by sorry

end focal_chord_length_perpendicular_l197_197483


namespace quadratic_min_value_l197_197925

theorem quadratic_min_value :
  ∃ x : ℝ, (∀ y : ℝ, y = x^2 - 4 * x + 7 → y ≥ 3) ∧ (x = 2 → (x^2 - 4 * x + 7 = 3)) :=
by
  sorry

end quadratic_min_value_l197_197925


namespace problem_concentric_circles_chord_probability_l197_197609

open ProbabilityTheory

noncomputable def probability_chord_intersects_inner_circle
  (r1 r2 : ℝ) (h : r1 < r2) : ℝ :=
1/6

theorem problem_concentric_circles_chord_probability :
  probability_chord_intersects_inner_circle 1.5 3 
  (by norm_num) = 1/6 :=
sorry

end problem_concentric_circles_chord_probability_l197_197609


namespace hyperbola_foci_l197_197583

-- Define the conditions and the question
def hyperbola_equation (x y : ℝ) : Prop := 
  x^2 - 4 * y^2 - 6 * x + 24 * y - 11 = 0

-- The foci of the hyperbola 
def foci (x1 x2 y1 y2 : ℝ) : Prop := 
  (x1, y1) = (3, 3 + 2 * Real.sqrt 5) ∨ (x2, y2) = (3, 3 - 2 * Real.sqrt 5)

-- The proof statement
theorem hyperbola_foci :
  ∃ x1 x2 y1 y2 : ℝ, hyperbola_equation x1 y1 ∧ foci x1 x2 y1 y2 :=
sorry

end hyperbola_foci_l197_197583


namespace sand_art_l197_197474

theorem sand_art (len_blue_rect : ℕ) (area_blue_rect : ℕ) (side_red_square : ℕ) (sand_per_sq_inch : ℕ) (h1 : len_blue_rect = 7) (h2 : area_blue_rect = 42) (h3 : side_red_square = 5) (h4 : sand_per_sq_inch = 3) :
  (area_blue_rect * sand_per_sq_inch) + (side_red_square * side_red_square * sand_per_sq_inch) = 201 :=
by
  sorry

end sand_art_l197_197474


namespace find_time_for_compound_interest_l197_197915

noncomputable def compound_interest_time 
  (A P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem find_time_for_compound_interest :
  compound_interest_time 500 453.51473922902494 0.05 1 = 2 :=
sorry

end find_time_for_compound_interest_l197_197915


namespace find_solutions_l197_197699

theorem find_solutions (a m n : ℕ) (h : a > 0) (h₁ : m > 0) (h₂ : n > 0) :
  (a^m + 1) ∣ (a + 1)^n → 
  ((a = 1 ∧ True) ∨ (True ∧ m = 1) ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2)) :=
by sorry

end find_solutions_l197_197699


namespace parabola_vertex_l197_197979

theorem parabola_vertex (y x : ℝ) : y^2 - 4*y + 3*x + 7 = 0 → (x = -1 ∧ y = 2) := 
sorry

end parabola_vertex_l197_197979


namespace seventh_term_geometric_sequence_l197_197320

theorem seventh_term_geometric_sequence (a : ℝ) (a3 : ℝ) (r : ℝ) (n : ℕ) (term : ℕ → ℝ)
    (h_a : a = 3)
    (h_a3 : a3 = 3 / 64)
    (h_term : ∀ n, term n = a * r ^ (n - 1))
    (h_r : r = 1 / 8) :
    term 7 = 3 / 262144 :=
by
  sorry

end seventh_term_geometric_sequence_l197_197320


namespace min_value_inequality_l197_197011

noncomputable def minValue : ℝ := 17 / 2

theorem min_value_inequality (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_cond : a + 2 * b = 1) :
  a^2 + 4 * b^2 + 1 / (a * b) = minValue := 
by
  sorry

end min_value_inequality_l197_197011


namespace prove_geomSeqSumFirst3_l197_197012

noncomputable def geomSeqSumFirst3 {a₁ a₆ : ℕ} (h₁ : a₁ = 1) (h₂ : a₆ = 32) : ℕ :=
  let r := 2 -- since r^5 = 32 which means r = 2
  let S3 := a₁ * (1 - r^3) / (1 - r)
  S3

theorem prove_geomSeqSumFirst3 : 
  geomSeqSumFirst3 (h₁ : 1 = 1) (h₂ : 32 = 32) = 7 := by
  sorry

end prove_geomSeqSumFirst3_l197_197012


namespace first_batch_price_is_50_max_number_of_type_a_tools_l197_197174

-- Define the conditions
def first_batch_cost : Nat := 2000
def second_batch_cost : Nat := 2200
def price_increase : Nat := 5
def max_total_cost : Nat := 2500
def type_b_cost : Nat := 40
def total_third_batch : Nat := 50

-- First batch price per tool
theorem first_batch_price_is_50 (x : Nat) (h1 : first_batch_cost * (x + price_increase) = second_batch_cost * x) :
  x = 50 :=
sorry

-- Second batch price per tool & maximum type A tools in third batch
theorem max_number_of_type_a_tools (y : Nat)
  (h2 : 55 * y + type_b_cost * (total_third_batch - y) ≤ max_total_cost) :
  y ≤ 33 :=
sorry

end first_batch_price_is_50_max_number_of_type_a_tools_l197_197174


namespace sin_2B_value_l197_197779

-- Define the triangle's internal angles and the tangent of angles
variables (A B C : ℝ) 

-- Given conditions from the problem
def tan_sequence (tanA tanB tanC : ℝ) : Prop :=
  tanA = (1/2) * tanB ∧
  tanC = (3/2) * tanB ∧
  2 * tanB = tanC + tanB + (tanC - tanA)

-- The statement to be proven
theorem sin_2B_value (h : tan_sequence (Real.tan A) (Real.tan B) (Real.tan C)) :
  Real.sin (2 * B) = 4 / 5 :=
sorry

end sin_2B_value_l197_197779


namespace book_set_cost_l197_197790

theorem book_set_cost (charge_per_sqft : ℝ) (lawn_length lawn_width : ℝ) (num_lawns : ℝ) (additional_area : ℝ) (total_cost : ℝ) :
  charge_per_sqft = 0.10 ∧ lawn_length = 20 ∧ lawn_width = 15 ∧ num_lawns = 3 ∧ additional_area = 600 ∧ total_cost = 150 →
  (num_lawns * (lawn_length * lawn_width) * charge_per_sqft + additional_area * charge_per_sqft = total_cost) :=
by
  sorry

end book_set_cost_l197_197790


namespace largest_n_satisfies_l197_197493

noncomputable def sin_plus_cos_bound (n : ℕ) (x : ℝ) : Prop :=
  (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (2 * Real.sqrt n)

theorem largest_n_satisfies :
  ∃ (n : ℕ), (∀ x : ℝ, sin_plus_cos_bound n x) ∧
  ∀ m : ℕ, (∀ x : ℝ, sin_plus_cos_bound m x) → m ≤ 2 := 
sorry

end largest_n_satisfies_l197_197493


namespace area_of_rhombus_l197_197589

noncomputable def diagonal_length_1 : ℕ := 30
noncomputable def diagonal_length_2 : ℕ := 14

theorem area_of_rhombus (d1 d2 : ℕ) (h1 : d1 = diagonal_length_1) (h2 : d2 = diagonal_length_2) : 
  (d1 * d2) / 2 = 210 :=
by 
  rw [h1, h2]
  sorry

end area_of_rhombus_l197_197589


namespace conditions_necessary_sufficient_l197_197528

variables (p q r s : Prop)

theorem conditions_necessary_sufficient :
  ((p → r) ∧ (¬ (r → p)) ∧ (q → r) ∧ (s → r) ∧ (q → s)) →
  ((s ↔ q) ∧ ((p → q) ∧ ¬ (q → p)) ∧ ((¬ p → ¬ s) ∧ ¬ (¬ s → ¬ p))) := by
  sorry

end conditions_necessary_sufficient_l197_197528


namespace rain_probability_weekend_l197_197576

theorem rain_probability_weekend :
  let p_rain_F := 0.60
  let p_rain_S := 0.70
  let p_rain_U := 0.40
  let p_no_rain_F := 1 - p_rain_F
  let p_no_rain_S := 1 - p_rain_S
  let p_no_rain_U := 1 - p_rain_U
  let p_no_rain_all_days := p_no_rain_F * p_no_rain_S * p_no_rain_U
  let p_rain_at_least_one_day := 1 - p_no_rain_all_days
  (p_rain_at_least_one_day * 100 = 92.8) := sorry

end rain_probability_weekend_l197_197576


namespace fraction_to_decimal_l197_197663

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l197_197663


namespace polynomial_real_roots_abs_c_geq_2_l197_197646

-- Definition of the polynomial P(x)
def P (x : ℝ) (a b c : ℝ) : ℝ := x^6 + a*x^5 + b*x^4 + c*x^3 + b*x^2 + a*x + 1

-- Statement of the problem: Given that P(x) has six distinct real roots, prove |c| ≥ 2
theorem polynomial_real_roots_abs_c_geq_2 (a b c : ℝ) :
  (∃ r1 r2 r3 r4 r5 r6 : ℝ, r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r1 ≠ r5 ∧ r1 ≠ r6 ∧
                           r2 ≠ r3 ∧ r2 ≠ r4 ∧ r2 ≠ r5 ∧ r2 ≠ r6 ∧
                           r3 ≠ r4 ∧ r3 ≠ r5 ∧ r3 ≠ r6 ∧
                           r4 ≠ r5 ∧ r4 ≠ r6 ∧
                           r5 ≠ r6 ∧
                           P r1 a b c = 0 ∧ P r2 a b c = 0 ∧ P r3 a b c = 0 ∧
                           P r4 a b c = 0 ∧ P r5 a b c = 0 ∧ P r6 a b c = 0) →
  |c| ≥ 2 := by
  sorry

end polynomial_real_roots_abs_c_geq_2_l197_197646


namespace partial_fraction_product_l197_197125

theorem partial_fraction_product :
  ∃ (A B C : ℚ), 
  (∀ x : ℚ, x ≠ 1 ∧ x ≠ -3 ∧ x ≠ 4 → 
    (x^2 - 4) / (x^3 + x^2 - 11 * x - 13) = A / (x - 1) + B / (x + 3) + C / (x - 4)) ∧
  A * B * C = 5 / 196 :=
sorry

end partial_fraction_product_l197_197125


namespace simplify_sqrt_expr_l197_197740

-- We need to prove that simplifying √(5 - 2√6) is equal to √3 - √2.
theorem simplify_sqrt_expr : 
  Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 :=
by 
  sorry

end simplify_sqrt_expr_l197_197740


namespace value_a_squared_plus_b_squared_l197_197142

-- Defining the problem with the given conditions
theorem value_a_squared_plus_b_squared (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 6) : a^2 + b^2 = 21 :=
by
  sorry

end value_a_squared_plus_b_squared_l197_197142


namespace students_taking_neither_l197_197546

variable (total_students math_students physics_students both_students : ℕ)
variable (h1 : total_students = 80)
variable (h2 : math_students = 50)
variable (h3 : physics_students = 40)
variable (h4 : both_students = 25)

theorem students_taking_neither (h1 : total_students = 80)
    (h2 : math_students = 50)
    (h3 : physics_students = 40)
    (h4 : both_students = 25) :
    total_students - (math_students - both_students + physics_students - both_students + both_students) = 15 :=
by
    sorry

end students_taking_neither_l197_197546


namespace Manny_lasagna_pieces_l197_197081

-- Define variables and conditions
variable (M : ℕ) -- Manny's desired number of pieces
variable (A : ℕ := 0) -- Aaron's pieces
variable (K : ℕ := 2 * M) -- Kai's pieces
variable (R : ℕ := M / 2) -- Raphael's pieces
variable (L : ℕ := 2 + R) -- Lisa's pieces

-- Prove that Manny wants 1 piece of lasagna
theorem Manny_lasagna_pieces (M : ℕ) (A : ℕ := 0) (K : ℕ := 2 * M) (R : ℕ := M / 2) (L : ℕ := 2 + R) 
  (h : M + A + K + R + L = 6) : M = 1 :=
by
  sorry

end Manny_lasagna_pieces_l197_197081


namespace sum_of_integers_remainder_l197_197021

-- Definitions of the integers and their properties
variables (a b c : ℕ)

-- Conditions
axiom h1 : a % 53 = 31
axiom h2 : b % 53 = 17
axiom h3 : c % 53 = 8
axiom h4 : a % 5 = 0

-- The proof goal
theorem sum_of_integers_remainder :
  (a + b + c) % 53 = 3 :=
by
  sorry -- Proof to be provided

end sum_of_integers_remainder_l197_197021


namespace new_person_weight_is_55_l197_197392

variable (W : ℝ) -- Total weight of the original 8 people
variable (new_person_weight : ℝ) -- Weight of the new person
variable (avg_increase : ℝ := 2.5) -- The average weight increase

-- Given conditions
def condition (W new_person_weight : ℝ) : Prop :=
  new_person_weight = W + (8 * avg_increase) + 35 - W

-- The proof statement
theorem new_person_weight_is_55 (W : ℝ) : (new_person_weight = 55) :=
by
  sorry

end new_person_weight_is_55_l197_197392


namespace find_distance_PQ_of_polar_coords_l197_197552

theorem find_distance_PQ_of_polar_coords (α β : ℝ) (h : β - α = 2 * Real.pi / 3) :
  let P := (5, α)
  let Q := (12, β)
  dist P Q = Real.sqrt 229 :=
by
  sorry

end find_distance_PQ_of_polar_coords_l197_197552


namespace function_symmetry_implies_even_l197_197239

theorem function_symmetry_implies_even (f : ℝ → ℝ) (h1 : ∃ x, f x ≠ 0)
  (h2 : ∀ x y, f x = y ↔ -f (-x) = -y) : ∀ x, f x = f (-x) :=
by
  sorry

end function_symmetry_implies_even_l197_197239


namespace additional_amount_needed_l197_197937

-- Define the amounts spent on shampoo, conditioner, and lotion
def shampoo_cost : ℝ := 10.00
def conditioner_cost : ℝ := 10.00
def lotion_cost_per_bottle : ℝ := 6.00
def lotion_quantity : ℕ := 3

-- Define the amount required for free shipping
def free_shipping_threshold : ℝ := 50.00

-- Calculate the total amount spent
def total_spent : ℝ := shampoo_cost + conditioner_cost + (lotion_quantity * lotion_cost_per_bottle)

-- Define the additional amount needed for free shipping
def additional_needed_for_shipping : ℝ := free_shipping_threshold - total_spent

-- The final goal to prove
theorem additional_amount_needed : additional_needed_for_shipping = 12.00 :=
by
  sorry

end additional_amount_needed_l197_197937


namespace correct_statements_l197_197901

theorem correct_statements (f : ℝ → ℝ) (t : ℝ)
  (h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0)
  (h2 : (∀ x : ℝ, f x = f (-x)) ∧ (∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → f x1 > f x2) ∧ f (-2) = 0)
  (h3 : ∀ x : ℝ, f (-x) = -f x)
  (h4 : ∀ x : ℝ, f (x - t) = f (x + t)) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 > f x2 ↔ x1 < x2) ∧
  (∀ x : ℝ, f x - f (|x|) = - (f (-x) - f (|x|))) :=
by
  sorry

end correct_statements_l197_197901


namespace smallest_c_value_l197_197700

theorem smallest_c_value (c d : ℝ) (h_nonneg_c : 0 ≤ c) (h_nonneg_d : 0 ≤ d)
  (h_eq_cos : ∀ x : ℤ, Real.cos (c * x - d) = Real.cos (35 * x)) :
  c = 35 := by
  sorry

end smallest_c_value_l197_197700


namespace original_sugar_amount_l197_197207

theorem original_sugar_amount (f : ℕ) (s t r : ℕ) (h1 : f = 5) (h2 : r = 10) (h3 : t = 14) (h4 : f * 2 = r):
  s = t / 2 := sorry

end original_sugar_amount_l197_197207


namespace intersection_of_A_and_B_l197_197386

def set_A : Set ℝ := {x : ℝ | x^2 - 5 * x + 6 > 0}
def set_B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | x < 1} :=
sorry

end intersection_of_A_and_B_l197_197386


namespace numerator_of_first_fraction_l197_197240

theorem numerator_of_first_fraction (y : ℝ) (h : y > 0) (x : ℝ) 
  (h_eq : (x / y) * y + (3 * y) / 10 = 0.35 * y) : x = 32 := 
by
  sorry

end numerator_of_first_fraction_l197_197240


namespace calculate_total_area_of_figure_l197_197874

-- Defining the lengths of the segments according to the problem conditions.
def length_1 : ℕ := 8
def length_2 : ℕ := 6
def length_3 : ℕ := 3
def length_4 : ℕ := 5
def length_5 : ℕ := 2
def length_6 : ℕ := 4

-- Using the given lengths to compute the areas of the smaller rectangles
def area_A : ℕ := length_1 * length_2
def area_B : ℕ := length_4 * (10 - 6)
def area_C : ℕ := (6 - 3) * (15 - 10)

-- The total area of the figure is the sum of the areas of the smaller rectangles
def total_area : ℕ := area_A + area_B + area_C

-- The statement to prove
theorem calculate_total_area_of_figure : total_area = 83 := by
  -- Proof goes here
  sorry

end calculate_total_area_of_figure_l197_197874


namespace area_of_three_layer_cover_l197_197920

-- Define the hall dimensions
def hall_width : ℕ := 10
def hall_height : ℕ := 10

-- Define the dimensions of the carpets
def carpet1_width : ℕ := 6
def carpet1_height : ℕ := 8
def carpet2_width : ℕ := 6
def carpet2_height : ℕ := 6
def carpet3_width : ℕ := 5
def carpet3_height : ℕ := 7

-- Theorem to prove area covered by the carpets in three layers
theorem area_of_three_layer_cover : 
  ∀ (w1 w2 w3 h1 h2 h3 : ℕ), w1 = carpet1_width → h1 = carpet1_height → w2 = carpet2_width → h2 = carpet2_height → w3 = carpet3_width → h3 = carpet3_height → 
  ∃ (area : ℕ), area = 6 :=
by
  intros w1 w2 w3 h1 h2 h3 hw1 hw2 hw3 hh1 hh2 hh3
  exact ⟨6, rfl⟩

#check area_of_three_layer_cover

end area_of_three_layer_cover_l197_197920


namespace arithmetic_sequence_index_l197_197120

theorem arithmetic_sequence_index {a : ℕ → ℕ} (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 3) (h₃ : a n = 2014) : n = 672 :=
by
  sorry

end arithmetic_sequence_index_l197_197120


namespace compute_expression_l197_197917

-- Definition of the imaginary unit i
class ImaginaryUnit (i : ℂ) where
  I_square : i * i = -1

-- Definition of non-zero real number a
variable (a : ℝ) (h_a : a ≠ 0)

-- Theorem to prove the equivalence
theorem compute_expression (i : ℂ) [ImaginaryUnit i] :
  (a * i - i⁻¹)⁻¹ = -i / (a + 1) :=
by
  sorry

end compute_expression_l197_197917


namespace exists_m_divisible_l197_197582

-- Define the function f
def f (x : ℕ) : ℕ := 3 * x + 2

-- Define the 100th iterate of f
def f_iter (n : ℕ) : ℕ := 3^n

-- Define the condition that needs to be proven
theorem exists_m_divisible : ∃ m : ℕ, (3^100 * m + (3^100 - 1)) % 1988 = 0 :=
sorry

end exists_m_divisible_l197_197582


namespace scientific_notation_of_3900000000_l197_197704

theorem scientific_notation_of_3900000000 : 3900000000 = 3.9 * 10^9 :=
by 
  sorry

end scientific_notation_of_3900000000_l197_197704


namespace clara_sells_total_cookies_l197_197825

theorem clara_sells_total_cookies :
  let cookies_per_box_1 := 12
  let cookies_per_box_2 := 20
  let cookies_per_box_3 := 16
  let cookies_per_box_4 := 18
  let cookies_per_box_5 := 22

  let boxes_sold_1 := 50.5
  let boxes_sold_2 := 80.25
  let boxes_sold_3 := 70.75
  let boxes_sold_4 := 65.5
  let boxes_sold_5 := 55.25

  let total_cookies_1 := cookies_per_box_1 * boxes_sold_1
  let total_cookies_2 := cookies_per_box_2 * boxes_sold_2
  let total_cookies_3 := cookies_per_box_3 * boxes_sold_3
  let total_cookies_4 := cookies_per_box_4 * boxes_sold_4
  let total_cookies_5 := cookies_per_box_5 * boxes_sold_5

  let total_cookies := total_cookies_1 + total_cookies_2 + total_cookies_3 + total_cookies_4 + total_cookies_5

  total_cookies = 5737.5 :=
by
  sorry

end clara_sells_total_cookies_l197_197825


namespace perpendicular_lines_slope_l197_197862

theorem perpendicular_lines_slope (a : ℝ) (h1 :  a * (a + 2) = -1) : a = -1 :=
by 
-- Perpendicularity condition given
sorry

end perpendicular_lines_slope_l197_197862


namespace greatest_prime_factor_187_l197_197562

theorem greatest_prime_factor_187 : ∃ p : ℕ, Prime p ∧ p ∣ 187 ∧ ∀ q : ℕ, Prime q ∧ q ∣ 187 → p ≥ q := by
  sorry

end greatest_prime_factor_187_l197_197562


namespace sum_of_arithmetic_sequence_l197_197586

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : a 5 + a 4 = 18) (hS_def : ∀ n, S n = n * (a 1 + a n) / 2) : S 8 = 72 := 
sorry

end sum_of_arithmetic_sequence_l197_197586


namespace water_depth_correct_l197_197105

noncomputable def water_depth (ron_height : ℝ) (dean_shorter_by : ℝ) : ℝ :=
  let dean_height := ron_height - dean_shorter_by
  2.5 * dean_height + 3

theorem water_depth_correct :
  water_depth 14.2 8.3 = 17.75 :=
by
  let ron_height := 14.2
  let dean_shorter_by := 8.3
  let dean_height := ron_height - dean_shorter_by
  let depth := 2.5 * dean_height + 3
  simp [water_depth, dean_height, depth]
  sorry

end water_depth_correct_l197_197105


namespace simplify_fraction_l197_197338

theorem simplify_fraction (a b : ℕ) (h₁ : a = 84) (h₂ : b = 144) :
  a / gcd a b = 7 ∧ b / gcd a b = 12 := 
by
  sorry

end simplify_fraction_l197_197338


namespace division_addition_example_l197_197176

theorem division_addition_example : 12 / (1 / 6) + 3 = 75 := by
  sorry

end division_addition_example_l197_197176


namespace triangle_area_given_conditions_l197_197481

theorem triangle_area_given_conditions (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6) (h2 : C = Real.pi / 3) : 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end triangle_area_given_conditions_l197_197481


namespace kolya_or_leva_l197_197089

theorem kolya_or_leva (k l : ℝ) (hkl : k > 0) (hll : l > 0) : 
  (k > l → ∃ a b c : ℝ, a = l + (2 / 3) * (k - l) ∧ b = (1 / 6) * (k - l) ∧ c = (1 / 6) * (k - l) ∧ a > b + c + l ∧ ¬(a < b + c + a)) ∨ 
  (k ≤ l → ∃ k1 k2 k3 : ℝ, k1 ≥ k2 ∧ k2 ≥ k3 ∧ k = k1 + k2 + k3 ∧ ∃ a' b' c' : ℝ, a' = k1 ∧ b' = (l - k1) / 2 ∧ c' = (l - k1) / 2 ∧ a' + a' > k2 ∧ b' + b' > k3) :=
by sorry

end kolya_or_leva_l197_197089


namespace larger_angle_at_3_30_l197_197495

def hour_hand_angle_3_30 : ℝ := 105.0
def minute_hand_angle_3_30 : ℝ := 180.0
def smaller_angle_between_hands : ℝ := abs (minute_hand_angle_3_30 - hour_hand_angle_3_30)
def larger_angle_between_hands : ℝ := 360.0 - smaller_angle_between_hands

theorem larger_angle_at_3_30 :
  larger_angle_between_hands = 285.0 := 
  sorry

end larger_angle_at_3_30_l197_197495


namespace hotdogs_needed_l197_197321

theorem hotdogs_needed 
  (ella_hotdogs : ℕ) (emma_hotdogs : ℕ)
  (luke_multiple : ℕ) (hunter_multiple : ℚ)
  (h_ella : ella_hotdogs = 2)
  (h_emma : emma_hotdogs = 2)
  (h_luke : luke_multiple = 2)
  (h_hunter : hunter_multiple = (3/2)) :
  ella_hotdogs + emma_hotdogs + luke_multiple * (ella_hotdogs + emma_hotdogs) + hunter_multiple * (ella_hotdogs + emma_hotdogs) = 18 := by
    sorry

end hotdogs_needed_l197_197321


namespace find_primes_l197_197230

theorem find_primes (p : ℕ) (hp : Nat.Prime p) :
  (∃ a b c k : ℤ, a^2 + b^2 + c^2 = p ∧ a^4 + b^4 + c^4 = k * p) ↔ (p = 2 ∨ p = 3) :=
by
  sorry

end find_primes_l197_197230


namespace cos_60_eq_half_l197_197232

theorem cos_60_eq_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_60_eq_half_l197_197232


namespace sum_term_ratio_equals_four_l197_197533

variable {a_n : ℕ → ℝ} -- The arithmetic sequence a_n
variable {S_n : ℕ → ℝ} -- The sum of the first n terms S_n
variable {d : ℝ} -- The common difference of the sequence
variable {a_1 : ℝ} -- The first term of the sequence

-- The conditions as hypotheses
axiom a_n_formula (n : ℕ) : a_n n = a_1 + (n - 1) * d
axiom S_n_formula (n : ℕ) : S_n n = n * (a_1 + (n - 1) * d / 2)
axiom non_zero_d : d ≠ 0
axiom condition_a10_S4 : a_n 10 = S_n 4

-- The proof statement
theorem sum_term_ratio_equals_four : (S_n 8) / (a_n 9) = 4 :=
by
  sorry

end sum_term_ratio_equals_four_l197_197533


namespace total_spectators_after_halftime_l197_197900

theorem total_spectators_after_halftime
  (initial_boys : ℕ := 300)
  (initial_girls : ℕ := 400)
  (initial_adults : ℕ := 300)
  (total_people : ℕ := 1000)
  (quarter_boys_leave_fraction : ℚ := 1 / 4)
  (quarter_girls_leave_fraction : ℚ := 1 / 8)
  (quarter_adults_leave_fraction : ℚ := 1 / 5)
  (halftime_new_boys : ℕ := 50)
  (halftime_new_girls : ℕ := 90)
  (halftime_adults_leave_fraction : ℚ := 3 / 100) :
  let boys_after_first_quarter := initial_boys - initial_boys * quarter_boys_leave_fraction
  let girls_after_first_quarter := initial_girls - initial_girls * quarter_girls_leave_fraction
  let adults_after_first_quarter := initial_adults - initial_adults * quarter_adults_leave_fraction
  let boys_after_halftime := boys_after_first_quarter + halftime_new_boys
  let girls_after_halftime := girls_after_first_quarter + halftime_new_girls
  let adults_after_halftime := adults_after_first_quarter * (1 - halftime_adults_leave_fraction)
  boys_after_halftime + girls_after_halftime + adults_after_halftime = 948 :=
by sorry

end total_spectators_after_halftime_l197_197900


namespace part1_solution_count_part2_solution_count_l197_197908

theorem part1_solution_count :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), solutions.card = 7 ∧
    ∀ (m n r : ℕ), (m, n, r) ∈ solutions ↔ mn + nr + mr = 2 * (m + n + r) := sorry

theorem part2_solution_count (k : ℕ) (h : 1 < k) :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), solutions.card ≥ 3 * k + 1 ∧
    ∀ (m n r : ℕ), (m, n, r) ∈ solutions ↔ mn + nr + mr = k * (m + n + r) := sorry

end part1_solution_count_part2_solution_count_l197_197908


namespace tan_alpha_plus_pi_over_4_equals_3_over_22_l197_197061

theorem tan_alpha_plus_pi_over_4_equals_3_over_22
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 :=
sorry

end tan_alpha_plus_pi_over_4_equals_3_over_22_l197_197061


namespace number_of_revolutions_wheel_half_mile_l197_197679

theorem number_of_revolutions_wheel_half_mile :
  let diameter := 10 * (1 : ℝ)
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let half_mile_in_feet := 2640
  (half_mile_in_feet / circumference) = 264 / Real.pi :=
by
  let diameter := 10 * (1 : ℝ)
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let half_mile_in_feet := 2640
  have h : (half_mile_in_feet / circumference) = 264 / Real.pi := by
    sorry
  exact h

end number_of_revolutions_wheel_half_mile_l197_197679


namespace bus_passengers_remaining_l197_197711

theorem bus_passengers_remaining (initial_passengers : ℕ := 22) 
                                 (boarding_alighting1 : (ℤ × ℤ) := (4, -8)) 
                                 (boarding_alighting2 : (ℤ × ℤ) := (6, -5)) : 
                                 (initial_passengers : ℤ) + 
                                 (boarding_alighting1.fst + boarding_alighting1.snd) + 
                                 (boarding_alighting2.fst + boarding_alighting2.snd) = 19 :=
by
  sorry

end bus_passengers_remaining_l197_197711


namespace stan_run_duration_l197_197258

def run_duration : ℕ := 100

def num_3_min_songs : ℕ := 10
def num_2_min_songs : ℕ := 15
def time_per_3_min_song : ℕ := 3
def time_per_2_min_song : ℕ := 2
def additional_time_needed : ℕ := 40

theorem stan_run_duration :
  (num_3_min_songs * time_per_3_min_song) + (num_2_min_songs * time_per_2_min_song) + additional_time_needed = run_duration := by
  sorry

end stan_run_duration_l197_197258


namespace calculate_subtraction_l197_197809

def base9_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 81 + ((n / 10) % 10) * 9 + (n % 10)

def base6_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

theorem calculate_subtraction : base9_to_base10 324 - base6_to_base10 231 = 174 :=
  by sorry

end calculate_subtraction_l197_197809


namespace walkway_area_correct_l197_197135

/-- Definitions as per problem conditions --/
def bed_length : ℕ := 8
def bed_width : ℕ := 3
def walkway_bed_width : ℕ := 2
def walkway_row_width : ℕ := 1
def num_beds_in_row : ℕ := 3
def num_rows : ℕ := 4

/-- Total dimensions including walkways --/
def total_width := num_beds_in_row * bed_length + (num_beds_in_row + 1) * walkway_bed_width
def total_height := num_rows * bed_width + (num_rows + 1) * walkway_row_width

/-- Total areas --/
def total_area := total_width * total_height
def bed_area := bed_length * bed_width
def total_bed_area := num_beds_in_row * num_rows * bed_area
def walkway_area := total_area - total_bed_area

theorem walkway_area_correct : walkway_area = 256 := by
  /- Import necessary libraries and skip the proof -/
  sorry

end walkway_area_correct_l197_197135


namespace complete_square_solution_l197_197984

theorem complete_square_solution :
  ∀ x : ℝ, x^2 - 4 * x - 22 = 0 → (x - 2)^2 = 26 :=
by
  intro x h
  sorry

end complete_square_solution_l197_197984


namespace intersect_not_A_B_l197_197191

open Set

-- Define the universal set U
def U := ℝ

-- Define set A
def A := {x : ℝ | x ≤ 3}

-- Define set B
def B := {x : ℝ | x ≤ 6}

-- Define the complement of A in U
def not_A := {x : ℝ | x > 3}

-- The proof problem
theorem intersect_not_A_B :
  (not_A ∩ B) = {x : ℝ | 3 < x ∧ x ≤ 6} :=
sorry

end intersect_not_A_B_l197_197191


namespace cos_in_third_quadrant_l197_197759

theorem cos_in_third_quadrant (B : ℝ) (h_sin_B : Real.sin B = -5/13) (h_quadrant : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 :=
by
  sorry

end cos_in_third_quadrant_l197_197759


namespace nesbitt_inequality_l197_197525

theorem nesbitt_inequality {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) ∧ (a = b ∧ b = c → a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2) :=
sorry

end nesbitt_inequality_l197_197525


namespace mr_callen_total_loss_l197_197588

noncomputable def total_loss : ℤ :=
  let bought_paintings_price := 15 * 60
  let bought_wooden_toys_price := 12 * 25
  let bought_handmade_hats_price := 20 * 15
  let total_bought_price := bought_paintings_price + bought_wooden_toys_price + bought_handmade_hats_price
  let sold_paintings_price := 15 * (60 - (60 * 18 / 100))
  let sold_wooden_toys_price := 12 * (25 - (25 * 25 / 100))
  let sold_handmade_hats_price := 20 * (15 - (15 * 10 / 100))
  let total_sold_price := sold_paintings_price + sold_wooden_toys_price + sold_handmade_hats_price
  total_bought_price - total_sold_price

theorem mr_callen_total_loss : total_loss = 267 := by
  sorry

end mr_callen_total_loss_l197_197588


namespace value_of_p_l197_197751

variable (m n p : ℝ)

-- The conditions from the problem
def first_point_on_line := m = (n / 6) - (2 / 5)
def second_point_on_line := m + p = ((n + 18) / 6) - (2 / 5)

-- The theorem to prove
theorem value_of_p (h1 : first_point_on_line m n) (h2 : second_point_on_line m n p) : p = 3 :=
  sorry

end value_of_p_l197_197751


namespace triangle_angle_C_triangle_max_area_l197_197022

noncomputable def cos (θ : Real) : Real := sorry
noncomputable def sin (θ : Real) : Real := sorry

theorem triangle_angle_C (a b c : Real) (A B C : Real) (h1: 0 < A ∧ A < Real.pi)
  (h2: 0 < B ∧ B < Real.pi) (h3: 0 < C ∧ C < Real.pi)
  (h4: (2 * a + b) * cos C + c * cos B = 0) : C = (2 * Real.pi) / 3 :=
sorry

theorem triangle_max_area (a b c : Real) (A B C : Real) (h1: 0 < A ∧ A < Real.pi)
  (h2: 0 < B ∧ B < Real.pi) (h3: 0 < C ∧ C < Real.pi)
  (h4: (2 * a + b) * cos C + c * cos B = 0) (hc : c = 6)
  (hC : C = (2 * Real.pi) / 3) : 
  ∃ (S : Real), S = 3 * Real.sqrt 3 := 
sorry

end triangle_angle_C_triangle_max_area_l197_197022


namespace consecutive_negative_product_sum_l197_197343

theorem consecutive_negative_product_sum (n : ℤ) (h : n * (n + 1) = 2850) : n + (n + 1) = -107 :=
sorry

end consecutive_negative_product_sum_l197_197343


namespace seashells_total_correct_l197_197744

-- Define the initial counts for Henry, John, and Adam.
def initial_seashells_Henry : ℕ := 11
def initial_seashells_John : ℕ := 24
def initial_seashells_Adam : ℕ := 17

-- Define the total initial seashells collected by all.
def total_initial_seashells : ℕ := 83

-- Calculate Leo's initial seashells.
def initial_seashells_Leo : ℕ := total_initial_seashells - (initial_seashells_Henry + initial_seashells_John + initial_seashells_Adam)

-- Define the changes occurred when they returned home.
def extra_seashells_Henry : ℕ := 3
def given_away_seashells_John : ℕ := 5
def percentage_given_away_Leo : ℕ := 40
def extra_seashells_Leo : ℕ := 5

-- Define the final number of seashells each person has.
def final_seashells_Henry : ℕ := initial_seashells_Henry + extra_seashells_Henry
def final_seashells_John : ℕ := initial_seashells_John - given_away_seashells_John
def given_away_seashells_Leo : ℕ := (initial_seashells_Leo * percentage_given_away_Leo) / 100
def final_seashells_Leo : ℕ := initial_seashells_Leo - given_away_seashells_Leo + extra_seashells_Leo
def final_seashells_Adam : ℕ := initial_seashells_Adam

-- Define the total number of seashells they have now.
def total_final_seashells : ℕ := final_seashells_Henry + final_seashells_John + final_seashells_Leo + final_seashells_Adam

-- Proposition that asserts the total number of seashells is 74.
theorem seashells_total_correct :
  total_final_seashells = 74 :=
sorry

end seashells_total_correct_l197_197744


namespace midpoint_product_l197_197883

-- Defining the endpoints of the line segment
def x1 : ℤ := 4
def y1 : ℤ := 7
def x2 : ℤ := -8
def y2 : ℤ := 9

-- Proof goal: show that the product of the coordinates of the midpoint is -16
theorem midpoint_product : ((x1 + x2) / 2) * ((y1 + y2) / 2) = -16 := 
by sorry

end midpoint_product_l197_197883


namespace find_x_l197_197026

-- Definitions for the vectors and their relationships
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)
def u (x : ℝ) : ℝ × ℝ := (a.1 + 2 * (b x).1, a.2 + 2 * (b x).2)
def v (x : ℝ) : ℝ × ℝ := (2 * a.1 - (b x).1, 2 * a.2 - (b x).2)

-- Given condition that u is parallel to v
def u_parallel_v (x : ℝ) : Prop := u x = v x

-- Prove that the value of x is 1/2
theorem find_x : ∃ x : ℝ, u_parallel_v x ∧ x = 1 / 2 := 
sorry

end find_x_l197_197026


namespace expected_sectors_pizza_l197_197739

/-- Let N be the total number of pizza slices and M be the number of slices taken randomly.
    Given N = 16 and M = 5, the expected number of sectors formed is 11/3. -/
theorem expected_sectors_pizza (N M : ℕ) (hN : N = 16) (hM : M = 5) :
  (N - M) * M / (N - 1) = 11 / 3 :=
  sorry

end expected_sectors_pizza_l197_197739


namespace value_of_c_over_b_l197_197460

def is_median (a b c : ℤ) (m : ℤ) : Prop :=
a < b ∧ b < c ∧ m = b

def in_geometric_progression (p q r : ℤ) : Prop :=
∃ k : ℤ, k ≠ 0 ∧ q = p * k ∧ r = q * k

theorem value_of_c_over_b (a b c p q r : ℤ) 
  (h1 : (a + b + c) / 3 = (b / 2))
  (h2 : a * b * c = 0)
  (h3 : a < b ∧ b < c ∧ a = 0)
  (h4 : p < q ∧ q < r ∧ r ≠ 0)
  (h5 : in_geometric_progression p q r)
  (h6 : a^2 + b^2 + c^2 = (p + q + r)^2) : 
  c / b = 2 := 
sorry

end value_of_c_over_b_l197_197460


namespace solution_set_f_less_x_plus_1_l197_197717

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_continuous : Continuous f
axiom f_at_1 : f 1 = 2
axiom f_derivative : ∀ x, deriv f x < 1

theorem solution_set_f_less_x_plus_1 : 
  ∀ x : ℝ, (f x < x + 1) ↔ (x > 1) :=
by
  sorry

end solution_set_f_less_x_plus_1_l197_197717


namespace sum_of_cubes_condition_l197_197911

theorem sum_of_cubes_condition (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 := 
by
  sorry

end sum_of_cubes_condition_l197_197911


namespace inequality_solution_set_l197_197434

theorem inequality_solution_set (x : ℝ) : 
  (x + 2) / (x - 1) ≤ 0 ↔ -2 ≤ x ∧ x < 1 := 
sorry

end inequality_solution_set_l197_197434


namespace total_cost_to_plant_flowers_l197_197050

noncomputable def flower_cost : ℕ := 9
noncomputable def clay_pot_cost : ℕ := flower_cost + 20
noncomputable def soil_bag_cost : ℕ := flower_cost - 2
noncomputable def total_cost : ℕ := flower_cost + clay_pot_cost + soil_bag_cost

theorem total_cost_to_plant_flowers : total_cost = 45 := by
  sorry

end total_cost_to_plant_flowers_l197_197050


namespace perimeter_percent_increase_l197_197169

noncomputable def side_increase (s₁ s₂_ratio s₃_ratio s₄_ratio s₅_ratio : ℝ) : ℝ :=
  let s₂ := s₂_ratio * s₁
  let s₃ := s₃_ratio * s₂
  let s₄ := s₄_ratio * s₃
  let s₅ := s₅_ratio * s₄
  s₅

theorem perimeter_percent_increase (s₁ : ℝ) (s₂_ratio s₃_ratio s₄_ratio s₅_ratio : ℝ) (P₁ := 3 * s₁)
    (P₅ := 3 * side_increase s₁ s₂_ratio s₃_ratio s₄_ratio s₅_ratio) :
    s₁ = 4 → s₂_ratio = 1.5 → s₃_ratio = 1.3 → s₄_ratio = 1.5 → s₅_ratio = 1.3 →
    P₅ = 45.63 →
    ((P₅ - P₁) / P₁) * 100 = 280.3 :=
by
  intros
  -- proof goes here
  sorry

end perimeter_percent_increase_l197_197169


namespace find_a4_l197_197238

variable (a_1 d : ℝ)

def a_n (n : ℕ) : ℝ :=
  a_1 + (n - 1) * d

axiom condition1 : (a_n a_1 d 2 + a_n a_1 d 6) / 2 = 5 * Real.sqrt 3
axiom condition2 : (a_n a_1 d 3 + a_n a_1 d 7) / 2 = 7 * Real.sqrt 3

theorem find_a4 : a_n a_1 d 4 = 5 * Real.sqrt 3 :=
by
  -- Proof should go here, but we insert "sorry" to mark it as incomplete.
  sorry

end find_a4_l197_197238


namespace shaded_region_area_correct_l197_197324

noncomputable def area_shaded_region : ℝ := 
  let side_length := 2
  let radius := 1
  let area_square := side_length^2
  let area_circle := Real.pi * radius^2
  area_square - area_circle

theorem shaded_region_area_correct : area_shaded_region = 4 - Real.pi :=
  by
    sorry

end shaded_region_area_correct_l197_197324


namespace negative_number_from_operations_l197_197448

theorem negative_number_from_operations :
  (∀ (a b : Int), a + b < 0 → a = -1 ∧ b = -3) ∧
  (∀ (a b : Int), a - b < 0 → a = 1 ∧ b = 4) ∧
  (∀ (a b : Int), a * b > 0 → a = 3 ∧ b = -2) ∧
  (∀ (a b : Int), a / b = 0 → a = 0 ∧ b = -7) :=
by
  sorry

end negative_number_from_operations_l197_197448


namespace k_values_for_perpendicular_lines_l197_197989

-- Definition of perpendicular condition for lines
def perpendicular_lines (k : ℝ) : Prop :=
  k * (k - 1) + (1 - k) * (2 * k + 3) = 0

-- Lean 4 statement representing the math proof problem
theorem k_values_for_perpendicular_lines (k : ℝ) :
  perpendicular_lines k ↔ k = -3 ∨ k = 1 :=
by
  sorry

end k_values_for_perpendicular_lines_l197_197989


namespace gcf_54_81_l197_197995

theorem gcf_54_81 : Nat.gcd 54 81 = 27 :=
by sorry

end gcf_54_81_l197_197995


namespace total_time_marco_6_laps_total_time_in_minutes_and_seconds_l197_197393

noncomputable def marco_running_time : ℕ :=
  let distance_1 := 150
  let speed_1 := 5
  let time_1 := distance_1 / speed_1

  let distance_2 := 300
  let speed_2 := 4
  let time_2 := distance_2 / speed_2

  let time_per_lap := time_1 + time_2
  let total_laps := 6
  let total_time_seconds := time_per_lap * total_laps

  total_time_seconds

theorem total_time_marco_6_laps : marco_running_time = 630 := sorry

theorem total_time_in_minutes_and_seconds : 10 * 60 + 30 = 630 := sorry

end total_time_marco_6_laps_total_time_in_minutes_and_seconds_l197_197393


namespace evaluate_expression_l197_197422

theorem evaluate_expression (b : ℝ) (hb : b = -3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b = 10 / 27 :=
by 
  -- Lean code typically begins the proof block here
  sorry  -- The proof itself is omitted

end evaluate_expression_l197_197422


namespace positive_number_square_roots_l197_197316

theorem positive_number_square_roots (m : ℝ) 
  (h : (2 * m - 1) + (2 - m) = 0) :
  (2 - m)^2 = 9 :=
by
  sorry

end positive_number_square_roots_l197_197316


namespace more_math_than_reading_l197_197146

def pages_reading := 4
def pages_math := 7

theorem more_math_than_reading : pages_math - pages_reading = 3 :=
by
  sorry

end more_math_than_reading_l197_197146


namespace prime_product_2002_l197_197505

theorem prime_product_2002 {a b c d : ℕ} (ha_prime : Prime a) (hb_prime : Prime b) (hc_prime : Prime c) (hd_prime : Prime d)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : a + c = d)
  (h2 : a * (a + b + c + d) = c * (d - b))
  (h3 : 1 + b * c + d = b * d) :
  a * b * c * d = 2002 := 
by 
  sorry

end prime_product_2002_l197_197505


namespace gcd_markers_l197_197250

variable (n1 n2 n3 : ℕ)

-- Let the markers Mary, Luis, and Ali bought be represented by n1, n2, and n3
def MaryMarkers : ℕ := 36
def LuisMarkers : ℕ := 45
def AliMarkers : ℕ := 75

theorem gcd_markers : Nat.gcd (Nat.gcd MaryMarkers LuisMarkers) AliMarkers = 3 := by
  sorry

end gcd_markers_l197_197250


namespace problem_statement_l197_197470

noncomputable def C : ℝ := 49
noncomputable def D : ℝ := 3.75

theorem problem_statement : C + D = 52.75 := by
  sorry

end problem_statement_l197_197470


namespace size_ratio_l197_197973

variable {U : ℝ} (h1 : C = 1.5 * U) (h2 : R = 4 / 3 * C)

theorem size_ratio : R = 8 / 3 * U :=
by
  sorry

end size_ratio_l197_197973


namespace license_plate_count_l197_197955

-- Define the conditions
def num_digits : ℕ := 5
def num_letters : ℕ := 2
def digit_choices : ℕ := 10
def letter_choices : ℕ := 26

-- Define the statement to prove the total number of distinct licenses plates
theorem license_plate_count : 
  (digit_choices ^ num_digits) * (letter_choices ^ num_letters) * 2 = 2704000 :=
by
  sorry

end license_plate_count_l197_197955


namespace log_sum_l197_197905

variable (m a b : ℝ)
variable (m_pos : 0 < m)
variable (m_ne_one : m ≠ 1)
variable (h1 : m^2 = a)
variable (h2 : m^3 = b)

theorem log_sum (m_pos : 0 < m) (m_ne_one : m ≠ 1) (h1 : m^2 = a) (h2 : m^3 = b) :
  2 * Real.log (a) / Real.log (m) + Real.log (b) / Real.log (m) = 7 := 
sorry

end log_sum_l197_197905


namespace lunch_customers_is_127_l197_197647

-- Define the conditions based on the given problem
def breakfast_customers : ℕ := 73
def dinner_customers : ℕ := 87
def total_customers_on_saturday : ℕ := 574
def total_customers_on_friday : ℕ := total_customers_on_saturday / 2

-- Define the variable representing the lunch customers
variable (L : ℕ)

-- State the proposition we want to prove
theorem lunch_customers_is_127 :
  breakfast_customers + L + dinner_customers = total_customers_on_friday → L = 127 := by {
  sorry
}

end lunch_customers_is_127_l197_197647


namespace ways_to_select_four_doctors_l197_197813

def num_ways_to_select_doctors (num_internists : ℕ) (num_surgeons : ℕ) (team_size : ℕ) : ℕ :=
  (Nat.choose num_internists 1 * Nat.choose num_surgeons (team_size - 1)) + 
  (Nat.choose num_internists 2 * Nat.choose num_surgeons (team_size - 2)) + 
  (Nat.choose num_internists 3 * Nat.choose num_surgeons (team_size - 3))

theorem ways_to_select_four_doctors : num_ways_to_select_doctors 5 6 4 = 310 := 
by
  sorry

end ways_to_select_four_doctors_l197_197813


namespace octal_67_equals_ternary_2001_l197_197350

def octalToDecimal (n : Nat) : Nat :=
  -- Definition of octal to decimal conversion omitted
  sorry

def decimalToTernary (n : Nat) : Nat :=
  -- Definition of decimal to ternary conversion omitted
  sorry

theorem octal_67_equals_ternary_2001 : 
  decimalToTernary (octalToDecimal 67) = 2001 :=
by
  -- Proof omitted
  sorry

end octal_67_equals_ternary_2001_l197_197350


namespace imaginary_part_of_complex_l197_197994

theorem imaginary_part_of_complex :
  let i := Complex.I
  let z := 10 * i / (3 + i)
  z.im = 3 :=
by
  sorry

end imaginary_part_of_complex_l197_197994


namespace problem_l197_197217

variable {x y : ℝ}

theorem problem (hx : 0 < x) (hy : 0 < y) (h : x^2 - y^2 = 3 * x * y) :
  (x^2 / y^2) + (y^2 / x^2) - 2 = 9 :=
sorry

end problem_l197_197217


namespace only_zero_solution_l197_197107

theorem only_zero_solution (a b c n : ℤ) (h_gcd : Int.gcd (Int.gcd (Int.gcd a b) c) n = 1)
  (h_eq : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
sorry

end only_zero_solution_l197_197107


namespace binomial_product_l197_197983

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end binomial_product_l197_197983


namespace initially_calculated_avg_height_l197_197362

theorem initially_calculated_avg_height
  (A : ℕ)
  (initially_calculated_total_height : ℕ := 35 * A)
  (wrong_height : ℕ := 166)
  (actual_height : ℕ := 106)
  (height_overestimation : ℕ := wrong_height - actual_height)
  (actual_avg_height : ℕ := 179)
  (correct_total_height : ℕ := 35 * actual_avg_height)
  (initially_calculate_total_height_is_more : initially_calculated_total_height = correct_total_height + height_overestimation) :
  A = 181 :=
by
  sorry

end initially_calculated_avg_height_l197_197362


namespace cell_phones_sold_l197_197637

theorem cell_phones_sold (init_samsung init_iphone final_samsung final_iphone defective_samsung defective_iphone : ℕ)
    (h1 : init_samsung = 14) 
    (h2 : init_iphone = 8) 
    (h3 : final_samsung = 10) 
    (h4 : final_iphone = 5) 
    (h5 : defective_samsung = 2) 
    (h6 : defective_iphone = 1) : 
    init_samsung - defective_samsung - final_samsung + 
    init_iphone - defective_iphone - final_iphone = 4 := 
by
  sorry

end cell_phones_sold_l197_197637


namespace triangle_XDE_area_l197_197314

theorem triangle_XDE_area 
  (XY YZ XZ : ℝ) (hXY : XY = 8) (hYZ : YZ = 12) (hXZ : XZ = 14)
  (D E : ℝ → ℝ) (XD XE : ℝ) (hXD : XD = 3) (hXE : XE = 9) :
  ∃ (A : ℝ), A = 1/2 * XD * XE * (15 * Real.sqrt 17 / 56) ∧ A = 405 * Real.sqrt 17 / 112 :=
  sorry

end triangle_XDE_area_l197_197314


namespace ajay_walks_distance_l197_197641

theorem ajay_walks_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h_speed : speed = 3) 
  (h_time : time = 16.666666666666668) : 
  distance = speed * time :=
by
  sorry

end ajay_walks_distance_l197_197641


namespace AB_passes_fixed_point_locus_of_N_l197_197284

-- Definition of the parabola
def parabola (x y : ℝ) := y^2 = 4 * x

-- Definition of the point M which is the right-angle vertex
def M : ℝ × ℝ := (1, 2)

-- Statement for Part 1: Prove line AB passes through a fixed point
theorem AB_passes_fixed_point 
    (A B : ℝ × ℝ)
    (hA : parabola A.1 A.2)
    (hB : parabola B.1 B.2)
    (hM : M = (1, 2))
    (hRightAngle : (A.1 - 1) * (B.1 - 1) + (A.2 - 2) * (B.2 - 2) = 0) :
    ∃ P : ℝ × ℝ, P = (5, -2) := sorry

-- Statement for Part 2: Find the locus of point N
theorem locus_of_N 
    (A B : ℝ × ℝ)
    (hA : parabola A.1 A.2)
    (hB : parabola B.1 B.2)
    (hM : M = (1, 2))
    (hRightAngle : (A.1 - 1) * (B.1 - 1) + (A.2 - 2) * (B.2 - 2) = 0) 
    (N : ℝ × ℝ)
    (hN : ∃ t : ℝ, N = (t, -(t - 3))) :
    (N.1 - 3)^2 + N.2^2 = 8 ∧ N.1 ≠ 1 := sorry

end AB_passes_fixed_point_locus_of_N_l197_197284


namespace workers_contribution_l197_197863

theorem workers_contribution (W C : ℕ) (h1 : W * C = 300000) (h2 : W * (C + 50) = 360000) : W = 1200 :=
by
  sorry

end workers_contribution_l197_197863


namespace harry_friday_speed_l197_197902

theorem harry_friday_speed :
  let monday_speed := 10
  let tuesday_thursday_speed := monday_speed + monday_speed * (50 / 100)
  let friday_speed := tuesday_thursday_speed + tuesday_thursday_speed * (60 / 100)
  friday_speed = 24 :=
by
  sorry

end harry_friday_speed_l197_197902


namespace loan_to_scholarship_ratio_l197_197791

noncomputable def tuition := 22000
noncomputable def parents_contribution := tuition / 2
noncomputable def scholarship := 3000
noncomputable def wage_per_hour := 10
noncomputable def working_hours := 200
noncomputable def earnings := wage_per_hour * working_hours
noncomputable def total_scholarship_and_work := scholarship + earnings
noncomputable def remaining_tuition := tuition - parents_contribution - total_scholarship_and_work
noncomputable def student_loan := remaining_tuition

theorem loan_to_scholarship_ratio :
  (student_loan / scholarship) = 2 := 
by
  sorry

end loan_to_scholarship_ratio_l197_197791


namespace real_part_of_z_l197_197893

open Complex

theorem real_part_of_z (z : ℂ) (h : I * z = 1 + 2 * I) : z.re = 2 :=
sorry

end real_part_of_z_l197_197893


namespace weight_of_replaced_person_l197_197530

/-- The weight of the person who was replaced is calculated given the average weight increase for 8 persons and the weight of the new person. --/
theorem weight_of_replaced_person
  (avg_weight_increase : ℝ)
  (num_persons : ℕ)
  (weight_new_person : ℝ) :
  avg_weight_increase = 3 → 
  num_persons = 8 →
  weight_new_person = 89 →
  weight_new_person - avg_weight_increase * num_persons = 65 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end weight_of_replaced_person_l197_197530


namespace count_valid_m_l197_197489

theorem count_valid_m (h : 1260 > 0) :
  ∃! (n : ℕ), n = 3 := by
  sorry

end count_valid_m_l197_197489


namespace geometric_sequence_l197_197755

theorem geometric_sequence (a : ℝ) (h1 : a > 0)
  (h2 : ∃ r : ℝ, 210 * r = a ∧ a * r = 63 / 40) :
  a = 18.1875 :=
by
  sorry

end geometric_sequence_l197_197755


namespace prime_has_property_p_l197_197103

theorem prime_has_property_p (n : ℕ) (hn : Prime n) (a : ℕ) (h : n ∣ a^n - 1) : n^2 ∣ a^n - 1 := by
  sorry

end prime_has_property_p_l197_197103


namespace factorize_expression_l197_197664

theorem factorize_expression (x y : ℝ) : 
  x * y^2 - 6 * x * y + 9 * x = x * (y - 3)^2 := 
by sorry

end factorize_expression_l197_197664


namespace hours_l197_197992

def mechanic_hours_charged (h : ℕ) : Prop :=
  45 * h + 225 = 450

theorem hours (h : ℕ) : mechanic_hours_charged h → h = 5 :=
by
  intro h_eq
  have : 45 * h + 225 = 450 := h_eq
  sorry

end hours_l197_197992


namespace new_average_weight_l197_197046

-- noncomputable theory can be enabled if necessary for real number calculations.
-- noncomputable theory

def original_players : Nat := 7
def original_avg_weight : Real := 103
def new_players : Nat := 2
def weight_first_new_player : Real := 110
def weight_second_new_player : Real := 60

theorem new_average_weight :
  let original_total_weight : Real := original_players * original_avg_weight
  let total_weight : Real := original_total_weight + weight_first_new_player + weight_second_new_player
  let total_players : Nat := original_players + new_players
  total_weight / total_players = 99 := by
  sorry

end new_average_weight_l197_197046


namespace eccentricity_sum_cannot_be_2sqrt2_l197_197341

noncomputable def e1 (a b : ℝ) := Real.sqrt (1 + (b^2) / (a^2))
noncomputable def e2 (a b : ℝ) := Real.sqrt (1 + (a^2) / (b^2))
noncomputable def e1_plus_e2 (a b : ℝ) := e1 a b + e2 a b

theorem eccentricity_sum_cannot_be_2sqrt2 (a b : ℝ) : e1_plus_e2 a b ≠ 2 * Real.sqrt 2 := by
  sorry

end eccentricity_sum_cannot_be_2sqrt2_l197_197341


namespace abba_divisible_by_11_l197_197339

-- Given any two-digit number with digits a and b
def is_divisible_by_11 (a b : ℕ) : Prop :=
  (1001 * a + 110 * b) % 11 = 0

theorem abba_divisible_by_11 (a b : ℕ) (ha : a < 10) (hb : b < 10) : is_divisible_by_11 a b :=
  sorry

end abba_divisible_by_11_l197_197339


namespace part1_part2_part3_l197_197127

noncomputable def f (x a : ℝ) : ℝ := x^2 + (x - 1) * |x - a|

-- Part 1
theorem part1 (a : ℝ) (x : ℝ) (h : a = -1) : 
  (f x a = 1) ↔ (x ≤ -1 ∨ x = 1) :=
sorry

-- Part 2
theorem part2 (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a < f y a) ↔ (a ≥ 1 / 3) :=
sorry

-- Part 3
theorem part3 (a : ℝ) (h1 : a < 1) (h2 : ∀ x : ℝ, f x a ≥ 2 * x - 3) : 
  -3 ≤ a ∧ a < 1 :=
sorry

end part1_part2_part3_l197_197127


namespace percentage_waiting_for_parts_l197_197317

def totalComputers : ℕ := 20
def unfixableComputers : ℕ := (20 * 20) / 100
def fixedRightAway : ℕ := 8
def waitingForParts : ℕ := totalComputers - (unfixableComputers + fixedRightAway)

theorem percentage_waiting_for_parts : (waitingForParts : ℝ) / totalComputers * 100 = 40 := 
by 
  have : waitingForParts = 8 := sorry
  have : (8 / 20 : ℝ) * 100 = 40 := sorry
  exact sorry

end percentage_waiting_for_parts_l197_197317


namespace atleast_one_genuine_l197_197272

noncomputable def products : ℕ := 12
noncomputable def genuine : ℕ := 10
noncomputable def defective : ℕ := 2
noncomputable def selected : ℕ := 3

theorem atleast_one_genuine :
  (selected = 3) →
  (genuine + defective = 12) →
  (genuine ≥ 3) →
  (selected ≥ 1) →
  ∃ g d : ℕ, g + d = 3 ∧ g > 0 ∧ d ≤ 2 :=
by
  -- Proof will go here.
  sorry

end atleast_one_genuine_l197_197272


namespace cyclic_quadrilateral_angle_D_l197_197733

theorem cyclic_quadrilateral_angle_D (A B C D : ℝ) (h1 : A + C = 180) (h2 : B + D = 180) (h3 : 3 * A = 4 * B) (h4 : 3 * A = 6 * C) : D = 100 :=
by
  sorry

end cyclic_quadrilateral_angle_D_l197_197733


namespace future_tech_high_absentee_percentage_l197_197311

theorem future_tech_high_absentee_percentage :
  let total_students := 180
  let boys := 100
  let girls := 80
  let absent_boys_fraction := 1 / 5
  let absent_girls_fraction := 1 / 4
  let absent_boys := absent_boys_fraction * boys
  let absent_girls := absent_girls_fraction * girls
  let total_absent_students := absent_boys + absent_girls
  let absent_percentage := (total_absent_students / total_students) * 100
  (absent_percentage = 22.22) := 
by
  sorry

end future_tech_high_absentee_percentage_l197_197311


namespace smallest_m_n_sum_l197_197928

theorem smallest_m_n_sum (m n : ℕ) (hmn : m > n) (div_condition : 4900 ∣ (2023 ^ m - 2023 ^ n)) : m + n = 24 :=
by
  sorry

end smallest_m_n_sum_l197_197928


namespace total_short_trees_l197_197225

def short_trees_initial := 41
def short_trees_planted := 57

theorem total_short_trees : short_trees_initial + short_trees_planted = 98 := by
  sorry

end total_short_trees_l197_197225


namespace weng_total_earnings_l197_197944

noncomputable def weng_earnings_usd : ℝ :=
  let usd_per_hr_job1 : ℝ := 12
  let eur_per_hr_job2 : ℝ := 13
  let gbp_per_hr_job3 : ℝ := 9
  let hr_job1 : ℝ := 2 + 15 / 60
  let hr_job2 : ℝ := 1 + 40 / 60
  let hr_job3 : ℝ := 3 + 10 / 60
  let usd_to_eur : ℝ := 0.85
  let usd_to_gbp : ℝ := 0.76
  let eur_to_usd : ℝ := 1.18
  let gbp_to_usd : ℝ := 1.32
  let earnings_job1 : ℝ := usd_per_hr_job1 * hr_job1
  let earnings_job2_eur : ℝ := eur_per_hr_job2 * hr_job2
  let earnings_job2_usd : ℝ := earnings_job2_eur * eur_to_usd
  let earnings_job3_gbp : ℝ := gbp_per_hr_job3 * hr_job3
  let earnings_job3_usd : ℝ := earnings_job3_gbp * gbp_to_usd
  earnings_job1 + earnings_job2_usd + earnings_job3_usd

theorem weng_total_earnings : weng_earnings_usd = 90.19 :=
by
  sorry

end weng_total_earnings_l197_197944


namespace evaluate_expression_l197_197833

theorem evaluate_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 :=
by
  sorry

end evaluate_expression_l197_197833


namespace garden_watering_system_pumps_l197_197572

-- Define conditions
def rate := 500 -- gallons per hour
def time := 30 / 60 -- hours, i.e., converting 30 minutes to hours

-- Theorem statement
theorem garden_watering_system_pumps :
  rate * time = 250 := by
  sorry

end garden_watering_system_pumps_l197_197572


namespace perfect_square_trinomial_t_l197_197879

theorem perfect_square_trinomial_t (a b t : ℝ) :
  (∃ (x y : ℝ), x = a ∧ y = 2 * b ∧ a^2 + (2 * t - 1) * a * b + 4 * b^2 = (x + y)^2) →
  (t = 5 / 2 ∨ t = -3 / 2) :=
by
  sorry

end perfect_square_trinomial_t_l197_197879


namespace triangle_perimeter_inequality_l197_197279

theorem triangle_perimeter_inequality (x : ℕ) (h₁ : 15 + 24 > x) (h₂ : 15 + x > 24) (h₃ : 24 + x > 15) 
    (h₄ : ∃ x : ℕ, x > 9 ∧ x < 39) : 15 + 24 + x = 49 :=
by { sorry }

end triangle_perimeter_inequality_l197_197279


namespace problem1_problem2_l197_197447

-- For problem 1: Prove the quotient is 5.
def f (n : ℕ) : ℕ := 
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a + b + c + a * b + b * c + c * a + a * b * c

theorem problem1 : (625 / f 625) = 5 :=
by
  sorry

-- For problem 2: Prove the set of numbers.
def three_digit_numbers_satisfying_quotient : Finset ℕ :=
  {199, 299, 399, 499, 599, 699, 799, 899, 999}

theorem problem2 (n : ℕ) : (100 ≤ n ∧ n < 1000) ∧ n / f n = 1 ↔ n ∈ three_digit_numbers_satisfying_quotient :=
by
  sorry

end problem1_problem2_l197_197447


namespace rose_share_correct_l197_197710

-- Define the conditions
def purity_share (P : ℝ) : ℝ := P
def sheila_share (P : ℝ) : ℝ := 5 * P
def rose_share (P : ℝ) : ℝ := 3 * P
def total_rent := 5400

-- The theorem to be proven
theorem rose_share_correct (P : ℝ) (h : purity_share P + sheila_share P + rose_share P = total_rent) : 
  rose_share P = 1800 :=
  sorry

end rose_share_correct_l197_197710


namespace pos_int_solns_to_eq_l197_197919

open Int

theorem pos_int_solns_to_eq (x y z : ℤ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x^2 + y^2 - z^2 = 9 - 2 * x * y ↔ 
    (x, y, z) = (5, 0, 4) ∨ (x, y, z) = (4, 1, 4) ∨ (x, y, z) = (3, 2, 4) ∨ 
    (x, y, z) = (2, 3, 4) ∨ (x, y, z) = (1, 4, 4) ∨ (x, y, z) = (0, 5, 4) ∨ 
    (x, y, z) = (3, 0, 0) ∨ (x, y, z) = (2, 1, 0) ∨ (x, y, z) = (1, 2, 0) ∨ 
    (x, y, z) = (0, 3, 0) :=
by sorry

end pos_int_solns_to_eq_l197_197919


namespace term_is_18_minimum_value_l197_197763

-- Define the sequence a_n
def a_n (n : ℕ) : ℤ := n^2 - 5 * n + 4

-- Prove that a_n = 18 implies n = 7
theorem term_is_18 (n : ℕ) (h : a_n n = 18) : n = 7 := 
by 
  sorry

-- Prove that the minimum value of a_n is -2 and it occurs at n = 2 or n = 3
theorem minimum_value (n : ℕ) : n = 2 ∨ n = 3 ∧ a_n n = -2 :=
by 
  sorry

end term_is_18_minimum_value_l197_197763


namespace anes_age_l197_197148

theorem anes_age (w w_d : ℤ) (n : ℤ) 
  (h1 : 1436 ≤ w ∧ w < 1445)
  (h2 : 1606 ≤ w_d ∧ w_d < 1615)
  (h3 : w_d = w + n * 40) : 
  n = 4 :=
sorry

end anes_age_l197_197148


namespace man_l197_197032

variable (V_m V_c : ℝ)

theorem man's_speed_against_current :
  (V_m + V_c = 21 ∧ V_c = 2.5) → (V_m - V_c = 16) :=
by
  sorry

end man_l197_197032


namespace rooks_non_attacking_kings_non_attacking_bishops_non_attacking_knights_non_attacking_queens_non_attacking_l197_197916

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

end rooks_non_attacking_kings_non_attacking_bishops_non_attacking_knights_non_attacking_queens_non_attacking_l197_197916


namespace find_theta_2phi_l197_197942

-- Given
variables {θ φ : ℝ}
variables (hθ_acute : 0 < θ ∧ θ < π / 2)
variables (hφ_acute : 0 < φ ∧ φ < π / 2)
variables (h_tanθ : Real.tan θ = 3 / 11)
variables (h_sinφ : Real.sin φ = 1 / 3)

-- To prove
theorem find_theta_2phi : 
  ∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ Real.tan x = (21 + 6 * Real.sqrt 2) / (77 - 6 * Real.sqrt 2) ∧ x = θ + 2 * φ := 
sorry

end find_theta_2phi_l197_197942


namespace odd_function_f_neg_x_l197_197673

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 2 * x else -(x^2 + 2 * x)

theorem odd_function_f_neg_x (x : ℝ) (hx : x < 0) :
  f x = -x^2 - 2 * x :=
by
  sorry

end odd_function_f_neg_x_l197_197673


namespace calc_f_xh_min_f_x_l197_197108

def f (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 1

theorem calc_f_xh_min_f_x (x h : ℝ) : f (x + h) - f x = h * (10 * x + 5 * h - 2) := 
by
  sorry

end calc_f_xh_min_f_x_l197_197108


namespace subtraction_decimal_nearest_hundredth_l197_197816

theorem subtraction_decimal_nearest_hundredth : 
  (845.59 - 249.27 : ℝ) = 596.32 :=
by
  sorry

end subtraction_decimal_nearest_hundredth_l197_197816


namespace fewest_colored_paper_l197_197283
   
   /-- Jungkook, Hoseok, and Seokjin shared colored paper. 
       Jungkook took 10 cards, Hoseok took 7, and Seokjin took 2 less than Jungkook. 
       Prove that Hoseok took the fewest pieces of colored paper. -/
   theorem fewest_colored_paper 
       (Jungkook Hoseok Seokjin : ℕ)
       (hj : Jungkook = 10)
       (hh : Hoseok = 7)
       (hs : Seokjin = Jungkook - 2) :
       Hoseok < Jungkook ∧ Hoseok < Seokjin :=
   by
     sorry
   
end fewest_colored_paper_l197_197283


namespace baker_weekend_hours_l197_197794

noncomputable def loaves_per_hour : ℕ := 5
noncomputable def ovens : ℕ := 4
noncomputable def weekday_hours : ℕ := 5
noncomputable def total_loaves : ℕ := 1740
noncomputable def weeks : ℕ := 3
noncomputable def weekday_days : ℕ := 5
noncomputable def weekend_days : ℕ := 2

theorem baker_weekend_hours :
  ((total_loaves - (weeks * weekday_days * weekday_hours * (loaves_per_hour * ovens))) / (weeks * (loaves_per_hour * ovens))) / weekend_days = 4 := by
  sorry

end baker_weekend_hours_l197_197794


namespace sufficient_but_not_necessary_condition_l197_197454

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x + 1) * (x - 3) < 0 → x > -1 ∧ ((x > -1) → (x + 1) * (x - 3) < 0) = false :=
sorry

end sufficient_but_not_necessary_condition_l197_197454


namespace factor_tree_X_value_l197_197865

def H : ℕ := 2 * 5
def J : ℕ := 3 * 7
def F : ℕ := 7 * H
def G : ℕ := 11 * J
def X : ℕ := F * G

theorem factor_tree_X_value : X = 16170 := by
  sorry

end factor_tree_X_value_l197_197865


namespace correct_option_is_C_l197_197578

theorem correct_option_is_C 
  (A : Prop)
  (B : Prop)
  (C : Prop)
  (D : Prop)
  (hA : ¬ A)
  (hB : ¬ B)
  (hD : ¬ D)
  (hC : C) :
  C := by
  exact hC

end correct_option_is_C_l197_197578


namespace evaluate_64_pow_7_over_6_l197_197520

theorem evaluate_64_pow_7_over_6 : (64 : ℝ)^(7 / 6) = 128 := by
  have h : (64 : ℝ) = 2^6 := by norm_num
  rw [h]
  norm_num
  sorry

end evaluate_64_pow_7_over_6_l197_197520


namespace find_smallest_n_l197_197796

-- Define costs and relationships
def cost_red (r : ℕ) : ℕ := 10 * r
def cost_green (g : ℕ) : ℕ := 18 * g
def cost_blue (b : ℕ) : ℕ := 20 * b
def cost_purple (n : ℕ) : ℕ := 24 * n

-- Define the mathematical problem
theorem find_smallest_n (r g b : ℕ) :
  ∃ n : ℕ, 24 * n = Nat.lcm (cost_red r) (Nat.lcm (cost_green g) (cost_blue b)) ∧ n = 15 :=
by
  sorry

end find_smallest_n_l197_197796


namespace remainder_y_div_13_l197_197742

def x (k : ℤ) : ℤ := 159 * k + 37
def y (x : ℤ) : ℤ := 5 * x^2 + 18 * x + 22

theorem remainder_y_div_13 (k : ℤ) : (y (x k)) % 13 = 8 := by
  sorry

end remainder_y_div_13_l197_197742


namespace product_of_roots_l197_197713

theorem product_of_roots :
  ∀ α β : ℝ, (Polynomial.roots (Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 5 * Polynomial.X + Polynomial.C (-12))).prod = -6 :=
by
  sorry

end product_of_roots_l197_197713


namespace nancy_earns_more_l197_197806

theorem nancy_earns_more (giraffe_jade : ℕ) (giraffe_price : ℕ) (elephant_jade : ℕ)
    (elephant_price : ℕ) (total_jade : ℕ) (giraffes : ℕ) (elephants : ℕ) (giraffe_total : ℕ) (elephant_total : ℕ)
    (diff : ℕ) :
    giraffe_jade = 120 →
    giraffe_price = 150 →
    elephant_jade = 240 →
    elephant_price = 350 →
    total_jade = 1920 →
    giraffes = total_jade / giraffe_jade →
    giraffe_total = giraffes * giraffe_price →
    elephants = total_jade / elephant_jade →
    elephant_total = elephants * elephant_price →
    diff = elephant_total - giraffe_total →
    diff = 400 :=
by
  intros
  sorry

end nancy_earns_more_l197_197806


namespace lacy_percentage_correct_l197_197550

variable (x : ℕ)

-- Definitions from the conditions
def total_problems := 8 * x
def missed_problems := 2 * x
def answered_problems := total_problems - missed_problems
def bonus_problems := x
def bonus_points := 2 * bonus_problems
def regular_points := answered_problems - bonus_problems
def total_points_scored := bonus_points + regular_points
def total_available_points := 8 * x + 2 * x

theorem lacy_percentage_correct :
  total_points_scored / total_available_points * 100 = 90 := by
  -- Proof steps would go here, but are not required per instructions.
  sorry

end lacy_percentage_correct_l197_197550


namespace mean_cost_of_diesel_l197_197446

-- Define the diesel rates and the number of years.
def dieselRates : List ℝ := [1.2, 1.3, 1.8, 2.1]
def years : ℕ := 4

-- Define the mean calculation and the proof requirement.
theorem mean_cost_of_diesel (h₁ : dieselRates = [1.2, 1.3, 1.8, 2.1]) 
                               (h₂ : years = 4) : 
  (dieselRates.sum / years) = 1.6 :=
by
  sorry

end mean_cost_of_diesel_l197_197446


namespace number_of_mowers_l197_197781

noncomputable section

def area_larger_meadow (A : ℝ) : ℝ := 2 * A

def team_half_day_work (K a : ℝ) : ℝ := (K * a) / 2

def team_remaining_larger_meadow (K a : ℝ) : ℝ := (K * a) / 2

def half_team_half_day_work (K a : ℝ) : ℝ := (K * a) / 4

def larger_meadow_area_leq_sum (K a A : ℝ) : Prop :=
  team_half_day_work K a + team_remaining_larger_meadow K a = 2 * A

def smaller_meadow_area_left (K a A : ℝ) : ℝ :=
  A - half_team_half_day_work K a

def one_mower_one_day_work_rate (K a : ℝ) : ℝ := (K * a) / 4

def eq_total_mowed_by_team (K a A : ℝ) : Prop :=
  larger_meadow_area_leq_sum K a A ∧ smaller_meadow_area_left K a A = (K * a) / 4

theorem number_of_mowers
  (K a A b : ℝ)
  (h1 : larger_meadow_area_leq_sum K a A)
  (h2 : smaller_meadow_area_left K a A = one_mower_one_day_work_rate K a)
  (h3 : one_mower_one_day_work_rate K a = b)
  (h4 : K * a = 2 * A)
  (h5 : 2 * A = 4 * b)
  : K = 8 :=
  sorry

end number_of_mowers_l197_197781


namespace solve_for_x_l197_197906

theorem solve_for_x (y : ℝ) : 
  ∃ x : ℝ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 ∧ 
           x = -21 / 38 :=
by
  sorry

end solve_for_x_l197_197906


namespace sugar_already_put_in_l197_197261

-- Definitions based on conditions
def required_sugar : ℕ := 13
def additional_sugar_needed : ℕ := 11

-- Theorem to be proven
theorem sugar_already_put_in :
  required_sugar - additional_sugar_needed = 2 := by
  sorry

end sugar_already_put_in_l197_197261


namespace geometric_common_ratio_l197_197101

theorem geometric_common_ratio (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 5 * d)^2 = a1 * (a1 + 20 * d)) : 
  (a1 + 5 * d) / a1 = 3 :=
by
  sorry

end geometric_common_ratio_l197_197101


namespace negation_of_universal_proposition_l197_197798

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l197_197798


namespace octal_to_decimal_l197_197554

theorem octal_to_decimal (n_octal : ℕ) (h : n_octal = 123) : 
  let d0 := 3 * 8^0
  let d1 := 2 * 8^1
  let d2 := 1 * 8^2
  n_octal = 64 + 16 + 3 :=
by
  sorry

end octal_to_decimal_l197_197554


namespace expression_value_l197_197129

def a : ℤ := 5
def b : ℤ := -3
def c : ℕ := 2

theorem expression_value : (3 * c) / (a + b) + c = 5 := by
  sorry

end expression_value_l197_197129


namespace complement_of_A_in_U_l197_197248

noncomputable def U : Set ℤ := {x : ℤ | x^2 ≤ 2*x + 3}
def A : Set ℤ := {0, 1, 2}

theorem complement_of_A_in_U : (U \ A) = {-1, 3} :=
by
  sorry

end complement_of_A_in_U_l197_197248


namespace solve_system_l197_197202

theorem solve_system :
    (∃ x y z : ℝ, 5 * x^2 + 3 * y^2 + 3 * x * y + 2 * x * z - y * z - 10 * y + 5 = 0 ∧
                49 * x^2 + 65 * y^2 + 49 * z^2 - 14 * x * y - 98 * x * z + 14 * y * z - 182 * x - 102 * y + 182 * z + 233 =0
                ∧ ((x = 0 ∧ y = 1 ∧ z = -2)
                   ∨ (x = 2/7 ∧ y = 1 ∧ z = -12/7))) :=
by
  sorry

end solve_system_l197_197202


namespace gaussian_guardians_total_points_l197_197529

theorem gaussian_guardians_total_points :
  let daniel := 7
  let curtis := 8
  let sid := 2
  let emily := 11
  let kalyn := 6
  let hyojeong := 12
  let ty := 1
  let winston := 7
  daniel + curtis + sid + emily + kalyn + hyojeong + ty + winston = 54 := by
  sorry

end gaussian_guardians_total_points_l197_197529


namespace find_white_balls_l197_197368

noncomputable def white_balls_in_bag (total_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) 
  (p_not_red_nor_purple : ℚ) : ℕ :=
total_balls - (red_balls + purple_balls) - (green_balls + yellow_balls)

theorem find_white_balls :
  let total_balls := 60
  let green_balls := 18
  let yellow_balls := 17
  let red_balls := 3
  let purple_balls := 1
  let p_not_red_nor_purple := 0.95
  white_balls_in_bag total_balls green_balls yellow_balls red_balls purple_balls p_not_red_nor_purple = 21 :=
by
  let total_balls := 60
  let green_balls := 18
  let yellow_balls := 17
  let red_balls := 3
  let purple_balls := 1
  let p_not_red_nor_purple := 0.95
  sorry

end find_white_balls_l197_197368


namespace max_pN_value_l197_197799

noncomputable def max_probability_units_digit (N: ℕ) (q2 q5 q10: ℚ) : ℚ :=
  let qk (k : ℕ) := (Nat.floor (N / k) : ℚ) / N
  q10 * (2 - q10) + 2 * (q2 - q10) * (q5 - q10)

theorem max_pN_value : ∃ (a b : ℕ), (a.gcd b = 1) ∧ (∀ N q2 q5 q10, max_probability_units_digit N q2 q5 q10 ≤  27 / 100) ∧ (100 * 27 + 100 = 2800) :=
by
  sorry

end max_pN_value_l197_197799


namespace loan_amount_l197_197858

theorem loan_amount
  (P : ℝ)
  (SI : ℝ := 704)
  (R : ℝ := 8)
  (T : ℝ := 8)
  (h : SI = (P * R * T) / 100) : P = 1100 :=
by
  sorry

end loan_amount_l197_197858


namespace Dima_floor_l197_197918

theorem Dima_floor (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 9)
  (h2 : 60 = (n - 1))
  (h3 : 70 = (n - 1) / (n - 1) * 60 + (n - n / 2) * 2 * 60)
  (h4 : ∀ m : ℕ, 1 ≤ m ∧ m ≤ 9 → (5 * n = 6 * m + 1) → (n = 7 ∧ m = 6)) :
  n = 7 :=
by
  sorry

end Dima_floor_l197_197918


namespace chickens_bought_l197_197307

theorem chickens_bought (total_spent : ℤ) (egg_count : ℤ) (egg_price : ℤ) (chicken_price : ℤ) (egg_cost : ℤ := egg_count * egg_price) (chicken_spent : ℤ := total_spent - egg_cost) : total_spent = 88 → egg_count = 20 → egg_price = 2 → chicken_price = 8 → chicken_spent / chicken_price = 6 :=
by
  intros
  sorry

end chickens_bought_l197_197307


namespace bullet_trains_crossing_time_l197_197179

theorem bullet_trains_crossing_time
  (length : ℝ)
  (time1 time2 : ℝ)
  (speed1 speed2 : ℝ)
  (relative_speed : ℝ)
  (total_distance : ℝ)
  (cross_time : ℝ)
  (h_length : length = 120)
  (h_time1 : time1 = 10)
  (h_time2 : time2 = 20)
  (h_speed1 : speed1 = length / time1)
  (h_speed2 : speed2 = length / time2)
  (h_relative_speed : relative_speed = speed1 + speed2)
  (h_total_distance : total_distance = length + length)
  (h_cross_time : cross_time = total_distance / relative_speed) :
  cross_time = 240 / 18 := 
by
  sorry

end bullet_trains_crossing_time_l197_197179


namespace value_of_expression_l197_197843

open Real

theorem value_of_expression (m n r t : ℝ) 
  (h1 : m / n = 7 / 5) 
  (h2 : r / t = 8 / 15) : 
  (5 * m * r - 2 * n * t) / (6 * n * t - 8 * m * r) = 65 := 
by
  sorry

end value_of_expression_l197_197843


namespace max_value_m_l197_197266

/-- Proof that the inequality (a^2 + 4(b^2 + c^2))(b^2 + 4(a^2 + c^2))(c^2 + 4(a^2 + b^2)) 
    is greater than or equal to 729 for all a, b, c ∈ ℝ \ {0} with 
    |1/a| + |1/b| + |1/c| ≤ 3. -/
theorem max_value_m (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h_cond : |1 / a| + |1 / b| + |1 / c| ≤ 3) :
  (a^2 + 4 * (b^2 + c^2)) * (b^2 + 4 * (a^2 + c^2)) * (c^2 + 4 * (a^2 + b^2)) ≥ 729 :=
by {
  sorry
}

end max_value_m_l197_197266


namespace fare_midpoint_to_b_l197_197507

-- Define the conditions
def initial_fare : ℕ := 5
def initial_distance : ℕ := 2
def additional_fare_per_km : ℕ := 2
def total_fare : ℕ := 35
def walked_distance_meters : ℕ := 800

-- Define the correct answer
def fare_from_midpoint_to_b : ℕ := 19

-- Prove that the fare from the midpoint between A and B to B is 19 yuan
theorem fare_midpoint_to_b (y : ℝ) (h1 : 16.8 < y ∧ y ≤ 17) : 
  let half_distance := y / 2
  let total_taxi_distance := half_distance - 2
  let total_additional_fare := ⌈total_taxi_distance⌉ * additional_fare_per_km
  initial_fare + total_additional_fare = fare_from_midpoint_to_b := 
by
  sorry

end fare_midpoint_to_b_l197_197507


namespace mason_internet_speed_l197_197068

-- Definitions based on the conditions
def total_data : ℕ := 880
def downloaded_data : ℕ := 310
def remaining_time : ℕ := 190

-- Statement: The speed of Mason's Internet connection after it slows down
theorem mason_internet_speed :
  (total_data - downloaded_data) / remaining_time = 3 :=
by
  sorry

end mason_internet_speed_l197_197068


namespace fraction_value_l197_197443

variable (x y : ℝ)

theorem fraction_value (h : 1/x + 1/y = 2) : (2*x + 5*x*y + 2*y) / (x - 3*x*y + y) = -9 := by
  sorry

end fraction_value_l197_197443


namespace A_and_B_finish_work_together_in_12_days_l197_197486

theorem A_and_B_finish_work_together_in_12_days 
  (T_B : ℕ) 
  (T_A : ℕ)
  (h1 : T_B = 18) 
  (h2 : T_A = 2 * T_B) : 
  1 / (1 / T_A + 1 / T_B) = 12 := 
by 
  sorry

end A_and_B_finish_work_together_in_12_days_l197_197486


namespace all_statements_correct_l197_197899

theorem all_statements_correct :
  (∀ (b h : ℝ), (3 * b * h = 3 * (b * h))) ∧
  (∀ (b h : ℝ), (1/2 * b * (1/2 * h) = 1/2 * (1/2 * b * h))) ∧
  (∀ (r : ℝ), (π * (2 * r) ^ 2 = 4 * (π * r ^ 2))) ∧
  (∀ (r : ℝ), (π * (3 * r) ^ 2 = 9 * (π * r ^ 2))) ∧
  (∀ (s : ℝ), ((2 * s) ^ 2 = 4 * (s ^ 2)))
  → False := 
by 
  intros h
  sorry

end all_statements_correct_l197_197899


namespace repair_cost_total_l197_197499

-- Define the inputs
def labor_cost_rate : ℤ := 75
def labor_hours : ℤ := 16
def part_cost : ℤ := 1200

-- Define the required computation and proof statement
def total_repair_cost : ℤ :=
  let labor_cost := labor_cost_rate * labor_hours
  labor_cost + part_cost

theorem repair_cost_total : total_repair_cost = 2400 := by
  -- Proof would go here
  sorry

end repair_cost_total_l197_197499


namespace part1_part2_part3_l197_197731

-- Definitions for the given functions
def y1 (x : ℝ) : ℝ := -x + 1
def y2 (x : ℝ) : ℝ := -3 * x + 2

-- Part (1)
theorem part1 (a : ℝ) : (∃ x : ℝ, y1 x = a + y2 x ∧ x > 0) ↔ (a > -1) := sorry

-- Part (2)
theorem part2 (x y : ℝ) (h1 : y = y1 x) (h2 : y = y2 x) : 12*x^2 + 12*x*y + 3*y^2 = 27/4 := sorry

-- Part (3)
theorem part3 (A B : ℝ) (x : ℝ) (h : (4 - 2 * x) / ((3 * x - 2) * (x - 1)) = A / y1 x + B / y2 x) : (A / B + B / A) = -17 / 4 := sorry

end part1_part2_part3_l197_197731


namespace g_difference_l197_197778

def g (n : ℕ) : ℚ :=
  (1 / 4 : ℚ) * n^2 * (n + 1) * (n + 3) + 1

theorem g_difference (m : ℕ) : 
  g m - g (m - 1) = (3 / 4 : ℚ) * m^2 * (m + 5 / 3) :=
by
  sorry

end g_difference_l197_197778


namespace circle_center_l197_197820

theorem circle_center (x y : ℝ) :
  x^2 + 4 * x + y^2 - 6 * y + 1 = 0 → (x + 2, y - 3) = (0, 0) :=
by
  sorry

end circle_center_l197_197820


namespace evaluate_expression_l197_197725

theorem evaluate_expression :
  (3 + 6 + 9) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 6 + 9) = 5 / 6 :=
by sorry

end evaluate_expression_l197_197725


namespace integer_pairs_perfect_squares_l197_197231

theorem integer_pairs_perfect_squares (a b : ℤ) :
  (∃ k : ℤ, (a, b) = (k^2, 0) ∨ (a, b) = (0, k^2) ∨ (a, b) = (k, 1-k) ∨ (a, b) = (-6, -5) ∨ (a, b) = (-5, -6) ∨ (a, b) = (-4, -4))
  ↔ 
  (∃ x1 x2 : ℤ, a^2 + 4*b = x1^2 ∧ b^2 + 4*a = x2^2) :=
sorry

end integer_pairs_perfect_squares_l197_197231


namespace exists_231_four_digit_integers_l197_197720

theorem exists_231_four_digit_integers (n : ℕ) : 
  (∃ A B C D : ℕ, 
     A ≠ 0 ∧ 
     1 ≤ A ∧ A ≤ 9 ∧ 
     0 ≤ B ∧ B ≤ 9 ∧ 
     0 ≤ C ∧ C ≤ 9 ∧ 
     0 ≤ D ∧ D ≤ 9 ∧ 
     999 * (A - D) + 90 * (B - C) = n^3) ↔ n = 231 :=
by sorry

end exists_231_four_digit_integers_l197_197720


namespace find_four_digit_numbers_l197_197151

def isFourDigitNumber (n : ℕ) : Prop := (1000 ≤ n) ∧ (n < 10000)

noncomputable def solveABCD (AB CD : ℕ) : ℕ := 100 * AB + CD

theorem find_four_digit_numbers :
  ∀ (AB CD : ℕ),
    isFourDigitNumber (solveABCD AB CD) →
    solveABCD AB CD = AB * CD + AB ^ 2 →
      solveABCD AB CD = 1296 ∨ solveABCD AB CD = 3468 :=
by
  intros AB CD h1 h2
  sorry

end find_four_digit_numbers_l197_197151


namespace triangle_area_l197_197841

theorem triangle_area
  (area_WXYZ : ℝ)
  (side_small_squares : ℝ)
  (AB_eq_AC : ℝ)
  (A_coincides_with_O : ℝ)
  (area : ℝ) :
  area_WXYZ = 49 →  -- The area of square WXYZ is 49 cm^2
  side_small_squares = 2 → -- Sides of the smaller squares are 2 cm long
  AB_eq_AC = AB_eq_AC → -- Triangle ABC is isosceles with AB = AC
  A_coincides_with_O = A_coincides_with_O → -- A coincides with O
  area = 45 / 4 := -- The area of triangle ABC is 45/4 cm^2
by
  sorry

end triangle_area_l197_197841


namespace product_increase_l197_197377

theorem product_increase (a b : ℝ) (h : (a + 1) * (b + 1) = 2 * a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  ((a^2 - 1) * (b^2 - 1) = 4 * a * b) :=
sorry

end product_increase_l197_197377


namespace equal_charges_at_x_l197_197312

theorem equal_charges_at_x (x : ℝ) : 
  (2.75 * x + 125 = 1.50 * x + 140) → (x = 12) := 
by
  sorry

end equal_charges_at_x_l197_197312


namespace symmetric_sum_eq_two_l197_197366

-- Definitions and conditions
def symmetric (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = -Q.2

def P : ℝ × ℝ := (sorry, 1)
def Q : ℝ × ℝ := (-3, sorry)

-- Problem statement
theorem symmetric_sum_eq_two (h : symmetric P Q) : P.1 + Q.2 = 2 :=
by
  -- Proof omitted
  sorry

end symmetric_sum_eq_two_l197_197366


namespace max_integer_solutions_l197_197846

def quad_func (x : ℝ) : ℝ := x^2 - 6 * x + 1

theorem max_integer_solutions (p : ℝ → ℝ) : 
  (p = quad_func) →
  (∃ n1 n2 n3 n4 : ℤ, 
    ((p n1 = p (n1 ^ 2)) ∧ (p n2 = p (n2 ^ 2)) ∧ 
    (p n3 = p (n3 ^ 2)) ∧ (p n4 = p (n4 ^ 2))) ∧ 
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ 
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ 
    n3 ≠ n4) :=
by
  sorry

end max_integer_solutions_l197_197846


namespace probability_two_red_marbles_drawn_l197_197579

/-- A jar contains two red marbles, three green marbles, and ten white marbles and no other marbles.
Two marbles are randomly drawn from this jar without replacement. -/
theorem probability_two_red_marbles_drawn (total_marbles red_marbles green_marbles white_marbles : ℕ)
    (draw_without_replacement : Bool) :
    total_marbles = 15 ∧ red_marbles = 2 ∧ green_marbles = 3 ∧ white_marbles = 10 ∧ draw_without_replacement = true →
    (2 / 15) * (1 / 14) = 1 / 105 :=
by
  intro h
  sorry

end probability_two_red_marbles_drawn_l197_197579


namespace correct_average_l197_197074

-- let's define the numbers as a list
def numbers : List ℕ := [1200, 1300, 1510, 1520, 1530, 1200]

-- the condition given in the problem: the stated average is 1380
def stated_average : ℕ := 1380

-- given the correct calculation of average, let's write the theorem statement
theorem correct_average : (numbers.foldr (· + ·) 0) / numbers.length = 1460 :=
by
  -- we would prove it here
  sorry

end correct_average_l197_197074


namespace at_least_one_tails_up_l197_197996

-- Define propositions p and q
variable (p q : Prop)

-- The theorem statement
theorem at_least_one_tails_up : (¬p ∨ ¬q) ↔ ¬(p ∧ q) := by
  sorry

end at_least_one_tails_up_l197_197996


namespace possible_values_2n_plus_m_l197_197601

theorem possible_values_2n_plus_m :
  ∀ (n m : ℤ), 3 * n - m < 5 → n + m > 26 → 3 * m - 2 * n < 46 → 2 * n + m = 36 :=
by sorry

end possible_values_2n_plus_m_l197_197601


namespace triangle_inequality_l197_197194

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b) / (a + b + c) > 1 / 2 :=
sorry

end triangle_inequality_l197_197194


namespace sqrt_floor_eq_l197_197608

theorem sqrt_floor_eq (n : ℕ) (hn : 0 < n) :
    ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧ 
    ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ ∧ 
    ⌊Real.sqrt (4 * n + 3)⌋ = ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ :=
by
  sorry

end sqrt_floor_eq_l197_197608


namespace larry_initial_money_l197_197038

theorem larry_initial_money
  (M : ℝ)
  (spent_maintenance : ℝ := 0.04 * M)
  (saved_for_emergencies : ℝ := 0.30 * M)
  (snack_cost : ℝ := 5)
  (souvenir_cost : ℝ := 25)
  (lunch_cost : ℝ := 12)
  (loan_cost : ℝ := 10)
  (remaining_money : ℝ := 368)
  (total_spent : ℝ := snack_cost + souvenir_cost + lunch_cost + loan_cost) :
  M - spent_maintenance - saved_for_emergencies - total_spent = remaining_money →
  M = 636.36 :=
by
  sorry

end larry_initial_money_l197_197038


namespace julia_played_with_kids_on_tuesday_l197_197542

theorem julia_played_with_kids_on_tuesday (total: ℕ) (monday: ℕ) (tuesday: ℕ) 
  (h1: total = 18) (h2: monday = 4) : 
  tuesday = (total - monday) :=
by
  sorry

end julia_played_with_kids_on_tuesday_l197_197542


namespace initial_speed_l197_197643

variable (v : ℝ)
variable (h1 : (v / 2) + 2 * v = 75)

theorem initial_speed (v : ℝ) (h1 : (v / 2) + 2 * v = 75) : v = 30 :=
sorry

end initial_speed_l197_197643


namespace area_of_equilateral_triangle_l197_197829

theorem area_of_equilateral_triangle
  (A B C D E : Type) 
  (side_length : ℝ) 
  (medians_perpendicular : Prop) 
  (BD CE : ℝ)
  (inscribed_circle : Prop)
  (equilateral_triangle : A = B ∧ B = C) 
  (s : side_length = 18) 
  (BD_len : BD = 15) 
  (CE_len : CE = 9) 
  : ∃ area, area = 81 * Real.sqrt 3
  :=
by {
  sorry
}

end area_of_equilateral_triangle_l197_197829


namespace misread_system_of_equations_solutions_l197_197716

theorem misread_system_of_equations_solutions (a b : ℤ) (x₁ y₁ x₂ y₂ : ℤ)
  (h1 : x₁ = -3) (h2 : y₁ = -1) (h3 : x₂ = 5) (h4 : y₂ = 4)
  (eq1 : a * x₂ + 5 * y₂ = 15)
  (eq2 : 4 * x₁ - b * y₁ = -2) :
  a = -1 ∧ b = 10 ∧ a ^ 2023 + (- (1 / 10 : ℚ) * b) ^ 2023 = -2 := by
  -- Translate misreading conditions into theorems we need to prove (note: skipping proof).
  have hb : b = 10 := by sorry
  have ha : a = -1 := by sorry
  exact ⟨ha, hb, by simp [ha, hb]; norm_num⟩

end misread_system_of_equations_solutions_l197_197716


namespace intersection_M_N_l197_197963

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1} :=
  by sorry

end intersection_M_N_l197_197963


namespace no_intersect_M1_M2_l197_197043

theorem no_intersect_M1_M2 (A B : ℤ) : ∃ C : ℤ, 
  ∀ x y : ℤ, (x^2 + A * x + B) ≠ (2 * y^2 + 2 * y + C) := by
  sorry

end no_intersect_M1_M2_l197_197043


namespace Mona_joined_groups_l197_197479

theorem Mona_joined_groups (G : ℕ) (h : G * 4 - 3 = 33) : G = 9 :=
by
  sorry

end Mona_joined_groups_l197_197479


namespace initial_unread_messages_correct_l197_197592

-- Definitions based on conditions
def messages_read_per_day := 20
def messages_new_per_day := 6
def duration_in_days := 7
def effective_reading_rate := messages_read_per_day - messages_new_per_day

-- The initial number of unread messages
def initial_unread_messages := duration_in_days * effective_reading_rate

-- The theorem we want to prove
theorem initial_unread_messages_correct :
  initial_unread_messages = 98 :=
sorry

end initial_unread_messages_correct_l197_197592


namespace ratio_Ford_to_Toyota_l197_197315

-- Definitions based on the conditions
variables (Ford Dodge Toyota VW : ℕ)

axiom h1 : Ford = (1/3 : ℚ) * Dodge
axiom h2 : VW = (1/2 : ℚ) * Toyota
axiom h3 : VW = 5
axiom h4 : Dodge = 60

-- Theorem statement to be proven
theorem ratio_Ford_to_Toyota : Ford / Toyota = 2 :=
by {
  sorry
}

end ratio_Ford_to_Toyota_l197_197315


namespace largest_value_among_expressions_l197_197288

theorem largest_value_among_expressions 
  (a1 a2 b1 b2 : ℝ)
  (h1 : 0 < a1) (h2 : a1 < a2) (h3 : a2 < 1)
  (h4 : 0 < b1) (h5 : b1 < b2) (h6 : b2 < 1)
  (ha : a1 + a2 = 1) (hb : b1 + b2 = 1) :
  a1 * b1 + a2 * b2 > a1 * a2 + b1 * b2 ∧ 
  a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 := 
sorry

end largest_value_among_expressions_l197_197288


namespace baseball_cards_per_friend_l197_197617

theorem baseball_cards_per_friend (total_cards : ℕ) (total_friends : ℕ) (h1 : total_cards = 24) (h2 : total_friends = 4) : (total_cards / total_friends) = 6 := 
by
  sorry

end baseball_cards_per_friend_l197_197617


namespace r_values_if_polynomial_divisible_l197_197490

noncomputable
def find_r_iff_divisible (r : ℝ) : Prop :=
  (10 * (r^2 * (1 - 2*r))) = -6 ∧ 
  (2 * r + (1 - 2*r)) = 1 ∧ 
  (r^2 + 2 * r * (1 - 2*r)) = -5.2

theorem r_values_if_polynomial_divisible (r : ℝ) :
  (find_r_iff_divisible r) ↔ 
  (r = (2 + Real.sqrt 30) / 5 ∨ r = (2 - Real.sqrt 30) / 5) := 
by
  sorry

end r_values_if_polynomial_divisible_l197_197490


namespace min_m_n_l197_197262

theorem min_m_n (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 108 * m = n^3) : m + n = 8 :=
sorry

end min_m_n_l197_197262


namespace blue_paint_amount_l197_197513

theorem blue_paint_amount
  (blue_white_ratio : ℚ := 4 / 5)
  (white_paint : ℚ := 15)
  (blue_paint : ℚ) :
  blue_paint = 12 :=
by
  sorry

end blue_paint_amount_l197_197513


namespace total_campers_went_rowing_l197_197730

theorem total_campers_went_rowing (morning_campers afternoon_campers : ℕ) (h_morning : morning_campers = 35) (h_afternoon : afternoon_campers = 27) : morning_campers + afternoon_campers = 62 := by
  -- handle the proof
  sorry

end total_campers_went_rowing_l197_197730


namespace prime_iff_factorial_mod_l197_197602

theorem prime_iff_factorial_mod (p : ℕ) : 
  Nat.Prime p ↔ (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end prime_iff_factorial_mod_l197_197602


namespace problem_statement_l197_197719

variables (x y : ℚ)

theorem problem_statement 
  (h1 : x + y = 8 / 15) 
  (h2 : x - y = 1 / 105) : 
  x^2 - y^2 = 8 / 1575 :=
sorry

end problem_statement_l197_197719


namespace find_xz_l197_197766

theorem find_xz (x y z : ℝ) (h1 : 2 * x + z = 15) (h2 : x - 2 * y = 8) : x + z = 15 :=
sorry

end find_xz_l197_197766


namespace arrange_BANANA_l197_197620

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l197_197620


namespace partition_cities_l197_197861

-- Define the type for cities and airlines.
variable (City : Type) (Airline : Type)

-- Define the number of cities and airlines
variable (n k : ℕ)

-- Define a relation to represent bidirectional direct flights
variable (flight : Airline → City → City → Prop)

-- Define the condition: Some pairs of cities are connected by exactly one direct flight operated by one of the airline companies
-- or there are no such flights between them.
axiom unique_flight : ∀ (a : Airline) (c1 c2 : City), flight a c1 c2 → ¬ (∃ (a' : Airline), flight a' c1 c2 ∧ a' ≠ a)

-- Define the condition: Any two direct flights operated by the same company share a common endpoint
axiom shared_endpoint :
  ∀ (a : Airline) (c1 c2 c3 c4 : City), flight a c1 c2 → flight a c3 c4 → (c1 = c3 ∨ c1 = c4 ∨ c2 = c3 ∨ c2 = c4)

-- The main theorem to prove
theorem partition_cities :
  ∃ (partition : City → Fin (k + 2)), ∀ (c1 c2 : City) (a : Airline), flight a c1 c2 → partition c1 ≠ partition c2 :=
sorry

end partition_cities_l197_197861


namespace rhombus_side_length_l197_197069

theorem rhombus_side_length (a b s K : ℝ)
  (h1 : b = 3 * a)
  (h2 : K = (1 / 2) * a * b)
  (h3 : s ^ 2 = (a / 2) ^ 2 + (3 * a / 2) ^ 2) :
  s = Real.sqrt (5 * K / 3) :=
by
  sorry

end rhombus_side_length_l197_197069


namespace set_intersection_l197_197071

def A := {x : ℝ | -5 < x ∧ x < 2}
def B := {x : ℝ | |x| < 3}

theorem set_intersection : {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | -3 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l197_197071


namespace increase_by_percentage_l197_197498

-- Define the initial number.
def initial_number : ℝ := 75

-- Define the percentage increase as a decimal.
def percentage_increase : ℝ := 1.5

-- Define the expected final result after applying the increase.
def expected_result : ℝ := 187.5

-- The proof statement.
theorem increase_by_percentage : initial_number * (1 + percentage_increase) = expected_result :=
by
  sorry

end increase_by_percentage_l197_197498


namespace one_fourth_of_eight_point_four_l197_197226

theorem one_fourth_of_eight_point_four : (8.4 / 4) = (21 / 10) :=
by
  -- The expected proof would go here
  sorry

end one_fourth_of_eight_point_four_l197_197226


namespace rectangle_width_squared_l197_197747

theorem rectangle_width_squared (w l : ℝ) (h1 : w^2 + l^2 = 400) (h2 : 4 * w^2 + l^2 = 484) : w^2 = 28 := 
by
  sorry

end rectangle_width_squared_l197_197747


namespace remainder_of_4123_div_by_32_l197_197969

theorem remainder_of_4123_div_by_32 : 
  ∃ r, 0 ≤ r ∧ r < 32 ∧ 4123 = 32 * (4123 / 32) + r ∧ r = 27 := by
  sorry

end remainder_of_4123_div_by_32_l197_197969


namespace circle_area_l197_197203

theorem circle_area (x y : ℝ) : 
  x^2 + y^2 - 18 * x + 8 * y = -72 → 
  ∃ r : ℝ, r = 5 ∧ π * r ^ 2 = 25 * π := 
by
  sorry

end circle_area_l197_197203


namespace negative_integer_is_minus_21_l197_197950

variable (n : ℤ) (hn : n < 0) (h : n * (-3) + 2 = 65)

theorem negative_integer_is_minus_21 : n = -21 :=
by
  sorry

end negative_integer_is_minus_21_l197_197950


namespace simplify_expression_l197_197441

theorem simplify_expression (x y : ℝ) (h : x ≠ y) : (x^2 - x * y) / (x - y)^2 = x / (x - y) :=
by sorry

end simplify_expression_l197_197441


namespace point_D_coordinates_l197_197351

-- Define the vectors and points
structure Point where
  x : Int
  y : Int

def vector_add (p1 p2 : Point) : Point :=
  { x := p1.x + p2.x, y := p1.y + p2.y }

def scalar_multiply (k : Int) (p : Point) : Point :=
  { x := k * p.x, y := k * p.y }

def ab := Point.mk 5 (-3)
def c := Point.mk (-1) 3
def cd := scalar_multiply 2 ab

def D : Point := vector_add c cd

-- Theorem statement
theorem point_D_coordinates :
  D = Point.mk 9 (-3) :=
sorry

end point_D_coordinates_l197_197351


namespace gcd_32_48_l197_197309

/--
The greatest common factor of 32 and 48 is 16.
-/
theorem gcd_32_48 : Int.gcd 32 48 = 16 :=
by
  sorry

end gcd_32_48_l197_197309


namespace symmetric_point_l197_197418

theorem symmetric_point (A B C : ℝ) (hA : A = Real.sqrt 7) (hB : B = 1) :
  C = 2 - Real.sqrt 7 ↔ (A + C) / 2 = B :=
by
  sorry

end symmetric_point_l197_197418


namespace length_of_other_train_is_correct_l197_197990

noncomputable def length_of_second_train 
  (length_first_train : ℝ) 
  (speed_first_train : ℝ) 
  (speed_second_train : ℝ) 
  (time_to_cross : ℝ) 
  : ℝ := 
  let speed_first_train_m_s := speed_first_train * (1000 / 3600)
  let speed_second_train_m_s := speed_second_train * (1000 / 3600)
  let relative_speed := speed_first_train_m_s + speed_second_train_m_s
  let total_distance := relative_speed * time_to_cross
  total_distance - length_first_train

theorem length_of_other_train_is_correct :
  length_of_second_train 250 120 80 9 = 249.95 :=
by
  unfold length_of_second_train
  simp
  sorry

end length_of_other_train_is_correct_l197_197990


namespace john_has_22_dimes_l197_197748

theorem john_has_22_dimes (d q : ℕ) (h1 : d = q + 4) (h2 : 10 * d + 25 * q = 680) : d = 22 :=
by
sorry

end john_has_22_dimes_l197_197748


namespace unique_n_for_solutions_l197_197471

theorem unique_n_for_solutions :
  ∃! (n : ℕ), (∀ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (3 * x + 3 * y + 2 * z = n)) → 
  ((∃ (s : ℕ), s = 10) ∧ (n = 17)) :=
sorry

end unique_n_for_solutions_l197_197471


namespace scaled_multiplication_l197_197849

theorem scaled_multiplication
  (h : 14.97 * 46 = 688.62) :
  1.497 * 4.6 = 6.8862 :=
by
  sorry

end scaled_multiplication_l197_197849


namespace distinct_positive_roots_l197_197384

noncomputable def f (a x : ℝ) : ℝ := x^4 - x^3 + 8 * a * x^2 - a * x + a^2

theorem distinct_positive_roots (a : ℝ) :
  0 < a ∧ a < 1/24 → (∀ x1 x2 x3 x4 : ℝ, f a x1 = 0 ∧ 0 < x1 ∧ f a x2 = 0 ∧ 0 < x2 ∧ f a x3 = 0 ∧ 0 < x3 ∧ f a x4 = 0 ∧ 0 < x4 ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ↔ (1/25 < a ∧ a < 1/24) :=
sorry

end distinct_positive_roots_l197_197384


namespace division_exponentiation_addition_l197_197278

theorem division_exponentiation_addition :
  6 / -3 + 2^2 * (1 - 4) = -14 := by
sorry

end division_exponentiation_addition_l197_197278


namespace simplify_expression_l197_197896

theorem simplify_expression (x y : ℝ) :
  (2 * x + 25) + (150 * x + 40) + (5 * y + 10) = 152 * x + 5 * y + 75 :=
by sorry

end simplify_expression_l197_197896


namespace ab_proof_l197_197956

theorem ab_proof (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 90 < a + b) (h4 : a + b < 99) 
  (h5 : 0.9 < (a : ℝ) / b) (h6 : (a : ℝ) / b < 0.91) : a * b = 2346 :=
sorry

end ab_proof_l197_197956


namespace closed_fishing_season_purpose_sustainable_l197_197079

-- Defining the options for the purpose of the closed fishing season
inductive FishingPurpose
| sustainable_development : FishingPurpose
| inspect_fishing_vessels : FishingPurpose
| prevent_red_tides : FishingPurpose
| zoning_management : FishingPurpose

-- Defining rational utilization of resources involving fishing seasons
def rational_utilization (closed_fishing_season: Bool) : FishingPurpose := 
  if closed_fishing_season then FishingPurpose.sustainable_development 
  else FishingPurpose.inspect_fishing_vessels -- fallback for contradiction; shouldn't be used

-- The theorem we want to prove
theorem closed_fishing_season_purpose_sustainable :
  rational_utilization true = FishingPurpose.sustainable_development :=
sorry

end closed_fishing_season_purpose_sustainable_l197_197079


namespace find_cos_minus_sin_l197_197727

variable (θ : ℝ)
variable (h1 : θ ∈ Set.Ioo (3 * Real.pi / 4) Real.pi)
variable (h2 : Real.sin θ * Real.cos θ = -Real.sqrt 3 / 2)

theorem find_cos_minus_sin : Real.cos θ - Real.sin θ = -Real.sqrt (1 + Real.sqrt 3) := by
  sorry

end find_cos_minus_sin_l197_197727


namespace queenie_worked_4_days_l197_197388

-- Conditions
def daily_earning : ℕ := 150
def overtime_rate : ℕ := 5
def overtime_hours : ℕ := 4
def total_pay : ℕ := 770

-- Question
def number_of_days_worked (d : ℕ) : Prop := 
  daily_earning * d + overtime_rate * overtime_hours * d = total_pay

-- Theorem statement
theorem queenie_worked_4_days : ∃ d : ℕ, number_of_days_worked d ∧ d = 4 := 
by 
  use 4
  unfold number_of_days_worked 
  sorry

end queenie_worked_4_days_l197_197388


namespace john_outside_doors_count_l197_197149

theorem john_outside_doors_count 
  (bedroom_doors : ℕ := 3) 
  (cost_outside_door : ℕ := 20) 
  (total_cost : ℕ := 70) 
  (cost_bedroom_door := cost_outside_door / 2) 
  (total_bedroom_cost := bedroom_doors * cost_bedroom_door) 
  (outside_doors := (total_cost - total_bedroom_cost) / cost_outside_door) : 
  outside_doors = 2 :=
by
  sorry

end john_outside_doors_count_l197_197149


namespace repeated_mul_eq_pow_l197_197086

-- Define the repeated multiplication of 2, n times
def repeated_mul (n : ℕ) : ℕ :=
  (List.replicate n 2).prod

-- State the theorem to prove
theorem repeated_mul_eq_pow (n : ℕ) : repeated_mul n = 2 ^ n :=
by
  sorry

end repeated_mul_eq_pow_l197_197086


namespace roots_are_reciprocals_eq_a_minus_one_l197_197006

theorem roots_are_reciprocals_eq_a_minus_one (a : ℝ) :
  (∀ x y : ℝ, x + y = -(a - 1) ∧ x * y = a^2 → x * y = 1) → a = -1 :=
by
  intro h
  sorry

end roots_are_reciprocals_eq_a_minus_one_l197_197006


namespace merchant_spent_for_belle_l197_197573

def dress_cost (S : ℤ) (H : ℤ) : ℤ := 6 * S + 3 * H
def hat_cost (S : ℤ) (H : ℤ) : ℤ := 3 * S + 5 * H
def belle_cost (S : ℤ) (H : ℤ) : ℤ := S + 2 * H

theorem merchant_spent_for_belle :
  ∃ (S H : ℤ), dress_cost S H = 105 ∧ hat_cost S H = 70 ∧ belle_cost S H = 25 :=
by
  sorry

end merchant_spent_for_belle_l197_197573


namespace roy_consumes_tablets_in_225_minutes_l197_197216

variables 
  (total_tablets : ℕ) 
  (time_per_tablet : ℕ)

def total_time_to_consume_all_tablets 
  (total_tablets : ℕ) 
  (time_per_tablet : ℕ) : ℕ :=
  (total_tablets - 1) * time_per_tablet

theorem roy_consumes_tablets_in_225_minutes 
  (h1 : total_tablets = 10) 
  (h2 : time_per_tablet = 25) : 
  total_time_to_consume_all_tablets total_tablets time_per_tablet = 225 :=
by
  -- Proof goes here
  sorry

end roy_consumes_tablets_in_225_minutes_l197_197216


namespace math_time_more_than_science_l197_197541

section ExamTimes

-- Define the number of questions and time in minutes for each subject
def num_english_questions := 60
def num_math_questions := 25
def num_science_questions := 35

def time_english_minutes := 100
def time_math_minutes := 120
def time_science_minutes := 110

-- Define the time per question for each subject
def time_per_question (total_time : ℕ) (num_questions : ℕ) : ℚ :=
  total_time / num_questions

def time_english_per_question := time_per_question time_english_minutes num_english_questions
def time_math_per_question := time_per_question time_math_minutes num_math_questions
def time_science_per_question := time_per_question time_science_minutes num_science_questions

-- Prove the additional time per Math question compared to Science question
theorem math_time_more_than_science : 
  (time_math_per_question - time_science_per_question) = 1.6571 := 
sorry

end ExamTimes

end math_time_more_than_science_l197_197541


namespace tank_filling_time_l197_197427

noncomputable def netWaterPerCycle (rateA rateB rateC : ℕ) : ℕ := rateA + rateB - rateC

noncomputable def totalTimeToFill (tankCapacity rateA rateB rateC cycleDuration : ℕ) : ℕ :=
  let netWater := netWaterPerCycle rateA rateB rateC
  let cyclesNeeded := tankCapacity / netWater
  cyclesNeeded * cycleDuration

theorem tank_filling_time :
  totalTimeToFill 750 40 30 20 3 = 45 :=
by
  -- replace "sorry" with the actual proof if required
  sorry

end tank_filling_time_l197_197427


namespace fixed_point_graph_l197_197228

theorem fixed_point_graph (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) : ∃ x y : ℝ, (x = 2 ∧ y = 2 ∧ y = a^(x-2) + 1) :=
by
  use 2
  use 2
  sorry

end fixed_point_graph_l197_197228


namespace rational_abs_eq_l197_197680

theorem rational_abs_eq (a : ℚ) (h : |-3 - a| = 3 + |a|) : 0 ≤ a := 
by
  sorry

end rational_abs_eq_l197_197680


namespace part_I_part_II_l197_197269

-- Part I: Inequality solution
theorem part_I (x : ℝ) : 
  (abs (x - 1) ≥ 4 - abs (x - 3)) ↔ (x ≤ 0 ∨ x ≥ 4) := 
sorry

-- Part II: Minimum value of mn
theorem part_II (m n : ℕ) (h1 : (1:ℝ)/m + (1:ℝ)/(2*n) = 1) (hm : 0 < m) (hn : 0 < n) :
  (mn : ℕ) = 2 :=
sorry

end part_I_part_II_l197_197269


namespace interest_rate_l197_197775

theorem interest_rate (P T R : ℝ) (SI CI : ℝ) (difference : ℝ)
  (hP : P = 1700)
  (hT : T = 1)
  (hdiff : difference = 4.25)
  (hSI : SI = P * R * T / 100)
  (hCI : CI = P * ((1 + R / 200)^2 - 1))
  (hDiff : CI - SI = difference) : 
  R = 10 := sorry

end interest_rate_l197_197775


namespace solve_equation_l197_197123

theorem solve_equation : ∀ x : ℝ, (x + 2) / 4 - 1 = (2 * x + 1) / 3 → x = -2 :=
by
  intro x
  intro h
  sorry  

end solve_equation_l197_197123


namespace single_dog_barks_per_minute_l197_197299

theorem single_dog_barks_per_minute (x : ℕ) (h : 10 * 2 * x = 600) : x = 30 :=
by
  sorry

end single_dog_barks_per_minute_l197_197299


namespace potato_bag_weight_l197_197237

theorem potato_bag_weight (w : ℕ) (h₁ : w = 36) : w = 36 :=
by
  sorry

end potato_bag_weight_l197_197237


namespace range_of_a_l197_197064

theorem range_of_a (a : Real) : 
  (∀ x y : Real, (x^2 + y^2 + 2 * a * x - 4 * a * y + 5 * a^2 - 4 = 0 → x < 0 ∧ y > 0)) ↔ (a > 2) := 
sorry

end range_of_a_l197_197064


namespace calculate_adult_chaperones_l197_197577

theorem calculate_adult_chaperones (students : ℕ) (student_fee : ℕ) (adult_fee : ℕ) (total_fee : ℕ) 
  (h_students : students = 35) 
  (h_student_fee : student_fee = 5) 
  (h_adult_fee : adult_fee = 6) 
  (h_total_fee : total_fee = 199) : 
  ∃ (A : ℕ), 35 * student_fee + A * adult_fee = 199 ∧ A = 4 := 
by
  sorry

end calculate_adult_chaperones_l197_197577


namespace molecular_weight_CCl4_l197_197139

theorem molecular_weight_CCl4 (MW_7moles_CCl4 : ℝ) (h : MW_7moles_CCl4 = 1064) : 
  MW_7moles_CCl4 / 7 = 152 :=
by
  sorry

end molecular_weight_CCl4_l197_197139


namespace find_N_l197_197887

theorem find_N : ∀ N : ℕ, (991 + 993 + 995 + 997 + 999 = 5000 - N) → N = 25 :=
by
  intro N h
  sorry

end find_N_l197_197887


namespace number_of_books_from_second_shop_l197_197677

theorem number_of_books_from_second_shop (books_first_shop : ℕ) (cost_first_shop : ℕ)
    (books_second_shop : ℕ) (cost_second_shop : ℕ) (average_price : ℕ) :
    books_first_shop = 50 →
    cost_first_shop = 1000 →
    cost_second_shop = 800 →
    average_price = 20 →
    average_price * (books_first_shop + books_second_shop) = cost_first_shop + cost_second_shop →
    books_second_shop = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end number_of_books_from_second_shop_l197_197677


namespace max_value_f_l197_197914

noncomputable def f (x : ℝ) : ℝ := Real.sin (2*x) - 2 * Real.sqrt 3 * (Real.sin x)^2

theorem max_value_f : ∃ x : ℝ, f x = 2 - Real.sqrt 3 :=
  sorry

end max_value_f_l197_197914


namespace third_number_is_forty_four_l197_197752

theorem third_number_is_forty_four (a b c d e : ℕ) (h1 : a = e + 1) (h2 : b = e) 
  (h3 : c = e - 1) (h4 : d = e - 2) (h5 : e = e - 3) 
  (h6 : (a + b + c) / 3 = 45) (h7 : (c + d + e) / 3 = 43) : 
  c = 44 := 
sorry

end third_number_is_forty_four_l197_197752


namespace coat_total_selling_price_l197_197014

theorem coat_total_selling_price :
  let original_price := 120
  let discount_percent := 30
  let tax_percent := 8
  let discount_amount := (discount_percent / 100) * original_price
  let sale_price := original_price - discount_amount
  let tax_amount := (tax_percent / 100) * sale_price
  let total_selling_price := sale_price + tax_amount
  total_selling_price = 90.72 :=
by
  sorry

end coat_total_selling_price_l197_197014


namespace milk_for_18_cookies_l197_197531

def milk_needed_to_bake_cookies (cookies : ℕ) (milk_per_24_cookies : ℚ) (quarts_to_pints : ℚ) : ℚ :=
  (milk_per_24_cookies * quarts_to_pints) * (cookies / 24)

theorem milk_for_18_cookies :
  milk_needed_to_bake_cookies 18 4.5 2 = 6.75 :=
by
  sorry

end milk_for_18_cookies_l197_197531


namespace negation_of_P_is_exists_ge_1_l197_197492

theorem negation_of_P_is_exists_ge_1 :
  let P := ∀ x : ℤ, x < 1
  ¬P ↔ ∃ x : ℤ, x ≥ 1 := by
  sorry

end negation_of_P_is_exists_ge_1_l197_197492


namespace opposite_sides_of_line_l197_197394

theorem opposite_sides_of_line (m : ℝ) 
  (ha : (m + 0 - 1) * (2 + m - 1) < 0): 
  -1 < m ∧ m < 1 :=
sorry

end opposite_sides_of_line_l197_197394


namespace problem_x2_plus_y2_l197_197381

theorem problem_x2_plus_y2 (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 :=
sorry

end problem_x2_plus_y2_l197_197381


namespace mangoes_in_shop_l197_197070

-- Define the conditions
def ratio_mango_to_apple := 10 / 3
def apples := 36

-- Problem statement to prove
theorem mangoes_in_shop : ∃ (m : ℕ), m = 120 ∧ m = apples * ratio_mango_to_apple :=
by
  sorry

end mangoes_in_shop_l197_197070


namespace Maurice_current_age_l197_197780

variable (Ron_now Maurice_now : ℕ)

theorem Maurice_current_age
  (h1 : Ron_now = 43)
  (h2 : ∀ t Ron_future Maurice_future : ℕ, Ron_future = 4 * Maurice_future → Ron_future = Ron_now + 5 → Maurice_future = Maurice_now + 5) :
  Maurice_now = 7 := 
sorry

end Maurice_current_age_l197_197780


namespace sum_of_positive_ks_l197_197063

theorem sum_of_positive_ks :
  ∃ (S : ℤ), S = 39 ∧ ∀ k : ℤ, 
  (∃ α β : ℤ, α * β = 18 ∧ α + β = k) →
  (k > 0 → S = 19 + 11 + 9) := sorry

end sum_of_positive_ks_l197_197063


namespace represent_sum_and_product_eq_231_l197_197945

theorem represent_sum_and_product_eq_231 :
  ∃ (x y z w : ℕ), x = 3 ∧ y = 7 ∧ z = 11 ∧ w = 210 ∧ (231 = x + y + z + w) ∧ (231 = x * y * z) :=
by
  -- The proof is omitted here.
  sorry

end represent_sum_and_product_eq_231_l197_197945


namespace percentage_of_75_eq_percent_of_450_l197_197555

theorem percentage_of_75_eq_percent_of_450 (x : ℝ) (h : (x / 100) * 75 = 0.025 * 450) : x = 15 := 
sorry

end percentage_of_75_eq_percent_of_450_l197_197555


namespace max_sin_sin2x_l197_197287

open Real

theorem max_sin_sin2x (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  ∃ x : ℝ, (0 < x ∧ x < π / 2) ∧ (sin x * sin (2 * x) = 4 * sqrt 3 / 9) := 
sorry

end max_sin_sin2x_l197_197287


namespace natural_number_pairs_sum_to_three_l197_197358

theorem natural_number_pairs_sum_to_three :
  {p : ℕ × ℕ | p.1 + p.2 = 3} = {(1, 2), (2, 1)} :=
by
  sorry

end natural_number_pairs_sum_to_three_l197_197358


namespace unique_solution_l197_197082

-- Define the system of equations
def system_of_equations (m x y : ℝ) := 
  (m + 1) * x - y - 3 * m = 0 ∧ 4 * x + (m - 1) * y + 7 = 0

-- Define the determinant condition
def determinant_nonzero (m : ℝ) := m^2 + 3 ≠ 0

-- Theorem to prove there is exactly one solution
theorem unique_solution (m x y : ℝ) : 
  determinant_nonzero m → ∃! (x y : ℝ), system_of_equations m x y :=
by
  sorry

end unique_solution_l197_197082


namespace angle_sum_triangle_l197_197456

theorem angle_sum_triangle (A B C : ℝ) (hA : A = 75) (hB : B = 40) (h_sum : A + B + C = 180) : C = 65 :=
by
  sorry

end angle_sum_triangle_l197_197456


namespace percent_students_prefer_golf_l197_197285

theorem percent_students_prefer_golf (students_north : ℕ) (students_south : ℕ)
  (percent_golf_north : ℚ) (percent_golf_south : ℚ) :
  students_north = 1800 →
  students_south = 2200 →
  percent_golf_north = 15 →
  percent_golf_south = 25 →
  (820 / 4000 : ℚ) = 20.5 :=
by
  intros h_north h_south h_percent_north h_percent_south
  sorry

end percent_students_prefer_golf_l197_197285


namespace double_angle_second_quadrant_l197_197276

theorem double_angle_second_quadrant (α : ℝ) (h : π/2 < α ∧ α < π) : 
  ¬((0 ≤ 2*α ∧ 2*α < π/2) ∨ (3*π/2 < 2*α ∧ 2*α < 2*π)) :=
sorry

end double_angle_second_quadrant_l197_197276


namespace first_group_hours_per_day_l197_197047

theorem first_group_hours_per_day :
  ∃ H : ℕ, 
    (39 * 12 * H = 30 * 26 * 3) ∧
    H = 5 :=
by sorry

end first_group_hours_per_day_l197_197047


namespace select_team_of_5_l197_197954

def boys : ℕ := 7
def girls : ℕ := 9
def total_students : ℕ := boys + girls

theorem select_team_of_5 (n : ℕ := total_students) (k : ℕ := 5) :
  (Nat.choose n k) = 4368 :=
by
  sorry

end select_team_of_5_l197_197954


namespace jamie_hours_each_time_l197_197173

theorem jamie_hours_each_time (hours_per_week := 2) (weeks := 6) (rate := 10) (total_earned := 360) : 
  ∃ (h : ℕ), h = 3 ∧ (hours_per_week * weeks * rate * h = total_earned) := 
by
  sorry

end jamie_hours_each_time_l197_197173


namespace determine_common_difference_l197_197622

variables {a : ℕ → ℤ} {d : ℤ}

-- Definition of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 + n * d

-- The given condition in the problem
def given_condition (a : ℕ → ℤ) (d : ℤ) : Prop :=
  3 * a 6 = a 3 + a 4 + a 5 + 6

-- The theorem to prove
theorem determine_common_difference
  (h_seq : arithmetic_seq a d)
  (h_cond : given_condition a d) :
  d = 1 :=
sorry

end determine_common_difference_l197_197622


namespace max_area_of_house_l197_197091

-- Definitions for conditions
def height_of_plates : ℝ := 2.5
def price_per_meter_colored : ℝ := 450
def price_per_meter_composite : ℝ := 200
def roof_cost_per_sqm : ℝ := 200
def cost_limit : ℝ := 32000

-- Definitions for the variables
variables (x y : ℝ) (P S : ℝ)

-- Definition for the material cost P
def material_cost (x y : ℝ) : ℝ := 900 * x + 400 * y + 200 * x * y

-- Maximum area S and corresponding x
theorem max_area_of_house (x y : ℝ) (h : material_cost x y ≤ cost_limit) :
  S = 100 ∧ x = 20 / 3 :=
sorry

end max_area_of_house_l197_197091


namespace sum_remainder_l197_197768

theorem sum_remainder (n : ℕ) (h : n = 102) :
  ((n * (n + 1) / 2) % 5250) = 3 :=
by
  sorry

end sum_remainder_l197_197768


namespace perpendicular_line_sum_l197_197508

theorem perpendicular_line_sum (a b c : ℝ) (h1 : a + 4 * c - 2 = 0) (h2 : 2 - 5 * c + b = 0) 
  (perpendicular : (a / -4) * (2 / 5) = -1) : a + b + c = -4 := 
sorry

end perpendicular_line_sum_l197_197508


namespace number_of_boxes_sold_on_saturday_l197_197463

theorem number_of_boxes_sold_on_saturday (S : ℝ) 
  (h : S + 1.5 * S + 1.95 * S + 2.34 * S + 2.574 * S = 720) : 
  S = 77 := 
sorry

end number_of_boxes_sold_on_saturday_l197_197463


namespace abs_eq_neg_l197_197497

theorem abs_eq_neg (x : ℝ) (h : |x + 6| = -(x + 6)) : x ≤ -6 :=
by 
  sorry

end abs_eq_neg_l197_197497


namespace complement_intersection_l197_197097

def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | x < 2 }
def CR (S : Set ℝ) : Set ℝ := { x | x ∉ S }

theorem complement_intersection :
  CR (M ∩ N) = { x | x < 1 } ∪ { x | x ≥ 2 } := by
  sorry

end complement_intersection_l197_197097


namespace problem_statement_l197_197236

theorem problem_statement (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end problem_statement_l197_197236


namespace box_volume_possible_l197_197438

theorem box_volume_possible (x : ℕ) (V : ℕ) (H1 : V = 40 * x^3) (H2 : (2 * x) * (4 * x) * (5 * x) = V) : 
  V = 320 :=
by 
  have x_possible_values := x
  -- checking if V = 320 and x = 2 satisfies the given conditions
  sorry

end box_volume_possible_l197_197438


namespace janice_remaining_time_l197_197556

theorem janice_remaining_time
  (homework_time : ℕ := 30)
  (clean_room_time : ℕ := homework_time / 2)
  (walk_dog_time : ℕ := homework_time + 5)
  (take_out_trash_time : ℕ := homework_time / 6)
  (total_time_before_movie : ℕ := 120) :
  (total_time_before_movie - (homework_time + clean_room_time + walk_dog_time + take_out_trash_time)) = 35 :=
by
  sorry

end janice_remaining_time_l197_197556


namespace sum_area_triangles_lt_total_area_l197_197802

noncomputable def G : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def A_k (k : ℕ+) : ℝ := sorry -- Assume we've defined A_k's expression correctly
noncomputable def S (S1 S2 : ℝ) : ℝ := 2 * S1 - S2

theorem sum_area_triangles_lt_total_area (k : ℕ+) (S1 S2 : ℝ) :
  (A_k k < S S1 S2) :=
sorry

end sum_area_triangles_lt_total_area_l197_197802


namespace gwen_spent_zero_l197_197400

theorem gwen_spent_zero 
  (m : ℕ) 
  (d : ℕ) 
  (S : ℕ) 
  (h1 : m = 8) 
  (h2 : d = 5)
  (h3 : (m - S) = (d - S) + 3) : 
  S = 0 :=
by
  sorry

end gwen_spent_zero_l197_197400


namespace length_of_ln_l197_197949

theorem length_of_ln (sin_N_eq : Real.sin angle_N = 3 / 5) (LM_eq : length_LM = 15) :
  length_LN = 25 :=
sorry

end length_of_ln_l197_197949


namespace bus_trip_length_l197_197413

theorem bus_trip_length (v T : ℝ) 
    (h1 : 2 * v + (T - 2 * v) * (3 / (2 * v)) + 1 = T / v + 5)
    (h2 : 2 + 30 / v + (T - (2 * v + 30)) * (3 / (2 * v)) + 1 = T / v + 4) : 
    T = 180 :=
    sorry

end bus_trip_length_l197_197413


namespace pablo_books_read_l197_197894

noncomputable def pages_per_book : ℕ := 150
noncomputable def cents_per_page : ℕ := 1
noncomputable def cost_of_candy : ℕ := 1500    -- $15 in cents
noncomputable def leftover_money : ℕ := 300    -- $3 in cents
noncomputable def total_money := cost_of_candy + leftover_money
noncomputable def earnings_per_book := pages_per_book * cents_per_page

theorem pablo_books_read : total_money / earnings_per_book = 12 := by
  sorry

end pablo_books_read_l197_197894


namespace series_sum_is_6_over_5_l197_197048

noncomputable def series_sum : ℝ := ∑' n : ℕ, if n % 4 == 0 then 1 / (4^(n/4)) else 
                                          if n % 4 == 1 then 1 / (2 * 4^(n/4)) else 
                                          if n % 4 == 2 then -1 / (4^(n/4) * 4^(1/2)) else 
                                          -1 / (2 * 4^(n/4 + 1/2))

theorem series_sum_is_6_over_5 : series_sum = 6 / 5 := 
  sorry

end series_sum_is_6_over_5_l197_197048


namespace triangular_array_of_coins_l197_197375

theorem triangular_array_of_coins (N : ℤ) (h : N * (N + 1) / 2 = 3003) : N = 77 :=
by
  sorry

end triangular_array_of_coins_l197_197375


namespace barrel_capacity_l197_197117

theorem barrel_capacity (x y : ℝ) (h1 : y = 45 / (3/5)) (h2 : 0.6*x = y*3/5) (h3 : 0.4*x = 18) : 
  y = 75 :=
by
  sorry

end barrel_capacity_l197_197117


namespace math_problem_l197_197485

theorem math_problem 
  (a1 : (10^4 + 500) = 100500)
  (a2 : (25^4 + 500) = 390625500)
  (a3 : (40^4 + 500) = 256000500)
  (a4 : (55^4 + 500) = 915062500)
  (a5 : (70^4 + 500) = 24010062500)
  (b1 : (5^4 + 500) = 625+500)
  (b2 : (20^4 + 500) = 160000500)
  (b3 : (35^4 + 500) = 150062500)
  (b4 : (50^4 + 500) = 625000500)
  (b5 : (65^4 + 500) = 1785062500) :
  ( (100500 * 390625500 * 256000500 * 915062500 * 24010062500) / (625+500 * 160000500 * 150062500 * 625000500 * 1785062500) = 240) :=
by
  sorry

end math_problem_l197_197485


namespace prime_factorization_sum_l197_197072

theorem prime_factorization_sum (w x y z k : ℕ) (h : 2^w * 3^x * 5^y * 7^z * 11^k = 2310) :
  2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 28 :=
sorry

end prime_factorization_sum_l197_197072


namespace range_of_m_l197_197971

theorem range_of_m (m : ℝ) : 
  (m - 1 < 0 ∧ 4 * m - 3 > 0) → (3 / 4 < m ∧ m < 1) := 
by
  sorry

end range_of_m_l197_197971


namespace sofa_love_seat_ratio_l197_197823

theorem sofa_love_seat_ratio (L S: ℕ) (h1: L = 148) (h2: S + L = 444): S = 2 * L := by
  sorry

end sofa_love_seat_ratio_l197_197823


namespace no_half_dimension_cuboid_l197_197503

theorem no_half_dimension_cuboid
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (a' b' c' : ℝ) (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0) :
  ¬ (a' * b' * c' = (1 / 2) * a * b * c ∧ 2 * (a' * b' + b' * c' + c' * a') = a * b + b * c + c * a) :=
by
  sorry

end no_half_dimension_cuboid_l197_197503


namespace ratio_A_B_l197_197869

-- Define constants for non-zero numbers A and B
variables {A B : ℕ} (h1 : A ≠ 0) (h2 : B ≠ 0)

-- Define the given condition
theorem ratio_A_B (h : (2 * A) * 7 = (3 * B) * 3) : A / B = 9 / 14 := by
  sorry

end ratio_A_B_l197_197869


namespace find_n_l197_197694

theorem find_n (n : ℕ) : (256 : ℝ) ^ (1 / 4 : ℝ) = 4 ^ n → 256 = (4 ^ 4 : ℝ) → n = 1 :=
by
  intros h₁ h₂
  sorry

end find_n_l197_197694


namespace real_roots_of_quadratic_l197_197304

theorem real_roots_of_quadratic (k : ℝ) : (k ≤ 0 ∨ 1 ≤ k) →
  ∃ x : ℝ, x^2 + 2 * k * x + k = 0 :=
by
  intro h
  sorry

end real_roots_of_quadratic_l197_197304


namespace exists_multiple_with_digits_0_or_1_l197_197215

theorem exists_multiple_with_digits_0_or_1 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (k % n = 0) ∧ (∀ digit ∈ k.digits 10, digit = 0 ∨ digit = 1) ∧ (k.digits 10).length ≤ n :=
sorry

end exists_multiple_with_digits_0_or_1_l197_197215


namespace inverse_47_mod_48_l197_197596

theorem inverse_47_mod_48 : ∃ x, x < 48 ∧ x > 0 ∧ 47 * x % 48 = 1 :=
sorry

end inverse_47_mod_48_l197_197596


namespace scientific_notation_per_capita_GDP_l197_197571

theorem scientific_notation_per_capita_GDP (GDP : ℝ) (h : GDP = 104000): 
  GDP = 1.04 * 10^5 := 
by
  sorry

end scientific_notation_per_capita_GDP_l197_197571


namespace find_AM_l197_197208

-- Definitions (conditions)
variables {A M B : ℝ}
variable  (collinear : A ≤ M ∧ M ≤ B ∨ B ≤ M ∧ M ≤ A ∨ A ≤ B ∧ B ≤ M)
          (h1 : abs (M - A) = 2 * abs (M - B)) 
          (h2 : abs (A - B) = 6)

-- Proof problem statement
theorem find_AM : (abs (M - A) = 4) ∨ (abs (M - A) = 12) :=
by 
  sorry

end find_AM_l197_197208


namespace paula_paint_cans_needed_l197_197567

-- Let's define the initial conditions and required computations in Lean.
def initial_rooms : ℕ := 48
def cans_lost : ℕ := 4
def remaining_rooms : ℕ := 36
def large_rooms_to_paint : ℕ := 8
def normal_rooms_to_paint : ℕ := 20
def paint_per_large_room : ℕ := 2 -- as each large room requires twice as much paint

-- Define a function to compute the number of cans required.
def cans_needed (initial_rooms remaining_rooms large_rooms_to_paint normal_rooms_to_paint paint_per_large_room : ℕ) : ℕ :=
  let rooms_lost := initial_rooms - remaining_rooms
  let cans_per_room := rooms_lost / cans_lost
  let total_room_equivalents := large_rooms_to_paint * paint_per_large_room + normal_rooms_to_paint
  total_room_equivalents / cans_per_room

theorem paula_paint_cans_needed : cans_needed initial_rooms remaining_rooms large_rooms_to_paint normal_rooms_to_paint paint_per_large_room = 12 :=
by
  -- The proof would go here
  sorry

end paula_paint_cans_needed_l197_197567


namespace find_M_l197_197115

theorem find_M (a b M : ℝ) (h : (a + 2 * b)^2 = (a - 2 * b)^2 + M) : M = 8 * a * b :=
by sorry

end find_M_l197_197115


namespace correct_transformation_l197_197524

theorem correct_transformation (a b c : ℝ) (h : (b / (a^2 + 1)) > (c / (a^2 + 1))) : b > c :=
by {
  -- Placeholder proof
  sorry
}

end correct_transformation_l197_197524


namespace diagonals_bisect_in_rhombus_l197_197600

axiom Rhombus : Type
axiom Parallelogram : Type

axiom isParallelogram : Rhombus → Parallelogram
axiom diagonalsBisectEachOther : Parallelogram → Prop

theorem diagonals_bisect_in_rhombus (R : Rhombus) :
  ∀ (P : Parallelogram), isParallelogram R = P → diagonalsBisectEachOther P → diagonalsBisectEachOther (isParallelogram R) :=
by
  sorry

end diagonals_bisect_in_rhombus_l197_197600


namespace find_modulus_l197_197259

open Complex -- Open the Complex namespace for convenience

noncomputable def modulus_of_z (a : ℝ) (h : (1 + 2 * Complex.I) * (a + Complex.I : ℂ) = Complex.re ((1 + 2 * Complex.I) * (a + Complex.I)) + Complex.im ((1 + 2 * Complex.I) * (a + Complex.I)) * Complex.I) : ℝ :=
  Complex.abs ((1 + 2 * Complex.I) * (a + Complex.I))

theorem find_modulus : modulus_of_z (-3) (by {
  -- Provide the condition that real part equals imaginary part
  admit -- This 'admit' serves as a placeholder for the proof of the condition 
}) = 5 * Real.sqrt 2 := sorry

end find_modulus_l197_197259


namespace concave_number_probability_l197_197986

/-- Definition of a concave number -/
def is_concave_number (a b c : ℕ) : Prop :=
  a > b ∧ c > b

/-- Set of possible digits -/
def digits : Finset ℕ := {4, 5, 6, 7, 8}

 /-- Total number of distinct three-digit combinations -/
def total_combinations : ℕ := 60

 /-- Number of concave numbers -/
def concave_numbers : ℕ := 20

 /-- Probability that a randomly chosen three-digit number is a concave number -/
def probability_concave : ℚ := concave_numbers / total_combinations

theorem concave_number_probability :
  probability_concave = 1 / 3 :=
by
  sorry

end concave_number_probability_l197_197986


namespace equivalent_discount_l197_197161

theorem equivalent_discount (original_price : ℝ) (d1 d2 single_discount : ℝ) :
  original_price = 50 →
  d1 = 0.15 →
  d2 = 0.10 →
  single_discount = 0.235 →
  original_price * (1 - d1) * (1 - d2) = original_price * (1 - single_discount) :=
by
  intros
  sorry

end equivalent_discount_l197_197161


namespace decreased_revenue_l197_197634

variable (T C : ℝ)
def Revenue (tax consumption : ℝ) : ℝ := tax * consumption

theorem decreased_revenue (hT_new : T_new = 0.9 * T) (hC_new : C_new = 1.1 * C) :
  Revenue T_new C_new = 0.99 * (Revenue T C) := 
sorry

end decreased_revenue_l197_197634


namespace triangle_perimeter_l197_197000

theorem triangle_perimeter (m : ℝ) (a b : ℝ) (h1 : 3 ^ 2 - 3 * (m + 1) + 2 * m = 0)
  (h2 : a ^ 2 - (m + 1) * a + 2 * m = 0)
  (h3 : b ^ 2 - (m + 1) * b + 2 * m = 0)
  (h4 : a = 3 ∨ b = 3)
  (h5 : a ≠ b ∨ a = b)
  (hAB : a ≠ b ∨ a = b) :
  (∀ s₁ s₂ : ℝ, s₁ = a ∨ s₁ = b ∧ s₂ = a ∨ s₂ = b ∧ s₁ ≠ s₂ → s₁ + s₁ + s₂ = 10 ∨ s₁ + s₁ + s₂ = 11) ∨
  (∀ s₁ s₂ : ℝ, s₁ = a ∨ s₁ = b ∧ s₂ = a ∨ s₂ = b ∧ s₁ = s₂ → b + b + a = 10 ∨ b + b + a = 11) := by
  sorry

end triangle_perimeter_l197_197000


namespace angles_around_point_sum_l197_197036

theorem angles_around_point_sum 
  (x y : ℝ)
  (h1 : 130 + x + y = 360)
  (h2 : y = x + 30) :
  x = 100 ∧ y = 130 :=
by
  sorry

end angles_around_point_sum_l197_197036


namespace sequence_from_625_to_629_l197_197188

def arrows_repeating_pattern (n : ℕ) : ℕ := n % 5

theorem sequence_from_625_to_629 :
  arrows_repeating_pattern 625 = 0 ∧ arrows_repeating_pattern 629 = 4 →
  ∃ (seq : ℕ → ℕ), 
    (seq 0 = arrows_repeating_pattern 625) ∧
    (seq 1 = arrows_repeating_pattern (625 + 1)) ∧
    (seq 2 = arrows_repeating_pattern (625 + 2)) ∧
    (seq 3 = arrows_repeating_pattern (625 + 3)) ∧
    (seq 4 = arrows_repeating_pattern 629) := 
sorry

end sequence_from_625_to_629_l197_197188


namespace number_of_girls_l197_197788

theorem number_of_girls (B G: ℕ) 
  (ratio : 8 * G = 5 * B) 
  (total : B + G = 780) :
  G = 300 := 
sorry

end number_of_girls_l197_197788


namespace translated_point_is_correct_l197_197246

-- Cartesian Point definition
structure Point where
  x : Int
  y : Int

-- Define the translation function
def translate (p : Point) (dx dy : Int) : Point :=
  Point.mk (p.x + dx) (p.y - dy)

-- Define the initial point A and the translation amounts
def A : Point := ⟨-3, 2⟩
def dx : Int := 3
def dy : Int := 2

-- The proof goal
theorem translated_point_is_correct :
  translate A dx dy = ⟨0, 0⟩ :=
by
  -- This is where the proof would normally go
  sorry

end translated_point_is_correct_l197_197246


namespace prime_power_condition_l197_197404

open Nat

theorem prime_power_condition (u v : ℕ) :
  (∃ p n : ℕ, p.Prime ∧ p^n = (u * v^3) / (u^2 + v^2)) ↔ ∃ k : ℕ, k ≥ 1 ∧ u = 2^k ∧ v = 2^k := by {
  sorry
}

end prime_power_condition_l197_197404


namespace mow_lawn_time_l197_197168

noncomputable def time_to_mow (lawn_length lawn_width: ℝ) 
(swat_width overlap width_conversion: ℝ) (speed: ℝ) : ℝ :=
(lawn_length * lawn_width) / (((swat_width - overlap) / width_conversion) * lawn_length * speed)

theorem mow_lawn_time : 
  time_to_mow 120 180 30 6 12 6000 = 1.8 := 
by
  -- Given:
  -- Lawn dimensions: 120 feet by 180 feet
  -- Mower swath: 30 inches with 6 inches overlap
  -- Walking speed: 6000 feet per hour
  -- Conversion factor: 12 inches = 1 foot
  sorry

end mow_lawn_time_l197_197168


namespace c_impossible_value_l197_197445

theorem c_impossible_value (a b c : ℤ) (h : (∀ x : ℤ, (x + a) * (x + b) = x^2 + c * x - 8)) : c ≠ 4 :=
by
  sorry

end c_impossible_value_l197_197445


namespace least_sum_exponents_of_1000_l197_197136

def sum_least_exponents (n : ℕ) : ℕ :=
  if n = 1000 then 38 else 0 -- Since we only care about the case for 1000.

theorem least_sum_exponents_of_1000 :
  sum_least_exponents 1000 = 38 := by
  sorry

end least_sum_exponents_of_1000_l197_197136


namespace sequence_count_l197_197122

theorem sequence_count :
  ∃ n : ℕ, 
  (∀ a : Fin 101 → ℤ, 
    a 1 = 0 ∧ 
    a 100 = 475 ∧ 
    (∀ k : ℕ, 1 ≤ k ∧ k < 100 → |a (k + 1) - a k| = 5) → 
    n = 4851) := 
sorry

end sequence_count_l197_197122


namespace max_sum_of_abc_l197_197184

theorem max_sum_of_abc (A B C : ℕ) (h₁ : A ≠ B) (h₂ : B ≠ C) (h₃ : A ≠ C) (h₄ : A * B * C = 2310) : 
  A + B + C ≤ 52 :=
sorry

end max_sum_of_abc_l197_197184


namespace min_t_value_l197_197523

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem min_t_value : 
  ∀ (x y : ℝ), x ∈ Set.Icc (-3 : ℝ) (2 : ℝ) → y ∈ Set.Icc (-3 : ℝ) (2 : ℝ)
  → |f (x) - f (y)| ≤ 20 :=
by
  sorry

end min_t_value_l197_197523


namespace solve_prime_equation_l197_197610

def is_prime (n : ℕ) : Prop := ∀ k, k < n ∧ k > 1 → n % k ≠ 0

theorem solve_prime_equation (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r)
  (h : 5 * p = q^3 - r^3) : p = 67 ∧ q = 7 ∧ r = 2 :=
sorry

end solve_prime_equation_l197_197610


namespace integer_solution_l197_197041

theorem integer_solution (a b : ℤ) (h : 6 * a * b = 9 * a - 10 * b + 303) : a + b = 15 :=
sorry

end integer_solution_l197_197041


namespace root_interval_l197_197323

noncomputable def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_interval (h1 : f 1 < 0) (h2 : f 1.5 > 0) (h3 : f 1.25 < 0) :
  ∃ c, 1.25 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  -- Proof by the Intermediate Value Theorem
  sorry

end root_interval_l197_197323


namespace minerals_now_l197_197221

def minerals_yesterday (M : ℕ) : Prop := (M / 2 = 21)

theorem minerals_now (M : ℕ) (H : minerals_yesterday M) : (M + 6 = 48) :=
by 
  unfold minerals_yesterday at H
  sorry

end minerals_now_l197_197221


namespace length_AB_l197_197800

theorem length_AB (r : ℝ) (A B : ℝ) (π : ℝ) : 
  r = 4 ∧ π = 3 ∧ (A = 8 ∧ B = 8) → (A = B ∧ A + B = 24 → AB = 6) :=
by
  intros
  sorry

end length_AB_l197_197800


namespace recurring_decimal_to_rational_l197_197621

theorem recurring_decimal_to_rational : 
  (0.125125125 : ℝ) = 125 / 999 :=
sorry

end recurring_decimal_to_rational_l197_197621


namespace tile_arrangement_probability_l197_197096

theorem tile_arrangement_probability :
  let X := 4  -- Number of tiles marked X
  let O := 2  -- Number of tiles marked O
  let total := 6  -- Total number of tiles
  let arrangement := [true, true, false, true, false, true]  -- XXOXOX represented as [X, X, O, X, O, X]
  (↑(X / total) * ↑((X - 1) / (total - 1)) * ↑((O / (total - 2))) * ↑((X - 2) / (total - 3)) * ↑((O - 1) / (total - 4)) * 1 : ℚ) = 1 / 15 :=
sorry

end tile_arrangement_probability_l197_197096


namespace hyperbola_equation_sum_l197_197389

theorem hyperbola_equation_sum (h k a c b : ℝ) (h_h : h = 1) (h_k : k = 1) (h_a : a = 3) (h_c : c = 9) (h_c2 : c^2 = a^2 + b^2) :
    h + k + a + b = 5 + 6 * Real.sqrt 2 :=
by
  sorry

end hyperbola_equation_sum_l197_197389


namespace sum_of_3digit_numbers_remainder_2_l197_197425

-- Define the smallest and largest three-digit numbers leaving remainder 2 when divided by 5
def smallest : ℕ := 102
def largest  : ℕ := 997
def common_diff : ℕ := 5

-- Define the arithmetic sequence
def seq_length : ℕ := ((largest - smallest) / common_diff) + 1
def sequence_sum : ℕ := seq_length * (smallest + largest) / 2

-- The theorem to be proven
theorem sum_of_3digit_numbers_remainder_2 : sequence_sum = 98910 :=
by
  sorry

end sum_of_3digit_numbers_remainder_2_l197_197425


namespace altitude_circumradius_relation_l197_197095

variable (a b c R ha : ℝ)
-- Assume S is the area of the triangle
variable (S : ℝ)
-- conditions
axiom area_circumradius : S = (a * b * c) / (4 * R)
axiom area_altitude : S = (a * ha) / 2

-- Prove the equivalence
theorem altitude_circumradius_relation 
  (area_circumradius : S = (a * b * c) / (4 * R)) 
  (area_altitude : S = (a * ha) / 2) : 
  ha = (b * c) / (2 * R) :=
sorry

end altitude_circumradius_relation_l197_197095


namespace coordinate_sum_l197_197090

theorem coordinate_sum (f : ℝ → ℝ) (x y : ℝ) (h₁ : f 9 = 7) (h₂ : 3 * y = f (3 * x) / 3 + 3) (h₃ : x = 3) : 
  x + y = 43 / 9 :=
by
  -- Proof goes here
  sorry

end coordinate_sum_l197_197090


namespace thre_digit_num_condition_l197_197265

theorem thre_digit_num_condition (n : ℕ) (h : n = 735) :
  (n % 35 = 0) ∧ (Nat.digits 10 n).sum = 15 := by
  sorry

end thre_digit_num_condition_l197_197265


namespace union_prob_inconsistency_l197_197873

noncomputable def p_a : ℚ := 2/15
noncomputable def p_b : ℚ := 4/15
noncomputable def p_b_given_a : ℚ := 3

theorem union_prob_inconsistency : p_a + p_b - p_b_given_a * p_a = 0 → false := by
  sorry

end union_prob_inconsistency_l197_197873


namespace ratio_of_areas_of_triangles_l197_197648

-- Define the given conditions
variables {X Y Z T : Type}
variable (distance_XY : ℝ)
variable (distance_XZ : ℝ)
variable (distance_YZ : ℝ)
variable (is_angle_bisector : Prop)

-- Define the correct answer as a goal
theorem ratio_of_areas_of_triangles (h1 : distance_XY = 15)
    (h2 : distance_XZ = 25)
    (h3 : distance_YZ = 34)
    (h4 : is_angle_bisector) : 
    -- Ratio of the areas of triangle XYT to triangle XZT
    ∃ (ratio : ℝ), ratio = 3 / 5 :=
by
  -- This is where the proof would go, omitted with "sorry"
  sorry

end ratio_of_areas_of_triangles_l197_197648


namespace outermost_diameter_l197_197143

def radius_of_fountain := 6 -- derived from the information that 12/2 = 6
def width_of_garden := 9
def width_of_inner_walking_path := 3
def width_of_outer_walking_path := 7

theorem outermost_diameter :
  2 * (radius_of_fountain + width_of_garden + width_of_inner_walking_path + width_of_outer_walking_path) = 50 :=
by
  sorry

end outermost_diameter_l197_197143


namespace initial_average_score_l197_197274

theorem initial_average_score (A : ℝ) :
  (∃ (A : ℝ), (16 * A = 15 * 64 + 24)) → A = 61.5 := 
by 
  sorry 

end initial_average_score_l197_197274


namespace smallest_square_contains_five_disks_l197_197764

noncomputable def smallest_side_length := 2 + 2 * Real.sqrt 2

theorem smallest_square_contains_five_disks :
  ∃ (a : ℝ), a = smallest_side_length ∧ (∃ (d : ℕ → ℝ × ℝ), 
    (∀ i, 0 ≤ i ∧ i < 5 → (d i).fst ^ 2 + (d i).snd ^ 2 < (a / 2 - 1) ^ 2) ∧ 
    (∀ i j, 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 ∧ i ≠ j → 
      (d i).fst ^ 2 + (d i).snd ^ 2 + (d j).fst ^ 2 + (d j).snd ^ 2 ≥ 4)) :=
sorry

end smallest_square_contains_five_disks_l197_197764


namespace length_of_segment_P_to_P_l197_197741

/-- Point P is given as (-4, 3) and P' is the reflection of P over the x-axis. 
    We need to prove that the length of the segment connecting P to P' is 6. -/
theorem length_of_segment_P_to_P' :
  let P := (-4, 3)
  let P' := (-4, -3)
  dist P P' = 6 :=
by
  sorry

end length_of_segment_P_to_P_l197_197741


namespace no_function_f_exists_l197_197685

theorem no_function_f_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2013 :=
by sorry

end no_function_f_exists_l197_197685


namespace correct_number_of_conclusions_l197_197669

def y (x : ℝ) := -5 * x + 1

def conclusion1 := y (-1) = 5
def conclusion2 := ∃ x1 x2 x3 : ℝ, y x1 > 0 ∧ y x2 > 0 ∧ y (x3) < 0 ∧ (x1 < 0) ∧ (x2 > 0) ∧ (x3 < x2)
def conclusion3 := ∀ x : ℝ, x > 1 → y x < 0
def conclusion4 := ∀ x1 x2 : ℝ, x1 < x2 → y x1 < y x2

-- We want to prove that exactly 2 of these conclusions are correct
theorem correct_number_of_conclusions : (¬ conclusion1 ∧ conclusion2 ∧ conclusion3 ∧ ¬ conclusion4) :=
by
  sorry

end correct_number_of_conclusions_l197_197669


namespace y_completes_work_in_seventy_days_l197_197624

def work_days (mahesh_days : ℕ) (mahesh_work_days : ℕ) (rajesh_days : ℕ) (y_days : ℕ) : Prop :=
  let mahesh_rate := (1:ℝ) / mahesh_days
  let rajesh_rate := (1:ℝ) / rajesh_days
  let work_done_by_mahesh := mahesh_rate * mahesh_work_days
  let remaining_work := (1:ℝ) - work_done_by_mahesh
  let rajesh_remaining_work_days := remaining_work / rajesh_rate
  let y_rate := (1:ℝ) / y_days
  y_rate = rajesh_rate

theorem y_completes_work_in_seventy_days :
  work_days 35 20 30 70 :=
by
  -- This is where the proof would go
  sorry

end y_completes_work_in_seventy_days_l197_197624


namespace probability_mixed_doubles_l197_197826

def num_athletes : ℕ := 6
def num_males : ℕ := 3
def num_females : ℕ := 3
def num_coaches : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of ways to select athletes
def total_ways : ℕ :=
  (choose num_athletes 2) * (choose (num_athletes - 2) 2) * (choose (num_athletes - 4) 2)

-- Number of favorable ways to select mixed doubles teams
def favorable_ways : ℕ :=
  (choose num_males 1) * (choose num_females 1) *
  (choose (num_males - 1) 1) * (choose (num_females - 1) 1) *
  (choose 1 1) * (choose 1 1)

-- Probability calculation
def probability : ℚ := favorable_ways / total_ways

theorem probability_mixed_doubles :
  probability = 2/5 :=
by
  sorry

end probability_mixed_doubles_l197_197826


namespace max_profit_at_max_price_l197_197028

-- Definitions based on the given problem's conditions
def cost_price : ℝ := 30
def profit_margin : ℝ := 0.5
def max_price : ℝ := cost_price * (1 + profit_margin)
def min_price : ℝ := 35
def base_sales : ℝ := 350
def sales_decrease_per_price_increase : ℝ := 50
def price_increase_step : ℝ := 5

-- Profit function based on the conditions
def profit (x : ℝ) : ℝ := (-10 * x^2 + 1000 * x - 21000)

-- Maximum profit and corresponding price
theorem max_profit_at_max_price :
  ∀ x, min_price ≤ x ∧ x ≤ max_price →
  profit x ≤ profit max_price ∧ profit max_price = 3750 :=
by sorry

end max_profit_at_max_price_l197_197028


namespace remainder_when_sum_divided_mod7_l197_197363

theorem remainder_when_sum_divided_mod7 (a b c : ℕ)
  (h1 : a < 7) (h2 : b < 7) (h3 : c < 7)
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0)
  (h7 : a * b * c % 7 = 2)
  (h8 : 3 * c % 7 = 1)
  (h9 : 4 * b % 7 = (2 + b) % 7) :
  (a + b + c) % 7 = 3 := by
  sorry

end remainder_when_sum_divided_mod7_l197_197363


namespace find_k_l197_197407

theorem find_k (x y k : ℝ) (h₁ : x = 2) (h₂ : y = -1) (h₃ : y - k * x = 7) : k = -4 :=
by
  sorry

end find_k_l197_197407


namespace arithmetic_progression_terms_even_sums_l197_197357

theorem arithmetic_progression_terms_even_sums (n a d : ℕ) (h_even : Even n) 
  (h_odd_sum : n * (a + (n - 2) * d) = 60) 
  (h_even_sum : n * (a + d + a + (n - 1) * d) = 72) 
  (h_last_first : (n - 1) * d = 12) : n = 8 := 
sorry

end arithmetic_progression_terms_even_sums_l197_197357


namespace total_cost_is_80_l197_197927

-- Conditions
def cost_flour := 3 * 3
def cost_eggs := 3 * 10
def cost_milk := 7 * 5
def cost_baking_soda := 2 * 3

-- Question and proof requirement
theorem total_cost_is_80 : cost_flour + cost_eggs + cost_milk + cost_baking_soda = 80 := by
  sorry

end total_cost_is_80_l197_197927


namespace part1_l197_197462

theorem part1 (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^2005 + z^2006 + z^2008 + z^2009 = -2 :=
  sorry

end part1_l197_197462


namespace double_rooms_percentage_l197_197466

theorem double_rooms_percentage (S : ℝ) (h1 : 0 < S)
  (h2 : ∃ Sd : ℝ, Sd = 0.75 * S)
  (h3 : ∃ Ss : ℝ, Ss = 0.25 * S):
  (0.375 * S) / (0.625 * S) * 100 = 60 := 
by 
  sorry

end double_rooms_percentage_l197_197466


namespace no_adjacent_standing_prob_l197_197737

def coin_flip_probability : ℚ :=
  let a2 := 3
  let a3 := 4
  let a4 := a3 + a2
  let a5 := a4 + a3
  let a6 := a5 + a4
  let a7 := a6 + a5
  let a8 := a7 + a6
  let a9 := a8 + a7
  let a10 := a9 + a8
  let favorable_outcomes := a10
  favorable_outcomes / (2 ^ 10)

theorem no_adjacent_standing_prob :
  coin_flip_probability = (123 / 1024 : ℚ) :=
by sorry

end no_adjacent_standing_prob_l197_197737


namespace parallelogram_height_l197_197547

theorem parallelogram_height
  (area : ℝ)
  (base : ℝ)
  (h_area : area = 375)
  (h_base : base = 25) :
  (area / base) = 15 :=
by
  sorry

end parallelogram_height_l197_197547


namespace Theresa_video_games_l197_197931

variable (Tory Julia Theresa : ℕ)

def condition1 : Prop := Tory = 6
def condition2 : Prop := Julia = Tory / 3
def condition3 : Prop := Theresa = (Julia * 3) + 5

theorem Theresa_video_games : condition1 Tory → condition2 Tory Julia → condition3 Julia Theresa → Theresa = 11 := by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  sorry

end Theresa_video_games_l197_197931


namespace positive_integer_base_conversion_l197_197840

theorem positive_integer_base_conversion (A B : ℕ) (h1 : A < 9) (h2 : B < 7) 
(h3 : 9 * A + B = 7 * B + A) : 9 * 3 + 4 = 31 :=
by sorry

end positive_integer_base_conversion_l197_197840


namespace neg_sqrt_17_bounds_l197_197590

theorem neg_sqrt_17_bounds :
  (16 < 17) ∧ (17 < 25) ∧ (16 = 4^2) ∧ (25 = 5^2) ∧ (4 < Real.sqrt 17) ∧ (Real.sqrt 17 < 5) →
  (-5 < -Real.sqrt 17) ∧ (-Real.sqrt 17 < -4) :=
by
  sorry

end neg_sqrt_17_bounds_l197_197590


namespace brick_length_is_50_l197_197277

theorem brick_length_is_50
  (x : ℝ)
  (brick_volume_eq : x * 11.25 * 6 * 3200 = 800 * 600 * 22.5) :
  x = 50 :=
by
  sorry

end brick_length_is_50_l197_197277


namespace find_constants_l197_197487

theorem find_constants :
  ∃ A B C D : ℚ,
    (∀ x : ℚ,
      x ≠ 2 → x ≠ 3 → x ≠ 5 → x ≠ -1 →
      (x^2 - 9) / ((x - 2) * (x - 3) * (x - 5) * (x + 1)) =
      A / (x - 2) + B / (x - 3) + C / (x - 5) + D / (x + 1)) ∧
  A = -5/9 ∧ B = 0 ∧ C = 4/9 ∧ D = -1/9 :=
by
  sorry

end find_constants_l197_197487


namespace solve_abs_inequality_l197_197729

theorem solve_abs_inequality (x : ℝ) :
  (|x-2| ≥ |x|) → x ≤ 1 :=
by
  sorry

end solve_abs_inequality_l197_197729


namespace added_number_is_five_l197_197786

def original_number := 19
def final_resultant := 129
def doubling_expression (x : ℕ) (y : ℕ) := 3 * (2 * x + y)

theorem added_number_is_five:
  ∃ y, doubling_expression original_number y = final_resultant ↔ y = 5 :=
sorry

end added_number_is_five_l197_197786


namespace exists_pos_int_such_sqrt_not_int_l197_197268

theorem exists_pos_int_such_sqrt_not_int (a b c : ℤ) : ∃ n : ℕ, 0 < n ∧ ¬∃ k : ℤ, k * k = n^3 + a * n^2 + b * n + c :=
by
  sorry

end exists_pos_int_such_sqrt_not_int_l197_197268


namespace sum_of_squares_of_consecutive_integers_l197_197553

theorem sum_of_squares_of_consecutive_integers (b : ℕ) (h : (b-1) * b * (b+1) = 12 * ((b-1) + b + (b+1))) : 
  (b - 1) * (b - 1) + b * b + (b + 1) * (b + 1) = 110 := 
by sorry

end sum_of_squares_of_consecutive_integers_l197_197553


namespace conditional_probability_l197_197199

-- Given probabilities:
def p_a : ℚ := 5/23
def p_b : ℚ := 7/23
def p_c : ℚ := 1/23
def p_a_and_b : ℚ := 2/23
def p_a_and_c : ℚ := 1/23
def p_b_and_c : ℚ := 1/23
def p_a_and_b_and_c : ℚ := 1/23

-- Theorem statement to prove:
theorem conditional_probability : p_a_and_b_and_c / p_a_and_c = 1 :=
by
  sorry

end conditional_probability_l197_197199


namespace digit_at_position_2020_l197_197098

def sequence_digit (n : Nat) : Nat :=
  -- Function to return the nth digit of the sequence formed by concatenating the integers from 1 to 1000
  sorry

theorem digit_at_position_2020 : sequence_digit 2020 = 7 :=
  sorry

end digit_at_position_2020_l197_197098


namespace mary_needs_more_apples_l197_197282

theorem mary_needs_more_apples (total_pies : ℕ) (apples_per_pie : ℕ) (harvested_apples : ℕ) (y : ℕ) :
  total_pies = 10 → apples_per_pie = 8 → harvested_apples = 50 → y = 30 :=
by
  intro h1 h2 h3
  have total_apples_needed := total_pies * apples_per_pie
  have apples_needed_to_buy := total_apples_needed - harvested_apples
  have proof_needed : apples_needed_to_buy = y := sorry
  have proof_given : y = 30 := sorry
  have apples_needed := total_pies * apples_per_pie - harvested_apples
  exact proof_given

end mary_needs_more_apples_l197_197282


namespace basketball_committee_l197_197619

theorem basketball_committee (total_players guards : ℕ) (choose_committee choose_guard : ℕ) :
  total_players = 12 → guards = 4 → choose_committee = 3 → choose_guard = 1 →
  (guards * ((total_players - guards).choose (choose_committee - choose_guard)) = 112) :=
by
  intros h_tp h_g h_cc h_cg
  rw [h_tp, h_g, h_cc, h_cg]
  simp
  norm_num
  sorry

end basketball_committee_l197_197619


namespace arithmetic_mean_of_arithmetic_progression_l197_197952

variable (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ)

/-- General term of an arithmetic progression -/
def arithmetic_progression (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

theorem arithmetic_mean_of_arithmetic_progression (k p : ℕ) (hk : 1 < k) :
  a k = (a (k - p) + a (k + p)) / 2 := by
  sorry

end arithmetic_mean_of_arithmetic_progression_l197_197952


namespace geometric_sequence_general_term_l197_197970

theorem geometric_sequence_general_term :
  ∀ (n : ℕ), (n > 0) →
  (∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ (k : ℕ), k > 0 → a (k+1) = 2 * a k) ∧ a n = 2^(n-1)) :=
by
  sorry

end geometric_sequence_general_term_l197_197970


namespace calculate_weight_l197_197472

theorem calculate_weight (W : ℝ) (h : 0.75 * W + 2 = 62) : W = 80 :=
by
  sorry

end calculate_weight_l197_197472


namespace square_perimeter_l197_197369

theorem square_perimeter (A_total : ℕ) (A_common : ℕ) (A_circle : ℕ) 
  (H1 : A_total = 329)
  (H2 : A_common = 101)
  (H3 : A_circle = 234) :
  4 * (Int.sqrt (A_total - A_circle + A_common)) = 56 :=
by
  -- Since we are only required to provide the statement, we can skip the proof steps.
  -- sorry to skip the proof.
  sorry

end square_perimeter_l197_197369


namespace woman_work_rate_l197_197644

theorem woman_work_rate (M W : ℝ) (h1 : 10 * M + 15 * W = 1 / 8) (h2 : M = 1 / 100) : W = 1 / 600 :=
by 
  sorry

end woman_work_rate_l197_197644


namespace frustum_volume_correct_l197_197761

noncomputable def volume_of_frustum 
(base_edge_original_pyramid : ℝ) (height_original_pyramid : ℝ) 
(base_edge_smaller_pyramid : ℝ) (height_smaller_pyramid : ℝ) : ℝ :=
  let base_area_original := base_edge_original_pyramid ^ 2
  let volume_original := 1 / 3 * base_area_original * height_original_pyramid
  let similarity_ratio := base_edge_smaller_pyramid / base_edge_original_pyramid
  let volume_smaller := volume_original * (similarity_ratio ^ 3)
  volume_original - volume_smaller

theorem frustum_volume_correct 
(base_edge_original_pyramid : ℝ) (height_original_pyramid : ℝ) 
(base_edge_smaller_pyramid : ℝ) (height_smaller_pyramid : ℝ) 
(h_orig_base_edge : base_edge_original_pyramid = 16) 
(h_orig_height : height_original_pyramid = 10) 
(h_smaller_base_edge : base_edge_smaller_pyramid = 8) 
(h_smaller_height : height_smaller_pyramid = 5) : 
  volume_of_frustum base_edge_original_pyramid height_original_pyramid base_edge_smaller_pyramid height_smaller_pyramid = 746.66 :=
by 
  sorry

end frustum_volume_correct_l197_197761


namespace total_cans_l197_197073

theorem total_cans (total_oil : ℕ) (oil_in_8_liter_cans : ℕ) (number_of_8_liter_cans : ℕ) (remaining_oil : ℕ) 
(oil_per_15_liter_can : ℕ) (number_of_15_liter_cans : ℕ) :
  total_oil = 290 ∧ oil_in_8_liter_cans = 8 ∧ number_of_8_liter_cans = 10 ∧ oil_per_15_liter_can = 15 ∧
  remaining_oil = total_oil - (number_of_8_liter_cans * oil_in_8_liter_cans) ∧
  number_of_15_liter_cans = remaining_oil / oil_per_15_liter_can →
  (number_of_8_liter_cans + number_of_15_liter_cans) = 24 := sorry

end total_cans_l197_197073


namespace units_digit_of_17_pow_28_l197_197793

theorem units_digit_of_17_pow_28 : ((17:ℕ) ^ 28) % 10 = 1 := by
  sorry

end units_digit_of_17_pow_28_l197_197793


namespace smallest_positive_integer_x_l197_197988

theorem smallest_positive_integer_x (x : ℕ) (h900 : ∃ a b c : ℕ, 900 = (2^a) * (3^b) * (5^c) ∧ a = 2 ∧ b = 2 ∧ c = 2) (h1152 : ∃ a b : ℕ, 1152 = (2^a) * (3^b) ∧ a = 7 ∧ b = 2) : x = 32 :=
by
  sorry

end smallest_positive_integer_x_l197_197988


namespace hotel_cost_l197_197476

theorem hotel_cost (x y : ℕ) (h1 : 3 * x + 6 * y = 1020) (h2 : x + 5 * y = 700) :
  5 * (x + y) = 1100 :=
sorry

end hotel_cost_l197_197476


namespace solution_unique_s_l197_197692

theorem solution_unique_s (s : ℝ) (hs : ⌊s⌋ + s = 22.7) : s = 11.7 :=
sorry

end solution_unique_s_l197_197692


namespace max_digit_sum_of_watch_display_l197_197336

-- Define the problem conditions
def valid_hour (h : ℕ) : Prop := 0 ≤ h ∧ h < 24
def valid_minute (m : ℕ) : Prop := 0 ≤ m ∧ m < 60
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the proof problem
theorem max_digit_sum_of_watch_display : 
  ∃ h m : ℕ, valid_hour h ∧ valid_minute m ∧ (digit_sum h + digit_sum m = 24) :=
sorry

end max_digit_sum_of_watch_display_l197_197336


namespace compare_logs_l197_197613

open Real

noncomputable def a := log 6 / log 3
noncomputable def b := 1 / log 5
noncomputable def c := log 14 / log 7

theorem compare_logs : a > b ∧ b > c := by
  sorry

end compare_logs_l197_197613


namespace libby_quarters_left_after_payment_l197_197402

noncomputable def quarters_needed (usd_target : ℝ) (usd_per_quarter : ℝ) : ℝ := 
  usd_target / usd_per_quarter

noncomputable def quarters_left (initial_quarters : ℝ) (used_quarters : ℝ) : ℝ := 
  initial_quarters - used_quarters

theorem libby_quarters_left_after_payment
  (initial_quarters : ℝ) (usd_target : ℝ) (usd_per_quarter : ℝ) 
  (h_initial : initial_quarters = 160) 
  (h_usd_target : usd_target = 35) 
  (h_usd_per_quarter : usd_per_quarter = 0.25) : 
  quarters_left initial_quarters (quarters_needed usd_target usd_per_quarter) = 20 := 
by
  sorry

end libby_quarters_left_after_payment_l197_197402


namespace pirate_treasure_probability_l197_197982

theorem pirate_treasure_probability :
  let p_treasure := 1 / 5
  let p_trap_no_treasure := 1 / 10
  let p_notreasure_notrap := 7 / 10
  let combinatorial_factor := Nat.choose 8 4
  let probability := (combinatorial_factor * (p_treasure ^ 4) * (p_notreasure_notrap ^ 4))
  probability = 33614 / 1250000 :=
by
  sorry

end pirate_treasure_probability_l197_197982


namespace total_trip_time_l197_197814

noncomputable def speed_coastal := 10 / 20  -- miles per minute
noncomputable def speed_highway := 4 * speed_coastal  -- miles per minute
noncomputable def time_highway := 50 / speed_highway  -- minutes
noncomputable def total_time := 20 + time_highway  -- minutes

theorem total_trip_time : total_time = 45 := 
by
  -- Proof omitted
  sorry

end total_trip_time_l197_197814


namespace bathing_suits_total_l197_197421

def men_bathing_suits : ℕ := 14797
def women_bathing_suits : ℕ := 4969
def total_bathing_suits : ℕ := 19766

theorem bathing_suits_total :
  men_bathing_suits + women_bathing_suits = total_bathing_suits := by
  sorry

end bathing_suits_total_l197_197421


namespace myrtle_eggs_count_l197_197480

-- Definition for daily egg production
def daily_eggs : ℕ := 3 * 3

-- Definition for the number of days Myrtle is gone
def days_gone : ℕ := 7

-- Definition for total eggs laid
def total_eggs : ℕ := daily_eggs * days_gone

-- Definition for eggs taken by neighbor
def eggs_taken_by_neighbor : ℕ := 12

-- Definition for eggs remaining after neighbor takes some
def eggs_after_neighbor : ℕ := total_eggs - eggs_taken_by_neighbor

-- Definition for eggs dropped by Myrtle
def eggs_dropped_by_myrtle : ℕ := 5

-- Definition for total remaining eggs Myrtle has
def eggs_remaining : ℕ := eggs_after_neighbor - eggs_dropped_by_myrtle

-- Theorem statement
theorem myrtle_eggs_count : eggs_remaining = 46 := by
  sorry

end myrtle_eggs_count_l197_197480


namespace inequality_solution_range_of_a_l197_197294

noncomputable def f (x : ℝ) : ℝ := |1 - 2 * x| - |1 + x| 

theorem inequality_solution (x : ℝ) : f x ≥ 4 ↔ x ≤ -2 ∨ x ≥ 6 := 
by sorry

theorem range_of_a (a x : ℝ) (h : a^2 + 2 * a + |1 + x| < f x) : -3 < a ∧ a < 1 :=
by sorry

end inequality_solution_range_of_a_l197_197294


namespace age_ratio_in_years_l197_197521

theorem age_ratio_in_years (p c x : ℕ) 
  (H1 : p - 2 = 3 * (c - 2)) 
  (H2 : p - 4 = 4 * (c - 4)) 
  (H3 : (p + x) / (c + x) = 2) : 
  x = 4 :=
sorry

end age_ratio_in_years_l197_197521


namespace parabola_translation_correct_l197_197267

variable (x : ℝ)

def original_parabola : ℝ := 5 * x^2

def translated_parabola : ℝ := 5 * (x - 2)^2 + 3

theorem parabola_translation_correct :
  translated_parabola x = 5 * (x - 2)^2 + 3 :=
by
  sorry

end parabola_translation_correct_l197_197267


namespace domain_of_log_base_5_range_of_3_pow_neg_l197_197875

theorem domain_of_log_base_5 (x : ℝ) : (1 - x > 0) -> (x < 1) :=
sorry

theorem range_of_3_pow_neg (y : ℝ) : (∃ x : ℝ, y = 3 ^ (-x)) -> (y > 0) :=
sorry

end domain_of_log_base_5_range_of_3_pow_neg_l197_197875


namespace sector_area_l197_197614

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) (area : ℝ) 
  (h1 : arc_length = 6) 
  (h2 : central_angle = 2) 
  (h3 : radius = arc_length / central_angle): 
  area = (1 / 2) * arc_length * radius := 
  sorry

end sector_area_l197_197614


namespace solve_for_y_l197_197757

theorem solve_for_y : ∃ y : ℕ, 8^4 = 2^y ∧ y = 12 := by
  sorry

end solve_for_y_l197_197757


namespace evaluate_expression_l197_197587

noncomputable def g (A B C D x : ℝ) : ℝ := A * x^3 + B * x^2 - C * x + D

theorem evaluate_expression (A B C D : ℝ) (h1 : g A B C D 2 = 5) (h2 : g A B C D (-1) = -8) (h3 : g A B C D 0 = 2) :
  -12 * A + 6 * B - 3 * C + D = 27.5 :=
by
  sorry

end evaluate_expression_l197_197587


namespace range_of_a_l197_197591

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ 2 * x^2 - a * x + 2 > 0) ↔ a < 5 := sorry

end range_of_a_l197_197591


namespace line_y2_not_pass_second_quadrant_l197_197559

theorem line_y2_not_pass_second_quadrant {a b : ℝ} (h1 : a < 0) (h2 : b > 0) :
  ¬∃ x : ℝ, x < 0 ∧ bx + a > 0 :=
by
  sorry

end line_y2_not_pass_second_quadrant_l197_197559


namespace find_m_l197_197322

theorem find_m
  (m : ℝ)
  (A B : ℝ × ℝ × ℝ)
  (hA : A = (m, 2, 3))
  (hB : B = (1, -1, 1))
  (h_dist : (Real.sqrt ((m - 1) ^ 2 + (2 - (-1)) ^ 2 + (3 - 1) ^ 2) = Real.sqrt 13)) :
  m = 1 := 
sorry

end find_m_l197_197322


namespace jackie_free_time_correct_l197_197639

noncomputable def jackie_free_time : ℕ :=
  let total_hours_in_a_day := 24
  let hours_working := 8
  let hours_exercising := 3
  let hours_sleeping := 8
  let total_activity_hours := hours_working + hours_exercising + hours_sleeping
  total_hours_in_a_day - total_activity_hours

theorem jackie_free_time_correct : jackie_free_time = 5 := by
  sorry

end jackie_free_time_correct_l197_197639


namespace catering_service_comparison_l197_197753

theorem catering_service_comparison :
  ∃ (x : ℕ), 150 + 18 * x > 250 + 15 * x ∧ (∀ y : ℕ, y < x -> (150 + 18 * y ≤ 250 + 15 * y)) ∧ x = 34 :=
sorry

end catering_service_comparison_l197_197753


namespace christine_savings_l197_197332

def commission_rate : ℝ := 0.12
def total_sales : ℝ := 24000
def personal_needs_percentage : ℝ := 0.60
def savings_percentage : ℝ := 1 - personal_needs_percentage

noncomputable def commission_earned : ℝ := total_sales * commission_rate
noncomputable def amount_saved : ℝ := commission_earned * savings_percentage

theorem christine_savings :
  amount_saved = 1152 :=
by
  sorry

end christine_savings_l197_197332


namespace max_expression_value_l197_197033

noncomputable def A : ℝ := 15682 + (1 / 3579)
noncomputable def B : ℝ := 15682 - (1 / 3579)
noncomputable def C : ℝ := 15682 * (1 / 3579)
noncomputable def D : ℝ := 15682 / (1 / 3579)
noncomputable def E : ℝ := 15682.3579

theorem max_expression_value :
  D = 56109138 ∧ D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end max_expression_value_l197_197033


namespace trig_relationship_l197_197147

theorem trig_relationship : 
  let a := Real.sin (145 * Real.pi / 180)
  let b := Real.cos (52 * Real.pi / 180)
  let c := Real.tan (47 * Real.pi / 180)
  a < b ∧ b < c :=
by 
  sorry

end trig_relationship_l197_197147


namespace sharon_trip_distance_l197_197762

noncomputable def usual_speed (x : ℝ) : ℝ := x / 180
noncomputable def reduced_speed (x : ℝ) : ℝ := usual_speed x - 25 / 60
noncomputable def increased_speed (x : ℝ) : ℝ := usual_speed x + 10 / 60
noncomputable def pre_storm_time : ℝ := 60
noncomputable def total_time : ℝ := 300

theorem sharon_trip_distance : 
  ∀ (x : ℝ), 
  60 + (x / 3) / reduced_speed x + (x / 3) / increased_speed x = 240 → 
  x = 135 :=
sorry

end sharon_trip_distance_l197_197762


namespace smallest_positive_n_l197_197981

theorem smallest_positive_n (n : ℕ) (h : 1023 * n % 30 = 2147 * n % 30) : n = 15 :=
by
  sorry

end smallest_positive_n_l197_197981


namespace arithmetic_seq_sum_l197_197898

/-- Given an arithmetic sequence {a_n} such that a_5 + a_6 + a_7 = 15,
prove that the sum of the first 11 terms of the sequence S_11 is 55. -/
theorem arithmetic_seq_sum (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₁ : a 5 + a 6 + a 7 = 15)
  (h₂ : ∀ n, S n = n * (a 1 + a n) / 2) :
  S 11 = 55 :=
sorry

end arithmetic_seq_sum_l197_197898


namespace xyz_poly_identity_l197_197655

theorem xyz_poly_identity (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)
  (h4 : x + y + z = 0) (h5 : xy + xz + yz ≠ 0) :
  (x^6 + y^6 + z^6) / (xyz * (xy + xz + yz)) = 6 :=
by
  sorry

end xyz_poly_identity_l197_197655


namespace incorrect_weight_conclusion_l197_197722

theorem incorrect_weight_conclusion (x y : ℝ) (h1 : y = 0.85 * x - 85.71) :
  ¬ (x = 160 → y = 50.29) :=
sorry

end incorrect_weight_conclusion_l197_197722


namespace total_pages_in_storybook_l197_197519

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

end total_pages_in_storybook_l197_197519


namespace lowest_point_in_fourth_quadrant_l197_197345

theorem lowest_point_in_fourth_quadrant (k : ℝ) (h : k < -1) :
    let x := - (k + 1) / 2
    let y := (4 * k - (k + 1) ^ 2) / 4
    y < 0 ∧ x > 0 :=
by
  let x := - (k + 1) / 2
  let y := (4 * k - (k + 1) ^ 2) / 4
  sorry

end lowest_point_in_fourth_quadrant_l197_197345


namespace arithmetic_seq_sum_l197_197330

theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d) (h_a5 : a 5 = 15) :
  a 3 + a 4 + a 6 + a 7 = 60 :=
sorry

end arithmetic_seq_sum_l197_197330


namespace hiking_packing_weight_l197_197698

theorem hiking_packing_weight
  (miles_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days : ℝ)
  (supply_per_mile : ℝ)
  (resupply_fraction : ℝ)
  (expected_first_pack_weight : ℝ)
  (hiking_hours : ℝ := hours_per_day * days)
  (total_miles : ℝ := miles_per_hour * hiking_hours)
  (total_weight_needed : ℝ := supply_per_mile * total_miles)
  (resupply_weight : ℝ := resupply_fraction * total_weight_needed)
  (first_pack_weight : ℝ := total_weight_needed - resupply_weight) :
  first_pack_weight = 37.5 :=
by
  -- The proof goes here, but is omitted since the proof is not required.
  sorry

end hiking_packing_weight_l197_197698


namespace total_morning_afternoon_emails_l197_197110

-- Define the conditions
def morning_emails : ℕ := 5
def afternoon_emails : ℕ := 8
def evening_emails : ℕ := 72

-- State the proof problem
theorem total_morning_afternoon_emails : 
  morning_emails + afternoon_emails = 13 := by
  sorry

end total_morning_afternoon_emails_l197_197110


namespace minimum_value_l197_197491

-- Define the geometric sequence and its conditions
variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (positive : ∀ n, 0 < a n)
variable (geometric_seq : ∀ n, a (n+1) = q * a n)
variable (condition1 : a 6 = a 5 + 2 * a 4)
variable (m n : ℕ)
variable (condition2 : ∀ m n, sqrt (a m * a n) = 2 * a 1 → a m = a n)

-- Prove that the minimum value of 1/m + 9/n is 4
theorem minimum_value : m + n = 4 → (∀ x y : ℝ, (0 < x ∧ 0 < y) → (1 / x + 9 / y) ≥ 4) :=
sorry

end minimum_value_l197_197491


namespace sin_two_alpha_sub_pi_eq_24_div_25_l197_197501

noncomputable def pi_div_2 : ℝ := Real.pi / 2

theorem sin_two_alpha_sub_pi_eq_24_div_25
  (α : ℝ) 
  (h1 : pi_div_2 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.tan (α + Real.pi / 4) = -1 / 7) : 
  Real.sin (2 * α - Real.pi) = 24 / 25 := 
sorry

end sin_two_alpha_sub_pi_eq_24_div_25_l197_197501


namespace sales_tax_difference_l197_197785

theorem sales_tax_difference
  (price : ℝ)
  (rate1 rate2 : ℝ)
  (h_rate1 : rate1 = 0.075)
  (h_rate2 : rate2 = 0.07)
  (h_price : price = 30) :
  (price * rate1 - price * rate2 = 0.15) :=
by
  sorry

end sales_tax_difference_l197_197785


namespace find_a_b_l197_197640

noncomputable def A : Set ℝ := { x : ℝ | -1 < x ∧ x < 3 }
noncomputable def B : Set ℝ := { x : ℝ | -3 < x ∧ x < 2 }
noncomputable def sol_set (a b : ℝ) : Set ℝ := { x : ℝ | x^2 + a * x + b < 0 }

theorem find_a_b :
  (sol_set (-2) (3 - 6)) = A ∩ B → (-1) + (-2) = -3 :=
by
  intros h1
  sorry

end find_a_b_l197_197640


namespace remaining_area_l197_197396

-- Given a regular hexagon and a rhombus composed of two equilateral triangles.
-- Hexagon area is 135 square centimeters.

variable (hexagon_area : ℝ) (rhombus_area : ℝ)
variable (is_regular_hexagon : Prop) (is_composed_of_two_equilateral_triangles : Prop)

-- The conditions
def correct_hexagon_area := hexagon_area = 135
def rhombus_is_composed := is_composed_of_two_equilateral_triangles = true
def hexagon_is_regular := is_regular_hexagon = true

-- Goal: Remaining area after cutting out the rhombus should be 75 square centimeters
theorem remaining_area : 
  correct_hexagon_area hexagon_area →
  hexagon_is_regular is_regular_hexagon →
  rhombus_is_composed is_composed_of_two_equilateral_triangles →
  hexagon_area - rhombus_area = 75 :=
by
  sorry

end remaining_area_l197_197396


namespace light_flash_fraction_l197_197435

theorem light_flash_fraction (flash_interval : ℕ) (total_flashes : ℕ) (seconds_in_hour : ℕ) (fraction_of_hour : ℚ) :
  flash_interval = 6 →
  total_flashes = 600 →
  seconds_in_hour = 3600 →
  fraction_of_hour = 1 →
  (total_flashes * flash_interval) / seconds_in_hour = fraction_of_hour := by
  sorry

end light_flash_fraction_l197_197435


namespace remainder_T2015_mod_12_eq_8_l197_197300

-- Define sequences of length n consisting of the letters A and B,
-- with no more than two A's in a row and no more than two B's in a row
def T : ℕ → ℕ :=
  sorry  -- Definition for T(n) must follow the given rules

-- Theorem to prove that T(2015) modulo 12 equals 8
theorem remainder_T2015_mod_12_eq_8 :
  (T 2015) % 12 = 8 :=
  sorry

end remainder_T2015_mod_12_eq_8_l197_197300


namespace runner_time_difference_l197_197966

theorem runner_time_difference 
  (v : ℝ)  -- runner's initial speed (miles per hour)
  (H1 : 0 < v)  -- speed is positive
  (d : ℝ)  -- total distance
  (H2 : d = 40)  -- total distance condition
  (t2 : ℝ)  -- time taken for the second half
  (H3 : t2 = 10)  -- second half time condition
  (H4 : v ≠ 0)  -- initial speed cannot be zero
  (H5: 20 = 10 * (v / 2))  -- equation derived from the second half conditions
  : (t2 - (20 / v)) = 5 := 
by
  sorry

end runner_time_difference_l197_197966


namespace square_measurement_error_l197_197645

theorem square_measurement_error (S S' : ℝ) (error_percentage : ℝ)
  (area_error_percentage : ℝ) (h1 : area_error_percentage = 2.01) :
  error_percentage = 1 :=
by
  sorry

end square_measurement_error_l197_197645


namespace least_five_digit_perfect_square_cube_l197_197190

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l197_197190


namespace circle_center_radius_l197_197987

open Real

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 6*x = 0 ↔ (x - 3)^2 + y^2 = 9 :=
by sorry

end circle_center_radius_l197_197987


namespace ceil_square_of_neg_seven_fourths_l197_197055

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l197_197055


namespace tangent_circle_equation_l197_197565

theorem tangent_circle_equation :
  (∃ m : Real, ∃ n : Real,
    (∀ x y : Real, (x - m)^2 + (y - n)^2 = 36) ∧ 
    ((m - 0)^2 + (n - 3)^2 = 25) ∧
    n = 6 ∧ (m = 4 ∨ m = -4)) :=
sorry

end tangent_circle_equation_l197_197565


namespace evaluate_expression_l197_197570

theorem evaluate_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ( (1 / a^2 + 1 / b^2)⁻¹ = a^2 * b^2 / (a^2 + b^2) ) :=
by
  sorry

end evaluate_expression_l197_197570


namespace problem_omega_pow_l197_197561

noncomputable def omega : ℂ := Complex.I -- Define a non-real root for x² = 1; an example choice could be i, the imaginary unit.

theorem problem_omega_pow :
  omega^2 = 1 → 
  (1 - omega + omega^2)^6 + (1 + omega - omega^2)^6 = 730 := 
by
  intro h1
  -- proof steps omitted
  sorry

end problem_omega_pow_l197_197561


namespace math_expression_evaluation_l197_197738

theorem math_expression_evaluation :
  |1 - Real.sqrt 3| + 3 * Real.tan (Real.pi / 6) - (1/2)⁻¹ + (3 - Real.pi)^0 = 3.732 + Real.sqrt 3 := by
  sorry

end math_expression_evaluation_l197_197738


namespace p_has_49_l197_197175

theorem p_has_49 (P : ℝ) (h : P = (2/7) * P + 35) : P = 49 :=
by
  sorry

end p_has_49_l197_197175


namespace parallel_lines_slope_eq_l197_197253

theorem parallel_lines_slope_eq (a : ℝ) : (∀ x y : ℝ, 3 * y - 4 * a = 8 * x) ∧ (∀ x y : ℝ, y - 2 = (a + 4) * x) → a = -4 / 3 :=
by
  sorry

end parallel_lines_slope_eq_l197_197253


namespace minimum_a_for_f_leq_one_range_of_a_for_max_value_l197_197812

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * log x - (1 / 3) * a * x^3 + 2 * x

theorem minimum_a_for_f_leq_one :
  ∀ {a : ℝ}, (a > 0) → (∀ x : ℝ, f a x ≤ 1) → (a ≥ 3) :=
sorry

theorem range_of_a_for_max_value :
  ∀ {a : ℝ}, (a > 0) → (∃ B : ℝ, ∀ x : ℝ, f a x ≤ B) ↔ (0 < a ∧ a ≤ (3 / 2) * exp 3) :=
sorry

end minimum_a_for_f_leq_one_range_of_a_for_max_value_l197_197812


namespace find_number_l197_197455

-- Define the conditions
variables (x : ℝ)
axiom condition : (4/3) * x = 45

-- Prove the main statement
theorem find_number : x = 135 / 4 :=
by
  sorry

end find_number_l197_197455


namespace solve_eqn_l197_197932

theorem solve_eqn (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 56) : x + y = 2 := by
  sorry

end solve_eqn_l197_197932


namespace circle_line_intersection_points_l197_197848

theorem circle_line_intersection_points :
  let circle_eqn : ℝ × ℝ → Prop := fun p => (p.1 - 1)^2 + p.2^2 = 16
  let line_eqn  : ℝ × ℝ → Prop := fun p => p.1 = 4
  ∃ (p₁ p₂ : ℝ × ℝ), 
    circle_eqn p₁ ∧ line_eqn p₁ ∧ circle_eqn p₂ ∧ line_eqn p₂ ∧ p₁ ≠ p₂ 
      → ∀ (p : ℝ × ℝ), circle_eqn p ∧ line_eqn p → 
        p = p₁ ∨ p = p₂ ∧ (p₁ ≠ p ∨ p₂ ≠ p)
 := sorry

end circle_line_intersection_points_l197_197848


namespace fg_3_eq_123_l197_197881

def f (x : ℤ) : ℤ := x^2 + 2
def g (x : ℤ) : ℤ := 3 * x + 2

theorem fg_3_eq_123 : f (g 3) = 123 := by
  sorry

end fg_3_eq_123_l197_197881


namespace beads_problem_l197_197442

noncomputable def number_of_blue_beads (total_beads : ℕ) (beads_with_blue_neighbor : ℕ) (beads_with_green_neighbor : ℕ) : ℕ :=
  let beads_with_both_neighbors := beads_with_blue_neighbor + beads_with_green_neighbor - total_beads
  let beads_with_only_blue_neighbor := beads_with_blue_neighbor - beads_with_both_neighbors
  (2 * beads_with_only_blue_neighbor + beads_with_both_neighbors) / 2

theorem beads_problem : number_of_blue_beads 30 26 20 = 18 := by 
  -- ...
  sorry

end beads_problem_l197_197442


namespace determine_correct_path_l197_197164

variable (A B C : Type)
variable (truthful : A → Prop)
variable (whimsical : A → Prop)
variable (answers : A → Prop)
variable (path_correct : A → Prop)

-- Conditions
axiom two_truthful_one_whimsical (x y z : A) : (truthful x ∧ truthful y ∧ whimsical z) ∨ 
                                                (truthful x ∧ truthful z ∧ whimsical y) ∨ 
                                                (truthful y ∧ truthful z ∧ whimsical x)

axiom traveler_aware : ∀ x y : A, truthful x → ¬ truthful y
axiom siblings : A → B → C → Prop
axiom ask_sibling : A → B → C → Prop

-- Conditions formalized
axiom ask_about_truthfulness (x y : A) : answers x → (truthful y ↔ ¬truthful y)

theorem determine_correct_path (x y z : A) :
  (truthful x ∧ ¬truthful y ∧ path_correct x) ∨
  (¬truthful x ∧ truthful y ∧ path_correct y) ∨
  (¬truthful x ∧ ¬truthful y ∧ truthful z ∧ path_correct z) :=
sorry

end determine_correct_path_l197_197164


namespace max_volume_of_pyramid_l197_197387

theorem max_volume_of_pyramid
  (a b c : ℝ)
  (h1 : a + b + c = 9)
  (h2 : ∀ (α β : ℝ), α = 30 ∧ β = 45)
  : ∃ V, V = (9 * Real.sqrt 2) / 4 ∧ V = (1/6) * (Real.sqrt 2 / 2) * a * b * c :=
by
  sorry

end max_volume_of_pyramid_l197_197387


namespace exponential_equation_solution_l197_197353

theorem exponential_equation_solution (b x : ℝ) (hb : b > 1) (hx : x > 0)
  (h : (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) :
  x = (3 / 5)^(Real.log b / Real.log (5 / 3)) :=
by
  sorry

end exponential_equation_solution_l197_197353


namespace find_the_number_l197_197204

-- Define the number we are trying to find
variable (x : ℝ)

-- Define the main condition from the problem
def main_condition : Prop := 0.7 * x - 40 = 30

-- Formalize the goal to prove
theorem find_the_number (h : main_condition x) : x = 100 :=
by
  -- Placeholder for the proof
  sorry

end find_the_number_l197_197204


namespace price_per_maple_tree_l197_197195

theorem price_per_maple_tree 
  (cabin_price : ℕ) (initial_cash : ℕ) (remaining_cash : ℕ)
  (num_cypress : ℕ) (price_cypress : ℕ)
  (num_pine : ℕ) (price_pine : ℕ)
  (num_maple : ℕ) 
  (total_raised_from_trees : ℕ) :
  cabin_price = 129000 ∧ 
  initial_cash = 150 ∧ 
  remaining_cash = 350 ∧ 
  num_cypress = 20 ∧ 
  price_cypress = 100 ∧ 
  num_pine = 600 ∧ 
  price_pine = 200 ∧ 
  num_maple = 24 ∧ 
  total_raised_from_trees = 129350 - initial_cash → 
  (price_maple : ℕ) = 300 :=
by 
  sorry

end price_per_maple_tree_l197_197195


namespace picture_books_count_l197_197408

-- Definitions based on the given conditions
def total_books : ℕ := 35
def fiction_books : ℕ := 5
def non_fiction_books : ℕ := fiction_books + 4
def autobiographies : ℕ := 2 * fiction_books
def total_non_picture_books : ℕ := fiction_books + non_fiction_books + autobiographies
def picture_books : ℕ := total_books - total_non_picture_books

-- Statement of the problem
theorem picture_books_count : picture_books = 11 :=
by sorry

end picture_books_count_l197_197408


namespace transport_cost_l197_197465

theorem transport_cost (cost_per_kg : ℝ) (weight_g : ℝ) : 
  (cost_per_kg = 30000) → (weight_g = 400) → 
  ((weight_g / 1000) * cost_per_kg = 12000) :=
by
  intros h1 h2
  sorry

end transport_cost_l197_197465


namespace carl_additional_gift_bags_l197_197545

theorem carl_additional_gift_bags (definite_visitors additional_visitors extravagant_bags average_bags total_bags_needed : ℕ) :
  definite_visitors = 50 →
  additional_visitors = 40 →
  extravagant_bags = 10 →
  average_bags = 20 →
  total_bags_needed = 90 →
  (total_bags_needed - (extravagant_bags + average_bags)) = 60 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end carl_additional_gift_bags_l197_197545
