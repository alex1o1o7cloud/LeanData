import Mathlib

namespace arithmetic_sequence_1001th_term_l1013_101366

theorem arithmetic_sequence_1001th_term (p q : ℤ)
  (h1 : 9 - p = (2 * q - 5))
  (h2 : (3 * p - q + 7) - 9 = (2 * q - 5)) :
  p + (1000 * (2 * q - 5)) = 5004 :=
by
  sorry

end arithmetic_sequence_1001th_term_l1013_101366


namespace sine_cos_suffices_sine_cos_necessary_l1013_101379

theorem sine_cos_suffices
  (a b c : ℝ)
  (h : ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) :
  c > Real.sqrt (a^2 + b^2) :=
sorry

theorem sine_cos_necessary
  (a b c : ℝ)
  (h : c > Real.sqrt (a^2 + b^2)) :
  ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0 :=
sorry

end sine_cos_suffices_sine_cos_necessary_l1013_101379


namespace profit_percentage_calculation_l1013_101372

noncomputable def profit_percentage (SP CP : ℝ) : ℝ := ((SP - CP) / CP) * 100

theorem profit_percentage_calculation (SP : ℝ) (h : CP = 0.92 * SP) : |profit_percentage SP (0.92 * SP) - 8.70| < 0.01 :=
by
  sorry

end profit_percentage_calculation_l1013_101372


namespace range_of_g_le_2_minus_x_l1013_101355

noncomputable def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ :=
if x ≥ 0 then f x else -f (-x)

theorem range_of_g_le_2_minus_x : {x : ℝ | g x ≤ 2 - x} = {x : ℝ | x ≤ 1} :=
by sorry

end range_of_g_le_2_minus_x_l1013_101355


namespace odd_square_divisors_l1013_101363

theorem odd_square_divisors (n : ℕ) (h_odd : n % 2 = 1) : 
  ∃ (f g : ℕ), (f > g) ∧ (∀ d, d ∣ (n * n) → d % 4 = 1 ↔ (0 < f)) ∧ (∀ d, d ∣ (n * n) → d % 4 = 3 ↔ (0 < g)) :=
by
  sorry

end odd_square_divisors_l1013_101363


namespace C_plus_D_l1013_101304

theorem C_plus_D (D C : ℚ) (h1 : ∀ x : ℚ, (Dx - 17) / ((x - 2) * (x - 4)) = C / (x - 2) + 4 / (x - 4))
  (h2 : ∀ x : ℚ, (x - 2) * (x - 4) = x^2 - 6 * x + 8) :
  C + D = 8.5 := sorry

end C_plus_D_l1013_101304


namespace last_three_digits_of_8_pow_104_l1013_101332

def last_three_digits_of_pow (x n : ℕ) : ℕ :=
  (x ^ n) % 1000

theorem last_three_digits_of_8_pow_104 : last_three_digits_of_pow 8 104 = 984 := 
by
  sorry

end last_three_digits_of_8_pow_104_l1013_101332


namespace sufficient_but_not_necessary_condition_l1013_101300

theorem sufficient_but_not_necessary_condition (a : ℝ) (h₁ : a > 2) : a ≥ 1 ∧ ¬(∀ (a : ℝ), a ≥ 1 → a > 2) := 
by
  sorry

end sufficient_but_not_necessary_condition_l1013_101300


namespace calc_f_five_times_l1013_101369

def f (x : ℕ) : ℕ := if x % 2 = 0 then x / 2 else 5 * x + 1

theorem calc_f_five_times : f (f (f (f (f 5)))) = 166 :=
by 
  sorry

end calc_f_five_times_l1013_101369


namespace part1_intersection_part2_range_of_m_l1013_101359

-- Define the universal set and the sets A and B
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0 ∨ x > 3}
def B (m : ℝ) : Set ℝ := {x | x < m - 1 ∨ x > 2 * m}

-- Part (1): When m = 3, find A ∩ B
theorem part1_intersection:
  A ∩ B 3 = {x | x < 0 ∨ x > 6} :=
sorry

-- Part (2): If B ∪ A = B, find the range of values for m
theorem part2_range_of_m (m : ℝ) :
  (B m ∪ A = B m) → (1 ≤ m ∧ m ≤ 3 / 2) :=
sorry

end part1_intersection_part2_range_of_m_l1013_101359


namespace original_price_l1013_101394

theorem original_price 
  (SP : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h1 : SP = 15)
  (h2 : gain_percent = 0.50)
  (h3 : SP = P * (1 + gain_percent)) :
  P = 10 :=
by
  sorry

end original_price_l1013_101394


namespace rolls_sold_to_uncle_l1013_101361

theorem rolls_sold_to_uncle (total_rolls : ℕ) (rolls_grandmother : ℕ) (rolls_neighbor : ℕ) (rolls_remaining : ℕ) (rolls_uncle : ℕ) :
  total_rolls = 12 →
  rolls_grandmother = 3 →
  rolls_neighbor = 3 →
  rolls_remaining = 2 →
  rolls_uncle = total_rolls - rolls_remaining - (rolls_grandmother + rolls_neighbor) →
  rolls_uncle = 4 :=
by
  intros h_total h_grandmother h_neighbor h_remaining h_compute
  rw [h_total, h_grandmother, h_neighbor, h_remaining] at h_compute
  exact h_compute

end rolls_sold_to_uncle_l1013_101361


namespace initial_books_l1013_101317

-- Definitions for the conditions.

def boxes (b : ℕ) : ℕ := 3 * b -- Box count
def booksInRoom : ℕ := 21 -- Books in the room
def booksOnTable : ℕ := 4 -- Books on the coffee table
def cookbooks : ℕ := 18 -- Cookbooks in the kitchen
def booksGrabbed : ℕ := 12 -- Books grabbed from the donation center
def booksNow : ℕ := 23 -- Books Henry has now

-- Define total number of books donated
def totalBooksDonated (inBoxes : ℕ) (additionalBooks : ℕ) : ℕ :=
  inBoxes + additionalBooks - booksGrabbed

-- Define number of books Henry initially had
def initialBooks (netDonated : ℕ) (booksCurrently : ℕ) : ℕ :=
  netDonated + booksCurrently

-- Proof goal
theorem initial_books (b : ℕ) (inBox : ℕ) (additionalBooks : ℕ) : 
  let totalBooks := booksInRoom + booksOnTable + cookbooks
  let inBoxes := boxes b
  let totalDonated := totalBooksDonated inBoxes totalBooks
  initialBooks totalDonated booksNow = 99 :=
by 
  simp [initialBooks, totalBooksDonated, boxes, booksInRoom, booksOnTable, cookbooks, booksGrabbed, booksNow]
  sorry

end initial_books_l1013_101317


namespace gf_three_l1013_101335

def f (x : ℕ) : ℕ := x^3 - 4 * x + 5
def g (x : ℕ) : ℕ := 3 * x^2 + x + 2

theorem gf_three : g (f 3) = 1222 :=
by {
  -- We would need to prove the given mathematical statement here.
  sorry
}

end gf_three_l1013_101335


namespace calculate_distribution_l1013_101371

theorem calculate_distribution (a b : ℝ) : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b :=
by
  sorry

end calculate_distribution_l1013_101371


namespace m_value_for_arithmetic_seq_min_value_t_smallest_T_periodic_seq_l1013_101390

-- Defining the sequence condition
def seq_condition (a : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n ≥ 2, a n ^ 2 - a (n + 1) * a (n - 1) = m * (a 2 - a 1) ^ 2

-- (1) Value of m for an arithmetic sequence with a non-zero common difference
theorem m_value_for_arithmetic_seq {a : ℕ → ℝ} (d : ℝ) (h_nonzero : d ≠ 0) :
  (∀ n, a (n + 1) = a n + d) → seq_condition a 1 :=
by
  sorry

-- (2) Minimum value of t given specific conditions
theorem min_value_t {t p : ℝ} (a : ℕ → ℝ) (h_p : 3 ≤ p ∧ p ≤ 5) :
  a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 4 ∧ (∀ n, t * a n + p ≥ n) → t = 1 / 32 :=
by
  sorry

-- (3) Smallest value of T for non-constant periodic sequence
theorem smallest_T_periodic_seq {a : ℕ → ℝ} {m : ℝ} (h_m_nonzero : m ≠ 0) :
  seq_condition a m → (∀ n, a (n + T) = a n) → (∃ T' > 0, ∀ T'', T'' > 0 → T'' = 3) :=
by
  sorry

end m_value_for_arithmetic_seq_min_value_t_smallest_T_periodic_seq_l1013_101390


namespace greatest_possible_value_of_x_l1013_101311

theorem greatest_possible_value_of_x : 
  (∀ x : ℚ, ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20) → 
  x ≤ 9/5 := sorry

end greatest_possible_value_of_x_l1013_101311


namespace melanies_mother_gave_l1013_101309

-- Define initial dimes, dad's contribution, and total dimes now
def initial_dimes : ℕ := 7
def dad_dimes : ℕ := 8
def total_dimes : ℕ := 19

-- Define the number of dimes the mother gave
def mother_dimes := total_dimes - (initial_dimes + dad_dimes)

-- Proof statement
theorem melanies_mother_gave : mother_dimes = 4 := by
  sorry

end melanies_mother_gave_l1013_101309


namespace problem1_problem2_problem2_equality_l1013_101305

variable {a b c d : ℝ}

-- Problem 1
theorem problem1 (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : (a - b) * (b - c) * (c - d) * (d - a) = -3) (h5 : a + b + c + d = 6) : d < 0.36 :=
sorry

-- Problem 2
theorem problem2 (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : (a - b) * (b - c) * (c - d) * (d - a) = -3) (h5 : a^2 + b^2 + c^2 + d^2 = 14) : (a + c) * (b + d) ≤ 8 :=
sorry

theorem problem2_equality (h1 : a = 3) (h2 : b = 2) (h3 : c = 1) (h4 : d = 0) : (a + c) * (b + d) = 8 :=
sorry

end problem1_problem2_problem2_equality_l1013_101305


namespace find_a_minus_b_l1013_101307

theorem find_a_minus_b (a b : ℝ) :
  (∀ (x : ℝ), x^4 - 8 * x^3 + a * x^2 + b * x + 16 = 0 → x > 0) →
  a - b = 56 :=
by
  sorry

end find_a_minus_b_l1013_101307


namespace max_k_C_l1013_101381

theorem max_k_C (n : ℕ) (h1 : Odd n) (h2 : 0 < n) :
  ∃ k : ℕ, (k = ((n + 1) / 2) ^ 2) := 
sorry

end max_k_C_l1013_101381


namespace expected_area_convex_hull_correct_l1013_101374

def point_placement (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 10

def convex_hull_area (points : Finset (ℕ × ℤ)) : ℚ := 
  -- Definition of the area calculation goes here. This is a placeholder.
  0  -- Placeholder for the actual calculation

noncomputable def expected_convex_hull_area : ℚ := 
  -- Calculation of the expected area, which is complex and requires integration of the probability.
  sorry  -- Placeholder for the actual expected value

theorem expected_area_convex_hull_correct : 
  expected_convex_hull_area = 1793 / 128 :=
sorry

end expected_area_convex_hull_correct_l1013_101374


namespace largest_prime_factor_4851_l1013_101334

theorem largest_prime_factor_4851 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 4851 ∧ (∀ q : ℕ, Nat.Prime q ∧ q ∣ 4851 → q ≤ p) :=
by
  -- todo: provide actual proof
  sorry

end largest_prime_factor_4851_l1013_101334


namespace multiple_of_students_in_restroom_l1013_101303

theorem multiple_of_students_in_restroom 
    (num_desks_per_row : ℕ)
    (num_rows : ℕ)
    (desk_fill_fraction : ℚ)
    (total_students : ℕ)
    (students_restroom : ℕ)
    (absent_students : ℕ)
    (m : ℕ) :
    num_desks_per_row = 6 →
    num_rows = 4 →
    desk_fill_fraction = 2 / 3 →
    total_students = 23 →
    students_restroom = 2 →
    (num_rows * num_desks_per_row : ℕ) * desk_fill_fraction = 16 →
    (16 - students_restroom) = 14 →
    total_students - 14 - 2 = absent_students →
    absent_students = 7 →
    2 * m - 1 = 7 →
    m = 4
:= by
    intros;
    sorry

end multiple_of_students_in_restroom_l1013_101303


namespace upper_limit_of_people_l1013_101350

theorem upper_limit_of_people (T : ℕ) (h1 : (3/7) * T = 24) (h2 : T > 50) : T ≤ 56 :=
by
  -- The steps to solve this proof would go here.
  sorry

end upper_limit_of_people_l1013_101350


namespace range_of_a_l1013_101388

-- Define conditions
def setA : Set ℝ := {x | x^2 - x ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | 2^(1 - x) + a ≤ 0}

-- Problem statement in Lean 4
theorem range_of_a (a : ℝ) (h : setA ⊆ setB a) : a ≤ -2 :=
by
  sorry

end range_of_a_l1013_101388


namespace stratified_sampling_l1013_101348

theorem stratified_sampling 
  (students_first_grade : ℕ)
  (students_second_grade : ℕ)
  (selected_first_grade : ℕ)
  (x : ℕ)
  (h1 : students_first_grade = 400)
  (h2 : students_second_grade = 360)
  (h3 : selected_first_grade = 60)
  (h4 : (selected_first_grade / students_first_grade : ℚ) = (x / students_second_grade : ℚ)) :
  x = 54 :=
sorry

end stratified_sampling_l1013_101348


namespace problem_statement_l1013_101326

noncomputable def f (n : ℕ) : ℝ := Real.log (n^2) / Real.log 3003

theorem problem_statement : f 33 + f 13 + f 7 = 2 := 
by
  sorry

end problem_statement_l1013_101326


namespace relationship_C1_C2_A_l1013_101325

variables (A B C C1 C2 : ℝ)

-- Given conditions
def TriangleABC : Prop := B = 2 * A
def AngleSumProperty : Prop := A + B + C = 180
def AltitudeDivides := C1 = 90 - A ∧ C2 = 90 - 2 * A

-- Theorem to prove the relationship between C1, C2, and A
theorem relationship_C1_C2_A (h1: TriangleABC A B) (h2: AngleSumProperty A B C) (h3: AltitudeDivides C1 C2 A) : 
  C1 - C2 = A :=
by sorry

end relationship_C1_C2_A_l1013_101325


namespace dog_food_weight_l1013_101397

/-- 
 Mike has 2 dogs, each dog eats 6 cups of dog food twice a day.
 Mike buys 9 bags of 20-pound dog food a month.
 Prove that a cup of dog food weighs 0.25 pounds.
-/
theorem dog_food_weight :
  let dogs := 2
  let cups_per_meal := 6
  let meals_per_day := 2
  let bags_per_month := 9
  let weight_per_bag := 20
  let days_per_month := 30
  let total_cups_per_day := cups_per_meal * meals_per_day * dogs
  let total_cups_per_month := total_cups_per_day * days_per_month
  let total_weight_per_month := bags_per_month * weight_per_bag
  (total_weight_per_month / total_cups_per_month : ℝ) = 0.25 :=
by
  sorry

end dog_food_weight_l1013_101397


namespace operation_correct_l1013_101338

def operation (x y : ℝ) := x^2 + y^2 + 12

theorem operation_correct :
  operation (Real.sqrt 6) (Real.sqrt 6) = 23.999999999999996 :=
by
  -- proof omitted
  sorry

end operation_correct_l1013_101338


namespace polygon_divided_l1013_101351

theorem polygon_divided (p q r : ℕ) : p - q + r = 1 :=
sorry

end polygon_divided_l1013_101351


namespace find_matches_in_second_set_l1013_101398

-- Conditions defined as Lean variables
variables (x : ℕ)
variables (avg_first_20 : ℚ := 40)
variables (avg_second_x : ℚ := 20)
variables (avg_all_30 : ℚ := 100 / 3)
variables (total_first_20 : ℚ := 20 * avg_first_20)
variables (total_all_30 : ℚ := 30 * avg_all_30)

-- Proof statement (question) along with conditions
theorem find_matches_in_second_set (x_value : x = 10) :
  avg_first_20 = 40 ∧ avg_second_x = 20 ∧ avg_all_30 = 100 / 3 →
  20 * avg_first_20 + x * avg_second_x = 30 * avg_all_30 → x = 10 := 
sorry

end find_matches_in_second_set_l1013_101398


namespace age_ratio_problem_l1013_101312

def age_condition (s a : ℕ) : Prop :=
  s - 2 = 2 * (a - 2) ∧ s - 4 = 3 * (a - 4)

def future_ratio (s a x : ℕ) : Prop :=
  (s + x) * 2 = (a + x) * 3

theorem age_ratio_problem :
  ∃ s a x : ℕ, age_condition s a ∧ future_ratio s a x ∧ x = 2 :=
by
  sorry

end age_ratio_problem_l1013_101312


namespace inequality_four_a_cubed_sub_l1013_101336

theorem inequality_four_a_cubed_sub (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  4 * a^3 * (a - b) ≥ a^4 - b^4 :=
sorry

end inequality_four_a_cubed_sub_l1013_101336


namespace cos_150_deg_eq_neg_half_l1013_101324

noncomputable def cos_of_angle (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

theorem cos_150_deg_eq_neg_half :
  cos_of_angle 150 = -1/2 :=
by
  /-
    The conditions used directly in the problem include:
    - θ = 150 (Given angle)
  -/
  sorry

end cos_150_deg_eq_neg_half_l1013_101324


namespace parallelogram_height_l1013_101360

theorem parallelogram_height
  (A b : ℝ)
  (h : ℝ)
  (h_area : A = 120)
  (h_base : b = 12)
  (h_formula : A = b * h) : h = 10 :=
by 
  sorry

end parallelogram_height_l1013_101360


namespace Maxwell_age_l1013_101365

theorem Maxwell_age :
  ∀ (sister_age maxwell_age : ℕ),
    (sister_age = 2) → 
    (maxwell_age + 2 = 2 * (sister_age + 2)) →
    (maxwell_age = 6) :=
by
  intros sister_age maxwell_age h1 h2
  -- Definitions and hypotheses come directly from conditions
  sorry

end Maxwell_age_l1013_101365


namespace rationalize_sqrt_fraction_l1013_101352

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l1013_101352


namespace cyclist_C_speed_l1013_101318

variable (c d : ℕ)

def distance_to_meeting (c d : ℕ) : Prop :=
  d = c + 6 ∧
  90 + 30 = 120 ∧
  ((90 - 30) / c) = (120 / d) ∧
  (60 / c) = (120 / (c + 6))

theorem cyclist_C_speed : distance_to_meeting c d → c = 6 :=
by
  intro h
  -- To be filled in with the proof using the conditions
  sorry

end cyclist_C_speed_l1013_101318


namespace WallLengthBy40Men_l1013_101373

-- Definitions based on the problem conditions
def men1 : ℕ := 20
def length1 : ℕ := 112
def days1 : ℕ := 6

def men2 : ℕ := 40
variable (y : ℕ)  -- given 'y' days

-- Establish the relationship based on the given conditions
theorem WallLengthBy40Men :
  ∃ x : ℕ, x = (men2 / men1) * length1 * (y / days1) :=
by
  sorry

end WallLengthBy40Men_l1013_101373


namespace range_m_condition_l1013_101344

theorem range_m_condition {x y m : ℝ} (h1 : x^2 + (y - 1)^2 = 1) (h2 : x + y + m ≥ 0) : -1 < m :=
by
  sorry

end range_m_condition_l1013_101344


namespace num_perfect_square_factors_l1013_101343

-- Define the exponents and their corresponding number of perfect square factors
def num_square_factors (exp : ℕ) : ℕ := exp / 2 + 1

-- Define the product of the prime factorization
def product : ℕ := 2^12 * 3^15 * 7^18

-- State the theorem
theorem num_perfect_square_factors :
  (num_square_factors 12) * (num_square_factors 15) * (num_square_factors 18) = 560 := by
  sorry

end num_perfect_square_factors_l1013_101343


namespace stools_chopped_up_l1013_101395

variable (chairs tables stools : ℕ)
variable (sticks_per_chair sticks_per_table sticks_per_stool : ℕ)
variable (sticks_per_hour hours total_sticks_from_chairs tables_sticks required_sticks : ℕ)

theorem stools_chopped_up (h1 : sticks_per_chair = 6)
                         (h2 : sticks_per_table = 9)
                         (h3 : sticks_per_stool = 2)
                         (h4 : sticks_per_hour = 5)
                         (h5 : chairs = 18)
                         (h6 : tables = 6)
                         (h7 : hours = 34)
                         (h8 : total_sticks_from_chairs = chairs * sticks_per_chair)
                         (h9 : tables_sticks = tables * sticks_per_table)
                         (h10 : required_sticks = hours * sticks_per_hour)
                         (h11 : total_sticks_from_chairs + tables_sticks = 162) :
                         stools = 4 := by
  sorry

end stools_chopped_up_l1013_101395


namespace negation_proposition_l1013_101384

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) :=
sorry

end negation_proposition_l1013_101384


namespace james_parking_tickets_l1013_101315

-- Define the conditions
def ticket_cost_1 := 150
def ticket_cost_2 := 150
def ticket_cost_3 := 1 / 3 * ticket_cost_1
def total_cost := ticket_cost_1 + ticket_cost_2 + ticket_cost_3
def roommate_pays := total_cost / 2
def james_remaining_money := 325
def james_original_money := james_remaining_money + roommate_pays

-- Define the theorem we want to prove
theorem james_parking_tickets (h1: ticket_cost_1 = 150)
                              (h2: ticket_cost_1 = ticket_cost_2)
                              (h3: ticket_cost_3 = 1 / 3 * ticket_cost_1)
                              (h4: total_cost = ticket_cost_1 + ticket_cost_2 + ticket_cost_3)
                              (h5: roommate_pays = total_cost / 2)
                              (h6: james_remaining_money = 325)
                              (h7: james_original_money = james_remaining_money + roommate_pays):
                              total_cost = 350 :=
by
  sorry

end james_parking_tickets_l1013_101315


namespace Carla_total_marbles_l1013_101322

def initial_marbles : ℝ := 187.0
def bought_marbles : ℝ := 134.0

theorem Carla_total_marbles : initial_marbles + bought_marbles = 321.0 := 
by 
  sorry

end Carla_total_marbles_l1013_101322


namespace systematic_sampling_interval_people_l1013_101392

theorem systematic_sampling_interval_people (total_employees : ℕ) (selected_employees : ℕ) (start_interval : ℕ) (end_interval : ℕ)
  (h_total : total_employees = 420)
  (h_selected : selected_employees = 21)
  (h_start_end : start_interval = 281)
  (h_end : end_interval = 420)
  : (end_interval - start_interval + 1) / (total_employees / selected_employees) = 7 := 
by
  -- sorry placeholder for proof
  sorry

end systematic_sampling_interval_people_l1013_101392


namespace cubic_roots_fraction_l1013_101386

theorem cubic_roots_fraction 
  (a b c d : ℝ)
  (h_eq : ∀ x: ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ (x = -2 ∨ x = 3 ∨ x = 4)) :
  c / d = -1 / 12 :=
by
  sorry

end cubic_roots_fraction_l1013_101386


namespace sum_of_star_tips_l1013_101364

/-- Given ten points that are evenly spaced on a circle and connected to form a 10-pointed star,
prove that the sum of the angle measurements of the ten tips of the star is 720 degrees. -/
theorem sum_of_star_tips (n : ℕ) (h : n = 10) :
  (10 * 72 = 720) :=
by
  sorry

end sum_of_star_tips_l1013_101364


namespace avg_age_new_students_l1013_101340

-- Definitions for the conditions
def initial_avg_age : ℕ := 14
def initial_student_count : ℕ := 10
def new_student_count : ℕ := 5
def new_avg_age : ℕ := initial_avg_age + 1

-- Lean statement for the proof problem
theorem avg_age_new_students :
  (initial_avg_age * initial_student_count + new_avg_age * new_student_count) / new_student_count = 17 :=
by
  sorry

end avg_age_new_students_l1013_101340


namespace mixture_volume_correct_l1013_101368

-- Define the input values
def water_volume : ℕ := 20
def vinegar_volume : ℕ := 18
def water_ratio : ℚ := 3/5
def vinegar_ratio : ℚ := 5/6

-- Calculate the mixture volume
def mixture_volume : ℚ :=
  (water_volume * water_ratio) + (vinegar_volume * vinegar_ratio)

-- Define the expected result
def expected_mixture_volume : ℚ := 27

-- State the theorem
theorem mixture_volume_correct : mixture_volume = expected_mixture_volume := by
  sorry

end mixture_volume_correct_l1013_101368


namespace find_m_l1013_101314

def l1 (m x y: ℝ) : Prop := 2 * x + m * y - 2 = 0
def l2 (m x y: ℝ) : Prop := m * x + 2 * y - 1 = 0
def perpendicular (m : ℝ) : Prop :=
  let slope_l1 := -2 / m
  let slope_l2 := -m / 2
  slope_l1 * slope_l2 = -1

theorem find_m (m : ℝ) (h : perpendicular m) : m = 2 :=
sorry

end find_m_l1013_101314


namespace sphere_volume_l1013_101321

theorem sphere_volume {r : ℝ} (h: 4 * Real.pi * r^2 = 256 * Real.pi) : (4 / 3) * Real.pi * r^3 = (2048 / 3) * Real.pi :=
by
  sorry

end sphere_volume_l1013_101321


namespace hyperbola_equation_l1013_101346

theorem hyperbola_equation (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) 
  (h₃ : (2^2 / a^2) - (1^2 / b^2) = 1) (h₄ : a^2 + b^2 = 3) :
  (∀ x y : ℝ,  (x^2 / 2) - y^2 = 1) :=
by 
  sorry

end hyperbola_equation_l1013_101346


namespace volume_of_box_l1013_101389

-- Define the dimensions of the box
variables (L W H : ℝ)

-- Define the conditions as hypotheses
def side_face_area : Prop := H * W = 288
def top_face_area : Prop := L * W = 1.5 * 288
def front_face_area : Prop := L * H = 0.5 * (L * W)

-- Define the volume of the box
def box_volume : ℝ := L * W * H

-- The proof statement
theorem volume_of_box (h1 : side_face_area H W) (h2 : top_face_area L W) (h3 : front_face_area L H W) : box_volume L W H = 5184 :=
by
  sorry

end volume_of_box_l1013_101389


namespace solve_eq1_solve_eq2_solve_eq3_l1013_101358

theorem solve_eq1 (x : ℝ) : 5 * x - 2.9 = 12 → x = 1.82 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

theorem solve_eq2 (x : ℝ) : 10.5 * x + 0.6 * x = 44 → x = 3 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

theorem solve_eq3 (x : ℝ) : 8 * x / 2 = 1.5 → x = 0.375 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

end solve_eq1_solve_eq2_solve_eq3_l1013_101358


namespace quadratic_one_pos_one_neg_l1013_101301

theorem quadratic_one_pos_one_neg (a : ℝ) : 
  (a < -1) → (∃ x1 x2 : ℝ, x1 * x2 < 0 ∧ x1 + x2 > 0 ∧ (x1^2 + x1 + a = 0 ∧ x2^2 + x2 + a = 0)) :=
sorry

end quadratic_one_pos_one_neg_l1013_101301


namespace reflect_across_x_axis_l1013_101367

-- Definitions for the problem conditions
def initial_point : ℝ × ℝ := (-2, 1)
def reflected_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- The statement to be proved
theorem reflect_across_x_axis :
  reflected_point initial_point = (-2, -1) :=
  sorry

end reflect_across_x_axis_l1013_101367


namespace intersection_S_T_l1013_101331

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l1013_101331


namespace power_greater_than_one_million_l1013_101370

theorem power_greater_than_one_million (α β γ δ : ℝ) (ε ζ η : ℕ)
  (h1 : α = 1.01) (h2 : β = 1.001) (h3 : γ = 1.000001) 
  (h4 : δ = 1000000) 
  (h_eps : ε = 99999900) (h_zet : ζ = 999999000) (h_eta : η = 999999000000) :
  α^ε > δ ∧ β^ζ > δ ∧ γ^η > δ :=
by
  sorry

end power_greater_than_one_million_l1013_101370


namespace find_b_l1013_101354

def operation (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem find_b (b : ℤ) (h : operation 11 b = 110) : b = 12 := 
by
  sorry

end find_b_l1013_101354


namespace identity_function_l1013_101342

theorem identity_function (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) : ∀ n : ℕ, f n = n :=
by
  sorry

end identity_function_l1013_101342


namespace number_of_senior_citizen_tickets_sold_on_first_day_l1013_101362

theorem number_of_senior_citizen_tickets_sold_on_first_day 
  (S : ℤ) (x : ℤ)
  (student_ticket_price : ℤ := 9)
  (first_day_sales : ℤ := 79)
  (second_day_sales : ℤ := 246) 
  (first_day_student_tickets_sold : ℤ := 3)
  (second_day_senior_tickets_sold : ℤ := 12)
  (second_day_student_tickets_sold : ℤ := 10) 
  (h1 : 12 * S + 10 * student_ticket_price = second_day_sales)
  (h2 : S * x + first_day_student_tickets_sold * student_ticket_price = first_day_sales) : 
  x = 4 :=
by
  sorry

end number_of_senior_citizen_tickets_sold_on_first_day_l1013_101362


namespace perimeter_of_octagon_l1013_101327

theorem perimeter_of_octagon :
  let base := 10
  let left_side := 9
  let right_side := 11
  let top_left_diagonal := 6
  let top_right_diagonal := 7
  let small_side1 := 2
  let small_side2 := 3
  let small_side3 := 4
  base + left_side + right_side + top_left_diagonal + top_right_diagonal + small_side1 + small_side2 + small_side3 = 52 :=
by
  -- This automatically assumes all the definitions and shows the equation
  sorry

end perimeter_of_octagon_l1013_101327


namespace cost_price_is_975_l1013_101337

-- Definitions from the conditions
def selling_price : ℝ := 1170
def profit_percentage : ℝ := 0.20

-- The proof statement
theorem cost_price_is_975 : (selling_price / (1 + profit_percentage)) = 975 := by
  sorry

end cost_price_is_975_l1013_101337


namespace find_x_l1013_101385

theorem find_x (x : ℝ) (h : 15 * x + 16 * x + 19 * x + 11 = 161) : x = 3 :=
sorry

end find_x_l1013_101385


namespace min_cans_for_gallon_l1013_101302

-- Define conditions
def can_capacity : ℕ := 12
def gallon_to_ounces : ℕ := 128

-- Define the minimum number of cans function.
def min_cans (capacity : ℕ) (required : ℕ) : ℕ :=
  (required + capacity - 1) / capacity -- This is the ceiling of required / capacity

-- Statement asserting the required minimum number of cans.
theorem min_cans_for_gallon (h : min_cans can_capacity gallon_to_ounces = 11) : 
  can_capacity > 0 ∧ gallon_to_ounces > 0 := by
  sorry

end min_cans_for_gallon_l1013_101302


namespace total_distance_traveled_l1013_101383

theorem total_distance_traveled
  (bike_time_min : ℕ) (bike_rate_mph : ℕ)
  (jog_time_min : ℕ) (jog_rate_mph : ℕ)
  (total_time_min : ℕ)
  (h_bike_time : bike_time_min = 30)
  (h_bike_rate : bike_rate_mph = 6)
  (h_jog_time : jog_time_min = 45)
  (h_jog_rate : jog_rate_mph = 8)
  (h_total_time : total_time_min = 75) :
  (bike_rate_mph * bike_time_min / 60) + (jog_rate_mph * jog_time_min / 60) = 9 :=
by sorry

end total_distance_traveled_l1013_101383


namespace identically_zero_on_interval_l1013_101308

variable (f : ℝ → ℝ) (a b : ℝ)
variable (h_cont : ContinuousOn f (Set.Icc a b))
variable (h_int : ∀ n : ℕ, ∫ x in a..b, (x : ℝ)^n * f x = 0)

theorem identically_zero_on_interval : ∀ x ∈ Set.Icc a b, f x = 0 := 
by 
  sorry

end identically_zero_on_interval_l1013_101308


namespace det_scaled_matrix_l1013_101316

theorem det_scaled_matrix (a b c d : ℝ) (h : a * d - b * c = 5) : 
  (3 * a) * (3 * d) - (3 * b) * (3 * c) = 45 :=
by 
  sorry

end det_scaled_matrix_l1013_101316


namespace range_a_l1013_101393

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2 * x else x - 1

theorem range_a (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 ≠ a^2 - 2 * a ∧ f x2 ≠ a^2 - 2 * a ∧ f x3 ≠ a^2 - 2 * a) ↔ (0 < a ∧ a < 1 ∨ 1 < a ∧ a < 2) :=
by
  sorry

end range_a_l1013_101393


namespace value_standard_deviations_from_mean_l1013_101380

-- Define the mean (µ)
def μ : ℝ := 15.5

-- Define the standard deviation (σ)
def σ : ℝ := 1.5

-- Define the value X
def X : ℝ := 12.5

-- Prove that the Z-score is -2
theorem value_standard_deviations_from_mean : (X - μ) / σ = -2 := by
  sorry

end value_standard_deviations_from_mean_l1013_101380


namespace parabola_slope_l1013_101310

theorem parabola_slope (p k : ℝ) (h1 : p > 0)
  (h_focus_distance : (p / 2) * (3^(1/2)) / (3 + 1^(1/2))^(1/2) = 3^(1/2))
  (h_AF_FB : exists A B : ℝ × ℝ, (A.1 = 2 - p / 2 ∧ 2 * (B.1 - 2) = 2)
    ∧ (A.2 = p - p / 2 ∧ A.2 = -2 * B.2)) :
  abs k = 2 * (2^(1/2)) :=
sorry

end parabola_slope_l1013_101310


namespace bn_is_arithmetic_an_general_formula_l1013_101333

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l1013_101333


namespace complex_expression_evaluation_l1013_101320

noncomputable def imaginary_i := Complex.I 

theorem complex_expression_evaluation : 
  ((2 + imaginary_i) / (1 - imaginary_i)) - (1 - imaginary_i) = -1/2 + (5/2) * imaginary_i :=
by 
  sorry

end complex_expression_evaluation_l1013_101320


namespace greater_number_is_84_l1013_101347

theorem greater_number_is_84
  (x y : ℕ)
  (h1 : x * y = 2688)
  (h2 : x + y - (x - y) = 64) :
  x = 84 :=
by sorry

end greater_number_is_84_l1013_101347


namespace min_cylinder_surface_area_l1013_101339

noncomputable def h := Real.sqrt (5^2 - 4^2)
noncomputable def V_cone := (1 / 3) * Real.pi * 4^2 * h
noncomputable def V_cylinder (r h': ℝ) := Real.pi * r^2 * h'
noncomputable def h' (r: ℝ) := 16 / r^2
noncomputable def S (r: ℝ) := 2 * Real.pi * r^2 + (32 * Real.pi) / r

theorem min_cylinder_surface_area : 
  ∃ r, r = 2 ∧ ∀ r', r' ≠ 2 → S r' > S 2 := sorry

end min_cylinder_surface_area_l1013_101339


namespace elder_age_is_30_l1013_101313

-- Define the ages of the younger and elder persons
variables (y e : ℕ)

-- We have the following conditions:
-- Condition 1: The elder's age is 16 years more than the younger's age
def age_difference := e = y + 16

-- Condition 2: Six years ago, the elder's age was three times the younger's age
def six_years_ago := e - 6 = 3 * (y - 6)

-- We need to prove that the present age of the elder person is 30
theorem elder_age_is_30 (y e : ℕ) (h1 : age_difference y e) (h2 : six_years_ago y e) : e = 30 :=
sorry

end elder_age_is_30_l1013_101313


namespace product_divisible_by_8_probability_l1013_101399

noncomputable def probability_product_divisible_by_8 (dice_rolls : Fin 6 → Fin 8) : ℚ :=
  -- Function to calculate the probability that the product of numbers is divisible by 8
  sorry

theorem product_divisible_by_8_probability :
  ∀ (dice_rolls : Fin 6 → Fin 8),
  probability_product_divisible_by_8 dice_rolls = 177 / 256 :=
sorry

end product_divisible_by_8_probability_l1013_101399


namespace complex_number_quadrant_l1013_101306

theorem complex_number_quadrant :
  let z := (2 - (1 * Complex.I)) / (1 + (1 * Complex.I))
  (z.re > 0 ∧ z.im < 0) :=
by
  sorry

end complex_number_quadrant_l1013_101306


namespace find_x_l1013_101341

def determinant (a b c d : ℚ) : ℚ := a * d - b * c

theorem find_x (x : ℚ) (h : determinant (2 * x) (-4) x 1 = 18) : x = 3 :=
  sorry

end find_x_l1013_101341


namespace ratio_of_border_to_tile_l1013_101323

variable {s d : ℝ}

theorem ratio_of_border_to_tile (h1 : 900 = 30 * 30)
  (h2 : 0.81 = (900 * s^2) / (30 * s + 60 * d)^2) :
  d / s = 1 / 18 := by {
  sorry }

end ratio_of_border_to_tile_l1013_101323


namespace simplify_fraction_l1013_101382

theorem simplify_fraction (x : ℤ) :
  (2 * x - 3) / 4 + (3 * x + 5) / 5 - (x - 1) / 2 = (12 * x + 15) / 20 :=
by sorry

end simplify_fraction_l1013_101382


namespace deepak_current_age_l1013_101345

theorem deepak_current_age (A D : ℕ) (h1 : A / D = 5 / 7) (h2 : A + 6 = 36) : D = 42 :=
sorry

end deepak_current_age_l1013_101345


namespace total_distance_correct_l1013_101376

def jonathan_distance : ℝ := 7.5

def mercedes_distance : ℝ := 2 * jonathan_distance

def davonte_distance : ℝ := mercedes_distance + 2

def total_distance : ℝ := mercedes_distance + davonte_distance

theorem total_distance_correct : total_distance = 32 := by
  rw [total_distance, mercedes_distance, davonte_distance]
  norm_num
  sorry

end total_distance_correct_l1013_101376


namespace g_eval_at_neg2_l1013_101391

def g (x : ℝ) : ℝ := x^3 + 2*x - 4

theorem g_eval_at_neg2 : g (-2) = -16 := by
  sorry

end g_eval_at_neg2_l1013_101391


namespace average_death_rate_l1013_101330

variable (birth_rate : ℕ) (net_increase_day : ℕ)

noncomputable def death_rate_per_two_seconds (birth_rate net_increase_day : ℕ) : ℕ :=
  let seconds_per_day := 86400
  let net_increase_per_second := net_increase_day / seconds_per_day
  let birth_rate_per_second := birth_rate / 2
  let death_rate_per_second := birth_rate_per_second - net_increase_per_second
  2 * death_rate_per_second

theorem average_death_rate
  (birth_rate : ℕ := 4) 
  (net_increase_day : ℕ := 86400) :
  death_rate_per_two_seconds birth_rate net_increase_day = 2 :=
sorry

end average_death_rate_l1013_101330


namespace calculate_expression_l1013_101356

-- Define the conditions
def exp1 : ℤ := (-1)^(53)
def exp2 : ℤ := 2^(2^4 + 5^2 - 4^3)

-- State and skip the proof
theorem calculate_expression :
  exp1 + exp2 = -1 + 1 / (2^23) :=
by sorry

#check calculate_expression

end calculate_expression_l1013_101356


namespace math_proof_l1013_101375

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def binom (n k : Nat) : Nat :=
  (factorial n) / ((factorial k) * (factorial (n - k)))

theorem math_proof :
  binom 20 6 * factorial 6 = 27907200 :=
by
  sorry

end math_proof_l1013_101375


namespace multiples_of_6_or_8_but_not_both_l1013_101319

theorem multiples_of_6_or_8_but_not_both (n : ℕ) : 
  n = 25 ∧ (n = 18) ∧ (n = 6) → (25 - 6) + (18 - 6) = 31 :=
by
  sorry

end multiples_of_6_or_8_but_not_both_l1013_101319


namespace find_multiple_of_sons_age_l1013_101396

theorem find_multiple_of_sons_age (F S k : ℕ) 
  (h1 : F = 33)
  (h2 : F = k * S + 3)
  (h3 : F + 3 = 2 * (S + 3) + 10) : 
  k = 3 :=
by
  sorry

end find_multiple_of_sons_age_l1013_101396


namespace additional_money_needed_for_free_shipping_l1013_101377

-- Define the prices of the books
def price_book1 : ℝ := 13.00
def price_book2 : ℝ := 15.00
def price_book3 : ℝ := 10.00
def price_book4 : ℝ := 10.00

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Calculate the discounted prices
def discounted_price_book1 : ℝ := price_book1 * (1 - discount_rate)
def discounted_price_book2 : ℝ := price_book2 * (1 - discount_rate)

-- Sum of discounted prices of books
def total_cost : ℝ := discounted_price_book1 + discounted_price_book2 + price_book3 + price_book4

-- Free shipping threshold
def free_shipping_threshold : ℝ := 50.00

-- Define the additional amount needed for free shipping
def additional_amount : ℝ := free_shipping_threshold - total_cost

-- The proof statement
theorem additional_money_needed_for_free_shipping : additional_amount = 9.00 := by
  -- calculation steps omitted
  sorry

end additional_money_needed_for_free_shipping_l1013_101377


namespace find_b_l1013_101357

open Real

variables {A B C a b c : ℝ}

theorem find_b 
  (hA : A = π / 4) 
  (h1 : 2 * b * sin B - c * sin C = 2 * a * sin A) 
  (h_area : 1 / 2 * b * c * sin A = 3) : 
  b = 3 := 
sorry

end find_b_l1013_101357


namespace probability_of_consecutive_blocks_drawn_l1013_101349

theorem probability_of_consecutive_blocks_drawn :
  let total_ways := (Nat.factorial 12)
  let favorable_ways := (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 5) * (Nat.factorial 3)
  (favorable_ways / total_ways) = 1 / 4620 :=
by
  sorry

end probability_of_consecutive_blocks_drawn_l1013_101349


namespace find_number_l1013_101387

variable (n : ℝ)

theorem find_number (h₁ : (0.47 * 1442 - 0.36 * n) + 63 = 3) : 
  n = 2049.28 := 
by 
  sorry

end find_number_l1013_101387


namespace tan_alpha_value_trigonometric_expression_value_l1013_101378

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.sin α = (2 * Real.sqrt 5) / 5) : 
  Real.tan α = 2 :=
sorry

theorem trigonometric_expression_value (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.sin α = (2 * Real.sqrt 5) / 5) : 
  (4 * Real.sin (π - α) + 2 * Real.cos (2 * π - α)) / (Real.sin (π / 2 - α) + Real.sin (-α)) = -10 := 
sorry

end tan_alpha_value_trigonometric_expression_value_l1013_101378


namespace arithmetic_mean_of_primes_l1013_101329

variable (list : List ℕ) 
variable (primes : List ℕ)
variable (h1 : list = [24, 25, 29, 31, 33])
variable (h2 : primes = [29, 31])

theorem arithmetic_mean_of_primes : (primes.sum / primes.length : ℝ) = 30 := by
  sorry

end arithmetic_mean_of_primes_l1013_101329


namespace min_sum_of_arithmetic_sequence_terms_l1013_101353

open Real

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ d : ℝ, a m = a n + d * (m - n)

theorem min_sum_of_arithmetic_sequence_terms (a : ℕ → ℝ) 
  (hpos : ∀ n, a n > 0) 
  (harith : arithmetic_sequence a) 
  (hprod : a 1 * a 20 = 100) : 
  a 7 + a 14 ≥ 20 := sorry

end min_sum_of_arithmetic_sequence_terms_l1013_101353


namespace solve_for_constants_l1013_101328

theorem solve_for_constants : 
  ∃ (t s : ℚ), (∀ x : ℚ, (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + 12) = 15 * x^4 + s * x^3 + 33 * x^2 + 12 * x + 108) ∧ 
  t = 37 / 5 ∧ 
  s = 11 / 5 :=
by
  sorry

end solve_for_constants_l1013_101328
