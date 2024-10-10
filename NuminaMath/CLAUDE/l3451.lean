import Mathlib

namespace intersection_of_A_and_B_l3451_345108

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by sorry

end intersection_of_A_and_B_l3451_345108


namespace is_integer_division_l3451_345160

theorem is_integer_division : ∃ k : ℤ, (19^92 - 91^29) / 90 = k := by
  sorry

end is_integer_division_l3451_345160


namespace stuffed_animal_sales_difference_l3451_345182

/-- Given the sales of stuffed animals by Quincy, Thor, and Jake, prove the difference between Quincy's and Jake's sales. -/
theorem stuffed_animal_sales_difference 
  (quincy thor jake : ℕ) 
  (h1 : quincy = 100 * thor) 
  (h2 : jake = thor + 15) 
  (h3 : quincy = 2000) : 
  quincy - jake = 1965 := by
sorry

end stuffed_animal_sales_difference_l3451_345182


namespace jack_vacation_budget_l3451_345114

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Represents the amount of money Jack has saved in base 8 -/
def jack_savings : ℕ := 3777

/-- Represents the cost of the airline ticket in base 10 -/
def ticket_cost : ℕ := 1200

/-- Calculates the remaining money after buying the ticket -/
def remaining_money : ℕ := base8_to_base10 jack_savings - ticket_cost

theorem jack_vacation_budget :
  remaining_money = 847 := by sorry

end jack_vacation_budget_l3451_345114


namespace second_point_x_coordinate_l3451_345181

/-- Given two points on a line, prove the x-coordinate of the second point -/
theorem second_point_x_coordinate 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 1 = 2 * (n + 0.5) + 5) : 
  m + 1 = 2 * n + 6 := by
  sorry

end second_point_x_coordinate_l3451_345181


namespace bryans_deposit_l3451_345122

theorem bryans_deposit (mark_deposit : ℕ) (bryan_deposit : ℕ) : 
  mark_deposit = 88 →
  bryan_deposit = 5 * mark_deposit - 40 →
  bryan_deposit = 400 := by
sorry

end bryans_deposit_l3451_345122


namespace employee_reduction_l3451_345188

theorem employee_reduction (original_employees : ℝ) (reduction_percentage : ℝ) : 
  original_employees = 243.75 → 
  reduction_percentage = 0.20 → 
  original_employees * (1 - reduction_percentage) = 195 := by
  sorry


end employee_reduction_l3451_345188


namespace jose_investment_is_225000_l3451_345162

/-- Calculates Jose's investment given the problem conditions -/
def calculate_jose_investment (tom_investment : ℕ) (tom_duration : ℕ) (jose_duration : ℕ) (total_profit : ℕ) (jose_profit : ℕ) : ℕ :=
  (tom_investment * tom_duration * (total_profit - jose_profit)) / (jose_profit * jose_duration)

/-- Proves that Jose's investment is 225000 given the problem conditions -/
theorem jose_investment_is_225000 :
  calculate_jose_investment 30000 12 10 27000 15000 = 225000 := by
  sorry

end jose_investment_is_225000_l3451_345162


namespace expected_left_handed_students_l3451_345105

theorem expected_left_handed_students
  (total_students : ℕ)
  (left_handed_proportion : ℚ)
  (h1 : total_students = 32)
  (h2 : left_handed_proportion = 3 / 8) :
  ↑total_students * left_handed_proportion = 12 :=
by sorry

end expected_left_handed_students_l3451_345105


namespace intersection_of_A_and_B_l3451_345186

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end intersection_of_A_and_B_l3451_345186


namespace relationship_abc_l3451_345180

theorem relationship_abc (a b c : ℝ) 
  (ha : a = 2^(1/5))
  (hb : b = 2^(3/10))
  (hc : c = Real.log 2 / Real.log 3) :
  c < a ∧ a < b :=
sorry

end relationship_abc_l3451_345180


namespace messages_cleared_in_seven_days_l3451_345155

/-- Given the initial number of unread messages, messages read per day,
    and new messages received per day, calculate the number of days
    required to read all unread messages. -/
def days_to_read_messages (initial_messages : ℕ) (messages_read_per_day : ℕ) (new_messages_per_day : ℕ) : ℕ :=
  initial_messages / (messages_read_per_day - new_messages_per_day)

/-- Theorem stating that it takes 7 days to read all unread messages
    under the given conditions. -/
theorem messages_cleared_in_seven_days :
  days_to_read_messages 98 20 6 = 7 := by
  sorry

#eval days_to_read_messages 98 20 6

end messages_cleared_in_seven_days_l3451_345155


namespace x_fourth_minus_reciprocal_l3451_345100

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end x_fourth_minus_reciprocal_l3451_345100


namespace correct_sample_l3451_345185

def random_number_table : List (List Nat) := [
  [16, 22, 77, 94, 39, 49, 54, 43, 54, 82, 17, 37, 93, 23, 78, 87, 35, 20, 96, 43, 84, 26, 34, 91, 64],
  [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76],
  [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79],
  [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38, 15, 51, 00, 13, 42, 99, 66, 02, 79, 54],
  [57, 60, 86, 32, 44, 09, 47, 27, 96, 54, 49, 17, 46, 09, 62, 90, 52, 84, 77, 27, 08, 02, 73, 43, 28]
]

def start_row : Nat := 5
def start_col : Nat := 4
def total_bottles : Nat := 80
def sample_size : Nat := 6

def is_valid_bottle (n : Nat) : Bool :=
  n < total_bottles

def select_sample (table : List (List Nat)) (row : Nat) (col : Nat) : List Nat :=
  sorry

theorem correct_sample :
  select_sample random_number_table start_row start_col = [77, 39, 49, 54, 43, 17] :=
by sorry

end correct_sample_l3451_345185


namespace ball_max_height_l3451_345192

/-- The height function of the ball's parabolic path -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

/-- Theorem stating that the maximum height of the ball is 116 feet -/
theorem ball_max_height : 
  ∃ (max : ℝ), max = 116 ∧ ∀ (t : ℝ), h t ≤ max :=
sorry

end ball_max_height_l3451_345192


namespace graphs_intersect_once_l3451_345151

/-- The value of b for which the graphs of y = bx^2 + 5x + 2 and y = -2x - 3 intersect at exactly one point -/
def b : ℚ := 49 / 20

/-- The quadratic function representing the first graph -/
def f (x : ℝ) : ℝ := b * x^2 + 5 * x + 2

/-- The linear function representing the second graph -/
def g (x : ℝ) : ℝ := -2 * x - 3

/-- Theorem stating that the graphs intersect at exactly one point when b = 49/20 -/
theorem graphs_intersect_once :
  ∃! x : ℝ, f x = g x :=
sorry

end graphs_intersect_once_l3451_345151


namespace product_mod_75_l3451_345189

theorem product_mod_75 : ∃ m : ℕ, 198 * 864 ≡ m [ZMOD 75] ∧ 0 ≤ m ∧ m < 75 :=
by
  use 72
  sorry

end product_mod_75_l3451_345189


namespace vector_difference_magnitude_l3451_345147

def m : Fin 2 → ℝ := ![(-1), 2]
def n (b : ℝ) : Fin 2 → ℝ := ![2, b]

theorem vector_difference_magnitude (b : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ m = k • n b) →
  ‖m - n b‖ = 3 * Real.sqrt 5 := by sorry

end vector_difference_magnitude_l3451_345147


namespace fourth_number_in_expression_l3451_345159

theorem fourth_number_in_expression (x : ℝ) : 
  0.3 * 0.8 + 0.1 * x = 0.29 → x = 0.5 := by
sorry

end fourth_number_in_expression_l3451_345159


namespace complex_number_quadrant_l3451_345173

theorem complex_number_quadrant : ∀ z : ℂ, 
  (3 - 2*I) * z = 4 + 3*I → 
  (0 < z.re ∧ 0 < z.im) := by
sorry

end complex_number_quadrant_l3451_345173


namespace perfect_square_fraction_solutions_l3451_345139

theorem perfect_square_fraction_solutions :
  ∀ m n p : ℕ+,
  p.val.Prime →
  (∃ k : ℕ+, ((5^(m.val) + 2^(n.val) * p.val) : ℚ) / (5^(m.val) - 2^(n.val) * p.val) = (k.val : ℚ)^2) →
  ((m = 1 ∧ n = 1 ∧ p = 2) ∨ (m = 2 ∧ n = 3 ∧ p = 3) ∨ (m = 2 ∧ n = 2 ∧ p = 5)) :=
by sorry

end perfect_square_fraction_solutions_l3451_345139


namespace least_possible_QR_length_l3451_345118

theorem least_possible_QR_length (PQ PR SR QS : ℝ) (hPQ : PQ = 7)
  (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 24) :
  ∃ (QR : ℕ), QR ≥ 14 ∧ ∀ (n : ℕ), n ≥ 14 → 
  (QR : ℝ) > PR - PQ ∧ (QR : ℝ) > QS - SR := by
  sorry

end least_possible_QR_length_l3451_345118


namespace age_of_35th_student_l3451_345197

/-- The age of the 35th student in a class, given the following conditions:
  - There are 35 students in total
  - The average age of all 35 students is 16.5 years
  - 10 students have an average age of 15.3 years
  - 17 students have an average age of 16.7 years
  - 6 students have an average age of 18.4 years
  - 1 student has an age of 14.7 years
-/
theorem age_of_35th_student 
  (total_students : Nat) 
  (avg_age_all : ℝ)
  (num_group1 : Nat) (avg_age_group1 : ℝ)
  (num_group2 : Nat) (avg_age_group2 : ℝ)
  (num_group3 : Nat) (avg_age_group3 : ℝ)
  (num_group4 : Nat) (age_group4 : ℝ)
  (h1 : total_students = 35)
  (h2 : avg_age_all = 16.5)
  (h3 : num_group1 = 10)
  (h4 : avg_age_group1 = 15.3)
  (h5 : num_group2 = 17)
  (h6 : avg_age_group2 = 16.7)
  (h7 : num_group3 = 6)
  (h8 : avg_age_group3 = 18.4)
  (h9 : num_group4 = 1)
  (h10 : age_group4 = 14.7)
  (h11 : num_group1 + num_group2 + num_group3 + num_group4 + 1 = total_students) :
  (total_students : ℝ) * avg_age_all - 
  ((num_group1 : ℝ) * avg_age_group1 + 
   (num_group2 : ℝ) * avg_age_group2 + 
   (num_group3 : ℝ) * avg_age_group3 + 
   (num_group4 : ℝ) * age_group4) = 15.5 := by
  sorry

end age_of_35th_student_l3451_345197


namespace area_circle_outside_square_l3451_345152

/-- The area inside a circle but outside a square with shared center -/
theorem area_circle_outside_square (r : ℝ) (d : ℝ) :
  r = 1 →  -- radius of circle is 1
  d = 2 →  -- diagonal of square is 2
  π - d^2 / 2 = π - 2 :=
by
  sorry

#check area_circle_outside_square

end area_circle_outside_square_l3451_345152


namespace souvenir_price_increase_l3451_345104

theorem souvenir_price_increase (original_price final_price : ℝ) 
  (h1 : original_price = 76.8)
  (h2 : final_price = 120)
  (h3 : ∃ x : ℝ, original_price * (1 + x)^2 = final_price) :
  ∃ x : ℝ, original_price * (1 + x)^2 = final_price ∧ x = 0.25 := by
sorry

end souvenir_price_increase_l3451_345104


namespace remaining_books_and_games_l3451_345165

/-- The number of remaining items to experience in a category -/
def remaining (total : ℕ) (experienced : ℕ) : ℕ := total - experienced

/-- The total number of remaining items to experience across categories -/
def total_remaining (remaining1 : ℕ) (remaining2 : ℕ) : ℕ := remaining1 + remaining2

/-- Proof that the number of remaining books and games to experience is 109 -/
theorem remaining_books_and_games :
  let total_books : ℕ := 150
  let total_games : ℕ := 50
  let books_read : ℕ := 74
  let games_played : ℕ := 17
  let remaining_books := remaining total_books books_read
  let remaining_games := remaining total_games games_played
  total_remaining remaining_books remaining_games = 109 := by
  sorry

end remaining_books_and_games_l3451_345165


namespace part1_part2_l3451_345141

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x, f a x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1 := by sorry

-- Part 2
theorem part2 (m : ℝ) : 
  (∃ n : ℝ, f 1 n ≤ m - f 1 (-n)) → m ≥ 4 := by sorry

end part1_part2_l3451_345141


namespace sum_of_repeating_decimals_l3451_345179

-- Define repeating decimals
def repeating_decimal_8 : ℚ := 8/9
def repeating_decimal_2 : ℚ := 2/9

-- Theorem statement
theorem sum_of_repeating_decimals : 
  repeating_decimal_8 + repeating_decimal_2 = 10/9 := by
  sorry

end sum_of_repeating_decimals_l3451_345179


namespace problem_statement_l3451_345109

noncomputable def f (x : ℝ) : ℝ := (x / (x + 4)) * Real.exp (x + 2)

noncomputable def g (a x : ℝ) : ℝ := (Real.exp (x + 2) - a * x - 3 * a) / ((x + 2)^2)

theorem problem_statement :
  (∀ x > -2, x * Real.exp (x + 2) + x + 4 > 0) ∧
  (∀ a ∈ Set.Icc 0 1, ∃ min_x > -2, ∀ x > -2, g a min_x ≤ g a x) ∧
  (∃ h : ℝ → ℝ, (∀ a ∈ Set.Icc 0 1, ∃ min_x > -2, h a = g a min_x) ∧
    Set.range h = Set.Ioo (1/2) (Real.exp 2 / 4)) := by
  sorry

end problem_statement_l3451_345109


namespace sum_cube_inequality_l3451_345125

theorem sum_cube_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end sum_cube_inequality_l3451_345125


namespace square_area_increase_l3451_345126

theorem square_area_increase (s : ℝ) (h1 : s^2 = 256) (h2 : s > 0) :
  (s + 2)^2 - s^2 = 68 := by
  sorry

end square_area_increase_l3451_345126


namespace manager_count_is_two_l3451_345199

/-- Represents the daily salary structure and employee count in a grocery store -/
structure GroceryStore where
  managerSalary : ℕ
  clerkSalary : ℕ
  clerkCount : ℕ
  totalSalary : ℕ

/-- Calculates the number of managers in the grocery store -/
def managerCount (store : GroceryStore) : ℕ :=
  (store.totalSalary - store.clerkSalary * store.clerkCount) / store.managerSalary

/-- Theorem stating that the number of managers in the given scenario is 2 -/
theorem manager_count_is_two :
  let store : GroceryStore := {
    managerSalary := 5,
    clerkSalary := 2,
    clerkCount := 3,
    totalSalary := 16
  }
  managerCount store = 2 := by sorry

end manager_count_is_two_l3451_345199


namespace modulo_residue_problem_l3451_345194

theorem modulo_residue_problem : (392 + 6 * 51 + 8 * 221 + 3^2 * 23) % 17 = 11 := by
  sorry

end modulo_residue_problem_l3451_345194


namespace equation_satisfies_condition_l3451_345136

theorem equation_satisfies_condition (x y z : ℤ) : 
  x = z - 2 ∧ y = x + 1 → x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
  sorry

end equation_satisfies_condition_l3451_345136


namespace flowers_used_for_bouquets_l3451_345177

theorem flowers_used_for_bouquets (tulips roses extra_flowers : ℕ) :
  tulips = 4 → roses = 11 → extra_flowers = 4 →
  tulips + roses - extra_flowers = 11 := by
  sorry

end flowers_used_for_bouquets_l3451_345177


namespace ball_hitting_ground_time_l3451_345175

/-- The time it takes for a ball to hit the ground when thrown upward -/
theorem ball_hitting_ground_time :
  let initial_speed : ℝ := 5
  let initial_height : ℝ := 10
  let gravity : ℝ := 9.8
  let motion_equation (t : ℝ) : ℝ := -4.9 * t^2 + initial_speed * t + initial_height
  ∃ (t : ℝ), t > 0 ∧ motion_equation t = 0 ∧ t = 10/7 :=
by sorry

end ball_hitting_ground_time_l3451_345175


namespace expand_expression_l3451_345120

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 := by
  sorry

end expand_expression_l3451_345120


namespace min_value_of_x_l3451_345154

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ Real.log 3 + (2/3) * Real.log x) : x ≥ 27 := by
  sorry

end min_value_of_x_l3451_345154


namespace f_47_mod_17_l3451_345156

def f (n : ℕ) : ℕ := 3^n + 7^n

theorem f_47_mod_17 : f 47 % 17 = 10 := by
  sorry

end f_47_mod_17_l3451_345156


namespace intersection_sum_l3451_345172

/-- Given two lines y = 2x + c and y = -x + d intersecting at (4, 12), prove that c + d = 20 -/
theorem intersection_sum (c d : ℝ) : 
  (∀ x y, y = 2*x + c → y = -x + d → (x = 4 ∧ y = 12)) → 
  c + d = 20 := by sorry

end intersection_sum_l3451_345172


namespace competition_results_l3451_345111

/-- Represents the score for a single competition -/
structure CompetitionScore where
  first : ℕ+
  second : ℕ+
  third : ℕ+
  first_gt_second : first > second
  second_gt_third : second > third

/-- Represents the results of all competitions -/
structure CompetitionResults where
  score : CompetitionScore
  num_competitions : ℕ+
  a_total_score : ℕ
  b_total_score : ℕ
  c_total_score : ℕ
  b_first_place_count : ℕ

/-- The main theorem statement -/
theorem competition_results 
  (res : CompetitionResults)
  (h1 : res.num_competitions = 6)
  (h2 : res.a_total_score = 26)
  (h3 : res.b_total_score = 11)
  (h4 : res.c_total_score = 11)
  (h5 : res.b_first_place_count = 1) :
  ∃ (b_third_place_count : ℕ), b_third_place_count = 4 := by
  sorry

end competition_results_l3451_345111


namespace vector_sum_zero_l3451_345176

variable {V : Type*} [AddCommGroup V]

theorem vector_sum_zero (A B C : V) : (B - A) + (A - C) - (B - C) = 0 := by
  sorry

end vector_sum_zero_l3451_345176


namespace people_owning_only_dogs_l3451_345163

theorem people_owning_only_dogs :
  let total_pet_owners : ℕ := 79
  let only_cats : ℕ := 10
  let cats_and_dogs : ℕ := 5
  let cats_dogs_snakes : ℕ := 3
  let total_snakes : ℕ := 49
  let only_dogs : ℕ := total_pet_owners - only_cats - cats_and_dogs - cats_dogs_snakes - (total_snakes - cats_dogs_snakes)
  only_dogs = 15 :=
by sorry

end people_owning_only_dogs_l3451_345163


namespace apple_quantity_proof_l3451_345129

/-- Calculates the final quantity of apples given initial quantity, sold quantity, and purchased quantity. -/
def final_quantity (initial : ℕ) (sold : ℕ) (purchased : ℕ) : ℕ :=
  initial - sold + purchased

/-- Theorem stating that given the specific quantities in the problem, the final quantity is 293 kg. -/
theorem apple_quantity_proof :
  final_quantity 280 132 145 = 293 := by
  sorry

end apple_quantity_proof_l3451_345129


namespace f_decreasing_on_interval_l3451_345174

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x^2 - 3 * x + 1) / Real.log (1/3)

theorem f_decreasing_on_interval : 
  ∀ x y, 1 < x → x < y → f y < f x := by sorry

end f_decreasing_on_interval_l3451_345174


namespace always_two_real_roots_root_geq_7_implies_k_leq_neg_5_l3451_345127

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : ℝ := x^2 + (k+1)*x + 3*k - 6

-- Theorem 1: The quadratic equation always has two real roots
theorem always_two_real_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ k = 0 ∧ quadratic_equation x₂ k = 0 :=
sorry

-- Theorem 2: If one root is not less than 7, then k ≤ -5
theorem root_geq_7_implies_k_leq_neg_5 (k : ℝ) :
  (∃ x : ℝ, quadratic_equation x k = 0 ∧ x ≥ 7) → k ≤ -5 :=
sorry

end always_two_real_roots_root_geq_7_implies_k_leq_neg_5_l3451_345127


namespace clownfish_display_count_l3451_345166

/-- Represents the number of fish in the aquarium -/
def total_fish : ℕ := 100

/-- Represents the number of blowfish that stay in their own tank -/
def blowfish_in_own_tank : ℕ := 26

/-- Calculates the number of clownfish in the display tank -/
def clownfish_in_display (total_fish : ℕ) (blowfish_in_own_tank : ℕ) : ℕ :=
  let total_per_species := total_fish / 2
  let blowfish_in_display := total_per_species - blowfish_in_own_tank
  let initial_clownfish_in_display := blowfish_in_display
  initial_clownfish_in_display - (initial_clownfish_in_display / 3)

theorem clownfish_display_count :
  clownfish_in_display total_fish blowfish_in_own_tank = 16 := by
  sorry

end clownfish_display_count_l3451_345166


namespace statement_a_is_correct_l3451_345130

theorem statement_a_is_correct (x y : ℝ) : x + y < 0 → x^2 - y > x := by
  sorry

end statement_a_is_correct_l3451_345130


namespace bridge_length_l3451_345145

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 156 →
  train_speed_kmh = 45 →
  crossing_time = 40 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 344 :=
by sorry

end bridge_length_l3451_345145


namespace total_cost_is_21_16_l3451_345132

def sandwich_price : ℝ := 2.49
def soda_price : ℝ := 1.87
def chips_price : ℝ := 1.25
def chocolate_price : ℝ := 0.99

def sandwich_quantity : ℕ := 2
def soda_quantity : ℕ := 4
def chips_quantity : ℕ := 3
def chocolate_quantity : ℕ := 5

def total_cost : ℝ :=
  sandwich_price * sandwich_quantity +
  soda_price * soda_quantity +
  chips_price * chips_quantity +
  chocolate_price * chocolate_quantity

theorem total_cost_is_21_16 : total_cost = 21.16 := by
  sorry

end total_cost_is_21_16_l3451_345132


namespace no_18_pretty_below_1500_l3451_345107

def is_m_pretty (n m : ℕ+) : Prop :=
  (Nat.divisors n).card = m ∧ m ∣ n

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem no_18_pretty_below_1500 :
  ∀ n : ℕ+,
  n < 1500 →
  (∃ (a b : ℕ) (k : ℕ+), 
    n = 2^a * 7^b * k ∧
    a ≥ 1 ∧
    b ≥ 1 ∧
    is_coprime k.val 14 ∧
    (Nat.divisors n.val).card = 18) →
  ¬(is_m_pretty n 18) :=
sorry

end no_18_pretty_below_1500_l3451_345107


namespace sin_20_cos_10_plus_sin_10_sin_70_l3451_345196

theorem sin_20_cos_10_plus_sin_10_sin_70 :
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) +
  Real.sin (10 * π / 180) * Real.sin (70 * π / 180) = 1 / 2 := by
  sorry

end sin_20_cos_10_plus_sin_10_sin_70_l3451_345196


namespace gcd_lcm_problem_l3451_345124

theorem gcd_lcm_problem : 
  (Nat.gcd 60 75 * Nat.lcm 48 18 + 5 = 2165) := by sorry

end gcd_lcm_problem_l3451_345124


namespace sequence_a_int_l3451_345178

def sequence_a (c : ℕ) : ℕ → ℤ
  | 0 => 2
  | n + 1 => c * sequence_a c n + Int.sqrt ((c^2 - 1) * (sequence_a c n^2 - 4))

theorem sequence_a_int (c : ℕ) (hc : c ≥ 1) :
  ∀ n : ℕ, ∃ k : ℤ, sequence_a c n = k :=
by sorry

end sequence_a_int_l3451_345178


namespace ice_cream_to_after_lunch_ratio_l3451_345123

def initial_money : ℚ := 30
def lunch_cost : ℚ := 10
def remaining_money : ℚ := 15

def money_after_lunch : ℚ := initial_money - lunch_cost
def ice_cream_cost : ℚ := money_after_lunch - remaining_money

theorem ice_cream_to_after_lunch_ratio :
  ice_cream_cost / money_after_lunch = 1 / 4 := by sorry

end ice_cream_to_after_lunch_ratio_l3451_345123


namespace problem_statement_l3451_345135

theorem problem_statement 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (heq : a^2 + 2*b^2 + 3*c^2 = 4) : 
  (a = c → a*b ≤ Real.sqrt 2 / 2) ∧ 
  (a + 2*b + 3*c ≤ 2 * Real.sqrt 6) := by
sorry

end problem_statement_l3451_345135


namespace jack_second_half_time_l3451_345115

/-- Jack and Jill's race up the hill -/
def hill_race (jack_first_half jack_total jill_total : ℕ) : Prop :=
  jack_first_half = 19 ∧
  jill_total = 32 ∧
  jack_total + 7 = jill_total

theorem jack_second_half_time 
  (jack_first_half jack_total jill_total : ℕ) 
  (h : hill_race jack_first_half jack_total jill_total) : 
  jack_total - jack_first_half = 6 := by
  sorry

end jack_second_half_time_l3451_345115


namespace unique_solution_l3451_345101

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def xyz_to_decimal (x y z : ℕ) : ℚ := (100 * x + 10 * y + z : ℚ) / 1000

theorem unique_solution (x y z : ℕ) :
  is_digit x ∧ is_digit y ∧ is_digit z →
  (1 : ℚ) / (x + y + z : ℚ) = xyz_to_decimal x y z →
  x = 1 ∧ y = 2 ∧ z = 5 := by sorry

end unique_solution_l3451_345101


namespace sister_reams_proof_l3451_345119

/-- The number of reams of paper bought for Haley -/
def reams_for_haley : ℕ := 2

/-- The total number of reams of paper bought -/
def total_reams : ℕ := 5

/-- The number of reams of paper bought for Haley's sister -/
def reams_for_sister : ℕ := total_reams - reams_for_haley

theorem sister_reams_proof : reams_for_sister = 3 := by
  sorry

end sister_reams_proof_l3451_345119


namespace smallest_two_digit_with_product_12_l3451_345190

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → n ≤ m :=
by
  sorry

end smallest_two_digit_with_product_12_l3451_345190


namespace koolaid_mixture_l3451_345138

theorem koolaid_mixture (W : ℝ) : 
  W > 4 →
  (2 : ℝ) / (2 + 4 * (W - 4)) = 0.04 →
  W = 16 := by
sorry

end koolaid_mixture_l3451_345138


namespace mikes_ride_length_l3451_345143

/-- Proves that Mike's ride was 36 miles long given the taxi fare conditions -/
theorem mikes_ride_length :
  let mike_base_fare : ℚ := 2.5
  let mike_per_mile : ℚ := 0.25
  let annie_base_fare : ℚ := 2.5
  let annie_toll : ℚ := 5
  let annie_per_mile : ℚ := 0.25
  let annie_miles : ℚ := 16
  ∀ m : ℚ,
    mike_base_fare + mike_per_mile * m = 
    annie_base_fare + annie_toll + annie_per_mile * annie_miles →
    m = 36 :=
by
  sorry


end mikes_ride_length_l3451_345143


namespace p_shape_point_count_l3451_345149

/-- Calculates the number of distinct points on a "P" shape derived from a square --/
def count_points_on_p_shape (side_length : ℕ) (point_interval : ℕ) : ℕ :=
  let points_per_side := side_length / point_interval + 1
  let total_points := points_per_side * 3
  total_points - 2

theorem p_shape_point_count :
  count_points_on_p_shape 10 1 = 31 := by
  sorry

#eval count_points_on_p_shape 10 1

end p_shape_point_count_l3451_345149


namespace interest_rate_proof_l3451_345161

theorem interest_rate_proof (P : ℝ) (n : ℕ) (diff : ℝ) (r : ℝ) : 
  P = 5399.999999999995 →
  n = 2 →
  P * ((1 + r)^n - 1) - P * r * n = diff →
  diff = 216 →
  r = 0.2 :=
sorry

end interest_rate_proof_l3451_345161


namespace perimeter_ratio_not_integer_l3451_345131

theorem perimeter_ratio_not_integer (a k l : ℕ+) (h : a ^ 2 = k * l) :
  ¬ ∃ (n : ℕ), (2 * (k + l) : ℚ) / (4 * a) = n := by
  sorry

end perimeter_ratio_not_integer_l3451_345131


namespace vectors_collinear_l3451_345167

def a : ℝ × ℝ × ℝ := (-1, 2, 8)
def b : ℝ × ℝ × ℝ := (3, 7, -1)
def c₁ : ℝ × ℝ × ℝ := (4 * a.1 - 3 * b.1, 4 * a.2.1 - 3 * b.2.1, 4 * a.2.2 - 3 * b.2.2)
def c₂ : ℝ × ℝ × ℝ := (9 * b.1 - 12 * a.1, 9 * b.2.1 - 12 * a.2.1, 9 * b.2.2 - 12 * a.2.2)

theorem vectors_collinear : ∃ (k : ℝ), c₁ = (k * c₂.1, k * c₂.2.1, k * c₂.2.2) := by
  sorry

end vectors_collinear_l3451_345167


namespace star_properties_l3451_345133

-- Define the * operation
def star (x y : ℝ) : ℝ := x - y

-- State the theorem
theorem star_properties :
  (∀ x : ℝ, star x x = 0) ∧
  (∀ x y z : ℝ, star x (star y z) = star x y + z) ∧
  (star 1993 1935 = 58) := by
  sorry

end star_properties_l3451_345133


namespace triangle_properties_l3451_345187

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem triangle_properties (t : Triangle) :
  (∃ k : ℝ, t.a = 2*k ∧ t.b = 3*k ∧ t.c = 4*k → Real.cos t.C < 0) ∧
  (Real.sin t.A > Real.sin t.B → t.A > t.B) ∧
  (t.C = π/3 ∧ t.b = 10 ∧ t.c = 9 → ∃ t1 t2 : Triangle, t1 ≠ t2 ∧ 
    t1.b = t.b ∧ t1.c = t.c ∧ t1.C = t.C ∧
    t2.b = t.b ∧ t2.c = t.c ∧ t2.C = t.C) :=
by sorry

end triangle_properties_l3451_345187


namespace total_bacon_needed_l3451_345158

/-- The number of eggs on each breakfast plate -/
def eggs_per_plate : ℕ := 2

/-- The number of customers ordering breakfast plates -/
def num_customers : ℕ := 14

/-- The number of bacon strips on each breakfast plate -/
def bacon_per_plate : ℕ := 2 * eggs_per_plate

/-- The total number of bacon strips needed -/
def total_bacon : ℕ := num_customers * bacon_per_plate

theorem total_bacon_needed : total_bacon = 56 := by
  sorry

end total_bacon_needed_l3451_345158


namespace parabola_intersection_points_l3451_345164

/-- The intersection points of two parabolas that also lie on a given line -/
theorem parabola_intersection_points (x y : ℝ) :
  (y = 3 * x^2 - 9 * x + 4) ∧ 
  (y = -x^2 + 3 * x + 6) ∧ 
  (y = x + 3) →
  ((x = (3 + Real.sqrt 11) / 2 ∧ y = (9 + Real.sqrt 11) / 2) ∨
   (x = (3 - Real.sqrt 11) / 2 ∧ y = (9 - Real.sqrt 11) / 2)) :=
by sorry

end parabola_intersection_points_l3451_345164


namespace range_of_a_l3451_345106

-- Define a decreasing function on (-1, 1)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : DecreasingFunction f) 
  (h2 : f (1 - a) < f (2 * a - 1)) :
  0 < a ∧ a < 2/3 := by
  sorry

end range_of_a_l3451_345106


namespace positive_X_value_l3451_345157

-- Define the # relation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Theorem statement
theorem positive_X_value (X : ℝ) (h : hash X 7 = 250) : X = Real.sqrt 201 :=
by sorry

end positive_X_value_l3451_345157


namespace pencils_per_student_l3451_345144

/-- Represents the distribution of pencils to students -/
def pencil_distribution (total_pencils : ℕ) (max_students : ℕ) : ℕ :=
  total_pencils / max_students

/-- Theorem stating that given 910 pencils and 91 students, each student receives 10 pencils -/
theorem pencils_per_student :
  pencil_distribution 910 91 = 10 := by
  sorry

#check pencils_per_student

end pencils_per_student_l3451_345144


namespace simplify_expression_l3451_345171

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^3 + b^3 = a + b - 1) : 
  a / b + b / a - 2 / (a * b) = -1 - 1 / (a * b) := by
  sorry

end simplify_expression_l3451_345171


namespace game_lives_theorem_l3451_345112

/-- Given a game with initial players, players who quit, and total remaining lives,
    calculate the number of lives per remaining player. -/
def lives_per_player (initial_players : ℕ) (players_quit : ℕ) (total_lives : ℕ) : ℕ :=
  total_lives / (initial_players - players_quit)

/-- Theorem: In a game with 16 initial players, 7 players quitting, and 72 total remaining lives,
    each remaining player has 8 lives. -/
theorem game_lives_theorem :
  lives_per_player 16 7 72 = 8 := by
  sorry

end game_lives_theorem_l3451_345112


namespace yarn_length_proof_l3451_345146

theorem yarn_length_proof (green_length red_length total_length : ℕ) : 
  green_length = 156 ∧ 
  red_length = 3 * green_length + 8 →
  total_length = green_length + red_length →
  total_length = 632 := by
sorry

end yarn_length_proof_l3451_345146


namespace f_inequality_l3451_345134

open Real

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem f_inequality (hf : ∀ x, x ∈ Set.Ici 0 → (x + 1) * f x + x * f' x ≥ 0)
  (hf_not_const : ¬∀ x y, f x = f y) :
  f 1 < 2 * ℯ * f 2 := by
  sorry

end f_inequality_l3451_345134


namespace sum_of_powers_and_mersenne_is_sum_of_squares_l3451_345116

/-- A Mersenne prime is a prime number of the form 2^k - 1 for some positive integer k. -/
def is_mersenne_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ k : ℕ, k > 0 ∧ p = 2^k - 1

/-- An integer n that is both the sum of two different powers of 2 
    and the sum of two different Mersenne primes is the sum of two different square numbers. -/
theorem sum_of_powers_and_mersenne_is_sum_of_squares (n : ℕ) 
  (h1 : ∃ a b : ℕ, a ≠ b ∧ n = 2^a + 2^b)
  (h2 : ∃ p q : ℕ, p ≠ q ∧ is_mersenne_prime p ∧ is_mersenne_prime q ∧ n = p + q) :
  ∃ x y : ℕ, x ≠ y ∧ n = x^2 + y^2 :=
sorry

end sum_of_powers_and_mersenne_is_sum_of_squares_l3451_345116


namespace hemisphere_base_area_l3451_345195

theorem hemisphere_base_area (r : ℝ) (h : r > 0) : 3 * Real.pi * r^2 = 9 → Real.pi * r^2 = 3 := by
  sorry

end hemisphere_base_area_l3451_345195


namespace geometric_sequence_sum_inequality_l3451_345102

/-- A geometric sequence with positive terms and common ratio not equal to 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ n, a (n + 1) = q * a n) ∧ 
  q ≠ 1

theorem geometric_sequence_sum_inequality 
  (a : ℕ → ℝ) (q : ℝ) (h : GeometricSequence a q) : 
  a 1 + a 4 > a 2 + a 3 := by
  sorry

end geometric_sequence_sum_inequality_l3451_345102


namespace fraction_division_and_addition_l3451_345168

theorem fraction_division_and_addition :
  (5 : ℚ) / 6 / (9 : ℚ) / 10 + 1 / 15 = 402 / 405 := by
  sorry

end fraction_division_and_addition_l3451_345168


namespace quadratic_root_property_l3451_345121

theorem quadratic_root_property (m : ℝ) : 
  m^2 + 2*m - 1 = 0 → 2*m^2 + 4*m = 2 := by
  sorry

end quadratic_root_property_l3451_345121


namespace ratio_sum_problem_l3451_345191

theorem ratio_sum_problem (a b c : ℕ) : 
  a + b + c = 108 → 
  5 * b = 3 * a → 
  4 * b = 3 * c → 
  b = 27 := by
sorry

end ratio_sum_problem_l3451_345191


namespace optimal_strategy_l3451_345137

/-- Represents the clothing types --/
inductive ClothingType
| A
| B

/-- Represents the cost and selling price of each clothing type --/
def clothingInfo : ClothingType → (ℕ × ℕ)
| ClothingType.A => (80, 120)
| ClothingType.B => (60, 90)

/-- The total number of clothing items --/
def totalClothing : ℕ := 100

/-- The maximum total cost allowed --/
def maxTotalCost : ℕ := 7500

/-- The minimum number of type A clothing --/
def minTypeA : ℕ := 65

/-- The maximum number of type A clothing --/
def maxTypeA : ℕ := 75

/-- Calculates the total profit given the number of type A clothing and the discount --/
def totalProfit (x : ℕ) (a : ℚ) : ℚ :=
  (10 - a) * x + 3000

/-- Represents the optimal purchase strategy --/
structure OptimalStrategy where
  typeACount : ℕ
  typeBCount : ℕ

/-- Theorem stating the optimal purchase strategy based on the discount --/
theorem optimal_strategy (a : ℚ) (h1 : 0 < a) (h2 : a < 20) :
  (∃ (strategy : OptimalStrategy),
    (0 < a ∧ a < 10 → strategy.typeACount = maxTypeA ∧ strategy.typeBCount = totalClothing - maxTypeA) ∧
    (a = 10 → strategy.typeACount ≥ minTypeA ∧ strategy.typeACount ≤ maxTypeA) ∧
    (10 < a ∧ a < 20 → strategy.typeACount = minTypeA ∧ strategy.typeBCount = totalClothing - minTypeA) ∧
    (∀ (x : ℕ), minTypeA ≤ x → x ≤ maxTypeA → totalProfit strategy.typeACount a ≥ totalProfit x a)) :=
sorry

end optimal_strategy_l3451_345137


namespace watermelon_seeds_l3451_345110

/-- Given a watermelon cut into 40 slices, with each slice having an equal number of black and white seeds,
    and a total of 1,600 seeds in the watermelon, prove that there are 20 black seeds in each slice. -/
theorem watermelon_seeds (slices : ℕ) (total_seeds : ℕ) (black_seeds_per_slice : ℕ) :
  slices = 40 →
  total_seeds = 1600 →
  total_seeds = 2 * slices * black_seeds_per_slice →
  black_seeds_per_slice = 20 := by
  sorry

#check watermelon_seeds

end watermelon_seeds_l3451_345110


namespace playground_area_l3451_345184

/-- The area of a rectangular playground with given conditions -/
theorem playground_area : 
  ∀ (width length : ℝ),
  length = 3 * width + 30 →
  2 * (width + length) = 730 →
  width * length = 23554.6875 := by
sorry

end playground_area_l3451_345184


namespace inequality_proof_l3451_345103

theorem inequality_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x^2 + y^2 + 1 = (x*y - 1)^2) : 
  (x + y ≥ 4) ∧ (x^2 + y^2 ≥ 8) ∧ (x + 4*y ≥ 9) := by
  sorry

end inequality_proof_l3451_345103


namespace H2O_weight_not_72_l3451_345148

/-- The molecular weight of H2O in g/mol -/
def molecular_weight_H2O : ℝ := 18.016

/-- The given incorrect molecular weight in g/mol -/
def given_weight : ℝ := 72

/-- Theorem stating that the molecular weight of H2O is not equal to the given weight -/
theorem H2O_weight_not_72 : molecular_weight_H2O ≠ given_weight := by
  sorry

end H2O_weight_not_72_l3451_345148


namespace chim_tu_survival_days_l3451_345117

/-- The number of distinct T-shirts --/
def n : ℕ := 4

/-- The number of days between outfit changes --/
def days_per_outfit : ℕ := 3

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The factorial of a natural number --/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of distinct outfits with exactly k T-shirts --/
def outfits_with_k (k : ℕ) : ℕ := choose n k * factorial k

/-- The total number of distinct outfits --/
def total_outfits : ℕ := outfits_with_k 3 + outfits_with_k 4

/-- The number of days Chim Tu can wear a unique outfit --/
def survival_days : ℕ := total_outfits * days_per_outfit

theorem chim_tu_survival_days : survival_days = 144 := by
  sorry

end chim_tu_survival_days_l3451_345117


namespace complex_number_proof_l3451_345170

theorem complex_number_proof (z : ℂ) :
  (∃ (z₁ : ℝ), z₁ = (z / (1 + z^2)).re ∧ (z / (1 + z^2)).im = 0) ∧
  (∃ (z₂ : ℝ), z₂ = (z^2 / (1 + z)).re ∧ (z^2 / (1 + z)).im = 0) →
  z = -1/2 + (Complex.I * Real.sqrt 3) / 2 ∨ z = -1/2 - (Complex.I * Real.sqrt 3) / 2 :=
by sorry

end complex_number_proof_l3451_345170


namespace blocks_and_colors_l3451_345198

theorem blocks_and_colors (total_blocks : ℕ) (blocks_per_color : ℕ) (colors_used : ℕ) : 
  total_blocks = 49 → blocks_per_color = 7 → colors_used = total_blocks / blocks_per_color →
  colors_used = 7 := by
  sorry

end blocks_and_colors_l3451_345198


namespace sqrt_two_plus_pi_irrational_l3451_345183

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ (p : ℝ) / (q : ℝ)

-- State the theorem
theorem sqrt_two_plus_pi_irrational :
  IsIrrational (Real.sqrt 2) → IsIrrational π → IsIrrational (Real.sqrt 2 + π) :=
by
  sorry

end sqrt_two_plus_pi_irrational_l3451_345183


namespace project_work_time_difference_l3451_345169

/-- Given three people working on a project for a total of 140 hours,
    with their working times in the ratio of 3:5:6,
    prove that the difference between the longest and shortest working times is 30 hours. -/
theorem project_work_time_difference (x : ℝ) 
  (h1 : 3 * x + 5 * x + 6 * x = 140) : 6 * x - 3 * x = 30 := by
  sorry

end project_work_time_difference_l3451_345169


namespace xyz_product_l3451_345153

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 6 * y = -24)
  (eq2 : y * z + 6 * z = -24)
  (eq3 : z * x + 6 * x = -24) :
  x * y * z = 144 := by
sorry

end xyz_product_l3451_345153


namespace maria_coin_difference_l3451_345128

/-- Represents the number of coins of each denomination -/
structure CoinCollection where
  five_cent : ℕ
  ten_cent : ℕ
  twenty_cent : ℕ
  twenty_five_cent : ℕ

/-- The conditions of Maria's coin collection -/
def maria_collection (c : CoinCollection) : Prop :=
  c.five_cent + c.ten_cent + c.twenty_cent + c.twenty_five_cent = 30 ∧
  c.ten_cent = 2 * c.five_cent ∧
  5 * c.five_cent + 10 * c.ten_cent + 20 * c.twenty_cent + 25 * c.twenty_five_cent = 410

theorem maria_coin_difference (c : CoinCollection) : 
  maria_collection c → c.twenty_five_cent - c.twenty_cent = 1 := by
  sorry

end maria_coin_difference_l3451_345128


namespace polynomial_inequality_l3451_345193

theorem polynomial_inequality (n : ℕ) (hn : n > 1) :
  ∀ x : ℝ, x > 0 → x^n - n*x + n - 1 ≥ 0 := by
  sorry

end polynomial_inequality_l3451_345193


namespace optimal_well_position_l3451_345150

open Real

/-- Represents the positions of 6 houses along a road -/
structure HousePositions where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ
  x₅ : ℝ
  x₆ : ℝ
  h₁₂ : x₁ < x₂
  h₂₃ : x₂ < x₃
  h₃₄ : x₃ < x₄
  h₄₅ : x₄ < x₅
  h₅₆ : x₅ < x₆

/-- The sum of absolute distances from a point x to all house positions -/
def sumOfDistances (hp : HousePositions) (x : ℝ) : ℝ :=
  |x - hp.x₁| + |x - hp.x₂| + |x - hp.x₃| + |x - hp.x₄| + |x - hp.x₅| + |x - hp.x₆|

/-- The theorem stating that the optimal well position is the average of x₃ and x₄ -/
theorem optimal_well_position (hp : HousePositions) :
  ∃ (x : ℝ), ∀ (y : ℝ), sumOfDistances hp x ≤ sumOfDistances hp y ∧ x = (hp.x₃ + hp.x₄) / 2 :=
sorry

end optimal_well_position_l3451_345150


namespace shirt_sweater_cost_l3451_345140

/-- The total cost of a shirt and a sweater given their price relationship -/
theorem shirt_sweater_cost (shirt_price sweater_price total_cost : ℝ) : 
  shirt_price = 36.46 →
  shirt_price = sweater_price - 7.43 →
  total_cost = shirt_price + sweater_price →
  total_cost = 80.35 := by
sorry

end shirt_sweater_cost_l3451_345140


namespace box_of_books_l3451_345113

theorem box_of_books (box_weight : ℕ) (book_weight : ℕ) (h1 : box_weight = 42) (h2 : book_weight = 3) :
  box_weight / book_weight = 14 := by
  sorry

end box_of_books_l3451_345113


namespace z_properties_l3451_345142

def z : ℂ := -(2 * Complex.I + 6) * Complex.I

theorem z_properties : 
  (z.re > 0 ∧ z.im < 0) ∧ 
  ∃ (y : ℝ), z - 2 = y * Complex.I :=
sorry

end z_properties_l3451_345142
