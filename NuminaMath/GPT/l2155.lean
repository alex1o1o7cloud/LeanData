import Mathlib

namespace candy_given_l2155_215526

theorem candy_given (A R G : ℕ) (h1 : A = 15) (h2 : R = 9) : G = 6 :=
by
  sorry

end candy_given_l2155_215526


namespace monday_to_sunday_ratio_l2155_215527

-- Define the number of pints Alice bought on Sunday
def sunday_pints : ℕ := 4

-- Define the number of pints Alice bought on Monday as a multiple of Sunday
def monday_pints (k : ℕ) : ℕ := 4 * k

-- Define the number of pints Alice bought on Tuesday
def tuesday_pints (k : ℕ) : ℚ := (4 * k) / 3

-- Define the number of pints Alice returned on Wednesday
def wednesday_return (k : ℕ) : ℚ := (2 * k) / 3

-- Define the total number of pints Alice had on Wednesday before returning the expired ones
def total_pre_return (k : ℕ) : ℚ := 18 + (2 * k) / 3

-- Define the total number of pints purchased from Sunday to Tuesday
def total_pints (k : ℕ) : ℚ := 4 + 4 * k + (4 * k) / 3

-- The statement to be proven
theorem monday_to_sunday_ratio : ∃ k : ℕ, 
  (4 * k + (4 * k) / 3 + 4 = 18 + (2 * k) / 3) ∧
  (4 * k) / 4 = 3 :=
by 
  sorry

end monday_to_sunday_ratio_l2155_215527


namespace sufficient_but_not_necessary_l2155_215538

theorem sufficient_but_not_necessary (x : ℝ) : 
  (x = 1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = 1) :=
by
  sorry

end sufficient_but_not_necessary_l2155_215538


namespace roots_in_intervals_l2155_215559

theorem roots_in_intervals {a b c : ℝ} (h₁ : a < b) (h₂ : b < c) :
  let f (x : ℝ) := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)
  -- statement that the roots are in the intervals (a, b) and (b, c)
  ∃ r₁ r₂, (a < r₁ ∧ r₁ < b) ∧ (b < r₂ ∧ r₂ < c) ∧ f r₁ = 0 ∧ f r₂ = 0 := 
sorry

end roots_in_intervals_l2155_215559


namespace ciphertext_to_plaintext_l2155_215521

theorem ciphertext_to_plaintext :
  ∃ (a b c d : ℕ), (a + 2 * b = 14) ∧ (2 * b + c = 9) ∧ (2 * c + 3 * d = 23) ∧ (4 * d = 28) ∧ a = 6 ∧ b = 4 ∧ c = 1 ∧ d = 7 :=
by 
  sorry

end ciphertext_to_plaintext_l2155_215521


namespace find_price_of_fourth_variety_theorem_l2155_215540

-- Define the variables and conditions
variables (P1 P2 P3 P4 : ℝ) (Q1 Q2 Q3 Q4 : ℝ) (P_avg : ℝ)

-- Given conditions
def price_of_fourth_variety : Prop :=
  P1 = 126 ∧
  P2 = 135 ∧
  P3 = 156 ∧
  P_avg = 165 ∧
  Q1 / Q2 = 2 / 3 ∧
  Q1 / Q3 = 2 / 4 ∧
  Q1 / Q4 = 2 / 5 ∧
  (P1 * Q1 + P2 * Q2 + P3 * Q3 + P4 * Q4) / (Q1 + Q2 + Q3 + Q4) = P_avg

-- Prove that the price of the fourth variety of tea is Rs. 205.8 per kg
theorem find_price_of_fourth_variety_theorem : price_of_fourth_variety P1 P2 P3 P4 Q1 Q2 Q3 Q4 P_avg → P4 = 205.8 :=
by {
  sorry
}

end find_price_of_fourth_variety_theorem_l2155_215540


namespace book_arrangement_ways_l2155_215597

open Nat

theorem book_arrangement_ways : 
  let m := 4  -- Number of math books
  let h := 6  -- Number of history books
  -- Number of ways to place a math book on both ends:
  let ways_ends := m * (m - 1)  -- Choices for the left end and right end
  -- Number of ways to arrange the remaining books:
  let ways_entities := 2!  -- Arrangements of the remaining entities
  -- Number of ways to arrange history books within the block:
  let arrange_history := factorial h
  -- Total arrangements
  let total_ways := ways_ends * ways_entities * arrange_history
  total_ways = 17280 := sorry

end book_arrangement_ways_l2155_215597


namespace find_lower_percentage_l2155_215501

theorem find_lower_percentage (P : ℝ) : 
  (12000 * 0.15 * 2 - 720 = 12000 * (P / 100) * 2) → P = 12 := by
  sorry

end find_lower_percentage_l2155_215501


namespace tickets_not_went_to_concert_l2155_215519

theorem tickets_not_went_to_concert :
  let total_tickets := 900
  let before_start := total_tickets * 3 / 4
  let remaining_after_start := total_tickets - before_start
  let after_first_song := remaining_after_start * 5 / 9
  let during_middle := 80
  remaining_after_start - (after_first_song + during_middle) = 20 := 
by
  let total_tickets := 900
  let before_start := total_tickets * 3 / 4
  let remaining_after_start := total_tickets - before_start
  let after_first_song := remaining_after_start * 5 / 9
  let during_middle := 80
  show remaining_after_start - (after_first_song + during_middle) = 20
  sorry

end tickets_not_went_to_concert_l2155_215519


namespace graham_crackers_leftover_l2155_215580

-- Definitions for the problem conditions
def initial_boxes_graham := 14
def initial_packets_oreos := 15
def initial_ounces_cream_cheese := 36

def boxes_per_cheesecake := 2
def packets_per_cheesecake := 3
def ounces_per_cheesecake := 4

-- Define the statement that needs to be proved
theorem graham_crackers_leftover :
  initial_boxes_graham - (min (initial_boxes_graham / boxes_per_cheesecake) (min (initial_packets_oreos / packets_per_cheesecake) (initial_ounces_cream_cheese / ounces_per_cheesecake)) * boxes_per_cheesecake) = 4 :=
by sorry

end graham_crackers_leftover_l2155_215580


namespace minimum_value_of_expression_l2155_215530

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : a > b) (h3 : ab = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := 
sorry

end minimum_value_of_expression_l2155_215530


namespace place_two_in_front_l2155_215564

-- Define the conditions: the original number has hundreds digit h, tens digit t, and units digit u.
variables (h t u : ℕ)

-- Define the function representing the placement of the digit 2 before the three-digit number.
def new_number (h t u : ℕ) : ℕ :=
  2000 + 100 * h + 10 * t + u

-- State the theorem that proves the new number formed is as stated.
theorem place_two_in_front : new_number h t u = 2000 + 100 * h + 10 * t + u :=
by sorry

end place_two_in_front_l2155_215564


namespace largest_sum_of_digits_l2155_215525

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) (h1: a < 10) (h2: b < 10) (h3: c < 10) (h4: 0 < y ∧ y ≤ 12) (h5: 1000 * y = abc) :
  a + b + c = 8 := by
  sorry

end largest_sum_of_digits_l2155_215525


namespace solve_equations_l2155_215578

theorem solve_equations (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 4 * a + 3 * b = 39) : a + b = 82 / 7 := by
  sorry

end solve_equations_l2155_215578


namespace min_x_minus_y_l2155_215571

theorem min_x_minus_y {x y : ℝ} (hx : 0 ≤ x) (hx2 : x ≤ 2 * Real.pi) (hy : 0 ≤ y) (hy2 : y ≤ 2 * Real.pi)
    (h : 2 * Real.sin x * Real.cos y - Real.sin x + Real.cos y = 1 / 2) : 
    x - y = -Real.pi / 2 := 
sorry

end min_x_minus_y_l2155_215571


namespace sequence_inequality_l2155_215596

theorem sequence_inequality 
  (a : ℕ → ℝ)
  (h_non_decreasing : ∀ i j : ℕ, i ≤ j → a i ≤ a j)
  (h_range : ∀ i, 1 ≤ i ∧ i ≤ 10 → a i = a (i - 1)) :
  (1 / 6) * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) ≤ (1 / 10) * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10) :=
by
  sorry

end sequence_inequality_l2155_215596


namespace Cody_money_final_l2155_215528

-- Define the initial amount of money Cody had
def Cody_initial : ℝ := 45.0

-- Define the birthday gift amount
def birthday_gift : ℝ := 9.0

-- Define the amount spent on the game
def game_expense : ℝ := 19.0

-- Define the percentage of remaining money spent on clothes as a fraction
def clothes_spending_fraction : ℝ := 0.40

-- Define the late birthday gift received
def late_birthday_gift : ℝ := 4.5

-- Define the final amount of money Cody has
def Cody_final : ℝ :=
  let after_birthday := Cody_initial + birthday_gift
  let after_game := after_birthday - game_expense
  let spent_on_clothes := clothes_spending_fraction * after_game
  let after_clothes := after_game - spent_on_clothes
  after_clothes + late_birthday_gift

theorem Cody_money_final : Cody_final = 25.5 := by
  sorry

end Cody_money_final_l2155_215528


namespace commuting_time_equation_l2155_215545

-- Definitions based on the conditions
def distance_to_cemetery : ℝ := 15
def cyclists_speed (x : ℝ) : ℝ := x
def car_speed (x : ℝ) : ℝ := 2 * x
def cyclists_start_time_earlier : ℝ := 0.5

-- The statement we need to prove
theorem commuting_time_equation (x : ℝ) (h : x > 0) :
  distance_to_cemetery / cyclists_speed x =
  (distance_to_cemetery / car_speed x) + cyclists_start_time_earlier :=
by
  sorry

end commuting_time_equation_l2155_215545


namespace johnson_vincent_work_together_l2155_215583

theorem johnson_vincent_work_together (work : Type) (time_johnson : ℕ) (time_vincent : ℕ) (combined_time : ℕ) :
  time_johnson = 10 → time_vincent = 40 → combined_time = 8 → 
  (1 / time_johnson + 1 / time_vincent) = 1 / combined_time :=
by
  intros h_johnson h_vincent h_combined
  sorry

end johnson_vincent_work_together_l2155_215583


namespace value_of_expression_l2155_215506

theorem value_of_expression (m : ℝ) (h : 1 / (m - 2) = 1) : (2 / (m - 2)) - m + 2 = 1 :=
sorry

end value_of_expression_l2155_215506


namespace max_three_digit_divisible_by_4_sequence_l2155_215541

theorem max_three_digit_divisible_by_4_sequence (a : ℕ → ℕ) (n : ℕ) (h1 : ∀ k ≤ n - 2, a (k + 2) = 3 * a (k + 1) - 2 * a k - 2)
(h2 : ∀ k1 k2, k1 < k2 → a k1 < a k2) (ha2022 : ∃ k, a k = 2022) (hn : n ≥ 3) :
  ∃ m : ℕ, ∀ k, 100 ≤ a k ∧ a k ≤ 999 → a k % 4 = 0 → m ≤ 225 := by
  sorry

end max_three_digit_divisible_by_4_sequence_l2155_215541


namespace problem_solution_l2155_215511

theorem problem_solution :
  2 ^ 2000 - 3 * 2 ^ 1999 + 2 ^ 1998 - 2 ^ 1997 + 2 ^ 1996 = -5 * 2 ^ 1996 :=
by  -- initiate the proof script
  sorry  -- means "proof is omitted"

end problem_solution_l2155_215511


namespace probability_different_color_and_label_sum_more_than_3_l2155_215532

-- Definitions for the conditions:
structure Coin :=
  (color : Bool) -- True for Yellow, False for Green
  (label : Nat)

def coins : List Coin := [
  Coin.mk true 1,
  Coin.mk true 2,
  Coin.mk false 1,
  Coin.mk false 2,
  Coin.mk false 3
]

def outcomes : List (Coin × Coin) :=
  [(coins[0], coins[1]), (coins[0], coins[2]), (coins[0], coins[3]), (coins[0], coins[4]),
   (coins[1], coins[2]), (coins[1], coins[3]), (coins[1], coins[4]),
   (coins[2], coins[3]), (coins[2], coins[4]), (coins[3], coins[4])]

def different_color_and_label_sum_more_than_3 (c1 c2 : Coin) : Bool :=
  c1.color ≠ c2.color ∧ (c1.label + c2.label > 3)

def valid_outcomes : List (Coin × Coin) :=
  outcomes.filter (λ p => different_color_and_label_sum_more_than_3 p.fst p.snd)

-- Proof statement:
theorem probability_different_color_and_label_sum_more_than_3 :
  (valid_outcomes.length : ℚ) / (outcomes.length : ℚ) = 3 / 10 :=
by
  sorry

end probability_different_color_and_label_sum_more_than_3_l2155_215532


namespace count_zeros_in_decimal_rep_l2155_215549

theorem count_zeros_in_decimal_rep (n : ℕ) (h : n = 2^3 * 5^7) : 
  ∀ (a b : ℕ), (∃ (a : ℕ) (b : ℕ), n = 10^b ∧ a < 10^b) → 
  6 = b - 1 := by
  sorry

end count_zeros_in_decimal_rep_l2155_215549


namespace sum_of_100_and_98_consecutive_diff_digits_l2155_215565

def S100 (n : ℕ) : ℕ := 50 * (2 * n + 99)
def S98 (n : ℕ) : ℕ := 49 * (2 * n + 297)

theorem sum_of_100_and_98_consecutive_diff_digits (n : ℕ) :
  ¬ (S100 n % 10 = S98 n % 10) :=
sorry

end sum_of_100_and_98_consecutive_diff_digits_l2155_215565


namespace max_value_of_f_l2155_215589

noncomputable def f (x : ℝ) : ℝ :=
  (2 * x + 1) / (4 * x ^ 2 + 1)

theorem max_value_of_f : ∃ (M : ℝ), ∀ (x : ℝ), x > 0 → f x ≤ M ∧ M = (Real.sqrt 2 + 1) / 2 :=
by
  sorry

end max_value_of_f_l2155_215589


namespace distinct_real_roots_form_geometric_progression_eq_170_l2155_215591

theorem distinct_real_roots_form_geometric_progression_eq_170 
  (a : ℝ) :
  (∃ (u : ℝ) (v : ℝ) (hu : u ≠ 0) (hv : v ≠ 0) (hv1 : |v| ≠ 1), 
  (16 * u^12 + (2 * a + 17) * u^6 * v^3 - a * u^9 * v - a * u^3 * v^9 + 16 = 0)) 
  → a = 170 :=
by sorry

end distinct_real_roots_form_geometric_progression_eq_170_l2155_215591


namespace a_18_value_l2155_215534

-- Define the concept of an "Equally Summed Sequence"
def equallySummedSequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = c

-- Define the specific conditions for a_1 and the common sum
def specific_sequence (a : ℕ → ℝ) : Prop :=
  equallySummedSequence a 5 ∧ a 1 = 2

-- The theorem we want to prove
theorem a_18_value (a : ℕ → ℝ) (h : specific_sequence a) : a 18 = 3 :=
sorry

end a_18_value_l2155_215534


namespace rectangle_area_288_l2155_215520

/-- A rectangle contains eight circles arranged in a 2x4 grid. Each circle has a radius of 3 inches.
    We are asked to prove that the area of the rectangle is 288 square inches. --/
noncomputable def circle_radius : ℝ := 3
noncomputable def circles_per_width : ℕ := 2
noncomputable def circles_per_length : ℕ := 4
noncomputable def circle_diameter : ℝ := 2 * circle_radius
noncomputable def rectangle_width : ℝ := circles_per_width * circle_diameter
noncomputable def rectangle_length : ℝ := circles_per_length * circle_diameter
noncomputable def rectangle_area : ℝ := rectangle_length * rectangle_width

theorem rectangle_area_288 :
  rectangle_area = 288 :=
by
  -- Proof of the area will be filled in here.
  sorry

end rectangle_area_288_l2155_215520


namespace fatima_donates_75_sq_inches_l2155_215547

/-- Fatima starts with 100 square inches of cloth and cuts it in half twice.
    The total amount of cloth she donates should be 75 square inches. -/
theorem fatima_donates_75_sq_inches:
  ∀ (cloth_initial cloth_after_first_cut cloth_after_second_cut cloth_donated_first cloth_donated_second: ℕ),
  cloth_initial = 100 → 
  cloth_after_first_cut = cloth_initial / 2 →
  cloth_donated_first = cloth_initial / 2 →
  cloth_after_second_cut = cloth_after_first_cut / 2 →
  cloth_donated_second = cloth_after_first_cut / 2 →
  cloth_donated_first + cloth_donated_second = 75 := 
by
  intros cloth_initial cloth_after_first_cut cloth_after_second_cut cloth_donated_first cloth_donated_second
  intros h_initial h_after_first h_donated_first h_after_second h_donated_second
  sorry

end fatima_donates_75_sq_inches_l2155_215547


namespace length_of_side_d_l2155_215507

variable (a b c d : ℕ)
variable (h_ratio1 : a / c = 3 / 4)
variable (h_ratio2 : b / d = 3 / 4)
variable (h_a : a = 3)
variable (h_b : b = 6)

theorem length_of_side_d (a b c d : ℕ)
  (h_ratio1 : a / c = 3 / 4)
  (h_ratio2 : b / d = 3 / 4)
  (h_a : a = 3)
  (h_b : b = 6) : d = 8 := 
sorry

end length_of_side_d_l2155_215507


namespace problems_per_page_l2155_215585

def total_problems : ℕ := 72
def finished_problems : ℕ := 32
def remaining_pages : ℕ := 5
def remaining_problems : ℕ := total_problems - finished_problems

theorem problems_per_page : remaining_problems / remaining_pages = 8 := 
by
  sorry

end problems_per_page_l2155_215585


namespace red_pencils_count_l2155_215599

theorem red_pencils_count 
  (packs : ℕ) 
  (pencils_per_pack : ℕ) 
  (extra_packs : ℕ) 
  (extra_pencils_per_pack : ℕ)
  (total_red_pencils : ℕ) 
  (h1 : packs = 15)
  (h2 : pencils_per_pack = 1)
  (h3 : extra_packs = 3)
  (h4 : extra_pencils_per_pack = 2)
  (h5 : total_red_pencils = packs * pencils_per_pack + extra_packs * extra_pencils_per_pack) : 
  total_red_pencils = 21 := 
  by sorry

end red_pencils_count_l2155_215599


namespace range_of_a_l2155_215575

noncomputable def f (x a : ℝ) : ℝ := 
  (1 / 2) * (Real.cos x + Real.sin x) * (Real.cos x - Real.sin x - 4 * a) + (4 * a - 3) * x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → 
  0 ≤ (Real.cos (2 * x) - 2 * a * (Real.sin x - Real.cos x) + 4 * a - 3)) ↔ (a ≥ 1.5) :=
sorry

end range_of_a_l2155_215575


namespace cubic_feet_per_bag_l2155_215536

-- Definitions
def length_bed := 8 -- in feet
def width_bed := 4 -- in feet
def height_bed := 1 -- in feet
def number_of_beds := 2
def number_of_bags := 16

-- Theorem statement
theorem cubic_feet_per_bag : 
  (length_bed * width_bed * height_bed * number_of_beds) / number_of_bags = 4 :=
by
  sorry

end cubic_feet_per_bag_l2155_215536


namespace measure_of_angle_S_l2155_215544

-- Define the angles in the pentagon PQRST
variables (P Q R S T : ℝ)
-- Assume the conditions from the problem
variables (h1 : P = Q)
variables (h2 : Q = R)
variables (h3 : S = T)
variables (h4 : P = S - 30)
-- Assume the sum of angles in a pentagon is 540 degrees
variables (h5 : P + Q + R + S + T = 540)

theorem measure_of_angle_S :
  S = 126 := by
  -- placeholder for the actual proof
  sorry

end measure_of_angle_S_l2155_215544


namespace maximum_height_when_isosceles_l2155_215561

variable (c : ℝ) (c1 c2 : ℝ)

def right_angled_triangle (c1 c2 c : ℝ) : Prop :=
  c1 * c1 + c2 * c2 = c * c

def isosceles_right_triangle (c1 c2 : ℝ) : Prop :=
  c1 = c2

noncomputable def height_relative_to_hypotenuse (c : ℝ) : ℝ :=
  c / 2

theorem maximum_height_when_isosceles 
  (c1 c2 c : ℝ) 
  (h_right : right_angled_triangle c1 c2 c) 
  (h_iso : isosceles_right_triangle c1 c2) :
  height_relative_to_hypotenuse c = c / 2 :=
  sorry

end maximum_height_when_isosceles_l2155_215561


namespace negation_proposition_l2155_215557

theorem negation_proposition (m : ℤ) :
  ¬(∃ x : ℤ, x^2 + 2*x + m < 0) ↔ ∀ x : ℤ, x^2 + 2*x + m ≥ 0 :=
by
  sorry

end negation_proposition_l2155_215557


namespace problem1_problem2_problem3_l2155_215579

theorem problem1 (x : ℤ) (h : 263 - x = 108) : x = 155 :=
by sorry

theorem problem2 (x : ℤ) (h : 25 * x = 1950) : x = 78 :=
by sorry

theorem problem3 (x : ℤ) (h : x / 15 = 64) : x = 960 :=
by sorry

end problem1_problem2_problem3_l2155_215579


namespace local_value_of_4_in_564823_l2155_215518

def face_value (d : ℕ) : ℕ := d
def place_value_of_thousands : ℕ := 1000
def local_value (d : ℕ) (p : ℕ) : ℕ := d * p

theorem local_value_of_4_in_564823 :
  local_value (face_value 4) place_value_of_thousands = 4000 :=
by 
  sorry

end local_value_of_4_in_564823_l2155_215518


namespace sqrt_seven_l2155_215550

theorem sqrt_seven (x : ℝ) : x^2 = 7 ↔ x = Real.sqrt 7 ∨ x = -Real.sqrt 7 := by
  sorry

end sqrt_seven_l2155_215550


namespace quadratic_no_real_roots_l2155_215588

theorem quadratic_no_real_roots (m : ℝ) : ¬ ∃ x : ℝ, x^2 + 2 * x - m = 0 → m < -1 := 
by {
  sorry
}

end quadratic_no_real_roots_l2155_215588


namespace climbing_stairs_l2155_215581

noncomputable def total_methods_to_climb_stairs : ℕ :=
  (Nat.choose 8 5) + (Nat.choose 8 6) + (Nat.choose 8 7) + 1

theorem climbing_stairs (n : ℕ := 9) (min_steps : ℕ := 6) (max_steps : ℕ := 9)
  (H1 : min_steps ≤ n)
  (H2 : n ≤ max_steps)
  : total_methods_to_climb_stairs = 93 := by
  sorry

end climbing_stairs_l2155_215581


namespace exam_students_count_l2155_215539

theorem exam_students_count (failed_students : ℕ) (failed_percentage : ℝ) (total_students : ℕ) 
    (h1 : failed_students = 260) 
    (h2 : failed_percentage = 0.65) 
    (h3 : (failed_percentage * total_students : ℝ) = (failed_students : ℝ)) : 
    total_students = 400 := 
by 
    sorry

end exam_students_count_l2155_215539


namespace divisible_by_11_and_smallest_n_implies_77_l2155_215515

theorem divisible_by_11_and_smallest_n_implies_77 (n : ℕ) (h₁ : n = 7) : ∃ m : ℕ, m = 11 * n := 
sorry

end divisible_by_11_and_smallest_n_implies_77_l2155_215515


namespace vanaspati_percentage_l2155_215590

theorem vanaspati_percentage (Q : ℝ) (h1 : 0.60 * Q > 0) (h2 : Q + 10 > 0) (h3 : Q = 10) :
    let total_ghee := Q + 10
    let pure_ghee := 0.60 * Q + 10
    let pure_ghee_fraction := pure_ghee / total_ghee
    pure_ghee_fraction = 0.80 → 
    let vanaspati_fraction := 1 - pure_ghee_fraction
    vanaspati_fraction * 100 = 40 :=
by
  intros
  sorry

end vanaspati_percentage_l2155_215590


namespace evaluation_expression_l2155_215524

theorem evaluation_expression : 
  20 * (10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5))) = 192.6 := 
by
  sorry

end evaluation_expression_l2155_215524


namespace probability_diagonals_intersect_l2155_215566

theorem probability_diagonals_intersect {n : ℕ} :
  (2 * n + 1 > 2) → 
  ∀ (total_diagonals : ℕ) (total_combinations : ℕ) (intersecting_pairs : ℕ),
    total_diagonals = 2 * n^2 - n - 1 →
    total_combinations = (total_diagonals * (total_diagonals - 1)) / 2 →
    intersecting_pairs = ((2 * n + 1) * n * (2 * n - 1) * (n - 1)) / 6 →
    (intersecting_pairs : ℚ) / (total_combinations : ℚ) = n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) := sorry

end probability_diagonals_intersect_l2155_215566


namespace grace_dimes_count_l2155_215574

-- Defining the conditions
def dimes_to_pennies (d : ℕ) : ℕ := 10 * d
def nickels_to_pennies : ℕ := 10 * 5
def total_pennies (d : ℕ) : ℕ := dimes_to_pennies d + nickels_to_pennies

-- The statement of the theorem
theorem grace_dimes_count (d : ℕ) (h : total_pennies d = 150) : d = 10 := 
sorry

end grace_dimes_count_l2155_215574


namespace trapezoid_area_l2155_215570

theorem trapezoid_area 
  (a b c : ℝ)
  (h_a : a = 5)
  (h_b : b = 15)
  (h_c : c = 13)
  : (1 / 2) * (a + b) * (Real.sqrt (c ^ 2 - ((b - a) / 2) ^ 2)) = 120 := by
  sorry

end trapezoid_area_l2155_215570


namespace inscribed_circle_radius_l2155_215522

variable (AB AC BC : ℝ) (r : ℝ)

theorem inscribed_circle_radius 
  (h1 : AB = 9) 
  (h2 : AC = 9) 
  (h3 : BC = 8) : r = (4 * Real.sqrt 65) / 13 := 
sorry

end inscribed_circle_radius_l2155_215522


namespace depth_of_water_is_60_l2155_215560

def dean_height : ℕ := 6
def depth_multiplier : ℕ := 10
def water_depth : ℕ := depth_multiplier * dean_height

theorem depth_of_water_is_60 : water_depth = 60 := by
  -- mathematical equivalent proof problem
  sorry

end depth_of_water_is_60_l2155_215560


namespace days_B_to_complete_remaining_work_l2155_215592

/-- 
  Given that:
  - A can complete a work in 20 days.
  - B can complete the same work in 12 days.
  - A and B worked together for 3 days before A left.
  
  We need to prove that B will require 7.2 days to complete the remaining work alone. 
--/
theorem days_B_to_complete_remaining_work : 
  (∃ (A_rate B_rate combined_rate work_done_in_3_days remaining_work d_B : ℚ), 
   A_rate = (1 / 20) ∧
   B_rate = (1 / 12) ∧
   combined_rate = A_rate + B_rate ∧
   work_done_in_3_days = 3 * combined_rate ∧
   remaining_work = 1 - work_done_in_3_days ∧
   d_B = remaining_work / B_rate ∧
   d_B = 7.2) := 
by 
  sorry

end days_B_to_complete_remaining_work_l2155_215592


namespace toby_candies_left_l2155_215573

def total_candies : ℕ := 56 + 132 + 8 + 300
def num_cousins : ℕ := 13

theorem toby_candies_left : total_candies % num_cousins = 2 :=
by sorry

end toby_candies_left_l2155_215573


namespace asparagus_spears_needed_l2155_215568

def BridgetteGuests : Nat := 84
def AlexGuests : Nat := (2 * BridgetteGuests) / 3
def TotalGuests : Nat := BridgetteGuests + AlexGuests
def ExtraPlates : Nat := 10
def TotalPlates : Nat := TotalGuests + ExtraPlates
def VegetarianPercent : Nat := 20
def LargePortionPercent : Nat := 10
def VegetarianMeals : Nat := (VegetarianPercent * TotalGuests) / 100
def LargePortionMeals : Nat := (LargePortionPercent * TotalGuests) / 100
def RegularMeals : Nat := TotalGuests - (VegetarianMeals + LargePortionMeals)
def AsparagusPerRegularMeal : Nat := 8
def AsparagusPerVegetarianMeal : Nat := 6
def AsparagusPerLargePortionMeal : Nat := 12

theorem asparagus_spears_needed : 
  RegularMeals * AsparagusPerRegularMeal + 
  VegetarianMeals * AsparagusPerVegetarianMeal + 
  LargePortionMeals * AsparagusPerLargePortionMeal = 1120 := by
  sorry

end asparagus_spears_needed_l2155_215568


namespace min_value_perpendicular_vectors_l2155_215551

theorem min_value_perpendicular_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (hperp : x + 3 * y = 1) : (1 / x + 1 / (3 * y)) = 4 :=
by sorry

end min_value_perpendicular_vectors_l2155_215551


namespace line_through_intersection_points_l2155_215500

def first_circle (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def second_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem line_through_intersection_points (x y : ℝ) :
  (first_circle x y ∧ second_circle x y) → x - y - 3 = 0 :=
by
  sorry

end line_through_intersection_points_l2155_215500


namespace xiao_zhang_complete_task_l2155_215516

open Nat

def xiaoZhangCharacters (n : ℕ) : ℕ :=
match n with
| 0 => 0
| (n+1) => 2 * (xiaoZhangCharacters n)

theorem xiao_zhang_complete_task :
  ∀ (total_chars : ℕ), (total_chars > 0) → 
  (xiaoZhangCharacters 5 = (total_chars / 3)) →
  (xiaoZhangCharacters 6 = total_chars) :=
by
  sorry

end xiao_zhang_complete_task_l2155_215516


namespace find_larger_number_l2155_215546

variables (x y : ℝ)

def sum_cond : Prop := x + y = 17
def diff_cond : Prop := x - y = 7

theorem find_larger_number (h1 : sum_cond x y) (h2 : diff_cond x y) : x = 12 :=
sorry

end find_larger_number_l2155_215546


namespace infinitude_of_composite_z_l2155_215510

theorem infinitude_of_composite_z (a : ℕ) (h : ∃ k : ℕ, k > 1 ∧ a = 4 * k^4) : 
  ∀ n : ℕ, ¬ Prime (n^4 + a) :=
by sorry

end infinitude_of_composite_z_l2155_215510


namespace range_of_a_l2155_215542

-- Function definition
def f (x a : ℝ) : ℝ := -x^3 + 3 * a^2 * x - 4 * a

-- Main theorem statement
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, f x a = 0) ↔ (a ∈ Set.Ioi (Real.sqrt 2)) :=
sorry

end range_of_a_l2155_215542


namespace largest_n_crates_same_number_oranges_l2155_215537

theorem largest_n_crates_same_number_oranges (total_crates : ℕ) 
  (crate_min_oranges : ℕ) (crate_max_oranges : ℕ) 
  (h1 : total_crates = 200) (h2 : crate_min_oranges = 100) (h3 : crate_max_oranges = 130) 
  : ∃ n : ℕ, n = 7 ∧ ∀ orange_count, crate_min_oranges ≤ orange_count ∧ orange_count ≤ crate_max_oranges → ∃ k, k = n ∧ ∃ t, t ≤ total_crates ∧ t ≥ k := 
sorry

end largest_n_crates_same_number_oranges_l2155_215537


namespace determine_OQ_l2155_215529

theorem determine_OQ (l m n p O A B C D Q : ℝ) (h0 : O = 0)
  (hA : A = l) (hB : B = m) (hC : C = n) (hD : D = p)
  (hQ : l ≤ Q ∧ Q ≤ m)
  (h_ratio : (|C - Q| / |Q - D|) = (|B - Q| / |Q - A|)) :
  Q = (l + m) / 2 :=
sorry

end determine_OQ_l2155_215529


namespace max_a_2017_2018_ge_2017_l2155_215594

def seq_a (a : ℕ → ℤ) (b : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ (∀ n, n ≥ 1 → 
  (b (n-1) = 1 → a (n+1) = a n * b n + a (n-1)) ∧ 
  (b (n-1) > 1 → a (n+1) = a n * b n - a (n-1)))

theorem max_a_2017_2018_ge_2017 (a : ℕ → ℤ) (b : ℕ → ℕ) (h : seq_a a b) :
  max (a 2017) (a 2018) ≥ 2017 :=
sorry

end max_a_2017_2018_ge_2017_l2155_215594


namespace algebraic_expression_eval_l2155_215509

theorem algebraic_expression_eval (a b : ℝ) 
  (h_eq : ∀ (x : ℝ), ¬(x ≠ 0 ∧ x ≠ 1 ∧ (x / (x - 1) + (x - 1) / x = (a + b * x) / (x^2 - x)))) :
  8 * a + 4 * b - 5 = 27 := 
sorry

end algebraic_expression_eval_l2155_215509


namespace measure_of_angle_Q_l2155_215504

-- Given conditions
variables (α β γ δ : ℝ)
axiom h1 : α = 130
axiom h2 : β = 95
axiom h3 : γ = 110
axiom h4 : δ = 104

-- Statement of the problem
theorem measure_of_angle_Q (Q : ℝ) (h5 : Q + α + β + γ + δ = 540) : Q = 101 := 
sorry

end measure_of_angle_Q_l2155_215504


namespace units_digit_expression_mod_10_l2155_215558

theorem units_digit_expression_mod_10 : ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 := 
by 
  -- Proof steps would go here
  sorry

end units_digit_expression_mod_10_l2155_215558


namespace infinite_positive_sequence_geometric_l2155_215567

theorem infinite_positive_sequence_geometric {a : ℕ → ℝ} (h : ∀ n ≥ 1, a (n + 2) = a n - a (n + 1)) 
  (h_pos : ∀ n, a n > 0) :
  ∃ (a1 : ℝ) (q : ℝ), q = (Real.sqrt 5 - 1) / 2 ∧ (∀ n, a n = a1 * q^(n - 1)) := by
  sorry

end infinite_positive_sequence_geometric_l2155_215567


namespace problem_quadratic_radicals_l2155_215533

theorem problem_quadratic_radicals (x y : ℝ) (h : 3 * y = x + 2 * y + 2) : x - y = -2 :=
sorry

end problem_quadratic_radicals_l2155_215533


namespace regina_earnings_l2155_215505

-- Definitions based on conditions
def num_cows := 20
def num_pigs := 4 * num_cows
def price_per_pig := 400
def price_per_cow := 800

-- Total earnings calculation based on definitions
def earnings_from_cows := num_cows * price_per_cow
def earnings_from_pigs := num_pigs * price_per_pig
def total_earnings := earnings_from_cows + earnings_from_pigs

-- Proof statement
theorem regina_earnings : total_earnings = 48000 := by
  sorry

end regina_earnings_l2155_215505


namespace find_coordinates_of_C_l2155_215562

def Point := (ℝ × ℝ)

def A : Point := (-2, -1)
def B : Point := (4, 7)

/-- A custom definition to express that point C divides the segment AB in the ratio 2:1 from point B. -/
def is_point_C (C : Point) : Prop :=
  ∃ k : ℝ, k = 2 / 3 ∧
  C = (k * A.1 + (1 - k) * B.1, k * A.2 + (1 - k) * B.2)

theorem find_coordinates_of_C (C : Point) (h : is_point_C C) : 
  C = (2, 13 / 3) :=
sorry

end find_coordinates_of_C_l2155_215562


namespace minimum_discount_l2155_215548

theorem minimum_discount (x : ℝ) (hx : x ≤ 10) : 
  let cost_price := 400 
  let selling_price := 500
  let discount_price := selling_price - (selling_price * (x / 100))
  let gross_profit := discount_price - cost_price 
  gross_profit ≥ cost_price * 0.125 :=
sorry

end minimum_discount_l2155_215548


namespace sin_cos_plus_one_l2155_215552

theorem sin_cos_plus_one (x : ℝ) (h : Real.tan x = 1 / 3) : Real.sin x * Real.cos x + 1 = 13 / 10 :=
by
  sorry

end sin_cos_plus_one_l2155_215552


namespace geometric_progression_solution_l2155_215517

theorem geometric_progression_solution (x : ℝ) :
  (2 * x + 10) ^ 2 = x * (5 * x + 10) → x = 15 + 5 * Real.sqrt 5 :=
by
  intro h
  sorry

end geometric_progression_solution_l2155_215517


namespace isosceles_triangle_side_length_l2155_215554

theorem isosceles_triangle_side_length :
  let a := 1
  let b := Real.sqrt 3
  let right_triangle_area := (1 / 2) * a * b
  let isosceles_triangle_area := right_triangle_area / 3
  ∃ s, s = Real.sqrt 109 / 6 ∧ 
    (∀ (base height : ℝ), 
      (base = a / 3 ∨ base = b / 3) ∧
      height = (2 * isosceles_triangle_area) / base → 
      1 / 2 * base * height = isosceles_triangle_area) :=
by
  sorry

end isosceles_triangle_side_length_l2155_215554


namespace linear_function_quadrants_l2155_215577

theorem linear_function_quadrants (k b : ℝ) (h : k * b < 0) : 
  (∀ x : ℝ, (k < 0 ∧ b > 0) → (k * x + b > 0 → x > 0) ∧ (k * x + b < 0 → x < 0)) ∧ 
  (∀ x : ℝ, (k > 0 ∧ b < 0) → (k * x + b > 0 → x > 0) ∧ (k * x + b < 0 → x < 0)) :=
sorry

end linear_function_quadrants_l2155_215577


namespace integer_pairs_satisfying_equation_and_nonnegative_product_l2155_215514

theorem integer_pairs_satisfying_equation_and_nonnegative_product :
  ∃ (pairs : List (ℤ × ℤ)), 
    (∀ p ∈ pairs, p.1 * p.2 ≥ 0 ∧ p.1^3 + p.2^3 + 99 * p.1 * p.2 = 33^3) ∧ 
    pairs.length = 35 :=
by sorry

end integer_pairs_satisfying_equation_and_nonnegative_product_l2155_215514


namespace expression_equality_l2155_215593

theorem expression_equality :
  (2^1001 + 5^1002)^2 - (2^1001 - 5^1002)^2 = 40 * 10^1001 := 
by
  sorry

end expression_equality_l2155_215593


namespace intersection_M_N_l2155_215586

def M := {m : ℤ | -3 < m ∧ m < 2}
def N := {x : ℤ | x * (x - 1) = 0}

theorem intersection_M_N : M ∩ N = {0, 1} := sorry

end intersection_M_N_l2155_215586


namespace find_n_l2155_215535

theorem find_n (n : ℝ) (h1 : ∀ m : ℝ, m = 4 → m^(m/2) = 4) : 
  n^(n/2) = 8 ↔ n = 2^Real.sqrt 6 :=
by
  sorry

end find_n_l2155_215535


namespace striped_jerseys_count_l2155_215508

noncomputable def totalSpent : ℕ := 80
noncomputable def longSleevedJerseyCost : ℕ := 15
noncomputable def stripedJerseyCost : ℕ := 10
noncomputable def numberOfLongSleevedJerseys : ℕ := 4

theorem striped_jerseys_count :
  (totalSpent - numberOfLongSleevedJerseys * longSleevedJerseyCost) / stripedJerseyCost = 2 := by
  sorry

end striped_jerseys_count_l2155_215508


namespace moles_of_AgOH_formed_l2155_215576

theorem moles_of_AgOH_formed (moles_AgNO3 : ℕ) (moles_NaOH : ℕ) 
  (reaction : moles_AgNO3 + moles_NaOH = 2) : moles_AgNO3 + 2 = 2 :=
by
  sorry

end moles_of_AgOH_formed_l2155_215576


namespace hyperbola_condition_l2155_215523

theorem hyperbola_condition (a : ℝ) (h : a > 0)
  (e : ℝ) (h_e : e = Real.sqrt (1 + 4 / (a^2))) :
  (e > Real.sqrt 2) ↔ (0 < a ∧ a < 1) := 
sorry

end hyperbola_condition_l2155_215523


namespace ratio_of_x_to_y_l2155_215531

-- Given condition: The percentage that y is less than x is 83.33333333333334%.
def percentage_less_than (x y : ℝ) : Prop := (x - y) / x = 0.8333333333333334

-- Prove: The ratio R = x / y is 1/6.
theorem ratio_of_x_to_y (x y : ℝ) (h : percentage_less_than x y) : x / y = 6 := 
by sorry

end ratio_of_x_to_y_l2155_215531


namespace number_of_boys_and_girls_l2155_215503

theorem number_of_boys_and_girls (b g : ℕ) 
    (h1 : ∀ n : ℕ, (n ≥ 1) → ∃ (a_n : ℕ), a_n = 2 * n + 1)
    (h2 : (2 * b + 1 = g))
    : b = (g - 1) / 2 :=
by
  sorry

end number_of_boys_and_girls_l2155_215503


namespace height_of_rectangular_block_l2155_215587

variable (V A h : ℕ)

theorem height_of_rectangular_block :
  V = 120 ∧ A = 24 ∧ V = A * h → h = 5 :=
by
  sorry

end height_of_rectangular_block_l2155_215587


namespace arnaldo_billion_difference_l2155_215502

theorem arnaldo_billion_difference :
  (10 ^ 12) - (10 ^ 9) = 999000000000 :=
by
  sorry

end arnaldo_billion_difference_l2155_215502


namespace workers_time_l2155_215555

variables (x y: ℝ)

theorem workers_time (h1 : (x > 0) ∧ (y > 0)) 
                     (h2 : (3/x + 2/y = 11/20)) 
                     (h3 : (1/x + 1/y = 1/2)) :
                     (x = 10 ∧ y = 8) := 
by
  sorry

end workers_time_l2155_215555


namespace solve_for_a_l2155_215569

theorem solve_for_a : ∃ a : ℝ, (∀ x : ℝ, x = -2 → x^2 - a * x + 7 = 0) → a = -11 / 2 :=
by 
  sorry

end solve_for_a_l2155_215569


namespace sin_2x_eq_7_div_25_l2155_215582

theorem sin_2x_eq_7_div_25 (x : ℝ) (h : Real.sin (Real.pi / 4 - x) = 3 / 5) :
    Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_2x_eq_7_div_25_l2155_215582


namespace value_is_correct_l2155_215595

-- Define the mean and standard deviation
def mean : ℝ := 14.0
def std_dev : ℝ := 1.5

-- Define the value that is 2 standard deviations less than the mean
def value : ℝ := mean - 2 * std_dev

-- Theorem stating that value = 11.0
theorem value_is_correct : value = 11.0 := by
  sorry

end value_is_correct_l2155_215595


namespace tan_45_degrees_l2155_215563

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l2155_215563


namespace probability_one_letter_from_each_l2155_215543

theorem probability_one_letter_from_each
  (total_cards : ℕ)
  (adam_cards : ℕ)
  (brian_cards : ℕ)
  (h1 : total_cards = 12)
  (h2 : adam_cards = 4)
  (h3 : brian_cards = 6)
  : (4/12 * 6/11) + (6/12 * 4/11) = 4/11 := by
  sorry

end probability_one_letter_from_each_l2155_215543


namespace angle_variance_less_than_bound_l2155_215584

noncomputable def angle_variance (α β γ : ℝ) : ℝ :=
  (1/3) * ((α - (2 * Real.pi / 3))^2 + (β - (2 * Real.pi / 3))^2 + (γ - (2 * Real.pi / 3))^2)

theorem angle_variance_less_than_bound (O A B C : ℝ → ℝ) :
  ∀ α β γ : ℝ, α + β + γ = 2 * Real.pi ∧ α ≥ β ∧ β ≥ γ → angle_variance α β γ < 2 * Real.pi^2 / 9 :=
by
  sorry

end angle_variance_less_than_bound_l2155_215584


namespace annual_interest_rate_continuous_compounding_l2155_215572

noncomputable def continuous_compounding_rate (A P : ℝ) (t : ℝ) : ℝ :=
  (Real.log (A / P)) / t

theorem annual_interest_rate_continuous_compounding :
  continuous_compounding_rate 8500 5000 10 = (Real.log (1.7)) / 10 :=
by
  sorry

end annual_interest_rate_continuous_compounding_l2155_215572


namespace min_total_rope_cut_l2155_215512

theorem min_total_rope_cut (len1 len2 len3 p1 p2 p3 p4: ℕ) (hl1 : len1 = 52) (hl2 : len2 = 37)
  (hl3 : len3 = 25) (hp1 : p1 = 7) (hp2 : p2 = 3) (hp3 : p3 = 1) 
  (hp4 : ∃ x y z : ℕ, x * p1 + y * p2 + z * p3 = len1 + len2 - len3 ∧ x + y + z ≤ 25) :
  p4 = 82 := 
sorry

end min_total_rope_cut_l2155_215512


namespace percentage_not_even_integers_l2155_215553

variable (T : ℝ) (E : ℝ)
variables (h1 : 0.36 * T = E * 0.60) -- Condition 1 translated: 36% of T are even multiples of 3.
variables (h2 : E * 0.40)            -- Condition 2 translated: 40% of E are not multiples of 3.

theorem percentage_not_even_integers : 0.40 * T = T - E :=
by
  sorry

end percentage_not_even_integers_l2155_215553


namespace tangent_line_eq_l2155_215556

theorem tangent_line_eq (x y : ℝ) (h : y = e^(-5 * x) + 2) :
  ∀ (t : ℝ), t = 0 → y = 3 → y = -5 * x + 3 :=
by
  sorry

end tangent_line_eq_l2155_215556


namespace longer_side_length_l2155_215513

-- Define the relevant entities: radius, area of the circle, and rectangle conditions.
noncomputable def radius : ℝ := 6
noncomputable def area_circle : ℝ := Real.pi * radius^2
noncomputable def area_rectangle : ℝ := 3 * area_circle
noncomputable def shorter_side : ℝ := 2 * radius

-- Prove that the length of the longer side of the rectangle is 9π cm.
theorem longer_side_length : ∃ (l : ℝ), (area_rectangle = l * shorter_side) → (l = 9 * Real.pi) :=
by
  sorry

end longer_side_length_l2155_215513


namespace bird_problem_l2155_215598

theorem bird_problem (B : ℕ) (h : (2 / 15) * B = 60) : B = 450 ∧ (2 / 15) * B = 60 :=
by
  sorry

end bird_problem_l2155_215598
