import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_a3_l757_75738

theorem geometric_sequence_a3 (q : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : a 5 = 4) (h3 : ∀ n, a n = a 1 * q ^ (n - 1)) : a 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3_l757_75738


namespace NUMINAMATH_GPT_combined_votes_l757_75770

theorem combined_votes {A B : ℕ} (h1 : A = 14) (h2 : 2 * B = A) : A + B = 21 := 
by 
sorry

end NUMINAMATH_GPT_combined_votes_l757_75770


namespace NUMINAMATH_GPT_find_a_l757_75749

noncomputable def f (x a : ℝ) : ℝ := x - Real.log (x + 2) + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem find_a (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ a = 3) → a = -Real.log 2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l757_75749


namespace NUMINAMATH_GPT_middle_tree_distance_l757_75786

theorem middle_tree_distance (d : ℕ) (b : ℕ) (c : ℕ) 
  (h_b : b = 84) (h_c : c = 91) 
  (h_right_triangle : d^2 + b^2 = c^2) : 
  d = 35 :=
by
  sorry

end NUMINAMATH_GPT_middle_tree_distance_l757_75786


namespace NUMINAMATH_GPT_calculateBooksRemaining_l757_75775

noncomputable def totalBooksRemaining
    (initialBooks : ℕ)
    (n : ℕ)
    (a₁ : ℕ)
    (d : ℕ)
    (borrowedBooks : ℕ)
    (returnedBooks : ℕ) : ℕ :=
  let sumDonations := n * (2 * a₁ + (n - 1) * d) / 2
  let totalAfterDonations := initialBooks + sumDonations
  totalAfterDonations - borrowedBooks + returnedBooks

theorem calculateBooksRemaining :
  totalBooksRemaining 1000 15 2 2 350 270 = 1160 :=
by
  sorry

end NUMINAMATH_GPT_calculateBooksRemaining_l757_75775


namespace NUMINAMATH_GPT_find_k_l757_75723

theorem find_k (k : ℝ) (h : (2 * (7:ℝ)^2) + 3 * 7 - k = 0) : k = 119 := by
  sorry

end NUMINAMATH_GPT_find_k_l757_75723


namespace NUMINAMATH_GPT_probability_B_in_A_is_17_over_24_l757_75709

open Set

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | abs p.1 + abs p.2 <= 2}
def set_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p ∈ set_A ∧ p.2 <= p.1 ^ 2}

noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry -- Assume we have means to compute the area of a set

theorem probability_B_in_A_is_17_over_24 :
  (area set_B / area set_A) = 17 / 24 :=
sorry

end NUMINAMATH_GPT_probability_B_in_A_is_17_over_24_l757_75709


namespace NUMINAMATH_GPT_symmetric_points_l757_75769

theorem symmetric_points (m n : ℤ) (h1 : m - 1 = -3) (h2 : 1 = n - 1) : m + n = 0 := by
  sorry

end NUMINAMATH_GPT_symmetric_points_l757_75769


namespace NUMINAMATH_GPT_f_decreasing_on_positive_reals_f_greater_than_2_div_x_plus_2_l757_75746

noncomputable def f (x : ℝ) : ℝ := if x > 0 then (Real.log (1 + x)) / x else 0

theorem f_decreasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x :=
sorry

theorem f_greater_than_2_div_x_plus_2 :
  ∀ x : ℝ, 0 < x → f x > 2 / (x + 2) :=
sorry

end NUMINAMATH_GPT_f_decreasing_on_positive_reals_f_greater_than_2_div_x_plus_2_l757_75746


namespace NUMINAMATH_GPT_largest_number_l757_75702

-- Define the set elements with b = -3
def neg_5b (b : ℤ) : ℤ := -5 * b
def pos_3b (b : ℤ) : ℤ := 3 * b
def frac_30_b (b : ℤ) : ℤ := 30 / b
def b_sq (b : ℤ) : ℤ := b * b

-- Prove that when b = -3, the largest element in the set {-5b, 3b, 30/b, b^2, 2} is 15
theorem largest_number (b : ℤ) (h : b = -3) : max (max (max (max (neg_5b b) (pos_3b b)) (frac_30_b b)) (b_sq b)) 2 = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_number_l757_75702


namespace NUMINAMATH_GPT_fruit_prices_l757_75754

theorem fruit_prices (x y : ℝ) 
  (h₁ : 3 * x + 2 * y = 40) 
  (h₂ : 2 * x + 3 * y = 35) : 
  x = 10 ∧ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_fruit_prices_l757_75754


namespace NUMINAMATH_GPT_solve_inequality_system_l757_75753

theorem solve_inequality_system (x : ℝ) 
  (h1 : 2 * (x - 1) < x + 3)
  (h2 : (x + 1) / 3 - x < 3) : 
  -4 < x ∧ x < 5 := 
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l757_75753


namespace NUMINAMATH_GPT_ratio_of_side_lengths_l757_75790

theorem ratio_of_side_lengths (w1 w2 : ℝ) (s1 s2 : ℝ)
  (h1 : w1 = 8) (h2 : w2 = 64)
  (v1 : w1 = s1 ^ 3)
  (v2 : w2 = s2 ^ 3) : 
  s2 / s1 = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_side_lengths_l757_75790


namespace NUMINAMATH_GPT_least_m_plus_n_l757_75791

theorem least_m_plus_n (m n : ℕ) (h1 : Nat.gcd (m + n) 231 = 1) 
                                  (h2 : m^m ∣ n^n) 
                                  (h3 : ¬ m ∣ n)
                                  : m + n = 75 :=
sorry

end NUMINAMATH_GPT_least_m_plus_n_l757_75791


namespace NUMINAMATH_GPT_no_monotonically_decreasing_l757_75742

variable (f : ℝ → ℝ)

theorem no_monotonically_decreasing (x1 x2 : ℝ) (h1 : ∃ x1 x2, x1 < x2 ∧ f x1 ≤ f x2) : ∀ x1 x2, x1 < x2 → f x1 > f x2 → False :=
by
  intros x1 x2 h2 h3
  obtain ⟨a, b, h4, h5⟩ := h1
  have contra := h5
  sorry

end NUMINAMATH_GPT_no_monotonically_decreasing_l757_75742


namespace NUMINAMATH_GPT_rebecca_end_of_day_money_eq_l757_75728

-- Define the costs for different services
def haircut_cost   := 30
def perm_cost      := 40
def dye_job_cost   := 60
def extension_cost := 80

-- Define the supply costs for the services
def haircut_supply_cost   := 5
def dye_job_supply_cost   := 10
def extension_supply_cost := 25

-- Today's appointments
def num_haircuts   := 5
def num_perms      := 3
def num_dye_jobs   := 2
def num_extensions := 1

-- Additional incomes and expenses
def tips           := 75
def daily_expenses := 45

-- Calculate the total earnings and costs
def total_service_revenue : ℕ := 
  num_haircuts * haircut_cost +
  num_perms * perm_cost +
  num_dye_jobs * dye_job_cost +
  num_extensions * extension_cost

def total_revenue : ℕ := total_service_revenue + tips

def total_supply_cost : ℕ := 
  num_haircuts * haircut_supply_cost +
  num_dye_jobs * dye_job_supply_cost +
  num_extensions * extension_supply_cost

def end_of_day_money : ℕ := total_revenue - total_supply_cost - daily_expenses

-- Lean statement to prove Rebecca will have $430 at the end of the day
theorem rebecca_end_of_day_money_eq : end_of_day_money = 430 := by
  sorry

end NUMINAMATH_GPT_rebecca_end_of_day_money_eq_l757_75728


namespace NUMINAMATH_GPT_percentage_error_calc_l757_75704

theorem percentage_error_calc (x : ℝ) (h : x ≠ 0) : 
  let correct_result := x * (5 / 3)
  let incorrect_result := x * (3 / 5)
  let percentage_error := (correct_result - incorrect_result) / correct_result * 100
  percentage_error = 64 := by
  sorry

end NUMINAMATH_GPT_percentage_error_calc_l757_75704


namespace NUMINAMATH_GPT_triangle_rectangle_ratio_l757_75772

-- Definitions of the perimeter conditions and the relationship between length and width of the rectangle.
def equilateral_triangle_side_length (t : ℕ) : Prop :=
  3 * t = 24

def rectangle_dimensions (l w : ℕ) : Prop :=
  2 * l + 2 * w = 24 ∧ l = 2 * w

-- The main theorem stating the desired ratio.
theorem triangle_rectangle_ratio (t l w : ℕ) 
  (ht : equilateral_triangle_side_length t) (hlw : rectangle_dimensions l w) : t / w = 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_rectangle_ratio_l757_75772


namespace NUMINAMATH_GPT_Dan_tshirts_total_l757_75712

theorem Dan_tshirts_total :
  (let rate1 := 1 / 12
   let rate2 := 1 / 6
   let hour := 60
   let tshirts_first_hour := hour * rate1
   let tshirts_second_hour := hour * rate2
   let total_tshirts := tshirts_first_hour + tshirts_second_hour
   total_tshirts) = 15 := by
  sorry

end NUMINAMATH_GPT_Dan_tshirts_total_l757_75712


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l757_75773

variable (x y : ℝ)

def A := {y : ℝ | ∃ x > 1, y = Real.log x / Real.log 2}
def B := {y : ℝ | ∃ x > 1, y = (1 / 2) ^ x}

theorem intersection_of_A_and_B :
  (A ∩ B) = {y : ℝ | 0 < y ∧ y < 1 / 2} :=
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l757_75773


namespace NUMINAMATH_GPT_triplet_solution_l757_75778

theorem triplet_solution (a b c : ℝ)
  (h1 : a^2 + b = c^2)
  (h2 : b^2 + c = a^2)
  (h3 : c^2 + a = b^2) :
  (a = 0 ∧ b = 0 ∧ c = 0) ∨
  (a = 0 ∧ b = 1 ∧ c = -1) ∨
  (a = -1 ∧ b = 0 ∧ c = 1) ∨
  (a = 1 ∧ b = -1 ∧ c = 0) :=
sorry

end NUMINAMATH_GPT_triplet_solution_l757_75778


namespace NUMINAMATH_GPT_geometric_sequence_S6_div_S3_l757_75761

theorem geometric_sequence_S6_div_S3 (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h1 : a 1 + a 3 = 5 / 4)
  (h2 : a 2 + a 4 = 5 / 2)
  (hS : ∀ n, S n = a 1 * (1 - (2:ℝ) ^ n) / (1 - 2)) :
  S 6 / S 3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_S6_div_S3_l757_75761


namespace NUMINAMATH_GPT_smallest_integer_inequality_l757_75724

theorem smallest_integer_inequality :
  (∃ n : ℤ, ∀ x y z : ℝ, (x + y + z)^2 ≤ (n:ℝ) * (x^2 + y^2 + z^2)) ∧
  ∀ m : ℤ, (∀ x y z : ℝ, (x + y + z)^2 ≤ (m:ℝ) * (x^2 + y^2 + z^2)) → 3 ≤ m :=
  sorry

end NUMINAMATH_GPT_smallest_integer_inequality_l757_75724


namespace NUMINAMATH_GPT_carterHas152Cards_l757_75795

-- Define the number of baseball cards Marcus has.
def marcusCards : Nat := 210

-- Define the number of baseball cards Carter has.
def carterCards : Nat := marcusCards - 58

-- Theorem to prove Carter's baseball cards total 152 given the conditions.
theorem carterHas152Cards (h1 : marcusCards = 210) (h2 : marcusCards = carterCards + 58) : carterCards = 152 :=
by
  -- Proof omitted for this exercise
  sorry

end NUMINAMATH_GPT_carterHas152Cards_l757_75795


namespace NUMINAMATH_GPT_find_f_neg12_add_f_14_l757_75771

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log (Real.sqrt (x^2 - 2*x + 2) - x + 1)

theorem find_f_neg12_add_f_14 : f (-12) + f 14 = 2 :=
by
  -- The hard part, the actual proof, is left as sorry.
  sorry

end NUMINAMATH_GPT_find_f_neg12_add_f_14_l757_75771


namespace NUMINAMATH_GPT_quadratic_real_equal_roots_l757_75794

theorem quadratic_real_equal_roots (m : ℝ) :
  (3*x^2 + (2 - m)*x + 5 = 0 → (3 : ℕ) * x^2 + ((2 : ℕ) - m) * x + (5 : ℕ) = 0) →
  ∃ m₁ m₂ : ℝ, m₁ = 2 - 2 * Real.sqrt 15 ∧ m₂ = 2 + 2 * Real.sqrt 15 ∧ 
    (∀ x : ℝ, (3 * x^2 + (2 - m₁) * x + 5 = 0) ∧ (3 * x^2 + (2 - m₂) * x + 5 = 0)) :=
sorry

end NUMINAMATH_GPT_quadratic_real_equal_roots_l757_75794


namespace NUMINAMATH_GPT_average_book_width_l757_75725

-- Define the widths of the books as given in the problem conditions
def widths : List ℝ := [3, 7.5, 1.25, 0.75, 4, 12]

-- Define the number of books from the problem conditions
def num_books : ℝ := 6

-- We prove that the average width of the books is equal to 4.75
theorem average_book_width : (widths.sum / num_books) = 4.75 :=
by
  sorry

end NUMINAMATH_GPT_average_book_width_l757_75725


namespace NUMINAMATH_GPT_other_asymptote_l757_75758

theorem other_asymptote (a b : ℝ) :
  (∀ x y : ℝ, y = 2 * x → y - b = a * (x - (-4))) ∧
  (∀ c d : ℝ, c = -4) →
  ∃ m b' : ℝ, m = -1/2 ∧ b' = -10 ∧ ∀ x y : ℝ, y = m * x + b' :=
by
  sorry

end NUMINAMATH_GPT_other_asymptote_l757_75758


namespace NUMINAMATH_GPT_tangent_parallel_l757_75747

theorem tangent_parallel (a b : ℝ) 
  (h1 : b = (1 / 3) * a^3 - (1 / 2) * a^2 + 1) 
  (h2 : (a^2 - a) = 2) : 
  a = 2 ∨ a = -1 :=
by {
  -- proof skipped
  sorry
}

end NUMINAMATH_GPT_tangent_parallel_l757_75747


namespace NUMINAMATH_GPT_amy_final_money_l757_75756

theorem amy_final_money :
  let initial_money := 2
  let chore_payment := 5 * 13
  let birthday_gift := 3
  let toy_cost := 12
  let remaining_money := initial_money + chore_payment + birthday_gift - toy_cost
  let grandparents_reward := 2 * remaining_money
  remaining_money + grandparents_reward = 174 := 
by
  sorry

end NUMINAMATH_GPT_amy_final_money_l757_75756


namespace NUMINAMATH_GPT_average_age_of_group_l757_75763

theorem average_age_of_group :
  let n_graders := 40
  let n_parents := 50
  let n_teachers := 10
  let avg_age_graders := 12
  let avg_age_parents := 35
  let avg_age_teachers := 45
  let total_individuals := n_graders + n_parents + n_teachers
  let total_age := n_graders * avg_age_graders + n_parents * avg_age_parents + n_teachers * avg_age_teachers
  (total_age : ℚ) / total_individuals = 26.8 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_group_l757_75763


namespace NUMINAMATH_GPT_maximize_xyplusxzplusyzplusy2_l757_75731

theorem maximize_xyplusxzplusyzplusy2 (x y z : ℝ) (h1 : x + 2 * y + z = 7) (h2 : y ≥ 0) :
  xy + xz + yz + y^2 ≤ 10.5 :=
sorry

end NUMINAMATH_GPT_maximize_xyplusxzplusyzplusy2_l757_75731


namespace NUMINAMATH_GPT_bonus_percentage_correct_l757_75751

/-
Tom serves 10 customers per hour and works for 8 hours, earning 16 bonus points.
We need to find the percentage of bonus points per customer served.
-/

def customers_per_hour : ℕ := 10
def hours_worked : ℕ := 8
def total_bonus_points : ℕ := 16

def total_customers_served : ℕ := customers_per_hour * hours_worked
def bonus_percentage : ℕ := (total_bonus_points * 100) / total_customers_served

theorem bonus_percentage_correct : bonus_percentage = 20 := by
  sorry

end NUMINAMATH_GPT_bonus_percentage_correct_l757_75751


namespace NUMINAMATH_GPT_martha_painting_rate_l757_75718

noncomputable def martha_square_feet_per_hour
  (width1 : ℕ) (width2 : ℕ) (height : ℕ) (coats : ℕ) (total_hours : ℕ) 
  (pair1_walls : ℕ) (pair2_walls : ℕ) : ℕ :=
  let pair1_total_area := width1 * height * pair1_walls
  let pair2_total_area := width2 * height * pair2_walls
  let total_area := pair1_total_area + pair2_total_area
  let total_paint_area := total_area * coats
  total_paint_area / total_hours

theorem martha_painting_rate :
  martha_square_feet_per_hour 12 16 10 3 42 2 2 = 40 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_martha_painting_rate_l757_75718


namespace NUMINAMATH_GPT_meaningful_range_fraction_l757_75789

theorem meaningful_range_fraction (x : ℝ) : 
  ¬ (x = 3) ↔ (∃ y, y = x / (x - 3)) :=
sorry

end NUMINAMATH_GPT_meaningful_range_fraction_l757_75789


namespace NUMINAMATH_GPT_kimiko_watched_4_videos_l757_75717

/-- Kimiko's videos. --/
def first_video_length := 120
def second_video_length := 270
def last_two_video_length := 60
def total_time_watched := 510

theorem kimiko_watched_4_videos :
  first_video_length + second_video_length + last_two_video_length + last_two_video_length = total_time_watched → 
  4 = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_kimiko_watched_4_videos_l757_75717


namespace NUMINAMATH_GPT_shelby_initial_money_l757_75737

-- Definitions based on conditions
def cost_of_first_book : ℕ := 8
def cost_of_second_book : ℕ := 4
def cost_of_each_poster : ℕ := 4
def number_of_posters : ℕ := 2

-- Number to prove (initial money)
def initial_money : ℕ := 20

-- Theorem statement
theorem shelby_initial_money :
    (cost_of_first_book + cost_of_second_book + (number_of_posters * cost_of_each_poster)) = initial_money := by
    sorry

end NUMINAMATH_GPT_shelby_initial_money_l757_75737


namespace NUMINAMATH_GPT_painters_workdays_l757_75743

theorem painters_workdays (five_painters_days : ℝ) (four_painters_days : ℝ) : 
  (5 * five_painters_days = 9) → (4 * four_painters_days = 9) → (four_painters_days = 2.25) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_painters_workdays_l757_75743


namespace NUMINAMATH_GPT_solution_l757_75711

def question (x : ℝ) : Prop := (x - 5) / ((x - 3) ^ 2) < 0

theorem solution :
  {x : ℝ | question x} = {x : ℝ | x < 3} ∪ {x : ℝ | 3 < x ∧ x < 5} :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_l757_75711


namespace NUMINAMATH_GPT_total_investment_amount_l757_75739

-- Define the conditions
def total_interest_in_one_year : ℝ := 1023
def invested_at_6_percent : ℝ := 8200
def interest_rate_6_percent : ℝ := 0.06
def interest_rate_7_5_percent : ℝ := 0.075

-- Define the equation based on the conditions
def interest_from_6_percent_investment : ℝ := invested_at_6_percent * interest_rate_6_percent

def total_investment_is_correct (T : ℝ) : Prop :=
  let interest_from_7_5_percent_investment := (T - invested_at_6_percent) * interest_rate_7_5_percent
  interest_from_6_percent_investment + interest_from_7_5_percent_investment = total_interest_in_one_year

-- Statement to prove
theorem total_investment_amount : total_investment_is_correct 15280 :=
by
  unfold total_investment_is_correct
  unfold interest_from_6_percent_investment
  simp
  sorry

end NUMINAMATH_GPT_total_investment_amount_l757_75739


namespace NUMINAMATH_GPT_sum_of_first_and_last_l757_75752

noncomputable section

variables {A B C D E F G H I : ℕ}

theorem sum_of_first_and_last :
  (D = 8) →
  (A + B + C + D = 50) →
  (B + C + D + E = 50) →
  (C + D + E + F = 50) →
  (D + E + F + G = 50) →
  (E + F + G + H = 50) →
  (F + G + H + I = 50) →
  (A + I = 92) :=
by
  intros hD h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_sum_of_first_and_last_l757_75752


namespace NUMINAMATH_GPT_find_temp_friday_l757_75701

-- Definitions for conditions
variables (M T W Th F : ℝ)

-- Condition 1: Average temperature for Monday to Thursday is 48 degrees
def avg_temp_mon_thu : Prop := (M + T + W + Th) / 4 = 48

-- Condition 2: Average temperature for Tuesday to Friday is 46 degrees
def avg_temp_tue_fri : Prop := (T + W + Th + F) / 4 = 46

-- Condition 3: Temperature on Monday is 39 degrees
def temp_monday : Prop := M = 39

-- Theorem: Temperature on Friday is 31 degrees
theorem find_temp_friday (h1 : avg_temp_mon_thu M T W Th)
                         (h2 : avg_temp_tue_fri T W Th F)
                         (h3 : temp_monday M) :
  F = 31 :=
sorry

end NUMINAMATH_GPT_find_temp_friday_l757_75701


namespace NUMINAMATH_GPT_sequence_periodic_l757_75788

theorem sequence_periodic (a : ℕ → ℚ) (h1 : a 1 = 4 / 5)
  (h2 : ∀ n, 0 ≤ a n ∧ a n ≤ 1 → 
    (a (n + 1) = if a n ≤ 1 / 2 then 2 * a n else 2 * a n - 1)) :
  a 2017 = 4 / 5 :=
sorry

end NUMINAMATH_GPT_sequence_periodic_l757_75788


namespace NUMINAMATH_GPT_tickets_difference_l757_75762

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

end NUMINAMATH_GPT_tickets_difference_l757_75762


namespace NUMINAMATH_GPT_problem_statement_l757_75735

theorem problem_statement {n d : ℕ} (hn : 0 < n) (hd : 0 < d) (h1 : d ∣ n) (h2 : d^2 * n + 1 ∣ n^2 + d^2) :
  n = d^2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l757_75735


namespace NUMINAMATH_GPT_find_values_of_M_l757_75713

theorem find_values_of_M :
  ∃ M : ℕ, 
    (M = 81 ∨ M = 92) ∧ 
    (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ M = 10 * a + b ∧
     (∃ k : ℕ, k ^ 3 = 9 * (a - b) ∧ k > 0)) :=
sorry

end NUMINAMATH_GPT_find_values_of_M_l757_75713


namespace NUMINAMATH_GPT_gasoline_price_percent_increase_l757_75783

theorem gasoline_price_percent_increase 
  (highest_price : ℕ) (lowest_price : ℕ) 
  (h_highest : highest_price = 17) 
  (h_lowest : lowest_price = 10) : 
  (highest_price - lowest_price) * 100 / lowest_price = 70 := 
by 
  sorry

end NUMINAMATH_GPT_gasoline_price_percent_increase_l757_75783


namespace NUMINAMATH_GPT_part1_part2_part3_l757_75792

section CircleLine

-- Given: Circle C with equation x^2 + y^2 - 2x - 2y + 1 = 0
-- Tangent to line l intersecting the x-axis at A and the y-axis at B
variable (a b : ℝ) (ha : a > 2) (hb : b > 2)

-- Ⅰ. Prove that (a - 2)(b - 2) = 2
theorem part1 : (a - 2) * (b - 2) = 2 :=
sorry

-- Ⅱ. Find the equation of the trajectory of the midpoint of segment AB
theorem part2 (x y : ℝ) (hx : x > 1) (hy : y > 1) : (x - 1) * (y - 1) = 1 :=
sorry

-- Ⅲ. Find the minimum value of the area of triangle AOB
theorem part3 : ∃ (area : ℝ), area = 6 :=
sorry

end CircleLine

end NUMINAMATH_GPT_part1_part2_part3_l757_75792


namespace NUMINAMATH_GPT_meal_preppers_activity_setters_count_l757_75708

-- Definitions for the problem conditions
def num_friends : ℕ := 6
def num_meal_preppers : ℕ := 3

-- Statement of the theorem
theorem meal_preppers_activity_setters_count :
  (num_friends.choose num_meal_preppers) = 20 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_meal_preppers_activity_setters_count_l757_75708


namespace NUMINAMATH_GPT_letters_with_both_l757_75722

/-
In a certain alphabet, some letters contain a dot and a straight line. 
36 letters contain a straight line but do not contain a dot. 
The alphabet has 60 letters, all of which contain either a dot or a straight line or both. 
There are 4 letters that contain a dot but do not contain a straight line. 
-/
def L_no_D : ℕ := 36
def D_no_L : ℕ := 4
def total_letters : ℕ := 60

theorem letters_with_both (DL : ℕ) : 
  total_letters = D_no_L + L_no_D + DL → 
  DL = 20 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_letters_with_both_l757_75722


namespace NUMINAMATH_GPT_evaluate_f_at_2_l757_75736

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 1

theorem evaluate_f_at_2 : f 2 = 5 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_2_l757_75736


namespace NUMINAMATH_GPT_bianca_points_earned_l757_75721

-- Define the constants and initial conditions
def points_per_bag : ℕ := 5
def total_bags : ℕ := 17
def not_recycled_bags : ℕ := 8

-- Define a function to calculate the number of recycled bags
def recycled_bags (total: ℕ) (not_recycled: ℕ) : ℕ :=
  total - not_recycled

-- Define a function to calculate the total points earned
def total_points_earned (bags: ℕ) (points_per_bag: ℕ) : ℕ :=
  bags * points_per_bag

-- State the theorem
theorem bianca_points_earned : total_points_earned (recycled_bags total_bags not_recycled_bags) points_per_bag = 45 :=
by
  sorry

end NUMINAMATH_GPT_bianca_points_earned_l757_75721


namespace NUMINAMATH_GPT_line_equation_intersections_l757_75767

theorem line_equation_intersections (m b k : ℝ) (h1 : b ≠ 0) 
  (h2 : m * 2 + b = 7) (h3 : abs (k^2 + 8*k + 7 - (m*k + b)) = 4) :
  m = 6 ∧ b = -5 :=
by {
  sorry
}

end NUMINAMATH_GPT_line_equation_intersections_l757_75767


namespace NUMINAMATH_GPT_coin_die_sum_probability_l757_75719

theorem coin_die_sum_probability : 
  let coin_sides := [5, 15]
  let die_sides := [1, 2, 3, 4, 5, 6]
  let ben_age := 18
  (1 / 2 : ℚ) * (1 / 6 : ℚ) = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_coin_die_sum_probability_l757_75719


namespace NUMINAMATH_GPT_simplified_sum_l757_75748

theorem simplified_sum :
  (-1 : ℤ) ^ 2002 + (-1 : ℤ) ^ 2003 + 2 ^ 2004 - 2 ^ 2003 = 2 ^ 2003 := 
by 
  sorry -- Proof skipped

end NUMINAMATH_GPT_simplified_sum_l757_75748


namespace NUMINAMATH_GPT_first_train_speed_l757_75744

noncomputable def speed_first_train (length1 length2 : ℝ) (speed2 time : ℝ) : ℝ :=
  let distance := (length1 + length2) / 1000
  let time_hours := time / 3600
  (distance / time_hours) - speed2

theorem first_train_speed :
  speed_first_train 100 280 30 18.998480121590273 = 42 :=
by
  sorry

end NUMINAMATH_GPT_first_train_speed_l757_75744


namespace NUMINAMATH_GPT_fraction_at_x_eq_4571_div_39_l757_75797

def numerator (x : ℕ) : ℕ := x^6 - 16 * x^3 + x^2 + 64
def denominator (x : ℕ) : ℕ := x^3 - 8

theorem fraction_at_x_eq_4571_div_39 : numerator 5 / denominator 5 = 4571 / 39 :=
by
  sorry

end NUMINAMATH_GPT_fraction_at_x_eq_4571_div_39_l757_75797


namespace NUMINAMATH_GPT_calculate_49_squared_l757_75720

theorem calculate_49_squared : 
  ∀ (a b : ℕ), a = 50 → b = 2 → (a - b)^2 = a^2 - 2 * a * b + b^2 → (49^2 = 50^2 - 196) :=
by
  intro a b h1 h2 h3
  sorry

end NUMINAMATH_GPT_calculate_49_squared_l757_75720


namespace NUMINAMATH_GPT_joe_purchased_360_gallons_l757_75707

def joe_initial_paint (P : ℝ) : Prop :=
  let first_week_paint := (1/4) * P
  let remaining_paint := (3/4) * P
  let second_week_paint := (1/2) * remaining_paint
  let total_used_paint := first_week_paint + second_week_paint
  total_used_paint = 225

theorem joe_purchased_360_gallons : ∃ P : ℝ, joe_initial_paint P ∧ P = 360 :=
by
  sorry

end NUMINAMATH_GPT_joe_purchased_360_gallons_l757_75707


namespace NUMINAMATH_GPT_exists_five_positive_integers_sum_20_product_420_l757_75745
-- Import the entirety of Mathlib to ensure all necessary definitions are available

-- Lean statement for the proof problem
theorem exists_five_positive_integers_sum_20_product_420 :
  ∃ (a b c d e : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ a + b + c + d + e = 20 ∧ a * b * c * d * e = 420 :=
sorry

end NUMINAMATH_GPT_exists_five_positive_integers_sum_20_product_420_l757_75745


namespace NUMINAMATH_GPT_c_difference_correct_l757_75768

noncomputable def find_c_difference (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 20) : ℝ :=
  2 * Real.sqrt 34

theorem c_difference_correct (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 20) :
  find_c_difference a b c h1 h2 = 2 * Real.sqrt 34 := 
sorry

end NUMINAMATH_GPT_c_difference_correct_l757_75768


namespace NUMINAMATH_GPT_c_share_l757_75799

theorem c_share (a b c d e : ℝ) (k : ℝ)
  (h1 : a + b + c + d + e = 1010)
  (h2 : a - 25 = 4 * k)
  (h3 : b - 10 = 3 * k)
  (h4 : c - 15 = 6 * k)
  (h5 : d - 20 = 2 * k)
  (h6 : e - 30 = 5 * k) :
  c = 288 :=
by
  -- proof with necessary steps
  sorry

end NUMINAMATH_GPT_c_share_l757_75799


namespace NUMINAMATH_GPT_g_neither_even_nor_odd_l757_75729

noncomputable def g (x : ℝ) : ℝ := 3 ^ (x^2 - 3) - |x| + Real.sin x

theorem g_neither_even_nor_odd : ∀ x : ℝ, g x ≠ g (-x) ∧ g x ≠ -g (-x) := 
by
  intro x
  sorry

end NUMINAMATH_GPT_g_neither_even_nor_odd_l757_75729


namespace NUMINAMATH_GPT_percentage_of_whole_is_10_l757_75705

def part : ℝ := 0.01
def whole : ℝ := 0.1

theorem percentage_of_whole_is_10 : (part / whole) * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_of_whole_is_10_l757_75705


namespace NUMINAMATH_GPT_average_speed_is_55_l757_75700

theorem average_speed_is_55 
  (initial_reading : ℕ) (final_reading : ℕ) (time_hours : ℕ)
  (H1 : initial_reading = 15951) 
  (H2 : final_reading = 16061)
  (H3 : time_hours = 2) : 
  (final_reading - initial_reading) / time_hours = 55 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_is_55_l757_75700


namespace NUMINAMATH_GPT_initial_blue_balls_l757_75784

theorem initial_blue_balls (B : ℕ) (h1 : 25 - 5 = 20) (h2 : (B - 5) / 20 = 1 / 5) : B = 9 :=
by
  sorry

end NUMINAMATH_GPT_initial_blue_balls_l757_75784


namespace NUMINAMATH_GPT_nested_g_of_2_l757_75730

def g (x : ℤ) : ℤ := x^2 - 4*x + 3

theorem nested_g_of_2 : g (g (g (g (g (g 2))))) = 1394486148248 := by
  sorry

end NUMINAMATH_GPT_nested_g_of_2_l757_75730


namespace NUMINAMATH_GPT_prob_at_least_6_heads_eq_l757_75766

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end NUMINAMATH_GPT_prob_at_least_6_heads_eq_l757_75766


namespace NUMINAMATH_GPT_count_ordered_pairs_l757_75710

theorem count_ordered_pairs : 
  ∃ n : ℕ, n = 136 ∧ 
  ∀ a b : ℝ, 
    (∃ x y : ℤ, a * x + b * y = 2 ∧ (x : ℝ)^2 + (y : ℝ)^2 = 65) → n = 136 := 
sorry

end NUMINAMATH_GPT_count_ordered_pairs_l757_75710


namespace NUMINAMATH_GPT_determinant_value_l757_75726

variable (a1 b1 b2 c1 c2 c3 d1 d2 d3 d4 : ℝ)

def matrix_det : ℝ :=
  Matrix.det ![
    ![a1, b1, c1, d1],
    ![a1, b2, c2, d2],
    ![a1, b2, c3, d3],
    ![a1, b2, c3, d4]
  ]

theorem determinant_value : 
  matrix_det a1 b1 b2 c1 c2 c3 d1 d2 d3 d4 = 
  a1 * (b2 - b1) * (c3 - c2) * (d4 - d3) :=
by
  sorry

end NUMINAMATH_GPT_determinant_value_l757_75726


namespace NUMINAMATH_GPT_sheep_count_l757_75796

theorem sheep_count {c s : ℕ} 
  (h1 : c + s = 20)
  (h2 : 2 * c + 4 * s = 60) : s = 10 :=
sorry

end NUMINAMATH_GPT_sheep_count_l757_75796


namespace NUMINAMATH_GPT_trigonometric_identity_l757_75732

open Real

theorem trigonometric_identity (α : ℝ) (h1 : cos α = -4 / 5) (h2 : π < α ∧ α < (3 * π / 2)) :
    (1 + tan (α / 2)) / (1 - tan (α / 2)) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l757_75732


namespace NUMINAMATH_GPT_find_d10_bills_l757_75782

variable (V : Int) (d10 d20 : Int)

-- Given conditions
def spent_money (d10 d20 : Int) : Int := 10 * d10 + 20 * d20

axiom spent_amount : spent_money d10 d20 = 80
axiom more_20_bills : d20 = d10 + 1

-- Question to prove
theorem find_d10_bills : d10 = 2 :=
by {
  -- We mark the theorem to be proven
  sorry
}

end NUMINAMATH_GPT_find_d10_bills_l757_75782


namespace NUMINAMATH_GPT_least_xy_value_l757_75785

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 96 :=
by sorry

end NUMINAMATH_GPT_least_xy_value_l757_75785


namespace NUMINAMATH_GPT_john_experience_when_mike_started_l757_75733

-- Definitions from the conditions
variable (J O M : ℕ)
variable (h1 : J = 20) -- James currently has 20 years of experience
variable (h2 : O - 8 = 2 * (J - 8)) -- 8 years ago, John had twice as much experience as James
variable (h3 : J + O + M = 68) -- Combined experience is 68 years

-- Theorem to prove
theorem john_experience_when_mike_started : O - M = 16 := 
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_john_experience_when_mike_started_l757_75733


namespace NUMINAMATH_GPT_at_least_one_angle_ge_60_l757_75760

theorem at_least_one_angle_ge_60 (A B C : ℝ) (hA : A < 60) (hB : B < 60) (hC : C < 60) (h_sum : A + B + C = 180) : false :=
sorry

end NUMINAMATH_GPT_at_least_one_angle_ge_60_l757_75760


namespace NUMINAMATH_GPT_valid_n_values_l757_75759

theorem valid_n_values (n x y : ℤ) (h1 : n * (x - 3) = y + 3) (h2 : x + n = 3 * (y - n)) :
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7 :=
sorry

end NUMINAMATH_GPT_valid_n_values_l757_75759


namespace NUMINAMATH_GPT_original_savings_l757_75780

variable (A B : ℕ)

-- A's savings are 5 times that of B's savings
def cond1 : Prop := A = 5 * B

-- If A withdraws 60 yuan and B deposits 60 yuan, then B's savings will be twice that of A's savings
def cond2 : Prop := (B + 60) = 2 * (A - 60)

-- Prove the original savings of A and B
theorem original_savings (h1 : cond1 A B) (h2 : cond2 A B) : A = 100 ∧ B = 20 := by
  sorry

end NUMINAMATH_GPT_original_savings_l757_75780


namespace NUMINAMATH_GPT_right_triangle_area_l757_75764

theorem right_triangle_area
  (hypotenuse : ℝ) (leg1 : ℝ) (leg2 : ℝ)
  (hypotenuse_eq : hypotenuse = 13)
  (leg1_eq : leg1 = 5)
  (pythagorean_eq : hypotenuse^2 = leg1^2 + leg2^2) :
  (1 / 2) * leg1 * leg2 = 30 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l757_75764


namespace NUMINAMATH_GPT_find_number_l757_75750

theorem find_number (x : ℤ) : 45 - (28 - (x - (15 - 16))) = 55 ↔ x = 37 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l757_75750


namespace NUMINAMATH_GPT_complement_union_l757_75798

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {3, 4, 5}

theorem complement_union :
  ((U \ A) ∪ B) = {1, 3, 4, 5, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l757_75798


namespace NUMINAMATH_GPT_algebraic_expression_value_l757_75755

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3*x - 1 = 0) :
  (x - 3)^2 - (2*x + 1)*(2*x - 1) - 3*x = 7 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l757_75755


namespace NUMINAMATH_GPT_number_of_12_digit_numbers_with_consecutive_digits_same_l757_75781

theorem number_of_12_digit_numbers_with_consecutive_digits_same : 
  let total := (2 : ℕ) ^ 12
  let excluded := 2
  total - excluded = 4094 :=
by
  let total := (2 : ℕ) ^ 12
  let excluded := 2
  have h : total = 4096 := by norm_num
  have h' : total - excluded = 4094 := by norm_num
  exact h'

end NUMINAMATH_GPT_number_of_12_digit_numbers_with_consecutive_digits_same_l757_75781


namespace NUMINAMATH_GPT_count_common_divisors_l757_75787

theorem count_common_divisors : 
  (Nat.divisors 60 ∩ Nat.divisors 90 ∩ Nat.divisors 30).card = 8 :=
by
  sorry

end NUMINAMATH_GPT_count_common_divisors_l757_75787


namespace NUMINAMATH_GPT_servings_per_pie_l757_75734

theorem servings_per_pie (serving_apples : ℝ) (guests : ℕ) (pies : ℕ) (apples_per_guest : ℝ)
  (H_servings: serving_apples = 1.5) 
  (H_guests: guests = 12)
  (H_pies: pies = 3)
  (H_apples_per_guest: apples_per_guest = 3) :
  (guests * apples_per_guest) / (serving_apples * pies) = 8 :=
by
  rw [H_servings, H_guests, H_pies, H_apples_per_guest]
  sorry

end NUMINAMATH_GPT_servings_per_pie_l757_75734


namespace NUMINAMATH_GPT_solve_cos_sin_eq_one_l757_75727

open Real

theorem solve_cos_sin_eq_one (n : ℕ) (hn : n > 0) :
  {x : ℝ | cos x ^ n - sin x ^ n = 1} = {x : ℝ | ∃ k : ℤ, x = k * π} :=
by
  sorry

end NUMINAMATH_GPT_solve_cos_sin_eq_one_l757_75727


namespace NUMINAMATH_GPT_johns_equation_l757_75765

theorem johns_equation (a b c d e : ℤ) (ha : a = 2) (hb : b = 3) 
  (hc : c = 4) (hd : d = 5) : 
  a - (b - (c * (d - e))) = a - b - c * d + e ↔ e = 8 := 
by
  sorry

end NUMINAMATH_GPT_johns_equation_l757_75765


namespace NUMINAMATH_GPT_soaked_part_solution_l757_75757

theorem soaked_part_solution 
  (a b : ℝ) (c : ℝ) 
  (h : c * (2/3) * a * b = 2 * a^2 * b^3 + (1/3) * a^3 * b^2) :
  c = 3 * a * b^2 + (1/2) * a^2 * b :=
by
  sorry

end NUMINAMATH_GPT_soaked_part_solution_l757_75757


namespace NUMINAMATH_GPT_num_factors_x_l757_75779

theorem num_factors_x (x : ℕ) (h : 2011^(2011^2012) = x^x) : ∃ n : ℕ, n = 2012 ∧  ∀ d : ℕ, d ∣ x -> d ≤ n :=
sorry

end NUMINAMATH_GPT_num_factors_x_l757_75779


namespace NUMINAMATH_GPT_minimum_dot_product_l757_75776

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 36) + (y^2 / 9) = 1

def K : (ℝ × ℝ) := (2, 0)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

theorem minimum_dot_product (M N : ℝ × ℝ) (hM : ellipse M.1 M.2) (hN : ellipse N.1 N.2) (h : dot_product (vector_sub M K) (vector_sub N K) = 0) :
  ∃ α β : ℝ, 0 ≤ α ∧ α < 2 * Real.pi ∧ 0 ≤ β ∧ β < 2 * Real.pi ∧ M = (6 * Real.cos α, 3 * Real.sin α) ∧ N = (6 * Real.cos β, 3 * Real.sin β) ∧
  (∃ C : ℝ, C = 23 / 3 ∧ ∀ M N, ellipse M.1 M.2 → ellipse N.1 N.2 → dot_product (vector_sub M K) (vector_sub N K) = 0 → dot_product (vector_sub M K) (vector_sub (vector_sub M N) K) >= C) :=
sorry

end NUMINAMATH_GPT_minimum_dot_product_l757_75776


namespace NUMINAMATH_GPT_no_solution_to_system_l757_75740

theorem no_solution_to_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 8) ∧ (6 * x - 8 * y = 18) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_to_system_l757_75740


namespace NUMINAMATH_GPT_evaluate_expression_l757_75793

theorem evaluate_expression : 
  let a := 3 
  let b := 2 
  (a^2 + b)^2 - (a^2 - b)^2 + 2*a*b = 78 := 
by
  let a := 3
  let b := 2
  sorry

end NUMINAMATH_GPT_evaluate_expression_l757_75793


namespace NUMINAMATH_GPT_ratio_problem_l757_75703

theorem ratio_problem (X : ℕ) :
  (18 : ℕ) * 360 = 9 * X → X = 720 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ratio_problem_l757_75703


namespace NUMINAMATH_GPT_new_table_capacity_is_six_l757_75741

-- Definitions based on the conditions
def total_tables : ℕ := 40
def extra_new_tables : ℕ := 12
def total_customers : ℕ := 212
def original_table_capacity : ℕ := 4

-- Main statement to prove
theorem new_table_capacity_is_six (O N C : ℕ) 
  (h1 : O + N = total_tables)
  (h2 : N = O + extra_new_tables)
  (h3 : O * original_table_capacity + N * C = total_customers) :
  C = 6 :=
sorry

end NUMINAMATH_GPT_new_table_capacity_is_six_l757_75741


namespace NUMINAMATH_GPT_greatest_divisor_arithmetic_sequence_sum_l757_75715

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end NUMINAMATH_GPT_greatest_divisor_arithmetic_sequence_sum_l757_75715


namespace NUMINAMATH_GPT_janet_wait_time_l757_75714

theorem janet_wait_time 
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : lake_width = 60) 
  :
  60 / 12 - 60 / 30 = 3 :=
by
  sorry

end NUMINAMATH_GPT_janet_wait_time_l757_75714


namespace NUMINAMATH_GPT_exists_prime_not_dividing_difference_l757_75774

theorem exists_prime_not_dividing_difference {m : ℕ} (hm : m ≠ 1) : 
  ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, ¬ p ∣ (n^n - m) := 
sorry

end NUMINAMATH_GPT_exists_prime_not_dividing_difference_l757_75774


namespace NUMINAMATH_GPT_prism_faces_eq_nine_l757_75777

-- Define the condition: a prism with 21 edges
def prism_edges (n : ℕ) := n = 21

-- Define the number of sides on each polygonal base
def num_sides (L : ℕ) := 3 * L = 21

-- Define the total number of faces
def total_faces (F : ℕ) (L : ℕ) := F = L + 2

-- The theorem we want to prove
theorem prism_faces_eq_nine (n L F : ℕ) 
  (h1 : prism_edges n)
  (h2 : num_sides L)
  (h3 : total_faces F L) :
  F = 9 := 
sorry

end NUMINAMATH_GPT_prism_faces_eq_nine_l757_75777


namespace NUMINAMATH_GPT_min_value_x_plus_y_l757_75716

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 4 / x + 9 / y = 1) : x + y = 25 :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_y_l757_75716


namespace NUMINAMATH_GPT_factorize_expr_l757_75706

theorem factorize_expr (a x y : ℝ) : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expr_l757_75706
