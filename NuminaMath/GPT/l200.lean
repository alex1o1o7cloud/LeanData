import Mathlib

namespace james_vegetable_intake_l200_200687

theorem james_vegetable_intake :
  let daily_asparagus := 0.25
  let daily_broccoli := 0.25
  let daily_intake := daily_asparagus + daily_broccoli
  let doubled_daily_intake := daily_intake * 2
  let weekly_intake_asparagus_broccoli := doubled_daily_intake * 7
  let weekly_kale := 3
  let total_weekly_intake := weekly_intake_asparagus_broccoli + weekly_kale
  total_weekly_intake = 10 := 
by
  sorry

end james_vegetable_intake_l200_200687


namespace paint_price_max_boxes_paint_A_l200_200535

theorem paint_price :
  ∃ x y : ℕ, (x + 2 * y = 56 ∧ 2 * x + y = 64) ∧ x = 24 ∧ y = 16 := by
  sorry

theorem max_boxes_paint_A (m : ℕ) :
  24 * m + 16 * (200 - m) ≤ 3920 → m ≤ 90 := by
  sorry

end paint_price_max_boxes_paint_A_l200_200535


namespace cone_generatrix_length_l200_200858

-- Define the conditions of the problem
def base_radius : ℝ := sqrt 2
def cone_base_circumference : ℝ := 2 * Real.pi * base_radius
def semicircle_arc_length (generatrix : ℝ) : ℝ := Real.pi * generatrix

-- The proof statement
theorem cone_generatrix_length : 
  ∃ (l : ℝ), 2 * Real.pi * sqrt 2 = Real.pi * l ∧ l = 2 * sqrt 2 :=
sorry

end cone_generatrix_length_l200_200858


namespace rowing_speed_in_still_water_l200_200029

theorem rowing_speed_in_still_water (v c : ℝ) (h1 : c = 1.5) 
(h2 : ∀ t : ℝ, (v + c) * t = (v - c) * 2 * t) : 
  v = 4.5 :=
by
  sorry

end rowing_speed_in_still_water_l200_200029


namespace maximum_numbers_no_divisible_difference_l200_200238

theorem maximum_numbers_no_divisible_difference :
  ∃ (S : Finset ℕ), S ⊆ Finset.range 236 ∧ 
  (∀ a b c ∈ S, a ≠ b → b ≠ c → c ≠ a → ¬ (a - b = 0) ∨ ¬ (c ∣ (a - b))) ∧ S.card ≤ 118 :=
by
  sorry

end maximum_numbers_no_divisible_difference_l200_200238


namespace range_of_a_l200_200480

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (-x)

theorem range_of_a (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) (h_ineq : f a (-2) > f a (-3)) : 0 < a ∧ a < 1 :=
by {
  sorry
}

end range_of_a_l200_200480


namespace coordinates_reflection_y_axis_l200_200393

theorem coordinates_reflection_y_axis :
  let M := (-5, 2) in
  reflect_y_axis M = (5, 2) :=
by
  sorry

end coordinates_reflection_y_axis_l200_200393


namespace p_eval_at_neg_one_l200_200512

noncomputable def p (x : ℝ) : ℝ :=
  x^2 - 2*x + 9

theorem p_eval_at_neg_one : p (-1) = 12 := by
  sorry

end p_eval_at_neg_one_l200_200512


namespace multimedia_sets_max_profit_l200_200154

-- Definitions of conditions:
def cost_A : ℝ := 3
def cost_B : ℝ := 2.4
def price_A : ℝ := 3.3
def price_B : ℝ := 2.8
def total_sets : ℕ := 50
def total_cost : ℝ := 132
def min_m : ℕ := 11

-- Problem 1: Prove the number of sets based on equations
theorem multimedia_sets (x y : ℕ) (h1 : x + y = total_sets) (h2 : cost_A * x + cost_B * y = total_cost) :
  x = 20 ∧ y = 30 :=
by sorry

-- Problem 2: Prove the maximum profit within a given range
theorem max_profit (m : ℕ) (h_m : 10 < m ∧ m < 20) :
  (-(0.1 : ℝ) * m + 20 = 18.9) ↔ m = min_m :=
by sorry

end multimedia_sets_max_profit_l200_200154


namespace quadratic_non_real_roots_iff_l200_200910

theorem quadratic_non_real_roots_iff (b : ℝ) : 
  (∀ x : ℂ, ¬ (x^2 + b*x + 16 = 0) → (b^2 - 4 * 1 * 16 < 0)) ↔ -8 < b ∧ b < 8 :=
by sorry

end quadratic_non_real_roots_iff_l200_200910


namespace rectangle_area_is_180_l200_200150

def area_of_square (side : ℕ) : ℕ := side * side
def length_of_rectangle (radius : ℕ) : ℕ := (2 * radius) / 5
def area_of_rectangle (length breadth : ℕ) : ℕ := length * breadth

theorem rectangle_area_is_180 :
  ∀ (side breadth : ℕ), 
    area_of_square side = 2025 → 
    breadth = 10 → 
    area_of_rectangle (length_of_rectangle side) breadth = 180 :=
by
  intros side breadth h_area h_breadth
  sorry

end rectangle_area_is_180_l200_200150


namespace find_b_l200_200477

theorem find_b (k a b : ℝ) (h1 : 1 + a + b = 3) (h2 : k = 3 + a) :
  b = 3 := 
sorry

end find_b_l200_200477


namespace calculate_sum_of_squares_l200_200823

theorem calculate_sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 268 := 
  sorry

end calculate_sum_of_squares_l200_200823


namespace rectangle_area_l200_200577

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end rectangle_area_l200_200577


namespace paint_price_max_boxes_paint_A_l200_200537

theorem paint_price :
  ∃ x y : ℕ, (x + 2 * y = 56 ∧ 2 * x + y = 64) ∧ x = 24 ∧ y = 16 := by
  sorry

theorem max_boxes_paint_A (m : ℕ) :
  24 * m + 16 * (200 - m) ≤ 3920 → m ≤ 90 := by
  sorry

end paint_price_max_boxes_paint_A_l200_200537


namespace carpet_area_l200_200239

/-- A rectangular floor with a length of 15 feet and a width of 12 feet needs 20 square yards of carpet to cover it. -/
theorem carpet_area (length_feet : ℕ) (width_feet : ℕ) (feet_per_yard : ℕ) (length_yards : ℕ) (width_yards : ℕ) (area_sq_yards : ℕ) :
  length_feet = 15 ∧
  width_feet = 12 ∧
  feet_per_yard = 3 ∧
  length_yards = length_feet / feet_per_yard ∧
  width_yards = width_feet / feet_per_yard ∧
  area_sq_yards = length_yards * width_yards → 
  area_sq_yards = 20 :=
by
  sorry

end carpet_area_l200_200239


namespace smallest_n_good_sequence_2014_l200_200809

-- Define the concept of a "good sequence"
def good_sequence (a : ℕ → ℝ) : Prop :=
  a 0 > 0 ∧
  ∀ i, a (i + 1) = 2 * a i + 1 ∨ a (i + 1) = a i / (a i + 2)

-- Define the smallest n such that a good sequence reaches 2014 at a_n
theorem smallest_n_good_sequence_2014 :
  ∃ (n : ℕ), (∀ a, good_sequence a → a n = 2014) ∧
  ∀ (m : ℕ), m < n → ∀ a, good_sequence a → a m ≠ 2014 :=
sorry

end smallest_n_good_sequence_2014_l200_200809


namespace rectangle_area_l200_200573

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end rectangle_area_l200_200573


namespace rectangle_area_l200_200560

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l200_200560


namespace congruence_a_b_mod_1008_l200_200717

theorem congruence_a_b_mod_1008
  (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : a ^ b - b ^ a = 1008) : a ≡ b [MOD 1008] :=
sorry

end congruence_a_b_mod_1008_l200_200717


namespace joaozinho_card_mariazinha_card_pedrinho_error_l200_200422

-- Define the card transformation function
def transform_card (number : ℕ) (color_adjustment : ℕ) : ℕ :=
  (number * 2 + 3) * 5 + color_adjustment

-- The proof problems
theorem joaozinho_card : transform_card 3 4 = 49 :=
by
  sorry

theorem mariazinha_card : ∃ number, ∃ color_adjustment, transform_card number color_adjustment = 76 :=
by
  sorry

theorem pedrinho_error : ∀ number color_adjustment, ¬ transform_card number color_adjustment = 61 :=
by
  sorry

end joaozinho_card_mariazinha_card_pedrinho_error_l200_200422


namespace total_cans_collected_l200_200722

theorem total_cans_collected (total_students : ℕ) (half_students_collecting_12 : ℕ) 
 (students_collecting_0 : ℕ) (remaining_students_collecting_4 : ℕ) 
 (cans_collected_by_half : ℕ) (cans_collected_by_remaining : ℕ) :
 total_students = 30 →
 half_students_collecting_12 = 15 →
 students_collecting_0 = 2 →
 remaining_students_collecting_4 = 13 →
 cans_collected_by_half = half_students_collecting_12 * 12 →
 cans_collected_by_remaining = remaining_students_collecting_4 * 4 →
 (cans_collected_by_half + students_collecting_0 * 0 + cans_collected_by_remaining) = 232 :=
 by
  intros h1 h2 h3 h4 h5 h6
  obtain rfl : 15 * 12 = 180 := rfl
  obtain rfl : 13 * 4 = 52 := rfl
  rw [h5, h6]
  simp
  sorry

end total_cans_collected_l200_200722


namespace abs_neg_2023_l200_200151

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l200_200151


namespace frank_original_money_l200_200468

theorem frank_original_money (X : ℝ) :
  (X - (1 / 5) * X - (1 / 4) * (X - (1 / 5) * X) = 360) → (X = 600) :=
by
  sorry

end frank_original_money_l200_200468


namespace pet_store_problem_l200_200286

noncomputable def num_ways_to_buy_pets (puppies kittens hamsters birds : ℕ) (people : ℕ) : ℕ :=
  (puppies * kittens * hamsters * birds) * (people.factorial)

theorem pet_store_problem :
  num_ways_to_buy_pets 12 10 5 3 4 = 43200 :=
by
  sorry

end pet_store_problem_l200_200286


namespace lily_received_books_l200_200100

def mike_books : ℕ := 45
def corey_books : ℕ := 2 * mike_books
def mike_gave_lily : ℕ := 10
def corey_gave_lily : ℕ := mike_gave_lily + 15
def lily_books_received : ℕ := mike_gave_lily + corey_gave_lily

theorem lily_received_books : lily_books_received = 35 := by
  sorry

end lily_received_books_l200_200100


namespace original_number_is_115_l200_200273

-- Define the original number N, the least number to be subtracted (given), and the divisor
variable (N : ℤ) (k : ℤ)

-- State the condition based on the problem's requirements
def least_number_condition := ∃ k : ℤ, N - 28 = 87 * k

-- State the proof problem: Given the condition, prove the original number
theorem original_number_is_115 (h : least_number_condition N) : N = 115 := 
by
  sorry

end original_number_is_115_l200_200273


namespace football_match_even_goals_l200_200796

noncomputable def poisson_even_probability (λ : ℝ) : ℝ :=
  (1 + Real.exp (-2 * λ)) / 2

theorem football_match_even_goals :
  poisson_even_probability 2.8 ≈ 0.502 :=
by
  -- Proof skipped
  sorry

end football_match_even_goals_l200_200796


namespace both_reunions_l200_200421

theorem both_reunions (U O H B : ℕ) 
  (hU : U = 100) 
  (hO : O = 50) 
  (hH : H = 62) 
  (attend_one : U = O + H - B) :  
  B = 12 := 
by 
  sorry

end both_reunions_l200_200421


namespace frank_money_l200_200470

theorem frank_money (X : ℝ) (h1 : (3/4) * (4/5) * X = 360) : X = 600 :=
sorry

end frank_money_l200_200470


namespace coordinates_A_B_l200_200949

theorem coordinates_A_B : 
  (∃ x, 7 * x + 2 * 3 = 41) ∧ (∃ y, 7 * (-5) + 2 * y = 41) → 
  ((∃ x, x = 5) ∧ (∃ y, y = 38)) :=
by
  sorry

end coordinates_A_B_l200_200949


namespace triangle_inequality_l200_200673

theorem triangle_inequality (a b c : ℝ) (h : a < b + c) : a^2 - b^2 - c^2 - 2*b*c < 0 := by
  sorry

end triangle_inequality_l200_200673


namespace find_m_n_l200_200657

-- Define the vectors OA, OB, OC
def vector_oa (m : ℝ) : ℝ × ℝ := (-2, m)
def vector_ob (n : ℝ) : ℝ × ℝ := (n, 1)
def vector_oc : ℝ × ℝ := (5, -1)

-- Define the condition that OA is perpendicular to OB
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Define the condition that points A, B, and C are collinear.
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (A.1 - B.1) * (C.2 - A.2) = k * ((C.1 - A.1) * (A.2 - B.2))

theorem find_m_n (m n : ℝ) :
  collinear (-2, m) (n, 1) (5, -1) ∧ perpendicular (-2, m) (n, 1) → m = 3 ∧ n = 3 / 2 := by
  intro h
  sorry

end find_m_n_l200_200657


namespace fewer_females_than_males_l200_200082

theorem fewer_females_than_males 
  (total_students : ℕ)
  (female_students : ℕ)
  (h_total : total_students = 280)
  (h_female : female_students = 127) :
  total_students - female_students - female_students = 26 := by
  sorry

end fewer_females_than_males_l200_200082


namespace gmat_test_takers_correctly_l200_200777

variable (A B : ℝ)
variable (intersection union : ℝ)

theorem gmat_test_takers_correctly :
  B = 0.8 ∧ intersection = 0.7 ∧ union = 0.95 → A = 0.85 :=
by 
  sorry

end gmat_test_takers_correctly_l200_200777


namespace max_profit_l200_200581

theorem max_profit (m : ℝ) :
  (m - 8) * (900 - 15 * m) = -15 * (m - 34) ^ 2 + 10140 :=
by
  sorry

end max_profit_l200_200581


namespace bisection_method_root_interval_l200_200973

noncomputable def f (x : ℝ) : ℝ := 2^x + 3 * x - 7

theorem bisection_method_root_interval :
  f 1 < 0 → f 2 > 0 → f 3 > 0 → ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 :=
by
  sorry

end bisection_method_root_interval_l200_200973


namespace cars_travel_same_distance_l200_200585

-- Define all the variables and conditions
def TimeR : ℝ := sorry -- the time taken by car R
def TimeP : ℝ := TimeR - 2
def SpeedR : ℝ := 58.4428877022476
def SpeedP : ℝ := SpeedR + 10

-- state the distance travelled by both cars
def DistanceR : ℝ := SpeedR * TimeR
def DistanceP : ℝ := SpeedP * TimeP

-- Prove that both distances are the same and equal to 800
theorem cars_travel_same_distance : DistanceR = 800 := by
  sorry

end cars_travel_same_distance_l200_200585


namespace gcf_of_lcm_9_21_and_10_22_eq_one_l200_200059

theorem gcf_of_lcm_9_21_and_10_22_eq_one :
  Nat.gcd (Nat.lcm 9 21) (Nat.lcm 10 22) = 1 :=
sorry

end gcf_of_lcm_9_21_and_10_22_eq_one_l200_200059


namespace complement_A_union_B_in_U_l200_200483

-- Define the universe set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | (x - 2) * (x + 1) ≤ 0}

-- Define set B
def B : Set ℝ := {x | 0 ≤ x ∧ x < 3}

-- Define the union of A and B
def A_union_B : Set ℝ := {x | (-1 ≤ x ∧ x < 3)}

-- Define the complement of A ∪ B in U
def C_U_A_union_B : Set ℝ := {x | x < -1 ∨ x ≥ 3}

-- Proof Statement
theorem complement_A_union_B_in_U :
  {x | x < -1 ∨ x ≥ 3} = {x | x ∈ U ∧ (x ∉ A_union_B)} :=
sorry

end complement_A_union_B_in_U_l200_200483


namespace max_profit_at_one_device_l200_200427

noncomputable def revenue (x : ℕ) : ℝ := 30 * x - 0.2 * x^2

def fixed_monthly_cost : ℝ := 40

def material_cost_per_device : ℝ := 5

noncomputable def cost (x : ℕ) : ℝ := fixed_monthly_cost + material_cost_per_device * x

noncomputable def profit_function (x : ℕ) : ℝ := (revenue x) - (cost x)

noncomputable def marginal_profit_function (x : ℕ) : ℝ :=
  profit_function (x + 1) - profit_function x

theorem max_profit_at_one_device :
  marginal_profit_function 1 = 24.4 ∧
  ∀ x : ℕ, marginal_profit_function x ≤ 24.4 := sorry

end max_profit_at_one_device_l200_200427


namespace range_of_m_l200_200379

noncomputable def quadratic_expr_never_equal (m : ℝ) : Prop :=
  ∀ (x : ℝ), 2 * x^2 + 4 * x + m ≠ 3 * x^2 - 2 * x + 6

theorem range_of_m (m : ℝ) : quadratic_expr_never_equal m ↔ m < -3 := 
by
  sorry

end range_of_m_l200_200379


namespace cube_root_numbers_less_than_15_l200_200885

/-- The number of positive whole numbers that have cube roots less than 15 is 3375. -/
theorem cube_root_numbers_less_than_15 : 
  { n : ℕ // n > 0 ∧ n < 15^3 } = 3375 :=
begin
  sorry
end

end cube_root_numbers_less_than_15_l200_200885


namespace proof_problem_l200_200020

-- Define the probability of green light and red light
def p_green : ℚ := 3 / 4
def p_red : ℚ := 1 / 4

-- Define the probability distribution
def prob_distribution (k : ℕ) : ℚ :=
  if k = 0 then p_red
  else if k = 1 then p_red * p_green
  else if k = 2 then p_green * p_green * p_red
  else if k = 3 then p_green * p_green * p_green * p_red
  else if k = 4 then p_green ^ 4
  else 0

-- Define the expected value
noncomputable def E_xi : ℚ := 
  0 * prob_distribution 0 +
  1 * prob_distribution 1 +
  2 * prob_distribution 2 +
  3 * prob_distribution 3 +
  4 * prob_distribution 4

-- Define the probability that the car stops at most after 3 intersections
noncomputable def P_xi_le_3 : ℚ :=
  prob_distribution 0 +
  prob_distribution 1 +
  prob_distribution 2 +
  prob_distribution 3

-- Putting it all together into Lean theorem statement
theorem proof_problem : 
  E_xi = 525 / 256 ∧
  P_xi_le_3 = 175 / 256 :=
by 
  sorry

end proof_problem_l200_200020


namespace cost_of_paving_is_correct_l200_200015

def length_of_room : ℝ := 5.5
def width_of_room : ℝ := 4
def rate_per_square_meter : ℝ := 950
def area_of_room : ℝ := length_of_room * width_of_room
def cost_of_paving : ℝ := area_of_room * rate_per_square_meter

theorem cost_of_paving_is_correct : cost_of_paving = 20900 := 
by
  sorry

end cost_of_paving_is_correct_l200_200015


namespace gcd_888_1147_l200_200999

/-- Use the Euclidean algorithm to find the greatest common divisor (GCD) of 888 and 1147. -/
theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end gcd_888_1147_l200_200999


namespace positive_whole_numbers_with_cube_roots_less_than_15_l200_200886

theorem positive_whole_numbers_with_cube_roots_less_than_15 :
  ∃ n : ℕ, n = 3374 ∧ ∀ x : ℕ, (x > 0 ∧ x < 3375) → x <= n :=
by sorry

end positive_whole_numbers_with_cube_roots_less_than_15_l200_200886


namespace minimum_common_correct_questions_l200_200493

open Finset

-- Define the students
inductive Student
| Xiaoxi | Xiaofei | Xianguan | Xialan
deriving DecidableEq

-- Define the problem set as a finset of questions
def questions : Finset ℕ := (range 10).toFinset

-- Define the set of questions each student answered correctly
variable (Xiaoxi_correct Xiaofei_correct Xianguan_correct Xialan_correct : Finset ℕ)

-- Define the conditions
axiom condition1 : Xiaoxi_correct.card = 8
axiom condition2 : Xiaofei_correct.card = 8
axiom condition3 : Xianguan_correct.card = 8
axiom condition4 : Xialan_correct.card = 8

-- Define the Lean statement to be proven
theorem minimum_common_correct_questions :
  ∃ common_questions : Finset ℕ, common_questions.card ≥ 2 ∧
    common_questions ⊆ Xiaoxi_correct ∧ 
    common_questions ⊆ Xiaofei_correct ∧ 
    common_questions ⊆ Xianguan_correct ∧ 
    common_questions ⊆ Xialan_correct :=
sorry

end minimum_common_correct_questions_l200_200493


namespace range_of_square_of_difference_of_roots_l200_200071

theorem range_of_square_of_difference_of_roots (a : ℝ) (h : (a - 1) * (a - 2) < 0) :
  ∃ (S : Set ℝ), S = { x | 0 < x ∧ x ≤ 1 } ∧ ∀ (x1 x2 : ℝ),
  x1 + x2 = 2 * a ∧ x1 * x2 = 2 * a^2 - 3 * a + 2 → (x1 - x2)^2 ∈ S :=
sorry

end range_of_square_of_difference_of_roots_l200_200071


namespace find_height_of_door_l200_200926

noncomputable def height_of_door (x : ℝ) (w : ℝ) (h : ℝ) : ℝ := h

theorem find_height_of_door :
  ∃ x w h, (w = x - 4) ∧ (h = x - 2) ∧ (x^2 = w^2 + h^2) ∧ height_of_door x w h = 8 :=
by {
  sorry
}

end find_height_of_door_l200_200926


namespace d_minus_b_equals_757_l200_200515

theorem d_minus_b_equals_757 
  (a b c d : ℕ) 
  (h1 : a^5 = b^4) 
  (h2 : c^3 = d^2) 
  (h3 : c - a = 19) : 
  d - b = 757 := 
by 
  sorry

end d_minus_b_equals_757_l200_200515


namespace maximum_value_of_n_with_positive_sequence_l200_200338

theorem maximum_value_of_n_with_positive_sequence (a : ℕ → ℝ) (h_seq : ∀ n : ℕ, 0 < a n) 
    (h_arithmetic : ∀ n : ℕ, a (n + 1)^2 - a n^2 = 1) : ∃ n : ℕ, n = 24 ∧ a n < 5 :=
by
  sorry

end maximum_value_of_n_with_positive_sequence_l200_200338


namespace find_g_l200_200961

noncomputable def g : ℝ → ℝ
| x => 2 * (4^x - 3^x)

theorem find_g :
  (g 1 = 2) ∧
  (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y) →
  ∀ x : ℝ, g x = 2 * (4^x - 3^x) := by
  sorry

end find_g_l200_200961


namespace sqrt_expression_equal_cos_half_theta_l200_200347

noncomputable def sqrt_half_plus_sqrt_half_cos2theta_minus_sqrt_one_minus_sintheta (θ : Real) : Real :=
  Real.sqrt (1 / 2 + 1 / 2 * Real.sqrt (1 / 2 + 1 / 2 * Real.cos (2 * θ))) - Real.sqrt (1 - Real.sin θ)

theorem sqrt_expression_equal_cos_half_theta (θ : Real) (h : π < θ) (h2 : θ < 3 * π / 2)
  (h3 : Real.cos θ < 0) (h4 : 0 < Real.sin (θ / 2)) (h5 : Real.cos (θ / 2) < 0) :
  sqrt_half_plus_sqrt_half_cos2theta_minus_sqrt_one_minus_sintheta θ = Real.cos (θ / 2) :=
by
  sorry

end sqrt_expression_equal_cos_half_theta_l200_200347


namespace height_of_flagpole_l200_200748

theorem height_of_flagpole 
  (house_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ) 
  (flagpole_shadow : ℝ) (house_height : ℝ)
  (h1 : house_shadow = 70)
  (h2 : tree_height = 28)
  (h3 : tree_shadow = 40)
  (h4 : flagpole_shadow = 25)
  (h5 : house_height = (tree_height * house_shadow) / tree_shadow) :
  round ((house_height * flagpole_shadow / house_shadow) : ℝ) = 18 := 
by
  sorry

end height_of_flagpole_l200_200748


namespace area_of_set_K_l200_200360

open Metric

def set_K :=
  {p : ℝ × ℝ | (abs p.1 + abs (3 * p.2) - 6) * (abs (3 * p.1) + abs p.2 - 6) ≤ 0}

def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry -- Define the area function for a general set s

theorem area_of_set_K : area set_K = 24 :=
  sorry

end area_of_set_K_l200_200360


namespace dilute_lotion_l200_200053

/-- Determine the number of ounces of water needed to dilute 12 ounces
    of a shaving lotion containing 60% alcohol to a lotion containing 45% alcohol. -/
theorem dilute_lotion (W : ℝ) : 
  ∃ W, 12 * (0.60 : ℝ) / (12 + W) = 0.45 ∧ W = 4 :=
by
  use 4
  sorry

end dilute_lotion_l200_200053


namespace quadratic_non_real_roots_l200_200900

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 16 = 0) ↔ b ∈ set.Ioo (-8 : ℝ) 8 :=
by
  sorry

end quadratic_non_real_roots_l200_200900


namespace unattainable_y_l200_200199

theorem unattainable_y (x : ℝ) (h : x ≠ -5/4) : ¬∃ y : ℝ, y = (2 - 3 * x) / (4 * x + 5) ∧ y = -3 / 4 :=
by
  sorry

end unattainable_y_l200_200199


namespace proof_q_is_true_l200_200490

variable (p q : Prop)

-- Assuming the conditions
axiom h1 : p ∨ q   -- p or q is true
axiom h2 : ¬ p     -- not p is true

-- Theorem statement to prove q is true
theorem proof_q_is_true : q :=
by
  sorry

end proof_q_is_true_l200_200490


namespace abs_quadratic_bound_l200_200655

theorem abs_quadratic_bound (a b : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + a * x + b) :
  (|f 1| ≥ (1 / 2)) ∨ (|f 2| ≥ (1 / 2)) ∨ (|f 3| ≥ (1 / 2)) :=
by
  sorry

end abs_quadratic_bound_l200_200655


namespace problem_statement_l200_200095

theorem problem_statement (n : ℕ) (hn : 0 < n) 
  (h : (1/2 : ℚ) + 1/3 + 1/7 + 1/n ∈ ℤ) : n = 42 ∧ ¬ (n > 84) := 
by
  sorry

end problem_statement_l200_200095


namespace odd_function_f_neg_9_l200_200341

noncomputable def f (x : ℝ) : ℝ := 
if x > 0 then x^(1/2) 
else -((-x)^(1/2))

theorem odd_function_f_neg_9 : f (-9) = -3 := by
  sorry

end odd_function_f_neg_9_l200_200341


namespace intersection_A_B_union_B_complement_A_l200_200882

open Set

variable (U A B : Set ℝ)

noncomputable def U_def : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}
noncomputable def A_def : Set ℝ := {x | 0 < x ∧ x ≤ 3}
noncomputable def B_def : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem intersection_A_B : (A ∩ B) = {x | 0 < x ∧ x ≤ 1} :=
sorry

theorem union_B_complement_A : B ∪ (U \ A) = {x | (-5 ≤ x ∧ x ≤ 1) ∨ (3 < x ∧ x ≤ 5)} :=
sorry

attribute [irreducible] U_def A_def B_def

end intersection_A_B_union_B_complement_A_l200_200882


namespace number_of_negative_x_l200_200647

theorem number_of_negative_x (n : ℤ) (hn : 1 ≤ n ∧ n * n < 200) : 
  ∃ m ≥ 1, m = 14 := sorry

end number_of_negative_x_l200_200647


namespace generatrix_length_l200_200876

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end generatrix_length_l200_200876


namespace function_relationship_value_of_x_when_y_is_1_l200_200852

variable (x y : ℝ) (k : ℝ)

-- Conditions
def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, y = k / (x - 3)

axiom condition_1 : inverse_proportion x y
axiom condition_2 : y = 5 ∧ x = 4

-- Statements to be proved
theorem function_relationship :
  ∃ k : ℝ, (y = k / (x - 3)) ∧ (y = 5 ∧ x = 4 → k = 5) :=
by
  sorry

theorem value_of_x_when_y_is_1 (hy : y = 1) :
  ∃ x : ℝ, (y = 5 / (x - 3)) ∧ x = 8 :=
by
  sorry

end function_relationship_value_of_x_when_y_is_1_l200_200852


namespace debby_bought_bottles_l200_200832

def bottles_per_day : ℕ := 109
def days_lasting : ℕ := 74

theorem debby_bought_bottles : bottles_per_day * days_lasting = 8066 := by
  sorry

end debby_bought_bottles_l200_200832


namespace kernels_needed_for_movie_night_l200_200505

structure PopcornPreferences where
  caramel_popcorn: ℝ
  butter_popcorn: ℝ
  cheese_popcorn: ℝ
  kettle_corn_popcorn: ℝ

noncomputable def total_kernels_needed (preferences: PopcornPreferences) : ℝ :=
  (preferences.caramel_popcorn / 6) * 3 +
  (preferences.butter_popcorn / 4) * 2 +
  (preferences.cheese_popcorn / 8) * 4 +
  (preferences.kettle_corn_popcorn / 3) * 1

theorem kernels_needed_for_movie_night :
  let preferences := PopcornPreferences.mk 3 4 6 3
  total_kernels_needed preferences = 7.5 :=
sorry

end kernels_needed_for_movie_night_l200_200505


namespace solution_set_of_absolute_value_inequality_l200_200004

theorem solution_set_of_absolute_value_inequality :
  { x : ℝ | |x + 1| - |x - 2| > 1 } = { x : ℝ | 1 < x } :=
by 
  sorry

end solution_set_of_absolute_value_inequality_l200_200004


namespace find_interest_rate_of_initial_investment_l200_200625

def initial_investment : ℝ := 1400
def additional_investment : ℝ := 700
def total_investment : ℝ := 2100
def additional_interest_rate : ℝ := 0.08
def target_total_income_rate : ℝ := 0.06
def target_total_income : ℝ := target_total_income_rate * total_investment

theorem find_interest_rate_of_initial_investment (r : ℝ) :
  (initial_investment * r + additional_investment * additional_interest_rate = target_total_income) → 
  (r = 0.05) :=
by
  sorry

end find_interest_rate_of_initial_investment_l200_200625


namespace intersection_A_B_union_A_complement_B_l200_200667

open Set

noncomputable def U : Set ℝ := Univ

def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}

def B : Set ℝ := {x | (x^2 - 2 * x - 3) ≥ 0}

def complement_B : Set ℝ := U \ B

-- Proving A ∩ B = {x | -5 < x ≤ -1}
theorem intersection_A_B :
  { x | -5 < x ∧ x ≤ -1 } = A ∩ B := by
  sorry

-- Proving A ∪ (complement_U B) = {x | -5 < x < 3}
theorem union_A_complement_B :
  { x | -5 < x ∧ x < 3 } = A ∪ complement_B := by
  sorry

end intersection_A_B_union_A_complement_B_l200_200667


namespace ray_reflection_and_distance_l200_200605

-- Define the initial conditions
def pointA : ℝ × ℝ := (-3, 3)
def circleC_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Definitions of the lines for incident and reflected rays
def incident_ray_line (x y : ℝ) : Prop := 4*x + 3*y + 3 = 0
def reflected_ray_line (x y : ℝ) : Prop := 3*x + 4*y - 3 = 0

-- Distance traveled by the ray
def distance_traveled (A T : ℝ × ℝ) := 7

theorem ray_reflection_and_distance :
  ∃ (x₁ y₁ : ℝ), incident_ray_line x₁ y₁ ∧ reflected_ray_line x₁ y₁ ∧ circleC_eq x₁ y₁ ∧ 
  (∀ (P : ℝ × ℝ), P = pointA → distance_traveled P (x₁, y₁) = 7) :=
sorry

end ray_reflection_and_distance_l200_200605


namespace problem_1_problem_2_l200_200663

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x - 3|

theorem problem_1 (a x : ℝ) (h1 : a < 3) (h2 : (∀ x, f x a >= 4 ↔ x ≤ 1/2 ∨ x ≥ 9/2)) : 
  a = 2 :=
sorry

theorem problem_2 (a : ℝ) (h1 : ∀ x : ℝ, f x a + |x - 3| ≥ 1) : 
  a ≤ 2 :=
sorry

end problem_1_problem_2_l200_200663


namespace roots_of_third_quadratic_l200_200070

/-- Given two quadratic equations with exactly one common root and a non-equal coefficient condition, 
prove that the other roots are roots of a third quadratic equation -/
theorem roots_of_third_quadratic 
  (a1 a2 a3 α β γ : ℝ)
  (h1 : α ≠ β)
  (h2 : α ≠ γ)
  (h3 : a1 ≠ a2)
  (h_eq1 : α^2 + a1*α + a2*a3 = 0)
  (h_eq2 : β^2 + a1*β + a2*a3 = 0)
  (h_eq3 : α^2 + a2*α + a1*a3 = 0)
  (h_eq4 : γ^2 + a2*γ + a1*a3 = 0) :
  β^2 + a3*β + a1*a2 = 0 ∧ γ^2 + a3*γ + a1*a2 = 0 :=
by
  sorry

end roots_of_third_quadratic_l200_200070


namespace number_of_factors_multiples_of_360_l200_200335

def n : ℕ := 2^10 * 3^14 * 5^8

theorem number_of_factors_multiples_of_360 (n : ℕ) (hn : n = 2^10 * 3^14 * 5^8) : 
  ∃ (k : ℕ), k = 832 ∧ 
  (∀ m : ℕ, m ∣ n → 360 ∣ m → k = 8 * 13 * 8) := 
sorry

end number_of_factors_multiples_of_360_l200_200335


namespace rectangle_area_l200_200558

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l200_200558


namespace sequence_sum_l200_200997

theorem sequence_sum : (1 - 3 + 5 - 7 + 9 - 11 + 13 - 15 + 17 - 19) = -10 :=
by
  sorry

end sequence_sum_l200_200997


namespace tan_sin_identity_l200_200310

theorem tan_sin_identity :
  (let θ := 30 * Real.pi / 180 in 
   let tan_θ := Real.sin θ / Real.cos θ in 
   let sin_θ := 1 / 2 in
   let cos_θ := Real.sqrt 3 / 2 in
   ((tan_θ ^ 2 - sin_θ ^ 2) / (tan_θ ^ 2 * sin_θ ^ 2) = 1)) :=
by
  let θ := 30 * Real.pi / 180
  let tan_θ := Real.sin θ / Real.cos θ
  let sin_θ := 1 / 2
  let cos_θ := Real.sqrt 3 / 2
  have h_tan_θ : tan_θ = 1 / Real.sqrt 3 := sorry  -- We skip the derivation proof.
  have h_sin_θ : sin_θ = 1 / 2 := rfl
  have h_cos_θ : cos_θ = Real.sqrt 3 / 2 := sorry  -- We skip the derivation proof.
  have numerator := tan_θ ^ 2 - sin_θ ^ 2
  have denominator := tan_θ ^ 2 * sin_θ ^ 2
  -- show equality
  have h : (numerator / denominator) = (1 : ℝ) := sorry  -- We skip the final proof.
  exact h

end tan_sin_identity_l200_200310


namespace ab_sum_l200_200189

theorem ab_sum (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 7) (h3 : |a - b| = b - a) : a + b = 10 ∨ a + b = 4 :=
by
  sorry

end ab_sum_l200_200189


namespace simplify_expression_l200_200146

theorem simplify_expression : 3000 * (3000 ^ 3000) + 3000 * (3000 ^ 3000) = 2 * 3000 ^ 3001 := 
by 
  sorry

end simplify_expression_l200_200146


namespace lamps_on_bridge_l200_200424

theorem lamps_on_bridge (bridge_length : ℕ) (lamp_spacing : ℕ) (num_intervals : ℕ) (num_lamps : ℕ) 
  (h1 : bridge_length = 30) 
  (h2 : lamp_spacing = 5)
  (h3 : num_intervals = bridge_length / lamp_spacing)
  (h4 : num_lamps = num_intervals + 1) :
  num_lamps = 7 := 
by
  sorry

end lamps_on_bridge_l200_200424


namespace smallest_x_for_multiple_of_720_l200_200754

theorem smallest_x_for_multiple_of_720 (x : ℕ) (h1 : 450 = 2^1 * 3^2 * 5^2) (h2 : 720 = 2^4 * 3^2 * 5^1) : x = 8 ↔ (450 * x) % 720 = 0 :=
by
  sorry

end smallest_x_for_multiple_of_720_l200_200754


namespace football_match_even_goals_l200_200794

noncomputable def poisson_even_probability (λ : ℝ) : ℝ :=
  (1 + Real.exp (-2 * λ)) / 2

theorem football_match_even_goals :
  poisson_even_probability 2.8 ≈ 0.502 :=
by
  -- Proof skipped
  sorry

end football_match_even_goals_l200_200794


namespace range_of_g_l200_200166

noncomputable def g (x : ℝ) : ℝ := (Real.cos x)^4 + (Real.sin x)^2

theorem range_of_g : Set.Icc (3 / 4) 1 = Set.range g :=
by
  sorry

end range_of_g_l200_200166


namespace num_pos_nums_with_cube_root_lt_15_l200_200890

theorem num_pos_nums_with_cube_root_lt_15 : 
  ∃ (N : ℕ), (∀ n : ℕ, 1 ≤ n ∧ n ≤ 3374 → ∛(n : ℝ) < 15) ∧ N = 3374 := 
sorry

end num_pos_nums_with_cube_root_lt_15_l200_200890


namespace cost_per_text_message_for_first_plan_l200_200415

theorem cost_per_text_message_for_first_plan (x : ℝ) : 
  (9 + 60 * x = 60 * 0.40) → (x = 0.25) :=
by
  intro h
  sorry

end cost_per_text_message_for_first_plan_l200_200415


namespace billy_sleep_total_l200_200208

def billy_sleep : Prop :=
  let first_night := 6
  let second_night := first_night + 2
  let third_night := second_night / 2
  let fourth_night := third_night * 3
  first_night + second_night + third_night + fourth_night = 30

theorem billy_sleep_total : billy_sleep := by
  sorry

end billy_sleep_total_l200_200208


namespace letter_arrangements_l200_200203

theorem letter_arrangements :
  (∑ j in Finset.range 5, ∑ m in Finset.range 6, 
  Nat.choose 5 j * Nat.choose 5 m * Nat.multichoose 5 [4 - j, 6 - 5 + j - m, m]) 
  = -- Required expression
  sorry

end letter_arrangements_l200_200203


namespace additive_inverse_of_half_l200_200120

theorem additive_inverse_of_half :
  - (1 / 2) = -1 / 2 :=
by
  sorry

end additive_inverse_of_half_l200_200120


namespace find_a33_in_arithmetic_sequence_grid_l200_200297

theorem find_a33_in_arithmetic_sequence_grid 
  (matrix : ℕ → ℕ → ℕ)
  (rows_are_arithmetic : ∀ i, ∃ a b, ∀ j, matrix i j = a + b * (j - 1))
  (columns_are_arithmetic : ∀ j, ∃ c d, ∀ i, matrix i j = c + d * (i - 1))
  : matrix 3 3 = 31 :=
sorry

end find_a33_in_arithmetic_sequence_grid_l200_200297


namespace symmetric_line_equation_l200_200116

theorem symmetric_line_equation (x y : ℝ) (h : 4 * x - 3 * y + 5 = 0):
  4 * x + 3 * y + 5 = 0 :=
sorry

end symmetric_line_equation_l200_200116


namespace football_even_goal_probability_l200_200797

noncomputable def poisson_even_goal_probability : ℝ :=
  let λ := 2.8 in
  (1 + Real.exp (-2 * λ)) / 2

theorem football_even_goal_probability :
  let λ := 2.8 in
  let N := Poisson λ in  -- Define N as a Poisson random variable with parameter λ
  (∑ k : ℕ in (range (2*k)).filter (λ k, N.P k > 0), N.P k) = 
  poisson_even_goal_probability :=
by
  sorry

end football_even_goal_probability_l200_200797


namespace P_finishes_in_15_minutes_more_l200_200016

variable (P Q : ℝ)

def rate_p := 1 / 4
def rate_q := 1 / 15
def time_together := 3
def total_job := 1

theorem P_finishes_in_15_minutes_more :
  let combined_rate := rate_p + rate_q
  let completed_job_in_3_hours := combined_rate * time_together
  let remaining_job := total_job - completed_job_in_3_hours
  let time_for_P_to_finish := remaining_job / rate_p
  let minutes_needed := time_for_P_to_finish * 60
  minutes_needed = 15 :=
by
  -- Proof steps go here
  sorry

end P_finishes_in_15_minutes_more_l200_200016


namespace legally_drive_after_hours_l200_200248

theorem legally_drive_after_hours (n : ℕ) :
  (∀ t ≥ n, 0.8 * (0.5 : ℝ) ^ t ≤ 0.2) ↔ n = 2 :=
by
  sorry

end legally_drive_after_hours_l200_200248


namespace melissa_earnings_from_sales_l200_200942

noncomputable def commission_earned (coupe_price suv_price commission_rate : ℕ) : ℕ :=
  (coupe_price + suv_price) * commission_rate / 100

theorem melissa_earnings_from_sales : 
  commission_earned 30000 60000 2 = 1800 :=
by
  sorry

end melissa_earnings_from_sales_l200_200942


namespace bucket_capacity_l200_200600

-- Given Conditions
variable (C : ℝ)
variable (h : (2 / 3) * C = 9)

-- Goal
theorem bucket_capacity : C = 13.5 := by
  sorry

end bucket_capacity_l200_200600


namespace reading_schedule_correct_l200_200220

-- Defining the conditions
def total_words : ℕ := 34685
def words_day1 (x : ℕ) : ℕ := x
def words_day2 (x : ℕ) : ℕ := 2 * x
def words_day3 (x : ℕ) : ℕ := 4 * x

-- Defining the main statement of the problem
theorem reading_schedule_correct (x : ℕ) : 
  words_day1 x + words_day2 x + words_day3 x = total_words := 
sorry

end reading_schedule_correct_l200_200220


namespace ones_digit_of_7_pow_53_l200_200591

theorem ones_digit_of_7_pow_53 : (7^53 % 10) = 7 := by
  sorry

end ones_digit_of_7_pow_53_l200_200591


namespace subset_proof_l200_200940

-- Define set M
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (x^2 + 2*x + 1)}

-- The problem statement
theorem subset_proof : M ⊆ N ∧ ∃ y ∈ N, y ∉ M :=
by
  sorry

end subset_proof_l200_200940


namespace find_speed_of_first_train_l200_200747

noncomputable def relative_speed (length1 length2 : ℕ) (time_seconds : ℝ) : ℝ :=
  let total_length_km := (length1 + length2) / 1000
  let time_hours := time_seconds / 3600
  total_length_km / time_hours

theorem find_speed_of_first_train
  (length1 : ℕ)   -- Length of the first train in meters
  (length2 : ℕ)   -- Length of the second train in meters
  (speed2 : ℝ)    -- Speed of the second train in km/h
  (time_seconds : ℝ)  -- Time in seconds to be clear from each other
  (correct_speed1 : ℝ)  -- Correct speed of the first train in km/h
  (h_length1 : length1 = 160)
  (h_length2 : length2 = 280)
  (h_speed2 : speed2 = 30)
  (h_time_seconds : time_seconds = 21.998240140788738)
  (h_correct_speed1 : correct_speed1 = 41.98) :
  relative_speed length1 length2 time_seconds = speed2 + correct_speed1 :=
by
  sorry

end find_speed_of_first_train_l200_200747


namespace trigonometric_identity_example_l200_200311

theorem trigonometric_identity_example :
  (let θ := 30 / 180 * Real.pi in
   (Real.tan θ)^2 - (Real.sin θ)^2) / ((Real.tan θ)^2 * (Real.sin θ)^2) = 4 / 3 :=
by
  let θ := 30 / 180 * Real.pi
  have h_tan2 := Real.tan θ * Real.tan θ
  have h_sin2 : Real.sin θ * Real.sin θ = 1/4 := by sorry
  have h_cos2 : Real.cos θ * Real.cos θ = 3/4 := by sorry
  sorry

end trigonometric_identity_example_l200_200311


namespace g_values_l200_200094

variable (g : ℝ → ℝ)

-- Condition: ∀ x y z ∈ ℝ, g(x^2 + y * g(z)) = x * g(x) + 2 * z * g(y)
axiom g_axiom : ∀ x y z : ℝ, g (x^2 + y * g z) = x * g x + 2 * z * g y

-- Proposition: The possible values of g(4) are 0 and 8.
theorem g_values : g 4 = 0 ∨ g 4 = 8 :=
by
  sorry

end g_values_l200_200094


namespace range_of_m_l200_200653

theorem range_of_m (k : ℝ) (m : ℝ) (y x : ℝ)
  (h1 : ∀ x, y = k * (x - 1) + m)
  (h2 : y = 3 ∧ x = -2)
  (h3 : (∃ x, x < 0 ∧ y > 0) ∧ (∃ x, x < 0 ∧ y < 0) ∧ (∃ x, x > 0 ∧ y < 0)) :
  m < - (3 / 2) :=
sorry

end range_of_m_l200_200653


namespace triangle_square_side_length_ratio_l200_200621

theorem triangle_square_side_length_ratio (t s : ℝ) (ht : 3 * t = 12) (hs : 4 * s = 12) : 
  t / s = 4 / 3 :=
by
  sorry

end triangle_square_side_length_ratio_l200_200621


namespace real_possible_b_values_quadratic_non_real_roots_l200_200907

theorem real_possible_b_values_quadratic_non_real_roots :
  {b : ℝ | ∃ (a c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4*a*c < 0)} = {b : ℝ | -8 < b ∧ b < 8} :=
by 
  sorry

end real_possible_b_values_quadratic_non_real_roots_l200_200907


namespace milkman_pure_milk_l200_200384

theorem milkman_pure_milk (x : ℝ) 
  (h_cost : 3.60 * x = 3 * (x + 5)) : x = 25 :=
  sorry

end milkman_pure_milk_l200_200384


namespace ratio_sequences_l200_200370

-- Define positive integers n and k, with k >= n and k - n even.
variables {n k : ℕ} (h_k_ge_n : k ≥ n) (h_even : (k - n) % 2 = 0)

-- Define the sets S_N and S_M
def S_N (n k : ℕ) : ℕ := sorry -- Placeholder for the cardinality of S_N
def S_M (n k : ℕ) : ℕ := sorry -- Placeholder for the cardinality of S_M

-- Main theorem: N / M = 2^(k - n)
theorem ratio_sequences (h_k_ge_n : k ≥ n) (h_even : (k - n) % 2 = 0) :
  (S_N n k : ℝ) / (S_M n k : ℝ) = 2^(k - n) := sorry

end ratio_sequences_l200_200370


namespace man_rate_in_still_water_l200_200030

-- The conditions
def speed_with_stream : ℝ := 20
def speed_against_stream : ℝ := 4

-- The problem rephrased as a Lean statement
theorem man_rate_in_still_water : 
  (speed_with_stream + speed_against_stream) / 2 = 12 := 
by
  sorry

end man_rate_in_still_water_l200_200030


namespace five_digit_number_count_l200_200456

theorem five_digit_number_count : ∃ n, n = 1134 ∧ ∀ (a b c d e : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧ 
  (a < b ∧ b < c ∧ c > d ∧ d > e) → n = 1134 :=
by 
  sorry

end five_digit_number_count_l200_200456


namespace continuous_iff_k_n_continuous_l200_200481

def k_n (n : ℝ) (x : ℝ) : ℝ :=
  if x ≤ -n then -n
  else if x < n then x
  else n

noncomputable def f_continuous_iff_k_n_f_continuous (f : ℝ → ℝ) : Prop :=
  (Continuous f) ↔ ∀ n : ℝ, Continuous (λ x, k_n n (f x))

theorem continuous_iff_k_n_continuous (f : ℝ → ℝ) : f_continuous_iff_k_n_f_continuous f :=
sorry

end continuous_iff_k_n_continuous_l200_200481


namespace additive_inverse_of_half_l200_200121

theorem additive_inverse_of_half :
  - (1 / 2) = -1 / 2 :=
by
  sorry

end additive_inverse_of_half_l200_200121


namespace tom_typing_time_l200_200266

theorem tom_typing_time (typing_speed : ℕ) (words_per_page : ℕ) (number_of_pages : ℕ) 
  (h1 : typing_speed = 90) 
  (h2 : words_per_page = 450) 
  (h3 : number_of_pages = 10) : 
  (words_per_page / typing_speed) * number_of_pages = 50 :=
by simp [h1, h2, h3]; norm_num

end tom_typing_time_l200_200266


namespace find_k_l200_200671

theorem find_k (k : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-1, k)
  (a.1 * (2 * a.1 - b.1) + a.2 * (2 * a.2 - b.2)) = 0 → k = 12 := sorry

end find_k_l200_200671


namespace option_A_is_correct_l200_200413

-- Define propositions p and q
variables (p q : Prop)

-- Option A
def isOptionACorrect: Prop := (¬p ∨ ¬q) → (¬p ∧ ¬q)

theorem option_A_is_correct: isOptionACorrect p q := sorry

end option_A_is_correct_l200_200413


namespace cannot_form_polygon_l200_200843

-- Define the stick lengths as a list
def stick_lengths : List ℕ := List.range 100 |>.map (λ n => 2^n)

-- Define the condition for forming a polygon
def can_form_polygon (lst : List ℕ) : Prop :=
  ∃ subset, subset ⊆ lst ∧ subset.length ≥ 3 ∧ (∀ s ∈ subset, s < (subset.sum - s))

-- The theorem to be proved
theorem cannot_form_polygon : ¬ can_form_polygon stick_lengths :=
by 
  sorry

end cannot_form_polygon_l200_200843


namespace rectangle_side_ratio_l200_200289

noncomputable def sin_30_deg := 1 / 2

theorem rectangle_side_ratio 
  (a b c : ℝ) 
  (h1 : a + b = 2 * c) 
  (h2 : a * b = (c ^ 2) / 2) :
  (a / b = 3 + 2 * Real.sqrt 2) ∨ (a / b = 3 - 2 * Real.sqrt 2) :=
by
  sorry

end rectangle_side_ratio_l200_200289


namespace find_m_l200_200044

noncomputable def f (x : ℝ) := 4 * x^2 - 3 * x + 5
noncomputable def g (x : ℝ) (m : ℝ) := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 20) : m = -14 :=
by
  sorry

end find_m_l200_200044


namespace december_sales_fraction_l200_200694

noncomputable def average_sales (A : ℝ) := 11 * A
noncomputable def december_sales (A : ℝ) := 3 * A
noncomputable def total_sales (A : ℝ) := average_sales A + december_sales A

theorem december_sales_fraction (A : ℝ) (h1 : december_sales A = 3 * A)
  (h2 : average_sales A = 11 * A) :
  december_sales A / total_sales A = 3 / 14 :=
by
  sorry

end december_sales_fraction_l200_200694


namespace candy_count_l200_200633

theorem candy_count (S : ℕ) (H1 : 32 + S - 35 = 39) : S = 42 :=
by
  sorry

end candy_count_l200_200633


namespace hyperbola_center_l200_200183

theorem hyperbola_center 
  (x y : ℝ)
  (h : 9 * x ^ 2 + 54 * x - 16 * y ^ 2 - 128 * y - 200 = 0) : 
  (x = -3) ∧ (y = -4) := 
sorry

end hyperbola_center_l200_200183


namespace k_value_if_function_not_in_first_quadrant_l200_200215

theorem k_value_if_function_not_in_first_quadrant : 
  ∀ k : ℝ, (∀ x : ℝ, x > 0 → (k - 2) * x ^ (|k|) + k ≤ 0) → k = -1 :=
by
  sorry

end k_value_if_function_not_in_first_quadrant_l200_200215


namespace opposite_of_half_l200_200118

theorem opposite_of_half : - (1 / 2) = -1 / 2 := 
by
  sorry

end opposite_of_half_l200_200118


namespace volume_rectangular_box_l200_200979

variables {l w h : ℝ}

theorem volume_rectangular_box (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end volume_rectangular_box_l200_200979


namespace fraction_of_3_4_is_4_27_l200_200134

theorem fraction_of_3_4_is_4_27 (a b : ℚ) (h1 : a = 3/4) (h2 : b = 1/9) :
  b / a = 4 / 27 :=
by
  sorry

end fraction_of_3_4_is_4_27_l200_200134


namespace tan_sin_identity_l200_200313

noncomputable def tan_deg : ℝ := tan (real.pi / 6)
noncomputable def sin_deg : ℝ := sin (real.pi / 6)

theorem tan_sin_identity :
  (tan_deg ^ 2 - sin_deg ^ 2) / (tan_deg ^ 2 * sin_deg ^ 2) = 1 :=
by
  have tan_30 := by
    simp [tan_deg, real.tan_pi_div_six]
    norm_num
  have sin_30 := by
    simp [sin_deg, real.sin_pi_div_six]
    norm_num
  sorry

end tan_sin_identity_l200_200313


namespace Petya_receives_last_wrapper_l200_200484

variable (a b c : ℝ)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem Petya_receives_last_wrapper
  (h1 : discriminant a b c ≥ 0)
  (h2 : discriminant a c b ≥ 0)
  (h3 : discriminant b a c ≥ 0)
  (h4 : discriminant c a b < 0)
  (h5 : discriminant b c a < 0) :
  discriminant c b a ≥ 0 :=
sorry

end Petya_receives_last_wrapper_l200_200484


namespace investment_difference_l200_200098

noncomputable def A_Maria : ℝ := 60000 * (1 + 0.045)^3
noncomputable def A_David : ℝ := 60000 * (1 + 0.0175)^6
noncomputable def investment_diff : ℝ := A_Maria - A_David

theorem investment_difference : abs (investment_diff - 1803.30) < 1 :=
by
  have hM : A_Maria = 60000 * (1 + 0.045)^3 := by rfl
  have hD : A_David = 60000 * (1 + 0.0175)^6 := by rfl
  have hDiff : investment_diff = A_Maria - A_David := by rfl
  -- Proof would go here; using the provided approximations
  sorry

end investment_difference_l200_200098


namespace smallest_b_l200_200750

theorem smallest_b (b: ℕ) (h1: b > 3) (h2: ∃ n: ℕ, n^3 = 2 * b + 3) : b = 12 :=
sorry

end smallest_b_l200_200750


namespace move_point_inside_with_25_reflections_cannot_move_point_inside_with_24_reflections_l200_200948

-- Define the initial conditions
def pointA := (50 : ℝ)
def radius := (1 : ℝ)
def origin := (0 : ℝ)

-- Statement for part (a)
theorem move_point_inside_with_25_reflections :
  ∃ (n : ℕ) (r : ℝ), n = 25 ∧ r = radius + 50 ∧ pointA ≤ r :=
by
  sorry

-- Statement for part (b)
theorem cannot_move_point_inside_with_24_reflections :
  ∀ (n : ℕ) (r : ℝ), n = 24 → r = radius + 48 → pointA > r :=
by
  sorry

end move_point_inside_with_25_reflections_cannot_move_point_inside_with_24_reflections_l200_200948


namespace prod_sum_leq_four_l200_200520

theorem prod_sum_leq_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 4) :
  ab + bc + cd + da ≤ 4 :=
sorry

end prod_sum_leq_four_l200_200520


namespace disjoint_sets_l200_200001

def P : ℕ → ℝ → ℝ
| 0, x => x
| 1, x => 4 * x^3 + 3 * x
| (n + 1), x => (4 * x^2 + 2) * P n x - P (n - 1) x

def A (m : ℝ) : Set ℝ := {x | ∃ n : ℕ, P n m = x }

theorem disjoint_sets (m : ℝ) : Disjoint (A m) (A (m + 4)) :=
by
  -- Proof goes here
  sorry

end disjoint_sets_l200_200001


namespace rectangle_area_l200_200556

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_area_l200_200556


namespace uncle_zhang_age_l200_200588

theorem uncle_zhang_age (z l : ℕ) (h1 : z + l = 56) (h2 : z = l - (l / 2)) : z = 24 :=
by sorry

end uncle_zhang_age_l200_200588


namespace generatrix_length_of_cone_l200_200865

theorem generatrix_length_of_cone (r l : ℝ) (h1 : r = real.sqrt 2) 
    (h2 : 2 * real.pi * r = real.pi * l) : l = 2 * real.sqrt 2 :=
by
  -- The proof steps would go here, but we don't need to provide them.
  sorry

end generatrix_length_of_cone_l200_200865


namespace empty_solution_set_l200_200813

theorem empty_solution_set 
  (x : ℝ) 
  (h : -2 + 3 * x - 2 * x^2 > 0) : 
  false :=
by
  -- Discriminant calculation to prove empty solution set
  let delta : ℝ := 9 - 4 * 2 * 2
  have h_delta : delta < 0 := by norm_num
  sorry

end empty_solution_set_l200_200813


namespace door_height_is_eight_l200_200929

/-- Statement of the problem: given a door with specified dimensions as conditions,
prove that the height of the door is 8 feet. -/
theorem door_height_is_eight (x : ℝ) (h₁ : x^2 = (x - 4)^2 + (x - 2)^2) : (x - 2) = 8 :=
by
  sorry

end door_height_is_eight_l200_200929


namespace problem_1_problem_2_l200_200964

theorem problem_1 :
  83 * 87 = 100 * 8 * (8 + 1) + 21 :=
by sorry

theorem problem_2 (n : ℕ) :
  (10 * n + 3) * (10 * n + 7) = 100 * n * (n + 1) + 21 :=
by sorry

end problem_1_problem_2_l200_200964


namespace growth_rate_correct_max_avg_visitors_correct_l200_200989

-- Define the conditions from part 1
def visitors_march : ℕ := 80000
def visitors_may : ℕ := 125000

-- Define the monthly average growth rate
def monthly_avg_growth_rate (x : ℝ) : Prop :=
(1 + x)^2 = (visitors_may / visitors_march : ℝ)

-- Define the condition for June
def visitors_june_1_10 : ℕ := 66250
def max_avg_visitors_per_day (y : ℝ) : Prop :=
6.625 + 20 * y ≤ 15.625

-- Prove the monthly growth rate
theorem growth_rate_correct : ∃ x : ℝ, monthly_avg_growth_rate x ∧ x = 0.25 := sorry

-- Prove the max average visitors per day in June
theorem max_avg_visitors_correct : ∃ y : ℝ, max_avg_visitors_per_day y ∧ y = 0.45 := sorry

end growth_rate_correct_max_avg_visitors_correct_l200_200989


namespace find_b_over_a_l200_200088

variables {a b c : ℝ}
variables {b₃ b₇ b₁₁ : ℝ}

-- Conditions
def roots_of_quadratic (a b c b₃ b₁₁ : ℝ) : Prop :=
  ∃ p q, p + q = -b / a ∧ p * q = c / a ∧ (p = b₃ ∨ p = b₁₁) ∧ (q = b₃ ∨ q = b₁₁)

def middle_term_value (b₇ : ℝ) : Prop :=
  b₇ = 3

-- The statement to be proved
theorem find_b_over_a
  (h1 : roots_of_quadratic a b c b₃ b₁₁)
  (h2 : middle_term_value b₇)
  (h3 : b₃ + b₁₁ = 2 * b₇) :
  b / a = -6 :=
sorry

end find_b_over_a_l200_200088


namespace actual_price_per_gallon_l200_200786

variable (x : ℝ)
variable (expected_price : ℝ := x) -- price per gallon that the motorist expected to pay
variable (total_cash : ℝ := 12 * x) -- total cash to buy 12 gallons at expected price
variable (actual_price : ℝ := x + 0.30) -- actual price per gallon
variable (equation : 12 * x = 10 * (x + 0.30)) -- total cash equals the cost of 10 gallons at actual price

theorem actual_price_per_gallon (x : ℝ) (h : 12 * x = 10 * (x + 0.30)) : x + 0.30 = 1.80 := 
by 
  sorry

end actual_price_per_gallon_l200_200786


namespace door_height_is_eight_l200_200930

/-- Statement of the problem: given a door with specified dimensions as conditions,
prove that the height of the door is 8 feet. -/
theorem door_height_is_eight (x : ℝ) (h₁ : x^2 = (x - 4)^2 + (x - 2)^2) : (x - 2) = 8 :=
by
  sorry

end door_height_is_eight_l200_200930


namespace odd_function_phi_l200_200726

theorem odd_function_phi (φ : ℝ) (hφ1 : 0 ≤ φ) (hφ2 : φ ≤ π) (h : ∀ x : ℝ, cos (x + φ) = - cos (-x + φ)) : φ = π / 2 :=
by
  sorry

end odd_function_phi_l200_200726


namespace find_a_value_l200_200476

theorem find_a_value
  (a : ℝ)
  (h : ∀ x, 0 ≤ x ∧ x ≤ (π / 2) → a * Real.sin x + Real.cos x ≤ 2)
  (h_max : ∃ x, 0 ≤ x ∧ x ≤ (π / 2) ∧ a * Real.sin x + Real.cos x = 2) :
  a = Real.sqrt 3 :=
sorry

end find_a_value_l200_200476


namespace sarah_score_l200_200242

theorem sarah_score (j g s : ℕ) 
  (h1 : g = 2 * j) 
  (h2 : s = g + 50) 
  (h3 : (s + g + j) / 3 = 110) : 
  s = 162 := 
by 
  sorry

end sarah_score_l200_200242


namespace point_reflection_l200_200392

-- Define the original point and the reflection function
structure Point where
  x : ℝ
  y : ℝ

def reflect_y_axis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

-- Define the original point
def M : Point := ⟨-5, 2⟩

-- State the theorem to prove the reflection
theorem point_reflection : reflect_y_axis M = ⟨5, 2⟩ :=
  sorry

end point_reflection_l200_200392


namespace range_of_sum_l200_200340

theorem range_of_sum (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b + 1 / a + 9 / b = 10) : 2 ≤ a + b ∧ a + b ≤ 8 :=
sorry

end range_of_sum_l200_200340


namespace q_range_l200_200251

def q (x : ℝ) : ℝ := (x^2 - 2)^2

theorem q_range : 
  ∀ y : ℝ, y ∈ Set.range q ↔ 0 ≤ y :=
by sorry

end q_range_l200_200251


namespace remainder_div_180_l200_200842

theorem remainder_div_180 {j : ℕ} (h1 : 0 < j) (h2 : 120 % (j^2) = 12) : 180 % j = 0 :=
by
  sorry

end remainder_div_180_l200_200842


namespace largest_integer_satisfying_inequality_l200_200052

theorem largest_integer_satisfying_inequality : ∃ (x : ℤ), (5 * x - 4 < 3 - 2 * x) ∧ (∀ (y : ℤ), (5 * y - 4 < 3 - 2 * y) → y ≤ x) ∧ x = 0 :=
by
  sorry

end largest_integer_satisfying_inequality_l200_200052


namespace football_even_goal_prob_l200_200806

noncomputable def poisson_even_goal_prob (λ : ℝ) : ℝ :=
  let p := ∑' k, (Real.exp (-λ) * (λ ^ (2 * k))) / (Real.fact (2 * k))
  in p

theorem football_even_goal_prob : 
  poisson_even_goal_prob 2.8 ≈ 0.502 :=
by
  -- Proof of the theorem
  sorry

end football_even_goal_prob_l200_200806


namespace length_of_RU_l200_200685

theorem length_of_RU
  (P Q R S T U : Point)
  (PQ QR RP : ℝ)
  (PQ_dist : dist P Q = 13)
  (QR_dist : dist Q R = 20)
  (RP_dist : dist R P = 15)
  (angle_bisector : is_angle_bisector P Q R S)
  (T_on_circumcircle : T ≠ P ∧ T ∈ circumcircle (triangle P Q R))
  (U_on_circumcircle : P ≠ U ∧ U ∈ circumcircle (triangle P S T))
  (U_on_PQ : collinear P U Q) :
  dist R U = 20 :=
sorry

end length_of_RU_l200_200685


namespace divisible_by_77_l200_200103

theorem divisible_by_77 (n : ℤ) : ∃ k : ℤ, n^18 - n^12 - n^8 + n^2 = 77 * k :=
by
  sorry

end divisible_by_77_l200_200103


namespace rectangle_area_l200_200553

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_area_l200_200553


namespace complete_collection_prob_l200_200743

noncomputable def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem complete_collection_prob : 
  let n := 18
  let k := 10
  let needed := 6
  let uncollected_combinations := C 6 6
  let collected_combinations := C 12 4
  let total_combinations := C 18 10
  (uncollected_combinations * collected_combinations) / total_combinations = 5 / 442 :=
by 
  sorry

end complete_collection_prob_l200_200743


namespace sqrt_expression_l200_200631

theorem sqrt_expression : Real.sqrt ((4^2) * (5^6)) = 500 := by
  sorry

end sqrt_expression_l200_200631


namespace right_triangle_with_inscribed_circle_l200_200727

-- Define the problem conditions and the result
theorem right_triangle_with_inscribed_circle (k : ℝ) (h : k > 0) :
  (∀ α : ℝ, 
    α = (π / 4 - arcsin ((√2 * (k - 1)) / (2 * (k + 1)))) ∨
    α = (π / 4 + arcsin ((√2 * (k - 1)) / (2 * (k + 1)))))
  :=
sorry

end right_triangle_with_inscribed_circle_l200_200727


namespace avg_transformation_l200_200191

theorem avg_transformation
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 2) :
  ((3 * x₁ + 1) + (3 * x₂ + 1) + (3 * x₃ + 1) + (3 * x₄ + 1) + (3 * x₅ + 1)) / 5 = 7 :=
by
  sorry

end avg_transformation_l200_200191


namespace calculate_sum_of_squares_l200_200825

theorem calculate_sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 268 := 
  sorry

end calculate_sum_of_squares_l200_200825


namespace box_volume_l200_200975

theorem box_volume (l w h : ℝ)
  (A1 : l * w = 30)
  (A2 : w * h = 20)
  (A3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end box_volume_l200_200975


namespace meadow_total_money_l200_200709

/-
  Meadow orders 30 boxes of diapers weekly.
  Each box contains 40 packs.
  Each pack contains 160 diapers.
  Meadow sells each diaper for $5.
  Prove that the total money Meadow makes from selling all her diapers is $960000.
-/

theorem meadow_total_money :
  let number_of_boxes := 30 in
  let packs_per_box := 40 in
  let diapers_per_pack := 160 in
  let price_per_diaper := 5 in
  let total_packs := number_of_boxes * packs_per_box in
  let total_diapers := total_packs * diapers_per_pack in
  let total_money := total_diapers * price_per_diaper in
  total_money = 960000 :=
by
  sorry

end meadow_total_money_l200_200709


namespace exists_integer_multiple_of_3_2008_l200_200104

theorem exists_integer_multiple_of_3_2008 :
  ∃ k : ℤ, 3 ^ 2008 ∣ (k ^ 3 - 36 * k ^ 2 + 51 * k - 97) :=
sorry

end exists_integer_multiple_of_3_2008_l200_200104


namespace max_value_a_n_l200_200068

noncomputable def a_seq : ℕ → ℕ
| 0     => 0  -- By Lean's 0-based indexing, a_1 corresponds to a_seq 1
| 1     => 3
| (n+2) => a_seq (n+1) + 1

def S_n (n : ℕ) : ℕ := (n * (n + 5)) / 2

theorem max_value_a_n : 
  ∃ n : ℕ, S_n n = 2023 ∧ a_seq n = 73 :=
by
  sorry

end max_value_a_n_l200_200068


namespace find_ratio_squares_l200_200651

variables (x y z a b c : ℝ)

theorem find_ratio_squares 
  (h1 : x / a + y / b + z / c = 5) 
  (h2 : a / x + b / y + c / z = 0) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := 
sorry

end find_ratio_squares_l200_200651


namespace max_frac_a_c_squared_l200_200508

theorem max_frac_a_c_squared 
  (a b c : ℝ) (y z : ℝ)
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order: a ≥ b ∧ b ≥ c)
  (h_system: a^2 + z^2 = c^2 + y^2 ∧ c^2 + y^2 = (a - y)^2 + (c - z)^2)
  (h_bounds: 0 ≤ y ∧ y < a ∧ 0 ≤ z ∧ z < c) :
  (a/c)^2 ≤ 4/3 :=
sorry

end max_frac_a_c_squared_l200_200508


namespace rectangle_area_l200_200574

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end rectangle_area_l200_200574


namespace mark_savings_l200_200941

-- Given conditions
def original_price : ℝ := 300
def discount_rate : ℝ := 0.20
def cheaper_lens_price : ℝ := 220

-- Definitions derived from conditions
def discount_amount : ℝ := original_price * discount_rate
def discounted_price : ℝ := original_price - discount_amount
def savings : ℝ := discounted_price - cheaper_lens_price

-- Statement to prove
theorem mark_savings : savings = 20 :=
by
  -- Definitions incorporated
  have h1 : discount_amount = 300 * 0.20 := rfl
  have h2 : discounted_price = 300 - discount_amount := rfl
  have h3 : cheaper_lens_price = 220 := rfl
  have h4 : savings = discounted_price - cheaper_lens_price := rfl
  sorry

end mark_savings_l200_200941


namespace probability_volleyball_is_one_third_l200_200579

-- Define the total number of test items
def total_test_items : ℕ := 3

-- Define the number of favorable outcomes for hitting the wall with a volleyball
def favorable_outcomes_volleyball : ℕ := 1

-- Define the probability calculation
def probability_hitting_wall_with_volleyball : ℚ :=
  favorable_outcomes_volleyball / total_test_items

-- Prove the probability is 1/3
theorem probability_volleyball_is_one_third :
  probability_hitting_wall_with_volleyball = 1 / 3 := 
sorry

end probability_volleyball_is_one_third_l200_200579


namespace tilly_bag_cost_l200_200128

theorem tilly_bag_cost (n : ℕ) (p : ℕ) (profit : ℕ) : 
  (n = 100) → (p = 10) → (profit = 300) → (n * p - profit) / n = 7 := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3]
  norm_num
  sorry

end tilly_bag_cost_l200_200128


namespace min_pie_pieces_l200_200253

theorem min_pie_pieces (p : ℕ) : 
  (∀ (k : ℕ), (k = 5 ∨ k = 7) → ∃ (m : ℕ), p = k * m ∨ p = m * k) → p = 11 := 
sorry

end min_pie_pieces_l200_200253


namespace polarEquationOfCircleCenter1_1Radius1_l200_200396

noncomputable def circleEquationInPolarCoordinates (θ : ℝ) : ℝ := 2 * Real.cos (θ - 1)

theorem polarEquationOfCircleCenter1_1Radius1 (ρ θ : ℝ) 
  (h : Real.sqrt ((ρ * Real.cos θ - Real.cos 1)^2 + (ρ * Real.sin θ - Real.sin 1)^2) = 1) :
  ρ = circleEquationInPolarCoordinates θ :=
by sorry

end polarEquationOfCircleCenter1_1Radius1_l200_200396


namespace area_of_inscribed_rectangle_l200_200282

theorem area_of_inscribed_rectangle (r : ℝ) (h : r = 6) (ratio : ℝ) (hr : ratio = 3 / 1) :
  ∃ (length width : ℝ), (width = 2 * r) ∧ (length = ratio * width) ∧ (length * width = 432) :=
by
  sorry

end area_of_inscribed_rectangle_l200_200282


namespace rhombus_area_l200_200589

-- Define the rhombus with given conditions
def rhombus (a d1 d2 : ℝ) : Prop :=
  a = 9 ∧ abs (d1 - d2) = 10 

-- The theorem stating the area of the rhombus
theorem rhombus_area (a d1 d2 : ℝ) (h : rhombus a d1 d2) : 
  (d1 * d2) / 2 = 72 :=
by
  sorry

#check rhombus_area

end rhombus_area_l200_200589


namespace probability_of_top_grade_product_l200_200032

-- Definitions for the problem conditions
def P_B : ℝ := 0.03
def P_C : ℝ := 0.01

-- Given that the sum of all probabilities is 1
axiom sum_of_probabilities (P_A P_B P_C : ℝ) : P_A + P_B + P_C = 1

-- Statement to be proved
theorem probability_of_top_grade_product : ∃ P_A : ℝ, P_A = 1 - P_B - P_C ∧ P_A = 0.96 :=
by
  -- Assuming the proof steps to derive the answer
  sorry

end probability_of_top_grade_product_l200_200032


namespace equilateral_triangle_square_ratio_l200_200623

theorem equilateral_triangle_square_ratio (t s : ℕ) (h_t : 3 * t = 12) (h_s : 4 * s = 12) :
  t / s = 4 / 3 := by
  sorry

end equilateral_triangle_square_ratio_l200_200623


namespace sum_of_slopes_correct_l200_200596

noncomputable def sum_of_slopes : ℚ :=
  let Γ1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 1}
  let Γ2 := {p : ℝ × ℝ | (p.1 - 10)^2 + (p.2 - 11)^2 = 1}
  let l := {k : ℝ | ∃ p1 ∈ Γ1, ∃ p2 ∈ Γ1, ∃ p3 ∈ Γ2, ∃ p4 ∈ Γ2, p1 ≠ p2 ∧ p3 ≠ p4 ∧ p1.2 = k * p1.1 ∧ p3.2 = k * p3.1}
  let valid_slopes := {k | k ∈ l ∧ (k = 11/10 ∨ k = 1 ∨ k = 5/4)}
  (11 / 10) + 1 + (5 / 4)

theorem sum_of_slopes_correct : sum_of_slopes = 67 / 20 := 
  by sorry

end sum_of_slopes_correct_l200_200596


namespace not_in_range_l200_200177

noncomputable def g (x c: ℝ) : ℝ := x^2 + c * x + 5

theorem not_in_range (c : ℝ) (hc : -2 * Real.sqrt 2 < c ∧ c < 2 * Real.sqrt 2) :
  ∀ x : ℝ, g x c ≠ 3 :=
by
  intros
  sorry

end not_in_range_l200_200177


namespace paint_price_max_boxes_paint_A_l200_200536

theorem paint_price :
  ∃ x y : ℕ, (x + 2 * y = 56 ∧ 2 * x + y = 64) ∧ x = 24 ∧ y = 16 := by
  sorry

theorem max_boxes_paint_A (m : ℕ) :
  24 * m + 16 * (200 - m) ≤ 3920 → m ≤ 90 := by
  sorry

end paint_price_max_boxes_paint_A_l200_200536


namespace not_divisible_a1a2_l200_200699

theorem not_divisible_a1a2 (a1 a2 b1 b2 : ℕ) (h1 : 1 < b1) (h2 : b1 < a1) (h3 : 1 < b2) (h4 : b2 < a2) (h5 : b1 ∣ a1) (h6 : b2 ∣ a2) :
  ¬ (a1 * a2 ∣ a1 * b1 + a2 * b2 - 1) :=
by
  sorry

end not_divisible_a1a2_l200_200699


namespace parabola_focus_l200_200328

open Real

theorem parabola_focus (a : ℝ) (h k : ℝ) (x y : ℝ) (f : ℝ) :
  (a = -1/4) → (h = 0) → (k = 0) → 
  (f = (1 / (4 * a))) →
  (y = a * (x - h) ^ 2 + k) → 
  (y = -1 / 4 * x ^ 2) → f = -1 := by
  intros h_a h_h h_k h_f parabola_eq _
  rw [h_a, h_h, h_k] at *
  sorry

end parabola_focus_l200_200328


namespace even_goal_probability_approximation_l200_200802

noncomputable def poisson_even_goal_probability (λ : ℝ) : ℝ :=
  (e^(-λ) * Real.cosh λ)

theorem even_goal_probability_approximation :
  poisson_even_goal_probability 2.8 ≈ 0.502 :=
by
  sorry

end even_goal_probability_approximation_l200_200802


namespace quadratic_has_negative_root_sufficiency_quadratic_has_negative_root_necessity_l200_200652

theorem quadratic_has_negative_root_sufficiency 
  (a : ℝ) : (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) → (a < 0) :=
sorry

theorem quadratic_has_negative_root_necessity 
  (a : ℝ) : (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ (a < 0) :=
sorry

end quadratic_has_negative_root_sufficiency_quadratic_has_negative_root_necessity_l200_200652


namespace fraction_of_crop_to_CD_is_correct_l200_200829

-- Define the trapezoid with given conditions
structure Trapezoid :=
  (AB CD AD BC : ℝ)
  (angleA angleD : ℝ)
  (h: ℝ) -- height
  (Area Trapezoid total_area close_area_to_CD: ℝ) 

-- Assumptions
axiom AB_eq_CD (T : Trapezoid) : T.AB = 150 
axiom CD_eq_CD (T : Trapezoid) : T.CD = 200
axiom AD_eq_CD (T : Trapezoid) : T.AD = 130
axiom BC_eq_CD (T : Trapezoid) : T.BC = 130
axiom angleA_eq_75 (T : Trapezoid) : T.angleA = 75
axiom angleD_eq_75 (T : Trapezoid) : T.angleD = 75

-- The fraction calculation
noncomputable def fraction_to_CD (T : Trapezoid) : ℝ :=
  T.close_area_to_CD / T.total_area

-- Theorem stating the fraction of the crop that is brought to the longer base CD is 15/28
theorem fraction_of_crop_to_CD_is_correct (T : Trapezoid) 
  (h_pos : 0 < T.h)
  (total_area_def : T.total_area = (T.AB + T.CD) * T.h / 2)
  (close_area_def : T.close_area_to_CD = ((T.h / 4) * (T.AB + T.CD))) : 
  fraction_to_CD T = 15 / 28 :=
  sorry

end fraction_of_crop_to_CD_is_correct_l200_200829


namespace math_problem_l200_200936

theorem math_problem
  (m : ℕ) (h₁ : m = 8^126) :
  (m * 16) / 64 = 16^94 :=
by
  sorry

end math_problem_l200_200936


namespace tom_purchases_mangoes_l200_200970

theorem tom_purchases_mangoes (m : ℕ) (h1 : 8 * 70 + m * 65 = 1145) : m = 9 :=
by
  sorry

end tom_purchases_mangoes_l200_200970


namespace solve_for_x_l200_200488

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l200_200488


namespace min_value_of_expression_l200_200704

theorem min_value_of_expression (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x + y + z = 5) : 
  (9 / x + 16 / y + 25 / z) ≥ 28.8 :=
by sorry

end min_value_of_expression_l200_200704


namespace tiffany_cans_l200_200969

variable {M : ℕ}

theorem tiffany_cans : (M + 12 = 2 * M) → (M = 12) :=
by
  intro h
  sorry

end tiffany_cans_l200_200969


namespace angles_in_interval_l200_200638

-- Define the main statement we need to prove
theorem angles_in_interval (theta : ℝ) (h1 : 0 ≤ theta) (h2 : theta ≤ 2 * Real.pi) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos theta - x * (1 - x) + (1-x)^2 * Real.sin theta < 0) →
  (Real.pi / 2 < theta ∧ theta < 3 * Real.pi / 2) :=
by
  sorry

end angles_in_interval_l200_200638


namespace quadratic_non_real_roots_b_values_l200_200898

theorem quadratic_non_real_roots_b_values (b : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by sorry

end quadratic_non_real_roots_b_values_l200_200898


namespace simplify_complex_expression_l200_200541

theorem simplify_complex_expression (i : ℂ) (h : i^2 = -1) : 
  7 * (4 - 2 * i) - 2 * i * (3 - 4 * i) = 20 - 20 * i := 
by
  sorry

end simplify_complex_expression_l200_200541


namespace running_time_difference_l200_200303

theorem running_time_difference :
  ∀ (distance speed usual_speed : ℝ), 
  distance = 30 →
  usual_speed = 10 →
  speed = (distance / (usual_speed / 2)) - (distance / (usual_speed * 1.5)) →
  speed = 4 :=
by
  intros distance speed usual_speed hd hu hs
  sorry

end running_time_difference_l200_200303


namespace max_profit_l200_200582

-- Define the conditions
def profit (m : ℝ) := (m - 8) * (900 - 15 * m)

-- State the theorem
theorem max_profit (m : ℝ) : 
  ∃ M, M = profit 34 ∧ ∀ x, profit x ≤ M :=
by
  -- the proof goes here
  sorry

end max_profit_l200_200582


namespace find_abcde_l200_200259

theorem find_abcde (N : ℕ) (a b c d e f : ℕ) (h : a ≠ 0) 
(h1 : N % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f)
(h2 : (N^2) % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f) :
    a * 10000 + b * 1000 + c * 100 + d * 10 + e = 48437 :=
by sorry

end find_abcde_l200_200259


namespace salad_dressing_percentage_l200_200713

variable (P Q : ℝ) -- P and Q are the amounts of dressings P and Q in grams

-- Conditions
variable (h1 : 0.3 * P + 0.1 * Q = 12) -- The combined vinegar percentage condition
variable (h2 : P + Q = 100)            -- The total weight condition

-- Statement to prove
theorem salad_dressing_percentage (P_percent : ℝ) 
    (h1 : 0.3 * P + 0.1 * Q = 12) (h2 : P + Q = 100) : 
    P / (P + Q) * 100 = 10 :=
sorry

end salad_dressing_percentage_l200_200713


namespace shadow_boundary_eqn_l200_200160

noncomputable def boundary_of_shadow (x : ℝ) : ℝ := x^2 / 10 - 1

theorem shadow_boundary_eqn (radius : ℝ) (center : ℝ × ℝ × ℝ) (light_source : ℝ × ℝ × ℝ) (x y: ℝ) :
  radius = 2 →
  center = (0, 0, 2) →
  light_source = (0, -2, 3) →
  y = boundary_of_shadow x :=
by
  intros hradius hcenter hlight
  sorry

end shadow_boundary_eqn_l200_200160


namespace option_d_is_right_triangle_l200_200363

noncomputable def right_triangle (a b c : ℝ) : Prop :=
  a^2 + c^2 = b^2

theorem option_d_is_right_triangle (a b c : ℝ) (h : a^2 = b^2 - c^2) :
  right_triangle a b c :=
by
  sorry

end option_d_is_right_triangle_l200_200363


namespace athena_total_spent_l200_200683

noncomputable def cost_sandwiches := 4 * 3.25
noncomputable def cost_fruit_drinks := 3 * 2.75
noncomputable def cost_cookies := 6 * 1.50
noncomputable def cost_chips := 2 * 1.85

noncomputable def total_cost := cost_sandwiches + cost_fruit_drinks + cost_cookies + cost_chips

theorem athena_total_spent : total_cost = 33.95 := 
by 
  simp [cost_sandwiches, cost_fruit_drinks, cost_cookies, cost_chips, total_cost]
  sorry

end athena_total_spent_l200_200683


namespace zongzi_problem_l200_200009

def zongzi_prices : Prop :=
  ∀ (x y : ℕ), -- x: price of red bean zongzi, y: price of meat zongzi
  10 * x + 12 * y = 136 → -- total cost for the first customer
  y = 2 * x →
  x = 4 ∧ y = 8 -- prices found

def discounted_zongzi_prices : Prop :=
  ∀ (a b : ℕ), -- a: discounted price of red bean zongzi, b: discounted price of meat zongzi
  20 * a + 30 * b = 270 → -- cost for Xiaohuan's mother
  30 * a + 20 * b = 230 → -- cost for Xiaole's mother
  a = 3 ∧ b = 7 -- discounted prices found

def zongzi_packages (m : ℕ) : Prop :=
  ∀ (a b : ℕ), -- a: discounted price of red bean zongzi, b: discounted price of meat zongzi
  a = 3 → b = 7 →
  (80 - 4 * m) * (m * a + (40 - m) * b) + (4 * m + 8) * ((40 - m) * a + m * b) = 17280 →
  m ≤ 20 / 2 → -- quantity constraint
  m = 10 -- final m value

-- Statement to prove all together
theorem zongzi_problem :
  zongzi_prices ∧ discounted_zongzi_prices ∧ ∃ (m : ℕ), zongzi_packages m :=
by sorry

end zongzi_problem_l200_200009


namespace minimum_cars_with_racing_stripes_l200_200087

-- Definitions and conditions
variable (numberOfCars : ℕ) (withoutAC : ℕ) (maxWithACWithoutStripes : ℕ)

axiom total_number_of_cars : numberOfCars = 100
axiom cars_without_ac : withoutAC = 49
axiom max_ac_without_stripes : maxWithACWithoutStripes = 49    

-- Proposition
theorem minimum_cars_with_racing_stripes 
  (total_number_of_cars : numberOfCars = 100) 
  (cars_without_ac : withoutAC = 49)
  (max_ac_without_stripes : maxWithACWithoutStripes = 49) :
  ∃ (R : ℕ), R = 2 :=
by
  sorry

end minimum_cars_with_racing_stripes_l200_200087


namespace pears_in_basket_l200_200954

def TaniaFruits (b1 b2 b3 b4 b5 : ℕ) : Prop :=
  b1 = 18 ∧ b2 = 12 ∧ b3 = 9 ∧ b4 = b3 ∧ b5 + b1 + b2 + b3 + b4 = 58

theorem pears_in_basket {b1 b2 b3 b4 b5 : ℕ} (h : TaniaFruits b1 b2 b3 b4 b5) : b5 = 10 :=
by 
  sorry

end pears_in_basket_l200_200954


namespace center_coordinates_l200_200959

noncomputable def center_of_circle (x y : ℝ) : Prop := 
  x^2 + y^2 + 2*x - 4*y = 0

theorem center_coordinates : center_of_circle (-1) 2 :=
by sorry

end center_coordinates_l200_200959


namespace surface_area_of_cross_shape_with_five_unit_cubes_l200_200292

noncomputable def unit_cube_surface_area : ℕ := 6
noncomputable def num_cubes : ℕ := 5
noncomputable def total_surface_area_iso_cubes : ℕ := num_cubes * unit_cube_surface_area
noncomputable def central_cube_exposed_faces : ℕ := 2
noncomputable def surrounding_cubes_exposed_faces : ℕ := 5
noncomputable def surrounding_cubes_count : ℕ := 4
noncomputable def cross_shape_surface_area : ℕ := 
  central_cube_exposed_faces + (surrounding_cubes_count * surrounding_cubes_exposed_faces)

theorem surface_area_of_cross_shape_with_five_unit_cubes : cross_shape_surface_area = 22 := 
by sorry

end surface_area_of_cross_shape_with_five_unit_cubes_l200_200292


namespace solution_set_of_inequality_l200_200336

theorem solution_set_of_inequality (a b x : ℝ) (h1 : 0 < a) (h2 : b = 2 * a) : ax > b ↔ x > -2 :=
by sorry

end solution_set_of_inequality_l200_200336


namespace probability_penny_dime_heads_l200_200718

-- Define the probabilities for individual coin flips
def coin_flip_outcomes : ℕ := 2
def total_outcomes (n : ℕ) : ℕ := coin_flip_outcomes ^ n
def successful_outcomes : ℕ := coin_flip_outcomes ^ 3

-- Statement of the problem
theorem probability_penny_dime_heads : 
  (successful_outcomes : ℚ) / (total_outcomes 5 : ℚ) = 1 / 4 :=
by
  -- Proof omitted
  sorry

end probability_penny_dime_heads_l200_200718


namespace johns_profit_l200_200692

noncomputable def selling_price : ℝ := 2
noncomputable def num_newspapers : ℕ := 500
noncomputable def sell_fraction : ℝ := 0.80
noncomputable def buy_discount : ℝ := 0.75

def buying_price_per_newspaper : ℝ := selling_price * (1 - buy_discount)
def total_cost : ℝ := num_newspapers * buying_price_per_newspaper
def num_sold_newspapers : ℕ := (sell_fraction * num_newspapers.to_real).to_nat
def revenue : ℝ := num_sold_newspapers * selling_price
def profit : ℝ := revenue - total_cost

theorem johns_profit :
  profit = 550 := by
  sorry

end johns_profit_l200_200692


namespace complex_coordinates_l200_200880

theorem complex_coordinates (i : ℂ) (z : ℂ) (h : i^2 = -1) (h_z : z = (1 + 2 * i^3) / (2 + i)) :
  z = -i := 
by {
  sorry
}

end complex_coordinates_l200_200880


namespace dimes_count_l200_200414

-- Definitions of types of coins and their values in cents.
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def halfDollar := 50

-- Condition statements as assumptions
variables (num_pennies num_nickels num_dimes num_quarters num_halfDollars : ℕ)

-- Sum of all coins and their values (in cents)
def total_value := num_pennies * penny + num_nickels * nickel + num_dimes * dime + num_quarters * quarter + num_halfDollars * halfDollar

-- Total number of coins
def total_coins := num_pennies + num_nickels + num_dimes + num_quarters + num_halfDollars

-- Proving the number of dimes is 5 given the conditions.
theorem dimes_count : 
  total_value = 163 ∧ 
  total_coins = 12 ∧ 
  num_pennies ≥ 1 ∧ 
  num_nickels ≥ 1 ∧ 
  num_dimes ≥ 1 ∧ 
  num_quarters ≥ 1 ∧ 
  num_halfDollars ≥ 1 → 
  num_dimes = 5 :=
by
  sorry

end dimes_count_l200_200414


namespace greatest_k_divides_n_l200_200288

theorem greatest_k_divides_n (n : ℕ) (h_pos : 0 < n) (h_divisors_n : Nat.totient n = 72) (h_divisors_5n : Nat.totient (5 * n) = 90) : ∃ k : ℕ, ∀ m : ℕ, (5^k ∣ n) → (5^(k+1) ∣ n) → k = 3 :=
by
  sorry

end greatest_k_divides_n_l200_200288


namespace total_money_l200_200706

theorem total_money (m c : ℝ) (hm : m = 5 / 8) (hc : c = 7 / 20) : m + c = 0.975 := sorry

end total_money_l200_200706


namespace least_pos_int_with_ten_factors_l200_200410

theorem least_pos_int_with_ten_factors : ∃ (n : ℕ), n > 0 ∧ (∀ m, (m > 0 ∧ ∃ d : ℕ, d∣n → d = 1 ∨ d = n) → m < n) ∧ ( ∃! n, ∃ d : ℕ, d∣n ) := sorry

end least_pos_int_with_ten_factors_l200_200410


namespace remainder_equivalence_l200_200593

theorem remainder_equivalence (x y q r : ℕ) (hxy : x = q * y + r) (hy_pos : 0 < y) (h_r : 0 ≤ r ∧ r < y) : 
  ((x - 3 * q * y) % y) = r := 
by 
  sorry

end remainder_equivalence_l200_200593


namespace plane_through_A_perpendicular_to_BC_l200_200760

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_between_points (P Q : Point3D) : Point3D :=
  { x := Q.x - P.x, y := Q.y - P.y, z := Q.z - P.z }

def plane_eq (n : Point3D) (P : Point3D) (x y z : ℝ) : ℝ :=
  n.x * (x - P.x) + n.y * (y - P.y) + n.z * (z - P.z)

def A := Point3D.mk 0 (-2) 8
def B := Point3D.mk 4 3 2
def C := Point3D.mk 1 4 3

def n := vector_between_points B C
def plane := plane_eq n A

theorem plane_through_A_perpendicular_to_BC :
  ∀ x y z : ℝ, plane x y z = 0 ↔ -3 * x + y + z - 6 = 0 :=
by
  sorry

end plane_through_A_perpendicular_to_BC_l200_200760


namespace problem_statement_l200_200634

theorem problem_statement :
  ∀ k : Nat, (∃ r s : Nat, r > 0 ∧ s > 0 ∧ (k^2 - 6 * k + 11)^(r - 1) = (2 * k - 7)^s) ↔ (k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 8) :=
by
  sorry

end problem_statement_l200_200634


namespace quotient_calculation_l200_200138

theorem quotient_calculation
  (dividend : ℕ)
  (divisor : ℕ)
  (remainder : ℕ)
  (h_dividend : dividend = 176)
  (h_divisor : divisor = 14)
  (h_remainder : remainder = 8) :
  ∃ q, dividend = divisor * q + remainder ∧ q = 12 :=
by
  sorry

end quotient_calculation_l200_200138


namespace smallest_area_ellipse_tangent_to_circle_l200_200815

-- Definitions for the problem conditions.
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Stating the problem in Lean 4
theorem smallest_area_ellipse_tangent_to_circle : 
  ∃ (a b : ℝ), (∀ x y, circle x y → ellipse a b x y) ∧ (π * a * b = 5 * π) :=
by sorry

end smallest_area_ellipse_tangent_to_circle_l200_200815


namespace apples_difference_l200_200779

-- Definitions for initial and remaining apples
def initial_apples : ℕ := 46
def remaining_apples : ℕ := 14

-- The theorem to prove the difference between initial and remaining apples is 32
theorem apples_difference : initial_apples - remaining_apples = 32 := by
  -- proof is omitted
  sorry

end apples_difference_l200_200779


namespace students_with_screws_neq_bolts_l200_200357

-- Let's define the main entities
def total_students : ℕ := 40
def nails_neq_bolts : ℕ := 15
def screws_eq_nails : ℕ := 10

-- Main theorem statement
theorem students_with_screws_neq_bolts (total : ℕ) (neq_nails_bolts : ℕ) (eq_screws_nails : ℕ) :
  total = 40 → neq_nails_bolts = 15 → eq_screws_nails = 10 → ∃ k, k ≥ 15 ∧ k ≤ 40 - eq_screws_nails - neq_nails_bolts := 
by
  intros
  sorry

end students_with_screws_neq_bolts_l200_200357


namespace symmetric_abs_necessary_not_sufficient_l200_200332

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def y_axis_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x, |f (-x)| = |f x|

theorem symmetric_abs_necessary_not_sufficient (f : ℝ → ℝ) :
  is_odd_function f → y_axis_symmetric f := sorry

end symmetric_abs_necessary_not_sufficient_l200_200332


namespace coris_aunt_age_today_l200_200048

variable (Cori_age_now : ℕ) (age_diff : ℕ)

theorem coris_aunt_age_today (H1 : Cori_age_now = 3) (H2 : ∀ (Cori_age5 Aunt_age5 : ℕ), Cori_age5 = Cori_age_now + 5 → Aunt_age5 = 3 * Cori_age5 → Aunt_age5 - 5 = age_diff) :
  age_diff = 19 := 
by
  intros
  sorry

end coris_aunt_age_today_l200_200048


namespace remainder_sum_mod_13_l200_200759

theorem remainder_sum_mod_13 (a b c d : ℕ) 
(h₁ : a % 13 = 3) (h₂ : b % 13 = 5) (h₃ : c % 13 = 7) (h₄ : d % 13 = 9) : 
  (a + b + c + d) % 13 = 11 :=
by sorry

end remainder_sum_mod_13_l200_200759


namespace distinct_digits_sum_l200_200525

theorem distinct_digits_sum (A B C D G : ℕ) (AB CD GGG : ℕ)
  (h1: AB = 10 * A + B)
  (h2: CD = 10 * C + D)
  (h3: GGG = 111 * G)
  (h4: AB * CD = GGG)
  (h5: A ≠ B)
  (h6: A ≠ C)
  (h7: A ≠ D)
  (h8: A ≠ G)
  (h9: B ≠ C)
  (h10: B ≠ D)
  (h11: B ≠ G)
  (h12: C ≠ D)
  (h13: C ≠ G)
  (h14: D ≠ G)
  (hA: A < 10)
  (hB: B < 10)
  (hC: C < 10)
  (hD: D < 10)
  (hG: G < 10)
  : A + B + C + D + G = 17 := sorry

end distinct_digits_sum_l200_200525


namespace increasing_range_of_a_l200_200190

noncomputable def f (x : ℝ) (a : ℝ) := 
  if x ≤ 1 then -x^2 + 4*a*x 
  else (2*a + 3)*x - 4*a + 5

theorem increasing_range_of_a :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a ≤ f x₂ a) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
sorry

end increasing_range_of_a_l200_200190


namespace find_15th_term_l200_200445

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ a 2 = 4 ∧ ∀ n, a (n + 2) = a n

theorem find_15th_term :
  ∃ a : ℕ → ℕ, seq a ∧ a 15 = 3 :=
by
  sorry

end find_15th_term_l200_200445


namespace sum_of_edges_l200_200586

theorem sum_of_edges (a r : ℝ) 
  (h_vol : (a / r) * a * (a * r) = 432) 
  (h_surf_area : 2 * ((a * a) / r + (a * a) * r + a * a) = 384) 
  (h_geom_prog : r ≠ 1) :
  4 * ((6 * Real.sqrt 2) / r + 6 * Real.sqrt 2 + (6 * Real.sqrt 2) * r) = 72 * (Real.sqrt 2) := 
sorry

end sum_of_edges_l200_200586


namespace number_of_cube_roots_lt_15_l200_200887

theorem number_of_cube_roots_lt_15 : 
  { n : ℕ | n > 0 ∧ ∃ x : ℕ, x = n ∧ (x^(1/3:ℝ) < 15) }.card = 3375 := by
  sorry

end number_of_cube_roots_lt_15_l200_200887


namespace cost_per_pumpkin_pie_l200_200436

theorem cost_per_pumpkin_pie
  (pumpkin_pies : ℕ)
  (cherry_pies : ℕ)
  (cost_cherry_pie : ℕ)
  (total_profit : ℕ)
  (selling_price : ℕ)
  (total_revenue : ℕ)
  (total_cost : ℕ)
  (cost_pumpkin_pie : ℕ)
  (H1 : pumpkin_pies = 10)
  (H2 : cherry_pies = 12)
  (H3 : cost_cherry_pie = 5)
  (H4 : total_profit = 20)
  (H5 : selling_price = 5)
  (H6 : total_revenue = (pumpkin_pies + cherry_pies) * selling_price)
  (H7 : total_cost = total_revenue - total_profit)
  (H8 : total_cost = pumpkin_pies * cost_pumpkin_pie + cherry_pies * cost_cherry_pie) :
  cost_pumpkin_pie = 3 :=
by
  -- Placeholder for proof
  sorry

end cost_per_pumpkin_pie_l200_200436


namespace cone_generatrix_length_l200_200869

/-- Given that the base radius of a cone is √2, 
    and its lateral surface is unfolded into a semicircle,
    prove that the length of the generatrix of the cone is 2√2. -/
theorem cone_generatrix_length (r l : ℝ) 
  (h1 : r = real.sqrt 2)
  (h2 : 2 * real.pi * r = real.pi * l) :
  l = 2 * real.sqrt 2 :=
by
  sorry -- Proof steps go here (not required in the current context)


end cone_generatrix_length_l200_200869


namespace island_knight_majority_villages_l200_200529

def NumVillages := 1000
def NumInhabitants := 99
def TotalKnights := 54054
def AnswersPerVillage : ℕ := 66 -- Number of villagers who answered "more knights"
def RemainingAnswersPerVillage : ℕ := 33 -- Number of villagers who answered "more liars"

theorem island_knight_majority_villages : 
  ∃ n : ℕ, n = 638 ∧ (66 * n + 33 * (NumVillages - n) = TotalKnights) :=
by -- Begin the proof
  sorry -- Proof to be filled in later

end island_knight_majority_villages_l200_200529


namespace vertical_angles_congruent_l200_200106

theorem vertical_angles_congruent (A B : Angle) (h : vertical_angles A B) : congruent A B := sorry

end vertical_angles_congruent_l200_200106


namespace distributi_l200_200404

def number_of_distributions (spots : ℕ) (classes : ℕ) (min_spot_per_class : ℕ) : ℕ :=
  Nat.choose (spots - min_spot_per_class * classes + (classes - 1)) (classes - 1)

theorem distributi.on_of_10_spots (A B C : ℕ) (hA : A ≥ 1) (hB : B ≥ 1) (hC : C ≥ 1) 
(h_total : A + B + C = 10) : number_of_distributions 10 3 1 = 36 :=
by
  sorry

end distributi_l200_200404


namespace even_goal_probability_approximation_l200_200801

noncomputable def poisson_even_goal_probability (λ : ℝ) : ℝ :=
  (e^(-λ) * Real.cosh λ)

theorem even_goal_probability_approximation :
  poisson_even_goal_probability 2.8 ≈ 0.502 :=
by
  sorry

end even_goal_probability_approximation_l200_200801


namespace volume_of_rectangular_box_l200_200983

theorem volume_of_rectangular_box 
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12) : 
  l * w * h = 60 :=
sorry

end volume_of_rectangular_box_l200_200983


namespace factor_tree_X_value_l200_200356

def H : ℕ := 2 * 5
def J : ℕ := 3 * 7
def F : ℕ := 7 * H
def G : ℕ := 11 * J
def X : ℕ := F * G

theorem factor_tree_X_value : X = 16170 := by
  sorry

end factor_tree_X_value_l200_200356


namespace special_day_jacket_price_l200_200402

noncomputable def original_price : ℝ := 240
noncomputable def first_discount_rate : ℝ := 0.4
noncomputable def special_day_discount_rate : ℝ := 0.25

noncomputable def first_discounted_price : ℝ :=
  original_price * (1 - first_discount_rate)
  
noncomputable def special_day_price : ℝ :=
  first_discounted_price * (1 - special_day_discount_rate)

theorem special_day_jacket_price : special_day_price = 108 := by
  -- definitions and calculations go here
  sorry

end special_day_jacket_price_l200_200402


namespace intersection_M_N_l200_200998

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := 
by
  -- Proof to be provided
  sorry

end intersection_M_N_l200_200998


namespace min_balls_to_draw_l200_200915

theorem min_balls_to_draw (red blue green yellow white black : ℕ) (h_red : red = 35) (h_blue : blue = 25) (h_green : green = 22) (h_yellow : yellow = 18) (h_white : white = 14) (h_black : black = 12) : 
  ∃ n, n = 95 ∧ ∀ (r b g y w bl : ℕ), r ≤ red ∧ b ≤ blue ∧ g ≤ green ∧ y ≤ yellow ∧ w ≤ white ∧ bl ≤ black → (r + b + g + y + w + bl = 95 → r ≥ 18 ∨ b ≥ 18 ∨ g ≥ 18 ∨ y ≥ 18 ∨ w ≥ 18 ∨ bl ≥ 18) :=
by sorry

end min_balls_to_draw_l200_200915


namespace insurance_coverage_is_80_percent_l200_200587

-- Definitions and conditions
def MRI_cost : ℕ := 1200
def doctor_hourly_fee : ℕ := 300
def doctor_examination_time : ℕ := 30  -- in minutes
def seen_fee : ℕ := 150
def amount_paid_by_tim : ℕ := 300

-- The total cost calculation
def total_cost : ℕ := MRI_cost + (doctor_hourly_fee * doctor_examination_time / 60) + seen_fee

-- The amount covered by insurance
def amount_covered_by_insurance : ℕ := total_cost - amount_paid_by_tim

-- The percentage of coverage by insurance
def insurance_coverage_percentage : ℕ := (amount_covered_by_insurance * 100) / total_cost

theorem insurance_coverage_is_80_percent : insurance_coverage_percentage = 80 := by
  sorry

end insurance_coverage_is_80_percent_l200_200587


namespace coordinates_of_P_l200_200339

def A : ℝ × ℝ := (3, -4)
def B : ℝ × ℝ := (-9, 2)
def P : ℝ × ℝ := (-1, -2)

theorem coordinates_of_P : P = (1 / 3 • (B.1 - A.1) + 2 / 3 • A.1, 1 / 3 • (B.2 - A.2) + 2 / 3 • A.2) :=
by
    rw [A, B, P]
    sorry

end coordinates_of_P_l200_200339


namespace smallest_x_for_multiple_of_720_l200_200753

theorem smallest_x_for_multiple_of_720 (x : ℕ) (h1 : 450 = 2^1 * 3^2 * 5^2) (h2 : 720 = 2^4 * 3^2 * 5^1) : x = 8 ↔ (450 * x) % 720 = 0 :=
by
  sorry

end smallest_x_for_multiple_of_720_l200_200753


namespace find_a_find_b_plus_c_l200_200502

variable (a b c : ℝ)
variable (A B C : ℝ) -- Angles in radians

-- Condition: Given that 2a / cos A = (3c - 2b) / cos B
axiom condition1 : 2 * a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B)

-- Condition 1: b = sqrt(5) * sin B
axiom condition2 : b = Real.sqrt 5 * (Real.sin B)

-- Proof problem for finding a
theorem find_a : a = 5 / 3 := by
  sorry

-- Condition 2: a = sqrt(6) and the area is sqrt(5) / 2
axiom condition3 : a = Real.sqrt 6
axiom condition4 : 1 / 2 * b * c * (Real.sin A) = Real.sqrt 5 / 2

-- Proof problem for finding b + c
theorem find_b_plus_c : b + c = 4 := by
  sorry

end find_a_find_b_plus_c_l200_200502


namespace boat_navigation_under_arch_l200_200007

theorem boat_navigation_under_arch (h_arch : ℝ) (w_arch: ℝ) (boat_width: ℝ) (boat_height: ℝ) (boat_above_water: ℝ) :
  (h_arch = 5) → 
  (w_arch = 8) → 
  (boat_width = 4) → 
  (boat_height = 2) → 
  (boat_above_water = 0.75) →
  (h_arch - 2 = 3) :=
by
  intros h_arch_eq w_arch_eq boat_w_eq boat_h_eq boat_above_water_eq
  sorry

end boat_navigation_under_arch_l200_200007


namespace distance_between_locations_A_and_B_l200_200972

-- Define the conditions
variables {x y s t : ℝ}

-- Conditions specified in the problem
axiom bus_a_meets_bus_b_after_85_km : 85 / x = (s - 85) / y 
axiom buses_meet_again_after_turnaround : (s - 85 + 65) / x + 1 / 2 = (85 + (s - 65)) / y + 1 / 2

-- The theorem to be proved
theorem distance_between_locations_A_and_B : s = 190 :=
by
  sorry

end distance_between_locations_A_and_B_l200_200972


namespace ab_bc_cd_da_le_four_l200_200524

theorem ab_bc_cd_da_le_four (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 :=
by
  sorry

end ab_bc_cd_da_le_four_l200_200524


namespace fifteenth_term_is_three_l200_200446

noncomputable def sequence (n : ℕ) : ℕ :=
  if n % 2 = 1 then 3 else 4

theorem fifteenth_term_is_three : sequence 15 = 3 :=
  by sorry

end fifteenth_term_is_three_l200_200446


namespace power_inequality_l200_200474

theorem power_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  a^5 + b^5 > a^2 * b^3 + a^3 * b^2 :=
sorry

end power_inequality_l200_200474


namespace similar_triangles_legs_l200_200607

theorem similar_triangles_legs (y : ℝ) (h : 12 / y = 9 / 7) : y = 84 / 9 := by
  sorry

end similar_triangles_legs_l200_200607


namespace cone_generatrix_length_l200_200857

-- Define the conditions of the problem
def base_radius : ℝ := sqrt 2
def cone_base_circumference : ℝ := 2 * Real.pi * base_radius
def semicircle_arc_length (generatrix : ℝ) : ℝ := Real.pi * generatrix

-- The proof statement
theorem cone_generatrix_length : 
  ∃ (l : ℝ), 2 * Real.pi * sqrt 2 = Real.pi * l ∧ l = 2 * sqrt 2 :=
sorry

end cone_generatrix_length_l200_200857


namespace age_of_b_l200_200419

variable (a b : ℕ)
variable (h1 : a * 3 = b * 5)
variable (h2 : (a + 2) * 2 = (b + 2) * 3)

theorem age_of_b : b = 6 :=
by
  sorry

end age_of_b_l200_200419


namespace num_positive_whole_numbers_with_cube_roots_less_than_15_l200_200889

theorem num_positive_whole_numbers_with_cube_roots_less_than_15 : 
  {n : ℕ // n > 0 ∧ n < 15 ^ 3}.card = 3374 := 
by 
  sorry

end num_positive_whole_numbers_with_cube_roots_less_than_15_l200_200889


namespace geometric_sequence_problem_l200_200373

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

-- Define the statement for the roots of the quadratic function
def is_root (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f a = 0

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ :=
  x^2 - x - 2013

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h1 : is_geometric_sequence a) 
  (h2 : is_root quadratic_function (a 2)) 
  (h3 : is_root quadratic_function (a 3)) : 
  a 1 * a 4 = -2013 :=
sorry

end geometric_sequence_problem_l200_200373


namespace quadratic_roots_real_and_values_l200_200665

theorem quadratic_roots_real_and_values (m : ℝ) (x : ℝ) :
  (x ^ 2 - x + 2 * m - 2 = 0) → (m ≤ 9 / 8) ∧ (m = 1 → (x = 0 ∨ x = 1)) :=
by
  sorry

end quadratic_roots_real_and_values_l200_200665


namespace algebraic_expression_value_l200_200850

variables (a b c d m : ℤ)

def opposite (a b : ℤ) : Prop := a + b = 0
def reciprocal (c d : ℤ) : Prop := c * d = 1
def abs_eq_2 (m : ℤ) : Prop := |m| = 2

theorem algebraic_expression_value {a b c d m : ℤ} 
  (h1 : opposite a b) 
  (h2 : reciprocal c d) 
  (h3 : abs_eq_2 m) :
  (2 * m - (a + b - 1) + 3 * c * d = 8 ∨ 2 * m - (a + b - 1) + 3 * c * d = 0) :=
by
  sorry

end algebraic_expression_value_l200_200850


namespace range_of_b_l200_200200

noncomputable def f (x a b : ℝ) : ℝ :=
  x + a / x + b

theorem range_of_b (b : ℝ) :
  (∀ (a x : ℝ), (1/2 ≤ a ∧ a ≤ 2) ∧ (1/4 ≤ x ∧ x ≤ 1) → f x a b ≤ 10) →
  b ≤ 7 / 4 :=
by
  sorry

end range_of_b_l200_200200


namespace remainder_of_h_x10_div_h_x_l200_200509

noncomputable def h (x : ℤ) : ℤ := x^5 - x^4 + x^3 - x^2 + x - 1

theorem remainder_of_h_x10_div_h_x (x : ℤ) : (h (x ^ 10)) % (h x) = -6 :=
by
  sorry

end remainder_of_h_x10_div_h_x_l200_200509


namespace new_bookstore_acquisition_l200_200601

theorem new_bookstore_acquisition (x : ℝ) 
  (h1 : (1 / 2) * x + (1 / 4) * x + 50 = x - 200) : x = 1000 :=
by {
  sorry
}

end new_bookstore_acquisition_l200_200601


namespace cost_of_siding_l200_200110

def area_of_wall (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def area_of_roof (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length * width)

def area_of_sheet (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def sheets_needed (total_area : ℕ) (sheet_area : ℕ) : ℕ :=
  (total_area + sheet_area - 1) / sheet_area  -- Cooling the ceiling with integer arithmetic

def total_cost (sheets : ℕ) (price_per_sheet : ℕ) : ℕ :=
  sheets * price_per_sheet

theorem cost_of_siding : 
  ∀ (length_wall width_wall length_roof width_roof length_sheet width_sheet price_per_sheet : ℕ),
  length_wall = 10 → width_wall = 7 →
  length_roof = 10 → width_roof = 6 →
  length_sheet = 10 → width_sheet = 14 →
  price_per_sheet = 50 →
  total_cost (sheets_needed (area_of_wall length_wall width_wall + area_of_roof length_roof width_roof) (area_of_sheet length_sheet width_sheet)) price_per_sheet = 100 :=
by
  intros
  simp [area_of_wall, area_of_roof, area_of_sheet, sheets_needed, total_cost]
  sorry

end cost_of_siding_l200_200110


namespace shuttle_speeds_l200_200432

def speed_at_altitude (speed_per_sec : ℕ) : ℕ :=
  speed_per_sec * 3600

theorem shuttle_speeds (speed_300 speed_800 avg_speed : ℕ) :
  speed_at_altitude 7 = 25200 ∧ 
  speed_at_altitude 6 = 21600 ∧ 
  avg_speed = (25200 + 21600) / 2 ∧ 
  avg_speed = 23400 := 
by
  sorry

end shuttle_speeds_l200_200432


namespace james_vegetable_intake_l200_200688

theorem james_vegetable_intake :
  let daily_asparagus := 0.25
  let daily_broccoli := 0.25
  let daily_intake := daily_asparagus + daily_broccoli
  let doubled_daily_intake := daily_intake * 2
  let weekly_intake_asparagus_broccoli := doubled_daily_intake * 7
  let weekly_kale := 3
  let total_weekly_intake := weekly_intake_asparagus_broccoli + weekly_kale
  total_weekly_intake = 10 := 
by
  sorry

end james_vegetable_intake_l200_200688


namespace sum_of_circle_areas_l200_200162

theorem sum_of_circle_areas (a b c: ℝ)
  (h1: a + b = 6)
  (h2: b + c = 8)
  (h3: a + c = 10) :
  π * a^2 + π * b^2 + π * c^2 = 56 * π := 
by
  sorry

end sum_of_circle_areas_l200_200162


namespace twenty_five_percent_less_than_80_one_fourth_more_l200_200971

theorem twenty_five_percent_less_than_80_one_fourth_more (n : ℕ) (h : (5 / 4 : ℝ) * n = 60) : n = 48 :=
by
  sorry

end twenty_five_percent_less_than_80_one_fourth_more_l200_200971


namespace impossible_condition_l200_200344

noncomputable def f (x : ℝ) : ℝ := |2^x - 1|

theorem impossible_condition (a b c : ℝ) (h : f a > f b ∧ f b > f c) : ¬ (b < a ∧ a < c) :=
by
  sorry

end impossible_condition_l200_200344


namespace vector_addition_l200_200076

variable {𝕍 : Type} [AddCommGroup 𝕍] [Module ℝ 𝕍]
variable (a b : 𝕍)

theorem vector_addition : 
  (1 / 2 : ℝ) • (2 • a - 4 • b) + 2 • b = a :=
by
  sorry

end vector_addition_l200_200076


namespace tan_add_pi_over_4_sin_over_expression_l200_200193

variable (α : ℝ)

theorem tan_add_pi_over_4 (h : Real.tan α = 2) : 
  Real.tan (α + π / 4) = -3 := 
  sorry

theorem sin_over_expression (h : Real.tan α = 2) : 
  (Real.sin (2 * α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1) = 1 := 
  sorry

end tan_add_pi_over_4_sin_over_expression_l200_200193


namespace calculate_sum_of_squares_l200_200824

theorem calculate_sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 268 := 
  sorry

end calculate_sum_of_squares_l200_200824


namespace door_height_eight_l200_200924

theorem door_height_eight (x : ℝ) (h w : ℝ) (H1 : w = x - 4) (H2 : h = x - 2) (H3 : x^2 = (x - 4)^2 + (x - 2)^2) : h = 8 :=
by
  sorry

end door_height_eight_l200_200924


namespace find_second_number_l200_200113

theorem find_second_number 
  (h₁ : (20 + 40 + 60) / 3 = (10 + x + 15) / 3 + 5) :
  x = 80 :=
  sorry

end find_second_number_l200_200113


namespace sum_of_integers_l200_200123

theorem sum_of_integers (a b c : ℤ) (h1 : a = (1 / 3) * (b + c)) (h2 : b = (1 / 5) * (a + c)) (h3 : c = 35) : a + b + c = 60 :=
by
  sorry

end sum_of_integers_l200_200123


namespace rectangle_area_l200_200575

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end rectangle_area_l200_200575


namespace non_real_roots_interval_l200_200905

theorem non_real_roots_interval (b : ℝ) : (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by
  sorry

end non_real_roots_interval_l200_200905


namespace part_i_part_ii_l200_200046

noncomputable theory

-- Statement for Part (i)
theorem part_i (a : ℕ) (h : a > 1) :
  ∃ P : ℕ → Prop, (∀ n, Prime (P n) ∧ ∃ k, P n ∣ (a ^ k + 1)) ∧ ∀ n, P n ≠ P (n + 1) :=
sorry

-- Statement for Part (ii)
theorem part_ii (a : ℕ) (h : a > 1) :
  ∃ Q : ℕ → Prop, (∀ n, Prime (Q n) ∧ ∀ k, ¬ (Q n ∣ (a ^ k + 1))) ∧ ∀ n, Q n ≠ Q (n + 1) :=
sorry

end part_i_part_ii_l200_200046


namespace jordon_machine_input_l200_200693

theorem jordon_machine_input (x : ℝ) : (3 * x - 6) / 2 + 9 = 27 → x = 14 := 
by
  sorry

end jordon_machine_input_l200_200693


namespace certain_number_is_3500_l200_200602

theorem certain_number_is_3500 :
  ∃ x : ℝ, x - (1000 / 20.50) = 3451.2195121951218 ∧ x = 3500 :=
by
  sorry

end certain_number_is_3500_l200_200602


namespace sum_of_fourth_powers_eq_square_of_sum_of_squares_l200_200240

theorem sum_of_fourth_powers_eq_square_of_sum_of_squares 
  (x1 x2 x3 : ℝ) (p q n : ℝ)
  (h1 : x1^3 + p*x1^2 + q*x1 + n = 0)
  (h2 : x2^3 + p*x2^2 + q*x2 + n = 0)
  (h3 : x3^3 + p*x3^2 + q*x3 + n = 0)
  (h_rel : q^2 = 2 * n * p) :
  x1^4 + x2^4 + x3^4 = (x1^2 + x2^2 + x3^2)^2 := 
sorry

end sum_of_fourth_powers_eq_square_of_sum_of_squares_l200_200240


namespace generatrix_length_l200_200875

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end generatrix_length_l200_200875


namespace walking_speed_l200_200429

theorem walking_speed (total_time : ℕ) (distance : ℕ) (rest_interval : ℕ) (rest_time : ℕ) (rest_periods: ℕ) 
  (total_rest_time: ℕ) (total_walking_time: ℕ) (hours: ℕ) 
  (H1 : total_time = 332) 
  (H2 : distance = 50) 
  (H3 : rest_interval = 10) 
  (H4 : rest_time = 8)
  (H5 : rest_periods = distance / rest_interval - 1) 
  (H6 : total_rest_time = rest_periods * rest_time)
  (H7 : total_walking_time = total_time - total_rest_time) 
  (H8 : hours = total_walking_time / 60) : 
  (distance / hours) = 10 :=
by {
  -- proof omitted
  sorry
}

end walking_speed_l200_200429


namespace prod_sum_leq_four_l200_200519

theorem prod_sum_leq_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 4) :
  ab + bc + cd + da ≤ 4 :=
sorry

end prod_sum_leq_four_l200_200519


namespace continuous_constant_function_l200_200051

noncomputable def f (x : ℝ) : ℝ := sorry

theorem continuous_constant_function :
  (∀ a b : ℝ, (a^2 + a*b + b^2) * (∫ x in a..b, f x) = 3 * (∫ x in a..b, x^2 * f x)) →
  continuous f →
  ∃ C : ℝ, ∀ x : ℝ, f x = C := 
sorry

end continuous_constant_function_l200_200051


namespace Jennifer_future_age_Jordana_future_age_Jordana_current_age_l200_200691

variable (Jennifer_age_now Jordana_age_now : ℕ)

-- Conditions
def age_in_ten_years (current_age : ℕ) : ℕ := current_age + 10
theorem Jennifer_future_age : age_in_ten_years Jennifer_age_now = 30 := sorry
theorem Jordana_future_age : age_in_ten_years Jordana_age_now = 3 * age_in_ten_years Jennifer_age_now := sorry

-- Question to prove
theorem Jordana_current_age : Jordana_age_now = 80 := sorry

end Jennifer_future_age_Jordana_future_age_Jordana_current_age_l200_200691


namespace plywood_perimeter_difference_l200_200772

theorem plywood_perimeter_difference :
  let l := 10
  let w := 6
  let n := 6
  ∃ p_max p_min, 
    (l * w) % n = 0 ∧
    (p_max = 24) ∧
    (p_min = 12.66) ∧
    p_max - p_min = 11.34 := 
by
  sorry

end plywood_perimeter_difference_l200_200772


namespace billy_sleep_total_l200_200212

theorem billy_sleep_total
  (h₁ : ∀ n : ℕ, n = 1 → ∃ h : ℕ, h = 6)
  (h₂ : ∀ n : ℕ, n = 2 → ∃ h : ℕ, h = (6 + 2))
  (h₃ : ∀ n : ℕ, n = 3 → ∃ h : ℕ, h = ((6 + 2) / 2))
  (h₄ : ∀ n : ℕ, n = 4 → ∃ h : ℕ, h = (((6 + 2) / 2) * 3)) :
  ∑ n in {1, 2, 3, 4}, (classical.some (h₁ n 1) + classical.some (h₂ n 2) + classical.some (h₃ n 3) + classical.some (h₄ n 4)) = 30 :=
by sorry

end billy_sleep_total_l200_200212


namespace rectangle_area_l200_200570

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l200_200570


namespace evaluate_expression_l200_200301

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -a - b^3 + a * b^2 = 59 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l200_200301


namespace simplify_expression_l200_200111

variable (t : ℝ)

theorem simplify_expression (ht : t > 0) (ht_ne : t ≠ 1 / 2) :
  (1 - Real.sqrt (2 * t)) / ( (1 - Real.sqrt (4 * t ^ (3 / 4))) / (1 - Real.sqrt (2 * t ^ (1 / 4))) - Real.sqrt (2 * t)) *
  (Real.sqrt (1 / (1 / 2) + Real.sqrt (4 * t ^ 2)) / (1 + Real.sqrt (1 / (2 * t))) - Real.sqrt (2 * t))⁻¹ = 1 :=
by
  sorry

end simplify_expression_l200_200111


namespace sequence_nth_term_l200_200662

theorem sequence_nth_term (a : ℕ → ℚ) (h : a 1 = 3 / 2 ∧ a 2 = 1 ∧ a 3 = 5 / 8 ∧ a 4 = 3 / 8) :
  ∀ n : ℕ, a n = (n^2 - 11*n + 34) / 16 := by
  sorry

end sequence_nth_term_l200_200662


namespace inequality_solution_sets_equivalence_l200_200879

theorem inequality_solution_sets_equivalence
  (a b : ℝ)
  (h1 : (∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - 5 * x + b > 0)) :
  (∀ x : ℝ, x < -1/3 ∨ x > 1/2 ↔ bx^2 - 5 * x + a > 0) :=
  sorry

end inequality_solution_sets_equivalence_l200_200879


namespace min_moves_to_balance_stacks_l200_200732

theorem min_moves_to_balance_stacks :
  let stack1 := 9
  let stack2 := 7
  let stack3 := 5
  let stack4 := 10
  let target := 8
  let total_coins := stack1 + stack2 + stack3 + stack4
  total_coins = 31 →
  ∃ moves, moves = 11 ∧
    (stack1 + 3 * moves = target) ∧
    (stack2 + 3 * moves = target) ∧
    (stack3 + 3 * moves = target) ∧
    (stack4 + 3 * moves = target) :=
sorry

end min_moves_to_balance_stacks_l200_200732


namespace complete_collection_prob_l200_200742

noncomputable def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem complete_collection_prob : 
  let n := 18
  let k := 10
  let needed := 6
  let uncollected_combinations := C 6 6
  let collected_combinations := C 12 4
  let total_combinations := C 18 10
  (uncollected_combinations * collected_combinations) / total_combinations = 5 / 442 :=
by 
  sorry

end complete_collection_prob_l200_200742


namespace find_height_of_door_l200_200925

noncomputable def height_of_door (x : ℝ) (w : ℝ) (h : ℝ) : ℝ := h

theorem find_height_of_door :
  ∃ x w h, (w = x - 4) ∧ (h = x - 2) ∧ (x^2 = w^2 + h^2) ∧ height_of_door x w h = 8 :=
by {
  sorry
}

end find_height_of_door_l200_200925


namespace depth_of_channel_l200_200767

theorem depth_of_channel (top_width bottom_width : ℝ) (area : ℝ) (h : ℝ) 
  (h_top : top_width = 14) (h_bottom : bottom_width = 8) (h_area : area = 770) :
  (1 / 2) * (top_width + bottom_width) * h = area → h = 70 :=
by
  intros h_trapezoid
  sorry

end depth_of_channel_l200_200767


namespace min_value_is_3_l200_200173

theorem min_value_is_3 (a b : ℝ) (h1 : a > b / 2) (h2 : 2 * a > b) : (2 * a + b) / a ≥ 3 :=
sorry

end min_value_is_3_l200_200173


namespace paint_price_and_max_boxes_l200_200538

theorem paint_price_and_max_boxes (x y m : ℕ) 
  (h1 : x + 2 * y = 56) 
  (h2 : 2 * x + y = 64) 
  (h3 : 24 * m + 16 * (200 - m) ≤ 3920) : 
  x = 24 ∧ y = 16 ∧ m ≤ 90 := 
  by 
    sorry

end paint_price_and_max_boxes_l200_200538


namespace batsman_average_after_17th_inning_l200_200598

variable (A : ℝ) (total_runs : ℝ) (new_average : ℝ)
hypothesis h1 : total_runs = 16 * A + 87
hypothesis h2 : new_average =  (total_runs / 17)
hypothesis h3 : new_average = A + 3

theorem batsman_average_after_17th_inning :
  new_average = 39 :=
by
  sorry

end batsman_average_after_17th_inning_l200_200598


namespace cost_per_bag_l200_200127

theorem cost_per_bag (total_bags : ℕ) (sale_price_per_bag : ℕ) (desired_profit : ℕ) (total_revenue : ℕ)
  (total_cost : ℕ) (cost_per_bag : ℕ) :
  total_bags = 100 → sale_price_per_bag = 10 → desired_profit = 300 →
  total_revenue = total_bags * sale_price_per_bag →
  total_cost = total_revenue - desired_profit →
  cost_per_bag = total_cost / total_bags →
  cost_per_bag = 7 := by
  sorry

end cost_per_bag_l200_200127


namespace james_vegetable_consumption_l200_200689

theorem james_vegetable_consumption : 
  (1/4 + 1/4) * 2 * 7 + 3 = 10 := 
by
  sorry

end james_vegetable_consumption_l200_200689


namespace fill_tank_in_12_minutes_l200_200595

theorem fill_tank_in_12_minutes (rate1 rate2 rate_out : ℝ) 
  (h1 : rate1 = 1 / 18) (h2 : rate2 = 1 / 20) (h_out : rate_out = 1 / 45) : 
  12 = 1 / (rate1 + rate2 - rate_out) :=
by
  -- sorry will be replaced with the actual proof.
  sorry

end fill_tank_in_12_minutes_l200_200595


namespace smallest_x_for_multiple_l200_200751

theorem smallest_x_for_multiple (x : ℕ) : (450 * x) % 720 = 0 ↔ x = 8 := 
by {
  sorry
}

end smallest_x_for_multiple_l200_200751


namespace find_number_l200_200126

theorem find_number (x : ℝ) 
(h : x * 13.26 + x * 9.43 + x * 77.31 = 470) : 
x = 4.7 := 
sorry

end find_number_l200_200126


namespace integer_solution_of_inequality_l200_200952

theorem integer_solution_of_inequality :
  ∀ (x : ℤ), 0 < (x - 1 : ℚ) * (x - 1) / (x + 1) ∧ (x - 1) * (x - 1) / (x + 1) < 1 →
  x > -1 ∧ x ≠ 1 ∧ x < 3 → 
  x = 2 :=
by
  sorry

end integer_solution_of_inequality_l200_200952


namespace total_built_up_area_l200_200254

theorem total_built_up_area
    (A1 A2 A3 A4 : ℕ)
    (hA1 : A1 = 480)
    (hA2 : A2 = 560)
    (hA3 : A3 = 200)
    (hA4 : A4 = 440)
    (total_plot_area : ℕ)
    (hplots : total_plot_area = 4 * (480 + 560 + 200 + 440) / 4)
    : 800 = total_plot_area - (A1 + A2 + A3 + A4) :=
by
  -- This is where the solution will be filled in
  sorry

end total_built_up_area_l200_200254


namespace vendor_has_1512_liters_of_sprite_l200_200781

-- Define the conditions
def liters_of_maaza := 60
def liters_of_pepsi := 144
def least_number_of_cans := 143
def gcd_maaza_pepsi := Nat.gcd liters_of_maaza liters_of_pepsi --let Lean compute GCD

-- Define the liters per can as the GCD of Maaza and Pepsi
def liters_per_can := gcd_maaza_pepsi

-- Define the number of cans for Maaza and Pepsi respectively
def cans_of_maaza := liters_of_maaza / liters_per_can
def cans_of_pepsi := liters_of_pepsi / liters_per_can

-- Define total cans for Maaza and Pepsi
def total_cans_for_maaza_and_pepsi := cans_of_maaza + cans_of_pepsi

-- Define the number of cans for Sprite
def cans_of_sprite := least_number_of_cans - total_cans_for_maaza_and_pepsi

-- The total liters of Sprite the vendor has
def liters_of_sprite := cans_of_sprite * liters_per_can

-- Statement to prove
theorem vendor_has_1512_liters_of_sprite : 
  liters_of_sprite = 1512 :=
by
  -- solution omitted 
  sorry

end vendor_has_1512_liters_of_sprite_l200_200781


namespace tan_sin_identity_l200_200309

theorem tan_sin_identity :
  (let θ := 30 * Real.pi / 180 in 
   let tan_θ := Real.sin θ / Real.cos θ in 
   let sin_θ := 1 / 2 in
   let cos_θ := Real.sqrt 3 / 2 in
   ((tan_θ ^ 2 - sin_θ ^ 2) / (tan_θ ^ 2 * sin_θ ^ 2) = 1)) :=
by
  let θ := 30 * Real.pi / 180
  let tan_θ := Real.sin θ / Real.cos θ
  let sin_θ := 1 / 2
  let cos_θ := Real.sqrt 3 / 2
  have h_tan_θ : tan_θ = 1 / Real.sqrt 3 := sorry  -- We skip the derivation proof.
  have h_sin_θ : sin_θ = 1 / 2 := rfl
  have h_cos_θ : cos_θ = Real.sqrt 3 / 2 := sorry  -- We skip the derivation proof.
  have numerator := tan_θ ^ 2 - sin_θ ^ 2
  have denominator := tan_θ ^ 2 * sin_θ ^ 2
  -- show equality
  have h : (numerator / denominator) = (1 : ℝ) := sorry  -- We skip the final proof.
  exact h

end tan_sin_identity_l200_200309


namespace angle_of_inclination_l200_200658

/--
Given the direction vector of line l as (-sqrt(3), 3),
prove that the angle of inclination α of line l is 120 degrees.
-/
theorem angle_of_inclination (α : ℝ) :
  let direction_vector : Real × Real := (-Real.sqrt 3, 3)
  let slope := direction_vector.2 / direction_vector.1
  slope = -Real.sqrt 3 → α = 120 :=
by
  sorry

end angle_of_inclination_l200_200658


namespace two_digit_ab_divisible_by_11_13_l200_200080

theorem two_digit_ab_divisible_by_11_13 (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10^5 * 2 + 10^4 * 0 + 10^3 * 1 + 10^2 * a + 10 * b + 7) % 11 = 0)
  (h4 : (10^5 * 2 + 10^4 * 0 + 10^3 * 1 + 10^2 * a + 10 * b + 7) % 13 = 0) :
  10 * a + b = 48 :=
sorry

end two_digit_ab_divisible_by_11_13_l200_200080


namespace right_triangle_condition_l200_200362

theorem right_triangle_condition (a b c : ℝ) : (a^2 = b^2 - c^2) → (∃ B : ℝ, B = 90) := 
sorry

end right_triangle_condition_l200_200362


namespace cookies_per_bag_l200_200109

-- Definitions of the given conditions
def c1 := 23  -- number of chocolate chip cookies
def c2 := 25  -- number of oatmeal cookies
def b := 8    -- number of baggies

-- Statement to prove
theorem cookies_per_bag : (c1 + c2) / b = 6 :=
by 
  sorry

end cookies_per_bag_l200_200109


namespace fraction_of_sum_l200_200035

theorem fraction_of_sum (P : ℝ) (R : ℝ) (T : ℝ) (H_R : R = 8.333333333333337) (H_T : T = 2) : 
  let SI := (P * R * T) / 100
  let A := P + SI
  A / P = 1.1666666666666667 :=
by
  sorry

end fraction_of_sum_l200_200035


namespace circle_center_coordinates_l200_200958

open Real

noncomputable def circle_center (x y : Real) : Prop := 
  x^2 + y^2 - 4*x + 6*y = 0

theorem circle_center_coordinates :
  ∃ (a b : Real), circle_center a b ∧ a = 2 ∧ b = -3 :=
by
  use 2, -3
  sorry

end circle_center_coordinates_l200_200958


namespace solution_set_correct_l200_200062

noncomputable def solution_set (x : ℝ) : Prop :=
  x + 2 / (x + 1) > 2

theorem solution_set_correct :
  {x : ℝ | solution_set x} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | x > 1} :=
by sorry

end solution_set_correct_l200_200062


namespace hyperbola_equation_l200_200844

theorem hyperbola_equation 
  {a b : ℝ} (ha : a > 0) (hb : b > 0) 
  (h_gt : a > b)
  (parallel_asymptote : ∃ k : ℝ, k = 2)
  (focus_on_line : ∃ cₓ : ℝ, ∃ c : ℝ, c = 5 ∧ cₓ = -5 ∧ (y = -2 * cₓ - 10)) :
  ∃ (a b : ℝ), (a^2 = 5) ∧ (b^2 = 20) ∧ (a^2 > b^2) ∧ c = 5 ∧ (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 / 5 - y^2 / 20 = 1)) :=
sorry

end hyperbola_equation_l200_200844


namespace price_of_72_cans_l200_200420

def regular_price_per_can : ℝ := 0.30
def discount_percentage : ℝ := 0.15
def discounted_price_per_can := regular_price_per_can * (1 - discount_percentage)
def cans_purchased : ℕ := 72

theorem price_of_72_cans :
  cans_purchased * discounted_price_per_can = 18.36 :=
by sorry

end price_of_72_cans_l200_200420


namespace solve_compound_inequality_l200_200388

noncomputable def compound_inequality_solution (x : ℝ) : Prop :=
  (3 - (1 / (3 * x + 4)) < 5) ∧ (2 * x + 1 > 0)

theorem solve_compound_inequality (x : ℝ) :
  compound_inequality_solution x ↔ (x > -1/2) :=
by
  sorry

end solve_compound_inequality_l200_200388


namespace find_value_of_k_l200_200841

theorem find_value_of_k (k x : ℝ) 
  (h : 1 / (4 - x ^ 2) + 2 = k / (x - 2)) : 
  k = -1 / 4 :=
by
  sorry

end find_value_of_k_l200_200841


namespace tan_sin_div_l200_200305

theorem tan_sin_div (tan_30_sq : Real := (Real.sin 30 / Real.cos 30) ^ 2)
                    (sin_30_sq : Real := (Real.sin 30) ^ 2):
                    tan_30_sq = (1/3) → sin_30_sq = (1/4) → 
                    (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end tan_sin_div_l200_200305


namespace election_votes_l200_200734

noncomputable def third_candidate_votes (total_votes first_candidate_votes second_candidate_votes : ℕ) (winning_fraction : ℚ) : ℕ :=
  total_votes - (first_candidate_votes + second_candidate_votes)

theorem election_votes :
  ∃ total_votes : ℕ, 
  ∃ first_candidate_votes : ℕ,
  ∃ second_candidate_votes : ℕ,
  ∃ winning_fraction : ℚ,
  first_candidate_votes = 5000 ∧ 
  second_candidate_votes = 15000 ∧ 
  winning_fraction = 2/3 ∧ 
  total_votes = 60000 ∧ 
  third_candidate_votes total_votes first_candidate_votes second_candidate_votes winning_fraction = 40000 :=
    sorry

end election_votes_l200_200734


namespace least_integer_with_ten_factors_l200_200411

def number_of_factors (n : ℕ) : ℕ :=
  (n.factors.map (λ k, k + 1)).prod

theorem least_integer_with_ten_factors : ∃ n, ∃ p q a b : ℕ, 
  prime p ∧ prime q ∧ p < q ∧
  n = p^a * q^b ∧
  a + 1 = 2 ∧ b + 1 = 5 ∧
  n = 48 :=
sorry

end least_integer_with_ten_factors_l200_200411


namespace abs_b_leq_one_l200_200374

theorem abs_b_leq_one (a b : ℝ) (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) : |b| ≤ 1 := 
sorry

end abs_b_leq_one_l200_200374


namespace geometric_common_ratio_l200_200551

theorem geometric_common_ratio (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 5 * d)^2 = a1 * (a1 + 20 * d)) : 
  (a1 + 5 * d) / a1 = 3 :=
by
  sorry

end geometric_common_ratio_l200_200551


namespace number_of_roses_sold_l200_200783

def initial_roses : ℕ := 50
def picked_roses : ℕ := 21
def final_roses : ℕ := 56

theorem number_of_roses_sold : ∃ x : ℕ, initial_roses - x + picked_roses = final_roses ∧ x = 15 :=
by {
  sorry
}

end number_of_roses_sold_l200_200783


namespace part_a_part_b_l200_200472

variables (A B C D: Type) 

def is_midpoint (B C D : Type) : Prop := sorry

def angle (X Y Z : Type) : angle := sorry

theorem part_a (h_midpoint: is_midpoint B C D)
  (h_angleB: angle A B C = 105)
  (h_angleBDA: angle B D A = 45) :
  angle A C B = 30 :=
sorry

theorem part_b (h_angleC: angle A C B = 30)
  (h_angleB: angle A B C = 105)
  (h_angleBDA: angle B D A = 45) :
  is_midpoint B C D :=
sorry

end part_a_part_b_l200_200472


namespace quadratic_non_real_roots_l200_200901

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 16 = 0) ↔ b ∈ set.Ioo (-8 : ℝ) 8 :=
by
  sorry

end quadratic_non_real_roots_l200_200901


namespace remainder_of_2n_div_10_l200_200766

theorem remainder_of_2n_div_10 (n : ℤ) (h : n % 20 = 11) : (2 * n) % 10 = 2 := by
  sorry

end remainder_of_2n_div_10_l200_200766


namespace volume_rectangular_box_l200_200981

variables {l w h : ℝ}

theorem volume_rectangular_box (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end volume_rectangular_box_l200_200981


namespace polygon_sides_eq_14_l200_200124

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem polygon_sides_eq_14 (n : ℕ) (h : n + num_diagonals n = 77) : n = 14 :=
by
  sorry

end polygon_sides_eq_14_l200_200124


namespace mul_powers_same_base_l200_200169

theorem mul_powers_same_base (x : ℝ) : (x ^ 8) * (x ^ 2) = x ^ 10 :=
by
  exact sorry

end mul_powers_same_base_l200_200169


namespace sqrt_diff_ineq_sum_sq_gt_sum_prod_l200_200019

-- First proof problem: Prove that sqrt(11) - 2 * sqrt(3) > 3 - sqrt(10)
theorem sqrt_diff_ineq : (Real.sqrt 11 - 2 * Real.sqrt 3) > (3 - Real.sqrt 10) := sorry

-- Second proof problem: Prove that a^2 + b^2 + c^2 > ab + bc + ca given a, b, and c are real numbers that are not all equal
theorem sum_sq_gt_sum_prod (a b c : ℝ) (h : ¬ (a = b ∧ b = c ∧ a = c)) : a^2 + b^2 + c^2 > a * b + b * c + c * a := sorry

end sqrt_diff_ineq_sum_sq_gt_sum_prod_l200_200019


namespace dexter_total_cards_l200_200323

-- Define the given conditions as constants and variables in Lean
constant num_basketball_boxes : ℕ := 9
constant cards_per_basketball_box : ℕ := 15
constant boxes_filled_less_with_football_cards : ℕ := 3
constant cards_per_football_box : ℕ := 20

-- Calculate the derived quantities based on the above conditions
def num_football_boxes : ℕ := num_basketball_boxes - boxes_filled_less_with_football_cards
def total_basketball_cards : ℕ := num_basketball_boxes * cards_per_basketball_box
def total_football_cards : ℕ := num_football_boxes * cards_per_football_box
def total_cards : ℕ := total_basketball_cards + total_football_cards

-- State the theorem to prove
theorem dexter_total_cards : total_cards = 255 := by
  sorry  -- proof placeholder; the goal is to establish that total_cards = 255

end dexter_total_cards_l200_200323


namespace calculate_land_tax_l200_200627

def plot_size : ℕ := 15
def cadastral_value_per_sotka : ℕ := 100000
def tax_rate : ℝ := 0.003

theorem calculate_land_tax :
  plot_size * cadastral_value_per_sotka * tax_rate = 4500 := 
by 
  sorry

end calculate_land_tax_l200_200627


namespace average_is_207_l200_200765

variable (x : ℕ)

theorem average_is_207 (h : (201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + 212 + x) / 10 = 207) :
  x = 212 :=
sorry

end average_is_207_l200_200765


namespace isoelectronic_problem_1_part_1_isoelectronic_problem_1_part_2_isoelectronic_problem_2_l200_200084

-- Definitions for number of valence electrons
def valence_electrons (atom : String) : ℕ :=
  if atom = "C" then 4
  else if atom = "N" then 5
  else if atom = "O" then 6
  else if atom = "F" then 7
  else if atom = "S" then 6
  else 0

-- Definitions for molecular valence count
def molecule_valence_electrons (molecule : List String) : ℕ :=
  molecule.foldr (λ x acc => acc + valence_electrons x) 0

-- Definitions for specific molecules
def N2_molecule := ["N", "N"]
def CO_molecule := ["C", "O"]
def N2O_molecule := ["N", "N", "O"]
def CO2_molecule := ["C", "O", "O"]
def NO2_minus_molecule := ["N", "O", "O"]
def SO2_molecule := ["S", "O", "O"]
def O3_molecule := ["O", "O", "O"]

-- Isoelectronic property definition
def isoelectronic (mol1 mol2 : List String) : Prop :=
  molecule_valence_electrons mol1 = molecule_valence_electrons mol2

theorem isoelectronic_problem_1_part_1 :
  isoelectronic N2_molecule CO_molecule := sorry

theorem isoelectronic_problem_1_part_2 :
  isoelectronic N2O_molecule CO2_molecule := sorry

theorem isoelectronic_problem_2 :
  isoelectronic NO2_minus_molecule SO2_molecule ∧
  isoelectronic NO2_minus_molecule O3_molecule := sorry

end isoelectronic_problem_1_part_1_isoelectronic_problem_1_part_2_isoelectronic_problem_2_l200_200084


namespace area_proof_l200_200563

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end area_proof_l200_200563


namespace digit_d_is_six_l200_200464

theorem digit_d_is_six (d : ℕ) (h_even : d % 2 = 0) (h_digits_sum : 7 + 4 + 8 + 2 + d % 9 = 0) : d = 6 :=
by 
  sorry

end digit_d_is_six_l200_200464


namespace range_of_f_l200_200214

noncomputable def f : ℝ → ℝ := sorry

lemma problem_conditions (x : ℝ) (hx : x > 0) : 
  f x > 0 ∧ f x < deriv f x ∧ deriv f x < 3 * f x :=
sorry

theorem range_of_f : 
  1 / real.exp 6 < f 1 / f 3 ∧ f 1 / f 3 < 1 / real.exp 2 :=
sorry

end range_of_f_l200_200214


namespace sqrt_123400_l200_200188

theorem sqrt_123400 (h1: Real.sqrt 12.34 = 3.512) : Real.sqrt 123400 = 351.2 :=
by 
  sorry

end sqrt_123400_l200_200188


namespace max_sum_composite_shape_l200_200102

theorem max_sum_composite_shape :
  let faces_hex_prism := 8
  let edges_hex_prism := 18
  let vertices_hex_prism := 12

  let faces_hex_with_pyramid := 8 - 1 + 6
  let edges_hex_with_pyramid := 18 + 6
  let vertices_hex_with_pyramid := 12 + 1
  let sum_hex_with_pyramid := faces_hex_with_pyramid + edges_hex_with_pyramid + vertices_hex_with_pyramid

  let faces_rec_with_pyramid := 8 - 1 + 5
  let edges_rec_with_pyramid := 18 + 4
  let vertices_rec_with_pyramid := 12 + 1
  let sum_rec_with_pyramid := faces_rec_with_pyramid + edges_rec_with_pyramid + vertices_rec_with_pyramid

  sum_hex_with_pyramid = 50 ∧ sum_rec_with_pyramid = 46 ∧ sum_hex_with_pyramid ≥ sum_rec_with_pyramid := 
by
  have faces_hex_prism := 8
  have edges_hex_prism := 18
  have vertices_hex_prism := 12

  have faces_hex_with_pyramid := 8 - 1 + 6
  have edges_hex_with_pyramid := 18 + 6
  have vertices_hex_with_pyramid := 12 + 1
  have sum_hex_with_pyramid := faces_hex_with_pyramid + edges_hex_with_pyramid + vertices_hex_with_pyramid

  have faces_rec_with_pyramid := 8 - 1 + 5
  have edges_rec_with_pyramid := 18 + 4
  have vertices_rec_with_pyramid := 12 + 1
  have sum_rec_with_pyramid := faces_rec_with_pyramid + edges_rec_with_pyramid + vertices_rec_with_pyramid

  sorry -- proof omitted

end max_sum_composite_shape_l200_200102


namespace greatest_four_digit_number_divisible_by_3_and_4_l200_200270

theorem greatest_four_digit_number_divisible_by_3_and_4 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (n % 12 = 0) ∧ (∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ (m % 12 = 0) → m ≤ 9996) :=
by sorry

end greatest_four_digit_number_divisible_by_3_and_4_l200_200270


namespace total_ants_correct_l200_200606

-- Define the conditions
def park_width_ft : ℕ := 450
def park_length_ft : ℕ := 600
def ants_per_sq_inch_first_half : ℕ := 2
def ants_per_sq_inch_second_half : ℕ := 4

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Convert width and length from feet to inches
def park_width_inch : ℕ := park_width_ft * feet_to_inches
def park_length_inch : ℕ := park_length_ft * feet_to_inches

-- Define the area of each half of the park in square inches
def half_length_inch : ℕ := park_length_inch / 2
def area_first_half_sq_inch : ℕ := park_width_inch * half_length_inch
def area_second_half_sq_inch : ℕ := park_width_inch * half_length_inch

-- Define the number of ants in each half
def ants_first_half : ℕ := ants_per_sq_inch_first_half * area_first_half_sq_inch
def ants_second_half : ℕ := ants_per_sq_inch_second_half * area_second_half_sq_inch

-- Define the total number of ants
def total_ants : ℕ := ants_first_half + ants_second_half

-- The proof problem
theorem total_ants_correct : total_ants = 116640000 := by
  sorry

end total_ants_correct_l200_200606


namespace combined_weight_is_150_l200_200010

-- Definitions based on conditions
def tracy_weight : ℕ := 52
def jake_weight : ℕ := tracy_weight + 8
def weight_range : ℕ := 14
def john_weight : ℕ := tracy_weight - 14

-- Proving the combined weight
theorem combined_weight_is_150 :
  tracy_weight + jake_weight + john_weight = 150 := by
  sorry

end combined_weight_is_150_l200_200010


namespace length_generatrix_cone_l200_200873

theorem length_generatrix_cone (base_radius : ℝ) (lateral_surface_unfolded : Prop)
    (h_base_radius : base_radius = real.sqrt 2)
    (h_lateral_surface_unfolded : lateral_surface_unfolded) :
    let l := 2 * real.sqrt 2 in
    2 * π * base_radius = π * l :=
by {
  -- assumptions incorporated,
  -- the proof goes here
  sorry
}

end length_generatrix_cone_l200_200873


namespace probability_of_divisor_of_6_is_two_thirds_l200_200782

noncomputable def probability_divisor_of_6 : ℚ :=
  have divisors_of_6 : Finset ℕ := {1, 2, 3, 6}
  have total_possible_outcomes : ℕ := 6
  have favorable_outcomes : ℕ := 4
  have probability_event : ℚ := favorable_outcomes / total_possible_outcomes
  2 / 3

theorem probability_of_divisor_of_6_is_two_thirds :
  probability_divisor_of_6 = 2 / 3 :=
sorry

end probability_of_divisor_of_6_is_two_thirds_l200_200782


namespace trigonometric_identity_l200_200279

theorem trigonometric_identity (t : ℝ) : 
  5.43 * Real.cos (22 * Real.pi / 180 - t) * Real.cos (82 * Real.pi / 180 - t) +
  Real.cos (112 * Real.pi / 180 - t) * Real.cos (172 * Real.pi / 180 - t) = 
  0.5 * (Real.sin t + Real.cos t) :=
sorry

end trigonometric_identity_l200_200279


namespace least_three_digit_12_heavy_number_l200_200294

def is_12_heavy (n : ℕ) : Prop :=
  n % 12 > 8

def three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem least_three_digit_12_heavy_number :
  ∃ n, three_digit n ∧ is_12_heavy n ∧ ∀ m, three_digit m ∧ is_12_heavy m → n ≤ m :=
  Exists.intro 105 (by
    sorry)

end least_three_digit_12_heavy_number_l200_200294


namespace box_volume_l200_200978

theorem box_volume (l w h : ℝ)
  (A1 : l * w = 30)
  (A2 : w * h = 20)
  (A3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end box_volume_l200_200978


namespace length_generatrix_cone_l200_200872

theorem length_generatrix_cone (base_radius : ℝ) (lateral_surface_unfolded : Prop)
    (h_base_radius : base_radius = real.sqrt 2)
    (h_lateral_surface_unfolded : lateral_surface_unfolded) :
    let l := 2 * real.sqrt 2 in
    2 * π * base_radius = π * l :=
by {
  -- assumptions incorporated,
  -- the proof goes here
  sorry
}

end length_generatrix_cone_l200_200872


namespace series_convergence_p_geq_2_l200_200417

noncomputable def ai_series_converges (a : ℕ → ℝ) : Prop :=
  ∃ l : ℝ, ∑' i, a i ^ 2 = l

noncomputable def bi_series_converges (b : ℕ → ℝ) : Prop :=
  ∃ l : ℝ, ∑' i, b i ^ 2 = l

theorem series_convergence_p_geq_2 
  (a b : ℕ → ℝ) 
  (h₁ : ai_series_converges a)
  (h₂ : bi_series_converges b) 
  (p : ℝ) (hp : p ≥ 2) : 
  ∃ l : ℝ, ∑' i, |a i - b i| ^ p = l := 
sorry

end series_convergence_p_geq_2_l200_200417


namespace problem_f_symmetry_problem_f_definition_problem_correct_answer_l200_200378

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then Real.log x else Real.log (2 - x)

theorem problem_f_symmetry (x : ℝ) : f (2 - x) = f x := 
sorry

theorem problem_f_definition (x : ℝ) (hx : x ≥ 1) : f x = Real.log x :=
sorry

theorem problem_correct_answer: 
  f (1 / 2) < f 2 ∧ f 2 < f (1 / 3) :=
sorry

end problem_f_symmetry_problem_f_definition_problem_correct_answer_l200_200378


namespace polynomial_solutions_l200_200186

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

end polynomial_solutions_l200_200186


namespace shortest_distance_between_circles_is_zero_l200_200271

open Real

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop :=
  x^2 - 12 * x + y^2 - 8 * y - 12 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop :=
  x^2 + 10 * x + y^2 - 10 * y + 34 = 0

-- Statement of the proof problem: 
-- Prove the shortest distance between the two circles defined by circle1 and circle2 is 0.
theorem shortest_distance_between_circles_is_zero :
    ∀ (x1 y1 x2 y2 : ℝ),
      circle1 x1 y1 →
      circle2 x2 y2 →
      0 = 0 :=
by
  intros x1 y1 x2 y2 h1 h2
  sorry

end shortest_distance_between_circles_is_zero_l200_200271


namespace maximum_daily_sales_revenue_l200_200426

noncomputable def P (t : ℕ) : ℤ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

noncomputable def Q (t : ℕ) : ℤ :=
  if 0 < t ∧ t ≤ 30 then -t + 40 else 0

noncomputable def y (t : ℕ) : ℤ := P t * Q t

theorem maximum_daily_sales_revenue : 
  ∃ (t : ℕ), 0 < t ∧ t ≤ 30 ∧ y t = 1125 :=
by
  sorry

end maximum_daily_sales_revenue_l200_200426


namespace tv_episode_length_l200_200834

theorem tv_episode_length :
  ∀ (E : ℕ), 
    600 = 3 * E + 270 + 2 * 105 + 45 → 
    E = 25 :=
by
  intros E h
  sorry

end tv_episode_length_l200_200834


namespace Mr_Tom_invested_in_fund_X_l200_200944

theorem Mr_Tom_invested_in_fund_X (a b : ℝ) (h1 : a + b = 100000) (h2 : 0.17 * b = 0.23 * a + 200) : a = 42000 := 
by
  sorry

end Mr_Tom_invested_in_fund_X_l200_200944


namespace richmond_tigers_revenue_l200_200956

theorem richmond_tigers_revenue
  (total_tickets : ℕ)
  (first_half_tickets : ℕ)
  (catA_first_half : ℕ)
  (catB_first_half : ℕ)
  (catC_first_half : ℕ)
  (priceA : ℕ)
  (priceB : ℕ)
  (priceC : ℕ)
  (catA_second_half : ℕ)
  (catB_second_half : ℕ)
  (catC_second_half : ℕ)
  (total_revenue_second_half : ℕ)
  (h_total_tickets : total_tickets = 9570)
  (h_first_half_tickets : first_half_tickets = 3867)
  (h_catA_first_half : catA_first_half = 1350)
  (h_catB_first_half : catB_first_half = 1150)
  (h_catC_first_half : catC_first_half = 1367)
  (h_priceA : priceA = 50)
  (h_priceB : priceB = 40)
  (h_priceC : priceC = 30)
  (h_catA_second_half : catA_second_half = 1350)
  (h_catB_second_half : catB_second_half = 1150)
  (h_catC_second_half : catC_second_half = 1367)
  (h_total_revenue_second_half : total_revenue_second_half = 154510)
  :
  catA_second_half * priceA + catB_second_half * priceB + catC_second_half * priceC = total_revenue_second_half :=
by
  sorry

end richmond_tigers_revenue_l200_200956


namespace pie_division_min_pieces_l200_200252

-- Define the problem as a Lean statement
theorem pie_division_min_pieces : ∃ n : ℕ, (∀ m ∈ {5, 7}, n % m = 0) ∧ n = 11 :=
by
  use 11
  split
  -- Prove for 5
  { intro m
    intro hm
    cases hm
    -- m = 5
    { exact Nat.mod_eq_zero_of_dvd (Nat.dvd_trans (Nat.dvd_refl 11) (Nat.dvd_of_mem_divisors hm)) }
    -- m = 7
    { exact Nat.mod_eq_zero_of_dvd (Nat.dvd_trans (Nat.dvd_refl 11) (Nat.dvd_of_mem_divisors hm)) }
    -- Impossible, there are only 5 and 7
    contradiction }
  -- Prove n = 11
  exact rfl

end pie_division_min_pieces_l200_200252


namespace polynomial_coefficients_l200_200666

theorem polynomial_coefficients (a_0 a_1 a_2 a_3 a_4 : ℤ) :
  (∀ x : ℤ, (x + 2)^5 = (x + 1)^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_0 = 31 ∧ a_1 = 75 :=
by
  sorry

end polynomial_coefficients_l200_200666


namespace find_math_marks_l200_200050

theorem find_math_marks (subjects : ℕ)
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℝ)
  (math_marks : ℕ) :
  subjects = 5 →
  english_marks = 96 →
  physics_marks = 99 →
  chemistry_marks = 100 →
  biology_marks = 98 →
  average_marks = 98.2 →
  math_marks = 98 :=
by
  intros h_subjects h_english h_physics h_chemistry h_biology h_average
  sorry

end find_math_marks_l200_200050


namespace factorization_l200_200838

theorem factorization (a b : ℝ) : a * b^2 - 4 * a * b + 4 * a = a * (b - 2)^2 := 
by sorry

end factorization_l200_200838


namespace car_distance_travelled_l200_200021

theorem car_distance_travelled (time_hours : ℝ) (time_minutes : ℝ) (time_seconds : ℝ)
    (actual_speed : ℝ) (reduced_speed : ℝ) (distance : ℝ) :
    time_hours = 1 → 
    time_minutes = 40 →
    time_seconds = 48 →
    actual_speed = 34.99999999999999 → 
    reduced_speed = (5 / 7) * actual_speed → 
    distance = reduced_speed * ((time_hours + time_minutes / 60 + time_seconds / 3600) : ℝ) →
    distance = 42 := sorry

end car_distance_travelled_l200_200021


namespace relationship_between_a_and_b_l200_200218

open Polynomial

noncomputable def quadratic_eq_with_common_root (a b t : ℚ) : Prop :=
  (X^2 + C a * X + C b = 0) ∧ (X^2 + C b * X + C a = 0) ∧ (∃ x : ℚ, x = t)

theorem relationship_between_a_and_b
  (a b : ℚ) (h_diff : a ≠ b) (h_common_root : ∃ t : ℚ,
    (eval t (X^2 + C a * X + C b) = 0) ∧
    (eval t (X^2 + C b * X + C a) = 0)) :
  a + b + 1 = 0 :=
sorry

end relationship_between_a_and_b_l200_200218


namespace arithmetic_sequence_common_difference_l200_200682

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h₁ : a 2 = 9) (h₂ : a 5 = 33) :
  ∀ d : ℤ, (∀ n : ℕ, a n = a 1 + (n - 1) * d) → d = 8 :=
by
  -- We state the theorem and provide a "sorry" proof placeholder
  sorry

end arithmetic_sequence_common_difference_l200_200682


namespace parametric_to_standard_l200_200174

theorem parametric_to_standard (θ : ℝ) (x y : ℝ)
  (h1 : x = 1 + 2 * Real.cos θ)
  (h2 : y = 2 * Real.sin θ) :
  (x - 1)^2 + y^2 = 4 := 
sorry

end parametric_to_standard_l200_200174


namespace total_fish_correct_l200_200369

def Leo_fish := 40
def Agrey_fish := Leo_fish + 20
def Sierra_fish := Agrey_fish + 15
def total_fish := Leo_fish + Agrey_fish + Sierra_fish

theorem total_fish_correct : total_fish = 175 := by
  sorry


end total_fish_correct_l200_200369


namespace greatest_possible_value_of_squares_l200_200092

theorem greatest_possible_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 15)
  (h2 : ab + c + d = 78)
  (h3 : ad + bc = 160)
  (h4 : cd = 96) :
  a^2 + b^2 + c^2 + d^2 ≤ 717 ∧ ∃ a b c d, a + b = 15 ∧ ab + c + d = 78 ∧ ad + bc = 160 ∧ cd = 96 ∧ a^2 + b^2 + c^2 + d^2 = 717 :=
sorry

end greatest_possible_value_of_squares_l200_200092


namespace three_digit_numbers_count_l200_200204

theorem three_digit_numbers_count : 
  ∃ (count : ℕ), count = 3 ∧ 
  ∀ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
             (n / 100 = 9) ∧ 
             (∃ a b c, n = 100 * a + 10 * b + c ∧ a + b + c = 27) ∧ 
             (n % 2 = 0) → count = 3 :=
sorry

end three_digit_numbers_count_l200_200204


namespace lattice_point_intersection_l200_200917

open Int

theorem lattice_point_intersection : ∃ (k_values : Finset ℤ), k_values.card = 4 ∧ ∀ k ∈ k_values, ∃ (x y : ℤ), y = 2 * x - 1 ∧ y = k * x + k :=
by
  sorry

end lattice_point_intersection_l200_200917


namespace trigonometric_identity_l200_200307

theorem trigonometric_identity :
  let θ := 30 * Real.pi / 180
  in (Real.tan θ ^ 2 - Real.sin θ ^ 2) / (Real.tan θ ^ 2 * Real.sin θ ^ 2) = 1 :=
by
  sorry

end trigonometric_identity_l200_200307


namespace range_of_a_l200_200878

theorem range_of_a (a : ℝ) (h : ¬ ∃ x : ℝ, x^2 + (a + 1) * x + 1 ≤ 0) : -3 < a ∧ a < 1 :=
sorry

end range_of_a_l200_200878


namespace football_even_goal_prob_l200_200807

noncomputable def poisson_even_goal_prob (λ : ℝ) : ℝ :=
  let p := ∑' k, (Real.exp (-λ) * (λ ^ (2 * k))) / (Real.fact (2 * k))
  in p

theorem football_even_goal_prob : 
  poisson_even_goal_prob 2.8 ≈ 0.502 :=
by
  -- Proof of the theorem
  sorry

end football_even_goal_prob_l200_200807


namespace Sheila_attends_picnic_probability_l200_200386

theorem Sheila_attends_picnic_probability :
  let P_rain := 0.5
  let P_no_rain := 0.5
  let P_Sheila_goes_if_rain := 0.3
  let P_Sheila_goes_if_no_rain := 0.7
  let P_friend_agrees := 0.5
  (P_rain * P_Sheila_goes_if_rain + P_no_rain * P_Sheila_goes_if_no_rain) * P_friend_agrees = 0.25 := 
by
  sorry

end Sheila_attends_picnic_probability_l200_200386


namespace fourth_vertex_of_square_l200_200262

def A : ℂ := 2 - 3 * Complex.I
def B : ℂ := 3 + 2 * Complex.I
def C : ℂ := -3 + 2 * Complex.I

theorem fourth_vertex_of_square : ∃ D : ℂ, 
  (D - B) = (B - A) * Complex.I ∧ 
  (D - C) = (C - A) * Complex.I ∧ 
  (D = -3 + 8 * Complex.I) :=
sorry

end fourth_vertex_of_square_l200_200262


namespace school_club_profit_l200_200431

def calculate_profit (bars_bought : ℕ) (cost_per_3_bars : ℚ) (bars_sold : ℕ) (price_per_4_bars : ℚ) : ℚ :=
  let cost_per_bar := cost_per_3_bars / 3
  let total_cost := bars_bought * cost_per_bar
  let price_per_bar := price_per_4_bars / 4
  let total_revenue := bars_sold * price_per_bar
  total_revenue - total_cost

theorem school_club_profit :
  calculate_profit 1200 1.50 1200 2.40 = 120 :=
by sorry

end school_club_profit_l200_200431


namespace generatrix_length_of_cone_l200_200867

theorem generatrix_length_of_cone (r l : ℝ) (h1 : r = real.sqrt 2) 
    (h2 : 2 * real.pi * r = real.pi * l) : l = 2 * real.sqrt 2 :=
by
  -- The proof steps would go here, but we don't need to provide them.
  sorry

end generatrix_length_of_cone_l200_200867


namespace find_sachin_age_l200_200418

-- Define Sachin's and Rahul's ages as variables
variables (S R : ℝ)

-- Define the conditions
def rahul_age := S + 9
def age_ratio := (S / R) = (7 / 9)

-- State the theorem for Sachin's age
theorem find_sachin_age (h1 : R = rahul_age S) (h2 : age_ratio S R) : S = 31.5 :=
by sorry

end find_sachin_age_l200_200418


namespace seating_arrangements_l200_200144

-- Define the participants
inductive Person : Type
| xiaoMing
| parent1
| parent2
| grandparent1
| grandparent2

open Person

-- Define the function to count seating arrangements
noncomputable def count_seating_arrangements : Nat :=
  let arrangements := [
    -- (Only one parent next to Xiao Ming, parents not next to each other)
    12,
    -- (Only one parent next to Xiao Ming, parents next to each other)
    24,
    -- (Both parents next to Xiao Ming)
    12
  ]
  arrangements.foldr (· + ·) 0

theorem seating_arrangements : count_seating_arrangements = 48 := by
  sorry

end seating_arrangements_l200_200144


namespace sqrt_meaningful_range_l200_200216

theorem sqrt_meaningful_range (x : ℝ) : 
  (x + 4) ≥ 0 ↔ x ≥ -4 :=
by sorry

end sqrt_meaningful_range_l200_200216


namespace unique_not_in_range_l200_200960

noncomputable def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_not_in_range (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0)
  (h₅ : f a b c d 23 = 23) (h₆ : f a b c d 101 = 101) (h₇ : ∀ x ≠ -d / c, f a b c d (f a b c d x) = x) :
  (a / c) = 62 := 
 sorry

end unique_not_in_range_l200_200960


namespace perimeter_of_triangle_l200_200457

def point (x y : ℝ) := (x, y)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def perimeter_triangle (a b c : ℝ × ℝ) : ℝ :=
  distance a b + distance b c + distance c a

theorem perimeter_of_triangle :
  let A := point 1 2
  let B := point 6 8
  let C := point 1 5
  perimeter_triangle A B C = Real.sqrt 61 + Real.sqrt 34 + 3 :=
by
  -- proof steps can be provided here
  sorry

end perimeter_of_triangle_l200_200457


namespace number_of_arrangements_l200_200105

def basil_plants := 2
def aloe_plants := 1
def cactus_plants := 1
def white_lamps := 2
def red_lamps := 2
def total_plants := basil_plants + aloe_plants + cactus_plants
def total_lamps := white_lamps + red_lamps

theorem number_of_arrangements : total_plants = 4 ∧ total_lamps = 4 →
  ∃ n : ℕ, n = 28 :=
by
  intro h
  sorry

end number_of_arrangements_l200_200105


namespace rectangle_area_l200_200571

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l200_200571


namespace paint_price_and_max_boxes_l200_200540

theorem paint_price_and_max_boxes (x y m : ℕ) 
  (h1 : x + 2 * y = 56) 
  (h2 : 2 * x + y = 64) 
  (h3 : 24 * m + 16 * (200 - m) ≤ 3920) : 
  x = 24 ∧ y = 16 ∧ m ≤ 90 := 
  by 
    sorry

end paint_price_and_max_boxes_l200_200540


namespace fraction_in_pairing_l200_200086

open Function

theorem fraction_in_pairing (s t : ℕ) (h : (t : ℚ) / 4 = s / 3) : 
  ((t / 4 : ℚ) + (s / 3)) / (t + s) = 2 / 7 :=
by sorry

end fraction_in_pairing_l200_200086


namespace solution_set_of_inequality_min_value_of_expression_l200_200664

def f (x : ℝ) : ℝ := |x + 1| - |2 * x - 2|

-- (I) Prove that the solution set of the inequality f(x) ≥ x - 1 is [0, 2]
theorem solution_set_of_inequality 
  (x : ℝ) : f x ≥ x - 1 ↔ 0 ≤ x ∧ x ≤ 2 := 
sorry

-- (II) Given the maximum value m of f(x) is 2 and a + b + c = 2, prove the minimum value of b^2/a + c^2/b + a^2/c is 2
theorem min_value_of_expression
  (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 2) :
  b^2 / a + c^2 / b + a^2 / c ≥ 2 :=
sorry

end solution_set_of_inequality_min_value_of_expression_l200_200664


namespace box_made_by_Bellini_or_son_l200_200953

-- Definitions of the conditions
variable (B : Prop) -- Bellini made the box
variable (S : Prop) -- Bellini's son made the box
variable (inscription_true : Prop) -- The inscription "I made this box" is truthful

-- The problem statement in Lean: Prove that B or S given the inscription is true
theorem box_made_by_Bellini_or_son (B S inscription_true : Prop) (h1 : inscription_true → (B ∨ S)) : B ∨ S :=
by
  sorry

end box_made_by_Bellini_or_son_l200_200953


namespace det_new_matrix_l200_200077

variables {a b c d : ℝ}

theorem det_new_matrix (h : a * d - b * c = 5) : (a - c) * d - (b - d) * c = 5 :=
by sorry

end det_new_matrix_l200_200077


namespace rectangle_area_l200_200568

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l200_200568


namespace aaron_ate_more_apples_l200_200387

-- Define the number of apples eaten by Aaron and Zeb
def apples_eaten_by_aaron : ℕ := 6
def apples_eaten_by_zeb : ℕ := 1

-- Theorem to prove the difference in apples eaten
theorem aaron_ate_more_apples :
  apples_eaten_by_aaron - apples_eaten_by_zeb = 5 :=
by
  sorry

end aaron_ate_more_apples_l200_200387


namespace cost_of_remaining_ingredients_l200_200125

theorem cost_of_remaining_ingredients :
  let cocoa_required := 0.4
  let sugar_required := 0.6
  let cake_weight := 450
  let given_cocoa := 259
  let cost_per_lb_cocoa := 3.50
  let cost_per_lb_sugar := 0.80
  let total_cocoa_needed := cake_weight * cocoa_required
  let total_sugar_needed := cake_weight * sugar_required
  let remaining_cocoa := max 0 (total_cocoa_needed - given_cocoa)
  let remaining_sugar := total_sugar_needed
  let total_cost := remaining_cocoa * cost_per_lb_cocoa + remaining_sugar * cost_per_lb_sugar
  total_cost = 216 := by
  sorry

end cost_of_remaining_ingredients_l200_200125


namespace car_speed_ratio_to_pedestrian_speed_l200_200617

noncomputable def ratio_of_speeds (length_pedestrian_car: ℝ) (length_bridge: ℝ) (speed_pedestrian: ℝ) (speed_car: ℝ) : ℝ :=
  speed_car / speed_pedestrian

theorem car_speed_ratio_to_pedestrian_speed
  (L : ℝ)  -- length of the bridge
  (vc vp : ℝ)  -- speeds of car and pedestrian respectively
  (h1 : (2 / 5) * L + (3 / 5) * vp = L)
  (h2 : (3 / 5) * L = vc * (L / (5 * vp))) :
  ratio_of_speeds L L vp vc = 5 :=
by
  sorry

end car_speed_ratio_to_pedestrian_speed_l200_200617


namespace jane_rejected_percentage_l200_200224

theorem jane_rejected_percentage (P : ℕ) (John_rejected : ℤ) (Jane_inspected_rejected : ℤ) :
  John_rejected = 7 * P ∧
  Jane_inspected_rejected = 5 * P ∧
  (John_rejected + Jane_inspected_rejected) = 75 * P → 
  Jane_inspected_rejected = P  :=
by sorry

end jane_rejected_percentage_l200_200224


namespace solve_abs_ineq_l200_200054

theorem solve_abs_ineq (x : ℝ) (h : x > 0) : |4 * x - 5| < 8 ↔ 0 < x ∧ x < 13 / 4 :=
by
  sorry

end solve_abs_ineq_l200_200054


namespace number_of_solutions_l200_200320

theorem number_of_solutions (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 5) :
  (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 4 → x = -2 :=
sorry

end number_of_solutions_l200_200320


namespace fraction_calculation_l200_200269

theorem fraction_calculation :
  (3 / 4) * (1 / 2) * (2 / 5) * 5060 = 759 :=
by
  sorry

end fraction_calculation_l200_200269


namespace lawn_care_company_expense_l200_200785

theorem lawn_care_company_expense (cost_blade : ℕ) (num_blades : ℕ) (cost_string : ℕ) :
  cost_blade = 8 → num_blades = 4 → cost_string = 7 → 
  (num_blades * cost_blade + cost_string = 39) :=
by
  intro h1 h2 h3
  sorry

end lawn_care_company_expense_l200_200785


namespace keiths_total_spending_l200_200932

theorem keiths_total_spending :
  let digimon_cost := 4 * 4.45
  let pokemon_cost := 3 * 5.25
  let yugioh_cost := 6 * 3.99
  let mtg_cost := 2 * 6.75
  let baseball_cost := 1 * 6.06
  let total_cost := digimon_cost + pokemon_cost + yugioh_cost + mtg_cost + baseball_cost
  total_cost = 77.05 :=
by
  let digimon_cost := 4 * 4.45
  let pokemon_cost := 3 * 5.25
  let yugioh_cost := 6 * 3.99
  let mtg_cost := 2 * 6.75
  let baseball_cost := 1 * 6.06
  let total_cost := digimon_cost + pokemon_cost + yugioh_cost + mtg_cost + baseball_cost
  have h : total_cost = 77.05 := sorry
  exact h

end keiths_total_spending_l200_200932


namespace volume_of_rectangular_box_l200_200986

theorem volume_of_rectangular_box 
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12) : 
  l * w * h = 60 :=
sorry

end volume_of_rectangular_box_l200_200986


namespace investment2_rate_l200_200428

-- Define the initial conditions
def total_investment : ℝ := 10000
def investment1 : ℝ := 4000
def rate1 : ℝ := 0.05
def investment2 : ℝ := 3500
def income1 : ℝ := investment1 * rate1
def yearly_income_goal : ℝ := 500
def remaining_investment : ℝ := total_investment - investment1 - investment2
def rate3 : ℝ := 0.064
def income3 : ℝ := remaining_investment * rate3

-- The main theorem
theorem investment2_rate (rate2 : ℝ) : 
  income1 + income3 + investment2 * (rate2 / 100) = yearly_income_goal → rate2 = 4 := 
by 
  sorry

end investment2_rate_l200_200428


namespace find_smaller_number_l200_200967

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 18) (h2 : a * b = 45) : a = 3 ∨ b = 3 := 
by 
  -- Proof would go here
  sorry

end find_smaller_number_l200_200967


namespace expression_value_l200_200135

   theorem expression_value :
     (20 - (2010 - 201)) + (2010 - (201 - 20)) = 40 := 
   by
     sorry
   
end expression_value_l200_200135


namespace finish_time_is_1_10_PM_l200_200089

-- Definitions of the problem conditions
def start_time := 9 * 60 -- 9:00 AM in minutes past midnight
def third_task_finish_time := 11 * 60 + 30 -- 11:30 AM in minutes past midnight
def num_tasks := 5
def tasks1_to_3_duration := third_task_finish_time - start_time
def one_task_duration := tasks1_to_3_duration / 3
def total_duration := one_task_duration * num_tasks

-- Statement to prove the final time when John finishes the fifth task
theorem finish_time_is_1_10_PM : 
  start_time + total_duration = 13 * 60 + 10 := 
by 
  sorry

end finish_time_is_1_10_PM_l200_200089


namespace season_duration_l200_200008

theorem season_duration (total_games : ℕ) (games_per_month : ℕ) (h1 : total_games = 323) (h2 : games_per_month = 19) :
  (total_games / games_per_month) = 17 :=
by
  sorry

end season_duration_l200_200008


namespace least_non_lucky_multiple_of_12_l200_200604

/- Defines what it means for a number to be a lucky integer -/
def isLucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

/- Proves the least positive multiple of 12 that is not a lucky integer is 96 -/
theorem least_non_lucky_multiple_of_12 : ∃ n, n % 12 = 0 ∧ ¬isLucky n ∧ ∀ m, m % 12 = 0 ∧ ¬isLucky m → n ≤ m :=
  by
  sorry

end least_non_lucky_multiple_of_12_l200_200604


namespace eccentricity_of_ellipse_l200_200853

open Real

theorem eccentricity_of_ellipse 
  (O B F : ℝ × ℝ)
  (a b : ℝ) 
  (h_a_gt_b: a > b)
  (h_b_gt_0: b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h_OB_eq_OF : dist O B = dist O F)
  (O_is_origin : O = (0,0))
  (B_is_upper_vertex : B = (0, b))
  (F_is_right_focus : F = (c, 0) ∧ c = Real.sqrt (a^2 - b^2)) :
 (c / a = sqrt 2 / 2)
:=
sorry

end eccentricity_of_ellipse_l200_200853


namespace sales_quota_50_l200_200466

theorem sales_quota_50 :
  let cars_sold_first_three_days := 5 * 3
  let cars_sold_next_four_days := 3 * 4
  let additional_cars_needed := 23
  let total_quota := cars_sold_first_three_days + cars_sold_next_four_days + additional_cars_needed
  total_quota = 50 :=
by
  -- proof goes here
  sorry

end sales_quota_50_l200_200466


namespace part1_part2_l200_200845

-- Definitions of sets A and B
def A (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < a + 1 }
def B : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 ≥ 0 }

-- Proving the first condition
theorem part1 (a : ℝ) : (A a ∩ B = ∅) ∧ (A a ∪ B = Set.univ) ↔ a = 2 :=
by
  sorry

-- Proving the second condition
theorem part2 (a : ℝ) : (A a ⊆ B) ↔ (a ≤ 0 ∨ a ≥ 4) :=
by
  sorry

end part1_part2_l200_200845


namespace repeating_decmials_sum_is_fraction_l200_200451

noncomputable def x : ℚ := 2/9
noncomputable def y : ℚ := 2/99
noncomputable def z : ℚ := 2/9999

theorem repeating_decmials_sum_is_fraction :
  (x + y + z) = 2426 / 9999 := by
  sorry

end repeating_decmials_sum_is_fraction_l200_200451


namespace cara_total_bread_l200_200064

theorem cara_total_bread 
  (d : ℕ) (L : ℕ) (B : ℕ) (S : ℕ) 
  (h_dinner : d = 240) 
  (h_lunch : d = 8 * L) 
  (h_breakfast : d = 6 * B) 
  (h_snack : d = 4 * S) : 
  d + L + B + S = 370 := 
sorry

end cara_total_bread_l200_200064


namespace find_k_l200_200207

theorem find_k (k : ℕ) : (1 / 3)^32 * (1 / 125)^k = 1 / 27^32 → k = 0 :=
by {
  sorry
}

end find_k_l200_200207


namespace tan_sin_identity_l200_200314

noncomputable def tan_deg : ℝ := tan (real.pi / 6)
noncomputable def sin_deg : ℝ := sin (real.pi / 6)

theorem tan_sin_identity :
  (tan_deg ^ 2 - sin_deg ^ 2) / (tan_deg ^ 2 * sin_deg ^ 2) = 1 :=
by
  have tan_30 := by
    simp [tan_deg, real.tan_pi_div_six]
    norm_num
  have sin_30 := by
    simp [sin_deg, real.sin_pi_div_six]
    norm_num
  sorry

end tan_sin_identity_l200_200314


namespace car_speed_ratio_to_pedestrian_speed_l200_200616

noncomputable def ratio_of_speeds (length_pedestrian_car: ℝ) (length_bridge: ℝ) (speed_pedestrian: ℝ) (speed_car: ℝ) : ℝ :=
  speed_car / speed_pedestrian

theorem car_speed_ratio_to_pedestrian_speed
  (L : ℝ)  -- length of the bridge
  (vc vp : ℝ)  -- speeds of car and pedestrian respectively
  (h1 : (2 / 5) * L + (3 / 5) * vp = L)
  (h2 : (3 / 5) * L = vc * (L / (5 * vp))) :
  ratio_of_speeds L L vp vc = 5 :=
by
  sorry

end car_speed_ratio_to_pedestrian_speed_l200_200616


namespace non_real_roots_bounded_l200_200909

theorem non_real_roots_bounded (b : ℝ) :
  (∀ x : ℝ, polynomial.has_roots (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2) → ¬ real.is_root (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2)) → 
  b ∈ set.Ioo (-(8 : ℝ)) 8 :=
by sorry

end non_real_roots_bounded_l200_200909


namespace find_k_l200_200227

-- Define the set A using a condition on the quadratic equation
def A (k : ℝ) : Set ℝ := {x | k * x ^ 2 + 4 * x + 4 = 0}

-- Define the condition for the set A to have exactly one element
def has_exactly_one_element (k : ℝ) : Prop :=
  ∃ x : ℝ, A k = {x}

-- The problem statement is to find the value of k for which A has exactly one element
theorem find_k : ∃ k : ℝ, has_exactly_one_element k ∧ k = 1 :=
by
  simp [has_exactly_one_element, A]
  sorry

end find_k_l200_200227


namespace series_sum_l200_200317

theorem series_sum :
  ∑' n : ℕ,  n ≠ 0 → (6 * n + 2) / ((6 * n - 1)^2 * (6 * n + 5)^2) = 1 / 600 := by
  sorry

end series_sum_l200_200317


namespace green_peaches_in_each_basket_l200_200179

theorem green_peaches_in_each_basket (G : ℕ) 
  (h1 : ∀ B : ℕ, B = 15) 
  (h2 : ∀ R : ℕ, R = 19) 
  (h3 : ∀ P : ℕ, P = 345) 
  (h_eq : 345 = 15 * (19 + G)) : 
  G = 4 := by
  sorry

end green_peaches_in_each_basket_l200_200179


namespace final_toy_count_correct_l200_200175

def initial_toy_count : ℝ := 5.3
def tuesday_toys_left (initial: ℝ) : ℝ := initial * 0.605
def tuesday_new_toys : ℝ := 3.6
def wednesday_toys_left (tuesday_total: ℝ) : ℝ := tuesday_total * 0.498
def wednesday_new_toys : ℝ := 2.4
def thursday_toys_left (wednesday_total: ℝ) : ℝ := wednesday_total * 0.692
def thursday_new_toys : ℝ := 4.5

def total_toys (initial: ℝ) : ℝ :=
  let after_tuesday := tuesday_toys_left initial + tuesday_new_toys
  let after_wednesday := wednesday_toys_left after_tuesday + wednesday_new_toys
  let after_thursday := thursday_toys_left after_wednesday + thursday_new_toys
  after_thursday

def toys_lost_tuesday (initial: ℝ) (left: ℝ) : ℝ := initial - left
def toys_lost_wednesday (tuesday_total: ℝ) (left: ℝ) : ℝ := tuesday_total - left
def toys_lost_thursday (wednesday_total: ℝ) (left: ℝ) : ℝ := wednesday_total - left
def total_lost_toys (initial: ℝ) : ℝ :=
  let tuesday_left := tuesday_toys_left initial
  let tuesday_total := tuesday_left + tuesday_new_toys
  let wednesday_left := wednesday_toys_left tuesday_total
  let wednesday_total := wednesday_left + wednesday_new_toys
  let thursday_left := thursday_toys_left wednesday_total
  let lost_tuesday := toys_lost_tuesday initial tuesday_left
  let lost_wednesday := toys_lost_wednesday tuesday_total wednesday_left
  let lost_thursday := toys_lost_thursday wednesday_total thursday_left
  lost_tuesday + lost_wednesday + lost_thursday

def final_toy_count (initial: ℝ) : ℝ :=
  let current_toys := total_toys initial
  let lost_toys := total_lost_toys initial
  current_toys + lost_toys

theorem final_toy_count_correct :
  final_toy_count initial_toy_count = 15.8 := sorry

end final_toy_count_correct_l200_200175


namespace calculate_expression_value_l200_200821

theorem calculate_expression_value :
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 :=
by
  sorry

end calculate_expression_value_l200_200821


namespace original_number_is_3199_l200_200987

theorem original_number_is_3199 (n : ℕ) (k : ℕ) (h1 : k = 3200) (h2 : (n + k) % 8 = 0) : n = 3199 :=
sorry

end original_number_is_3199_l200_200987


namespace circle_center_radius_sum_18_l200_200934

-- Conditions from the problem statement
def circle_eq (x y : ℝ) : Prop := x^2 + 2 * y - 9 = -y^2 + 18 * x + 9

-- Goal is to prove a + b + r = 18
theorem circle_center_radius_sum_18 :
  (∃ a b r : ℝ, 
     (∀ x y : ℝ, circle_eq x y ↔ (x - a)^2 + (y - b)^2 = r^2) ∧ 
     a + b + r = 18) :=
sorry

end circle_center_radius_sum_18_l200_200934


namespace base_7_minus_base_8_l200_200326

def convert_base_7 (n : ℕ) : ℕ :=
  match n with
  | 543210 => 5 * 7^5 + 4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0
  | _ => 0

def convert_base_8 (n : ℕ) : ℕ :=
  match n with
  | 45321 => 4 * 8^4 + 5 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1 * 8^0
  | _ => 0

theorem base_7_minus_base_8 : convert_base_7 543210 - convert_base_8 45321 = 75620 := by
  sorry

end base_7_minus_base_8_l200_200326


namespace probability_average_is_five_l200_200648

-- Definitions and conditions
def numbers : List ℕ := [1, 3, 4, 6, 7, 9]

def average_is_five (a b : ℕ) : Prop := (a + b) / 2 = 5

-- Desired statement
theorem probability_average_is_five : 
  ∃ p : ℚ, p = 1 / 5 ∧ (∃ a b : ℕ, a ∈ numbers ∧ b ∈ numbers ∧ average_is_five a b) := 
sorry

end probability_average_is_five_l200_200648


namespace sixth_number_of_11_consecutive_odd_sum_1991_is_181_l200_200676

theorem sixth_number_of_11_consecutive_odd_sum_1991_is_181 :
  (∃ (n : ℤ), (2 * n + 1) + (2 * n + 3) + (2 * n + 5) + (2 * n + 7) + (2 * n + 9) + (2 * n + 11) + (2 * n + 13) + (2 * n + 15) + (2 * n + 17) + (2 * n + 19) + (2 * n + 21) = 1991) →
  2 * 85 + 11 = 181 := 
by
  sorry

end sixth_number_of_11_consecutive_odd_sum_1991_is_181_l200_200676


namespace production_growth_rate_eq_l200_200187

theorem production_growth_rate_eq 
  (x : ℝ)
  (H : 100 + 100 * (1 + x) + 100 * (1 + x) ^ 2 = 364) : 
  100 + 100 * (1 + x) + 100 * (1 + x) ^ 2 = 364 :=
by {
  sorry
}

end production_growth_rate_eq_l200_200187


namespace opposite_of_half_l200_200119

theorem opposite_of_half : - (1 / 2) = -1 / 2 := 
by
  sorry

end opposite_of_half_l200_200119


namespace calculate_expression_value_l200_200820

theorem calculate_expression_value :
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 :=
by
  sorry

end calculate_expression_value_l200_200820


namespace Dexter_card_count_l200_200322

theorem Dexter_card_count : 
  let basketball_boxes := 9
  let cards_per_basketball_box := 15
  let football_boxes := basketball_boxes - 3
  let cards_per_football_box := 20
  let basketball_cards := basketball_boxes * cards_per_basketball_box
  let football_cards := football_boxes * cards_per_football_box
  let total_cards := basketball_cards + football_cards
  total_cards = 255 :=
sorry

end Dexter_card_count_l200_200322


namespace calculate_expression_l200_200826

theorem calculate_expression : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := 
by
  sorry

end calculate_expression_l200_200826


namespace task1_task2_task3_task4_l200_200609

-- Definitions of the given conditions
def cost_price : ℝ := 16
def selling_price_range (x : ℝ) : Prop := 16 ≤ x ∧ x ≤ 48
def init_selling_price : ℝ := 20
def init_sales_volume : ℝ := 360
def decreasing_sales_rate : ℝ := 10
def daily_sales_vol (x : ℝ) : ℝ := 360 - 10 * (x - 20)
def daily_total_profit (x : ℝ) (y : ℝ) : ℝ := y * (x - cost_price)

-- Proof task (1)
theorem task1 : daily_sales_vol 25 = 310 ∧ daily_total_profit 25 (daily_sales_vol 25) = 2790 := 
by 
    -- Your code here
    sorry

-- Proof task (2)
theorem task2 : ∀ x, daily_sales_vol x = -10 * x + 560 := 
by 
    -- Your code here
    sorry

-- Proof task (3)
theorem task3 : ∀ x, 
    W = (x - 16) * (daily_sales_vol x) 
    ∧ W = -10 * x ^ 2 + 720 * x - 8960 
    ∧ (∃ x, -10 * x ^ 2 + 720 * x - 8960 = 4000 ∧ selling_price_range x) := 
by 
    -- Your code here 
    sorry

-- Proof task (4)
theorem task4 : ∃ x, 
    -10 * (x - 36) ^ 2 + 4000 = 3000 
    ∧ selling_price_range x 
    ∧ (x = 26 ∨ x = 46) := 
by 
    -- Your code here 
    sorry

end task1_task2_task3_task4_l200_200609


namespace find_number_of_children_l200_200594

theorem find_number_of_children (C B : ℕ) (H1 : B = 2 * C) (H2 : B = 4 * (C - 360)) : C = 720 := 
by
  sorry

end find_number_of_children_l200_200594


namespace find_value_a_prove_inequality_l200_200471

noncomputable def arithmetic_sequence (a : ℕ) (S : ℕ → ℕ) (a_n : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 2 → S n * S n = 3 * n ^ 2 * a_n n + S (n - 1) * S (n - 1) ∧ a_n n ≠ 0

theorem find_value_a {S : ℕ → ℕ} {a_n : ℕ → ℕ} :
  (∃ (a : ℕ), arithmetic_sequence a S a_n) → a = 3 :=
sorry

noncomputable def sequence_bn (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) :=
  ∀ n : ℕ, b_n n = 1 / ((a_n n - 1) * (a_n n + 2))

theorem prove_inequality {S : ℕ → ℕ} {a_n : ℕ → ℕ} {b_n : ℕ → ℕ} {T : ℕ → ℕ} :
  (∃ (a : ℕ), arithmetic_sequence a S a_n) →
  (sequence_bn a_n b_n) →
  ∀ n : ℕ, T n < 1 / 6 :=
sorry

end find_value_a_prove_inequality_l200_200471


namespace sum_squares_bound_l200_200517

theorem sum_squares_bound (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) : 
  ab + bc + cd + da ≤ 4 :=
sorry

end sum_squares_bound_l200_200517


namespace clay_boys_proof_l200_200223

variable (total_students : ℕ)
variable (total_boys : ℕ)
variable (total_girls : ℕ)
variable (jonas_students : ℕ)
variable (clay_students : ℕ)
variable (birch_students : ℕ)
variable (jonas_boys : ℕ)
variable (birch_girls : ℕ)

noncomputable def boys_from_clay (total_students total_boys total_girls jonas_students clay_students birch_students jonas_boys birch_girls : ℕ) : ℕ :=
  let birch_boys := birch_students - birch_girls
  let clay_boys := total_boys - (jonas_boys + birch_boys)
  clay_boys

theorem clay_boys_proof (h1 : total_students = 180) (h2 : total_boys = 94) 
    (h3 : total_girls = 86) (h4 : jonas_students = 60) 
    (h5 : clay_students = 80) (h6 : birch_students = 40) 
    (h7 : jonas_boys = 30) (h8 : birch_girls = 24) : 
  boys_from_clay total_students total_boys total_girls jonas_students clay_students birch_students jonas_boys birch_girls = 48 := 
by 
  simp [boys_from_clay] 
  sorry

end clay_boys_proof_l200_200223


namespace orcs_per_squad_is_eight_l200_200390

-- Defining the conditions
def total_weight_of_swords := 1200
def weight_each_orc_can_carry := 15
def number_of_squads := 10

-- Proof statement to demonstrate the answer
theorem orcs_per_squad_is_eight :
  (total_weight_of_swords / weight_each_orc_can_carry) / number_of_squads = 8 := by
  sorry

end orcs_per_squad_is_eight_l200_200390


namespace add_fractions_l200_200181

-- Define the two fractions
def frac1 := 7 / 8
def frac2 := 9 / 12

-- The problem: addition of the two fractions and expressing in simplest form
theorem add_fractions : frac1 + frac2 = (13 : ℚ) / 8 := 
by 
  sorry

end add_fractions_l200_200181


namespace typing_time_l200_200265

theorem typing_time (typing_speed : ℕ) (words_per_page : ℕ) (number_of_pages : ℕ) (h1 : typing_speed = 90) (h2 : words_per_page = 450) (h3 : number_of_pages = 10) : 
  (words_per_page / typing_speed) * number_of_pages = 50 := 
by
  sorry

end typing_time_l200_200265


namespace physics_marks_l200_200992

theorem physics_marks (P C M : ℕ) 
  (h1 : (P + C + M) = 255)
  (h2 : (P + M) = 180)
  (h3 : (P + C) = 140) : 
  P = 65 :=
by
  sorry

end physics_marks_l200_200992


namespace problem_l200_200228

def remainder_when_divided_by_20 (a b : ℕ) : ℕ := (a + b) % 20

theorem problem (a b : ℕ) (n m : ℤ) (h1 : a = 60 * n + 53) (h2 : b = 50 * m + 24) : 
  remainder_when_divided_by_20 a b = 17 := 
by
  -- Proof would go here
  sorry

end problem_l200_200228


namespace simplify_expression_l200_200168

variable (a : ℝ)

theorem simplify_expression : 
  (a^2 / (a^(1/2) * a^(2/3))) = a^(5/6) :=
by
  sorry

end simplify_expression_l200_200168


namespace final_song_count_l200_200620

theorem final_song_count {init_songs added_songs removed_songs doubled_songs final_songs : ℕ} 
    (h1 : init_songs = 500)
    (h2 : added_songs = 500)
    (h3 : doubled_songs = (init_songs + added_songs) * 2)
    (h4 : removed_songs = 50)
    (h_final : final_songs = doubled_songs - removed_songs) : 
    final_songs = 2950 :=
by
  sorry

end final_song_count_l200_200620


namespace find_height_of_door_l200_200927

noncomputable def height_of_door (x : ℝ) (w : ℝ) (h : ℝ) : ℝ := h

theorem find_height_of_door :
  ∃ x w h, (w = x - 4) ∧ (h = x - 2) ∧ (x^2 = w^2 + h^2) ∧ height_of_door x w h = 8 :=
by {
  sorry
}

end find_height_of_door_l200_200927


namespace largest_value_is_E_l200_200140

-- Define the given values
def A := 1 - 0.1
def B := 1 - 0.01
def C := 1 - 0.001
def D := 1 - 0.0001
def E := 1 - 0.00001

-- Main theorem statement
theorem largest_value_is_E : E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  sorry

end largest_value_is_E_l200_200140


namespace helicopter_rental_cost_l200_200131

theorem helicopter_rental_cost
  (hours_per_day : ℕ)
  (total_days : ℕ)
  (total_cost : ℕ)
  (H1 : hours_per_day = 2)
  (H2 : total_days = 3)
  (H3 : total_cost = 450) :
  total_cost / (hours_per_day * total_days) = 75 :=
by
  sorry

end helicopter_rental_cost_l200_200131


namespace rowing_speed_in_still_water_l200_200028

variable (v c t : ℝ)
variable (h1 : c = 1.3)
variable (h2 : 2 * ((v - c) * t) = ((v + c) * t))

theorem rowing_speed_in_still_water : v = 3.9 := by
  sorry

end rowing_speed_in_still_water_l200_200028


namespace minimize_shoes_l200_200380

-- Definitions for inhabitants, one-legged inhabitants, and shoe calculations
def total_inhabitants := 10000
def P (percent_one_legged : ℕ) := (percent_one_legged * total_inhabitants) / 100
def non_one_legged (percent_one_legged : ℕ) := total_inhabitants - (P percent_one_legged)
def non_one_legged_with_shoes (percent_one_legged : ℕ) := (non_one_legged percent_one_legged) / 2
def shoes_needed (percent_one_legged : ℕ) := 
  (P percent_one_legged) + 2 * (non_one_legged_with_shoes percent_one_legged)

-- Theorem to prove that 100% one-legged minimizes the shoes required
theorem minimize_shoes : ∀ (percent_one_legged : ℕ), shoes_needed percent_one_legged = total_inhabitants → percent_one_legged = 100 :=
by
  intros percent_one_legged h
  sorry

end minimize_shoes_l200_200380


namespace bruno_initial_books_l200_200040

theorem bruno_initial_books :
  ∃ (B : ℕ), B - 4 + 10 = 39 → B = 33 :=
by
  use 33
  intro h
  linarith [h]

end bruno_initial_books_l200_200040


namespace distinct_elements_triangle_not_isosceles_l200_200482

theorem distinct_elements_triangle_not_isosceles
  {a b c : ℝ} (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  ¬(a = b ∨ b = c ∨ a = c) := by
  sorry

end distinct_elements_triangle_not_isosceles_l200_200482


namespace remainder_proof_l200_200139

noncomputable def problem (n : ℤ) : Prop :=
  n % 9 = 4

noncomputable def solution (n : ℤ) : ℤ :=
  (4 * n - 11) % 9

theorem remainder_proof (n : ℤ) (h : problem n) : solution n = 5 := by
  sorry

end remainder_proof_l200_200139


namespace find_f_1_l200_200206

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_1 : (∀ x : ℝ, f x + 3 * f (-x) = Real.logb 2 (x + 3)) → f 1 = 1 / 8 := 
by 
  sorry

end find_f_1_l200_200206


namespace max_units_of_material_A_l200_200990

theorem max_units_of_material_A (x y z : ℕ) 
    (h1 : 3 * x + 5 * y + 7 * z = 62)
    (h2 : 2 * x + 4 * y + 6 * z = 50) : x ≤ 5 :=
by
    sorry 

end max_units_of_material_A_l200_200990


namespace math_problem_l200_200041

theorem math_problem : 1999^2 - 2000 * 1998 = 1 := 
by
  sorry

end math_problem_l200_200041


namespace rectangle_area_l200_200572

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l200_200572


namespace geometric_sequence_first_term_l200_200660

theorem geometric_sequence_first_term
  (a : ℕ → ℝ) -- sequence a_n
  (r : ℝ) -- common ratio
  (h1 : r = 2) -- given common ratio
  (h2 : a 4 = 16) -- given a_4 = 16
  (h3 : ∀ n, a n = a 1 * r^(n-1)) -- definition of geometric sequence
  : a 1 = 2 := 
sorry

end geometric_sequence_first_term_l200_200660


namespace central_angle_of_sector_l200_200496

-- Define the given conditions
def radius : ℝ := 10
def area : ℝ := 100

-- The statement to be proved
theorem central_angle_of_sector (α : ℝ) (h : area = (1 / 2) * α * radius ^ 2) : α = 2 :=
by
  sorry

end central_angle_of_sector_l200_200496


namespace max_profit_l200_200433

-- Define the given conditions
def cost_price : ℝ := 80
def sales_relationship (x : ℝ) : ℝ := -0.5 * x + 160
def selling_price_range (x : ℝ) : Prop := 120 ≤ x ∧ x ≤ 180

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_relationship x

-- The goal: prove the maximum profit and the selling price that achieves it
theorem max_profit : ∃ (x : ℝ), selling_price_range x ∧ profit x = 7000 := 
  sorry

end max_profit_l200_200433


namespace clock_correct_time_fraction_l200_200153

/-- A 12-hour digital clock problem:
A 12-hour digital clock displays the hour and minute of a day.
Whenever it is supposed to display a '1' or a '2', it mistakenly displays a '9'.
The fraction of the day during which the clock shows the correct time is 7/24.
-/
theorem clock_correct_time_fraction : (7 : ℚ) / 24 = 7 / 24 :=
by sorry

end clock_correct_time_fraction_l200_200153


namespace football_match_even_goals_l200_200795

noncomputable def poisson_even_probability (λ : ℝ) : ℝ :=
  (1 + Real.exp (-2 * λ)) / 2

theorem football_match_even_goals :
  poisson_even_probability 2.8 ≈ 0.502 :=
by
  -- Proof skipped
  sorry

end football_match_even_goals_l200_200795


namespace binomial_coefficient_multiple_of_4_l200_200462

theorem binomial_coefficient_multiple_of_4 :
  ∃ (S : Finset ℕ), (∀ k ∈ S, 0 ≤ k ∧ k ≤ 2014 ∧ (Nat.choose 2014 k) % 4 = 0) ∧ S.card = 991 :=
sorry

end binomial_coefficient_multiple_of_4_l200_200462


namespace rectangle_area_l200_200569

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l200_200569


namespace max_profit_l200_200583

-- Define the conditions
def profit (m : ℝ) := (m - 8) * (900 - 15 * m)

-- State the theorem
theorem max_profit (m : ℝ) : 
  ∃ M, M = profit 34 ∧ ∀ x, profit x ≤ M :=
by
  -- the proof goes here
  sorry

end max_profit_l200_200583


namespace mileage_on_city_streets_l200_200101

-- Defining the given conditions
def distance_on_highways : ℝ := 210
def mileage_on_highways : ℝ := 35
def total_gas_used : ℝ := 9
def distance_on_city_streets : ℝ := 54

-- Proving the mileage on city streets
theorem mileage_on_city_streets :
  ∃ x : ℝ, 
    (distance_on_highways / mileage_on_highways + distance_on_city_streets / x = total_gas_used)
    ∧ x = 18 :=
by
  sorry

end mileage_on_city_streets_l200_200101


namespace cost_price_of_article_l200_200300

theorem cost_price_of_article (SP CP : ℝ) (h1 : SP = 150) (h2 : SP = CP + (1 / 4) * CP) : CP = 120 :=
by
  sorry

end cost_price_of_article_l200_200300


namespace smaller_circle_circumference_l200_200780

theorem smaller_circle_circumference (r r2 : ℝ) : 
  (60:ℝ) / 360 * 2 * Real.pi * r = 8 →
  r = 24 / Real.pi →
  1 / 4 * (24 / Real.pi)^2 = (24 / Real.pi - 2 * r2) * (24 / Real.pi) →
  2 * Real.pi * r2 = 36 :=
  by
    intros h1 h2 h3
    sorry

end smaller_circle_circumference_l200_200780


namespace toms_age_ratio_l200_200267

variable (T N : ℕ)

def toms_age_condition : Prop :=
  T = 3 * (T - 4 * N) + N

theorem toms_age_ratio (h : toms_age_condition T N) : T / N = 11 / 2 :=
by sorry

end toms_age_ratio_l200_200267


namespace eighteen_women_time_l200_200674

theorem eighteen_women_time (h : ∀ (n : ℕ), n = 6 → ∀ (t : ℕ), t = 60 → true) : ∀ (n : ℕ), n = 18 → ∀ (t : ℕ), t = 20 → true :=
by
  sorry

end eighteen_women_time_l200_200674


namespace cone_generatrix_length_l200_200868

/-- Given that the base radius of a cone is √2, 
    and its lateral surface is unfolded into a semicircle,
    prove that the length of the generatrix of the cone is 2√2. -/
theorem cone_generatrix_length (r l : ℝ) 
  (h1 : r = real.sqrt 2)
  (h2 : 2 * real.pi * r = real.pi * l) :
  l = 2 * real.sqrt 2 :=
by
  sorry -- Proof steps go here (not required in the current context)


end cone_generatrix_length_l200_200868


namespace fourth_term_in_arithmetic_sequence_l200_200081

theorem fourth_term_in_arithmetic_sequence (a d : ℝ) (h : 2 * a + 6 * d = 20) : a + 3 * d = 10 :=
sorry

end fourth_term_in_arithmetic_sequence_l200_200081


namespace probability_even_goals_is_approximately_l200_200791

noncomputable def probability_even_goals (λ : ℝ) : ℝ :=
  let p : ℝ := ∑ k in (nat.filter even), (λ ^ k * real.exp (-λ)) / (nat.fact k)
  in p

def λ : ℝ := 2.8

theorem probability_even_goals_is_approximately:
  probability_even_goals λ ≈ 0.502 :=
sorry

end probability_even_goals_is_approximately_l200_200791


namespace cocoa_powder_total_l200_200403

variable (already_has : ℕ) (still_needs : ℕ)

theorem cocoa_powder_total (h₁ : already_has = 259) (h₂ : still_needs = 47) : already_has + still_needs = 306 :=
by
  sorry

end cocoa_powder_total_l200_200403


namespace evaluate_expression_l200_200835

theorem evaluate_expression : (527 * 527 - 526 * 528) = 1 := by
  sorry

end evaluate_expression_l200_200835


namespace adiabatic_compression_work_l200_200198

noncomputable def adiabatic_work (p1 V1 V2 k : ℝ) (h₁ : k > 1) (h₂ : V1 > 0) (h₃ : V2 > 0) : ℝ :=
  (p1 * V1) / (k - 1) * (1 - (V1 / V2)^(k - 1))

theorem adiabatic_compression_work (p1 V1 V2 k W : ℝ) (h₁ : k > 1) (h₂ : V1 > 0) (h₃ : V2 > 0) :
  W = adiabatic_work p1 V1 V2 k h₁ h₂ h₃ :=
sorry

end adiabatic_compression_work_l200_200198


namespace sum_first_last_l200_200114

theorem sum_first_last (A B C D : ℕ) (h1 : (A + B + C) / 3 = 6) (h2 : (B + C + D) / 3 = 5) (h3 : D = 4) : A + D = 11 :=
by
  sorry

end sum_first_last_l200_200114


namespace percentage_increase_l200_200260

theorem percentage_increase (original_price new_price : ℝ) (h₀ : original_price = 300) (h₁ : new_price = 420) :
  ((new_price - original_price) / original_price) * 100 = 40 :=
by
  -- Insert the proof here
  sorry

end percentage_increase_l200_200260


namespace smaller_successive_number_l200_200995

noncomputable def solve_successive_numbers : ℕ :=
  let n := 51
  n

theorem smaller_successive_number (n : ℕ) (h : n * (n + 1) = 2652) : n = solve_successive_numbers :=
  sorry

end smaller_successive_number_l200_200995


namespace delores_remaining_money_l200_200176

variable (delores_money : ℕ := 450)
variable (computer_price : ℕ := 1000)
variable (computer_discount : ℝ := 0.30)
variable (printer_price : ℕ := 100)
variable (printer_tax_rate : ℝ := 0.15)
variable (table_price_euros : ℕ := 200)
variable (exchange_rate : ℝ := 1.2)

def computer_sale_price : ℝ := computer_price * (1 - computer_discount)
def printer_total_cost : ℝ := printer_price * (1 + printer_tax_rate)
def table_cost_dollars : ℝ := table_price_euros * exchange_rate
def total_cost : ℝ := computer_sale_price + printer_total_cost + table_cost_dollars
def remaining_money : ℝ := delores_money - total_cost

theorem delores_remaining_money : remaining_money = -605 := by
  sorry

end delores_remaining_money_l200_200176


namespace find_certain_number_l200_200349

theorem find_certain_number (h1 : 213 * 16 = 3408) (x : ℝ) (h2 : x * 2.13 = 0.03408) : x = 0.016 :=
by
  sorry

end find_certain_number_l200_200349


namespace cubic_has_three_zeros_l200_200398

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem cubic_has_three_zeros : (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) :=
sorry

end cubic_has_three_zeros_l200_200398


namespace number_of_bottle_caps_l200_200394

def total_cost : ℝ := 25
def cost_per_bottle_cap : ℝ := 5

theorem number_of_bottle_caps : total_cost / cost_per_bottle_cap = 5 := 
by 
  sorry

end number_of_bottle_caps_l200_200394


namespace price_of_paint_models_max_boxes_of_paint_A_l200_200532

-- Define the conditions as hypotheses
def price_paint_model_A_B (x y : ℕ) : Prop :=
  x + 2 * y = 56 ∧ 2 * x + y = 64

def total_cost_constraint (x y m : ℕ) : Prop :=
  24 * m + 16 * (200 - m) ≤ 3920

-- Prove the prices of the paint models given the conditions
theorem price_of_paint_models :
  ∃ (x y : ℕ), price_paint_model_A_B x y ∧ x = 24 ∧ y = 16 :=
by
  sorry

-- Prove the maximum number of boxes of paint model A the school can purchase given the total cost constraint
theorem max_boxes_of_paint_A :
  ∃ (m : ℕ), total_cost_constraint 24 16 m ∧ m = 90 :=
by
  sorry

end price_of_paint_models_max_boxes_of_paint_A_l200_200532


namespace correctTechnologyUsedForVolcanicAshMonitoring_l200_200000

-- Define the choices
inductive Technology
| RemoteSensing : Technology
| GPS : Technology
| GIS : Technology
| DigitalEarth : Technology

-- Define the problem conditions
def primaryTechnologyUsedForVolcanicAshMonitoring := Technology.RemoteSensing

-- The statement to prove
theorem correctTechnologyUsedForVolcanicAshMonitoring : primaryTechnologyUsedForVolcanicAshMonitoring = Technology.RemoteSensing :=
by
  sorry

end correctTechnologyUsedForVolcanicAshMonitoring_l200_200000


namespace expression_simplifies_to_49_l200_200895

theorem expression_simplifies_to_49 (x : ℝ) : 
  (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end expression_simplifies_to_49_l200_200895


namespace sticker_probability_l200_200739

theorem sticker_probability 
  (n : ℕ) (k : ℕ) (uncollected : ℕ) (collected : ℕ) (C : ℕ → ℕ → ℕ) :
  n = 18 → k = 10 → uncollected = 6 → collected = 12 → 
  (C uncollected uncollected) * (C collected (k - uncollected)) = 495 → 
  C n k = 43758 → 
  (495 : ℚ) / 43758 = 5 / 442 := 
by
  intros h_n h_k h_uncollected h_collected h_C1 h_C2
  sorry

end sticker_probability_l200_200739


namespace altitudes_bounded_by_perimeter_l200_200353

theorem altitudes_bounded_by_perimeter (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a + b + c = 2) :
  ¬ (∀ (ha hb hc : ℝ), ha = 2 / a * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     hb = 2 / b * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     hc = 2 / c * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     ha > 1 / Real.sqrt 3 ∧ 
                     hb > 1 / Real.sqrt 3 ∧ 
                     hc > 1 / Real.sqrt 3 ) :=
sorry

end altitudes_bounded_by_perimeter_l200_200353


namespace multiply_expression_l200_200229

theorem multiply_expression (x : ℝ) : 
  (x^4 + 49 * x^2 + 2401) * (x^2 - 49) = x^6 - 117649 :=
by
  sorry

end multiply_expression_l200_200229


namespace petya_wins_last_l200_200485

--- Definitions and conditions
variables {a b c : ℝ}
variable h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a
variable h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
variable discriminant : ℝ × ℝ × ℝ → ℝ := λ (coeffs : ℝ × ℝ × ℝ), coeffs.2^2 - 4 * coeffs.1 * coeffs.3

noncomputable def petya_turn (coeffs : ℝ × ℝ × ℝ) : Prop :=
discriminant coeffs ≥ 0

noncomputable def vasya_turn (coeffs : ℝ × ℝ × ℝ) : Prop :=
discriminant coeffs < 0

variable order : {coeffs // petya_turn coeffs} ⊕ {coeffs // vasya_turn coeffs}  -- permutation results assuming order in such way 5 already settled
variable sequence : list ({coeffs // petya_turn coeffs} ⊕ {coeffs // vasya_turn coeffs})

axiom petya_three_first : sequence.take 3 = [sum.inl _, sum.inl _, sum.inl _]
axiom vasya_two_next : sequence.drop 3 = [sum.inr _, sum.inr _]

theorem petya_wins_last [inhabited sequence] :
  sequence.length = 5 → sum.inl (_ : {coeffs // petya_turn coeffs}) ∈ (sequence.take 6).nth_le 5 sorry :=
sorry

end petya_wins_last_l200_200485


namespace initial_value_l200_200133

theorem initial_value (x k : ℤ) (h : x + 335 = k * 456) : x = 121 := sorry

end initial_value_l200_200133


namespace find_x_l200_200661

open Real

def vector (a b : ℝ) : ℝ × ℝ := (a, b)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

def problem_statement (x : ℝ) : Prop :=
  let m := vector 2 x
  let n := vector 4 (-2)
  let m_minus_n := vector (2 - 4) (x - (-2))
  perpendicular m m_minus_n → x = -1 + sqrt 5 ∨ x = -1 - sqrt 5

-- We assert the theorem based on the problem statement
theorem find_x (x : ℝ) : problem_statement x :=
  sorry

end find_x_l200_200661


namespace time_reduced_fraction_l200_200026

theorem time_reduced_fraction
  (T : ℝ)
  (V : ℝ)
  (hV : V = 42)
  (D : ℝ)
  (hD_1 : D = V * T)
  (V' : ℝ)
  (hV' : V' = V + 21)
  (T' : ℝ)
  (hD_2 : D = V' * T') :
  (T - T') / T = 1 / 3 :=
by
  -- Proof omitted
  sorry

end time_reduced_fraction_l200_200026


namespace find_least_positive_n_l200_200840

theorem find_least_positive_n (n : ℕ) : 
  let m := 143
  m = 11 * 13 → 
  (3^5 ≡ 1 [MOD m^2]) →
  (3^39 ≡ 1 [MOD (13^2)]) →
  n = 195 :=
sorry

end find_least_positive_n_l200_200840


namespace max_arithmetic_subsequences_l200_200714

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d c : ℤ), ∀ n : ℕ, a n = d * n + c

-- Condition that the sum of the indices is even
def sum_indices_even (n m : ℕ) : Prop :=
  (n % 2 = 0 ∧ m % 2 = 0) ∨ (n % 2 = 1 ∧ m % 2 = 1)

-- Maximum count of 3-term arithmetic sequences in a sequence of 20 terms
theorem max_arithmetic_subsequences (a : ℕ → ℤ) (h_arith : arithmetic_sequence a) :
  ∃ n : ℕ, n = 180 :=
by
  sorry

end max_arithmetic_subsequences_l200_200714


namespace moles_of_hcl_l200_200329

-- Definitions according to the conditions
def methane := 1 -- 1 mole of methane (CH₄)
def chlorine := 2 -- 2 moles of chlorine (Cl₂)
def hcl := 1 -- The expected number of moles of Hydrochloric acid (HCl)

-- The Lean 4 statement (no proof required)
theorem moles_of_hcl (methane chlorine : ℕ) : hcl = 1 :=
by sorry

end moles_of_hcl_l200_200329


namespace find_four_numbers_l200_200454

theorem find_four_numbers (a b c d : ℕ) 
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7)
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) :
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := 
  by
    sorry

end find_four_numbers_l200_200454


namespace tyler_brother_age_difference_l200_200012

-- Definitions of Tyler's age and the sum of their ages:
def tyler_age : ℕ := 7
def sum_of_ages (brother_age : ℕ) : Prop := tyler_age + brother_age = 11

-- Proof problem: Prove that Tyler's brother's age minus Tyler's age equals 4 years.
theorem tyler_brother_age_difference (B : ℕ) (h : sum_of_ages B) : B - tyler_age = 4 :=
by
  sorry

end tyler_brother_age_difference_l200_200012


namespace percent_within_one_std_dev_l200_200022

theorem percent_within_one_std_dev (m d : ℝ) (dist : ℝ → ℝ)
  (symm : ∀ x, dist (m + x) = dist (m - x))
  (less_than_upper_bound : ∀ x, (x < (m + d)) → dist x < 0.92) :
  ∃ p : ℝ, p = 0.84 :=
by
  sorry

end percent_within_one_std_dev_l200_200022


namespace no_infinite_subdivision_exists_l200_200158

theorem no_infinite_subdivision_exists : ¬ ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (∀ n : ℕ,
    ∃ (ai bi : ℝ), ai > bi ∧ bi > 0 ∧ ai * bi = a * b ∧
    (ai / bi = a / b ∨ bi / ai = a / b)) :=
sorry

end no_infinite_subdivision_exists_l200_200158


namespace ellipse_standard_equation_l200_200770

theorem ellipse_standard_equation (a b : ℝ) (h1 : 2 * a = 2 * (2 * b)) (h2 : (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} ∨ (2, 0) ∈ {p : ℝ × ℝ | (p.2^2 / a^2) + (p.1^2 / b^2) = 1}) :
  (∃ a b : ℝ, (a > b ∧ a > 0 ∧ b > 0 ∧ (2 * a = 2 * (2 * b)) ∧ (2, 0) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} ∧ (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1} → (x^2 / 4 + y^2 / 1 = 1)) ∨ (x^2 / 16 + y^2 / 4 = 1))) :=
  sorry

end ellipse_standard_equation_l200_200770


namespace termite_ridden_not_collapsing_l200_200231

theorem termite_ridden_not_collapsing
  (total_termite_ridden : ℚ)
  (termite_ridden_collapsing : ℚ) :
  total_termite_ridden = 1/3 →
  termite_ridden_collapsing = 5/8 →
  total_termite_ridden - (total_termite_ridden * termite_ridden_collapsing) = 1/8 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end termite_ridden_not_collapsing_l200_200231


namespace smallest_sum_l200_200400

theorem smallest_sum (r s t : ℕ) (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_pos_t : 0 < t) 
  (h_prod : r * s * t = 1230) : r + s + t = 52 :=
sorry

end smallest_sum_l200_200400


namespace max_projection_area_l200_200382

noncomputable def maxProjectionArea (a : ℝ) : ℝ :=
  if a > (Real.sqrt 3 / 3) ∧ a <= (Real.sqrt 3 / 2) then
    Real.sqrt 3 / 4
  else if a >= (Real.sqrt 3 / 2) then
    a / 2
  else 
    0  -- if the condition for a is not met, it's an edge case which shouldn't logically occur here

theorem max_projection_area (a : ℝ) (h1 : a > Real.sqrt 3 / 3) (h2 : a <= Real.sqrt 3 / 2 ∨ a >= Real.sqrt 3 / 2) :
  maxProjectionArea a = 
    if a > Real.sqrt 3 / 3 ∧ a <= Real.sqrt 3 / 2 then Real.sqrt 3 / 4
    else if a >= Real.sqrt 3 / 2 then a / 2
    else
      sorry :=
by sorry

end max_projection_area_l200_200382


namespace triangle_square_side_length_ratio_l200_200622

theorem triangle_square_side_length_ratio (t s : ℝ) (ht : 3 * t = 12) (hs : 4 * s = 12) : 
  t / s = 4 / 3 :=
by
  sorry

end triangle_square_side_length_ratio_l200_200622


namespace simplify_expression_l200_200246

theorem simplify_expression (x : ℝ) : 3 * x + 5 * x ^ 2 + 2 - (9 - 4 * x - 5 * x ^ 2) = 10 * x ^ 2 + 7 * x - 7 :=
by
  sorry

end simplify_expression_l200_200246


namespace area_of_tangent_triangle_l200_200957

noncomputable def tangentTriangleArea : ℝ :=
  let y := λ x : ℝ => x^3 + x
  let dy := λ x : ℝ => 3 * x^2 + 1
  let slope := dy 1
  let y_intercept := 2 - slope * 1
  let x_intercept := - y_intercept / slope
  let base := x_intercept
  let height := - y_intercept
  0.5 * base * height

theorem area_of_tangent_triangle :
  tangentTriangleArea = 1 / 2 :=
by
  sorry

end area_of_tangent_triangle_l200_200957


namespace existence_of_large_independent_subset_l200_200855

theorem existence_of_large_independent_subset (X : Type) (S : set (set X)) (n : ℕ) (hX : fintype X) (hS : ∀ A B ∈ S, A ≠ B → (A ∩ B).card ≤ 1) (hX_card : (fintype.card X) = n) :
  ∃ A ⊆ X, A ∉ S ∧ fintype.card A ≥ nat.floor (real.sqrt (2 * n)) :=
begin
  sorry,
end

end existence_of_large_independent_subset_l200_200855


namespace tan_A_eq_11_l200_200083

variable (A B C : ℝ)

theorem tan_A_eq_11
  (h1 : Real.sin A = 10 * Real.sin B * Real.sin C)
  (h2 : Real.cos A = 10 * Real.cos B * Real.cos C) :
  Real.tan A = 11 := 
sorry

end tan_A_eq_11_l200_200083


namespace discrete_rv_X_hit_rings_l200_200141

-- Defining the problem in Lean
theorem discrete_rv_X_hit_rings 
  (X : ℕ → ℝ)        -- X is a function taking natural numbers to real numbers representing probabilities
  (hx : ∀ n, n ≤ 10 → X n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})  -- Values X can take are within 0 to 10
  : true := sorry    -- Placeholder for the proof which is not required

end discrete_rv_X_hit_rings_l200_200141


namespace jerry_cut_pine_trees_l200_200367

theorem jerry_cut_pine_trees (P : ℕ)
  (h1 : 3 * 60 = 180)
  (h2 : 4 * 100 = 400)
  (h3 : 80 * P + 180 + 400 = 1220) :
  P = 8 :=
by {
  sorry -- Proof not required as per the instructions
}

end jerry_cut_pine_trees_l200_200367


namespace football_even_goal_probability_l200_200800

noncomputable def poisson_even_goal_probability : ℝ :=
  let λ := 2.8 in
  (1 + Real.exp (-2 * λ)) / 2

theorem football_even_goal_probability :
  let λ := 2.8 in
  let N := Poisson λ in  -- Define N as a Poisson random variable with parameter λ
  (∑ k : ℕ in (range (2*k)).filter (λ k, N.P k > 0), N.P k) = 
  poisson_even_goal_probability :=
by
  sorry

end football_even_goal_probability_l200_200800


namespace g_one_third_value_l200_200715

noncomputable def g : ℚ → ℚ := sorry

theorem g_one_third_value : (∀ (x : ℚ), x ≠ 0 → (4 * g (1 / x) + 3 * g x / x^2 = x^3)) → g (1 / 3) = 21 / 44 := by
  intro h
  sorry

end g_one_third_value_l200_200715


namespace num_students_in_second_class_l200_200249

theorem num_students_in_second_class 
  (avg1 : ℕ) (num1 : ℕ) (avg2 : ℕ) (overall_avg : ℕ) (n : ℕ) :
  avg1 = 50 → num1 = 30 → avg2 = 60 → overall_avg = 5625 → 
  (num1 * avg1 + n * avg2) = (num1 + n) * overall_avg → n = 50 :=
by sorry

end num_students_in_second_class_l200_200249


namespace solve_problem_1_solve_problem_2_l200_200542

-- Problem statement 1: Prove that the solutions to x(x-2) = x-2 are x = 1 and x = 2.
theorem solve_problem_1 (x : ℝ) : (x * (x - 2) = x - 2) ↔ (x = 1 ∨ x = 2) :=
  sorry

-- Problem statement 2: Prove that the solutions to 2x^2 + 3x - 5 = 0 are x = 1 and x = -5/2.
theorem solve_problem_2 (x : ℝ) : (2 * x^2 + 3 * x - 5 = 0) ↔ (x = 1 ∨ x = -5 / 2) :=
  sorry

end solve_problem_1_solve_problem_2_l200_200542


namespace generatrix_length_of_cone_l200_200861

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end generatrix_length_of_cone_l200_200861


namespace bus_waiting_probability_l200_200425

-- Definitions
def arrival_time_range := (0, 90)  -- minutes from 1:00 to 2:30
def bus_wait_time := 20             -- bus waits for 20 minutes

noncomputable def probability_bus_there_when_Laura_arrives : ℚ :=
  let total_area := 90 * 90
  let trapezoid_area := 1400
  let triangle_area := 200
  (trapezoid_area + triangle_area) / total_area

-- Theorem statement
theorem bus_waiting_probability : probability_bus_there_when_Laura_arrives = 16 / 81 := by
  sorry

end bus_waiting_probability_l200_200425


namespace hexagon_angles_sum_l200_200684

theorem hexagon_angles_sum (mA mB mC : ℤ) (x y : ℤ)
  (hA : mA = 35) (hB : mB = 80) (hC : mC = 30)
  (hSum : (6 - 2) * 180 = 720)
  (hAdjacentA : 90 + 90 = 180)
  (hAdjacentC : 90 - mC = 60) :
  x + y = 95 := by
  sorry

end hexagon_angles_sum_l200_200684


namespace AD_perpendicular_to_BS_l200_200226

open Point Triangle Angle Circle Line Segment

variables {A B C D E F G S : Point}
variables (ABC : Triangle)
variables (k : Circle)
variables (h1 : ABC.angle_bisector A B C D)
variables (h2 : k.passes_through D)
variables (h3 : k.tangent_to_line AC E)
variables (h4 : k.tangent_to_line AB F)
variables (h5 : k.second_intersection_with BC G)
variables (h6 : Line.through E G S)
variables (h7 : Line.through D F S)
variables (h8 : ABC.side_length B C > ABC.side_length C A)

theorem AD_perpendicular_to_BS : Line.perpendicular (Line.through A D) (Line.through B S) :=
by
  sorry

end AD_perpendicular_to_BS_l200_200226


namespace range_of_a_l200_200846

theorem range_of_a (b c a : ℝ) (h_intersect : ∀ x : ℝ, 
  (x ^ 2 - 2 * b * x + b ^ 2 + c = 1 - x → x = b )) 
  (h_vertex : c = a * b ^ 2) :
  a ≥ (-1 / 5) ∧ a ≠ 0 := 
by 
-- Proof skipped
sorry

end range_of_a_l200_200846


namespace problem_gets_solved_prob_l200_200578

-- Define conditions for probabilities
def P_A_solves := 2 / 3
def P_B_solves := 3 / 4

-- Calculate the probability that the problem is solved
theorem problem_gets_solved_prob :
  let P_A_not_solves := 1 - P_A_solves
  let P_B_not_solves := 1 - P_B_solves
  let P_both_not_solve := P_A_not_solves * P_B_not_solves
  let P_solved := 1 - P_both_not_solve
  P_solved = 11 / 12 :=
by
  -- Skip proof
  sorry

end problem_gets_solved_prob_l200_200578


namespace Rajesh_days_to_complete_l200_200097

theorem Rajesh_days_to_complete (Mahesh_days : ℕ) (Rajesh_days : ℕ) (Total_days : ℕ)
  (h1 : Mahesh_days = 45) (h2 : Total_days - 20 = Rajesh_days) (h3 : Total_days = 54) :
  Rajesh_days = 34 :=
by
  sorry

end Rajesh_days_to_complete_l200_200097


namespace subtraction_equals_eleven_l200_200501

theorem subtraction_equals_eleven (K A N G R O : ℕ) (h1: K ≠ A) (h2: K ≠ N) (h3: K ≠ G) (h4: K ≠ R) (h5: K ≠ O) (h6: A ≠ N) (h7: A ≠ G) (h8: A ≠ R) (h9: A ≠ O) (h10: N ≠ G) (h11: N ≠ R) (h12: N ≠ O) (h13: G ≠ R) (h14: G ≠ O) (h15: R ≠ O) (sum_eq : 100 * K + 10 * A + N + 10 * G + A = 100 * R + 10 * O + O) : 
  (10 * R + N) - (10 * K + G) = 11 := 
by 
  sorry

end subtraction_equals_eleven_l200_200501


namespace complex_inequality_l200_200697

open Complex

noncomputable def condition (a b c : ℂ) := a * Complex.abs (b * c) + b * Complex.abs (c * a) + c * Complex.abs (a * b) = 0

theorem complex_inequality (a b c : ℂ) (h : condition a b c) :
  Complex.abs ((a - b) * (b - c) * (c - a)) ≥ 3 * Real.sqrt 3 * Complex.abs (a * b * c) := 
sorry

end complex_inequality_l200_200697


namespace evaluate_expression_l200_200836

theorem evaluate_expression : (527 * 527 - 526 * 528) = 1 := by
  sorry

end evaluate_expression_l200_200836


namespace thirty_six_forty_five_nine_eighteen_l200_200063

theorem thirty_six_forty_five_nine_eighteen :
  18 * 36 + 45 * 18 - 9 * 18 = 1296 :=
by
  sorry

end thirty_six_forty_five_nine_eighteen_l200_200063


namespace generatrix_length_l200_200874

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end generatrix_length_l200_200874


namespace intersection_complement_l200_200670

open Set

def U : Set ℝ := univ
def A : Set ℤ := {x : ℤ | -3 < x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement (U A B : Set ℝ) : A ∩ (U \ B) = {0, 1} := sorry

end intersection_complement_l200_200670


namespace cube_face_area_l200_200584

-- Definition for the condition of the cube's surface area
def cube_surface_area (s : ℝ) : Prop := s = 36

-- Definition stating a cube has 6 faces
def cube_faces : ℝ := 6

-- The target proposition to prove
theorem cube_face_area (s : ℝ) (area_of_one_face : ℝ) (h1 : cube_surface_area s) (h2 : cube_faces = 6) : area_of_one_face = s / 6 :=
by
  sorry

end cube_face_area_l200_200584


namespace find_y_solution_l200_200217

noncomputable def series_sum (y : ℝ) : ℝ :=
1 + 3 * y + 5 * y^2 + 7 * y^3 + ∑' n : ℕ, ((2 * n + 1) * y^n)

theorem find_y_solution : 
  ∃ y : ℝ, (series_sum y = 16) ∧ (y = (33 - Real.sqrt 129) / 32) :=
begin
  use (33 - Real.sqrt 129) / 32,
  split,
  {
    -- The statement that series_sum ((33 - sqrt 129) / 32) equals 16 should be proved here.
    sorry
  },
  refl,
end

end find_y_solution_l200_200217


namespace tan_identity_given_condition_l200_200473

variable (α : Real)

theorem tan_identity_given_condition :
  (Real.tan α + 1 / Real.tan α = 9 / 4) →
  (Real.tan α ^ 2 + 1 / (Real.sin α * Real.cos α) + 1 / Real.tan α ^ 2 = 85 / 16) := 
by
  sorry

end tan_identity_given_condition_l200_200473


namespace paint_quantity_l200_200649

variable (totalPaint : ℕ) (blueRatio greenRatio whiteRatio : ℕ)

theorem paint_quantity 
  (h_total_paint : totalPaint = 45)
  (h_ratio_blue : blueRatio = 5)
  (h_ratio_green : greenRatio = 3)
  (h_ratio_white : whiteRatio = 7) :
  let totalRatio := blueRatio + greenRatio + whiteRatio
  let partQuantity := totalPaint / totalRatio
  let bluePaint := blueRatio * partQuantity
  let greenPaint := greenRatio * partQuantity
  let whitePaint := whiteRatio * partQuantity
  bluePaint = 15 ∧ greenPaint = 9 ∧ whitePaint = 21 :=
by
  sorry

end paint_quantity_l200_200649


namespace door_height_l200_200919

theorem door_height
  (x : ℝ) -- length of the pole
  (pole_eq_diag : x = real.sqrt ((x - 4)^2 + (x - 2)^2)) -- condition 3 (Pythagorean theorem)
  : real.sqrt ((x - 4)^2 + (x - 2)^2) = 10 :=
by
  sorry

end door_height_l200_200919


namespace right_triangle_area_l200_200197

theorem right_triangle_area (A B C : ℝ) (hA : A = 64) (hB : B = 49) (hC : C = 225) :
  let a := Real.sqrt A
  let b := Real.sqrt B
  let c := Real.sqrt C
  ∃ (area : ℝ), area = (1 / 2) * a * b ∧ area = 28 :=
by
  sorry

end right_triangle_area_l200_200197


namespace inequality_of_positive_reals_l200_200514

variable {a b c : ℝ}

theorem inequality_of_positive_reals (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3 / 2 :=
sorry

end inequality_of_positive_reals_l200_200514


namespace combined_towel_weight_l200_200708

/-
Given:
1. Mary has 5 times as many towels as Frances.
2. Mary has 3 times as many towels as John.
3. The total weight of their towels is 145 pounds.
4. Mary has 60 towels.

To prove: 
The combined weight of Frances's and John's towels is 22.863 kilograms.
-/

theorem combined_towel_weight (total_weight_pounds : ℝ) (mary_towels frances_towels john_towels : ℕ) 
  (conversion_factor : ℝ) (combined_weight_kilograms : ℝ) :
  mary_towels = 60 →
  mary_towels = 5 * frances_towels →
  mary_towels = 3 * john_towels →
  total_weight_pounds = 145 →
  conversion_factor = 0.453592 →
  combined_weight_kilograms = 22.863 :=
by
  sorry

end combined_towel_weight_l200_200708


namespace part1_part2_l200_200597

noncomputable def is_monotonically_increasing (f' : ℝ → ℝ) := ∀ x, f' x ≥ 0

noncomputable def is_monotonically_decreasing (f' : ℝ → ℝ) (I : Set ℝ) := ∀ x ∈ I, f' x ≤ 0

def f' (a x : ℝ) : ℝ := 3 * x ^ 2 - a

theorem part1 (a : ℝ) : 
  is_monotonically_increasing (f' a) ↔ a ≤ 0 := sorry

theorem part2 (a : ℝ) : 
  is_monotonically_decreasing (f' a) (Set.Ioo (-1 : ℝ) (1 : ℝ)) ↔ a ≥ 3 := sorry

end part1_part2_l200_200597


namespace compute_sum_l200_200440

open BigOperators

theorem compute_sum : 
  (1 / 2 ^ 2010 : ℝ) * ∑ n in Finset.range 1006, (-3 : ℝ) ^ n * (Nat.choose 2010 (2 * n)) = -1 / 2 :=
by
  sorry

end compute_sum_l200_200440


namespace find_AD_l200_200771

-- Defining points and distances in the context of a triangle
variables {A B C D: Type*}
variables (dist_AB : ℝ) (dist_AC : ℝ) (dist_BC : ℝ) (midpoint_D : Prop)

-- Given conditions
def triangle_conditions : Prop :=
  dist_AB = 26 ∧
  dist_AC = 26 ∧
  dist_BC = 24 ∧
  midpoint_D

-- Problem statement as a Lean theorem
theorem find_AD
  (h : triangle_conditions dist_AB dist_AC dist_BC midpoint_D) :
  ∃ (AD : ℝ), AD = 2 * Real.sqrt 133 :=
sorry

end find_AD_l200_200771


namespace premium_percentage_on_shares_l200_200156

theorem premium_percentage_on_shares
    (investment : ℕ)
    (share_price : ℕ)
    (premium_percentage : ℕ)
    (dividend_percentage : ℕ)
    (total_dividend : ℕ)
    (number_of_shares : ℕ)
    (investment_eq : investment = number_of_shares * (share_price + premium_percentage))
    (dividend_eq : total_dividend = number_of_shares * (share_price * dividend_percentage / 100))
    (investment_val : investment = 14400)
    (share_price_val : share_price = 100)
    (dividend_percentage_val : dividend_percentage = 5)
    (total_dividend_val : total_dividend = 600)
    (number_of_shares_val : number_of_shares = 600 / 5) :
    premium_percentage = 20 :=
by
  sorry

end premium_percentage_on_shares_l200_200156


namespace billy_sleep_total_l200_200209

def billy_sleep : Prop :=
  let first_night := 6
  let second_night := first_night + 2
  let third_night := second_night / 2
  let fourth_night := third_night * 3
  first_night + second_night + third_night + fourth_night = 30

theorem billy_sleep_total : billy_sleep := by
  sorry

end billy_sleep_total_l200_200209


namespace sum_first_10_terms_arith_seq_l200_200700

theorem sum_first_10_terms_arith_seq (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 3 = 5)
  (h2 : a 7 = 13)
  (h3 : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h4 : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  S 10 = 100 :=
sorry

end sum_first_10_terms_arith_seq_l200_200700


namespace original_polygon_sides_l200_200614

theorem original_polygon_sides {n : ℕ} 
  (h : (n - 2) * 180 = 1620) : n = 10 ∨ n = 11 ∨ n = 12 :=
sorry

end original_polygon_sides_l200_200614


namespace cookies_eaten_is_correct_l200_200991

-- Define initial and remaining cookies
def initial_cookies : ℕ := 7
def remaining_cookies : ℕ := 5
def cookies_eaten : ℕ := initial_cookies - remaining_cookies

-- The theorem we need to prove
theorem cookies_eaten_is_correct : cookies_eaten = 2 :=
by
  -- Here we would provide the proof
  sorry

end cookies_eaten_is_correct_l200_200991


namespace similar_triangle_perimeter_l200_200011

/-
  Given an isosceles triangle with two equal sides of 18 inches and a base of 12 inches, 
  and a similar triangle with the shortest side of 30 inches, 
  prove that the perimeter of the similar triangle is 120 inches.
-/

def is_isosceles (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem similar_triangle_perimeter
  (a b c : ℕ) (a' b' c' : ℕ) (h1 : is_isosceles a b c)
  (h2 : a = 12) (h3 : b = 18) (h4 : c = 18)
  (h5 : a' = 30) (h6 : a' * 18 = a * b')
  (h7 : a' * 18 = a * c') :
  a' + b' + c' = 120 :=
by {
  sorry
}

end similar_triangle_perimeter_l200_200011


namespace box_volume_l200_200977

theorem box_volume (l w h : ℝ)
  (A1 : l * w = 30)
  (A2 : w * h = 20)
  (A3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end box_volume_l200_200977


namespace equivalent_expression_l200_200507

noncomputable def problem_statement (α β γ δ p q : ℝ) :=
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -2 * (p^2 - q^2) + 4

theorem equivalent_expression
  (α β γ δ p q : ℝ)
  (h1 : ∀ x, x^2 + p * x + 2 = 0 → (x = α ∨ x = β))
  (h2 : ∀ x, x^2 + q * x + 2 = 0 → (x = γ ∨ x = δ)) :
  problem_statement α β γ δ p q :=
by sorry

end equivalent_expression_l200_200507


namespace smallest_x_for_multiple_l200_200752

theorem smallest_x_for_multiple (x : ℕ) : (450 * x) % 720 = 0 ↔ x = 8 := 
by {
  sorry
}

end smallest_x_for_multiple_l200_200752


namespace max_remaining_numbers_l200_200237

theorem max_remaining_numbers : 
  ∃ s : Finset ℕ, s ⊆ (Finset.range 236) ∧ (∀ x y ∈ s, x ≠ y → ¬ (x - y).abs ∣ x) ∧ s.card = 118 := 
by
  sorry

end max_remaining_numbers_l200_200237


namespace rectangular_coordinates_correct_l200_200287

noncomputable def rectangular_coordinates : ℝ × ℝ → ℝ × ℝ :=
λ (p : ℝ × ℝ),
let r := Real.sqrt (p.1 ^ 2 + p.2 ^ 2),
    theta := Real.arctan (p.2 / p.1),
    cos_theta := p.1 / r,
    sin_theta := p.2 / r,
    r2 := r * r,
    cos_3theta := 4 * cos_theta ^ 3 - 3 * cos_theta,
    sin_3theta := 3 * sin_theta - 4 * sin_theta ^ 3 in
(r2 * cos_3theta, r2 * sin_3theta)

-- Given the conditions
def p : ℝ × ℝ := (12, 5)

-- Prove that the rectangular coordinates of (r^2, 3θ) are as expected
theorem rectangular_coordinates_correct :
  rectangular_coordinates p = ( - 494004 / 2197, 4441555 / 2197 ) :=
by
  sorry

end rectangular_coordinates_correct_l200_200287


namespace not_square_or_cube_l200_200511

theorem not_square_or_cube (n : ℕ) (h : n > 1) : 
  ¬ (∃ a : ℕ, 2^n - 1 = a^2) ∧ ¬ (∃ a : ℕ, 2^n - 1 = a^3) :=
by
  sorry

end not_square_or_cube_l200_200511


namespace add_to_fraction_l200_200274

theorem add_to_fraction (x : ℕ) :
  (3 + x) / (11 + x) = 5 / 9 ↔ x = 7 :=
by
  sorry

end add_to_fraction_l200_200274


namespace ab_bc_cd_da_le_four_l200_200522

theorem ab_bc_cd_da_le_four (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 :=
by
  sorry

end ab_bc_cd_da_le_four_l200_200522


namespace train_speed_l200_200613

theorem train_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 400) (h_time : time = 40) : distance / time = 10 := by
  rw [h_distance, h_time]
  norm_num

end train_speed_l200_200613


namespace range_of_t_l200_200489

theorem range_of_t (t : ℝ) (x : ℝ) : (1 < x ∧ x ≤ 4) → (|x - t| < 1 ↔ 2 ≤ t ∧ t ≤ 3) :=
by
  sorry

end range_of_t_l200_200489


namespace wire_length_is_180_l200_200295

def wire_problem (length1 length2 : ℕ) (h1 : length1 = 106) (h2 : length2 = 74) (h3 : length1 = length2 + 32) : Prop :=
  (length1 + length2 = 180)

-- Use the definition as an assumption to write the theorem.
theorem wire_length_is_180 (length1 length2 : ℕ) 
  (h1 : length1 = 106) 
  (h2 : length2 = 74) 
  (h3 : length1 = length2 + 32) : 
  length1 + length2 = 180 :=
by
  rw [h1, h2] at h3
  sorry

end wire_length_is_180_l200_200295


namespace cycling_distance_l200_200397

-- Define the conditions
def cycling_time : ℕ := 40  -- Total cycling time in minutes
def time_per_interval : ℕ := 10  -- Time per interval in minutes
def distance_per_interval : ℕ := 2  -- Distance per interval in miles

-- Proof statement
theorem cycling_distance : (cycling_time / time_per_interval) * distance_per_interval = 8 := by
  sorry

end cycling_distance_l200_200397


namespace complete_collection_prob_l200_200741

noncomputable def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem complete_collection_prob : 
  let n := 18
  let k := 10
  let needed := 6
  let uncollected_combinations := C 6 6
  let collected_combinations := C 12 4
  let total_combinations := C 18 10
  (uncollected_combinations * collected_combinations) / total_combinations = 5 / 442 :=
by 
  sorry

end complete_collection_prob_l200_200741


namespace area_of_triangle_l200_200839

namespace TriangleArea

structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

noncomputable def area (A B C : Point3D) : ℚ :=
  let x1 := A.x
  let y1 := A.y
  let z1 := A.z
  let x2 := B.x
  let y2 := B.y
  let z2 := B.z
  let x3 := C.x
  let y3 := C.y
  let z3 := C.z
  1 / 2 * ( (x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)) )

def A : Point3D := ⟨0, 3, 6⟩
def B : Point3D := ⟨-2, 2, 2⟩
def C : Point3D := ⟨-5, 5, 2⟩

theorem area_of_triangle : area A B C = 4.5 :=
by
  sorry

end TriangleArea

end area_of_triangle_l200_200839


namespace value_of_2a_plus_b_l200_200897

theorem value_of_2a_plus_b (a b : ℤ) (h1 : |a - 1| = 4) (h2 : |b| = 7) (h3 : |a + b| ≠ a + b) :
  2 * a + b = 3 ∨ 2 * a + b = -13 := sorry

end value_of_2a_plus_b_l200_200897


namespace students_in_sample_l200_200358

theorem students_in_sample (T : ℕ) (S : ℕ) (F : ℕ) (J : ℕ) (se : ℕ)
  (h1 : J = 22 * T / 100)
  (h2 : S = 25 * T / 100)
  (h3 : se = 160)
  (h4 : F = S + 64)
  (h5 : ∀ x, x ∈ ({F, S, J, se} : Finset ℕ) → x ≤ T ∧  x ≥ 0):
  T = 800 :=
by
  have h6 : T = F + S + J + se := sorry
  sorry

end students_in_sample_l200_200358


namespace fraction_of_canvas_painted_blue_l200_200406

noncomputable def square_canvas_blue_fraction : ℚ :=
  sorry

theorem fraction_of_canvas_painted_blue :
  square_canvas_blue_fraction = 3 / 8 :=
  sorry

end fraction_of_canvas_painted_blue_l200_200406


namespace probability_complete_collection_l200_200745

def combination (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem probability_complete_collection : 
  combination 6 6 * combination 12 4 / combination 18 10 = 5 / 442 :=
by
  have h1 : combination 6 6 = 1 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h2 : combination 12 4 = 495 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h3 : combination 18 10 = 43758 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  rw [h1, h2, h3]
  norm_num
  sorry

end probability_complete_collection_l200_200745


namespace inequality_solution_l200_200003

def solution_set_of_inequality (x : ℝ) : Prop :=
  x * (x - 1) < 0

theorem inequality_solution :
  { x : ℝ | solution_set_of_inequality x } = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end inequality_solution_l200_200003


namespace triangle_area_l200_200060

-- Given coordinates of vertices of triangle ABC
def A := (0, 0)
def B := (1424233, 2848467)
def C := (1424234, 2848469)

-- Define a mathematical proof statement to prove the area of the triangle ABC
theorem triangle_area :
  let area_ABC := (1 / 2 : ℝ) * Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) * (1 / Real.sqrt (2^2 + (-1)^2))
  (Float.to_string 0.50) = "0.50" :=
by
  let area_ABC := (1 / 2 : ℝ) * Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) * (1 / Real.sqrt (2^2 + (-1)^2))
  sorry
    

end triangle_area_l200_200060


namespace volume_of_rectangular_box_l200_200985

theorem volume_of_rectangular_box 
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12) : 
  l * w * h = 60 :=
sorry

end volume_of_rectangular_box_l200_200985


namespace digits_count_concatenated_l200_200255

-- Define the conditions for the digit count of 2^n and 5^n
def digits_count_2n (n p : ℕ) : Prop := 10^(p-1) ≤ 2^n ∧ 2^n < 10^p
def digits_count_5n (n q : ℕ) : Prop := 10^(q-1) ≤ 5^n ∧ 5^n < 10^q

-- The main theorem to prove the number of digits when 2^n and 5^n are concatenated
theorem digits_count_concatenated (n p q : ℕ) 
  (h1 : digits_count_2n n p) 
  (h2 : digits_count_5n n q): 
  p + q = n + 1 := by 
  sorry

end digits_count_concatenated_l200_200255


namespace non_real_roots_interval_l200_200903

theorem non_real_roots_interval (b : ℝ) : 
  (∀ x : ℝ, ∃ (a c : ℝ), a ≠ 0 ∧ b^2 - 4 * a * c < 0 ∧ (x^2 + b * x + c = 0))  →
  b ∈ Ioo (-8 : ℝ) (8 : ℝ) :=
by
  sorry

end non_real_roots_interval_l200_200903


namespace line_equation_l200_200455

variable (t : ℝ)
variable (x y : ℝ)

def param_x (t : ℝ) : ℝ := 3 * t + 2
def param_y (t : ℝ) : ℝ := 5 * t - 7

theorem line_equation :
  ∃ m b : ℝ, ∀ t : ℝ, y = param_y t ∧ x = param_x t → y = m * x + b := by
  use (5 / 3)
  use (-31 / 3)
  sorry

end line_equation_l200_200455


namespace non_real_roots_bounded_l200_200908

theorem non_real_roots_bounded (b : ℝ) :
  (∀ x : ℝ, polynomial.has_roots (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2) → ¬ real.is_root (polynomial.C 16 + polynomial.linear (b : ℝ) + polynomial.X^2)) → 
  b ∈ set.Ioo (-(8 : ℝ)) 8 :=
by sorry

end non_real_roots_bounded_l200_200908


namespace maximum_numbers_up_to_235_l200_200233

def max_remaining_numbers : ℕ := 118

theorem maximum_numbers_up_to_235 (numbers : set ℕ) (h₁ : ∀ n ∈ numbers, n ≤ 235)
  (h₂ : ∀ a b ∈ numbers, a ≠ b → ¬ (a - b).abs ∣ a) :
  numbers.card ≤ max_remaining_numbers :=
sorry

end maximum_numbers_up_to_235_l200_200233


namespace range_of_a_l200_200350

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2 * a * x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_a_l200_200350


namespace y_intercept_line_l200_200610

theorem y_intercept_line : 
  ∃ m b : ℝ, 
  (2 * m + b = -3) ∧ 
  (6 * m + b = 5) ∧ 
  b = -7 :=
by 
  sorry

end y_intercept_line_l200_200610


namespace arithmetic_sequence_a4_l200_200359

/-- Given an arithmetic sequence {a_n}, where S₁₀ = 60 and a₇ = 7, prove that a₄ = 5. -/
theorem arithmetic_sequence_a4 (a₁ d : ℝ) 
  (h1 : 10 * a₁ + 45 * d = 60) 
  (h2 : a₁ + 6 * d = 7) : 
  a₁ + 3 * d = 5 :=
  sorry

end arithmetic_sequence_a4_l200_200359


namespace coeff_b_l200_200552

noncomputable def g (a b c d e : ℝ) (x : ℝ) :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem coeff_b (a b c d e : ℝ):
  -- The function g(x) has roots at x = -1, 0, 1, 2
  (g a b c d e (-1) = 0) →
  (g a b c d e 0 = 0) →
  (g a b c d e 1 = 0) →
  (g a b c d e 2 = 0) →
  -- The function passes through the point (0, 3)
  (g a b c d e 0 = 3) →
  -- Assuming a = 1
  (a = 1) →
  -- Prove that b = -2
  b = -2 :=
by
  intros _ _ _ _ _ a_eq_1
  -- Proof omitted
  sorry

end coeff_b_l200_200552


namespace exterior_angle_DEF_l200_200385

theorem exterior_angle_DEF :
  let heptagon_angle := (180 * (7 - 2)) / 7
  let octagon_angle := (180 * (8 - 2)) / 8
  let total_degrees := 360
  total_degrees - (heptagon_angle + octagon_angle) = 96.43 :=
by
  sorry

end exterior_angle_DEF_l200_200385


namespace vertical_angles_congruent_l200_200107

theorem vertical_angles_congruent (a b : Angle) (h : VerticalAngles a b) : CongruentAngles a b :=
sorry

end vertical_angles_congruent_l200_200107


namespace labor_hired_l200_200494

noncomputable def Q_d (P : ℝ) : ℝ := 60 - 14 * P
noncomputable def Q_s (P : ℝ) : ℝ := 20 + 6 * P
noncomputable def MPL (L : ℝ) : ℝ := 160 / (L^2)
def wage : ℝ := 5

theorem labor_hired (L P : ℝ) (h_eq_price: 60 - 14 * P = 20 + 6 * P) (h_eq_wage: 160 / (L^2) * 2 = wage) :
  L = 8 :=
by
  have h1 : 60 - 14 * P = 20 + 6 * P := h_eq_price
  have h2 : 160 / (L^2) * 2 = wage := h_eq_wage
  sorry

end labor_hired_l200_200494


namespace arithmetic_expression_eval_l200_200171

theorem arithmetic_expression_eval : 
  (1000 * 0.09999) / 10 * 999 = 998001 := 
by 
  sorry

end arithmetic_expression_eval_l200_200171


namespace fuel_A_added_l200_200993

noncomputable def total_tank_capacity : ℝ := 218

noncomputable def ethanol_fraction_A : ℝ := 0.12
noncomputable def ethanol_fraction_B : ℝ := 0.16

noncomputable def total_ethanol : ℝ := 30

theorem fuel_A_added (x : ℝ) 
    (hA : 0 ≤ x) 
    (hA_le_capacity : x ≤ total_tank_capacity) 
    (h_eq : 0.12 * x + 0.16 * (total_tank_capacity - x) = total_ethanol) : 
    x = 122 := 
sorry

end fuel_A_added_l200_200993


namespace same_sign_iff_product_positive_different_sign_iff_product_negative_l200_200202

variable (a b : ℝ)

theorem same_sign_iff_product_positive :
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0)) ↔ (a * b > 0) :=
sorry

theorem different_sign_iff_product_negative :
  ((a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) ↔ (a * b < 0) :=
sorry

end same_sign_iff_product_positive_different_sign_iff_product_negative_l200_200202


namespace min_value_expr_l200_200642

open Real

theorem min_value_expr (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) :
  3 * cos θ + 2 / sin θ + 2 * sqrt 2 * tan θ ≥ (3 : ℝ) * (12 * sqrt 2)^((1 : ℝ) / (3 : ℝ)) := sorry

end min_value_expr_l200_200642


namespace size_of_can_of_concentrate_l200_200037

theorem size_of_can_of_concentrate
  (can_to_water_ratio : ℕ := 1 + 3)
  (servings_needed : ℕ := 320)
  (serving_size : ℕ := 6)
  (total_volume : ℕ := servings_needed * serving_size) :
  ∃ C : ℕ, C = total_volume / can_to_water_ratio :=
by
  sorry

end size_of_can_of_concentrate_l200_200037


namespace daily_rental_cost_l200_200774

def daily_cost (x : ℝ) (miles : ℝ) (cost_per_mile : ℝ) : ℝ :=
  x + miles * cost_per_mile

theorem daily_rental_cost (x : ℝ) (miles : ℝ) (cost_per_mile : ℝ) (total_budget : ℝ) 
  (h : daily_cost x miles cost_per_mile = total_budget) : x = 30 :=
by
  let constant_miles := 200
  let constant_cost_per_mile := 0.23
  let constant_budget := 76
  sorry

end daily_rental_cost_l200_200774


namespace population_increase_rate_l200_200401

theorem population_increase_rate (P₀ P₁ : ℕ) (rate : ℚ) (h₁ : P₀ = 220) (h₂ : P₁ = 242) :
  rate = ((P₁ - P₀ : ℚ) / P₀) * 100 := by
  sorry

end population_increase_rate_l200_200401


namespace susan_age_is_11_l200_200817

theorem susan_age_is_11 (S A : ℕ) 
  (h1 : A = S + 5) 
  (h2 : A + S = 27) : 
  S = 11 := 
by 
  sorry

end susan_age_is_11_l200_200817


namespace volume_rectangular_box_l200_200980

variables {l w h : ℝ}

theorem volume_rectangular_box (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end volume_rectangular_box_l200_200980


namespace divide_rope_into_parts_l200_200178

theorem divide_rope_into_parts:
  (∀ rope_length : ℝ, rope_length = 5 -> ∀ parts : ℕ, parts = 4 -> (∀ i : ℕ, i < parts -> ((rope_length / parts) = (5 / 4)))) :=
by sorry

end divide_rope_into_parts_l200_200178


namespace frank_original_money_l200_200467

theorem frank_original_money (X : ℝ) :
  (X - (1 / 5) * X - (1 / 4) * (X - (1 / 5) * X) = 360) → (X = 600) :=
by
  sorry

end frank_original_money_l200_200467


namespace round_time_of_A_l200_200763

theorem round_time_of_A (T_a T_b : ℝ) 
  (h1 : 4 * T_b = 5 * T_a) 
  (h2 : 4 * T_b = 4 * T_a + 10) : T_a = 10 :=
by
  sorry

end round_time_of_A_l200_200763


namespace sum_of_primes_between_30_and_50_l200_200756

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- List of prime numbers between 30 and 50
def prime_numbers_between_30_and_50 : List ℕ := [31, 37, 41, 43, 47]

-- Sum of prime numbers between 30 and 50
def sum_prime_numbers_between_30_and_50 : ℕ :=
  prime_numbers_between_30_and_50.sum

-- Theorem: The sum of prime numbers between 30 and 50 is 199
theorem sum_of_primes_between_30_and_50 :
  sum_prime_numbers_between_30_and_50 = 199 := by
    sorry

end sum_of_primes_between_30_and_50_l200_200756


namespace sum_a4_a6_l200_200067

variable (a : ℕ → ℝ) (d : ℝ)
variable (h_arith : ∀ n : ℕ, a (n + 1) = a 1 + n * d)
variable (h_sum : a 2 + a 3 + a 7 + a 8 = 8)

theorem sum_a4_a6 : a 4 + a 6 = 4 :=
by
  sorry

end sum_a4_a6_l200_200067


namespace sufficient_but_not_necessary_l200_200854

variables {p q : Prop}

theorem sufficient_but_not_necessary :
  (p → q) ∧ (¬q → ¬p) ∧ ¬(q → p) → (¬q → ¬p) ∧ (¬(q → p)) :=
by
  sorry

end sufficient_but_not_necessary_l200_200854


namespace sticker_probability_l200_200738

theorem sticker_probability 
  (n : ℕ) (k : ℕ) (uncollected : ℕ) (collected : ℕ) (C : ℕ → ℕ → ℕ) :
  n = 18 → k = 10 → uncollected = 6 → collected = 12 → 
  (C uncollected uncollected) * (C collected (k - uncollected)) = 495 → 
  C n k = 43758 → 
  (495 : ℚ) / 43758 = 5 / 442 := 
by
  intros h_n h_k h_uncollected h_collected h_C1 h_C2
  sorry

end sticker_probability_l200_200738


namespace total_population_l200_200491

theorem total_population (b g t : ℕ) (h₁ : b = 6 * g) (h₂ : g = 5 * t) :
  b + g + t = 36 * t :=
by
  sorry

end total_population_l200_200491


namespace pen_cost_difference_l200_200787

theorem pen_cost_difference :
  ∀ (P : ℕ), (P + 2 = 13) → (P - 2 = 9) :=
by
  intro P
  intro h
  sorry

end pen_cost_difference_l200_200787


namespace fatima_probability_l200_200137

theorem fatima_probability :
  let y := (Nat.choose 12 6 : ℚ) / 2^12 in
  2 * (793 / 2048 : ℚ) + y = 1 → (793 / 2048 : ℚ) = (Nat.choose 12 6 : ℚ) / 2^12 → 
  (793 / 2048 = 793 / 2048 : ℚ) := 
by
  intros y h1 h2
  sorry

end fatima_probability_l200_200137


namespace polynomial_value_at_minus_2_l200_200757

variable (a b : ℝ)

def polynomial (x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem polynomial_value_at_minus_2 :
  (polynomial a b (-2) = -21) :=
  sorry

end polynomial_value_at_minus_2_l200_200757


namespace molecular_weight_neutralization_l200_200590

def molecular_weight_acetic_acid : ℝ := 
  (12.01 * 2) + (1.008 * 4) + (16.00 * 2)

def molecular_weight_sodium_hydroxide : ℝ := 
  22.99 + 16.00 + 1.008

def total_weight_acetic_acid (moles : ℝ) : ℝ := 
  molecular_weight_acetic_acid * moles

def total_weight_sodium_hydroxide (moles : ℝ) : ℝ := 
  molecular_weight_sodium_hydroxide * moles

def total_molecular_weight (moles_ac: ℝ) (moles_naoh : ℝ) : ℝ :=
  total_weight_acetic_acid moles_ac + 
  total_weight_sodium_hydroxide moles_naoh

theorem molecular_weight_neutralization :
  total_molecular_weight 7 10 = 820.344 :=
by
  sorry

end molecular_weight_neutralization_l200_200590


namespace question1_question2_l200_200334

variables (θ : ℝ)

-- Condition: tan θ = 2
def tan_theta_eq : Prop := Real.tan θ = 2

-- Question 1: Prove (4 * sin θ - 2 * cos θ) / (3 * sin θ + 5 * cos θ) = 6 / 11
theorem question1 (h : tan_theta_eq θ) : (4 * Real.sin θ - 2 * Real.cos θ) / (3 * Real.sin θ + 5 * Real.cos θ) = 6 / 11 :=
by
  sorry

-- Question 2: Prove 1 - 4 * sin θ * cos θ + 2 * cos² θ = -1 / 5
theorem question2 (h : tan_theta_eq θ) : 1 - 4 * Real.sin θ * Real.cos θ + 2 * (Real.cos θ)^2 = -1 / 5 :=
by
  sorry

end question1_question2_l200_200334


namespace faye_total_books_l200_200996

def initial_books : ℕ := 34
def books_given_away : ℕ := 3
def books_bought : ℕ := 48

theorem faye_total_books : initial_books - books_given_away + books_bought = 79 :=
by
  sorry

end faye_total_books_l200_200996


namespace max_profit_l200_200580

theorem max_profit (m : ℝ) :
  (m - 8) * (900 - 15 * m) = -15 * (m - 34) ^ 2 + 10140 :=
by
  sorry

end max_profit_l200_200580


namespace polygon_sides_count_l200_200318

def sides_square : ℕ := 4
def sides_triangle : ℕ := 3
def sides_hexagon : ℕ := 6
def sides_heptagon : ℕ := 7
def sides_octagon : ℕ := 8
def sides_nonagon : ℕ := 9

def total_sides_exposed : ℕ :=
  let adjacent_1side := sides_square + sides_nonagon - 2 * 1
  let adjacent_2sides :=
    sides_triangle + sides_hexagon +
    sides_heptagon + sides_octagon - 4 * 2
  adjacent_1side + adjacent_2sides

theorem polygon_sides_count : total_sides_exposed = 27 := by
  sorry

end polygon_sides_count_l200_200318


namespace fraction_evaluation_l200_200450

theorem fraction_evaluation :
  (1/5 - 1/7) / (3/8 + 2/9) = 144/1505 := 
  by 
    sorry

end fraction_evaluation_l200_200450


namespace first_number_is_105_percent_of_second_kilograms_reduced_by_10_percent_l200_200157

-- Proof problem 1: Given a number is 5% more than another number
theorem first_number_is_105_percent_of_second (x y : ℚ) (h : x = y * 1.05) : x = y * (1 + 0.05) :=
by {
  -- proof here
  sorry
}

-- Proof problem 2: 10 kilograms reduced by 10%
theorem kilograms_reduced_by_10_percent (kg : ℚ) (h : kg = 10) : kg * (1 - 0.1) = 9 :=
by {
  -- proof here
  sorry
}

end first_number_is_105_percent_of_second_kilograms_reduced_by_10_percent_l200_200157


namespace problem1_problem2_l200_200479

def f (x a : ℝ) := x^2 + 2 * a * x + 2

theorem problem1 (a : ℝ) (h : a = -1) : 
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a ≤ 37) ∧ (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x a = 37) ∧
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a ≥ 1) ∧ (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x a = 1) :=
by
  sorry

theorem problem2 (a : ℝ) : 
  (∀ x1 x2 : ℝ, -5 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 5 → f x1 a > f x2 a) ↔ a ≤ -5 :=
by
  sorry

end problem1_problem2_l200_200479


namespace expression_evaluation_l200_200891

-- Define the property to be proved:
theorem expression_evaluation (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by sorry

end expression_evaluation_l200_200891


namespace M_is_real_l200_200939

open Complex

-- Define the condition that characterizes the set M
def M (Z : ℂ) : Prop := (Z - 1)^2 = abs (Z - 1)^2

-- Prove that M is exactly the set of real numbers
theorem M_is_real : ∀ (Z : ℂ), M Z ↔ Z.im = 0 :=
by
  sorry

end M_is_real_l200_200939


namespace sum_sq_roots_cubic_l200_200830

noncomputable def sum_sq_roots (r s t : ℝ) : ℝ :=
  r^2 + s^2 + t^2

theorem sum_sq_roots_cubic :
  ∀ r s t, (2 * r^3 + 3 * r^2 - 5 * r + 1 = 0) →
           (2 * s^3 + 3 * s^2 - 5 * s + 1 = 0) →
           (2 * t^3 + 3 * t^2 - 5 * t + 1 = 0) →
           (r + s + t = -3 / 2) →
           (r * s + r * t + s * t = 5 / 2) →
           sum_sq_roots r s t = -11 / 4 :=
by 
  intros r s t h₁ h₂ h₃ sum_roots prod_roots
  sorry

end sum_sq_roots_cubic_l200_200830


namespace emily_total_spent_l200_200636

def total_cost (art_supplies_cost skirt_cost : ℕ) (number_of_skirts : ℕ) : ℕ :=
  art_supplies_cost + (skirt_cost * number_of_skirts)

theorem emily_total_spent :
  total_cost 20 15 2 = 50 :=
by
  sorry

end emily_total_spent_l200_200636


namespace difference_one_third_0_333_l200_200974

theorem difference_one_third_0_333 :
  let one_third : ℚ := 1 / 3
  let three_hundred_thirty_three_thousandth : ℚ := 333 / 1000
  one_third - three_hundred_thirty_three_thousandth = 1 / 3000 :=
by
  sorry

end difference_one_third_0_333_l200_200974


namespace electricity_fee_l200_200492

theorem electricity_fee (a b : ℝ) : 
  let base_usage := 100
  let additional_usage := 160 - base_usage
  let base_cost := base_usage * a
  let additional_cost := additional_usage * b
  base_cost + additional_cost = 100 * a + 60 * b :=
by
  sorry

end electricity_fee_l200_200492


namespace find_z_l200_200912

theorem find_z (x : ℕ) (z : ℚ) (h1 : x = 103)
               (h2 : x^3 * z - 3 * x^2 * z + 2 * x * z = 208170) 
               : z = 5 / 265 := 
by 
  sorry

end find_z_l200_200912


namespace cost_of_goat_l200_200129

theorem cost_of_goat (G : ℝ) (goat_count : ℕ) (llama_count : ℕ) (llama_multiplier : ℝ) (total_cost : ℝ) 
    (h1 : goat_count = 3)
    (h2 : llama_count = 2 * goat_count)
    (h3 : llama_multiplier = 1.5)
    (h4 : total_cost = 4800) : G = 400 :=
by
  sorry

end cost_of_goat_l200_200129


namespace determine_coefficients_l200_200833

theorem determine_coefficients (p q : ℝ) :
  (∃ x : ℝ, x^2 + p * x + q = 0 ∧ x = p) ∧ (∃ y : ℝ, y^2 + p * y + q = 0 ∧ y = q)
  ↔ (p = 0 ∧ q = 0) ∨ (p = 1 ∧ q = -2) := by
sorry

end determine_coefficients_l200_200833


namespace area_of_trapezium_l200_200639

-- Defining the lengths of the sides and the distance
def a : ℝ := 12  -- 12 cm
def b : ℝ := 16  -- 16 cm
def h : ℝ := 14  -- 14 cm

-- Statement that the area of the trapezium is 196 cm²
theorem area_of_trapezium : (1 / 2) * (a + b) * h = 196 :=
by
  sorry

end area_of_trapezium_l200_200639


namespace total_whales_observed_l200_200366

-- Define the conditions
def trip1_male_whales : ℕ := 28
def trip1_female_whales : ℕ := 2 * trip1_male_whales
def trip1_total_whales : ℕ := trip1_male_whales + trip1_female_whales

def baby_whales_trip2 : ℕ := 8
def adult_whales_trip2 : ℕ := 2 * baby_whales_trip2
def trip2_total_whales : ℕ := baby_whales_trip2 + adult_whales_trip2

def trip3_male_whales : ℕ := trip1_male_whales / 2
def trip3_female_whales : ℕ := trip1_female_whales
def trip3_total_whales : ℕ := trip3_male_whales + trip3_female_whales

-- Prove the total number of whales observed
theorem total_whales_observed : trip1_total_whales + trip2_total_whales + trip3_total_whales = 178 := by
  -- Assuming all intermediate steps are correct
  sorry

end total_whales_observed_l200_200366


namespace billy_sleep_total_l200_200211

theorem billy_sleep_total :
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  day1 + day2 + day3 + day4 = 30 :=
by
  -- Definitions
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  -- Assertion
  have h : day1 + day2 + day3 + day4 = 30 := sorry
  exact h

end billy_sleep_total_l200_200211


namespace circle_condition_l200_200395

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + m = 0) → m < 1 := 
by
  sorry

end circle_condition_l200_200395


namespace probability_complete_collection_l200_200746

def combination (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem probability_complete_collection : 
  combination 6 6 * combination 12 4 / combination 18 10 = 5 / 442 :=
by
  have h1 : combination 6 6 = 1 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h2 : combination 12 4 = 495 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h3 : combination 18 10 = 43758 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  rw [h1, h2, h3]
  norm_num
  sorry

end probability_complete_collection_l200_200746


namespace find_missing_value_l200_200280

theorem find_missing_value :
  300 * 2 + (12 + 4) * 1 / 8 = 602 :=
by
  sorry

end find_missing_value_l200_200280


namespace total_balloons_l200_200261

theorem total_balloons (T : ℕ) 
    (h1 : T / 4 = 100)
    : T = 400 := 
by
  sorry

end total_balloons_l200_200261


namespace factorization1_factorization2_factorization3_l200_200452

-- Problem 1
theorem factorization1 (a x : ℝ) : a * x^2 - 4 * a = a * (x + 2) * (x - 2) :=
sorry

-- Problem 2
theorem factorization2 (m x y : ℝ) : m * x^2 + 2 * m * x * y + m * y^2 = m * (x + y)^2 :=
sorry

-- Problem 3
theorem factorization3 (a b : ℝ) : (1 / 2) * a^2 - a * b + (1 / 2) * b^2 = (1 / 2) * (a - b)^2 :=
sorry

end factorization1_factorization2_factorization3_l200_200452


namespace rectangle_lengths_correct_l200_200721

-- Definitions of the parameters and their relationships
noncomputable def AB := 1200
noncomputable def BC := 150
noncomputable def AB_ext := AB
noncomputable def BC_ext := BC + 350
noncomputable def CD := AB
noncomputable def DA := BC

-- Definitions of the calculated distances using the conditions
noncomputable def AP := Real.sqrt (AB^2 + BC_ext^2)
noncomputable def PD := Real.sqrt (BC_ext^2 + AB^2)

-- Using similarity of triangles for PQ and CQ
noncomputable def PQ := (350 / 500) * AP
noncomputable def CQ := (350 / 500) * AB

-- The theorem to prove the final results
theorem rectangle_lengths_correct :
    AP = 1300 ∧
    PD = 1250 ∧
    PQ = 910 ∧
    CQ = 840 :=
    by
    sorry

end rectangle_lengths_correct_l200_200721


namespace log_minus_one_has_one_zero_l200_200728

theorem log_minus_one_has_one_zero : ∃! x : ℝ, x > 0 ∧ (Real.log x - 1 = 0) :=
sorry

end log_minus_one_has_one_zero_l200_200728


namespace prob_qualified_prod_by_A_l200_200500

variable (p_A : ℝ) (p_not_A : ℝ) (p_B_given_A : ℝ) (p_B_given_not_A : ℝ)
variable (p_A_and_B : ℝ)

axiom prob_A : p_A = 0.7
axiom prob_not_A : p_not_A = 0.3
axiom prob_B_given_A : p_B_given_A = 0.95
axiom prob_B_given_not_A : p_B_given_not_A = 0.8

theorem prob_qualified_prod_by_A :
  p_A_and_B = p_A * p_B_given_A :=
  sorry

end prob_qualified_prod_by_A_l200_200500


namespace find_smallest_x_l200_200331

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

theorem find_smallest_x (x: ℕ) (h1: 2 * x = 144) (h2: 3 * x = 216) : x = 72 :=
by
  sorry

end find_smallest_x_l200_200331


namespace exists_solution_for_lambda_9_l200_200465

theorem exists_solution_for_lambda_9 :
  ∃ x y : ℝ, (x^2 + y^2 = 8 * x + 6 * y) ∧ (9 * x^2 + y^2 = 6 * y) ∧ (y^2 + 9 = 9 * x + 6 * y + 9) :=
by
  sorry

end exists_solution_for_lambda_9_l200_200465


namespace expression_constant_for_large_x_l200_200412

theorem expression_constant_for_large_x (x : ℝ) (h : x ≥ 4 / 7) : 
  -4 * x + |4 - 7 * x| - |1 - 3 * x| + 4 = 1 :=
by
  sorry

end expression_constant_for_large_x_l200_200412


namespace curve_equation_l200_200640

noncomputable def satisfies_conditions (f : ℝ → ℝ) (M₀ : ℝ × ℝ) : Prop :=
  (f M₀.1 = M₀.2) ∧ 
  (∀ (x y : ℝ) (h_tangent : ∀ x y, y = (f x) → x * y - 2 * (f x) * x = 0),
    y = f x → x * y / (y / x) = 2 * x)

theorem curve_equation (f : ℝ → ℝ) :
  satisfies_conditions f (1, 4) →
  (∀ x : ℝ, f x * x = 4) :=
by
  intro h
  sorry

end curve_equation_l200_200640


namespace not_net_of_cuboid_l200_200222

noncomputable def cuboid_closed_path (c : Type) (f : c → c) :=
∀ (x1 x2 : c), ∃ (y : c), f x1 = y ∧ f x2 = y

theorem not_net_of_cuboid (c : Type) [Nonempty c] [DecidableEq c] (net : c → Set c) (f : c → c) :
  cuboid_closed_path c f → ¬ (∀ x, net x = {x}) :=
by
  sorry

end not_net_of_cuboid_l200_200222


namespace original_numbers_l200_200733

theorem original_numbers (a b c d : ℕ) (x : ℕ)
  (h1 : a + b + c + d = 45)
  (h2 : a + 2 = x)
  (h3 : b - 2 = x)
  (h4 : 2 * c = x)
  (h5 : d / 2 = x) : 
  (a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20) :=
sorry

end original_numbers_l200_200733


namespace find_pq_l200_200321

variable (p q : ℝ)

def vec1 : Fin 3 → ℝ := ![3, p, -4]
def vec2 : Fin 3 → ℝ := ![6, 5, q]
def cross_product (v₁ v₂ : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![v₁ 1 * v₂ 2 - v₁ 2 * v₂ 1, 
    v₁ 2 * v₂ 0 - v₁ 0 * v₂ 2, 
    v₁ 0 * v₂ 1 - v₁ 1 * v₂ 0]

theorem find_pq :
  cross_product (vec1 p q) (vec2 p q) = ![0, 0, 0] →
  p = 5 / 2 ∧ q = -8 := by
  sorry

end find_pq_l200_200321


namespace coral_three_night_total_pages_l200_200047

-- Definitions based on conditions in the problem
def night1_pages : ℕ := 30
def night2_pages : ℕ := 2 * night1_pages - 2
def night3_pages : ℕ := night1_pages + night2_pages + 3
def total_pages : ℕ := night1_pages + night2_pages + night3_pages

-- The statement we want to prove
theorem coral_three_night_total_pages : total_pages = 179 := by
  sorry

end coral_three_night_total_pages_l200_200047


namespace proportional_x_y2_y_z2_l200_200762

variable {x y z k m c : ℝ}

theorem proportional_x_y2_y_z2 (h1 : x = k * y^2) (h2 : y = m / z^2) (h3 : x = 2) (hz4 : z = 4) (hz16 : z = 16):
  x = 1/128 :=
by
  sorry

end proportional_x_y2_y_z2_l200_200762


namespace tangent_half_angle_sum_eq_product_l200_200950

variable {α β γ : ℝ}

theorem tangent_half_angle_sum_eq_product (h : α + β + γ = 2 * Real.pi) :
  Real.tan (α / 2) + Real.tan (β / 2) + Real.tan (γ / 2) =
  Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) :=
sorry

end tangent_half_angle_sum_eq_product_l200_200950


namespace calculate_expression_value_l200_200822

theorem calculate_expression_value :
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 :=
by
  sorry

end calculate_expression_value_l200_200822


namespace distribute_cookies_l200_200498

theorem distribute_cookies :
  (∑ i in Finset.range 5, (3 + 0)) ≤ 30 →
  (∑ i in Finset.range 5, y i) + 15 = 30 →
  (∑ i in Finset.range 5, y i) = 15 →
  nat.choose (15 + 5 - 1) (5 - 1) = 3876 :=
by {
  sorry
}

end distribute_cookies_l200_200498


namespace area_proof_l200_200566

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end area_proof_l200_200566


namespace n_leq_84_l200_200096

theorem n_leq_84 (n : ℕ) (hn : 0 < n) (h: (1 / 2 + 1 / 3 + 1 / 7 + 1 / ↑n : ℚ).den ≤ 1): n ≤ 84 :=
sorry

end n_leq_84_l200_200096


namespace correct_total_annual_cost_l200_200368

def cost_after_coverage (cost: ℕ) (coverage: ℕ) : ℕ :=
  cost - (cost * coverage / 100)

def epiPen_costs : ℕ :=
  (cost_after_coverage 500 75) +
  (cost_after_coverage 550 60) +
  (cost_after_coverage 480 70) +
  (cost_after_coverage 520 65)

def monthly_medical_expenses : ℕ :=
  (cost_after_coverage 250 80) +
  (cost_after_coverage 180 70) +
  (cost_after_coverage 300 75) +
  (cost_after_coverage 350 60) +
  (cost_after_coverage 200 70) +
  (cost_after_coverage 400 80) +
  (cost_after_coverage 150 90) +
  (cost_after_coverage 100 100) +
  (cost_after_coverage 300 60) +
  (cost_after_coverage 350 90) +
  (cost_after_coverage 450 85) +
  (cost_after_coverage 500 65)

def total_annual_cost : ℕ :=
  epiPen_costs + monthly_medical_expenses

theorem correct_total_annual_cost :
  total_annual_cost = 1542 :=
  by sorry

end correct_total_annual_cost_l200_200368


namespace cube_sum_equals_36_l200_200375

variable {a b c k : ℝ}

theorem cube_sum_equals_36 (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
    (heq : (a^3 - 12) / a = (b^3 - 12) / b)
    (heq_another : (b^3 - 12) / b = (c^3 - 12) / c) :
    a^3 + b^3 + c^3 = 36 := by
  sorry

end cube_sum_equals_36_l200_200375


namespace center_of_tangent_circle_lies_on_hyperbola_l200_200548

open Real

def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 6*y + 24 = 0

noncomputable def locus_of_center : Set (ℝ × ℝ) :=
  {P | ∃ (r : ℝ), ∀ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x2 y2 → 
    dist P (x1, y1) = r + 2 ∧ dist P (x2, y2) = r + 1}

theorem center_of_tangent_circle_lies_on_hyperbola :
  ∀ P : ℝ × ℝ, P ∈ locus_of_center → ∃ (a b : ℝ) (F1 F2 : ℝ × ℝ), ∀ Q : ℝ × ℝ,
    dist Q F1 - dist Q F2 = 1 ∧ 
    dist F1 F2 = 5 ∧
    P ∈ {Q | dist Q F1 - dist Q F2 = 1} :=
sorry

end center_of_tangent_circle_lies_on_hyperbola_l200_200548


namespace expression_eval_l200_200550

theorem expression_eval : 2 * 3 + 2 * 3 = 12 := by
  sorry

end expression_eval_l200_200550


namespace median_salary_is_correct_l200_200167

-- Define a structure for position and salary data
structure EmployeePosition :=
  (title : String)
  (count : Nat)
  (salary : Int)

-- Given data
def employeePositions : List EmployeePosition :=
  [{title := "CEO", count := 1, salary := 150000},
   {title := "General Manager", count := 4, salary := 95000},
   {title := "Manager", count := 12, salary := 80000},
   {title := "Assistant Manager", count := 8, salary := 55000},
   {title := "Clerk", count := 40, salary := 25000}]

noncomputable def totalEmployees : Nat :=
  employeePositions.foldl (fun acc pos => acc + pos.count) 0

noncomputable def medianSalary (positions : List EmployeePosition) : Int :=
  let sortedEmployees := positions.flatMap (fun pos => List.replicate pos.count pos.salary)
  let sortedSalaries := sortedEmployees.sort (<=)
  sortedSalaries.get! (sortedSalaries.length / 2)

theorem median_salary_is_correct : medianSalary employeePositions = 25000 := by
  sorry

end median_salary_is_correct_l200_200167


namespace prod_sum_leq_four_l200_200521

theorem prod_sum_leq_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 4) :
  ab + bc + cd + da ≤ 4 :=
sorry

end prod_sum_leq_four_l200_200521


namespace price_of_tea_mixture_l200_200389

noncomputable def price_of_mixture (price1 price2 price3 : ℝ) (ratio1 ratio2 ratio3 : ℝ) : ℝ :=
  (price1 * ratio1 + price2 * ratio2 + price3 * ratio3) / (ratio1 + ratio2 + ratio3)

theorem price_of_tea_mixture :
  price_of_mixture 126 135 175.5 1 1 2 = 153 := 
by
  sorry

end price_of_tea_mixture_l200_200389


namespace quadratic_non_real_roots_iff_l200_200911

theorem quadratic_non_real_roots_iff (b : ℝ) : 
  (∀ x : ℂ, ¬ (x^2 + b*x + 16 = 0) → (b^2 - 4 * 1 * 16 < 0)) ↔ -8 < b ∧ b < 8 :=
by sorry

end quadratic_non_real_roots_iff_l200_200911


namespace Chris_buys_48_golf_balls_l200_200547

theorem Chris_buys_48_golf_balls (total_golf_balls : ℕ) (dozen_to_balls : ℕ → ℕ)
  (dan_buys : ℕ) (gus_buys : ℕ) (chris_buys : ℕ) :
  dozen_to_balls 1 = 12 →
  dan_buys = 5 →
  gus_buys = 2 →
  total_golf_balls = 132 →
  (chris_buys * 12) + (dan_buys * 12) + (gus_buys * 12) = total_golf_balls →
  chris_buys * 12 = 48 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Chris_buys_48_golf_balls_l200_200547


namespace arithmetic_sequence_common_difference_l200_200377

theorem arithmetic_sequence_common_difference 
  (d : ℝ) (h : d ≠ 0) (a : ℕ → ℝ)
  (h1 : a 1 = 9 * d)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (k : ℕ) :
  (a k)^2 = (a 1) * (a (2 * k)) → k = 4 :=
by
  sorry

end arithmetic_sequence_common_difference_l200_200377


namespace sticker_probability_l200_200740

theorem sticker_probability 
  (n : ℕ) (k : ℕ) (uncollected : ℕ) (collected : ℕ) (C : ℕ → ℕ → ℕ) :
  n = 18 → k = 10 → uncollected = 6 → collected = 12 → 
  (C uncollected uncollected) * (C collected (k - uncollected)) = 495 → 
  C n k = 43758 → 
  (495 : ℚ) / 43758 = 5 / 442 := 
by
  intros h_n h_k h_uncollected h_collected h_C1 h_C2
  sorry

end sticker_probability_l200_200740


namespace prime_square_minus_one_divisible_by_24_l200_200711

theorem prime_square_minus_one_divisible_by_24 (n : ℕ) (h_prime : Prime n) (h_n_neq_2 : n ≠ 2) (h_n_neq_3 : n ≠ 3) : 24 ∣ (n^2 - 1) :=
sorry

end prime_square_minus_one_divisible_by_24_l200_200711


namespace john_needs_more_money_l200_200278

def total_needed : ℝ := 2.50
def current_amount : ℝ := 0.75
def remaining_amount : ℝ := 1.75

theorem john_needs_more_money : total_needed - current_amount = remaining_amount :=
by
  sorry

end john_needs_more_money_l200_200278


namespace sum_squares_bound_l200_200516

theorem sum_squares_bound (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) : 
  ab + bc + cd + da ≤ 4 :=
sorry

end sum_squares_bound_l200_200516


namespace total_bills_proof_l200_200635

variable (a : ℝ) (total_may : ℝ) (total_june_may_june : ℝ)

-- The total bill in May is 140 yuan.
def total_bill_may (a : ℝ) := 140

-- The water bill increases by 10% in June.
def water_bill_june (a : ℝ) := 1.1 * a

-- The electricity bill in May.
def electricity_bill_may (a : ℝ) := 140 - a

-- The electricity bill increases by 20% in June.
def electricity_bill_june (a : ℝ) := (140 - a) * 1.2

-- Total electricity bills in June.
def total_electricity_june (a : ℝ) := (140 - a) + 0.2 * (140 - a)

-- Total water and electricity bills in June.
def total_water_electricity_june (a : ℝ) := 1.1 * a + 168 - 1.2 * a

-- Total water and electricity bills for May and June.
def total_water_electricity_may_june (a : ℝ) := a + (1.1 * a) + (140 - a) + ((140 - a) * 1.2)

-- When a = 40, the total water and electricity bills for May and June.
theorem total_bills_proof : ∀ a : ℝ, a = 40 → total_water_electricity_may_june a = 304 := 
by
  intros a ha
  rw [ha]
  sorry

end total_bills_proof_l200_200635


namespace part_a_part_b_part_c_l200_200769

variable (f g : ℝ → ℝ)
variable (x₀ : ℝ)
variable [DifferentiableAt ℝ f x₀]
variable [DifferentiableAt ℝ g x₀]

theorem part_a : deriv (λ x, f x + g x) x₀ = deriv f x₀ + deriv g x₀ := 
sorry

theorem part_b : deriv (λ x, f x * g x) x₀ = deriv f x₀ * g x₀ + f x₀ * deriv g x₀ := 
sorry

theorem part_c (h : g x₀ ≠ 0) : deriv (λ x, f x / g x) x₀ = (deriv f x₀ * g x₀ - f x₀ * deriv g x₀) / (g x₀)^2 := 
sorry

end part_a_part_b_part_c_l200_200769


namespace intersection_complement_eq_l200_200201

open Set

-- Definitions from the problem conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | x ≥ 0}
def C_U_N : Set ℝ := {x | x < 0}

-- Statement of the proof problem
theorem intersection_complement_eq : M ∩ C_U_N = {x | -1 ≤ x ∧ x < 0} :=
by
  sorry

end intersection_complement_eq_l200_200201


namespace segments_either_disjoint_or_common_point_l200_200006

theorem segments_either_disjoint_or_common_point (n : ℕ) (segments : List (ℝ × ℝ)) 
  (h_len : segments.length = n^2 + 1) : 
  (∃ (disjoint_segments : List (ℝ × ℝ)), disjoint_segments.length ≥ n + 1 ∧ 
    (∀ (s1 s2 : (ℝ × ℝ)), s1 ∈ disjoint_segments → s2 ∈ disjoint_segments 
    → s1 ≠ s2 → ¬ (s1.1 ≤ s2.2 ∧ s2.1 ≤ s1.2))) 
  ∨ 
  (∃ (common_point_segments : List (ℝ × ℝ)), common_point_segments.length ≥ n + 1 ∧ 
    (∃ (p : ℝ), ∀ (s : (ℝ × ℝ)), s ∈ common_point_segments → s.1 ≤ p ∧ p ≤ s.2)) :=
sorry

end segments_either_disjoint_or_common_point_l200_200006


namespace next_in_sequence_is_65_by_19_l200_200230

section
  open Int

  -- Definitions for numerators
  def numerator_sequence : ℕ → ℤ
  | 0 => -3
  | 1 => 5
  | 2 => -9
  | 3 => 17
  | 4 => -33
  | (n + 5) => numerator_sequence n * (-2) + 1

  -- Definitions for denominators
  def denominator_sequence : ℕ → ℕ
  | 0 => 4
  | 1 => 7
  | 2 => 10
  | 3 => 13
  | 4 => 16
  | (n + 5) => denominator_sequence n + 3

  -- Next term in the sequence
  def next_term (n : ℕ) : ℚ :=
    (numerator_sequence (n + 5) : ℚ) / (denominator_sequence (n + 5) : ℚ)

  -- Theorem stating the next number in the sequence
  theorem next_in_sequence_is_65_by_19 :
    next_term 0 = 65 / 19 :=
  by
    unfold next_term
    simp [numerator_sequence, denominator_sequence]
    sorry
end

end next_in_sequence_is_65_by_19_l200_200230


namespace trigonometric_identity_l200_200308

theorem trigonometric_identity :
  let θ := 30 * Real.pi / 180
  in (Real.tan θ ^ 2 - Real.sin θ ^ 2) / (Real.tan θ ^ 2 * Real.sin θ ^ 2) = 1 :=
by
  sorry

end trigonometric_identity_l200_200308


namespace food_company_total_food_l200_200283

theorem food_company_total_food (boxes : ℕ) (kg_per_box : ℕ) (full_boxes : boxes = 388) (weight_per_box : kg_per_box = 2) :
  boxes * kg_per_box = 776 :=
by
  -- the proof would go here
  sorry

end food_company_total_food_l200_200283


namespace arithmetic_seq_sum_l200_200194

variable (a : ℕ → ℝ)

open ArithSeq

-- Given conditions
axiom h_arith_seq : ∀ n, a (n+1) - a n = a 1 - a 0
axiom h_sum : a 1 + a 5 + a 9 = 6

-- The statement to prove
theorem arithmetic_seq_sum : a 2 + a 8 = 4 := by
  sorry -- proof to be completed

end arithmetic_seq_sum_l200_200194


namespace div_240_of_prime_diff_l200_200933

-- Definitions
def is_prime (n : ℕ) : Prop := ∃ p : ℕ, p = n ∧ Prime p
def prime_with_two_digits (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ is_prime n

-- The theorem statement
theorem div_240_of_prime_diff (a b : ℕ) (ha : prime_with_two_digits a) (hb : prime_with_two_digits b) (h : a > b) :
  240 ∣ (a^4 - b^4) ∧ ∀ d : ℕ, (d ∣ (a^4 - b^4) → (∀ m n : ℕ, prime_with_two_digits m → prime_with_two_digits n → m > n → d ∣ (m^4 - n^4) ) → d ≤ 240) :=
by
  sorry

end div_240_of_prime_diff_l200_200933


namespace greatest_candies_to_office_l200_200611

-- Problem statement: Prove that the greatest possible number of candies given to the office is 7 when distributing candies among 8 students.

theorem greatest_candies_to_office (n : ℕ) : 
  ∃ k : ℕ, k = n % 8 ∧ k ≤ 7 ∧ k = 7 :=
by
  sorry

end greatest_candies_to_office_l200_200611


namespace door_height_l200_200921

theorem door_height
  (x : ℝ) -- length of the pole
  (pole_eq_diag : x = real.sqrt ((x - 4)^2 + (x - 2)^2)) -- condition 3 (Pythagorean theorem)
  : real.sqrt ((x - 4)^2 + (x - 2)^2) = 10 :=
by
  sorry

end door_height_l200_200921


namespace tan_a_over_tan_b_plus_tan_b_over_tan_a_l200_200702

theorem tan_a_over_tan_b_plus_tan_b_over_tan_a {a b : ℝ} 
  (h1 : (Real.sin a / Real.cos b) + (Real.sin b / Real.cos a) = 2)
  (h2 : (Real.cos a / Real.sin b) + (Real.cos b / Real.sin a) = 4) :
  (Real.tan a / Real.tan b) + (Real.tan b / Real.tan a) = 44 / 5 :=
sorry

end tan_a_over_tan_b_plus_tan_b_over_tan_a_l200_200702


namespace no_even_integers_of_form_3k_plus_4_and_5m_plus_2_l200_200884

theorem no_even_integers_of_form_3k_plus_4_and_5m_plus_2 (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 1000) (h2 : ∃ k : ℕ, n = 3 * k + 4) (h3 : ∃ m : ℕ, n = 5 * m + 2) (h4 : n % 2 = 0) : false :=
sorry

end no_even_integers_of_form_3k_plus_4_and_5m_plus_2_l200_200884


namespace tan_120_deg_l200_200441

theorem tan_120_deg : Real.tan (120 * Real.pi / 180) = -Real.sqrt 3 := by
  sorry

end tan_120_deg_l200_200441


namespace smallest_tournament_with_ordered_group_l200_200497

-- Define the concept of a tennis tournament with n players
def tennis_tournament (n : ℕ) := 
  ∀ (i j : ℕ), (i < n) → (j < n) → (i ≠ j) → (i < j) ∨ (j < i)

-- Define what it means for a group of four players to be "ordered"
def ordered_group (p1 p2 p3 p4 : ℕ) : Prop := 
  ∃ (winner : ℕ), ∃ (loser : ℕ), 
    (winner ≠ loser) ∧ (winner = p1 ∨ winner = p2 ∨ winner = p3 ∨ winner = p4) ∧ 
    (loser = p1 ∨ loser = p2 ∨ loser = p3 ∨ loser = p4)

-- Prove that any tennis tournament with 8 players has an ordered group
theorem smallest_tournament_with_ordered_group : 
  ∀ (n : ℕ), ∀ (tournament : tennis_tournament n), 
    (n ≥ 8) → 
    (∃ (p1 p2 p3 p4 : ℕ), ordered_group p1 p2 p3 p4) :=
  by
  -- proof omitted
  sorry

end smallest_tournament_with_ordered_group_l200_200497


namespace area_proof_l200_200565

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end area_proof_l200_200565


namespace quadratic_rewrite_l200_200729

theorem quadratic_rewrite (x : ℝ) (b c : ℝ) : 
  (x^2 + 1560 * x + 2400 = (x + b)^2 + c) → 
  c / b = -300 :=
by
  sorry

end quadratic_rewrite_l200_200729


namespace distance_covered_downstream_l200_200423

noncomputable def speed_in_still_water := 16 -- km/hr
noncomputable def speed_of_stream := 5 -- km/hr
noncomputable def time_taken := 5 -- hours
noncomputable def effective_speed_downstream := speed_in_still_water + speed_of_stream -- km/hr

theorem distance_covered_downstream :
  (effective_speed_downstream * time_taken = 105) :=
by
  sorry

end distance_covered_downstream_l200_200423


namespace expression_evaluation_l200_200893

-- Define the property to be proved:
theorem expression_evaluation (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by sorry

end expression_evaluation_l200_200893


namespace solve_for_s_l200_200487

theorem solve_for_s (s t : ℚ) (h1 : 7 * s + 8 * t = 150) (h2 : s = 2 * t + 3) : s = 162 / 11 := 
by
  sorry

end solve_for_s_l200_200487


namespace intersection_A_B_union_B_complement_A_l200_200883

open Set

variable (U A B : Set ℝ)

noncomputable def U_def : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}
noncomputable def A_def : Set ℝ := {x | 0 < x ∧ x ≤ 3}
noncomputable def B_def : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem intersection_A_B : (A ∩ B) = {x | 0 < x ∧ x ≤ 1} :=
sorry

theorem union_B_complement_A : B ∪ (U \ A) = {x | (-5 ≤ x ∧ x ≤ 1) ∨ (3 < x ∧ x ≤ 5)} :=
sorry

attribute [irreducible] U_def A_def B_def

end intersection_A_B_union_B_complement_A_l200_200883


namespace product_of_first_two_terms_l200_200966

theorem product_of_first_two_terms (a_7 : ℕ) (d : ℕ) (a_7_eq : a_7 = 17) (d_eq : d = 2) :
  let a_1 := a_7 - 6 * d
  let a_2 := a_1 + d
  a_1 * a_2 = 35 :=
by
  sorry

end product_of_first_two_terms_l200_200966


namespace garden_ratio_l200_200430

theorem garden_ratio (L W : ℝ) (h1 : 2 * L + 2 * W = 180) (h2 : L = 60) : L / W = 2 :=
by
  -- this is where you would put the proof
  sorry

end garden_ratio_l200_200430


namespace sum_of_roots_l200_200527

theorem sum_of_roots (f : ℝ → ℝ) (h_symmetric : ∀ x, f (3 + x) = f (3 - x)) (h_roots : ∃ (roots : Finset ℝ), roots.card = 6 ∧ ∀ r ∈ roots, f r = 0) : 
  ∃ S, S = 18 :=
by
  sorry

end sum_of_roots_l200_200527


namespace number_of_convex_quadrilaterals_with_parallel_sides_l200_200847

-- Define a regular 20-sided polygon
def regular_20_sided_polygon : Type := 
  { p : ℕ // 0 < p ∧ p ≤ 20 }

-- The main theorem statement
theorem number_of_convex_quadrilaterals_with_parallel_sides : 
  ∃ (n : ℕ), n = 765 :=
sorry

end number_of_convex_quadrilaterals_with_parallel_sides_l200_200847


namespace football_even_goal_prob_l200_200805

noncomputable def poisson_even_goal_prob (λ : ℝ) : ℝ :=
  let p := ∑' k, (Real.exp (-λ) * (λ ^ (2 * k))) / (Real.fact (2 * k))
  in p

theorem football_even_goal_prob : 
  poisson_even_goal_prob 2.8 ≈ 0.502 :=
by
  -- Proof of the theorem
  sorry

end football_even_goal_prob_l200_200805


namespace molecular_weight_N2O_correct_l200_200136

-- Define the atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of N2O
def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O

-- Prove the statement
theorem molecular_weight_N2O_correct : molecular_weight_N2O = 44.02 := by
  -- We leave the proof as an exercise (or assumption)
  sorry

end molecular_weight_N2O_correct_l200_200136


namespace perfect_square_A_perfect_square_D_l200_200072

def is_even (n : ℕ) : Prop := n % 2 = 0

def A : ℕ := 2^10 * 3^12 * 7^14
def D : ℕ := 2^20 * 3^16 * 7^12

theorem perfect_square_A : ∃ k : ℕ, A = k^2 :=
by
  sorry

theorem perfect_square_D : ∃ k : ℕ, D = k^2 :=
by
  sorry

end perfect_square_A_perfect_square_D_l200_200072


namespace hyperbola_foci_distance_l200_200720

theorem hyperbola_foci_distance (c : ℝ) (h : c = Real.sqrt 2) : 
  let f1 := (c * Real.sqrt 2, c * Real.sqrt 2)
  let f2 := (-c * Real.sqrt 2, -c * Real.sqrt 2)
  Real.sqrt ((f2.1 - f1.1) ^ 2 + (f2.2 - f1.2) ^ 2) = 4 * Real.sqrt 2 := 
by
  sorry

end hyperbola_foci_distance_l200_200720


namespace sector_angle_l200_200877

-- Defining the conditions
def perimeter (r l : ℝ) : Prop := 2 * r + l = 8
def area (r l : ℝ) : Prop := (1 / 2) * l * r = 4

-- Lean theorem statement
theorem sector_angle (r l θ : ℝ) :
  (perimeter r l) → (area r l) → (θ = l / r) → |θ| = 2 :=
by sorry

end sector_angle_l200_200877


namespace average_speed_joey_round_trip_l200_200686

noncomputable def average_speed_round_trip
  (d : ℝ) (t₁ : ℝ) (r : ℝ) (s₂ : ℝ) : ℝ :=
  2 * d / (t₁ + d / s₂)

-- Lean statement for the proof problem
theorem average_speed_joey_round_trip :
  average_speed_round_trip 6 1 6 12 = 8 := sorry

end average_speed_joey_round_trip_l200_200686


namespace complement_union_correct_l200_200669

-- Define the universal set U, set A, and set B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- The complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The union of the complement of A and set B
def union_complement_U_A_B : Set ℕ := complement_U_A ∪ B

-- State the theorem to prove
theorem complement_union_correct : union_complement_U_A_B = {2, 3, 4, 5} := 
by 
  sorry

end complement_union_correct_l200_200669


namespace perpendicular_vectors_l200_200672

variable {t : ℝ}

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (6, -4)

theorem perpendicular_vectors (ht : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) : t = -5 :=
sorry

end perpendicular_vectors_l200_200672


namespace rectangle_area_l200_200576

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end rectangle_area_l200_200576


namespace neg_exists_n_sq_gt_two_pow_n_l200_200526

open Classical

theorem neg_exists_n_sq_gt_two_pow_n :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by
  sorry

end neg_exists_n_sq_gt_two_pow_n_l200_200526


namespace value_of_s_l200_200444

-- Define the variables as integers (they represent non-zero digits)
variables {a p v e s r : ℕ}

-- Define the conditions as hypotheses
theorem value_of_s (h1 : a + p = v) (h2 : v + e = s) (h3 : s + a = r) (h4 : p + e + r = 14) :
  s = 7 :=
by
  sorry

end value_of_s_l200_200444


namespace no_power_of_q_l200_200337

theorem no_power_of_q (n : ℕ) (hn : n > 0) (q : ℕ) (hq : Prime q) : ¬ (∃ k : ℕ, n^q + ((n-1)/2)^2 = q^k) := 
by
  sorry  -- proof steps are not required as per instructions

end no_power_of_q_l200_200337


namespace percentage_proof_l200_200758

/-- Lean 4 statement proving the percentage -/
theorem percentage_proof :
  ∃ P : ℝ, (800 - (P / 100) * 8000) = 796 ∧ P = 0.05 :=
by
  use 0.05
  sorry

end percentage_proof_l200_200758


namespace number_of_quadratic_PQ_equal_to_PR_l200_200372

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4)

def is_quadratic (Q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, Q = λ x => a * x^2 + b * x + c

theorem number_of_quadratic_PQ_equal_to_PR :
  let possible_Qx_fwds := 4^4
  let non_quadratic_cases := 6
  possible_Qx_fwds - non_quadratic_cases = 250 :=
by
  sorry

end number_of_quadratic_PQ_equal_to_PR_l200_200372


namespace sum_series_eq_one_quarter_l200_200170

noncomputable def series_term (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))

theorem sum_series_eq_one_quarter : 
  (∑' n, series_term (n + 1)) = 1 / 4 :=
by
  sorry

end sum_series_eq_one_quarter_l200_200170


namespace chromium_percentage_in_second_alloy_l200_200680

theorem chromium_percentage_in_second_alloy
  (x : ℝ)
  (h1 : chromium_percentage_in_first_alloy = 15)
  (h2 : weight_first_alloy = 15)
  (h3 : weight_second_alloy = 35)
  (h4 : chromium_percentage_in_new_alloy = 10.1)
  (h5 : total_weight = weight_first_alloy + weight_second_alloy)
  (h6 : chromium_in_new_alloy = chromium_percentage_in_new_alloy / 100 * total_weight)
  (h7 : chromium_in_first_alloy = chromium_percentage_in_first_alloy / 100 * weight_first_alloy)
  (h8 : chromium_in_second_alloy = x / 100 * weight_second_alloy)
  (h9 : chromium_in_new_alloy = chromium_in_first_alloy + chromium_in_second_alloy) :
  x = 8 := by
  sorry

end chromium_percentage_in_second_alloy_l200_200680


namespace trigonometric_identity_example_l200_200312

theorem trigonometric_identity_example :
  (let θ := 30 / 180 * Real.pi in
   (Real.tan θ)^2 - (Real.sin θ)^2) / ((Real.tan θ)^2 * (Real.sin θ)^2) = 4 / 3 :=
by
  let θ := 30 / 180 * Real.pi
  have h_tan2 := Real.tan θ * Real.tan θ
  have h_sin2 : Real.sin θ * Real.sin θ = 1/4 := by sorry
  have h_cos2 : Real.cos θ * Real.cos θ = 3/4 := by sorry
  sorry

end trigonometric_identity_example_l200_200312


namespace smallest_b_l200_200749

theorem smallest_b (b: ℕ) (h1: b > 3) (h2: ∃ n: ℕ, n^3 = 2 * b + 3) : b = 12 :=
sorry

end smallest_b_l200_200749


namespace cubs_win_series_probability_l200_200955

theorem cubs_win_series_probability :
  ∑ k in Finset.range 5, (Nat.choose (4 + k) k * ((3:ℚ) / 5)^5 * ((2:ℚ) / 5)^k) = 243 / 625 :=
by
  sorry

end cubs_win_series_probability_l200_200955


namespace inscribed_circle_radius_l200_200965

noncomputable def calculate_r (a b c : ℝ) : ℝ :=
  let term1 := 1 / a
  let term2 := 1 / b
  let term3 := 1 / c
  let term4 := 3 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c))
  1 / (term1 + term2 + term3 + term4)

theorem inscribed_circle_radius :
  calculate_r 6 10 15 = 30 / (10 * Real.sqrt 26 + 3) :=
by
  sorry

end inscribed_circle_radius_l200_200965


namespace percentage_increase_l200_200434

theorem percentage_increase (new_wage original_wage : ℝ) (h₁ : new_wage = 42) (h₂ : original_wage = 28) :
  ((new_wage - original_wage) / original_wage) * 100 = 50 := by
  sorry

end percentage_increase_l200_200434


namespace max_weak_quartets_120_l200_200296

noncomputable def max_weak_quartets (n : ℕ) : ℕ :=
  -- Placeholder definition to represent the maximum weak quartets
  sorry  -- To be replaced with the actual mathematical definition

theorem max_weak_quartets_120 : max_weak_quartets 120 = 4769280 := by
  sorry

end max_weak_quartets_120_l200_200296


namespace find_initial_amount_l200_200184

-- defining conditions
def compound_interest (A P : ℝ) (r : ℝ) (n t : ℕ) : ℝ :=
  A - P

-- main theorem to prove the principal amount
theorem find_initial_amount 
  (A P : ℝ) (r : ℝ)
  (n t : ℕ)
  (h_P : A = P * (1 + r / n)^t)
  (compound_interest_eq : A - P = 1785.98)
  (r_eq : r = 0.20)
  (n_eq : n = 1)
  (t_eq : t = 5) :
  P = 1200 :=
by
  sorry

end find_initial_amount_l200_200184


namespace probability_even_goals_is_approximately_l200_200790

noncomputable def probability_even_goals (λ : ℝ) : ℝ :=
  let p : ℝ := ∑ k in (nat.filter even), (λ ^ k * real.exp (-λ)) / (nat.fact k)
  in p

def λ : ℝ := 2.8

theorem probability_even_goals_is_approximately:
  probability_even_goals λ ≈ 0.502 :=
sorry

end probability_even_goals_is_approximately_l200_200790


namespace add_and_simplify_fractions_l200_200245

theorem add_and_simplify_fractions :
  (1 / 462) + (23 / 42) = 127 / 231 :=
by
  sorry

end add_and_simplify_fractions_l200_200245


namespace two_digit_number_representation_l200_200036

-- Define the conditions and the problem statement in Lean 4
def units_digit (n : ℕ) := n % 10
def tens_digit (n : ℕ) := (n / 10) % 10

theorem two_digit_number_representation (x : ℕ) (h : x < 10) :
  ∃ n : ℕ, units_digit n = x ∧ tens_digit n = 2 * x ^ 2 ∧ n = 20 * x ^ 2 + x :=
by {
  sorry
}

end two_digit_number_representation_l200_200036


namespace max_remaining_numbers_l200_200235

/-- 
The board initially has numbers 1, 2, 3, ..., 235.
Among the remaining numbers, no number is divisible by the difference of any two others.
Prove that the maximum number of numbers that could remain on the board is 118.
-/
theorem max_remaining_numbers : 
  ∃ S : set ℕ, (∀ a ∈ S, 1 ≤ a ∧ a ≤ 235) ∧ (∀ a b ∈ S, a ≠ b → ¬ ∃ d, d ∣ (a - b)) ∧ 
  ∃ T : set ℕ, S ⊆ T ∧ T ⊆ finset.range 236 ∧ T.card = 118 := 
sorry

end max_remaining_numbers_l200_200235


namespace price_of_paint_models_max_boxes_of_paint_A_l200_200534

-- Define the conditions as hypotheses
def price_paint_model_A_B (x y : ℕ) : Prop :=
  x + 2 * y = 56 ∧ 2 * x + y = 64

def total_cost_constraint (x y m : ℕ) : Prop :=
  24 * m + 16 * (200 - m) ≤ 3920

-- Prove the prices of the paint models given the conditions
theorem price_of_paint_models :
  ∃ (x y : ℕ), price_paint_model_A_B x y ∧ x = 24 ∧ y = 16 :=
by
  sorry

-- Prove the maximum number of boxes of paint model A the school can purchase given the total cost constraint
theorem max_boxes_of_paint_A :
  ∃ (m : ℕ), total_cost_constraint 24 16 m ∧ m = 90 :=
by
  sorry

end price_of_paint_models_max_boxes_of_paint_A_l200_200534


namespace bus_speed_including_stoppages_l200_200637

theorem bus_speed_including_stoppages :
  ∀ (s t : ℝ), s = 75 → t = 24 → (s * ((60 - t) / 60)) = 45 :=
by
  intros s t hs ht
  rw [hs, ht]
  sorry

end bus_speed_including_stoppages_l200_200637


namespace subcommittee_count_l200_200773

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem subcommittee_count : 
  let R := 10
  let D := 4
  let subR := 4
  let subD := 2
  binomial R subR * binomial D subD = 1260 := 
by
  sorry

end subcommittee_count_l200_200773


namespace rectangle_area_l200_200554

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_area_l200_200554


namespace rectangle_area_l200_200561

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l200_200561


namespace cubes_sum_l200_200716

theorem cubes_sum (a b c : ℝ) (h1 : a + b + c = 8) (h2 : a * b + a * c + b * c = 9) (h3 : a * b * c = -18) :
  a^3 + b^3 + c^3 = 242 :=
by
  sorry

end cubes_sum_l200_200716


namespace first_day_bacteria_exceeds_200_l200_200355

noncomputable def bacteria_growth (n : ℕ) : ℕ := 2 * 3^n

theorem first_day_bacteria_exceeds_200 : ∃ n : ℕ, 2 * 3^n > 200 ∧ ∀ m : ℕ, m < n → 2 * 3^m ≤ 200 :=
by
  -- sorry for skipping proof
  sorry

end first_day_bacteria_exceeds_200_l200_200355


namespace quadratic_has_exactly_one_root_l200_200376

noncomputable def discriminant (b c : ℝ) : ℝ :=
b^2 - 4 * c

noncomputable def f (x b c : ℝ) : ℝ :=
x^2 + b * x + c

noncomputable def transformed_f (x b c : ℝ) : ℝ :=
(x - 2020)^2 + b * (x - 2020) + c

theorem quadratic_has_exactly_one_root (b c : ℝ)
  (h_discriminant : discriminant b c = 2020) :
  ∃! x : ℝ, f (x - 2020) b c + f x b c = 0 :=
sorry

end quadratic_has_exactly_one_root_l200_200376


namespace longer_diagonal_of_rhombus_l200_200034

theorem longer_diagonal_of_rhombus
  (A : ℝ) (r1 r2 : ℝ) (x : ℝ)
  (hA : A = 135)
  (h_ratio : r1 = 5) (h_ratio2 : r2 = 3)
  (h_area : (1/2) * (r1 * x) * (r2 * x) = A) :
  r1 * x = 15 :=
by
  sorry

end longer_diagonal_of_rhombus_l200_200034


namespace log_9_256_eq_4_log_2_3_l200_200448

noncomputable def logBase9Base2Proof : Prop :=
  (Real.log 256 / Real.log 9 = 4 * (Real.log 3 / Real.log 2))

theorem log_9_256_eq_4_log_2_3 : logBase9Base2Proof :=
by
  sorry

end log_9_256_eq_4_log_2_3_l200_200448


namespace natalia_total_distance_l200_200099

theorem natalia_total_distance :
  let dist_mon := 40
  let bonus_mon := 0.05 * dist_mon
  let effective_mon := dist_mon + bonus_mon
  
  let dist_tue := 50
  let bonus_tue := 0.03 * dist_tue
  let effective_tue := dist_tue + bonus_tue
  
  let dist_wed := dist_tue / 2
  let bonus_wed := 0.07 * dist_wed
  let effective_wed := dist_wed + bonus_wed
  
  let dist_thu := dist_mon + dist_wed
  let bonus_thu := 0.04 * dist_thu
  let effective_thu := dist_thu + bonus_thu
  
  let dist_fri := 1.2 * dist_thu
  let bonus_fri := 0.06 * dist_fri
  let effective_fri := dist_fri + bonus_fri
  
  let dist_sat := 0.75 * dist_fri
  let bonus_sat := 0.02 * dist_sat
  let effective_sat := dist_sat + bonus_sat
  
  let dist_sun := dist_sat - dist_wed
  let bonus_sun := 0.10 * dist_sun
  let effective_sun := dist_sun + bonus_sun
  
  effective_mon + effective_tue + effective_wed + effective_thu + effective_fri + effective_sat + effective_sun = 367.05 :=
by
  sorry

end natalia_total_distance_l200_200099


namespace correct_calculation_l200_200988

theorem correct_calculation (a b x y : ℝ) :
  (7 * a^2 * b - 7 * b * a^2 = 0) ∧ 
  (¬ (6 * a + 4 * b = 10 * a * b)) ∧ 
  (¬ (7 * x^2 * y - 3 * x^2 * y = 4 * x^4 * y^2)) ∧ 
  (¬ (8 * x^2 + 8 * x^2 = 16 * x^4)) :=
sorry

end correct_calculation_l200_200988


namespace minimize_y_l200_200225

variable (a b x : ℝ)

def y := (x - a)^2 + (x - b)^2

theorem minimize_y : ∃ x : ℝ, (∀ (x' : ℝ), y x a b ≤ y x' a b) ∧ x = (a + b) / 2 := by
  sorry

end minimize_y_l200_200225


namespace percentage_female_guests_from_jay_family_l200_200707

def total_guests : ℕ := 240
def female_guests_percentage : ℕ := 60
def female_guests_from_jay_family : ℕ := 72

theorem percentage_female_guests_from_jay_family :
  (female_guests_from_jay_family : ℚ) / (total_guests * (female_guests_percentage / 100) : ℚ) * 100 = 50 := by
  sorry

end percentage_female_guests_from_jay_family_l200_200707


namespace sequence_problem_l200_200045

theorem sequence_problem :
  7 * 9 * 11 + (7 + 9 + 11) = 720 :=
by
  sorry

end sequence_problem_l200_200045


namespace largest_product_of_three_l200_200814

theorem largest_product_of_three :
  ∃ (a b c : ℤ), a ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 b ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 c ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
                 a * b * c = 90 := 
sorry

end largest_product_of_three_l200_200814


namespace temperature_difference_correct_l200_200276

def refrigerator_temp : ℝ := 3
def freezer_temp : ℝ := -10
def temperature_difference : ℝ := refrigerator_temp - freezer_temp

theorem temperature_difference_correct : temperature_difference = 13 := 
by
  sorry

end temperature_difference_correct_l200_200276


namespace girls_in_class_l200_200731

theorem girls_in_class (B G : ℕ) 
  (h1 : G = B + 3) 
  (h2 : G + B = 41) : 
  G = 22 := 
sorry

end girls_in_class_l200_200731


namespace divisibility_equiv_l200_200650

open Nat

theorem divisibility_equiv (m n : ℕ) : 
  (2^n - 1) % ((2^m - 1)^2) = 0 ↔ n % (m * (2^m - 1)) = 0 := 
sorry

end divisibility_equiv_l200_200650


namespace city_map_distance_example_l200_200381

variable (distance_on_map : ℝ)
variable (scale : ℝ)
variable (actual_distance : ℝ)

theorem city_map_distance_example
  (h1 : distance_on_map = 16)
  (h2 : scale = 1 / 10000)
  (h3 : actual_distance = distance_on_map / scale) :
  actual_distance = 1.6 * 10^3 :=
by
  sorry

end city_map_distance_example_l200_200381


namespace not_necessarily_circle_l200_200025

open Set

-- Definitions of conditions
def convex_figure (F : Set ℝ) : Prop :=
  Convex ℝ F

def equilateral_triangle (T : Set ℝ) : Prop :=
  -- A placeholder definition for an equilateral triangle with side length 1
  sorry 

def can_translate_triangle_to_boundary (F T : Set ℝ) : Prop :=
  ∃ t : ℝ × ℝ, ∀ v ∈ T, v + t ∈ boundary F

-- Main proof problem
theorem not_necessarily_circle (F : Set ℝ) :
  convex_figure F →
  (∀ T, equilateral_triangle T → can_translate_triangle_to_boundary F T) →
  ¬(∀ x, F = metric.ball x 1) :=
by
  intros h_convex h_property
  -- To be proved: there exists non-circle convex figures with the given property.
  sorry

end not_necessarily_circle_l200_200025


namespace pipes_fill_tank_in_one_hour_l200_200947

theorem pipes_fill_tank_in_one_hour (p q r s : ℝ) (hp : p = 1/2) (hq : q = 1/4) (hr : r = 1/12) (hs : s = 1/6) :
  1 / (p + q + r + s) = 1 :=
by
  sorry

end pipes_fill_tank_in_one_hour_l200_200947


namespace door_height_eight_l200_200922

theorem door_height_eight (x : ℝ) (h w : ℝ) (H1 : w = x - 4) (H2 : h = x - 2) (H3 : x^2 = (x - 4)^2 + (x - 2)^2) : h = 8 :=
by
  sorry

end door_height_eight_l200_200922


namespace negation_correct_l200_200275

theorem negation_correct (x : ℝ) : -(3 * x - 2) = -3 * x + 2 := 
by sorry

end negation_correct_l200_200275


namespace maxwell_age_l200_200677

theorem maxwell_age (M : ℕ) (h1 : ∃ n : ℕ, n = M + 2) (h2 : ∃ k : ℕ, k = 4) (h3 : (M + 2) = 2 * 4) : M = 6 :=
sorry

end maxwell_age_l200_200677


namespace price_of_paint_models_max_boxes_of_paint_A_l200_200533

-- Define the conditions as hypotheses
def price_paint_model_A_B (x y : ℕ) : Prop :=
  x + 2 * y = 56 ∧ 2 * x + y = 64

def total_cost_constraint (x y m : ℕ) : Prop :=
  24 * m + 16 * (200 - m) ≤ 3920

-- Prove the prices of the paint models given the conditions
theorem price_of_paint_models :
  ∃ (x y : ℕ), price_paint_model_A_B x y ∧ x = 24 ∧ y = 16 :=
by
  sorry

-- Prove the maximum number of boxes of paint model A the school can purchase given the total cost constraint
theorem max_boxes_of_paint_A :
  ∃ (m : ℕ), total_cost_constraint 24 16 m ∧ m = 90 :=
by
  sorry

end price_of_paint_models_max_boxes_of_paint_A_l200_200533


namespace evaluate_expression_l200_200449

theorem evaluate_expression : (2^2010 * 3^2012 * 25) / 6^2011 = 37.5 := by
  sorry

end evaluate_expression_l200_200449


namespace option_d_is_right_triangle_l200_200364

noncomputable def right_triangle (a b c : ℝ) : Prop :=
  a^2 + c^2 = b^2

theorem option_d_is_right_triangle (a b c : ℝ) (h : a^2 = b^2 - c^2) :
  right_triangle a b c :=
by
  sorry

end option_d_is_right_triangle_l200_200364


namespace necessary_but_not_sufficient_l200_200346

def quadratic_inequality (x : ℝ) : Prop :=
  x^2 - 3 * x + 2 < 0

def necessary_condition_A (x : ℝ) : Prop :=
  -1 < x ∧ x < 2

def necessary_condition_D (x : ℝ) : Prop :=
  -2 < x ∧ x < 2

theorem necessary_but_not_sufficient :
  (∀ x, quadratic_inequality x → ∃ x, necessary_condition_A x ∧ ¬(quadratic_inequality x ∧ necessary_condition_A x)) ∧ 
  (∀ x, quadratic_inequality x → ∃ x, necessary_condition_D x ∧ ¬(quadratic_inequality x ∧ necessary_condition_D x)) :=
sorry

end necessary_but_not_sufficient_l200_200346


namespace students_count_geometry_history_science_l200_200453

noncomputable def number_of_students (geometry_only history_only science_only 
                                      geometry_and_history geometry_and_science : ℕ) : ℕ :=
  geometry_only + history_only + science_only

theorem students_count_geometry_history_science (geometry_total history_only science_only 
                                                 geometry_and_history geometry_and_science : ℕ) :
  geometry_total = 30 →
  geometry_and_history = 15 →
  history_only = 15 →
  geometry_and_science = 8 →
  science_only = 10 →
  number_of_students (geometry_total - geometry_and_history - geometry_and_science)
                     history_only
                     science_only = 32 :=
by
  sorry

end students_count_geometry_history_science_l200_200453


namespace find_G16_l200_200935

variable (G : ℝ → ℝ)

def condition1 : Prop := G 8 = 28

def condition2 : Prop := ∀ x : ℝ, 
  (x^2 + 8*x + 16) ≠ 0 → 
  (G (4*x) / G (x + 4) = 16 - (64*x + 80) / (x^2 + 8*x + 16))

theorem find_G16 (h1 : condition1 G) (h2 : condition2 G) : G 16 = 120 :=
sorry

end find_G16_l200_200935


namespace complete_collection_probability_l200_200735

namespace Stickers

open scoped Classical

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.Ico 1 (n + 1))

def combinations (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_complete_collection :
  ℚ := (combinations 6 6 * combinations 12 4) / combinations 18 10

theorem complete_collection_probability :
  probability_complete_collection = (5 / 442 : ℚ) := by
  sorry

end Stickers

end complete_collection_probability_l200_200735


namespace Geraldine_more_than_Jazmin_l200_200066

-- Define the number of dolls Geraldine and Jazmin have
def Geraldine_dolls : ℝ := 2186.0
def Jazmin_dolls : ℝ := 1209.0

-- State the theorem we need to prove
theorem Geraldine_more_than_Jazmin :
  Geraldine_dolls - Jazmin_dolls = 977.0 := 
by
  sorry

end Geraldine_more_than_Jazmin_l200_200066


namespace right_angled_triangle_setB_l200_200142

def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c

theorem right_angled_triangle_setB :
  isRightAngledTriangle 1 1 (Real.sqrt 2) ∧
  ¬isRightAngledTriangle 1 2 3 ∧
  ¬isRightAngledTriangle 6 8 11 ∧
  ¬isRightAngledTriangle 2 3 4 :=
by
  sorry

end right_angled_triangle_setB_l200_200142


namespace player1_wins_game_533_player1_wins_game_1000_l200_200409

-- Defining a structure for the game conditions
structure Game :=
  (target_sum : ℕ)
  (player1_wins_optimal : Bool)

-- Definition of the game scenarios
def game_533 := Game.mk 533 true
def game_1000 := Game.mk 1000 true

-- Theorem statements for the respective games
theorem player1_wins_game_533 : game_533.player1_wins_optimal :=
by sorry

theorem player1_wins_game_1000 : game_1000.player1_wins_optimal :=
by sorry

end player1_wins_game_533_player1_wins_game_1000_l200_200409


namespace simplify_expression_l200_200951

theorem simplify_expression :
  (256:ℝ)^(1/4) * (125:ℝ)^(1/2) = 20 * Real.sqrt 5 := 
by {
  have h1 : (256:ℝ) = 2^8 := by norm_num,
  have h2 : (125:ℝ) = 5^3 := by norm_num,
  rw [h1, h2], 
  simp,
  have h3 : (2 : ℝ) ^ 8 ^ (1 / 4) = 4 := by norm_num,
  have h4 : (5 : ℝ) ^ 3 ^ (1 / 2) = 5 * Real.sqrt 5 := by norm_num,
  rw [h3, h4],
  norm_num,
}

end simplify_expression_l200_200951


namespace Tom_money_made_l200_200090

theorem Tom_money_made (money_last_week money_now : ℕ) (h1 : money_last_week = 74) (h2 : money_now = 160) : 
  (money_now - money_last_week = 86) :=
by 
  sorry

end Tom_money_made_l200_200090


namespace probability_gcd_is_one_l200_200408

-- Definitions for the problem
def set := {1, 2, 3, 4, 5, 6, 7, 8}
def pairs := { (a, b) | a ∈ set ∧ b ∈ set ∧ a < b }
def gcd_is_one (a b : ℕ) : Prop := Nat.gcd a b = 1
def valid_pairs := { (a, b) ∈ pairs | gcd_is_one a b }

-- Lean 4 statement for the proof problem
theorem probability_gcd_is_one : 
  (valid_pairs.card : ℚ) / (pairs.card : ℚ) = 3 / 4 :=
sorry -- To be proven

end probability_gcd_is_one_l200_200408


namespace max_remained_numbers_l200_200236

theorem max_remained_numbers (S : Finset ℕ) (hSubset : S ⊆ Finset.range 236)
  (hCondition : ∀ a b c ∈ S, a ≠ b → a ≠ c → b ≠ c → ¬(b - a ∣ c)) : S.card ≤ 118 := 
sorry

end max_remained_numbers_l200_200236


namespace valid_rod_count_l200_200931

open Nat

theorem valid_rod_count :
  ∃ valid_rods : Finset ℕ,
    (∀ d ∈ valid_rods, 6 ≤ d ∧ d < 35 ∧ d ≠ 5 ∧ d ≠ 10 ∧ d ≠ 20) ∧ 
    valid_rods.card = 26 := sorry

end valid_rod_count_l200_200931


namespace box_volume_l200_200976

theorem box_volume (l w h : ℝ)
  (A1 : l * w = 30)
  (A2 : w * h = 20)
  (A3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end box_volume_l200_200976


namespace billy_sleep_total_l200_200213

theorem billy_sleep_total
  (h₁ : ∀ n : ℕ, n = 1 → ∃ h : ℕ, h = 6)
  (h₂ : ∀ n : ℕ, n = 2 → ∃ h : ℕ, h = (6 + 2))
  (h₃ : ∀ n : ℕ, n = 3 → ∃ h : ℕ, h = ((6 + 2) / 2))
  (h₄ : ∀ n : ℕ, n = 4 → ∃ h : ℕ, h = (((6 + 2) / 2) * 3)) :
  ∑ n in {1, 2, 3, 4}, (classical.some (h₁ n 1) + classical.some (h₂ n 2) + classical.some (h₃ n 3) + classical.some (h₄ n 4)) = 30 :=
by sorry

end billy_sleep_total_l200_200213


namespace different_movies_count_l200_200319

theorem different_movies_count 
    (d_movies : ℕ) (h_movies : ℕ) (a_movies : ℕ) (b_movies : ℕ) (c_movies : ℕ) 
    (together_movies : ℕ) (dha_movies : ℕ) (bc_movies : ℕ) 
    (db_movies : ℕ) (ac_movies : ℕ)
    (H_d : d_movies = 20) (H_h : h_movies = 26) (H_a : a_movies = 35) 
    (H_b : b_movies = 29) (H_c : c_movies = 16)
    (H_together : together_movies = 5)
    (H_dha : dha_movies = 4) (H_bc : bc_movies = 3) 
    (H_db : db_movies = 2) (H_ac : ac_movies = 4) :
    d_movies + h_movies + a_movies + b_movies + c_movies 
    - 4 * together_movies - 3 * dha_movies - 2 * bc_movies - db_movies - 3 * ac_movies = 74 := by sorry

end different_movies_count_l200_200319


namespace probability_even_goals_is_approximately_l200_200789

noncomputable def probability_even_goals (λ : ℝ) : ℝ :=
  let p : ℝ := ∑ k in (nat.filter even), (λ ^ k * real.exp (-λ)) / (nat.fact k)
  in p

def λ : ℝ := 2.8

theorem probability_even_goals_is_approximately:
  probability_even_goals λ ≈ 0.502 :=
sorry

end probability_even_goals_is_approximately_l200_200789


namespace find_a_l200_200343

theorem find_a (x : ℝ) (a : ℝ)
  (h1 : 3 * x - 4 = a)
  (h2 : (x + a) / 3 = 1)
  (h3 : (x = (a + 4) / 3) → (x = 3 - a → ((a + 4) / 3 = 2 * (3 - a)))) :
  a = 2 :=
sorry

end find_a_l200_200343


namespace find_number_l200_200460

noncomputable def N := 953.87

theorem find_number (h : (0.47 * N - 0.36 * 1412) + 65 = 5) : N = 953.87 := sorry

end find_number_l200_200460


namespace find_b_l200_200654

noncomputable def ellipse_foci (a b : ℝ) (hb : b > 0) (hab : a > b) : Prop :=
∃ (F1 F2 P : ℝ×ℝ), 
    (∃ (h : a > b), (2 * b^2 + 9 = a^2)) ∧ 
    (dist P F1 + dist P F2 = 2 * a) ∧ 
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧ 
    (2 * 4 * (a^2 - b^2) = 36)

theorem find_b (a b : ℝ) (hb : b > 0) (hab : a > b) : 
    ellipse_foci a b hb hab → b = 3 :=
by
  sorry

end find_b_l200_200654


namespace sum_a_c_e_l200_200078

theorem sum_a_c_e {a b c d e f : ℝ} 
  (h1 : a / b = 2) 
  (h2 : c / d = 2) 
  (h3 : e / f = 2) 
  (h4 : b + d + f = 5) : 
  a + c + e = 10 :=
by
  -- Proof goes here
  sorry

end sum_a_c_e_l200_200078


namespace trigonometric_identity_l200_200316

theorem trigonometric_identity :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  t30 = 1 / Real.sqrt 3 →
  s30 = 1 / 2 →
  (t30 ^ 2 - s30 ^ 2) / (t30 ^ 2 * s30 ^ 2) = 1 :=
by
  intros t30_eq s30_eq
  -- Proof would go here
  sorry

end trigonometric_identity_l200_200316


namespace cindy_arrival_speed_l200_200042

def cindy_speed (d t1 t2 t3: ℕ) : Prop :=
  (d = 20 * t1) ∧ 
  (d = 10 * (t2 + 3 / 4)) ∧
  (t3 = t1 + 1 / 2) ∧
  (20 * t1 = 10 * (t2 + 3 / 4)) -> 
  (d / (t3) = 12)

theorem cindy_arrival_speed (t1 t2: ℕ) (h₁: t2 = t1 + 3 / 4) (d: ℕ) (h2: d = 20 * t1) (h3: t3 = t1 + 1 / 2) :
  cindy_speed d t1 t2 t3 := by
  sorry

end cindy_arrival_speed_l200_200042


namespace quadratic_condition_l200_200250

theorem quadratic_condition (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * x + 3 = 0) → a ≠ 0 :=
by 
  intro h
  -- Proof will be here
  sorry

end quadratic_condition_l200_200250


namespace cone_generatrix_length_l200_200870

/-- Given that the base radius of a cone is √2, 
    and its lateral surface is unfolded into a semicircle,
    prove that the length of the generatrix of the cone is 2√2. -/
theorem cone_generatrix_length (r l : ℝ) 
  (h1 : r = real.sqrt 2)
  (h2 : 2 * real.pi * r = real.pi * l) :
  l = 2 * real.sqrt 2 :=
by
  sorry -- Proof steps go here (not required in the current context)


end cone_generatrix_length_l200_200870


namespace integer_type_l200_200285

theorem integer_type (f : ℕ) (h : f = 14) (x : ℕ) (hx : 3150 * f = x * x) : f > 0 :=
by
  sorry

end integer_type_l200_200285


namespace intersection_and_complement_find_m_l200_200073

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

end intersection_and_complement_find_m_l200_200073


namespace determine_a_l200_200725

theorem determine_a (x y a : ℝ) 
  (h1 : x + y = a)
  (h2 : x^3 + y^3 = a)
  (h3 : x^5 + y^5 = a) : 
  a = 0 := 
sorry

end determine_a_l200_200725


namespace average_retail_price_l200_200304

theorem average_retail_price 
  (products : Fin 20 → ℝ)
  (h1 : ∀ i, 400 ≤ products i) 
  (h2 : ∃ s : Finset (Fin 20), s.card = 10 ∧ ∀ i ∈ s, products i < 1000)
  (h3 : ∃ i, products i = 11000): 
  (Finset.univ.sum products) / 20 = 1200 := 
by
  sorry

end average_retail_price_l200_200304


namespace total_selling_price_is_18000_l200_200293

def cost_price_per_meter : ℕ := 50
def loss_per_meter : ℕ := 5
def meters_sold : ℕ := 400

def selling_price_per_meter := cost_price_per_meter - loss_per_meter

def total_selling_price := selling_price_per_meter * meters_sold

theorem total_selling_price_is_18000 :
  total_selling_price = 18000 :=
sorry

end total_selling_price_is_18000_l200_200293


namespace intersection_P_Q_range_a_l200_200074

def set_P : Set ℝ := { x | 2 * x^2 - 3 * x + 1 ≤ 0 }
def set_Q (a : ℝ) : Set ℝ := { x | (x - a) * (x - a - 1) ≤ 0 }

theorem intersection_P_Q (a : ℝ) (h_a : a = 1) :
  set_P ∩ set_Q 1 = {1} :=
sorry

theorem range_a (a : ℝ) :
  (∀ x : ℝ, x ∈ set_P → x ∈ set_Q a) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end intersection_P_Q_range_a_l200_200074


namespace football_even_goal_probability_l200_200799

noncomputable def poisson_even_goal_probability : ℝ :=
  let λ := 2.8 in
  (1 + Real.exp (-2 * λ)) / 2

theorem football_even_goal_probability :
  let λ := 2.8 in
  let N := Poisson λ in  -- Define N as a Poisson random variable with parameter λ
  (∑ k : ℕ in (range (2*k)).filter (λ k, N.P k > 0), N.P k) = 
  poisson_even_goal_probability :=
by
  sorry

end football_even_goal_probability_l200_200799


namespace range_of_a_inequality_l200_200351

theorem range_of_a_inequality (a : ℝ) (h : ∀ x ∈ Ioo 0 1, (x + real.log a) / real.exp x - (a * real.log x) / x > 0) :
  a ∈ set.Ico (1 / real.exp 1) 1 :=
sorry

end range_of_a_inequality_l200_200351


namespace max_value_of_f_at_0_min_value_of_f_on_neg_inf_to_0_range_of_a_for_ineq_l200_200478

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 5*x + 5) / Real.exp x

theorem max_value_of_f_at_0 :
  f 0 = 5 := by
  sorry

theorem min_value_of_f_on_neg_inf_to_0 :
  f (-3) = -Real.exp 3 := by
  sorry

theorem range_of_a_for_ineq :
  ∀ x : ℝ, x^2 + 5*x + 5 - a * Real.exp x ≥ 0 ↔ a ≤ -Real.exp 3 := by
  sorry

end max_value_of_f_at_0_min_value_of_f_on_neg_inf_to_0_range_of_a_for_ineq_l200_200478


namespace smallest_pos_mult_of_31_mod_97_l200_200755

theorem smallest_pos_mult_of_31_mod_97 {k : ℕ} (h : 31 * k % 97 = 6) : 31 * k = 2015 :=
sorry

end smallest_pos_mult_of_31_mod_97_l200_200755


namespace football_even_goal_prob_l200_200808

noncomputable def poisson_even_goal_prob (λ : ℝ) : ℝ :=
  let p := ∑' k, (Real.exp (-λ) * (λ ^ (2 * k))) / (Real.fact (2 * k))
  in p

theorem football_even_goal_prob : 
  poisson_even_goal_prob 2.8 ≈ 0.502 :=
by
  -- Proof of the theorem
  sorry

end football_even_goal_prob_l200_200808


namespace non_real_roots_interval_l200_200902

theorem non_real_roots_interval (b : ℝ) : 
  (∀ x : ℝ, ∃ (a c : ℝ), a ≠ 0 ∧ b^2 - 4 * a * c < 0 ∧ (x^2 + b * x + c = 0))  →
  b ∈ Ioo (-8 : ℝ) (8 : ℝ) :=
by
  sorry

end non_real_roots_interval_l200_200902


namespace shifted_polynomial_sum_l200_200013

theorem shifted_polynomial_sum (a b c : ℝ) :
  (∀ x : ℝ, (3 * x^2 + 2 * x + 5) = (a * (x + 5)^2 + b * (x + 5) + c)) →
  a + b + c = 125 :=
by
  sorry

end shifted_polynomial_sum_l200_200013


namespace calculate_expression_l200_200828

theorem calculate_expression : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := 
by
  sorry

end calculate_expression_l200_200828


namespace tan_sin_div_l200_200306

theorem tan_sin_div (tan_30_sq : Real := (Real.sin 30 / Real.cos 30) ^ 2)
                    (sin_30_sq : Real := (Real.sin 30) ^ 2):
                    tan_30_sq = (1/3) → sin_30_sq = (1/4) → 
                    (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end tan_sin_div_l200_200306


namespace jill_marathon_time_l200_200503

def jack_marathon_distance : ℝ := 42
def jack_marathon_time : ℝ := 6
def speed_ratio : ℝ := 0.7

theorem jill_marathon_time :
  ∃ t_jill : ℝ, (t_jill = jack_marathon_distance / (jack_marathon_distance / jack_marathon_time / speed_ratio)) ∧
  t_jill = 4.2 :=
by
  -- The proof goes here
  sorry

end jill_marathon_time_l200_200503


namespace area_proof_l200_200567

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end area_proof_l200_200567


namespace lines_perpendicular_if_one_perpendicular_and_one_parallel_l200_200475

def Line : Type := sorry  -- Define the type representing lines
def Plane : Type := sorry  -- Define the type representing planes

def is_perpendicular_to_plane (a : Line) (α : Plane) : Prop := sorry  -- Definition for a line being perpendicular to a plane
def is_parallel_to_plane (b : Line) (α : Plane) : Prop := sorry  -- Definition for a line being parallel to a plane
def is_perpendicular (a b : Line) : Prop := sorry  -- Definition for a line being perpendicular to another line

theorem lines_perpendicular_if_one_perpendicular_and_one_parallel 
  (a b : Line) (α : Plane) 
  (h1 : is_perpendicular_to_plane a α) 
  (h2 : is_parallel_to_plane b α) : 
  is_perpendicular a b := 
sorry

end lines_perpendicular_if_one_perpendicular_and_one_parallel_l200_200475


namespace geometric_sequence_sum_l200_200196

open Real

variable {a a5 a3 a4 S4 q : ℝ}

theorem geometric_sequence_sum (h1 : q < 1)
                             (h2 : a + a5 = 20)
                             (h3 : a3 * a5 = 64) :
                             S4 = 120 := by
  sorry

end geometric_sequence_sum_l200_200196


namespace solution_set_abs_inequality_l200_200122

theorem solution_set_abs_inequality :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 0} :=
sorry

end solution_set_abs_inequality_l200_200122


namespace michael_birth_year_l200_200117

theorem michael_birth_year (first_AMC8_year : ℕ) (tenth_AMC8_year : ℕ) (age_during_tenth_AMC8 : ℕ) 
  (h1 : first_AMC8_year = 1985) (h2 : tenth_AMC8_year = (first_AMC8_year + 9)) (h3 : age_during_tenth_AMC8 = 15) :
  (tenth_AMC8_year - age_during_tenth_AMC8) = 1979 :=
by
  sorry

end michael_birth_year_l200_200117


namespace author_earnings_calculation_l200_200165

open Real

namespace AuthorEarnings

def paperCoverCopies  : ℕ := 32000
def paperCoverPrice   : ℝ := 0.20
def paperCoverPercent : ℝ := 0.06

def hardCoverCopies   : ℕ := 15000
def hardCoverPrice    : ℝ := 0.40
def hardCoverPercent  : ℝ := 0.12

def total_earnings_paper_cover : ℝ := paperCoverCopies * paperCoverPrice
def earnings_paper_cover : ℝ := total_earnings_paper_cover * paperCoverPercent

def total_earnings_hard_cover : ℝ := hardCoverCopies * hardCoverPrice
def earnings_hard_cover : ℝ := total_earnings_hard_cover * hardCoverPercent

def author_total_earnings : ℝ := earnings_paper_cover + earnings_hard_cover

theorem author_earnings_calculation : author_total_earnings = 1104 := by
  sorry

end AuthorEarnings

end author_earnings_calculation_l200_200165


namespace billy_sleep_total_l200_200210

theorem billy_sleep_total :
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  day1 + day2 + day3 + day4 = 30 :=
by
  -- Definitions
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  -- Assertion
  have h : day1 + day2 + day3 + day4 = 30 := sorry
  exact h

end billy_sleep_total_l200_200210


namespace calculate_expression_l200_200827

theorem calculate_expression : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := 
by
  sorry

end calculate_expression_l200_200827


namespace everyone_can_cross_l200_200383

-- Define each agent
inductive Agent
| C   -- Princess Sonya
| K (i : Fin 8) -- Knights numbered 1 to 7

open Agent

-- Define friendships
def friends (a b : Agent) : Prop :=
  match a, b with
  | C, (K 4) => False
  | (K 4), C => False
  | _, _ => (∃ i : Fin 8, a = K i ∧ b = K (i+1)) ∨ (∃ i : Fin 7, a = K (i+1) ∧ b = K i) ∨ a = C ∨ b = C

-- Define the crossing conditions
def boatCanCarry : List Agent → Prop
| [a, b] => friends a b
| [a, b, c] => friends a b ∧ friends b c ∧ friends a c
| _ => False

-- The main statement to prove
theorem everyone_can_cross (agents : List Agent) (steps : List (List Agent)) :
  agents = [C, K 0, K 1, K 2, K 3, K 4, K 5, K 6, K 7] →
  (∀ step ∈ steps, boatCanCarry step) →
  (∃ final_state : List (List Agent), final_state = [[C, K 0, K 1, K 2, K 3, K 4, K 5, K 6, K 7]]) :=
by 
  -- The proof is omitted.
  sorry

end everyone_can_cross_l200_200383


namespace EDTA_Ca2_complex_weight_l200_200628

-- Definitions of atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_Ca : ℝ := 40.08

-- Number of atoms in EDTA
def num_atoms_C : ℝ := 10
def num_atoms_H : ℝ := 16
def num_atoms_N : ℝ := 2
def num_atoms_O : ℝ := 8

-- Molecular weight of EDTA
def molecular_weight_EDTA : ℝ :=
  num_atoms_C * atomic_weight_C +
  num_atoms_H * atomic_weight_H +
  num_atoms_N * atomic_weight_N +
  num_atoms_O * atomic_weight_O

-- Proof that the molecular weight of the complex is 332.328 g/mol
theorem EDTA_Ca2_complex_weight : molecular_weight_EDTA + atomic_weight_Ca = 332.328 := by
  sorry

end EDTA_Ca2_complex_weight_l200_200628


namespace apps_difference_l200_200831

variable (initial_apps : ℕ) (added_apps : ℕ) (apps_left : ℕ)
variable (total_apps : ℕ := initial_apps + added_apps)
variable (deleted_apps : ℕ := total_apps - apps_left)
variable (difference : ℕ := added_apps - deleted_apps)

theorem apps_difference (h1 : initial_apps = 115) (h2 : added_apps = 235) (h3 : apps_left = 178) : 
  difference = 63 := by
  sorry

end apps_difference_l200_200831


namespace square_side_length_l200_200112

-- Define the conditions
def rectangle_width : ℝ := 4
def rectangle_length : ℝ := 4
def area_rectangle : ℝ := rectangle_width * rectangle_length
def area_square : ℝ := area_rectangle

-- Prove the side length of the square
theorem square_side_length :
  ∃ s : ℝ, s * s = area_square ∧ s = 4 := 
  by {
    -- Here you'd write the proof step, but it's omitted as per instructions
    sorry
  }

end square_side_length_l200_200112


namespace find_abc_sum_l200_200371

theorem find_abc_sum :
  ∃ (a b c : ℤ), 2 * a + 3 * b = 52 ∧ 3 * b + c = 41 ∧ b * c = 60 ∧ a + b + c = 25 :=
by
  use 8, 12, 5
  sorry

end find_abc_sum_l200_200371


namespace train_cross_signal_pole_time_l200_200281

theorem train_cross_signal_pole_time :
  ∀ (train_length platform_length platform_cross_time signal_cross_time : ℝ),
  train_length = 300 →
  platform_length = 300 →
  platform_cross_time = 36 →
  signal_cross_time = train_length / ((train_length + platform_length) / platform_cross_time) →
  signal_cross_time = 18 :=
by
  intros train_length platform_length platform_cross_time signal_cross_time h_train_length h_platform_length h_platform_cross_time h_signal_cross_time
  rw [h_train_length, h_platform_length, h_platform_cross_time] at h_signal_cross_time
  sorry

end train_cross_signal_pole_time_l200_200281


namespace find_a_iff_l200_200055

def non_deg_ellipse (k : ℝ) : Prop :=
  ∀ x y : ℝ, 9 * (x^2) + (y^2) - 36 * x + 8 * y = k → 
  (∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0))

theorem find_a_iff (k : ℝ) : non_deg_ellipse k ↔ k > -52 := by
  sorry

end find_a_iff_l200_200055


namespace num_ordered_pairs_l200_200963

theorem num_ordered_pairs : 
  {p : ℤ × ℤ // p.1 * p.2 ≥ 0 ∧ p.1^3 + p.2^3 + 99 * p.1 * p.2 = 33^3}.to_finset.card = 35 := 
sorry

end num_ordered_pairs_l200_200963


namespace central_angle_of_sector_l200_200914

theorem central_angle_of_sector :
  ∃ R α : ℝ, (2 * R + α * R = 4) ∧ (1 / 2 * R ^ 2 * α = 1) ∧ α = 2 :=
by
  sorry

end central_angle_of_sector_l200_200914


namespace capital_growth_rate_l200_200612

theorem capital_growth_rate
  (loan_amount : ℝ) (interest_rate : ℝ) (repayment_period : ℝ) (surplus : ℝ) (growth_rate : ℝ) :
  loan_amount = 2000000 ∧ interest_rate = 0.08 ∧ repayment_period = 2 ∧ surplus = 720000 ∧
  (loan_amount * (1 + growth_rate)^repayment_period = loan_amount * (1 + interest_rate) + surplus) →
  growth_rate = 0.2 :=
by
  sorry

end capital_growth_rate_l200_200612


namespace gcd_n_squared_plus_4_n_plus_3_l200_200645

theorem gcd_n_squared_plus_4_n_plus_3 (n : ℕ) (hn_gt_four : n > 4) : 
  (gcd (n^2 + 4) (n + 3)) = if n % 13 = 10 then 13 else 1 := 
sorry

end gcd_n_squared_plus_4_n_plus_3_l200_200645


namespace cone_altitude_to_radius_ratio_l200_200291

theorem cone_altitude_to_radius_ratio (r h : ℝ) (V_cone V_sphere : ℝ)
  (h1 : V_sphere = (4 / 3) * Real.pi * r^3)
  (h2 : V_cone = (1 / 3) * Real.pi * r^2 * h)
  (h3 : V_cone = (1 / 3) * V_sphere) :
  h / r = 4 / 3 :=
by
  sorry

end cone_altitude_to_radius_ratio_l200_200291


namespace meadow_income_is_960000_l200_200710

theorem meadow_income_is_960000 :
  let boxes := 30
  let packs_per_box := 40
  let diapers_per_pack := 160
  let price_per_diaper := 5
  (boxes * packs_per_box * diapers_per_pack * price_per_diaper) = 960000 := 
by
  sorry

end meadow_income_is_960000_l200_200710


namespace sum_squares_bound_l200_200518

theorem sum_squares_bound (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) : 
  ab + bc + cd + da ≤ 4 :=
sorry

end sum_squares_bound_l200_200518


namespace set_intersection_complement_l200_200705

def setA : Set ℝ := {-2, -1, 0, 1, 2}
def setB : Set ℝ := { x : ℝ | x^2 + 2*x < 0 }
def complementB : Set ℝ := { x : ℝ | x ≥ 0 ∨ x ≤ -2 }

theorem set_intersection_complement :
  setA ∩ complementB = {-2, 0, 1, 2} :=
by
  sorry

end set_intersection_complement_l200_200705


namespace fraction_of_students_with_buddy_l200_200495

variable (s n : ℕ)

theorem fraction_of_students_with_buddy (h : n = 4 * s / 3) : 
  (n / 4 + s / 3) / (n + s) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_buddy_l200_200495


namespace unique_prime_value_l200_200696

def T : ℤ := 2161

theorem unique_prime_value :
  ∃ p : ℕ, (∃ n : ℤ, n^4 - 898 * n^2 + T - 2160 = p) ∧ Prime p ∧ (∀ q, (∃ n : ℤ, n^4 - 898 * n^2 + T - 2160 = q) → q = p) :=
  sorry

end unique_prime_value_l200_200696


namespace average_girls_score_l200_200038

open Function

variable (C c D d : ℕ)
variable (avgCedarBoys avgCedarGirls avgCedarCombined avgDeltaBoys avgDeltaGirls avgDeltaCombined avgCombinedBoys : ℤ)

-- Conditions
def CedarBoys := avgCedarBoys = 85
def CedarGirls := avgCedarGirls = 80
def CedarCombined := avgCedarCombined = 83
def DeltaBoys := avgDeltaBoys = 76
def DeltaGirls := avgDeltaGirls = 95
def DeltaCombined := avgDeltaCombined = 87
def CombinedBoys := avgCombinedBoys = 73

-- Correct answer
def CombinedGirls (avgCombinedGirls : ℤ) := avgCombinedGirls = 86

-- Final statement
theorem average_girls_score (C c D d : ℕ)
    (avgCedarBoys avgCedarGirls avgCedarCombined avgDeltaBoys avgDeltaGirls avgDeltaCombined avgCombinedBoys : ℤ)
    (H1 : CedarBoys avgCedarBoys)
    (H2 : CedarGirls avgCedarGirls)
    (H3 : CedarCombined avgCedarCombined)
    (H4 : DeltaBoys avgDeltaBoys)
    (H5 : DeltaGirls avgDeltaGirls)
    (H6 : DeltaCombined avgDeltaCombined)
    (H7 : CombinedBoys avgCombinedBoys) :
    ∃ avgCombinedGirls, CombinedGirls avgCombinedGirls :=
sorry

end average_girls_score_l200_200038


namespace fraction_of_two_bedroom_l200_200354

theorem fraction_of_two_bedroom {x : ℝ} 
    (h1 : 0.17 + x = 0.5) : x = 0.33 :=
by
  sorry

end fraction_of_two_bedroom_l200_200354


namespace probability_complete_collection_l200_200744

def combination (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem probability_complete_collection : 
  combination 6 6 * combination 12 4 / combination 18 10 = 5 / 442 :=
by
  have h1 : combination 6 6 = 1 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h2 : combination 12 4 = 495 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  have h3 : combination 18 10 = 43758 := by
    unfold combination
    simp [Nat.factorial, Nat.div, Nat.mul]

  rw [h1, h2, h3]
  norm_num
  sorry

end probability_complete_collection_l200_200744


namespace cost_per_sq_meter_l200_200161

def tank_dimensions : ℝ × ℝ × ℝ := (25, 12, 6)
def total_plastering_cost : ℝ := 186
def total_plastering_area : ℝ :=
  let (length, width, height) := tank_dimensions
  let area_bottom := length * width
  let area_longer_walls := length * height * 2
  let area_shorter_walls := width * height * 2
  area_bottom + area_longer_walls + area_shorter_walls

theorem cost_per_sq_meter : total_plastering_cost / total_plastering_area = 0.25 := by
  sorry

end cost_per_sq_meter_l200_200161


namespace find_n_cosine_l200_200641

theorem find_n_cosine : ∃ (n : ℤ), -180 ≤ n ∧ n ≤ 180 ∧ (∃ m : ℤ, n = 25 + 360 * m ∨ n = -25 + 360 * m) :=
by
  sorry

end find_n_cosine_l200_200641


namespace lim_sup_eq_Union_lim_inf_l200_200180

open Set

theorem lim_sup_eq_Union_lim_inf
  (Ω : Type*)
  (A : ℕ → Set Ω) :
  (⋂ n, ⋃ k ≥ n, A k) = ⋃ (n_infty : ℕ → ℕ) (hn : StrictMono n_infty), ⋃ n, ⋂ k ≥ n, A (n_infty k) :=
by
  sorry

end lim_sup_eq_Union_lim_inf_l200_200180


namespace door_height_eight_l200_200923

theorem door_height_eight (x : ℝ) (h w : ℝ) (H1 : w = x - 4) (H2 : h = x - 2) (H3 : x^2 = (x - 4)^2 + (x - 2)^2) : h = 8 :=
by
  sorry

end door_height_eight_l200_200923


namespace original_perimeter_l200_200788

theorem original_perimeter (a b : ℝ) (h : a / 2 + b / 2 = 129 / 2) : 2 * (a + b) = 258 :=
by
  sorry

end original_perimeter_l200_200788


namespace angles_does_not_exist_l200_200656

theorem angles_does_not_exist (a1 a2 a3 : ℝ) 
  (h1 : a1 + a2 = 90) 
  (h2 : a2 + a3 = 180) 
  (h3 : a3 = 18) : False :=
by
  sorry

end angles_does_not_exist_l200_200656


namespace D_working_alone_completion_time_l200_200768

variable (A_rate D_rate : ℝ)
variable (A_job_hours D_job_hours : ℝ)

-- Conditions
def A_can_complete_in_15_hours : Prop := (A_job_hours = 15)
def A_and_D_together_complete_in_10_hours : Prop := (1/A_rate + 1/D_rate = 10)

-- Proof statement
theorem D_working_alone_completion_time
  (hA : A_job_hours = 15)
  (hAD : 1/A_rate + 1/D_rate = 10) :
  D_job_hours = 30 := sorry

end D_working_alone_completion_time_l200_200768


namespace even_goal_probability_approximation_l200_200803

noncomputable def poisson_even_goal_probability (λ : ℝ) : ℝ :=
  (e^(-λ) * Real.cosh λ)

theorem even_goal_probability_approximation :
  poisson_even_goal_probability 2.8 ≈ 0.502 :=
by
  sorry

end even_goal_probability_approximation_l200_200803


namespace max_combinations_for_n_20_l200_200024

def num_combinations (s n k : ℕ) : ℕ :=
if n = 0 then if s = 0 then 1 else 0
else if s < n then 0
else if k = 0 then 0
else num_combinations (s - k) (n - 1) (k - 1) + num_combinations s n (k - 1)

theorem max_combinations_for_n_20 : ∀ s k, s = 20 ∧ k = 9 → num_combinations s 4 k = 12 :=
by
  intros s k h
  cases h
  sorry

end max_combinations_for_n_20_l200_200024


namespace complete_collection_probability_l200_200736

namespace Stickers

open scoped Classical

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.Ico 1 (n + 1))

def combinations (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_complete_collection :
  ℚ := (combinations 6 6 * combinations 12 4) / combinations 18 10

theorem complete_collection_probability :
  probability_complete_collection = (5 / 442 : ℚ) := by
  sorry

end Stickers

end complete_collection_probability_l200_200736


namespace interest_rate_calculation_l200_200549

-- Define the problem conditions and proof statement in Lean
theorem interest_rate_calculation 
  (P : ℝ) (r : ℝ) (T : ℝ) (CI SI diff : ℝ) 
  (principal_condition : P = 6000.000000000128)
  (time_condition : T = 2)
  (diff_condition : diff = 15)
  (CI_formula : CI = P * (1 + r)^T - P)
  (SI_formula : SI = P * r * T)
  (difference_condition : CI - SI = diff) : 
  r = 0.05 := 
by 
  sorry

end interest_rate_calculation_l200_200549


namespace quadratic_non_real_roots_b_values_l200_200899

theorem quadratic_non_real_roots_b_values (b : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by sorry

end quadratic_non_real_roots_b_values_l200_200899


namespace train_length_is_correct_l200_200764

noncomputable def length_of_train (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_s

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 :=
by 
  -- Here, a proof would be provided, eventually using the definitions and conditions given
  sorry

end train_length_is_correct_l200_200764


namespace contrapositive_l200_200143

theorem contrapositive (a : ℝ) : (a > 0 → a > 1) → (a ≤ 1 → a ≤ 0) :=
by sorry

end contrapositive_l200_200143


namespace dog_has_fewer_lives_than_cat_l200_200775

noncomputable def cat_lives : ℕ := 9
noncomputable def mouse_lives : ℕ := 13
noncomputable def dog_lives : ℕ := mouse_lives - 7
noncomputable def dog_less_lives : ℕ := cat_lives - dog_lives

theorem dog_has_fewer_lives_than_cat : dog_less_lives = 3 := by
  sorry

end dog_has_fewer_lives_than_cat_l200_200775


namespace volume_in_barrel_l200_200145

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem volume_in_barrel (x : ℕ) (V : ℕ) (hx : V = 30) 
  (h1 : V = x / 2 + x / 3 + x / 4 + x / 5 + x / 6) 
  (h2 : is_divisible (87 * x) 60) : 
  V = 29 := 
sorry

end volume_in_barrel_l200_200145


namespace perfect_squares_with_property_l200_200182

open Nat

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k : ℕ, p.Prime ∧ k > 0 ∧ n = p^k

def satisfies_property (n : ℕ) : Prop :=
  ∀ a : ℕ, a ∣ n → a ≥ 15 → is_prime_power (a + 15)

theorem perfect_squares_with_property :
  {n | satisfies_property n ∧ ∃ k : ℕ, n = k^2} = {1, 4, 9, 16, 49, 64, 196} :=
by
  sorry

end perfect_squares_with_property_l200_200182


namespace attendants_both_tools_l200_200149

theorem attendants_both_tools (pencil_users pen_users only_one_type total_attendants both_types : ℕ)
  (h1 : pencil_users = 25) 
  (h2 : pen_users = 15) 
  (h3 : only_one_type = 20) 
  (h4 : total_attendants = only_one_type + both_types) 
  (h5 : total_attendants = pencil_users + pen_users - both_types) 
  : both_types = 10 :=
by
  -- Fill in the proof sub-steps here if needed
  sorry

end attendants_both_tools_l200_200149


namespace non_real_roots_interval_l200_200904

theorem non_real_roots_interval (b : ℝ) : (∀ x : ℂ, x^2 + (b : ℂ) * x + 16 = 0 → ¬ is_real x) ↔ b ∈ Ioo (-8 : ℝ) 8 :=
by
  sorry

end non_real_roots_interval_l200_200904


namespace sum_of_roots_eq_h_over_4_l200_200513

theorem sum_of_roots_eq_h_over_4 (x1 x2 h b : ℝ) (h_ne : x1 ≠ x2)
  (hx1 : 4 * x1 ^ 2 - h * x1 = b) (hx2 : 4 * x2 ^ 2 - h * x2 = b) : x1 + x2 = h / 4 :=
sorry

end sum_of_roots_eq_h_over_4_l200_200513


namespace generatrix_length_of_cone_l200_200866

theorem generatrix_length_of_cone (r l : ℝ) (h1 : r = real.sqrt 2) 
    (h2 : 2 * real.pi * r = real.pi * l) : l = 2 * real.sqrt 2 :=
by
  -- The proof steps would go here, but we don't need to provide them.
  sorry

end generatrix_length_of_cone_l200_200866


namespace maximize_garden_area_length_l200_200290

noncomputable def length_parallel_to_wall (cost_per_foot : ℝ) (fence_cost : ℝ) : ℝ :=
  let total_length := fence_cost / cost_per_foot 
  let y := total_length / 4 
  let length_parallel := total_length - 2 * y
  length_parallel

theorem maximize_garden_area_length :
  ∀ (cost_per_foot fence_cost : ℝ), cost_per_foot = 10 → fence_cost = 1500 → 
  length_parallel_to_wall cost_per_foot fence_cost = 75 :=
by
  intros
  simp [length_parallel_to_wall, *]
  sorry

end maximize_garden_area_length_l200_200290


namespace ab_bc_cd_da_le_four_l200_200523

theorem ab_bc_cd_da_le_four (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 :=
by
  sorry

end ab_bc_cd_da_le_four_l200_200523


namespace arithmetic_to_geometric_l200_200018

theorem arithmetic_to_geometric (a1 a2 a3 a4 d : ℝ)
  (h_arithmetic : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d)
  (h_d_nonzero : d ≠ 0):
  ((a2^2 = a1 * a3 ∨ a2^2 = a1 * a4 ∨ a3^2 = a1 * a4 ∨ a3^2 = a2 * a4) → (a1 / d = 1 ∨ a1 / d = -4)) :=
by {
  sorry
}

end arithmetic_to_geometric_l200_200018


namespace q_at_1_is_zero_l200_200818

-- Define the function q : ℝ → ℝ
-- The conditions imply q(1) = 0
axiom q : ℝ → ℝ

-- Given that (1, 0) is on the graph of y = q(x)
axiom q_condition : q 1 = 0

-- Prove q(1) = 0 given the condition that (1, 0) is on the graph
theorem q_at_1_is_zero : q 1 = 0 :=
by
  exact q_condition

end q_at_1_is_zero_l200_200818


namespace false_propositions_count_l200_200812

-- Definitions of the propositions
def proposition1 (A B : Prop) (P : Prop) : Prop :=
  P ∧ ¬ P

def proposition2 (A B : Prop) (P : Prop) : Prop :=
  P ∧ ¬ P

def proposition3 (A B : Prop) : Prop :=
  ¬ (A ∧ B)

def proposition4 (A B : Prop) : Prop :=
  A ∧ B

-- Theorem to prove the total number of false propositions
theorem false_propositions_count (A B : Prop) (P1 P2 P3 P4 : Prop) :
  ¬ (proposition1 A B P1) ∧ ¬ (proposition2 A B P2) ∧ ¬ (proposition3 A B) ∧ proposition4 A B → 3 = 3 :=
by
  intro h
  sorry

end false_propositions_count_l200_200812


namespace problem_integer_square_l200_200848

theorem problem_integer_square 
  (a b c d A : ℤ) 
  (H1 : a^2 + A = b^2) 
  (H2 : c^2 + A = d^2) : 
  ∃ (k : ℕ), 2 * (a + b) * (c + d) * (a * c + b * d - A) = k^2 :=
by
  sorry

end problem_integer_square_l200_200848


namespace right_triangle_condition_l200_200361

theorem right_triangle_condition (a b c : ℝ) : (a^2 = b^2 - c^2) → (∃ B : ℝ, B = 90) := 
sorry

end right_triangle_condition_l200_200361


namespace cos_sum_arithmetic_seq_l200_200849

theorem cos_sum_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + a 5 + a 9 = 5 * Real.pi) : 
  Real.cos (a 2 + a 8) = -1 / 2 :=
  sorry

end cos_sum_arithmetic_seq_l200_200849


namespace num_int_values_N_l200_200463

theorem num_int_values_N (N : ℕ) : 
  (∃ M, M ∣ 72 ∧ M > 3 ∧ N = M - 3) ↔ N ∈ ({1, 3, 5, 6, 9, 15, 21, 33, 69} : Finset ℕ) :=
by
  sorry

end num_int_values_N_l200_200463


namespace tiffany_total_bags_l200_200264

-- Define the initial and additional bags correctly
def bags_on_monday : ℕ := 10
def bags_next_day : ℕ := 3
def bags_day_after : ℕ := 7

-- Define the total bags calculation
def total_bags (initial : ℕ) (next : ℕ) (after : ℕ) : ℕ :=
  initial + next + after

-- Prove that the total bags collected is 20
theorem tiffany_total_bags : total_bags bags_on_monday bags_next_day bags_day_after = 20 :=
by
  sorry

end tiffany_total_bags_l200_200264


namespace intersection_of_sets_l200_200668

def set_M := { y : ℝ | y ≥ 0 }
def set_N := { y : ℝ | ∃ x : ℝ, y = -x^2 + 1 }

theorem intersection_of_sets : set_M ∩ set_N = { y : ℝ | 0 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_of_sets_l200_200668


namespace rectangle_area_l200_200557

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_area_l200_200557


namespace problem_proof_l200_200195

noncomputable def problem (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y ≥ 0 ∧ x ^ 2019 + y = 1) → (x + y ^ 2019 > 1 - 1 / 300)

theorem problem_proof (x y : ℝ) : problem x y :=
by
  intros h
  sorry

end problem_proof_l200_200195


namespace range_of_a_l200_200352

-- Define the main proof problem statement
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ∈ set.Ioo 0 1 → (x + real.log a) / real.exp x - (a * real.log x) / x > 0) : 
  a ∈ set.Ico (1 / real.exp 1) 1 :=
sorry

end range_of_a_l200_200352


namespace soccer_team_starters_l200_200946

theorem soccer_team_starters (players : Finset ℕ) (quadruplets : Finset ℕ) :
  players.card = 16 ∧ quadruplets.card = 4 ∧ quadruplets ⊆ players →
  let starters := Finset.filter (λ p, p ∈ quadruplets) players in
  starters.card = 2 → 
  let non_quadruplets := players \ quadruplets in
  ∃ chosen_starters : Finset ℕ,
    chosen_starters.card = 6 ∧
    starters ⊆ chosen_starters ∧
    ∃ combination_count : ℕ,
      combination_count = ((quadruplets.card.choose 2) * (non_quadruplets.card.choose 4)) ∧
      combination_count = 2970 := 
sorry

end soccer_team_starters_l200_200946


namespace binomial_sum_real_part_l200_200439

theorem binomial_sum_real_part :
  (1 / (2:ℝ)^(2010)) * ∑ n in Finset.range (1005 + 1), (-3:ℝ)^n * Nat.choose 2010 (2 * n) = -1 / 2 := 
sorry

end binomial_sum_real_part_l200_200439


namespace age_of_father_now_l200_200031

variable (M F : ℕ)

theorem age_of_father_now :
  (M = 2 * F / 5) ∧ (M + 14 = (F + 14) / 2) → F = 70 :=
by 
sorry

end age_of_father_now_l200_200031


namespace car_speed_ratio_l200_200618

-- Assuming the bridge length as L, pedestrian's speed as v_p, and car's speed as v_c.
variables (L v_p v_c : ℝ)

-- Mathematically equivalent proof problem statement in Lean 4.
theorem car_speed_ratio (h1 : 2/5 * L = 2/5 * L)
                       (h2 : (L - 2/5 * L) / v_p = L / v_c) :
    v_c = 5 * v_p := 
  sorry

end car_speed_ratio_l200_200618


namespace measureable_weights_count_l200_200132

theorem measureable_weights_count (a b c : ℕ) (ha : a = 1) (hb : b = 3) (hc : c = 9) :
  ∃ s : Finset ℕ, s.card = 13 ∧ ∀ x ∈ s, x ≥ 1 ∧ x ≤ 13 := 
sorry

end measureable_weights_count_l200_200132


namespace length_generatrix_cone_l200_200871

theorem length_generatrix_cone (base_radius : ℝ) (lateral_surface_unfolded : Prop)
    (h_base_radius : base_radius = real.sqrt 2)
    (h_lateral_surface_unfolded : lateral_surface_unfolded) :
    let l := 2 * real.sqrt 2 in
    2 * π * base_radius = π * l :=
by {
  -- assumptions incorporated,
  -- the proof goes here
  sorry
}

end length_generatrix_cone_l200_200871


namespace intersection_A_B_l200_200093

def A : Set ℝ := {x | x < 3 * x - 1}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_A_B : (A ∩ B) = {x | x > 1 / 2 ∧ x < 3} :=
by sorry

end intersection_A_B_l200_200093


namespace leak_empty_time_l200_200284

variable (inlet_rate : ℕ := 6) -- litres per minute
variable (total_capacity : ℕ := 12960) -- litres
variable (empty_time_with_inlet_open : ℕ := 12) -- hours

def inlet_rate_per_hour := inlet_rate * 60 -- litres per hour
def net_emptying_rate := total_capacity / empty_time_with_inlet_open -- litres per hour
def leak_rate := net_emptying_rate + inlet_rate_per_hour -- litres per hour

theorem leak_empty_time : total_capacity / leak_rate = 9 := by
  sorry

end leak_empty_time_l200_200284


namespace right_triangle_properties_l200_200219

theorem right_triangle_properties (a b c h : ℝ)
  (ha: a = 5) (hb: b = 12) (h_right_angle: a^2 + b^2 = c^2)
  (h_area: (1/2) * a * b = (1/2) * c * h) :
  c = 13 ∧ h = 60 / 13 :=
by
  sorry

end right_triangle_properties_l200_200219


namespace cone_generatrix_length_l200_200856

-- Define the conditions of the problem
def base_radius : ℝ := sqrt 2
def cone_base_circumference : ℝ := 2 * Real.pi * base_radius
def semicircle_arc_length (generatrix : ℝ) : ℝ := Real.pi * generatrix

-- The proof statement
theorem cone_generatrix_length : 
  ∃ (l : ℝ), 2 * Real.pi * sqrt 2 = Real.pi * l ∧ l = 2 * sqrt 2 :=
sorry

end cone_generatrix_length_l200_200856


namespace cone_generatrix_length_is_2sqrt2_l200_200862

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_is_2sqrt2_l200_200862


namespace complete_collection_probability_l200_200737

namespace Stickers

open scoped Classical

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.Ico 1 (n + 1))

def combinations (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_complete_collection :
  ℚ := (combinations 6 6 * combinations 12 4) / combinations 18 10

theorem complete_collection_probability :
  probability_complete_collection = (5 / 442 : ℚ) := by
  sorry

end Stickers

end complete_collection_probability_l200_200737


namespace even_numbers_set_l200_200837

-- Define the set of all even numbers in set-builder notation
def even_set : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}

-- Theorem stating that this set is the set of all even numbers
theorem even_numbers_set :
  ∀ x : ℤ, (x ∈ even_set ↔ ∃ n : ℤ, x = 2 * n) := by
  sorry

end even_numbers_set_l200_200837


namespace evaluate_expression_at_two_l200_200272

theorem evaluate_expression_at_two : (2 * (2:ℝ)^2 - 3 * 2 + 4) = 6 := by
  sorry

end evaluate_expression_at_two_l200_200272


namespace generatrix_length_of_cone_l200_200859

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end generatrix_length_of_cone_l200_200859


namespace total_area_of_L_shaped_figure_l200_200679

-- Define the specific lengths for each segment
def bottom_rect_length : ℕ := 10
def bottom_rect_width : ℕ := 6
def central_rect_length : ℕ := 4
def central_rect_width : ℕ := 4
def top_rect_length : ℕ := 5
def top_rect_width : ℕ := 1

-- Calculate the area of each rectangle
def bottom_rect_area : ℕ := bottom_rect_length * bottom_rect_width
def central_rect_area : ℕ := central_rect_length * central_rect_width
def top_rect_area : ℕ := top_rect_length * top_rect_width

-- Given the length and width of the rectangles, calculate the total area of the L-shaped figure
theorem total_area_of_L_shaped_figure : 
  bottom_rect_area + central_rect_area + top_rect_area = 81 := by
  sorry

end total_area_of_L_shaped_figure_l200_200679


namespace volume_of_rectangular_box_l200_200984

theorem volume_of_rectangular_box 
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12) : 
  l * w * h = 60 :=
sorry

end volume_of_rectangular_box_l200_200984


namespace value_of_expression_l200_200205

theorem value_of_expression (x y z : ℝ) (hz : z ≠ 0) 
    (h1 : 2 * x - 3 * y - z = 0) 
    (h2 : x + 3 * y - 14 * z = 0) : 
    (x^2 + 3 * x * y) / (y^2 + z^2) = 7 := 
by 
  sorry

end value_of_expression_l200_200205


namespace area_proof_l200_200564

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end area_proof_l200_200564


namespace door_height_is_eight_l200_200928

/-- Statement of the problem: given a door with specified dimensions as conditions,
prove that the height of the door is 8 feet. -/
theorem door_height_is_eight (x : ℝ) (h₁ : x^2 = (x - 4)^2 + (x - 2)^2) : (x - 2) = 8 :=
by
  sorry

end door_height_is_eight_l200_200928


namespace max_remaining_numbers_l200_200234

theorem max_remaining_numbers : 
  ∃ (S ⊆ {n | 1 ≤ n ∧ n ≤ 235}), (∀ x ∈ S, ∀ y ∈ S, x ≠ y → ¬(x ∣ (y - x))) ∧ card S = 118 :=
by
  sorry

end max_remaining_numbers_l200_200234


namespace min_value_quadratic_expr_l200_200643

theorem min_value_quadratic_expr (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : a > 0) 
  (h2 : x₁ ≠ x₂) 
  (h3 : x₁^2 - 4*a*x₁ + 3*a^2 < 0) 
  (h4 : x₂^2 - 4*a*x₂ + 3*a^2 < 0)
  (h5 : x₁ + x₂ = 4*a)
  (h6 : x₁ * x₂ = 3*a^2) : 
  x₁ + x₂ + a / (x₁ * x₂) = 4 * a + 1 / (3 * a) := 
sorry

end min_value_quadratic_expr_l200_200643


namespace transformation_invariant_l200_200407

-- Define the initial and transformed parabolas
def initial_parabola (x : ℝ) : ℝ := 2 * x^2
def transformed_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 3

-- Define the transformation process
def move_right_1 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - 1)
def move_up_3 (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 3

-- Concatenate transformations to form the final transformation
def combined_transformation (x : ℝ) : ℝ :=
  move_up_3 (move_right_1 initial_parabola) x

-- Statement to prove
theorem transformation_invariant :
  ∀ x : ℝ, combined_transformation x = transformed_parabola x := 
by {
  sorry
}

end transformation_invariant_l200_200407


namespace positive_difference_largest_prime_factors_l200_200592

theorem positive_difference_largest_prime_factors :
  let p1 := 139
  let p2 := 29
  p1 - p2 = 110 := sorry

end positive_difference_largest_prime_factors_l200_200592


namespace derivative_of_y_l200_200185

-- Define the function y
def y (x : ℝ) : ℝ := - (1 / (3 * (Real.sin x)^3)) - (1 / (Real.sin x)) + (1 / 2) * Real.log ((1 + Real.sin x) / (1 - Real.sin x))

-- Statement to prove the derivative of y
theorem derivative_of_y (x : ℝ) : deriv y x = 1 / (Real.cos x * (Real.sin x)^4) := by
  sorry

end derivative_of_y_l200_200185


namespace quadratic_polynomial_solution_l200_200330

theorem quadratic_polynomial_solution :
  ∃ a b c : ℚ, 
    (∀ x : ℚ, ax*x + bx + c = 8 ↔ x = -2) ∧ 
    (∀ x : ℚ, ax*x + bx + c = 2 ↔ x = 1) ∧ 
    (∀ x : ℚ, ax*x + bx + c = 10 ↔ x = 3) ∧ 
    a = 6 / 5 ∧ 
    b = -4 / 5 ∧ 
    c = 8 / 5 :=
by {
  sorry
}

end quadratic_polynomial_solution_l200_200330


namespace sum_of_final_numbers_l200_200730

theorem sum_of_final_numbers (x y : ℝ) (S : ℝ) (h : x + y = S) : 
  3 * (x + 4) + 3 * (y + 4) = 3 * S + 24 := by
  sorry

end sum_of_final_numbers_l200_200730


namespace figure_total_area_l200_200724

theorem figure_total_area (a : ℝ) (h : a^2 - (3/2 * a^2) = 0.6) : 
  5 * a^2 = 6 :=
by
  sorry

end figure_total_area_l200_200724


namespace boat_speed_in_still_water_l200_200005

def speed_of_stream : ℝ := 8
def downstream_distance : ℝ := 64
def upstream_distance : ℝ := 32

theorem boat_speed_in_still_water (x : ℝ) (t : ℝ) 
  (HS_downstream : t = downstream_distance / (x + speed_of_stream)) 
  (HS_upstream : t = upstream_distance / (x - speed_of_stream)) :
  x = 24 := by
  sorry

end boat_speed_in_still_water_l200_200005


namespace distance_between_intersections_l200_200256

-- Define the parabola and circle equations
def parabola (y : ℝ) : ℝ := (y^2 / 12)
def circle (x y : ℝ) : ℝ := (x^2 + y^2 - 4*x - 6*y)

-- Define the proof problem
theorem distance_between_intersections :
  ∃ P Q : ℝ × ℝ, P ≠ Q ∧ 
              P.2^2 = 12 * P.1 ∧ 
              circle P.1 P.2 = 0 ∧
              Q.2^2 = 12 * Q.1 ∧ 
              circle Q.1 Q.2 = 0 ∧
              dist P Q = 3 * real.sqrt 13 :=
sorry

end distance_between_intersections_l200_200256


namespace cupcakes_for_children_l200_200049

-- Definitions for the conditions
def packs15 : Nat := 4
def packs10 : Nat := 4
def cupcakes_per_pack15 : Nat := 15
def cupcakes_per_pack10 : Nat := 10

-- Proposition to prove the total number of cupcakes is 100
theorem cupcakes_for_children :
  (packs15 * cupcakes_per_pack15) + (packs10 * cupcakes_per_pack10) = 100 := by
  sorry

end cupcakes_for_children_l200_200049


namespace quadratic_inequality_l200_200192

theorem quadratic_inequality (a : ℝ) :
  (¬ (∃ x : ℝ, a * x^2 + 2 * x + 3 ≤ 0)) ↔ (a > 1 / 3) :=
by 
  sorry

end quadratic_inequality_l200_200192


namespace time_period_for_investment_l200_200778

variable (P R₁₅ R₁₀ I₁₅ I₁₀ : ℝ)
variable (T : ℝ)

noncomputable def principal := 8400
noncomputable def rate15 := 15
noncomputable def rate10 := 10
noncomputable def interestDifference := 840

theorem time_period_for_investment :
  ∀ (T : ℝ),
    P = principal →
    R₁₅ = rate15 →
    R₁₀ = rate10 →
    I₁₅ = P * (R₁₅ / 100) * T →
    I₁₀ = P * (R₁₀ / 100) * T →
    (I₁₅ - I₁₀) = interestDifference →
    T = 2 :=
  sorry

end time_period_for_investment_l200_200778


namespace groceries_spent_l200_200615

/-- Defining parameters from the conditions provided -/
def rent : ℝ := 5000
def milk : ℝ := 1500
def education : ℝ := 2500
def petrol : ℝ := 2000
def miscellaneous : ℝ := 700
def savings_rate : ℝ := 0.10
def savings : ℝ := 1800

/-- Adding an assertion for the total spent on groceries -/
def groceries : ℝ := 4500

theorem groceries_spent (total_salary total_expenses : ℝ) :
  total_salary = savings / savings_rate →
  total_expenses = rent + milk + education + petrol + miscellaneous →
  groceries = total_salary - (total_expenses + savings) :=
by
  intros h_salary h_expenses
  sorry

end groceries_spent_l200_200615


namespace count_cube_roots_less_than_15_l200_200888

theorem count_cube_roots_less_than_15 :
  {n : ℕ | n > 0 ∧ real.cbrt n < 15}.to_finset.card = 3374 :=
by sorry

end count_cube_roots_less_than_15_l200_200888


namespace even_function_expression_l200_200342

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x then 2*x + 1 else -2*x + 1

theorem even_function_expression (x : ℝ) (hx : x < 0) :
  f x = -2*x + 1 :=
by sorry

end even_function_expression_l200_200342


namespace rate_mangoes_correct_l200_200130

-- Define the conditions
def weight_apples : ℕ := 8
def rate_apples : ℕ := 70
def cost_apples := weight_apples * rate_apples

def total_payment : ℕ := 1145
def weight_mangoes : ℕ := 9
def cost_mangoes := total_payment - cost_apples

-- Define the rate per kg of mangoes
def rate_mangoes := cost_mangoes / weight_mangoes

-- Prove the rate per kg for mangoes
theorem rate_mangoes_correct : rate_mangoes = 65 := by
  -- all conditions and intermediate calculations already stated
  sorry

end rate_mangoes_correct_l200_200130


namespace correct_probability_l200_200698

noncomputable def T : ℕ := 44
noncomputable def num_books : ℕ := T - 35
noncomputable def n : ℕ := 9
noncomputable def favorable_outcomes : ℕ := (Nat.choose n 6) * 2
noncomputable def total_arrangements : ℕ := (Nat.factorial n)
noncomputable def probability : Rat := (favorable_outcomes : ℚ) / (total_arrangements : ℚ)
noncomputable def m : ℕ := 1
noncomputable def p : Nat := Nat.gcd 168 362880
noncomputable def final_prob_form : Rat := 1 / 2160
noncomputable def answer : ℕ := m + 2160

theorem correct_probability : 
  probability = final_prob_form ∧ answer = 2161 := 
by
  sorry

end correct_probability_l200_200698


namespace pictures_at_dolphin_show_l200_200416

def taken_before : Int := 28
def total_pictures_taken : Int := 44

theorem pictures_at_dolphin_show : total_pictures_taken - taken_before = 16 := by
  -- solution proof goes here
  sorry

end pictures_at_dolphin_show_l200_200416


namespace even_goal_probability_approximation_l200_200804

noncomputable def poisson_even_goal_probability (λ : ℝ) : ℝ :=
  (e^(-λ) * Real.cosh λ)

theorem even_goal_probability_approximation :
  poisson_even_goal_probability 2.8 ≈ 0.502 :=
by
  sorry

end even_goal_probability_approximation_l200_200804


namespace quintuplets_babies_l200_200681

theorem quintuplets_babies (t r q : ℕ) (h1 : r = 6 * q)
  (h2 : t = 2 * r)
  (h3 : 2 * t + 3 * r + 5 * q = 1500) :
  5 * q = 160 :=
by
  sorry

end quintuplets_babies_l200_200681


namespace solve_r_minus_s_l200_200937

noncomputable def r := 20
noncomputable def s := 4

theorem solve_r_minus_s
  (h1 : r^2 - 24 * r + 80 = 0)
  (h2 : s^2 - 24 * s + 80 = 0)
  (h3 : r > s) : r - s = 16 :=
by
  sorry

end solve_r_minus_s_l200_200937


namespace count_odd_integers_in_range_l200_200459

theorem count_odd_integers_in_range : (set_of (λ n : ℤ, 25 < n^2 ∧ n^2 < 144 ∧ odd n)).card = 6 :=
by
  sorry

end count_odd_integers_in_range_l200_200459


namespace xy_yz_zx_nonzero_l200_200091

theorem xy_yz_zx_nonzero (x y z : ℝ)
  (h1 : 1 / |x^2 + 2 * y * z| + 1 / |y^2 + 2 * z * x| > 1 / |z^2 + 2 * x * y|)
  (h2 : 1 / |y^2 + 2 * z * x| + 1 / |z^2 + 2 * x * y| > 1 / |x^2 + 2 * y * z|)
  (h3 : 1 / |z^2 + 2 * x * y| + 1 / |x^2 + 2 * y * z| > 1 / |y^2 + 2 * z * x|) :
  x * y + y * z + z * x ≠ 0 := by
  sorry

end xy_yz_zx_nonzero_l200_200091


namespace cone_generatrix_length_is_2sqrt2_l200_200863

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_is_2sqrt2_l200_200863


namespace knight_tour_impossible_49_squares_l200_200365

-- Define the size of the chessboard
def boardSize : ℕ := 7

-- Define the total number of squares on the chessboard
def totalSquares : ℕ := boardSize * boardSize

-- Define the condition for a knight's tour on the 49-square board
def knight_tour_possible (n : ℕ) : Prop :=
  n = totalSquares ∧ 
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 
  -- add condition representing knight's tour and ending
  -- adjacent condition can be mathematically proved here 
  -- but we'll skip here as we asked just to state the problem not the proof.
  sorry -- Placeholder for the precise condition

-- Define the final theorem statement
theorem knight_tour_impossible_49_squares : ¬ knight_tour_possible totalSquares :=
by sorry

end knight_tour_impossible_49_squares_l200_200365


namespace rate_percent_per_annum_l200_200277

theorem rate_percent_per_annum (P : ℝ) (SI_increase : ℝ) (T_increase : ℝ) (R : ℝ) 
  (hP : P = 2000) (hSI_increase : SI_increase = 40) (hT_increase : T_increase = 4) 
  (h : SI_increase = P * R * T_increase / 100) : R = 0.5 :=
by  
  sorry

end rate_percent_per_annum_l200_200277


namespace paint_price_and_max_boxes_l200_200539

theorem paint_price_and_max_boxes (x y m : ℕ) 
  (h1 : x + 2 * y = 56) 
  (h2 : 2 * x + y = 64) 
  (h3 : 24 * m + 16 * (200 - m) ≤ 3920) : 
  x = 24 ∧ y = 16 ∧ m ≤ 90 := 
  by 
    sorry

end paint_price_and_max_boxes_l200_200539


namespace arithmetic_seq_slope_l200_200918

theorem arithmetic_seq_slope {a : ℕ → ℤ} (h : a 2 - a 4 = 2) : ∃ a1 : ℤ, ∀ n : ℕ, a n = -n + (a 1) + 1 := 
by {
  sorry
}

end arithmetic_seq_slope_l200_200918


namespace remainder_h_x_10_div_h_x_l200_200510

noncomputable def h (x : ℤ) : ℤ := x^5 - x^4 + x^3 - x^2 + x - 1

theorem remainder_h_x_10_div_h_x (x : ℤ) : polynomial.div_mod_by_monic (h (x)) (h (x)) (h (x^{10})) = (x, 5) :=
by
  -- Proof omitted.
  sorry

end remainder_h_x_10_div_h_x_l200_200510


namespace costForFirstKgs_l200_200435

noncomputable def applePrice (l : ℝ) (q : ℝ) (x : ℝ) (totalWeight : ℝ) : ℝ :=
  if totalWeight <= x then l * totalWeight else l * x + q * (totalWeight - x)

theorem costForFirstKgs (l q x : ℝ) :
  l = 10 ∧ q = 11 ∧ (applePrice l q x 33 = 333) ∧ (applePrice l q x 36 = 366) ∧ (applePrice l q 15 15 = 150) → x = 30 := 
by
  sorry

end costForFirstKgs_l200_200435


namespace one_bag_covers_250_sqfeet_l200_200324

noncomputable def lawn_length : ℝ := 22
noncomputable def lawn_width : ℝ := 36
noncomputable def bags_count : ℝ := 4
noncomputable def extra_area : ℝ := 208

noncomputable def lawn_area : ℝ := lawn_length * lawn_width
noncomputable def total_covered_area : ℝ := lawn_area + extra_area
noncomputable def one_bag_area : ℝ := total_covered_area / bags_count

theorem one_bag_covers_250_sqfeet :
  one_bag_area = 250 := 
by
  sorry

end one_bag_covers_250_sqfeet_l200_200324


namespace limo_gas_price_l200_200486

theorem limo_gas_price
  (hourly_wage : ℕ := 15)
  (ride_payment : ℕ := 5)
  (review_bonus : ℕ := 20)
  (hours_worked : ℕ := 8)
  (rides_given : ℕ := 3)
  (gallons_gas : ℕ := 17)
  (good_reviews : ℕ := 2)
  (total_owed : ℕ := 226) :
  total_owed = (hours_worked * hourly_wage) + (rides_given * ride_payment) + (good_reviews * review_bonus) + (gallons_gas * 3) :=
by
  sorry

end limo_gas_price_l200_200486


namespace mixture_weight_l200_200115

theorem mixture_weight (C : ℚ) (W : ℚ)
  (H1: C > 0) -- C represents the cost per pound of milk powder and coffee in June, and is a positive number
  (H2: C * 0.2 = 0.2) -- The price per pound of milk powder in July
  (H3: (W / 2) * 0.2 + (W / 2) * 4 * C = 6.30) -- The cost of the mixture in July

  : W = 3 := 
sorry

end mixture_weight_l200_200115


namespace perpendicular_vectors_x_value_l200_200075

-- Define the vectors a and b
def a : ℝ × ℝ := (3, -1)
def b (x : ℝ) : ℝ × ℝ := (1, x)

-- Define the dot product function for vectors in ℝ^2
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

-- The mathematical statement to prove
theorem perpendicular_vectors_x_value (x : ℝ) (h : dot_product a (b x) = 0) : x = 3 :=
by
  sorry

end perpendicular_vectors_x_value_l200_200075


namespace expression_evaluation_l200_200892

-- Define the property to be proved:
theorem expression_evaluation (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by sorry

end expression_evaluation_l200_200892


namespace sqrt_of_product_of_powers_l200_200630

theorem sqrt_of_product_of_powers :
  (Real.sqrt (4^2 * 5^6) = 500) :=
by
  sorry

end sqrt_of_product_of_powers_l200_200630


namespace sixteen_grams_on_left_pan_l200_200945

theorem sixteen_grams_on_left_pan :
  ∃ (weights : ℕ → ℕ) (pans : ℕ → ℕ) (n : ℕ),
    weights n = 16 ∧
    pans 0 = 11111 ∧
    ∃ k, (∀ i < k, weights i = 2 ^ i) ∧
    (∀ i < k, (pans 1 + weights i = 38) ∧ (pans 0 + 11111 = weights i + skeletal)) ∧
    k = 6 := by
  sorry

end sixteen_grams_on_left_pan_l200_200945


namespace correct_quotient_is_48_l200_200994

theorem correct_quotient_is_48 (dividend : ℕ) (incorrect_divisor : ℕ) (incorrect_quotient : ℕ) (correct_divisor : ℕ) (correct_quotient : ℕ) :
  incorrect_divisor = 72 → 
  incorrect_quotient = 24 → 
  correct_divisor = 36 →
  dividend = incorrect_divisor * incorrect_quotient →
  correct_quotient = dividend / correct_divisor →
  correct_quotient = 48 :=
by
  sorry

end correct_quotient_is_48_l200_200994


namespace product_of_roots_cubic_eq_l200_200043

theorem product_of_roots_cubic_eq (α : Type _) [Field α] :
  (∃ (r1 r2 r3 : α), (r1 * r2 * r3 = 6) ∧ (r1 + r2 + r3 = 6) ∧ (r1 * r2 + r1 * r3 + r2 * r3 = 11)) :=
by
  sorry

end product_of_roots_cubic_eq_l200_200043


namespace max_remaining_numbers_l200_200232

def numbers (n : ℕ) : set ℕ := {k | 1 ≤ k ∧ k ≤ n}

def valid_subset (s : set ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → (a - b) ≠ 0 → ¬ (a - b) ∣ c

theorem max_remaining_numbers : ∃ s : set ℕ, s ⊆ numbers 235 ∧ valid_subset s ∧ card s = 118 := 
sorry

end max_remaining_numbers_l200_200232


namespace volume_of_rect_prism_l200_200033

variables {a b c V : ℝ}

theorem volume_of_rect_prism :
  (∃ (a b c : ℝ), (a * b = Real.sqrt 2) ∧ (b * c = Real.sqrt 3) ∧ (a * c = Real.sqrt 6) ∧ V = a * b * c) →
  V = Real.sqrt 6 :=
by
  sorry

end volume_of_rect_prism_l200_200033


namespace heads_count_l200_200152

theorem heads_count (H T : ℕ) (h1 : H + T = 128) (h2 : H = T + 12) : H = 70 := by
  sorry

end heads_count_l200_200152


namespace moles_of_C2H6_are_1_l200_200776

def moles_of_C2H6_reacted (n_C2H6: ℕ) (n_Cl2: ℕ) (n_C2Cl6: ℕ): Prop :=
  n_Cl2 = 6 ∧ n_C2Cl6 = 1 ∧ (n_C2H6 + 6 * (n_Cl2 - 1) = n_C2Cl6 + 6 * (n_Cl2 - 1))

theorem moles_of_C2H6_are_1:
  ∀ (n_C2H6 n_Cl2 n_C2Cl6: ℕ), moles_of_C2H6_reacted n_C2H6 n_Cl2 n_C2Cl6 → n_C2H6 = 1 :=
by
  intros n_C2H6 n_Cl2 n_C2Cl6 h
  sorry

end moles_of_C2H6_are_1_l200_200776


namespace yogurt_cost_l200_200243

-- Define the conditions given in the problem
def total_cost_ice_cream : ℕ := 20 * 6
def spent_difference : ℕ := 118

theorem yogurt_cost (y : ℕ) 
  (h1 : total_cost_ice_cream = 2 * y + spent_difference) : 
  y = 1 :=
  sorry

end yogurt_cost_l200_200243


namespace probability_of_picking_grain_buds_l200_200761

theorem probability_of_picking_grain_buds :
  let num_stamps := 3
  let num_grain_buds := 1
  let probability := num_grain_buds / num_stamps
  probability = 1 / 3 :=
by
  sorry

end probability_of_picking_grain_buds_l200_200761


namespace cube_volume_l200_200257

/-- Given the perimeter of one face of a cube, proving the volume of the cube -/

theorem cube_volume (h : ∀ (s : ℝ), 4 * s = 28) : (∃ (v : ℝ), v = (7 : ℝ) ^ 3) :=
by
  sorry

end cube_volume_l200_200257


namespace always_composite_l200_200442

theorem always_composite (p : ℕ) (hp : Nat.Prime p) : ¬Nat.Prime (p^2 + 35) ∧ ¬Nat.Prime (p^2 + 55) :=
by
  sorry

end always_composite_l200_200442


namespace sum_of_coefficients_condition_l200_200703

theorem sum_of_coefficients_condition 
  (t : ℕ → ℤ) 
  (d e f : ℤ) 
  (h0 : t 0 = 3) 
  (h1 : t 1 = 7) 
  (h2 : t 2 = 17) 
  (h3 : t 3 = 86)
  (rec_relation : ∀ k ≥ 2, t (k + 1) = d * t k + e * t (k - 1) + f * t (k - 2)) : 
  d + e + f = 14 :=
by
  sorry

end sum_of_coefficients_condition_l200_200703


namespace add_and_simplify_fractions_l200_200244

theorem add_and_simplify_fractions :
  (1 / 462) + (23 / 42) = 127 / 231 :=
by
  sorry

end add_and_simplify_fractions_l200_200244


namespace bhanu_income_problem_l200_200298

-- Define the total income
def total_income (I : ℝ) : Prop :=
  let petrol_spent := 300
  let house_rent := 70
  (0.10 * (I - petrol_spent) = house_rent)

-- Define the percentage of income spent on petrol
def petrol_percentage (P : ℝ) (I : ℝ) : Prop :=
  0.01 * P * I = 300

-- The theorem we aim to prove
theorem bhanu_income_problem : 
  ∃ I P, total_income I ∧ petrol_percentage P I ∧ P = 30 :=
by
  sorry

end bhanu_income_problem_l200_200298


namespace find_x_l200_200147

theorem find_x (x : ℝ) : 
  (1 + x) * 0.20 = x * 0.4 → x = 1 :=
by
  intros h
  sorry

end find_x_l200_200147


namespace kanul_spent_on_raw_materials_eq_500_l200_200506

variable (total_amount : ℕ)
variable (machinery_cost : ℕ)
variable (cash_percentage : ℕ)

def amount_spent_on_raw_materials (total_amount machinery_cost cash_percentage : ℕ) : ℕ :=
  total_amount - machinery_cost - (total_amount * cash_percentage / 100)

theorem kanul_spent_on_raw_materials_eq_500 :
  total_amount = 1000 →
  machinery_cost = 400 →
  cash_percentage = 10 →
  amount_spent_on_raw_materials total_amount machinery_cost cash_percentage = 500 :=
by
  intros
  sorry

end kanul_spent_on_raw_materials_eq_500_l200_200506


namespace pentagon_area_calc_l200_200819

noncomputable def pentagon_area : ℝ :=
  let triangle1 := (1 / 2) * 18 * 22
  let triangle2 := (1 / 2) * 30 * 26
  let trapezoid := (1 / 2) * (22 + 30) * 10
  triangle1 + triangle2 + trapezoid

theorem pentagon_area_calc :
  pentagon_area = 848 := by
  sorry

end pentagon_area_calc_l200_200819


namespace calculate_fraction_square_mul_l200_200437

theorem calculate_fraction_square_mul :
  ((8 / 9) ^ 2) * ((1 / 3) ^ 2) = 64 / 729 :=
by
  sorry

end calculate_fraction_square_mul_l200_200437


namespace function_periodicity_l200_200913

theorem function_periodicity (f : ℝ → ℝ) (h1 : ∀ x, f (-x) + f x = 0)
  (h2 : ∀ x, f (x + 1) = f (1 - x)) (h3 : f 1 = 5) : f 2015 = -5 :=
sorry

end function_periodicity_l200_200913


namespace football_match_even_goals_l200_200793

noncomputable def poisson_even_probability (λ : ℝ) : ℝ :=
  (1 + Real.exp (-2 * λ)) / 2

theorem football_match_even_goals :
  poisson_even_probability 2.8 ≈ 0.502 :=
by
  -- Proof skipped
  sorry

end football_match_even_goals_l200_200793


namespace system_no_solution_l200_200057

theorem system_no_solution (n : ℝ) :
  ∃ x y z : ℝ, (n * x + y = 1) ∧ (1 / 2 * n * y + z = 1) ∧ (x + 1 / 2 * n * z = 2) ↔ n = -1 := 
sorry

end system_no_solution_l200_200057


namespace ratio_quadrilateral_l200_200391

theorem ratio_quadrilateral
  (ABCD_area : ℝ)
  (h_ABCD : ABCD_area = 40)
  (K L M N : Type)
  (AK KB : ℝ)
  (h_ratio : AK / KB = BL / LC ∧ BL / LC = CM / MD ∧ CM / MD = DN / NA)
  (KLMN_area : ℝ)
  (h_KLMN : KLMN_area = 25) :
  (AK / (AK + KB) = 1 / 4 ∨ AK / (AK + KB) = 3 / 4) :=
sorry

end ratio_quadrilateral_l200_200391


namespace bicycle_route_total_length_l200_200221

theorem bicycle_route_total_length :
  let horizontal_length := 13 
  let vertical_length := 13 
  2 * horizontal_length + 2 * vertical_length = 52 :=
by
  let horizontal_length := 13
  let vertical_length := 13
  sorry

end bicycle_route_total_length_l200_200221


namespace incorrect_statement_l200_200164

noncomputable def first_line_of_defense := "Skin and mucous membranes"
noncomputable def second_line_of_defense := "Antimicrobial substances and phagocytic cells in body fluids"
noncomputable def third_line_of_defense := "Immune organs and immune cells"
noncomputable def non_specific_immunity := "First and second line of defense"
noncomputable def specific_immunity := "Third line of defense"
noncomputable def d_statement := "The defensive actions performed by the three lines of defense in the human body are called non-specific immunity"

theorem incorrect_statement : d_statement ≠ specific_immunity ∧ d_statement ≠ non_specific_immunity := by
  sorry

end incorrect_statement_l200_200164


namespace each_child_plays_40_minutes_l200_200916

variable (TotalMinutes : ℕ)
variable (NumChildren : ℕ)
variable (ChildPairs : ℕ)

theorem each_child_plays_40_minutes (h1 : TotalMinutes = 120) 
                                    (h2 : NumChildren = 6) 
                                    (h3 : ChildPairs = 2) :
  (ChildPairs * TotalMinutes) / NumChildren = 40 :=
by
  sorry

end each_child_plays_40_minutes_l200_200916


namespace football_even_goal_probability_l200_200798

noncomputable def poisson_even_goal_probability : ℝ :=
  let λ := 2.8 in
  (1 + Real.exp (-2 * λ)) / 2

theorem football_even_goal_probability :
  let λ := 2.8 in
  let N := Poisson λ in  -- Define N as a Poisson random variable with parameter λ
  (∑ k : ℕ in (range (2*k)).filter (λ k, N.P k > 0), N.P k) = 
  poisson_even_goal_probability :=
by
  sorry

end football_even_goal_probability_l200_200798


namespace equilateral_triangle_square_ratio_l200_200624

theorem equilateral_triangle_square_ratio (t s : ℕ) (h_t : 3 * t = 12) (h_s : 4 * s = 12) :
  t / s = 4 / 3 := by
  sorry

end equilateral_triangle_square_ratio_l200_200624


namespace no_positive_reals_satisfy_conditions_l200_200058

theorem no_positive_reals_satisfy_conditions :
  ¬ ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  4 * (a * b + b * c + c * a) - 1 ≥ a^2 + b^2 + c^2 ∧ 
  a^2 + b^2 + c^2 ≥ 3 * (a^3 + b^3 + c^3) :=
by
  sorry

end no_positive_reals_satisfy_conditions_l200_200058


namespace solve_for_k_l200_200056

theorem solve_for_k :
  ∀ (k : ℝ), (∃ x : ℝ, (3*x + 8)*(x - 6) = -50 + k*x) ↔
    k = -10 + 2*Real.sqrt 6 ∨ k = -10 - 2*Real.sqrt 6 := by
  sorry

end solve_for_k_l200_200056


namespace polynomial_roots_r_l200_200544

theorem polynomial_roots_r :
  ∀ (α β γ : ℝ),
    (polynomial.root_set (polynomial.C (-14) + polynomial.C 5 * polynomial.X 
                          + polynomial.C 4 * polynomial.X^2 + polynomial.X^3) ℝ = 
     {α, β, γ}) →
    ∃ p q r : ℝ, 
      (polynomial.root_set (polynomial.C r + polynomial.C q * polynomial.X 
                            + polynomial.C p * polynomial.X^2 + polynomial.X^3) ℝ = 
       {α + β, β + γ, γ + α}) ∧ 
      r = 34 :=
begin
  sorry
end

end polynomial_roots_r_l200_200544


namespace problem_l200_200851

def f (x : ℚ) : ℚ :=
  x⁻¹ - (x⁻¹ / (1 - x⁻¹))

theorem problem : f (f (-3)) = 6 / 5 :=
by
  sorry

end problem_l200_200851


namespace average_after_17th_inning_l200_200599

-- Define the conditions.
variable (A : ℚ) -- The initial average after 16 innings

-- Define the score in the 17th inning and the increment in the average.
def runs_in_17th_inning : ℚ := 87
def increment_in_average : ℚ := 3

-- Define the equation derived from the conditions.
theorem average_after_17th_inning :
  (16 * A + runs_in_17th_inning) / 17 = A + increment_in_average →
  A + increment_in_average = 39 :=
sorry

end average_after_17th_inning_l200_200599


namespace fourth_root_is_four_l200_200263

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^4 - 8 * x^3 - 7 * x^2 + 9 * x + 11

-- Conditions that must be true for the given problem
@[simp] def f_neg1_zero : f (-1) = 0 := by sorry
@[simp] def f_2_zero : f (2) = 0 := by sorry
@[simp] def f_neg3_zero : f (-3) = 0 := by sorry

-- The theorem stating the fourth root
theorem fourth_root_is_four (root4 : ℝ) (H : f root4 = 0) : root4 = 4 := by sorry

end fourth_root_is_four_l200_200263


namespace scientific_notation_4947_66_billion_l200_200810

theorem scientific_notation_4947_66_billion :
  4947.66 * 10^8 = 4.94766 * 10^11 :=
sorry

end scientific_notation_4947_66_billion_l200_200810


namespace circles_internally_tangent_l200_200258

theorem circles_internally_tangent :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 6 * x + 4 * y + 12 = 0) ∧ (x^2 + y^2 - 14 * x - 2 * y + 14 = 0) →
  ∃ (C1 C2 : ℝ × ℝ) (r1 r2 : ℝ),
  C1 = (3, -2) ∧ r1 = 1 ∧
  C2 = (7, 1) ∧ r2 = 6 ∧
  dist C1 C2 = r2 - r1 :=
by
  sorry

end circles_internally_tangent_l200_200258


namespace find_x_value_l200_200405

theorem find_x_value (x : ℝ) 
  (h₁ : 1 / (x + 8) + 2 / (3 * x) + 1 / (2 * x) = 1 / (2 * x)) :
  x = (-1 + Real.sqrt 97) / 6 :=
sorry

end find_x_value_l200_200405


namespace calories_per_orange_is_correct_l200_200504

noncomputable def calories_per_orange
  (oranges pieces_per_orange num_people calories_per_person : ℕ)
  (h_oranges : oranges = 5)
  (h_pieces_per_orange : pieces_per_orange = 8)
  (h_num_people : num_people = 4)
  (h_calories_per_person : calories_per_person = 100) : ℕ :=
by
  -- Definitions derived from conditions
  let total_pieces := oranges * pieces_per_orange
  let pieces_per_person := total_pieces / num_people
  let total_calories := calories_per_person
  have calories_per_piece := total_calories / pieces_per_person

  -- Conclusion
  have calories_per_orange := pieces_per_orange * calories_per_piece
  exact calories_per_orange

theorem calories_per_orange_is_correct
  (oranges pieces_per_orange num_people calories_per_person : ℕ)
  (h_oranges : oranges = 5)
  (h_pieces_per_orange : pieces_per_orange = 8)
  (h_num_people : num_people = 4)
  (h_calories_per_person : calories_per_person = 100) :
  calories_per_orange oranges pieces_per_orange num_people calories_per_person
    h_oranges h_pieces_per_orange h_num_people h_calories_per_person = 100 :=
by
  simp [calories_per_orange]
  sorry  -- Proof omitted

end calories_per_orange_is_correct_l200_200504


namespace sum_of_solutions_eq_zero_l200_200458

theorem sum_of_solutions_eq_zero :
  ∀ x : ℝ, (-π ≤ x ∧ x ≤ 3 * π ∧ (1 / Real.sin x + 1 / Real.cos x = 4))
  → x = 0 := sorry

end sum_of_solutions_eq_zero_l200_200458


namespace marble_total_weight_l200_200531

theorem marble_total_weight :
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 + 0.21666666666666667 + 0.4583333333333333 + 0.12777777777777778 = 1.5527777777777777 :=
by
  sorry

end marble_total_weight_l200_200531


namespace find_smaller_number_l200_200968

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 18) (h2 : a * b = 45) : a = 3 ∨ b = 3 := 
by 
  -- Proof would go here
  sorry

end find_smaller_number_l200_200968


namespace A_50_correct_l200_200695

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := 
  ![![3, 2], 
    ![-8, -5]]

-- The theorem to prove
theorem A_50_correct : A^50 = ![![(-199 : ℤ), -100], 
                                 ![400, 201]] := 
by
  sorry

end A_50_correct_l200_200695


namespace range_of_a_l200_200333

def A : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }

theorem range_of_a (a : ℝ) (h : a ∈ A) : -1 ≤ a ∧ a ≤ 3 := 
sorry

end range_of_a_l200_200333


namespace trigonometric_identity_l200_200315

theorem trigonometric_identity :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  t30 = 1 / Real.sqrt 3 →
  s30 = 1 / 2 →
  (t30 ^ 2 - s30 ^ 2) / (t30 ^ 2 * s30 ^ 2) = 1 :=
by
  intros t30_eq s30_eq
  -- Proof would go here
  sorry

end trigonometric_identity_l200_200315


namespace num_ways_to_choose_officers_same_gender_l200_200530

-- Definitions based on conditions
def num_members : Nat := 24
def num_boys : Nat := 12
def num_girls : Nat := 12
def num_officers : Nat := 3

-- Theorem statement using these definitions
theorem num_ways_to_choose_officers_same_gender :
  (num_boys * (num_boys-1) * (num_boys-2) * 2) = 2640 :=
by
  sorry

end num_ways_to_choose_officers_same_gender_l200_200530


namespace geo_seq_sum_l200_200545

theorem geo_seq_sum (a : ℤ) (S : ℕ → ℤ) :
  (∀ n : ℕ, S n = 2^n + a) →
  a = -1 :=
sorry

end geo_seq_sum_l200_200545


namespace count_three_digit_values_with_double_sum_eq_six_l200_200701

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_three_digit (x : ℕ) : Prop := 
  100 ≤ x ∧ x < 1000

theorem count_three_digit_values_with_double_sum_eq_six :
  ∃ count : ℕ, is_three_digit count ∧ (
    (∀ x, is_three_digit x → sum_of_digits (sum_of_digits x) = 6) ↔ count = 30
  ) :=
sorry

end count_three_digit_values_with_double_sum_eq_six_l200_200701


namespace pqrs_sum_l200_200065

theorem pqrs_sum (p q r s : ℤ)
  (h1 : r + p = -1)
  (h2 : s + p * r + q = 3)
  (h3 : p * s + q * r = -4)
  (h4 : q * s = 4) :
  p + q + r + s = -1 :=
sorry

end pqrs_sum_l200_200065


namespace jordan_rectangle_length_l200_200302

def rectangle_area (length width : ℝ) : ℝ := length * width

theorem jordan_rectangle_length :
  let carol_length := 8
  let carol_width := 15
  let jordan_width := 30
  let carol_area := rectangle_area carol_length carol_width
  ∃ jordan_length, rectangle_area jordan_length jordan_width = carol_area →
  jordan_length = 4 :=
by
  sorry

end jordan_rectangle_length_l200_200302


namespace generatrix_length_of_cone_l200_200860

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end generatrix_length_of_cone_l200_200860


namespace expression_simplifies_to_49_l200_200894

theorem expression_simplifies_to_49 (x : ℝ) : 
  (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end expression_simplifies_to_49_l200_200894


namespace students_in_class_l200_200528

theorem students_in_class (y : ℕ) (H : 2 * y^2 + 6 * y + 9 = 490) : 
  y + (y + 3) = 31 := by
  sorry

end students_in_class_l200_200528


namespace rectangle_area_l200_200562

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l200_200562


namespace volume_rectangular_box_l200_200982

variables {l w h : ℝ}

theorem volume_rectangular_box (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end volume_rectangular_box_l200_200982


namespace car_speed_ratio_l200_200619

-- Assuming the bridge length as L, pedestrian's speed as v_p, and car's speed as v_c.
variables (L v_p v_c : ℝ)

-- Mathematically equivalent proof problem statement in Lean 4.
theorem car_speed_ratio (h1 : 2/5 * L = 2/5 * L)
                       (h2 : (L - 2/5 * L) / v_p = L / v_c) :
    v_c = 5 * v_p := 
  sorry

end car_speed_ratio_l200_200619


namespace simplify_expression_1_simplify_expression_2_l200_200247

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) : 4 * (a + b) + 2 * (a + b) - (a + b) = 5 * a + 5 * b :=
  sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) : (3 * m / 2) - (5 * m / 2 - 1) + 3 * (4 - m) = -4 * m + 13 :=
  sorry

end simplify_expression_1_simplify_expression_2_l200_200247


namespace cone_generatrix_length_is_2sqrt2_l200_200864

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_is_2sqrt2_l200_200864


namespace total_amount_shared_l200_200039

theorem total_amount_shared (A B C : ℕ) (h1 : A = 24) (h2 : 2 * A = 3 * B) (h3 : 8 * A = 4 * C) :
  A + B + C = 156 :=
sorry

end total_amount_shared_l200_200039


namespace Melissa_commission_l200_200943

theorem Melissa_commission 
  (coupe_price : ℝ)
  (suv_multiplier : ℝ)
  (commission_rate : ℝ) :
  (coupe_price = 30000) →
  (suv_multiplier = 2) →
  (commission_rate = 0.02) →
  let suv_price := suv_multiplier * coupe_price in
  let total_sales := coupe_price + suv_price in
  let commission := commission_rate * total_sales in
  commission = 1800 :=
begin
  sorry
end

end Melissa_commission_l200_200943


namespace area_of_triangle_ABC_l200_200061

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1424233, 2848467)
def C : point := (1424234, 2848469)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_ABC : triangle_area A B C = 0.50 := by
  sorry

end area_of_triangle_ABC_l200_200061


namespace no_pos_int_squares_l200_200172

open Nat

theorem no_pos_int_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬(∃ k m : ℕ, k ^ 2 = a ^ 2 + b ∧ m ^ 2 = b ^ 2 + a) :=
sorry

end no_pos_int_squares_l200_200172


namespace abs_inequality_solution_rational_inequality_solution_l200_200543

theorem abs_inequality_solution (x : ℝ) : (|x - 2| + |2 * x - 3| < 4) ↔ (1 / 3 < x ∧ x < 3) :=
sorry

theorem rational_inequality_solution (x : ℝ) : 
  (x^2 - 3 * x) / (x^2 - x - 2) ≤ x ↔ (x ∈ Set.Icc (-1) 0 ∪ {1} ∪ Set.Ioi 2) := 
sorry

#check abs_inequality_solution
#check rational_inequality_solution

end abs_inequality_solution_rational_inequality_solution_l200_200543


namespace incorrect_statement_D_l200_200659

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + x
else -(x^2 + x)

theorem incorrect_statement_D : ¬(∀ x : ℝ, x ≤ 0 → f x = x^2 + x) :=
by
  sorry

end incorrect_statement_D_l200_200659


namespace max_sum_disjoint_subsets_l200_200938

open Finset

theorem max_sum_disjoint_subsets :
  ∃ (M : Finset ℕ), M ⊆ range 1 26 ∧
  (∀ (A B : Finset ℕ), A ⊆ M → B ⊆ M → A ∩ B = ∅ → A.sum (+) ≠ B.sum (+)) ∧
  M.sum (+) = 123 := sorry

end max_sum_disjoint_subsets_l200_200938


namespace remaining_amount_to_be_paid_l200_200675

theorem remaining_amount_to_be_paid (p : ℝ) (deposit : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) (final_payment : ℝ) :
  deposit = 80 ∧ tax_rate = 0.07 ∧ discount_rate = 0.05 ∧ deposit = 0.1 * p ∧ 
  final_payment = (p - (discount_rate * p)) * (1 + tax_rate) - deposit → 
  final_payment = 733.20 :=
by
  sorry

end remaining_amount_to_be_paid_l200_200675


namespace four_digit_square_l200_200784

/-- A four-digit square number that satisfies the given conditions -/
theorem four_digit_square (a b c d : ℕ) (h₁ : b + c = a) (h₂ : a + c = 10 * d) :
  1000 * a + 100 * b + 10 * c + d = 6241 :=
sorry

end four_digit_square_l200_200784


namespace store_incur_loss_of_one_percent_l200_200159

theorem store_incur_loss_of_one_percent
    (a b x : ℝ)
    (h1 : x = a * 1.1)
    (h2 : x = b * 0.9)
    : (2 * x - (a + b)) / (a + b) = -0.01 :=
by
  -- Proof goes here
  sorry

end store_incur_loss_of_one_percent_l200_200159


namespace find_s_l200_200327

theorem find_s (s : Real) (h : ⌊s⌋ + s = 15.4) : s = 7.4 :=
sorry

end find_s_l200_200327


namespace transportation_inverse_proportion_l200_200546

theorem transportation_inverse_proportion (V t : ℝ) (h: V * t = 10^5) : V = 10^5 / t :=
by
  sorry

end transportation_inverse_proportion_l200_200546


namespace james_vegetable_consumption_l200_200690

theorem james_vegetable_consumption : 
  (1/4 + 1/4) * 2 * 7 + 3 = 10 := 
by
  sorry

end james_vegetable_consumption_l200_200690


namespace sqrt_domain_l200_200499

def inequality_holds (x : ℝ) : Prop := x + 5 ≥ 0

theorem sqrt_domain (x : ℝ) : inequality_holds x ↔ x ≥ -5 := by
  sorry

end sqrt_domain_l200_200499


namespace vertical_angles_congruent_l200_200108

namespace VerticalAngles

-- Define what it means for two angles to be vertical
def Vertical (α β : Real) : Prop :=
  -- Definition of vertical angles goes here
  sorry

-- Hypothesis: If two angles are vertical angles
theorem vertical_angles_congruent (α β : Real) (h : Vertical α β) : α = β :=
sorry

end VerticalAngles

end vertical_angles_congruent_l200_200108


namespace CarlosAndDianaReceivedAs_l200_200811

variables (Alan Beth Carlos Diana : Prop)
variable (num_A : ℕ)

-- Condition 1: Alan => Beth
axiom AlanImpliesBeth : Alan → Beth

-- Condition 2: Beth => Carlos
axiom BethImpliesCarlos : Beth → Carlos

-- Condition 3: Carlos => Diana
axiom CarlosImpliesDiana : Carlos → Diana

-- Condition 4: Only two students received an A
axiom OnlyTwoReceivedAs : num_A = 2

-- Theorem: Carlos and Diana received A's
theorem CarlosAndDianaReceivedAs : ((Alan ∧ Beth ∧ Carlos ∧ Diana → False) ∧
                                   (Beth ∧ Carlos ∧ Diana → False) ∧
                                   (Alan ∧ Beth ∧ Diana → False) ∧
                                   (Alan ∧ Beth ∧ Carlos → False) ∧
                                   (Alan ∧ Diana → False) ∧
                                   (Beth ∧ Carlos → False) ∧
                                   (Alan ∧ Carlos → False) ∧
                                   (Beth ∧ Diana → False)) → (Carlos ∧ Diana) :=
by
  intros h
  have h1 := AlanImpliesBeth
  have h2 := BethImpliesCarlos
  have h3 := CarlosImpliesDiana
  have h4 := OnlyTwoReceivedAs
  sorry

end CarlosAndDianaReceivedAs_l200_200811


namespace cans_collected_is_232_l200_200723

-- Definitions of the conditions
def total_students : ℕ := 30
def half_students : ℕ := total_students / 2
def cans_per_half_student : ℕ := 12
def remaining_students : ℕ := 13
def cans_per_remaining_student : ℕ := 4

-- Calculate total cans collected
def total_cans_collected : ℕ := (half_students * cans_per_half_student) + (remaining_students * cans_per_remaining_student)

-- The theorem to be proved
theorem cans_collected_is_232 : total_cans_collected = 232 := by
  -- Proof would go here
  sorry

end cans_collected_is_232_l200_200723


namespace rectangle_area_l200_200555

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_area_l200_200555


namespace rectangle_area_l200_200559

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l200_200559


namespace cryptarithm_problem_l200_200678

theorem cryptarithm_problem (F E D : ℤ) (h1 : F - E = D - 1) (h2 : D + E + F = 16) (h3 : F - E = D) : 
    F - E = 5 :=
by sorry

end cryptarithm_problem_l200_200678


namespace minimum_value_f_inequality_proof_l200_200345

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 1)

-- The minimal value of f(x)
def m : ℝ := 4

theorem minimum_value_f :
  (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, -3 ≤ x ∧ x ≤ 1 ∧ f x = m) :=
by
  sorry -- Proof that the minimum value of f(x) is 4 and occurs in the range -3 ≤ x ≤ 1

variables (p q r : ℝ)

-- Given condition that p^2 + 2q^2 + r^2 = 4
theorem inequality_proof (h : p^2 + 2 * q^2 + r^2 = m) : q * (p + r) ≤ 2 :=
by
  sorry -- Proof that q(p + r) ≤ 2 given p^2 + 2q^2 + r^2 = 4

end minimum_value_f_inequality_proof_l200_200345


namespace quadratic_solution_range_l200_200447

theorem quadratic_solution_range :
  ∃ x : ℝ, x^2 + 12 * x - 15 = 0 ∧ 1.1 < x ∧ x < 1.2 :=
sorry

end quadratic_solution_range_l200_200447


namespace marble_problem_l200_200816

theorem marble_problem (a : ℚ) (total : ℚ) 
  (h1 : total = a + 2 * a + 6 * a + 42 * a) :
  a = 42 / 17 :=
by 
  sorry

end marble_problem_l200_200816


namespace ronald_laundry_frequency_l200_200241

variable (Tim_laundry_frequency Ronald_laundry_frequency : ℕ)

theorem ronald_laundry_frequency :
  (Tim_laundry_frequency = 9) →
  (18 % Ronald_laundry_frequency = 0) →
  (18 % Tim_laundry_frequency = 0) →
  (Ronald_laundry_frequency ≠ 1) →
  (Ronald_laundry_frequency ≠ 18) →
  (Ronald_laundry_frequency ≠ 9) →
  (Ronald_laundry_frequency = 3) :=
by
  intros hTim hRonaldMultiple hTimMultiple hNot1 hNot18 hNot9
  sorry

end ronald_laundry_frequency_l200_200241


namespace find_digit_A_l200_200002

-- Define the six-digit number for any digit A
def six_digit_number (A : ℕ) : ℕ := 103200 + A * 10 + 4
-- Define the condition that a number is prime
def is_prime (n : ℕ) : Prop := (2 ≤ n) ∧ ∀ m : ℕ, 2 ≤ m → m * m ≤ n → ¬ (m ∣ n)

-- The main theorem stating that A must equal 1 for the number to be prime
theorem find_digit_A (A : ℕ) : A = 1 ↔ is_prime (six_digit_number A) :=
by
  sorry -- Proof to be filled in


end find_digit_A_l200_200002


namespace real_possible_b_values_quadratic_non_real_roots_l200_200906

theorem real_possible_b_values_quadratic_non_real_roots :
  {b : ℝ | ∃ (a c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4*a*c < 0)} = {b : ℝ | -8 < b ∧ b < 8} :=
by 
  sorry

end real_possible_b_values_quadratic_non_real_roots_l200_200906


namespace blue_pill_cost_l200_200626

theorem blue_pill_cost :
  ∃ (y : ℝ), (∀ (d : ℝ), d = 45) ∧
  (∀ (b : ℝ) (r : ℝ), b = y ∧ r = y - 2) ∧
  ((21 : ℝ) * 45 = 945) ∧
  (b + r = 45) ∧
  y = 23.5 := 
by
  sorry

end blue_pill_cost_l200_200626


namespace dealership_sedan_sales_l200_200603

-- Definitions based on conditions:
def sports_cars_ratio : ℕ := 3
def sedans_ratio : ℕ := 5
def anticipated_sports_cars : ℕ := 36

-- Proof problem statement
theorem dealership_sedan_sales :
    (anticipated_sports_cars * sedans_ratio) / sports_cars_ratio = 60 :=
by
  -- Proof goes here
  sorry

end dealership_sedan_sales_l200_200603


namespace vines_painted_l200_200325

-- Definitions based on the conditions in the problem statement
def time_per_lily : ℕ := 5
def time_per_rose : ℕ := 7
def time_per_orchid : ℕ := 3
def time_per_vine : ℕ := 2
def total_time_spent : ℕ := 213
def lilies_painted : ℕ := 17
def roses_painted : ℕ := 10
def orchids_painted : ℕ := 6

-- The theorem to prove the number of vines painted
theorem vines_painted (vines_painted : ℕ) : 
  213 = (17 * 5) + (10 * 7) + (6 * 3) + (vines_painted * 2) → 
  vines_painted = 20 :=
by
  intros h
  sorry

end vines_painted_l200_200325


namespace frank_money_l200_200469

theorem frank_money (X : ℝ) (h1 : (3/4) * (4/5) * X = 360) : X = 600 :=
sorry

end frank_money_l200_200469


namespace sqrt_of_product_of_powers_l200_200629

theorem sqrt_of_product_of_powers :
  (Real.sqrt (4^2 * 5^6) = 500) :=
by
  sorry

end sqrt_of_product_of_powers_l200_200629


namespace no_odd_prime_pn_plus_1_eq_2m_no_odd_prime_pn_minus_1_eq_2m_l200_200712

open Nat

theorem no_odd_prime_pn_plus_1_eq_2m (n p m : ℕ)
  (hn : n > 1) (hp : p.Prime) (hp_odd : Odd p) (hm : m > 0) :
  p^n + 1 ≠ 2^m := by
  sorry

theorem no_odd_prime_pn_minus_1_eq_2m (n p m : ℕ)
  (hn : n > 2) (hp : p.Prime) (hp_odd : Odd p) (hm : m > 0) :
  p^n - 1 ≠ 2^m := by
  sorry

end no_odd_prime_pn_plus_1_eq_2m_no_odd_prime_pn_minus_1_eq_2m_l200_200712


namespace find_f_10_l200_200069

-- Defining the function f as an odd, periodic function with period 2
def odd_func_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = - f x) ∧ (∀ x : ℝ, f (x + 2) = f x)

-- Stating the theorem that f(10) is 0 given the conditions
theorem find_f_10 (f : ℝ → ℝ) (h1 : odd_func_periodic f) : f 10 = 0 :=
sorry

end find_f_10_l200_200069


namespace max_value_of_6_f_x_plus_2012_l200_200646

noncomputable def f (x : ℝ) : ℝ :=
  min (min (4*x + 1) (x + 2)) (-2*x + 4)

theorem max_value_of_6_f_x_plus_2012 : ∃ x : ℝ, 6 * f x + 2012 = 2028 :=
sorry

end max_value_of_6_f_x_plus_2012_l200_200646


namespace base_length_of_parallelogram_l200_200719

theorem base_length_of_parallelogram 
  (area : ℝ) (base altitude : ℝ) 
  (h_area : area = 242)
  (h_altitude : altitude = 2 * base) :
  base = 11 :=
by
  sorry

end base_length_of_parallelogram_l200_200719


namespace price_per_litre_of_second_oil_l200_200348

-- Define the conditions given in the problem
def oil1_volume : ℝ := 10 -- 10 litres of first oil
def oil1_rate : ℝ := 50 -- Rs. 50 per litre

def oil2_volume : ℝ := 5 -- 5 litres of the second oil
def total_mixed_volume : ℝ := oil1_volume + oil2_volume -- Total volume of mixed oil

def mixed_rate : ℝ := 55.33 -- Rs. 55.33 per litre for the mixed oil

-- Define the target value to prove: price per litre of the second oil
def price_of_second_oil : ℝ := 65.99

-- Prove the statement
theorem price_per_litre_of_second_oil : 
  (oil1_volume * oil1_rate + oil2_volume * price_of_second_oil) = total_mixed_volume * mixed_rate :=
by 
  sorry -- actual proof to be provided

end price_per_litre_of_second_oil_l200_200348


namespace imaginaria_city_population_l200_200399

theorem imaginaria_city_population (a b c : ℕ) (h₁ : a^2 + 225 = b^2 + 1) (h₂ : b^2 + 1 + 75 = c^2) : 5 ∣ a^2 :=
by
  sorry

end imaginaria_city_population_l200_200399


namespace dog_older_than_max_by_18_l200_200461

-- Definition of the conditions
def human_to_dog_years_ratio : ℕ := 7
def max_age : ℕ := 3
def dog_age_in_human_years : ℕ := 3

-- Translate the question: How much older, in dog years, will Max's dog be?
def age_difference_in_dog_years : ℕ :=
  dog_age_in_human_years * human_to_dog_years_ratio - max_age

-- The proof statement
theorem dog_older_than_max_by_18 : age_difference_in_dog_years = 18 := by
  sorry

end dog_older_than_max_by_18_l200_200461


namespace probability_even_goals_is_approximately_l200_200792

noncomputable def probability_even_goals (λ : ℝ) : ℝ :=
  let p : ℝ := ∑ k in (nat.filter even), (λ ^ k * real.exp (-λ)) / (nat.fact k)
  in p

def λ : ℝ := 2.8

theorem probability_even_goals_is_approximately:
  probability_even_goals λ ≈ 0.502 :=
sorry

end probability_even_goals_is_approximately_l200_200792


namespace amber_josh_departure_time_l200_200163

def latest_departure_time (flight_time : ℕ) (check_in_time : ℕ) (drive_time : ℕ) (parking_time : ℕ) :=
  flight_time - check_in_time - drive_time - parking_time

theorem amber_josh_departure_time :
  latest_departure_time 20 2 (45 / 60) (15 / 60) = 17 :=
by
  -- Placeholder for actual proof
  sorry

end amber_josh_departure_time_l200_200163


namespace remainder_when_divided_by_x_minus_2_l200_200644

-- Define the polynomial
def f (x : ℕ) : ℕ := x^3 - x^2 + 4 * x - 1

-- Statement of the problem: Prove f(2) = 11 using the Remainder Theorem
theorem remainder_when_divided_by_x_minus_2 : f 2 = 11 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l200_200644


namespace original_wage_before_increase_l200_200148

theorem original_wage_before_increase (new_wage : ℝ) (increase_rate : ℝ) (original_wage : ℝ) (h : new_wage = original_wage + increase_rate * original_wage) : 
  new_wage = 42 → increase_rate = 0.50 → original_wage = 28 :=
by
  intros h_new_wage h_increase_rate
  have h1 : new_wage = 42 := h_new_wage
  have h2 : increase_rate = 0.50 := h_increase_rate
  have h3 : new_wage = original_wage + increase_rate * original_wage := h
  sorry

end original_wage_before_increase_l200_200148


namespace notebooks_multiple_of_3_l200_200027

theorem notebooks_multiple_of_3 (N : ℕ) (h1 : ∃ k : ℕ, N = 3 * k) :
  ∃ k : ℕ, N = 3 * k :=
by
  sorry

end notebooks_multiple_of_3_l200_200027


namespace factor_polynomial_l200_200443

noncomputable def polynomial (x y n : ℤ) : ℤ := x^2 + 4 * x * y + 2 * x + n * y - n

theorem factor_polynomial (n : ℤ) :
  (∃ A B C D E F : ℤ, polynomial A B C = (A * x + B * y + C) * (D * x + E * y + F)) ↔ n = 0 :=
sorry

end factor_polynomial_l200_200443


namespace fat_rings_per_group_l200_200299

theorem fat_rings_per_group (F : ℕ)
  (h1 : ∀ F, (70 * (F + 4)) = (40 * (F + 4)) + 180)
  : F = 2 :=
sorry

end fat_rings_per_group_l200_200299


namespace cost_prices_max_units_B_possible_scenarios_l200_200155

-- Part 1: Prove cost prices of Product A and B
theorem cost_prices (x : ℝ) (A B : ℝ) 
  (h₁ : B = x ∧ A = x - 2) 
  (h₂ : 80 / A = 100 / B) 
  : B = 10 ∧ A = 8 :=
by 
  sorry

-- Part 2: Prove maximum units of product B that can be purchased
theorem max_units_B (y : ℕ) 
  (h₁ : ∀ y : ℕ, 3 * y - 5 + y ≤ 95) 
  : y ≤ 25 :=
by 
  sorry

-- Part 3: Prove possible scenarios for purchasing products A and B
theorem possible_scenarios (y : ℕ) 
  (h₁ : y > 23 * 9/17 ∧ y ≤ 25) 
  : y = 24 ∨ y = 25 :=
by 
  sorry

end cost_prices_max_units_B_possible_scenarios_l200_200155


namespace tangent_line_eq_f_positive_find_a_l200_200881

noncomputable def f (x a : ℝ) : ℝ := 1 - (a * x^2) / (Real.exp x)
noncomputable def f' (x a : ℝ) : ℝ := (a * x * (x - 2)) / (Real.exp x)

-- Part 1: equation of tangent line
theorem tangent_line_eq (a : ℝ) (h1 : f' 1 a = 1) (hx : f 1 a = 2) : ∀ x, f 1 a + f' 1 a * (x - 1) = x + 1 :=
sorry

-- Part 2: f(x) > 0 for x > 0 when a = 1
theorem f_positive (x : ℝ) (h : x > 0) : f x 1 > 0 :=
sorry

-- Part 3: minimum value of f(x) is -3, find a
theorem find_a (a : ℝ) (h : ∀ x, f x a ≥ -3) : a = Real.exp 2 :=
sorry

end tangent_line_eq_f_positive_find_a_l200_200881


namespace find_y_l200_200079

theorem find_y (x y : ℝ) (h1 : x^2 = 2 * y - 6) (h2 : x = 7) : y = 55 / 2 :=
by
  sorry

end find_y_l200_200079


namespace expression_simplifies_to_49_l200_200896

theorem expression_simplifies_to_49 (x : ℝ) : 
  (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end expression_simplifies_to_49_l200_200896


namespace largest_nonrepresentable_integer_l200_200017

theorem largest_nonrepresentable_integer :
  (∀ a b : ℕ, 8 * a + 15 * b ≠ 97) ∧ (∀ n : ℕ, n > 97 → ∃ a b : ℕ, n = 8 * a + 15 * b) :=
sorry

end largest_nonrepresentable_integer_l200_200017


namespace max_areas_divided_by_disk_l200_200023

-- Define the problem and the required conditions for the disk division
variable (n : ℕ)
axiom n_pos : n > 0

-- Assume the disk is divided by 2n equally spaced radii, one secant line, and one chord that does not intersect the secant
def max_non_overlapping_areas (n : ℕ) : ℕ := 4 * n - 1

-- The theorem stating that the maximum number of non-overlapping areas is 4n - 1
theorem max_areas_divided_by_disk :
  max_non_overlapping_areas n = 4 * n - 1 :=
sorry

end max_areas_divided_by_disk_l200_200023


namespace sqrt_expression_l200_200632

theorem sqrt_expression : Real.sqrt ((4^2) * (5^6)) = 500 := by
  sorry

end sqrt_expression_l200_200632


namespace eq_abs_piecewise_l200_200014

theorem eq_abs_piecewise (x : ℝ) : (|x| = if x >= 0 then x else -x) :=
by
  sorry

end eq_abs_piecewise_l200_200014


namespace negation_exists_l200_200962

theorem negation_exists :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ x) ↔ ∃ x : ℝ, x^2 + 1 < x :=
sorry

end negation_exists_l200_200962


namespace average_snowfall_dec_1861_l200_200085

theorem average_snowfall_dec_1861 (snowfall : ℕ) (days_in_dec : ℕ) (hours_in_day : ℕ) 
  (time_period : ℕ) (Avg_inch_per_hour : ℚ) : 
  snowfall = 492 ∧ days_in_dec = 31 ∧ hours_in_day = 24 ∧ time_period = days_in_dec * hours_in_day ∧ 
  Avg_inch_per_hour = snowfall / time_period → 
  Avg_inch_per_hour = 492 / (31 * 24) :=
by sorry

end average_snowfall_dec_1861_l200_200085


namespace cost_of_5_spoons_l200_200608

theorem cost_of_5_spoons (cost_per_set : ℕ) (num_spoons_per_set : ℕ) (num_spoons_needed : ℕ)
  (h1 : cost_per_set = 21) (h2 : num_spoons_per_set = 7) (h3 : num_spoons_needed = 5) :
  (cost_per_set / num_spoons_per_set) * num_spoons_needed = 15 :=
by
  sorry

end cost_of_5_spoons_l200_200608


namespace christopher_age_l200_200438

variables (C G : ℕ)

theorem christopher_age :
  (C = 2 * G) ∧ (C - 9 = 5 * (G - 9)) → C = 24 :=
by
  intro h
  sorry

end christopher_age_l200_200438


namespace door_height_l200_200920

theorem door_height
  (x : ℝ) -- length of the pole
  (pole_eq_diag : x = real.sqrt ((x - 4)^2 + (x - 2)^2)) -- condition 3 (Pythagorean theorem)
  : real.sqrt ((x - 4)^2 + (x - 2)^2) = 10 :=
by
  sorry

end door_height_l200_200920


namespace area_of_triangle_ABC_l200_200268

def point : Type := ℝ × ℝ

def A : point := (2, 1)
def B : point := (1, 4)
def on_line (C : point) : Prop := C.1 + C.2 = 9
def area_triangle (A B C : point) : ℝ := 0.5 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - B.1 * A.2 - C.1 * B.2 - A.1 * C.2)

theorem area_of_triangle_ABC :
  ∃ C : point, on_line C ∧ area_triangle A B C = 2 :=
sorry

end area_of_triangle_ABC_l200_200268
