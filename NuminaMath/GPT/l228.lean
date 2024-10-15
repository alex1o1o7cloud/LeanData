import Mathlib

namespace NUMINAMATH_GPT_find_k_l228_22849

-- Defining the conditions used in the problem context
def line_condition (k a b : ℝ) : Prop :=
  (b = 4 * k + 1) ∧ (5 = k * a + 1) ∧ (b + 1 = k * a + 1)

-- The statement of the theorem
theorem find_k (a b k : ℝ) (h : line_condition k a b) : k = 3 / 4 :=
by sorry

end NUMINAMATH_GPT_find_k_l228_22849


namespace NUMINAMATH_GPT_min_cost_for_boxes_l228_22831

theorem min_cost_for_boxes
  (box_length: ℕ) (box_width: ℕ) (box_height: ℕ)
  (cost_per_box: ℝ) (total_volume: ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 15)
  (h4 : cost_per_box = 1.30)
  (h5 : total_volume = 3060000) :
  ∃ cost: ℝ, cost = 663 :=
by
  sorry

end NUMINAMATH_GPT_min_cost_for_boxes_l228_22831


namespace NUMINAMATH_GPT_spring_expenses_l228_22859

noncomputable def expense_by_end_of_february : ℝ := 0.6
noncomputable def expense_by_end_of_may : ℝ := 1.8
noncomputable def spending_during_spring_months := expense_by_end_of_may - expense_by_end_of_february

-- Lean statement for the proof problem
theorem spring_expenses : spending_during_spring_months = 1.2 := by
  sorry

end NUMINAMATH_GPT_spring_expenses_l228_22859


namespace NUMINAMATH_GPT_distance_between_points_l228_22890

theorem distance_between_points :
  let p1 := (-4, 17)
  let p2 := (12, -1)
  let distance := Real.sqrt ((12 - (-4))^2 + (-1 - 17)^2)
  distance = 2 * Real.sqrt 145 := sorry

end NUMINAMATH_GPT_distance_between_points_l228_22890


namespace NUMINAMATH_GPT_smallest_positive_integer_n_l228_22814

theorem smallest_positive_integer_n :
  ∃ n : ℕ, (n > 0 ∧ 17 * n % 7 = 2) ∧ ∀ m : ℕ, (m > 0 ∧ 17 * m % 7 = 2) → n ≤ m := 
sorry

end NUMINAMATH_GPT_smallest_positive_integer_n_l228_22814


namespace NUMINAMATH_GPT_minimum_value_frac_sum_l228_22824

theorem minimum_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 / y = 3) :
  (2 / x + y) ≥ 8 / 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_frac_sum_l228_22824


namespace NUMINAMATH_GPT_number_is_twenty_l228_22841

-- We state that if \( \frac{30}{100}x = \frac{15}{100} \times 40 \), then \( x = 20 \)
theorem number_is_twenty (x : ℝ) (h : (30 / 100) * x = (15 / 100) * 40) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_is_twenty_l228_22841


namespace NUMINAMATH_GPT_xiaolong_average_speed_l228_22820

noncomputable def averageSpeed (dist_home_store : ℕ) (time_home_store : ℕ) 
                               (speed_store_playground : ℕ) (time_store_playground : ℕ) 
                               (dist_playground_school : ℕ) (speed_playground_school : ℕ) 
                               (total_time : ℕ) : ℕ :=
  let dist_store_playground := speed_store_playground * time_store_playground
  let time_playground_school := dist_playground_school / speed_playground_school
  let total_distance := dist_home_store + dist_store_playground + dist_playground_school
  total_distance / total_time

theorem xiaolong_average_speed :
  averageSpeed 500 7 80 8 300 60 20 = 72 := by
  sorry

end NUMINAMATH_GPT_xiaolong_average_speed_l228_22820


namespace NUMINAMATH_GPT_area_of_trapezium_eq_336_l228_22848

-- Define the lengths of the parallel sides and the distance between them
def a := 30 -- length of one parallel side in cm
def b := 12 -- length of the other parallel side in cm
def h := 16 -- distance between the parallel sides (height) in cm

-- Define the expected area
def expectedArea := 336 -- area in square cm

-- State the theorem to prove
theorem area_of_trapezium_eq_336 : (1/2 : ℝ) * (a + b) * h = expectedArea := 
by 
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_area_of_trapezium_eq_336_l228_22848


namespace NUMINAMATH_GPT_find_setC_l228_22837

def setA := {x : ℝ | x^2 - 3 * x + 2 = 0}
def setB (a : ℝ) := {x : ℝ | a * x - 2 = 0}
def union_condition (a : ℝ) : Prop := (setA ∪ setB a) = setA
def setC := {a : ℝ | union_condition a}

theorem find_setC : setC = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_find_setC_l228_22837


namespace NUMINAMATH_GPT_floor_length_l228_22874

variable (b l : ℝ)

theorem floor_length :
  (l = 3 * b) →
  (3 * b ^ 2 = 128) →
  l = 19.59 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_floor_length_l228_22874


namespace NUMINAMATH_GPT_probability_of_sum_17_is_correct_l228_22806

def probability_sum_17 : ℚ :=
  let favourable_outcomes := 2
  let total_outcomes := 81
  favourable_outcomes / total_outcomes

theorem probability_of_sum_17_is_correct :
  probability_sum_17 = 2 / 81 :=
by
  -- The proof steps are not required for this task
  sorry

end NUMINAMATH_GPT_probability_of_sum_17_is_correct_l228_22806


namespace NUMINAMATH_GPT_longest_segment_cylinder_l228_22893

theorem longest_segment_cylinder (r h : ℤ) (c : ℝ) (hr : r = 4) (hh : h = 9) : 
  c = Real.sqrt (2 * r * r + h * h) ↔ c = Real.sqrt 145 :=
by
  sorry

end NUMINAMATH_GPT_longest_segment_cylinder_l228_22893


namespace NUMINAMATH_GPT_coin_flip_heads_probability_l228_22842

theorem coin_flip_heads_probability :
  let coins : List String := ["penny", "nickel", "dime", "quarter", "half-dollar"]
  let independent_event (coin : String) : Prop := True
  let outcomes := (2 : ℕ) ^ (List.length coins)
  let successful_outcomes := 4
  let probability := successful_outcomes / outcomes
  probability = 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_coin_flip_heads_probability_l228_22842


namespace NUMINAMATH_GPT_factory_output_exceeds_by_20_percent_l228_22896

theorem factory_output_exceeds_by_20_percent 
  (planned_output : ℝ) (actual_output : ℝ)
  (h_planned : planned_output = 20)
  (h_actual : actual_output = 24) :
  ((actual_output - planned_output) / planned_output) * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_factory_output_exceeds_by_20_percent_l228_22896


namespace NUMINAMATH_GPT_ratio_of_friday_to_thursday_l228_22827

theorem ratio_of_friday_to_thursday
  (wednesday_copies : ℕ)
  (total_copies : ℕ)
  (ratio : ℚ)
  (h1 : wednesday_copies = 15)
  (h2 : total_copies = 69)
  (h3 : ratio = 1 / 5) :
  (total_copies - wednesday_copies - 3 * wednesday_copies) / (3 * wednesday_copies) = ratio :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_friday_to_thursday_l228_22827


namespace NUMINAMATH_GPT_find_n_tangent_eq_1234_l228_22854

theorem find_n_tangent_eq_1234 (n : ℤ) (h1 : -90 < n ∧ n < 90) (h2 : Real.tan (n * Real.pi / 180) = Real.tan (1234 * Real.pi / 180)) : n = -26 := 
by 
  sorry

end NUMINAMATH_GPT_find_n_tangent_eq_1234_l228_22854


namespace NUMINAMATH_GPT_speed_of_second_car_l228_22822

/-!
Two cars started from the same point, at 5 am, traveling in opposite directions. 
One car was traveling at 50 mph, and they were 450 miles apart at 10 am. 
Prove that the speed of the other car is 40 mph.
-/

variable (S : ℝ) -- Speed of the second car

theorem speed_of_second_car
    (h1 : ∀ t : ℝ, t = 5) -- The time of travel from 5 am to 10 am is 5 hours 
    (h2 : ∀ d₁ : ℝ, d₁ = 50 * 5) -- Distance traveled by the first car
    (h3 : ∀ d₂ : ℝ, d₂ = S * 5) -- Distance traveled by the second car
    (h4 : 450 = 50 * 5 + S * 5) -- Total distance between the two cars
    : S = 40 := sorry

end NUMINAMATH_GPT_speed_of_second_car_l228_22822


namespace NUMINAMATH_GPT_smallest_five_digit_divisible_by_2_3_8_9_l228_22872

-- Definitions for the conditions given in the problem
def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000
def divisible_by (n d : ℕ) : Prop := d ∣ n

-- The main theorem stating the problem
theorem smallest_five_digit_divisible_by_2_3_8_9 :
  ∃ n : ℕ, is_five_digit n ∧ divisible_by n 2 ∧ divisible_by n 3 ∧ divisible_by n 8 ∧ divisible_by n 9 ∧ n = 10008 :=
sorry

end NUMINAMATH_GPT_smallest_five_digit_divisible_by_2_3_8_9_l228_22872


namespace NUMINAMATH_GPT_rocket_parachute_opens_l228_22801

theorem rocket_parachute_opens (h t : ℝ) : h = -t^2 + 12 * t + 1 ∧ h = 37 -> t = 6 :=
by sorry

end NUMINAMATH_GPT_rocket_parachute_opens_l228_22801


namespace NUMINAMATH_GPT_geometric_diff_l228_22873

-- Definitions based on conditions
def is_geometric (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ (d2 * d2 = d1 * d3)

-- Problem statement
theorem geometric_diff :
  let largest_geometric := 964
  let smallest_geometric := 124
  is_geometric largest_geometric ∧ is_geometric smallest_geometric ∧
  (largest_geometric - smallest_geometric = 840) :=
by
  sorry

end NUMINAMATH_GPT_geometric_diff_l228_22873


namespace NUMINAMATH_GPT_solve_for_x_l228_22847

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l228_22847


namespace NUMINAMATH_GPT_correct_statements_l228_22836

-- Define the function and the given conditions
def f : ℝ → ℝ := sorry

lemma not_constant (h: ∃ x y: ℝ, x ≠ y ∧ f x ≠ f y) : true := sorry
lemma periodic (x : ℝ) : f (x - 1) = f (x + 1) := sorry
lemma symmetric (x : ℝ) : f (2 - x) = f x := sorry

-- The statements we want to prove
theorem correct_statements : 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, f (1 - x) = f (1 + x)) ∧ 
  (∃ T > 0, ∀ x, f (x + T) = f x)
:= by
  sorry

end NUMINAMATH_GPT_correct_statements_l228_22836


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l228_22891

-- Problem 1
theorem problem1 (x : ℝ) (h : x * (5 * x + 4) = 5 * x + 4) : x = -4 / 5 ∨ x = 1 := 
sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : -3 * x^2 + 22 * x - 24 = 0) : x = 6 ∨ x = 4 / 3 := 
sorry

-- Problem 3
theorem problem3 (x : ℝ) (h : (x + 8) * (x + 1) = -12) : x = -4 ∨ x = -5 := 
sorry

-- Problem 4
theorem problem4 (x : ℝ) (h : (3 * x + 2) * (x + 3) = x + 14) : x = -4 ∨ x = 2 / 3 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l228_22891


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_not_sufficient_condition_l228_22875

theorem necessary_not_sufficient_condition (x : ℝ) :
  (1 < x ∧ x < 4) → (|x - 2| < 1) := sorry

theorem not_sufficient_condition (x : ℝ) :
  (|x - 2| < 1) → (1 < x ∧ x < 4) := sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_not_sufficient_condition_l228_22875


namespace NUMINAMATH_GPT_increasing_on_interval_l228_22817

theorem increasing_on_interval (a : ℝ) : (∀ x : ℝ, x > 1/2 → (2 * x + a + 1 / x^2) ≥ 0) → a ≥ -3 :=
by
  intros h
  -- Rest of the proof would go here
  sorry

end NUMINAMATH_GPT_increasing_on_interval_l228_22817


namespace NUMINAMATH_GPT_gray_eyed_black_haired_students_l228_22843

theorem gray_eyed_black_haired_students (total_students : ℕ) 
  (green_eyed_red_haired : ℕ) (black_haired : ℕ) (gray_eyed : ℕ) 
  (h_total : total_students = 50)
  (h_green_eyed_red_haired : green_eyed_red_haired = 17)
  (h_black_haired : black_haired = 27)
  (h_gray_eyed : gray_eyed = 23) :
  ∃ (gray_eyed_black_haired : ℕ), gray_eyed_black_haired = 17 :=
by sorry

end NUMINAMATH_GPT_gray_eyed_black_haired_students_l228_22843


namespace NUMINAMATH_GPT_find_s2_length_l228_22853

variables (s r : ℝ)

def condition1 : Prop := 2 * r + s = 2420
def condition2 : Prop := 2 * r + 3 * s = 4040

theorem find_s2_length (h1 : condition1 s r) (h2 : condition2 s r) : s = 810 :=
sorry

end NUMINAMATH_GPT_find_s2_length_l228_22853


namespace NUMINAMATH_GPT_sandy_initial_carrots_l228_22858

-- Defining the conditions
def sam_took : ℕ := 3
def sandy_left : ℕ := 3

-- The statement to be proven
theorem sandy_initial_carrots :
  (sandy_left + sam_took = 6) :=
by
  sorry

end NUMINAMATH_GPT_sandy_initial_carrots_l228_22858


namespace NUMINAMATH_GPT_range_of_a_l228_22818

-- Define sets P and M
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def M (a : ℝ) : Set ℝ := {x | (2 - a) ≤ x ∧ x ≤ (1 + a)}

-- Prove the range of a
theorem range_of_a (a : ℝ) : (P ∩ (M a) = P) ↔ (a ≥ 1) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l228_22818


namespace NUMINAMATH_GPT_inequality_abcd_l228_22809

theorem inequality_abcd (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) :
    (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) >= 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_abcd_l228_22809


namespace NUMINAMATH_GPT_average_of_first_n_multiples_of_8_is_88_l228_22884

theorem average_of_first_n_multiples_of_8_is_88 (n : ℕ) (h : (n / 2) * (8 + 8 * n) / n = 88) : n = 21 :=
sorry

end NUMINAMATH_GPT_average_of_first_n_multiples_of_8_is_88_l228_22884


namespace NUMINAMATH_GPT_students_distribution_l228_22839

theorem students_distribution (students villages : ℕ) (h_students : students = 4) (h_villages : villages = 3) :
  ∃ schemes : ℕ, schemes = 36 := 
sorry

end NUMINAMATH_GPT_students_distribution_l228_22839


namespace NUMINAMATH_GPT_titu_andreescu_inequality_l228_22816

theorem titu_andreescu_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 :=
sorry

end NUMINAMATH_GPT_titu_andreescu_inequality_l228_22816


namespace NUMINAMATH_GPT_probability_of_white_first_red_second_l228_22803

noncomputable def probability_white_first_red_second : ℚ :=
let totalBalls := 6
let probWhiteFirst := 1 / totalBalls
let remainingBalls := totalBalls - 1
let probRedSecond := 1 / remainingBalls
probWhiteFirst * probRedSecond

theorem probability_of_white_first_red_second :
  probability_white_first_red_second = 1 / 30 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_white_first_red_second_l228_22803


namespace NUMINAMATH_GPT_red_flowers_count_l228_22895

-- Let's define the given conditions
def total_flowers : ℕ := 10
def white_flowers : ℕ := 2
def blue_percentage : ℕ := 40

-- Calculate the number of blue flowers
def blue_flowers : ℕ := (blue_percentage * total_flowers) / 100

-- The property we want to prove is the number of red flowers
theorem red_flowers_count :
  total_flowers - (blue_flowers + white_flowers) = 4 :=
by
  sorry

end NUMINAMATH_GPT_red_flowers_count_l228_22895


namespace NUMINAMATH_GPT_percentage_decrease_is_correct_l228_22813

variable (P : ℝ)

-- Condition 1: After the first year, the price increased by 30%
def price_after_first_year : ℝ := 1.30 * P

-- Condition 2: At the end of the 2-year period, the price of the painting is 110.5% of the original price
def price_after_second_year : ℝ := 1.105 * P

-- Condition 3: Let D be the percentage decrease during the second year
def D : ℝ := 0.15

-- Goal: Prove that the percentage decrease during the second year is 15%
theorem percentage_decrease_is_correct : 
  1.30 * P - D * 1.30 * P = 1.105 * P → D = 0.15 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_is_correct_l228_22813


namespace NUMINAMATH_GPT_wolves_total_games_l228_22832

theorem wolves_total_games
  (x y : ℕ) -- Before district play, the Wolves had won x games out of y games.
  (hx : x = 40 * y / 100) -- The Wolves had won 40% of their basketball games before district play.
  (hx' : 5 * x = 2 * y)
  (hy : 60 * (y + 10) / 100 = x + 9) -- They finished the season having won 60% of their total games.
  : y + 10 = 25 := by
  sorry

end NUMINAMATH_GPT_wolves_total_games_l228_22832


namespace NUMINAMATH_GPT_chewing_gum_company_revenue_l228_22880

theorem chewing_gum_company_revenue (R : ℝ) :
  let projected_revenue := 1.25 * R
  let actual_revenue := 0.75 * R
  (actual_revenue / projected_revenue) * 100 = 60 := 
by
  sorry

end NUMINAMATH_GPT_chewing_gum_company_revenue_l228_22880


namespace NUMINAMATH_GPT_multiple_of_four_and_six_prime_sum_even_l228_22865

theorem multiple_of_four_and_six_prime_sum_even {a b : ℤ} 
  (h_a : ∃ m : ℤ, a = 4 * m) 
  (h_b1 : ∃ n : ℤ, b = 6 * n) 
  (h_b2 : Prime b) : 
  Even (a + b) := 
  by sorry

end NUMINAMATH_GPT_multiple_of_four_and_six_prime_sum_even_l228_22865


namespace NUMINAMATH_GPT_area_of_largest_circle_l228_22863

theorem area_of_largest_circle (side_length : ℝ) (h : side_length = 2) : 
  (Real.pi * (side_length / 2)^2 = 3.14) :=
by
  sorry

end NUMINAMATH_GPT_area_of_largest_circle_l228_22863


namespace NUMINAMATH_GPT_hexagon_inscribed_circumscribed_symmetric_l228_22869

-- Define the conditions of the problem
variables (R r c : ℝ)

-- Define the main assertion of the problem
theorem hexagon_inscribed_circumscribed_symmetric :
  3 * (R^2 - c^2)^4 - 4 * r^2 * (R^2 - c^2)^2 * (R^2 + c^2) - 16 * R^2 * c^2 * r^4 = 0 :=
by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_hexagon_inscribed_circumscribed_symmetric_l228_22869


namespace NUMINAMATH_GPT_geometric_sequence_sum_div_l228_22899

theorem geometric_sequence_sum_div :
  ∀ {a : ℕ → ℝ} {q : ℝ},
  (∀ n, a (n + 1) = a n * q) →
  q = -1 / 3 →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by
  intros a q geometric_seq common_ratio
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_div_l228_22899


namespace NUMINAMATH_GPT_domain_of_f_l228_22879

open Set

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 1)

theorem domain_of_f :
  {x : ℝ | x^2 - 1 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l228_22879


namespace NUMINAMATH_GPT_multiplier_eq_l228_22819

-- Definitions of the given conditions
def length (w : ℝ) (m : ℝ) : ℝ := m * w + 2
def perimeter (l : ℝ) (w : ℝ) : ℝ := 2 * l + 2 * w

-- Condition definitions
def l : ℝ := 38
def P : ℝ := 100

-- Proof statement
theorem multiplier_eq (m w : ℝ) (h1 : length w m = l) (h2 : perimeter l w = P) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_multiplier_eq_l228_22819


namespace NUMINAMATH_GPT_perpendicular_vectors_l228_22829

/-- Given vectors a and b, prove that m = 6 if a is perpendicular to b -/
theorem perpendicular_vectors {m : ℝ} (h₁ : (1, 5, -2) = (1, 5, -2)) (h₂ : ∃ m : ℝ, (m, 2, m+2) = (m, 2, m+2)) (h₃ : (1 * m + 5 * 2 + (-2) * (m + 2) = 0)) :
  m = 6 :=
sorry

end NUMINAMATH_GPT_perpendicular_vectors_l228_22829


namespace NUMINAMATH_GPT_problem_2535_l228_22808

theorem problem_2535 (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 1) :
  a + b + (a^3 / b^2) + (b^3 / a^2) = 2535 := sorry

end NUMINAMATH_GPT_problem_2535_l228_22808


namespace NUMINAMATH_GPT_sum_of_A_B_C_l228_22882

theorem sum_of_A_B_C (A B C : ℕ) (h_pos_A : 0 < A) (h_pos_B : 0 < B) (h_pos_C : 0 < C) (h_rel_prime : Nat.gcd A (Nat.gcd B C) = 1) (h_eq : A * Real.log 3 / Real.log 180 + B * Real.log 5 / Real.log 180 = C) : A + B + C = 4 :=
sorry

end NUMINAMATH_GPT_sum_of_A_B_C_l228_22882


namespace NUMINAMATH_GPT_part1_part2_l228_22800

def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}
def C : Set ℝ := {x | -1 < x ∧ x < 4}

theorem part1 : A ∩ (B 3)ᶜ = Set.Icc 3 5 := by
  sorry

theorem part2 : A ∩ B m = C → m = 8 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l228_22800


namespace NUMINAMATH_GPT_largest_consecutive_integers_sum_to_45_l228_22812

theorem largest_consecutive_integers_sum_to_45 (x n : ℕ) (h : 45 = n * (2 * x + n - 1) / 2) : n ≤ 9 :=
sorry

end NUMINAMATH_GPT_largest_consecutive_integers_sum_to_45_l228_22812


namespace NUMINAMATH_GPT_yoongi_number_division_l228_22861

theorem yoongi_number_division (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 :=
by
  sorry

end NUMINAMATH_GPT_yoongi_number_division_l228_22861


namespace NUMINAMATH_GPT_find_a_l228_22852

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }
def setB : Set ℝ := { x | Real.log (x^2 - 5 * x + 8) / Real.log 2 = 1 }
def setC (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }

-- Proof statement to find the value of a
theorem find_a (a : ℝ) : setA ∩ setC a = ∅ → setB ∩ setC a ≠ ∅ → a = -2 := by
  sorry

end NUMINAMATH_GPT_find_a_l228_22852


namespace NUMINAMATH_GPT_min_value_fraction_l228_22826

theorem min_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : a < (2 / 3) * b) (h3 : c ≥ b^2 / (3 * a)) : 
  ∃ x : ℝ, (∀ y : ℝ, y ≥ x → y ≥ 1) ∧ (x = 1) :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l228_22826


namespace NUMINAMATH_GPT_Tim_pencils_value_l228_22877

variable (Sarah_pencils : ℕ)
variable (Tyrah_pencils : ℕ)
variable (Tim_pencils : ℕ)

axiom Tyrah_condition : Tyrah_pencils = 6 * Sarah_pencils
axiom Tim_condition : Tim_pencils = 8 * Sarah_pencils
axiom Tyrah_pencils_value : Tyrah_pencils = 12

theorem Tim_pencils_value : Tim_pencils = 16 :=
by
  sorry

end NUMINAMATH_GPT_Tim_pencils_value_l228_22877


namespace NUMINAMATH_GPT_verify_value_l228_22805

theorem verify_value (a b c d m : ℝ) 
  (h₁ : a = -b) 
  (h₂ : c * d = 1) 
  (h₃ : |m| = 3) :
  3 * c * d + (a + b) / (c * d) - m = 0 ∨ 
  3 * c * d + (a + b) / (c * d) - m = 6 := 
sorry

end NUMINAMATH_GPT_verify_value_l228_22805


namespace NUMINAMATH_GPT_rectangle_area_eq_2a_squared_l228_22887

variable {α : Type} [Semiring α] (a : α)

-- Conditions
def width (a : α) : α := a
def length (a : α) : α := 2 * a

-- Proof statement
theorem rectangle_area_eq_2a_squared (a : α) : (length a) * (width a) = 2 * a^2 := 
sorry

end NUMINAMATH_GPT_rectangle_area_eq_2a_squared_l228_22887


namespace NUMINAMATH_GPT_magnesium_is_limiting_l228_22846

-- Define the conditions
def moles_Mg : ℕ := 4
def moles_CO2 : ℕ := 2
def moles_O2 : ℕ := 2 -- represent excess O2, irrelevant to limiting reagent
def mag_ox_reaction (mg : ℕ) (o2 : ℕ) (mgo : ℕ) : Prop := 2 * mg + o2 = 2 * mgo
def mag_carbon_reaction (mg : ℕ) (co2 : ℕ) (mgco3 : ℕ) : Prop := mg + co2 = mgco3

-- Assume Magnesium is the limiting reagent for both reactions
theorem magnesium_is_limiting (mgo : ℕ) (mgco3 : ℕ) :
  mag_ox_reaction moles_Mg moles_O2 mgo ∧ mag_carbon_reaction moles_Mg moles_CO2 mgco3 →
  mgo = 4 ∧ mgco3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_magnesium_is_limiting_l228_22846


namespace NUMINAMATH_GPT_quadratic_minimization_l228_22866

theorem quadratic_minimization : 
  ∃ x : ℝ, ∀ y : ℝ, (x^2 - 12 * x + 36 ≤ y^2 - 12 * y + 36) ∧ x^2 - 12 * x + 36 = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_minimization_l228_22866


namespace NUMINAMATH_GPT_geometric_sequence_property_l228_22834

theorem geometric_sequence_property (a : ℕ → ℝ) (h : ∀ n, a (n + 1) / a n = a 1 / a 0) (h₁ : a 5 * a 14 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_property_l228_22834


namespace NUMINAMATH_GPT_probability_two_cards_l228_22883

noncomputable def probability_first_spade_second_ace : ℚ :=
  let total_cards := 52
  let total_spades := 13
  let total_aces := 4
  let remaining_cards := total_cards - 1
  
  let first_spade_non_ace := (total_spades - 1) / total_cards
  let second_ace_after_non_ace := total_aces / remaining_cards
  
  let probability_case1 := first_spade_non_ace * second_ace_after_non_ace
  
  let first_ace_spade := 1 / total_cards
  let second_ace_after_ace := (total_aces - 1) / remaining_cards
  
  let probability_case2 := first_ace_spade * second_ace_after_ace
  
  probability_case1 + probability_case2

theorem probability_two_cards {p : ℚ} (h : p = 1 / 52) : 
  probability_first_spade_second_ace = p := 
by 
  simp only [probability_first_spade_second_ace]
  sorry

end NUMINAMATH_GPT_probability_two_cards_l228_22883


namespace NUMINAMATH_GPT_opposite_neg_9_l228_22823

theorem opposite_neg_9 : 
  ∃ y : Int, -9 + y = 0 ∧ y = 9 :=
by
  sorry

end NUMINAMATH_GPT_opposite_neg_9_l228_22823


namespace NUMINAMATH_GPT_friends_pets_ratio_l228_22876

theorem friends_pets_ratio (pets_total : ℕ) (pets_taylor : ℕ) (pets_friend4 : ℕ) (pets_friend5 : ℕ)
  (pets_first3_total : ℕ) : pets_total = 32 → pets_taylor = 4 → pets_friend4 = 2 → pets_friend5 = 2 →
  pets_first3_total = pets_total - pets_taylor - pets_friend4 - pets_friend5 →
  (pets_first3_total : ℚ) / pets_taylor = 6 :=
by
  sorry

end NUMINAMATH_GPT_friends_pets_ratio_l228_22876


namespace NUMINAMATH_GPT_Karen_has_fewer_nail_polishes_than_Kim_l228_22864

theorem Karen_has_fewer_nail_polishes_than_Kim :
  ∀ (Kim Heidi Karen : ℕ), Kim = 12 → Heidi = Kim + 5 → Karen + Heidi = 25 → (Kim - Karen) = 4 :=
by
  intros Kim Heidi Karen hK hH hKH
  sorry

end NUMINAMATH_GPT_Karen_has_fewer_nail_polishes_than_Kim_l228_22864


namespace NUMINAMATH_GPT_oa_dot_ob_eq_neg2_l228_22894

/-!
# Problem Statement
Given AB as the diameter of the smallest radius circle centered at C(0,1) that intersects 
the graph of y = 1 / (|x| - 1), where O is the origin. Prove that the dot product 
\overrightarrow{OA} · \overrightarrow{OB} equals -2.
-/

noncomputable def smallest_radius_circle_eqn (x : ℝ) : ℝ :=
  x^2 + ((1 / (|x| - 1)) - 1)^2

noncomputable def radius_of_circle (x : ℝ) : ℝ :=
  Real.sqrt (smallest_radius_circle_eqn x)

noncomputable def OA (x : ℝ) : ℝ × ℝ :=
  (x, (1 / (|x| - 1)) + 1)

noncomputable def OB (x : ℝ) : ℝ × ℝ :=
  (-x, 1 - (1 / (|x| - 1)))

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem oa_dot_ob_eq_neg2 (x : ℝ) (hx : |x| > 1) :
  let a := OA x
  let b := OB x
  dot_product a b = -2 :=
by
  sorry

end NUMINAMATH_GPT_oa_dot_ob_eq_neg2_l228_22894


namespace NUMINAMATH_GPT_cathy_can_win_l228_22835

theorem cathy_can_win (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  (∃ (f : ℕ → ℕ) (hf : ∀ i, f i < n + 1), (∀ i j, (i < j) → (f i < f j) → (f j = f i + 1)) → n ≤ 2^(k-1)) :=
sorry

end NUMINAMATH_GPT_cathy_can_win_l228_22835


namespace NUMINAMATH_GPT_div_sub_eq_l228_22825

theorem div_sub_eq : 0.24 / 0.004 - 0.1 = 59.9 := by
  sorry

end NUMINAMATH_GPT_div_sub_eq_l228_22825


namespace NUMINAMATH_GPT_abs_ab_cd_leq_one_fourth_l228_22888

theorem abs_ab_cd_leq_one_fourth (a b c d : ℝ) (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) (h3 : 0 ≤ d) (h_sum : a + b + c + d = 1) :
  |a * b - c * d| ≤ 1 / 4 :=
sorry

end NUMINAMATH_GPT_abs_ab_cd_leq_one_fourth_l228_22888


namespace NUMINAMATH_GPT_kayden_total_processed_l228_22811

-- Definition of the given conditions and final proof problem statement in Lean 4
variable (x : ℕ)  -- x is the number of cartons delivered to each customer

theorem kayden_total_processed (h : 4 * (x - 60) = 160) : 4 * x = 400 :=
by
  sorry

end NUMINAMATH_GPT_kayden_total_processed_l228_22811


namespace NUMINAMATH_GPT_tangent_series_identity_l228_22828

noncomputable def series_tangent (x : ℝ) : ℝ := ∑' n, (1 / (2 ^ n)) * Real.tan (x / (2 ^ n))

theorem tangent_series_identity (x : ℝ) : 
  (1 / x) - (1 / Real.tan x) = series_tangent x := 
sorry

end NUMINAMATH_GPT_tangent_series_identity_l228_22828


namespace NUMINAMATH_GPT_find_angle_measure_l228_22868

def complement_more_condition (x : ℝ) : Prop :=
  90 - x = (1 / 7) * x + 26

theorem find_angle_measure (x : ℝ) (h : complement_more_condition x) : x = 56 :=
sorry

end NUMINAMATH_GPT_find_angle_measure_l228_22868


namespace NUMINAMATH_GPT_probability_of_region_l228_22804

-- Definition of the bounds
def bounds (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8

-- Definition of the region where x + y <= 5
def region (x y : ℝ) : Prop := x + y ≤ 5

-- The proof statement
theorem probability_of_region : 
  (∃ (x y : ℝ), bounds x y ∧ region x y) →
  ∃ (p : ℚ), p = 3/8 :=
by sorry

end NUMINAMATH_GPT_probability_of_region_l228_22804


namespace NUMINAMATH_GPT_proposition_false_at_4_l228_22860

theorem proposition_false_at_4 (P : ℕ → Prop) (hp : ∀ k : ℕ, k > 0 → (P k → P (k + 1))) (h4 : ¬ P 5) : ¬ P 4 :=
by {
    sorry
}

end NUMINAMATH_GPT_proposition_false_at_4_l228_22860


namespace NUMINAMATH_GPT_x_y_value_l228_22855

theorem x_y_value (x y : ℝ) (h : x^2 + y^2 = 8 * x - 4 * y - 30) : x + y = 2 :=
sorry

end NUMINAMATH_GPT_x_y_value_l228_22855


namespace NUMINAMATH_GPT_line_through_two_points_l228_22892

theorem line_through_two_points (A B : ℝ × ℝ)
  (hA : A = (2, -3))
  (hB : B = (1, 4)) :
  ∃ (m b : ℝ), (∀ x y : ℝ, (y = m * x + b) ↔ ((x, y) = A ∨ (x, y) = B)) ∧ m = -7 ∧ b = 11 := by
  sorry

end NUMINAMATH_GPT_line_through_two_points_l228_22892


namespace NUMINAMATH_GPT_compare_sqrt_difference_minimize_materials_plan_compare_a_inv_l228_22878

-- Problem 1
theorem compare_sqrt_difference : 3 - Real.sqrt 2 > 4 - 2 * Real.sqrt 2 := 
  sorry

-- Problem 2
theorem minimize_materials_plan (x y : ℝ) (h : x > y) : 
  4 * x + 6 * y > 3 * x + 7 * y := 
  sorry

-- Problem 3
theorem compare_a_inv (a : ℝ) (h : a > 0) : 
  (0 < a ∧ a < 1) → a < 1 / a ∧ (a = 1 → a = 1 / a) ∧ (a > 1 → a > 1 / a) :=
  sorry

end NUMINAMATH_GPT_compare_sqrt_difference_minimize_materials_plan_compare_a_inv_l228_22878


namespace NUMINAMATH_GPT_statues_at_end_of_fourth_year_l228_22881

def initial_statues : ℕ := 4
def statues_after_second_year : ℕ := initial_statues * 4
def statues_added_third_year : ℕ := 12
def broken_statues_third_year : ℕ := 3
def statues_removed_third_year : ℕ := broken_statues_third_year
def statues_added_fourth_year : ℕ := broken_statues_third_year * 2

def statues_end_of_first_year : ℕ := initial_statues
def statues_end_of_second_year : ℕ := statues_after_second_year
def statues_end_of_third_year : ℕ := statues_end_of_second_year + statues_added_third_year - statues_removed_third_year
def statues_end_of_fourth_year : ℕ := statues_end_of_third_year + statues_added_fourth_year

theorem statues_at_end_of_fourth_year : statues_end_of_fourth_year = 31 :=
by
  sorry

end NUMINAMATH_GPT_statues_at_end_of_fourth_year_l228_22881


namespace NUMINAMATH_GPT_product_and_quotient_l228_22886

theorem product_and_quotient : (16 * 0.0625 / 4 * 0.5 * 2) = (1 / 4) :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_product_and_quotient_l228_22886


namespace NUMINAMATH_GPT_ribbons_problem_l228_22857

/-
    In a large box of ribbons, 1/3 are yellow, 1/4 are purple, 1/6 are orange, and the remaining 40 ribbons are black.
    Prove that the total number of orange ribbons is 27.
-/

theorem ribbons_problem :
  ∀ (total : ℕ), 
    (1 / 3 : ℚ) * total + (1 / 4 : ℚ) * total + (1 / 6 : ℚ) * total + 40 = total →
    (1 / 6 : ℚ) * total = 27 := sorry

end NUMINAMATH_GPT_ribbons_problem_l228_22857


namespace NUMINAMATH_GPT_exists_not_perfect_square_l228_22802

theorem exists_not_perfect_square (a b c : ℤ) : ∃ (n : ℕ), n > 0 ∧ ¬ ∃ k : ℕ, n^3 + a * n^2 + b * n + c = k^2 :=
by
  sorry

end NUMINAMATH_GPT_exists_not_perfect_square_l228_22802


namespace NUMINAMATH_GPT_function_characterization_l228_22844

def isRelativelyPrime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem function_characterization (f : ℕ → ℤ) (hyp : ∀ x y, isRelativelyPrime x y → f (x + y) = f (x + 1) + f (y + 1)) :
  ∃ a b : ℤ, ∀ n : ℕ, f (2 * n) = (n - 1) * b ∧ f (2 * n + 1) = (n - 1) * b + a :=
by
  sorry

end NUMINAMATH_GPT_function_characterization_l228_22844


namespace NUMINAMATH_GPT_khali_shovels_snow_l228_22856

theorem khali_shovels_snow :
  let section1_length := 30
  let section1_width := 3
  let section1_depth := 1
  let section2_length := 15
  let section2_width := 2
  let section2_depth := 0.5
  let volume1 := section1_length * section1_width * section1_depth
  let volume2 := section2_length * section2_width * section2_depth
  volume1 + volume2 = 105 :=
by 
  sorry

end NUMINAMATH_GPT_khali_shovels_snow_l228_22856


namespace NUMINAMATH_GPT_triangular_25_eq_325_l228_22840

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_25_eq_325 : triangular_number 25 = 325 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_triangular_25_eq_325_l228_22840


namespace NUMINAMATH_GPT_balls_per_pack_l228_22870

theorem balls_per_pack (total_packs total_cost cost_per_ball total_balls balls_per_pack : ℕ)
  (h1 : total_packs = 4)
  (h2 : total_cost = 24)
  (h3 : cost_per_ball = 2)
  (h4 : total_balls = total_cost / cost_per_ball)
  (h5 : total_balls = 12)
  (h6 : balls_per_pack = total_balls / total_packs) :
  balls_per_pack = 3 := by 
  sorry

end NUMINAMATH_GPT_balls_per_pack_l228_22870


namespace NUMINAMATH_GPT_rooms_already_painted_l228_22830

-- Define the conditions as variables and hypotheses
variables (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ)
variables (h1 : total_rooms = 10)
variables (h2 : hours_per_room = 8)
variables (h3 : remaining_hours = 16)

-- Define the theorem stating the number of rooms already painted
theorem rooms_already_painted (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ)
  (h1 : total_rooms = 10) (h2 : hours_per_room = 8) (h3 : remaining_hours = 16) :
  (total_rooms - (remaining_hours / hours_per_room) = 8) :=
sorry

end NUMINAMATH_GPT_rooms_already_painted_l228_22830


namespace NUMINAMATH_GPT_conic_section_pair_of_lines_l228_22897

theorem conic_section_pair_of_lines : 
  (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = 0 → (2 * x - 3 * y = 0 ∨ 2 * x + 3 * y = 0)) :=
by
  sorry

end NUMINAMATH_GPT_conic_section_pair_of_lines_l228_22897


namespace NUMINAMATH_GPT_polynomial_div_6_l228_22850

theorem polynomial_div_6 (n : ℕ) : 6 ∣ (2 * n ^ 3 + 9 * n ^ 2 + 13 * n) := 
sorry

end NUMINAMATH_GPT_polynomial_div_6_l228_22850


namespace NUMINAMATH_GPT_length_of_first_platform_l228_22889

theorem length_of_first_platform 
  (t1 t2 : ℝ) 
  (length_train : ℝ) 
  (length_second_platform : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (speed_eq : (t1 + length_train) / time1 = (length_second_platform + length_train) / time2) 
  (time1_eq : time1 = 15) 
  (time2_eq : time2 = 20) 
  (length_train_eq : length_train = 100) 
  (length_second_platform_eq: length_second_platform = 500) :
  t1 = 350 := 
  by 
  sorry

end NUMINAMATH_GPT_length_of_first_platform_l228_22889


namespace NUMINAMATH_GPT_num_factors_of_1320_l228_22871

theorem num_factors_of_1320 : ∃ n : ℕ, (n = 24) ∧ (∃ a b c d : ℕ, 1320 = 2^a * 3^b * 5^c * 11^d ∧ (a + 1) * (b + 1) * (c + 1) * (d + 1) = 24) :=
by
  sorry

end NUMINAMATH_GPT_num_factors_of_1320_l228_22871


namespace NUMINAMATH_GPT_frog_eyes_count_l228_22851

def total_frog_eyes (a b c : ℕ) (eyesA eyesB eyesC : ℕ) : ℕ :=
  a * eyesA + b * eyesB + c * eyesC

theorem frog_eyes_count :
  let a := 2
  let b := 1
  let c := 3
  let eyesA := 2
  let eyesB := 3
  let eyesC := 4
  total_frog_eyes a b c eyesA eyesB eyesC = 19 := by
  sorry

end NUMINAMATH_GPT_frog_eyes_count_l228_22851


namespace NUMINAMATH_GPT_orange_juice_percentage_l228_22815

theorem orange_juice_percentage 
  (V : ℝ) 
  (W : ℝ) 
  (G : ℝ)
  (hV : V = 300)
  (hW: W = 0.4 * V)
  (hG: G = 105) : 
  (V - W - G) / V * 100 = 25 := 
by 
  -- We will need to use sorry to skip the proof and focus just on the statement
  sorry

end NUMINAMATH_GPT_orange_juice_percentage_l228_22815


namespace NUMINAMATH_GPT_first_cyclist_speed_l228_22807

theorem first_cyclist_speed (v₁ v₂ : ℕ) (c t : ℕ) 
  (h1 : v₂ = 8) 
  (h2 : c = 675) 
  (h3 : t = 45) 
  (h4 : v₁ * t + v₂ * t = c) : 
  v₁ = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_first_cyclist_speed_l228_22807


namespace NUMINAMATH_GPT_evaluate_expression_l228_22898

def binom (n k : ℕ) : ℕ := if h : k ≤ n then Nat.choose n k else 0

theorem evaluate_expression : 
  (binom 2 5 * 3 ^ 5) / binom 10 5 = 0 := by
  -- Given conditions:
  have h1 : binom 2 5 = 0 := by sorry
  have h2 : binom 10 5 = 252 := by sorry
  -- Proof goal:
  sorry

end NUMINAMATH_GPT_evaluate_expression_l228_22898


namespace NUMINAMATH_GPT_daniel_age_is_13_l228_22838

-- Define Aunt Emily's age
def aunt_emily_age : ℕ := 48

-- Define Brianna's age as a third of Aunt Emily's age
def brianna_age : ℕ := aunt_emily_age / 3

-- Define that Daniel's age is 3 years less than Brianna's age
def daniel_age : ℕ := brianna_age - 3

-- Theorem to prove Daniel's age is 13 given the conditions
theorem daniel_age_is_13 :
  brianna_age = aunt_emily_age / 3 →
  daniel_age = brianna_age - 3 →
  daniel_age = 13 :=
  sorry

end NUMINAMATH_GPT_daniel_age_is_13_l228_22838


namespace NUMINAMATH_GPT_mult_63_37_l228_22845

theorem mult_63_37 : 63 * 37 = 2331 :=
by {
  sorry
}

end NUMINAMATH_GPT_mult_63_37_l228_22845


namespace NUMINAMATH_GPT_x_coordinate_D_l228_22810

noncomputable def find_x_coordinate_D (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : ℝ := 
  let l := -a * b
  let x := l / c
  x

theorem x_coordinate_D (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (D_on_parabola : d^2 = (a + b) * (d) + l)
  (lines_intersect_y_axis : ∃ l : ℝ, (a^2 = (b + a) * a + l) ∧ (b^2 = (b + a) * b + l) ∧ (c^2 = (d + c) * c + l)) :
  d = (a * b) / c :=
by sorry

end NUMINAMATH_GPT_x_coordinate_D_l228_22810


namespace NUMINAMATH_GPT_ball_hits_ground_time_l228_22821

theorem ball_hits_ground_time (h : ℝ → ℝ) (t : ℝ) :
  (∀ (t : ℝ), h t = -16 * t ^ 2 - 30 * t + 200) → h t = 0 → t = 2.5 :=
by
  -- Placeholder for the formal proof
  sorry

end NUMINAMATH_GPT_ball_hits_ground_time_l228_22821


namespace NUMINAMATH_GPT_find_starting_number_l228_22833

theorem find_starting_number :
  ∃ startnum : ℕ, startnum % 5 = 0 ∧ (∀ k : ℕ, 0 ≤ k ∧ k < 20 → startnum + 5 * k ≤ 100) ∧ startnum = 10 :=
sorry

end NUMINAMATH_GPT_find_starting_number_l228_22833


namespace NUMINAMATH_GPT_more_spent_on_keychains_bracelets_than_tshirts_l228_22867

-- Define the conditions as variables
variable (spent_keychains_bracelets spent_total_spent : ℝ)
variable (spent_keychains_bracelets_eq : spent_keychains_bracelets = 347.00)
variable (spent_total_spent_eq : spent_total_spent = 548.00)

-- Using these conditions, define the problem to prove the desired result
theorem more_spent_on_keychains_bracelets_than_tshirts :
  spent_keychains_bracelets - (spent_total_spent - spent_keychains_bracelets) = 146.00 :=
by
  rw [spent_keychains_bracelets_eq, spent_total_spent_eq]
  sorry

end NUMINAMATH_GPT_more_spent_on_keychains_bracelets_than_tshirts_l228_22867


namespace NUMINAMATH_GPT_union_when_m_equals_4_subset_implies_m_range_l228_22885

-- Define the sets and conditions
def set_A := { x : ℝ | -2 ≤ x ∧ x ≤ 5 }
def set_B (m : ℝ) := { x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Problem 1: When m = 4, find the union of A and B
theorem union_when_m_equals_4 : ∀ x, x ∈ set_A ∪ set_B 4 ↔ -2 ≤ x ∧ x ≤ 7 :=
by sorry

-- Problem 2: If B ⊆ A, find the range of the real number m
theorem subset_implies_m_range (m : ℝ) : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ m ≤ 3 :=
by sorry

end NUMINAMATH_GPT_union_when_m_equals_4_subset_implies_m_range_l228_22885


namespace NUMINAMATH_GPT_g_g_x_has_two_distinct_real_roots_iff_l228_22862

noncomputable def g (d x : ℝ) := x^2 - 4 * x + d

def has_two_distinct_real_roots (f : ℝ → ℝ) : Prop := 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

theorem g_g_x_has_two_distinct_real_roots_iff (d : ℝ) :
  has_two_distinct_real_roots (g d ∘ g d) ↔ d = 8 := sorry

end NUMINAMATH_GPT_g_g_x_has_two_distinct_real_roots_iff_l228_22862
