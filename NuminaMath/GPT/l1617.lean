import Mathlib

namespace percentage_increase_in_area_l1617_161726

variable (L W : Real)

theorem percentage_increase_in_area (hL : L > 0) (hW : W > 0) :
  ((1 + 0.25) * L * (1 + 0.25) * W - L * W) / (L * W) * 100 = 56.25 := by
  sorry

end percentage_increase_in_area_l1617_161726


namespace gcd_sequence_property_l1617_161779

theorem gcd_sequence_property (a : ℕ → ℕ) (m n : ℕ) (h : ∀ m n, m > n → Nat.gcd (a m) (a n) = Nat.gcd (a (m - n)) (a n)) : 
  Nat.gcd (a m) (a n) = a (Nat.gcd m n) :=
by
  sorry

end gcd_sequence_property_l1617_161779


namespace bird_families_flew_away_for_winter_l1617_161700

def bird_families_africa : ℕ := 38
def bird_families_asia : ℕ := 80
def total_bird_families_flew_away : ℕ := bird_families_africa + bird_families_asia

theorem bird_families_flew_away_for_winter : total_bird_families_flew_away = 118 := by
  -- proof goes here (not required)
  sorry

end bird_families_flew_away_for_winter_l1617_161700


namespace largest_integer_value_x_l1617_161765

theorem largest_integer_value_x : ∀ (x : ℤ), (5 - 4 * x > 17) → x ≤ -4 := sorry

end largest_integer_value_x_l1617_161765


namespace sum_ages_in_five_years_l1617_161769

theorem sum_ages_in_five_years (L J : ℕ) (hL : L = 13) (h_relation : L = 2 * J + 3) : 
  (L + 5) + (J + 5) = 28 := 
by 
  sorry

end sum_ages_in_five_years_l1617_161769


namespace false_proposition_of_quadratic_l1617_161744

theorem false_proposition_of_quadratic
  (a : ℝ) (h0 : a ≠ 0)
  (h1 : ¬(5 = a * (1/2)^2 + (-a^2 - 1) * (1/2) + a))
  (h2 : (a^2 + 1) / (2 * a) > 0)
  (h3 : (0, a) = (0, x) ∧ x > 0)
  (h4 : ∀ x : ℝ, a * x^2 + (-a^2 - 1) * x + a ≤ 0) :
  false :=
sorry

end false_proposition_of_quadratic_l1617_161744


namespace range_of_a_l1617_161725

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ 2 - a * x
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def h (x : ℝ) : ℝ := Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ f x a = h x) →
  1 ≤ a ∧ a ≤ Real.exp 1 + 1 / Real.exp 1 :=
sorry

end range_of_a_l1617_161725


namespace sum_of_series_l1617_161727

noncomputable def sum_term (k : ℕ) : ℝ :=
  (7 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))

theorem sum_of_series : (∑' k : ℕ, sum_term (k + 1)) = 7 / 4 := by
  sorry

end sum_of_series_l1617_161727


namespace point_divides_segment_l1617_161787

theorem point_divides_segment (x₁ y₁ x₂ y₂ m n : ℝ) (h₁ : (x₁, y₁) = (3, 7)) (h₂ : (x₂, y₂) = (5, 1)) (h₃ : m = 1) (h₄ : n = 3) :
  ( (m * x₂ + n * x₁) / (m + n), (m * y₂ + n * y₁) / (m + n) ) = (3.5, 5.5) :=
by
  sorry

end point_divides_segment_l1617_161787


namespace cats_awake_l1617_161701

theorem cats_awake (total_cats asleep_cats cats_awake : ℕ) (h1 : total_cats = 98) (h2 : asleep_cats = 92) (h3 : cats_awake = total_cats - asleep_cats) : cats_awake = 6 :=
by
  -- Definitions and conditions
  subst h1
  subst h2
  subst h3
  -- The statement we need to prove
  sorry

end cats_awake_l1617_161701


namespace find_S10_l1617_161790

def sequence_sums (S : ℕ → ℚ) (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = 3 * S n - S (n + 1) - 1)

theorem find_S10 (S a : ℕ → ℚ) (h : sequence_sums S a) : S 10 = 513 / 2 :=
  sorry

end find_S10_l1617_161790


namespace Xingyou_age_is_3_l1617_161740

theorem Xingyou_age_is_3 (x : ℕ) (h1 : x = x) (h2 : x + 3 = 2 * x) : x = 3 :=
by
  sorry

end Xingyou_age_is_3_l1617_161740


namespace student_marks_l1617_161778

variable (x : ℕ)
variable (passing_marks : ℕ)
variable (max_marks : ℕ := 400)
variable (fail_by : ℕ := 14)

theorem student_marks :
  (passing_marks = 36 * max_marks / 100) →
  (x + fail_by = passing_marks) →
  x = 130 :=
by sorry

end student_marks_l1617_161778


namespace center_of_circle_l1617_161707

theorem center_of_circle : ∃ c : ℝ × ℝ, 
  (∃ r : ℝ, ∀ x y : ℝ, (x - c.1) * (x - c.1) + (y - c.2) * (y - c.2) = r ↔ x^2 + y^2 - 6*x - 2*y - 15 = 0) → c = (3, 1) :=
by 
  sorry

end center_of_circle_l1617_161707


namespace shop_makes_off_each_jersey_l1617_161721

theorem shop_makes_off_each_jersey :
  ∀ (T : ℝ) (jersey_earnings : ℝ),
  (T = 25) →
  (jersey_earnings = T + 90) →
  jersey_earnings = 115 := by
  intros T jersey_earnings ht hj
  sorry

end shop_makes_off_each_jersey_l1617_161721


namespace number_when_added_by_5_is_30_l1617_161729

theorem number_when_added_by_5_is_30 (x: ℕ) (h: x - 10 = 15) : x + 5 = 30 :=
by
  sorry

end number_when_added_by_5_is_30_l1617_161729


namespace pieces_not_chewed_l1617_161780

theorem pieces_not_chewed : 
  (8 * 7 - 54) = 2 := 
by 
  sorry

end pieces_not_chewed_l1617_161780


namespace find_smallest_result_l1617_161716

namespace small_result

def num_set : Set Int := { -10, -4, 0, 2, 7 }

def all_results : Set Int := 
  { z | ∃ x ∈ num_set, ∃ y ∈ num_set, z = x * y ∨ z = x + y }

def smallest_result := -70

theorem find_smallest_result : ∃ z ∈ all_results, z = smallest_result :=
by
  sorry

end small_result

end find_smallest_result_l1617_161716


namespace two_faucets_fill_60_gallons_l1617_161747

def four_faucets_fill (tub_volume : ℕ) (time_minutes : ℕ) : Prop :=
  4 * (tub_volume / time_minutes) = 120 / 5

def two_faucets_fill (tub_volume : ℕ) (time_minutes : ℕ) : Prop :=
  2 * (tub_volume / time_minutes) = 60 / time_minutes

theorem two_faucets_fill_60_gallons :
  (four_faucets_fill 120 5) → ∃ t: ℕ, two_faucets_fill 60 t ∧ t = 5 :=
by {
  sorry
}

end two_faucets_fill_60_gallons_l1617_161747


namespace total_fruit_weight_l1617_161781

-- Definitions for the conditions
def mario_ounces : ℕ := 8
def lydia_ounces : ℕ := 24
def nicolai_pounds : ℕ := 6
def ounces_per_pound : ℕ := 16

-- Theorem statement
theorem total_fruit_weight : 
  ((mario_ounces / ounces_per_pound : ℚ) + 
   (lydia_ounces / ounces_per_pound : ℚ) + 
   (nicolai_pounds : ℚ)) = 8 := 
sorry

end total_fruit_weight_l1617_161781


namespace largest_tile_size_l1617_161793

theorem largest_tile_size
  (length width : ℕ)
  (H1 : length = 378)
  (H2 : width = 595) :
  Nat.gcd length width = 7 :=
by
  sorry

end largest_tile_size_l1617_161793


namespace smallest_positive_period_maximum_f_B_l1617_161794

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi / 3) - Real.sqrt 3 / 2

theorem smallest_positive_period (x : ℝ) : 
  (∀ T, (f (x + T) = f x) → (T ≥ 0) → T = Real.pi) := 
sorry

variable {a b c : ℝ}

lemma cos_law_cos_B (h : b^2 = a * c) : 
  ∀ (B : ℝ), (B > 0) ∧ (B < Real.pi) → 
  (1 / 2) ≤ Real.cos B ∧ Real.cos B < 1 := 
sorry

theorem maximum_f_B (h : b^2 = a * c) :
  ∀ (B : ℝ), (B > 0) ∧ (B < Real.pi) → 
  f B ≤ 1 := 
sorry

end smallest_positive_period_maximum_f_B_l1617_161794


namespace math_problem_l1617_161739

theorem math_problem (x y : ℕ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end math_problem_l1617_161739


namespace find_j_l1617_161741

theorem find_j (n j : ℕ) (h1 : n % j = 28) (h2 : (n : ℝ) / j = 142.07) : j = 400 :=
by
  sorry

end find_j_l1617_161741


namespace rectangle_area_correct_l1617_161796

-- Definitions of side lengths
def sideOne : ℝ := 5.9
def sideTwo : ℝ := 3

-- Definition of the area calculation for a rectangle
def rectangleArea (a b : ℝ) : ℝ :=
  a * b

-- The main theorem stating the area is as calculated
theorem rectangle_area_correct :
  rectangleArea sideOne sideTwo = 17.7 := by
  sorry

end rectangle_area_correct_l1617_161796


namespace max_value_of_function_l1617_161749

theorem max_value_of_function (x : ℝ) (h : x < 1 / 2) : 
  ∃ y, y = 2 * x + 1 / (2 * x - 1) ∧ y ≤ -1 :=
by
  sorry

end max_value_of_function_l1617_161749


namespace problem_1_problem_2_l1617_161732

def A : Set ℝ := { x | x^2 - 3 * x + 2 < 0 }

def B (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 3 * a + 1 }

-- Problem 1
theorem problem_1 (a : ℝ) (h : a = 1 / 4) : A ∩ B a = { x | 1 < x ∧ x < 7 / 4 } :=
by
  rw [h]
  sorry

-- Problem 2
theorem problem_2 : (∀ x, A x → B a x) → ∀ a, 1 / 3 ≤ a ∧ a ≤ 2 :=
by
  sorry

end problem_1_problem_2_l1617_161732


namespace side_length_square_l1617_161751

theorem side_length_square (A : ℝ) (s : ℝ) (h1 : A = 30) (h2 : A = s^2) : 5 < s ∧ s < 6 :=
by
  -- the proof would go here
  sorry

end side_length_square_l1617_161751


namespace find_p_value_l1617_161798

noncomputable def solve_p (m p : ℕ) :=
  (1^m / 5^m) * (1^16 / 4^16) = 1 / (2 * p^31)

theorem find_p_value (m p : ℕ) (hm : m = 31) :
  solve_p m p ↔ p = 10 :=
by
  sorry

end find_p_value_l1617_161798


namespace total_area_of_hexagon_is_693_l1617_161799

-- Conditions
def hexagon_side1_length := 3
def hexagon_side2_length := 2
def angle_between_length3_sides := 120
def all_internal_triangles_are_equilateral := true
def number_of_triangles := 6

-- Define the problem statement
theorem total_area_of_hexagon_is_693 
  (a1 : hexagon_side1_length = 3)
  (a2 : hexagon_side2_length = 2)
  (a3 : angle_between_length3_sides = 120)
  (a4 : all_internal_triangles_are_equilateral = true)
  (a5 : number_of_triangles = 6) :
  total_area_of_hexagon = 693 :=
by
  sorry

end total_area_of_hexagon_is_693_l1617_161799


namespace approx_average_sqft_per_person_l1617_161719

noncomputable def average_sqft_per_person 
  (population : ℕ) 
  (land_area_sqmi : ℕ) 
  (sqft_per_sqmi : ℕ) : ℕ :=
(sqft_per_sqmi * land_area_sqmi) / population

theorem approx_average_sqft_per_person :
  average_sqft_per_person 331000000 3796742 (5280 ^ 2) = 319697 := 
sorry

end approx_average_sqft_per_person_l1617_161719


namespace abcd_sum_l1617_161734

theorem abcd_sum : 
  ∃ (a b c d : ℕ), 
    (∃ x y : ℝ, x + y = 5 ∧ 2 * x * y = 6 ∧ 
      (x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d)) →
    a + b + c + d = 21 :=
by
  sorry

end abcd_sum_l1617_161734


namespace probability_at_least_9_heads_in_12_flips_l1617_161763

theorem probability_at_least_9_heads_in_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := (Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)
  let probability := favorable_outcomes / total_outcomes
  probability = 299 / 4096 := 
by
  sorry

end probability_at_least_9_heads_in_12_flips_l1617_161763


namespace perimeter_increase_ratio_of_sides_l1617_161738

def width_increase (a : ℝ) : ℝ := 1.1 * a
def length_increase (b : ℝ) : ℝ := 1.2 * b
def original_perimeter (a b : ℝ) : ℝ := 2 * (a + b)
def new_perimeter (a b : ℝ) : ℝ := 2 * (1.1 * a + 1.2 * b)

theorem perimeter_increase : ∀ a b : ℝ, 
  (a > 0) → (b > 0) → 
  (new_perimeter a b - original_perimeter a b) / (original_perimeter a b) * 100 < 20 := 
by
  sorry

theorem ratio_of_sides (a b : ℝ) (h : new_perimeter a b = 1.18 * original_perimeter a b) : a / b = 1 / 4 := 
by
  sorry

end perimeter_increase_ratio_of_sides_l1617_161738


namespace min_value_expression_l1617_161756

theorem min_value_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 7) ^ 2 + (3 * Real.sin α + 4 * Real.cos β - 12) ^ 2 ≥ 36 :=
sorry

end min_value_expression_l1617_161756


namespace point_inside_circle_l1617_161702

theorem point_inside_circle :
  let center := (2, 3)
  let radius := 2
  let P := (1, 2)
  let distance_squared := (P.1 - center.1)^2 + (P.2 - center.2)^2
  distance_squared < radius^2 :=
by
  -- Definitions
  let center := (2, 3)
  let radius := 2
  let P := (1, 2)
  let distance_squared := (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2

  -- Goal
  show distance_squared < radius ^ 2
  
  -- Skip Proof
  sorry

end point_inside_circle_l1617_161702


namespace number_of_ordered_pairs_l1617_161703

theorem number_of_ordered_pairs : ∃ (s : Finset (ℂ × ℂ)), 
    (∀ (a b : ℂ), (a, b) ∈ s → a^5 * b^3 = 1 ∧ a^9 * b^2 = 1) ∧ 
    s.card = 17 := 
by
  sorry

end number_of_ordered_pairs_l1617_161703


namespace sufficient_not_necessary_condition_l1617_161767

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x > 2) → ((x + 1) * (x - 2) > 0) ∧ ¬(∀ y, (y + 1) * (y - 2) > 0 → y > 2) := 
sorry

end sufficient_not_necessary_condition_l1617_161767


namespace coin_diameter_l1617_161715

theorem coin_diameter (r : ℝ) (h : r = 7) : 2 * r = 14 := by
  rw [h]
  norm_num

end coin_diameter_l1617_161715


namespace cube_volume_and_surface_area_l1617_161795

theorem cube_volume_and_surface_area (e : ℕ) (h : 12 * e = 72) :
  (e^3 = 216) ∧ (6 * e^2 = 216) := by
  sorry

end cube_volume_and_surface_area_l1617_161795


namespace total_number_of_red_and_white_jelly_beans_in_fishbowl_l1617_161783

def number_of_red_jelly_beans_in_bag := 24
def number_of_white_jelly_beans_in_bag := 18
def number_of_bags := 3

theorem total_number_of_red_and_white_jelly_beans_in_fishbowl :
  number_of_red_jelly_beans_in_bag * number_of_bags + number_of_white_jelly_beans_in_bag * number_of_bags = 126 := by
  sorry

end total_number_of_red_and_white_jelly_beans_in_fishbowl_l1617_161783


namespace find_numbers_l1617_161758

theorem find_numbers (A B C D : ℚ) 
  (h1 : A + B = 44)
  (h2 : 5 * A = 6 * B)
  (h3 : C = 2 * (A - B))
  (h4 : D = (A + B + C) / 3 + 3) :
  A = 24 ∧ B = 20 ∧ C = 8 ∧ D = 61 / 3 := 
  by 
    sorry

end find_numbers_l1617_161758


namespace sum_integers_neg50_to_60_l1617_161754

theorem sum_integers_neg50_to_60 : 
  (Finset.sum (Finset.Icc (-50 : ℤ) 60) id) = 555 := 
by
  -- Placeholder for the actual proof
  sorry

end sum_integers_neg50_to_60_l1617_161754


namespace shaded_areas_equal_l1617_161784

theorem shaded_areas_equal (φ : ℝ) (hφ1 : 0 < φ) (hφ2 : φ < π / 4) : 
  (Real.tan φ) = 2 * φ :=
sorry

end shaded_areas_equal_l1617_161784


namespace walnut_trees_total_l1617_161742

theorem walnut_trees_total : 33 + 44 = 77 :=
by
  sorry

end walnut_trees_total_l1617_161742


namespace positive_difference_x_coordinates_lines_l1617_161776

theorem positive_difference_x_coordinates_lines :
  let l := fun x : ℝ => -2 * x + 4
  let m := fun x : ℝ => - (1 / 5) * x + 1
  let x_l := (- (10 - 4) / 2)
  let x_m := (- (10 - 1) * 5)
  abs (x_l - x_m) = 42 := by
  sorry

end positive_difference_x_coordinates_lines_l1617_161776


namespace count_integers_six_times_sum_of_digits_l1617_161745

theorem count_integers_six_times_sum_of_digits (n : ℕ) (h : n < 1000) 
    (digit_sum : ℕ → ℕ)
    (digit_sum_correct : ∀ (n : ℕ), digit_sum n = (n % 10) + ((n / 10) % 10) + (n / 100)) :
    ∃! n, n < 1000 ∧ n = 6 * digit_sum n :=
sorry

end count_integers_six_times_sum_of_digits_l1617_161745


namespace A_worked_alone_after_B_left_l1617_161755

/-- A and B can together finish a work in 40 days. They worked together for 10 days and then B left.
    A alone can finish the job in 80 days. We need to find out how many days did A work alone after B left. -/
theorem A_worked_alone_after_B_left
  (W : ℝ)
  (A_work_rate : ℝ := W / 80)
  (B_work_rate : ℝ := W / 80)
  (AB_work_rate : ℝ := W / 40)
  (work_done_together_in_10_days : ℝ := 10 * (W / 40))
  (remaining_work : ℝ := W - work_done_together_in_10_days)
  (A_rate_alone : ℝ := W / 80) :
  ∃ D : ℝ, D * (W / 80) = remaining_work → D = 60 :=
by
  sorry

end A_worked_alone_after_B_left_l1617_161755


namespace cricket_runs_l1617_161782

theorem cricket_runs (A B C : ℕ) (h1 : A / B = 1 / 3) (h2 : B / C = 1 / 5) (h3 : A + B + C = 95) : C = 75 :=
by
  -- Skipping proof details
  sorry

end cricket_runs_l1617_161782


namespace pairings_count_l1617_161791

-- Define the problem's conditions explicitly
def number_of_bowls : Nat := 6
def number_of_glasses : Nat := 6

-- The theorem stating that the number of pairings is 36
theorem pairings_count : number_of_bowls * number_of_glasses = 36 := by
  sorry

end pairings_count_l1617_161791


namespace cary_strips_ivy_l1617_161757

variable (strip_per_day : ℕ) (grow_per_night : ℕ) (total_ivy : ℕ)

theorem cary_strips_ivy (h1 : strip_per_day = 6) (h2 : grow_per_night = 2) (h3 : total_ivy = 40) :
  (total_ivy / (strip_per_day - grow_per_night)) = 10 := by
  sorry

end cary_strips_ivy_l1617_161757


namespace first_number_is_nine_l1617_161711

theorem first_number_is_nine (x : ℤ) (h : 11 * x = 3 * (x + 4) + 16 + 4 * (x + 2)) : x = 9 :=
by {
  sorry
}

end first_number_is_nine_l1617_161711


namespace abs_difference_of_squares_l1617_161753

theorem abs_difference_of_squares : abs ((102: ℤ) ^ 2 - (98: ℤ) ^ 2) = 800 := by
  sorry

end abs_difference_of_squares_l1617_161753


namespace Ella_food_each_day_l1617_161775

variable {E : ℕ} -- Define E as the number of pounds of food Ella eats each day

def food_dog_eats (E : ℕ) : ℕ := 4 * E -- Definition of food the dog eats each day

def total_food_eaten_in_10_days (E : ℕ) : ℕ := 10 * E + 10 * (food_dog_eats E) -- Total food (Ella and dog) in 10 days

theorem Ella_food_each_day : total_food_eaten_in_10_days E = 1000 → E = 20 :=
by
  intros h -- Assume the given condition
  sorry -- Skip the actual proof

end Ella_food_each_day_l1617_161775


namespace product_of_roots_eq_neg_14_l1617_161792

theorem product_of_roots_eq_neg_14 :
  ∀ (x : ℝ), 25 * x^2 + 60 * x - 350 = 0 → ((-350) / 25) = -14 :=
by
  intros x h
  sorry

end product_of_roots_eq_neg_14_l1617_161792


namespace one_positive_real_solution_l1617_161750

theorem one_positive_real_solution : 
    ∃! x : ℝ, 0 < x ∧ (x ^ 10 + 7 * x ^ 9 + 14 * x ^ 8 + 1729 * x ^ 7 - 1379 * x ^ 6 = 0) :=
sorry

end one_positive_real_solution_l1617_161750


namespace sequence_bounds_l1617_161736

theorem sequence_bounds (θ : ℝ) (n : ℕ) (a : ℕ → ℝ) (hθ : 0 < θ ∧ θ < π / 2) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1 - 2 * (Real.sin θ * Real.cos θ)^2) 
  (h_recurrence : ∀ n, a (n + 2) - a (n + 1) + a n * (Real.sin θ * Real.cos θ)^2 = 0) :
  1 / 2 ^ (n - 1) ≤ a n ∧ a n ≤ 1 - (Real.sin (2 * θ))^n * (1 - 1 / 2 ^ (n - 1)) := 
sorry

end sequence_bounds_l1617_161736


namespace soccer_team_lineups_l1617_161705

noncomputable def num_starting_lineups (n k t g : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) k)

theorem soccer_team_lineups :
  num_starting_lineups 18 9 1 1 = 3501120 := by
    sorry

end soccer_team_lineups_l1617_161705


namespace part1_part2_part3_l1617_161788

def pointM (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m + 3)

-- Part 1
theorem part1 (m : ℝ) (h : 2 * m + 3 = 0) : pointM m = (-5 / 2, 0) :=
  sorry

-- Part 2
theorem part2 (m : ℝ) (h : 2 * m + 3 = -1) : pointM m = (-3, -1) :=
  sorry

-- Part 3
theorem part3 (m : ℝ) (h1 : |m - 1| = 2) : pointM m = (2, 9) ∨ pointM m = (-2, 1) :=
  sorry

end part1_part2_part3_l1617_161788


namespace abc_inequality_l1617_161728

theorem abc_inequality (x y z : ℝ) (a b c : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : a = (x * (y - z) ^ 2) ^ 2) (h2 : b = (y * (z - x) ^ 2) ^ 2) (h3 : c = (z * (x - y) ^ 2) ^ 2) :
  a^2 + b^2 + c^2 ≥ 2 * (a * b + b * c + c * a) :=
by {
  sorry
}

end abc_inequality_l1617_161728


namespace combined_number_of_fasteners_l1617_161773

def lorenzo_full_cans_total_fasteners
  (thumbtacks_cans : ℕ)
  (pushpins_cans : ℕ)
  (staples_cans : ℕ)
  (thumbtacks_per_board : ℕ)
  (pushpins_per_board : ℕ)
  (staples_per_board : ℕ)
  (boards_tested : ℕ)
  (thumbtacks_remaining : ℕ)
  (pushpins_remaining : ℕ)
  (staples_remaining : ℕ) :
  ℕ :=
  let thumbtacks_used := thumbtacks_per_board * boards_tested
  let pushpins_used := pushpins_per_board * boards_tested
  let staples_used := staples_per_board * boards_tested
  let thumbtacks_per_can := thumbtacks_used + thumbtacks_remaining
  let pushpins_per_can := pushpins_used + pushpins_remaining
  let staples_per_can := staples_used + staples_remaining
  let total_thumbtacks := thumbtacks_per_can * thumbtacks_cans
  let total_pushpins := pushpins_per_can * pushpins_cans
  let total_staples := staples_per_can * staples_cans
  total_thumbtacks + total_pushpins + total_staples

theorem combined_number_of_fasteners :
  lorenzo_full_cans_total_fasteners 5 3 2 3 2 4 150 45 35 25 = 4730 :=
  by
  sorry

end combined_number_of_fasteners_l1617_161773


namespace sphere_radius_is_16_25_l1617_161712

def sphere_in_cylinder_radius (r : ℝ) : Prop := 
  ∃ (x : ℝ), (x ^ 2 + 15 ^ 2 = r ^ 2) ∧ ((x + 10) ^ 2 = r ^ 2) ∧ (r = 16.25)

theorem sphere_radius_is_16_25 : 
  sphere_in_cylinder_radius 16.25 :=
sorry

end sphere_radius_is_16_25_l1617_161712


namespace merchant_markup_l1617_161768

theorem merchant_markup (C : ℝ) (M : ℝ) (h1 : (1 + M / 100 - 0.40 * (1 + M / 100)) * C = 1.05 * C) : 
  M = 75 := sorry

end merchant_markup_l1617_161768


namespace smallest_base_for_101_l1617_161723

theorem smallest_base_for_101 : ∃ b : ℕ, b = 10 ∧ b ≤ 101 ∧ 101 < b^2 :=
by
  -- We state the simplest form of the theorem,
  -- then use the answer from the solution step.
  use 10
  sorry

end smallest_base_for_101_l1617_161723


namespace exists_x0_lt_l1617_161789

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d
noncomputable def Q (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem exists_x0_lt {a b c d p q r s : ℝ} (h1 : r < s) (h2 : s - r > 2)
  (h3 : ∀ x, r < x ∧ x < s → P x a b c d < 0 ∧ Q x p q < 0)
  (h4 : ∀ x, x < r ∨ x > s → P x a b c d >= 0 ∧ Q x p q >= 0) :
  ∃ x0, r < x0 ∧ x0 < s ∧ P x0 a b c d < Q x0 p q :=
sorry

end exists_x0_lt_l1617_161789


namespace cube_edge_length_l1617_161708

theorem cube_edge_length (total_edge_length : ℕ) (num_edges : ℕ) (h1 : total_edge_length = 108) (h2 : num_edges = 12) : total_edge_length / num_edges = 9 := by 
  -- additional formal mathematical steps can follow here
  sorry

end cube_edge_length_l1617_161708


namespace fish_worth_rice_l1617_161771

variables (f l r : ℝ)

-- Conditions based on the problem statement
def fish_for_bread : Prop := 3 * f = 2 * l
def bread_for_rice : Prop := l = 4 * r

-- Statement to be proven
theorem fish_worth_rice (h₁ : fish_for_bread f l) (h₂ : bread_for_rice l r) : f = (8 / 3) * r :=
  sorry

end fish_worth_rice_l1617_161771


namespace find_a7_l1617_161752

variable {a : ℕ → ℕ}  -- Define the geometric sequence as a function from natural numbers to natural numbers.
variable (h_geo_seq : ∀ (n k : ℕ), a n ^ 2 = a (n - k) * a (n + k)) -- property of geometric sequences
variable (h_a3 : a 3 = 2) -- given a₃ = 2
variable (h_a5 : a 5 = 8) -- given a₅ = 8

theorem find_a7 : a 7 = 32 :=
by
  sorry

end find_a7_l1617_161752


namespace positive_number_is_nine_l1617_161772

theorem positive_number_is_nine (x : ℝ) (n : ℝ) (hx : x > 0) (hn : n > 0)
  (sqrt1 : x^2 = n) (sqrt2 : (x - 6)^2 = n) : 
  n = 9 :=
by
  sorry

end positive_number_is_nine_l1617_161772


namespace Victoria_money_left_l1617_161746

noncomputable def Victoria_initial_money : ℝ := 10000
noncomputable def jacket_price : ℝ := 250
noncomputable def trousers_price : ℝ := 180
noncomputable def purse_price : ℝ := 450
noncomputable def jackets_bought : ℕ := 8
noncomputable def trousers_bought : ℕ := 15
noncomputable def purses_bought : ℕ := 4
noncomputable def discount_rate : ℝ := 0.15
noncomputable def dinner_bill_inclusive : ℝ := 552.50
noncomputable def dinner_service_charge_rate : ℝ := 0.15

theorem Victoria_money_left : 
  Victoria_initial_money - 
  ((jackets_bought * jacket_price + trousers_bought * trousers_price) * (1 - discount_rate) + 
   purses_bought * purse_price + 
   dinner_bill_inclusive / (1 + dinner_service_charge_rate)) = 3725 := 
by 
  sorry

end Victoria_money_left_l1617_161746


namespace tangent_line_at_1_2_l1617_161714

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

def tangent_eq (x y : ℝ) : Prop := y = 2 * x

theorem tangent_line_at_1_2 : tangent_eq 1 2 :=
by
  have f_1 := 1
  have f'_1 := 2
  sorry

end tangent_line_at_1_2_l1617_161714


namespace solve_for_y_l1617_161777

theorem solve_for_y : ∃ (y : ℚ), y + 2 - 2 / 3 = 4 * y - (y + 2) ∧ y = 5 / 3 :=
by
  sorry

end solve_for_y_l1617_161777


namespace average_distinct_k_values_l1617_161713

theorem average_distinct_k_values (k : ℕ) (h : ∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ r1 > 0 ∧ r2 > 0) : k = 15 :=
sorry

end average_distinct_k_values_l1617_161713


namespace fourth_powers_count_l1617_161733

theorem fourth_powers_count (n m : ℕ) (h₁ : n^4 ≥ 100) (h₂ : m^4 ≤ 10000) :
  ∃ k, k = m - n + 1 ∧ k = 7 :=
by
  sorry

end fourth_powers_count_l1617_161733


namespace lion_king_cost_l1617_161761

theorem lion_king_cost
  (LK_earned : ℕ := 200) -- The Lion King earned 200 million
  (LK_profit : ℕ := 190) -- The Lion King profit calculated from half of Star Wars' profit
  (SW_cost : ℕ := 25)    -- Star Wars cost 25 million
  (SW_earned : ℕ := 405) -- Star Wars earned 405 million
  (SW_profit : SW_earned - SW_cost = 380) -- Star Wars profit
  (LK_profit_from_SW : LK_profit = 1/2 * (SW_earned - SW_cost)) -- The Lion King profit calculation
  (LK_cost : ℕ := LK_earned - LK_profit) -- The Lion King cost calculation
  : LK_cost = 10 := 
sorry

end lion_king_cost_l1617_161761


namespace lines_perpendicular_l1617_161770

-- Definition of lines and their relationships
def Line : Type := ℝ × ℝ × ℝ → Prop

variables (a b c : Line)

-- Condition 1: a is perpendicular to b
axiom perp (a b : Line) : Prop
-- Condition 2: b is parallel to c
axiom parallel (b c : Line) : Prop

-- Theorem to prove: 
theorem lines_perpendicular (h1 : perp a b) (h2 : parallel b c) : perp a c :=
sorry

end lines_perpendicular_l1617_161770


namespace constant_seq_is_arith_not_always_geom_l1617_161718

theorem constant_seq_is_arith_not_always_geom (c : ℝ) (seq : ℕ → ℝ) (h : ∀ n, seq n = c) :
  (∀ n, seq (n + 1) - seq n = 0) ∧ (c = 0 ∨ (∀ n, seq (n + 1) / seq n = 1)) :=
by
  sorry

end constant_seq_is_arith_not_always_geom_l1617_161718


namespace product_of_coordinates_of_D_l1617_161717

theorem product_of_coordinates_of_D (D : ℝ × ℝ) (N : ℝ × ℝ) (C : ℝ × ℝ) 
  (hN : N = (4, 3)) (hC : C = (5, -1)) (midpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 * D.2 = 21 :=
by
  sorry

end product_of_coordinates_of_D_l1617_161717


namespace round_robin_games_l1617_161731

theorem round_robin_games (x : ℕ) (h : ∃ (n : ℕ), n = 15) : (x * (x - 1)) / 2 = 15 :=
sorry

end round_robin_games_l1617_161731


namespace power_of_two_l1617_161766

theorem power_of_two (Number : ℕ) (h1 : Number = 128) (h2 : Number * (1/4 : ℝ) = 2^5) :
  ∃ power : ℕ, 2^power = 128 := 
by
  use 7
  sorry

end power_of_two_l1617_161766


namespace max_min_product_l1617_161710

theorem max_min_product (A B : ℕ) (h : A + B = 100) : 
  (∃ (maxProd : ℕ), maxProd = 2500 ∧ (∀ (A B : ℕ), A + B = 100 → A * B ≤ maxProd)) ∧
  (∃ (minProd : ℕ), minProd = 0 ∧ (∀ (A B : ℕ), A + B = 100 → minProd ≤ A * B)) :=
by 
  -- Proof omitted
  sorry

end max_min_product_l1617_161710


namespace sum_of_rel_prime_greater_than_one_l1617_161748

theorem sum_of_rel_prime_greater_than_one (a : ℕ) (h : a > 6) : 
  ∃ b c : ℕ, a = b + c ∧ b > 1 ∧ c > 1 ∧ Nat.gcd b c = 1 :=
sorry

end sum_of_rel_prime_greater_than_one_l1617_161748


namespace correct_money_calculation_l1617_161760

structure BootSale :=
(initial_money : ℕ)
(price_per_boot : ℕ)
(total_taken : ℕ)
(total_returned : ℕ)
(money_spent : ℕ)
(remaining_money_to_return : ℕ)

theorem correct_money_calculation (bs : BootSale) :
  bs.initial_money = 25 →
  bs.price_per_boot = 12 →
  bs.total_taken = 25 →
  bs.total_returned = 5 →
  bs.money_spent = 3 →
  bs.remaining_money_to_return = 2 →
  bs.total_taken - bs.total_returned + bs.money_spent = 23 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end correct_money_calculation_l1617_161760


namespace different_values_count_l1617_161724

theorem different_values_count (i : ℕ) (h : 1 ≤ i ∧ i ≤ 2015) : 
  ∃ l : Finset ℕ, (∀ j ∈ l, ∃ i : ℕ, (1 ≤ i ∧ i ≤ 2015) ∧ j = (i^2 / 2015)) ∧
  l.card = 2016 := 
sorry

end different_values_count_l1617_161724


namespace minimum_value_l1617_161720

theorem minimum_value (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  (2 / (x + 3 * y) + 1 / (x - y)) = (3 + 2 * Real.sqrt 2) / 2 := sorry

end minimum_value_l1617_161720


namespace part_I_part_II_part_III_l1617_161785

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (1 / 2) * a * x^2

-- Part (Ⅰ)
theorem part_I (x : ℝ) : (0 < x) → (f 1 x < f 1 (x+1)) := sorry

-- Part (Ⅱ)
theorem part_II (f_has_two_distinct_extreme_values : ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ (f a x = f a y))) : 0 < a ∧ a < 1 := sorry

-- Part (Ⅲ)
theorem part_III (f_has_two_distinct_zeros : ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0)) : 0 < a ∧ a < (2 / Real.exp 1) := sorry

end part_I_part_II_part_III_l1617_161785


namespace compute_cd_l1617_161722

-- Define the variables c and d as real numbers
variables (c d : ℝ)

-- Define the conditions
def condition1 : Prop := c + d = 10
def condition2 : Prop := c^3 + d^3 = 370

-- State the theorem we need to prove
theorem compute_cd (h1 : condition1 c d) (h2 : condition2 c d) : c * d = 21 :=
by
  sorry

end compute_cd_l1617_161722


namespace diane_15_cents_arrangement_l1617_161759

def stamps : List (ℕ × ℕ) := 
  [(1, 1), 
   (2, 2), 
   (3, 3), 
   (4, 4), 
   (5, 5), 
   (6, 6), 
   (7, 7), 
   (8, 8), 
   (9, 9), 
   (10, 10), 
   (11, 11), 
   (12, 12)]

def number_of_arrangements (value : ℕ) (stamps : List (ℕ × ℕ)) : ℕ := sorry

theorem diane_15_cents_arrangement : number_of_arrangements 15 stamps = 32 := 
sorry

end diane_15_cents_arrangement_l1617_161759


namespace multiple_is_eight_l1617_161737

theorem multiple_is_eight (m : ℝ) (h : 17 = m * 2.625 - 4) : m = 8 :=
by
  sorry

end multiple_is_eight_l1617_161737


namespace Gina_gave_fraction_to_mom_l1617_161735

variable (M : ℝ)

theorem Gina_gave_fraction_to_mom :
  (∃ M, M + (1/8 : ℝ) * 400 + (1/5 : ℝ) * 400 + 170 = 400) →
  M / 400 = 1/4 :=
by
  intro h
  sorry

end Gina_gave_fraction_to_mom_l1617_161735


namespace number_of_streams_l1617_161774

theorem number_of_streams (S A B C D : Type) (f : S → A) (f1 : A → B) :
  (∀ (x : ℕ), x = 1000 → 
  (x * 375 / 1000 = 375 ∧ x * 625 / 1000 = 625) ∧ 
  (S ≠ C ∧ S ≠ D ∧ C ≠ D)) →
  -- Introduce some conditions to represent the described transition process
  -- Specifically the conditions mentioning the lakes and transitions 
  ∀ (transition_count : ℕ), 
    (transition_count = 4) →
    ∃ (number_of_streams : ℕ), number_of_streams = 3 := 
sorry

end number_of_streams_l1617_161774


namespace quadratic_roots_inverse_sum_l1617_161704

theorem quadratic_roots_inverse_sum (t q α β : ℝ) (h1 : α + β = t) (h2 : α * β = q) 
  (h3 : ∀ n : ℕ, n ≥ 1 → α^n + β^n = t) : (1 / α^2011 + 1 / β^2011) = 2 := 
by 
  sorry

end quadratic_roots_inverse_sum_l1617_161704


namespace fractions_order_l1617_161764

theorem fractions_order :
  (25 / 21 < 23 / 19) ∧ (23 / 19 < 21 / 17) :=
by {
  sorry
}

end fractions_order_l1617_161764


namespace ratio_of_cube_volumes_l1617_161786

theorem ratio_of_cube_volumes (a b : ℕ) (ha : a = 10) (hb : b = 25) :
  (a^3 : ℚ) / (b^3 : ℚ) = 8 / 125 := by
  sorry

end ratio_of_cube_volumes_l1617_161786


namespace alice_age_l1617_161709

theorem alice_age (x : ℕ) (h1 : ∃ n : ℕ, x - 4 = n^2) (h2 : ∃ m : ℕ, x + 2 = m^3) : x = 58 :=
sorry

end alice_age_l1617_161709


namespace wrap_XL_boxes_per_roll_l1617_161797

-- Conditions
def rolls_per_shirt_box : ℕ := 5
def num_shirt_boxes : ℕ := 20
def num_XL_boxes : ℕ := 12
def cost_per_roll : ℕ := 4
def total_cost : ℕ := 32

-- Prove that one roll of wrapping paper can wrap 3 XL boxes
theorem wrap_XL_boxes_per_roll : (num_XL_boxes / ((total_cost / cost_per_roll) - (num_shirt_boxes / rolls_per_shirt_box))) = 3 := 
sorry

end wrap_XL_boxes_per_roll_l1617_161797


namespace correct_statement_l1617_161762

noncomputable def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  let rec b_aux (m : ℕ) :=
    match m with
    | 0     => 0
    | m + 1 => 1 + 1 / (α m + b_aux m)
  b_aux n

theorem correct_statement (α : ℕ → ℕ) (h : ∀ k, α k > 0) : b 4 α < b 7 α :=
by sorry

end correct_statement_l1617_161762


namespace grant_earnings_l1617_161743

theorem grant_earnings 
  (baseball_cards_sale : ℕ) 
  (baseball_bat_sale : ℕ) 
  (baseball_glove_price : ℕ) 
  (baseball_glove_discount : ℕ) 
  (baseball_cleats_sale : ℕ) : 
  baseball_cards_sale + baseball_bat_sale + (baseball_glove_price - baseball_glove_discount) + 2 * baseball_cleats_sale = 79 :=
by
  let baseball_cards_sale := 25
  let baseball_bat_sale := 10
  let baseball_glove_price := 30
  let baseball_glove_discount := (30 * 20) / 100
  let baseball_cleats_sale := 10
  sorry

end grant_earnings_l1617_161743


namespace actual_distance_traveled_l1617_161706

theorem actual_distance_traveled (T : ℝ) :
  ∀ D : ℝ, (D = 4 * T) → (D + 6 = 5 * T) → D = 24 :=
by
  intro D h1 h2
  sorry

end actual_distance_traveled_l1617_161706


namespace swimming_class_attendance_l1617_161730

theorem swimming_class_attendance (total_students : ℕ) (chess_percentage : ℝ) (swimming_percentage : ℝ) 
  (H1 : total_students = 1000) 
  (H2 : chess_percentage = 0.20) 
  (H3 : swimming_percentage = 0.10) : 
  200 * 0.10 = 20 := 
by sorry

end swimming_class_attendance_l1617_161730
