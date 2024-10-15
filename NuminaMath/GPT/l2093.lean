import Mathlib

namespace NUMINAMATH_GPT_volume_of_prism_l2093_209306

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 18) (h2 : b * c = 20) (h3 : c * a = 12) (h4 : a + b + c = 11) :
  a * b * c = 12 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l2093_209306


namespace NUMINAMATH_GPT_exists_k_for_blocks_of_2022_l2093_209395

theorem exists_k_for_blocks_of_2022 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (0 < k) ∧ (∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → (∃ j, 
  k^i / 10^j % 10^4 = 2022)) :=
sorry

end NUMINAMATH_GPT_exists_k_for_blocks_of_2022_l2093_209395


namespace NUMINAMATH_GPT_susan_backward_spaces_l2093_209391

variable (spaces_to_win total_spaces : ℕ)
variables (first_turn second_turn_forward second_turn_back third_turn : ℕ)

theorem susan_backward_spaces :
  ∀ (total_spaces first_turn second_turn_forward second_turn_back third_turn win_left : ℕ),
  total_spaces = 48 →
  first_turn = 8 →
  second_turn_forward = 2 →
  third_turn = 6 →
  win_left = 37 →
  first_turn + second_turn_forward + third_turn - second_turn_back + win_left = total_spaces →
  second_turn_back = 6 :=
by
  intros total_spaces first_turn second_turn_forward second_turn_back third_turn win_left
  intros h_total h_first h_second_forward h_third h_win h_eq
  rw [h_total, h_first, h_second_forward, h_third, h_win] at h_eq
  sorry

end NUMINAMATH_GPT_susan_backward_spaces_l2093_209391


namespace NUMINAMATH_GPT_cubic_roots_solution_sum_l2093_209355

theorem cubic_roots_solution_sum (u v w : ℝ) (h1 : (u - 2) * (u - 3) * (u - 4) = 1 / 2)
                                     (h2 : (v - 2) * (v - 3) * (v - 4) = 1 / 2)
                                     (h3 : (w - 2) * (w - 3) * (w - 4) = 1 / 2)
                                     (distinct_roots : u ≠ v ∧ v ≠ w ∧ u ≠ w) :
  u^3 + v^3 + w^3 = -42 :=
sorry

end NUMINAMATH_GPT_cubic_roots_solution_sum_l2093_209355


namespace NUMINAMATH_GPT_sugar_water_inequality_acute_triangle_inequality_l2093_209396

-- Part 1: Proving the inequality \(\frac{a}{b} < \frac{a+m}{b+m}\)
theorem sugar_water_inequality (a b m : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m) : 
  a / b < (a + m) / (b + m) :=
by
  sorry

-- Part 2: Proving the inequality in an acute triangle \(\triangle ABC\)
theorem acute_triangle_inequality (A B C : ℝ) (hA : A < B + C) (hB : B < C + A) (hC : C < A + B) : 
  (A / (B + C)) + (B / (C + A)) + (C / (A + B)) < 2 :=
by
  sorry

end NUMINAMATH_GPT_sugar_water_inequality_acute_triangle_inequality_l2093_209396


namespace NUMINAMATH_GPT_veranda_area_l2093_209376

/-- The width of the veranda on all sides of the room. -/
def width_of_veranda : ℝ := 2

/-- The length of the room. -/
def length_of_room : ℝ := 21

/-- The width of the room. -/
def width_of_room : ℝ := 12

/-- The area of the veranda given the conditions. -/
theorem veranda_area (length_of_room width_of_room width_of_veranda : ℝ) :
  (length_of_room + 2 * width_of_veranda) * (width_of_room + 2 * width_of_veranda) - length_of_room * width_of_room = 148 :=
by
  sorry

end NUMINAMATH_GPT_veranda_area_l2093_209376


namespace NUMINAMATH_GPT_yan_distance_ratio_l2093_209308

-- Define conditions
variable (x z w: ℝ)  -- x: distance from Yan to his home, z: distance from Yan to the school, w: Yan's walking speed
variable (h1: z / w = x / w + (x + z) / (5 * w))  -- Both choices require the same amount of time

-- The ratio of Yan's distance from his home to his distance from the school is 2/3
theorem yan_distance_ratio :
    x / z = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_yan_distance_ratio_l2093_209308


namespace NUMINAMATH_GPT_count_C_sets_l2093_209359

-- Definitions of sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 2}

-- The predicate that a set C satisfies B ∪ C = A
def satisfies_condition (C : Set ℕ) : Prop := B ∪ C = A

-- The claim that there are exactly 4 such sets C
theorem count_C_sets : 
  ∃ (C1 C2 C3 C4 : Set ℕ), 
    (satisfies_condition C1 ∧ satisfies_condition C2 ∧ satisfies_condition C3 ∧ satisfies_condition C4) 
    ∧ 
    (∀ C', satisfies_condition C' → C' = C1 ∨ C' = C2 ∨ C' = C3 ∨ C' = C4)
    ∧ 
    (C1 ≠ C2 ∧ C1 ≠ C3 ∧ C1 ≠ C4 ∧ C2 ≠ C3 ∧ C2 ≠ C4 ∧ C3 ≠ C4) := 
sorry

end NUMINAMATH_GPT_count_C_sets_l2093_209359


namespace NUMINAMATH_GPT_quadratic_max_value_4_at_2_l2093_209331

theorem quadratic_max_value_4_at_2 (a b c : ℝ) (m : ℝ)
  (h1 : ∀ x : ℝ, x ≠ 2 → (a * 2^2 + b * 2 + c) = 4)
  (h2 : a * 0^2 + b * 0 + c = -20)
  (h3 : a * 5^2 + b * 5 + c = m) :
  m = -50 :=
sorry

end NUMINAMATH_GPT_quadratic_max_value_4_at_2_l2093_209331


namespace NUMINAMATH_GPT_percentage_increase_l2093_209343

variable {α : Type} [LinearOrderedField α]

theorem percentage_increase (x y : α) (h : x = 0.5 * y) : y = x + x :=
by
  -- The steps of the proof are omitted and 'sorry' is used to skip actual proof.
  sorry

end NUMINAMATH_GPT_percentage_increase_l2093_209343


namespace NUMINAMATH_GPT_not_age_of_child_l2093_209367

noncomputable def sum_from_1_to_n (n : ℕ) := n * (n + 1) / 2

theorem not_age_of_child (N : ℕ) (S : Finset ℕ) (a b : ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11} ∧
  N = 1100 * a + 11 * b ∧
  a ≠ b ∧
  N ≥ 1000 ∧ N < 10000 ∧
  ((S.sum id) = N) ∧
  (∀ age ∈ S, N % age = 0) →
  10 ∉ S := 
by
  sorry

end NUMINAMATH_GPT_not_age_of_child_l2093_209367


namespace NUMINAMATH_GPT_sum_of_natural_numbers_l2093_209370

theorem sum_of_natural_numbers (n : ℕ) (h : n * (n + 1) = 812) : n = 28 := by
  sorry

end NUMINAMATH_GPT_sum_of_natural_numbers_l2093_209370


namespace NUMINAMATH_GPT_emily_sixth_quiz_score_l2093_209365

-- Define the scores Emily has received
def scores : List ℕ := [92, 96, 87, 89, 100]

-- Define the number of quizzes
def num_quizzes : ℕ := 6

-- Define the desired average score
def desired_average : ℕ := 94

-- The theorem to prove the score Emily needs on her sixth quiz to achieve the desired average
theorem emily_sixth_quiz_score : ∃ (x : ℕ), List.sum scores + x = desired_average * num_quizzes := by
  sorry

end NUMINAMATH_GPT_emily_sixth_quiz_score_l2093_209365


namespace NUMINAMATH_GPT_calculate_volume_from_measurements_l2093_209314

variables (r h : ℝ) (P : ℝ × ℝ)

noncomputable def volume_truncated_cylinder (area_base : ℝ) (height_segment : ℝ) : ℝ :=
  area_base * height_segment

theorem calculate_volume_from_measurements
    (radius : ℝ) (height : ℝ)
    (area_base : ℝ := π * radius^2)
    (P : ℝ × ℝ)  -- intersection point on the axis
    (height_segment : ℝ) : 
    volume_truncated_cylinder area_base height_segment = area_base * height_segment :=
by
  -- The proof would involve demonstrating the relationship mathematically
  sorry

end NUMINAMATH_GPT_calculate_volume_from_measurements_l2093_209314


namespace NUMINAMATH_GPT_kim_candy_bars_saved_l2093_209323

theorem kim_candy_bars_saved
  (n : ℕ)
  (c : ℕ)
  (w : ℕ)
  (total_bought : ℕ := n * c)
  (total_eaten : ℕ := n / w)
  (candy_bars_saved : ℕ := total_bought - total_eaten) :
  candy_bars_saved = 28 :=
by
  sorry

end NUMINAMATH_GPT_kim_candy_bars_saved_l2093_209323


namespace NUMINAMATH_GPT_shaded_region_area_l2093_209340

variables (a b : ℕ) 
variable (A : Type) 

def AD := 5
def CD := 2
def semi_major_axis := 6
def semi_minor_axis := 4

noncomputable def area_ellipse := Real.pi * semi_major_axis * semi_minor_axis
noncomputable def area_rectangle := AD * CD
noncomputable def area_shaded_region := area_ellipse - area_rectangle

theorem shaded_region_area : area_shaded_region = 24 * Real.pi - 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_shaded_region_area_l2093_209340


namespace NUMINAMATH_GPT_smallest_hot_dog_packages_l2093_209346

theorem smallest_hot_dog_packages (d : ℕ) (b : ℕ) (hd : d = 10) (hb : b = 15) :
  ∃ n : ℕ, n * d = m * b ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_hot_dog_packages_l2093_209346


namespace NUMINAMATH_GPT_cost_price_is_50_l2093_209307

-- Define the conditions
def selling_price : ℝ := 80
def profit_rate : ℝ := 0.6

-- The cost price should be proven to be 50
def cost_price (C : ℝ) : Prop :=
  selling_price = C + (C * profit_rate)

theorem cost_price_is_50 : ∃ C : ℝ, cost_price C ∧ C = 50 := by
  sorry

end NUMINAMATH_GPT_cost_price_is_50_l2093_209307


namespace NUMINAMATH_GPT_combined_height_of_trees_is_correct_l2093_209392

noncomputable def original_height_of_trees 
  (h1_current : ℝ) (h1_growth_rate : ℝ)
  (h2_current : ℝ) (h2_growth_rate : ℝ)
  (h3_current : ℝ) (h3_growth_rate : ℝ)
  (conversion_rate : ℝ) : ℝ :=
  let h1 := h1_current / (1 + h1_growth_rate)
  let h2 := h2_current / (1 + h2_growth_rate)
  let h3 := h3_current / (1 + h3_growth_rate)
  (h1 + h2 + h3) / conversion_rate

theorem combined_height_of_trees_is_correct :
  original_height_of_trees 240 0.70 300 0.50 180 0.60 12 = 37.81 :=
by
  sorry

end NUMINAMATH_GPT_combined_height_of_trees_is_correct_l2093_209392


namespace NUMINAMATH_GPT_positive_integer_solutions_l2093_209366

theorem positive_integer_solutions
  (m n k : ℕ)
  (hm : 0 < m) (hn : 0 < n) (hk : 0 < k) :
  3 * m + 4 * n = 5 * k ↔ (m = 1 ∧ n = 2 ∧ k = 2) := 
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_l2093_209366


namespace NUMINAMATH_GPT_inequality_solution_l2093_209302

theorem inequality_solution (x : ℝ) :
  (-4 ≤ x ∧ x < -3 / 2) ↔ (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2093_209302


namespace NUMINAMATH_GPT_equation_has_no_solution_l2093_209321

theorem equation_has_no_solution (k : ℝ) : ¬ (∃ x : ℝ , (x ≠ 3 ∧ x ≠ 4) ∧ (x - 1) / (x - 3) = (x - k) / (x - 4)) ↔ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_equation_has_no_solution_l2093_209321


namespace NUMINAMATH_GPT_min_bottles_needed_l2093_209337

theorem min_bottles_needed (fluid_ounces_needed : ℝ) (bottle_size_ml : ℝ) (conversion_factor : ℝ) :
  fluid_ounces_needed = 60 ∧ bottle_size_ml = 250 ∧ conversion_factor = 33.8 →
  ∃ (n : ℕ), n = 8 ∧ (fluid_ounces_needed / conversion_factor * 1000 / bottle_size_ml) <= ↑n :=
by
  sorry

end NUMINAMATH_GPT_min_bottles_needed_l2093_209337


namespace NUMINAMATH_GPT_charlie_acorns_l2093_209347

theorem charlie_acorns (x y : ℕ) (hc hs : ℕ)
  (h5 : x = 5 * hc)
  (h7 : y = 7 * hs)
  (total : x + y = 145)
  (holes : hs = hc - 3) :
  x = 70 :=
by
  sorry

end NUMINAMATH_GPT_charlie_acorns_l2093_209347


namespace NUMINAMATH_GPT_sum_of_possible_values_l2093_209379

theorem sum_of_possible_values (x : ℝ) (h : (x + 3) * (x - 4) = 24) : 
  ∃ x1 x2 : ℝ, (x1 + 3) * (x1 - 4) = 24 ∧ (x2 + 3) * (x2 - 4) = 24 ∧ x1 + x2 = 1 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_l2093_209379


namespace NUMINAMATH_GPT_Q1_no_such_a_b_Q2_no_such_a_b_c_l2093_209311

theorem Q1_no_such_a_b :
  ∀ (a b : ℕ), (0 < a) ∧ (0 < b) → ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, k^2 = 2^n * a + 5^n * b) := sorry

theorem Q2_no_such_a_b_c :
  ∀ (a b c : ℕ), (0 < a) ∧ (0 < b) ∧ (0 < c) → ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, k^2 = 2^n * a + 5^n * b + c) := sorry

end NUMINAMATH_GPT_Q1_no_such_a_b_Q2_no_such_a_b_c_l2093_209311


namespace NUMINAMATH_GPT_fraction_cows_sold_is_one_fourth_l2093_209388

def num_cows : ℕ := 184
def num_dogs (C : ℕ) : ℕ := C / 2
def remaining_animals : ℕ := 161
def fraction_dogs_sold : ℚ := 3 / 4
def fraction_cows_sold (C remaining_cows : ℕ) : ℚ := (C - remaining_cows) / C

theorem fraction_cows_sold_is_one_fourth :
  ∀ (C remaining_dogs remaining_cows: ℕ),
    C = 184 →
    remaining_animals = 161 →
    remaining_dogs = (1 - fraction_dogs_sold) * num_dogs C →
    remaining_cows = remaining_animals - remaining_dogs →
    fraction_cows_sold C remaining_cows = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_fraction_cows_sold_is_one_fourth_l2093_209388


namespace NUMINAMATH_GPT_largest_negative_integer_is_neg_one_l2093_209378

def is_negative_integer (n : Int) : Prop := n < 0

def is_largest_negative_integer (n : Int) : Prop := 
  is_negative_integer n ∧ ∀ m : Int, is_negative_integer m → m ≤ n

theorem largest_negative_integer_is_neg_one : 
  is_largest_negative_integer (-1) := by
  sorry

end NUMINAMATH_GPT_largest_negative_integer_is_neg_one_l2093_209378


namespace NUMINAMATH_GPT_find_asymptote_slope_l2093_209373

theorem find_asymptote_slope (x y : ℝ) (h : (y^2) / 9 - (x^2) / 4 = 1) : y = 3 / 2 * x :=
sorry

end NUMINAMATH_GPT_find_asymptote_slope_l2093_209373


namespace NUMINAMATH_GPT_prove_problem_statement_l2093_209364

noncomputable def problem_statement : Prop :=
  let E := (0, 0)
  let F := (2, 4)
  let G := (6, 2)
  let H := (7, 0)
  let line_through_E x y := y = -2 * x + 14
  let intersection_x := 37 / 8
  let intersection_y := 19 / 4
  let intersection_point := (intersection_x, intersection_y)
  let u := 37
  let v := 8
  let w := 19
  let z := 4
  u + v + w + z = 68

theorem prove_problem_statement : problem_statement :=
  sorry

end NUMINAMATH_GPT_prove_problem_statement_l2093_209364


namespace NUMINAMATH_GPT_initial_pigs_l2093_209389

theorem initial_pigs (x : ℕ) (h1 : x + 22 = 86) : x = 64 :=
by
  sorry

end NUMINAMATH_GPT_initial_pigs_l2093_209389


namespace NUMINAMATH_GPT_candy_cost_l2093_209341

theorem candy_cost
    (grape_candies : ℕ)
    (cherry_candies : ℕ)
    (apple_candies : ℕ)
    (total_cost : ℝ)
    (total_candies : ℕ)
    (cost_per_candy : ℝ)
    (h1 : grape_candies = 24)
    (h2 : grape_candies = 3 * cherry_candies)
    (h3 : apple_candies = 2 * grape_candies)
    (h4 : total_cost = 200)
    (h5 : total_candies = cherry_candies + grape_candies + apple_candies)
    (h6 : cost_per_candy = total_cost / total_candies) :
    cost_per_candy = 2.50 :=
by
    sorry

end NUMINAMATH_GPT_candy_cost_l2093_209341


namespace NUMINAMATH_GPT_lesser_of_two_numbers_l2093_209351

theorem lesser_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
by
  sorry

end NUMINAMATH_GPT_lesser_of_two_numbers_l2093_209351


namespace NUMINAMATH_GPT_correct_option_l2093_209300

theorem correct_option : (∃ x, x = -3 ∧ x^3 = -27) :=
by {
  -- Given conditions
  let x := -3
  use x
  constructor
  . rfl
  . norm_num
}

end NUMINAMATH_GPT_correct_option_l2093_209300


namespace NUMINAMATH_GPT_square_side_length_l2093_209332

variables (s : ℝ) (π : ℝ)
  
theorem square_side_length (h : 4 * s = π * s^2 / 2) : s = 8 / π :=
by sorry

end NUMINAMATH_GPT_square_side_length_l2093_209332


namespace NUMINAMATH_GPT_triangles_area_possibilities_unique_l2093_209397

noncomputable def triangle_area_possibilities : ℕ :=
  -- Define lengths of segments on the first line
  let AB := 1
  let BC := 2
  let CD := 3
  -- Sum to get total lengths
  let AC := AB + BC -- 3
  let AD := AB + BC + CD -- 6
  -- Define length of the segment on the second line
  let EF := 2
  -- GH is a segment not parallel to the first two lines
  let GH := 1
  -- The number of unique possible triangle areas
  4

theorem triangles_area_possibilities_unique :
  triangle_area_possibilities = 4 := 
sorry

end NUMINAMATH_GPT_triangles_area_possibilities_unique_l2093_209397


namespace NUMINAMATH_GPT_cost_per_bag_l2093_209338

theorem cost_per_bag (total_friends: ℕ) (amount_paid_per_friend: ℕ) (total_bags: ℕ) 
  (h1 : total_friends = 3) (h2 : amount_paid_per_friend = 5) (h3 : total_bags = 5) 
  : total_friends * amount_paid_per_friend / total_bags = 3 := by
  sorry

end NUMINAMATH_GPT_cost_per_bag_l2093_209338


namespace NUMINAMATH_GPT_maurice_age_l2093_209390

theorem maurice_age (M : ℕ) 
  (h₁ : 48 = 4 * (M + 5)) : M = 7 := 
by
  sorry

end NUMINAMATH_GPT_maurice_age_l2093_209390


namespace NUMINAMATH_GPT_part_I_part_II_l2093_209322

noncomputable def f (x : ℝ) := |x - 2| - |2 * x + 1|

theorem part_I :
  { x : ℝ | f x ≤ 0 } = { x : ℝ | x ≤ -3 ∨ x ≥ (1 : ℝ) / 3 } :=
by
  sorry

theorem part_II :
  ∀ x : ℝ, f x - 2 * m^2 ≤ 4 * m :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l2093_209322


namespace NUMINAMATH_GPT_digits_partition_impossible_l2093_209357

theorem digits_partition_impossible : 
  ¬ ∃ (A B : Finset ℕ), 
    A.card = 4 ∧ B.card = 4 ∧ A ∪ B = {1, 2, 3, 4, 5, 7, 8, 9} ∧ A ∩ B = ∅ ∧ 
    A.sum id = B.sum id := 
by
  sorry

end NUMINAMATH_GPT_digits_partition_impossible_l2093_209357


namespace NUMINAMATH_GPT_scientist_born_on_saturday_l2093_209313

noncomputable def day_of_week := List String

noncomputable def calculate_day := 
  let days_in_regular_years := 113
  let days_in_leap_years := 2 * 37
  let total_days_back := days_in_regular_years + days_in_leap_years
  total_days_back % 7

theorem scientist_born_on_saturday :
  let anniversary_day := 4  -- 0=Sunday, 1=Monday, ..., 4=Thursday
  calculate_day = 5 → 
  let birth_day := (anniversary_day + 7 - calculate_day) % 7 
  birth_day = 6 := sorry

end NUMINAMATH_GPT_scientist_born_on_saturday_l2093_209313


namespace NUMINAMATH_GPT_no_real_root_for_3_in_g_l2093_209344

noncomputable def g (x c : ℝ) : ℝ := x^2 + 3 * x + c

theorem no_real_root_for_3_in_g (c : ℝ) :
  (21 - 4 * c) < 0 ↔ c > 21 / 4 := by
sorry

end NUMINAMATH_GPT_no_real_root_for_3_in_g_l2093_209344


namespace NUMINAMATH_GPT_smallest_number_l2093_209356

theorem smallest_number (x : ℕ) (h1 : 2 * x = third) (h2 : 4 * x = second) (h3 : 7 * x = fourth) (h4 : (x + second + third + fourth) / 4 = 77) :
  x = 22 :=
by sorry

end NUMINAMATH_GPT_smallest_number_l2093_209356


namespace NUMINAMATH_GPT_monotonic_criteria_l2093_209352

noncomputable def monotonic_interval (m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 4 → 
  (-2 * x₁^2 + m * x₁ + 1) ≤ (-2 * x₂^2 + m * x₂ + 1)

theorem monotonic_criteria (m : ℝ) : 
  (m ≤ -4 ∨ m ≥ 16) ↔ monotonic_interval m := 
sorry

end NUMINAMATH_GPT_monotonic_criteria_l2093_209352


namespace NUMINAMATH_GPT_find_velocity_of_current_l2093_209318

-- Define the conditions given in the problem
def rowing_speed_in_still_water : ℤ := 10
def distance_to_place : ℤ := 48
def total_travel_time : ℤ := 10

-- Define the primary goal, which is to find the velocity of the current given the conditions
theorem find_velocity_of_current (v : ℤ) 
  (h1 : rowing_speed_in_still_water = 10)
  (h2 : distance_to_place = 48)
  (h3 : total_travel_time = 10) 
  (h4 : rowing_speed_in_still_water * 2 + v * 0 = 
   rowing_speed_in_still_water - v) :
  v = 2 := 
sorry

end NUMINAMATH_GPT_find_velocity_of_current_l2093_209318


namespace NUMINAMATH_GPT_area_diff_circle_square_l2093_209363

theorem area_diff_circle_square (s r : ℝ) (A_square A_circle : ℝ) (d : ℝ) (pi : ℝ) 
  (h1 : d = 8) -- diagonal of the square
  (h2 : d = 2 * r) -- diameter of the circle is 8, so radius is 4
  (h3 : s^2 + s^2 = d^2) -- Pythagorean Theorem for the square
  (h4 : A_square = s^2) -- area of the square
  (h5 : A_circle = pi * r^2) -- area of the circle
  (h6 : pi = 3.14159) -- approximation for π
  : abs (A_circle - A_square) - 18.3 < 0.1 := sorry

end NUMINAMATH_GPT_area_diff_circle_square_l2093_209363


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2093_209393

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x = 2 ∧ y = -1) → (x + y - 1 = 0) ∧ ¬(∀ x y, x + y - 1 = 0 → (x = 2 ∧ y = -1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2093_209393


namespace NUMINAMATH_GPT_angelas_insects_l2093_209383

variable (DeanInsects : ℕ) (JacobInsects : ℕ) (AngelaInsects : ℕ)

theorem angelas_insects
  (h1 : DeanInsects = 30)
  (h2 : JacobInsects = 5 * DeanInsects)
  (h3 : AngelaInsects = JacobInsects / 2):
  AngelaInsects = 75 := 
by
  sorry

end NUMINAMATH_GPT_angelas_insects_l2093_209383


namespace NUMINAMATH_GPT_problem1_problem2_l2093_209374

-- First Problem
theorem problem1 : 
  Real.cos (Real.pi / 3) + Real.sin (Real.pi / 4) - Real.tan (Real.pi / 4) = (-1 + Real.sqrt 2) / 2 :=
by
  sorry

-- Second Problem
theorem problem2 : 
  6 * (Real.tan (Real.pi / 6))^2 - Real.sqrt 3 * Real.sin (Real.pi / 3) - 2 * Real.cos (Real.pi / 4) = 1 / 2 - Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2093_209374


namespace NUMINAMATH_GPT_option_D_correct_l2093_209317

theorem option_D_correct (a : ℝ) :
  3 * a ^ 2 - a ≠ 2 * a ∧
  a - (1 - 2 * a) ≠ a - 1 ∧
  -5 * (1 - a ^ 2) ≠ -5 - 5 * a ^ 2 ∧
  a ^ 3 + 7 * a ^ 3 - 5 * a ^ 3 = 3 * a ^ 3 :=
by
  sorry

end NUMINAMATH_GPT_option_D_correct_l2093_209317


namespace NUMINAMATH_GPT_total_gulbis_is_correct_l2093_209380

-- Definitions based on given conditions
def num_dureums : ℕ := 156
def num_gulbis_in_one_dureum : ℕ := 20

-- Definition of total gulbis calculated
def total_gulbis : ℕ := num_dureums * num_gulbis_in_one_dureum

-- Statement to prove
theorem total_gulbis_is_correct : total_gulbis = 3120 := by
  -- The actual proof would go here
  sorry

end NUMINAMATH_GPT_total_gulbis_is_correct_l2093_209380


namespace NUMINAMATH_GPT_sequence_sixth_term_l2093_209385

theorem sequence_sixth_term :
  ∃ (a : ℕ → ℕ),
    a 1 = 3 ∧
    a 5 = 43 ∧
    (∀ n, a (n + 1) = (1/4) * (a n + a (n + 2))) →
    a 6 = 129 :=
sorry

end NUMINAMATH_GPT_sequence_sixth_term_l2093_209385


namespace NUMINAMATH_GPT_moscow_probability_higher_l2093_209353

def total_combinations : ℕ := 64 * 63

def invalid_combinations_ural : ℕ := 8 * 7 + 8 * 7

def valid_combinations_moscow : ℕ := total_combinations

def valid_combinations_ural : ℕ := total_combinations - invalid_combinations_ural

def probability_moscow : ℚ := valid_combinations_moscow / total_combinations

def probability_ural : ℚ := valid_combinations_ural / total_combinations

theorem moscow_probability_higher :
  probability_moscow > probability_ural :=
by
  unfold probability_moscow probability_ural
  unfold valid_combinations_moscow valid_combinations_ural invalid_combinations_ural total_combinations
  sorry

end NUMINAMATH_GPT_moscow_probability_higher_l2093_209353


namespace NUMINAMATH_GPT_f_periodic_l2093_209333

noncomputable def f : ℝ → ℝ := sorry

variable (a : ℝ) (h_a : 0 < a)
variable (h_cond : ∀ x : ℝ, f (x + a) = 1 / 2 + sqrt (f x - (f x)^2))

theorem f_periodic : ∀ x : ℝ, f (x + 2 * a) = f x := sorry

end NUMINAMATH_GPT_f_periodic_l2093_209333


namespace NUMINAMATH_GPT_combinations_medical_team_l2093_209336

noncomputable def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem combinations_medical_team : 
  let maleDoctors := 6
  let femaleDoctors := 5
  let numWaysMale := num_combinations maleDoctors 2
  let numWaysFemale := num_combinations femaleDoctors 1
  numWaysMale * numWaysFemale = 75 :=
by
  let maleDoctors := 6
  let femaleDoctors := 5
  let numWaysMale := num_combinations maleDoctors 2
  let numWaysFemale := num_combinations femaleDoctors 1
  show numWaysMale * numWaysFemale = 75 
  sorry

end NUMINAMATH_GPT_combinations_medical_team_l2093_209336


namespace NUMINAMATH_GPT_simplify_expression_l2093_209381

theorem simplify_expression : 2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := 
by 
  sorry 

end NUMINAMATH_GPT_simplify_expression_l2093_209381


namespace NUMINAMATH_GPT_volume_ratio_remainder_520_l2093_209371

noncomputable def simplex_ratio_mod : Nat :=
  let m := 2 ^ 2015 - 2016
  let n := 2 ^ 2015
  (m + n) % 1000

theorem volume_ratio_remainder_520 :
  let m := 2 ^ 2015 - 2016
  let n := 2 ^ 2015
  (m + n) % 1000 = 520 :=
by 
  sorry

end NUMINAMATH_GPT_volume_ratio_remainder_520_l2093_209371


namespace NUMINAMATH_GPT_problem_solution_l2093_209312

theorem problem_solution:
  2019 ^ Real.log (Real.log 2019) - Real.log 2019 ^ Real.log 2019 = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2093_209312


namespace NUMINAMATH_GPT_additional_cost_per_kg_l2093_209345

theorem additional_cost_per_kg (l m : ℝ) 
  (h1 : 168 = 30 * l + 3 * m) 
  (h2 : 186 = 30 * l + 6 * m) 
  (h3 : 20 * l = 100) : 
  m = 6 := 
by
  sorry

end NUMINAMATH_GPT_additional_cost_per_kg_l2093_209345


namespace NUMINAMATH_GPT_percentage_of_x_l2093_209377

variable {x y : ℝ}
variable {P : ℝ}

theorem percentage_of_x (h1 : (P / 100) * x = (20 / 100) * y) (h2 : x / y = 2) : P = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_of_x_l2093_209377


namespace NUMINAMATH_GPT_polynomial_solution_l2093_209398
-- Import necessary library

-- Define the property to be checked
def polynomial_property (P : Real → Real) : Prop :=
  ∀ a b c : Real, 
    P (a + b - 2 * c) + P (b + c - 2 * a) + P (c + a - 2 * b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

-- The statement that needs to be proven
theorem polynomial_solution (a b : Real) : polynomial_property (λ x => a * x^2 + b * x) := 
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l2093_209398


namespace NUMINAMATH_GPT_domain_of_f_l2093_209326

noncomputable def f (x : ℝ) : ℝ := (x + 6) / Real.sqrt (x^2 - 5*x + 6)

theorem domain_of_f : 
  {x : ℝ | x^2 - 5*x + 6 > 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2093_209326


namespace NUMINAMATH_GPT_find_y1_l2093_209360

theorem find_y1
  (y1 y2 y3 : ℝ)
  (h1 : 0 ≤ y3)
  (h2 : y3 ≤ y2)
  (h3 : y2 ≤ y1)
  (h4 : y1 ≤ 1)
  (h5 : (1 - y1)^2 + 2 * (y1 - y2)^2 + 2 * (y2 - y3)^2 + y3^2 = 1 / 2) :
  y1 = 3 / 4 :=
sorry

end NUMINAMATH_GPT_find_y1_l2093_209360


namespace NUMINAMATH_GPT_population_percentage_l2093_209368

-- Definitions based on the given conditions
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Conditions from the problem statement
def part_population : ℕ := 23040
def total_population : ℕ := 25600

-- The theorem stating that the percentage is 90
theorem population_percentage : percentage part_population total_population = 90 :=
  by
    -- Proof steps would go here, we only need to state the theorem
    sorry

end NUMINAMATH_GPT_population_percentage_l2093_209368


namespace NUMINAMATH_GPT_total_dots_on_left_faces_l2093_209325

-- Define the number of dots on the faces A, B, C, and D
def d_A : ℕ := 3
def d_B : ℕ := 5
def d_C : ℕ := 6
def d_D : ℕ := 5

-- The statement we need to prove
theorem total_dots_on_left_faces : d_A + d_B + d_C + d_D = 19 := by
  sorry

end NUMINAMATH_GPT_total_dots_on_left_faces_l2093_209325


namespace NUMINAMATH_GPT_polynomial_irreducible_over_Z_iff_Q_l2093_209319

theorem polynomial_irreducible_over_Z_iff_Q (f : Polynomial ℤ) :
  Irreducible f ↔ Irreducible (f.map (Int.castRingHom ℚ)) :=
sorry

end NUMINAMATH_GPT_polynomial_irreducible_over_Z_iff_Q_l2093_209319


namespace NUMINAMATH_GPT_sequence_length_div_by_four_l2093_209362

theorem sequence_length_div_by_four (a : ℕ) (h0 : a = 11664) (H : ∀ n, a = (4 ^ n) * b → b ≠ 0 ∧ n ≤ 3) : 
  ∃ n, n + 1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_sequence_length_div_by_four_l2093_209362


namespace NUMINAMATH_GPT_ratio_of_apple_to_orange_cost_l2093_209369

-- Define the costs of fruits based on the given conditions.
def cost_per_kg_oranges : ℝ := 12
def cost_per_kg_apples : ℝ := 2

-- The theorem to prove.
theorem ratio_of_apple_to_orange_cost : cost_per_kg_apples / cost_per_kg_oranges = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_apple_to_orange_cost_l2093_209369


namespace NUMINAMATH_GPT_collinear_points_in_cube_l2093_209316

def collinear_groups_in_cube : Prop :=
  let vertices := 8
  let edge_midpoints := 12
  let face_centers := 6
  let center_point := 1
  let total_groups :=
    (vertices * (vertices - 1) / 2) + (face_centers * 1 / 2) + (edge_midpoints * 3 / 2)
  total_groups = 49

theorem collinear_points_in_cube : collinear_groups_in_cube :=
  by
    sorry

end NUMINAMATH_GPT_collinear_points_in_cube_l2093_209316


namespace NUMINAMATH_GPT_four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime_l2093_209301

theorem four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime
  (N : ℕ) (hN : N ≥ 2) :
  (∀ n : ℕ, n < N → ¬ ∃ k : ℕ, k^2 = 4 * n * (N - n) + 1) ↔ Nat.Prime (N^2 + 1) :=
by sorry

end NUMINAMATH_GPT_four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime_l2093_209301


namespace NUMINAMATH_GPT_quadratic_complete_square_l2093_209309

theorem quadratic_complete_square :
  ∃ d e : ℝ, (∀ x : ℝ, x^2 + 800 * x + 500 = (x + d)^2 + e) ∧
    (e / d = -398.75) :=
by
  use 400
  use -159500
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l2093_209309


namespace NUMINAMATH_GPT_unique_reversible_six_digit_number_exists_l2093_209349

theorem unique_reversible_six_digit_number_exists :
  ∃! (N : ℤ), 100000 ≤ N ∧ N < 1000000 ∧
  ∃ (f e d c b a : ℤ), 
  N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧ 
  9 * N = 100000 * f + 10000 * e + 1000 * d + 100 * c + 10 * b + a := 
sorry

end NUMINAMATH_GPT_unique_reversible_six_digit_number_exists_l2093_209349


namespace NUMINAMATH_GPT_not_on_graph_ln_l2093_209303

theorem not_on_graph_ln {a b : ℝ} (h : b = Real.log a) : ¬ (1 + b = Real.log (a + Real.exp 1)) :=
by
  sorry

end NUMINAMATH_GPT_not_on_graph_ln_l2093_209303


namespace NUMINAMATH_GPT_problem_equivalent_proof_l2093_209320

def sequence_row1 (n : ℕ) : ℤ := 2 * (-2)^(n - 1)
def sequence_row2 (n : ℕ) : ℤ := sequence_row1 n - 1
def sequence_row3 (n : ℕ) : ℤ := (-2)^n - sequence_row2 n

theorem problem_equivalent_proof :
  let a := sequence_row1 7
  let b := sequence_row2 7
  let c := sequence_row3 7
  a - b + c = -254 :=
by
  sorry

end NUMINAMATH_GPT_problem_equivalent_proof_l2093_209320


namespace NUMINAMATH_GPT_num_quarters_left_l2093_209328

-- Define initial amounts and costs
def initial_amount : ℝ := 40
def pizza_cost : ℝ := 2.75
def soda_cost : ℝ := 1.50
def jeans_cost : ℝ := 11.50
def quarter_value : ℝ := 0.25

-- Define the total amount spent
def total_spent : ℝ := pizza_cost + soda_cost + jeans_cost

-- Define the remaining amount
def remaining_amount : ℝ := initial_amount - total_spent

-- Prove the number of quarters left
theorem num_quarters_left : remaining_amount / quarter_value = 97 :=
by
  sorry

end NUMINAMATH_GPT_num_quarters_left_l2093_209328


namespace NUMINAMATH_GPT_denominator_of_speed_l2093_209382

theorem denominator_of_speed (h : 0.8 = 8 / d * 3600 / 1000) : d = 36 := 
by
  sorry

end NUMINAMATH_GPT_denominator_of_speed_l2093_209382


namespace NUMINAMATH_GPT_erdos_problem_l2093_209304

variable (X : Type) [Infinite X] (𝓗 : Set (Set X))
variable (h1 : ∀ (A : Set X) (hA : A.Finite), ∃ (H1 H2 : Set X) (hH1 : H1 ∈ 𝓗) (hH2 : H2 ∈ 𝓗), H1 ∩ H2 = ∅ ∧ H1 ∪ H2 = A)

theorem erdos_problem (k : ℕ) (hk : k > 0) : 
  ∃ (A : Set X) (ways : Finset (Set X × Set X)), A.Finite ∧ (∀ (p : Set X × Set X), p ∈ ways → p.1 ∈ 𝓗 ∧ p.2 ∈ 𝓗 ∧ p.1 ∩ p.2 = ∅ ∧ p.1 ∪ p.2 = A) ∧ ways.card ≥ k :=
by
  sorry

end NUMINAMATH_GPT_erdos_problem_l2093_209304


namespace NUMINAMATH_GPT_find_initial_number_of_girls_l2093_209329

theorem find_initial_number_of_girls (b g : ℕ) : 
  (b = 3 * (g - 12)) ∧ (4 * (b - 36) = g - 12) → g = 25 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_initial_number_of_girls_l2093_209329


namespace NUMINAMATH_GPT_cage_chicken_problem_l2093_209327

theorem cage_chicken_problem :
  (∃ x : ℕ, 6 ≤ x ∧ x ≤ 10 ∧ (4 * x + 1 = 5 * (x - 1))) ∧
  (∀ x : ℕ, 6 ≤ x ∧ x ≤ 10 → (4 * x + 1 ≥ 25 ∧ 4 * x + 1 ≤ 41)) :=
by
  sorry

end NUMINAMATH_GPT_cage_chicken_problem_l2093_209327


namespace NUMINAMATH_GPT_sum_of_remainders_l2093_209335

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : ((n % 4) + (n % 5) = 4) :=
sorry

end NUMINAMATH_GPT_sum_of_remainders_l2093_209335


namespace NUMINAMATH_GPT_items_priced_at_9_yuan_l2093_209384

theorem items_priced_at_9_yuan (equal_number_items : ℕ)
  (total_cost : ℕ)
  (price_8_yuan : ℕ)
  (price_9_yuan : ℕ)
  (price_8_yuan_count : ℕ)
  (price_9_yuan_count : ℕ) :
  equal_number_items * 2 = price_8_yuan_count + price_9_yuan_count ∧
  (price_8_yuan_count * price_8_yuan + price_9_yuan_count * price_9_yuan = total_cost) ∧
  (price_8_yuan = 8) ∧
  (price_9_yuan = 9) ∧
  (total_cost = 172) →
  price_9_yuan_count = 12 :=
by
  sorry

end NUMINAMATH_GPT_items_priced_at_9_yuan_l2093_209384


namespace NUMINAMATH_GPT_max_sum_of_digits_of_S_l2093_209315

def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def distinctDigits (n : ℕ) : Prop :=
  let digits := (n.digits 10).toFinset
  digits.card = (n.digits 10).length

def digitsRange (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → 1 ≤ d ∧ d ≤ 9

theorem max_sum_of_digits_of_S : ∃ a b S, 
  isThreeDigit a ∧ 
  isThreeDigit b ∧ 
  distinctDigits a ∧ 
  distinctDigits b ∧ 
  digitsRange a ∧ 
  digitsRange b ∧ 
  isThreeDigit S ∧ 
  S = a + b ∧ 
  (S.digits 10).sum = 12 :=
sorry

end NUMINAMATH_GPT_max_sum_of_digits_of_S_l2093_209315


namespace NUMINAMATH_GPT_triangle_is_right_angle_l2093_209342

theorem triangle_is_right_angle (A B C : ℝ) : 
  (A / B = 2 / 3) ∧ (A / C = 2 / 5) ∧ (A + B + C = 180) →
  (A = 36) ∧ (B = 54) ∧ (C = 90) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_triangle_is_right_angle_l2093_209342


namespace NUMINAMATH_GPT_hvac_cost_per_vent_l2093_209348

theorem hvac_cost_per_vent (cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h_cost : cost = 20000) (h_zones : zones = 2) (h_vents_per_zone : vents_per_zone = 5) :
  (cost / (zones * vents_per_zone) = 2000) :=
by
  sorry

end NUMINAMATH_GPT_hvac_cost_per_vent_l2093_209348


namespace NUMINAMATH_GPT_range_of_a_quadratic_root_conditions_l2093_209305

theorem range_of_a_quadratic_root_conditions (a : ℝ) :
  ((∃ x₁ x₂ : ℝ, x₁ > 2 ∧ x₂ < 2 ∧ (ax^2 - 2*(a+1)*x + a-1 = 0)) ↔ (0 < a ∧ a < 5)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_quadratic_root_conditions_l2093_209305


namespace NUMINAMATH_GPT_solve_fraction_l2093_209310

theorem solve_fraction (x : ℝ) (h1 : x + 2 = 0) (h2 : 2 * x - 4 ≠ 0) : x = -2 := 
by 
  sorry

end NUMINAMATH_GPT_solve_fraction_l2093_209310


namespace NUMINAMATH_GPT_min_cubes_needed_l2093_209394

def minimum_cubes_for_views (front_view side_view : ℕ) : ℕ :=
  4

theorem min_cubes_needed (front_view_cond side_view_cond : ℕ) :
  front_view_cond = 2 ∧ side_view_cond = 3 → minimum_cubes_for_views front_view_cond side_view_cond = 4 :=
by
  intro h
  cases h
  -- Proving the condition based on provided views
  sorry

end NUMINAMATH_GPT_min_cubes_needed_l2093_209394


namespace NUMINAMATH_GPT_original_number_is_14_l2093_209386

theorem original_number_is_14 (x : ℝ) (h : (2 * x + 2) / 3 = 10) : x = 14 := by
  sorry

end NUMINAMATH_GPT_original_number_is_14_l2093_209386


namespace NUMINAMATH_GPT_percent_calculation_l2093_209387

-- Given conditions
def part : ℝ := 120.5
def whole : ℝ := 80.75

-- Theorem statement
theorem percent_calculation : (part / whole) * 100 = 149.26 := 
sorry

end NUMINAMATH_GPT_percent_calculation_l2093_209387


namespace NUMINAMATH_GPT_range_of_square_of_difference_of_roots_l2093_209330

theorem range_of_square_of_difference_of_roots (a : ℝ) (h : (a - 1) * (a - 2) < 0) :
  ∃ (S : Set ℝ), S = { x | 0 < x ∧ x ≤ 1 } ∧ ∀ (x1 x2 : ℝ),
  x1 + x2 = 2 * a ∧ x1 * x2 = 2 * a^2 - 3 * a + 2 → (x1 - x2)^2 ∈ S :=
sorry

end NUMINAMATH_GPT_range_of_square_of_difference_of_roots_l2093_209330


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2093_209399

variable (p q : Prop)

theorem necessary_but_not_sufficient_condition (h : ¬p) : p ∨ q ↔ true :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2093_209399


namespace NUMINAMATH_GPT_updated_mean_l2093_209375

theorem updated_mean (n : ℕ) (observation_mean decrement : ℕ) 
  (h1 : n = 50) (h2 : observation_mean = 200) (h3 : decrement = 15) : 
  ((observation_mean * n - decrement * n) / n = 185) :=
by
  sorry

end NUMINAMATH_GPT_updated_mean_l2093_209375


namespace NUMINAMATH_GPT_isosceles_triangle_height_l2093_209361

theorem isosceles_triangle_height (l w h : ℝ) 
  (h1 : l * w = (1 / 2) * w * h) : h = 2 * l :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_height_l2093_209361


namespace NUMINAMATH_GPT_score_order_l2093_209354

-- Definitions that come from the problem conditions
variables (M Q S K : ℝ)
variables (hQK : Q = K) (hMK : M > K) (hSK : S < K)

-- The theorem to prove
theorem score_order (hQK : Q = K) (hMK : M > K) (hSK : S < K) : S < Q ∧ Q < M :=
by {
  sorry
}

end NUMINAMATH_GPT_score_order_l2093_209354


namespace NUMINAMATH_GPT_hoodies_ownership_l2093_209334

-- Step a): Defining conditions
variables (Fiona_casey_hoodies_total: ℕ) (Casey_difference: ℕ) (Alex_hoodies: ℕ)

-- Functions representing the constraints
def hoodies_owned_by_Fiona (F : ℕ) : Prop :=
  (F + (F + 2) + 3 = 15)

-- Step c): Prove the correct number of hoodies owned by each
theorem hoodies_ownership (F : ℕ) (H1 : hoodies_owned_by_Fiona F) : 
  F = 5 ∧ (F + 2 = 7) ∧ (3 = 3) :=
by {
  -- Skipping proof details
  sorry
}

end NUMINAMATH_GPT_hoodies_ownership_l2093_209334


namespace NUMINAMATH_GPT_trapezoid_LM_sqrt2_l2093_209324

theorem trapezoid_LM_sqrt2 (K L M N P Q : Type*)
  (KM : ℝ)
  (KN MQ LM MP : ℝ)
  (h_KM : KM = 1)
  (h_KN_MQ : KN = MQ)
  (h_LM_MP : LM = MP) 
  (h_KP_1 : KN = 1) 
  (h_MQ_1 : MQ = 1) :
  LM = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_LM_sqrt2_l2093_209324


namespace NUMINAMATH_GPT_chris_remaining_money_l2093_209339

variable (video_game_cost : ℝ)
variable (discount_rate : ℝ)
variable (candy_cost : ℝ)
variable (tax_rate : ℝ)
variable (shipping_fee : ℝ)
variable (hourly_rate : ℝ)
variable (hours_worked : ℝ)

noncomputable def remaining_money (video_game_cost discount_rate candy_cost tax_rate shipping_fee hourly_rate hours_worked : ℝ) : ℝ :=
  let discount := discount_rate * video_game_cost
  let discounted_price := video_game_cost - discount
  let total_video_game_cost := discounted_price + shipping_fee
  let video_tax := tax_rate * total_video_game_cost
  let candy_tax := tax_rate * candy_cost
  let total_cost := (total_video_game_cost + video_tax) + (candy_cost + candy_tax)
  let earnings := hourly_rate * hours_worked
  earnings - total_cost

theorem chris_remaining_money : remaining_money 60 0.15 5 0.10 3 8 9 = 7.1 :=
by
  sorry

end NUMINAMATH_GPT_chris_remaining_money_l2093_209339


namespace NUMINAMATH_GPT_probability_white_ball_from_first_urn_correct_l2093_209350

noncomputable def probability_white_ball_from_first_urn : ℝ :=
  let p_H1 : ℝ := 0.5
  let p_H2 : ℝ := 0.5
  let p_A_given_H1 : ℝ := 0.7
  let p_A_given_H2 : ℝ := 0.6
  let p_A : ℝ := p_H1 * p_A_given_H1 + p_H2 * p_A_given_H2
  p_H1 * p_A_given_H1 / p_A

theorem probability_white_ball_from_first_urn_correct :
  probability_white_ball_from_first_urn = 0.538 :=
sorry

end NUMINAMATH_GPT_probability_white_ball_from_first_urn_correct_l2093_209350


namespace NUMINAMATH_GPT_imo_1988_problem_29_l2093_209372

variable (d r : ℕ)
variable (h1 : d > 1)
variable (h2 : 1059 % d = r)
variable (h3 : 1417 % d = r)
variable (h4 : 2312 % d = r)

theorem imo_1988_problem_29 :
  d - r = 15 := by sorry

end NUMINAMATH_GPT_imo_1988_problem_29_l2093_209372


namespace NUMINAMATH_GPT_ball_draw_probability_red_is_one_ninth_l2093_209358

theorem ball_draw_probability_red_is_one_ninth :
  let A_red := 4
  let A_white := 2
  let B_red := 1
  let B_white := 5
  let P_red_A := A_red / (A_red + A_white)
  let P_red_B := B_red / (B_red + B_white)
  P_red_A * P_red_B = 1 / 9 := by
    -- Proof here
    sorry

end NUMINAMATH_GPT_ball_draw_probability_red_is_one_ninth_l2093_209358
