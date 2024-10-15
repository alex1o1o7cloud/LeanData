import Mathlib

namespace NUMINAMATH_GPT_min_value_theorem_l2195_219594

noncomputable def min_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) : ℝ :=
  1/a + 2/b

theorem min_value_theorem (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  min_value a b h₀ h₁ h₂ ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_theorem_l2195_219594


namespace NUMINAMATH_GPT_parallel_lines_implies_value_of_m_l2195_219558

theorem parallel_lines_implies_value_of_m :
  ∀ (m : ℝ), (∀ (x y : ℝ), 3 * x + 2 * y - 2 = 0) ∧ (∀ (x y : ℝ), (2 * m - 1) * x + m * y + 1 = 0) → 
  m = 2 := 
by
  sorry

end NUMINAMATH_GPT_parallel_lines_implies_value_of_m_l2195_219558


namespace NUMINAMATH_GPT_probability_walk_450_feet_or_less_l2195_219528

theorem probability_walk_450_feet_or_less 
  (gates : List ℕ) (initial_gate new_gate : ℕ) 
  (n : ℕ) (dist_between_adjacent_gates : ℕ) 
  (valid_gates : gates.length = n)
  (distance : dist_between_adjacent_gates = 90) :
  n = 15 → 
  (initial_gate ∈ gates ∧ new_gate ∈ gates) → 
  ∃ (m1 m2 : ℕ), m1 = 59 ∧ m2 = 105 ∧ gcd m1 m2 = 1 ∧ 
  (∃ probability : ℚ, probability = (59 / 105 : ℚ) ∧ 
  (∃ sum_m1_m2 : ℕ, sum_m1_m2 = m1 + m2 ∧ sum_m1_m2 = 164)) :=
by
  sorry

end NUMINAMATH_GPT_probability_walk_450_feet_or_less_l2195_219528


namespace NUMINAMATH_GPT_sum_of_consecutive_evens_l2195_219541

theorem sum_of_consecutive_evens (E1 E2 E3 E4 : ℕ) (h1 : E4 = 38) (h2 : E3 = E4 - 2) (h3 : E2 = E3 - 2) (h4 : E1 = E2 - 2) : 
  E1 + E2 + E3 + E4 = 140 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_evens_l2195_219541


namespace NUMINAMATH_GPT_total_students_l2195_219542

-- Definition of the problem conditions
def buses : ℕ := 18
def seats_per_bus : ℕ := 15
def empty_seats_per_bus : ℕ := 3

-- Formulating the mathematically equivalent proof problem
theorem total_students :
  (buses * (seats_per_bus - empty_seats_per_bus) = 216) :=
by
  sorry

end NUMINAMATH_GPT_total_students_l2195_219542


namespace NUMINAMATH_GPT_sara_dozen_quarters_l2195_219516

theorem sara_dozen_quarters (dollars : ℕ) (quarters_per_dollar : ℕ) (quarters_per_dozen : ℕ) 
  (h1 : dollars = 9) (h2 : quarters_per_dollar = 4) (h3 : quarters_per_dozen = 12) : 
  dollars * quarters_per_dollar / quarters_per_dozen = 3 := 
by 
  sorry

end NUMINAMATH_GPT_sara_dozen_quarters_l2195_219516


namespace NUMINAMATH_GPT_solve_equation_l2195_219587

theorem solve_equation (x : ℝ) : 4 * (x - 1) ^ 2 = 9 ↔ x = 5 / 2 ∨ x = -1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l2195_219587


namespace NUMINAMATH_GPT_time_distribution_l2195_219548

noncomputable def total_hours_at_work (hours_task1 day : ℕ) (hours_task2 day : ℕ) (work_days : ℕ) (reduce_per_week : ℕ) : ℕ :=
  (hours_task1 + hours_task2) * work_days

theorem time_distribution (h1 : 5 = 5) (h2 : 3 = 3) (days : 5 = 5) (reduction : 5 = 5) :
  total_hours_at_work 5 3 5 5 = 40 :=
by
  sorry

end NUMINAMATH_GPT_time_distribution_l2195_219548


namespace NUMINAMATH_GPT_find_f_zero_l2195_219532

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_zero (h : ∀ x : ℝ, x ≠ 0 → f (2 * x - 1) = (1 - x^2) / x^2) : f 0 = 3 :=
sorry

end NUMINAMATH_GPT_find_f_zero_l2195_219532


namespace NUMINAMATH_GPT_trapezium_other_side_length_l2195_219590

theorem trapezium_other_side_length :
  ∃ (x : ℝ), 1/2 * (18 + x) * 17 = 323 ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_other_side_length_l2195_219590


namespace NUMINAMATH_GPT_probability_of_drawing_3_black_and_2_white_l2195_219569

noncomputable def total_ways_to_draw_5_balls : ℕ := Nat.choose 27 5
noncomputable def ways_to_choose_3_black : ℕ := Nat.choose 10 3
noncomputable def ways_to_choose_2_white : ℕ := Nat.choose 12 2
noncomputable def favorable_outcomes : ℕ := ways_to_choose_3_black * ways_to_choose_2_white
noncomputable def desired_probability : ℚ := favorable_outcomes / total_ways_to_draw_5_balls

theorem probability_of_drawing_3_black_and_2_white :
  desired_probability = 132 / 1345 := by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_3_black_and_2_white_l2195_219569


namespace NUMINAMATH_GPT_carol_is_inviting_friends_l2195_219566

theorem carol_is_inviting_friends :
  ∀ (invitations_per_pack packs_needed friends_invited : ℕ), 
  invitations_per_pack = 2 → 
  packs_needed = 5 → 
  friends_invited = invitations_per_pack * packs_needed → 
  friends_invited = 10 :=
by
  intros invitations_per_pack packs_needed friends_invited h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_carol_is_inviting_friends_l2195_219566


namespace NUMINAMATH_GPT_thabo_total_books_l2195_219556

-- Definitions and conditions mapped from the problem
def H : ℕ := 35
def P_NF : ℕ := H + 20
def P_F : ℕ := 2 * P_NF
def total_books : ℕ := H + P_NF + P_F

-- The theorem proving the total number of books
theorem thabo_total_books : total_books = 200 := by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_thabo_total_books_l2195_219556


namespace NUMINAMATH_GPT_safe_zone_inequality_l2195_219518

theorem safe_zone_inequality (x : ℝ) (fuse_burn_rate : ℝ) (run_speed : ℝ) (safe_zone_dist : ℝ) (H1: fuse_burn_rate = 0.5) (H2: run_speed = 4) (H3: safe_zone_dist = 150) :
  run_speed * (x / fuse_burn_rate) ≥ safe_zone_dist :=
sorry

end NUMINAMATH_GPT_safe_zone_inequality_l2195_219518


namespace NUMINAMATH_GPT_expected_winnings_is_0_25_l2195_219565

def prob_heads : ℚ := 3 / 8
def prob_tails : ℚ := 1 / 4
def prob_edge  : ℚ := 1 / 8
def prob_disappear : ℚ := 1 / 4

def winnings_heads : ℚ := 2
def winnings_tails : ℚ := 5
def winnings_edge  : ℚ := -2
def winnings_disappear : ℚ := -6

def expected_winnings : ℚ := 
  prob_heads * winnings_heads +
  prob_tails * winnings_tails +
  prob_edge  * winnings_edge +
  prob_disappear * winnings_disappear

theorem expected_winnings_is_0_25 : expected_winnings = 0.25 := by
  sorry

end NUMINAMATH_GPT_expected_winnings_is_0_25_l2195_219565


namespace NUMINAMATH_GPT_volume_of_new_cube_is_2744_l2195_219561

-- Define the volume function for a cube given side length
def volume_of_cube (side : ℝ) : ℝ := side ^ 3

-- Given the original cube with a specific volume
def original_volume : ℝ := 343

-- Find the side length of the original cube by taking the cube root of the volume
def original_side_length := (original_volume : ℝ)^(1/3)

-- The side length of the new cube is twice the side length of the original cube
def new_side_length := 2 * original_side_length

-- The volume of the new cube should be calculated
def new_volume := volume_of_cube new_side_length

-- Theorem stating that the new volume is 2744 cubic feet
theorem volume_of_new_cube_is_2744 : new_volume = 2744 := sorry

end NUMINAMATH_GPT_volume_of_new_cube_is_2744_l2195_219561


namespace NUMINAMATH_GPT_compute_fraction_l2195_219583

theorem compute_fraction : (1922^2 - 1913^2) / (1930^2 - 1905^2) = (9 : ℚ) / 25 := by
  sorry

end NUMINAMATH_GPT_compute_fraction_l2195_219583


namespace NUMINAMATH_GPT_total_eggs_today_l2195_219527

def eggs_morning : ℕ := 816
def eggs_afternoon : ℕ := 523

theorem total_eggs_today : eggs_morning + eggs_afternoon = 1339 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_eggs_today_l2195_219527


namespace NUMINAMATH_GPT_third_podcast_length_correct_l2195_219596

def first_podcast_length : ℕ := 45
def fourth_podcast_length : ℕ := 60
def next_podcast_length : ℕ := 60
def total_drive_time : ℕ := 360

def second_podcast_length := 2 * first_podcast_length

def total_time_other_than_third := first_podcast_length + second_podcast_length + fourth_podcast_length + next_podcast_length

theorem third_podcast_length_correct :
  total_drive_time - total_time_other_than_third = 105 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_third_podcast_length_correct_l2195_219596


namespace NUMINAMATH_GPT_loaned_books_count_l2195_219533

variable (x : ℕ)

def initial_books : ℕ := 75
def percentage_returned : ℝ := 0.65
def end_books : ℕ := 54
def non_returned_books : ℕ := initial_books - end_books
def percentage_non_returned : ℝ := 1 - percentage_returned

theorem loaned_books_count :
  percentage_non_returned * (x:ℝ) = non_returned_books → x = 60 :=
by
  sorry

end NUMINAMATH_GPT_loaned_books_count_l2195_219533


namespace NUMINAMATH_GPT_trapezium_area_l2195_219525

theorem trapezium_area (a b h : ℝ) (ha : a = 24) (hb : b = 18) (hh : h = 15) : 
  1/2 * (a + b) * h = 315 ∧ h = 15 :=
by 
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_trapezium_area_l2195_219525


namespace NUMINAMATH_GPT_log_sum_correct_l2195_219553

noncomputable def log_sum : Prop :=
  let x := (3/2)
  let y := (5/3)
  (x + y) = (19/6)

theorem log_sum_correct : log_sum :=
by
  sorry

end NUMINAMATH_GPT_log_sum_correct_l2195_219553


namespace NUMINAMATH_GPT_pyramid_addition_totals_l2195_219551

theorem pyramid_addition_totals 
  (initial_faces : ℕ) (initial_edges : ℕ) (initial_vertices : ℕ)
  (first_pyramid_new_faces : ℕ) (first_pyramid_new_edges : ℕ) (first_pyramid_new_vertices : ℕ)
  (second_pyramid_new_faces : ℕ) (second_pyramid_new_edges : ℕ) (second_pyramid_new_vertices : ℕ)
  (cancelling_faces_first : ℕ) (cancelling_faces_second : ℕ) :
  initial_faces = 5 → 
  initial_edges = 9 → 
  initial_vertices = 6 → 
  first_pyramid_new_faces = 3 →
  first_pyramid_new_edges = 3 →
  first_pyramid_new_vertices = 1 →
  second_pyramid_new_faces = 4 →
  second_pyramid_new_edges = 4 →
  second_pyramid_new_vertices = 1 →
  cancelling_faces_first = 1 →
  cancelling_faces_second = 1 →
  initial_faces + first_pyramid_new_faces - cancelling_faces_first 
  + second_pyramid_new_faces - cancelling_faces_second 
  + initial_edges + first_pyramid_new_edges + second_pyramid_new_edges
  + initial_vertices + first_pyramid_new_vertices + second_pyramid_new_vertices 
  = 34 := by sorry

end NUMINAMATH_GPT_pyramid_addition_totals_l2195_219551


namespace NUMINAMATH_GPT_simplify_fraction_l2195_219560

theorem simplify_fraction : 
  ((2^12)^2 - (2^10)^2) / ((2^11)^2 - (2^9)^2) = 4 := 
by sorry

end NUMINAMATH_GPT_simplify_fraction_l2195_219560


namespace NUMINAMATH_GPT_sufficient_not_necessary_l2195_219568

theorem sufficient_not_necessary (a b : ℝ) (h : b > a ∧ a > 0) : (1 / a > 1 / b) :=
by {
  sorry -- the proof steps are intentionally omitted
}

end NUMINAMATH_GPT_sufficient_not_necessary_l2195_219568


namespace NUMINAMATH_GPT_jessies_initial_weight_l2195_219522

-- Definitions based on the conditions
def weight_lost : ℕ := 126
def current_weight : ℕ := 66

-- The statement to prove
theorem jessies_initial_weight :
  (weight_lost + current_weight = 192) :=
by 
  sorry

end NUMINAMATH_GPT_jessies_initial_weight_l2195_219522


namespace NUMINAMATH_GPT_ratio_eq_l2195_219580

variable (a b c d : ℚ)

theorem ratio_eq :
  (a / b = 5 / 2) →
  (c / d = 7 / 3) →
  (d / b = 5 / 4) →
  (a / c = 6 / 7) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_ratio_eq_l2195_219580


namespace NUMINAMATH_GPT_average_score_of_seniors_l2195_219591

theorem average_score_of_seniors
    (total_students : ℕ)
    (average_score_all : ℚ)
    (num_seniors num_non_seniors : ℕ)
    (mean_score_senior mean_score_non_senior : ℚ)
    (h1 : total_students = 120)
    (h2 : average_score_all = 84)
    (h3 : num_non_seniors = 2 * num_seniors)
    (h4 : mean_score_senior = 2 * mean_score_non_senior)
    (h5 : num_seniors + num_non_seniors = total_students)
    (h6 : num_seniors * mean_score_senior + num_non_seniors * mean_score_non_senior = total_students * average_score_all) :
  mean_score_senior = 126 :=
by
  sorry

end NUMINAMATH_GPT_average_score_of_seniors_l2195_219591


namespace NUMINAMATH_GPT_anya_can_obtain_any_composite_number_l2195_219581

theorem anya_can_obtain_any_composite_number (n : ℕ) (h : ∃ k, k > 1 ∧ k < n ∧ n % k = 0) : ∃ m ≥ 4, ∀ k, k > 1 → k < m → m % k = 0 → m = n :=
by
  sorry

end NUMINAMATH_GPT_anya_can_obtain_any_composite_number_l2195_219581


namespace NUMINAMATH_GPT_algebraic_expression_value_l2195_219537

theorem algebraic_expression_value (x y : ℝ) (h : x^2 - 4 * x - 1 = 0) : 
  (2 * x - 3) ^ 2 - (x + y) * (x - y) - y ^ 2 = 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_algebraic_expression_value_l2195_219537


namespace NUMINAMATH_GPT_original_cost_price_l2195_219597

theorem original_cost_price ( C S : ℝ )
  (h1 : S = 1.05 * C)
  (h2 : S - 3 = 1.10 * 0.95 * C)
  : C = 600 :=
sorry

end NUMINAMATH_GPT_original_cost_price_l2195_219597


namespace NUMINAMATH_GPT_condition_sufficient_but_not_necessary_l2195_219508

variable (a b : ℝ)

theorem condition_sufficient_but_not_necessary :
  (|a| < 1 ∧ |b| < 1) → (|1 - a * b| > |a - b|) ∧
  ((|1 - a * b| > |a - b|) → (|a| < 1 ∧ |b| < 1) ∨ (|a| ≥ 1 ∧ |b| ≥ 1)) :=
by
  sorry

end NUMINAMATH_GPT_condition_sufficient_but_not_necessary_l2195_219508


namespace NUMINAMATH_GPT_worksheets_already_graded_l2195_219576

theorem worksheets_already_graded {total_worksheets problems_per_worksheet problems_left_to_grade : ℕ} :
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left_to_grade = 16 →
  (total_worksheets - (problems_left_to_grade / problems_per_worksheet)) = 5 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_worksheets_already_graded_l2195_219576


namespace NUMINAMATH_GPT_number_of_classmates_late_l2195_219513

-- Definitions based on conditions from problem statement
def charlizeLate : ℕ := 20
def classmateLate : ℕ := charlizeLate + 10
def totalLateTime : ℕ := 140

-- The proof statement
theorem number_of_classmates_late (x : ℕ) (h1 : totalLateTime = charlizeLate + x * classmateLate) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_classmates_late_l2195_219513


namespace NUMINAMATH_GPT_weight_of_a_l2195_219534

-- Define conditions
def weight_of_b : ℕ := 750 -- weight of one liter of ghee packet of brand 'b' in grams
def ratio_a_to_b : ℕ × ℕ := (3, 2)
def total_volume_liters : ℕ := 4 -- total volume of the mixture in liters
def total_weight_grams : ℕ := 3360 -- total weight of the mixture in grams

-- Target proof statement
theorem weight_of_a (W_a : ℕ) 
  (h_ratio : (ratio_a_to_b.1 + ratio_a_to_b.2) = 5)
  (h_mix_vol_a : (ratio_a_to_b.1 * total_volume_liters) = 12)
  (h_mix_vol_b : (ratio_a_to_b.2 * total_volume_liters) = 8)
  (h_weight_eq : (ratio_a_to_b.1 * W_a * total_volume_liters + ratio_a_to_b.2 * weight_of_b * total_volume_liters) = total_weight_grams * 5) : 
  W_a = 900 :=
by {
  sorry
}

end NUMINAMATH_GPT_weight_of_a_l2195_219534


namespace NUMINAMATH_GPT_simplify_polynomial_l2195_219517

variable (r : ℝ)

theorem simplify_polynomial : (2 * r^2 + 5 * r - 7) - (r^2 + 9 * r - 3) = r^2 - 4 * r - 4 := by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l2195_219517


namespace NUMINAMATH_GPT_hyperbola_m_range_l2195_219523

-- Define the equation of the hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / (m + 2)) - (y^2 / (m - 1)) = 1

-- State the equivalent range problem
theorem hyperbola_m_range (m : ℝ) :
  is_hyperbola m ↔ (m < -2 ∨ m > 1) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_m_range_l2195_219523


namespace NUMINAMATH_GPT_polynomial_constant_l2195_219538

theorem polynomial_constant (P : ℝ → ℝ → ℝ) (h : ∀ x y : ℝ, P (x + y) (y - x) = P x y) : 
  ∃ c : ℝ, ∀ x y : ℝ, P x y = c := 
sorry

end NUMINAMATH_GPT_polynomial_constant_l2195_219538


namespace NUMINAMATH_GPT_line_intersects_circle_midpoint_trajectory_l2195_219519

-- Definitions based on conditions
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

def line_eq (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Statement of the problem
theorem line_intersects_circle :
  ∀ m : ℝ, ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
sorry

theorem midpoint_trajectory :
  ∀ (x y : ℝ), 
  (∃ (xa ya xb yb : ℝ), circle_eq xa ya ∧ line_eq m xa ya ∧ 
   circle_eq xb yb ∧ line_eq m xb yb ∧ (x, y) = ((xa + xb) / 2, (ya + yb) / 2)) ↔
   ( x - 1 / 2)^2 + (y - 1)^2 = 1 / 4 :=
sorry

end NUMINAMATH_GPT_line_intersects_circle_midpoint_trajectory_l2195_219519


namespace NUMINAMATH_GPT_binomial_7_4_l2195_219503

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_7_4 : binomial 7 4 = 35 := by
  sorry

end NUMINAMATH_GPT_binomial_7_4_l2195_219503


namespace NUMINAMATH_GPT_younger_person_age_l2195_219586

/-- Let E be the present age of the elder person and Y be the present age of the younger person.
Given the conditions :
1) E - Y = 20
2) E - 15 = 2 * (Y - 15)
Prove that Y = 35. -/
theorem younger_person_age (E Y : ℕ) 
  (h1 : E - Y = 20) 
  (h2 : E - 15 = 2 * (Y - 15)) : 
  Y = 35 :=
sorry

end NUMINAMATH_GPT_younger_person_age_l2195_219586


namespace NUMINAMATH_GPT_puppies_per_dog_l2195_219540

def dogs := 15
def puppies := 75

theorem puppies_per_dog : puppies / dogs = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_puppies_per_dog_l2195_219540


namespace NUMINAMATH_GPT_three_digit_with_five_is_divisible_by_five_l2195_219595

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_in_five (n : ℕ) : Prop := n % 10 = 5

def divisible_by_five (n : ℕ) : Prop := n % 5 = 0

theorem three_digit_with_five_is_divisible_by_five (M : ℕ) :
  is_three_digit M ∧ ends_in_five M → divisible_by_five M :=
by
  sorry

end NUMINAMATH_GPT_three_digit_with_five_is_divisible_by_five_l2195_219595


namespace NUMINAMATH_GPT_total_animal_legs_is_12_l2195_219543

-- Define the number of legs per dog and chicken
def legs_per_dog : Nat := 4
def legs_per_chicken : Nat := 2

-- Define the number of dogs and chickens Mrs. Hilt saw
def number_of_dogs : Nat := 2
def number_of_chickens : Nat := 2

-- Calculate the total number of legs seen
def total_legs_seen : Nat :=
  (number_of_dogs * legs_per_dog) + (number_of_chickens * legs_per_chicken)

-- The theorem to be proven
theorem total_animal_legs_is_12 : total_legs_seen = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_animal_legs_is_12_l2195_219543


namespace NUMINAMATH_GPT_sin_cos_identity_l2195_219536

theorem sin_cos_identity (θ : ℝ) (h : Real.tan (θ + (Real.pi / 4)) = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = -7/5 := 
by 
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l2195_219536


namespace NUMINAMATH_GPT_rank_identity_l2195_219589

theorem rank_identity (n p : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) 
  (h1: 2 ≤ n) (h2: 2 ≤ p) (h3: A^(p+1) = A) : 
  Matrix.rank A + Matrix.rank (1 - A^p) = n := 
  sorry

end NUMINAMATH_GPT_rank_identity_l2195_219589


namespace NUMINAMATH_GPT_no_function_satisfies_condition_l2195_219549

theorem no_function_satisfies_condition :
  ¬ ∃ f : ℤ → ℤ, ∀ x y : ℤ, f (x + f y) = f x - y :=
sorry

end NUMINAMATH_GPT_no_function_satisfies_condition_l2195_219549


namespace NUMINAMATH_GPT_quadratic_to_square_form_l2195_219577

theorem quadratic_to_square_form (x : ℝ) :
  (x^2 - 6*x + 7 = 0) ↔ ((x - 3)^2 = 2) :=
sorry

end NUMINAMATH_GPT_quadratic_to_square_form_l2195_219577


namespace NUMINAMATH_GPT_max_omega_l2195_219529

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x)

theorem max_omega :
  (∃ ω > 0, (∃ k : ℤ, (f ω (2 * π / 3) = 0) ∧ (ω = 3 / 2 * k)) ∧ (0 < ω * π / 14 ∧ ω * π / 14 ≤ π / 2)) →
  ∃ ω, ω = 6 :=
by
  sorry

end NUMINAMATH_GPT_max_omega_l2195_219529


namespace NUMINAMATH_GPT_smallest_number_of_students_l2195_219515

theorem smallest_number_of_students
    (g11 g10 g9 : Nat)
    (h_ratio1 : 4 * g9 = 3 * g11)
    (h_ratio2 : 6 * g10 = 5 * g11) :
  g11 + g10 + g9 = 31 :=
sorry

end NUMINAMATH_GPT_smallest_number_of_students_l2195_219515


namespace NUMINAMATH_GPT_bert_made_1_dollar_l2195_219567

def bert_earnings (selling_price tax_rate markup : ℝ) : ℝ :=
  selling_price - (tax_rate * selling_price) - (selling_price - markup)

theorem bert_made_1_dollar :
  bert_earnings 90 0.1 10 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_bert_made_1_dollar_l2195_219567


namespace NUMINAMATH_GPT_problem1_problem2_l2195_219511

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + 2 * (a - 2) * x + 4

theorem problem1 (a : ℝ) :
  (∀ x, f a x > 0) → 0 < a ∧ a < 4 :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x, -3 <= x ∧ x <= 1 → f a x > 0) → (-1/2 < a ∧ a < 4) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2195_219511


namespace NUMINAMATH_GPT_stadium_length_in_yards_l2195_219599

def length_in_feet := 183
def feet_per_yard := 3

theorem stadium_length_in_yards : length_in_feet / feet_per_yard = 61 := by
  sorry

end NUMINAMATH_GPT_stadium_length_in_yards_l2195_219599


namespace NUMINAMATH_GPT_smallest_positive_integer_l2195_219555

theorem smallest_positive_integer (x : ℕ) (hx_pos : x > 0) (h : x < 15) : x = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l2195_219555


namespace NUMINAMATH_GPT_count_positive_integers_l2195_219554

theorem count_positive_integers (n : ℕ) (m : ℕ) :
  (∀ (k : ℕ), 1 ≤ k ∧ k < 100 ∧ (∃ (n : ℕ), n = 2 * k + 1 ∧ n < 200) 
  ∧ (∃ (m : ℤ), m = k * (k + 1) ∧ m % 5 = 0)) → 
  ∃ (cnt : ℕ), cnt = 20 :=
by
  sorry

end NUMINAMATH_GPT_count_positive_integers_l2195_219554


namespace NUMINAMATH_GPT_avg_speed_l2195_219593

noncomputable def jane_total_distance : ℝ := 120
noncomputable def time_period_hours : ℝ := 7

theorem avg_speed :
  jane_total_distance / time_period_hours = (120 / 7 : ℝ):=
by
  sorry

end NUMINAMATH_GPT_avg_speed_l2195_219593


namespace NUMINAMATH_GPT_total_area_equals_total_frequency_l2195_219573

-- Definition of frequency and frequency distribution histogram
def frequency_distribution_histogram (frequencies : List ℕ) := ∀ i, (i < frequencies.length) → ℕ

-- Definition that the total area of the small rectangles is the sum of the frequencies
def total_area_of_rectangles (frequencies : List ℕ) : ℕ := frequencies.sum

-- Theorem stating the equivalence
theorem total_area_equals_total_frequency (frequencies : List ℕ) :
  total_area_of_rectangles frequencies = frequencies.sum := 
by
  sorry

end NUMINAMATH_GPT_total_area_equals_total_frequency_l2195_219573


namespace NUMINAMATH_GPT_calculation_l2195_219578

theorem calculation :
  5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 :=
  by
    sorry

end NUMINAMATH_GPT_calculation_l2195_219578


namespace NUMINAMATH_GPT_find_x_value_l2195_219564

theorem find_x_value (x : ℝ) (y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x - 4) : x = 7 / 2 := 
sorry

end NUMINAMATH_GPT_find_x_value_l2195_219564


namespace NUMINAMATH_GPT_intersection_A_B_union_A_B_range_of_a_l2195_219514

open Set

-- Definitions for the given sets
def Universal : Set ℝ := univ
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 7}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 6}

-- Propositions to prove
theorem intersection_A_B : 
  A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7} := 
  sorry

theorem union_A_B : 
  A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := 
  sorry

theorem range_of_a (a : ℝ) : 
  (A ∪ C a = C a) → (2 ≤ a ∧ a < 3) := 
  sorry

end NUMINAMATH_GPT_intersection_A_B_union_A_B_range_of_a_l2195_219514


namespace NUMINAMATH_GPT_geometric_sequence_a6_l2195_219592

theorem geometric_sequence_a6 (a : ℕ → ℝ) (a1 r : ℝ) (h1 : ∀ n, a n = a1 * r ^ (n - 1)) (h2 : (a 2) * (a 4) * (a 12) = 64) : a 6 = 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l2195_219592


namespace NUMINAMATH_GPT_total_paid_is_201_l2195_219550

def adult_ticket_price : ℕ := 8
def child_ticket_price : ℕ := 5
def total_tickets : ℕ := 33
def child_tickets : ℕ := 21
def adult_tickets : ℕ := total_tickets - child_tickets
def total_paid : ℕ := (child_tickets * child_ticket_price) + (adult_tickets * adult_ticket_price)

theorem total_paid_is_201 : total_paid = 201 :=
by
  sorry

end NUMINAMATH_GPT_total_paid_is_201_l2195_219550


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2195_219588

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, (0 < x ∧ x < 2) → (x < 2)) ∧ ¬(∀ x : ℝ, (0 < x ∧ x < 2) ↔ (x < 2)) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2195_219588


namespace NUMINAMATH_GPT_parabola_tangent_line_l2195_219547

noncomputable def verify_a_value (a : ℝ) : Prop :=
  ∃ x₀ y₀ : ℝ, (y₀ = a * x₀^2) ∧ (x₀ - y₀ - 1 = 0) ∧ (2 * a * x₀ = 1)

theorem parabola_tangent_line :
  verify_a_value (1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_parabola_tangent_line_l2195_219547


namespace NUMINAMATH_GPT_largest_n_factors_l2195_219559

theorem largest_n_factors (n : ℤ) :
  (∃ A B : ℤ, 3 * B + A = n ∧ A * B = 72) → n ≤ 217 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_n_factors_l2195_219559


namespace NUMINAMATH_GPT_math_problem_l2195_219531

variable (a b c : ℤ)

theorem math_problem
  (h₁ : 3 * a + 4 * b + 5 * c = 0)
  (h₂ : |a| = 1)
  (h₃ : |b| = 1)
  (h₄ : |c| = 1) :
  a * (b + c) = - (3 / 5) :=
sorry

end NUMINAMATH_GPT_math_problem_l2195_219531


namespace NUMINAMATH_GPT_Alyssa_weekly_allowance_l2195_219510

theorem Alyssa_weekly_allowance
  (A : ℝ)
  (h1 : A / 2 + 8 = 12) :
  A = 8 := 
sorry

end NUMINAMATH_GPT_Alyssa_weekly_allowance_l2195_219510


namespace NUMINAMATH_GPT_satisfies_conditions_l2195_219572

theorem satisfies_conditions : ∃ (n : ℤ), 0 ≤ n ∧ n < 31 ∧ -250 % 31 = n % 31 ∧ n = 29 :=
by
  sorry

end NUMINAMATH_GPT_satisfies_conditions_l2195_219572


namespace NUMINAMATH_GPT_no_intersection_tangent_graph_l2195_219557

theorem no_intersection_tangent_graph (k : ℝ) (m : ℤ) : 
  (∀ x: ℝ, x = (k * Real.pi) / 2 → (¬ 4 * k ≠ 4 * m + 1)) → 
  (-1 ≤ k ∧ k ≤ 1) →
  (k = 1 / 4 ∨ k = -3 / 4) :=
sorry

end NUMINAMATH_GPT_no_intersection_tangent_graph_l2195_219557


namespace NUMINAMATH_GPT_number_of_classmates_l2195_219571

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end NUMINAMATH_GPT_number_of_classmates_l2195_219571


namespace NUMINAMATH_GPT_remaining_wire_length_l2195_219526

theorem remaining_wire_length (total_length : ℝ) (fraction_cut : ℝ) (remaining_length : ℝ) (h1 : total_length = 3) (h2 : fraction_cut = 1 / 3) (h3 : remaining_length = 2) :
  total_length * (1 - fraction_cut) = remaining_length :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_remaining_wire_length_l2195_219526


namespace NUMINAMATH_GPT_total_missed_questions_l2195_219552

-- Definitions
def missed_by_you : ℕ := 36
def missed_by_friend : ℕ := 7
def missed_by_you_friends : ℕ := missed_by_you + missed_by_friend

-- Theorem
theorem total_missed_questions (h1 : missed_by_you = 5 * missed_by_friend) :
  missed_by_you_friends = 43 :=
by
  sorry

end NUMINAMATH_GPT_total_missed_questions_l2195_219552


namespace NUMINAMATH_GPT_mod_1237_17_l2195_219512

theorem mod_1237_17 : 1237 % 17 = 13 := by
  sorry

end NUMINAMATH_GPT_mod_1237_17_l2195_219512


namespace NUMINAMATH_GPT_algebra_geometry_probabilities_l2195_219506

theorem algebra_geometry_probabilities :
  let total := 5
  let algebra := 2
  let geometry := 3
  let prob_first_algebra := algebra / total
  let prob_second_geometry_after_algebra := geometry / (total - 1)
  let prob_both := prob_first_algebra * prob_second_geometry_after_algebra
  let total_after_first_algebra := total - 1
  let remaining_geometry := geometry
  prob_both = 3 / 10 ∧ remaining_geometry / total_after_first_algebra = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_algebra_geometry_probabilities_l2195_219506


namespace NUMINAMATH_GPT_total_games_is_24_l2195_219570

-- Definitions of conditions
def games_this_month : Nat := 9
def games_last_month : Nat := 8
def games_next_month : Nat := 7

-- Total games attended
def total_games_attended : Nat :=
  games_this_month + games_last_month + games_next_month

-- Problem statement
theorem total_games_is_24 : total_games_attended = 24 := by
  sorry

end NUMINAMATH_GPT_total_games_is_24_l2195_219570


namespace NUMINAMATH_GPT_sqrt_inequality_l2195_219520

theorem sqrt_inequality (x : ℝ) (h₁ : 3 / 2 ≤ x) (h₂ : x ≤ 5) : 
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := 
sorry

end NUMINAMATH_GPT_sqrt_inequality_l2195_219520


namespace NUMINAMATH_GPT_intersection_has_one_element_l2195_219504

noncomputable def A (a : ℝ) : Set ℝ := {1, a, 5}
noncomputable def B (a : ℝ) : Set ℝ := {2, a^2 + 1}

theorem intersection_has_one_element (a : ℝ) (h : ∃ x, A a ∩ B a = {x}) : a = 0 ∨ a = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_has_one_element_l2195_219504


namespace NUMINAMATH_GPT_number_of_cats_l2195_219546

theorem number_of_cats (total_animals : ℕ) (dogs : ℕ) (cats : ℕ) 
  (h1 : total_animals = 1212) 
  (h2 : dogs = 567) 
  (h3 : cats = total_animals - dogs) : 
  cats = 645 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_cats_l2195_219546


namespace NUMINAMATH_GPT_two_trains_meet_at_distance_l2195_219507

theorem two_trains_meet_at_distance 
  (D_slow D_fast : ℕ)  -- Distances traveled by the slower and faster trains
  (T : ℕ)  -- Time taken to meet
  (h0 : 16 * T = D_slow)  -- Distance formula for slower train
  (h1 : 21 * T = D_fast)  -- Distance formula for faster train
  (h2 : D_fast = D_slow + 60)  -- Faster train travels 60 km more than slower train
  : (D_slow + D_fast = 444) := sorry

end NUMINAMATH_GPT_two_trains_meet_at_distance_l2195_219507


namespace NUMINAMATH_GPT_prime_square_sub_one_divisible_by_24_l2195_219574

theorem prime_square_sub_one_divisible_by_24 (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.Prime p) : 24 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_GPT_prime_square_sub_one_divisible_by_24_l2195_219574


namespace NUMINAMATH_GPT_hawks_score_l2195_219563

theorem hawks_score (x y : ℕ) (h1 : x + y = 82) (h2 : x - y = 18) : y = 32 :=
sorry

end NUMINAMATH_GPT_hawks_score_l2195_219563


namespace NUMINAMATH_GPT_power_function_value_at_neg2_l2195_219584

theorem power_function_value_at_neg2 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : ∀ x : ℝ, f x = x^a)
  (h2 : f 2 = 1 / 4) 
  : f (-2) = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_power_function_value_at_neg2_l2195_219584


namespace NUMINAMATH_GPT_pairs_of_polygons_with_angle_difference_l2195_219579

theorem pairs_of_polygons_with_angle_difference :
  ∃ (pairs : ℕ), pairs = 52 ∧ ∀ (n k : ℕ), n > k ∧ (360 / k - 360 / n = 1) :=
sorry

end NUMINAMATH_GPT_pairs_of_polygons_with_angle_difference_l2195_219579


namespace NUMINAMATH_GPT_regular_polygon_perimeter_l2195_219524

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_perimeter_l2195_219524


namespace NUMINAMATH_GPT_min_employees_needed_l2195_219509

theorem min_employees_needed (forest_jobs : ℕ) (marine_jobs : ℕ) (both_jobs : ℕ)
    (h1 : forest_jobs = 95) (h2 : marine_jobs = 80) (h3 : both_jobs = 35) :
    (forest_jobs - both_jobs) + (marine_jobs - both_jobs) + both_jobs = 140 :=
by
  sorry

end NUMINAMATH_GPT_min_employees_needed_l2195_219509


namespace NUMINAMATH_GPT_boris_climbs_needed_l2195_219539

-- Definitions
def elevation_hugo : ℕ := 10000
def shorter_difference : ℕ := 2500
def climbs_hugo : ℕ := 3

-- Derived Definitions
def elevation_boris : ℕ := elevation_hugo - shorter_difference
def total_climbed_hugo : ℕ := climbs_hugo * elevation_hugo

-- Theorem
theorem boris_climbs_needed : (total_climbed_hugo / elevation_boris) = 4 :=
by
  -- conditions and definitions are used here
  sorry

end NUMINAMATH_GPT_boris_climbs_needed_l2195_219539


namespace NUMINAMATH_GPT_complex_eq_l2195_219505

theorem complex_eq (a b : ℝ) (i : ℂ) (hi : i^2 = -1) (h : (a + 2 * i) / i = b + i) : a + b = 1 :=
sorry

end NUMINAMATH_GPT_complex_eq_l2195_219505


namespace NUMINAMATH_GPT_area_of_hexagon_l2195_219500

theorem area_of_hexagon (c d : ℝ) (a b : ℝ)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : a + b = d) : 
  (c^2 + d^2 = c^2 + a^2 + b^2 + 2*a*b) :=
by
  sorry

end NUMINAMATH_GPT_area_of_hexagon_l2195_219500


namespace NUMINAMATH_GPT_carly_dogs_total_l2195_219501

theorem carly_dogs_total (total_nails : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) (h2 : three_legged_dogs = 3) (h3 : nails_per_paw = 4) : total_dogs = 11 :=
by
  sorry

end NUMINAMATH_GPT_carly_dogs_total_l2195_219501


namespace NUMINAMATH_GPT_max_value_of_f_on_interval_l2195_219598

noncomputable def f (x : ℝ) : ℝ := 2^x + x * Real.log (1/4)

theorem max_value_of_f_on_interval :
  ∃ x ∈ Set.Icc (-2:ℝ) 2, f x = (1/4:ℝ) + 4 * Real.log 2 := 
sorry

end NUMINAMATH_GPT_max_value_of_f_on_interval_l2195_219598


namespace NUMINAMATH_GPT_earliest_time_100_degrees_l2195_219530

def temperature (t : ℝ) : ℝ := -t^2 + 15 * t + 40

theorem earliest_time_100_degrees :
  ∃ t : ℝ, temperature t = 100 ∧ (∀ t' : ℝ, temperature t' = 100 → t' ≥ t) :=
by
  sorry

end NUMINAMATH_GPT_earliest_time_100_degrees_l2195_219530


namespace NUMINAMATH_GPT_greatest_possible_red_points_l2195_219585

theorem greatest_possible_red_points (R B : ℕ) (h1 : R + B = 25)
    (h2 : ∀ r1 r2, r1 < R → r2 < R → r1 ≠ r2 → ∃ (n : ℕ), (∃ b1 : ℕ, b1 < B) ∧ ¬∃ b2 : ℕ, b2 < B) :
  R ≤ 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_possible_red_points_l2195_219585


namespace NUMINAMATH_GPT_line_tangent_to_parabola_l2195_219521

theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, y^2 = 16 * x ∧ 4 * x + 3 * y + k = 0 → ∀ y, y^2 + 12 * y + 4 * k = 0 → (12)^2 - 4 * 1 * 4 * k = 0) → 
  k = 9 :=
by
  sorry

end NUMINAMATH_GPT_line_tangent_to_parabola_l2195_219521


namespace NUMINAMATH_GPT_find_length_of_AB_l2195_219544

open Real

theorem find_length_of_AB (A B C : ℝ) 
    (h1 : tan A = 3 / 4) 
    (h2 : B = 6) 
    (h3 : C = π / 2) : sqrt (B^2 + ((3/4) * B)^2) = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_find_length_of_AB_l2195_219544


namespace NUMINAMATH_GPT_accurate_measurement_l2195_219562

-- Define the properties of Dr. Sharadek's tape
structure SharadekTape where
  startsWithHalfCM : Bool -- indicates if the tape starts with a half-centimeter bracket
  potentialError : ℝ -- potential measurement error

-- Define the conditions as an instance of the structure
noncomputable def drSharadekTape : SharadekTape :=
  { startsWithHalfCM := true,
    potentialError := 0.5 }

-- Define a segment with a known precise measurement
structure Segment where
  length : ℝ

noncomputable def AB (N : ℕ) : Segment :=
  { length := N + 0.5 }

-- The theorem stating the correct answer under the given conditions
theorem accurate_measurement (N : ℕ) : 
  ∃ AB : Segment, AB.length = N + 0.5 :=
by
  existsi AB N
  exact rfl

end NUMINAMATH_GPT_accurate_measurement_l2195_219562


namespace NUMINAMATH_GPT_simplify_expression_l2195_219582

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem simplify_expression :
  (1/2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b) = 6 • b - 3 • a :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2195_219582


namespace NUMINAMATH_GPT_find_x_when_water_added_l2195_219535

variable (m x : ℝ)

theorem find_x_when_water_added 
  (h1 : m > 25)
  (h2 : (m * m / 100) = ((m - 15) / 100) * (m + x)) :
  x = 15 * m / (m - 15) :=
sorry

end NUMINAMATH_GPT_find_x_when_water_added_l2195_219535


namespace NUMINAMATH_GPT_terry_nora_age_relation_l2195_219545

variable {N : ℕ} -- Nora's current age

theorem terry_nora_age_relation (h₁ : Terry_current_age = 30) (h₂ : Terry_future_age = 4 * N) : N = 10 :=
by
  --- additional assumptions
  have Terry_future_age_def : Terry_future_age = 30 + 10 := by sorry
  rw [Terry_future_age_def] at h₂
  linarith

end NUMINAMATH_GPT_terry_nora_age_relation_l2195_219545


namespace NUMINAMATH_GPT_total_fish_sold_l2195_219575

-- Define the conditions
def w1 : ℕ := 50
def w2 : ℕ := 3 * w1

-- Define the statement to prove
theorem total_fish_sold : w1 + w2 = 200 := by
  -- Insert the proof here 
  -- (proof omitted as per the instructions)
  sorry

end NUMINAMATH_GPT_total_fish_sold_l2195_219575


namespace NUMINAMATH_GPT_complex_equality_l2195_219502

theorem complex_equality (a b : ℝ) (h : (⟨0, 1⟩ : ℂ) ^ 3 = ⟨a, -b⟩) : a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_complex_equality_l2195_219502
