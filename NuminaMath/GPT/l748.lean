import Mathlib

namespace positive_integers_of_inequality_l748_74894

theorem positive_integers_of_inequality (x : ℕ) (h : 9 - 3 * x > 0) : x = 1 ∨ x = 2 :=
sorry

end positive_integers_of_inequality_l748_74894


namespace thief_speed_l748_74867

theorem thief_speed (v : ℝ) (hv : v > 0) : 
  let head_start_duration := (1/2 : ℝ)  -- 30 minutes, converted to hours
  let owner_speed := (75 : ℝ)  -- speed of owner in kmph
  let chase_duration := (2 : ℝ)  -- duration of the chase in hours
  let distance_by_owner := owner_speed * chase_duration  -- distance covered by the owner
  let total_distance_thief := head_start_duration * v + chase_duration * v  -- total distance covered by the thief
  distance_by_owner = 150 ->  -- given that owner covers 150 km
  total_distance_thief = 150  -- and so should the thief
  -> v = 60 := sorry

end thief_speed_l748_74867


namespace sum_of_circle_areas_l748_74898

theorem sum_of_circle_areas 
    (r s t : ℝ)
    (h1 : r + s = 6)
    (h2 : r + t = 8)
    (h3 : s + t = 10) : 
    (π * r^2 + π * s^2 + π * t^2) = 36 * π := 
by
    sorry

end sum_of_circle_areas_l748_74898


namespace weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900_l748_74897

noncomputable def Wa : ℕ := 
  let volume_a := (3/5) * 4
  let volume_b := (2/5) * 4
  let weight_b := 700
  let total_weight := 3280
  (total_weight - (weight_b * volume_b)) / volume_a

theorem weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900 :
  Wa = 900 := 
by
  sorry

end weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900_l748_74897


namespace problem_1_problem_2_problem_3_l748_74855

-- Definitions of assumptions and conditions.
structure Problem :=
  (boys : ℕ) -- number of boys
  (girls : ℕ) -- number of girls
  (subjects : ℕ) -- number of subjects
  (boyA_not_math : Prop) -- Boy A can't be a representative of the mathematics course
  (girlB_chinese : Prop) -- Girl B must be a representative of the Chinese language course

-- Problem 1: Calculate the number of ways satisfying condition (1)
theorem problem_1 (p : Problem) (h1 : p.girls < p.boys) :
  ∃ n : ℕ, n = 5520 := sorry

-- Problem 2: Calculate the number of ways satisfying condition (2)
theorem problem_2 (p : Problem) (h1 : p.boys ≥ 1) (h2 : p.boyA_not_math) :
  ∃ n : ℕ, n = 3360 := sorry

-- Problem 3: Calculate the number of ways satisfying condition (3)
theorem problem_3 (p : Problem) (h1 : p.boys ≥ 1) (h2 : p.boyA_not_math) (h3 : p.girlB_chinese) :
  ∃ n : ℕ, n = 360 := sorry

end problem_1_problem_2_problem_3_l748_74855


namespace product_three_consecutive_not_power_l748_74887

theorem product_three_consecutive_not_power (n k m : ℕ) (hn : n > 0) (hm : m ≥ 2) : 
  (n-1) * n * (n+1) ≠ k^m :=
by sorry

end product_three_consecutive_not_power_l748_74887


namespace prove_M_l748_74815

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {2, 3}
def M : Set ℕ := {x | x ∈ P ∧ x ∉ Q}

theorem prove_M :
  M = {1} :=
by
  sorry

end prove_M_l748_74815


namespace souvenir_prices_total_profit_l748_74835

variables (x y m n : ℝ)

-- Conditions for the first part
def conditions_part1 : Prop :=
  7 * x + 8 * y = 380 ∧
  10 * x + 6 * y = 380

-- Result for the first part
def result_part1 : Prop :=
  x = 20 ∧ y = 30

-- Conditions for the second part
def conditions_part2 : Prop :=
  m + n = 40 ∧
  20 * m + 30 * n = 900 

-- Result for the second part
def result_part2 : Prop :=
  30 * 5 + 10 * 7 = 220

theorem souvenir_prices (x y : ℝ) (h : conditions_part1 x y) : result_part1 x y :=
by { sorry }

theorem total_profit (m n : ℝ) (h : conditions_part2 m n) : result_part2 :=
by { sorry }

end souvenir_prices_total_profit_l748_74835


namespace value_of_expression_l748_74872

theorem value_of_expression (x y : ℝ) (h : x + 2 * y = 30) : (x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3) = 16 :=
by
  sorry

end value_of_expression_l748_74872


namespace horner_method_poly_at_neg2_l748_74883

-- Define the polynomial using the given conditions and Horner's method transformation
def polynomial : ℤ → ℤ := fun x => (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 3) * x + 2

-- State the theorem
theorem horner_method_poly_at_neg2 : polynomial (-2) = -40 := by
  sorry

end horner_method_poly_at_neg2_l748_74883


namespace joey_speed_return_l748_74833

/--
Joey the postman takes 1 hour to run a 5-mile-long route every day, delivering packages along the way.
On his return, he must climb a steep hill covering 3 miles and then navigate a rough, muddy terrain spanning 2 miles.
If the average speed of the entire round trip is 8 miles per hour, prove that the speed with which Joey returns along the path is 20 miles per hour.
-/
theorem joey_speed_return
  (dist_out : ℝ := 5)
  (time_out : ℝ := 1)
  (dist_hill : ℝ := 3)
  (dist_terrain : ℝ := 2)
  (avg_speed_round : ℝ := 8)
  (total_dist : ℝ := dist_out * 2)
  (total_time : ℝ := total_dist / avg_speed_round)
  (time_return : ℝ := total_time - time_out)
  (dist_return : ℝ := dist_hill + dist_terrain) :
  (dist_return / time_return = 20) := 
sorry

end joey_speed_return_l748_74833


namespace total_candies_is_36_l748_74805

-- Defining the conditions
def candies_per_day (day : String) : Nat :=
  if day = "Monday" ∨ day = "Wednesday" then 2 else 1

def total_candies_per_week : Nat :=
  (candies_per_day "Monday" + candies_per_day "Tuesday"
  + candies_per_day "Wednesday" + candies_per_day "Thursday"
  + candies_per_day "Friday" + candies_per_day "Saturday"
  + candies_per_day "Sunday")

def total_candies_in_weeks (weeks : Nat) : Nat :=
  weeks * total_candies_per_week

-- Stating the theorem
theorem total_candies_is_36 : total_candies_in_weeks 4 = 36 :=
  sorry

end total_candies_is_36_l748_74805


namespace work_completion_l748_74842

theorem work_completion (A : ℝ) (B : ℝ) (work_duration : ℝ) (total_days : ℝ) (B_days : ℝ) :
  B_days = 28 ∧ total_days = 8 ∧ (A * 2 + (A * 6 + B * 6) = work_duration) →
  A = 84 / 11 :=
by
  sorry

end work_completion_l748_74842


namespace fractional_difference_l748_74858

def recurring72 : ℚ := 72 / 99
def decimal72 : ℚ := 72 / 100

theorem fractional_difference : recurring72 - decimal72 = 2 / 275 := by
  sorry

end fractional_difference_l748_74858


namespace simplify_trig_expression_l748_74890

theorem simplify_trig_expression : 2 * Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 2 := 
sorry

end simplify_trig_expression_l748_74890


namespace range_of_a_l748_74856

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x^2 + 2 * a * x + 1) → -1 ≤ a ∧ a ≤ 1 :=
by
  intro h
  sorry

end range_of_a_l748_74856


namespace marcy_total_time_l748_74845

theorem marcy_total_time 
    (petting_time : ℝ)
    (fraction_combing : ℝ)
    (H1 : petting_time = 12)
    (H2 : fraction_combing = 1/3) :
    (petting_time + (fraction_combing * petting_time) = 16) :=
  sorry

end marcy_total_time_l748_74845


namespace circles_intersect_l748_74825

noncomputable def positional_relationship (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : String :=
  let d := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  if radius1 + radius2 > d ∧ d > abs (radius1 - radius2) then "Intersecting"
  else if radius1 + radius2 = d then "Externally tangent"
  else if abs (radius1 - radius2) = d then "Internally tangent"
  else "Separate"

theorem circles_intersect :
  positional_relationship (0, 1) (1, 2) 1 2 = "Intersecting" :=
by
  sorry

end circles_intersect_l748_74825


namespace scooter_gain_percent_l748_74838

def initial_cost : ℝ := 900
def first_repair_cost : ℝ := 150
def second_repair_cost : ℝ := 75
def third_repair_cost : ℝ := 225
def selling_price : ℝ := 1800

theorem scooter_gain_percent :
  let total_cost := initial_cost + first_repair_cost + second_repair_cost + third_repair_cost
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 33.33 :=
by
  sorry

end scooter_gain_percent_l748_74838


namespace evaluate_expression_l748_74862

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 :=
by
  sorry

end evaluate_expression_l748_74862


namespace problem_l748_74857

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * b * x + c

def f_prime (a b x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + 3 * b

theorem problem (a b c : ℝ) (h1 : f_prime a b 2 = 0) (h2 : f_prime a b 1 = -3) :
  a = -1 ∧ b = 0 ∧ (let f_min := f (-1) 0 c 2 
                   let f_max := 0 
                   f_max - f_min = 4) :=
by
  sorry

end problem_l748_74857


namespace difference_of_scores_correct_l748_74841

-- Define the parameters
def num_innings : ℕ := 46
def batting_avg : ℕ := 63
def highest_score : ℕ := 248
def reduced_avg : ℕ := 58
def excluded_innings : ℕ := num_innings - 2

-- Necessary calculations
def total_runs := batting_avg * num_innings
def reduced_total_runs := reduced_avg * excluded_innings
def sum_highest_lowest := total_runs - reduced_total_runs
def lowest_score := sum_highest_lowest - highest_score

-- The correct answer to prove
def expected_difference := highest_score - lowest_score
def correct_answer := 150

-- Define the proof problem
theorem difference_of_scores_correct :
  expected_difference = correct_answer := by
  sorry

end difference_of_scores_correct_l748_74841


namespace count_sequences_of_length_15_l748_74859

def countingValidSequences (n : ℕ) : ℕ := sorry

theorem count_sequences_of_length_15 :
  countingValidSequences 15 = 266 :=
  sorry

end count_sequences_of_length_15_l748_74859


namespace value_of_expression_l748_74851

theorem value_of_expression (r s : ℝ) (h₁ : 3 * r^2 - 5 * r - 7 = 0) (h₂ : 3 * s^2 - 5 * s - 7 = 0) : 
  (9 * r^2 - 9 * s^2) / (r - s) = 15 :=
sorry

end value_of_expression_l748_74851


namespace right_triangle_side_length_l748_74822

theorem right_triangle_side_length (area : ℝ) (side1 : ℝ) (side2 : ℝ) (h_area : area = 8) (h_side1 : side1 = Real.sqrt 10) (h_area_eq : area = 0.5 * side1 * side2) :
  side2 = 1.6 * Real.sqrt 10 :=
by 
  sorry

end right_triangle_side_length_l748_74822


namespace boxes_produced_by_machine_A_in_10_minutes_l748_74843

-- Define the variables and constants involved
variables {A : ℕ} -- number of boxes machine A produces in 10 minutes

-- Define the condition that machine B produces 4*A boxes in 10 minutes
def boxes_produced_by_machine_B_in_10_minutes := 4 * A

-- Define the combined production working together for 20 minutes
def combined_production_in_20_minutes := 10 * A

-- Statement to prove that machine A produces A boxes in 10 minutes
theorem boxes_produced_by_machine_A_in_10_minutes :
  ∀ (boxes_produced_by_machine_B_in_10_minutes : ℕ) (combined_production_in_20_minutes : ℕ),
    boxes_produced_by_machine_B_in_10_minutes = 4 * A →
    combined_production_in_20_minutes = 10 * A →
    A = A :=
by
  intros _ _ hB hC
  sorry

end boxes_produced_by_machine_A_in_10_minutes_l748_74843


namespace simplify_fraction_l748_74813

theorem simplify_fraction : 
  (2 * Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5 := 
by
  sorry

end simplify_fraction_l748_74813


namespace smallest_prime_divisor_of_sum_of_powers_l748_74844

theorem smallest_prime_divisor_of_sum_of_powers :
  let a := 5
  let b := 7
  let n := 23
  let m := 17
  Nat.minFac (a^n + b^m) = 2 := by
  sorry

end smallest_prime_divisor_of_sum_of_powers_l748_74844


namespace total_food_consumed_l748_74865

theorem total_food_consumed (n1 n2 f1 f2 : ℕ) (h1 : n1 = 4000) (h2 : n2 = n1 - 500) (h3 : f1 = 10) (h4 : f2 = f1 - 2) : 
    n1 * f1 + n2 * f2 = 68000 := by 
  sorry

end total_food_consumed_l748_74865


namespace magnitude_fourth_power_l748_74803

open Complex

noncomputable def complex_magnitude_example : ℂ := 4 + 3 * Real.sqrt 3 * Complex.I

theorem magnitude_fourth_power :
  ‖complex_magnitude_example ^ 4‖ = 1849 := by
  sorry

end magnitude_fourth_power_l748_74803


namespace median_to_longest_side_l748_74836

theorem median_to_longest_side
  (a b c : ℕ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26)
  (h4 : a^2 + b^2 = c^2) :
  ∃ m : ℕ, m = c / 2 ∧ m = 13 := 
by {
  sorry
}

end median_to_longest_side_l748_74836


namespace deer_families_stayed_l748_74830

-- Define the initial number of deer families
def initial_deer_families : ℕ := 79

-- Define the number of deer families that moved out
def moved_out_deer_families : ℕ := 34

-- The theorem stating how many deer families stayed
theorem deer_families_stayed : initial_deer_families - moved_out_deer_families = 45 :=
by
  -- Proof will be provided here
  sorry

end deer_families_stayed_l748_74830


namespace distance_and_ratio_correct_l748_74812

noncomputable def distance_and_ratio (a : ℝ) : ℝ × ℝ :=
  let dist : ℝ := a / Real.sqrt 3
  let ratio : ℝ := 1 / 2
  ⟨dist, ratio⟩

theorem distance_and_ratio_correct (a : ℝ) :
  distance_and_ratio a = (a / Real.sqrt 3, 1 / 2) := by
  -- Proof omitted
  sorry

end distance_and_ratio_correct_l748_74812


namespace inequality_solution_set_l748_74826

theorem inequality_solution_set :
  {x : ℝ | 3 * x + 9 > 0 ∧ 2 * x < 6} = {x : ℝ | -3 < x ∧ x < 3} := 
by
  sorry

end inequality_solution_set_l748_74826


namespace circumcircle_eq_l748_74861

-- Definitions and conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4
def point_P : (ℝ × ℝ) := (4, 2)
def is_tangent_point (x y : ℝ) : Prop := sorry -- You need a proper definition for tangency

theorem circumcircle_eq :
  ∃ (hA : is_tangent_point 0 2) (hB : ∃ x y, is_tangent_point x y),
  ∃ (x y : ℝ), (circle_eq 0 2 ∧ circle_eq x y) ∧ (x-2)^2 + (y-1)^2 = 5 :=
  sorry

end circumcircle_eq_l748_74861


namespace no_playful_two_digit_numbers_l748_74870

def is_playful (a b : ℕ) : Prop := 10 * a + b = a^3 + b^2

theorem no_playful_two_digit_numbers :
  (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → ¬ is_playful a b) :=
by {
  sorry
}

end no_playful_two_digit_numbers_l748_74870


namespace linear_regression_decrease_l748_74820

theorem linear_regression_decrease (x : ℝ) (y : ℝ) (h : y = 2 - 1.5 * x) : 
  y = 2 - 1.5 * (x + 1) -> (y - (2 - 1.5 * (x +1))) = -1.5 :=
by
  sorry

end linear_regression_decrease_l748_74820


namespace find_k_for_min_value_zero_l748_74879

theorem find_k_for_min_value_zero :
  (∃ k : ℝ, ∀ x y : ℝ, 9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 ≥ 0 ∧
                         ∃ x y : ℝ, 9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 = 0) →
  k = 3 / 2 :=
sorry

end find_k_for_min_value_zero_l748_74879


namespace find_third_side_l748_74824

theorem find_third_side
  (cubes : ℕ) (cube_volume : ℚ) (side1 side2 : ℚ)
  (fits : cubes = 24) (vol_cube : cube_volume = 27)
  (dim1 : side1 = 8) (dim2 : side2 = 9) :
  (side1 * side2 * (cube_volume * cubes) / (side1 * side2)) = 9 := by
  sorry

end find_third_side_l748_74824


namespace remainder_of_pencils_l748_74839

def number_of_pencils : ℕ := 13254839
def packages : ℕ := 7

theorem remainder_of_pencils :
  number_of_pencils % packages = 3 := by
  sorry

end remainder_of_pencils_l748_74839


namespace inequality_proof_l748_74884

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a < a - b :=
by
  sorry

end inequality_proof_l748_74884


namespace fraction_representation_correct_l748_74809

theorem fraction_representation_correct (h : ∀ (x y z w: ℕ), 9*x = y ∧ 47*z = w ∧ 2*47*5 = 235):
  (18: ℚ) / (9 * 47 * 5) = (2: ℚ) / 235 :=
by
  sorry

end fraction_representation_correct_l748_74809


namespace yellow_fraction_after_changes_l748_74889

theorem yellow_fraction_after_changes (y : ℕ) :
  let green_initial := (4 / 7 : ℚ) * y
  let yellow_initial := (3 / 7 : ℚ) * y
  let yellow_new := 3 * yellow_initial
  let green_new := green_initial + (1 / 2) * green_initial
  let total_new := green_new + yellow_new
  yellow_new / total_new = (3 / 5 : ℚ) :=
by
  sorry

end yellow_fraction_after_changes_l748_74889


namespace female_members_count_l748_74808

theorem female_members_count (M F : ℕ) (h1 : F = 2 * M) (h2 : F + M = 18) : F = 12 :=
by
  -- the proof will go here
  sorry

end female_members_count_l748_74808


namespace solve_N1N2_identity_l748_74874

theorem solve_N1N2_identity :
  (∃ N1 N2 : ℚ,
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 3 →
      (42 * x - 37) / (x^2 - 4 * x + 3) =
      N1 / (x - 1) + N2 / (x - 3)) ∧ 
      N1 * N2 = -445 / 4) :=
by
  sorry

end solve_N1N2_identity_l748_74874


namespace problem_solution_l748_74899

variable (U : Set Real) (a b : Real) (t : Real)
variable (A B : Set Real)

-- Conditions
def condition1 : U = Set.univ := sorry

def condition2 : ∀ x, a ≠ 0 → ax^2 + 2 * x + b > 0 ↔ x ≠ -1 / a := sorry

def condition3 : a > b := sorry

def condition4 : t = (a^2 + b^2) / (a - b) := sorry

def condition5 : ∀ m, (∀ x, |x + 1| - |x - 3| ≤ m^2 - 3 * m) → m ∈ B := sorry

-- To Prove
theorem problem_solution : A ∩ (Set.univ \ B) = {m : Real | 2 * Real.sqrt 2 ≤ m ∧ m < 4} := sorry

end problem_solution_l748_74899


namespace evaluate_expression_l748_74821

theorem evaluate_expression :
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = 137 / 52 := by
  sorry

end evaluate_expression_l748_74821


namespace fraction_relationship_l748_74885

theorem fraction_relationship (a b c : ℚ)
  (h1 : a / b = 3 / 5)
  (h2 : b / c = 2 / 7) :
  c / a = 35 / 6 :=
by
  sorry

end fraction_relationship_l748_74885


namespace not_product_of_two_primes_l748_74854

theorem not_product_of_two_primes (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : ∃ n : ℕ, a^3 + b^3 = n^2) :
  ¬ (∃ p q : ℕ, p ≠ q ∧ Prime p ∧ Prime q ∧ a + b = p * q) :=
by
  sorry

end not_product_of_two_primes_l748_74854


namespace min_value_x_squared_plus_y_squared_l748_74877

theorem min_value_x_squared_plus_y_squared {x y : ℝ} 
  (h : x^2 + y^2 - 4*x - 6*y + 12 = 0) : 
  ∃ m : ℝ, m = 14 - 2 * Real.sqrt 13 ∧ ∀ u v : ℝ, (u^2 + v^2 - 4*u - 6*v + 12 = 0) → (u^2 + v^2 ≥ m) :=
by
  sorry

end min_value_x_squared_plus_y_squared_l748_74877


namespace betty_age_l748_74871

theorem betty_age {A M B : ℕ} (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 14) : B = 7 :=
sorry

end betty_age_l748_74871


namespace max_value_proof_l748_74847

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + 2 * x * y + 3 * y^2

theorem max_value_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 12) : 
  max_value x y = 24 + 12 * Real.sqrt 3 := 
sorry

end max_value_proof_l748_74847


namespace binomial_sum_l748_74829

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_sum (n : ℤ) (h1 : binomial 25 n.natAbs + binomial 25 12 = binomial 26 13 ∧ n ≥ 0) : 
    (n = 12 ∨ n = 13) → n.succ + n = 25 := 
    sorry

end binomial_sum_l748_74829


namespace union_M_N_l748_74828

def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

theorem union_M_N :
  M ∪ N = {x | -3 ≤ x ∧ x < 1} := 
sorry

end union_M_N_l748_74828


namespace sphere_diameter_l748_74893

theorem sphere_diameter 
  (shadow_sphere : ℝ)
  (height_pole : ℝ)
  (shadow_pole : ℝ)
  (parallel_rays : Prop)
  (vertical_objects : Prop)
  (tan_theta : ℝ) :
  shadow_sphere = 12 →
  height_pole = 1.5 →
  shadow_pole = 3 →
  (tan_theta = height_pole / shadow_pole) →
  parallel_rays →
  vertical_objects →
  2 * (shadow_sphere * tan_theta) = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end sphere_diameter_l748_74893


namespace vector_cross_product_coordinates_l748_74810

variables (a1 a2 a3 b1 b2 b3 : ℝ)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

theorem vector_cross_product_coordinates :
  cross_product (a1, a2, a3) (b1, b2, b3) = 
    (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1) :=
by
sorry

end vector_cross_product_coordinates_l748_74810


namespace visibility_beach_to_hill_visibility_ferry_to_tree_l748_74882

noncomputable def altitude_lake : ℝ := 104
noncomputable def altitude_hill_tree : ℝ := 154
noncomputable def map_distance_1 : ℝ := 70 / 100 -- Convert cm to meters
noncomputable def map_distance_2 : ℝ := 38.5 / 100 -- Convert cm to meters
noncomputable def map_scale : ℝ := 95000
noncomputable def earth_circumference : ℝ := 40000000 -- Convert km to meters

noncomputable def earth_radius : ℝ := earth_circumference / (2 * Real.pi)

noncomputable def visible_distance (height : ℝ) : ℝ :=
  Real.sqrt (2 * earth_radius * height)

noncomputable def actual_distance_1 : ℝ := map_distance_1 * map_scale
noncomputable def actual_distance_2 : ℝ := map_distance_2 * map_scale

theorem visibility_beach_to_hill :
  actual_distance_1 > visible_distance (altitude_hill_tree - altitude_lake) :=
by
  sorry

theorem visibility_ferry_to_tree :
  actual_distance_2 > visible_distance (altitude_hill_tree - altitude_lake) :=
by
  sorry

end visibility_beach_to_hill_visibility_ferry_to_tree_l748_74882


namespace range_of_a_l748_74892

theorem range_of_a (a : ℝ) (h1 : ∃ x : ℝ, x > 0 ∧ |x| = a * x - a) (h2 : ∀ x : ℝ, x < 0 → |x| ≠ a * x - a) : a > 1 :=
sorry

end range_of_a_l748_74892


namespace min_days_required_l748_74891

theorem min_days_required (n : ℕ) (h1 : n ≥ 1) (h2 : 2 * (2^n - 1) ≥ 100) : n = 6 :=
sorry

end min_days_required_l748_74891


namespace cube_edge_length_l748_74837

theorem cube_edge_length (a : ℝ) (base_length : ℝ) (base_width : ℝ) (rise_height : ℝ) 
  (h_conditions : base_length = 20 ∧ base_width = 15 ∧ rise_height = 11.25 ∧ 
                  (base_length * base_width * rise_height) = a^3) : 
  a = 15 := 
by
  sorry

end cube_edge_length_l748_74837


namespace inequality_proof_l748_74866

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (a + c) ≥ 2 * Real.sqrt (a * b * c * (a + b + c)) := 
sorry

end inequality_proof_l748_74866


namespace parallelogram_isosceles_angles_l748_74801

def angle_sum_isosceles_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ (a = b ∨ b = c ∨ a = c)

theorem parallelogram_isosceles_angles :
  ∀ (A B C D P : Type) (AB BC CD DA BD : ℝ)
    (angle_DAB angle_BCD angle_ABC angle_CDA angle_ABP angle_BAP angle_PBD angle_BDP angle_CBD angle_BCD : ℝ),
  AB ≠ BC →
  angle_DAB = 72 →
  angle_BCD = 72 →
  angle_ABC = 108 →
  angle_CDA = 108 →
  angle_sum_isosceles_triangle angle_ABP angle_BAP 108 →
  angle_sum_isosceles_triangle 72 72 angle_BDP →
  angle_sum_isosceles_triangle 108 36 36 →
  ∃! (ABP BPD BCD : Type),
   (angle_ABP = 36 ∧ angle_BAP = 36 ∧ angle_PBA = 108) ∧
   (angle_PBD = 72 ∧ angle_PDB = 72 ∧ angle_BPD = 36) ∧
   (angle_CBD = 108 ∧ angle_BCD = 36 ∧ angle_BDC = 36) :=
sorry

end parallelogram_isosceles_angles_l748_74801


namespace sum_of_ages_l748_74888

/--
Given:
- Beckett's age is 12.
- Olaf is 3 years older than Beckett.
- Shannen is 2 years younger than Olaf.
- Jack is 5 more than twice as old as Shannen.

Prove that the sum of the ages of Beckett, Olaf, Shannen, and Jack is 71 years.
-/
theorem sum_of_ages :
  let Beckett := 12
  let Olaf := Beckett + 3
  let Shannen := Olaf - 2
  let Jack := 2 * Shannen + 5
  Beckett + Olaf + Shannen + Jack = 71 :=
by
  let Beckett := 12
  let Olaf := Beckett + 3
  let Shannen := Olaf - 2
  let Jack := 2 * Shannen + 5
  show Beckett + Olaf + Shannen + Jack = 71
  sorry

end sum_of_ages_l748_74888


namespace f_g_2_eq_1_l748_74886

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := -2 * x + 5

theorem f_g_2_eq_1 : f (g 2) = 1 :=
by
  sorry

end f_g_2_eq_1_l748_74886


namespace correct_statement_l748_74834

noncomputable def f (x : ℝ) := Real.exp x - x
noncomputable def g (x : ℝ) := Real.log x + x + 1

def proposition_p := ∀ x : ℝ, f x > 0
def proposition_q := ∃ x0 : ℝ, 0 < x0 ∧ g x0 = 0

theorem correct_statement : (proposition_p ∧ proposition_q) :=
by
  sorry

end correct_statement_l748_74834


namespace base5_division_l748_74817

-- Given conditions in decimal:
def n1_base10 : ℕ := 214
def n2_base10 : ℕ := 7

-- Convert the result back to base 5
def result_base5 : ℕ := 30  -- since 30 in decimal is 110 in base 5

theorem base5_division (h1 : 1324 = 214) (h2 : 12 = 7) : 1324 / 12 = 110 :=
by {
  -- these conditions help us bridge to the proof (intentionally left unproven here)
  sorry
}

end base5_division_l748_74817


namespace intersection_area_correct_l748_74827

noncomputable def intersection_area (XY YE FX EX FY : ℕ) : ℚ :=
  if XY = 12 ∧ YE = FX ∧ YE = 15 ∧ EX = FY ∧ EX = 20 then
    18
  else
    0

theorem intersection_area_correct {XY YE FX EX FY : ℕ} (h1 : XY = 12) (h2 : YE = FX) (h3 : YE = 15) (h4 : EX = FY) (h5 : EX = 20) : 
  intersection_area XY YE FX EX FY = 18 := 
by {
  sorry
}

end intersection_area_correct_l748_74827


namespace rectangle_width_l748_74840

theorem rectangle_width
  (L W : ℝ)
  (h1 : W = L + 2)
  (h2 : 2 * L + 2 * W = 16) :
  W = 5 :=
by
  sorry

end rectangle_width_l748_74840


namespace final_price_percentage_l748_74896

theorem final_price_percentage (P : ℝ) (h₀ : P > 0)
  (h₁ : ∃ P₁, P₁ = 0.80 * P)
  (h₂ : ∃ P₂, P₁ = 0.80 * P ∧ P₂ = P₁ - 0.10 * P₁) :
  P₂ = 0.72 * P :=
by
  sorry

end final_price_percentage_l748_74896


namespace change_calculation_l748_74878

/-!
# Problem
Adam has $5 to buy an airplane that costs $4.28. How much change will he get after buying the airplane?

# Conditions
Adam has $5.
The airplane costs $4.28.

# Statement
Prove that the change Adam will get is $0.72.
-/

theorem change_calculation : 
  let amount := 5.00
  let cost := 4.28
  let change := 0.72
  amount - cost = change :=
by 
  sorry

end change_calculation_l748_74878


namespace three_students_with_B_l748_74807

-- Define the students and their statements as propositions
variables (Eva B_Frank B_Gina B_Harry : Prop)

-- Condition 1: Eva said, "If I get a B, then Frank will get a B."
axiom Eva_statement : Eva → B_Frank

-- Condition 2: Frank said, "If I get a B, then Gina will get a B."
axiom Frank_statement : B_Frank → B_Gina

-- Condition 3: Gina said, "If I get a B, then Harry will get a B."
axiom Gina_statement : B_Gina → B_Harry

-- Condition 4: Only three students received a B.
axiom only_three_Bs : (Eva ∧ B_Frank ∧ B_Gina ∧ B_Harry) → False

-- The theorem we need to prove: The three students who received B's are Frank, Gina, and Harry.
theorem three_students_with_B (h_B_Frank : B_Frank) (h_B_Gina : B_Gina) (h_B_Harry : B_Harry) : ¬Eva :=
by
  sorry

end three_students_with_B_l748_74807


namespace students_circle_no_regular_exists_zero_regular_school_students_l748_74875

noncomputable def students_circle_no_regular (n : ℕ) 
    (student : ℕ → String)
    (neighbor_right : ℕ → ℕ)
    (lies_to : ℕ → ℕ → Bool) : Prop :=
  ∀ i, student i = "Gymnasium student" →
    (if lies_to i (neighbor_right i)
     then (student (neighbor_right i) ≠ "Gymnasium student")
     else student (neighbor_right i) = "Gymnasium student") →
    (if lies_to (neighbor_right i) i
     then (student i ≠ "Gymnasium student")
     else student i = "Gymnasium student")

theorem students_circle_no_regular_exists_zero_regular_school_students
  (n : ℕ) 
  (student : ℕ → String)
  (neighbor_right : ℕ → ℕ)
  (lies_to : ℕ → ℕ → Bool)
  (h : students_circle_no_regular n student neighbor_right lies_to)
  : (∀ i, student i ≠ "Regular school student") :=
sorry

end students_circle_no_regular_exists_zero_regular_school_students_l748_74875


namespace ways_to_start_writing_l748_74895

def ratio_of_pens_to_notebooks (pens notebooks : ℕ) : Prop := 
    pens * 4 = notebooks * 5

theorem ways_to_start_writing 
    (pens notebooks : ℕ) 
    (h_ratio : ratio_of_pens_to_notebooks pens notebooks) 
    (h_pens : pens = 50)
    (h_notebooks : notebooks = 40) : 
    ∃ ways : ℕ, ways = 40 :=
by
  sorry

end ways_to_start_writing_l748_74895


namespace speed_of_train_A_is_90_kmph_l748_74863

-- Definitions based on the conditions
def train_length_A := 225 -- in meters
def train_length_B := 150 -- in meters
def crossing_time := 15 -- in seconds

-- The total distance covered by train A to cross train B
def total_distance := train_length_A + train_length_B

-- The speed of train A in m/s
def speed_in_mps := total_distance / crossing_time

-- Conversion factor from m/s to km/hr
def mps_to_kmph (mps: ℕ) := mps * 36 / 10

-- The speed of train A in km/hr
def speed_in_kmph := mps_to_kmph speed_in_mps

-- The theorem to be proved
theorem speed_of_train_A_is_90_kmph : speed_in_kmph = 90 := by
  -- Proof steps go here
  sorry

end speed_of_train_A_is_90_kmph_l748_74863


namespace students_per_group_l748_74868

-- Define the conditions:
def total_students : ℕ := 120
def not_picked_students : ℕ := 22
def groups : ℕ := 14

-- Calculate the picked students:
def picked_students : ℕ := total_students - not_picked_students

-- Statement of the problem:
theorem students_per_group : picked_students / groups = 7 :=
  by sorry

end students_per_group_l748_74868


namespace packages_per_hour_A_B_max_A_robots_l748_74849

-- Define the number of packages sorted by each unit of type A and B robots
def packages_by_A_robot (x : ℕ) := x
def packages_by_B_robot (y : ℕ) := y

-- Problem conditions
def cond1 (x y : ℕ) : Prop := 80 * x + 100 * y = 8200
def cond2 (x y : ℕ) : Prop := 50 * x + 50 * y = 4500

-- Part 1: to prove type A and type B robot's packages per hour
theorem packages_per_hour_A_B (x y : ℕ) (h1 : cond1 x y) (h2 : cond2 x y) : x = 40 ∧ y = 50 :=
by sorry

-- Part 2: prove maximum units of type A robots when purchasing 200 robots ensuring not < 9000 packages/hour
def cond3 (m : ℕ) : Prop := 40 * m + 50 * (200 - m) ≥ 9000

theorem max_A_robots (m : ℕ) (h3 : cond3 m) : m ≤ 100 :=
by sorry

end packages_per_hour_A_B_max_A_robots_l748_74849


namespace tan_ratio_l748_74800

theorem tan_ratio (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5 / 8)
  (h2 : Real.sin (a - b) = 1 / 4) : 
  Real.tan a / Real.tan b = 7 / 3 := 
sorry

end tan_ratio_l748_74800


namespace g_cross_horizontal_asymptote_at_l748_74873

noncomputable def g (x : ℝ) : ℝ :=
  (3 * x^2 - 7 * x - 8) / (x^2 - 5 * x + 6)

theorem g_cross_horizontal_asymptote_at (x : ℝ) : g x = 3 ↔ x = 13 / 4 :=
by
  sorry

end g_cross_horizontal_asymptote_at_l748_74873


namespace smallest_n_for_isosceles_trapezoid_coloring_l748_74852

def isIsoscelesTrapezoid (a b c d : ℕ) : Prop :=
  -- conditions to check if vertices a, b, c, d form an isosceles trapezoid in a regular n-gon
  sorry  -- definition of an isosceles trapezoid

def vertexColors (n : ℕ) : Fin n → Fin 3 :=
  sorry  -- vertex coloring function

theorem smallest_n_for_isosceles_trapezoid_coloring :
  ∃ n : ℕ, (∀ (vertices : Fin n → Fin 3), ∃ (a b c d : Fin n),
    vertexColors n a = vertexColors n b ∧
    vertexColors n b = vertexColors n c ∧
    vertexColors n c = vertexColors n d ∧
    isIsoscelesTrapezoid a b c d) ∧ n = 17 :=
by
  sorry

end smallest_n_for_isosceles_trapezoid_coloring_l748_74852


namespace number_of_pounds_colombian_beans_l748_74804

def cost_per_pound_colombian : ℝ := 5.50
def cost_per_pound_peruvian : ℝ := 4.25
def total_weight : ℝ := 40
def desired_cost_per_pound : ℝ := 4.60
noncomputable def amount_colombian_beans (C : ℝ) : Prop := 
  let P := total_weight - C
  cost_per_pound_colombian * C + cost_per_pound_peruvian * P = desired_cost_per_pound * total_weight

theorem number_of_pounds_colombian_beans : ∃ C, amount_colombian_beans C ∧ C = 11.2 :=
sorry

end number_of_pounds_colombian_beans_l748_74804


namespace bills_average_speed_l748_74880

theorem bills_average_speed :
  ∃ v t : ℝ, 
      (v + 5) * (t + 2) + v * t = 680 ∧ 
      (t + 2) + t = 18 ∧ 
      v = 35 :=
by
  sorry

end bills_average_speed_l748_74880


namespace students_taking_statistics_l748_74806

-- Definitions based on conditions
def total_students := 89
def history_students := 36
def history_or_statistics := 59
def history_not_statistics := 27

-- The proof problem
theorem students_taking_statistics : ∃ S : ℕ, S = 32 ∧
  ((history_students - history_not_statistics) + S - (history_students - history_not_statistics)) = history_or_statistics :=
by
  use 32
  sorry

end students_taking_statistics_l748_74806


namespace inequality_proof_l748_74876

theorem inequality_proof (a b c A α : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (h_sum : a + b + c = A) (hA : A ≤ 1) (hα : α > 0) :
  ( (1 / a - a) ^ α + (1 / b - b) ^ α + (1 / c - c) ^ α ) ≥ 3 * ( (3 / A) - (A / 3) ) ^ α :=
by
  sorry

end inequality_proof_l748_74876


namespace bouquet_branches_l748_74850

variable (w : ℕ) (b : ℕ)

theorem bouquet_branches :
  (w + b = 7) → 
  (w ≥ 1) → 
  (∀ x y, x ≠ y → (x = w ∨ y = w) → (x = b ∨ y = b)) → 
  (w = 1 ∧ b = 6) :=
by
  intro h1 h2 h3
  sorry

end bouquet_branches_l748_74850


namespace history_paper_pages_l748_74848

/-
Stacy has a history paper due in 3 days.
She has to write 21 pages per day to finish on time.
Prove that the total number of pages for the history paper is 63.
-/

theorem history_paper_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 21) (h2 : days = 3) : total_pages = 63 :=
by
  -- We would include the proof here, but for now, we use sorry to skip the proof.
  sorry

end history_paper_pages_l748_74848


namespace find_a_l748_74869

open Set Real

-- Defining sets A and B, and the condition A ∩ B = {3}
def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

-- Mathematically equivalent proof statement
theorem find_a (a : ℝ) (h : A ∩ B a = {3}) : a = 1 :=
  sorry

end find_a_l748_74869


namespace monogram_count_is_correct_l748_74832

def count_possible_monograms : ℕ :=
  Nat.choose 23 2

theorem monogram_count_is_correct : 
  count_possible_monograms = 253 := 
by 
  -- The proof will show this matches the combination formula calculation
  -- The final proof is left incomplete as per the instructions
  sorry

end monogram_count_is_correct_l748_74832


namespace lcm_18_35_l748_74823

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := 
by 
  sorry

end lcm_18_35_l748_74823


namespace min_value_x2y3z_l748_74864

theorem min_value_x2y3z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 2 / x + 3 / y + 1 / z = 12) :
  x^2 * y^3 * z ≥ 1 / 64 :=
by
  sorry

end min_value_x2y3z_l748_74864


namespace cost_of_ABC_book_l748_74860

theorem cost_of_ABC_book (x : ℕ) 
  (h₁ : 8 = 8)  -- Cost of "TOP" book is 8 dollars
  (h₂ : 13 * 8 = 104)  -- Thirteen "TOP" books sold last week
  (h₃ : 104 - 4 * x = 12)  -- Difference in earnings is $12
  : x = 23 :=
sorry

end cost_of_ABC_book_l748_74860


namespace solve_inequality_l748_74818

theorem solve_inequality :
  {x : ℝ | -3 * x^2 + 5 * x + 4 < 0} = {x : ℝ | x < 3 / 4} ∪ {x : ℝ | 1 < x} :=
by
  sorry

end solve_inequality_l748_74818


namespace downstream_speed_is_45_l748_74811

-- Define the conditions
def upstream_speed := 35 -- The man can row upstream at 35 kmph
def still_water_speed := 40 -- The speed of the man in still water is 40 kmph

-- Define the speed of the stream based on the given conditions
def stream_speed := still_water_speed - upstream_speed 

-- Define the speed of the man rowing downstream
def downstream_speed := still_water_speed + stream_speed

-- The assertion to prove
theorem downstream_speed_is_45 : downstream_speed = 45 := by
  sorry

end downstream_speed_is_45_l748_74811


namespace alice_still_needs_to_fold_l748_74802

theorem alice_still_needs_to_fold (total_cranes alice_folds friend_folds remains: ℕ) 
  (h1 : total_cranes = 1000)
  (h2 : alice_folds = total_cranes / 2)
  (h3 : friend_folds = (total_cranes - alice_folds) / 5)
  (h4 : remains = total_cranes - alice_folds - friend_folds) :
  remains = 400 := 
  by
    sorry

end alice_still_needs_to_fold_l748_74802


namespace negation_of_existence_statement_l748_74853

theorem negation_of_existence_statement :
  (¬ ∃ x : ℝ, x^2 - 8 * x + 18 < 0) ↔ (∀ x : ℝ, x^2 - 8 * x + 18 ≥ 0) :=
by
  sorry

end negation_of_existence_statement_l748_74853


namespace find_xy_value_l748_74881

theorem find_xy_value (x y z w : ℕ) (h1 : x = w) (h2 : y = z) (h3 : w + w = z * w) (h4 : y = w)
    (h5 : w + w = w * w) (h6 : z = 3) : x * y = 4 := by
  -- Given that w = 2 based on the conditions
  sorry

end find_xy_value_l748_74881


namespace necessary_condition_l748_74819

theorem necessary_condition :
  ∃ x : ℝ, (x < 0 ∨ x > 2) → (2 * x^2 - 5 * x - 3 ≥ 0) :=
sorry

end necessary_condition_l748_74819


namespace valid_lineups_l748_74846

def total_players : ℕ := 15
def k : ℕ := 2  -- number of twins
def total_chosen : ℕ := 7
def remaining_players := total_players - k

def nCr (n r : ℕ) : ℕ :=
  if r > n then 0
  else Nat.choose n r

def total_choices : ℕ := nCr total_players total_chosen
def restricted_choices : ℕ := nCr remaining_players (total_chosen - k)

theorem valid_lineups : total_choices - restricted_choices = 5148 := by
  sorry

end valid_lineups_l748_74846


namespace exist_unique_rectangular_prism_Q_l748_74814

variable (a b c : ℝ) (h_lt : a < b ∧ b < c)
variable (x y z : ℝ) (hx_lt : x < y ∧ y < z ∧ z < a)

theorem exist_unique_rectangular_prism_Q :
  (2 * (x*y + y*z + z*x) = 0.5 * (a*b + b*c + c*a) ∧ x*y*z = 0.25 * a*b*c) ∧ (x < y ∧ y < z ∧ z < a) → 
  ∃! x y z, (2 * (x*y + y*z + z*x) = 0.5 * (a*b + b*c + c*a) ∧ x*y*z = 0.25 * a*b*c) :=
sorry

end exist_unique_rectangular_prism_Q_l748_74814


namespace lisa_interest_earned_l748_74816

/-- Lisa's interest earned after three years from Bank of Springfield's Super High Yield savings account -/
theorem lisa_interest_earned :
  let P := 2000
  let r := 0.02
  let n := 3
  let A := P * (1 + r)^n
  A - P = 122 := by
  sorry

end lisa_interest_earned_l748_74816


namespace books_about_fish_l748_74831

theorem books_about_fish (F : ℕ) (spent : ℕ) (cost_whale_books : ℕ) (cost_magazines : ℕ) (cost_fish_books_per_unit : ℕ) (whale_books : ℕ) (magazines : ℕ) :
  whale_books = 9 →
  magazines = 3 →
  cost_whale_books = 11 →
  cost_magazines = 1 →
  spent = 179 →
  99 + 11 * F + 3 = spent → F = 7 :=
by
  sorry

end books_about_fish_l748_74831
