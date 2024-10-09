import Mathlib

namespace product_of_ages_l1175_117566

theorem product_of_ages (O Y : ℕ) (h1 : O - Y = 12) (h2 : O + Y = (O - Y) + 40) : O * Y = 640 := by
  sorry

end product_of_ages_l1175_117566


namespace problem_solution_l1175_117599

theorem problem_solution (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a + b = 1 :=
sorry

end problem_solution_l1175_117599


namespace triangle_ABC_properties_l1175_117535

theorem triangle_ABC_properties 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 2 * Real.sin B * Real.sin C * Real.cos A + Real.cos A = 3 * Real.sin A ^ 2 - Real.cos (B - C)) : 
  (2 * a = b + c) ∧ 
  (b + c = 2) →
  (Real.cos A = 3/5) → 
  (1 / 2 * b * c * Real.sin A = 3 / 8) :=
by
  sorry

end triangle_ABC_properties_l1175_117535


namespace find_x_plus_y_l1175_117554

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2010) (h2 : x + 2010 * Real.sin y = 2009) (hy : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 :=
by
  sorry

end find_x_plus_y_l1175_117554


namespace jameson_badminton_medals_l1175_117508

theorem jameson_badminton_medals (total_medals track_medals : ℕ) (swimming_medals : ℕ) :
  total_medals = 20 →
  track_medals = 5 →
  swimming_medals = 2 * track_medals →
  total_medals - (track_medals + swimming_medals) = 5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end jameson_badminton_medals_l1175_117508


namespace bart_earned_14_l1175_117537

variable (questions_per_survey money_per_question surveys_monday surveys_tuesday : ℕ → ℝ)
variable (total_surveys total_questions money_earned : ℕ → ℝ)

noncomputable def conditions :=
  let questions_per_survey := 10
  let money_per_question := 0.2
  let surveys_monday := 3
  let surveys_tuesday := 4
  let total_surveys := surveys_monday + surveys_tuesday
  let total_questions := questions_per_survey * total_surveys
  let money_earned := total_questions * money_per_question
  money_earned = 14

theorem bart_earned_14 : conditions :=
by
  -- proof steps
  sorry

end bart_earned_14_l1175_117537


namespace jackson_grade_l1175_117575

open Function

theorem jackson_grade :
  ∃ (grade : ℕ), 
  ∀ (hours_playing hours_studying : ℕ), 
    (hours_playing = 9) ∧ 
    (hours_studying = hours_playing / 3) ∧ 
    (grade = hours_studying * 15) →
    grade = 45 := 
by {
  sorry
}

end jackson_grade_l1175_117575


namespace solve_rectangular_field_problem_l1175_117512

-- Define the problem
def f (L W : ℝ) := L * W = 80 ∧ 2 * W + L = 28

-- Define the length of the uncovered side
def length_of_uncovered_side (L: ℝ) := L = 20

-- The statement we need to prove
theorem solve_rectangular_field_problem (L W : ℝ) (h : f L W) : length_of_uncovered_side L :=
by
  sorry

end solve_rectangular_field_problem_l1175_117512


namespace terminal_side_in_third_quadrant_l1175_117546

theorem terminal_side_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  (∃ k : ℤ, α = k * π + π / 2 + π) := sorry

end terminal_side_in_third_quadrant_l1175_117546


namespace manfred_total_paychecks_l1175_117555

-- Define the conditions
def first_paychecks : ℕ := 6
def first_paycheck_amount : ℕ := 750
def remaining_paycheck_amount : ℕ := first_paycheck_amount + 20
def average_amount : ℝ := 765.38

-- Main theorem statement
theorem manfred_total_paychecks (x : ℕ) (h : (first_paychecks * first_paycheck_amount + x * remaining_paycheck_amount) / (first_paychecks + x) = average_amount) : first_paychecks + x = 26 :=
by
  sorry

end manfred_total_paychecks_l1175_117555


namespace printing_machine_completion_time_l1175_117544

-- Definitions of times in hours
def start_time : ℕ := 9 -- 9:00 AM
def half_job_time : ℕ := 12 -- 12:00 PM
def completion_time : ℕ := 15 -- 3:00 PM

-- Time taken to complete half the job
def half_job_duration : ℕ := half_job_time - start_time

-- Total time to complete the entire job
def total_job_duration : ℕ := 2 * half_job_duration

-- Proof that the machine will complete the job at 3:00 PM
theorem printing_machine_completion_time : 
    start_time + total_job_duration = completion_time :=
sorry

end printing_machine_completion_time_l1175_117544


namespace like_terms_implies_a_plus_2b_eq_3_l1175_117584

theorem like_terms_implies_a_plus_2b_eq_3 (a b : ℤ) (h1 : 2 * a + b = 6) (h2 : a - b = 3) : a + 2 * b = 3 :=
sorry

end like_terms_implies_a_plus_2b_eq_3_l1175_117584


namespace intersection_of_P_with_complement_Q_l1175_117565

-- Define the universal set U, and sets P and Q
def U : List ℕ := [1, 2, 3, 4]
def P : List ℕ := [1, 2]
def Q : List ℕ := [2, 3]

-- Define the complement of Q with respect to U
def complement (U Q : List ℕ) : List ℕ := U.filter (λ x => x ∉ Q)

-- Define the intersection of two sets
def intersection (A B : List ℕ) : List ℕ := A.filter (λ x => x ∈ B)

-- The proof statement we need to show
theorem intersection_of_P_with_complement_Q : intersection P (complement U Q) = [1] := by
  sorry

end intersection_of_P_with_complement_Q_l1175_117565


namespace evaluate_expression_l1175_117515

theorem evaluate_expression :
  let x := (1/4 : ℚ)
  let y := (4/5 : ℚ)
  let z := (-2 : ℚ)
  x^3 * y^2 * z^2 = 1/25 :=
by
  sorry

end evaluate_expression_l1175_117515


namespace no_real_solutions_l1175_117511

-- Define the equation
def equation (x : ℝ) : Prop :=
  (2 * x ^ 2 - 6 * x + 5) ^ 2 + 1 = -|x|

-- Declare the theorem which states there are no real solutions to the given equation
theorem no_real_solutions : ∀ x : ℝ, ¬ equation x :=
by
  intro x
  sorry

end no_real_solutions_l1175_117511


namespace moles_NaCl_formed_in_reaction_l1175_117506

noncomputable def moles_of_NaCl_formed (moles_NaOH moles_HCl : ℕ) : ℕ :=
  if moles_NaOH = 1 ∧ moles_HCl = 1 then 1 else 0

theorem moles_NaCl_formed_in_reaction : moles_of_NaCl_formed 1 1 = 1 := 
by
  sorry

end moles_NaCl_formed_in_reaction_l1175_117506


namespace subway_length_in_meters_l1175_117510

noncomputable def subway_speed : ℝ := 1.6 -- km per minute
noncomputable def crossing_time : ℝ := 3 + 15 / 60 -- minutes
noncomputable def bridge_length : ℝ := 4.85 -- km

theorem subway_length_in_meters :
  let total_distance_traveled := subway_speed * crossing_time
  let subway_length_km := total_distance_traveled - bridge_length
  let subway_length_m := subway_length_km * 1000
  subway_length_m = 350 :=
by
  sorry

end subway_length_in_meters_l1175_117510


namespace temperature_decrease_2C_l1175_117530

variable (increase_3 : ℤ := 3)
variable (decrease_2 : ℤ := -2)

theorem temperature_decrease_2C :
  decrease_2 = -2 :=
by
  -- This is where the proof would go
  sorry

end temperature_decrease_2C_l1175_117530


namespace solution_set_of_inequality_l1175_117570

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} :=
by
  sorry

end solution_set_of_inequality_l1175_117570


namespace infinite_series_eq_5_over_16_l1175_117571

noncomputable def infinite_series_sum : ℝ :=
  ∑' (n : ℕ), (n + 1 : ℝ) / (5 ^ (n + 1))

theorem infinite_series_eq_5_over_16 :
  infinite_series_sum = 5 / 16 :=
sorry

end infinite_series_eq_5_over_16_l1175_117571


namespace reservoir_water_l1175_117585

-- Conditions definitions
def total_capacity (C : ℝ) : Prop :=
  ∃ (x : ℝ), x = C

def normal_level (C : ℝ) : ℝ :=
  C - 20

def water_end_of_month (C : ℝ) : ℝ :=
  0.75 * C

def condition_equation (C : ℝ) : Prop :=
  water_end_of_month C = 2 * normal_level C

-- The theorem proving the amount of water at the end of the month is 24 million gallons given the conditions
theorem reservoir_water (C : ℝ) (hC : total_capacity C) (h_condition : condition_equation C) : water_end_of_month C = 24 :=
by
  sorry

end reservoir_water_l1175_117585


namespace book_arrangement_count_l1175_117507

theorem book_arrangement_count :
  let n := 6
  let identical_pairs := 2
  let total_arrangements_if_unique := n.factorial
  let ident_pair_correction := (identical_pairs.factorial * identical_pairs.factorial)
  (total_arrangements_if_unique / ident_pair_correction) = 180 := by
  sorry

end book_arrangement_count_l1175_117507


namespace total_distance_travelled_l1175_117572

/-- Proving that the total horizontal distance traveled by the centers of two wheels with radii 1 m and 2 m 
    after one complete revolution is 6π meters. -/
theorem total_distance_travelled (R1 R2 : ℝ) (h1 : R1 = 1) (h2 : R2 = 2) : 
    2 * Real.pi * R1 + 2 * Real.pi * R2 = 6 * Real.pi :=
by
  sorry

end total_distance_travelled_l1175_117572


namespace men_at_conference_l1175_117519

theorem men_at_conference (M : ℕ) 
  (num_women : ℕ) (num_children : ℕ)
  (indian_men_fraction : ℚ) (indian_women_fraction : ℚ)
  (indian_children_fraction : ℚ) (non_indian_fraction : ℚ)
  (num_women_eq : num_women = 300)
  (num_children_eq : num_children = 500)
  (indian_men_fraction_eq : indian_men_fraction = 0.10)
  (indian_women_fraction_eq : indian_women_fraction = 0.60)
  (indian_children_fraction_eq : indian_children_fraction = 0.70)
  (non_indian_fraction_eq : non_indian_fraction = 0.5538461538461539) :
  M = 500 :=
by
  sorry

end men_at_conference_l1175_117519


namespace flag_pole_height_eq_150_l1175_117590

-- Define the conditions
def tree_height : ℝ := 12
def tree_shadow_length : ℝ := 8
def flag_pole_shadow_length : ℝ := 100

-- Problem statement: prove the height of the flag pole equals 150 meters
theorem flag_pole_height_eq_150 :
  ∃ (F : ℝ), (tree_height / tree_shadow_length) = (F / flag_pole_shadow_length) ∧ F = 150 :=
by
  -- Setup the proof scaffold
  have h : (tree_height / tree_shadow_length) = (150 / flag_pole_shadow_length) := by sorry
  exact ⟨150, h, rfl⟩

end flag_pole_height_eq_150_l1175_117590


namespace book_surface_area_l1175_117541

variables (L : ℕ) (T : ℕ) (A1 : ℕ) (A2 : ℕ) (W : ℕ) (S : ℕ)

theorem book_surface_area (hL : L = 5) (hT : T = 2) 
                         (hA1 : A1 = L * W) (hA1_val : A1 = 50)
                         (hA2 : A2 = T * W) (hA2_val : A2 = 10) :
  S = 2 * A1 + A2 + 2 * (L * T) :=
sorry

end book_surface_area_l1175_117541


namespace max_unique_rankings_l1175_117503

theorem max_unique_rankings (n : ℕ) : 
  ∃ (contestants : ℕ), 
    (∀ (scores : ℕ → ℕ), 
      (∀ i, 0 ≤ scores i ∧ scores i ≤ contestants) ∧
      (∀ i j, i ≠ j → scores i ≠ scores j)) 
    → contestants = 2^n := 
sorry

end max_unique_rankings_l1175_117503


namespace train_crossing_time_l1175_117593

theorem train_crossing_time :
  ∀ (length_train1 length_train2 : ℕ) 
    (speed_train1_kmph speed_train2_kmph : ℝ), 
  length_train1 = 420 →
  speed_train1_kmph = 72 →
  length_train2 = 640 →
  speed_train2_kmph = 36 →
  (length_train1 + length_train2) / ((speed_train1_kmph - speed_train2_kmph) * (1000 / 3600)) = 106 :=
by
  intros
  sorry

end train_crossing_time_l1175_117593


namespace boys_chairs_problem_l1175_117505

theorem boys_chairs_problem :
  ∃ (n k : ℕ), n * k = 123 ∧ (∀ p q : ℕ, p * q = 123 → p = n ∧ q = k ∨ p = k ∧ q = n) :=
by
  sorry

end boys_chairs_problem_l1175_117505


namespace sequence_term_4_l1175_117525

noncomputable def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * sequence n

theorem sequence_term_4 : sequence 3 = 8 := 
by
  sorry

end sequence_term_4_l1175_117525


namespace right_angled_triangle_l1175_117588

-- Define the lengths of the sides of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- The theorem to prove that these lengths form a right-angled triangle
theorem right_angled_triangle : a^2 + b^2 = c^2 :=
by
  sorry

end right_angled_triangle_l1175_117588


namespace jose_to_haylee_ratio_l1175_117589

variable (J : ℕ)

def haylee_guppies := 36
def charliz_guppies := J / 3
def nicolai_guppies := 4 * (J / 3)
def total_guppies := haylee_guppies + J + charliz_guppies + nicolai_guppies

theorem jose_to_haylee_ratio :
  haylee_guppies = 36 ∧ total_guppies = 84 →
  J / haylee_guppies = 1 / 2 :=
by
  intro h
  sorry

end jose_to_haylee_ratio_l1175_117589


namespace geometric_sequence_sum_l1175_117539

noncomputable def geometric_sequence (a : ℕ → ℝ) (r: ℝ): Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (r: ℝ)
  (h_geometric : geometric_sequence a r)
  (h_ratio : r = 2)
  (h_sum_condition : a 1 + a 4 + a 7 = 10) :
  a 3 + a 6 + a 9 = 20 := 
sorry

end geometric_sequence_sum_l1175_117539


namespace linear_dependency_k_l1175_117591

theorem linear_dependency_k (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧
    (c1 * 1 + c2 * 4 = 0) ∧
    (c1 * 2 + c2 * k = 0) ∧
    (c1 * 3 + c2 * 6 = 0)) ↔ k = 8 :=
by
  sorry

end linear_dependency_k_l1175_117591


namespace cost_per_package_l1175_117524

theorem cost_per_package
  (parents : ℕ)
  (brothers : ℕ)
  (spouses_per_brother : ℕ)
  (children_per_brother : ℕ)
  (total_cost : ℕ)
  (num_packages : ℕ)
  (h1 : parents = 2)
  (h2 : brothers = 3)
  (h3 : spouses_per_brother = 1)
  (h4 : children_per_brother = 2)
  (h5 : total_cost = 70)
  (h6 : num_packages = parents + brothers + brothers * spouses_per_brother + brothers * children_per_brother) :
  total_cost / num_packages = 5 :=
by
  -- Proof goes here
  sorry

end cost_per_package_l1175_117524


namespace part1_part2_l1175_117500

-- Part 1: Prove that the range of values for k is k ≤ 1/4
theorem part1 (f : ℝ → ℝ) (k : ℝ) 
  (h1 : ∀ x0 : ℝ, f x0 ≥ |k+3| - |k-2|)
  (h2 : ∀ x : ℝ, f x = |2*x - 1| + |x - 2| ) : 
  k ≤ 1/4 := 
sorry

-- Part 2: Show that the minimum value of m+n is 8/3
theorem part2 (f : ℝ → ℝ) (m n : ℝ) 
  (h1 : ∀ x : ℝ, f x ≥ 1/m + 1/n)
  (h2 : ∀ x : ℝ, f x = |2*x - 1| + |x - 2| ) : 
  m + n ≥ 8/3 := 
sorry

end part1_part2_l1175_117500


namespace boys_left_is_31_l1175_117534

def initial_children : ℕ := 85
def girls_came_in : ℕ := 24
def final_children : ℕ := 78

noncomputable def compute_boys_left (initial : ℕ) (girls_in : ℕ) (final : ℕ) : ℕ :=
  (initial + girls_in) - final

theorem boys_left_is_31 :
  compute_boys_left initial_children girls_came_in final_children = 31 :=
by
  sorry

end boys_left_is_31_l1175_117534


namespace elder_three_times_younger_l1175_117569

-- Definitions based on conditions
def age_difference := 16
def elder_present_age := 30
def younger_present_age := elder_present_age - age_difference

-- The problem statement to prove the correct value of n (years ago)
theorem elder_three_times_younger (n : ℕ) 
  (h1 : elder_present_age = younger_present_age + age_difference)
  (h2 : elder_present_age - n = 3 * (younger_present_age - n)) : 
  n = 6 := 
sorry

end elder_three_times_younger_l1175_117569


namespace total_fish_correct_l1175_117563

def Leo_fish := 40
def Agrey_fish := Leo_fish + 20
def Sierra_fish := Agrey_fish + 15
def total_fish := Leo_fish + Agrey_fish + Sierra_fish

theorem total_fish_correct : total_fish = 175 := by
  sorry


end total_fish_correct_l1175_117563


namespace largest_integer_x_l1175_117557

theorem largest_integer_x (x : ℤ) (h : 3 - 5 * x > 22) : x ≤ -4 :=
by
  sorry

end largest_integer_x_l1175_117557


namespace candy_ratio_l1175_117598

theorem candy_ratio (chocolate_bars M_and_Ms marshmallows total_candies : ℕ)
  (h1 : chocolate_bars = 5)
  (h2 : M_and_Ms = 7 * chocolate_bars)
  (h3 : total_candies = 25 * 10)
  (h4 : marshmallows = total_candies - chocolate_bars - M_and_Ms) :
  marshmallows / M_and_Ms = 6 :=
by
  sorry

end candy_ratio_l1175_117598


namespace find_sister_candy_initially_l1175_117592

-- Defining the initial pieces of candy Katie had.
def katie_candy : ℕ := 8

-- Defining the pieces of candy Katie's sister had initially.
def sister_candy_initially : ℕ := sorry -- To be determined

-- The total number of candy pieces they had after eating 8 pieces.
def total_remaining_candy : ℕ := 23

theorem find_sister_candy_initially : 
  (katie_candy + sister_candy_initially - 8 = total_remaining_candy) → (sister_candy_initially = 23) :=
by
  sorry

end find_sister_candy_initially_l1175_117592


namespace compute_result_l1175_117581

-- Define the operations a # b and b # c
def operation (a b : ℤ) : ℤ := a * b - b + b^2

-- Define the expression for (3 # 8) # z given the operations
def evaluate (z : ℤ) : ℤ := operation (operation 3 8) z

-- Prove that (3 # 8) # z = 79z + z^2
theorem compute_result (z : ℤ) : evaluate z = 79 * z + z^2 := 
by
  sorry

end compute_result_l1175_117581


namespace sqrt_16_l1175_117561

theorem sqrt_16 : {x : ℝ | x^2 = 16} = {4, -4} :=
by
  sorry

end sqrt_16_l1175_117561


namespace cauliflower_difference_is_401_l1175_117543

-- Definitions using conditions from part a)
def garden_area_this_year : ℕ := 40401
def side_length_this_year : ℕ := Nat.sqrt garden_area_this_year
def side_length_last_year : ℕ := side_length_this_year - 1
def garden_area_last_year : ℕ := side_length_last_year ^ 2
def cauliflowers_difference : ℕ := garden_area_this_year - garden_area_last_year

-- Problem statement claiming that the difference in cauliflowers produced is 401
theorem cauliflower_difference_is_401 :
  garden_area_this_year = 40401 →
  side_length_this_year = 201 →
  side_length_last_year = 200 →
  garden_area_last_year = 40000 →
  cauliflowers_difference = 401 :=
by
  intros
  sorry

end cauliflower_difference_is_401_l1175_117543


namespace division_multiplication_result_l1175_117586

theorem division_multiplication_result : (180 / 6) * 3 = 90 := by
  sorry

end division_multiplication_result_l1175_117586


namespace greatest_prime_factor_of_5_pow_7_plus_6_pow_6_l1175_117548

def greatest_prime_factor (n : ℕ) : ℕ :=
  sorry -- Implementation of finding greatest prime factor goes here

theorem greatest_prime_factor_of_5_pow_7_plus_6_pow_6 : 
  greatest_prime_factor (5^7 + 6^6) = 211 := 
by 
  sorry -- Proof of the theorem goes here

end greatest_prime_factor_of_5_pow_7_plus_6_pow_6_l1175_117548


namespace total_goals_l1175_117523

-- Define constants for goals scored in respective seasons
def goalsLastSeason : ℕ := 156
def goalsThisSeason : ℕ := 187

-- Define the theorem for the total number of goals
theorem total_goals : goalsLastSeason + goalsThisSeason = 343 :=
by
  -- Proof is omitted
  sorry

end total_goals_l1175_117523


namespace units_digit_k_squared_plus_three_to_the_k_mod_10_l1175_117560

def k := 2025^2 + 3^2025

theorem units_digit_k_squared_plus_three_to_the_k_mod_10 : 
  (k^2 + 3^k) % 10 = 5 := by
sorry

end units_digit_k_squared_plus_three_to_the_k_mod_10_l1175_117560


namespace speed_in_still_water_l1175_117568

-- Given conditions
def upstream_speed : ℝ := 60
def downstream_speed : ℝ := 90

-- Proof that the speed of the man in still water is 75 kmph
theorem speed_in_still_water :
  (upstream_speed + downstream_speed) / 2 = 75 := 
by
  sorry

end speed_in_still_water_l1175_117568


namespace train_length_l1175_117509

theorem train_length (L V : ℝ) (h1 : L = V * 120) (h2 : L + 1000 = V * 220) : L = 1200 := 
by
  sorry

end train_length_l1175_117509


namespace gcd_max_value_l1175_117502

theorem gcd_max_value : ∀ (n : ℕ), n > 0 → ∃ (d : ℕ), d = 9 ∧ d ∣ gcd (13 * n + 4) (8 * n + 3) :=
by
  sorry

end gcd_max_value_l1175_117502


namespace age_difference_l1175_117562

def age1 : ℕ := 10
def age2 : ℕ := age1 - 2
def age3 : ℕ := age2 + 4
def age4 : ℕ := age3 / 2
def age5 : ℕ := age4 + 20
def avg : ℕ := (age1 + age5) / 2

theorem age_difference :
  (age3 - age2) = 4 ∧ avg = 18 := by
  sorry

end age_difference_l1175_117562


namespace average_weight_estimate_l1175_117551

noncomputable def average_weight (female_students male_students : ℕ) (avg_weight_female avg_weight_male : ℕ) : ℝ :=
  (female_students / (female_students + male_students) : ℝ) * avg_weight_female +
  (male_students / (female_students + male_students) : ℝ) * avg_weight_male

theorem average_weight_estimate :
  average_weight 504 596 49 57 = (504 / 1100 : ℝ) * 49 + (596 / 1100 : ℝ) * 57 :=
by
  sorry

end average_weight_estimate_l1175_117551


namespace election_votes_l1175_117516

theorem election_votes (V : ℝ) 
    (h1 : ∃ c1 c2 : ℝ, c1 + c2 = V ∧ c1 = 0.60 * V ∧ c2 = 0.40 * V)
    (h2 : ∃ m : ℝ, m = 280 ∧ 0.60 * V - 0.40 * V = m) : 
    V = 1400 :=
by
  sorry

end election_votes_l1175_117516


namespace average_mark_of_remaining_students_l1175_117514

theorem average_mark_of_remaining_students
  (n : ℕ) (A : ℕ) (m : ℕ) (B : ℕ) (total_students : n = 10)
  (avg_class : A = 80) (excluded_students : m = 5) (avg_excluded : B = 70) :
  (A * n - B * m) / (n - m) = 90 :=
by
  sorry

end average_mark_of_remaining_students_l1175_117514


namespace integer_points_on_segment_l1175_117522

noncomputable def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

theorem integer_points_on_segment (n : ℕ) (h : 0 < n) :
  f n = if n % 3 = 0 then 2 else 0 :=
by
  sorry

end integer_points_on_segment_l1175_117522


namespace sphere_ratios_l1175_117550

theorem sphere_ratios (r1 r2 : ℝ) (h : r1 / r2 = 1 / 3) :
  (4 * π * r1^2) / (4 * π * r2^2) = 1 / 9 ∧ (4 / 3 * π * r1^3) / (4 / 3 * π * r2^3) = 1 / 27 :=
by
  sorry

end sphere_ratios_l1175_117550


namespace num_men_scenario1_is_15_l1175_117579

-- Definitions based on the conditions
def hours_per_day_scenario1 : ℕ := 9
def days_scenario1 : ℕ := 16
def men_scenario2 : ℕ := 18
def hours_per_day_scenario2 : ℕ := 8
def days_scenario2 : ℕ := 15
def total_work_done : ℕ := men_scenario2 * hours_per_day_scenario2 * days_scenario2

-- Definition of the number of men M in the first scenario
noncomputable def men_scenario1 : ℕ := total_work_done / (hours_per_day_scenario1 * days_scenario1)

-- Statement of desired proof: prove that the number of men in the first scenario is 15
theorem num_men_scenario1_is_15 :
  men_scenario1 = 15 := by
  sorry

end num_men_scenario1_is_15_l1175_117579


namespace rectangles_equal_area_implies_value_l1175_117531

theorem rectangles_equal_area_implies_value (x y : ℝ) (h1 : x < 9) (h2 : y < 4)
  (h3 : x * (4 - y) = y * (9 - x)) : 360 * x / y = 810 :=
by
  -- We only need to state the theorem, the proof is not required.
  sorry

end rectangles_equal_area_implies_value_l1175_117531


namespace books_taken_off_l1175_117521

def books_initially : ℝ := 38.0
def books_remaining : ℝ := 28.0

theorem books_taken_off : books_initially - books_remaining = 10 := by
  sorry

end books_taken_off_l1175_117521


namespace twice_x_plus_one_third_y_l1175_117583

theorem twice_x_plus_one_third_y (x y : ℝ) : 2 * x + (1 / 3) * y = 2 * x + (1 / 3) * y := 
by 
  sorry

end twice_x_plus_one_third_y_l1175_117583


namespace Nicky_profit_l1175_117513

-- Definitions for conditions
def card1_value : ℕ := 8
def card2_value : ℕ := 8
def received_card_value : ℕ := 21

-- The statement to be proven
theorem Nicky_profit : (received_card_value - (card1_value + card2_value)) = 5 :=
by
  sorry

end Nicky_profit_l1175_117513


namespace blueberry_pies_count_l1175_117578

-- Definitions and conditions
def total_pies := 30
def ratio_parts := 10
def pies_per_part := total_pies / ratio_parts
def blueberry_ratio := 3

-- Problem statement
theorem blueberry_pies_count :
  blueberry_ratio * pies_per_part = 9 := by
  -- The solution step that leads to the proof
  sorry

end blueberry_pies_count_l1175_117578


namespace find_a_l1175_117574

theorem find_a (a : ℝ) (h1 : a^2 + 2 * a - 15 = 0) (h2 : a^2 + 4 * a - 5 ≠ 0) :
  a = 3 :=
by
sorry

end find_a_l1175_117574


namespace geometric_sequence_sum_point_on_line_l1175_117532

theorem geometric_sequence_sum_point_on_line
  (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) (r : ℝ)
  (h1 : a 1 = t)
  (h2 : ∀ n : ℕ, a (n + 1) = t * r ^ n)
  (h3 : ∀ n : ℕ, S n = t * (1 - r ^ n) / (1 - r))
  (h4 : ∀ n : ℕ, (S n, a (n + 1)) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 + 1})
  : t = 1 :=
by
  sorry

end geometric_sequence_sum_point_on_line_l1175_117532


namespace calculate_x_minus_y_l1175_117547

theorem calculate_x_minus_y (x y z : ℝ) 
    (h1 : x - y + z = 23) 
    (h2 : x - y - z = 7) : 
    x - y = 15 :=
by
  sorry

end calculate_x_minus_y_l1175_117547


namespace quadruple_equation_solution_count_l1175_117517

theorem quadruple_equation_solution_count (
    a b c d : ℕ
) (h_pos: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_order: a < b ∧ b < c ∧ c < d) 
  (h_equation: 2 * a + 2 * b + 2 * c + 2 * d = d^2 - c^2 + b^2 - a^2) : 
  num_correct_statements = 2 :=
sorry

end quadruple_equation_solution_count_l1175_117517


namespace lisa_time_to_complete_l1175_117576

theorem lisa_time_to_complete 
  (hotdogs_record : ℕ) 
  (eaten_so_far : ℕ) 
  (rate_per_minute : ℕ) 
  (remaining_hotdogs : ℕ) 
  (time_to_complete : ℕ) 
  (h1 : hotdogs_record = 75) 
  (h2 : eaten_so_far = 20) 
  (h3 : rate_per_minute = 11) 
  (h4 : remaining_hotdogs = hotdogs_record - eaten_so_far)
  (h5 : time_to_complete = remaining_hotdogs / rate_per_minute) :
  time_to_complete = 5 :=
sorry

end lisa_time_to_complete_l1175_117576


namespace three_digit_cubes_divisible_by_4_l1175_117533

-- Let's define the conditions in Lean
def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Let's combine these conditions to define the target predicate in Lean
def is_target_number (n : ℕ) : Prop := is_three_digit n ∧ is_perfect_cube n ∧ is_divisible_by_4 n

-- The statement to be proven: that there is only one such number
theorem three_digit_cubes_divisible_by_4 : 
  (∃! n, is_target_number n) :=
sorry

end three_digit_cubes_divisible_by_4_l1175_117533


namespace arrangements_with_AB_together_l1175_117564

theorem arrangements_with_AB_together (students : Finset α) (A B : α) (hA : A ∈ students) (hB : B ∈ students) (h_students : students.card = 5) : 
  ∃ n, n = 48 :=
sorry

end arrangements_with_AB_together_l1175_117564


namespace train_passes_jogger_time_l1175_117573

theorem train_passes_jogger_time (speed_jogger_kmph : ℝ) 
                                (speed_train_kmph : ℝ) 
                                (distance_ahead_m : ℝ) 
                                (length_train_m : ℝ) : 
  speed_jogger_kmph = 9 → 
  speed_train_kmph = 45 →
  distance_ahead_m = 250 →
  length_train_m = 120 →
  (distance_ahead_m + length_train_m) / (speed_train_kmph - speed_jogger_kmph) * (1000 / 3600) = 37 :=
by
  intros h1 h2 h3 h4
  sorry

end train_passes_jogger_time_l1175_117573


namespace distinct_arrangements_TOOL_l1175_117596

/-- The word "TOOL" consists of four letters where "O" is repeated twice. 
Prove that the number of distinct arrangements of the letters in the word is 12. -/
theorem distinct_arrangements_TOOL : 
  let total_letters := 4
  let repeated_O := 2
  (Nat.factorial total_letters / Nat.factorial repeated_O) = 12 := 
by
  sorry

end distinct_arrangements_TOOL_l1175_117596


namespace find_S_l1175_117538

noncomputable def S : ℕ+ → ℝ := sorry
noncomputable def a : ℕ+ → ℝ := sorry

axiom h : ∀ n : ℕ+, 2 * S n = 3 * a n + 4

theorem find_S : ∀ n : ℕ+, S n = 2 - 2 * 3 ^ (n : ℕ) :=
  sorry

end find_S_l1175_117538


namespace both_owners_count_l1175_117529

-- Define the sets and counts as given in the conditions
variable (total_students : ℕ) (rabbit_owners : ℕ) (guinea_pig_owners : ℕ) (both_owners : ℕ)

-- Assume the values given in the problem
axiom total : total_students = 50
axiom rabbits : rabbit_owners = 35
axiom guinea_pigs : guinea_pig_owners = 40

-- The theorem to prove
theorem both_owners_count : both_owners = rabbit_owners + guinea_pig_owners - total_students := by
  sorry

end both_owners_count_l1175_117529


namespace negation_of_p_is_neg_p_l1175_117520

-- Define proposition p
def p : Prop := ∃ x : ℝ, x^2 + x - 1 ≥ 0

-- Define the negation of p
def neg_p : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

theorem negation_of_p_is_neg_p : ¬p = neg_p := by
  -- The proof is omitted as per the instruction
  sorry

end negation_of_p_is_neg_p_l1175_117520


namespace technicians_count_l1175_117582

theorem technicians_count 
  (T R : ℕ) 
  (h1 : T + R = 14) 
  (h2 : 12000 * T + 6000 * R = 9000 * 14) : 
  T = 7 :=
by
  sorry

end technicians_count_l1175_117582


namespace express_x2_y2_z2_in_terms_of_sigma1_sigma2_l1175_117577

variable (x y z : ℝ)
def sigma1 := x + y + z
def sigma2 := x * y + y * z + z * x

theorem express_x2_y2_z2_in_terms_of_sigma1_sigma2 :
  x^2 + y^2 + z^2 = sigma1 x y z ^ 2 - 2 * sigma2 x y z := by
  sorry

end express_x2_y2_z2_in_terms_of_sigma1_sigma2_l1175_117577


namespace find_original_number_l1175_117595

theorem find_original_number (x : ℝ) : ((x - 3) / 6) * 12 = 8 → x = 7 :=
by
  intro h
  sorry

end find_original_number_l1175_117595


namespace expression_value_l1175_117558

-- Step c: Definitions based on conditions
def base1 : ℤ := -2
def exponent1 : ℕ := 4^2
def base2 : ℕ := 1
def exponent2 : ℕ := 3^3

-- The Lean statement for the problem
theorem expression_value :
  base1 ^ exponent1 + base2 ^ exponent2 = 65537 := by
  sorry

end expression_value_l1175_117558


namespace gcd_subtraction_result_l1175_117536

theorem gcd_subtraction_result : gcd 8100 270 - 8 = 262 := by
  sorry

end gcd_subtraction_result_l1175_117536


namespace page_copy_cost_l1175_117552

theorem page_copy_cost (cost_per_4_pages : ℕ) (page_count : ℕ) (dollar_to_cents : ℕ) : cost_per_4_pages = 8 → page_count = 4 → dollar_to_cents = 100 → (1500 * (page_count / cost_per_4_pages) = 750) :=
by
  intros
  sorry

end page_copy_cost_l1175_117552


namespace find_weight_of_sausages_l1175_117542

variable (packages : ℕ) (cost_per_pound : ℕ) (total_cost : ℕ) (total_weight : ℕ) (weight_per_package : ℕ)

-- Defining the given conditions
def jake_buys_packages (packages : ℕ) : Prop := packages = 3
def cost_of_sausages (cost_per_pound : ℕ) : Prop := cost_per_pound = 4
def amount_paid (total_cost : ℕ) : Prop := total_cost = 24

-- Derived condition to find total weight
def total_weight_of_sausages (total_cost : ℕ) (cost_per_pound : ℕ) : ℕ := total_cost / cost_per_pound

-- Derived condition to find weight per package
def weight_of_each_package (total_weight : ℕ) (packages : ℕ) : ℕ := total_weight / packages

-- The theorem statement
theorem find_weight_of_sausages
  (h1 : jake_buys_packages packages)
  (h2 : cost_of_sausages cost_per_pound)
  (h3 : amount_paid total_cost) :
  weight_of_each_package (total_weight_of_sausages total_cost cost_per_pound) packages = 2 :=
by
  sorry  -- Proof placeholder

end find_weight_of_sausages_l1175_117542


namespace find_johns_allowance_l1175_117594

variable (A : ℝ)  -- John's weekly allowance

noncomputable def johns_allowance : Prop :=
  let arcade_spent := (3 / 5) * A
  let remaining_after_arcade := (2 / 5) * A
  let toy_store_spent := (1 / 3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spent
  let final_spent := 0.88
  final_spent = remaining_after_toy_store → A = 3.30

theorem find_johns_allowance : johns_allowance A := by
  sorry

end find_johns_allowance_l1175_117594


namespace sum_of_roots_l1175_117501

theorem sum_of_roots (f : ℝ → ℝ) (h_symmetric : ∀ x, f (3 + x) = f (3 - x)) (h_roots : ∃ (roots : Finset ℝ), roots.card = 6 ∧ ∀ r ∈ roots, f r = 0) : 
  ∃ S, S = 18 :=
by
  sorry

end sum_of_roots_l1175_117501


namespace shares_of_stocks_they_can_buy_l1175_117504

def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def months_of_savings : ℕ := 4
def cost_per_share : ℕ := 50

theorem shares_of_stocks_they_can_buy :
  (((weekly_savings_wife * 4) + monthly_savings_husband) * months_of_savings / 2) / cost_per_share = 25 :=
by
  -- sorry for the implementation
  sorry

end shares_of_stocks_they_can_buy_l1175_117504


namespace total_cost_over_8_weeks_l1175_117549

def cost_per_weekday_edition : ℝ := 0.50
def cost_per_sunday_edition : ℝ := 2.00
def num_weekday_editions_per_week : ℕ := 3
def duration_in_weeks : ℕ := 8

theorem total_cost_over_8_weeks :
  (num_weekday_editions_per_week * cost_per_weekday_edition + cost_per_sunday_edition) * duration_in_weeks = 28.00 := by
  sorry

end total_cost_over_8_weeks_l1175_117549


namespace sum_of_y_for_f_l1175_117556

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + 5

theorem sum_of_y_for_f (y1 y2 y3 : ℝ) :
  (∀ y, 64 * y^3 - 8 * y + 5 = 7) →
  y1 + y2 + y3 = 0 :=
by
  -- placeholder for actual proof
  sorry

end sum_of_y_for_f_l1175_117556


namespace exponent_equality_l1175_117540

theorem exponent_equality (s m : ℕ) (h : (2^16) * (25^s) = 5 * (10^m)) : m = 16 :=
by sorry

end exponent_equality_l1175_117540


namespace percentage_increase_salary_l1175_117567

theorem percentage_increase_salary (S : ℝ) (P : ℝ) (h1 : 1.16 * S = 348) (h2 : S + P * S = 375) : P = 0.25 :=
by
  sorry

end percentage_increase_salary_l1175_117567


namespace tan_ratio_l1175_117527

open Real

variable (x y : ℝ)

-- Conditions
def sin_add_eq : sin (x + y) = 5 / 8 := sorry
def sin_sub_eq : sin (x - y) = 1 / 4 := sorry

-- Proof problem statement
theorem tan_ratio : sin (x + y) = 5 / 8 → sin (x - y) = 1 / 4 → (tan x) / (tan y) = 2 := sorry

end tan_ratio_l1175_117527


namespace contrapositive_example_l1175_117553

theorem contrapositive_example (x : ℝ) (h : x = 1 → x^2 - 3 * x + 2 = 0) :
  x^2 - 3 * x + 2 ≠ 0 → x ≠ 1 :=
by
  intro h₀
  intro h₁
  have h₂ := h h₁
  contradiction

end contrapositive_example_l1175_117553


namespace quadratic_sum_of_coefficients_l1175_117580

theorem quadratic_sum_of_coefficients (x : ℝ) : 
  let a := 1
  let b := 1
  let c := -4
  a + b + c = -2 :=
by
  sorry

end quadratic_sum_of_coefficients_l1175_117580


namespace smallest_positive_integer_solution_l1175_117526

theorem smallest_positive_integer_solution (x : ℕ) (h : 5 * x ≡ 17 [MOD 29]) : x = 15 :=
sorry

end smallest_positive_integer_solution_l1175_117526


namespace average_gas_mileage_round_trip_l1175_117545

noncomputable def average_gas_mileage
  (d1 d2 : ℕ) (m1 m2 : ℕ) : ℚ :=
  let total_distance := d1 + d2
  let total_fuel := (d1 / m1) + (d2 / m2)
  total_distance / total_fuel

theorem average_gas_mileage_round_trip :
  average_gas_mileage 150 180 25 15 = 18.3 := by
  sorry

end average_gas_mileage_round_trip_l1175_117545


namespace remainder_73_to_73_plus73_div137_l1175_117587

theorem remainder_73_to_73_plus73_div137 :
  ((73 ^ 73 + 73) % 137) = 9 := by
  sorry

end remainder_73_to_73_plus73_div137_l1175_117587


namespace no_n_repeats_stock_price_l1175_117528

-- Problem statement translation
theorem no_n_repeats_stock_price (n : ℕ) (h1 : n < 100) : ¬ ∃ k l : ℕ, (100 + n) ^ k * (100 - n) ^ l = 100 ^ (k + l) :=
by
  sorry

end no_n_repeats_stock_price_l1175_117528


namespace intersection_points_form_line_l1175_117559

theorem intersection_points_form_line :
  ∀ (x y : ℝ), ((x * y = 12) ∧ ((x^2 / 16) + (y^2 / 36) = 1)) →
  ∃ (x1 x2 : ℝ) (y1 y2 : ℝ), (x, y) = (x1, y1) ∨ (x, y) = (x2, y2) ∧ (x2 - x1) * (y2 - y1) = x1 * y1 - x2 * y2 :=
by
  sorry

end intersection_points_form_line_l1175_117559


namespace complement_intersection_l1175_117518

open Set

variable {U : Set ℝ} (A B : Set ℝ)

def A_def : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B_def : Set ℝ := { x | 2 < x ∧ x < 10 }

theorem complement_intersection :
  (U = univ ∧ A = A_def ∧ B = B_def) →
  (compl (A ∩ B) = {x | x < 3 ∨ x ≥ 7}) :=
by
  sorry

end complement_intersection_l1175_117518


namespace simplify_expression_l1175_117597

variable (a b : ℝ)

theorem simplify_expression (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (a ^ (7 / 3) - 2 * a ^ (5 / 3) * b ^ (2 / 3) + a * b ^ (4 / 3)) / 
  (a ^ (5 / 3) - a ^ (4 / 3) * b ^ (1 / 3) - a * b ^ (2 / 3) + a ^ (2 / 3) * b) / 
  a ^ (1 / 3) =
  a ^ (1 / 3) + b ^ (1 / 3) :=
sorry

end simplify_expression_l1175_117597
