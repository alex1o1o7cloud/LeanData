import Mathlib

namespace regular_polygon_sides_l1156_115656

theorem regular_polygon_sides (angle : ℝ) (h1 : angle = 150) : ∃ n : ℕ, (180 * (n - 2)) / n = angle ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l1156_115656


namespace problem1_problem2_problem3_problem4_l1156_115696

theorem problem1 (x : ℝ) : x^2 - 2 * x + 1 = 0 ↔ x = 1 := 
by sorry

theorem problem2 (x : ℝ) : x^2 + 2 * x - 3 = 0 ↔ x = 1 ∨ x = -3 :=
by sorry

theorem problem3 (x : ℝ) : 2 * x^2 + 5 * x - 1 = 0 ↔ x = (-5 + Real.sqrt 33) / 4 ∨ x = (-5 - Real.sqrt 33) / 4 :=
by sorry

theorem problem4 (x : ℝ) : 2 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 :=
by sorry

end problem1_problem2_problem3_problem4_l1156_115696


namespace abs_difference_l1156_115699

theorem abs_difference (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 8) : 
  |a - b| = 2 * Real.sqrt 10 :=
by
  sorry

end abs_difference_l1156_115699


namespace rebecca_groups_of_eggs_l1156_115608

def eggs : Nat := 16
def group_size : Nat := 2

theorem rebecca_groups_of_eggs : (eggs / group_size) = 8 := by
  sorry

end rebecca_groups_of_eggs_l1156_115608


namespace merchant_articles_l1156_115673

theorem merchant_articles (N CP SP : ℝ) (h1 : N * CP = 16 * SP) (h2 : SP = CP * 1.0625) (h3 : CP ≠ 0) : N = 17 :=
by
  sorry

end merchant_articles_l1156_115673


namespace service_center_location_l1156_115662

theorem service_center_location : 
  ∀ (milepost4 milepost9 : ℕ), 
  milepost4 = 30 → milepost9 = 150 → 
  (∃ milepost_service_center : ℕ, milepost_service_center = milepost4 + ((milepost9 - milepost4) / 2)) → 
  milepost_service_center = 90 :=
by
  intros milepost4 milepost9 h4 h9 hsc
  sorry

end service_center_location_l1156_115662


namespace average_second_pair_l1156_115679

theorem average_second_pair 
  (avg_six : ℝ) (avg_first_pair : ℝ) (avg_third_pair : ℝ) (avg_second_pair : ℝ) 
  (h1 : avg_six = 3.95) 
  (h2 : avg_first_pair = 4.2) 
  (h3 : avg_third_pair = 3.8000000000000007) : 
  avg_second_pair = 3.85 :=
by
  sorry

end average_second_pair_l1156_115679


namespace lifespan_represents_sample_l1156_115601

-- Definitions
def survey_population := 2500
def provinces_and_cities := 11

-- Theorem stating that the lifespan of the urban residents surveyed represents a sample
theorem lifespan_represents_sample
  (number_of_residents : ℕ) (num_provinces : ℕ) 
  (h₁ : number_of_residents = survey_population)
  (h₂ : num_provinces = provinces_and_cities) :
  "Sample" = "Sample" :=
by 
  -- Proof skipped
  sorry

end lifespan_represents_sample_l1156_115601


namespace problem1_problem2_l1156_115620

theorem problem1 : (-(3 / 4) - (5 / 8) + (9 / 12)) * (-24) = 15 := by
  sorry

theorem problem2 : (-1 ^ 6 + |(-2) ^ 3 - 10| - (-3) / (-1) ^ 2023) = 14 := by
  sorry

end problem1_problem2_l1156_115620


namespace compute_difference_l1156_115688

def distinct_solutions (p q : ℝ) : Prop :=
  (p ≠ q) ∧ (∃ (x : ℝ), (x = p ∨ x = q) ∧ (x-3)*(x+3) = 21*x - 63) ∧
  (p > q)

theorem compute_difference (p q : ℝ) (h : distinct_solutions p q) : p - q = 15 :=
by
  sorry

end compute_difference_l1156_115688


namespace find_k_l1156_115613

theorem find_k
  (angle_C : ℝ)
  (AB : ℝ × ℝ)
  (AC : ℝ × ℝ)
  (h1 : angle_C = 90)
  (h2 : AB = (k, 1))
  (h3 : AC = (2, 3)) :
  k = 5 := by
  sorry

end find_k_l1156_115613


namespace digit_H_value_l1156_115676

theorem digit_H_value (E F G H : ℕ) (h1 : E < 10) (h2 : F < 10) (h3 : G < 10) (h4 : H < 10)
  (cond1 : 10 * E + F + 10 * G + E = 10 * H + E)
  (cond2 : 10 * E + F - (10 * G + E) = E)
  (cond3 : E + G = H + 1) : H = 8 :=
sorry

end digit_H_value_l1156_115676


namespace minimum_value_of_f_l1156_115626

-- Define the function y = f(x)
def f (x : ℝ) : ℝ := x^2 + 8 * x + 25

-- We need to prove that the minimum value of f(x) is 9
theorem minimum_value_of_f : ∃ x : ℝ, f x = 9 ∧ ∀ y : ℝ, f y ≥ 9 :=
by
  sorry

end minimum_value_of_f_l1156_115626


namespace horse_buying_problem_l1156_115695

variable (x y z : ℚ)

theorem horse_buying_problem :
  (x + 1/2 * y + 1/2 * z = 12) →
  (y + 1/3 * x + 1/3 * z = 12) →
  (z + 1/4 * x + 1/4 * y = 12) →
  x = 60/17 ∧ y = 136/17 ∧ z = 156/17 :=
by
  sorry

end horse_buying_problem_l1156_115695


namespace counterexample_to_conjecture_l1156_115614

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, 1 < m ∧ m < n → ¬(m ∣ n)

def is_power_of_two (k : ℕ) : Prop := ∃ m : ℕ, m > 0 ∧ k = 2 ^ m

theorem counterexample_to_conjecture :
  ∃ n : ℤ, n > 5 ∧ ¬ (3 ∣ n) ∧ ¬ (∃ p k : ℕ, is_prime p ∧ is_power_of_two k ∧ n = p + k) :=
sorry

end counterexample_to_conjecture_l1156_115614


namespace real_roots_of_polynomial_l1156_115694

theorem real_roots_of_polynomial :
  {x : ℝ | (x^4 - 4*x^3 + 5*x^2 - 2*x + 2) = 0} = {1, -1} :=
sorry

end real_roots_of_polynomial_l1156_115694


namespace rectangle_length_width_l1156_115640

theorem rectangle_length_width 
  (x y : ℚ)
  (h1 : x - 5 = y + 2)
  (h2 : x * y = (x - 5) * (y + 2)) :
  x = 25 / 3 ∧ y = 4 / 3 :=
by
  sorry

end rectangle_length_width_l1156_115640


namespace geometric_progression_theorem_l1156_115680

variables {a b c : ℝ} {n : ℕ} {q : ℝ}

-- Define the terms in the geometric progression
def nth_term (a q : ℝ) (n : ℕ) : ℝ := a * q^n
def second_nth_term (a q : ℝ) (n : ℕ) : ℝ := a * q^(2 * n)
def fourth_nth_term (a q : ℝ) (n : ℕ) : ℝ := a * q^(4 * n)

-- Conditions
axiom nth_term_def : b = nth_term a q n
axiom second_nth_term_def : b = second_nth_term a q n
axiom fourth_nth_term_def : c = fourth_nth_term a q n

-- Statement to prove
theorem geometric_progression_theorem :
  b * (b^2 - a^2) = a^2 * (c - b) :=
sorry

end geometric_progression_theorem_l1156_115680


namespace bus_sarah_probability_l1156_115672

-- Define the probability of Sarah arriving while the bus is still there
theorem bus_sarah_probability :
  let total_minutes := 60
  let bus_waiting_time := 15
  let total_area := (total_minutes * total_minutes : ℕ)
  let triangle_area := (1 / 2 : ℝ) * 45 * 15
  let rectangle_area := 15 * 15
  let shaded_area := triangle_area + rectangle_area
  (shaded_area / total_area : ℝ) = (5 / 32 : ℝ) :=
by
  sorry

end bus_sarah_probability_l1156_115672


namespace min_value_inequality_l1156_115633

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ( (x^2 + 4*x + 2) * (y^2 + 5*y + 3) * (z^2 + 6*z + 4) ) / (x * y * z) ≥ 336 := 
by
  sorry

end min_value_inequality_l1156_115633


namespace markup_percentage_l1156_115667

theorem markup_percentage {C : ℝ} (hC0: 0 < C) (h1: 0 < 1.125 * C) : 
  ∃ (x : ℝ), 0.75 * (1.20 * C * (1 + x / 100)) = 1.125 * C ∧ x = 25 := 
by
  have h2 : 1.20 = (6 / 5 : ℝ) := by norm_num
  have h3 : 0.75 = (3 / 4 : ℝ) := by norm_num
  sorry

end markup_percentage_l1156_115667


namespace problem_g_eq_l1156_115647

noncomputable def g : ℝ → ℝ := sorry

theorem problem_g_eq :
  (∀ x ≠ 0, g x - 3 * g (1 / x) = 3^x + x) →
  g 3 = ( -31 - 3 * 3^(1/3)) / 8 :=
by
  intro h
  -- proof goes here
  sorry

end problem_g_eq_l1156_115647


namespace S_is_multiples_of_six_l1156_115619

-- Defining the problem.
def S : Set ℝ :=
  { t | ∃ n : ℤ, t = 6 * n }

-- We are given that S is non-empty
axiom S_non_empty : ∃ x, x ∈ S

-- Condition: For any x, y ∈ S, both x + y ∈ S and x - y ∈ S.
axiom S_closed_add_sub : ∀ x y, x ∈ S → y ∈ S → (x + y ∈ S ∧ x - y ∈ S)

-- The smallest positive number in S is 6.
axiom S_smallest : ∀ ε, ε > 0 → ∃ x, x ∈ S ∧ x = 6

-- The goal is to prove that S is exactly the set of all multiples of 6.
theorem S_is_multiples_of_six : ∀ t, t ∈ S ↔ ∃ n : ℤ, t = 6 * n :=
by
  sorry

end S_is_multiples_of_six_l1156_115619


namespace floor_problem_solution_l1156_115685

noncomputable def floor_problem (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋

theorem floor_problem_solution :
  { x : ℝ | floor_problem x } = { x : ℝ | 2 ≤ x ∧ x < 7 / 3 } :=
by sorry

end floor_problem_solution_l1156_115685


namespace yards_gained_l1156_115698

variable {G : ℤ}

theorem yards_gained (h : -5 + G = 3) : G = 8 :=
  by
  sorry

end yards_gained_l1156_115698


namespace ratio_of_ages_in_two_years_l1156_115631

-- Define the constants
def son_age : ℕ := 24
def age_difference : ℕ := 26

-- Define the equations based on conditions
def man_age := son_age + age_difference
def son_future_age := son_age + 2
def man_future_age := man_age + 2

-- State the theorem for the required ratio
theorem ratio_of_ages_in_two_years : man_future_age / son_future_age = 2 := by
  sorry

end ratio_of_ages_in_two_years_l1156_115631


namespace trig_identity_l1156_115637

theorem trig_identity : 
  Real.sin (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) + 
  Real.cos (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 :=
by 
  sorry

end trig_identity_l1156_115637


namespace product_divisible_by_49_l1156_115687

theorem product_divisible_by_49 (a b : ℕ) (h : (a^2 + b^2) % 7 = 0) : (a * b) % 49 = 0 :=
sorry

end product_divisible_by_49_l1156_115687


namespace eunsung_sungmin_menu_cases_l1156_115692

theorem eunsung_sungmin_menu_cases :
  let kinds_of_chicken := 4
  let kinds_of_pizza := 3
  let same_chicken_different_pizza :=
    kinds_of_chicken * (kinds_of_pizza * (kinds_of_pizza - 1))
  let same_pizza_different_chicken :=
    kinds_of_pizza * (kinds_of_chicken * (kinds_of_chicken - 1))
  same_chicken_different_pizza + same_pizza_different_chicken = 60 :=
by
  sorry

end eunsung_sungmin_menu_cases_l1156_115692


namespace second_athlete_high_jump_eq_eight_l1156_115636

theorem second_athlete_high_jump_eq_eight :
  let first_athlete_long_jump := 26
  let first_athlete_triple_jump := 30
  let first_athlete_high_jump := 7
  let second_athlete_long_jump := 24
  let second_athlete_triple_jump := 34
  let winner_average_jump := 22
  (first_athlete_long_jump + first_athlete_triple_jump + first_athlete_high_jump) / 3 < winner_average_jump →
  ∃ (second_athlete_high_jump : ℝ), 
    second_athlete_high_jump = 
    (winner_average_jump * 3 - (second_athlete_long_jump + second_athlete_triple_jump)) ∧ 
    second_athlete_high_jump = 8 :=
by
  intros 
  sorry

end second_athlete_high_jump_eq_eight_l1156_115636


namespace golu_distance_travelled_l1156_115678

theorem golu_distance_travelled 
  (b : ℝ) (c : ℝ) (h : c^2 = x^2 + b^2) : x = 8 := by
  sorry

end golu_distance_travelled_l1156_115678


namespace find_root_power_117_l1156_115602

noncomputable def problem (a b c : ℝ) (x1 x2 : ℝ) :=
  (3 * a - b) / c * x1^2 + c * (3 * a + b) / (3 * a - b) = 0 ∧
  (3 * a - b) / c * x2^2 + c * (3 * a + b) / (3 * a - b) = 0 ∧
  x1 + x2 = 0

theorem find_root_power_117 (a b c : ℝ) (x1 x2 : ℝ) (h : problem a b c x1 x2) : 
  x1 ^ 117 + x2 ^ 117 = 0 :=
sorry

end find_root_power_117_l1156_115602


namespace equivalent_proof_problem_l1156_115622

def op (a b : ℝ) : ℝ := (a + b) ^ 2

theorem equivalent_proof_problem (x y : ℝ) : 
  op ((x + y) ^ 2) ((x - y) ^ 2) = 4 * (x ^ 2 + y ^ 2) ^ 2 := 
by 
  sorry

end equivalent_proof_problem_l1156_115622


namespace min_prime_factor_sum_l1156_115649

theorem min_prime_factor_sum (x y a b c d : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : 5 * x^7 = 13 * y^11)
  (h4 : x = 13^6 * 5^7) (h5 : a = 13) (h6 : b = 5) (h7 : c = 6) (h8 : d = 7) : 
  a + b + c + d = 31 :=
by
  sorry

end min_prime_factor_sum_l1156_115649


namespace no_nat_numbers_m_n_satisfy_eq_l1156_115645

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l1156_115645


namespace reciprocal_of_2023_l1156_115669

theorem reciprocal_of_2023 : (2023 : ℝ)⁻¹ = 1 / 2023 :=
by
  sorry

end reciprocal_of_2023_l1156_115669


namespace hyperbola_m_value_l1156_115604

theorem hyperbola_m_value (m k : ℝ) (h₀ : k > 0) (h₁ : 0 < -m) 
  (h₂ : 2 * k = Real.sqrt (1 + m)) : 
  m = -3 := 
by {
  sorry
}

end hyperbola_m_value_l1156_115604


namespace circle_center_radius_l1156_115639

theorem circle_center_radius : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  center = (2, 0) ∧ radius = 2 ∧ ∀ (x y : ℝ), x^2 + y^2 - 4 * x = 0 ↔ (x - 2)^2 + y^2 = 4 :=
by
  sorry

end circle_center_radius_l1156_115639


namespace units_digit_17_pow_35_l1156_115670

theorem units_digit_17_pow_35 : (17 ^ 35) % 10 = 3 := by
sorry

end units_digit_17_pow_35_l1156_115670


namespace completing_the_square_l1156_115683

theorem completing_the_square (x : ℝ) : x^2 + 8*x + 7 = 0 → (x + 4)^2 = 9 :=
by {
  sorry
}

end completing_the_square_l1156_115683


namespace polynomial_roots_expression_l1156_115658

theorem polynomial_roots_expression 
  (a b α β γ δ : ℝ)
  (h1 : α^2 - a*α - 1 = 0)
  (h2 : β^2 - a*β - 1 = 0)
  (h3 : γ^2 - b*γ - 1 = 0)
  (h4 : δ^2 - b*δ - 1 = 0) :
  ((α - γ)^2 * (β - γ)^2 * (α + δ)^2 * (β + δ)^2) = (b^2 - a^2)^2 :=
sorry

end polynomial_roots_expression_l1156_115658


namespace sum_non_solutions_l1156_115600

theorem sum_non_solutions (A B C : ℝ) (h : ∀ x, (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9) → x ≠ -12) :
  -12 = -12 := 
sorry

end sum_non_solutions_l1156_115600


namespace eq_sin_intersect_16_solutions_l1156_115684

theorem eq_sin_intersect_16_solutions :
  ∃ S : Finset ℝ, (∀ x ∈ S, 0 ≤ x ∧ x ≤ 50 ∧ (x / 50 = Real.sin x)) ∧ (S.card = 16) :=
  sorry

end eq_sin_intersect_16_solutions_l1156_115684


namespace initial_number_of_students_l1156_115612

/-- 
Theorem: If the average mark of the students of a class in an exam is 90, and 2 students whose average mark is 45 are excluded, resulting in the average mark of the remaining students being 95, then the initial number of students is 20.
-/
theorem initial_number_of_students (N : ℕ) (T : ℕ)
  (h1 : T = N * 90)
  (h2 : (T - 90) / (N - 2) = 95) : 
  N = 20 :=
sorry

end initial_number_of_students_l1156_115612


namespace candies_problem_max_children_l1156_115681

theorem candies_problem_max_children (u v : ℕ → ℕ) (n : ℕ) :
  (∀ i : ℕ, u i = v i + 2) →
  (∀ i : ℕ, u i + 2 = u (i + 1)) →
  (u (n - 1) / u 0 = 13) →
  n = 25 :=
by
  -- Proof not required as per the instructions.
  sorry

end candies_problem_max_children_l1156_115681


namespace triangle_at_most_one_obtuse_l1156_115664

theorem triangle_at_most_one_obtuse 
  (A B C : ℝ)
  (h_sum : A + B + C = 180) 
  (h_obtuse_A : A > 90) 
  (h_obtuse_B : B > 90) 
  (h_obtuse_C : C > 90) :
  false :=
by 
  sorry

end triangle_at_most_one_obtuse_l1156_115664


namespace initial_quantity_l1156_115625

variables {A : ℝ} -- initial quantity of acidic liquid
variables {W : ℝ} -- quantity of water removed

theorem initial_quantity (h1: A * 0.6 = W + 25) (h2: W = 9) : A = 27 :=
by
  sorry

end initial_quantity_l1156_115625


namespace instantaneous_rate_of_change_at_e_l1156_115677

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem instantaneous_rate_of_change_at_e : deriv f e = 0 := by
  sorry

end instantaneous_rate_of_change_at_e_l1156_115677


namespace brick_fence_depth_l1156_115628

theorem brick_fence_depth (length height total_bricks : ℕ) 
    (h1 : length = 20) 
    (h2 : height = 5) 
    (h3 : total_bricks = 800) : 
    (total_bricks / (4 * length * height) = 2) := 
by
  sorry

end brick_fence_depth_l1156_115628


namespace sandy_books_cost_l1156_115617

theorem sandy_books_cost :
  ∀ (x : ℕ),
  (1280 + 880) / (x + 55) = 18 → 
  x = 65 :=
by
  intros x h
  sorry

end sandy_books_cost_l1156_115617


namespace inverse_function_correct_l1156_115641

noncomputable def f (x : ℝ) : ℝ :=
  (x - 1) ^ 2 + 1

noncomputable def f_inv (y : ℝ) : ℝ :=
  1 - Real.sqrt (y - 1)

theorem inverse_function_correct (x : ℝ) (hx : x ≥ 2) :
  f_inv x = 1 - Real.sqrt (x - 1) ∧ ∀ y : ℝ, (y ≤ 0) → f y = x → y = f_inv x :=
by {
  sorry
}

end inverse_function_correct_l1156_115641


namespace tom_gets_correct_share_l1156_115616

def total_savings : ℝ := 18500.0
def natalie_share : ℝ := 0.35 * total_savings
def remaining_after_natalie : ℝ := total_savings - natalie_share
def rick_share : ℝ := 0.30 * remaining_after_natalie
def remaining_after_rick : ℝ := remaining_after_natalie - rick_share
def lucy_share : ℝ := 0.40 * remaining_after_rick
def remaining_after_lucy : ℝ := remaining_after_rick - lucy_share
def minimum_share : ℝ := 1000.0
def tom_share : ℝ := remaining_after_lucy

theorem tom_gets_correct_share :
  (natalie_share ≥ minimum_share) ∧ (rick_share ≥ minimum_share) ∧ (lucy_share ≥ minimum_share) →
  tom_share = 5050.50 :=
by
  sorry

end tom_gets_correct_share_l1156_115616


namespace judah_crayons_l1156_115659

theorem judah_crayons (karen beatrice gilbert judah : ℕ) 
  (h1 : karen = 128)
  (h2 : karen = 2 * beatrice)
  (h3 : beatrice = 2 * gilbert)
  (h4 : gilbert = 4 * judah) : 
  judah = 8 :=
by
  sorry

end judah_crayons_l1156_115659


namespace program_output_l1156_115653

theorem program_output :
  ∃ a b : ℕ, a = 10 ∧ b = a - 8 ∧ a = a - b ∧ a = 8 :=
by
  let a := 10
  let b := a - 8
  let a := a - b
  use a
  use b
  sorry

end program_output_l1156_115653


namespace find_k_l1156_115657

theorem find_k (k : ℝ) : (∀ x y : ℝ, (x + k * y - 2 * k = 0) → (k * x - (k - 2) * y + 1 = 0) → x * k + y * (-1 / k) + y * 2 = 0) →
  (k = 0 ∨ k = 3) :=
by
  sorry

end find_k_l1156_115657


namespace range_of_a_l1156_115652

-- Define the propositions p and q
def p (a : ℝ) := ∀ x : ℝ, 0 ≤ x → x ≤ 1 → a ≥ Real.exp x
def q (a : ℝ) := ∃ x : ℝ, x^2 + 4 * x + a = 0

-- The proof statement
theorem range_of_a (a : ℝ) : (p a ∧ q a) → a ∈ Set.Icc (Real.exp 1) 4 := by
  intro h
  sorry

end range_of_a_l1156_115652


namespace guess_probability_l1156_115689

-- Definitions based on the problem conditions
def even_digits : Set ℕ := {0, 2, 4, 6, 8}

def possible_attempts : ℕ := (5 * 4) -- A^2_5

def favorable_outcomes : ℕ := (4 * 2) -- C^1_4 * A^2_2

noncomputable def probability_correct_guess : ℝ :=
  (favorable_outcomes : ℝ) / (possible_attempts : ℝ)

-- Lean statement for the proof problem
theorem guess_probability : probability_correct_guess = 2 / 5 := by
  sorry

end guess_probability_l1156_115689


namespace fraction_addition_l1156_115646

theorem fraction_addition : 
  (2 : ℚ) / 5 + (3 : ℚ) / 8 + 1 = 71 / 40 :=
by
  sorry

end fraction_addition_l1156_115646


namespace find_m_for_even_function_l1156_115671

def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + (m + 2) * m * x + 2

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem find_m_for_even_function :
  ∃ m : ℝ, is_even_function (quadratic_function m) ∧ m = -2 :=
by
  sorry

end find_m_for_even_function_l1156_115671


namespace ellipse_and_line_properties_l1156_115615

theorem ellipse_and_line_properties :
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ a * a = 4 ∧ b * b = 3 ∧
  ∀ x y : ℝ, (x, y) = (1, 3/2) → x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∃ k : ℝ, k = 1 / 2 ∧ ∀ x y : ℝ, (x, y) = (2, 1) →
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
  (x1 - 2) * (x2 - 2) + (k * (x1 - 2) + 1 - 1) * (k * (x2 - 2) + 1 - 1) = 5 / 4) :=
sorry

end ellipse_and_line_properties_l1156_115615


namespace number_of_boys_l1156_115642

-- Define the conditions given in the problem
def total_people := 41
def total_amount := 460
def boy_amount := 12
def girl_amount := 8

-- Define the proof statement that needs to be proven
theorem number_of_boys (B G : ℕ) (h1 : B + G = total_people) (h2 : boy_amount * B + girl_amount * G = total_amount) : B = 33 := 
by {
  -- The actual proof will go here
  sorry
}

end number_of_boys_l1156_115642


namespace bob_needs_50_planks_l1156_115648

-- Define the raised bed dimensions and requirements
structure RaisedBedDimensions where
  height : ℕ -- in feet
  width : ℕ  -- in feet
  length : ℕ -- in feet

def plank_length : ℕ := 8  -- length of each plank in feet
def plank_width : ℕ := 1  -- width of each plank in feet
def num_beds : ℕ := 10

def planks_needed (bed : RaisedBedDimensions) : ℕ :=
  let long_sides := 2  -- 2 long sides per bed
  let short_sides := 2 * (bed.width / plank_length)  -- 1/4 plank per short side if width is 2 feet
  let total_sides := long_sides + short_sides
  let stacked_sides := total_sides * (bed.height / plank_width)  -- stacked to match height
  stacked_sides

def raised_bed : RaisedBedDimensions := {height := 2, width := 2, length := 8}

theorem bob_needs_50_planks : planks_needed raised_bed * num_beds = 50 := by
  sorry

end bob_needs_50_planks_l1156_115648


namespace length_PR_l1156_115643

variable (P Q R : Type) [Inhabited P] [Inhabited Q] [Inhabited R]
variable {xPR xQR xsinR : ℝ}
variable (hypotenuse_opposite_ratio : xsinR = (3/5))
variable (sideQR : xQR = 9)
variable (rightAngle : ∀ (P Q R : Type), P ≠ Q → Q ∈ line_through Q R)

theorem length_PR : (∃ xPR : ℝ, xPR = 15) :=
by
  sorry

end length_PR_l1156_115643


namespace original_number_of_people_l1156_115650

/-- Initially, one-third of the people in a room left.
Then, one-fourth of those remaining started to dance.
There were then 18 people who were not dancing.
What was the original number of people in the room? -/
theorem original_number_of_people (x : ℕ) 
  (h_one_third_left : ∀ y : ℕ, 2 * y / 3 = x) 
  (h_one_fourth_dancing : ∀ y : ℕ, y / 4 = x) 
  (h_non_dancers : x / 2 = 18) : 
  x = 36 :=
sorry

end original_number_of_people_l1156_115650


namespace value_of_expression_l1156_115623

variable {a : ℝ}

theorem value_of_expression (h : a^2 + 2 * a - 1 = 0) : 2 * a^2 + 4 * a - 2024 = -2022 :=
by
  sorry

end value_of_expression_l1156_115623


namespace prove_f_three_eq_neg_three_l1156_115624

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.tan x + 1

theorem prove_f_three_eq_neg_three (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -3 := by
  sorry

end prove_f_three_eq_neg_three_l1156_115624


namespace chess_piece_problem_l1156_115610

theorem chess_piece_problem
  (a b c : ℕ)
  (h1 : b = b * 2 - a)
  (h2 : c = c * 2)
  (h3 : a = a * 2 - b)
  (h4 : c = c * 2 - a + b)
  (h5 : a * 2 = 16)
  (h6 : b * 2 = 16)
  (h7 : c * 2 = 16) : 
  a = 26 ∧ b = 14 ∧ c = 8 := 
sorry

end chess_piece_problem_l1156_115610


namespace find_a_l1156_115635

noncomputable def f (x : ℝ) : ℝ := x^2 + 10

noncomputable def g (x : ℝ) : ℝ := x^2 - 6

theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : f (g a) = 12) :
    a = Real.sqrt (6 + Real.sqrt 2) ∨ a = Real.sqrt (6 - Real.sqrt 2) :=
sorry

end find_a_l1156_115635


namespace scrooge_mcduck_max_box_l1156_115618

-- Define Fibonacci numbers
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

-- The problem statement: for a given positive integer k (number of coins initially),
-- the maximum box index n into which Scrooge McDuck can place a coin
-- is F_{k+2} - 1.
theorem scrooge_mcduck_max_box (k : ℕ) (h_pos : 0 < k) :
  ∃ n, n = fib (k + 2) - 1 :=
sorry

end scrooge_mcduck_max_box_l1156_115618


namespace abs_triangle_inequality_l1156_115603

theorem abs_triangle_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| :=
by sorry

end abs_triangle_inequality_l1156_115603


namespace approximate_number_of_fish_in_pond_l1156_115654

-- Define the conditions as hypotheses.
def tagged_fish_caught_first : ℕ := 50
def total_fish_caught_second : ℕ := 50
def tagged_fish_found_second : ℕ := 5

-- Define total fish in the pond.
def total_fish_in_pond (N : ℝ) : Prop :=
  tagged_fish_found_second / total_fish_caught_second = tagged_fish_caught_first / N

-- The statement to be proved.
theorem approximate_number_of_fish_in_pond (N : ℝ) (h : total_fish_in_pond N) : N = 500 :=
sorry

end approximate_number_of_fish_in_pond_l1156_115654


namespace calculate_k_l1156_115690

theorem calculate_k (β : ℝ) (hβ : (Real.tan β + 1 / Real.tan β) ^ 2 = k + 1) : k = 1 := by
  sorry

end calculate_k_l1156_115690


namespace parallelogram_area_example_l1156_115693

def point := (ℚ × ℚ)
def parallelogram_area (A B C D : point) : ℚ :=
  let base := B.1 - A.1
  let height := C.2 - A.2
  base * height

theorem parallelogram_area_example : 
  parallelogram_area (1, 1) (7, 1) (4, 9) (10, 9) = 48 := by
  sorry

end parallelogram_area_example_l1156_115693


namespace third_number_hcf_lcm_l1156_115666

theorem third_number_hcf_lcm (N : ℕ) 
  (HCF : Nat.gcd (Nat.gcd 136 144) N = 8)
  (LCM : Nat.lcm (Nat.lcm 136 144) N = 2^4 * 3^2 * 17 * 7) : 
  N = 7 := 
  sorry

end third_number_hcf_lcm_l1156_115666


namespace nearest_integer_to_x_plus_2y_l1156_115691

theorem nearest_integer_to_x_plus_2y
  (x y : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (h1 : |x| + 2 * y = 6)
  (h2 : |x| * y + x^3 = 2) :
  Int.floor (x + 2 * y + 0.5) = 6 :=
by sorry

end nearest_integer_to_x_plus_2y_l1156_115691


namespace inequality_not_true_l1156_115697

variable {x y : ℝ}

theorem inequality_not_true (h : x > y) : ¬(-3 * x + 6 > -3 * y + 6) :=
by
  sorry

end inequality_not_true_l1156_115697


namespace min_trips_required_l1156_115668

def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def load_capacity : ℕ := 190

theorem min_trips_required :
  ∃ (trips : ℕ), 
  (∀ partition : List (List ℕ), (∀ group : List ℕ, group ∈ partition → 
  group.sum ≤ load_capacity) ∧ partition.join = masses → 
  partition.length ≥ 6) :=
sorry

end min_trips_required_l1156_115668


namespace angle_in_fourth_quadrant_l1156_115675

theorem angle_in_fourth_quadrant (θ : ℝ) (hθ : θ = 300) : 270 < θ ∧ θ < 360 :=
by
  -- theta equals 300
  have h1 : θ = 300 := hθ
  -- check that 300 degrees lies between 270 and 360
  sorry

end angle_in_fourth_quadrant_l1156_115675


namespace min_deliveries_l1156_115661

theorem min_deliveries (cost_per_delivery_income: ℕ) (cost_per_delivery_gas: ℕ) (van_cost: ℕ) (d: ℕ) : 
  (d * (cost_per_delivery_income - cost_per_delivery_gas) ≥ van_cost) ↔ (d ≥ van_cost / (cost_per_delivery_income - cost_per_delivery_gas)) :=
by
  sorry

def john_deliveries : ℕ := 7500 / (15 - 5)

example : john_deliveries = 750 :=
by
  sorry

end min_deliveries_l1156_115661


namespace xy_addition_equals_13_l1156_115674

theorem xy_addition_equals_13 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt_15 : x < 15) (hy_lt_15 : y < 15) (hxy : x + y + x * y = 49) : x + y = 13 :=
by
  sorry

end xy_addition_equals_13_l1156_115674


namespace tom_tim_typing_ratio_l1156_115660

theorem tom_tim_typing_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) :
  M / T = 5 :=
by
  -- Proof to be completed
  sorry

end tom_tim_typing_ratio_l1156_115660


namespace alice_average_speed_l1156_115605

/-- Alice cycled 40 miles at 8 miles per hour and 20 miles at 40 miles per hour. 
    The average speed for the entire trip --/
theorem alice_average_speed :
  let distance1 := 40
  let speed1 := 8
  let distance2 := 20
  let speed2 := 40
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  (total_distance / total_time) = (120 / 11) := 
by
  sorry -- proof steps would go here

end alice_average_speed_l1156_115605


namespace probability_different_colors_l1156_115663

theorem probability_different_colors :
  let total_chips := 16
  let prob_blue := (7 : ℚ) / total_chips
  let prob_yellow := (5 : ℚ) / total_chips
  let prob_red := (4 : ℚ) / total_chips
  let prob_blue_then_nonblue := prob_blue * ((prob_yellow + prob_red) : ℚ)
  let prob_yellow_then_non_yellow := prob_yellow * ((prob_blue + prob_red) : ℚ)
  let prob_red_then_non_red := prob_red * ((prob_blue + prob_yellow) : ℚ)
  let total_prob := prob_blue_then_nonblue + prob_yellow_then_non_yellow + prob_red_then_non_red
  total_prob = (83 : ℚ) / 128 := 
by
  sorry

end probability_different_colors_l1156_115663


namespace last_term_arithmetic_progression_eq_62_l1156_115607

theorem last_term_arithmetic_progression_eq_62
  (a : ℕ) (d : ℕ) (n : ℕ) 
  (h_a : a = 2)
  (h_d : d = 2)
  (h_n : n = 31) : 
  a + (n - 1) * d = 62 :=
by
  sorry

end last_term_arithmetic_progression_eq_62_l1156_115607


namespace real_solutions_of_fraction_eqn_l1156_115638

theorem real_solutions_of_fraction_eqn (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 7) :
  ( x = 3 + Real.sqrt 3 ∨ x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5 ) ↔
    ((x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)) / ((x - 3) * (x - 7) * (x - 3)) = 1 :=
sorry

end real_solutions_of_fraction_eqn_l1156_115638


namespace volume_of_rectangular_prism_l1156_115630

theorem volume_of_rectangular_prism (l w h : ℕ) (x : ℕ) 
  (h_ratio : l = 3 * x ∧ w = 2 * x ∧ h = x)
  (h_edges : 4 * l + 4 * w + 4 * h = 72) : 
  l * w * h = 162 := 
by
  sorry

end volume_of_rectangular_prism_l1156_115630


namespace triangle_side_length_l1156_115655

theorem triangle_side_length (P Q R : Type) (cos_Q : ℝ) (PQ QR : ℝ) 
  (sin_Q : ℝ) (h_cos_Q : cos_Q = 0.6) (h_PQ : PQ = 10) (h_sin_Q : sin_Q = 0.8) : 
  QR = 50 / 3 :=
by
  sorry

end triangle_side_length_l1156_115655


namespace min_ticket_gates_l1156_115686

theorem min_ticket_gates (a x y : ℕ) (h_pos: a > 0) :
  (a = 30 * x) ∧ (y = 2 * x) → ∃ n : ℕ, (n ≥ 4) ∧ (a + 5 * x ≤ 5 * n * y) :=
by
  sorry

end min_ticket_gates_l1156_115686


namespace number_of_pupils_wrong_entry_l1156_115634

theorem number_of_pupils_wrong_entry 
  (n : ℕ) (A : ℝ) 
  (h_wrong_entry : ∀ m, (m = 85 → n * (A + 1 / 2) = n * A + 52))
  (h_increase : ∀ m, (m = 33 → n * (A + 1 / 2) = n * A + 52)) 
  : n = 104 := 
sorry

end number_of_pupils_wrong_entry_l1156_115634


namespace suraya_picked_more_apples_l1156_115606

theorem suraya_picked_more_apples (suraya caleb kayla : ℕ) 
  (h1 : suraya = caleb + 12)
  (h2 : caleb = kayla - 5)
  (h3 : kayla = 20) : suraya - kayla = 7 := by
  sorry

end suraya_picked_more_apples_l1156_115606


namespace fraction_division_l1156_115644

theorem fraction_division:
  (1 / 4) / (1 / 8) = 2 :=
by
  sorry

end fraction_division_l1156_115644


namespace total_movies_seen_l1156_115621

theorem total_movies_seen (d h a c : ℕ) (hd : d = 7) (hh : h = 12) (ha : a = 15) (hc : c = 2) :
  (c + (d - c) + (h - c) + (a - c)) = 30 :=
by
  sorry

end total_movies_seen_l1156_115621


namespace part_I_part_II_l1156_115651

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - (a * x) / (x + 1)

theorem part_I (a : ℝ) : (∀ x, f a 0 ≤ f a x) → a = 1 := by
  sorry

theorem part_II (a : ℝ) : (∀ x > 0, f a x > 0) → a ≤ 1 := by
  sorry

end part_I_part_II_l1156_115651


namespace Isabel_subtasks_remaining_l1156_115632

-- Definition of the known quantities
def Total_problems : ℕ := 72
def Completed_problems : ℕ := 32
def Subtasks_per_problem : ℕ := 5

-- Definition of the calculations
def Total_subtasks : ℕ := Total_problems * Subtasks_per_problem
def Completed_subtasks : ℕ := Completed_problems * Subtasks_per_problem
def Remaining_subtasks : ℕ := Total_subtasks - Completed_subtasks

-- The theorem we need to prove
theorem Isabel_subtasks_remaining : Remaining_subtasks = 200 := by
  -- Proof would go here, but we'll use sorry to indicate it's omitted
  sorry

end Isabel_subtasks_remaining_l1156_115632


namespace general_formula_sum_b_l1156_115665

-- Define the arithmetic sequence
def arithmetic_sequence (a d: ℕ) (n: ℕ) := a + (n - 1) * d

-- Given conditions
def a1 : ℕ := 1
def d : ℕ := 2
def a (n : ℕ) : ℕ := arithmetic_sequence a1 d n
def b (n : ℕ) : ℕ := 2 ^ a n

-- Formula for the arithmetic sequence
theorem general_formula (n : ℕ) : a n = 2 * n - 1 := 
by sorry

-- Sum of the first n terms of b_n
theorem sum_b (n : ℕ) : (Finset.range n).sum b = (2 / 3) * (4 ^ n - 1) :=
by sorry

end general_formula_sum_b_l1156_115665


namespace tangent_line_at_point_l1156_115609

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = x^3 - 2 * x + 1) (h_point : (x, y) = (1, 0)) :
  y = x - 1 :=
sorry

end tangent_line_at_point_l1156_115609


namespace jim_total_weight_per_hour_l1156_115627

theorem jim_total_weight_per_hour :
  let hours := 8
  let gold_chest := 100
  let gold_bag := 50
  let gold_extra := 30 + 20 + 10
  let silver := 30
  let bronze := 50
  let weight_gold := 10
  let weight_silver := 5
  let weight_bronze := 2
  let total_gold := gold_chest + 2 * gold_bag + gold_extra
  let total_weight := total_gold * weight_gold + silver * weight_silver + bronze * weight_bronze
  total_weight / hours = 356.25 := by
  sorry

end jim_total_weight_per_hour_l1156_115627


namespace marks_in_mathematics_l1156_115611

-- Definitions for the given conditions in the problem
def marks_in_english : ℝ := 86
def marks_in_physics : ℝ := 82
def marks_in_chemistry : ℝ := 87
def marks_in_biology : ℝ := 81
def average_marks : ℝ := 85
def number_of_subjects : ℕ := 5

-- Defining the total marks based on the provided conditions
def total_marks : ℝ := average_marks * number_of_subjects

-- Proving that the marks in mathematics are 89
theorem marks_in_mathematics : total_marks - (marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology) = 89 :=
by
  sorry

end marks_in_mathematics_l1156_115611


namespace smallest_value_l1156_115629

theorem smallest_value (x : ℝ) (h : 3 * x^2 + 33 * x - 90 = x * (x + 18)) : x ≥ -10.5 :=
sorry

end smallest_value_l1156_115629


namespace solve_equation_l1156_115682

theorem solve_equation (x : ℝ) (h : x ≠ -1) :
  (x = -1 / 2 ∨ x = 2) ↔ (∃ x : ℝ, x ≠ -1 ∧ (x^3 - x^2)/(x^2 + 2*x + 1) + x = -2) :=
sorry

end solve_equation_l1156_115682
