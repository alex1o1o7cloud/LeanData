import Mathlib

namespace find_number_l1464_146440

theorem find_number (x a_3 a_4 : ℕ) (h1 : x + a_4 = 5574) (h2 : x + a_3 = 557) : x = 5567 :=
  sorry

end find_number_l1464_146440


namespace problem1_problem2_l1464_146479

-- Problem I
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 12 = 0}
def B : Set ℝ := {-2, 4}

theorem problem1 (a : ℝ) (h : A a = B) : a = 2 :=
sorry

-- Problem II
def C (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}
def B' : Set ℝ := {-2, 4}

theorem problem2 (m : ℝ) (h : B' ∪ C m = B') : 
  m = -1/2 ∨ m = -1/4 ∨ m = 0 :=
sorry

end problem1_problem2_l1464_146479


namespace mult_base7_correct_l1464_146472

def base7_to_base10 (n : ℕ) : ℕ :=
  -- assume conversion from base-7 to base-10 is already defined
  sorry 

def base10_to_base7 (n : ℕ) : ℕ :=
  -- assume conversion from base-10 to base-7 is already defined
  sorry

theorem mult_base7_correct : (base7_to_base10 325) * (base7_to_base10 4) = base7_to_base10 1656 :=
by
  sorry

end mult_base7_correct_l1464_146472


namespace max_integer_k_l1464_146411

-- Definitions of the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * x^2 - 2 * x
noncomputable def g' (x : ℝ) : ℝ := x - 2

-- Definition of the inequality condition
theorem max_integer_k (k : ℝ) : 
  (∀ x : ℝ, x > 2 → k * (x - 2) < x * f x + 2 * g' x + 3) ↔
  k ≤ 5 :=
sorry

end max_integer_k_l1464_146411


namespace complement_A_union_B_in_U_l1464_146417

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

end complement_A_union_B_in_U_l1464_146417


namespace quadratic_roots_interval_l1464_146442

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l1464_146442


namespace ear_muffs_total_l1464_146452

theorem ear_muffs_total (a b : ℕ) (h1 : a = 1346) (h2 : b = 6444) : a + b = 7790 :=
by
  sorry

end ear_muffs_total_l1464_146452


namespace crayons_left_l1464_146403

theorem crayons_left (start_crayons lost_crayons left_crayons : ℕ) 
  (h1 : start_crayons = 479) 
  (h2 : lost_crayons = 345) 
  (h3 : left_crayons = start_crayons - lost_crayons) : 
  left_crayons = 134 :=
sorry

end crayons_left_l1464_146403


namespace find_g5_l1464_146483

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end find_g5_l1464_146483


namespace number_of_real_zeros_l1464_146436

def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

theorem number_of_real_zeros : ∃! x : ℝ, f x = 0 := sorry

end number_of_real_zeros_l1464_146436


namespace math_problem_l1464_146441

theorem math_problem
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : x = z * (1 / y)) : 
  (x - z / x) * (y + 1 / (z * y)) = (x^4 - z^3 + x^2 * (z^2 - z)) / (z * x^2) :=
by
  sorry

end math_problem_l1464_146441


namespace ratio_boys_girls_l1464_146478

theorem ratio_boys_girls
  (B G : ℕ)  -- Number of boys and girls
  (h_ratio : 75 * G = 80 * B)
  (h_total_no_scholarship : 100 * (3 * B + 4 * G) = 7772727272727272 * (B + G)) :
  B = 5 * G := sorry

end ratio_boys_girls_l1464_146478


namespace speed_in_still_water_l1464_146451

variable (v_m v_s : ℝ)

def swims_downstream (v_m v_s : ℝ) : Prop :=
  54 = (v_m + v_s) * 3

def swims_upstream (v_m v_s : ℝ) : Prop :=
  18 = (v_m - v_s) * 3

theorem speed_in_still_water : swims_downstream v_m v_s ∧ swims_upstream v_m v_s → v_m = 12 :=
by
  sorry

end speed_in_still_water_l1464_146451


namespace probability_x_y_less_than_3_l1464_146418

theorem probability_x_y_less_than_3 :
  let A := 6 * 2
  let triangle_area := (1 / 2) * 3 * 2
  let P := triangle_area / A
  P = 1 / 4 := by sorry

end probability_x_y_less_than_3_l1464_146418


namespace train_length_is_140_l1464_146402

noncomputable def train_length (speed_kmh : ℕ) (time_s : ℕ) (bridge_length_m : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  let distance := speed_ms * time_s
  distance - bridge_length_m

theorem train_length_is_140 :
  train_length 45 30 235 = 140 := by
  sorry

end train_length_is_140_l1464_146402


namespace quadratic_no_discriminant_23_l1464_146467

theorem quadratic_no_discriminant_23 (a b c : ℤ) (h_eq : b^2 - 4 * a * c = 23) : False := sorry

end quadratic_no_discriminant_23_l1464_146467


namespace product_of_areas_eq_square_of_volume_l1464_146475

theorem product_of_areas_eq_square_of_volume (w : ℝ) :
  let l := 2 * w
  let h := 3 * w
  let A_bottom := l * w
  let A_side := w * h
  let A_front := l * h
  let volume := l * w * h
  A_bottom * A_side * A_front = volume^2 :=
by
  sorry

end product_of_areas_eq_square_of_volume_l1464_146475


namespace part1_part2_part3_l1464_146445

noncomputable def quadratic_has_real_roots (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1^2 - 2*k*x1 + k^2 + k + 1 = 0 ∧ x2^2 - 2*k*x2 + k^2 + k + 1 = 0

theorem part1 (k : ℝ) :
  quadratic_has_real_roots k → k ≤ -1 :=
sorry

theorem part2 (k : ℝ) (x1 x2 : ℝ) :
  quadratic_has_real_roots k ∧ x1^2 + x2^2 = 10 → k = -2 :=
sorry

theorem part3 (k : ℝ) (x1 x2 : ℝ) :
  quadratic_has_real_roots k ∧ (|x1| + |x2| = 2) → k = -1 :=
sorry

end part1_part2_part3_l1464_146445


namespace find_a3_l1464_146450

variable {α : Type} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) :=
  ∃ r : α, ∀ n, a (n + 1) = a n * r

theorem find_a3 (a : ℕ → α) (h : geometric_sequence a) (h1 : a 0 * a 4 = 16) :
  a 2 = 4 ∨ a 2 = -4 :=
by
  sorry

end find_a3_l1464_146450


namespace alpha_add_beta_eq_pi_div_two_l1464_146485

open Real

theorem alpha_add_beta_eq_pi_div_two (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : (sin α) ^ 4 / (cos β) ^ 2 + (cos α) ^ 4 / (sin β) ^ 2 = 1) :
  α + β = π / 2 :=
sorry

end alpha_add_beta_eq_pi_div_two_l1464_146485


namespace binomial_coefficient_x5_l1464_146496

theorem binomial_coefficient_x5 :
  let binomial_term (r : ℕ) : ℕ := Nat.choose 7 r * (21 - 4 * r)
  35 = binomial_term 4 :=
by
  sorry

end binomial_coefficient_x5_l1464_146496


namespace ratio_third_to_others_l1464_146422

-- Definitions of the heights
def H1 := 600
def H2 := 2 * H1
def H3 := 7200 - (H1 + H2)

-- Definition of the ratio to be proved
def ratio := H3 / (H1 + H2)

-- The theorem statement in Lean 4
theorem ratio_third_to_others : ratio = 3 := by
  have hH1 : H1 = 600 := rfl
  have hH2 : H2 = 2 * 600 := rfl
  have hH3 : H3 = 7200 - (600 + 1200) := rfl
  have h_total : 600 + 1200 + H3 = 7200 := sorry
  have h_ratio : (7200 - (600 + 1200)) / (600 + 1200) = 3 := by sorry
  sorry

end ratio_third_to_others_l1464_146422


namespace scientific_notation_l1464_146406

theorem scientific_notation : (10374 * 10^9 : Real) = 1.037 * 10^13 :=
by
  sorry

end scientific_notation_l1464_146406


namespace number_is_2_point_5_l1464_146482

theorem number_is_2_point_5 (x : ℝ) (h: x^2 + 50 = (x - 10)^2) : x = 2.5 := 
by
  sorry

end number_is_2_point_5_l1464_146482


namespace evaluate_expression_l1464_146419

theorem evaluate_expression : 
  (16 = 2^4) → 
  (32 = 2^5) → 
  (16^24 / 32^12 = 8^12) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end evaluate_expression_l1464_146419


namespace simplify_expression_of_triangle_side_lengths_l1464_146420

theorem simplify_expression_of_triangle_side_lengths
  (a b c : ℝ) 
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  |a - b - c| - |c - a + b| = 0 :=
by
  sorry

end simplify_expression_of_triangle_side_lengths_l1464_146420


namespace parallel_lines_if_and_only_if_perpendicular_lines_if_and_only_if_l1464_146423

variable (m x y : ℝ)

def l1_eq : Prop := (3 - m) * x + 2 * m * y + 1 = 0
def l2_eq : Prop := 2 * m * x + 2 * y + m = 0

theorem parallel_lines_if_and_only_if : l1_eq m x y → l2_eq m x y → (m = -3/2) :=
by sorry

theorem perpendicular_lines_if_and_only_if : l1_eq m x y → l2_eq m x y → (m = 0 ∨ m = 5) :=
by sorry

end parallel_lines_if_and_only_if_perpendicular_lines_if_and_only_if_l1464_146423


namespace max_m_value_l1464_146458

theorem max_m_value (a b m : ℝ) (ha : a > 0) (hb : b > 0) (H : (3/a + 1/b) ≥ m / (a + 3 * b)) : m ≤ 12 :=
sorry

end max_m_value_l1464_146458


namespace calculate_expression_l1464_146470

theorem calculate_expression :
  4 * Real.sqrt 24 * (Real.sqrt 6 / 8) / Real.sqrt 3 - 3 * Real.sqrt 3 = -Real.sqrt 3 :=
by
  sorry

end calculate_expression_l1464_146470


namespace dodecagon_area_constraint_l1464_146431

theorem dodecagon_area_constraint 
    (a : ℕ) -- side length of the square
    (N : ℕ) -- a large number with 2017 digits, breaking it down as 2 * (10^2017 - 1) / 9
    (hN : N = (2 * (10^2017 - 1)) / 9) 
    (H : ∃ n : ℕ, (n * n) = 3 * a^2 / 2) :
    False :=
by
    sorry

end dodecagon_area_constraint_l1464_146431


namespace max_distance_between_vertices_l1464_146491

theorem max_distance_between_vertices (inner_perimeter outer_perimeter : ℕ) 
  (inner_perimeter_eq : inner_perimeter = 20) 
  (outer_perimeter_eq : outer_perimeter = 28) : 
  ∃ x y, x + y = 7 ∧ x^2 + y^2 = 25 ∧ (x^2 + (x + y)^2 = 65) :=
by
  sorry

end max_distance_between_vertices_l1464_146491


namespace graph_of_abs_g_l1464_146449

noncomputable def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -1 then x + 3
  else if -1 < x ∧ x ≤ 1 then -x^2 + 2
  else if 1 < x ∧ x ≤ 4 then x - 2
  else 0

noncomputable def abs_g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -3 then -(x + 3)
  else if -3 < x ∧ x ≤ -1 then x + 3
  else if -1 < x ∧ x ≤ 1 then -x^2 + 2
  else if 1 < x ∧ x ≤ 2 then -(x - 2)
  else if 2 < x ∧ x ≤ 4 then x - 2
  else 0

theorem graph_of_abs_g :
  ∀ x : ℝ, abs_g x = |g x| :=
by
  sorry

end graph_of_abs_g_l1464_146449


namespace min_area_triangle_l1464_146407

-- Define the points and line equation
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (30, 10)
def line (x : ℤ) : ℤ := 2 * x - 5

-- Define a function to calculate the area using Shoelace formula
noncomputable def area (C : ℤ × ℤ) : ℝ :=
  (1 / 2) * |(A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1)|

-- Prove that the minimum area of the triangle with the given conditions is 15
theorem min_area_triangle : ∃ (C : ℤ × ℤ), C.2 = line C.1 ∧ area C = 15 := sorry

end min_area_triangle_l1464_146407


namespace max_possible_value_l1464_146434

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end max_possible_value_l1464_146434


namespace number_of_ways_to_choose_teams_l1464_146462

theorem number_of_ways_to_choose_teams : 
  ∃ (n : ℕ), n = Nat.choose 5 2 ∧ n = 10 :=
by
  have h : Nat.choose 5 2 = 10 := by sorry
  use 10
  exact ⟨h, rfl⟩

end number_of_ways_to_choose_teams_l1464_146462


namespace greatest_product_of_two_integers_sum_2006_l1464_146408

theorem greatest_product_of_two_integers_sum_2006 :
  ∃ (x y : ℤ), x + y = 2006 ∧ x * y = 1006009 :=
by
  sorry

end greatest_product_of_two_integers_sum_2006_l1464_146408


namespace prob_bigger_number_correct_l1464_146463

def bernardo_picks := {n | 1 ≤ n ∧ n ≤ 10}
def silvia_picks := {n | 1 ≤ n ∧ n ≤ 8}

noncomputable def prob_bigger_number : ℚ :=
  let prob_bern_picks_10 : ℚ := 3 / 10
  let prob_bern_not_10_larger_silvia : ℚ := 55 / 112
  let prob_bern_not_picks_10 : ℚ := 7 / 10
  prob_bern_picks_10 + prob_bern_not_10_larger_silvia * prob_bern_not_picks_10

theorem prob_bigger_number_correct :
  prob_bigger_number = 9 / 14 := by
  sorry

end prob_bigger_number_correct_l1464_146463


namespace football_daily_practice_hours_l1464_146435

-- Define the total practice hours and the days missed.
def total_hours := 30
def days_missed := 1
def days_in_week := 7

-- Calculate the number of days practiced.
def days_practiced := days_in_week - days_missed

-- Define the daily practice hours.
def daily_practice_hours := total_hours / days_practiced

-- State the proposition.
theorem football_daily_practice_hours :
  daily_practice_hours = 5 := sorry

end football_daily_practice_hours_l1464_146435


namespace relationship_x_y_l1464_146453

theorem relationship_x_y (a b c : ℝ) (h₀ : a > b) (h₁ : b > c) (h₂ : x = Real.sqrt ((a - b) * (b - c))) (h₃ : y = (a - c) / 2) : 
  x ≤ y :=
by
  sorry

end relationship_x_y_l1464_146453


namespace range_of_m_l1464_146413

def p (m : ℝ) : Prop := ∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0)
def q (m : ℝ) : Prop := ∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1 ∨ ∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0))
  ∧ (¬ (∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0) → ∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1)) ↔
  (-1 ≤ m ∧ m ≤ 0) ∨ (1 < m ∧ m < 2) :=
  sorry

end range_of_m_l1464_146413


namespace triplet_not_equal_to_one_l1464_146400

def A := (1/2, 1/3, 1/6)
def B := (2, -2, 1)
def C := (0.1, 0.3, 0.6)
def D := (1.1, -2.1, 1.0)
def E := (-3/2, -5/2, 5)

theorem triplet_not_equal_to_one (ha : A = (1/2, 1/3, 1/6))
                                (hb : B = (2, -2, 1))
                                (hc : C = (0.1, 0.3, 0.6))
                                (hd : D = (1.1, -2.1, 1.0))
                                (he : E = (-3/2, -5/2, 5)) :
  (1/2 + 1/3 + 1/6 = 1) ∧
  (2 + -2 + 1 = 1) ∧
  (0.1 + 0.3 + 0.6 = 1) ∧
  (1.1 + -2.1 + 1.0 ≠ 1) ∧
  (-3/2 + -5/2 + 5 = 1) :=
by {
  sorry
}

end triplet_not_equal_to_one_l1464_146400


namespace probabilities_equal_l1464_146457

def roll := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

def is_successful (r : roll) : Prop := r.val ≥ 3

def prob_successful : ℚ := 4 / 6

def prob_unsuccessful : ℚ := 1 - prob_successful

def prob_at_least_one_success_two_rolls : ℚ := 1 - (prob_unsuccessful ^ 2)

def prob_at_least_two_success_four_rolls : ℚ :=
  let zero_success := prob_unsuccessful ^ 4
  let one_success := 4 * (prob_unsuccessful ^ 3) * prob_successful
  1 - (zero_success + one_success)

theorem probabilities_equal :
  prob_at_least_one_success_two_rolls = prob_at_least_two_success_four_rolls := by
  sorry

end probabilities_equal_l1464_146457


namespace range_of_MF_plus_MN_l1464_146490

open Real

noncomputable def point_on_parabola (x y : ℝ) : Prop := y^2 = 4 * x

theorem range_of_MF_plus_MN (M : ℝ × ℝ) (N : ℝ × ℝ) (F : ℝ × ℝ) (hM : point_on_parabola M.1 M.2) (hN : N = (2, 2)) (hF : F = (1, 0)) :
  ∃ y : ℝ, y ≥ 3 ∧ ∀ MF MN : ℝ, MF = abs (M.1 - F.1) + abs (M.2 - F.2) ∧ MN = abs (M.1 - N.1) + abs (M.2 - N.2) → MF + MN = y :=
sorry

end range_of_MF_plus_MN_l1464_146490


namespace inequality_abc_l1464_146405

theorem inequality_abc 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) : 
  a^2 * b^2 + b^2 * c^2 + a^2 * c^2 ≥ a * b * c * (a + b + c) := 
by 
  sorry

end inequality_abc_l1464_146405


namespace quadratic_distinct_roots_l1464_146492

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end quadratic_distinct_roots_l1464_146492


namespace exists_four_distinct_natural_numbers_sum_any_three_prime_l1464_146414

theorem exists_four_distinct_natural_numbers_sum_any_three_prime :
  ∃ a b c d : ℕ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (Prime (a + b + c) ∧ Prime (a + b + d) ∧ Prime (a + c + d) ∧ Prime (b + c + d)) :=
sorry

end exists_four_distinct_natural_numbers_sum_any_three_prime_l1464_146414


namespace maximum_range_of_temperatures_l1464_146456

variable (T1 T2 T3 T4 T5 : ℝ)

-- Given conditions
def average_condition : Prop := (T1 + T2 + T3 + T4 + T5) / 5 = 50
def lowest_temperature_condition : Prop := T1 = 45

-- Question to prove
def possible_maximum_range : Prop := T5 - T1 = 25

-- The final theorem statement
theorem maximum_range_of_temperatures 
  (h_avg : average_condition T1 T2 T3 T4 T5) 
  (h_lowest : lowest_temperature_condition T1) 
  : possible_maximum_range T1 T5 := by
  sorry

end maximum_range_of_temperatures_l1464_146456


namespace combination_10_3_l1464_146446

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end combination_10_3_l1464_146446


namespace index_cards_per_pack_l1464_146426

-- Definitions of the conditions
def students_per_period := 30
def periods_per_day := 6
def index_cards_per_student := 10
def total_spent := 108
def pack_cost := 3

-- Helper Definitions
def total_students := periods_per_day * students_per_period
def total_index_cards_needed := total_students * index_cards_per_student
def packs_bought := total_spent / pack_cost

-- Theorem to prove
theorem index_cards_per_pack :
  total_index_cards_needed / packs_bought = 50 := by
  sorry

end index_cards_per_pack_l1464_146426


namespace number_of_action_figures_removed_l1464_146464

-- Definitions for conditions
def initial : ℕ := 15
def added : ℕ := 2
def current : ℕ := 10

-- The proof statement
theorem number_of_action_figures_removed (initial added current : ℕ) : 
  (initial + added - current) = 7 := by
  sorry

end number_of_action_figures_removed_l1464_146464


namespace osmotic_pressure_independence_l1464_146494

-- definitions for conditions
def osmotic_pressure_depends_on (osmotic_pressure protein_content Na_content Cl_content : Prop) : Prop :=
  (osmotic_pressure = protein_content ∧ osmotic_pressure = Na_content ∧ osmotic_pressure = Cl_content)

-- statement of the problem to be proved
theorem osmotic_pressure_independence 
  (osmotic_pressure : Prop) 
  (protein_content : Prop) 
  (Na_content : Prop) 
  (Cl_content : Prop) 
  (mw_plasma_protein : Prop)
  (dependence : osmotic_pressure_depends_on osmotic_pressure protein_content Na_content Cl_content) :
  ¬(osmotic_pressure = mw_plasma_protein) :=
sorry

end osmotic_pressure_independence_l1464_146494


namespace similar_triangles_iff_sides_proportional_l1464_146427

theorem similar_triangles_iff_sides_proportional
  (a b c a1 b1 c1 : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < a1 ∧ 0 < b1 ∧ 0 < c1) :
  (Real.sqrt (a * a1) + Real.sqrt (b * b1) + Real.sqrt (c * c1) =
   Real.sqrt ((a + b + c) * (a1 + b1 + c1))) ↔
  (a / a1 = b / b1 ∧ b / b1 = c / c1) :=
by
  sorry

end similar_triangles_iff_sides_proportional_l1464_146427


namespace radius_inscribed_circle_ABC_l1464_146499

noncomputable def radius_of_inscribed_circle (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem radius_inscribed_circle_ABC (hAB : AB = 18) (hAC : AC = 18) (hBC : BC = 24) :
  radius_of_inscribed_circle 18 18 24 = 2 * Real.sqrt 6 := by
  sorry

end radius_inscribed_circle_ABC_l1464_146499


namespace avg_age_assist_coaches_l1464_146401

-- Define the conditions given in the problem

def total_members := 50
def avg_age_total := 22
def girls := 30
def boys := 15
def coaches := 5
def avg_age_girls := 18
def avg_age_boys := 20
def head_coaches := 3
def assist_coaches := 2
def avg_age_head_coaches := 30

-- Define the target theorem to prove
theorem avg_age_assist_coaches : 
  (avg_age_total * total_members - avg_age_girls * girls - avg_age_boys * boys - avg_age_head_coaches * head_coaches) / assist_coaches = 85 := 
  by
    sorry

end avg_age_assist_coaches_l1464_146401


namespace small_boxes_in_large_box_l1464_146410

def number_of_chocolate_bars_in_small_box := 25
def total_number_of_chocolate_bars := 375

theorem small_boxes_in_large_box : total_number_of_chocolate_bars / number_of_chocolate_bars_in_small_box = 15 := by
  sorry

end small_boxes_in_large_box_l1464_146410


namespace moles_of_HCl_needed_l1464_146488

theorem moles_of_HCl_needed : ∀ (moles_KOH : ℕ), moles_KOH = 2 →
  (moles_HCl : ℕ) → moles_HCl = 2 :=
by
  sorry

end moles_of_HCl_needed_l1464_146488


namespace no_2000_digit_perfect_square_with_1999_digits_of_5_l1464_146493

theorem no_2000_digit_perfect_square_with_1999_digits_of_5 :
  ¬ (∃ n : ℕ,
      (Nat.digits 10 n).length = 2000 ∧
      ∃ k : ℕ, n = k * k ∧
      (Nat.digits 10 n).count 5 ≥ 1999) :=
sorry

end no_2000_digit_perfect_square_with_1999_digits_of_5_l1464_146493


namespace min_value_of_m_l1464_146437

noncomputable def a_n (n : ℕ) : ℝ := 2 * 3^(n - 1)
noncomputable def b_n (n : ℕ) : ℝ := 2 * n - 9
noncomputable def c_n (n : ℕ) : ℝ := b_n n / a_n n

theorem min_value_of_m (m : ℝ) : (∀ n : ℕ, c_n n ≤ m) → m ≥ 1/162 :=
by
  sorry

end min_value_of_m_l1464_146437


namespace jesse_gave_pencils_l1464_146461

theorem jesse_gave_pencils (initial_pencils : ℕ) (final_pencils : ℕ) (pencils_given : ℕ) :
  initial_pencils = 78 → final_pencils = 34 → pencils_given = initial_pencils - final_pencils → pencils_given = 44 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jesse_gave_pencils_l1464_146461


namespace sum_of_other_endpoint_coords_l1464_146465

theorem sum_of_other_endpoint_coords (x y : ℝ) (hx : (6 + x) / 2 = 5) (hy : (2 + y) / 2 = 7) : x + y = 16 := 
  sorry

end sum_of_other_endpoint_coords_l1464_146465


namespace andrey_wins_iff_irreducible_fraction_l1464_146474

def is_irreducible_fraction (p : ℝ) : Prop :=
  ∃ m n : ℕ, p = m / 2^n ∧ gcd m (2^n) = 1

def can_reach_0_or_1 (p : ℝ) : Prop :=
  ∀ move : ℝ, ∃ dir : ℝ, (p + dir * move = 0 ∨ p + dir * move = 1)

theorem andrey_wins_iff_irreducible_fraction (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∃ move_sequence : ℕ → ℝ, ∀ n, can_reach_0_or_1 (move_sequence n)) ↔ is_irreducible_fraction p :=
sorry

end andrey_wins_iff_irreducible_fraction_l1464_146474


namespace shorts_cost_l1464_146497

theorem shorts_cost :
  let football_cost := 3.75
  let shoes_cost := 11.85
  let zachary_money := 10
  let additional_needed := 8
  ∃ S, football_cost + shoes_cost + S = zachary_money + additional_needed ∧ S = 2.40 :=
by
  sorry

end shorts_cost_l1464_146497


namespace hiker_walked_distance_first_day_l1464_146429

theorem hiker_walked_distance_first_day (h d_1 d_2 d_3 : ℕ) (H₁ : d_1 = 3 * h)
    (H₂ : d_2 = 4 * (h - 1)) (H₃ : d_3 = 30) (H₄ : d_1 + d_2 + d_3 = 68) :
    d_1 = 18 := 
by 
  sorry

end hiker_walked_distance_first_day_l1464_146429


namespace SUCCESSOR_arrangement_count_l1464_146447

theorem SUCCESSOR_arrangement_count :
  (Nat.factorial 9) / (Nat.factorial 3 * Nat.factorial 2) = 30240 :=
by
  sorry

end SUCCESSOR_arrangement_count_l1464_146447


namespace sum_of_fifths_divisible_by_30_l1464_146471

open BigOperators

theorem sum_of_fifths_divisible_by_30 {a : ℕ → ℕ} {n : ℕ} 
  (h : 30 ∣ ∑ i in Finset.range n, a i) : 
  30 ∣ ∑ i in Finset.range n, (a i) ^ 5 := 
by sorry

end sum_of_fifths_divisible_by_30_l1464_146471


namespace functional_eq_linear_l1464_146421

theorem functional_eq_linear {f : ℝ → ℝ} (h : ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = (x + y) * (f x - f y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end functional_eq_linear_l1464_146421


namespace smallest_area_right_triangle_l1464_146448

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 7) (hb : b = 10): 
  ∃ (A : ℕ), A = 35 :=
  by
    have hab := 1/2 * a * b
    sorry

-- Note: "sorry" is used as a placeholder for the proof.

end smallest_area_right_triangle_l1464_146448


namespace sum_abs_of_roots_l1464_146404

variables {p q r : ℤ}

theorem sum_abs_of_roots:
  p + q + r = 0 →
  p * q + q * r + r * p = -2023 →
  |p| + |q| + |r| = 94 := by
  intro h1 h2
  sorry

end sum_abs_of_roots_l1464_146404


namespace star_property_l1464_146415

-- Define the custom operation
def star (a b : ℝ) : ℝ := a^2 - b

-- Define the property to prove
theorem star_property (x y : ℝ) : star (x - y) (x + y) = x^2 - x - 2 * x * y + y^2 - y :=
by sorry

end star_property_l1464_146415


namespace expression_undefined_at_12_l1464_146424

theorem expression_undefined_at_12 :
  ¬ ∃ x : ℝ, x = 12 ∧ (x^2 - 24 * x + 144 = 0) →
  (∃ y : ℝ, y = (3 * x^3 + 5) / (x^2 - 24 * x + 144)) :=
by
  sorry

end expression_undefined_at_12_l1464_146424


namespace mean_score_of_seniors_l1464_146425

theorem mean_score_of_seniors 
  (s n : ℕ)
  (ms mn : ℝ)
  (h1 : s + n = 120)
  (h2 : n = 2 * s)
  (h3 : ms = 1.5 * mn)
  (h4 : (s : ℝ) * ms + (n : ℝ) * mn = 13200)
  : ms = 141.43 :=
by
  sorry

end mean_score_of_seniors_l1464_146425


namespace kittens_weight_problem_l1464_146412

theorem kittens_weight_problem
  (w_lightest : ℕ)
  (w_heaviest : ℕ)
  (w_total : ℕ)
  (total_lightest : w_lightest = 80)
  (total_heaviest : w_heaviest = 200)
  (total_weight : w_total = 500) :
  ∃ (n : ℕ), n = 11 :=
by sorry

end kittens_weight_problem_l1464_146412


namespace locus_of_midpoint_of_chord_l1464_146495

theorem locus_of_midpoint_of_chord
  (x y : ℝ)
  (hx : (x - 1)^2 + y^2 ≠ 0)
  : (x - 1) * (x - 1) + y * y = 1 :=
by
  sorry

end locus_of_midpoint_of_chord_l1464_146495


namespace find_a_l1464_146489

theorem find_a {S : ℕ → ℤ} (a : ℤ)
  (hS : ∀ n : ℕ, S n = 5 ^ (n + 1) + a) : a = -5 :=
sorry

end find_a_l1464_146489


namespace calc_problem1_calc_problem2_l1464_146468

-- Proof Problem 1
theorem calc_problem1 : 
  (Real.sqrt 3 + 2 * Real.sqrt 2) - (3 * Real.sqrt 3 + Real.sqrt 2) = -2 * Real.sqrt 3 + Real.sqrt 2 := 
by 
  sorry

-- Proof Problem 2
theorem calc_problem2 : 
  Real.sqrt 2 * (Real.sqrt 2 + 1 / Real.sqrt 2) - abs (2 - Real.sqrt 6) = 5 - Real.sqrt 6 := 
by 
  sorry

end calc_problem1_calc_problem2_l1464_146468


namespace total_reading_materials_l1464_146430

theorem total_reading_materials 
  (magazines : ℕ) 
  (newspapers : ℕ) 
  (h_magazines : magazines = 425) 
  (h_newspapers : newspapers = 275) : 
  magazines + newspapers = 700 := 
by 
  sorry

end total_reading_materials_l1464_146430


namespace james_total_money_l1464_146432

section
-- Conditions
def number_of_bills : ℕ := 3
def value_of_each_bill : ℕ := 20
def initial_wallet_amount : ℕ := 75

-- Question:
-- What is the total amount of money James has now?
def total_value_of_bills : ℕ := number_of_bills * value_of_each_bill
def total_money_now : ℕ := initial_wallet_amount + total_value_of_bills

-- Theorem stating that he has $135 now.
theorem james_total_money : total_money_now = 135 := 
  by
    sorry
end

end james_total_money_l1464_146432


namespace lily_petals_l1464_146438

theorem lily_petals (L : ℕ) (h1 : 8 * L + 15 = 63) : L = 6 :=
by sorry

end lily_petals_l1464_146438


namespace range_of_f_l1464_146477

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ |x + 1|

theorem range_of_f : Set.Ioo 0 1 ∪ {1} = {y : ℝ | ∃ x : ℝ, f x = y} :=
by
  sorry

end range_of_f_l1464_146477


namespace K_travel_time_40_miles_l1464_146433

noncomputable def K_time (x : ℝ) : ℝ := 40 / x

theorem K_travel_time_40_miles (x : ℝ) (d : ℝ) (Δt : ℝ)
  (h1 : d = 40)
  (h2 : Δt = 1 / 3)
  (h3 : ∃ (Kmiles_r : ℝ) (Mmiles_r : ℝ), Kmiles_r = x ∧ Mmiles_r = x - 0.5)
  (h4 : ∃ (Ktime : ℝ) (Mtime : ℝ), Ktime = d / x ∧ Mtime = d / (x - 0.5) ∧ Mtime - Ktime = Δt) :
  K_time x = 5 := sorry

end K_travel_time_40_miles_l1464_146433


namespace arrangement_count_l1464_146484

-- Definitions corresponding to the given problem conditions
def numMathBooks : Nat := 3
def numPhysicsBooks : Nat := 2
def numChemistryBooks : Nat := 1
def totalArrangements : Nat := 2592

-- Statement of the theorem
theorem arrangement_count :
  ∃ (numM numP numC : Nat), 
    numM = 3 ∧ 
    numP = 2 ∧ 
    numC = 1 ∧ 
    (numM + numP + numC = 6) ∧ 
    allMathBooksAdjacent ∧ 
    physicsBooksNonAdjacent → 
    totalArrangements = 2592 :=
by
  sorry

end arrangement_count_l1464_146484


namespace cube_edge_length_l1464_146460

theorem cube_edge_length {e : ℝ} (h : 12 * e = 108) : e = 9 :=
by sorry

end cube_edge_length_l1464_146460


namespace polygon_sides_l1464_146487

-- Define the conditions
def side_length : ℝ := 7
def perimeter : ℝ := 42

-- The statement to prove: number of sides is 6
theorem polygon_sides : (perimeter / side_length) = 6 := by
  sorry

end polygon_sides_l1464_146487


namespace value_of_a_l1464_146416

-- Definition of the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

-- Definition of the derivative f'(-1)
def f_prime_at_neg1 (a : ℝ) : ℝ := 3 * a - 6

-- The theorem to prove the value of a
theorem value_of_a (a : ℝ) (h : f_prime_at_neg1 a = 3) : a = 3 :=
by
  sorry

end value_of_a_l1464_146416


namespace coin_combinations_l1464_146459

-- Define the coins and their counts
def one_cent_count := 1
def two_cent_count := 1
def five_cent_count := 1
def ten_cent_count := 4
def fifty_cent_count := 2

-- Define the expected number of different possible amounts
def expected_amounts := 119

-- Prove that the expected number of possible amounts can be achieved given the coins
theorem coin_combinations : 
  (∃ sums : Finset ℕ, 
    sums.card = expected_amounts ∧ 
    (∀ n ∈ sums, n = one_cent_count * 1 + 
                          two_cent_count * 2 + 
                          five_cent_count * 5 + 
                          ten_cent_count * 10 + 
                          fifty_cent_count * 50)) :=
sorry

end coin_combinations_l1464_146459


namespace problem_statement_l1464_146498

-- Define the function f(x)
variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 4) = -f x
axiom increasing_on_0_2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y

-- Theorem to prove
theorem problem_statement : f (-10) < f 40 ∧ f 40 < f 3 :=
by
  sorry

end problem_statement_l1464_146498


namespace braking_distance_non_alcohol_reaction_time_after_alcohol_braking_distance_after_alcohol_increase_in_braking_distance_max_reaction_time_l1464_146476

section braking_distance

variables {t k v s : ℝ}

-- Problem 1
theorem braking_distance_non_alcohol: 
  (t = 0.5) ∧ (v = 10) ∧ (k = 0.1) → (s = t * v + k * v^2) → (s = 15) :=
by intros; sorry

-- Problem 2a
theorem reaction_time_after_alcohol:
  (v = 15) ∧ (s = 52.5) ∧ (k = 0.1) → (s = t * v + k * v^2) → (t = 2) :=
by intros; sorry

-- Problem 2b
theorem braking_distance_after_alcohol:
  (t = 2) ∧ (v = 10) ∧ (k = 0.1) → (s = t * v + k * v^2) → (s = 30) :=
by intros; sorry

-- Problem 2c
theorem increase_in_braking_distance:
  (s_after = 30) ∧ (s_before = 15) → (diff = s_after - s_before) → (diff = 15) :=
by intros; sorry

-- Problem 3
theorem max_reaction_time:
  (v = 12) ∧ (k = 0.1) ∧ (s ≤ 42) → (s = t * v + k * v^2) → (t ≤ 2.3) :=
by intros; sorry

end braking_distance

end braking_distance_non_alcohol_reaction_time_after_alcohol_braking_distance_after_alcohol_increase_in_braking_distance_max_reaction_time_l1464_146476


namespace andy_late_l1464_146486

theorem andy_late
  (school_start : ℕ := 480) -- 8:00 AM in minutes since midnight
  (normal_travel_time : ℕ := 30)
  (red_lights : ℕ := 4)
  (red_light_wait_time : ℕ := 3)
  (construction_wait_time : ℕ := 10)
  (departure_time : ℕ := 435) -- 7:15 AM in minutes since midnight
  : ((school_start - departure_time) < (normal_travel_time + (red_lights * red_light_wait_time) + construction_wait_time)) →
    school_start + (normal_travel_time + (red_lights * red_light_wait_time) + construction_wait_time - (school_start - departure_time)) = school_start + 7 :=
by
  -- This skips the proof part
  sorry

end andy_late_l1464_146486


namespace system_infinite_solutions_l1464_146480

theorem system_infinite_solutions :
  ∃ (x y : ℚ), (3 * x - 4 * y = 5) ∧ (9 * x - 12 * y = 15) ↔ (3 * x - 4 * y = 5) :=
by
  sorry

end system_infinite_solutions_l1464_146480


namespace correct_calculation_l1464_146466

theorem correct_calculation :
  (- (4 + 2 / 3) - (1 + 5 / 6) - (- (18 + 1 / 2)) + (- (13 + 3 / 4))) = - (7 / 4) :=
by 
  sorry

end correct_calculation_l1464_146466


namespace axis_of_symmetry_parabola_l1464_146439

theorem axis_of_symmetry_parabola : 
  ∀ (x : ℝ), 2 * (x - 3)^2 - 5 = 2 * (x - 3)^2 - 5 → (∃ h : ℝ, h = 3 ∧ ∀ x : ℝ, h = 3) :=
by
  sorry

end axis_of_symmetry_parabola_l1464_146439


namespace relationship_abc_l1464_146443

noncomputable def a : ℝ := 5 ^ (Real.log 3.4 / Real.log 3)
noncomputable def b : ℝ := 5 ^ (Real.log 3.6 / Real.log 3)
noncomputable def c : ℝ := (1 / 5) ^ (Real.log 0.5 / Real.log 3)

theorem relationship_abc : b > a ∧ a > c :=
by
  -- Assumptions derived from logarithmic properties.
  have h1 : Real.log 2 < Real.log 3.4 := sorry
  have h2 : Real.log 3.4 < Real.log 3.6 := sorry
  have h3 : Real.log 0.5 < 0 := sorry
  have h4 : Real.log 2 / Real.log 3 = Real.log 2 := sorry
  have h5 : Real.log 0.5 / Real.log 3 = -Real.log 2 := sorry

  -- Monotonicity of exponential function.
  apply And.intro
  { exact sorry }
  { exact sorry }

end relationship_abc_l1464_146443


namespace father_l1464_146455

variable (S F : ℕ)

theorem father's_age (h1 : F = 3 * S + 3) (h2 : F + 3 = 2 * (S + 3) + 10) : F = 33 := by
  sorry

end father_l1464_146455


namespace quadratic_solution_l1464_146481

theorem quadratic_solution (m n x : ℝ)
  (h1 : (x - m)^2 + n = 0) 
  (h2 : ∃ (a b : ℝ), a ≠ b ∧ (x = a ∨ x = b) ∧ (a - m)^2 + n = 0 ∧ (b - m)^2 + n = 0
    ∧ (a = -1 ∨ a = 3) ∧ (b = -1 ∨ b = 3)) :
  x = -3 ∨ x = 1 :=
by {
  sorry
}

end quadratic_solution_l1464_146481


namespace cos_sin_fraction_l1464_146444

theorem cos_sin_fraction (α β : ℝ) (h1 : Real.tan (α + β) = 2 / 5) 
                         (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := 
  sorry

end cos_sin_fraction_l1464_146444


namespace max_gas_tank_capacity_l1464_146469

-- Definitions based on conditions
def start_gas : ℕ := 10
def gas_used_store : ℕ := 6
def gas_used_doctor : ℕ := 2
def refill_needed : ℕ := 10

-- Theorem statement based on the equivalence proof problem
theorem max_gas_tank_capacity : 
  start_gas - (gas_used_store + gas_used_doctor) + refill_needed = 12 :=
by
  -- Proof steps go here
  sorry

end max_gas_tank_capacity_l1464_146469


namespace moles_of_HCl_combined_l1464_146428

/-- Prove the number of moles of Hydrochloric acid combined is 1, given that 
1 mole of Sodium hydroxide and some moles of Hydrochloric acid react to produce 
1 mole of Water, based on the balanced chemical equation: NaOH + HCl → NaCl + H2O -/
theorem moles_of_HCl_combined (moles_NaOH : ℕ) (moles_HCl : ℕ) (moles_H2O : ℕ)
  (h1 : moles_NaOH = 1) (h2 : moles_H2O = 1) 
  (balanced_eq : moles_NaOH = moles_HCl ∧ moles_HCl = moles_H2O) : 
  moles_HCl = 1 :=
by
  sorry

end moles_of_HCl_combined_l1464_146428


namespace find_c_l1464_146409

theorem find_c (c : ℝ)
  (h1 : ∃ y : ℝ, y = (-2)^2 - (-2) + c)
  (h2 : ∃ m : ℝ, m = 2 * (-2) - 1)
  (h3 : ∃ x y, y - (4 + c) = -5 * (x + 2) ∧ x = 0 ∧ y = 0) :
  c = 4 :=
sorry

end find_c_l1464_146409


namespace sum_of_two_numbers_l1464_146473

theorem sum_of_two_numbers (x y : ℕ) (h : x = 11) (h1 : y = 3 * x + 11) : x + y = 55 := by
  sorry

end sum_of_two_numbers_l1464_146473


namespace smallest_missing_unit_digit_l1464_146454

theorem smallest_missing_unit_digit :
  (∀ n, n ∈ [0, 1, 4, 5, 6, 9]) → ∃ smallest_digit, smallest_digit = 2 :=
by
  sorry

end smallest_missing_unit_digit_l1464_146454
