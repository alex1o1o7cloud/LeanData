import Mathlib

namespace min_value_of_z_l7_780

theorem min_value_of_z (a x y : ℝ) (h1 : a > 0) (h2 : x ≥ 1) (h3 : x + y ≤ 3) (h4 : y ≥ a * (x - 3)) :
  (∃ (x y : ℝ), 2 * x + y = 1) → a = 1 / 2 :=
by {
  sorry
}

end min_value_of_z_l7_780


namespace tan_double_angle_identity_l7_719

theorem tan_double_angle_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) : 
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := 
  sorry

end tan_double_angle_identity_l7_719


namespace pleasant_goat_paths_l7_705

-- Define the grid points A, B, and C
structure Point :=
  (x : ℕ)
  (y : ℕ)

def A : Point := { x := 0, y := 0 }
def C : Point := { x := 3, y := 3 }  -- assuming some grid layout
def B : Point := { x := 1, y := 1 }

-- Define a statement to count the number of shortest paths
def shortest_paths_count (A B C : Point) : ℕ := sorry

-- Proving the shortest paths from A to C avoiding B is 81
theorem pleasant_goat_paths : shortest_paths_count A B C = 81 := 
sorry

end pleasant_goat_paths_l7_705


namespace central_angle_of_sector_l7_792

open Real

theorem central_angle_of_sector (l S : ℝ) (α R : ℝ) (hl : l = 4) (hS : S = 4) (h1 : l = α * R) (h2 : S = 1/2 * α * R^2) : 
  α = 2 :=
by
  -- Proof will be supplied here
  sorry

end central_angle_of_sector_l7_792


namespace marta_candies_received_l7_706

theorem marta_candies_received:
  ∃ x y : ℕ, x + y = 200 ∧ x < 100 ∧ x > (4 * y) / 5 ∧ (x % 8 = 0) ∧ (y % 8 = 0) ∧ x = 96 ∧ y = 104 := 
sorry

end marta_candies_received_l7_706


namespace smallest_prime_perimeter_l7_740

def is_prime (n : ℕ) := Nat.Prime n
def is_triangle (a b c : ℕ) := a + b > c ∧ a + c > b ∧ b + c > a
def is_scalene (a b c : ℕ) := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_prime_perimeter :
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ is_scalene a b c ∧ a ≥ 5
  ∧ is_prime (a + b + c) ∧ a + b + c = 23 :=
by
  sorry

end smallest_prime_perimeter_l7_740


namespace problem_not_equivalent_l7_768

theorem problem_not_equivalent :
  (0.0000396 ≠ 3.9 * 10^(-5)) ∧ 
  (0.0000396 = 3.96 * 10^(-5)) ∧ 
  (0.0000396 = 396 * 10^(-7)) ∧ 
  (0.0000396 = (793 / 20000) * 10^(-5)) ∧ 
  (0.0000396 = 198 / 5000000) :=
by
  sorry

end problem_not_equivalent_l7_768


namespace pauly_cannot_make_more_omelets_l7_761

-- Pauly's omelet data
def total_eggs : ℕ := 36
def plain_omelet_eggs : ℕ := 3
def cheese_omelet_eggs : ℕ := 4
def vegetable_omelet_eggs : ℕ := 5

-- Requested omelets
def requested_plain_omelets : ℕ := 4
def requested_cheese_omelets : ℕ := 2
def requested_vegetable_omelets : ℕ := 3

-- Number of eggs used for each type of requested omelet
def total_requested_eggs : ℕ :=
  (requested_plain_omelets * plain_omelet_eggs) +
  (requested_cheese_omelets * cheese_omelet_eggs) +
  (requested_vegetable_omelets * vegetable_omelet_eggs)

-- The remaining number of eggs
def remaining_eggs : ℕ := total_eggs - total_requested_eggs

theorem pauly_cannot_make_more_omelets :
  remaining_eggs < min plain_omelet_eggs (min cheese_omelet_eggs vegetable_omelet_eggs) :=
by
  sorry

end pauly_cannot_make_more_omelets_l7_761


namespace no_possible_arrangement_l7_701

theorem no_possible_arrangement :
  ¬ ∃ (a : Fin 9 → ℕ),
    (∀ i, 1 ≤ a i ∧ a i ≤ 9) ∧
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) % 3 = 0) ∧
    (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) > 12) :=
  sorry

end no_possible_arrangement_l7_701


namespace claire_earnings_l7_765

theorem claire_earnings
  (total_flowers : ℕ)
  (tulips : ℕ)
  (white_roses : ℕ)
  (price_per_red_rose : ℚ)
  (sell_fraction : ℚ)
  (h1 : total_flowers = 400)
  (h2 : tulips = 120)
  (h3 : white_roses = 80)
  (h4 : price_per_red_rose = 0.75)
  (h5 : sell_fraction = 1/2) : 
  (total_flowers - tulips - white_roses) * sell_fraction * price_per_red_rose = 75 :=
by
  sorry

end claire_earnings_l7_765


namespace smallest_prime_dividing_7pow15_plus_9pow17_l7_785

theorem smallest_prime_dividing_7pow15_plus_9pow17 :
  Nat.Prime 2 ∧ (∀ p : ℕ, Nat.Prime p → p ∣ (7^15 + 9^17) → 2 ≤ p) :=
by
  sorry

end smallest_prime_dividing_7pow15_plus_9pow17_l7_785


namespace solution_set_of_inequality_l7_746

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_mono : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_f1_zero : f 1 = 0) : 
  { x | f x > 0 } = { x | x < -1 ∨ 1 < x } := 
by
  sorry

end solution_set_of_inequality_l7_746


namespace two_pairs_of_dice_probability_l7_751

noncomputable def two_pairs_probability : ℚ :=
  5 / 36

theorem two_pairs_of_dice_probability :
  ∃ p : ℚ, p = two_pairs_probability := 
by 
  use 5 / 36
  sorry

end two_pairs_of_dice_probability_l7_751


namespace edges_sum_l7_783

def edges_triangular_pyramid : ℕ := 6
def edges_triangular_prism : ℕ := 9

theorem edges_sum : edges_triangular_pyramid + edges_triangular_prism = 15 :=
by
  sorry

end edges_sum_l7_783


namespace gcd_lcm_product_l7_723

theorem gcd_lcm_product (a b : ℤ) (h1 : Int.gcd a b = 8) (h2 : Int.lcm a b = 24) : a * b = 192 := by
  sorry

end gcd_lcm_product_l7_723


namespace total_votes_l7_779

variable (V : ℝ)

theorem total_votes (h : 0.70 * V - 0.30 * V = 160) : V = 400 := by
  sorry

end total_votes_l7_779


namespace people_stools_chairs_l7_724

def numberOfPeopleStoolsAndChairs (x y z : ℕ) : Prop :=
  2 * x + 3 * y + 4 * z = 32 ∧
  x > y ∧
  x > z ∧
  x < y + z

theorem people_stools_chairs :
  ∃ (x y z : ℕ), numberOfPeopleStoolsAndChairs x y z ∧ x = 5 ∧ y = 2 ∧ z = 4 :=
by
  sorry

end people_stools_chairs_l7_724


namespace max_objective_function_value_l7_777

def objective_function (x1 x2 : ℝ) := 4 * x1 + 6 * x2

theorem max_objective_function_value :
  ∃ x1 x2 : ℝ, 
    (x1 >= 0) ∧ 
    (x2 >= 0) ∧ 
    (x1 + x2 <= 18) ∧ 
    (0.5 * x1 + x2 <= 12) ∧ 
    (2 * x1 <= 24) ∧ 
    (2 * x2 <= 18) ∧ 
    (∀ y1 y2 : ℝ, 
      (y1 >= 0) ∧ 
      (y2 >= 0) ∧ 
      (y1 + y2 <= 18) ∧ 
      (0.5 * y1 + y2 <= 12) ∧ 
      (2 * y1 <= 24) ∧ 
      (2 * y2 <= 18) -> 
      objective_function y1 y2 <= objective_function x1 x2) ∧
    (objective_function x1 x2 = 84) :=
by
  use 12, 6
  sorry

end max_objective_function_value_l7_777


namespace tangent_line_eq_l7_773

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 - 2 * x - 1)

theorem tangent_line_eq :
  let x := 1
  let y := f x
  ∃ (m : ℝ), m = -2 * Real.exp 1 ∧ (∀ (x y : ℝ), y = m * (x - 1) + f 1) :=
by
  sorry

end tangent_line_eq_l7_773


namespace cos_alpha_plus_beta_l7_702

variable (α β : ℝ)
variable (hα : Real.sin α = (Real.sqrt 5) / 5)
variable (hβ : Real.sin β = (Real.sqrt 10) / 10)
variable (hα_obtuse : π / 2 < α ∧ α < π)
variable (hβ_obtuse : π / 2 < β ∧ β < π)

theorem cos_alpha_plus_beta : Real.cos (α + β) = Real.sqrt 2 / 2 ∧ α + β = 7 * π / 4 := by
  sorry

end cos_alpha_plus_beta_l7_702


namespace rahul_share_of_payment_l7_788

def work_rate_rahul : ℚ := 1 / 3
def work_rate_rajesh : ℚ := 1 / 2
def total_payment : ℚ := 150

theorem rahul_share_of_payment : (work_rate_rahul / (work_rate_rahul + work_rate_rajesh)) * total_payment = 60 := by
  sorry

end rahul_share_of_payment_l7_788


namespace ellipse_foci_y_axis_range_l7_739

theorem ellipse_foci_y_axis_range (k : ℝ) :
  (∃ x y : ℝ, x^2 + k * y^2 = 4 ∧ (∃ c1 c2 : ℝ, y = 0 → c1^2 + c2^2 = 4)) ↔ 0 < k ∧ k < 1 :=
by
  sorry

end ellipse_foci_y_axis_range_l7_739


namespace correct_factorization_option_A_l7_712

variable (x y : ℝ)

theorem correct_factorization_option_A :
  (2 * x^2 + 3 * x + 1 = (2 * x + 1) * (x + 1)) :=
by {
  sorry
}

end correct_factorization_option_A_l7_712


namespace polynomial_expansion_abs_sum_l7_716

theorem polynomial_expansion_abs_sum :
  let a_0 := 1
  let a_1 := -8
  let a_2 := 24
  let a_3 := -32
  let a_4 := 16
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| = 81 :=
by
  sorry

end polynomial_expansion_abs_sum_l7_716


namespace find_line_equation_proj_origin_l7_733

theorem find_line_equation_proj_origin (P : ℝ × ℝ) (hP : P = (-2, 1)) :
    ∃ (a b c : ℝ), a * 2 + b * (-1) + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = 5 := 
by
  sorry

end find_line_equation_proj_origin_l7_733


namespace range_m_l7_778

def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

noncomputable def problem :=
  ∀ (m : ℝ), (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 3)

theorem range_m (m : ℝ) : problem := 
  sorry

end range_m_l7_778


namespace monotonicity_and_range_l7_737

noncomputable def f (a x : ℝ) : ℝ := (a * x - 2) * Real.exp x - Real.exp (a - 2)

theorem monotonicity_and_range (a x : ℝ) :
  ( (a = 0 → ∀ x, f a x < f a (x + 1)) ∧
  (a > 0 → ∀ x < (2 - a) / a, f a x < f a (x + 1) ∧ ∀ x > (2 - a) / a, f a x > f a (x + 1) ) ∧
  (a < 0 → ∀ x > (2 - a) / a, f a x < f a (x + 1) ∧ ∀ x < (2 - a) / a, f a x > f a (x + 1) ) ∧
  (∀ x > 1, f a x > 0 ↔ a ∈ Set.Ici 1)) 
:=
sorry

end monotonicity_and_range_l7_737


namespace minimum_value_of_f_l7_738

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) / x

theorem minimum_value_of_f (h : 1 < x) : ∃ y, f x = y ∧ (∀ z, (f z) ≥ 2*sqrt 2) :=
by
  sorry

end minimum_value_of_f_l7_738


namespace positive_integers_satisfying_inequality_l7_731

-- Define the assertion that there are exactly 5 positive integers x satisfying the given inequality
theorem positive_integers_satisfying_inequality :
  (∃! x : ℕ, 4 < x ∧ x < 10 ∧ (10 * x)^4 > x^8 ∧ x^8 > 2^16) :=
sorry

end positive_integers_satisfying_inequality_l7_731


namespace average_chore_time_l7_743

theorem average_chore_time 
  (times : List ℕ := [4, 3, 2, 1, 0])
  (counts : List ℕ := [2, 4, 2, 1, 1]) 
  (total_students : ℕ := 10)
  (total_time : ℕ := List.sum (List.zipWith (λ t c => t * c) times counts)) :
  (total_time : ℚ) / total_students = 2.5 := by
  sorry

end average_chore_time_l7_743


namespace total_husk_is_30_bags_l7_796

-- Define the total number of cows and the number of days.
def numCows : ℕ := 30
def numDays : ℕ := 30

-- Define the rate of consumption: one cow eats one bag in 30 days.
def consumptionRate (cows : ℕ) (days : ℕ) : ℕ := cows / days

-- Define the total amount of husk consumed in 30 days by 30 cows.
def totalHusk (cows : ℕ) (days : ℕ) (rate : ℕ) : ℕ := cows * rate

-- State the problem in a theorem.
theorem total_husk_is_30_bags : totalHusk numCows numDays 1 = 30 := by
  sorry

end total_husk_is_30_bags_l7_796


namespace other_equation_l7_774

-- Define the variables for the length of the rope and the depth of the well
variables (x y : ℝ)

-- Given condition
def cond1 : Prop := (1/4) * x = y + 3

-- The proof goal
theorem other_equation (h : cond1 x y) : (1/5) * x = y + 2 :=
sorry

end other_equation_l7_774


namespace composite_a2_b2_l7_735

-- Introduce the main definitions according to the conditions stated in a)
theorem composite_a2_b2 (x1 x2 : ℕ) (h1 : x1 > 0) (h2 : x2 > 0) (a b : ℤ) 
  (ha : a = -(x1 + x2)) (hb : b = x1 * x2 - 1) : 
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ (a^2 + b^2) = m * n := 
by 
  sorry

end composite_a2_b2_l7_735


namespace negation_of_exists_sin_gt_one_l7_757

theorem negation_of_exists_sin_gt_one : 
  (¬ ∃ x : ℝ, Real.sin x > 1) ↔ (∀ x : ℝ, Real.sin x ≤ 1) := 
by
  sorry

end negation_of_exists_sin_gt_one_l7_757


namespace inequality_solution_set_l7_769

theorem inequality_solution_set : 
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end inequality_solution_set_l7_769


namespace impossible_ratio_5_11_l7_771

theorem impossible_ratio_5_11:
  ∀ (b g: ℕ), 
  b + g ≥ 66 →
  b + 11 = g - 13 →
  ¬(5 * b = 11 * (b + 24) ∧ b ≥ 21) := 
by
  intros b g h1 h2 h3
  sorry

end impossible_ratio_5_11_l7_771


namespace john_growth_l7_710

theorem john_growth 
  (InitialHeight : ℤ)
  (GrowthRate : ℤ)
  (FinalHeight : ℤ)
  (h1 : InitialHeight = 66)
  (h2 : GrowthRate = 2)
  (h3 : FinalHeight = 72) :
  (FinalHeight - InitialHeight) / GrowthRate = 3 :=
by
  sorry

end john_growth_l7_710


namespace isosceles_triangle_perimeter_l7_708

-- Defining the given conditions
def is_isosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ c = a
def triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

-- Stating the problem and goal
theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h_iso: is_isosceles a b c)
  (h_len1: a = 3 ∨ a = 6)
  (h_len2: b = 3 ∨ b = 6)
  (h_triangle: triangle a b c): a + b + c = 15 :=
sorry

end isosceles_triangle_perimeter_l7_708


namespace gym_distance_l7_714

def distance_to_work : ℕ := 10
def distance_to_gym (dist : ℕ) : ℕ := (dist / 2) + 2

theorem gym_distance :
  distance_to_gym distance_to_work = 7 :=
sorry

end gym_distance_l7_714


namespace num_factors_48_l7_770

theorem num_factors_48 : 
  ∀ (n : ℕ), n = 48 → (∃ k : ℕ, k = 10 ∧ ∀ d : ℕ, d ∣ n → (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 16 ∨ d = 24 ∨ d = 48)) :=
  by
    intros n h
    sorry

end num_factors_48_l7_770


namespace subset_implies_range_l7_790

open Set

-- Definitions based on the problem statement
def A : Set ℝ := { x : ℝ | x < 5 }
def B (a : ℝ) : Set ℝ := { x : ℝ | x < a }

-- Theorem statement
theorem subset_implies_range (a : ℝ) (h : A ⊆ B a) : a ≥ 5 :=
sorry

end subset_implies_range_l7_790


namespace product_not_50_l7_772

theorem product_not_50 :
  (1 / 2 * 100 = 50) ∧
  (-5 * -10 = 50) ∧
  ¬(5 * 11 = 50) ∧
  (2 * 25 = 50) ∧
  (5 / 2 * 20 = 50) :=
by
  sorry

end product_not_50_l7_772


namespace factor_expression_l7_793

variable (x : ℤ)

theorem factor_expression : 63 * x - 21 = 21 * (3 * x - 1) := 
by 
  sorry

end factor_expression_l7_793


namespace final_volume_of_syrup_l7_775

-- Definitions based on conditions extracted from step a)
def quarts_to_cups (q : ℚ) : ℚ := q * 4
def reduce_volume (v : ℚ) : ℚ := v / 12
def add_sugar (v : ℚ) (s : ℚ) : ℚ := v + s

theorem final_volume_of_syrup :
  let initial_volume_in_quarts := 6
  let sugar_added := 1
  let initial_volume_in_cups := quarts_to_cups initial_volume_in_quarts
  let reduced_volume := reduce_volume initial_volume_in_cups
  add_sugar reduced_volume sugar_added = 3 :=
by
  sorry

end final_volume_of_syrup_l7_775


namespace sin_neg_765_eq_neg_sqrt2_div_2_l7_720

theorem sin_neg_765_eq_neg_sqrt2_div_2 :
  Real.sin (-765 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_765_eq_neg_sqrt2_div_2_l7_720


namespace find_Z_l7_742

theorem find_Z (Z : ℝ) (h : (100 + 20 / Z) * Z = 9020) : Z = 90 :=
sorry

end find_Z_l7_742


namespace boy_late_l7_734

noncomputable def time_late (D V1 V2 : ℝ) (early : ℝ) : ℝ :=
  let T1 := D / V1
  let T2 := D / V2
  let T1_mins := T1 * 60
  let T2_mins := T2 * 60
  let actual_on_time := T2_mins + early
  T1_mins - actual_on_time

theorem boy_late :
  time_late 2.5 5 10 10 = 5 :=
by
  sorry

end boy_late_l7_734


namespace find_length_of_field_l7_760

variables (L : ℝ) -- Length of the field
variables (width_field : ℝ := 55) -- Width of the field, given as 55 meters.
variables (width_path : ℝ := 2.5) -- Width of the path around the field, given as 2.5 meters.
variables (area_path : ℝ := 1200) -- Area of the path, given as 1200 square meters.

theorem find_length_of_field
  (h : area_path = (L + 2 * width_path) * (width_field + 2 * width_path) - L * width_field)
  : L = 180 :=
by sorry

end find_length_of_field_l7_760


namespace domain_f_l7_745

noncomputable def f (x : ℝ) := Real.sqrt (3 - x) + Real.log (x - 1)

theorem domain_f : { x : ℝ | 1 < x ∧ x ≤ 3 } = { x : ℝ | True } ∩ { x : ℝ | x ≤ 3 } ∩ { x : ℝ | x > 1 } :=
by
  sorry

end domain_f_l7_745


namespace seating_sessions_l7_794

theorem seating_sessions (num_parents num_pupils morning_parents afternoon_parents morning_pupils mid_day_pupils evening_pupils session_capacity total_sessions : ℕ) 
  (h1 : num_parents = 61)
  (h2 : num_pupils = 177)
  (h3 : session_capacity = 44)
  (h4 : morning_parents = 35)
  (h5 : afternoon_parents = 26)
  (h6 : morning_pupils = 65)
  (h7 : mid_day_pupils = 57)
  (h8 : evening_pupils = 55)
  (h9 : total_sessions = 8) :
  ∃ (parent_sessions pupil_sessions : ℕ), 
    parent_sessions + pupil_sessions = total_sessions ∧
    parent_sessions = (morning_parents + session_capacity - 1) / session_capacity + (afternoon_parents + session_capacity - 1) / session_capacity ∧
    pupil_sessions = (morning_pupils + session_capacity - 1) / session_capacity + (mid_day_pupils + session_capacity - 1) / session_capacity + (evening_pupils + session_capacity - 1) / session_capacity := 
by
  sorry

end seating_sessions_l7_794


namespace evaluate_expression_l7_787

theorem evaluate_expression :
  -25 + 7 * ((8 / 4) ^ 2) = 3 :=
by
  sorry

end evaluate_expression_l7_787


namespace least_four_digit_9_heavy_l7_721

def is_9_heavy (n : ℕ) : Prop := n % 9 > 5

def four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

theorem least_four_digit_9_heavy : ∃ n, four_digit n ∧ is_9_heavy n ∧ ∀ m, (four_digit m ∧ is_9_heavy m) → n ≤ m :=
by
  exists 1005
  sorry

end least_four_digit_9_heavy_l7_721


namespace calculation_l7_759

noncomputable def distance_from_sphere_center_to_plane (S P Q R : Point) (r PQ QR RP : ℝ) : ℝ := 
  let a := PQ / 2
  let b := QR / 2
  let c := RP / 2
  let s := (PQ + QR + RP) / 2
  let K := Real.sqrt (s * (s - PQ) * (s - QR) * (s - RP))
  let R := (PQ * QR * RP) / (4 * K)
  Real.sqrt (r^2 - R^2)

theorem calculation 
  (P Q R S : Point) 
  (r : ℝ) 
  (PQ QR RP : ℝ)
  (h1 : PQ = 17)
  (h2 : QR = 18)
  (h3 : RP = 19)
  (h4 : r = 25) :
  distance_from_sphere_center_to_plane S P Q R r PQ QR RP = 35 * Real.sqrt 7 / 8 → 
  ∃ (x y z : ℕ), x + y + z = 50 ∧ (x.gcd z = 1) ∧ ¬ ∃ p : ℕ, Nat.Prime p ∧ p^2 ∣ y := 
by {
  sorry
}

end calculation_l7_759


namespace range_of_a_l7_736

theorem range_of_a (x y : ℝ) (a : ℝ) :
  (0 < x ∧ x ≤ 2) ∧ (0 < y ∧ y ≤ 2) ∧ (x * y = 2) ∧ (6 - 2 * x - y ≥ a * (2 - x) * (4 - y)) →
  a ≤ 1 :=
by sorry

end range_of_a_l7_736


namespace gcd_of_840_and_1764_l7_711

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := 
by {
  sorry
}

end gcd_of_840_and_1764_l7_711


namespace candy_from_sister_is_5_l7_713

noncomputable def candy_received_from_sister (candy_from_neighbors : ℝ) (pieces_per_day : ℝ) (days : ℕ) : ℝ :=
  pieces_per_day * days - candy_from_neighbors

theorem candy_from_sister_is_5 :
  candy_received_from_sister 11.0 8.0 2 = 5.0 :=
by
  sorry

end candy_from_sister_is_5_l7_713


namespace sadies_average_speed_l7_707

def sadie_time : ℝ := 2
def ariana_speed : ℝ := 6
def ariana_time : ℝ := 0.5
def sarah_speed : ℝ := 4
def total_time : ℝ := 4.5
def total_distance : ℝ := 17

theorem sadies_average_speed :
  ((total_distance - ((ariana_speed * ariana_time) + (sarah_speed * (total_time - sadie_time - ariana_time)))) / sadie_time) = 3 := 
by sorry

end sadies_average_speed_l7_707


namespace remainder_when_squared_l7_727

theorem remainder_when_squared (n : ℕ) (h : n % 8 = 6) : (n * n) % 32 = 4 := by
  sorry

end remainder_when_squared_l7_727


namespace geese_flew_away_l7_744

theorem geese_flew_away (initial remaining flown_away : ℕ) (h_initial: initial = 51) (h_remaining: remaining = 23) : flown_away = 28 :=
by
  sorry

end geese_flew_away_l7_744


namespace combine_heaps_l7_764

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l7_764


namespace x_squared_eq_1_iff_x_eq_1_l7_728

theorem x_squared_eq_1_iff_x_eq_1 (x : ℝ) : (x^2 = 1 → x = 1) ↔ false ∧ (x = 1 → x^2 = 1) :=
by
  sorry

end x_squared_eq_1_iff_x_eq_1_l7_728


namespace bricks_required_l7_730

theorem bricks_required (L_courtyard W_courtyard L_brick W_brick : Real)
  (hcourtyard : L_courtyard = 35) 
  (wcourtyard : W_courtyard = 24) 
  (hbrick_len : L_brick = 0.15) 
  (hbrick_wid : W_brick = 0.08) : 
  (L_courtyard * W_courtyard) / (L_brick * W_brick) = 70000 := 
by
  sorry

end bricks_required_l7_730


namespace fraction_spent_by_Rica_is_one_fifth_l7_722

-- Define the conditions
variable (totalPrizeMoney : ℝ) (fractionReceived : ℝ) (amountLeft : ℝ)
variable (h1 : totalPrizeMoney = 1000) (h2 : fractionReceived = 3 / 8) (h3 : amountLeft = 300)

-- Define Rica's original prize money
noncomputable def RicaOriginalPrizeMoney (totalPrizeMoney fractionReceived : ℝ) : ℝ :=
  fractionReceived * totalPrizeMoney

-- Define amount spent by Rica
noncomputable def AmountSpent (originalPrizeMoney amountLeft : ℝ) : ℝ :=
  originalPrizeMoney - amountLeft

-- Define the fraction of prize money spent by Rica
noncomputable def FractionSpent (amountSpent originalPrizeMoney : ℝ) : ℝ :=
  amountSpent / originalPrizeMoney

-- Main theorem to prove
theorem fraction_spent_by_Rica_is_one_fifth :
  let totalPrizeMoney := 1000
  let fractionReceived := 3 / 8
  let amountLeft := 300
  let RicaOriginalPrizeMoney := fractionReceived * totalPrizeMoney
  let AmountSpent := RicaOriginalPrizeMoney - amountLeft
  let FractionSpent := AmountSpent / RicaOriginalPrizeMoney
  FractionSpent = 1 / 5 :=
by {
  -- Proof details are omitted as per instructions
  sorry
}

end fraction_spent_by_Rica_is_one_fifth_l7_722


namespace imaginary_part_of_complex_number_l7_766

open Complex

theorem imaginary_part_of_complex_number :
  ∀ (i : ℂ), i^2 = -1 → im ((2 * I) / (2 + I^3)) = 4 / 5 :=
by
  intro i hi
  sorry

end imaginary_part_of_complex_number_l7_766


namespace find_number_of_hens_l7_749

theorem find_number_of_hens
  (H C : ℕ)
  (h1 : H + C = 48)
  (h2 : 2 * H + 4 * C = 140) :
  H = 26 :=
by
  sorry

end find_number_of_hens_l7_749


namespace find_polynomial_l7_754

noncomputable def polynomial_satisfies_conditions (P : Polynomial ℝ) : Prop :=
  P.eval 0 = 0 ∧ ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1

theorem find_polynomial (P : Polynomial ℝ) (h : polynomial_satisfies_conditions P) : P = Polynomial.X :=
  sorry

end find_polynomial_l7_754


namespace find_a_10_l7_776

def seq (a : ℕ → ℚ) : Prop :=
∀ n, a (n + 1) = 2 * a n / (a n + 2)

def initial_value (a : ℕ → ℚ) : Prop :=
a 1 = 1

theorem find_a_10 (a : ℕ → ℚ) (h1 : initial_value a) (h2 : seq a) : 
  a 10 = 2 / 11 := 
sorry

end find_a_10_l7_776


namespace exists_naturals_l7_799

def sum_of_digits (a : ℕ) : ℕ := sorry

theorem exists_naturals (R : ℕ) (hR : R > 0) :
  ∃ n : ℕ, n > 0 ∧ (sum_of_digits (n^2)) / (sum_of_digits n) = R :=
by
  sorry

end exists_naturals_l7_799


namespace missing_number_in_proportion_l7_717

theorem missing_number_in_proportion (x : ℝ) :
  (2 / x) = ((4 / 3) / (10 / 3)) → x = 5 :=
by sorry

end missing_number_in_proportion_l7_717


namespace fraction_calculation_l7_758

noncomputable def improper_frac_1 : ℚ := 21 / 8
noncomputable def improper_frac_2 : ℚ := 33 / 14
noncomputable def improper_frac_3 : ℚ := 37 / 12
noncomputable def improper_frac_4 : ℚ := 35 / 8
noncomputable def improper_frac_5 : ℚ := 179 / 9

theorem fraction_calculation :
  (improper_frac_1 - (2 / 3) * improper_frac_2) / ((improper_frac_3 + improper_frac_4) / improper_frac_5) = 59 / 21 :=
by
  sorry

end fraction_calculation_l7_758


namespace determine_function_l7_798

noncomputable def functional_solution (f : ℝ → ℝ) : Prop := 
  ∃ (C₁ C₂ : ℝ), ∀ (x : ℝ), 0 < x → f x = C₁ * x + C₂ / x 

theorem determine_function (f : ℝ → ℝ) :
  (∀ (x y : ℝ), 0 < x → 0 < y → (x + 1 / x) * f y = f (x * y) + f (y / x)) →
  functional_solution f :=
sorry

end determine_function_l7_798


namespace jennifer_remaining_money_l7_762

noncomputable def money_spent_on_sandwich (initial_money : ℝ) : ℝ :=
  let sandwich_cost := (1/5) * initial_money
  let discount := (10/100) * sandwich_cost
  sandwich_cost - discount

noncomputable def money_spent_on_ticket (initial_money : ℝ) : ℝ :=
  (1/6) * initial_money

noncomputable def money_spent_on_book (initial_money : ℝ) : ℝ :=
  (1/2) * initial_money

noncomputable def money_after_initial_expenses (initial_money : ℝ) (gift : ℝ) : ℝ :=
  initial_money - money_spent_on_sandwich initial_money - money_spent_on_ticket initial_money - money_spent_on_book initial_money + gift

noncomputable def money_spent_on_cosmetics (remaining_money : ℝ) : ℝ :=
  (1/4) * remaining_money

noncomputable def money_after_cosmetics (remaining_money : ℝ) : ℝ :=
  remaining_money - money_spent_on_cosmetics remaining_money

noncomputable def money_spent_on_tshirt (remaining_money : ℝ) : ℝ :=
  let tshirt_cost := (1/3) * remaining_money
  let tax := (5/100) * tshirt_cost
  tshirt_cost + tax

noncomputable def remaining_money (initial_money : ℝ) (gift : ℝ) : ℝ :=
  let after_initial := money_after_initial_expenses initial_money gift
  let after_cosmetics := after_initial - money_spent_on_cosmetics after_initial
  after_cosmetics - money_spent_on_tshirt after_cosmetics

theorem jennifer_remaining_money : remaining_money 90 30 = 21.35 := by
  sorry

end jennifer_remaining_money_l7_762


namespace inequality_a_b_l7_750

theorem inequality_a_b (a b : ℝ) (h : a > b ∧ b > 0) : (1/a) < (1/b) := 
by
  sorry

end inequality_a_b_l7_750


namespace teacher_age_frequency_l7_732

theorem teacher_age_frequency (f_less_than_30 : ℝ) (f_between_30_and_50 : ℝ) (h1 : f_less_than_30 = 0.3) (h2 : f_between_30_and_50 = 0.5) :
  1 - f_less_than_30 - f_between_30_and_50 = 0.2 :=
by
  rw [h1, h2]
  norm_num

end teacher_age_frequency_l7_732


namespace quadratic_function_series_sum_l7_789

open Real

noncomputable def P (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 7

theorem quadratic_function_series_sum :
  (∀ (x : ℝ), 0 < x ∧ x < 1 →
    (∑' n, P n * x^n) = (16 * x^2 - 11 * x + 7) / (1 - x)^3) :=
sorry

end quadratic_function_series_sum_l7_789


namespace find_certain_number_l7_729

-- Define the given operation a # b
def sOperation (a b : ℝ) : ℝ :=
  a * b - b + b^2

-- State the theorem to find the value of the certain number
theorem find_certain_number (x : ℝ) (h : sOperation 3 x = 48) : x = 6 :=
sorry

end find_certain_number_l7_729


namespace difference_of_digits_l7_747

theorem difference_of_digits (X Y : ℕ) (h1 : 10 * X + Y < 100) 
  (h2 : 72 = (10 * X + Y) - (10 * Y + X)) : (X - Y) = 8 :=
sorry

end difference_of_digits_l7_747


namespace Jose_age_proof_l7_715

-- Definitions based on the conditions
def Inez_age : ℕ := 15
def Zack_age : ℕ := Inez_age + 5
def Jose_age : ℕ := Zack_age - 7

theorem Jose_age_proof : Jose_age = 13 :=
by
  -- Proof omitted
  sorry

end Jose_age_proof_l7_715


namespace remainder_3n_mod_7_l7_767

theorem remainder_3n_mod_7 (n : ℤ) (k : ℤ) (h : n = 7*k + 1) :
  (3 * n) % 7 = 3 := by
  sorry

end remainder_3n_mod_7_l7_767


namespace two_digit_integer_plus_LCM_of_3_4_5_l7_718

theorem two_digit_integer_plus_LCM_of_3_4_5 (x : ℕ) (h1 : 9 < x) (h2 : x < 100) (h3 : ∃ k, x = 60 * k + 2) :
  x = 62 :=
by {
  sorry
}

end two_digit_integer_plus_LCM_of_3_4_5_l7_718


namespace simplify_fraction_l7_784

theorem simplify_fraction : (2 / 520) + (23 / 40) = 301 / 520 := by
  sorry

end simplify_fraction_l7_784


namespace listK_consecutive_integers_count_l7_700

-- Given conditions
def listK := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] -- A list K consisting of consecutive integers
def leastInt : Int := -5 -- The least integer in list K
def rangePosInt : Nat := 5 -- The range of the positive integers in list K

-- The theorem to prove
theorem listK_consecutive_integers_count : listK.length = 11 := by
  -- skipping the proof
  sorry

end listK_consecutive_integers_count_l7_700


namespace find_smallest_number_l7_781

theorem find_smallest_number 
  : ∃ x : ℕ, (x - 18) % 14 = 0 ∧ (x - 18) % 26 = 0 ∧ (x - 18) % 28 = 0 ∧ (x - 18) / Nat.lcm 14 (Nat.lcm 26 28) = 746 ∧ x = 271562 := by
  sorry

end find_smallest_number_l7_781


namespace Suma_work_time_l7_786

theorem Suma_work_time (W : ℝ) (h1 : W > 0) :
  let renu_rate := W / 8
  let combined_rate := W / 4
  let suma_rate := combined_rate - renu_rate
  let suma_time := W / suma_rate
  suma_time = 8 :=
by 
  let renu_rate := W / 8
  let combined_rate := W / 4
  let suma_rate := combined_rate - renu_rate
  let suma_time := W / suma_rate
  exact sorry

end Suma_work_time_l7_786


namespace sum_of_consecutive_integers_product_336_l7_782

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l7_782


namespace product_gcf_lcm_l7_726

def gcf (a b c : Nat) : Nat := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : Nat) : Nat := Nat.lcm (Nat.lcm a b) c

theorem product_gcf_lcm :
  let A := gcf 6 18 24
  let B := lcm 6 18 24
  A * B = 432 :=
by
  let A := gcf 6 18 24
  let B := lcm 6 18 24
  have hA : A = Nat.gcd (Nat.gcd 6 18) 24 := rfl
  have hB : B = Nat.lcm (Nat.lcm 6 18) 24 := rfl
  sorry

end product_gcf_lcm_l7_726


namespace total_blocks_per_day_l7_763

def blocks_to_park : ℕ := 4
def blocks_to_hs : ℕ := 7
def blocks_to_home : ℕ := 11
def walks_per_day : ℕ := 3

theorem total_blocks_per_day :
  (blocks_to_park + blocks_to_hs + blocks_to_home) * walks_per_day = 66 :=
by
  sorry

end total_blocks_per_day_l7_763


namespace largest_fraction_of_consecutive_odds_is_three_l7_753

theorem largest_fraction_of_consecutive_odds_is_three
  (p q r s : ℕ)
  (h1 : 0 < p)
  (h2 : p < q)
  (h3 : q < r)
  (h4 : r < s)
  (h_odd1 : p % 2 = 1)
  (h_odd2 : q % 2 = 1)
  (h_odd3 : r % 2 = 1)
  (h_odd4 : s % 2 = 1)
  (h_consecutive1 : q = p + 2)
  (h_consecutive2 : r = q + 2)
  (h_consecutive3 : s = r + 2) :
  (r + s) / (p + q) = 3 :=
sorry

end largest_fraction_of_consecutive_odds_is_three_l7_753


namespace wheels_on_floor_l7_755

def number_of_wheels (n_people : Nat) (w_per_person : Nat) : Nat :=
  n_people * w_per_person

theorem wheels_on_floor (n_people : Nat) (w_per_person : Nat) (h_people : n_people = 40) (h_wheels : w_per_person = 4) :
  number_of_wheels n_people w_per_person = 160 := by
  sorry

end wheels_on_floor_l7_755


namespace car_distance_l7_725

variable (v_x v_y : ℝ) (Δt_x : ℝ) (d_x : ℝ)

theorem car_distance (h_vx : v_x = 35) (h_vy : v_y = 50) (h_Δt : Δt_x = 1.2)
  (h_dx : d_x = v_x * Δt_x):
  d_x + v_x * (d_x / (v_y - v_x)) = 98 := 
by sorry

end car_distance_l7_725


namespace union_sets_l7_752

def M (a : ℕ) : Set ℕ := {a, 0}
def N : Set ℕ := {1, 2}

theorem union_sets (a : ℕ) (h_inter : M a ∩ N = {2}) : M a ∪ N = {0, 1, 2} :=
by
  sorry

end union_sets_l7_752


namespace find_m_n_l7_797

theorem find_m_n (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) 
  (h1 : m * n ∣ 3 ^ m + 1) (h2 : m * n ∣ 3 ^ n + 1) : 
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) :=
by
  sorry

end find_m_n_l7_797


namespace block3_reaches_target_l7_704

-- Type representing the position of a block on a 3x7 grid
structure Position where
  row : Nat
  col : Nat
  deriving DecidableEq, Repr

-- Defining the initial positions of blocks
def Block1Start : Position := ⟨2, 2⟩
def Block2Start : Position := ⟨3, 5⟩
def Block3Start : Position := ⟨1, 4⟩

-- The target position in the center of the board
def TargetPosition : Position := ⟨3, 5⟩

-- A function to represent if blocks collide or not
def canMove (current : Position) (target : Position) (blocks : List Position) : Prop :=
  target.row < 3 ∧ target.col < 7 ∧ ¬(target ∈ blocks)

-- Main theorem stating the goal
theorem block3_reaches_target : ∃ (steps : Nat → Position), steps 0 = Block3Start ∧ steps 7 = TargetPosition :=
  sorry

end block3_reaches_target_l7_704


namespace equation_of_circle_correct_l7_709

open Real

def equation_of_circle_through_points (x y : ℝ) :=
  x^2 + y^2 - 4 * x - 6 * y

theorem equation_of_circle_correct :
  ∀ (x y : ℝ),
    (equation_of_circle_through_points (0 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (4 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (-1 : ℝ) (1 : ℝ) = 0) →
    (equation_of_circle_through_points x y = 0) :=
by 
  sorry

end equation_of_circle_correct_l7_709


namespace rest_area_location_l7_795

theorem rest_area_location :
  ∃ (rest_area : ℝ), rest_area = 35 + (95 - 35) / 2 :=
by
  -- Here we set the variables for the conditions
  let fifth_exit := 35
  let seventh_exit := 95
  let rest_area := 35 + (95 - 35) / 2
  use rest_area
  sorry

end rest_area_location_l7_795


namespace sufficient_but_not_necessary_condition_l7_748

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a^2 ≠ 4) → (a ≠ 2) ∧ ¬ ((a ≠ 2) → (a^2 ≠ 4)) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l7_748


namespace mustard_found_at_third_table_l7_741

variable (a b T : ℝ)
def found_mustard_at_first_table := (a = 0.25)
def found_mustard_at_second_table := (b = 0.25)
def total_mustard_found := (T = 0.88)

theorem mustard_found_at_third_table
  (h1 : found_mustard_at_first_table a)
  (h2 : found_mustard_at_second_table b)
  (h3 : total_mustard_found T) :
  T - (a + b) = 0.38 := by
  sorry

end mustard_found_at_third_table_l7_741


namespace carter_average_goals_l7_791

theorem carter_average_goals (C : ℝ)
  (h1 : C + (1 / 2) * C + (C - 3) = 7) : C = 4 :=
by
  sorry

end carter_average_goals_l7_791


namespace cone_base_diameter_l7_756

theorem cone_base_diameter {r l : ℝ} 
  (h₁ : π * r * l + π * r^2 = 3 * π) 
  (h₂ : 2 * π * r = π * l) : 
  2 * r = 2 :=
by
  sorry

end cone_base_diameter_l7_756


namespace sum_f_values_l7_703

noncomputable def f : ℝ → ℝ := sorry

axiom odd_property (x : ℝ) : f (-x) = -f (x)
axiom periodicity (x : ℝ) : f (x) = f (x + 4)
axiom f1 : f 1 = -1

theorem sum_f_values : f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry

end sum_f_values_l7_703
