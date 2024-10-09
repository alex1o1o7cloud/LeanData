import Mathlib

namespace distance_between_two_cars_l1105_110509

theorem distance_between_two_cars 
    (initial_distance : ℝ) 
    (first_car_distance1 : ℝ) 
    (first_car_distance2 : ℝ)
    (second_car_distance : ℝ) 
    (final_distance : ℝ) :
    initial_distance = 150 →
    first_car_distance1 = 25 →
    first_car_distance2 = 25 →
    second_car_distance = 35 →
    final_distance = initial_distance - (first_car_distance1 + first_car_distance2 + second_car_distance) →
    final_distance = 65 :=
by
  intros h_initial h_first1 h_first2 h_second h_final
  sorry

end distance_between_two_cars_l1105_110509


namespace Yan_ratio_distance_l1105_110525

theorem Yan_ratio_distance (w x y : ℕ) (h : w > 0) (h_eq : y/w = x/w + (x + y)/(5 * w)) : x/y = 2/3 := by
  sorry

end Yan_ratio_distance_l1105_110525


namespace find_c_l1105_110584

theorem find_c 
  (a b c : ℝ) 
  (h_vertex : ∀ x y, y = a * x^2 + b * x + c → 
    (∃ k l, l = b / (2 * a) ∧ k = a * l^2 + b * l + c ∧ k = 3 ∧ l = -2))
  (h_pass : ∀ x y, y = a * x^2 + b * x + c → 
    (x = 2 ∧ y = 7)) : c = 4 :=
by sorry

end find_c_l1105_110584


namespace number_of_cows_l1105_110567

theorem number_of_cows (C H : ℕ) (hcnd : 4 * C + 2 * H = 2 * (C + H) + 18) :
  C = 9 :=
sorry

end number_of_cows_l1105_110567


namespace problem_l1105_110518

noncomputable def f (ω x : ℝ) : ℝ := (Real.sin (ω * x / 2))^2 + (1 / 2) * Real.sin (ω * x) - 1 / 2

theorem problem (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, x ∈ Set.Ioo (Real.pi : ℝ) (2 * Real.pi) → f ω x ≠ 0) →
  ω ∈ Set.Icc 0 (1 / 8) ∪ Set.Icc (1 / 4) (5 / 8) :=
by
  sorry

end problem_l1105_110518


namespace students_in_class_l1105_110500

theorem students_in_class
  (B : ℕ) (E : ℕ) (G : ℕ)
  (h1 : B = 12)
  (h2 : G + B = 22)
  (h3 : E = 10) :
  G + E + B = 32 :=
by
  sorry

end students_in_class_l1105_110500


namespace restore_price_by_percentage_l1105_110589

theorem restore_price_by_percentage 
  (p : ℝ) -- original price
  (h₀ : p > 0) -- condition that price is positive
  (r₁ : ℝ := 0.25) -- reduction of 25%
  (r₁_applied : ℝ := p * (1 - r₁)) -- first reduction
  (r₂ : ℝ := 0.20) -- additional reduction of 20%
  (r₂_applied : ℝ := r₁_applied * (1 - r₂)) -- second reduction
  (final_price : ℝ := r₂_applied) -- final price after two reductions
  (increase_needed : ℝ := p - final_price) -- amount to increase to restore the price
  (percent_increase : ℝ := (increase_needed / final_price) * 100) -- percentage increase needed
  : abs (percent_increase - 66.67) < 0.01 := -- proof that percentage increase is approximately 66.67%
sorry

end restore_price_by_percentage_l1105_110589


namespace positive_integer_condition_l1105_110539

theorem positive_integer_condition (n : ℕ) (h : 15 * n = n^2 + 56) : n = 8 :=
sorry

end positive_integer_condition_l1105_110539


namespace num_ways_award_medals_l1105_110557

-- There are 8 sprinters in total
def num_sprinters : ℕ := 8

-- Three of the sprinters are Americans
def num_americans : ℕ := 3

-- The number of non-American sprinters
def num_non_americans : ℕ := num_sprinters - num_americans

-- The question to prove: the number of ways the medals can be awarded if at most one American gets a medal
theorem num_ways_award_medals 
  (n : ℕ) (m : ℕ) (k : ℕ) (h1 : n = num_sprinters) (h2 : m = num_americans) 
  (h3 : k = num_non_americans) 
  (no_american : ℕ := k * (k - 1) * (k - 2)) 
  (one_american : ℕ := m * 3 * k * (k - 1)) 
  : no_american + one_american = 240 :=
sorry

end num_ways_award_medals_l1105_110557


namespace initial_bananas_tree_l1105_110549

-- Definitions for the conditions
def bananas_left_on_tree : ℕ := 100
def bananas_eaten_by_raj : ℕ := 70
def bananas_in_basket_of_raj := 2 * bananas_eaten_by_raj
def bananas_cut_from_tree := bananas_eaten_by_raj + bananas_in_basket_of_raj
def initial_bananas_on_tree := bananas_cut_from_tree + bananas_left_on_tree

-- The theorem to be proven
theorem initial_bananas_tree : initial_bananas_on_tree = 310 :=
by sorry

end initial_bananas_tree_l1105_110549


namespace sequence_formula_l1105_110553

theorem sequence_formula (a : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a n ≠ 0)
  (h2 : a 1 = 1)
  (h3 : ∀ n : ℕ, n > 0 → a (n + 1) = 1 / (n + 1 + 1 / (a n))) :
  ∀ n : ℕ, n > 0 → a n = 2 / ((n : ℝ) ^ 2 - n + 2) :=
by
  sorry

end sequence_formula_l1105_110553


namespace daniel_utility_equation_solution_l1105_110535

theorem daniel_utility_equation_solution (t : ℚ) :
  t * (10 - t) = (4 - t) * (t + 4) → t = 8 / 5 := by
  sorry

end daniel_utility_equation_solution_l1105_110535


namespace find_A_l1105_110554

theorem find_A :
  ∃ A B C D : ℕ, A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
               A * B = 72 ∧ C * D = 72 ∧
               A + B = C - D ∧ A = 4 :=
by
  sorry

end find_A_l1105_110554


namespace b_distance_behind_proof_l1105_110599

-- Given conditions
def race_distance : ℕ := 1000
def a_time : ℕ := 40
def b_delay : ℕ := 10

def a_speed : ℕ := race_distance / a_time
def b_distance_behind : ℕ := a_speed * b_delay

theorem b_distance_behind_proof : b_distance_behind = 250 := by
  -- Prove that b_distance_behind = 250
  sorry

end b_distance_behind_proof_l1105_110599


namespace total_minutes_exercised_l1105_110534

-- Defining the conditions
def Javier_minutes_per_day : Nat := 50
def Javier_days : Nat := 10

def Sanda_minutes_day_90 : Nat := 90
def Sanda_days_90 : Nat := 3

def Sanda_minutes_day_75 : Nat := 75
def Sanda_days_75 : Nat := 2

def Sanda_minutes_day_45 : Nat := 45
def Sanda_days_45 : Nat := 4

-- Main statement to prove
theorem total_minutes_exercised : 
  (Javier_minutes_per_day * Javier_days) + 
  (Sanda_minutes_day_90 * Sanda_days_90) +
  (Sanda_minutes_day_75 * Sanda_days_75) +
  (Sanda_minutes_day_45 * Sanda_days_45) = 1100 := by
  sorry

end total_minutes_exercised_l1105_110534


namespace steve_ate_bags_l1105_110566

-- Given conditions
def total_macaroons : Nat := 12
def weight_per_macaroon : Nat := 5
def num_bags : Nat := 4
def total_weight_remaining : Nat := 45

-- Derived conditions
def total_weight_macaroons : Nat := total_macaroons * weight_per_macaroon
def macaroons_per_bag : Nat := total_macaroons / num_bags
def weight_per_bag : Nat := macaroons_per_bag * weight_per_macaroon
def bags_remaining : Nat := total_weight_remaining / weight_per_bag

-- Proof statement
theorem steve_ate_bags : num_bags - bags_remaining = 1 := by
  sorry

end steve_ate_bags_l1105_110566


namespace horse_food_needed_l1105_110532

theorem horse_food_needed
  (ratio_sheep_horses : ℕ := 6)
  (ratio_horses_sheep : ℕ := 7)
  (horse_food_per_day : ℕ := 230)
  (sheep_on_farm : ℕ := 48)
  (units : ℕ := sheep_on_farm / ratio_sheep_horses)
  (horses_on_farm : ℕ := units * ratio_horses_sheep) :
  horses_on_farm * horse_food_per_day = 12880 := by
  sorry

end horse_food_needed_l1105_110532


namespace parametric_function_f_l1105_110533

theorem parametric_function_f (f : ℚ → ℚ)
  (x y : ℝ) (t : ℚ) :
  y = 20 * t - 10 →
  y = (3 / 4 : ℝ) * x - 15 →
  x = f t →
  f t = (80 / 3) * t + 20 / 3 :=
by
  sorry

end parametric_function_f_l1105_110533


namespace parabolas_equation_l1105_110516

theorem parabolas_equation (vertex_origin : (0, 0) ∈ {(x, y) | y = x^2} ∨ (0, 0) ∈ {(x, y) | x = -y^2})
  (focus_on_axis : ∀ F : ℝ × ℝ, (F ∈ {(x, y) | y = x^2} ∨ F ∈ {(x, y) | x = -y^2}) → (F.1 = 0 ∨ F.2 = 0))
  (through_point : (-2, 4) ∈ {(x, y) | y = x^2} ∨ (-2, 4) ∈ {(x, y) | x = -y^2}) :
  {(x, y) | y = x^2} ∪ {(x, y) | x = -y^2} ≠ ∅ :=
by
  sorry

end parabolas_equation_l1105_110516


namespace part1_part2_l1105_110592

noncomputable def point_M (m : ℝ) : ℝ × ℝ := (2 * m + 1, m - 4)
def point_N : ℝ × ℝ := (5, 2)

theorem part1 (m : ℝ) (h : m - 4 = 2) : point_M m = (13, 2) := by
  sorry

theorem part2 (m : ℝ) (h : 2 * m + 1 = 3) : point_M m = (3, -3) := by
  sorry

end part1_part2_l1105_110592


namespace problem1_problem2_l1105_110544

-- Problem 1
theorem problem1 : 
  (-2.8) - (-3.6) + (-1.5) - (3.6) = -4.3 := 
by 
  sorry

-- Problem 2
theorem problem2 :
  (- (5 / 6 : ℚ) + (1 / 3 : ℚ) - (3 / 4 : ℚ)) * (-24) = 30 := 
by 
  sorry

end problem1_problem2_l1105_110544


namespace max_magnitude_vector_sub_l1105_110531

open Real

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
sqrt (v.1^2 + v.2^2)

noncomputable def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
(v1.1 - v2.1, v1.2 - v2.2)

theorem max_magnitude_vector_sub (a b : ℝ × ℝ)
  (ha : vector_magnitude a = 2)
  (hb : vector_magnitude b = 1) :
  ∃ θ : ℝ, |vector_magnitude (vector_sub a b)| = 3 :=
by
  use π  -- θ = π to minimize cos θ to be -1
  sorry

end max_magnitude_vector_sub_l1105_110531


namespace ratio_sheep_to_horses_l1105_110586

theorem ratio_sheep_to_horses (sheep horses : ℕ) (total_horse_food daily_food_per_horse : ℕ)
  (h1 : sheep = 16)
  (h2 : total_horse_food = 12880)
  (h3 : daily_food_per_horse = 230)
  (h4 : horses = total_horse_food / daily_food_per_horse) :
  (sheep / gcd sheep horses) / (horses / gcd sheep horses) = 2 / 7 := by
  sorry

end ratio_sheep_to_horses_l1105_110586


namespace total_students_is_correct_l1105_110508

-- Define the number of students in each class based on the conditions
def number_of_students_finley := 24
def number_of_students_johnson := (number_of_students_finley / 2) + 10
def number_of_students_garcia := 2 * number_of_students_johnson
def number_of_students_smith := number_of_students_finley / 3
def number_of_students_patel := (3 / 4) * (number_of_students_finley + number_of_students_johnson + number_of_students_garcia)

-- Define the total number of students in all five classes combined
def total_number_of_students := 
  number_of_students_finley + 
  number_of_students_johnson + 
  number_of_students_garcia +
  number_of_students_smith + 
  number_of_students_patel

-- The theorem statement to prove
theorem total_students_is_correct : total_number_of_students = 166 := by
  sorry

end total_students_is_correct_l1105_110508


namespace no_nondegenerate_triangle_l1105_110515

def distinct_positive_integers (a b c : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def nondegenerate_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem no_nondegenerate_triangle (a b c : ℕ)
  (h_distinct : distinct_positive_integers a b c)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 1)
  (h1 : a ∣ (b - c) ^ 2)
  (h2 : b ∣ (c - a) ^ 2)
  (h3 : c ∣ (a - b) ^ 2) :
  ¬nondegenerate_triangle a b c :=
sorry

end no_nondegenerate_triangle_l1105_110515


namespace inradius_of_triangle_l1105_110540

theorem inradius_of_triangle (p A r : ℝ) (h1 : p = 20) (h2 : A = 25) : r = 2.5 :=
sorry

end inradius_of_triangle_l1105_110540


namespace count_possible_values_l1105_110579

open Nat

def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

def is_valid_addition (A B C D : ℕ) : Prop :=
  ∀ x y z w v u : ℕ, 
  (x = A) ∧ (y = B) ∧ (z = C) ∧ (w = D) ∧ (v = B) ∧ (u = D) →
  (A + C = D) ∧ (A + D = B) ∧ (B + B = D) ∧ (D + D = C)

theorem count_possible_values : ∀ (A B C D : ℕ), 
  distinct_digits A B C D → is_valid_addition A B C D → num_of_possible_D = 4 :=
by
  intro A B C D hd hv
  sorry

end count_possible_values_l1105_110579


namespace find_sum_of_relatively_prime_integers_l1105_110568

theorem find_sum_of_relatively_prime_integers :
  ∃ (x y : ℕ), x * y + x + y = 119 ∧ x < 25 ∧ y < 25 ∧ Nat.gcd x y = 1 ∧ x + y = 20 :=
by
  sorry

end find_sum_of_relatively_prime_integers_l1105_110568


namespace solve_inequality_l1105_110574

open Set

theorem solve_inequality (x : ℝ) (h : -3 * x^2 + 5 * x + 4 < 0 ∧ x > 0) : x ∈ Ioo 0 1 := by
  sorry

end solve_inequality_l1105_110574


namespace flowers_per_bouquet_l1105_110590

theorem flowers_per_bouquet (narcissus chrysanthemums bouquets : ℕ) 
  (h1: narcissus = 75) 
  (h2: chrysanthemums = 90) 
  (h3: bouquets = 33) 
  : (narcissus + chrysanthemums) / bouquets = 5 := 
by 
  sorry

end flowers_per_bouquet_l1105_110590


namespace prove_b_is_neg_two_l1105_110519

-- Define the conditions
variables (b : ℝ)

-- Hypothesis: The real and imaginary parts of the complex number (2 - b * I) * I are opposites
def complex_opposite_parts (b : ℝ) : Prop :=
  b = -2

-- The theorem statement
theorem prove_b_is_neg_two : complex_opposite_parts b :=
sorry

end prove_b_is_neg_two_l1105_110519


namespace profit_percentage_l1105_110569

theorem profit_percentage (cost_price marked_price : ℝ) (discount_rate : ℝ) 
  (h1 : cost_price = 66.5) (h2 : marked_price = 87.5) (h3 : discount_rate = 0.05) : 
  (100 * ((marked_price * (1 - discount_rate) - cost_price) / cost_price)) = 25 :=
by
  sorry

end profit_percentage_l1105_110569


namespace cost_of_27_lilies_l1105_110597

theorem cost_of_27_lilies
  (cost_18 : ℕ)
  (price_ratio : ℕ → ℕ → Prop)
  (h_cost_18 : cost_18 = 30)
  (h_price_ratio : ∀ n m c : ℕ, price_ratio n m ↔ c = n * 5 / 3 ∧ m = c * 3 / 5) :
  ∃ c : ℕ, price_ratio 27 c ∧ c = 45 := 
by
  sorry

end cost_of_27_lilies_l1105_110597


namespace exactly_two_overlap_l1105_110522

-- Define the concept of rectangles
structure Rectangle :=
  (width : ℕ)
  (height : ℕ)

-- Define the given rectangles
def rect1 : Rectangle := ⟨4, 6⟩
def rect2 : Rectangle := ⟨4, 6⟩
def rect3 : Rectangle := ⟨4, 6⟩

-- Hypothesis defining the overlapping areas
def overlap1_2 : ℕ := 4 * 2 -- first and second rectangles overlap in 8 cells
def overlap2_3 : ℕ := 2 * 6 -- second and third rectangles overlap in 12 cells
def overlap1_3 : ℕ := 0    -- first and third rectangles do not directly overlap

-- Total overlap calculation
def total_exactly_two_overlap : ℕ := (overlap1_2 + overlap2_3)

-- The theorem we need to prove
theorem exactly_two_overlap (rect1 rect2 rect3 : Rectangle) : total_exactly_two_overlap = 14 := sorry

end exactly_two_overlap_l1105_110522


namespace honor_students_count_l1105_110558

noncomputable def number_of_honor_students (G B Eg Eb : ℕ) (p_girl p_boy : ℚ) : ℕ :=
  if G < 30 ∧ B < 30 ∧ Eg = (3 / 13) * G ∧ Eb = (4 / 11) * B ∧ G + B < 30 then
    Eg + Eb
  else
    0

theorem honor_students_count :
  ∃ (G B Eg Eb : ℕ), (G < 30 ∧ B < 30 ∧ G % 13 = 0 ∧ B % 11 = 0 ∧ Eg = (3 * G / 13) ∧ Eb = (4 * B / 11) ∧ G + B < 30 ∧ number_of_honor_students G B Eg Eb (3 / 13) (4 / 11) = 7) :=
by {
  sorry
}

end honor_students_count_l1105_110558


namespace solution_set_inequality_l1105_110536

theorem solution_set_inequality : {x : ℝ | (x-1)*(x-2) ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end solution_set_inequality_l1105_110536


namespace b_coordinates_bc_equation_l1105_110570

section GeometryProof

-- Define point A
def A : ℝ × ℝ := (1, 1)

-- Altitude CD has the equation: 3x + y - 12 = 0
def altitude_CD (x y : ℝ) : Prop := 3 * x + y - 12 = 0

-- Angle bisector BE has the equation: x - 2y + 4 = 0
def angle_bisector_BE (x y : ℝ) : Prop := x - 2 * y + 4 = 0

-- Coordinates of point B
def B : ℝ × ℝ := (-8, -2)

-- Equation of line BC
def line_BC (x y : ℝ) : Prop := 9 * x - 13 * y + 46 = 0

-- Proof statement for the coordinates of point B
theorem b_coordinates : ∃ x y : ℝ, (x, y) = B :=
by sorry

-- Proof statement for the equation of line BC
theorem bc_equation : ∃ (f : ℝ → ℝ → Prop), f = line_BC :=
by sorry

end GeometryProof

end b_coordinates_bc_equation_l1105_110570


namespace xsq_plus_ysq_l1105_110580

theorem xsq_plus_ysq (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : x^2 + y^2 = 25 :=
by
  sorry

end xsq_plus_ysq_l1105_110580


namespace average_infection_rate_infected_computers_exceed_700_l1105_110511

theorem average_infection_rate (h : (1 + x) ^ 2 = 81) : x = 8 := by
  sorry

theorem infected_computers_exceed_700 (h_infection_rate : 8 = 8) : (1 + 8) ^ 3 > 700 := by
  sorry

end average_infection_rate_infected_computers_exceed_700_l1105_110511


namespace fourth_power_square_prime_l1105_110543

noncomputable def fourth_smallest_prime := 7

theorem fourth_power_square_prime :
  (fourth_smallest_prime ^ 2) ^ 4 = 5764801 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end fourth_power_square_prime_l1105_110543


namespace max_unsealed_windows_l1105_110538

-- Definitions of conditions for the problem
def windows : Nat := 15
def panes : Nat := 15

-- Definition of the matching and selection process conditions
def matched_panes (window pane : Nat) : Prop :=
  pane >= window

-- Proof problem statement
theorem max_unsealed_windows 
  (glazier_approaches_window : ∀ (current_window : Nat), ∃ pane : Nat, pane >= current_window) :
  ∃ (max_unsealed : Nat), max_unsealed = 7 :=
by
  sorry

end max_unsealed_windows_l1105_110538


namespace total_birds_in_marsh_l1105_110541

def number_of_geese : Nat := 58
def number_of_ducks : Nat := 37

theorem total_birds_in_marsh :
  number_of_geese + number_of_ducks = 95 :=
sorry

end total_birds_in_marsh_l1105_110541


namespace exists_nonneg_integers_l1105_110526

theorem exists_nonneg_integers (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (x y z t : ℕ), (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ t ≠ 0) ∧ t < p ∧ x^2 + y^2 + z^2 = t * p :=
sorry

end exists_nonneg_integers_l1105_110526


namespace rearrange_expression_l1105_110560

theorem rearrange_expression :
  1 - 2 - 3 - 4 - (5 - 6 - 7) = 0 :=
by
  sorry

end rearrange_expression_l1105_110560


namespace arc_length_EF_l1105_110556

-- Definitions based on the conditions
def angle_DEF_degrees : ℝ := 45
def circumference_D : ℝ := 80
def total_circle_degrees : ℝ := 360

-- Theorems/lemmata needed to prove the required statement
theorem arc_length_EF :
  let proportion := angle_DEF_degrees / total_circle_degrees
  let arc_length := proportion * circumference_D
  arc_length = 10 :=
by
  -- Placeholder for the proof
  sorry

end arc_length_EF_l1105_110556


namespace rope_segment_length_l1105_110587

theorem rope_segment_length (L : ℕ) (half_fold_times : ℕ) (dm_to_cm : ℕ → ℕ) 
  (hL : L = 8) (h_half_fold_times : half_fold_times = 2) (h_dm_to_cm : dm_to_cm 1 = 10)
  : dm_to_cm (L / 2 ^ half_fold_times) = 20 := 
by 
  sorry

end rope_segment_length_l1105_110587


namespace x_coordinate_second_point_l1105_110561

theorem x_coordinate_second_point (m n : ℝ) 
(h₁ : m = 2 * n + 5)
(h₂ : m + 2 = 2 * (n + 1) + 5) : 
  (m + 2) = 2 * n + 7 :=
by sorry

end x_coordinate_second_point_l1105_110561


namespace baseball_cards_per_friend_l1105_110565

theorem baseball_cards_per_friend (total_cards friends : ℕ) (h_total : total_cards = 24) (h_friends : friends = 4) : total_cards / friends = 6 :=
by
  sorry

end baseball_cards_per_friend_l1105_110565


namespace EdProblem_l1105_110513

/- Define the conditions -/
def EdConditions := 
  ∃ (m : ℕ) (N : ℕ), 
    m = 16 ∧ 
    N = Nat.choose 15 5 ∧
    N % 1000 = 3

/- The statement to be proven -/
theorem EdProblem : EdConditions :=
  sorry

end EdProblem_l1105_110513


namespace turnip_difference_l1105_110503

theorem turnip_difference (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) : melanie_turnips - benny_turnips = 26 := by
  sorry

end turnip_difference_l1105_110503


namespace solve_for_x_l1105_110581

theorem solve_for_x (x : ℝ) (h_pos : 0 < x) (h : (x / 100) * (x ^ 2) = 9) : x = 10 * (3 ^ (1 / 3)) :=
by
  sorry

end solve_for_x_l1105_110581


namespace find_C_l1105_110596

theorem find_C (A B C : ℕ)
  (hA : A = 348)
  (hB : B = A + 173)
  (hC : C = B + 299) :
  C = 820 :=
sorry

end find_C_l1105_110596


namespace number_of_ways_to_sign_up_probability_student_A_online_journalists_l1105_110517

-- Definitions for the conditions
def students : Finset String := {"A", "B", "C", "D", "E"}
def projects : Finset String := {"Online Journalists", "Robot Action", "Sounds of Music"}

-- Function to calculate combinations (nCr)
def combinations (n k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate arrangements
def arrangements (n : ℕ) : ℕ := Nat.factorial n

-- Proof opportunity for part 1
theorem number_of_ways_to_sign_up : 
  (combinations 5 3 * arrangements 3) + ((combinations 5 2 * combinations 3 2) / arrangements 2 * arrangements 3) = 150 :=
sorry

-- Proof opportunity for part 2
theorem probability_student_A_online_journalists
  (h : (combinations 5 3 * arrangements 3 + combinations 5 3 * combinations 3 2 * arrangements 2 * arrangements 3) = 243) : 
  ((combinations 4 3 * arrangements 2) * projects.card ^ 3) / 
  (combinations 5 3 * arrangements 3 + combinations 5 3 * combinations 3 2 * arrangements 2 * arrangements 3) = 1 / 15 :=
sorry

end number_of_ways_to_sign_up_probability_student_A_online_journalists_l1105_110517


namespace weekend_price_of_coat_l1105_110528

-- Definitions based on conditions
def original_price : ℝ := 250
def sale_price_discount : ℝ := 0.4
def weekend_additional_discount : ℝ := 0.3

-- To prove the final weekend price
theorem weekend_price_of_coat :
  (original_price * (1 - sale_price_discount) * (1 - weekend_additional_discount)) = 105 := by
  sorry

end weekend_price_of_coat_l1105_110528


namespace cos_alpha_minus_pi_six_l1105_110521

theorem cos_alpha_minus_pi_six (α : ℝ) (h : Real.sin (α + Real.pi / 3) = 4 / 5) : 
  Real.cos (α - Real.pi / 6) = 4 / 5 :=
sorry

end cos_alpha_minus_pi_six_l1105_110521


namespace sector_angle_given_circumference_and_area_max_sector_area_given_circumference_l1105_110595

-- Problem (1)
theorem sector_angle_given_circumference_and_area :
  (∀ (r l : ℝ), 2 * r + l = 10 ∧ (1 / 2) * l * r = 4 → l / r = (1 / 2)) := by
  sorry

-- Problem (2)
theorem max_sector_area_given_circumference :
  (∀ (r l : ℝ), 2 * r + l = 40 → (r = 10 ∧ l = 20 ∧ (1 / 2) * l * r = 100 ∧ l / r = 2)) := by
  sorry

end sector_angle_given_circumference_and_area_max_sector_area_given_circumference_l1105_110595


namespace bricks_needed_for_room_floor_l1105_110583

-- Conditions
def length : ℕ := 4
def breadth : ℕ := 5
def bricks_per_square_meter : ℕ := 17

-- Question and Answer (Proof Problem)
theorem bricks_needed_for_room_floor : 
  (length * breadth) * bricks_per_square_meter = 340 := by
  sorry

end bricks_needed_for_room_floor_l1105_110583


namespace divisibility_by_n_l1105_110546

variable (a b c : ℤ) (n : ℕ)

theorem divisibility_by_n
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2 * n + 1) :
  ∃ k : ℤ, a^3 + b^2 - a^2 - b^3 = k * ↑n := 
sorry

end divisibility_by_n_l1105_110546


namespace true_proposition_l1105_110563

open Real

-- Proposition p
def p : Prop := ∀ x > 0, log x + 4 * x ≥ 3

-- Proposition q
def q : Prop := ∃ x > 0, 8 * x + 1 / (2 * x) ≤ 4

theorem true_proposition : ¬ p ∧ q := by
  sorry

end true_proposition_l1105_110563


namespace cost_price_bicycle_A_l1105_110588

variable {CP_A CP_B SP_C : ℝ}

theorem cost_price_bicycle_A (h1 : CP_B = 1.25 * CP_A) (h2 : SP_C = 1.25 * CP_B) (h3 : SP_C = 225) :
  CP_A = 144 :=
by
  sorry

end cost_price_bicycle_A_l1105_110588


namespace profit_without_discount_l1105_110573

theorem profit_without_discount (CP SP_with_discount SP_without_discount : ℝ) (h1 : CP = 100) (h2 : SP_with_discount = CP + 0.235 * CP) (h3 : SP_with_discount = 0.95 * SP_without_discount) : (SP_without_discount - CP) / CP * 100 = 30 :=
by
  sorry

end profit_without_discount_l1105_110573


namespace robin_initial_gum_l1105_110529

theorem robin_initial_gum (x : ℕ) (h1 : x + 26 = 44) : x = 18 := 
by 
  sorry

end robin_initial_gum_l1105_110529


namespace slope_of_line_l1105_110545

theorem slope_of_line (x y : ℝ) (h : 4 * y = 5 * x + 20) : y = (5/4) * x + 5 :=
by {
  sorry
}

end slope_of_line_l1105_110545


namespace cupcakes_gluten_nut_nonvegan_l1105_110559

-- Definitions based on conditions
def total_cupcakes := 120
def gluten_free_cupcakes := total_cupcakes / 3
def vegan_cupcakes := total_cupcakes / 4
def nut_free_cupcakes := total_cupcakes / 5
def gluten_and_vegan_cupcakes := 15
def vegan_and_nut_free_cupcakes := 10

-- Defining the theorem to prove the main question
theorem cupcakes_gluten_nut_nonvegan : 
  total_cupcakes - ((gluten_free_cupcakes + (vegan_cupcakes - gluten_and_vegan_cupcakes)) - vegan_and_nut_free_cupcakes) = 65 :=
by sorry

end cupcakes_gluten_nut_nonvegan_l1105_110559


namespace polynomial_expansion_l1105_110537

variable (t : ℝ)

theorem polynomial_expansion :
  (3 * t^3 + 2 * t^2 - 4 * t + 3) * (-4 * t^3 + 3 * t - 5) = -12 * t^6 - 8 * t^5 + 25 * t^4 - 21 * t^3 - 22 * t^2 + 29 * t - 15 :=
by {
  sorry
}

end polynomial_expansion_l1105_110537


namespace bankers_gain_is_60_l1105_110542

def banker's_gain (BD F PV R T : ℝ) : ℝ :=
  let TD := F - PV
  BD - TD

theorem bankers_gain_is_60 (BD F PV R T BG : ℝ) (h₁ : BD = 260) (h₂ : R = 0.10) (h₃ : T = 3)
  (h₄ : F = 260 / 0.3) (h₅ : PV = F / (1 + (R * T))) :
  banker's_gain BD F PV R T = 60 :=
by
  rw [banker's_gain, h₄, h₅]
  -- Further simplifications and exact equality steps would be added here with actual proof steps
  sorry

end bankers_gain_is_60_l1105_110542


namespace card_draw_probability_l1105_110505

-- Define a function to compute the probability of a sequence of draws
noncomputable def probability_of_event : Rat :=
  (4 / 52) * (4 / 51) * (1 / 50)

theorem card_draw_probability :
  probability_of_event = 4 / 33150 :=
by
  -- Proof goes here
  sorry

end card_draw_probability_l1105_110505


namespace slower_whale_length_is_101_25_l1105_110510

def length_of_slower_whale (v_i_f v_i_s a_f a_s t : ℝ) : ℝ :=
  let D_f := v_i_f * t + 0.5 * a_f * t^2
  let D_s := v_i_s * t + 0.5 * a_s * t^2
  D_f - D_s

theorem slower_whale_length_is_101_25
  (v_i_f v_i_s a_f a_s t L : ℝ)
  (h1 : v_i_f = 18)
  (h2 : v_i_s = 15)
  (h3 : a_f = 1)
  (h4 : a_s = 0.5)
  (h5 : t = 15)
  (h6 : length_of_slower_whale v_i_f v_i_s a_f a_s t = L) :
  L = 101.25 :=
by
  sorry

end slower_whale_length_is_101_25_l1105_110510


namespace parametric_equation_correct_max_min_x_plus_y_l1105_110504

noncomputable def parametric_equation (φ : ℝ) : ℝ × ℝ :=
  (2 + Real.sqrt 2 * Real.cos φ, 2 + Real.sqrt 2 * Real.sin φ)

theorem parametric_equation_correct (ρ θ : ℝ) (h : ρ^2 - 4 * Real.sqrt 2 * Real.cos (θ - π/4) + 6 = 0) :
  ∃ (φ : ℝ), parametric_equation φ = ( 2 + Real.sqrt 2 * Real.cos φ, 2 + Real.sqrt 2 * Real.sin φ) := 
sorry

theorem max_min_x_plus_y (P : ℝ × ℝ) (hP : ∃ (φ : ℝ), P = parametric_equation φ) :
  ∃ f : ℝ, (P.fst + P.snd) = f ∧ (f = 6 ∨ f = 2) :=
sorry

end parametric_equation_correct_max_min_x_plus_y_l1105_110504


namespace part1_part2_l1105_110572

-- Define set A and set B for m = 3
def setA : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def setB_m3 : Set ℝ := {x | x^2 - 2 * x - 3 < 0}

-- Define the complement of B in ℝ and the intersection of complements
def complB_m3 : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}
def intersection_complB_A : Set ℝ := complB_m3 ∩ setA

-- Verify that the intersection of the complement of B and A equals the given set
theorem part1 : intersection_complB_A = {x | 3 ≤ x ∧ x ≤ 5} :=
by
  sorry

-- Define set A and the intersection of A and B
def setA' : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def setAB : Set ℝ := {x | -1 < x ∧ x < 4}

-- Given A ∩ B = {x | -1 < x < 4}, determine m such that B = {x | -1 < x < 4}
theorem part2 : ∃ m : ℝ, (setA' ∩ {x | x^2 - 2 * x - m < 0} = setAB) ∧ m = 8 :=
by
  sorry

end part1_part2_l1105_110572


namespace exists_zero_in_interval_l1105_110585

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem exists_zero_in_interval : 
  (f 2) * (f 3) < 0 := by
  sorry

end exists_zero_in_interval_l1105_110585


namespace range_of_m_l1105_110552

theorem range_of_m (k : ℝ) (m : ℝ) (y x : ℝ)
  (h1 : ∀ x, y = k * (x - 1) + m)
  (h2 : y = 3 ∧ x = -2)
  (h3 : (∃ x, x < 0 ∧ y > 0) ∧ (∃ x, x < 0 ∧ y < 0) ∧ (∃ x, x > 0 ∧ y < 0)) :
  m < - (3 / 2) :=
sorry

end range_of_m_l1105_110552


namespace express_set_l1105_110548

open Set

/-- Define the set of natural numbers for which an expression is also a natural number. -/
theorem express_set : {x : ℕ | ∃ y : ℕ, 6 = y * (5 - x)} = {2, 3, 4} :=
by
  sorry

end express_set_l1105_110548


namespace angle_is_10_l1105_110506

theorem angle_is_10 (x : ℕ) (h1 : 180 - x = 2 * (90 - x) + 10) : x = 10 := 
by sorry

end angle_is_10_l1105_110506


namespace sin_alpha_eq_sqrt_5_div_3_l1105_110593

variable (α : ℝ)

theorem sin_alpha_eq_sqrt_5_div_3
  (hα : 0 < α ∧ α < Real.pi)
  (h : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) :
  Real.sin α = Real.sqrt 5 / 3 := 
by 
  sorry

end sin_alpha_eq_sqrt_5_div_3_l1105_110593


namespace parabola_x_intercepts_count_l1105_110562

theorem parabola_x_intercepts_count :
  let a := -3
  let b := 4
  let c := -1
  let discriminant := b ^ 2 - 4 * a * c
  discriminant ≥ 0 →
  let num_roots := if discriminant > 0 then 2 else if discriminant = 0 then 1 else 0
  num_roots = 2 := 
by {
  sorry
}

end parabola_x_intercepts_count_l1105_110562


namespace distance_travelled_is_960_l1105_110520

-- Definitions based on conditions
def speed_slower := 60 -- Speed of slower bike in km/h
def speed_faster := 64 -- Speed of faster bike in km/h
def time_diff := 1 -- Time difference in hours

-- Problem statement: Prove that the distance covered by both bikes is 960 km.
theorem distance_travelled_is_960 (T : ℝ) (D : ℝ) 
  (h1 : D = speed_slower * T)
  (h2 : D = speed_faster * (T - time_diff)) :
  D = 960 := 
sorry

end distance_travelled_is_960_l1105_110520


namespace birches_planted_l1105_110530

variable 
  (G B X : ℕ) -- G: number of girls, B: number of boys, X: number of birches

-- Conditions:
variable
  (h1 : G + B = 24) -- Total number of students
  (h2 : 3 * G + X = 24) -- Total number of plants
  (h3 : X = B / 3) -- Birches planted by boys

-- Proof statement:
theorem birches_planted : X = 6 :=
by 
  sorry

end birches_planted_l1105_110530


namespace smallest_four_digit_palindrome_divisible_by_8_l1105_110527

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def divisible_by_8 (n : ℕ) : Prop :=
  n % 8 = 0

theorem smallest_four_digit_palindrome_divisible_by_8 : ∃ (n : ℕ), is_palindrome n ∧ is_four_digit n ∧ divisible_by_8 n ∧ n = 4004 := by
  sorry

end smallest_four_digit_palindrome_divisible_by_8_l1105_110527


namespace sequence_general_term_l1105_110582

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 0 else 2 * n - 4

def S (n : ℕ) : ℤ :=
  n ^ 2 - 3 * n + 2

theorem sequence_general_term (n : ℕ) : a n = 
  if n = 1 then S n 
  else S n - S (n - 1) := by
  sorry

end sequence_general_term_l1105_110582


namespace part_1_prob_excellent_part_2_rounds_pvalues_l1105_110591

-- Definition of the probability of an excellent pair
def prob_excellent (p1 p2 : ℚ) : ℚ :=
  2 * p1 * (1 - p1) * p2 * p2 + p1 * p1 * 2 * p2 * (1 - p2) + p1 * p1 * p2 * p2

-- Part (1) statement: Prove the probability that they achieve "excellent pair" status in the first round
theorem part_1_prob_excellent (p1 p2 : ℚ) (hp1 : p1 = 3/4) (hp2 : p2 = 2/3) :
  prob_excellent p1 p2 = 2/3 := by
  rw [hp1, hp2]
  sorry

-- Part (2) statement: Prove the minimum number of rounds and values of p1 and p2
theorem part_2_rounds_pvalues (n : ℕ) (p1 p2 : ℚ) (h_sum : p1 + p2 = 4/3)
  (h_goal : n * prob_excellent p1 p2 ≥ 16) :
  (n = 27) ∧ (p1 = 2/3) ∧ (p2 = 2/3) := by
  sorry

end part_1_prob_excellent_part_2_rounds_pvalues_l1105_110591


namespace problem_solution_l1105_110514

/-- Let f be an even function on ℝ such that f(x + 2) = f(x) and f(x) = x - 2 for x ∈ [3, 4]. 
    Then f(sin 1) < f(cos 1). -/
theorem problem_solution (f : ℝ → ℝ) 
  (h1 : ∀ x, f (-x) = f x)
  (h2 : ∀ x, f (x + 2) = f x)
  (h3 : ∀ x, 3 ≤ x ∧ x ≤ 4 → f x = x - 2) :
  f (Real.sin 1) < f (Real.cos 1) :=
sorry

end problem_solution_l1105_110514


namespace solve_fractional_equation_l1105_110502

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 2 / 3) :
  (3 * x + 2) / (3 * x^2 + 4 * x - 4) = (3 * x) / (3 * x - 2) ↔ x = 1 / 3 ∨ x = -2 := by
  sorry

end solve_fractional_equation_l1105_110502


namespace mixing_ratios_l1105_110524

theorem mixing_ratios (V : ℝ) (hV : 0 < V) :
  (4 * V / 5 + 7 * V / 10) / (V / 5 + 3 * V / 10) = 3 :=
by
  sorry

end mixing_ratios_l1105_110524


namespace proof_problem_l1105_110507

-- Definitions of the conditions
def domain_R (f : ℝ → ℝ) : Prop := ∀ x : ℝ, true

def symmetric_graph_pt (f : ℝ → ℝ) (a : ℝ) (b : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) = 2 * b - f (a + x)

def symmetric (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = -f (x)

def symmetric_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (2*a - x) = f (x)

-- Definitions of the statements to prove
def statement_1 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (y = f (x - 1) → y = f (1 - x) → x = 1)

def statement_2 (f : ℝ → ℝ) : Prop :=
  symmetric_line f (3 / 2)

def statement_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) = -f (x)

-- Main proof problem
theorem proof_problem (f : ℝ → ℝ) 
  (h_domain : domain_R f)
  (h_symmetric_pt : symmetric_graph_pt f (-3 / 4) 0)
  (h_symmetric : ∀ x : ℝ, f (x + 3 / 2) = -f (x))
  (h_property : ∀ x : ℝ, f (x + 2) = -f (-x + 4)) :
  statement_1 f ∧ statement_2 f ∧ statement_3 f :=
sorry

end proof_problem_l1105_110507


namespace part_I_part_II_l1105_110594

noncomputable def M : Set ℝ := { x | |x + 1| + |x - 1| ≤ 2 }

theorem part_I : M = Set.Icc (-1 : ℝ) (1 : ℝ) := 
sorry

theorem part_II (x y z : ℝ) (hx : x ∈ M) (hy : |y| ≤ (1/6)) (hz : |z| ≤ (1/9)) :
  |x + 2 * y - 3 * z| ≤ (5/3) :=
by
  sorry

end part_I_part_II_l1105_110594


namespace find_start_number_l1105_110547

def count_even_not_divisible_by_3 (start end_ : ℕ) : ℕ :=
  (end_ / 2 + 1) - (end_ / 6 + 1) - (if start = 0 then start / 2 else start / 2 + 1 - (start - 1) / 6 - 1)

theorem find_start_number (start end_ : ℕ) (h1 : end_ = 170) (h2 : count_even_not_divisible_by_3 start end_ = 54) : start = 8 :=
by 
  rw [h1] at h2
  sorry

end find_start_number_l1105_110547


namespace tangent_parallel_to_line_l1105_110564

theorem tangent_parallel_to_line (x y : ℝ) :
  (y = x^3 + x - 1) ∧ (3 * x^2 + 1 = 4) → (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -3) := by
  sorry

end tangent_parallel_to_line_l1105_110564


namespace cupcakes_left_l1105_110555

def num_packages : ℝ := 3.5
def cupcakes_per_package : ℝ := 7
def cupcakes_eaten : ℝ := 5.75

theorem cupcakes_left :
  num_packages * cupcakes_per_package - cupcakes_eaten = 18.75 :=
by
  sorry

end cupcakes_left_l1105_110555


namespace diamonds_G20_l1105_110577

def diamonds_in_figure (n : ℕ) : ℕ :=
if n = 1 then 1 else 4 * n^2 + 4 * n - 7

theorem diamonds_G20 : diamonds_in_figure 20 = 1673 :=
by sorry

end diamonds_G20_l1105_110577


namespace change_received_l1105_110598

theorem change_received (cost_cat_toy : ℝ) (cost_cage : ℝ) (total_paid : ℝ) (change : ℝ) :
  cost_cat_toy = 8.77 →
  cost_cage = 10.97 →
  total_paid = 20.00 →
  change = 0.26 →
  total_paid - (cost_cat_toy + cost_cage) = change := by
sorry

end change_received_l1105_110598


namespace solve_for_x_l1105_110512

theorem solve_for_x (x : ℝ) (h : (2 * x + 7) / 6 = 13) : x = 35.5 :=
by
  -- Proof steps would go here
  sorry

end solve_for_x_l1105_110512


namespace total_clothes_donated_l1105_110550

theorem total_clothes_donated
  (pants : ℕ) (jumpers : ℕ) (pajama_sets : ℕ) (tshirts : ℕ)
  (friends : ℕ)
  (adam_donation : ℕ)
  (half_adam_donated : ℕ)
  (friends_donation : ℕ)
  (total_donation : ℕ)
  (h1 : pants = 4) 
  (h2 : jumpers = 4) 
  (h3 : pajama_sets = 4 * 2) 
  (h4 : tshirts = 20) 
  (h5 : friends = 3)
  (h6 : adam_donation = pants + jumpers + pajama_sets + tshirts) 
  (h7 : half_adam_donated = adam_donation / 2) 
  (h8 : friends_donation = friends * adam_donation) 
  (h9 : total_donation = friends_donation + half_adam_donated) :
  total_donation = 126 :=
by
  sorry

end total_clothes_donated_l1105_110550


namespace worm_in_apple_l1105_110576

theorem worm_in_apple (radius : ℝ) (travel_distance : ℝ) (h_radius : radius = 31) (h_travel_distance : travel_distance = 61) :
  ∃ S : Set ℝ, ∀ point_on_path : ℝ, (point_on_path ∈ S) → false :=
by
  sorry

end worm_in_apple_l1105_110576


namespace evaluate_expression_l1105_110523

theorem evaluate_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end evaluate_expression_l1105_110523


namespace radius_ratio_eq_inv_sqrt_5_l1105_110501

noncomputable def ratio_of_radii (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : ℝ :=
  a / b

theorem radius_ratio_eq_inv_sqrt_5 (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : 
  ratio_of_radii a b h = 1 / Real.sqrt 5 :=
sorry

end radius_ratio_eq_inv_sqrt_5_l1105_110501


namespace boiling_temperature_l1105_110551

-- Definitions according to conditions
def initial_temperature : ℕ := 41

def temperature_increase_per_minute : ℕ := 3

def pasta_cooking_time : ℕ := 12

def mixing_and_salad_time : ℕ := pasta_cooking_time / 3

def total_evening_time : ℕ := 73

-- Conditions and the problem statement in Lean
theorem boiling_temperature :
  initial_temperature + (total_evening_time - (pasta_cooking_time + mixing_and_salad_time)) * temperature_increase_per_minute = 212 :=
by
  -- Here would be the proof, skipped with sorry
  sorry

end boiling_temperature_l1105_110551


namespace sacks_per_day_l1105_110578

theorem sacks_per_day (total_sacks : ℕ) (days : ℕ) (h1 : total_sacks = 56) (h2 : days = 4) : total_sacks / days = 14 := by
  sorry

end sacks_per_day_l1105_110578


namespace largest_angle_of_pentagon_l1105_110575

theorem largest_angle_of_pentagon (x : ℝ) : 
  (2*x + 2) + 3*x + 4*x + 5*x + (6*x - 2) = 540 → 
  6*x - 2 = 160 :=
by
  intro h
  sorry

end largest_angle_of_pentagon_l1105_110575


namespace value_of_1_plus_i_cubed_l1105_110571

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- The main statement to verify
theorem value_of_1_plus_i_cubed : (1 + i ^ 3) = (1 - i) :=
by {  
  -- Use given conditions here if needed
  sorry
}

end value_of_1_plus_i_cubed_l1105_110571
