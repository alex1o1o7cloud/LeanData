import Mathlib

namespace houses_before_boom_l86_86592

theorem houses_before_boom (current_houses built_during_boom houses_before : ℕ) 
  (h1 : current_houses = 2000)
  (h2 : built_during_boom = 574)
  (h3 : current_houses = houses_before + built_during_boom) : 
  houses_before = 1426 := 
by
  -- Proof omitted
  sorry

end houses_before_boom_l86_86592


namespace smallest_prime_less_than_perfect_square_l86_86358

theorem smallest_prime_less_than_perfect_square : ∃ (n : ℕ), ∃ (k : ℤ), n = (k^2 - 12 : ℤ) ∧ nat.prime n ∧ n > 0 ∧ ∀ (m : ℕ), (∃ (j : ℤ), m = (j^2 - 12 : ℤ) ∧ nat.prime m ∧ m > 0) → n ≤ m :=
begin
  sorry
end

end smallest_prime_less_than_perfect_square_l86_86358


namespace average_speed_is_6_point_5_l86_86239

-- Define the given values
def total_distance : ℝ := 42
def riding_time : ℝ := 6
def break_time : ℝ := 0.5

-- Prove the average speed given the conditions
theorem average_speed_is_6_point_5 :
  (total_distance / (riding_time + break_time)) = 6.5 :=
by
  sorry

end average_speed_is_6_point_5_l86_86239


namespace mulch_cost_l86_86302

-- Definitions based on conditions
def cost_per_cubic_foot : ℕ := 8
def cubic_yard_to_cubic_feet : ℕ := 27
def volume_in_cubic_yards : ℕ := 7

-- Target statement to prove
theorem mulch_cost :
    (volume_in_cubic_yards * cubic_yard_to_cubic_feet) * cost_per_cubic_foot = 1512 := by
  sorry

end mulch_cost_l86_86302


namespace reciprocal_of_neg_four_l86_86741

def is_reciprocal (x y : ℚ) : Prop := x * y = 1

theorem reciprocal_of_neg_four : is_reciprocal (-4) (-1/4) :=
by
  sorry

end reciprocal_of_neg_four_l86_86741


namespace multiple_of_5_or_7_probability_l86_86103

theorem multiple_of_5_or_7_probability : 
  (let numbers := Finset.range 51 -- The set of numbers from 0 to 50
   let multiples_5 := numbers.filter (λ n, n % 5 = 0)
   let multiples_7 := numbers.filter (λ n, n % 7 = 0)
   let multiples_35 := numbers.filter (λ n, n % 35 = 0)
   let favorable_count := multiples_5.card + multiples_7.card - multiples_35.card
   let total_count := numbers.card - 1
   (favorable_count : ℚ) / total_count = 8 / 25) := sorry

end multiple_of_5_or_7_probability_l86_86103


namespace solution_of_xyz_l86_86629

theorem solution_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) : x + y + z = 48 := 
sorry

end solution_of_xyz_l86_86629


namespace distance_focus_directrix_l86_86440

theorem distance_focus_directrix (θ : ℝ) : 
  (∃ d : ℝ, (∀ (ρ : ℝ), ρ = 5 / (3 - 2 * Real.cos θ)) ∧ d = 5 / 2) :=
sorry

end distance_focus_directrix_l86_86440


namespace sum_of_eight_numbers_l86_86526

theorem sum_of_eight_numbers (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 := 
begin 
  sorry 
end

end sum_of_eight_numbers_l86_86526


namespace P_subset_Q_l86_86009

-- Define the set P
def P := {x : ℝ | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 1}

-- Define the set Q
def Q := {x : ℝ | x ≤ 2}

-- Prove P ⊆ Q
theorem P_subset_Q : P ⊆ Q :=
by
  sorry

end P_subset_Q_l86_86009


namespace reciprocal_of_neg_2023_l86_86990

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86990


namespace units_digit_of_33_pow_33_mul_7_pow_7_l86_86594

theorem units_digit_of_33_pow_33_mul_7_pow_7 : (33 ^ (33 * (7 ^ 7))) % 10 = 7 := 
  sorry

end units_digit_of_33_pow_33_mul_7_pow_7_l86_86594


namespace max_MB_value_l86_86429

open Real

-- Define the conditions of the problem
variables {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : sqrt 6 / 3 = sqrt (1 - b^2 / a^2))

-- Define the point M and the vertex B on the ellipse
variables (M : ℝ × ℝ) (hM : (M.1)^2 / (a)^2 + (M.2)^2 / (b)^2 = 1)
def B : ℝ × ℝ := (0, -b)

-- The task is to prove the maximum value of |MB| given the conditions
theorem max_MB_value : ∃ (maxMB : ℝ), maxMB = (3 * sqrt 2 / 2) * b :=
sorry

end max_MB_value_l86_86429


namespace final_price_is_correct_l86_86860

def initial_price : ℝ := 15
def first_discount_rate : ℝ := 0.2
def second_discount_rate : ℝ := 0.25

def first_discount : ℝ := initial_price * first_discount_rate
def price_after_first_discount : ℝ := initial_price - first_discount

def second_discount : ℝ := price_after_first_discount * second_discount_rate
def final_price : ℝ := price_after_first_discount - second_discount

theorem final_price_is_correct :
  final_price = 9 :=
by
  -- The actual proof steps will go here.
  sorry

end final_price_is_correct_l86_86860


namespace locus_of_point_P_l86_86267

/-- Given three points in the coordinate plane A(0,3), B(-√3, 0), and C(√3, 0), 
    and a point P on the coordinate plane such that PA = PB + PC, 
    determine the equation of the locus of point P. -/
noncomputable def locus_equation : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | (P.1^2 + (P.2 - 1)^2 = 4) ∧ (P.2 ≤ 0)}

theorem locus_of_point_P :
  ∀ (P : ℝ × ℝ),
  (∃ A B C : ℝ × ℝ, A = (0, 3) ∧ B = (-Real.sqrt 3, 0) ∧ C = (Real.sqrt 3, 0) ∧ 
     dist P A = dist P B + dist P C) →
  P ∈ locus_equation :=
by
  intros P hp
  sorry

end locus_of_point_P_l86_86267


namespace latitude_approx_l86_86818

noncomputable def calculate_latitude (R h : ℝ) (θ : ℝ) : ℝ :=
  if h = 0 then θ else Real.arccos (1 / (2 * Real.pi))

theorem latitude_approx (R h θ : ℝ) (h_nonzero : h ≠ 0)
  (r1 : ℝ := R * Real.cos θ)
  (r2 : ℝ := (R + h) * Real.cos θ)
  (s : ℝ := 2 * Real.pi * h * Real.cos θ)
  (condition : s = h) :
  θ = Real.arccos (1 / (2 * Real.pi)) := by
  sorry

end latitude_approx_l86_86818


namespace expression_range_l86_86260

open Real -- Open the real number namespace

theorem expression_range (x y : ℝ) (h : (x - 1)^2 + (y - 4)^2 = 1) : 
  0 ≤ (x * y - x) / (x^2 + (y - 1)^2) ∧ (x * y - x) / (x^2 + (y - 1)^2) ≤ 12 / 25 :=
sorry -- Proof to be filled in.

end expression_range_l86_86260


namespace smallest_prime_12_less_than_square_l86_86339

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l86_86339


namespace expected_value_range_of_p_l86_86812

theorem expected_value_range_of_p (p : ℝ) (X : ℕ → ℝ) :
  (∀ n, (n = 1 → X n = p) ∧ 
        (n = 2 → X n = p * (1 - p)) ∧ 
        (n = 3 → X n = (1 - p) ^ 2)) →
  (p^2 - 3 * p + 3 > 1.75) → 
  0 < p ∧ p < 0.5 := by
  intros hprob hexp
  -- Proof would be filled in here
  sorry

end expected_value_range_of_p_l86_86812


namespace smallest_integer_in_set_l86_86692

theorem smallest_integer_in_set (median : ℤ) (greatest : ℤ) (h1 : median = 157) (h2 : greatest = 169) :
  ∃ (smallest : ℤ), smallest = 145 :=
by
  -- Setup the conditions
  have set_cons_odd : True := trivial
  -- Known facts
  have h_median : median = 157 := by exact h1
  have h_greatest : greatest = 169 := by exact h2
  -- We must prove
  existsi 145
  sorry

end smallest_integer_in_set_l86_86692


namespace paint_area_correct_l86_86023

-- Definitions for the conditions of the problem
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def door_height : ℕ := 3
def door_length : ℕ := 5

-- Define the total area of the wall (without considering the door)
def wall_area : ℕ := wall_height * wall_length

-- Define the area of the door
def door_area : ℕ := door_height * door_length

-- Define the area that needs to be painted
def area_to_paint : ℕ := wall_area - door_area

-- The proof problem: Prove that Sandy needs to paint 135 square feet
theorem paint_area_correct : area_to_paint = 135 := 
by
  -- Sorry will be replaced with an actual proof
  sorry

end paint_area_correct_l86_86023


namespace crayons_total_correct_l86_86195

-- Definitions from the conditions
def initial_crayons : ℕ := 9
def added_crayons : ℕ := 3

-- Expected total crayons as per the conditions and the correct answer
def total_crayons_expected : ℕ := 12

-- The proof statement
theorem crayons_total_correct :
  initial_crayons + added_crayons = total_crayons_expected :=
by
  -- Proof details here
  sorry

end crayons_total_correct_l86_86195


namespace weights_balance_l86_86699

theorem weights_balance (k : ℕ) 
    (m n : ℕ → ℝ) 
    (h1 : ∀ i : ℕ, i < k → m i > n i) 
    (h2 : ∀ i : ℕ, i < k → ∃ j : ℕ, j ≠ i ∧ (m i + n j = n i + m j 
                                               ∨ m j + n i = n j + m i)) 
    : k = 1 ∨ k = 2 := 
by sorry

end weights_balance_l86_86699


namespace solve_for_x_l86_86538

theorem solve_for_x (x : ℝ) (h : x ≠ -2) :
  (4 * x) / (x + 2) - 2 / (x + 2) = 3 / (x + 2) → x = 5 / 4 := by
  sorry

end solve_for_x_l86_86538


namespace sum_of_vars_l86_86765

theorem sum_of_vars (x y z : ℝ) (h1 : y = 2 * x) (h2 : z = 2 * y) : x + y + z = 7 * x := 
by 
  sorry

end sum_of_vars_l86_86765


namespace tulips_for_each_eye_l86_86397

theorem tulips_for_each_eye (R : ℕ) : 2 * R + 18 + 9 * 18 = 196 → R = 8 :=
by
  intro h
  sorry

end tulips_for_each_eye_l86_86397


namespace perfect_squares_between_50_and_200_l86_86451

theorem perfect_squares_between_50_and_200 : ∃ n m : ℕ, (8 ≤ n ∧ n ≤ 14) ∧ (m - n + 1 = 7) :=
by {
  use 8, 14,
  split,
  {
    exact ⟨by norm_num, by norm_num⟩,
  },
  {
    norm_num,
  },
  sorry
}

end perfect_squares_between_50_and_200_l86_86451


namespace point_in_second_quadrant_l86_86933

theorem point_in_second_quadrant (m : ℝ) (h1 : 3 - m < 0) (h2 : m - 1 > 0) : m > 3 :=
by
  sorry

end point_in_second_quadrant_l86_86933


namespace negation_of_implication_l86_86185

theorem negation_of_implication (a b c : ℝ) :
  ¬ (a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) :=
by sorry

end negation_of_implication_l86_86185


namespace least_wins_to_40_points_l86_86046

theorem least_wins_to_40_points 
  (points_per_victory : ℕ)
  (points_per_draw : ℕ)
  (points_per_defeat : ℕ)
  (total_matches : ℕ)
  (initial_points : ℕ)
  (matches_played : ℕ)
  (target_points : ℕ) :
  points_per_victory = 3 →
  points_per_draw = 1 →
  points_per_defeat = 0 →
  total_matches = 20 →
  initial_points = 12 →
  matches_played = 5 →
  target_points = 40 →
  ∃ wins_needed : ℕ, wins_needed = 10 :=
by
  sorry

end least_wins_to_40_points_l86_86046


namespace mark_total_young_fish_l86_86795

-- Define the conditions
def num_tanks : ℕ := 5
def fish_per_tank : ℕ := 6
def young_per_fish : ℕ := 25

-- Define the total number of young fish
def total_young_fish := num_tanks * fish_per_tank * young_per_fish

-- The theorem statement
theorem mark_total_young_fish : total_young_fish = 750 :=
  by
    sorry

end mark_total_young_fish_l86_86795


namespace amanda_final_score_l86_86590

theorem amanda_final_score
  (average1 : ℕ) (quizzes1 : ℕ) (average_required : ℕ) (quizzes_total : ℕ)
  (H1 : average1 = 92) (H2 : quizzes1 = 4) (H3 : average_required = 93) (H4 : quizzes_total = 5) :
  let total_points1 := quizzes1 * average1,
      total_points_required := quizzes_total * average_required,
      final_score_needed := total_points_required - total_points1
  in final_score_needed = 97 := by
  sorry

end amanda_final_score_l86_86590


namespace polynomial_multiplication_correct_l86_86100

noncomputable def polynomial_expansion : Polynomial ℤ :=
  (Polynomial.C (3 : ℤ) * Polynomial.X ^ 3 + Polynomial.C (4 : ℤ) * Polynomial.X ^ 2 - Polynomial.C (8 : ℤ) * Polynomial.X - Polynomial.C (5 : ℤ)) *
  (Polynomial.C (2 : ℤ) * Polynomial.X ^ 4 - Polynomial.C (3 : ℤ) * Polynomial.X ^ 2 + Polynomial.C (1 : ℤ))

theorem polynomial_multiplication_correct :
  polynomial_expansion = Polynomial.C (6 : ℤ) * Polynomial.X ^ 7 +
                         Polynomial.C (12 : ℤ) * Polynomial.X ^ 6 -
                         Polynomial.C (25 : ℤ) * Polynomial.X ^ 5 -
                         Polynomial.C (20 : ℤ) * Polynomial.X ^ 4 +
                         Polynomial.C (34 : ℤ) * Polynomial.X ^ 2 -
                         Polynomial.C (8 : ℤ) * Polynomial.X -
                         Polynomial.C (5 : ℤ) :=
by
  sorry

end polynomial_multiplication_correct_l86_86100


namespace total_candies_is_829_l86_86801

-- Conditions as definitions
def Adam : ℕ := 6
def James : ℕ := 3 * Adam
def Rubert : ℕ := 4 * James
def Lisa : ℕ := 2 * Rubert
def Chris : ℕ := Lisa + 5
def Emily : ℕ := 3 * Chris - 7

-- Total candies
def total_candies : ℕ := Adam + James + Rubert + Lisa + Chris + Emily

-- Theorem to prove
theorem total_candies_is_829 : total_candies = 829 :=
by
  -- skipping the proof
  sorry

end total_candies_is_829_l86_86801


namespace right_triangle_conditions_l86_86598

theorem right_triangle_conditions (x y z h α β : ℝ) : 
  x - y = α → 
  z - h = β → 
  x^2 + y^2 = z^2 → 
  x * y = h * z → 
  β > α :=
by 
sorry

end right_triangle_conditions_l86_86598


namespace pat_peano_maximum_pages_l86_86799

noncomputable def count_fives_in_range : ℕ → ℕ := sorry

theorem pat_peano_maximum_pages (n : ℕ) : 
  (count_fives_in_range 54) = 15 → n ≤ 54 :=
sorry

end pat_peano_maximum_pages_l86_86799


namespace parabola_translation_shift_downwards_l86_86694

theorem parabola_translation_shift_downwards :
  ∀ (x y : ℝ), (y = x^2 - 5) ↔ ((∃ (k : ℝ), k = -5 ∧ y = x^2 + k)) :=
by
  sorry

end parabola_translation_shift_downwards_l86_86694


namespace original_average_rent_l86_86544

theorem original_average_rent
    (A : ℝ) -- original average rent per person
    (h1 : 4 * A + 200 = 3400) -- condition derived from the rent problem
    : A = 800 := 
sorry

end original_average_rent_l86_86544


namespace point_in_second_quadrant_l86_86934

theorem point_in_second_quadrant (m : ℝ) (h1 : 3 - m < 0) (h2 : m - 1 > 0) : m > 3 :=
by
  sorry

end point_in_second_quadrant_l86_86934


namespace factor_expression_l86_86597

theorem factor_expression (x y z : ℝ) :
  x^3 * (y^2 - z^2) - y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x * y + z^2 - z * x) :=
by
  sorry

end factor_expression_l86_86597


namespace find_x_ineq_solution_l86_86734

open Set

theorem find_x_ineq_solution :
  {x : ℝ | (x - 2) / (x - 4) ≥ 3} = Ioc 4 5 := 
sorry

end find_x_ineq_solution_l86_86734


namespace distance_origin_to_point_l86_86642

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_origin_to_point :
  distance (0, 0) (-15, 8) = 17 :=
by 
  sorry

end distance_origin_to_point_l86_86642


namespace animal_shelter_cats_l86_86476

theorem animal_shelter_cats (D C x : ℕ) (h1 : 15 * C = 7 * D) (h2 : 15 * (C + x) = 11 * D) (h3 : D = 60) : x = 16 :=
by
  sorry

end animal_shelter_cats_l86_86476


namespace banana_popsicles_count_l86_86782

theorem banana_popsicles_count 
  (grape_popsicles cherry_popsicles total_popsicles : ℕ)
  (h1 : grape_popsicles = 2)
  (h2 : cherry_popsicles = 13)
  (h3 : total_popsicles = 17) :
  total_popsicles - (grape_popsicles + cherry_popsicles) = 2 := by
  sorry

end banana_popsicles_count_l86_86782


namespace inequality_solution_reciprocal_inequality_l86_86835

-- Proof Problem (1)
theorem inequality_solution (x : ℝ) : |x-1| + (1/2)*|x-3| < 2 ↔ (1 < x ∧ x < 3) :=
sorry

-- Proof Problem (2)
theorem reciprocal_inequality (a b c : ℝ) (h : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 2) : 
  (1/a) + (1/b) + (1/c) ≥ 9/2 :=
sorry

end inequality_solution_reciprocal_inequality_l86_86835


namespace group_arrangement_l86_86774

-- Definition of the problem parameters
def men : Finset ℕ := {0, 1, 2, 3}
def women : Finset ℕ := {0, 1, 2, 3, 4}

-- Definition of the proof statement
theorem group_arrangement : 
  let group_conditions := 
    (λ grp1 grp2 grp3 : Finset ℕ, grp1.card = 3 ∧ grp2.card = 2 ∧ grp3.card = 2 ∧
                                   grp1 ∩ grp2 = ∅ ∧ grp1 ∩ grp3 = ∅ ∧ grp2 ∩ grp3 = ∅ ∧
                                   ∃ (m1 m2 m3 wl : ℕ) (grp1 ⊆ men ∪ women), 
                                     grp1 = {m1, m2, wl} ∧
                                     grp2 = {men.erase m1, women.erase wl} ∧ 
                                     grp3 = {men.erase m2, women.erase wl}) in
  ∀ (grps : Finset (Finset ℕ)), grps.card = 3 → group_conditions grps = 240 :=
sorry

end group_arrangement_l86_86774


namespace perfect_squares_count_between_50_and_200_l86_86463

theorem perfect_squares_count_between_50_and_200 :
  ∃ (N : ℕ), N = (finset.Ico 8 15).card ∧ N = 7 :=
by
  sorry

end perfect_squares_count_between_50_and_200_l86_86463


namespace factorial_trailing_digits_l86_86090

theorem factorial_trailing_digits (n : ℕ) :
  ¬ ∃ k : ℕ, (n! / 10^k) % 10000 = 1976 ∧ k > 0 := 
sorry

end factorial_trailing_digits_l86_86090


namespace expected_value_matches_variance_matches_l86_86840

variables {N : ℕ} (I : Fin N → Bool)

-- Define the probability that a randomly chosen pair of cards matches
def p_match : ℝ := 1 / N

-- Define the indicator variable I_k
def I_k (k : Fin N) : ℝ :=
if I k then 1 else 0

-- Define the sum S of all the indicator variables
def S : ℝ := (Finset.univ.sum I_k)

-- Expected value E[I_k] is 1/N
def E_I_k : ℝ := 1 / N

-- Expected value E[S] is the sum of E[I_k] over all k, which is 1
theorem expected_value_matches : ∑ k, E_I_k = 1 := sorry

-- Variance calculation: Var[S] = E[S^2] - (E[S])^2
def E_S_sq : ℝ := (Finset.univ.sum (λ k, I_k k * I_k k)) + 
                  2 * (Finset.univ.sum (λ (jk : Fin N × Fin N), if jk.1 < jk.2 then I_k jk.1 * I_k jk.2 else 0))

theorem variance_matches : (E_S_sq - 1) = 1 := sorry

end expected_value_matches_variance_matches_l86_86840


namespace part1_part2_l86_86129

def A (x : ℝ) : Prop := -2 < x ∧ x < 10
def B (x a : ℝ) : Prop := (x ≥ 1 + a ∨ x ≤ 1 - a) ∧ a > 0
def p (x : ℝ) : Prop := A x
def q (x a : ℝ) : Prop := B x a

theorem part1 (a : ℝ) (hA : ∀ x, A x → ¬ B x a) : a ≥ 9 :=
sorry

theorem part2 (a : ℝ) (hSuff : ∀ x, (x ≥ 10 ∨ x ≤ -2) → B x a) (hNotNec : ∃ x, ¬ (x ≥ 10 ∨ x ≤ -2) ∧ B x a) : 0 < a ∧ a ≤ 3 :=
sorry

end part1_part2_l86_86129


namespace max_sum_first_n_terms_l86_86145

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem max_sum_first_n_terms (a1 : ℝ) (h1 : a1 > 0)
  (h2 : 5 * a_n a1 d 8 = 8 * a_n a1 d 13) :
  ∃ n : ℕ, n = 21 ∧ ∀ m : ℕ, S_n a1 d m ≤ S_n a1 d n :=
by
  sorry

end max_sum_first_n_terms_l86_86145


namespace solution_set_l86_86736

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x - 2) / (x - 4) ≥ 3

theorem solution_set :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | 4 < x ∧ x ≤ 5} :=
by
  sorry

end solution_set_l86_86736


namespace solve_fraction_equation_l86_86418

theorem solve_fraction_equation (x : ℝ) (hx1 : 0 < x) (hx2 : (x - 6) / 12 = 6 / (x - 12)) : x = 18 := 
sorry

end solve_fraction_equation_l86_86418


namespace probability_sum_18_l86_86926

theorem probability_sum_18:
  (∑ k in {1,2,3,4,5,6}, k = 6)^4 * (probability {d₁ d₂ d₃ d₄ : ℕ | d₁ + d₂ + d₃ + d₄ = 18} 6 6) = 5 / 216 := 
sorry

end probability_sum_18_l86_86926


namespace number_added_to_x_is_2_l86_86902

/-- Prove that in a set of integers {x, x + y, x + 4, x + 7, x + 22}, 
    where the mean is 3 greater than the median, the number added to x 
    to get the second integer is 2. --/

theorem number_added_to_x_is_2 (x y : ℤ) (h_pos : 0 < x ∧ 0 < y) 
  (h_median : (x + 4) = ((x + y) + (x + (x + y) + (x + 4) + (x + 7) + (x + 22)) / 5 - 3)) : 
  y = 2 := by
  sorry

end number_added_to_x_is_2_l86_86902


namespace smallest_prime_12_less_than_square_l86_86336

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l86_86336


namespace binom_squared_l86_86408

theorem binom_squared :
  (Nat.choose 12 11) ^ 2 = 144 := 
by
  -- Mathematical steps would go here.
  sorry

end binom_squared_l86_86408


namespace sin_225_correct_l86_86879

-- Define the condition of point being on the unit circle at 225 degrees.
noncomputable def P_225 := Complex.polar 1 (Real.pi + Real.pi / 4)

-- Define the goal statement that translates the question and correct answer.
theorem sin_225_correct : Complex.sin (Real.pi + Real.pi / 4) = -Real.sqrt 2 / 2 := 
by sorry

end sin_225_correct_l86_86879


namespace johns_total_cost_l86_86652

-- Definitions for the prices and quantities
def price_shirt : ℝ := 15.75
def price_tie : ℝ := 9.40
def quantity_shirts : ℕ := 3
def quantity_ties : ℕ := 2

-- Definition for the total cost calculation
def total_cost (price_shirt price_tie : ℝ) (quantity_shirts quantity_ties : ℕ) : ℝ :=
  (price_shirt * quantity_shirts) + (price_tie * quantity_ties)

-- Theorem stating the total cost calculation for John's purchase
theorem johns_total_cost : total_cost price_shirt price_tie quantity_shirts quantity_ties = 66.05 :=
by
  sorry

end johns_total_cost_l86_86652


namespace problem1_problem2_l86_86089

-- Define the main assumptions and the proof problem for Lean 4
theorem problem1 (a : ℝ) (h : a ≠ 0) : (a^2)^3 / (-a)^2 = a^4 := sorry

theorem problem2 (a b : ℝ) : (a + 2 * b) * (a + b) - 3 * a * (a + b) = -2 * a^2 + 2 * b^2 := sorry

end problem1_problem2_l86_86089


namespace f_monotonically_increasing_l86_86914

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.log x + 1 / x

-- Prove that f is monotonically increasing on (1, +∞)
theorem f_monotonically_increasing : (∀ x : ℝ, x > 1 ↔ ∃ δ > 0, ∀ (h : ℝ), h < x + δ ∧ h > x - δ → f' h > 0) := 
sorry

end f_monotonically_increasing_l86_86914


namespace non_parallel_lines_a_l86_86441

theorem non_parallel_lines_a (a : ℝ) :
  ¬ (a * -(1 / (a+2))) = a →
  ¬ (-1 / (a+2)) = 2 →
  a = 0 ∨ a = -3 :=
by
  sorry

end non_parallel_lines_a_l86_86441


namespace bucket_full_weight_l86_86713

theorem bucket_full_weight (c d : ℝ) (x y : ℝ) (h1 : x + (1 / 4) * y = c) (h2 : x + (3 / 4) * y = d) : 
  x + y = (3 * d - 3 * c) / 2 :=
by
  sorry

end bucket_full_weight_l86_86713


namespace inequality_solution_l86_86768

theorem inequality_solution {a b x : ℝ} 
  (h_sol_set : -1 < x ∧ x < 1) 
  (h1 : x - a > 2) 
  (h2 : b - 2 * x > 0) : 
  (a + b) ^ 2021 = -1 := 
by 
  sorry 

end inequality_solution_l86_86768


namespace minimum_vehicles_l86_86170

theorem minimum_vehicles (students adults : ℕ) (van_capacity minibus_capacity : ℕ)
    (severe_allergies_students : ℕ) (vehicle_requires_adult : Prop)
    (h_students : students = 24) (h_adults : adults = 3)
    (h_van_capacity : van_capacity = 8) (h_minibus_capacity : minibus_capacity = 14)
    (h_severe_allergies_students : severe_allergies_students = 2)
    (h_vehicle_requires_adult : vehicle_requires_adult)
    : ∃ (min_vehicles : ℕ), min_vehicles = 5 :=
by
  sorry

end minimum_vehicles_l86_86170


namespace equation_of_line_l86_86697

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1 * v.1 + v.2 * v.2
  (dot_product / norm_sq * v.1, dot_product / norm_sq * v.2)

theorem equation_of_line (x y : ℝ) :
  projection (x, y) (7, 3) = (-7, -3) →
  y = -7/3 * x - 58/3 :=
by
  intro h
  sorry

end equation_of_line_l86_86697


namespace batteries_on_flashlights_l86_86825

variable (b_flashlights b_toys b_controllers b_total : ℕ)

theorem batteries_on_flashlights :
  b_toys = 15 → 
  b_controllers = 2 → 
  b_total = 19 → 
  b_total = b_flashlights + b_toys + b_controllers → 
  b_flashlights = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end batteries_on_flashlights_l86_86825


namespace range_of_a_l86_86747

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : ∃ x : ℝ, |x - 4| + |x + 3| < a) : a > 7 :=
sorry

end range_of_a_l86_86747


namespace arithmetic_sequence_fifth_term_l86_86918

theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℕ), (∀ n, a n.succ = a n + 2) → a 1 = 2 → a 5 = 10 :=
by
  intros a h1 h2
  sorry

end arithmetic_sequence_fifth_term_l86_86918


namespace sum_of_eight_numbers_l86_86524

theorem sum_of_eight_numbers (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 := 
begin 
  sorry 
end

end sum_of_eight_numbers_l86_86524


namespace miles_to_friends_house_l86_86290

-- Define the conditions as constants
def miles_per_gallon : ℕ := 19
def gallons : ℕ := 2
def miles_to_school : ℕ := 15
def miles_to_softball_park : ℕ := 6
def miles_to_burger_restaurant : ℕ := 2
def miles_home : ℕ := 11

-- Define the total miles driven
def total_miles_driven (miles_to_friend : ℕ) :=
  miles_to_school + miles_to_softball_park + miles_to_burger_restaurant + miles_to_friend + miles_home

-- Define the total miles possible with given gallons of gas
def total_miles_possible : ℕ :=
  miles_per_gallon * gallons

-- Prove that the miles driven to the friend's house is 4
theorem miles_to_friends_house : 
  ∃ miles_to_friend, total_miles_driven miles_to_friend = total_miles_possible ∧ miles_to_friend = 4 :=
by
  sorry

end miles_to_friends_house_l86_86290


namespace find_X_l86_86663

variable {α : Type} -- considering sets of some type α
variables (A B X : Set α)

theorem find_X (h1 : A ∩ X = B ∩ X ∧ B ∩ X = A ∩ B)
               (h2 : A ∪ B ∪ X = A ∪ B) : X = A ∩ B :=
by {
    sorry
}

end find_X_l86_86663


namespace fred_has_18_stickers_l86_86006

def jerry_stickers := 36
def george_stickers (jerry : ℕ) := jerry / 3
def fred_stickers (george : ℕ) := george + 6

theorem fred_has_18_stickers :
  let j := jerry_stickers
  let g := george_stickers j 
  fred_stickers g = 18 :=
by
  sorry

end fred_has_18_stickers_l86_86006


namespace smallest_prime_12_less_perfect_square_l86_86340

def is_prime (n : ℕ) : Prop := nat.prime n

def is_perfect_square_less_12 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 12

def smallest_prime (P : ℕ → Prop) : ℕ :=
  if h : ∃ n, P n then classical.some h else 0

def satisfies_conditions (n : ℕ) : Prop :=
  is_prime n ∧ is_perfect_square_less_12 n

theorem smallest_prime_12_less_perfect_square :
  smallest_prime satisfies_conditions = 13 :=
  sorry

end smallest_prime_12_less_perfect_square_l86_86340


namespace law_I_law_II_l86_86492

section
variable (x y z : ℝ)

def op_at (a b : ℝ) : ℝ := a + 2 * b
def op_hash (a b : ℝ) : ℝ := 2 * a - b

theorem law_I (x y z : ℝ) : op_at x (op_hash y z) = op_hash (op_at x y) (op_at x z) := 
by
  unfold op_at op_hash
  sorry

theorem law_II (x y z : ℝ) : x + op_at y z ≠ op_at (x + y) (x + z) := 
by
  unfold op_at
  sorry

end

end law_I_law_II_l86_86492


namespace expected_value_matches_variance_matches_l86_86839

variables {N : ℕ} (I : Fin N → Bool)

-- Define the probability that a randomly chosen pair of cards matches
def p_match : ℝ := 1 / N

-- Define the indicator variable I_k
def I_k (k : Fin N) : ℝ :=
if I k then 1 else 0

-- Define the sum S of all the indicator variables
def S : ℝ := (Finset.univ.sum I_k)

-- Expected value E[I_k] is 1/N
def E_I_k : ℝ := 1 / N

-- Expected value E[S] is the sum of E[I_k] over all k, which is 1
theorem expected_value_matches : ∑ k, E_I_k = 1 := sorry

-- Variance calculation: Var[S] = E[S^2] - (E[S])^2
def E_S_sq : ℝ := (Finset.univ.sum (λ k, I_k k * I_k k)) + 
                  2 * (Finset.univ.sum (λ (jk : Fin N × Fin N), if jk.1 < jk.2 then I_k jk.1 * I_k jk.2 else 0))

theorem variance_matches : (E_S_sq - 1) = 1 := sorry

end expected_value_matches_variance_matches_l86_86839


namespace find_two_numbers_l86_86190

theorem find_two_numbers (x y : ℕ) :
  (x + y = 667 ∧ Nat.lcm x y / Nat.gcd x y = 120) ↔
  (x = 232 ∧ y = 435) ∨ (x = 552 ∧ y = 115) :=
by
  sorry

end find_two_numbers_l86_86190


namespace incorrect_statement_B_l86_86094

-- Define the plane vector operation "☉".
def vector_operation (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

-- Define the mathematical problem based on the given conditions.
theorem incorrect_statement_B (a b : ℝ × ℝ) : vector_operation a b ≠ vector_operation b a := by
  sorry

end incorrect_statement_B_l86_86094


namespace percent_profit_l86_86138

variable (C S : ℝ)

theorem percent_profit (h : 72 * C = 60 * S) : ((S - C) / C) * 100 = 20 := by
  sorry

end percent_profit_l86_86138


namespace probability_of_B_l86_86186

variables (A B : Prop)
variables (P : Prop → ℝ) -- Probability Measure

axiom A_and_B : P (A ∧ B) = 0.15
axiom not_A_and_not_B : P (¬A ∧ ¬B) = 0.6

theorem probability_of_B : P B = 0.15 :=
by
  sorry

end probability_of_B_l86_86186


namespace function_increasing_probability_l86_86908

noncomputable def is_increasing_on_interval (a b : ℤ) : Prop :=
∀ x : ℝ, x > 1 → 2 * a * x - 2 * b > 0

noncomputable def valid_pairs : List (ℤ × ℤ) :=
[(0, -1), (1, -1), (1, 1), (2, -1), (2, 1)]

noncomputable def total_pairs : ℕ :=
3 * 4

noncomputable def probability_of_increasing_function : ℚ :=
(valid_pairs.length : ℚ) / total_pairs

theorem function_increasing_probability :
  probability_of_increasing_function = 5 / 12 :=
by
  sorry

end function_increasing_probability_l86_86908


namespace abs_diff_simplification_l86_86468

theorem abs_diff_simplification (a b : ℝ) (h1 : a < 0) (h2 : a * b < 0) : |b - a + 1| - |a - b - 5| = -4 :=
  sorry

end abs_diff_simplification_l86_86468


namespace sum_of_numbers_on_cards_l86_86506

-- Define the natural numbers condition
variables {a b c d e f g h : ℕ}

-- The theorem statement
theorem sum_of_numbers_on_cards (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_numbers_on_cards_l86_86506


namespace count_multiples_less_than_300_l86_86269

theorem count_multiples_less_than_300 : ∀ n : ℕ, n < 300 → (2 * 3 * 5 * 7 ∣ n) ↔ n = 210 :=
by
  sorry

end count_multiples_less_than_300_l86_86269


namespace area_of_smaller_circle_l86_86561

/-
  Variables and assumptions:
  r: Radius of the smaller circle
  R: Radius of the larger circle which is three times the smaller circle. Hence, R = 3 * r.
  PA = AB = 6: Lengths of the tangent segments
  Area: Calculated area of the smaller circle
-/

theorem area_of_smaller_circle (r : ℝ) (h1 : 6 = r) (h2 : 3 * 6 = R) (h3 : 6 = r) : 
  ∃ (area : ℝ), area = (36 * Real.pi) / 7 :=
by
  sorry 

end area_of_smaller_circle_l86_86561


namespace solve_for_x_l86_86175

theorem solve_for_x (x : ℝ) (h : 1 - 1 / (1 - x) ^ 3 = 1 / (1 - x)) : x = 1 :=
sorry

end solve_for_x_l86_86175


namespace positive_integers_sum_of_squares_l86_86042

theorem positive_integers_sum_of_squares
  (a b c d : ℤ)
  (h1 : a^2 + b^2 + c^2 + d^2 = 90)
  (h2 : a + b + c + d = 16) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d := 
by
  sorry

end positive_integers_sum_of_squares_l86_86042


namespace sin_225_eq_neg_sqrt_two_div_two_l86_86880

theorem sin_225_eq_neg_sqrt_two_div_two :
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt_two_div_two_l86_86880


namespace sam_runs_more_than_sarah_sue_runs_less_than_sarah_l86_86638

-- Definitions based on the problem conditions
def street_width : ℝ := 25
def block_side_length : ℝ := 500
def sarah_perimeter : ℝ := 4 * block_side_length
def sam_perimeter : ℝ := 4 * (block_side_length + 2 * street_width)
def sue_perimeter : ℝ := 4 * (block_side_length - 2 * street_width)

-- The proof problem statements
theorem sam_runs_more_than_sarah : sam_perimeter - sarah_perimeter = 200 := by
  sorry

theorem sue_runs_less_than_sarah : sarah_perimeter - sue_perimeter = 200 := by
  sorry

end sam_runs_more_than_sarah_sue_runs_less_than_sarah_l86_86638


namespace hannah_late_times_l86_86760

variable (hourly_rate : ℝ)
variable (hours_worked : ℝ)
variable (dock_per_late : ℝ)
variable (actual_pay : ℝ)

theorem hannah_late_times (h1 : hourly_rate = 30)
                          (h2 : hours_worked = 18)
                          (h3 : dock_per_late = 5)
                          (h4 : actual_pay = 525) :
  ((hourly_rate * hours_worked - actual_pay) / dock_per_late) = 3 := 
by
  sorry

end hannah_late_times_l86_86760


namespace log_inequality_l86_86789

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 2 / Real.log 5
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem log_inequality : c > a ∧ a > b := 
by
  sorry

end log_inequality_l86_86789


namespace yardsCatchingPasses_l86_86813

-- Definitions from conditions in a)
def totalYardage : ℕ := 150
def runningYardage : ℕ := 90

-- Problem statement (Proof will follow)
theorem yardsCatchingPasses : totalYardage - runningYardage = 60 := by
  sorry

end yardsCatchingPasses_l86_86813


namespace polynomial_divisibility_l86_86612

theorem polynomial_divisibility 
  (a b c : ℤ)
  (P : ℤ → ℤ)
  (root_condition : ∃ u v : ℤ, u * v * (u + v) = -c ∧ u * v = b) 
  (P_def : ∀ x, P x = x^3 + a * x^2 + b * x + c) :
  2 * P (-1) ∣ (P 1 + P (-1) - 2 * (1 + P 0)) :=
by
  sorry

end polynomial_divisibility_l86_86612


namespace baker_number_of_eggs_l86_86069

theorem baker_number_of_eggs (flour cups eggs : ℕ) (h1 : eggs = 3 * (flour / 2)) (h2 : flour = 6) : eggs = 9 :=
by
  sorry

end baker_number_of_eggs_l86_86069


namespace triangle_length_l86_86487

theorem triangle_length (DE DF : ℝ) (Median_to_EF : ℝ) (EF : ℝ) :
  DE = 2 ∧ DF = 3 ∧ Median_to_EF = EF → EF = (13:ℝ).sqrt / 5 := by
  sorry

end triangle_length_l86_86487


namespace prob_sum_is_18_l86_86928

theorem prob_sum_is_18 : 
  let num_faces := 6
  let num_dice := 4
  let total_outcomes := num_faces ^ num_dice
  ∑ (d1 d2 d3 d4 : ℕ) in finset.Icc 1 num_faces, 
  if d1 + d2 + d3 + d4 = 18 then 1 else 0 = 35 → 
  (35 : ℚ) / total_outcomes = 35 / 648 :=
by
  sorry

end prob_sum_is_18_l86_86928


namespace smallest_prime_less_than_perfect_square_is_13_l86_86343

noncomputable def smallest_prime_less_than_perfect_square : ℕ :=
  Inf {p : ℕ | ∃ k : ℕ, p = k^2 - 12 ∧ p > 0 ∧ Nat.Prime p}

theorem smallest_prime_less_than_perfect_square_is_13 :
  smallest_prime_less_than_perfect_square = 13 := by
  sorry

end smallest_prime_less_than_perfect_square_is_13_l86_86343


namespace set_intersection_complement_l86_86155
open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}
def comp_T : Set ℕ := U \ T

theorem set_intersection_complement :
  S ∩ comp_T = {1, 5} := by
  sorry

end set_intersection_complement_l86_86155


namespace train_length_l86_86570

/-- Given problem conditions -/
def speed_kmh := 72
def length_platform_m := 270
def time_sec := 26

/-- Convert speed to meters per second -/
def speed_mps := speed_kmh * 1000 / 3600

/-- Calculate the total distance covered -/
def distance_covered := speed_mps * time_sec

theorem train_length :
  (distance_covered - length_platform_m) = 250 :=
by
  sorry

end train_length_l86_86570


namespace domain_of_f_intervals_of_monotonicity_extremal_values_l86_86265

noncomputable def f (x : ℝ) := (1 / 2) * x ^ 2 - 5 * x + 4 * Real.log x 

theorem domain_of_f : ∀ x, 0 < x → f x = (1 / 2) * x ^ 2 - 5 * x + 4 * Real.log x :=
by
  intro x hx
  exact rfl

theorem intervals_of_monotonicity :
  (∀ x, 0 < x ∧ x < 1 → f x < f 1) ∧
  (∀ x, 1 < x ∧ x < 4 → f x > f 1 ∧ f x < f 4) ∧
  (∀ x, 4 < x → f x > f 4) :=
sorry

theorem extremal_values :
  (f 1 = - (9 / 2)) ∧ 
  (f 4 = -12 + 4 * Real.log 4) :=
sorry

end domain_of_f_intervals_of_monotonicity_extremal_values_l86_86265


namespace smallest_prime_12_less_than_perfect_square_l86_86348

theorem smallest_prime_12_less_than_perfect_square : ∃ n : ℕ, prime n ∧ ∃ k : ℕ, k^2 - n = 12 ∧ n = 13 :=
by {
  use 13,
  split,
  { exact prime_def.2 ⟨nat.prime_def_lt.mp nat.prime_two⟩, -- Proof that 13 is prime, simplified
  { use 5,
    split,
    { calc
      5 ^ 2 - 13
         = 25 - 13 : by rfl
    ... = 12 : by rfl,
    { refl,
    }
  }
end

end smallest_prime_12_less_than_perfect_square_l86_86348


namespace area_AMN_eq_56_25_l86_86936

open_locale real -- for π
open_locale euclidean_geometry -- for Euclidean geometry

noncomputable section

-- Define the necessary geometrical points and angles
variables (A B C H M D N : Point ℝ³)

-- Main theorem
theorem area_AMN_eq_56_25 :
  ∀ (A B C H M D N : Point ℝ³), 
    dist A B = 15 ∧ ∠ BAC = π/4 ∧ ∠ BCA = π/6 ∧ 
    foot A (line3 B C) H ∧ mid_point (line3 B C) M ∧
    mid_point (line3 H M) N -> 
  (area (triangle3 A M N) = 56.25) :=
by
  intros
  {
    sorry
  }

end area_AMN_eq_56_25_l86_86936


namespace product_units_digit_mod_10_l86_86058

theorem product_units_digit_mod_10
  (u1 u2 u3 : ℕ)
  (hu1 : u1 = 2583 % 10)
  (hu2 : u2 = 7462 % 10)
  (hu3 : u3 = 93215 % 10) :
  ((2583 * 7462 * 93215) % 10) = 0 :=
by
  have h_units1 : u1 = 3 := by sorry
  have h_units2 : u2 = 2 := by sorry
  have h_units3 : u3 = 5 := by sorry
  have h_produce_units : ((3 * 2 * 5) % 10) = 0 := by sorry
  exact h_produce_units

end product_units_digit_mod_10_l86_86058


namespace min_moves_to_find_treasure_l86_86957

theorem min_moves_to_find_treasure (cells : List ℕ) (h1 : cells = [5, 5, 5]) : 
  ∃ n, n = 2 ∧ (∀ moves, moves ≥ n → true) := sorry

end min_moves_to_find_treasure_l86_86957


namespace smallest_prime_less_than_square_l86_86357

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l86_86357


namespace find_daily_rate_second_company_l86_86803

def daily_rate_second_company (x : ℝ) : Prop :=
  let total_cost_1 := 21.95 + 0.19 * 150
  let total_cost_2 := x + 0.21 * 150
  total_cost_1 = total_cost_2

theorem find_daily_rate_second_company : daily_rate_second_company 18.95 :=
  by
  unfold daily_rate_second_company
  sorry

end find_daily_rate_second_company_l86_86803


namespace car_speed_second_hour_l86_86189

/-- The speed of the car in the first hour is 85 km/h, the average speed is 65 km/h over 2 hours,
proving that the speed of the car in the second hour is 45 km/h. -/
theorem car_speed_second_hour (v1 : ℕ) (v_avg : ℕ) (t : ℕ) (d1 : ℕ) (d2 : ℕ) 
  (h1 : v1 = 85) (h2 : v_avg = 65) (h3 : t = 2) (h4 : d1 = v1 * 1) (h5 : d2 = (v_avg * t) - d1) :
  d2 = 45 :=
sorry

end car_speed_second_hour_l86_86189


namespace strictly_increasing_interval_l86_86183

noncomputable def f (x : ℝ) : ℝ := x - x * Real.log x

theorem strictly_increasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x - x * Real.log x → ∀ y : ℝ, (0 < y ∧ y < 1 ∧ y > x) → f y > f x :=
sorry

end strictly_increasing_interval_l86_86183


namespace benny_missed_games_l86_86727

def total_games : ℕ := 39
def attended_games : ℕ := 14
def missed_games : ℕ := total_games - attended_games

theorem benny_missed_games : missed_games = 25 := by
  sorry

end benny_missed_games_l86_86727


namespace choose_2_out_of_8_l86_86285

def n : ℕ := 8
def k : ℕ := 2

theorem choose_2_out_of_8 : choose n k = 28 :=
by
  simp [n, k]
  sorry

end choose_2_out_of_8_l86_86285


namespace applesauce_ratio_is_half_l86_86022

-- Define the weights and number of pies
def total_weight : ℕ := 120
def weight_per_pie : ℕ := 4
def num_pies : ℕ := 15

-- Calculate weights used for pies and applesauce
def weight_for_pies : ℕ := num_pies * weight_per_pie
def weight_for_applesauce : ℕ := total_weight - weight_for_pies

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- Statement to prove
theorem applesauce_ratio_is_half :
  ratio weight_for_applesauce total_weight = 1 / 2 :=
by
  -- The proof goes here
  sorry

end applesauce_ratio_is_half_l86_86022


namespace determine_gx_l86_86732

/-
  Given two polynomials f(x) and h(x), we need to show that g(x) is a certain polynomial
  when f(x) + g(x) = h(x).
-/

def f (x : ℝ) : ℝ := 4 * x^5 + 3 * x^3 + x - 2
def h (x : ℝ) : ℝ := 7 * x^3 - 5 * x + 4
def g (x : ℝ) : ℝ := -4 * x^5 + 4 * x^3 - 4 * x + 6

theorem determine_gx (x : ℝ) : f x + g x = h x :=
by
  -- proof will go here
  sorry

end determine_gx_l86_86732


namespace min_m_for_four_elements_l86_86613

open Set

theorem min_m_for_four_elements (n : ℕ) (hn : n ≥ 2) :
  ∃ m, m = 2 * n + 2 ∧ 
  (∀ (S : Finset ℕ), S.card = m → 
    (∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a = b + c + d)) :=
by
  sorry

end min_m_for_four_elements_l86_86613


namespace rectangular_prism_pairs_l86_86705

def total_pairs_of_edges_in_rect_prism_different_dimensions (length width height : ℝ) 
  (h1 : length ≠ width) 
  (h2 : width ≠ height) 
  (h3 : height ≠ length) 
  : ℕ :=
66

theorem rectangular_prism_pairs (length width height : ℝ) 
  (h1 : length ≠ width) 
  (h2 : width ≠ height) 
  (h3 : height ≠ length) 
  : total_pairs_of_edges_in_rect_prism_different_dimensions length width height h1 h2 h3 = 66 := 
sorry

end rectangular_prism_pairs_l86_86705


namespace max_yes_answers_100_l86_86086

-- Define the maximum number of "Yes" answers that could be given in a lineup of n people
def maxYesAnswers (n : ℕ) : ℕ :=
  if n = 0 then 0 else 1 + (n - 2)

theorem max_yes_answers_100 : maxYesAnswers 100 = 99 :=
  by sorry

end max_yes_answers_100_l86_86086


namespace original_triangle_area_l86_86388

theorem original_triangle_area :
  let S_perspective := (1 / 2) * 1 * 1 * Real.sin (Real.pi / 3)
  let S_ratio := Real.sqrt 2 / 4
  let S_perspective_value := Real.sqrt 3 / 4
  let S_original := S_perspective_value / S_ratio
  S_original = Real.sqrt 6 / 2 :=
by
  sorry

end original_triangle_area_l86_86388


namespace matt_skips_correctly_l86_86161

-- Definitions based on conditions
def skips_per_second := 3
def jumping_time_minutes := 10
def seconds_per_minute := 60
def total_jumping_seconds := jumping_time_minutes * seconds_per_minute
def expected_skips := total_jumping_seconds * skips_per_second

-- Proof statement
theorem matt_skips_correctly :
  expected_skips = 1800 :=
by
  sorry

end matt_skips_correctly_l86_86161


namespace necessary_but_not_sufficient_l86_86834

theorem necessary_but_not_sufficient (x : ℝ) : (x > 1 → x > 2) = (false) ∧ (x > 2 → x > 1) = (true) := by
  sorry

end necessary_but_not_sufficient_l86_86834


namespace monomials_like_terms_l86_86634

theorem monomials_like_terms (a b : ℤ) (h1 : a + 1 = 2) (h2 : b - 2 = 3) : a + b = 6 :=
sorry

end monomials_like_terms_l86_86634


namespace find_number_l86_86862

theorem find_number (x : ℕ) (h : x + 3 * x = 20) : x = 5 :=
by
  sorry

end find_number_l86_86862


namespace part_1_part_2_l86_86610

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

theorem part_1 (m : ℝ) : (A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3}) → (m = 2) := 
by
  sorry

theorem part_2 (m : ℝ) : (A ⊆ (Set.univ \ B m)) → (m > 5 ∨ m < -3) := 
by
  sorry

end part_1_part_2_l86_86610


namespace evaluate_expression_l86_86891

theorem evaluate_expression (x : ℤ) (z : ℤ) (hx : x = 4) (hz : z = -2) : z * (z - 4 * x) = 36 :=
by
  sorry

end evaluate_expression_l86_86891


namespace latitude_approx_l86_86817

noncomputable def calculate_latitude (R h : ℝ) (θ : ℝ) : ℝ :=
  if h = 0 then θ else Real.arccos (1 / (2 * Real.pi))

theorem latitude_approx (R h θ : ℝ) (h_nonzero : h ≠ 0)
  (r1 : ℝ := R * Real.cos θ)
  (r2 : ℝ := (R + h) * Real.cos θ)
  (s : ℝ := 2 * Real.pi * h * Real.cos θ)
  (condition : s = h) :
  θ = Real.arccos (1 / (2 * Real.pi)) := by
  sorry

end latitude_approx_l86_86817


namespace red_tint_percentage_new_mixture_l86_86378

-- Definitions of the initial conditions
def original_volume : ℝ := 50
def red_tint_percentage : ℝ := 0.20
def added_red_tint : ℝ := 6

-- Definition for the proof
theorem red_tint_percentage_new_mixture : 
  let original_red_tint := red_tint_percentage * original_volume
  let new_red_tint := original_red_tint + added_red_tint
  let new_total_volume := original_volume + added_red_tint
  (new_red_tint / new_total_volume) * 100 = 28.57 :=
by
  sorry

end red_tint_percentage_new_mixture_l86_86378


namespace parabola_has_one_x_intercept_l86_86622

-- Define the equation of the parabola.
def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 4

-- Prove that the number of x-intercepts of the graph of the parabola is 1.
theorem parabola_has_one_x_intercept : (∃! y : ℝ, parabola y = 4) :=
by
  sorry

end parabola_has_one_x_intercept_l86_86622


namespace amanda_final_quiz_score_l86_86589

theorem amanda_final_quiz_score
  (average_score_4quizzes : ℕ)
  (total_quizzes : ℕ)
  (average_a : ℕ)
  (current_score : ℕ)
  (required_total_score : ℕ)
  (required_score_final_quiz : ℕ) :
  average_score_4quizzes = 92 →
  total_quizzes = 5 →
  average_a = 93 →
  current_score = 4 * average_score_4quizzes →
  required_total_score = total_quizzes * average_a →
  required_score_final_quiz = required_total_score - current_score →
  required_score_final_quiz = 97 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end amanda_final_quiz_score_l86_86589


namespace g_inv_g_inv_14_l86_86496

noncomputable def g (x : ℝ) := 3 * x - 4

noncomputable def g_inv (x : ℝ) := (x + 4) / 3

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by sorry

end g_inv_g_inv_14_l86_86496


namespace expected_value_matches_variance_matches_l86_86838

variables {N : ℕ} (I : Fin N → Bool)

-- Define the probability that a randomly chosen pair of cards matches
def p_match : ℝ := 1 / N

-- Define the indicator variable I_k
def I_k (k : Fin N) : ℝ :=
if I k then 1 else 0

-- Define the sum S of all the indicator variables
def S : ℝ := (Finset.univ.sum I_k)

-- Expected value E[I_k] is 1/N
def E_I_k : ℝ := 1 / N

-- Expected value E[S] is the sum of E[I_k] over all k, which is 1
theorem expected_value_matches : ∑ k, E_I_k = 1 := sorry

-- Variance calculation: Var[S] = E[S^2] - (E[S])^2
def E_S_sq : ℝ := (Finset.univ.sum (λ k, I_k k * I_k k)) + 
                  2 * (Finset.univ.sum (λ (jk : Fin N × Fin N), if jk.1 < jk.2 then I_k jk.1 * I_k jk.2 else 0))

theorem variance_matches : (E_S_sq - 1) = 1 := sorry

end expected_value_matches_variance_matches_l86_86838


namespace expected_correct_guesses_l86_86479

theorem expected_correct_guesses 
    (num_matches : ℕ) (num_outcomes : ℕ) 
    (prob_outcome_eq : ∀ i, (i < num_matches) → (PMF.uniform (Fin num_outcomes)).pmf i = (1 / num_outcomes : ℝ))
    (h_num_matches : num_matches = 12)
    (h_num_outcomes : num_outcomes = 3) : 
    E((uniform (Fin num_outcomes).replicate num_matches).toOuterMeasure) = 4 :=
by
  sorry

end expected_correct_guesses_l86_86479


namespace y_values_l86_86412

noncomputable def y (x : ℝ) : ℝ :=
  (Real.sin x / |Real.sin x|) + (|Real.cos x| / Real.cos x) + (Real.tan x / |Real.tan x|)

theorem y_values (x : ℝ) (h1 : 0 < x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x ≠ 0) (h4 : Real.cos x ≠ 0) (h5 : Real.tan x ≠ 0) :
  y x = 3 ∨ y x = -1 :=
sorry

end y_values_l86_86412


namespace value_of_a_plus_d_l86_86572

theorem value_of_a_plus_d 
  (a b c d : ℤ)
  (h1 : a + b = 12) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) 
  : a + d = 9 := 
  sorry

end value_of_a_plus_d_l86_86572


namespace increase_in_cases_second_day_l86_86079

-- Define the initial number of cases.
def initial_cases : ℕ := 2000

-- Define the number of recoveries on the second day.
def recoveries_day2 : ℕ := 50

-- Define the number of new cases on the third day and the recoveries on the third day.
def new_cases_day3 : ℕ := 1500
def recoveries_day3 : ℕ := 200

-- Define the total number of positive cases after the third day.
def total_cases_day3 : ℕ := 3750

-- Lean statement to prove the increase in cases on the second day is 750.
theorem increase_in_cases_second_day : 
  ∃ x : ℕ, initial_cases + x - recoveries_day2 + new_cases_day3 - recoveries_day3 = total_cases_day3 ∧ x = 750 :=
by
  sorry

end increase_in_cases_second_day_l86_86079


namespace xy_condition_l86_86371

theorem xy_condition : (∀ x y : ℝ, x^2 + y^2 = 0 → xy = 0) ∧ ¬ (∀ x y : ℝ, xy = 0 → x^2 + y^2 = 0) := 
by
  sorry

end xy_condition_l86_86371


namespace find_number_l86_86197

theorem find_number (x : ℝ) : 35 + 3 * x^2 = 89 ↔ x = 3 * Real.sqrt 2 ∨ x = -3 * Real.sqrt 2 := by
  sorry

end find_number_l86_86197


namespace sequence_general_formula_l86_86258

-- Definitions according to conditions in a)
def seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n + 1

def S (n : ℕ) (seq : ℕ → ℕ) : ℕ :=
  n * seq (n + 1) - 3 * n^2 - 4 * n

-- The proof goal
theorem sequence_general_formula (n : ℕ) (h : 0 < n) :
  seq n = 2 * n + 1 :=
by
  sorry

end sequence_general_formula_l86_86258


namespace smaller_number_l86_86049

theorem smaller_number (x y : ℤ) (h1 : x + y = 22) (h2 : x - y = 16) : y = 3 :=
by
  sorry

end smaller_number_l86_86049


namespace point_in_plane_region_l86_86082

theorem point_in_plane_region :
  let P := (0, 0)
  let Q := (2, 4)
  let R := (-1, 4)
  let S := (1, 8)
  (P.1 + P.2 - 1 < 0) ∧ ¬(Q.1 + Q.2 - 1 < 0) ∧ ¬(R.1 + R.2 - 1 < 0) ∧ ¬(S.1 + S.2 - 1 < 0) :=
by
  sorry

end point_in_plane_region_l86_86082


namespace smallest_z_minus_w_l86_86114

theorem smallest_z_minus_w {w x y z : ℕ} 
  (h1 : w * x * y * z = 9!)
  (h2 : w < x) 
  (h3 : x < y) 
  (h4 : y < z) : 
  z - w = 12 :=
sorry

end smallest_z_minus_w_l86_86114


namespace intersection_at_one_point_l86_86184

-- Define the quadratic equation derived from the intersection condition
def quadratic (y k : ℝ) : ℝ :=
  3 * y^2 - 2 * y + (k - 4)

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ :=
  (-2)^2 - 4 * 3 * (k - 4)

-- The statement of the problem in Lean
theorem intersection_at_one_point (k : ℝ) :
  (∃ y : ℝ, quadratic y k = 0 ∧ discriminant k = 0) ↔ k = 13 / 3 :=
by 
  sorry

end intersection_at_one_point_l86_86184


namespace cos_double_angle_sum_l86_86493

theorem cos_double_angle_sum (α : ℝ) (hα : 0 < α ∧ α < π / 2) 
  (h : Real.sin (α + π/6) = 3/5) : 
  Real.cos (2*α + π/12) = 31 / 50 * Real.sqrt 2 := sorry

end cos_double_angle_sum_l86_86493


namespace shape_area_l86_86287

-- Define the conditions as Lean definitions
def side_length : ℝ := 3
def num_squares : ℕ := 4

-- Prove that the area of the shape is 36 cm² given the conditions
theorem shape_area : num_squares * (side_length * side_length) = 36 := by
    -- The proof is skipped with sorry
    sorry

end shape_area_l86_86287


namespace box_cubes_no_green_face_l86_86680

theorem box_cubes_no_green_face (a b c : ℕ) (h_a2 : a > 2) (h_b2 : b > 2) (h_c2 : c > 2)
  (h_no_green_face : (a-2)*(b-2)*(c-2) = (a*b*c) / 3) :
  (a, b, c) = (7, 30, 4) ∨ (a, b, c) = (8, 18, 4) ∨ (a, b, c) = (9, 14, 4) ∨
  (a, b, c) = (10, 12, 4) ∨ (a, b, c) = (5, 27, 5) ∨ (a, b, c) = (6, 12, 5) ∨
  (a, b, c) = (7, 9, 5) ∨ (a, b, c) = (6, 8, 6) :=
sorry

end box_cubes_no_green_face_l86_86680


namespace set_of_x_satisfying_inequality_l86_86095

theorem set_of_x_satisfying_inequality : 
  {x : ℝ | (x - 2)^2 < 9} = {x : ℝ | -1 < x ∧ x < 5} :=
by
  sorry

end set_of_x_satisfying_inequality_l86_86095


namespace unique_x_inequality_l86_86910

theorem unique_x_inequality (a : ℝ) : (∀ x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2 → (a = 1 ∨ a = 2)) :=
by
  sorry

end unique_x_inequality_l86_86910


namespace interval_of_a_l86_86915

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then Real.exp x + x^2 else Real.exp (-x) + x^2

theorem interval_of_a (a : ℝ) :
  f (-a) + f a ≤ 2 * f 1 → -1 ≤ a ∧ a ≤ 1 :=
sorry

end interval_of_a_l86_86915


namespace no_integers_exist_l86_86085

theorem no_integers_exist :
  ¬ ∃ a b : ℤ, ∃ x y : ℤ, a^5 * b + 3 = x^3 ∧ a * b^5 + 3 = y^3 :=
by
  sorry

end no_integers_exist_l86_86085


namespace chemical_transport_problem_l86_86776

theorem chemical_transport_problem :
  (∀ (w r : ℕ), r = w + 420 →
  (900 / r) = (600 / (10 * w)) →
  w = 30 ∧ r = 450) ∧ 
  (∀ (x : ℕ), x + 450 * 3 * 2 + 60 * x ≥ 3600 → x = 15) := by
  sorry

end chemical_transport_problem_l86_86776


namespace casey_nail_decorating_time_l86_86091

theorem casey_nail_decorating_time :
  let coat_application_time := 20
  let coat_drying_time := 20
  let pattern_time := 40
  let total_time := 3 * (coat_application_time + coat_drying_time) + pattern_time
  total_time = 160 :=
by
  let coat_application_time := 20
  let coat_drying_time := 20
  let pattern_time := 40
  let total_time := 3 * (coat_application_time + coat_drying_time) + pattern_time
  trivial

end casey_nail_decorating_time_l86_86091


namespace find_k_l86_86016

theorem find_k (d : ℤ) (h : d ≠ 0) (a : ℤ → ℤ) 
  (a_def : ∀ n, a n = 4 * d + (n - 1) * d) 
  (geom_mean_condition : ∃ k, a k * a k = a 1 * a 6) : 
  ∃ k, k = 3 := 
by
  sorry

end find_k_l86_86016


namespace dodecagon_area_l86_86395

theorem dodecagon_area (s : ℝ) (n : ℕ) (angles : ℕ → ℝ)
  (h_s : s = 10) (h_n : n = 12) 
  (h_angles : ∀ i, angles i = if i % 3 == 2 then 270 else 90) :
  ∃ area : ℝ, area = 500 := 
sorry

end dodecagon_area_l86_86395


namespace sum_of_eight_numbers_l86_86523

theorem sum_of_eight_numbers (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 := 
begin 
  sorry 
end

end sum_of_eight_numbers_l86_86523


namespace max_cars_quotient_div_10_l86_86667

theorem max_cars_quotient_div_10 (n : ℕ) (h1 : ∀ v : ℕ, v ≥ 20 * n) (h2 : ∀ d : ℕ, d = 5* (n + 1)) :
  (4000 / 10 = 400) := by
  sorry

end max_cars_quotient_div_10_l86_86667


namespace smallest_prime_12_less_than_perfect_square_l86_86353

theorem smallest_prime_12_less_than_perfect_square : ∃ p : ℕ, prime p ∧ ∃ n : ℤ, p = n^2 - 12 ∧ p = 13 := 
by
  sorry

end smallest_prime_12_less_than_perfect_square_l86_86353


namespace X_investment_l86_86366

theorem X_investment (P : ℝ) 
  (Y_investment : ℝ := 42000)
  (Z_investment : ℝ := 48000)
  (Z_joins_at : ℝ := 4)
  (total_profit : ℝ := 14300)
  (Z_share : ℝ := 4160) :
  (P * 12 / (P * 12 + Y_investment * 12 + Z_investment * (12 - Z_joins_at))) * total_profit = Z_share → P = 35700 :=
by
  sorry

end X_investment_l86_86366


namespace sum_of_eight_numbers_on_cards_l86_86507

theorem sum_of_eight_numbers_on_cards :
  ∃ (a b c d e f g h : ℕ),
  (a + b) * (c + d) * (e + f) * (g + h) = 330 ∧
  (a + b + c + d + e + f + g + h) = 21 :=
by
  sorry

end sum_of_eight_numbers_on_cards_l86_86507


namespace total_amount_spent_l86_86558

/-
  Define the original prices of the games, discount rate, and tax rate.
-/
def batman_game_price : ℝ := 13.60
def superman_game_price : ℝ := 5.06
def discount_rate : ℝ := 0.20
def tax_rate : ℝ := 0.08

/-
  Prove that the total amount spent including discounts and taxes equals $16.12.
-/
theorem total_amount_spent :
  let batman_discount := batman_game_price * discount_rate
  let superman_discount := superman_game_price * discount_rate
  let batman_discounted_price := batman_game_price - batman_discount
  let superman_discounted_price := superman_game_price - superman_discount
  let total_before_tax := batman_discounted_price + superman_discounted_price
  let sales_tax := total_before_tax * tax_rate
  let total_amount := total_before_tax + sales_tax
  total_amount = 16.12 :=
by
  sorry

end total_amount_spent_l86_86558


namespace perfect_squares_count_between_50_and_200_l86_86462

theorem perfect_squares_count_between_50_and_200 :
  ∃ (N : ℕ), N = (finset.Ico 8 15).card ∧ N = 7 :=
by
  sorry

end perfect_squares_count_between_50_and_200_l86_86462


namespace part_a_39x55_5x11_l86_86836

theorem part_a_39x55_5x11 :
  ¬ (∃ (a1 a2 b1 b2 : ℕ), 
    39 = 5 * a1 + 11 * b1 ∧ 
    55 = 5 * a2 + 11 * b2) := 
  by sorry

end part_a_39x55_5x11_l86_86836


namespace sum_geometric_sequence_l86_86124

theorem sum_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (a1 : ℝ)
  (h1 : a 5 = -2) (h2 : a 8 = 16)
  (hq : q^3 = a 8 / a 5) (ha1 : a 1 = a1)
  (hS : S n = a1 * (1 - q^n) / (1 - q))
  : S 6 = 21 / 8 :=
sorry

end sum_geometric_sequence_l86_86124


namespace sum_infinite_series_eq_l86_86743

theorem sum_infinite_series_eq {x : ℝ} (hx : |x| < 1) :
  (∑' n : ℕ, (n + 1) * x^n) = 1 / (1 - x)^2 :=
by
  sorry

end sum_infinite_series_eq_l86_86743


namespace ratio_of_speeds_l86_86063

theorem ratio_of_speeds (va vb L : ℝ) (h1 : 0 < L) (h2 : 0 < va) (h3 : 0 < vb)
  (h4 : ∀ t : ℝ, t = L / va ↔ t = (L - 0.09523809523809523 * L) / vb) :
  va / vb = 21 / 19 :=
by
  sorry

end ratio_of_speeds_l86_86063


namespace riya_speed_l86_86960

theorem riya_speed 
  (R : ℝ)
  (priya_speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ)
  (h_priya_speed : priya_speed = 22)
  (h_time : time = 1)
  (h_distance : distance = 43)
  : R + priya_speed * time = distance → R = 21 :=
by 
  sorry

end riya_speed_l86_86960


namespace perpendicular_lines_condition_l86_86712

theorem perpendicular_lines_condition (m : ℝ) :
  (m = -1) ↔ ((m * 2 + 1 * m * (m - 1)) = 0) :=
sorry

end perpendicular_lines_condition_l86_86712


namespace johns_profit_is_200_l86_86149

def num_woodburnings : ℕ := 20
def price_per_woodburning : ℕ := 15
def cost_of_wood : ℕ := 100
def total_revenue : ℕ := num_woodburnings * price_per_woodburning
def profit : ℕ := total_revenue - cost_of_wood

theorem johns_profit_is_200 : profit = 200 :=
by
  -- proof steps go here
  sorry

end johns_profit_is_200_l86_86149


namespace Todd_ate_5_cupcakes_l86_86229

theorem Todd_ate_5_cupcakes (original_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) (remaining_cupcakes : ℕ) :
  original_cupcakes = 50 ∧ packages = 9 ∧ cupcakes_per_package = 5 ∧ remaining_cupcakes = packages * cupcakes_per_package →
  original_cupcakes - remaining_cupcakes = 5 :=
by
  sorry

end Todd_ate_5_cupcakes_l86_86229


namespace arithmetic_sequence_sum_l86_86435

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  a 1 + a 2 = -1 →
  a 3 = 4 →
  (a 1 + 2 * d = 4) →
  ∀ n, a n = a 1 + (n - 1) * d →
  a 4 + a 5 = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end arithmetic_sequence_sum_l86_86435


namespace probability_X_gt_4_l86_86437

noncomputable def normal_dist (μ σ : ℝ) : Measure ℝ := sorry
noncomputable def prob_within_interval (X : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

def X := normal_dist 3 1

theorem probability_X_gt_4 :
  prob_within_interval X 2 4 = 0.6826 → prob_within_interval X 4 (4 + ∞) = 0.1587 :=
sorry

end probability_X_gt_4_l86_86437


namespace tangent_line_is_x_minus_y_eq_zero_l86_86319

theorem tangent_line_is_x_minus_y_eq_zero : 
  ∀ (f : ℝ → ℝ) (x y : ℝ), 
  f x = x^3 - 2 * x → 
  (x, y) = (1, 1) → 
  (∃ (m : ℝ), m = 3 * (1:ℝ)^2 - 2 ∧ (y - 1) = m * (x - 1)) → 
  x - y = 0 :=
by
  intros f x y h_func h_point h_tangent
  sorry

end tangent_line_is_x_minus_y_eq_zero_l86_86319


namespace price_of_second_candy_l86_86568

variables (X P : ℝ)

-- Conditions
def total_weight (X : ℝ) := X + 6.25 = 10
def total_value (X P : ℝ) := 3.50 * X + 6.25 * P = 40

-- Proof problem
theorem price_of_second_candy (h1 : total_weight X) (h2 : total_value X P) : P = 4.30 :=
by 
  sorry

end price_of_second_candy_l86_86568


namespace largest_base8_3digit_to_base10_l86_86563

theorem largest_base8_3digit_to_base10 : (7 * 8^2 + 7 * 8^1 + 7 * 8^0) = 511 := by
  sorry

end largest_base8_3digit_to_base10_l86_86563


namespace new_average_contribution_75_l86_86490

-- Define the conditions given in the problem
def original_contributions : ℝ := 1
def johns_donation : ℝ := 100
def increase_rate : ℝ := 1.5

-- Define a function to calculate the new average contribution size
def new_total_contributions (A : ℝ) := A + johns_donation
def new_average_contribution (A : ℝ) := increase_rate * A

-- Theorem to prove that the new average contribution size is $75
theorem new_average_contribution_75 (A : ℝ) :
  new_total_contributions A / (original_contributions + 1) = increase_rate * A →
  A = 50 →
  new_average_contribution A = 75 :=
by
  intros h1 h2
  rw [new_average_contribution, h2]
  sorry

end new_average_contribution_75_l86_86490


namespace number_of_perfect_squares_between_50_and_200_l86_86452

theorem number_of_perfect_squares_between_50_and_200 :
  ∃ n: ℕ, 50 < n^2 ∧ n^2 < 200 ∧ (14 - 8 + 1 = 7) := sorry

end number_of_perfect_squares_between_50_and_200_l86_86452


namespace solve_inequality_for_a_l86_86249

theorem solve_inequality_for_a (a : ℝ) :
  (∀ x : ℝ, abs (x^2 + 3 * a * x + 4 * a) ≤ 3 → x = -3 * a / 2)
  ↔ (a = 8 + 2 * Real.sqrt 13 ∨ a = 8 - 2 * Real.sqrt 13) :=
by 
  sorry

end solve_inequality_for_a_l86_86249


namespace ending_number_divisible_by_9_l86_86050

theorem ending_number_divisible_by_9 (E : ℕ) 
  (h1 : ∀ n, 10 ≤ n → n ≤ E → n % 9 = 0 → ∃ m ≥ 1, n = 18 + 9 * (m - 1)) 
  (h2 : (E - 18) / 9 + 1 = 111110) : 
  E = 999999 :=
by
  sorry

end ending_number_divisible_by_9_l86_86050


namespace proof_problem_l86_86326

-- Definitions coming from the conditions
def num_large_divisions := 12
def num_small_divisions_per_large := 5
def seconds_per_small_division := 1
def seconds_per_large_division := num_small_divisions_per_large * seconds_per_small_division
def start_position := 5
def end_position := 9
def divisions_moved := end_position - start_position
def total_seconds_actual := divisions_moved * seconds_per_large_division
def total_seconds_claimed := 4

-- The theorem stating the false claim
theorem proof_problem : total_seconds_actual ≠ total_seconds_claimed :=
by {
  -- We skip the actual proof as instructed
  sorry
}

end proof_problem_l86_86326


namespace farm_distance_is_6_l86_86403

noncomputable def distance_to_farm (initial_gallons : ℕ) 
  (consumption_rate : ℕ) (supermarket_distance : ℕ) 
  (outbound_distance : ℕ) (remaining_gallons : ℕ) : ℕ :=
initial_gallons * consumption_rate - 
  (2 * supermarket_distance + 2 * outbound_distance - remaining_gallons * consumption_rate)

theorem farm_distance_is_6 : 
  distance_to_farm 12 2 5 2 2 = 6 :=
by
  sorry

end farm_distance_is_6_l86_86403


namespace reciprocal_of_negative_2023_l86_86986

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l86_86986


namespace mean_cost_of_diesel_l86_86707

-- Define the diesel rates and the number of years.
def dieselRates : List ℝ := [1.2, 1.3, 1.8, 2.1]
def years : ℕ := 4

-- Define the mean calculation and the proof requirement.
theorem mean_cost_of_diesel (h₁ : dieselRates = [1.2, 1.3, 1.8, 2.1]) 
                               (h₂ : years = 4) : 
  (dieselRates.sum / years) = 1.6 :=
by
  sorry

end mean_cost_of_diesel_l86_86707


namespace find_k_l86_86158

theorem find_k 
  (m_eq : ∀ x : ℝ, ∃ y : ℝ, y = 4 * x + 2)
  (n_eq : ∀ x : ℝ, ∃ y : ℝ, y = k * x - 8)
  (intersect : ∃ x y : ℝ, x = -2 ∧ y = -6 ∧ 4 * x + 2 = y ∧ k * x - 8 = y) :
  k = -1 := 
sorry

end find_k_l86_86158


namespace smallest_number_divisibility_l86_86206

theorem smallest_number_divisibility :
  ∃ x, (x + 3) % 70 = 0 ∧ (x + 3) % 100 = 0 ∧ (x + 3) % 84 = 0 ∧ x = 6303 :=
sorry

end smallest_number_divisibility_l86_86206


namespace probability_sum_18_l86_86925

theorem probability_sum_18:
  (∑ k in {1,2,3,4,5,6}, k = 6)^4 * (probability {d₁ d₂ d₃ d₄ : ℕ | d₁ + d₂ + d₃ + d₄ = 18} 6 6) = 5 / 216 := 
sorry

end probability_sum_18_l86_86925


namespace least_positive_integer_n_l86_86752

theorem least_positive_integer_n : ∃ (n : ℕ), (1 / (n : ℝ) - 1 / (n + 1) < 1 / 100) ∧ ∀ m, m < n → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 100) :=
sorry

end least_positive_integer_n_l86_86752


namespace width_of_cistern_is_6_l86_86715

-- Length of the cistern
def length : ℝ := 8

-- Breadth of the water surface
def breadth : ℝ := 1.85

-- Total wet surface area
def total_wet_surface_area : ℝ := 99.8

-- Let w be the width of the cistern
def width (w : ℝ) : Prop :=
  total_wet_surface_area = (length * w) + 2 * (length * breadth) + 2 * (w * breadth)

theorem width_of_cistern_is_6 : width 6 :=
  by
    -- This proof is omitted. The statement asserts that the width is 6 meters.
    sorry

end width_of_cistern_is_6_l86_86715


namespace reciprocal_of_neg_2023_l86_86988

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86988


namespace sum_of_numbers_on_cards_l86_86503

-- Define the natural numbers condition
variables {a b c d e f g h : ℕ}

-- The theorem statement
theorem sum_of_numbers_on_cards (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_numbers_on_cards_l86_86503


namespace cards_sum_l86_86521

theorem cards_sum (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
by
  sorry

end cards_sum_l86_86521


namespace symmetric_line_equation_l86_86180

theorem symmetric_line_equation (x y : ℝ) :
  (2 : ℝ) * (2 - x) + (3 : ℝ) * (-2 - y) - 6 = 0 → 2 * x + 3 * y + 8 = 0 :=
by
  sorry

end symmetric_line_equation_l86_86180


namespace reciprocal_of_x_l86_86924

theorem reciprocal_of_x (x : ℝ) (h1 : x^3 - 2 * x^2 = 0) (h2 : x ≠ 0) : x = 2 → (1 / x = 1 / 2) :=
by {
  sorry
}

end reciprocal_of_x_l86_86924


namespace find_painted_stencils_l86_86396

variable (hourly_wage racquet_wage grommet_wage stencil_wage total_earnings hours_worked racquets_restrung grommets_changed : ℕ)
variable (painted_stencils: ℕ)

axiom condition_hourly_wage : hourly_wage = 9
axiom condition_racquet_wage : racquet_wage = 15
axiom condition_grommet_wage : grommet_wage = 10
axiom condition_stencil_wage : stencil_wage = 1
axiom condition_total_earnings : total_earnings = 202
axiom condition_hours_worked : hours_worked = 8
axiom condition_racquets_restrung : racquets_restrung = 7
axiom condition_grommets_changed : grommets_changed = 2

theorem find_painted_stencils :
  painted_stencils = 5 :=
by
  -- Given:
  -- hourly_wage = 9
  -- racquet_wage = 15
  -- grommet_wage = 10
  -- stencil_wage = 1
  -- total_earnings = 202
  -- hours_worked = 8
  -- racquets_restrung = 7
  -- grommets_changed = 2

  -- We need to prove:
  -- painted_stencils = 5
  
  sorry

end find_painted_stencils_l86_86396


namespace xyz_sum_48_l86_86626

theorem xyz_sum_48 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : z * x + y = 47) : 
  x + y + z = 48 :=
sorry

end xyz_sum_48_l86_86626


namespace fence_length_l86_86811

theorem fence_length {w l : ℕ} (h1 : l = 2 * w) (h2 : 30 = 2 * l + 2 * w) : l = 10 := by
  sorry

end fence_length_l86_86811


namespace mother_returns_home_at_8_05_l86_86829

noncomputable
def xiaoMing_home_time : Nat := 7 * 60 -- 7:00 AM in minutes
def xiaoMing_speed : Nat := 40 -- in meters per minute
def mother_home_time : Nat := 7 * 60 + 20 -- 7:20 AM in minutes
def meet_point : Nat := 1600 -- in meters
def stay_time : Nat := 5 -- in minutes
def return_duration_by_bike : Nat := 20 -- in minutes

theorem mother_returns_home_at_8_05 :
    (xiaoMing_home_time + (meet_point / xiaoMing_speed) + stay_time + return_duration_by_bike) = (8 * 60 + 5) :=
by
    sorry

end mother_returns_home_at_8_05_l86_86829


namespace find_value_of_p_l86_86552

theorem find_value_of_p (p q : ℚ) (h1 : p + q = 3 / 4)
    (h2 : 45 * p^8 * q^2 = 120 * p^7 * q^3) : p = 6 / 11 :=
by
    sorry

end find_value_of_p_l86_86552


namespace largest_digit_divisible_by_6_l86_86204

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ 4517 * 10 + N % 6 = 0 ∧ ∀ m : ℕ, m ≤ 9 ∧ 4517 * 10 + m % 6 = 0 → m ≤ N :=
by
  -- Proof omitted, replace with actual proof
  sorry

end largest_digit_divisible_by_6_l86_86204


namespace part1_part2_l86_86962
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (3 * x - 1) + a * x + 3

theorem part1 (x : ℝ) : (f x 1) ≤ 5 ↔ (-1/2 : ℝ) ≤ x ∧ x ≤ 3/4 := by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, ∀ y : ℝ, f x a ≥ f y a) ↔ (-3 : ℝ) ≤ a ∧ a ≤ 3 := by
  sorry

end part1_part2_l86_86962


namespace perfect_squares_between_50_and_200_l86_86449

theorem perfect_squares_between_50_and_200 : ∃ n m : ℕ, (8 ≤ n ∧ n ≤ 14) ∧ (m - n + 1 = 7) :=
by {
  use 8, 14,
  split,
  {
    exact ⟨by norm_num, by norm_num⟩,
  },
  {
    norm_num,
  },
  sorry
}

end perfect_squares_between_50_and_200_l86_86449


namespace fill_pipe_fraction_l86_86374

theorem fill_pipe_fraction (t : ℕ) (f : ℝ) (h : t = 30) (h' : f = 1) : f = 1 :=
by
  sorry

end fill_pipe_fraction_l86_86374


namespace original_team_players_l86_86823

theorem original_team_players (n : ℕ) (W : ℝ)
    (h1 : W = n * 76)
    (h2 : (W + 110 + 60) / (n + 2) = 78) : n = 7 :=
  sorry

end original_team_players_l86_86823


namespace proposition_false_at_9_l86_86072

theorem proposition_false_at_9 (P : ℕ → Prop) 
  (h : ∀ k : ℕ, k ≥ 1 → P k → P (k + 1))
  (hne10 : ¬ P 10) : ¬ P 9 :=
by
  intro hp9
  have hp10 : P 10 := h _ (by norm_num) hp9
  contradiction

end proposition_false_at_9_l86_86072


namespace angle_A_value_sin_2B_plus_A_l86_86432

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h1 : a = 3)
variable (h2 : b = 2 * Real.sqrt 2)
variable (triangle_condition : b / (a + c) = 1 - (Real.sin C / (Real.sin A + Real.sin B)))

theorem angle_A_value : A = Real.pi / 3 :=
sorry

theorem sin_2B_plus_A (hA : A = Real.pi / 3) : 
  Real.sin (2 * B + A) = (2 * Real.sqrt 2 - Real.sqrt 3) / 6 :=
sorry

end angle_A_value_sin_2B_plus_A_l86_86432


namespace repeating_decimal_sum_l86_86101

-- Definitions from conditions
def repeating_decimal_1_3 : ℚ := 1 / 3
def repeating_decimal_2_99 : ℚ := 2 / 99

-- Statement to prove
theorem repeating_decimal_sum : repeating_decimal_1_3 + repeating_decimal_2_99 = 35 / 99 :=
by sorry

end repeating_decimal_sum_l86_86101


namespace max_ab_min_expr_l86_86494

variable {a b : ℝ}

-- Conditions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom add_eq_2 : a + b = 2

-- Statements to prove
theorem max_ab : (a * b) ≤ 1 := sorry
theorem min_expr : (2 / a + 8 / b) ≥ 9 := sorry

end max_ab_min_expr_l86_86494


namespace distinct_numbers_in_T_l86_86157

-- Definitions of sequences as functions
def seq1 (k: ℕ) : ℕ := 5 * k - 3
def seq2 (l: ℕ) : ℕ := 8 * l - 5

-- Definition of sets A and B
def A : Finset ℕ := Finset.image seq1 (Finset.range 3000)
def B : Finset ℕ := Finset.image seq2 (Finset.range 3000)

-- Definition of set T as the union of A and B
def T := A ∪ B

-- Proof statement
theorem distinct_numbers_in_T : T.card = 5400 := by
  sorry

end distinct_numbers_in_T_l86_86157


namespace find_length_l86_86710

-- Let's define the conditions given in the problem
variables (b l : ℝ)

-- Length is more than breadth by 200%
def length_eq_breadth_plus_200_percent (b l : ℝ) : Prop := l = 3 * b

-- Total cost and rate per square meter
def cost_eq_area_times_rate (total_cost rate area : ℝ) : Prop := total_cost = rate * area

-- Given values
def total_cost : ℝ := 529
def rate_per_sq_meter : ℝ := 3

-- We need to prove that the length l is approximately 23 meters
theorem find_length (h1 : length_eq_breadth_plus_200_percent b l) 
    (h2 : cost_eq_area_times_rate total_cost rate_per_sq_meter (3 * b^2)) : 
    abs (l - 23) < 1 :=
by
  sorry -- Proof to be filled

end find_length_l86_86710


namespace apples_count_l86_86187

theorem apples_count : (23 - 20 + 6 = 9) :=
by
  sorry

end apples_count_l86_86187


namespace joan_dimes_spent_l86_86946

theorem joan_dimes_spent (initial_dimes remaining_dimes spent_dimes : ℕ) 
    (h_initial: initial_dimes = 5) 
    (h_remaining: remaining_dimes = 3) : 
    spent_dimes = initial_dimes - remaining_dimes := 
by 
    sorry

end joan_dimes_spent_l86_86946


namespace impossible_to_place_integers_35x35_l86_86489

theorem impossible_to_place_integers_35x35 (f : Fin 35 → Fin 35 → ℤ) :
  (∀ i j, abs (f i j - f (i + 1) j) ≤ 18 ∧ abs (f i j - f i (j + 1)) ≤ 18) →
  ∃ i j, i ≠ j ∧ f i j = f i j → False :=
by sorry

end impossible_to_place_integers_35x35_l86_86489


namespace average_a_b_l86_86807

-- Defining the variables A, B, C
variables (A B C : ℝ)

-- Given conditions
def condition1 : Prop := (A + B + C) / 3 = 45
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 31

-- The theorem stating that the average weight of a and b is 40 kg
theorem average_a_b (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 B) : (A + B) / 2 = 40 :=
sorry

end average_a_b_l86_86807


namespace smallest_x_l86_86471

theorem smallest_x (x y : ℝ) (h1 : 4 < x) (h2 : x < 8) (h3 : 8 < y) (h4 : y < 12) (h5 : y - x = 7) :
  ∃ ε > 0, x = 4 + ε :=
by
  sorry

end smallest_x_l86_86471


namespace correct_transformation_l86_86364

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0) : (a / b = 2 * a / 2 * b) :=
by
  sorry

end correct_transformation_l86_86364


namespace second_player_always_wins_l86_86192

open Nat

theorem second_player_always_wins (cards : Finset ℕ) (h_card_count : cards.card = 16) :
  ∃ strategy : ℕ → ℕ, ∀ total_score : ℕ,
  total_score ≤ 22 → (total_score + strategy total_score > 22 ∨ 
  (∃ next_score : ℕ, total_score + next_score ≤ 22 ∧ strategy (total_score + next_score) = 1)) :=
sorry

end second_player_always_wins_l86_86192


namespace points_collinear_l86_86128

theorem points_collinear 
  {a b c : ℝ} (h1 : 0 < b) (h2 : b < a) (h3 : c = Real.sqrt (a^2 - b^2))
  (α β : ℝ)
  (P : ℝ × ℝ) (hP : P = (a^2 / c, 0)) 
  (A : ℝ × ℝ) (hA : A = (a * Real.cos α, b * Real.sin α)) 
  (B : ℝ × ℝ) (hB : B = (a * Real.cos β, b * Real.sin β)) 
  (Q : ℝ × ℝ) (hQ : Q = (a * Real.cos α, -b * Real.sin α)) 
  (F : ℝ × ℝ) (hF : F = (c, 0))
  (line_through_F : (A.1 - F.1) * (B.2 - F.2) = (A.2 - F.2) * (B.1 - F.1)) :
  ∃ (k : ℝ), k * (Q.1 - P.1) = Q.2 - P.2 ∧ k * (B.1 - P.1) = B.2 - P.2 :=
by {
  sorry
}

end points_collinear_l86_86128


namespace transform_f_to_shift_left_l86_86060

theorem transform_f_to_shift_left (f : ℝ → ℝ) :
  ∀ x : ℝ, f (2 * x - 1) = f (2 * (x - 1) + 1) := by
  sorry

end transform_f_to_shift_left_l86_86060


namespace find_f_2_l86_86619

theorem find_f_2 (f : ℝ → ℝ) (h : ∀ x, f (1 / x + 1) = 2 * x + 3) : f 2 = 5 :=
by
  sorry

end find_f_2_l86_86619


namespace geometric_seq_problem_l86_86001

-- Definitions to capture the geometric sequence and the known condition
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

variables (a : ℕ → ℝ)

-- Given the condition a_1 * a_8^3 * a_15 = 243
axiom geom_seq_condition : a 1 * (a 8)^3 * a 15 = 243

theorem geometric_seq_problem 
  (h : is_geometric_sequence a) : (a 9)^3 / (a 11) = 9 :=
sorry

end geometric_seq_problem_l86_86001


namespace trapezoid_division_areas_l86_86177

open Classical

variable (area_trapezoid : ℝ) (base1 base2 : ℝ)
variable (triangle1 triangle2 triangle3 triangle4 : ℝ)

theorem trapezoid_division_areas 
  (h1 : area_trapezoid = 3) 
  (h2 : base1 = 1) 
  (h3 : base2 = 2) 
  (h4 : triangle1 = 1 / 3)
  (h5 : triangle2 = 2 / 3)
  (h6 : triangle3 = 2 / 3)
  (h7 : triangle4 = 4 / 3) :
  triangle1 + triangle2 + triangle3 + triangle4 = area_trapezoid :=
by
  sorry

end trapezoid_division_areas_l86_86177


namespace perfect_squares_count_between_50_and_200_l86_86448

theorem perfect_squares_count_between_50_and_200 : 
  let count := (λ n m : ℤ, n - m + 1) in
  ∃ n m : ℕ, 50 < n^2 ∧ n^2 < 200 ∧ 50 < m^2 ∧ m^2 < 200 ∧ count m n = 7 :=
begin
  sorry
end

end perfect_squares_count_between_50_and_200_l86_86448


namespace factorization_example_l86_86181

theorem factorization_example :
  (x : ℝ) → (x^2 + 6 * x + 9 = (x + 3)^2) :=
by
  sorry

end factorization_example_l86_86181


namespace value_of_f_ln6_l86_86011

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x + Real.exp x else -(x + Real.exp (-x))

theorem value_of_f_ln6 : (f (Real.log 6)) = Real.log 6 - (1/6) :=
by
  sorry

end value_of_f_ln6_l86_86011


namespace number_of_perfect_squares_between_50_and_200_l86_86454

theorem number_of_perfect_squares_between_50_and_200 :
  ∃ n: ℕ, 50 < n^2 ∧ n^2 < 200 ∧ (14 - 8 + 1 = 7) := sorry

end number_of_perfect_squares_between_50_and_200_l86_86454


namespace clock_angle_34030_l86_86087

noncomputable def calculate_angle (h m s : ℕ) : ℚ :=
  abs ((60 * h - 11 * (m + s / 60)) / 2)

theorem clock_angle_34030 : calculate_angle 3 40 30 = 130 :=
by
  sorry

end clock_angle_34030_l86_86087


namespace trip_time_40mph_l86_86556

noncomputable def trip_time_80mph : ℝ := 6.75
noncomputable def speed_80mph : ℝ := 80
noncomputable def speed_40mph : ℝ := 40

noncomputable def distance : ℝ := speed_80mph * trip_time_80mph

theorem trip_time_40mph : distance / speed_40mph = 13.50 :=
by
  sorry

end trip_time_40mph_l86_86556


namespace find_k_l86_86947

def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (k : ℤ)
  (h1 : f a b c 1 = 0)
  (h2 : 60 < f a b c 9 ∧ f a b c 9 < 70)
  (h3 : 90 < f a b c 10 ∧ f a b c 10 < 100)
  (h4 : ∃ k : ℤ, 10000 * k < f a b c 100 ∧ f a b c 100 < 10000 * (k + 1))
  : k = 2 :=
sorry

end find_k_l86_86947


namespace find_x_l86_86636

theorem find_x (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 :=
by
  sorry

end find_x_l86_86636


namespace count_perfect_squares_50_to_200_l86_86460

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem count_perfect_squares_50_to_200 :
  {n : ℕ | 50 < n ∧ n < 200 ∧ is_perfect_square n}.to_finset.card = 7 :=
by
  sorry

end count_perfect_squares_50_to_200_l86_86460


namespace lasagna_pieces_l86_86793

-- Definition of the conditions
def manny_piece := 1
def aaron_piece := 0
def kai_piece := 2 * manny_piece
def raphael_piece := 0.5 * manny_piece
def lisa_piece := 2 + 0.5 * raphael_piece

-- The main theorem statement proving the total number of pieces
theorem lasagna_pieces : 
  manny_piece + aaron_piece + kai_piece + raphael_piece + lisa_piece = 6 :=
by
  -- Proof is omitted
  sorry

end lasagna_pieces_l86_86793


namespace geometric_sequence_proof_l86_86485

-- Define a geometric sequence with first term 1 and common ratio q with |q| ≠ 1
noncomputable def geometric_sequence (q : ℝ) (n : ℕ) : ℝ :=
  if h : |q| ≠ 1 then (1 : ℝ) * q ^ (n - 1) else 0

-- m should be 11 given the conditions
theorem geometric_sequence_proof (q : ℝ) (m : ℕ) (h : |q| ≠ 1) 
  (hm : geometric_sequence q m = geometric_sequence q 1 * geometric_sequence q 2 * geometric_sequence q 3 * geometric_sequence q 4 * geometric_sequence q 5 ) : 
  m = 11 :=
by
  sorry

end geometric_sequence_proof_l86_86485


namespace anna_has_9_cupcakes_left_l86_86232

def cupcakes_left (initial : ℕ) (given_away_fraction : ℚ) (eaten : ℕ) : ℕ :=
  let remaining = initial * (1 - given_away_fraction)
  remaining - eaten

theorem anna_has_9_cupcakes_left :
  cupcakes_left 60 (4/5 : ℚ) 3 = 9 := by
  sorry

end anna_has_9_cupcakes_left_l86_86232


namespace merchant_profit_l86_86580

theorem merchant_profit 
  (CP MP SP profit : ℝ)
  (markup_percentage discount_percentage : ℝ)
  (h1 : CP = 100)
  (h2 : markup_percentage = 0.40)
  (h3 : discount_percentage = 0.10)
  (h4 : MP = CP + (markup_percentage * CP))
  (h5 : SP = MP - (discount_percentage * MP))
  (h6 : profit = SP - CP) :
  profit / CP * 100 = 26 :=
by sorry

end merchant_profit_l86_86580


namespace find_theta_l86_86815

theorem find_theta (R h : ℝ) (θ : ℝ) 
  (r1_def : r1 = R * Real.cos θ)
  (r2_def : r2 = (R + h) * Real.cos θ)
  (s_def : s = 2 * π * h * Real.cos θ)
  (s_eq_h : s = h) : 
  θ = Real.arccos (1 / (2 * π)) :=
by
  sorry

end find_theta_l86_86815


namespace sum_of_eight_numbers_l86_86522

theorem sum_of_eight_numbers (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 := 
begin 
  sorry 
end

end sum_of_eight_numbers_l86_86522


namespace range_of_b2_plus_c2_l86_86003

theorem range_of_b2_plus_c2 (A B C : ℝ) (a b c : ℝ) 
  (h1 : (a - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C)
  (ha : a = Real.sqrt 3)
  (hAcute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π) :
  (∃ x, 5 < x ∧ x ≤ 6 ∧ x = b^2 + c^2) :=
sorry

end range_of_b2_plus_c2_l86_86003


namespace roots_of_polynomial_l86_86887

noncomputable def polynomial (m z : ℝ) : ℝ :=
  z^3 - (m^2 - m + 7) * z - (3 * m^2 - 3 * m - 6)

theorem roots_of_polynomial (m z : ℝ) (h : polynomial m (-1) = 0) :
  (m = 3 ∧ z = 4 ∨ z = -3) ∨ (m = -2 ∧ sorry) :=
sorry

end roots_of_polynomial_l86_86887


namespace smallest_prime_12_less_than_square_l86_86337

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l86_86337


namespace probability_750_occurrences_probability_710_occurrences_l86_86045

noncomputable section

-- Define the binomial distribution parameters
def p : ℝ := 0.8
def q : ℝ := 1 - p
def n : ℕ := 900
def mean : ℝ := n * p
def variance : ℝ := n * p * q
def std_dev : ℝ := Real.sqrt variance

-- Question (a): Prove the probability of event A occurring 750 times is approximately 0.00146
theorem probability_750_occurrences : 
  Pr (binomial n p) 750 ≈ 0.00146 :=
  sorry

-- Question (b): Prove the probability of event A occurring 710 times is approximately 0.0236
theorem probability_710_occurrences : 
  Pr (binomial n p) 710 ≈ 0.0236 :=
  sorry

end probability_750_occurrences_probability_710_occurrences_l86_86045


namespace minimum_n_divisible_20_l86_86949

theorem minimum_n_divisible_20 :
  ∃ (n : ℕ), (∀ (l : List ℕ), l.length = n → 
    ∃ (a b c d : ℕ), a ∈ l ∧ b ∈ l ∧ c ∈ l ∧ d ∈ l ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b - c - d) % 20 = 0) ∧ 
  (∀ m, m < n → ¬(∀ (l : List ℕ), l.length = m → 
    ∃ (a b c d : ℕ), a ∈ l ∧ b ∈ l ∧ c ∈ l ∧ d ∈ l ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b - c - d) % 20 = 0)) := 
⟨9, 
  by sorry, 
  by sorry⟩

end minimum_n_divisible_20_l86_86949


namespace reciprocal_of_neg_2023_l86_86999

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86999


namespace smallest_prime_12_less_than_perfect_square_l86_86354

theorem smallest_prime_12_less_than_perfect_square : ∃ p : ℕ, prime p ∧ ∃ n : ℤ, p = n^2 - 12 ∧ p = 13 := 
by
  sorry

end smallest_prime_12_less_than_perfect_square_l86_86354


namespace max_sum_of_squares_l86_86410

theorem max_sum_of_squares :
  ∃ m n : ℕ, (m ∈ Finset.range 101) ∧ (n ∈ Finset.range 101) ∧ ((n^2 - n * m - m^2)^2 = 1) ∧ (m^2 + n^2 = 10946) :=
by
  sorry

end max_sum_of_squares_l86_86410


namespace quadratic_trinomial_negative_value_l86_86669

theorem quadratic_trinomial_negative_value
  (a b c : ℝ)
  (h1 : b^2 ≥ 4 * c)
  (h2 : 1 ≥ 4 * a * c)
  (h3 : b^2 ≥ 4 * a) :
  ∃ x : ℝ, a * x^2 + b * x + c < 0 :=
by
  sorry

end quadratic_trinomial_negative_value_l86_86669


namespace sum_of_numbers_on_cards_l86_86502

-- Define the natural numbers condition
variables {a b c d e f g h : ℕ}

-- The theorem statement
theorem sum_of_numbers_on_cards (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_numbers_on_cards_l86_86502


namespace pos_integers_divisible_by_2_3_5_7_less_than_300_l86_86271

theorem pos_integers_divisible_by_2_3_5_7_less_than_300 : 
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, k < 300 → 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → k = n * (210 : ℕ) :=
by
  sorry

end pos_integers_divisible_by_2_3_5_7_less_than_300_l86_86271


namespace total_crayons_in_drawer_l86_86194

-- Definitions of conditions from a)
def initial_crayons : Nat := 9
def additional_crayons : Nat := 3

-- Statement to prove that total crayons in the drawer is 12
theorem total_crayons_in_drawer : initial_crayons + additional_crayons = 12 := sorry

end total_crayons_in_drawer_l86_86194


namespace max_value_f_max_value_at_maximum_value_of_f_l86_86264

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2) / (x^2)

theorem max_value_f : ∀ x > 0, ∃ c : ℝ, f x = (Real.log x + c) / x^2 ∧ f 1 = 2 :=
by {
  sorry
}

theorem max_value_at : f (Real.exp (-3/2)) = (Real.exp 3) / 2 :=
by {
  exact rfl
}

theorem maximum_value_of_f {x : ℝ} (hx : 0 < x) :
  ∃ y, y = (Real.exp 3) / 2 ∧ ∀ z, z > 0 → f z ≤ y :=
by {
  use f (Real.exp (-3 / 2)),
  split,
  { rw max_value_at },
  { intro z,
    rw ←max_value_at,
    sorry }
}

end max_value_f_max_value_at_maximum_value_of_f_l86_86264


namespace matrix_solution_l86_86107

-- Define the matrices
def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, 5; -1, 4]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![3, -8; 4, -11]
def P : Matrix (Fin 2) (Fin 2) ℚ := !![4/13, -31/13; 5/13, -42/13]

-- The desired Lean theorem
theorem matrix_solution :
  P * A = B :=
by sorry

end matrix_solution_l86_86107


namespace maximize_profit_at_14_yuan_and_720_l86_86865

def initial_cost : ℝ := 8
def initial_price : ℝ := 10
def initial_units_sold : ℝ := 200
def decrease_units_per_half_yuan_increase : ℝ := 10
def increase_price_per_step : ℝ := 0.5

noncomputable def profit (x : ℝ) : ℝ := 
  let selling_price := initial_price + increase_price_per_step * x
  let units_sold := initial_units_sold - decrease_units_per_half_yuan_increase * x
  (selling_price - initial_cost) * units_sold

theorem maximize_profit_at_14_yuan_and_720 :
  profit 8 = 720 ∧ (initial_price + increase_price_per_step * 8 = 14) :=
by
  sorry

end maximize_profit_at_14_yuan_and_720_l86_86865


namespace Cherry_weekly_earnings_l86_86240

theorem Cherry_weekly_earnings :
  let cost_3_5 := 2.50
  let cost_6_8 := 4.00
  let cost_9_12 := 6.00
  let cost_13_15 := 8.00
  let num_5kg := 4
  let num_8kg := 2
  let num_10kg := 3
  let num_14kg := 1
  let daily_earnings :=
    (num_5kg * cost_3_5) + (num_8kg * cost_6_8) + (num_10kg * cost_9_12) + (num_14kg * cost_13_15)
  let weekly_earnings := daily_earnings * 7
  weekly_earnings = 308 := by
  sorry

end Cherry_weekly_earnings_l86_86240


namespace value_of_k_l86_86132

theorem value_of_k (x z k : ℝ) (h1 : 2 * x - (-1) + 3 * z = 9) 
                   (h2 : x + 2 * (-1) - z = k) 
                   (h3 : -x + (-1) + 4 * z = 6) : 
                   k = -3 :=
by
  sorry

end value_of_k_l86_86132


namespace shape_of_fixed_phi_l86_86899

open EuclideanGeometry

def spherical_coordinates (ρ θ φ : ℝ) : Point ℝ :=
  ⟨ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ⟩

theorem shape_of_fixed_phi (c : ℝ) :
    {p : Point ℝ | ∃ ρ θ, p = spherical_coordinates ρ θ c} = cone :=
by sorry

end shape_of_fixed_phi_l86_86899


namespace find_n_l86_86166

variable (a b c n : ℤ)
variable (h1 : a + b + c = 100)
variable (h2 : a + b / 2 = 40)

theorem find_n : n = a - c := by
  sorry

end find_n_l86_86166


namespace inequality_solution_set_l86_86742

theorem inequality_solution_set :
  { x : ℝ | -x^2 + 2*x > 0 } = { x : ℝ | 0 < x ∧ x < 2 } :=
sorry

end inequality_solution_set_l86_86742


namespace polynomial_remainder_l86_86662

theorem polynomial_remainder :
  ∀ (q : Polynomial ℚ),
  (q.eval 2 = 8) →
  (q.eval (-3) = -10) →
  ∃ c d : ℚ, (q = (Polynomial.C (c : ℚ) * (Polynomial.X - Polynomial.C 2) * (Polynomial.X + Polynomial.C 3)) + (Polynomial.C 3.6 * Polynomial.X + Polynomial.C 0.8)) :=
by intros q h1 h2; sorry

end polynomial_remainder_l86_86662


namespace basketball_player_probability_l86_86853

def probability_makes_shot (p : ℚ) := 1 - (1 - p)

theorem basketball_player_probability :
  let p_free_throw := (4 : ℚ) / 5
  let p_HS_3pointer := (1 : ℚ) / 2
  let p_Pro_3pointer := (1 : ℚ) / 3
  let p_miss_all := (1 - p_free_throw) * (1 - p_HS_3pointer) * (1 - p_Pro_3pointer)
  let p_make_at_least_one := 1 - p_miss_all
  in p_make_at_least_one = 14 / 15 :=
by
  sorry

end basketball_player_probability_l86_86853


namespace johns_profit_l86_86152

def profit (n : ℕ) (p c : ℕ) : ℕ :=
  n * p - c

theorem johns_profit :
  profit 20 15 100 = 200 :=
by
  sorry

end johns_profit_l86_86152


namespace characteristic_function_a_characteristic_function_b_l86_86010

-- Definition for part (a):
theorem characteristic_function_a (X : ℝ → ℝ) (φ : ℝ → ℂ)
  (hφ : ∀ t_n : ℕ → ℝ, (∃ n, t_n n → 0) → (| φ (t_n n) |= 1 + o((t_n n)^2))) :
  ∃ a : ℝ, X = a :=
by sorry

-- Definition for part (b):
theorem characteristic_function_b (X : ℝ → ℝ) (φ : ℝ → ℂ)
  (hφ : ∀ t_n : ℕ → ℝ, (∃ n, t_n n → 0) → (| φ (t_n n) |= 1 + O((t_n n)^2))) :
  integrable X :=
by sorry

end characteristic_function_a_characteristic_function_b_l86_86010


namespace number_of_arrangements_l86_86714

-- Definitions of teachers and schools
inductive Teacher
| A | B | C | D | E

inductive School
| A | B | C | D

open Teacher School

-- Condition: Each school must have at least one teacher
def nonempty_schools (assignment : Teacher → School) : Prop :=
  (∃ t, assignment t = A) ∧ (∃ t, assignment t = B) ∧ (∃ t, assignment t = C) ∧ (∃ t, assignment t = D)

-- Condition: Teachers A, B, and C do not go to school B
def restriction (assignment : Teacher → School) : Prop :=
  (assignment A ≠ B) ∧ (assignment B ≠ B) ∧ (assignment C ≠ B)

-- The main statement
theorem number_of_arrangements : 
  ∃ (assignment : Teacher → School), nonempty_schools assignment ∧ restriction assignment :=
sorry

end number_of_arrangements_l86_86714


namespace domain_of_sqrt_sum_l86_86038

theorem domain_of_sqrt_sum (x : ℝ) : (1 ≤ x ∧ x ≤ 3) ↔ (x - 1 ≥ 0 ∧ 3 - x ≥ 0) := by
  sorry

end domain_of_sqrt_sum_l86_86038


namespace valentine_count_initial_l86_86956

def valentines_given : ℕ := 42
def valentines_left : ℕ := 16
def valentines_initial := valentines_given + valentines_left

theorem valentine_count_initial :
  valentines_initial = 58 :=
by
  sorry

end valentine_count_initial_l86_86956


namespace perfect_squares_between_50_and_200_l86_86466

theorem perfect_squares_between_50_and_200 : ∃ (n : ℕ), n = 7 := by
  let count := (range 15).filter (λ n, n^2 ≥ 50 ∧ n^2 ≤ 200)).length
  have h : count = 7 := by sorry
  use count
  exact h

end perfect_squares_between_50_and_200_l86_86466


namespace each_niece_gets_fifty_ice_cream_sandwiches_l86_86053

theorem each_niece_gets_fifty_ice_cream_sandwiches
  (total_sandwiches : ℕ)
  (total_nieces : ℕ)
  (h1 : total_sandwiches = 1857)
  (h2 : total_nieces = 37) :
  (total_sandwiches / total_nieces) = 50 :=
by
  sorry

end each_niece_gets_fifty_ice_cream_sandwiches_l86_86053


namespace fluctuations_B_greater_than_A_l86_86127

variable (A B : Type)
variable (mean_A mean_B : ℝ)
variable (var_A var_B : ℝ)

-- Given conditions
axiom avg_A : mean_A = 5
axiom avg_B : mean_B = 5
axiom variance_A : var_A = 0.1
axiom variance_B : var_B = 0.2

-- The proof problem statement
theorem fluctuations_B_greater_than_A : var_A < var_B :=
by sorry

end fluctuations_B_greater_than_A_l86_86127


namespace acute_angle_probability_l86_86579

noncomputable def prob_acute_angle : ℝ :=
  let m_values := [1, 2, 3, 4, 5, 6]
  let outcomes_count := (36 : ℝ)
  let good_outcomes_count := (15 : ℝ)
  good_outcomes_count / outcomes_count

theorem acute_angle_probability :
  prob_acute_angle = 5 / 12 :=
by
  sorry

end acute_angle_probability_l86_86579


namespace geometric_mean_of_4_and_9_l86_86970

theorem geometric_mean_of_4_and_9 : ∃ (G : ℝ), G = 6 ∨ G = -6 :=
by
  sorry

end geometric_mean_of_4_and_9_l86_86970


namespace shirt_price_percentage_l86_86721

variable (original_price : ℝ) (final_price : ℝ)

def calculate_sale_price (p : ℝ) : ℝ := 0.80 * p

def calculate_new_sale_price (p : ℝ) : ℝ := 0.80 * p

def calculate_final_price (p : ℝ) : ℝ := 0.85 * p

theorem shirt_price_percentage :
  (original_price = 60) →
  (final_price = calculate_final_price (calculate_new_sale_price (calculate_sale_price original_price))) →
  (final_price / original_price) * 100 = 54.4 :=
by
  intros h₁ h₂
  sorry

end shirt_price_percentage_l86_86721


namespace range_of_f_l86_86411

noncomputable def f (x : ℝ) := Real.sqrt (-x^2 - 6*x - 5)

theorem range_of_f : Set.range f = Set.Icc 0 2 := 
by
  sorry

end range_of_f_l86_86411


namespace max_distance_from_point_to_line_l86_86770

theorem max_distance_from_point_to_line (θ m : ℝ) :
  let P := (Real.cos θ, Real.sin θ)
  let d := (P.1 - m * P.2 - 2) / Real.sqrt (1 + m^2)
  ∃ (θ m : ℝ), d ≤ 3 := sorry

end max_distance_from_point_to_line_l86_86770


namespace remainder_when_four_times_n_minus_9_l86_86062

theorem remainder_when_four_times_n_minus_9
  (n : ℤ) (h : n % 5 = 3) : (4 * n - 9) % 5 = 3 := 
by 
  sorry

end remainder_when_four_times_n_minus_9_l86_86062


namespace point_in_second_quadrant_l86_86932

variable (m : ℝ)

-- Defining the conditions
def x_negative (m : ℝ) := 3 - m < 0
def y_positive (m : ℝ) := m - 1 > 0

theorem point_in_second_quadrant (h1 : x_negative m) (h2 : y_positive m) : m > 3 :=
by
  sorry

end point_in_second_quadrant_l86_86932


namespace original_number_is_24_l86_86361

theorem original_number_is_24 (N : ℕ) 
  (h1 : (N + 1) % 25 = 0)
  (h2 : 1 = 1) : N = 24 := 
sorry

end original_number_is_24_l86_86361


namespace working_mom_work_percentage_l86_86585

theorem working_mom_work_percentage :
  let total_hours_in_day := 24
  let work_hours := 8
  let gym_hours := 2
  let cooking_hours := 1.5
  let bath_hours := 0.5
  let homework_hours := 1
  let packing_hours := 0.5
  let cleaning_hours := 0.5
  let leisure_hours := 2
  let total_activity_hours := work_hours + gym_hours + cooking_hours + bath_hours + homework_hours + packing_hours + cleaning_hours + leisure_hours
  16 = total_activity_hours →
  (work_hours / total_hours_in_day) * 100 = 33.33 :=
by
  sorry

end working_mom_work_percentage_l86_86585


namespace parallel_lines_have_equal_slopes_l86_86767

theorem parallel_lines_have_equal_slopes (a : ℝ) :
  (∃ a : ℝ, (∀ y : ℝ, 2 * a * y - 1 = 0) ∧ (∃ x y : ℝ, (3 * a - 1) * x + y - 1 = 0) 
  → (∃ a : ℝ, (1 / (2 * a)) = - (3 * a - 1))) 
→ a = 1/2 :=
by
  sorry

end parallel_lines_have_equal_slopes_l86_86767


namespace option_C_correct_inequality_l86_86394

theorem option_C_correct_inequality (x : ℝ) : 
  (1 / ((x + 1) * (x - 1)) ≤ 0) ↔ (-1 < x ∧ x < 1) :=
sorry

end option_C_correct_inequality_l86_86394


namespace pencils_count_l86_86554

theorem pencils_count (pens pencils : ℕ) 
  (h_ratio : 6 * pens = 5 * pencils) 
  (h_difference : pencils = pens + 6) : 
  pencils = 36 := 
by 
  sorry

end pencils_count_l86_86554


namespace rectangle_image_l86_86012

-- A mathematically equivalent Lean 4 proof problem statement

variable (x y : ℝ)

def rectangle_OABC (x y : ℝ) : Prop :=
  (x = 0 ∧ (0 ≤ y ∧ y ≤ 3)) ∨
  (y = 0 ∧ (0 ≤ x ∧ x ≤ 2)) ∨
  (x = 2 ∧ (0 ≤ y ∧ y ≤ 3)) ∨
  (y = 3 ∧ (0 ≤ x ∧ x ≤ 2))

def transform_u (x y : ℝ) : ℝ := x^2 - y^2 + 1
def transform_v (x y : ℝ) : ℝ := x * y

theorem rectangle_image (u v : ℝ) :
  (∃ (x y : ℝ), rectangle_OABC x y ∧ u = transform_u x y ∧ v = transform_v x y) ↔
  (u, v) = (-8, 0) ∨
  (u, v) = (1, 0) ∨
  (u, v) = (5, 0) ∨
  (u, v) = (-4, 6) :=
sorry

end rectangle_image_l86_86012


namespace final_amounts_total_l86_86943

variable {Ben_initial Tom_initial Max_initial: ℕ}
variable {Ben_final Tom_final Max_final: ℕ}

theorem final_amounts_total (h1: Ben_initial = 48) 
                           (h2: Max_initial = 48) 
                           (h3: Ben_final = ((Ben_initial - Tom_initial - Max_initial) * 3 / 2))
                           (h4: Max_final = ((Max_initial * 3 / 2))) 
                           (h5: Tom_final = (Tom_initial * 2 - ((Ben_initial - Tom_initial - Max_initial) / 2) - 48))
                           (h6: Max_final = 48) :
  Ben_final + Tom_final + Max_final = 144 := 
by 
  sorry

end final_amounts_total_l86_86943


namespace proof_multiple_l86_86539

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

theorem proof_multiple (a b : ℕ) 
  (h₁ : is_multiple a 5) 
  (h₂ : is_multiple b 10) : 
  is_multiple b 5 ∧ 
  is_multiple (a + b) 5 ∧ 
  is_multiple (a + b) 2 :=
by
  sorry

end proof_multiple_l86_86539


namespace find_smallest_integer_in_set_l86_86691

def is_odd (n : ℤ) : Prop := n % 2 = 1
def median (s : Set ℤ) (m : ℤ) : Prop := 
  (∃ l u : Finset ℤ, 
      (∀ x ∈ l, x < m) ∧ 
      (∀ y ∈ u, y > m) ∧ 
      l.card = u.card ∧ 
      (l ∪ u).card % 2 = 0 ∧ 
      s = l ∪ {m} ∪ u ∧ 
      l.card + 1 + u.card = s.card)
      
def greatest (s : Set ℤ) (g : ℤ) : Prop :=
  ∃ x ∈ s, ∀ y ∈ s, y ≤ g ∧ x = g
  
theorem find_smallest_integer_in_set 
  (s : Set ℤ)
  (h1 : median s 157)
  (h2 : greatest s 169) : 
  ∃ x ∈ s, ∀ y ∈ s, x ≤ y ∧ x = 151 := 
by 
  sorry

end find_smallest_integer_in_set_l86_86691


namespace find_number_l86_86470

theorem find_number (some_number : ℤ) (h : some_number + 9 = 54) : some_number = 45 :=
sorry

end find_number_l86_86470


namespace area_proof_l86_86165

def square_side_length : ℕ := 2
def triangle_leg_length : ℕ := 2

-- Definition of the initial square area
def square_area (side_length : ℕ) : ℕ := side_length * side_length

-- Definition of the area for one isosceles right triangle
def triangle_area (leg_length : ℕ) : ℕ := (leg_length * leg_length) / 2

-- Area of the initial square
def R_square_area : ℕ := square_area square_side_length

-- Area of the 12 isosceles right triangles
def total_triangle_area : ℕ := 12 * triangle_area triangle_leg_length

-- Total area of region R
def R_area : ℕ := R_square_area + total_triangle_area

-- Smallest convex polygon S is a larger square with side length 8
def S_area : ℕ := square_area (4 * square_side_length)

-- Area inside S but outside R
def area_inside_S_outside_R : ℕ := S_area - R_area

theorem area_proof : area_inside_S_outside_R = 36 :=
by
  sorry

end area_proof_l86_86165


namespace ordered_pair_solution_l86_86605

theorem ordered_pair_solution :
  ∃ x y : ℤ, (x + y = (3 - x) + (3 - y)) ∧ (x - y = (x - 2) + (y - 2)) ∧ (x = 2) ∧ (y = 1) :=
by
  use 2, 1
  repeat { sorry }

end ordered_pair_solution_l86_86605


namespace age_ratio_proof_l86_86702

variable (j a x : ℕ)

/-- Given conditions about Jack and Alex's ages. -/
axiom h1 : j - 3 = 2 * (a - 3)
axiom h2 : j - 5 = 3 * (a - 5)

def age_ratio_in_years : Prop :=
  (3 * (a + x) = 2 * (j + x)) → (x = 1)

theorem age_ratio_proof : age_ratio_in_years j a x := by
  sorry

end age_ratio_proof_l86_86702


namespace find_number_l86_86104

theorem find_number : ∃ (x : ℝ), x + 0.303 + 0.432 = 5.485 ↔ x = 4.750 := 
sorry

end find_number_l86_86104


namespace min_n_coloring_property_l86_86898

theorem min_n_coloring_property : ∃ n : ℕ, (∀ (coloring : ℕ → Bool), 
  (∀ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧ coloring a = coloring b ∧ coloring b = coloring c → 2 * a + b = c)) ∧ n = 15 := 
sorry

end min_n_coloring_property_l86_86898


namespace num_arrangements_thm1_num_arrangements_thm2_num_arrangements_thm3_l86_86606

open Nat

def num_arrangements_A_middle (n : ℕ) : ℕ :=
  if n = 4 then factorial 4 else 0

def num_arrangements_A_not_adj_B (n : ℕ) : ℕ :=
  if n = 5 then (factorial 3) * (factorial 4 / factorial 2) else 0

def num_arrangements_A_B_not_ends (n : ℕ) : ℕ :=
  if n = 5 then (factorial 3 / factorial 2) * factorial 3 else 0

theorem num_arrangements_thm1 : num_arrangements_A_middle 4 = 24 := 
  sorry

theorem num_arrangements_thm2 : num_arrangements_A_not_adj_B 5 = 72 := 
  sorry

theorem num_arrangements_thm3 : num_arrangements_A_B_not_ends 5 = 36 := 
  sorry

end num_arrangements_thm1_num_arrangements_thm2_num_arrangements_thm3_l86_86606


namespace number_of_n_such_that_n_div_25_minus_n_is_square_l86_86607

theorem number_of_n_such_that_n_div_25_minus_n_is_square :
  ∃! n1 n2 : ℤ, ∀ n : ℤ, (n = n1 ∨ n = n2) ↔ ∃ k : ℤ, k^2 = n / (25 - n) :=
sorry

end number_of_n_such_that_n_div_25_minus_n_is_square_l86_86607


namespace min_value_of_polynomial_l86_86324

theorem min_value_of_polynomial (a : ℝ) : 
  (∀ x : ℝ, (2 * x^3 - 3 * x^2 + a) ≥ 5) → a = 6 :=
by
  sorry   -- Proof omitted

end min_value_of_polynomial_l86_86324


namespace airplane_shot_down_l86_86701

def P_A : ℝ := 0.4
def P_B : ℝ := 0.5
def P_C : ℝ := 0.8

def P_one_hit : ℝ := 0.4
def P_two_hit : ℝ := 0.7
def P_three_hit : ℝ := 1

def P_one : ℝ := (P_A * (1 - P_B) * (1 - P_C)) + ((1 - P_A) * P_B * (1 - P_C)) + ((1 - P_A) * (1 - P_B) * P_C)
def P_two : ℝ := (P_A * P_B * (1 - P_C)) + (P_A * (1 - P_B) * P_C) + ((1 - P_A) * P_B * P_C)
def P_three : ℝ := P_A * P_B * P_C

def total_probability := (P_one * P_one_hit) + (P_two * P_two_hit) + (P_three * P_three_hit)

theorem airplane_shot_down : total_probability = 0.604 := by
  sorry

end airplane_shot_down_l86_86701


namespace reciprocal_of_negative_2023_l86_86981

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l86_86981


namespace bulb_cheaper_than_lamp_by_4_l86_86007

/-- Jim bought a $7 lamp and a bulb. The bulb cost a certain amount less than the lamp. 
    He bought 2 lamps and 6 bulbs and paid $32 in all. 
    The amount by which the bulb is cheaper than the lamp is $4. -/
theorem bulb_cheaper_than_lamp_by_4
  (lamp_price bulb_price : ℝ)
  (h1 : lamp_price = 7)
  (h2 : bulb_price = 7 - 4)
  (h3 : 2 * lamp_price + 6 * bulb_price = 32) :
  (7 - bulb_price = 4) :=
by {
  sorry
}

end bulb_cheaper_than_lamp_by_4_l86_86007


namespace possible_b4b7_products_l86_86658

theorem possible_b4b7_products (b : ℕ → ℤ) (d : ℤ)
  (h_arith_sequence : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b (n + 1) > b n)
  (h_product_21 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = 21 :=
by
  sorry

end possible_b4b7_products_l86_86658


namespace sum_of_eight_numbers_l86_86513

theorem sum_of_eight_numbers (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_eight_numbers_l86_86513


namespace find_min_m_n_l86_86967

theorem find_min_m_n (m n : ℕ) (h1 : m > 1) (h2 : -1 ≤ Real.logBase m (n * x)) (h3 : Real.logBase m (n * x) ≤ 1) (h4 : ∀ x, (1 : ℝ) / (m * n) ≤ x ∧ x ≤ (m : ℝ) / n) :
  m + n = 4333 :=
by 
  sorry

end find_min_m_n_l86_86967


namespace find_Q_digit_l86_86028

theorem find_Q_digit (P Q R S T U : ℕ) (h1 : P ≠ Q) (h2 : P ≠ R) (h3 : P ≠ S)
  (h4 : P ≠ T) (h5 : P ≠ U) (h6 : Q ≠ R) (h7 : Q ≠ S) (h8 : Q ≠ T)
  (h9 : Q ≠ U) (h10 : R ≠ S) (h11 : R ≠ T) (h12 : R ≠ U) (h13 : S ≠ T)
  (h14 : S ≠ U) (h15 : T ≠ U) (h_range_P : 4 ≤ P ∧ P ≤ 9)
  (h_range_Q : 4 ≤ Q ∧ Q ≤ 9) (h_range_R : 4 ≤ R ∧ R ≤ 9)
  (h_range_S : 4 ≤ S ∧ S ≤ 9) (h_range_T : 4 ≤ T ∧ T ≤ 9)
  (h_range_U : 4 ≤ U ∧ U ≤ 9) 
  (h_sum_lines : 3 * P + 2 * Q + 3 * S + R + T + 2 * U = 100)
  (h_sum_digits : P + Q + S + R + T + U = 39) : Q = 6 :=
sorry  -- proof to be provided

end find_Q_digit_l86_86028


namespace no_solution_exists_l86_86202

theorem no_solution_exists :
  ∀ a b : ℕ, a - b = 5 ∨ b - a = 5 → a * b = 132 → false :=
by
  sorry

end no_solution_exists_l86_86202


namespace exists_tangent_circle_l86_86488

theorem exists_tangent_circle
  (A B C D P : Point ℝ)
  (h1 : ∠P B C = ∠P D A)
  (h2 : ∠P C B = ∠P A D)
  (ω : Circle ℝ)
  (hω : ω.circumscribes_convex (ConvexPolygon Victoria_corners) ABCD) : 
  ∃ Γ : Circle ℝ, 
    is_tangent Γ (Line ℝ A B) ∧ 
    is_tangent Γ (Line ℝ C D) ∧
    is_tangent Γ (Circumcircle (Triangle ℝ A B P) ∧
    is_tangent Γ (Circumcircle (Triangle ℝ C D P))

end exists_tangent_circle_l86_86488


namespace sugar_measurement_l86_86080

theorem sugar_measurement :
  let total_sugar_needed := (5 : ℚ)/2
  let cup_capacity := (1 : ℚ)/4
  (total_sugar_needed / cup_capacity) = 10 := 
by
  sorry

end sugar_measurement_l86_86080


namespace geometric_seq_second_term_l86_86682

-- Definitions
def fifth_term : ℕ → ℝ := λ n, if n = 5 then 48 else 0
def sixth_term : ℕ → ℝ := λ n, if n = 6 then 72 else 0

-- Theorem Statement
theorem geometric_seq_second_term :
  let r := sixth_term 6 / fifth_term 5,
      a := (fifth_term 5) / (r ^ 4),
      a2 := a * r in
  sixth_term 6 = 72 ∧ fifth_term 5 = 48 →
  a2 = 384 / 27 := 
begin
  sorry
end

end geometric_seq_second_term_l86_86682


namespace smallest_prime_12_less_than_square_l86_86335

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l86_86335


namespace TV_cost_exact_l86_86300

theorem TV_cost_exact (savings : ℝ) (fraction_furniture : ℝ) (fraction_tv : ℝ) (original_savings : ℝ) (tv_cost : ℝ) :
  savings = 880 →
  fraction_furniture = 3 / 4 →
  fraction_tv = 1 - fraction_furniture →
  tv_cost = fraction_tv * savings →
  tv_cost = 220 :=
by
  sorry

end TV_cost_exact_l86_86300


namespace probability_sum_18_is_1_over_54_l86_86929

open Finset

-- Definitions for a 6-faced die, four rolls, and a probability space.
def faces := {1, 2, 3, 4, 5, 6}
def dice_rolls : Finset (Finset ℕ) := product faces (product faces (product faces faces))

def valid_sum : ℕ := 18

noncomputable def probability_of_sum_18 : ℚ :=
  (dice_rolls.filter (λ r, r.sum = valid_sum)).card / dice_rolls.card

theorem probability_sum_18_is_1_over_54 :
  probability_of_sum_18 = 1 / 54 := 
  sorry

end probability_sum_18_is_1_over_54_l86_86929


namespace find_second_term_l86_86687

-- Define the terms and common ratio in the geometric sequence
def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

-- Given the fifth and sixth terms
variables (a r : ℚ)
axiom fifth_term : geometric_sequence a r 5 = 48
axiom sixth_term : geometric_sequence a r 6 = 72

-- Prove that the second term is 128/9
theorem find_second_term : geometric_sequence a r 2 = 128 / 9 :=
sorry

end find_second_term_l86_86687


namespace eval_expr_l86_86708
open Real

theorem eval_expr : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 8000 := by
  sorry

end eval_expr_l86_86708


namespace determine_all_functions_l86_86655

-- Define the natural numbers (ℕ) as positive integers
def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, k * k = x

theorem determine_all_functions (g : ℕ → ℕ) :
  (∀ m n : ℕ, is_perfect_square ((g m + n) * (m + g n))) →
  ∃ c : ℕ, ∀ n : ℕ, g n = n + c :=
by
  sorry

end determine_all_functions_l86_86655


namespace sum_of_eight_numbers_on_cards_l86_86511

theorem sum_of_eight_numbers_on_cards :
  ∃ (a b c d e f g h : ℕ),
  (a + b) * (c + d) * (e + f) * (g + h) = 330 ∧
  (a + b + c + d + e + f + g + h) = 21 :=
by
  sorry

end sum_of_eight_numbers_on_cards_l86_86511


namespace range_of_m_l86_86061

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 → ((m^2 - m) * 4^x - 2^x < 0)) → (-1 < m ∧ m < 2) :=
by
  sorry

end range_of_m_l86_86061


namespace quadratic_root_m_eq_neg_fourteen_l86_86433

theorem quadratic_root_m_eq_neg_fourteen : ∀ (m : ℝ), (∃ x : ℝ, x = 2 ∧ x^2 + 5 * x + m = 0) → m = -14 :=
by
  sorry

end quadratic_root_m_eq_neg_fourteen_l86_86433


namespace servant_leaves_after_nine_months_l86_86920

-- Definitions based on conditions
def yearly_salary : ℕ := 90 + 90
def monthly_salary : ℕ := yearly_salary / 12
def amount_received : ℕ := 45 + 90

-- The theorem to prove
theorem servant_leaves_after_nine_months :
    amount_received / monthly_salary = 9 :=
by
  -- Using the provided conditions, we establish the equality we need.
  sorry

end servant_leaves_after_nine_months_l86_86920


namespace smallest_number_exists_l86_86066

theorem smallest_number_exists (x : ℤ) :
  (x + 3) % 18 = 0 ∧ 
  (x + 3) % 70 = 0 ∧ 
  (x + 3) % 100 = 0 ∧ 
  (x + 3) % 84 = 0 → 
  x = 6297 :=
by
  sorry

end smallest_number_exists_l86_86066


namespace domain_of_f_l86_86969

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log x + Real.sqrt (2 - x)

theorem domain_of_f :
  { x : ℝ | 0 < x ∧ x ≤ 2 ∧ x ≠ 1 } = { x : ℝ | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) } :=
by
  sorry

end domain_of_f_l86_86969


namespace correct_subtraction_l86_86274

/-- Given a number n where subtracting 63 results in 8,
we aim to find the result of subtracting 36 from n
and proving that the result is 35. -/
theorem correct_subtraction (n : ℕ) (h : n - 63 = 8) : n - 36 = 35 :=
by
  sorry

end correct_subtraction_l86_86274


namespace good_committees_count_l86_86428

noncomputable def number_of_good_committees : ℕ :=
  let total_members := 30
  let enemies := 6
  let good_committees := (30 * (15 + 253)) / 3
  good_committees

theorem good_committees_count :
  number_of_good_committees = 1990 :=
by
  unfold number_of_good_committees
  -- Ensure the formula for calculating the number of good committees
  -- matches the conditions and calculations provided in the problem
  have h1 : (30 * (15 + 253)) = 8040 := by norm_num
  have h2 : 8040 / 3 = 2680 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end good_committees_count_l86_86428


namespace solve_for_x_l86_86759

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (-2, x)
def add_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def sub_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def is_parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

theorem solve_for_x : ∀ x : ℝ, is_parallel (add_vectors a (b x)) (sub_vectors a (b x)) → x = -4 :=
by
  intros x h_par
  sorry

end solve_for_x_l86_86759


namespace smallest_prime_less_than_perfect_square_is_13_l86_86344

noncomputable def smallest_prime_less_than_perfect_square : ℕ :=
  Inf {p : ℕ | ∃ k : ℕ, p = k^2 - 12 ∧ p > 0 ∧ Nat.Prime p}

theorem smallest_prime_less_than_perfect_square_is_13 :
  smallest_prime_less_than_perfect_square = 13 := by
  sorry

end smallest_prime_less_than_perfect_square_is_13_l86_86344


namespace range_of_a_l86_86495

noncomputable section

def f (a x : ℝ) := a * x^2 + 2 * a * x - Real.log (x + 1)
def g (x : ℝ) := (Real.exp x - x - 1) / (Real.exp x * (x + 1))

theorem range_of_a
  (a : ℝ)
  (h : ∀ x > 0, f a x + Real.exp (-a) > 1 / (x + 1)) : a ∈ Set.Ici (1 / 2) := 
sorry

end range_of_a_l86_86495


namespace bernardo_receives_l86_86959

theorem bernardo_receives :
  let amount_distributed (n : ℕ) : ℕ := (n * (n + 1)) / 2
  let is_valid (n : ℕ) : Prop := amount_distributed n ≤ 1000
  let bernardo_amount (k : ℕ) : ℕ := (k * (2 + (k - 1) * 3)) / 2
  ∃ k : ℕ, is_valid (15 * 3) ∧ bernardo_amount 15 = 345 :=
sorry

end bernardo_receives_l86_86959


namespace zero_polynomial_is_solution_l86_86417

noncomputable def polynomial_zero (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (p.eval x)^2 + (p.eval (1/x))^2 = (p.eval (x^2)) * (p.eval (1/(x^2))) → p = 0

theorem zero_polynomial_is_solution : ∀ p : Polynomial ℝ, (∀ x : ℝ, x ≠ 0 → (p.eval x)^2 + (p.eval (1/x))^2 = (p.eval (x^2)) * (p.eval (1/(x^2)))) → p = 0 :=
by
  sorry

end zero_polynomial_is_solution_l86_86417


namespace sufficient_but_not_necessary_l86_86261

theorem sufficient_but_not_necessary (x y : ℝ) :
  (x + y = 1 → xy ≤ 1 / 4) ∧ (∃ x y : ℝ, xy ≤ 1 / 4 ∧ x + y ≠ 1) := by
  sorry

end sufficient_but_not_necessary_l86_86261


namespace perfect_squares_50_to_200_l86_86456

theorem perfect_squares_50_to_200 : 
  ∃ (k : ℕ), k = 7 ∧ ∀ n : ℤ, 50 < n^2 ∧ n^2 < 200 -> (8 ≤ n ∧ n ≤ 14) := 
by
  sorry

end perfect_squares_50_to_200_l86_86456


namespace cat_and_dog_positions_l86_86648

def cat_position_after_365_moves : Nat :=
  let cycle_length := 9
  365 % cycle_length

def dog_position_after_365_moves : Nat :=
  let cycle_length := 16
  365 % cycle_length

theorem cat_and_dog_positions :
  cat_position_after_365_moves = 5 ∧ dog_position_after_365_moves = 13 :=
by
  sorry

end cat_and_dog_positions_l86_86648


namespace reciprocal_of_neg_2023_l86_86979

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l86_86979


namespace L_shape_area_correct_l86_86863

noncomputable def large_rectangle_area : ℕ := 12 * 7
noncomputable def small_rectangle_area : ℕ := 4 * 3
noncomputable def L_shape_area := large_rectangle_area - small_rectangle_area

theorem L_shape_area_correct : L_shape_area = 72 := by
  -- here goes your solution
  sorry

end L_shape_area_correct_l86_86863


namespace parabola_x_intercepts_l86_86624

theorem parabola_x_intercepts :
  ∃! y : ℝ, -3 * y^2 + 2 * y + 4 = y := 
by
  sorry

end parabola_x_intercepts_l86_86624


namespace oliver_boxes_total_l86_86532

theorem oliver_boxes_total (initial_boxes : ℕ := 8) (additional_boxes : ℕ := 6) : initial_boxes + additional_boxes = 14 := 
by 
  sorry

end oliver_boxes_total_l86_86532


namespace claire_sleep_hours_l86_86242

def hours_in_day := 24
def cleaning_hours := 4
def cooking_hours := 2
def crafting_hours := 5
def tailoring_hours := crafting_hours

theorem claire_sleep_hours :
  hours_in_day - (cleaning_hours + cooking_hours + crafting_hours + tailoring_hours) = 8 := by
  sorry

end claire_sleep_hours_l86_86242


namespace proof_problem_l86_86763

variable (a b c m : ℝ)

-- Condition
def condition : Prop := m = (c * a * b) / (a + b)

-- Question
def question : Prop := b = (m * a) / (c * a - m)

-- Proof statement
theorem proof_problem (h : condition a b c m) : question a b c m := 
sorry

end proof_problem_l86_86763


namespace customers_left_proof_l86_86078

def initial_customers : ℕ := 21
def tables : ℕ := 3
def people_per_table : ℕ := 3
def remaining_customers : ℕ := tables * people_per_table
def customers_left (initial remaining : ℕ) : ℕ := initial - remaining

theorem customers_left_proof : customers_left initial_customers remaining_customers = 12 := sorry

end customers_left_proof_l86_86078


namespace discount_percentage_l86_86037

theorem discount_percentage (MP CP SP : ℝ)
  (h1 : CP = 0.64 * MP)
  (h2 : SP = CP * 1.375)
  (gain_percent : 37.5 = (SP - CP) / CP * 100) :
  (MP - SP) / MP * 100 = 12 :=
by
  sorry

end discount_percentage_l86_86037


namespace remainder_when_divided_by_x_minus_2_l86_86057

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^4 - 8 * x^3 + 12 * x^2 + 20 * x - 10

-- State the theorem about the remainder when f(x) is divided by x-2
theorem remainder_when_divided_by_x_minus_2 : f 2 = 30 := by
  -- This is where the proof would go, but we use sorry to skip the proof.
  sorry

end remainder_when_divided_by_x_minus_2_l86_86057


namespace prove_0_in_A_prove_13_in_A_prove_74_in_A_prove_A_is_Z_l86_86369

variable {A : Set Int}

-- Assuming set A is closed under subtraction
axiom A_closed_under_subtraction : ∀ x y, x ∈ A → y ∈ A → x - y ∈ A
axiom A_contains_4 : 4 ∈ A
axiom A_contains_9 : 9 ∈ A

theorem prove_0_in_A : 0 ∈ A :=
sorry

theorem prove_13_in_A : 13 ∈ A :=
sorry

theorem prove_74_in_A : 74 ∈ A :=
sorry

theorem prove_A_is_Z : A = Set.univ :=
sorry

end prove_0_in_A_prove_13_in_A_prove_74_in_A_prove_A_is_Z_l86_86369


namespace moles_of_HCl_is_one_l86_86604

def moles_of_HCl_combined 
  (moles_NaHSO3 : ℝ) 
  (moles_H2O_formed : ℝ)
  (reaction_completes : moles_H2O_formed = 1) 
  (one_mole_NaHSO3_used : moles_NaHSO3 = 1) 
  : ℝ := 
by 
  sorry

theorem moles_of_HCl_is_one 
  (moles_NaHSO3 : ℝ) 
  (moles_H2O_formed : ℝ)
  (reaction_completes : moles_H2O_formed = 1) 
  (one_mole_NaHSO3_used : moles_NaHSO3 = 1) 
  : moles_of_HCl_combined moles_NaHSO3 moles_H2O_formed reaction_completes one_mole_NaHSO3_used = 1 := 
by 
  sorry

end moles_of_HCl_is_one_l86_86604


namespace arithmetic_sequence_product_l86_86656

theorem arithmetic_sequence_product (b : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ n, b (n + 1) > b n)
  (h2 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
sorry

end arithmetic_sequence_product_l86_86656


namespace women_bathing_suits_count_l86_86854

theorem women_bathing_suits_count :
  ∀ (total_bathing_suits men_bathing_suits women_bathing_suits : ℕ),
    total_bathing_suits = 19766 →
    men_bathing_suits = 14797 →
    women_bathing_suits = total_bathing_suits - men_bathing_suits →
    women_bathing_suits = 4969 := by
sorry

end women_bathing_suits_count_l86_86854


namespace total_working_days_l86_86201

theorem total_working_days 
  (D : ℕ)
  (A : ℝ)
  (B : ℝ)
  (h1 : A * (D - 2) = 80)
  (h2 : B * (D - 5) = 63)
  (h3 : A * (D - 5) = B * (D - 2) + 2) :
  D = 32 := 
sorry

end total_working_days_l86_86201


namespace reciprocal_of_neg_2023_l86_86998

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86998


namespace proof_l86_86121

def statement : Prop :=
  ∀ (a : ℝ),
    (¬ (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 - 3 * a - x + 1 > 0) ∧
    ¬ (a^2 - 4 ≥ 0 ∧
    (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 - 3 * a - x + 1 > 0)))
    → (1 ≤ a ∧ a < 2)

theorem proof : statement :=
by
  sorry

end proof_l86_86121


namespace expected_number_of_matches_variance_of_number_of_matches_l86_86846

-- Definitions of conditions
def num_pairs (N : ℕ) : Type := Fin N -> bool -- Type representing pairs of cards matching or not for an N-pair scenario

def indicator_variable (N : ℕ) (k : Fin N) : num_pairs N -> Prop :=
  λ (pairs : num_pairs N), pairs k

def matching_probability (N : ℕ) : ℝ :=
  1 / (N : ℝ)

-- Statement of the first proof problem
theorem expected_number_of_matches (N : ℕ) (pairs : num_pairs N) : 
  (∑ k, (if indicator_variable N k pairs then 1 else 0)) / N = 1 :=
sorry

-- Statement of the second proof problem
theorem variance_of_number_of_matches (N : ℕ) (pairs : num_pairs N) :
  (∑ k, (if indicator_variable N k pairs then 1 else 0) * (if indicator_variable N k pairs then 1 else 0) + 
  2 * ∑ i j, if i ≠ j then 
  (if indicator_variable N i pairs then 1 else 0) * (if indicator_variable N j pairs then 1 else 0) else 0) - 
  ((∑ k, (if indicator_variable N k pairs then 1 else 0)) / N) ^ 2 = 1 :=
sorry

end expected_number_of_matches_variance_of_number_of_matches_l86_86846


namespace geo_vs_ari_seq_l86_86940

theorem geo_vs_ari_seq (a b r d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  let a5 := a * r^4,
      b5 := b + 4 * d in
  a5 > b5 :=
by
  let a3 := a * r^2,
      b3 := b + 2 * d;
  have ha3b3 : a3 = b3, from sorry,
  have ha1b1 : a = b, from sorry,
  sorry

end geo_vs_ari_seq_l86_86940


namespace choose_8_from_16_l86_86376

theorem choose_8_from_16 :
  Nat.choose 16 8 = 12870 :=
sorry

end choose_8_from_16_l86_86376


namespace expected_number_of_matches_variance_of_number_of_matches_l86_86844

-- Definitions of conditions
def num_pairs (N : ℕ) : Type := Fin N -> bool -- Type representing pairs of cards matching or not for an N-pair scenario

def indicator_variable (N : ℕ) (k : Fin N) : num_pairs N -> Prop :=
  λ (pairs : num_pairs N), pairs k

def matching_probability (N : ℕ) : ℝ :=
  1 / (N : ℝ)

-- Statement of the first proof problem
theorem expected_number_of_matches (N : ℕ) (pairs : num_pairs N) : 
  (∑ k, (if indicator_variable N k pairs then 1 else 0)) / N = 1 :=
sorry

-- Statement of the second proof problem
theorem variance_of_number_of_matches (N : ℕ) (pairs : num_pairs N) :
  (∑ k, (if indicator_variable N k pairs then 1 else 0) * (if indicator_variable N k pairs then 1 else 0) + 
  2 * ∑ i j, if i ≠ j then 
  (if indicator_variable N i pairs then 1 else 0) * (if indicator_variable N j pairs then 1 else 0) else 0) - 
  ((∑ k, (if indicator_variable N k pairs then 1 else 0)) / N) ^ 2 = 1 :=
sorry

end expected_number_of_matches_variance_of_number_of_matches_l86_86844


namespace regular_price_of_tire_l86_86695

theorem regular_price_of_tire (x : ℝ) (h : 3 * x + 3 = 240) : x = 79 :=
by
  sorry

end regular_price_of_tire_l86_86695


namespace employee_salary_percentage_l86_86331

theorem employee_salary_percentage (A B : ℝ)
    (h1 : A + B = 450)
    (h2 : B = 180) : (A / B) * 100 = 150 := by
  sorry

end employee_salary_percentage_l86_86331


namespace length_of_DG_l86_86939

theorem length_of_DG {AB BC DG DF : ℝ} (h1 : AB = 8) (h2 : BC = 10) (h3 : DG = DF) 
  (h4 : 1/5 * (AB * BC) = 1/2 * DG^2) : DG = 4 * Real.sqrt 2 :=
by sorry

end length_of_DG_l86_86939


namespace expected_number_of_matches_variance_of_number_of_matches_l86_86848

-- Defining the conditions first, and then posing the proof statements
namespace MatchingPairs

open ProbabilityTheory

-- Probabilistic setup for indicator variables
variable (N : ℕ) (prob : ℝ := 1 / N)

-- Indicator variable Ik representing matches
@[simp] def I (k : ℕ) : ℝ := if k < N then prob else 0

-- Define the sum of expected matches S
@[simp] def S : ℝ := ∑ k in finset.range N, I N k

-- Statement: The expectation of the number of matching pairs is 1
theorem expected_number_of_matches : E[S] = 1 := sorry

-- Statement: The variance of the number of matching pairs is 1
theorem variance_of_number_of_matches : Var S = 1 := sorry

end MatchingPairs

end expected_number_of_matches_variance_of_number_of_matches_l86_86848


namespace investment_plans_count_l86_86717

theorem investment_plans_count :
  let binom := Nat.choose
  ∃ (cnt : Nat), cnt = binom 5 3 * 3! + binom 5 1 * binom 4 1 * 3 ∧ cnt = 120 :=
by
  sorry

end investment_plans_count_l86_86717


namespace heating_rate_l86_86230

/-- 
 Andy is making fudge. He needs to raise the temperature of the candy mixture from 60 degrees to 240 degrees. 
 Then, he needs to cool it down to 170 degrees. The candy heats at a certain rate and cools at a rate of 7 degrees/minute.
 It takes 46 minutes for the candy to be done. Prove that the heating rate is 5 degrees per minute.
-/
theorem heating_rate (initial_temp heating_temp cooling_temp : ℝ) (cooling_rate total_time : ℝ) 
  (h1 : initial_temp = 60) (h2 : heating_temp = 240) (h3 : cooling_temp = 170) 
  (h4 : cooling_rate = 7) (h5 : total_time = 46) : 
  ∃ (H : ℝ), H = 5 :=
by 
  -- We declare here that the rate H exists and is 5 degrees per minute.
  let H : ℝ := 5
  existsi H
  sorry

end heating_rate_l86_86230


namespace transport_cost_l86_86963

-- Define the conditions
def cost_per_kg : ℕ := 15000
def grams_per_kg : ℕ := 1000
def weight_in_grams : ℕ := 500

-- Define the main theorem stating the proof problem
theorem transport_cost
  (c : ℕ := cost_per_kg)
  (gpk : ℕ := grams_per_kg)
  (w : ℕ := weight_in_grams)
  : c * w / gpk = 7500 :=
by
  -- Since we are not required to provide the proof, adding sorry here
  sorry

end transport_cost_l86_86963


namespace possible_b4b7_products_l86_86659

theorem possible_b4b7_products (b : ℕ → ℤ) (d : ℤ)
  (h_arith_sequence : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b (n + 1) > b n)
  (h_product_21 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = 21 :=
by
  sorry

end possible_b4b7_products_l86_86659


namespace sum_of_eight_numbers_l86_86528

-- Definitions used in the conditions
variables {a b c d e f g h : ℕ}

-- Given condition
axiom product_condition : (a + b) * (c + d) * (e + f) * (g + h) = 330

-- Define individual sums
def ab_sum := a + b
def cd_sum := c + d
def ef_sum := e + f
def gh_sum := g + h

-- Define the total sum of the eight numbers on the cards
def total_sum := ab_sum + cd_sum + ef_sum + gh_sum

-- The theorem to prove
theorem sum_of_eight_numbers : total_sum = 21 := by
  have ab_sum_eq : ab_sum = 2 := sorry
  have cd_sum_eq : cd_sum = 3 := sorry
  have ef_sum_eq : ef_sum = 5 := sorry
  have gh_sum_eq : gh_sum = 11 := sorry
  rw [ab_sum_eq, cd_sum_eq, ef_sum_eq, gh_sum_eq]
  norm_num

end sum_of_eight_numbers_l86_86528


namespace find_sum_of_a_and_b_l86_86576

variable (a b w y z S : ℕ)

-- Conditions based on problem statement
axiom condition1 : 19 + w + 23 = S
axiom condition2 : 22 + y + a = S
axiom condition3 : b + 18 + z = S
axiom condition4 : 19 + 22 + b = S
axiom condition5 : w + y + 18 = S
axiom condition6 : 23 + a + z = S
axiom condition7 : 19 + y + z = S
axiom condition8 : 23 + y + b = S

theorem find_sum_of_a_and_b : a + b = 23 :=
by
  sorry  -- To be provided with the actual proof later

end find_sum_of_a_and_b_l86_86576


namespace correct_transformation_l86_86365

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0) : (a / b = 2 * a / 2 * b) :=
by
  sorry

end correct_transformation_l86_86365


namespace max_nested_fraction_value_l86_86019

-- Define the problem conditions
def numbers := (List.range 100).map (λ n => n + 1)

-- Define the nested fraction function
noncomputable def nested_fraction (l : List ℕ) : ℚ :=
  l.foldr (λ x acc => x / acc) 1

-- Prove that the maximum value of the nested fraction from 1 to 100 is 100! / 4
theorem max_nested_fraction_value :
  nested_fraction numbers = (Nat.factorial 100) / 4 :=
sorry

end max_nested_fraction_value_l86_86019


namespace smallest_n_identity_matrix_l86_86111

-- Define the rotation matrix for 150 degrees
def rotation_matrix_150 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos (150 * Real.pi / 180), -Real.sin (150 * Real.pi / 180)], 
    ![Real.sin (150 * Real.pi / 180), Real.cos (150 * Real.pi / 180)]]

-- Define the identity matrix of size 2
def I_two : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, 1]]

-- Statement of the theorem
theorem smallest_n_identity_matrix : ∃ n : ℕ, n > 0 ∧ (rotation_matrix_150 ^ n = I_two) ∧ 
  ∀ m : ℕ, (m > 0 ∧ rotation_matrix_150 ^ m = I_two) → m ≥ n := 
sorry

end smallest_n_identity_matrix_l86_86111


namespace min_value_of_function_l86_86897

noncomputable def y (x : ℝ) : ℝ := (Real.cos x) * (Real.sin (2 * x))

theorem min_value_of_function :
  ∃ x ∈ Set.Icc (-Real.pi) Real.pi, y x = -4 * Real.sqrt 3 / 9 :=
sorry

end min_value_of_function_l86_86897


namespace find_number_l86_86893

theorem find_number (x : ℝ) (h : x^2 + 50 = (x - 10)^2) : x = 2.5 :=
sorry

end find_number_l86_86893


namespace divisible_by_a_minus_one_squared_l86_86673

theorem divisible_by_a_minus_one_squared (a n : ℕ) (h : n > 0) :
  (a^(n+1) - n * (a - 1) - a) % (a - 1)^2 = 0 :=
by
  sorry

end divisible_by_a_minus_one_squared_l86_86673


namespace alice_unanswered_questions_l86_86588

theorem alice_unanswered_questions :
  ∃ (c w u : ℕ), (5 * c - 2 * w = 54) ∧ (2 * c + u = 36) ∧ (c + w + u = 30) ∧ (u = 8) :=
by
  -- proof omitted
  sorry

end alice_unanswered_questions_l86_86588


namespace toy_problem_l86_86501

theorem toy_problem :
  ∃ (n m : ℕ), 
    1500 ≤ n ∧ n ≤ 2000 ∧ 
    n % 15 = 5 ∧ n % 20 = 5 ∧ n % 30 = 5 ∧ 
    (n + m) % 12 = 0 ∧ (n + m) % 18 = 0 ∧ 
    n + m ≤ 2100 ∧ m = 31 := 
sorry

end toy_problem_l86_86501


namespace production_today_is_correct_l86_86425

theorem production_today_is_correct (n : ℕ) (P : ℕ) (T : ℕ) (average_daily_production : ℕ) (new_average_daily_production : ℕ) (h1 : n = 3) (h2 : average_daily_production = 70) (h3 : new_average_daily_production = 75) (h4 : P = n * average_daily_production) (h5 : P + T = (n + 1) * new_average_daily_production) : T = 90 :=
by
  sorry

end production_today_is_correct_l86_86425


namespace B_is_brownian_motion_l86_86660

open ProbabilityTheory

-- Given: B^{ \circ} is a Brownian bridge, which is a centered Gaussian process with a specific covariance structure.
variables (B_circ : ℝ → ℝ)
  [is_gaussian B_circ]
  (h_B_circ_zero_mean : ∀ t, t ∈ set.Icc 0 1 → 𝔼[B_circ t] = 0)
  (h_B_circ_cov : ∀ s t, s ∈ set.Icc 0 1 → t ∈ set.Icc 0 1 → 𝔼[B_circ s * B_circ t] = s * (1 - t))

-- Defining the transformed process B_t
noncomputable def B (t : ℝ) : ℝ := (1 + t) * B_circ (t / (1 + t))

-- Statement: Show that the process B = (B_t)_{t ≥ 0} is a Brownian motion
theorem B_is_brownian_motion : IsBrownianMotion (λ t, B B_circ t) :=
sorry

end B_is_brownian_motion_l86_86660


namespace ways_to_seat_people_l86_86199

noncomputable def number_of_ways : ℕ :=
  let choose_people := (Nat.choose 12 8)
  let divide_groups := (Nat.choose 8 4)
  let arrange_circular_table := (Nat.factorial 3)
  choose_people * divide_groups * (arrange_circular_table * arrange_circular_table)

theorem ways_to_seat_people :
  number_of_ways = 1247400 :=
by 
  -- proof goes here
  sorry

end ways_to_seat_people_l86_86199


namespace percentage_spent_on_household_items_eq_50_l86_86586

-- Definitions for the conditions in the problem
def MonthlyIncome : ℝ := 90000
def ClothesPercentage : ℝ := 0.25
def MedicinesPercentage : ℝ := 0.15
def Savings : ℝ := 9000

-- Definition of the statement where we need to calculate the percentage spent on household items
theorem percentage_spent_on_household_items_eq_50 :
  let ClothesExpense := ClothesPercentage * MonthlyIncome
  let MedicinesExpense := MedicinesPercentage * MonthlyIncome
  let TotalExpense := ClothesExpense + MedicinesExpense + Savings
  let HouseholdItemsExpense := MonthlyIncome - TotalExpense
  let TotalIncome := MonthlyIncome
  (HouseholdItemsExpense / TotalIncome) * 100 = 50 :=
by
  sorry

end percentage_spent_on_household_items_eq_50_l86_86586


namespace odd_n_divides_pow_fact_sub_one_l86_86298

theorem odd_n_divides_pow_fact_sub_one
  {n : ℕ} (hn_pos : n > 0) (hn_odd : n % 2 = 1)
  : n ∣ (2 ^ (Nat.factorial n) - 1) :=
sorry

end odd_n_divides_pow_fact_sub_one_l86_86298


namespace marble_158th_is_gray_l86_86386

def marble_color (n : ℕ) : String :=
  if (n % 12 < 5) then "gray"
  else if (n % 12 < 9) then "white"
  else "black"

theorem marble_158th_is_gray : marble_color 157 = "gray" := 
by
  sorry

end marble_158th_is_gray_l86_86386


namespace projection_matrix_l86_86251

open Matrix

noncomputable def P : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![4/9, -4/9, -2/9],
    ![-4/9, 4/9, 2/9],
    ![-2/9, 2/9, 1/9]]

def vector_v (x y z : ℝ) : Fin 3 → ℝ
| 0 => x
| 1 => y
| 2 => z

theorem projection_matrix (x y z : ℝ) :
  let v := vector_v x y z
  let u := vector_v 2 (-2) (-1)
in  (P.mulVec v) = (u.mul (dot_product v u / dot_product u u)) :=
by
  sorry

end projection_matrix_l86_86251


namespace jack_valid_sequences_l86_86148

-- Definitions based strictly on the conditions from Step a)
def valid_sequence_count : ℕ :=
  -- Count the valid paths under given conditions (mock placeholder definition)
  1  -- This represents the proof statement

-- The main theorem stating the proof problem
theorem jack_valid_sequences :
  valid_sequence_count = 1 := 
  sorry  -- Proof placeholder

end jack_valid_sequences_l86_86148


namespace probability_odd_sum_l86_86704

noncomputable def favorable_outcomes : ℕ := 18
noncomputable def total_outcomes : ℕ := 6 * 6

theorem probability_odd_sum :
  favorable_outcomes / total_outcomes = 1 / 2 := by
  sorry

end probability_odd_sum_l86_86704


namespace initial_books_l86_86550

variable (B : ℤ)

theorem initial_books (h1 : 4 / 6 * B = B - 3300) (h2 : 3300 = 2 / 6 * B) : B = 9900 :=
by
  sorry

end initial_books_l86_86550


namespace reciprocal_of_neg_2023_l86_86997

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86997


namespace value_of_y_l86_86133

theorem value_of_y (y : ℝ) (h : (y / 5) / 3 = 5 / (y / 3)) : y = 15 ∨ y = -15 :=
by
  sorry

end value_of_y_l86_86133


namespace school_dance_attendance_l86_86557

theorem school_dance_attendance (P : ℝ)
  (h1 : 0.1 * P = (P - (0.9 * P)))
  (h2 : 0.9 * P = (2/3) * (0.9 * P) + (1/3) * (0.9 * P))
  (h3 : 30 = (1/3) * (0.9 * P)) :
  P = 100 :=
by
  sorry

end school_dance_attendance_l86_86557


namespace arc_length_l86_86477

theorem arc_length (r : ℝ) (α : ℝ) (h_r : r = 2) (h_α : α = π / 7) : (α * r) = 2 * π / 7 := by
  sorry

end arc_length_l86_86477


namespace Christine_picked_10_pounds_l86_86875

-- Variable declarations for the quantities involved
variable (C : ℝ) -- Pounds of strawberries Christine picked
variable (pieStrawberries : ℝ := 3) -- Pounds of strawberries per pie
variable (pies : ℝ := 10) -- Number of pies
variable (totalStrawberries : ℝ := 30) -- Total pounds of strawberries for pies

-- The condition that Rachel picked twice as many strawberries as Christine
variable (R : ℝ := 2 * C)

-- The condition for the total pounds of strawberries picked by Christine and Rachel
axiom strawberries_eq : C + R = totalStrawberries

-- The goal is to prove that Christine picked 10 pounds of strawberries
theorem Christine_picked_10_pounds : C = 10 := by
  sorry

end Christine_picked_10_pounds_l86_86875


namespace problem_inequality_l86_86256

theorem problem_inequality (a b c : ℝ) : a^2 + b^2 + c^2 + 4 ≥ ab + 3*b + 2*c := 
by 
  sorry

end problem_inequality_l86_86256


namespace find_b_value_l86_86207

theorem find_b_value
  (b : ℝ)
  (eq1 : ∀ y x, 3 * y - 3 * b = 9 * x)
  (eq2 : ∀ y x, y - 2 = (b + 9) * x)
  (parallel : ∀ y1 y2 x1 x2, 
    (3 * y1 - 3 * b = 9 * x1) ∧ (y2 - 2 = (b + 9) * x2) → 
    ((3 * x1 = (b + 9) * x2) ↔ (3 = b + 9)))
  : b = -6 := 
  sorry

end find_b_value_l86_86207


namespace sum_of_eight_numbers_on_cards_l86_86510

theorem sum_of_eight_numbers_on_cards :
  ∃ (a b c d e f g h : ℕ),
  (a + b) * (c + d) * (e + f) * (g + h) = 330 ∧
  (a + b + c + d + e + f + g + h) = 21 :=
by
  sorry

end sum_of_eight_numbers_on_cards_l86_86510


namespace num_valid_m_values_for_distributing_marbles_l86_86027

theorem num_valid_m_values_for_distributing_marbles : 
  ∃ (m_values : Finset ℕ), m_values.card = 22 ∧ 
  ∀ m ∈ m_values, ∃ n : ℕ, m * n = 360 ∧ n > 1 ∧ m > 1 :=
by
  sorry

end num_valid_m_values_for_distributing_marbles_l86_86027


namespace derivative_f_monotonicity_and_extreme_points_l86_86439

open Function

-- Define the function f(x) = x^3 + x^2 - 8x + 7
def f (x : ℝ) : ℝ := x^3 + x^2 - 8*x + 7

-- Statement: Derivative of the function
theorem derivative_f : 
  ∀ x, deriv f x = 3*x^2 + 2*x - 8 :=
by 
  intro x
  simp [f]
  sorry

-- Statement: Intervals of monotonicity and extreme points
theorem monotonicity_and_extreme_points :
  (∀ x, (x < -2) → (deriv f x < 0)) ∧
  (∀ x, (-2 < x ∧ x < 4/3) → (deriv f x < 0)) ∧
  (∀ x, (x > 4/3) → (deriv f x > 0)) ∧
  (∃ x₁, x₁ = -2 ∧ is_local_max f x₁) ∧
  (∃ x₂, x₂ = 4/3 ∧ is_local_min f x₂) :=
by 
  sorry

end derivative_f_monotonicity_and_extreme_points_l86_86439


namespace initial_number_proof_l86_86314

def initial_number : ℕ := 7899665
def result : ℕ := 7899593
def factor1 : ℕ := 12
def factor2 : ℕ := 3
def factor3 : ℕ := 2

def certain_value : ℕ := (factor1 * factor2) * factor3

theorem initial_number_proof :
  initial_number - certain_value = result := by
  sorry

end initial_number_proof_l86_86314


namespace fraction_spent_on_house_rent_l86_86377

noncomputable def total_salary : ℚ :=
  let food_expense_ratio := (3 / 10 : ℚ)
  let conveyance_expense_ratio := (1 / 8 : ℚ)
  let total_expense := 3400
  total_expense * (40 / 17)

noncomputable def expenditure_on_house_rent (salary: ℚ) : ℚ :=
  let remaining_amount := 1400
  let food_and_conveyance_expense := 3400
  salary - remaining_amount - food_and_conveyance_expense

theorem fraction_spent_on_house_rent : expenditure_on_house_rent total_salary / total_salary = (2 / 5 : ℚ) := by
  sorry

end fraction_spent_on_house_rent_l86_86377


namespace trajectory_of_moving_circle_l86_86257

noncomputable def ellipse_trajectory_eq (x y : ℝ) : Prop :=
  (x^2)/25 + (y^2)/9 = 1

theorem trajectory_of_moving_circle
  (x y : ℝ)
  (A : ℝ × ℝ)
  (C : ℝ × ℝ)
  (radius_C : ℝ)
  (hC : (x + 4)^2 + y^2 = 100)
  (hA : A = (4, 0))
  (radius_C_eq : radius_C = 10) :
  ellipse_trajectory_eq x y :=
sorry

end trajectory_of_moving_circle_l86_86257


namespace train_length_l86_86387

theorem train_length (t_post t_platform l_platform : ℕ) (L : ℚ) : 
  t_post = 15 → t_platform = 25 → l_platform = 100 →
  (L / t_post) = (L + l_platform) / t_platform → 
  L = 150 :=
by 
  intros h1 h2 h3 h4
  -- Proof steps would go here
  sorry

end train_length_l86_86387


namespace geometric_progression_sum_eq_l86_86048

theorem geometric_progression_sum_eq
  (a q b : ℝ) (n : ℕ)
  (hq : q ≠ 1)
  (h : (a * (q^2^n - 1)) / (q - 1) = (b * (q^(2*n) - 1)) / (q^2 - 1)) :
  b = a + a * q :=
by
  sorry

end geometric_progression_sum_eq_l86_86048


namespace average_beef_sales_l86_86303

def ground_beef_sales.Thur : ℕ := 210
def ground_beef_sales.Fri : ℕ := 2 * ground_beef_sales.Thur
def ground_beef_sales.Sat : ℕ := 150
def ground_beef_sales.total : ℕ := ground_beef_sales.Thur + ground_beef_sales.Fri + ground_beef_sales.Sat
def ground_beef_sales.days : ℕ := 3
def ground_beef_sales.average : ℕ := ground_beef_sales.total / ground_beef_sales.days

theorem average_beef_sales (thur : ℕ) (fri : ℕ) (sat : ℕ) (days : ℕ) (total : ℕ) (avg : ℕ) :
  thur = 210 → 
  fri = 2 * thur → 
  sat = 150 → 
  total = thur + fri + sat → 
  days = 3 → 
  avg = total / days → 
  avg = 260 := by
    sorry

end average_beef_sales_l86_86303


namespace exists_five_integers_l86_86413

theorem exists_five_integers :
  ∃ (a b c d e : ℤ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
    c ≠ d ∧ c ≠ e ∧ 
    d ≠ e ∧
    ∃ (k1 k2 k3 k4 k5 : ℕ), 
      k1^2 = (a + b + c + d) ∧ 
      k2^2 = (a + b + c + e) ∧ 
      k3^2 = (a + b + d + e) ∧ 
      k4^2 = (a + c + d + e) ∧ 
      k5^2 = (b + c + d + e) := 
sorry

end exists_five_integers_l86_86413


namespace sum_series_and_convergence_l86_86744

theorem sum_series_and_convergence (x : ℝ) (h : -1 < x ∧ x < 1) :
  ∑' n, (n + 6) * x^(7 * n) = (6 - 5 * x^7) / (1 - x^7)^2 :=
by
  sorry

end sum_series_and_convergence_l86_86744


namespace expected_matches_is_one_variance_matches_is_one_l86_86842

noncomputable def indicator (k : ℕ) (matches : Finset ℕ) : ℕ :=
  if k ∈ matches then 1 else 0

def expected_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  (Finset.range N).sum (λ k, indicator k matches / N)

def variance_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  let E_S := expected_matches N matches in
  let E_S2 := (Finset.range N).sum (λ k, (indicator k matches) ^ 2 / N) in
  E_S2 - E_S ^ 2

theorem expected_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  expected_matches N matches = 1 := sorry

theorem variance_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  variance_matches N matches = 1 := sorry

end expected_matches_is_one_variance_matches_is_one_l86_86842


namespace machine_C_time_l86_86664

theorem machine_C_time (T_c : ℝ) : 
  (1/4) + (1/3) + (1/T_c) = (3/4) → T_c = 6 := 
by 
  sorry

end machine_C_time_l86_86664


namespace smallest_positive_prime_12_less_than_square_is_13_l86_86349

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_prime_12_less_than_square (k : ℕ) : Prop :=
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = k * k - 12

theorem smallest_positive_prime_12_less_than_square_is_13 :
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = 5 * 5 - 12 ∧ (∀ n : ℕ, is_prime n ∧ n > 0 ∧ n = k * k - 12 → n ≥ m) :=
begin
  use 13,
  sorry
end

end smallest_positive_prime_12_less_than_square_is_13_l86_86349


namespace factor_expression_l86_86730

theorem factor_expression (x : ℝ) : 18 * x^2 + 9 * x - 3 = 3 * (6 * x^2 + 3 * x - 1) :=
by
  sorry

end factor_expression_l86_86730


namespace probability_lakers_win_in_7_games_l86_86964

theorem probability_lakers_win_in_7_games (prob_celtics_win : ℚ) (prob_lakers_win : ℚ) (combinations_6_3 : ℕ) :
  prob_celtics_win = 3 / 4 →
  prob_lakers_win = 1 / 4 →
  combinations_6_3 = 20 →
  (combinations_6_3 * (prob_lakers_win ^ 3) * (prob_celtics_win ^ 3) * prob_lakers_win = 135 / 4096) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have binomial_prob := 20 * ((1/4:ℚ)^3) * ((3/4:ℚ)^3)
  have final_prob := binomial_prob * (1/4:ℚ)
  exact final_prob = 135 / 4096
  sorry

end probability_lakers_win_in_7_games_l86_86964


namespace floor_sqrt_20_squared_eq_16_l86_86099

theorem floor_sqrt_20_squared_eq_16 : (Int.floor (Real.sqrt 20))^2 = 16 := by
  sorry

end floor_sqrt_20_squared_eq_16_l86_86099


namespace equal_parts_division_l86_86409

theorem equal_parts_division (n : ℕ) (h : (n * n) % 4 = 0) : 
  ∃ parts : ℕ, parts = 4 ∧ ∀ (i : ℕ), i < parts → 
    ∃ p : ℕ, p = (n * n) / parts :=
by sorry

end equal_parts_division_l86_86409


namespace scientific_notation_14nm_l86_86802

theorem scientific_notation_14nm :
  0.000000014 = 1.4 * 10^(-8) := 
by 
  sorry

end scientific_notation_14nm_l86_86802


namespace pos_integers_divisible_by_2_3_5_7_less_than_300_l86_86272

theorem pos_integers_divisible_by_2_3_5_7_less_than_300 : 
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, k < 300 → 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → k = n * (210 : ℕ) :=
by
  sorry

end pos_integers_divisible_by_2_3_5_7_less_than_300_l86_86272


namespace perfect_squares_between_50_and_200_l86_86465

theorem perfect_squares_between_50_and_200 : ∃ (n : ℕ), n = 7 := by
  let count := (range 15).filter (λ n, n^2 ≥ 50 ∧ n^2 ≤ 200)).length
  have h : count = 7 := by sorry
  use count
  exact h

end perfect_squares_between_50_and_200_l86_86465


namespace intersection_P_Q_l86_86254

open Set

noncomputable def P : Set ℝ := {-1, 0, Real.sqrt 2}

def Q : Set ℝ := {y | ∃ θ : ℝ, y = Real.sin θ}

theorem intersection_P_Q : P ∩ Q = {-1, 0} :=
by
  sorry

end intersection_P_Q_l86_86254


namespace intersection_M_N_l86_86757

def M (x : ℝ) : Prop := x^2 ≥ x

def N (x : ℝ) (y : ℝ) : Prop := y = 3^x + 1

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | ∃ y : ℝ, N x y ∧ y > 1} = {x : ℝ | x > 1} :=
by {
  sorry
}

end intersection_M_N_l86_86757


namespace truncated_cone_volume_l86_86215

theorem truncated_cone_volume 
  (V_initial : ℝ)
  (r_ratio : ℝ)
  (V_final : ℝ)
  (r_ratio_eq : r_ratio = 1 / 2)
  (V_initial_eq : V_initial = 1) :
  V_final = 7 / 8 :=
  sorry

end truncated_cone_volume_l86_86215


namespace green_flowers_count_l86_86002

theorem green_flowers_count :
  ∀ (G R B Y T : ℕ),
    T = 96 →
    R = 3 * G →
    B = 48 →
    Y = 12 →
    G + R + B + Y = T →
    G = 9 :=
by
  intros G R B Y T
  intro hT
  intro hR
  intro hB
  intro hY
  intro hSum
  sorry

end green_flowers_count_l86_86002


namespace perfect_squares_between_50_and_200_l86_86464

theorem perfect_squares_between_50_and_200 : ∃ (n : ℕ), n = 7 := by
  let count := (range 15).filter (λ n, n^2 ≥ 50 ∧ n^2 ≤ 200)).length
  have h : count = 7 := by sorry
  use count
  exact h

end perfect_squares_between_50_and_200_l86_86464


namespace intersection_of_A_and_B_l86_86430

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {x | -Real.sqrt 3 < x ∧ x < Real.sqrt 3}

theorem intersection_of_A_and_B : (A ∩ B) = {x | -Real.sqrt 3 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l86_86430


namespace triangle_angles_and_sides_l86_86851

theorem triangle_angles_and_sides (A B C a b c : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hAngleSum : A + B + C = 180) (hle_ABC : A ≤ B ∧ B ≤ C) 
  (hle_abc : a ≤ b ∧ b ≤ c) (hArea : (a * b * (Real.sin C))/2 = 1) :
  (0 < A ∧ A ≤ 60) ∧ (0 < B ∧ B < 90) ∧ (60 ≤ C ∧ C < 180) ∧ 
  (0 < a) ∧ (sqrt 2 ≤ b) ∧ (frac 2 (Real/*c.root ⟨4, rfl⟩) > c) := 
by
  sorry

end triangle_angles_and_sides_l86_86851


namespace ratio_of_A_to_B_l86_86672

theorem ratio_of_A_to_B (A B C : ℝ) (h1 : A + B + C = 544) (h2 : B = (1/4) * C) (hA : A = 64) (hB : B = 96) (hC : C = 384) : A / B = 2 / 3 :=
by 
  sorry

end ratio_of_A_to_B_l86_86672


namespace lasagna_pieces_l86_86794

theorem lasagna_pieces (m a k r l : ℕ → ℝ)
  (hm : m 1 = 1)                -- Manny's consumption
  (ha : a 0 = 0)                -- Aaron's consumption
  (hk : ∀ n, k n = 2 * (m 1))   -- Kai's consumption
  (hr : ∀ n, r n = (1 / 2) * (m 1)) -- Raphael's consumption
  (hl : ∀ n, l n = 2 + (r n))   -- Lisa's consumption
  : m 1 + a 0 + k 1 + r 1 + l 1 = 6 :=
by
  -- Proof goes here
  sorry

end lasagna_pieces_l86_86794


namespace cycling_race_difference_l86_86478

-- Define the speeds and time
def s_Chloe : ℝ := 18
def s_David : ℝ := 15
def t : ℝ := 5

-- Define the distances based on the speeds and time
def d_Chloe : ℝ := s_Chloe * t
def d_David : ℝ := s_David * t
def distance_difference : ℝ := d_Chloe - d_David

-- The theorem to prove
theorem cycling_race_difference :
  distance_difference = 15 := by
  sorry

end cycling_race_difference_l86_86478


namespace expression_value_l86_86015

noncomputable def expression (x y z : ℝ) : ℝ :=
  (x^7 + y^7 + z^7) / (x * y * z * (x * y + x * z + y * z))

theorem expression_value
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod_nonzero : x * y + x * z + y * z ≠ 0) :
  expression x y z = -7 :=
by 
  sorry

end expression_value_l86_86015


namespace cube_inscribed_circumscribed_volume_ratio_l86_86644

theorem cube_inscribed_circumscribed_volume_ratio
  (S_1 S_2 V_1 V_2 : ℝ)
  (h : S_1 / S_2 = (1 / Real.sqrt 2) ^ 2) :
  V_1 / V_2 = (Real.sqrt 3 / 3) ^ 3 :=
sorry

end cube_inscribed_circumscribed_volume_ratio_l86_86644


namespace symmetric_to_y_axis_circle_l86_86321

open Real

-- Definition of the original circle's equation
def original_circle (x y : ℝ) : Prop := x^2 - 2 * x + y^2 = 3

-- Definition of the symmetric circle's equation with respect to the y-axis
def symmetric_circle (x y : ℝ) : Prop := x^2 + 2 * x + y^2 = 3

-- Theorem stating that the symmetric circle has the given equation
theorem symmetric_to_y_axis_circle (x y : ℝ) : 
  (symmetric_circle x y) ↔ (original_circle ((-x) - 2) y) :=
sorry

end symmetric_to_y_axis_circle_l86_86321


namespace number_of_perfect_squares_between_50_and_200_l86_86453

theorem number_of_perfect_squares_between_50_and_200 :
  ∃ n: ℕ, 50 < n^2 ∧ n^2 < 200 ∧ (14 - 8 + 1 = 7) := sorry

end number_of_perfect_squares_between_50_and_200_l86_86453


namespace functional_relationship_selling_price_l86_86541

open Real

-- Definitions used from conditions
def cost_price : ℝ := 20
def daily_sales_quantity (x : ℝ) : ℝ := -2 * x + 80

-- Functional relationship between daily sales profit W and selling price x
def daily_sales_profit (x : ℝ) : ℝ :=
  (x - cost_price) * daily_sales_quantity x

-- Part (1): Prove the functional relationship
theorem functional_relationship (x : ℝ) :
  daily_sales_profit x = -2 * x^2 + 120 * x - 1600 :=
by {
  sorry
}

-- Part (2): Prove the selling price should be $25 to achieve $150 profit with condition x ≤ 30
theorem selling_price (x : ℝ) :
  daily_sales_profit x = 150 ∧ x ≤ 30 → x = 25 :=
by {
  sorry
}

end functional_relationship_selling_price_l86_86541


namespace road_repair_equation_l86_86780

variable (x : ℝ) 

-- Original problem conditions
def total_road_length := 150
def extra_repair_per_day := 5
def days_ahead := 5

-- The proof problem to show that the schedule differential equals 5 days ahead
theorem road_repair_equation :
  (total_road_length / x) - (total_road_length / (x + extra_repair_per_day)) = days_ahead :=
sorry

end road_repair_equation_l86_86780


namespace find_polynomial_l86_86108

-- Define the polynomial function and the constant
variables {F : Type*} [Field F]

-- The main condition of the problem
def satisfies_condition (p : F → F) (c : F) :=
  ∀ x : F, p (p x) = x * p x + c * x^2

-- Prove the correct answers
theorem find_polynomial (p : F → F) (c : F) : 
  (c = 0 → ∀ x, p x = x) ∧ (c = -2 → ∀ x, p x = -x) :=
by
  sorry

end find_polynomial_l86_86108


namespace three_digit_numbers_left_l86_86625

def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def isABAForm (n : ℕ) : Prop :=
  ∃ A B : ℕ, A ≠ 0 ∧ A ≠ B ∧ n = 100 * A + 10 * B + A

def isAABOrBAAForm (n : ℕ) : Prop :=
  ∃ A B : ℕ, A ≠ 0 ∧ A ≠ B ∧ (n = 100 * A + 10 * A + B ∨ n = 100 * B + 10 * A + A)

def totalThreeDigitNumbers : ℕ := 900

def countABA : ℕ := 81

def countAABAndBAA : ℕ := 153

theorem three_digit_numbers_left : 
  (totalThreeDigitNumbers - countABA - countAABAndBAA) = 666 := 
by
   sorry

end three_digit_numbers_left_l86_86625


namespace expected_number_of_matches_variance_of_number_of_matches_l86_86847

-- Defining the conditions first, and then posing the proof statements
namespace MatchingPairs

open ProbabilityTheory

-- Probabilistic setup for indicator variables
variable (N : ℕ) (prob : ℝ := 1 / N)

-- Indicator variable Ik representing matches
@[simp] def I (k : ℕ) : ℝ := if k < N then prob else 0

-- Define the sum of expected matches S
@[simp] def S : ℝ := ∑ k in finset.range N, I N k

-- Statement: The expectation of the number of matching pairs is 1
theorem expected_number_of_matches : E[S] = 1 := sorry

-- Statement: The variance of the number of matching pairs is 1
theorem variance_of_number_of_matches : Var S = 1 := sorry

end MatchingPairs

end expected_number_of_matches_variance_of_number_of_matches_l86_86847


namespace cards_sum_l86_86519

theorem cards_sum (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
by
  sorry

end cards_sum_l86_86519


namespace smallest_prime_12_less_perfect_square_l86_86341

def is_prime (n : ℕ) : Prop := nat.prime n

def is_perfect_square_less_12 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 12

def smallest_prime (P : ℕ → Prop) : ℕ :=
  if h : ∃ n, P n then classical.some h else 0

def satisfies_conditions (n : ℕ) : Prop :=
  is_prime n ∧ is_perfect_square_less_12 n

theorem smallest_prime_12_less_perfect_square :
  smallest_prime satisfies_conditions = 13 :=
  sorry

end smallest_prime_12_less_perfect_square_l86_86341


namespace marble_problem_l86_86859

theorem marble_problem
  (h1 : ∀ x : ℕ, x > 0 → (x + 2) * ((220 / x) - 1) = 220) :
  ∃ x : ℕ, x > 0 ∧ (x + 2) * ((220 / ↑x) - 1) = 220 ∧ x = 20 :=
by
  sorry

end marble_problem_l86_86859


namespace log_cut_problem_l86_86328

theorem log_cut_problem (x y : ℕ) (h1 : x + y = 30) (h2 : 3 * x + 4 * y = 100) :
  2 * x + 3 * y = 70 := by
  sorry

end log_cut_problem_l86_86328


namespace sasha_lives_on_seventh_floor_l86_86312

theorem sasha_lives_on_seventh_floor (N : ℕ) (x : ℕ) 
(h1 : x = (1/3 : ℝ) * N) 
(h2 : N - ((1/3 : ℝ) * N + 1) = (1/2 : ℝ) * N) :
  N + 1 = 7 := 
sorry

end sasha_lives_on_seventh_floor_l86_86312


namespace count_perfect_squares_50_to_200_l86_86459

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem count_perfect_squares_50_to_200 :
  {n : ℕ | 50 < n ∧ n < 200 ∧ is_perfect_square n}.to_finset.card = 7 :=
by
  sorry

end count_perfect_squares_50_to_200_l86_86459


namespace base3_sum_l86_86870

theorem base3_sum : 
  (1 * 3^0 - 2 * 3^1 - 2 * 3^0 + 2 * 3^2 + 1 * 3^1 - 1 * 3^0 - 1 * 3^3) = (2 * 3^2 + 1 * 3^1 + 0 * 3^0) := 
by 
  sorry

end base3_sum_l86_86870


namespace B_subscribed_fraction_correct_l86_86228

-- Define the total capital and the shares of A, C
variables (X : ℝ) (profit : ℝ) (A_share : ℝ) (C_share : ℝ)

-- Define the conditions as given in the problem
def A_capital_share := 1 / 3
def C_capital_share := 1 / 5
def total_profit := 2430
def A_profit_share := 810

-- Define the calculation of B's share
def B_capital_share := 1 - (A_capital_share + C_capital_share)

-- Define the expected correct answer for B's share
def expected_B_share := 7 / 15

-- Theorem statement
theorem B_subscribed_fraction_correct :
  B_capital_share = expected_B_share :=
by
  sorry

end B_subscribed_fraction_correct_l86_86228


namespace product_of_roots_l86_86728

open Real

theorem product_of_roots : (sqrt (Real.exp (1 / 4 * log (16)))) * (sqrt (Real.exp (1 / 6 * log (64)))) = 4 :=
by
  -- sorry is used to bypass the actual proof implementation
  sorry

end product_of_roots_l86_86728


namespace speed_ratio_l86_86864

theorem speed_ratio (v1 v2 : ℝ) (t1 t2 : ℝ) (dist_before dist_after : ℝ) (total_dist : ℝ)
  (h1 : dist_before + dist_after = total_dist)
  (h2 : dist_before = 20)
  (h3 : dist_after = 20)
  (h4 : t2 = t1 + 11)
  (h5 : t2 = 22)
  (h6 : t1 = dist_before / v1)
  (h7 : t2 = dist_after / v2) :
  v1 / v2 = 2 := 
sorry

end speed_ratio_l86_86864


namespace initial_amount_solution_l86_86415

noncomputable def initialAmount (P : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then P else (1 + 1/8) * initialAmount P (n - 1)

theorem initial_amount_solution (P : ℝ) (h₁ : initialAmount P 2 = 2025) : P = 1600 :=
  sorry

end initial_amount_solution_l86_86415


namespace vartan_recreation_percent_l86_86786

noncomputable def percent_recreation_week (last_week_wages current_week_wages current_week_recreation last_week_recreation : ℝ) : ℝ :=
  (current_week_recreation / current_week_wages) * 100

theorem vartan_recreation_percent 
  (W : ℝ) 
  (h1 : last_week_wages = W)  
  (h2 : last_week_recreation = 0.15 * W)
  (h3 : current_week_wages = 0.90 * W)
  (h4 : current_week_recreation = 1.80 * last_week_recreation) :
  percent_recreation_week last_week_wages current_week_wages current_week_recreation last_week_recreation = 30 :=
by
  sorry

end vartan_recreation_percent_l86_86786


namespace extreme_points_exactly_one_zero_in_positive_interval_l86_86916

noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - (1 / 3) * a * x^3

theorem extreme_points (a : ℝ) (h : a > Real.exp 1) :
  ∃ (x1 x2 x3 : ℝ), (0 < x1) ∧ (x1 < x2) ∧ (x2 < x3) ∧ (deriv (f x) = 0) := sorry

theorem exactly_one_zero_in_positive_interval (a : ℝ) (h : a > Real.exp 1) :
  ∃! x : ℝ, (0 < x) ∧ (f x a = 0) := sorry

end extreme_points_exactly_one_zero_in_positive_interval_l86_86916


namespace flower_bouquet_violets_percentage_l86_86858

theorem flower_bouquet_violets_percentage
  (total_flowers yellow_flowers purple_flowers : ℕ)
  (yellow_daisies yellow_tulips purple_violets : ℕ)
  (h_yellow_flowers : yellow_flowers = (total_flowers / 2))
  (h_purple_flowers : purple_flowers = (total_flowers / 2))
  (h_yellow_daisies : yellow_daisies = (yellow_flowers / 5))
  (h_yellow_tulips : yellow_tulips = yellow_flowers - yellow_daisies)
  (h_purple_violets : purple_violets = (purple_flowers / 2)) :
  ((purple_violets : ℚ) / total_flowers) * 100 = 25 :=
by
  sorry

end flower_bouquet_violets_percentage_l86_86858


namespace solution_set_of_inequality_l86_86119

noncomputable def f (x : ℝ) : ℝ := sorry

lemma odd_function (x : ℝ) : f (-x) = -f x := sorry
lemma functional_value_at_1 : f 1 = Real.exp 1 := sorry
lemma monotonic_condition (x : ℝ) (hx : 0 ≤ x) : (x-1) * f x < x * (deriv f x) := sorry

theorem solution_set_of_inequality : { x : ℝ | x * f x - Real.exp (|x|) > 0 } =
  { x : ℝ | x < -1 } ∪ { x : ℝ | 1 < x } :=
by
  sorry

end solution_set_of_inequality_l86_86119


namespace distances_perimeter_inequality_l86_86668

variable {Point Polygon : Type}

-- Definitions for the conditions
variables (O : Point) (M : Polygon)
variable (ρ : ℝ) -- perimeter of M
variable (d : ℝ) -- sum of distances to each vertex of M from O
variable (h : ℝ) -- sum of distances to each side of M from O

-- The theorem statement
theorem distances_perimeter_inequality :
  d^2 - h^2 ≥ ρ^2 / 4 :=
by
  sorry

end distances_perimeter_inequality_l86_86668


namespace quadrant_of_angle_l86_86912

theorem quadrant_of_angle (θ : ℝ) (h1 : Real.cos θ = -3 / 5) (h2 : Real.tan θ = 4 / 3) :
    θ ∈ Set.Icc (π : ℝ) (3 * π / 2) := sorry

end quadrant_of_angle_l86_86912


namespace a_is_perfect_square_l86_86787

theorem a_is_perfect_square {a : ℕ} (h : ∀ n : ℕ, ∃ d : ℕ, d ≠ 1 ∧ d % n = 1 ∧ d ∣ n ^ 2 * a - 1) : ∃ k : ℕ, a = k ^ 2 :=
by
  sorry

end a_is_perfect_square_l86_86787


namespace water_force_on_dam_l86_86593

-- Given conditions
def density : Real := 1000  -- kg/m^3
def gravity : Real := 10    -- m/s^2
def a : Real := 5.7         -- m
def b : Real := 9.0         -- m
def h : Real := 4.0         -- m

-- Prove that the force is 544000 N under the given conditions
theorem water_force_on_dam : ∃ (F : Real), F = 544000 :=
by
  sorry  -- proof goes here

end water_force_on_dam_l86_86593


namespace calendars_ordered_l86_86222

theorem calendars_ordered 
  (C D : ℝ) 
  (h1 : C + D = 500) 
  (h2 : 0.75 * C + 0.50 * D = 300) 
  : C = 200 :=
by
  sorry

end calendars_ordered_l86_86222


namespace subscription_total_eq_14036_l86_86571

noncomputable def total_subscription (x : ℕ) : ℕ :=
  3 * x + 14000

theorem subscription_total_eq_14036 (c : ℕ) (profit_b : ℕ) (total_profit : ℕ) 
  (h1 : profit_b = 10200)
  (h2 : total_profit = 30000) 
  (h3 : (profit_b : ℝ) / (total_profit : ℝ) = (c + 5000 : ℝ) / (total_subscription c : ℝ)) :
  total_subscription c = 14036 :=
by
  sorry

end subscription_total_eq_14036_l86_86571


namespace part_I_part_II_l86_86917

noncomputable def f (a x : ℝ) : ℝ := |a * x - 1| + |x + 2|

theorem part_I (h₁ : ∀ x : ℝ, f 1 x ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2) : True :=
by sorry

theorem part_II (h₂ : ∃ a : ℝ, a > 0 ∧ (∀ x, f a x ≥ 2) ∧ (∀ b : ℝ, b > 0 ∧ (∀ x, f b x ≥ 2) → a ≤ b) ) : True :=
by sorry

end part_I_part_II_l86_86917


namespace ratio_of_populations_l86_86797

theorem ratio_of_populations (ne_pop : ℕ) (combined_pop : ℕ) (ny_pop : ℕ) (h1 : ne_pop = 2100000) 
                            (h2 : combined_pop = 3500000) (h3 : ny_pop = combined_pop - ne_pop) :
                            (ny_pop * 3 = ne_pop * 2) :=
by
  sorry

end ratio_of_populations_l86_86797


namespace johns_shell_arrangements_l86_86785

-- Define the total number of arrangements without considering symmetries
def totalArrangements := Nat.factorial 12

-- Define the number of equivalent arrangements due to symmetries
def symmetries := 6 * 2

-- Define the number of distinct arrangements
def distinctArrangements : Nat := totalArrangements / symmetries

-- State the theorem
theorem johns_shell_arrangements : distinctArrangements = 479001600 :=
by
  sorry

end johns_shell_arrangements_l86_86785


namespace find_number_l86_86542

theorem find_number (x : ℝ) :
  (10 + 30 + 50) / 3 = 30 →
  ((x + 40 + 6) / 3 = (10 + 30 + 50) / 3 - 8) →
  x = 20 :=
by
  intros h_avg1 h_avg2
  sorry

end find_number_l86_86542


namespace Debby_drinks_five_bottles_per_day_l86_86093

theorem Debby_drinks_five_bottles_per_day (total_bottles : ℕ) (days : ℕ) (h1 : total_bottles = 355) (h2 : days = 71) : (total_bottles / days) = 5 :=
by 
  sorry

end Debby_drinks_five_bottles_per_day_l86_86093


namespace quadratic_eq_has_double_root_l86_86253

theorem quadratic_eq_has_double_root (m : ℝ) :
  (m - 2) ^ 2 - 4 * (m + 1) = 0 ↔ m = 0 ∨ m = 8 := 
by
  sorry

end quadratic_eq_has_double_root_l86_86253


namespace village_transportation_problem_l86_86081

noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

variable (total odd : ℕ) (a : ℕ)

theorem village_transportation_problem 
  (h_total : total = 15)
  (h_odd : odd = 7)
  (h_selected : 10 = 10)
  (h_eq : (comb 7 4) * (comb 8 6) / (comb 15 10) = (comb 7 (10 - a)) * (comb 8 a) / (comb 15 10)) :
  a = 6 := 
sorry

end village_transportation_problem_l86_86081


namespace anna_cupcakes_remaining_l86_86231

theorem anna_cupcakes_remaining :
  let total_cupcakes := 60
  let cupcakes_given_away := (4 / 5 : ℝ) * total_cupcakes
  let cupcakes_after_giving := total_cupcakes - cupcakes_given_away
  let cupcakes_eaten := 3
  let cupcakes_left := cupcakes_after_giving - cupcakes_eaten
  cupcakes_left = 9 :=
by
  sorry

end anna_cupcakes_remaining_l86_86231


namespace bisections_needed_l86_86472

theorem bisections_needed (ε : ℝ) (ε_pos : ε = 0.01) (h : 0 < ε) : 
  ∃ n : ℕ, n ≤ 7 ∧ 1 / (2^n) < ε :=
by
  sorry

end bisections_needed_l86_86472


namespace range_of_a_l86_86750

structure PropositionP (a : ℝ) : Prop :=
  (h : 2 * a + 1 > 5)

structure PropositionQ (a : ℝ) : Prop :=
  (h : -1 ≤ a ∧ a ≤ 3)

theorem range_of_a (a : ℝ) (hp : PropositionP a ∨ PropositionQ a) (hq : ¬(PropositionP a ∧ PropositionQ a)) :
  (-1 ≤ a ∧ a ≤ 2) ∨ (a > 3) :=
sorry

end range_of_a_l86_86750


namespace a5_gt_b5_l86_86941

variables {a_n b_n : ℕ → ℝ}
variables {a1 b1 a3 b3 : ℝ}
variables {q : ℝ} {d : ℝ}

/-- Given conditions -/
axiom h1 : a1 = b1
axiom h2 : a1 > 0
axiom h3 : a3 = b3
axiom h4 : a3 = a1 * q^2
axiom h5 : b3 = a1 + 2 * d
axiom h6 : a1 ≠ a3

/-- Prove that a_5 is greater than b_5 -/
theorem a5_gt_b5 : a1 * q^4 > a1 + 4 * d :=
by sorry

end a5_gt_b5_l86_86941


namespace number_of_puppies_sold_l86_86591

variables (P : ℕ) (p_0 : ℕ) (k_0 : ℕ) (r : ℕ) (k_s : ℕ)

theorem number_of_puppies_sold 
  (h1 : p_0 = 7) 
  (h2 : k_0 = 6) 
  (h3 : r = 8) 
  (h4 : k_s = 3) : 
  P = p_0 - (r - (k_0 - k_s)) :=
by sorry

end number_of_puppies_sold_l86_86591


namespace min_value_frac_l86_86137

theorem min_value_frac (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ (x : ℝ), x = 16 ∧ (forall y, y = 9 / a + 1 / b → x ≤ y) :=
sorry

end min_value_frac_l86_86137


namespace total_revenue_correct_l86_86379

-- Define the conditions
def charge_per_slice : ℕ := 5
def slices_per_pie : ℕ := 4
def pies_sold : ℕ := 9

-- Prove the question: total revenue
theorem total_revenue_correct : charge_per_slice * slices_per_pie * pies_sold = 180 :=
by
  sorry

end total_revenue_correct_l86_86379


namespace geometric_seq_second_term_l86_86681

-- Definitions
def fifth_term : ℕ → ℝ := λ n, if n = 5 then 48 else 0
def sixth_term : ℕ → ℝ := λ n, if n = 6 then 72 else 0

-- Theorem Statement
theorem geometric_seq_second_term :
  let r := sixth_term 6 / fifth_term 5,
      a := (fifth_term 5) / (r ^ 4),
      a2 := a * r in
  sixth_term 6 = 72 ∧ fifth_term 5 = 48 →
  a2 = 384 / 27 := 
begin
  sorry
end

end geometric_seq_second_term_l86_86681


namespace age_ratio_albert_mary_l86_86587

variable (A M B : ℕ) 

theorem age_ratio_albert_mary
    (h1 : A = 4 * B)
    (h2 : M = A - 10)
    (h3 : B = 5) :
    A = 2 * M :=
by
    sorry

end age_ratio_albert_mary_l86_86587


namespace monthly_revenue_l86_86164

variable (R : ℝ) -- The monthly revenue

-- Conditions
def after_taxes (R : ℝ) : ℝ := R * 0.90
def after_marketing (R : ℝ) : ℝ := (after_taxes R) * 0.95
def after_operational_costs (R : ℝ) : ℝ := (after_marketing R) * 0.80
def total_employee_wages (R : ℝ) : ℝ := (after_operational_costs R) * 0.15

-- Number of employees and their wages
def number_of_employees : ℝ := 10
def wage_per_employee : ℝ := 4104
def total_wages : ℝ := number_of_employees * wage_per_employee

-- Proof problem
theorem monthly_revenue : R = 400000 ↔ total_employee_wages R = total_wages := by
  sorry

end monthly_revenue_l86_86164


namespace smallest_positive_prime_12_less_than_square_is_13_l86_86350

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_prime_12_less_than_square (k : ℕ) : Prop :=
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = k * k - 12

theorem smallest_positive_prime_12_less_than_square_is_13 :
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = 5 * 5 - 12 ∧ (∀ n : ℕ, is_prime n ∧ n > 0 ∧ n = k * k - 12 → n ≥ m) :=
begin
  use 13,
  sorry
end

end smallest_positive_prime_12_less_than_square_is_13_l86_86350


namespace smallest_prime_12_less_than_perfect_square_l86_86352

theorem smallest_prime_12_less_than_perfect_square : ∃ p : ℕ, prime p ∧ ∃ n : ℤ, p = n^2 - 12 ∧ p = 13 := 
by
  sorry

end smallest_prime_12_less_than_perfect_square_l86_86352


namespace find_constant_k_l86_86064

theorem find_constant_k (k : ℤ) :
    (∀ x : ℝ, -x^2 - (k + 7) * x - 8 = - (x - 2) * (x - 4)) → k = -13 :=
by 
    intros h
    sorry

end find_constant_k_l86_86064


namespace max_r_value_l86_86025

theorem max_r_value (r : ℕ) (hr : r ≥ 2)
  (m n : Fin r → ℤ)
  (h : ∀ i j : Fin r, i < j → |m i * n j - m j * n i| = 1) :
  r ≤ 3 := 
sorry

end max_r_value_l86_86025


namespace distinct_points_l86_86740

namespace IntersectionPoints

open Set

theorem distinct_points :
  ∃ (S : Set (ℝ × ℝ)), 
    (∀ x y : ℝ, ((x + 2*y - 8) * (3*x - y + 6) = 0) → (x, y) ∈ S) ∧
    (∀ x y : ℝ, ((2*x - 3*y + 2) * (4*x + y - 16) = 0) → (x, y) ∈ S) ∧
    S.card = 4 := by
  sorry

end IntersectionPoints

end distinct_points_l86_86740


namespace transform_cubic_eq_trig_solutions_l86_86833

section

variable {p q x t r : ℝ}

/-- Statement (a): Show that for p < 0, the equation x³ + px + q = 0 can be transformed by substituting x = 2 * sqrt(-p / 3) * t into the equation 4 * t³ - 3 * t - r = 0 in the variable t -/
theorem transform_cubic_eq (hp : p < 0) (ht : t = 2 * sqrt(-p / 3) * x) :
  x^3 + p * x + q = 0 ↔ 4 * t^3 - 3 * t - r = 0 :=
sorry

/-- Statement (b): Prove that for 4p³ + 27q² ≤ 0, the solutions to the equation 4t³ - 3t - r = 0 will be t₁ = cos(φ / 3), t₂ = cos((φ + 2 * π) / 3), t₃ = cos((φ + 4 * π) / 3) where φ = arccos(r) -/
theorem trig_solutions (hineq : 4 * p^3 + 27 * q^2 ≤ 0) (φ : ℝ) (hφ : φ = arccos r) :
  (4 * (cos(φ / 3))^3 - 3 * cos(φ / 3) - r = 0) ∧
  (4 * (cos((φ + 2 * real.pi) / 3))^3 - 3 * cos((φ + 2 * real.pi) / 3) - r = 0) ∧
  (4 * (cos((φ + 4 * real.pi) / 3))^3 - 3 * cos((φ + 4 * real.pi) / 3) - r = 0) :=
sorry

end

end transform_cubic_eq_trig_solutions_l86_86833


namespace solution_l86_86426

theorem solution {a : ℕ → ℝ} 
  (h : a 1 = 1)
  (h2 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    a n - 4 * a (if n = 100 then 1 else n + 1) + 3 * a (if n = 99 then 1 else if n = 100 then 2 else n + 2) ≥ 0) :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → a n = 1 :=
by
  sorry

end solution_l86_86426


namespace sum_of_intercepts_modulo_13_l86_86067

theorem sum_of_intercepts_modulo_13 :
  ∃ (x0 y0 : ℤ), 0 ≤ x0 ∧ x0 < 13 ∧ 0 ≤ y0 ∧ y0 < 13 ∧
    (4 * x0 ≡ 1 [ZMOD 13]) ∧ (3 * y0 ≡ 12 [ZMOD 13]) ∧ (x0 + y0 = 14) := 
sorry

end sum_of_intercepts_modulo_13_l86_86067


namespace reciprocal_of_neg_2023_l86_86991

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86991


namespace certain_number_sum_421_l86_86856

theorem certain_number_sum_421 :
  ∃ n, (∃ k, n = 423 * k) ∧ k = 2 →
  n + 421 = 1267 :=
by
  sorry

end certain_number_sum_421_l86_86856


namespace fraction_checked_by_worker_y_l86_86211

variables (P X Y : ℕ)
variables (defective_rate_x defective_rate_y total_defective_rate : ℚ)
variables (h1 : X + Y = P)
variables (h2 : defective_rate_x = 0.005)
variables (h3 : defective_rate_y = 0.008)
variables (h4 : total_defective_rate = 0.007)
variables (defective_x : ℚ := 0.005 * X)
variables (defective_y : ℚ := 0.008 * Y)
variables (total_defective_products : ℚ := 0.007 * P)
variables (h5 : defective_x + defective_y = total_defective_products)

theorem fraction_checked_by_worker_y : Y / P = 2 / 3 :=
by sorry

end fraction_checked_by_worker_y_l86_86211


namespace options_not_equal_l86_86828

theorem options_not_equal (a b c d e : ℚ)
  (ha : a = 14 / 10)
  (hb : b = 1 + 2 / 5)
  (hc : c = 1 + 7 / 25)
  (hd : d = 1 + 2 / 10)
  (he : e = 1 + 14 / 70) :
  a = 7 / 5 ∧ b = 7 / 5 ∧ c ≠ 7 / 5 ∧ d ≠ 7 / 5 ∧ e ≠ 7 / 5 :=
by sorry

end options_not_equal_l86_86828


namespace motorcycle_licenses_count_l86_86583

theorem motorcycle_licenses_count : (3 * (10 ^ 6) = 3000000) :=
by
  sorry -- Proof would go here.

end motorcycle_licenses_count_l86_86583


namespace count_four_digit_integers_with_conditions_l86_86445

def is_four_digit_integer (n : Nat) : Prop := 1000 ≤ n ∧ n < 10000

def thousands_digit_is_seven (n : Nat) : Prop := 
  (n / 1000) % 10 = 7

def hundreds_digit_is_odd (n : Nat) : Prop := 
  let hd := (n / 100) % 10
  hd % 2 = 1

theorem count_four_digit_integers_with_conditions : 
  (Nat.card {n : Nat // is_four_digit_integer n ∧ thousands_digit_is_seven n ∧ hundreds_digit_is_odd n}) = 500 :=
by
  sorry

end count_four_digit_integers_with_conditions_l86_86445


namespace sphere_surface_area_is_36pi_l86_86280

-- Define the edge length of the cube
def edge_length : ℝ := 2 * Real.sqrt 3

-- Define the diagonal of the cube
def cube_diagonal : ℝ := edge_length * Real.sqrt 3

-- Define the radius of the sphere circumscribing the cube
def sphere_radius : ℝ := cube_diagonal / 2

-- Define the surface area of the sphere
noncomputable def sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius^2

-- Prove that the surface area of the sphere is 36π
theorem sphere_surface_area_is_36pi : sphere_surface_area = 36 * Real.pi := by
  sorry

end sphere_surface_area_is_36pi_l86_86280


namespace sin_225_eq_neg_sqrt_two_div_two_l86_86881

theorem sin_225_eq_neg_sqrt_two_div_two :
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt_two_div_two_l86_86881


namespace increasing_function_l86_86040

theorem increasing_function (k b : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k + 1) * x1 + b < (2 * k + 1) * x2 + b) ↔ k > -1/2 := 
by
  sorry

end increasing_function_l86_86040


namespace granddaughter_fraction_l86_86401

noncomputable def betty_age : ℕ := 60
def fraction_younger (p : ℕ) : ℕ := (p * 40) / 100
noncomputable def daughter_age : ℕ := betty_age - fraction_younger betty_age
def granddaughter_age : ℕ := 12
def fraction (a b : ℕ) : ℚ := a / b

theorem granddaughter_fraction :
  fraction granddaughter_age daughter_age = 1 / 3 := 
by
  sorry

end granddaughter_fraction_l86_86401


namespace smallest_positive_n_l86_86110

open Real

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos θ, -sin θ], ![sin θ, cos θ]]

theorem smallest_positive_n (n : ℕ) :
    (rotation_matrix (150 * π / 180)) ^ n = 1 :=
    n = 12 := sorry

end smallest_positive_n_l86_86110


namespace domain_of_composed_function_l86_86039

theorem domain_of_composed_function {f : ℝ → ℝ} (h : ∀ x, -1 < x ∧ x < 1 → f x ∈ Set.Ioo (-1:ℝ) 1) :
  ∀ x, 0 < x ∧ x < 1 → f (2*x-1) ∈ Set.Ioo (-1:ℝ) 1 := by
  sorry

end domain_of_composed_function_l86_86039


namespace investment_plans_count_l86_86716

theorem investment_plans_count :
  ∃ (projects cities : ℕ) (no_more_than_two : ℕ → ℕ → Prop),
    no_more_than_two projects cities →
    projects = 3 →
    cities = 5 →
    (projects ≤ 2 ∧ projects > 0) →
    ( (3.choose 2) * 5 * 4 + 5.choose 3 ) = 120 :=
by
  sorry

end investment_plans_count_l86_86716


namespace girls_percentage_l86_86937

theorem girls_percentage (total_students girls boys : ℕ) 
    (total_eq : total_students = 42)
    (ratio : 3 * girls = 4 * boys)
    (total_students_eq : total_students = girls + boys) : 
    (girls * 100 / total_students : ℚ) = 57.14 := 
by 
  sorry

end girls_percentage_l86_86937


namespace discount_percentage_l86_86289

theorem discount_percentage (coach_cost sectional_cost other_cost paid : ℕ) 
  (h1 : coach_cost = 2500) 
  (h2 : sectional_cost = 3500) 
  (h3 : other_cost = 2000) 
  (h4 : paid = 7200) : 
  ((coach_cost + sectional_cost + other_cost - paid) * 100) / (coach_cost + sectional_cost + other_cost) = 10 :=
by
  sorry

end discount_percentage_l86_86289


namespace geometric_seq_second_term_l86_86683

-- Definitions
def fifth_term : ℕ → ℝ := λ n, if n = 5 then 48 else 0
def sixth_term : ℕ → ℝ := λ n, if n = 6 then 72 else 0

-- Theorem Statement
theorem geometric_seq_second_term :
  let r := sixth_term 6 / fifth_term 5,
      a := (fifth_term 5) / (r ^ 4),
      a2 := a * r in
  sixth_term 6 = 72 ∧ fifth_term 5 = 48 →
  a2 = 384 / 27 := 
begin
  sorry
end

end geometric_seq_second_term_l86_86683


namespace cost_backpack_is_100_l86_86491

-- Definitions based on the conditions
def cost_wallet : ℕ := 50
def cost_sneakers_per_pair : ℕ := 100
def num_sneakers_pairs : ℕ := 2
def cost_jeans_per_pair : ℕ := 50
def num_jeans_pairs : ℕ := 2
def total_spent : ℕ := 450

-- The problem statement
theorem cost_backpack_is_100 (x : ℕ) 
  (leonard_total : ℕ := cost_wallet + num_sneakers_pairs * cost_sneakers_per_pair) 
  (michael_non_backpack_total : ℕ := num_jeans_pairs * cost_jeans_per_pair) :
  total_spent = leonard_total + michael_non_backpack_total + x → x = 100 := 
by
  unfold cost_wallet cost_sneakers_per_pair num_sneakers_pairs total_spent cost_jeans_per_pair num_jeans_pairs
  intro h
  sorry

end cost_backpack_is_100_l86_86491


namespace inequality_proof_equality_condition_l86_86497

theorem inequality_proof (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) :=
sorry

theorem equality_condition (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b)) → a = b :=
sorry

end inequality_proof_equality_condition_l86_86497


namespace reciprocal_of_neg_2023_l86_86977

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l86_86977


namespace solution_of_xyz_l86_86630

theorem solution_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) : x + y + z = 48 := 
sorry

end solution_of_xyz_l86_86630


namespace sum_of_numbers_on_cards_l86_86505

-- Define the natural numbers condition
variables {a b c d e f g h : ℕ}

-- The theorem statement
theorem sum_of_numbers_on_cards (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_numbers_on_cards_l86_86505


namespace add_gold_coins_l86_86481

open Nat

theorem add_gold_coins (G S X : ℕ) 
  (h₁ : G = S / 3) 
  (h₂ : (G + X) / S = 1 / 2) 
  (h₃ : G + X + S = 135) : 
  X = 15 := 
sorry

end add_gold_coins_l86_86481


namespace sum_of_eight_numbers_l86_86527

-- Definitions used in the conditions
variables {a b c d e f g h : ℕ}

-- Given condition
axiom product_condition : (a + b) * (c + d) * (e + f) * (g + h) = 330

-- Define individual sums
def ab_sum := a + b
def cd_sum := c + d
def ef_sum := e + f
def gh_sum := g + h

-- Define the total sum of the eight numbers on the cards
def total_sum := ab_sum + cd_sum + ef_sum + gh_sum

-- The theorem to prove
theorem sum_of_eight_numbers : total_sum = 21 := by
  have ab_sum_eq : ab_sum = 2 := sorry
  have cd_sum_eq : cd_sum = 3 := sorry
  have ef_sum_eq : ef_sum = 5 := sorry
  have gh_sum_eq : gh_sum = 11 := sorry
  rw [ab_sum_eq, cd_sum_eq, ef_sum_eq, gh_sum_eq]
  norm_num

end sum_of_eight_numbers_l86_86527


namespace coin_arrangements_l86_86533

/-- We define the conditions for Robert's coin arrangement problem. -/
def gold_coins := 5
def silver_coins := 5
def total_coins := gold_coins + silver_coins

/-- We define the number of ways to arrange 5 gold coins and 5 silver coins in 10 positions,
using the binomial coefficient. -/
def arrangements_colors : ℕ := Nat.choose total_coins gold_coins

/-- We define the number of possible configurations for the orientation of the coins
such that no two adjacent coins are face to face. -/
def arrangements_orientation : ℕ := 11

/-- The total number of distinguishable arrangements of the coins. -/
def total_arrangements : ℕ := arrangements_colors * arrangements_orientation

theorem coin_arrangements : total_arrangements = 2772 := by
  -- The proof is omitted.
  sorry

end coin_arrangements_l86_86533


namespace average_age_of_adults_l86_86806

theorem average_age_of_adults 
  (total_members : ℕ)
  (avg_age_total : ℕ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (num_adults : ℕ)
  (avg_age_girls : ℕ)
  (avg_age_boys : ℕ)
  (total_sum_ages : ℕ := total_members * avg_age_total)
  (sum_ages_girls : ℕ := num_girls * avg_age_girls)
  (sum_ages_boys : ℕ := num_boys * avg_age_boys)
  (sum_ages_adults : ℕ := total_sum_ages - sum_ages_girls - sum_ages_boys)
  : (num_adults = 10) → (avg_age_total = 20) → (num_girls = 30) → (avg_age_girls = 18) → (num_boys = 20) → (avg_age_boys = 22) → (total_sum_ages = 1200) → (sum_ages_girls = 540) → (sum_ages_boys = 440) → (sum_ages_adults = 220) → (sum_ages_adults / num_adults = 22) :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end average_age_of_adults_l86_86806


namespace max_value_of_quadratic_l86_86808

theorem max_value_of_quadratic :
  ∃ y : ℚ, ∀ x : ℚ, -x^2 - 3 * x + 4 ≤ y :=
sorry

end max_value_of_quadratic_l86_86808


namespace shiela_bottles_l86_86537

theorem shiela_bottles (num_stars : ℕ) (stars_per_bottle : ℕ) (num_bottles : ℕ) 
  (h1 : num_stars = 45) (h2 : stars_per_bottle = 5) : num_bottles = 9 :=
sorry

end shiela_bottles_l86_86537


namespace find_second_term_l86_86688

-- Define the terms and common ratio in the geometric sequence
def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

-- Given the fifth and sixth terms
variables (a r : ℚ)
axiom fifth_term : geometric_sequence a r 5 = 48
axiom sixth_term : geometric_sequence a r 6 = 72

-- Prove that the second term is 128/9
theorem find_second_term : geometric_sequence a r 2 = 128 / 9 :=
sorry

end find_second_term_l86_86688


namespace num_squares_figure8_perimeter_figure12_perimeter_figureC_eq_38_ratio_perimeter_figure29_figureD_l86_86372

-- Condition: Figure 1 is formed by 3 identical squares of side length 1 cm.
def squares_in_figure1 : ℕ := 3

-- Condition: Perimeter of Figure 1 is 8 cm.
def perimeter_figure1 : ℝ := 8

-- Condition: Each subsequent figure adds 2 squares.
def squares_in_figure (n : ℕ) : ℕ :=
  squares_in_figure1 + 2 * (n - 1)

-- Condition: Each subsequent figure increases perimeter by 2 cm.
def perimeter_figure (n : ℕ) : ℝ :=
  perimeter_figure1 + 2 * (n - 1)

-- Proof problem (a): Prove that the number of squares in Figure 8 is 17.
theorem num_squares_figure8 :
  squares_in_figure 8 = 17 :=
sorry

-- Proof problem (b): Prove that the perimeter of Figure 12 is 30 cm.
theorem perimeter_figure12 :
  perimeter_figure 12 = 30 :=
sorry

-- Proof problem (c): Prove that the positive integer C for which the perimeter of Figure C is 38 cm is 16.
theorem perimeter_figureC_eq_38 :
  ∃ C : ℕ, perimeter_figure C = 38 :=
sorry

-- Proof problem (d): Prove that the positive integer D for which the ratio of the perimeter of Figure 29 to the perimeter of Figure D is 4/11 is 85.
theorem ratio_perimeter_figure29_figureD :
  ∃ D : ℕ, (perimeter_figure 29 / perimeter_figure D) = (4 / 11) :=
sorry

end num_squares_figure8_perimeter_figure12_perimeter_figureC_eq_38_ratio_perimeter_figure29_figureD_l86_86372


namespace fraction_zero_l86_86549

theorem fraction_zero (x : ℝ) (h₁ : x - 3 = 0) (h₂ : x ≠ 0) : (x - 3) / (4 * x) = 0 :=
by
  sorry

end fraction_zero_l86_86549


namespace mindy_emails_l86_86954

theorem mindy_emails (P E : ℕ) 
    (h1 : E = 9 * P - 7)
    (h2 : E + P = 93) :
    E = 83 := 
    sorry

end mindy_emails_l86_86954


namespace quadratic_real_roots_l86_86424

theorem quadratic_real_roots (a: ℝ) :
  ∀ x: ℝ, (a-6) * x^2 - 8 * x + 9 = 0 ↔ (a ≤ 70/9 ∧ a ≠ 6) :=
  sorry

end quadratic_real_roots_l86_86424


namespace smallest_prime_12_less_than_perfect_square_l86_86346

theorem smallest_prime_12_less_than_perfect_square : ∃ n : ℕ, prime n ∧ ∃ k : ℕ, k^2 - n = 12 ∧ n = 13 :=
by {
  use 13,
  split,
  { exact prime_def.2 ⟨nat.prime_def_lt.mp nat.prime_two⟩, -- Proof that 13 is prime, simplified
  { use 5,
    split,
    { calc
      5 ^ 2 - 13
         = 25 - 13 : by rfl
    ... = 12 : by rfl,
    { refl,
    }
  }
end

end smallest_prime_12_less_than_perfect_square_l86_86346


namespace cone_to_cylinder_ratio_l86_86427

theorem cone_to_cylinder_ratio (r : ℝ) (h_cyl : ℝ) (h_cone : ℝ) 
  (V_cyl : ℝ) (V_cone : ℝ) 
  (h_cyl_eq : h_cyl = 18)
  (r_eq : r = 5)
  (h_cone_eq : h_cone = h_cyl / 3)
  (volume_cyl_eq : V_cyl = π * r^2 * h_cyl)
  (volume_cone_eq : V_cone = 1/3 * π * r^2 * h_cone) :
  V_cone / V_cyl = 1 / 9 := by
  sorry

end cone_to_cylinder_ratio_l86_86427


namespace smallest_positive_prime_12_less_than_square_is_13_l86_86351

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_prime_12_less_than_square (k : ℕ) : Prop :=
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = k * k - 12

theorem smallest_positive_prime_12_less_than_square_is_13 :
  ∃ m : ℕ, is_prime m ∧ m > 0 ∧ m = 5 * 5 - 12 ∧ (∀ n : ℕ, is_prime n ∧ n > 0 ∧ n = k * k - 12 → n ≥ m) :=
begin
  use 13,
  sorry
end

end smallest_positive_prime_12_less_than_square_is_13_l86_86351


namespace find_all_pos_integers_l86_86603

theorem find_all_pos_integers (M : ℕ) (h1 : M > 0) (h2 : M < 10) :
  (5 ∣ (1989^M + M^1989)) ↔ (M = 1) ∨ (M = 4) :=
by
  sorry

end find_all_pos_integers_l86_86603


namespace smallest_prime_less_than_perfect_square_l86_86360

theorem smallest_prime_less_than_perfect_square : ∃ (n : ℕ), ∃ (k : ℤ), n = (k^2 - 12 : ℤ) ∧ nat.prime n ∧ n > 0 ∧ ∀ (m : ℕ), (∃ (j : ℤ), m = (j^2 - 12 : ℤ) ∧ nat.prime m ∧ m > 0) → n ≤ m :=
begin
  sorry
end

end smallest_prime_less_than_perfect_square_l86_86360


namespace inequalities_in_quadrants_l86_86696

theorem inequalities_in_quadrants (x y : ℝ) :
  (y > - (1 / 2) * x + 6) ∧ (y > 3 * x - 4) → (x > 0) ∧ (y > 0) :=
  sorry

end inequalities_in_quadrants_l86_86696


namespace mean_of_counts_is_7_l86_86500

theorem mean_of_counts_is_7 (counts : List ℕ) (h : counts = [6, 12, 1, 12, 7, 3, 8]) :
  counts.sum / counts.length = 7 :=
by
  sorry

end mean_of_counts_is_7_l86_86500


namespace find_analytical_expression_of_f_l86_86913

-- Given conditions: f(1/x) = 1/(x+1)
def f (x : ℝ) : ℝ := sorry

-- Domain statement (optional for additional clarity):
def domain (x : ℝ) := x ≠ 0 ∧ x ≠ -1

-- Proof obligation: Prove that f(x) = x / (x + 1)
theorem find_analytical_expression_of_f :
  ∀ x : ℝ, domain x → f x = x / (x + 1) := sorry

end find_analytical_expression_of_f_l86_86913


namespace bottom_row_bricks_l86_86141

theorem bottom_row_bricks (n : ℕ) 
  (h1 : (n + (n-1) + (n-2) + (n-3) + (n-4) = 200)) : 
  n = 42 := 
by sorry

end bottom_row_bricks_l86_86141


namespace find_theta_l86_86819

variables (R h θ : ℝ)
hypothesis h1 : (r₁ = R * cos θ)
hypothesis h2 : (r₂ = (R + h) * cos θ)
hypothesis h3 : (s = 2 * π * r₂ - 2 * π * r₁)
hypothesis h4 : (s = 2 * π * h * cos θ)
hypothesis h5 : (s = h)

theorem find_theta : θ = real.arccos (1 / (2 * π)) :=
by
  sorry

end find_theta_l86_86819


namespace angle_WYZ_correct_l86_86291

-- Define the angles as constants
def angle_XYZ : ℝ := 36
def angle_XYW : ℝ := 15

-- Theorem statement asserting the solution
theorem angle_WYZ_correct :
  (angle_XYZ - angle_XYW = 21) := 
by
  -- This is where the proof would go, but we use 'sorry' as instructed
  sorry

end angle_WYZ_correct_l86_86291


namespace total_kids_l86_86573

theorem total_kids (girls boys: ℕ) (h1: girls = 3) (h2: boys = 6) : girls + boys = 9 :=
by
  sorry

end total_kids_l86_86573


namespace parabola_coeff_sum_l86_86182

theorem parabola_coeff_sum (a b c : ℤ) (h₁ : a * (1:ℤ)^2 + b * 1 + c = 3)
                                      (h₂ : a * (-1)^2 + b * (-1) + c = 5)
                                      (vertex : ∀ x, a * (x + 1)^2 + 1 = a * x^2 + bx + c) :
a + b + c = 3 := 
sorry

end parabola_coeff_sum_l86_86182


namespace sum_of_digits_1197_l86_86640

theorem sum_of_digits_1197 : (1 + 1 + 9 + 7 = 18) := by sorry

end sum_of_digits_1197_l86_86640


namespace reciprocal_of_neg_2023_l86_86975

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l86_86975


namespace water_tank_height_l86_86821

theorem water_tank_height (r h : ℝ) (V : ℝ) (V_water : ℝ) (a b : ℕ) 
  (h_tank : h = 120) (r_tank : r = 20) (V_tank : V = (1/3) * π * r^2 * h) 
  (V_water_capacity : V_water = 0.4 * V) :
  a = 48 ∧ b = 2 ∧ V = 16000 * π ∧ V_water = 6400 * π ∧ 
  h_water = 48 * (2^(1/3) / 1) ∧ (a + b = 50) :=
by
  sorry

end water_tank_height_l86_86821


namespace expected_matches_is_one_variance_matches_is_one_l86_86843

noncomputable def indicator (k : ℕ) (matches : Finset ℕ) : ℕ :=
  if k ∈ matches then 1 else 0

def expected_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  (Finset.range N).sum (λ k, indicator k matches / N)

def variance_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  let E_S := expected_matches N matches in
  let E_S2 := (Finset.range N).sum (λ k, (indicator k matches) ^ 2 / N) in
  E_S2 - E_S ^ 2

theorem expected_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  expected_matches N matches = 1 := sorry

theorem variance_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  variance_matches N matches = 1 := sorry

end expected_matches_is_one_variance_matches_is_one_l86_86843


namespace grocery_cost_l86_86958

/-- Potatoes and celery costs problem. -/
theorem grocery_cost (a b : ℝ) (potato_cost_per_kg celery_cost_per_kg : ℝ) 
(h1 : potato_cost_per_kg = 1) (h2 : celery_cost_per_kg = 0.7) :
  potato_cost_per_kg * a + celery_cost_per_kg * b = a + 0.7 * b :=
by
  rw [h1, h2]
  sorry

end grocery_cost_l86_86958


namespace find_p_l86_86809

theorem find_p (p q : ℝ) (h1 : p + 2 * q = 1) (h2 : p > 0) (h3 : q > 0) (h4 : 10 * p^9 * q = 45 * p^8 * q^2): 
  p = 9 / 13 :=
by
  sorry

end find_p_l86_86809


namespace find_y_l86_86766

theorem find_y (x y : ℤ) (h1 : x = -4) (h2 : x^2 + 3 * x + 7 = y - 5) : y = 16 := 
by
  sorry

end find_y_l86_86766


namespace initial_horses_to_cows_ratio_l86_86304

theorem initial_horses_to_cows_ratio (H C : ℕ) (h₁ : (H - 15) / (C + 15) = 13 / 7) (h₂ : H - 15 = C + 45) :
  H / C = 4 / 1 := 
sorry

end initial_horses_to_cows_ratio_l86_86304


namespace range_of_m_l86_86120

namespace ProofProblem

-- Define propositions P and Q in Lean
def P (m : ℝ) : Prop := 2 * m > 1
def Q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

-- Assumptions
variables (m : ℝ)
axiom hP_or_Q : P m ∨ Q m
axiom hP_and_Q_false : ¬(P m ∧ Q m)

-- We need to prove the range of m
theorem range_of_m : m ∈ (Set.Icc (-2 : ℝ) (1 / 2 : ℝ) ∪ Set.Ioi (2 : ℝ)) :=
sorry

end ProofProblem

end range_of_m_l86_86120


namespace minimum_value_of_sum_l86_86753

open Real

theorem minimum_value_of_sum {a b : ℝ} (h₁ : 0 < a) (h₂ : 0 < b) (h : log a / log 2 + log b / log 2 ≥ 6) :
  a + b ≥ 16 :=
sorry

end minimum_value_of_sum_l86_86753


namespace swim_club_percentage_l86_86070

theorem swim_club_percentage (P : ℕ) (total_members : ℕ) (not_passed_taken_course : ℕ) (not_passed_not_taken_course : ℕ) :
  total_members = 50 →
  not_passed_taken_course = 5 →
  not_passed_not_taken_course = 30 →
  (total_members - (total_members * P / 100) = not_passed_taken_course + not_passed_not_taken_course) →
  P = 30 :=
by
  sorry

end swim_club_percentage_l86_86070


namespace square_garden_area_l86_86676

theorem square_garden_area (P A : ℕ)
  (h1 : P = 40)
  (h2 : A = 2 * P + 20) :
  A = 100 :=
by
  rw [h1] at h2 -- Substitute h1 (P = 40) into h2 (A = 2P + 20)
  norm_num at h2 -- Normalize numeric expressions in h2
  exact h2 -- Conclude by showing h2 (A = 100) holds

-- The output should be able to build successfully without solving the proof.

end square_garden_area_l86_86676


namespace find_f_neg3_l86_86248

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x^2 - 2 * x else -(x^2 - 2 * -x)

theorem find_f_neg3 (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_pos : ∀ x : ℝ, 0 < x → f x = x^2 - 2 * x) : f (-3) = -3 :=
by
  sorry

end find_f_neg3_l86_86248


namespace total_fleas_l86_86906

-- Definitions based on conditions provided
def fleas_Gertrude : Nat := 10
def fleas_Olive : Nat := fleas_Gertrude / 2
def fleas_Maud : Nat := 5 * fleas_Olive

-- Prove the total number of fleas on all three chickens
theorem total_fleas :
  fleas_Gertrude + fleas_Olive + fleas_Maud = 40 :=
by sorry

end total_fleas_l86_86906


namespace solve_x_l86_86907

def δ (x : ℝ) : ℝ := 4 * x + 6
def φ (x : ℝ) : ℝ := 5 * x + 4

theorem solve_x : ∃ x: ℝ, δ (φ x) = 3 → x = -19 / 20 := by
  sorry

end solve_x_l86_86907


namespace find_principal_amount_l86_86074

theorem find_principal_amount 
  (total_interest : ℝ)
  (rate1 rate2 : ℝ)
  (years1 years2 : ℕ)
  (P : ℝ)
  (A1 A2 : ℝ) 
  (hA1 : A1 = P * (1 + rate1/100)^years1)
  (hA2 : A2 = A1 * (1 + rate2/100)^years2)
  (hInterest : A2 = P + total_interest) : 
  P = 25252.57 :=
by
  -- Given the conditions above, we prove the main statement.
  sorry

end find_principal_amount_l86_86074


namespace total_tickets_correct_l86_86032

-- Define the initial number of tickets Tate has
def initial_tickets_Tate : ℕ := 32

-- Define the additional tickets Tate buys
def additional_tickets_Tate : ℕ := 2

-- Calculate the total number of tickets Tate has
def total_tickets_Tate : ℕ := initial_tickets_Tate + additional_tickets_Tate

-- Define the number of tickets Peyton has (half of Tate's total tickets)
def tickets_Peyton : ℕ := total_tickets_Tate / 2

-- Calculate the total number of tickets Tate and Peyton have together
def total_tickets_together : ℕ := total_tickets_Tate + tickets_Peyton

-- Prove that the total number of tickets together equals 51
theorem total_tickets_correct : total_tickets_together = 51 := by
  sorry

end total_tickets_correct_l86_86032


namespace min_value_expression_l86_86749

-- Define the given problem conditions and statement
theorem min_value_expression :
  ∀ (x y : ℝ), 0 < x → 0 < y → 6 ≤ (y / x) + (16 * x / (2 * x + y)) :=
by
  sorry

end min_value_expression_l86_86749


namespace equivalent_expression_l86_86764

-- Define the conditions and the statement that needs to be proven
theorem equivalent_expression (x : ℝ) (h : x^2 - 2 * x + 1 = 0) : 2 * x^2 - 4 * x = -2 := 
  by
    sorry

end equivalent_expression_l86_86764


namespace exact_consecutive_hits_l86_86167

/-
Prove the number of ways to arrange 8 shots with exactly 3 hits such that exactly 2 out of the 3 hits are consecutive is 30.
-/

def count_distinct_sequences (total_shots : ℕ) (hits : ℕ) (consecutive_hits : ℕ) : ℕ :=
  if total_shots = 8 ∧ hits = 3 ∧ consecutive_hits = 2 then 30 else 0

theorem exact_consecutive_hits :
  count_distinct_sequences 8 3 2 = 30 :=
by
  -- The proof is omitted.
  sorry

end exact_consecutive_hits_l86_86167


namespace angle_EMC_ninety_degrees_l86_86288

open EuclideanGeometry

noncomputable def incircle_tangent_points (A B C : Point) :=
  let ic := incircle ⟨A, B, C⟩
  let ⟨C1⟩ := is_incircle_tangent_to_AB A B ic
  let ⟨B1⟩ := is_incircle_tangent_to_AC A C ic
  let ⟨A1⟩ := is_incircle_tangent_to_BC B C ic
  (C1, B1, A1)

noncomputable def point_E (A A1 : Point) (ic : Circle) : Point :=
  intersection_point (line A A1) ic

noncomputable def midpoint_N (B1 A1 : Point) : Point :=
  midpoint B1 A1

noncomputable def point_M (N A A1 : Point) : Point :=
  reflection_of_point_across_line (line A A1) N

theorem angle_EMC_ninety_degrees (A B C : Point) :
  let (C1, B1, A1) := incircle_tangent_points A B C
  let ic := incircle ⟨A, B, C⟩
  let E := point_E A A1 ic
  let N := midpoint_N B1 A1
  let M := point_M N A A1
  ∠ (line E M) (line E C) = 90 :=
by sorry

end angle_EMC_ninety_degrees_l86_86288


namespace problem_statement_l86_86283

theorem problem_statement (x θ : ℝ) (h : Real.logb 2 x + Real.cos θ = 2) : |x - 8| + |x + 2| = 10 :=
sorry

end problem_statement_l86_86283


namespace parts_drawn_l86_86745

-- Given that a sample of 30 parts is drawn and each part has a 25% chance of being drawn,
-- prove that the total number of parts N is 120.

theorem parts_drawn (N : ℕ) (h : (30 : ℚ) / N = 0.25) : N = 120 :=
sorry

end parts_drawn_l86_86745


namespace payment_first_trip_payment_second_trip_l86_86224

-- Define conditions and questions
variables {x y : ℝ}

-- Conditions: discounts and expenditure
def discount_1st_trip (x : ℝ) := 0.9 * x
def discount_2nd_trip (y : ℝ) := 300 * 0.9 + (y - 300) * 0.8

def combined_discount (x y : ℝ) := 300 * 0.9 + (x + y - 300) * 0.8

-- Given conditions as equations
axiom eq1 : discount_1st_trip x + discount_2nd_trip y - combined_discount x y = 19
axiom eq2 : x + y - (discount_1st_trip x + discount_2nd_trip y) = 67

-- The proof statements
theorem payment_first_trip : discount_1st_trip 190 = 171 := by sorry

theorem payment_second_trip : discount_2nd_trip 390 = 342 := by sorry

end payment_first_trip_payment_second_trip_l86_86224


namespace divisible_by_10_l86_86783

theorem divisible_by_10 : (11 * 21 * 31 * 41 * 51 - 1) % 10 = 0 := by
  sorry

end divisible_by_10_l86_86783


namespace water_usage_correct_l86_86390

variable (y : ℝ) (C₁ : ℝ) (C₂ : ℝ) (x : ℝ)

noncomputable def water_bill : ℝ :=
  if x ≤ 4 then C₁ * x else 4 * C₁ + C₂ * (x - 4)

theorem water_usage_correct (h1 : y = 12.8) (h2 : C₁ = 1.2) (h3 : C₂ = 1.6) : x = 9 :=
by
  have h4 : x > 4 := sorry
  sorry

end water_usage_correct_l86_86390


namespace parallel_line_equation_perpendicular_line_equation_l86_86263

theorem parallel_line_equation {x y : ℝ} (P : ∃ x y, 2 * x + y - 5 = 0 ∧ x - 2 * y = 0) :
  (∃ (l : ℝ), ∀ x y, 4 * x - y - 7 = 0) :=
sorry

theorem perpendicular_line_equation {x y : ℝ} (P : ∃ x y, 2 * x + y - 5 = 0 ∧ x - 2 * y = 0) :
  (∃ (l : ℝ), ∀ x y, x + 4 * y - 6 = 0) :=
sorry

end parallel_line_equation_perpendicular_line_equation_l86_86263


namespace siblings_pizza_order_l86_86609

theorem siblings_pizza_order 
  (hAlex : ℚ := 1/7) 
  (hBeth : ℚ := 2/5) 
  (hCyril : ℚ := 3/10) 
  (hLeftover : ℚ := 1 - (hAlex + hBeth + hCyril)) 
  (hDan : ℚ := 2 * hLeftover) 
  (slices : ℚ := 70)
  (alex_slices : ℚ := hAlex * slices)
  (beth_slices : ℚ := hBeth * slices)
  (cyril_slices : ℚ := hCyril * slices)
  (dan_slices : ℚ := hDan * slices):
  [beth_slices, dan_slices, cyril_slices, alex_slices].sort (≥) = [28, 22, 21, 10] := 
by 
  sorry

end siblings_pizza_order_l86_86609


namespace full_house_probability_l86_86055

open Nat

theorem full_house_probability :
  let total_outcomes := choose 52 5,
      heart_rank_ways := 13,
      heart_card_ways := choose 4 3,
      club_rank_ways := 12,
      club_card_ways := choose 4 2,
      successful_outcomes := heart_rank_ways * heart_card_ways * club_rank_ways * club_card_ways
  in
  (successful_outcomes : ℚ) / total_outcomes = 6 / 4165 := by
  sorry

end full_house_probability_l86_86055


namespace find_y_l86_86220

theorem find_y (y : ℝ) (h_cond : y = (1 / y) * (-y) - 3) : y = -4 := 
sorry

end find_y_l86_86220


namespace number_of_small_branches_l86_86578

-- Define the number of small branches grown by each branch as a variable
variable (x : ℕ)

-- Define the total number of main stems, branches, and small branches
def total := 1 + x + x * x

theorem number_of_small_branches (h : total x = 91) : x = 9 :=
by
  -- Proof is not required as per instructions
  sorry

end number_of_small_branches_l86_86578


namespace perfect_squares_between_50_and_200_l86_86450

theorem perfect_squares_between_50_and_200 : ∃ n m : ℕ, (8 ≤ n ∧ n ≤ 14) ∧ (m - n + 1 = 7) :=
by {
  use 8, 14,
  split,
  {
    exact ⟨by norm_num, by norm_num⟩,
  },
  {
    norm_num,
  },
  sorry
}

end perfect_squares_between_50_and_200_l86_86450


namespace smallest_logarithmic_term_l86_86617

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem smallest_logarithmic_term (x₀ : ℝ) (hx₀ : f x₀ = 0) (h_interval : 2 < x₀ ∧ x₀ < Real.exp 1) :
  min (min (Real.log x₀) (Real.log (Real.sqrt x₀))) (min (Real.log (Real.log x₀)) ((Real.log x₀)^2)) = Real.log (Real.log x₀) := 
by
  sorry

end smallest_logarithmic_term_l86_86617


namespace smallest_prime_12_less_perfect_square_l86_86342

def is_prime (n : ℕ) : Prop := nat.prime n

def is_perfect_square_less_12 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 12

def smallest_prime (P : ℕ → Prop) : ℕ :=
  if h : ∃ n, P n then classical.some h else 0

def satisfies_conditions (n : ℕ) : Prop :=
  is_prime n ∧ is_perfect_square_less_12 n

theorem smallest_prime_12_less_perfect_square :
  smallest_prime satisfies_conditions = 13 :=
  sorry

end smallest_prime_12_less_perfect_square_l86_86342


namespace sum_of_eight_numbers_l86_86531

-- Definitions used in the conditions
variables {a b c d e f g h : ℕ}

-- Given condition
axiom product_condition : (a + b) * (c + d) * (e + f) * (g + h) = 330

-- Define individual sums
def ab_sum := a + b
def cd_sum := c + d
def ef_sum := e + f
def gh_sum := g + h

-- Define the total sum of the eight numbers on the cards
def total_sum := ab_sum + cd_sum + ef_sum + gh_sum

-- The theorem to prove
theorem sum_of_eight_numbers : total_sum = 21 := by
  have ab_sum_eq : ab_sum = 2 := sorry
  have cd_sum_eq : cd_sum = 3 := sorry
  have ef_sum_eq : ef_sum = 5 := sorry
  have gh_sum_eq : gh_sum = 11 := sorry
  rw [ab_sum_eq, cd_sum_eq, ef_sum_eq, gh_sum_eq]
  norm_num

end sum_of_eight_numbers_l86_86531


namespace team_a_daily_work_rate_l86_86675

theorem team_a_daily_work_rate
  (L : ℕ) (D1 : ℕ) (D2 : ℕ) (w : ℕ → ℕ)
  (hL : L = 8250)
  (hD1 : D1 = 4)
  (hD2 : D2 = 7)
  (hwB : ∀ (x : ℕ), w x = x + 150)
  (hwork : ∀ (x : ℕ), D1 * x + D2 * (x + (w x)) = L) :
  ∃ x : ℕ, x = 400 :=
by
  sorry

end team_a_daily_work_rate_l86_86675


namespace cos_4_arccos_fraction_l86_86894

theorem cos_4_arccos_fraction :
  (Real.cos (4 * Real.arccos (2 / 5))) = (-47 / 625) :=
by
  sorry

end cos_4_arccos_fraction_l86_86894


namespace cards_sum_l86_86517

theorem cards_sum (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
by
  sorry

end cards_sum_l86_86517


namespace parabola_has_one_x_intercept_l86_86621

-- Define the equation of the parabola.
def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 4

-- Prove that the number of x-intercepts of the graph of the parabola is 1.
theorem parabola_has_one_x_intercept : (∃! y : ℝ, parabola y = 4) :=
by
  sorry

end parabola_has_one_x_intercept_l86_86621


namespace no_real_roots_of_ffx_or_ggx_l86_86054

noncomputable def is_unitary_quadratic_trinomial (p : ℝ → ℝ) : Prop :=
∃ b c : ℝ, ∀ x : ℝ, p x = x^2 + b*x + c

theorem no_real_roots_of_ffx_or_ggx 
    (f g : ℝ → ℝ) 
    (hf : is_unitary_quadratic_trinomial f) 
    (hg : is_unitary_quadratic_trinomial g)
    (hf_ng : ∀ x : ℝ, f (g x) ≠ 0)
    (hg_nf : ∀ x : ℝ, g (f x) ≠ 0) :
    (∀ x : ℝ, f (f x) ≠ 0) ∨ (∀ x : ℝ, g (g x) ≠ 0) :=
sorry

end no_real_roots_of_ffx_or_ggx_l86_86054


namespace three_digit_number_ends_with_same_three_digits_l86_86895

theorem three_digit_number_ends_with_same_three_digits (N : ℕ) (hN : 100 ≤ N ∧ N < 1000) :
  (∀ k : ℕ, k ≥ 1 → N^k % 1000 = N % 1000) ↔ (N = 376 ∨ N = 625) := 
sorry

end three_digit_number_ends_with_same_three_digits_l86_86895


namespace problem1_problem2_l86_86131

noncomputable def A : Set ℝ := Set.Icc 1 4
noncomputable def B (a : ℝ) : Set ℝ := Set.Iio a

-- Problem 1
theorem problem1 (A := A) (B := B 4) : A ∩ B = Set.Icc 1 4 := by
  sorry 

-- Problem 2
theorem problem2 (A := A) : ∀ a : ℝ, (A ⊆ B a) → (4 ≤ a) := by
  sorry

end problem1_problem2_l86_86131


namespace arithmetic_sequence_product_l86_86657

theorem arithmetic_sequence_product (b : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ n, b (n + 1) > b n)
  (h2 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
sorry

end arithmetic_sequence_product_l86_86657


namespace sum_first_nine_primes_l86_86112

theorem sum_first_nine_primes : 
  2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 = 100 :=
by
  sorry

end sum_first_nine_primes_l86_86112


namespace craig_apples_total_l86_86247

-- Conditions
def initial_apples := 20.0
def additional_apples := 7.0

-- Question turned into a proof problem
theorem craig_apples_total : initial_apples + additional_apples = 27.0 :=
by
  sorry

end craig_apples_total_l86_86247


namespace area_covered_by_three_layers_l86_86711

theorem area_covered_by_three_layers (A B C : ℕ) (total_wallpaper : ℕ := 300)
  (wall_area : ℕ := 180) (two_layer_coverage : ℕ := 30) :
  A + 2 * B + 3 * C = total_wallpaper ∧ B + C = total_wallpaper - wall_area ∧ B = two_layer_coverage → 
  C = 90 :=
by
  sorry

end area_covered_by_three_layers_l86_86711


namespace cost_price_per_metre_l86_86385

theorem cost_price_per_metre (total_metres total_sale total_loss_per_metre total_sell_price : ℕ) (h1: total_metres = 500) (h2: total_sell_price = 15000) (h3: total_loss_per_metre = 10) : total_sell_price + (total_loss_per_metre * total_metres) / total_metres = 40 :=
by
  sorry

end cost_price_per_metre_l86_86385


namespace smallest_prime_less_than_perfect_square_is_13_l86_86345

noncomputable def smallest_prime_less_than_perfect_square : ℕ :=
  Inf {p : ℕ | ∃ k : ℕ, p = k^2 - 12 ∧ p > 0 ∧ Nat.Prime p}

theorem smallest_prime_less_than_perfect_square_is_13 :
  smallest_prime_less_than_perfect_square = 13 := by
  sorry

end smallest_prime_less_than_perfect_square_is_13_l86_86345


namespace sum_of_eight_numbers_l86_86516

theorem sum_of_eight_numbers (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_eight_numbers_l86_86516


namespace problem_l86_86885

theorem problem (a b : ℚ) (x : ℚ) (hx : 0 < x) :
  (a / (10^(x+1) - 1) + b / (10^(x+1) + 3) = (3 * 10^x + 4) / ((10^(x+1) - 1) * (10^(x+1) + 3))) →
  a - b = 37 / 20 :=
sorry

end problem_l86_86885


namespace probability_sum_18_is_1_over_54_l86_86930

open Finset

-- Definitions for a 6-faced die, four rolls, and a probability space.
def faces := {1, 2, 3, 4, 5, 6}
def dice_rolls : Finset (Finset ℕ) := product faces (product faces (product faces faces))

def valid_sum : ℕ := 18

noncomputable def probability_of_sum_18 : ℚ :=
  (dice_rolls.filter (λ r, r.sum = valid_sum)).card / dice_rolls.card

theorem probability_sum_18_is_1_over_54 :
  probability_of_sum_18 = 1 / 54 := 
  sorry

end probability_sum_18_is_1_over_54_l86_86930


namespace smallest_prime_12_less_than_square_l86_86334

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l86_86334


namespace perfect_squares_count_between_50_and_200_l86_86447

theorem perfect_squares_count_between_50_and_200 : 
  let count := (λ n m : ℤ, n - m + 1) in
  ∃ n m : ℕ, 50 < n^2 ∧ n^2 < 200 ∧ 50 < m^2 ∧ m^2 < 200 ∧ count m n = 7 :=
begin
  sorry
end

end perfect_squares_count_between_50_and_200_l86_86447


namespace fraction_remains_unchanged_l86_86632

theorem fraction_remains_unchanged (x y : ℝ) : 
  (3 * (3 * x)) / (2 * (3 * x) - 3 * y) = (3 * x) / (2 * x - y) :=
by
  sorry

end fraction_remains_unchanged_l86_86632


namespace sum_of_eight_numbers_on_cards_l86_86508

theorem sum_of_eight_numbers_on_cards :
  ∃ (a b c d e f g h : ℕ),
  (a + b) * (c + d) * (e + f) * (g + h) = 330 ∧
  (a + b + c + d + e + f + g + h) = 21 :=
by
  sorry

end sum_of_eight_numbers_on_cards_l86_86508


namespace problem_one_problem_two_l86_86266

noncomputable def f (x m : ℝ) : ℝ := x^2 - (m-1) * x + 2 * m

theorem problem_one (m : ℝ) : (∀ x : ℝ, 0 < x → f x m > 0) ↔ (-2 * Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2 * Real.sqrt 6 + 5) :=
by
  sorry

theorem problem_two (m : ℝ) : (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x m = 0) ↔ (m ∈ Set.Ioo (-2 : ℝ) 0) :=
by
  sorry

end problem_one_problem_two_l86_86266


namespace extended_hexagon_area_l86_86333

theorem extended_hexagon_area (original_area : ℝ) (side_length_extension : ℝ)
  (original_side_length : ℝ) (new_side_length : ℝ) :
  original_area = 18 ∧ side_length_extension = 1 ∧ original_side_length = 2 
  ∧ new_side_length = original_side_length + 2 * side_length_extension →
  36 = original_area + 6 * (0.5 * side_length_extension * (original_side_length + 1)) := 
sorry

end extended_hexagon_area_l86_86333


namespace field_trip_bread_l86_86225

theorem field_trip_bread (group_size : ℕ) (groups : ℕ) 
    (students_per_group : group_size = 5 + 1)
    (total_groups : groups = 5)
    (sandwiches_per_student : ℕ := 2) 
    (bread_per_sandwich : ℕ := 2) : 
    (groups * group_size * sandwiches_per_student * bread_per_sandwich) = 120 := 
by 
    have students_per_group_lemma : group_size = 6 := by sorry
    have total_students := groups * group_size
    have _ : total_students = 30 := by sorry
    have total_sandwiches := total_students * sandwiches_per_student
    have _ : total_sandwiches = 60 := by sorry
    have total_bread := total_sandwiches * bread_per_sandwich
    have _ : total_bread = 120 := by sorry
    exact id 120

end field_trip_bread_l86_86225


namespace x_intercept_is_2_l86_86822

noncomputable def x_intercept_of_line : ℝ :=
  by
  sorry -- This is where the proof would go

theorem x_intercept_is_2 :
  (∀ x y : ℝ, 5 * x - 2 * y - 10 = 0 → y = 0 → x = 2) :=
  by
  intro x y H_eq H_y0
  rw [H_y0] at H_eq
  simp at H_eq
  sorry -- This is where the proof would go

end x_intercept_is_2_l86_86822


namespace perfect_squares_50_to_200_l86_86457

theorem perfect_squares_50_to_200 : 
  ∃ (k : ℕ), k = 7 ∧ ∀ n : ℤ, 50 < n^2 ∧ n^2 < 200 -> (8 ≤ n ∧ n ≤ 14) := 
by
  sorry

end perfect_squares_50_to_200_l86_86457


namespace sum_of_fourth_powers_eq_174_fourth_l86_86700

theorem sum_of_fourth_powers_eq_174_fourth :
  120 ^ 4 + 97 ^ 4 + 84 ^ 4 + 27 ^ 4 = 174 ^ 4 :=
by
  sorry

end sum_of_fourth_powers_eq_174_fourth_l86_86700


namespace unique_exponential_solution_l86_86296

theorem unique_exponential_solution (a x : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hx_pos : 0 < x) :
  ∃! y : ℝ, a^y = x :=
by
  sorry

end unique_exponential_solution_l86_86296


namespace inverse_matrix_l86_86616

theorem inverse_matrix
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (B : Matrix (Fin 2) (Fin 2) ℚ)
  (H : A * B = ![![1, 2], ![0, 6]]) :
  A⁻¹ = ![![-1, 0], ![0, 2]] :=
sorry

end inverse_matrix_l86_86616


namespace music_player_winner_l86_86569

theorem music_player_winner (n : ℕ) (h1 : ∀ k, k % n = 0 → k = 35) (h2 : 35 % 7 = 0) (h3 : 35 % n = 0) (h4 : n ≠ 1) (h5 : n ≠ 7) (h6 : n ≠ 35) : n = 5 := 
sorry

end music_player_winner_l86_86569


namespace ratio_of_areas_l86_86971

variables (s : ℝ)

def side_length_square := s
def longer_side_rect := 1.2 * s
def shorter_side_rect := 0.8 * s

noncomputable def area_rectangle := longer_side_rect s * shorter_side_rect s
noncomputable def area_triangle := (1 / 2) * (longer_side_rect s * shorter_side_rect s)

theorem ratio_of_areas :
  (area_triangle s) / (area_rectangle s) = 1 / 2 :=
by
  sorry

end ratio_of_areas_l86_86971


namespace exists_five_distinct_natural_numbers_product_eq_1000_l86_86595

theorem exists_five_distinct_natural_numbers_product_eq_1000 :
  ∃ (a b c d e : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a * b * c * d * e = 1000 := sorry

end exists_five_distinct_natural_numbers_product_eq_1000_l86_86595


namespace dilation_result_l86_86179

-- Definitions based on conditions:
def center : ℂ := 1 + 2 * complex.I
def scale_factor : ℂ := 4
def original_point : ℂ := -2 - 2 * complex.I

-- Statement:
theorem dilation_result :
  (scale_factor * (original_point - center) + center) = -11 - 14 * complex.I :=
by
  -- placeholder for the actual proof
  sorry

end dilation_result_l86_86179


namespace largest_y_coordinate_ellipse_l86_86884

theorem largest_y_coordinate_ellipse (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 := 
by
  -- proof to be filled in
  sorry

end largest_y_coordinate_ellipse_l86_86884


namespace molecular_weight_of_10_moles_of_Al2S3_l86_86564

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06

-- Define the molecular weight calculation for Al2S3
def molecular_weight_Al2S3 : ℝ :=
  (2 * atomic_weight_Al) + (3 * atomic_weight_S)

-- Define the molecular weight for 10 moles of Al2S3
def molecular_weight_10_moles_Al2S3 : ℝ :=
  10 * molecular_weight_Al2S3

-- The theorem to prove
theorem molecular_weight_of_10_moles_of_Al2S3 :
  molecular_weight_10_moles_Al2S3 = 1501.4 :=
by
  -- skip the proof
  sorry

end molecular_weight_of_10_moles_of_Al2S3_l86_86564


namespace faye_pencils_l86_86416

theorem faye_pencils :
  ∀ (packs : ℕ) (pencils_per_pack : ℕ) (rows : ℕ) (total_pencils pencils_per_row : ℕ),
  packs = 35 →
  pencils_per_pack = 4 →
  rows = 70 →
  total_pencils = packs * pencils_per_pack →
  pencils_per_row = total_pencils / rows →
  pencils_per_row = 2 :=
by
  intros packs pencils_per_pack rows total_pencils pencils_per_row
  intros packs_eq pencils_per_pack_eq rows_eq total_pencils_eq pencils_per_row_eq
  sorry

end faye_pencils_l86_86416


namespace geometric_sequence_second_term_l86_86685

theorem geometric_sequence_second_term (a r : ℝ) 
  (h_fifth_term : a * r^4 = 48) 
  (h_sixth_term : a * r^5 = 72) : 
  a * r = 1152 / 81 := by
  sorry

end geometric_sequence_second_term_l86_86685


namespace even_factors_count_l86_86442

theorem even_factors_count (n : ℕ) (h : n = 2^3 * 3 * 7^2 * 5) : 
  ∃ k, k = 36 ∧ 
       (∀ a b c d : ℕ, 1 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 0 ≤ d ∧ d ≤ 1 →
       ∃ m, m = 2^a * 3^b * 7^c * 5^d ∧ 2 ∣ m ∧ m ∣ n) := sorry

end even_factors_count_l86_86442


namespace total_rainfall_recorded_l86_86088

-- Define the conditions based on the rainfall amounts for each day
def rainfall_monday : ℝ := 0.16666666666666666
def rainfall_tuesday : ℝ := 0.4166666666666667
def rainfall_wednesday : ℝ := 0.08333333333333333

-- State the theorem: the total rainfall recorded over the three days is 0.6666666666666667 cm.
theorem total_rainfall_recorded :
  (rainfall_monday + rainfall_tuesday + rainfall_wednesday) = 0.6666666666666667 := by
  sorry

end total_rainfall_recorded_l86_86088


namespace erwin_chocolates_weeks_l86_86601

-- Define weekdays chocolates and weekends chocolates
def weekdays_chocolates := 2
def weekends_chocolates := 1

-- Define the total chocolates Erwin ate
def total_chocolates := 24

-- Define the number of weekdays and weekend days in a week
def weekdays := 5
def weekends := 2

-- Define the total chocolates Erwin eats in a week
def chocolates_per_week : Nat := (weekdays * weekdays_chocolates) + (weekends * weekends_chocolates)

-- Prove that Erwin finishes all chocolates in 2 weeks
theorem erwin_chocolates_weeks : (total_chocolates / chocolates_per_week) = 2 := by
  sorry

end erwin_chocolates_weeks_l86_86601


namespace number_of_6mb_pictures_l86_86021

theorem number_of_6mb_pictures
    (n : ℕ)             -- initial number of pictures
    (size_old : ℕ)      -- size of old pictures in megabytes
    (size_new : ℕ)      -- size of new pictures in megabytes
    (total_capacity : ℕ)  -- total capacity of the memory card in megabytes
    (h1 : n = 3000)      -- given memory card can hold 3000 pictures
    (h2 : size_old = 8)  -- each old picture is 8 megabytes
    (h3 : size_new = 6)  -- each new picture is 6 megabytes
    (h4 : total_capacity = n * size_old)  -- total capacity calculated from old pictures
    : total_capacity / size_new = 4000 :=  -- the number of new pictures that can be held
by
  sorry

end number_of_6mb_pictures_l86_86021


namespace person_last_name_length_l86_86804

theorem person_last_name_length (samantha_lastname: ℕ) (bobbie_lastname: ℕ) (person_lastname: ℕ) 
  (h1: samantha_lastname + 3 = bobbie_lastname)
  (h2: bobbie_lastname - 2 = 2 * person_lastname)
  (h3: samantha_lastname = 7) :
  person_lastname = 4 :=
by 
  sorry

end person_last_name_length_l86_86804


namespace product_plus_one_square_l86_86772

theorem product_plus_one_square (n : ℕ):
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 := 
  sorry

end product_plus_one_square_l86_86772


namespace total_cost_is_correct_l86_86796

noncomputable def total_cost_of_gifts : ℝ :=
  let polo_shirts := 3 * 26
  let necklaces := 2 * 83
  let computer_game := 90
  let socks := 4 * 7
  let books := 3 * 15
  let scarves := 2 * 22
  let mugs := 5 * 8
  let sneakers := 65

  let cost_before_discounts := polo_shirts + necklaces + computer_game + socks + books + scarves + mugs + sneakers

  let discount_books := 0.10 * books
  let discount_sneakers := 0.15 * sneakers
  let cost_after_discounts := cost_before_discounts - discount_books - discount_sneakers

  let sales_tax := 0.065 * cost_after_discounts
  let cost_after_tax := cost_after_discounts + sales_tax

  let final_cost := cost_after_tax - 12

  final_cost

theorem total_cost_is_correct :
  total_cost_of_gifts = 564.96 := by
sorry

end total_cost_is_correct_l86_86796


namespace range_of_m_l86_86574

theorem range_of_m (m : ℝ) : (1^2 + 2*1 - m ≤ 0) ∧ (2^2 + 2*2 - m > 0) → 3 ≤ m ∧ m < 8 := by
  sorry

end range_of_m_l86_86574


namespace cars_meet_in_3_hours_l86_86330

theorem cars_meet_in_3_hours
  (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (t : ℝ)
  (h_distance: distance = 333)
  (h_speed1: speed1 = 54)
  (h_speed2: speed2 = 57)
  (h_equation: speed1 * t + speed2 * t = distance) :
  t = 3 :=
sorry

end cars_meet_in_3_hours_l86_86330


namespace systematic_sampling_questionnaire_B_count_l86_86332

theorem systematic_sampling_questionnaire_B_count (n : ℕ) (N : ℕ) (first_random : ℕ) (range_A_start range_A_end range_B_start range_B_end : ℕ) 
  (h1 : n = 32) (h2 : N = 960) (h3 : first_random = 9) (h4 : range_A_start = 1) (h5 : range_A_end = 460) 
  (h6 : range_B_start = 461) (h7 : range_B_end = 761) :
  ∃ count : ℕ, count = 10 := by
  sorry

end systematic_sampling_questionnaire_B_count_l86_86332


namespace translate_function_l86_86560

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (4 * x + 13 * Real.pi / 12) - 1

theorem translate_function :
  ∀ x : ℝ, f (x) = 2 * Real.sin (4 * x + 13 * Real.pi / 12) - 1 :=
by
  intro x
  sorry

end translate_function_l86_86560


namespace sum_of_c_l86_86756

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * n - 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℕ :=
  2^(n - 1)

-- Define the sequence c_n
def c (n : ℕ) : ℕ :=
  a n * b n

-- Define the sum S_n of the first n terms of c_n
def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => c (i + 1))

-- The main Lean statement
theorem sum_of_c (n : ℕ) : S n = 3 + (n - 1) * 2^(n + 1) :=
  sorry

end sum_of_c_l86_86756


namespace implication_a_lt_b_implies_a_lt_b_plus_1_l86_86909

theorem implication_a_lt_b_implies_a_lt_b_plus_1 (a b : ℝ) (h : a < b) : a < b + 1 := by
  sorry

end implication_a_lt_b_implies_a_lt_b_plus_1_l86_86909


namespace ufo_convention_attendees_l86_86235

theorem ufo_convention_attendees 
  (F M : ℕ) 
  (h1 : F + M = 450) 
  (h2 : M = F + 26) : 
  M = 238 := 
sorry

end ufo_convention_attendees_l86_86235


namespace initial_population_l86_86044

theorem initial_population (P : ℝ) (h1 : ∀ t : ℕ, P * (1.10 : ℝ) ^ t = 26620 → t = 3) : P = 20000 := by
  have h2 : P * (1.10) ^ 3 = 26620 := sorry
  sorry

end initial_population_l86_86044


namespace parabola_axis_of_symmetry_l86_86035

theorem parabola_axis_of_symmetry : 
  ∀ (x : ℝ), x = -1 → (∃ y : ℝ, y = -x^2 - 2*x - 3) :=
by
  sorry

end parabola_axis_of_symmetry_l86_86035


namespace min_sides_regular_polygon_l86_86373

/-- A regular polygon can accurately be placed back in its original position 
    when rotated by 50°.  Prove that the minimum number of sides the polygon 
    should have is 36. -/

theorem min_sides_regular_polygon (n : ℕ) (h : ∃ k : ℕ, 50 * k = 360 / n) : n = 36 :=
  sorry

end min_sides_regular_polygon_l86_86373


namespace general_formula_a_n_general_formula_b_n_l86_86614

-- Prove general formula for the sequence a_n
theorem general_formula_a_n (S : Nat → Nat) (a : Nat → Nat) (h₁ : ∀ n, S n = 2^(n+1) - 2) :
  (∀ n, a n = S n - S (n - 1)) → ∀ n, a n = 2^n :=
by
  sorry

-- Prove general formula for the sequence b_n
theorem general_formula_b_n (a b : Nat → Nat) (h₁ : ∀ n, a n = 2^n) :
  (∀ n, b n = a n + a (n + 1)) → ∀ n, b n = 3 * 2^n :=
by
  sorry

end general_formula_a_n_general_formula_b_n_l86_86614


namespace smallest_prime_12_less_than_square_l86_86338

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l86_86338


namespace sum_of_eight_numbers_l86_86512

theorem sum_of_eight_numbers (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_eight_numbers_l86_86512


namespace shape_is_cone_l86_86900

-- Definition: spherical coordinates and constant phi
def spherical_coords (ρ θ φ : ℝ) : Type := ℝ × ℝ × ℝ
def phi_constant (c : ℝ) (φ : ℝ) : Prop := φ = c

-- Theorem: shape described by φ = c in spherical coordinates is a cone
theorem shape_is_cone (ρ θ c : ℝ) (h₁ : c ∈ set.Icc 0 real.pi) : 
  (∃ (ρ θ : ℝ), spherical_coords ρ θ c = (ρ, θ, c)) → 
  (∀ φ, phi_constant c φ) → 
  shape_is_cone := sorry

end shape_is_cone_l86_86900


namespace solution_set_l86_86737

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x - 2) / (x - 4) ≥ 3

theorem solution_set :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | 4 < x ∧ x ≤ 5} :=
by
  sorry

end solution_set_l86_86737


namespace certain_event_idiom_l86_86567

theorem certain_event_idiom : 
  ∃ (idiom : String), idiom = "Catching a turtle in a jar" ∧ 
  ∀ (option : String), 
    option = "Catching a turtle in a jar" ∨ 
    option = "Carving a boat to find a sword" ∨ 
    option = "Waiting by a tree stump for a rabbit" ∨ 
    option = "Fishing for the moon in the water" → 
    (option = idiom ↔ (option = "Catching a turtle in a jar")) := 
by
  sorry

end certain_event_idiom_l86_86567


namespace sum_of_possible_values_l86_86781

theorem sum_of_possible_values (x : ℝ) (h : |x - 5| + 2 = 4) :
  x = 7 ∨ x = 3 → x = 10 := 
by sorry

end sum_of_possible_values_l86_86781


namespace number_of_even_factors_l86_86443

theorem number_of_even_factors :
  let n := 2^3 * 3^1 * 7^2 * 5^1 in
  let even_factors :=
    ∑ a in finset.range 4 \{0}, -- 1 ≤ a ≤ 3
    ∑ b in finset.range 2,      -- 0 ≤ b ≤ 1
    ∑ c in finset.range 3,      -- 0 ≤ c ≤ 2
    ∑ d in finset.range 2,      -- 0 ≤ d ≤ 1
    (2^a * 3^b * 7^c * 5^d : ℕ) | n % (2^a * 3^b * 7^c * 5^d) = 0 in
  even_factors.card = 36 :=
by
  sorry

end number_of_even_factors_l86_86443


namespace mona_biked_monday_l86_86163

-- Define the constants and conditions
def distance_biked_weekly : ℕ := 30
def distance_biked_wednesday : ℕ := 12
def speed_flat_road : ℕ := 15
def speed_reduction_percentage : ℕ := 20

-- Define the main problem and conditions in Lean
theorem mona_biked_monday (M : ℕ)
  (h1 : 2 * M + distance_biked_wednesday + M = distance_biked_weekly)  -- total distance biked in the week
  (h2 : 2 * M * (100 - speed_reduction_percentage) / 100 / 15 = 2 * M / 12)  -- speed reduction effect
  : M = 6 :=
sorry 

end mona_biked_monday_l86_86163


namespace olympiad_even_group_l86_86724

theorem olympiad_even_group (P : Type) [Fintype P] [Nonempty P] (knows : P → P → Prop)
  (h : ∀ p, (Finset.filter (knows p) Finset.univ).card ≥ 3) :
  ∃ (G : Finset P), G.card > 2 ∧ G.card % 2 = 0 ∧ ∀ p ∈ G, ∃ q₁ q₂ ∈ G, q₁ ≠ p ∧ q₂ ≠ p ∧ knows p q₁ ∧ knows p q₂ :=
by
  sorry

end olympiad_even_group_l86_86724


namespace four_digit_unique_count_l86_86608

theorem four_digit_unique_count : 
  (∃ k : ℕ, k = 14 ∧ ∃ lst : List ℕ, lst.length = 4 ∧ 
    (∀ d ∈ lst, d = 2 ∨ d = 3) ∧ (2 ∈ lst) ∧ (3 ∈ lst)) :=
by
  sorry

end four_digit_unique_count_l86_86608


namespace find_second_term_l86_86689

-- Define the terms and common ratio in the geometric sequence
def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

-- Given the fifth and sixth terms
variables (a r : ℚ)
axiom fifth_term : geometric_sequence a r 5 = 48
axiom sixth_term : geometric_sequence a r 6 = 72

-- Prove that the second term is 128/9
theorem find_second_term : geometric_sequence a r 2 = 128 / 9 :=
sorry

end find_second_term_l86_86689


namespace field_trip_bread_pieces_l86_86226

theorem field_trip_bread_pieces :
  (students_per_group : ℕ) (num_groups : ℕ) (sandwiches_per_student : ℕ) (pieces_per_sandwich : ℕ)
  (H1 : students_per_group = 6)
  (H2 : num_groups = 5)
  (H3 : sandwiches_per_student = 2)
  (H4 : pieces_per_sandwich = 2)
  : 
  let total_students := num_groups * students_per_group in
  let total_sandwiches := total_students * sandwiches_per_student in
  let total_pieces_bread := total_sandwiches * pieces_per_sandwich in
  total_pieces_bread = 120 :=
by
  let total_students := num_groups * students_per_group
  let total_sandwiches := total_students * sandwiches_per_student
  let total_pieces_bread := total_sandwiches * pieces_per_sandwich
  sorry

end field_trip_bread_pieces_l86_86226


namespace turtle_distance_during_rabbit_rest_l86_86389

theorem turtle_distance_during_rabbit_rest
  (D : ℕ)
  (vr vt : ℕ)
  (rabbit_speed_multiple : vr = 15 * vt)
  (rabbit_remaining_distance : D - 100 = 900)
  (turtle_finish_time : true)
  (rabbit_to_be_break : true)
  (turtle_finish_during_rabbit_rest : true) :
  (D - (900 / 15) = 940) :=
by
  sorry

end turtle_distance_during_rabbit_rest_l86_86389


namespace sin_cos_tan_min_value_l86_86739

open Real

theorem sin_cos_tan_min_value :
  ∀ x : ℝ, (sin x)^2 + (cos x)^2 = 1 → (sin x)^4 + (cos x)^4 + (tan x)^2 ≥ 3/2 :=
by
  sorry

end sin_cos_tan_min_value_l86_86739


namespace valueOf_seq_l86_86118

variable (a : ℕ → ℝ)
variable (h_arith_seq : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n)
variable (h_positive : ∀ n : ℕ, a n > 0)
variable (h_arith_subseq : 2 * a 5 = a 3 + a 6)

theorem valueOf_seq (a : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arith_subseq : 2 * a 5 = a 3 + a 6) :
  (∃ q : ℝ, q = 1 ∨ q = (1 + Real.sqrt 5) / 2 ∧ (a 3 + a 5) / (a 4 + a 6) = 1 / q) → 
  (∃ q : ℝ, (a 3 + a 5) / (a 4 + a 6) = 1 ∨ (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2) :=
by
  sorry

end valueOf_seq_l86_86118


namespace find_distance_between_B_and_C_l86_86559

def problem_statement : Prop :=
  ∃ (x y : ℝ),
  (y / 75 + x / 145 = 4.8) ∧ 
  ((x + y) / 100 = 2 + y / 70) ∧ 
  x = 290

theorem find_distance_between_B_and_C : problem_statement :=
by
  sorry

end find_distance_between_B_and_C_l86_86559


namespace sum_first_11_terms_eq_99_l86_86483

variable {a_n : ℕ → ℝ} -- assuming the sequence values are real numbers
variable (S : ℕ → ℝ) -- sum of the first n terms
variable (a₃ a₆ a₉ : ℝ)
variable (h_sequence : ∀ n, a_n n = aₙ 1 + (n - 1) * (a_n 2 - aₙ 1)) -- sequence is arithmetic
variable (h_condition : a₃ + a₉ = 27 - a₆) -- given condition

theorem sum_first_11_terms_eq_99 
  (h_a₃ : a₃ = a_n 3) 
  (h_a₆ : a₆ = a_n 6) 
  (h_a₉ : a₉ = a_n 9) 
  (h_S : S 11 = 11 * a₆) : 
  S 11 = 99 := 
by 
  sorry


end sum_first_11_terms_eq_99_l86_86483


namespace tables_needed_l86_86178

open Nat

def base7_to_base10 (n : Nat) : Nat := 
  3 * 7^2 + 1 * 7^1 + 2 * 7^0

theorem tables_needed (attendees_base7 : Nat) (attendees_base10 : Nat) (tables : Nat) :
  attendees_base7 = 312 ∧ attendees_base10 = base7_to_base10 attendees_base7 ∧ attendees_base10 = 156 ∧ tables = attendees_base10 / 3 → tables = 52 := 
by
  intros
  sorry

end tables_needed_l86_86178


namespace value_of_a_l86_86116

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem value_of_a (a : ℝ) : f' (-1) a = 4 → a = 10 / 3 := by
  sorry

end value_of_a_l86_86116


namespace percent_of_value_l86_86212

theorem percent_of_value (decimal_form : Real) (value : Nat) (expected_result : Real) : 
  decimal_form = 0.25 ∧ value = 300 ∧ expected_result = 75 → 
  decimal_form * value = expected_result := by
  sorry

end percent_of_value_l86_86212


namespace susan_hourly_rate_l86_86176

-- Definitions based on conditions
def vacation_workdays : ℕ := 10 -- Susan is taking a two-week vacation equivalent to 10 workdays

def weekly_workdays : ℕ := 5 -- Susan works 5 days a week

def paid_vacation_days : ℕ := 6 -- Susan has 6 days of paid vacation

def hours_per_day : ℕ := 8 -- Susan works 8 hours a day

def missed_pay_total : ℕ := 480 -- Susan will miss $480 pay on her unpaid vacation days

-- Calculations
def unpaid_vacation_days : ℕ := vacation_workdays - paid_vacation_days

def daily_lost_pay : ℕ := missed_pay_total / unpaid_vacation_days

def hourly_rate : ℕ := daily_lost_pay / hours_per_day

theorem susan_hourly_rate :
  hourly_rate = 15 := by sorry

end susan_hourly_rate_l86_86176


namespace find_m_divisors_l86_86154

/-- Define the set S of positive integer divisors of 25^8. -/
def S : Finset ℕ := (Finset.range (17)).map ⟨λ k, 5^k, Nat.pow_injective_of_injective (Nat.prime.pow_injective (Nat.prime_of_prime 5))⟩

/-- Define a1, a2, a3 being chosen from S such that a1 divides a2 and a2 divides a3 -/
def valid_division_count : ℕ :=
  (Finset.range (17 + 3)).card

/-- Probability that a1 divides a2 and a2 divides a3 is m/n where m and n are relatively prime -/
theorem find_m_divisors (m n : ℕ) (hmc : Nat.gcd m n = 1) : m = 1 :=
  let total_possible_choices := (S.card)^3
  have h : S.card = 17 := rfl
  let total_valid_choices := valid_division_count
  have h1 : valid_division_count = 969 := rfl
  let prob := total_valid_choices / total_possible_choices
  have hprob : prob = 969 / 4913 := by norm_num1
  have h2 : Nat.gcd 969 4913 = 1 := by norm_num1
  sorry

end find_m_divisors_l86_86154


namespace add_complex_eq_required_complex_addition_l86_86059

theorem add_complex_eq (a b c d : ℝ) (i : ℂ) (h : i ^ 2 = -1) :
  (a + b * i) + (c + d * i) = (a + c) + (b + d) * i :=
by sorry

theorem required_complex_addition :
  let a : ℂ := 5 - 3 * i
  let b : ℂ := 2 + 12 * i
  a + b = 7 + 9 * i := 
by sorry

end add_complex_eq_required_complex_addition_l86_86059


namespace div_by_1897_l86_86308

theorem div_by_1897 (n : ℕ) : 1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end div_by_1897_l86_86308


namespace gcd_mult_product_is_perfect_square_l86_86950

-- The statement of the problem in Lean 4
theorem gcd_mult_product_is_perfect_square
  (x y z : ℕ)
  (h : 1/x - 1/y = 1/z) : 
  ∃ k : ℕ, k^2 = Nat.gcd x (Nat.gcd y z) * x * y * z :=
by 
  sorry

end gcd_mult_product_is_perfect_square_l86_86950


namespace circle_diameter_equality_l86_86370

theorem circle_diameter_equality (r d : ℝ) (h₁ : d = 2 * r) (h₂ : π * d = π * r^2) : d = 4 :=
by
  sorry

end circle_diameter_equality_l86_86370


namespace range_of_sum_of_reciprocals_l86_86748

theorem range_of_sum_of_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  ∃ (r : ℝ), ∀ t ∈ Set.Ici (3 + 2 * Real.sqrt 2), t = (1 / x + 1 / y) := 
sorry

end range_of_sum_of_reciprocals_l86_86748


namespace price_of_each_movie_in_first_box_l86_86414

theorem price_of_each_movie_in_first_box (P : ℝ) (total_movies_box1 : ℕ) (total_movies_box2 : ℕ) (price_per_movie_box2 : ℝ) (average_price : ℝ) (total_movies : ℕ) :
  total_movies_box1 = 10 →
  total_movies_box2 = 5 →
  price_per_movie_box2 = 5 →
  average_price = 3 →
  total_movies = 15 →
  10 * P + 5 * price_per_movie_box2 = average_price * total_movies →
  P = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end price_of_each_movie_in_first_box_l86_86414


namespace num_ways_distribute_balls_l86_86273

-- Definition of the combinatorial problem
def indistinguishableBallsIntoBoxes : ℕ := 11

-- Main theorem statement
theorem num_ways_distribute_balls : indistinguishableBallsIntoBoxes = 11 := by
  sorry

end num_ways_distribute_balls_l86_86273


namespace number_of_diagonals_in_decagon_l86_86143

-- Definition of the problem condition: a polygon with n = 10 sides
def n : ℕ := 10

-- Theorem stating the number of diagonals in a regular decagon
theorem number_of_diagonals_in_decagon : (n * (n - 3)) / 2 = 35 :=
by
  -- Proof steps will go here
  sorry

end number_of_diagonals_in_decagon_l86_86143


namespace xyz_sum_48_l86_86628

theorem xyz_sum_48 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : z * x + y = 47) : 
  x + y + z = 48 :=
sorry

end xyz_sum_48_l86_86628


namespace triangle_internal_angle_A_l86_86475

theorem triangle_internal_angle_A {B C A : ℝ} (hB : Real.tan B = -2) (hC : Real.tan C = 1 / 3) (h_sum: A = π - B - C) : A = π / 4 :=
by
  sorry

end triangle_internal_angle_A_l86_86475


namespace contradiction_proof_l86_86198

theorem contradiction_proof (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by sorry

end contradiction_proof_l86_86198


namespace total_number_of_fleas_l86_86903

theorem total_number_of_fleas :
  let G_fleas := 10
  let O_fleas := G_fleas / 2
  let M_fleas := 5 * O_fleas
  G_fleas + O_fleas + M_fleas = 40 := rfl

end total_number_of_fleas_l86_86903


namespace royalty_amount_l86_86693

theorem royalty_amount (x : ℝ) (h1 : x > 800) (h2 : x ≤ 4000) (h3 : (x - 800) * 0.14 = 420) :
  x = 3800 :=
by
  sorry

end royalty_amount_l86_86693


namespace total_revenue_correct_l86_86380

-- Define the conditions
def charge_per_slice : ℕ := 5
def slices_per_pie : ℕ := 4
def pies_sold : ℕ := 9

-- Prove the question: total revenue
theorem total_revenue_correct : charge_per_slice * slices_per_pie * pies_sold = 180 :=
by
  sorry

end total_revenue_correct_l86_86380


namespace sodium_bicarbonate_moles_needed_l86_86761

-- Definitions for the problem.
def balanced_reaction : Prop := 
  ∀ (NaHCO₃ HCl NaCl H₂O CO₂ : Type) (moles_NaHCO₃ moles_HCl moles_NaCl moles_H₂O moles_CO₂ : Nat),
  (moles_NaHCO₃ = moles_HCl) → 
  (moles_NaCl = moles_HCl) → 
  (moles_H₂O = moles_HCl) → 
  (moles_CO₂ = moles_HCl)

-- Given condition: 3 moles of HCl
def moles_HCl : Nat := 3

-- The theorem statement
theorem sodium_bicarbonate_moles_needed : 
  balanced_reaction → moles_HCl = 3 → ∃ moles_NaHCO₃, moles_NaHCO₃ = 3 :=
by 
  -- Proof will be provided here.
  sorry

end sodium_bicarbonate_moles_needed_l86_86761


namespace sum_ab_eq_negative_two_l86_86438

def f (x : ℝ) := x^3 + 3 * x^2 + 6 * x + 4

theorem sum_ab_eq_negative_two (a b : ℝ) (h1 : f a = 14) (h2 : f b = -14) : a + b = -2 := 
by 
  sorry

end sum_ab_eq_negative_two_l86_86438


namespace probability_of_genuine_after_defective_first_draw_l86_86051

-- Definitions used in conditions
def total_products : ℕ := 7
def genuine_products : ℕ := 4
def defective_products : ℕ := 3
def remaining_products_after_first_draw : ℕ := 6
def remaining_genuine_products_after_first_draw : ℕ := 4

-- Main statement
theorem probability_of_genuine_after_defective_first_draw :
  (remaining_genuine_products_after_first_draw / remaining_products_after_first_draw : ℚ) = 2 / 3 := 
by
  sorry

end probability_of_genuine_after_defective_first_draw_l86_86051


namespace thomas_total_blocks_l86_86824

def stack1 := 7
def stack2 := stack1 + 3
def stack3 := stack2 - 6
def stack4 := stack3 + 10
def stack5 := stack2 * 2

theorem thomas_total_blocks : stack1 + stack2 + stack3 + stack4 + stack5 = 55 := by
  sorry

end thomas_total_blocks_l86_86824


namespace spring_length_increase_l86_86869

-- Define the weight (x) and length (y) data points
def weights : List ℝ := [0, 1, 2, 3, 4, 5]
def lengths : List ℝ := [20, 20.5, 21, 21.5, 22, 22.5]

-- Prove that for each increase of 1 kg in weight, the length of the spring increases by 0.5 cm
theorem spring_length_increase (h : weights.length = lengths.length) :
  ∀ i, i < weights.length - 1 → (lengths.get! (i+1) - lengths.get! i) = 0.5 :=
by
  -- Proof goes here, omitted for now
  sorry

end spring_length_increase_l86_86869


namespace chandler_saves_weeks_l86_86406

theorem chandler_saves_weeks 
  (cost_of_bike : ℕ) 
  (grandparents_money : ℕ) 
  (aunt_money : ℕ) 
  (cousin_money : ℕ) 
  (weekly_earnings : ℕ)
  (total_birthday_money : ℕ := grandparents_money + aunt_money + cousin_money) 
  (total_money : ℕ := total_birthday_money + weekly_earnings * 24):
  (cost_of_bike = 600) → 
  (grandparents_money = 60) → 
  (aunt_money = 40) → 
  (cousin_money = 20) → 
  (weekly_earnings = 20) → 
  (total_money = cost_of_bike) → 
  24 = ((cost_of_bike - total_birthday_money) / weekly_earnings) := 
by 
  intros; 
  sorry

end chandler_saves_weeks_l86_86406


namespace division_value_l86_86221

theorem division_value (x : ℝ) (h : 1376 / x - 160 = 12) : x = 8 := 
by sorry

end division_value_l86_86221


namespace julia_birth_year_l86_86637

open Nat

theorem julia_birth_year (w_age : ℕ) (p_diff : ℕ) (j_diff : ℕ) (current_year : ℕ) :
  w_age = 37 →
  p_diff = 3 →
  j_diff = 2 →
  current_year = 2021 →
  (current_year - w_age) - p_diff - j_diff = 1979 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end julia_birth_year_l86_86637


namespace minimum_m_n_sum_l86_86029

theorem minimum_m_n_sum:
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ 90 * m = n ^ 3 ∧ m + n = 330 :=
sorry

end minimum_m_n_sum_l86_86029


namespace solids_with_triangular_front_view_l86_86935

-- Definitions based on given conditions
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

def can_have_triangular_front_view : Solid → Prop
  | Solid.TriangularPyramid => true
  | Solid.SquarePyramid => true
  | Solid.TriangularPrism => true
  | Solid.SquarePrism => false
  | Solid.Cone => true
  | Solid.Cylinder => false

-- Theorem statement
theorem solids_with_triangular_front_view :
  {s : Solid | can_have_triangular_front_view s} = 
  {Solid.TriangularPyramid, Solid.SquarePyramid, Solid.TriangularPrism, Solid.Cone} :=
by
  sorry

end solids_with_triangular_front_view_l86_86935


namespace sum_of_numbers_on_cards_l86_86504

-- Define the natural numbers condition
variables {a b c d e f g h : ℕ}

-- The theorem statement
theorem sum_of_numbers_on_cards (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_numbers_on_cards_l86_86504


namespace eric_has_more_than_500_paperclips_on_saturday_l86_86890

theorem eric_has_more_than_500_paperclips_on_saturday :
  ∃ k : ℕ, (4 * 3 ^ k > 500) ∧ (∀ m : ℕ, m < k → 4 * 3 ^ m ≤ 500) ∧ ((k + 1) % 7 = 6) :=
by
  sorry

end eric_has_more_than_500_paperclips_on_saturday_l86_86890


namespace cost_of_building_fence_square_plot_l86_86209

-- Definition of conditions
def area_of_square_plot : ℕ := 289
def price_per_foot : ℕ := 60

-- Resulting theorem statement
theorem cost_of_building_fence_square_plot : 
  let side_length := Int.sqrt area_of_square_plot
  let perimeter := 4 * side_length
  let cost := perimeter * price_per_foot
  cost = 4080 := 
by
  -- Placeholder for the actual proof
  sorry

end cost_of_building_fence_square_plot_l86_86209


namespace unique_prime_p_l86_86105

theorem unique_prime_p (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^2 + 2)) : p = 3 := 
by 
  sorry

end unique_prime_p_l86_86105


namespace simplify_expression_l86_86174

variable (y : ℝ)

theorem simplify_expression : (3 * y^4)^2 = 9 * y^8 :=
by 
  sorry

end simplify_expression_l86_86174


namespace find_xyz_l86_86792

variable (x y z : ℝ)
variable (h1 : x = 80 + 0.11 * 80)
variable (h2 : y = 120 - 0.15 * 120)
variable (h3 : z = 0.20 * (0.40 * (x + y)) + 0.40 * (x + y))

theorem find_xyz (hx : x = 88.8) (hy : y = 102) (hz : z = 91.584) : 
  x = 88.8 ∧ y = 102 ∧ z = 91.584 := by
  sorry

end find_xyz_l86_86792


namespace problem_correct_l86_86233

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
def is_nat_lt_10 (n : ℕ) : Prop := n < 10
def not_zero (n : ℕ) : Prop := n ≠ 0

structure Matrix4x4 :=
  (a₀₀ a₀₁ a₀₂ a₀₃ : ℕ)
  (a₁₀ a₁₁ a₁₂ a₁₃ : ℕ)
  (a₂₀ a₂₁ a₂₂ a₂₃ : ℕ)
  (a₃₀ a₃₁ a₃₂ a₃₃ : ℕ)

def valid_matrix (M : Matrix4x4) : Prop :=
  -- Each cell must be a natural number less than 10
  is_nat_lt_10 M.a₀₀ ∧ is_nat_lt_10 M.a₀₁ ∧ is_nat_lt_10 M.a₀₂ ∧ is_nat_lt_10 M.a₀₃ ∧
  is_nat_lt_10 M.a₁₀ ∧ is_nat_lt_10 M.a₁₁ ∧ is_nat_lt_10 M.a₁₂ ∧ is_nat_lt_10 M.a₁₃ ∧
  is_nat_lt_10 M.a₂₀ ∧ is_nat_lt_10 M.a₂₁ ∧ is_nat_lt_10 M.a₂₂ ∧ is_nat_lt_10 M.a₂₃ ∧
  is_nat_lt_10 M.a₃₀ ∧ is_nat_lt_10 M.a₃₁ ∧ is_nat_lt_10 M.a₃₂ ∧ is_nat_lt_10 M.a₃₃ ∧

  -- Cells in the same region must contain the same number
  M.a₀₀ = M.a₁₀ ∧ M.a₀₀ = M.a₂₀ ∧ M.a₀₀ = M.a₃₀ ∧
  M.a₂₀ = M.a₂₁ ∧
  M.a₂₂ = M.a₂₃ ∧ M.a₂₂ = M.a₃₂ ∧ M.a₂₂ = M.a₃₃ ∧
  M.a₀₃ = M.a₁₃ ∧
  
  -- Cells in the leftmost column cannot contain the number 0
  not_zero M.a₀₀ ∧ not_zero M.a₁₀ ∧ not_zero M.a₂₀ ∧ not_zero M.a₃₀ ∧

  -- The four-digit number formed by the first row is 2187
  is_four_digit (M.a₀₀ * 1000 + M.a₀₁ * 100 + M.a₀₂ * 10 + M.a₀₃) ∧ 
  (M.a₀₀ * 1000 + M.a₀₁ * 100 + M.a₀₂ * 10 + M.a₀₃ = 2187) ∧
  
  -- The four-digit number formed by the second row is 7387
  is_four_digit (M.a₁₀ * 1000 + M.a₁₁ * 100 + M.a₁₂ * 10 + M.a₁₃) ∧ 
  (M.a₁₀ * 1000 + M.a₁₁ * 100 + M.a₁₂ * 10 + M.a₁₃ = 7387) ∧
  
  -- The four-digit number formed by the third row is 7744
  is_four_digit (M.a₂₀ * 1000 + M.a₂₁ * 100 + M.a₂₂ * 10 + M.a₂₃) ∧ 
  (M.a₂₀ * 1000 + M.a₂₁ * 100 + M.a₂₂ * 10 + M.a₂₃ = 7744) ∧
  
  -- The four-digit number formed by the fourth row is 7844
  is_four_digit (M.a₃₀ * 1000 + M.a₃₁ * 100 + M.a₃₂ * 10 + M.a₃₃) ∧ 
  (M.a₃₀ * 1000 + M.a₃₁ * 100 + M.a₃₂ * 10 + M.a₃₃ = 7844)

noncomputable def problem_solution : Matrix4x4 :=
{ a₀₀ := 2, a₀₁ := 1, a₀₂ := 8, a₀₃ := 7,
  a₁₀ := 7, a₁₁ := 3, a₁₂ := 8, a₁₃ := 7,
  a₂₀ := 7, a₂₁ := 7, a₂₂ := 4, a₂₃ := 4,
  a₃₀ := 7, a₃₁ := 8, a₃₂ := 4, a₃₃ := 4 }

theorem problem_correct : valid_matrix problem_solution :=
by
  -- The proof would go here to show that problem_solution meets valid_matrix
  sorry

end problem_correct_l86_86233


namespace cubic_root_conditions_l86_86245

-- Define the cubic polynomial
def cubic (a b : ℝ) (x : ℝ) : ℝ := x^3 + a * x + b

-- Define a predicate for the cubic equation having exactly one real root
def has_one_real_root (a b : ℝ) : Prop :=
  ∀ y : ℝ, cubic a b y = 0 → ∃! x : ℝ, cubic a b x = 0

-- Theorem statement
theorem cubic_root_conditions (a b : ℝ) :
  (a = -3 ∧ b = -3) ∨ (a = -3 ∧ b > 2) ∨ (a = 0 ∧ b = 2) → has_one_real_root a b :=
sorry

end cubic_root_conditions_l86_86245


namespace hats_in_shipment_l86_86855

theorem hats_in_shipment (H : ℝ) (h_condition : 0.75 * H = 90) : H = 120 :=
sorry

end hats_in_shipment_l86_86855


namespace find_square_value_l86_86467

variable (a b : ℝ)
variable (square : ℝ)

-- Conditions: Given the equation square * 3 * a = -3 * a^2 * b
axiom condition : square * 3 * a = -3 * a^2 * b

-- Theorem: Prove that square = -a * b
theorem find_square_value (a b : ℝ) (square : ℝ) (h : square * 3 * a = -3 * a^2 * b) : 
    square = -a * b :=
by
  exact sorry

end find_square_value_l86_86467


namespace reciprocal_of_negative_2023_l86_86980

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l86_86980


namespace annika_return_time_l86_86398

-- Define the rate at which Annika hikes.
def hiking_rate := 10 -- minutes per kilometer

-- Define the distances mentioned in the problem.
def initial_distance_east := 2.5 -- kilometers
def total_distance_east := 3.5 -- kilometers

-- Define the time calculations.
def additional_distance_east := total_distance_east - initial_distance_east

-- Calculate the total time required for Annika to get back to the start.
theorem annika_return_time (rate : ℝ) (initial_dist : ℝ) (total_dist : ℝ) (additional_dist : ℝ) : 
  initial_dist = 2.5 → total_dist = 3.5 → rate = 10 → additional_dist = total_dist - initial_dist → 
  (2.5 * rate + additional_dist * rate * 2) = 45 :=
by
-- Since this is just the statement and no proof is needed, we use sorry
sorry

end annika_return_time_l86_86398


namespace quadratic_roots_l86_86052

theorem quadratic_roots {x y : ℝ} (h1 : x + y = 8) (h2 : |x - y| = 10) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (x^2 - 8*x - 9 = 0) ∧ (y^2 - 8*y - 9 = 0) :=
by
  sorry

end quadratic_roots_l86_86052


namespace exists_nat_with_digit_sum_l86_86888

-- Definitions of the necessary functions
def digit_sum (n : ℕ) : ℕ := sorry -- Assume this is the sum of the digits of n

theorem exists_nat_with_digit_sum :
  ∃ n : ℕ, digit_sum n = 1000 ∧ digit_sum (n^2) = 1000000 :=
by
  sorry

end exists_nat_with_digit_sum_l86_86888


namespace animals_total_l86_86234

-- Given definitions and conditions
def ducks : ℕ := 25
def rabbits : ℕ := 8
def chickens := 4 * ducks

-- Proof statement
theorem animals_total (h1 : chickens = 4 * ducks)
                     (h2 : ducks - 17 = rabbits)
                     (h3 : rabbits = 8) :
  chickens + ducks + rabbits = 133 := by
  sorry

end animals_total_l86_86234


namespace cube_inverse_sum_l86_86279

theorem cube_inverse_sum (x : ℂ) (h : x + 1/x = -3) : x^3 + (1/x)^3 = -18 :=
by
  sorry

end cube_inverse_sum_l86_86279


namespace max_abs_value_l86_86252

theorem max_abs_value (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) : |x - 2 * y + 1| ≤ 5 :=
by
  sorry

end max_abs_value_l86_86252


namespace radius_of_circle_l86_86392

open Complex

theorem radius_of_circle (z : ℂ) (h : (z + 2)^4 = 16 * z^4) : abs z = 2 / Real.sqrt 3 :=
sorry

end radius_of_circle_l86_86392


namespace area_of_circle_portion_l86_86056

theorem area_of_circle_portion :
  (∀ x y : ℝ, (x^2 + 6 * x + y^2 = 50) → y ≤ x - 3 → y ≤ 0 → (y^2 + (x + 3)^2 ≤ 59)) →
  (∃ area : ℝ, area = (59 * Real.pi / 4)) :=
by
  sorry

end area_of_circle_portion_l86_86056


namespace average_prime_numbers_l86_86400

-- Definitions of the visible numbers.
def visible1 : ℕ := 51
def visible2 : ℕ := 72
def visible3 : ℕ := 43

-- Definitions of the hidden numbers as prime numbers.
def hidden1 : ℕ := 2
def hidden2 : ℕ := 23
def hidden3 : ℕ := 31

-- Common sum of the numbers on each card.
def common_sum : ℕ := 74

-- Establishing the conditions given in the problem.
def condition1 : hidden1 + visible2 = common_sum := by sorry
def condition2 : hidden2 + visible1 = common_sum := by sorry
def condition3 : hidden3 + visible3 = common_sum := by sorry

-- Calculate the average of the hidden prime numbers.
def average_hidden_primes : ℚ := (hidden1 + hidden2 + hidden3) / 3

-- The proof statement that the average of the hidden prime numbers is 56/3.
theorem average_prime_numbers : average_hidden_primes = 56 / 3 := by
  sorry

end average_prime_numbers_l86_86400


namespace value_of_a_plus_b_l86_86611

-- Define the main problem conditions
variables (a b : ℝ)

-- State the problem in Lean
theorem value_of_a_plus_b (h1 : |a| = 2) (h2 : |b| = 3) (h3 : |a - b| = - (a - b)) :
  a + b = 5 ∨ a + b = 1 :=
sorry

end value_of_a_plus_b_l86_86611


namespace surface_area_of_sphere_l86_86281

theorem surface_area_of_sphere (a : Real) (h : a = 2 * Real.sqrt 3) : 
  (4 * Real.pi * ((Real.sqrt 3 * a / 2) ^ 2)) = 36 * Real.pi :=
by
  sorry

end surface_area_of_sphere_l86_86281


namespace books_finished_l86_86666

theorem books_finished (miles_traveled : ℕ) (miles_per_book : ℕ) (h_travel : miles_traveled = 6760) (h_rate : miles_per_book = 450) : (miles_traveled / miles_per_book) = 15 :=
by {
  -- Proof will be inserted here
  sorry
}

end books_finished_l86_86666


namespace coloring_points_l86_86284

theorem coloring_points
  (A : ℤ × ℤ) (B : ℤ × ℤ) (C : ℤ × ℤ)
  (hA : A.fst % 2 = 1 ∧ A.snd % 2 = 1)
  (hB : (B.fst % 2 = 1 ∧ B.snd % 2 = 0) ∨ (B.fst % 2 = 0 ∧ B.snd % 2 = 1))
  (hC : C.fst % 2 = 0 ∧ C.snd % 2 = 0) :
  ∃ D : ℤ × ℤ,
    (D.fst % 2 = 1 ∧ D.snd % 2 = 0) ∨ (D.fst % 2 = 0 ∧ D.snd % 2 = 1) ∧
    (A.fst + C.fst = B.fst + D.fst) ∧
    (A.snd + C.snd = B.snd + D.snd) := 
sorry

end coloring_points_l86_86284


namespace crayons_left_l86_86800

theorem crayons_left (initial_crayons erasers_left more_crayons_than_erasers : ℕ)
    (H1 : initial_crayons = 531)
    (H2 : erasers_left = 38)
    (H3 : more_crayons_than_erasers = 353) :
    (initial_crayons - (initial_crayons - (erasers_left + more_crayons_than_erasers)) = 391) :=
by 
  sorry

end crayons_left_l86_86800


namespace constant_abs_difference_l86_86295

variable (a : ℕ → ℝ)

-- Define the condition for the recurrence relation
def recurrence_relation : Prop := ∀ n ≥ 1, a (n + 2) = a (n + 1) + a n

-- State the theorem
theorem constant_abs_difference (h : recurrence_relation a) : ∃ C : ℝ, ∀ n ≥ 2, |(a n)^2 - (a (n-1)) * (a (n+1))| = C :=
    sorry

end constant_abs_difference_l86_86295


namespace chemical_transport_problem_l86_86777

theorem chemical_transport_problem :
  (∀ (w r : ℕ), r = w + 420 →
  (900 / r) = (600 / (10 * w)) →
  w = 30 ∧ r = 450) ∧ 
  (∀ (x : ℕ), x + 450 * 3 * 2 + 60 * x ≥ 3600 → x = 15) := by
  sorry

end chemical_transport_problem_l86_86777


namespace problem_solution_l86_86297

theorem problem_solution
  {a b c d : ℝ}
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h2 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2011)
  (h3 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2011) :
  (c * d)^2012 - (a * b)^2012 = 2011 :=
by
  sorry

end problem_solution_l86_86297


namespace prob_sum_is_18_l86_86927

theorem prob_sum_is_18 : 
  let num_faces := 6
  let num_dice := 4
  let total_outcomes := num_faces ^ num_dice
  ∑ (d1 d2 d3 d4 : ℕ) in finset.Icc 1 num_faces, 
  if d1 + d2 + d3 + d4 = 18 then 1 else 0 = 35 → 
  (35 : ℚ) / total_outcomes = 35 / 648 :=
by
  sorry

end prob_sum_is_18_l86_86927


namespace binomial_constant_term_l86_86866

theorem binomial_constant_term (n : ℕ) (h : n > 0) :
  (∃ r : ℕ, n = 2 * r) ↔ (n = 6) :=
by
  sorry

end binomial_constant_term_l86_86866


namespace cindy_correct_answer_l86_86241

-- Define the conditions given in the problem
def x : ℤ := 272 -- Cindy's miscalculated number

-- The outcome of Cindy's incorrect operation
def cindy_incorrect (x : ℤ) : Prop := (x - 7) = 53 * 5

-- The outcome of Cindy's correct operation
def cindy_correct (x : ℤ) : ℤ := (x - 5) / 7

-- The main theorem to prove
theorem cindy_correct_answer : cindy_incorrect x → cindy_correct x = 38 :=
by
  sorry

end cindy_correct_answer_l86_86241


namespace ratio_longer_to_shorter_side_l86_86581

-- Definitions of the problem
variables (l s : ℝ)
def rect_sheet_fold : Prop :=
  l = Real.sqrt (s^2 + (s^2 / l)^2)

-- The to-be-proved theorem
theorem ratio_longer_to_shorter_side (h : rect_sheet_fold l s) :
  l / s = Real.sqrt ((2 : ℝ) / (Real.sqrt 5 - 1)) :=
sorry

end ratio_longer_to_shorter_side_l86_86581


namespace average_income_of_other_40_customers_l86_86706

/-
Given:
1. The average income of 50 customers is $45,000.
2. The average income of the wealthiest 10 customers is $55,000.

Prove:
1. The average income of the other 40 customers is $42,500.
-/

theorem average_income_of_other_40_customers 
  (avg_income_50 : ℝ)
  (wealthiest_10_avg : ℝ) 
  (total_customers : ℕ)
  (wealthiest_customers : ℕ)
  (remaining_customers : ℕ)
  (h1 : avg_income_50 = 45000)
  (h2 : wealthiest_10_avg = 55000)
  (h3 : total_customers = 50)
  (h4 : wealthiest_customers = 10)
  (h5 : remaining_customers = 40) :
  let total_income_50 := total_customers * avg_income_50
  let total_income_wealthiest_10 := wealthiest_customers * wealthiest_10_avg
  let income_remaining_customers := total_income_50 - total_income_wealthiest_10
  let avg_income_remaining := income_remaining_customers / remaining_customers
  avg_income_remaining = 42500 := 
sorry

end average_income_of_other_40_customers_l86_86706


namespace sum_of_digits_divisible_by_45_l86_86486

theorem sum_of_digits_divisible_by_45 (a b : ℕ) (h1 : b = 0 ∨ b = 5) (h2 : (21 + a + b) % 9 = 0) : a + b = 6 :=
by
  sorry

end sum_of_digits_divisible_by_45_l86_86486


namespace abs_sub_lt_five_solution_set_l86_86555

theorem abs_sub_lt_five_solution_set (x : ℝ) : |x - 3| < 5 ↔ -2 < x ∧ x < 8 :=
by sorry

end abs_sub_lt_five_solution_set_l86_86555


namespace ratio_triangle_BFD_to_square_ABCE_l86_86005

-- Defining necessary components for the mathematical problem
def square_ABCE (x : ℝ) : ℝ := 16 * x^2
def triangle_BFD_area (x : ℝ) : ℝ := 7 * x^2

-- The theorem that needs to be proven, stating the ratio of the areas
theorem ratio_triangle_BFD_to_square_ABCE (x : ℝ) (hx : x > 0) :
  (triangle_BFD_area x) / (square_ABCE x) = 7 / 16 :=
by
  sorry

end ratio_triangle_BFD_to_square_ABCE_l86_86005


namespace area_of_equilateral_triangle_example_l86_86798

noncomputable def area_of_equilateral_triangle_with_internal_point (a b c : ℝ) (d_pa : ℝ) (d_pb : ℝ) (d_pc : ℝ) : ℝ :=
  if h : ((d_pa = 3) ∧ (d_pb = 4) ∧ (d_pc = 5)) then
    (9 + (25 * Real.sqrt 3)/4)
  else
    0

theorem area_of_equilateral_triangle_example :
  area_of_equilateral_triangle_with_internal_point 3 4 5 3 4 5 = 9 + (25 * Real.sqrt 3)/4 :=
  by sorry

end area_of_equilateral_triangle_example_l86_86798


namespace volume_of_new_pyramid_is_108_l86_86582

noncomputable def volume_of_cut_pyramid : ℝ :=
  let base_edge_length := 12 * Real.sqrt 2
  let slant_edge_length := 15
  let cut_height := 4.5
  -- Calculate the height of the original pyramid using Pythagorean theorem
  let original_height := Real.sqrt (slant_edge_length^2 - (base_edge_length/2 * Real.sqrt 2)^2)
  -- Calculate the remaining height of the smaller pyramid
  let remaining_height := original_height - cut_height
  -- Calculate the scale factor
  let scale_factor := remaining_height / original_height
  -- New base edge length
  let new_base_edge_length := base_edge_length * scale_factor
  -- New base area
  let new_base_area := (new_base_edge_length)^2
  -- Volume of the new pyramid
  (1 / 3) * new_base_area * remaining_height

-- Define the statement to prove
theorem volume_of_new_pyramid_is_108 :
  volume_of_cut_pyramid = 108 :=
by
  sorry

end volume_of_new_pyramid_is_108_l86_86582


namespace reciprocal_of_neg_2023_l86_86987

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86987


namespace expected_number_of_matches_variance_of_number_of_matches_l86_86845

-- Definitions of conditions
def num_pairs (N : ℕ) : Type := Fin N -> bool -- Type representing pairs of cards matching or not for an N-pair scenario

def indicator_variable (N : ℕ) (k : Fin N) : num_pairs N -> Prop :=
  λ (pairs : num_pairs N), pairs k

def matching_probability (N : ℕ) : ℝ :=
  1 / (N : ℝ)

-- Statement of the first proof problem
theorem expected_number_of_matches (N : ℕ) (pairs : num_pairs N) : 
  (∑ k, (if indicator_variable N k pairs then 1 else 0)) / N = 1 :=
sorry

-- Statement of the second proof problem
theorem variance_of_number_of_matches (N : ℕ) (pairs : num_pairs N) :
  (∑ k, (if indicator_variable N k pairs then 1 else 0) * (if indicator_variable N k pairs then 1 else 0) + 
  2 * ∑ i j, if i ≠ j then 
  (if indicator_variable N i pairs then 1 else 0) * (if indicator_variable N j pairs then 1 else 0) else 0) - 
  ((∑ k, (if indicator_variable N k pairs then 1 else 0)) / N) ^ 2 = 1 :=
sorry

end expected_number_of_matches_variance_of_number_of_matches_l86_86845


namespace average_speed_l86_86368

theorem average_speed (v1 v2 : ℝ) (h1 : v1 = 110) (h2 : v2 = 88) : 
  (2 * v1 * v2) / (v1 + v2) = 97.78 := 
by sorry

end average_speed_l86_86368


namespace birds_in_tree_l86_86837

def initialBirds : Nat := 14
def additionalBirds : Nat := 21
def totalBirds := initialBirds + additionalBirds

theorem birds_in_tree : totalBirds = 35 := by
  sorry

end birds_in_tree_l86_86837


namespace simplify_expression_l86_86173

variable (y : ℝ)

theorem simplify_expression : (3 * y^4)^2 = 9 * y^8 :=
by 
  sorry

end simplify_expression_l86_86173


namespace bob_friends_l86_86873

-- Define the total price and the amount paid by each person
def total_price : ℕ := 40
def amount_per_person : ℕ := 8

-- Define the total number of people who paid
def total_people : ℕ := total_price / amount_per_person

-- Define Bob's presence and require proving the number of his friends
theorem bob_friends (total_price amount_per_person total_people : ℕ) (h1 : total_price = 40)
  (h2 : amount_per_person = 8) (h3 : total_people = total_price / amount_per_person) : 
  total_people - 1 = 4 :=
by
  sorry

end bob_friends_l86_86873


namespace smallest_prime_less_than_square_l86_86356

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l86_86356


namespace volume_of_tetrahedron_ABCD_l86_86827

noncomputable section

open Matrix

def volume_of_tetrahedron (a b c d : ℝ^3) : ℝ :=
  (1 / 6) * abs (det.matrix ![![b - a, c - a, d - a]])

theorem volume_of_tetrahedron_ABCD :
  ∀ (A B C D : ℝ^3), 
    dist A B = 4 ∧
    dist A C = 5 ∧
    dist A D = 6 ∧
    dist B C = 2 * Real.sqrt 7 ∧
    dist B D = 5 ∧
    dist C D = Real.sqrt 34 → 
    volume_of_tetrahedron A B C D = 6 * Real.sqrt 1301 :=
by
  intros A B C D h,
  sorry

end volume_of_tetrahedron_ABCD_l86_86827


namespace contrapositive_question_l86_86546

theorem contrapositive_question (x : ℝ) :
  (x = 2 → x^2 - 3 * x + 2 = 0) ↔ (x^2 - 3 * x + 2 ≠ 0 → x ≠ 2) := 
sorry

end contrapositive_question_l86_86546


namespace cricketer_average_score_l86_86317

theorem cricketer_average_score
  (avg1 : ℕ)
  (matches1 : ℕ)
  (avg2 : ℕ)
  (matches2 : ℕ)
  (total_matches : ℕ)
  (total_avg : ℕ)
  (h1 : avg1 = 20)
  (h2 : matches1 = 2)
  (h3 : avg2 = 30)
  (h4 : matches2 = 3)
  (h5 : total_matches = 5)
  (h6 : total_avg = 26)
  (h_total_runs : total_avg * total_matches = avg1 * matches1 + avg2 * matches2) :
  total_avg = 26 := 
sorry

end cricketer_average_score_l86_86317


namespace count_perfect_squares_50_to_200_l86_86458

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem count_perfect_squares_50_to_200 :
  {n : ℕ | 50 < n ∧ n < 200 ∧ is_perfect_square n}.to_finset.card = 7 :=
by
  sorry

end count_perfect_squares_50_to_200_l86_86458


namespace polynomial_operations_l86_86320

-- Define the given options for M, N, and P
def A (x : ℝ) : ℝ := 2 * x - 6
def B (x : ℝ) : ℝ := 3 * x + 5
def C (x : ℝ) : ℝ := -5 * x - 21

-- Define the original expression and its simplified form
def original_expr (M N : ℝ → ℝ) (x : ℝ) : ℝ :=
  2 * M x - 3 * N x

-- Define the simplified target expression
def simplified_expr (x : ℝ) : ℝ := -5 * x - 21

theorem polynomial_operations :
  ∀ (M N P : ℝ → ℝ),
  (original_expr M N = simplified_expr) →
  (M = A ∨ N = B ∨ P = C)
:= by
  intros M N P H
  sorry

end polynomial_operations_l86_86320


namespace min_value_function_l86_86126

theorem min_value_function (x y : ℝ) (h1 : -2 < x ∧ x < 2) (h2 : -2 < y ∧ y < 2) (h3 : x * y = -1) :
  ∃ u : ℝ, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) ∧ u = 12 / 5 :=
by
  sorry

end min_value_function_l86_86126


namespace books_bought_at_bookstore_l86_86018

-- Define the initial count of books
def initial_books : ℕ := 72

-- Define the number of books received each month from the book club
def books_from_club (months : ℕ) : ℕ := months

-- Number of books received as gifts
def books_from_daughter : ℕ := 1
def books_from_mother : ℕ := 4

-- Number of books bought
def books_from_yard_sales : ℕ := 2

-- Number of books donated and sold
def books_donated : ℕ := 12
def books_sold : ℕ := 3

-- Final total count of books
def final_books : ℕ := 81

-- Calculate the number of books acquired and then removed, and prove 
-- the number of books bought at the bookstore halfway through the year
theorem books_bought_at_bookstore (months : ℕ) (b : ℕ) :
  initial_books + books_from_club months + books_from_daughter + books_from_mother + books_from_yard_sales + b - books_donated - books_sold = final_books → b = 5 :=
by sorry

end books_bought_at_bookstore_l86_86018


namespace group_made_l86_86690

-- Definitions based on the problem's conditions
def teachers_made : Nat := 28
def total_products : Nat := 93

-- Theorem to prove that the group made 65 recycled materials
theorem group_made : total_products - teachers_made = 65 := by
  sorry

end group_made_l86_86690


namespace bronchitis_option_D_correct_l86_86286

noncomputable def smoking_related_to_bronchitis : Prop :=
  -- Conclusion that "smoking is related to chronic bronchitis"
sorry

noncomputable def confidence_level : ℝ :=
  -- Confidence level in the conclusion
  0.99

theorem bronchitis_option_D_correct :
  smoking_related_to_bronchitis →
  (confidence_level > 0.99) →
  -- Option D is correct: "Among 100 smokers, it is possible that not a single person has chronic bronchitis"
  ∃ (P : ℕ → Prop), (∀ n : ℕ, n ≤ 100 → P n = False) :=
by sorry

end bronchitis_option_D_correct_l86_86286


namespace pie_shop_revenue_l86_86381

def costPerSlice : Int := 5
def slicesPerPie : Int := 4
def piesSold : Int := 9

theorem pie_shop_revenue : (costPerSlice * slicesPerPie * piesSold) = 180 := 
by
  sorry

end pie_shop_revenue_l86_86381


namespace vertex_in_fourth_quadrant_l86_86473

theorem vertex_in_fourth_quadrant (m : ℝ) (h : m < 0) : 
  (0 < -m) ∧ (-1 < 0) :=
by
  sorry

end vertex_in_fourth_quadrant_l86_86473


namespace sin_225_eq_neg_sqrt2_div_2_l86_86882

noncomputable def sin_225_deg := real.sin (225 * real.pi / 180)

theorem sin_225_eq_neg_sqrt2_div_2 : sin_225_deg = -real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt2_div_2_l86_86882


namespace scientific_notation_of_1_300_000_l86_86722

-- Define the condition: 1.3 million equals 1,300,000
def one_point_three_million : ℝ := 1300000

-- The theorem statement for the question
theorem scientific_notation_of_1_300_000 :
  one_point_three_million = 1.3 * 10^6 :=
sorry

end scientific_notation_of_1_300_000_l86_86722


namespace expected_matches_is_one_variance_matches_is_one_l86_86841

noncomputable def indicator (k : ℕ) (matches : Finset ℕ) : ℕ :=
  if k ∈ matches then 1 else 0

def expected_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  (Finset.range N).sum (λ k, indicator k matches / N)

def variance_matches (N : ℕ) (matches : Finset ℕ) : ℝ :=
  let E_S := expected_matches N matches in
  let E_S2 := (Finset.range N).sum (λ k, (indicator k matches) ^ 2 / N) in
  E_S2 - E_S ^ 2

theorem expected_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  expected_matches N matches = 1 := sorry

theorem variance_matches_is_one (N : ℕ) (matches : Finset ℕ) :
  variance_matches N matches = 1 := sorry

end expected_matches_is_one_variance_matches_is_one_l86_86841


namespace smallest_prime_less_than_perfect_square_l86_86359

theorem smallest_prime_less_than_perfect_square : ∃ (n : ℕ), ∃ (k : ℤ), n = (k^2 - 12 : ℤ) ∧ nat.prime n ∧ n > 0 ∧ ∀ (m : ℕ), (∃ (j : ℤ), m = (j^2 - 12 : ℤ) ∧ nat.prime m ∧ m > 0) → n ≤ m :=
begin
  sorry
end

end smallest_prime_less_than_perfect_square_l86_86359


namespace solve_for_y_l86_86674

theorem solve_for_y (y : ℚ) (h : (4 / 7) * (1 / 5) * y - 2 = 10) : y = 105 := by
  sorry

end solve_for_y_l86_86674


namespace range_of_a_l86_86140

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) := 
by
  sorry

end range_of_a_l86_86140


namespace sin_225_eq_neg_sqrt2_div_2_l86_86883

noncomputable def sin_225_deg := real.sin (225 * real.pi / 180)

theorem sin_225_eq_neg_sqrt2_div_2 : sin_225_deg = -real.sqrt 2 / 2 :=
by
  sorry

end sin_225_eq_neg_sqrt2_div_2_l86_86883


namespace discount_is_twelve_l86_86036

def markedPrice := Real
def costPrice (MP : markedPrice) : Real := 0.64 * MP
def gainPercent : Real := 37.5

def discountPercentage (MP : markedPrice) : Real :=
  let CP := costPrice MP
  let gain := gainPercent / 100 * CP
  let SP := CP + gain
  ((MP - SP) / MP) * 100

theorem discount_is_twelve (MP : markedPrice) : discountPercentage MP = 12 :=
by
  sorry

end discount_is_twelve_l86_86036


namespace find_value_of_x_l86_86213

theorem find_value_of_x :
  (0.47 * 1442 - 0.36 * 1412) + 65 = 234.42 := 
by
  sorry

end find_value_of_x_l86_86213


namespace B_subset_A_A_inter_B_empty_l86_86122

-- Definitions for the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}

-- Proof statement for Part (1)
theorem B_subset_A (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ (-1 / 2 < a ∧ a < 1) := sorry

-- Proof statement for Part (2)
theorem A_inter_B_empty (a : ℝ) : (∀ x, ¬(x ∈ A ∧ x ∈ B a)) ↔ (a ≤ -4 ∨ a ≥ 2) := sorry

end B_subset_A_A_inter_B_empty_l86_86122


namespace people_in_group_10_l86_86545

-- Let n represent the number of people in the group.
def number_of_people_in_group (n : ℕ) : Prop :=
  let average_increase : ℚ := 3.2
  let weight_of_replaced_person : ℚ := 65
  let weight_of_new_person : ℚ := 97
  let weight_increase : ℚ := weight_of_new_person - weight_of_replaced_person
  weight_increase = average_increase * n

theorem people_in_group_10 :
  ∃ n : ℕ, number_of_people_in_group n ∧ n = 10 :=
by
  sorry

end people_in_group_10_l86_86545


namespace total_crayons_in_drawer_l86_86193

-- Definitions of conditions from a)
def initial_crayons : Nat := 9
def additional_crayons : Nat := 3

-- Statement to prove that total crayons in the drawer is 12
theorem total_crayons_in_drawer : initial_crayons + additional_crayons = 12 := sorry

end total_crayons_in_drawer_l86_86193


namespace largest_prime_divisor_of_sum_of_cyclic_sequence_is_101_l86_86223

-- Define the sequence and its cyclic property
def cyclicSequence (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq (n + 4) = 1000 * (seq n % 10) + 100 * (seq (n + 1) % 10) + 10 * (seq (n + 2) % 10) + (seq (n + 3) % 10)

-- Define the property of T being the sum of the sequence
def sumOfSequence (seq : ℕ → ℕ) (T : ℕ) : Prop :=
  T = seq 0 + seq 1 + seq 2 + seq 3

-- Define the statement that T is always divisible by 101
theorem largest_prime_divisor_of_sum_of_cyclic_sequence_is_101
  (seq : ℕ → ℕ) (T : ℕ)
  (h1 : cyclicSequence seq)
  (h2 : sumOfSequence seq T) :
  (101 ∣ T) := 
sorry

end largest_prime_divisor_of_sum_of_cyclic_sequence_is_101_l86_86223


namespace liu_xing_statement_incorrect_l86_86480

-- Definitions of the initial statistics of the classes
def avg_score_class_91 : ℝ := 79.5
def avg_score_class_92 : ℝ := 80.2

-- Definitions of corrections applied
def correction_gain_class_91 : ℝ := 0.6 * 3
def correction_loss_class_91 : ℝ := 0.2 * 3
def correction_gain_class_92 : ℝ := 0.5 * 3
def correction_loss_class_92 : ℝ := 0.3 * 3

-- Definitions of corrected averages
def corrected_avg_class_91 : ℝ := avg_score_class_91 + correction_gain_class_91 - correction_loss_class_91
def corrected_avg_class_92 : ℝ := avg_score_class_92 + correction_gain_class_92 - correction_loss_class_92

-- Proof statement
theorem liu_xing_statement_incorrect : corrected_avg_class_91 ≤ corrected_avg_class_92 :=
by {
  -- Additional hints and preliminary calculations could be done here.
  sorry
}

end liu_xing_statement_incorrect_l86_86480


namespace leah_daily_savings_l86_86653

theorem leah_daily_savings 
  (L : ℝ)
  (h1 : 0.25 * 24 = 6)
  (h2 : ∀ (L : ℝ), (L * 20) = 20 * L)
  (h3 : ∀ (L : ℝ), 2 * L * 12 = 24 * L)
  (h4 :  6 + 20 * L + 24 * L = 28) 
: L = 0.5 :=
by
  sorry

end leah_daily_savings_l86_86653


namespace Seokjin_paper_count_l86_86945

theorem Seokjin_paper_count (Jimin_paper : ℕ) (h1 : Jimin_paper = 41) (h2 : ∀ x : ℕ, Seokjin_paper = Jimin_paper - 1) : Seokjin_paper = 40 :=
by {
  sorry
}

end Seokjin_paper_count_l86_86945


namespace parabola_x_intercepts_l86_86623

theorem parabola_x_intercepts :
  ∃! y : ℝ, -3 * y^2 + 2 * y + 4 = y := 
by
  sorry

end parabola_x_intercepts_l86_86623


namespace three_digit_ends_in_5_divisible_by_5_l86_86076

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_in_5 (n : ℕ) : Prop := n % 10 = 5

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem three_digit_ends_in_5_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit N) 
  (h2 : ends_in_5 N) : is_divisible_by_5 N := 
sorry

end three_digit_ends_in_5_divisible_by_5_l86_86076


namespace second_number_is_40_l86_86814

-- Defining the problem
theorem second_number_is_40
  (a b c : ℚ)
  (h1 : a + b + c = 120)
  (h2 : a = (3/4 : ℚ) * b)
  (h3 : c = (5/4 : ℚ) * b) :
  b = 40 :=
sorry

end second_number_is_40_l86_86814


namespace sum_of_eight_numbers_l86_86529

-- Definitions used in the conditions
variables {a b c d e f g h : ℕ}

-- Given condition
axiom product_condition : (a + b) * (c + d) * (e + f) * (g + h) = 330

-- Define individual sums
def ab_sum := a + b
def cd_sum := c + d
def ef_sum := e + f
def gh_sum := g + h

-- Define the total sum of the eight numbers on the cards
def total_sum := ab_sum + cd_sum + ef_sum + gh_sum

-- The theorem to prove
theorem sum_of_eight_numbers : total_sum = 21 := by
  have ab_sum_eq : ab_sum = 2 := sorry
  have cd_sum_eq : cd_sum = 3 := sorry
  have ef_sum_eq : ef_sum = 5 := sorry
  have gh_sum_eq : gh_sum = 11 := sorry
  rw [ab_sum_eq, cd_sum_eq, ef_sum_eq, gh_sum_eq]
  norm_num

end sum_of_eight_numbers_l86_86529


namespace smallest_n_condition_l86_86596

-- Define the conditions
def condition1 (x : ℤ) : Prop := 2 * x - 3 ≡ 0 [ZMOD 13]
def condition2 (y : ℤ) : Prop := 3 * y + 4 ≡ 0 [ZMOD 13]

-- Problem statement: finding n such that the expression is a multiple of 13
theorem smallest_n_condition (x y : ℤ) (n : ℤ) :
  condition1 x → condition2 y → x^2 - x * y + y^2 + n ≡ 0 [ZMOD 13] → n = 1 := 
by
  sorry

end smallest_n_condition_l86_86596


namespace solve_fractional_equation_l86_86313

theorem solve_fractional_equation : 
  ∃ x : ℝ, (x - 1) / 2 = 1 - (3 * x + 2) / 5 ↔ x = 1 := 
sorry

end solve_fractional_equation_l86_86313


namespace cos_inequality_m_range_l86_86565

theorem cos_inequality_m_range (m : ℝ) : 
  (-1 ≤ 1 - m ∧ 1 - m ≤ 1) ↔ (0 ≤ m ∧ m ≤ 2) :=
sorry

end cos_inequality_m_range_l86_86565


namespace number_of_ways_to_elect_officers_l86_86236

theorem number_of_ways_to_elect_officers (total_candidates past_officers positions : ℕ)
  (total_candidates_eq : total_candidates = 18)
  (past_officers_eq : past_officers = 8)
  (positions_eq : positions = 6) :
  ∃ k : ℕ, k = 16338 := 
by
  have h1 : nat.choose 18 6 = 18564 := by sorry
  have h2 : nat.choose 10 6 = 210 := by sorry
  have h3 : 8 * (nat.choose 10 5) = 2016 := by sorry
  have h4 : nat.choose 18 6 - nat.choose 10 6 - 8 * (nat.choose 10 5) = 16338 := by sorry
  use 16338
  exact h4

end number_of_ways_to_elect_officers_l86_86236


namespace crayons_total_correct_l86_86196

-- Definitions from the conditions
def initial_crayons : ℕ := 9
def added_crayons : ℕ := 3

-- Expected total crayons as per the conditions and the correct answer
def total_crayons_expected : ℕ := 12

-- The proof statement
theorem crayons_total_correct :
  initial_crayons + added_crayons = total_crayons_expected :=
by
  -- Proof details here
  sorry

end crayons_total_correct_l86_86196


namespace xyz_sum_48_l86_86627

theorem xyz_sum_48 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : z * x + y = 47) : 
  x + y + z = 48 :=
sorry

end xyz_sum_48_l86_86627


namespace part1_part2_l86_86791

namespace Problem

open Real

def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2*x - 8 > 0)

theorem part1 (h : p 1 x ∧ q x) : 2 < x ∧ x < 3:= 
sorry

theorem part2 (hpq : ∀ x, ¬ p a x → ¬ q x) : 
   1 < a ∧ a ≤ 2 := 
sorry

end Problem

end part1_part2_l86_86791


namespace imaginary_part_inv_z_l86_86755

def z : ℂ := 1 - 2 * Complex.I

theorem imaginary_part_inv_z : Complex.im (1 / z) = 2 / 5 :=
by
  -- proof to be filled in
  sorry

end imaginary_part_inv_z_l86_86755


namespace Ivanka_more_months_l86_86649

variable (I : ℕ) (W : ℕ)

theorem Ivanka_more_months (hW : W = 18) (hI_W : I + W = 39) : I - W = 3 :=
by
  sorry

end Ivanka_more_months_l86_86649


namespace cost_of_orchestra_seat_l86_86075

-- Define the variables according to the conditions in the problem
def orchestra_ticket_count (y : ℕ) : Prop := (2 * y + 115 = 355)
def total_ticket_cost (x y : ℕ) : Prop := (120 * x + 235 * 8 = 3320)
def balcony_ticket_relation (y : ℕ) : Prop := (y + 115 = 355 - y)

-- Main theorem statement: Prove that the cost of a seat in the orchestra is 12 dollars
theorem cost_of_orchestra_seat : ∃ x y : ℕ, orchestra_ticket_count y ∧ total_ticket_cost x y ∧ (x = 12) :=
by sorry

end cost_of_orchestra_seat_l86_86075


namespace find_theta_l86_86816

theorem find_theta (R h : ℝ) (θ : ℝ) 
  (r1_def : r1 = R * Real.cos θ)
  (r2_def : r2 = (R + h) * Real.cos θ)
  (s_def : s = 2 * π * h * Real.cos θ)
  (s_eq_h : s = h) : 
  θ = Real.arccos (1 / (2 * π)) :=
by
  sorry

end find_theta_l86_86816


namespace not_necessarily_a_squared_lt_b_squared_l86_86134
-- Import the necessary library

-- Define the variables and the condition
variables {a b : ℝ}
axiom h : a < b

-- The theorem statement that needs to be proved/disproved
theorem not_necessarily_a_squared_lt_b_squared (a b : ℝ) (h : a < b) : ¬ (a^2 < b^2) :=
sorry

end not_necessarily_a_squared_lt_b_squared_l86_86134


namespace evaluate_expression_l86_86633

theorem evaluate_expression
  (p q r s : ℚ)
  (h1 : p / q = 4 / 5)
  (h2 : r / s = 3 / 7) :
  (18 / 7) + ((2 * q - p) / (2 * q + p)) - ((3 * s + r) / (3 * s - r)) = 5 / 3 := by
  sorry

end evaluate_expression_l86_86633


namespace exists_valid_circle_group_l86_86726

variable {P : Type}
variable (knows : P → P → Prop)

def knows_at_least_three (p : P) : Prop :=
  ∃ (p₁ p₂ p₃ : P), p₁ ≠ p ∧ p₂ ≠ p ∧ p₃ ≠ p ∧ knows p p₁ ∧ knows p p₂ ∧ knows p p₃

def valid_circle_group (G : List P) : Prop :=
  (2 < G.length) ∧ (G.length % 2 = 0) ∧ (∀ i, knows (G.nthLe i sorry) (G.nthLe ((i + 1) % G.length) sorry) ∧ knows (G.nthLe i sorry) (G.nthLe ((i - 1 + G.length) % G.length) sorry))

theorem exists_valid_circle_group (H : ∀ p : P, knows_at_least_three knows p) : 
  ∃ G : List P, valid_circle_group knows G := 
sorry

end exists_valid_circle_group_l86_86726


namespace ratio_perimeter_to_breadth_l86_86553

-- Definitions of the conditions
def area_of_rectangle (length breadth : ℝ) := length * breadth
def perimeter_of_rectangle (length breadth : ℝ) := 2 * (length + breadth)

-- The problem statement: prove the ratio of perimeter to breadth
theorem ratio_perimeter_to_breadth (L B : ℝ) (hL : L = 18) (hA : area_of_rectangle L B = 216) :
  (perimeter_of_rectangle L B) / B = 5 :=
by 
  -- Given definitions and conditions, we skip the proof.
  sorry

end ratio_perimeter_to_breadth_l86_86553


namespace complete_the_square_l86_86210

theorem complete_the_square (x : ℝ) : x^2 + 2*x - 3 = 0 ↔ (x + 1)^2 = 4 :=
by sorry

end complete_the_square_l86_86210


namespace cricketer_boundaries_l86_86857

theorem cricketer_boundaries (total_runs : ℕ) (sixes : ℕ) (percent_runs_by_running : ℝ)
  (h1 : total_runs = 152)
  (h2 : sixes = 2)
  (h3 : percent_runs_by_running = 60.526315789473685) :
  let runs_by_running := round (total_runs * percent_runs_by_running / 100)
  let runs_from_sixes := sixes * 6
  let runs_from_boundaries := total_runs - runs_by_running - runs_from_sixes
  let boundaries := runs_from_boundaries / 4
  boundaries = 12 :=
by
  sorry

end cricketer_boundaries_l86_86857


namespace sum_of_digits_N_l86_86584

-- Define the main problem conditions and the result statement
theorem sum_of_digits_N {N : ℕ} 
  (h₁ : (N * (N + 1)) / 2 = 5103) : 
  (N.digits 10).sum = 2 :=
sorry

end sum_of_digits_N_l86_86584


namespace height_of_trapezoid_l86_86643

-- Define the condition that a trapezoid has diagonals of given lengths and a given midline.
def trapezoid_conditions (AC BD ML : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : Prop := 
  AC = 6 ∧ BD = 8 ∧ ML = 5

-- Define the height of the trapezoid.
def trapezoid_height (AC BD ML : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : ℝ :=
  4.8

-- The theorem statement
theorem height_of_trapezoid (AC BD ML h : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : 
  trapezoid_conditions AC BD ML h_d1 h_d2 h_ml 
  → trapezoid_height AC BD ML h_d1 h_d2 h_ml = 4.8 := 
by
  intros
  sorry

end height_of_trapezoid_l86_86643


namespace sin_225_eq_neg_sqrt2_over_2_l86_86876

theorem sin_225_eq_neg_sqrt2_over_2 : Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end sin_225_eq_neg_sqrt2_over_2_l86_86876


namespace reciprocal_of_negative_2023_l86_86982

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l86_86982


namespace mod_equivalence_l86_86013

theorem mod_equivalence (a b : ℤ) (d : ℕ) (hd : d ≠ 0) 
  (a' b' : ℕ) (ha' : a % d = a') (hb' : b % d = b') : (a ≡ b [ZMOD d]) ↔ a' = b' := 
sorry

end mod_equivalence_l86_86013


namespace ratio_of_screws_l86_86499

def initial_screws : Nat := 8
def total_required_screws : Nat := 4 * 6
def screws_to_buy : Nat := total_required_screws - initial_screws

theorem ratio_of_screws :
  (screws_to_buy : ℚ) / initial_screws = 2 :=
by
  simp [initial_screws, total_required_screws, screws_to_buy]
  sorry

end ratio_of_screws_l86_86499


namespace horse_saddle_ratio_l86_86218

theorem horse_saddle_ratio (total_cost : ℕ) (saddle_cost : ℕ) (horse_cost : ℕ) 
  (h_total : total_cost = 5000)
  (h_saddle : saddle_cost = 1000)
  (h_sum : horse_cost + saddle_cost = total_cost) : 
  horse_cost / saddle_cost = 4 :=
by sorry

end horse_saddle_ratio_l86_86218


namespace sum_mod_6_l86_86419

theorem sum_mod_6 :
  (60123 + 60124 + 60125 + 60126 + 60127 + 60128 + 60129 + 60130) % 6 = 4 :=
by
  sorry

end sum_mod_6_l86_86419


namespace jellybean_total_count_l86_86852

theorem jellybean_total_count :
  let black := 8
  let green := 2 * black
  let orange := (2 * green) - 5
  let red := orange + 3
  let yellow := black / 2
  let purple := red + 4
  let brown := (green + purple) - 3
  black + green + orange + red + yellow + purple + brown = 166 := by
  -- skipping proof for brevity
  sorry

end jellybean_total_count_l86_86852


namespace lamp_count_and_profit_l86_86071

-- Define the parameters given in the problem
def total_lamps : ℕ := 50
def total_cost : ℕ := 2500
def cost_A : ℕ := 40
def cost_B : ℕ := 65
def marked_A : ℕ := 60
def marked_B : ℕ := 100
def discount_A : ℕ := 10 -- percent
def discount_B : ℕ := 30 -- percent

-- Derived definitions from the solution
def lamps_A : ℕ := 30
def lamps_B : ℕ := 20
def selling_price_A : ℕ := marked_A * (100 - discount_A) / 100
def selling_price_B : ℕ := marked_B * (100 - discount_B) / 100
def profit_A : ℕ := selling_price_A - cost_A
def profit_B : ℕ := selling_price_B - cost_B
def total_profit : ℕ := (profit_A * lamps_A) + (profit_B * lamps_B)

-- Lean statement
theorem lamp_count_and_profit :
  lamps_A + lamps_B = total_lamps ∧
  (cost_A * lamps_A + cost_B * lamps_B) = total_cost ∧
  total_profit = 520 := by
  -- proofs will go here
  sorry

end lamp_count_and_profit_l86_86071


namespace sin_225_eq_neg_sqrt2_over_2_l86_86877

theorem sin_225_eq_neg_sqrt2_over_2 : Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end sin_225_eq_neg_sqrt2_over_2_l86_86877


namespace total_fleas_l86_86905

-- Definitions based on conditions provided
def fleas_Gertrude : Nat := 10
def fleas_Olive : Nat := fleas_Gertrude / 2
def fleas_Maud : Nat := 5 * fleas_Olive

-- Prove the total number of fleas on all three chickens
theorem total_fleas :
  fleas_Gertrude + fleas_Olive + fleas_Maud = 40 :=
by sorry

end total_fleas_l86_86905


namespace tangent_line_eq_l86_86547

noncomputable def f (x : Real) : Real := x^4 - x

def P : Real × Real := (1, 0)

theorem tangent_line_eq :
  let m := 4 * (1 : Real) ^ 3 - 1 in
  let y1 := 0 in
  let x1 := 1 in
  ∀ (x y : Real), y = m * (x - x1) + y1 ↔ 3 * x - y - 3 = 0 :=
by
  intro x y
  sorry

end tangent_line_eq_l86_86547


namespace exists_group_round_table_l86_86725

open Finset Function

variable (P : Finset ℤ) (knows : ℤ → ℤ → Prop)

def has_at_least_three_friends (P : Finset ℤ) (knows : ℤ → ℤ → Prop) : Prop :=
  ∀ p ∈ P, (P.filter (knows p)).card ≥ 3

noncomputable def exists_even_group (P : Finset ℤ) (knows : ℤ → ℤ → Prop) : Prop :=
  ∃ S : Finset ℤ, (S ⊆ P) ∧ (2 < S.card) ∧ (Even S.card) ∧ (∀ p ∈ S, ∀ q ∈ S, Edge_connected p q knows S)

theorem exists_group_round_table (P : Finset ℤ) (knows : ℤ → ℤ → Prop) 
  (h : has_at_least_three_friends P knows) : 
  exists_even_group P knows :=
sorry

end exists_group_round_table_l86_86725


namespace pie_price_l86_86889

theorem pie_price (cakes_sold : ℕ) (cake_price : ℕ) (cakes_total_earnings : ℕ)
                  (pies_sold : ℕ) (total_earnings : ℕ) (price_per_pie : ℕ)
                  (H1 : cakes_sold = 453)
                  (H2 : cake_price = 12)
                  (H3 : pies_sold = 126)
                  (H4 : total_earnings = 6318)
                  (H5 : cakes_total_earnings = cakes_sold * cake_price)
                  (H6 : price_per_pie * pies_sold = total_earnings - cakes_total_earnings) :
    price_per_pie = 7 := by
    sorry

end pie_price_l86_86889


namespace sector_area_l86_86384

theorem sector_area (r : ℝ) (α : ℝ) (h_r : r = 6) (h_α : α = π / 3) : (1 / 2) * (α * r) * r = 6 * π :=
by
  rw [h_r, h_α]
  sorry

end sector_area_l86_86384


namespace angle_parallel_lines_l86_86125

variables {Line : Type} (a b c : Line) (theta : ℝ)
variable (angle_between : Line → Line → ℝ)

def is_parallel (a b : Line) : Prop := sorry

theorem angle_parallel_lines (h_parallel : is_parallel a b) (h_angle : angle_between a c = theta) : angle_between b c = theta := 
sorry

end angle_parallel_lines_l86_86125


namespace problem1_solution_problem2_solution_l86_86024

theorem problem1_solution (x : ℝ): 2 * x^2 + x - 3 = 0 → (x = 1 ∨ x = -3 / 2) :=
by
  intro h
  -- Proof skipped
  sorry

theorem problem2_solution (x : ℝ): (x - 3)^2 = 2 * x * (3 - x) → (x = 3 ∨ x = 1) :=
by
  intro h
  -- Proof skipped
  sorry

end problem1_solution_problem2_solution_l86_86024


namespace train_crossing_time_l86_86867

/-!
## Problem Statement
A train 400 m in length crosses a telegraph post. The speed of the train is 90 km/h. Prove that it takes 16 seconds for the train to cross the telegraph post.
-/

-- Defining the given definitions based on the conditions in a)
def train_length : ℕ := 400
def train_speed_kmh : ℕ := 90
def train_speed_ms : ℚ := 25 -- Converting 90 km/h to 25 m/s

-- Proving the problem statement
theorem train_crossing_time : train_length / train_speed_ms = 16 := 
by
  -- convert conditions and show expected result
  sorry

end train_crossing_time_l86_86867


namespace third_row_number_of_trees_l86_86375

theorem third_row_number_of_trees (n : ℕ) 
  (divisible_by_7 : 84 % 7 = 0) 
  (divisible_by_6 : 84 % 6 = 0) 
  (divisible_by_n : 84 % n = 0) 
  (least_trees : 84 = 84): 
  n = 4 := 
sorry

end third_row_number_of_trees_l86_86375


namespace enclosed_area_correct_l86_86738

noncomputable def enclosedArea : ℝ := ∫ x in (1 / Real.exp 1)..Real.exp 1, 1 / x

theorem enclosed_area_correct : enclosedArea = 2 := by
  sorry

end enclosed_area_correct_l86_86738


namespace thabo_books_l86_86316

variable (P F H : Nat)

theorem thabo_books :
  P > 55 ∧ F = 2 * P ∧ H = 55 ∧ H + P + F = 280 → P - H = 20 :=
by
  sorry

end thabo_books_l86_86316


namespace percentage_ownership_l86_86144

theorem percentage_ownership (total students_cats students_dogs : ℕ) (h1 : total = 500) (h2 : students_cats = 75) (h3 : students_dogs = 125):
  (students_cats / total : ℝ) = 0.15 ∧
  (students_dogs / total : ℝ) = 0.25 :=
by
  sorry

end percentage_ownership_l86_86144


namespace stamp_collection_cost_l86_86068

def cost_brazil_per_stamp : ℝ := 0.08
def cost_peru_per_stamp : ℝ := 0.05
def num_brazil_stamps_60s : ℕ := 7
def num_peru_stamps_60s : ℕ := 4
def num_brazil_stamps_70s : ℕ := 12
def num_peru_stamps_70s : ℕ := 6

theorem stamp_collection_cost :
  num_brazil_stamps_60s * cost_brazil_per_stamp +
  num_peru_stamps_60s * cost_peru_per_stamp +
  num_brazil_stamps_70s * cost_brazil_per_stamp +
  num_peru_stamps_70s * cost_peru_per_stamp =
  2.02 :=
by
  -- Skipping proof steps.
  sorry

end stamp_collection_cost_l86_86068


namespace num_friends_with_exactly_four_gifts_l86_86096

open Finset
open SimpleGraph

-- Consider a set of six friends used to represent vertices of the graph
def friends : Finset (Fin 6) := univ

-- There are 13 exchanges, and each exchange involves mutual gifting between two friends.
-- Represent this scenario using an undirected graph
def exchanges : SimpleGraph (Fin 6) :=
{ adj := λ a b, (a ≠ b),
  sym := λ a b h, h.symm,
  loopless := λ a h, by { cases h } }

-- Assume there are 13 edges in our graph
axiom exchanges_card : (edges exchanges).card = 13

-- Question: prove there are exactly 2 or 4 friends who received exactly 4 gifts.
theorem num_friends_with_exactly_four_gifts :
  ∃ (n : ℕ), n ∈ {2, 4} ∧ (friends.filter (λ f, (exchanges.neighborFinset f).card = 4)).card = n :=
sorry

end num_friends_with_exactly_four_gifts_l86_86096


namespace probability_of_two_same_color_l86_86577

noncomputable def probability_at_least_two_same_color (reds whites blues greens : ℕ) (total_draws : ℕ) : ℚ :=
  have total_marbles := reds + whites + blues + greens
  let total_combinations := Nat.choose total_marbles total_draws
  let two_reds := Nat.choose reds 2 * (total_marbles - 2)
  let two_whites := Nat.choose whites 2 * (total_marbles - 2)
  let two_blues := Nat.choose blues 2 * (total_marbles - 2)
  let two_greens := Nat.choose greens 2 * (total_marbles - 2)
  
  let all_reds := Nat.choose reds 3
  let all_whites := Nat.choose whites 3
  let all_blues := Nat.choose blues 3
  let all_greens := Nat.choose greens 3
  
  let desired_outcomes := two_reds + two_whites + two_blues + two_greens +
                          all_reds + all_whites + all_blues + all_greens
                          
  (desired_outcomes : ℚ) / (total_combinations : ℚ)

theorem probability_of_two_same_color : probability_at_least_two_same_color 6 7 8 4 3 = 69 / 115 := 
by
  sorry

end probability_of_two_same_color_l86_86577


namespace y_intercept_of_line_l86_86896

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) (hx : x = 0) : y = 4 :=
by
  -- The proof goes here
  sorry

end y_intercept_of_line_l86_86896


namespace right_angled_triangle_solution_l86_86117

theorem right_angled_triangle_solution:
  ∃ (a b c : ℕ),
    (a^2 + b^2 = c^2) ∧
    (a + b + c = (a * b) / 2) ∧
    ((a, b, c) = (6, 8, 10) ∨ (a, b, c) = (5, 12, 13)) :=
by
  sorry

end right_angled_triangle_solution_l86_86117


namespace cards_sum_l86_86518

theorem cards_sum (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
by
  sorry

end cards_sum_l86_86518


namespace discount_percentage_l86_86077

theorem discount_percentage (P D : ℝ) 
  (h1 : P > 0)
  (h2 : D = (1 - 0.28000000000000004 / 0.60)) :
  D = 0.5333333333333333 :=
by
  sorry

end discount_percentage_l86_86077


namespace alyssa_puppies_l86_86723

theorem alyssa_puppies (total_puppies : ℕ) (given_away : ℕ) (remaining_puppies : ℕ) 
  (h1 : total_puppies = 7) (h2 : given_away = 5) 
  : remaining_puppies = total_puppies - given_away → remaining_puppies = 2 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end alyssa_puppies_l86_86723


namespace vehicles_with_cd_player_but_no_pw_or_ab_l86_86831

-- Definitions based on conditions from step a)
def P : ℝ := 0.60 -- percentage of vehicles with power windows
def A : ℝ := 0.25 -- percentage of vehicles with anti-lock brakes
def C : ℝ := 0.75 -- percentage of vehicles with a CD player
def PA : ℝ := 0.10 -- percentage of vehicles with both power windows and anti-lock brakes
def AC : ℝ := 0.15 -- percentage of vehicles with both anti-lock brakes and a CD player
def PC : ℝ := 0.22 -- percentage of vehicles with both power windows and a CD player
def PAC : ℝ := 0.00 -- no vehicle has all 3 features

-- The statement we want to prove
theorem vehicles_with_cd_player_but_no_pw_or_ab : C - (PC + AC) = 0.38 := by
  sorry

end vehicles_with_cd_player_but_no_pw_or_ab_l86_86831


namespace solutions_to_equation_l86_86698

theorem solutions_to_equation :
  ∀ x : ℝ, (x + 1) * (x - 2) = x + 1 ↔ x = -1 ∨ x = 3 :=
by
  sorry

end solutions_to_equation_l86_86698


namespace difference_in_square_sides_square_side_length_square_area_greater_than_rectangle_l86_86367

-- Exploration 1
theorem difference_in_square_sides (a b : ℝ) (h1 : a + b = 20) (h2 : a^2 - b^2 = 40) : a - b = 2 :=
by sorry

-- Exploration 2
theorem square_side_length (x y : ℝ) : (2 * x + 2 * y) / 4 = (x + y) / 2 :=
by sorry

theorem square_area_greater_than_rectangle (x y : ℝ) (h : x > y) : ( (x + y) / 2 ) ^ 2 > x * y :=
by sorry

end difference_in_square_sides_square_side_length_square_area_greater_than_rectangle_l86_86367


namespace mason_grandmother_age_l86_86160

-- Defining the ages of Mason, Sydney, Mason's father, and Mason's grandmother
def mason_age : ℕ := 20

def sydney_age (S : ℕ) : Prop :=
  mason_age = S / 3

def father_age (S F : ℕ) : Prop :=
  F = S + 6

def grandmother_age (F G : ℕ) : Prop :=
  G = 2 * F

theorem mason_grandmother_age (S F G : ℕ) (h1 : sydney_age S) (h2 : father_age S F) (h3 : grandmother_age F G) : G = 132 :=
by
  -- leaving the proof as a sorry
  sorry

end mason_grandmother_age_l86_86160


namespace value_of_x_l86_86208

theorem value_of_x (v w z y x : ℤ) 
  (h1 : v = 90)
  (h2 : w = v + 30)
  (h3 : z = w + 21)
  (h4 : y = z + 11)
  (h5 : x = y + 6) : 
  x = 158 :=
by 
  sorry

end value_of_x_l86_86208


namespace johns_profit_is_200_l86_86150

def num_woodburnings : ℕ := 20
def price_per_woodburning : ℕ := 15
def cost_of_wood : ℕ := 100
def total_revenue : ℕ := num_woodburnings * price_per_woodburning
def profit : ℕ := total_revenue - cost_of_wood

theorem johns_profit_is_200 : profit = 200 :=
by
  -- proof steps go here
  sorry

end johns_profit_is_200_l86_86150


namespace seconds_in_9_point_4_minutes_l86_86921

def seconds_in_minute : ℕ := 60
def minutes : ℝ := 9.4
def expected_seconds : ℝ := 564

theorem seconds_in_9_point_4_minutes : minutes * seconds_in_minute = expected_seconds :=
by 
  sorry

end seconds_in_9_point_4_minutes_l86_86921


namespace container_volume_ratio_l86_86237

theorem container_volume_ratio (V1 V2 : ℚ)
  (h1 : (3 / 5) * V1 = (2 / 3) * V2) :
  V1 / V2 = 10 / 9 :=
by sorry

end container_volume_ratio_l86_86237


namespace sequence_conjecture_l86_86919

theorem sequence_conjecture (a : ℕ → ℝ) (h₁ : a 1 = 7)
  (h₂ : ∀ n, a (n + 1) = 7 * a n / (a n + 7)) :
  ∀ n, a n = 7 / n :=
by
  sorry

end sequence_conjecture_l86_86919


namespace cubic_identity_l86_86276

theorem cubic_identity (x : ℝ) (h : x + (1/x) = -3) : x^3 + (1/x^3) = -18 :=
by
  sorry

end cubic_identity_l86_86276


namespace remaining_numbers_l86_86543

theorem remaining_numbers (S S3 S2 N : ℕ) (h1 : S / 5 = 8) (h2 : S3 / 3 = 4) (h3 : S2 / N = 14) 
(hS  : S = 5 * 8) (hS3 : S3 = 3 * 4) (hS2 : S2 = S - S3) : N = 2 := by
  sorry

end remaining_numbers_l86_86543


namespace avg_salary_officers_l86_86773

-- Definitions of the given conditions
def avg_salary_employees := 120
def avg_salary_non_officers := 110
def num_officers := 15
def num_non_officers := 495

-- The statement to be proven
theorem avg_salary_officers : (15 * (15 * X) / (15 + 495)) = 450 :=
by
  sorry

end avg_salary_officers_l86_86773


namespace square_points_sum_of_squares_l86_86534

theorem square_points_sum_of_squares 
  (a b c d : ℝ) 
  (h₀_a : 0 ≤ a ∧ a ≤ 1)
  (h₀_b : 0 ≤ b ∧ b ≤ 1)
  (h₀_c : 0 ≤ c ∧ c ≤ 1)
  (h₀_d : 0 ≤ d ∧ d ≤ 1) 
  :
  2 ≤ a^2 + (1 - d)^2 + b^2 + (1 - a)^2 + c^2 + (1 - b)^2 + d^2 + (1 - c)^2 ∧
  a^2 + (1 - d)^2 + b^2 + (1 - a)^2 + c^2 + (1 - b)^2 + d^2 + (1 - c)^2 ≤ 4 := 
by
  sorry

end square_points_sum_of_squares_l86_86534


namespace measure_of_angle_B_find_a_and_c_find_perimeter_l86_86147

theorem measure_of_angle_B (a b c : ℝ) (A B C : ℝ) 
    (h : c / (b - a) = (Real.sin A + Real.sin B) / (Real.sin A + Real.sin C)) 
    (cos_B : Real.cos B = -1 / 2) : B = 2 * Real.pi / 3 :=
by
  sorry

theorem find_a_and_c (a c A C : ℝ) (S : ℝ) 
    (h1 : Real.sin C = 2 * Real.sin A) (h2 : S = 2 * Real.sqrt 3) 
    (A' : a * c = 8) : a = 2 ∧ c = 4 :=
by
  sorry

theorem find_perimeter (a b c : ℝ) 
    (h1 : b = Real.sqrt 3) (h2 : a * c = 1) 
    (h3 : a + c = 2) : a + b + c = 2 + Real.sqrt 3 :=
by
  sorry

end measure_of_angle_B_find_a_and_c_find_perimeter_l86_86147


namespace smallest_n_for_rotation_identity_l86_86109

noncomputable def rot_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]

theorem smallest_n_for_rotation_identity :
  ∃ (n : Nat), n > 0 ∧ (rot_matrix (150 * Real.pi / 180)) ^ n = 1 ∧
  ∀ (m : Nat), m > 0 ∧ (rot_matrix (150 * Real.pi / 180)) ^ m = 1 → n ≤ m :=
begin
  sorry
end

end smallest_n_for_rotation_identity_l86_86109


namespace triangle_inequality_l86_86259

variables (a b c : ℝ)

theorem triangle_inequality (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0)
  (h₃ : a + b > c) (h₄ : b + c > a) (h₅ : c + a > b) :
  (|a^2 - b^2| / c) + (|b^2 - c^2| / a) ≥ (|c^2 - a^2| / b) :=
by
  sorry

end triangle_inequality_l86_86259


namespace number_of_even_factors_l86_86444

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def count_even_factors (n : ℕ) : ℕ :=
  ( finset.range  (4)).filter_map (λ a, 
  (finset.range  (2)).filter_map (λ b, 
  (finset.range  (3)).filter_map (λ c, 
  (finset.range  (2)).filter_map (λ d, 
  if is_even (2^a * 3^b * 7^c * 5^d) 
  then some (2^a * 3^b * 7^c * 5^d)
  else none)).card * (finset.range  (2)).card * (finset.range  (3)).card * (finset.range  (2)).card

theorem number_of_even_factors :
    count_even_factors (2^3 * 3^1 * 7^2 * 5^1) = 36 :=
sorry

end number_of_even_factors_l86_86444


namespace area_of_triangle_abe_l86_86618

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
10 -- Dummy definition, in actual scenario appropriate area calculation will be required.

def length_AD : ℝ := 2
def length_BD : ℝ := 3

def areas_equal (S_ABE S_DBFE : ℝ) : Prop :=
    S_ABE = S_DBFE

theorem area_of_triangle_abe
  (area_abc : ℝ)
  (length_ad length_bd : ℝ)
  (equal_areas : areas_equal (triangle_area 1 1 1) 1) -- Dummy values, should be substituted with correct arguments
  : triangle_area 1 1 1 = 6 :=
sorry -- proof will be filled later

end area_of_triangle_abe_l86_86618


namespace largest_constant_inequality_l86_86106

theorem largest_constant_inequality (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) :
  (y*z + z*x + x*y)^2 * (x + y + z) ≥ 4 * x*y*z * (x^2 + y^2 + z^2) :=
sorry

end largest_constant_inequality_l86_86106


namespace old_edition_pages_l86_86718

theorem old_edition_pages :
  ∃ (x : ℕ), let y := 3 * x^2 - 90 in
    450 = 2 * x - 230 ∧
    y ≥ ((11:ℕ) * x / 10) ∧
    y.isNat :=
by
  -- Proof will be added here
  sorry

end old_edition_pages_l86_86718


namespace shape_of_constant_phi_l86_86901

-- Define the spherical coordinates structure
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the condition that φ is a constant c
def constant_phi (c : ℝ) (coords : SphericalCoordinates) : Prop :=
  coords.φ = c

-- Define the type for shapes
inductive Shape
  | Line : Shape
  | Circle : Shape
  | Plane : Shape
  | Sphere : Shape
  | Cylinder : Shape
  | Cone : Shape

-- The theorem statement
theorem shape_of_constant_phi (c : ℝ) (coords : SphericalCoordinates) 
  (h : constant_phi c coords) : Shape :=
  Shape.Cone

end shape_of_constant_phi_l86_86901


namespace whale_sixth_hour_consumption_l86_86868

-- Definitions based on the given conditions
def consumption (x : ℕ) (hour : ℕ) : ℕ := x + 3 * (hour - 1)

def total_consumption (x : ℕ) : ℕ := 
  (consumption x 1) + (consumption x 2) + (consumption x 3) +
  (consumption x 4) + (consumption x 5) + (consumption x 6) + 
  (consumption x 7) + (consumption x 8) + (consumption x 9)

-- Given problem translated to Lean
theorem whale_sixth_hour_consumption (x : ℕ) (h1 : total_consumption x = 270) :
  consumption x 6 = 33 :=
sorry

end whale_sixth_hour_consumption_l86_86868


namespace reciprocal_of_negative_2023_l86_86983

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l86_86983


namespace b_is_arithmetic_sequence_l86_86188

theorem b_is_arithmetic_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) :
  a 1 = 1 →
  a 2 = 2 →
  (∀ n, a (n + 2) = 2 * a (n + 1) - a n + 2) →
  (∀ n, b n = a (n + 1) - a n) →
  ∃ d, ∀ n, b (n + 1) = b n + d :=
by
  intros h1 h2 h3 h4
  use 2
  sorry

end b_is_arithmetic_sequence_l86_86188


namespace parallel_lines_m_eq_minus_seven_l86_86268

theorem parallel_lines_m_eq_minus_seven
  (m : ℝ)
  (l₁ : ∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m)
  (l₂ : ∀ x y : ℝ, 2 * x + (5 + m) * y = 8)
  (parallel : ∀ x y : ℝ, (3 + m) * 4 = 2 * (5 + m)) :
  m = -7 :=
sorry

end parallel_lines_m_eq_minus_seven_l86_86268


namespace cubic_identity_l86_86277

theorem cubic_identity (x : ℝ) (h : x + (1/x) = -3) : x^3 + (1/x^3) = -18 :=
by
  sorry

end cubic_identity_l86_86277


namespace rooster_weight_l86_86600

variable (W : ℝ)  -- The weight of the first rooster

theorem rooster_weight (h1 : 0.50 * W + 0.50 * 40 = 35) : W = 30 :=
by
  sorry

end rooster_weight_l86_86600


namespace no_solution_for_n_eq_neg1_l86_86246

theorem no_solution_for_n_eq_neg1 (x y z : ℝ) : ¬ (∃ x y z, (-1) * x^2 + y = 2 ∧ (-1) * y^2 + z = 2 ∧ (-1) * z^2 + x = 2) :=
by
  sorry

end no_solution_for_n_eq_neg1_l86_86246


namespace victory_circle_count_l86_86004

   -- Define the conditions
   def num_runners : ℕ := 8
   def num_medals : ℕ := 5
   def medals : List String := ["gold", "silver", "bronze", "titanium", "copper"]
   
   -- Define the scenarios
   def scenario1 : ℕ := 2 * 6 -- 2! * 3!
   def scenario2 : ℕ := 6 * 2 -- 3! * 2!
   def scenario3 : ℕ := 2 * 2 * 1 -- 2! * 2! * 1!

   -- Calculate the total number of victory circles
   def total_victory_circles : ℕ := scenario1 + scenario2 + scenario3

   theorem victory_circle_count : total_victory_circles = 28 := by
     sorry
   
end victory_circle_count_l86_86004


namespace smallest_prime_12_less_than_perfect_square_l86_86347

theorem smallest_prime_12_less_than_perfect_square : ∃ n : ℕ, prime n ∧ ∃ k : ℕ, k^2 - n = 12 ∧ n = 13 :=
by {
  use 13,
  split,
  { exact prime_def.2 ⟨nat.prime_def_lt.mp nat.prime_two⟩, -- Proof that 13 is prime, simplified
  { use 5,
    split,
    { calc
      5 ^ 2 - 13
         = 25 - 13 : by rfl
    ... = 12 : by rfl,
    { refl,
    }
  }
end

end smallest_prime_12_less_than_perfect_square_l86_86347


namespace point_in_second_quadrant_l86_86931

variable (m : ℝ)

-- Defining the conditions
def x_negative (m : ℝ) := 3 - m < 0
def y_positive (m : ℝ) := m - 1 > 0

theorem point_in_second_quadrant (h1 : x_negative m) (h2 : y_positive m) : m > 3 :=
by
  sorry

end point_in_second_quadrant_l86_86931


namespace mary_needs_6_cups_of_flour_l86_86953

-- Define the necessary constants according to the conditions.
def flour_needed : ℕ := 6
def sugar_needed : ℕ := 13
def flour_more_than_sugar : ℕ := 8

-- Define the number of cups of flour Mary needs to add.
def flour_to_add (flour_put_in : ℕ) : ℕ := flour_needed - flour_put_in

-- Prove that Mary needs to add 6 more cups of flour.
theorem mary_needs_6_cups_of_flour (flour_put_in : ℕ) (h : flour_more_than_sugar = 8): flour_to_add flour_put_in = 6 :=
by {
  sorry -- the proof is omitted.
}

end mary_needs_6_cups_of_flour_l86_86953


namespace population_multiple_of_seven_l86_86325

theorem population_multiple_of_seven 
  (a b c : ℕ) 
  (h1 : a^2 + 100 = b^2 + 1) 
  (h2 : b^2 + 1 + 100 = c^2) : 
  (∃ k : ℕ, a = 7 * k) :=
sorry

end population_multiple_of_seven_l86_86325


namespace arithmetic_sequence_sum_l86_86948

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (h_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 5) (h_a5 : a 5 = 9) :
  S 7 = 49 :=
sorry

end arithmetic_sequence_sum_l86_86948


namespace reciprocal_of_neg_2023_l86_86974

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l86_86974


namespace minimum_employment_age_l86_86551

/-- This structure represents the conditions of the problem -/
structure EmploymentConditions where
  jane_current_age : ℕ  -- Jane's current age
  years_until_dara_half_age : ℕ  -- Years until Dara is half Jane's age
  years_until_dara_min_age : ℕ  -- Years until Dara reaches minimum employment age

/-- The proof problem statement -/
theorem minimum_employment_age (conds : EmploymentConditions)
  (h_jane : conds.jane_current_age = 28)
  (h_half_age : conds.years_until_dara_half_age = 6)
  (h_min_age : conds.years_until_dara_min_age = 14) :
  let jane_in_six := conds.jane_current_age + conds.years_until_dara_half_age
  let dara_in_six := jane_in_six / 2
  let dara_now := dara_in_six - conds.years_until_dara_half_age
  let M := dara_now + conds.years_until_dara_min_age
  M = 25 :=
by
  sorry

end minimum_employment_age_l86_86551


namespace tangent_line_at_point_l86_86548

noncomputable def tangent_line_equation (f : ℝ → ℝ) (P : ℝ × ℝ) :=
let f' := deriv f in
let slope := f' P.1 in
let point_slope_form := λ x, slope * (x - P.1) + P.2 in
λ y, y = slope * (y - P.1) + P.2

theorem tangent_line_at_point (f : ℝ → ℝ) (P : ℝ × ℝ) (h : f = λ x, x^4 - x) (hP : P = (1, 0)) :
  tangent_line_equation f P = (λ x y, 3 * x - y - 3 = 0) :=
by
  sorry

end tangent_line_at_point_l86_86548


namespace expand_and_simplify_l86_86892

theorem expand_and_simplify (y : ℚ) (h : y ≠ 0) :
  (3/4 * (8/y - 6*y^2 + 3*y)) = (6/y - 9*y^2/2 + 9*y/4) :=
by
  sorry

end expand_and_simplify_l86_86892


namespace Lois_books_total_l86_86952

-- Definitions based on the conditions
def initial_books : ℕ := 150
def books_given_to_nephew : ℕ := initial_books / 4
def remaining_books : ℕ := initial_books - books_given_to_nephew
def non_fiction_books : ℕ := remaining_books * 60 / 100
def kept_non_fiction_books : ℕ := non_fiction_books / 2
def fiction_books : ℕ := remaining_books - non_fiction_books
def lent_fiction_books : ℕ := fiction_books / 3
def remaining_fiction_books : ℕ := fiction_books - lent_fiction_books
def newly_purchased_books : ℕ := 12

-- The total number of books Lois has now
def total_books_now : ℕ := kept_non_fiction_books + remaining_fiction_books + newly_purchased_books

-- Theorem statement
theorem Lois_books_total : total_books_now = 76 := by
  sorry

end Lois_books_total_l86_86952


namespace alissa_total_amount_spent_correct_l86_86871
-- Import necessary Lean library

-- Define the costs of individual items
def football_cost : ℝ := 8.25
def marbles_cost : ℝ := 6.59
def puzzle_cost : ℝ := 12.10
def action_figure_cost : ℝ := 15.29
def board_game_cost : ℝ := 23.47

-- Define the discount rate and the sales tax rate
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.06

-- Define the total cost before discount
def total_cost_before_discount : ℝ :=
  football_cost + marbles_cost + puzzle_cost + action_figure_cost + board_game_cost

-- Define the discount amount
def discount : ℝ := total_cost_before_discount * discount_rate

-- Define the total cost after discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount

-- Define the sales tax amount
def sales_tax : ℝ := total_cost_after_discount * sales_tax_rate

-- Define the total amount spent
def total_amount_spent : ℝ := total_cost_after_discount + sales_tax

-- Prove that the total amount spent is $62.68
theorem alissa_total_amount_spent_correct : total_amount_spent = 62.68 := 
  by 
    sorry

end alissa_total_amount_spent_correct_l86_86871


namespace relationship_between_a_b_c_l86_86469

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1 / 3)
noncomputable def c : ℝ := Real.log 3 / Real.log (1 / 2)

theorem relationship_between_a_b_c : c < b ∧ b < a :=
by
  sorry

end relationship_between_a_b_c_l86_86469


namespace evaluate_x_squared_when_y_equals_4_l86_86315

open Real

theorem evaluate_x_squared_when_y_equals_4 :
  ∀ (x y k : ℝ), (x = 5) → (y = 2) → (x^2 * y^4 = k) → (y = 4) → (x^2 = 25 / 16) :=
by
  intros x y k hx hy h1 hy' 
  -- Additional constraints and steps for the proof
  sorry

end evaluate_x_squared_when_y_equals_4_l86_86315


namespace gambler_target_win_percentage_l86_86216

-- Define the initial conditions
def initial_games_played : ℕ := 20
def initial_win_rate : ℚ := 0.40

def additional_games_played : ℕ := 20
def additional_win_rate : ℚ := 0.80

-- Define the proof problem statement
theorem gambler_target_win_percentage 
  (initial_wins : ℚ := initial_win_rate * initial_games_played)
  (additional_wins : ℚ := additional_win_rate * additional_games_played)
  (total_games_played : ℕ := initial_games_played + additional_games_played)
  (total_wins : ℚ := initial_wins + additional_wins) :
  ((total_wins / total_games_played) * 100 : ℚ) = 60 := 
by
  -- Skipping the proof steps
  sorry

end gambler_target_win_percentage_l86_86216


namespace johns_profit_l86_86151

def profit (n : ℕ) (p c : ℕ) : ℕ :=
  n * p - c

theorem johns_profit :
  profit 20 15 100 = 200 :=
by
  sorry

end johns_profit_l86_86151


namespace choose_four_socks_from_seven_l86_86171

theorem choose_four_socks_from_seven : (Nat.choose 7 4) = 35 :=
by
  sorry

end choose_four_socks_from_seven_l86_86171


namespace geese_survived_first_year_l86_86020

-- Definitions based on the conditions
def total_eggs := 900
def hatch_rate := 2 / 3
def survive_first_month_rate := 3 / 4
def survive_first_year_rate := 2 / 5

-- Definitions derived from the conditions
def hatched_geese := total_eggs * hatch_rate
def survived_first_month := hatched_geese * survive_first_month_rate
def survived_first_year := survived_first_month * survive_first_year_rate

-- Target proof statement
theorem geese_survived_first_year : survived_first_year = 180 := by
  sorry

end geese_survived_first_year_l86_86020


namespace reciprocal_of_neg_2023_l86_86978

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l86_86978


namespace solution_comparison_l86_86026

theorem solution_comparison (a a' b b' k : ℝ) (h1 : a ≠ 0) (h2 : a' ≠ 0) (h3 : 0 < k) :
  (k * b * a') > (a * b') :=
sorry

end solution_comparison_l86_86026


namespace range_of_k_is_l86_86968

noncomputable def range_of_k (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : Set ℝ :=
{k : ℝ | ∀ x : ℝ, a^x + 4 * a^(-x) - k > 0}

theorem range_of_k_is (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  range_of_k a h₁ h₂ = { k : ℝ | k < 4 ∧ k ≠ 3 } :=
sorry

end range_of_k_is_l86_86968


namespace number_of_interviewees_l86_86203

theorem number_of_interviewees (n : ℕ) (h : (6 : ℚ) / (n * (n - 1)) = 1 / 70) : n = 21 :=
sorry

end number_of_interviewees_l86_86203


namespace sheila_attends_l86_86536

noncomputable def probability_sheila_attends (P_Rain P_Attend_Rain P_Sunny P_Attend_Sunny P_Strike : ℝ) :=
  let P_Attend_without_Strike := P_Rain * P_Attend_Rain + P_Sunny * P_Attend_Sunny in
  P_Attend_without_Strike * (1 - P_Strike)

theorem sheila_attends :
  probability_sheila_attends 0.50 0.25 0.50 0.80 0.10 = 0.4725 :=
by
  -- conditions
  let P_Rain := 0.50
  let P_Attend_Rain := 0.25
  let P_Sunny := 1 - P_Rain
  let P_Attend_Sunny := 0.80
  let P_Strike := 0.10

  have P_Attend_without_Strike: ℝ := P_Rain * P_Attend_Rain + P_Sunny * P_Attend_Sunny
  have result: ℝ := P_Attend_without_Strike * (1 - P_Strike)

  have eq1 : P_Sunny = 0.50 := by calc
    P_Sunny = 1 - P_Rain   : by rfl
    ...      = 1 - 0.50    : by rfl
    ...      = 0.50        : by rfl
  have eq2 : P_Attend_without_Strike = 0.525 := by calc
    P_Attend_without_Strike = P_Rain * P_Attend_Rain + P_Sunny * P_Attend_Sunny : by rfl
    ...          = 0.50 * 0.25 + P_Sunny * 0.80                                  : by rfl
    ...          = 0.50 * 0.25 + 0.50 * 0.80                                     : by rw eq1
    ...          = 0.125 + 0.40                                                 : by rfl
    ...          = 0.525                                                        : by rfl
  have eq3 : result = 0.4725 := by calc
    result = P_Attend_without_Strike * (1 - P_Strike) : by rfl
    ...    = 0.525 * (1 - 0.10)                        : by rw eq2
    ...    = 0.525 * 0.90                              : by rfl
    ...    = 0.4725                                    : by rfl

  exact eq3 ░ sorry

end sheila_attends_l86_86536


namespace exists_integers_a_b_for_m_l86_86422

theorem exists_integers_a_b_for_m (m : ℕ) (h : 0 < m) :
  ∃ a b : ℤ, |a| ≤ m ∧ |b| ≤ m ∧ 0 < a + b * Real.sqrt 2 ∧ a + b * Real.sqrt 2 ≤ (1 + Real.sqrt 2) / (m + 2) :=
by
  sorry

end exists_integers_a_b_for_m_l86_86422


namespace avg_difference_l86_86805

def avg (a b c : ℕ) := (a + b + c) / 3

theorem avg_difference : avg 14 32 53 - avg 21 47 22 = 3 :=
by
  sorry

end avg_difference_l86_86805


namespace files_per_folder_l86_86301

theorem files_per_folder
    (initial_files : ℕ)
    (deleted_files : ℕ)
    (folders : ℕ)
    (remaining_files : ℕ)
    (files_per_folder : ℕ)
    (initial_files_eq : initial_files = 93)
    (deleted_files_eq : deleted_files = 21)
    (folders_eq : folders = 9)
    (remaining_files_eq : remaining_files = initial_files - deleted_files)
    (files_per_folder_eq : files_per_folder = remaining_files / folders) :
    files_per_folder = 8 :=
by
    -- Here, sorry is used to skip the actual proof steps 
    sorry

end files_per_folder_l86_86301


namespace eggs_used_afternoon_l86_86162

theorem eggs_used_afternoon (eggs_pumpkin eggs_apple eggs_cherry eggs_total : ℕ)
  (h_pumpkin : eggs_pumpkin = 816)
  (h_apple : eggs_apple = 384)
  (h_cherry : eggs_cherry = 120)
  (h_total : eggs_total = 1820) :
  eggs_total - (eggs_pumpkin + eggs_apple + eggs_cherry) = 500 :=
by
  sorry

end eggs_used_afternoon_l86_86162


namespace function_sqrt_plus_one_l86_86275

variable (f : ℝ → ℝ)
variable (x : ℝ)

theorem function_sqrt_plus_one (h1 : ∀ x : ℝ, f x = 3) (h2 : x ≥ 0) : f (Real.sqrt x) + 1 = 4 :=
by
  sorry

end function_sqrt_plus_one_l86_86275


namespace cube_inverse_sum_l86_86278

theorem cube_inverse_sum (x : ℂ) (h : x + 1/x = -3) : x^3 + (1/x)^3 = -18 :=
by
  sorry

end cube_inverse_sum_l86_86278


namespace div_pow_eq_l86_86292

theorem div_pow_eq (n : ℕ) (h : n = 16 ^ 2023) : n / 4 = 4 ^ 4045 :=
by
  rw [h]
  sorry

end div_pow_eq_l86_86292


namespace interest_rate_per_annum_l86_86318

theorem interest_rate_per_annum (P T : ℝ) (r : ℝ) 
  (h1 : P = 15000) 
  (h2 : T = 2)
  (h3 : P * (1 + r)^T - P - (P * r * T) = 150) : 
  r = 0.1 :=
by
  sorry

end interest_rate_per_annum_l86_86318


namespace total_tickets_correct_l86_86031

-- Define the initial number of tickets Tate has
def initial_tickets_Tate : ℕ := 32

-- Define the additional tickets Tate buys
def additional_tickets_Tate : ℕ := 2

-- Calculate the total number of tickets Tate has
def total_tickets_Tate : ℕ := initial_tickets_Tate + additional_tickets_Tate

-- Define the number of tickets Peyton has (half of Tate's total tickets)
def tickets_Peyton : ℕ := total_tickets_Tate / 2

-- Calculate the total number of tickets Tate and Peyton have together
def total_tickets_together : ℕ := total_tickets_Tate + tickets_Peyton

-- Prove that the total number of tickets together equals 51
theorem total_tickets_correct : total_tickets_together = 51 := by
  sorry

end total_tickets_correct_l86_86031


namespace rhombus_region_area_l86_86670

noncomputable def region_area (s : ℝ) (angleB : ℝ) : ℝ :=
  let h := (s / 2) * (Real.sin (angleB / 2))
  let area_triangle := (1 / 2) * (s / 2) * h
  3 * area_triangle

theorem rhombus_region_area : region_area 3 150 = 0.87345 := by
    sorry

end rhombus_region_area_l86_86670


namespace expression_divisible_by_1897_l86_86310

theorem expression_divisible_by_1897 (n : ℕ) :
  1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end expression_divisible_by_1897_l86_86310


namespace reciprocal_of_neg_2023_l86_86996

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86996


namespace triangle_groups_count_l86_86214

theorem triangle_groups_count (total_points collinear_groups groups_of_three total_combinations : ℕ)
    (h1 : total_points = 12)
    (h2 : collinear_groups = 16)
    (h3 : groups_of_three = (total_points.choose 3))
    (h4 : total_combinations = groups_of_three - collinear_groups) :
    total_combinations = 204 :=
by
  -- This is where the proof would go
  sorry

end triangle_groups_count_l86_86214


namespace reciprocal_of_neg_2023_l86_86995

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86995


namespace range_of_a_l86_86399

noncomputable def in_range (a : ℝ) : Prop :=
  (0 < a ∧ a < 1) ∨ (a ≥ 1)

theorem range_of_a (a : ℝ) (p q : Prop) (h1 : p ↔ (0 < a ∧ a < 1)) (h2 : q ↔ (a ≥ 1 / 2)) (h3 : p ∨ q) (h4 : ¬ (p ∧ q)) :
  in_range a :=
by
  sorry

end range_of_a_l86_86399


namespace riverview_problem_l86_86942

theorem riverview_problem (h c : Nat) (p : Nat := 4 * h) (s : Nat := 5 * c) (d : Nat := 4 * p) :
  (p + h + s + c + d = 52 → false) :=
by {
  sorry
}

end riverview_problem_l86_86942


namespace max_pawns_19x19_l86_86191

def maxPawnsOnChessboard (n : ℕ) := 
  n * n

theorem max_pawns_19x19 :
  maxPawnsOnChessboard 19 = 361 := 
by
  sorry

end max_pawns_19x19_l86_86191


namespace children_count_l86_86775

theorem children_count 
  (A B C : Finset ℕ)
  (hA : A.card = 7)
  (hB : B.card = 6)
  (hC : C.card = 5)
  (hA_inter_B : (A ∩ B).card = 4)
  (hA_inter_C : (A ∩ C).card = 3)
  (hB_inter_C : (B ∩ C).card = 2)
  (hA_inter_B_inter_C : (A ∩ B ∩ C).card = 1) :
  (A ∪ B ∪ C).card = 10 := 
by
  sorry

end children_count_l86_86775


namespace find_m_value_l86_86436

-- Define the quadratic function
def quadratic (x m : ℝ) : ℝ := x^2 - 6 * x + m

-- Define the condition that the quadratic function has a minimum value of 1
def has_minimum_value_of_one (m : ℝ) : Prop := ∃ x : ℝ, quadratic x m = 1

-- The main theorem statement
theorem find_m_value : ∀ m : ℝ, has_minimum_value_of_one m → m = 10 :=
by sorry

end find_m_value_l86_86436


namespace calculate_a5_l86_86420

variable {a1 : ℝ} -- geometric sequence first term
variable {a : ℕ → ℝ} -- geometric sequence
variable {n : ℕ} -- sequence index
variable {r : ℝ} -- common ratio

-- Definitions based on the given conditions
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a1 * r ^ n

-- Given conditions
axiom common_ratio_is_two : r = 2
axiom product_condition : a 2 * a 10 = 16 -- indices offset by 1, so a3 = a 2 and a11 = a 10
axiom positive_terms : ∀ n, a n > 0

-- Goal: calculate a 4
theorem calculate_a5 : a 4 = 1 :=
sorry

end calculate_a5_l86_86420


namespace sum_of_eight_numbers_l86_86525

theorem sum_of_eight_numbers (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 := 
begin 
  sorry 
end

end sum_of_eight_numbers_l86_86525


namespace v_at_one_l86_86293

def u (x : ℝ) : ℝ := 4 * x - 9

def v (y : ℝ) : ℝ := let x := (y + 9) / 4 in x^2 + 4 * x - 5

theorem v_at_one : v 1 = 11.25 :=
by
  -- placeholder for the proof
  sorry

end v_at_one_l86_86293


namespace total_number_of_fleas_l86_86904

theorem total_number_of_fleas :
  let G_fleas := 10
  let O_fleas := G_fleas / 2
  let M_fleas := 5 * O_fleas
  G_fleas + O_fleas + M_fleas = 40 := rfl

end total_number_of_fleas_l86_86904


namespace correct_transformation_l86_86363

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0): (a / b = 2 * a / (2 * b)) :=
by
  sorry

end correct_transformation_l86_86363


namespace log_term_evaluation_l86_86238

theorem log_term_evaluation : (Real.log 2)^2 + (Real.log 5)^2 + 2 * (Real.log 2) * (Real.log 5) = 1 := by
  sorry

end log_term_evaluation_l86_86238


namespace sum_values_l86_86083

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (x + 2) = -f x
axiom value_at_one : f 1 = 8

theorem sum_values :
  f 2008 + f 2009 + f 2010 = 8 :=
sorry

end sum_values_l86_86083


namespace problem1_problem2_l86_86874

-- Problem 1: y(x + y) + (x + y)(x - y) = x^2
theorem problem1 (x y : ℝ) : y * (x + y) + (x + y) * (x - y) = x^2 := 
by sorry

-- Problem 2: ( (2m + 1) / (m + 1) + m - 1 ) ÷ ( (m + 2) / (m^2 + 2m + 1) ) = m^2 + m
theorem problem2 (m : ℝ) (h1 : m ≠ -1) : 
  ( (2 * m + 1) / (m + 1) + m - 1 ) / ( (m + 2) / ((m + 1)^2) ) = m^2 + m := 
by sorry

end problem1_problem2_l86_86874


namespace math_problem_l86_86923

theorem math_problem (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x + y + x * y = 1) :
  x * y + 1 / (x * y) - y / x - x / y = 4 :=
sorry

end math_problem_l86_86923


namespace supremum_neg_frac_l86_86421

noncomputable def supremum_expression (a b : ℝ) : ℝ :=
  - (1 / (2 * a) + 2 / b)

theorem supremum_neg_frac {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) :
  ∃ M : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → supremum_expression x y ≤ M)
  ∧ (∀ N : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → supremum_expression x y ≤ N) → M ≤ N)
  ∧ M = -9 / 2 :=
sorry

end supremum_neg_frac_l86_86421


namespace P_at_7_eq_5760_l86_86951

noncomputable def P (x : ℝ) : ℝ :=
  12 * (x - 1) * (x - 2) * (x - 3)^2 * (x - 6)^4

theorem P_at_7_eq_5760 : P 7 = 5760 :=
by
  -- Proof goes here
  sorry

end P_at_7_eq_5760_l86_86951


namespace inequality_proof_l86_86014

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (ab / (a + b)) + (bc / (b + c)) + (ca / (c + a)) ≤ (3 * (ab + bc + ca)) / (2 * (a + b + c)) :=
by
  sorry

end inequality_proof_l86_86014


namespace vector_subtraction_l86_86758

theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (3, 5)) (h2 : b = (-2, 1)) :
  a - (2 : ℝ) • b = (7, 3) :=
by
  rw [h1, h2]
  simp
  sorry

end vector_subtraction_l86_86758


namespace inequality_0_lt_a_lt_1_l86_86168

theorem inequality_0_lt_a_lt_1 (a : ℝ) (h : 0 < a ∧ a < 1) : 
  (1 / a) + (4 / (1 - a)) ≥ 9 :=
by
  sorry

end inequality_0_lt_a_lt_1_l86_86168


namespace gcd_840_1764_l86_86322

def a : ℕ := 840
def b : ℕ := 1764

theorem gcd_840_1764 : Nat.gcd a b = 84 := by
  -- Proof omitted
  sorry

end gcd_840_1764_l86_86322


namespace jemma_total_grasshoppers_l86_86650

def number_of_grasshoppers_on_plant : Nat := 7
def number_of_dozen_baby_grasshoppers : Nat := 2
def number_in_a_dozen : Nat := 12

theorem jemma_total_grasshoppers :
  number_of_grasshoppers_on_plant + number_of_dozen_baby_grasshoppers * number_in_a_dozen = 31 := by
  sorry

end jemma_total_grasshoppers_l86_86650


namespace div_by_1897_l86_86309

theorem div_by_1897 (n : ℕ) : 1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end div_by_1897_l86_86309


namespace find_v1_l86_86294

def u (x : ℝ) : ℝ := 4 * x - 9

def v (y : ℝ) : ℝ := y^2 + 4 * y - 5

theorem find_v1 : v 1 = 11.25 := by
  sorry

end find_v1_l86_86294


namespace elizabeth_bananas_eaten_l86_86599

theorem elizabeth_bananas_eaten (initial_bananas remaining_bananas eaten_bananas : ℕ) 
    (h1 : initial_bananas = 12) 
    (h2 : remaining_bananas = 8) 
    (h3 : eaten_bananas = initial_bananas - remaining_bananas) :
    eaten_bananas = 4 := 
sorry

end elizabeth_bananas_eaten_l86_86599


namespace quadratic_single_intersection_l86_86282

theorem quadratic_single_intersection (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x + m = 0 → x^2 - 2 * x + m = (x-1)^2) :=
sorry

end quadratic_single_intersection_l86_86282


namespace breadth_of_room_is_6_l86_86709

theorem breadth_of_room_is_6 
(the_room_length : ℝ) 
(the_carpet_width : ℝ) 
(cost_per_meter : ℝ) 
(total_cost : ℝ) 
(h1 : the_room_length = 15) 
(h2 : the_carpet_width = 0.75) 
(h3 : cost_per_meter = 0.30) 
(h4 : total_cost = 36) : 
  ∃ (breadth_of_room : ℝ), breadth_of_room = 6 :=
sorry

end breadth_of_room_is_6_l86_86709


namespace additional_cost_per_person_l86_86217

-- Define the initial conditions and variables used in the problem
def base_cost := 1700
def discount_per_person := 50
def car_wash_earnings := 500
def initial_friends := 6
def final_friends := initial_friends - 1

-- Calculate initial cost per person with all friends
def discounted_base_cost_initial := base_cost - (initial_friends * discount_per_person)
def total_cost_after_car_wash_initial := discounted_base_cost_initial - car_wash_earnings
def cost_per_person_initial := total_cost_after_car_wash_initial / initial_friends

-- Calculate final cost per person after Brad leaves
def discounted_base_cost_final := base_cost - (final_friends * discount_per_person)
def total_cost_after_car_wash_final := discounted_base_cost_final - car_wash_earnings
def cost_per_person_final := total_cost_after_car_wash_final / final_friends

-- Proving the amount each friend has to pay more after Brad leaves
theorem additional_cost_per_person : cost_per_person_final - cost_per_person_initial = 40 := 
by
  sorry

end additional_cost_per_person_l86_86217


namespace bread_needed_for_sandwiches_l86_86227

def students_per_group := 5
def groups := 5
def sandwiches_per_student := 2
def pieces_of_bread_per_sandwich := 2

theorem bread_needed_for_sandwiches : 
  students_per_group * groups * sandwiches_per_student * pieces_of_bread_per_sandwich = 100 := 
by
  sorry

end bread_needed_for_sandwiches_l86_86227


namespace count_multiples_less_than_300_l86_86270

theorem count_multiples_less_than_300 : ∀ n : ℕ, n < 300 → (2 * 3 * 5 * 7 ∣ n) ↔ n = 210 :=
by
  sorry

end count_multiples_less_than_300_l86_86270


namespace f_3_2_plus_f_5_1_l86_86886

def f (a b : ℤ) : ℚ :=
  if a - b ≤ 2 then (a * b - a - 1) / (3 * a)
  else (a * b + b - 1) / (-3 * b)

theorem f_3_2_plus_f_5_1 :
  f 3 2 + f 5 1 = -13 / 9 :=
by
  sorry

end f_3_2_plus_f_5_1_l86_86886


namespace trigonometric_identity_l86_86751

theorem trigonometric_identity (x : ℝ) (h : (1 + Real.sin x) / Real.cos x = -1/2) : 
  Real.cos x / (Real.sin x - 1) = 1/2 := 
sorry

end trigonometric_identity_l86_86751


namespace sum_of_eight_numbers_l86_86530

-- Definitions used in the conditions
variables {a b c d e f g h : ℕ}

-- Given condition
axiom product_condition : (a + b) * (c + d) * (e + f) * (g + h) = 330

-- Define individual sums
def ab_sum := a + b
def cd_sum := c + d
def ef_sum := e + f
def gh_sum := g + h

-- Define the total sum of the eight numbers on the cards
def total_sum := ab_sum + cd_sum + ef_sum + gh_sum

-- The theorem to prove
theorem sum_of_eight_numbers : total_sum = 21 := by
  have ab_sum_eq : ab_sum = 2 := sorry
  have cd_sum_eq : cd_sum = 3 := sorry
  have ef_sum_eq : ef_sum = 5 := sorry
  have gh_sum_eq : gh_sum = 11 := sorry
  rw [ab_sum_eq, cd_sum_eq, ef_sum_eq, gh_sum_eq]
  norm_num

end sum_of_eight_numbers_l86_86530


namespace smallest_number_diminished_by_8_divisible_by_9_6_18_equals_62_l86_86566

theorem smallest_number_diminished_by_8_divisible_by_9_6_18_equals_62
  (n: ℕ) (h1: n - 8 = 44) 
  (h2: (n - 8) % 9 = 0)
  (h3: (n - 8) % 6 = 0)
  (h4: (n - 8) % 18 = 0) : 
  n = 62 :=
sorry

end smallest_number_diminished_by_8_divisible_by_9_6_18_equals_62_l86_86566


namespace probability_multiple_of_100_l86_86474

open ProbabilityTheory
open Classical

-- Define the set of numbers
def numberSet : Finset ℕ := {2, 4, 10, 12, 15, 20, 50}

-- Define the condition for a product to be a multiple of 100
def isMultipleOf100 (a b : ℕ) : Prop := (a * b) % 100 = 0

-- Define the event space for two distinct elements from the set
def eventSpace : Finset (ℕ × ℕ) :=
  (numberSet.product numberSet).filter (λ (x : ℕ × ℕ), x.fst ≠ x.snd)

-- Define the successful events where the product is a multiple of 100
def successfulEvents : Finset (ℕ × ℕ) :=
  eventSpace.filter (λ (x : ℕ × ℕ), isMultipleOf100 x.fst x.snd)

-- Define the probability of picking such a pair
def probability : ℚ :=
  successfulEvents.card.toRat / eventSpace.card.toRat

-- The statement to prove
theorem probability_multiple_of_100 : probability = 1 / 3 :=
begin
  sorry
end

end probability_multiple_of_100_l86_86474


namespace rachel_essay_time_spent_l86_86169

noncomputable def time_spent_writing (pages : ℕ) (time_per_page : ℕ) : ℕ :=
  pages * time_per_page

noncomputable def total_time_spent (research_time : ℕ) (writing_time : ℕ) (editing_time : ℕ) : ℕ :=
  research_time + writing_time + editing_time

noncomputable def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

theorem rachel_essay_time_spent :
  let research_time := 45
  let writing_time := time_spent_writing 6 30
  let editing_time := 75
  let total_minutes := total_time_spent research_time writing_time editing_time
  minutes_to_hours total_minutes = 5 :=
by
  -- Definitions and intermediate steps
  let research_time := 45
  let writing_time := time_spent_writing 6 30
  let editing_time := 75
  let total_minutes := total_time_spent research_time writing_time editing_time
  have h_writing_time : writing_time = 6 * 30 := rfl
  have h_total_minutes : total_minutes = 45 + 180 + 75 := by
    rw [h_writing_time]
    rfl
  have h_total_minutes_calc : total_minutes = 300 := by
    exact h_total_minutes
  have h_hours : minutes_to_hours total_minutes = 5 := by
    rw [h_total_minutes_calc]
    rfl
  exact h_hours

end rachel_essay_time_spent_l86_86169


namespace minimum_prism_volume_l86_86719

theorem minimum_prism_volume (l m n : ℕ) (h1 : l > 0) (h2 : m > 0) (h3 : n > 0)
    (hidden_volume_condition : (l - 1) * (m - 1) * (n - 1) = 420) :
    ∃ N : ℕ, N = l * m * n ∧ N = 630 := by
  sorry

end minimum_prism_volume_l86_86719


namespace stadium_breadth_l86_86635

theorem stadium_breadth (P L B : ℕ) (h1 : P = 800) (h2 : L = 100) :
  2 * (L + B) = P → B = 300 :=
by
  sorry

end stadium_breadth_l86_86635


namespace daily_sales_profit_function_selling_price_for_given_profit_l86_86540

noncomputable def profit (x : ℝ) (y : ℝ) := x * y - 20 * y

theorem daily_sales_profit_function (x : ℝ) :
  let y := -2 * x + 80
  in profit x y = -2 * x^2 + 120 * x - 1600 := by
  let y := -2 * x + 80
  calc
    profit x y = x * y - 20 * y : rfl
          ... = x * (-2 * x + 80) - 20 * (-2 * x + 80) : by rw y
          ... = x * (-2 * x + 80) - 20 * (-2 * x + 80) : rfl
          ... = -2 * x^2 + 80 * x + 40 * x - 1600 : by ring
          ... = -2 * x^2 + 120 * x - 1600 : by ring

theorem selling_price_for_given_profit (W : ℝ) (x : ℝ) :
  W = -2 * x^2 + 120 * x - 1600 → x ≤ 30 → W = 150 → x = 25 := by
  intros h₁ h₂ h₃
  have h := congr_arg (λ W, W - 150) h₁
  rw h₃ at h
  calc
    _ - 150 = -2 * x^2 + 120 * x - 1600 - 150 : h
        ... = -2 * x^2 + 120 * x - 1750 : by ring
        ... = 0 : by exact h₃

  have h₄ : x^2 - 60 * x + 875 = 0 :=
    by
      have h₅ := congr_arg (λ W, -W) h
      rw [neg_sub, sub_eq_add_neg, neg_neg] at h₅
      exact h₅
  have h₆ : (x - 25) * (x - 35) = 0 :=
    by
      apply (Int.exists_two_squares_add 25 h₄).symm
      sorry
  cases h₆ with h₇ h₈
  exact h₇
  exfalso
  linarith only [h₂, h₈]

end daily_sales_profit_function_selling_price_for_given_profit_l86_86540


namespace solve_inequality_l86_86250

theorem solve_inequality (x : ℝ) : (1 ≤ |x + 3| ∧ |x + 3| ≤ 4) ↔ (-7 ≤ x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_l86_86250


namespace geometric_sequence_second_term_l86_86684

theorem geometric_sequence_second_term (a r : ℝ) 
  (h_fifth_term : a * r^4 = 48) 
  (h_sixth_term : a * r^5 = 72) : 
  a * r = 1152 / 81 := by
  sorry

end geometric_sequence_second_term_l86_86684


namespace min_vertical_distance_between_graphs_l86_86041

noncomputable def absolute_value (x : ℝ) : ℝ :=
if x >= 0 then x else -x

theorem min_vertical_distance_between_graphs : 
  ∃ d : ℝ, d = 3 / 4 ∧ ∀ x : ℝ, ∃ dist : ℝ, dist = absolute_value x - (- x^2 - 4 * x - 3) ∧ dist >= d :=
by
  sorry

end min_vertical_distance_between_graphs_l86_86041


namespace original_grape_jelly_beans_l86_86405

namespace JellyBeans

-- Definition of the problem conditions
variables (g c : ℕ)
axiom h1 : g = 3 * c
axiom h2 : g - 15 = 5 * (c - 5)

-- Proof goal statement
theorem original_grape_jelly_beans : g = 15 :=
by
  sorry

end JellyBeans

end original_grape_jelly_beans_l86_86405


namespace geometric_sequence_second_term_l86_86686

theorem geometric_sequence_second_term (a r : ℝ) 
  (h_fifth_term : a * r^4 = 48) 
  (h_sixth_term : a * r^5 = 72) : 
  a * r = 1152 / 81 := by
  sorry

end geometric_sequence_second_term_l86_86686


namespace value_of_x_l86_86136

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l86_86136


namespace min_value_M_proof_l86_86754

noncomputable def min_value_M (a b c d e f g M : ℝ) : Prop :=
  (∀ (a b c d e f g : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ g ≥ 0 ∧ 
    a + b + c + d + e + f + g = 1 ∧ 
    M = max (max (max (max (a + b + c) (b + c + d)) (c + d + e)) (d + e + f)) (e + f + g)
  → M ≥ (1 / 3))

theorem min_value_M_proof : min_value_M a b c d e f g M :=
by
  sorry

end min_value_M_proof_l86_86754


namespace sum_first_11_even_numbers_is_132_l86_86065

def sum_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (n + 1)

theorem sum_first_11_even_numbers_is_132 : sum_first_n_even_numbers 11 = 132 := 
  by
    sorry

end sum_first_11_even_numbers_is_132_l86_86065


namespace a2_value_is_42_l86_86620

noncomputable def a₂_value (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :=
  a_2

theorem a2_value_is_42 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (x^3 + x^10 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + a_4 * (x + 1)^4 +
                a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + 
                a_9 * (x + 1)^9 + a_10 * (x + 1)^10) →
  a₂_value a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 = 42 :=
by
  sorry

end a2_value_is_42_l86_86620


namespace total_tickets_l86_86033

-- Define the initial number of tickets Tate has.
def tate_initial_tickets : ℕ := 32

-- Define the number of tickets Tate buys additionally.
def additional_tickets : ℕ := 2

-- Define the total number of tickets Tate has after buying more.
def tate_total_tickets : ℕ := tate_initial_tickets + additional_tickets

-- Define the total number of tickets Peyton has.
def peyton_tickets : ℕ := tate_total_tickets / 2

-- State the theorem to prove the total number of tickets Tate and Peyton have together.
theorem total_tickets : tate_total_tickets + peyton_tickets = 51 := by
  -- Placeholder for the proof
  sorry

end total_tickets_l86_86033


namespace equations_have_same_solution_l86_86679

theorem equations_have_same_solution (x c : ℝ) 
  (h1 : 3 * x + 9 = 0) (h2 : c * x + 15 = 3) : c = 4 :=
by
  sorry

end equations_have_same_solution_l86_86679


namespace reciprocal_of_neg_2023_l86_86989

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86989


namespace new_phone_plan_cost_l86_86159

def old_plan_cost : ℝ := 150
def increase_percentage : ℝ := 0.30
def new_plan_cost := old_plan_cost + (increase_percentage * old_plan_cost)

theorem new_phone_plan_cost : new_plan_cost = 195 := by
  -- From the condition that the old plan cost is $150 and the increase percentage is 30%
  -- We should prove that the new plan cost is $195
  sorry

end new_phone_plan_cost_l86_86159


namespace correct_transformation_l86_86362

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0): (a / b = 2 * a / (2 * b)) :=
by
  sorry

end correct_transformation_l86_86362


namespace joe_needs_more_cars_l86_86784

-- Definitions based on conditions
def current_cars : ℕ := 50
def total_cars : ℕ := 62

-- Theorem based on the problem question and correct answer
theorem joe_needs_more_cars : (total_cars - current_cars) = 12 :=
by
  sorry

end joe_needs_more_cars_l86_86784


namespace circle_center_radius_l86_86966

theorem circle_center_radius {x y : ℝ} :
  (∃ r : ℝ, (x - 1)^2 + y^2 = r^2) ↔ (x^2 + y^2 - 2*x - 5 = 0) :=
by sorry

end circle_center_radius_l86_86966


namespace reciprocal_of_neg_2023_l86_86993

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86993


namespace vector_calculation_l86_86255

def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (1, -1)
def vec_result : ℝ × ℝ := (3 * vec_a.fst - 2 * vec_b.fst, 3 * vec_a.snd - 2 * vec_b.snd)
def target_vec : ℝ × ℝ := (1, 5)

theorem vector_calculation :
  vec_result = target_vec :=
sorry

end vector_calculation_l86_86255


namespace sum_of_eight_numbers_l86_86514

theorem sum_of_eight_numbers (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_eight_numbers_l86_86514


namespace sqrt_expression_simplify_l86_86244

theorem sqrt_expression_simplify : 
  2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / (10 * Real.sqrt 2) = 3 * Real.sqrt 2 / 20 :=
by 
  sorry

end sqrt_expression_simplify_l86_86244


namespace miles_round_trip_time_l86_86139

theorem miles_round_trip_time : 
  ∀ (d : ℝ), d = 57 →
  ∀ (t : ℝ), t = 40 →
  ∀ (x : ℝ), x = 4 →
  10 = ((2 * d * x) / t) * 2 := 
by
  intros d hd t ht x hx
  rw [hd, ht, hx]
  sorry

end miles_round_trip_time_l86_86139


namespace not_necessarily_a_squared_lt_b_squared_l86_86135
-- Import the necessary library

-- Define the variables and the condition
variables {a b : ℝ}
axiom h : a < b

-- The theorem statement that needs to be proved/disproved
theorem not_necessarily_a_squared_lt_b_squared (a b : ℝ) (h : a < b) : ¬ (a^2 < b^2) :=
sorry

end not_necessarily_a_squared_lt_b_squared_l86_86135


namespace eval_f_3_minus_f_neg_3_l86_86762

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 7 * x

-- State the theorem
theorem eval_f_3_minus_f_neg_3 : f 3 - f (-3) = 690 := by
  sorry

end eval_f_3_minus_f_neg_3_l86_86762


namespace cos_75_deg_identity_l86_86243

theorem cos_75_deg_identity :
  real.cos (75 * real.pi / 180) = (real.sqrt 6 - real.sqrt 2) / 4 :=
by 
  -- We skip the actual proof.
  sorry

end cos_75_deg_identity_l86_86243


namespace box_height_is_55_cm_l86_86402

noncomputable def height_of_box 
  (ceiling_height_m : ℝ) 
  (light_fixture_below_ceiling_cm : ℝ) 
  (bob_height_m : ℝ) 
  (bob_reach_cm : ℝ) 
  : ℝ :=
  let ceiling_height_cm := ceiling_height_m * 100
  let bob_height_cm := bob_height_m * 100
  let light_fixture_from_floor := ceiling_height_cm - light_fixture_below_ceiling_cm
  let bob_total_reach := bob_height_cm + bob_reach_cm
  light_fixture_from_floor - bob_total_reach

-- Theorem statement
theorem box_height_is_55_cm 
  (ceiling_height_m : ℝ) 
  (light_fixture_below_ceiling_cm : ℝ) 
  (bob_height_m : ℝ) 
  (bob_reach_cm : ℝ) 
  (h : height_of_box ceiling_height_m light_fixture_below_ceiling_cm bob_height_m bob_reach_cm = 55) 
  : height_of_box 3 15 1.8 50 = 55 :=
by
  unfold height_of_box
  sorry

end box_height_is_55_cm_l86_86402


namespace john_umbrella_in_car_l86_86651

variable (UmbrellasInHouse : Nat)
variable (CostPerUmbrella : Nat)
variable (TotalAmountPaid : Nat)

theorem john_umbrella_in_car
  (h1 : UmbrellasInHouse = 2)
  (h2 : CostPerUmbrella = 8)
  (h3 : TotalAmountPaid = 24) :
  (TotalAmountPaid / CostPerUmbrella) - UmbrellasInHouse = 1 := by
  sorry

end john_umbrella_in_car_l86_86651


namespace reciprocal_of_neg_2023_l86_86976

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l86_86976


namespace point_value_of_other_questions_l86_86830

theorem point_value_of_other_questions (x y p : ℕ) 
  (h1 : x = 10) 
  (h2 : x + y = 40) 
  (h3 : 40 + 30 * p = 100) : 
  p = 2 := 
  sorry

end point_value_of_other_questions_l86_86830


namespace robin_cut_hair_l86_86671

-- Definitions as per the given conditions
def initial_length := 17
def current_length := 13

-- Statement of the proof problem
theorem robin_cut_hair : initial_length - current_length = 4 := 
by 
  sorry

end robin_cut_hair_l86_86671


namespace games_played_so_far_l86_86641

-- Definitions based on conditions
def total_matches := 20
def points_for_victory := 3
def points_for_draw := 1
def points_for_defeat := 0
def points_scored_so_far := 14
def points_needed := 40
def required_wins := 6

-- The proof problem
theorem games_played_so_far : 
  ∃ W D L : ℕ, 3 * W + D + 0 * L = points_scored_so_far ∧ 
  ∃ W' D' L' : ℕ, 3 * W' + D' + 0 * L' + 3 * required_wins = points_needed ∧ 
  (total_matches - required_wins = 14) :=
by 
  sorry

end games_played_so_far_l86_86641


namespace height_relationship_l86_86562

theorem height_relationship 
  (r₁ h₁ r₂ h₂ : ℝ)
  (h_volume : π * r₁^2 * h₁ = π * r₂^2 * h₂)
  (h_radius : r₂ = (6/5) * r₁) :
  h₁ = 1.44 * h₂ :=
by
  sorry

end height_relationship_l86_86562


namespace total_tickets_l86_86034

-- Define the initial number of tickets Tate has.
def tate_initial_tickets : ℕ := 32

-- Define the number of tickets Tate buys additionally.
def additional_tickets : ℕ := 2

-- Define the total number of tickets Tate has after buying more.
def tate_total_tickets : ℕ := tate_initial_tickets + additional_tickets

-- Define the total number of tickets Peyton has.
def peyton_tickets : ℕ := tate_total_tickets / 2

-- State the theorem to prove the total number of tickets Tate and Peyton have together.
theorem total_tickets : tate_total_tickets + peyton_tickets = 51 := by
  -- Placeholder for the proof
  sorry

end total_tickets_l86_86034


namespace find_function_l86_86661

theorem find_function (f : ℝ → ℝ) :
  (∀ x y z : ℝ, x + y + z = 0 → f (x^3) + f (y)^3 + f (z)^3 = 3 * x * y * z) → 
  f = id :=
by sorry

end find_function_l86_86661


namespace arithmetic_seq_finite_negative_terms_l86_86615

theorem arithmetic_seq_finite_negative_terms (a d : ℝ) :
  (∃ N : ℕ, ∀ n : ℕ, n > N → a + n * d ≥ 0) ↔ (a < 0 ∧ d > 0) :=
by
  sorry

end arithmetic_seq_finite_negative_terms_l86_86615


namespace mistaken_divisor_l86_86000

theorem mistaken_divisor (x : ℕ) (h1 : ∀ (d : ℕ), d ∣ 840 → d = 21 ∨ d = x) 
(h2 : 840 = 70 * x) : x = 12 := 
by sorry

end mistaken_divisor_l86_86000


namespace reciprocal_of_negative_2023_l86_86985

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l86_86985


namespace evaluate_expression_l86_86602

variables (x y : ℕ)

theorem evaluate_expression : x = 2 → y = 4 → y * (y - 2 * x + 1) = 4 :=
by
  intro h1 h2
  sorry

end evaluate_expression_l86_86602


namespace find_x_l86_86790

noncomputable def h (x : ℚ) : ℚ :=
  (5 * ((x - 2) / 3) - 3)

theorem find_x : h (19/2) = 19/2 :=
by
  sorry

end find_x_l86_86790


namespace flower_beds_fraction_correct_l86_86073

noncomputable def flower_beds_fraction (yard_length : ℝ) (yard_width : ℝ) (trapezoid_parallel_side1 : ℝ) (trapezoid_parallel_side2 : ℝ) : ℝ :=
  let leg_length := (trapezoid_parallel_side2 - trapezoid_parallel_side1) / 2
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 2 * triangle_area
  let yard_area := yard_length * yard_width
  total_flower_bed_area / yard_area

theorem flower_beds_fraction_correct :
  flower_beds_fraction 30 5 20 30 = 1 / 6 :=
by
  sorry

end flower_beds_fraction_correct_l86_86073


namespace solution_set_of_abs_x_minus_1_lt_1_l86_86047

theorem solution_set_of_abs_x_minus_1_lt_1 : {x : ℝ | |x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_abs_x_minus_1_lt_1_l86_86047


namespace prime_polynomial_l86_86788

theorem prime_polynomial (n : ℕ) (h1 : 2 ≤ n)
  (h2 : ∀ k : ℕ, k ≤ Nat.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
sorry

end prime_polynomial_l86_86788


namespace jon_found_marbles_l86_86665

-- Definitions based on the conditions
variables (M J B : ℕ)

-- Prove that Jon found 110 marbles
theorem jon_found_marbles
  (h1 : M + J = 66)
  (h2 : M = 2 * J)
  (h3 : J + B = 3 * M) :
  B = 110 :=
by
  sorry -- proof to be completed

end jon_found_marbles_l86_86665


namespace no_int_solutions_for_quadratics_l86_86084

theorem no_int_solutions_for_quadratics :
  ¬ ∃ a b c : ℤ, (∃ x1 x2 : ℤ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) ∧
                (∃ y1 y2 : ℤ, (a + 1) * y1^2 + (b + 1) * y1 + (c + 1) = 0 ∧ 
                              (a + 1) * y2^2 + (b + 1) * y2 + (c + 1) = 0) :=
by
  sorry

end no_int_solutions_for_quadratics_l86_86084


namespace chi_square_relationship_l86_86423

noncomputable def chi_square_statistic {X Y : Type*} (data : X → Y → ℝ) : ℝ := 
  sorry -- Actual definition is omitted for simplicity.

theorem chi_square_relationship (X Y : Type*) (data : X → Y → ℝ) :
  ( ∀ Χ2 : ℝ, Χ2 = chi_square_statistic data →
  (Χ2 = 0 → ∃ (credible : Prop), ¬credible)) → 
  (Χ2 > 0 → ∃ (credible : Prop), credible) :=
sorry

end chi_square_relationship_l86_86423


namespace solution_of_xyz_l86_86631

theorem solution_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) : x + y + z = 48 := 
sorry

end solution_of_xyz_l86_86631


namespace find_multiple_l86_86938

-- Definitions based on the problem's conditions
def n_drunk_drivers : ℕ := 6
def total_students : ℕ := 45
def num_speeders (M : ℕ) : ℕ := M * n_drunk_drivers - 3

-- The theorem that we need to prove
theorem find_multiple (M : ℕ) (h1: total_students = n_drunk_drivers + num_speeders M) : M = 7 :=
by
  sorry

end find_multiple_l86_86938


namespace cyclic_quadrilateral_sides_equal_l86_86305

theorem cyclic_quadrilateral_sides_equal
  (A B C D P : ℝ) -- Points represented as reals for simplicity
  (AB CD BC AD : ℝ) -- Lengths of sides AB, CD, BC, AD
  (a b c d e θ : ℝ) -- Various lengths and angle as given in the solution
  (h1 : a + e = b + c + d)
  (h2 : (1 / 2) * a * e * Real.sin θ = (1 / 2) * b * e * Real.sin θ + (1 / 2) * c * d * Real.sin θ) :
  c = e ∨ d = e := sorry

end cyclic_quadrilateral_sides_equal_l86_86305


namespace sin_C_of_right_triangle_l86_86482

theorem sin_C_of_right_triangle (A B C: ℝ) (sinA: ℝ) (sinB: ℝ) (sinC: ℝ) :
  (sinA = 8/17) →
  (sinB = 1) →
  (A + B + C = π) →
  (B = π / 2) →
  (sinC = 15/17) :=
  
by
  intro h_sinA h_sinB h_triangle h_B
  sorry -- Proof is not required

end sin_C_of_right_triangle_l86_86482


namespace maximum_value_of_a_l86_86746

theorem maximum_value_of_a {x y a : ℝ} (hx : x > 1 / 3) (hy : y > 1) :
  (∀ x y, x > 1 / 3 → y > 1 → 9 * x^2 / (a^2 * (y - 1)) + y^2 / (a^2 * (3 * x - 1)) ≥ 1)
  ↔ a ≤ 2 * Real.sqrt 2 :=
sorry

end maximum_value_of_a_l86_86746


namespace expression_divisible_by_1897_l86_86311

theorem expression_divisible_by_1897 (n : ℕ) :
  1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end expression_divisible_by_1897_l86_86311


namespace George_says_365_l86_86535

-- Definitions based on conditions
def skips_Alice (n : Nat) : Prop :=
  ∃ k, n = 3 * k - 1

def skips_Barbara (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * k - 1) - 1
  
def skips_Candice (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * k - 1) - 1) - 1

def skips_Debbie (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1

def skips_Eliza (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1) - 1

def skips_Fatima (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1) - 1) - 1

def numbers_said_by_students (n : Nat) : Prop :=
  skips_Alice n ∨ skips_Barbara n ∨ skips_Candice n ∨ skips_Debbie n ∨ skips_Eliza n ∨ skips_Fatima n

-- The proof statement
theorem George_says_365 : ¬numbers_said_by_students 365 :=
sorry

end George_says_365_l86_86535


namespace pie_shop_revenue_l86_86382

def costPerSlice : Int := 5
def slicesPerPie : Int := 4
def piesSold : Int := 9

theorem pie_shop_revenue : (costPerSlice * slicesPerPie * piesSold) = 180 := 
by
  sorry

end pie_shop_revenue_l86_86382


namespace jennifer_fish_tank_problem_l86_86944

theorem jennifer_fish_tank_problem :
  let built_tanks := 3
  let fish_per_built_tank := 15
  let planned_tanks := 3
  let fish_per_planned_tank := 10
  let total_built_fish := built_tanks * fish_per_built_tank
  let total_planned_fish := planned_tanks * fish_per_planned_tank
  let total_fish := total_built_fish + total_planned_fish
  total_fish = 75 := by
    let built_tanks := 3
    let fish_per_built_tank := 15
    let planned_tanks := 3
    let fish_per_planned_tank := 10
    let total_built_fish := built_tanks * fish_per_built_tank
    let total_planned_fish := planned_tanks * fish_per_planned_tank
    let total_fish := total_built_fish + total_planned_fish
    have h₁ : total_built_fish = 45 := by sorry
    have h₂ : total_planned_fish = 30 := by sorry
    have h₃ : total_fish = 75 := by sorry
    exact h₃

end jennifer_fish_tank_problem_l86_86944


namespace weight_of_seventh_person_l86_86832

noncomputable def weight_of_six_people : ℕ := 6 * 156
noncomputable def new_average_weight (x : ℕ) : Prop := (weight_of_six_people + x) / 7 = 151

theorem weight_of_seventh_person (x : ℕ) (h : new_average_weight x) : x = 121 :=
by
  sorry

end weight_of_seventh_person_l86_86832


namespace gianna_saved_for_365_days_l86_86115

-- Define the total amount saved and the amount saved each day
def total_amount_saved : ℕ := 14235
def amount_saved_each_day : ℕ := 39

-- Define the problem statement to prove the number of days saved
theorem gianna_saved_for_365_days :
  (total_amount_saved / amount_saved_each_day) = 365 :=
sorry

end gianna_saved_for_365_days_l86_86115


namespace smallest_prime_less_than_square_l86_86355

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end smallest_prime_less_than_square_l86_86355


namespace find_value_of_x_l86_86850

theorem find_value_of_x (x : ℝ) : (45 * x = 0.4 * 900) -> x = 8 :=
by
  intro h
  sorry

end find_value_of_x_l86_86850


namespace reciprocal_of_neg_2023_l86_86994

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86994


namespace mindy_emails_l86_86955

theorem mindy_emails (P E : ℕ) 
    (h1 : E = 9 * P - 7)
    (h2 : E + P = 93) :
    E = 83 := 
    sorry

end mindy_emails_l86_86955


namespace min_students_in_group_l86_86200

theorem min_students_in_group 
  (g1 g2 : ℕ) 
  (n1 n2 e1 e2 f1 f2 : ℕ)
  (H_equal_groups : g1 = g2)
  (H_both_languages_g1 : n1 = 5)
  (H_both_languages_g2 : n2 = 5)
  (H_french_students : f1 * 3 = f2)
  (H_english_students : e1 = 4 * e2)
  (H_total_g1 : g1 = f1 + e1 - n1)
  (H_total_g2 : g2 = f2 + e2 - n2) 
: g1 = 28 :=
sorry

end min_students_in_group_l86_86200


namespace find_n_if_pow_eqn_l86_86922

theorem find_n_if_pow_eqn (n : ℕ) :
  6 ^ 3 = 9 ^ n → n = 3 :=
by 
  sorry

end find_n_if_pow_eqn_l86_86922


namespace analyze_properties_l86_86961

noncomputable def eq_condition (x a : ℝ) : Prop :=
x ≠ 0 ∧ a = (x - 1) / (x^2)

noncomputable def first_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x = 1

noncomputable def second_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x > 1

noncomputable def third_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x < 1

theorem analyze_properties (x a : ℝ) (h1 : eq_condition x a):
(first_condition x a) ∧ ¬(second_condition x a) ∧ ¬(third_condition x a) :=
by
  sorry

end analyze_properties_l86_86961


namespace marcus_batches_l86_86017

theorem marcus_batches (B : ℕ) : (5 * B = 35) ∧ (35 - 8 = 27) → B = 7 :=
by {
  sorry
}

end marcus_batches_l86_86017


namespace reciprocal_of_negative_2023_l86_86984

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l86_86984


namespace sin_225_correct_l86_86878

-- Define the condition of point being on the unit circle at 225 degrees.
noncomputable def P_225 := Complex.polar 1 (Real.pi + Real.pi / 4)

-- Define the goal statement that translates the question and correct answer.
theorem sin_225_correct : Complex.sin (Real.pi + Real.pi / 4) = -Real.sqrt 2 / 2 := 
by sorry

end sin_225_correct_l86_86878


namespace find_m_l86_86113

theorem find_m (m a : ℝ) (h : (2:ℝ) * 1^2 - 3 * 1 + a = 0) 
  (h_roots : ∀ x : ℝ, 2 * x^2 - 3 * x + a = 0 → (x = 1 ∨ x = m)) :
  m = 1 / 2 :=
by
  sorry

end find_m_l86_86113


namespace sequences_meet_at_2017_l86_86306

-- Define the sequences for Paul and Penny
def paul_sequence (n : ℕ) : ℕ := 3 * n - 2
def penny_sequence (m : ℕ) : ℕ := 2022 - 5 * m

-- Statement to be proven
theorem sequences_meet_at_2017 : ∃ n m : ℕ, paul_sequence n = 2017 ∧ penny_sequence m = 2017 := by
  sorry

end sequences_meet_at_2017_l86_86306


namespace expected_number_of_matches_variance_of_number_of_matches_l86_86849

-- Defining the conditions first, and then posing the proof statements
namespace MatchingPairs

open ProbabilityTheory

-- Probabilistic setup for indicator variables
variable (N : ℕ) (prob : ℝ := 1 / N)

-- Indicator variable Ik representing matches
@[simp] def I (k : ℕ) : ℝ := if k < N then prob else 0

-- Define the sum of expected matches S
@[simp] def S : ℝ := ∑ k in finset.range N, I N k

-- Statement: The expectation of the number of matching pairs is 1
theorem expected_number_of_matches : E[S] = 1 := sorry

-- Statement: The variance of the number of matching pairs is 1
theorem variance_of_number_of_matches : Var S = 1 := sorry

end MatchingPairs

end expected_number_of_matches_variance_of_number_of_matches_l86_86849


namespace find_x_ineq_solution_l86_86735

open Set

theorem find_x_ineq_solution :
  {x : ℝ | (x - 2) / (x - 4) ≥ 3} = Ioc 4 5 := 
sorry

end find_x_ineq_solution_l86_86735


namespace sum_of_eight_numbers_l86_86515

theorem sum_of_eight_numbers (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_eight_numbers_l86_86515


namespace line_does_not_pass_through_third_quadrant_l86_86645

def line (x : ℝ) : ℝ := -x + 1

-- A line passes through the point (1, 0) and has a slope of -1
def passes_through_point (P : ℝ × ℝ) : Prop :=
  ∃ m b, m = -1 ∧ P.2 = m * P.1 + b ∧ line P.1 = P.2

def third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem line_does_not_pass_through_third_quadrant :
  ¬ ∃ p : ℝ × ℝ, passes_through_point p ∧ third_quadrant p :=
sorry

end line_does_not_pass_through_third_quadrant_l86_86645


namespace binary_to_decimal_and_octal_conversion_l86_86092

-- Definition of the binary number in question
def bin_num : ℕ := 0b1011

-- The expected decimal equivalent
def dec_num : ℕ := 11

-- The expected octal equivalent
def oct_num : ℤ := 0o13

-- Proof problem statement
theorem binary_to_decimal_and_octal_conversion :
  bin_num = dec_num ∧ dec_num = oct_num := 
by 
  sorry

end binary_to_decimal_and_octal_conversion_l86_86092


namespace problem_l86_86130

noncomputable def f (x : ℝ) (a b : ℝ) := (b - 2^x) / (2^(x+1) + a)

theorem problem (a b k : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) →
  (f 0 a b = 0) → (f (-1) a b = -f 1 a b) → 
  a = 2 ∧ b = 1 ∧ 
  (∀ x y : ℝ, x < y → f x a b > f y a b) ∧ 
  (∀ x : ℝ, x ≥ 1 → f (k * 3^x) a b + f (3^x - 9^x + 2) a b > 0 → k < 4 / 3) :=
by
  sorry

end problem_l86_86130


namespace line_through_point_parallel_l86_86678

/-
Given the point P(2, 0) and a line x - 2y + 3 = 0,
prove that the equation of the line passing through 
P and parallel to the given line is 2y - x + 2 = 0.
-/
theorem line_through_point_parallel
  (P : ℝ × ℝ)
  (x y : ℝ)
  (line_eq : x - 2*y + 3 = 0)
  (P_eq : P = (2, 0)) :
  ∃ (a b c : ℝ), a * y - b * x + c = 0 :=
sorry

end line_through_point_parallel_l86_86678


namespace gcd_seq_l86_86156

noncomputable theory
open Polynomial

variables {R : Type*} [CommRing R]

def a_seq (P : R[X]) (a : ℕ → R) : Prop :=
a 0 = 0 ∧ ∀ n > 0, a n = eval (a (n - 1)) P

theorem gcd_seq (P : ℤ[X]) (hP : ∀ x ≥ 0, eval x P > 0) (a : ℕ → ℤ) (hseq : a_seq P a)
(m n : ℕ) (hm : m > 0) (hn : n > 0) :
∃ d, d = Nat.gcd m n ∧ Nat.gcd (a m) (a n) = a d :=
sorry

end gcd_seq_l86_86156


namespace blocks_added_l86_86391

theorem blocks_added (original_blocks new_blocks added_blocks : ℕ) 
  (h1 : original_blocks = 35) 
  (h2 : new_blocks = 65) 
  (h3 : new_blocks = original_blocks + added_blocks) : 
  added_blocks = 30 :=
by
  -- We use the given conditions to prove the statement
  sorry

end blocks_added_l86_86391


namespace reciprocal_of_neg_2023_l86_86973

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l86_86973


namespace appropriate_speech_length_l86_86153

-- Condition 1: Speech duration in minutes
def speech_duration_min : ℝ := 30
def speech_duration_max : ℝ := 45

-- Condition 2: Ideal rate of speech in words per minute
def ideal_rate : ℝ := 150

-- Question translated into Lean proof statement
theorem appropriate_speech_length (n : ℝ) (h : n = 5650) :
  speech_duration_min * ideal_rate ≤ n ∧ n ≤ speech_duration_max * ideal_rate :=
by
  sorry

end appropriate_speech_length_l86_86153


namespace sample_size_l86_86972

variable (x n : ℕ)

-- Conditions as definitions
def staff_ratio : Prop := 15 * x + 3 * x + 2 * x = 20 * x
def sales_staff : Prop := 30 / n = 15 / 20

-- Main statement to prove
theorem sample_size (h1: staff_ratio x) (h2: sales_staff n) : n = 40 := by
  sorry

end sample_size_l86_86972


namespace range_of_a_in_fourth_quadrant_l86_86434

-- Define the fourth quadrant condition
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Define the point P(a+1, a-1) and state the theorem
theorem range_of_a_in_fourth_quadrant (a : ℝ) :
  in_fourth_quadrant (a + 1) (a - 1) → -1 < a ∧ a < 1 :=
by
  intro h
  have h1 : a + 1 > 0 := h.1
  have h2 : a - 1 < 0 := h.2
  have h3 : a > -1 := by linarith
  have h4 : a < 1 := by linarith
  exact ⟨h3, h4⟩

end range_of_a_in_fourth_quadrant_l86_86434


namespace B_pow_2021_eq_B_l86_86008

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![1 / 2, 0, -Real.sqrt 3 / 2],
  ![0, -1, 0],
  ![Real.sqrt 3 / 2, 0, 1 / 2]
]

theorem B_pow_2021_eq_B : B ^ 2021 = B := 
by sorry

end B_pow_2021_eq_B_l86_86008


namespace suraya_picked_more_apples_l86_86030

theorem suraya_picked_more_apples (suraya caleb kayla : ℕ) 
  (h1 : suraya = caleb + 12)
  (h2 : caleb = kayla - 5)
  (h3 : kayla = 20) : suraya - kayla = 7 := by
  sorry

end suraya_picked_more_apples_l86_86030


namespace equal_if_fraction_is_positive_integer_l86_86654

theorem equal_if_fraction_is_positive_integer
  (a b : ℕ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (K : ℝ := Real.sqrt ((a^2 + b^2:ℕ)/2))
  (A : ℝ := (a + b:ℕ)/2)
  (h_int_pos : ∃ (n : ℕ), n > 0 ∧ K / A = n) :
  a = b := sorry

end equal_if_fraction_is_positive_integer_l86_86654


namespace left_square_side_length_l86_86647

theorem left_square_side_length 
  (x y z : ℝ)
  (H1 : y = x + 17)
  (H2 : z = x + 11)
  (H3 : x + y + z = 52) : 
  x = 8 := by
  sorry

end left_square_side_length_l86_86647


namespace remainder_7_pow_137_mod_11_l86_86826

theorem remainder_7_pow_137_mod_11 :
    (137 = 13 * 10 + 7) →
    (7^10 ≡ 1 [MOD 11]) →
    (7^137 ≡ 6 [MOD 11]) :=
by
  intros h1 h2
  sorry

end remainder_7_pow_137_mod_11_l86_86826


namespace nested_f_has_zero_l86_86498

def f (x : ℝ) : ℝ := x^2 + 2017 * x + 1

theorem nested_f_has_zero (n : ℕ) (hn : n ≥ 1) : ∃ x : ℝ, (Nat.iterate f n x) = 0 :=
by
  sorry

end nested_f_has_zero_l86_86498


namespace correct_angle_calculation_l86_86911

theorem correct_angle_calculation (α β : ℝ) (hα : 0 < α ∧ α < 90) (hβ : 90 < β ∧ β < 180) :
    22.5 < 0.25 * (α + β) ∧ 0.25 * (α + β) < 67.5 → 0.25 * (α + β) = 45.3 :=
by
  sorry

end correct_angle_calculation_l86_86911


namespace bacteria_growth_returns_six_l86_86142

theorem bacteria_growth_returns_six (n : ℕ) (h : (4 * 2 ^ n > 200)) : n = 6 :=
sorry

end bacteria_growth_returns_six_l86_86142


namespace find_y_l86_86431

noncomputable def imaginary_unit : ℂ := Complex.I

noncomputable def z1 (y : ℝ) : ℂ := 3 + y * imaginary_unit

noncomputable def z2 : ℂ := 2 - imaginary_unit

theorem find_y (y : ℝ) (h : z1 y / z2 = 1 + imaginary_unit) : y = 1 :=
by
  sorry

end find_y_l86_86431


namespace middle_digit_base5_l86_86219

theorem middle_digit_base5 {M : ℕ} (x y z : ℕ) (hx : 0 ≤ x ∧ x < 5) (hy : 0 ≤ y ∧ y < 5) (hz : 0 ≤ z ∧ z < 5)
    (h_base5 : M = 25 * x + 5 * y + z) (h_base8 : M = 64 * z + 8 * y + x) : y = 0 :=
sorry

end middle_digit_base5_l86_86219


namespace worker_and_robot_capacity_additional_workers_needed_l86_86778

-- Definitions and conditions
def worker_capacity (x : ℕ) : Prop :=
  (1 : ℕ) * x + 420 = 420 + x

def time_equivalence (x : ℕ) : Prop :=
  900 * 10 * x = 600 * (x + 420)

-- First part of the proof problem
theorem worker_and_robot_capacity (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  x = 30 ∧ x + 420 = 450 :=
by
  sorry

-- Second part of the proof problem
theorem additional_workers_needed (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  3 * (x + 420) * 2 < 3600 →
  2 * 30 * 15 ≥ 3600 - 2 * 3 * (x + 420) :=
by
  sorry

end worker_and_robot_capacity_additional_workers_needed_l86_86778


namespace number_of_white_balls_l86_86146

theorem number_of_white_balls (x : ℕ) (h : (x : ℚ) / (x + 12) = 2 / 3) : x = 24 :=
sorry

end number_of_white_balls_l86_86146


namespace machine_minutes_worked_l86_86872

theorem machine_minutes_worked {x : ℕ} 
  (h_rate : ∀ y : ℕ, 6 * y = number_of_shirts_machine_makes_yesterday)
  (h_today : 14 = number_of_shirts_machine_makes_today)
  (h_total : number_of_shirts_machine_makes_yesterday + number_of_shirts_machine_makes_today = 156) : 
  x = 23 :=
by
  sorry

end machine_minutes_worked_l86_86872


namespace discount_is_one_percent_l86_86720

/-
  Assuming the following:
  - market_price is the price of one pen in dollars.
  - num_pens is the number of pens bought.
  - cost_price is the total cost price paid by the retailer.
  - profit_percentage is the profit made by the retailer.
  We need to prove that the discount percentage is 1.
-/

noncomputable def discount_percentage
  (market_price : ℝ)
  (num_pens : ℕ)
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (SP_per_pen : ℝ) : ℝ :=
  ((market_price - SP_per_pen) / market_price) * 100

theorem discount_is_one_percent
  (market_price : ℝ)
  (num_pens : ℕ)
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (buying_condition : cost_price = (market_price * num_pens * (36 / 60)))
  (SP : ℝ)
  (selling_condition : SP = cost_price * (1 + profit_percentage / 100))
  (SP_per_pen : ℝ)
  (sp_per_pen_condition : SP_per_pen = SP / num_pens)
  (profit_condition : profit_percentage = 65) :
  discount_percentage market_price num_pens cost_price profit_percentage SP_per_pen = 1 := by
  sorry

end discount_is_one_percent_l86_86720


namespace gingerbreads_per_tray_l86_86733

theorem gingerbreads_per_tray (x : ℕ) (h : 4 * x + 3 * 20 = 160) : x = 25 :=
by
  sorry

end gingerbreads_per_tray_l86_86733


namespace dragons_legs_l86_86729

theorem dragons_legs :
  ∃ (n : ℤ), ∀ (x y : ℤ), x + 3 * y = 26
                       → 40 * x + n * y = 298
                       → n = 14 :=
by
  sorry

end dragons_legs_l86_86729


namespace second_order_arithmetic_sequence_20th_term_l86_86323

theorem second_order_arithmetic_sequence_20th_term :
  (∀ a : ℕ → ℕ,
    a 1 = 1 ∧
    a 2 = 4 ∧
    a 3 = 9 ∧
    a 4 = 16 ∧
    (∀ n, 2 ≤ n → a n - a (n - 1) = 2 * n - 1) →
    a 20 = 400) :=
by 
  sorry

end second_order_arithmetic_sequence_20th_term_l86_86323


namespace evaluate_expression_l86_86097

theorem evaluate_expression (x : ℝ) : (x+2)^2 + 2*(x+2)*(4-x) + (4-x)^2 = 36 :=
by sorry

end evaluate_expression_l86_86097


namespace find_theta_l86_86820

variables (R h θ : ℝ)
hypothesis h1 : (r₁ = R * cos θ)
hypothesis h2 : (r₂ = (R + h) * cos θ)
hypothesis h3 : (s = 2 * π * r₂ - 2 * π * r₁)
hypothesis h4 : (s = 2 * π * h * cos θ)
hypothesis h5 : (s = h)

theorem find_theta : θ = real.arccos (1 / (2 * π)) :=
by
  sorry

end find_theta_l86_86820


namespace repeating_decimals_sum_l86_86102

noncomputable def repeating_decimals_sum_as_fraction : ℚ :=
  let d1 := "0.333333..." -- Represents 0.\overline{3}
  let d2 := "0.020202..." -- Represents 0.\overline{02}
  let sum := "0.353535..." -- Represents 0.\overline{35}
  by sorry

theorem repeating_decimals_sum (d1 d2 : ℚ)
  (h1 : d1 = 0.\overline{3})
  (h2 : d2 = 0.\overline{02}) :
  d1 + d2 = (35 / 99) := by sorry

end repeating_decimals_sum_l86_86102


namespace solve_for_real_a_l86_86123

theorem solve_for_real_a (a : ℝ) (i : ℂ) (h : i^2 = -1) (h1 : (a - i)^2 = 2 * i) : a = -1 :=
by sorry

end solve_for_real_a_l86_86123


namespace quadratic_has_distinct_real_roots_l86_86383

theorem quadratic_has_distinct_real_roots :
  let a := 5
  let b := 14
  let c := 5
  let discriminant := b^2 - 4 * a * c
  discriminant > 0 := 
by
  sorry

end quadratic_has_distinct_real_roots_l86_86383


namespace fraction_sum_of_two_reciprocals_fraction_sum_of_equal_reciprocals_fraction_difference_of_two_reciprocals_l86_86172

theorem fraction_sum_of_two_reciprocals (n : ℕ) (hn : n > 0) : 
  ∃ a b : ℕ, (a ≠ b) ∧ (3 * 5 * n * (a + b) = a * b) :=
sorry

theorem fraction_sum_of_equal_reciprocals (n : ℕ) : 
  ∃ a : ℕ, 3 * 5 * n * 2 = a * a ↔ (∃ k : ℕ, n = 2 * k) :=
sorry

theorem fraction_difference_of_two_reciprocals (n : ℕ) (hn : n > 0) : 
  ∃ a b : ℕ, (a ≠ b) ∧ 3 * 5 * n * (a - b) = a * b :=
sorry

end fraction_sum_of_two_reciprocals_fraction_sum_of_equal_reciprocals_fraction_difference_of_two_reciprocals_l86_86172


namespace christineTravelDistance_l86_86407

-- Definition of Christine's speed and time
def christineSpeed : ℝ := 20
def christineTime : ℝ := 4

-- Theorem to prove the distance Christine traveled
theorem christineTravelDistance : christineSpeed * christineTime = 80 := by
  -- The proof is omitted
  sorry

end christineTravelDistance_l86_86407


namespace ratio_of_areas_l86_86205

theorem ratio_of_areas (r : ℝ) (s1 s2 : ℝ) 
  (h1 : s1^2 = 4 / 5 * r^2)
  (h2 : s2^2 = 2 * r^2) :
  (s1^2 / s2^2) = 2 / 5 := by
  sorry

end ratio_of_areas_l86_86205


namespace floor_sqrt_20_squared_l86_86098

theorem floor_sqrt_20_squared : (⌊real.sqrt 20⌋ : ℝ)^2 = 16 :=
by
  have h1 : 4 < real.sqrt 20 := sorry
  have h2 : real.sqrt 20 < 5 := sorry
  have h3 : ⌊real.sqrt 20⌋ = 4 := sorry
  exact sorry

end floor_sqrt_20_squared_l86_86098


namespace sum_of_24_terms_l86_86646

variable (a_1 d : ℝ)

def a (n : ℕ) : ℝ := a_1 + (n - 1) * d

theorem sum_of_24_terms 
  (h : (a 5 + a 10 + a 15 + a 20 = 20)) : 
  (12 * (2 * a_1 + 23 * d) = 120) :=
by
  sorry

end sum_of_24_terms_l86_86646


namespace correct_representations_l86_86043

open Set

theorem correct_representations : 
  let S1 := {2, 3} ≠ ({3, 2} : Set ℕ)
  let S2 := ({(x, y) | x + y = 1} : Set (ℕ × ℕ)) = {y | ∃ x, x + y = 1}
  let S3 := ({x | x > 1} : Set ℕ) = {y | y > 1}
  let S4 := ({x | ∃ y, x + y = 1} : Set ℕ) = {y | ∃ x, x + y = 1}
  (¬S1 ∧ ¬S2 ∧ S3 ∧ S4) :=
by
  let S1 := {2, 3} ≠ ({3, 2} : Set ℕ)
  let S2 := ({(x, y) | x + y = 1} : Set (ℕ × ℕ)) = {y | ∃ x, x + y = 1}
  let S3 := ({x | x > 1} : Set ℕ) = {y | y > 1}
  let S4 := ({x | ∃ y, x + y = 1} : Set ℕ) = {y | ∃ x, x + y = 1}
  exact sorry

end correct_representations_l86_86043


namespace find_present_age_of_eldest_l86_86810

noncomputable def eldest_present_age (x : ℕ) : ℕ :=
  8 * x

theorem find_present_age_of_eldest :
  ∃ x : ℕ, 20 * x - 21 = 59 ∧ eldest_present_age x = 32 :=
by
  sorry

end find_present_age_of_eldest_l86_86810


namespace perfect_squares_50_to_200_l86_86455

theorem perfect_squares_50_to_200 : 
  ∃ (k : ℕ), k = 7 ∧ ∀ n : ℤ, 50 < n^2 ∧ n^2 < 200 -> (8 ≤ n ∧ n ≤ 14) := 
by
  sorry

end perfect_squares_50_to_200_l86_86455


namespace find_p_tilde_one_l86_86731

noncomputable def p (x : ℝ) : ℝ :=
  let r : ℝ := -1 / 9
  let s : ℝ := 1
  x^2 - (r + s) * x + (r * s)

theorem find_p_tilde_one : p 1 = 0 := by
  sorry

end find_p_tilde_one_l86_86731


namespace avg_of_sequence_is_x_l86_86677

noncomputable def sum_naturals (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem avg_of_sequence_is_x (x : ℝ) :
  let n := 100
  let sum := sum_naturals n
  (sum + x) / (n + 1) = 50 * x → 
  x = 5050 / 5049 :=
by
  intro n sum h
  exact sorry

end avg_of_sequence_is_x_l86_86677


namespace perfect_squares_count_between_50_and_200_l86_86446

theorem perfect_squares_count_between_50_and_200 : 
  let count := (λ n m : ℤ, n - m + 1) in
  ∃ n m : ℕ, 50 < n^2 ∧ n^2 < 200 ∧ 50 < m^2 ∧ m^2 < 200 ∧ count m n = 7 :=
begin
  sorry
end

end perfect_squares_count_between_50_and_200_l86_86446


namespace angle_bisectors_and_median_inequality_l86_86307

open Real

variables (A B C : Point)
variables (a b c : ℝ) -- sides of the triangle
variables (p : ℝ) -- semi-perimeter of the triangle
variables (la lb mc : ℝ) -- angle bisectors and median lengths

-- Assume the given conditions
axiom angle_bisector_la (A B C : Point) : ℝ -- lengths of the angle bisector of ∠BAC
axiom angle_bisector_lb (A B C : Point) : ℝ -- lengths of the angle bisector of ∠ABC
axiom median_mc (A B C : Point) : ℝ -- length of the median from vertex C
axiom semi_perimeter (a b c : ℝ) : ℝ -- semi-perimeter of the triangle

-- The statement of the theorem
theorem angle_bisectors_and_median_inequality (la lb mc p : ℝ) :
  la + lb + mc ≤ sqrt 3 * p :=
sorry

end angle_bisectors_and_median_inequality_l86_86307


namespace athlete_a_catches_up_and_race_duration_l86_86639

-- Track is 1000 meters
def track_length : ℕ := 1000

-- Athlete A's speed: first minute, increasing until 5th minute and decreasing until 600 meters/min
def athlete_A_speed (minute : ℕ) : ℕ :=
  match minute with
  | 0 => 1000
  | 1 => 1000
  | 2 => 1200
  | 3 => 1400
  | 4 => 1600
  | 5 => 1400
  | 6 => 1200
  | 7 => 1000
  | 8 => 800
  | 9 => 600
  | _ => 600

-- Athlete B's constant speed
def athlete_B_speed : ℕ := 1200

-- Function to compute distance covered in given minutes, assuming starts at 0
def total_distance (speed : ℕ → ℕ) (minutes : ℕ) : ℕ :=
  (List.range minutes).map speed |>.sum

-- Defining the maximum speed moment for A
def athlete_A_max_speed_distance : ℕ := total_distance athlete_A_speed 4
def athlete_B_max_speed_distance : ℕ := athlete_B_speed * 4

-- Proof calculation for target time 10 2/3 minutes
def time_catch : ℚ := 10 + 2 / 3

-- Defining the theorem to be proven
theorem athlete_a_catches_up_and_race_duration :
  athlete_A_max_speed_distance > athlete_B_max_speed_distance ∧ time_catch = 32 / 3 :=
by
  -- Place holder for the proof's details
  sorry

end athlete_a_catches_up_and_race_duration_l86_86639


namespace option_b_has_two_distinct_real_roots_l86_86393

def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  let Δ := b^2 - 4 * a * c
  Δ > 0

theorem option_b_has_two_distinct_real_roots :
  has_two_distinct_real_roots 1 (-2) (-3) :=
by
  sorry

end option_b_has_two_distinct_real_roots_l86_86393


namespace tin_to_copper_ratio_l86_86575

theorem tin_to_copper_ratio (L_A T_A T_B C_B : ℝ) 
  (h_total_mass_A : L_A + T_A = 90)
  (h_ratio_A : L_A / T_A = 3 / 4)
  (h_total_mass_B : T_B + C_B = 140)
  (h_total_tin : T_A + T_B = 91.42857142857143) :
  T_B / C_B = 2 / 5 :=
sorry

end tin_to_copper_ratio_l86_86575


namespace max_groups_l86_86327

-- Define the conditions
def valid_eq (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ (3 * a + b = 13)

-- The proof problem: No need for the proof body, just statement
theorem max_groups : ∃! (l : List (ℕ × ℕ)), (∀ ab ∈ l, valid_eq ab.fst ab.snd) ∧ l.length = 3 := sorry

end max_groups_l86_86327


namespace area_of_estate_l86_86861

theorem area_of_estate (side_length_in_inches : ℝ) (scale : ℝ) (real_side_length : ℝ) (area : ℝ) :
  side_length_in_inches = 12 →
  scale = 100 →
  real_side_length = side_length_in_inches * scale →
  area = real_side_length ^ 2 →
  area = 1440000 :=
by
  sorry

end area_of_estate_l86_86861


namespace worker_and_robot_capacity_additional_workers_needed_l86_86779

-- Definitions and conditions
def worker_capacity (x : ℕ) : Prop :=
  (1 : ℕ) * x + 420 = 420 + x

def time_equivalence (x : ℕ) : Prop :=
  900 * 10 * x = 600 * (x + 420)

-- First part of the proof problem
theorem worker_and_robot_capacity (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  x = 30 ∧ x + 420 = 450 :=
by
  sorry

-- Second part of the proof problem
theorem additional_workers_needed (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  3 * (x + 420) * 2 < 3600 →
  2 * 30 * 15 ≥ 3600 - 2 * 3 * (x + 420) :=
by
  sorry

end worker_and_robot_capacity_additional_workers_needed_l86_86779


namespace reciprocal_of_neg_2023_l86_86992

theorem reciprocal_of_neg_2023 : (-2023: ℝ) * (-1 / 2023) = 1 := 
by sorry

end reciprocal_of_neg_2023_l86_86992


namespace tan_double_angle_third_quadrant_l86_86262

open Real

theorem tan_double_angle_third_quadrant (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2) 
  (h2 : sin (π - α) = - (3 / 5)) : 
  tan (2 * α) = 24 / 7 :=
by
  sorry

end tan_double_angle_third_quadrant_l86_86262


namespace solution_1_solution_2_l86_86404

noncomputable def problem_1 : Real :=
  Real.log 25 + Real.log 2 * Real.log 50 + (Real.log 2)^2

noncomputable def problem_2 : Real :=
  (Real.logb 3 2 + Real.logb 9 2) * (Real.logb 4 3 + Real.logb 8 3)

theorem solution_1 : problem_1 = 2 := by
  sorry

theorem solution_2 : problem_2 = 5 / 4 := by
  sorry

end solution_1_solution_2_l86_86404


namespace percentage_difference_max_min_l86_86965

-- Definitions for the sector angles of each department
def angle_manufacturing := 162.0
def angle_sales := 108.0
def angle_research_and_development := 54.0
def angle_administration := 36.0

-- Full circle in degrees
def full_circle := 360.0

-- Compute the percentage representations of each department
def percentage_manufacturing := (angle_manufacturing / full_circle) * 100
def percentage_sales := (angle_sales / full_circle) * 100
def percentage_research_and_development := (angle_research_and_development / full_circle) * 100
def percentage_administration := (angle_administration / full_circle) * 100

-- Prove that the percentage difference between the department with the maximum and minimum number of employees is 35%
theorem percentage_difference_max_min : 
  percentage_manufacturing - percentage_administration = 35.0 :=
by
  -- placeholder for the actual proof
  sorry

end percentage_difference_max_min_l86_86965


namespace four_transformations_of_1989_l86_86329

-- Definition of the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Initial number
def initial_number : ℕ := 1989

-- Theorem statement
theorem four_transformations_of_1989 : 
  sum_of_digits (sum_of_digits (sum_of_digits (sum_of_digits initial_number))) = 9 :=
by
  sorry

end four_transformations_of_1989_l86_86329


namespace max_tan_B_minus_C_l86_86769

theorem max_tan_B_minus_C (A B C a b c : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C)
  (h_sides : a = 2 * b * Real.cos C - 3 * c * Real.cos B) :
  Real.tan (B - C) ≤ 3 / 4 := 
sorry

end max_tan_B_minus_C_l86_86769


namespace cards_sum_l86_86520

theorem cards_sum (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) :
  a + b + c + d + e + f + g + h = 21 :=
by
  sorry

end cards_sum_l86_86520


namespace growth_rate_equation_l86_86771

-- Given conditions
def revenue_january : ℕ := 36
def revenue_march : ℕ := 48

-- Problem statement
theorem growth_rate_equation (x : ℝ) 
  (h_january : revenue_january = 36)
  (h_march : revenue_march = 48) :
  36 * (1 + x) ^ 2 = 48 :=
sorry

end growth_rate_equation_l86_86771


namespace translated_function_l86_86703

def translate_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x, f (x + a)

def translate_down (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ :=
  λ x, f x - b

def translate_left_and_down (f : ℝ → ℝ) (a b : ℝ) : ℝ → ℝ :=
  translate_down (translate_left f a) b

theorem translated_function (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 2^x) →
  translate_left_and_down f 2 3 = λ x, 2^(x + 2) - 3 :=
by
  intro h
  funext x
  rw [translate_left_and_down, translate_left, translate_down]
  simp [h]
  sorry

end translated_function_l86_86703


namespace values_of_m_l86_86299

theorem values_of_m (m n : ℕ) (hmn : m * n = 900) (hm: m > 1) (hn: n ≥ 1) : 
  (∃ (k : ℕ), ∀ (m : ℕ), (1 < m ∧ (900 / m) ≥ 1 ∧ 900 % m = 0) ↔ k = 25) :=
sorry

end values_of_m_l86_86299


namespace sum_of_eight_numbers_on_cards_l86_86509

theorem sum_of_eight_numbers_on_cards :
  ∃ (a b c d e f g h : ℕ),
  (a + b) * (c + d) * (e + f) * (g + h) = 330 ∧
  (a + b + c + d + e + f + g + h) = 21 :=
by
  sorry

end sum_of_eight_numbers_on_cards_l86_86509


namespace geometric_sequence_common_ratio_l86_86484

variable {α : Type*} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) (q : α) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → α)
  (q : α)
  (h1 : is_geometric_sequence a q)
  (h2 : a 3 = 6)
  (h3 : a 0 + a 1 + a 2 = 18) :
  q = 1 ∨ q = - (1 / 2) := 
sorry

end geometric_sequence_common_ratio_l86_86484


namespace perfect_squares_count_between_50_and_200_l86_86461

theorem perfect_squares_count_between_50_and_200 :
  ∃ (N : ℕ), N = (finset.Ico 8 15).card ∧ N = 7 :=
by
  sorry

end perfect_squares_count_between_50_and_200_l86_86461
