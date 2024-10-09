import Mathlib

namespace find_special_number_l255_25551

theorem find_special_number:
  ∃ (n : ℕ), (n > 0) ∧ (∃ (k : ℕ), 2 * n = k^2)
           ∧ (∃ (m : ℕ), 3 * n = m^3)
           ∧ (∃ (p : ℕ), 5 * n = p^5)
           ∧ n = 1085 :=
by
  sorry

end find_special_number_l255_25551


namespace roots_of_equation_l255_25515

theorem roots_of_equation :
  ∀ x : ℝ, (21 / (x^2 - 9) - 3 / (x - 3) = 1) ↔ (x = 3 ∨ x = -7) :=
by
  sorry

end roots_of_equation_l255_25515


namespace right_triangle_angle_l255_25588

theorem right_triangle_angle {A B C : ℝ} (hABC : A + B + C = 180) (hC : C = 90) (hA : A = 70) : B = 20 :=
sorry

end right_triangle_angle_l255_25588


namespace find_n_l255_25532

theorem find_n (n : ℚ) (h : (1 / (n + 2) + 3 / (n + 2) + n / (n + 2) = 4)) : n = -4 / 3 :=
by
  sorry

end find_n_l255_25532


namespace find_x_l255_25528

-- Define the problem conditions.
def workers := ℕ
def gadgets := ℕ
def gizmos := ℕ
def hours := ℕ

-- Given conditions
def condition1 (g h : ℝ) := (1 / g = 2) ∧ (1 / h = 3)
def condition2 (g h : ℝ) := (100 * 3 / g = 900) ∧ (100 * 3 / h = 600)
def condition3 (x : ℕ) (g h : ℝ) := (40 * 4 / g = x) ∧ (40 * 4 / h = 480)

-- Proof problem statement
theorem find_x (g h : ℝ) (x : ℕ) : 
  condition1 g h → condition2 g h → condition3 x g h → x = 320 :=
by 
  intros h1 h2 h3
  sorry

end find_x_l255_25528


namespace find_the_number_l255_25581

-- Statement
theorem find_the_number (x : ℤ) (h : 2 * x = 3 * x - 25) : x = 25 :=
  sorry

end find_the_number_l255_25581


namespace arithmetic_sequence_problem_l255_25516

theorem arithmetic_sequence_problem 
  (a_n b_n : ℕ → ℕ) 
  (S_n T_n : ℕ → ℕ) 
  (h1: ∀ n, S_n n = (n * (a_n n + a_n (n-1))) / 2)
  (h2: ∀ n, T_n n = (n * (b_n n + b_n (n-1))) / 2)
  (h3: ∀ n, (S_n n) / (T_n n) = (7 * n + 2) / (n + 3)):
  (a_n 4) / (b_n 4) = 51 / 10 := 
sorry

end arithmetic_sequence_problem_l255_25516


namespace incorrect_median_l255_25525

def data_set : List ℕ := [7, 11, 10, 11, 6, 14, 11, 10, 11, 9]

noncomputable def median (l : List ℕ) : ℚ := 
  let sorted := l.toArray.qsort (· ≤ ·) 
  if sorted.size % 2 = 0 then
    (sorted.get! (sorted.size / 2 - 1) + sorted.get! (sorted.size / 2)) / 2
  else
    sorted.get! (sorted.size / 2)

theorem incorrect_median :
  median data_set ≠ 10 := by
  sorry

end incorrect_median_l255_25525


namespace find_ab_l255_25590

theorem find_ab
  (a b c : ℝ)
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 27)
  (h3 : a + b + c = 10)
  (h4 : a^3 - b^3 = 36)
  : a * b = -15 :=
by
  sorry

end find_ab_l255_25590


namespace least_common_multiple_135_195_l255_25539

def leastCommonMultiple (a b : ℕ) : ℕ :=
  Nat.lcm a b

theorem least_common_multiple_135_195 : leastCommonMultiple 135 195 = 1755 := by
  sorry

end least_common_multiple_135_195_l255_25539


namespace possible_values_of_x_l255_25586

-- Definitions representing the initial conditions
def condition1 (x : ℕ) : Prop := 203 % x = 13
def condition2 (x : ℕ) : Prop := 298 % x = 13

-- Main theorem statement
theorem possible_values_of_x (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 ∨ x = 95 := 
by
  sorry

end possible_values_of_x_l255_25586


namespace options_necessarily_positive_l255_25521

variable (x y z : ℝ)

theorem options_necessarily_positive (h₁ : -1 < x) (h₂ : x < 0) (h₃ : 0 < y) (h₄ : y < 1) (h₅ : 2 < z) (h₆ : z < 3) :
  y + x^2 * z > 0 ∧
  y + x^2 > 0 ∧
  y + y^2 > 0 ∧
  y + 2 * z > 0 := 
  sorry

end options_necessarily_positive_l255_25521


namespace determine_n_l255_25591

noncomputable def S : ℕ → ℝ := sorry -- define arithmetic series sum
noncomputable def a_1 : ℝ := sorry -- define first term
noncomputable def d : ℝ := sorry -- define common difference

axiom S_6 : S 6 = 36
axiom S_n {n : ℕ} (h : n > 0) : S n = 324
axiom S_n_minus_6 {n : ℕ} (h : n > 6) : S (n - 6) = 144

theorem determine_n (n : ℕ) (h : n > 0) : n = 18 := by {
  sorry
}

end determine_n_l255_25591


namespace no_valid_triangle_exists_l255_25576

-- Variables representing the sides and altitudes of the triangle
variables (a b c h_a h_b h_c : ℕ)

-- Definition of the perimeter condition
def perimeter_condition : Prop := a + b + c = 1995

-- Definition of integer altitudes condition (simplified)
def integer_altitudes_condition : Prop := 
  ∃ (h_a h_b h_c : ℕ), (h_a * 4 * a ^ 2 = 2 * a ^ 2 * b ^ 2 + 2 * a ^ 2 * c ^ 2 + 2 * c ^ 2 * b ^ 2 - a ^ 4 - b ^ 4 - c ^ 4)

-- The main theorem to prove no valid triangle exists
theorem no_valid_triangle_exists : ¬ (∃ (a b c : ℕ), perimeter_condition a b c ∧ integer_altitudes_condition a b c) :=
sorry

end no_valid_triangle_exists_l255_25576


namespace kyle_origami_stars_l255_25542

/-- Kyle bought 2 glass bottles, each can hold 15 origami stars,
    then bought another 3 identical glass bottles.
    Prove that the total number of origami stars needed to fill them is 75. -/
theorem kyle_origami_stars : (2 * 15) + (3 * 15) = 75 := by
  sorry

end kyle_origami_stars_l255_25542


namespace sum_of_solutions_eq_8_l255_25587

theorem sum_of_solutions_eq_8 :
    let a : ℝ := 1
    let b : ℝ := -8
    let c : ℝ := -26
    ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 0) ∧ (a * x2^2 + b * x2 + c = 0) →
      x1 + x2 = 8 :=
sorry

end sum_of_solutions_eq_8_l255_25587


namespace smallest_integer_N_l255_25507

theorem smallest_integer_N : ∃ (N : ℕ), 
  (∀ (a : ℕ → ℕ), ((∀ (i : ℕ), i < 125 -> a i > 0 ∧ a i ≤ N) ∧
  (∀ (i : ℕ), 1 ≤ i ∧ i < 124 → a i > (a (i - 1) + a (i + 1)) / 2) ∧
  (∀ (i j : ℕ), i < 125 ∧ j < 125 ∧ i ≠ j → a i ≠ a j)) → N = 2016) :=
sorry

end smallest_integer_N_l255_25507


namespace div_by_19_l255_25545

theorem div_by_19 (n : ℕ) : 19 ∣ (26^n - 7^n) :=
sorry

end div_by_19_l255_25545


namespace lollipop_distribution_l255_25579

theorem lollipop_distribution :
  let n := 42
  let initial_lollipops := 650
  let required_lollipops := n * (n + 1) / 2
  (required_lollipops - initial_lollipops) = 253 :=
by
  let n := 42
  let initial_lollipops := 650
  let required_lollipops := n * (n + 1) / 2
  have h : required_lollipops = 903 := by norm_num
  have h2 : (required_lollipops - initial_lollipops) = 253 := by norm_num
  exact h2

end lollipop_distribution_l255_25579


namespace profit_percent_l255_25504

theorem profit_percent (P C : ℝ) (h : (2 / 3) * P = 0.88 * C) : P - C = 0.32 * C → (P - C) / C * 100 = 32 := by
  sorry

end profit_percent_l255_25504


namespace ferries_are_divisible_by_4_l255_25537

theorem ferries_are_divisible_by_4 (t T : ℕ) (H : ∃ n : ℕ, T = n * t) :
  ∃ N : ℕ, N = 4 * (T / t) ∧ N % 4 = 0 :=
by
  sorry

end ferries_are_divisible_by_4_l255_25537


namespace pyramid_vertices_l255_25553

theorem pyramid_vertices (n : ℕ) (h : 2 * n = 14) : n + 1 = 8 :=
by {
  sorry
}

end pyramid_vertices_l255_25553


namespace distance_from_origin_to_point_l255_25524

def point : ℝ × ℝ := (12, -16)
def origin : ℝ × ℝ := (0, 0)

theorem distance_from_origin_to_point : 
  let (x1, y1) := origin
  let (x2, y2) := point 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 20 :=
by
  sorry

end distance_from_origin_to_point_l255_25524


namespace average_mileage_is_correct_l255_25552

noncomputable def total_distance : ℝ := 150 + 200
noncomputable def sedan_efficiency : ℝ := 25
noncomputable def truck_efficiency : ℝ := 15
noncomputable def sedan_miles : ℝ := 150
noncomputable def truck_miles : ℝ := 200

noncomputable def total_gas_used : ℝ := (sedan_miles / sedan_efficiency) + (truck_miles / truck_efficiency)
noncomputable def average_gas_mileage : ℝ := total_distance / total_gas_used

theorem average_mileage_is_correct :
  average_gas_mileage = 18.1 := 
by
  sorry

end average_mileage_is_correct_l255_25552


namespace difference_between_numbers_l255_25599

theorem difference_between_numbers :
  ∃ S : ℝ, L = 1650 ∧ L = 6 * S + 15 ∧ L - S = 1377.5 :=
sorry

end difference_between_numbers_l255_25599


namespace volume_of_rectangular_prism_l255_25558

-- Definition of the given conditions
variables (a b c : ℝ)

def condition1 : Prop := a * b = 24
def condition2 : Prop := b * c = 15
def condition3 : Prop := a * c = 10

-- The statement we want to prove
theorem volume_of_rectangular_prism
  (h1 : condition1 a b)
  (h2 : condition2 b c)
  (h3 : condition3 a c) :
  a * b * c = 60 :=
by sorry

end volume_of_rectangular_prism_l255_25558


namespace servings_of_peanut_butter_l255_25547

theorem servings_of_peanut_butter :
  let peanutButterInJar := 37 + 4 / 5
  let oneServing := 1 + 1 / 2
  let servings := 25 + 1 / 5
  (peanutButterInJar / oneServing) = servings :=
by
  let peanutButterInJar := 37 + 4 / 5
  let oneServing := 1 + 1 / 2
  let servings := 25 + 1 / 5
  sorry

end servings_of_peanut_butter_l255_25547


namespace length_of_side_divisible_by_4_l255_25577

theorem length_of_side_divisible_by_4 {m n : ℕ} 
  (h : ∀ k : ℕ, (m * k) + (n * k) % 4 = 0 ) : 
  m % 4 = 0 ∨ n % 4 = 0 :=
by
  sorry

end length_of_side_divisible_by_4_l255_25577


namespace gcd_204_85_l255_25520

theorem gcd_204_85: Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l255_25520


namespace negation_of_every_square_positive_l255_25564

theorem negation_of_every_square_positive :
  ¬(∀ n : ℕ, n^2 > 0) ↔ ∃ n : ℕ, n^2 ≤ 0 := sorry

end negation_of_every_square_positive_l255_25564


namespace license_plate_combinations_l255_25562

-- Definitions based on the conditions
def num_letters := 26
def num_digits := 10
def num_positions := 5

def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Main theorem statement
theorem license_plate_combinations :
  choose num_letters 2 * (num_letters - 2) * choose num_positions 2 * choose (num_positions - 2) 2 * num_digits * (num_digits - 1) * (num_digits - 2) = 7776000 :=
by
  sorry

end license_plate_combinations_l255_25562


namespace remainder_of_c_plus_d_l255_25550

-- Definitions based on conditions
def c (k : ℕ) : ℕ := 60 * k + 53
def d (m : ℕ) : ℕ := 40 * m + 29

-- Statement of the problem
theorem remainder_of_c_plus_d (k m : ℕ) :
  ((c k + d m) % 20) = 2 :=
by
  unfold c
  unfold d
  sorry

end remainder_of_c_plus_d_l255_25550


namespace sun_city_population_greater_than_twice_roseville_l255_25511

-- Conditions
def willowdale_population : ℕ := 2000
def roseville_population : ℕ := 3 * willowdale_population - 500
def sun_city_population : ℕ := 12000

-- Theorem
theorem sun_city_population_greater_than_twice_roseville :
  sun_city_population = 2 * roseville_population + 1000 :=
by
  -- The proof is omitted as per the problem statement
  sorry

end sun_city_population_greater_than_twice_roseville_l255_25511


namespace daisies_per_bouquet_is_7_l255_25512

/-
Each bouquet of roses contains 12 roses.
Each bouquet of daisies contains an equal number of daisies.
The flower shop sells 20 bouquets today.
10 of the bouquets are rose bouquets and 10 are daisy bouquets.
The flower shop sold 190 flowers in total today.
-/

def num_daisies_per_bouquet (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ) : ℕ :=
  (total_flowers_sold - total_roses_sold) / bouquets_sold 

theorem daisies_per_bouquet_is_7 :
  ∀ (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ),
  (roses_per_bouquet = 12) →
  (bouquets_sold = 10) →
  (total_roses_sold = bouquets_sold * roses_per_bouquet) →
  (total_flowers_sold = 190) →
  num_daisies_per_bouquet roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold = 7 :=
by
  intros
  -- Placeholder for the actual proof
  sorry

end daisies_per_bouquet_is_7_l255_25512


namespace compute_difference_of_reciprocals_l255_25522

theorem compute_difference_of_reciprocals
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x / y) :
  (1 / x) - (1 / y) = - (1 / y^2) :=
by
  sorry

end compute_difference_of_reciprocals_l255_25522


namespace compare_M_N_l255_25565

variable (a : ℝ)

def M : ℝ := 2 * a^2 - 4 * a
def N : ℝ := a^2 - 2 * a - 3

theorem compare_M_N : M a > N a := by
  sorry

end compare_M_N_l255_25565


namespace g_possible_values_l255_25585

noncomputable def g (x y z : ℝ) : ℝ :=
  (x + y) / x + (y + z) / y + (z + x) / z

theorem g_possible_values (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  6 ≤ g x y z :=
by
  sorry

end g_possible_values_l255_25585


namespace committee_count_is_252_l255_25578

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Problem statement: Number of ways to choose a 5-person committee from a club of 10 people is 252 -/
theorem committee_count_is_252 : binom 10 5 = 252 :=
by
  sorry

end committee_count_is_252_l255_25578


namespace second_chapter_pages_l255_25572

/-- A book has 2 chapters across 81 pages. The first chapter is 13 pages long. -/
theorem second_chapter_pages (total_pages : ℕ) (first_chapter_pages : ℕ) (second_chapter_pages : ℕ) : 
  total_pages = 81 → 
  first_chapter_pages = 13 → 
  second_chapter_pages = total_pages - first_chapter_pages → 
  second_chapter_pages = 68 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end second_chapter_pages_l255_25572


namespace max_bottles_drunk_l255_25535

theorem max_bottles_drunk (e b : ℕ) (h1 : e = 16) (h2 : b = 4) : 
  ∃ n : ℕ, n = 5 :=
by
  sorry

end max_bottles_drunk_l255_25535


namespace necessary_but_not_sufficient_l255_25556

theorem necessary_but_not_sufficient (x : ℝ) (h : x < 4) : x < 0 ∨ true :=
by
  sorry

end necessary_but_not_sufficient_l255_25556


namespace simplify_complex_fraction_l255_25574

theorem simplify_complex_fraction :
  (⟨-4, -6⟩ : ℂ) / (⟨5, -2⟩ : ℂ) = ⟨-(32 : ℚ) / 21, -(38 : ℚ) / 21⟩ := 
sorry

end simplify_complex_fraction_l255_25574


namespace lines_parallel_l255_25582

theorem lines_parallel 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : Real.log (Real.sin α) + Real.log (Real.sin γ) = 2 * Real.log (Real.sin β)) :
  (∀ x y : ℝ, ∀ a b c : ℝ, 
    (x * (Real.sin α)^2 + y * Real.sin α = a) → 
    (x * (Real.sin β)^2 + y * Real.sin γ = c) →
    (-Real.sin α = -((Real.sin β)^2 / Real.sin γ))) :=
sorry

end lines_parallel_l255_25582


namespace value_of_five_l255_25555

variable (f : ℝ → ℝ)

-- Conditions of the problem
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 2) = f (x)

theorem value_of_five (hf_odd : odd_function f) (hf_periodic : periodic_function f) : f 5 = 0 :=
by 
  sorry

end value_of_five_l255_25555


namespace circle_condition_l255_25598

def represents_circle (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x + 1/2)^2 + (y + m)^2 = 5/4 - m

theorem circle_condition (m : ℝ) : represents_circle m ↔ m < 5/4 :=
by sorry

end circle_condition_l255_25598


namespace solution_set_l255_25595

-- Define the system of equations
def system_of_equations (x y : ℤ) : Prop :=
  4 * x^2 = y^2 + 2 * y + 4 ∧
  (2 * x)^2 - (y + 1)^2 = 3 ∧
  (2 * x - (y + 1)) * (2 * x + (y + 1)) = 3

-- Prove that the solutions to the system are the set we expect
theorem solution_set : 
  { (x, y) : ℤ × ℤ | system_of_equations x y } = { (1, 0), (1, -2), (-1, 0), (-1, -2) } := 
by 
  -- Proof omitted
  sorry

end solution_set_l255_25595


namespace congruence_equivalence_l255_25546

theorem congruence_equivalence (m n a b : ℤ) (h_coprime : Int.gcd m n = 1) :
  a ≡ b [ZMOD m * n] ↔ (a ≡ b [ZMOD m] ∧ a ≡ b [ZMOD n]) :=
sorry

end congruence_equivalence_l255_25546


namespace Jolene_cars_washed_proof_l255_25501

-- Definitions for conditions
def number_of_families : ℕ := 4
def babysitting_rate : ℕ := 30 -- in dollars
def car_wash_rate : ℕ := 12 -- in dollars
def total_money_raised : ℕ := 180 -- in dollars

-- Mathematical representation of the problem:
def babysitting_earnings : ℕ := number_of_families * babysitting_rate
def earnings_from_cars : ℕ := total_money_raised - babysitting_earnings
def number_of_cars_washed : ℕ := earnings_from_cars / car_wash_rate

-- The proof statement
theorem Jolene_cars_washed_proof : number_of_cars_washed = 5 := 
sorry

end Jolene_cars_washed_proof_l255_25501


namespace domain_of_f_l255_25500

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

theorem domain_of_f :
  { x : ℝ | 2 * x - 1 > 0 } = { x : ℝ | x > 1 / 2 } :=
by
  sorry

end domain_of_f_l255_25500


namespace values_of_x_defined_l255_25544

noncomputable def problem_statement (x : ℝ) : Prop :=
  (2 * x - 3 > 0) ∧ (5 - 2 * x > 0)

theorem values_of_x_defined (x : ℝ) :
  problem_statement x ↔ (3 / 2 < x ∧ x < 5 / 2) :=
by sorry

end values_of_x_defined_l255_25544


namespace probability_computation_l255_25569

noncomputable def probability_two_equal_three : ℚ :=
  let p_one_digit : ℚ := 3 / 4
  let p_two_digit : ℚ := 1 / 4
  let number_of_dice : ℕ := 5
  let ways_to_choose_two_digit := Nat.choose number_of_dice 2
  ways_to_choose_two_digit * (p_two_digit^2) * (p_one_digit^3)

theorem probability_computation :
  probability_two_equal_three = 135 / 512 :=
by
  sorry

end probability_computation_l255_25569


namespace probability_at_least_three_aces_l255_25509

open Nat

noncomputable def combination (n k : ℕ) : ℕ :=
  n.choose k

theorem probability_at_least_three_aces :
  (combination 4 3 * combination 48 2 + combination 4 4 * combination 48 1) / combination 52 5 = (combination 4 3 * combination 48 2 + combination 4 4 * combination 48 1 : ℚ) / combination 52 5 :=
by
  sorry

end probability_at_least_three_aces_l255_25509


namespace investor_wait_time_l255_25541

noncomputable def compound_interest_time (P A r : ℝ) (n : ℕ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem investor_wait_time :
  compound_interest_time 600 661.5 0.10 2 = 1 := 
sorry

end investor_wait_time_l255_25541


namespace radius_of_third_circle_l255_25543

noncomputable def circle_radius {r1 r2 : ℝ} (h1 : r1 = 15) (h2 : r2 = 25) : ℝ :=
  let A_shaded := (25^2 * Real.pi) - (15^2 * Real.pi)
  let r := Real.sqrt (A_shaded / Real.pi)
  r

theorem radius_of_third_circle (r1 r2 r3 : ℝ) (h1 : r1 = 15) (h2 : r2 = 25) :
  circle_radius h1 h2 = 20 :=
by 
  sorry

end radius_of_third_circle_l255_25543


namespace find_profit_percentage_l255_25592

theorem find_profit_percentage (h : (m + 8) / (1 - 0.08) = m + 10) : m = 15 := sorry

end find_profit_percentage_l255_25592


namespace math_problem_l255_25533

theorem math_problem 
  (a b c d : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c ≥ d) 
  (h4 : d > 0) 
  (h5 : a + b + c + d = 1) : 
  (a + 2*b + 3*c + 4*d) * a^a * b^b * c^c * d^d < 1 := 
sorry

end math_problem_l255_25533


namespace alberto_biked_more_than_bjorn_l255_25563

-- Define the distances traveled by Bjorn and Alberto after 5 hours.
def b_distance : ℝ := 75
def a_distance : ℝ := 100

-- Statement to prove the distance difference after 5 hours.
theorem alberto_biked_more_than_bjorn : a_distance - b_distance = 25 := 
by
  -- Proof is skipped, focusing only on the statement.
  sorry

end alberto_biked_more_than_bjorn_l255_25563


namespace problem_1_a_problem_1_b_problem_2_l255_25560

def set_A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def set_B : Set ℝ := {x | 2 < x ∧ x < 9}
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}
def set_union (s₁ s₂ : Set ℝ) := {x | x ∈ s₁ ∨ x ∈ s₂}
def set_inter (s₁ s₂ : Set ℝ) := {x | x ∈ s₁ ∧ x ∈ s₂}

theorem problem_1_a :
  set_inter set_A set_B = {x : ℝ | 3 ≤ x ∧ x < 6} :=
sorry

theorem problem_1_b :
  set_union complement_B set_A = {x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
sorry

def set_C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem problem_2 (a : ℝ) :
  (set_C a ⊆ set_B) → (2 ≤ a ∧ a ≤ 8) :=
sorry

end problem_1_a_problem_1_b_problem_2_l255_25560


namespace evaluate_expression_l255_25584

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6

theorem evaluate_expression :
  ( (1 / a) + (1 / b) + (1 / c) + (1 / d) ) ^ 2 = 96 / 529 :=
by
  sorry

end evaluate_expression_l255_25584


namespace initial_bacteria_count_l255_25597

theorem initial_bacteria_count (d: ℕ) (t_final: ℕ) (N_final: ℕ) 
    (h1: t_final = 4 * 60)  -- 4 minutes equals 240 seconds
    (h2: d = 15)            -- Doubling interval is 15 seconds
    (h3: N_final = 2097152) -- Final bacteria count is 2,097,152
    :
    ∃ n: ℕ, N_final = n * 2^((t_final / d)) ∧ n = 32 :=
by
  sorry

end initial_bacteria_count_l255_25597


namespace large_green_curlers_l255_25508

-- Define the number of total curlers
def total_curlers : ℕ := 16

-- Define the fraction for pink curlers
def pink_fraction : ℕ := 1 / 4

-- Define the number of pink curlers
def pink_curlers : ℕ := pink_fraction * total_curlers

-- Define the number of blue curlers
def blue_curlers : ℕ := 2 * pink_curlers

-- Define the total number of pink and blue curlers
def pink_and_blue_curlers : ℕ := pink_curlers + blue_curlers

-- Define the number of green curlers
def green_curlers : ℕ := total_curlers - pink_and_blue_curlers

-- Theorem stating the number of green curlers is 4
theorem large_green_curlers : green_curlers = 4 := by
  -- Proof would go here
  sorry

end large_green_curlers_l255_25508


namespace inequality_2n_squared_plus_3n_plus_1_l255_25557

theorem inequality_2n_squared_plus_3n_plus_1 (n : ℕ) (h: n > 0) : (2 * n^2 + 3 * n + 1)^n ≥ 6^n * (n! * n!) := 
by sorry

end inequality_2n_squared_plus_3n_plus_1_l255_25557


namespace container_volume_ratio_l255_25568

theorem container_volume_ratio
  (A B : ℝ)
  (h : (5 / 6) * A = (3 / 4) * B) :
  (A / B = 9 / 10) :=
sorry

end container_volume_ratio_l255_25568


namespace solution_set_of_inequality_g_geq_2_l255_25575

-- Definition of the function f
def f (x a : ℝ) := |x - a|

-- Definition of the function g
def g (x a : ℝ) := f x a + f (x + 2) a

-- Proof Problem I
theorem solution_set_of_inequality (a : ℝ) (x : ℝ) :
  a = -1 → (f x a ≥ 4 - |2 * x - 1|) ↔ (x ≤ -4/3 ∨ x ≥ 4/3) :=
by sorry

-- Proof Problem II
theorem g_geq_2 (a : ℝ) (x : ℝ) :
  (∀ x, f x a ≤ 1 → (0 ≤ x ∧ x ≤ 2)) → a = 1 → g x a ≥ 2 :=
by sorry

end solution_set_of_inequality_g_geq_2_l255_25575


namespace proof_problem_l255_25502

theorem proof_problem (a b c d x : ℝ)
  (h1 : c = 6 * d)
  (h2 : 2 * a = 1 / (-b))
  (h3 : abs x = 9) :
  (2 * a * b - 6 * d + c - x / 3 = -4) ∨ (2 * a * b - 6 * d + c - x / 3 = 2) :=
by
  sorry

end proof_problem_l255_25502


namespace operation_4_3_is_5_l255_25526

def custom_operation (m n : ℕ) : ℕ := n ^ 2 - m

theorem operation_4_3_is_5 : custom_operation 4 3 = 5 :=
by
  -- Proof goes here
  sorry

end operation_4_3_is_5_l255_25526


namespace max_value_proof_l255_25510

noncomputable def max_value (x y z : ℝ) : ℝ :=
  1 / x + 2 / y + 3 / z

theorem max_value_proof (x y z : ℝ) (h1 : 2 / 5 ≤ z ∧ z ≤ min x y)
    (h2 : x * z ≥ 4 / 15) (h3 : y * z ≥ 1 / 5) : max_value x y z ≤ 13 := 
by
  sorry

end max_value_proof_l255_25510


namespace find_k_l255_25567

theorem find_k (k x y : ℕ) (h : k * 2 + 1 = 5) : k = 2 :=
by {
  -- Proof will go here
  sorry
}

end find_k_l255_25567


namespace probability_shaded_region_l255_25530

def triangle_game :=
  let total_regions := 6
  let shaded_regions := 3
  shaded_regions / total_regions

theorem probability_shaded_region:
  triangle_game = 1 / 2 := by
  sorry

end probability_shaded_region_l255_25530


namespace binary_to_base4_conversion_l255_25559

theorem binary_to_base4_conversion : ∀ (a b c d e : ℕ), 
  1101101101 = (11 * 2^8) + (01 * 2^6) + (10 * 2^4) + (11 * 2^2) + 01 -> 
  a = 3 -> b = 1 -> c = 2 -> d = 3 -> e = 1 -> 
  (a*10000 + b*1000 + c*100 + d*10 + e : ℕ) = 31131 :=
by
  -- proof will go here
  sorry

end binary_to_base4_conversion_l255_25559


namespace min_value_l255_25589

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y + 1 / x + 2 / y = 13 / 2) :
  x - 1 / y ≥ - 1 / 2 :=
sorry

end min_value_l255_25589


namespace janice_remaining_hours_l255_25513

def homework_time : ℕ := 30
def clean_room_time : ℕ := homework_time / 2
def walk_dog_time : ℕ := homework_time + 5
def trash_time : ℕ := homework_time / 6
def total_task_time : ℕ := homework_time + clean_room_time + walk_dog_time + trash_time
def remaining_minutes : ℕ := 35

theorem janice_remaining_hours : (remaining_minutes : ℚ) / 60 = (7 / 12 : ℚ) :=
by
  sorry

end janice_remaining_hours_l255_25513


namespace system_of_equations_solution_exists_l255_25519

theorem system_of_equations_solution_exists :
  ∃ (x y z : ℤ), 
    (x + y - 2018 = (y - 2019) * x) ∧
    (y + z - 2017 = (y - 2019) * z) ∧
    (x + z + 5 = x * z) ∧
    (x = 3 ∧ y = 2021 ∧ z = 4 ∨ 
    x = -1 ∧ y = 2019 ∧ z = -2) := 
sorry

end system_of_equations_solution_exists_l255_25519


namespace ratio_of_area_to_breadth_l255_25529

theorem ratio_of_area_to_breadth (b l A : ℝ) (h₁ : b = 10) (h₂ : l - b = 10) (h₃ : A = l * b) : A / b = 20 := 
by
  sorry

end ratio_of_area_to_breadth_l255_25529


namespace max_value_frac_sixth_roots_eq_two_l255_25549

noncomputable def max_value_frac_sixth_roots (α β : ℝ) (t : ℝ) (q : ℝ) : ℝ :=
  if α + β = t ∧ α^2 + β^2 = t ∧ α^3 + β^3 = t ∧ α^4 + β^4 = t ∧ α^5 + β^5 = t then
    max (1 / α^6 + 1 / β^6) 2
  else
    0

theorem max_value_frac_sixth_roots_eq_two (α β : ℝ) (t : ℝ) (q : ℝ) :
  (α + β = t ∧ α^2 + β^2 = t ∧ α^3 + β^3 = t ∧ α^4 + β^4 = t ∧ α^5 + β^5 = t) →
  ∃ m, max_value_frac_sixth_roots α β t q = m ∧ m = 2 :=
sorry

end max_value_frac_sixth_roots_eq_two_l255_25549


namespace find_a_l255_25561

noncomputable def f (x : ℝ) (a : ℝ) := (2 / x) - 2 + 2 * a * Real.log x

theorem find_a (a : ℝ) (h : ∃ x ∈ Set.Icc (1/2 : ℝ) 2, f x a = 0) : a = 1 := by
  sorry

end find_a_l255_25561


namespace function_domain_l255_25518

noncomputable def sqrt_domain : Set ℝ :=
  {x | x + 1 ≥ 0 ∧ 2 - x > 0 ∧ 2 - x ≠ 1}

theorem function_domain :
  sqrt_domain = {x | -1 ≤ x ∧ x < 1} ∪ {x | 1 < x ∧ x < 2} :=
by
  sorry

end function_domain_l255_25518


namespace equivalent_knicks_l255_25573

theorem equivalent_knicks (knicks knacks knocks : ℕ) (h1 : 5 * knicks = 3 * knacks) (h2 : 4 * knacks = 6 * knocks) :
  36 * knocks = 40 * knicks :=
by
  sorry

end equivalent_knicks_l255_25573


namespace proof_set_intersection_l255_25570

noncomputable def U := ℝ
noncomputable def M := {x : ℝ | 0 ≤ x ∧ x < 5}
noncomputable def N := {x : ℝ | x ≥ 2}
noncomputable def compl_U_N := {x : ℝ | x < 2}
noncomputable def intersection := { x : ℝ | 0 ≤ x ∧ x < 2 }

theorem proof_set_intersection : ((compl_U_N ∩ M) = {x : ℝ | 0 ≤ x ∧ x < 2}) :=
by
  sorry

end proof_set_intersection_l255_25570


namespace hall_volume_l255_25593

theorem hall_volume (length breadth : ℝ) (h : ℝ)
  (h_length : length = 15) (h_breadth : breadth = 12)
  (h_area : 2 * (length * breadth) = 2 * (breadth * h) + 2 * (length * h)) :
  length * breadth * h = 8004 := 
by
  -- Proof not required
  sorry

end hall_volume_l255_25593


namespace tan_15_deg_product_l255_25596

theorem tan_15_deg_product : (1 + Real.tan 15) * (1 + Real.tan 15) = 2.1433 := by
  sorry

end tan_15_deg_product_l255_25596


namespace min_n_Sn_l255_25531

/--
Given an arithmetic sequence {a_n}, let S_n denote the sum of its first n terms.
If S_4 = -2, S_5 = 0, and S_6 = 3, then the minimum value of n * S_n is -9.
-/
theorem min_n_Sn (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h₁ : S 4 = -2)
  (h₂ : S 5 = 0)
  (h₃ : S 6 = 3)
  (h₄ : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  : ∃ n : ℕ, n * S n = -9 := 
sorry

end min_n_Sn_l255_25531


namespace find_value_of_x_l255_25536

theorem find_value_of_x (x y z : ℤ) (h1 : x > y) (h2 : y > z) (h3 : z = 3)
  (h4 : 2 * x + 3 * y + 3 * z = 5 * y + 11) (h5 : (x = y + 1) ∧ (y = z + 1)) :
  x = 5 := 
sorry

end find_value_of_x_l255_25536


namespace sophomores_in_program_l255_25534

-- Define variables
variable (P S : ℕ)

-- Conditions for the problem
def total_students (P S : ℕ) : Prop := P + S = 36
def percent_sophomores_club (P S : ℕ) (x : ℕ) : Prop := x = 3 * P / 10
def percent_seniors_club (P S : ℕ) (y : ℕ) : Prop := y = S / 4
def equal_club_members (x y : ℕ) : Prop := x = y

-- Theorem stating the problem and proof goal
theorem sophomores_in_program
  (x y : ℕ)
  (h1 : total_students P S)
  (h2 : percent_sophomores_club P S x)
  (h3 : percent_seniors_club P S y)
  (h4 : equal_club_members x y) :
  P = 15 := 
sorry

end sophomores_in_program_l255_25534


namespace no_rational_xyz_satisfies_l255_25523

theorem no_rational_xyz_satisfies:
  ¬ ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
  (1 / (x - y) ^ 2 + 1 / (y - z) ^ 2 + 1 / (z - x) ^ 2 = 2014) :=
by
  -- The proof will go here
  sorry

end no_rational_xyz_satisfies_l255_25523


namespace point_after_rotation_l255_25571

-- Definitions based on conditions
def point_N : ℝ × ℝ := (-1, -2)
def origin_O : ℝ × ℝ := (0, 0)
def rotation_180 (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- The statement to be proved
theorem point_after_rotation :
  rotation_180 point_N = (1, 2) :=
by
  sorry

end point_after_rotation_l255_25571


namespace liters_conversion_hours_to_days_cubic_meters_to_cubic_cm_l255_25514

-- Define the conversions and the corresponding proofs
theorem liters_conversion : 8.32 = 8 + 320 / 1000 := sorry

theorem hours_to_days : 6 = 1 / 4 * 24 := sorry

theorem cubic_meters_to_cubic_cm : 0.75 * 10^6 = 750000 := sorry

end liters_conversion_hours_to_days_cubic_meters_to_cubic_cm_l255_25514


namespace jackson_money_proof_l255_25527

noncomputable def jackson_money (W : ℝ) := 7 * W
noncomputable def lucy_money (W : ℝ) := 3 * W
noncomputable def ethan_money (W : ℝ) := 3 * W + 20

theorem jackson_money_proof : ∀ (W : ℝ), (W + 7 * W + 3 * W + (3 * W + 20) = 600) → jackson_money W = 290.01 :=
by 
  intros W h
  have total_eq := h
  sorry

end jackson_money_proof_l255_25527


namespace tan_390_correct_l255_25503

-- We assume basic trigonometric functions and their properties
noncomputable def tan_390_equals_sqrt3_div3 : Prop :=
  Real.tan (390 * Real.pi / 180) = Real.sqrt 3 / 3

theorem tan_390_correct : tan_390_equals_sqrt3_div3 :=
  by
  -- Proof is omitted
  sorry

end tan_390_correct_l255_25503


namespace multiplicative_inverse_l255_25538

theorem multiplicative_inverse (a b n : ℤ) (h₁ : a = 208) (h₂ : b = 240) (h₃ : n = 307) : 
  (a * b) % n = 1 :=
by
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end multiplicative_inverse_l255_25538


namespace final_expression_simplified_l255_25517

variable (a : ℝ)

theorem final_expression_simplified : 
  (2 * a + 6 - 3 * a) / 2 = -a / 2 + 3 := 
by 
sorry

end final_expression_simplified_l255_25517


namespace find_m_l255_25580

theorem find_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x < 0) → m = -1 :=
by sorry

end find_m_l255_25580


namespace find_number_l255_25594

theorem find_number (x : ℝ) (h : 2 * x - 2.6 * 4 = 10) : x = 10.2 :=
sorry

end find_number_l255_25594


namespace base8_arithmetic_l255_25554

-- Define the numbers in base 8
def num1 : ℕ := 0o453
def num2 : ℕ := 0o267
def num3 : ℕ := 0o512
def expected_result : ℕ := 0o232

-- Prove that (num1 + num2) - num3 = expected_result in base 8
theorem base8_arithmetic : ((num1 + num2) - num3) = expected_result := by
  sorry

end base8_arithmetic_l255_25554


namespace principal_amount_invested_l255_25505

noncomputable def calculate_principal : ℕ := sorry

theorem principal_amount_invested (P : ℝ) (y : ℝ) 
    (h1 : 300 = P * y * 2 / 100) -- Condition for simple interest
    (h2 : 307.50 = P * ((1 + y/100)^2 - 1)) -- Condition for compound interest
    : P = 73.53 := 
sorry

end principal_amount_invested_l255_25505


namespace arithmetic_sequence_sum_l255_25540

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 0 + n * d) 
  (h2 : ∀ n, S_n n = n * (a 0 + a n) / 2) 
  (h3 : 2 * a 6 = 5 + a 8) :
  S_n 9 = 45 := 
by 
  sorry

end arithmetic_sequence_sum_l255_25540


namespace cos_triple_angle_l255_25548

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l255_25548


namespace circle_passes_through_fixed_point_l255_25566

theorem circle_passes_through_fixed_point :
  ∀ (C : ℝ × ℝ), (C.2 ^ 2 = 4 * C.1) ∧ (C.1 = -1 + (C.1 + 1)) → ∃ P : ℝ × ℝ, P = (1, 0) ∧
    (P.1 - C.1) ^ 2 + (P.2 - C.2) ^ 2 = (C.1 + 1) ^ 2 + (0 - C.2) ^ 2 :=
by
  sorry

end circle_passes_through_fixed_point_l255_25566


namespace students_still_in_school_l255_25506

theorem students_still_in_school
  (total_students : ℕ)
  (half_trip : total_students / 2 > 0)
  (half_remaining_sent_home : (total_students / 2) / 2 > 0)
  (total_students_val : total_students = 1000)
  :
  let students_still_in_school := total_students - (total_students / 2) - ((total_students - (total_students / 2)) / 2)
  students_still_in_school = 250 :=
by
  sorry

end students_still_in_school_l255_25506


namespace correct_intersection_l255_25583

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem correct_intersection : M ∩ N = {2, 3} := by sorry

end correct_intersection_l255_25583
