import Mathlib

namespace NUMINAMATH_GPT_least_four_digit_perfect_square_and_fourth_power_l3_344

theorem least_four_digit_perfect_square_and_fourth_power : 
    ∃ (n : ℕ), (1000 ≤ n) ∧ (n < 10000) ∧ (∃ a : ℕ, n = a^2) ∧ (∃ b : ℕ, n = b^4) ∧ 
    (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (∃ a : ℕ, m = a^2) ∧ (∃ b : ℕ, m = b^4) → n ≤ m) ∧ n = 6561 :=
by
  sorry

end NUMINAMATH_GPT_least_four_digit_perfect_square_and_fourth_power_l3_344


namespace NUMINAMATH_GPT_suitable_for_census_l3_355

-- Definitions based on the conditions in a)
def survey_A := "The service life of a batch of batteries"
def survey_B := "The height of all classmates in the class"
def survey_C := "The content of preservatives in a batch of food"
def survey_D := "The favorite mathematician of elementary and middle school students in the city"

-- The main statement to prove
theorem suitable_for_census : survey_B = "The height of all classmates in the class" := by
  -- We assert that the height of all classmates is the suitable survey for a census based on given conditions
  sorry

end NUMINAMATH_GPT_suitable_for_census_l3_355


namespace NUMINAMATH_GPT_floor_sum_even_l3_324

theorem floor_sum_even (a b c : ℕ) (h1 : a^2 + b^2 + 1 = c^2) : 
    ((a / 2) + (c / 2)) % 2 = 0 := 
  sorry

end NUMINAMATH_GPT_floor_sum_even_l3_324


namespace NUMINAMATH_GPT_boys_and_girls_original_total_l3_386

theorem boys_and_girls_original_total (b g : ℕ) 
(h1 : b = 3 * g) 
(h2 : b - 4 = 5 * (g - 4)) : 
b + g = 32 := 
sorry

end NUMINAMATH_GPT_boys_and_girls_original_total_l3_386


namespace NUMINAMATH_GPT_parallel_lines_l3_378

-- Definitions of lines and plane
variable {Line : Type}
variable {Plane : Type}
variable (a b c : Line)
variable (α : Plane)

-- Parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelPlane : Line → Plane → Prop)

-- Given conditions
variable (h1 : parallel a c)
variable (h2 : parallel b c)

-- Theorem statement
theorem parallel_lines (a b c : Line) 
                       (α : Plane) 
                       (parallel : Line → Line → Prop) 
                       (perpendicular : Line → Line → Prop) 
                       (parallelPlane : Line → Plane → Prop)
                       (h1 : parallel a c) 
                       (h2 : parallel b c) : 
                       parallel a b :=
sorry

end NUMINAMATH_GPT_parallel_lines_l3_378


namespace NUMINAMATH_GPT_tamara_total_earnings_l3_337

-- Definitions derived from the conditions in the problem statement.
def pans : ℕ := 2
def pieces_per_pan : ℕ := 8
def price_per_piece : ℕ := 2

-- Theorem stating the required proof problem.
theorem tamara_total_earnings : 
  (pans * pieces_per_pan * price_per_piece) = 32 :=
by
  sorry

end NUMINAMATH_GPT_tamara_total_earnings_l3_337


namespace NUMINAMATH_GPT_probability_all_five_dice_even_l3_306

-- Definitions of conditions
def standard_six_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6}
def even_numbers : Set ℕ := {2, 4, 6}

-- The statement to be proven
theorem probability_all_five_dice_even : 
  (∀ die ∈ standard_six_sided_die, (∃ n ∈ even_numbers, die = n)) → (1 / 32) = (1 / 2) ^ 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_probability_all_five_dice_even_l3_306


namespace NUMINAMATH_GPT_valid_votes_correct_l3_307

noncomputable def Total_votes : ℕ := 560000
noncomputable def Percentages_received : Fin 4 → ℚ 
| 0 => 0.4
| 1 => 0.35
| 2 => 0.15
| 3 => 0.1

noncomputable def Percentages_invalid : Fin 4 → ℚ 
| 0 => 0.12
| 1 => 0.18
| 2 => 0.25
| 3 => 0.3

noncomputable def Votes_received (i : Fin 4) : ℚ := Total_votes * Percentages_received i

noncomputable def Invalid_votes (i : Fin 4) : ℚ := Votes_received i * Percentages_invalid i

noncomputable def Valid_votes (i : Fin 4) : ℚ := Votes_received i - Invalid_votes i

def A_valid_votes := 197120
def B_valid_votes := 160720
def C_valid_votes := 63000
def D_valid_votes := 39200

theorem valid_votes_correct :
  Valid_votes 0 = A_valid_votes ∧
  Valid_votes 1 = B_valid_votes ∧
  Valid_votes 2 = C_valid_votes ∧
  Valid_votes 3 = D_valid_votes := by
  sorry

end NUMINAMATH_GPT_valid_votes_correct_l3_307


namespace NUMINAMATH_GPT_swimming_pool_width_l3_347

theorem swimming_pool_width 
  (V : ℝ) (L : ℝ) (B1 : ℝ) (B2 : ℝ) (h : ℝ)
  (h_volume : V = (h / 2) * (B1 + B2) * L) 
  (h_V : V = 270) 
  (h_L : L = 12) 
  (h_B1 : B1 = 1) 
  (h_B2 : B2 = 4) : 
  h = 9 :=
  sorry

end NUMINAMATH_GPT_swimming_pool_width_l3_347


namespace NUMINAMATH_GPT_correct_statements_l3_365

-- Define the universal set U as ℤ (integers)
noncomputable def U : Set ℤ := Set.univ

-- Conditions
def is_subset_of_int : Prop := {0} ⊆ (Set.univ : Set ℤ)

def counterexample_subsets (A B : Set ℤ) : Prop :=
  (A = {1, 2} ∧ B = {1, 2, 3}) ∧ (B ∩ (U \ A) ≠ ∅)

def negation_correct_1 : Prop :=
  ¬(∀ x : ℤ, x^2 > 0) ↔ ∃ x : ℤ, x^2 ≤ 0

def negation_correct_2 : Prop :=
  ¬(∀ x : ℤ, x^2 > 0) ↔ ¬(∀ x : ℤ, x^2 < 0)

-- The theorem to prove the equivalence of correct statements
theorem correct_statements :
  (is_subset_of_int ∧
   ∀ A B : Set ℤ, A ⊆ U → B ⊆ U → (A ⊆ B → counterexample_subsets A B) ∧
   negation_correct_1 ∧
   ¬negation_correct_2) ↔
  (true) :=
by 
  sorry

end NUMINAMATH_GPT_correct_statements_l3_365


namespace NUMINAMATH_GPT_product_of_four_integers_l3_315

theorem product_of_four_integers (A B C D : ℕ) (h_pos_A : 0 < A) (h_pos_B : 0 < B) (h_pos_C : 0 < C) (h_pos_D : 0 < D)
  (h_sum : A + B + C + D = 36)
  (h_eq1 : A + 2 = B - 2)
  (h_eq2 : B - 2 = C * 2)
  (h_eq3 : C * 2 = D / 2) :
  A * B * C * D = 3840 :=
by
  sorry

end NUMINAMATH_GPT_product_of_four_integers_l3_315


namespace NUMINAMATH_GPT_customers_in_other_countries_l3_331

-- Definitions for conditions
def total_customers : ℕ := 7422
def us_customers : ℕ := 723

-- Statement to prove
theorem customers_in_other_countries : total_customers - us_customers = 6699 :=
by
  sorry

end NUMINAMATH_GPT_customers_in_other_countries_l3_331


namespace NUMINAMATH_GPT_smallest_n_area_gt_2500_l3_354

noncomputable def triangle_area (n : ℕ) : ℝ :=
  (1/2 : ℝ) * (|(n : ℝ) * (2 * n) + (n^2 - 1 : ℝ) * (3 * n^2 - 1) + (n^3 - 3 * n) * 1
  - (1 : ℝ) * (n^2 - 1) - (2 * n) * (n^3 - 3 * n) - (3 * n^2 - 1) * (n : ℝ)|)

theorem smallest_n_area_gt_2500 : ∃ n : ℕ, (∀ m : ℕ, 0 < m ∧ m < n → triangle_area m <= 2500) ∧ triangle_area n > 2500 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_area_gt_2500_l3_354


namespace NUMINAMATH_GPT_pair1_equivalent_pair2_non_equivalent_pair3_equivalent_pair4_equivalent_pair5_non_equivalent_pair6_equivalent_l3_368

theorem pair1_equivalent (x : ℝ) : (x^2 + 5 * x < 4) ↔ (x^2 + 5 * x + 3 * x < 4 + 3 * x) :=
sorry

theorem pair2_non_equivalent (x : ℝ) (hx : x ≠ 0) : (x^2 + 5 * x < 4) ↔ (x^2 + 5 * x + 1 / x < 4 + 1 / x) :=
sorry

theorem pair3_equivalent (x : ℝ) (hx : x ≥ 3) : (x ≥ 3) ↔ (x * (x + 5)^2 ≥ 3 * (x + 5)^2) :=
sorry

theorem pair4_equivalent (x : ℝ) (hx : x ≥ 3) : (x ≥ 3) ↔ (x * (x - 5)^2 ≥ 3 * (x - 5)^2) :=
sorry

theorem pair5_non_equivalent (x : ℝ) (hx : x ≠ -1) : (x + 3 > 0) ↔ ( (x + 3) * (x + 1) / (x + 1) > 0) :=
sorry

theorem pair6_equivalent (x : ℝ) (hx : x ≠ -2) : (x - 3 > 0) ↔ ( (x + 2) * (x - 3) / (x + 2) > 0) :=
sorry

end NUMINAMATH_GPT_pair1_equivalent_pair2_non_equivalent_pair3_equivalent_pair4_equivalent_pair5_non_equivalent_pair6_equivalent_l3_368


namespace NUMINAMATH_GPT_odd_numbers_le_twice_switch_pairs_l3_396

-- Number of odd elements in row n is denoted as numOdd n
def numOdd (n : ℕ) : ℕ := -- Definition of numOdd function
sorry

-- Number of switch pairs in row n is denoted as numSwitchPairs n
def numSwitchPairs (n : ℕ) : ℕ := -- Definition of numSwitchPairs function
sorry

-- Definition of Pascal's Triangle and conditions
def binom (n k : ℕ) : ℕ := if k > n then 0 else if k = 0 ∨ k = n then 1 else binom (n-1) (k-1) + binom (n-1) k

-- Check even or odd
def isOdd (n : ℕ) : Bool := n % 2 = 1

-- Definition of switch pair check
def isSwitchPair (a b : ℕ) : Prop := (isOdd a ∧ ¬isOdd b) ∨ (¬isOdd a ∧ isOdd b)

theorem odd_numbers_le_twice_switch_pairs (n : ℕ) :
  numOdd n ≤ 2 * numSwitchPairs (n-1) :=
sorry

end NUMINAMATH_GPT_odd_numbers_le_twice_switch_pairs_l3_396


namespace NUMINAMATH_GPT_log_50_between_consecutive_integers_l3_348

theorem log_50_between_consecutive_integers :
    (∃ (m n : ℤ), m < n ∧ m < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < n ∧ m + n = 3) :=
by
  have log_10_eq_1 : Real.log 10 / Real.log 10 = 1 := by sorry
  have log_100_eq_2 : Real.log 100 / Real.log 10 = 2 := by sorry
  have log_increasing : ∀ (x y : ℝ), x < y → Real.log x / Real.log 10 < Real.log y / Real.log 10 := by sorry
  have interval : 10 < 50 ∧ 50 < 100 := by sorry
  use 1
  use 2
  sorry

end NUMINAMATH_GPT_log_50_between_consecutive_integers_l3_348


namespace NUMINAMATH_GPT_last_digit_sum_chessboard_segments_l3_391

theorem last_digit_sum_chessboard_segments {N : ℕ} (tile_count : ℕ) (segment_count : ℕ := 112) (dominos_per_tiling : ℕ := 32) (segments_per_domino : ℕ := 2) (N := tile_count / N) :
  (80 * N) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_sum_chessboard_segments_l3_391


namespace NUMINAMATH_GPT_pencils_in_drawer_l3_369

/-- 
If there were originally 2 pencils in the drawer and there are now 5 pencils in total, 
then Tim must have placed 3 pencils in the drawer.
-/
theorem pencils_in_drawer (original_pencils tim_pencils total_pencils : ℕ) 
  (h1 : original_pencils = 2) 
  (h2 : total_pencils = 5) 
  (h3 : total_pencils = original_pencils + tim_pencils) : 
  tim_pencils = 3 := 
by
  rw [h1, h2] at h3
  linarith

end NUMINAMATH_GPT_pencils_in_drawer_l3_369


namespace NUMINAMATH_GPT_find_f2_l3_357

-- Define f as an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define g based on f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f x + 9

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : odd_function f)
variable (h_g_neg2 : g f (-2) = 3)

-- Theorem statement
theorem find_f2 : f 2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l3_357


namespace NUMINAMATH_GPT_sarees_original_price_l3_374

theorem sarees_original_price (P : ℝ) (h : 0.75 * 0.85 * P = 248.625) : P = 390 :=
by
  sorry

end NUMINAMATH_GPT_sarees_original_price_l3_374


namespace NUMINAMATH_GPT_traditionalist_fraction_l3_364

theorem traditionalist_fraction (T P : ℕ) 
  (h1 : ∀ prov : ℕ, prov < 6 → T = P / 9) 
  (h2 : P + 6 * T > 0) :
  6 * T / (P + 6 * T) = 2 / 5 := 
by
  sorry

end NUMINAMATH_GPT_traditionalist_fraction_l3_364


namespace NUMINAMATH_GPT_Joan_initial_money_l3_320

def cost_hummus (containers : ℕ) (price_per_container : ℕ) : ℕ := containers * price_per_container
def cost_apple (quantity : ℕ) (price_per_apple : ℕ) : ℕ := quantity * price_per_apple

theorem Joan_initial_money 
  (containers_of_hummus : ℕ)
  (price_per_hummus : ℕ)
  (cost_chicken : ℕ)
  (cost_bacon : ℕ)
  (cost_vegetables : ℕ)
  (quantity_apple : ℕ)
  (price_per_apple : ℕ)
  (total_cost : ℕ)
  (remaining_money : ℕ):
  containers_of_hummus = 2 →
  price_per_hummus = 5 →
  cost_chicken = 20 →
  cost_bacon = 10 →
  cost_vegetables = 10 →
  quantity_apple = 5 →
  price_per_apple = 2 →
  remaining_money = cost_apple quantity_apple price_per_apple →
  total_cost = cost_hummus containers_of_hummus price_per_hummus + cost_chicken + cost_bacon + cost_vegetables + remaining_money →
  total_cost = 60 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Joan_initial_money_l3_320


namespace NUMINAMATH_GPT_import_tax_paid_l3_338

theorem import_tax_paid (total_value excess_value tax_rate tax_paid : ℝ)
  (h₁ : total_value = 2590)
  (h₂ : excess_value = total_value - 1000)
  (h₃ : tax_rate = 0.07)
  (h₄ : tax_paid = excess_value * tax_rate) : 
  tax_paid = 111.30 := by
  -- variables
  sorry

end NUMINAMATH_GPT_import_tax_paid_l3_338


namespace NUMINAMATH_GPT_system_of_equations_solution_l3_387

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
    (2 * x - 3 * y = 1) ∧ 
    (5 * x + 4 * y = 6) ∧ 
    (x + 2 * y = 2) ∧
    x = 2 / 3 ∧ y = 2 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_system_of_equations_solution_l3_387


namespace NUMINAMATH_GPT_person_walks_distance_l3_393

theorem person_walks_distance {D t : ℝ} (h1 : 5 * t = D) (h2 : 10 * t = D + 20) : D = 20 :=
by
  sorry

end NUMINAMATH_GPT_person_walks_distance_l3_393


namespace NUMINAMATH_GPT_a_term_b_value_c_value_d_value_l3_326

theorem a_term (a x : ℝ) (h1 : a * (x + 1) = x^3 + 3 * x^2 + 3 * x + 1) : a = x^2 + 2 * x + 1 :=
sorry

theorem b_value (a x b : ℝ) (h1 : a - 1 = 0) (h2 : x = 0 ∨ x = b) : b = -2 :=
sorry

theorem c_value (p c b : ℝ) (h1 : p * c^4 = 32) (h2 : p * c = b^2) (h3 : 0 < c) : c = 2 :=
sorry

theorem d_value (A B d : ℝ) (P : ℝ → ℝ) (c : ℝ) (h1 : P (A * B) = P A + P B) (h2 : P A = 1) (h3 : P B = c) (h4 : A = 10^ P A) (h5 : B = 10^ P B) (h6 : d = A * B) : d = 1000 :=
sorry

end NUMINAMATH_GPT_a_term_b_value_c_value_d_value_l3_326


namespace NUMINAMATH_GPT_amaya_movie_watching_time_l3_345

theorem amaya_movie_watching_time :
  let t1 := 30 + 5
  let t2 := 20 + 7
  let t3 := 10 + 12
  let t4 := 15 + 8
  let t5 := 25 + 15
  let t6 := 15 + 10
  t1 + t2 + t3 + t4 + t5 + t6 = 172 :=
by
  sorry

end NUMINAMATH_GPT_amaya_movie_watching_time_l3_345


namespace NUMINAMATH_GPT_trains_meet_in_time_l3_336

noncomputable def time_to_meet (length1 length2 distance_between speed1 speed2 : ℝ) : ℝ :=
  let speed1_mps := speed1 * 1000 / 3600
  let speed2_mps := speed2 * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := distance_between + length1 + length2
  total_distance / relative_speed

theorem trains_meet_in_time :
  time_to_meet 150 250 850 110 130 = 18.75 :=
by 
  -- here would go the proof steps, but since we are not required,
  sorry

end NUMINAMATH_GPT_trains_meet_in_time_l3_336


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l3_380

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x - (Real.pi / 3))

theorem necessary_but_not_sufficient (ω : ℝ) :
  (∀ x : ℝ, f ω (x + Real.pi) = f ω x) ↔ (ω = 2) ∨ (∃ ω ≠ 2, ∀ x : ℝ, f ω (x + Real.pi) = f ω x) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l3_380


namespace NUMINAMATH_GPT_solve_equation_l3_390

-- Define the equation and the conditions
def problem_equation (x : ℝ) : Prop :=
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 2)

def valid_solution (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -6

-- State the theorem that solutions x = 3 and x = -4 solve the problem under the conditions
theorem solve_equation : ∀ x : ℝ, valid_solution x → (x = 3 ∨ x = -4 ∧ problem_equation x) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l3_390


namespace NUMINAMATH_GPT_range_of_a_l3_342

/-- 
Proof problem statement derived from the given math problem and solution:
Prove that if the conditions:
1. ∀ x > 0, x + 1/x > a
2. ∃ x0 ∈ ℝ, x0^2 - 2*a*x0 + 1 ≤ 0
3. ¬ (∃ x0 ∈ ℝ, x0^2 - 2*a*x0 + 1 ≤ 0) is false
4. (∀ x > 0, x + 1/x > a) ∧ (∃ x0 ∈ ℝ, x0^2 - 2*a*x0 + 1 ≤ 0) is false
hold, then a ≥ 2.
-/
theorem range_of_a (a : ℝ)
  (h1 : ∀ x : ℝ, x > 0 → x + 1 / x > a)
  (h2 : ∃ x0 : ℝ, x0^2 - 2 * a * x0 + 1 ≤ 0)
  (h3 : ¬ (¬ (∃ x0 : ℝ, x0^2 - 2 * a * x0 + 1 ≤ 0)))
  (h4 : ¬ ((∀ x : ℝ, x > 0 → x + 1 / x > a) ∧ (∃ x0 : ℝ, x0^2 - 2 * a * x0 + 1 ≤ 0))) :
  a ≥ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l3_342


namespace NUMINAMATH_GPT_words_per_page_eq_106_l3_383

-- Definition of conditions as per the problem statement
def pages : ℕ := 224
def max_words_per_page : ℕ := 150
def total_words_congruence : ℕ := 156
def modulus : ℕ := 253

theorem words_per_page_eq_106 (p : ℕ) : 
  (224 * p % 253 = 156) ∧ (p ≤ 150) → p = 106 :=
by 
  sorry

end NUMINAMATH_GPT_words_per_page_eq_106_l3_383


namespace NUMINAMATH_GPT_primes_eq_condition_l3_327

theorem primes_eq_condition (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) : 
  p + q^2 = r^4 → p = 7 ∧ q = 3 ∧ r = 2 := 
by
  sorry

end NUMINAMATH_GPT_primes_eq_condition_l3_327


namespace NUMINAMATH_GPT_find_number_l3_341

theorem find_number (x : ℤ) (h : 27 + 2 * x = 39) : x = 6 :=
sorry

end NUMINAMATH_GPT_find_number_l3_341


namespace NUMINAMATH_GPT_intersection_eq_l3_303

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_eq : M ∩ N = {5, 7, 9} := sorry

end NUMINAMATH_GPT_intersection_eq_l3_303


namespace NUMINAMATH_GPT_perfect_square_proof_l3_334

theorem perfect_square_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := 
sorry

end NUMINAMATH_GPT_perfect_square_proof_l3_334


namespace NUMINAMATH_GPT_true_propositions_l3_395

-- Defining the propositions as functions for clarity
def proposition1 (L1 L2 P: Prop) : Prop := 
  (L1 ∧ L2 → P) → (P)

def proposition2 (plane1 plane2 line: Prop) : Prop := 
  (line → (plane1 ∧ plane2)) → (plane1 ∧ plane2)

def proposition3 (L1 L2 L3: Prop) : Prop := 
  (L1 ∧ L2 → L3) → L1

def proposition4 (plane1 plane2 line: Prop) : Prop := 
  (plane1 ∧ plane2 → (line → ¬ (plane1 ∧ plane2)))

-- Assuming the required mathematical hypothesis was valid within our formal system 
theorem true_propositions : proposition2 plane1 plane2 line ∧ proposition4 plane1 plane2 line := 
by sorry

end NUMINAMATH_GPT_true_propositions_l3_395


namespace NUMINAMATH_GPT_intersection_slopes_l3_318

theorem intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ 
  m ∈ Set.Iic (-Real.sqrt (4 / 41)) ∨ m ∈ Set.Ici (Real.sqrt (4 / 41)) := 
sorry

end NUMINAMATH_GPT_intersection_slopes_l3_318


namespace NUMINAMATH_GPT_min_value_of_f_l3_371

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem min_value_of_f : ∃ x : ℝ, (f x = -(1 / Real.exp 1)) ∧ (∀ y : ℝ, f y ≥ f x) := by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l3_371


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l3_372

def f (x a : ℝ) : ℝ := x^3 + 3 * a * x

def slope_tangent_at_one (a : ℝ) : ℝ := 3 * 1^2 + 3 * a

def are_perpendicular (a : ℝ) : Prop := -a = -1

theorem necessary_and_sufficient_condition (a : ℝ) :
  (slope_tangent_at_one a = 6) ↔ (are_perpendicular a) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l3_372


namespace NUMINAMATH_GPT_doughnuts_served_initially_l3_353

def initial_doughnuts_served (staff_count : Nat) (doughnuts_per_staff : Nat) (doughnuts_left : Nat) : Nat :=
  staff_count * doughnuts_per_staff + doughnuts_left

theorem doughnuts_served_initially :
  ∀ (staff_count doughnuts_per_staff doughnuts_left : Nat), staff_count = 19 → doughnuts_per_staff = 2 → doughnuts_left = 12 →
  initial_doughnuts_served staff_count doughnuts_per_staff doughnuts_left = 50 :=
by
  intros staff_count doughnuts_per_staff doughnuts_left hstaff hdonuts hleft
  rw [hstaff, hdonuts, hleft]
  rfl

#check doughnuts_served_initially

end NUMINAMATH_GPT_doughnuts_served_initially_l3_353


namespace NUMINAMATH_GPT_raffle_tickets_sold_l3_363

theorem raffle_tickets_sold (total_amount : ℕ) (ticket_cost : ℕ) (tickets_sold : ℕ) 
    (h1 : total_amount = 620) (h2 : ticket_cost = 4) : tickets_sold = 155 :=
by {
  sorry
}

end NUMINAMATH_GPT_raffle_tickets_sold_l3_363


namespace NUMINAMATH_GPT_infinite_non_expressible_integers_l3_376

theorem infinite_non_expressible_integers :
  ∃ (S : Set ℤ), S.Infinite ∧ (∀ n ∈ S, ∀ a b c : ℕ, n ≠ 2^a + 3^b - 5^c) :=
sorry

end NUMINAMATH_GPT_infinite_non_expressible_integers_l3_376


namespace NUMINAMATH_GPT_square_field_area_l3_375

noncomputable def area_of_square_field(speed_kph : ℝ) (time_hrs : ℝ) : ℝ :=
  let speed_mps := (speed_kph * 1000) / 3600
  let distance := speed_mps * (time_hrs * 3600)
  let side_length := distance / Real.sqrt 2
  side_length ^ 2

theorem square_field_area 
  (speed_kph : ℝ := 2.4)
  (time_hrs : ℝ := 3.0004166666666667) :
  area_of_square_field speed_kph time_hrs = 25939764.41 := 
by 
  -- This is a placeholder for the proof. 
  sorry

end NUMINAMATH_GPT_square_field_area_l3_375


namespace NUMINAMATH_GPT_area_enclosed_by_circle_l3_397

theorem area_enclosed_by_circle : 
  (∀ x y : ℝ, x^2 + y^2 + 10 * x + 24 * y = 0) → 
  (π * 13^2 = 169 * π):=
by
  intro h
  sorry

end NUMINAMATH_GPT_area_enclosed_by_circle_l3_397


namespace NUMINAMATH_GPT_projectile_first_reaches_70_feet_l3_325

theorem projectile_first_reaches_70_feet :
  ∃ t : ℝ , (t > 0) ∧ (-16 * t^2 + 80 * t = 70) ∧ (∀ t' : ℝ, (t' > 0) ∧ (-16 * t'^2 + 80 * t' = 70) → t ≤ t') :=
sorry

end NUMINAMATH_GPT_projectile_first_reaches_70_feet_l3_325


namespace NUMINAMATH_GPT_glove_ratio_l3_333

theorem glove_ratio (P : ℕ) (G : ℕ) (hf : P = 43) (hg : G = 2 * P) : G / P = 2 := by
  rw [hf, hg]
  norm_num
  sorry

end NUMINAMATH_GPT_glove_ratio_l3_333


namespace NUMINAMATH_GPT_log_base_30_of_8_l3_394

theorem log_base_30_of_8 (a b : Real) (h1 : Real.log 5 = a) (h2 : Real.log 3 = b) : 
    Real.logb 30 8 = 3 * (1 - a) / (b + 1) := 
  sorry

end NUMINAMATH_GPT_log_base_30_of_8_l3_394


namespace NUMINAMATH_GPT_total_dollar_amount_l3_328

/-- Definitions of base 5 numbers given in the problem -/
def pearls := 1 * 5^0 + 2 * 5^1 + 3 * 5^2 + 4 * 5^3
def silk := 1 * 5^0 + 1 * 5^1 + 1 * 5^2 + 1 * 5^3
def spices := 1 * 5^0 + 2 * 5^1 + 2 * 5^2
def maps := 0 * 5^0 + 1 * 5^1

/-- The theorem to prove the total dollar amount in base 10 -/
theorem total_dollar_amount : pearls + silk + spices + maps = 808 :=
by
  sorry

end NUMINAMATH_GPT_total_dollar_amount_l3_328


namespace NUMINAMATH_GPT_square_numbers_divisible_by_5_between_20_and_110_l3_335

theorem square_numbers_divisible_by_5_between_20_and_110 :
  ∃ (y : ℕ), (y = 25 ∨ y = 100) ∧ (∃ (n : ℕ), y = n^2) ∧ 5 ∣ y ∧ 20 < y ∧ y < 110 :=
by
  sorry

end NUMINAMATH_GPT_square_numbers_divisible_by_5_between_20_and_110_l3_335


namespace NUMINAMATH_GPT_find_e_l3_351

theorem find_e (d e f : ℕ) (hd : d > 1) (he : e > 1) (hf : f > 1) :
  (∀ M : ℝ, M ≠ 1 → (M^(1/d) * (M^(1/e) * (M^(1/f)))^(1/e)^(1/d)) = (M^(17/24))^(1/24)) → e = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_e_l3_351


namespace NUMINAMATH_GPT_train_cross_pole_time_l3_356

-- Definitions based on the conditions
def train_speed_kmh := 54
def train_length_m := 105
def train_speed_ms := (train_speed_kmh * 1000) / 3600
def expected_time := train_length_m / train_speed_ms

-- Theorem statement, encapsulating the problem
theorem train_cross_pole_time : expected_time = 7 := by
  sorry

end NUMINAMATH_GPT_train_cross_pole_time_l3_356


namespace NUMINAMATH_GPT_problem1_problem2_l3_319

open Real -- Open the Real namespace for trigonometric functions

-- Part 1: Prove cos(5π + α) * tan(α - 7π) = 4/5 given π < α < 2π and cos α = 3/5
theorem problem1 (α : ℝ) (hα1 : π < α) (hα2 : α < 2 * π) (hcos : cos α = 3 / 5) : 
  cos (5 * π + α) * tan (α - 7 * π) = 4 / 5 := sorry

-- Part 2: Prove sin(π/3 + α) = √3/3 given cos (π/6 - α) = √3/3
theorem problem2 (α : ℝ) (hcos : cos (π / 6 - α) = sqrt 3 / 3) : 
  sin (π / 3 + α) = sqrt 3 / 3 := sorry

end NUMINAMATH_GPT_problem1_problem2_l3_319


namespace NUMINAMATH_GPT_line_intersects_circle_and_angle_conditions_l3_323

noncomputable def line_circle_intersection_condition (k : ℝ) : Prop :=
  - (Real.sqrt 3) / 3 ≤ k ∧ k ≤ (Real.sqrt 3) / 3

noncomputable def inclination_angle_condition (θ : ℝ) : Prop :=
  (0 ≤ θ ∧ θ ≤ Real.pi / 6) ∨ (5 * Real.pi / 6 ≤ θ ∧ θ < Real.pi)

theorem line_intersects_circle_and_angle_conditions (k θ : ℝ) :
  line_circle_intersection_condition k →
  inclination_angle_condition θ →
  ∃ x y : ℝ, (y = k * (x + 1)) ∧ ((x - 1)^2 + y^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_circle_and_angle_conditions_l3_323


namespace NUMINAMATH_GPT_no_valid_pairs_l3_313

open Nat

theorem no_valid_pairs (l y : ℕ) (h1 : y % 30 = 0) (h2 : l > 1) :
  (∃ n m : ℕ, 180 - 360 / n = y ∧ 180 - 360 / m = l * y ∧ y * l ≤ 180) → False := 
by
  intro h
  sorry

end NUMINAMATH_GPT_no_valid_pairs_l3_313


namespace NUMINAMATH_GPT_bottle_cost_l3_314

-- Definitions of the conditions
def total_cost := 30
def wine_extra_cost := 26

-- Statement of the problem in Lean 4
theorem bottle_cost : 
  ∃ x : ℕ, (x + (x + wine_extra_cost) = total_cost) ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_bottle_cost_l3_314


namespace NUMINAMATH_GPT_min_value_l3_367

noncomputable def min_expression_value (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 2 ≤ k) : ℝ :=
  (x^2 + k*x + 1) * (y^2 + k*y + 1) * (z^2 + k*z + 1) / (x * y * z)

theorem min_value (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 2 ≤ k) :
  min_expression_value x y z k hx hy hz hk ≥ (2 + k)^3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_l3_367


namespace NUMINAMATH_GPT_bob_day3_miles_l3_340

noncomputable def total_miles : ℕ := 70
noncomputable def day1_miles : ℕ := total_miles * 20 / 100
noncomputable def remaining_after_day1 : ℕ := total_miles - day1_miles
noncomputable def day2_miles : ℕ := remaining_after_day1 * 50 / 100
noncomputable def remaining_after_day2 : ℕ := remaining_after_day1 - day2_miles
noncomputable def day3_miles : ℕ := remaining_after_day2

theorem bob_day3_miles : day3_miles = 28 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_bob_day3_miles_l3_340


namespace NUMINAMATH_GPT_overall_average_score_l3_350

def students_monday := 24
def students_tuesday := 4
def total_students := 28
def mean_score_monday := 82
def mean_score_tuesday := 90

theorem overall_average_score :
  (students_monday * mean_score_monday + students_tuesday * mean_score_tuesday) / total_students = 83 := by
sorry

end NUMINAMATH_GPT_overall_average_score_l3_350


namespace NUMINAMATH_GPT_find_k_l3_317

noncomputable def a_squared : ℝ := 9
noncomputable def b_squared (k : ℝ) : ℝ := 4 + k
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

noncomputable def c_squared_1 (k : ℝ) : ℝ := 5 - k
noncomputable def c_squared_2 (k : ℝ) : ℝ := k - 5

theorem find_k (k : ℝ) :
  (eccentricity (Real.sqrt (c_squared_1 k)) (Real.sqrt a_squared) = 4 / 5 →
   k = -19 / 25) ∨ 
  (eccentricity (Real.sqrt (c_squared_2 k)) (Real.sqrt (b_squared k)) = 4 / 5 →
   k = 21) :=
sorry

end NUMINAMATH_GPT_find_k_l3_317


namespace NUMINAMATH_GPT_real_roots_range_of_k_l3_349

theorem real_roots_range_of_k (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 - 2 * k * x + (k + 3) = 0) ↔ (k ≤ 3 / 2) :=
sorry

end NUMINAMATH_GPT_real_roots_range_of_k_l3_349


namespace NUMINAMATH_GPT_average_velocity_eq_l3_304

noncomputable def motion_eq : ℝ → ℝ := λ t => 1 - t + t^2

theorem average_velocity_eq (Δt : ℝ) :
  (motion_eq (3 + Δt) - motion_eq 3) / Δt = 5 + Δt :=
by
  sorry

end NUMINAMATH_GPT_average_velocity_eq_l3_304


namespace NUMINAMATH_GPT_cuboid_ratio_l3_310

theorem cuboid_ratio (length breadth height: ℕ) (h_length: length = 90) (h_breadth: breadth = 75) (h_height: height = 60) : 
(length / Nat.gcd length (Nat.gcd breadth height) = 6) ∧ 
(breadth / Nat.gcd length (Nat.gcd breadth height) = 5) ∧ 
(height / Nat.gcd length (Nat.gcd breadth height) = 4) := by 
  -- intentionally skipped proof 
  sorry

end NUMINAMATH_GPT_cuboid_ratio_l3_310


namespace NUMINAMATH_GPT_largest_four_digit_number_divisible_by_2_5_9_11_l3_346

theorem largest_four_digit_number_divisible_by_2_5_9_11 : ∃ n : ℤ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (∀ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (n % 2 = 0) ∧ 
  (n % 5 = 0) ∧ 
  (n % 9 = 0) ∧ 
  (n % 11 = 0) ∧ 
  (n = 8910) := 
by
  sorry

end NUMINAMATH_GPT_largest_four_digit_number_divisible_by_2_5_9_11_l3_346


namespace NUMINAMATH_GPT_perpendicular_vector_solution_l3_388

theorem perpendicular_vector_solution 
    (a b : ℝ × ℝ) (m : ℝ) 
    (h_a : a = (1, -1)) 
    (h_b : b = (-2, 3)) 
    (h_perp : a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) 
    : m = 2 / 5 := 
sorry

end NUMINAMATH_GPT_perpendicular_vector_solution_l3_388


namespace NUMINAMATH_GPT_find_different_mass_part_l3_379

-- Definitions for the parts a1, a2, a3, a4 and their masses
variable {α : Type}
variables (a₁ a₂ a₃ a₄ : α)
variable [LinearOrder α]

-- Definition of the problem conditions
def different_mass_part (a₁ a₂ a₃ a₄ : α) : Prop :=
  (a₁ ≠ a₂ ∨ a₁ ≠ a₃ ∨ a₁ ≠ a₄ ∨ a₂ ≠ a₃ ∨ a₂ ≠ a₄ ∨ a₃ ≠ a₄)

-- Theorem statement assuming we can identify the differing part using two weighings on a pan balance
theorem find_different_mass_part (h : different_mass_part a₁ a₂ a₃ a₄) :
  ∃ (part : α), part = a₁ ∨ part = a₂ ∨ part = a₃ ∨ part = a₄ :=
sorry

end NUMINAMATH_GPT_find_different_mass_part_l3_379


namespace NUMINAMATH_GPT_degrees_to_radians_750_l3_382

theorem degrees_to_radians_750 (π : ℝ) (deg_750 : ℝ) 
  (h : 180 = π) : 
  750 * (π / 180) = 25 / 6 * π :=
by
  sorry

end NUMINAMATH_GPT_degrees_to_radians_750_l3_382


namespace NUMINAMATH_GPT_expr1_val_expr2_val_l3_377

noncomputable def expr1 : ℝ :=
  (1 / Real.sin (10 * Real.pi / 180)) - (Real.sqrt 3 / Real.cos (10 * Real.pi / 180))

theorem expr1_val : expr1 = 4 :=
  sorry

noncomputable def expr2 : ℝ :=
  (Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) - Real.cos (20 * Real.pi / 180)) /
  (Real.cos (80 * Real.pi / 180) * Real.sqrt (1 - Real.cos (20 * Real.pi / 180)))

theorem expr2_val : expr2 = Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_expr1_val_expr2_val_l3_377


namespace NUMINAMATH_GPT_rowing_speed_l3_362

theorem rowing_speed (V_m V_w V_upstream V_downstream : ℝ)
  (h1 : V_upstream = 25)
  (h2 : V_downstream = 65)
  (h3 : V_w = 5) :
  V_m = 45 :=
by
  -- Lean will verify the theorem given the conditions
  sorry

end NUMINAMATH_GPT_rowing_speed_l3_362


namespace NUMINAMATH_GPT_find_number_l3_352

theorem find_number :
  ∃ (N : ℝ), (5 / 4) * N = (4 / 5) * N + 45 ∧ N = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l3_352


namespace NUMINAMATH_GPT_trip_length_l3_339

theorem trip_length 
  (total_time : ℝ) (canoe_speed : ℝ) (hike_speed : ℝ) (hike_distance : ℝ)
  (hike_time_eq : hike_distance / hike_speed = 5.4) 
  (canoe_time_eq : total_time - hike_distance / hike_speed = 0.1)
  (canoe_distance_eq : canoe_speed * (total_time - hike_distance / hike_speed) = 1.2)
  (total_time_val : total_time = 5.5)
  (canoe_speed_val : canoe_speed = 12)
  (hike_speed_val : hike_speed = 5)
  (hike_distance_val : hike_distance = 27) :
  total_time = 5.5 → canoe_speed = 12 → hike_speed = 5 → hike_distance = 27 → hike_distance + canoe_speed * (total_time - hike_distance / hike_speed) = 28.2 := 
by
  intro h_total_time h_canoe_speed h_hike_speed h_hike_distance
  rw [h_total_time, h_canoe_speed, h_hike_speed, h_hike_distance]
  sorry

end NUMINAMATH_GPT_trip_length_l3_339


namespace NUMINAMATH_GPT_distinct_real_roots_l3_384

theorem distinct_real_roots (k : ℝ) :
  (∀ x : ℝ, (k - 2) * x^2 + 2 * x - 1 = 0 → ∃ y : ℝ, (k - 2) * y^2 + 2 * y - 1 = 0 ∧ y ≠ x) ↔
  (k > 1 ∧ k ≠ 2) := 
by sorry

end NUMINAMATH_GPT_distinct_real_roots_l3_384


namespace NUMINAMATH_GPT_find_n_l3_302

theorem find_n : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 11] ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l3_302


namespace NUMINAMATH_GPT_galya_overtakes_sasha_l3_300

variable {L : ℝ} -- Length of the track
variable (Sasha_uphill_speed : ℝ := 8)
variable (Sasha_downhill_speed : ℝ := 24)
variable (Galya_uphill_speed : ℝ := 16)
variable (Galya_downhill_speed : ℝ := 18)

noncomputable def average_speed (uphill_speed: ℝ) (downhill_speed: ℝ) : ℝ :=
  1 / ((1 / (4 * uphill_speed)) + (3 / (4 * downhill_speed)))

noncomputable def time_for_one_lap (L: ℝ) (speed: ℝ) : ℝ :=
  L / speed

theorem galya_overtakes_sasha 
  (L_pos : 0 < L) :
  let v_Sasha := average_speed Sasha_uphill_speed Sasha_downhill_speed
  let v_Galya := average_speed Galya_uphill_speed Galya_downhill_speed
  let t_Sasha := time_for_one_lap L v_Sasha
  let t_Galya := time_for_one_lap L v_Galya
  (L * 11 / v_Galya) < (L * 10 / v_Sasha) :=
by
  sorry

end NUMINAMATH_GPT_galya_overtakes_sasha_l3_300


namespace NUMINAMATH_GPT_exists_increasing_seq_with_sum_square_diff_l3_373

/-- There exists an increasing sequence of natural numbers in which
  the sum of any two consecutive terms is equal to the square of their
  difference. -/
theorem exists_increasing_seq_with_sum_square_diff :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, a n < a (n + 1)) ∧ (∀ n : ℕ, a n + a (n + 1) = (a (n + 1) - a n) ^ 2) :=
sorry

end NUMINAMATH_GPT_exists_increasing_seq_with_sum_square_diff_l3_373


namespace NUMINAMATH_GPT_pieces_of_chocolate_left_l3_366

theorem pieces_of_chocolate_left (initial_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ) 
    (h1 : initial_boxes = 14) (h2 : given_away_boxes = 8) (h3 : pieces_per_box = 3) : 
    (initial_boxes - given_away_boxes) * pieces_per_box = 18 := 
by 
  -- The proof will be here
  sorry

end NUMINAMATH_GPT_pieces_of_chocolate_left_l3_366


namespace NUMINAMATH_GPT_p_expression_l3_385

theorem p_expression (m n p : ℤ) (r1 r2 : ℝ) 
  (h1 : r1 + r2 = m) 
  (h2 : r1 * r2 = n) 
  (h3 : r1^2 + r2^2 = p) : 
  p = m^2 - 2 * n := by
  sorry

end NUMINAMATH_GPT_p_expression_l3_385


namespace NUMINAMATH_GPT_coordinates_reflect_y_axis_l3_343

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem coordinates_reflect_y_axis (p : ℝ × ℝ) (h : p = (5, 2)) : reflect_y_axis p = (-5, 2) :=
by
  rw [h]
  rfl

end NUMINAMATH_GPT_coordinates_reflect_y_axis_l3_343


namespace NUMINAMATH_GPT_right_triangle_arithmetic_sequence_side_length_l3_398

theorem right_triangle_arithmetic_sequence_side_length :
  ∃ (a b c : ℕ), (a < b ∧ b < c) ∧ (b - a = c - b) ∧ (a^2 + b^2 = c^2) ∧ (b = 81) :=
sorry

end NUMINAMATH_GPT_right_triangle_arithmetic_sequence_side_length_l3_398


namespace NUMINAMATH_GPT_fraction_decomposition_roots_sum_l3_305

theorem fraction_decomposition_roots_sum :
  ∀ (p q r A B C : ℝ),
  p ≠ q → p ≠ r → q ≠ r →
  (∀ (s : ℝ), s ≠ p → s ≠ q → s ≠ r →
          1 / (s^3 - 15 * s^2 + 50 * s - 56) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 225 :=
by
  intros p q r A B C hpq hpr hqr hDecomp
  -- Skip proof
  sorry

end NUMINAMATH_GPT_fraction_decomposition_roots_sum_l3_305


namespace NUMINAMATH_GPT_Y_minus_X_eq_92_l3_332

def arithmetic_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def X : ℕ := arithmetic_sum 10 2 46
def Y : ℕ := arithmetic_sum 12 2 46

theorem Y_minus_X_eq_92 : Y - X = 92 := by
  sorry

end NUMINAMATH_GPT_Y_minus_X_eq_92_l3_332


namespace NUMINAMATH_GPT_triangular_array_sum_of_digits_l3_399

def triangular_sum (N : ℕ) : ℕ := N * (N + 1) / 2

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem triangular_array_sum_of_digits :
  ∃ N : ℕ, triangular_sum N = 2080 ∧ sum_of_digits N = 10 :=
by
  sorry

end NUMINAMATH_GPT_triangular_array_sum_of_digits_l3_399


namespace NUMINAMATH_GPT_tan_alpha_add_pi_div_four_l3_312

theorem tan_alpha_add_pi_div_four {α : ℝ} (h1 : α ∈ Set.Ioo 0 (Real.pi)) (h2 : Real.cos α = -4/5) :
  Real.tan (α + Real.pi / 4) = 1 / 7 :=
sorry

end NUMINAMATH_GPT_tan_alpha_add_pi_div_four_l3_312


namespace NUMINAMATH_GPT_find_m_l3_321

-- Define vectors a and b
def a (m : ℝ) : ℝ × ℝ := (2, -m)
def b : ℝ × ℝ := (1, 3)

-- Define the condition for perpendicular vectors
def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- State the problem
theorem find_m (m : ℝ) (h : is_perpendicular (a m + b) b) : m = 4 :=
sorry -- proof omitted

end NUMINAMATH_GPT_find_m_l3_321


namespace NUMINAMATH_GPT_price_difference_is_correct_l3_360

-- Definitions from the problem conditions
def list_price : ℝ := 58.80
def tech_shop_discount : ℝ := 12.00
def value_mart_discount_rate : ℝ := 0.20

-- Calculating the sale prices from definitions
def tech_shop_sale_price : ℝ := list_price - tech_shop_discount
def value_mart_sale_price : ℝ := list_price * (1 - value_mart_discount_rate)

-- The proof problem statement
theorem price_difference_is_correct :
  value_mart_sale_price - tech_shop_sale_price = 0.24 :=
by
  sorry

end NUMINAMATH_GPT_price_difference_is_correct_l3_360


namespace NUMINAMATH_GPT_trigonometric_problem_l3_389

theorem trigonometric_problem
  (α : ℝ)
  (h1 : Real.tan α = Real.sqrt 3)
  (h2 : π < α)
  (h3 : α < 3 * π / 2) :
  Real.cos (2 * π - α) - Real.sin α = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_GPT_trigonometric_problem_l3_389


namespace NUMINAMATH_GPT_find_g_l3_311

noncomputable def g (x : ℝ) : ℝ := 2 - 4 * x

theorem find_g :
  g 0 = 2 ∧ (∀ x y : ℝ, g (x * y) = g ((3 * x ^ 2 + y ^ 2) / 4) + 3 * (x - y) ^ 2) → ∀ x : ℝ, g x = 2 - 4 * x :=
by
  sorry

end NUMINAMATH_GPT_find_g_l3_311


namespace NUMINAMATH_GPT_rounding_estimation_correct_l3_359

theorem rounding_estimation_correct (a b d : ℕ)
  (ha : a > 0) (hb : b > 0) (hd : d > 0)
  (a_round : ℕ) (b_round : ℕ) (d_round : ℕ)
  (h_round_a : a_round ≥ a) (h_round_b : b_round ≤ b) (h_round_d : d_round ≤ d) :
  (Real.sqrt (a_round / b_round) - Real.sqrt d_round) > (Real.sqrt (a / b) - Real.sqrt d) :=
by
  sorry

end NUMINAMATH_GPT_rounding_estimation_correct_l3_359


namespace NUMINAMATH_GPT_target_average_income_l3_308

variable (past_incomes : List ℕ) (next_average : ℕ)

def total_past_income := past_incomes.sum
def total_next_income := next_average * 5
def total_ten_week_income := total_past_income past_incomes + total_next_income next_average

theorem target_average_income (h1 : past_incomes = [406, 413, 420, 436, 395])
                              (h2 : next_average = 586) :
  total_ten_week_income past_incomes next_average / 10 = 500 := by
  sorry

end NUMINAMATH_GPT_target_average_income_l3_308


namespace NUMINAMATH_GPT_school_B_saving_l3_330

def cost_A (kg_price : ℚ) (kg_amount : ℚ) : ℚ :=
  kg_price * kg_amount

def effective_kg_B (total_kg : ℚ) (extra_percentage : ℚ) : ℚ :=
  total_kg / (1 + extra_percentage)

def cost_B (kg_price : ℚ) (effective_kg : ℚ) : ℚ :=
  kg_price * effective_kg

theorem school_B_saving
  (kg_amount : ℚ) (price_A: ℚ) (discount: ℚ) (extra_percentage : ℚ) 
  (expected_saving : ℚ)
  (h1 : kg_amount = 56)
  (h2 : price_A = 8.06)
  (h3 : discount = 0.56)
  (h4 : extra_percentage = 0.05)
  (h5 : expected_saving = 51.36) :
  cost_A price_A kg_amount - cost_B (price_A - discount) (effective_kg_B kg_amount extra_percentage) = expected_saving := 
by 
  sorry

end NUMINAMATH_GPT_school_B_saving_l3_330


namespace NUMINAMATH_GPT_minimum_days_to_pay_back_l3_322

theorem minimum_days_to_pay_back (x : ℕ) : 
  (50 + 5 * x ≥ 150) → x = 20 :=
sorry

end NUMINAMATH_GPT_minimum_days_to_pay_back_l3_322


namespace NUMINAMATH_GPT_solve_problem_l3_358

def spadesuit (x y : ℝ) : ℝ := x^2 + y^2

theorem solve_problem : spadesuit (spadesuit 3 5) 4 = 1172 := by
  sorry

end NUMINAMATH_GPT_solve_problem_l3_358


namespace NUMINAMATH_GPT_arithmetic_progression_l3_301

theorem arithmetic_progression (a b c : ℝ) (h : a + c = 2 * b) :
  3 * (a^2 + b^2 + c^2) = 6 * (a - b)^2 + (a + b + c)^2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_l3_301


namespace NUMINAMATH_GPT_vectors_orthogonal_dot_product_l3_329

theorem vectors_orthogonal_dot_product (y : ℤ) :
  (3 * -2) + (4 * y) + (-1 * 5) = 0 → y = 11 / 4 :=
by
  sorry

end NUMINAMATH_GPT_vectors_orthogonal_dot_product_l3_329


namespace NUMINAMATH_GPT_find_x_min_construction_cost_l3_381

-- Define the conditions for Team A and Team B
def Team_A_Daily_Construction (x : ℕ) : ℕ := x + 300
def Team_A_Daily_Cost : ℕ := 3600
def Team_B_Daily_Construction (x : ℕ) : ℕ := x
def Team_B_Daily_Cost : ℕ := 2200

-- Condition: The number of days Team A needs to construct 1800m^2 is equal to the number of days Team B needs to construct 1200m^2
def construction_days (x : ℕ) : Prop := 
  1800 / (x + 300) = 1200 / x

-- Define the total days worked and the minimum construction area condition
def total_days : ℕ := 22
def min_construction_area : ℕ := 15000

-- Define the construction cost function given the number of days each team works
def construction_cost (m : ℕ) : ℕ := 
  3600 * m + 2200 * (total_days - m)

-- Main theorem: Prove that x = 600 satisfies the conditions
theorem find_x (x : ℕ) (h : x = 600) : construction_days x := by sorry

-- Second theorem: Prove that the minimum construction cost is 56800 yuan
theorem min_construction_cost (m : ℕ) (h : m ≥ 6) : construction_cost m = 56800 := by sorry

end NUMINAMATH_GPT_find_x_min_construction_cost_l3_381


namespace NUMINAMATH_GPT_running_distance_l3_361

theorem running_distance (D : ℕ) 
  (hA_time : ∀ (A_time : ℕ), A_time = 28) 
  (hB_time : ∀ (B_time : ℕ), B_time = 32) 
  (h_lead : ∀ (lead : ℕ), lead = 28) 
  (hA_speed : ∀ (A_speed : ℚ), A_speed = D / 28) 
  (hB_speed : ∀ (B_speed : ℚ), B_speed = D / 32) 
  (hB_dist : ∀ (B_dist : ℚ), B_dist = D - 28) 
  (h_eq : ∀ (B_dist : ℚ), B_dist = D * (28 / 32)) :
  D = 224 :=
by 
  sorry

end NUMINAMATH_GPT_running_distance_l3_361


namespace NUMINAMATH_GPT_length_of_purple_part_l3_309

variables (P : ℝ) (black : ℝ) (blue : ℝ) (total_len : ℝ)

-- The conditions
def conditions := 
  black = 0.5 ∧ 
  blue = 2 ∧ 
  total_len = 4 ∧ 
  P + black + blue = total_len

-- The proof problem statement
theorem length_of_purple_part (h : conditions P 0.5 2 4) : P = 1.5 :=
sorry

end NUMINAMATH_GPT_length_of_purple_part_l3_309


namespace NUMINAMATH_GPT_prime_gt_three_modulus_l3_316

theorem prime_gt_three_modulus (p : ℕ) (hp : Prime p) (hp_gt3 : p > 3) : (p^2 + 12) % 12 = 1 := by
  sorry

end NUMINAMATH_GPT_prime_gt_three_modulus_l3_316


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_equilateral_triangle_l3_392

theorem eccentricity_of_ellipse_equilateral_triangle (c b a e : ℝ)
  (h1 : b = Real.sqrt (3 * c))
  (h2 : a = Real.sqrt (b^2 + c^2)) 
  (h3 : e = c / a) :
  e = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_eccentricity_of_ellipse_equilateral_triangle_l3_392


namespace NUMINAMATH_GPT_female_democrats_l3_370

theorem female_democrats (F M D_f: ℕ) 
  (h1 : F + M = 780)
  (h2 : D_f = (1/2) * F)
  (h3 : (1/3) * 780 = 260)
  (h4 : 260 = (1/2) * F + (1/4) * M) : 
  D_f = 130 := 
by
  sorry

end NUMINAMATH_GPT_female_democrats_l3_370
