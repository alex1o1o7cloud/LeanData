import Mathlib

namespace algorithm_output_l455_45572

noncomputable def algorithm (x : ℝ) : ℝ :=
if x < 0 then x + 1 else -x^2

theorem algorithm_output :
  algorithm (-2) = -1 ∧ algorithm 3 = -9 :=
by
  -- proof omitted using sorry
  sorry

end algorithm_output_l455_45572


namespace number_of_passed_candidates_l455_45512

theorem number_of_passed_candidates
  (P F : ℕ) 
  (h1 : P + F = 120)
  (h2 : 39 * P + 15 * F = 4200) : P = 100 :=
sorry

end number_of_passed_candidates_l455_45512


namespace negation_of_existential_l455_45525

theorem negation_of_existential (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) :=
by
  sorry

end negation_of_existential_l455_45525


namespace domain_all_real_iff_l455_45577

theorem domain_all_real_iff (k : ℝ) :
  (∀ x : ℝ, -3 * x ^ 2 - x + k ≠ 0 ) ↔ k < -1 / 12 :=
by
  sorry

end domain_all_real_iff_l455_45577


namespace brian_total_video_length_l455_45578

theorem brian_total_video_length :
  let cat_length := 4
  let dog_length := 2 * cat_length
  let gorilla_length := cat_length ^ 2
  let elephant_length := cat_length + dog_length + gorilla_length
  let cat_dog_gorilla_elephant_sum := cat_length + dog_length + gorilla_length + elephant_length
  let penguin_length := cat_dog_gorilla_elephant_sum ^ 3
  let dolphin_length := cat_length + dog_length + gorilla_length + elephant_length + penguin_length
  let total_length := cat_length + dog_length + gorilla_length + elephant_length + penguin_length + dolphin_length
  total_length = 351344 := by
    sorry

end brian_total_video_length_l455_45578


namespace smallest_positive_value_l455_45530

noncomputable def exprA := 30 - 4 * Real.sqrt 14
noncomputable def exprB := 4 * Real.sqrt 14 - 30
noncomputable def exprC := 25 - 6 * Real.sqrt 15
noncomputable def exprD := 75 - 15 * Real.sqrt 30
noncomputable def exprE := 15 * Real.sqrt 30 - 75

theorem smallest_positive_value :
  exprC = 25 - 6 * Real.sqrt 15 ∧
  exprC < exprA ∧
  exprC < exprB ∧
  exprC < exprD ∧
  exprC < exprE ∧
  exprC > 0 :=
by sorry

end smallest_positive_value_l455_45530


namespace drunk_drivers_count_l455_45501

theorem drunk_drivers_count (D S : ℕ) (h1 : S = 7 * D - 3) (h2 : D + S = 45) : D = 6 :=
by
  sorry

end drunk_drivers_count_l455_45501


namespace union_sets_l455_45545

-- Definitions of sets A and B
def set_A : Set ℝ := {x | x / (x - 1) < 0}
def set_B : Set ℝ := {x | abs (1 - x) > 1 / 2}

-- The problem: prove that the union of sets A and B is (-∞, 1) ∪ (3/2, ∞)
theorem union_sets :
  set_A ∪ set_B = {x | x < 1} ∪ {x | x > 3 / 2} :=
by
  sorry

end union_sets_l455_45545


namespace integer_count_in_interval_l455_45521

theorem integer_count_in_interval : 
  let lower_bound := Int.floor (-7 * Real.pi)
  let upper_bound := Int.ceil (12 * Real.pi)
  upper_bound - lower_bound + 1 = 61 :=
by
  let lower_bound := Int.floor (-7 * Real.pi)
  let upper_bound := Int.ceil (12 * Real.pi)
  have : upper_bound - lower_bound + 1 = 61 := sorry
  exact this

end integer_count_in_interval_l455_45521


namespace initial_quantity_of_milk_l455_45535

theorem initial_quantity_of_milk (A B C : ℝ) 
    (h1 : B = 0.375 * A)
    (h2 : C = 0.625 * A)
    (h3 : B + 148 = C - 148) : A = 1184 :=
by
  sorry

end initial_quantity_of_milk_l455_45535


namespace correct_calculation_A_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_correct_answer_is_A_l455_45584

theorem correct_calculation_A : (Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6) :=
by { sorry }

theorem incorrect_calculation_B : (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) :=
by { sorry }

theorem incorrect_calculation_C : ((Real.sqrt 2)^2 ≠ 2 * Real.sqrt 2) :=
by { sorry }

theorem incorrect_calculation_D : (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) :=
by { sorry }

theorem correct_answer_is_A :
  (Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6) ∧
  (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) ∧
  ((Real.sqrt 2)^2 ≠ 2 * Real.sqrt 2) ∧
  (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) :=
by {
  exact ⟨correct_calculation_A, incorrect_calculation_B, incorrect_calculation_C, incorrect_calculation_D⟩
}

end correct_calculation_A_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_correct_answer_is_A_l455_45584


namespace sampling_interval_l455_45593

theorem sampling_interval 
  (total_population : ℕ) 
  (individuals_removed : ℕ) 
  (population_after_removal : ℕ)
  (sampling_interval : ℕ) :
  total_population = 102 →
  individuals_removed = 2 →
  population_after_removal = total_population - individuals_removed →
  population_after_removal = 100 →
  ∃ s : ℕ, population_after_removal % s = 0 ∧ s = 10 := 
by
  sorry

end sampling_interval_l455_45593


namespace solve_eq_l455_45579

theorem solve_eq :
  { x : ℝ | (14 * x - x^2) / (x + 2) * (x + (14 - x) / (x + 2)) = 48 } =
  {4, (1 + Real.sqrt 193) / 2, (1 - Real.sqrt 193) / 2} :=
by
  sorry

end solve_eq_l455_45579


namespace train_length_is_1400_l455_45598

theorem train_length_is_1400
  (L : ℝ) 
  (h1 : ∃ speed, speed = L / 100) 
  (h2 : ∃ speed, speed = (L + 700) / 150) :
  L = 1400 :=
by sorry

end train_length_is_1400_l455_45598


namespace Abby_sits_in_seat_3_l455_45565

theorem Abby_sits_in_seat_3:
  ∃ (positions : Fin 5 → String),
  (positions 3 = "Abby") ∧
  (positions 4 = "Bret") ∧
  ¬ ((positions 3 = "Dana") ∨ (positions 5 = "Dana")) ∧
  ¬ ((positions 2 = "Erin") ∧ (positions 3 = "Carl") ∨
    (positions 3 = "Erin") ∧ (positions 5 = "Carl")) :=
  sorry

end Abby_sits_in_seat_3_l455_45565


namespace minimum_number_is_correct_l455_45534

-- Define the operations and conditions on the digits
def transform (n : ℕ) : ℕ :=
if 2 ≤ n then n - 2 + 1 else n

noncomputable def minimum_transformed_number (l : List ℕ) : List ℕ :=
l.map transform

def initial_number : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def expected_number : List ℕ := [1, 0, 1, 0, 1, 0, 1, 0, 1]

theorem minimum_number_is_correct :
  minimum_transformed_number initial_number = expected_number := 
by
  -- sorry is a placeholder for the proof
  sorry

end minimum_number_is_correct_l455_45534


namespace percentage_increase_biographies_l455_45594

variable (B b n : ℝ)
variable (h1 : b = 0.20 * B)
variable (h2 : b + n = 0.32 * (B + n))

theorem percentage_increase_biographies (B b n : ℝ) (h1 : b = 0.20 * B) (h2 : b + n = 0.32 * (B + n)) :
  n / b * 100 = 88.24 := by
  sorry

end percentage_increase_biographies_l455_45594


namespace whale_length_l455_45552

theorem whale_length
  (velocity_fast : ℕ)
  (velocity_slow : ℕ)
  (time : ℕ)
  (h1 : velocity_fast = 18)
  (h2 : velocity_slow = 15)
  (h3 : time = 15) :
  (velocity_fast - velocity_slow) * time = 45 := 
by
  sorry

end whale_length_l455_45552


namespace largest_difference_l455_45575

noncomputable def A := 3 * (2010: ℕ) ^ 2011
noncomputable def B := (2010: ℕ) ^ 2011
noncomputable def C := 2009 * (2010: ℕ) ^ 2010
noncomputable def D := 3 * (2010: ℕ) ^ 2010
noncomputable def E := (2010: ℕ) ^ 2010
noncomputable def F := (2010: ℕ) ^ 2009

theorem largest_difference :
  (A - B) > (B - C) ∧ (A - B) > (C - D) ∧ (A - B) > (D - E) ∧ (A - B) > (E - F) :=
by
  sorry

end largest_difference_l455_45575


namespace man_salary_problem_l455_45583

-- Define the problem in Lean 4
theorem man_salary_problem (S : ℝ) :
  (1/3 * S) + (1/4 * S) + (1/5 * S) + 1760 = S → 
  S = 8123.08 :=
sorry

end man_salary_problem_l455_45583


namespace required_cups_of_sugar_l455_45596

-- Define the original ratios
def original_flour_water_sugar_ratio : Rat := 10 / 6 / 3
def new_flour_water_ratio : Rat := 2 * (10 / 6)
def new_flour_sugar_ratio : Rat := (1 / 2) * (10 / 3)

-- Given conditions
def cups_of_water : Rat := 2

-- Problem statement: prove the amount of sugar required
theorem required_cups_of_sugar : ∀ (sugar_cups : Rat),
  original_flour_water_sugar_ratio = 10 / 6 / 3 ∧
  new_flour_water_ratio = 2 * (10 / 6) ∧
  new_flour_sugar_ratio = (1 / 2) * (10 / 3) ∧
  cups_of_water = 2 ∧
  (6 / 12) = (2 / sugar_cups) → sugar_cups = 4 := by
  intro sugar_cups
  sorry

end required_cups_of_sugar_l455_45596


namespace compare_neg_rationals_l455_45527

theorem compare_neg_rationals : - (3 / 4 : ℚ) > - (6 / 5 : ℚ) :=
by sorry

end compare_neg_rationals_l455_45527


namespace total_coins_l455_45551

theorem total_coins (total_value : ℕ) (value_2_coins : ℕ) (num_2_coins : ℕ) (num_1_coins : ℕ) : 
  total_value = 402 ∧ value_2_coins = 2 * num_2_coins ∧ num_2_coins = 148 ∧ total_value = value_2_coins + num_1_coins →
  num_1_coins + num_2_coins = 254 :=
by
  intros h
  sorry

end total_coins_l455_45551


namespace students_did_not_eat_2_l455_45582

-- Define the given conditions
def total_students : ℕ := 20
def total_crackers_eaten : ℕ := 180
def crackers_per_pack : ℕ := 10

-- Calculate the number of packs eaten
def packs_eaten : ℕ := total_crackers_eaten / crackers_per_pack

-- Calculate the number of students who did not eat their animal crackers
def students_who_did_not_eat : ℕ := total_students - packs_eaten

-- Prove that the number of students who did not eat their animal crackers is 2
theorem students_did_not_eat_2 :
  students_who_did_not_eat = 2 :=
  by
    sorry

end students_did_not_eat_2_l455_45582


namespace length_of_segment_NB_l455_45526

variable (L W x : ℝ)
variable (h1 : 0 < L) (h2 : 0 < W) (h3 : x * W / 2 = 0.4 * (L * W))

theorem length_of_segment_NB (L W x : ℝ) (h1 : 0 < L) (h2 : 0 < W) (h3 : x * W / 2 = 0.4 * (L * W)) : 
  x = 0.8 * L :=
by
  sorry

end length_of_segment_NB_l455_45526


namespace pentagon_right_angles_l455_45544

theorem pentagon_right_angles (angles : Finset ℕ) :
  angles = {0, 1, 2, 3} ↔ ∀ (k : ℕ), k ∈ angles ↔ ∃ (a b c d e : ℕ), 
  a + b + c + d + e = 540 ∧ (a = 90 ∨ b = 90 ∨ c = 90 ∨ d = 90 ∨ e = 90) 
  ∧ Finset.card (Finset.filter (λ x => x = 90) {a, b, c, d, e}) = k := 
sorry

end pentagon_right_angles_l455_45544


namespace dasha_strip_dimensions_l455_45574

theorem dasha_strip_dimensions (a b c : ℕ) (h1 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43) : 
  (a = 1 ∧ (b + c = 22)) ∨ (a = 22 ∧ (b + c = 1)) :=
by sorry

end dasha_strip_dimensions_l455_45574


namespace odd_positive_multiples_of_7_with_units_digit_1_lt_200_count_l455_45515

theorem odd_positive_multiples_of_7_with_units_digit_1_lt_200_count : 
  ∃ (count : ℕ), count = 3 ∧
  ∀ n : ℕ, (n % 2 = 1) → (n % 7 = 0) → (n < 200) → (n % 10 = 1) → count = 3 :=
sorry

end odd_positive_multiples_of_7_with_units_digit_1_lt_200_count_l455_45515


namespace find_x_pos_integer_l455_45523

theorem find_x_pos_integer (x : ℕ) (h : 0 < x) (n d : ℕ)
    (h1 : n = x^2 + 4 * x + 29)
    (h2 : d = 4 * x + 9)
    (h3 : n = d * x + 13) : 
    x = 2 := 
sorry

end find_x_pos_integer_l455_45523


namespace find_sum_of_squares_l455_45555

theorem find_sum_of_squares (x y : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x * y + x + y = 35) (h4 : x^2 * y + x * y^2 = 210) : x^2 + y^2 = 154 :=
sorry

end find_sum_of_squares_l455_45555


namespace periodic_odd_function_value_at_7_l455_45541

noncomputable def f : ℝ → ℝ := sorry -- Need to define f appropriately, skipped for brevity

theorem periodic_odd_function_value_at_7
    (f_odd : ∀ x : ℝ, f (-x) = -f x)
    (f_periodic : ∀ x : ℝ, f (x + 4) = f x)
    (f_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x) :
    f 7 = -1 := sorry

end periodic_odd_function_value_at_7_l455_45541


namespace real_roots_range_l455_45588

theorem real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k*x^2 - 6*x + 9 = 0) ↔ k ≤ 1 :=
sorry

end real_roots_range_l455_45588


namespace lulu_cash_left_l455_45519

-- Define the initial amount
def initial_amount : ℕ := 65

-- Define the amount spent on ice cream
def spent_on_ice_cream : ℕ := 5

-- Define the amount spent on a t-shirt
def spent_on_tshirt (remaining_after_ice_cream : ℕ) : ℕ := remaining_after_ice_cream / 2

-- Define the amount deposited in the bank
def deposited_in_bank (remaining_after_tshirt : ℕ) : ℕ := remaining_after_tshirt / 5

-- Define the remaining cash after all transactions
def remaining_cash (initial : ℕ) (spent_ice_cream : ℕ) (spent_tshirt: ℕ) (deposited: ℕ) :ℕ :=
  initial - spent_ice_cream - spent_tshirt - deposited

-- Theorem statement to prove
theorem lulu_cash_left : remaining_cash initial_amount spent_on_ice_cream (spent_on_tshirt (initial_amount - spent_on_ice_cream)) 
(deposited_in_bank ((initial_amount - spent_on_ice_cream) - (spent_on_tshirt (initial_amount - spent_on_ice_cream)))) = 24 :=
by
  sorry

end lulu_cash_left_l455_45519


namespace remainder_of_difference_divided_by_prime_l455_45536

def largest_three_digit_number : ℕ := 999
def smallest_five_digit_number : ℕ := 10000
def smallest_prime_greater_than_1000 : ℕ := 1009

theorem remainder_of_difference_divided_by_prime :
  (smallest_five_digit_number - largest_three_digit_number) % smallest_prime_greater_than_1000 = 945 :=
by
  -- The proof will be filled in here
  sorry

end remainder_of_difference_divided_by_prime_l455_45536


namespace sum_of_exterior_angles_of_convex_quadrilateral_l455_45557

theorem sum_of_exterior_angles_of_convex_quadrilateral:
  ∀ (α β γ δ : ℝ),
  (α + β + γ + δ = 360) → 
  (∀ (θ₁ θ₂ θ₃ θ₄ : ℝ),
    (θ₁ = 180 - α ∧ θ₂ = 180 - β ∧ θ₃ = 180 - γ ∧ θ₄ = 180 - δ) → 
    θ₁ + θ₂ + θ₃ + θ₄ = 360) := 
by 
  intros α β γ δ h1 θ₁ θ₂ θ₃ θ₄ h2
  rcases h2 with ⟨hα, hβ, hγ, hδ⟩
  rw [hα, hβ, hγ, hδ]
  linarith

end sum_of_exterior_angles_of_convex_quadrilateral_l455_45557


namespace value_of_number_l455_45502

theorem value_of_number (x : ℤ) (number : ℚ) (h₁ : x = 32) (h₂ : 35 - (23 - (15 - x)) = 12 * number / (1/2)) : number = -5/6 :=
by
  sorry

end value_of_number_l455_45502


namespace probability_cello_viola_same_tree_l455_45549

noncomputable section

def cellos : ℕ := 800
def violas : ℕ := 600
def cello_viola_pairs_same_tree : ℕ := 100

theorem probability_cello_viola_same_tree : 
  (cello_viola_pairs_same_tree: ℝ) / ((cellos * violas : ℕ) : ℝ) = 1 / 4800 := 
by
  sorry

end probability_cello_viola_same_tree_l455_45549


namespace delta_five_three_l455_45563

def Δ (a b : ℕ) : ℕ := 4 * a - 6 * b

theorem delta_five_three :
  Δ 5 3 = 2 := by
  sorry

end delta_five_three_l455_45563


namespace total_lambs_l455_45517

def num_initial_lambs : ℕ := 6
def num_baby_lambs_per_mother : ℕ := 2
def num_mothers : ℕ := 2
def traded_lambs : ℕ := 3
def extra_lambs : ℕ := 7

theorem total_lambs :
  num_initial_lambs + (num_baby_lambs_per_mother * num_mothers) - traded_lambs + extra_lambs = 14 :=
by
  sorry

end total_lambs_l455_45517


namespace gcd_of_repeated_three_digit_integers_is_1001001_l455_45537

theorem gcd_of_repeated_three_digit_integers_is_1001001 :
  ∀ (n : ℕ), (100 ≤ n ∧ n <= 999) →
  ∃ d : ℕ, d = 1001001 ∧
    (∀ m : ℕ, m = n * 1001001 →
      ∃ k : ℕ, m = k * d) :=
by
  sorry

end gcd_of_repeated_three_digit_integers_is_1001001_l455_45537


namespace sum_fractions_lt_one_l455_45510

theorem sum_fractions_lt_one (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  0 < (a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c)) ∧
  (a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c)) < 1 :=
by
  sorry

end sum_fractions_lt_one_l455_45510


namespace part1_part2_l455_45538

def A : Set ℝ := {x | x^2 + x - 12 < 0}
def B : Set ℝ := {x | 4 / (x + 3) ≤ 1}
def C (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x + m^2 - 1 ≤ 0}

theorem part1 : A ∩ B = {x | -4 < x ∧ x < -3 ∨ 1 ≤ x ∧ x < 3} := sorry

theorem part2 (m : ℝ) : (-3 < m ∧ m < 2) ↔ ∀ x, (x ∈ A → x ∈ C m) ∧ ∃ x, x ∈ C m ∧ x ∉ A := sorry

end part1_part2_l455_45538


namespace f_comp_f_neg1_l455_45524

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then (1 / 4) ^ x else Real.log x / Real.log (1 / 2)

theorem f_comp_f_neg1 : f (f (-1)) = -2 := 
by
  sorry

end f_comp_f_neg1_l455_45524


namespace angle_between_bisectors_is_zero_l455_45528

-- Let's define the properties of the triangle and the required proof.

open Real

-- Define the side lengths of the isosceles triangle
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧ a > 0 ∧ b > 0 ∧ c > 0

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c) ∧ is_triangle a b c

-- Define the specific isosceles triangle in the problem
def triangle_ABC : Prop := is_isosceles 5 5 6

-- Prove that the angle φ between the two lines is 0°
theorem angle_between_bisectors_is_zero :
  triangle_ABC → ∃ φ : ℝ, φ = 0 :=
by sorry

end angle_between_bisectors_is_zero_l455_45528


namespace least_number_to_add_l455_45548

theorem least_number_to_add {n : ℕ} (h : n = 1202) : (∃ k : ℕ, (n + k) % 4 = 0 ∧ ∀ m : ℕ, (m < k → (n + m) % 4 ≠ 0)) ∧ k = 2 := by
  sorry

end least_number_to_add_l455_45548


namespace simplify_fraction_l455_45546

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) : (x^2 / (x - 2) - 4 / (x - 2)) = x + 2 := by
  sorry

end simplify_fraction_l455_45546


namespace range_of_angle_of_inclination_l455_45514

theorem range_of_angle_of_inclination (α : ℝ) :
  ∃ θ : ℝ, θ ∈ (Set.Icc 0 (Real.pi / 4) ∪ Set.Ico (3 * Real.pi / 4) Real.pi) ∧
           ∀ x : ℝ, ∃ y : ℝ, y = x * Real.sin α + 1 := by
  sorry

end range_of_angle_of_inclination_l455_45514


namespace angle_of_inclination_l455_45585

/--
Given the direction vector of line l as (-sqrt(3), 3),
prove that the angle of inclination α of line l is 120 degrees.
-/
theorem angle_of_inclination (α : ℝ) :
  let direction_vector : Real × Real := (-Real.sqrt 3, 3)
  let slope := direction_vector.2 / direction_vector.1
  slope = -Real.sqrt 3 → α = 120 :=
by
  sorry

end angle_of_inclination_l455_45585


namespace find_f_of_3_l455_45561

theorem find_f_of_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end find_f_of_3_l455_45561


namespace rahul_salary_l455_45509

variable (X : ℝ)

def house_rent_deduction (salary : ℝ) : ℝ := salary * 0.8
def education_expense (remaining_after_rent : ℝ) : ℝ := remaining_after_rent * 0.9
def clothing_expense (remaining_after_education : ℝ) : ℝ := remaining_after_education * 0.9

theorem rahul_salary : (X * 0.8 * 0.9 * 0.9 = 1377) → X = 2125 :=
by
  intros h
  sorry

end rahul_salary_l455_45509


namespace sum_of_products_l455_45599

variable (a b c : ℝ)

theorem sum_of_products (h1 : a^2 + b^2 + c^2 = 250) (h2 : a + b + c = 16) : 
  ab + bc + ca = 3 :=
sorry

end sum_of_products_l455_45599


namespace intersection_complement_eq_l455_45542

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - 5 * x + 4 < 0}

theorem intersection_complement_eq :
  A ∩ {x | x ≤ 1 ∨ x ≥ 4} = {0, 1} := by
  sorry

end intersection_complement_eq_l455_45542


namespace integer_root_abs_sum_l455_45529

noncomputable def solve_abs_sum (p q r : ℤ) : ℤ := |p| + |q| + |r|

theorem integer_root_abs_sum (p q r m : ℤ) 
  (h1 : p + q + r = 0)
  (h2 : p * q + q * r + r * p = -2024)
  (h3 : ∃ m, ∀ x, x^3 - 2024 * x + m = (x - p) * (x - q) * (x - r)) :
  solve_abs_sum p q r = 104 :=
by sorry

end integer_root_abs_sum_l455_45529


namespace arc_length_correct_l455_45533

noncomputable def radius : ℝ :=
  5

noncomputable def area_of_sector : ℝ :=
  8.75

noncomputable def arc_length (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * 2 * Real.pi * r

theorem arc_length_correct :
  ∃ θ, arc_length θ radius = 3.5 ∧ (θ / 360) * Real.pi * radius^2 = area_of_sector :=
by
  sorry

end arc_length_correct_l455_45533


namespace sum_arithmetic_sequence_ge_four_l455_45562

theorem sum_arithmetic_sequence_ge_four
  (a_n : ℕ → ℚ) -- arithmetic sequence
  (S : ℕ → ℚ) -- sum of the first n terms of the sequence
  (h_arith_seq : ∀ n, S n = (n * a_n 1) + (n * (n - 1) / 2) * (a_n 2 - a_n 1))
  (p q : ℕ)
  (hpq_ne : p ≠ q)
  (h_sp : S p = p / q)
  (h_sq : S q = q / p) :
  S (p + q) ≥ 4 :=
by
  sorry

end sum_arithmetic_sequence_ge_four_l455_45562


namespace henry_collected_points_l455_45505

def points_from_wins (wins : ℕ) : ℕ := wins * 5
def points_from_losses (losses : ℕ) : ℕ := losses * 2
def points_from_draws (draws : ℕ) : ℕ := draws * 3

def total_points (wins losses draws : ℕ) : ℕ := 
  points_from_wins wins + points_from_losses losses + points_from_draws draws

theorem henry_collected_points :
  total_points 2 2 10 = 44 := by
  -- The proof will go here
  sorry

end henry_collected_points_l455_45505


namespace find_arithmetic_progression_terms_l455_45560

noncomputable def arithmetic_progression_terms (a1 a2 a3 : ℕ) (d : ℕ) 
  (condition1 : a1 + (a1 + d) = 3 * 2^2) 
  (condition2 : a1 + (a1 + d) + (a1 + 2 * d) = 3 * 3^2) : Prop := 
  a1 = 3 ∧ a2 = 9 ∧ a3 = 15

theorem find_arithmetic_progression_terms
  (a1 a2 a3 : ℕ) (d : ℕ)
  (cond1 : a1 + (a1 + d) = 3 * 2^2)
  (cond2 : a1 + (a1 + d) + (a1 + 2 * d) = 3 * 3^2) :
  arithmetic_progression_terms a1 a2 a3 d cond1 cond2 :=
sorry

end find_arithmetic_progression_terms_l455_45560


namespace joe_speed_l455_45566

theorem joe_speed (P : ℝ) (J : ℝ) (h1 : J = 2 * P) (h2 : 2 * P * (2 / 3) + P * (2 / 3) = 16) : J = 16 := 
by
  sorry

end joe_speed_l455_45566


namespace number_multiplied_by_9_l455_45554

theorem number_multiplied_by_9 (x : ℕ) (h : 50 = x + 26) : 9 * x = 216 := by
  sorry

end number_multiplied_by_9_l455_45554


namespace N_eq_M_union_P_l455_45558

open Set

def M : Set ℝ := { x | ∃ n : ℤ, x = n }
def N : Set ℝ := { x | ∃ n : ℤ, x = n / 2 }
def P : Set ℝ := { x | ∃ n : ℤ, x = n + 1/2 }

theorem N_eq_M_union_P : N = M ∪ P := 
sorry

end N_eq_M_union_P_l455_45558


namespace unique_painted_cube_l455_45591

/-- Determine the number of distinct ways to paint a cube where:
  - One side is yellow,
  - Two sides are purple,
  - Three sides are orange.
  Taking into account that two cubes are considered identical if they can be rotated to match. -/
theorem unique_painted_cube :
  ∃ unique n : ℕ, n = 1 ∧
    (∃ (c : Fin 6 → Fin 3), 
      (∃ (i : Fin 6), c i = 0) ∧ 
      (∃ (j k : Fin 6), j ≠ k ∧ c j = 1 ∧ c k = 1) ∧ 
      (∃ (m p q : Fin 6), m ≠ p ∧ m ≠ q ∧ p ≠ q ∧ c m = 2 ∧ c p = 2 ∧ c q = 2)
    ) :=
sorry

end unique_painted_cube_l455_45591


namespace find_m_l455_45531

theorem find_m (m : ℝ) (h : (4 * (-1)^3 + 3 * m * (-1)^2 + 6 * (-1) = 2)) :
  m = 4 :=
by
  sorry

end find_m_l455_45531


namespace triangle_area_eq_l455_45556

noncomputable def areaOfTriangle (a b c A B C: ℝ): ℝ :=
1 / 2 * a * c * (Real.sin A)

theorem triangle_area_eq
  (a b c A B C : ℝ)
  (h1 : a = 2)
  (h2 : A = Real.pi / 3)
  (h3 : Real.sqrt 3 / 2 - Real.sin (B - C) = Real.sin (2 * B)) :
  areaOfTriangle a b c A B C = Real.sqrt 3 ∨ areaOfTriangle a b c A B C = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end triangle_area_eq_l455_45556


namespace joan_games_l455_45586

theorem joan_games (games_this_year games_total games_last_year : ℕ) 
  (h1 : games_this_year = 4) 
  (h2 : games_total = 9) 
  (h3 : games_total = games_this_year + games_last_year) :
  games_last_year = 5 :=
by {
  -- The proof goes here
  sorry
}

end joan_games_l455_45586


namespace find_sum_lent_l455_45522

theorem find_sum_lent (P : ℝ) : 
  (∃ R T : ℝ, R = 4 ∧ T = 8 ∧ I = P - 170 ∧ I = (P * 8) / 25) → P = 250 :=
by
  sorry

end find_sum_lent_l455_45522


namespace find_k_l455_45568

theorem find_k (a b c k : ℝ) 
  (h : ∀ x : ℝ, 
    (a * x^2 + b * x + c + b * x^2 + a * x - 7 + k * x^2 + c * x + 3) / (x^2 - 2 * x - 5) = (x^2 - 2*x - 5)) :
  k = 2 :=
by
  sorry

end find_k_l455_45568


namespace georgia_black_buttons_l455_45576

theorem georgia_black_buttons : 
  ∀ (B : ℕ), 
  (4 + B + 3 = 9) → 
  B = 2 :=
by
  introv h
  linarith

end georgia_black_buttons_l455_45576


namespace find_angle_FYD_l455_45573

noncomputable def angle_FYD (AB CD AXF FYG : ℝ) : ℝ := 180 - AXF

theorem find_angle_FYD (AB CD : ℝ) (AXF : ℝ) (FYG : ℝ) (h1 : AB = CD) (h2 : AXF = 125) (h3 : FYG = 40) :
  angle_FYD AB CD AXF FYG = 55 :=
by
  sorry

end find_angle_FYD_l455_45573


namespace exists_quadratic_function_l455_45564

theorem exists_quadratic_function :
  (∃ (a b c : ℝ), ∀ (k : ℕ), k > 0 → (a * (5 / 9 * (10^k - 1))^2 + b * (5 / 9 * (10^k - 1)) + c = 5/9 * (10^(2*k) - 1))) :=
by
  have a := 9 / 5
  have b := 2
  have c := 0
  use a, b, c
  intros k hk
  sorry

end exists_quadratic_function_l455_45564


namespace complex_expression_evaluation_l455_45590

-- Defining the imaginary unit
def i : ℂ := Complex.I

-- Defining the complex number z
def z : ℂ := 1 - i

-- Stating the theorem to prove
theorem complex_expression_evaluation : z^2 + (2 / z) = 1 - i := by
  sorry

end complex_expression_evaluation_l455_45590


namespace chocolate_bars_remaining_l455_45518

theorem chocolate_bars_remaining (total_bars sold_week1 sold_week2 : ℕ) (h_total : total_bars = 18) (h_sold1 : sold_week1 = 5) (h_sold2 : sold_week2 = 7) : total_bars - (sold_week1 + sold_week2) = 6 :=
by {
  sorry
}

end chocolate_bars_remaining_l455_45518


namespace shortest_side_of_right_triangle_l455_45504

theorem shortest_side_of_right_triangle (a b : ℝ) (h : a = 9 ∧ b = 12) : ∃ c : ℝ, (c = min a b) ∧ c = 9 :=
by
  sorry

end shortest_side_of_right_triangle_l455_45504


namespace odd_f_neg1_l455_45550

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 
  if 0 ≤ x 
  then 2^x + 2 * x + b 
  else - (2^(-x) + 2 * (-x) + b)

theorem odd_f_neg1 (b : ℝ) (h : f 0 b = 0) : f (-1) b = -3 :=
by
  sorry

end odd_f_neg1_l455_45550


namespace line_tangent_to_circle_l455_45587

open Real

theorem line_tangent_to_circle :
    ∃ (x y : ℝ), (3 * x - 4 * y - 5 = 0) ∧ ((x - 1)^2 + (y + 3)^2 - 4 = 0) ∧ 
    (∃ (t r : ℝ), (t = 0 ∧ r ≠ 0) ∧ 
     (3 * t - 4 * (r + t * 3 / 4) - 5 = 0) ∧ ((r + t * 3 / 4 - 1)^2 + (3 * (-1) + t - 3)^2 = 0)) 
  :=
sorry

end line_tangent_to_circle_l455_45587


namespace monotone_function_sol_l455_45508

noncomputable def monotone_function (f : ℤ → ℤ) :=
  ∀ x y : ℤ, f x ≤ f y → x ≤ y

theorem monotone_function_sol
  (f : ℤ → ℤ)
  (H1 : monotone_function f)
  (H2 : ∀ x y : ℤ, f (x^2005 + y^2005) = f x ^ 2005 + f y ^ 2005) :
  (∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x) :=
sorry

end monotone_function_sol_l455_45508


namespace find_real_x_l455_45506

noncomputable def solution_set (x : ℝ) := (5 ≤ x) ∧ (x < 5.25)

theorem find_real_x (x : ℝ) :
  (⌊x * ⌊x⌋⌋ = 20) ↔ solution_set x :=
by
  sorry

end find_real_x_l455_45506


namespace sum_of_three_smallest_two_digit_primes_l455_45513

theorem sum_of_three_smallest_two_digit_primes :
  11 + 13 + 17 = 41 :=
by
  sorry

end sum_of_three_smallest_two_digit_primes_l455_45513


namespace value_of_f_at_3_l455_45559

def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem value_of_f_at_3 : f 3 = 15 :=
by
  -- This proof needs to be filled in
  sorry

end value_of_f_at_3_l455_45559


namespace hyperbola_midpoint_exists_l455_45547

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l455_45547


namespace a7_arithmetic_sequence_l455_45570

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def a1 : ℝ := 2
def a4 : ℝ := 5

theorem a7_arithmetic_sequence : ∃ d : ℝ, is_arithmetic_sequence a d ∧ a 1 = a1 ∧ a 4 = a4 → a 7 = 8 :=
by
  sorry

end a7_arithmetic_sequence_l455_45570


namespace shaded_area_of_octagon_l455_45539

noncomputable def areaOfShadedRegion (s : ℝ) (r : ℝ) (theta : ℝ) : ℝ :=
  let n := 8
  let octagonArea := n * 0.5 * s^2 * (Real.sin (Real.pi/n) / Real.sin (Real.pi/(2 * n)))
  let sectorArea := n * 0.5 * r^2 * (theta / (2 * Real.pi))
  octagonArea - sectorArea

theorem shaded_area_of_octagon (h_s : 5 = 5) (h_r : 3 = 3) (h_theta : 45 = 45) :
  areaOfShadedRegion 5 3 (45 * (Real.pi / 180)) = 100 - 9 * Real.pi := by
  sorry

end shaded_area_of_octagon_l455_45539


namespace like_term_l455_45595

theorem like_term (a : ℝ) : ∃ (a : ℝ), a * x ^ 5 * y ^ 3 = a * x ^ 5 * y ^ 3 :=
by sorry

end like_term_l455_45595


namespace gcd_12569_36975_l455_45516

-- Define the integers for which we need to find the gcd
def num1 : ℕ := 12569
def num2 : ℕ := 36975

-- The statement that the gcd of these two numbers is 1
theorem gcd_12569_36975 : Nat.gcd num1 num2 = 1 := by
  sorry

end gcd_12569_36975_l455_45516


namespace area_increase_l455_45581

theorem area_increase (original_length original_width new_length : ℝ)
  (h1 : original_length = 20)
  (h2 : original_width = 5)
  (h3 : new_length = original_length + 10) :
  (new_length * original_width - original_length * original_width) = 50 := by
  sorry

end area_increase_l455_45581


namespace smallest_integer_in_set_l455_45511

theorem smallest_integer_in_set :
  ∀ (n : ℤ), (n + 6 > 2 * ((n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7)) → n = -1 :=
by
  intros n h
  sorry

end smallest_integer_in_set_l455_45511


namespace total_number_of_balls_l455_45507

theorem total_number_of_balls 
(b : ℕ) (P_blue : ℚ) (h1 : b = 8) (h2 : P_blue = 1/3) : 
  ∃ g : ℕ, b + g = 24 := by
  sorry

end total_number_of_balls_l455_45507


namespace cost_price_article_l455_45532

variable (SP : ℝ := 21000)
variable (d : ℝ := 0.10)
variable (p : ℝ := 0.08)

theorem cost_price_article : (SP * (1 - d)) / (1 + p) = 17500 := by
  sorry

end cost_price_article_l455_45532


namespace relay_race_time_l455_45540

theorem relay_race_time (R S D : ℕ) (h1 : S = R + 2) (h2 : D = R - 3) (h3 : R + S + D = 71) : R = 24 :=
by
  sorry

end relay_race_time_l455_45540


namespace problem1_problem2_problem3_problem4_problem5_l455_45553

-- Definitions and conditions
variable (a : ℝ) (b : ℝ) (ha : a > 0) (hb : b > 0) (hineq : a - 2 * Real.sqrt b > 0)

-- Problem 1: √(a - 2√b) = √m - √n
theorem problem1 (h₁ : a = 5) (h₂ : b = 6) : Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 := sorry

-- Problem 2: √(a + 2√b) = √m + √n
theorem problem2 (h₁ : a = 12) (h₂ : b = 35) : Real.sqrt (12 + 2 * Real.sqrt 35) = Real.sqrt 7 + Real.sqrt 5 := sorry

-- Problem 3: √(a + 6√b) = √m + √n
theorem problem3 (h₁ : a = 9) (h₂ : b = 6) : Real.sqrt (9 + 6 * Real.sqrt 2) = Real.sqrt 6 + Real.sqrt 3 := sorry

-- Problem 4: √(a - 4√b) = √m - √n
theorem problem4 (h₁ : a = 16) (h₂ : b = 60) : Real.sqrt (16 - 4 * Real.sqrt 15) = Real.sqrt 10 - Real.sqrt 6 := sorry

-- Problem 5: √(a - √b) + √(c + √d)
theorem problem5 (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 2) (h₄ : d = 3) 
  : Real.sqrt (3 - Real.sqrt 5) + Real.sqrt (2 + Real.sqrt 3) = (Real.sqrt 10 + Real.sqrt 6) / 2 := sorry

end problem1_problem2_problem3_problem4_problem5_l455_45553


namespace billy_tickets_used_l455_45567

-- Definitions for the number of rides and cost per ride
def ferris_wheel_rides : Nat := 7
def bumper_car_rides : Nat := 3
def ticket_per_ride : Nat := 5

-- Total number of rides
def total_rides : Nat := ferris_wheel_rides + bumper_car_rides

-- Total tickets used
def total_tickets : Nat := total_rides * ticket_per_ride

-- Theorem stating the number of tickets Billy used in total
theorem billy_tickets_used : total_tickets = 50 := by
  sorry

end billy_tickets_used_l455_45567


namespace eq_of_op_star_l455_45592

theorem eq_of_op_star (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a^b^2)^n = a^(bn)^2 ↔ n = 1 := by
sorry

end eq_of_op_star_l455_45592


namespace concentrate_amount_l455_45589

def parts_concentrate : ℤ := 1
def parts_water : ℤ := 5
def part_ratio : ℤ := parts_concentrate + parts_water -- Total parts
def servings : ℤ := 375
def volume_per_serving : ℤ := 150
def total_volume : ℤ := servings * volume_per_serving -- Total volume of orange juice
def volume_per_part : ℤ := total_volume / part_ratio -- Volume per part of mixture

theorem concentrate_amount :
  volume_per_part = 9375 :=
by
  sorry

end concentrate_amount_l455_45589


namespace solve_equation_l455_45580

theorem solve_equation (x : ℝ) : x * (x + 5)^3 * (5 - x) = 0 ↔ x = 0 ∨ x = -5 ∨ x = 5 := by
  sorry

end solve_equation_l455_45580


namespace slope_of_tangent_line_at_zero_l455_45520

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem slope_of_tangent_line_at_zero : (deriv f 0) = 1 := 
by
  sorry

end slope_of_tangent_line_at_zero_l455_45520


namespace prime_sum_of_composites_l455_45597

def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ m * k = n
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def can_be_expressed_as_sum_of_two_composites (p : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ p = a + b

theorem prime_sum_of_composites :
  can_be_expressed_as_sum_of_two_composites 13 ∧ 
  ∀ p : ℕ, is_prime p ∧ p > 13 → can_be_expressed_as_sum_of_two_composites p :=
by 
  sorry

end prime_sum_of_composites_l455_45597


namespace ophelia_average_pay_l455_45503

theorem ophelia_average_pay : ∀ (n : ℕ), 
  (51 + 100 * (n - 1)) / n = 93 ↔ n = 7 :=
by
  sorry

end ophelia_average_pay_l455_45503


namespace total_gifts_l455_45569

theorem total_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end total_gifts_l455_45569


namespace exists_zero_point_in_interval_l455_45571

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x - 2 * x

theorem exists_zero_point_in_interval :
  ∃ c ∈ Set.Ioo 1 (Real.pi / 2), f c = 0 := 
sorry

end exists_zero_point_in_interval_l455_45571


namespace max_chips_with_constraints_l455_45543

theorem max_chips_with_constraints (n : ℕ) (h1 : n > 0) 
  (h2 : ∀ i j : ℕ, (i < n) → (j = i + 10 ∨ j = i + 15) → ((i % 25) = 0 ∨ (j % 25) = 0)) :
  n ≤ 25 := 
sorry

end max_chips_with_constraints_l455_45543


namespace solution_exists_real_solution_31_l455_45500

theorem solution_exists_real_solution_31 :
  ∃ x : ℝ, (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10 ∧ 
            (x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15) :=
sorry

end solution_exists_real_solution_31_l455_45500
