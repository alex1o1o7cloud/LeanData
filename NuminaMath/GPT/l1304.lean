import Mathlib

namespace solve_equation_l1304_130430

theorem solve_equation :
  ∃ a b x : ℤ, 
  ((a * x^2 + b * x + 14)^2 + (b * x^2 + a * x + 8)^2 = 0) 
  ↔ (a = -6 ∧ b = -5 ∧ x = -2) :=
by {
  sorry
}

end solve_equation_l1304_130430


namespace log_product_eq_two_l1304_130455

open Real

theorem log_product_eq_two
  : log 5 / log 3 * log 6 / log 5 * log 9 / log 6 = 2 := by
  sorry

end log_product_eq_two_l1304_130455


namespace inequality_proof_l1304_130448

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b * c = 1) : 
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l1304_130448


namespace infinite_polynomial_pairs_l1304_130492

open Polynomial

theorem infinite_polynomial_pairs :
  ∀ n : ℕ, ∃ (fn gn : ℤ[X]), fn^2 - (X^4 - 2 * X) * gn^2 = 1 :=
sorry

end infinite_polynomial_pairs_l1304_130492


namespace least_element_of_special_set_l1304_130411

theorem least_element_of_special_set :
  ∃ T : Finset ℕ, T ⊆ Finset.range 16 ∧ T.card = 7 ∧
    (∀ {x y : ℕ}, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0)) ∧ 
    (∀ {z : ℕ}, z ∈ T → ∀ {x y : ℕ}, x ≠ y → x ∈ T → y ∈ T → z ≠ x + y) ∧
    ∀ (x : ℕ), x ∈ T → x ≥ 4 :=
sorry

end least_element_of_special_set_l1304_130411


namespace boat_shipments_divisor_l1304_130432

/-- 
Given:
1. There exists an integer B representing the number of boxes that can be divided into S equal shipments by boat.
2. B can be divided into 24 equal shipments by truck.
3. The smallest number of boxes B is 120.
Prove that S, the number of equal shipments by boat, is 60.
--/
theorem boat_shipments_divisor (B S : ℕ) (h1 : B % S = 0) (h2 : B % 24 = 0) (h3 : B = 120) : S = 60 := 
sorry

end boat_shipments_divisor_l1304_130432


namespace older_brother_allowance_l1304_130486

theorem older_brother_allowance 
  (sum_allowance : ℕ)
  (difference : ℕ)
  (total_sum : sum_allowance = 12000)
  (additional_amount : difference = 1000) :
  ∃ (older_brother_allowance younger_brother_allowance : ℕ), 
    older_brother_allowance = younger_brother_allowance + difference ∧
    younger_brother_allowance + older_brother_allowance = sum_allowance ∧
    older_brother_allowance = 6500 :=
by {
  sorry
}

end older_brother_allowance_l1304_130486


namespace missing_fraction_of_coins_l1304_130408

-- Defining the initial conditions
def total_coins (x : ℕ) := x
def lost_coins (x : ℕ) := (1 / 2) * x
def found_coins (x : ℕ) := (3 / 8) * x

-- Theorem statement
theorem missing_fraction_of_coins (x : ℕ) : 
  (total_coins x - lost_coins x + found_coins x) = (7 / 8) * x :=
by
  sorry  -- proof is omitted as per the instructions

end missing_fraction_of_coins_l1304_130408


namespace f_is_periodic_with_period_4a_l1304_130407

variable (f : ℝ → ℝ) (a : ℝ)

theorem f_is_periodic_with_period_4a (h : ∀ x : ℝ, f (x + a) = (1 + f x) / (1 - f x)) : ∀ x : ℝ, f (x + 4 * a) = f x :=
by
  sorry

end f_is_periodic_with_period_4a_l1304_130407


namespace depak_bank_account_l1304_130446

theorem depak_bank_account :
  ∃ (n : ℕ), (x + 1 = 6 * n) ∧ n = 1 → x = 5 := 
sorry

end depak_bank_account_l1304_130446


namespace complex_square_eq_l1304_130426

open Complex

theorem complex_square_eq {a b : ℝ} (h : (a + b * Complex.I)^2 = Complex.mk 3 4) : a^2 + b^2 = 5 :=
by {
  sorry
}

end complex_square_eq_l1304_130426


namespace range_of_a_l1304_130491

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, x ≤ y → f x a ≥ f y a) : a ≤ -3 :=
by
  sorry

end range_of_a_l1304_130491


namespace find_a_l1304_130483

noncomputable def A (a : ℝ) : Set ℝ := {-1, a^2 + 1, a^2 - 3}
noncomputable def B (a : ℝ) : Set ℝ := {-4, a - 1, a + 1}

theorem find_a (a : ℝ) (h : A a ∩ B a = {-2}) : a = -1 :=
sorry

end find_a_l1304_130483


namespace eval_f_nested_l1304_130499

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 0 then x + 1 else x ^ 2

theorem eval_f_nested : f (f (-2)) = 0 := by
  sorry

end eval_f_nested_l1304_130499


namespace left_side_value_l1304_130436

-- Define the relevant variables and conditions
variable (L R B : ℕ)

-- Assuming conditions
def sum_of_sides (L R B : ℕ) : Prop := L + R + B = 50
def right_side_relation (L R : ℕ) : Prop := R = L + 2
def base_value (B : ℕ) : Prop := B = 24

-- Main theorem statement
theorem left_side_value (L R B : ℕ) (h1 : sum_of_sides L R B) (h2 : right_side_relation L R) (h3 : base_value B) : L = 12 :=
sorry

end left_side_value_l1304_130436


namespace find_n_l1304_130460

theorem find_n (n : ℕ) (h : 1 < n) :
  (∀ a b : ℕ, Nat.gcd a b = 1 → (a % n = b % n ↔ (a * b) % n = 1)) →
  (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 12 ∨ n = 24) :=
by
  sorry

end find_n_l1304_130460


namespace smallest_c_geometric_arithmetic_progression_l1304_130410

theorem smallest_c_geometric_arithmetic_progression (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : 0 < c) 
(h4 : b ^ 2 = a * c) (h5 : a + b = 2 * c) : c = 1 :=
sorry

end smallest_c_geometric_arithmetic_progression_l1304_130410


namespace simplify_expression_l1304_130402

variable (y : ℝ)

theorem simplify_expression : (5 * y + 6 * y + 7 * y + 2) = (18 * y + 2) := 
by
  sorry

end simplify_expression_l1304_130402


namespace trajectory_midpoint_l1304_130493

/-- Let A and B be two moving points on the circle x^2 + y^2 = 4, and AB = 2. 
    The equation of the trajectory of the midpoint M of the line segment AB is x^2 + y^2 = 3. -/
theorem trajectory_midpoint (A B : ℝ × ℝ) (M : ℝ × ℝ)
    (hA : A.1^2 + A.2^2 = 4)
    (hB : B.1^2 + B.2^2 = 4)
    (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4)
    (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
    M.1^2 + M.2^2 = 3 :=
sorry

end trajectory_midpoint_l1304_130493


namespace remainder_x1001_mod_poly_l1304_130404

noncomputable def remainder_poly_div (n k : ℕ) (f g : Polynomial ℚ) : Polynomial ℚ :=
  Polynomial.modByMonic f g

theorem remainder_x1001_mod_poly :
  remainder_poly_div 1001 3 (Polynomial.X ^ 1001) (Polynomial.X ^ 3 - Polynomial.X ^ 2 - Polynomial.X + 1) = Polynomial.X ^ 2 :=
by
  sorry

end remainder_x1001_mod_poly_l1304_130404


namespace harrys_morning_routine_time_l1304_130487

theorem harrys_morning_routine_time :
  (15 + 20 + 25 + 2 * 15 = 90) :=
by
  sorry

end harrys_morning_routine_time_l1304_130487


namespace units_digit_17_pow_39_l1304_130437

theorem units_digit_17_pow_39 : 
  ∃ d : ℕ, d < 10 ∧ (17^39 % 10 = d) ∧ d = 3 :=
by
  sorry

end units_digit_17_pow_39_l1304_130437


namespace laboratory_spent_on_flasks_l1304_130421

theorem laboratory_spent_on_flasks:
  ∀ (F : ℝ), (∃ cost_test_tubes : ℝ, cost_test_tubes = (2 / 3) * F) →
  (∃ cost_safety_gear : ℝ, cost_safety_gear = (1 / 3) * F) →
  2 * F = 300 → F = 150 :=
by
  intros F h1 h2 h3
  sorry

end laboratory_spent_on_flasks_l1304_130421


namespace comic_books_collection_l1304_130405

theorem comic_books_collection (initial_ky: ℕ) (rate_ky: ℕ) (initial_la: ℕ) (rate_la: ℕ) (months: ℕ) :
  initial_ky = 50 → rate_ky = 1 → initial_la = 20 → rate_la = 7 → months = 33 →
  initial_la + rate_la * months = 3 * (initial_ky + rate_ky * months) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end comic_books_collection_l1304_130405


namespace find_f_zero_l1304_130480

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_zero (a : ℝ) (h1 : ∀ x : ℝ, f (x - a) = x^3 + 1)
  (h2 : ∀ x : ℝ, f x + f (2 - x) = 2) : 
  f 0 = 0 :=
sorry

end find_f_zero_l1304_130480


namespace complement_U_A_correct_l1304_130434

-- Step 1: Define the universal set U
def U (x : ℝ) := x > 0

-- Step 2: Define the set A
def A (x : ℝ) := 0 < x ∧ x < 1

-- Step 3: Define the complement of A in U
def complement_U_A (x : ℝ) := U x ∧ ¬ A x

-- Step 4: Define the expected complement
def expected_complement (x : ℝ) := x ≥ 1

-- Step 5: The proof problem statement
theorem complement_U_A_correct (x : ℝ) : complement_U_A x = expected_complement x := by
  sorry

end complement_U_A_correct_l1304_130434


namespace solution_l1304_130485

noncomputable def f (x : ℝ) := 
  10 / (Real.sqrt (x - 5) - 10) + 
  2 / (Real.sqrt (x - 5) - 5) + 
  9 / (Real.sqrt (x - 5) + 5) + 
  18 / (Real.sqrt (x - 5) + 10)

theorem solution : 
  f (1230 / 121) = 0 := sorry

end solution_l1304_130485


namespace distance_between_towns_in_kilometers_l1304_130445

theorem distance_between_towns_in_kilometers :
  (20 * 5) * 1.60934 = 160.934 :=
by
  sorry

end distance_between_towns_in_kilometers_l1304_130445


namespace find_c_for_square_of_binomial_l1304_130470

theorem find_c_for_square_of_binomial (c : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 + 50 * x + c = (x + b)^2) → c = 625 :=
by
  intro h
  obtain ⟨b, h⟩ := h
  sorry

end find_c_for_square_of_binomial_l1304_130470


namespace find_f_inv_64_l1304_130416

noncomputable def f : ℝ → ℝ :=
  sorry  -- We don't know the exact form of f.

axiom f_property_1 : f 5 = 2

axiom f_property_2 : ∀ x : ℝ, f (2 * x) = 2 * f x

def f_inv (y : ℝ) : ℝ :=
  sorry  -- We define the inverse function in terms of y.

theorem find_f_inv_64 : f_inv 64 = 160 :=
by {
  -- Main statement to be proved.
  sorry
}

end find_f_inv_64_l1304_130416


namespace tan_120_deg_l1304_130447

theorem tan_120_deg : Real.tan (120 * Real.pi / 180) = -Real.sqrt 3 := by
  sorry

end tan_120_deg_l1304_130447


namespace sid_spent_on_computer_accessories_l1304_130478

def initial_money : ℕ := 48
def snacks_cost : ℕ := 8
def remaining_money_more_than_half : ℕ := 4

theorem sid_spent_on_computer_accessories : 
  ∀ (m s r : ℕ), m = initial_money → s = snacks_cost → r = remaining_money_more_than_half →
  m - (r + m / 2 + s) = 12 :=
by
  intros m s r h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sid_spent_on_computer_accessories_l1304_130478


namespace smallest_number_is_a_l1304_130433

def smallest_number_among_options : ℤ :=
  let a: ℤ := -3
  let b: ℤ := 0
  let c: ℤ := -(-1)
  let d: ℤ := (-1)^2
  min a (min b (min c d))

theorem smallest_number_is_a : smallest_number_among_options = -3 :=
  by
    sorry

end smallest_number_is_a_l1304_130433


namespace opposite_of_neg2_is_2_l1304_130419

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_neg2_is_2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_is_2_l1304_130419


namespace total_distance_covered_l1304_130496

-- Define the basic conditions
def num_marathons : Nat := 15
def miles_per_marathon : Nat := 26
def yards_per_marathon : Nat := 385
def yards_per_mile : Nat := 1760

-- Define the total miles and total yards covered
def total_miles : Nat := num_marathons * miles_per_marathon
def total_yards : Nat := num_marathons * yards_per_marathon

-- Convert excess yards into miles and calculate the remaining yards
def extra_miles : Nat := total_yards / yards_per_mile
def remaining_yards : Nat := total_yards % yards_per_mile

-- Compute the final total distance
def total_distance_miles : Nat := total_miles + extra_miles
def total_distance_yards : Nat := remaining_yards

-- The theorem that needs to be proven
theorem total_distance_covered :
  total_distance_miles = 393 ∧ total_distance_yards = 495 :=
by
  sorry

end total_distance_covered_l1304_130496


namespace find_f_two_l1304_130412

-- The function f is defined on (0, +∞) and takes positive values
noncomputable def f : ℝ → ℝ := sorry

-- The given condition that areas of triangle AOB and trapezoid ABH_BH_A are equal
axiom equalAreas (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) : 
  (1 / 2) * |x1 * f x2 - x2 * f x1| = (1 / 2) * (x2 - x1) * (f x1 + f x2)

-- The specific given value
axiom f_one : f 1 = 4

-- The theorem we need to prove
theorem find_f_two : f 2 = 2 :=
sorry

end find_f_two_l1304_130412


namespace sarah_cupcakes_l1304_130477

theorem sarah_cupcakes (c k d : ℕ) (h1 : c + k = 6) (h2 : 90 * c + 40 * k = 100 * d) : c = 4 ∨ c = 6 :=
by {
  sorry -- Proof is omitted as requested.
}

end sarah_cupcakes_l1304_130477


namespace sum_of_remainders_l1304_130431

theorem sum_of_remainders (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 5) 
  (h3 : c % 30 = 20) : 
  (a + b + c) % 30 = 10 := 
by sorry

end sum_of_remainders_l1304_130431


namespace ice_cream_depth_l1304_130457

theorem ice_cream_depth 
  (r_sphere : ℝ) 
  (r_cylinder : ℝ) 
  (h_cylinder : ℝ) 
  (V_sphere : ℝ) 
  (V_cylinder : ℝ) 
  (constant_density : V_sphere = V_cylinder)
  (r_sphere_eq : r_sphere = 2) 
  (r_cylinder_eq : r_cylinder = 8) 
  (V_sphere_def : V_sphere = (4 / 3) * Real.pi * r_sphere^3) 
  (V_cylinder_def : V_cylinder = Real.pi * r_cylinder^2 * h_cylinder) 
  : h_cylinder = 1 / 6 := 
by 
  sorry

end ice_cream_depth_l1304_130457


namespace olivia_possible_amount_l1304_130497

theorem olivia_possible_amount (k : ℕ) :
  ∃ k : ℕ, 1 + 79 * k = 1984 :=
by
  -- Prove that there exists a non-negative integer k such that the equation holds
  sorry

end olivia_possible_amount_l1304_130497


namespace min_value_x_add_one_div_y_l1304_130467

theorem min_value_x_add_one_div_y (x y : ℝ) (h1 : x > 1) (h2 : x - y = 1) : 
x + 1 / y ≥ 3 :=
sorry

end min_value_x_add_one_div_y_l1304_130467


namespace a_work_days_alone_l1304_130459

-- Definitions based on conditions
def work_days_a   (a: ℝ)    : Prop := ∃ (x:ℝ), a = x
def work_days_b   (b: ℝ)    : Prop := b = 36
def alternate_work (a b W x: ℝ) : Prop := 9 * (W / 36 + W / x) = W ∧ x > 0

-- The main theorem to prove
theorem a_work_days_alone (x W: ℝ) (b: ℝ) (h_work_days_b: work_days_b b)
                          (h_alternate_work: alternate_work a b W x) : 
                          work_days_a a → a = 12 :=
by sorry

end a_work_days_alone_l1304_130459


namespace no_solution_exists_l1304_130428

theorem no_solution_exists : 
  ¬ ∃ (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0), 
    45 * x = (35 / 100) * 900 ∧
    y^2 + x = 100 ∧
    z = x^3 * y - (2 * x + 1) / (y + 4) :=
by
  sorry

end no_solution_exists_l1304_130428


namespace club_membership_l1304_130417

theorem club_membership (n : ℕ) : 
  n ≡ 6 [MOD 10] → n ≡ 6 [MOD 11] → 200 ≤ n ∧ n ≤ 300 → n = 226 :=
by
  intros h1 h2 h3
  sorry

end club_membership_l1304_130417


namespace number_exceeds_percent_l1304_130400

theorem number_exceeds_percent (x : ℝ) (h : x = 0.12 * x + 52.8) : x = 60 :=
by {
  sorry
}

end number_exceeds_percent_l1304_130400


namespace flower_bed_dimensions_l1304_130418

variable (l w : ℕ)

theorem flower_bed_dimensions :
  (l + 3) * (w + 2) = l * w + 64 →
  (l + 2) * (w + 3) = l * w + 68 →
  l = 14 ∧ w = 10 :=
by
  intro h1 h2
  sorry

end flower_bed_dimensions_l1304_130418


namespace ratio_population_X_to_Z_l1304_130476

-- Given definitions
def population_of_Z : ℕ := sorry
def population_of_Y : ℕ := 2 * population_of_Z
def population_of_X : ℕ := 5 * population_of_Y

-- Theorem to prove
theorem ratio_population_X_to_Z : population_of_X / population_of_Z = 10 :=
by
  sorry

end ratio_population_X_to_Z_l1304_130476


namespace rachel_earnings_one_hour_l1304_130472

-- Define Rachel's hourly wage
def rachelWage : ℝ := 12.00

-- Define the number of people Rachel serves in one hour
def peopleServed : ℕ := 20

-- Define the tip amount per person
def tipPerPerson : ℝ := 1.25

-- Calculate the total tips received
def totalTips : ℝ := (peopleServed : ℝ) * tipPerPerson

-- Calculate the total amount Rachel makes in one hour
def totalEarnings : ℝ := rachelWage + totalTips

-- The theorem to state Rachel's total earnings in one hour
theorem rachel_earnings_one_hour : totalEarnings = 37.00 := 
by
  sorry

end rachel_earnings_one_hour_l1304_130472


namespace expression_value_l1304_130406

theorem expression_value : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end expression_value_l1304_130406


namespace misread_weight_l1304_130441

-- Definitions based on given conditions in part (a)
def initial_avg_weight : ℝ := 58.4
def num_boys : ℕ := 20
def correct_weight : ℝ := 61
def correct_avg_weight : ℝ := 58.65

-- The Lean theorem statement that needs to be proved
theorem misread_weight :
  let incorrect_total_weight := initial_avg_weight * num_boys
  let correct_total_weight := correct_avg_weight * num_boys
  let weight_diff := correct_total_weight - incorrect_total_weight
  correct_weight - weight_diff = 56 := sorry

end misread_weight_l1304_130441


namespace senate_subcommittee_l1304_130481

/-- 
Proof of the number of ways to form a Senate subcommittee consisting of 7 Republicans
and 2 Democrats from the available 12 Republicans and 6 Democrats.
-/
theorem senate_subcommittee (R D : ℕ) (choose_R choose_D : ℕ) (hR : R = 12) (hD : D = 6) 
  (h_choose_R : choose_R = 7) (h_choose_D : choose_D = 2) : 
  (Nat.choose R choose_R) * (Nat.choose D choose_D) = 11880 := by
  sorry

end senate_subcommittee_l1304_130481


namespace words_per_page_l1304_130440

theorem words_per_page (p : ℕ) (hp : p ≤ 120) (h : 150 * p ≡ 210 [MOD 221]) : p = 98 := by
  sorry

end words_per_page_l1304_130440


namespace range_of_k_l1304_130403

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x - 2| + |x - 3| > |k - 1|) → 0 < k ∧ k < 2 :=
by
  sorry

end range_of_k_l1304_130403


namespace cube_volume_split_l1304_130414

theorem cube_volume_split (x y z : ℝ) (h : x > 0) :
  ∃ y z : ℝ, y > 0 ∧ z > 0 ∧ y^3 + z^3 = x^3 :=
sorry

end cube_volume_split_l1304_130414


namespace product_bc_l1304_130468

theorem product_bc {b c : ℤ} (h1 : ∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) :
    b * c = 110 :=
sorry

end product_bc_l1304_130468


namespace cos_sin_exp_l1304_130444

theorem cos_sin_exp (n : ℕ) (t : ℝ) (h : n ≤ 1000) :
  (Complex.exp (t * Complex.I)) ^ n = Complex.exp (n * t * Complex.I) :=
by
  sorry

end cos_sin_exp_l1304_130444


namespace seeds_total_l1304_130424

-- Define the conditions as given in the problem.
def Bom_seeds : ℕ := 300
def Gwi_seeds : ℕ := Bom_seeds + 40
def Yeon_seeds : ℕ := 3 * Gwi_seeds

-- Lean statement to prove the total number of seeds.
theorem seeds_total : Bom_seeds + Gwi_seeds + Yeon_seeds = 1660 := 
by
  -- Assuming all given definitions and conditions are true,
  -- we aim to prove the final theorem statement.
  sorry

end seeds_total_l1304_130424


namespace train_speed_is_28_l1304_130495

-- Define the given conditions
def train_length : ℕ := 1200
def overbridge_length : ℕ := 200
def crossing_time : ℕ := 50

-- Define the total distance
def total_distance := train_length + overbridge_length

-- Define the speed calculation function
def speed (distance time : ℕ) : ℕ := 
  distance / time

-- State the theorem to be proven
theorem train_speed_is_28 : speed total_distance crossing_time = 28 := 
by
  -- Proof to be provided
  sorry

end train_speed_is_28_l1304_130495


namespace age_difference_l1304_130415

theorem age_difference (A B C : ℕ) (h1 : B = 10) (h2 : B = 2 * C) (h3 : A + B + C = 27) : A - B = 2 :=
 by
  sorry

end age_difference_l1304_130415


namespace necessary_and_sufficient_condition_for_x2_ne_y2_l1304_130435

theorem necessary_and_sufficient_condition_for_x2_ne_y2 (x y : ℤ) :
  (x ^ 2 ≠ y ^ 2) ↔ (x ≠ y ∧ x ≠ -y) :=
by
  sorry

end necessary_and_sufficient_condition_for_x2_ne_y2_l1304_130435


namespace fraction_power_equals_l1304_130438

theorem fraction_power_equals :
  (5 / 7) ^ 7 = (78125 : ℚ) / 823543 := 
by
  sorry

end fraction_power_equals_l1304_130438


namespace area_of_triangle_PQR_l1304_130475

structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 2, y := 2 }
def Q : Point := { x := 7, y := 2 }
def R : Point := { x := 5, y := 9 }

noncomputable def triangleArea (A B C : Point) : ℝ :=
  (1 / 2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem area_of_triangle_PQR : triangleArea P Q R = 17.5 := by
  sorry

end area_of_triangle_PQR_l1304_130475


namespace bird_families_left_l1304_130456

theorem bird_families_left (B_initial B_flew_away : ℕ) (h_initial : B_initial = 41) (h_flew_away : B_flew_away = 27) :
  B_initial - B_flew_away = 14 :=
by
  sorry

end bird_families_left_l1304_130456


namespace ratio_of_sums_l1304_130442

/-- Define the relevant arithmetic sequences and sums -/

-- Sequence 1: 3, 6, 9, ..., 45
def seq1 : ℕ → ℕ
| n => 3 * n + 3

-- Sequence 2: 4, 8, 12, ..., 64
def seq2 : ℕ → ℕ
| n => 4 * n + 4

-- Sum function for arithmetic sequences
def sum_arith_seq (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n-1) * d) / 2

noncomputable def sum_seq1 : ℕ := sum_arith_seq 3 3 15 -- 3 + 6 + ... + 45
noncomputable def sum_seq2 : ℕ := sum_arith_seq 4 4 16 -- 4 + 8 + ... + 64

-- Prove that the ratio of sums is 45/68
theorem ratio_of_sums : (sum_seq1 : ℚ) / sum_seq2 = 45 / 68 :=
  sorry

end ratio_of_sums_l1304_130442


namespace total_distance_hiked_east_l1304_130429

-- Define Annika's constant rate of hiking
def constant_rate : ℝ := 10 -- minutes per kilometer

-- Define already hiked distance
def distance_hiked : ℝ := 2.75 -- kilometers

-- Define total available time to return
def total_time : ℝ := 45 -- minutes

-- Prove that the total distance hiked east is 4.5 kilometers
theorem total_distance_hiked_east : distance_hiked + (total_time - distance_hiked * constant_rate) / constant_rate = 4.5 :=
by
  sorry

end total_distance_hiked_east_l1304_130429


namespace singh_gain_l1304_130427

def initial_amounts (B A S : ℕ) : Prop :=
  B = 70 ∧ A = 70 ∧ S = 70

def ratio_Ashtikar_Singh (A S : ℕ) : Prop :=
  2 * A = S

def ratio_Singh_Bhatia (S B : ℕ) : Prop :=
  4 * B = S

def total_conservation (A S B : ℕ) : Prop :=
  A + S + B = 210

theorem singh_gain : ∀ B A S fA fB fS : ℕ,
  initial_amounts B A S →
  ratio_Ashtikar_Singh fA fS →
  ratio_Singh_Bhatia fS fB →
  total_conservation fA fS fB →
  fS - S = 50 :=
by
  intros B A S fA fB fS
  intros i rA rS tC
  sorry

end singh_gain_l1304_130427


namespace number_of_dozen_eggs_to_mall_l1304_130471

-- Define the conditions as assumptions
def number_of_dozen_eggs_collected (x : Nat) : Prop :=
  x = 2 * 8

def number_of_dozen_eggs_to_market (x : Nat) : Prop :=
  x = 3

def number_of_dozen_eggs_for_pie (x : Nat) : Prop :=
  x = 4

def number_of_dozen_eggs_to_charity (x : Nat) : Prop :=
  x = 4

-- The theorem stating the answer to the problem
theorem number_of_dozen_eggs_to_mall 
  (h1 : ∃ x, number_of_dozen_eggs_collected x)
  (h2 : ∃ x, number_of_dozen_eggs_to_market x)
  (h3 : ∃ x, number_of_dozen_eggs_for_pie x)
  (h4 : ∃ x, number_of_dozen_eggs_to_charity x)
  : ∃ z, z = 5 := 
sorry

end number_of_dozen_eggs_to_mall_l1304_130471


namespace birds_initially_l1304_130469

-- Definitions of the conditions
def initial_birds (B : Nat) := B
def initial_storks := 4
def additional_storks := 6
def total := 13

-- The theorem we need to prove
theorem birds_initially (B : Nat) (h : initial_birds B + initial_storks + additional_storks = total) : initial_birds B = 3 :=
by
  -- The proof can go here
  sorry

end birds_initially_l1304_130469


namespace guacamole_serving_and_cost_l1304_130452

theorem guacamole_serving_and_cost 
  (initial_avocados : ℕ) 
  (additional_avocados : ℕ) 
  (avocados_per_serving : ℕ) 
  (x : ℝ) 
  (h_initial : initial_avocados = 5) 
  (h_additional : additional_avocados = 4) 
  (h_serving : avocados_per_serving = 3) :
  (initial_avocados + additional_avocados) / avocados_per_serving = 3 
  ∧ additional_avocados * x = 4 * x := by
  sorry

end guacamole_serving_and_cost_l1304_130452


namespace inverse_proportion_neg_k_l1304_130463

theorem inverse_proportion_neg_k (x1 x2 y1 y2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (h3 : y1 > y2) :
  ∃ k : ℝ, k < 0 ∧ (∀ x, (x = x1 → y1 = k / x) ∧ (x = x2 → y2 = k / x)) := by
  use -1
  sorry

end inverse_proportion_neg_k_l1304_130463


namespace butterfly_black_dots_l1304_130489

theorem butterfly_black_dots (b f : ℕ) (total_butterflies : b = 397) (total_black_dots : f = 4764) : f / b = 12 :=
by
  sorry

end butterfly_black_dots_l1304_130489


namespace Trevor_tip_l1304_130450

variable (Uber Lyft Taxi : ℕ)
variable (TotalCost : ℕ)

theorem Trevor_tip 
  (h1 : Uber = Lyft + 3) 
  (h2 : Lyft = Taxi + 4) 
  (h3 : Uber = 22) 
  (h4 : TotalCost = 18)
  (h5 : Taxi = 15) :
  (TotalCost - Taxi) * 100 / Taxi = 20 := by
  sorry

end Trevor_tip_l1304_130450


namespace square_perimeter_l1304_130439

theorem square_perimeter (s : ℝ) (h : s^2 = s) : 4 * s = 4 :=
by
  sorry

end square_perimeter_l1304_130439


namespace minimal_reciprocal_sum_l1304_130479

theorem minimal_reciprocal_sum (m n : ℕ) (hm : m > 0) (hn : n > 0) :
    (4 / m) + (1 / n) = (30 / (m * n)) → m = 10 ∧ n = 5 :=
sorry

end minimal_reciprocal_sum_l1304_130479


namespace parallel_vectors_eq_l1304_130465

theorem parallel_vectors_eq (m : ℤ) (h : (m, 4) = (3 * k, -2 * k)) : m = -6 :=
by
  sorry

end parallel_vectors_eq_l1304_130465


namespace roots_of_cubic_equation_l1304_130425

theorem roots_of_cubic_equation 
  (k m : ℝ) 
  (h : ∀r1 r2 r3: ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
  r1 + r2 + r3 = 7 ∧ r1 * r2 * r3 = m ∧ (r1 * r2 + r2 * r3 + r1 * r3) = k) : 
  k + m = 22 := sorry

end roots_of_cubic_equation_l1304_130425


namespace zeros_at_end_of_product1_value_of_product2_l1304_130464

-- Definitions and conditions
def product1 := 360 * 5
def product2 := 250 * 4

-- Statements of the proof problems
theorem zeros_at_end_of_product1 : Nat.digits 10 product1 = [0, 0, 8, 1] := by
  sorry

theorem value_of_product2 : product2 = 1000 := by
  sorry

end zeros_at_end_of_product1_value_of_product2_l1304_130464


namespace quadratic_inequality_solution_l1304_130401

theorem quadratic_inequality_solution : 
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x < -1 ∨ x > 3 / 2} :=
by
  sorry

end quadratic_inequality_solution_l1304_130401


namespace least_perimeter_of_triangle_l1304_130490

theorem least_perimeter_of_triangle (c : ℕ) (h1 : 24 + 51 > c) (h2 : c > 27) : 24 + 51 + c = 103 :=
by
  sorry

end least_perimeter_of_triangle_l1304_130490


namespace part1_part2_part3_l1304_130494

variable {a b c : ℝ}

-- Part (1)
theorem part1 (a b c : ℝ) : a * (b - c) ^ 2 + b * (c - a) ^ 2 + c * (a - b) ^ 2 + 4 * a * b * c > a ^ 3 + b ^ 3 + c ^ 3 :=
sorry

-- Part (2)
theorem part2 (a b c : ℝ) : 2 * a ^ 2 * b ^ 2 + 2 * b ^ 2 * c ^ 2 + 2 * c ^ 2 * a ^ 2 > a ^ 4 + b ^ 4 + c ^ 4 :=
sorry

-- Part (3)
theorem part3 (a b c : ℝ) : 2 * a * b + 2 * b * c + 2 * c * a > a ^ 2 + b ^ 2 + c ^ 2 :=
sorry

end part1_part2_part3_l1304_130494


namespace evaluate_expression_l1304_130451

-- Defining the conditions for the cosine and sine values
def cos_0 : Real := 1
def sin_3pi_2 : Real := -1

-- Proving the given expression equals -1
theorem evaluate_expression : 3 * cos_0 + 4 * sin_3pi_2 = -1 :=
by 
  -- Given the definitions, this will simplify as expected.
  sorry

end evaluate_expression_l1304_130451


namespace box_dimensions_sum_l1304_130461

theorem box_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 30) 
  (h2 : A * C = 50)
  (h3 : B * C = 90) : 
  A + B + C = (58 * Real.sqrt 15) / 3 :=
sorry

end box_dimensions_sum_l1304_130461


namespace correct_option_l1304_130449

-- Define the given conditions
def a : ℕ := 7^5
def b : ℕ := 5^7

-- State the theorem to be proven
theorem correct_option : a^7 * b^5 = 35^35 := by
  -- insert the proof here
  sorry

end correct_option_l1304_130449


namespace range_of_a_l1304_130420

noncomputable def f (x a : ℝ) : ℝ := x * abs (x - a)

theorem range_of_a (a : ℝ) :
  (∀ (x1 x2 : ℝ), 3 ≤ x1 ∧ 3 ≤ x2 ∧ x1 ≠ x2 → (x1 - x2) * (f x1 a - f x2 a) > 0) → a ≤ 3 :=
by sorry

end range_of_a_l1304_130420


namespace count_divisible_by_five_l1304_130488

theorem count_divisible_by_five : 
  ∃ n : ℕ, (∀ x, 1 ≤ x ∧ x ≤ 1000 → (x % 5 = 0 → (n = 200))) :=
by
  sorry

end count_divisible_by_five_l1304_130488


namespace number_of_voters_in_election_l1304_130498

theorem number_of_voters_in_election
  (total_membership : ℕ)
  (votes_cast : ℕ)
  (winning_percentage_cast : ℚ)
  (percentage_of_total : ℚ)
  (h_total : total_membership = 1600)
  (h_winning_percentage : winning_percentage_cast = 0.60)
  (h_percentage_of_total : percentage_of_total = 0.196875)
  (h_votes : winning_percentage_cast * votes_cast = percentage_of_total * total_membership) :
  votes_cast = 525 :=
by
  sorry

end number_of_voters_in_election_l1304_130498


namespace no_other_distinct_prime_products_l1304_130484

theorem no_other_distinct_prime_products :
  ∀ (q1 q2 q3 : Nat), 
  Prime q1 ∧ Prime q2 ∧ Prime q3 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧ q1 * q2 * q3 ≠ 17 * 11 * 23 → 
  q1 + q2 + q3 ≠ 51 :=
by
  intros q1 q2 q3 h
  sorry

end no_other_distinct_prime_products_l1304_130484


namespace nova_monthly_donation_l1304_130474

def total_annual_donation : ℕ := 20484
def months_in_year : ℕ := 12
def monthly_donation : ℕ := total_annual_donation / months_in_year

theorem nova_monthly_donation :
  monthly_donation = 1707 :=
by
  unfold monthly_donation
  sorry

end nova_monthly_donation_l1304_130474


namespace students_taking_all_three_l1304_130466

-- Definitions and Conditions
def total_students : ℕ := 25
def coding_students : ℕ := 12
def chess_students : ℕ := 15
def photography_students : ℕ := 10
def at_least_two_classes : ℕ := 10

-- Request to prove: Number of students taking all three classes
theorem students_taking_all_three (x y w z : ℕ) :
  (x + y + z + w = 10) →
  (coding_students - (10 - y) + chess_students - (10 - w) + (10 - x) = 21) →
  z = 4 :=
by
  intros
  -- Proof will go here
  sorry

end students_taking_all_three_l1304_130466


namespace alice_probability_same_color_l1304_130482

def total_ways_to_draw : ℕ := 
  Nat.choose 9 3 * Nat.choose 6 3 * Nat.choose 3 3

def favorable_outcomes_for_alice : ℕ := 
  3 * Nat.choose 6 3 * Nat.choose 3 3

def probability_alice_same_color : ℚ := 
  favorable_outcomes_for_alice / total_ways_to_draw

theorem alice_probability_same_color : probability_alice_same_color = 1 / 28 := 
by
  -- Proof is omitted as per instructions
  sorry

end alice_probability_same_color_l1304_130482


namespace mrs_hilt_chapters_read_l1304_130454

def number_of_books : ℝ := 4.0
def chapters_per_book : ℝ := 4.25
def total_chapters_read : ℝ := number_of_books * chapters_per_book

theorem mrs_hilt_chapters_read : total_chapters_read = 17 :=
by
  unfold total_chapters_read
  norm_num
  sorry

end mrs_hilt_chapters_read_l1304_130454


namespace number_of_testing_methods_l1304_130473

-- Definitions based on conditions
def num_genuine_items : ℕ := 6
def num_defective_items : ℕ := 4
def total_tests : ℕ := 5

-- Theorem stating the number of testing methods
theorem number_of_testing_methods 
    (h1 : total_tests = 5) 
    (h2 : num_genuine_items = 6) 
    (h3 : num_defective_items = 4) :
    ∃ n : ℕ, n = 576 := 
sorry

end number_of_testing_methods_l1304_130473


namespace quadratic_function_even_l1304_130422

theorem quadratic_function_even (a b : ℝ) (h1 : ∀ x : ℝ, x^2 + (a-1)*x + a + b = x^2 - (a-1)*x + a + b) (h2 : 4 + (a-1)*2 + a + b = 0) : a + b = -4 := 
sorry

end quadratic_function_even_l1304_130422


namespace average_speed_round_trip_l1304_130413

theorem average_speed_round_trip (v1 v2 : ℝ) (h1 : v1 = 60) (h2 : v2 = 100) :
  (2 * v1 * v2) / (v1 + v2) = 75 :=
by
  sorry

end average_speed_round_trip_l1304_130413


namespace ellipse_nec_but_not_suff_l1304_130409

-- Definitions and conditions
def isEllipse (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c

/-- Given that the sum of the distances from a moving point P in the plane to two fixed points is constant,
the condition is necessary but not sufficient for the trajectory of the moving point P being an ellipse. -/
theorem ellipse_nec_but_not_suff (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (c : ℝ) :
  (∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c) →
  (c > dist F1 F2 → ¬ isEllipse P F1 F2) ∧ (isEllipse P F1 F2 → ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c) :=
by
  sorry

end ellipse_nec_but_not_suff_l1304_130409


namespace dan_job_time_l1304_130453

theorem dan_job_time
  (Annie_time : ℝ) (Dan_work_time : ℝ) (Annie_work_remain : ℝ) (total_work : ℝ)
  (Annie_time_cond : Annie_time = 9)
  (Dan_work_time_cond : Dan_work_time = 8)
  (Annie_work_remain_cond : Annie_work_remain = 3.0000000000000004)
  (total_work_cond : total_work = 1) :
  ∃ (Dan_time : ℝ), Dan_time = 12 := by
  sorry

end dan_job_time_l1304_130453


namespace largest_integer_is_59_l1304_130443

theorem largest_integer_is_59 
  {w x y z : ℤ} 
  (h₁ : (w + x + y) / 3 = 32)
  (h₂ : (w + x + z) / 3 = 39)
  (h₃ : (w + y + z) / 3 = 40)
  (h₄ : (x + y + z) / 3 = 44) :
  max (max w x) (max y z) = 59 :=
by {
  sorry
}

end largest_integer_is_59_l1304_130443


namespace total_cost_of_projectors_and_computers_l1304_130458

theorem total_cost_of_projectors_and_computers :
  let n_p := 8
  let c_p := 7500
  let n_c := 32
  let c_c := 3600
  (n_p * c_p + n_c * c_c) = 175200 := by
  let n_p := 8
  let c_p := 7500
  let n_c := 32
  let c_c := 3600
  sorry 

end total_cost_of_projectors_and_computers_l1304_130458


namespace find_k_l1304_130423

theorem find_k (k : ℝ) (A B : ℝ → ℝ)
  (hA : ∀ x, A x = 2 * x^2 + k * x - 6 * x)
  (hB : ∀ x, B x = -x^2 + k * x - 1)
  (hIndependent : ∀ x, ∃ C : ℝ, A x + 2 * B x = C) :
  k = 2 :=
by 
  sorry

end find_k_l1304_130423


namespace Piglet_ate_one_l1304_130462

theorem Piglet_ate_one (V S K P : ℕ) (h1 : V + S + K + P = 70)
  (h2 : S + K = 45) (h3 : V > S) (h4 : V > K) (h5 : V > P) 
  (h6 : V ≥ 1) (h7 : S ≥ 1) (h8 : K ≥ 1) (h9 : P ≥ 1) : P = 1 :=
sorry

end Piglet_ate_one_l1304_130462
