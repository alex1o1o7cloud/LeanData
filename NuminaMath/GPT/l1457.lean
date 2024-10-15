import Mathlib

namespace NUMINAMATH_GPT_logic_problem_l1457_145742

variable (p q : Prop)

theorem logic_problem (h₁ : ¬ p) (h₂ : p ∨ q) : p = False ∧ q = True :=
by
  sorry

end NUMINAMATH_GPT_logic_problem_l1457_145742


namespace NUMINAMATH_GPT_Sally_quarters_l1457_145753

theorem Sally_quarters : 760 + 418 - 152 = 1026 := 
by norm_num

end NUMINAMATH_GPT_Sally_quarters_l1457_145753


namespace NUMINAMATH_GPT_sum_of_lengths_of_edges_geometric_progression_l1457_145793

theorem sum_of_lengths_of_edges_geometric_progression :
  ∃ (a r : ℝ), (a / r) * a * (a * r) = 8 ∧ 2 * (a / r * a + a * a * r + a * r * a / r) = 32 ∧ 
  4 * ((a / r) + a + (a * r)) = 32 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_lengths_of_edges_geometric_progression_l1457_145793


namespace NUMINAMATH_GPT_forest_leaves_count_correct_l1457_145721

def number_of_trees : ℕ := 20
def number_of_main_branches_per_tree : ℕ := 15
def number_of_sub_branches_per_main_branch : ℕ := 25
def number_of_tertiary_branches_per_sub_branch : ℕ := 30
def number_of_leaves_per_sub_branch : ℕ := 75
def number_of_leaves_per_tertiary_branch : ℕ := 45

def total_leaves_on_sub_branches_per_tree :=
  number_of_main_branches_per_tree * number_of_sub_branches_per_main_branch * number_of_leaves_per_sub_branch

def total_sub_branches_per_tree :=
  number_of_main_branches_per_tree * number_of_sub_branches_per_main_branch

def total_leaves_on_tertiary_branches_per_tree :=
  total_sub_branches_per_tree * number_of_tertiary_branches_per_sub_branch * number_of_leaves_per_tertiary_branch

def total_leaves_per_tree :=
  total_leaves_on_sub_branches_per_tree + total_leaves_on_tertiary_branches_per_tree

def total_leaves_in_forest :=
  total_leaves_per_tree * number_of_trees

theorem forest_leaves_count_correct :
  total_leaves_in_forest = 10687500 := 
by sorry

end NUMINAMATH_GPT_forest_leaves_count_correct_l1457_145721


namespace NUMINAMATH_GPT_remaining_amount_is_correct_l1457_145756

-- Define the original price based on the deposit paid
def original_price : ℝ := 1500

-- Define the discount percentage
def discount_percentage : ℝ := 0.05

-- Define the sales tax percentage
def tax_percentage : ℝ := 0.075

-- Define the deposit already paid
def deposit_paid : ℝ := 150

-- Define the discounted price
def discounted_price : ℝ := original_price * (1 - discount_percentage)

-- Define the sales tax amount
def sales_tax : ℝ := discounted_price * tax_percentage

-- Define the final cost after adding sales tax
def final_cost : ℝ := discounted_price + sales_tax

-- Define the remaining amount to be paid
def remaining_amount : ℝ := final_cost - deposit_paid

-- The statement we need to prove
theorem remaining_amount_is_correct : remaining_amount = 1381.875 :=
by
  -- We'd normally write the proof here, but that's not required for this task.
  sorry

end NUMINAMATH_GPT_remaining_amount_is_correct_l1457_145756


namespace NUMINAMATH_GPT_total_squares_after_erasing_lines_l1457_145734

theorem total_squares_after_erasing_lines :
  ∀ (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ), a = 16 → b = 4 → c = 9 → d = 2 → 
  a - b + c - d + (a / 16) = 22 := 
by
  intro a b c d h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end NUMINAMATH_GPT_total_squares_after_erasing_lines_l1457_145734


namespace NUMINAMATH_GPT_negation_proof_l1457_145727

theorem negation_proof :
  (¬ ∃ x : ℝ, x ≤ 1 ∨ x^2 > 4) ↔ (∀ x : ℝ, x > 1 ∧ x^2 ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l1457_145727


namespace NUMINAMATH_GPT_value_of_expression_l1457_145759

theorem value_of_expression (a b c d x y : ℤ) 
  (h1 : a = -b) 
  (h2 : c * d = 1)
  (h3 : abs x = 3)
  (h4 : y = -1) : 
  2 * x - c * d + 6 * (a + b) - abs y = 4 ∨ 2 * x - c * d + 6 * (a + b) - abs y = -8 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l1457_145759


namespace NUMINAMATH_GPT_length_of_unfenced_side_l1457_145773

theorem length_of_unfenced_side :
  ∃ L W : ℝ, L * W = 320 ∧ 2 * W + L = 56 ∧ L = 40 :=
by
  sorry

end NUMINAMATH_GPT_length_of_unfenced_side_l1457_145773


namespace NUMINAMATH_GPT_interest_rate_first_part_l1457_145798

theorem interest_rate_first_part (A A1 A2 I : ℝ) (r : ℝ) :
  A = 3200 →
  A1 = 800 →
  A2 = A - A1 →
  I = 144 →
  (A1 * r / 100 + A2 * 5 / 100 = I) →
  r = 3 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_interest_rate_first_part_l1457_145798


namespace NUMINAMATH_GPT_time_to_cross_platform_l1457_145737

-- Definitions based on the given conditions
def train_length : ℝ := 300
def time_to_cross_pole : ℝ := 18
def platform_length : ℝ := 350

-- The question reformulated as a theorem in Lean 4
theorem time_to_cross_platform 
  (l_train : ℝ := train_length)
  (t_pole_cross : ℝ := time_to_cross_pole)
  (l_platform : ℝ := platform_length) :
  (l_train / t_pole_cross * (l_train + l_platform) = 39) :=
sorry

end NUMINAMATH_GPT_time_to_cross_platform_l1457_145737


namespace NUMINAMATH_GPT_milkman_pure_milk_l1457_145792

theorem milkman_pure_milk (x : ℝ) 
  (h_cost : 3.60 * x = 3 * (x + 5)) : x = 25 :=
  sorry

end NUMINAMATH_GPT_milkman_pure_milk_l1457_145792


namespace NUMINAMATH_GPT_chipmunk_families_went_away_l1457_145796

theorem chipmunk_families_went_away :
  ∀ (total_families left_families went_away_families : ℕ),
  total_families = 86 →
  left_families = 21 →
  went_away_families = total_families - left_families →
  went_away_families = 65 :=
by
  intros total_families left_families went_away_families ht hl hw
  rw [ht, hl] at hw
  exact hw

end NUMINAMATH_GPT_chipmunk_families_went_away_l1457_145796


namespace NUMINAMATH_GPT_sum_of_possible_values_of_k_l1457_145730

open Complex

theorem sum_of_possible_values_of_k (x y z k : ℂ) (hxyz : x ≠ y ∧ y ≠ z ∧ z ≠ x)
    (h : x / (1 - y + z) = k ∧ y / (1 - z + x) = k ∧ z / (1 - x + y) = k) : k = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_k_l1457_145730


namespace NUMINAMATH_GPT_find_minimum_a_l1457_145777

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x

theorem find_minimum_a (a : ℝ) :
  (∀ x, 1 ≤ x → 0 ≤ 3 * x^2 + a) ↔ a ≥ -3 :=
by
  sorry

end NUMINAMATH_GPT_find_minimum_a_l1457_145777


namespace NUMINAMATH_GPT_container_emptying_l1457_145764

theorem container_emptying (a b c : ℕ) : ∃ m n k : ℕ,
  (m = 0 ∨ n = 0 ∨ k = 0) ∧
  (∀ a' b' c', 
    (a' = a ∧ b' = b ∧ c' = c) ∨ 
    (a' + 2 * b' = a' ∧ b' = b ∧ c' + 2 * b' = c') ∨ 
    (a' + 2 * c' = a' ∧ b' + 2 * c' = b' ∧ c' = c') ∨ 
    (a + 2 * b' + c' = a' + 2 * m * (a + b') ∧ b' = n * (a + b') ∧ c' = k * (a + b')) 
  -> (a' = 0 ∨ b' = 0 ∨ c' = 0)) :=
sorry

end NUMINAMATH_GPT_container_emptying_l1457_145764


namespace NUMINAMATH_GPT_prime_pairs_square_l1457_145776

noncomputable def is_square (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

theorem prime_pairs_square (a b : ℤ) (ha : is_prime a) (hb : is_prime b) :
  is_square (3 * a^2 * b + 16 * a * b^2) ↔ (a = 19 ∧ b = 19) ∨ (a = 2 ∧ b = 3) :=
by
  sorry

end NUMINAMATH_GPT_prime_pairs_square_l1457_145776


namespace NUMINAMATH_GPT_decimal_fraction_eq_l1457_145704

theorem decimal_fraction_eq {b : ℕ} (hb : 0 < b) :
  (4 * b + 19 : ℚ) / (6 * b + 11) = 0.76 → b = 19 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_decimal_fraction_eq_l1457_145704


namespace NUMINAMATH_GPT_cricket_team_average_age_l1457_145762

open Real

-- Definitions based on the conditions given
def team_size := 11
def captain_age := 27
def wicket_keeper_age := 30
def remaining_players_size := team_size - 2

-- The mathematically equivalent proof problem in Lean statement
theorem cricket_team_average_age :
  ∃ A : ℝ,
    (A - 1) * remaining_players_size = (A * team_size) - (captain_age + wicket_keeper_age) ∧
    A = 24 :=
by
  sorry

end NUMINAMATH_GPT_cricket_team_average_age_l1457_145762


namespace NUMINAMATH_GPT_find_triangle_base_l1457_145750

theorem find_triangle_base (left_side : ℝ) (right_side : ℝ) (base : ℝ) 
  (h_left : left_side = 12) 
  (h_right : right_side = left_side + 2)
  (h_sum : left_side + right_side + base = 50) :
  base = 24 := 
sorry

end NUMINAMATH_GPT_find_triangle_base_l1457_145750


namespace NUMINAMATH_GPT_arrangement_count_l1457_145797

/-- April has five different basil plants and five different tomato plants. --/
def basil_plants : ℕ := 5
def tomato_plants : ℕ := 5

/-- All tomato plants must be placed next to each other. --/
def tomatoes_next_to_each_other := true

/-- The row must start with a basil plant. --/
def starts_with_basil := true

/-- The number of ways to arrange the plants in a row under the given conditions is 11520. --/
theorem arrangement_count :
  basil_plants = 5 ∧ tomato_plants = 5 ∧ tomatoes_next_to_each_other ∧ starts_with_basil → 
  ∃ arrangements : ℕ, arrangements = 11520 :=
by 
  sorry

end NUMINAMATH_GPT_arrangement_count_l1457_145797


namespace NUMINAMATH_GPT_rectangle_percentage_excess_l1457_145740

variable (L W : ℝ) -- The lengths of the sides of the rectangle
variable (x : ℝ) -- The percentage excess for the first side (what we want to prove)

theorem rectangle_percentage_excess 
    (h1 : W' = W * 0.95)                    -- Condition: second side is taken with 5% deficit
    (h2 : L' = L * (1 + x/100))             -- Condition: first side is taken with x% excess
    (h3 : A = L * W)                        -- Actual area of the rectangle
    (h4 : 1.064 = (L' * W') / A) :           -- Condition: error percentage in the area is 6.4%
    x = 12 :=                                -- Proof that x equals 12
sorry

end NUMINAMATH_GPT_rectangle_percentage_excess_l1457_145740


namespace NUMINAMATH_GPT_number_of_honey_bees_l1457_145714

theorem number_of_honey_bees (total_honey : ℕ) (honey_one_bee : ℕ) (days : ℕ) (h1 : total_honey = 30) (h2 : honey_one_bee = 1) (h3 : days = 30) : 
  (total_honey / honey_one_bee) = 30 :=
by
  -- Given total_honey = 30 grams in 30 days
  -- Given honey_one_bee = 1 gram in 30 days
  -- We need to prove (total_honey / honey_one_bee) = 30
  sorry

end NUMINAMATH_GPT_number_of_honey_bees_l1457_145714


namespace NUMINAMATH_GPT_travel_speed_is_four_l1457_145788
-- Import the required library

-- Define the conditions
def jacksSpeed (x : ℝ) : ℝ := x^2 - 13 * x - 26
def jillsDistance (x : ℝ) : ℝ := x^2 - 5 * x - 66
def jillsTime (x : ℝ) : ℝ := x + 8

-- Prove the equivalent statement
theorem travel_speed_is_four (x : ℝ) (h : x = 15) :
  jillsDistance x / jillsTime x = 4 ∧ jacksSpeed x = 4 := 
by sorry

end NUMINAMATH_GPT_travel_speed_is_four_l1457_145788


namespace NUMINAMATH_GPT_fedora_cleaning_time_l1457_145703

-- Definitions based on given conditions
def cleaning_time_per_section (total_time sections_cleaned : ℕ) : ℕ :=
  total_time / sections_cleaned

def remaining_sections (total_sections cleaned_sections : ℕ) : ℕ :=
  total_sections - cleaned_sections

def total_cleaning_time (remaining_sections time_per_section : ℕ) : ℕ :=
  remaining_sections * time_per_section

-- Theorem statement
theorem fedora_cleaning_time 
  (total_time : ℕ) 
  (sections_cleaned : ℕ)
  (additional_time : ℕ)
  (additional_sections : ℕ)
  (cleaned_sections : ℕ)
  (total_sections : ℕ)
  (h1 : total_time = 33)
  (h2 : sections_cleaned = 3)
  (h3 : additional_time = 165)
  (h4 : additional_sections = 15)
  (h5 : cleaned_sections = 3)
  (h6 : total_sections = 18)
  (h7 : cleaning_time_per_section total_time sections_cleaned = 11)
  (h8 : remaining_sections total_sections cleaned_sections = additional_sections)
  : total_cleaning_time additional_sections (cleaning_time_per_section total_time sections_cleaned) = additional_time := sorry

end NUMINAMATH_GPT_fedora_cleaning_time_l1457_145703


namespace NUMINAMATH_GPT_geometric_sequence_from_second_term_l1457_145724

theorem geometric_sequence_from_second_term (S : ℕ → ℕ) (a : ℕ → ℕ) :
  S 1 = 1 ∧ S 2 = 2 ∧ (∀ n, n ≥ 2 → S (n + 1) - 3 * S n + 2 * S (n - 1) = 0) →
  (∀ n, n ≥ 2 → a (n + 1) = 2 * a n) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_from_second_term_l1457_145724


namespace NUMINAMATH_GPT_inequality_proof_l1457_145700

variable {a b c : ℝ}

theorem inequality_proof (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: a^2 = b^2 + c^2) :
  a^3 + b^3 + c^3 ≥ (2*Real.sqrt 2 + 1) / 7 * (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1457_145700


namespace NUMINAMATH_GPT_express_1997_using_elevent_fours_l1457_145743

def number_expression_uses_eleven_fours : Prop :=
  (4 * 444 + 44 * 4 + 44 + 4 / 4 = 1997)
  
theorem express_1997_using_elevent_fours : number_expression_uses_eleven_fours :=
by
  sorry

end NUMINAMATH_GPT_express_1997_using_elevent_fours_l1457_145743


namespace NUMINAMATH_GPT_acute_triangle_tangent_sum_geq_3_sqrt_3_l1457_145790

theorem acute_triangle_tangent_sum_geq_3_sqrt_3 {α β γ : ℝ} (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) (h_sum : α + β + γ = π)
  (acute_α : α < π / 2) (acute_β : β < π / 2) (acute_γ : γ < π / 2) :
  Real.tan α + Real.tan β + Real.tan γ >= 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_acute_triangle_tangent_sum_geq_3_sqrt_3_l1457_145790


namespace NUMINAMATH_GPT_problem_l1457_145718

variable {a b c d : ℝ}

theorem problem (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (9 / (a - d)) :=
sorry

end NUMINAMATH_GPT_problem_l1457_145718


namespace NUMINAMATH_GPT_sum_of_digits_divisible_by_9_l1457_145757

theorem sum_of_digits_divisible_by_9 (D E : ℕ) (hD : D < 10) (hE : E < 10) : 
  (D + E + 37) % 9 = 0 → ((D + E = 8) ∨ (D + E = 17)) →
  (8 + 17 = 25) := 
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_sum_of_digits_divisible_by_9_l1457_145757


namespace NUMINAMATH_GPT_component_probability_l1457_145722

theorem component_probability (p : ℝ) 
  (h : (1 - p)^3 = 0.001) : 
  p = 0.9 :=
sorry

end NUMINAMATH_GPT_component_probability_l1457_145722


namespace NUMINAMATH_GPT_no_consecutive_primes_sum_65_l1457_145766

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p q : ℕ) : Prop := 
  is_prime p ∧ is_prime q ∧ (q = p + 2 ∨ q = p - 2)

theorem no_consecutive_primes_sum_65 : 
  ¬ ∃ p q : ℕ, consecutive_primes p q ∧ p + q = 65 :=
by 
  sorry

end NUMINAMATH_GPT_no_consecutive_primes_sum_65_l1457_145766


namespace NUMINAMATH_GPT_n_mult_n_plus_1_eq_square_l1457_145781

theorem n_mult_n_plus_1_eq_square (n : ℤ) : (∃ k : ℤ, n * (n + 1) = k^2) ↔ (n = 0 ∨ n = -1) := 
by sorry

end NUMINAMATH_GPT_n_mult_n_plus_1_eq_square_l1457_145781


namespace NUMINAMATH_GPT_intersection_A_B_l1457_145728

def A : Set ℤ := { -2, -1, 0, 1, 2 }
def B : Set ℤ := { x : ℤ | x < 1 }

theorem intersection_A_B : A ∩ B = { -2, -1, 0 } :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l1457_145728


namespace NUMINAMATH_GPT_time_to_upload_file_l1457_145709

-- Define the conditions
def file_size : ℕ := 160
def upload_speed : ℕ := 8

-- Define the question as a proof goal
theorem time_to_upload_file :
  file_size / upload_speed = 20 := 
sorry

end NUMINAMATH_GPT_time_to_upload_file_l1457_145709


namespace NUMINAMATH_GPT_smallest_prime_factor_of_setC_l1457_145754

def setC : Set ℕ := {51, 53, 54, 56, 57}

def prime_factors (n : ℕ) : Set ℕ :=
  { p | p.Prime ∧ p ∣ n }

theorem smallest_prime_factor_of_setC :
  (∃ n ∈ setC, ∀ m ∈ setC, ∀ p ∈ prime_factors n, ∀ q ∈ prime_factors m, p ≤ q) ∧
  (∃ m ∈ setC, ∀ p ∈ prime_factors 54, ∀ q ∈ prime_factors m, p = q) := 
sorry

end NUMINAMATH_GPT_smallest_prime_factor_of_setC_l1457_145754


namespace NUMINAMATH_GPT_neg_p_l1457_145761

variable {α : Type}
variable (x : α)

def p (x : Real) : Prop := ∀ x : Real, x > 1 → x^2 - 1 > 0

theorem neg_p : ¬( ∀ x : Real, x > 1 → x^2 - 1 > 0) ↔ ∃ x : Real, x > 1 ∧ x^2 - 1 ≤ 0 := 
by 
  sorry

end NUMINAMATH_GPT_neg_p_l1457_145761


namespace NUMINAMATH_GPT_monotonic_solution_l1457_145768

-- Definition of a monotonic function
def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- The main theorem
theorem monotonic_solution (f : ℝ → ℝ) 
  (mon : monotonic f) 
  (h : ∀ x y : ℝ, f (f x - y) + f (x + y) = 0) : 
  (∀ x, f x = 0) ∨ (∀ x, f x = -x) :=
sorry

end NUMINAMATH_GPT_monotonic_solution_l1457_145768


namespace NUMINAMATH_GPT_middle_number_in_8th_row_l1457_145701

-- Define a function that describes the number on the far right of the nth row.
def far_right_number (n : ℕ) : ℕ := n^2

-- Define a function that calculates the number of elements in the nth row.
def row_length (n : ℕ) : ℕ := 2 * n - 1

-- Define the middle number in the nth row.
def middle_number (n : ℕ) : ℕ := 
  let mid_index := (row_length n + 1) / 2
  far_right_number (n - 1) + mid_index

-- Statement to prove the middle number in the 8th row is 57
theorem middle_number_in_8th_row : middle_number 8 = 57 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_middle_number_in_8th_row_l1457_145701


namespace NUMINAMATH_GPT_xyz_inequality_l1457_145774

theorem xyz_inequality (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  3 * (x^2 * y^2 + x^2 * z^2 + y^2 * z^2) - 2 * x * y * z * (x + y + z) ≤ 3 := by
  sorry

end NUMINAMATH_GPT_xyz_inequality_l1457_145774


namespace NUMINAMATH_GPT_green_toads_per_acre_l1457_145799

theorem green_toads_per_acre (brown_toads spotted_brown_toads green_toads : ℕ) 
  (h1 : ∀ g, 25 * g = brown_toads) 
  (h2 : spotted_brown_toads = brown_toads / 4) 
  (h3 : spotted_brown_toads = 50) : 
  green_toads = 8 :=
by
  sorry

end NUMINAMATH_GPT_green_toads_per_acre_l1457_145799


namespace NUMINAMATH_GPT_hcf_of_two_numbers_l1457_145758

noncomputable def number1 : ℕ := 414

noncomputable def lcm_factors : Set ℕ := {13, 18}

noncomputable def hcf (a b : ℕ) : ℕ := Nat.gcd a b

-- Statement to prove
theorem hcf_of_two_numbers (Y : ℕ) 
  (H : ℕ) 
  (lcm : ℕ) 
  (H_lcm_factors : lcm = H * 13 * 18)
  (H_lcm_prop : lcm = (number1 * Y) / H)
  (H_Y : Y = (H^2 * 13 * 18) / 414)
  : H = 23 := 
sorry

end NUMINAMATH_GPT_hcf_of_two_numbers_l1457_145758


namespace NUMINAMATH_GPT_cookie_difference_l1457_145748

def AlyssaCookies : ℕ := 129
def AiyannaCookies : ℕ := 140
def Difference : ℕ := 11

theorem cookie_difference : AiyannaCookies - AlyssaCookies = Difference := by
  sorry

end NUMINAMATH_GPT_cookie_difference_l1457_145748


namespace NUMINAMATH_GPT_number_of_integer_solutions_l1457_145706

theorem number_of_integer_solutions
    (a : ℤ)
    (x : ℤ)
    (h1 : ∃ x : ℤ, (1 - a) / (x - 2) + 2 = 1 / (2 - x))
    (h2 : ∀ x : ℤ, 4 * x ≥ 3 * (x - 1) ∧ x + (2 * x - 1) / 2 < (a - 1) / 2) :
    (a = 4) :=
sorry

end NUMINAMATH_GPT_number_of_integer_solutions_l1457_145706


namespace NUMINAMATH_GPT_min_a2_b2_l1457_145779

noncomputable def minimum_a2_b2 (a b : ℝ) : Prop :=
  (∃ a b : ℝ, (|(-2*a - 2*b + 4)|) / (Real.sqrt (a^2 + (2*b)^2)) = 2) → (a^2 + b^2 = 2)

theorem min_a2_b2 : minimum_a2_b2 a b :=
by
  sorry

end NUMINAMATH_GPT_min_a2_b2_l1457_145779


namespace NUMINAMATH_GPT_basketball_teams_l1457_145755

theorem basketball_teams (boys girls : ℕ) (total_players : ℕ) (team_size : ℕ) (ways : ℕ) :
  boys = 7 → girls = 3 → total_players = 10 → team_size = 5 → ways = 105 → 
  ∃ (girls_in_team1 girls_in_team2 : ℕ), 
    girls_in_team1 + girls_in_team2 = 3 ∧ 
    1 ≤ girls_in_team1 ∧ 
    1 ≤ girls_in_team2 ∧ 
    girls_in_team1 ≠ 0 ∧ 
    girls_in_team2 ≠ 0 ∧ 
    ways = 105 :=
by 
  sorry

end NUMINAMATH_GPT_basketball_teams_l1457_145755


namespace NUMINAMATH_GPT_find_number_l1457_145716

theorem find_number (x : ℤ) (h : x = 1) : x + 1 = 2 :=
  by
  sorry

end NUMINAMATH_GPT_find_number_l1457_145716


namespace NUMINAMATH_GPT_parabola_point_l1457_145733

theorem parabola_point (a b c : ℝ) (hA : 0.64 * a - 0.8 * b + c = 4.132)
  (hB : 1.44 * a + 1.2 * b + c = -1.948) (hC : 7.84 * a + 2.8 * b + c = -3.932) :
  0.5 * (1.8)^2 - 3.24 * 1.8 + 1.22 = -2.992 :=
by
  -- Proof is intentionally omitted
  sorry

end NUMINAMATH_GPT_parabola_point_l1457_145733


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1457_145772

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ)
  (h_seq : arithmetic_sequence a)
  (h1 : a 2 + a 3 = 4)
  (h2 : a 4 + a 5 = 6) :
  a 9 + a 10 = 11 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1457_145772


namespace NUMINAMATH_GPT_cones_to_cylinder_volume_ratio_l1457_145778

theorem cones_to_cylinder_volume_ratio :
  let π := Real.pi
  let r_cylinder := 4
  let h_cylinder := 18
  let r_cone := 4
  let h_cone1 := 6
  let h_cone2 := 9
  let V_cylinder := π * r_cylinder^2 * h_cylinder
  let V_cone1 := (1 / 3) * π * r_cone^2 * h_cone1
  let V_cone2 := (1 / 3) * π * r_cone^2 * h_cone2
  let V_totalCones := V_cone1 + V_cone2
  V_totalCones / V_cylinder = 5 / 18 :=
by
  sorry

end NUMINAMATH_GPT_cones_to_cylinder_volume_ratio_l1457_145778


namespace NUMINAMATH_GPT_right_triangle_third_side_square_l1457_145723

theorem right_triangle_third_side_square (a b : ℕ) (c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2) :
  c^2 = 28 ∨ c^2 = 100 :=
by { sorry }

end NUMINAMATH_GPT_right_triangle_third_side_square_l1457_145723


namespace NUMINAMATH_GPT_maximum_black_squares_l1457_145744

theorem maximum_black_squares (n : ℕ) (h : n ≥ 2) : 
  (n % 2 = 0 → ∃ b : ℕ, b = (n^2 - 4) / 2) ∧ 
  (n % 2 = 1 → ∃ b : ℕ, b = (n^2 - 1) / 2) := 
by sorry

end NUMINAMATH_GPT_maximum_black_squares_l1457_145744


namespace NUMINAMATH_GPT_remainder_9_5_4_6_5_7_mod_7_l1457_145711

theorem remainder_9_5_4_6_5_7_mod_7 :
  ((9^5 + 4^6 + 5^7) % 7) = 2 :=
by sorry

end NUMINAMATH_GPT_remainder_9_5_4_6_5_7_mod_7_l1457_145711


namespace NUMINAMATH_GPT_rick_ironed_27_pieces_l1457_145767

def pieces_of_clothing_ironed (dress_shirts_per_hour : ℕ) (hours_ironing_shirts : ℕ) 
                              (dress_pants_per_hour : ℕ) (hours_ironing_pants : ℕ) : ℕ :=
  dress_shirts_per_hour * hours_ironing_shirts + dress_pants_per_hour * hours_ironing_pants

theorem rick_ironed_27_pieces :
  pieces_of_clothing_ironed 4 3 3 5 = 27 :=
by sorry

end NUMINAMATH_GPT_rick_ironed_27_pieces_l1457_145767


namespace NUMINAMATH_GPT_negation_of_inequality_l1457_145710

theorem negation_of_inequality :
  ¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 := 
sorry

end NUMINAMATH_GPT_negation_of_inequality_l1457_145710


namespace NUMINAMATH_GPT_problem_solution_l1457_145791

theorem problem_solution :
  3 ^ (0 ^ (2 ^ 2)) + ((3 ^ 1) ^ 0) ^ 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1457_145791


namespace NUMINAMATH_GPT_parrot_seeds_consumed_l1457_145765

theorem parrot_seeds_consumed (H1 : ∃ T : ℝ, 0.40 * T = 8) : 
  (∃ T : ℝ, 0.40 * T = 8 ∧ 2 * T = 40) :=
sorry

end NUMINAMATH_GPT_parrot_seeds_consumed_l1457_145765


namespace NUMINAMATH_GPT_elmer_saving_percent_l1457_145775

theorem elmer_saving_percent (x c : ℝ) (hx : x > 0) (hc : c > 0) :
  let old_car_fuel_efficiency := x
  let new_car_fuel_efficiency := 1.6 * x
  let gasoline_cost := c
  let diesel_cost := 1.25 * c
  let trip_distance := 300
  let old_car_fuel_needed := trip_distance / old_car_fuel_efficiency
  let new_car_fuel_needed := trip_distance / new_car_fuel_efficiency
  let old_car_cost := old_car_fuel_needed * gasoline_cost
  let new_car_cost := new_car_fuel_needed * diesel_cost
  let cost_saving := old_car_cost - new_car_cost
  let percent_saving := (cost_saving / old_car_cost) * 100
  percent_saving = 21.875 :=
by
  sorry

end NUMINAMATH_GPT_elmer_saving_percent_l1457_145775


namespace NUMINAMATH_GPT_ratio_platform_to_pole_l1457_145719

variables (l t T v : ℝ)
-- Conditions
axiom constant_velocity : ∀ t l, l = v * t
axiom pass_pole : l = v * t
axiom pass_platform : 6 * l = v * T 

theorem ratio_platform_to_pole (h1 : l = v * t) (h2 : 6 * l = v * T) : T / t = 6 := 
  by sorry

end NUMINAMATH_GPT_ratio_platform_to_pole_l1457_145719


namespace NUMINAMATH_GPT_bus_seats_capacity_l1457_145783

-- Define the conditions
variable (x : ℕ) -- number of people each seat can hold
def left_side_seats := 15
def right_side_seats := left_side_seats - 3
def back_seat_capacity := 7
def total_capacity := left_side_seats * x + right_side_seats * x + back_seat_capacity

-- State the theorem
theorem bus_seats_capacity :
  total_capacity x = 88 → x = 3 := by
  sorry

end NUMINAMATH_GPT_bus_seats_capacity_l1457_145783


namespace NUMINAMATH_GPT_movies_left_to_watch_l1457_145736

theorem movies_left_to_watch (total_movies : ℕ) (movies_watched : ℕ) : total_movies = 17 ∧ movies_watched = 7 → (total_movies - movies_watched) = 10 :=
by
  sorry

end NUMINAMATH_GPT_movies_left_to_watch_l1457_145736


namespace NUMINAMATH_GPT_attendance_proof_l1457_145787

noncomputable def next_year_attendance (this_year: ℕ) := 2 * this_year
noncomputable def last_year_attendance (next_year: ℕ) := next_year - 200
noncomputable def total_attendance (last_year this_year next_year: ℕ) := last_year + this_year + next_year

theorem attendance_proof (this_year: ℕ) (h1: this_year = 600):
    total_attendance (last_year_attendance (next_year_attendance this_year)) this_year (next_year_attendance this_year) = 2800 :=
by
  sorry

end NUMINAMATH_GPT_attendance_proof_l1457_145787


namespace NUMINAMATH_GPT_regression_lines_intersect_at_average_l1457_145705

theorem regression_lines_intersect_at_average
  {x_vals1 x_vals2 : List ℝ} {y_vals1 y_vals2 : List ℝ}
  (n1 : x_vals1.length = 100) (n2 : x_vals2.length = 150)
  (mean_x1 : (List.sum x_vals1 / 100) = s) (mean_x2 : (List.sum x_vals2 / 150) = s)
  (mean_y1 : (List.sum y_vals1 / 100) = t) (mean_y2 : (List.sum y_vals2 / 150) = t)
  (regression_line1 : ℝ → ℝ)
  (regression_line2 : ℝ → ℝ)
  (on_line1 : ∀ x, regression_line1 x = (a1 * x + b1))
  (on_line2 : ∀ x, regression_line2 x = (a2 * x + b2))
  (sample_center1 : regression_line1 s = t)
  (sample_center2 : regression_line2 s = t) :
  regression_line1 s = regression_line2 s := sorry

end NUMINAMATH_GPT_regression_lines_intersect_at_average_l1457_145705


namespace NUMINAMATH_GPT_smith_boxes_l1457_145784

theorem smith_boxes (initial_markers : ℕ) (markers_per_box : ℕ) (total_markers : ℕ)
  (h1 : initial_markers = 32) (h2 : markers_per_box = 9) (h3 : total_markers = 86) :
  total_markers - initial_markers = 6 * markers_per_box :=
by
  -- We state our assumptions explicitly for better clarity
  have h_total : total_markers = 86 := h3
  have h_initial : initial_markers = 32 := h1
  have h_box : markers_per_box = 9 := h2
  sorry

end NUMINAMATH_GPT_smith_boxes_l1457_145784


namespace NUMINAMATH_GPT_symmetric_point_l1457_145707

theorem symmetric_point (P : ℝ × ℝ) (a b : ℝ) (h1 : P = (2, 7)) (h2 : 1 * (a - 2) + (b - 7) * (-1) = 0) (h3 : (a + 2) / 2 + (b + 7) / 2 + 1 = 0) :
  (a, b) = (-8, -3) :=
sorry

end NUMINAMATH_GPT_symmetric_point_l1457_145707


namespace NUMINAMATH_GPT_triangle_angle_sum_l1457_145715

theorem triangle_angle_sum (y : ℝ) (h : 40 + 3 * y + (y + 10) = 180) : y = 32.5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l1457_145715


namespace NUMINAMATH_GPT_magic_trick_constant_l1457_145712

theorem magic_trick_constant (a : ℚ) : ((2 * a + 8) / 4 - a / 2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_magic_trick_constant_l1457_145712


namespace NUMINAMATH_GPT_partition_count_l1457_145789

theorem partition_count (A B : Finset ℕ) :
  (∀ n, n ∈ A ∨ n ∈ B) ∧ 
  (∀ n, n ∈ A → 1 ≤ n ∧ n ≤ 9) ∧ 
  (∀ n, n ∈ B → 1 ≤ n ∧ n ≤ 9) ∧ 
  (A ∩ B = ∅) ∧ 
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
  (8 * A.sum id = B.sum id) ∧ 
  (A.sum id + B.sum id = 45) → 
  ∃! (num_ways : ℕ), num_ways = 3 :=
sorry

end NUMINAMATH_GPT_partition_count_l1457_145789


namespace NUMINAMATH_GPT_total_fish_in_lake_l1457_145760

-- Given conditions:
def initiallyTaggedFish : ℕ := 100
def capturedFish : ℕ := 100
def taggedFishInAugust : ℕ := 5
def taggedFishMortalityRate : ℝ := 0.3
def newcomerFishRate : ℝ := 0.2

-- Proof to show that the total number of fish at the beginning of April is 1120
theorem total_fish_in_lake (initiallyTaggedFish capturedFish taggedFishInAugust : ℕ) 
  (taggedFishMortalityRate newcomerFishRate : ℝ) : 
  (taggedFishInAugust : ℝ) / (capturedFish * (1 - newcomerFishRate)) = 
  ((initiallyTaggedFish * (1 - taggedFishMortalityRate)) : ℝ) / (1120 : ℝ) :=
by 
  sorry

end NUMINAMATH_GPT_total_fish_in_lake_l1457_145760


namespace NUMINAMATH_GPT_ethan_presents_l1457_145725

variable (A E : ℝ)

theorem ethan_presents (h1 : A = 9) (h2 : A = E - 22.0) : E = 31 := 
by
  sorry

end NUMINAMATH_GPT_ethan_presents_l1457_145725


namespace NUMINAMATH_GPT_min_value_expression_l1457_145720

theorem min_value_expression (x1 x2 x3 x4 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h_sum : x1 + x2 + x3 + x4 = Real.pi) :
  (2 * (Real.sin x1) ^ 2 + 1 / (Real.sin x1) ^ 2) *
  (2 * (Real.sin x2) ^ 2 + 1 / (Real.sin x2) ^ 2) *
  (2 * (Real.sin x3) ^ 2 + 1 / (Real.sin x3) ^ 2) *
  (2 * (Real.sin x4) ^ 2 + 1 / (Real.sin x4) ^ 2) = 81 := 
sorry

end NUMINAMATH_GPT_min_value_expression_l1457_145720


namespace NUMINAMATH_GPT_find_m_l1457_145751

-- Mathematical definitions from the given conditions
def condition1 (m : ℝ) : Prop := m^2 - 2 * m - 2 = 1
def condition2 (m : ℝ) : Prop := m + 1/2 * m^2 > 0

-- The proof problem summary
theorem find_m (m : ℝ) (h1 : condition1 m) (h2 : condition2 m) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1457_145751


namespace NUMINAMATH_GPT_radius_increase_l1457_145794

/-- Proving that the radius increases by 7/π inches when the circumference increases from 50 inches to 64 inches -/
theorem radius_increase (C₁ C₂ : ℝ) (h₁ : C₁ = 50) (h₂ : C₂ = 64) :
  (C₂ / (2 * Real.pi) - C₁ / (2 * Real.pi)) = 7 / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_radius_increase_l1457_145794


namespace NUMINAMATH_GPT_inequality_proof_l1457_145782

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (2 * a^2) / (1 + a + a * b)^2 + (2 * b^2) / (1 + b + b * c)^2 + (2 * c^2) / (1 + c + c * a)^2 +
  9 / ((1 + a + a * b) * (1 + b + b * c) * (1 + c + c * a)) ≥ 1 :=
by {
  sorry -- The proof goes here
}

end NUMINAMATH_GPT_inequality_proof_l1457_145782


namespace NUMINAMATH_GPT_total_strawberries_l1457_145763

-- Define the number of original strawberries and the number of picked strawberries
def original_strawberries : ℕ := 42
def picked_strawberries : ℕ := 78

-- Prove the total number of strawberries
theorem total_strawberries : original_strawberries + picked_strawberries = 120 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_strawberries_l1457_145763


namespace NUMINAMATH_GPT_sum_of_inverses_gt_one_l1457_145739

variable (a1 a2 a3 S : ℝ)

theorem sum_of_inverses_gt_one
  (h1 : a1 > 1)
  (h2 : a2 > 1)
  (h3 : a3 > 1)
  (h_sum : a1 + a2 + a3 = S)
  (ineq1 : a1^2 / (a1 - 1) > S)
  (ineq2 : a2^2 / (a2 - 1) > S)
  (ineq3 : a3^2 / (a3 - 1) > S) :
  1 / (a1 + a2) + 1 / (a2 + a3) + 1 / (a3 + a1) > 1 := by
  sorry

end NUMINAMATH_GPT_sum_of_inverses_gt_one_l1457_145739


namespace NUMINAMATH_GPT_no_such_function_exists_l1457_145726

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, (f 0 > 0) ∧ (∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) :=
by
  -- proof to be completed
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l1457_145726


namespace NUMINAMATH_GPT_total_number_of_plugs_l1457_145717

variables (pairs_mittens pairs_plugs : ℕ)

-- Conditions
def initial_pairs_mittens : ℕ := 150
def initial_pairs_plugs : ℕ := initial_pairs_mittens + 20
def added_pairs_plugs : ℕ := 30
def total_pairs_plugs : ℕ := initial_pairs_plugs + added_pairs_plugs

-- The proposition we're going to prove:
theorem total_number_of_plugs : initial_pairs_mittens = 150 ∧ initial_pairs_plugs = initial_pairs_mittens + 20 ∧ added_pairs_plugs = 30 → 
  total_pairs_plugs * 2 = 400 := sorry

end NUMINAMATH_GPT_total_number_of_plugs_l1457_145717


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1457_145702

theorem solution_set_of_inequality (a t : ℝ) (h1 : ∀ x : ℝ, x^2 - 2 * a * x + a > 0) : 
  a > 0 ∧ a < 1 → (a^(2*t + 1) < a^(t^2 + 2*t - 3) ↔ -2 < t ∧ t < 2) :=
by
  intro ha
  have h : (0 < a ∧ a < 1) := sorry
  exact sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1457_145702


namespace NUMINAMATH_GPT_david_marks_in_biology_l1457_145747

theorem david_marks_in_biology (marks_english marks_math marks_physics marks_chemistry : ℕ)
  (average_marks num_subjects total_marks_known : ℕ)
  (h1 : marks_english = 76)
  (h2 : marks_math = 65)
  (h3 : marks_physics = 82)
  (h4 : marks_chemistry = 67)
  (h5 : average_marks = 75)
  (h6 : num_subjects = 5)
  (h7 : total_marks_known = marks_english + marks_math + marks_physics + marks_chemistry)
  (h8 : total_marks_known = 290)
  : ∃ biology_marks : ℕ, biology_marks = 85 ∧ biology_marks = (average_marks * num_subjects) - total_marks_known :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_david_marks_in_biology_l1457_145747


namespace NUMINAMATH_GPT_toothpicks_at_20th_stage_l1457_145785

def toothpicks_in_stage (n : ℕ) : ℕ :=
  4 + 3 * (n - 1)

theorem toothpicks_at_20th_stage : toothpicks_in_stage 20 = 61 :=
by 
  sorry

end NUMINAMATH_GPT_toothpicks_at_20th_stage_l1457_145785


namespace NUMINAMATH_GPT_packets_of_chips_l1457_145708

theorem packets_of_chips (x : ℕ) 
  (h1 : ∀ x, 2 * (x : ℝ) + 1.5 * (10 : ℝ) = 45) : 
  x = 15 := 
by 
  sorry

end NUMINAMATH_GPT_packets_of_chips_l1457_145708


namespace NUMINAMATH_GPT_min_expression_value_l1457_145752

theorem min_expression_value (a b c : ℝ) (ha : 1 ≤ a) (hbc : b ≥ a) (hcb : c ≥ b) (hc5 : c ≤ 5) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (5 / c - 1)^2 ≥ 2 * Real.sqrt 5 - 4 * Real.sqrt (5^(1/4)) + 4 :=
sorry

end NUMINAMATH_GPT_min_expression_value_l1457_145752


namespace NUMINAMATH_GPT_train_speed_kmph_l1457_145746

noncomputable def train_speed_mps : ℝ := 60.0048

def conversion_factor : ℝ := 3.6

theorem train_speed_kmph : train_speed_mps * conversion_factor = 216.01728 := by
  sorry

end NUMINAMATH_GPT_train_speed_kmph_l1457_145746


namespace NUMINAMATH_GPT_lipstick_cost_correct_l1457_145732

noncomputable def cost_of_lipsticks (total_cost: ℕ) (cost_slippers: ℚ) (cost_hair_color: ℚ) (paid: ℚ) (number_lipsticks: ℕ) : ℚ :=
  (paid - (6 * cost_slippers + 8 * cost_hair_color)) / number_lipsticks

theorem lipstick_cost_correct :
  cost_of_lipsticks 6 (2.5:ℚ) (3:ℚ) (44:ℚ) 4 = 1.25 := by
  sorry

end NUMINAMATH_GPT_lipstick_cost_correct_l1457_145732


namespace NUMINAMATH_GPT_each_monkey_gets_bananas_l1457_145786

-- Define the conditions
def total_monkeys : ℕ := 12
def total_piles : ℕ := 10
def first_piles : ℕ := 6
def first_pile_hands : ℕ := 9
def first_hand_bananas : ℕ := 14
def remaining_piles : ℕ := total_piles - first_piles
def remaining_pile_hands : ℕ := 12
def remaining_hand_bananas : ℕ := 9

-- Define the number of bananas in each type of pile
def bananas_in_first_piles : ℕ := first_piles * first_pile_hands * first_hand_bananas
def bananas_in_remaining_piles : ℕ := remaining_piles * remaining_pile_hands * remaining_hand_bananas
def total_bananas : ℕ := bananas_in_first_piles + bananas_in_remaining_piles

-- Define the main theorem to be proved
theorem each_monkey_gets_bananas : total_bananas / total_monkeys = 99 := by
  sorry

end NUMINAMATH_GPT_each_monkey_gets_bananas_l1457_145786


namespace NUMINAMATH_GPT_stella_dolls_count_l1457_145731

variables (D : ℕ) (clocks glasses P_doll P_clock P_glass cost profit : ℕ)

theorem stella_dolls_count (h_clocks : clocks = 2)
                     (h_glasses : glasses = 5)
                     (h_P_doll : P_doll = 5)
                     (h_P_clock : P_clock = 15)
                     (h_P_glass : P_glass = 4)
                     (h_cost : cost = 40)
                     (h_profit : profit = 25) :
  D = 3 :=
by sorry

end NUMINAMATH_GPT_stella_dolls_count_l1457_145731


namespace NUMINAMATH_GPT_find_solution_set_l1457_145738

noncomputable def is_solution (x : ℝ) : Prop :=
(1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < 1 / 4

theorem find_solution_set :
  { x : ℝ | is_solution x } = { x : ℝ | x < -2 } ∪ { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end NUMINAMATH_GPT_find_solution_set_l1457_145738


namespace NUMINAMATH_GPT_guppies_to_angelfish_ratio_l1457_145729

noncomputable def goldfish : ℕ := 8
noncomputable def angelfish : ℕ := goldfish + 4
noncomputable def total_fish : ℕ := 44
noncomputable def guppies : ℕ := total_fish - (goldfish + angelfish)

theorem guppies_to_angelfish_ratio :
    guppies / angelfish = 2 := by
    sorry

end NUMINAMATH_GPT_guppies_to_angelfish_ratio_l1457_145729


namespace NUMINAMATH_GPT_chocolate_milk_container_size_l1457_145735

/-- Holly's chocolate milk consumption conditions and container size -/
theorem chocolate_milk_container_size
  (morning_initial: ℝ)  -- Initial amount in the morning
  (morning_drink: ℝ)    -- Amount drank in the morning with breakfast
  (lunch_drink: ℝ)      -- Amount drank at lunch
  (dinner_drink: ℝ)     -- Amount drank with dinner
  (end_of_day: ℝ)       -- Amount she ends the day with
  (lunch_container_size: ℝ) -- Size of the container bought at lunch
  (C: ℝ)                -- Container size she bought at lunch
  (h_initial: morning_initial = 16)
  (h_morning_drink: morning_drink = 8)
  (h_lunch_drink: lunch_drink = 8)
  (h_dinner_drink: dinner_drink = 8)
  (h_end_of_day: end_of_day = 56) :
  (morning_initial - morning_drink) + C - lunch_drink - dinner_drink = end_of_day → 
  lunch_container_size = 64 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_milk_container_size_l1457_145735


namespace NUMINAMATH_GPT_xy_inequality_l1457_145770

theorem xy_inequality (x y : ℝ) (h: x^8 + y^8 ≤ 2) : 
  x^2 * y^2 + |x^2 - y^2| ≤ π / 2 :=
sorry

end NUMINAMATH_GPT_xy_inequality_l1457_145770


namespace NUMINAMATH_GPT_inequality_unequal_positive_numbers_l1457_145713

theorem inequality_unequal_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  (a + b) / 2 > (2 * a * b) / (a + b) :=
by
sorry

end NUMINAMATH_GPT_inequality_unequal_positive_numbers_l1457_145713


namespace NUMINAMATH_GPT_solve_inequality_system_l1457_145771

theorem solve_inequality_system (x : ℝ) :
  (x / 3 + 2 > 0) ∧ (2 * x + 5 ≥ 3) ↔ (x ≥ -1) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l1457_145771


namespace NUMINAMATH_GPT_alice_meets_bob_at_25_km_l1457_145769

-- Define variables for times, speeds, and distances
variables (t : ℕ) (d : ℕ)

-- Conditions
def distance_between_homes := 41
def alice_speed := 5
def bob_speed := 4
def alice_start_time := 1

-- Relating the distances covered by Alice and Bob when they meet
def alice_walk_distance := alice_speed * (t + alice_start_time)
def bob_walk_distance := bob_speed * t
def total_walk_distance := alice_walk_distance + bob_walk_distance

-- Alexander walks 25 kilometers before meeting Bob
theorem alice_meets_bob_at_25_km :
  total_walk_distance = distance_between_homes → alice_walk_distance = 25 :=
by
  sorry

end NUMINAMATH_GPT_alice_meets_bob_at_25_km_l1457_145769


namespace NUMINAMATH_GPT_restaurant_sales_l1457_145795

theorem restaurant_sales :
  let meals_sold_8 := 10
  let price_per_meal_8 := 8
  let meals_sold_10 := 5
  let price_per_meal_10 := 10
  let meals_sold_4 := 20
  let price_per_meal_4 := 4
  let total_sales := meals_sold_8 * price_per_meal_8 + meals_sold_10 * price_per_meal_10 + meals_sold_4 * price_per_meal_4
  total_sales = 210 :=
by
  sorry

end NUMINAMATH_GPT_restaurant_sales_l1457_145795


namespace NUMINAMATH_GPT_credibility_of_relationship_l1457_145780

theorem credibility_of_relationship
  (sample_size : ℕ)
  (chi_squared_value : ℝ)
  (table : ℕ → ℝ × ℝ)
  (h_sample : sample_size = 5000)
  (h_chi_squared : chi_squared_value = 6.109)
  (h_table : table 5 = (5.024, 0.025) ∧ table 6 = (6.635, 0.010)) :
  credible_percent = 97.5 :=
by
  sorry

end NUMINAMATH_GPT_credibility_of_relationship_l1457_145780


namespace NUMINAMATH_GPT_prove_zero_l1457_145741

variable {a b c : ℝ}

theorem prove_zero (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c) ^ 2 + b / (c - a) ^ 2 + c / (a - b) ^ 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_prove_zero_l1457_145741


namespace NUMINAMATH_GPT_lindsey_exercise_bands_l1457_145745

theorem lindsey_exercise_bands (x : ℕ) 
  (h1 : ∀ n, n = 5 * x) 
  (h2 : ∀ m, m = 10 * x) 
  (h3 : ∀ d, d = m + 10) 
  (h4 : d = 30) : 
  x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_lindsey_exercise_bands_l1457_145745


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l1457_145749

theorem sum_arithmetic_sequence :
  let a : ℤ := -25
  let d : ℤ := 4
  let a_n : ℤ := 19
  let n : ℤ := (a_n - a) / d + 1
  let S : ℤ := n * (a + a_n) / 2
  S = -36 :=
by 
  let a := -25
  let d := 4
  let a_n := 19
  let n := (a_n - a) / d + 1
  let S := n * (a + a_n) / 2
  show S = -36
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_l1457_145749
