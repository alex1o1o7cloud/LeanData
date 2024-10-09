import Mathlib

namespace count_multiples_of_15_l336_33602

theorem count_multiples_of_15 : ∃ n : ℕ, ∀ k, 12 < k ∧ k < 202 ∧ k % 15 = 0 ↔ k = 15 * n ∧ n = 13 := sorry

end count_multiples_of_15_l336_33602


namespace optionD_is_quad_eq_in_one_var_l336_33623

/-- Define a predicate for being a quadratic equation in one variable --/
def is_quad_eq_in_one_var (eq : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ (a b c : ℕ), a ≠ 0 ∧ ∀ x : ℕ, eq a b c

/-- Options as given predicates --/
def optionA (a b c : ℕ) : Prop := 3 * a^2 - 6 * b + 2 = 0
def optionB (a b c : ℕ) : Prop := a * a^2 - b * a + c = 0
def optionC (a b c : ℕ) : Prop := (1 / a^2) + b = c
def optionD (a b c : ℕ) : Prop := a^2 = 0

/-- Prove that Option D is a quadratic equation in one variable --/
theorem optionD_is_quad_eq_in_one_var : is_quad_eq_in_one_var optionD :=
sorry

end optionD_is_quad_eq_in_one_var_l336_33623


namespace abs_eq_sum_condition_l336_33631

theorem abs_eq_sum_condition (x y : ℝ) (h : |x - y^2| = x + y^2) : x = 0 ∧ y = 0 :=
  sorry

end abs_eq_sum_condition_l336_33631


namespace renu_suma_work_together_l336_33694

-- Define the time it takes for Renu to do the work by herself
def renu_days : ℕ := 6

-- Define the time it takes for Suma to do the work by herself
def suma_days : ℕ := 12

-- Define the work rate for Renu
def renu_work_rate : ℚ := 1 / renu_days

-- Define the work rate for Suma
def suma_work_rate : ℚ := 1 / suma_days

-- Define the combined work rate
def combined_work_rate : ℚ := renu_work_rate + suma_work_rate

-- Define the days it takes for both Renu and Suma to complete the work together
def days_to_complete_together : ℚ := 1 / combined_work_rate

-- The theorem stating that Renu and Suma can complete the work together in 4 days
theorem renu_suma_work_together : days_to_complete_together = 4 :=
by
  have h1 : renu_days = 6 := rfl
  have h2 : suma_days = 12 := rfl
  have h3 : renu_work_rate = 1 / 6 := by simp [renu_work_rate, h1]
  have h4 : suma_work_rate = 1 / 12 := by simp [suma_work_rate, h2]
  have h5 : combined_work_rate = 1 / 6 + 1 / 12 := by simp [combined_work_rate, h3, h4]
  have h6 : combined_work_rate = 1 / 4 := by norm_num [h5]
  have h7 : days_to_complete_together = 1 / (1 / 4) := by simp [days_to_complete_together, h6]
  have h8 : days_to_complete_together = 4 := by norm_num [h7]
  exact h8

end renu_suma_work_together_l336_33694


namespace set_elements_l336_33669

def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, b = k * a

theorem set_elements:
  {x : ℤ | ∃ d : ℤ, is_divisor d 12 ∧ d = 6 - x ∧ x ≥ 0} = 
  {0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 18} :=
by {
  sorry
}

end set_elements_l336_33669


namespace Paige_team_players_l336_33673

/-- Paige's team won their dodgeball game and scored 41 points total.
    If Paige scored 11 points and everyone else scored 6 points each,
    prove that the total number of players on the team was 6. -/
theorem Paige_team_players (total_points paige_points other_points : ℕ) (x : ℕ) (H1 : total_points = 41) (H2 : paige_points = 11) (H3 : other_points = 6) (H4 : paige_points + other_points * x = total_points) : x + 1 = 6 :=
by {
  sorry
}

end Paige_team_players_l336_33673


namespace factorize_equivalence_l336_33686

-- declaring that the following definition may not be computable
noncomputable def factorize_expression (x y : ℝ) : Prop :=
  x^2 * y + x * y^2 = x * y * (x + y)

-- theorem to state the proof problem
theorem factorize_equivalence (x y : ℝ) : factorize_expression x y :=
sorry

end factorize_equivalence_l336_33686


namespace three_digit_integers_211_421_l336_33671

def is_one_more_than_multiple_of (n k : ℕ) : Prop :=
  ∃ m : ℕ, n = m * k + 1

theorem three_digit_integers_211_421
  (n : ℕ) (h1 : (100 ≤ n) ∧ (n ≤ 999))
  (h2 : is_one_more_than_multiple_of n 2)
  (h3 : is_one_more_than_multiple_of n 3)
  (h4 : is_one_more_than_multiple_of n 5)
  (h5 : is_one_more_than_multiple_of n 7) :
  n = 211 ∨ n = 421 :=
sorry

end three_digit_integers_211_421_l336_33671


namespace quadratic_inverse_sum_roots_l336_33693

theorem quadratic_inverse_sum_roots (x1 x2 : ℝ) (h1 : x1^2 - 2023 * x1 + 1 = 0) (h2 : x2^2 - 2023 * x2 + 1 = 0) : 
  (1/x1 + 1/x2) = 2023 :=
by
  -- We outline the proof steps that should be accomplished.
  -- These will be placeholders and not part of the actual statement.
  -- sorry allows us to skip the proof.
  sorry

end quadratic_inverse_sum_roots_l336_33693


namespace scientific_notation_of_86_million_l336_33610

theorem scientific_notation_of_86_million :
  86000000 = 8.6 * 10^7 :=
sorry

end scientific_notation_of_86_million_l336_33610


namespace josh_payment_correct_l336_33600

/-- Josh's purchase calculation -/
def josh_total_payment : ℝ :=
  let string_cheese_cost := 0.10
  let number_of_cheeses_per_pack := 20
  let packs_bought := 3
  let sales_tax_rate := 0.12
  let cost_before_tax := packs_bought * number_of_cheeses_per_pack * string_cheese_cost
  let sales_tax := sales_tax_rate * cost_before_tax
  cost_before_tax + sales_tax

theorem josh_payment_correct :
  josh_total_payment = 6.72 := by
  sorry

end josh_payment_correct_l336_33600


namespace children_difference_l336_33659

-- Axiom definitions based on conditions
def initial_children : ℕ := 36
def first_stop_got_off : ℕ := 45
def first_stop_got_on : ℕ := 25
def second_stop_got_off : ℕ := 68
def final_children : ℕ := 12

-- Mathematical formulation of the problem and its proof statement
theorem children_difference :
  ∃ (x : ℕ), 
    initial_children - first_stop_got_off + first_stop_got_on - second_stop_got_off + x = final_children ∧ 
    (first_stop_got_off + second_stop_got_off) - (first_stop_got_on + x) = 24 :=
by 
  sorry

end children_difference_l336_33659


namespace trigon_expr_correct_l336_33683

noncomputable def trigon_expr : ℝ :=
  1 / Real.sin (Real.pi / 6) - 4 * Real.sin (Real.pi / 3)

theorem trigon_expr_correct : trigon_expr = 2 - 2 * Real.sqrt 3 := by
  sorry

end trigon_expr_correct_l336_33683


namespace max_val_neg_5000_l336_33628

noncomputable def max_val_expression (x y : ℝ) : ℝ :=
  (x^2 + (1 / y^2)) * (x^2 + (1 / y^2) - 100) + (y^2 + (1 / x^2)) * (y^2 + (1 / x^2) - 100)

theorem max_val_neg_5000 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ x y, x > 0 ∧ y > 0 ∧ max_val_expression x y = -5000 :=
by
  sorry

end max_val_neg_5000_l336_33628


namespace grade_assignment_ways_l336_33666

theorem grade_assignment_ways (n_students : ℕ) (n_grades : ℕ) (h_students : n_students = 12) (h_grades : n_grades = 4) :
  (n_grades ^ n_students) = 16777216 := by
  rw [h_students, h_grades]
  rfl

end grade_assignment_ways_l336_33666


namespace photos_in_each_album_l336_33612

theorem photos_in_each_album (total_photos : ℕ) (number_of_albums : ℕ) (photos_per_album : ℕ) 
    (h1 : total_photos = 2560) 
    (h2 : number_of_albums = 32) 
    (h3 : total_photos = number_of_albums * photos_per_album) : 
    photos_per_album = 80 := 
by 
    sorry

end photos_in_each_album_l336_33612


namespace cuboid_edge_sum_l336_33692

-- Define the properties of a cuboid
structure Cuboid (α : Type) [LinearOrderedField α] where
  length : α
  width : α
  height : α

-- Define the volume of a cuboid
def volume {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  c.length * c.width * c.height

-- Define the surface area of a cuboid
def surface_area {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  2 * (c.length * c.width + c.width * c.height + c.height * c.length)

-- Define the sum of all edges of a cuboid
def edge_sum {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  4 * (c.length + c.width + c.height)

-- Given a geometric progression property
def gp_property {α : Type} [LinearOrderedField α] (c : Cuboid α) (q a : α) : Prop :=
  c.length = q * a ∧ c.width = a ∧ c.height = a / q

-- The main problem to be stated in Lean
theorem cuboid_edge_sum (α : Type) [LinearOrderedField α] (c : Cuboid α) (a q : α)
  (h1 : volume c = 8)
  (h2 : surface_area c = 32)
  (h3 : gp_property c q a) :
  edge_sum c = 32 := by
    sorry

end cuboid_edge_sum_l336_33692


namespace bathroom_area_l336_33655

def tile_size : ℝ := 0.5 -- Each tile is 0.5 feet

structure Section :=
  (width : ℕ)
  (length : ℕ)

def longer_section : Section := ⟨15, 25⟩
def alcove : Section := ⟨10, 8⟩

def area (s : Section) : ℝ := (s.width * tile_size) * (s.length * tile_size)

theorem bathroom_area :
  area longer_section + area alcove = 113.75 := by
  sorry

end bathroom_area_l336_33655


namespace coin_flip_probability_l336_33674

theorem coin_flip_probability :
  let total_flips := 8
  let num_heads := 6
  let total_outcomes := (2: ℝ) ^ total_flips
  let favorable_outcomes := (Nat.choose total_flips num_heads)
  let probability := favorable_outcomes / total_outcomes
  probability = (7 / 64 : ℝ) :=
by
  sorry

end coin_flip_probability_l336_33674


namespace interval_length_l336_33679

theorem interval_length (a b : ℝ) (h : ∀ x : ℝ, a + 1 ≤ 3 * x + 6 ∧ 3 * x + 6 ≤ b - 2) :
  (b - a = 57) :=
sorry

end interval_length_l336_33679


namespace average_monthly_balance_l336_33614

def january_balance : ℕ := 100
def february_balance : ℕ := 200
def march_balance : ℕ := 150
def april_balance : ℕ := 150
def may_balance : ℕ := 180
def number_of_months : ℕ := 5
def total_balance : ℕ := january_balance + february_balance + march_balance + april_balance + may_balance

theorem average_monthly_balance :
  (january_balance + february_balance + march_balance + april_balance + may_balance) / number_of_months = 156 := by
  sorry

end average_monthly_balance_l336_33614


namespace unique_function_and_sum_calculate_n_times_s_l336_33636

def f : ℝ → ℝ := sorry

theorem unique_function_and_sum :
  (∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z) →
  (∃! g : ℝ → ℝ, ∀ x, f x = g x) ∧ f 3 = 0 :=
sorry

theorem calculate_n_times_s :
  ∃ n s : ℕ, (∃! f : ℝ → ℝ, ∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z) ∧ n = 1 ∧ s = (0 : ℝ) ∧ n * s = 0 :=
sorry

end unique_function_and_sum_calculate_n_times_s_l336_33636


namespace sum_first_8_terms_eq_8_l336_33606

noncomputable def arithmetic_sequence_sum (n : ℕ) (a₁ d : ℤ) : ℤ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

theorem sum_first_8_terms_eq_8
  (a : ℕ → ℤ)
  (h_arith_seq : ∀ n : ℕ, a (n + 1) = a 1 + ↑n * d)
  (h_a1 : a 1 = 8)
  (h_a4_a6 : a 4 + a 6 = 0) :
  arithmetic_sequence_sum 8 8 (-2) = 8 := 
by
  sorry

end sum_first_8_terms_eq_8_l336_33606


namespace lemon_juice_calculation_l336_33624

noncomputable def lemon_juice_per_lemon (table_per_dozen : ℕ) (dozens : ℕ) (lemons : ℕ) : ℕ :=
  (table_per_dozen * dozens) / lemons

theorem lemon_juice_calculation :
  lemon_juice_per_lemon 12 3 9 = 4 :=
by
  -- proof would be here
  sorry

end lemon_juice_calculation_l336_33624


namespace distinct_three_digit_numbers_count_l336_33696

theorem distinct_three_digit_numbers_count : 
  ∃! n : ℕ, n = 5 * 4 * 3 :=
by
  use 60
  sorry

end distinct_three_digit_numbers_count_l336_33696


namespace problem1_domain_valid_problem2_domain_valid_l336_33646

-- Definition of the domains as sets.

def domain1 (x : ℝ) : Prop := ∃ k : ℤ, x = 2 * k * Real.pi

def domain2 (x : ℝ) : Prop := (-3 ≤ x ∧ x < -Real.pi / 2) ∨ (0 < x ∧ x < Real.pi / 2)

-- Theorem statements for the domains.

theorem problem1_domain_valid (x : ℝ) : (∀ y : ℝ, y = Real.log (Real.cos x) → y ≥ 0) ↔ domain1 x := sorry

theorem problem2_domain_valid (x : ℝ) : 
  (∀ y : ℝ, y = Real.log (Real.sin (2 * x)) + Real.sqrt (9 - x ^ 2) → y ∈ Set.Icc (-3) 3) ↔ domain2 x := sorry

end problem1_domain_valid_problem2_domain_valid_l336_33646


namespace degree_to_radian_conversion_l336_33637

theorem degree_to_radian_conversion : (-330 : ℝ) * (π / 180) = -(11 * π / 6) :=
by 
  sorry

end degree_to_radian_conversion_l336_33637


namespace circles_tangent_internally_l336_33670

theorem circles_tangent_internally 
  (x y : ℝ) 
  (h : x^4 - 16 * x^2 + 2 * x^2 * y^2 - 16 * y^2 + y^4 = 4 * x^3 + 4 * x * y^2 - 64 * x) :
  ∃ c₁ c₂ : ℝ × ℝ, 
    (c₁ = (0, 0)) ∧ (c₂ = (2, 0)) ∧ 
    ((x - c₁.1)^2 + (y - c₁.2)^2 = 16) ∧ 
    ((x - c₂.1)^2 + (y - c₂.2)^2 = 4) ∧
    dist c₁ c₂ = 2 := 
sorry

end circles_tangent_internally_l336_33670


namespace pyramid_height_l336_33690

-- Define the heights of individual blocks and the structure of the pyramid.
def block_height := 10 -- in centimeters
def num_layers := 3

-- Define the total height of the pyramid as the sum of the heights of all blocks.
def total_height (block_height : Nat) (num_layers : Nat) := block_height * num_layers

-- The theorem stating that the total height of the stack is 30 cm given the conditions.
theorem pyramid_height : total_height block_height num_layers = 30 := by
  sorry

end pyramid_height_l336_33690


namespace matrix_sum_correct_l336_33638

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, 3], ![-2, 1]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![-1, 5], ![8, -3]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 8], ![6, -2]]

theorem matrix_sum_correct : A + B = C := by
  sorry

end matrix_sum_correct_l336_33638


namespace base_16_zeros_in_15_factorial_l336_33672

-- Definition of the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definition of the power function to generalize \( a^b \)
def power (a b : ℕ) : ℕ :=
  if b = 0 then 1 else a * power a (b - 1)

-- The constraints of the problem
def k_zeros_base_16 (n : ℕ) (k : ℕ) : Prop :=
  ∃ p, factorial n = p * power 16 k ∧ ¬ (∃ q, factorial n = q * power 16 (k + 1))

-- The main theorem we want to prove
theorem base_16_zeros_in_15_factorial : ∃ k, k_zeros_base_16 15 k ∧ k = 3 :=
by 
  sorry -- Proof to be found

end base_16_zeros_in_15_factorial_l336_33672


namespace probability_at_least_one_first_class_part_l336_33691

-- Define the problem constants
def total_parts : ℕ := 6
def first_class_parts : ℕ := 4
def second_class_parts : ℕ := 2
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Define the target probability
def target_probability : ℚ := 14 / 15

-- Statement of the problem as a Lean theorem
theorem probability_at_least_one_first_class_part :
  (1 - (choose second_class_parts 2 : ℚ) / (choose total_parts 2 : ℚ)) = target_probability :=
by
  -- the proof is omitted
  sorry

end probability_at_least_one_first_class_part_l336_33691


namespace bill_annual_healthcare_cost_l336_33687

def hourly_wage := 25
def weekly_hours := 30
def weeks_per_month := 4
def months_per_year := 12
def normal_monthly_price := 500
def annual_income := hourly_wage * weekly_hours * weeks_per_month * months_per_year
def subsidy (income : ℕ) : ℕ :=
  if income < 10000 then 90
  else if income ≤ 40000 then 50
  else if income > 50000 then 20
  else 0
def monthly_cost_after_subsidy := (normal_monthly_price * (100 - subsidy annual_income)) / 100
def annual_cost := monthly_cost_after_subsidy * months_per_year

theorem bill_annual_healthcare_cost : annual_cost = 3000 := by
  sorry

end bill_annual_healthcare_cost_l336_33687


namespace five_g_speeds_l336_33680

theorem five_g_speeds (m : ℝ) :
  (1400 / 50) - (1400 / (50 * m)) = 24 → m = 7 :=
by
  sorry

end five_g_speeds_l336_33680


namespace money_distribution_l336_33698

variable (A B C : ℕ)

theorem money_distribution :
  A + B + C = 500 →
  B + C = 360 →
  C = 60 →
  A + C = 200 :=
by
  intros h1 h2 h3
  sorry

end money_distribution_l336_33698


namespace average_birds_seen_correct_l336_33685

-- Define the number of birds seen by each person
def birds_seen_by_marcus : ℕ := 7
def birds_seen_by_humphrey : ℕ := 11
def birds_seen_by_darrel : ℕ := 9

-- Define the number of people
def number_of_people : ℕ := 3

-- Calculate the total number of birds seen
def total_birds_seen : ℕ := birds_seen_by_marcus + birds_seen_by_humphrey + birds_seen_by_darrel

-- Calculate the average number of birds seen
def average_birds_seen : ℕ := total_birds_seen / number_of_people

-- Proof statement
theorem average_birds_seen_correct :
  average_birds_seen = 9 :=
by
  -- Leaving the proof out as instructed
  sorry

end average_birds_seen_correct_l336_33685


namespace imaginary_part_of_z_l336_33650

-- Define the problem conditions and what to prove
theorem imaginary_part_of_z (z : ℂ) (h : (1 - I) * z = I) : z.im = 1 / 2 :=
sorry

end imaginary_part_of_z_l336_33650


namespace positive_number_decreased_by_4_is_21_times_reciprocal_l336_33662

theorem positive_number_decreased_by_4_is_21_times_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x - 4 = 21 * (1 / x)) : x = 7 := 
sorry

end positive_number_decreased_by_4_is_21_times_reciprocal_l336_33662


namespace eccentricities_ellipse_hyperbola_l336_33613

theorem eccentricities_ellipse_hyperbola :
  let a := 2
  let b := -5
  let c := 2
  let delta := b^2 - 4 * a * c
  let x1 := (-b + Real.sqrt delta) / (2 * a)
  let x2 := (-b - Real.sqrt delta) / (2 * a)
  (x1 > 1) ∧ (0 < x2) ∧ (x2 < 1) :=
sorry

end eccentricities_ellipse_hyperbola_l336_33613


namespace pet_store_cages_l336_33664

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 78) (h2 : sold_puppies = 30) (h3 : puppies_per_cage = 8) : 
  (initial_puppies - sold_puppies) / puppies_per_cage = 6 := 
by 
  sorry

end pet_store_cages_l336_33664


namespace rain_probability_tel_aviv_l336_33676

open scoped Classical

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv :
  binomial_probability 6 4 0.5 = 0.234375 :=
by 
  sorry

end rain_probability_tel_aviv_l336_33676


namespace common_tangents_l336_33603

noncomputable def radius1 := 8
noncomputable def radius2 := 6
noncomputable def distance := 2

theorem common_tangents (r1 r2 d : ℕ) 
  (h1 : r1 = radius1) 
  (h2 : r2 = radius2) 
  (h3 : d = distance) :
  (d = r1 - r2) → 1 = 1 := by 
  sorry

end common_tangents_l336_33603


namespace range_of_m_if_forall_x_gt_0_l336_33651

open Real

theorem range_of_m_if_forall_x_gt_0 (m : ℝ) :
  (∀ x : ℝ, 0 < x → x + 1/x - m > 0) ↔ m < 2 :=
by
  -- Placeholder proof
  sorry

end range_of_m_if_forall_x_gt_0_l336_33651


namespace complement_union_eq_l336_33625

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end complement_union_eq_l336_33625


namespace measure_of_side_XY_l336_33601

theorem measure_of_side_XY 
  (a b c : ℝ) 
  (Area : ℝ)
  (h1 : a = 30)
  (h2 : b = 60)
  (h3 : c = 90)
  (h4 : a + b + c = 180)
  (h_area : Area = 36)
  : (∀ (XY YZ XZ : ℝ), XY = 4.56) :=
by
  sorry

end measure_of_side_XY_l336_33601


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l336_33621

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l336_33621


namespace value_of_f_at_2_l336_33682

def f (x : ℝ) := x^2 + 2 * x - 3

theorem value_of_f_at_2 : f 2 = 5 :=
by
  sorry

end value_of_f_at_2_l336_33682


namespace money_left_after_expenses_l336_33626

theorem money_left_after_expenses :
  let salary := 8123.08
  let food_expense := (1:ℝ) / 3 * salary
  let rent_expense := (1:ℝ) / 4 * salary
  let clothes_expense := (1:ℝ) / 5 * salary
  let total_expense := food_expense + rent_expense + clothes_expense
  let money_left := salary - total_expense
  money_left = 1759.00 :=
sorry

end money_left_after_expenses_l336_33626


namespace vector_decomposition_l336_33689

def x : ℝ×ℝ×ℝ := (8, 0, 5)
def p : ℝ×ℝ×ℝ := (2, 0, 1)
def q : ℝ×ℝ×ℝ := (1, 1, 0)
def r : ℝ×ℝ×ℝ := (4, 1, 2)

theorem vector_decomposition :
  x = (1:ℝ) • p + (-2:ℝ) • q + (2:ℝ) • r :=
by
  sorry

end vector_decomposition_l336_33689


namespace problem_l336_33639

variable {f : ℝ → ℝ}

-- Condition: f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Condition: f is monotonically decreasing on (0, +∞)
def monotone_decreasing_on_pos (f : ℝ → ℝ) : Prop := 
  ∀ ⦃x y : ℝ⦄, 0 < x → 0 < y → x < y → f y < f x

theorem problem (h_even : even_function f) (h_mon_dec : monotone_decreasing_on_pos f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end problem_l336_33639


namespace diagonals_in_nonagon_l336_33660

-- Define the properties of the polygon
def convex : Prop := true
def sides (n : ℕ) : Prop := n = 9
def right_angles (count : ℕ) : Prop := count = 2

-- Define the formula for the number of diagonals in a polygon with 'n' sides
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The theorem definition
theorem diagonals_in_nonagon :
  convex →
  (sides 9) →
  (right_angles 2) →
  number_of_diagonals 9 = 27 :=
by
  sorry

end diagonals_in_nonagon_l336_33660


namespace corrected_mean_is_correct_l336_33634

-- Define the initial conditions
def initial_mean : ℝ := 36
def n_obs : ℝ := 50
def incorrect_obs : ℝ := 23
def correct_obs : ℝ := 45

-- Calculate the incorrect total sum
def incorrect_total_sum : ℝ := initial_mean * n_obs

-- Define the corrected total sum
def corrected_total_sum : ℝ := incorrect_total_sum - incorrect_obs + correct_obs

-- State the main theorem to be proved
theorem corrected_mean_is_correct : corrected_total_sum / n_obs = 36.44 := by
  sorry

end corrected_mean_is_correct_l336_33634


namespace total_coffee_blend_cost_l336_33675

-- Define the cost per pound of coffee types A and B
def cost_per_pound_A := 4.60
def cost_per_pound_B := 5.95

-- Given the pounds of coffee for Type A and the blend condition for Type B
def pounds_A := 67.52
def pounds_B := 2 * pounds_A

-- Total cost calculation
def total_cost := (pounds_A * cost_per_pound_A) + (pounds_B * cost_per_pound_B)

-- Theorem statement: The total cost of the coffee blend is $1114.08
theorem total_coffee_blend_cost : total_cost = 1114.08 := by
  -- Proof omitted
  sorry

end total_coffee_blend_cost_l336_33675


namespace bathroom_square_footage_l336_33684

theorem bathroom_square_footage
  (tiles_width : ℕ)
  (tiles_length : ℕ)
  (tile_size_inches : ℕ)
  (inches_per_foot : ℕ)
  (h1 : tiles_width = 10)
  (h2 : tiles_length = 20)
  (h3 : tile_size_inches = 6)
  (h4 : inches_per_foot = 12)
: (tiles_length * tile_size_inches / inches_per_foot) * (tiles_width * tile_size_inches / inches_per_foot) = 50 := 
by
  sorry

end bathroom_square_footage_l336_33684


namespace tan_sin_cos_identity_l336_33642

theorem tan_sin_cos_identity {x : ℝ} (htan : Real.tan x = 1 / 3) : Real.sin x * Real.cos x + 1 = 13 / 10 :=
by
  sorry

end tan_sin_cos_identity_l336_33642


namespace perimeter_equals_interior_tiles_l336_33652

theorem perimeter_equals_interior_tiles (m n : ℕ) (h : m ≤ n) :
  (2 * m + 2 * n - 4 = 2 * (m * n) - (2 * m + 2 * n - 4)) ↔ (m = 5 ∧ n = 12 ∨ m = 6 ∧ n = 8) :=
by sorry

end perimeter_equals_interior_tiles_l336_33652


namespace one_fourth_div_one_eighth_l336_33635

theorem one_fourth_div_one_eighth : (1 / 4) / (1 / 8) = 2 := by
  sorry

end one_fourth_div_one_eighth_l336_33635


namespace infinite_squares_in_arithmetic_sequence_l336_33618

open Nat Int

theorem infinite_squares_in_arithmetic_sequence
  (a d : ℤ) (h_d_nonneg : d ≥ 0) (x : ℤ) 
  (hx_square : ∃ n : ℕ, a + n * d = x * x) :
  ∃ (infinitely_many_n : ℕ → Prop), 
    (∀ k : ℕ, ∃ n : ℕ, infinitely_many_n n ∧ a + n * d = (x + k * d) * (x + k * d)) :=
sorry

end infinite_squares_in_arithmetic_sequence_l336_33618


namespace all_n_eq_one_l336_33604

theorem all_n_eq_one (k : ℕ) (n : ℕ → ℕ)
  (h₁ : k ≥ 2)
  (h₂ : ∀ i, 1 ≤ i ∧ i < k → (n (i + 1)) ∣ 2^(n i) - 1)
  (h₃ : (n 1) ∣ 2^(n k) - 1) :
  ∀ i, 1 ≤ i ∧ i ≤ k → n i = 1 := 
sorry

end all_n_eq_one_l336_33604


namespace no_rational_roots_l336_33677

theorem no_rational_roots {p q : ℤ} (hp : p % 2 = 1) (hq : q % 2 = 1) :
  ¬ ∃ x : ℚ, x^2 + (2 * p) * x + (2 * q) = 0 :=
by
  -- proof using contradiction technique
  sorry

end no_rational_roots_l336_33677


namespace range_of_2x_plus_y_l336_33647

theorem range_of_2x_plus_y {x y: ℝ} (h: x^2 / 4 + y^2 = 1) : -Real.sqrt 17 ≤ 2 * x + y ∧ 2 * x + y ≤ Real.sqrt 17 :=
sorry

end range_of_2x_plus_y_l336_33647


namespace intersection_of_A_and_B_l336_33609

def A : Set ℝ := { x | 0 < x ∧ x < 2 }
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B : A ∩ B = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l336_33609


namespace problem_statement_l336_33608

theorem problem_statement : 
  (∀ x y : ℤ, y = 2 * x^2 - 3 * x + 4 ∧ y = 6 ∧ x = 2) → (2 * 2 - 3 * (-3) + 4 * 4 = 29) :=
by
  intro h
  sorry

end problem_statement_l336_33608


namespace cow_count_l336_33611

theorem cow_count
  (initial_cows : ℕ) (cows_died : ℕ) (cows_sold : ℕ)
  (increase_cows : ℕ) (gift_cows : ℕ) (final_cows : ℕ) (bought_cows : ℕ) :
  initial_cows = 39 ∧ cows_died = 25 ∧ cows_sold = 6 ∧
  increase_cows = 24 ∧ gift_cows = 8 ∧ final_cows = 83 →
  bought_cows = 43 :=
by
  sorry

end cow_count_l336_33611


namespace find_k_l336_33641

theorem find_k (k : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-1, k)
  (a.1 * (2 * a.1 - b.1) + a.2 * (2 * a.2 - b.2)) = 0 → k = 12 := sorry

end find_k_l336_33641


namespace men_work_days_l336_33658

theorem men_work_days (M : ℕ) (W : ℕ) (h : W / (M * 40) = W / ((M - 5) * 50)) : M = 25 :=
by
  -- Will add the proof later
  sorry

end men_work_days_l336_33658


namespace range_of_a_l336_33617

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - (a + 1) * x + a ≤ 0 → -4 ≤ x ∧ x ≤ 3) ↔ (-4 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l336_33617


namespace cosine_expression_value_l336_33649

noncomputable def c : ℝ := 2 * Real.pi / 7

theorem cosine_expression_value :
  (Real.cos (3 * c) * Real.cos (5 * c) * Real.cos (6 * c)) / 
  (Real.cos c * Real.cos (2 * c) * Real.cos (3 * c)) = 1 :=
by
  sorry

end cosine_expression_value_l336_33649


namespace pascal_row_20_fifth_sixth_sum_l336_33665

-- Conditions from the problem
def pascal_element (n k : ℕ) : ℕ := Nat.choose n k

-- Question translated to a Lean theorem
theorem pascal_row_20_fifth_sixth_sum :
  pascal_element 20 4 + pascal_element 20 5 = 20349 :=
by
  sorry

end pascal_row_20_fifth_sixth_sum_l336_33665


namespace func_symmetry_monotonicity_range_of_m_l336_33622

open Real

theorem func_symmetry_monotonicity (f : ℝ → ℝ)
  (h1 : ∀ x, f (3 + x) = f (1 - x))
  (h2 : ∀ x1 x2, 2 < x1 → 2 < x2 → (f x1 - f x2) / (x1 - x2) > 0) :
  (∀ x, f (2 + x) = f (2 - x)) ∧
  (∀ x1 x2, (x1 > 2 ∧ x2 > 2 → f x1 < f x2 → x1 < x2) ∧
            (x2 > 2 ∧ x1 > x2 → f x2 < f x1 → x2 < x1)) := 
sorry

theorem range_of_m (f : ℝ → ℝ)
  (h : ∀ θ : ℝ, f (cos θ ^ 2 + 2 * (m : ℝ) ^ 2 + 2) < f (sin θ + m ^ 2 - 3 * m - 2)) :
  ∀ m, (3 - sqrt 42) / 6 < m ∧ m < (3 + sqrt 42) / 6 :=
sorry

end func_symmetry_monotonicity_range_of_m_l336_33622


namespace haley_initial_music_files_l336_33620

theorem haley_initial_music_files (M : ℕ) 
  (h1 : M + 42 - 11 = 58) : M = 27 := 
by
  sorry

end haley_initial_music_files_l336_33620


namespace math_problem_l336_33619

variable (x Q : ℝ)

theorem math_problem (h : 5 * (6 * x - 3 * Real.pi) = Q) :
  15 * (18 * x - 9 * Real.pi) = 9 * Q := 
by
  sorry

end math_problem_l336_33619


namespace mod_81256_eq_16_l336_33605

theorem mod_81256_eq_16 : ∃ n : ℤ, 0 ≤ n ∧ n < 31 ∧ 81256 % 31 = n := by
  use 16
  sorry

end mod_81256_eq_16_l336_33605


namespace fraction_of_termite_ridden_homes_collapsing_l336_33695

variable (T : ℕ) -- T represents the total number of homes
variable (termiteRiddenFraction : ℚ := 1/3) -- Fraction of homes that are termite-ridden
variable (termiteRiddenNotCollapsingFraction : ℚ := 1/7) -- Fraction of homes that are termite-ridden but not collapsing

theorem fraction_of_termite_ridden_homes_collapsing :
  termiteRiddenFraction - termiteRiddenNotCollapsingFraction = 4/21 :=
by
  -- Proof goes here
  sorry

end fraction_of_termite_ridden_homes_collapsing_l336_33695


namespace graveling_cost_l336_33629

theorem graveling_cost
  (length_lawn : ℝ) (width_lawn : ℝ)
  (width_road : ℝ)
  (cost_per_sq_m : ℝ)
  (h1: length_lawn = 80) (h2: width_lawn = 40) (h3: width_road = 10) (h4: cost_per_sq_m = 3) :
  (length_lawn * width_road + width_lawn * width_road - width_road * width_road) * cost_per_sq_m = 3900 := 
by
  sorry

end graveling_cost_l336_33629


namespace find_original_number_l336_33645

theorem find_original_number (x : ℕ) (h : 3 * (2 * x + 9) = 57) : x = 5 := 
by sorry

end find_original_number_l336_33645


namespace original_recipe_calls_for_4_tablespoons_l336_33653

def key_limes := 8
def juice_per_lime := 1 -- in tablespoons
def juice_doubled := key_limes * juice_per_lime
def original_juice_amount := juice_doubled / 2

theorem original_recipe_calls_for_4_tablespoons :
  original_juice_amount = 4 :=
by
  sorry

end original_recipe_calls_for_4_tablespoons_l336_33653


namespace additional_male_students_l336_33667

variable (a : ℕ)

theorem additional_male_students (h : a > 20) : a - 20 = (a - 20) := 
by 
  sorry

end additional_male_students_l336_33667


namespace employees_in_factory_l336_33681

theorem employees_in_factory (initial_total : ℕ) (init_prod : ℕ) (init_admin : ℕ)
  (increase_prod_frac : ℚ) (increase_admin_frac : ℚ) :
  initial_total = 1200 →
  init_prod = 800 →
  init_admin = 400 →
  increase_prod_frac = 0.35 →
  increase_admin_frac = 3 / 5 →
  init_prod + init_prod * increase_prod_frac +
  init_admin + init_admin * increase_admin_frac = 1720 := by
  intros h_total h_prod h_admin h_inc_prod h_inc_admin
  sorry

end employees_in_factory_l336_33681


namespace lars_total_breads_per_day_l336_33644

def loaves_per_hour : ℕ := 10
def baguettes_per_two_hours : ℕ := 30
def hours_per_day : ℕ := 6

theorem lars_total_breads_per_day :
  (loaves_per_hour * hours_per_day) + ((hours_per_day / 2) * baguettes_per_two_hours) = 150 :=
  by 
  sorry

end lars_total_breads_per_day_l336_33644


namespace scenario1_scenario2_scenario3_l336_33678

noncomputable def scenario1_possible_situations : Nat :=
  12

noncomputable def scenario2_possible_situations : Nat :=
  144

noncomputable def scenario3_possible_situations : Nat :=
  50

theorem scenario1 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) (not_consecutive : Prop) :
  shots = 10 ∧ hits = 7 ∧ consecutive_hits = 5 ∧ remaining_hits = 2 ∧ not_consecutive → 
  scenario1_possible_situations = 12 := by
  sorry

theorem scenario2 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) :
  shots = 10 ∧ hits = 7 ∧ consecutive_hits = 4 ∧ remaining_hits = 3 → 
  scenario2_possible_situations = 144 := by
  sorry

theorem scenario3 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) :
  shots = 10 ∧ hits = 6 ∧ consecutive_hits = 4 ∧ remaining_hits = 2 → 
  scenario3_possible_situations = 50 := by
  sorry

end scenario1_scenario2_scenario3_l336_33678


namespace disproving_rearranged_sum_l336_33632

noncomputable section

open scoped BigOperators

variable {a : ℕ → ℝ} {f : ℕ → ℕ}

-- Conditions
def summable_a (a : ℕ → ℝ) : Prop :=
  ∑' i, a i = 1

def strictly_decreasing_abs (a : ℕ → ℝ) : Prop :=
  ∀ n m, n < m → abs (a n) > abs (a m)

def bijection (f : ℕ → ℕ) : Prop :=
  ∀ n, ∃ m, f m = n

def limit_condition (a : ℕ → ℝ) (f : ℕ → ℕ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, abs ((f n : ℤ) - (n : ℤ)) * abs (a n) < ε

-- Statement
theorem disproving_rearranged_sum :
  summable_a a ∧
  strictly_decreasing_abs a ∧
  bijection f ∧
  limit_condition a f →
  ∑' i, a (f i) ≠ 1 :=
sorry

end disproving_rearranged_sum_l336_33632


namespace remaining_area_exclude_smaller_rectangles_l336_33697

-- Conditions from part a)
variables (x : ℕ)
def large_rectangle_area := (x + 8) * (x + 6)
def small1_rectangle_area := (2 * x - 1) * (x - 1)
def small2_rectangle_area := (x - 3) * (x - 5)

-- Proof statement from part c)
theorem remaining_area_exclude_smaller_rectangles :
  large_rectangle_area x - (small1_rectangle_area x - small2_rectangle_area x) = 25 * x + 62 :=
by
  sorry

end remaining_area_exclude_smaller_rectangles_l336_33697


namespace plantable_area_l336_33661

noncomputable def flowerbed_r := 10
noncomputable def path_w := 4
noncomputable def full_area := 100 * Real.pi
noncomputable def segment_area := 20.67 * Real.pi * 2 -- each path affects two segments

theorem plantable_area :
  full_area - segment_area = 58.66 * Real.pi := 
by sorry

end plantable_area_l336_33661


namespace problem_statement_l336_33643

/-
Definitions of the given conditions:
- Circle P: (x-1)^2 + y^2 = 8, center C.
- Point M(-1,0).
- Line y = kx + m intersects trajectory at points A and B.
- k_{OA} \cdot k_{OB} = -1/2.
-/

noncomputable def Circle_P : Set (ℝ × ℝ) :=
  { p | (p.1 - 1)^2 + p.2^2 = 8 }

def Point_M : (ℝ × ℝ) := (-1, 0)

def Trajectory_C : Set (ℝ × ℝ) :=
  { p | p.1^2 / 2 + p.2^2 = 1 }

def Line_kx_m (k m : ℝ) : Set (ℝ × ℝ) :=
  { p | p.2 = k * p.1 + m }

def k_OA_OB (k_OA k_OB : ℝ) : Prop :=
  k_OA * k_OB = -1/2

/-
Mathematical equivalence proof problem:
- Prove the trajectory of center C is an ellipse with equation x^2/2 + y^2 = 1.
- Prove that if line y=kx+m intersects with the trajectory, the area of the triangle AOB is a fixed value.
-/

theorem problem_statement (k m : ℝ)
    (h_intersects : ∃ A B : ℝ × ℝ, A ∈ (Trajectory_C ∩ Line_kx_m k m) ∧ B ∈ (Trajectory_C ∩ Line_kx_m k m))
    (k_OA k_OB : ℝ) (h_k_OA_k_OB : k_OA_OB k_OA k_OB) :
  ∃ (C_center_trajectory : Trajectory_C),
  ∃ (area_AOB : ℝ), area_AOB = (3 * Real.sqrt 2) / 2 :=
sorry

end problem_statement_l336_33643


namespace union_of_sets_l336_33663

def setA : Set ℕ := {0, 1}
def setB : Set ℕ := {0, 2}

theorem union_of_sets : setA ∪ setB = {0, 1, 2} := 
sorry

end union_of_sets_l336_33663


namespace original_price_of_item_l336_33656

theorem original_price_of_item (P : ℝ) 
(selling_price : ℝ) 
(h1 : 0.9 * P = selling_price) 
(h2 : selling_price = 675) : 
P = 750 := sorry

end original_price_of_item_l336_33656


namespace range_of_a_l336_33657

theorem range_of_a (a b c : ℝ) (h₁ : a + b + c = 2) (h₂ : a^2 + b^2 + c^2 = 4) (h₃ : a > b ∧ b > c) :
  (2 / 3 < a ∧ a < 2) :=
sorry

end range_of_a_l336_33657


namespace anna_money_left_eur_l336_33654

noncomputable def total_cost_usd : ℝ := 4 * 1.50 + 7 * 2.25 + 3 * 0.75 + 3.00 * 0.80
def sales_tax_rate : ℝ := 0.075
def exchange_rate : ℝ := 0.85
def initial_amount_usd : ℝ := 50

noncomputable def total_cost_with_tax_usd : ℝ := total_cost_usd * (1 + sales_tax_rate)
noncomputable def total_cost_eur : ℝ := total_cost_with_tax_usd * exchange_rate
noncomputable def initial_amount_eur : ℝ := initial_amount_usd * exchange_rate

noncomputable def money_left_eur : ℝ := initial_amount_eur - total_cost_eur

theorem anna_money_left_eur : abs (money_left_eur - 18.38) < 0.01 := by
  -- Add proof steps here
  sorry

end anna_money_left_eur_l336_33654


namespace intersection_of_M_and_N_l336_33633

open Set

def M : Set ℕ := {0, 1, 2, 3}
def N : Set ℕ := {2, 3}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := 
by 
  sorry

end intersection_of_M_and_N_l336_33633


namespace range_of_m_l336_33648

open Real

-- Defining conditions as propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m*x + 1 ≠ 0
def q (m : ℝ) : Prop := m > 1
def p_or_q (m : ℝ) : Prop := p m ∨ q m
def p_and_q (m : ℝ) : Prop := p m ∧ q m

-- Mathematically equivalent proof problem
theorem range_of_m (m : ℝ) (H1 : p_or_q m) (H2 : ¬p_and_q m) : -2 < m ∧ m ≤ 1 ∨ 2 ≤ m :=
by
  sorry

end range_of_m_l336_33648


namespace train_platform_length_l336_33640

noncomputable def kmph_to_mps (v : ℕ) : ℕ := v * 1000 / 3600

theorem train_platform_length :
  ∀ (train_length speed_kmph time_sec : ℕ),
    speed_kmph = 36 →
    train_length = 175 →
    time_sec = 40 →
    let speed_mps := kmph_to_mps speed_kmph
    let total_distance := speed_mps * time_sec
    let platform_length := total_distance - train_length
    platform_length = 225 :=
by
  intros train_length speed_kmph time_sec h_speed h_train h_time
  let speed_mps := kmph_to_mps speed_kmph
  let total_distance := speed_mps * time_sec
  let platform_length := total_distance - train_length
  sorry

end train_platform_length_l336_33640


namespace rick_books_total_l336_33688

theorem rick_books_total 
  (N : ℕ)
  (h : N / 16 = 25) : 
  N = 400 := 
  sorry

end rick_books_total_l336_33688


namespace union_of_M_N_is_real_set_l336_33607

-- Define the set M
def M : Set ℝ := { x | x^2 + 3 * x + 2 > 0 }

-- Define the set N
def N : Set ℝ := { x | (1 / 2 : ℝ) ^ x ≤ 4 }

-- The goal is to prove that the union of M and N is the set of all real numbers
theorem union_of_M_N_is_real_set : M ∪ N = Set.univ :=
by
  sorry

end union_of_M_N_is_real_set_l336_33607


namespace value_of_X_is_one_l336_33699

-- Problem: Given the numbers 28 at the start of a row, 17 in the middle, and -15 in the same column as X,
-- we show the value of X must be 1 because the sequences are arithmetic.

theorem value_of_X_is_one (d : ℤ) (X : ℤ) :
  -- Conditions
  (17 - X = d) ∧ 
  (X - (-15) = d) ∧ 
  (d = 16) →
  -- Conclusion: X must be 1
  X = 1 :=
by 
  sorry

end value_of_X_is_one_l336_33699


namespace dealership_sales_prediction_l336_33627

theorem dealership_sales_prediction (sports_cars_sold sedans SUVs : ℕ) 
    (ratio_sc_sedans : 3 * sedans = 5 * sports_cars_sold) 
    (ratio_sc_SUVs : sports_cars_sold = 2 * SUVs) 
    (sports_cars_sold_next_month : sports_cars_sold = 36) :
    (sedans = 60 ∧ SUVs = 72) :=
sorry

end dealership_sales_prediction_l336_33627


namespace bridget_initial_skittles_l336_33668

theorem bridget_initial_skittles (b : ℕ) (h : b + 4 = 8) : b = 4 :=
by {
  sorry
}

end bridget_initial_skittles_l336_33668


namespace arithmetic_mean_after_removal_l336_33616

theorem arithmetic_mean_after_removal 
  (mean_original : ℝ) (num_original : ℕ) 
  (nums_removed : List ℝ) (mean_new : ℝ)
  (h1 : mean_original = 50) 
  (h2 : num_original = 60) 
  (h3 : nums_removed = [60, 65, 70, 40]) 
  (h4 : mean_new = 49.38) :
  let sum_original := mean_original * num_original
  let num_remaining := num_original - nums_removed.length
  let sum_removed := List.sum nums_removed
  let sum_new := sum_original - sum_removed
  
  mean_new = sum_new / num_remaining :=
sorry

end arithmetic_mean_after_removal_l336_33616


namespace third_butcher_delivered_8_packages_l336_33630

variables (x y z t1 t2 t3 : ℕ)

-- Given Conditions
axiom h1 : x = 10
axiom h2 : y = 7
axiom h3 : 4 * x + 4 * y + 4 * z = 100
axiom t1_time : t1 = 8
axiom t2_time : t2 = 10
axiom t3_time : t3 = 18

-- Proof Problem
theorem third_butcher_delivered_8_packages :
  z = 8 :=
by
  -- proof to be filled
  sorry

end third_butcher_delivered_8_packages_l336_33630


namespace average_speed_with_stoppages_l336_33615

theorem average_speed_with_stoppages
  (avg_speed_without_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ)
  (moving_time_per_hour : ℝ)
  (total_distance_moved : ℝ)
  (total_time_with_stoppages : ℝ) :
  avg_speed_without_stoppages = 60 → 
  stoppage_time_per_hour = 45 / 60 →
  moving_time_per_hour = 15 / 60 →
  total_distance_moved = avg_speed_without_stoppages * moving_time_per_hour →
  total_time_with_stoppages = 1 →
  (total_distance_moved / total_time_with_stoppages) = 15 :=
by
  intros
  sorry

end average_speed_with_stoppages_l336_33615
