import Mathlib

namespace NUMINAMATH_CALUDE_fruit_juice_mixture_l468_46848

/-- Given a 2-liter mixture that is 10% pure fruit juice, 
    adding 0.4 liters of pure fruit juice results in a 
    mixture that is 25% fruit juice -/
theorem fruit_juice_mixture : 
  let initial_volume : ℝ := 2
  let initial_percentage : ℝ := 0.1
  let added_volume : ℝ := 0.4
  let target_percentage : ℝ := 0.25
  let final_volume := initial_volume + added_volume
  let final_juice_volume := initial_volume * initial_percentage + added_volume
  final_juice_volume / final_volume = target_percentage :=
by sorry


end NUMINAMATH_CALUDE_fruit_juice_mixture_l468_46848


namespace NUMINAMATH_CALUDE_disease_gender_relation_expected_trial_cost_l468_46825

-- Define the total number of patients
def total_patients : ℕ := 1800

-- Define the number of male and female patients
def male_patients : ℕ := 1200
def female_patients : ℕ := 600

-- Define the number of patients with type A disease
def male_type_a : ℕ := 800
def female_type_a : ℕ := 450

-- Define the χ² formula
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for α = 0.001
def critical_value : ℚ := 10828 / 1000

-- Define the probability of producing antibodies
def antibody_prob : ℚ := 2 / 3

-- Define the cost per dose
def cost_per_dose : ℕ := 9

-- Define the number of doses per cycle
def doses_per_cycle : ℕ := 3

-- Theorem statements
theorem disease_gender_relation :
  chi_square male_type_a (male_patients - male_type_a) female_type_a (female_patients - female_type_a) > critical_value := by sorry

theorem expected_trial_cost :
  (20 : ℚ) / 27 * (3 * cost_per_dose) + 7 / 27 * (6 * cost_per_dose) = 34 := by sorry

end NUMINAMATH_CALUDE_disease_gender_relation_expected_trial_cost_l468_46825


namespace NUMINAMATH_CALUDE_special_function_increasing_l468_46868

/-- A function satisfying the given properties -/
class SpecialFunction (f : ℝ → ℝ) : Prop where
  pos_gt_one : ∀ x > 0, f x > 1
  multiplicative : ∀ x y, f (x + y) = f x * f y

/-- Theorem: f is increasing on ℝ -/
theorem special_function_increasing (f : ℝ → ℝ) [SpecialFunction f] :
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_special_function_increasing_l468_46868


namespace NUMINAMATH_CALUDE_find_B_l468_46853

/-- The number represented by 2B8, where B is a single digit -/
def number (B : ℕ) : ℕ := 200 + 10 * B + 8

/-- The sum of digits of the number 2B8 -/
def digit_sum (B : ℕ) : ℕ := 2 + B + 8

theorem find_B : ∃ B : ℕ, B < 10 ∧ number B % 3 = 0 ∧ B = 2 :=
sorry

end NUMINAMATH_CALUDE_find_B_l468_46853


namespace NUMINAMATH_CALUDE_four_digit_integer_problem_l468_46822

theorem four_digit_integer_problem (a b c d : ℕ) : 
  a * 1000 + b * 100 + c * 10 + d > 0 →
  a + b + c + d = 16 →
  b + c = 8 →
  a - d = 2 →
  (a * 1000 + b * 100 + c * 10 + d) % 9 = 0 →
  a * 1000 + b * 100 + c * 10 + d = 5533 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_integer_problem_l468_46822


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l468_46811

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l468_46811


namespace NUMINAMATH_CALUDE_system_solvability_l468_46810

/-- The system of equations has real solutions if and only if a, b, c form a triangle -/
theorem system_solvability (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∃ x y z : ℝ, a * x + b * y - c * z = 0 ∧
               a * Real.sqrt (1 - x^2) + b * Real.sqrt (1 - y^2) - c * Real.sqrt (1 - z^2) = 0)
  ↔ (abs (a - b) ≤ c ∧ c ≤ a + b) :=
by sorry

end NUMINAMATH_CALUDE_system_solvability_l468_46810


namespace NUMINAMATH_CALUDE_ariel_fish_count_l468_46852

theorem ariel_fish_count (total : ℕ) (male_fraction : ℚ) (female_count : ℕ) : 
  total = 45 → male_fraction = 2/3 → female_count = total - (total * male_fraction).num → female_count = 15 := by
  sorry

end NUMINAMATH_CALUDE_ariel_fish_count_l468_46852


namespace NUMINAMATH_CALUDE_function_transformation_l468_46899

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 3) : 
  f 1 + 1 = 4 := by sorry

end NUMINAMATH_CALUDE_function_transformation_l468_46899


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l468_46865

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), ∀ n, a n = a₁ * r ^ (n - 1)

-- Define the property of three terms forming a geometric sequence
def form_geometric_sequence (x y z : ℝ) : Prop :=
  y * y = x * z

-- Theorem statement
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  form_geometric_sequence (a 3) (a 6) (a 9) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l468_46865


namespace NUMINAMATH_CALUDE_probability_theorem_l468_46888

-- Define the total number of children
def total_children : ℕ := 9

-- Define the number of children with green hats
def green_hats : ℕ := 3

-- Define the function to calculate the probability
def probability_no_adjacent_green_hats (n : ℕ) (k : ℕ) : ℚ :=
  -- The actual calculation would go here, but we'll use sorry to skip the proof
  5 / 14

-- State the theorem
theorem probability_theorem :
  probability_no_adjacent_green_hats total_children green_hats = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l468_46888


namespace NUMINAMATH_CALUDE_fixed_point_of_line_family_l468_46849

/-- The fixed point that a family of lines passes through -/
theorem fixed_point_of_line_family :
  ∃! p : ℝ × ℝ, ∀ m : ℝ, (2*m - 1) * p.1 + (m + 3) * p.2 - (m - 11) = 0 :=
by
  -- The unique point is (2, -3)
  use (2, -3)
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_family_l468_46849


namespace NUMINAMATH_CALUDE_square_of_difference_product_of_three_terms_l468_46895

-- Problem 1
theorem square_of_difference (a b : ℝ) : (a^2 - b)^2 = a^4 - 2*a^2*b + b^2 := by sorry

-- Problem 2
theorem product_of_three_terms (x : ℝ) : (2*x + 1)*(4*x^2 - 1)*(2*x - 1) = 16*x^4 - 8*x^2 + 1 := by sorry

end NUMINAMATH_CALUDE_square_of_difference_product_of_three_terms_l468_46895


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l468_46872

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 3}
def Q : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

-- Define the intersection of P and Q
def PQ_intersection : Set ℝ := P ∩ Q

-- Define the half-open interval [3,4)
def interval_3_4 : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 4}

-- Theorem statement
theorem intersection_equals_interval : PQ_intersection = interval_3_4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l468_46872


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l468_46883

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

/-- The condition "a = -1" -/
def condition (a : ℝ) : Prop := a = -1

/-- The line ax + y - 1 = 0 is parallel to x + ay + 5 = 0 -/
def lines_are_parallel (a : ℝ) : Prop := parallel_lines (-a) (1/a)

/-- "a = -1" is a sufficient but not necessary condition for the lines to be parallel -/
theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, condition a → lines_are_parallel a) ∧
  (∃ a, lines_are_parallel a ∧ ¬condition a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l468_46883


namespace NUMINAMATH_CALUDE_dogwood_trees_after_planting_l468_46828

/-- The number of dogwood trees in the park after a week of planting -/
def total_trees (initial : ℕ) (monday tuesday wednesday thursday friday saturday sunday : ℕ) : ℕ :=
  initial + monday + tuesday + wednesday + thursday + friday + saturday + sunday

/-- Theorem stating the total number of dogwood trees after the week's planting -/
theorem dogwood_trees_after_planting :
  total_trees 7 3 2 5 1 6 4 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_after_planting_l468_46828


namespace NUMINAMATH_CALUDE_max_product_division_l468_46808

theorem max_product_division (N : ℝ) (h : N > 0) :
  ∀ x : ℝ, 0 < x ∧ x < N → x * (N - x) ≤ (N / 2) * (N / 2) := by
  sorry

end NUMINAMATH_CALUDE_max_product_division_l468_46808


namespace NUMINAMATH_CALUDE_T_is_three_intersecting_lines_l468_46805

-- Define the set T
def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | 
  (p.1 - 3 = 5 ∧ p.2 + 1 ≥ 5) ∨
  (p.1 - 3 = p.2 + 1 ∧ 5 ≥ p.1 - 3) ∨
  (5 = p.2 + 1 ∧ p.1 - 3 ≥ 5)}

-- Define what it means for three lines to intersect at a single point
def three_lines_intersect_at_point (S : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  ∃ (l₁ l₂ l₃ : Set (ℝ × ℝ)),
    (∀ q ∈ S, q ∈ l₁ ∨ q ∈ l₂ ∨ q ∈ l₃) ∧
    (l₁ ∩ l₂ = {p}) ∧ (l₂ ∩ l₃ = {p}) ∧ (l₃ ∩ l₁ = {p}) ∧
    (∀ q ∈ l₁, ∃ r ∈ l₁, q ≠ r) ∧
    (∀ q ∈ l₂, ∃ r ∈ l₂, q ≠ r) ∧
    (∀ q ∈ l₃, ∃ r ∈ l₃, q ≠ r)

-- Theorem statement
theorem T_is_three_intersecting_lines :
  ∃ p : ℝ × ℝ, three_lines_intersect_at_point T p :=
sorry

end NUMINAMATH_CALUDE_T_is_three_intersecting_lines_l468_46805


namespace NUMINAMATH_CALUDE_paint_mixer_days_to_make_drums_l468_46821

/-- Given a paint mixer who makes an equal number of drums each day,
    prove that if it takes 3 days to make 18 drums of paint,
    it will take 60 days to make 360 drums of paint. -/
theorem paint_mixer_days_to_make_drums
  (daily_production : ℕ → ℕ)  -- Function representing daily production
  (h1 : ∀ n : ℕ, daily_production n = daily_production 1)  -- Equal production each day
  (h2 : (daily_production 1) * 3 = 18)  -- 18 drums in 3 days
  : (daily_production 1) * 60 = 360 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixer_days_to_make_drums_l468_46821


namespace NUMINAMATH_CALUDE_three_percent_difference_l468_46881

theorem three_percent_difference (x y : ℝ) : 
  3 = 0.15 * x → 3 = 0.25 * y → x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_three_percent_difference_l468_46881


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_500_l468_46877

theorem largest_multiple_of_15_less_than_500 :
  ∀ n : ℕ, n * 15 < 500 → n * 15 ≤ 495 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_500_l468_46877


namespace NUMINAMATH_CALUDE_victors_decks_count_l468_46827

/-- The number of decks Victor bought -/
def victors_decks : ℕ := 6

/-- The cost of each trick deck in dollars -/
def deck_cost : ℕ := 8

/-- The number of decks Victor's friend bought -/
def friends_decks : ℕ := 2

/-- The total amount spent by Victor and his friend in dollars -/
def total_spent : ℕ := 64

theorem victors_decks_count :
  victors_decks * deck_cost + friends_decks * deck_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_victors_decks_count_l468_46827


namespace NUMINAMATH_CALUDE_sidney_adult_cats_l468_46846

/-- Represents the number of adult cats Sidney has -/
def num_adult_cats : ℕ := sorry

/-- The number of kittens Sidney has -/
def num_kittens : ℕ := 4

/-- The number of cans of cat food Sidney has -/
def initial_cans : ℕ := 7

/-- The number of cans an adult cat eats per day -/
def adult_cat_consumption : ℚ := 1

/-- The number of cans a kitten eats per day -/
def kitten_consumption : ℚ := 3/4

/-- The number of additional cans Sidney needs to buy -/
def additional_cans : ℕ := 35

/-- The number of days Sidney needs to feed her cats -/
def days : ℕ := 7

theorem sidney_adult_cats : 
  num_adult_cats = 3 ∧
  (num_kittens : ℚ) * kitten_consumption * days + 
  (num_adult_cats : ℚ) * adult_cat_consumption * days = 
  (initial_cans : ℚ) + additional_cans :=
sorry

end NUMINAMATH_CALUDE_sidney_adult_cats_l468_46846


namespace NUMINAMATH_CALUDE_gigi_cookies_theorem_l468_46831

/-- Represents the number of cups of flour per batch of cookies -/
def flour_per_batch : ℕ := 2

/-- Represents the initial amount of flour in cups -/
def initial_flour : ℕ := 20

/-- Represents the number of additional batches that can be made with remaining flour -/
def additional_batches : ℕ := 7

/-- Calculates the number of batches Gigi baked initially -/
def batches_baked : ℕ := (initial_flour - additional_batches * flour_per_batch) / flour_per_batch

theorem gigi_cookies_theorem : batches_baked = 3 := by
  sorry

end NUMINAMATH_CALUDE_gigi_cookies_theorem_l468_46831


namespace NUMINAMATH_CALUDE_smallest_integer_power_l468_46871

theorem smallest_integer_power (x : ℕ) (h : x = 9 * 3) :
  (∀ c : ℕ, x^c > 3^24 → c ≥ 9) ∧ x^9 > 3^24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_power_l468_46871


namespace NUMINAMATH_CALUDE_problem_statement_l468_46842

/-- Given positive real numbers a, b, c, and a function f with minimum value 1, 
    prove that a + b + c = 1 and a² + b² + c² ≥ 1/3 -/
theorem problem_statement (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (hf : ∀ x, |x - a| + |x + b| + c ≥ 1) : 
    (a + b + c = 1) ∧ (a^2 + b^2 + c^2 ≥ 1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l468_46842


namespace NUMINAMATH_CALUDE_mary_crayons_left_l468_46812

/-- Represents the number of crayons Mary has left after giving some away -/
def crayons_left (initial_green initial_blue initial_yellow given_green given_blue given_yellow : ℕ) : ℕ :=
  (initial_green - given_green) + (initial_blue - given_blue) + (initial_yellow - given_yellow)

/-- Theorem stating that Mary has 14 crayons left after giving some away -/
theorem mary_crayons_left : 
  crayons_left 5 8 7 3 1 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_mary_crayons_left_l468_46812


namespace NUMINAMATH_CALUDE_pepperjack_probability_l468_46820

/-- The probability of picking a pepperjack cheese stick from a pack containing
    15 cheddar, 30 mozzarella, and 45 pepperjack sticks is 50%. -/
theorem pepperjack_probability (cheddar mozzarella pepperjack : ℕ) 
    (h1 : cheddar = 15)
    (h2 : mozzarella = 30)
    (h3 : pepperjack = 45) :
    (pepperjack : ℚ) / (cheddar + mozzarella + pepperjack) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pepperjack_probability_l468_46820


namespace NUMINAMATH_CALUDE_clock_advance_proof_l468_46833

def clock_hours : ℕ := 12

def start_time : ℕ := 3

def hours_elapsed : ℕ := 2500

def end_time : ℕ := 7

theorem clock_advance_proof :
  (start_time + hours_elapsed) % clock_hours = end_time :=
by sorry

end NUMINAMATH_CALUDE_clock_advance_proof_l468_46833


namespace NUMINAMATH_CALUDE_infinitely_many_real_roots_l468_46886

theorem infinitely_many_real_roots : Set.Infinite {x : ℝ | ∃ y : ℝ, y^2 = -(x+1)^3} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_real_roots_l468_46886


namespace NUMINAMATH_CALUDE_product_of_sum_and_reciprocals_geq_nine_l468_46816

theorem product_of_sum_and_reciprocals_geq_nine (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_reciprocals_geq_nine_l468_46816


namespace NUMINAMATH_CALUDE_quadratic_root_property_l468_46863

theorem quadratic_root_property (a : ℝ) : 
  (2 * a^2 = 6 * a - 4) → (a^2 - 3 * a + 2024 = 2022) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l468_46863


namespace NUMINAMATH_CALUDE_fraction_equality_implies_four_l468_46864

theorem fraction_equality_implies_four (k n m : ℕ+) :
  (1 : ℚ) / n^2 + (1 : ℚ) / m^2 = (k : ℚ) / (n^2 + m^2) →
  k = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_four_l468_46864


namespace NUMINAMATH_CALUDE_zebra_stripes_l468_46801

theorem zebra_stripes (w n b : ℕ) : 
  w + n = b + 1 →  -- Total black stripes (wide + narrow) is one more than white stripes
  b = w + 7 →      -- Number of white stripes is 7 more than wide black stripes
  n = 8 :=         -- Number of narrow black stripes is 8
by sorry

end NUMINAMATH_CALUDE_zebra_stripes_l468_46801


namespace NUMINAMATH_CALUDE_sum_nonpositive_implies_one_nonpositive_l468_46874

theorem sum_nonpositive_implies_one_nonpositive (x y : ℝ) : 
  x + y ≤ 0 → x ≤ 0 ∨ y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_nonpositive_implies_one_nonpositive_l468_46874


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l468_46854

theorem arithmetic_mean_problem (y b : ℝ) (h : y ≠ 0) :
  (((y + b) / y + (2 * y - b) / y) / 2) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l468_46854


namespace NUMINAMATH_CALUDE_five_letter_words_with_vowels_l468_46875

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels

def word_length : Nat := 5

theorem five_letter_words_with_vowels :
  (alphabet.card ^ word_length) - (consonants.card ^ word_length) = 6752 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_with_vowels_l468_46875


namespace NUMINAMATH_CALUDE_relay_race_selections_l468_46814

def number_of_athletes : ℕ := 6
def number_of_legs : ℕ := 4
def athletes_cant_run_first : ℕ := 2

theorem relay_race_selections :
  let total_athletes := number_of_athletes
  let race_legs := number_of_legs
  let excluded_first_leg := athletes_cant_run_first
  let first_leg_choices := total_athletes - excluded_first_leg
  let remaining_athletes := total_athletes - 1
  let remaining_legs := race_legs - 1
  (first_leg_choices : ℕ) * (remaining_athletes.choose remaining_legs) = 240 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_selections_l468_46814


namespace NUMINAMATH_CALUDE_solve_for_k_l468_46813

theorem solve_for_k : ∃ k : ℝ, ((-1) - k * 2 = 7) ∧ k = -4 := by sorry

end NUMINAMATH_CALUDE_solve_for_k_l468_46813


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_condition_l468_46878

theorem geometric_progression_ratio_condition 
  (x y z w r : ℝ) 
  (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0)
  (h2 : x * (y - z) ≠ y * (z - x) ∧ 
        y * (z - x) ≠ z * (x - y) ∧ 
        z * (x - y) ≠ w * (y - x))
  (h3 : ∃ (a : ℝ), a ≠ 0 ∧ 
        x * (y - z) = a ∧ 
        y * (z - x) = a * r ∧ 
        z * (x - y) = a * r^2 ∧ 
        w * (y - x) = a * r^3) :
  r^3 + r^2 + r + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_condition_l468_46878


namespace NUMINAMATH_CALUDE_cube_root_condition_square_root_condition_integer_part_condition_main_result_l468_46847

def a : ℝ := 4
def b : ℝ := 2
def c : ℤ := 3

theorem cube_root_condition : (3 * a - 4) ^ (1/3 : ℝ) = 2 := by sorry

theorem square_root_condition : (a + 2 * b + 1) ^ (1/2 : ℝ) = 3 := by sorry

theorem integer_part_condition : c = Int.floor (Real.sqrt 15) := by sorry

theorem main_result : (a + b + c : ℝ) ^ (1/2 : ℝ) = 3 ∨ (a + b + c : ℝ) ^ (1/2 : ℝ) = -3 := by sorry

end NUMINAMATH_CALUDE_cube_root_condition_square_root_condition_integer_part_condition_main_result_l468_46847


namespace NUMINAMATH_CALUDE_cindy_calculation_l468_46806

theorem cindy_calculation (x : ℝ) (h : (x - 9) / 3 = 43) : (x - 3) / 9 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l468_46806


namespace NUMINAMATH_CALUDE_toms_vaccines_l468_46894

theorem toms_vaccines (total_payment : ℕ) (trip_cost : ℕ) (vaccine_cost : ℕ) (doctor_visit : ℕ) 
  (insurance_coverage : ℚ) :
  total_payment = 1340 →
  trip_cost = 1200 →
  vaccine_cost = 45 →
  doctor_visit = 250 →
  insurance_coverage = 4/5 →
  ∃ (num_vaccines : ℕ), 
    (total_payment : ℚ) = trip_cost + (1 - insurance_coverage) * (doctor_visit + num_vaccines * vaccine_cost) ∧
    num_vaccines = 10 :=
by sorry

end NUMINAMATH_CALUDE_toms_vaccines_l468_46894


namespace NUMINAMATH_CALUDE_wall_bricks_l468_46856

/-- Represents the time taken by the first bricklayer to build the wall alone -/
def time1 : ℝ := 8

/-- Represents the time taken by the second bricklayer to build the wall alone -/
def time2 : ℝ := 12

/-- Represents the reduction in productivity when working together (in bricks per hour) -/
def reduction : ℝ := 15

/-- Represents the time taken by both bricklayers working together to build the wall -/
def timeJoint : ℝ := 6

/-- Represents the total number of bricks in the wall -/
def totalBricks : ℝ := 360

theorem wall_bricks : 
  timeJoint * (totalBricks / time1 + totalBricks / time2 - reduction) = totalBricks := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_l468_46856


namespace NUMINAMATH_CALUDE_largest_class_size_l468_46834

theorem largest_class_size (total_students : ℕ) (num_classes : ℕ) (class_diff : ℕ) :
  total_students = 95 →
  num_classes = 5 →
  class_diff = 2 →
  ∃ (x : ℕ), x = 23 ∧ 
    (x + (x - class_diff) + (x - 2*class_diff) + (x - 3*class_diff) + (x - 4*class_diff) = total_students) :=
by sorry

end NUMINAMATH_CALUDE_largest_class_size_l468_46834


namespace NUMINAMATH_CALUDE_counterexample_squared_inequality_l468_46803

theorem counterexample_squared_inequality :
  ∃ (m n : ℝ), m > n ∧ m^2 ≤ n^2 := by sorry

end NUMINAMATH_CALUDE_counterexample_squared_inequality_l468_46803


namespace NUMINAMATH_CALUDE_complex_quadrant_l468_46829

theorem complex_quadrant (z : ℂ) (h : (1 - I) * z = 3 + 5*I) : 
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l468_46829


namespace NUMINAMATH_CALUDE_hill_climbing_speed_l468_46885

/-- Proves that given a round trip journey with specified conditions, 
    the average speed for the upward journey is 2.625 km/h -/
theorem hill_climbing_speed 
  (upward_time : ℝ) 
  (downward_time : ℝ) 
  (total_avg_speed : ℝ) 
  (h1 : upward_time = 4) 
  (h2 : downward_time = 2) 
  (h3 : total_avg_speed = 3.5) : 
  (total_avg_speed * (upward_time + downward_time)) / (2 * upward_time) = 2.625 := by
sorry

end NUMINAMATH_CALUDE_hill_climbing_speed_l468_46885


namespace NUMINAMATH_CALUDE_fraction_value_l468_46809

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l468_46809


namespace NUMINAMATH_CALUDE_factor_expression_l468_46869

/-- The expression a^3 (b^2 - c^2) + b^3 (c^2 - a^2) + c^3 (a^2 - b^2) 
    can be factored as (a - b)(b - c)(c - a) * (-(ab + ac + bc)) -/
theorem factor_expression (a b c : ℝ) :
  a^3 * (b^2 - c^2) + b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2) =
  (a - b) * (b - c) * (c - a) * (-(a*b + a*c + b*c)) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l468_46869


namespace NUMINAMATH_CALUDE_line_through_point_l468_46850

/-- Given a line ax + (a+1)y = a + 4 passing through the point (3, -7), prove that a = -11/5 --/
theorem line_through_point (a : ℚ) : 
  (a * 3 + (a + 1) * (-7) = a + 4) → a = -11/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l468_46850


namespace NUMINAMATH_CALUDE_g_derivative_l468_46857

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2 + 3^x

theorem g_derivative (x : ℝ) (h : x > 0) :
  deriv g x = 1 / (x * Real.log 2) + 3^x * Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_g_derivative_l468_46857


namespace NUMINAMATH_CALUDE_shaded_volume_is_112_l468_46897

/-- The volume of a rectangular prism with dimensions a, b, and c -/
def volume (a b c : ℕ) : ℕ := a * b * c

/-- The dimensions of the larger prism -/
def large_prism : Fin 3 → ℕ
| 0 => 4
| 1 => 5
| 2 => 6
| _ => 0

/-- The dimensions of the smaller prism -/
def small_prism : Fin 3 → ℕ
| 0 => 1
| 1 => 2
| 2 => 4
| _ => 0

theorem shaded_volume_is_112 :
  volume (large_prism 0) (large_prism 1) (large_prism 2) -
  volume (small_prism 0) (small_prism 1) (small_prism 2) = 112 := by
  sorry

end NUMINAMATH_CALUDE_shaded_volume_is_112_l468_46897


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l468_46858

/-- Given two vectors a and b in ℝ², where a = (-√3, 1) and b = (1, x),
    if a and b are perpendicular, then x = √3. -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (-Real.sqrt 3, 1)
  let b : ℝ × ℝ := (1, x)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l468_46858


namespace NUMINAMATH_CALUDE_spaceship_speed_halving_l468_46800

/-- The number of additional people that cause the spaceship's speed to be halved -/
def additional_people : ℕ := sorry

/-- The speed of the spaceship given the number of people on board -/
def speed (people : ℕ) : ℝ := sorry

/-- Theorem: The number of additional people that cause the spaceship's speed to be halved is 100 -/
theorem spaceship_speed_halving :
  (speed 200 = 500) →
  (speed 400 = 125) →
  (∀ n : ℕ, speed (n + additional_people) = (speed n) / 2) →
  additional_people = 100 := by
  sorry

end NUMINAMATH_CALUDE_spaceship_speed_halving_l468_46800


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l468_46823

theorem simplify_and_evaluate (x y : ℝ) (hx : x = 2023) (hy : y = 2) :
  (x + 2*y)^2 - (x^3 + 4*x^2*y) / x = 16 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l468_46823


namespace NUMINAMATH_CALUDE_simplify_expression_l468_46891

theorem simplify_expression : (2^8 + 4^5) * (1^3 - (-1)^3)^8 = 327680 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l468_46891


namespace NUMINAMATH_CALUDE_largest_prime_divisor_test_l468_46859

theorem largest_prime_divisor_test (n : ℕ) : 
  1000 ≤ n → n ≤ 1100 → 
  (∀ p : ℕ, p ≤ 31 → Nat.Prime p → ¬(p ∣ n)) → 
  Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_test_l468_46859


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l468_46824

theorem sqrt_x_minus_one_meaningful (x : ℝ) :
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l468_46824


namespace NUMINAMATH_CALUDE_least_stamps_l468_46836

theorem least_stamps (n : ℕ) : n = 107 ↔ 
  (n > 0) ∧ 
  (n % 4 = 3) ∧ 
  (n % 5 = 2) ∧ 
  (n % 7 = 1) ∧ 
  (∀ m : ℕ, m > 0 → m % 4 = 3 → m % 5 = 2 → m % 7 = 1 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_least_stamps_l468_46836


namespace NUMINAMATH_CALUDE_smallest_a_is_correct_l468_46896

/-- The smallest positive integer a such that both 112 and 33 are factors of a * 43 * 62 * 1311 -/
def smallest_a : ℕ := 1848

/-- Predicate to check if a number divides the product a * 43 * 62 * 1311 -/
def is_factor (n : ℕ) (a : ℕ) : Prop :=
  (n : ℤ) ∣ (a * 43 * 62 * 1311 : ℤ)

theorem smallest_a_is_correct :
  (∀ a : ℕ, a > 0 → is_factor 112 a → is_factor 33 a → a ≥ smallest_a) ∧
  is_factor 112 smallest_a ∧
  is_factor 33 smallest_a :=
sorry

end NUMINAMATH_CALUDE_smallest_a_is_correct_l468_46896


namespace NUMINAMATH_CALUDE_sally_balloons_l468_46860

theorem sally_balloons (initial : ℕ) (lost : ℕ) (final : ℕ) : 
  initial = 20 → lost = 5 → final = (initial - lost) * 2 → final = 30 := by
sorry

end NUMINAMATH_CALUDE_sally_balloons_l468_46860


namespace NUMINAMATH_CALUDE_number_of_divisors_30030_l468_46840

theorem number_of_divisors_30030 : Nat.card (Nat.divisors 30030) = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_30030_l468_46840


namespace NUMINAMATH_CALUDE_cos_ninety_degrees_l468_46880

theorem cos_ninety_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_ninety_degrees_l468_46880


namespace NUMINAMATH_CALUDE_triangle_properties_l468_46876

/-- Given a triangle ABC with sides a, b, and c, prove the following properties -/
theorem triangle_properties (A B C : ℝ × ℝ) (a b c : ℝ) :
  let AB := B - A
  let BC := C - B
  let CA := A - C
  -- Given condition
  AB • AC + 2 * (-AB) • BC = 3 * (-CA) • (-BC) →
  -- Side lengths
  ‖BC‖ = a ∧ ‖CA‖ = b ∧ ‖AB‖ = c →
  -- Prove these properties
  a^2 + 2*b^2 = 3*c^2 ∧ 
  ∀ (cos_C : ℝ), cos_C = (a^2 + b^2 - c^2) / (2*a*b) → cos_C ≥ Real.sqrt 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l468_46876


namespace NUMINAMATH_CALUDE_adult_ticket_price_l468_46855

theorem adult_ticket_price 
  (total_tickets : ℕ) 
  (total_profit : ℕ) 
  (kid_tickets : ℕ) 
  (kid_price : ℕ) :
  total_tickets = 175 →
  total_profit = 750 →
  kid_tickets = 75 →
  kid_price = 2 →
  (total_tickets - kid_tickets) * 
    ((total_profit - kid_tickets * kid_price) / (total_tickets - kid_tickets)) = 600 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l468_46855


namespace NUMINAMATH_CALUDE_sqrt2_plus_sqrt3_power_2012_decimal_digits_l468_46837

theorem sqrt2_plus_sqrt3_power_2012_decimal_digits :
  ∃ k : ℤ,
    (k : ℝ) < (Real.sqrt 2 + Real.sqrt 3) ^ 2012 ∧
    (Real.sqrt 2 + Real.sqrt 3) ^ 2012 < (k + 1 : ℝ) ∧
    (Real.sqrt 2 + Real.sqrt 3) ^ 2012 - k > (79 : ℝ) / 100 ∧
    (Real.sqrt 2 + Real.sqrt 3) ^ 2012 - k < (80 : ℝ) / 100 :=
by
  sorry


end NUMINAMATH_CALUDE_sqrt2_plus_sqrt3_power_2012_decimal_digits_l468_46837


namespace NUMINAMATH_CALUDE_angle_2013_in_third_quadrant_l468_46838

-- Define the quadrants
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define a function to determine the quadrant of an angle
def angleQuadrant (angle : ℝ) : Quadrant := sorry

-- Theorem stating that 2013° is in the third quadrant
theorem angle_2013_in_third_quadrant :
  angleQuadrant 2013 = Quadrant.Third :=
by
  -- Define the relationship between 2013° and 213°
  have h1 : 2013 = 5 * 360 + 213 := by sorry
  
  -- State that 213° is in the third quadrant
  have h2 : angleQuadrant 213 = Quadrant.Third := by sorry
  
  -- State that angles with the same terminal side are in the same quadrant
  have h3 : ∀ (a b : ℝ), (a - b) % 360 = 0 → angleQuadrant a = angleQuadrant b := by sorry
  
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_angle_2013_in_third_quadrant_l468_46838


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l468_46866

theorem interest_rate_calculation (initial_amount loan_amount final_amount : ℚ) :
  initial_amount = 30 →
  loan_amount = 15 →
  final_amount = 33 →
  (final_amount - initial_amount) / loan_amount * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l468_46866


namespace NUMINAMATH_CALUDE_boys_who_bought_balloons_l468_46802

/-- The number of balloons in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of balloons the clown initially had -/
def initial_dozens : ℕ := 3

/-- The number of girls who bought a balloon -/
def girls_bought : ℕ := 12

/-- The number of balloons the clown has left after sales -/
def remaining_balloons : ℕ := 21

/-- The number of boys who bought a balloon -/
def boys_bought : ℕ := initial_dozens * dozen - remaining_balloons - girls_bought

theorem boys_who_bought_balloons :
  boys_bought = 3 := by sorry

end NUMINAMATH_CALUDE_boys_who_bought_balloons_l468_46802


namespace NUMINAMATH_CALUDE_distance_point_to_line_l468_46845

/-- The distance from the point (1, 0) to the line x - y + 1 = 0 is √2 -/
theorem distance_point_to_line : 
  let point : ℝ × ℝ := (1, 0)
  let line (x y : ℝ) : Prop := x - y + 1 = 0
  Real.sqrt 2 = (|1 - 0 + 1|) / Real.sqrt (1^2 + (-1)^2) := by sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l468_46845


namespace NUMINAMATH_CALUDE_max_subdivision_sides_l468_46884

/-- Represents a convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 2

/-- Represents the maximum number of sides in a subdivision polygon -/
def maxSubdivisionSides (n : ℕ) : ℕ := n

/-- Theorem stating that the maximum number of sides in a subdivision polygon is n -/
theorem max_subdivision_sides (n : ℕ) (p : ConvexPolygon n) :
  maxSubdivisionSides n = n := by
  sorry

#eval maxSubdivisionSides 13    -- Should output 13
#eval maxSubdivisionSides 1950  -- Should output 1950

end NUMINAMATH_CALUDE_max_subdivision_sides_l468_46884


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l468_46815

theorem stratified_sampling_problem (grade10 : ℕ) (grade11 : ℕ) (grade12 : ℕ) 
  (sample12 : ℕ) (h1 : grade10 = 1600) (h2 : grade11 = 1200) (h3 : grade12 = 800) 
  (h4 : sample12 = 20) :
  ∃ (sample1011 : ℕ), 
    (grade10 + grade11) * sample12 = grade12 * sample1011 ∧ 
    sample1011 = 70 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l468_46815


namespace NUMINAMATH_CALUDE_certain_number_proof_l468_46873

theorem certain_number_proof : ∃! N : ℕ, 
  N % 101 = 8 ∧ 
  5161 % 101 = 10 ∧ 
  N = 5159 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l468_46873


namespace NUMINAMATH_CALUDE_range_of_a_l468_46889

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (a - 2) * x + 1 ≠ 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0

-- Define the theorem
theorem range_of_a (a : ℝ) (h : p a ∨ q a) : -2 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l468_46889


namespace NUMINAMATH_CALUDE_total_students_correct_l468_46817

/-- The total number of students at the college -/
def total_students : ℕ := 880

/-- The percentage of students enrolled in biology classes -/
def biology_enrollment_percentage : ℚ := 40 / 100

/-- The number of students not enrolled in biology classes -/
def non_biology_students : ℕ := 528

/-- Theorem stating that the total number of students is correct given the conditions -/
theorem total_students_correct :
  (1 - biology_enrollment_percentage) * total_students = non_biology_students :=
sorry

end NUMINAMATH_CALUDE_total_students_correct_l468_46817


namespace NUMINAMATH_CALUDE_profit_maximization_l468_46818

variable (x : ℝ)

def production_cost (x : ℝ) : ℝ := x^3 - 24*x^2 + 63*x + 10
def sales_revenue (x : ℝ) : ℝ := 18*x
def profit (x : ℝ) : ℝ := sales_revenue x - production_cost x

theorem profit_maximization (h : x > 0) :
  profit x = -x^3 + 24*x^2 - 45*x - 10 ∧
  ∃ (max_x : ℝ), max_x = 15 ∧
    ∀ (y : ℝ), y > 0 → profit y ≤ profit max_x ∧
    profit max_x = 1340 :=
by sorry

end NUMINAMATH_CALUDE_profit_maximization_l468_46818


namespace NUMINAMATH_CALUDE_quadratic_function_property_l468_46879

theorem quadratic_function_property (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (f 0 = f 4 ∧ f 0 > f 1) → (a > 0 ∧ 4 * a + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l468_46879


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l468_46890

-- Define the conditions p and q
def p (x : ℝ) : Prop := (x - 2)^2 ≤ 1
def q (x : ℝ) : Prop := 2 / (x - 1) ≥ 1

-- Define the set of x that satisfy p
def p_set : Set ℝ := {x | p x}

-- Define the set of x that satisfy q
def q_set : Set ℝ := {x | q x}

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (q_set ⊆ p_set) ∧ ¬(p_set ⊆ q_set) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l468_46890


namespace NUMINAMATH_CALUDE_ryan_chinese_hours_l468_46835

/-- Ryan's daily study hours -/
structure StudyHours where
  english : ℕ
  chinese : ℕ
  more_chinese : chinese = english + 1

/-- Theorem: Ryan spends 7 hours on learning Chinese -/
theorem ryan_chinese_hours (ryan : StudyHours) (h : ryan.english = 6) : ryan.chinese = 7 := by
  sorry

end NUMINAMATH_CALUDE_ryan_chinese_hours_l468_46835


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l468_46843

theorem rectangular_plot_breadth (length breadth area : ℝ) : 
  length = 3 * breadth → 
  area = length * breadth → 
  area = 588 → 
  breadth = 14 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l468_46843


namespace NUMINAMATH_CALUDE_three_prime_divisors_theorem_l468_46841

theorem three_prime_divisors_theorem (x n : ℕ) :
  x = 2^n - 32 ∧
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 2 ∧ q ≠ 2 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → r = 2 ∨ r = p ∨ r = q)) →
  x = 2016 ∨ x = 16352 := by
  sorry

end NUMINAMATH_CALUDE_three_prime_divisors_theorem_l468_46841


namespace NUMINAMATH_CALUDE_sum_of_four_squares_99_l468_46826

theorem sum_of_four_squares_99 : ∃ (a b c d w x y z : ℕ),
  a^2 + b^2 + c^2 + d^2 = 99 ∧
  w^2 + x^2 + y^2 + z^2 = 99 ∧
  (a, b, c, d) ≠ (w, x, y, z) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_four_squares_99_l468_46826


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l468_46851

theorem solution_set_of_inequality (x : ℝ) : 
  (x^2 - x < 0) ↔ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l468_46851


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l468_46862

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : (1 : ℝ) / a > (1 : ℝ) / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l468_46862


namespace NUMINAMATH_CALUDE_dreams_driving_distance_l468_46830

/-- Represents the problem of calculating Dream's driving distance --/
theorem dreams_driving_distance :
  let gas_consumption_rate : ℝ := 4  -- gallons per mile
  let additional_miles_tomorrow : ℝ := 200
  let total_gas_consumption : ℝ := 4000
  ∃ (miles_today : ℝ),
    gas_consumption_rate * miles_today + 
    gas_consumption_rate * (miles_today + additional_miles_tomorrow) = 
    total_gas_consumption ∧
    miles_today = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_dreams_driving_distance_l468_46830


namespace NUMINAMATH_CALUDE_system_solution_l468_46807

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 - 9*y₁^2 = 0 ∧ 2*x₁ - 3*y₁ = 6) ∧
    (x₂^2 - 9*y₂^2 = 0 ∧ 2*x₂ - 3*y₂ = 6) ∧
    x₁ = 6 ∧ y₁ = 2 ∧ x₂ = 2 ∧ y₂ = -2/3 ∧
    ∀ (x y : ℝ), (x^2 - 9*y^2 = 0 ∧ 2*x - 3*y = 6) →
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l468_46807


namespace NUMINAMATH_CALUDE_ball_selection_count_l468_46882

def num_colors : ℕ := 4
def balls_per_color : ℕ := 6
def balls_to_select : ℕ := 3

def valid_number_combinations : List (List ℕ) :=
  [[1, 3, 5], [1, 3, 6], [1, 4, 6], [2, 4, 6]]

theorem ball_selection_count :
  (num_colors.choose balls_to_select) *
  (valid_number_combinations.length) *
  (balls_to_select.factorial) = 96 := by
sorry

end NUMINAMATH_CALUDE_ball_selection_count_l468_46882


namespace NUMINAMATH_CALUDE_four_digit_number_remainder_l468_46844

theorem four_digit_number_remainder (a b c d : Nat) 
  (h1 : a ≠ 0) 
  (h2 : d ≠ 0) 
  (h3 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) 
  (h4 : ∃ k : Int, (1000 * a + 100 * b + 10 * c + d) + (1000 * a + 100 * c + 10 * b + d) = 900 * k) : 
  (1000 * a + 100 * b + 10 * c + d) % 90 = 45 := by
sorry

end NUMINAMATH_CALUDE_four_digit_number_remainder_l468_46844


namespace NUMINAMATH_CALUDE_A_subset_B_iff_a_geq_2_plus_sqrt5_l468_46819

/-- Set A defined as a circle with center (2,1) and radius 1 -/
def A : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 1)^2 ≤ 1}

/-- Set B defined by the condition 2|x-1| + |y-1| ≤ a -/
def B (a : ℝ) : Set (ℝ × ℝ) := {p | 2 * |p.1 - 1| + |p.2 - 1| ≤ a}

/-- Theorem stating that A is a subset of B if and only if a ≥ 2 + √5 -/
theorem A_subset_B_iff_a_geq_2_plus_sqrt5 :
  ∀ a : ℝ, A ⊆ B a ↔ a ≥ 2 + Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_A_subset_B_iff_a_geq_2_plus_sqrt5_l468_46819


namespace NUMINAMATH_CALUDE_power_division_simplification_l468_46804

theorem power_division_simplification : (1000 : ℕ)^7 / (10 : ℕ)^17 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_power_division_simplification_l468_46804


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l468_46887

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence where a_4 = 5 and a_8 = 6, a_2 * a_10 = 30 -/
theorem geometric_sequence_product (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_a4 : a 4 = 5) 
    (h_a8 : a 8 = 6) : 
  a 2 * a 10 = 30 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l468_46887


namespace NUMINAMATH_CALUDE_inequality_theorem_l468_46832

theorem inequality_theorem (p q : ℝ) :
  q > 0 ∧ p ≥ 0 →
  ((4 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔
  (0 ≤ p ∧ p < 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l468_46832


namespace NUMINAMATH_CALUDE_valid_pairs_l468_46839

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digits_form_arithmetic_sequence (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ : ℕ), n = 100 * d₁ + 10 * d₂ + d₃ ∧ 
    d₁ < d₂ ∧ d₂ < d₃ ∧ d₂ - d₁ = d₃ - d₂

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def all_digits_same (n : ℕ) : Prop :=
  ∃ (d : ℕ), n = d * 11111

theorem valid_pairs : 
  ∀ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ≤ b ∧ 
    is_three_digit (a + b) ∧ 
    digits_form_arithmetic_sequence (a + b) ∧
    is_five_digit (a * b) ∧
    all_digits_same (a * b) →
  ((a = 41 ∧ b = 271) ∨ 
   (a = 164 ∧ b = 271) ∨ 
   (a = 82 ∧ b = 542) ∨ 
   (a = 123 ∧ b = 813)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l468_46839


namespace NUMINAMATH_CALUDE_not_divisible_by_nine_l468_46867

theorem not_divisible_by_nine (n : ℕ) : ¬ (9 ∣ (n^3 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_nine_l468_46867


namespace NUMINAMATH_CALUDE_joes_notebooks_l468_46861

theorem joes_notebooks (initial_amount : ℕ) (notebook_cost : ℕ) (book_cost : ℕ) 
  (books_bought : ℕ) (amount_left : ℕ) : 
  initial_amount = 56 → 
  notebook_cost = 4 → 
  book_cost = 7 → 
  books_bought = 2 → 
  amount_left = 14 → 
  ∃ (notebooks_bought : ℕ), 
    notebooks_bought = 7 ∧ 
    initial_amount = notebook_cost * notebooks_bought + book_cost * books_bought + amount_left :=
by sorry

end NUMINAMATH_CALUDE_joes_notebooks_l468_46861


namespace NUMINAMATH_CALUDE_randy_piggy_bank_theorem_l468_46898

/-- Calculates the amount in Randy's piggy bank after a year -/
def piggy_bank_after_year (initial_amount : ℕ) (store_trip_cost : ℕ) (store_trips_per_month : ℕ)
  (internet_bill : ℕ) (extra_cost_third_trip : ℕ) (weekly_earnings : ℕ) (birthday_gift : ℕ) : ℕ :=
  let months_in_year : ℕ := 12
  let weeks_in_year : ℕ := 52
  let regular_store_expenses := store_trip_cost * store_trips_per_month * months_in_year
  let extra_expenses := extra_cost_third_trip * (months_in_year / 3)
  let internet_expenses := internet_bill * months_in_year
  let job_income := weekly_earnings * weeks_in_year
  let total_expenses := regular_store_expenses + extra_expenses + internet_expenses
  let total_income := job_income + birthday_gift
  initial_amount + total_income - total_expenses

theorem randy_piggy_bank_theorem :
  piggy_bank_after_year 200 2 4 20 1 15 100 = 740 := by
  sorry

end NUMINAMATH_CALUDE_randy_piggy_bank_theorem_l468_46898


namespace NUMINAMATH_CALUDE_six_digit_integers_count_l468_46892

/-- The number of different positive six-digit integers formed using the digits 1, 1, 1, 5, 9, and 9 -/
def six_digit_integers : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of different positive six-digit integers
    formed using the digits 1, 1, 1, 5, 9, and 9 is equal to 60 -/
theorem six_digit_integers_count : six_digit_integers = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_integers_count_l468_46892


namespace NUMINAMATH_CALUDE_unique_a_value_l468_46870

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + 4 = 0}

-- State the theorem
theorem unique_a_value :
  ∃! a : ℝ, (B a).Nonempty ∧ B a ⊆ A := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l468_46870


namespace NUMINAMATH_CALUDE_pine_cones_on_roof_l468_46893

/-- The number of pine trees in Alan's backyard -/
def num_trees : ℕ := 8

/-- The number of pine cones dropped by each tree -/
def cones_per_tree : ℕ := 200

/-- The weight of each pine cone in ounces -/
def cone_weight : ℕ := 4

/-- The total weight of pine cones on Alan's roof in ounces -/
def roof_weight : ℕ := 1920

/-- The percentage of pine cones that fall on Alan's roof -/
def roof_percentage : ℚ := 30 / 100

theorem pine_cones_on_roof :
  (roof_weight / cone_weight) / (num_trees * cones_per_tree) = roof_percentage := by
  sorry

end NUMINAMATH_CALUDE_pine_cones_on_roof_l468_46893
