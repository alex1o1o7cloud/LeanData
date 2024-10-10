import Mathlib

namespace speed_conversion_l1640_164024

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- Given speed in meters per second -/
def given_speed_mps : ℝ := 23.3352

/-- Calculated speed in kilometers per hour -/
def calculated_speed_kmph : ℝ := 84.00672

theorem speed_conversion : given_speed_mps * mps_to_kmph = calculated_speed_kmph := by
  sorry

end speed_conversion_l1640_164024


namespace four_vertex_cycle_exists_l1640_164092

/-- A graph with n ≥ 4 vertices where each vertex has degree between 1 and n-2 (inclusive) --/
structure CompanyGraph (n : ℕ) where
  (vertices : Finset (Fin n))
  (edges : Finset (Fin n × Fin n))
  (h1 : n ≥ 4)
  (h2 : ∀ v ∈ vertices, 1 ≤ (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card)
  (h3 : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≤ n - 2)
  (h4 : ∀ e ∈ edges, e.1 ∈ vertices ∧ e.2 ∈ vertices)
  (h5 : ∀ e ∈ edges, (e.2, e.1) ∈ edges)  -- Knowledge is mutual

/-- A cycle of four vertices in the graph --/
structure FourVertexCycle (n : ℕ) (G : CompanyGraph n) where
  (v1 v2 v3 v4 : Fin n)
  (h1 : v1 ∈ G.vertices ∧ v2 ∈ G.vertices ∧ v3 ∈ G.vertices ∧ v4 ∈ G.vertices)
  (h2 : (v1, v2) ∈ G.edges ∧ (v2, v3) ∈ G.edges ∧ (v3, v4) ∈ G.edges ∧ (v4, v1) ∈ G.edges)
  (h3 : (v1, v3) ∉ G.edges ∧ (v2, v4) ∉ G.edges)

/-- The main theorem --/
theorem four_vertex_cycle_exists (n : ℕ) (G : CompanyGraph n) : 
  ∃ c : FourVertexCycle n G, True :=
sorry

end four_vertex_cycle_exists_l1640_164092


namespace graces_nickels_l1640_164054

theorem graces_nickels (dimes : ℕ) (nickels : ℕ) : 
  dimes = 10 →
  dimes * 10 + nickels * 5 = 150 →
  nickels = 10 := by
sorry

end graces_nickels_l1640_164054


namespace simplify_fraction_product_l1640_164089

theorem simplify_fraction_product : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by sorry

end simplify_fraction_product_l1640_164089


namespace smallest_multiple_of_5_and_711_l1640_164084

theorem smallest_multiple_of_5_and_711 : 
  ∀ n : ℕ, n > 0 ∧ 5 ∣ n ∧ 711 ∣ n → n ≥ 3555 := by
  sorry

end smallest_multiple_of_5_and_711_l1640_164084


namespace pure_gala_trees_l1640_164016

/-- Represents the apple orchard problem --/
def apple_orchard_problem (T F G : ℕ) : Prop :=
  (F : ℚ) + 0.1 * T = 238 ∧
  F = (3/4 : ℚ) * T ∧
  G = T - F

/-- Theorem stating the number of pure Gala trees --/
theorem pure_gala_trees : ∃ T F G : ℕ, 
  apple_orchard_problem T F G ∧ G = 70 := by
  sorry

end pure_gala_trees_l1640_164016


namespace intersection_with_complement_l1640_164003

def A : Set ℝ := {1, 2, 3, 4, 5, 6}
def B : Set ℝ := {x | 2 < x ∧ x < 5}

theorem intersection_with_complement :
  A ∩ (Set.univ \ B) = {1, 2, 5, 6} := by sorry

end intersection_with_complement_l1640_164003


namespace books_read_second_week_l1640_164087

theorem books_read_second_week :
  ∀ (total_books : ℕ) 
    (first_week_books : ℕ) 
    (later_weeks_books : ℕ) 
    (total_weeks : ℕ),
  total_books = 54 →
  first_week_books = 6 →
  later_weeks_books = 9 →
  total_weeks = 7 →
  total_books = first_week_books + (total_weeks - 2) * later_weeks_books + 3 :=
by
  sorry

end books_read_second_week_l1640_164087


namespace problem_solution_l1640_164005

theorem problem_solution (a : ℝ) (f g : ℝ → ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, f x = x^2 + 9)
  (h3 : ∀ x, g x = x^2 - 3)
  (h4 : f (g a) = 9) :
  a = Real.sqrt 3 := by
sorry

end problem_solution_l1640_164005


namespace wedge_volume_specific_case_l1640_164081

/-- Represents a cylindrical log with a wedge cut out. -/
structure WedgedLog where
  diameter : ℝ
  firstCutAngle : ℝ
  secondCutAngle : ℝ
  intersectionPoint : ℕ

/-- Calculates the volume of the wedge cut from the log. -/
def wedgeVolume (log : WedgedLog) : ℝ :=
  sorry

/-- Theorem stating the volume of the wedge under specific conditions. -/
theorem wedge_volume_specific_case :
  let log : WedgedLog := {
    diameter := 16,
    firstCutAngle := 90,  -- perpendicular cut
    secondCutAngle := 60,
    intersectionPoint := 1
  }
  wedgeVolume log = 512 * Real.pi := by sorry

end wedge_volume_specific_case_l1640_164081


namespace isosceles_right_triangle_angles_l1640_164039

theorem isosceles_right_triangle_angles (α : ℝ) :
  α > 0 ∧ α < 90 →
  (α + α + 90 = 180) →
  α = 45 := by
sorry

end isosceles_right_triangle_angles_l1640_164039


namespace dolphin_count_dolphin_count_proof_l1640_164078

theorem dolphin_count : ℕ → Prop :=
  fun total_dolphins =>
    let fully_trained := total_dolphins / 4
    let remaining := total_dolphins - fully_trained
    let in_training := (2 * remaining) / 3
    let untrained := remaining - in_training
    (fully_trained = total_dolphins / 4) ∧
    (in_training = (2 * remaining) / 3) ∧
    (untrained = 5) →
    total_dolphins = 20

-- The proof goes here
theorem dolphin_count_proof : dolphin_count 20 := by
  sorry

end dolphin_count_dolphin_count_proof_l1640_164078


namespace g_one_half_l1640_164086

/-- Given a function g : ℝ → ℝ satisfying certain properties, prove that g(1/2) = 1/2 -/
theorem g_one_half (g : ℝ → ℝ) 
  (h1 : g 2 = 2)
  (h2 : ∀ x y : ℝ, g (x * y + g x) = y * g x + g x) : 
  g (1/2) = 1/2 := by
  sorry

end g_one_half_l1640_164086


namespace binomial_gcd_divisibility_l1640_164096

theorem binomial_gcd_divisibility (n k : ℕ+) :
  ∃ m : ℕ, m * n = Nat.choose n.val k.val * Nat.gcd n.val k.val := by
  sorry

end binomial_gcd_divisibility_l1640_164096


namespace kitchen_broken_fraction_l1640_164042

theorem kitchen_broken_fraction :
  let foyer_broken : ℕ := 10
  let kitchen_total : ℕ := 35
  let total_not_broken : ℕ := 34
  let foyer_total : ℕ := foyer_broken * 3
  let total_bulbs : ℕ := foyer_total + kitchen_total
  let total_broken : ℕ := total_bulbs - total_not_broken
  let kitchen_broken : ℕ := total_broken - foyer_broken
  (kitchen_broken : ℚ) / kitchen_total = 3 / 5 :=
by sorry

end kitchen_broken_fraction_l1640_164042


namespace expression_evaluation_l1640_164070

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 11) + 2 = -x^4 + 3*x^3 - 5*x^2 + 11*x + 2 :=
by sorry

end expression_evaluation_l1640_164070


namespace percentage_of_b_grades_l1640_164048

def scores : List Nat := [91, 68, 59, 99, 82, 88, 86, 79, 72, 60, 87, 85, 83, 76, 81, 93, 65, 89, 78, 74]

def is_grade_b (score : Nat) : Bool :=
  83 ≤ score ∧ score ≤ 92

def count_grade_b (scores : List Nat) : Nat :=
  scores.filter is_grade_b |>.length

theorem percentage_of_b_grades (scores : List Nat) :
  scores.length = 20 →
  (count_grade_b scores : Rat) / scores.length * 100 = 30 := by
  sorry

end percentage_of_b_grades_l1640_164048


namespace min_value_a_l1640_164051

theorem min_value_a : 
  (∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 2 ∧ ∃ a : ℝ, a * 3^x ≥ x - 1) → 
  (∃ a_min : ℝ, a_min = -6 ∧ ∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 2 ∧ a * 3^x ≥ x - 1) → a ≥ a_min) :=
by sorry

end min_value_a_l1640_164051


namespace can_form_triangle_l1640_164093

theorem can_form_triangle (a b c : ℝ) (ha : a = 6) (hb : b = 8) (hc : c = 12) :
  a + b > c ∧ b + c > a ∧ c + a > b := by
  sorry

end can_form_triangle_l1640_164093


namespace square_sum_geq_double_product_l1640_164045

theorem square_sum_geq_double_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by
  sorry

end square_sum_geq_double_product_l1640_164045


namespace cube_root_sum_equals_five_l1640_164014

theorem cube_root_sum_equals_five :
  (Real.rpow (25 + 10 * Real.sqrt 5) (1/3 : ℝ)) + (Real.rpow (25 - 10 * Real.sqrt 5) (1/3 : ℝ)) = 5 :=
by sorry

end cube_root_sum_equals_five_l1640_164014


namespace halloween_candy_count_l1640_164064

/-- Represents the number of candies each person has -/
structure CandyCount where
  bob : Nat
  mary : Nat
  john : Nat
  sue : Nat
  sam : Nat

/-- The total number of candies for all friends -/
def totalCandies (cc : CandyCount) : Nat :=
  cc.bob + cc.mary + cc.john + cc.sue + cc.sam

/-- Theorem stating that the total number of candies is 50 -/
theorem halloween_candy_count :
  ∃ (cc : CandyCount),
    cc.bob = 10 ∧
    cc.mary = 5 ∧
    cc.john = 5 ∧
    cc.sue = 20 ∧
    cc.sam = 10 ∧
    totalCandies cc = 50 := by
  sorry

end halloween_candy_count_l1640_164064


namespace min_green_beads_exact_min_green_beads_l1640_164006

/-- Represents a necklace with red, blue, and green beads. -/
structure Necklace where
  total : Nat
  red : Nat
  blue : Nat
  green : Nat
  sum_eq_total : red + blue + green = total
  red_between_blue : red ≥ blue
  green_between_red : green ≥ red

/-- The minimum number of green beads in a necklace of 80 beads satisfying the given conditions. -/
theorem min_green_beads (n : Necklace) (h : n.total = 80) : n.green ≥ 27 := by
  sorry

/-- The minimum number of green beads is exactly 27. -/
theorem exact_min_green_beads : ∃ n : Necklace, n.total = 80 ∧ n.green = 27 := by
  sorry

end min_green_beads_exact_min_green_beads_l1640_164006


namespace halloween_candy_distribution_l1640_164057

theorem halloween_candy_distribution (initial_candy : ℕ) (eaten_candy : ℕ) (num_piles : ℕ) 
  (h1 : initial_candy = 108)
  (h2 : eaten_candy = 36)
  (h3 : num_piles = 8) :
  (initial_candy - eaten_candy) / num_piles = 9 := by
  sorry

end halloween_candy_distribution_l1640_164057


namespace no_real_solutions_l1640_164066

theorem no_real_solutions : ¬∃ (x : ℝ), -3 * x - 8 = 8 * x^2 + 2 := by
  sorry

end no_real_solutions_l1640_164066


namespace check_problem_l1640_164062

/-- The check problem -/
theorem check_problem (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →
  (10 ≤ y ∧ y ≤ 99) →
  (100 * y + x) - (100 * x + y) = 2058 →
  (10 ≤ x ∧ x ≤ 78) ∧ y = x + 21 :=
by sorry

end check_problem_l1640_164062


namespace smallest_n_congruence_l1640_164020

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ k : ℕ+, k < n → (7 ^ k.val : ℤ) % 3 ≠ (k.val ^ 7 : ℤ) % 3) ∧ 
  (7 ^ n.val : ℤ) % 3 = (n.val ^ 7 : ℤ) % 3 → 
  n = 1 :=
sorry

end smallest_n_congruence_l1640_164020


namespace num_choices_eq_ten_l1640_164076

/-- The number of science subjects -/
def num_science : ℕ := 3

/-- The number of humanities subjects -/
def num_humanities : ℕ := 3

/-- The total number of subjects to choose from -/
def total_subjects : ℕ := num_science + num_humanities

/-- The number of subjects that must be chosen -/
def subjects_to_choose : ℕ := 3

/-- The minimum number of science subjects that must be chosen -/
def min_science : ℕ := 2

/-- The function that calculates the number of ways to choose subjects -/
def num_choices : ℕ := sorry

theorem num_choices_eq_ten : num_choices = 10 := by sorry

end num_choices_eq_ten_l1640_164076


namespace min_value_theorem_l1640_164050

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b + a * c + b * c + 2 * Real.sqrt 5 = 6 - a ^ 2) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 :=
by sorry

end min_value_theorem_l1640_164050


namespace no_solution_in_interval_l1640_164058

theorem no_solution_in_interval : ¬∃ x : ℝ, 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 := by
  sorry

end no_solution_in_interval_l1640_164058


namespace smallest_number_l1640_164094

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

def number_a : Nat := base_to_decimal [5, 8] 9
def number_b : Nat := base_to_decimal [0, 1, 2] 6
def number_c : Nat := base_to_decimal [0, 0, 0, 1] 4
def number_d : Nat := base_to_decimal [1, 1, 1, 1, 1, 1] 2

theorem smallest_number :
  number_d < number_a ∧ number_d < number_b ∧ number_d < number_c :=
sorry

end smallest_number_l1640_164094


namespace count_valid_m_l1640_164033

def is_valid (m : ℕ+) : Prop :=
  ∃ k : ℕ+, (2310 : ℚ) / ((m : ℚ)^2 - 2) = k

theorem count_valid_m :
  ∃! (s : Finset ℕ+), s.card = 3 ∧ ∀ m : ℕ+, m ∈ s ↔ is_valid m :=
sorry

end count_valid_m_l1640_164033


namespace quadratic_discriminant_equality_l1640_164098

theorem quadratic_discriminant_equality (a b c x : ℝ) (h1 : a ≠ 0) (h2 : a * x^2 + b * x + c = 0) : 
  b^2 - 4*a*c = (2*a*x + b)^2 := by
  sorry

end quadratic_discriminant_equality_l1640_164098


namespace existence_of_n_consecutive_representable_l1640_164026

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Part 1: Existence of n such that n + S(n) = 1980
theorem existence_of_n : ∃ n : ℕ, n + S n = 1980 := by sorry

-- Part 2: For any m, either m or m+1 can be expressed as n + S(n)
theorem consecutive_representable (m : ℕ) : 
  (∃ n : ℕ, n + S n = m) ∨ (∃ n : ℕ, n + S n = m + 1) := by sorry

end existence_of_n_consecutive_representable_l1640_164026


namespace zhukov_birth_year_l1640_164019

theorem zhukov_birth_year (total_years : ℕ) (years_diff : ℕ) (birth_year : ℕ) :
  total_years = 78 →
  years_diff = 70 →
  birth_year = 1900 - (total_years - years_diff) / 2 →
  birth_year = 1896 :=
by sorry

end zhukov_birth_year_l1640_164019


namespace probability_two_teachers_in_A_l1640_164040

def num_teachers : ℕ := 3
def num_places : ℕ := 2

def total_assignments : ℕ := num_places ^ num_teachers

def assignments_with_two_in_A : ℕ := (Nat.choose num_teachers 2)

theorem probability_two_teachers_in_A :
  (assignments_with_two_in_A : ℚ) / total_assignments = 3 / 8 := by
  sorry

end probability_two_teachers_in_A_l1640_164040


namespace rectangular_prism_prime_edges_l1640_164009

theorem rectangular_prism_prime_edges (a b c : ℕ) (k : ℕ) : 
  Prime a → Prime b → Prime c →
  ∃ p n : ℕ, Prime p ∧ 2 * (a * b + b * c + c * a) = p^n →
  (a = 2^k - 1 ∧ Prime (2^k - 1) ∧ b = 2 ∧ c = 2) ∨
  (b = 2^k - 1 ∧ Prime (2^k - 1) ∧ a = 2 ∧ c = 2) ∨
  (c = 2^k - 1 ∧ Prime (2^k - 1) ∧ a = 2 ∧ b = 2) :=
sorry

end rectangular_prism_prime_edges_l1640_164009


namespace circle_radius_proof_l1640_164049

theorem circle_radius_proof (A₁ A₂ : ℝ) (h1 : A₁ > 0) (h2 : A₂ > 0) : 
  (A₁ + A₂ = π * 5^2) →
  (A₂ = (A₁ + A₂) / 2) →
  ∃ r : ℝ, r > 0 ∧ A₁ = π * r^2 ∧ r = 5 * Real.sqrt 3 / 3 :=
by sorry

end circle_radius_proof_l1640_164049


namespace credit_sales_ratio_l1640_164088

theorem credit_sales_ratio (total_sales cash_sales : ℚ) 
  (h1 : total_sales = 80)
  (h2 : cash_sales = 48) :
  (total_sales - cash_sales) / total_sales = 2 / 5 := by
  sorry

end credit_sales_ratio_l1640_164088


namespace actual_average_height_l1640_164017

/-- The number of boys in the class -/
def num_boys : ℕ := 60

/-- The initial calculated average height in cm -/
def initial_avg : ℝ := 185

/-- The recorded heights of the three boys with errors -/
def recorded_heights : Fin 3 → ℝ := ![170, 195, 160]

/-- The actual heights of the three boys -/
def actual_heights : Fin 3 → ℝ := ![140, 165, 190]

/-- The actual average height of the boys in the class -/
def actual_avg : ℝ := 184.50

theorem actual_average_height :
  let total_initial := initial_avg * num_boys
  let total_difference := (recorded_heights 0 - actual_heights 0) +
                          (recorded_heights 1 - actual_heights 1) +
                          (recorded_heights 2 - actual_heights 2)
  let corrected_total := total_initial - total_difference
  corrected_total / num_boys = actual_avg := by sorry

end actual_average_height_l1640_164017


namespace range_of_a_l1640_164082

def f (x : ℝ) := x^2 - 4*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4) a, f x ∈ Set.Icc (-4) 32) →
  (∀ y ∈ Set.Icc (-4) 32, ∃ x ∈ Set.Icc (-4) a, f x = y) →
  a ∈ Set.Icc 2 8 := by
sorry

end range_of_a_l1640_164082


namespace lateral_surface_area_of_cone_l1640_164000

theorem lateral_surface_area_of_cone (slant_height base_radius : ℝ) 
  (h1 : slant_height = 4)
  (h2 : base_radius = 2) :
  (1/2) * slant_height * (2 * Real.pi * base_radius) = 8 * Real.pi :=
sorry

end lateral_surface_area_of_cone_l1640_164000


namespace shopping_trip_solution_l1640_164075

/-- The exchange rate from USD to CAD -/
def exchange_rate : ℚ := 8 / 5

/-- The amount spent in CAD -/
def amount_spent : ℕ := 80

/-- The function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem stating the solution to the problem -/
theorem shopping_trip_solution (d : ℕ) : 
  (exchange_rate * d - amount_spent = d) → sum_of_digits d = 7 := by
  sorry

#eval sum_of_digits 133  -- This should output 7

end shopping_trip_solution_l1640_164075


namespace rotary_club_omelet_eggs_rotary_club_omelet_eggs_proof_l1640_164072

/-- Calculate the number of eggs needed for the Rotary Club Omelet Breakfast -/
theorem rotary_club_omelet_eggs : ℕ :=
  let small_children := 53
  let older_children := 35
  let adults := 75
  let seniors := 37
  let small_children_omelets := 0.5
  let older_children_omelets := 1
  let adults_omelets := 2
  let seniors_omelets := 1.5
  let extra_omelets := 25
  let eggs_per_omelet := 2

  let total_omelets := small_children * small_children_omelets +
                       older_children * older_children_omelets +
                       adults * adults_omelets +
                       seniors * seniors_omelets +
                       extra_omelets

  let total_eggs := total_omelets * eggs_per_omelet

  584

theorem rotary_club_omelet_eggs_proof : rotary_club_omelet_eggs = 584 := by
  sorry

end rotary_club_omelet_eggs_rotary_club_omelet_eggs_proof_l1640_164072


namespace one_fifth_equals_point_two_l1640_164025

theorem one_fifth_equals_point_two : (1 : ℚ) / 5 = 0.200000 := by sorry

end one_fifth_equals_point_two_l1640_164025


namespace license_plate_combinations_l1640_164011

def number_of_letter_combinations : ℕ :=
  (Nat.choose 26 2) * 24 * (5 * 4 * 3 / (2 * 2))

def number_of_digit_combinations : ℕ := 10 * 9 * 8

theorem license_plate_combinations :
  number_of_letter_combinations * number_of_digit_combinations = 5644800 := by
  sorry

end license_plate_combinations_l1640_164011


namespace parallel_vectors_k_value_l1640_164085

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the non-collinearity condition
variable (h_non_collinear : ¬ ∃ (c : ℝ), e₁ = c • e₂)

-- Define vectors a and b
def a (e₁ e₂ : V) : V := 2 • e₁ + 3 • e₂
def b (k : ℝ) (e₁ e₂ : V) : V := k • e₁ - 4 • e₂

-- State the theorem
theorem parallel_vectors_k_value 
  (h_parallel : ∃ (m : ℝ), a e₁ e₂ = m • (b k e₁ e₂)) :
  k = -8/3 := by
  sorry

end parallel_vectors_k_value_l1640_164085


namespace soccer_substitutions_remainder_l1640_164069

/-- Number of ways to make n substitutions in a soccer game -/
def substitutions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => 11 * (12 - k) * substitutions k

/-- Total number of ways to make 0 to 4 substitutions -/
def totalSubstitutions : ℕ :=
  (List.range 5).map substitutions |>.sum

/-- The remainder when the total number of substitutions is divided by 1000 -/
theorem soccer_substitutions_remainder :
  totalSubstitutions % 1000 = 522 := by
  sorry

end soccer_substitutions_remainder_l1640_164069


namespace emily_earnings_is_twenty_l1640_164038

/-- The number of chocolate bars in a box -/
def total_bars : ℕ := 8

/-- The cost of each chocolate bar in dollars -/
def cost_per_bar : ℕ := 4

/-- The number of unsold bars -/
def unsold_bars : ℕ := 3

/-- Emily's earnings from selling chocolate bars -/
def emily_earnings : ℕ := (total_bars - unsold_bars) * cost_per_bar

/-- Theorem stating that Emily's earnings are $20 -/
theorem emily_earnings_is_twenty : emily_earnings = 20 := by
  sorry

end emily_earnings_is_twenty_l1640_164038


namespace polynomial_division_theorem_l1640_164021

theorem polynomial_division_theorem (x : ℝ) : 
  x^5 - 25*x^3 + 14*x^2 - 20*x + 15 = (x - 3)*(x^4 + 3*x^3 - 16*x^2 - 34*x - 122) + (-291) := by
  sorry

end polynomial_division_theorem_l1640_164021


namespace rope_remaining_lengths_l1640_164027

/-- Calculates the remaining lengths of two ropes after giving away portions. -/
theorem rope_remaining_lengths (x y : ℝ) (p q : ℝ) : 
  p = 0.40 * x ∧ q = 0.5625 * y := by
  sorry

#check rope_remaining_lengths

end rope_remaining_lengths_l1640_164027


namespace quotient_problem_l1640_164023

theorem quotient_problem (dividend : ℕ) (k : ℕ) (divisor : ℕ) :
  dividend = 64 → k = 8 → divisor = k → dividend / divisor = 8 := by
  sorry

end quotient_problem_l1640_164023


namespace amusement_park_visitors_l1640_164056

/-- Represents the amusement park ticket sales problem -/
theorem amusement_park_visitors 
  (ticket_price : ℕ) 
  (saturday_visitors : ℕ) 
  (sunday_visitors : ℕ) 
  (total_revenue : ℕ) 
  (h1 : ticket_price = 3)
  (h2 : saturday_visitors = 200)
  (h3 : sunday_visitors = 300)
  (h4 : total_revenue = 3000) :
  ∃ (daily_visitors : ℕ), 
    daily_visitors * 5 * ticket_price + (saturday_visitors + sunday_visitors) * ticket_price = total_revenue ∧ 
    daily_visitors = 100 := by
  sorry


end amusement_park_visitors_l1640_164056


namespace largest_integer_satisfying_inequality_l1640_164059

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (8 - 5 * x > 22) → x ≤ -3 ∧ 8 - 5 * (-3) > 22 :=
by
  sorry

end largest_integer_satisfying_inequality_l1640_164059


namespace contrapositive_equivalence_l1640_164002

theorem contrapositive_equivalence (a b c d : ℝ) :
  ((a = b ∧ c = d) → a + c = b + d) ↔ (a + c ≠ b + d → a ≠ b ∨ c ≠ d) :=
by sorry

end contrapositive_equivalence_l1640_164002


namespace simplify_and_evaluate_l1640_164053

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 3) :
  (m - (m + 9) / (m + 1)) / ((m^2 + 3*m) / (m + 1)) = 1 - m := by
  sorry

end simplify_and_evaluate_l1640_164053


namespace final_value_16_l1640_164080

/-- A function that simulates the loop behavior --/
def loop_iteration (b : ℕ) : ℕ := b + 3

/-- The loop condition --/
def loop_condition (b : ℕ) : Prop := b < 16

/-- The theorem statement --/
theorem final_value_16 :
  ∃ (n : ℕ), 
    let b := 10
    let final_b := (loop_iteration^[n] b)
    (∀ k < n, loop_condition ((loop_iteration^[k]) b)) ∧
    ¬(loop_condition final_b) ∧
    final_b = 16 :=
sorry

end final_value_16_l1640_164080


namespace students_in_neither_clubs_l1640_164035

/-- Represents the number of students in various categories in a class --/
structure ClassMembers where
  total : ℕ
  chinese : ℕ
  math : ℕ
  both : ℕ

/-- Calculates the number of students in neither the Chinese nor Math club --/
def studentsInNeither (c : ClassMembers) : ℕ :=
  c.total - (c.chinese + c.math - c.both)

/-- Theorem stating the number of students in neither club for the given scenario --/
theorem students_in_neither_clubs (c : ClassMembers) 
  (h_total : c.total = 55)
  (h_chinese : c.chinese = 32)
  (h_math : c.math = 36)
  (h_both : c.both = 18) :
  studentsInNeither c = 5 := by
  sorry

#eval studentsInNeither { total := 55, chinese := 32, math := 36, both := 18 }

end students_in_neither_clubs_l1640_164035


namespace super_soup_expansion_l1640_164097

/-- The number of new stores opened by Super Soup in 2020 -/
def new_stores_2020 (initial_2018 : ℕ) (opened_2019 closed_2019 closed_2020 final_2020 : ℕ) : ℕ :=
  final_2020 - (initial_2018 + opened_2019 - closed_2019 - closed_2020)

/-- Theorem stating that Super Soup opened 10 new stores in 2020 -/
theorem super_soup_expansion :
  new_stores_2020 23 5 2 6 30 = 10 := by
  sorry

end super_soup_expansion_l1640_164097


namespace special_triangle_side_length_l1640_164073

/-- An equilateral triangle with a special interior point -/
structure SpecialTriangle where
  /-- The side length of the equilateral triangle -/
  t : ℝ
  /-- The distance from vertex D to point Q -/
  DQ : ℝ
  /-- The distance from vertex E to point Q -/
  EQ : ℝ
  /-- The distance from vertex F to point Q -/
  FQ : ℝ
  /-- The triangle is equilateral -/
  equilateral : t > 0
  /-- The point Q is inside the triangle -/
  interior : DQ > 0 ∧ EQ > 0 ∧ FQ > 0
  /-- The distances from Q to the vertices -/
  distances : DQ = 2 ∧ EQ = Real.sqrt 5 ∧ FQ = 3

/-- The theorem stating that the side length of the special triangle is 2√3 -/
theorem special_triangle_side_length (T : SpecialTriangle) : T.t = 2 * Real.sqrt 3 := by
  sorry

end special_triangle_side_length_l1640_164073


namespace james_muffins_count_l1640_164055

/-- The number of muffins Arthur baked -/
def arthur_muffins : ℕ := 115

/-- The factor by which James baked more muffins than Arthur -/
def james_factor : ℕ := 12

/-- The number of muffins James baked -/
def james_muffins : ℕ := arthur_muffins * james_factor

theorem james_muffins_count : james_muffins = 1380 := by
  sorry

end james_muffins_count_l1640_164055


namespace max_gel_pens_l1640_164071

/-- Represents the number of pens of each type -/
structure PenCounts where
  ballpoint : ℕ
  gel : ℕ
  fountain : ℕ

/-- Checks if the given pen counts satisfy all conditions -/
def is_valid_count (counts : PenCounts) : Prop :=
  counts.ballpoint + counts.gel + counts.fountain = 20 ∧
  counts.ballpoint > 0 ∧ counts.gel > 0 ∧ counts.fountain > 0 ∧
  10 * counts.ballpoint + 50 * counts.gel + 80 * counts.fountain = 1000

/-- Theorem stating that the maximum number of gel pens is 13 -/
theorem max_gel_pens : 
  (∃ (counts : PenCounts), is_valid_count counts ∧ counts.gel = 13) ∧
  (∀ (counts : PenCounts), is_valid_count counts → counts.gel ≤ 13) :=
sorry

end max_gel_pens_l1640_164071


namespace arithmetic_progression_cosine_squared_l1640_164067

open Real

theorem arithmetic_progression_cosine_squared (x y z : ℝ) (α : ℝ) :
  (∃ k : ℝ, y = x + α ∧ z = y + α) →  -- x, y, z form an arithmetic progression
  α = arccos (-1/3) →                 -- common difference
  (∃ m : ℝ, 3 / cos y = 1 / cos x + m ∧ 1 / cos z = 3 / cos y + m) →  -- 1/cos(x), 3/cos(y), 1/cos(z) form an AP
  cos y ^ 2 = 4/5 := by
sorry

end arithmetic_progression_cosine_squared_l1640_164067


namespace rug_profit_calculation_l1640_164079

theorem rug_profit_calculation (buying_price selling_price num_rugs tax_rate transport_fee : ℚ) 
  (h1 : buying_price = 40)
  (h2 : selling_price = 60)
  (h3 : num_rugs = 20)
  (h4 : tax_rate = 1/10)
  (h5 : transport_fee = 5) :
  let total_cost := buying_price * num_rugs + transport_fee * num_rugs
  let total_revenue := selling_price * num_rugs * (1 + tax_rate)
  let profit := total_revenue - total_cost
  profit = 420 := by sorry

end rug_profit_calculation_l1640_164079


namespace monotonic_increase_interval_l1640_164028

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2 * Real.log x

theorem monotonic_increase_interval :
  ∀ x : ℝ, x > 0 → (∀ y : ℝ, y > x → f y > f x) ↔ x > Real.sqrt 3 / 3 :=
by sorry

end monotonic_increase_interval_l1640_164028


namespace repeating_decimal_calculation_l1640_164083

/-- Represents a repeating decimal with a two-digit repeating part -/
def RepeatingDecimal (a b : ℕ) : ℚ := a * 10 + b / 99

/-- The main theorem to prove -/
theorem repeating_decimal_calculation :
  let x : ℚ := RepeatingDecimal 5 4
  let y : ℚ := RepeatingDecimal 1 8
  (x / y) * (1 / 2) = 3 / 2 := by sorry

end repeating_decimal_calculation_l1640_164083


namespace corner_sum_is_164_l1640_164007

/-- Represents a square grid with side length n -/
structure Grid (n : ℕ) where
  size : ℕ
  size_eq : size = n * n

/-- The value of a cell in the grid given its row and column -/
def cellValue (g : Grid 9) (row col : ℕ) : ℕ :=
  (row - 1) * 9 + col

/-- The sum of the corner values in a 9x9 grid -/
def cornerSum (g : Grid 9) : ℕ :=
  cellValue g 1 1 + cellValue g 1 9 + cellValue g 9 1 + cellValue g 9 9

/-- Theorem: The sum of the corner values in a 9x9 grid is 164 -/
theorem corner_sum_is_164 (g : Grid 9) : cornerSum g = 164 := by
  sorry

end corner_sum_is_164_l1640_164007


namespace gcd_f_x_l1640_164044

def f (x : ℤ) : ℤ := (5*x+3)*(11*x+2)*(14*x+7)*(3*x+8)

theorem gcd_f_x (x : ℤ) (h : ∃ k : ℤ, x = 3456 * k) : 
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 48 := by
  sorry

end gcd_f_x_l1640_164044


namespace binomial_congruence_l1640_164043

theorem binomial_congruence (p n : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) (hn : n > 0) 
  (h_cong : n ≡ 1 [MOD p]) : 
  (Nat.choose (n * p) p) ≡ n [MOD p^4] := by sorry

end binomial_congruence_l1640_164043


namespace power_twentyseven_x_plus_one_l1640_164052

theorem power_twentyseven_x_plus_one (x : ℝ) (h : (3 : ℝ) ^ (2 * x) = 5) : 
  (27 : ℝ) ^ (x + 1) = 135 := by sorry

end power_twentyseven_x_plus_one_l1640_164052


namespace third_score_calculation_l1640_164074

theorem third_score_calculation (score1 score2 score4 : ℕ) (average : ℚ) :
  score1 = 65 →
  score2 = 67 →
  score4 = 85 →
  average = 75 →
  ∃ score3 : ℕ, (score1 + score2 + score3 + score4) / 4 = average ∧ score3 = 83 := by
  sorry

end third_score_calculation_l1640_164074


namespace new_person_weight_l1640_164061

/-- Represents a group of people with their weights -/
structure WeightGroup where
  size : Nat
  total_weight : ℝ
  avg_weight : ℝ

/-- Represents the change in the group when a person is replaced -/
structure WeightChange where
  old_weight : ℝ
  new_weight : ℝ
  avg_increase : ℝ

/-- Theorem stating the weight of the new person -/
theorem new_person_weight 
  (group : WeightGroup)
  (change : WeightChange)
  (h1 : group.size = 8)
  (h2 : change.old_weight = 65)
  (h3 : change.avg_increase = 3.5)
  (h4 : ∀ w E, (w * (1 + E / 100) - w) ≤ change.avg_increase) :
  change.new_weight = 93 := by
sorry


end new_person_weight_l1640_164061


namespace gcd_282_470_l1640_164068

theorem gcd_282_470 : Nat.gcd 282 470 = 94 := by
  sorry

end gcd_282_470_l1640_164068


namespace solve_linear_equation_l1640_164047

theorem solve_linear_equation (n : ℚ) (h : 2 * n + 5 = 16) : 2 * n - 3 = 8 := by
  sorry

end solve_linear_equation_l1640_164047


namespace power_of_two_between_powers_of_ten_l1640_164046

theorem power_of_two_between_powers_of_ten (t : ℕ+) : 
  (10 ^ (t.val - 1 : ℕ) < 2 ^ 64) ∧ (2 ^ 64 < 10 ^ t.val) → t = 20 := by
  sorry

end power_of_two_between_powers_of_ten_l1640_164046


namespace problem_solution_l1640_164001

open Real

noncomputable def α : ℝ := sorry

-- Given conditions
axiom cond1 : (sin (π/2 - α) + sin (-π - α)) / (3 * cos (2*π + α) + cos (3*π/2 - α)) = 3
axiom cond2 : ∃ (a : ℝ), ∀ (x y : ℝ), (x - a)^2 + y^2 = 7 → y = 0
axiom cond3 : ∃ (a : ℝ), abs (2*a) / sqrt 5 = sqrt 5
axiom cond4 : ∃ (a r : ℝ), r > 0 ∧ (2*sqrt 2)^2 + (sqrt 5)^2 = (2*r)^2 ∧ ∀ (x y : ℝ), (x - a)^2 + y^2 = r^2

-- Theorem to prove
theorem problem_solution :
  (sin α - 3*cos α) / (sin α + cos α) = -1/3 ∧
  ∃ (a : ℝ), (∀ (x y : ℝ), (x - a)^2 + y^2 = 7 ∨ (x + a)^2 + y^2 = 7) :=
sorry

end problem_solution_l1640_164001


namespace food_consumption_reduction_l1640_164008

/-- Calculates the required reduction in food consumption per student to maintain the same total cost, given a decrease in the number of students and an increase in food price. -/
theorem food_consumption_reduction 
  (student_decrease_rate : ℝ) 
  (food_price_increase_rate : ℝ) 
  (ε : ℝ) -- tolerance for approximation
  (h1 : student_decrease_rate = 0.05)
  (h2 : food_price_increase_rate = 0.20)
  (h3 : ε > 0)
  : ∃ (reduction_rate : ℝ), 
    abs (reduction_rate - (1 - 1 / ((1 - student_decrease_rate) * (1 + food_price_increase_rate)))) < ε ∧ 
    abs (reduction_rate - 0.1228) < ε := by
  sorry

end food_consumption_reduction_l1640_164008


namespace not_even_not_odd_composition_l1640_164015

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Statement of the theorem
theorem not_even_not_odd_composition (f : ℝ → ℝ) (c : ℝ) (h : OddFunction f) :
  ¬ (EvenFunction (fun x ↦ f (f (x + c)))) ∧ ¬ (OddFunction (fun x ↦ f (f (x + c)))) :=
sorry

end not_even_not_odd_composition_l1640_164015


namespace flag_distance_l1640_164060

theorem flag_distance (road_length : ℝ) (total_flags : ℕ) (h1 : road_length = 191.8) (h2 : total_flags = 58) :
  let intervals := total_flags / 2 - 1
  road_length / intervals = 6.85 := by
sorry

end flag_distance_l1640_164060


namespace matrix_square_zero_implication_l1640_164090

theorem matrix_square_zero_implication (n : ℕ) (M N : Matrix (Fin n) (Fin n) ℝ) 
  (h : (M * N)^2 = 0) :
  (n = 2 → (N * M)^2 = 0) ∧ 
  (n ≥ 3 → ∃ (M' N' : Matrix (Fin n) (Fin n) ℝ), (M' * N')^2 = 0 ∧ (N' * M')^2 ≠ 0) :=
by sorry

end matrix_square_zero_implication_l1640_164090


namespace equations_solutions_l1640_164018

-- Define the equations
def equation1 (x : ℝ) : Prop := 6 * x - 7 = 4 * x - 5
def equation2 (x : ℝ) : Prop := (3 * x - 1) / 4 - 1 = (5 * x - 7) / 6

-- State the theorem
theorem equations_solutions :
  (∃ x : ℝ, equation1 x ∧ x = 1) ∧
  (∃ x : ℝ, equation2 x ∧ x = -1) :=
sorry

end equations_solutions_l1640_164018


namespace function_properties_l1640_164095

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + x^2

theorem function_properties :
  ∀ a : ℝ,
  (∀ x : ℝ, x ∈ Set.Icc 1 (exp 1) → f a x ∈ Set.Icc (f a 1) (f a (exp 1))) ∧
  (a = -4 → ∀ x : ℝ, x ∈ Set.Icc 1 (exp 1) → f a x ≤ f a (exp 1)) ∧
  (a = -4 → f a (exp 1) = (exp 1)^2 - 4) ∧
  (∃ n : ℕ, n ≤ 2 ∧ (∃ S : Finset ℝ, S.card = n ∧ ∀ x ∈ S, x ∈ Set.Icc 1 (exp 1) ∧ f a x = 0)) ∧
  (a > 0 → ¬∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 (exp 1) → x₂ ∈ Set.Icc 1 (exp 1) →
    |f a x₁ - f a x₂| ≤ |1/x₁ - 1/x₂|) :=
by sorry

end function_properties_l1640_164095


namespace gumballs_per_pair_is_9_l1640_164031

/-- The number of gumballs Kim gets for each pair of earrings --/
def gumballs_per_pair : ℕ :=
  let earrings_day1 : ℕ := 3
  let earrings_day2 : ℕ := 2 * earrings_day1
  let earrings_day3 : ℕ := earrings_day2 - 1
  let total_earrings : ℕ := earrings_day1 + earrings_day2 + earrings_day3
  let gumballs_per_day : ℕ := 3
  let total_days : ℕ := 42
  let total_gumballs : ℕ := gumballs_per_day * total_days
  total_gumballs / total_earrings

theorem gumballs_per_pair_is_9 : gumballs_per_pair = 9 := by
  sorry

end gumballs_per_pair_is_9_l1640_164031


namespace triangle_existence_and_uniqueness_l1640_164010

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the given points
variable (D E F : Point)

-- Define the conditions
def is_midpoint (M : Point) (A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_trisection_point (E B C : Point) : Prop :=
  E.x = B.x + (C.x - B.x) / 3 ∧ E.y = B.y + (C.y - B.y) / 3

def is_quarter_point (F C A : Point) : Prop :=
  F.x = C.x + 3 * (A.x - C.x) / 4 ∧ F.y = C.y + 3 * (A.y - C.y) / 4

-- State the theorem
theorem triangle_existence_and_uniqueness :
  ∃! (ABC : Triangle),
    is_midpoint D ABC.A ABC.B ∧
    is_trisection_point E ABC.B ABC.C ∧
    is_quarter_point F ABC.C ABC.A :=
sorry

end triangle_existence_and_uniqueness_l1640_164010


namespace bank_comparison_l1640_164037

/-- Calculates the annual yield given a quarterly interest rate -/
def annual_yield_quarterly (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate) ^ 4 - 1

/-- Calculates the annual yield given an annual interest rate -/
def annual_yield_annual (annual_rate : ℝ) : ℝ :=
  annual_rate

theorem bank_comparison (bank1_quarterly_rate : ℝ) (bank2_annual_rate : ℝ)
    (h1 : bank1_quarterly_rate = 0.8)
    (h2 : bank2_annual_rate = -9) :
    annual_yield_quarterly bank1_quarterly_rate > annual_yield_annual bank2_annual_rate := by
  sorry

#eval annual_yield_quarterly 0.8
#eval annual_yield_annual (-9)

end bank_comparison_l1640_164037


namespace factoring_a_squared_minus_nine_l1640_164030

theorem factoring_a_squared_minus_nine (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := by
  sorry

end factoring_a_squared_minus_nine_l1640_164030


namespace simplify_expression_l1640_164099

theorem simplify_expression (m : ℝ) : (-m^4)^5 / m^5 * m = -m^14 := by
  sorry

end simplify_expression_l1640_164099


namespace not_all_trihedral_angles_form_equilateral_triangles_l1640_164063

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

/-- Represents a plane intersecting a trihedral angle -/
structure Intersection where
  angle : TrihedralAngle
  plane : Unit  -- We don't need to define the plane explicitly for this problem

/-- Predicate to check if an intersection forms an equilateral triangle -/
def forms_equilateral_triangle (i : Intersection) : Prop :=
  -- This would involve complex geometric calculations in reality
  sorry

/-- Theorem stating that not all trihedral angles can be intersected to form equilateral triangles -/
theorem not_all_trihedral_angles_form_equilateral_triangles :
  ∃ (t : TrihedralAngle), ∀ (p : Unit), ¬(forms_equilateral_triangle ⟨t, p⟩) :=
sorry

end not_all_trihedral_angles_form_equilateral_triangles_l1640_164063


namespace man_speed_calculation_man_speed_approximately_5_004_l1640_164041

/-- Calculates the speed of a man walking opposite to a train, given the train's length, speed, and time to cross the man. -/
theorem man_speed_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh

/-- The speed of the man is approximately 5.004 km/h -/
theorem man_speed_approximately_5_004 :
  ∃ ε > 0, |man_speed_calculation 200 114.99 6 - 5.004| < ε :=
sorry

end man_speed_calculation_man_speed_approximately_5_004_l1640_164041


namespace square_side_length_l1640_164091

theorem square_side_length (diagonal_inches : ℝ) (h : diagonal_inches = 2 * Real.sqrt 2) :
  let diagonal_feet := diagonal_inches / 12
  let side_feet := diagonal_feet / Real.sqrt 2
  side_feet = 1 / 6 := by sorry

end square_side_length_l1640_164091


namespace timi_ears_count_l1640_164004

-- Define the inhabitants
structure Inhabitant where
  name : String
  ears_seen : Nat

-- Define the problem setup
def zog_problem : List Inhabitant :=
  [{ name := "Imi", ears_seen := 8 },
   { name := "Dimi", ears_seen := 7 },
   { name := "Timi", ears_seen := 5 }]

-- Theorem: Timi has 5 ears
theorem timi_ears_count (problem : List Inhabitant) : 
  problem = zog_problem → 
  (problem.find? (fun i => i.name = "Timi")).map (fun i => 
    List.sum (problem.map (fun j => j.ears_seen)) / 2 - i.ears_seen) = some 5 := by
  sorry

end timi_ears_count_l1640_164004


namespace stock_price_change_l1640_164034

theorem stock_price_change (total_stocks : ℕ) (higher_percentage : ℚ) :
  total_stocks = 1980 →
  higher_percentage = 120 / 100 →
  ∃ (higher lower : ℕ),
    higher + lower = total_stocks ∧
    higher = (higher_percentage * lower).num ∧
    higher = 1080 :=
by sorry

end stock_price_change_l1640_164034


namespace simple_interest_rate_interest_rate_problem_l1640_164022

/-- Simple interest calculation --/
theorem simple_interest_rate (principal amount : ℚ) (time : ℕ) (rate : ℚ) : 
  principal * (1 + rate * time) = amount →
  rate = (amount - principal) / (principal * time) :=
by
  sorry

/-- Prove that the interest rate is 5% given the problem conditions --/
theorem interest_rate_problem :
  let principal : ℚ := 600
  let amount : ℚ := 720
  let time : ℕ := 4
  let rate : ℚ := (amount - principal) / (principal * time)
  rate = 5 / 100 :=
by
  sorry

end simple_interest_rate_interest_rate_problem_l1640_164022


namespace same_color_probability_l1640_164032

def total_balls : ℕ := 20
def blue_balls : ℕ := 8
def green_balls : ℕ := 5
def red_balls : ℕ := 7

theorem same_color_probability :
  let prob_blue := (blue_balls / total_balls) ^ 2
  let prob_green := (green_balls / total_balls) ^ 2
  let prob_red := (red_balls / total_balls) ^ 2
  prob_blue + prob_green + prob_red = 117 / 200 := by
sorry

end same_color_probability_l1640_164032


namespace ending_number_proof_l1640_164036

theorem ending_number_proof (n : ℕ) : 
  (n > 100) ∧ 
  (∃ (count : ℕ), count = 33 ∧ 
    (∀ k : ℕ, 100 < k ∧ k ≤ n ∧ k % 3 = 0 → 
      ∃ i : ℕ, i ≤ count ∧ k = 100 + 3 * i)) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ (count : ℕ), count = 33 ∧ 
      (∀ k : ℕ, 100 < k ∧ k ≤ m ∧ k % 3 = 0 → 
        ∃ i : ℕ, i ≤ count ∧ k = 100 + 3 * i))) →
  n = 198 := by
sorry

end ending_number_proof_l1640_164036


namespace binary_to_base4_correct_l1640_164012

/-- Converts a binary number to base 4 --/
def binary_to_base4 (b : ℕ) : ℕ := sorry

/-- The binary representation of the number --/
def binary_num : ℕ := 110110100

/-- The base 4 representation of the number --/
def base4_num : ℕ := 31220

/-- Theorem stating that the conversion of the binary number to base 4 is correct --/
theorem binary_to_base4_correct : binary_to_base4 binary_num = base4_num := by sorry

end binary_to_base4_correct_l1640_164012


namespace phil_coin_collection_l1640_164065

def initial_coins : ℕ := 250
def years_tripling : ℕ := 3
def weeks_in_year : ℕ := 52
def days_in_year : ℕ := 365
def coins_per_week_4th_year : ℕ := 5
def coins_every_second_day_5th_year : ℕ := 2
def coins_per_day_6th_year : ℕ := 1
def loss_fraction : ℚ := 1/3

def coins_after_loss : ℕ := 1160

theorem phil_coin_collection :
  let coins_after_3_years := initial_coins * (2^years_tripling)
  let coins_4th_year := coins_after_3_years + coins_per_week_4th_year * weeks_in_year
  let coins_5th_year := coins_4th_year + coins_every_second_day_5th_year * (days_in_year / 2)
  let coins_6th_year := coins_5th_year + coins_per_day_6th_year * days_in_year
  let coins_before_loss := coins_6th_year
  coins_after_loss = coins_before_loss - ⌊coins_before_loss * loss_fraction⌋ :=
by sorry

end phil_coin_collection_l1640_164065


namespace matches_left_after_2022_l1640_164013

/-- The number of matchsticks needed to form a digit --/
def matchsticks_for_digit (d : Nat) : Nat :=
  if d = 2 then 5
  else if d = 0 then 6
  else 0  -- We only care about 2 and 0 for this problem

/-- The number of matchsticks needed to form the year 2022 --/
def matchsticks_for_2022 : Nat :=
  matchsticks_for_digit 2 * 3 + matchsticks_for_digit 0

/-- The initial number of matches in the box --/
def initial_matches : Nat := 30

/-- Theorem: After forming 2022 with matchsticks, 9 matches will be left --/
theorem matches_left_after_2022 :
  initial_matches - matchsticks_for_2022 = 9 := by
  sorry


end matches_left_after_2022_l1640_164013


namespace quadratic_inequality_range_l1640_164029

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
by sorry

end quadratic_inequality_range_l1640_164029


namespace yuri_puppies_count_l1640_164077

/-- The number of puppies Yuri adopted in the first week -/
def first_week : ℕ := 20

/-- The number of puppies Yuri adopted in the second week -/
def second_week : ℕ := (2 * first_week) / 5

/-- The number of puppies Yuri adopted in the third week -/
def third_week : ℕ := (3 * second_week) / 8

/-- The number of puppies Yuri adopted in the fourth week -/
def fourth_week : ℕ := 2 * second_week

/-- The number of puppies Yuri adopted in the fifth week -/
def fifth_week : ℕ := first_week + 10

/-- The number of puppies Yuri adopted in the sixth week -/
def sixth_week : ℕ := 2 * third_week - 5

/-- The number of puppies Yuri adopted in the seventh week -/
def seventh_week : ℕ := 2 * sixth_week

/-- The number of puppies Yuri adopted in half of the eighth week -/
def eighth_week_half : ℕ := (5 * seventh_week) / 6

/-- The total number of puppies Yuri adopted -/
def total_puppies : ℕ := first_week + second_week + third_week + fourth_week + 
                         fifth_week + sixth_week + seventh_week + eighth_week_half

theorem yuri_puppies_count : total_puppies = 81 := by
  sorry

end yuri_puppies_count_l1640_164077
