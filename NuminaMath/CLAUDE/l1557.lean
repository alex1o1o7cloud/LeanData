import Mathlib

namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1557_155713

-- Define the two hyperbolas
def hyperbola1 (x y : ℝ) : Prop := x^2/16 - y^2/9 = 1
def hyperbola2 (x y : ℝ) : Prop := y^2/9 - x^2/16 = 1

-- Define the asymptotes for a hyperbola
def asymptotes (a b : ℝ) (x y : ℝ) : Prop := y = (b/a)*x ∨ y = -(b/a)*x

-- Theorem stating that both hyperbolas have the same asymptotes
theorem hyperbolas_same_asymptotes :
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    (hyperbola1 x y ∨ hyperbola2 x y) → asymptotes a b x y :=
sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1557_155713


namespace NUMINAMATH_CALUDE_find_a_l1557_155723

theorem find_a : ∃ a : ℤ, 
  (∃ x : ℤ, (2 * x - a = 3) ∧ 
    (∀ y : ℤ, (1 - (y - 2) / 2 : ℚ) < ((1 + y) / 3 : ℚ) → y ≥ x) ∧
    (1 - (x - 2) / 2 : ℚ) < ((1 + x) / 3 : ℚ)) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_find_a_l1557_155723


namespace NUMINAMATH_CALUDE_fraction_problem_l1557_155776

theorem fraction_problem (numerator : ℕ) : 
  (numerator : ℚ) / (2 * numerator + 4) = 3 / 7 → numerator = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1557_155776


namespace NUMINAMATH_CALUDE_average_of_9_15_N_l1557_155790

theorem average_of_9_15_N (N : ℝ) (h1 : 12 < N) (h2 : N < 25) :
  let avg := (9 + 15 + N) / 3
  avg = 15 ∨ avg = 17 :=
by sorry

end NUMINAMATH_CALUDE_average_of_9_15_N_l1557_155790


namespace NUMINAMATH_CALUDE_system_of_equations_l1557_155750

theorem system_of_equations (y : ℝ) :
  ∃ (x z : ℝ),
    (19 * (x + y) + 17 = 19 * (-x + y) - 21) ∧
    (5 * x - 3 * z = 11 * y - 7) ∧
    (x = -1) ∧
    (z = -11 * y / 3 + 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l1557_155750


namespace NUMINAMATH_CALUDE_geometric_series_sum_of_cubes_l1557_155702

theorem geometric_series_sum_of_cubes 
  (a : ℝ) (r : ℝ) (hr : -1 < r ∧ r < 1) 
  (h1 : a / (1 - r) = 2) 
  (h2 : a^2 / (1 - r^2) = 6) : 
  a^3 / (1 - r^3) = 96/7 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_of_cubes_l1557_155702


namespace NUMINAMATH_CALUDE_surface_area_of_sliced_prism_l1557_155781

/-- Right prism with equilateral triangle base -/
structure RightPrism :=
  (height : ℝ)
  (base_side_length : ℝ)

/-- Point on an edge of the prism -/
structure EdgePoint :=
  (distance_ratio : ℝ)

/-- Solid formed by slicing off part of the prism -/
structure SlicedSolid :=
  (prism : RightPrism)
  (point1 : EdgePoint)
  (point2 : EdgePoint)
  (point3 : EdgePoint)

/-- Surface area of the sliced solid -/
def surface_area (solid : SlicedSolid) : ℝ :=
  sorry

theorem surface_area_of_sliced_prism (solid : SlicedSolid) 
  (h1 : solid.prism.height = 24)
  (h2 : solid.prism.base_side_length = 18)
  (h3 : solid.point1.distance_ratio = 1/3)
  (h4 : solid.point2.distance_ratio = 1/3)
  (h5 : solid.point3.distance_ratio = 1/3) :
  surface_area solid = 96 + 9 * Real.sqrt 3 + 9 * Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_sliced_prism_l1557_155781


namespace NUMINAMATH_CALUDE_parabola_distance_l1557_155797

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def point_B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem parabola_distance (A : ℝ × ℝ) :
  parabola A.1 A.2 →
  ‖A - focus‖ = ‖point_B - focus‖ →
  ‖A - point_B‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_distance_l1557_155797


namespace NUMINAMATH_CALUDE_temperature_difference_product_of_N_values_l1557_155796

theorem temperature_difference (B D N : ℝ) : 
  B = D - N →
  (∃ k, k = (D - N + 10) - (D - 4) ∧ (k = 1 ∨ k = -1)) →
  (N = 13 ∨ N = 15) :=
sorry

theorem product_of_N_values : 
  (∃ N₁ N₂ : ℝ, (N₁ = 13 ∧ N₂ = 15) ∧ N₁ * N₂ = 195) :=
sorry

end NUMINAMATH_CALUDE_temperature_difference_product_of_N_values_l1557_155796


namespace NUMINAMATH_CALUDE_pauls_strawberries_l1557_155771

/-- Given an initial count of strawberries and an additional number picked,
    calculate the total number of strawberries. -/
def total_strawberries (initial : ℕ) (picked : ℕ) : ℕ :=
  initial + picked

/-- Theorem: Paul's total strawberries after picking more -/
theorem pauls_strawberries :
  total_strawberries 42 78 = 120 := by
  sorry

end NUMINAMATH_CALUDE_pauls_strawberries_l1557_155771


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1557_155716

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1557_155716


namespace NUMINAMATH_CALUDE_number_problem_l1557_155765

theorem number_problem : ∃ x : ℝ, (x / 3) * 12 = 9 ∧ x = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1557_155765


namespace NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l1557_155724

def number_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_number_with_2020_divisors :
  ∀ m : ℕ, m < 2^100 * 3^4 * 5 * 7 →
    number_of_divisors m ≠ 2020 ∧
    number_of_divisors (2^100 * 3^4 * 5 * 7) = 2020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l1557_155724


namespace NUMINAMATH_CALUDE_class_size_proof_l1557_155756

theorem class_size_proof (original_average : ℝ) (new_students : ℕ) (new_students_average : ℝ) (average_decrease : ℝ) :
  original_average = 40 →
  new_students = 8 →
  new_students_average = 32 →
  average_decrease = 4 →
  ∃ (original_size : ℕ), 
    (original_size * original_average + new_students * new_students_average) / (original_size + new_students) = original_average - average_decrease ∧
    original_size = 8 :=
by sorry

end NUMINAMATH_CALUDE_class_size_proof_l1557_155756


namespace NUMINAMATH_CALUDE_share_calculation_l1557_155744

theorem share_calculation (total : ℝ) (a b c : ℝ) : 
  total = 300 →
  a + b + c = total →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a = 120 := by sorry

end NUMINAMATH_CALUDE_share_calculation_l1557_155744


namespace NUMINAMATH_CALUDE_total_hamburger_combinations_l1557_155719

/-- The number of condiments available. -/
def num_condiments : ℕ := 10

/-- The number of choices for each condiment (include or not include). -/
def choices_per_condiment : ℕ := 2

/-- The number of choices for meat patties. -/
def meat_patty_choices : ℕ := 3

/-- The number of choices for bun types. -/
def bun_choices : ℕ := 3

/-- Theorem stating the total number of different hamburger combinations. -/
theorem total_hamburger_combinations :
  (choices_per_condiment ^ num_condiments) * meat_patty_choices * bun_choices = 9216 := by
  sorry


end NUMINAMATH_CALUDE_total_hamburger_combinations_l1557_155719


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l1557_155789

theorem quadratic_root_implies_k (k : ℝ) : 
  (2 * (5 : ℝ)^2 + 3 * 5 - k = 0) → k = 65 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l1557_155789


namespace NUMINAMATH_CALUDE_wally_bear_cost_l1557_155787

/-- Calculates the total cost of bears given the number of bears, initial price, and discount per bear. -/
def total_cost (num_bears : ℕ) (initial_price : ℚ) (discount : ℚ) : ℚ :=
  initial_price + (num_bears - 1 : ℚ) * (initial_price - discount)

/-- Theorem stating that the total cost for 101 bears is $354, given the specified pricing scheme. -/
theorem wally_bear_cost :
  total_cost 101 4 0.5 = 354 := by
  sorry

end NUMINAMATH_CALUDE_wally_bear_cost_l1557_155787


namespace NUMINAMATH_CALUDE_snake_count_theorem_l1557_155786

/-- Represents the number of pet owners for different combinations of pets --/
structure PetOwners where
  total : Nat
  onlyDogs : Nat
  onlyCats : Nat
  catsAndDogs : Nat
  catsDogsSnakes : Nat

/-- Given the pet ownership data, proves that the minimum number of snakes is 3
    and that the total number of snakes cannot be determined --/
theorem snake_count_theorem (po : PetOwners)
  (h1 : po.total = 79)
  (h2 : po.onlyDogs = 15)
  (h3 : po.onlyCats = 10)
  (h4 : po.catsAndDogs = 5)
  (h5 : po.catsDogsSnakes = 3) :
  ∃ (minSnakes : Nat), minSnakes = 3 ∧ 
  ¬∃ (totalSnakes : Nat), ∀ (n : Nat), n ≥ minSnakes → n = totalSnakes :=
by sorry

end NUMINAMATH_CALUDE_snake_count_theorem_l1557_155786


namespace NUMINAMATH_CALUDE_dog_food_weight_l1557_155751

theorem dog_food_weight (initial_amount : ℝ) (second_bag_weight : ℝ) (final_amount : ℝ) 
  (h1 : initial_amount = 15)
  (h2 : second_bag_weight = 10)
  (h3 : final_amount = 40) :
  ∃ (first_bag_weight : ℝ), 
    initial_amount + first_bag_weight + second_bag_weight = final_amount ∧ 
    first_bag_weight = 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_weight_l1557_155751


namespace NUMINAMATH_CALUDE_football_scoring_problem_l1557_155778

/-- Represents the football scoring problem with Gina and Tom -/
theorem football_scoring_problem 
  (gina_day1 : ℕ) 
  (tom_day1 : ℕ) 
  (tom_day2 : ℕ) 
  (gina_day2 : ℕ) 
  (h1 : gina_day1 = 2)
  (h2 : tom_day1 = gina_day1 + 3)
  (h3 : tom_day2 = 6)
  (h4 : gina_day2 < tom_day2)
  (h5 : gina_day1 + tom_day1 + gina_day2 + tom_day2 = 17) :
  tom_day2 - gina_day2 = 2 := by
sorry

end NUMINAMATH_CALUDE_football_scoring_problem_l1557_155778


namespace NUMINAMATH_CALUDE_mary_bought_24_cards_l1557_155740

/-- The number of baseball cards Mary bought -/
def cards_bought (initial_cards promised_cards remaining_cards : ℝ) : ℝ :=
  remaining_cards - (initial_cards - promised_cards)

/-- Theorem: Mary bought 24.0 baseball cards -/
theorem mary_bought_24_cards :
  cards_bought 18.0 26.0 32.0 = 24.0 := by
  sorry

end NUMINAMATH_CALUDE_mary_bought_24_cards_l1557_155740


namespace NUMINAMATH_CALUDE_jenny_house_improvements_l1557_155725

/-- Represents the problem of calculating the maximum value of improvements Jenny can make to her house. -/
theorem jenny_house_improvements
  (tax_rate : ℝ)
  (initial_house_value : ℝ)
  (rail_project_increase : ℝ)
  (max_affordable_tax : ℝ)
  (h1 : tax_rate = 0.02)
  (h2 : initial_house_value = 400000)
  (h3 : rail_project_increase = 0.25)
  (h4 : max_affordable_tax = 15000) :
  let new_house_value := initial_house_value * (1 + rail_project_increase)
  let max_house_value := max_affordable_tax / tax_rate
  max_house_value - new_house_value = 250000 :=
by sorry

end NUMINAMATH_CALUDE_jenny_house_improvements_l1557_155725


namespace NUMINAMATH_CALUDE_school_play_attendance_l1557_155774

theorem school_play_attendance : 
  let num_girls : ℕ := 10
  let num_boys : ℕ := 12
  let family_members_per_kid : ℕ := 3
  let kids_with_stepparent : ℕ := 5
  let kids_with_grandparents : ℕ := 3
  let additional_grandparents_per_kid : ℕ := 2

  (num_girls + num_boys) * family_members_per_kid + 
  kids_with_stepparent + 
  kids_with_grandparents * additional_grandparents_per_kid = 77 :=
by sorry

end NUMINAMATH_CALUDE_school_play_attendance_l1557_155774


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_line_l1557_155783

/-- A line that passes through the point (3, 2) and forms an isosceles right triangle with the coordinate axes has the equation x - y - 1 = 0 or x + y - 5 = 0 -/
theorem isosceles_right_triangle_line : 
  ∀ (l : Set (ℝ × ℝ)), 
  ((3, 2) ∈ l) → 
  (∃ (a b : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ a * x + b * y = 1) →
  (∃ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l ∧ p.1 = 0 ∧ q.2 = 0 ∧ p.2 = q.1 ∧ 
    (p.2 - 3)^2 + (q.1 - 3)^2 = (3 - 0)^2 + (2 - 0)^2) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (x - y = 1 ∨ x + y = 5)) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_line_l1557_155783


namespace NUMINAMATH_CALUDE_rachel_adam_weight_difference_l1557_155788

/-- Given the weights of three people Rachel, Jimmy, and Adam, prove that Rachel weighs 15 pounds more than Adam. -/
theorem rachel_adam_weight_difference (R J A : ℝ) : 
  R = 75 →  -- Rachel weighs 75 pounds
  R = J - 6 →  -- Rachel weighs 6 pounds less than Jimmy
  R > A →  -- Rachel weighs more than Adam
  (R + J + A) / 3 = 72 →  -- The average weight of the three people is 72 pounds
  R - A = 15 :=  -- Rachel weighs 15 pounds more than Adam
by sorry

end NUMINAMATH_CALUDE_rachel_adam_weight_difference_l1557_155788


namespace NUMINAMATH_CALUDE_total_village_tax_l1557_155703

/-- Represents the farm tax collected from a village -/
structure FarmTax where
  total_tax : ℝ
  willam_tax : ℝ
  willam_land_percentage : ℝ

/-- Theorem stating the total tax collected from the village -/
theorem total_village_tax (ft : FarmTax) 
  (h1 : ft.willam_tax = 480)
  (h2 : ft.willam_land_percentage = 25) :
  ft.total_tax = 1920 := by
  sorry

end NUMINAMATH_CALUDE_total_village_tax_l1557_155703


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1557_155718

theorem inequality_solution_set (x : ℝ) : x - 3 > 4*x ↔ x < -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1557_155718


namespace NUMINAMATH_CALUDE_zion_age_is_8_dad_age_relation_future_age_relation_l1557_155727

/-- Zion's current age in years -/
def zion_age : ℕ := 8

/-- Zion's dad's current age in years -/
def dad_age : ℕ := 4 * zion_age + 3

theorem zion_age_is_8 : zion_age = 8 := by sorry

theorem dad_age_relation : dad_age = 4 * zion_age + 3 := by sorry

theorem future_age_relation : dad_age + 10 = (zion_age + 10) + 27 := by sorry

end NUMINAMATH_CALUDE_zion_age_is_8_dad_age_relation_future_age_relation_l1557_155727


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1557_155794

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (n : ℕ) :
  (∀ k : ℕ, a (k + 2) - a (k + 1) = a (k + 1) - a k) →  -- arithmetic sequence condition
  a 1 = 1/3 →
  a 2 + a 5 = 4 →
  a n = 33 →
  n = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1557_155794


namespace NUMINAMATH_CALUDE_count_possible_denominators_l1557_155721

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def repeating_decimal_to_fraction (c d : ℕ) : ℚ :=
  (10 * c + d : ℚ) / 99

theorem count_possible_denominators :
  ∃ (S : Finset ℕ),
    (∀ c d : ℕ, is_valid_digit c → is_valid_digit d →
      (c ≠ 8 ∨ d ≠ 8) → (c ≠ 0 ∨ d ≠ 0) →
      (repeating_decimal_to_fraction c d).den ∈ S) ∧
    S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_possible_denominators_l1557_155721


namespace NUMINAMATH_CALUDE_grain_milling_problem_l1557_155733

theorem grain_milling_problem (grain_weight : ℚ) : 
  (grain_weight * (1 - 1/10) = 100) → grain_weight = 1000/9 := by
  sorry

end NUMINAMATH_CALUDE_grain_milling_problem_l1557_155733


namespace NUMINAMATH_CALUDE_true_discount_equals_bankers_gain_l1557_155770

/-- Present worth of a sum due -/
def present_worth : ℝ := 576

/-- Banker's gain -/
def bankers_gain : ℝ := 16

/-- True discount -/
def true_discount : ℝ := bankers_gain

theorem true_discount_equals_bankers_gain :
  true_discount = bankers_gain :=
by sorry

end NUMINAMATH_CALUDE_true_discount_equals_bankers_gain_l1557_155770


namespace NUMINAMATH_CALUDE_num_outfits_l1557_155763

/-- Number of shirts available -/
def num_shirts : ℕ := 8

/-- Number of ties available -/
def num_ties : ℕ := 5

/-- Number of pairs of pants available -/
def num_pants : ℕ := 4

/-- Number of jackets available -/
def num_jackets : ℕ := 2

/-- Number of tie options (including no tie) -/
def tie_options : ℕ := num_ties + 1

/-- Number of jacket options (including no jacket) -/
def jacket_options : ℕ := num_jackets + 1

/-- Theorem stating the number of distinct outfits -/
theorem num_outfits : num_shirts * num_pants * tie_options * jacket_options = 576 := by
  sorry

end NUMINAMATH_CALUDE_num_outfits_l1557_155763


namespace NUMINAMATH_CALUDE_max_b_value_l1557_155768

theorem max_b_value (a b c : ℕ) (h1 : a * b * c = 360) (h2 : 1 < c) (h3 : c < b) (h4 : b < a) :
  b ≤ 12 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = 360 ∧ 1 < c' ∧ c' < b' ∧ b' < a' ∧ b' = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l1557_155768


namespace NUMINAMATH_CALUDE_coin_division_problem_l1557_155766

theorem coin_division_problem : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 8 = 6) ∧ 
  (n % 7 = 5) ∧ 
  (n % 9 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 8 = 6 ∧ m % 7 = 5 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_coin_division_problem_l1557_155766


namespace NUMINAMATH_CALUDE_common_root_values_l1557_155728

theorem common_root_values (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * k^3 + b * k^2 + c * k + d = 0)
  (h2 : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_common_root_values_l1557_155728


namespace NUMINAMATH_CALUDE_no_prime_factor_6k_plus_5_l1557_155711

theorem no_prime_factor_6k_plus_5 (n k : ℕ+) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_factor : p ∣ n^2 - n + 1) : p ≠ 6 * k + 5 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_factor_6k_plus_5_l1557_155711


namespace NUMINAMATH_CALUDE_marks_trees_l1557_155762

theorem marks_trees (initial_trees : ℕ) (new_trees_per_existing : ℕ) : 
  initial_trees = 93 → new_trees_per_existing = 8 → 
  initial_trees + initial_trees * new_trees_per_existing = 837 := by
sorry

end NUMINAMATH_CALUDE_marks_trees_l1557_155762


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l1557_155785

def tangent_sequence (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, k > 0 → a (k + 1) = (3 / 2) * a k

theorem sum_of_specific_terms 
  (a : ℕ → ℝ) 
  (h1 : tangent_sequence a) 
  (h2 : a 1 = 16) : 
  a 1 + a 3 + a 5 = 133 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l1557_155785


namespace NUMINAMATH_CALUDE_centroid_trace_area_centroid_trace_area_diameter_30_l1557_155741

/-- The area of the region bounded by the curve traced by the centroid of a triangle
    inscribed in a circle, where the base of the triangle is a diameter of the circle. -/
theorem centroid_trace_area (r : ℝ) (h : r > 0) : 
  (π * (r / 3)^2) = (25 * π / 9) * r^2 := by
  sorry

/-- The specific case where the diameter of the circle is 30 -/
theorem centroid_trace_area_diameter_30 : 
  (π * 5^2) = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_centroid_trace_area_centroid_trace_area_diameter_30_l1557_155741


namespace NUMINAMATH_CALUDE_set_operations_and_conditions_l1557_155742

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 8}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Define the theorem
theorem set_operations_and_conditions :
  -- Part 1
  (A 0 ∩ B = {x | 5 < x ∧ x ≤ 8}) ∧
  (A 0 ∪ Bᶜ = {x | -1 ≤ x ∧ x ≤ 8}) ∧
  -- Part 2
  (∀ a : ℝ, A a ∪ B = B ↔ a < -9 ∨ a > 5) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_conditions_l1557_155742


namespace NUMINAMATH_CALUDE_modulus_of_z_is_5_l1557_155709

-- Define the complex number z
def z : ℂ := (2 - Complex.I) ^ 2

-- Theorem stating that the modulus of z is 5
theorem modulus_of_z_is_5 : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_is_5_l1557_155709


namespace NUMINAMATH_CALUDE_pig_profit_is_960_l1557_155757

/-- Calculates the profit from selling pigs given the specified conditions -/
def calculate_pig_profit (num_piglets : ℕ) (sale_price : ℕ) (feeding_cost : ℕ) 
  (months_group1 : ℕ) (months_group2 : ℕ) : ℕ :=
  let revenue := num_piglets * sale_price
  let cost_group1 := (num_piglets / 2) * feeding_cost * months_group1
  let cost_group2 := (num_piglets / 2) * feeding_cost * months_group2
  let total_cost := cost_group1 + cost_group2
  revenue - total_cost

/-- The profit from selling pigs under the given conditions is $960 -/
theorem pig_profit_is_960 : 
  calculate_pig_profit 6 300 10 12 16 = 960 := by
  sorry

end NUMINAMATH_CALUDE_pig_profit_is_960_l1557_155757


namespace NUMINAMATH_CALUDE_birthday_celebration_friends_l1557_155747

/-- The number of friends attending Paolo and Sevilla's birthday celebration -/
def num_friends : ℕ := sorry

/-- The total bill amount -/
def total_bill : ℕ := sorry

theorem birthday_celebration_friends :
  (total_bill = 12 * (num_friends + 2)) ∧
  (total_bill = 16 * num_friends) →
  num_friends = 6 := by sorry

end NUMINAMATH_CALUDE_birthday_celebration_friends_l1557_155747


namespace NUMINAMATH_CALUDE_infinitely_many_composites_l1557_155726

/-- A strictly increasing sequence of natural numbers where each number from
    the third one onwards is the sum of some two preceding numbers. -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ n ≥ 2, ∃ i j, i < n ∧ j < n ∧ a n = a i + a j)

/-- A number is composite if it's not prime and greater than 1. -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- The main theorem stating that there are infinitely many composite numbers
    in a special sequence. -/
theorem infinitely_many_composites (a : ℕ → ℕ) (h : SpecialSequence a) :
    ∀ N, ∃ n > N, IsComposite (a n) :=
  sorry

end NUMINAMATH_CALUDE_infinitely_many_composites_l1557_155726


namespace NUMINAMATH_CALUDE_triangle_properties_l1557_155722

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  t.c^2 + t.a * t.b = t.c * (t.a * Real.cos t.B - t.b * Real.cos t.A) + 2 * t.b^2

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : t.c = 2 * Real.sqrt 3) : 
  t.C = π / 3 ∧ 
  ∃ (x : ℝ), -2 * Real.sqrt 3 < x ∧ x < 2 * Real.sqrt 3 ∧ x = 4 * Real.sin t.B - t.a :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1557_155722


namespace NUMINAMATH_CALUDE_travel_time_difference_l1557_155798

/-- Proves that the difference in travel time between a 400-mile trip and a 360-mile trip,
    when traveling at a constant speed of 40 miles per hour, is 60 minutes. -/
theorem travel_time_difference (speed : ℝ) (dist1 : ℝ) (dist2 : ℝ) :
  speed = 40 → dist1 = 400 → dist2 = 360 →
  (dist1 / speed - dist2 / speed) * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_difference_l1557_155798


namespace NUMINAMATH_CALUDE_scale_division_theorem_l1557_155745

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 6 * 12 + 8

/-- Represents the number of parts the scale is divided into -/
def num_parts : ℕ := 4

/-- Represents the length of each part in inches -/
def part_length : ℕ := scale_length / num_parts

/-- Proves that each part of the scale is 20 inches (1 foot 8 inches) long -/
theorem scale_division_theorem : part_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_theorem_l1557_155745


namespace NUMINAMATH_CALUDE_diagonal_division_l1557_155739

/-- A regular polygon with 2018 vertices, labeled clockwise from 1 to 2018 -/
structure RegularPolygon2018 where
  vertices : Fin 2018

/-- The number of vertices between two given vertices in a clockwise direction -/
def verticesBetween (a b : Fin 2018) : ℕ :=
  if b.val ≥ a.val then
    b.val - a.val + 1
  else
    (2018 - a.val) + b.val + 1

/-- The result of drawing diagonals in the polygon -/
def diagonalResult (p : RegularPolygon2018) : Prop :=
  let polygon1 := verticesBetween 18 1018
  let polygon2 := verticesBetween 1018 2000
  let polygon3 := verticesBetween 2000 18 + 1  -- Adding 1 for vertex 1018
  polygon1 = 1001 ∧ polygon2 = 983 ∧ polygon3 = 38

theorem diagonal_division (p : RegularPolygon2018) : diagonalResult p := by
  sorry

end NUMINAMATH_CALUDE_diagonal_division_l1557_155739


namespace NUMINAMATH_CALUDE_petyas_friends_l1557_155729

/-- The number of stickers Petya gives to each friend in the first scenario -/
def stickers_per_friend_scenario1 : ℕ := 5

/-- The number of stickers Petya has left in the first scenario -/
def stickers_left_scenario1 : ℕ := 8

/-- The number of stickers Petya gives to each friend in the second scenario -/
def stickers_per_friend_scenario2 : ℕ := 6

/-- The number of additional stickers Petya needs in the second scenario -/
def additional_stickers_needed : ℕ := 11

/-- Petya's number of friends -/
def number_of_friends : ℕ := 19

theorem petyas_friends :
  (stickers_per_friend_scenario1 * number_of_friends + stickers_left_scenario1 =
   stickers_per_friend_scenario2 * number_of_friends - additional_stickers_needed) ∧
  (number_of_friends = 19) :=
by sorry

end NUMINAMATH_CALUDE_petyas_friends_l1557_155729


namespace NUMINAMATH_CALUDE_squirrel_solution_l1557_155775

/-- The number of walnuts the girl squirrel ate -/
def squirrel_problem (initial : ℕ) (boy_gathered : ℕ) (boy_dropped : ℕ) (girl_brought : ℕ) (final : ℕ) : ℕ :=
  initial + boy_gathered - boy_dropped + girl_brought - final

/-- Theorem stating the solution to the squirrel problem -/
theorem squirrel_solution : squirrel_problem 12 6 1 5 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_solution_l1557_155775


namespace NUMINAMATH_CALUDE_nancy_and_rose_bracelets_l1557_155773

/-- The number of beads required for each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has -/
def nancy_pearl_beads : ℕ := nancy_metal_beads + 20

/-- The number of crystal beads Rose has -/
def rose_crystal_beads : ℕ := 20

/-- The number of stone beads Rose has -/
def rose_stone_beads : ℕ := 2 * rose_crystal_beads

/-- The total number of beads Nancy and Rose have -/
def total_beads : ℕ := nancy_metal_beads + nancy_pearl_beads + rose_crystal_beads + rose_stone_beads

/-- The number of bracelets Nancy and Rose can make -/
def bracelets_made : ℕ := total_beads / beads_per_bracelet

theorem nancy_and_rose_bracelets : bracelets_made = 20 := by
  sorry

end NUMINAMATH_CALUDE_nancy_and_rose_bracelets_l1557_155773


namespace NUMINAMATH_CALUDE_average_initial_price_is_54_l1557_155759

/-- Represents the price and quantity of fruit. -/
structure FruitInfo where
  applePrice : ℕ
  orangePrice : ℕ
  totalFruit : ℕ
  orangesPutBack : ℕ
  avgPriceKept : ℕ

/-- Calculates the average price of initially selected fruit. -/
def averageInitialPrice (info : FruitInfo) : ℚ :=
  let apples := info.totalFruit - (info.totalFruit - info.orangesPutBack - 
    (info.avgPriceKept * (info.totalFruit - info.orangesPutBack) - 
    info.orangePrice * (info.totalFruit - info.orangesPutBack - info.orangesPutBack)) / 
    (info.applePrice - info.orangePrice))
  let oranges := info.totalFruit - apples
  (info.applePrice * apples + info.orangePrice * oranges) / info.totalFruit

/-- Theorem stating that the average initial price is 54 cents. -/
theorem average_initial_price_is_54 (info : FruitInfo) 
    (h1 : info.applePrice = 40)
    (h2 : info.orangePrice = 60)
    (h3 : info.totalFruit = 10)
    (h4 : info.orangesPutBack = 6)
    (h5 : info.avgPriceKept = 45) :
  averageInitialPrice info = 54 := by
  sorry

end NUMINAMATH_CALUDE_average_initial_price_is_54_l1557_155759


namespace NUMINAMATH_CALUDE_charity_ticket_sales_l1557_155779

theorem charity_ticket_sales (full_price_tickets half_price_tickets : ℕ) 
  (full_price half_price : ℚ) : 
  full_price_tickets + half_price_tickets = 160 →
  full_price_tickets * full_price + half_price_tickets * half_price = 2400 →
  half_price = full_price / 2 →
  full_price_tickets * full_price = 960 := by
sorry

end NUMINAMATH_CALUDE_charity_ticket_sales_l1557_155779


namespace NUMINAMATH_CALUDE_eggs_taken_l1557_155755

theorem eggs_taken (initial_eggs : ℕ) (remaining_eggs : ℕ) (h1 : initial_eggs = 47) (h2 : remaining_eggs = 42) :
  initial_eggs - remaining_eggs = 5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_taken_l1557_155755


namespace NUMINAMATH_CALUDE_cookie_sharing_proof_l1557_155730

/-- The number of people sharing cookies baked by Beth --/
def number_of_people : ℕ :=
  let batches : ℕ := 4
  let dozens_per_batch : ℕ := 2
  let cookies_per_dozen : ℕ := 12
  let cookies_per_person : ℕ := 6
  let total_cookies : ℕ := batches * dozens_per_batch * cookies_per_dozen
  total_cookies / cookies_per_person

/-- Proof that the number of people sharing the cookies is 16 --/
theorem cookie_sharing_proof : number_of_people = 16 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sharing_proof_l1557_155730


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l1557_155791

/-- 
Given a right triangle with sides 6, 8, and 10, and an inscribed square with side length a 
where one vertex of the square coincides with the right angle of the triangle,
and an isosceles right triangle with legs 6 and 6, and an inscribed square with side length b 
where one side of the square lies on the hypotenuse of the triangle,
the ratio of a to b is √2/3.
-/
theorem inscribed_squares_ratio : 
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), x + y = 10 ∧ x^2 + y^2 = 10^2 ∧ x * y = 48 ∧ a * (x - a) = a * (y - a)) →
  (∃ (z : ℝ), z^2 = 72 ∧ b + b = z) →
  a / b = Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l1557_155791


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l1557_155732

def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

theorem derivative_f_at_one :
  deriv f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l1557_155732


namespace NUMINAMATH_CALUDE_monochromatic_solution_exists_l1557_155754

def Color := Bool

def NumberSet : Set Nat := {1, 2, 3, 4, 5}

def Coloring := Nat → Color

theorem monochromatic_solution_exists (c : Coloring) : 
  ∃ (x y z : Nat), x ∈ NumberSet ∧ y ∈ NumberSet ∧ z ∈ NumberSet ∧ 
  x + y = z ∧ c x = c y ∧ c y = c z :=
sorry

end NUMINAMATH_CALUDE_monochromatic_solution_exists_l1557_155754


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1557_155712

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → x * y > 0) ∧
  (∃ x y : ℝ, x * y > 0 ∧ ¬(x > 0 ∧ y > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1557_155712


namespace NUMINAMATH_CALUDE_remainder_theorem_l1557_155758

theorem remainder_theorem (r : ℤ) : (r^11 - 1) % (r - 2) = 2047 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1557_155758


namespace NUMINAMATH_CALUDE_sequence_2017th_term_l1557_155761

theorem sequence_2017th_term (a : ℕ+ → ℚ) 
  (h1 : a 1 = 0)
  (h2 : ∀ n : ℕ+, n ≥ 2 → (1 / (1 - a n) - 1 / (1 - a (n - 1)) = 1)) :
  a 2017 = 2016 / 2017 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2017th_term_l1557_155761


namespace NUMINAMATH_CALUDE_weak_to_strong_ratio_l1557_155708

/-- Represents the amount of coffee used for different strengths --/
structure CoffeeUsage where
  weak_per_cup : ℕ
  strong_per_cup : ℕ
  cups_each : ℕ
  total_tablespoons : ℕ

/-- Theorem stating the ratio of weak to strong coffee usage --/
theorem weak_to_strong_ratio (c : CoffeeUsage) 
  (h1 : c.weak_per_cup = 1)
  (h2 : c.strong_per_cup = 2)
  (h3 : c.cups_each = 12)
  (h4 : c.total_tablespoons = 36) :
  (c.weak_per_cup * c.cups_each) / (c.strong_per_cup * c.cups_each) = 1 / 2 := by
  sorry

#check weak_to_strong_ratio

end NUMINAMATH_CALUDE_weak_to_strong_ratio_l1557_155708


namespace NUMINAMATH_CALUDE_f_increasing_l1557_155707

def f (x : ℝ) : ℝ := x^2 + 4*x + 3

theorem f_increasing : ∀ x y, 0 < x → x < y → f x < f y := by sorry

end NUMINAMATH_CALUDE_f_increasing_l1557_155707


namespace NUMINAMATH_CALUDE_number_of_digits_in_N_l1557_155764

theorem number_of_digits_in_N : ∃ (N : ℕ), 
  N = 2^12 * 5^8 ∧ (Nat.log 10 N + 1 = 10) := by sorry

end NUMINAMATH_CALUDE_number_of_digits_in_N_l1557_155764


namespace NUMINAMATH_CALUDE_ball_count_proof_l1557_155738

theorem ball_count_proof (white green yellow red purple : ℕ)
  (h1 : white = 50)
  (h2 : green = 30)
  (h3 : yellow = 8)
  (h4 : red = 9)
  (h5 : purple = 3)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 88/100) :
  white + green + yellow + red + purple = 100 := by
sorry

end NUMINAMATH_CALUDE_ball_count_proof_l1557_155738


namespace NUMINAMATH_CALUDE_connected_vessels_equilibrium_l1557_155700

/-- Represents the final levels of liquids in two connected vessels after equilibrium -/
def FinalLevels (H : ℝ) : ℝ × ℝ :=
  (0.69 * H, H)

/-- Proves that the given final levels are correct for the connected vessels problem -/
theorem connected_vessels_equilibrium 
  (H : ℝ) 
  (h_positive : H > 0) 
  (ρ_water : ℝ) 
  (ρ_gasoline : ℝ) 
  (h_water_density : ρ_water = 1000) 
  (h_gasoline_density : ρ_gasoline = 600) 
  (h_initial_level : ℝ) 
  (h_initial : h_initial = 0.9 * H) 
  (h_tap_height : ℝ) 
  (h_tap : h_tap_height = 0.2 * H) : 
  FinalLevels H = (0.69 * H, H) :=
sorry

#check connected_vessels_equilibrium

end NUMINAMATH_CALUDE_connected_vessels_equilibrium_l1557_155700


namespace NUMINAMATH_CALUDE_sum_of_exponents_is_eight_l1557_155735

/-- Represents the exponents of variables in a simplified cube root expression -/
structure SimplifiedCubeRootExponents where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Simplifies the cube root of 40a^6b^7c^14 and returns the exponents of variables outside the radical -/
def simplify_cube_root : SimplifiedCubeRootExponents := {
  a := 2,
  b := 2,
  c := 4
}

/-- The sum of exponents outside the radical after simplifying ∛(40a^6b^7c^14) is 8 -/
theorem sum_of_exponents_is_eight :
  (simplify_cube_root.a + simplify_cube_root.b + simplify_cube_root.c) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_is_eight_l1557_155735


namespace NUMINAMATH_CALUDE_prob_at_most_one_red_l1557_155705

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def red_balls : ℕ := 2
def drawn_balls : ℕ := 3

theorem prob_at_most_one_red :
  (1 : ℚ) - (Nat.choose white_balls 1 * Nat.choose red_balls 2 : ℚ) / Nat.choose total_balls drawn_balls = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_red_l1557_155705


namespace NUMINAMATH_CALUDE_jack_emails_l1557_155784

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 3

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 5

/-- The total number of emails Jack received in the day -/
def total_emails : ℕ := morning_emails + afternoon_emails

theorem jack_emails : total_emails = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_emails_l1557_155784


namespace NUMINAMATH_CALUDE_xiao_ming_foot_length_l1557_155720

/-- The relationship between a person's height and foot length -/
def height_foot_relation (h d : ℝ) : Prop := h = 7 * d

/-- Xiao Ming's height in cm -/
def xiao_ming_height : ℝ := 171.5

theorem xiao_ming_foot_length :
  ∃ d : ℝ, height_foot_relation xiao_ming_height d ∧ d = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_foot_length_l1557_155720


namespace NUMINAMATH_CALUDE_dryer_cost_l1557_155782

theorem dryer_cost (washer dryer : ℕ) : 
  washer + dryer = 600 →
  washer = 3 * dryer →
  dryer = 150 := by
sorry

end NUMINAMATH_CALUDE_dryer_cost_l1557_155782


namespace NUMINAMATH_CALUDE_fourth_person_height_l1557_155752

/-- Given four people with heights in increasing order, prove the height of the fourth person. -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- Heights in increasing order
  h₂ - h₁ = 2 →  -- Difference between 1st and 2nd
  h₃ - h₂ = 2 →  -- Difference between 2nd and 3rd
  h₄ - h₃ = 6 →  -- Difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 79 →  -- Average height
  h₄ = 85 :=
by sorry

end NUMINAMATH_CALUDE_fourth_person_height_l1557_155752


namespace NUMINAMATH_CALUDE_probability_standard_deck_l1557_155777

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (face_cards : Nat)
  (number_cards : Nat)

/-- Define a standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    face_cards := 12,
    number_cards := 40 }

/-- Calculate the probability of drawing a face card first and a number card second -/
def probability_face_then_number (d : Deck) : Rat :=
  (d.face_cards * d.number_cards : Rat) / (d.total_cards * (d.total_cards - 1))

/-- Theorem stating the probability for a standard deck -/
theorem probability_standard_deck :
  probability_face_then_number standard_deck = 40 / 221 := by
  sorry


end NUMINAMATH_CALUDE_probability_standard_deck_l1557_155777


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1557_155731

theorem largest_angle_in_special_triangle (A B C : Real) (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = π) (h3 : Real.sin A / Real.sin B = 3 / 5)
  (h4 : Real.sin B / Real.sin C = 5 / 7) :
  max A (max B C) = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1557_155731


namespace NUMINAMATH_CALUDE_probability_multiple_of_7_l1557_155793

def is_multiple_of_7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

def count_pairs (n : ℕ) : ℕ := n.choose 2

theorem probability_multiple_of_7 : 
  let total_pairs := count_pairs 100
  let valid_pairs := total_pairs - count_pairs (100 - 14)
  (valid_pairs : ℚ) / total_pairs = 259 / 990 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_of_7_l1557_155793


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l1557_155772

theorem largest_divisor_five_consecutive_integers : 
  ∃ (k : ℕ), k = 60 ∧ 
  (∀ (n : ℤ), ∃ (m : ℤ), (n * (n+1) * (n+2) * (n+3) * (n+4)) = m * k) ∧
  (∀ (l : ℕ), l > k → ∃ (n : ℤ), ∀ (m : ℤ), (n * (n+1) * (n+2) * (n+3) * (n+4)) ≠ m * l) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l1557_155772


namespace NUMINAMATH_CALUDE_least_multiple_25_with_digit_product_125_l1557_155714

def is_multiple_of_25 (n : ℕ) : Prop := ∃ k : ℕ, n = 25 * k

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  let digits := n.digits 10
  digits.prod

theorem least_multiple_25_with_digit_product_125 :
  ∀ n : ℕ, n > 0 → is_multiple_of_25 n → digit_product n = 125 → n ≥ 555 :=
sorry

end NUMINAMATH_CALUDE_least_multiple_25_with_digit_product_125_l1557_155714


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1557_155792

theorem diophantine_equation_solution :
  ∃ (a b c d : ℕ+), 4^(a : ℕ) * 5^(b : ℕ) - 3^(c : ℕ) * 11^(d : ℕ) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1557_155792


namespace NUMINAMATH_CALUDE_smallest_m_for_partition_l1557_155734

theorem smallest_m_for_partition (r : ℕ+) :
  (∃ (m : ℕ+), ∀ (A : Fin r → Set ℕ),
    (∀ (i j : Fin r), i ≠ j → A i ∩ A j = ∅) →
    (⋃ (i : Fin r), A i) = Finset.range m →
    (∃ (k : Fin r) (a b : ℕ), a ∈ A k ∧ b ∈ A k ∧ a ≠ 0 ∧ 1 ≤ b / a ∧ b / a ≤ 1 + 1 / 2022)) ∧
  (∀ (m : ℕ+), m < 2023 * r →
    ¬(∀ (A : Fin r → Set ℕ),
      (∀ (i j : Fin r), i ≠ j → A i ∩ A j = ∅) →
      (⋃ (i : Fin r), A i) = Finset.range m →
      (∃ (k : Fin r) (a b : ℕ), a ∈ A k ∧ b ∈ A k ∧ a ≠ 0 ∧ 1 ≤ b / a ∧ b / a ≤ 1 + 1 / 2022))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_partition_l1557_155734


namespace NUMINAMATH_CALUDE_range_of_a_l1557_155715

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → a ∈ Set.Ioo (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1557_155715


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1557_155748

theorem cubic_root_sum (a b c : ℝ) : 
  (15 * a^3 - 30 * a^2 + 20 * a - 2 = 0) →
  (15 * b^3 - 30 * b^2 + 20 * b - 2 = 0) →
  (15 * c^3 - 30 * c^2 + 20 * c - 2 = 0) →
  (0 < a ∧ a < 1) →
  (0 < b ∧ b < 1) →
  (0 < c ∧ c < 1) →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1557_155748


namespace NUMINAMATH_CALUDE_tea_mixture_price_l1557_155799

/-- The price of the first variety of tea in Rs per kg -/
def price_first : ℝ := 126

/-- The price of the second variety of tea in Rs per kg -/
def price_second : ℝ := 135

/-- The price of the third variety of tea in Rs per kg -/
def price_third : ℝ := 175.5

/-- The price of the mixture in Rs per kg -/
def price_mixture : ℝ := 153

/-- The ratio of the first variety in the mixture -/
def ratio_first : ℝ := 1

/-- The ratio of the second variety in the mixture -/
def ratio_second : ℝ := 1

/-- The ratio of the third variety in the mixture -/
def ratio_third : ℝ := 2

/-- The total ratio sum -/
def ratio_total : ℝ := ratio_first + ratio_second + ratio_third

theorem tea_mixture_price :
  (ratio_first * price_first + ratio_second * price_second + ratio_third * price_third) / ratio_total = price_mixture := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l1557_155799


namespace NUMINAMATH_CALUDE_celine_collected_ten_erasers_l1557_155749

/-- The number of erasers collected by each person -/
structure EraserCollection where
  gabriel : ℕ
  celine : ℕ
  julian : ℕ
  erica : ℕ
  david : ℕ

/-- The conditions of the eraser collection problem -/
def valid_collection (ec : EraserCollection) : Prop :=
  ec.celine = 2 * ec.gabriel ∧
  ec.julian = 2 * ec.celine ∧
  ec.erica = 3 * ec.julian ∧
  ec.david = 5 * ec.erica ∧
  ec.gabriel ≥ 1 ∧ ec.celine ≥ 1 ∧ ec.julian ≥ 1 ∧ ec.erica ≥ 1 ∧ ec.david ≥ 1 ∧
  ec.gabriel + ec.celine + ec.julian + ec.erica + ec.david = 380

/-- The theorem stating that Celine collected 10 erasers -/
theorem celine_collected_ten_erasers (ec : EraserCollection) 
  (h : valid_collection ec) : ec.celine = 10 := by
  sorry

end NUMINAMATH_CALUDE_celine_collected_ten_erasers_l1557_155749


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1557_155769

theorem simplify_and_evaluate (x : ℝ) (h : x = 1) : 
  (4 / (x^2 - 4)) / (2 / (x - 2)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1557_155769


namespace NUMINAMATH_CALUDE_monkey_percentage_after_events_l1557_155780

/-- Represents the counts of animals in the tree --/
structure AnimalCounts where
  monkeys : ℕ
  birds : ℕ
  squirrels : ℕ
  cats : ℕ

/-- Calculates the total number of animals --/
def totalAnimals (counts : AnimalCounts) : ℕ :=
  counts.monkeys + counts.birds + counts.squirrels + counts.cats

/-- Applies the events described in the problem --/
def applyEvents (initial : AnimalCounts) : AnimalCounts :=
  { monkeys := initial.monkeys,
    birds := initial.birds - 2 - 2,  -- 2 eaten by monkeys, 2 chased away
    squirrels := initial.squirrels - 1,  -- 1 chased away
    cats := initial.cats }

/-- Calculates the percentage of monkeys after the events --/
def monkeyPercentage (initial : AnimalCounts) : ℚ :=
  let final := applyEvents initial
  (final.monkeys : ℚ) / (totalAnimals final : ℚ) * 100

theorem monkey_percentage_after_events :
  let initial : AnimalCounts := { monkeys := 6, birds := 9, squirrels := 3, cats := 5 }
  monkeyPercentage initial = 100/3 := by sorry

end NUMINAMATH_CALUDE_monkey_percentage_after_events_l1557_155780


namespace NUMINAMATH_CALUDE_cross_tangential_cubic_cross_tangential_sine_cross_tangential_tangent_l1557_155704

-- Define the concept of cross-tangential intersection
def cross_tangential_intersection (l : ℝ → ℝ) (c : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := p
  -- Condition (i): The line l is tangent to the curve C at the point P(x₀, y₀)
  (deriv c x₀ = deriv l x₀) ∧
  -- Condition (ii): The curve C lies on both sides of the line l near point P
  (∃ δ > 0, ∀ x ∈ Set.Ioo (x₀ - δ) (x₀ + δ),
    (x < x₀ → c x < l x) ∧ (x > x₀ → c x > l x) ∨
    (x < x₀ → c x > l x) ∧ (x > x₀ → c x < l x))

-- Statement 1
theorem cross_tangential_cubic :
  cross_tangential_intersection (λ _ => 0) (λ x => x^3) (0, 0) :=
sorry

-- Statement 3
theorem cross_tangential_sine :
  cross_tangential_intersection (λ x => x) Real.sin (0, 0) :=
sorry

-- Statement 4
theorem cross_tangential_tangent :
  cross_tangential_intersection (λ x => x) Real.tan (0, 0) :=
sorry

end NUMINAMATH_CALUDE_cross_tangential_cubic_cross_tangential_sine_cross_tangential_tangent_l1557_155704


namespace NUMINAMATH_CALUDE_driver_distance_theorem_l1557_155753

/-- Calculates the total distance traveled by a driver given their speed and driving durations. -/
def total_distance_traveled (speed : ℝ) (first_duration second_duration : ℝ) : ℝ :=
  speed * (first_duration + second_duration)

/-- Theorem stating that a driver traveling at 60 mph for 4 hours and 9 hours will cover 780 miles. -/
theorem driver_distance_theorem :
  let speed := 60
  let first_duration := 4
  let second_duration := 9
  total_distance_traveled speed first_duration second_duration = 780 := by
  sorry

#check driver_distance_theorem

end NUMINAMATH_CALUDE_driver_distance_theorem_l1557_155753


namespace NUMINAMATH_CALUDE_square_sum_equals_four_l1557_155795

theorem square_sum_equals_four (x y : ℝ) (h : x^2 + y^2 + x^2*y^2 - 4*x*y + 1 = 0) : 
  (x + y)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_four_l1557_155795


namespace NUMINAMATH_CALUDE_members_neither_subject_count_l1557_155706

/-- The number of club members taking neither computer science nor robotics -/
def membersNeitherSubject (totalMembers csMembers roboticsMembers bothSubjects : ℕ) : ℕ :=
  totalMembers - (csMembers + roboticsMembers - bothSubjects)

/-- Theorem stating the number of club members taking neither subject -/
theorem members_neither_subject_count :
  membersNeitherSubject 150 80 70 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_members_neither_subject_count_l1557_155706


namespace NUMINAMATH_CALUDE_g_is_odd_l1557_155701

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define g as f(x) - f(-x)
def g (x : ℝ) : ℝ := f x - f (-x)

-- Theorem: g is an odd function
theorem g_is_odd : ∀ x : ℝ, g f (-x) = -(g f x) := by
  sorry

end NUMINAMATH_CALUDE_g_is_odd_l1557_155701


namespace NUMINAMATH_CALUDE_operations_correct_l1557_155717

-- Define the operations
def operation3 (x : ℝ) : Prop := x ≠ 0 → x^6 / x^3 = x^3
def operation4 (x : ℝ) : Prop := (x^3)^2 = x^6

-- Theorem stating that both operations are correct
theorem operations_correct : 
  (∀ x : ℝ, operation3 x) ∧ (∀ x : ℝ, operation4 x) := by sorry

end NUMINAMATH_CALUDE_operations_correct_l1557_155717


namespace NUMINAMATH_CALUDE_wage_increase_l1557_155736

theorem wage_increase (original_wage : ℝ) (increase_percentage : ℝ) (new_wage : ℝ) :
  original_wage = 20 →
  increase_percentage = 40 →
  new_wage = original_wage * (1 + increase_percentage / 100) →
  new_wage = 28 := by
sorry

end NUMINAMATH_CALUDE_wage_increase_l1557_155736


namespace NUMINAMATH_CALUDE_reasoning_is_analogical_l1557_155746

/-- A type representing different reasoning methods -/
inductive ReasoningMethod
  | Inductive
  | Analogical
  | Deductive
  | None

/-- A circle with radius R -/
structure Circle (R : ℝ) where
  radius : R > 0

/-- A rectangle inscribed in a circle -/
structure InscribedRectangle (R : ℝ) extends Circle R where
  width : ℝ
  height : ℝ
  inscribed : width^2 + height^2 ≤ 4 * R^2

/-- A sphere with radius R -/
structure Sphere (R : ℝ) where
  radius : R > 0

/-- A rectangular solid inscribed in a sphere -/
structure InscribedRectangularSolid (R : ℝ) extends Sphere R where
  length : ℝ
  width : ℝ
  height : ℝ
  inscribed : length^2 + width^2 + height^2 ≤ 4 * R^2

/-- Theorem about maximum area rectangle in a circle -/
axiom max_area_square_in_circle (R : ℝ) :
  ∀ (rect : InscribedRectangle R), rect.width * rect.height ≤ 2 * R^2

/-- The reasoning method used to deduce the theorem about cubes in spheres -/
def reasoning_method : ReasoningMethod := by sorry

/-- The main theorem stating that the reasoning method is analogical -/
theorem reasoning_is_analogical :
  reasoning_method = ReasoningMethod.Analogical := by sorry

end NUMINAMATH_CALUDE_reasoning_is_analogical_l1557_155746


namespace NUMINAMATH_CALUDE_all_zeros_not_pronounced_l1557_155737

/-- Represents a natural number in decimal notation --/
def DecimalNumber : Type := List Nat

/-- Rules for reading integers --/
structure ReadingRules where
  readHighestToLowest : Bool
  skipEndZeros : Bool
  readConsecutiveZerosAsOne : Bool

/-- Function to determine if a digit should be pronounced --/
def shouldPronounce (rules : ReadingRules) (num : DecimalNumber) (index : Nat) : Bool :=
  sorry

/-- The number 3,406,000 in decimal notation --/
def number : DecimalNumber := [3, 4, 0, 6, 0, 0, 0]

/-- The rules for reading integers as described in the problem --/
def integerReadingRules : ReadingRules := {
  readHighestToLowest := true,
  skipEndZeros := true,
  readConsecutiveZerosAsOne := true
}

/-- Theorem stating that all zeros in 3,406,000 are not pronounced --/
theorem all_zeros_not_pronounced : 
  ∀ i, i ∈ [2, 4, 5, 6] → ¬(shouldPronounce integerReadingRules number i) :=
sorry

end NUMINAMATH_CALUDE_all_zeros_not_pronounced_l1557_155737


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l1557_155743

theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = !![1, 4; -2, -7] →
  (A^3)⁻¹ = !![41, 144; -72, -247] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l1557_155743


namespace NUMINAMATH_CALUDE_ninth_root_of_unity_l1557_155767

theorem ninth_root_of_unity (y : ℂ) : 
  y = Complex.exp (2 * Real.pi * I / 9) → y^9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ninth_root_of_unity_l1557_155767


namespace NUMINAMATH_CALUDE_total_non_defective_engines_l1557_155760

/-- Represents a batch of engines with their total count and defect rate -/
structure Batch where
  total : ℕ
  defect_rate : ℚ
  defect_rate_valid : 0 ≤ defect_rate ∧ defect_rate ≤ 1

/-- Calculates the number of non-defective engines in a batch -/
def non_defective (b : Batch) : ℚ :=
  b.total * (1 - b.defect_rate)

/-- The list of batches with their respective data -/
def batches : List Batch := [
  ⟨140, 12/100, by norm_num⟩,
  ⟨150, 18/100, by norm_num⟩,
  ⟨170, 22/100, by norm_num⟩,
  ⟨180, 28/100, by norm_num⟩,
  ⟨190, 32/100, by norm_num⟩,
  ⟨210, 36/100, by norm_num⟩,
  ⟨220, 41/100, by norm_num⟩
]

/-- The theorem stating the total number of non-defective engines -/
theorem total_non_defective_engines :
  Int.floor (batches.map non_defective).sum = 902 := by
  sorry

end NUMINAMATH_CALUDE_total_non_defective_engines_l1557_155760


namespace NUMINAMATH_CALUDE_apple_eating_contest_l1557_155710

def classroom (n : ℕ) (total_apples : ℕ) (aaron_apples : ℕ) (zeb_apples : ℕ) : Prop :=
  n = 8 ∧
  total_apples > 20 ∧
  ∀ student, student ≠ aaron_apples → aaron_apples ≥ student ∧
  ∀ student, student ≠ zeb_apples → student ≥ zeb_apples

theorem apple_eating_contest (n : ℕ) (total_apples : ℕ) (aaron_apples : ℕ) (zeb_apples : ℕ) 
  (h : classroom n total_apples aaron_apples zeb_apples) : 
  aaron_apples - zeb_apples = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_eating_contest_l1557_155710
