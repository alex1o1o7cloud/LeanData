import Mathlib

namespace NUMINAMATH_CALUDE_cafeteria_extra_apples_l1904_190483

/-- The number of extra apples in the cafeteria -/
def extra_apples (red_apples green_apples students_wanting_fruit : ℕ) : ℕ :=
  red_apples + green_apples - students_wanting_fruit

/-- Theorem: The cafeteria ends up with 32 extra apples -/
theorem cafeteria_extra_apples :
  extra_apples 25 17 10 = 32 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_extra_apples_l1904_190483


namespace NUMINAMATH_CALUDE_intersection_condition_l1904_190457

/-- Two curves intersect at exactly two distinct points -/
def HasTwoDistinctIntersections (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ (f x₁ y₁ ∧ g x₁ y₁) ∧ (f x₂ y₂ ∧ g x₂ y₂) ∧
  ∀ (x y : ℝ), (f x y ∧ g x y) → ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))

/-- The circle equation -/
def Circle (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = b^2

/-- The parabola equation -/
def Parabola (b : ℝ) (x y : ℝ) : Prop := y = -x^2 + b

theorem intersection_condition (b : ℝ) :
  HasTwoDistinctIntersections (Circle b) (Parabola b) ↔ b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l1904_190457


namespace NUMINAMATH_CALUDE_min_value_expression_l1904_190412

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≤ b + c) (hbc : b ≤ a + c) (hca : c ≤ a + b) :
  c / (a + b) + b / c ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1904_190412


namespace NUMINAMATH_CALUDE_dogwood_trees_in_other_part_l1904_190495

/-- The number of dogwood trees in the first part of the park -/
def trees_in_first_part : ℝ := 5.0

/-- The number of trees park workers plan to cut down -/
def planned_trees_to_cut : ℝ := 7.0

/-- The number of park workers on the job -/
def park_workers : ℝ := 8.0

/-- The number of dogwood trees left in the park after the work is done -/
def trees_left_after_work : ℝ := 2.0

/-- The number of dogwood trees in the other part of the park -/
def trees_in_other_part : ℝ := trees_left_after_work

theorem dogwood_trees_in_other_part : 
  trees_in_other_part = 2.0 :=
by sorry

end NUMINAMATH_CALUDE_dogwood_trees_in_other_part_l1904_190495


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1904_190480

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^2 + 3*x*y - 2*y^2 = 122 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1904_190480


namespace NUMINAMATH_CALUDE_prism_volume_l1904_190428

theorem prism_volume (a b c h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a * b = 100 → a * h = 50 → b * h = 40 → h = 10 →
  a * b * h = 200 := by sorry

end NUMINAMATH_CALUDE_prism_volume_l1904_190428


namespace NUMINAMATH_CALUDE_smallest_k_for_800_digit_sum_l1904_190475

/-- Represents a number consisting of k digits of 7 -/
def seventySevenNumber (k : ℕ) : ℕ :=
  (7 * (10^k - 1)) / 9

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved -/
theorem smallest_k_for_800_digit_sum :
  ∃ k : ℕ, k = 88 ∧
  (∀ m : ℕ, m < k → sumOfDigits (5 * seventySevenNumber m) < 800) ∧
  sumOfDigits (5 * seventySevenNumber k) = 800 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_800_digit_sum_l1904_190475


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l1904_190439

theorem parallel_lines_m_value (x y : ℝ) :
  (∀ x y, 2*x + 3*y + 1 = 0 ↔ m*x + 6*y - 5 = 0) →
  m = 4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l1904_190439


namespace NUMINAMATH_CALUDE_power_of_product_l1904_190423

theorem power_of_product (a b : ℝ) (n : ℕ) : (a * b) ^ n = a ^ n * b ^ n := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1904_190423


namespace NUMINAMATH_CALUDE_item_list_price_l1904_190473

theorem item_list_price : ∃ (list_price : ℝ), 
  list_price > 0 ∧
  0.15 * (list_price - 15) = 0.25 * (list_price - 25) ∧
  list_price = 40 := by
sorry

end NUMINAMATH_CALUDE_item_list_price_l1904_190473


namespace NUMINAMATH_CALUDE_quadratic_root_distance_translation_l1904_190491

/-- Given a quadratic function f(x) = ax^2 + bx + c with a > 0 and two distinct roots
    a distance p apart, the downward translation needed to make the distance
    between the roots 2p is (3b^2)/(4a) - 3c. -/
theorem quadratic_root_distance_translation
  (a b c p : ℝ)
  (h_a_pos : a > 0)
  (h_distinct_roots : ∃ x y, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0)
  (h_distance : ∃ x y, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ |x - y| = p) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := a * x^2 + b * x + (c - ((3 * b^2) / (4 * a) - 3 * c))
  ∃ x y, x ≠ y ∧ g x = 0 ∧ g y = 0 ∧ |x - y| = 2 * p :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_distance_translation_l1904_190491


namespace NUMINAMATH_CALUDE_wrapping_paper_cost_l1904_190453

-- Define the problem parameters
def shirtBoxesPerRoll : ℕ := 5
def xlBoxesPerRoll : ℕ := 3
def totalShirtBoxes : ℕ := 20
def totalXlBoxes : ℕ := 12
def totalCost : ℚ := 32

-- Define the theorem
theorem wrapping_paper_cost :
  let rollsForShirtBoxes := totalShirtBoxes / shirtBoxesPerRoll
  let rollsForXlBoxes := totalXlBoxes / xlBoxesPerRoll
  let totalRolls := rollsForShirtBoxes + rollsForXlBoxes
  totalCost / totalRolls = 4 := by sorry

end NUMINAMATH_CALUDE_wrapping_paper_cost_l1904_190453


namespace NUMINAMATH_CALUDE_animal_shelter_cats_l1904_190459

theorem animal_shelter_cats (dogs : ℕ) (initial_ratio_dogs initial_ratio_cats : ℕ) 
  (final_ratio_dogs final_ratio_cats : ℕ) (additional_cats : ℕ) : 
  dogs = 75 →
  initial_ratio_dogs = 15 →
  initial_ratio_cats = 7 →
  final_ratio_dogs = 15 →
  final_ratio_cats = 11 →
  (dogs : ℚ) / (dogs * initial_ratio_cats / initial_ratio_dogs : ℚ) = 
    initial_ratio_dogs / initial_ratio_cats →
  (dogs : ℚ) / (dogs * initial_ratio_cats / initial_ratio_dogs + additional_cats : ℚ) = 
    final_ratio_dogs / final_ratio_cats →
  additional_cats = 20 := by
  sorry

end NUMINAMATH_CALUDE_animal_shelter_cats_l1904_190459


namespace NUMINAMATH_CALUDE_min_value_theorem_l1904_190458

theorem min_value_theorem (m n p x y z : ℝ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0)
  (hmnp : m * n * p = 8)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxyz : x * y * z = 8) :
  ∃ (min : ℝ), min = 12 + 4 * (m + n + p) ∧
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a * b * c = 8 →
    a^2 + b^2 + c^2 + m*a*b + n*a*c + p*b*c ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1904_190458


namespace NUMINAMATH_CALUDE_bookArrangements_eq_48_l1904_190438

/-- The number of ways to arrange 3 different math books and 2 different Chinese books in a row,
    with the Chinese books placed next to each other. -/
def bookArrangements : ℕ :=
  (Nat.factorial 4) * (Nat.factorial 2)

/-- The total number of arrangements is 48. -/
theorem bookArrangements_eq_48 : bookArrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_bookArrangements_eq_48_l1904_190438


namespace NUMINAMATH_CALUDE_total_stars_is_580_l1904_190429

/-- The number of stars needed to fill all bottles Kyle bought -/
def total_stars : ℕ :=
  let type_a_initial := 3
  let type_a_later := 5
  let type_b := 4
  let type_c := 2
  let capacity_a := 30
  let capacity_b := 50
  let capacity_c := 70
  (type_a_initial + type_a_later) * capacity_a + type_b * capacity_b + type_c * capacity_c

theorem total_stars_is_580 : total_stars = 580 := by
  sorry

end NUMINAMATH_CALUDE_total_stars_is_580_l1904_190429


namespace NUMINAMATH_CALUDE_textbook_selection_ways_l1904_190404

/-- The number of ways to select textbooks from two categories -/
def select_textbooks (required : ℕ) (selective : ℕ) (total : ℕ) : ℕ :=
  (required.choose 1 * selective.choose 2) + (required.choose 2 * selective.choose 1)

/-- Theorem stating that selecting 3 textbooks from 2 required and 3 selective, 
    with at least one from each category, can be done in 9 ways -/
theorem textbook_selection_ways :
  select_textbooks 2 3 3 = 9 := by
  sorry

#eval select_textbooks 2 3 3

end NUMINAMATH_CALUDE_textbook_selection_ways_l1904_190404


namespace NUMINAMATH_CALUDE_uncertain_mushrooms_l1904_190420

/-- Given the total number of mushrooms, the number of safe mushrooms, and the relationship
    between safe and poisonous mushrooms, prove that the number of uncertain mushrooms is 5. -/
theorem uncertain_mushrooms (total : ℕ) (safe : ℕ) (poisonous : ℕ) :
  total = 32 →
  safe = 9 →
  poisonous = 2 * safe →
  total - (safe + poisonous) = 5 := by
  sorry

end NUMINAMATH_CALUDE_uncertain_mushrooms_l1904_190420


namespace NUMINAMATH_CALUDE_percent_to_decimal_four_percent_to_decimal_l1904_190410

theorem percent_to_decimal (x : ℚ) :
  x / 100 = x * (1 / 100) := by sorry

theorem four_percent_to_decimal :
  (4 : ℚ) / 100 = (4 : ℚ) * (1 / 100) ∧ (4 : ℚ) * (1 / 100) = 0.04 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_four_percent_to_decimal_l1904_190410


namespace NUMINAMATH_CALUDE_larger_number_proof_l1904_190451

theorem larger_number_proof (x y : ℝ) : 
  y > x → 4 * y = 3 * x → y - x = 12 → y = -36 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1904_190451


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1904_190463

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (h : a > b) :
  (a + b) / 2 > Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1904_190463


namespace NUMINAMATH_CALUDE_population_increase_theorem_l1904_190454

-- Define the increase factors
def increase_factor_0_to_2 : ℝ := 1.1
def increase_factor_2_to_5 : ℝ := 1.2

-- Define the total increase factor
def total_increase_factor : ℝ := increase_factor_0_to_2 * increase_factor_2_to_5

-- Theorem statement
theorem population_increase_theorem :
  (total_increase_factor - 1) * 100 = 32 := by
  sorry


end NUMINAMATH_CALUDE_population_increase_theorem_l1904_190454


namespace NUMINAMATH_CALUDE_a7_not_prime_l1904_190446

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Defines the sequence a_n -/
def a : ℕ → ℕ
  | 0 => 170  -- Initial value
  | n + 1 => a n + reverseDigits (a n)

/-- States that a_7 is not prime -/
theorem a7_not_prime : ¬ Nat.Prime (a 7) := by sorry

end NUMINAMATH_CALUDE_a7_not_prime_l1904_190446


namespace NUMINAMATH_CALUDE_third_month_sales_l1904_190445

def sales_1 : ℕ := 3435
def sales_2 : ℕ := 3927
def sales_4 : ℕ := 4230
def sales_5 : ℕ := 3562
def sales_6 : ℕ := 1991
def target_average : ℕ := 3500
def num_months : ℕ := 6

theorem third_month_sales :
  sales_1 + sales_2 + sales_4 + sales_5 + sales_6 + 3855 = target_average * num_months :=
by sorry

end NUMINAMATH_CALUDE_third_month_sales_l1904_190445


namespace NUMINAMATH_CALUDE_binary_to_octal_l1904_190466

-- Define the binary number
def binary_number : ℕ := 0b110101

-- Define the octal number
def octal_number : ℕ := 66

-- Theorem stating that the binary number is equal to the octal number
theorem binary_to_octal : binary_number = octal_number := by
  sorry

end NUMINAMATH_CALUDE_binary_to_octal_l1904_190466


namespace NUMINAMATH_CALUDE_find_c_l1904_190435

theorem find_c : ∃ c : ℝ, 
  (∃ n : ℤ, Int.floor c = n ∧ 3 * (n : ℝ)^2 + 12 * (n : ℝ) - 27 = 0) ∧ 
  (let frac := c - Int.floor c
   4 * frac^2 - 12 * frac + 5 = 0) ∧
  (0 ≤ c - Int.floor c ∧ c - Int.floor c < 1) ∧
  c = -8.5 := by
  sorry

end NUMINAMATH_CALUDE_find_c_l1904_190435


namespace NUMINAMATH_CALUDE_plant_arrangement_count_l1904_190416

/-- Represents the number of ways to arrange plants under lamps -/
def plant_arrangements : ℕ := 49

/-- The number of basil plants -/
def num_basil : ℕ := 3

/-- The number of aloe plants -/
def num_aloe : ℕ := 1

/-- The number of lamp colors -/
def num_lamp_colors : ℕ := 3

/-- The number of lamps per color -/
def lamps_per_color : ℕ := 2

/-- The total number of lamps -/
def total_lamps : ℕ := num_lamp_colors * lamps_per_color

/-- The total number of plants -/
def total_plants : ℕ := num_basil + num_aloe

theorem plant_arrangement_count :
  (num_basil = 3) →
  (num_aloe = 1) →
  (num_lamp_colors = 3) →
  (lamps_per_color = 2) →
  (total_plants ≤ total_lamps) →
  (plant_arrangements = 49) := by
  sorry

end NUMINAMATH_CALUDE_plant_arrangement_count_l1904_190416


namespace NUMINAMATH_CALUDE_mixed_gender_selections_l1904_190472

-- Define the number of male and female students
def num_male_students : Nat := 5
def num_female_students : Nat := 3

-- Define the total number of students
def total_students : Nat := num_male_students + num_female_students

-- Define the number of students to be selected
def students_to_select : Nat := 3

-- Function to calculate combinations
def combination (n : Nat) (r : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Theorem statement
theorem mixed_gender_selections :
  combination total_students students_to_select -
  combination num_male_students students_to_select -
  combination num_female_students students_to_select = 45 := by
  sorry


end NUMINAMATH_CALUDE_mixed_gender_selections_l1904_190472


namespace NUMINAMATH_CALUDE_ambulance_ride_cost_ambulance_cost_proof_l1904_190425

/-- Calculates the cost of an ambulance ride given a hospital bill breakdown -/
theorem ambulance_ride_cost (total_bill : ℝ) (medication_percentage : ℝ) 
  (overnight_percentage : ℝ) (food_cost : ℝ) : ℝ :=
  let medication_cost := medication_percentage * total_bill
  let remaining_after_medication := total_bill - medication_cost
  let overnight_cost := overnight_percentage * remaining_after_medication
  let ambulance_cost := total_bill - medication_cost - overnight_cost - food_cost
  ambulance_cost

/-- Proves that the ambulance ride cost is $1700 given specific bill details -/
theorem ambulance_cost_proof :
  ambulance_ride_cost 5000 0.5 0.25 175 = 1700 := by
  sorry

end NUMINAMATH_CALUDE_ambulance_ride_cost_ambulance_cost_proof_l1904_190425


namespace NUMINAMATH_CALUDE_sum_of_ratios_is_four_l1904_190444

/-- Given two nonconstant geometric sequences with common first term,
    if the difference of their third terms is four times the difference of their second terms,
    then the sum of their common ratios is 4. -/
theorem sum_of_ratios_is_four (k p r : ℝ) (h_k : k ≠ 0) (h_p : p ≠ 1) (h_r : r ≠ 1) 
    (h_eq : k * p^2 - k * r^2 = 4 * (k * p - k * r)) :
  p + r = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ratios_is_four_l1904_190444


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1904_190422

/-- Given a quadratic function y = ax^2 + bx + 3 where a and b are constants and a ≠ 0,
    prove that if (-m,0) and (3m,0) lie on the graph of this function, then b^2 + 4a = 0. -/
theorem quadratic_function_property (a b m : ℝ) (h_a : a ≠ 0) :
  (a * (-m)^2 + b * (-m) + 3 = 0) →
  (a * (3*m)^2 + b * (3*m) + 3 = 0) →
  b^2 + 4*a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1904_190422


namespace NUMINAMATH_CALUDE_ladybugs_on_tuesday_l1904_190489

theorem ladybugs_on_tuesday (monday_ladybugs : ℕ) (dots_per_ladybug : ℕ) (total_dots : ℕ) :
  monday_ladybugs = 8 →
  dots_per_ladybug = 6 →
  total_dots = 78 →
  ∃ tuesday_ladybugs : ℕ, 
    tuesday_ladybugs = 5 ∧
    total_dots = monday_ladybugs * dots_per_ladybug + tuesday_ladybugs * dots_per_ladybug :=
by sorry

end NUMINAMATH_CALUDE_ladybugs_on_tuesday_l1904_190489


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l1904_190478

def numbers : List ℕ := [18, 24, 42]

theorem arithmetic_mean_of_numbers :
  (numbers.sum / numbers.length : ℚ) = 28 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l1904_190478


namespace NUMINAMATH_CALUDE_cat_weight_ratio_l1904_190474

def megs_cat_weight : ℕ := 20
def weight_difference : ℕ := 8

def annes_cat_weight : ℕ := megs_cat_weight + weight_difference

theorem cat_weight_ratio :
  (megs_cat_weight : ℚ) / annes_cat_weight = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_cat_weight_ratio_l1904_190474


namespace NUMINAMATH_CALUDE_cubic_roots_property_l1904_190421

/-- 
Given a cubic polynomial x^3 + cx^2 + dx + 16c where c and d are nonzero integers,
if two of its roots coincide and all three roots are integers, then |cd| = 2560.
-/
theorem cubic_roots_property (c d : ℤ) (hc : c ≠ 0) (hd : d ≠ 0) : 
  (∃ p q : ℤ, (∀ x : ℝ, x^3 + c*x^2 + d*x + 16*c = (x - p)^2 * (x - q))) →
  |c*d| = 2560 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_property_l1904_190421


namespace NUMINAMATH_CALUDE_cloth_selling_price_l1904_190493

/-- Represents the selling price calculation for cloth --/
def total_selling_price (metres : ℕ) (cost_price : ℕ) (loss : ℕ) : ℕ :=
  metres * (cost_price - loss)

/-- Theorem stating the total selling price for the given conditions --/
theorem cloth_selling_price :
  total_selling_price 300 65 5 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l1904_190493


namespace NUMINAMATH_CALUDE_cone_lateral_surface_is_sector_l1904_190492

/-- Represents the possible shapes of an unfolded lateral surface of a cone -/
inductive UnfoldedShape
  | Triangle
  | Rectangle
  | Square
  | Sector

/-- Represents a cone -/
structure Cone where
  -- Add any necessary properties of a cone here

/-- The lateral surface of a cone when unfolded -/
def lateralSurface (c : Cone) : UnfoldedShape := sorry

/-- Theorem stating that the lateral surface of a cone, when unfolded, is shaped like a sector -/
theorem cone_lateral_surface_is_sector (c : Cone) : lateralSurface c = UnfoldedShape.Sector := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_is_sector_l1904_190492


namespace NUMINAMATH_CALUDE_opposite_sides_range_l1904_190400

theorem opposite_sides_range (m : ℝ) : 
  let A : ℝ × ℝ := (m, 2)
  let B : ℝ × ℝ := (2, m)
  let line (x y : ℝ) := x + 2*y - 4
  (line A.1 A.2) * (line B.1 B.2) < 0 → 0 < m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l1904_190400


namespace NUMINAMATH_CALUDE_extreme_values_range_c_l1904_190494

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 2*c*x^2 + x

-- Define the property of having extreme values
def has_extreme_values (c : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (3*x₁^2 - 4*c*x₁ + 1 = 0) ∧ 
    (3*x₂^2 - 4*c*x₂ + 1 = 0)

-- State the theorem
theorem extreme_values_range_c :
  ∀ c : ℝ, has_extreme_values c ↔ (c < -Real.sqrt 3 / 2 ∨ c > Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_range_c_l1904_190494


namespace NUMINAMATH_CALUDE_crabby_squido_ratio_l1904_190497

def squido_oysters : ℕ := 200
def total_oysters : ℕ := 600

def crabby_oysters : ℕ := total_oysters - squido_oysters

theorem crabby_squido_ratio : 
  (crabby_oysters : ℚ) / squido_oysters = 2 := by sorry

end NUMINAMATH_CALUDE_crabby_squido_ratio_l1904_190497


namespace NUMINAMATH_CALUDE_num_lineups_eq_2277_l1904_190465

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def special_players : ℕ := 3

/-- The number of possible lineups given the constraints -/
def num_lineups : ℕ :=
  3 * (Nat.choose (team_size - special_players) (lineup_size - 1)) +
  Nat.choose (team_size - special_players) lineup_size

/-- Theorem stating that the number of possible lineups is 2277 -/
theorem num_lineups_eq_2277 : num_lineups = 2277 := by
  sorry

end NUMINAMATH_CALUDE_num_lineups_eq_2277_l1904_190465


namespace NUMINAMATH_CALUDE_power_function_m_value_l1904_190447

theorem power_function_m_value : ∃! m : ℝ, m^2 - 9*m + 19 = 1 ∧ 2*m^2 - 7*m - 9 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_power_function_m_value_l1904_190447


namespace NUMINAMATH_CALUDE_blanket_donation_ratio_l1904_190468

/-- The ratio of blankets collected on the second day to the first day -/
def blanket_ratio (team_size : ℕ) (first_day_per_person : ℕ) (last_day_total : ℕ) (total_blankets : ℕ) : ℚ :=
  let first_day := team_size * first_day_per_person
  let second_day := total_blankets - first_day - last_day_total
  (second_day : ℚ) / first_day

/-- Proves that the ratio of blankets collected on the second day to the first day is 3 -/
theorem blanket_donation_ratio :
  blanket_ratio 15 2 22 142 = 3 := by
  sorry

end NUMINAMATH_CALUDE_blanket_donation_ratio_l1904_190468


namespace NUMINAMATH_CALUDE_odd_function_property_l1904_190414

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the function g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (hf : IsOdd f) :
  g f 1 = 1 → g f (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1904_190414


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l1904_190442

theorem min_value_of_function (x : ℝ) (h : x > 1) : 1 / (x - 1) + x ≥ 3 := by
  sorry

theorem equality_condition (x : ℝ) (h : x > 1) : 1 / (x - 1) + x = 3 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l1904_190442


namespace NUMINAMATH_CALUDE_daisy_toys_count_l1904_190487

def monday_toys : ℕ := 5
def tuesday_toys_left : ℕ := 3
def tuesday_toys_bought : ℕ := 3
def wednesday_toys_bought : ℕ := 5

def total_toys : ℕ := monday_toys + (monday_toys - tuesday_toys_left) + tuesday_toys_bought + wednesday_toys_bought

theorem daisy_toys_count : total_toys = 15 := by sorry

end NUMINAMATH_CALUDE_daisy_toys_count_l1904_190487


namespace NUMINAMATH_CALUDE_caramel_candy_probability_l1904_190417

/-- The probability of selecting a caramel-flavored candy from a set of candies -/
theorem caramel_candy_probability 
  (total_candies : ℕ) 
  (caramel_candies : ℕ) 
  (lemon_candies : ℕ) 
  (h1 : total_candies = caramel_candies + lemon_candies)
  (h2 : caramel_candies = 3)
  (h3 : lemon_candies = 4) :
  (caramel_candies : ℚ) / total_candies = 3 / 7 :=
sorry

end NUMINAMATH_CALUDE_caramel_candy_probability_l1904_190417


namespace NUMINAMATH_CALUDE_absolute_value_of_z_l1904_190484

theorem absolute_value_of_z (r : ℝ) (z : ℂ) (h1 : |r| < 3) (h2 : z + 1/z = r) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_z_l1904_190484


namespace NUMINAMATH_CALUDE_seating_arrangement_solution_l1904_190488

/-- Represents a seating arrangement with rows of 7 or 9 seats. -/
structure SeatingArrangement where
  rows_of_nine : ℕ
  rows_of_seven : ℕ

/-- 
  Theorem: Given a seating arrangement where each row seats either 7 or 9 people, 
  and 61 people are to be seated with every seat occupied, 
  the number of rows seating exactly 9 people is 6.
-/
theorem seating_arrangement_solution : 
  ∃ (arrangement : SeatingArrangement),
    arrangement.rows_of_nine * 9 + arrangement.rows_of_seven * 7 = 61 ∧
    arrangement.rows_of_nine = 6 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_solution_l1904_190488


namespace NUMINAMATH_CALUDE_complex_multiplication_l1904_190441

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1904_190441


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1904_190405

theorem z_in_fourth_quadrant (z : ℂ) (h : z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3)) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1904_190405


namespace NUMINAMATH_CALUDE_enemy_count_l1904_190419

theorem enemy_count (points_per_enemy : ℕ) (points_earned : ℕ) (enemies_left : ℕ) :
  points_per_enemy = 8 →
  enemies_left = 2 →
  points_earned = 40 →
  ∃ (total_enemies : ℕ), total_enemies = 7 ∧ points_per_enemy * (total_enemies - enemies_left) = points_earned :=
by sorry

end NUMINAMATH_CALUDE_enemy_count_l1904_190419


namespace NUMINAMATH_CALUDE_fidos_yard_area_fraction_l1904_190461

theorem fidos_yard_area_fraction :
  let square_side : ℝ := 2  -- Arbitrary side length
  let circle_radius : ℝ := 1  -- Half of the square side
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  circle_area / square_area = π :=
by sorry

end NUMINAMATH_CALUDE_fidos_yard_area_fraction_l1904_190461


namespace NUMINAMATH_CALUDE_at_least_one_positive_l1904_190406

theorem at_least_one_positive (x y z : ℝ) : 
  let a := x^2 - 2*x + π/2
  let b := y^2 - 2*y + π/3
  let c := z^2 - 2*z + π/6
  (a > 0) ∨ (b > 0) ∨ (c > 0) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_positive_l1904_190406


namespace NUMINAMATH_CALUDE_equation_is_linear_l1904_190496

/-- A linear equation with two variables is of the form ax + by = c, where a and b are not both zero -/
def is_linear_equation_two_vars (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation x + y = 2 -/
def equation (x y : ℝ) : Prop := x + y = 2

theorem equation_is_linear : is_linear_equation_two_vars equation :=
sorry

end NUMINAMATH_CALUDE_equation_is_linear_l1904_190496


namespace NUMINAMATH_CALUDE_circle_distance_inequality_l1904_190408

theorem circle_distance_inequality (x y : ℝ) : 
  x^2 + y^2 + 2*x - 6*y = 6 → (x - 1)^2 + (y - 2)^2 ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_circle_distance_inequality_l1904_190408


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_arithmetic_progression_l1904_190432

/-- Given a triangle with sides a, b, c forming an arithmetic progression,
    prove that the radius of the inscribed circle is one-third of the altitude to side b. -/
theorem inscribed_circle_radius_arithmetic_progression
  (a b c : ℝ) (h_order : a ≤ b ∧ b ≤ c) (h_arithmetic : 2 * b = a + c)
  (r : ℝ) (h_b : ℝ) (S : ℝ) 
  (h_area_inradius : 2 * S = r * (a + b + c))
  (h_area_altitude : 2 * S = h_b * b) :
  r = h_b / 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_arithmetic_progression_l1904_190432


namespace NUMINAMATH_CALUDE_surface_area_of_sliced_solid_l1904_190431

/-- Right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ

/-- Point on an edge of the prism -/
structure EdgePoint where
  edge : Fin 3  -- 0: AC, 1: BC, 2: DC
  position : ℝ  -- Fraction of the way along the edge

/-- Solid formed by slicing the prism -/
def SlicedSolid (prism : RightPrism) (p q r : EdgePoint) : Type := sorry

/-- Surface area of a sliced solid -/
noncomputable def surfaceArea (prism : RightPrism) (solid : SlicedSolid prism p q r) : ℝ := sorry

/-- The main theorem -/
theorem surface_area_of_sliced_solid 
  (prism : RightPrism)
  (h_height : prism.height = 20)
  (h_base : prism.baseSideLength = 10)
  (p : EdgePoint)
  (h_p : p.edge = 0 ∧ p.position = 1/3)
  (q : EdgePoint)
  (h_q : q.edge = 1 ∧ q.position = 1/3)
  (r : EdgePoint)
  (h_r : r.edge = 2 ∧ r.position = 1/2)
  (solid : SlicedSolid prism p q r) :
  surfaceArea prism solid = (50 * Real.sqrt 3 + 25 * Real.sqrt 2 / 3 + 50 * Real.sqrt 10) / 3 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_sliced_solid_l1904_190431


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l1904_190456

theorem quadratic_form_ratio (j : ℝ) : ∃ (c p q : ℝ),
  (6 * j^2 - 4 * j + 12 = c * (j + p)^2 + q) ∧ (q / p = -34) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l1904_190456


namespace NUMINAMATH_CALUDE_cylinder_volume_l1904_190413

/-- Given a cylinder with lateral surface area 100π cm² and an inscribed rectangular solid
    with diagonal length 10√2 cm, prove that the cylinder's volume is 250π cm³. -/
theorem cylinder_volume (r h : ℝ) (lateral_area : 2 * Real.pi * r * h = 100 * Real.pi)
    (diagonal_length : 4 * r^2 + h^2 = 200) : Real.pi * r^2 * h = 250 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l1904_190413


namespace NUMINAMATH_CALUDE_chernomor_salary_manipulation_l1904_190476

/-- Represents a salary proposal for a single month -/
structure SalaryProposal where
  warrior_salaries : Fin 33 → ℝ
  chernomor_salary : ℝ

/-- The voting function: returns true if the majority of warriors vote in favor -/
def majority_vote (current : SalaryProposal) (proposal : SalaryProposal) : Prop :=
  (Finset.filter (fun i => proposal.warrior_salaries i > current.warrior_salaries i) Finset.univ).card > 16

/-- The theorem stating that Chernomor can achieve his goal -/
theorem chernomor_salary_manipulation :
  ∃ (initial : SalaryProposal) (proposals : Fin 36 → SalaryProposal),
    (∀ i : Fin 35, majority_vote (proposals i) (proposals (i + 1))) ∧
    (proposals 35).chernomor_salary = 10 * initial.chernomor_salary ∧
    (∀ j : Fin 33, (proposals 35).warrior_salaries j ≤ initial.warrior_salaries j / 10) :=
sorry

end NUMINAMATH_CALUDE_chernomor_salary_manipulation_l1904_190476


namespace NUMINAMATH_CALUDE_marks_ratio_polly_willy_l1904_190402

/-- Given the ratios of marks between students, prove the ratio between Polly and Willy -/
theorem marks_ratio_polly_willy (p s w : ℝ) 
  (h1 : p / s = 4 / 5) 
  (h2 : s / w = 5 / 2) : 
  p / w = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_marks_ratio_polly_willy_l1904_190402


namespace NUMINAMATH_CALUDE_smallest_n_with_all_digit_sums_l1904_190467

-- Define a function to calculate the sum of digits of a number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define a function to get all divisors of a number
def divisors (n : ℕ) : Set ℕ := sorry

-- Define a function to get the set of sums of digits of all divisors
def sumsOfDigitsOfDivisors (n : ℕ) : Set ℕ := sorry

-- Main theorem
theorem smallest_n_with_all_digit_sums :
  ∀ n : ℕ, n < 288 →
    ¬(∀ k : ℕ, k ∈ Finset.range 9 → (k + 1) ∈ sumsOfDigitsOfDivisors n) ∧
  (∀ k : ℕ, k ∈ Finset.range 9 → (k + 1) ∈ sumsOfDigitsOfDivisors 288) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_all_digit_sums_l1904_190467


namespace NUMINAMATH_CALUDE_inequality_properties_l1904_190401

theorem inequality_properties (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧
  (1 / (a - b) < 1 / a) ∧
  (|a| > -b) ∧
  (Real.sqrt (-a) > Real.sqrt (-b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l1904_190401


namespace NUMINAMATH_CALUDE_unique_outfits_count_l1904_190464

def number_of_shirts : ℕ := 10
def number_of_ties : ℕ := 8
def shirts_per_outfit : ℕ := 5
def ties_per_outfit : ℕ := 4

theorem unique_outfits_count : 
  (Nat.choose number_of_shirts shirts_per_outfit) * 
  (Nat.choose number_of_ties ties_per_outfit) = 17640 := by
  sorry

end NUMINAMATH_CALUDE_unique_outfits_count_l1904_190464


namespace NUMINAMATH_CALUDE_square_transformation_2007_l1904_190499

-- Define the vertex order as a list of characters
def VertexOrder := List Char

-- Define the transformation operations
def rotate90Clockwise (order : VertexOrder) : VertexOrder :=
  match order with
  | [a, b, c, d] => [d, a, b, c]
  | _ => order

def reflectVertical (order : VertexOrder) : VertexOrder :=
  match order with
  | [a, b, c, d] => [d, c, b, a]
  | _ => order

def reflectHorizontal (order : VertexOrder) : VertexOrder :=
  match order with
  | [a, b, c, d] => [c, b, a, d]
  | _ => order

-- Define the complete transformation sequence
def transformSequence (order : VertexOrder) : VertexOrder :=
  reflectHorizontal (reflectVertical (rotate90Clockwise order))

-- Define a function to apply the transformation sequence n times
def applyTransformSequence (order : VertexOrder) (n : Nat) : VertexOrder :=
  match n with
  | 0 => order
  | n + 1 => applyTransformSequence (transformSequence order) n

-- Theorem statement
theorem square_transformation_2007 :
  applyTransformSequence ['A', 'B', 'C', 'D'] 2007 = ['D', 'C', 'B', 'A'] := by
  sorry


end NUMINAMATH_CALUDE_square_transformation_2007_l1904_190499


namespace NUMINAMATH_CALUDE_samantha_pet_food_difference_l1904_190443

/-- Proves that Samantha bought 49 more cans of cat food than dog and bird food combined. -/
theorem samantha_pet_food_difference : 
  let cat_packages : ℕ := 8
  let dog_packages : ℕ := 5
  let bird_packages : ℕ := 3
  let cat_cans_per_package : ℕ := 12
  let dog_cans_per_package : ℕ := 7
  let bird_cans_per_package : ℕ := 4
  let total_cat_cans := cat_packages * cat_cans_per_package
  let total_dog_cans := dog_packages * dog_cans_per_package
  let total_bird_cans := bird_packages * bird_cans_per_package
  total_cat_cans - (total_dog_cans + total_bird_cans) = 49 := by
  sorry

#eval 8 * 12 - (5 * 7 + 3 * 4)  -- Should output 49

end NUMINAMATH_CALUDE_samantha_pet_food_difference_l1904_190443


namespace NUMINAMATH_CALUDE_percentage_change_equivalence_l1904_190481

theorem percentage_change_equivalence (p q N : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_bound : q < 50) (hN : N > 0) :
  N * (1 + p / 100) * (1 - q / 100) < N ↔ p < (100 * q) / (100 - q) := by
  sorry

end NUMINAMATH_CALUDE_percentage_change_equivalence_l1904_190481


namespace NUMINAMATH_CALUDE_complement_M_equals_expected_l1904_190482

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set Nat := {1, 2}

-- Define the complement of M with respect to U
def complement_M : Set Nat := U \ M

-- Theorem statement
theorem complement_M_equals_expected : complement_M = {3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_M_equals_expected_l1904_190482


namespace NUMINAMATH_CALUDE_kaleb_ferris_wheel_cost_l1904_190411

/-- The amount of money Kaleb spent on the ferris wheel ride -/
def ferris_wheel_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * ticket_price

/-- Theorem: Kaleb spent 27 dollars on the ferris wheel ride -/
theorem kaleb_ferris_wheel_cost :
  ferris_wheel_cost 6 3 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_ferris_wheel_cost_l1904_190411


namespace NUMINAMATH_CALUDE_square_equality_l1904_190430

theorem square_equality (a b : ℝ) : a = b → a^2 = b^2 := by sorry

end NUMINAMATH_CALUDE_square_equality_l1904_190430


namespace NUMINAMATH_CALUDE_vincent_laundry_theorem_l1904_190498

/-- Represents the types of laundry loads --/
inductive LoadType
  | Regular
  | Delicate
  | Heavy

/-- Represents a day's laundry schedule --/
structure DaySchedule where
  regular : Nat
  delicate : Nat
  heavy : Nat

/-- Calculate total loads for a day --/
def totalLoads (schedule : DaySchedule) : Nat :=
  schedule.regular + schedule.delicate + schedule.heavy

/-- Vincent's laundry week --/
def laundryWeek : List DaySchedule :=
  [
    { regular := 2, delicate := 1, heavy := 3 },  -- Wednesday
    { regular := 4, delicate := 2, heavy := 4 },  -- Thursday
    { regular := 2, delicate := 1, heavy := 0 },  -- Friday
    { regular := 0, delicate := 0, heavy := 1 }   -- Saturday
  ]

theorem vincent_laundry_theorem :
  (laundryWeek.map totalLoads).sum = 20 := by
  sorry

#eval (laundryWeek.map totalLoads).sum

end NUMINAMATH_CALUDE_vincent_laundry_theorem_l1904_190498


namespace NUMINAMATH_CALUDE_union_of_sets_l1904_190462

theorem union_of_sets (M N : Set ℕ) (hM : M = {1, 2}) (hN : N = {2, 3}) :
  M ∪ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l1904_190462


namespace NUMINAMATH_CALUDE_inequality_proof_l1904_190471

theorem inequality_proof (x : ℝ) (n : ℕ) 
  (h1 : |x| < 1) (h2 : n ≥ 2) : (1 + x)^n + (1 - x)^n < 2^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1904_190471


namespace NUMINAMATH_CALUDE_brians_trip_distance_l1904_190452

/-- Calculates the distance traveled given car efficiency and gas used -/
def distance_traveled (efficiency : ℝ) (gas_used : ℝ) : ℝ :=
  efficiency * gas_used

/-- Proves that given a car efficiency of 20 miles per gallon and 
    a gas usage of 3 gallons, the distance traveled is 60 miles -/
theorem brians_trip_distance : 
  distance_traveled 20 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_brians_trip_distance_l1904_190452


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1904_190440

/-- Given a fractional equation (2x - m) / (x + 1) = 3 where x is positive,
    prove that m < -3 -/
theorem fractional_equation_solution_range (x m : ℝ) :
  (2 * x - m) / (x + 1) = 3 → x > 0 → m < -3 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1904_190440


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l1904_190409

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_m_value :
  ∀ m : ℝ,
  let p1 : Point := ⟨3, -4⟩
  let p2 : Point := ⟨6, 5⟩
  let p3 : Point := ⟨8, m⟩
  collinear p1 p2 p3 → m = 11 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l1904_190409


namespace NUMINAMATH_CALUDE_final_arrangement_decreasing_l1904_190407

/-- Represents a child with a unique height -/
structure Child :=
  (height : ℕ)

/-- Represents a row of children -/
def Row := List Child

/-- The operation of grouping and rearranging children -/
def groupAndRearrange (row : Row) : Row :=
  sorry

/-- Checks if a row is in decreasing order of height -/
def isDecreasingOrder (row : Row) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem final_arrangement_decreasing (n : ℕ) (initial_row : Row) :
  initial_row.length = n →
  (∀ i j, i ≠ j → (initial_row.get i).height ≠ (initial_row.get j).height) →
  isDecreasingOrder ((groupAndRearrange^[n-1]) initial_row) :=
sorry

end NUMINAMATH_CALUDE_final_arrangement_decreasing_l1904_190407


namespace NUMINAMATH_CALUDE_triangle_area_is_2_sqrt_6_l1904_190486

/-- A triangle with integral sides and perimeter 12 --/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  perimeter_eq : a + b + c = 12
  triangle_ineq_ab : a + b > c
  triangle_ineq_bc : b + c > a
  triangle_ineq_ca : c + a > b

/-- The area of a triangle with integral sides and perimeter 12 is 2√6 --/
theorem triangle_area_is_2_sqrt_6 (t : Triangle) : 
  ∃ (area : ℝ), area = 2 * Real.sqrt 6 ∧ area = (t.a * t.b * Real.sin (π / 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_2_sqrt_6_l1904_190486


namespace NUMINAMATH_CALUDE_new_persons_combined_weight_l1904_190418

/-- The combined weight of two new persons in a group, given specific conditions --/
theorem new_persons_combined_weight
  (n : ℕ)
  (avg_increase : ℝ)
  (old_weight1 old_weight2 : ℝ)
  (h1 : n = 15)
  (h2 : avg_increase = 5.2)
  (h3 : old_weight1 = 68)
  (h4 : old_weight2 = 70) :
  ∃ (w1 w2 : ℝ), w1 + w2 = 216 :=
sorry

end NUMINAMATH_CALUDE_new_persons_combined_weight_l1904_190418


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l1904_190450

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := a^2 + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- Theorem statement
theorem diamond_equation_solution :
  ∃ h : ℝ, diamond 3 h = 12 ∧ h = 6 := by sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l1904_190450


namespace NUMINAMATH_CALUDE_sin_cos_inverse_equation_l1904_190448

theorem sin_cos_inverse_equation (t : ℝ) :
  (Real.sin (2 * t) - Real.arcsin (2 * t))^2 + (Real.arccos (2 * t) - Real.cos (2 * t))^2 = 1 ↔
  ∃ k : ℤ, t = (π / 8) * (2 * ↑k + 1) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_inverse_equation_l1904_190448


namespace NUMINAMATH_CALUDE_parabola_vertex_on_axis_l1904_190434

/-- A parabola with equation y = x^2 - kx + k - 1 has its vertex on a coordinate axis if and only if k = 2 or k = 0 -/
theorem parabola_vertex_on_axis (k : ℝ) : 
  (∃ x y : ℝ, (y = x^2 - k*x + k - 1) ∧ 
    ((x = 0 ∧ y = k - 1) ∨ (y = 0 ∧ x = k/2)) ∧
    (∀ x' y' : ℝ, y' = x'^2 - k*x' + k - 1 → y' ≥ y)) ↔ 
  (k = 2 ∨ k = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_axis_l1904_190434


namespace NUMINAMATH_CALUDE_eighth_term_value_l1904_190455

/-- The nth term of a geometric sequence -/
def geometric_term (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r ^ (n - 1)

/-- The 8th term of the specific geometric sequence -/
def eighth_term : ℚ :=
  geometric_term 3 (3/2) 8

theorem eighth_term_value : eighth_term = 6561 / 128 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l1904_190455


namespace NUMINAMATH_CALUDE_semicircle_radius_prove_semicircle_radius_l1904_190449

theorem semicircle_radius : ℝ → Prop :=
fun r : ℝ =>
  (3 * (2 * r) + 2 * 12 = 2 * (2 * r) + 22 + 16 + 22) → r = 18

theorem prove_semicircle_radius : semicircle_radius 18 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_prove_semicircle_radius_l1904_190449


namespace NUMINAMATH_CALUDE_die_roll_probability_l1904_190415

theorem die_roll_probability : 
  let p_two : ℚ := 1 / 6  -- probability of rolling a 2
  let p_not_two : ℚ := 5 / 6  -- probability of not rolling a 2
  let num_rolls : ℕ := 5  -- number of rolls
  let num_twos : ℕ := 4  -- number of 2s we want
  
  -- probability of rolling exactly four 2s in first four rolls and not a 2 in last roll
  p_two ^ num_twos * p_not_two = 5 / 7776 :=
by sorry

end NUMINAMATH_CALUDE_die_roll_probability_l1904_190415


namespace NUMINAMATH_CALUDE_intersection_of_specific_sets_l1904_190490

theorem intersection_of_specific_sets :
  let A : Set ℕ := {1, 2, 5}
  let B : Set ℕ := {1, 3, 5}
  A ∩ B = {1, 5} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_specific_sets_l1904_190490


namespace NUMINAMATH_CALUDE_carlos_laundry_time_l1904_190470

/-- The total time for Carlos's laundry process -/
def laundry_time (wash_times : List Nat) (dry_times : List Nat) : Nat :=
  wash_times.sum + dry_times.sum

/-- Theorem stating that Carlos's laundry takes 380 minutes in total -/
theorem carlos_laundry_time :
  laundry_time [30, 45, 40, 50, 35] [85, 95] = 380 := by
  sorry

end NUMINAMATH_CALUDE_carlos_laundry_time_l1904_190470


namespace NUMINAMATH_CALUDE_opposite_of_three_l1904_190460

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℝ) : ℝ := -a

theorem opposite_of_three : opposite 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l1904_190460


namespace NUMINAMATH_CALUDE_min_PM_AB_implies_AB_equation_l1904_190427

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x + y + 2 = 0

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define tangent points A and B on circle M
def tangent_points (xA yA xB yB : ℝ) : Prop :=
  circle_M xA yA ∧ circle_M xB yB

-- Define the minimization condition
def min_condition (xP yP xM yM xA yA xB yB : ℝ) : Prop :=
  ∀ x y, point_P x y →
    (x - xM)^2 + (y - yM)^2 ≤ (xP - xM)^2 + (yP - yM)^2

-- Theorem statement
theorem min_PM_AB_implies_AB_equation :
  ∀ xP yP xM yM xA yA xB yB,
    point_P xP yP →
    tangent_points xA yA xB yB →
    min_condition xP yP xM yM xA yA xB yB →
    2*xA + yA + 1 = 0 ∧ 2*xB + yB + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_PM_AB_implies_AB_equation_l1904_190427


namespace NUMINAMATH_CALUDE_sum_and_difference_squares_l1904_190479

theorem sum_and_difference_squares (x y : ℝ) (h1 : x + y = 24) (h2 : x - y = 8) :
  (x + y)^2 + (x - y)^2 = 640 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_difference_squares_l1904_190479


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1904_190485

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (6 * a^3 + 500 * a + 1001 = 0) →
  (6 * b^3 + 500 * b + 1001 = 0) →
  (6 * c^3 + 500 * c + 1001 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 500.5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1904_190485


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l1904_190469

theorem max_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1)
  (hax : a^x = 3) (hby : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  (∀ z w : ℝ, a^z = 3 → b^w = 3 → 1/z + 1/w ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l1904_190469


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l1904_190437

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = 1.1 * x^2 - 2.1 * x + 5) ∧
    q (-1) = 4 ∧
    q 2 = 1 ∧
    q 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l1904_190437


namespace NUMINAMATH_CALUDE_no_real_solutions_l1904_190426

theorem no_real_solutions : ¬∃ (x : ℝ), x + 48 / (x - 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1904_190426


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1904_190477

theorem quadratic_inequality_solution (a : ℝ) (h : a < 0) :
  let solution := {x : ℝ | a * x^2 - (a - 1) * x - 1 < 0}
  ((-1 < a ∧ a < 0) → solution = {x | x < 1 ∨ x > -1/a}) ∧
  (a = -1 → solution = {x | x ≠ 1}) ∧
  (a < -1 → solution = {x | x < -1/a ∨ x > 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1904_190477


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l1904_190436

/-- The function f(x) = x^3 - 2x^2 + 5x - 1 -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 1

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x + 5

theorem f_derivative_at_one : f' 1 = 4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l1904_190436


namespace NUMINAMATH_CALUDE_hill_climbing_speed_l1904_190403

/-- Proves that given a round trip journey where the total time is 6 hours (4 hours up, 2 hours down)
    and the average speed for the whole journey is 4 km/h, then the average speed while climbing
    to the top is 3 km/h. -/
theorem hill_climbing_speed
  (total_time : ℝ)
  (up_time : ℝ)
  (down_time : ℝ)
  (average_speed : ℝ)
  (h1 : total_time = up_time + down_time)
  (h2 : total_time = 6)
  (h3 : up_time = 4)
  (h4 : down_time = 2)
  (h5 : average_speed = 4) :
  (average_speed * total_time) / (2 * up_time) = 3 := by
  sorry

end NUMINAMATH_CALUDE_hill_climbing_speed_l1904_190403


namespace NUMINAMATH_CALUDE_caleb_caught_two_trouts_l1904_190424

/-- The number of trouts Caleb caught -/
def caleb_trouts : ℕ := 2

/-- The number of trouts Caleb's dad caught -/
def dad_trouts : ℕ := 3 * caleb_trouts

theorem caleb_caught_two_trouts :
  (dad_trouts = 3 * caleb_trouts) ∧
  (dad_trouts = caleb_trouts + 4) →
  caleb_trouts = 2 := by
  sorry

end NUMINAMATH_CALUDE_caleb_caught_two_trouts_l1904_190424


namespace NUMINAMATH_CALUDE_rational_numbers_include_integers_and_fractions_l1904_190433

/-- A rational number is a number that can be expressed as the quotient of two integers, where the denominator is non-zero. -/
def IsRational (x : ℚ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

/-- An integer is a whole number (positive, negative, or zero) without a fractional component. -/
def IsInteger (x : ℚ) : Prop := ∃ (n : ℤ), x = n

/-- A fraction is a rational number that is not an integer. -/
def IsFraction (x : ℚ) : Prop := IsRational x ∧ ¬IsInteger x

theorem rational_numbers_include_integers_and_fractions :
  (∀ x : ℚ, IsInteger x → IsRational x) ∧
  (∀ x : ℚ, IsFraction x → IsRational x) :=
sorry

end NUMINAMATH_CALUDE_rational_numbers_include_integers_and_fractions_l1904_190433
