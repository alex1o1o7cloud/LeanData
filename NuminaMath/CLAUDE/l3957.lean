import Mathlib

namespace NUMINAMATH_CALUDE_h_of_3_eq_3_l3957_395741

-- Define the function h
noncomputable def h : ℝ → ℝ := fun x => 
  if x = 1 then 0  -- Handle the case when x = 1 separately
  else ((x + 1) * (x^2 + 1) * (x^3 + 1) * (x^4 + 1) * (x^5 + 1) * (x^6 + 1) * (x^7 + 1) * (x^8 + 1) * (x^9 + 1) - 1) / (x^26 - 1)

-- Theorem statement
theorem h_of_3_eq_3 : h 3 = 3 := by sorry

end NUMINAMATH_CALUDE_h_of_3_eq_3_l3957_395741


namespace NUMINAMATH_CALUDE_sin_double_minus_cos_half_squared_l3957_395764

theorem sin_double_minus_cos_half_squared 
  (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : Real.sin (Real.pi - α) = 4 / 5) : 
  Real.sin (2 * α) - Real.cos (α / 2) ^ 2 = 4 / 25 := by
sorry

end NUMINAMATH_CALUDE_sin_double_minus_cos_half_squared_l3957_395764


namespace NUMINAMATH_CALUDE_field_planting_fraction_l3957_395789

theorem field_planting_fraction :
  ∀ (a b c x : ℝ),
  a = 5 ∧ b = 12 ∧ c^2 = a^2 + b^2 →
  x^2 * c = 3 * (a * b) →
  (a * b - x^2) / (a * b) = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_field_planting_fraction_l3957_395789


namespace NUMINAMATH_CALUDE_pie_remainder_pie_problem_l3957_395788

theorem pie_remainder (carlos_portion : Real) (maria_fraction : Real) : Real :=
  let remaining_after_carlos := 1 - carlos_portion
  let maria_portion := maria_fraction * remaining_after_carlos
  let final_remainder := remaining_after_carlos - maria_portion
  
  final_remainder

theorem pie_problem :
  pie_remainder 0.6 0.25 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_pie_remainder_pie_problem_l3957_395788


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l3957_395771

-- Define the quadratic equations
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the set (-2,0) ∪ (1,3)
def target_set (m : ℝ) : Prop := (m > -2 ∧ m < 0) ∨ (m > 1 ∧ m < 3)

-- State the theorem
theorem quadratic_roots_theorem (m : ℝ) :
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ target_set m :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l3957_395771


namespace NUMINAMATH_CALUDE_unique_three_digit_sum_l3957_395730

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Theorem stating that 198 is the only three-digit number equal to 11 times the sum of its digits -/
theorem unique_three_digit_sum : ∃! n : ℕ, isThreeDigit n ∧ n = 11 * sumOfDigits n := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_sum_l3957_395730


namespace NUMINAMATH_CALUDE_probability_of_event_A_is_half_events_A_and_C_mutually_exclusive_l3957_395738

/-- Represents the labels on the balls -/
inductive Label : Type
  | one : Label
  | two : Label
  | three : Label

/-- Represents a pair of drawn balls -/
structure DrawnBalls :=
  (fromA : Label)
  (fromB : Label)

/-- The sample space of all possible outcomes -/
def sampleSpace : List DrawnBalls := sorry

/-- Event A: sum of labels < 4 -/
def eventA (db : DrawnBalls) : Prop := sorry

/-- Event C: product of labels > 3 -/
def eventC (db : DrawnBalls) : Prop := sorry

/-- The probability of an event -/
def probability (event : DrawnBalls → Prop) : ℚ := sorry

theorem probability_of_event_A_is_half :
  probability eventA = 1 / 2 := sorry

theorem events_A_and_C_mutually_exclusive :
  ∀ db : DrawnBalls, ¬(eventA db ∧ eventC db) := sorry

end NUMINAMATH_CALUDE_probability_of_event_A_is_half_events_A_and_C_mutually_exclusive_l3957_395738


namespace NUMINAMATH_CALUDE_triangle_max_area_l3957_395734

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  3 * b * Real.sin C - 5 * c * Real.sin B * Real.cos A = 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3957_395734


namespace NUMINAMATH_CALUDE_mindys_tax_rate_l3957_395784

/-- Given Mork's tax rate, Mindy's income relative to Mork's, and their combined tax rate,
    calculate Mindy's tax rate. -/
theorem mindys_tax_rate
  (morks_tax_rate : ℝ)
  (mindys_income_ratio : ℝ)
  (combined_tax_rate : ℝ)
  (h1 : morks_tax_rate = 0.40)
  (h2 : mindys_income_ratio = 3)
  (h3 : combined_tax_rate = 0.325) :
  let mindys_tax_rate := (combined_tax_rate * (1 + mindys_income_ratio) - morks_tax_rate) / mindys_income_ratio
  mindys_tax_rate = 0.30 :=
by sorry

end NUMINAMATH_CALUDE_mindys_tax_rate_l3957_395784


namespace NUMINAMATH_CALUDE_no_prime_p_and_p6_plus_6_prime_l3957_395744

theorem no_prime_p_and_p6_plus_6_prime :
  ¬ ∃ (p : ℕ), Nat.Prime p ∧ Nat.Prime (p^6 + 6) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_p_and_p6_plus_6_prime_l3957_395744


namespace NUMINAMATH_CALUDE_min_apples_in_basket_l3957_395733

theorem min_apples_in_basket (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 4 = 3) ∧ (x % 5 = 2) → x ≥ 67 :=
by sorry

end NUMINAMATH_CALUDE_min_apples_in_basket_l3957_395733


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l3957_395779

theorem least_subtrahend_for_divisibility (n m : ℕ) (hn : n = 13602) (hm : m = 87) :
  ∃ (k : ℕ), k = 30 ∧ 
  (∀ (x : ℕ), x < k → ¬(∃ (q : ℕ), n - x = m * q)) ∧
  (∃ (q : ℕ), n - k = m * q) :=
sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l3957_395779


namespace NUMINAMATH_CALUDE_ribbon_pieces_l3957_395778

theorem ribbon_pieces (original_length : ℝ) (piece_length : ℝ) (remaining_length : ℝ) : 
  original_length = 51 →
  piece_length = 0.15 →
  remaining_length = 36 →
  (original_length - remaining_length) / piece_length = 100 := by
sorry

end NUMINAMATH_CALUDE_ribbon_pieces_l3957_395778


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l3957_395716

/-- The number of ways to choose lines forming a rectangle with color constraints -/
def rectangleChoices (totalHorizontal totalVertical redHorizontal blueVertical : ℕ) : ℕ :=
  let horizontalChoices := (redHorizontal.choose 1 * (totalHorizontal - redHorizontal + 1).choose 1) +
                           redHorizontal.choose 2
  let verticalChoices := (blueVertical.choose 1 * (totalVertical - blueVertical + 1).choose 1) +
                         blueVertical.choose 2
  horizontalChoices * verticalChoices

/-- Theorem stating the number of ways to form a rectangle with given constraints -/
theorem rectangle_formation_count :
  rectangleChoices 6 5 3 2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l3957_395716


namespace NUMINAMATH_CALUDE_three_in_range_of_g_l3957_395785

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

-- Theorem statement
theorem three_in_range_of_g (a : ℝ) : ∃ x : ℝ, g a x = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_in_range_of_g_l3957_395785


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l3957_395777

theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let R := Real.sqrt ((a^2 + b^2 + c^2) / 4)
  4 * Real.pi * R^2 = 50 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l3957_395777


namespace NUMINAMATH_CALUDE_pool_cleaning_tip_percentage_l3957_395790

/-- Calculates the tip percentage for pool cleaning sessions -/
theorem pool_cleaning_tip_percentage
  (days_between_cleanings : ℕ)
  (cost_per_cleaning : ℕ)
  (chemical_cost : ℕ)
  (chemical_frequency : ℕ)
  (total_monthly_cost : ℕ)
  (days_in_month : ℕ := 30)  -- Assumption from the problem
  (h1 : days_between_cleanings = 3)
  (h2 : cost_per_cleaning = 150)
  (h3 : chemical_cost = 200)
  (h4 : chemical_frequency = 2)
  (h5 : total_monthly_cost = 2050)
  : (total_monthly_cost - (days_in_month / days_between_cleanings * cost_per_cleaning + chemical_frequency * chemical_cost)) / (days_in_month / days_between_cleanings * cost_per_cleaning) * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_pool_cleaning_tip_percentage_l3957_395790


namespace NUMINAMATH_CALUDE_train_speed_ratio_l3957_395706

/-- Given two trains running in opposite directions, prove that their speed ratio is 39:5 -/
theorem train_speed_ratio :
  ∀ (v₁ v₂ : ℝ), v₁ > 0 → v₂ > 0 →
  ∃ (l₁ l₂ : ℝ), l₁ > 0 ∧ l₂ > 0 ∧
  (l₁ / v₁ = 27) ∧ (l₂ / v₂ = 17) ∧ ((l₁ + l₂) / (v₁ + v₂) = 22) →
  v₁ / v₂ = 39 / 5 := by
sorry

end NUMINAMATH_CALUDE_train_speed_ratio_l3957_395706


namespace NUMINAMATH_CALUDE_ceiling_sqrt_162_l3957_395755

theorem ceiling_sqrt_162 : ⌈Real.sqrt 162⌉ = 13 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_162_l3957_395755


namespace NUMINAMATH_CALUDE_base9_85_equals_77_l3957_395747

-- Define a function to convert a base-9 number to base-10
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

-- Theorem statement
theorem base9_85_equals_77 :
  base9ToBase10 [5, 8] = 77 := by
  sorry

end NUMINAMATH_CALUDE_base9_85_equals_77_l3957_395747


namespace NUMINAMATH_CALUDE_calculate_expression_l3957_395786

theorem calculate_expression : 
  75 * (4 + 1/3 - (5 + 1/4)) / (3 + 1/2 + 2 + 1/5) = -5/31 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3957_395786


namespace NUMINAMATH_CALUDE_debby_photos_l3957_395763

/-- Calculates the number of photographs Debby kept after her vacation -/
theorem debby_photos (N : ℝ) : 
  let zoo_percent : ℝ := 0.60
  let museum_percent : ℝ := 0.25
  let gallery_percent : ℝ := 0.15
  let zoo_keep : ℝ := 0.70
  let museum_keep : ℝ := 0.50
  let gallery_keep : ℝ := 1

  let zoo_photos : ℝ := zoo_percent * N
  let museum_photos : ℝ := museum_percent * N
  let gallery_photos : ℝ := gallery_percent * N

  let kept_zoo : ℝ := zoo_keep * zoo_photos
  let kept_museum : ℝ := museum_keep * museum_photos
  let kept_gallery : ℝ := gallery_keep * gallery_photos

  let total_kept : ℝ := kept_zoo + kept_museum + kept_gallery

  total_kept = 0.695 * N :=
by sorry

end NUMINAMATH_CALUDE_debby_photos_l3957_395763


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3957_395752

theorem decimal_to_fraction_sum (a b : ℕ+) :
  (a : ℚ) / (b : ℚ) = 0.3421 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 0.3421 → a ≤ c ∧ b ≤ d →
  a + b = 13421 :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3957_395752


namespace NUMINAMATH_CALUDE_orange_selling_price_l3957_395787

/-- Proves that the selling price of each orange is 60 cents given the conditions -/
theorem orange_selling_price (total_cost : ℚ) (num_oranges : ℕ) (profit_per_orange : ℚ) :
  total_cost = 25 / 2 →
  num_oranges = 25 →
  profit_per_orange = 1 / 10 →
  (total_cost / num_oranges + profit_per_orange) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_orange_selling_price_l3957_395787


namespace NUMINAMATH_CALUDE_misread_weight_l3957_395756

/-- Proves that the misread weight is 56 kg given the conditions of the problem -/
theorem misread_weight (n : ℕ) (initial_avg correct_avg : ℝ) (correct_weight : ℝ) :
  n = 20 ∧ 
  initial_avg = 58.4 ∧ 
  correct_avg = 59 ∧ 
  correct_weight = 68 →
  ∃ x : ℝ, x = 56 ∧ n * correct_avg - n * initial_avg = correct_weight - x :=
by sorry

end NUMINAMATH_CALUDE_misread_weight_l3957_395756


namespace NUMINAMATH_CALUDE_cube_surface_area_l3957_395758

/-- Given a cube with side perimeter 24 cm, its surface area is 216 cm² -/
theorem cube_surface_area (side_perimeter : ℝ) (h : side_perimeter = 24) :
  6 * (side_perimeter / 4)^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3957_395758


namespace NUMINAMATH_CALUDE_geometric_progression_seventh_term_l3957_395748

theorem geometric_progression_seventh_term 
  (b₁ q : ℚ) 
  (sum_first_three : b₁ + b₁*q + b₁*q^2 = 91)
  (arithmetic_progression : 2*(b₁*q + 27) = (b₁ + 25) + (b₁*q^2 + 1)) :
  b₁*q^6 = (35 * 46656) / 117649 ∨ b₁*q^6 = (63 * 4096) / 117649 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_seventh_term_l3957_395748


namespace NUMINAMATH_CALUDE_new_home_library_capacity_l3957_395722

theorem new_home_library_capacity 
  (M : ℚ) -- Millicent's total number of books
  (H : ℚ) -- Harold's total number of books
  (harold_ratio : H = (1/2) * M) -- Harold has 1/2 as many books as Millicent
  (harold_brings : ℚ) -- Number of books Harold brings
  (millicent_brings : ℚ) -- Number of books Millicent brings
  (harold_brings_def : harold_brings = (1/3) * H) -- Harold brings 1/3 of his books
  (millicent_brings_def : millicent_brings = (1/2) * M) -- Millicent brings 1/2 of her books
  : harold_brings + millicent_brings = (2/3) * M := by
  sorry

end NUMINAMATH_CALUDE_new_home_library_capacity_l3957_395722


namespace NUMINAMATH_CALUDE_hindi_speakers_count_l3957_395700

/-- Represents the number of students who can speak a certain number of languages -/
structure LanguageSpeakers where
  total : ℕ  -- Total number of students in the class
  gujarati : ℕ  -- Number of students who can speak Gujarati
  marathi : ℕ  -- Number of students who can speak Marathi
  twoLanguages : ℕ  -- Number of students who can speak two languages
  allThree : ℕ  -- Number of students who can speak all three languages

/-- Calculates the number of Hindi speakers given the language distribution in the class -/
def numHindiSpeakers (ls : LanguageSpeakers) : ℕ :=
  ls.total - (ls.gujarati + ls.marathi - ls.twoLanguages + ls.allThree)

/-- Theorem stating that the number of Hindi speakers is 10 given the problem conditions -/
theorem hindi_speakers_count (ls : LanguageSpeakers) 
  (h_total : ls.total = 22)
  (h_gujarati : ls.gujarati = 6)
  (h_marathi : ls.marathi = 6)
  (h_two : ls.twoLanguages = 2)
  (h_all : ls.allThree = 1) :
  numHindiSpeakers ls = 10 := by
  sorry


end NUMINAMATH_CALUDE_hindi_speakers_count_l3957_395700


namespace NUMINAMATH_CALUDE_fish_remaining_l3957_395712

theorem fish_remaining (initial : ℝ) (moved : ℝ) :
  initial ≥ moved →
  initial - moved = initial - moved :=
by sorry

end NUMINAMATH_CALUDE_fish_remaining_l3957_395712


namespace NUMINAMATH_CALUDE_min_garden_width_proof_l3957_395761

/-- The minimum width of a rectangular garden satisfying the given conditions -/
def min_garden_width : ℝ := 4

/-- The length of the garden in terms of its width -/
def garden_length (w : ℝ) : ℝ := w + 20

/-- The area of the garden in terms of its width -/
def garden_area (w : ℝ) : ℝ := w * garden_length w

theorem min_garden_width_proof :
  (∀ w : ℝ, w > 0 → garden_area w ≥ 120 → w ≥ min_garden_width) ∧
  garden_area min_garden_width ≥ 120 :=
sorry

end NUMINAMATH_CALUDE_min_garden_width_proof_l3957_395761


namespace NUMINAMATH_CALUDE_sum_of_cubes_equals_ten_squared_l3957_395746

theorem sum_of_cubes_equals_ten_squared (h1 : 1 + 2 + 3 + 4 = 10) 
  (h2 : ∃ n : ℕ, 1^3 + 2^3 + 3^3 + 4^3 = 10^n) : 
  ∃ n : ℕ, 1^3 + 2^3 + 3^3 + 4^3 = 10^n ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equals_ten_squared_l3957_395746


namespace NUMINAMATH_CALUDE_solution_symmetry_l3957_395725

theorem solution_symmetry (x y : ℝ) : 
  ((x - y) * (x^2 - y^2) = 160 ∧ (x + y) * (x^2 + y^2) = 580) →
  ((3 - 7) * (3^2 - 7^2) = 160 ∧ (3 + 7) * (3^2 + 7^2) = 580) →
  ((7 - 3) * (7^2 - 3^2) = 160 ∧ (7 + 3) * (7^2 + 3^2) = 580) := by
sorry

end NUMINAMATH_CALUDE_solution_symmetry_l3957_395725


namespace NUMINAMATH_CALUDE_addition_equality_l3957_395729

theorem addition_equality : 731 + 672 = 1403 := by
  sorry

end NUMINAMATH_CALUDE_addition_equality_l3957_395729


namespace NUMINAMATH_CALUDE_range_of_M_l3957_395793

theorem range_of_M (x y z : ℝ) 
  (h1 : x + y + z = 30)
  (h2 : 3 * x + y - z = 50)
  (h3 : x ≥ 0)
  (h4 : y ≥ 0)
  (h5 : z ≥ 0) :
  120 ≤ 5 * x + 4 * y + 2 * z ∧ 5 * x + 4 * y + 2 * z ≤ 130 :=
by sorry

end NUMINAMATH_CALUDE_range_of_M_l3957_395793


namespace NUMINAMATH_CALUDE_modular_inverse_11_mod_1000_l3957_395798

theorem modular_inverse_11_mod_1000 : ∃ x : ℕ, x < 1000 ∧ (11 * x) % 1000 = 1 :=
  by
  use 91
  sorry

end NUMINAMATH_CALUDE_modular_inverse_11_mod_1000_l3957_395798


namespace NUMINAMATH_CALUDE_difference_of_squares_75_25_l3957_395792

theorem difference_of_squares_75_25 : 75^2 - 25^2 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_75_25_l3957_395792


namespace NUMINAMATH_CALUDE_youngest_child_age_l3957_395765

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem youngest_child_age (children : Fin 6 → ℕ) : 
  (∀ i : Fin 6, is_prime (children i)) →
  (∃ y : ℕ, children 0 = y ∧ 
            children 1 = y + 2 ∧
            children 2 = y + 6 ∧
            children 3 = y + 8 ∧
            children 4 = y + 12 ∧
            children 5 = y + 14) →
  children 0 = 5 :=
by sorry

end NUMINAMATH_CALUDE_youngest_child_age_l3957_395765


namespace NUMINAMATH_CALUDE_three_times_first_minus_second_l3957_395742

theorem three_times_first_minus_second (x y : ℕ) : 
  x + y = 48 → y = 17 → 3 * x - y = 76 := by
  sorry

end NUMINAMATH_CALUDE_three_times_first_minus_second_l3957_395742


namespace NUMINAMATH_CALUDE_book_price_is_two_l3957_395750

/-- The price of a book in rubles -/
def book_price : ℝ := 2

/-- The amount paid for the book in rubles -/
def amount_paid : ℝ := 1

/-- The remaining amount to be paid for the book -/
def remaining_amount : ℝ := book_price - amount_paid

theorem book_price_is_two :
  book_price = 2 ∧
  amount_paid = 1 ∧
  remaining_amount = book_price - amount_paid ∧
  remaining_amount = amount_paid + (book_price - (book_price - amount_paid)) :=
by sorry

end NUMINAMATH_CALUDE_book_price_is_two_l3957_395750


namespace NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l3957_395707

/-- Hyperbola type representing xy = 1 -/
structure Hyperbola where
  C₁ : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.1 * p.2 = 1}
  C₂ : Set (ℝ × ℝ) := {p | p.1 < 0 ∧ p.1 * p.2 = 1}

/-- Predicate to check if three points form an equilateral triangle -/
def IsEquilateralTriangle (p q r : ℝ × ℝ) : Prop :=
  let d₁ := (p.1 - q.1)^2 + (p.2 - q.2)^2
  let d₂ := (q.1 - r.1)^2 + (q.2 - r.2)^2
  let d₃ := (r.1 - p.1)^2 + (r.2 - p.2)^2
  d₁ = d₂ ∧ d₂ = d₃

/-- Main theorem statement -/
theorem hyperbola_equilateral_triangle (h : Hyperbola) (p q r : ℝ × ℝ) 
  (hp : p = (-1, -1) ∧ p ∈ h.C₂)
  (hq : q ∈ h.C₁)
  (hr : r ∈ h.C₁)
  (heq : IsEquilateralTriangle p q r) :
  (¬ (p ∈ h.C₁ ∧ q ∈ h.C₁ ∧ r ∈ h.C₁)) ∧
  (q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ r = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l3957_395707


namespace NUMINAMATH_CALUDE_rickshaw_charge_calculation_l3957_395753

/-- Rickshaw charge calculation -/
theorem rickshaw_charge_calculation 
  (initial_charge : ℝ) 
  (additional_charge : ℝ) 
  (total_distance : ℝ) 
  (total_charge : ℝ) :
  initial_charge = 13.5 →
  additional_charge = 2.5 →
  total_distance = 13 →
  total_charge = 103.5 →
  initial_charge + additional_charge * (total_distance - 1) = total_charge :=
by sorry

end NUMINAMATH_CALUDE_rickshaw_charge_calculation_l3957_395753


namespace NUMINAMATH_CALUDE_equation_solution_l3957_395773

theorem equation_solution : 
  {x : ℝ | ∃ (a b : ℝ), a^4 = 59 - 2*x ∧ b^4 = 23 + 2*x ∧ a + b = 4} = {-8, 29} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3957_395773


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l3957_395715

/-- The number of sides in a dodecagon -/
def n : ℕ := 12

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a regular dodecagon is 54 -/
theorem dodecagon_diagonals : num_diagonals n = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l3957_395715


namespace NUMINAMATH_CALUDE_average_rate_of_change_f_on_1_5_l3957_395799

def f (x : ℝ) : ℝ := 2 * x + 1

theorem average_rate_of_change_f_on_1_5 :
  (f 5 - f 1) / (5 - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_of_change_f_on_1_5_l3957_395799


namespace NUMINAMATH_CALUDE_equal_area_triangle_square_l3957_395735

/-- A square with vertices O, S, U, V -/
structure Square (O S U V : ℝ × ℝ) : Prop where
  is_square : true  -- We assume OSUV is a square without proving it

/-- The area of a triangle given three points -/
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

/-- The area of a square given its side length -/
def square_area (side : ℝ) : ℝ := side * side

theorem equal_area_triangle_square 
  (O S U V W : ℝ × ℝ) 
  (h_square : Square O S U V)
  (h_O : O = (0, 0))
  (h_U : U = (3, 3))
  (h_W : W = (3, 9)) : 
  triangle_area S V W = square_area 3 := by
  sorry

#check equal_area_triangle_square

end NUMINAMATH_CALUDE_equal_area_triangle_square_l3957_395735


namespace NUMINAMATH_CALUDE_telescope_visibility_increase_l3957_395772

theorem telescope_visibility_increase (min_without max_without min_with max_with : ℝ) 
  (h1 : min_without = 100)
  (h2 : max_without = 110)
  (h3 : min_with = 150)
  (h4 : max_with = 165) :
  let avg_without := (min_without + max_without) / 2
  let avg_with := (min_with + max_with) / 2
  (avg_with - avg_without) / avg_without * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visibility_increase_l3957_395772


namespace NUMINAMATH_CALUDE_lens_curve_properties_l3957_395769

/-- A lens-shaped curve consisting of two equal circular arcs -/
structure LensCurve where
  radius : ℝ
  arc_angle : ℝ
  h_positive_radius : 0 < radius
  h_arc_angle : arc_angle = 2 * Real.pi / 3

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  h_positive_side : 0 < side_length

/-- Predicate to check if a curve is closed and non-self-intersecting -/
def is_closed_non_self_intersecting (curve : Type) : Prop := sorry

/-- Predicate to check if a curve is different from a circle -/
def is_not_circle (curve : Type) : Prop := sorry

/-- Predicate to check if a triangle can be moved inside a curve with vertices tracing the curve -/
def can_move_triangle_inside (curve : Type) (triangle : Type) : Prop := sorry

theorem lens_curve_properties (l : LensCurve) (t : EquilateralTriangle) 
  (h : l.radius = t.side_length) : 
  is_closed_non_self_intersecting LensCurve ∧ 
  is_not_circle LensCurve ∧ 
  can_move_triangle_inside LensCurve EquilateralTriangle := by
  sorry

end NUMINAMATH_CALUDE_lens_curve_properties_l3957_395769


namespace NUMINAMATH_CALUDE_second_bus_students_l3957_395721

theorem second_bus_students (first_bus : ℕ) (second_bus : ℕ) : 
  first_bus = 38 →
  second_bus - 4 = (first_bus + 4) + 2 →
  second_bus = 44 := by
sorry

end NUMINAMATH_CALUDE_second_bus_students_l3957_395721


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3957_395727

def p (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 4

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 4) ∧
  (∀ x : ℝ, (x - 1) * (x + 1) * (x - 4) = p x) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3957_395727


namespace NUMINAMATH_CALUDE_elena_pen_purchase_l3957_395728

theorem elena_pen_purchase (cost_x : ℝ) (cost_y : ℝ) (total_pens : ℕ) (total_cost : ℝ) :
  cost_x = 4 →
  cost_y = 2.8 →
  total_pens = 12 →
  total_cost = 40 →
  ∃ (x y : ℕ), x + y = total_pens ∧ x * cost_x + y * cost_y = total_cost ∧ x = 5 :=
by sorry

end NUMINAMATH_CALUDE_elena_pen_purchase_l3957_395728


namespace NUMINAMATH_CALUDE_sin_four_arcsin_one_fourth_l3957_395776

theorem sin_four_arcsin_one_fourth :
  Real.sin (4 * Real.arcsin (1/4)) = 7 * Real.sqrt 15 / 32 := by
  sorry

end NUMINAMATH_CALUDE_sin_four_arcsin_one_fourth_l3957_395776


namespace NUMINAMATH_CALUDE_parabola_directrix_l3957_395732

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := x^2 = 4*y

/-- The directrix equation -/
def directrix_eq (y : ℝ) : Prop := y = -1

/-- Theorem: The directrix of the parabola x^2 = 4y is y = -1 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_eq x y → ∃ d : ℝ, directrix_eq d ∧ 
  (∀ p : ℝ × ℝ, p.1^2 = 4*p.2 → (p.1^2 + (p.2 - d)^2) = (p.2 - d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3957_395732


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l3957_395754

theorem not_p_sufficient_not_necessary_for_q :
  (∃ a : ℝ, (a < -1 → ∀ x > 0, a ≤ (x^2 + 1) / x) ∧
   (∀ x > 0, a ≤ (x^2 + 1) / x → ¬(a < -1))) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l3957_395754


namespace NUMINAMATH_CALUDE_candy_necklaces_remaining_l3957_395743

/-- Proves that given 9 packs of candy necklaces with 8 necklaces in each pack,
    if 4 packs are opened, then at least 40 candy necklaces remain unopened. -/
theorem candy_necklaces_remaining (total_packs : ℕ) (necklaces_per_pack : ℕ) (opened_packs : ℕ) :
  total_packs = 9 →
  necklaces_per_pack = 8 →
  opened_packs = 4 →
  (total_packs - opened_packs) * necklaces_per_pack ≥ 40 := by
  sorry

end NUMINAMATH_CALUDE_candy_necklaces_remaining_l3957_395743


namespace NUMINAMATH_CALUDE_range_of_m_l3957_395724

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 4 → x > 2 * m^2 - 3) ∧
  (∃ x : ℝ, x > 2 * m^2 - 3 ∧ (x ≤ -1 ∨ x ≥ 4)) →
  -1 ≤ m ∧ m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3957_395724


namespace NUMINAMATH_CALUDE_ending_number_is_48_l3957_395760

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def sequence_start : ℕ := 10

def sequence_length : ℕ := 13

theorem ending_number_is_48 :
  ∃ (seq : ℕ → ℕ),
    (∀ i, seq i ≥ sequence_start) ∧
    (∀ i, is_divisible_by_3 (seq i)) ∧
    (∀ i j, i < j → seq i < seq j) ∧
    (seq 0 = (sequence_start + 2)) ∧
    (seq (sequence_length - 1) = 48) :=
sorry

end NUMINAMATH_CALUDE_ending_number_is_48_l3957_395760


namespace NUMINAMATH_CALUDE_speed_conversion_l3957_395709

/-- Conversion factor from m/s to km/h -/
def mps_to_kmh : ℝ := 3.6

/-- The given speed in km/h -/
def given_speed_kmh : ℝ := 1.1076923076923078

/-- The speed in m/s to be proven -/
def speed_mps : ℝ := 0.3076923076923077

theorem speed_conversion :
  speed_mps * mps_to_kmh = given_speed_kmh := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l3957_395709


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l3957_395751

theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 2310 → 
  ∀ (a b c : ℕ+), a * b * c = 2310 → l + w + h ≤ a + b + c → 
  l + w + h = 42 :=
sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l3957_395751


namespace NUMINAMATH_CALUDE_brady_earnings_correct_l3957_395781

/-- Calculates the total earnings for Brady's transcription work -/
def brady_earnings (basic_cards : ℕ) (gourmet_cards : ℕ) : ℚ :=
  let basic_rate : ℚ := 70 / 100
  let gourmet_rate : ℚ := 90 / 100
  let basic_earnings := basic_rate * basic_cards
  let gourmet_earnings := gourmet_rate * gourmet_cards
  let card_earnings := basic_earnings + gourmet_earnings
  let total_cards := basic_cards + gourmet_cards
  let bonus_count := total_cards / 100
  let bonus_base := 10
  let bonus_increment := 5
  let bonus_total := bonus_count * bonus_base + (bonus_count * (bonus_count - 1) / 2) * bonus_increment
  card_earnings + bonus_total

theorem brady_earnings_correct :
  brady_earnings 120 80 = 181 := by
  sorry

end NUMINAMATH_CALUDE_brady_earnings_correct_l3957_395781


namespace NUMINAMATH_CALUDE_athlete_distance_l3957_395736

/-- Proves that an athlete running at 18 km/h for 40 seconds covers 200 meters -/
theorem athlete_distance (speed_kmh : ℝ) (time_s : ℝ) (distance_m : ℝ) : 
  speed_kmh = 18 → time_s = 40 → distance_m = speed_kmh * (1000 / 3600) * time_s → distance_m = 200 := by
  sorry

#check athlete_distance

end NUMINAMATH_CALUDE_athlete_distance_l3957_395736


namespace NUMINAMATH_CALUDE_parallel_line_slope_l3957_395796

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line given its equation
def slope_of_line (f : ℝ → ℝ → Prop) : ℝ :=
  sorry

-- Theorem: The slope of a line parallel to 3x - 6y = 12 is 1/2
theorem parallel_line_slope :
  slope_of_line line_equation = 1/2 := by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l3957_395796


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3957_395762

theorem arithmetic_sequence_sum : 
  ∀ (a₁ l d : ℤ) (n : ℕ),
    a₁ = -48 →
    l = 0 →
    d = 2 →
    n = 25 →
    l = a₁ + (n - 1) * d →
    (n : ℤ) * (a₁ + l) / 2 = -600 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3957_395762


namespace NUMINAMATH_CALUDE_preimage_of_two_zero_l3957_395740

/-- The mapping f from ℝ² to ℝ² defined by f(x, y) = (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that (1, 1) is the preimage of (2, 0) under f -/
theorem preimage_of_two_zero :
  f (1, 1) = (2, 0) ∧ ∀ p : ℝ × ℝ, f p = (2, 0) → p = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_two_zero_l3957_395740


namespace NUMINAMATH_CALUDE_smallest_denominator_between_fractions_l3957_395780

theorem smallest_denominator_between_fractions : 
  ∃ (p q : ℕ), 
    (1 : ℚ) / 2014 < (p : ℚ) / q ∧ 
    (p : ℚ) / q < (1 : ℚ) / 2013 ∧
    q = 4027 ∧
    (∀ (p' q' : ℕ), (1 : ℚ) / 2014 < (p' : ℚ) / q' → (p' : ℚ) / q' < (1 : ℚ) / 2013 → q ≤ q') :=
by sorry

end NUMINAMATH_CALUDE_smallest_denominator_between_fractions_l3957_395780


namespace NUMINAMATH_CALUDE_jimmy_has_more_sheets_l3957_395783

/-- Represents the number of sheets each person has -/
structure Sheets where
  jimmy : ℕ
  tommy : ℕ
  ashton : ℕ

/-- The initial state of sheet distribution -/
def initial : Sheets where
  jimmy := 58
  tommy := 58 + 25
  ashton := 85

/-- The state after Ashton gives sheets to Jimmy -/
def final (s : Sheets) : Sheets where
  jimmy := s.jimmy + s.ashton
  tommy := s.tommy
  ashton := 0

/-- Theorem stating that Jimmy will have 60 more sheets than Tommy after receiving sheets from Ashton -/
theorem jimmy_has_more_sheets : (final initial).jimmy - (final initial).tommy = 60 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_has_more_sheets_l3957_395783


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3957_395704

-- Define the ellipse
def ellipse (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 2) + y^2 / (k + 1) = 1

-- Define the foci
structure Foci (k : ℝ) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

-- Define the chord AB
structure Chord (k : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ

-- Define the eccentricity
def eccentricity (k : ℝ) : ℝ := sorry

theorem ellipse_eccentricity 
  (k : ℝ) 
  (hk : k > -1) 
  (f : Foci k)
  (c : Chord k)
  (hF₁ : c.A.1 = f.F₁.1 ∧ c.A.2 = f.F₁.2) -- Chord AB passes through F₁
  (hPerimeter : Real.sqrt ((c.A.1 - f.F₂.1)^2 + (c.A.2 - f.F₂.2)^2) + 
                Real.sqrt ((c.B.1 - f.F₂.1)^2 + (c.B.2 - f.F₂.2)^2) + 
                Real.sqrt ((c.A.1 - c.B.1)^2 + (c.A.2 - c.B.2)^2) = 8) -- Perimeter of ABF₂ is 8
  : eccentricity k = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_eccentricity_l3957_395704


namespace NUMINAMATH_CALUDE_root_zero_implies_a_half_l3957_395795

theorem root_zero_implies_a_half (a : ℝ) : 
  (∀ x : ℝ, x^2 + x + 2*a - 1 = 0 → x = 0 ∨ x ≠ 0) →
  (0^2 + 0 + 2*a - 1 = 0) →
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_root_zero_implies_a_half_l3957_395795


namespace NUMINAMATH_CALUDE_triangle_area_product_l3957_395791

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → (1/2 * (8/a) * (8/b) = 8) → a * b = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_product_l3957_395791


namespace NUMINAMATH_CALUDE_shaded_area_proof_l3957_395713

theorem shaded_area_proof (carpet_side : ℝ) (S T : ℝ) : 
  carpet_side = 9 →
  carpet_side / S = 3 →
  S / T = 3 →
  S * S + 8 * T * T = 17 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l3957_395713


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3957_395731

theorem algebraic_expression_value (x y : ℝ) 
  (sum_eq : x + y = 2) 
  (diff_eq : x - y = 4) : 
  1 + x^2 - y^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3957_395731


namespace NUMINAMATH_CALUDE_solve_equation_l3957_395710

-- Define the * operation
def star (a b : ℚ) : ℚ := 2 * a + 3 * b

-- Theorem statement
theorem solve_equation (x : ℚ) :
  star 5 (star 7 x) = -4 → x = -56/9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3957_395710


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3957_395768

/-- The quadratic function f(x) = x^2 - 14x + 45 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 45

theorem quadratic_minimum :
  (∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x) ∧
  (∃ (x : ℝ), f x = -4) ∧
  (∀ (y : ℝ), f y ≥ -4) ∧
  f 7 = -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3957_395768


namespace NUMINAMATH_CALUDE_distance_to_directrix_l3957_395705

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ

/-- Point type representing (x, y) coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Distance from point A to directrix of parabola C -/
theorem distance_to_directrix 
  (C : Parabola) 
  (A : Point) 
  (h1 : A.y^2 = 2 * C.p * A.x) 
  (h2 : A.x = 1) 
  (h3 : A.y = Real.sqrt 5) : 
  A.x + C.p / 2 = 9 / 4 := by
  sorry

#check distance_to_directrix

end NUMINAMATH_CALUDE_distance_to_directrix_l3957_395705


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l3957_395774

theorem mean_of_added_numbers (original_list : List ℝ) (x y z : ℝ) :
  original_list.length = 7 →
  original_list.sum / original_list.length = 48 →
  let new_list := original_list ++ [x, y, z]
  new_list.sum / new_list.length = 55 →
  (x + y + z) / 3 = 71 + 1/3 := by
sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l3957_395774


namespace NUMINAMATH_CALUDE_multiply_algebraic_expressions_l3957_395767

theorem multiply_algebraic_expressions (x y : ℝ) :
  6 * x * y^3 * (-1/2 * x^3 * y^2) = -3 * x^4 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_algebraic_expressions_l3957_395767


namespace NUMINAMATH_CALUDE_david_more_than_zachary_pushup_difference_is_thirty_l3957_395703

/-- The number of push-ups David did -/
def david_pushups : ℕ := 37

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 7

/-- David did more push-ups than Zachary -/
theorem david_more_than_zachary : david_pushups > zachary_pushups := by sorry

/-- The difference in push-ups between David and Zachary -/
def pushup_difference : ℕ := david_pushups - zachary_pushups

theorem pushup_difference_is_thirty : pushup_difference = 30 := by sorry

end NUMINAMATH_CALUDE_david_more_than_zachary_pushup_difference_is_thirty_l3957_395703


namespace NUMINAMATH_CALUDE_simplify_expression_l3957_395702

theorem simplify_expression : 5000 * (5000^9) * 2^1000 = 5000^10 * 2^1000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3957_395702


namespace NUMINAMATH_CALUDE_books_in_box_l3957_395737

theorem books_in_box (total : ℕ) (difference : ℕ) (books_a : ℕ) (books_b : ℕ) : 
  total = 20 → 
  difference = 4 → 
  books_a + books_b = total → 
  books_a = books_b + difference → 
  books_a = 12 := by
sorry

end NUMINAMATH_CALUDE_books_in_box_l3957_395737


namespace NUMINAMATH_CALUDE_ten_boys_handshakes_l3957_395797

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n.choose 2

/-- Theorem: When 10 boys each shake hands once with every other boy, there are 45 handshakes -/
theorem ten_boys_handshakes : handshakes 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_boys_handshakes_l3957_395797


namespace NUMINAMATH_CALUDE_joans_clothing_expenditure_l3957_395739

/-- The total amount Joan spent on clothing --/
def total_spent (shorts jacket shirt shoes hat belt : ℝ)
  (jacket_discount shirt_discount : ℝ) (shoes_coupon : ℝ) : ℝ :=
  shorts + (jacket * (1 - jacket_discount)) + (shirt * shirt_discount) +
  (shoes - shoes_coupon) + hat + belt

/-- Theorem stating the total amount Joan spent on clothing --/
theorem joans_clothing_expenditure :
  let shorts : ℝ := 15
  let jacket : ℝ := 14.82
  let shirt : ℝ := 12.51
  let shoes : ℝ := 21.67
  let hat : ℝ := 8.75
  let belt : ℝ := 6.34
  let jacket_discount : ℝ := 0.1  -- 10% discount on jacket
  let shirt_discount : ℝ := 0.5   -- half price for shirt
  let shoes_coupon : ℝ := 3       -- $3 off coupon for shoes
  total_spent shorts jacket shirt shoes hat belt jacket_discount shirt_discount shoes_coupon = 68.353 := by
  sorry


end NUMINAMATH_CALUDE_joans_clothing_expenditure_l3957_395739


namespace NUMINAMATH_CALUDE_fewest_tiles_required_l3957_395766

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ :=
  d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ :=
  feet * 12

/-- The dimensions of a tile in inches -/
def tileDimensions : Dimensions :=
  { length := 2, width := 3 }

/-- The dimensions of the region in feet -/
def regionDimensionsFeet : Dimensions :=
  { length := 4, width := 6 }

/-- The dimensions of the region in inches -/
def regionDimensionsInches : Dimensions :=
  { length := feetToInches regionDimensionsFeet.length,
    width := feetToInches regionDimensionsFeet.width }

/-- Theorem: The fewest number of tiles required to cover the region is 576 -/
theorem fewest_tiles_required :
  (area regionDimensionsInches) / (area tileDimensions) = 576 := by
  sorry

end NUMINAMATH_CALUDE_fewest_tiles_required_l3957_395766


namespace NUMINAMATH_CALUDE_final_number_of_boys_l3957_395794

/-- Given the initial number of boys and additional boys in a school, 
    prove that the final number of boys is the sum of these two numbers. -/
theorem final_number_of_boys 
  (initial_boys : ℕ) 
  (additional_boys : ℕ) : 
  initial_boys + additional_boys = initial_boys + additional_boys :=
by sorry

end NUMINAMATH_CALUDE_final_number_of_boys_l3957_395794


namespace NUMINAMATH_CALUDE_sine_cosine_sum_equals_one_l3957_395720

theorem sine_cosine_sum_equals_one : 
  Real.sin (π / 2 + π / 3) + Real.cos (π / 2 - π / 6) = 1 := by sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_equals_one_l3957_395720


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l3957_395757

theorem smallest_whole_number_above_sum : ∃ (n : ℕ), 
  (n : ℚ) > (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/5 : ℚ) + (6 + 1/6 : ℚ) ∧ 
  n = 19 ∧ 
  ∀ (m : ℕ), m < n → (m : ℚ) ≤ (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/5 : ℚ) + (6 + 1/6 : ℚ) :=
by sorry

#check smallest_whole_number_above_sum

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l3957_395757


namespace NUMINAMATH_CALUDE_lego_volume_proof_l3957_395723

/-- The number of rows of Legos -/
def num_rows : ℕ := 7

/-- The number of columns of Legos -/
def num_columns : ℕ := 5

/-- The number of layers of Legos -/
def num_layers : ℕ := 3

/-- The length of a single Lego in centimeters -/
def lego_length : ℝ := 1

/-- The width of a single Lego in centimeters -/
def lego_width : ℝ := 1

/-- The height of a single Lego in centimeters -/
def lego_height : ℝ := 1

/-- The total volume of stacked Legos in cubic centimeters -/
def total_volume : ℝ := num_rows * num_columns * num_layers * lego_length * lego_width * lego_height

theorem lego_volume_proof : total_volume = 105 := by
  sorry

end NUMINAMATH_CALUDE_lego_volume_proof_l3957_395723


namespace NUMINAMATH_CALUDE_eleventh_number_is_137_l3957_395708

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits add up to 11 -/
def nth_number_with_digit_sum_11 (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 11th number in the sequence is 137 -/
theorem eleventh_number_is_137 : nth_number_with_digit_sum_11 11 = 137 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_is_137_l3957_395708


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_ninth_l3957_395759

theorem last_digit_of_one_over_three_to_ninth (n : ℕ) : n = 3^9 → (1000000000 / n) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_ninth_l3957_395759


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l3957_395711

theorem smallest_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 125 * k)) ∧ 
  (∃ k₁ k₂ k₃ : ℕ, (n + 7) = 125 * k₁ ∧ (n + 7) = 11 * k₂ ∧ (n + 7) = 24 * k₃) ∧
  (∃ k : ℕ, n = 8 * k) ∧
  (n + 7 = 257) → 
  n = 250 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l3957_395711


namespace NUMINAMATH_CALUDE_least_four_digit_with_conditions_l3957_395775

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

def contains_digit (n d : ℕ) : Prop :=
  d ∈ n.digits 10

def divisible_by_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % d = 0

theorem least_four_digit_with_conditions :
  ∀ n : ℕ,
    is_four_digit n ∧
    has_different_digits n ∧
    contains_digit n 5 ∧
    divisible_by_digits n →
    5124 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_with_conditions_l3957_395775


namespace NUMINAMATH_CALUDE_two_thousand_thirteenth_underlined_pair_l3957_395782

/-- The sequence of n values where n and 3^n have the same units digit -/
def underlined_sequence : ℕ → ℕ
| 0 => 1
| n + 1 => underlined_sequence n + 2

/-- The nth pair in the sequence of underlined pairs -/
def nth_underlined_pair (n : ℕ) : ℕ × ℕ :=
  let m := underlined_sequence (n - 1)
  (m, 3^m)

theorem two_thousand_thirteenth_underlined_pair :
  nth_underlined_pair 2013 = (4025, 3^4025) := by
  sorry

end NUMINAMATH_CALUDE_two_thousand_thirteenth_underlined_pair_l3957_395782


namespace NUMINAMATH_CALUDE_increasing_function_composition_l3957_395718

theorem increasing_function_composition (f : ℝ → ℝ) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x > y → f x - f y > x - y) →
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x > y → f (x^2) - f (y^2) > x^6 - y^6) →
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x > y → f (x^3) - f (y^3) > (Real.sqrt 3 / 2) * (x^6 - y^6)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_composition_l3957_395718


namespace NUMINAMATH_CALUDE_spliced_wire_length_l3957_395717

theorem spliced_wire_length 
  (num_pieces : ℕ) 
  (piece_length : ℝ) 
  (overlap : ℝ) 
  (h1 : num_pieces = 15) 
  (h2 : piece_length = 25) 
  (h3 : overlap = 0.5) : 
  (num_pieces * piece_length - (num_pieces - 1) * overlap) / 100 = 3.68 := by
sorry

end NUMINAMATH_CALUDE_spliced_wire_length_l3957_395717


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3957_395770

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x - 1 ≤ 0) ↔ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3957_395770


namespace NUMINAMATH_CALUDE_estimate_red_balls_l3957_395701

/-- Represents the result of drawing a ball -/
inductive BallColor
| Red
| White

/-- Represents a bag of balls -/
structure BallBag where
  totalBalls : Nat
  redBalls : Nat
  whiteBalls : Nat
  totalBalls_eq : totalBalls = redBalls + whiteBalls

/-- Represents the result of multiple draws -/
structure DrawResult where
  totalDraws : Nat
  redDraws : Nat
  whiteDraws : Nat
  totalDraws_eq : totalDraws = redDraws + whiteDraws

/-- Theorem stating the estimated number of red balls -/
theorem estimate_red_balls 
  (bag : BallBag) 
  (draws : DrawResult) 
  (h1 : bag.totalBalls = 8) 
  (h2 : draws.totalDraws = 100) 
  (h3 : draws.redDraws = 75) :
  (bag.totalBalls : ℚ) * (draws.redDraws : ℚ) / (draws.totalDraws : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_estimate_red_balls_l3957_395701


namespace NUMINAMATH_CALUDE_triangle_base_length_l3957_395749

/-- Given a triangle with area 16 m² and height 8 m, prove its base length is 4 m -/
theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 16 → height = 8 → area = (base * height) / 2 → base = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3957_395749


namespace NUMINAMATH_CALUDE_max_square_in_unit_triangle_l3957_395714

/-- A triangle with base and height both equal to √2 maximizes the area of the inscribed square among all unit-area triangles. -/
theorem max_square_in_unit_triangle :
  ∀ (base height : ℝ) (square_side : ℝ),
    base > 0 → height > 0 → square_side > 0 →
    (1/2) * base * height = 1 →
    square_side^2 ≤ 1/2 →
    square_side^2 ≤ (base * height) / (base + height)^2 →
    square_side^2 ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_square_in_unit_triangle_l3957_395714


namespace NUMINAMATH_CALUDE_johns_walking_distance_l3957_395745

/-- Represents the journey of John to his workplace -/
def Johns_Journey (total_distance : ℝ) (skateboard_speed : ℝ) (walking_speed : ℝ) (total_time : ℝ) : Prop :=
  ∃ (skateboard_distance : ℝ) (walking_distance : ℝ),
    skateboard_distance + walking_distance = total_distance ∧
    skateboard_distance / skateboard_speed + walking_distance / walking_speed = total_time ∧
    walking_distance = 5.0

theorem johns_walking_distance :
  Johns_Journey 10 10 6 (66/60) →
  ∃ (walking_distance : ℝ), walking_distance = 5.0 :=
by
  sorry


end NUMINAMATH_CALUDE_johns_walking_distance_l3957_395745


namespace NUMINAMATH_CALUDE_nicole_collected_400_cards_l3957_395719

/-- The number of Pokemon cards Nicole collected -/
def nicole_cards : ℕ := 400

/-- The number of Pokemon cards Cindy collected -/
def cindy_cards : ℕ := 2 * nicole_cards

/-- The number of Pokemon cards Rex collected -/
def rex_cards : ℕ := (nicole_cards + cindy_cards) / 2

/-- The number of people Rex divided his cards among (himself and 3 siblings) -/
def num_people : ℕ := 4

/-- The number of cards Rex has left after dividing -/
def rex_leftover : ℕ := 150

theorem nicole_collected_400_cards :
  nicole_cards = 400 ∧
  cindy_cards = 2 * nicole_cards ∧
  rex_cards = (nicole_cards + cindy_cards) / 2 ∧
  rex_cards = num_people * rex_leftover :=
sorry

end NUMINAMATH_CALUDE_nicole_collected_400_cards_l3957_395719


namespace NUMINAMATH_CALUDE_function_solution_set_l3957_395726

theorem function_solution_set (a : ℝ) : 
  (∀ x : ℝ, (|2*x - a| + a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_solution_set_l3957_395726
