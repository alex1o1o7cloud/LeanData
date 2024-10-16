import Mathlib

namespace NUMINAMATH_CALUDE_special_circle_equation_l3986_398625

/-- A circle with center on the line 2x - y - 3 = 0 passing through (5, 2) and (3, -2) -/
def special_circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - (2*a - 3))^2 = ((5 - a)^2 + (2 - (2*a - 3))^2)}

/-- The equation of the circle is (x-2)^2 + (y-1)^2 = 10 -/
theorem special_circle_equation :
  ∃ a : ℝ, special_circle a = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 10} ∧
    (5, 2) ∈ special_circle a ∧ (3, -2) ∈ special_circle a :=
by
  sorry

end NUMINAMATH_CALUDE_special_circle_equation_l3986_398625


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3986_398613

theorem fraction_subtraction : (18 : ℚ) / 45 - 3 / 8 = 1 / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3986_398613


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_a_in_range_l3986_398619

/-- Given sets A and B, prove that their intersection is non-empty if and only if -1 < a < 3 -/
theorem intersection_nonempty_iff_a_in_range (a : ℝ) :
  let A := {x : ℝ | x - 1 > a^2}
  let B := {x : ℝ | x - 4 < 2*a}
  (∃ x, x ∈ A ∩ B) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_a_in_range_l3986_398619


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l3986_398665

-- Define y as a positive real number
variable (y : ℝ) (hy : y > 0)

-- State the theorem
theorem fourth_root_equivalence : (y^2 * y^(1/2))^(1/4) = y^(5/8) := by sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l3986_398665


namespace NUMINAMATH_CALUDE_inequality_range_l3986_398648

open Real

theorem inequality_range (a : ℝ) : 
  (∀ x > 1, a * log x > 1 - 1/x) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l3986_398648


namespace NUMINAMATH_CALUDE_triangulation_count_equals_catalan_l3986_398672

/-- The number of ways to triangulate a convex polygon -/
def triangulationCount (n : ℕ) : ℕ := sorry

/-- The n-th Catalan number -/
def catalanNumber (n : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to triangulate a convex (n+2)-gon
    is equal to the (n-1)-th Catalan number -/
theorem triangulation_count_equals_catalan (n : ℕ) :
  triangulationCount (n + 2) = catalanNumber (n - 1) := by sorry

end NUMINAMATH_CALUDE_triangulation_count_equals_catalan_l3986_398672


namespace NUMINAMATH_CALUDE_negative_of_negative_is_positive_l3986_398686

theorem negative_of_negative_is_positive (x : ℝ) : x < 0 → -x > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_is_positive_l3986_398686


namespace NUMINAMATH_CALUDE_max_product_with_constraint_l3986_398663

theorem max_product_with_constraint (a b c : ℕ+) (h : a + 2*b + 3*c = 100) :
  a * b * c ≤ 6171 :=
sorry

end NUMINAMATH_CALUDE_max_product_with_constraint_l3986_398663


namespace NUMINAMATH_CALUDE_tea_set_cost_optimization_l3986_398605

/-- Represents the cost calculation for tea sets and bowls under different offers -/
def tea_cost (x : ℕ) : ℕ → ℕ
| 1 => 20 * x + 5400  -- Offer 1
| 2 => 19 * x + 5700  -- Offer 2
| _ => 0  -- Invalid offer

/-- Represents the most cost-effective option -/
def best_option (x : ℕ) : ℕ := 6000 + (x - 30) * 19

theorem tea_set_cost_optimization (x : ℕ) (h : x > 30) :
  best_option x ≤ min (tea_cost x 1) (tea_cost x 2) ∧
  best_option 40 = 6190 := by
  sorry

#eval best_option 40  -- Should output 6190

end NUMINAMATH_CALUDE_tea_set_cost_optimization_l3986_398605


namespace NUMINAMATH_CALUDE_parabola_line_tangency_l3986_398684

theorem parabola_line_tangency (m : ℝ) : 
  (∃ x y : ℝ, y = x^2 ∧ x + y = Real.sqrt m ∧ 
   (∀ x' y' : ℝ, y' = x'^2 → x' + y' = Real.sqrt m → (x', y') = (x, y))) → 
  m = 1/16 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_tangency_l3986_398684


namespace NUMINAMATH_CALUDE_medicine_survey_l3986_398602

theorem medicine_survey (total : ℕ) (cold : ℕ) (stomach : ℕ) 
  (h_total : total = 100)
  (h_cold : cold = 75)
  (h_stomach : stomach = 80)
  (h_cold_le_total : cold ≤ total)
  (h_stomach_le_total : stomach ≤ total) :
  ∃ (max_both min_both : ℕ),
    max_both ≤ cold ∧
    max_both ≤ stomach ∧
    cold + stomach - max_both ≤ total ∧
    max_both = 75 ∧
    min_both ≥ 0 ∧
    min_both ≤ cold ∧
    min_both ≤ stomach ∧
    cold + stomach - min_both ≥ total ∧
    min_both = 55 := by
  sorry

end NUMINAMATH_CALUDE_medicine_survey_l3986_398602


namespace NUMINAMATH_CALUDE_plane_division_l3986_398606

/-- The maximum number of regions that can be created by n lines in a plane -/
def max_regions (n : ℕ) : ℕ := sorry

/-- The number of additional regions created by adding a line that intersects all existing lines -/
def additional_regions (n : ℕ) : ℕ := sorry

theorem plane_division (total_lines : ℕ) (parallel_lines : ℕ) 
  (h1 : total_lines = 10) 
  (h2 : parallel_lines = 4) 
  (h3 : parallel_lines ≤ total_lines) :
  max_regions (total_lines - parallel_lines) + 
  (parallel_lines * additional_regions (total_lines - parallel_lines)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_plane_division_l3986_398606


namespace NUMINAMATH_CALUDE_reducible_factorial_fraction_l3986_398632

theorem reducible_factorial_fraction (n : ℕ) :
  (∃ k : ℕ, k > 1 ∧ k ∣ n.factorial ∧ k ∣ (n + 1)) ↔
  (n % 2 = 1 ∧ n > 1) ∨ (n % 2 = 0 ∧ ¬(Nat.Prime (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_reducible_factorial_fraction_l3986_398632


namespace NUMINAMATH_CALUDE_book_arrangements_eq_34560_l3986_398654

/-- The number of ways to arrange 11 books (3 Arabic, 2 English, 4 Spanish, and 2 French) on a shelf,
    keeping Arabic, Spanish, and English books together respectively. -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 11
  let arabic_books : ℕ := 3
  let english_books : ℕ := 2
  let spanish_books : ℕ := 4
  let french_books : ℕ := 2
  let group_arrangements : ℕ := Nat.factorial 5
  let arabic_internal_arrangements : ℕ := Nat.factorial arabic_books
  let english_internal_arrangements : ℕ := Nat.factorial english_books
  let spanish_internal_arrangements : ℕ := Nat.factorial spanish_books
  group_arrangements * arabic_internal_arrangements * english_internal_arrangements * spanish_internal_arrangements

theorem book_arrangements_eq_34560 : book_arrangements = 34560 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangements_eq_34560_l3986_398654


namespace NUMINAMATH_CALUDE_total_spent_is_450_l3986_398652

/-- The total amount spent by Leonard and Michael on gifts for their father -/
def total_spent (leonard_wallet : ℕ) (leonard_sneakers : ℕ) (leonard_sneakers_count : ℕ)
                (michael_backpack : ℕ) (michael_jeans : ℕ) (michael_jeans_count : ℕ) : ℕ :=
  leonard_wallet + leonard_sneakers * leonard_sneakers_count +
  michael_backpack + michael_jeans * michael_jeans_count

/-- Theorem stating that the total amount spent by Leonard and Michael is $450 -/
theorem total_spent_is_450 :
  total_spent 50 100 2 100 50 2 = 450 := by
  sorry


end NUMINAMATH_CALUDE_total_spent_is_450_l3986_398652


namespace NUMINAMATH_CALUDE_maggies_portion_l3986_398644

theorem maggies_portion (total : ℝ) (maggies_share : ℝ) (debbys_portion : ℝ) :
  total = 6000 →
  maggies_share = 4500 →
  debbys_portion = 0.25 →
  maggies_share / total = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_maggies_portion_l3986_398644


namespace NUMINAMATH_CALUDE_min_abs_z_minus_one_l3986_398682

/-- For any complex number Z satisfying |Z-1| = |Z+1|, the minimum value of |Z-1| is 1. -/
theorem min_abs_z_minus_one (Z : ℂ) (h : Complex.abs (Z - 1) = Complex.abs (Z + 1)) :
  ∃ (min : ℝ), min = 1 ∧ ∀ (W : ℂ), Complex.abs (W - 1) = Complex.abs (W + 1) → Complex.abs (W - 1) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_minus_one_l3986_398682


namespace NUMINAMATH_CALUDE_youtube_views_problem_l3986_398658

/-- Calculates the additional views after the fourth day given the initial views,
    increase factor, and total views after 6 days. -/
def additional_views_after_fourth_day (initial_views : ℕ) (increase_factor : ℕ) (total_views_after_six_days : ℕ) : ℕ :=
  total_views_after_six_days - (initial_views + increase_factor * initial_views)

/-- Theorem stating that given the specific conditions of the problem,
    the additional views after the fourth day is 50000. -/
theorem youtube_views_problem :
  additional_views_after_fourth_day 4000 10 94000 = 50000 := by
  sorry

end NUMINAMATH_CALUDE_youtube_views_problem_l3986_398658


namespace NUMINAMATH_CALUDE_initial_payment_calculation_l3986_398628

theorem initial_payment_calculation (car_cost installment_amount : ℕ) (num_installments : ℕ) 
  (h1 : car_cost = 18000)
  (h2 : installment_amount = 2500)
  (h3 : num_installments = 6) :
  car_cost - (num_installments * installment_amount) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_initial_payment_calculation_l3986_398628


namespace NUMINAMATH_CALUDE_airbnb_rental_cost_l3986_398600

/-- Calculates the Airbnb rental cost for a vacation -/
theorem airbnb_rental_cost 
  (num_people : ℕ) 
  (car_rental_cost : ℕ) 
  (share_per_person : ℕ) 
  (h1 : num_people = 8)
  (h2 : car_rental_cost = 800)
  (h3 : share_per_person = 500) :
  num_people * share_per_person - car_rental_cost = 3200 := by
  sorry

end NUMINAMATH_CALUDE_airbnb_rental_cost_l3986_398600


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3986_398657

def A : Set ℝ := {x | x ≤ 2}
def B : Set ℝ := {x | x^2 - 3*x ≤ 0}

theorem set_intersection_theorem :
  (Set.univ \ A) ∩ B = {x | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3986_398657


namespace NUMINAMATH_CALUDE_count_maximal_arithmetic_sequences_correct_l3986_398604

/-- 
Given a positive integer n, count_maximal_arithmetic_sequences returns the number of 
maximal arithmetic sequences that can be formed from the set {1, 2, ..., n}.
A maximal arithmetic sequence is defined as an arithmetic sequence with a positive 
difference, containing at least two terms from the set, and to which no other element 
from the set can be added while maintaining the arithmetic progression.
-/
def count_maximal_arithmetic_sequences (n : ℕ) : ℕ :=
  (n^2) / 4

theorem count_maximal_arithmetic_sequences_correct (n : ℕ) :
  count_maximal_arithmetic_sequences n = ⌊(n^2 : ℚ) / 4⌋ := by
  sorry

#eval count_maximal_arithmetic_sequences 10  -- Expected output: 25

end NUMINAMATH_CALUDE_count_maximal_arithmetic_sequences_correct_l3986_398604


namespace NUMINAMATH_CALUDE_smallest_valid_side_l3986_398615

-- Define the triangle sides
def a : ℝ := 7.5
def b : ℝ := 14.5

-- Define the property of s being a valid side length
def is_valid_side (s : ℕ) : Prop :=
  (a + s > b) ∧ (a + b > s) ∧ (b + s > a)

-- State the theorem
theorem smallest_valid_side :
  ∃ (s : ℕ), is_valid_side s ∧ ∀ (t : ℕ), t < s → ¬is_valid_side t :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_side_l3986_398615


namespace NUMINAMATH_CALUDE_hyperbola_center_is_3_5_l3986_398697

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 - 24 * x - 25 * y^2 + 250 * y - 489 = 0

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 5)

/-- Theorem: The center of the hyperbola is (3, 5) -/
theorem hyperbola_center_is_3_5 :
  ∀ (x y : ℝ), hyperbola_equation x y →
  ∃ (a b : ℝ), (x - hyperbola_center.1)^2 / a^2 - (y - hyperbola_center.2)^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_3_5_l3986_398697


namespace NUMINAMATH_CALUDE_max_product_constraint_l3986_398637

theorem max_product_constraint (a b : ℝ) (h : a + b = 5) : 
  a * b ≤ 25 / 4 ∧ (a * b = 25 / 4 ↔ a = 5 / 2 ∧ b = 5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l3986_398637


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l3986_398626

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 5)
  (hdb : d / b = 2 / 5) :
  a / c = 125 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l3986_398626


namespace NUMINAMATH_CALUDE_bicycle_installation_problem_l3986_398660

/-- The number of bicycles a skilled worker can install per day -/
def x : ℕ := sorry

/-- The number of bicycles a new worker can install per day -/
def y : ℕ := sorry

/-- The number of skilled workers -/
def a : ℕ := sorry

/-- The number of new workers -/
def b : ℕ := sorry

/-- Theorem stating the conditions and expected results -/
theorem bicycle_installation_problem :
  (2 * x + 3 * y = 44) ∧
  (4 * x = 5 * y) ∧
  (25 * (a * x + b * y) = 3500) →
  (x = 10 ∧ y = 8) ∧
  ((a = 2 ∧ b = 15) ∨ (a = 6 ∧ b = 10) ∨ (a = 10 ∧ b = 5)) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_installation_problem_l3986_398660


namespace NUMINAMATH_CALUDE_largest_possible_b_l3986_398610

theorem largest_possible_b (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (Nat.Prime c) →
  (∀ b' : ℕ, (∃ a' c' : ℕ, 
    (a' * b' * c' = 360) ∧
    (1 < c') ∧
    (c' < b') ∧
    (b' < a') ∧
    (Nat.Prime c')) → b' ≤ b) →
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_largest_possible_b_l3986_398610


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l3986_398659

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 8 + a 9 = 32)
  (h_seventh : a 7 = 1) :
  a 10 = 31 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l3986_398659


namespace NUMINAMATH_CALUDE_symmetry_of_transformed_functions_l3986_398616

/-- Given a function f, prove that the graphs of f(x-1) and f(-x+1) are symmetric with respect to the line x = 1 -/
theorem symmetry_of_transformed_functions (f : ℝ → ℝ) : 
  ∀ (x y : ℝ), f (x - 1) = y ↔ f (-(x - 1)) = y :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_transformed_functions_l3986_398616


namespace NUMINAMATH_CALUDE_leah_bird_feeding_l3986_398624

/-- The number of weeks Leah can feed her birds without going back to the store -/
def weeks_of_feed (boxes_bought : ℕ) (boxes_in_pantry : ℕ) (parrot_consumption : ℕ) (cockatiel_consumption : ℕ) (grams_per_box : ℕ) : ℕ :=
  let total_boxes := boxes_bought + boxes_in_pantry
  let total_grams := total_boxes * grams_per_box
  let weekly_consumption := parrot_consumption + cockatiel_consumption
  total_grams / weekly_consumption

/-- Theorem stating that Leah can feed her birds for 12 weeks -/
theorem leah_bird_feeding :
  weeks_of_feed 3 5 100 50 225 = 12 := by
sorry

end NUMINAMATH_CALUDE_leah_bird_feeding_l3986_398624


namespace NUMINAMATH_CALUDE_ship_capacity_and_tax_calculation_l3986_398691

/-- Represents the types of cargo --/
inductive CargoType
  | Steel
  | Timber
  | Electronics
  | Textiles

/-- Represents a cargo load with its type and weight --/
structure CargoLoad :=
  (type : CargoType)
  (weight : Nat)

/-- Calculates the total weight of a list of cargo loads --/
def totalWeight (loads : List CargoLoad) : Nat :=
  loads.foldl (fun acc load => acc + load.weight) 0

/-- Calculates the import tax for a single cargo load --/
def importTax (load : CargoLoad) : Nat :=
  match load.type with
  | CargoType.Steel => load.weight * 50
  | CargoType.Timber => load.weight * 75
  | CargoType.Electronics => load.weight * 100
  | CargoType.Textiles => load.weight * 40

/-- Calculates the total import tax for a list of cargo loads --/
def totalImportTax (loads : List CargoLoad) : Nat :=
  loads.foldl (fun acc load => acc + importTax load) 0

/-- The main theorem to prove --/
theorem ship_capacity_and_tax_calculation 
  (maxCapacity : Nat)
  (initialCargo : List CargoLoad)
  (additionalCargo : List CargoLoad) :
  maxCapacity = 20000 →
  initialCargo = [
    ⟨CargoType.Steel, 3428⟩,
    ⟨CargoType.Timber, 1244⟩,
    ⟨CargoType.Electronics, 1301⟩
  ] →
  additionalCargo = [
    ⟨CargoType.Steel, 3057⟩,
    ⟨CargoType.Textiles, 2364⟩,
    ⟨CargoType.Timber, 1517⟩,
    ⟨CargoType.Electronics, 1785⟩
  ] →
  totalWeight (initialCargo ++ additionalCargo) ≤ maxCapacity ∧
  totalImportTax (initialCargo ++ additionalCargo) = 934485 :=
by sorry


end NUMINAMATH_CALUDE_ship_capacity_and_tax_calculation_l3986_398691


namespace NUMINAMATH_CALUDE_product_of_primes_l3986_398639

theorem product_of_primes : 
  ∃ (p q : Nat), 
    Prime p ∧ 
    Prime q ∧ 
    p = 1021031 ∧ 
    q = 237019 ∧ 
    p * q = 241940557349 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_l3986_398639


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3986_398675

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 400)
  (h2 : rate = 15)
  (h3 : time = 2) :
  (principal * rate * time) / 100 = 60 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3986_398675


namespace NUMINAMATH_CALUDE_square_area_ratio_l3986_398690

theorem square_area_ratio (small_side : ℝ) (large_side : ℝ) 
  (h1 : small_side = 2) 
  (h2 : large_side = 5) : 
  (small_side^2) / ((large_side^2 / 2) - (small_side^2 / 2)) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3986_398690


namespace NUMINAMATH_CALUDE_total_candies_l3986_398685

/-- The number of candies each person has -/
structure Candies where
  adam : ℕ
  james : ℕ
  rubert : ℕ
  lisa : ℕ
  chris : ℕ
  max : ℕ
  emily : ℕ

/-- The conditions of the candy distribution -/
def candy_conditions (c : Candies) : Prop :=
  c.adam = 6 ∧
  c.james = 3 * c.adam ∧
  c.rubert = 4 * c.james ∧
  c.lisa = 2 * c.rubert - 5 ∧
  c.chris = c.lisa / 2 + 7 ∧
  c.max = c.rubert + c.chris + 2 ∧
  c.emily = 3 * c.chris - (c.max - c.lisa)

/-- The theorem stating the total number of candies -/
theorem total_candies (c : Candies) (h : candy_conditions c) : 
  c.adam + c.james + c.rubert + c.lisa + c.chris + c.max + c.emily = 678 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l3986_398685


namespace NUMINAMATH_CALUDE_sum_of_xy_is_30_l3986_398687

-- Define the matrix
def matrix (x y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, 5, 10],
    ![4, x, y],
    ![4, y, x]]

-- State the theorem
theorem sum_of_xy_is_30 (x y : ℝ) (h1 : x ≠ y) (h2 : Matrix.det (matrix x y) = 0) :
  x + y = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_is_30_l3986_398687


namespace NUMINAMATH_CALUDE_right_triangle_area_l3986_398621

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 5) (h3 : a = 4) :
  (1/2) * a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3986_398621


namespace NUMINAMATH_CALUDE_parabola_from_circles_l3986_398670

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + y - 3 = 0

-- Define the directrix
def directrix (y : ℝ) : Prop := y = -1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

theorem parabola_from_circles :
  ∀ (x y : ℝ),
  (∃ (x₁ y₁ x₂ y₂ : ℝ), circle1 x₁ y₁ ∧ circle2 x₂ y₂ ∧ directrix y₁ ∧ directrix y₂) →
  parabola x y :=
by sorry

end NUMINAMATH_CALUDE_parabola_from_circles_l3986_398670


namespace NUMINAMATH_CALUDE_fraction_between_l3986_398651

theorem fraction_between (p q : ℕ+) (h1 : (6 : ℚ) / 11 < p / q) (h2 : p / q < (5 : ℚ) / 9) 
  (h3 : ∀ (r s : ℕ+), (6 : ℚ) / 11 < r / s → r / s < (5 : ℚ) / 9 → s ≥ q) : 
  p + q = 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_between_l3986_398651


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l3986_398693

theorem difference_of_squares_example : (527 : ℤ) * 527 - 526 * 528 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l3986_398693


namespace NUMINAMATH_CALUDE_fraction_simplification_l3986_398641

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hab : a ≠ b) :
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3986_398641


namespace NUMINAMATH_CALUDE_triangle_congruence_criteria_triangle_congruence_criteria_2_l3986_398608

-- Define the structure for a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define side lengths
def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define angle measure
def angle_measure (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem triangle_congruence_criteria (ABC A'B'C' : Triangle) :
  (side_length ABC.A ABC.B = side_length A'B'C'.A A'B'C'.B ∧
   side_length ABC.B ABC.C = side_length A'B'C'.B A'B'C'.C ∧
   side_length ABC.A ABC.C = side_length A'B'C'.A A'B'C'.C) →
  congruent ABC A'B'C' :=
sorry

theorem triangle_congruence_criteria_2 (ABC A'B'C' : Triangle) :
  (side_length ABC.A ABC.B = side_length A'B'C'.A A'B'C'.B ∧
   angle_measure ABC.A ABC.B ABC.C = angle_measure A'B'C'.A A'B'C'.B A'B'C'.C ∧
   angle_measure ABC.B ABC.C ABC.A = angle_measure A'B'C'.B A'B'C'.C A'B'C'.A) →
  congruent ABC A'B'C' :=
sorry

end NUMINAMATH_CALUDE_triangle_congruence_criteria_triangle_congruence_criteria_2_l3986_398608


namespace NUMINAMATH_CALUDE_divisor_between_l3986_398681

theorem divisor_between (n a b : ℕ) (hn : n > 8) (ha : 0 < a) (hb : 0 < b)
  (hab : a < b) (hdiv_a : a ∣ n) (hdiv_b : b ∣ n) (hneq : a ≠ b)
  (heq : n = a^2 + b) : 
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b := by
sorry

end NUMINAMATH_CALUDE_divisor_between_l3986_398681


namespace NUMINAMATH_CALUDE_inclination_angle_range_l3986_398646

theorem inclination_angle_range (α : Real) (h : α ∈ Set.Icc (π / 6) (2 * π / 3)) :
  let θ := Real.arctan (2 * Real.cos α)
  θ ∈ Set.Icc 0 (π / 3) ∪ Set.Ico (3 * π / 4) π := by
  sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l3986_398646


namespace NUMINAMATH_CALUDE_polyhedron_space_diagonals_l3986_398662

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem: A convex polyhedron Q with 30 vertices, 72 edges, 44 faces 
    (30 triangular and 14 quadrilateral) has 335 space diagonals -/
theorem polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 30,
    quadrilateral_faces := 14
  }
  space_diagonals Q = 335 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_space_diagonals_l3986_398662


namespace NUMINAMATH_CALUDE_average_bracelets_per_day_l3986_398603

def bike_cost : ℕ := 112
def selling_weeks : ℕ := 2
def bracelet_price : ℕ := 1
def days_per_week : ℕ := 7

theorem average_bracelets_per_day :
  (bike_cost / (selling_weeks * days_per_week)) / bracelet_price = 8 :=
by sorry

end NUMINAMATH_CALUDE_average_bracelets_per_day_l3986_398603


namespace NUMINAMATH_CALUDE_negation_of_existence_cube_lt_pow_three_negation_l3986_398612

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem cube_lt_pow_three_negation :
  (¬ ∃ x : ℕ, x^3 < 3^x) ↔ (∀ x : ℕ, x^3 ≥ 3^x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_cube_lt_pow_three_negation_l3986_398612


namespace NUMINAMATH_CALUDE_unique_five_numbers_l3986_398695

theorem unique_five_numbers : 
  ∃! (a b c d e : ℕ), 
    a < b ∧ b < c ∧ c < d ∧ d < e ∧
    a * b > 25 ∧ d * e < 75 ∧
    a = 5 ∧ b = 6 ∧ c = 7 ∧ d = 8 ∧ e = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_five_numbers_l3986_398695


namespace NUMINAMATH_CALUDE_plane_contains_points_plane_uniqueness_l3986_398620

/-- A plane in 3D space defined by its equation coefficients -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- A point in 3D space with integer coordinates -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Check if a point lies on a plane -/
def Point3D.liesOn (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- The specific plane we're proving about -/
def targetPlane : Plane := {
  A := 1
  B := 3
  C := -2
  D := -11
  A_pos := by simp
  gcd_one := by sorry
}

/-- The three points given in the problem -/
def p1 : Point3D := ⟨0, 3, -1⟩
def p2 : Point3D := ⟨2, 3, 1⟩
def p3 : Point3D := ⟨4, 1, 0⟩

/-- The main theorem stating that the target plane contains the three given points -/
theorem plane_contains_points : 
  p1.liesOn targetPlane ∧ p2.liesOn targetPlane ∧ p3.liesOn targetPlane :=
by sorry

/-- The theorem stating that the target plane is unique -/
theorem plane_uniqueness (plane : Plane) :
  p1.liesOn plane ∧ p2.liesOn plane ∧ p3.liesOn plane → plane = targetPlane :=
by sorry

end NUMINAMATH_CALUDE_plane_contains_points_plane_uniqueness_l3986_398620


namespace NUMINAMATH_CALUDE_range_of_f_less_than_one_l3986_398630

-- Define the function f
def f (x : ℝ) := x^3

-- State the theorem
theorem range_of_f_less_than_one :
  {x : ℝ | f x < 1} = Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_less_than_one_l3986_398630


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3986_398683

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, 2 * x^2 + 1 > 0)) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3986_398683


namespace NUMINAMATH_CALUDE_v_2002_equals_2_l3986_398640

def g : ℕ → ℕ
  | 1 => 5
  | 2 => 3
  | 3 => 1
  | 4 => 2
  | 5 => 4
  | _ => 0  -- Default case for completeness

def v : ℕ → ℕ
  | 0 => 5
  | n + 1 => g (v n)

theorem v_2002_equals_2 : v 2002 = 2 := by
  sorry

end NUMINAMATH_CALUDE_v_2002_equals_2_l3986_398640


namespace NUMINAMATH_CALUDE_tree_height_problem_l3986_398622

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 20 →  -- One tree is 20 feet taller than the other
  h₂ / h₁ = 5 / 7 →  -- The heights are in the ratio 5:7
  h₁ = 70 :=  -- The taller tree is 70 feet tall
by
  sorry

end NUMINAMATH_CALUDE_tree_height_problem_l3986_398622


namespace NUMINAMATH_CALUDE_negation_equivalence_l3986_398631

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3986_398631


namespace NUMINAMATH_CALUDE_f_composition_one_third_l3986_398634

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 8^x

-- State the theorem
theorem f_composition_one_third : f (f (1/3)) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_one_third_l3986_398634


namespace NUMINAMATH_CALUDE_simplified_expression_evaluation_l3986_398696

theorem simplified_expression_evaluation (x y : ℝ) 
  (hx : x = -1) (hy : y = 1/2) : 
  2 * (3 * x^2 + x * y^2) - 3 * (2 * x * y^2 - x^2) - 10 * x^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_evaluation_l3986_398696


namespace NUMINAMATH_CALUDE_square_count_figure_100_l3986_398679

/-- Represents the number of squares in the nth figure -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem square_count_figure_100 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 100 = 30301 := by
  sorry

end NUMINAMATH_CALUDE_square_count_figure_100_l3986_398679


namespace NUMINAMATH_CALUDE_base_7_addition_l3986_398647

/-- Addition in base 7 -/
def add_base_7 (a b : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 7 -/
def to_base_7 (n : ℕ) : ℕ := sorry

/-- Conversion from base 7 to base 10 -/
def from_base_7 (n : ℕ) : ℕ := sorry

theorem base_7_addition : add_base_7 (from_base_7 25) (from_base_7 256) = from_base_7 544 := by
  sorry

end NUMINAMATH_CALUDE_base_7_addition_l3986_398647


namespace NUMINAMATH_CALUDE_range_of_m_l3986_398669

def P (x : ℝ) : Prop := x^2 - 4*x - 12 ≤ 0

def Q (x m : ℝ) : Prop := |x - m| ≤ m^2

theorem range_of_m : 
  ∀ m : ℝ, (∀ x : ℝ, P x → Q x m) ∧ 
            (∃ x : ℝ, Q x m ∧ ¬P x) ∧ 
            (∃ x : ℝ, P x) 
  ↔ m ≤ -3 ∨ m > 2 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l3986_398669


namespace NUMINAMATH_CALUDE_count_decimal_parts_l3986_398677

theorem count_decimal_parts : 
  (0.1 / 0.001 = 100) ∧ (1 / 0.01 = 100) := by
  sorry

end NUMINAMATH_CALUDE_count_decimal_parts_l3986_398677


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l3986_398661

theorem right_triangle_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x ≤ x + y → x + y ≤ x + 3*y →
  (x + 3*y)^2 = x^2 + (x + y)^2 →
  x / y = 1 + Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l3986_398661


namespace NUMINAMATH_CALUDE_signal_count_theorem_l3986_398614

/-- Represents the number of indicator lights --/
def num_lights : Nat := 6

/-- Represents the number of lights that light up each time --/
def lights_lit : Nat := 3

/-- Represents the number of possible colors for each light --/
def num_colors : Nat := 3

/-- Calculates the total number of different signals that can be displayed --/
def total_signals : Nat :=
  -- The actual calculation is not provided, so we use a placeholder
  324

/-- Theorem stating that the total number of different signals is 324 --/
theorem signal_count_theorem :
  total_signals = 324 := by
  sorry

end NUMINAMATH_CALUDE_signal_count_theorem_l3986_398614


namespace NUMINAMATH_CALUDE_travel_group_average_age_l3986_398633

theorem travel_group_average_age 
  (num_men : ℕ) 
  (num_women : ℕ) 
  (avg_age_men : ℚ) 
  (avg_age_women : ℚ) 
  (h1 : num_men = 6) 
  (h2 : num_women = 9) 
  (h3 : avg_age_men = 57) 
  (h4 : avg_age_women = 52) :
  (num_men * avg_age_men + num_women * avg_age_women) / (num_men + num_women) = 54 := by
sorry

end NUMINAMATH_CALUDE_travel_group_average_age_l3986_398633


namespace NUMINAMATH_CALUDE_common_divisors_9240_10010_l3986_398609

theorem common_divisors_9240_10010 : 
  (Finset.filter (fun d => d ∣ 9240 ∧ d ∣ 10010) (Finset.range 10011)).card = 32 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_9240_10010_l3986_398609


namespace NUMINAMATH_CALUDE_difference_of_squares_l3986_398650

theorem difference_of_squares (x y : ℝ) : (x + 2*y) * (x - 2*y) = x^2 - 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3986_398650


namespace NUMINAMATH_CALUDE_odd_digits_base4_157_l3986_398643

/-- Converts a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of odd digits in the base-4 representation of 157 is 3 -/
theorem odd_digits_base4_157 : countOddDigits (toBase4 157) = 3 :=
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_157_l3986_398643


namespace NUMINAMATH_CALUDE_total_octopus_legs_l3986_398649

/-- The number of legs an octopus has -/
def legs_per_octopus : ℕ := 8

/-- The number of octopuses Carson saw -/
def octopuses_seen : ℕ := 5

/-- The total number of octopus legs Carson saw -/
def total_legs : ℕ := octopuses_seen * legs_per_octopus

theorem total_octopus_legs : total_legs = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_octopus_legs_l3986_398649


namespace NUMINAMATH_CALUDE_intersection_locus_is_circle_l3986_398668

/-- The locus of points (x, y) satisfying both equations 2ux - 3y - 2u = 0 and x - 3uy + 2 = 0,
    where u is a real parameter, is a circle. -/
theorem intersection_locus_is_circle :
  ∀ (x y u : ℝ), (2 * u * x - 3 * y - 2 * u = 0) ∧ (x - 3 * u * y + 2 = 0) →
  ∃ (c : ℝ × ℝ) (r : ℝ), (x - c.1)^2 + (y - c.2)^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_locus_is_circle_l3986_398668


namespace NUMINAMATH_CALUDE_height_pillar_E_l3986_398636

/-- Regular octagon with pillars -/
structure OctagonWithPillars where
  /-- Side length of the octagon -/
  side_length : ℝ
  /-- Height of pillar at vertex A -/
  height_A : ℝ
  /-- Height of pillar at vertex B -/
  height_B : ℝ
  /-- Height of pillar at vertex C -/
  height_C : ℝ

/-- Theorem: Height of pillar at E in a regular octagon with given pillar heights -/
theorem height_pillar_E (octagon : OctagonWithPillars) 
  (h_A : octagon.height_A = 15)
  (h_B : octagon.height_B = 12)
  (h_C : octagon.height_C = 13) :
  ∃ (height_E : ℝ), height_E = 5 := by
  sorry

end NUMINAMATH_CALUDE_height_pillar_E_l3986_398636


namespace NUMINAMATH_CALUDE_least_largest_factor_l3986_398664

theorem least_largest_factor (a b c d e : ℕ+) : 
  a * b * c * d * e = 55 * 60 * 65 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e →
  (∀ x y z w v : ℕ+, 
    x * y * z * w * v = 55 * 60 * 65 ∧ 
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧ 
    y ≠ z ∧ y ≠ w ∧ y ≠ v ∧ 
    z ≠ w ∧ z ≠ v ∧ 
    w ≠ v →
    max a (max b (max c (max d e))) ≤ max x (max y (max z (max w v)))) →
  max a (max b (max c (max d e))) = 13 :=
by sorry

end NUMINAMATH_CALUDE_least_largest_factor_l3986_398664


namespace NUMINAMATH_CALUDE_equality_in_different_bases_l3986_398635

theorem equality_in_different_bases : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 
  (3 * a^2 + 4 * a + 2 : ℕ) = (9 * b + 7 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_equality_in_different_bases_l3986_398635


namespace NUMINAMATH_CALUDE_probability_three_red_balls_l3986_398698

/-- The probability of picking 3 red balls from a bag containing 4 red, 5 blue, and 3 green balls -/
theorem probability_three_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) :
  total_balls = red_balls + blue_balls + green_balls →
  red_balls = 4 →
  blue_balls = 5 →
  green_balls = 3 →
  (red_balls : ℚ) / total_balls * (red_balls - 1) / (total_balls - 1) * (red_balls - 2) / (total_balls - 2) = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_red_balls_l3986_398698


namespace NUMINAMATH_CALUDE_student_rabbit_difference_l3986_398666

-- Define the number of students per classroom
def students_per_classroom : ℕ := 24

-- Define the number of rabbits per classroom
def rabbits_per_classroom : ℕ := 3

-- Define the total number of classrooms
def total_classrooms : ℕ := 5

-- Define the number of absent rabbits
def absent_rabbits : ℕ := 1

-- Theorem statement
theorem student_rabbit_difference :
  students_per_classroom * total_classrooms - 
  (rabbits_per_classroom * total_classrooms) = 105 := by
  sorry


end NUMINAMATH_CALUDE_student_rabbit_difference_l3986_398666


namespace NUMINAMATH_CALUDE_f_leq_one_iff_a_range_l3986_398618

-- Define the function f
def f (a x : ℝ) : ℝ := 5 - |x + a| - |x - 2|

-- State the theorem
theorem f_leq_one_iff_a_range (a : ℝ) :
  (∀ x, f a x ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_f_leq_one_iff_a_range_l3986_398618


namespace NUMINAMATH_CALUDE_certain_number_problem_l3986_398678

theorem certain_number_problem (y : ℝ) : 
  (0.25 * 780 = 0.15 * y - 30) → y = 1500 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3986_398678


namespace NUMINAMATH_CALUDE_distance_between_cities_l3986_398671

theorem distance_between_cities (v1 v2 t_diff : ℝ) (h1 : v1 = 60) (h2 : v2 = 70) (h3 : t_diff = 0.25) :
  let t := (v2 * t_diff) / (v2 - v1)
  v1 * t = 105 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_cities_l3986_398671


namespace NUMINAMATH_CALUDE_cubic_coefficient_b_value_l3986_398623

/-- A cubic function passing through specific points -/
def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- Theorem stating that for a cubic function passing through (2,0), (-1,0), and (1,4), b = 6 -/
theorem cubic_coefficient_b_value 
  (a b c d : ℝ) 
  (h1 : g a b c d 2 = 0)
  (h2 : g a b c d (-1) = 0)
  (h3 : g a b c d 1 = 4) :
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_coefficient_b_value_l3986_398623


namespace NUMINAMATH_CALUDE_smaller_number_proof_l3986_398601

theorem smaller_number_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 45) (h4 : y = 4 * x) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l3986_398601


namespace NUMINAMATH_CALUDE_max_salary_is_400000_l3986_398656

/-- Represents a baseball team -/
structure BaseballTeam where
  players : ℕ
  minSalary : ℕ
  maxTotalSalary : ℕ

/-- Calculates the maximum possible salary for a single player -/
def maxSinglePlayerSalary (team : BaseballTeam) : ℕ :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem: The maximum salary for a single player in the given conditions is $400,000 -/
theorem max_salary_is_400000 (team : BaseballTeam)
  (h1 : team.players = 21)
  (h2 : team.minSalary = 15000)
  (h3 : team.maxTotalSalary = 700000) :
  maxSinglePlayerSalary team = 400000 := by
  sorry

#eval maxSinglePlayerSalary { players := 21, minSalary := 15000, maxTotalSalary := 700000 }

end NUMINAMATH_CALUDE_max_salary_is_400000_l3986_398656


namespace NUMINAMATH_CALUDE_expression_simplification_l3986_398655

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = 2 - Real.sqrt 2) : 
  (a / (a^2 - b^2) - 1 / (a + b)) / (b / (b - a)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3986_398655


namespace NUMINAMATH_CALUDE_unique_number_with_specific_factors_l3986_398694

theorem unique_number_with_specific_factors :
  ∀ (x n : ℕ),
  x = 7^n + 1 →
  Odd n →
  (∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2 * 11 * p * q) →
  x = 16808 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_specific_factors_l3986_398694


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3986_398667

/-- A point M with coordinates (a-2, a+1) lies on the x-axis if and only if its coordinates are (-3, 0) -/
theorem point_on_x_axis (a : ℝ) : 
  (a + 1 = 0 ∧ (a - 2, a + 1) = (-3, 0)) ↔ (a - 2, a + 1) = (-3, 0) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3986_398667


namespace NUMINAMATH_CALUDE_abs_not_always_positive_l3986_398611

theorem abs_not_always_positive : ¬ (∀ x : ℝ, |x| > 0) := by
  sorry

end NUMINAMATH_CALUDE_abs_not_always_positive_l3986_398611


namespace NUMINAMATH_CALUDE_triangle_problem_l3986_398607

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  2 * a * Real.sin A = (2 * b + c) * Real.sin B + (2 * c + b) * Real.sin C →
  a = 7 →
  a * (15 * Real.sqrt 3 / 14) / 2 = b * c * Real.sin A →
  (A = 2 * π / 3 ∧ ((b = 3 ∧ c = 5) ∨ (b = 5 ∧ c = 3))) :=
by sorry


end NUMINAMATH_CALUDE_triangle_problem_l3986_398607


namespace NUMINAMATH_CALUDE_max_figures_9x9_grid_l3986_398699

/-- Represents a square grid -/
structure Grid (n : ℕ) :=
  (size : ℕ)
  (h_size : size = n)

/-- Represents a square figure -/
structure Figure (m : ℕ) :=
  (size : ℕ)
  (h_size : size = m)

/-- The maximum number of non-overlapping figures that can fit in a grid -/
def max_figures (g : Grid n) (f : Figure m) : ℕ :=
  (g.size / f.size) ^ 2

/-- Theorem: The maximum number of non-overlapping 2x2 squares in a 9x9 grid is 16 -/
theorem max_figures_9x9_grid :
  ∀ (g : Grid 9) (f : Figure 2),
    max_figures g f = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_figures_9x9_grid_l3986_398699


namespace NUMINAMATH_CALUDE_problem_solution_l3986_398673

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi/6) + Real.cos (x - Real.pi/3)

noncomputable def g (x : ℝ) : ℝ := 2 * (Real.sin (x/2))^2

theorem problem_solution (θ : ℝ) (k : ℤ) :
  (0 < θ ∧ θ < Real.pi/2) →  -- θ is in the first quadrant
  f θ = 3 * Real.sqrt 3 / 5 →
  g θ = 1/5 ∧
  (∀ x, f x ≥ g x ↔ ∃ k, 2 * k * Real.pi ≤ x ∧ x ≤ 2 * k * Real.pi + 2 * Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3986_398673


namespace NUMINAMATH_CALUDE_fourth_episode_duration_l3986_398680

theorem fourth_episode_duration (episode1 episode2 episode3 : ℕ) 
  (total_duration : ℕ) (h1 : episode1 = 58) (h2 : episode2 = 62) 
  (h3 : episode3 = 65) (h4 : total_duration = 4 * 60) : 
  total_duration - (episode1 + episode2 + episode3) = 55 := by
  sorry

end NUMINAMATH_CALUDE_fourth_episode_duration_l3986_398680


namespace NUMINAMATH_CALUDE_max_volume_right_prism_l3986_398674

theorem max_volume_right_prism (b c : ℝ) (h1 : b + c = 8) (h2 : b > 0) (h3 : c > 0) :
  let volume := fun x => (1/2) * b * x^2
  let x := Real.sqrt (64 - 16*b)
  (∀ y, volume y ≤ volume x) ∧ volume x = 32 := by
  sorry

end NUMINAMATH_CALUDE_max_volume_right_prism_l3986_398674


namespace NUMINAMATH_CALUDE_trapezoid_cd_length_l3986_398689

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  /-- Length of side BD -/
  bd : ℝ
  /-- Angle DBA in radians -/
  angle_dba : ℝ
  /-- Angle BDC in radians -/
  angle_bdc : ℝ
  /-- Ratio of BC to AD -/
  ratio_bc_ad : ℝ
  /-- AD is parallel to BC -/
  ad_parallel_bc : True
  /-- BD equals 3 -/
  bd_eq_three : bd = 3
  /-- Angle DBA equals 30 degrees (π/6 radians) -/
  angle_dba_eq_thirty_deg : angle_dba = Real.pi / 6
  /-- Angle BDC equals 60 degrees (π/3 radians) -/
  angle_bdc_eq_sixty_deg : angle_bdc = Real.pi / 3
  /-- Ratio of BC to AD is 7:4 -/
  ratio_bc_ad_eq_seven_four : ratio_bc_ad = 7 / 4

/-- Theorem: In the given trapezoid, CD equals 9/4 -/
theorem trapezoid_cd_length (t : Trapezoid) : ∃ cd : ℝ, cd = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_cd_length_l3986_398689


namespace NUMINAMATH_CALUDE_remainder_1234567_div_12_l3986_398638

theorem remainder_1234567_div_12 : 1234567 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567_div_12_l3986_398638


namespace NUMINAMATH_CALUDE_range_of_expressions_l3986_398676

theorem range_of_expressions (a b : ℝ) 
  (ha : -6 < a ∧ a < 8) 
  (hb : 2 < b ∧ b < 3) : 
  (-10 < 2*a + b ∧ 2*a + b < 19) ∧ 
  (-9 < a - b ∧ a - b < 6) ∧ 
  (-2 < a / b ∧ a / b < 4) := by
sorry

end NUMINAMATH_CALUDE_range_of_expressions_l3986_398676


namespace NUMINAMATH_CALUDE_puppies_calculation_l3986_398692

/-- The number of puppies Alyssa initially had -/
def initial_puppies : ℕ := 12

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has now -/
def remaining_puppies : ℕ := initial_puppies - puppies_given_away

theorem puppies_calculation : remaining_puppies = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppies_calculation_l3986_398692


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3986_398629

theorem smallest_integer_with_given_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
    x % 5 = 2 ∧ 
    x % 3 = 1 ∧ 
    x % 7 = 3 ∧
    (∀ y : ℕ, y > 0 → y % 5 = 2 → y % 3 = 1 → y % 7 = 3 → x ≤ y) ∧
    x = 22 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3986_398629


namespace NUMINAMATH_CALUDE_parallel_to_a_l3986_398642

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

/-- The vector a is defined as (-5, 4) -/
def a : ℝ × ℝ := (-5, 4)

/-- Theorem: A vector (x, y) is parallel to a = (-5, 4) if and only if
    there exists a real number k such that (x, y) = (-5k, 4k) -/
theorem parallel_to_a (x y : ℝ) :
  parallel (x, y) a ↔ ∃ k : ℝ, (x, y) = (-5 * k, 4 * k) :=
sorry

end NUMINAMATH_CALUDE_parallel_to_a_l3986_398642


namespace NUMINAMATH_CALUDE_peters_children_l3986_398627

theorem peters_children (initial_savings : ℕ) (addition : ℕ) (num_children : ℕ) : 
  initial_savings = 642986 →
  addition = 642987 →
  (initial_savings + addition) % num_children = 0 →
  num_children = 642987 := by
sorry

end NUMINAMATH_CALUDE_peters_children_l3986_398627


namespace NUMINAMATH_CALUDE_hiker_catches_cyclist_l3986_398617

/-- Proves that a hiker catches up to a cyclist in 15 minutes under specific conditions --/
theorem hiker_catches_cyclist (hiker_speed cyclist_speed : ℝ) (stop_time : ℝ) : 
  hiker_speed = 7 →
  cyclist_speed = 28 →
  stop_time = 5 / 60 →
  let distance_cyclist := cyclist_speed * stop_time
  let distance_hiker := hiker_speed * stop_time
  let distance_difference := distance_cyclist - distance_hiker
  let catch_up_time := distance_difference / hiker_speed
  catch_up_time * 60 = 15 := by
  sorry

#check hiker_catches_cyclist

end NUMINAMATH_CALUDE_hiker_catches_cyclist_l3986_398617


namespace NUMINAMATH_CALUDE_unique_a_value_l3986_398645

-- Define the sets M and N as functions of a
def M (a : ℝ) : Set ℝ := {1, 2, a^2 - 3*a - 1}
def N (a : ℝ) : Set ℝ := {-1, a, 3}

-- State the theorem
theorem unique_a_value : ∃! a : ℝ, (M a ∩ N a = {3} ∧ a ≠ -1) := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l3986_398645


namespace NUMINAMATH_CALUDE_vegetable_ghee_weight_l3986_398653

/-- The weight of one liter of vegetable ghee for brand 'b' in grams -/
def weight_b : ℝ := 850

/-- The ratio of brand 'a' to brand 'b' in the mixture by volume -/
def mixture_ratio : ℚ := 3/2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3640

/-- The weight of one liter of vegetable ghee for brand 'a' in grams -/
def weight_a : ℝ := 950

theorem vegetable_ghee_weight : 
  (weight_a * (mixture_ratio / (mixture_ratio + 1)) * total_volume) + 
  (weight_b * (1 / (mixture_ratio + 1)) * total_volume) = total_weight :=
sorry

end NUMINAMATH_CALUDE_vegetable_ghee_weight_l3986_398653


namespace NUMINAMATH_CALUDE_total_symbol_count_is_62_l3986_398688

/-- The number of distinct symbols that can be represented by a sequence of dots and dashes of a given length. -/
def symbolCount (length : Nat) : Nat :=
  2^length

/-- The total number of distinct symbols that can be represented using sequences of 1 to 5 dots and/or dashes. -/
def totalSymbolCount : Nat :=
  (symbolCount 1) + (symbolCount 2) + (symbolCount 3) + (symbolCount 4) + (symbolCount 5)

/-- Theorem stating that the total number of distinct symbols is 62. -/
theorem total_symbol_count_is_62 : totalSymbolCount = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_symbol_count_is_62_l3986_398688
