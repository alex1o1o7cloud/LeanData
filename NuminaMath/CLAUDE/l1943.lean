import Mathlib

namespace NUMINAMATH_CALUDE_linear_function_midpoint_property_quadratic_function_midpoint_property_l1943_194321

/-- Linear function property -/
theorem linear_function_midpoint_property (a b x₁ x₂ : ℝ) :
  let f := fun x => a * x + b
  f ((x₁ + x₂) / 2) = (f x₁ + f x₂) / 2 := by sorry

/-- Quadratic function property -/
theorem quadratic_function_midpoint_property (a b x₁ x₂ : ℝ) :
  let g := fun x => x^2 + a * x + b
  g ((x₁ + x₂) / 2) ≤ (g x₁ + g x₂) / 2 := by sorry

end NUMINAMATH_CALUDE_linear_function_midpoint_property_quadratic_function_midpoint_property_l1943_194321


namespace NUMINAMATH_CALUDE_min_buses_for_535_students_l1943_194380

/-- The minimum number of buses needed to transport a given number of students -/
def min_buses (capacity : ℕ) (students : ℕ) : ℕ :=
  (students + capacity - 1) / capacity

/-- Theorem: Given a bus capacity of 45 students and 535 students to transport,
    the minimum number of buses needed is 12 -/
theorem min_buses_for_535_students :
  min_buses 45 535 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_buses_for_535_students_l1943_194380


namespace NUMINAMATH_CALUDE_remainder_of_x_l1943_194346

theorem remainder_of_x (x : ℤ) 
  (h1 : (2 + x) % (4^3) = 3^2 % (4^3))
  (h2 : (3 + x) % (5^3) = 5^2 % (5^3))
  (h3 : (6 + x) % (11^3) = 7^2 % (11^3)) :
  x % 220 = 192 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_x_l1943_194346


namespace NUMINAMATH_CALUDE_solve_equation_l1943_194330

theorem solve_equation (y : ℚ) (h : 1/3 - 1/4 = 1/y) : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1943_194330


namespace NUMINAMATH_CALUDE_walking_distance_l1943_194387

theorem walking_distance (x y : ℝ) 
  (h1 : x / 4 + y / 3 + y / 6 + x / 4 = 5) : x + y = 10 ∧ 2 * (x + y) = 20 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_l1943_194387


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1943_194326

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {9, a-5, 1-a}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {9} → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1943_194326


namespace NUMINAMATH_CALUDE_ball_radius_for_given_hole_l1943_194393

/-- The radius of a spherical ball that leaves a circular hole with given dimensions -/
def ball_radius (hole_diameter : ℝ) (hole_depth : ℝ) : ℝ :=
  13

/-- Theorem stating that a ball leaving a hole with diameter 24 cm and depth 8 cm has a radius of 13 cm -/
theorem ball_radius_for_given_hole : ball_radius 24 8 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ball_radius_for_given_hole_l1943_194393


namespace NUMINAMATH_CALUDE_debelyn_gave_two_dolls_l1943_194374

/-- Represents the number of dolls each person has --/
structure DollCount where
  debelyn_initial : ℕ
  christel_initial : ℕ
  christel_to_andrena : ℕ
  debelyn_to_andrena : ℕ

/-- The conditions of the problem --/
def problem_conditions (d : DollCount) : Prop :=
  d.debelyn_initial = 20 ∧
  d.christel_initial = 24 ∧
  d.christel_to_andrena = 5 ∧
  d.debelyn_initial - d.debelyn_to_andrena + 3 = d.christel_initial - d.christel_to_andrena + 2

theorem debelyn_gave_two_dolls (d : DollCount) 
  (h : problem_conditions d) : d.debelyn_to_andrena = 2 := by
  sorry

#check debelyn_gave_two_dolls

end NUMINAMATH_CALUDE_debelyn_gave_two_dolls_l1943_194374


namespace NUMINAMATH_CALUDE_hiker_speed_l1943_194322

theorem hiker_speed (supplies_per_mile : Real) (first_pack : Real) (resupply_ratio : Real)
  (hours_per_day : Real) (num_days : Real) :
  supplies_per_mile = 0.5 →
  first_pack = 40 →
  resupply_ratio = 0.25 →
  hours_per_day = 8 →
  num_days = 5 →
  (first_pack + first_pack * resupply_ratio) / supplies_per_mile / (hours_per_day * num_days) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_hiker_speed_l1943_194322


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1943_194382

theorem cubic_roots_sum_cubes (u v w : ℝ) : 
  (5 * u^3 + 500 * u + 1005 = 0) →
  (5 * v^3 + 500 * v + 1005 = 0) →
  (5 * w^3 + 500 * w + 1005 = 0) →
  (u + v)^3 + (v + w)^3 + (w + u)^3 = 603 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1943_194382


namespace NUMINAMATH_CALUDE_four_digit_square_palindromes_l1943_194372

theorem four_digit_square_palindromes :
  (∃! (s : Finset Nat), 
    ∀ n, n ∈ s ↔ 
      32 ≤ n ∧ n ≤ 99 ∧ 
      1000 ≤ n^2 ∧ n^2 ≤ 9999 ∧ 
      (∃ a b : Nat, n^2 = a * 1000 + b * 100 + b * 10 + a)) ∧ 
  (∃ s : Finset Nat, 
    (∀ n, n ∈ s ↔ 
      32 ≤ n ∧ n ≤ 99 ∧ 
      1000 ≤ n^2 ∧ n^2 ≤ 9999 ∧ 
      (∃ a b : Nat, n^2 = a * 1000 + b * 100 + b * 10 + a)) ∧ 
    s.card = 2) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_square_palindromes_l1943_194372


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1943_194354

/-- Given a cubic equation x√x - 9x + 8√x - 2 = 0 with all roots real and positive,
    the sum of the squares of its roots is 65. -/
theorem sum_of_squares_of_roots : ∃ (r s t : ℝ), 
  (∀ x : ℝ, x > 0 → (x * Real.sqrt x - 9*x + 8*Real.sqrt x - 2 = 0) ↔ (x = r ∨ x = s ∨ x = t)) →
  r > 0 ∧ s > 0 ∧ t > 0 →
  r^2 + s^2 + t^2 = 65 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1943_194354


namespace NUMINAMATH_CALUDE_total_owed_is_790_l1943_194395

/-- Calculates the total amount owed for three overdue bills -/
def total_amount_owed (bill1_principal : ℝ) (bill1_interest_rate : ℝ) (bill1_months : ℕ)
                      (bill2_principal : ℝ) (bill2_late_fee : ℝ) (bill2_months : ℕ)
                      (bill3_first_month_fee : ℝ) (bill3_months : ℕ) : ℝ :=
  let bill1_total := bill1_principal * (1 + bill1_interest_rate * bill1_months)
  let bill2_total := bill2_principal + bill2_late_fee * bill2_months
  let bill3_total := bill3_first_month_fee * (1 + (bill3_months - 1) * 2)
  bill1_total + bill2_total + bill3_total

/-- Theorem stating the total amount owed is $790 given the specific bill conditions -/
theorem total_owed_is_790 :
  total_amount_owed 200 0.1 2 130 50 6 40 2 = 790 := by
  sorry

end NUMINAMATH_CALUDE_total_owed_is_790_l1943_194395


namespace NUMINAMATH_CALUDE_max_abs_z_quadratic_equation_l1943_194376

open Complex

theorem max_abs_z_quadratic_equation (a b c z : ℂ) 
  (h1 : abs a = 1) (h2 : abs b = 1) (h3 : abs c = 1)
  (h4 : arg c = arg a + arg b)
  (h5 : a * z^2 + b * z + c = 0) :
  abs z ≤ (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_quadratic_equation_l1943_194376


namespace NUMINAMATH_CALUDE_smallest_class_size_l1943_194351

theorem smallest_class_size (n : ℕ) (scores : Fin n → ℕ) : 
  (∀ i, scores i ≥ 50) →   -- Each student scored at least 50
  (∃ s : Finset (Fin n), s.card = 4 ∧ ∀ i ∈ s, scores i = 80) →   -- Four students achieved the maximum score
  (Finset.sum Finset.univ scores) / n = 65 →   -- The average score was 65
  n ≥ 8 ∧ 
  (∃ scores_8 : Fin 8 → ℕ, 
    (∀ i, scores_8 i ≥ 50) ∧ 
    (∃ s : Finset (Fin 8), s.card = 4 ∧ ∀ i ∈ s, scores_8 i = 80) ∧ 
    (Finset.sum Finset.univ scores_8) / 8 = 65) :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_l1943_194351


namespace NUMINAMATH_CALUDE_complex_average_equals_three_halves_l1943_194388

/-- Average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- Average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- The main theorem to prove -/
theorem complex_average_equals_three_halves :
  avg3 (avg3 (avg2 2 2) 3 1) (avg2 1 2) 1 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_average_equals_three_halves_l1943_194388


namespace NUMINAMATH_CALUDE_dividend_division_theorem_l1943_194383

theorem dividend_division_theorem : ∃ (q r : ℕ), 
  220025 = (555 + 445) * q + r ∧ 
  r < (555 + 445) ∧ 
  r = 25 ∧ 
  q = 2 * (555 - 445) := by
  sorry

end NUMINAMATH_CALUDE_dividend_division_theorem_l1943_194383


namespace NUMINAMATH_CALUDE_sin_cos_inequality_l1943_194331

theorem sin_cos_inequality (x : ℝ) : 
  2 - Real.sqrt 2 ≤ Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2 
  ∧ Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2 ≤ 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_inequality_l1943_194331


namespace NUMINAMATH_CALUDE_car_value_after_depreciation_l1943_194391

def initial_value : ℝ := 10000

def depreciation_rates : List ℝ := [0.20, 0.15, 0.10, 0.08, 0.05]

def calculate_value (initial : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (fun acc rate => acc * (1 - rate)) initial

theorem car_value_after_depreciation :
  calculate_value initial_value depreciation_rates = 5348.88 := by
  sorry

end NUMINAMATH_CALUDE_car_value_after_depreciation_l1943_194391


namespace NUMINAMATH_CALUDE_widget_price_reduction_l1943_194320

theorem widget_price_reduction (total_money : ℝ) (original_quantity : ℕ) (reduced_quantity : ℕ) :
  total_money = 27.60 ∧ original_quantity = 6 ∧ reduced_quantity = 8 →
  (total_money / original_quantity) - (total_money / reduced_quantity) = 1.15 := by
  sorry

end NUMINAMATH_CALUDE_widget_price_reduction_l1943_194320


namespace NUMINAMATH_CALUDE_charcoal_drawings_count_l1943_194363

theorem charcoal_drawings_count (total : ℕ) (colored_pencil : ℕ) (blending_marker : ℕ) 
  (h1 : total = 25)
  (h2 : colored_pencil = 14)
  (h3 : blending_marker = 7) :
  total - (colored_pencil + blending_marker) = 4 := by
  sorry

end NUMINAMATH_CALUDE_charcoal_drawings_count_l1943_194363


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l1943_194344

-- Define the siblings
inductive Sibling : Type
| Alex : Sibling
| Beth : Sibling
| Cyril : Sibling
| Daria : Sibling
| Ed : Sibling

-- Define a function to represent the fraction of pizza eaten by each sibling
def pizza_fraction (s : Sibling) : ℚ :=
  match s with
  | Sibling.Alex => 1/6
  | Sibling.Beth => 1/4
  | Sibling.Cyril => 1/3
  | Sibling.Daria => 1/8
  | Sibling.Ed => 1 - (1/6 + 1/4 + 1/3 + 1/8)

-- Define the theorem
theorem pizza_consumption_order :
  ∃ (l : List Sibling),
    l = [Sibling.Cyril, Sibling.Beth, Sibling.Alex, Sibling.Daria, Sibling.Ed] ∧
    ∀ (i j : Nat), i < j → j < l.length →
      pizza_fraction (l.get ⟨i, by sorry⟩) ≥ pizza_fraction (l.get ⟨j, by sorry⟩) :=
by sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l1943_194344


namespace NUMINAMATH_CALUDE_divisors_of_10n_l1943_194328

/-- Given a natural number n where 100n^2 has exactly 55 different natural divisors,
    prove that 10n has exactly 18 natural divisors. -/
theorem divisors_of_10n (n : ℕ) (h : (Nat.divisors (100 * n^2)).card = 55) :
  (Nat.divisors (10 * n)).card = 18 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_10n_l1943_194328


namespace NUMINAMATH_CALUDE_sams_adventure_books_l1943_194385

/-- The number of adventure books Sam bought at the school's book fair -/
def adventure_books : ℕ := sorry

/-- The number of mystery books Sam bought -/
def mystery_books : ℕ := 17

/-- The total number of books Sam bought -/
def total_books : ℕ := 30

theorem sams_adventure_books :
  adventure_books = total_books - mystery_books ∧ adventure_books = 13 := by sorry

end NUMINAMATH_CALUDE_sams_adventure_books_l1943_194385


namespace NUMINAMATH_CALUDE_disjoint_sets_cardinality_relation_l1943_194397

theorem disjoint_sets_cardinality_relation 
  (a b : ℕ+) 
  (A B : Finset ℤ) 
  (h_disjoint : Disjoint A B)
  (h_membership : ∀ i ∈ A ∪ B, (i + a) ∈ A ∨ (i - b) ∈ B) :
  a * A.card = b * B.card := by
  sorry

end NUMINAMATH_CALUDE_disjoint_sets_cardinality_relation_l1943_194397


namespace NUMINAMATH_CALUDE_min_distance_parabola_to_line_l1943_194392

/-- The minimum distance from a point on the parabola y^2 = 4x to the line 3x + 4y + 15 = 0 -/
theorem min_distance_parabola_to_line :
  let parabola := {P : ℝ × ℝ | P.2^2 = 4 * P.1}
  let line := {P : ℝ × ℝ | 3 * P.1 + 4 * P.2 + 15 = 0}
  (∃ (d : ℝ), d > 0 ∧
    (∀ P ∈ parabola, ∀ Q ∈ line, d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) ∧
    (∃ P ∈ parabola, ∃ Q ∈ line, d = Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))) ∧
  (∀ d' : ℝ, (∀ P ∈ parabola, ∀ Q ∈ line, d' ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) →
    d' ≥ 29/15) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_parabola_to_line_l1943_194392


namespace NUMINAMATH_CALUDE_vegan_soy_free_dishes_l1943_194362

theorem vegan_soy_free_dishes 
  (total_dishes : ℕ) 
  (vegan_ratio : ℚ) 
  (soy_ratio : ℚ) 
  (h1 : vegan_ratio = 1 / 3) 
  (h2 : soy_ratio = 5 / 6) : 
  ↑total_dishes * vegan_ratio * (1 - soy_ratio) = ↑total_dishes * (1 / 18) := by
sorry

end NUMINAMATH_CALUDE_vegan_soy_free_dishes_l1943_194362


namespace NUMINAMATH_CALUDE_triple_application_equals_six_l1943_194350

/-- The function f defined as f(p) = 2p - 20 --/
def f (p : ℝ) : ℝ := 2 * p - 20

/-- Theorem stating that there exists a unique real number p such that f(f(f(p))) = 6 --/
theorem triple_application_equals_six :
  ∃! p : ℝ, f (f (f p)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_application_equals_six_l1943_194350


namespace NUMINAMATH_CALUDE_sequence_ratio_l1943_194304

/-- Given two sequences, one arithmetic and one geometric, prove that (a₂ - a₁) / b₂ = 1/2 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℚ) : 
  ((-1 : ℚ) - a₁ = a₁ - a₂) ∧ 
  (a₁ - a₂ = a₂ - (-4)) ∧ 
  ((-1 : ℚ) * b₁ = b₁ * b₂) ∧ 
  (b₁ * b₂ = b₂ * b₃) ∧ 
  (b₂ * b₃ = b₃ * (-4)) → 
  (a₂ - a₁) / b₂ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_ratio_l1943_194304


namespace NUMINAMATH_CALUDE_equation_solver_l1943_194345

theorem equation_solver (m n : ℕ) : 
  ((1^(m+1))/(5^(m+1))) * ((1^n)/(4^n)) = 1/(2*(10^35)) ∧ m = 34 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solver_l1943_194345


namespace NUMINAMATH_CALUDE_min_translation_for_symmetry_l1943_194375

theorem min_translation_for_symmetry :
  let f (x m : ℝ) := Real.sin (2 * (x - m) - π / 6)
  ∀ m : ℝ, m > 0 →
    (∀ x : ℝ, f x m = f (-x) m) →
    m ≥ π / 6 :=
by sorry

end NUMINAMATH_CALUDE_min_translation_for_symmetry_l1943_194375


namespace NUMINAMATH_CALUDE_shopping_money_l1943_194377

theorem shopping_money (initial_amount : ℝ) : 
  (0.7 * initial_amount = 350) → initial_amount = 500 := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_l1943_194377


namespace NUMINAMATH_CALUDE_expression_value_at_sqrt3_over_2_l1943_194305

theorem expression_value_at_sqrt3_over_2 :
  let x : ℝ := Real.sqrt 3 / 2
  (1 + x) / (1 + Real.sqrt (1 + x)) + (1 - x) / (1 - Real.sqrt (1 - x)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_sqrt3_over_2_l1943_194305


namespace NUMINAMATH_CALUDE_bob_weighs_165_l1943_194398

/-- Bob's weight given the conditions -/
def bobs_weight (jim_weight bob_weight : ℕ) : Prop :=
  (jim_weight + bob_weight = 220) ∧ 
  (bob_weight - jim_weight = 2 * jim_weight) ∧
  (bob_weight = 165)

/-- Theorem stating that Bob's weight is 165 pounds given the conditions -/
theorem bob_weighs_165 :
  ∃ (jim_weight bob_weight : ℕ), bobs_weight jim_weight bob_weight :=
by
  sorry

end NUMINAMATH_CALUDE_bob_weighs_165_l1943_194398


namespace NUMINAMATH_CALUDE_remaining_distance_l1943_194347

def total_distance : ℝ := 300
def speed : ℝ := 60
def time : ℝ := 2

theorem remaining_distance : total_distance - speed * time = 180 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l1943_194347


namespace NUMINAMATH_CALUDE_integers_between_cubes_l1943_194340

theorem integers_between_cubes : ∃ n : ℕ, n = (⌊(10.1 : ℝ)^3⌋ - ⌈(9.8 : ℝ)^3⌉ + 1) ∧ n = 89 := by sorry

end NUMINAMATH_CALUDE_integers_between_cubes_l1943_194340


namespace NUMINAMATH_CALUDE_largest_root_of_equation_l1943_194311

theorem largest_root_of_equation (x : ℝ) :
  (x - 37)^2 - 169 = 0 → x ≤ 50 ∧ ∃ y, (y - 37)^2 - 169 = 0 ∧ y = 50 := by
  sorry

end NUMINAMATH_CALUDE_largest_root_of_equation_l1943_194311


namespace NUMINAMATH_CALUDE_additional_discount_percentage_l1943_194399

/-- Proves that the additional discount percentage for mothers with 3 or more children is 4% -/
theorem additional_discount_percentage : 
  let original_price : ℚ := 125
  let mothers_day_discount : ℚ := 10 / 100
  let final_price : ℚ := 108
  let price_after_initial_discount : ℚ := original_price * (1 - mothers_day_discount)
  let additional_discount_amount : ℚ := price_after_initial_discount - final_price
  let additional_discount_percentage : ℚ := additional_discount_amount / price_after_initial_discount * 100
  additional_discount_percentage = 4 := by sorry

end NUMINAMATH_CALUDE_additional_discount_percentage_l1943_194399


namespace NUMINAMATH_CALUDE_garden_width_l1943_194378

theorem garden_width (playground_side : ℕ) (garden_length : ℕ) (total_fencing : ℕ) :
  playground_side = 27 →
  garden_length = 12 →
  total_fencing = 150 →
  4 * playground_side + 2 * garden_length + 2 * (total_fencing - 4 * playground_side - 2 * garden_length) / 2 = 150 →
  (total_fencing - 4 * playground_side - 2 * garden_length) / 2 = 9 := by
  sorry

#check garden_width

end NUMINAMATH_CALUDE_garden_width_l1943_194378


namespace NUMINAMATH_CALUDE_sum_of_squares_l1943_194307

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 88) : x^2 + y^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1943_194307


namespace NUMINAMATH_CALUDE_first_worker_time_l1943_194390

/-- Given three workers who make parts with the following conditions:
    1. They need to make 80 identical parts in total.
    2. Together, they produce 20 parts per hour.
    3. The first worker makes 20 parts, taking more than 3 hours.
    4. The remaining work is completed by the second and third workers together.
    5. The total time taken to complete the work is 8 hours.
    
    This theorem proves that it would take the first worker 16 hours to make all 80 parts by himself. -/
theorem first_worker_time (x y z : ℝ) (h1 : x + y + z = 20) 
  (h2 : 20 / x > 3) (h3 : 20 / x + 60 / (y + z) = 8) : 80 / x = 16 := by
  sorry

end NUMINAMATH_CALUDE_first_worker_time_l1943_194390


namespace NUMINAMATH_CALUDE_ratio_calculation_l1943_194339

theorem ratio_calculation (A B C : ℚ) (h : A = 2 * B ∧ C = 4 * B) :
  (3 * A + 2 * B) / (4 * C - A) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l1943_194339


namespace NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l1943_194310

/-- Given the conversion rates between knicks, knacks, and knocks, 
    prove that 36 knocks are equal to 40 knicks. -/
theorem knicks_knacks_knocks_conversion :
  (∀ (knicks knacks knocks : ℚ),
    5 * knicks = 3 * knacks →
    4 * knacks = 6 * knocks →
    36 * knocks = 40 * knicks) :=
by sorry

end NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l1943_194310


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1943_194359

theorem triangle_angle_measure (A B C : ℝ) (h1 : A + C = 2 * B) (h2 : C - A = 80) 
  (h3 : A + B + C = 180) : C = 100 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1943_194359


namespace NUMINAMATH_CALUDE_two_layer_triangle_structure_l1943_194327

/-- Calculates the number of small triangles in a layer given the number of triangles in the base row -/
def trianglesInLayer (baseTriangles : ℕ) : ℕ :=
  (baseTriangles * (baseTriangles + 1)) / 2

/-- Calculates the total number of toothpicks required for the two-layer structure -/
def totalToothpicks (lowerBaseTriangles upperBaseTriangles : ℕ) : ℕ :=
  let lowerTriangles := trianglesInLayer lowerBaseTriangles
  let upperTriangles := trianglesInLayer upperBaseTriangles
  let totalTriangles := lowerTriangles + upperTriangles
  let totalEdges := 3 * totalTriangles
  let boundaryEdges := 3 * lowerBaseTriangles + 3 * upperBaseTriangles - 3
  (totalEdges - boundaryEdges) / 2 + boundaryEdges

/-- The main theorem stating that the structure with 100 triangles in the lower base
    and 99 in the upper base requires 15596 toothpicks -/
theorem two_layer_triangle_structure :
  totalToothpicks 100 99 = 15596 := by
  sorry


end NUMINAMATH_CALUDE_two_layer_triangle_structure_l1943_194327


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1943_194361

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2*Complex.I → z = -1 + (3/2)*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1943_194361


namespace NUMINAMATH_CALUDE_used_car_seller_problem_l1943_194369

theorem used_car_seller_problem (num_clients : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) : 
  num_clients = 18 → cars_per_client = 3 → selections_per_car = 3 →
  num_clients * cars_per_client / selections_per_car = 18 := by
sorry

end NUMINAMATH_CALUDE_used_car_seller_problem_l1943_194369


namespace NUMINAMATH_CALUDE_equation_solutions_l1943_194371

theorem equation_solutions :
  (∃ x : ℝ, 4 * x + x = 19.5 ∧ x = 3.9) ∧
  (∃ x : ℝ, 26.4 - 3 * x = 14.4 ∧ x = 4) ∧
  (∃ x : ℝ, 2 * x - 0.5 * 2 = 0.8 ∧ x = 0.9) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1943_194371


namespace NUMINAMATH_CALUDE_original_red_marbles_l1943_194349

-- Define the initial number of red and green marbles
variable (r g : ℚ)

-- Define the conditions
def initial_ratio : Prop := r / g = 3 / 2
def new_ratio : Prop := (r - 15) / (g + 25) = 2 / 5

-- State the theorem
theorem original_red_marbles 
  (h1 : initial_ratio r g) 
  (h2 : new_ratio r g) : 
  r = 375 / 11 := by
  sorry

end NUMINAMATH_CALUDE_original_red_marbles_l1943_194349


namespace NUMINAMATH_CALUDE_jenna_one_way_distance_l1943_194338

/-- Calculates the one-way distance for a truck driver's round trip. -/
def one_way_distance (pay_rate : ℚ) (total_payment : ℚ) : ℚ :=
  (total_payment / pay_rate) / 2

/-- Proves that given a pay rate of $0.40 per mile and a total payment of $320 for a round trip, the one-way distance is 400 miles. -/
theorem jenna_one_way_distance :
  one_way_distance (40 / 100) 320 = 400 := by
  sorry

end NUMINAMATH_CALUDE_jenna_one_way_distance_l1943_194338


namespace NUMINAMATH_CALUDE_integer_sum_problem_l1943_194315

theorem integer_sum_problem (x y : ℤ) : 
  x > 0 → y > 0 → x - y = 8 → x * y = 120 → x + y = 4 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l1943_194315


namespace NUMINAMATH_CALUDE_grant_baseball_gear_sale_total_l1943_194348

def baseball_cards_price : ℝ := 25
def baseball_bat_price : ℝ := 10
def baseball_glove_original_price : ℝ := 30
def baseball_glove_discount : ℝ := 0.20
def baseball_cleats_original_price : ℝ := 10
def usd_to_eur_rate : ℝ := 0.85
def baseball_cleats_discount : ℝ := 0.15

theorem grant_baseball_gear_sale_total :
  let baseball_glove_sale_price := baseball_glove_original_price * (1 - baseball_glove_discount)
  let cleats_eur_price := baseball_cleats_original_price * usd_to_eur_rate
  let cleats_discounted_price := baseball_cleats_original_price * (1 - baseball_cleats_discount)
  baseball_cards_price + baseball_bat_price + baseball_glove_sale_price + cleats_eur_price + cleats_discounted_price = 76 := by
  sorry

end NUMINAMATH_CALUDE_grant_baseball_gear_sale_total_l1943_194348


namespace NUMINAMATH_CALUDE_inequality_proof_l1943_194314

theorem inequality_proof (a b c d : ℝ) (h1 : a * b = 1) (h2 : a * c + b * d = 2) : c * d ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1943_194314


namespace NUMINAMATH_CALUDE_grandmother_truth_lies_consistent_solution_l1943_194360

-- Define the type for grandmothers
inductive Grandmother
| Emilia
| Leonie
| Gabrielle

-- Define a function to represent the number of grandchildren for each grandmother
def grandchildren : Grandmother → ℕ
| Grandmother.Emilia => 8
| Grandmother.Leonie => 7
| Grandmother.Gabrielle => 10

-- Define a function to represent the statements made by each grandmother
def statements : Grandmother → List (Grandmother → Bool)
| Grandmother.Emilia => [
    fun g => grandchildren g = 7,
    fun g => grandchildren g = 8,
    fun g => grandchildren Grandmother.Gabrielle = 10
  ]
| Grandmother.Leonie => [
    fun g => grandchildren Grandmother.Emilia = 8,
    fun g => grandchildren g = 6,
    fun g => grandchildren g = 7
  ]
| Grandmother.Gabrielle => [
    fun g => grandchildren Grandmother.Emilia = 7,
    fun g => grandchildren g = 9,
    fun g => grandchildren g = 10
  ]

-- Define a function to count true statements for each grandmother
def countTrueStatements (g : Grandmother) : ℕ :=
  (statements g).filter (fun s => s g) |>.length

-- Theorem stating that each grandmother tells the truth twice and lies once
theorem grandmother_truth_lies :
  ∀ g : Grandmother, countTrueStatements g = 2 :=
sorry

-- Main theorem proving the consistency of the solution
theorem consistent_solution :
  (grandchildren Grandmother.Emilia = 8) ∧
  (grandchildren Grandmother.Leonie = 7) ∧
  (grandchildren Grandmother.Gabrielle = 10) :=
sorry

end NUMINAMATH_CALUDE_grandmother_truth_lies_consistent_solution_l1943_194360


namespace NUMINAMATH_CALUDE_cos_105_degrees_l1943_194306

theorem cos_105_degrees :
  Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l1943_194306


namespace NUMINAMATH_CALUDE_min_clerks_needed_is_84_l1943_194389

/-- The number of forms a clerk can process per hour -/
def forms_per_hour : ℕ := 25

/-- The time in minutes to process a type A form -/
def time_per_type_a : ℕ := 3

/-- The time in minutes to process a type B form -/
def time_per_type_b : ℕ := 4

/-- The number of type A forms to process -/
def num_type_a : ℕ := 3000

/-- The number of type B forms to process -/
def num_type_b : ℕ := 4000

/-- The number of hours in a workday -/
def hours_per_day : ℕ := 5

/-- The function to calculate the minimum number of clerks needed -/
def min_clerks_needed : ℕ :=
  let total_minutes := num_type_a * time_per_type_a + num_type_b * time_per_type_b
  let total_hours := (total_minutes + 59) / 60  -- Ceiling division
  (total_hours + hours_per_day - 1) / hours_per_day  -- Ceiling division

theorem min_clerks_needed_is_84 : min_clerks_needed = 84 := by
  sorry

end NUMINAMATH_CALUDE_min_clerks_needed_is_84_l1943_194389


namespace NUMINAMATH_CALUDE_basketball_not_table_tennis_l1943_194302

theorem basketball_not_table_tennis 
  (total : ℕ) 
  (basketball : ℕ) 
  (table_tennis : ℕ) 
  (neither : ℕ) 
  (h1 : total = 30) 
  (h2 : basketball = 15) 
  (h3 : table_tennis = 10) 
  (h4 : neither = 8) :
  ∃ (both : ℕ), 
    basketball - both = 12 ∧ 
    total = (basketball - both) + (table_tennis - both) + both + neither :=
by sorry

end NUMINAMATH_CALUDE_basketball_not_table_tennis_l1943_194302


namespace NUMINAMATH_CALUDE_mod_inverse_sum_equals_26_l1943_194319

theorem mod_inverse_sum_equals_26 : ∃ x y : ℤ, 
  (0 ≤ x ∧ x < 31) ∧ 
  (0 ≤ y ∧ y < 31) ∧ 
  (5 * x ≡ 1 [ZMOD 31]) ∧ 
  (5 * 5 * y ≡ 1 [ZMOD 31]) ∧ 
  ((x + y) % 31 = 26) := by
  sorry

end NUMINAMATH_CALUDE_mod_inverse_sum_equals_26_l1943_194319


namespace NUMINAMATH_CALUDE_line_through_points_l1943_194334

/-- 
A line in a rectangular coordinate system is defined by the equation x = 5y + 5.
This line passes through two points (m, n) and (m + 2, n + p).
The theorem proves that under these conditions, p must equal 2/5.
-/
theorem line_through_points (m n : ℝ) : 
  (m = 5 * n + 5) → 
  (m + 2 = 5 * (n + p) + 5) → 
  p = 2/5 :=
by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1943_194334


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l1943_194303

theorem perfect_square_divisibility (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) :
  ∃ k : ℕ, a = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l1943_194303


namespace NUMINAMATH_CALUDE_nine_valid_sets_l1943_194343

def count_valid_sets : ℕ → Prop :=
  λ n => ∃ S : Finset (ℕ × ℕ × ℕ),
    (∀ (a b c : ℕ), (a, b, c) ∈ S ↔ 
      (Nat.gcd a b = 4 ∧ 
       Nat.lcm a c = 100 ∧ 
       Nat.lcm b c = 100 ∧ 
       a ≤ b)) ∧
    S.card = n

theorem nine_valid_sets : count_valid_sets 9 := by sorry

end NUMINAMATH_CALUDE_nine_valid_sets_l1943_194343


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1943_194364

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^6 + 2*x^5 - 3*x^4 - 4*x^3 + 5*x^2 - 6*x + 7 = 
  (x^2 - 1) * (x - 2) * q + (-3*x^2 - 8*x + 13) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1943_194364


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_conditions_l1943_194379

theorem smallest_positive_integer_satisfying_conditions (a : ℕ) :
  (∀ b : ℕ, b > 0 ∧ b % 3 = 1 ∧ 5 ∣ b → a ≤ b) ∧
  a > 0 ∧ a % 3 = 1 ∧ 5 ∣ a →
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_conditions_l1943_194379


namespace NUMINAMATH_CALUDE_average_age_of_friends_l1943_194396

theorem average_age_of_friends (age1 age2 age3 : ℕ) : 
  age1 = 40 →
  age2 = 30 →
  age3 = age1 + 10 →
  (age1 + age2 + age3) / 3 = 40 := by
sorry

end NUMINAMATH_CALUDE_average_age_of_friends_l1943_194396


namespace NUMINAMATH_CALUDE_train_length_l1943_194365

/-- Given a train that crosses a platform in a certain time and a signal pole in another time,
    this theorem calculates the length of the train. -/
theorem train_length
  (platform_length : ℝ)
  (platform_time : ℝ)
  (pole_time : ℝ)
  (h1 : platform_length = 400)
  (h2 : platform_time = 42)
  (h3 : pole_time = 18) :
  ∃ (train_length : ℝ),
    train_length = 300 ∧
    train_length * (1 / pole_time) * platform_time = train_length + platform_length :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1943_194365


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l1943_194301

/-- Given two digits A and B in base d > 6, if ̅AB_d + ̅AA_d = 162_d, then A_d - B_d = 3_d -/
theorem digit_difference_in_base_d (d A B : ℕ) (h1 : d > 6) 
  (h2 : A < d) (h3 : B < d) 
  (h4 : A * d + B + A * d + A = 1 * d^2 + 6 * d + 2) : 
  A - B = 3 := by
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l1943_194301


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1943_194352

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors a and b
variable (a b : V)

-- State the theorem
theorem angle_between_vectors
  (h1 : ‖a‖ = 2)
  (h2 : ‖b‖ = (1/3) * ‖a‖)
  (h3 : ‖a - (1/2) • b‖ = Real.sqrt 43 / 3) :
  Real.arccos (inner a b / (‖a‖ * ‖b‖)) = (2 * Real.pi) / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1943_194352


namespace NUMINAMATH_CALUDE_optimal_loquat_variety_l1943_194386

/-- Represents a variety of loquat trees -/
structure LoquatVariety where
  name : String
  average_yield : ℝ
  variance : ℝ

/-- Determines if one variety is better than another based on yield and stability -/
def is_better (v1 v2 : LoquatVariety) : Prop :=
  (v1.average_yield > v2.average_yield) ∨ 
  (v1.average_yield = v2.average_yield ∧ v1.variance < v2.variance)

/-- Determines if a variety is the best among a list of varieties -/
def is_best (v : LoquatVariety) (vs : List LoquatVariety) : Prop :=
  ∀ v' ∈ vs, v ≠ v' → is_better v v'

theorem optimal_loquat_variety (A B C : LoquatVariety)
  (hA : A = { name := "A", average_yield := 42, variance := 1.8 })
  (hB : B = { name := "B", average_yield := 45, variance := 23 })
  (hC : C = { name := "C", average_yield := 45, variance := 1.8 }) :
  is_best C [A, B, C] := by
  sorry

#check optimal_loquat_variety

end NUMINAMATH_CALUDE_optimal_loquat_variety_l1943_194386


namespace NUMINAMATH_CALUDE_integer_floor_equation_l1943_194313

theorem integer_floor_equation (m n : ℕ+) :
  (⌊(m : ℝ)^2 / n⌋ + ⌊(n : ℝ)^2 / m⌋ = ⌊(m : ℝ) / n + (n : ℝ) / m⌋ + m * n) ↔
  (∃ k : ℕ+, (m = k ∧ n = k^2 + 1) ∨ (m = k^2 + 1 ∧ n = k)) :=
sorry

end NUMINAMATH_CALUDE_integer_floor_equation_l1943_194313


namespace NUMINAMATH_CALUDE_equation_condition_l1943_194312

theorem equation_condition (x y z : ℕ) 
  (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  (10 * x + y) * (10 * x + z) = 100 * x^2 + 110 * x + y * z ↔ y + z = 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_condition_l1943_194312


namespace NUMINAMATH_CALUDE_jasmine_purchase_cost_l1943_194341

/-- The cost calculation for Jasmine's purchase of coffee beans and milk. -/
theorem jasmine_purchase_cost :
  let coffee_beans_pounds : ℕ := 4
  let milk_gallons : ℕ := 2
  let coffee_bean_price_per_pound : ℚ := 5/2
  let milk_price_per_gallon : ℚ := 7/2
  let total_cost : ℚ := coffee_beans_pounds * coffee_bean_price_per_pound + milk_gallons * milk_price_per_gallon
  total_cost = 17 := by sorry

end NUMINAMATH_CALUDE_jasmine_purchase_cost_l1943_194341


namespace NUMINAMATH_CALUDE_inequality_proof_l1943_194342

theorem inequality_proof (x y : ℝ) (n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) (h4 : n ≥ 2) :
  x^n / (x + y^3) + y^n / (x^3 + y) ≥ 2^(4-n) / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1943_194342


namespace NUMINAMATH_CALUDE_intersection_points_polar_equations_l1943_194355

/-- The number of intersection points between r = 3 cos θ and r = 6 sin θ -/
theorem intersection_points_polar_equations : ∃ (n : ℕ), n = 2 ∧
  ∀ (x y : ℝ),
    ((x - 3/2)^2 + y^2 = 9/4 ∨ x^2 + (y - 3)^2 = 9) →
    (∃ (θ : ℝ), 
      (x = 3 * Real.cos θ * Real.cos θ ∧ y = 3 * Real.sin θ * Real.cos θ) ∨
      (x = 6 * Real.sin θ * Real.cos θ ∧ y = 6 * Real.sin θ * Real.sin θ)) :=
by sorry


end NUMINAMATH_CALUDE_intersection_points_polar_equations_l1943_194355


namespace NUMINAMATH_CALUDE_parabola_equation_l1943_194367

-- Define the parabola and its properties
structure Parabola where
  focus : ℝ × ℝ
  vertex : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the theorem
theorem parabola_equation (p : Parabola) :
  p.vertex = (0, 0) →
  p.focus.1 > 0 →
  p.focus.2 = 0 →
  (p.A.1 - p.focus.1, p.A.2 - p.focus.2) +
  (p.B.1 - p.focus.1, p.B.2 - p.focus.2) +
  (p.C.1 - p.focus.1, p.C.2 - p.focus.2) = (0, 0) →
  Real.sqrt ((p.A.1 - p.focus.1)^2 + (p.A.2 - p.focus.2)^2) +
  Real.sqrt ((p.B.1 - p.focus.1)^2 + (p.B.2 - p.focus.2)^2) +
  Real.sqrt ((p.C.1 - p.focus.1)^2 + (p.C.2 - p.focus.2)^2) = 6 →
  ∀ (x y : ℝ), (x, y) ∈ {(x, y) | y^2 = 8*x} ↔
    Real.sqrt ((x - p.focus.1)^2 + y^2) = x + p.focus.1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1943_194367


namespace NUMINAMATH_CALUDE_bayonet_on_third_draw_l1943_194332

/-- Represents the number of screw base bulbs initially in the box -/
def screw_bulbs : ℕ := 3

/-- Represents the number of bayonet base bulbs initially in the box -/
def bayonet_bulbs : ℕ := 7

/-- Represents the total number of bulbs initially in the box -/
def total_bulbs : ℕ := screw_bulbs + bayonet_bulbs

/-- The probability of selecting a bayonet base bulb on the third draw,
    given that the first two draws were screw base bulbs -/
def prob_bayonet_third : ℚ := 7 / 120

theorem bayonet_on_third_draw :
  (screw_bulbs / total_bulbs) *
  ((screw_bulbs - 1) / (total_bulbs - 1)) *
  (bayonet_bulbs / (total_bulbs - 2)) = prob_bayonet_third := by
  sorry

end NUMINAMATH_CALUDE_bayonet_on_third_draw_l1943_194332


namespace NUMINAMATH_CALUDE_flyer_multiple_l1943_194324

theorem flyer_multiple (maisie_flyers donna_flyers : ℕ) (h1 : maisie_flyers = 33) (h2 : donna_flyers = 71) :
  ∃ x : ℕ, donna_flyers = 5 + x * maisie_flyers ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_flyer_multiple_l1943_194324


namespace NUMINAMATH_CALUDE_sphere_cross_section_distance_l1943_194336

theorem sphere_cross_section_distance
  (V : ℝ) (A : ℝ) (d : ℝ)
  (hV : V = 4 * Real.sqrt 3 * Real.pi)
  (hA : A = Real.pi) :
  d = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sphere_cross_section_distance_l1943_194336


namespace NUMINAMATH_CALUDE_shoes_theorem_l1943_194316

/-- The number of pairs of shoes Ellie, Riley, and Jordan have in total -/
def total_shoes (ellie riley jordan : ℕ) : ℕ := ellie + riley + jordan

/-- The theorem stating the total number of shoes given the conditions -/
theorem shoes_theorem (ellie riley jordan : ℕ) 
  (h1 : ellie = 8)
  (h2 : riley = ellie - 3)
  (h3 : jordan = ((ellie + riley) * 3) / 2) :
  total_shoes ellie riley jordan = 32 := by
  sorry

end NUMINAMATH_CALUDE_shoes_theorem_l1943_194316


namespace NUMINAMATH_CALUDE_blue_card_value_is_five_l1943_194356

/-- The value of a blue card in credits -/
def blue_card_value (total_credits : ℕ) (total_cards : ℕ) (red_card_value : ℕ) (red_cards : ℕ) : ℕ :=
  (total_credits - red_card_value * red_cards) / (total_cards - red_cards)

/-- Theorem stating that the value of a blue card is 5 credits -/
theorem blue_card_value_is_five :
  blue_card_value 84 20 3 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_card_value_is_five_l1943_194356


namespace NUMINAMATH_CALUDE_cube_sum_over_product_is_18_l1943_194394

theorem cube_sum_over_product_is_18 
  (x y z : ℂ) 
  (nonzero_x : x ≠ 0) 
  (nonzero_y : y ≠ 0) 
  (nonzero_z : z ≠ 0) 
  (sum_30 : x + y + z = 30) 
  (squared_diff_sum : (x - y)^2 + (x - z)^2 + (y - z)^2 + x*y*z = 2*x*y*z) : 
  (x^3 + y^3 + z^3) / (x*y*z) = 18 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_is_18_l1943_194394


namespace NUMINAMATH_CALUDE_arithmetic_verification_l1943_194353

theorem arithmetic_verification (A B C M N P : ℝ) : 
  (A - B = C → C + B = A ∧ A - C = B) ∧ 
  (M * N = P → P / N = M ∧ P / M = N) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_verification_l1943_194353


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_y_l1943_194384

theorem max_value_of_x_plus_y (x y : ℝ) 
  (h1 : 5 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 6 * y ≤ 12) : 
  x + y ≤ 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_y_l1943_194384


namespace NUMINAMATH_CALUDE_y_minus_x_value_l1943_194368

theorem y_minus_x_value (x y : ℚ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) : 
  y - x = (15 : ℚ) / 2 := by sorry

end NUMINAMATH_CALUDE_y_minus_x_value_l1943_194368


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l1943_194370

theorem arithmetic_sequence_solution :
  let a₁ : ℚ := 3/4
  let a₂ : ℚ → ℚ := λ x => x - 2
  let a₃ : ℚ → ℚ := λ x => 5*x
  ∀ x : ℚ, (a₂ x - a₁ = a₃ x - a₂ x) → x = -19/12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l1943_194370


namespace NUMINAMATH_CALUDE_curve_C_and_point_Q_existence_l1943_194333

noncomputable section

-- Define the circle O
def circle_O : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the curve C
def curve_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

-- Define the fixed point (0, 1/2)
def fixed_point : ℝ × ℝ := (0, 1/2)

-- Define the point Q
def Q : ℝ × ℝ := (0, 6)

-- State the theorem
theorem curve_C_and_point_Q_existence :
  ∀ (P : ℝ × ℝ),
  (∃ (center : ℝ × ℝ), (center.1 - F.1)^2 + (center.2 - F.2)^2 = (center.1 - P.1)^2 + (center.2 - P.2)^2 ∧
                       ∃ (T : ℝ × ℝ), T ∈ circle_O ∧ (center.1 - T.1)^2 + (center.2 - T.2)^2 = (F.1 - P.1)^2 / 4 + (F.2 - P.2)^2 / 4) →
  P ∈ curve_C ∧
  ∀ (M N : ℝ × ℝ), M ∈ curve_C → N ∈ curve_C →
    (N.2 - M.2) * fixed_point.1 = (N.1 - M.1) * (fixed_point.2 - M.2) + M.1 * (N.2 - M.2) →
    (M.2 - Q.2) / (M.1 - Q.1) + (N.2 - Q.2) / (N.1 - Q.1) = 0 :=
by sorry

end

end NUMINAMATH_CALUDE_curve_C_and_point_Q_existence_l1943_194333


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l1943_194325

/-- Given a box of pens, calculate the probability of selecting two non-defective pens. -/
theorem probability_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (selected_pens : ℕ) 
  (h1 : total_pens = 10) 
  (h2 : defective_pens = 3) 
  (h3 : selected_pens = 2) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 7 / 15 := by
sorry


end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l1943_194325


namespace NUMINAMATH_CALUDE_probability_of_one_in_twenty_rows_l1943_194323

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) := sorry

/-- Counts the number of ones in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Counts the total number of elements in the first n rows of Pascal's Triangle -/
def countElements (n : ℕ) : ℕ := sorry

/-- The probability of randomly selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ :=
  (countOnes n : ℚ) / (countElements n : ℚ)

theorem probability_of_one_in_twenty_rows :
  probabilityOfOne 20 = 13 / 70 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_in_twenty_rows_l1943_194323


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l1943_194300

/-- Given an arithmetic sequence with first term 2 and common difference 3,
    the 20th term of the sequence is 59. -/
theorem arithmetic_sequence_20th_term : 
  ∀ (a : ℕ → ℤ), 
    (a 1 = 2) →  -- First term is 2
    (∀ n, a (n + 1) - a n = 3) →  -- Common difference is 3
    a 20 = 59 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l1943_194300


namespace NUMINAMATH_CALUDE_kimikos_age_l1943_194308

theorem kimikos_age (kayla_age kimiko_age min_driving_age wait_time : ℕ) : 
  kayla_age = kimiko_age / 2 →
  min_driving_age = 18 →
  kayla_age + wait_time = min_driving_age →
  wait_time = 5 →
  kimiko_age = 26 := by
sorry

end NUMINAMATH_CALUDE_kimikos_age_l1943_194308


namespace NUMINAMATH_CALUDE_leah_lost_money_l1943_194335

def total_earned : ℚ := 28
def milkshake_fraction : ℚ := 1/7
def savings_fraction : ℚ := 1/2
def remaining_in_wallet : ℚ := 1

theorem leah_lost_money : 
  let milkshake_cost := total_earned * milkshake_fraction
  let after_milkshake := total_earned - milkshake_cost
  let savings := after_milkshake * savings_fraction
  let in_wallet := after_milkshake - savings
  in_wallet - remaining_in_wallet = 11 := by sorry

end NUMINAMATH_CALUDE_leah_lost_money_l1943_194335


namespace NUMINAMATH_CALUDE_tenth_finger_number_l1943_194366

-- Define the function g based on the graph points
def g : ℕ → ℕ
| 0 => 5
| 1 => 0
| 2 => 4
| 3 => 8
| 4 => 3
| 5 => 7
| 6 => 2
| 7 => 6
| 8 => 1
| 9 => 5
| n => n  -- Default case for numbers not explicitly defined

-- Define a function that applies g n times to an initial value
def apply_g_n_times (n : ℕ) (initial : ℕ) : ℕ :=
  match n with
  | 0 => initial
  | k + 1 => g (apply_g_n_times k initial)

-- Theorem statement
theorem tenth_finger_number : apply_g_n_times 10 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tenth_finger_number_l1943_194366


namespace NUMINAMATH_CALUDE_hyperbola_asymptotic_lines_l1943_194329

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := 3 * x^2 - y^2 = 3

/-- The asymptotic lines equation -/
def asymptotic_lines_eq (x y : ℝ) : Prop := y^2 = 3 * x^2

/-- Theorem: The asymptotic lines of the hyperbola 3x^2 - y^2 = 3 are y = ± √3x -/
theorem hyperbola_asymptotic_lines :
  ∀ x y : ℝ, hyperbola_eq x y → asymptotic_lines_eq x y :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotic_lines_l1943_194329


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l1943_194318

/-- Given an arithmetic sequence {aₙ}, Sₙ represents the sum of its first n terms -/
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := sorry

/-- aₙ is an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := sorry

theorem fifth_term_of_arithmetic_sequence (a : ℕ → ℝ) :
  is_arithmetic_sequence a → S a 9 = 45 → a 5 = 5 := by sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l1943_194318


namespace NUMINAMATH_CALUDE_light_glows_165_times_l1943_194357

/-- Represents the glow pattern of the light in seconds -/
def glowPattern : List Nat := [15, 25, 35, 45]

/-- Calculates the total seconds in the glow pattern -/
def patternDuration : Nat := glowPattern.sum

/-- Converts a time in hours, minutes, seconds to total seconds -/
def timeToSeconds (hours minutes seconds : Nat) : Nat :=
  hours * 3600 + minutes * 60 + seconds

/-- Calculates the duration between two times in seconds -/
def durationBetween (startHours startMinutes startSeconds endHours endMinutes endSeconds : Nat) : Nat :=
  timeToSeconds endHours endMinutes endSeconds - timeToSeconds startHours startMinutes startSeconds

/-- Calculates the number of complete cycles in a given duration -/
def completeCycles (duration : Nat) : Nat :=
  duration / patternDuration

/-- Calculates the remaining seconds after complete cycles -/
def remainingSeconds (duration : Nat) : Nat :=
  duration % patternDuration

/-- Counts the number of glows in the remaining seconds -/
def countRemainingGlows (seconds : Nat) : Nat :=
  glowPattern.foldl (fun count interval => if seconds ≥ interval then count + 1 else count) 0

/-- Theorem: The light glows 165 times between 1:57:58 AM and 3:20:47 AM -/
theorem light_glows_165_times : 
  (completeCycles (durationBetween 1 57 58 3 20 47) * glowPattern.length) + 
  countRemainingGlows (remainingSeconds (durationBetween 1 57 58 3 20 47)) = 165 := by
  sorry


end NUMINAMATH_CALUDE_light_glows_165_times_l1943_194357


namespace NUMINAMATH_CALUDE_tangent_line_inverse_l1943_194373

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
variable (h_inverse : Function.RightInverse f_inv f ∧ Function.LeftInverse f_inv f)

-- Define a point x
variable (x : ℝ)

-- Define the tangent line to f at (x, f(x))
def tangent_line_f (t : ℝ) : ℝ := 2 * t - 3

-- State the theorem
theorem tangent_line_inverse (h_tangent : ∀ t, f x = tangent_line_f t - t) :
  ∀ t, x = t - 2 * (f_inv t) - 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_inverse_l1943_194373


namespace NUMINAMATH_CALUDE_problem_statement_l1943_194381

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.exp x = 0.1

-- Define the perpendicularity condition for two lines
def perpendicular (a : ℝ) : Prop :=
  ∀ x y : ℝ, (x - a * y = 0) → (2 * x + a * y - 1 = 0) → 
  (1 / a) * (-2 / a) = -1

-- Define proposition q
def q : Prop := ∀ a : ℝ, perpendicular a ↔ a = Real.sqrt 2

-- The theorem to be proved
theorem problem_statement : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1943_194381


namespace NUMINAMATH_CALUDE_consecutive_odd_product_square_l1943_194309

theorem consecutive_odd_product_square : 
  ∃ (n : ℤ), (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_square_l1943_194309


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1943_194358

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

/-- The problem statement -/
theorem symmetric_points_sum (m n : ℝ) 
  (h : symmetric_wrt_origin (-3, m) (n, 2)) : 
  m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1943_194358


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_gt_zero_l1943_194317

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) :=
by sorry

theorem negation_of_gt_zero :
  (¬∀ x : ℝ, x^2 - 2*x + 1 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_gt_zero_l1943_194317


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l1943_194337

theorem square_sum_given_product_and_sum (a b : ℝ) 
  (h1 : a * b = 16) 
  (h2 : a + b = 10) : 
  a^2 + b^2 = 68 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l1943_194337
