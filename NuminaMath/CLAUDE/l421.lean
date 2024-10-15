import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_l421_42144

theorem unique_solution : ∃! x : ℝ, x^2 + 50 = (x - 10)^2 ∧ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l421_42144


namespace NUMINAMATH_CALUDE_am_gm_inequality_special_case_l421_42102

theorem am_gm_inequality_special_case (x : ℝ) (h : x > 0) :
  x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_am_gm_inequality_special_case_l421_42102


namespace NUMINAMATH_CALUDE_joyce_gave_six_pencils_l421_42176

/-- The number of pencils Joyce gave to Eugene -/
def pencils_from_joyce (initial_pencils final_pencils : ℕ) : ℕ :=
  final_pencils - initial_pencils

/-- Theorem stating that Joyce gave Eugene 6 pencils -/
theorem joyce_gave_six_pencils (h1 : pencils_from_joyce 51 57 = 6) : 
  pencils_from_joyce 51 57 = 6 := by
  sorry

end NUMINAMATH_CALUDE_joyce_gave_six_pencils_l421_42176


namespace NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l421_42181

theorem vector_addition_and_scalar_multiplication :
  (3 : ℝ) • (⟨4, -2⟩ : ℝ × ℝ) + ⟨-3, 5⟩ = ⟨9, -1⟩ := by sorry

end NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l421_42181


namespace NUMINAMATH_CALUDE_secant_radius_ratio_l421_42122

/-- Represents a circle with a secant line and two squares inside --/
structure SecantSquaresCircle where
  /-- Radius of the circle --/
  radius : ℝ
  /-- Length of the secant line --/
  secant_length : ℝ
  /-- Side length of the smaller square --/
  small_square_side : ℝ
  /-- Side length of the larger square --/
  large_square_side : ℝ
  /-- The squares have two corners on the secant line and two on the circumference --/
  squares_position : Bool
  /-- The ratio of the square's side lengths is 5:9 --/
  side_ratio : small_square_side / large_square_side = 5 / 9

/-- The theorem stating the relationship between secant length and radius --/
theorem secant_radius_ratio (c : SecantSquaresCircle) :
  c.squares_position → c.secant_length / c.radius = 3 * Real.sqrt 10 / 5 :=
by sorry

end NUMINAMATH_CALUDE_secant_radius_ratio_l421_42122


namespace NUMINAMATH_CALUDE_john_gave_money_to_two_friends_l421_42165

/-- Calculates the number of friends John gave money to -/
def number_of_friends (initial_amount spent_on_sweets amount_per_friend final_amount : ℚ) : ℕ :=
  ((initial_amount - spent_on_sweets - final_amount) / amount_per_friend).num.toNat

/-- Proves that John gave money to 2 friends -/
theorem john_gave_money_to_two_friends :
  number_of_friends 7.10 1.05 1.00 4.05 = 2 := by
  sorry

#eval number_of_friends 7.10 1.05 1.00 4.05

end NUMINAMATH_CALUDE_john_gave_money_to_two_friends_l421_42165


namespace NUMINAMATH_CALUDE_f_range_theorem_l421_42179

noncomputable def f (x : ℝ) : ℝ := x + (Real.exp x)⁻¹

theorem f_range_theorem :
  {a : ℝ | ∀ x, f x > a * x} = Set.Ioo (1 - Real.exp 1) 1 :=
sorry

end NUMINAMATH_CALUDE_f_range_theorem_l421_42179


namespace NUMINAMATH_CALUDE_coffee_stock_decaf_percentage_l421_42198

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
theorem coffee_stock_decaf_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (additional_stock : ℝ)
  (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 70) :
  let total_stock := initial_stock + additional_stock
  let total_decaf := (initial_stock * initial_decaf_percent / 100) +
                     (additional_stock * additional_decaf_percent / 100)
  total_decaf / total_stock * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_coffee_stock_decaf_percentage_l421_42198


namespace NUMINAMATH_CALUDE_quadratic_maximum_l421_42105

theorem quadratic_maximum : 
  (∀ s : ℝ, -3 * s^2 + 36 * s + 7 ≤ 115) ∧ 
  (∃ s : ℝ, -3 * s^2 + 36 * s + 7 = 115) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l421_42105


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l421_42156

/-- A geometric sequence with the given property has common ratio 2 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)  -- The geometric sequence
  (h : ∀ n, a n * a (n + 1) = 4^n)  -- The given condition
  : (∃ q : ℝ, q = 2 ∧ ∀ n, a (n + 1) = q * a n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l421_42156


namespace NUMINAMATH_CALUDE_whitney_money_left_l421_42130

/-- The amount of money Whitney has left over after her purchase at the school book fair -/
def money_left_over : ℕ :=
  let initial_money : ℕ := 2 * 20
  let poster_cost : ℕ := 5
  let notebook_cost : ℕ := 4
  let bookmark_cost : ℕ := 2
  let num_posters : ℕ := 2
  let num_notebooks : ℕ := 3
  let num_bookmarks : ℕ := 2
  let total_cost : ℕ := poster_cost * num_posters + notebook_cost * num_notebooks + bookmark_cost * num_bookmarks
  initial_money - total_cost

theorem whitney_money_left : money_left_over = 14 := by
  sorry

end NUMINAMATH_CALUDE_whitney_money_left_l421_42130


namespace NUMINAMATH_CALUDE_coefficient_sum_l421_42148

variables (a b c d e : ℝ)

/-- The polynomial equation -/
def polynomial (x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- Theorem stating the relationship between the coefficients of the polynomial -/
theorem coefficient_sum (h1 : a ≠ 0)
  (h2 : polynomial 5 = 0)
  (h3 : polynomial (-3) = 0)
  (h4 : polynomial 1 = 0) :
  (b + c + d) / a = -7 := by sorry

end NUMINAMATH_CALUDE_coefficient_sum_l421_42148


namespace NUMINAMATH_CALUDE_sum_of_roots_l421_42142

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 15*a^2 + 25*a - 75 = 0)
  (hb : 8*b^3 - 60*b^2 - 310*b + 2675 = 0) :
  a + b = 15/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l421_42142


namespace NUMINAMATH_CALUDE_manuscript_cost_is_1350_l421_42139

/-- Calculates the total cost of typing a manuscript with given parameters. -/
def manuscript_typing_cost (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) 
  (first_time_cost : ℕ) (revision_cost : ℕ) : ℕ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let first_time_total := total_pages * first_time_cost
  let revision_once_total := pages_revised_once * revision_cost
  let revision_twice_total := pages_revised_twice * revision_cost * 2
  first_time_total + revision_once_total + revision_twice_total

/-- The total cost of typing the manuscript is $1350. -/
theorem manuscript_cost_is_1350 : 
  manuscript_typing_cost 100 30 20 10 5 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_is_1350_l421_42139


namespace NUMINAMATH_CALUDE_maria_towels_l421_42109

theorem maria_towels (green_towels white_towels given_away : ℕ) 
  (h1 : green_towels = 125)
  (h2 : white_towels = 130)
  (h3 : given_away = 180) :
  green_towels + white_towels - given_away = 75 := by
  sorry

end NUMINAMATH_CALUDE_maria_towels_l421_42109


namespace NUMINAMATH_CALUDE_lunch_break_duration_l421_42184

/-- Represents the painting rate of a person in house/hour -/
structure PaintingRate :=
  (rate : ℝ)

/-- Represents a day's painting session -/
structure PaintingDay :=
  (duration : ℝ)  -- in hours
  (totalRate : ℝ)  -- combined rate of painters
  (portionPainted : ℝ)  -- portion of house painted

def calculateLunchBreak (monday : PaintingDay) (tuesday : PaintingDay) (wednesday : PaintingDay) : ℝ :=
  sorry

theorem lunch_break_duration :
  let paula : PaintingRate := ⟨0.5⟩
  let helpers : PaintingRate := ⟨0.25⟩  -- Combined rate of two helpers
  let apprentice : PaintingRate := ⟨0⟩
  let monday : PaintingDay := ⟨9, paula.rate + helpers.rate + apprentice.rate, 0.6⟩
  let tuesday : PaintingDay := ⟨7, helpers.rate + apprentice.rate, 0.3⟩
  let wednesday : PaintingDay := ⟨1.2, paula.rate + apprentice.rate, 0.1⟩
  calculateLunchBreak monday tuesday wednesday = 1.4 :=
by sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l421_42184


namespace NUMINAMATH_CALUDE_shaded_area_in_square_configuration_l421_42193

/-- The area of the shaded region in a square configuration -/
theorem shaded_area_in_square_configuration 
  (total_area : ℝ) 
  (overlap_area : ℝ) 
  (area_ratio : ℝ) 
  (h1 : total_area = 196) 
  (h2 : overlap_area = 1) 
  (h3 : area_ratio = 4) : 
  ∃ (shaded_area : ℝ), shaded_area = 72 ∧ 
  ∃ (small_square_area large_square_area : ℝ),
    large_square_area = area_ratio * small_square_area ∧
    shaded_area = large_square_area + small_square_area - overlap_area ∧
    large_square_area + small_square_area - overlap_area < total_area := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_in_square_configuration_l421_42193


namespace NUMINAMATH_CALUDE_pharmaceutical_optimization_l421_42127

/-- Calculates the minimum number of experiments required for the fractional method -/
def min_experiments (lower_temp upper_temp accuracy : ℝ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of experiments for the given conditions -/
theorem pharmaceutical_optimization :
  min_experiments 29 63 1 = 7 := by sorry

end NUMINAMATH_CALUDE_pharmaceutical_optimization_l421_42127


namespace NUMINAMATH_CALUDE_convex_polygon_area_bounds_l421_42150

/-- A convex polygon -/
structure ConvexPolygon where
  area : ℝ
  area_pos : area > 0

/-- A rectangle -/
structure Rectangle where
  area : ℝ
  area_pos : area > 0

/-- A parallelogram -/
structure Parallelogram where
  area : ℝ
  area_pos : area > 0

/-- Theorem: For any convex polygon, there exists an enclosing rectangle with area no more than twice the polygon's area, 
    and an inscribed parallelogram with area at least half the polygon's area -/
theorem convex_polygon_area_bounds (P : ConvexPolygon) :
  (∃ R : Rectangle, R.area ≤ 2 * P.area) ∧ 
  (∃ Q : Parallelogram, Q.area ≥ P.area / 2) :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_area_bounds_l421_42150


namespace NUMINAMATH_CALUDE_ef_is_one_eighth_of_gh_l421_42183

/-- Given a line segment GH with points E and F on it, prove that EF is 1/8 of GH -/
theorem ef_is_one_eighth_of_gh (G E F H : Real) :
  (E ≥ G) → (F ≥ G) → (H ≥ E) → (H ≥ F) →  -- E and F lie on GH
  (E - G = 3 * (H - E)) →  -- GE = 3EH
  (F - G = 7 * (H - F)) →  -- GF = 7FH
  abs (E - F) = (1/8) * (H - G) := by sorry

end NUMINAMATH_CALUDE_ef_is_one_eighth_of_gh_l421_42183


namespace NUMINAMATH_CALUDE_negation_of_union_membership_l421_42143

theorem negation_of_union_membership {α : Type*} (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B := by
  sorry

end NUMINAMATH_CALUDE_negation_of_union_membership_l421_42143


namespace NUMINAMATH_CALUDE_locus_of_points_with_constant_sum_of_distances_l421_42149

/-- Represents a line in a plane -/
structure Line where
  -- Add necessary fields for a line

/-- Represents a point in a plane -/
structure Point where
  -- Add necessary fields for a point

/-- Perpendicular distance between a point and a line -/
def perpendicularDistance (p : Point) (l : Line) : ℝ := sorry

/-- Check if two lines are parallel -/
def areParallel (l1 l2 : Line) : Prop := sorry

/-- The locus of points satisfying the given conditions -/
inductive Locus
  | Empty
  | Parallelogram
  | CentrallySymmetricOctagon

/-- The main theorem statement -/
theorem locus_of_points_with_constant_sum_of_distances
  (L₁ L₂ L₃ L₄ : Line)
  (h_distinct : L₁ ≠ L₂ ∧ L₁ ≠ L₃ ∧ L₁ ≠ L₄ ∧ L₂ ≠ L₃ ∧ L₂ ≠ L₄ ∧ L₃ ≠ L₄)
  (h_parallel₁ : areParallel L₁ L₃)
  (h_parallel₂ : areParallel L₂ L₄)
  (D : ℝ)
  (x : ℝ) -- perpendicular distance between L₁ and L₃
  (y : ℝ) -- perpendicular distance between L₂ and L₄
  (h_x : ∀ (p : Point), perpendicularDistance p L₁ + perpendicularDistance p L₃ = x)
  (h_y : ∀ (p : Point), perpendicularDistance p L₂ + perpendicularDistance p L₄ = y) :
  (∀ (p : Point), 
    perpendicularDistance p L₁ + perpendicularDistance p L₂ + 
    perpendicularDistance p L₃ + perpendicularDistance p L₄ = D) →
  Locus :=
by sorry

end NUMINAMATH_CALUDE_locus_of_points_with_constant_sum_of_distances_l421_42149


namespace NUMINAMATH_CALUDE_password_matches_stored_sequence_l421_42123

/-- Represents a 32-letter alphabet where each letter is encoded as a pair of digits. -/
def Alphabet : Type := Fin 32

/-- Converts a letter to its ordinal number representation. -/
def toOrdinal (a : Alphabet) : Fin 100 := sorry

/-- Represents the remainder when dividing by 10. -/
def r10 (x : ℕ) : Fin 10 := sorry

/-- Generates the x_i sequence based on the given recurrence relation. -/
def genX (a b : ℕ) : ℕ → Fin 10
  | 0 => sorry
  | n + 1 => r10 (a * (genX a b n).val + b)

/-- Generates the c_i sequence based on x_i and y_i. -/
def genC (x : ℕ → Fin 10) (y : ℕ → Fin 100) : ℕ → Fin 10 :=
  fun i => r10 (x i + (y i).val)

/-- Converts a string to a sequence of ordinal numbers. -/
def stringToOrdinals (s : String) : ℕ → Fin 100 := sorry

/-- The stored sequence c_i. -/
def storedSequence : List (Fin 10) :=
  [2, 8, 5, 2, 8, 3, 1, 9, 8, 4, 1, 8, 4, 9, 7, 5]

/-- The password in lowercase letters. -/
def password : String := "яхта"

theorem password_matches_stored_sequence :
  ∃ (a b : ℕ),
    (∀ i, i ≥ 10 → genX a b i = genX a b (i - 10)) ∧
    genC (genX a b) (stringToOrdinals (password ++ password)) = fun i =>
      if h : i < storedSequence.length then
        storedSequence[i]'h
      else
        0 := by sorry

end NUMINAMATH_CALUDE_password_matches_stored_sequence_l421_42123


namespace NUMINAMATH_CALUDE_cost_per_mile_calculation_l421_42175

/-- Calculates the cost per mile for a car rental --/
theorem cost_per_mile_calculation
  (daily_rental_fee : ℚ)
  (daily_budget : ℚ)
  (distance : ℚ)
  (h1 : daily_rental_fee = 30)
  (h2 : daily_budget = 76)
  (h3 : distance = 200)
  : (daily_budget - daily_rental_fee) / distance = 23 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_mile_calculation_l421_42175


namespace NUMINAMATH_CALUDE_pet_farm_hamsters_prove_hamster_count_l421_42195

/-- Given a pet farm with rabbits and hamsters, prove the number of hamsters. -/
theorem pet_farm_hamsters (rabbit_count : ℕ) (ratio_rabbits : ℕ) (ratio_hamsters : ℕ) : ℕ :=
  let hamster_count := (rabbit_count * ratio_hamsters) / ratio_rabbits
  hamster_count

/-- Prove that there are 25 hamsters given the conditions. -/
theorem prove_hamster_count : pet_farm_hamsters 20 4 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_pet_farm_hamsters_prove_hamster_count_l421_42195


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l421_42191

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

theorem smallest_dual_palindrome : ∃ (n : ℕ),
  n > 10 ∧
  is_palindrome n 2 ∧
  is_palindrome n 8 ∧
  (∀ m : ℕ, m > 10 ∧ is_palindrome m 2 ∧ is_palindrome m 8 → n ≤ m) ∧
  n = 63 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l421_42191


namespace NUMINAMATH_CALUDE_triangle_properties_l421_42168

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.c * Real.cos t.B = 2 * t.a + t.b) 
  (h2 : t.a = t.b) 
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (Real.sqrt 3 / 2) * t.c) :
  (t.C = 2 * Real.pi / 3) ∧ 
  (t.a + t.b + t.c = 6 + 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l421_42168


namespace NUMINAMATH_CALUDE_tuna_sales_difference_l421_42192

/-- Calculates the difference in daily sales between peak and low seasons for tuna fish. -/
theorem tuna_sales_difference (peak_rate : ℕ) (low_rate : ℕ) (price : ℕ) (hours : ℕ) 
  (h1 : peak_rate = 6)
  (h2 : low_rate = 4)
  (h3 : price = 60)
  (h4 : hours = 15) :
  peak_rate * price * hours - low_rate * price * hours = 1800 := by
  sorry

#check tuna_sales_difference

end NUMINAMATH_CALUDE_tuna_sales_difference_l421_42192


namespace NUMINAMATH_CALUDE_find_y_value_l421_42135

theorem find_y_value (x y z : ℤ) 
  (eq1 : x + y + z = 270)
  (eq2 : x - y + z = 200)
  (eq3 : x + y - z = 150) :
  y = 35 := by
sorry

end NUMINAMATH_CALUDE_find_y_value_l421_42135


namespace NUMINAMATH_CALUDE_cubic_sum_l421_42178

theorem cubic_sum (a b c : ℝ) 
  (h1 : a + b + c = 8) 
  (h2 : a * b + a * c + b * c = 9) 
  (h3 : a * b * c = -18) : 
  a^3 + b^3 + c^3 = 242 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_l421_42178


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l421_42188

theorem fraction_to_decimal : 
  (52 : ℚ) / 180 = 0.1444444444444444 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l421_42188


namespace NUMINAMATH_CALUDE_toy_production_difference_l421_42158

/-- The difference in daily toy production between two machines -/
theorem toy_production_difference : 
  let machine_a_total : ℕ := 288
  let machine_a_days : ℕ := 12
  let machine_b_total : ℕ := 243
  let machine_b_days : ℕ := 9
  let machine_a_daily : ℚ := machine_a_total / machine_a_days
  let machine_b_daily : ℚ := machine_b_total / machine_b_days
  machine_b_daily - machine_a_daily = 3 := by
sorry

end NUMINAMATH_CALUDE_toy_production_difference_l421_42158


namespace NUMINAMATH_CALUDE_no_real_solutions_for_sqrt_equation_l421_42121

theorem no_real_solutions_for_sqrt_equation (b : ℝ) (h : b > 2) :
  ¬∃ (a x : ℝ), Real.sqrt (b - Real.cos (a + x)) = x := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_sqrt_equation_l421_42121


namespace NUMINAMATH_CALUDE_muffin_selling_price_l421_42161

/-- Represents the daily muffin order quantity -/
def daily_order : ℕ := 12

/-- Represents the cost of each muffin in cents -/
def cost_per_muffin : ℕ := 75

/-- Represents the weekly profit in cents -/
def weekly_profit : ℕ := 6300

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Calculates the selling price of each muffin in cents -/
def selling_price : ℕ :=
  let total_cost := daily_order * cost_per_muffin * days_per_week
  let total_revenue := total_cost + weekly_profit
  let total_muffins := daily_order * days_per_week
  total_revenue / total_muffins

theorem muffin_selling_price :
  selling_price = 150 := by sorry

end NUMINAMATH_CALUDE_muffin_selling_price_l421_42161


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l421_42126

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 9240 →
  Nat.gcd a b = 33 →
  a = 231 →
  b = 1320 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l421_42126


namespace NUMINAMATH_CALUDE_students_without_eyewear_l421_42190

/-- Given a population of students with specified percentages wearing glasses, contact lenses, or both,
    calculate the number of students not wearing any eyewear. -/
theorem students_without_eyewear
  (total_students : ℕ)
  (glasses_percent : ℚ)
  (contacts_percent : ℚ)
  (both_percent : ℚ)
  (h_total : total_students = 1200)
  (h_glasses : glasses_percent = 35 / 100)
  (h_contacts : contacts_percent = 25 / 100)
  (h_both : both_percent = 10 / 100) :
  (total_students : ℚ) * (1 - (glasses_percent + contacts_percent - both_percent)) = 600 :=
sorry

end NUMINAMATH_CALUDE_students_without_eyewear_l421_42190


namespace NUMINAMATH_CALUDE_magnitude_of_difference_l421_42164

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, 6]

theorem magnitude_of_difference (x : ℝ) :
  (vector_a 0 / vector_b x 0 = vector_a 1 / vector_b x 1) →
  Real.sqrt ((vector_a 0 - vector_b x 0)^2 + (vector_a 1 - vector_b x 1)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_difference_l421_42164


namespace NUMINAMATH_CALUDE_chase_travel_time_l421_42186

-- Define the speeds of Chase, Cameron, and Danielle relative to Chase's speed
def chase_speed : ℝ := 1
def cameron_speed : ℝ := 2 * chase_speed
def danielle_speed : ℝ := 3 * cameron_speed

-- Define Danielle's travel time
def danielle_time : ℝ := 30

-- Theorem to prove
theorem chase_travel_time :
  let chase_time := danielle_time * (danielle_speed / chase_speed)
  chase_time = 180 := by sorry

end NUMINAMATH_CALUDE_chase_travel_time_l421_42186


namespace NUMINAMATH_CALUDE_impossibility_of_triangle_formation_l421_42113

theorem impossibility_of_triangle_formation (n : ℕ) (h : n = 10) :
  ∃ (segments : Fin n → ℝ),
    ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k →
      ¬(segments i + segments j > segments k ∧
        segments j + segments k > segments i ∧
        segments k + segments i > segments j) :=
by sorry

end NUMINAMATH_CALUDE_impossibility_of_triangle_formation_l421_42113


namespace NUMINAMATH_CALUDE_system_two_solutions_iff_l421_42107

def system_has_two_solutions (a : ℝ) : Prop :=
  ∃! (s₁ s₂ : ℝ × ℝ), s₁ ≠ s₂ ∧
    (∀ (x y : ℝ), (x, y) = s₁ ∨ (x, y) = s₂ →
      a^2 - 2*a*x + 10*y + x^2 + y^2 = 0 ∧
      (|x| - 12)^2 + (|y| - 5)^2 = 169)

theorem system_two_solutions_iff (a : ℝ) :
  system_has_two_solutions a ↔ 
  (a > -30 ∧ a < -20) ∨ a = 0 ∨ (a > 20 ∧ a < 30) :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_iff_l421_42107


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l421_42185

/-- The number of different rectangles with sides parallel to the grid
    that can be formed by connecting four dots in a 5x5 square array of dots. -/
def num_rectangles_in_5x5_grid : ℕ :=
  (Nat.choose 5 2) * (Nat.choose 5 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100. -/
theorem rectangles_in_5x5_grid :
  num_rectangles_in_5x5_grid = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l421_42185


namespace NUMINAMATH_CALUDE_range_of_k_l421_42136

noncomputable def h (x : ℝ) : ℝ := 5 * x + 3

noncomputable def k (x : ℝ) : ℝ := h (h (h x))

theorem range_of_k :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3,
  ∃ y ∈ Set.Icc (-32 : ℝ) 468,
  k x = y ∧
  ∀ z ∈ Set.Icc (-32 : ℝ) 468,
  ∃ w ∈ Set.Icc (-1 : ℝ) 3,
  k w = z :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l421_42136


namespace NUMINAMATH_CALUDE_total_school_population_l421_42172

/-- Represents the number of people in different categories in a school -/
structure SchoolPopulation where
  male_students : ℕ
  female_students : ℕ
  staff : ℕ

/-- The conditions of the school population -/
def school_conditions (p : SchoolPopulation) : Prop :=
  p.male_students = 4 * p.female_students ∧
  p.female_students = 7 * p.staff

/-- The theorem stating the total number of people in the school -/
theorem total_school_population (p : SchoolPopulation) 
  (h : school_conditions p) : 
  p.male_students + p.female_students + p.staff = (9 / 7 : ℚ) * p.male_students :=
by
  sorry


end NUMINAMATH_CALUDE_total_school_population_l421_42172


namespace NUMINAMATH_CALUDE_greatest_difference_units_digit_l421_42112

/-- Given a three-digit integer in the form 72x that is a multiple of 3,
    the greatest possible difference between two possibilities for the units digit is 9. -/
theorem greatest_difference_units_digit :
  ∀ x : ℕ,
  x < 10 →
  (720 + x) % 3 = 0 →
  ∃ y z : ℕ,
  y < 10 ∧ z < 10 ∧
  (720 + y) % 3 = 0 ∧
  (720 + z) % 3 = 0 ∧
  y - z = 9 ∧
  ∀ a b : ℕ,
  a < 10 → b < 10 →
  (720 + a) % 3 = 0 →
  (720 + b) % 3 = 0 →
  a - b ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_difference_units_digit_l421_42112


namespace NUMINAMATH_CALUDE_orthocenters_collinear_l421_42177

-- Define a type for points in a plane
variable (Point : Type*)

-- Define a type for lines in a plane
variable (Line : Type*)

-- Define a function to determine if three lines form a triangle
variable (form_triangle : Line → Line → Line → Prop)

-- Define a function to get the orthocenter of a triangle formed by three lines
variable (orthocenter : Line → Line → Line → Point)

-- Define a function to check if points are collinear
variable (collinear : List Point → Prop)

-- Theorem statement
theorem orthocenters_collinear 
  (l₁ l₂ l₃ l₄ : Line)
  (h : ∀ (a b c : Line), a ∈ [l₁, l₂, l₃, l₄] → b ∈ [l₁, l₂, l₃, l₄] → c ∈ [l₁, l₂, l₃, l₄] → 
     a ≠ b ∧ b ≠ c ∧ a ≠ c → form_triangle a b c) :
  collinear [
    orthocenter l₁ l₂ l₃,
    orthocenter l₁ l₂ l₄,
    orthocenter l₁ l₃ l₄,
    orthocenter l₂ l₃ l₄
  ] := by
  sorry

end NUMINAMATH_CALUDE_orthocenters_collinear_l421_42177


namespace NUMINAMATH_CALUDE_count_divisors_not_mult_14_l421_42132

def n : ℕ := sorry

-- n is the smallest positive integer satisfying the conditions
axiom n_minimal : ∀ m : ℕ, m > 0 → m < n →
  ¬(∃ k : ℕ, m / 2 = k ^ 2) ∨
  ¬(∃ k : ℕ, m / 3 = k ^ 3) ∨
  ¬(∃ k : ℕ, m / 5 = k ^ 5) ∨
  ¬(∃ k : ℕ, m / 7 = k ^ 7)

-- n satisfies the conditions
axiom n_div_2_square : ∃ k : ℕ, n / 2 = k ^ 2
axiom n_div_3_cube : ∃ k : ℕ, n / 3 = k ^ 3
axiom n_div_5_fifth : ∃ k : ℕ, n / 5 = k ^ 5
axiom n_div_7_seventh : ∃ k : ℕ, n / 7 = k ^ 7

def divisors_not_mult_14 (n : ℕ) : ℕ := sorry

theorem count_divisors_not_mult_14 : divisors_not_mult_14 n = 19005 := by sorry

end NUMINAMATH_CALUDE_count_divisors_not_mult_14_l421_42132


namespace NUMINAMATH_CALUDE_incorrect_inequality_l421_42151

theorem incorrect_inequality (x y : ℝ) (h : x > y) : ¬(3 - x > 3 - y) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l421_42151


namespace NUMINAMATH_CALUDE_vector_dot_product_inequality_l421_42159

theorem vector_dot_product_inequality (α β γ α₁ β₁ γ₁ : ℝ) 
  (h1 : α^2 + β^2 + γ^2 = 1) 
  (h2 : α₁^2 + β₁^2 + γ₁^2 = 1) : 
  α * α₁ + β * β₁ + γ * γ₁ ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_inequality_l421_42159


namespace NUMINAMATH_CALUDE_absolute_value_and_exponent_simplification_l421_42134

theorem absolute_value_and_exponent_simplification :
  |(-3 : ℝ)| + (3 - Real.sqrt 3) ^ (0 : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponent_simplification_l421_42134


namespace NUMINAMATH_CALUDE_power_of_1024_l421_42125

theorem power_of_1024 : 
  (1024 : ℝ) ^ (0.25 : ℝ) * (1024 : ℝ) ^ (0.2 : ℝ) = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_1024_l421_42125


namespace NUMINAMATH_CALUDE_completing_square_transformation_l421_42197

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l421_42197


namespace NUMINAMATH_CALUDE_student_number_problem_l421_42171

theorem student_number_problem (x : ℝ) : 2 * x - 200 = 110 → x = 155 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l421_42171


namespace NUMINAMATH_CALUDE_joint_savings_theorem_l421_42170

/-- Calculates the total amount in a joint savings account given two people's earnings and savings rates. -/
def joint_savings (kimmie_earnings : ℝ) (zahra_reduction_rate : ℝ) (savings_rate : ℝ) : ℝ :=
  let zahra_earnings := kimmie_earnings * (1 - zahra_reduction_rate)
  let kimmie_savings := kimmie_earnings * savings_rate
  let zahra_savings := zahra_earnings * savings_rate
  kimmie_savings + zahra_savings

/-- Theorem stating that under given conditions, the joint savings amount to $375. -/
theorem joint_savings_theorem :
  joint_savings 450 (1/3) (1/2) = 375 := by
  sorry

end NUMINAMATH_CALUDE_joint_savings_theorem_l421_42170


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l421_42163

open Set
open Function
open Real

theorem solution_set_of_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : f 1 = 4) (h3 : ∀ x, deriv f x < 3) :
  {x : ℝ | f (Real.log x) > 3 * Real.log x + 1} = Ioo 0 (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l421_42163


namespace NUMINAMATH_CALUDE_grocer_average_sale_l421_42137

theorem grocer_average_sale 
  (sales : List ℕ) 
  (h1 : sales = [5266, 5744, 5864, 6122, 6588, 4916]) :
  (sales.sum / sales.length : ℚ) = 5750 := by
  sorry

end NUMINAMATH_CALUDE_grocer_average_sale_l421_42137


namespace NUMINAMATH_CALUDE_equation_solution_l421_42108

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (5 * x) ^ 15 = (25 * x) ^ 5 ↔ x = Real.sqrt (1 / 5) ∨ x = -Real.sqrt (1 / 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l421_42108


namespace NUMINAMATH_CALUDE_alien_running_time_l421_42141

/-- The time taken by an Alien to run a certain distance when chasing different animals -/
theorem alien_running_time 
  (speed_rabbit : ℝ) 
  (speed_frog : ℝ) 
  (time_difference : ℝ) 
  (h1 : speed_rabbit = 15) 
  (h2 : speed_frog = 10) 
  (h3 : time_difference = 0.5) :
  ∃ (distance : ℝ) (time_rabbit : ℝ),
    distance = speed_rabbit * time_rabbit ∧
    distance = speed_frog * (time_rabbit + time_difference) ∧
    time_rabbit + time_difference = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_alien_running_time_l421_42141


namespace NUMINAMATH_CALUDE_square_root_sum_equals_six_l421_42169

theorem square_root_sum_equals_six :
  Real.sqrt ((3 - 2 * Real.sqrt 3) ^ 2) + Real.sqrt ((3 + 2 * Real.sqrt 3) ^ 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_six_l421_42169


namespace NUMINAMATH_CALUDE_hash_3_8_l421_42189

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- Theorem statement
theorem hash_3_8 : hash 3 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_hash_3_8_l421_42189


namespace NUMINAMATH_CALUDE_rounded_number_bounds_l421_42157

def rounded_number : ℕ := 180000

theorem rounded_number_bounds :
  ∃ (min max : ℕ),
    (min ≤ rounded_number ∧ rounded_number < min + 5000) ∧
    (max - 5000 < rounded_number ∧ rounded_number ≤ max) ∧
    min = 175000 ∧ max = 184999 :=
by sorry

end NUMINAMATH_CALUDE_rounded_number_bounds_l421_42157


namespace NUMINAMATH_CALUDE_wombat_count_l421_42199

/-- The number of wombats in the enclosure -/
def num_wombats : ℕ := 9

/-- The number of rheas in the enclosure -/
def num_rheas : ℕ := 3

/-- The number of times each wombat claws -/
def wombat_claws : ℕ := 4

/-- The number of times each rhea claws -/
def rhea_claws : ℕ := 1

/-- The total number of claws -/
def total_claws : ℕ := 39

theorem wombat_count : 
  num_wombats * wombat_claws + num_rheas * rhea_claws = total_claws :=
by sorry

end NUMINAMATH_CALUDE_wombat_count_l421_42199


namespace NUMINAMATH_CALUDE_inverse_of_42_mod_43_and_59_l421_42174

theorem inverse_of_42_mod_43_and_59 :
  (∃ x : ℤ, (42 * x) % 43 = 1) ∧ (∃ y : ℤ, (42 * y) % 59 = 1) := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_42_mod_43_and_59_l421_42174


namespace NUMINAMATH_CALUDE_vector_difference_l421_42154

/-- Given two vectors AB and AC in 2D space, prove that BC is their difference --/
theorem vector_difference (AB AC : Fin 2 → ℝ) (h1 : AB = ![2, 3]) (h2 : AC = ![4, 7]) :
  AC - AB = ![2, 4] := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_l421_42154


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l421_42140

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4/y + 1/x = 4) :
  ∀ z w : ℝ, z > 0 → w > 0 → 4/w + 1/z = 4 → x + y ≤ z + w ∧ x + y = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l421_42140


namespace NUMINAMATH_CALUDE_inequality_proof_l421_42162

theorem inequality_proof (a b c : ℕ) (ha : a = 8^53) (hb : b = 16^41) (hc : c = 64^27) :
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l421_42162


namespace NUMINAMATH_CALUDE_max_degree_theorem_l421_42116

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The degree of a polynomial -/
def degree (p : RealPolynomial) : ℕ := sorry

/-- The number of coefficients equal to 1 in a polynomial -/
def num_coeff_one (p : RealPolynomial) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def num_real_roots (p : RealPolynomial) : ℕ := sorry

/-- The maximum degree of a polynomial satisfying the given conditions -/
def max_degree : ℕ := 4

theorem max_degree_theorem :
  ∀ (p : RealPolynomial),
    num_coeff_one p ≥ degree p →
    num_real_roots p = degree p →
    degree p ≤ max_degree :=
by sorry

end NUMINAMATH_CALUDE_max_degree_theorem_l421_42116


namespace NUMINAMATH_CALUDE_percentage_problem_l421_42104

theorem percentage_problem (P : ℝ) : P = 20 := by
  have h1 : 50 = P / 100 * 15 + 47 := by sorry
  have h2 : 15 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l421_42104


namespace NUMINAMATH_CALUDE_l_shape_placements_l421_42100

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Bool

/-- Represents an L-shaped figure composed of 3 small squares --/
structure LShape :=
  (orientation : Fin 4)
  (position : Fin 3 × Fin 3)

/-- Checks if an L-shaped figure is valid within the grid --/
def isValidPlacement (g : Grid) (l : LShape) : Bool :=
  sorry

/-- Counts the number of valid placements of the L-shaped figure in the grid --/
def countValidPlacements (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that there are 48 ways to place the L-shaped figure --/
theorem l_shape_placements :
  ∀ g : Grid, countValidPlacements g = 48 :=
sorry

end NUMINAMATH_CALUDE_l_shape_placements_l421_42100


namespace NUMINAMATH_CALUDE_point_in_triangle_property_l421_42117

/-- Triangle in the xy-plane with vertices (0,0), (4,0), and (4,10) -/
def Triangle : Set (ℝ × ℝ) :=
  {p | ∃ (t1 t2 : ℝ), 0 ≤ t1 ∧ 0 ≤ t2 ∧ t1 + t2 ≤ 1 ∧
       p.1 = 4 * t1 + 4 * t2 ∧
       p.2 = 10 * t2}

/-- The theorem states that for any point (a, b) in the defined triangle, a - b ≤ 0 -/
theorem point_in_triangle_property (p : ℝ × ℝ) (h : p ∈ Triangle) : p.1 - p.2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_triangle_property_l421_42117


namespace NUMINAMATH_CALUDE_megan_total_songs_l421_42103

/-- The number of country albums Megan bought -/
def country_albums : ℕ := 2

/-- The number of pop albums Megan bought -/
def pop_albums : ℕ := 8

/-- The number of songs in each album -/
def songs_per_album : ℕ := 7

/-- The total number of songs Megan bought -/
def total_songs : ℕ := (country_albums + pop_albums) * songs_per_album

theorem megan_total_songs : total_songs = 70 := by
  sorry

end NUMINAMATH_CALUDE_megan_total_songs_l421_42103


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l421_42128

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | (x+2)*(4-x) ≥ 0}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a+1}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) : B ∪ C a = B → a ∈ Set.Icc (-2) 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l421_42128


namespace NUMINAMATH_CALUDE_surface_area_of_joined_cubes_l421_42194

/-- The surface area of a cuboid formed by joining two cubes with side length b -/
def cuboid_surface_area (b : ℝ) : ℝ := 10 * b^2

/-- Theorem: The surface area of a cuboid formed by joining two cubes with side length b is 10b^2 -/
theorem surface_area_of_joined_cubes (b : ℝ) (h : b > 0) :
  cuboid_surface_area b = 2 * ((2*b * b) + (2*b * b) + (b * b)) :=
by sorry

end NUMINAMATH_CALUDE_surface_area_of_joined_cubes_l421_42194


namespace NUMINAMATH_CALUDE_prob_different_grades_is_two_thirds_l421_42114

/-- Represents the number of students in each grade -/
def students_per_grade : ℕ := 2

/-- Represents the total number of students -/
def total_students : ℕ := 2 * students_per_grade

/-- Represents the number of students to be selected -/
def selected_students : ℕ := 2

/-- Calculates the probability of selecting two students from different grades -/
def prob_different_grades : ℚ :=
  (students_per_grade * students_per_grade) / (total_students.choose selected_students)

theorem prob_different_grades_is_two_thirds :
  prob_different_grades = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_different_grades_is_two_thirds_l421_42114


namespace NUMINAMATH_CALUDE_number_percentage_equality_l421_42133

theorem number_percentage_equality (x : ℚ) : 
  (35 : ℚ) / 100 * x = (25 : ℚ) / 100 * 40 → x = 200 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_equality_l421_42133


namespace NUMINAMATH_CALUDE_circledot_not_commutative_l421_42152

-- Define a planar vector
structure PlanarVector where
  x : ℝ
  y : ℝ

-- Define the ⊙ operation
def circledot (a b : PlanarVector) : ℝ :=
  a.x * b.y - a.y * b.x

-- Theorem: The ⊙ operation is not commutative
theorem circledot_not_commutative : ¬ ∀ (a b : PlanarVector), circledot a b = circledot b a := by
  sorry

end NUMINAMATH_CALUDE_circledot_not_commutative_l421_42152


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l421_42155

theorem min_sum_absolute_values : 
  ∃ (x : ℝ), (∀ (y : ℝ), |y - 1| + |y - 2| + |y - 3| ≥ |x - 1| + |x - 2| + |x - 3|) ∧ 
  |x - 1| + |x - 2| + |x - 3| = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l421_42155


namespace NUMINAMATH_CALUDE_min_value_of_expression_l421_42180

/-- The minimum value of 2y^2 + x^2 given the system of equations and parameter range -/
theorem min_value_of_expression (a : ℝ) (x y : ℝ) 
  (h1 : a * x - y = 2 * a + 1)
  (h2 : -x + a * y = a)
  (h3 : -0.5 ≤ a ∧ a ≤ 2) :
  ∃ (min_val : ℝ), min_val = -2/9 ∧ 2 * y^2 + x^2 ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l421_42180


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l421_42110

theorem closest_integer_to_cube_root_250 : 
  ∀ n : ℤ, |n - (250 : ℝ)^(1/3)| ≥ |6 - (250 : ℝ)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l421_42110


namespace NUMINAMATH_CALUDE_merchant_loss_l421_42118

theorem merchant_loss (C S : ℝ) (h : C > 0) :
  40 * C = 25 * S → (S - C) / C * 100 = -20 := by
  sorry

end NUMINAMATH_CALUDE_merchant_loss_l421_42118


namespace NUMINAMATH_CALUDE_boat_stream_speed_ratio_l421_42153

/-- Given a boat rowing in a stream, prove that the ratio of the boat's speed in still water
    to the average speed of the stream is 3:1, under certain conditions. -/
theorem boat_stream_speed_ratio :
  ∀ (B S : ℝ), 
    B > 0 → -- The boat's speed in still water is positive
    S > 0 → -- The stream's average speed is positive
    (B - S) / (B + S) = 1 / 2 → -- It takes twice as long to row against the stream as with it
    B / S = 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_stream_speed_ratio_l421_42153


namespace NUMINAMATH_CALUDE_quadratic_roots_shift_l421_42120

theorem quadratic_roots_shift (b c : ℝ) : 
  ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂ ∧ x₁^2 + b*x₁ + c = 0 ∧ x₂^2 + b*x₂ + c = 0) → 
  ¬∃ (y₁ y₂ : ℝ), (y₁ ≠ y₂ ∧ y₁^2 + (b+1)*y₁ + (c+1) = 0 ∧ y₂^2 + (b+1)*y₂ + (c+1) = 0 ∧ 
                   y₁ = x₁ + 1 ∧ y₂ = x₂ + 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_shift_l421_42120


namespace NUMINAMATH_CALUDE_cell_phone_customers_l421_42173

theorem cell_phone_customers (total : ℕ) (us_customers : ℕ) 
  (h1 : total = 7422) (h2 : us_customers = 723) :
  total - us_customers = 6699 := by
  sorry

end NUMINAMATH_CALUDE_cell_phone_customers_l421_42173


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l421_42138

-- Define an isosceles triangle with side lengths 4 and 7
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 4 ∧ b = 7 ∧ a = c) ∨ (a = 7 ∧ b = 4 ∧ a = c)

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → Perimeter a b c = 15 ∨ Perimeter a b c = 18 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l421_42138


namespace NUMINAMATH_CALUDE_jungkook_paper_arrangement_l421_42187

/-- Given the following:
    - Number of bundles of colored paper
    - Number of pieces per bundle
    - Number of rows for arrangement
    - Number of sheets per row
    Calculate the number of additional sheets needed -/
def additional_sheets_needed (bundles : ℕ) (pieces_per_bundle : ℕ) (rows : ℕ) (sheets_per_row : ℕ) : ℕ :=
  (rows * sheets_per_row) - (bundles * pieces_per_bundle)

/-- Theorem stating that given 5 bundles of 8 pieces of colored paper,
    to arrange them into 9 rows of 6 sheets each, 14 additional sheets are needed -/
theorem jungkook_paper_arrangement :
  additional_sheets_needed 5 8 9 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_paper_arrangement_l421_42187


namespace NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l421_42129

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem sum_of_coordinates_after_reflection (x : ℝ) :
  let A : ℝ × ℝ := (x, 8)
  let B : ℝ × ℝ := reflect_over_y_axis A
  A.1 + A.2 + B.1 + B.2 = 16 := by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l421_42129


namespace NUMINAMATH_CALUDE_trick_decks_total_spent_l421_42131

/-- The total amount spent by Victor and his friend on trick decks -/
def total_spent (price_per_deck : ℕ) (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  price_per_deck * (victor_decks + friend_decks)

/-- Theorem stating the total amount spent by Victor and his friend -/
theorem trick_decks_total_spent :
  total_spent 8 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_trick_decks_total_spent_l421_42131


namespace NUMINAMATH_CALUDE_class_average_score_l421_42196

theorem class_average_score (scores : List ℝ) (avg_others : ℝ) : 
  scores.length = 4 →
  scores = [90, 85, 88, 80] →
  avg_others = 82 →
  let total_students : ℕ := 30
  let sum_scores : ℝ := scores.sum + (total_students - 4 : ℕ) * avg_others
  sum_scores / total_students = 82.5 := by
  sorry

end NUMINAMATH_CALUDE_class_average_score_l421_42196


namespace NUMINAMATH_CALUDE_curve_is_rhombus_not_square_l421_42145

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the curve defined by the equation (|x+y|)/(2a) + (|x-y|)/(2b) = 1 -/
def Curve (a b : ℝ) : Set Point :=
  {p : Point | (|p.x + p.y|) / (2 * a) + (|p.x - p.y|) / (2 * b) = 1}

/-- Checks if a quadrilateral is a rhombus -/
def is_rhombus (A B C D : Point) : Prop :=
  let AB := ((B.x - A.x)^2 + (B.y - A.y)^2).sqrt
  let BC := ((C.x - B.x)^2 + (C.y - B.y)^2).sqrt
  let CD := ((D.x - C.x)^2 + (D.y - C.y)^2).sqrt
  let DA := ((A.x - D.x)^2 + (A.y - D.y)^2).sqrt
  AB = BC ∧ BC = CD ∧ CD = DA

/-- Checks if a quadrilateral is a square -/
def is_square (A B C D : Point) : Prop :=
  is_rhombus A B C D ∧
  let AC := ((C.x - A.x)^2 + (C.y - A.y)^2).sqrt
  let BD := ((D.x - B.x)^2 + (D.y - B.y)^2).sqrt
  AC = BD

theorem curve_is_rhombus_not_square (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∃ A B C D : Point,
    A ∈ Curve a b ∧ B ∈ Curve a b ∧ C ∈ Curve a b ∧ D ∈ Curve a b ∧
    is_rhombus A B C D ∧
    ¬is_square A B C D :=
  sorry

end NUMINAMATH_CALUDE_curve_is_rhombus_not_square_l421_42145


namespace NUMINAMATH_CALUDE_second_oldest_age_l421_42147

/-- Represents the ages of three brothers -/
structure BrothersAges where
  youngest : ℕ
  secondOldest : ℕ
  oldest : ℕ

/-- Defines the conditions for the brothers' ages -/
def validAges (ages : BrothersAges) : Prop :=
  ages.youngest + ages.secondOldest + ages.oldest = 34 ∧
  ages.oldest = 3 * ages.youngest ∧
  ages.secondOldest = 2 * ages.youngest - 2

/-- Theorem stating that the second oldest brother is 10 years old -/
theorem second_oldest_age (ages : BrothersAges) (h : validAges ages) : ages.secondOldest = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_oldest_age_l421_42147


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l421_42115

/-- Given two concentric circles where the area between them is 50π square inches
    and the radius of the smaller circle is 5 inches, the length of a chord of the
    larger circle that is tangent to the smaller circle is 10√2 inches. -/
theorem chord_length_concentric_circles (R r : ℝ) (h1 : r = 5) 
    (h2 : π * R^2 - π * r^2 = 50 * π) : 
  ∃ (c : ℝ), c = 10 * Real.sqrt 2 ∧ c^2 = 4 * (R^2 - r^2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l421_42115


namespace NUMINAMATH_CALUDE_log_base_10_of_7_l421_42124

theorem log_base_10_of_7 (p q : ℝ) (hp : Real.log 3 / Real.log 4 = p) (hq : Real.log 7 / Real.log 3 = q) :
  Real.log 7 / Real.log 10 = 2 * p * q / (2 * p * q + 1) := by
  sorry

end NUMINAMATH_CALUDE_log_base_10_of_7_l421_42124


namespace NUMINAMATH_CALUDE_base_conversion_difference_l421_42106

-- Define a function to convert a number from base b to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + base * acc) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [3, 0, 5]
def base1 : Nat := 8

def num2 : List Nat := [1, 6, 5]
def base2 : Nat := 7

-- Theorem statement
theorem base_conversion_difference :
  to_base_10 num1 base1 - to_base_10 num2 base2 = 101 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_difference_l421_42106


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l421_42146

def total_students : ℕ := 500
def cat_owners : ℕ := 75

theorem percentage_of_cat_owners : 
  (cat_owners : ℚ) / total_students * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l421_42146


namespace NUMINAMATH_CALUDE_park_tree_count_l421_42160

/-- Given an initial number of trees and a number of trees to be planted,
    this function calculates the final number of trees. -/
def final_tree_count (initial_trees planted_trees : ℕ) : ℕ :=
  initial_trees + planted_trees

/-- Theorem stating that given 25 initial trees and 73 trees to be planted,
    the final number of trees will be 98. -/
theorem park_tree_count : final_tree_count 25 73 = 98 := by
  sorry

end NUMINAMATH_CALUDE_park_tree_count_l421_42160


namespace NUMINAMATH_CALUDE_tetrahedron_volume_order_l421_42166

/-- Represents a triangle with side lengths a, b, c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2
  ordered : a > b ∧ b > c

/-- Represents the volumes of tetrahedrons formed by folding the triangle --/
structure TetrahedronVolumes where
  V₁ : ℝ
  V₂ : ℝ
  V₃ : ℝ

/-- Calculates the volumes of tetrahedrons formed by folding the triangle --/
def calculateVolumes (t : Triangle) (θ : ℝ) (hθ : 0 < θ ∧ θ < π) : TetrahedronVolumes :=
  sorry

/-- Theorem: The volumes of tetrahedrons satisfy V₁ > V₂ > V₃ --/
theorem tetrahedron_volume_order (t : Triangle) (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
  let v := calculateVolumes t θ hθ
  v.V₁ > v.V₂ ∧ v.V₂ > v.V₃ := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_order_l421_42166


namespace NUMINAMATH_CALUDE_pet_store_bird_count_l421_42167

/-- Given a pet store with bird cages, each containing parrots and parakeets,
    calculate the total number of birds. -/
theorem pet_store_bird_count (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) :
  num_cages = 6 →
  parrots_per_cage = 2 →
  parakeets_per_cage = 7 →
  num_cages * (parrots_per_cage + parakeets_per_cage) = 54 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_bird_count_l421_42167


namespace NUMINAMATH_CALUDE_ball_pit_problem_l421_42182

theorem ball_pit_problem (total : ℕ) (red_fraction : ℚ) (blue_fraction : ℚ) : 
  total = 360 →
  red_fraction = 1/4 →
  blue_fraction = 1/5 →
  ∃ (red blue neither : ℕ),
    red = total * red_fraction ∧
    blue = (total - red) * blue_fraction ∧
    neither = total - red - blue ∧
    neither = 216 :=
by sorry

end NUMINAMATH_CALUDE_ball_pit_problem_l421_42182


namespace NUMINAMATH_CALUDE_initial_boys_on_team_l421_42119

theorem initial_boys_on_team (initial_girls : ℕ) (girls_joined : ℕ) (boys_quit : ℕ) (final_total : ℕ) : 
  initial_girls = 18 →
  girls_joined = 7 →
  boys_quit = 4 →
  final_total = 36 →
  ∃ initial_boys : ℕ, 
    initial_boys = 15 ∧ 
    (initial_girls + initial_boys) + girls_joined - boys_quit = final_total :=
by
  sorry

end NUMINAMATH_CALUDE_initial_boys_on_team_l421_42119


namespace NUMINAMATH_CALUDE_square_root_problem_l421_42111

-- Define the variables
variable (a b c : ℝ)

-- Define the conditions
def condition1 : Prop := (5 * a + 2) ^ (1/3 : ℝ) = 3
def condition2 : Prop := (3 * a + b - 1).sqrt = 4
def condition3 : Prop := c = ⌊(13 : ℝ).sqrt⌋

-- State the theorem
theorem square_root_problem (h1 : condition1 a) (h2 : condition2 a b) (h3 : condition3 c) :
  (3 * a - b + c).sqrt = 4 ∨ (3 * a - b + c).sqrt = -4 :=
sorry

end NUMINAMATH_CALUDE_square_root_problem_l421_42111


namespace NUMINAMATH_CALUDE_difference_of_squares_l421_42101

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l421_42101
