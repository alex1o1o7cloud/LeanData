import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_existence_and_values_l1132_113212

/-- Given a system of equations with parameters α₁, α₂, α₃, α₄, prove that a solution exists
    if and only if α₁ = α₂ = α₃ = α or α₄ = α, and find the solution in this case. -/
theorem system_solution_existence_and_values (α₁ α₂ α₃ α₄ : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    x₁ + x₂ = α₁ * α₂ ∧
    x₁ + x₃ = α₁ * α₃ ∧
    x₁ + x₄ = α₁ * α₄ ∧
    x₂ + x₃ = α₂ * α₃ ∧
    x₂ + x₄ = α₂ * α₄ ∧
    x₃ + x₄ = α₃ * α₄) ↔
  ((α₁ = α₂ ∧ α₂ = α₃) ∨ α₄ = α₂) ∧
  (∃ α β : ℝ,
    (α = α₁ ∧ β = α₄) ∨ (α = α₂ ∧ β = α₁) ∧
    x₁ = α^2 / 2 ∧
    x₂ = α^2 / 2 ∧
    x₃ = α^2 / 2 ∧
    x₄ = α * (β - α / 2)) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_existence_and_values_l1132_113212


namespace NUMINAMATH_CALUDE_largest_red_socks_proof_l1132_113268

/-- The largest number of red socks satisfying the given conditions -/
def largest_red_socks : ℕ := 1164

/-- The total number of socks -/
def total_socks : ℕ := 1936

/-- Probability of selecting two socks of the same color -/
def same_color_prob : ℚ := 3/5

theorem largest_red_socks_proof :
  (total_socks ≤ 2500) ∧
  (largest_red_socks > (total_socks - largest_red_socks)) ∧
  (largest_red_socks * (largest_red_socks - 1) + 
   (total_socks - largest_red_socks) * (total_socks - largest_red_socks - 1)) / 
   (total_socks * (total_socks - 1)) = same_color_prob ∧
  (∀ r : ℕ, r > largest_red_socks → 
    (r ≤ total_socks ∧ r > (total_socks - r) ∧
     (r * (r - 1) + (total_socks - r) * (total_socks - r - 1)) / 
     (total_socks * (total_socks - 1)) = same_color_prob) → false) :=
by sorry

end NUMINAMATH_CALUDE_largest_red_socks_proof_l1132_113268


namespace NUMINAMATH_CALUDE_profit_equation_correct_l1132_113202

/-- Represents the profit scenario for a product with varying price and sales volume. -/
def profit_equation (x : ℝ) : Prop :=
  let initial_purchase_price : ℝ := 35
  let initial_selling_price : ℝ := 40
  let initial_sales_volume : ℝ := 200
  let price_increase : ℝ := x
  let sales_volume_decrease : ℝ := 5 * x
  let new_profit_per_unit : ℝ := (initial_selling_price + price_increase) - initial_purchase_price
  let new_sales_volume : ℝ := initial_sales_volume - sales_volume_decrease
  let total_profit : ℝ := 1870
  (new_profit_per_unit * new_sales_volume) = total_profit

/-- Theorem stating that the given equation correctly represents the profit scenario. -/
theorem profit_equation_correct :
  ∀ x : ℝ, profit_equation x ↔ (x + 5) * (200 - 5 * x) = 1870 :=
sorry

end NUMINAMATH_CALUDE_profit_equation_correct_l1132_113202


namespace NUMINAMATH_CALUDE_flour_in_mixing_bowl_l1132_113233

theorem flour_in_mixing_bowl (total_sugar : ℚ) (total_flour : ℚ) 
  (h1 : total_sugar = 5)
  (h2 : total_flour = 18)
  (h3 : total_flour - total_sugar = 5) :
  total_flour - (total_sugar + 5) = 8 := by
  sorry

end NUMINAMATH_CALUDE_flour_in_mixing_bowl_l1132_113233


namespace NUMINAMATH_CALUDE_veranda_area_is_196_l1132_113224

/-- Represents the dimensions and characteristics of a room with a trapezoidal veranda. -/
structure RoomWithVeranda where
  room_length : ℝ
  room_width : ℝ
  veranda_short_side : ℝ
  veranda_long_side : ℝ

/-- Calculates the area of the trapezoidal veranda surrounding the room. -/
def verandaArea (r : RoomWithVeranda) : ℝ :=
  (r.room_length + 2 * r.veranda_long_side) * (r.room_width + 2 * r.veranda_short_side) - r.room_length * r.room_width

/-- Theorem stating that the area of the trapezoidal veranda is 196 m² for the given dimensions. -/
theorem veranda_area_is_196 (r : RoomWithVeranda)
    (h1 : r.room_length = 17)
    (h2 : r.room_width = 12)
    (h3 : r.veranda_short_side = 2)
    (h4 : r.veranda_long_side = 4) :
    verandaArea r = 196 := by
  sorry

#eval verandaArea { room_length := 17, room_width := 12, veranda_short_side := 2, veranda_long_side := 4 }

end NUMINAMATH_CALUDE_veranda_area_is_196_l1132_113224


namespace NUMINAMATH_CALUDE_no_clax_is_snapp_l1132_113284

-- Define the sets
variable (U : Type) -- Universe set
variable (Clax Ell Snapp Plott : Set U)

-- Define the conditions
variable (h1 : Clax ⊆ Ellᶜ)
variable (h2 : ∃ x, x ∈ Ell ∩ Snapp)
variable (h3 : Snapp ∩ Plott = ∅)

-- State the theorem
theorem no_clax_is_snapp : Clax ∩ Snapp = ∅ := by
  sorry

end NUMINAMATH_CALUDE_no_clax_is_snapp_l1132_113284


namespace NUMINAMATH_CALUDE_exists_multicolor_triangle_l1132_113252

/-- Represents the three possible colors for vertices -/
inductive Color
| Red
| Blue
| Yellow

/-- Represents a vertex in the triangle -/
structure Vertex where
  x : ℝ
  y : ℝ
  color : Color

/-- Represents a small equilateral triangle -/
structure SmallTriangle where
  v1 : Vertex
  v2 : Vertex
  v3 : Vertex

/-- Represents the large equilateral triangle ABC -/
structure LargeTriangle where
  n : ℕ
  smallTriangles : Array SmallTriangle

/-- Predicate to check if a vertex is on side BC -/
def onSideBC (v : Vertex) : Prop := sorry

/-- Predicate to check if a vertex is on side CA -/
def onSideCA (v : Vertex) : Prop := sorry

/-- Predicate to check if a vertex is on side AB -/
def onSideAB (v : Vertex) : Prop := sorry

/-- The main theorem to be proved -/
theorem exists_multicolor_triangle (ABC : LargeTriangle) : 
  (∀ v : Vertex, onSideBC v → v.color ≠ Color.Red) →
  (∀ v : Vertex, onSideCA v → v.color ≠ Color.Blue) →
  (∀ v : Vertex, onSideAB v → v.color ≠ Color.Yellow) →
  ∃ t : SmallTriangle, t ∈ ABC.smallTriangles ∧ 
    t.v1.color ≠ t.v2.color ∧ t.v2.color ≠ t.v3.color ∧ t.v1.color ≠ t.v3.color :=
sorry

end NUMINAMATH_CALUDE_exists_multicolor_triangle_l1132_113252


namespace NUMINAMATH_CALUDE_no_four_consecutive_integers_product_perfect_square_l1132_113210

theorem no_four_consecutive_integers_product_perfect_square :
  ∀ x : ℕ+, ¬∃ y : ℕ, x * (x + 1) * (x + 2) * (x + 3) = y^2 := by
sorry

end NUMINAMATH_CALUDE_no_four_consecutive_integers_product_perfect_square_l1132_113210


namespace NUMINAMATH_CALUDE_unshaded_parts_sum_l1132_113256

theorem unshaded_parts_sum (square_area shaded_area : ℝ) 
  (h1 : square_area = 36) 
  (h2 : shaded_area = 27) 
  (p q r s : ℝ) :
  p + q + r + s = 9 := by sorry

end NUMINAMATH_CALUDE_unshaded_parts_sum_l1132_113256


namespace NUMINAMATH_CALUDE_roots_equality_l1132_113278

theorem roots_equality (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂) 
  (h2 : x₁^2 - x₁ + a = 0) 
  (h3 : x₂^2 - x₂ + a = 0) : 
  |x₁^2 - x₂^2| = 1 ↔ |x₁^3 - x₂^3| = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_equality_l1132_113278


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1132_113209

universe u

def U : Set ℤ := {-1, 0, 2}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1132_113209


namespace NUMINAMATH_CALUDE_max_discount_rate_l1132_113298

theorem max_discount_rate (cost : ℝ) (original_price : ℝ) 
  (h1 : cost = 4) (h2 : original_price = 5) : 
  ∃ (max_discount : ℝ), 
    (∀ (discount : ℝ), discount ≤ max_discount → 
      (original_price * (1 - discount / 100) - cost) / cost ≥ 0.1) ∧
    (∀ (discount : ℝ), discount > max_discount → 
      (original_price * (1 - discount / 100) - cost) / cost < 0.1) ∧
    max_discount = 12 :=
sorry

end NUMINAMATH_CALUDE_max_discount_rate_l1132_113298


namespace NUMINAMATH_CALUDE_real_solutions_condition_l1132_113294

theorem real_solutions_condition (x : ℝ) :
  (∃ y : ℝ, y^2 + 6*x*y + x + 8 = 0) ↔ (x ≤ -8/9 ∨ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_real_solutions_condition_l1132_113294


namespace NUMINAMATH_CALUDE_second_question_percentage_l1132_113251

theorem second_question_percentage 
  (first_correct : ℝ) 
  (neither_correct : ℝ) 
  (both_correct : ℝ) 
  (h1 : first_correct = 63) 
  (h2 : neither_correct = 20) 
  (h3 : both_correct = 33) : 
  ∃ second_correct : ℝ, 
    second_correct = 50 ∧ 
    first_correct + second_correct - both_correct = 100 - neither_correct :=
by sorry

end NUMINAMATH_CALUDE_second_question_percentage_l1132_113251


namespace NUMINAMATH_CALUDE_gyroscope_spin_rate_doubling_time_l1132_113269

/-- The time interval for which a gyroscope's spin rate doubles -/
theorem gyroscope_spin_rate_doubling_time (v₀ v t : ℝ) (h₁ : v₀ = 6.25) (h₂ : v = 400) (h₃ : t = 90) :
  ∃ T : ℝ, v = v₀ * 2^(t/T) ∧ T = 15 := by
  sorry

end NUMINAMATH_CALUDE_gyroscope_spin_rate_doubling_time_l1132_113269


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1132_113292

/-- The quadratic function f(x) = x^2 - 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

theorem quadratic_minimum :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1132_113292


namespace NUMINAMATH_CALUDE_tumbler_price_l1132_113289

theorem tumbler_price (num_tumblers : ℕ) (num_bills : ℕ) (bill_value : ℕ) (change : ℕ) :
  num_tumblers = 10 →
  num_bills = 5 →
  bill_value = 100 →
  change = 50 →
  (num_bills * bill_value - change) / num_tumblers = 45 := by
sorry

end NUMINAMATH_CALUDE_tumbler_price_l1132_113289


namespace NUMINAMATH_CALUDE_expand_product_l1132_113236

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1132_113236


namespace NUMINAMATH_CALUDE_camp_attendance_l1132_113296

theorem camp_attendance (stay_home : ℕ) (difference : ℕ) (camp : ℕ) : 
  stay_home = 777622 → difference = 574664 → camp + difference = stay_home → camp = 202958 := by
sorry

end NUMINAMATH_CALUDE_camp_attendance_l1132_113296


namespace NUMINAMATH_CALUDE_factorization_equality_l1132_113283

theorem factorization_equality (c : ℝ) : 196 * c^3 + 28 * c^2 = 28 * c^2 * (7 * c + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1132_113283


namespace NUMINAMATH_CALUDE_bisection_exact_solution_possible_l1132_113207

/-- The bisection method can potentially find an exact solution -/
theorem bisection_exact_solution_possible
  {f : ℝ → ℝ} {a b : ℝ} (hab : a < b)
  (hf : Continuous f) (hfab : f a * f b < 0) :
  ∃ x ∈ Set.Icc a b, f x = 0 ∧ ∃ n : ℕ, x = (a + b) / 2^(n + 1) :=
sorry

end NUMINAMATH_CALUDE_bisection_exact_solution_possible_l1132_113207


namespace NUMINAMATH_CALUDE_G_fraction_difference_l1132_113226

/-- G is defined as the infinite repeating decimal 0.871871871... -/
def G : ℚ := 871 / 999

/-- The difference between the denominator and numerator when G is expressed as a fraction in lowest terms -/
def denominator_numerator_difference : ℕ := 999 - 871

theorem G_fraction_difference : denominator_numerator_difference = 128 := by
  sorry

end NUMINAMATH_CALUDE_G_fraction_difference_l1132_113226


namespace NUMINAMATH_CALUDE_election_votes_theorem_l1132_113238

theorem election_votes_theorem (total_votes : ℕ) 
  (h1 : ∃ (candidate_votes : ℕ), candidate_votes = (30 * total_votes) / 100)
  (h2 : ∃ (rival_votes : ℕ), rival_votes = (70 * total_votes) / 100)
  (h3 : ∃ (candidate_votes rival_votes : ℕ), rival_votes = candidate_votes + 4000) :
  total_votes = 10000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l1132_113238


namespace NUMINAMATH_CALUDE_complex_modulus_equal_parts_l1132_113265

theorem complex_modulus_equal_parts (b : ℝ) :
  let z : ℂ := (3 - b * Complex.I) / Complex.I
  (z.re = z.im) → Complex.abs z = 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_equal_parts_l1132_113265


namespace NUMINAMATH_CALUDE_profit_percentage_l1132_113211

theorem profit_percentage (C P : ℝ) (h : (2/3) * P = 0.95 * C) : 
  (P - C) / C * 100 = 42.5 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l1132_113211


namespace NUMINAMATH_CALUDE_linda_savings_l1132_113276

theorem linda_savings (savings : ℝ) : 
  (5 / 6 : ℝ) * savings + 500 = savings → savings = 3000 := by
  sorry

end NUMINAMATH_CALUDE_linda_savings_l1132_113276


namespace NUMINAMATH_CALUDE_g_of_two_equals_six_l1132_113201

/-- Given a function g where g(x) = 5x - 4 for all x, prove that g(2) = 6 -/
theorem g_of_two_equals_six (g : ℝ → ℝ) (h : ∀ x, g x = 5 * x - 4) : g 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_equals_six_l1132_113201


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1132_113204

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ x : ℝ, x > 1 → x ≠ 1) ∧ (∃ y : ℝ, y ≠ 1 ∧ ¬(y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1132_113204


namespace NUMINAMATH_CALUDE_max_value_theorem_l1132_113266

theorem max_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : 2*x + 2*y + z = 1) : 
  3*x*y + y*z + z*x ≤ 1/5 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1132_113266


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l1132_113220

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l1132_113220


namespace NUMINAMATH_CALUDE_average_book_width_l1132_113230

def book_widths : List ℝ := [4, 0.5, 1.2, 3, 7.5, 2, 5, 9]

theorem average_book_width :
  (book_widths.sum / book_widths.length : ℝ) = 4.025 := by
  sorry

end NUMINAMATH_CALUDE_average_book_width_l1132_113230


namespace NUMINAMATH_CALUDE_smallest_multiple_l1132_113218

theorem smallest_multiple (n : ℕ) (h : n = 5) : 
  (∃ m : ℕ, m * n - 15 > 2 * n ∧ ∀ k : ℕ, k < m → k * n - 15 ≤ 2 * n) → 
  (∃ m : ℕ, m * n - 15 > 2 * n ∧ ∀ k : ℕ, k < m → k * n - 15 ≤ 2 * n ∧ m = 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1132_113218


namespace NUMINAMATH_CALUDE_pumpkin_relationship_other_orchard_pumpkins_l1132_113246

/-- Represents the number of pumpkins at Sunshine Orchard -/
def sunshine_pumpkins : ℕ := 54

/-- Represents the number of pumpkins at the other orchard -/
def other_pumpkins : ℕ := 14

/-- Theorem stating the relationship between the number of pumpkins at Sunshine Orchard and the other orchard -/
theorem pumpkin_relationship : sunshine_pumpkins = 3 * other_pumpkins + 12 := by
  sorry

/-- Theorem proving that the other orchard has 14 pumpkins given the conditions -/
theorem other_orchard_pumpkins : other_pumpkins = 14 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_relationship_other_orchard_pumpkins_l1132_113246


namespace NUMINAMATH_CALUDE_specific_field_perimeter_l1132_113235

/-- A rectangular field with specific properties -/
structure RectangularField where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_eq : length = breadth + 30
  area_eq : area = length * breadth

/-- The perimeter of a rectangular field -/
def perimeter (field : RectangularField) : ℝ :=
  2 * (field.length + field.breadth)

/-- Theorem stating the perimeter of the specific field is 540 meters -/
theorem specific_field_perimeter :
  ∃ (field : RectangularField), field.area = 18000 ∧ perimeter field = 540 := by
  sorry

end NUMINAMATH_CALUDE_specific_field_perimeter_l1132_113235


namespace NUMINAMATH_CALUDE_match_duration_l1132_113215

theorem match_duration (goals_per_interval : ℝ) (interval_duration : ℝ) (total_goals : ℝ) :
  goals_per_interval = 2 →
  interval_duration = 15 →
  total_goals = 16 →
  (total_goals / goals_per_interval) * interval_duration = 120 :=
by
  sorry

#check match_duration

end NUMINAMATH_CALUDE_match_duration_l1132_113215


namespace NUMINAMATH_CALUDE_two_valid_m_values_l1132_113293

theorem two_valid_m_values : 
  ∃! (s : Finset ℕ), 
    (∀ m ∈ s, m > 0 ∧ (3087 : ℤ) ∣ (m^2 - 3)) ∧ 
    (∀ m : ℕ, m > 0 ∧ (3087 : ℤ) ∣ (m^2 - 3) → m ∈ s) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_valid_m_values_l1132_113293


namespace NUMINAMATH_CALUDE_cubeTowerSurfaceArea_8_l1132_113295

/-- Calculates the surface area of a cube tower -/
def cubeTowerSurfaceArea (n : Nat) : Nat :=
  let sideAreas : Nat → Nat := fun i => 6 * i^2
  let bottomAreas : Nat → Nat := fun i => i^2
  let adjustedAreas : List Nat := (List.range n).map (fun i =>
    if i = 0 then sideAreas (i + 1)
    else sideAreas (i + 1) - bottomAreas (i + 1))
  adjustedAreas.sum

/-- The surface area of a tower of 8 cubes with side lengths 1 to 8 is 1021 -/
theorem cubeTowerSurfaceArea_8 :
  cubeTowerSurfaceArea 8 = 1021 := by
  sorry

end NUMINAMATH_CALUDE_cubeTowerSurfaceArea_8_l1132_113295


namespace NUMINAMATH_CALUDE_ashley_pies_eaten_l1132_113227

theorem ashley_pies_eaten (pies_per_day : ℕ) (days : ℕ) (remaining_pies : ℕ) :
  pies_per_day = 7 → days = 12 → remaining_pies = 34 →
  pies_per_day * days - remaining_pies = 50 := by
  sorry

end NUMINAMATH_CALUDE_ashley_pies_eaten_l1132_113227


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_intersection_A_B_empty_intersection_A_B_equals_A_l1132_113244

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

-- Theorem for part (1)
theorem intersection_A_complement_B (m : ℝ) (h : m = 3) :
  A ∩ (Set.univ \ B m) = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem for part (2)
theorem intersection_A_B_empty (m : ℝ) :
  A ∩ B m = ∅ ↔ m ≤ -2 := by sorry

-- Theorem for part (3)
theorem intersection_A_B_equals_A (m : ℝ) :
  A ∩ B m = A ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_intersection_A_B_empty_intersection_A_B_equals_A_l1132_113244


namespace NUMINAMATH_CALUDE_ashwin_rental_hours_verify_solution_l1132_113203

/-- Calculates the total rental hours given the rental conditions and total amount paid --/
def rental_hours (first_hour_cost : ℕ) (additional_hour_cost : ℕ) (total_paid : ℕ) : ℕ :=
  let additional_hours := (total_paid - first_hour_cost) / additional_hour_cost
  1 + additional_hours

/-- Proves that Ashwin rented the tool for 11 hours given the specified conditions --/
theorem ashwin_rental_hours :
  rental_hours 25 10 125 = 11 := by
  sorry

/-- Verifies the solution satisfies the original problem conditions --/
theorem verify_solution :
  25 + 10 * (rental_hours 25 10 125 - 1) = 125 := by
  sorry

end NUMINAMATH_CALUDE_ashwin_rental_hours_verify_solution_l1132_113203


namespace NUMINAMATH_CALUDE_julio_mocktail_days_l1132_113225

/-- The number of days Julio made mocktails given the specified conditions -/
def mocktail_days (lime_juice_per_mocktail : ℚ) (juice_per_lime : ℚ) (limes_per_dollar : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent * limes_per_dollar * juice_per_lime) / lime_juice_per_mocktail

/-- Theorem stating that Julio made mocktails for 30 days under the given conditions -/
theorem julio_mocktail_days :
  mocktail_days 1 2 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_julio_mocktail_days_l1132_113225


namespace NUMINAMATH_CALUDE_inequality_proof_l1132_113275

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) : 
  x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1132_113275


namespace NUMINAMATH_CALUDE_subset_sums_determine_set_l1132_113282

def three_element_subset_sums (A : Finset ℤ) : Finset ℤ :=
  (A.powerset.filter (λ s => s.card = 3)).image (λ s => s.sum id)

theorem subset_sums_determine_set :
  ∀ A : Finset ℤ,
    A.card = 4 →
    three_element_subset_sums A = {-1, 3, 5, 8} →
    A = {-3, 0, 2, 6} := by
  sorry

end NUMINAMATH_CALUDE_subset_sums_determine_set_l1132_113282


namespace NUMINAMATH_CALUDE_group_size_calculation_l1132_113216

/-- Proves that the number of people in a group is 5, given the average weight increase and weight difference of replaced individuals. -/
theorem group_size_calculation (average_increase : ℝ) (weight_difference : ℝ) : 
  average_increase = 1.5 → weight_difference = 7.5 → 
  (weight_difference / average_increase : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l1132_113216


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l1132_113258

theorem quadratic_form_h_value (x : ℝ) :
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 15 = a * (x + 3/2)^2 + k :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l1132_113258


namespace NUMINAMATH_CALUDE_alloy_mixture_l1132_113299

/-- The percentage of chromium in the first alloy -/
def chromium_percent_1 : ℝ := 12

/-- The percentage of chromium in the second alloy -/
def chromium_percent_2 : ℝ := 10

/-- The amount of the first alloy used (in kg) -/
def amount_1 : ℝ := 15

/-- The percentage of chromium in the new alloy -/
def chromium_percent_new : ℝ := 10.6

/-- The amount of the second alloy used (in kg) -/
def amount_2 : ℝ := 35

theorem alloy_mixture :
  chromium_percent_1 * amount_1 / 100 + chromium_percent_2 * amount_2 / 100 =
  chromium_percent_new * (amount_1 + amount_2) / 100 :=
by sorry

end NUMINAMATH_CALUDE_alloy_mixture_l1132_113299


namespace NUMINAMATH_CALUDE_slope_of_line_l1132_113248

theorem slope_of_line (x y : ℝ) :
  3 * x + 4 * y + 12 = 0 → (y - 0) / (x - 0) = -3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l1132_113248


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1132_113234

/-- For any natural number n, the polynomial 
    x^(2n) - n^2 * x^(n+1) + 2(n^2 - 1) * x^n + 1 - n^2 * x^(n-1) 
    is divisible by (x-1)^3. -/
theorem polynomial_divisibility (n : ℕ) : 
  ∃ q : Polynomial ℚ, (X : Polynomial ℚ)^(2*n) - n^2 * X^(n+1) + 2*(n^2 - 1) * X^n + 1 - n^2 * X^(n-1) = 
    (X - 1)^3 * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1132_113234


namespace NUMINAMATH_CALUDE_student_selection_properties_l1132_113273

/-- Represents the selection of students from different grades -/
structure StudentSelection where
  total : Nat
  first_year : Nat
  second_year : Nat
  third_year : Nat
  selected : Nat

/-- Calculate the probability of selecting students from different grades -/
def prob_different_grades (s : StudentSelection) : Rat :=
  (s.first_year.choose 1 * s.second_year.choose 1 * s.third_year.choose 1) /
  (s.total.choose s.selected)

/-- Calculate the mathematical expectation of the number of first-year students selected -/
def expectation_first_year (s : StudentSelection) : Rat :=
  (0 * (s.total - s.first_year).choose s.selected +
   1 * (s.first_year.choose 1 * (s.total - s.first_year).choose (s.selected - 1)) +
   2 * (s.first_year.choose 2 * (s.total - s.first_year).choose (s.selected - 2))) /
  (s.total.choose s.selected)

/-- The main theorem stating the properties of the student selection problem -/
theorem student_selection_properties (s : StudentSelection) 
  (h1 : s.total = 5)
  (h2 : s.first_year = 2)
  (h3 : s.second_year = 2)
  (h4 : s.third_year = 1)
  (h5 : s.selected = 3) :
  prob_different_grades s = 2/5 ∧ expectation_first_year s = 6/5 := by
  sorry

#eval prob_different_grades ⟨5, 2, 2, 1, 3⟩
#eval expectation_first_year ⟨5, 2, 2, 1, 3⟩

end NUMINAMATH_CALUDE_student_selection_properties_l1132_113273


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1132_113245

/-- A quadratic equation x^2 - 2kx + k^2 + k + 1 = 0 with two real roots -/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  x^2 - 2*k*x + k^2 + k + 1 = 0

/-- The equation has two real roots -/
def has_two_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ ∧ quadratic_equation k x₂

theorem quadratic_equation_properties :
  (∀ k : ℝ, has_two_real_roots k → k ≤ -1) ∧
  (∀ k : ℝ, has_two_real_roots k → (∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ x₁^2 + x₂^2 = 10) → k = -2) ∧
  (∀ k : ℝ, has_two_real_roots k → (∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ |x₁| + |x₂| = 2) → k = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1132_113245


namespace NUMINAMATH_CALUDE_equal_value_nickels_l1132_113279

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters in the first set -/
def quarters_set1 : ℕ := 30

/-- The number of nickels in the first set -/
def nickels_set1 : ℕ := 15

/-- The number of quarters in the second set -/
def quarters_set2 : ℕ := 15

theorem equal_value_nickels : 
  ∃ n : ℕ, 
    quarters_set1 * quarter_value + nickels_set1 * nickel_value = 
    quarters_set2 * quarter_value + n * nickel_value ∧ 
    n = 90 := by
  sorry

end NUMINAMATH_CALUDE_equal_value_nickels_l1132_113279


namespace NUMINAMATH_CALUDE_inverse_on_negative_T_to_0_l1132_113286

-- Define a periodic function f with period T
def isPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Define the smallest positive period
def isSmallestPositivePeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ isPeriodic f T ∧ ∀ S, 0 < S ∧ S < T → ¬isPeriodic f S

-- Define the inverse function on (0, T)
def inverseOn0T (f : ℝ → ℝ) (T : ℝ) (D : Set ℝ) (fInv : ℝ → ℝ) : Prop :=
  ∀ x ∈ D, 0 < x ∧ x < T → f (fInv x) = x ∧ fInv (f x) = x

-- Main theorem
theorem inverse_on_negative_T_to_0
  (f : ℝ → ℝ) (T : ℝ) (D : Set ℝ) (fInv : ℝ → ℝ)
  (h_periodic : isPeriodic f T)
  (h_smallest : isSmallestPositivePeriod f T)
  (h_inverse : inverseOn0T f T D fInv) :
  ∀ x ∈ D, -T < x ∧ x < 0 → f (fInv x - T) = x ∧ fInv x - T = f⁻¹ x :=
sorry

end NUMINAMATH_CALUDE_inverse_on_negative_T_to_0_l1132_113286


namespace NUMINAMATH_CALUDE_probability_sum_eleven_l1132_113240

def seven_sided_die : Finset Nat := Finset.range 7
def five_sided_die : Finset Nat := Finset.range 5

def total_outcomes : Nat := seven_sided_die.card * five_sided_die.card

def successful_outcomes : Finset (Nat × Nat) :=
  {(4, 4), (5, 3), (6, 2)}

theorem probability_sum_eleven :
  (successful_outcomes.card : ℚ) / total_outcomes = 3 / 35 := by
sorry

end NUMINAMATH_CALUDE_probability_sum_eleven_l1132_113240


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1132_113264

theorem complex_magnitude_problem : 
  let z : ℂ := (2 - Complex.I)^2 / Complex.I
  Complex.abs z = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1132_113264


namespace NUMINAMATH_CALUDE_intersection_M_N_l1132_113270

open Set Real

-- Define the sets M and N
def M : Set ℝ := {x | 2 - x > 0}
def N : Set ℝ := Icc 1 3

-- State the theorem
theorem intersection_M_N : M ∩ N = Ico 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1132_113270


namespace NUMINAMATH_CALUDE_distance_between_points_on_lines_l1132_113242

/-- The distance between two points on specific lines -/
theorem distance_between_points_on_lines (a c m k : ℝ) :
  let b := 2 * m * a + k
  let d := -m * c + k
  (((c - a)^2 + (d - b)^2) : ℝ).sqrt = ((1 + m^2 * (c + 2*a)^2) * (c - a)^2 : ℝ).sqrt :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_on_lines_l1132_113242


namespace NUMINAMATH_CALUDE_quadratic_properties_quadratic_max_conditions_l1132_113250

-- Define the quadratic function
def quadratic_function (b c x : ℝ) : ℝ := -x^2 + b*x + c

-- Theorem for part 1
theorem quadratic_properties :
  let f := quadratic_function 4 3
  ∃ (vertex_x vertex_y : ℝ),
    (∀ x, f x ≤ f vertex_x) ∧
    vertex_x = 2 ∧
    vertex_y = 7 ∧
    (∀ x, -1 ≤ x ∧ x ≤ 3 → -2 ≤ f x ∧ f x ≤ 7) :=
sorry

-- Theorem for part 2
theorem quadratic_max_conditions :
  ∃ (b c : ℝ),
    (∀ x ≤ 0, quadratic_function b c x ≤ 2) ∧
    (∀ x > 0, quadratic_function b c x ≤ 3) ∧
    (∃ x ≤ 0, quadratic_function b c x = 2) ∧
    (∃ x > 0, quadratic_function b c x = 3) ∧
    b = 2 ∧
    c = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_quadratic_max_conditions_l1132_113250


namespace NUMINAMATH_CALUDE_wife_weekly_contribution_l1132_113290

def husband_weekly_contribution : ℕ := 335
def savings_weeks : ℕ := 24
def children_count : ℕ := 4
def child_receives : ℕ := 1680

theorem wife_weekly_contribution (wife_contribution : ℕ) :
  (husband_weekly_contribution * savings_weeks + wife_contribution * savings_weeks) / 2 =
  children_count * child_receives →
  wife_contribution = 225 := by
  sorry

end NUMINAMATH_CALUDE_wife_weekly_contribution_l1132_113290


namespace NUMINAMATH_CALUDE_work_completion_time_l1132_113262

theorem work_completion_time (a b : ℝ) (h1 : a = 2 * b) (h2 : 1 / (a + b) = 1 / 20) :
  1 / a = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1132_113262


namespace NUMINAMATH_CALUDE_train_speed_l1132_113213

/-- Proves that a train of given length crossing a bridge of given length in a given time has a specific speed -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  bridge_length = 132 →
  crossing_time = 16.13204276991174 →
  (train_length + bridge_length) / crossing_time * 3.6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1132_113213


namespace NUMINAMATH_CALUDE_mixed_repeating_decimal_denominator_divisibility_l1132_113222

/-- Represents a mixed repeating decimal -/
structure MixedRepeatingDecimal where
  non_repeating : ℕ
  repeating : ℕ

/-- Theorem: For any mixed repeating decimal that can be expressed as an irreducible fraction p/q,
    the denominator q is divisible by 2 or 5, or both. -/
theorem mixed_repeating_decimal_denominator_divisibility
  (x : MixedRepeatingDecimal)
  (p q : ℕ)
  (h_irreducible : Nat.Coprime p q)
  (h_fraction : (p : ℚ) / q = x.non_repeating + (x.repeating : ℚ) / (10^x.non_repeating.succ * (10^x.repeating.succ - 1))) :
  2 ∣ q ∨ 5 ∣ q :=
sorry

end NUMINAMATH_CALUDE_mixed_repeating_decimal_denominator_divisibility_l1132_113222


namespace NUMINAMATH_CALUDE_profit_share_ratio_l1132_113200

theorem profit_share_ratio (total_profit : ℝ) (difference : ℝ) 
  (h_total : total_profit = 1000)
  (h_diff : difference = 200) :
  ∃ (x y : ℝ), 
    x + y = total_profit ∧ 
    x - y = difference ∧ 
    x / total_profit = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l1132_113200


namespace NUMINAMATH_CALUDE_eight_digit_increasing_remainder_l1132_113263

theorem eight_digit_increasing_remainder (n : ℕ) (h : n = 8) :
  (Nat.choose (n + 9 - 1) n) % 1000 = 870 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_increasing_remainder_l1132_113263


namespace NUMINAMATH_CALUDE_conditional_inequality_1_conditional_inequality_2_conditional_log_inequality_conditional_reciprocal_inequality_l1132_113206

-- Statement 1
theorem conditional_inequality_1 (a b c : ℝ) (h1 : a > b) (h2 : c ≤ 0) :
  a * c ≤ b * c := by sorry

-- Statement 2
theorem conditional_inequality_2 (a b c : ℝ) (h1 : a * c^2 > b * c^2) (h2 : b ≥ 0) :
  a^2 > b^2 := by sorry

-- Statement 3
theorem conditional_log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > -1) :
  Real.log (a + 1) > Real.log (b + 1) := by sorry

-- Statement 4
theorem conditional_reciprocal_inequality (a b : ℝ) (h1 : a > b) (h2 : a * b > 0) :
  1 / a < 1 / b := by sorry

end NUMINAMATH_CALUDE_conditional_inequality_1_conditional_inequality_2_conditional_log_inequality_conditional_reciprocal_inequality_l1132_113206


namespace NUMINAMATH_CALUDE_abc_inequality_l1132_113237

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1132_113237


namespace NUMINAMATH_CALUDE_solution_part_i_solution_part_ii_l1132_113297

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

-- Theorem for part (I)
theorem solution_part_i :
  {x : ℝ | f x ≤ 4} = {x : ℝ | -5/3 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part (II)
theorem solution_part_ii :
  {a : ℝ | ∀ x ∈ {x : ℝ | f x ≤ 4}, |x + 3| + |x + a| < x + 6} =
  {a : ℝ | -4/3 < a ∧ a < 2} := by sorry

end NUMINAMATH_CALUDE_solution_part_i_solution_part_ii_l1132_113297


namespace NUMINAMATH_CALUDE_justin_jersey_cost_l1132_113205

/-- The total cost of jerseys bought by Justin -/
def total_cost (long_sleeve_count : ℕ) (long_sleeve_price : ℕ) (striped_count : ℕ) (striped_price : ℕ) : ℕ :=
  long_sleeve_count * long_sleeve_price + striped_count * striped_price

/-- Theorem stating that Justin's total cost for jerseys is $80 -/
theorem justin_jersey_cost :
  total_cost 4 15 2 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_justin_jersey_cost_l1132_113205


namespace NUMINAMATH_CALUDE_complex_modulus_equality_not_implies_square_equality_l1132_113271

theorem complex_modulus_equality_not_implies_square_equality :
  ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_not_implies_square_equality_l1132_113271


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1132_113221

theorem polynomial_factorization (x : ℝ) : 
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3)*(8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1132_113221


namespace NUMINAMATH_CALUDE_figure_100_cubes_l1132_113261

-- Define the sequence of unit cubes for the first four figures
def cube_sequence : Fin 4 → ℕ
  | 0 => 1
  | 1 => 8
  | 2 => 27
  | 3 => 64

-- Define the general formula for the number of cubes in figure n
def num_cubes (n : ℕ) : ℕ := n^3

-- Theorem statement
theorem figure_100_cubes :
  (∀ k : Fin 4, cube_sequence k = num_cubes k) →
  num_cubes 100 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_figure_100_cubes_l1132_113261


namespace NUMINAMATH_CALUDE_function_properties_l1132_113231

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x - 1

-- State the theorem
theorem function_properties (a : ℝ) (h_a : a ≠ 0) :
  -- f(x) has an extremum at x = -1
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -1 ∧ |x + 1| < ε → f a x ≤ f a (-1)) →
  -- The line y = m intersects the graph of y = f(x) at three distinct points
  (∃ (m : ℝ), ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m) →
  -- 1. When a < 0, f(x) is increasing on (-∞, +∞)
  (a < 0 → ∀ (x y : ℝ), x < y → f a x < f a y) ∧
  -- 2. When a > 0, f(x) is increasing on (-∞, -√a) ∪ (√a, +∞) and decreasing on (-√a, √a)
  (a > 0 → (∀ (x y : ℝ), (x < y ∧ y < -Real.sqrt a) ∨ (x > Real.sqrt a ∧ y > x) → f a x < f a y) ∧
           (∀ (x y : ℝ), -Real.sqrt a < x ∧ x < y ∧ y < Real.sqrt a → f a x > f a y)) ∧
  -- 3. The range of values for m is (-3, 1)
  (∃ (m : ℝ), -3 < m ∧ m < 1 ∧
    ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m) ∧
  (∀ (m : ℝ), (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m) → -3 < m ∧ m < 1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1132_113231


namespace NUMINAMATH_CALUDE_del_oranges_per_day_l1132_113272

theorem del_oranges_per_day (total : ℕ) (juan : ℕ) (del_days : ℕ) 
  (h_total : total = 107)
  (h_juan : juan = 61)
  (h_del_days : del_days = 2) :
  (total - juan) / del_days = 23 := by
  sorry

end NUMINAMATH_CALUDE_del_oranges_per_day_l1132_113272


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1132_113260

/-- Proves that a train with given length and speed takes a specific time to cross a bridge -/
theorem train_bridge_crossing_time
  (train_length : Real)
  (train_speed_kmh : Real)
  (total_length : Real)
  (h1 : train_length = 100)
  (h2 : train_speed_kmh = 45)
  (h3 : total_length = 275) :
  let train_speed_ms : Real := train_speed_kmh * 1000 / 3600
  let bridge_length : Real := total_length - train_length
  let distance_to_cross : Real := train_length + bridge_length
  let time_to_cross : Real := distance_to_cross / train_speed_ms
  time_to_cross = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1132_113260


namespace NUMINAMATH_CALUDE_positive_reals_inequality_l1132_113267

theorem positive_reals_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) : 
  (a + b + c ≥ 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c) ∧ 
  (a^2 + b^2 + c^2 ≥ Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequality_l1132_113267


namespace NUMINAMATH_CALUDE_circle_equation_m_range_l1132_113274

/-- Given an equation x^2 + y^2 - 2x - 4y + m = 0 that represents a circle, prove that m < 5 -/
theorem circle_equation_m_range (m : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0 ↔ (x - 1)^2 + (y - 2)^2 = r^2) →
  m < 5 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_m_range_l1132_113274


namespace NUMINAMATH_CALUDE_train_speed_l1132_113249

/-- The speed of a train given its length and time to pass an observer -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) :
  train_length = 100 →
  passing_time = 12 →
  (train_length / 1000) / (passing_time / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1132_113249


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1132_113239

theorem inequality_system_solution_set :
  {x : ℝ | -2*x ≤ 6 ∧ x + 1 < 0} = {x : ℝ | -3 ≤ x ∧ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1132_113239


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1132_113291

theorem rectangle_perimeter (length width : ℝ) 
  (h1 : length * width = 360)
  (h2 : (length + 10) * (width - 6) = 360) :
  2 * (length + width) = 76 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1132_113291


namespace NUMINAMATH_CALUDE_spencer_burritos_l1132_113253

/-- Represents the number of ways to make burritos with given constraints -/
def burrito_combinations (total_burritos : ℕ) (max_beef : ℕ) (max_chicken : ℕ) (available_wraps : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 25 ways to make exactly 5 burritos with the given constraints -/
theorem spencer_burritos : burrito_combinations 5 4 3 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_spencer_burritos_l1132_113253


namespace NUMINAMATH_CALUDE_fraction_equality_l1132_113281

theorem fraction_equality (p q r : ℕ+) 
  (h : (p : ℚ) + 1 / ((q : ℚ) + 1 / (r : ℚ)) = 25 / 19) : 
  q = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1132_113281


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l1132_113243

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (sum_squares : a^2 + b^2 + c^2 = 0.1) :
  a^4 + b^4 + c^4 = 0.005 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l1132_113243


namespace NUMINAMATH_CALUDE_circle_equation_l1132_113254

/-- Theorem: Given a circle with center (a, 0) where a < 0, radius √5, and tangent to the line x + 2y = 0, 
    the equation of the circle is (x + 5)² + y² = 5. -/
theorem circle_equation (a : ℝ) (h1 : a < 0) :
  let r : ℝ := Real.sqrt 5
  let d : ℝ → ℝ → ℝ := λ x y => |x + 2*y| / Real.sqrt 5
  (d a 0 = r) → 
  (∀ x y, (x - a)^2 + y^2 = 5 ↔ (x + 5)^2 + y^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1132_113254


namespace NUMINAMATH_CALUDE_matrix_sum_proof_l1132_113277

theorem matrix_sum_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 0, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; 7, -10]
  A + B = !![-2, 5; 7, -5] := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_proof_l1132_113277


namespace NUMINAMATH_CALUDE_circle_min_area_l1132_113257

/-- Given positive real numbers x and y satisfying the equation 3/(2+x) + 3/(2+y) = 1,
    this theorem states that (x-4)^2 + (y-4)^2 = 256 is the equation of the circle
    with center (x,y) and radius xy when its area is minimized. -/
theorem circle_min_area (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h : 3 / (2 + x) + 3 / (2 + y) = 1) :
    ∃ (center_x center_y : ℝ),
      (x - center_x)^2 + (y - center_y)^2 = (x * y)^2 ∧
      center_x = 4 ∧ center_y = 4 ∧ x * y = 16 ∧
      ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 3 / (2 + x') + 3 / (2 + y') = 1 →
        x' * y' ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_min_area_l1132_113257


namespace NUMINAMATH_CALUDE_mouse_jump_difference_l1132_113280

/-- Proves that the mouse jumped 12 inches less than the frog in the jumping contest. -/
theorem mouse_jump_difference (grasshopper_jump : ℕ) (grasshopper_frog_diff : ℕ) (mouse_jump : ℕ)
  (h1 : grasshopper_jump = 39)
  (h2 : grasshopper_frog_diff = 19)
  (h3 : mouse_jump = 8) :
  grasshopper_jump - grasshopper_frog_diff - mouse_jump = 12 := by
  sorry

end NUMINAMATH_CALUDE_mouse_jump_difference_l1132_113280


namespace NUMINAMATH_CALUDE_percent_equivalence_l1132_113287

theorem percent_equivalence (x : ℝ) (h : 0.3 * 0.05 * x = 18) : 0.05 * 0.3 * x = 18 := by
  sorry

end NUMINAMATH_CALUDE_percent_equivalence_l1132_113287


namespace NUMINAMATH_CALUDE_sum_of_specific_series_l1132_113241

def arithmetic_series (a₁ : ℕ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => a₁ + i * d)

def alternating_sum (l : List ℤ) : ℤ :=
  l.enum.foldl (λ acc (i, x) => acc + (if i % 2 == 0 then x else -x)) 0

theorem sum_of_specific_series :
  let series := arithmetic_series 100 (-2) 50
  alternating_sum series = 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_series_l1132_113241


namespace NUMINAMATH_CALUDE_count_divisible_by_2_3_or_5_count_divisible_by_2_3_or_5_is_74_l1132_113219

theorem count_divisible_by_2_3_or_5 : ℕ :=
  let n := 100
  let A₂ := n / 2
  let A₃ := n / 3
  let A₅ := n / 5
  let A₂₃ := n / 6
  let A₂₅ := n / 10
  let A₃₅ := n / 15
  let A₂₃₅ := n / 30
  A₂ + A₃ + A₅ - A₂₃ - A₂₅ - A₃₅ + A₂₃₅

theorem count_divisible_by_2_3_or_5_is_74 : 
  count_divisible_by_2_3_or_5 = 74 := by sorry

end NUMINAMATH_CALUDE_count_divisible_by_2_3_or_5_count_divisible_by_2_3_or_5_is_74_l1132_113219


namespace NUMINAMATH_CALUDE_selection_methods_count_l1132_113208

def num_students : ℕ := 5
def num_selected : ℕ := 4
def num_days : ℕ := 3
def num_friday : ℕ := 2
def num_saturday : ℕ := 1
def num_sunday : ℕ := 1

theorem selection_methods_count :
  (num_students.choose num_friday) *
  ((num_students - num_friday).choose num_saturday) *
  ((num_students - num_friday - num_saturday).choose num_sunday) = 60 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l1132_113208


namespace NUMINAMATH_CALUDE_carys_savings_l1132_113288

/-- Problem: Cary's Lawn Mowing Savings --/
theorem carys_savings (shoe_cost : ℕ) (saved : ℕ) (earnings_per_lawn : ℕ) (lawns_per_weekend : ℕ)
  (h1 : shoe_cost = 120)
  (h2 : saved = 30)
  (h3 : earnings_per_lawn = 5)
  (h4 : lawns_per_weekend = 3) :
  (shoe_cost - saved) / (lawns_per_weekend * earnings_per_lawn) = 6 := by
  sorry

end NUMINAMATH_CALUDE_carys_savings_l1132_113288


namespace NUMINAMATH_CALUDE_sum_divides_product_iff_not_odd_prime_l1132_113229

theorem sum_divides_product_iff_not_odd_prime (n : ℕ) : 
  (n * (n + 1) / 2) ∣ n! ↔ ¬(Nat.Prime (n + 1) ∧ Odd (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_sum_divides_product_iff_not_odd_prime_l1132_113229


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1132_113223

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 + a 8 = 2 →
  a 6 * a 7 = -8 →
  a 2 + a 11 = -7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1132_113223


namespace NUMINAMATH_CALUDE_not_divisible_by_seven_l1132_113247

theorem not_divisible_by_seven (n : ℤ) : ¬(7 ∣ (n^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_seven_l1132_113247


namespace NUMINAMATH_CALUDE_fraction_inequality_l1132_113285

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  a / d > b / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1132_113285


namespace NUMINAMATH_CALUDE_power_multiplication_l1132_113232

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_l1132_113232


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1132_113217

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + x + 1

-- Define the point through which the tangent line passes
def point : ℝ × ℝ := (-1, 1)

-- Theorem statement
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := (2 * x₀ + 1)  -- Slope of the tangent line
  (∀ x y, y - y₀ = m * (x - x₀)) ↔ (∀ x y, x + y = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1132_113217


namespace NUMINAMATH_CALUDE_existence_of_indistinguishable_arrangements_l1132_113255

/-- Represents the type of a tree -/
inductive TreeType
| Oak
| Baobab

/-- Represents a row of trees -/
def TreeRow := List TreeType

/-- Counts the number of oaks in a group of three adjacent trees -/
def countOaks (trees : TreeRow) (index : Nat) : Nat :=
  match trees.get? index, trees.get? (index + 1), trees.get? (index + 2) with
  | some TreeType.Oak, _, _ => 1
  | _, some TreeType.Oak, _ => 1
  | _, _, some TreeType.Oak => 1
  | _, _, _ => 0

/-- Generates the sequence of tag numbers for a given row of trees -/
def generateTags (trees : TreeRow) : List Nat :=
  List.range trees.length |>.map (countOaks trees)

/-- Theorem stating that there exist two different arrangements of trees
    with the same tag sequence -/
theorem existence_of_indistinguishable_arrangements :
  ∃ (row1 row2 : TreeRow),
    row1.length = 2000 ∧
    row2.length = 2000 ∧
    row1 ≠ row2 ∧
    generateTags row1 = generateTags row2 :=
sorry

end NUMINAMATH_CALUDE_existence_of_indistinguishable_arrangements_l1132_113255


namespace NUMINAMATH_CALUDE_qt_squared_eq_three_l1132_113259

-- Define the points
variable (X Y Z W P Q R S T U : ℝ × ℝ)

-- Define the square XYZW
def is_square (X Y Z W : ℝ × ℝ) : Prop := sorry

-- Define that P and S lie on XZ and XW respectively
def on_line (P X Z : ℝ × ℝ) : Prop := sorry
def on_line' (S X W : ℝ × ℝ) : Prop := sorry

-- Define XP = XS = √3
def distance_eq_sqrt3 (X P S : ℝ × ℝ) : Prop := sorry

-- Define Q and R lie on YZ and YW respectively
def on_line'' (Q Y Z : ℝ × ℝ) : Prop := sorry
def on_line''' (R Y W : ℝ × ℝ) : Prop := sorry

-- Define T and U lie on PS
def on_line'''' (T P S : ℝ × ℝ) : Prop := sorry
def on_line''''' (U P S : ℝ × ℝ) : Prop := sorry

-- Define QT ⊥ PS and RU ⊥ PS
def perpendicular (Q T P S : ℝ × ℝ) : Prop := sorry
def perpendicular' (R U P S : ℝ × ℝ) : Prop := sorry

-- Define areas of the shapes
def area_eq_1_5 (X P S : ℝ × ℝ) : Prop := sorry
def area_eq_1_5' (Y Q T P : ℝ × ℝ) : Prop := sorry
def area_eq_1_5'' (W S U R : ℝ × ℝ) : Prop := sorry
def area_eq_1_5''' (Y R U T Q : ℝ × ℝ) : Prop := sorry

-- The theorem to prove
theorem qt_squared_eq_three 
  (h1 : is_square X Y Z W)
  (h2 : on_line P X Z)
  (h3 : on_line' S X W)
  (h4 : distance_eq_sqrt3 X P S)
  (h5 : on_line'' Q Y Z)
  (h6 : on_line''' R Y W)
  (h7 : on_line'''' T P S)
  (h8 : on_line''''' U P S)
  (h9 : perpendicular Q T P S)
  (h10 : perpendicular' R U P S)
  (h11 : area_eq_1_5 X P S)
  (h12 : area_eq_1_5' Y Q T P)
  (h13 : area_eq_1_5'' W S U R)
  (h14 : area_eq_1_5''' Y R U T Q) :
  (Q.1 - T.1)^2 + (Q.2 - T.2)^2 = 3 := by sorry

end NUMINAMATH_CALUDE_qt_squared_eq_three_l1132_113259


namespace NUMINAMATH_CALUDE_system_solution_l1132_113214

theorem system_solution (x y m : ℝ) : 
  (2 * x + y = 5) → 
  (x - 2 * y = m) → 
  (2 * x - 3 * y = 1) → 
  (m = 0) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1132_113214


namespace NUMINAMATH_CALUDE_max_y_over_x_on_circle_l1132_113228

theorem max_y_over_x_on_circle (z : ℂ) (x y : ℝ) :
  z = x + y * I →
  x ≠ 0 →
  Complex.abs (z - 2) = Real.sqrt 3 →
  ∃ (k : ℝ), ∀ (w : ℂ) (u v : ℝ),
    w = u + v * I →
    u ≠ 0 →
    Complex.abs (w - 2) = Real.sqrt 3 →
    |v / u| ≤ k ∧
    k = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_y_over_x_on_circle_l1132_113228
