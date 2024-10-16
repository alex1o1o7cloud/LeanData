import Mathlib

namespace NUMINAMATH_CALUDE_difference_of_numbers_l1360_136058

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 504) : 
  |x - y| = 3 := by sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l1360_136058


namespace NUMINAMATH_CALUDE_pear_sales_l1360_136063

theorem pear_sales (morning_sales afternoon_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  morning_sales + afternoon_sales = 360 →
  afternoon_sales = 240 := by
sorry

end NUMINAMATH_CALUDE_pear_sales_l1360_136063


namespace NUMINAMATH_CALUDE_max_dogs_and_fish_l1360_136037

/-- Represents the count of each animal type in the pet shop -/
structure PetShop where
  dogs : ℕ
  cats : ℕ
  bunnies : ℕ
  parrots : ℕ
  fish : ℕ

/-- Checks if the given pet shop counts satisfy the ratio constraint -/
def satisfiesRatio (shop : PetShop) : Prop :=
  7 * shop.cats = 7 * shop.dogs ∧
  8 * shop.cats = 7 * shop.bunnies ∧
  3 * shop.cats = 7 * shop.parrots ∧
  5 * shop.cats = 7 * shop.fish

/-- Checks if the total number of dogs and bunnies is 330 -/
def totalDogsAndBunnies330 (shop : PetShop) : Prop :=
  shop.dogs + shop.bunnies = 330

/-- Checks if there are at least twice as many fish as cats -/
def twiceAsManyFishAsCats (shop : PetShop) : Prop :=
  shop.fish ≥ 2 * shop.cats

/-- Theorem stating the maximum number of dogs and corresponding number of fish -/
theorem max_dogs_and_fish (shop : PetShop) 
  (h1 : satisfiesRatio shop) 
  (h2 : totalDogsAndBunnies330 shop) 
  (h3 : twiceAsManyFishAsCats shop) :
  shop.dogs ≤ 154 ∧ (shop.dogs = 154 → shop.fish = 308) :=
sorry

end NUMINAMATH_CALUDE_max_dogs_and_fish_l1360_136037


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l1360_136045

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a large rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  large : Rectangle
  small1 : Rectangle
  small2 : Rectangle
  small3 : Rectangle
  small4 : Rectangle

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: If a rectangle is divided into four smaller rectangles, and three of them have
    areas 20, 12, and 16, then the fourth rectangle has an area of 16 -/
theorem fourth_rectangle_area
  (dr : DividedRectangle)
  (h1 : area dr.small1 = 20)
  (h2 : area dr.small2 = 12)
  (h3 : area dr.small3 = 16)
  (h_sum : area dr.large = area dr.small1 + area dr.small2 + area dr.small3 + area dr.small4)
  : area dr.small4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l1360_136045


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l1360_136038

/-- Represents the cost structure and group composition at a restaurant --/
structure RestaurantBill where
  adult_meal_costs : Fin 3 → ℕ
  adult_beverage_cost : ℕ
  kid_beverage_cost : ℕ
  total_people : ℕ
  kids_count : ℕ
  adult_meal_counts : Fin 3 → ℕ
  total_beverages : ℕ

/-- Calculates the total bill for a group at the restaurant --/
def calculate_total_bill (bill : RestaurantBill) : ℕ :=
  let adult_meals_cost := (bill.adult_meal_costs 0 * bill.adult_meal_counts 0) +
                          (bill.adult_meal_costs 1 * bill.adult_meal_counts 1) +
                          (bill.adult_meal_costs 2 * bill.adult_meal_counts 2)
  let adult_beverages_cost := min (bill.total_people - bill.kids_count) bill.total_beverages * bill.adult_beverage_cost
  let kid_beverages_cost := (bill.total_beverages - min (bill.total_people - bill.kids_count) bill.total_beverages) * bill.kid_beverage_cost
  adult_meals_cost + adult_beverages_cost + kid_beverages_cost

/-- Theorem stating that the total bill for the given group is $59 --/
theorem restaurant_bill_theorem (bill : RestaurantBill)
  (h1 : bill.adult_meal_costs = ![5, 7, 9])
  (h2 : bill.adult_beverage_cost = 2)
  (h3 : bill.kid_beverage_cost = 1)
  (h4 : bill.total_people = 14)
  (h5 : bill.kids_count = 7)
  (h6 : bill.adult_meal_counts = ![4, 2, 1])
  (h7 : bill.total_beverages = 9) :
  calculate_total_bill bill = 59 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l1360_136038


namespace NUMINAMATH_CALUDE_inequality_proof_l1360_136025

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a + b + c + 3) / 4 ≥ 1 / (a + b) + 1 / (b + c) + 1 / (c + a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1360_136025


namespace NUMINAMATH_CALUDE_short_trees_to_plant_l1360_136097

theorem short_trees_to_plant (current_short_trees : ℕ) (total_short_trees_after : ℕ) 
  (h1 : current_short_trees = 112)
  (h2 : total_short_trees_after = 217) :
  total_short_trees_after - current_short_trees = 105 := by
  sorry

#check short_trees_to_plant

end NUMINAMATH_CALUDE_short_trees_to_plant_l1360_136097


namespace NUMINAMATH_CALUDE_square_of_x_minus_three_l1360_136087

theorem square_of_x_minus_three (x : ℝ) (h : x = -3) : (x - 3)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_of_x_minus_three_l1360_136087


namespace NUMINAMATH_CALUDE_factorization_equality_l1360_136080

theorem factorization_equality (x : ℝ) : 5*x*(x-2) + 9*(x-2) = (x-2)*(5*x+9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1360_136080


namespace NUMINAMATH_CALUDE_fibonacci_properties_l1360_136007

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_properties :
  (∃ f : ℕ → ℕ → ℕ, ∀ k : ℕ, ∃ p q : ℕ, p > k ∧ q > k ∧ ∃ m : ℕ, (fibonacci p * fibonacci q - 1 = m^2)) ∧
  (∃ g : ℕ → ℕ → ℕ × ℕ, ∀ k : ℕ, ∃ m n : ℕ, m > k ∧ n > k ∧ 
    (fibonacci m ∣ fibonacci n^2 + 1) ∧ (fibonacci n ∣ fibonacci m^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_properties_l1360_136007


namespace NUMINAMATH_CALUDE_negation_of_forall_inequality_l1360_136057

theorem negation_of_forall_inequality (x : ℝ) :
  ¬(∀ x > 1, x - 1 > Real.log x) ↔ ∃ x > 1, x - 1 ≤ Real.log x :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_inequality_l1360_136057


namespace NUMINAMATH_CALUDE_line_through_two_points_l1360_136011

/-- 
Given two distinct points P₁(x₁, y₁) and P₂(x₂, y₂) in the plane,
the equation (x-x₁)(y₂-y₁) = (y-y₁)(x₂-x₁) represents the line passing through these points.
-/
theorem line_through_two_points (x₁ y₁ x₂ y₂ : ℝ) (h : (x₁, y₁) ≠ (x₂, y₂)) :
  ∀ x y : ℝ, (x - x₁) * (y₂ - y₁) = (y - y₁) * (x₂ - x₁) ↔ 
  ∃ t : ℝ, x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁) :=
by sorry

end NUMINAMATH_CALUDE_line_through_two_points_l1360_136011


namespace NUMINAMATH_CALUDE_complex_modulus_l1360_136096

theorem complex_modulus (z : ℂ) : z = (1 + Complex.I) / (2 - Complex.I) → Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1360_136096


namespace NUMINAMATH_CALUDE_mike_work_hours_l1360_136065

/-- Given that Mike worked 3 hours each day for 5 days, prove that his total work hours is 15. -/
theorem mike_work_hours (hours_per_day : ℕ) (days_worked : ℕ) (total_hours : ℕ) : 
  hours_per_day = 3 → days_worked = 5 → total_hours = hours_per_day * days_worked → total_hours = 15 := by
  sorry

end NUMINAMATH_CALUDE_mike_work_hours_l1360_136065


namespace NUMINAMATH_CALUDE_tangent_line_of_odd_cubic_l1360_136062

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = x^3 + (a-1)x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a-1)*x^2 + a*x

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*(a-1)*x + a

/-- The tangent line equation at (0,0) is y = mx, where m is the slope at x = 0 -/
def TangentLineAt0 (f : ℝ → ℝ) (f' : ℝ → ℝ) : ℝ → ℝ := fun x ↦ (f' 0) * x

theorem tangent_line_of_odd_cubic (a : ℝ) :
  IsOdd (f a) → TangentLineAt0 (f a) (f' a) = fun x ↦ x := by sorry

end NUMINAMATH_CALUDE_tangent_line_of_odd_cubic_l1360_136062


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_two_min_value_achieved_l1360_136092

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x + y) / (x*y) = 7/2 + Real.sqrt 6) : 
  ∀ a b : ℝ, a > 0 → b > 0 → (2*a + b) / (a*b) = 7/2 + Real.sqrt 6 → x + 3*y ≤ a + 3*b :=
sorry

theorem min_value_is_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x + y) / (x*y) = 7/2 + Real.sqrt 6) : 
  x + 3*y ≥ 2 :=
sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x + y) / (x*y) = 7/2 + Real.sqrt 6) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (2*a + b) / (a*b) = 7/2 + Real.sqrt 6 ∧ a + 3*b = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_two_min_value_achieved_l1360_136092


namespace NUMINAMATH_CALUDE_southern_tents_l1360_136014

/-- Represents the number of tents in different parts of the campsite -/
structure Campsite where
  total : ℕ
  north : ℕ
  east : ℕ
  center : ℕ
  south : ℕ

/-- Theorem stating the number of tents in the southern part of the campsite -/
theorem southern_tents (c : Campsite) 
  (h_total : c.total = 900)
  (h_north : c.north = 100)
  (h_east : c.east = 2 * c.north)
  (h_center : c.center = 4 * c.north)
  (h_sum : c.total = c.north + c.east + c.center + c.south) : 
  c.south = 200 := by
  sorry


end NUMINAMATH_CALUDE_southern_tents_l1360_136014


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1360_136072

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a with a₂ = 4 and a₆ = 6, prove that a₁₀ = 9 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_a2 : a 2 = 4) 
    (h_a6 : a 6 = 6) : 
  a 10 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1360_136072


namespace NUMINAMATH_CALUDE_complex_sum_roots_of_unity_l1360_136009

theorem complex_sum_roots_of_unity (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  (Finset.range 16).sum (λ k => ω^(20 + 4*k)) = -ω^2 := by sorry

end NUMINAMATH_CALUDE_complex_sum_roots_of_unity_l1360_136009


namespace NUMINAMATH_CALUDE_solid_is_triangular_prism_l1360_136066

/-- Represents a three-dimensional solid -/
structure Solid :=
  (front_view : Shape)
  (side_view : Shape)

/-- Represents geometric shapes -/
inductive Shape
  | Triangle
  | Quadrilateral
  | Other

/-- Defines a triangular prism -/
def is_triangular_prism (s : Solid) : Prop :=
  s.front_view = Shape.Triangle ∧ s.side_view = Shape.Quadrilateral

/-- Theorem: A solid with triangular front view and quadrilateral side view is a triangular prism -/
theorem solid_is_triangular_prism (s : Solid) 
  (h1 : s.front_view = Shape.Triangle) 
  (h2 : s.side_view = Shape.Quadrilateral) : 
  is_triangular_prism s := by
  sorry

end NUMINAMATH_CALUDE_solid_is_triangular_prism_l1360_136066


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l1360_136031

theorem quadratic_root_zero (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + a^2 - 9 = 0 → x = 0 ∨ x ≠ 0) →
  (0^2 - 2*0 + a^2 - 9 = 0) →
  (a = 3 ∨ a = -3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l1360_136031


namespace NUMINAMATH_CALUDE_overall_percentage_l1360_136018

theorem overall_percentage (grade1 grade2 grade3 : ℚ) 
  (h1 : grade1 = 50 / 100)
  (h2 : grade2 = 70 / 100)
  (h3 : grade3 = 90 / 100) :
  (grade1 + grade2 + grade3) / 3 = 70 / 100 := by
  sorry

end NUMINAMATH_CALUDE_overall_percentage_l1360_136018


namespace NUMINAMATH_CALUDE_power_zero_is_one_l1360_136036

theorem power_zero_is_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_is_one_l1360_136036


namespace NUMINAMATH_CALUDE_kara_forgotten_doses_l1360_136069

/-- The number of times Kara takes medication per day -/
def doses_per_day : ℕ := 3

/-- The amount of water in ounces Kara drinks with each dose -/
def water_per_dose : ℕ := 4

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total amount of water in ounces Kara drank with her medication over two weeks -/
def total_water_drunk : ℕ := 160

/-- The number of times Kara forgot to take her medication on one day in the second week -/
def forgotten_doses : ℕ := 2

theorem kara_forgotten_doses :
  (doses_per_day * water_per_dose * days_in_week * 2) - total_water_drunk = forgotten_doses * water_per_dose :=
by sorry

end NUMINAMATH_CALUDE_kara_forgotten_doses_l1360_136069


namespace NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_1_l1360_136027

theorem infinitely_many_primes_4k_plus_1 :
  ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4*k + 1) →
  ∃ q, Nat.Prime q ∧ (∃ m, q = 4*m + 1) ∧ q ∉ S :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_1_l1360_136027


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_range_l1360_136043

def f (a x : ℝ) : ℝ := x^2 + 2*(a-2)*x + 5

theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 4 ≤ x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) →
  a ∈ Set.Ici (-2) :=
by sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_range_l1360_136043


namespace NUMINAMATH_CALUDE_classmate_reading_comprehensive_only_classmate_reading_comprehensive_l1360_136099

/-- Represents a survey activity -/
inductive SurveyActivity
| SocketLifespan
| TreePlantingSurvival
| ClassmateReading
| DocumentaryViewership

/-- Determines if a survey activity is suitable for a comprehensive survey -/
def isSuitableForComprehensiveSurvey (activity : SurveyActivity) : Prop :=
  match activity with
  | SurveyActivity.ClassmateReading => true
  | _ => false

/-- Theorem stating that the classmate reading survey is suitable for a comprehensive survey -/
theorem classmate_reading_comprehensive :
  isSuitableForComprehensiveSurvey SurveyActivity.ClassmateReading :=
by sorry

/-- Theorem stating that the classmate reading survey is the only suitable activity for a comprehensive survey -/
theorem only_classmate_reading_comprehensive (activity : SurveyActivity) :
  isSuitableForComprehensiveSurvey activity ↔ activity = SurveyActivity.ClassmateReading :=
by sorry

end NUMINAMATH_CALUDE_classmate_reading_comprehensive_only_classmate_reading_comprehensive_l1360_136099


namespace NUMINAMATH_CALUDE_ab_value_l1360_136008

theorem ab_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b - 2| = 9) (h3 : a + b > 0) :
  ab = 33 ∨ ab = -33 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1360_136008


namespace NUMINAMATH_CALUDE_derek_average_increase_l1360_136085

def derek_scores : List ℝ := [92, 86, 89, 94, 91]

theorem derek_average_increase :
  let first_three := derek_scores.take 3
  let all_five := derek_scores
  (all_five.sum / all_five.length) - (first_three.sum / first_three.length) = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_derek_average_increase_l1360_136085


namespace NUMINAMATH_CALUDE_equation_condition_l1360_136032

theorem equation_condition (a b c d : ℝ) :
  (a^2 + b) / (b + c^2) = (c^2 + d) / (d + a^2) →
  (a = c ∨ a^2 + d + 2*b = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_condition_l1360_136032


namespace NUMINAMATH_CALUDE_initial_water_percentage_l1360_136071

theorem initial_water_percentage
  (capacity : ℝ)
  (added_water : ℝ)
  (final_fraction : ℝ)
  (h1 : capacity = 80)
  (h2 : added_water = 28)
  (h3 : final_fraction = 3/4)
  (h4 : final_fraction * capacity = (initial_percentage / 100) * capacity + added_water) :
  initial_percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l1360_136071


namespace NUMINAMATH_CALUDE_john_annual_maintenance_expenses_l1360_136047

/-- Represents John's annual car maintenance expenses --/
def annual_maintenance_expenses (
  annual_mileage : ℕ)
  (oil_change_interval : ℕ)
  (free_oil_changes : ℕ)
  (oil_change_cost : ℕ)
  (tire_rotation_interval : ℕ)
  (tire_rotation_cost : ℕ)
  (brake_pad_interval : ℕ)
  (brake_pad_cost : ℕ) : ℕ :=
  let paid_oil_changes := annual_mileage / oil_change_interval - free_oil_changes
  let annual_oil_change_cost := paid_oil_changes * oil_change_cost
  let annual_tire_rotation_cost := (annual_mileage / tire_rotation_interval) * tire_rotation_cost
  let annual_brake_pad_cost := (annual_mileage * brake_pad_cost) / brake_pad_interval
  annual_oil_change_cost + annual_tire_rotation_cost + annual_brake_pad_cost

/-- Theorem stating John's annual maintenance expenses --/
theorem john_annual_maintenance_expenses :
  annual_maintenance_expenses 12000 3000 1 50 6000 40 24000 200 = 330 := by
  sorry

end NUMINAMATH_CALUDE_john_annual_maintenance_expenses_l1360_136047


namespace NUMINAMATH_CALUDE_f_geq_half_iff_in_range_triangle_side_a_l1360_136012

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ :=
  let m_vec := m x
  let n_vec := n x
  (m_vec.1 * (1/2 * m_vec.1 - n_vec.1) + m_vec.2 * (1/2 * m_vec.2 - n_vec.2))

def in_range (x : ℝ) : Prop :=
  ∃ k : ℤ, x ∈ Set.Icc (k * Real.pi - Real.pi / 2) (k * Real.pi + Real.pi / 6)

theorem f_geq_half_iff_in_range (x : ℝ) :
  f x ≥ 1/2 ↔ in_range x := by sorry

theorem triangle_side_a (A B C : ℝ) (a b c : ℝ) :
  f (B / 2) = 1 → b = 1 → c = Real.sqrt 3 → a = 1 := by sorry

end NUMINAMATH_CALUDE_f_geq_half_iff_in_range_triangle_side_a_l1360_136012


namespace NUMINAMATH_CALUDE_min_value_of_f_l1360_136078

/-- The function f(x) with parameter p -/
def f (p : ℝ) (x : ℝ) : ℝ := x^2 - 2*p*x + 2*p^2 + 2*p - 1

/-- Theorem stating that the minimum value of f(x) is -2 for any real p -/
theorem min_value_of_f (p : ℝ) : 
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f p x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1360_136078


namespace NUMINAMATH_CALUDE_infinitely_many_primes_with_quadratic_nonresidue_l1360_136017

theorem infinitely_many_primes_with_quadratic_nonresidue (a : ℤ) 
  (h_odd : Odd a) (h_not_square : ∀ n : ℤ, n ^ 2 ≠ a) :
  ∃ (S : Set ℕ), (∀ p ∈ S, Prime p) ∧ 
  Set.Infinite S ∧ 
  (∀ p ∈ S, ¬ ∃ x : ℤ, x ^ 2 ≡ a [ZMOD p]) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_with_quadratic_nonresidue_l1360_136017


namespace NUMINAMATH_CALUDE_inequality_theorem_l1360_136082

theorem inequality_theorem (m n : ℕ) (h : m > n) :
  (1 + 1 / m : ℝ) ^ m > (1 + 1 / n : ℝ) ^ n ∧
  (1 + 1 / m : ℝ) ^ (m + 1) < (1 + 1 / n : ℝ) ^ (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1360_136082


namespace NUMINAMATH_CALUDE_sum_and_round_equals_130_l1360_136030

def round_to_nearest_ten (n : Int) : Int :=
  (n + 5) / 10 * 10

theorem sum_and_round_equals_130 : round_to_nearest_ten (68 + 57) = 130 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_round_equals_130_l1360_136030


namespace NUMINAMATH_CALUDE_emerald_woods_circuit_length_l1360_136044

/-- Proves that the total length of the Emerald Woods Circuit is 43 miles given the hiking conditions --/
theorem emerald_woods_circuit_length :
  ∀ (a b c d e : ℝ),
    a + b + c = 28 →
    c + d = 24 →
    c + d + e = 39 →
    a + d = 30 →
    a + b + c + d + e = 43 := by
  sorry

end NUMINAMATH_CALUDE_emerald_woods_circuit_length_l1360_136044


namespace NUMINAMATH_CALUDE_max_chocolates_buyable_l1360_136076

def total_money : ℚ := 24.50
def chocolate_price : ℚ := 2.20

theorem max_chocolates_buyable : 
  ⌊total_money / chocolate_price⌋ = 11 := by sorry

end NUMINAMATH_CALUDE_max_chocolates_buyable_l1360_136076


namespace NUMINAMATH_CALUDE_cyclic_power_inequality_l1360_136075

theorem cyclic_power_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^4*b + b^4*c + c^4*d + d^4*a ≥ a*b*c*d*(a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_power_inequality_l1360_136075


namespace NUMINAMATH_CALUDE_five_Z_three_equals_twelve_l1360_136052

-- Define the Z operation
def Z (a b : ℝ) : ℝ := 3 * (a - b)^2

-- Theorem statement
theorem five_Z_three_equals_twelve : Z 5 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_five_Z_three_equals_twelve_l1360_136052


namespace NUMINAMATH_CALUDE_barrys_age_l1360_136064

theorem barrys_age (sisters_average_age : ℕ) (total_average_age : ℕ) : 
  sisters_average_age = 27 → total_average_age = 28 → 
  (3 * sisters_average_age + 31) / 4 = total_average_age :=
by
  sorry

#check barrys_age

end NUMINAMATH_CALUDE_barrys_age_l1360_136064


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1360_136040

theorem min_value_sum_of_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  1 / (a^2 + 2*b^2) + 1 / (b^2 + 2*c^2) + 1 / (c^2 + 2*a^2) ≥ 9 ∧
  (1 / (a^2 + 2*b^2) + 1 / (b^2 + 2*c^2) + 1 / (c^2 + 2*a^2) = 9 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1360_136040


namespace NUMINAMATH_CALUDE_triangle_side_sum_max_l1360_136093

theorem triangle_side_sum_max (a c : ℝ) : 
  let B : ℝ := π / 3
  let b : ℝ := 2
  0 < a ∧ 0 < c ∧ 
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  a + c ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_max_l1360_136093


namespace NUMINAMATH_CALUDE_jan_skips_proof_l1360_136019

def initial_speed : ℕ := 70
def time_period : ℕ := 5

theorem jan_skips_proof (doubled_speed : ℕ) (total_skips : ℕ) 
  (h1 : doubled_speed = 2 * initial_speed) 
  (h2 : total_skips = doubled_speed * time_period) : 
  total_skips = 700 := by sorry

end NUMINAMATH_CALUDE_jan_skips_proof_l1360_136019


namespace NUMINAMATH_CALUDE_valid_p_values_l1360_136059

def is_valid_p (p : ℤ) : Prop :=
  ∃ (k : ℤ), k > 0 ∧ (4 * p + 20) = k * (3 * p - 6)

theorem valid_p_values :
  {p : ℤ | is_valid_p p} = {3, 4, 15, 28} :=
by sorry

end NUMINAMATH_CALUDE_valid_p_values_l1360_136059


namespace NUMINAMATH_CALUDE_smallest_third_term_geometric_prog_l1360_136042

/-- 
Given an arithmetic progression with first term 9, if adding 5 to the second term 
and 25 to the third term results in a geometric progression, then the smallest 
possible value for the third term of the resulting geometric progression is -3.
-/
theorem smallest_third_term_geometric_prog (d : ℝ) : 
  let a₁ := 9
  let a₂ := 9 + d
  let a₃ := 9 + 2*d
  let g₁ := a₁
  let g₂ := a₂ + 5
  let g₃ := a₃ + 25
  (g₂^2 = g₁ * g₃) →  -- Condition for geometric progression
  g₃ ≥ -3 ∧ (∃ d : ℝ, g₃ = -3) := by
sorry

end NUMINAMATH_CALUDE_smallest_third_term_geometric_prog_l1360_136042


namespace NUMINAMATH_CALUDE_wheel_rotation_l1360_136056

/-- Proves that a wheel with given radius and arc length rotates by the calculated number of radians -/
theorem wheel_rotation (radius : ℝ) (arc_length : ℝ) (rotation : ℝ) 
  (h1 : radius = 20)
  (h2 : arc_length = 40)
  (h3 : rotation = arc_length / radius)
  (h4 : rotation > 0) : -- represents counterclockwise rotation
  rotation = 2 := by
  sorry

end NUMINAMATH_CALUDE_wheel_rotation_l1360_136056


namespace NUMINAMATH_CALUDE_power_product_eight_l1360_136016

theorem power_product_eight (a b : ℕ+) (h : (2 ^ a.val) ^ b.val = 2 ^ 2) :
  2 ^ a.val * 2 ^ b.val = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_product_eight_l1360_136016


namespace NUMINAMATH_CALUDE_lucas_units_digit_l1360_136060

-- Define Lucas numbers
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

-- Theorem statement
theorem lucas_units_digit :
  unitsDigit (lucas (lucas 9)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_lucas_units_digit_l1360_136060


namespace NUMINAMATH_CALUDE_last_nonzero_digit_of_b_d_is_five_l1360_136022

/-- Definition of b_n -/
def b (n : ℕ+) : ℕ := 2 * (Nat.factorial (n + 10) / Nat.factorial (n + 2))

/-- The last nonzero digit of a natural number -/
def lastNonzeroDigit (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- The smallest positive integer d such that the last nonzero digit of b(d) is odd -/
def d : ℕ+ := sorry

theorem last_nonzero_digit_of_b_d_is_five :
  lastNonzeroDigit (b d) = 5 := by sorry

end NUMINAMATH_CALUDE_last_nonzero_digit_of_b_d_is_five_l1360_136022


namespace NUMINAMATH_CALUDE_binomial_150_150_l1360_136070

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_l1360_136070


namespace NUMINAMATH_CALUDE_inequality_preservation_l1360_136067

theorem inequality_preservation (x y z : ℝ) (k : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x * y * z = 1) (h5 : 1/x + 1/y + 1/z ≥ x + y + z) :
  1/(x^k) + 1/(y^k) + 1/(z^k) ≥ x^k + y^k + z^k := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1360_136067


namespace NUMINAMATH_CALUDE_function_relation_characterization_l1360_136084

theorem function_relation_characterization 
  (f g : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f m - f n = (m - n) * (g m + g n)) :
  ∃ a b c : ℕ, 
    (∀ n : ℕ, f n = a * n^2 + 2 * b * n + c) ∧ 
    (∀ n : ℕ, g n = a * n + b) :=
by sorry

end NUMINAMATH_CALUDE_function_relation_characterization_l1360_136084


namespace NUMINAMATH_CALUDE_unique_prime_solution_l1360_136024

theorem unique_prime_solution : 
  ∃! (p m : ℕ), 
    Prime p ∧ 
    m > 0 ∧ 
    p^3 + m*(p + 2) = m^2 + p + 1 ∧ 
    p = 2 ∧ 
    m = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l1360_136024


namespace NUMINAMATH_CALUDE_candy_sampling_percentage_l1360_136034

/-- The percentage of customers caught sampling candy -/
def caught_percent : ℝ := 22

/-- The percentage of candy samplers who are not caught -/
def not_caught_percent : ℝ := 20

/-- The total percentage of customers who sample candy -/
def total_sample_percent : ℝ := 28

/-- Theorem stating that the total percentage of customers who sample candy is 28% -/
theorem candy_sampling_percentage :
  total_sample_percent = caught_percent / (1 - not_caught_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_candy_sampling_percentage_l1360_136034


namespace NUMINAMATH_CALUDE_cost_AB_flight_l1360_136049

-- Define the distances
def distance_AC : ℝ := 3000
def distance_AB : ℝ := 3250

-- Define the cost structure
def bus_cost_per_km : ℝ := 0.15
def plane_cost_per_km : ℝ := 0.10
def plane_booking_fee : ℝ := 100

-- Define the function to calculate flight cost
def flight_cost (distance : ℝ) : ℝ :=
  distance * plane_cost_per_km + plane_booking_fee

-- Theorem to prove
theorem cost_AB_flight : flight_cost distance_AB = 425 := by
  sorry

end NUMINAMATH_CALUDE_cost_AB_flight_l1360_136049


namespace NUMINAMATH_CALUDE_sum_from_interest_and_discount_l1360_136006

/-- Given a sum, rate, and time, if the simple interest is 88 and the true discount is 80, then the sum is 880. -/
theorem sum_from_interest_and_discount (P r t : ℝ) 
  (h1 : P * r * t / 100 = 88)
  (h2 : P * r * t / (100 + r * t) = 80) : 
  P = 880 := by
  sorry

#check sum_from_interest_and_discount

end NUMINAMATH_CALUDE_sum_from_interest_and_discount_l1360_136006


namespace NUMINAMATH_CALUDE_total_players_l1360_136021

theorem total_players (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ) 
  (h1 : kabadi = 10) 
  (h2 : kho_kho_only = 20) 
  (h3 : both = 5) : 
  kabadi + kho_kho_only - both = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_players_l1360_136021


namespace NUMINAMATH_CALUDE_angle_with_special_supplement_complement_l1360_136046

theorem angle_with_special_supplement_complement : ∃ (x : ℝ), 
  (0 < x) ∧ (x < 180) ∧ (180 - x = 4 * (90 - x)) ∧ (x = 60) := by
  sorry

end NUMINAMATH_CALUDE_angle_with_special_supplement_complement_l1360_136046


namespace NUMINAMATH_CALUDE_problem_solution_l1360_136091

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = 144^(1/5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1360_136091


namespace NUMINAMATH_CALUDE_square_inequality_l1360_136003

theorem square_inequality (α : ℝ) (x : ℝ) (h1 : α ≥ 0) (h2 : (x + 1)^2 ≥ α * (α + 1)) :
  x^2 ≥ α * (α - 1) := by
sorry

end NUMINAMATH_CALUDE_square_inequality_l1360_136003


namespace NUMINAMATH_CALUDE_parabola_directrix_l1360_136086

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the points
def origin : ℝ × ℝ := (0, 0)
def point_D : ℝ × ℝ := (1, 2)

-- Define the perpendicularity condition
def perpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  (p2.1 - p1.1) * (p4.1 - p3.1) + (p2.2 - p1.2) * (p4.2 - p3.2) = 0

-- State the theorem
theorem parabola_directrix (p : ℝ) (A B : ℝ × ℝ) :
  parabola p A.1 A.2 ∧ 
  parabola p B.1 B.2 ∧ 
  perpendicular origin A origin B ∧
  perpendicular origin point_D A B →
  ∃ (x : ℝ), x = -5/4 ∧ ∀ (y : ℝ), parabola p x y → x = -p/2 :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1360_136086


namespace NUMINAMATH_CALUDE_trees_planted_today_is_41_l1360_136015

-- Define the initial number of trees
def initial_trees : Nat := 39

-- Define the number of trees to be planted tomorrow
def trees_tomorrow : Nat := 20

-- Define the final number of trees
def final_trees : Nat := 100

-- Define the number of trees planted today
def trees_planted_today : Nat := final_trees - initial_trees - trees_tomorrow

-- Theorem to prove
theorem trees_planted_today_is_41 : trees_planted_today = 41 := by
  sorry

end NUMINAMATH_CALUDE_trees_planted_today_is_41_l1360_136015


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1360_136048

theorem inequality_equivalence (x y : ℝ) : y - x^2 < |x| ↔ y < x^2 + |x| := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1360_136048


namespace NUMINAMATH_CALUDE_cinema_ticket_cost_l1360_136077

/-- Given Samuel and Kevin's cinema outing expenses, prove their combined ticket cost --/
theorem cinema_ticket_cost (total_budget : ℕ) 
  (samuel_food_drink : ℕ) (kevin_drink : ℕ) (kevin_food : ℕ) 
  (h1 : total_budget = 20)
  (h2 : samuel_food_drink = 6)
  (h3 : kevin_drink = 2)
  (h4 : kevin_food = 4) :
  ∃ (samuel_ticket kevin_ticket : ℕ),
    samuel_ticket + kevin_ticket = total_budget - (samuel_food_drink + kevin_drink + kevin_food) :=
by sorry

end NUMINAMATH_CALUDE_cinema_ticket_cost_l1360_136077


namespace NUMINAMATH_CALUDE_triangle_area_fraction_l1360_136020

-- Define the grid dimensions
def grid_width : ℕ := 8
def grid_height : ℕ := 6

-- Define the triangle vertices
def point_A : ℚ × ℚ := (2, 5)
def point_B : ℚ × ℚ := (7, 2)
def point_C : ℚ × ℚ := (6, 6)

-- Function to calculate the area of a triangle using the Shoelace formula
def triangle_area (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

-- Theorem statement
theorem triangle_area_fraction :
  (triangle_area point_A point_B point_C) / (grid_width * grid_height : ℚ) = 17/96 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_fraction_l1360_136020


namespace NUMINAMATH_CALUDE_equation_solution_l1360_136095

theorem equation_solution :
  let f (x : ℝ) := (x^3 - x^2 - 4*x) / (x^2 + 5*x + 6) + x
  ∀ x : ℝ, f x = -4 ↔ x = (3 + Real.sqrt 105) / 4 ∨ x = (3 - Real.sqrt 105) / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1360_136095


namespace NUMINAMATH_CALUDE_nesbitts_inequality_l1360_136098

theorem nesbitts_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_nesbitts_inequality_l1360_136098


namespace NUMINAMATH_CALUDE_max_value_cosine_sine_fraction_l1360_136051

theorem max_value_cosine_sine_fraction :
  ∀ x : ℝ, (1 + Real.cos x) / (Real.sin x + Real.cos x + 2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cosine_sine_fraction_l1360_136051


namespace NUMINAMATH_CALUDE_appliance_cost_theorem_l1360_136029

/-- Represents the cost of appliances in Joseph's house -/
structure ApplianceCosts where
  waterHeater : ℝ
  refrigerator : ℝ
  electricOven : ℝ
  airConditioner : ℝ
  washingMachine : ℝ

/-- Calculates the total cost of all appliances -/
def totalCost (costs : ApplianceCosts) : ℝ :=
  costs.waterHeater + costs.refrigerator + costs.electricOven + costs.airConditioner + costs.washingMachine

/-- Theorem stating the total cost of appliances is $1900 -/
theorem appliance_cost_theorem (costs : ApplianceCosts) : 
  costs.refrigerator = 3 * costs.waterHeater →
  costs.electricOven = 500 →
  costs.electricOven = 2 * costs.waterHeater →
  costs.airConditioner = 300 →
  costs.airConditioner = costs.refrigerator / 2 →
  costs.washingMachine = 100 →
  costs.washingMachine = costs.waterHeater / 5 →
  totalCost costs = 1900 := by
  sorry

end NUMINAMATH_CALUDE_appliance_cost_theorem_l1360_136029


namespace NUMINAMATH_CALUDE_min_value_expression_l1360_136033

theorem min_value_expression (a b c : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 4) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (4/c - 1)^2 ≥ 12 - 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1360_136033


namespace NUMINAMATH_CALUDE_sqrt_200_equals_10_l1360_136073

theorem sqrt_200_equals_10 : Real.sqrt 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_equals_10_l1360_136073


namespace NUMINAMATH_CALUDE_average_length_of_strings_l1360_136089

/-- The average length of three pieces of string -/
def average_length (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- Theorem: The average length of three pieces of string with lengths 2 inches, 5 inches, and 3 inches is equal to 10/3 inches -/
theorem average_length_of_strings : average_length 2 5 3 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_length_of_strings_l1360_136089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1360_136039

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {aₙ}, if a₁ + a₉ = 10, then a₅ = 5 -/
theorem arithmetic_sequence_middle_term 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 1 + a 9 = 10) : 
  a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1360_136039


namespace NUMINAMATH_CALUDE_cow_heart_ratio_is_32_to_1_l1360_136028

/-- The number of hearts on a standard deck of 52 playing cards -/
def hearts_in_deck : ℕ := 13

/-- The number of cows in Devonshire -/
def cows_in_devonshire : ℕ := hearts_in_deck

/-- The cost of each cow in dollars -/
def cost_per_cow : ℕ := 200

/-- The total cost of all cows in dollars -/
def total_cost : ℕ := 83200

/-- The ratio of cows to hearts -/
def cow_heart_ratio : ℚ := cows_in_devonshire / hearts_in_deck

theorem cow_heart_ratio_is_32_to_1 :
  cow_heart_ratio = 32 / 1 := by sorry

end NUMINAMATH_CALUDE_cow_heart_ratio_is_32_to_1_l1360_136028


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l1360_136000

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l1360_136000


namespace NUMINAMATH_CALUDE_parabola_intersection_l1360_136061

theorem parabola_intersection :
  let f (x : ℝ) := 2 * x^2 + 5 * x - 3
  let g (x : ℝ) := x^2 + 8
  let x₁ := (-5 - Real.sqrt 69) / 2
  let x₂ := (-5 + Real.sqrt 69) / 2
  let y₁ := f x₁
  let y₂ := f x₂
  (∀ x, f x = g x ↔ x = x₁ ∨ x = x₂) ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1360_136061


namespace NUMINAMATH_CALUDE_alpha_plus_two_beta_eq_pi_over_four_l1360_136090

theorem alpha_plus_two_beta_eq_pi_over_four 
  (α β : Real) 
  (acute_α : 0 < α ∧ α < π / 2) 
  (acute_β : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 1 / 7) 
  (sin_β : Real.sin β = Real.sqrt 10 / 10) : 
  α + 2 * β = π / 4 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_two_beta_eq_pi_over_four_l1360_136090


namespace NUMINAMATH_CALUDE_hundreds_digit_of_8_pow_1234_l1360_136068

-- Define a function to get the last three digits of 8^n
def lastThreeDigits (n : ℕ) : ℕ := 8^n % 1000

-- Define the cycle length of the last three digits of 8^n
def cycleLengthOf8 : ℕ := 20

-- Theorem statement
theorem hundreds_digit_of_8_pow_1234 :
  (lastThreeDigits 1234) / 100 = 1 :=
sorry

end NUMINAMATH_CALUDE_hundreds_digit_of_8_pow_1234_l1360_136068


namespace NUMINAMATH_CALUDE_new_person_weight_l1360_136026

/-- The weight of a new person joining a group, given certain conditions -/
theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : 
  n = 8 → 
  avg_increase = 3.5 →
  replaced_weight = 65 →
  (n : ℝ) * avg_increase + replaced_weight = 93 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1360_136026


namespace NUMINAMATH_CALUDE_dog_kennel_problem_l1360_136005

theorem dog_kennel_problem (total : ℕ) (long_fur : ℕ) (brown : ℕ) (neither : ℕ) 
  (h1 : total = 45)
  (h2 : long_fur = 36)
  (h3 : brown = 27)
  (h4 : neither = 8)
  : total - neither - (long_fur + brown - (total - neither)) = 26 := by
  sorry

end NUMINAMATH_CALUDE_dog_kennel_problem_l1360_136005


namespace NUMINAMATH_CALUDE_seven_digit_divisibility_l1360_136074

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

theorem seven_digit_divisibility :
  ∀ (a b c d e f : ℕ),
    (is_divisible_by_8 (2300000 + a * 10000 + b * 1000 + 372) = false) ∧
    (is_divisible_by_8 (5300000 + c * 10000 + d * 1000 + 164) = false) ∧
    (is_divisible_by_8 (5000000 + e * 10000 + f * 1000 + 3416) = true) ∧
    (is_divisible_by_8 (7100000 + a * 10000 + b * 1000 + 172) = false) :=
by
  sorry

#check seven_digit_divisibility

end NUMINAMATH_CALUDE_seven_digit_divisibility_l1360_136074


namespace NUMINAMATH_CALUDE_cauliflower_earnings_l1360_136054

/-- Earnings from farmers' market --/
structure MarketEarnings where
  total : ℕ
  broccoli : ℕ
  carrots : ℕ
  spinach : ℕ
  cauliflower : ℕ

/-- Conditions for the farmers' market earnings --/
def validMarketEarnings (e : MarketEarnings) : Prop :=
  e.total = 380 ∧
  e.broccoli = 57 ∧
  e.carrots = 2 * e.broccoli ∧
  e.spinach = (e.carrots / 2) + 16 ∧
  e.total = e.broccoli + e.carrots + e.spinach + e.cauliflower

theorem cauliflower_earnings (e : MarketEarnings) (h : validMarketEarnings e) :
  e.cauliflower = 136 := by
  sorry

end NUMINAMATH_CALUDE_cauliflower_earnings_l1360_136054


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1360_136010

theorem complex_fraction_simplification :
  (2 + 2 * Complex.I) / (-3 + 4 * Complex.I) = -14/25 - 14/25 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1360_136010


namespace NUMINAMATH_CALUDE_store_a_cheaper_for_300_l1360_136088

def cost_store_a (x : ℕ) : ℝ :=
  if x ≤ 100 then 5 * x else 4 * x + 100

def cost_store_b (x : ℕ) : ℝ :=
  4.5 * x

theorem store_a_cheaper_for_300 :
  cost_store_a 300 < cost_store_b 300 :=
sorry

end NUMINAMATH_CALUDE_store_a_cheaper_for_300_l1360_136088


namespace NUMINAMATH_CALUDE_candy_bar_distribution_l1360_136004

theorem candy_bar_distribution (total_candy_bars : ℝ) (num_people : ℝ) 
  (h1 : total_candy_bars = 5.0) 
  (h2 : num_people = 3.0) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |total_candy_bars / num_people - 1.67| < ε :=
sorry

end NUMINAMATH_CALUDE_candy_bar_distribution_l1360_136004


namespace NUMINAMATH_CALUDE_gcd_13247_36874_l1360_136083

theorem gcd_13247_36874 : Nat.gcd 13247 36874 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_13247_36874_l1360_136083


namespace NUMINAMATH_CALUDE_arccos_cos_three_l1360_136002

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_three_l1360_136002


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1360_136053

theorem no_integer_solutions (n : ℤ) (s : ℕ) (h_s : Odd s) :
  ¬ ∃ x : ℤ, x^2 - 16*n*x + 7^s = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1360_136053


namespace NUMINAMATH_CALUDE_scenario_proof_l1360_136081

theorem scenario_proof (a b c d : ℝ) 
  (h1 : a * b * c * d > 0) 
  (h2 : a < c) 
  (h3 : b * c * d < 0) : 
  a < 0 ∧ b > 0 ∧ c < 0 ∧ d > 0 :=
by sorry

end NUMINAMATH_CALUDE_scenario_proof_l1360_136081


namespace NUMINAMATH_CALUDE_brownies_problem_l1360_136035

theorem brownies_problem (total_brownies : ℕ) (tina_per_day : ℕ) (husband_per_day : ℕ) 
  (shared : ℕ) (left : ℕ) :
  total_brownies = 24 →
  tina_per_day = 2 →
  husband_per_day = 1 →
  shared = 4 →
  left = 5 →
  ∃ (days : ℕ), days = 5 ∧ 
    total_brownies = days * (tina_per_day + husband_per_day) + shared + left :=
by sorry

end NUMINAMATH_CALUDE_brownies_problem_l1360_136035


namespace NUMINAMATH_CALUDE_sum_of_powers_divisibility_l1360_136094

theorem sum_of_powers_divisibility 
  (a₁ a₂ a₃ a₄ : ℤ) 
  (h : a₁^3 + a₂^3 + a₃^3 + a₄^3 = 0) :
  ∀ k : ℕ, k % 2 = 1 → (6 : ℤ) ∣ (a₁^k + a₂^k + a₃^k + a₄^k) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_divisibility_l1360_136094


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1360_136001

theorem solution_set_of_inequality (x : ℝ) :
  (2 / (x - 3) ≤ 5) ↔ (x < 3 ∨ x ≥ 17/5) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1360_136001


namespace NUMINAMATH_CALUDE_loose_coins_amount_l1360_136050

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def bills_given : ℕ := 40
def change_received : ℕ := 10

theorem loose_coins_amount : 
  (flour_cost + cake_stand_cost + change_received) - bills_given = 3 := by
  sorry

end NUMINAMATH_CALUDE_loose_coins_amount_l1360_136050


namespace NUMINAMATH_CALUDE_calendar_problem_l1360_136041

theorem calendar_problem (x : ℕ) : 
  let date_C := x
  let date_A := x + 1
  let date_B := x + 13
  let date_P := x + 14
  date_C + date_P = date_A + date_B := by sorry

end NUMINAMATH_CALUDE_calendar_problem_l1360_136041


namespace NUMINAMATH_CALUDE_lemonade_cost_calculation_l1360_136055

/-- The cost of lemonade purchased by Coach Mike -/
def lemonade_cost : ℕ := sorry

/-- The amount Coach Mike gave to the girls -/
def amount_given : ℕ := 75

/-- The change Coach Mike received -/
def change_received : ℕ := 17

/-- Theorem stating that the lemonade cost is equal to the amount given minus the change received -/
theorem lemonade_cost_calculation : 
  lemonade_cost = amount_given - change_received := by sorry

end NUMINAMATH_CALUDE_lemonade_cost_calculation_l1360_136055


namespace NUMINAMATH_CALUDE_arithmetic_mistakes_calculation_difference_l1360_136079

theorem arithmetic_mistakes (x : ℤ) : 
  ((-1 - 8) * 2 - x = -24) → (x = 6) :=
by sorry

theorem calculation_difference : 
  ((-1 - 8) + 2 - 5) - ((-1 - 8) * 2 - 5) = 11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mistakes_calculation_difference_l1360_136079


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1360_136013

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, (3 * x^2 + 8 = 4 * x - 7) ↔ (x = a + b * I ∨ x = a - b * I)) →
  a + b^2 = 47/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1360_136013


namespace NUMINAMATH_CALUDE_jons_textbooks_weight_l1360_136023

theorem jons_textbooks_weight (brandon_weight : ℝ) (jon_weight : ℝ) : 
  brandon_weight = 8 → jon_weight = 3 * brandon_weight → jon_weight = 24 := by
  sorry

end NUMINAMATH_CALUDE_jons_textbooks_weight_l1360_136023
