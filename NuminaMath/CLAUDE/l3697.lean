import Mathlib

namespace NUMINAMATH_CALUDE_sin_shift_minimum_value_l3697_369750

open Real

theorem sin_shift_minimum_value (a : ℝ) :
  (a > 0) →
  (∀ x, sin (2 * x - π / 3) = sin (2 * (x - a))) →
  a = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_sin_shift_minimum_value_l3697_369750


namespace NUMINAMATH_CALUDE_choose_three_from_eight_l3697_369778

theorem choose_three_from_eight : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_eight_l3697_369778


namespace NUMINAMATH_CALUDE_evaluate_nested_expression_l3697_369766

def f (x : ℕ) : ℕ := 3 * (3 * (3 * (3 * (3 * x + 2) + 2) + 2) + 2) + 2

theorem evaluate_nested_expression :
  f 5 = 1457 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_nested_expression_l3697_369766


namespace NUMINAMATH_CALUDE_country_club_monthly_cost_l3697_369762

/-- Calculates the monthly cost per person for a country club membership --/
def monthly_cost_per_person (
  num_people : ℕ
  ) (initial_fee_per_person : ℚ
  ) (john_payment : ℚ
  ) : ℚ :=
  let total_cost := 2 * john_payment
  let total_initial_fee := num_people * initial_fee_per_person
  let total_monthly_cost := total_cost - total_initial_fee
  let yearly_cost_per_person := total_monthly_cost / num_people
  yearly_cost_per_person / 12

theorem country_club_monthly_cost :
  monthly_cost_per_person 4 4000 32000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_country_club_monthly_cost_l3697_369762


namespace NUMINAMATH_CALUDE_correct_position_probability_l3697_369789

/-- The number of books -/
def n : ℕ := 9

/-- The number of books to be in the correct position -/
def k : ℕ := 6

/-- The probability of exactly k books being in their correct position when n books are randomly rearranged -/
def probability (n k : ℕ) : ℚ := sorry

theorem correct_position_probability : probability n k = 1 / 2160 := by sorry

end NUMINAMATH_CALUDE_correct_position_probability_l3697_369789


namespace NUMINAMATH_CALUDE_intersection_line_circle_l3697_369714

/-- Given a line ax + y - 2 = 0 intersecting a circle (x-1)² + (y-a)² = 4 at points A and B,
    where AB is the diameter of the circle, prove that a = 1. -/
theorem intersection_line_circle (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + A.2 - 2 = 0) ∧ 
    ((A.1 - 1)^2 + (A.2 - a)^2 = 4) ∧
    (a * B.1 + B.2 - 2 = 0) ∧ 
    ((B.1 - 1)^2 + (B.2 - a)^2 = 4) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l3697_369714


namespace NUMINAMATH_CALUDE_shopping_problem_l3697_369710

/-- Shopping problem -/
theorem shopping_problem (initial_amount : ℝ) (baguette_cost : ℝ) (water_cost : ℝ)
  (chocolate_cost : ℝ) (milk_cost : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) :
  initial_amount = 50 →
  baguette_cost = 2 →
  water_cost = 1 →
  chocolate_cost = 1.5 →
  milk_cost = 3.5 →
  discount_rate = 0.1 →
  tax_rate = 0.07 →
  let baguette_total := 2 * baguette_cost
  let water_total := 2 * water_cost
  let chocolate_total := 2 * chocolate_cost
  let milk_total := milk_cost * (1 - discount_rate)
  let subtotal := baguette_total + water_total + chocolate_total + milk_total
  let tax := chocolate_total * tax_rate
  let total_cost := subtotal + tax
  initial_amount - total_cost = 37.64 := by
  sorry

end NUMINAMATH_CALUDE_shopping_problem_l3697_369710


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3697_369736

/-- The minimum distance between points on y = x^2 + 1 and y = √(x - 1) -/
theorem min_distance_between_curves : 
  let P : ℝ × ℝ → Prop := λ p => ∃ x : ℝ, x ≥ 0 ∧ p = (x, x^2 + 1)
  let Q : ℝ × ℝ → Prop := λ q => ∃ y : ℝ, y ≥ 1 ∧ q = (y, Real.sqrt (y - 1))
  ∀ p q : ℝ × ℝ, P p → Q q → 
    ∃ p' q' : ℝ × ℝ, P p' ∧ Q q' ∧ 
      Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) = 3 * Real.sqrt 2 / 4 ∧
      ∀ p'' q'' : ℝ × ℝ, P p'' → Q q'' → 
        Real.sqrt ((p''.1 - q''.1)^2 + (p''.2 - q''.2)^2) ≥ 3 * Real.sqrt 2 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_between_curves_l3697_369736


namespace NUMINAMATH_CALUDE_max_grain_mass_on_platform_l3697_369769

/-- Represents a rectangular platform --/
structure Platform where
  length : ℝ
  width : ℝ

/-- Represents the properties of grain --/
structure Grain where
  density : ℝ
  max_angle : ℝ

/-- Calculates the maximum mass of grain that can be loaded onto a platform --/
def max_grain_mass (p : Platform) (g : Grain) : ℝ :=
  sorry

/-- Theorem stating the maximum mass of grain on the given platform --/
theorem max_grain_mass_on_platform :
  let p : Platform := { length := 10, width := 5 }
  let g : Grain := { density := 1200, max_angle := 45 }
  max_grain_mass p g = 175000 := by
  sorry

end NUMINAMATH_CALUDE_max_grain_mass_on_platform_l3697_369769


namespace NUMINAMATH_CALUDE_strawberry_harvest_l3697_369745

/-- Calculates the total number of strawberries harvested from a rectangular garden -/
theorem strawberry_harvest (length width : ℕ) (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) :
  length = 10 →
  width = 7 →
  plants_per_sqft = 3 →
  strawberries_per_plant = 12 →
  length * width * plants_per_sqft * strawberries_per_plant = 2520 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_harvest_l3697_369745


namespace NUMINAMATH_CALUDE_distance_to_complex_point_l3697_369706

theorem distance_to_complex_point : ∃ (z : ℂ), z = 3 / (2 - Complex.I)^2 ∧ Complex.abs z = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_complex_point_l3697_369706


namespace NUMINAMATH_CALUDE_fraction_equality_l3697_369715

theorem fraction_equality : (2 * 0.24) / (20 * 2.4) = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3697_369715


namespace NUMINAMATH_CALUDE_winnie_balloons_l3697_369717

theorem winnie_balloons (red white green chartreuse : ℕ) 
  (h1 : red = 17) 
  (h2 : white = 33) 
  (h3 : green = 65) 
  (h4 : chartreuse = 83) 
  (friends : ℕ) 
  (h5 : friends = 8) : 
  (red + white + green + chartreuse) % friends = 6 := by
sorry

end NUMINAMATH_CALUDE_winnie_balloons_l3697_369717


namespace NUMINAMATH_CALUDE_or_not_implies_q_l3697_369727

theorem or_not_implies_q (p q : Prop) : (p ∨ q) → ¬p → q := by
  sorry

end NUMINAMATH_CALUDE_or_not_implies_q_l3697_369727


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l3697_369783

theorem circle_diameter_ratio (D C : Real) (h1 : D = 20) 
  (h2 : C > 0) (h3 : C < D) 
  (h4 : (π * D^2 / 4 - π * C^2 / 4) / (π * C^2 / 4) = 4) : 
  C = 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l3697_369783


namespace NUMINAMATH_CALUDE_school_trip_theorem_l3697_369777

/-- The number of school buses -/
def num_buses : ℕ := 95

/-- The number of seats on each bus -/
def seats_per_bus : ℕ := 118

/-- All buses are fully filled -/
axiom buses_full : True

/-- The total number of students in the school -/
def total_students : ℕ := num_buses * seats_per_bus

theorem school_trip_theorem : total_students = 11210 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_theorem_l3697_369777


namespace NUMINAMATH_CALUDE_complex_power_six_l3697_369738

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Statement: (1 + i)^6 = -8i -/
theorem complex_power_six : (1 + i)^6 = -8 * i := by sorry

end NUMINAMATH_CALUDE_complex_power_six_l3697_369738


namespace NUMINAMATH_CALUDE_fuel_cost_solution_l3697_369776

/-- Represents the fuel cost calculation problem --/
def fuel_cost_problem (truck_capacity : ℝ) (car_capacity : ℝ) (hybrid_capacity : ℝ)
  (truck_fullness : ℝ) (car_fullness : ℝ) (hybrid_fullness : ℝ)
  (diesel_price : ℝ) (gas_price : ℝ)
  (diesel_discount : ℝ) (gas_discount : ℝ) : Prop :=
  let truck_to_fill := truck_capacity * (1 - truck_fullness)
  let car_to_fill := car_capacity * (1 - car_fullness)
  let hybrid_to_fill := hybrid_capacity * (1 - hybrid_fullness)
  let diesel_discounted := diesel_price - diesel_discount
  let gas_discounted := gas_price - gas_discount
  let total_cost := truck_to_fill * diesel_discounted +
                    car_to_fill * gas_discounted +
                    hybrid_to_fill * gas_discounted
  total_cost = 95.88

/-- The main theorem stating the solution to the fuel cost problem --/
theorem fuel_cost_solution :
  fuel_cost_problem 25 15 10 0.5 (1/3) 0.25 3.5 3.2 0.1 0.15 :=
by sorry

end NUMINAMATH_CALUDE_fuel_cost_solution_l3697_369776


namespace NUMINAMATH_CALUDE_flag_arrangement_congruence_l3697_369747

/-- Number of blue flags -/
def blue_flags : ℕ := 10

/-- Number of green flags -/
def green_flags : ℕ := 10

/-- Total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- 
  N is the number of distinguishable arrangements of flags on two distinguishable flagpoles,
  where each flagpole has at least one green flag and no two green flags on either pole are adjacent
-/
def N : ℕ := sorry

theorem flag_arrangement_congruence : N ≡ 77 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_flag_arrangement_congruence_l3697_369747


namespace NUMINAMATH_CALUDE_line_up_five_people_two_youngest_not_first_l3697_369720

/-- The number of ways to arrange 5 people in a line with restrictions -/
def lineUpWays (n : ℕ) (y : ℕ) (f : ℕ) : ℕ :=
  (n - y) * (n - 1) * (n - 2) * (n - 3) * (n - 4)

/-- Theorem: There are 72 ways for 5 people to line up when 2 youngest can't be first -/
theorem line_up_five_people_two_youngest_not_first :
  lineUpWays 5 2 1 = 72 := by sorry

end NUMINAMATH_CALUDE_line_up_five_people_two_youngest_not_first_l3697_369720


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3697_369775

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (c : ℝ), c^2 = 5 * a^2 ∧ 
   ∃ (S : ℝ), S = 20 ∧ S = (1/2) * c * (4 * c)) →
  a^2 = 2 ∧ b^2 = 8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3697_369775


namespace NUMINAMATH_CALUDE_multiply_by_213_equals_3408_l3697_369739

theorem multiply_by_213_equals_3408 (x : ℝ) : 213 * x = 3408 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_213_equals_3408_l3697_369739


namespace NUMINAMATH_CALUDE_better_fit_example_l3697_369716

/-- Represents a regression model with its RSS (Residual Sum of Squares) -/
structure RegressionModel where
  rss : ℝ

/-- Determines if one model has a better fit than another based on RSS -/
def better_fit (model1 model2 : RegressionModel) : Prop :=
  model1.rss < model2.rss

theorem better_fit_example :
  let model1 : RegressionModel := ⟨168⟩
  let model2 : RegressionModel := ⟨197⟩
  better_fit model1 model2 := by
  sorry

end NUMINAMATH_CALUDE_better_fit_example_l3697_369716


namespace NUMINAMATH_CALUDE_classroom_ratio_l3697_369740

theorem classroom_ratio (total_students : ℕ) (num_boys : ℕ) (h1 : total_students > 0) (h2 : num_boys ≤ total_students) :
  let prob_boy := num_boys / total_students
  let prob_girl := (total_students - num_boys) / total_students
  (prob_boy / prob_girl = 3 / 4) → (num_boys / total_students = 3 / 7) := by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_l3697_369740


namespace NUMINAMATH_CALUDE_triangle_inequality_check_l3697_369733

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem stating that among the given sets, only {17, 17, 25} can form a triangle -/
theorem triangle_inequality_check : 
  ¬(can_form_triangle 3 4 8) ∧ 
  ¬(can_form_triangle 5 6 11) ∧ 
  ¬(can_form_triangle 6 8 16) ∧ 
  can_form_triangle 17 17 25 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_check_l3697_369733


namespace NUMINAMATH_CALUDE_customer_difference_l3697_369743

theorem customer_difference (initial : Nat) (remaining : Nat) : 
  initial = 11 → remaining = 3 → (initial - remaining) - remaining = 5 := by
  sorry

end NUMINAMATH_CALUDE_customer_difference_l3697_369743


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3697_369752

theorem quadratic_roots_properties (a b c x₁ x₂ : ℝ) (ha : a ≠ 0)
  (hx₁ : a * x₁^2 + b * x₁ + c = 0) (hx₂ : a * x₂^2 + b * x₂ + c = 0) :
  x₁^2 + x₂^2 = (b^2 - 2*a*c) / a^2 ∧ x₁^3 + x₂^3 = (3*a*b*c - b^3) / a^3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3697_369752


namespace NUMINAMATH_CALUDE_percentage_of_men_in_company_l3697_369735

theorem percentage_of_men_in_company 
  (total_employees : ℝ) 
  (men : ℝ) 
  (women : ℝ) 
  (h1 : men + women = total_employees)
  (h2 : men * 0.5 + women * 0.1666666666666669 = total_employees * 0.4)
  (h3 : men > 0)
  (h4 : women > 0)
  (h5 : total_employees > 0) : 
  men / total_employees = 0.7 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_men_in_company_l3697_369735


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l3697_369701

/-- Represents a hyperbola -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop

/-- The hyperbola passes through a given point -/
def passes_through (h : Hyperbola) (x y : ℝ) : Prop :=
  h.equation x y

/-- The asymptotes of the hyperbola -/
def has_asymptotes (h : Hyperbola) (f g : ℝ → ℝ) : Prop :=
  ∀ x, (h.equation x (f x) ∨ h.equation x (g x))

/-- The foci of the hyperbola -/
def foci (h : Hyperbola) : ℝ × ℝ := sorry

/-- Distance from a point to a line -/
def distance_to_line (x y : ℝ) (m b : ℝ) : ℝ := sorry

theorem hyperbola_theorem (h : Hyperbola) 
  (center_origin : h.equation 0 0)
  (asymptotes : has_asymptotes h (λ x => Real.sqrt 3 * x) (λ x => -Real.sqrt 3 * x))
  (point : passes_through h (Real.sqrt 2) (Real.sqrt 3)) :
  (∀ x y, h.equation x y ↔ x^2 - y^2/3 = 1) ∧ 
  (let (fx, fy) := foci h
   distance_to_line fx fy (Real.sqrt 3) 0 = Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l3697_369701


namespace NUMINAMATH_CALUDE_min_surface_pips_is_58_l3697_369712

/-- Represents a standard die -/
structure StandardDie :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)

/-- Represents four dice glued in a 2x2 square configuration -/
structure GluedDice :=
  (dice : Fin 4 → StandardDie)

/-- Calculates the number of pips on the surface of glued dice -/
def surface_pips (gd : GluedDice) : ℕ :=
  sorry

/-- The minimum number of pips on the surface of glued dice -/
def min_surface_pips : ℕ :=
  sorry

theorem min_surface_pips_is_58 : min_surface_pips = 58 :=
  sorry

end NUMINAMATH_CALUDE_min_surface_pips_is_58_l3697_369712


namespace NUMINAMATH_CALUDE_heater_purchase_comparison_l3697_369742

/-- Represents the total cost of purchasing heaters from a store -/
structure HeaterPurchase where
  aPrice : ℝ  -- Price of A type heater
  bPrice : ℝ  -- Price of B type heater
  aShipping : ℝ  -- Shipping cost for A type heater
  bShipping : ℝ  -- Shipping cost for B type heater

/-- Calculate the total cost for a given number of A type heaters -/
def totalCost (p : HeaterPurchase) (x : ℝ) : ℝ :=
  (p.aPrice + p.aShipping) * x + (p.bPrice + p.bShipping) * (100 - x)

/-- Store A's pricing -/
def storeA : HeaterPurchase :=
  { aPrice := 100, bPrice := 200, aShipping := 10, bShipping := 10 }

/-- Store B's pricing -/
def storeB : HeaterPurchase :=
  { aPrice := 120, bPrice := 190, aShipping := 0, bShipping := 12 }

theorem heater_purchase_comparison :
  (∀ x, totalCost storeA x = -100 * x + 21000) ∧
  (∀ x, totalCost storeB x = -82 * x + 20200) ∧
  (totalCost storeA 60 < totalCost storeB 60) := by
  sorry

end NUMINAMATH_CALUDE_heater_purchase_comparison_l3697_369742


namespace NUMINAMATH_CALUDE_expression_nonnegative_iff_x_in_interval_l3697_369741

/-- The expression (x-12x^2+36x^3)/(9-x^3) is nonnegative if and only if x is in the interval [0, 3). -/
theorem expression_nonnegative_iff_x_in_interval :
  ∀ x : ℝ, (x - 12 * x^2 + 36 * x^3) / (9 - x^3) ≥ 0 ↔ x ∈ Set.Icc 0 3 ∧ x ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_nonnegative_iff_x_in_interval_l3697_369741


namespace NUMINAMATH_CALUDE_average_marks_chem_math_l3697_369700

/-- Given that the total marks in physics, chemistry, and mathematics is 150 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 75. -/
theorem average_marks_chem_math (P C M : ℝ) 
  (h : P + C + M = P + 150) : (C + M) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chem_math_l3697_369700


namespace NUMINAMATH_CALUDE_unique_divisor_l3697_369719

def is_valid_divisor (d : ℕ) : Prop :=
  ∃ (sequence : Finset ℕ),
    (sequence.card = 8) ∧
    (∀ n ∈ sequence, 29 ≤ n ∧ n ≤ 119) ∧
    (∀ n ∈ sequence, n % d = 0)

theorem unique_divisor :
  ∃! d : ℕ, is_valid_divisor d ∧ d = 13 := by sorry

end NUMINAMATH_CALUDE_unique_divisor_l3697_369719


namespace NUMINAMATH_CALUDE_annie_cookies_l3697_369704

theorem annie_cookies (x : ℝ) 
  (h1 : x + 2*x + 2.8*x = 29) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_annie_cookies_l3697_369704


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3697_369774

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  C = 2 * Real.pi / 3 →
  c = 5 →
  a = Real.sqrt 5 * b * Real.sin A →
  b = 2 * Real.sqrt 15 / 3 ∧
  Real.tan (B + Real.pi / 4) = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3697_369774


namespace NUMINAMATH_CALUDE_smallest_class_size_l3697_369732

/-- Represents the number of students in a physical education class with the given arrangement. -/
def class_size (n : ℕ) : ℕ := 5 * n + 2

/-- Proves that the smallest possible class size satisfying the given conditions is 42 students. -/
theorem smallest_class_size :
  ∃ (n : ℕ), 
    (class_size n > 40) ∧ 
    (∀ m : ℕ, class_size m > 40 → m ≥ class_size n) ∧
    (class_size n = 42) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_l3697_369732


namespace NUMINAMATH_CALUDE_triangle_angle_and_vector_properties_l3697_369725

theorem triangle_angle_and_vector_properties 
  (A B C : ℝ) 
  (h_triangle : A + B + C = Real.pi)
  (m : ℝ × ℝ)
  (h_m : m = (Real.tan A + Real.tan B, Real.sqrt 3))
  (n : ℝ × ℝ)
  (h_n : n = (1, 1 - Real.tan A * Real.tan B))
  (h_perp : m.1 * n.1 + m.2 * n.2 = 0)
  (a : ℝ × ℝ)
  (h_a : a = (Real.sqrt 2 * Real.cos ((A + B) / 2), Real.sin ((A - B) / 2)))
  (h_norm_a : a.1^2 + a.2^2 = 3/2) : 
  C = Real.pi / 3 ∧ Real.tan A * Real.tan B = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_and_vector_properties_l3697_369725


namespace NUMINAMATH_CALUDE_probability_both_selected_l3697_369797

theorem probability_both_selected (prob_X prob_Y : ℚ) 
  (h1 : prob_X = 1/7) 
  (h2 : prob_Y = 2/5) : 
  prob_X * prob_Y = 2/35 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l3697_369797


namespace NUMINAMATH_CALUDE_three_Z_seven_l3697_369787

-- Define the operation Z
def Z (a b : ℝ) : ℝ := b + 5 * a - 2 * a^2

-- Theorem to prove
theorem three_Z_seven : Z 3 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_three_Z_seven_l3697_369787


namespace NUMINAMATH_CALUDE_shifted_roots_polynomial_l3697_369767

theorem shifted_roots_polynomial (r₁ r₂ : ℝ) (h_sum : r₁ + r₂ = 15) (h_prod : r₁ * r₂ = 36) :
  (X - (r₁ + 3)) * (X - (r₂ + 3)) = X^2 - 21*X + 90 :=
by sorry

end NUMINAMATH_CALUDE_shifted_roots_polynomial_l3697_369767


namespace NUMINAMATH_CALUDE_nephews_count_l3697_369748

/-- The number of nephews Alden had 10 years ago -/
def alden_nephews_10_years_ago : ℕ := 50

/-- The number of nephews Alden has now -/
def alden_nephews_now : ℕ := 2 * alden_nephews_10_years_ago

/-- The number of nephews Vihaan has now -/
def vihaan_nephews : ℕ := alden_nephews_now + 60

/-- The total number of nephews Alden and Vihaan have -/
def total_nephews : ℕ := alden_nephews_now + vihaan_nephews

theorem nephews_count : total_nephews = 260 := by
  sorry

end NUMINAMATH_CALUDE_nephews_count_l3697_369748


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3697_369773

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 5)^7 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 129 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3697_369773


namespace NUMINAMATH_CALUDE_no_multiples_of_2310_in_power_difference_form_l3697_369760

theorem no_multiples_of_2310_in_power_difference_form :
  ¬ ∃ (k i j : ℕ), 
    0 ≤ i ∧ i < j ∧ j ≤ 50 ∧ 
    k * 2310 = 2^j - 2^i ∧ 
    k > 0 :=
by sorry

end NUMINAMATH_CALUDE_no_multiples_of_2310_in_power_difference_form_l3697_369760


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_products_l3697_369792

theorem root_sum_reciprocal_products (p q r s : ℂ) : 
  (p^4 + 6*p^3 - 4*p^2 + 7*p + 3 = 0) →
  (q^4 + 6*q^3 - 4*q^2 + 7*q + 3 = 0) →
  (r^4 + 6*r^3 - 4*r^2 + 7*r + 3 = 0) →
  (s^4 + 6*s^3 - 4*s^2 + 7*s + 3 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_products_l3697_369792


namespace NUMINAMATH_CALUDE_west_8m_is_negative_8m_l3697_369799

/-- Represents the direction of movement --/
inductive Direction
  | East
  | West

/-- Represents a movement with magnitude and direction --/
structure Movement where
  magnitude : ℝ
  direction : Direction

/-- Convention for representing movement as a signed real number --/
def movementValue (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.magnitude
  | Direction.West => -m.magnitude

/-- Theorem stating that moving west 8m is equivalent to -8m --/
theorem west_8m_is_negative_8m :
  let west8m : Movement := { magnitude := 8, direction := Direction.West }
  movementValue west8m = -8 := by
  sorry

end NUMINAMATH_CALUDE_west_8m_is_negative_8m_l3697_369799


namespace NUMINAMATH_CALUDE_polygon_area_is_two_l3697_369795

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Calculates the area of a polygon given its vertices -/
noncomputable def polygonArea (vertices : List Point) : ℚ :=
  sorry

/-- The list of vertices of the polygon -/
def polygonVertices : List Point := [
  ⟨0, 0⟩, ⟨1, 0⟩, ⟨2, 1⟩, ⟨2, 0⟩, ⟨3, 0⟩, ⟨3, 1⟩,
  ⟨3, 2⟩, ⟨2, 2⟩, ⟨2, 3⟩, ⟨1, 2⟩, ⟨0, 2⟩, ⟨0, 1⟩
]

/-- The theorem stating that the area of the polygon is 2 square units -/
theorem polygon_area_is_two :
  polygonArea polygonVertices = 2 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_is_two_l3697_369795


namespace NUMINAMATH_CALUDE_sheet_width_calculation_l3697_369768

/-- The width of a rectangular sheet of paper with specific properties -/
def sheet_width : ℝ := sorry

theorem sheet_width_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  abs (sheet_width - 6.6) < ε ∧
  sheet_width * 13 * 2 = 6.5 * 11 + 100 := by sorry

end NUMINAMATH_CALUDE_sheet_width_calculation_l3697_369768


namespace NUMINAMATH_CALUDE_max_value_of_operation_l3697_369791

theorem max_value_of_operation : 
  ∃ (max : ℕ), max = 1200 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → 3 * (500 - n) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_operation_l3697_369791


namespace NUMINAMATH_CALUDE_symmetric_cubic_homogeneous_decomposition_non_negative_equivalence_l3697_369780

-- Define the symmetric polynomials g₁, g₂, g₃
def g₁ (x y z : ℝ) : ℝ := x * (x - y) * (x - z) + y * (y - x) * (y - z) + z * (z - x) * (z - y)
def g₂ (x y z : ℝ) : ℝ := (y + z) * (x - y) * (x - z) + (x + z) * (y - x) * (y - z) + (x + y) * (z - x) * (z - y)
def g₃ (x y z : ℝ) : ℝ := x * y * z

-- Define a ternary symmetric cubic homogeneous polynomial
def SymmetricCubicHomogeneous (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f x y z = f y z x ∧ f x y z = f y x z ∧ ∀ t : ℝ, f (t*x) (t*y) (t*z) = t^3 * f x y z

theorem symmetric_cubic_homogeneous_decomposition
  (f : ℝ → ℝ → ℝ → ℝ) (h : SymmetricCubicHomogeneous f) :
  ∃! (a b c : ℝ), ∀ x y z : ℝ, f x y z = a * g₁ x y z + b * g₂ x y z + c * g₃ x y z :=
sorry

theorem non_negative_equivalence
  (f : ℝ → ℝ → ℝ → ℝ) (h : SymmetricCubicHomogeneous f)
  (a b c : ℝ) (h_decomp : ∀ x y z : ℝ, f x y z = a * g₁ x y z + b * g₂ x y z + c * g₃ x y z) :
  (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → f x y z ≥ 0) ↔ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_symmetric_cubic_homogeneous_decomposition_non_negative_equivalence_l3697_369780


namespace NUMINAMATH_CALUDE_largest_value_l3697_369702

theorem largest_value : 
  ∀ (a b c d : ℤ), a = 2^3 ∧ b = -3^2 ∧ c = (-3)^2 ∧ d = (-2)^3 →
  (c ≥ a ∧ c ≥ b ∧ c ≥ d) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l3697_369702


namespace NUMINAMATH_CALUDE_min_button_presses_correct_l3697_369703

/-- Represents the time difference in minutes between the correct time and the displayed time -/
def time_difference : ℤ := 13

/-- Represents the increase in minutes when the first button is pressed -/
def button1_adjustment : ℤ := 9

/-- Represents the decrease in minutes when the second button is pressed -/
def button2_adjustment : ℤ := 20

/-- Represents the equation for adjusting the clock -/
def clock_adjustment (a b : ℤ) : Prop :=
  button1_adjustment * a - button2_adjustment * b = time_difference

/-- The minimum number of button presses required -/
def min_button_presses : ℕ := 24

/-- Theorem stating that the minimum number of button presses to correctly set the clock is 24 -/
theorem min_button_presses_correct :
  ∃ (a b : ℤ), clock_adjustment a b ∧ a ≥ 0 ∧ b ≥ 0 ∧ a + b = min_button_presses ∧
  (∀ (c d : ℤ), clock_adjustment c d → c ≥ 0 → d ≥ 0 → c + d ≥ min_button_presses) :=
by sorry

end NUMINAMATH_CALUDE_min_button_presses_correct_l3697_369703


namespace NUMINAMATH_CALUDE_square_of_real_not_always_positive_l3697_369765

theorem square_of_real_not_always_positive : 
  ¬ (∀ a : ℝ, a^2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_square_of_real_not_always_positive_l3697_369765


namespace NUMINAMATH_CALUDE_chemistry_class_size_l3697_369731

/-- Represents the number of students in a school with chemistry and biology classes -/
structure School where
  total : ℕ
  chemistry : ℕ
  biology : ℕ
  both : ℕ

/-- The conditions of the problem -/
def school_conditions (s : School) : Prop :=
  s.total = 43 ∧
  s.both = 5 ∧
  s.chemistry = 3 * s.biology ∧
  s.total = (s.chemistry - s.both) + (s.biology - s.both) + s.both

/-- The theorem to be proved -/
theorem chemistry_class_size (s : School) :
  school_conditions s → s.chemistry = 36 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_class_size_l3697_369731


namespace NUMINAMATH_CALUDE_solution_exists_l3697_369761

theorem solution_exists : ∃ M : ℤ, (14 : ℤ)^2 * (35 : ℤ)^2 = (10 : ℤ)^2 * (M - 10)^2 := by
  use 59
  sorry

#check solution_exists

end NUMINAMATH_CALUDE_solution_exists_l3697_369761


namespace NUMINAMATH_CALUDE_ring_toss_daily_income_l3697_369708

theorem ring_toss_daily_income (total_income : ℕ) (num_days : ℕ) (daily_income : ℕ) : 
  total_income = 7560 → 
  num_days = 12 → 
  total_income = daily_income * num_days →
  daily_income = 630 := by
sorry

end NUMINAMATH_CALUDE_ring_toss_daily_income_l3697_369708


namespace NUMINAMATH_CALUDE_square_sum_given_cube_sum_and_product_l3697_369779

theorem square_sum_given_cube_sum_and_product (x y : ℝ) : 
  (x + y)^3 = 8 → x * y = 5 → x^2 + y^2 = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_cube_sum_and_product_l3697_369779


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l3697_369770

/-- Proves that adding 2.4 liters of pure alcohol to a 6-liter solution that is 30% alcohol
    will result in a solution that is 50% alcohol. -/
theorem alcohol_solution_proof (initial_volume : ℝ) (initial_concentration : ℝ)
  (added_alcohol : ℝ) (target_concentration : ℝ)
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.3)
  (h3 : added_alcohol = 2.4)
  (h4 : target_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = target_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l3697_369770


namespace NUMINAMATH_CALUDE_cake_distribution_l3697_369793

theorem cake_distribution (total_cake : ℕ) (friends : ℕ) (pieces_per_friend : ℕ) 
  (h1 : total_cake = 150)
  (h2 : friends = 50)
  (h3 : pieces_per_friend * friends = total_cake) :
  pieces_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_cake_distribution_l3697_369793


namespace NUMINAMATH_CALUDE_odd_function_with_period_4_l3697_369707

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_with_period_4 (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 4) 
  (h_min_period : ∀ p, 0 < p → p < 4 → ¬ has_period f p) : 
  f 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_with_period_4_l3697_369707


namespace NUMINAMATH_CALUDE_inequality_proof_l3697_369713

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a / Real.sqrt (b^2 + 1) + b / Real.sqrt (a^2 + 1) ≥ (a + b) / Real.sqrt (a * b + 1) ∧
  (a / Real.sqrt (b^2 + 1) + b / Real.sqrt (a^2 + 1) = (a + b) / Real.sqrt (a * b + 1) ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3697_369713


namespace NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l3697_369724

theorem largest_whole_number_nine_times_less_than_150 :
  ∃ (x : ℕ), x = 16 ∧ (∀ y : ℕ, 9 * y < 150 → y ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l3697_369724


namespace NUMINAMATH_CALUDE_dihedral_angle_range_l3697_369786

/-- The dihedral angle between adjacent faces in a regular n-prism -/
def dihedral_angle (n : ℕ) (θ : ℝ) : Prop :=
  n > 2 ∧ ((n - 2 : ℝ) / n) * Real.pi < θ ∧ θ < Real.pi

/-- Theorem stating the range of dihedral angles in a regular n-prism -/
theorem dihedral_angle_range (n : ℕ) :
  ∃ θ : ℝ, dihedral_angle n θ :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_range_l3697_369786


namespace NUMINAMATH_CALUDE_catch_up_distance_l3697_369746

/-- The problem of two people traveling at different speeds --/
theorem catch_up_distance
  (speed_a : ℝ)
  (speed_b : ℝ)
  (delay : ℝ)
  (h1 : speed_a = 10)
  (h2 : speed_b = 20)
  (h3 : delay = 6)
  : speed_b * (speed_a * delay / (speed_b - speed_a)) = 120 :=
by sorry

end NUMINAMATH_CALUDE_catch_up_distance_l3697_369746


namespace NUMINAMATH_CALUDE_right_triangle_with_median_condition_l3697_369782

theorem right_triangle_with_median_condition (c : ℝ) (h : c > 0) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    a^2 + b^2 = c^2 ∧
    (c / 2)^2 = a * b ∧
    a = (c * (Real.sqrt 6 + Real.sqrt 2)) / 4 ∧
    b = (c * (Real.sqrt 6 - Real.sqrt 2)) / 4 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_median_condition_l3697_369782


namespace NUMINAMATH_CALUDE_product_and_sum_of_factors_l3697_369756

theorem product_and_sum_of_factors : ∃ a b : ℕ, 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8670 ∧ 
  a + b = 136 := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_of_factors_l3697_369756


namespace NUMINAMATH_CALUDE_waiter_customers_l3697_369749

/-- Calculates the number of remaining customers given the initial number and the number who left. -/
def remaining_customers (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

/-- Theorem stating that for a waiter with 14 initial customers, after 5 leave, 9 remain. -/
theorem waiter_customers : remaining_customers 14 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l3697_369749


namespace NUMINAMATH_CALUDE_sarah_earnings_l3697_369737

/-- Sarah's earnings for the week given her work hours and pay rates -/
theorem sarah_earnings : 
  let weekday_hours := 1.75 + 65/60 + 2.75 + 45/60
  let weekend_hours := 2
  let weekday_rate := 4
  let weekend_rate := 6
  (weekday_hours * weekday_rate + weekend_hours * weekend_rate : ℝ) = 37.33 := by
  sorry

end NUMINAMATH_CALUDE_sarah_earnings_l3697_369737


namespace NUMINAMATH_CALUDE_function_properties_l3697_369723

noncomputable def f (x : ℝ) : ℝ := 2^(Real.sin x)

theorem function_properties :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < π ∧ 0 < x₂ ∧ x₂ < π ∧ f x₁ + f x₂ = 2) ∨
  (∀ x₁ x₂ : ℝ, -π/2 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/2 → f x₁ < f x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3697_369723


namespace NUMINAMATH_CALUDE_compound_prop_evaluation_l3697_369755

-- Define the propositions
variable (p q : Prop)

-- Define the truth values of p and q
axiom p_true : p
axiom q_false : ¬q

-- Define the compound propositions
def prop1 := p ∨ q
def prop2 := p ∧ q
def prop3 := ¬p ∧ q
def prop4 := ¬p ∨ ¬q

-- State the theorem
theorem compound_prop_evaluation :
  prop1 p q ∧ prop4 p q ∧ ¬(prop2 p q) ∧ ¬(prop3 p q) :=
sorry

end NUMINAMATH_CALUDE_compound_prop_evaluation_l3697_369755


namespace NUMINAMATH_CALUDE_candied_fruit_earnings_l3697_369751

/-- The number of candied apples made -/
def num_apples : ℕ := 15

/-- The price of each candied apple in dollars -/
def price_apple : ℚ := 2

/-- The number of candied grapes made -/
def num_grapes : ℕ := 12

/-- The price of each candied grape in dollars -/
def price_grape : ℚ := (3 : ℚ) / 2

/-- The total earnings from selling all candied apples and grapes -/
def total_earnings : ℚ := num_apples * price_apple + num_grapes * price_grape

theorem candied_fruit_earnings : total_earnings = 48 := by
  sorry

end NUMINAMATH_CALUDE_candied_fruit_earnings_l3697_369751


namespace NUMINAMATH_CALUDE_range_of_f_l3697_369788

/-- The diamond operation -/
def diamond (x y : ℝ) : ℝ := (x + y)^2 - x * y

/-- The function f -/
def f (a x : ℝ) : ℝ := diamond a x

theorem range_of_f (a : ℝ) (h : diamond 1 a = 3) :
  ∀ y : ℝ, y > 1 → ∃ x : ℝ, x > 0 ∧ f a x = y ∧
  ∀ z : ℝ, z > 0 → f a z ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3697_369788


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3697_369763

/-- A hyperbola with center at the origin, focus on the x-axis, and an asymptote tangent to a specific circle has eccentricity 2. -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 4*y + 3 = 0 → (b*x - a*y)^2 ≤ (a^2 + b^2) * ((x - 0)^2 + (y - 2)^2)) → 
  (∃ x : ℝ, x ≠ 0 ∧ (0, x) ∈ {(x, y) | (x/a)^2 - (y/b)^2 = 1}) →
  c^2 = a^2 + b^2 →
  c / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3697_369763


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3697_369726

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p ∣ (20^4 + 15^4 - 10^5) ∧
    ∀ (q : ℕ), Nat.Prime q → q ∣ (20^4 + 15^4 - 10^5) → q ≤ p ∧
    p = 59 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3697_369726


namespace NUMINAMATH_CALUDE_keith_bought_cards_l3697_369709

/-- The number of baseball cards Keith bought -/
def cards_bought : ℕ := sorry

/-- Fred's initial number of baseball cards -/
def initial_cards : ℕ := 40

/-- Fred's current number of baseball cards -/
def current_cards : ℕ := 18

/-- Theorem: The number of cards Keith bought is equal to the difference
    between Fred's initial and current number of cards -/
theorem keith_bought_cards : 
  cards_bought = initial_cards - current_cards := by sorry

end NUMINAMATH_CALUDE_keith_bought_cards_l3697_369709


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3697_369729

-- Define the set M
def M : Set ℝ := {x | 2/x < 1}

-- Define the set N
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

-- The theorem to prove
theorem complement_M_intersect_N : (Set.univ \ M) ∩ N = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3697_369729


namespace NUMINAMATH_CALUDE_existence_of_divisible_difference_l3697_369744

theorem existence_of_divisible_difference (x : Fin 2022 → ℤ) :
  ∃ i j : Fin 2022, i ≠ j ∧ (2021 : ℤ) ∣ (x j - x i) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_divisible_difference_l3697_369744


namespace NUMINAMATH_CALUDE_smallest_integers_difference_smallest_integers_difference_is_27720_l3697_369734

theorem smallest_integers_difference : ℕ → Prop :=
  fun d =>
    ∃ n₁ n₂ : ℕ,
      n₁ > 1 ∧ n₂ > 1 ∧
      n₁ < n₂ ∧
      (∀ k : ℕ, 2 ≤ k → k ≤ 11 → n₁ % k = 1) ∧
      (∀ k : ℕ, 2 ≤ k → k ≤ 11 → n₂ % k = 1) ∧
      (∀ m : ℕ, m > 1 → m < n₂ → m ≠ n₁ → ∃ k : ℕ, 2 ≤ k ∧ k ≤ 11 ∧ m % k ≠ 1) ∧
      d = n₂ - n₁

theorem smallest_integers_difference_is_27720 : smallest_integers_difference 27720 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integers_difference_smallest_integers_difference_is_27720_l3697_369734


namespace NUMINAMATH_CALUDE_median_length_inequality_l3697_369705

theorem median_length_inequality (a b c s_a : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Triangle sides are positive
  a + b > c ∧ b + c > a ∧ a + c > b ∧  -- Triangle inequality
  s_a > 0 ∧  -- Median length is positive
  s_a^2 = (b^2 + c^2) / 4 - a^2 / 16  -- Median length formula
  →
  s_a < (b + c) / 2 := by
sorry

end NUMINAMATH_CALUDE_median_length_inequality_l3697_369705


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3697_369711

theorem quadratic_function_property (a m : ℝ) (h1 : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - x + a
  f m < 0 → f (m - 1) > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3697_369711


namespace NUMINAMATH_CALUDE_ellipse_equation_l3697_369721

theorem ellipse_equation (major_axis : ℝ) (eccentricity : ℝ) :
  major_axis = 8 →
  eccentricity = 3/4 →
  (∃ x y : ℝ, x^2/16 + y^2/7 = 1) ∨ (∃ x y : ℝ, x^2/7 + y^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3697_369721


namespace NUMINAMATH_CALUDE_sequence_monotonicity_l3697_369753

theorem sequence_monotonicity (k : ℝ) : 
  (∀ n : ℕ, (n + 1)^2 + k*(n + 1) + 2 > n^2 + k*n + 2) → k > -3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_monotonicity_l3697_369753


namespace NUMINAMATH_CALUDE_base_8_243_equals_163_l3697_369754

def base_8_to_10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

theorem base_8_243_equals_163 :
  base_8_to_10 2 4 3 = 163 := by
  sorry

end NUMINAMATH_CALUDE_base_8_243_equals_163_l3697_369754


namespace NUMINAMATH_CALUDE_range_of_f_l3697_369772

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3697_369772


namespace NUMINAMATH_CALUDE_complex_power_2019_l3697_369771

-- Define the imaginary unit i
variable (i : ℂ)

-- Define the property of i being the imaginary unit
axiom i_squared : i^2 = -1

-- State the theorem
theorem complex_power_2019 : (((1 + i) / (1 - i)) ^ 2019 : ℂ) = -i := by sorry

end NUMINAMATH_CALUDE_complex_power_2019_l3697_369771


namespace NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l3697_369784

theorem angle_measure_in_special_quadrilateral :
  ∀ (P Q R S : ℝ),
  P = 3 * Q →
  P = 4 * R →
  P = 6 * S →
  P + Q + R + S = 360 →
  P = 206 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l3697_369784


namespace NUMINAMATH_CALUDE_overlap_area_theorem_l3697_369758

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.sideLength * s.sideLength

/-- The overlap configuration of two squares -/
structure SquareOverlap where
  largeSquare : Square
  smallSquare : Square
  smallSquareTouchesCenter : smallSquare.sideLength = largeSquare.sideLength / 2

/-- The area covered only by the larger square in the overlap configuration -/
def SquareOverlap.areaOnlyLarger (so : SquareOverlap) : ℝ :=
  so.largeSquare.area - so.smallSquare.area

/-- The main theorem -/
theorem overlap_area_theorem (so : SquareOverlap) 
    (h1 : so.largeSquare.sideLength = 8) 
    (h2 : so.smallSquare.sideLength = 4) : 
    so.areaOnlyLarger = 48 := by
  sorry


end NUMINAMATH_CALUDE_overlap_area_theorem_l3697_369758


namespace NUMINAMATH_CALUDE_line_m_equation_l3697_369798

/-- Two distinct lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflection of a point about a line -/
def reflect (p : Point) (l : Line) : Point := sorry

/-- The theorem statement -/
theorem line_m_equation (l m : Line) (Q Q'' : Point) :
  l.a = 3 ∧ l.b = -4 ∧ l.c = 0 →  -- Line ℓ: 3x - 4y = 0
  Q.x = 3 ∧ Q.y = -2 →  -- Point Q(3, -2)
  Q''.x = 2 ∧ Q''.y = 5 →  -- Point Q''(2, 5)
  (∃ Q' : Point, reflect Q l = Q' ∧ reflect Q' m = Q'') →  -- Reflection conditions
  m.a = 1 ∧ m.b = 7 ∧ m.c = 0  -- Line m: x + 7y = 0
  := by sorry

end NUMINAMATH_CALUDE_line_m_equation_l3697_369798


namespace NUMINAMATH_CALUDE_tan_thirteen_pi_fourth_l3697_369794

theorem tan_thirteen_pi_fourth : Real.tan (13 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirteen_pi_fourth_l3697_369794


namespace NUMINAMATH_CALUDE_representative_count_l3697_369785

/-- The number of ways to choose a math class representative from a class with a given number of boys and girls. -/
def choose_representative (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  num_boys + num_girls

/-- Theorem: The number of ways to choose a math class representative from a class with 26 boys and 24 girls is 50. -/
theorem representative_count : choose_representative 26 24 = 50 := by
  sorry

end NUMINAMATH_CALUDE_representative_count_l3697_369785


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3697_369757

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + a 2 = 7)
  (h_diff : a 1 - a 3 = -6) :
  a 5 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3697_369757


namespace NUMINAMATH_CALUDE_average_percentages_correct_l3697_369790

-- Define the subjects
inductive Subject
  | English
  | Mathematics
  | Physics
  | Chemistry
  | Biology
  | History
  | Geography

-- Define the marks and total marks for each subject
def marks (s : Subject) : ℕ :=
  match s with
  | Subject.English => 76
  | Subject.Mathematics => 65
  | Subject.Physics => 82
  | Subject.Chemistry => 67
  | Subject.Biology => 85
  | Subject.History => 92
  | Subject.Geography => 58

def totalMarks (s : Subject) : ℕ :=
  match s with
  | Subject.English => 120
  | Subject.Mathematics => 150
  | Subject.Physics => 100
  | Subject.Chemistry => 80
  | Subject.Biology => 100
  | Subject.History => 150
  | Subject.Geography => 75

-- Define the average percentage calculation
def averagePercentage (s : Subject) : ℚ :=
  (marks s : ℚ) / (totalMarks s : ℚ) * 100

-- Theorem to prove the correctness of average percentages
theorem average_percentages_correct :
  averagePercentage Subject.English = 63.33 ∧
  averagePercentage Subject.Mathematics = 43.33 ∧
  averagePercentage Subject.Physics = 82 ∧
  averagePercentage Subject.Chemistry = 83.75 ∧
  averagePercentage Subject.Biology = 85 ∧
  averagePercentage Subject.History = 61.33 ∧
  averagePercentage Subject.Geography = 77.33 := by
  sorry


end NUMINAMATH_CALUDE_average_percentages_correct_l3697_369790


namespace NUMINAMATH_CALUDE_largest_share_proof_l3697_369796

def profit_distribution (ratio : List Nat) (total_profit : Nat) : List Nat :=
  let total_parts := ratio.sum
  let part_value := total_profit / total_parts
  ratio.map (· * part_value)

theorem largest_share_proof (ratio : List Nat) (total_profit : Nat) :
  ratio = [3, 3, 4, 5, 6] → total_profit = 42000 →
  (profit_distribution ratio total_profit).maximum? = some 12000 := by
  sorry

end NUMINAMATH_CALUDE_largest_share_proof_l3697_369796


namespace NUMINAMATH_CALUDE_cos_180_degrees_l3697_369759

theorem cos_180_degrees : Real.cos (π) = -1 := by sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l3697_369759


namespace NUMINAMATH_CALUDE_sum_of_factorization_coefficients_l3697_369728

theorem sum_of_factorization_coefficients :
  ∀ (a b c : ℤ),
  (∀ x : ℝ, x^2 + 17*x + 70 = (x + a) * (x + b)) →
  (∀ x : ℝ, x^2 - 19*x + 84 = (x - b) * (x - c)) →
  a + b + c = 29 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factorization_coefficients_l3697_369728


namespace NUMINAMATH_CALUDE_chess_tournament_solutions_l3697_369781

def chess_tournament (x : ℕ) : Prop :=
  ∃ y : ℕ,
    -- Two 7th graders scored 8 points in total
    8 + x * y = (x + 2) * (x + 1) / 2 ∧
    -- y is the number of points each 8th grader scored
    y > 0

theorem chess_tournament_solutions :
  ∀ x : ℕ, chess_tournament x ↔ (x = 7 ∨ x = 14) :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_solutions_l3697_369781


namespace NUMINAMATH_CALUDE_inequality_proof_l3697_369764

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3697_369764


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3697_369730

theorem average_speed_calculation (distance1 distance2 time1 time2 : ℝ) 
  (h1 : distance1 = 90)
  (h2 : distance2 = 80)
  (h3 : time1 = 1)
  (h4 : time2 = 1) :
  (distance1 + distance2) / (time1 + time2) = 85 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3697_369730


namespace NUMINAMATH_CALUDE_pencil_price_l3697_369718

theorem pencil_price 
  (total_items : ℕ) 
  (pen_count : ℕ) 
  (pencil_count : ℕ) 
  (total_cost : ℚ) 
  (avg_pen_price : ℚ) 
  (h1 : total_items = pen_count + pencil_count)
  (h2 : total_items = 105)
  (h3 : pen_count = 30)
  (h4 : pencil_count = 75)
  (h5 : total_cost = 750)
  (h6 : avg_pen_price = 20) :
  (total_cost - pen_count * avg_pen_price) / pencil_count = 2 := by
sorry


end NUMINAMATH_CALUDE_pencil_price_l3697_369718


namespace NUMINAMATH_CALUDE_constant_d_value_l3697_369722

theorem constant_d_value (a d : ℝ) (h : ∀ x : ℝ, (x - 3) * (x + a) = x^2 + d*x - 18) : d = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_d_value_l3697_369722
