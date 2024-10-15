import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l4044_404407

/-- The radius of the inscribed circle in a triangle with side lengths 8, 10, and 14 is √6. -/
theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 8) (h2 : DF = 10) (h3 : EF = 14) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l4044_404407


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l4044_404453

theorem rectangular_plot_breadth (b : ℝ) (l : ℝ) (A : ℝ) : 
  A = 23 * b →  -- Area is 23 times the breadth
  l = b + 10 →  -- Length is 10 meters more than breadth
  A = l * b →   -- Area formula for rectangle
  b = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l4044_404453


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l4044_404451

theorem cubic_equation_solutions :
  ∀ m n : ℤ, m^3 - n^3 = 2*m*n + 8 ↔ (m = 0 ∧ n = -2) ∨ (m = 2 ∧ n = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l4044_404451


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l4044_404481

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_prod : a 5 * a 13 = 6)
  (h_sum : a 4 + a 14 = 5) :
  a 80 / a 90 = 2/3 ∨ a 80 / a 90 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l4044_404481


namespace NUMINAMATH_CALUDE_preference_change_difference_l4044_404420

theorem preference_change_difference (initial_online initial_traditional final_online final_traditional : ℚ) 
  (h_initial_sum : initial_online + initial_traditional = 1)
  (h_final_sum : final_online + final_traditional = 1)
  (h_initial_online : initial_online = 2/5)
  (h_initial_traditional : initial_traditional = 3/5)
  (h_final_online : final_online = 4/5)
  (h_final_traditional : final_traditional = 1/5) :
  let min_change := |final_online - initial_online|
  let max_change := min initial_traditional (1 - initial_online)
  max_change - min_change = 2/5 := by
sorry

#eval (2 : ℚ) / 5 -- This should evaluate to 0.4, which is 40%

end NUMINAMATH_CALUDE_preference_change_difference_l4044_404420


namespace NUMINAMATH_CALUDE_parabola_satisfies_equation_l4044_404406

/-- A parabola with vertex at the origin, symmetric about coordinate axes, passing through (2, -3) -/
structure Parabola where
  /-- The parabola passes through the point (2, -3) -/
  passes_through : (2 : ℝ)^2 + (-3 : ℝ)^2 ≠ 0

/-- The equation of the parabola -/
def parabola_equation (p : Parabola) : Prop :=
  (∀ x y : ℝ, y^2 = 9/2 * x) ∨ (∀ x y : ℝ, x^2 = -4/3 * y)

/-- Theorem stating that the parabola satisfies the given equation -/
theorem parabola_satisfies_equation (p : Parabola) : parabola_equation p := by
  sorry

end NUMINAMATH_CALUDE_parabola_satisfies_equation_l4044_404406


namespace NUMINAMATH_CALUDE_jessicas_initial_quarters_l4044_404472

/-- 
Given that Jessica received some quarters from her sister and now has a certain number of quarters,
this theorem proves the number of quarters Jessica had initially.
-/
theorem jessicas_initial_quarters 
  (quarters_from_sister : ℕ) -- Number of quarters Jessica received from her sister
  (current_quarters : ℕ) -- Number of quarters Jessica has now
  (h1 : quarters_from_sister = 3) -- Jessica received 3 quarters from her sister
  (h2 : current_quarters = 11) -- Jessica now has 11 quarters
  : current_quarters - quarters_from_sister = 8 := by
  sorry

end NUMINAMATH_CALUDE_jessicas_initial_quarters_l4044_404472


namespace NUMINAMATH_CALUDE_short_stack_customers_count_l4044_404495

/-- The number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- The number of pancakes in a short stack -/
def short_stack : ℕ := 3

/-- The number of customers who ordered the big stack -/
def big_stack_customers : ℕ := 6

/-- The total number of pancakes made -/
def total_pancakes : ℕ := 57

/-- The number of customers who ordered the short stack -/
def short_stack_customers : ℕ := 9

theorem short_stack_customers_count :
  short_stack_customers * short_stack + big_stack_customers * big_stack = total_pancakes := by
  sorry

end NUMINAMATH_CALUDE_short_stack_customers_count_l4044_404495


namespace NUMINAMATH_CALUDE_bounded_difference_exists_l4044_404429

/-- A monotonous function satisfying the given inequality condition. -/
structure MonotonousFunction (f : ℝ → ℝ) (c₁ c₂ : ℝ) : Prop :=
  (mono : Monotone f)
  (pos_const : c₁ > 0 ∧ c₂ > 0)
  (ineq : ∀ x y : ℝ, f x + f y - c₁ ≤ f (x + y) ∧ f (x + y) ≤ f x + f y + c₂)

/-- The main theorem stating the existence of k such that f(x) - kx is bounded. -/
theorem bounded_difference_exists (f : ℝ → ℝ) (c₁ c₂ : ℝ) 
  (hf : MonotonousFunction f c₁ c₂) : 
  ∃ k : ℝ, ∃ M : ℝ, ∀ x : ℝ, |f x - k * x| ≤ M :=
sorry

end NUMINAMATH_CALUDE_bounded_difference_exists_l4044_404429


namespace NUMINAMATH_CALUDE_graces_age_l4044_404414

/-- Grace's age problem -/
theorem graces_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ) :
  mother_age = 80 →
  grandmother_age = 2 * mother_age →
  grace_age = (3 * grandmother_age) / 8 →
  grace_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_graces_age_l4044_404414


namespace NUMINAMATH_CALUDE_product_four_consecutive_even_divisible_by_96_largest_divisor_four_consecutive_even_l4044_404461

/-- The product of four consecutive even natural numbers is always divisible by 96 -/
theorem product_four_consecutive_even_divisible_by_96 :
  ∀ n : ℕ, 96 ∣ (2*n) * (2*n + 2) * (2*n + 4) * (2*n + 6) :=
by sorry

/-- 96 is the largest natural number that always divides the product of four consecutive even natural numbers -/
theorem largest_divisor_four_consecutive_even :
  ∀ m : ℕ, (∀ n : ℕ, m ∣ (2*n) * (2*n + 2) * (2*n + 4) * (2*n + 6)) → m ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_product_four_consecutive_even_divisible_by_96_largest_divisor_four_consecutive_even_l4044_404461


namespace NUMINAMATH_CALUDE_trisection_intersection_x_coordinate_l4044_404421

theorem trisection_intersection_x_coordinate : 
  let f : ℝ → ℝ := λ x => Real.log x
  let x₁ : ℝ := 2
  let x₂ : ℝ := 500
  let y₁ : ℝ := f x₁
  let y₂ : ℝ := f x₂
  let yC : ℝ := (2/3) * y₁ + (1/3) * y₂
  ∃ x₃ : ℝ, f x₃ = yC ∧ x₃ = 10 * (2^(2/3)) * (5^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_trisection_intersection_x_coordinate_l4044_404421


namespace NUMINAMATH_CALUDE_nose_spray_cost_l4044_404489

/-- Calculates the cost per nose spray in a "buy one get one free" promotion -/
def costPerNoseSpray (totalPaid : ℚ) (totalBought : ℕ) : ℚ :=
  totalPaid / (totalBought / 2)

theorem nose_spray_cost :
  let totalPaid : ℚ := 15
  let totalBought : ℕ := 10
  costPerNoseSpray totalPaid totalBought = 3 := by
  sorry

end NUMINAMATH_CALUDE_nose_spray_cost_l4044_404489


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l4044_404442

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Calculate total sum after simple interest -/
def total_sum (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal + simple_interest principal rate time

theorem simple_interest_calculation (P : ℚ) :
  total_sum P (5 : ℚ) (5 : ℚ) = 16065 →
  simple_interest P (5 : ℚ) (5 : ℚ) = 3213 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l4044_404442


namespace NUMINAMATH_CALUDE_exists_real_not_in_geometric_sequence_l4044_404499

/-- A geometric sequence is a sequence where the ratio of each term to its preceding term is constant (not zero) -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- There exists a real number that cannot be a term in any geometric sequence -/
theorem exists_real_not_in_geometric_sequence :
  ∃ x : ℝ, ∀ a : ℕ → ℝ, IsGeometricSequence a → ∀ n : ℕ, a n ≠ x :=
sorry

end NUMINAMATH_CALUDE_exists_real_not_in_geometric_sequence_l4044_404499


namespace NUMINAMATH_CALUDE_min_xyz_value_l4044_404434

/-- Given real numbers x, y, z satisfying the given conditions, 
    the minimum value of xyz is 9√11 - 32 -/
theorem min_xyz_value (x y z : ℝ) 
    (h1 : x * y + 2 * z = 1) 
    (h2 : x^2 + y^2 + z^2 = 5) : 
  ∀ (a b c : ℝ), a * b + 2 * c = 1 → a^2 + b^2 + c^2 = 5 → 
    x * y * z ≤ a * b * c ∧ 
    ∃ (x₀ y₀ z₀ : ℝ), x₀ * y₀ + 2 * z₀ = 1 ∧ x₀^2 + y₀^2 + z₀^2 = 5 ∧ 
      x₀ * y₀ * z₀ = 9 * Real.sqrt 11 - 32 :=
by
  sorry

#check min_xyz_value

end NUMINAMATH_CALUDE_min_xyz_value_l4044_404434


namespace NUMINAMATH_CALUDE_plywood_area_l4044_404402

theorem plywood_area (width length area : ℝ) :
  width = 6 →
  length = 4 →
  area = width * length →
  area = 24 :=
by sorry

end NUMINAMATH_CALUDE_plywood_area_l4044_404402


namespace NUMINAMATH_CALUDE_max_value_expression_l4044_404447

theorem max_value_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l4044_404447


namespace NUMINAMATH_CALUDE_exists_composite_carmichael_number_l4044_404431

theorem exists_composite_carmichael_number : ∃ n : ℕ, 
  n > 1 ∧ 
  ¬ Nat.Prime n ∧ 
  ∀ a : ℤ, (n : ℤ) ∣ (a^n - a) := by
  sorry

end NUMINAMATH_CALUDE_exists_composite_carmichael_number_l4044_404431


namespace NUMINAMATH_CALUDE_max_volume_rectangular_prism_l4044_404425

/-- Represents a right prism with a rectangular base -/
structure RectangularPrism where
  a : ℝ  -- length of the base
  b : ℝ  -- width of the base
  h : ℝ  -- height of the prism

/-- The sum of areas of two lateral faces and the base face is 32 -/
def area_constraint (p : RectangularPrism) : Prop :=
  p.a * p.h + p.b * p.h + p.a * p.b = 32

/-- The volume of the prism -/
def volume (p : RectangularPrism) : ℝ :=
  p.a * p.b * p.h

/-- Theorem stating the maximum volume of the prism -/
theorem max_volume_rectangular_prism :
  ∀ p : RectangularPrism, area_constraint p →
  volume p ≤ (128 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_rectangular_prism_l4044_404425


namespace NUMINAMATH_CALUDE_trapezoid_area_l4044_404432

/-- The area of a trapezoid with height h, bases 4h + 2 and 5h is (9h^2 + 2h) / 2 -/
theorem trapezoid_area (h : ℝ) : 
  let base1 := 4 * h + 2
  let base2 := 5 * h
  ((base1 + base2) / 2) * h = (9 * h^2 + 2 * h) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l4044_404432


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l4044_404450

theorem quadratic_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - x + 2 = 0 ∧ a * y^2 - y + 2 = 0) ↔ (a < 1/8 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l4044_404450


namespace NUMINAMATH_CALUDE_optimal_rental_plan_l4044_404460

/-- Represents the rental plan for cars -/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a rental plan is valid according to the given conditions -/
def isValidPlan (plan : RentalPlan) : Prop :=
  plan.typeA + plan.typeB = 8 ∧
  35 * plan.typeA + 30 * plan.typeB ≥ 255 ∧
  400 * plan.typeA + 320 * plan.typeB ≤ 3000

/-- Calculates the total cost of a rental plan -/
def totalCost (plan : RentalPlan) : ℕ :=
  400 * plan.typeA + 320 * plan.typeB

/-- The optimal rental plan -/
def optimalPlan : RentalPlan :=
  { typeA := 3, typeB := 5 }

theorem optimal_rental_plan :
  isValidPlan optimalPlan ∧
  totalCost optimalPlan = 2800 ∧
  ∀ plan, isValidPlan plan → totalCost plan ≥ totalCost optimalPlan :=
by sorry

end NUMINAMATH_CALUDE_optimal_rental_plan_l4044_404460


namespace NUMINAMATH_CALUDE_point_outside_intersecting_line_l4044_404485

/-- A line ax + by = 1 intersects a unit circle if and only if the distance
    from the origin to the line is less than 1 -/
def line_intersects_circle (a b : ℝ) : Prop :=
  (|1| / Real.sqrt (a^2 + b^2)) < 1

/-- A point (x,y) is outside the unit circle if its distance from the origin is greater than 1 -/
def point_outside_circle (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) > 1

theorem point_outside_intersecting_line (a b : ℝ) :
  line_intersects_circle a b → point_outside_circle a b :=
by sorry

end NUMINAMATH_CALUDE_point_outside_intersecting_line_l4044_404485


namespace NUMINAMATH_CALUDE_towel_folding_theorem_l4044_404467

/-- Represents the number of towels a person can fold in a given time -/
structure FoldingRate where
  towels : ℕ
  minutes : ℕ

/-- Calculates the number of towels folded in one hour given a folding rate -/
def towelsPerHour (rate : FoldingRate) : ℕ :=
  (60 / rate.minutes) * rate.towels

/-- The total number of towels folded by all three people in one hour -/
def totalTowelsPerHour (jane kyla anthony : FoldingRate) : ℕ :=
  towelsPerHour jane + towelsPerHour kyla + towelsPerHour anthony

theorem towel_folding_theorem (jane kyla anthony : FoldingRate)
  (h1 : jane = ⟨3, 5⟩)
  (h2 : kyla = ⟨5, 10⟩)
  (h3 : anthony = ⟨7, 20⟩) :
  totalTowelsPerHour jane kyla anthony = 87 := by
  sorry

#eval totalTowelsPerHour ⟨3, 5⟩ ⟨5, 10⟩ ⟨7, 20⟩

end NUMINAMATH_CALUDE_towel_folding_theorem_l4044_404467


namespace NUMINAMATH_CALUDE_sin_double_angle_l4044_404477

theorem sin_double_angle (α : ℝ) (h : Real.sin (α - π/4) = 3/5) : 
  Real.sin (2 * α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_l4044_404477


namespace NUMINAMATH_CALUDE_product_always_even_l4044_404455

theorem product_always_even (a b c : ℤ) : 
  Even ((a - b) * (b - c) * (c - a)) := by
  sorry

end NUMINAMATH_CALUDE_product_always_even_l4044_404455


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l4044_404439

/-- Given a parabola y = -(x+2)^2 - 3, its axis of symmetry is the line x = -2 -/
theorem parabola_axis_of_symmetry :
  let f : ℝ → ℝ := λ x => -(x + 2)^2 - 3
  ∃! a : ℝ, ∀ x y : ℝ, f x = f y → (x + y) / 2 = a :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l4044_404439


namespace NUMINAMATH_CALUDE_recipe_total_cups_l4044_404404

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℚ
  flour : ℚ
  sugar : ℚ

/-- Calculates the total cups of ingredients given a recipe ratio and cups of sugar used -/
def totalCups (ratio : RecipeRatio) (sugarCups : ℚ) : ℚ :=
  let partValue := sugarCups / ratio.sugar
  ratio.butter * partValue + ratio.flour * partValue + sugarCups

/-- Theorem: Given the specified recipe ratio and sugar amount, the total cups is 27.5 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := { butter := 1, flour := 6, sugar := 4 }
  totalCups ratio 10 = 27.5 := by
  sorry

#eval totalCups { butter := 1, flour := 6, sugar := 4 } 10

end NUMINAMATH_CALUDE_recipe_total_cups_l4044_404404


namespace NUMINAMATH_CALUDE_opposite_of_neg_three_l4044_404493

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem stating that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_three_l4044_404493


namespace NUMINAMATH_CALUDE_area_of_fourth_square_l4044_404409

/-- Given two right triangles PQR and PRS with a common hypotenuse PR,
    where the squares of the sides have areas 25, 64, 49, and an unknown value,
    prove that the area of the square on PS is 138 square units. -/
theorem area_of_fourth_square (PQ PR QR RS PS : ℝ) : 
  PQ^2 = 25 → QR^2 = 49 → RS^2 = 64 → 
  PQ^2 + QR^2 = PR^2 → PR^2 + RS^2 = PS^2 →
  PS^2 = 138 := by
  sorry

end NUMINAMATH_CALUDE_area_of_fourth_square_l4044_404409


namespace NUMINAMATH_CALUDE_bill_age_is_nineteen_l4044_404482

/-- Represents the ages of family members -/
structure FamilyAges where
  caroline : ℕ
  bill : ℕ
  daniel : ℕ
  alex : ℕ
  grandmother : ℕ

/-- Defines the conditions for the family ages problem -/
def ValidFamilyAges (ages : FamilyAges) : Prop :=
  ages.bill = 2 * ages.caroline - 1 ∧
  ages.daniel = ages.caroline / 2 ∧
  ages.alex = ages.bill ∧
  ages.grandmother = 4 * ages.caroline ∧
  ages.caroline + ages.bill + ages.daniel + ages.alex + ages.grandmother = 108

/-- Theorem stating that if the family ages are valid, Bill's age is 19 -/
theorem bill_age_is_nineteen (ages : FamilyAges) (h : ValidFamilyAges ages) : ages.bill = 19 := by
  sorry

end NUMINAMATH_CALUDE_bill_age_is_nineteen_l4044_404482


namespace NUMINAMATH_CALUDE_thomas_savings_years_l4044_404443

/-- Represents the savings scenario for Thomas --/
structure SavingsScenario where
  allowance : ℕ  -- Weekly allowance in the first year
  wage : ℕ       -- Hourly wage from the second year
  hours : ℕ      -- Weekly work hours from the second year
  carCost : ℕ    -- Cost of the car
  spending : ℕ   -- Weekly spending
  remaining : ℕ  -- Amount still needed to buy the car

/-- Calculates the number of years Thomas has been saving --/
def yearsOfSaving (s : SavingsScenario) : ℕ :=
  2  -- This is the value we want to prove

/-- Theorem stating that Thomas has been saving for 2 years --/
theorem thomas_savings_years (s : SavingsScenario) 
  (h1 : s.allowance = 50)
  (h2 : s.wage = 9)
  (h3 : s.hours = 30)
  (h4 : s.carCost = 15000)
  (h5 : s.spending = 35)
  (h6 : s.remaining = 2000) :
  yearsOfSaving s = 2 := by
  sorry

#check thomas_savings_years

end NUMINAMATH_CALUDE_thomas_savings_years_l4044_404443


namespace NUMINAMATH_CALUDE_parabola_shift_l4044_404478

/-- A parabola shifted left and down -/
def shifted_parabola (x y : ℝ) : Prop :=
  y = -(x + 2)^2 - 3

/-- The original parabola -/
def original_parabola (x y : ℝ) : Prop :=
  y = -x^2

/-- Theorem stating that the shifted parabola is equivalent to
    the original parabola shifted 2 units left and 3 units down -/
theorem parabola_shift :
  ∀ x y : ℝ, shifted_parabola x y ↔ original_parabola (x + 2) (y + 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l4044_404478


namespace NUMINAMATH_CALUDE_jason_seashell_count_l4044_404452

def seashell_count (initial : ℕ) (given_tim : ℕ) (given_lily : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial - given_tim - given_lily + found - lost

theorem jason_seashell_count : 
  seashell_count 49 13 7 15 5 = 39 := by sorry

end NUMINAMATH_CALUDE_jason_seashell_count_l4044_404452


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l4044_404474

theorem sum_of_specific_numbers : 
  12345 + 23451 + 34512 + 45123 + 51234 = 166665 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l4044_404474


namespace NUMINAMATH_CALUDE_f_properties_l4044_404465

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  ∃ (period : ℝ),
    (∀ (x : ℝ), f (x + period) = f x) ∧
    (∀ (p : ℝ), (∀ (x : ℝ), f (x + p) = f x) → period ≤ p) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 2) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) ∧
    (∃ (x₁ : ℝ), x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₁ = 2) ∧
    (∃ (x₂ : ℝ), x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₂ = -1) ∧
    (∀ (x₀ : ℝ), x₀ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → 
      f x₀ = 6/5 → Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4044_404465


namespace NUMINAMATH_CALUDE_max_n_with_2013_trailing_zeros_l4044_404410

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The maximum value of N such that N! has exactly 2013 trailing zeros -/
theorem max_n_with_2013_trailing_zeros :
  ∀ n : ℕ, n > 8069 → trailingZeros n > 2013 ∧
  trailingZeros 8069 = 2013 :=
by sorry

end NUMINAMATH_CALUDE_max_n_with_2013_trailing_zeros_l4044_404410


namespace NUMINAMATH_CALUDE_binomial_9_5_l4044_404497

theorem binomial_9_5 : Nat.choose 9 5 = 756 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_5_l4044_404497


namespace NUMINAMATH_CALUDE_nth_root_power_comparison_l4044_404487

theorem nth_root_power_comparison (a : ℝ) (n m : ℕ) (h1 : 0 < a) (h2 : 0 < n) (h3 : 0 < m) :
  (a > 1 → a^(m/n) > 1) ∧ (a < 1 → a^(m/n) < 1) := by
  sorry

end NUMINAMATH_CALUDE_nth_root_power_comparison_l4044_404487


namespace NUMINAMATH_CALUDE_number_of_shooting_orders_l4044_404458

/-- Represents the number of targets in each column -/
def targets_per_column : Fin 3 → ℕ
  | 0 => 4  -- Column A
  | 1 => 3  -- Column B
  | 2 => 3  -- Column C

/-- The total number of targets -/
def total_targets : ℕ := 10

/-- The number of initial shooting sequences -/
def initial_sequences : ℕ := 2

/-- Calculates the number of permutations for the remaining shots -/
def remaining_permutations : ℕ :=
  Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 3)

/-- Theorem stating the total number of different orders to break the targets -/
theorem number_of_shooting_orders :
  initial_sequences * remaining_permutations = 1120 := by sorry

end NUMINAMATH_CALUDE_number_of_shooting_orders_l4044_404458


namespace NUMINAMATH_CALUDE_both_sports_lovers_l4044_404403

/-- The number of students who like basketball -/
def basketball_lovers : ℕ := 7

/-- The number of students who like cricket -/
def cricket_lovers : ℕ := 8

/-- The number of students who like basketball or cricket or both -/
def total_lovers : ℕ := 12

/-- The number of students who like both basketball and cricket -/
def both_lovers : ℕ := basketball_lovers + cricket_lovers - total_lovers

theorem both_sports_lovers : both_lovers = 3 := by sorry

end NUMINAMATH_CALUDE_both_sports_lovers_l4044_404403


namespace NUMINAMATH_CALUDE_max_quotient_value_l4044_404484

theorem max_quotient_value (a b : ℝ) 
  (ha : 210 ≤ a ∧ a ≤ 430) 
  (hb : 590 ≤ b ∧ b ≤ 1190) : 
  (∀ x y, 210 ≤ x ∧ x ≤ 430 ∧ 590 ≤ y ∧ y ≤ 1190 → y / x ≤ 1190 / 210) :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l4044_404484


namespace NUMINAMATH_CALUDE_arc_length_sixty_degrees_l4044_404459

theorem arc_length_sixty_degrees (r : ℝ) (h : r = 1) :
  let angle : ℝ := π / 3
  let arc_length : ℝ := r * angle
  arc_length = π / 3 := by sorry

end NUMINAMATH_CALUDE_arc_length_sixty_degrees_l4044_404459


namespace NUMINAMATH_CALUDE_percent_students_with_cats_l4044_404475

/-- Given a school with 500 students where 75 students own cats,
    prove that 15% of the students own cats. -/
theorem percent_students_with_cats :
  let total_students : ℕ := 500
  let cat_owners : ℕ := 75
  (cat_owners : ℚ) / total_students * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percent_students_with_cats_l4044_404475


namespace NUMINAMATH_CALUDE_second_player_wins_l4044_404430

/-- Represents the state of the game -/
structure GameState :=
  (boxes : Fin 11 → ℕ)

/-- Represents a move in the game -/
structure Move :=
  (skipped : Fin 11)

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  { boxes := λ i => if i = move.skipped then state.boxes i else state.boxes i + 1 }

/-- Checks if the game is won -/
def is_won (state : GameState) : Prop :=
  ∃ i, state.boxes i = 21

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Represents the game play -/
def play (initial_state : GameState) (strategy1 strategy2 : Strategy) : Prop :=
  ∃ (n : ℕ) (states : ℕ → GameState),
    states 0 = initial_state ∧
    (∀ k, states (k+1) = 
      if k % 2 = 0
      then apply_move (states k) (strategy1 (states k))
      else apply_move (states k) (strategy2 (states k))) ∧
    is_won (states (2*n + 1)) ∧ ¬is_won (states (2*n))

/-- The theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy2 : Strategy), ∀ (strategy1 : Strategy),
    play { boxes := λ _ => 0 } strategy1 strategy2 :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l4044_404430


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l4044_404445

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {-1, 2, 3}
  let B : Set ℝ := {a + 2, a^2 + 2}
  A ∩ B = {3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l4044_404445


namespace NUMINAMATH_CALUDE_trip_equation_correct_l4044_404415

/-- Represents a car trip with a stop -/
structure CarTrip where
  totalDistance : ℝ
  totalTime : ℝ
  stopDuration : ℝ
  speedBefore : ℝ
  speedAfter : ℝ

/-- The equation for the trip is correct -/
theorem trip_equation_correct (trip : CarTrip) 
    (h1 : trip.totalDistance = 300)
    (h2 : trip.totalTime = 4)
    (h3 : trip.stopDuration = 0.5)
    (h4 : trip.speedBefore = 60)
    (h5 : trip.speedAfter = 90) :
  ∃ t : ℝ, 
    t ≥ 0 ∧ 
    t ≤ trip.totalTime - trip.stopDuration ∧
    trip.speedBefore * t + trip.speedAfter * (trip.totalTime - trip.stopDuration - t) = trip.totalDistance :=
by sorry

end NUMINAMATH_CALUDE_trip_equation_correct_l4044_404415


namespace NUMINAMATH_CALUDE_product_of_fractions_l4044_404479

theorem product_of_fractions : 
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 2) * (f 3) * (f 4) * (f 5) * (f 6) = 43 / 63 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l4044_404479


namespace NUMINAMATH_CALUDE_function_minimum_value_l4044_404464

theorem function_minimum_value (a : ℝ) :
  (∃ x₀ : ℝ, (x₀ + a)^2 + (Real.exp x₀ + a / Real.exp 1)^2 ≤ 4 / (Real.exp 2 + 1)) →
  a = (Real.exp 2 - 1) / (Real.exp 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_value_l4044_404464


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_C_R_A_B_l4044_404437

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

-- Define the complement of A in ℝ
def C_R_A : Set ℝ := {x : ℝ | x < 3 ∨ x ≥ 7}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 7} := by sorry

-- Theorem for (C_R A) ∩ B
theorem intersection_C_R_A_B : (C_R_A ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_C_R_A_B_l4044_404437


namespace NUMINAMATH_CALUDE_total_pages_in_book_l4044_404448

/-- The number of pages Suzanne read on Monday -/
def monday_pages : ℝ := 15.5

/-- The number of pages Suzanne read on Tuesday -/
def tuesday_pages : ℝ := 1.5 * monday_pages + 16

/-- The total number of pages Suzanne read in two days -/
def total_pages_read : ℝ := monday_pages + tuesday_pages

/-- The theorem stating the total number of pages in the book -/
theorem total_pages_in_book : total_pages_read * 2 = 109.5 := by sorry

end NUMINAMATH_CALUDE_total_pages_in_book_l4044_404448


namespace NUMINAMATH_CALUDE_ellipse_properties_l4044_404454

/-- Ellipse C in the Cartesian coordinate system α -/
def C (b : ℝ) (x y : ℝ) : Prop :=
  0 < b ∧ b < 2 ∧ x^2 / 4 + y^2 / b^2 = 1

/-- Point A is the right vertex of C -/
def A : ℝ × ℝ := (2, 0)

/-- Line l passing through O with non-zero slope -/
def l (m : ℝ) (x y : ℝ) : Prop :=
  m ≠ 0 ∧ y = m * x

/-- P and Q are intersection points of l and C -/
def intersectionPoints (b m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C b x₁ y₁ ∧ C b x₂ y₂ ∧ l m x₁ y₁ ∧ l m x₂ y₂

/-- M and N are intersections of AP, AQ with y-axis -/
def MN (b m : ℝ) : Prop :=
  ∃ (y₁ y₂ : ℝ), (∃ (x₁ y₁' : ℝ), C b x₁ y₁' ∧ l m x₁ y₁' ∧ y₁ = (y₁' / (x₁ - 2)) * (-2)) ∧
                 (∃ (x₂ y₂' : ℝ), C b x₂ y₂' ∧ l m x₂ y₂' ∧ y₂ = (y₂' / (x₂ - 2)) * (-2))

theorem ellipse_properties (b m : ℝ) :
  C b 1 1 → intersectionPoints b m → (∃ (x y : ℝ), C b x y ∧ l m x y ∧ (x^2 + y^2)^(1/2) = 2 * ((x - 2)^2 + y^2)^(1/2)) →
  b^2 = 4/3 ∧ (∀ y₁ y₂ : ℝ, MN b m → y₁ * y₂ = b^2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l4044_404454


namespace NUMINAMATH_CALUDE_locker_count_l4044_404473

/-- Calculates the cost of labeling lockers given the number of lockers -/
def labelingCost (n : ℕ) : ℚ :=
  let cost1 := (min n 9 : ℚ) * 2 / 100
  let cost2 := (min (max (n - 9) 0) 90 : ℚ) * 4 / 100
  let cost3 := (min (max (n - 99) 0) 900 : ℚ) * 6 / 100
  let cost4 := (max (n - 999) 0 : ℚ) * 8 / 100
  cost1 + cost2 + cost3 + cost4

theorem locker_count : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    labelingCost n = 13794 / 100 ∧ 
    n = 2001 :=
sorry

end NUMINAMATH_CALUDE_locker_count_l4044_404473


namespace NUMINAMATH_CALUDE_sum_of_five_reals_l4044_404411

theorem sum_of_five_reals (a b c d e : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e)
  (eq1 : a + b = c)
  (eq2 : a + b + c = d)
  (eq3 : a + b + c + d = e)
  (c_val : c = 5) : 
  a + b + c + d + e = 40 := by
sorry

end NUMINAMATH_CALUDE_sum_of_five_reals_l4044_404411


namespace NUMINAMATH_CALUDE_smallest_interesting_number_l4044_404449

/-- A natural number is interesting if 2n is a perfect square and 15n is a perfect cube. -/
def is_interesting (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 2 * n = a ^ 2 ∧ 15 * n = b ^ 3

/-- 1800 is the smallest interesting number. -/
theorem smallest_interesting_number : 
  is_interesting 1800 ∧ ∀ m < 1800, ¬is_interesting m :=
by sorry

end NUMINAMATH_CALUDE_smallest_interesting_number_l4044_404449


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l4044_404469

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 15120 →
  e = 9 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l4044_404469


namespace NUMINAMATH_CALUDE_ordering_of_logarithms_and_exponential_l4044_404436

theorem ordering_of_logarithms_and_exponential : 
  let a := Real.log 3 / Real.log 5
  let b := Real.log 8 / Real.log 13
  let c := Real.exp (-1/2)
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_ordering_of_logarithms_and_exponential_l4044_404436


namespace NUMINAMATH_CALUDE_four_cubic_yards_to_cubic_inches_l4044_404490

-- Define the conversion factors
def yard_to_foot : ℝ := 3
def foot_to_inch : ℝ := 12

-- Define the volume conversion function
def cubic_yards_to_cubic_inches (cubic_yards : ℝ) : ℝ :=
  cubic_yards * (yard_to_foot ^ 3) * (foot_to_inch ^ 3)

-- Theorem statement
theorem four_cubic_yards_to_cubic_inches :
  cubic_yards_to_cubic_inches 4 = 186624 := by
  sorry

end NUMINAMATH_CALUDE_four_cubic_yards_to_cubic_inches_l4044_404490


namespace NUMINAMATH_CALUDE_other_communities_count_l4044_404426

theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total = 300 →
  muslim_percent = 44/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  (total : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 54 := by
sorry

end NUMINAMATH_CALUDE_other_communities_count_l4044_404426


namespace NUMINAMATH_CALUDE_pave_hall_l4044_404438

/-- The number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℚ :=
  (hall_length * hall_width * 100) / (stone_length * stone_width)

/-- Theorem stating that 2700 stones are required to pave the given hall -/
theorem pave_hall : stones_required 36 15 4 5 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_pave_hall_l4044_404438


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l4044_404422

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^2000 - 1) (2^1990 - 1) = 2^10 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l4044_404422


namespace NUMINAMATH_CALUDE_smallest_group_size_l4044_404435

theorem smallest_group_size (n : ℕ) : 
  (n % 18 = 0 ∧ n % 60 = 0) → n ≥ Nat.lcm 18 60 := by
  sorry

#eval Nat.lcm 18 60

end NUMINAMATH_CALUDE_smallest_group_size_l4044_404435


namespace NUMINAMATH_CALUDE_geometric_sequence_a8_l4044_404456

def is_geometric_sequence (a : ℕ+ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ+, a (n + 1) = a n * q

theorem geometric_sequence_a8 (a : ℕ+ → ℚ) :
  is_geometric_sequence a →
  a 2 = 1 / 16 →
  a 5 = 1 / 2 →
  a 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a8_l4044_404456


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l4044_404424

theorem quadratic_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 4*x + 3 = 0 ∧ a * y^2 - 4*y + 3 = 0) ↔ 
  (a < 4/3 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l4044_404424


namespace NUMINAMATH_CALUDE_max_length_valid_progression_l4044_404492

/-- An arithmetic progression of natural numbers. -/
structure ArithmeticProgression :=
  (first : ℕ)
  (diff : ℕ)
  (len : ℕ)

/-- Check if a natural number contains the digit 9. -/
def containsNine (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 9

/-- An arithmetic progression satisfying the given conditions. -/
def ValidProgression (ap : ArithmeticProgression) : Prop :=
  ap.diff ≠ 0 ∧
  ∀ i : ℕ, i < ap.len → ¬containsNine (ap.first + i * ap.diff)

/-- The main theorem: The maximum length of a valid progression is 72. -/
theorem max_length_valid_progression :
  ∀ ap : ArithmeticProgression, ValidProgression ap → ap.len ≤ 72 :=
sorry

end NUMINAMATH_CALUDE_max_length_valid_progression_l4044_404492


namespace NUMINAMATH_CALUDE_min_additional_coins_alex_coin_distribution_l4044_404444

theorem min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let min_required := (num_friends * (num_friends + 1)) / 2
  if min_required > initial_coins then
    min_required - initial_coins
  else
    0

theorem alex_coin_distribution : min_additional_coins 15 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_min_additional_coins_alex_coin_distribution_l4044_404444


namespace NUMINAMATH_CALUDE_trigonometric_equality_l4044_404468

theorem trigonometric_equality : 
  4 * Real.sin (30 * π / 180) - Real.sqrt 2 * Real.cos (45 * π / 180) - 
  Real.sqrt 3 * Real.tan (30 * π / 180) + 2 * Real.sin (60 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l4044_404468


namespace NUMINAMATH_CALUDE_max_side_length_of_triangle_l4044_404416

theorem max_side_length_of_triangle (a b c : ℕ) : 
  a < b → b < c →  -- Three different integer side lengths
  a + b + c = 24 → -- Perimeter is 24 units
  a + b > c →      -- Triangle inequality
  b + c > a →      -- Triangle inequality
  a + c > b →      -- Triangle inequality
  c ≤ 11 :=        -- Maximum length of any side is 11
by sorry

end NUMINAMATH_CALUDE_max_side_length_of_triangle_l4044_404416


namespace NUMINAMATH_CALUDE_max_consecutive_semiprimes_l4044_404488

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def isSemiPrime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ p ≠ q ∧ n = p + q

theorem max_consecutive_semiprimes :
  (∃ start : ℕ, ∀ i : ℕ, i < 5 → isSemiPrime (start + i)) ∧
  (¬∃ start : ℕ, ∀ i : ℕ, i < 6 → isSemiPrime (start + i)) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_semiprimes_l4044_404488


namespace NUMINAMATH_CALUDE_quadratic_roots_proof_l4044_404476

theorem quadratic_roots_proof (x₁ x₂ : ℝ) : x₁ = -1 ∧ x₂ = 6 →
  (x₁^2 - 5*x₁ - 6 = 0) ∧ (x₂^2 - 5*x₂ - 6 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_proof_l4044_404476


namespace NUMINAMATH_CALUDE_triangle_problem_l4044_404463

-- Define the triangle ABC
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Add conditions for a valid triangle if necessary
  True

-- Define the theorem
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_a : a = 8)
  (h_bc : b - c = 2)
  (h_cosA : Real.cos A = -1/4) :
  Real.sin B = (3 * Real.sqrt 15) / 16 ∧ 
  Real.cos (2 * A + π/6) = -(7 * Real.sqrt 3) / 16 - (Real.sqrt 15) / 16 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l4044_404463


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l4044_404413

theorem complex_modulus_problem (z : ℂ) (h : (1 - I) * z = 1 + I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l4044_404413


namespace NUMINAMATH_CALUDE_divisible_by_thirty_l4044_404400

theorem divisible_by_thirty (a b : ℤ) : 
  30 ∣ (a * b * (a^4 - b^4)) := by sorry

end NUMINAMATH_CALUDE_divisible_by_thirty_l4044_404400


namespace NUMINAMATH_CALUDE_cone_height_for_given_volume_and_angle_l4044_404423

/-- Represents a cone with given volume and vertex angle -/
structure Cone where
  volume : ℝ
  vertexAngle : ℝ

/-- Calculates the height of a cone given its volume and vertex angle -/
def coneHeight (c : Cone) : ℝ :=
  sorry

/-- Theorem stating that a cone with volume 19683π and vertex angle 90° has height 39 -/
theorem cone_height_for_given_volume_and_angle :
  let c : Cone := { volume := 19683 * Real.pi, vertexAngle := 90 }
  coneHeight c = 39 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_for_given_volume_and_angle_l4044_404423


namespace NUMINAMATH_CALUDE_competition_problems_l4044_404408

/-- The total number of problems in the competition. -/
def total_problems : ℕ := 71

/-- The number of problems Lukáš correctly solved. -/
def solved_problems : ℕ := 12

/-- The additional points Lukáš would have gained if he solved the last 12 problems. -/
def additional_points : ℕ := 708

/-- The sum of the first n natural numbers. -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the total number of problems in the competition. -/
theorem competition_problems :
  (sum_first_n solved_problems) +
  (sum_first_n solved_problems + additional_points) =
  sum_first_n total_problems - sum_first_n (total_problems - solved_problems) :=
by sorry

end NUMINAMATH_CALUDE_competition_problems_l4044_404408


namespace NUMINAMATH_CALUDE_complex_power_sum_l4044_404498

theorem complex_power_sum (w : ℂ) (hw : w^2 - w + 1 = 0) :
  w^102 + w^103 + w^104 + w^105 + w^106 = 2*w + 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l4044_404498


namespace NUMINAMATH_CALUDE_prob_not_all_same_five_eight_sided_dice_l4044_404480

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The number of dice rolled -/
def k : ℕ := 5

/-- The probability of not getting all the same numbers when rolling k n-sided dice -/
def prob_not_all_same (n k : ℕ) : ℚ :=
  1 - (n : ℚ) / (n ^ k : ℚ)

/-- Theorem: The probability of not getting all the same numbers when rolling 
    five fair 8-sided dice is 4095/4096 -/
theorem prob_not_all_same_five_eight_sided_dice :
  prob_not_all_same n k = 4095 / 4096 := by sorry

end NUMINAMATH_CALUDE_prob_not_all_same_five_eight_sided_dice_l4044_404480


namespace NUMINAMATH_CALUDE_range_of_f_l4044_404405

theorem range_of_f (x : ℝ) : 
  let f := fun (x : ℝ) => Real.sin x^4 - Real.sin x * Real.cos x + Real.cos x^4
  0 ≤ f x ∧ f x ≤ 9/8 ∧ 
  (∃ y : ℝ, f y = 0) ∧ 
  (∃ z : ℝ, f z = 9/8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l4044_404405


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4044_404494

theorem polynomial_simplification (x : ℝ) :
  3 * x^3 + 4 * x^2 + 2 * x + 5 - (2 * x^3 - 5 * x^2 + x - 3) + (x^3 - 2 * x^2 - 4 * x + 6) =
  2 * x^3 + 7 * x^2 - 3 * x + 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4044_404494


namespace NUMINAMATH_CALUDE_locus_is_circle_l4044_404401

/-- Given a right triangle with sides s, s, and s√2, this function represents the locus of points P 
    such that the sum of squares of distances from P to the vertices equals a. -/
def locus (s a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (A B C : ℝ × ℝ), 
    -- A, B, C form a right triangle with sides s, s, s√2
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = s^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = s^2 ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = 2*s^2 ∧
    -- Sum of squares of distances from P to vertices equals a
    (p.1 - A.1)^2 + (p.2 - A.2)^2 + 
    (p.1 - B.1)^2 + (p.2 - B.2)^2 + 
    (p.1 - C.1)^2 + (p.2 - C.2)^2 = a}

/-- The constant K dependent on the triangle's dimensions -/
def K (s : ℝ) : ℝ := 2 * s^2

/-- Theorem stating that the locus is a circle if and only if a > K -/
theorem locus_is_circle (s a : ℝ) (h_s : s > 0) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), locus s a = Metric.ball center radius ↔ a > K s :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l4044_404401


namespace NUMINAMATH_CALUDE_exists_initial_points_for_82_l4044_404417

/-- The function that calculates the number of points after one application of the procedure -/
def points_after_one_step (n : ℕ) : ℕ := 3 * n - 2

/-- The function that calculates the number of points after two applications of the procedure -/
def points_after_two_steps (n : ℕ) : ℕ := 9 * n - 8

/-- Theorem stating that there exists an initial number of points that results in 82 points after two steps -/
theorem exists_initial_points_for_82 : ∃ n : ℕ, n > 0 ∧ points_after_two_steps n = 82 := by
  sorry

end NUMINAMATH_CALUDE_exists_initial_points_for_82_l4044_404417


namespace NUMINAMATH_CALUDE_frequency_third_group_l4044_404433

theorem frequency_third_group (m : ℕ) (h1 : m ≥ 3) : 
  let total_frequency : ℝ := 1
  let third_rectangle_area : ℝ := (1 / 4) * (total_frequency - third_rectangle_area)
  let sample_size : ℕ := 100
  (third_rectangle_area * sample_size : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_frequency_third_group_l4044_404433


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l4044_404457

theorem binomial_expansion_problem (m n : ℕ) (hm : m ≠ 0) (hn : n ≥ 2) :
  (∀ k, k ∈ Finset.range (n + 1) → k ≠ 5 → Nat.choose n k ≤ Nat.choose n 5) ∧
  Nat.choose n 2 * m^2 = 9 * Nat.choose n 1 * m →
  m = 2 ∧ n = 10 ∧ ((-17)^n) % 6 = 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l4044_404457


namespace NUMINAMATH_CALUDE_cafe_order_combinations_l4044_404462

-- Define the number of menu items
def menu_items : ℕ := 12

-- Define the number of people ordering
def num_people : ℕ := 3

-- Theorem statement
theorem cafe_order_combinations :
  menu_items ^ num_people = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cafe_order_combinations_l4044_404462


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4044_404428

theorem inequality_solution_set :
  {x : ℝ | x^2 + 2*x - 3 ≥ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4044_404428


namespace NUMINAMATH_CALUDE_ruby_candies_l4044_404418

/-- The number of friends Ruby shares her candies with -/
def num_friends : ℕ := 9

/-- The number of candies each friend receives -/
def candies_per_friend : ℕ := 4

/-- The initial number of candies Ruby has -/
def initial_candies : ℕ := num_friends * candies_per_friend

theorem ruby_candies : initial_candies = 36 := by
  sorry

end NUMINAMATH_CALUDE_ruby_candies_l4044_404418


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_5_range_of_m_for_f_geq_g_set_iic_1_equiv_interval_l4044_404419

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| + 2
def g (m : ℝ) (x : ℝ) : ℝ := m * |x|

-- Theorem for part I
theorem solution_set_f_greater_than_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -1 ∨ x > 5} := by sorry

-- Theorem for part II
theorem range_of_m_for_f_geq_g :
  {m : ℝ | ∀ x, f x ≥ g m x} = Set.Iic 1 := by sorry

-- Additional helper theorem to show that Set.Iic 1 is equivalent to (-∞, 1]
theorem set_iic_1_equiv_interval :
  Set.Iic 1 = {m : ℝ | m ≤ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_5_range_of_m_for_f_geq_g_set_iic_1_equiv_interval_l4044_404419


namespace NUMINAMATH_CALUDE_probability_greater_than_four_l4044_404441

-- Define a standard six-sided die
def standardDie : Finset Nat := Finset.range 6

-- Define the probability of an event on the die
def probability (event : Finset Nat) : Rat :=
  event.card / standardDie.card

-- Define the event of rolling a number greater than 4
def greaterThanFour : Finset Nat := Finset.filter (λ x => x > 4) standardDie

-- Theorem statement
theorem probability_greater_than_four :
  probability greaterThanFour = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_greater_than_four_l4044_404441


namespace NUMINAMATH_CALUDE_fruit_basket_count_l4044_404471

/-- The number of ways to choose from n identical items -/
def chooseFromIdentical (n : ℕ) : ℕ := n + 1

/-- The number of fruit baskets with at least one fruit -/
def fruitBaskets (pears bananas : ℕ) : ℕ :=
  chooseFromIdentical pears * chooseFromIdentical bananas - 1

theorem fruit_basket_count :
  fruitBaskets 8 12 = 116 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l4044_404471


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l4044_404427

theorem logarithm_expression_equality : 
  (Real.log 243 / Real.log 3) / (Real.log 3 / Real.log 81) - 
  (Real.log 729 / Real.log 3) / (Real.log 3 / Real.log 27) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l4044_404427


namespace NUMINAMATH_CALUDE_fruit_store_problem_l4044_404483

/-- Fruit store problem -/
theorem fruit_store_problem 
  (cost_price : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (price_change : ℝ) 
  (sales_change : ℝ)
  (h1 : cost_price = 30)
  (h2 : initial_price = 40)
  (h3 : initial_sales = 400)
  (h4 : price_change = 1)
  (h5 : sales_change = 10) :
  let sales_at_price (p : ℝ) := initial_sales - (p - initial_price) * sales_change
  let profit (p : ℝ) := (p - cost_price) * (sales_at_price p)
  (
    -- 1. Monthly sales at 45 yuan/kg is 350 kilograms
    sales_at_price 45 = 350 ∧ 
    -- 2. Selling price for 5250 yuan profit is 45 or 65 yuan/kg
    (∃ p, profit p = 5250 ↔ (p = 45 ∨ p = 65)) ∧
    -- 3. Maximum profit occurs at 55 yuan/kg and is 6250 yuan
    (∀ p, profit p ≤ 6250) ∧ profit 55 = 6250
  ) := by sorry

end NUMINAMATH_CALUDE_fruit_store_problem_l4044_404483


namespace NUMINAMATH_CALUDE_distance_to_midpoint_l4044_404446

-- Define the triangle PQR
structure RightTriangle where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  is_right : PQ^2 = PR^2 + QR^2

-- Define the specific triangle given in the problem
def triangle_PQR : RightTriangle :=
  { PQ := 15
    PR := 9
    QR := 12
    is_right := by norm_num }

-- Theorem statement
theorem distance_to_midpoint (t : RightTriangle) (h : t = triangle_PQR) :
  (t.PQ / 2 : ℝ) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_midpoint_l4044_404446


namespace NUMINAMATH_CALUDE_jack_bake_sale_goal_l4044_404466

/-- The price of a brownie that allows Jack to reach his sales goal -/
def brownie_price : ℚ := by sorry

theorem jack_bake_sale_goal (num_brownies : ℕ) (num_lemon_squares : ℕ) (lemon_square_price : ℚ)
  (num_cookies : ℕ) (cookie_price : ℚ) (total_goal : ℚ) :
  num_brownies = 4 →
  num_lemon_squares = 5 →
  lemon_square_price = 2 →
  num_cookies = 7 →
  cookie_price = 4 →
  total_goal = 50 →
  num_brownies * brownie_price + num_lemon_squares * lemon_square_price + num_cookies * cookie_price = total_goal →
  brownie_price = 3 := by sorry

end NUMINAMATH_CALUDE_jack_bake_sale_goal_l4044_404466


namespace NUMINAMATH_CALUDE_unique_perpendicular_line_l4044_404491

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

structure Line3D where
  point : Point3D
  direction : Point3D

-- Define perpendicularity between a line and a plane
def isPerpendicular (l : Line3D) (p : Plane) : Prop :=
  l.direction.x * p.a + l.direction.y * p.b + l.direction.z * p.c = 0

-- State the theorem
theorem unique_perpendicular_line 
  (P : Point3D) (π : Plane) : 
  ∃! l : Line3D, l.point = P ∧ isPerpendicular l π :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_line_l4044_404491


namespace NUMINAMATH_CALUDE_tangent_lines_through_origin_l4044_404496

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

-- Theorem statement
theorem tangent_lines_through_origin :
  ∃ (x₀ : ℝ), 
    (f x₀ = x₀ * (f' x₀)) ∧ 
    ((f' 0 = 2 ∧ f 0 = 0) ∨ 
     (f' x₀ = -1/4 ∧ f x₀ = -1/4 * x₀)) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_through_origin_l4044_404496


namespace NUMINAMATH_CALUDE_squirrel_pine_cones_l4044_404412

/-- The number of pine cones the squirrel planned to eat per day -/
def planned_daily_cones : ℕ := 6

/-- The additional number of pine cones the squirrel actually ate per day -/
def additional_daily_cones : ℕ := 2

/-- The number of days earlier the pine cones were finished -/
def days_earlier : ℕ := 5

/-- The total number of pine cones stored by the squirrel -/
def total_cones : ℕ := 120

theorem squirrel_pine_cones :
  ∃ (planned_days : ℕ),
    planned_days * planned_daily_cones =
    (planned_days - days_earlier) * (planned_daily_cones + additional_daily_cones) ∧
    total_cones = planned_days * planned_daily_cones :=
by sorry

end NUMINAMATH_CALUDE_squirrel_pine_cones_l4044_404412


namespace NUMINAMATH_CALUDE_xy_max_value_l4044_404440

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 2 * y = 12) :
  x * y ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_xy_max_value_l4044_404440


namespace NUMINAMATH_CALUDE_willy_lucy_crayon_difference_l4044_404486

theorem willy_lucy_crayon_difference :
  let willy_crayons : ℕ := 1400
  let lucy_crayons : ℕ := 290
  willy_crayons - lucy_crayons = 1110 :=
by sorry

end NUMINAMATH_CALUDE_willy_lucy_crayon_difference_l4044_404486


namespace NUMINAMATH_CALUDE_double_root_values_l4044_404470

/-- A polynomial with integer coefficients of the form x^4 + a₃x³ + a₂x² + a₁x + 18 -/
def P (a₃ a₂ a₁ : ℤ) (x : ℝ) : ℝ := x^4 + a₃*x^3 + a₂*x^2 + a₁*x + 18

/-- r is a double root of P if (x - r)² divides P -/
def is_double_root (r : ℤ) (a₃ a₂ a₁ : ℤ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, P a₃ a₂ a₁ x = (x - r)^2 * q x

theorem double_root_values (a₃ a₂ a₁ : ℤ) (r : ℤ) :
  is_double_root r a₃ a₂ a₁ → r = -3 ∨ r = -1 ∨ r = 1 ∨ r = 3 := by
  sorry

end NUMINAMATH_CALUDE_double_root_values_l4044_404470
