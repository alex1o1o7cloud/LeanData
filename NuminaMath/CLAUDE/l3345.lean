import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_condition_l3345_334580

theorem divisibility_condition (m : ℕ) (h1 : m > 2022) 
  (h2 : (2022 + m) ∣ (2022 * m)) : m = 1011 ∨ m = 2022 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3345_334580


namespace NUMINAMATH_CALUDE_recycle_theorem_l3345_334567

/-- The number of cans needed to recycle into one new can -/
def cans_per_recycle : ℕ := 5

/-- The initial number of cans -/
def initial_cans : ℕ := 3125

/-- The function that calculates the number of new cans produced in each recycling round -/
def new_cans_in_round (n : ℕ) : ℕ := 
  (initial_cans / cans_per_recycle ^ n) % cans_per_recycle

/-- The total number of new cans produced through all recycling rounds -/
def total_new_cans : ℕ := 
  (Finset.range 5).sum new_cans_in_round

theorem recycle_theorem : total_new_cans = 781 := by
  sorry

end NUMINAMATH_CALUDE_recycle_theorem_l3345_334567


namespace NUMINAMATH_CALUDE_cars_with_both_features_l3345_334553

/-- Represents the car lot scenario -/
structure CarLot where
  total : Nat
  with_airbag : Nat
  with_power_windows : Nat
  with_neither : Nat

/-- Theorem stating the number of cars with both air-bag and power windows -/
theorem cars_with_both_features (lot : CarLot) 
  (h1 : lot.total = 65)
  (h2 : lot.with_airbag = 45)
  (h3 : lot.with_power_windows = 30)
  (h4 : lot.with_neither = 2) :
  lot.with_airbag + lot.with_power_windows - (lot.total - lot.with_neither) = 12 := by
  sorry

#check cars_with_both_features

end NUMINAMATH_CALUDE_cars_with_both_features_l3345_334553


namespace NUMINAMATH_CALUDE_hannah_mugs_theorem_l3345_334514

def hannah_mugs (total_mugs : ℕ) (total_colors : ℕ) (yellow_mugs : ℕ) : Prop :=
  ∃ (red_mugs blue_mugs other_mugs : ℕ),
    total_mugs = red_mugs + blue_mugs + yellow_mugs + other_mugs ∧
    blue_mugs = 3 * red_mugs ∧
    red_mugs = yellow_mugs / 2 ∧
    other_mugs = 4

theorem hannah_mugs_theorem :
  hannah_mugs 40 4 12 :=
by sorry

end NUMINAMATH_CALUDE_hannah_mugs_theorem_l3345_334514


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3345_334538

theorem coefficient_x_squared_in_expansion : 
  let expansion := (fun x : ℝ => (x - x⁻¹)^6)
  ∃ (a b c : ℝ), ∀ x : ℝ, x ≠ 0 → 
    expansion x = a*x^3 + 15*x^2 + b*x + c + (x⁻¹ * (1 + x⁻¹ * (1 + x⁻¹ * (1)))) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3345_334538


namespace NUMINAMATH_CALUDE_roots_of_composite_quadratic_l3345_334529

/-- A quadratic function with real coefficients -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the quadratic function at a given point -/
def evaluate (f : QuadraticFunction) (x : ℂ) : ℂ :=
  f.a * x^2 + f.b * x + f.c

/-- Predicate stating that a complex number is purely imaginary -/
def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Predicate stating that all roots of the equation f(x) = 0 are purely imaginary -/
def hasPurelyImaginaryRoots (f : QuadraticFunction) : Prop :=
  ∀ x : ℂ, evaluate f x = 0 → isPurelyImaginary x

/-- Theorem stating the nature of roots for f(f(x)) = 0 -/
theorem roots_of_composite_quadratic
  (f : QuadraticFunction)
  (h : hasPurelyImaginaryRoots f) :
  ∀ x : ℂ, evaluate f (evaluate f x) = 0 →
    (¬ x.im = 0) ∧ ¬ isPurelyImaginary x :=
sorry

end NUMINAMATH_CALUDE_roots_of_composite_quadratic_l3345_334529


namespace NUMINAMATH_CALUDE_equation_solution_l3345_334506

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = (-7 + Real.sqrt 105) / 4) ∧ 
    (x₂ = (-7 - Real.sqrt 105) / 4) ∧ 
    (∀ x : ℝ, (4 * x^2 + 8 * x - 5 ≠ 0) → (2 * x - 1 ≠ 0) → 
      ((3 * x - 7) / (4 * x^2 + 8 * x - 5) = x / (2 * x - 1)) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3345_334506


namespace NUMINAMATH_CALUDE_book_pages_l3345_334540

theorem book_pages (x : ℕ) (h1 : x > 0) (h2 : x + (x + 1) = 137) : x + 1 = 69 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l3345_334540


namespace NUMINAMATH_CALUDE_ad_bc_ratio_l3345_334561

-- Define the triangle ABC
structure EquilateralTriangle :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define the triangle BCD
structure IsoscelesTriangle :=
  (side : ℝ)
  (angle : ℝ)
  (side_positive : side > 0)
  (angle_value : angle = 2 * Real.pi / 3)  -- 120° in radians

-- Define the configuration
structure TriangleConfiguration :=
  (abc : EquilateralTriangle)
  (bcd : IsoscelesTriangle)
  (shared_side : abc.side = bcd.side)

-- State the theorem
theorem ad_bc_ratio (config : TriangleConfiguration) :
  ∃ (ad bc : ℝ), ad / bc = 1 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ad_bc_ratio_l3345_334561


namespace NUMINAMATH_CALUDE_g_composition_equals_1200_l3345_334523

def g (x : ℝ) : ℝ := 7 * x + 3

theorem g_composition_equals_1200 : g (g (g 3)) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_equals_1200_l3345_334523


namespace NUMINAMATH_CALUDE_triangle_side_range_l3345_334577

theorem triangle_side_range (a : ℝ) : 
  (3 : ℝ) > 0 ∧ (5 : ℝ) > 0 ∧ (1 - 2*a : ℝ) > 0 ∧
  3 + 5 > 1 - 2*a ∧
  3 + (1 - 2*a) > 5 ∧
  5 + (1 - 2*a) > 3 →
  -7/2 < a ∧ a < -1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l3345_334577


namespace NUMINAMATH_CALUDE_inequalities_solution_l3345_334588

theorem inequalities_solution :
  (∀ x : ℝ, x * (9 - x) > 0 ↔ 0 < x ∧ x < 9) ∧
  (∀ x : ℝ, 16 - x^2 ≤ 0 ↔ x ≤ -4 ∨ x ≥ 4) := by sorry

end NUMINAMATH_CALUDE_inequalities_solution_l3345_334588


namespace NUMINAMATH_CALUDE_gum_pack_size_l3345_334537

-- Define the number of cherry and grape gum pieces
def cherry_gum : ℚ := 25
def grape_gum : ℚ := 35

-- Define the number of packs of grape gum found
def grape_packs_found : ℚ := 6

-- Define the variable x as the number of pieces in a complete pack
variable (x : ℚ)

-- Define the equality condition
def equality_condition (x : ℚ) : Prop :=
  (cherry_gum - x) / grape_gum = cherry_gum / (grape_gum + grape_packs_found * x)

-- Theorem statement
theorem gum_pack_size :
  equality_condition x → x = 115 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_gum_pack_size_l3345_334537


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l3345_334582

-- Define LCM function
def LCM (a b : ℕ) : ℕ := sorry

-- Define GCF function
def GCF (a b : ℕ) : ℕ := sorry

-- Theorem statement
theorem gcf_of_lcms : GCF (LCM 9 15) (LCM 14 25) = 5 := by sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l3345_334582


namespace NUMINAMATH_CALUDE_big_al_bananas_l3345_334532

theorem big_al_bananas (a : ℕ) (h : a + 2*a + 4*a + 8*a + 16*a = 155) : 16*a = 80 := by
  sorry

end NUMINAMATH_CALUDE_big_al_bananas_l3345_334532


namespace NUMINAMATH_CALUDE_right_triangle_area_and_height_l3345_334521

theorem right_triangle_area_and_height :
  let a : ℝ := 9
  let b : ℝ := 40
  let c : ℝ := 41
  -- Condition: it's a right triangle
  a ^ 2 + b ^ 2 = c ^ 2 →
  -- Prove the area
  (1 / 2 : ℝ) * a * b = 180 ∧
  -- Prove the height
  (2 * ((1 / 2 : ℝ) * a * b)) / c = 360 / 41 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_and_height_l3345_334521


namespace NUMINAMATH_CALUDE_log_product_equals_five_thirds_l3345_334555

theorem log_product_equals_five_thirds :
  Real.log 9 / Real.log 8 * (Real.log 32 / Real.log 9) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_five_thirds_l3345_334555


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l3345_334559

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l3345_334559


namespace NUMINAMATH_CALUDE_company_employees_l3345_334593

/-- The number of employees in a company satisfying certain conditions. -/
theorem company_employees : 
  ∀ (total_females : ℕ) 
    (advanced_degrees : ℕ) 
    (males_college_only : ℕ) 
    (females_advanced : ℕ),
  total_females = 110 →
  advanced_degrees = 90 →
  males_college_only = 35 →
  females_advanced = 55 →
  ∃ (total_employees : ℕ),
    total_employees = 180 ∧
    total_employees = advanced_degrees + (males_college_only + (total_females - females_advanced)) :=
by sorry

end NUMINAMATH_CALUDE_company_employees_l3345_334593


namespace NUMINAMATH_CALUDE_no_such_function_exists_l3345_334515

theorem no_such_function_exists : ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, (f^[f x]) x = x + 1 := by
  sorry

#check no_such_function_exists

end NUMINAMATH_CALUDE_no_such_function_exists_l3345_334515


namespace NUMINAMATH_CALUDE_smallest_four_digit_different_digits_l3345_334557

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

theorem smallest_four_digit_different_digits :
  ∀ n : ℕ, is_four_digit n → has_different_digits n → 1023 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_different_digits_l3345_334557


namespace NUMINAMATH_CALUDE_cable_section_length_l3345_334505

theorem cable_section_length (total_length : ℝ) (sections_kept : ℕ) : 
  total_length = 1000 ∧ sections_kept = 15 →
  ∃ (section_length : ℝ),
    section_length > 0 ∧
    (total_length / section_length : ℝ) * (3/4) * (1/2) = sections_kept ∧
    section_length = 25 := by
  sorry

end NUMINAMATH_CALUDE_cable_section_length_l3345_334505


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l3345_334572

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that the sum of their common ratios is 3. -/
theorem sum_of_common_ratios_is_three
  (k p r : ℝ)
  (h_nonconstant : k ≠ 0)
  (h_different_ratios : p ≠ r)
  (h_condition : k * p^2 - k * r^2 = 3 * (k * p - k * r)) :
  p + r = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l3345_334572


namespace NUMINAMATH_CALUDE_puppy_feeding_theorem_l3345_334511

/-- Given the number of puppies, portions of formula, and days, 
    calculates the number of times each puppy should be fed per day. -/
def feeding_frequency (puppies : ℕ) (portions : ℕ) (days : ℕ) : ℕ :=
  (portions / days) / puppies

/-- Proves that for 7 puppies, 105 portions, and 5 days, 
    the feeding frequency is 3 times per day. -/
theorem puppy_feeding_theorem :
  feeding_frequency 7 105 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_puppy_feeding_theorem_l3345_334511


namespace NUMINAMATH_CALUDE_retailer_profit_percent_l3345_334571

/-- Calculates the profit percent for a retailer given the purchase price, overhead expenses, and selling price. -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- The profit percent of the retailer is approximately 22.45% -/
theorem retailer_profit_percent :
  let ε := 0.01
  ∃ (x : ℚ), abs (x - profit_percent 225 20 300) < ε ∧ abs (x - 22.45) < ε :=
by sorry

end NUMINAMATH_CALUDE_retailer_profit_percent_l3345_334571


namespace NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l3345_334504

theorem divisibility_implies_multiple_of_three (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 2^n + 1) : 
  3 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l3345_334504


namespace NUMINAMATH_CALUDE_min_distance_complex_unit_circle_l3345_334597

theorem min_distance_complex_unit_circle (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ w : ℂ, Complex.abs w = 1 → Complex.abs (w + 4*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_unit_circle_l3345_334597


namespace NUMINAMATH_CALUDE_initial_marbles_l3345_334595

theorem initial_marbles (M : ℚ) : 
  (2 / 5 : ℚ) * M = 30 →
  (1 / 2 : ℚ) * ((2 / 5 : ℚ) * M) = 15 →
  M = 75 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_l3345_334595


namespace NUMINAMATH_CALUDE_planes_perpendicular_from_line_l3345_334549

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_from_line (a : Line) (M N : Plane) :
  perpendicular a M → parallel a N → perpendicularPlanes N M :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_from_line_l3345_334549


namespace NUMINAMATH_CALUDE_marks_percentage_raise_l3345_334585

/-- Calculates the percentage raise Mark received at his job -/
theorem marks_percentage_raise
  (original_hourly_rate : ℚ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (old_weekly_bills : ℚ)
  (new_weekly_expense : ℚ)
  (new_leftover_amount : ℚ)
  (h1 : original_hourly_rate = 40)
  (h2 : hours_per_day = 8)
  (h3 : days_per_week = 5)
  (h4 : old_weekly_bills = 600)
  (h5 : new_weekly_expense = 100)
  (h6 : new_leftover_amount = 980) :
  (new_leftover_amount + old_weekly_bills + new_weekly_expense - 
   (original_hourly_rate * hours_per_day * days_per_week)) / 
  (original_hourly_rate * hours_per_day * days_per_week) = 1/20 :=
by sorry

end NUMINAMATH_CALUDE_marks_percentage_raise_l3345_334585


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l3345_334578

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 900 → (n - 2) * 180 = angle_sum → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l3345_334578


namespace NUMINAMATH_CALUDE_race_outcomes_eq_210_l3345_334531

/-- The number of participants in the race -/
def num_participants : ℕ := 7

/-- The number of podium positions (1st, 2nd, 3rd) -/
def podium_positions : ℕ := 3

/-- Calculates the number of permutations of k elements from n elements -/
def permutations (n k : ℕ) : ℕ :=
  if k > n then 0
  else List.range k |>.foldl (fun acc i => acc * (n - i)) 1

/-- The number of different 1st-2nd-3rd place outcomes in a race with no ties -/
def race_outcomes : ℕ := permutations num_participants podium_positions

/-- Theorem: The number of different 1st-2nd-3rd place outcomes in a race
    with 7 participants and no ties is equal to 210 -/
theorem race_outcomes_eq_210 : race_outcomes = 210 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_eq_210_l3345_334531


namespace NUMINAMATH_CALUDE_percentage_equality_l3345_334558

theorem percentage_equality (x : ℝ) : (80 / 100 * 600 = 50 / 100 * x) → x = 960 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l3345_334558


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_existence_l3345_334581

theorem arithmetic_and_geometric_sequences_existence :
  ∃ (a b c : ℝ) (d r : ℝ),
    d ≠ 0 ∧ r ≠ 0 ∧ r ≠ 1 ∧
    (b - a = d ∧ c - b = d) ∧
    (∃ (x y : ℝ), x * r = y ∧ y * r = a ∧ a * r = b ∧ b * r = c) ∧
    ((a * r = b ∧ b * r = c) ∨ (b * r = a ∧ a * r = c) ∨ (c * r = a ∧ a * r = b)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_existence_l3345_334581


namespace NUMINAMATH_CALUDE_urn_problem_l3345_334569

def urn1_green : ℚ := 5
def urn1_blue : ℚ := 7
def urn2_green : ℚ := 20
def urn1_total : ℚ := urn1_green + urn1_blue
def same_color_prob : ℚ := 62/100

theorem urn_problem (M : ℚ) :
  (urn1_green / urn1_total) * (urn2_green / (urn2_green + M)) +
  (urn1_blue / urn1_total) * (M / (urn2_green + M)) = same_color_prob →
  M = 610/1657 := by
sorry

end NUMINAMATH_CALUDE_urn_problem_l3345_334569


namespace NUMINAMATH_CALUDE_output_is_72_l3345_334594

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 38 then step1 * 2 else step1 - 10

theorem output_is_72 : function_machine 12 = 72 := by
  sorry

end NUMINAMATH_CALUDE_output_is_72_l3345_334594


namespace NUMINAMATH_CALUDE_harry_snails_collection_l3345_334534

/-- Represents the number of sea stars Harry collected initially -/
def sea_stars : ℕ := 34

/-- Represents the number of seashells Harry collected initially -/
def seashells : ℕ := 21

/-- Represents the total number of items Harry had at the end of his walk -/
def total_items_left : ℕ := 59

/-- Represents the number of sea creatures Harry lost during his walk -/
def lost_sea_creatures : ℕ := 25

/-- Represents the number of snails Harry collected initially -/
def snails_collected : ℕ := total_items_left - (sea_stars + seashells - lost_sea_creatures)

theorem harry_snails_collection :
  snails_collected = 29 :=
sorry

end NUMINAMATH_CALUDE_harry_snails_collection_l3345_334534


namespace NUMINAMATH_CALUDE_coffee_cost_l3345_334560

theorem coffee_cost (sandwich_cost coffee_cost : ℕ) : 
  (3 * sandwich_cost + 2 * coffee_cost = 630) →
  (2 * sandwich_cost + 3 * coffee_cost = 690) →
  coffee_cost = 162 := by
sorry

end NUMINAMATH_CALUDE_coffee_cost_l3345_334560


namespace NUMINAMATH_CALUDE_weeks_to_afford_laptop_l3345_334518

/-- The minimum number of whole weeks needed to afford a laptop -/
def weeks_needed (laptop_cost birthday_money weekly_earnings : ℕ) : ℕ :=
  (laptop_cost - birthday_money + weekly_earnings - 1) / weekly_earnings

/-- Proof that 34 weeks are needed to afford the laptop -/
theorem weeks_to_afford_laptop :
  weeks_needed 800 125 20 = 34 := by
  sorry

end NUMINAMATH_CALUDE_weeks_to_afford_laptop_l3345_334518


namespace NUMINAMATH_CALUDE_special_line_equation_l3345_334548

/-- A line passing through a point and intersecting coordinate axes at points with negative reciprocal intercepts -/
structure SpecialLine where
  a : ℝ × ℝ  -- The point A that the line passes through
  eq : ℝ → ℝ → Prop  -- The equation of the line

/-- The condition for the line to have negative reciprocal intercepts -/
def hasNegativeReciprocalIntercepts (l : SpecialLine) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (l.eq k 0 ∧ l.eq 0 (-k) ∨ l.eq (-k) 0 ∧ l.eq 0 k)

/-- The main theorem stating the equation of the special line -/
theorem special_line_equation (l : SpecialLine) 
    (h1 : l.a = (5, 2))
    (h2 : hasNegativeReciprocalIntercepts l) :
    (∀ x y, l.eq x y ↔ 2*x - 5*y = -8) ∨
    (∀ x y, l.eq x y ↔ x - y = 3) := by
  sorry


end NUMINAMATH_CALUDE_special_line_equation_l3345_334548


namespace NUMINAMATH_CALUDE_christine_travel_time_l3345_334590

/-- Given Christine's travel scenario, prove the time she wandered. -/
theorem christine_travel_time (speed : ℝ) (distance : ℝ) (h1 : speed = 20) (h2 : distance = 80) :
  distance / speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_christine_travel_time_l3345_334590


namespace NUMINAMATH_CALUDE_lcm_of_5_6_8_9_l3345_334535

theorem lcm_of_5_6_8_9 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_8_9_l3345_334535


namespace NUMINAMATH_CALUDE_spelling_contest_problem_l3345_334503

theorem spelling_contest_problem (drew_wrong carla_correct total : ℕ) 
  (h1 : drew_wrong = 6)
  (h2 : carla_correct = 14)
  (h3 : total = 52)
  (h4 : 2 * drew_wrong = carla_correct + (total - (carla_correct + drew_wrong + (total - (2 * drew_wrong + carla_correct))))) :
  total - (2 * drew_wrong + carla_correct) = 20 := by
  sorry

end NUMINAMATH_CALUDE_spelling_contest_problem_l3345_334503


namespace NUMINAMATH_CALUDE_tomatoes_for_sale_tuesday_l3345_334550

/-- Calculates the amount of tomatoes ready for sale on Tuesday given specific conditions --/
theorem tomatoes_for_sale_tuesday 
  (initial_shipment : ℝ)
  (saturday_selling_rate : ℝ)
  (sunday_spoilage_rate : ℝ)
  (monday_shipment_multiplier : ℝ)
  (monday_selling_rate : ℝ)
  (tuesday_spoilage_rate : ℝ)
  (h1 : initial_shipment = 1000)
  (h2 : saturday_selling_rate = 0.6)
  (h3 : sunday_spoilage_rate = 0.2)
  (h4 : monday_shipment_multiplier = 1.5)
  (h5 : monday_selling_rate = 0.4)
  (h6 : tuesday_spoilage_rate = 0.15) :
  ∃ (tomatoes_tuesday : ℝ), tomatoes_tuesday = 928.2 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_for_sale_tuesday_l3345_334550


namespace NUMINAMATH_CALUDE_bees_in_hive_l3345_334596

/-- The total number of bees in a hive after more bees fly in -/
def total_bees (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: The total number of bees is 24 when there are initially 16 bees and 8 more fly in -/
theorem bees_in_hive : total_bees 16 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_bees_in_hive_l3345_334596


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3345_334530

/-- The complex number z = (2-i)/(1+i) is located in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - Complex.I) / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3345_334530


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3345_334562

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let e := (2 * Real.sqrt 3) / 3
  let line_distance := Real.sqrt 3 / 2
  let eccentricity_eq := e^2 = 1 + b^2 / a^2
  let distance_eq := (a * b)^2 / (a^2 + b^2) = line_distance^2
  eccentricity_eq ∧ distance_eq →
  a^2 = 3 ∧ b^2 = 1 :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l3345_334562


namespace NUMINAMATH_CALUDE_janes_number_exists_and_unique_l3345_334563

theorem janes_number_exists_and_unique :
  ∃! n : ℕ,
    200 ∣ n ∧
    45 ∣ n ∧
    500 < n ∧
    n < 2500 ∧
    Even n :=
by
  sorry

end NUMINAMATH_CALUDE_janes_number_exists_and_unique_l3345_334563


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3345_334587

theorem quadratic_equation_solution (x : ℝ) : x^2 + 2*x - 8 = 0 ↔ x = -4 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3345_334587


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3345_334564

theorem fraction_to_decimal : (17 : ℚ) / 625 = 0.0272 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3345_334564


namespace NUMINAMATH_CALUDE_monthly_sales_fraction_l3345_334544

theorem monthly_sales_fraction (december_sales : ℝ) (monthly_sales : ℝ) (total_sales : ℝ) :
  december_sales = 6 * monthly_sales →
  december_sales = 0.35294117647058826 * total_sales →
  monthly_sales = (1 / 17) * total_sales := by
sorry

end NUMINAMATH_CALUDE_monthly_sales_fraction_l3345_334544


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3345_334519

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 →
  q^3 - 8*q^2 + 10*q - 3 = 0 →
  r^3 - 8*r^2 + 10*r - 3 = 0 →
  p/(q*r + 2) + q/(p*r + 2) + r/(p*q + 2) = 4 + 9/20 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3345_334519


namespace NUMINAMATH_CALUDE_equation_solutions_l3345_334526

theorem equation_solutions :
  (∃ x : ℝ, 2 * x^3 = 16 ∧ x = 2) ∧
  (∃ x₁ x₂ : ℝ, (x₁ - 1)^2 = 4 ∧ (x₂ - 1)^2 = 4 ∧ x₁ = 3 ∧ x₂ = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3345_334526


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3345_334551

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3345_334551


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3345_334583

theorem arithmetic_mean_problem (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 89 → a = 34 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3345_334583


namespace NUMINAMATH_CALUDE_intersection_line_equation_l3345_334528

/-- Given two lines that intersect at (2, 3), prove that the line passing through
    the points defined by their coefficients has the equation 2x + 3y - 1 = 0 -/
theorem intersection_line_equation (A₁ B₁ A₂ B₂ : ℝ) :
  (A₁ * 2 + B₁ * 3 = 1) →
  (A₂ * 2 + B₂ * 3 = 1) →
  ∃ (k : ℝ), k ≠ 0 ∧ (A₁ - A₂) * 2 + (B₁ - B₂) * 3 = k * (2 * (A₁ - A₂) + 3 * (B₁ - B₂) - 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l3345_334528


namespace NUMINAMATH_CALUDE_pythagorean_sum_inequality_l3345_334575

theorem pythagorean_sum_inequality (a b c x y z : ℕ) 
  (h1 : a^2 + b^2 = c^2) (h2 : x^2 + y^2 = z^2) :
  (a + x)^2 + (b + y)^2 ≤ (c + z)^2 ∧ 
  ((a + x)^2 + (b + y)^2 = (c + z)^2 ↔ (a * z = c * x ∧ b * z = c * y)) :=
sorry

end NUMINAMATH_CALUDE_pythagorean_sum_inequality_l3345_334575


namespace NUMINAMATH_CALUDE_locus_of_vertex_C_l3345_334522

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define an equilateral triangle
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def IsEquilateral (t : EquilateralTriangle) : Prop :=
  let d_AB := ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)^(1/2)
  let d_BC := ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)^(1/2)
  let d_CA := ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)^(1/2)
  d_AB = d_BC ∧ d_BC = d_CA

-- Define the theorem
theorem locus_of_vertex_C (c : Circle) (t : EquilateralTriangle) :
  IsEquilateral t →
  PointOnCircle c t.A →
  PointOnCircle c t.B →
  ∃ c1 c2 : Circle,
    c1.center = c.center ∧
    c2.center = c.center ∧
    c1.radius = c.radius ∧
    c2.radius = c.radius ∧
    PointOnCircle c1 t.C ∨ PointOnCircle c2 t.C :=
by sorry

end NUMINAMATH_CALUDE_locus_of_vertex_C_l3345_334522


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l3345_334512

theorem consecutive_integers_product (x : ℕ) :
  x > 0 ∧ x * (x + 1) = 812 → (x + 1)^2 - x = 813 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l3345_334512


namespace NUMINAMATH_CALUDE_three_A_students_l3345_334539

-- Define the students
inductive Student : Type
| Edward : Student
| Fiona : Student
| George : Student
| Hannah : Student
| Ian : Student

-- Define a predicate for getting an A
def got_A : Student → Prop := sorry

-- Define the statements
axiom Edward_statement : got_A Student.Edward → got_A Student.Fiona
axiom Fiona_statement : got_A Student.Fiona → got_A Student.George
axiom George_statement : got_A Student.George → got_A Student.Hannah
axiom Hannah_statement : got_A Student.Hannah → got_A Student.Ian

-- Define the condition that exactly three students got an A
axiom three_A : ∃ (s1 s2 s3 : Student), 
  (s1 ≠ s2 ∧ s1 ≠ s3 ∧ s2 ≠ s3) ∧
  got_A s1 ∧ got_A s2 ∧ got_A s3 ∧
  (∀ (s : Student), got_A s → (s = s1 ∨ s = s2 ∨ s = s3))

-- The theorem to prove
theorem three_A_students : 
  got_A Student.George ∧ got_A Student.Hannah ∧ got_A Student.Ian ∧
  ¬got_A Student.Edward ∧ ¬got_A Student.Fiona :=
sorry

end NUMINAMATH_CALUDE_three_A_students_l3345_334539


namespace NUMINAMATH_CALUDE_clearance_sale_gain_percentage_l3345_334525

-- Define the original selling price
def original_selling_price : ℝ := 30

-- Define the original gain percentage
def original_gain_percentage : ℝ := 20

-- Define the discount percentage during clearance sale
def clearance_discount_percentage : ℝ := 10

-- Theorem statement
theorem clearance_sale_gain_percentage :
  let cost_price := original_selling_price / (1 + original_gain_percentage / 100)
  let discounted_price := original_selling_price * (1 - clearance_discount_percentage / 100)
  let new_gain := discounted_price - cost_price
  let new_gain_percentage := (new_gain / cost_price) * 100
  new_gain_percentage = 8 := by sorry

end NUMINAMATH_CALUDE_clearance_sale_gain_percentage_l3345_334525


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l3345_334570

theorem sqrt_equation_solutions (x : ℝ) : 
  Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l3345_334570


namespace NUMINAMATH_CALUDE_steak_entree_cost_l3345_334502

theorem steak_entree_cost 
  (total_guests : ℕ) 
  (chicken_cost : ℕ) 
  (total_budget : ℕ) 
  (h1 : total_guests = 80)
  (h2 : chicken_cost = 18)
  (h3 : total_budget = 1860)
  : (total_budget - (total_guests / 4 * chicken_cost)) / (3 * total_guests / 4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_steak_entree_cost_l3345_334502


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l3345_334598

theorem sqrt_equation_solutions :
  {x : ℝ | x ≥ 2 ∧ Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 10 - 8 * Real.sqrt (x - 2)) = 2} =
  {8.25, 22.25} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l3345_334598


namespace NUMINAMATH_CALUDE_books_count_proof_l3345_334584

/-- Given a ratio of items and a total count, calculates the number of items for a specific part of the ratio. -/
def calculate_items (ratio : List Nat) (total_items : Nat) (part_index : Nat) : Nat :=
  let total_parts := ratio.sum
  let items_per_part := total_items / total_parts
  items_per_part * (ratio.get! part_index)

/-- Proves that given the ratio 7:3:2 for books, pens, and notebooks, and a total of 600 items, the number of books is 350. -/
theorem books_count_proof :
  let ratio := [7, 3, 2]
  let total_items := 600
  let books_index := 0
  calculate_items ratio total_items books_index = 350 := by
  sorry

end NUMINAMATH_CALUDE_books_count_proof_l3345_334584


namespace NUMINAMATH_CALUDE_square_puzzle_l3345_334599

theorem square_puzzle (n : ℕ) 
  (h1 : n^2 + 20 = (n + 1)^2 - 9) : n = 14 ∧ n^2 + 20 = 216 := by
  sorry

#check square_puzzle

end NUMINAMATH_CALUDE_square_puzzle_l3345_334599


namespace NUMINAMATH_CALUDE_smallest_y_with_given_remainders_l3345_334527

theorem smallest_y_with_given_remainders : ∃ y : ℕ, 
  y > 0 ∧ 
  y % 3 = 2 ∧ 
  y % 7 = 6 ∧ 
  y % 8 = 7 ∧ 
  ∀ z : ℕ, z > 0 ∧ z % 3 = 2 ∧ z % 7 = 6 ∧ z % 8 = 7 → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_with_given_remainders_l3345_334527


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3345_334517

theorem complex_equation_sum (a b : ℝ) : 
  (a + 2 * Complex.I) / Complex.I = b + Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3345_334517


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l3345_334579

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 200 →
  a * b + b * c + c * d ≤ 10000 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l3345_334579


namespace NUMINAMATH_CALUDE_ted_losses_l3345_334533

/-- Represents a player in the game --/
inductive Player
| Carl
| James
| Saif
| Ted

/-- Records the number of wins and losses for a player --/
structure PlayerRecord where
  wins : Nat
  losses : Nat

/-- Represents the game results for all players --/
def GameResults := Player → PlayerRecord

theorem ted_losses (results : GameResults) :
  (results Player.Carl).wins = 5 ∧
  (results Player.Carl).losses = 0 ∧
  (results Player.James).wins = 4 ∧
  (results Player.James).losses = 2 ∧
  (results Player.Saif).wins = 1 ∧
  (results Player.Saif).losses = 6 ∧
  (results Player.Ted).wins = 4 ∧
  (∀ p : Player, (results p).wins + (results p).losses = 
    (results Player.Carl).wins + (results Player.James).wins + 
    (results Player.Saif).wins + (results Player.Ted).wins) →
  (results Player.Ted).losses = 6 := by
  sorry

end NUMINAMATH_CALUDE_ted_losses_l3345_334533


namespace NUMINAMATH_CALUDE_no_such_polynomials_l3345_334500

/-- A polynomial is a perfect square if it's the square of another non-constant polynomial -/
def IsPerfectSquare (p : Polynomial ℝ) : Prop :=
  ∃ q : Polynomial ℝ, q.degree > 0 ∧ p = q^2

theorem no_such_polynomials :
  ¬∃ (f g : Polynomial ℝ),
    f.degree > 0 ∧ g.degree > 0 ∧
    ¬IsPerfectSquare f ∧
    ¬IsPerfectSquare g ∧
    IsPerfectSquare (f.comp g) ∧
    IsPerfectSquare (g.comp f) :=
by sorry

end NUMINAMATH_CALUDE_no_such_polynomials_l3345_334500


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3345_334509

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) : 
  A = 400 * Real.pi → d = 40 → A = Real.pi * (d / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3345_334509


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l3345_334574

theorem initial_markup_percentage (initial_price : ℝ) (additional_increase : ℝ) : 
  initial_price = 45 →
  additional_increase = 5 →
  initial_price + additional_increase = 2 * (initial_price - (initial_price - (initial_price / (1 + 8)))) →
  (initial_price - (initial_price / (1 + 8))) / (initial_price / (1 + 8)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l3345_334574


namespace NUMINAMATH_CALUDE_probability_three_hearts_is_correct_l3345_334565

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of hearts in a standard deck -/
def heartsCount : ℕ := 13

/-- Calculates the probability of drawing three hearts in a row from a standard deck without replacement -/
def probabilityThreeHearts : ℚ :=
  (heartsCount : ℚ) / deckSize *
  ((heartsCount - 1) : ℚ) / (deckSize - 1) *
  ((heartsCount - 2) : ℚ) / (deckSize - 2)

/-- Theorem stating that the probability of drawing three hearts in a row is 26/2025 -/
theorem probability_three_hearts_is_correct :
  probabilityThreeHearts = 26 / 2025 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_hearts_is_correct_l3345_334565


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l3345_334501

/-- The surface area of a cuboid with given dimensions -/
def cuboidSurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

/-- Theorem: The surface area of a cuboid with edges 4 cm, 5 cm, and 6 cm is 148 cm² -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 4 5 6 = 148 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l3345_334501


namespace NUMINAMATH_CALUDE_sum_of_combinations_l3345_334507

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem sum_of_combinations : 
  (∀ m n : ℕ, binomial m n + binomial (m - 1) n = binomial m (n + 1)) →
  (binomial 3 3 + binomial 4 3 + binomial 5 3 + binomial 6 3 + 
   binomial 7 3 + binomial 8 3 + binomial 9 3 + binomial 10 3 = 330) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_combinations_l3345_334507


namespace NUMINAMATH_CALUDE_angle_trigonometry_l3345_334543

open Real

-- Define the angle θ
variable (θ : ℝ)

-- Define the condition that the terminal side of θ lies on y = 2x (x ≥ 0)
def terminal_side_condition (θ : ℝ) : Prop :=
  ∃ (x : ℝ), x ≥ 0 ∧ tan θ = 2

-- Theorem statement
theorem angle_trigonometry (h : terminal_side_condition θ) :
  (tan θ = 2) ∧
  ((2 * cos θ + 3 * sin θ) / (cos θ - 3 * sin θ) + sin θ * cos θ = -6/5) := by
  sorry

end NUMINAMATH_CALUDE_angle_trigonometry_l3345_334543


namespace NUMINAMATH_CALUDE_inequality_proof_l3345_334576

theorem inequality_proof (x y : ℝ) : 2^(-Real.cos x^2) + 2^(-Real.sin x^2) ≥ Real.sin y + Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3345_334576


namespace NUMINAMATH_CALUDE_expression_evaluation_l3345_334510

theorem expression_evaluation : (18 * 3 + 6) / (6 - 3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3345_334510


namespace NUMINAMATH_CALUDE_inequality_solution_l3345_334568

theorem inequality_solution (x m : ℝ) : 
  (x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0) → 
  (∀ x, x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0 → 2*x^2 - 9*x + m < 0) → 
  m < 9 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3345_334568


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3345_334554

/-- A geometric sequence with first term 3 and specific arithmetic property -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) ∧
  a 1 = 3 ∧
  ∃ d : ℝ, 2 * a 2 = 4 * a 1 + d ∧ a 3 = 2 * a 2 + d

theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 3 + a 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3345_334554


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_l3345_334556

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_l3345_334556


namespace NUMINAMATH_CALUDE_cookie_box_duration_l3345_334536

/-- Given a box of cookies and daily consumption, calculate how many days the box will last -/
def cookiesDuration (totalCookies : ℕ) (oldestSonCookies : ℕ) (youngestSonCookies : ℕ) : ℕ :=
  totalCookies / (oldestSonCookies + youngestSonCookies)

/-- Prove that a box of 54 cookies lasts 9 days when 4 cookies are given to the oldest son
    and 2 cookies are given to the youngest son daily -/
theorem cookie_box_duration :
  cookiesDuration 54 4 2 = 9 := by
  sorry

#eval cookiesDuration 54 4 2

end NUMINAMATH_CALUDE_cookie_box_duration_l3345_334536


namespace NUMINAMATH_CALUDE_tank_width_is_twelve_l3345_334516

/-- Represents the dimensions and plastering cost of a tank. -/
structure Tank where
  length : ℝ
  depth : ℝ
  width : ℝ
  plasteringRate : ℝ
  totalCost : ℝ

/-- Calculates the total surface area of the tank. -/
def surfaceArea (t : Tank) : ℝ :=
  2 * (t.length * t.depth) + 2 * (t.width * t.depth) + (t.length * t.width)

/-- Theorem stating that for a tank with given dimensions and plastering cost,
    the width is 12 meters. -/
theorem tank_width_is_twelve (t : Tank)
  (h1 : t.length = 25)
  (h2 : t.depth = 6)
  (h3 : t.plasteringRate = 0.25)
  (h4 : t.totalCost = 186)
  (h5 : t.totalCost = t.plasteringRate * surfaceArea t) :
  t.width = 12 := by
  sorry

#check tank_width_is_twelve

end NUMINAMATH_CALUDE_tank_width_is_twelve_l3345_334516


namespace NUMINAMATH_CALUDE_half_radius_circle_y_l3345_334547

-- Define the circles
def circle_x : Real → Prop := λ r => 2 * Real.pi * r = 14 * Real.pi
def circle_y : Real → Prop := λ r => True  -- We don't have specific information about y's circumference

-- Theorem statement
theorem half_radius_circle_y : 
  ∃ (rx ry : Real), 
    circle_x rx ∧ 
    circle_y ry ∧ 
    (Real.pi * rx^2 = Real.pi * ry^2) ∧  -- Same area
    (ry / 2 = 3.5) := by
  sorry

end NUMINAMATH_CALUDE_half_radius_circle_y_l3345_334547


namespace NUMINAMATH_CALUDE_compare_negative_roots_l3345_334541

theorem compare_negative_roots : -3 * Real.sqrt 3 > -2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_roots_l3345_334541


namespace NUMINAMATH_CALUDE_equation_solutions_l3345_334508

-- Define the equation
def equation (a x : ℝ) : Prop :=
  ((1 - x^2)^2 + 2*a^2 + 5*a)^7 - ((3*a + 2)*(1 - x^2) + 3)^7 = 
  5 - 2*a - (3*a + 2)*x^2 - 2*a^2 - (1 - x^2)^2

-- Define the interval
def in_interval (x : ℝ) : Prop :=
  -Real.sqrt 6 / 2 ≤ x ∧ x ≤ Real.sqrt 2

-- Define the condition for two distinct solutions
def has_two_distinct_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ in_interval x₁ ∧ in_interval x₂ ∧ 
  equation a x₁ ∧ equation a x₂

-- State the theorem
theorem equation_solutions :
  ∀ a : ℝ, has_two_distinct_solutions a ↔ 
  (0.25 ≤ a ∧ a < 1) ∨ (-3.5 ≤ a ∧ a < -2) := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3345_334508


namespace NUMINAMATH_CALUDE_return_flight_speed_l3345_334573

/-- Proves that given a round trip flight with specified conditions, the return flight speed is 500 mph -/
theorem return_flight_speed 
  (total_distance : ℝ) 
  (outbound_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : total_distance = 3000) 
  (h2 : outbound_speed = 300) 
  (h3 : total_time = 8) : 
  (total_distance / 2) / (total_time - (total_distance / 2) / outbound_speed) = 500 := by
  sorry

#check return_flight_speed

end NUMINAMATH_CALUDE_return_flight_speed_l3345_334573


namespace NUMINAMATH_CALUDE_jar_weight_theorem_l3345_334552

theorem jar_weight_theorem (jar_weight : ℝ) (full_weight : ℝ) 
  (h1 : jar_weight = 0.1 * full_weight)
  (h2 : 0 < full_weight) :
  let remaining_fraction : ℝ := 0.5555555555555556
  let remaining_weight := jar_weight + remaining_fraction * (full_weight - jar_weight)
  remaining_weight / full_weight = 0.6 := by sorry

end NUMINAMATH_CALUDE_jar_weight_theorem_l3345_334552


namespace NUMINAMATH_CALUDE_greatest_multiple_less_than_700_l3345_334520

theorem greatest_multiple_less_than_700 : ∃ n : ℕ, n = 680 ∧ 
  (∀ m : ℕ, m < 700 ∧ 5 ∣ m ∧ 4 ∣ m → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_less_than_700_l3345_334520


namespace NUMINAMATH_CALUDE_point_N_coordinates_l3345_334586

def M : ℝ × ℝ := (0, -1)

def N : ℝ × ℝ → Prop := fun p => p.1 - p.2 + 1 = 0

def perpendicular (p q r s : ℝ × ℝ) : Prop :=
  (q.1 - p.1) * (s.1 - r.1) + (q.2 - p.2) * (s.2 - r.2) = 0

def line_x_plus_2y_minus_3 (p : ℝ × ℝ) : Prop :=
  p.1 + 2 * p.2 - 3 = 0

theorem point_N_coordinates :
  ∃ n : ℝ × ℝ, N n ∧ perpendicular M n (0, 0) (1, -2) ∧ n = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_point_N_coordinates_l3345_334586


namespace NUMINAMATH_CALUDE_no_cover_with_changed_tiles_l3345_334589

/-- Represents a rectangular floor -/
structure Floor :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a tile configuration -/
structure TileConfig :=
  (twoBytwo : ℕ)  -- number of 2x2 tiles
  (fourByOne : ℕ) -- number of 4x1 tiles

/-- Predicate to check if a floor can be covered by a given tile configuration -/
def canCover (f : Floor) (tc : TileConfig) : Prop :=
  4 * tc.twoBytwo + 4 * tc.fourByOne = f.length * f.width

/-- Main theorem: If a floor can be covered by a tile configuration,
    it cannot be covered by changing the number of tiles by ±1 for each type -/
theorem no_cover_with_changed_tiles (f : Floor) (tc : TileConfig) :
  canCover f tc →
  ¬(canCover f { twoBytwo := tc.twoBytwo + 1, fourByOne := tc.fourByOne - 1 } ∨
    canCover f { twoBytwo := tc.twoBytwo - 1, fourByOne := tc.fourByOne + 1 }) :=
by
  sorry

#check no_cover_with_changed_tiles

end NUMINAMATH_CALUDE_no_cover_with_changed_tiles_l3345_334589


namespace NUMINAMATH_CALUDE_product_of_cube_and_square_l3345_334545

theorem product_of_cube_and_square (x : ℝ) : 2 * x^3 * (-3 * x)^2 = 18 * x^5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cube_and_square_l3345_334545


namespace NUMINAMATH_CALUDE_negate_sum_diff_l3345_334546

theorem negate_sum_diff (a b c : ℝ) : -(a - b + c) = -a + b - c := by sorry

end NUMINAMATH_CALUDE_negate_sum_diff_l3345_334546


namespace NUMINAMATH_CALUDE_problem_statement_l3345_334592

theorem problem_statement :
  (∃ n : ℤ, 15 = 3 * n) ∧
  (∃ m : ℤ, 121 = 11 * m) ∧ (¬ ∃ k : ℤ, 60 = 11 * k) ∧
  (∃ p : ℤ, 63 = 7 * p) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3345_334592


namespace NUMINAMATH_CALUDE_largest_coefficient_term_l3345_334591

theorem largest_coefficient_term (n : ℕ+) :
  ∀ k : ℕ, k ≠ n + 1 →
    Nat.choose (2 * n) (n + 1) ≥ Nat.choose (2 * n) k := by
  sorry

end NUMINAMATH_CALUDE_largest_coefficient_term_l3345_334591


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3345_334566

theorem arithmetic_calculation : 12 / 4 - 3 - 16 + 4 * 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3345_334566


namespace NUMINAMATH_CALUDE_find_number_l3345_334524

theorem find_number (N : ℚ) : 
  (N / (4/5) = (4/5) * N + 36) → N = 80 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3345_334524


namespace NUMINAMATH_CALUDE_initial_sale_percentage_l3345_334542

theorem initial_sale_percentage (P : ℝ) (x : ℝ) (h : x ≥ 0 ∧ x ≤ 1) : 
  ((1 - x) * P * 0.9 = 0.45 * P) → x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_initial_sale_percentage_l3345_334542


namespace NUMINAMATH_CALUDE_paul_gave_35_books_l3345_334513

/-- The number of books Paul gave to his friend -/
def books_given_to_friend (initial_books sold_books remaining_books : ℕ) : ℕ :=
  initial_books - sold_books - remaining_books

/-- Theorem stating that Paul gave 35 books to his friend -/
theorem paul_gave_35_books : books_given_to_friend 108 11 62 = 35 := by
  sorry

end NUMINAMATH_CALUDE_paul_gave_35_books_l3345_334513
