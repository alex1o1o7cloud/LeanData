import Mathlib

namespace NUMINAMATH_CALUDE_factorization_correctness_l1307_130747

theorem factorization_correctness (x y : ℝ) : 
  (∃! n : ℕ, n = (if x^3 + 2*x*y + x = x*(x^2 + 2*y) then 1 else 0) + 
             (if x^2 + 4*x + 4 = (x + 2)^2 then 1 else 0) + 
             (if -x^2 + y^2 = (x + y)*(x - y) then 1 else 0) ∧ 
             n = 1) := by sorry

end NUMINAMATH_CALUDE_factorization_correctness_l1307_130747


namespace NUMINAMATH_CALUDE_work_days_calculation_l1307_130786

/-- Represents the number of days worked by each person -/
structure WorkDays where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the daily wages of each person -/
structure DailyWages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The main theorem statement -/
theorem work_days_calculation 
  (work_days : WorkDays)
  (daily_wages : DailyWages)
  (total_earning : ℕ)
  (h1 : work_days.a = 6)
  (h2 : work_days.c = 4)
  (h3 : daily_wages.a * 4 = daily_wages.b * 3)
  (h4 : daily_wages.b * 5 = daily_wages.c * 4)
  (h5 : daily_wages.c = 95)
  (h6 : work_days.a * daily_wages.a + work_days.b * daily_wages.b + work_days.c * daily_wages.c = total_earning)
  (h7 : total_earning = 1406)
  : work_days.b = 9 := by
  sorry


end NUMINAMATH_CALUDE_work_days_calculation_l1307_130786


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1307_130745

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1307_130745


namespace NUMINAMATH_CALUDE_proposition_and_negation_l1307_130783

theorem proposition_and_negation :
  (∃ x : ℝ, x^2 - x = 0) ∧
  (¬(∃ x : ℝ, x^2 - x = 0) ↔ (∀ x : ℝ, x^2 - x ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_negation_l1307_130783


namespace NUMINAMATH_CALUDE_compute_expression_l1307_130785

theorem compute_expression : 9 * (-5) - (7 * -2) + (8 * -6) = -79 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1307_130785


namespace NUMINAMATH_CALUDE_gas_usage_multiple_l1307_130726

theorem gas_usage_multiple (felicity_usage adhira_usage : ℕ) 
  (h1 : felicity_usage = 23)
  (h2 : adhira_usage = 7)
  (h3 : ∃ m : ℕ, felicity_usage = m * adhira_usage - 5) :
  ∃ m : ℕ, m = 4 ∧ felicity_usage = m * adhira_usage - 5 :=
by sorry

end NUMINAMATH_CALUDE_gas_usage_multiple_l1307_130726


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_in_interval_l1307_130794

theorem inequality_holds_iff_p_in_interval (p q : ℝ) :
  q > 0 →
  2*p + q ≠ 0 →
  (4*(2*p*q^2 + p^2*q + 4*q^2 + 4*p*q) / (2*p + q) > 3*p^2*q) ↔
  (0 ≤ p ∧ p < 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_in_interval_l1307_130794


namespace NUMINAMATH_CALUDE_short_trees_to_plant_l1307_130799

theorem short_trees_to_plant (current_short_trees : ℕ) (total_short_trees_after : ℕ) 
  (h1 : current_short_trees = 112)
  (h2 : total_short_trees_after = 217) :
  total_short_trees_after - current_short_trees = 105 := by
  sorry

#check short_trees_to_plant

end NUMINAMATH_CALUDE_short_trees_to_plant_l1307_130799


namespace NUMINAMATH_CALUDE_prime_triple_equation_l1307_130790

theorem prime_triple_equation (p q n : ℕ) : 
  p.Prime → q.Prime → p > 0 → q > 0 → n > 0 →
  p * (p + 1) + q * (q + 1) = n * (n + 1) →
  ((p = 5 ∧ q = 3 ∧ n = 6) ∨ (p = 3 ∧ q = 5 ∧ n = 6)) :=
by sorry

end NUMINAMATH_CALUDE_prime_triple_equation_l1307_130790


namespace NUMINAMATH_CALUDE_fireflies_problem_l1307_130773

theorem fireflies_problem (initial : ℕ) : 
  (initial + 8 - 2 = 9) → initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_fireflies_problem_l1307_130773


namespace NUMINAMATH_CALUDE_initial_money_calculation_l1307_130787

theorem initial_money_calculation (initial_amount : ℚ) : 
  (initial_amount * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = 500) → 
  initial_amount = 1250 := by
sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l1307_130787


namespace NUMINAMATH_CALUDE_bathtub_guests_l1307_130784

/-- Proves that given a bathtub with 10 liters capacity, after 3 guests use 1.5 liters each
    and 1 guest uses 1.75 liters, the remaining water can be used by exactly 3 more guests
    if each uses 1.25 liters. -/
theorem bathtub_guests (bathtub_capacity : ℝ) (guests_1 : ℕ) (water_1 : ℝ)
                        (guests_2 : ℕ) (water_2 : ℝ) (water_per_remaining_guest : ℝ) :
  bathtub_capacity = 10 →
  guests_1 = 3 →
  water_1 = 1.5 →
  guests_2 = 1 →
  water_2 = 1.75 →
  water_per_remaining_guest = 1.25 →
  (bathtub_capacity - (guests_1 * water_1 + guests_2 * water_2)) / water_per_remaining_guest = 3 :=
by sorry

end NUMINAMATH_CALUDE_bathtub_guests_l1307_130784


namespace NUMINAMATH_CALUDE_eleven_operations_to_equal_l1307_130737

/-- The number of operations required to make two numbers equal --/
def operations_to_equal (a b : ℕ) (sub_a add_b : ℕ) : ℕ :=
  (a - b) / (sub_a + add_b)

/-- Theorem stating that it takes 11 operations to make the numbers equal --/
theorem eleven_operations_to_equal :
  operations_to_equal 365 24 19 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_operations_to_equal_l1307_130737


namespace NUMINAMATH_CALUDE_hyperbola_y_axis_condition_l1307_130795

/-- Represents a conic section of the form mx^2 + ny^2 = 1 -/
structure ConicSection (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1

/-- Predicate for a hyperbola with foci on the y-axis -/
def IsHyperbolaOnYAxis (m n : ℝ) : Prop :=
  m < 0 ∧ n > 0

theorem hyperbola_y_axis_condition (m n : ℝ) :
  (IsHyperbolaOnYAxis m n → m * n < 0) ∧
  ¬(m * n < 0 → IsHyperbolaOnYAxis m n) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_y_axis_condition_l1307_130795


namespace NUMINAMATH_CALUDE_annual_growth_rate_l1307_130734

theorem annual_growth_rate (initial_value final_value : ℝ) (h1 : initial_value = 70400) 
  (h2 : final_value = 89100) : ∃ r : ℝ, initial_value * (1 + r)^2 = final_value ∧ r = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_annual_growth_rate_l1307_130734


namespace NUMINAMATH_CALUDE_dog_food_consumption_l1307_130741

/-- The amount of dog food one dog eats per day, in scoops -/
def dog_food_per_dog : ℝ := 0.12

/-- The number of dogs Ella owns -/
def number_of_dogs : ℕ := 2

/-- The total amount of dog food consumed by all dogs in a day, in scoops -/
def total_food_consumed : ℝ := dog_food_per_dog * number_of_dogs

theorem dog_food_consumption :
  total_food_consumed = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_consumption_l1307_130741


namespace NUMINAMATH_CALUDE_prime_product_divisible_by_four_l1307_130772

theorem prime_product_divisible_by_four (p q : ℕ) : 
  Prime p → Prime q → Prime (p * q + 1) → 
  4 ∣ ((2 * p + q) * (p + 2 * q)) := by
sorry

end NUMINAMATH_CALUDE_prime_product_divisible_by_four_l1307_130772


namespace NUMINAMATH_CALUDE_root_quadruples_l1307_130709

theorem root_quadruples : ∀ a b c d : ℝ,
  (a ≠ b ∧ 
   2 * a^2 - 3 * c * a + 8 * d = 0 ∧
   2 * b^2 - 3 * c * b + 8 * d = 0 ∧
   c ≠ d ∧
   2 * c^2 - 3 * a * c + 8 * b = 0 ∧
   2 * d^2 - 3 * a * d + 8 * b = 0) →
  ((a = 4 ∧ b = 8 ∧ c = 4 ∧ d = 8) ∨
   (a = -2 ∧ b = -22 ∧ c = -8 ∧ d = 11) ∨
   (a = -8 ∧ b = 2 ∧ c = -2 ∧ d = -4)) :=
by sorry

end NUMINAMATH_CALUDE_root_quadruples_l1307_130709


namespace NUMINAMATH_CALUDE_problem_solution_l1307_130788

theorem problem_solution (a b c : ℝ) 
  (h1 : a + b + c = 150)
  (h2 : a + 10 = b - 5)
  (h3 : b - 5 = c^2) :
  b = (1322 - 2 * Real.sqrt 1241) / 16 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1307_130788


namespace NUMINAMATH_CALUDE_acute_angle_in_first_quadrant_l1307_130735

-- Define what an acute angle is
def is_acute_angle (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Define what it means for an angle to be in the first quadrant
def in_first_quadrant (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Theorem stating that an acute angle is in the first quadrant
theorem acute_angle_in_first_quadrant (θ : Real) : 
  is_acute_angle θ → in_first_quadrant θ := by
  sorry


end NUMINAMATH_CALUDE_acute_angle_in_first_quadrant_l1307_130735


namespace NUMINAMATH_CALUDE_min_n_for_120n_divisibility_l1307_130710

theorem min_n_for_120n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (4 ∣ 120 * m) ∧ (8 ∣ 120 * m) ∧ (12 ∣ 120 * m) → n ≤ m) ∧
  (4 ∣ 120 * n) ∧ (8 ∣ 120 * n) ∧ (12 ∣ 120 * n) :=
by
  -- Proof goes here
  sorry

#check min_n_for_120n_divisibility

end NUMINAMATH_CALUDE_min_n_for_120n_divisibility_l1307_130710


namespace NUMINAMATH_CALUDE_world_cup_teams_l1307_130725

theorem world_cup_teams (total_gifts : ℕ) (gifts_per_team : ℕ) : 
  total_gifts = 14 → 
  gifts_per_team = 2 → 
  total_gifts / gifts_per_team = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_world_cup_teams_l1307_130725


namespace NUMINAMATH_CALUDE_only_B_forms_grid_l1307_130716

/-- Represents a shape that can be used in the puzzle game -/
inductive Shape
  | A
  | B
  | C

/-- Represents a 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a shape can form a complete 4x4 grid without gaps or overlaps -/
def canFormGrid (s : Shape) : Prop :=
  ∃ (g : Grid), ∀ (i j : Fin 4), g i j = true

/-- Theorem stating that only shape B can form a complete 4x4 grid -/
theorem only_B_forms_grid :
  (canFormGrid Shape.B) ∧ 
  (¬ canFormGrid Shape.A) ∧ 
  (¬ canFormGrid Shape.C) :=
sorry

end NUMINAMATH_CALUDE_only_B_forms_grid_l1307_130716


namespace NUMINAMATH_CALUDE_cow_count_is_sixteen_l1307_130776

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem: If the total number of legs is 32 more than twice the number of heads,
    then the number of cows is 16 -/
theorem cow_count_is_sixteen (count : AnimalCount) :
  totalLegs count = 2 * totalHeads count + 32 → count.cows = 16 := by
  sorry


end NUMINAMATH_CALUDE_cow_count_is_sixteen_l1307_130776


namespace NUMINAMATH_CALUDE_oil_tank_depth_l1307_130707

/-- Represents a right frustum oil tank -/
structure RightFrustumTank where
  volume : ℝ  -- Volume in liters
  top_edge : ℝ  -- Length of top edge in cm
  bottom_edge : ℝ  -- Length of bottom edge in cm

/-- Calculates the depth of a right frustum oil tank -/
def calculate_depth (tank : RightFrustumTank) : ℝ :=
  sorry

/-- Theorem stating that the depth of the given oil tank is 75 cm -/
theorem oil_tank_depth (tank : RightFrustumTank) 
  (h1 : tank.volume = 190)
  (h2 : tank.top_edge = 60)
  (h3 : tank.bottom_edge = 40) :
  calculate_depth tank = 75 :=
sorry

end NUMINAMATH_CALUDE_oil_tank_depth_l1307_130707


namespace NUMINAMATH_CALUDE_sum_parity_when_sum_of_squares_even_l1307_130704

theorem sum_parity_when_sum_of_squares_even (m n : ℤ) : 
  Even (m^2 + n^2) → Even (m + n) :=
by sorry

end NUMINAMATH_CALUDE_sum_parity_when_sum_of_squares_even_l1307_130704


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1307_130731

theorem arithmetic_sequence_sum : 
  ∀ (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) (n : ℕ),
    a₁ = -25 →
    aₙ = 19 →
    d = 4 →
    aₙ = a₁ + (n - 1) * d →
    (n : ℤ) * (a₁ + aₙ) / 2 = -36 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1307_130731


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_negative_two_l1307_130711

theorem sum_of_roots_equals_negative_two
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (h1 : c^2 + a*c + b = 0)
  (h2 : d^2 + a*d + b = 0)
  (h3 : a^2 + c*a + d = 0)
  (h4 : b^2 + c*b + d = 0) :
  a + b + c + d = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_negative_two_l1307_130711


namespace NUMINAMATH_CALUDE_simplify_sqrt_x_squared_y_second_quadrant_l1307_130743

theorem simplify_sqrt_x_squared_y_second_quadrant (x y : ℝ) (h1 : x < 0) (h2 : y > 0) :
  Real.sqrt (x^2 * y) = -x * Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_x_squared_y_second_quadrant_l1307_130743


namespace NUMINAMATH_CALUDE_line_AB_equation_l1307_130728

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

/-- Point P -/
def P : ℝ × ℝ := (8, 1)

/-- A point lies on the line AB -/
def on_line_AB (x y : ℝ) : Prop := ∃ (t : ℝ), x = 8 + t ∧ y = 1 + 2*t

/-- A and B are intersection points of line AB and the hyperbola -/
def A_B_intersection (A B : ℝ × ℝ) : Prop :=
  on_line_AB A.1 A.2 ∧ on_line_AB B.1 B.2 ∧
  hyperbola A.1 A.2 ∧ hyperbola B.1 B.2

/-- P is the midpoint of AB -/
def P_is_midpoint (A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

/-- The main theorem -/
theorem line_AB_equation :
  ∃ (A B : ℝ × ℝ), A_B_intersection A B ∧ P_is_midpoint A B →
  ∀ (x y : ℝ), on_line_AB x y ↔ 2*x - y - 15 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_AB_equation_l1307_130728


namespace NUMINAMATH_CALUDE_brittany_age_after_vacation_l1307_130746

/-- Given that Rebecca is 25 years old and Brittany is 3 years older than Rebecca,
    prove that Brittany's age after returning from a 4-year vacation is 32 years old. -/
theorem brittany_age_after_vacation (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ)
  (h1 : rebecca_age = 25)
  (h2 : age_difference = 3)
  (h3 : vacation_duration = 4) :
  rebecca_age + age_difference + vacation_duration = 32 :=
by sorry

end NUMINAMATH_CALUDE_brittany_age_after_vacation_l1307_130746


namespace NUMINAMATH_CALUDE_book_writing_time_l1307_130724

/-- Calculates the number of weeks required to write a book -/
def weeks_to_write_book (pages_per_hour : ℕ) (hours_per_day : ℕ) (total_pages : ℕ) : ℕ :=
  (total_pages / (pages_per_hour * hours_per_day) + 6) / 7

/-- Theorem: It takes 7 weeks to write a 735-page book at 5 pages per hour, 3 hours per day -/
theorem book_writing_time :
  weeks_to_write_book 5 3 735 = 7 := by
  sorry

#eval weeks_to_write_book 5 3 735

end NUMINAMATH_CALUDE_book_writing_time_l1307_130724


namespace NUMINAMATH_CALUDE_meeting_gender_ratio_l1307_130700

theorem meeting_gender_ratio (total_population : ℕ) (females_attending : ℕ) : 
  total_population = 300 →
  females_attending = 50 →
  (total_population / 2 - females_attending) / females_attending = 2 := by
  sorry

end NUMINAMATH_CALUDE_meeting_gender_ratio_l1307_130700


namespace NUMINAMATH_CALUDE_store_revenue_l1307_130718

theorem store_revenue (N D J : ℝ) 
  (h1 : N = (3/5) * D) 
  (h2 : D = (20/7) * ((N + J) / 2)) : 
  J = (1/6) * N := by
sorry

end NUMINAMATH_CALUDE_store_revenue_l1307_130718


namespace NUMINAMATH_CALUDE_equation_solution_l1307_130761

theorem equation_solution : ∃ x : ℕ, 16^5 + 16^5 + 16^5 = 4^x ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1307_130761


namespace NUMINAMATH_CALUDE_rectangle_circle_union_area_l1307_130729

/-- The area of the union of a rectangle and a circle with specific dimensions -/
theorem rectangle_circle_union_area :
  let rectangle_width : ℝ := 8
  let rectangle_length : ℝ := 12
  let circle_radius : ℝ := 12
  let rectangle_area : ℝ := rectangle_width * rectangle_length
  let circle_area : ℝ := π * circle_radius^2
  let overlap_area : ℝ := (1/4) * circle_area
  rectangle_area + circle_area - overlap_area = 96 + 108 * π := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_union_area_l1307_130729


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1307_130757

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x + y = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 2006 * x - 2007
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x + y = -1003) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1307_130757


namespace NUMINAMATH_CALUDE_blue_candy_count_l1307_130713

theorem blue_candy_count (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total = 3409)
  (h2 : red = 145)
  (h3 : blue = total - red) :
  blue = 3264 := by
  sorry

end NUMINAMATH_CALUDE_blue_candy_count_l1307_130713


namespace NUMINAMATH_CALUDE_okeydokey_investment_l1307_130768

/-- Represents the investment scenario for earthworms -/
structure EarthwormInvestment where
  total_earthworms : ℕ
  artichokey_apples : ℕ
  okeydokey_earthworms : ℕ

/-- Calculates the number of apples Okeydokey invested -/
def okeydokey_apples (investment : EarthwormInvestment) : ℕ :=
  (investment.okeydokey_earthworms * (investment.artichokey_apples + investment.okeydokey_earthworms)) / 
  (investment.total_earthworms - investment.okeydokey_earthworms)

/-- Theorem stating that Okeydokey invested 5 apples -/
theorem okeydokey_investment (investment : EarthwormInvestment) 
  (h1 : investment.total_earthworms = 60)
  (h2 : investment.artichokey_apples = 7)
  (h3 : investment.okeydokey_earthworms = 25) : 
  okeydokey_apples investment = 5 := by
  sorry

#eval okeydokey_apples { total_earthworms := 60, artichokey_apples := 7, okeydokey_earthworms := 25 }

end NUMINAMATH_CALUDE_okeydokey_investment_l1307_130768


namespace NUMINAMATH_CALUDE_candy_probability_l1307_130714

def total_candies : ℕ := 20
def red_candies : ℕ := 12
def blue_candies : ℕ := 8

def same_color_probability : ℚ :=
  678 / 1735

theorem candy_probability :
  let first_pick := 2
  let second_pick := 2
  let remaining_candies := total_candies - first_pick
  (red_candies.choose first_pick * (red_candies - first_pick).choose second_pick +
   blue_candies.choose first_pick * (blue_candies - first_pick).choose second_pick +
   (red_candies.choose 1 * blue_candies.choose 1) * 
   ((red_candies - 1).choose 1 * (blue_candies - 1).choose 1)) /
  (total_candies.choose first_pick * remaining_candies.choose second_pick) =
  same_color_probability := by
sorry

end NUMINAMATH_CALUDE_candy_probability_l1307_130714


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l1307_130797

theorem arithmetic_evaluation : (7 + 5 + 8) / 3 - 2 / 3 + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l1307_130797


namespace NUMINAMATH_CALUDE_sphere_volume_reduction_line_tangent_to_circle_l1307_130752

-- Proposition 1
theorem sphere_volume_reduction (r : ℝ) (V : ℝ → ℝ) (h : V r = (4/3) * π * r^3) :
  V (r/2) = (1/8) * V r := by sorry

-- Proposition 3
theorem line_tangent_to_circle :
  let d := (1 : ℝ) / Real.sqrt 2
  (d = Real.sqrt ((1/2) : ℝ)) ∧ 
  (∀ x y : ℝ, x + y + 1 = 0 → x^2 + y^2 = 1/2 → 
    (x^2 + y^2 = d^2 ∨ x^2 + y^2 > d^2)) := by sorry

end NUMINAMATH_CALUDE_sphere_volume_reduction_line_tangent_to_circle_l1307_130752


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1307_130774

/-- Proves that a boat's speed in still water is 51 kmph given the conditions -/
theorem boat_speed_in_still_water 
  (upstream_time : ℝ) 
  (downstream_time : ℝ) 
  (stream_speed : ℝ) 
  (h1 : upstream_time = 2 * downstream_time)
  (h2 : stream_speed = 17) : 
  ∃ (boat_speed : ℝ), boat_speed = 51 ∧ 
    (boat_speed + stream_speed) * downstream_time = 
    (boat_speed - stream_speed) * upstream_time := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1307_130774


namespace NUMINAMATH_CALUDE_max_base_seven_digit_sum_l1307_130756

/-- Represents a positive integer in base 7 --/
def BaseSevenDigits := List Nat

/-- Converts a positive integer to its base 7 representation --/
def toBaseSeven (n : Nat) : BaseSevenDigits :=
  sorry

/-- Calculates the sum of digits in a base 7 representation --/
def sumBaseSevenDigits (digits : BaseSevenDigits) : Nat :=
  sorry

/-- Checks if a base 7 representation is valid (all digits < 7) --/
def isValidBaseSeven (digits : BaseSevenDigits) : Prop :=
  sorry

/-- Converts a base 7 representation back to a natural number --/
def fromBaseSeven (digits : BaseSevenDigits) : Nat :=
  sorry

/-- The main theorem --/
theorem max_base_seven_digit_sum :
  ∀ n : Nat, n > 0 → n < 3000 →
    ∃ (max : Nat),
      max = 24 ∧
      sumBaseSevenDigits (toBaseSeven n) ≤ max ∧
      (∀ m : Nat, m > 0 → m < 3000 →
        sumBaseSevenDigits (toBaseSeven m) ≤ max) :=
by sorry

end NUMINAMATH_CALUDE_max_base_seven_digit_sum_l1307_130756


namespace NUMINAMATH_CALUDE_triangle_special_condition_right_angle_l1307_130792

/-- Given a triangle ABC, if b cos C + c cos B = a sin A, then angle A is 90° -/
theorem triangle_special_condition_right_angle 
  (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧ 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  b * Real.cos C + c * Real.cos B = a * Real.sin A → 
  A = π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_condition_right_angle_l1307_130792


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1307_130715

/-- A quadratic function f(x) = ax^2 + bx + c with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_property (a b c : ℤ) 
  (h1 : QuadraticFunction a b c 2 = 5)
  (h2 : ∀ x, QuadraticFunction a b c x ≥ QuadraticFunction a b c 1)
  (h3 : QuadraticFunction a b c 1 = 3) :
  a - b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1307_130715


namespace NUMINAMATH_CALUDE_red_balls_drawn_is_random_variable_l1307_130755

/-- A bag containing black and red balls -/
structure Bag where
  black : ℕ
  red : ℕ

/-- The result of drawing balls from the bag -/
structure DrawResult where
  total : ℕ
  red : ℕ

/-- A random variable is a function that assigns a real number to each outcome of a random experiment -/
def RandomVariable (α : Type) := α → ℝ

/-- The bag containing 2 black balls and 6 red balls -/
def bag : Bag := { black := 2, red := 6 }

/-- The number of balls drawn -/
def numDrawn : ℕ := 2

/-- The function that counts the number of red balls drawn -/
def countRedBalls : DrawResult → ℕ := fun r => r.red

/-- Statement: The number of red balls drawn is a random variable -/
theorem red_balls_drawn_is_random_variable :
  ∃ (rv : RandomVariable DrawResult), ∀ (result : DrawResult),
    result.total = numDrawn ∧ result.red ≤ bag.red →
      rv result = (countRedBalls result : ℝ) :=
sorry

end NUMINAMATH_CALUDE_red_balls_drawn_is_random_variable_l1307_130755


namespace NUMINAMATH_CALUDE_tv_screen_area_l1307_130791

theorem tv_screen_area : 
  let trapezoid_short_base : ℝ := 3
  let trapezoid_long_base : ℝ := 5
  let trapezoid_height : ℝ := 2
  let triangle_base : ℝ := trapezoid_long_base
  let triangle_height : ℝ := 4
  let trapezoid_area := (trapezoid_short_base + trapezoid_long_base) * trapezoid_height / 2
  let triangle_area := triangle_base * triangle_height / 2
  trapezoid_area + triangle_area = 18 := by
sorry

end NUMINAMATH_CALUDE_tv_screen_area_l1307_130791


namespace NUMINAMATH_CALUDE_cosine_function_minimum_l1307_130720

theorem cosine_function_minimum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.cos (b * x + c) ≥ a * Real.cos c) ∧ 
  (∀ ε > 0, ∃ x, a * Real.cos (b * x + (c - ε)) > a * Real.cos (c - ε)) →
  c = π :=
sorry

end NUMINAMATH_CALUDE_cosine_function_minimum_l1307_130720


namespace NUMINAMATH_CALUDE_sequence_2015th_term_l1307_130798

/-- Given a sequence {a_n} satisfying the conditions:
    1) a₁ = 1
    2) a₂ = 1/2
    3) 2/a_{n+1} = 1/a_n + 1/a_{n+2} for all n ∈ ℕ*
    Prove that a₂₀₁₅ = 1/2015 -/
theorem sequence_2015th_term (a : ℕ → ℚ) 
  (h1 : a 1 = 1)
  (h2 : a 2 = 1/2)
  (h3 : ∀ n : ℕ, n ≥ 1 → 2 / (a (n + 1)) = 1 / (a n) + 1 / (a (n + 2))) :
  a 2015 = 1 / 2015 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2015th_term_l1307_130798


namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_half_l1307_130751

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the angle α
variable (α : ℝ)

-- State the theorem
theorem sin_alpha_plus_pi_half (h : ∃ (t : ℝ), t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) : 
  Real.sin (α + π/2) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_half_l1307_130751


namespace NUMINAMATH_CALUDE_right_triangle_special_angles_l1307_130736

-- Define a right triangle
structure RightTriangle where
  a : ℝ  -- leg 1
  b : ℝ  -- leg 2
  c : ℝ  -- hypotenuse
  h : ℝ  -- altitude to hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem
  altitude_condition : h = c / 4  -- altitude is 4 times smaller than hypotenuse

-- Define the theorem
theorem right_triangle_special_angles (t : RightTriangle) :
  let angle1 := Real.arcsin (t.h / t.c)
  let angle2 := Real.arcsin (t.a / t.c)
  (angle1 = 15 * π / 180 ∧ angle2 = 75 * π / 180) ∨
  (angle1 = 75 * π / 180 ∧ angle2 = 15 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_special_angles_l1307_130736


namespace NUMINAMATH_CALUDE_periodic_product_quotient_iff_commensurable_l1307_130703

theorem periodic_product_quotient_iff_commensurable 
  (f g : ℝ → ℝ) (T₁ T₂ : ℝ) 
  (hf : ∀ x, f (x + T₁) = f x) 
  (hg : ∀ x, g (x + T₂) = g x)
  (hpos_f : ∀ x, f x > 0)
  (hpos_g : ∀ x, g x > 0) :
  (∃ T, ∀ x, (f x * g x) = (f (x + T) * g (x + T)) ∧ 
            (f x / g x) = (f (x + T) / g (x + T))) ↔ 
  (∃ m n : ℤ, m ≠ 0 ∧ n ≠ 0 ∧ m * T₁ = n * T₂) :=
sorry

end NUMINAMATH_CALUDE_periodic_product_quotient_iff_commensurable_l1307_130703


namespace NUMINAMATH_CALUDE_h_range_l1307_130719

-- Define the function h(x)
def h (x : ℝ) : ℝ := 2 * (x - 3)

-- Define the domain of h(x)
def dom_h : Set ℝ := {x : ℝ | x ≠ -7}

-- Define the range of h(x)
def range_h : Set ℝ := {y : ℝ | y ≠ -20}

-- Theorem statement
theorem h_range : 
  {y : ℝ | ∃ x ∈ dom_h, h x = y} = range_h :=
sorry

end NUMINAMATH_CALUDE_h_range_l1307_130719


namespace NUMINAMATH_CALUDE_square_root_of_average_squares_ge_arithmetic_mean_l1307_130742

theorem square_root_of_average_squares_ge_arithmetic_mean
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_average_squares_ge_arithmetic_mean_l1307_130742


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1307_130760

theorem complex_equation_solution (z : ℂ) (h : (1 + 2*I)*z = 4 + 3*I) : z = 2 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1307_130760


namespace NUMINAMATH_CALUDE_box_cubes_count_l1307_130769

/-- The minimum number of cubes required to build a box -/
def min_cubes (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height) / cube_volume

/-- Theorem: The minimum number of 3 cubic cm cubes required to build a box
    with dimensions 12 cm × 16 cm × 6 cm is 384. -/
theorem box_cubes_count :
  min_cubes 12 16 6 3 = 384 := by
  sorry

end NUMINAMATH_CALUDE_box_cubes_count_l1307_130769


namespace NUMINAMATH_CALUDE_find_number_l1307_130727

theorem find_number : ∃ x : ℝ, 4 * x - 23 = 33 ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1307_130727


namespace NUMINAMATH_CALUDE_nonempty_set_implies_nonnegative_a_l1307_130793

theorem nonempty_set_implies_nonnegative_a (a : ℝ) :
  (∅ : Set ℝ) ⊂ {x : ℝ | x^2 ≤ a} → a ∈ Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_nonempty_set_implies_nonnegative_a_l1307_130793


namespace NUMINAMATH_CALUDE_playground_count_l1307_130723

theorem playground_count (numbers : List Nat) : 
  numbers.length = 6 ∧ 
  numbers.take 5 = [6, 12, 1, 12, 7] ∧ 
  (numbers.sum / numbers.length : ℚ) = 7 →
  numbers.getLast! = 4 := by
sorry

end NUMINAMATH_CALUDE_playground_count_l1307_130723


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1307_130758

/-- An isosceles triangle with side lengths m-2, 2m+1, and 8 has a perimeter of 17.5 -/
theorem isosceles_triangle_perimeter : ∀ m : ℝ,
  let a := m - 2
  let b := 2 * m + 1
  let c := 8
  (a = c ∨ b = c) → -- isosceles condition
  (a + b > c ∧ b + c > a ∧ c + a > b) → -- triangle inequality
  a + b + c = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1307_130758


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1307_130763

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, 1 + x^5 = a₀ + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1307_130763


namespace NUMINAMATH_CALUDE_evaluate_f_l1307_130717

def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_f : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l1307_130717


namespace NUMINAMATH_CALUDE_f_properties_imply_l1307_130782

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x > 0 → f x < 0) ∧
  (f 1 = -2)

theorem f_properties_imply (f : ℝ → ℝ) (h : f_properties f) :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ ∀ y ∈ Set.Icc (-3) 3, f y ≤ f x ∧ f x = 6) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ ∀ y ∈ Set.Icc (-3) 3, f y ≥ f x ∧ f x = -6) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_imply_l1307_130782


namespace NUMINAMATH_CALUDE_min_value_a_l1307_130706

theorem min_value_a (a b : ℕ) (h : 1176 * a = b^3) : 63 ≤ a := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l1307_130706


namespace NUMINAMATH_CALUDE_problem_solution_l1307_130777

theorem problem_solution (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1307_130777


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l1307_130771

theorem concentric_circles_ratio (a b : ℝ) (h : a > 0) (h' : b > 0) 
  (h_area : π * b^2 - π * a^2 = 4 * (π * a^2)) : 
  a / b = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l1307_130771


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_sum_of_specific_quadratic_roots_l1307_130764

theorem sum_of_quadratic_roots (a b c : ℚ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a :=
by sorry

theorem sum_of_specific_quadratic_roots :
  let a : ℚ := -48
  let b : ℚ := 108
  let c : ℚ := 162
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_sum_of_specific_quadratic_roots_l1307_130764


namespace NUMINAMATH_CALUDE_scott_distance_l1307_130778

/-- Given a 100-meter race where Scott runs 4 meters for every 5 meters that Chris runs,
    prove that Scott will have run 80 meters when Chris crosses the finish line. -/
theorem scott_distance (race_length : ℕ) (scott_ratio chris_ratio : ℕ) : 
  race_length = 100 →
  scott_ratio = 4 →
  chris_ratio = 5 →
  (scott_ratio * race_length) / chris_ratio = 80 := by
sorry

end NUMINAMATH_CALUDE_scott_distance_l1307_130778


namespace NUMINAMATH_CALUDE_inaccurate_tape_measurement_l1307_130750

theorem inaccurate_tape_measurement 
  (wholesale_price : ℝ) 
  (tape_length : ℝ) 
  (retail_markup : ℝ) 
  (actual_profit : ℝ) 
  (h1 : retail_markup = 0.4)
  (h2 : actual_profit = 0.39)
  (h3 : ((1 + retail_markup) * wholesale_price - tape_length * wholesale_price) / (tape_length * wholesale_price) = actual_profit) :
  tape_length = 140 / 139 :=
sorry

end NUMINAMATH_CALUDE_inaccurate_tape_measurement_l1307_130750


namespace NUMINAMATH_CALUDE_nabla_example_l1307_130749

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- State the theorem
theorem nabla_example : nabla (nabla 2 3) 2 = 4099 := by
  sorry

end NUMINAMATH_CALUDE_nabla_example_l1307_130749


namespace NUMINAMATH_CALUDE_f_is_direct_proportion_l1307_130779

/-- Definition of direct proportion --/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function we want to prove is directly proportional --/
def f (x : ℝ) : ℝ := -0.1 * x

/-- Theorem stating that f is a direct proportion --/
theorem f_is_direct_proportion : is_direct_proportion f := by
  sorry

end NUMINAMATH_CALUDE_f_is_direct_proportion_l1307_130779


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1307_130748

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 > x^2) ↔ (∃ x : ℝ, x^3 ≤ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1307_130748


namespace NUMINAMATH_CALUDE_bobs_improvement_percentage_l1307_130766

theorem bobs_improvement_percentage (bob_time sister_time : ℕ) 
  (h1 : bob_time = 640) 
  (h2 : sister_time = 320) : 
  (bob_time - sister_time) / bob_time * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bobs_improvement_percentage_l1307_130766


namespace NUMINAMATH_CALUDE_vector_sum_necessary_not_sufficient_l1307_130767

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def form_triangle (a b c : V) : Prop := sorry

theorem vector_sum_necessary_not_sufficient (a b c : V) :
  (form_triangle a b c → a + b + c = 0) ∧
  ¬(a + b + c = 0 → form_triangle a b c) :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_necessary_not_sufficient_l1307_130767


namespace NUMINAMATH_CALUDE_contest_ranking_l1307_130740

theorem contest_ranking (A B C D : ℝ) 
  (non_negative : A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ D ≥ 0)
  (sum_equality : B + D = A + C)
  (interchange_inequality : A + B > C + D)
  (dick_exceeds : D > B + C) :
  A > D ∧ D > B ∧ B > C := by
sorry

end NUMINAMATH_CALUDE_contest_ranking_l1307_130740


namespace NUMINAMATH_CALUDE_birth_year_problem_l1307_130730

theorem birth_year_problem (x : ℕ) : 
  (1850 ≤ x^2 + x) ∧ (x^2 + x < 1900) → -- Born in second half of 19th century
  (x^2 + 2*x - x = x^2 + x) →           -- x years old in year x^2 + 2x
  x^2 + x = 1892                        -- Year of birth is 1892
:= by sorry

end NUMINAMATH_CALUDE_birth_year_problem_l1307_130730


namespace NUMINAMATH_CALUDE_particle_movement_probability_l1307_130775

/-- The probability of a particle moving from (0, 0) to (2, 3) in 5 steps,
    where each step has an equal probability of 1/2 of moving right or up. -/
theorem particle_movement_probability :
  let n : ℕ := 5  -- Total number of steps
  let k : ℕ := 2  -- Number of steps to the right
  let p : ℚ := 1/2  -- Probability of moving right (or up)
  Nat.choose n k * p^n = (1/2)^5 := by
  sorry

end NUMINAMATH_CALUDE_particle_movement_probability_l1307_130775


namespace NUMINAMATH_CALUDE_determinant_zero_l1307_130721

theorem determinant_zero (α β : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![0, Real.cos α, -Real.sin α],
                                        ![-Real.cos α, 0, Real.cos β],
                                        ![Real.sin α, -Real.cos β, 0]]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_determinant_zero_l1307_130721


namespace NUMINAMATH_CALUDE_good_goods_sufficient_condition_l1307_130733

-- Define propositions
variable (G : Prop) -- G represents "goods are good"
variable (C : Prop) -- C represents "goods are cheap"

-- Define the statement "Good goods are not cheap"
def good_goods_not_cheap : Prop := G → ¬C

-- Theorem to prove
theorem good_goods_sufficient_condition (h : good_goods_not_cheap G C) : 
  G → ¬C :=
by
  sorry


end NUMINAMATH_CALUDE_good_goods_sufficient_condition_l1307_130733


namespace NUMINAMATH_CALUDE_baseball_cards_problem_l1307_130738

theorem baseball_cards_problem (X : ℚ) : 3 * (X - (X + 1) / 2 - 1) = 18 ↔ X = 15 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_problem_l1307_130738


namespace NUMINAMATH_CALUDE_max_trip_weight_is_750_l1307_130780

/-- Represents the number of crates on a trip -/
inductive NumCrates
  | three
  | four
  | five

/-- The minimum weight of a single crate in kg -/
def minCrateWeight : ℝ := 150

/-- Calculates the maximum weight of crates on a single trip -/
def maxTripWeight (n : NumCrates) : ℝ :=
  match n with
  | .three => 3 * minCrateWeight
  | .four => 4 * minCrateWeight
  | .five => 5 * minCrateWeight

/-- Theorem: The maximum weight of crates on a single trip is 750 kg -/
theorem max_trip_weight_is_750 :
  ∀ n : NumCrates, maxTripWeight n ≤ 750 ∧ ∃ m : NumCrates, maxTripWeight m = 750 :=
by sorry

end NUMINAMATH_CALUDE_max_trip_weight_is_750_l1307_130780


namespace NUMINAMATH_CALUDE_loan_amount_proof_l1307_130732

/-- The annual interest rate A charges B -/
def interest_rate_A : ℝ := 0.10

/-- The annual interest rate B charges C -/
def interest_rate_B : ℝ := 0.115

/-- The number of years for which the loan is considered -/
def years : ℝ := 3

/-- B's gain over the loan period -/
def gain : ℝ := 1125

/-- The amount lent by A to B -/
def amount : ℝ := 25000

theorem loan_amount_proof :
  gain = (interest_rate_B - interest_rate_A) * years * amount := by sorry

end NUMINAMATH_CALUDE_loan_amount_proof_l1307_130732


namespace NUMINAMATH_CALUDE_fifth_term_sequence_l1307_130739

theorem fifth_term_sequence (n : ℕ) : 
  let a : ℕ → ℕ := λ k => k * (k + 1) / 2
  a 5 = 15 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_sequence_l1307_130739


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1307_130759

theorem inequality_solution_set (m n : ℝ) : 
  (∀ x, mx - n > 0 ↔ x < 1/3) → 
  (∀ x, (m + n) * x < n - m ↔ x > -1/2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1307_130759


namespace NUMINAMATH_CALUDE_binary_10110100_is_180_l1307_130796

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10110100_is_180 :
  binary_to_decimal [false, false, true, false, true, true, false, true] = 180 := by
  sorry

end NUMINAMATH_CALUDE_binary_10110100_is_180_l1307_130796


namespace NUMINAMATH_CALUDE_florist_roses_theorem_l1307_130712

/-- Represents the number of roses picked in the first picking -/
def first_picking : ℝ := 16.0

theorem florist_roses_theorem (initial : ℝ) (second_picking : ℝ) (final_total : ℝ) :
  initial = 37.0 →
  second_picking = 19.0 →
  final_total = 72 →
  initial + first_picking + second_picking = final_total :=
by
  sorry

#check florist_roses_theorem

end NUMINAMATH_CALUDE_florist_roses_theorem_l1307_130712


namespace NUMINAMATH_CALUDE_chimney_bricks_proof_l1307_130765

/-- The time it takes Brenda to build the chimney alone (in hours) -/
def brenda_time : ℝ := 9

/-- The time it takes Brandon to build the chimney alone (in hours) -/
def brandon_time : ℝ := 10

/-- The decrease in combined output when working together (in bricks per hour) -/
def output_decrease : ℝ := 10

/-- The time it takes Brenda and Brandon to build the chimney together (in hours) -/
def combined_time : ℝ := 5

/-- The number of bricks in the chimney -/
def chimney_bricks : ℝ := 900

theorem chimney_bricks_proof :
  let brenda_rate := chimney_bricks / brenda_time
  let brandon_rate := chimney_bricks / brandon_time
  let combined_rate := brenda_rate + brandon_rate - output_decrease
  chimney_bricks = combined_rate * combined_time := by
  sorry

end NUMINAMATH_CALUDE_chimney_bricks_proof_l1307_130765


namespace NUMINAMATH_CALUDE_find_d_l1307_130702

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c - d + 3)) : 
  d = 75/16 := by
sorry

end NUMINAMATH_CALUDE_find_d_l1307_130702


namespace NUMINAMATH_CALUDE_no_prime_divisible_by_39_l1307_130762

theorem no_prime_divisible_by_39 : ∀ p : ℕ, Prime p → ¬(39 ∣ p) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_divisible_by_39_l1307_130762


namespace NUMINAMATH_CALUDE_candy_division_l1307_130722

theorem candy_division (mark peter susan john lucy : ℝ) 
  (h1 : mark = 90)
  (h2 : peter = 120.5)
  (h3 : susan = 74.75)
  (h4 : john = 150)
  (h5 : lucy = 85.25)
  (total_people : ℕ)
  (h6 : total_people = 10) :
  (mark + peter + susan + john + lucy) / total_people = 52.05 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_l1307_130722


namespace NUMINAMATH_CALUDE_log2_odd_and_increasing_l1307_130770

-- Define the logarithm base 2 function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log2_odd_and_increasing :
  (∀ x > 0, log2 (-x) = -log2 x) ∧
  (∀ x y, 0 ≤ x → x ≤ y → log2 x ≤ log2 y) :=
by sorry

end NUMINAMATH_CALUDE_log2_odd_and_increasing_l1307_130770


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1307_130705

/-- Represents a hyperbola with foci on the y-axis -/
structure Hyperbola where
  /-- The distance from the center to a focus -/
  c : ℝ
  /-- The length of the semi-major axis -/
  a : ℝ
  /-- The length of the semi-minor axis -/
  b : ℝ
  /-- One focus lies on the line 5x-2y+20=0 -/
  focus_on_line : c = 10
  /-- The ratio of c to a is 5/3 -/
  c_a_ratio : c / a = 5 / 3
  /-- Relationship between a, b, and c -/
  abc_relation : b^2 = c^2 - a^2

/-- The equation of the hyperbola is x²/64 - y²/36 = -1 -/
theorem hyperbola_equation (h : Hyperbola) :
  ∀ x y : ℝ, (x^2 / 64 - y^2 / 36 = -1) ↔ h.b^2 * y^2 - h.a^2 * x^2 = h.a^2 * h.b^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1307_130705


namespace NUMINAMATH_CALUDE_total_employees_l1307_130789

theorem total_employees (part_time full_time : ℕ) 
  (h1 : part_time = 2041) 
  (h2 : full_time = 63093) : 
  part_time + full_time = 65134 := by
  sorry

end NUMINAMATH_CALUDE_total_employees_l1307_130789


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l1307_130781

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular n β) : 
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l1307_130781


namespace NUMINAMATH_CALUDE_six_meetings_in_middle_l1307_130708

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℕ  -- Speed in meters per minute

/-- Calculates the number of meetings in the middle for two runners -/
def numberOfMeetings (runner1 runner2 : Runner) : ℕ :=
  sorry

/-- Theorem stating that two runners with given speeds meet 6 times in the middle -/
theorem six_meetings_in_middle :
  let runner1 : Runner := ⟨240⟩
  let runner2 : Runner := ⟨180⟩
  numberOfMeetings runner1 runner2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_six_meetings_in_middle_l1307_130708


namespace NUMINAMATH_CALUDE_triangle_inequality_l1307_130744

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  a + b > c ∧ a + c > b ∧ b + c > a := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1307_130744


namespace NUMINAMATH_CALUDE_min_value_expression_l1307_130701

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1 / 2) :
  x^3 + 4*x*y + 16*y^3 + 8*y*z + 3*z^3 ≥ 18 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 / 2 ∧
    x₀^3 + 4*x₀*y₀ + 16*y₀^3 + 8*y₀*z₀ + 3*z₀^3 = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1307_130701


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1307_130754

/-- 
A line x = m intersects a parabola x = -3y^2 - 4y + 7 at exactly one point 
if and only if m = 25/3
-/
theorem parabola_line_intersection (m : ℝ) : 
  (∃! y : ℝ, m = -3 * y^2 - 4 * y + 7) ↔ m = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1307_130754


namespace NUMINAMATH_CALUDE_hiker_distance_l1307_130753

theorem hiker_distance (north east south east2 : ℝ) 
  (h_north : north = 15)
  (h_east : east = 8)
  (h_south : south = 9)
  (h_east2 : east2 = 2) :
  Real.sqrt ((north - south)^2 + (east + east2)^2) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_l1307_130753
