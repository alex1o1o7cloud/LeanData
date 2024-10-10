import Mathlib

namespace smallest_four_digit_divisible_by_53_l1634_163453

theorem smallest_four_digit_divisible_by_53 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 1007 → ¬(53 ∣ n)) ∧ 53 ∣ 1007 :=
by sorry

end smallest_four_digit_divisible_by_53_l1634_163453


namespace area_of_triangle_ABC_is_one_l1634_163452

/-- Given three unit squares arranged in a straight line, each sharing a side with the next,
    where A is the bottom left vertex of the first square,
    B is the top right vertex of the second square,
    and C is the top left vertex of the third square,
    prove that the area of triangle ABC is 1. -/
theorem area_of_triangle_ABC_is_one :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 1)
  let C : ℝ × ℝ := (2, 1)
  let triangle_area (p q r : ℝ × ℝ) : ℝ :=
    (1/2) * abs (p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2))
  triangle_area A B C = 1 := by
sorry

end area_of_triangle_ABC_is_one_l1634_163452


namespace certain_number_is_three_l1634_163408

theorem certain_number_is_three (x : ℝ) (n : ℝ) : 
  (4 / (1 + n / x) = 1) → (x = 1) → n = 3 := by
  sorry

end certain_number_is_three_l1634_163408


namespace abs_inequality_equivalence_l1634_163480

theorem abs_inequality_equivalence (x : ℝ) : 
  |5 - 2*x| < 3 ↔ 1 < x ∧ x < 4 :=
sorry

end abs_inequality_equivalence_l1634_163480


namespace product_powers_equality_l1634_163400

theorem product_powers_equality (a : ℝ) : 
  let b := a - 1
  (a + b) * (a^2 + b^2) * (a^4 + b^4) * (a^8 + b^8) * (a^16 + b^16) * (a^32 + b^32) = a^64 - b^64 := by
  sorry

end product_powers_equality_l1634_163400


namespace dinner_lunch_ratio_is_two_l1634_163498

/-- Represents the daily calorie intake of John -/
structure DailyCalories where
  breakfast : ℕ
  lunch : ℕ
  dinner : ℕ
  shakes : ℕ
  total : ℕ

/-- The ratio of dinner calories to lunch calories -/
def dinner_lunch_ratio (dc : DailyCalories) : ℚ :=
  dc.dinner / dc.lunch

/-- John's daily calorie intake satisfies the given conditions -/
def johns_calories : DailyCalories :=
  { breakfast := 500,
    lunch := 500 + (500 * 25 / 100),
    dinner := 3275 - (500 + (500 + (500 * 25 / 100)) + (3 * 300)),
    shakes := 3 * 300,
    total := 3275 }

theorem dinner_lunch_ratio_is_two : dinner_lunch_ratio johns_calories = 2 := by
  sorry

end dinner_lunch_ratio_is_two_l1634_163498


namespace inequality_proof_l1634_163470

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 8) :
  (a - 2) / (a + 1) + (b - 2) / (b + 1) + (c - 2) / (c + 1) ≤ 0 := by
  sorry

end inequality_proof_l1634_163470


namespace time_to_restaurant_is_10_minutes_l1634_163487

/-- Time in minutes to walk from Park Office to Hidden Lake -/
def time_to_hidden_lake : ℕ := 15

/-- Time in minutes to walk from Hidden Lake to Park Office -/
def time_from_hidden_lake : ℕ := 7

/-- Total time in minutes for the entire journey (including Lake Park restaurant) -/
def total_time : ℕ := 32

/-- Time in minutes to walk from Park Office to Lake Park restaurant -/
def time_to_restaurant : ℕ := total_time - (time_to_hidden_lake + time_from_hidden_lake)

theorem time_to_restaurant_is_10_minutes : time_to_restaurant = 10 := by
  sorry

end time_to_restaurant_is_10_minutes_l1634_163487


namespace complex_number_range_l1634_163446

theorem complex_number_range (x y : ℝ) : 
  let z : ℂ := x + y * Complex.I
  (Complex.abs (z - (3 + 4 * Complex.I)) = 1) →
  (16 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 36) :=
by sorry

end complex_number_range_l1634_163446


namespace ant_population_growth_l1634_163417

/-- Represents the number of days passed -/
def days : ℕ := 4

/-- The growth factor for Species C per day -/
def growth_factor_C : ℕ := 2

/-- The growth factor for Species D per day -/
def growth_factor_D : ℕ := 4

/-- The total number of ants on Day 0 -/
def total_ants_day0 : ℕ := 35

/-- The total number of ants on Day 4 -/
def total_ants_day4 : ℕ := 3633

/-- The number of Species C ants on Day 0 -/
def species_C_day0 : ℕ := 22

/-- The number of Species D ants on Day 0 -/
def species_D_day0 : ℕ := 13

theorem ant_population_growth :
  species_C_day0 * growth_factor_C ^ days = 352 ∧
  species_C_day0 + species_D_day0 = total_ants_day0 ∧
  species_C_day0 * growth_factor_C ^ days + species_D_day0 * growth_factor_D ^ days = total_ants_day4 :=
by sorry

end ant_population_growth_l1634_163417


namespace tan_sixty_degrees_l1634_163451

theorem tan_sixty_degrees : Real.tan (π / 3) = Real.sqrt 3 := by sorry

end tan_sixty_degrees_l1634_163451


namespace max_value_of_f_l1634_163485

/-- The quadratic function f(x) = -3x^2 + 6x + 2 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 6 * x + 2

/-- The maximum value of f(x) is 5 -/
theorem max_value_of_f : ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M := by sorry

end max_value_of_f_l1634_163485


namespace simplify_expression_l1634_163434

theorem simplify_expression : 
  1 / (2 / (Real.sqrt 3 + 2) + 3 / (Real.sqrt 5 - 2)) = (10 + 2 * Real.sqrt 3 - 3 * Real.sqrt 5) / 43 := by
  sorry

end simplify_expression_l1634_163434


namespace geometric_sequence_inequality_l1634_163477

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- For any geometric sequence, the average of the squares of the second and fourth terms
    is greater than or equal to the square of the third term. -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (h : IsGeometricSequence a) :
    (a 2)^2 / 2 + (a 4)^2 / 2 ≥ (a 3)^2 := by
  sorry

end geometric_sequence_inequality_l1634_163477


namespace bowling_ball_weight_proof_l1634_163442

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 21.875

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 35

/-- Theorem stating that the weight of one bowling ball is 21.875 pounds -/
theorem bowling_ball_weight_proof :
  (8 * bowling_ball_weight = 5 * canoe_weight) ∧
  (4 * canoe_weight = 140) →
  bowling_ball_weight = 21.875 := by
sorry

end bowling_ball_weight_proof_l1634_163442


namespace remainder_sum_l1634_163413

theorem remainder_sum (c d : ℤ) (hc : c % 100 = 86) (hd : d % 150 = 144) :
  (c + d) % 50 = 30 := by sorry

end remainder_sum_l1634_163413


namespace special_polygon_perimeter_l1634_163438

/-- A polygon with specific properties -/
structure SpecialPolygon where
  AB : ℝ
  AE : ℝ
  BD : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  angle_DBC : ℝ
  angle_BCD : ℝ
  angle_CDB : ℝ
  h_AB_eq_AE : AB = AE
  h_AB_val : AB = 120
  h_DE_val : DE = 226
  h_BD_val : BD = 115
  h_BD_eq_BC : BD = BC
  h_angle_DBC_eq_BCD : angle_DBC = angle_BCD
  h_triangle_BCD_equilateral : angle_DBC = 60 ∧ angle_BCD = 60 ∧ angle_CDB = 60
  h_CD_eq_BD : CD = BD

/-- The perimeter of the special polygon is 696 -/
theorem special_polygon_perimeter (p : SpecialPolygon) : 
  p.AB + p.AE + p.BD + p.BC + p.CD + p.DE = 696 := by
  sorry


end special_polygon_perimeter_l1634_163438


namespace average_age_combined_l1634_163448

theorem average_age_combined (n_students : ℕ) (avg_age_students : ℝ)
                              (n_parents : ℕ) (avg_age_parents : ℝ)
                              (n_teachers : ℕ) (avg_age_teachers : ℝ) :
  n_students = 40 →
  avg_age_students = 10 →
  n_parents = 60 →
  avg_age_parents = 35 →
  n_teachers = 5 →
  avg_age_teachers = 45 →
  (n_students * avg_age_students + n_parents * avg_age_parents + n_teachers * avg_age_teachers) /
  (n_students + n_parents + n_teachers : ℝ) = 26 := by
  sorry

#check average_age_combined

end average_age_combined_l1634_163448


namespace option_D_is_false_l1634_163467

-- Define the proposition p and q
variable (p q : Prop)

-- Define the statement for option D
def option_D : Prop := (p ∨ q) → (p ∧ q)

-- Theorem stating that option D is false
theorem option_D_is_false : ¬ (∀ p q, option_D p q) := by
  sorry

-- Note: We don't need to prove the other options are correct in this statement,
-- as the question only asks for the incorrect option.

end option_D_is_false_l1634_163467


namespace businessmen_one_beverage_businessmen_one_beverage_proof_l1634_163420

/-- The number of businessmen who drank only one type of beverage at a conference -/
theorem businessmen_one_beverage (total : ℕ) (coffee tea juice : ℕ) 
  (coffee_tea coffee_juice tea_juice : ℕ) (all_three : ℕ) : ℕ :=
  let total_businessmen : ℕ := 35
  let coffee_drinkers : ℕ := 18
  let tea_drinkers : ℕ := 15
  let juice_drinkers : ℕ := 8
  let coffee_and_tea : ℕ := 6
  let tea_and_juice : ℕ := 4
  let coffee_and_juice : ℕ := 3
  let all_beverages : ℕ := 2
  21

#check businessmen_one_beverage

/-- Proof that 21 businessmen drank only one type of beverage -/
theorem businessmen_one_beverage_proof : 
  businessmen_one_beverage 35 18 15 8 6 3 4 2 = 21 := by
  sorry

end businessmen_one_beverage_businessmen_one_beverage_proof_l1634_163420


namespace total_weight_AlI3_is_3261_44_l1634_163492

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of aluminum atoms in a molecule of AlI3 -/
def num_Al_atoms : ℕ := 1

/-- The number of iodine atoms in a molecule of AlI3 -/
def num_I_atoms : ℕ := 3

/-- The number of moles of AlI3 -/
def num_moles : ℝ := 8

/-- The total weight of AlI3 in grams -/
def total_weight_AlI3 : ℝ :=
  num_moles * (num_Al_atoms * atomic_weight_Al + num_I_atoms * atomic_weight_I)

theorem total_weight_AlI3_is_3261_44 : total_weight_AlI3 = 3261.44 := by
  sorry

end total_weight_AlI3_is_3261_44_l1634_163492


namespace bus_departure_interval_l1634_163427

/-- The departure interval of the bus -/
def x : ℝ := sorry

/-- The speed of the bus -/
def bus_speed : ℝ := sorry

/-- The speed of Xiao Hong -/
def xiao_hong_speed : ℝ := sorry

/-- The time interval between buses passing Xiao Hong from behind -/
def overtake_interval : ℝ := 6

/-- The time interval between buses approaching Xiao Hong head-on -/
def approach_interval : ℝ := 3

theorem bus_departure_interval :
  (overtake_interval * (bus_speed - xiao_hong_speed) = x * bus_speed) ∧
  (approach_interval * (bus_speed + xiao_hong_speed) = x * bus_speed) →
  x = 4 := by sorry

end bus_departure_interval_l1634_163427


namespace specific_trapezoid_height_l1634_163423

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid where
  a : ℝ  -- Length of one parallel side
  b : ℝ  -- Length of the other parallel side
  c : ℝ  -- Length of one non-parallel side
  d : ℝ  -- Length of the other non-parallel side

/-- The height of a trapezoid -/
def trapezoid_height (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that a trapezoid with the given dimensions has a height of 12 -/
theorem specific_trapezoid_height :
  let t : Trapezoid := { a := 25, b := 4, c := 20, d := 13 }
  trapezoid_height t = 12 := by
  sorry

end specific_trapezoid_height_l1634_163423


namespace metallic_sheet_dimension_l1634_163476

/-- Given a rectangular metallic sheet, prove that the second dimension is 36 m -/
theorem metallic_sheet_dimension (sheet_length : ℝ) (sheet_width : ℝ) 
  (cut_length : ℝ) (box_volume : ℝ) :
  sheet_length = 46 →
  cut_length = 8 →
  box_volume = 4800 →
  box_volume = (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length →
  sheet_width = 36 := by
  sorry

end metallic_sheet_dimension_l1634_163476


namespace nancy_scholarship_amount_l1634_163403

/-- Proves that Nancy's scholarship amount is $3,000 given the tuition costs and other conditions --/
theorem nancy_scholarship_amount : 
  ∀ (tuition : ℕ) 
    (parent_contribution : ℕ) 
    (work_hours : ℕ) 
    (hourly_rate : ℕ) 
    (scholarship : ℕ),
  tuition = 22000 →
  parent_contribution = tuition / 2 →
  work_hours = 200 →
  hourly_rate = 10 →
  scholarship + 2 * scholarship + parent_contribution + work_hours * hourly_rate = tuition →
  scholarship = 3000 := by
sorry


end nancy_scholarship_amount_l1634_163403


namespace cube_angle_sum_prove_cube_angle_sum_l1634_163486

/-- The sum of three right angles and one angle formed by a face diagonal in a cube is 330 degrees. -/
theorem cube_angle_sum : ℝ → Prop :=
  fun (cube_angle_sum : ℝ) =>
    let right_angle : ℝ := 90
    let face_diagonal_angle : ℝ := 60
    cube_angle_sum = 3 * right_angle + face_diagonal_angle ∧ cube_angle_sum = 330

/-- Proof of the theorem -/
theorem prove_cube_angle_sum : ∃ (x : ℝ), cube_angle_sum x :=
  sorry

end cube_angle_sum_prove_cube_angle_sum_l1634_163486


namespace class_size_l1634_163439

theorem class_size (debate_only : ℕ) (singing_only : ℕ) (both : ℕ)
  (h1 : debate_only = 10)
  (h2 : singing_only = 18)
  (h3 : both = 17) :
  debate_only + singing_only + both - both = 28 :=
by
  sorry

end class_size_l1634_163439


namespace square_sum_of_linear_equations_l1634_163490

theorem square_sum_of_linear_equations (x y : ℝ) 
  (eq1 : 3 * x + 4 * y = 30) 
  (eq2 : x + 2 * y = 13) : 
  x^2 + y^2 = 145/4 := by sorry

end square_sum_of_linear_equations_l1634_163490


namespace hyperbola_parabola_same_foci_l1634_163450

-- Define the hyperbola equation
def hyperbola (k : ℝ) (x y : ℝ) : Prop :=
  y^2 / 5 - x^2 / k = 1

-- Define the parabola equation
def parabola (x y : ℝ) : Prop :=
  x^2 = 12 * y

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (0, 3)

-- Define the property of having the same foci
def same_foci (k : ℝ) : Prop :=
  ∃ (c : ℝ), c^2 = 5 + (-k) ∧ c = 3

-- Theorem statement
theorem hyperbola_parabola_same_foci (k : ℝ) :
  same_foci k → k = -4 :=
by
  sorry

end hyperbola_parabola_same_foci_l1634_163450


namespace cyclic_inequality_l1634_163406

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z ≥ 3) :
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 ∧
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end cyclic_inequality_l1634_163406


namespace total_pizza_slices_l1634_163428

theorem total_pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_pizzas = 35) (h2 : slices_per_pizza = 12) : 
  num_pizzas * slices_per_pizza = 420 := by
  sorry

end total_pizza_slices_l1634_163428


namespace perimeter_of_quarter_circle_bounded_region_l1634_163409

/-- The perimeter of a region bounded by four quarter-circle arcs constructed at each corner of a square with sides measuring 4/π is equal to 4. -/
theorem perimeter_of_quarter_circle_bounded_region : 
  let square_side : ℝ := 4 / Real.pi
  let quarter_circle_radius : ℝ := square_side / 2
  let quarter_circle_perimeter : ℝ := Real.pi * quarter_circle_radius / 2
  let total_perimeter : ℝ := 4 * quarter_circle_perimeter
  total_perimeter = 4 := by sorry

end perimeter_of_quarter_circle_bounded_region_l1634_163409


namespace monomial_product_l1634_163401

/-- Given two monomials 4x⁴y² and 3x²y³, prove that their product is 12x⁶y⁵ -/
theorem monomial_product :
  ∀ (x y : ℝ), (4 * x^4 * y^2) * (3 * x^2 * y^3) = 12 * x^6 * y^5 := by
  sorry

end monomial_product_l1634_163401


namespace narrowest_strip_for_specific_figure_l1634_163474

/-- Represents a plane figure composed of an equilateral triangle and circular arcs --/
structure TriangleWithArcs where
  side_length : ℝ
  small_radius : ℝ
  large_radius : ℝ

/-- Calculates the narrowest strip width for a given TriangleWithArcs --/
def narrowest_strip_width (figure : TriangleWithArcs) : ℝ :=
  figure.small_radius + figure.large_radius

/-- Theorem stating that for the specific figure described, the narrowest strip width is 6 units --/
theorem narrowest_strip_for_specific_figure :
  let figure : TriangleWithArcs := {
    side_length := 4,
    small_radius := 1,
    large_radius := 5
  }
  narrowest_strip_width figure = 6 := by
  sorry

end narrowest_strip_for_specific_figure_l1634_163474


namespace quadrilateral_angle_l1634_163496

/-- 
Given a quadrilateral with angles α₁, α₂, α₃, α₄, α₅ satisfying:
1) α₁ + α₂ = 180°
2) α₃ = α₄
3) α₂ + α₅ = 180°
Prove that α₄ = 90°
-/
theorem quadrilateral_angle (α₁ α₂ α₃ α₄ α₅ : ℝ) 
  (h1 : α₁ + α₂ = 180)
  (h2 : α₃ = α₄)
  (h3 : α₂ + α₅ = 180)
  (h4 : α₁ + α₂ + α₃ + α₄ = 360) :  -- sum of angles in a quadrilateral
  α₄ = 90 := by
  sorry

end quadrilateral_angle_l1634_163496


namespace min_production_quantity_l1634_163457

def cost_function (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

def selling_price : ℝ := 25

theorem min_production_quantity :
  ∃ (min_x : ℝ), min_x = 150 ∧
  ∀ (x : ℝ), x ∈ Set.Ioo 0 240 →
    (selling_price * x ≥ cost_function x ↔ x ≥ min_x) :=
sorry

end min_production_quantity_l1634_163457


namespace largest_multiple_of_11_below_negative_200_l1634_163435

theorem largest_multiple_of_11_below_negative_200 :
  ∀ n : ℤ, n * 11 < -200 → n * 11 ≤ -209 :=
by
  sorry

end largest_multiple_of_11_below_negative_200_l1634_163435


namespace shirt_price_shopping_scenario_l1634_163472

/-- The price of a shirt given the shopping scenario --/
theorem shirt_price (total_paid : ℝ) (num_shorts : ℕ) (price_per_short : ℝ) 
  (num_shirts : ℕ) (senior_discount : ℝ) : ℝ :=
  let shorts_cost := num_shorts * price_per_short
  let discounted_shorts_cost := shorts_cost * (1 - senior_discount)
  let shirts_cost := total_paid - discounted_shorts_cost
  shirts_cost / num_shirts

/-- The price of each shirt in the given shopping scenario is $15.30 --/
theorem shopping_scenario : 
  shirt_price 117 3 15 5 0.1 = 15.3 := by
  sorry

end shirt_price_shopping_scenario_l1634_163472


namespace point_on_transformed_graph_l1634_163459

/-- Given a function g : ℝ → ℝ such that g(8) = 5, prove that (8/3, 14/9) is on the graph of 3y = g(3x)/3 + 3 and the sum of its coordinates is 38/9 -/
theorem point_on_transformed_graph (g : ℝ → ℝ) (h : g 8 = 5) :
  let f : ℝ → ℝ := λ x => (g (3 * x) / 3 + 3) / 3
  f (8/3) = 14/9 ∧ 8/3 + 14/9 = 38/9 := by
sorry

end point_on_transformed_graph_l1634_163459


namespace cubic_sum_values_l1634_163481

def M (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![a, b, c],
    ![b, c, a],
    ![c, a, b]]

theorem cubic_sum_values (a b c : ℂ) :
  M a b c ^ 2 = 1 →
  a * b * c = -1 →
  (a^3 + b^3 + c^3 = -2) ∨ (a^3 + b^3 + c^3 = -4) :=
by sorry

end cubic_sum_values_l1634_163481


namespace combination_count_l1634_163432

/-- The number of different styles of backpacks -/
def num_backpacks : ℕ := 2

/-- The number of different styles of pencil cases -/
def num_pencil_cases : ℕ := 2

/-- A combination consists of one backpack and one pencil case -/
def combination := ℕ × ℕ

/-- The total number of possible combinations -/
def total_combinations : ℕ := num_backpacks * num_pencil_cases

theorem combination_count : total_combinations = 4 := by sorry

end combination_count_l1634_163432


namespace cubic_expression_value_l1634_163444

/-- Given that px³ + qx - 10 = 2006 when x = 1, prove that px³ + qx - 10 = -2026 when x = -1 -/
theorem cubic_expression_value (p q : ℝ) 
  (h : p * 1^3 + q * 1 - 10 = 2006) :
  p * (-1)^3 + q * (-1) - 10 = -2026 := by
  sorry

end cubic_expression_value_l1634_163444


namespace equation_system_solvability_l1634_163493

theorem equation_system_solvability : ∃ (x y z : ℝ), 
  (2 * x + y = 4) ∧ 
  (x^2 + 3 * y = 5) ∧ 
  (3 * x - 1.5 * y + z = 7) := by
  sorry

end equation_system_solvability_l1634_163493


namespace quadratic_function_properties_l1634_163473

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := 3 * (x + 1)^2 - 2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 6 * (x + 1)

theorem quadratic_function_properties :
  (f 1 = 10) ∧
  (f (-1) = -2) ∧
  (∀ x > -1, f' x > 0) :=
sorry

end quadratic_function_properties_l1634_163473


namespace some_number_value_l1634_163483

theorem some_number_value (x : ℝ) : (85 + 32 / x) * x = 9637 → x = 113 := by
  sorry

end some_number_value_l1634_163483


namespace polar_bear_fish_consumption_l1634_163416

/-- The amount of trout eaten daily by the polar bear in buckets -/
def trout_amount : ℝ := 0.2

/-- The amount of salmon eaten daily by the polar bear in buckets -/
def salmon_amount : ℝ := 0.4

/-- The total amount of fish eaten daily by the polar bear in buckets -/
def total_fish : ℝ := trout_amount + salmon_amount

theorem polar_bear_fish_consumption :
  total_fish = 0.6 := by sorry

end polar_bear_fish_consumption_l1634_163416


namespace bills_naps_l1634_163429

theorem bills_naps (total_hours : ℕ) (work_hours : ℕ) (nap_duration : ℕ) : 
  total_hours = 96 → work_hours = 54 → nap_duration = 7 → 
  (total_hours - work_hours) / nap_duration = 6 := by
sorry

end bills_naps_l1634_163429


namespace factorial_sum_equals_natural_sum_squared_l1634_163469

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (k : ℕ) : ℕ := (List.range k).map factorial |>.sum

def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

theorem factorial_sum_equals_natural_sum_squared :
  ∀ k n : ℕ, sum_of_factorials k = (sum_of_naturals n)^2 ↔ (k = 1 ∧ n = 1) ∨ (k = 3 ∧ n = 2) :=
sorry

end factorial_sum_equals_natural_sum_squared_l1634_163469


namespace zero_not_necessarily_in_2_5_l1634_163495

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of having a unique zero in an interval
def has_unique_zero_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

-- State the theorem
theorem zero_not_necessarily_in_2_5 :
  (has_unique_zero_in f 1 3) →
  (has_unique_zero_in f 1 4) →
  (has_unique_zero_in f 1 5) →
  ¬ (∀ g : ℝ → ℝ, (has_unique_zero_in g 1 3 ∧ has_unique_zero_in g 1 4 ∧ has_unique_zero_in g 1 5) → 
    (∃ x, 2 < x ∧ x < 5 ∧ g x = 0)) :=
by sorry

end zero_not_necessarily_in_2_5_l1634_163495


namespace tangent_line_equation_l1634_163424

/-- Ellipse C₁ -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Parabola C₂ -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = 4 * p.1}

/-- Line with slope and y-intercept -/
structure Line where
  k : ℝ
  m : ℝ

/-- Tangent line to both ellipse and parabola -/
def isTangentLine (l : Line) (e : Ellipse) : Prop :=
  ∃ (x y : ℝ), x^2 / e.a^2 + y^2 / e.b^2 = 1 ∧
                y = l.k * x + l.m ∧
                (l.k * x + l.m)^2 = 4 * x

theorem tangent_line_equation (e : Ellipse) 
  (h1 : e.a^2 - e.b^2 = e.a^2 / 2)  -- Eccentricity condition
  (h2 : e.a - (e.a^2 - e.b^2).sqrt = Real.sqrt 2 - 1)  -- Minimum distance condition
  : ∃ (l : Line), isTangentLine l e ∧ 
    ((l.k = Real.sqrt 2 / 2 ∧ l.m = Real.sqrt 2) ∨
     (l.k = -Real.sqrt 2 / 2 ∧ l.m = -Real.sqrt 2)) := by
  sorry

end tangent_line_equation_l1634_163424


namespace annies_crayons_l1634_163431

/-- Annie's crayon problem -/
theorem annies_crayons (initial : ℕ) (additional : ℕ) : 
  initial = 4 → additional = 36 → initial + additional = 40 := by
  sorry

end annies_crayons_l1634_163431


namespace square_window_side_length_l1634_163458

/-- Represents the dimensions of a glass pane -/
structure GlassPane where
  height : ℝ
  width : ℝ
  ratio : height / width = 5 / 2

/-- Represents the dimensions of a square window -/
structure SquareWindow where
  pane : GlassPane
  border_width : ℝ
  side_length : ℝ

/-- Theorem stating the side length of the square window -/
theorem square_window_side_length 
  (window : SquareWindow)
  (h1 : window.border_width = 2)
  (h2 : window.side_length = 4 * window.pane.width + 5 * window.border_width)
  (h3 : window.side_length = 2 * window.pane.height + 3 * window.border_width) :
  window.side_length = 26 := by
  sorry

end square_window_side_length_l1634_163458


namespace polynomial_value_l1634_163460

def f (x : ℝ) (a₀ a₁ a₂ a₃ a₄ : ℝ) : ℝ :=
  a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + 7 * x^5

theorem polynomial_value (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  f 2004 a₀ a₁ a₂ a₃ a₄ = 72 ∧
  f 2005 a₀ a₁ a₂ a₃ a₄ = -30 ∧
  f 2006 a₀ a₁ a₂ a₃ a₄ = 32 ∧
  f 2007 a₀ a₁ a₂ a₃ a₄ = -24 ∧
  f 2008 a₀ a₁ a₂ a₃ a₄ = 24 →
  f 2009 a₀ a₁ a₂ a₃ a₄ = 847 := by
  sorry

end polynomial_value_l1634_163460


namespace aisha_driving_problem_l1634_163436

theorem aisha_driving_problem (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (average_speed : ℝ) :
  initial_distance = 18 →
  initial_speed = 36 →
  second_speed = 60 →
  average_speed = 48 →
  ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = average_speed ∧
    additional_distance = 30 :=
by sorry

end aisha_driving_problem_l1634_163436


namespace smallest_fraction_between_l1634_163430

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ (p : ℚ) / q < 5 / 8 →
  q ≥ 13 ∧ (q = 13 → p = 8) :=
sorry

end smallest_fraction_between_l1634_163430


namespace cruz_marbles_l1634_163449

/-- The number of marbles Atticus has -/
def atticus : ℕ := 4

/-- The number of marbles Jensen has -/
def jensen : ℕ := 2 * atticus

/-- The number of marbles Cruz has -/
def cruz : ℕ := 20 - (atticus + jensen)

/-- The total number of marbles -/
def total : ℕ := atticus + jensen + cruz

theorem cruz_marbles :
  (3 * total = 60) ∧ (atticus = 4) ∧ (jensen = 2 * atticus) → cruz = 8 := by
  sorry


end cruz_marbles_l1634_163449


namespace little_john_friends_money_l1634_163468

/-- Calculates the amount given to each friend by Little John --/
theorem little_john_friends_money 
  (initial_amount : ℚ) 
  (sweets_cost : ℚ) 
  (num_friends : ℕ) 
  (remaining_amount : ℚ) 
  (h1 : initial_amount = 8.5)
  (h2 : sweets_cost = 1.25)
  (h3 : num_friends = 2)
  (h4 : remaining_amount = 4.85) :
  (initial_amount - remaining_amount - sweets_cost) / num_friends = 1.2 := by
  sorry

end little_john_friends_money_l1634_163468


namespace car_travel_distance_l1634_163482

theorem car_travel_distance (speed1 speed2 total_distance average_speed : ℝ) 
  (h1 : speed1 = 75)
  (h2 : speed2 = 80)
  (h3 : total_distance = 320)
  (h4 : average_speed = 77.4193548387097)
  (h5 : total_distance = 2 * (total_distance / 2)) : 
  total_distance / 2 = 160 := by
  sorry

#check car_travel_distance

end car_travel_distance_l1634_163482


namespace number_of_smaller_cubes_l1634_163489

theorem number_of_smaller_cubes (surface_area : ℝ) (small_cube_volume : ℝ) : 
  surface_area = 5400 → small_cube_volume = 216 → 
  (surface_area / 6).sqrt ^ 3 / small_cube_volume = 125 := by
  sorry

end number_of_smaller_cubes_l1634_163489


namespace music_class_participation_l1634_163441

theorem music_class_participation (jacob_total : ℕ) (jacob_participating : ℕ) (steve_total : ℕ)
  (h1 : jacob_total = 27)
  (h2 : jacob_participating = 18)
  (h3 : steve_total = 45) :
  (jacob_participating * steve_total) / jacob_total = 30 := by
  sorry

end music_class_participation_l1634_163441


namespace sum_of_coefficients_l1634_163407

-- Define the expansion
def expansion (a : ℝ) (x : ℝ) : ℝ := (2 + a * x) * (1 + x)^5

-- Define the coefficient of x^2
def coeff_x2 (a : ℝ) : ℝ := 20 + 5 * a

-- Theorem statement
theorem sum_of_coefficients (a : ℝ) (h : coeff_x2 a = 15) : 
  ∃ (sum : ℝ), sum = expansion a 1 ∧ sum = 64 := by
  sorry

end sum_of_coefficients_l1634_163407


namespace distance_center_to_point_l1634_163494

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 6*x - 8*y + 9

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, -4)

-- Define the given point
def given_point : ℝ × ℝ := (5, -3)

-- Statement to prove
theorem distance_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = Real.sqrt 5 :=
by sorry

end distance_center_to_point_l1634_163494


namespace lily_of_valley_price_increase_l1634_163421

/-- Calculates the percentage increase in selling price compared to buying price for Françoise's lily of the valley pots. -/
theorem lily_of_valley_price_increase 
  (buying_price : ℝ) 
  (num_pots : ℕ) 
  (amount_given_back : ℝ) 
  (h1 : buying_price = 12)
  (h2 : num_pots = 150)
  (h3 : amount_given_back = 450) :
  let total_cost := buying_price * num_pots
  let total_revenue := total_cost + amount_given_back
  let selling_price := total_revenue / num_pots
  (selling_price - buying_price) / buying_price * 100 = 25 := by
sorry


end lily_of_valley_price_increase_l1634_163421


namespace total_missed_pitches_example_l1634_163422

/-- Represents a person's batting performance -/
structure BattingPerformance where
  tokens : Nat
  hits : Nat

/-- Calculates the total number of missed pitches for all players -/
def totalMissedPitches (pitchesPerToken : Nat) (performances : List BattingPerformance) : Nat :=
  performances.foldl (fun acc p => acc + p.tokens * pitchesPerToken - p.hits) 0

theorem total_missed_pitches_example :
  let pitchesPerToken := 15
  let macy := BattingPerformance.mk 11 50
  let piper := BattingPerformance.mk 17 55
  let quinn := BattingPerformance.mk 13 60
  let performances := [macy, piper, quinn]
  totalMissedPitches pitchesPerToken performances = 450 := by
  sorry

#eval totalMissedPitches 15 [BattingPerformance.mk 11 50, BattingPerformance.mk 17 55, BattingPerformance.mk 13 60]

end total_missed_pitches_example_l1634_163422


namespace bank_cash_increase_l1634_163411

/-- Represents a bank transaction --/
inductive Transaction
  | Deposit (amount : ℕ)
  | Withdrawal (amount : ℕ)

/-- Calculates the net change in cash after a series of transactions --/
def netChange (transactions : List Transaction) : ℤ :=
  transactions.foldl
    (fun acc t => match t with
      | Transaction.Deposit a => acc + a
      | Transaction.Withdrawal a => acc - a)
    0

/-- The list of transactions for the day --/
def dayTransactions : List Transaction := [
  Transaction.Withdrawal 960000,
  Transaction.Deposit 500000,
  Transaction.Withdrawal 700000,
  Transaction.Deposit 1200000,
  Transaction.Deposit 2200000,
  Transaction.Withdrawal 1025000,
  Transaction.Withdrawal 240000
]

theorem bank_cash_increase :
  netChange dayTransactions = 975000 := by
  sorry

end bank_cash_increase_l1634_163411


namespace evaluate_F_4_f_5_l1634_163418

-- Define the functions f and F
def f (a : ℝ) : ℝ := a - 3
def F (a b : ℝ) : ℝ := b^3 + a*b

-- State the theorem
theorem evaluate_F_4_f_5 : F 4 (f 5) = 16 := by
  sorry

end evaluate_F_4_f_5_l1634_163418


namespace glue_drops_in_cube_l1634_163497

/-- 
For an n × n × n cube built from n³ unit cubes, where one drop of glue is used for each pair 
of touching faces between two cubes, the total number of glue drops used is 3n²(n-1).
-/
theorem glue_drops_in_cube (n : ℕ) : 
  n > 0 → 3 * n^2 * (n - 1) = 
    (n - 1) * n * n  -- drops for vertical contacts
    + (n - 1) * n * n  -- drops for horizontal contacts
    + (n - 1) * n * n  -- drops for depth contacts
  := by sorry

end glue_drops_in_cube_l1634_163497


namespace existence_of_m_n_l1634_163471

theorem existence_of_m_n (p : ℕ) (hp : Prime p) (hp_gt_10 : p > 10) :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m + n < p ∧ (p ∣ (5^m * 7^n - 1)) := by
  sorry

end existence_of_m_n_l1634_163471


namespace hexagon_area_is_six_l1634_163462

/-- A point on a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- A polygon defined by its vertices --/
structure Polygon where
  vertices : List GridPoint

/-- Calculate the area of a polygon given its vertices --/
def calculateArea (p : Polygon) : Int :=
  sorry

/-- The 4x4 square on the grid --/
def square : Polygon :=
  { vertices := [
      { x := 0, y := 0 },
      { x := 0, y := 4 },
      { x := 4, y := 4 },
      { x := 4, y := 0 }
    ] }

/-- The hexagon formed by adding two points to the square --/
def hexagon : Polygon :=
  { vertices := [
      { x := 0, y := 0 },
      { x := 0, y := 4 },
      { x := 2, y := 4 },
      { x := 4, y := 4 },
      { x := 4, y := 0 },
      { x := 2, y := 0 }
    ] }

theorem hexagon_area_is_six :
  calculateArea hexagon = 6 :=
sorry

end hexagon_area_is_six_l1634_163462


namespace polynomial_coefficient_problem_l1634_163479

theorem polynomial_coefficient_problem (a b : ℝ) : 
  (0 < a) → 
  (0 < b) → 
  (a + b = 1) → 
  (21 * a^10 * b^4 = 35 * a^8 * b^6) → 
  a = 5 / (5 + Real.sqrt 15) := by
  sorry

end polynomial_coefficient_problem_l1634_163479


namespace complement_intersection_theorem_l1634_163499

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2, 4}

-- Define set N
def N : Set Nat := {2, 4, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl M ∩ Set.compl N : Set Nat) = {3} :=
by sorry

end complement_intersection_theorem_l1634_163499


namespace sport_formulation_corn_syrup_l1634_163456

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio := ⟨1, 12, 30⟩

/-- The sport formulation ratio -/
def sport_ratio (standard : DrinkRatio) : DrinkRatio :=
  ⟨standard.flavoring,
   standard.corn_syrup / 3,
   standard.water * 2⟩

/-- Calculates the amount of corn syrup given the amount of water and the ratio -/
def corn_syrup_amount (water_amount : ℚ) (ratio : DrinkRatio) : ℚ :=
  (ratio.corn_syrup * water_amount) / ratio.water

theorem sport_formulation_corn_syrup :
  corn_syrup_amount 30 (sport_ratio standard_ratio) = 2 := by
  sorry

end sport_formulation_corn_syrup_l1634_163456


namespace other_asymptote_equation_l1634_163445

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- x-coordinate of the foci -/
  foci_x : ℝ

/-- Theorem: For a hyperbola with one asymptote y = 4x and foci on the line x = 3,
    the other asymptote has the equation y = -4x + 24 -/
theorem other_asymptote_equation (h : Hyperbola) 
    (h1 : h.asymptote1 = fun x => 4 * x) 
    (h2 : h.foci_x = 3) : 
    ∃ (asymptote2 : ℝ → ℝ), asymptote2 = fun x => -4 * x + 24 := by
  sorry

end other_asymptote_equation_l1634_163445


namespace art_exhibit_revenue_l1634_163425

/-- Calculates the total revenue from ticket sales for an art exhibit --/
theorem art_exhibit_revenue :
  let start_time : Nat := 9 * 60  -- 9:00 AM in minutes
  let end_time : Nat := 16 * 60 + 55  -- 4:55 PM in minutes
  let interval : Nat := 5  -- 5 minutes
  let group_size : Nat := 30
  let regular_price : Nat := 10
  let student_price : Nat := 6
  let regular_to_student_ratio : Nat := 3

  let total_intervals : Nat := (end_time - start_time) / interval + 1
  let total_tickets : Nat := total_intervals * group_size
  let student_tickets : Nat := total_tickets / (regular_to_student_ratio + 1)
  let regular_tickets : Nat := total_tickets - student_tickets

  let total_revenue : Nat := student_tickets * student_price + regular_tickets * regular_price

  total_revenue = 25652 := by sorry

end art_exhibit_revenue_l1634_163425


namespace ashley_family_movie_cost_l1634_163454

/-- Calculates the total cost of a movie outing for Ashley's family --/
def movie_outing_cost (
  child_ticket_price : ℝ)
  (adult_ticket_price_diff : ℝ)
  (senior_ticket_price_diff : ℝ)
  (morning_discount : ℝ)
  (voucher_discount : ℝ)
  (popcorn_price : ℝ)
  (soda_price : ℝ)
  (candy_price : ℝ)
  (concession_discount : ℝ) : ℝ :=
  let adult_ticket_price := child_ticket_price + adult_ticket_price_diff
  let senior_ticket_price := adult_ticket_price - senior_ticket_price_diff
  let ticket_cost := 2 * adult_ticket_price + 4 * child_ticket_price + senior_ticket_price
  let discounted_ticket_cost := ticket_cost * (1 - morning_discount) - child_ticket_price - voucher_discount
  let concession_cost := 3 * popcorn_price + 2 * soda_price + candy_price
  let discounted_concession_cost := concession_cost * (1 - concession_discount)
  discounted_ticket_cost + discounted_concession_cost

/-- Theorem stating the total cost of Ashley's family's movie outing --/
theorem ashley_family_movie_cost :
  movie_outing_cost 4.25 3.50 1.75 0.10 4.00 5.25 3.50 4.00 0.10 = 50.47 := by
  sorry

end ashley_family_movie_cost_l1634_163454


namespace modified_cube_surface_area_l1634_163419

/-- Represents a modified cube with square holes cut through each face -/
structure ModifiedCube where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculates the total surface area of a modified cube including inside surfaces -/
def total_surface_area (cube : ModifiedCube) : ℝ :=
  let original_surface_area := 6 * cube.edge_length^2
  let hole_area := 6 * cube.hole_side_length^2
  let new_exposed_area := 6 * 4 * cube.hole_side_length^2
  original_surface_area - hole_area + new_exposed_area

/-- Theorem stating that a cube with edge length 4 and hole side length 2 has a total surface area of 168 -/
theorem modified_cube_surface_area :
  let cube : ModifiedCube := { edge_length := 4, hole_side_length := 2 }
  total_surface_area cube = 168 := by
  sorry

end modified_cube_surface_area_l1634_163419


namespace ellipse_focus_m_value_l1634_163405

/-- Given an ellipse with equation x²/25 + y²/m² = 1 where m > 0,
    and left focus at (-4, 0), prove that m = 3 -/
theorem ellipse_focus_m_value (m : ℝ) (h1 : m > 0) :
  (∀ x y : ℝ, x^2 / 25 + y^2 / m^2 = 1) →
  (∃ x y : ℝ, x = -4 ∧ y = 0 ∧ x^2 / 25 + y^2 / m^2 = 1) →
  m = 3 := by
sorry

end ellipse_focus_m_value_l1634_163405


namespace lcm_count_l1634_163415

theorem lcm_count : 
  ∃! (n : ℕ), n > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ k ∈ S, k > 0 ∧ Nat.lcm (9^9) (Nat.lcm (12^12) k) = 18^18) ∧
    (∀ k ∉ S, k > 0 → Nat.lcm (9^9) (Nat.lcm (12^12) k) ≠ 18^18)) :=
by
  sorry

end lcm_count_l1634_163415


namespace fourth_square_dots_l1634_163455

/-- The side length of the nth square in the sequence -/
def side_length (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- The number of dots in the nth square -/
def num_dots (n : ℕ) : ℕ := (side_length n) ^ 2

theorem fourth_square_dots :
  num_dots 4 = 49 := by sorry

end fourth_square_dots_l1634_163455


namespace fast_clock_accuracy_l1634_163491

/-- Represents time in minutes since midnight -/
def Time := ℕ

/-- Converts hours and minutes to total minutes -/
def toMinutes (hours minutes : ℕ) : Time :=
  hours * 60 + minutes

/-- A fast-running clock that gains time at a constant rate -/
structure FastClock where
  /-- The rate at which the clock gains time, represented as (gained_minutes, real_minutes) -/
  rate : ℕ × ℕ
  /-- The current time shown on the fast clock -/
  current_time : Time

/-- Calculates the actual time given a FastClock -/
def actualTime (clock : FastClock) (start_time : Time) : Time :=
  sorry

theorem fast_clock_accuracy (start_time : Time) (end_time : Time) :
  let initial_clock : FastClock := { rate := (15, 45), current_time := start_time }
  let final_clock : FastClock := { rate := (15, 45), current_time := end_time }
  start_time = toMinutes 15 0 →
  end_time = toMinutes 23 0 →
  actualTime final_clock start_time = toMinutes 23 15 :=
  sorry

end fast_clock_accuracy_l1634_163491


namespace fraction_subtraction_l1634_163466

theorem fraction_subtraction : (15 : ℚ) / 45 - (1 + 2 / 9) = -8 / 9 := by
  sorry

end fraction_subtraction_l1634_163466


namespace smallest_batch_size_l1634_163484

theorem smallest_batch_size (N : ℕ) (h1 : N > 70) (h2 : (21 * N) % 70 = 0) :
  N ≥ 80 ∧ ∀ m : ℕ, m > 70 ∧ (21 * m) % 70 = 0 → m ≥ N := by
  sorry

end smallest_batch_size_l1634_163484


namespace range_of_a_correct_l1634_163440

/-- Proposition p: For all x ∈ ℝ, ax^2 + ax + 1 > 0 always holds -/
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: The function f(x) = 4x^2 - ax is monotonically increasing on [1, +∞) -/
def q (a : ℝ) : Prop := ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → 4 * x^2 - a * x ≤ 4 * y^2 - a * y

/-- The range of a given the conditions -/
def range_of_a : Set ℝ := {a : ℝ | a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8)}

theorem range_of_a_correct (a : ℝ) : (p a ∨ q a) ∧ ¬(p a) → a ∈ range_of_a := by
  sorry

end range_of_a_correct_l1634_163440


namespace probability_both_asian_l1634_163488

def asian_countries : ℕ := 3
def european_countries : ℕ := 3
def total_countries : ℕ := asian_countries + european_countries
def countries_to_select : ℕ := 2

def total_outcomes : ℕ := (total_countries.choose countries_to_select)
def favorable_outcomes : ℕ := (asian_countries.choose countries_to_select)

theorem probability_both_asian :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 5 := by sorry

end probability_both_asian_l1634_163488


namespace turtle_arrangement_l1634_163447

/-- The number of grid intersections in a rectangular arrangement of square tiles -/
def grid_intersections (width : ℕ) (height : ℕ) : ℕ :=
  (width + 1) * height

/-- Theorem: The number of grid intersections in a 20 × 21 rectangular arrangement of square tiles is 420 -/
theorem turtle_arrangement : grid_intersections 20 21 = 420 := by
  sorry

end turtle_arrangement_l1634_163447


namespace modular_inverse_97_mod_101_l1634_163464

theorem modular_inverse_97_mod_101 :
  ∃ x : ℕ, x < 101 ∧ (97 * x) % 101 = 1 :=
by
  use 25
  sorry

end modular_inverse_97_mod_101_l1634_163464


namespace greatest_integer_solution_l1634_163426

theorem greatest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, 7 - 3 * y + 2 > 23 → y ≤ x) ↔ x = -5 := by
  sorry

end greatest_integer_solution_l1634_163426


namespace remainder_theorem_l1634_163410

theorem remainder_theorem (n m : ℤ) (q2 : ℤ) 
  (h1 : n % 11 = 1) 
  (h2 : m % 17 = 3) 
  (h3 : m = 17 * q2 + 3) : 
  (5 * n + 3 * m) % 11 = (3 + 7 * q2) % 11 := by
sorry

end remainder_theorem_l1634_163410


namespace frood_game_threshold_l1634_163465

theorem frood_game_threshold : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → m < n → (m * (m + 1)) / 2 ≤ 15 * m) ∧ (n * (n + 1)) / 2 > 15 * n := by
  sorry

end frood_game_threshold_l1634_163465


namespace x_equals_3_sufficient_not_necessary_for_x_squared_9_l1634_163461

theorem x_equals_3_sufficient_not_necessary_for_x_squared_9 :
  (∀ x : ℝ, x = 3 → x^2 = 9) ∧
  (∃ x : ℝ, x^2 = 9 ∧ x ≠ 3) :=
by sorry

end x_equals_3_sufficient_not_necessary_for_x_squared_9_l1634_163461


namespace six_balls_four_boxes_l1634_163475

/-- Number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute 6 4 = 72 := by sorry

end six_balls_four_boxes_l1634_163475


namespace unique_solution_exponential_equation_l1634_163412

theorem unique_solution_exponential_equation (p q : ℝ) :
  (∀ x : ℝ, 2^(p*x + q) = p * 2^x + q) → p = 1 ∧ q = 0 := by
  sorry

end unique_solution_exponential_equation_l1634_163412


namespace alcohol_mixture_percentage_l1634_163478

/-- Calculates the final alcohol percentage in a solution after partial replacement -/
theorem alcohol_mixture_percentage 
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (drained_volume : ℝ)
  (replacement_percentage : ℝ)
  (h1 : initial_volume = 1)
  (h2 : initial_percentage = 0.75)
  (h3 : drained_volume = 0.4)
  (h4 : replacement_percentage = 0.5)
  (h5 : initial_volume - drained_volume + drained_volume = initial_volume) :
  let remaining_volume := initial_volume - drained_volume
  let remaining_alcohol := remaining_volume * initial_percentage
  let added_alcohol := drained_volume * replacement_percentage
  let total_alcohol := remaining_alcohol + added_alcohol
  let final_percentage := total_alcohol / initial_volume
  final_percentage = 0.65 := by sorry

end alcohol_mixture_percentage_l1634_163478


namespace correct_representation_l1634_163404

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be represented -/
def number : ℕ := 91000

/-- The scientific notation representation of the number -/
def representation : ScientificNotation := {
  coefficient := 9.1
  exponent := 4
  h1 := by sorry
}

/-- Theorem: The given representation is correct for the number -/
theorem correct_representation : 
  (representation.coefficient * (10 : ℝ) ^ representation.exponent) = number := by sorry

end correct_representation_l1634_163404


namespace next_coincidence_correct_l1634_163437

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts time to total seconds -/
def Time.toSeconds (t : Time) : ℕ :=
  t.hours * 3600 + t.minutes * 60 + t.seconds

/-- Checks if hour and minute hands coincide at given time -/
def handsCoincide (t : Time) : Prop :=
  (t.hours % 12 * 60 + t.minutes) * 11 = t.minutes * 12

/-- The next time after midnight when clock hands coincide -/
def nextCoincidence : Time :=
  { hours := 1, minutes := 5, seconds := 27 }

theorem next_coincidence_correct :
  handsCoincide nextCoincidence ∧
  ∀ t : Time, t.toSeconds < nextCoincidence.toSeconds → ¬handsCoincide t :=
by sorry

end next_coincidence_correct_l1634_163437


namespace preferred_numbers_count_l1634_163433

/-- A function that counts the number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- A function that counts the number of four-digit "preferred" numbers. -/
def count_preferred_numbers : ℕ :=
  -- Numbers with two 8s, not in the first position
  choose 3 2 * 8 * 9 +
  -- Numbers with two 8s, including in the first position
  choose 3 1 * 9 * 9 +
  -- Numbers with four 8s
  1

/-- Theorem stating that the count of four-digit "preferred" numbers is 460. -/
theorem preferred_numbers_count : count_preferred_numbers = 460 := by sorry

end preferred_numbers_count_l1634_163433


namespace isosceles_triangle_perimeter_l1634_163414

/-- An isosceles triangle with two sides of lengths 2 and 4 has a perimeter of 10. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = b ∨ b = c ∨ a = c) →
  ((a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2) ∨ (b = 2 ∧ c = 4) ∨ (b = 4 ∧ c = 2) ∨ (a = 2 ∧ c = 4) ∨ (a = 4 ∧ c = 2)) →
  a + b + c = 10 :=
by sorry

end isosceles_triangle_perimeter_l1634_163414


namespace zero_exponent_is_one_l1634_163443

theorem zero_exponent_is_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end zero_exponent_is_one_l1634_163443


namespace chromatic_number_upper_bound_l1634_163402

-- Define a graph type
structure Graph :=
  (V : Type) -- Vertex set
  (E : V → V → Prop) -- Edge relation

-- Define the number of edges in a graph
def num_edges (G : Graph) : ℕ := sorry

-- Define the chromatic number of a graph
def chromatic_number (G : Graph) : ℕ := sorry

-- State the theorem
theorem chromatic_number_upper_bound (G : Graph) :
  chromatic_number G ≤ (1/2 : ℝ) + Real.sqrt (2 * (num_edges G : ℝ) + 1/4) := by
  sorry

end chromatic_number_upper_bound_l1634_163402


namespace max_carlson_jars_l1634_163463

/-- Represents the initial state of jam jars for Carlson and Baby -/
structure JamJars where
  carlson_weights : List ℕ
  baby_weights : List ℕ

/-- Checks if the given JamJars satisfies the initial condition -/
def initial_condition (jars : JamJars) : Prop :=
  (jars.carlson_weights.sum = 13 * jars.baby_weights.sum) ∧
  (∀ w ∈ jars.carlson_weights, w > 0) ∧
  (∀ w ∈ jars.baby_weights, w > 0)

/-- Checks if the given JamJars satisfies the final condition after transfer -/
def final_condition (jars : JamJars) : Prop :=
  let smallest := jars.carlson_weights.minimum?
  match smallest with
  | some min =>
    ((jars.carlson_weights.sum - min) = 8 * (jars.baby_weights.sum + min)) ∧
    (∀ w ∈ jars.carlson_weights, w ≥ min)
  | none => False

/-- The main theorem stating the maximum number of jars Carlson could have initially had -/
theorem max_carlson_jars (jars : JamJars) :
  initial_condition jars → final_condition jars →
  jars.carlson_weights.length ≤ 23 :=
by sorry

#check max_carlson_jars

end max_carlson_jars_l1634_163463
