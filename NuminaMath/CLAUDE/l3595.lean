import Mathlib

namespace apple_students_count_l3595_359544

/-- Represents the total number of degrees in a circle -/
def total_degrees : ℕ := 360

/-- Represents the number of degrees in a right angle -/
def right_angle : ℕ := 90

/-- Represents the number of students who chose bananas -/
def banana_students : ℕ := 168

/-- Calculates the number of students who chose apples given the conditions -/
def apple_students : ℕ :=
  (right_angle * (banana_students * 4 / 3)) / total_degrees

theorem apple_students_count : apple_students = 56 := by
  sorry

end apple_students_count_l3595_359544


namespace largest_value_l3595_359504

def expr_a : ℝ := 3 - 1 + 4 + 6
def expr_b : ℝ := 3 - 1 * 4 + 6
def expr_c : ℝ := 3 - (1 + 4) * 6
def expr_d : ℝ := 3 - 1 + 4 * 6
def expr_e : ℝ := 3 * (1 - 4) + 6

theorem largest_value :
  expr_d = 26 ∧
  expr_d > expr_a ∧
  expr_d > expr_b ∧
  expr_d > expr_c ∧
  expr_d > expr_e :=
by sorry

end largest_value_l3595_359504


namespace not_sufficient_not_necessary_condition_l3595_359533

theorem not_sufficient_not_necessary_condition (a b : ℝ) :
  ¬(∀ a b : ℝ, a + b > 0 → a * b > 0) ∧ ¬(∀ a b : ℝ, a * b > 0 → a + b > 0) := by
  sorry

end not_sufficient_not_necessary_condition_l3595_359533


namespace power_of_five_l3595_359512

theorem power_of_five (x : ℕ) : 121 * (5^x) = 75625 ↔ x = 4 := by
  sorry

end power_of_five_l3595_359512


namespace water_saved_in_june_john_water_savings_l3595_359588

/-- Calculates the water saved in June by replacing an inefficient toilet with a more efficient one. -/
theorem water_saved_in_june (old_toilet_usage : ℝ) (flushes_per_day : ℕ) (water_reduction_percentage : ℝ) (days_in_june : ℕ) : ℝ :=
  let new_toilet_usage := old_toilet_usage * (1 - water_reduction_percentage)
  let daily_old_usage := old_toilet_usage * flushes_per_day
  let daily_new_usage := new_toilet_usage * flushes_per_day
  let june_old_usage := daily_old_usage * days_in_june
  let june_new_usage := daily_new_usage * days_in_june
  june_old_usage - june_new_usage

/-- Proves that John saved 1800 gallons of water in June by replacing his old toilet. -/
theorem john_water_savings : water_saved_in_june 5 15 0.8 30 = 1800 := by
  sorry

end water_saved_in_june_john_water_savings_l3595_359588


namespace ellipse_sum_l3595_359534

/-- Represents an ellipse with center (h, k) and semi-axes a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.b^2 = 1

theorem ellipse_sum (e : Ellipse) :
  e.h = 5 ∧ e.k = -3 ∧ e.a = 7 ∧ e.b = 4 →
  e.h + e.k + e.a + e.b = 13 := by
  sorry

end ellipse_sum_l3595_359534


namespace fraction_sum_and_multiply_l3595_359595

theorem fraction_sum_and_multiply :
  3 * (2 / 10 + 4 / 20 + 6 / 30) = 9 / 5 := by
  sorry

end fraction_sum_and_multiply_l3595_359595


namespace expression_value_l3595_359555

theorem expression_value : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 := by
  sorry

end expression_value_l3595_359555


namespace oliver_presentation_appropriate_l3595_359525

/-- Represents a presentation with a given word count. -/
structure Presentation where
  word_count : ℕ

/-- Checks if a presentation is appropriate given the speaking rate and time constraints. -/
def is_appropriate_presentation (p : Presentation) (speaking_rate : ℕ) (min_time : ℕ) (max_time : ℕ) : Prop :=
  let min_words := speaking_rate * min_time
  let max_words := speaking_rate * max_time
  min_words ≤ p.word_count ∧ p.word_count ≤ max_words

theorem oliver_presentation_appropriate :
  let speaking_rate := 120
  let min_time := 40
  let max_time := 55
  let presentation1 := Presentation.mk 5000
  let presentation2 := Presentation.mk 6200
  is_appropriate_presentation presentation1 speaking_rate min_time max_time ∧
  is_appropriate_presentation presentation2 speaking_rate min_time max_time :=
by sorry

end oliver_presentation_appropriate_l3595_359525


namespace text_message_plan_cost_l3595_359539

/-- The cost per text message for the first plan -/
def cost_per_text_plan1 : ℚ := 1/4

/-- The monthly fee for the first plan -/
def monthly_fee_plan1 : ℚ := 9

/-- The cost per text message for the second plan -/
def cost_per_text_plan2 : ℚ := 2/5

/-- The number of text messages for which both plans cost the same -/
def equal_cost_messages : ℕ := 60

theorem text_message_plan_cost : 
  monthly_fee_plan1 + equal_cost_messages * cost_per_text_plan1 = 
  equal_cost_messages * cost_per_text_plan2 :=
by sorry

end text_message_plan_cost_l3595_359539


namespace quadrilateral_diagonal_length_l3595_359500

theorem quadrilateral_diagonal_length 
  (area : ℝ) 
  (offset1 : ℝ) 
  (offset2 : ℝ) 
  (h1 : area = 300) 
  (h2 : offset1 = 9) 
  (h3 : offset2 = 6) : 
  area = (1/2) * (offset1 + offset2) * 40 :=
by sorry

end quadrilateral_diagonal_length_l3595_359500


namespace smallest_resolvable_debt_l3595_359501

theorem smallest_resolvable_debt (pig_value goat_value : ℕ) 
  (h_pig : pig_value = 350) (h_goat : goat_value = 240) :
  ∃ (debt : ℕ), debt > 0 ∧ 
  (∀ (d : ℕ), d > 0 → (∃ (p g : ℤ), d = pig_value * p + goat_value * g) → d ≥ debt) ∧
  (∃ (p g : ℤ), debt = pig_value * p + goat_value * g) :=
sorry

end smallest_resolvable_debt_l3595_359501


namespace trigonometric_identity_l3595_359519

theorem trigonometric_identity (α : Real) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * (Real.cos (π / 6 + α / 2))^2 - 1 = 1 / 3 := by
  sorry

end trigonometric_identity_l3595_359519


namespace magnitude_unit_vector_times_vector_l3595_359593

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

/-- Given a unit vector e and a non-zero vector b, prove that |e|*b = b -/
theorem magnitude_unit_vector_times_vector (e b : n) 
  (h_unit : ‖e‖ = 1) (h_nonzero : b ≠ 0) : 
  ‖e‖ • b = b := by
  sorry

end magnitude_unit_vector_times_vector_l3595_359593


namespace x_value_from_fraction_equality_l3595_359557

theorem x_value_from_fraction_equality (x y : ℝ) :
  x / (x - 1) = (y^2 + 2*y + 3) / (y^2 + 2*y + 2) →
  x = y^2 + 2*y + 3 := by
sorry

end x_value_from_fraction_equality_l3595_359557


namespace not_necessarily_right_triangle_l3595_359546

/-- Given a triangle ABC with angle ratio ∠A:∠B:∠C = 3:4:5, it cannot be concluded that ABC is a right triangle. -/
theorem not_necessarily_right_triangle (A B C : ℝ) (h : A / (A + B + C) = 3 / 12 ∧ B / (A + B + C) = 4 / 12 ∧ C / (A + B + C) = 5 / 12) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end not_necessarily_right_triangle_l3595_359546


namespace compound_proposition_truth_l3595_359586

theorem compound_proposition_truth : 
  (∀ x : ℝ, x < 0 → (2 : ℝ)^x > (3 : ℝ)^x) ∧ 
  (∃ x : ℝ, x > 0 ∧ Real.sqrt x > x^3) := by
sorry

end compound_proposition_truth_l3595_359586


namespace oil_drop_probability_l3595_359537

theorem oil_drop_probability (circle_diameter : ℝ) (square_side : ℝ) 
  (h1 : circle_diameter = 3) 
  (h2 : square_side = 1) : 
  (square_side ^ 2) / (π * (circle_diameter / 2) ^ 2) = 4 / (9 * π) :=
sorry

end oil_drop_probability_l3595_359537


namespace salary_reduction_percentage_l3595_359503

theorem salary_reduction_percentage (x : ℝ) : 
  (100 - x + (100 - x) * (11.11111111111111 / 100) = 100) → x = 10 := by
  sorry

end salary_reduction_percentage_l3595_359503


namespace bicycle_wheel_revolutions_l3595_359508

/-- Calculates the number of revolutions of the back wheel given the diameters of both wheels and the number of revolutions of the front wheel. -/
theorem bicycle_wheel_revolutions 
  (front_diameter : ℝ) 
  (back_diameter : ℝ) 
  (front_revolutions : ℝ) : 
  front_diameter = 28 →
  back_diameter = 20 →
  front_revolutions = 50 →
  (back_diameter / front_diameter) * front_revolutions = 70 := by
sorry

end bicycle_wheel_revolutions_l3595_359508


namespace sin_225_degrees_l3595_359580

theorem sin_225_degrees :
  Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_225_degrees_l3595_359580


namespace doughnuts_per_box_l3595_359548

theorem doughnuts_per_box (total_doughnuts : ℕ) (num_boxes : ℕ) (doughnuts_per_box : ℕ) : 
  total_doughnuts = 48 → 
  num_boxes = 4 → 
  total_doughnuts = num_boxes * doughnuts_per_box →
  doughnuts_per_box = 12 := by
  sorry

end doughnuts_per_box_l3595_359548


namespace actual_distance_scientific_notation_l3595_359577

/-- The scale of the map -/
def map_scale : ℚ := 1 / 8000000

/-- The distance between A and B on the map in centimeters -/
def map_distance : ℚ := 3.5

/-- The actual distance between A and B in centimeters -/
def actual_distance : ℕ := 28000000

/-- Theorem stating that the actual distance is equal to 2.8 × 10^7 -/
theorem actual_distance_scientific_notation : 
  (actual_distance : ℝ) = 2.8 * (10 : ℝ)^7 := by
  sorry

end actual_distance_scientific_notation_l3595_359577


namespace fraction_multiplication_result_l3595_359516

theorem fraction_multiplication_result : 
  (5 / 8 : ℚ) * (7 / 12 : ℚ) * (3 / 7 : ℚ) * 1350 = 210.9375 := by
  sorry

end fraction_multiplication_result_l3595_359516


namespace waiter_customers_l3595_359571

/-- Calculates the number of customers a waiter has after some tables leave --/
def customers_remaining (initial_tables : Float) (tables_left : Float) (customers_per_table : Float) : Float :=
  (initial_tables - tables_left) * customers_per_table

/-- Theorem: Given the initial conditions, the waiter has 256.0 customers --/
theorem waiter_customers :
  customers_remaining 44.0 12.0 8.0 = 256.0 := by
  sorry

end waiter_customers_l3595_359571


namespace megan_cupcakes_per_package_l3595_359545

/-- Calculates the number of cupcakes per package given the initial number of cupcakes,
    the number of cupcakes eaten, and the number of packages. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Proves that given 68 initial cupcakes, 32 cupcakes eaten, and 6 packages,
    the number of cupcakes in each package is 6. -/
theorem megan_cupcakes_per_package :
  cupcakes_per_package 68 32 6 = 6 := by
  sorry

end megan_cupcakes_per_package_l3595_359545


namespace percent_difference_l3595_359569

theorem percent_difference (x y p : ℝ) (h : x = y * (1 + p / 100)) : 
  p = 100 * ((x - y) / y) := by
  sorry

end percent_difference_l3595_359569


namespace exists_solution_for_calendar_equation_l3595_359538

theorem exists_solution_for_calendar_equation :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
  sorry

end exists_solution_for_calendar_equation_l3595_359538


namespace arithmetic_sqrt_of_three_l3595_359592

theorem arithmetic_sqrt_of_three (x : ℝ) : x = Real.sqrt 3 ↔ x ≥ 0 ∧ x ^ 2 = 3 := by sorry

end arithmetic_sqrt_of_three_l3595_359592


namespace men_count_in_alternating_arrangement_l3595_359573

/-- Represents the number of arrangements for a given number of men and women -/
def arrangements (men : ℕ) (women : ℕ) : ℕ := sorry

/-- Represents whether men and women are alternating in an arrangement -/
def isAlternating (men : ℕ) (women : ℕ) : Prop := sorry

theorem men_count_in_alternating_arrangement :
  ∀ (men : ℕ),
  (women : ℕ) → women = 2 →
  isAlternating men women →
  arrangements men women = 12 →
  men = 4 := by sorry

end men_count_in_alternating_arrangement_l3595_359573


namespace oliver_william_money_difference_l3595_359584

/-- Calculates the total amount of money given the number of bills of different denominations -/
def calculate_total (twenty_bills ten_bills five_bills : ℕ) : ℕ :=
  20 * twenty_bills + 10 * ten_bills + 5 * five_bills

/-- Represents the problem of comparing Oliver's and William's money -/
theorem oliver_william_money_difference :
  let oliver_total := calculate_total 10 0 3
  let william_total := calculate_total 0 15 4
  oliver_total - william_total = 45 := by sorry

end oliver_william_money_difference_l3595_359584


namespace probability_of_ravi_selection_l3595_359529

theorem probability_of_ravi_selection 
  (p_ram : ℝ) 
  (p_both : ℝ) 
  (h1 : p_ram = 6/7)
  (h2 : p_both = 0.17142857142857143) :
  p_both / p_ram = 0.2 :=
sorry

end probability_of_ravi_selection_l3595_359529


namespace oysters_with_pearls_percentage_l3595_359521

/-- The percentage of oysters with pearls, given the number of oysters collected per dive,
    the number of dives, and the total number of pearls collected. -/
def percentage_oysters_with_pearls (oysters_per_dive : ℕ) (num_dives : ℕ) (total_pearls : ℕ) : ℚ :=
  (total_pearls : ℚ) / ((oysters_per_dive * num_dives) : ℚ) * 100

/-- Theorem stating that the percentage of oysters with pearls is 25%,
    given the specific conditions from the problem. -/
theorem oysters_with_pearls_percentage :
  percentage_oysters_with_pearls 16 14 56 = 25 := by
  sorry


end oysters_with_pearls_percentage_l3595_359521


namespace table_cost_l3595_359515

/-- The cost of furniture items and payment details --/
structure FurniturePurchase where
  couch_cost : ℕ
  lamp_cost : ℕ
  initial_payment : ℕ
  remaining_balance : ℕ

/-- Theorem stating the cost of the table --/
theorem table_cost (purchase : FurniturePurchase)
  (h1 : purchase.couch_cost = 750)
  (h2 : purchase.lamp_cost = 50)
  (h3 : purchase.initial_payment = 500)
  (h4 : purchase.remaining_balance = 400) :
  ∃ (table_cost : ℕ), 
    purchase.couch_cost + table_cost + purchase.lamp_cost - purchase.initial_payment = purchase.remaining_balance ∧
    table_cost = 100 :=
sorry

end table_cost_l3595_359515


namespace stating_paint_usage_calculation_l3595_359565

/-- 
Given an initial amount of paint and usage fractions for two weeks,
calculate the total amount of paint used.
-/
def paint_used (initial_paint : ℝ) (week1_fraction : ℝ) (week2_fraction : ℝ) : ℝ :=
  let week1_usage := initial_paint * week1_fraction
  let remaining_paint := initial_paint - week1_usage
  let week2_usage := remaining_paint * week2_fraction
  week1_usage + week2_usage

/-- 
Theorem stating that given 360 gallons of initial paint, 
using 1/4 of all paint in the first week and 1/6 of the remaining paint 
in the second week results in a total usage of 135 gallons of paint.
-/
theorem paint_usage_calculation :
  paint_used 360 (1/4) (1/6) = 135 := by
  sorry


end stating_paint_usage_calculation_l3595_359565


namespace quadratic_factorization_l3595_359559

theorem quadratic_factorization (m : ℝ) : m^2 - 2*m + 1 = (m - 1)^2 := by
  sorry

end quadratic_factorization_l3595_359559


namespace cos_2alpha_plus_pi_third_l3595_359564

theorem cos_2alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/3) = 2/3) : 
  Real.cos (2*α + π/3) = -1/9 := by
sorry

end cos_2alpha_plus_pi_third_l3595_359564


namespace peters_remaining_money_l3595_359518

/-- Calculates Peter's remaining money after shopping at the market. -/
def remaining_money (initial_amount : ℕ) (potato_kg : ℕ) (potato_price : ℕ) 
  (tomato_kg : ℕ) (tomato_price : ℕ) (cucumber_kg : ℕ) (cucumber_price : ℕ) 
  (banana_kg : ℕ) (banana_price : ℕ) : ℕ :=
  initial_amount - (potato_kg * potato_price + tomato_kg * tomato_price + 
    cucumber_kg * cucumber_price + banana_kg * banana_price)

/-- Proves that Peter's remaining money after shopping is $426. -/
theorem peters_remaining_money : 
  remaining_money 500 6 2 9 3 5 4 3 5 = 426 := by
  sorry

end peters_remaining_money_l3595_359518


namespace area_of_triangle_FQH_area_of_triangle_FQH_proof_l3595_359522

-- Define the rectangle EFGH
structure Rectangle where
  EF : ℝ
  EH : ℝ

-- Define the trapezoid PRHG
structure Trapezoid where
  EP : ℝ
  area : ℝ

-- Define the problem setup
def problem (rect : Rectangle) (trap : Trapezoid) : Prop :=
  rect.EF = 16 ∧ 
  trap.EP = 8 ∧
  trap.area = 160

-- Theorem statement
theorem area_of_triangle_FQH (rect : Rectangle) (trap : Trapezoid) 
  (h : problem rect trap) : ℝ :=
  80

-- Proof
theorem area_of_triangle_FQH_proof (rect : Rectangle) (trap : Trapezoid) 
  (h : problem rect trap) : area_of_triangle_FQH rect trap h = 80 := by
  sorry

end area_of_triangle_FQH_area_of_triangle_FQH_proof_l3595_359522


namespace min_abs_plus_2023_min_value_abs_plus_2023_l3595_359549

theorem min_abs_plus_2023 (a : ℚ) : 
  (|a| + 2023 : ℚ) ≥ 2023 := by sorry

theorem min_value_abs_plus_2023 : 
  ∃ (m : ℚ), ∀ (a : ℚ), (|a| + 2023 : ℚ) ≥ m ∧ ∃ (b : ℚ), (|b| + 2023 : ℚ) = m := by
  use 2023
  sorry

end min_abs_plus_2023_min_value_abs_plus_2023_l3595_359549


namespace split_cost_12_cupcakes_at_1_50_l3595_359578

/-- The amount each person pays when two people buy cupcakes and split the cost evenly -/
def split_cost (num_cupcakes : ℕ) (price_per_cupcake : ℚ) : ℚ :=
  (num_cupcakes : ℚ) * price_per_cupcake / 2

/-- Theorem: When two people buy 12 cupcakes at $1.50 each and split the cost evenly, each person pays $9.00 -/
theorem split_cost_12_cupcakes_at_1_50 :
  split_cost 12 (3/2) = 9 := by
  sorry

end split_cost_12_cupcakes_at_1_50_l3595_359578


namespace max_cable_connections_l3595_359526

/-- Represents the number of computers of brand A -/
def brand_a_count : Nat := 28

/-- Represents the number of computers of brand B -/
def brand_b_count : Nat := 12

/-- Represents the minimum number of connections required per computer -/
def min_connections : Nat := 2

/-- Theorem stating the maximum number of distinct cable connections -/
theorem max_cable_connections :
  brand_a_count * brand_b_count = 336 ∧
  brand_a_count * brand_b_count ≥ brand_a_count * min_connections ∧
  brand_a_count * brand_b_count ≥ brand_b_count * min_connections :=
sorry

end max_cable_connections_l3595_359526


namespace least_subtraction_for_divisibility_l3595_359598

theorem least_subtraction_for_divisibility :
  ∃ (x : ℕ), x = 7 ∧
  12 ∣ (652543 - x) ∧
  ∀ (y : ℕ), y < x → ¬(12 ∣ (652543 - y)) :=
by sorry

end least_subtraction_for_divisibility_l3595_359598


namespace parabola_intersection_slope_l3595_359583

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Theorem: For a parabola y² = 2px with p > 0, if a line through M(-p/2, 0) with slope k
    intersects the parabola at A(x₀, y₀) such that |AM| = 5/4 * |AF|, then k = ±3/4 -/
theorem parabola_intersection_slope (C : Parabola) (A : ParabolaPoint C) (k : ℝ) :
  let M : ℝ × ℝ := (-C.p/2, 0)
  let F : ℝ × ℝ := (C.p/2, 0)
  let AM := Real.sqrt ((A.x + C.p/2)^2 + A.y^2)
  let AF := A.x + C.p/2
  (A.y - 0) / (A.x - (-C.p/2)) = k →
  AM = 5/4 * AF →
  k = 3/4 ∨ k = -3/4 := by
sorry

end parabola_intersection_slope_l3595_359583


namespace intersection_sum_l3595_359587

theorem intersection_sum (n c : ℝ) : 
  (∀ x y : ℝ, y = n * x + 3 → y = 4 * x + c → x = 4 ∧ y = 7) → 
  c + n = -8 := by
sorry

end intersection_sum_l3595_359587


namespace differential_of_y_l3595_359590

open Real

noncomputable def y (x : ℝ) : ℝ := cos x * log (tan x) - log (tan (x / 2))

theorem differential_of_y (x : ℝ) (h : x ≠ 0) (h' : x ≠ π/2) :
  deriv y x = -sin x * log (tan x) :=
by sorry

end differential_of_y_l3595_359590


namespace triangle_side_values_l3595_359575

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_side_values :
  ∀ x : ℕ+, 
    (triangle_exists 5 (x.val ^ 2) 12) ↔ (x = 3 ∨ x = 4) :=
by sorry

end triangle_side_values_l3595_359575


namespace max_distance_product_l3595_359536

/-- Fixed point A -/
def A : ℝ × ℝ := (0, 0)

/-- Fixed point B -/
def B : ℝ × ℝ := (1, 3)

/-- Line through A -/
def line_A (m : ℝ) (x y : ℝ) : Prop := x + m * y = 0

/-- Line through B -/
def line_B (m : ℝ) (x y : ℝ) : Prop := m * x - y - m + 3 = 0

/-- Intersection point P -/
def P (m : ℝ) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Product of distances PA and PB -/
def distance_product (m : ℝ) : ℝ := distance (P m) A * distance (P m) B

/-- Theorem: Maximum value of |PA| * |PB| is 5 -/
theorem max_distance_product : 
  ∃ (m : ℝ), ∀ (n : ℝ), distance_product n ≤ distance_product m ∧ distance_product m = 5 := by
  sorry

end max_distance_product_l3595_359536


namespace bridget_apples_bridget_apples_proof_l3595_359585

theorem bridget_apples : ℕ → Prop :=
  fun total_apples =>
    ∃ (ann_apples cassie_apples : ℕ),
      -- Bridget gave 4 apples to Tom
      -- She split the remaining apples equally between Ann and Cassie
      ann_apples = cassie_apples ∧
      -- After distribution, she was left with 5 apples
      total_apples = 4 + ann_apples + cassie_apples + 5 ∧
      -- The total number of apples is 13
      total_apples = 13

theorem bridget_apples_proof : bridget_apples 13 := by
  sorry

end bridget_apples_bridget_apples_proof_l3595_359585


namespace product_of_repeating_decimal_and_eight_l3595_359550

/-- The decimal representation of 0.456̄ as a rational number -/
def repeating_decimal : ℚ := 456 / 999

theorem product_of_repeating_decimal_and_eight :
  repeating_decimal * 8 = 1216 / 333 := by sorry

end product_of_repeating_decimal_and_eight_l3595_359550


namespace range_of_a_l3595_359542

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 21 = 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | 5*x - a ≥ 3*x + 2}

-- Theorem statement
theorem range_of_a (a : ℝ) : A ∪ B a = B a → a ≤ -8 :=
by sorry

end range_of_a_l3595_359542


namespace less_number_proof_l3595_359579

theorem less_number_proof (x y : ℝ) (h1 : y = 2 * x) (h2 : x + y = 96) : x = 32 := by
  sorry

end less_number_proof_l3595_359579


namespace first_year_after_2010_with_sum_10_l3595_359589

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_first_year_after_2010_with_sum_10 (year : ℕ) : Prop :=
  year > 2010 ∧
  sum_of_digits year = 10 ∧
  ∀ y, 2010 < y ∧ y < year → sum_of_digits y ≠ 10

theorem first_year_after_2010_with_sum_10 :
  is_first_year_after_2010_with_sum_10 2017 :=
sorry

end first_year_after_2010_with_sum_10_l3595_359589


namespace lottery_profit_l3595_359520

-- Define the card colors
inductive Color
| Black
| Red

-- Define the card values
inductive Value
| One
| Two
| Three
| Four

-- Define a card as a pair of color and value
structure Card where
  color : Color
  value : Value

-- Define the set of cards
def cards : Finset Card := sorry

-- Define the categories
inductive Category
| A  -- Flush
| B  -- Same color
| C  -- Straight
| D  -- Pair
| E  -- Others

-- Function to determine the category of a pair of cards
def categorize : Card → Card → Category := sorry

-- Function to calculate the probability of a category
def probability (c : Category) : Rat := sorry

-- Define the prize values
def prizeValue : Category → Nat
| Category.D => 9  -- First prize
| Category.B => 3  -- Second prize
| _ => 1           -- Third prize

-- Number of participants
def participants : Nat := 300

-- Theorem to prove
theorem lottery_profit :
  (∀ c : Category, c ≠ Category.D → probability Category.D ≤ probability c) ∧
  (∀ c : Category, c ≠ Category.B → probability c ≤ probability Category.B) ∧
  (participants * 3 - (participants * probability Category.D * prizeValue Category.D +
                       participants * probability Category.B * prizeValue Category.B +
                       participants * (1 - probability Category.D - probability Category.B) * 1) = 120) := by
  sorry

end lottery_profit_l3595_359520


namespace five_sqrt_two_gt_three_sqrt_three_l3595_359551

theorem five_sqrt_two_gt_three_sqrt_three : 5 * Real.sqrt 2 > 3 * Real.sqrt 3 := by
  sorry

end five_sqrt_two_gt_three_sqrt_three_l3595_359551


namespace gavin_dreams_total_l3595_359572

/-- The number of dreams Gavin has per day this year -/
def dreams_per_day : ℕ := 4

/-- The number of days in a year -/
def days_per_year : ℕ := 365

/-- The number of dreams Gavin had this year -/
def dreams_this_year : ℕ := dreams_per_day * days_per_year

/-- The number of dreams Gavin had last year -/
def dreams_last_year : ℕ := 2 * dreams_this_year

/-- The total number of dreams Gavin had in two years -/
def total_dreams : ℕ := dreams_this_year + dreams_last_year

theorem gavin_dreams_total : total_dreams = 4380 := by
  sorry

end gavin_dreams_total_l3595_359572


namespace three_number_ratio_problem_l3595_359561

theorem three_number_ratio_problem (a b c : ℝ) 
  (h_sum : a + b + c = 120)
  (h_ratio1 : a / b = 3 / 4)
  (h_ratio2 : b / c = 3 / 5)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  b = 1440 / 41 := by
sorry

end three_number_ratio_problem_l3595_359561


namespace shorts_savings_l3595_359558

/-- Calculates the savings when buying shorts with a discount compared to buying individually -/
def savings (price : ℝ) (quantity : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_cost := price * quantity
  let discounted_cost := total_cost * (1 - discount_rate)
  total_cost - discounted_cost

/-- Proves that the savings when buying 3 pairs of shorts at $10 each with a 10% discount is $3 -/
theorem shorts_savings : savings 10 3 0.1 = 3 := by
  sorry

end shorts_savings_l3595_359558


namespace shawnas_workout_goal_l3595_359599

/-- Shawna's workout goal in situps -/
def workout_goal : ℕ := sorry

/-- Number of situps Shawna did on Monday -/
def monday_situps : ℕ := 12

/-- Number of situps Shawna did on Tuesday -/
def tuesday_situps : ℕ := 19

/-- Number of situps Shawna needs to do on Wednesday to meet her goal -/
def wednesday_situps : ℕ := 59

/-- Theorem stating that Shawna's workout goal is 90 situps -/
theorem shawnas_workout_goal :
  workout_goal = monday_situps + tuesday_situps + wednesday_situps ∧
  workout_goal = 90 := by sorry

end shawnas_workout_goal_l3595_359599


namespace parallel_vectors_magnitude_l3595_359505

/-- Given two parallel vectors a and b in R², prove that the magnitude of b is 2√5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, 2) → 
  b.1 = -2 → 
  (a.1 * b.2 = a.2 * b.1) → 
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end parallel_vectors_magnitude_l3595_359505


namespace cricket_team_size_l3595_359567

theorem cricket_team_size :
  ∀ (n : ℕ),
  let captain_age : ℕ := 24
  let wicket_keeper_age : ℕ := captain_age + 3
  let team_average_age : ℕ := 21
  let remaining_players_average_age : ℕ := team_average_age - 1
  (n : ℝ) * team_average_age = 
    (n - 2 : ℝ) * remaining_players_average_age + captain_age + wicket_keeper_age →
  n = 11 :=
by sorry

end cricket_team_size_l3595_359567


namespace arithmetic_sequences_equal_sum_l3595_359514

/-- Sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
def arithmetic_sum (a d n : ℤ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequences_equal_sum :
  ∃! (n : ℕ), n > 0 ∧ arithmetic_sum 5 4 n = arithmetic_sum 12 3 n :=
by sorry

end arithmetic_sequences_equal_sum_l3595_359514


namespace security_system_probability_l3595_359568

theorem security_system_probability (p : ℝ) : 
  (1/8 : ℝ) * (1 - p) + (1 - 1/8) * p = 9/40 → p = 2/15 := by
  sorry

end security_system_probability_l3595_359568


namespace middle_number_proof_l3595_359594

theorem middle_number_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : a + b = 12) (h4 : a + c = 17) (h5 : b + c = 19) : b = 7 := by
  sorry

end middle_number_proof_l3595_359594


namespace expression_value_l3595_359591

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  2 * x^2 + 3 * y^2 - 4 * z^2 + 5 * x * y = -35 := by
  sorry

end expression_value_l3595_359591


namespace lowest_possible_score_l3595_359576

def total_tests : ℕ := 6
def max_score : ℕ := 200
def target_average : ℕ := 170

def first_four_scores : List ℕ := [150, 180, 175, 160]

theorem lowest_possible_score :
  ∃ (score1 score2 : ℕ),
    score1 ≤ max_score ∧ 
    score2 ≤ max_score ∧
    (List.sum first_four_scores + score1 + score2) / total_tests = target_average ∧
    (∀ (s1 s2 : ℕ), 
      s1 ≤ max_score → 
      s2 ≤ max_score → 
      (List.sum first_four_scores + s1 + s2) / total_tests = target_average → 
      min s1 s2 ≥ min score1 score2) ∧
    min score1 score2 = 155 :=
by sorry

end lowest_possible_score_l3595_359576


namespace square_sum_of_special_integers_l3595_359541

theorem square_sum_of_special_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 47)
  (h2 : x^2 * y + x * y^2 = 506) : 
  x^2 + y^2 = 101 := by sorry

end square_sum_of_special_integers_l3595_359541


namespace range_of_a_for_nonempty_solution_set_l3595_359506

theorem range_of_a_for_nonempty_solution_set :
  (∃ (a : ℝ), ∃ (x : ℝ), |x + 2| + |x| ≤ a) →
  (∀ (a : ℝ), (∃ (x : ℝ), |x + 2| + |x| ≤ a) ↔ a ∈ Set.Ici 2) :=
by sorry

end range_of_a_for_nonempty_solution_set_l3595_359506


namespace circumscribed_sphere_surface_area_l3595_359510

/-- The surface area of a sphere circumscribing a right circular cone -/
theorem circumscribed_sphere_surface_area (h : ℝ) (s : ℝ) (π : ℝ) : 
  h = 3 → s = 2 → π = Real.pi → 
  (4 * π * ((s^2 * 3 / 9) + (h^2 / 4))) = (43 * π / 3) := by
  sorry

end circumscribed_sphere_surface_area_l3595_359510


namespace chichikov_game_l3595_359574

theorem chichikov_game (total_nuts : ℕ) (box1 box2 : ℕ) : total_nuts = 222 → box1 + box2 = total_nuts →
  ∃ N : ℕ, 1 ≤ N ∧ N ≤ 222 ∧
  (∀ move : ℕ, move < 37 →
    ¬(∃ new_box1 new_box2 new_box3 : ℕ,
      new_box1 + new_box2 + new_box3 = total_nuts ∧
      (new_box1 = N ∨ new_box2 = N ∨ new_box3 = N ∨ new_box1 + new_box2 = N ∨ new_box1 + new_box3 = N ∨ new_box2 + new_box3 = N) ∧
      new_box1 + new_box2 + move = box1 + box2)) ∧
  (∀ N : ℕ, 1 ≤ N ∧ N ≤ 222 →
    ∃ new_box1 new_box2 new_box3 : ℕ,
      new_box1 + new_box2 + new_box3 = total_nuts ∧
      (new_box1 = N ∨ new_box2 = N ∨ new_box3 = N ∨ new_box1 + new_box2 = N ∨ new_box1 + new_box3 = N ∨ new_box2 + new_box3 = N) ∧
      new_box1 + new_box2 + 37 ≥ box1 + box2) :=
by
  sorry

end chichikov_game_l3595_359574


namespace quadratic_inequality_range_l3595_359507

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → 
  -2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2 := by
  sorry

end quadratic_inequality_range_l3595_359507


namespace comic_book_collections_equal_l3595_359532

/-- Kymbrea's initial comic book collection size -/
def kymbrea_initial : ℕ := 40

/-- Kymbrea's monthly comic book addition rate -/
def kymbrea_rate : ℕ := 3

/-- LaShawn's initial comic book collection size -/
def lashawn_initial : ℕ := 20

/-- LaShawn's monthly comic book addition rate -/
def lashawn_rate : ℕ := 5

/-- The number of months after which LaShawn's collection will be three times Kymbrea's -/
def months : ℕ := 25

theorem comic_book_collections_equal : 
  lashawn_initial + lashawn_rate * months = 3 * (kymbrea_initial + kymbrea_rate * months) := by
  sorry

end comic_book_collections_equal_l3595_359532


namespace inequality_proof_l3595_359554

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := by
  sorry

end inequality_proof_l3595_359554


namespace light_glow_interval_l3595_359524

def seconds_past_midnight (hours minutes seconds : ℕ) : ℕ :=
  hours * 3600 + minutes * 60 + seconds

def start_time : ℕ := seconds_past_midnight 1 57 58
def end_time : ℕ := seconds_past_midnight 3 20 47
def num_glows : ℝ := 354.92857142857144

theorem light_glow_interval :
  let total_time : ℕ := end_time - start_time
  let interval : ℝ := (total_time : ℝ) / num_glows
  ⌊interval⌋ = 14 := by sorry

end light_glow_interval_l3595_359524


namespace valid_K_values_l3595_359530

/-- The sum of the first n positive integers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate for K being a valid solution -/
def is_valid_K (K : ℕ) : Prop :=
  ∃ (N : ℕ), N < 50 ∧ triangular_sum K = N^2

theorem valid_K_values :
  {K : ℕ | is_valid_K K} = {1, 8, 49} := by sorry

end valid_K_values_l3595_359530


namespace minimum_value_implies_a_range_l3595_359535

/-- The function f(x) = x^3 - 3x has a minimum value on the interval (a, 6-a^2) -/
def has_minimum_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo a (6 - a^2), ∀ y ∈ Set.Ioo a (6 - a^2), f x ≤ f y

/-- The main theorem -/
theorem minimum_value_implies_a_range (a : ℝ) :
  has_minimum_on_interval (fun x => x^3 - 3*x) a → a ∈ Set.Icc (-2) 1 :=
sorry

end minimum_value_implies_a_range_l3595_359535


namespace quadratic_equation_roots_range_l3595_359560

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ↔ 
  m < -2 ∨ m > 2 :=
sorry

end quadratic_equation_roots_range_l3595_359560


namespace sphere_volume_ratio_l3595_359517

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * π * r₁^2) / (4 * π * r₂^2) = 4 / 9 →
  ((4 / 3) * π * r₁^3) / ((4 / 3) * π * r₂^3) = 8 / 27 := by
  sorry

end sphere_volume_ratio_l3595_359517


namespace vacant_seats_l3595_359511

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) 
  (h1 : total_seats = 600) 
  (h2 : filled_percentage = 75 / 100) : 
  ℕ := by
  sorry

end vacant_seats_l3595_359511


namespace exam_score_proof_l3595_359597

/-- Represents the average score of students who took the exam on the assigned day -/
def average_score_assigned_day : ℝ := 55

theorem exam_score_proof (total_students : ℕ) (assigned_day_percentage : ℝ) 
  (makeup_percentage : ℝ) (makeup_average : ℝ) (class_average : ℝ) : 
  total_students = 100 →
  assigned_day_percentage = 70 →
  makeup_percentage = 30 →
  makeup_average = 95 →
  class_average = 67 →
  (assigned_day_percentage * average_score_assigned_day + 
   makeup_percentage * makeup_average) / 100 = class_average :=
by
  sorry

#check exam_score_proof

end exam_score_proof_l3595_359597


namespace y_divisibility_l3595_359527

def y : ℕ := 80 + 120 + 160 + 200 + 360 + 440 + 4040

theorem y_divisibility : 
  (∃ k : ℕ, y = 5 * k) ∧ 
  (∃ k : ℕ, y = 10 * k) ∧ 
  (∃ k : ℕ, y = 20 * k) ∧ 
  (∃ k : ℕ, y = 40 * k) := by
  sorry

end y_divisibility_l3595_359527


namespace four_roots_implies_a_in_open_interval_l3595_359523

def f (x : ℝ) : ℝ := |x^2 + x - 2|

theorem four_roots_implies_a_in_open_interval (a : ℝ) :
  (∃ (w x y z : ℝ), w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    f w - a * |w - 2| = 0 ∧
    f x - a * |x - 2| = 0 ∧
    f y - a * |y - 2| = 0 ∧
    f z - a * |z - 2| = 0 ∧
    (∀ t : ℝ, f t - a * |t - 2| = 0 → t = w ∨ t = x ∨ t = y ∨ t = z)) →
  0 < a ∧ a < 3 :=
sorry

end four_roots_implies_a_in_open_interval_l3595_359523


namespace rowing_speeds_calculation_l3595_359528

/-- Represents the rowing speeds and wind effects for a man rowing a boat -/
structure RowingScenario where
  withStreamSpeed : ℝ
  againstStreamSpeed : ℝ
  windSpeedDownstream : ℝ
  windReductionAgainstStream : ℝ
  windIncreaseWithStream : ℝ

/-- Calculates the effective rowing speeds given a RowingScenario -/
def effectiveRowingSpeeds (scenario : RowingScenario) : ℝ × ℝ :=
  let effectiveAgainstStream := scenario.againstStreamSpeed * (1 - scenario.windReductionAgainstStream)
  let effectiveWithStream := scenario.withStreamSpeed * (1 + scenario.windIncreaseWithStream)
  (effectiveWithStream, effectiveAgainstStream)

/-- Theorem stating the effective rowing speeds for the given scenario -/
theorem rowing_speeds_calculation (scenario : RowingScenario) 
    (h1 : scenario.withStreamSpeed = 8)
    (h2 : scenario.againstStreamSpeed = 4)
    (h3 : scenario.windSpeedDownstream = 2)
    (h4 : scenario.windReductionAgainstStream = 0.2)
    (h5 : scenario.windIncreaseWithStream = 0.1) :
    effectiveRowingSpeeds scenario = (8.8, 3.2) := by
  sorry

#eval effectiveRowingSpeeds { 
  withStreamSpeed := 8, 
  againstStreamSpeed := 4, 
  windSpeedDownstream := 2, 
  windReductionAgainstStream := 0.2, 
  windIncreaseWithStream := 0.1 
}

end rowing_speeds_calculation_l3595_359528


namespace rectangular_garden_width_l3595_359540

theorem rectangular_garden_width (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * width → 
  area = length * width → 
  area = 768 → 
  width = 16 := by
sorry

end rectangular_garden_width_l3595_359540


namespace chocolate_squares_difference_l3595_359543

theorem chocolate_squares_difference (mike_squares jenny_squares : ℕ) 
  (h1 : mike_squares = 20) 
  (h2 : jenny_squares = 65) : 
  jenny_squares - 3 * mike_squares = 5 := by
  sorry

end chocolate_squares_difference_l3595_359543


namespace toby_change_is_seven_l3595_359566

/-- Represents the cost of a meal for two people -/
structure MealCost where
  cheeseburger_price : ℚ
  milkshake_price : ℚ
  coke_price : ℚ
  fries_price : ℚ
  cookie_price : ℚ
  cookie_quantity : ℕ
  tax : ℚ

/-- Calculates the change Toby brings home after splitting the bill -/
def toby_change (meal : MealCost) (toby_initial_amount : ℚ) : ℚ :=
  let total_cost := 2 * meal.cheeseburger_price + meal.milkshake_price + meal.coke_price +
                    meal.fries_price + meal.cookie_price * meal.cookie_quantity + meal.tax
  let toby_share := total_cost / 2
  toby_initial_amount - toby_share

/-- Theorem stating that Toby's change is $7 given the specific meal costs -/
theorem toby_change_is_seven :
  let meal := MealCost.mk 3.65 2 1 4 0.5 3 0.2
  toby_change meal 15 = 7 := by sorry


end toby_change_is_seven_l3595_359566


namespace simplify_expression_l3595_359509

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : x + y + z = 3) :
  (1 / (y^2 + z^2 - x^2)) + (1 / (x^2 + z^2 - y^2)) + (1 / (x^2 + y^2 - z^2)) =
  3 / (-9 + 6*y + 6*z - 2*y*z) := by
  sorry

end simplify_expression_l3595_359509


namespace sams_football_games_l3595_359556

/-- Given that Sam went to 14 football games this year and 43 games in total,
    prove that he went to 29 games last year. -/
theorem sams_football_games (games_this_year games_total : ℕ) 
    (h1 : games_this_year = 14)
    (h2 : games_total = 43) :
    games_total - games_this_year = 29 := by
  sorry

end sams_football_games_l3595_359556


namespace complex_condition_l3595_359552

theorem complex_condition (a b : ℝ) (hb : b ≠ 0) :
  let z : ℂ := Complex.mk a b
  (z^2 - 4*b*z).im = 0 → a = 2*b := by
  sorry

end complex_condition_l3595_359552


namespace larry_lost_stickers_l3595_359553

/-- Given that Larry starts with 93 stickers and ends up with 87 stickers,
    prove that he lost 6 stickers. -/
theorem larry_lost_stickers (initial : ℕ) (final : ℕ) (h1 : initial = 93) (h2 : final = 87) :
  initial - final = 6 := by
  sorry

end larry_lost_stickers_l3595_359553


namespace plywood_cut_perimeter_difference_l3595_359563

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem plywood_cut_perimeter_difference :
  let plywood := Rectangle.mk 12 6
  let area := plywood.length * plywood.width
  ∀ (piece : Rectangle),
    (6 * piece.length * piece.width = area) →
    (∃ (max_piece min_piece : Rectangle),
      (6 * max_piece.length * max_piece.width = area) ∧
      (6 * min_piece.length * min_piece.width = area) ∧
      (∀ (r : Rectangle), (6 * r.length * r.width = area) →
        perimeter r ≤ perimeter max_piece ∧
        perimeter r ≥ perimeter min_piece)) →
    (perimeter max_piece - perimeter min_piece = 14) := by
  sorry

end plywood_cut_perimeter_difference_l3595_359563


namespace xyz_sum_l3595_359502

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 147)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 163) :
  x*y + y*z + x*z = 56 := by
sorry

end xyz_sum_l3595_359502


namespace perfect_square_condition_l3595_359596

theorem perfect_square_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + (a - 1)*x + 16 = (x + b)^2) → (a = 9 ∨ a = -7) := by
sorry

end perfect_square_condition_l3595_359596


namespace union_complement_equals_set_l3595_359570

def I : Set Int := {x | -3 < x ∧ x < 3}
def A : Set Int := {1, 2}
def B : Set Int := {-2, -1, 2}

theorem union_complement_equals_set : A ∪ (I \ B) = {0, 1, 2} := by sorry

end union_complement_equals_set_l3595_359570


namespace single_elimination_tournament_matches_l3595_359513

/-- Represents a single-elimination tournament. -/
structure Tournament where
  teams : ℕ
  matches_played : ℕ

/-- The number of teams eliminated in a single-elimination tournament. -/
def eliminated_teams (t : Tournament) : ℕ := t.matches_played

/-- A tournament is complete when there is only one team remaining. -/
def is_complete (t : Tournament) : Prop := t.teams - eliminated_teams t = 1

theorem single_elimination_tournament_matches (t : Tournament) :
  t.teams = 128 → is_complete t → t.matches_played = 127 := by
  sorry

end single_elimination_tournament_matches_l3595_359513


namespace min_value_sum_reciprocals_l3595_359562

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_condition : 2*a + 2*b + 2*c = 3) :
  (1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a)) ≥ 2 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ 
    2*a₀ + 2*b₀ + 2*c₀ = 3 ∧
    1 / (2*a₀ + b₀) + 1 / (2*b₀ + c₀) + 1 / (2*c₀ + a₀) = 2 := by
  sorry

end min_value_sum_reciprocals_l3595_359562


namespace block_stacks_height_difference_main_theorem_l3595_359582

/-- Proves that the height difference between the final stack and the second stack is 7 blocks -/
theorem block_stacks_height_difference : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (first_stack : ℕ) (second_stack : ℕ) (final_stack : ℕ) (fallen_blocks : ℕ) (height_diff : ℕ) =>
    first_stack = 7 ∧
    second_stack = first_stack + 5 ∧
    final_stack = second_stack + height_diff ∧
    fallen_blocks = first_stack + (second_stack - 2) + (final_stack - 3) ∧
    fallen_blocks = 33 →
    height_diff = 7

/-- The main theorem stating that the height difference is 7 blocks -/
theorem main_theorem : ∃ (first_stack second_stack final_stack fallen_blocks : ℕ),
  block_stacks_height_difference first_stack second_stack final_stack fallen_blocks 7 :=
sorry

end block_stacks_height_difference_main_theorem_l3595_359582


namespace female_athletes_in_sample_l3595_359581

/-- Calculates the number of female athletes in a stratified sample -/
def femaleAthletesSample (totalAthletes maleAthletes femaleAthletes sampleSize : ℕ) : ℕ :=
  (femaleAthletes * sampleSize) / totalAthletes

/-- Theorem stating the number of female athletes in the sample -/
theorem female_athletes_in_sample :
  femaleAthletesSample 84 48 36 21 = 9 := by
  sorry

#eval femaleAthletesSample 84 48 36 21

end female_athletes_in_sample_l3595_359581


namespace vehicle_speeds_l3595_359547

/-- A structure representing a vehicle with its speed -/
structure Vehicle where
  speed : ℝ
  speed_pos : speed > 0

/-- The problem setup -/
def VehicleProblem (v₁ v₄ : ℝ) : Prop :=
  v₁ > 0 ∧ v₄ > 0 ∧ v₁ > v₄

/-- The theorem statement -/
theorem vehicle_speeds (v₁ v₄ : ℝ) (h : VehicleProblem v₁ v₄) :
  ∃ (v₂ v₃ : ℝ),
    v₂ = 3 * v₁ * v₄ / (2 * v₄ + v₁) ∧
    v₃ = 3 * v₁ * v₄ / (v₄ + 2 * v₁) ∧
    v₁ > v₂ ∧ v₂ > v₃ ∧ v₃ > v₄ :=
  sorry

end vehicle_speeds_l3595_359547


namespace absolute_value_equality_l3595_359531

theorem absolute_value_equality (y : ℝ) : 
  |y - 3| = |y - 5| → y = 4 := by
sorry

end absolute_value_equality_l3595_359531
