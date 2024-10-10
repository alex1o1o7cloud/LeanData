import Mathlib

namespace bert_sandwiches_l3317_331735

/-- The number of sandwiches remaining after two days of eating -/
def sandwiches_remaining (initial : ℕ) : ℕ :=
  initial - (initial / 2) - (initial / 2 - 2)

/-- Theorem stating that given 12 initial sandwiches, 2 remain after two days of eating -/
theorem bert_sandwiches : sandwiches_remaining 12 = 2 := by
  sorry

end bert_sandwiches_l3317_331735


namespace fourth_power_of_cube_of_third_smallest_prime_l3317_331790

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- State the theorem
theorem fourth_power_of_cube_of_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 4 = 244140625 := by
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l3317_331790


namespace kia_vehicles_count_l3317_331744

theorem kia_vehicles_count (total : ℕ) (dodge : ℕ) (hyundai : ℕ) (kia : ℕ) : 
  total = 400 →
  dodge = total / 2 →
  hyundai = dodge / 2 →
  kia = total - (dodge + hyundai) →
  kia = 100 := by
  sorry

end kia_vehicles_count_l3317_331744


namespace probability_one_from_each_set_l3317_331702

theorem probability_one_from_each_set (n : ℕ) :
  let total := 2 * n
  let prob_first := n / total
  let prob_second := n / (total - 1)
  2 * (prob_first * prob_second) = n / (n + 1) :=
by
  sorry

#check probability_one_from_each_set 6

end probability_one_from_each_set_l3317_331702


namespace greatest_c_not_in_range_l3317_331765

def f (c : ℝ) (x : ℝ) : ℝ := x^2 + c*x + 15

theorem greatest_c_not_in_range : 
  ∀ c : ℤ, (∀ x : ℝ, f c x ≠ -9) ↔ c ≤ 9 :=
by sorry

end greatest_c_not_in_range_l3317_331765


namespace eggs_per_plate_count_l3317_331737

def breakfast_plate (num_customers : ℕ) (total_bacon : ℕ) : ℕ → Prop :=
  λ eggs_per_plate : ℕ =>
    eggs_per_plate > 0 ∧
    2 * eggs_per_plate * num_customers = total_bacon

theorem eggs_per_plate_count (num_customers : ℕ) (total_bacon : ℕ) 
    (h1 : num_customers = 14) (h2 : total_bacon = 56) :
    ∃ eggs_per_plate : ℕ, breakfast_plate num_customers total_bacon eggs_per_plate ∧ 
    eggs_per_plate = 2 := by
  sorry

end eggs_per_plate_count_l3317_331737


namespace instantaneous_speed_at_4_l3317_331708

-- Define the motion equation
def s (t : ℝ) : ℝ := t^2 - 2*t + 5

-- Define the instantaneous speed (derivative of s)
def v (t : ℝ) : ℝ := 2*t - 2

-- Theorem: The instantaneous speed at t = 4 is 6 m/s
theorem instantaneous_speed_at_4 : v 4 = 6 := by
  sorry

end instantaneous_speed_at_4_l3317_331708


namespace product_inequality_l3317_331795

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1) * (b + 1) * (a + c) * (b + c) ≥ 16 * a * b * c := by
  sorry

end product_inequality_l3317_331795


namespace jerry_candy_boxes_l3317_331713

theorem jerry_candy_boxes (initial boxes_sold boxes_left : ℕ) :
  boxes_sold = 5 →
  boxes_left = 5 →
  initial = boxes_sold + boxes_left →
  initial = 10 :=
by sorry

end jerry_candy_boxes_l3317_331713


namespace girls_after_joining_l3317_331703

def initial_girls : ℕ := 732
def new_girls : ℕ := 682

theorem girls_after_joining (initial_girls new_girls : ℕ) : 
  initial_girls + new_girls = 1414 :=
sorry

end girls_after_joining_l3317_331703


namespace race_time_difference_l3317_331764

/-- Proves that in a 1000-meter race where runner A finishes in 90 seconds and is 100 meters ahead of runner B at the finish line, A beats B by 9 seconds. -/
theorem race_time_difference (race_length : ℝ) (a_time : ℝ) (distance_difference : ℝ) : 
  race_length = 1000 →
  a_time = 90 →
  distance_difference = 100 →
  (race_length / a_time) * (distance_difference / race_length) * a_time = 9 :=
by
  sorry

#check race_time_difference

end race_time_difference_l3317_331764


namespace binary_to_decimal_conversion_l3317_331707

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (digits : List Bool) : ℕ :=
  digits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- The binary representation of the number we want to convert. -/
def binaryNumber : List Bool :=
  [true, true, true, false, true, true, false, false, true, false, false, true]

theorem binary_to_decimal_conversion :
  binaryToNat binaryNumber = 3785 := by
  sorry

end binary_to_decimal_conversion_l3317_331707


namespace four_people_name_condition_l3317_331709

/-- Represents a person with a first name, patronymic, and last name -/
structure Person where
  firstName : String
  patronymic : String
  lastName : String

/-- Checks if two people share any attribute -/
def shareAttribute (p1 p2 : Person) : Prop :=
  p1.firstName = p2.firstName ∨ p1.patronymic = p2.patronymic ∨ p1.lastName = p2.lastName

/-- Theorem stating the existence of 4 people satisfying the given conditions -/
theorem four_people_name_condition : ∃ (people : Finset Person),
  (Finset.card people = 4) ∧
  (∀ (attr : Person → String),
    ∀ (p1 p2 p3 : Person),
      p1 ∈ people → p2 ∈ people → p3 ∈ people →
      p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
      ¬(attr p1 = attr p2 ∧ attr p2 = attr p3)) ∧
  (∀ (p1 p2 : Person),
    p1 ∈ people → p2 ∈ people → p1 ≠ p2 →
    shareAttribute p1 p2) :=
by sorry

end four_people_name_condition_l3317_331709


namespace absolute_value_ratio_l3317_331769

theorem absolute_value_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 5*x*y) :
  |((x+y)/(x-y))| = Real.sqrt (7/3) := by
  sorry

end absolute_value_ratio_l3317_331769


namespace angle_bisector_implies_line_AC_l3317_331734

-- Define points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-1, 2)

-- Define the angle bisector equation
def angle_bisector (x y : ℝ) : Prop := y = x + 1

-- Define the equation of line AC
def line_AC (x y : ℝ) : Prop := x - 2*y - 1 = 0

theorem angle_bisector_implies_line_AC :
  ∀ C : ℝ × ℝ,
  angle_bisector C.1 C.2 →
  line_AC C.1 C.2 :=
by sorry

end angle_bisector_implies_line_AC_l3317_331734


namespace divisors_of_8_factorial_l3317_331718

theorem divisors_of_8_factorial (n : ℕ) : n = 8 → (Finset.card (Nat.divisors (Nat.factorial n))) = 96 := by
  sorry

end divisors_of_8_factorial_l3317_331718


namespace slope_angle_vertical_line_l3317_331786

-- Define a vertical line
def vertical_line (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = a}

-- Define the slope angle of a line
def slope_angle (L : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem slope_angle_vertical_line :
  slope_angle (vertical_line 2) = π / 2 := by
  sorry

end slope_angle_vertical_line_l3317_331786


namespace bill_harry_nuts_ratio_l3317_331776

theorem bill_harry_nuts_ratio : 
  ∀ (sue_nuts harry_nuts bill_nuts : ℕ),
    sue_nuts = 48 →
    harry_nuts = 2 * sue_nuts →
    bill_nuts + harry_nuts = 672 →
    bill_nuts = 6 * harry_nuts :=
by
  sorry

end bill_harry_nuts_ratio_l3317_331776


namespace find_c_l3317_331767

theorem find_c (p q : ℝ → ℝ) (c : ℝ) : 
  (∀ x, p x = 4 * x - 3) →
  (∀ x, q x = 5 * x - c) →
  p (q 3) = 53 →
  c = 1 := by
sorry

end find_c_l3317_331767


namespace smallest_cover_l3317_331784

/-- The side length of the rectangle. -/
def rectangle_width : ℕ := 3

/-- The height of the rectangle. -/
def rectangle_height : ℕ := 4

/-- The area of a single rectangle. -/
def rectangle_area : ℕ := rectangle_width * rectangle_height

/-- The side length of the square that can be covered exactly by the rectangles. -/
def square_side : ℕ := 12

/-- The area of the square. -/
def square_area : ℕ := square_side * square_side

/-- The number of rectangles needed to cover the square. -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem smallest_cover :
  (∀ n : ℕ, n < square_side → n * n % rectangle_area ≠ 0) ∧
  square_area % rectangle_area = 0 ∧
  num_rectangles = 12 := by
  sorry

#check smallest_cover

end smallest_cover_l3317_331784


namespace gcd_15378_21333_48906_l3317_331751

theorem gcd_15378_21333_48906 : Nat.gcd 15378 (Nat.gcd 21333 48906) = 3 := by
  sorry

end gcd_15378_21333_48906_l3317_331751


namespace wrapping_paper_division_l3317_331749

theorem wrapping_paper_division (total_used : ℚ) (num_presents : ℕ) 
  (h1 : total_used = 1/2)
  (h2 : num_presents = 5) :
  total_used / num_presents = 1/10 := by
  sorry

end wrapping_paper_division_l3317_331749


namespace brown_eyed_brunettes_l3317_331747

/-- The total number of girls -/
def total_girls : ℕ := 60

/-- The number of green-eyed redheads -/
def green_eyed_redheads : ℕ := 20

/-- The number of brunettes -/
def brunettes : ℕ := 35

/-- The number of brown-eyed girls -/
def brown_eyed : ℕ := 25

/-- Theorem: The number of brown-eyed brunettes is 20 -/
theorem brown_eyed_brunettes : 
  total_girls - (green_eyed_redheads + (brunettes - (brown_eyed - (total_girls - brunettes - green_eyed_redheads)))) = 20 := by
  sorry

end brown_eyed_brunettes_l3317_331747


namespace cos_double_angle_unit_circle_l3317_331775

theorem cos_double_angle_unit_circle (α : Real) :
  (Real.cos α = -Real.sqrt 3 / 2 ∧ Real.sin α = 1 / 2) →
  Real.cos (2 * α) = 1 / 2 := by
sorry

end cos_double_angle_unit_circle_l3317_331775


namespace gumball_cost_l3317_331757

/-- Given that Melanie sells 4 gumballs for a total of 32 cents, prove that each gumball costs 8 cents. -/
theorem gumball_cost (num_gumballs : ℕ) (total_cents : ℕ) (h1 : num_gumballs = 4) (h2 : total_cents = 32) :
  total_cents / num_gumballs = 8 := by
  sorry

end gumball_cost_l3317_331757


namespace john_annual_profit_l3317_331726

/-- Calculates John's annual profit from subletting his apartment -/
def annual_profit (tenant_a_rent tenant_b_rent tenant_c_rent john_rent utilities maintenance : ℕ) : ℕ :=
  let monthly_income := tenant_a_rent + tenant_b_rent + tenant_c_rent
  let monthly_expenses := john_rent + utilities + maintenance
  let monthly_profit := monthly_income - monthly_expenses
  12 * monthly_profit

/-- Theorem stating John's annual profit given his rental income and expenses -/
theorem john_annual_profit :
  annual_profit 350 400 450 900 100 50 = 1800 := by
  sorry

end john_annual_profit_l3317_331726


namespace grayson_collection_l3317_331798

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 48

/-- The number of boxes Abigail collected -/
def abigail_boxes : ℕ := 2

/-- The number of boxes Olivia collected -/
def olivia_boxes : ℕ := 3

/-- The total number of cookies collected -/
def total_cookies : ℕ := 276

/-- The fraction of a box Grayson collected -/
def grayson_fraction : ℚ := 3/4

theorem grayson_collection :
  grayson_fraction * cookies_per_box = 
    total_cookies - (abigail_boxes + olivia_boxes) * cookies_per_box :=
by sorry

end grayson_collection_l3317_331798


namespace sum_of_solutions_equals_six_l3317_331797

theorem sum_of_solutions_equals_six :
  ∃ (x₁ x₂ : ℝ), 
    (3 : ℝ) ^ (x₁^2 - 4*x₁ - 3) = 9 ^ (x₁ - 5) ∧
    (3 : ℝ) ^ (x₂^2 - 4*x₂ - 3) = 9 ^ (x₂ - 5) ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 6 ∧
    ∀ (x : ℝ), (3 : ℝ) ^ (x^2 - 4*x - 3) = 9 ^ (x - 5) → x = x₁ ∨ x = x₂ :=
by sorry

end sum_of_solutions_equals_six_l3317_331797


namespace turtle_reaches_waterhole_in_28_minutes_l3317_331733

/-- Represents the scenario with two lion cubs and a turtle moving towards a watering hole -/
structure WaterholeProblem where
  /-- Distance of the first lion cub from the watering hole in minutes -/
  lion1_distance : ℝ
  /-- Speed multiplier of the second lion cub compared to the first -/
  lion2_speed_multiplier : ℝ
  /-- Distance of the turtle from the watering hole in minutes -/
  turtle_distance : ℝ

/-- Calculates the time it takes for the turtle to reach the watering hole after meeting the lion cubs -/
def timeToWaterhole (problem : WaterholeProblem) : ℝ :=
  sorry

/-- Theorem stating that given the specific problem conditions, the turtle reaches the watering hole 28 minutes after meeting the lion cubs -/
theorem turtle_reaches_waterhole_in_28_minutes :
  let problem : WaterholeProblem :=
    { lion1_distance := 5
      lion2_speed_multiplier := 1.5
      turtle_distance := 30 }
  timeToWaterhole problem = 28 :=
sorry

end turtle_reaches_waterhole_in_28_minutes_l3317_331733


namespace smallest_circle_theorem_l3317_331741

/-- Given two circles in the xy-plane, this function returns the equation of the circle 
    with the smallest area that passes through their intersection points. -/
def smallest_circle_through_intersections (c1 c2 : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  sorry

/-- The first given circle -/
def circle1 (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 + 4*p.1 + p.2 = -1

/-- The second given circle -/
def circle2 (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 + 2*p.1 + 2*p.2 + 1 = 0

/-- The resulting circle with the smallest area -/
def result_circle (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 + (6/5)*p.1 + (12/5)*p.2 + 1 = 0

theorem smallest_circle_theorem :
  smallest_circle_through_intersections circle1 circle2 = result_circle :=
sorry

end smallest_circle_theorem_l3317_331741


namespace triangle_altitude_specific_triangle_altitude_l3317_331710

/-- The altitude of a triangle given its area and base -/
theorem triangle_altitude (area : ℝ) (base : ℝ) (h_area : area > 0) (h_base : base > 0) :
  area = (1/2) * base * (2 * area / base) :=
by sorry

/-- The altitude of a specific triangle with area 800 and base 40 -/
theorem specific_triangle_altitude :
  let area : ℝ := 800
  let base : ℝ := 40
  let altitude : ℝ := 2 * area / base
  altitude = 40 :=
by sorry

end triangle_altitude_specific_triangle_altitude_l3317_331710


namespace archaeopteryx_humerus_estimate_l3317_331706

/-- Represents the linear regression equation for Archaeopteryx fossil specimens -/
def archaeopteryx_regression (x : ℝ) : ℝ := 1.197 * x - 3.660

/-- Theorem stating the estimated humerus length for a given femur length -/
theorem archaeopteryx_humerus_estimate :
  archaeopteryx_regression 50 = 56.19 := by
  sorry

end archaeopteryx_humerus_estimate_l3317_331706


namespace unshaded_perimeter_l3317_331794

/-- Given an L-shaped region formed by two adjoining rectangles with the following properties:
  - The total area of the L-shape is 240 square inches
  - The area of the shaded region is 65 square inches
  - The total length of the combined rectangles is 20 inches
  - The total width at the widest point is 12 inches
  - The width of the inner shaded rectangle is 5 inches
  - All rectangles contain right angles

  This theorem proves that the perimeter of the unshaded region is 64 inches. -/
theorem unshaded_perimeter (total_area : ℝ) (shaded_area : ℝ) (total_length : ℝ) (total_width : ℝ) (inner_width : ℝ)
  (h_total_area : total_area = 240)
  (h_shaded_area : shaded_area = 65)
  (h_total_length : total_length = 20)
  (h_total_width : total_width = 12)
  (h_inner_width : inner_width = 5) :
  2 * ((total_width - inner_width) + (total_area - shaded_area) / (total_width - inner_width)) = 64 :=
by sorry

end unshaded_perimeter_l3317_331794


namespace chord_equation_of_ellipse_l3317_331719

/-- Given an ellipse and a point on a bisecting chord, prove the equation of the line containing the chord. -/
theorem chord_equation_of_ellipse (x y : ℝ) :
  let ellipse := fun (x y : ℝ) => x^2/16 + y^2/4 = 1
  let P := (-2, 1)
  let chord_bisector := fun (x y : ℝ) => ∃ (x1 y1 x2 y2 : ℝ),
    ellipse x1 y1 ∧ ellipse x2 y2 ∧ 
    x = (x1 + x2)/2 ∧ y = (y1 + y2)/2
  chord_bisector P.1 P.2 →
  x - 2*y + 4 = 0 := by sorry

end chord_equation_of_ellipse_l3317_331719


namespace triangle_side_length_l3317_331754

/-- Given two triangles ABC and DEF with specified side lengths and angles,
    prove that the length of EF is 3.75 units when the area of DEF is half that of ABC. -/
theorem triangle_side_length (AB DE AC DF : ℝ) (angleBAC angleEDF : ℝ) :
  AB = 5 →
  DE = 2 →
  AC = 6 →
  DF = 3 →
  angleBAC = 30 * π / 180 →
  angleEDF = 45 * π / 180 →
  (1 / 2 * DE * DF * Real.sin angleEDF) = (1 / 4 * AB * AC * Real.sin angleBAC) →
  ∃ (EF : ℝ), EF = 3.75 :=
by sorry

end triangle_side_length_l3317_331754


namespace clock_time_l3317_331712

/-- Represents a clock with a specific ticking pattern -/
structure Clock where
  ticks_at_hour : ℕ
  time_between_first_and_last : ℝ
  time_at_12 : ℝ

/-- The number of ticks at 12 o'clock -/
def ticks_at_12 : ℕ := 12

theorem clock_time (c : Clock) (h1 : c.ticks_at_hour = 6) 
  (h2 : c.time_between_first_and_last = 25) 
  (h3 : c.time_at_12 = 55) : 
  c.ticks_at_hour = 6 := by sorry

end clock_time_l3317_331712


namespace students_playing_both_football_and_cricket_l3317_331723

/-- The number of students playing both football and cricket -/
def students_playing_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

/-- Proof that 140 students play both football and cricket -/
theorem students_playing_both_football_and_cricket :
  students_playing_both 410 325 175 50 = 140 := by
  sorry

end students_playing_both_football_and_cricket_l3317_331723


namespace linear_function_inequality_l3317_331714

theorem linear_function_inequality (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x + b) →
  (∀ x, f (f x) ≥ x - 3) ↔
  ((a = -1 ∧ b ∈ Set.univ) ∨ (a = 1 ∧ b ≥ -3/2)) :=
by sorry

end linear_function_inequality_l3317_331714


namespace largest_multiple_of_12_using_all_digits_l3317_331755

/-- A function that checks if a number uses each digit from 0 to 9 exactly once -/
def usesAllDigitsOnce (n : ℕ) : Prop := sorry

/-- A function that returns the largest number that can be formed using each digit from 0 to 9 exactly once and is a multiple of 12 -/
def largestMultipleOf12UsingAllDigits : ℕ := sorry

theorem largest_multiple_of_12_using_all_digits :
  largestMultipleOf12UsingAllDigits = 987654320 ∧
  usesAllDigitsOnce largestMultipleOf12UsingAllDigits ∧
  largestMultipleOf12UsingAllDigits % 12 = 0 ∧
  ∀ m : ℕ, usesAllDigitsOnce m ∧ m % 12 = 0 → m ≤ largestMultipleOf12UsingAllDigits :=
by sorry

end largest_multiple_of_12_using_all_digits_l3317_331755


namespace quadratic_roots_and_fraction_l3317_331756

theorem quadratic_roots_and_fraction (a b p q : ℝ) : 
  (∃ (x : ℂ), x^2 + p*x + q = 0 ∧ (x = 2 + a*I ∨ x = b + I)) →
  (a = -1 ∧ b = 2 ∧ p = -4 ∧ q = 5) ∧
  (a + b*I) / (p + q*I) = 3/41 + 6/41*I :=
by sorry

end quadratic_roots_and_fraction_l3317_331756


namespace mod_37_5_l3317_331762

theorem mod_37_5 : 37 % 5 = 2 := by
  sorry

end mod_37_5_l3317_331762


namespace tims_total_expense_l3317_331701

/-- Calculates Tim's total out-of-pocket expense for medical visits -/
theorem tims_total_expense (tims_visit_cost : ℝ) (tims_insurance_coverage : ℝ) 
  (cats_visit_cost : ℝ) (cats_insurance_coverage : ℝ) 
  (h1 : tims_visit_cost = 300)
  (h2 : tims_insurance_coverage = 0.75 * tims_visit_cost)
  (h3 : cats_visit_cost = 120)
  (h4 : cats_insurance_coverage = 60) : 
  tims_visit_cost - tims_insurance_coverage + cats_visit_cost - cats_insurance_coverage = 135 := by
  sorry


end tims_total_expense_l3317_331701


namespace train_passes_jogger_train_passes_jogger_time_l3317_331704

/-- The time taken for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The time taken for the train to pass the jogger is 40 seconds -/
theorem train_passes_jogger_time : train_passes_jogger 9 45 200 200 = 40 := by
  sorry

end train_passes_jogger_train_passes_jogger_time_l3317_331704


namespace max_value_circle_center_l3317_331792

/-- Circle C with center (a,b) and radius 1 -/
def Circle (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = 1}

/-- Region Ω -/
def Ω : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 - 7 ≤ 0 ∧ p.1 - p.2 + 3 ≥ 0 ∧ p.2 ≥ 0}

/-- The maximum value of a^2 + b^2 given the conditions -/
theorem max_value_circle_center (a b : ℝ) :
  (a, b) ∈ Ω →
  b = 1 →
  (∃ (x : ℝ), (x, 0) ∈ Circle a b) →
  a^2 + b^2 ≤ 37 :=
sorry

end max_value_circle_center_l3317_331792


namespace noah_class_size_l3317_331778

theorem noah_class_size (n : ℕ) (noah_rank_best : ℕ) (noah_rank_worst : ℕ) 
  (h1 : noah_rank_best = 40)
  (h2 : noah_rank_worst = 40)
  (h3 : n = noah_rank_best + noah_rank_worst - 1) :
  n = 79 := by
  sorry

end noah_class_size_l3317_331778


namespace angle_C_is_120_max_area_condition_l3317_331728

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_condition : (2 * a + b) * Real.cos C + c * Real.cos B = 0
axiom positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Part 1: Prove that angle C is 120°
theorem angle_C_is_120 : C = 2 * π / 3 := by sorry

-- Part 2: Prove that when c = 4, area is maximized when a = b = (4√3)/3
theorem max_area_condition (h : c = 4) :
  (∀ a' b', a' > 0 → b' > 0 → a' * b' * Real.sin C ≤ a * b * Real.sin C) →
  a = 4 * Real.sqrt 3 / 3 ∧ b = 4 * Real.sqrt 3 / 3 := by sorry

end angle_C_is_120_max_area_condition_l3317_331728


namespace diagonal_triangle_area_l3317_331716

/-- Represents a rectangular prism with given face areas -/
structure RectangularPrism where
  face_area_1 : ℝ
  face_area_2 : ℝ
  face_area_3 : ℝ

/-- Calculates the area of the triangle formed by the diagonals of the prism's faces -/
noncomputable def triangle_area (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem stating that for a rectangular prism with face areas 24, 30, and 32,
    the triangle formed by the diagonals of these faces has an area of 25 -/
theorem diagonal_triangle_area :
  let prism : RectangularPrism := ⟨24, 30, 32⟩
  triangle_area prism = 25 := by
  sorry

end diagonal_triangle_area_l3317_331716


namespace power_of_eight_division_l3317_331725

theorem power_of_eight_division (n : ℕ) : 8^(n+1) / 8 = 8^n := by
  sorry

end power_of_eight_division_l3317_331725


namespace divisibility_by_three_l3317_331717

theorem divisibility_by_three (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ)
  (h1 : A ^ 2 + B ^ 2 = A * B)
  (h2 : IsUnit (B * A - A * B)) :
  3 ∣ n := by
  sorry

end divisibility_by_three_l3317_331717


namespace b_not_two_l3317_331742

theorem b_not_two (b : ℝ) (h : ∀ x : ℝ, x ∈ Set.Icc 0 1 → |x + b| ≤ 2) : b ≠ 2 := by
  sorry

end b_not_two_l3317_331742


namespace root_implies_m_value_l3317_331748

theorem root_implies_m_value (m : ℚ) : 
  (∃ x : ℚ, x^2 - 6*x - 3*m - 5 = 0) ∧ 
  ((-1 : ℚ)^2 - 6*(-1) - 3*m - 5 = 0) → 
  m = 2/3 := by
sorry

end root_implies_m_value_l3317_331748


namespace sunflower_seed_tins_l3317_331761

theorem sunflower_seed_tins (candy_bags : ℕ) (candies_per_bag : ℕ) (seeds_per_tin : ℕ) (total_items : ℕ) : 
  candy_bags = 19 →
  candies_per_bag = 46 →
  seeds_per_tin = 170 →
  total_items = 1894 →
  (total_items - candy_bags * candies_per_bag) / seeds_per_tin = 6 :=
by sorry

end sunflower_seed_tins_l3317_331761


namespace sqrt_inequality_and_sum_of_squares_l3317_331766

theorem sqrt_inequality_and_sum_of_squares (a b c : ℝ) : 
  (Real.sqrt 6 + Real.sqrt 10 > 2 * Real.sqrt 3 + 2) ∧ 
  (a^2 + b^2 + c^2 ≥ a*b + b*c + a*c) := by
  sorry

end sqrt_inequality_and_sum_of_squares_l3317_331766


namespace stamp_cost_problem_l3317_331740

theorem stamp_cost_problem (total_stamps : ℕ) (high_denom : ℕ) (total_cost : ℚ) (high_denom_count : ℕ) :
  total_stamps = 20 →
  high_denom = 37 →
  total_cost = 706/100 →
  high_denom_count = 18 →
  ∃ (low_denom : ℕ),
    low_denom * (total_stamps - high_denom_count) = (total_cost * 100 - high_denom * high_denom_count : ℚ) ∧
    low_denom = 20 :=
by sorry

end stamp_cost_problem_l3317_331740


namespace max_gcd_sum_1085_l3317_331781

theorem max_gcd_sum_1085 :
  ∃ (m n : ℕ+), m + n = 1085 ∧ 
  ∀ (a b : ℕ+), a + b = 1085 → Nat.gcd a b ≤ Nat.gcd m n ∧
  Nat.gcd m n = 217 :=
by sorry

end max_gcd_sum_1085_l3317_331781


namespace semicircle_area_with_inscribed_rectangle_l3317_331753

theorem semicircle_area_with_inscribed_rectangle (r : ℝ) : 
  r > 0 → 
  (∃ (w h : ℝ), w > 0 ∧ h > 0 ∧ w = 1 ∧ h = 3 ∧ h = r) → 
  (π * r^2) / 2 = 9 * π / 2 := by
sorry

end semicircle_area_with_inscribed_rectangle_l3317_331753


namespace jake_present_weight_l3317_331758

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 156

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 224 - jake_weight

/-- Theorem stating Jake's present weight is 156 pounds -/
theorem jake_present_weight : jake_weight = 156 := by
  have h1 : jake_weight - 20 = 2 * sister_weight := by sorry
  have h2 : jake_weight + sister_weight = 224 := by sorry
  sorry

#check jake_present_weight

end jake_present_weight_l3317_331758


namespace min_distance_of_sine_extrema_l3317_331738

open Real

theorem min_distance_of_sine_extrema :
  ∀ (f : ℝ → ℝ) (x₁ x₂ : ℝ),
  (∀ x, f x = sin (π * x)) →
  (∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  (∃ (d : ℝ), d > 0 ∧ ∀ (y₁ y₂ : ℝ), (∀ x, f y₁ ≤ f x ∧ f x ≤ f y₂) → |y₁ - y₂| ≥ d) →
  (∀ (y₁ y₂ : ℝ), (∀ x, f y₁ ≤ f x ∧ f x ≤ f y₂) → |y₁ - y₂| ≥ 1) →
  |x₁ - x₂| = 1 := by
sorry

end min_distance_of_sine_extrema_l3317_331738


namespace system_solution_l3317_331722

theorem system_solution (a b c k x y z : ℝ) 
  (h1 : a * x + b * y + c * z = k)
  (h2 : a^2 * x + b^2 * y + c^2 * z = k^2)
  (h3 : a^3 * x + b^3 * y + c^3 * z = k^3)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  x = k * (k - c) * (k - b) / (a * (a - c) * (a - b)) ∧
  y = k * (k - c) * (k - a) / (b * (b - c) * (b - a)) ∧
  z = k * (k - a) * (k - b) / (c * (c - a) * (c - b)) := by
  sorry

end system_solution_l3317_331722


namespace dogwood_trees_planting_l3317_331793

theorem dogwood_trees_planting (initial_trees : ℕ) (planted_today : ℕ) (final_total : ℕ) 
  (h1 : initial_trees = 39)
  (h2 : planted_today = 41)
  (h3 : final_total = 100) :
  final_total - (initial_trees + planted_today) = 20 :=
by sorry

end dogwood_trees_planting_l3317_331793


namespace incorrect_quotient_calculation_l3317_331799

theorem incorrect_quotient_calculation (dividend : ℕ) (correct_divisor incorrect_divisor correct_quotient : ℕ) 
  (h1 : dividend = correct_divisor * correct_quotient)
  (h2 : correct_divisor = 21)
  (h3 : incorrect_divisor = 12)
  (h4 : correct_quotient = 28) :
  dividend / incorrect_divisor = 49 := by
sorry

end incorrect_quotient_calculation_l3317_331799


namespace total_waiting_time_bounds_l3317_331732

/-- 
Represents the total waiting time for a queue with Slowpokes and Quickies.
m: number of Slowpokes
n: number of Quickies
a: time taken by a Quickie
b: time taken by a Slowpoke
-/
def TotalWaitingTime (m n : ℕ) (a b : ℝ) : Prop :=
  let total := m + n
  ∀ (t_min t_max t_exp : ℝ),
    b > a →
    t_min = a * (n.choose 2) + a * m * n + b * (m.choose 2) →
    t_max = a * (n.choose 2) + b * m * n + b * (m.choose 2) →
    t_exp = (total.choose 2 : ℝ) * (b * m + a * n) / total →
    (t_min ≤ t_exp ∧ t_exp ≤ t_max)

theorem total_waiting_time_bounds {m n : ℕ} {a b : ℝ} :
  TotalWaitingTime m n a b :=
sorry

end total_waiting_time_bounds_l3317_331732


namespace arithmetic_sequence_common_difference_l3317_331783

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 2 + a 6 = 8 ∧ 
  a 3 + a 4 = 3

/-- The common difference of the arithmetic sequence is 5 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 5 := by
  sorry

end arithmetic_sequence_common_difference_l3317_331783


namespace bridge_building_time_l3317_331788

/-- If a crew of m workers can build a bridge in d days, then a crew of 2m workers can build the same bridge in d/2 days. -/
theorem bridge_building_time (m d : ℝ) (h1 : m > 0) (h2 : d > 0) :
  let initial_crew := m
  let initial_time := d
  let new_crew := 2 * m
  let new_time := d / 2
  initial_crew * initial_time = new_crew * new_time :=
by sorry

end bridge_building_time_l3317_331788


namespace odd_function_monotonicity_l3317_331780

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 1 / (4^x + 1)

theorem odd_function_monotonicity (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = -1/2 ∧ ∀ x₁ x₂, x₁ < x₂ → f a x₁ > f a x₂) := by sorry

end odd_function_monotonicity_l3317_331780


namespace north_village_conscripts_l3317_331724

/-- The number of people to be conscripted from a village, given its population and the total population and conscription numbers. -/
def conscriptsFromVillage (villagePopulation totalPopulation totalConscripts : ℕ) : ℕ :=
  (villagePopulation * totalConscripts) / totalPopulation

/-- Theorem stating that given the specific village populations and total conscripts, 
    the number of conscripts from the north village is 108. -/
theorem north_village_conscripts :
  let northPopulation : ℕ := 8100
  let westPopulation : ℕ := 7488
  let southPopulation : ℕ := 6912
  let totalConscripts : ℕ := 300
  let totalPopulation : ℕ := northPopulation + westPopulation + southPopulation
  conscriptsFromVillage northPopulation totalPopulation totalConscripts = 108 := by
  sorry

end north_village_conscripts_l3317_331724


namespace only_zero_is_purely_imaginary_l3317_331771

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number parameterized by m. -/
def complexNumber (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m, m^2 - 5*m + 6⟩

theorem only_zero_is_purely_imaginary :
  ∃! m : ℝ, isPurelyImaginary (complexNumber m) ∧ m = 0 := by
  sorry

end only_zero_is_purely_imaginary_l3317_331771


namespace square_difference_cubed_l3317_331760

theorem square_difference_cubed : (7^2 - 5^2)^3 = 13824 := by
  sorry

end square_difference_cubed_l3317_331760


namespace fraction_sum_equality_l3317_331768

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (25 - a) + b / (65 - b) + c / (60 - c) = 7) :
  5 / (25 - a) + 13 / (65 - b) + 12 / (60 - c) = 2 := by
  sorry

end fraction_sum_equality_l3317_331768


namespace correct_number_of_pitbulls_l3317_331774

/-- Represents the number of pitbulls James has -/
def num_pitbulls : ℕ := 2

/-- Represents the number of huskies James has -/
def num_huskies : ℕ := 5

/-- Represents the number of golden retrievers James has -/
def num_golden_retrievers : ℕ := 4

/-- Represents the number of pups each husky and pitbull has -/
def pups_per_husky_pitbull : ℕ := 3

/-- Represents the additional number of pups each golden retriever has compared to huskies -/
def additional_pups_golden : ℕ := 2

/-- Represents the difference between total pups and adult dogs -/
def pup_adult_difference : ℕ := 30

theorem correct_number_of_pitbulls :
  (num_huskies * pups_per_husky_pitbull) +
  (num_golden_retrievers * (pups_per_husky_pitbull + additional_pups_golden)) +
  (num_pitbulls * pups_per_husky_pitbull) =
  (num_huskies + num_golden_retrievers + num_pitbulls) + pup_adult_difference := by
  sorry

end correct_number_of_pitbulls_l3317_331774


namespace unique_modular_congruence_l3317_331750

theorem unique_modular_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ 12019 [ZMOD 13] := by
  sorry

end unique_modular_congruence_l3317_331750


namespace matches_played_eq_teams_minus_one_l3317_331779

/-- Represents an elimination tournament. -/
structure EliminationTournament where
  num_teams : ℕ
  no_replays : Bool

/-- The number of matches played in an elimination tournament. -/
def matches_played (t : EliminationTournament) : ℕ := sorry

/-- Theorem stating that in an elimination tournament with no replays, 
    the number of matches played is one less than the number of teams. -/
theorem matches_played_eq_teams_minus_one (t : EliminationTournament) 
  (h : t.no_replays = true) : matches_played t = t.num_teams - 1 := by sorry

end matches_played_eq_teams_minus_one_l3317_331779


namespace no_prime_sum_10003_l3317_331731

/-- A function that returns the number of ways to write n as the sum of two primes -/
def countPrimeSumWays (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range n)).card

/-- Theorem stating that 10003 cannot be written as the sum of two primes -/
theorem no_prime_sum_10003 : countPrimeSumWays 10003 = 0 := by
  sorry

end no_prime_sum_10003_l3317_331731


namespace circle_radius_from_area_l3317_331752

/-- Given a circle with area 49π, prove its radius is 7 -/
theorem circle_radius_from_area (A : ℝ) (r : ℝ) : 
  A = 49 * Real.pi → A = Real.pi * r^2 → r = 7 := by
  sorry

end circle_radius_from_area_l3317_331752


namespace power_equation_solution_l3317_331763

theorem power_equation_solution (p : ℕ) : (81 ^ 10 : ℕ) = 3 ^ p → p = 40 := by
  sorry

end power_equation_solution_l3317_331763


namespace condition_relationship_l3317_331777

theorem condition_relationship (x : ℝ) :
  (∀ x, x > 1 → 1 / x < 1) ∧ 
  (∃ x, 1 / x < 1 ∧ ¬(x > 1)) := by
  sorry

end condition_relationship_l3317_331777


namespace opposite_direction_speed_l3317_331772

/-- Given two people moving in opposite directions for 1 hour, 
    where one moves at 35 km/h and they end up 60 km apart,
    prove that the speed of the other person is 25 km/h. -/
theorem opposite_direction_speed 
  (speed_person1 : ℝ) 
  (speed_person2 : ℝ) 
  (time : ℝ) 
  (total_distance : ℝ) 
  (h1 : speed_person2 = 35) 
  (h2 : time = 1) 
  (h3 : total_distance = 60) 
  (h4 : speed_person1 * time + speed_person2 * time = total_distance) : 
  speed_person1 = 25 := by
  sorry

#check opposite_direction_speed

end opposite_direction_speed_l3317_331772


namespace winter_hamburger_sales_l3317_331729

/-- Given the total annual sales and percentages for spring and summer,
    calculate the number of hamburgers sold in winter. -/
theorem winter_hamburger_sales
  (total_sales : ℝ)
  (spring_percent : ℝ)
  (summer_percent : ℝ)
  (h_total : total_sales = 20)
  (h_spring : spring_percent = 0.3)
  (h_summer : summer_percent = 0.35) :
  total_sales - (spring_percent * total_sales + summer_percent * total_sales + (1 - spring_percent - summer_percent) / 2 * total_sales) = 3.5 :=
sorry

end winter_hamburger_sales_l3317_331729


namespace digit_value_difference_l3317_331721

/-- The numeral we are working with -/
def numeral : ℕ := 657903

/-- The digit we are focusing on -/
def digit : ℕ := 7

/-- The position of the digit in the numeral (counting from right, starting at 0) -/
def position : ℕ := 4

/-- The local value of a digit in a given position -/
def local_value (d : ℕ) (pos : ℕ) : ℕ := d * (10 ^ pos)

/-- The face value of a digit -/
def face_value (d : ℕ) : ℕ := d

/-- The difference between local value and face value -/
def value_difference (d : ℕ) (pos : ℕ) : ℕ := local_value d pos - face_value d

theorem digit_value_difference :
  value_difference digit position = 69993 := by sorry

end digit_value_difference_l3317_331721


namespace triangle_angle_inequality_l3317_331705

theorem triangle_angle_inequality (f : ℝ → ℝ) (α β : ℝ) : 
  (∀ x y, x ∈ [-1, 1] → y ∈ [-1, 1] → x < y → f x > f y) →  -- f is decreasing on [-1,1]
  0 < α →                                                   -- α is positive
  0 < β →                                                   -- β is positive
  α < π / 2 →                                               -- α is less than π/2
  β < π / 2 →                                               -- β is less than π/2
  α + β > π / 2 →                                           -- sum of α and β is greater than π/2
  α ≠ β →                                                   -- α and β are distinct
  f (Real.cos α) > f (Real.sin β) :=                        -- prove this inequality
by sorry

end triangle_angle_inequality_l3317_331705


namespace choir_arrangement_l3317_331789

theorem choir_arrangement (n : ℕ) : 
  (∃ k : ℕ, n = 9 * k) ∧ 
  (∃ k : ℕ, n = 10 * k) ∧ 
  (∃ k : ℕ, n = 11 * k) ↔ 
  n ≥ 990 ∧ n % 990 = 0 :=
by sorry

end choir_arrangement_l3317_331789


namespace function_translation_transformation_result_l3317_331746

-- Define the original function
def f (x : ℝ) : ℝ := 2 * (x + 1)^2 - 3

-- Define the transformed function
def g (x : ℝ) : ℝ := 2 * x^2

-- Theorem stating that g is the result of translating f
theorem function_translation (x : ℝ) : 
  g x = f (x - 1) + 3 := by
  sorry

-- Prove that the transformation results in g
theorem transformation_result : 
  ∀ x, g x = 2 * x^2 := by
  sorry

end function_translation_transformation_result_l3317_331746


namespace system_solution_l3317_331770

theorem system_solution (x y z : ℚ) : 
  (1/x + 1/y = 6) ∧ (1/y + 1/z = 4) ∧ (1/z + 1/x = 5) → 
  (x = 2/7) ∧ (y = 2/5) ∧ (z = 2/3) := by
sorry

end system_solution_l3317_331770


namespace yolanda_free_throws_l3317_331700

/-- Calculates the average number of free throws per game given the total points,
    number of games, and average two-point and three-point baskets per game. -/
def avg_free_throws (total_points : ℕ) (num_games : ℕ) 
                    (avg_two_point : ℕ) (avg_three_point : ℕ) : ℕ :=
  let avg_points_per_game := total_points / num_games
  let points_from_two_point := avg_two_point * 2
  let points_from_three_point := avg_three_point * 3
  avg_points_per_game - (points_from_two_point + points_from_three_point)

theorem yolanda_free_throws : 
  avg_free_throws 345 15 5 3 = 4 := by
  sorry

end yolanda_free_throws_l3317_331700


namespace perpendicular_tangents_intersection_l3317_331787

/-- The y-coordinate of the intersection point of perpendicular tangents on y = 4x^2 -/
theorem perpendicular_tangents_intersection (a b : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    A.2 = 4 * A.1^2 ∧ 
    B.2 = 4 * B.1^2 ∧ 
    A.1 = a ∧ 
    B.1 = b ∧ 
    (8 * a) * (8 * b) = -1) → 
  ∃ P : ℝ × ℝ, 
    (P.1 = (a + b) / 2) ∧ 
    (P.2 = -2) := by
  sorry

end perpendicular_tangents_intersection_l3317_331787


namespace correct_algorithm_structures_l3317_331785

-- Define the possible algorithm structures
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop
  | Flow
  | Nested

-- Define a function that checks if a list of structures is correct
def isCorrectStructureList (list : List AlgorithmStructure) : Prop :=
  list = [AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop]

-- State the theorem
theorem correct_algorithm_structures :
  isCorrectStructureList [AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop] :=
by sorry


end correct_algorithm_structures_l3317_331785


namespace principal_amount_proof_l3317_331796

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement --/
theorem principal_amount_proof :
  let final_amount : ℝ := 8820
  let rate : ℝ := 0.05
  let time : ℕ := 2
  ∃ (principal : ℝ), principal = 8000 ∧ compound_interest principal rate time = final_amount := by
sorry

end principal_amount_proof_l3317_331796


namespace stadium_fee_difference_l3317_331743

/-- Calculates the difference in total fees collected between full capacity and 3/4 capacity for a stadium. -/
def fee_difference (capacity : ℕ) (entry_fee : ℕ) : ℕ :=
  capacity * entry_fee - (capacity * 3 / 4) * entry_fee

/-- Proves that the fee difference for a stadium with 2000 capacity and $20 entry fee is $10,000. -/
theorem stadium_fee_difference :
  fee_difference 2000 20 = 10000 := by
  sorry

#eval fee_difference 2000 20

end stadium_fee_difference_l3317_331743


namespace book_pages_calculation_l3317_331782

theorem book_pages_calculation (pages_read : ℕ) (fraction_read : ℚ) (h1 : pages_read = 16) (h2 : fraction_read = 0.4) : 
  (pages_read : ℚ) / fraction_read = 40 := by
sorry

end book_pages_calculation_l3317_331782


namespace abc_value_l3317_331711

noncomputable def A (x : ℝ) : ℝ := ∑' k, x^(3*k) / (3*k).factorial
noncomputable def B (x : ℝ) : ℝ := ∑' k, x^(3*k+1) / (3*k+1).factorial
noncomputable def C (x : ℝ) : ℝ := ∑' k, x^(3*k+2) / (3*k+2).factorial

theorem abc_value (x : ℝ) (hx : x > 0) :
  (A x)^3 + (B x)^3 + (C x)^3 + 8*(A x)*(B x)*(C x) = 2014 →
  (A x)*(B x)*(C x) = 183 := by
sorry

end abc_value_l3317_331711


namespace ten_balls_distribution_l3317_331736

/-- The number of ways to distribute n identical balls into 3 boxes numbered 1, 2, and 3,
    where each box must contain at least as many balls as its number. -/
def distributionWays (n : ℕ) : ℕ :=
  let remainingBalls := n - (1 + 2 + 3)
  (remainingBalls + 3 - 1).choose 2

/-- Theorem: There are 15 ways to distribute 10 identical balls into 3 boxes numbered 1, 2, and 3,
    where each box must contain at least as many balls as its number. -/
theorem ten_balls_distribution : distributionWays 10 = 15 := by
  sorry

#eval distributionWays 10  -- Should output 15

end ten_balls_distribution_l3317_331736


namespace probability_x_less_than_2y_l3317_331773

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

-- Define the region where x < 2y
def region : Set (ℝ × ℝ) :=
  {p ∈ rectangle | p.1 < 2 * p.2}

-- Define the probability measure on the rectangle
noncomputable def prob : MeasureTheory.Measure (ℝ × ℝ) :=
  sorry

-- State the theorem
theorem probability_x_less_than_2y :
  prob region / prob rectangle = 1 / 2 := by
  sorry

end probability_x_less_than_2y_l3317_331773


namespace pricing_equation_l3317_331745

/-- 
Given an item with:
- cost price x (in yuan)
- markup percentage m (as a decimal)
- discount percentage d (as a decimal)
- final selling price s (in yuan)

This theorem states that the equation relating these values is:
x * (1 + m) * (1 - d) = s
-/
theorem pricing_equation (x m d s : ℝ) 
  (markup : m = 0.3)
  (discount : d = 0.2)
  (selling_price : s = 2080) :
  x * (1 + m) * (1 - d) = s :=
sorry

end pricing_equation_l3317_331745


namespace total_red_cards_l3317_331720

/-- The number of decks the shopkeeper has -/
def num_decks : ℕ := 7

/-- The number of red cards in one deck -/
def red_cards_per_deck : ℕ := 26

/-- Theorem: The total number of red cards the shopkeeper has is 182 -/
theorem total_red_cards : num_decks * red_cards_per_deck = 182 := by
  sorry

end total_red_cards_l3317_331720


namespace output_value_scientific_notation_l3317_331715

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem output_value_scientific_notation :
  toScientificNotation 110000000000 = ScientificNotation.mk 1.1 10 (by norm_num) :=
sorry

end output_value_scientific_notation_l3317_331715


namespace multiples_of_ten_not_twenty_l3317_331759

def count_numbers (n : ℕ) : ℕ :=
  (n.div 10 + 1).div 2

theorem multiples_of_ten_not_twenty (upper_bound : ℕ) (h : upper_bound = 500) :
  count_numbers upper_bound = 25 := by
  sorry

end multiples_of_ten_not_twenty_l3317_331759


namespace min_players_sum_divisible_by_10_l3317_331739

/-- Represents a 3x9 grid of distinct non-negative integers -/
def Grid := Matrix (Fin 3) (Fin 9) ℕ

/-- Predicate to check if all elements in a grid are distinct -/
def all_distinct (g : Grid) : Prop :=
  ∀ i j i' j', (i ≠ i' ∨ j ≠ j') → g i j ≠ g i' j'

/-- Predicate to check if a sum is divisible by 10 -/
def sum_divisible_by_10 (a b : ℕ) : Prop :=
  (a + b) % 10 = 0

/-- Main theorem statement -/
theorem min_players_sum_divisible_by_10 (g : Grid) (h : all_distinct g) :
  ∃ i j i' j', sum_divisible_by_10 (g i j) (g i' j') :=
sorry

end min_players_sum_divisible_by_10_l3317_331739


namespace smallest_prime_divides_infinitely_many_and_all_l3317_331791

def a (n : ℕ) : ℕ := 4^(2*n+1) + 3^(n+2)

def is_divisible_by (m n : ℕ) : Prop := ∃ k, m = n * k

def divides_infinitely_many (p : ℕ) : Prop :=
  ∀ N, ∃ n ≥ N, is_divisible_by (a n) p

def divides_all (p : ℕ) : Prop :=
  ∀ n, n ≥ 1 → is_divisible_by (a n) p

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m, 1 < m → m < p → ¬(is_divisible_by p m)

theorem smallest_prime_divides_infinitely_many_and_all :
  ∃ (p q : ℕ),
    is_prime p ∧
    is_prime q ∧
    divides_infinitely_many p ∧
    divides_all q ∧
    (∀ p', is_prime p' → divides_infinitely_many p' → p ≤ p') ∧
    (∀ q', is_prime q' → divides_all q' → q ≤ q') ∧
    p = 5 ∧
    q = 13 :=
  sorry

end smallest_prime_divides_infinitely_many_and_all_l3317_331791


namespace steps_to_eleventh_floor_l3317_331727

/-- Given that there are 42 steps between the 3rd and 5th floors of a building,
    prove that there are 210 steps from the ground floor to the 11th floor. -/
theorem steps_to_eleventh_floor :
  let steps_between_3_and_5 : ℕ := 42
  let floor_xiao_dong_lives : ℕ := 11
  let ground_floor : ℕ := 1
  let steps_to_xiao_dong : ℕ := (floor_xiao_dong_lives - ground_floor) * 
    (steps_between_3_and_5 / (5 - 3))
  steps_to_xiao_dong = 210 := by
  sorry


end steps_to_eleventh_floor_l3317_331727


namespace cube_volume_from_space_diagonal_l3317_331730

/-- The volume of a cube with a space diagonal of 6√3 units is 216 cubic units. -/
theorem cube_volume_from_space_diagonal :
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 6 * Real.sqrt 3 → s^3 = 216 := by
  sorry

end cube_volume_from_space_diagonal_l3317_331730
