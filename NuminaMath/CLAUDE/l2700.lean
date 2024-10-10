import Mathlib

namespace quadratic_equations_count_l2700_270041

variable (p : ℕ) [Fact (Nat.Prime p)]

/-- The number of quadratic equations with two distinct roots in p-arithmetic -/
def two_roots (p : ℕ) : ℕ := p * (p - 1) / 2

/-- The number of quadratic equations with exactly one root in p-arithmetic -/
def one_root (p : ℕ) : ℕ := p

/-- The number of quadratic equations with no roots in p-arithmetic -/
def no_roots (p : ℕ) : ℕ := p * (p - 1) / 2

/-- The total number of distinct quadratic equations in p-arithmetic -/
def total_equations (p : ℕ) : ℕ := p^2

theorem quadratic_equations_count (p : ℕ) [Fact (Nat.Prime p)] :
  two_roots p + one_root p + no_roots p = total_equations p :=
sorry

end quadratic_equations_count_l2700_270041


namespace prime_sum_theorem_l2700_270072

theorem prime_sum_theorem (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h_less : p < q) (h_eq : p * q + p^2 + q^2 = 199) : 
  (Finset.range (q - p)).sum (fun k => 2 / ((p + k) * (p + k + 1))) = 11 / 13 := by
  sorry

end prime_sum_theorem_l2700_270072


namespace burger_calories_l2700_270079

/-- Calculates the number of calories per burger given the following conditions:
  * 10 burritos cost $6
  * Each burrito has 120 calories
  * 5 burgers cost $8
  * Burgers provide 50 more calories per dollar than burritos
-/
theorem burger_calories :
  let burrito_count : ℕ := 10
  let burrito_cost : ℚ := 6
  let burrito_calories : ℕ := 120
  let burger_count : ℕ := 5
  let burger_cost : ℚ := 8
  let calorie_difference_per_dollar : ℕ := 50
  
  let burrito_calories_per_dollar : ℚ := (burrito_count * burrito_calories : ℚ) / burrito_cost
  let burger_calories_per_dollar : ℚ := burrito_calories_per_dollar + calorie_difference_per_dollar
  let total_burger_calories : ℚ := burger_calories_per_dollar * burger_cost
  let calories_per_burger : ℚ := total_burger_calories / burger_count
  
  calories_per_burger = 400 := by
    sorry

end burger_calories_l2700_270079


namespace flower_count_l2700_270059

theorem flower_count (minyoung_flowers yoojung_flowers : ℕ) : 
  minyoung_flowers = 24 → 
  minyoung_flowers = 4 * yoojung_flowers → 
  minyoung_flowers + yoojung_flowers = 30 := by
sorry

end flower_count_l2700_270059


namespace perpendicular_bisecting_diagonals_not_imply_square_l2700_270087

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define properties of quadrilaterals
def has_perpendicular_bisecting_diagonals (q : Quadrilateral) : Prop :=
  sorry

def is_square (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_bisecting_diagonals_not_imply_square :
  ¬ (∀ q : Quadrilateral, has_perpendicular_bisecting_diagonals q → is_square q) :=
sorry

end perpendicular_bisecting_diagonals_not_imply_square_l2700_270087


namespace quadratic_equation_solution_l2700_270025

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 6 ∧ x₂ = -2) ∧ 
  (x₁^2 - 4*x₁ - 12 = 0) ∧ 
  (x₂^2 - 4*x₂ - 12 = 0) :=
by sorry

end quadratic_equation_solution_l2700_270025


namespace negative_fraction_comparison_l2700_270098

theorem negative_fraction_comparison : -3/4 > -5/6 := by
  sorry

end negative_fraction_comparison_l2700_270098


namespace kendra_shirts_theorem_l2700_270020

/-- Calculates the number of shirts Kendra needs for two weeks -/
def shirts_needed : ℕ :=
  let school_days := 5
  let after_school_days := 3
  let saturday_shirts := 1
  let sunday_shirts := 2
  let weeks := 2
  (school_days + after_school_days + saturday_shirts + sunday_shirts) * weeks

/-- Theorem stating that Kendra needs 22 shirts for two weeks -/
theorem kendra_shirts_theorem : shirts_needed = 22 := by
  sorry

end kendra_shirts_theorem_l2700_270020


namespace switch_strategy_wins_l2700_270052

/-- Represents the three boxes in the game -/
inductive Box
| A
| B
| C

/-- Represents the possible states of a box -/
inductive BoxState
| Prize
| Empty

/-- Represents the game state -/
structure GameState where
  boxes : Box → BoxState
  initialChoice : Box
  hostOpened : Box
  finalChoice : Box

/-- The probability of winning by switching in the three-box game -/
def winProbabilityBySwitch (game : GameState) : ℚ :=
  2/3

/-- Theorem stating that the probability of winning by switching is greater than 1/2 -/
theorem switch_strategy_wins (game : GameState) :
  winProbabilityBySwitch game > 1/2 := by
  sorry

end switch_strategy_wins_l2700_270052


namespace average_age_of_five_students_l2700_270021

/-- Proves that the average age of 5 students is 14 years given the conditions of the problem -/
theorem average_age_of_five_students
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_students_with_known_average : Nat)
  (average_age_known : ℝ)
  (age_of_twelfth_student : ℕ)
  (h1 : total_students = 16)
  (h2 : average_age_all = 16)
  (h3 : num_students_with_known_average = 9)
  (h4 : average_age_known = 16)
  (h5 : age_of_twelfth_student = 42)
  : (total_students * average_age_all - num_students_with_known_average * average_age_known - age_of_twelfth_student) / (total_students - num_students_with_known_average - 1) = 14 := by
  sorry

end average_age_of_five_students_l2700_270021


namespace sqrt_3_irrational_l2700_270034

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_3_irrational_l2700_270034


namespace correct_sum_after_mistake_l2700_270046

/-- Given two two-digit numbers where a ones digit 7 is mistaken for 1
    and a tens digit 4 is mistaken for 6, resulting in a sum of 146,
    prove that the correct sum is 132. -/
theorem correct_sum_after_mistake (a b c d : ℕ) : 
  a ≤ 9 → b ≤ 9 → c ≤ 9 → d ≤ 9 →  -- Ensure all digits are single-digit
  (10 * a + 7) + (40 + d) = 146 →  -- Mistaken sum equation
  (10 * a + 7) + (40 + d) = 132 := by
sorry

end correct_sum_after_mistake_l2700_270046


namespace toilet_paper_supply_duration_l2700_270066

/-- Calculates the number of days a toilet paper supply will last for a family -/
def toilet_paper_duration (bill_usage : ℕ) (wife_usage : ℕ) (kid_usage : ℕ) (num_kids : ℕ) (num_rolls : ℕ) (squares_per_roll : ℕ) : ℕ :=
  let total_squares := num_rolls * squares_per_roll
  let daily_usage := bill_usage + wife_usage + kid_usage * num_kids
  total_squares / daily_usage

theorem toilet_paper_supply_duration :
  toilet_paper_duration 15 32 30 2 1000 300 = 2803 := by
  sorry

end toilet_paper_supply_duration_l2700_270066


namespace shoe_discount_ratio_l2700_270081

theorem shoe_discount_ratio (price1 price2 final_price : ℚ) : 
  price1 = 40 →
  price2 = 60 →
  final_price = 60 →
  let total := price1 + price2
  let extra_discount := total / 4
  let discounted_total := total - extra_discount
  let cheaper_discount := discounted_total - final_price
  (cheaper_discount / price1) = 3 / 8 := by
sorry

end shoe_discount_ratio_l2700_270081


namespace parabola_focus_l2700_270029

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = -4 * x^2 + 4 * x - 1

/-- The focus of a parabola -/
def is_focus (f_x f_y : ℝ) : Prop :=
  f_x = 1/2 ∧ f_y = -1/8

/-- Theorem: The focus of the parabola y = -4x^2 + 4x - 1 is (1/2, -1/8) -/
theorem parabola_focus :
  ∃ (f_x f_y : ℝ), (∀ x y, parabola_equation x y → is_focus f_x f_y) :=
sorry

end parabola_focus_l2700_270029


namespace adults_fed_is_22_l2700_270010

/-- Represents the resources and feeding capabilities of a community center -/
structure CommunityCenter where
  soup_cans : ℕ
  bread_loaves : ℕ
  adults_per_can : ℕ
  children_per_can : ℕ
  adults_per_loaf : ℕ
  children_per_loaf : ℕ

/-- Calculates the number of adults that can be fed with remaining resources -/
def adults_fed_after_children (cc : CommunityCenter) (children_to_feed : ℕ) : ℕ :=
  let cans_for_children := (children_to_feed + cc.children_per_can - 1) / cc.children_per_can
  let remaining_cans := cc.soup_cans - cans_for_children
  let adults_fed_by_cans := remaining_cans * cc.adults_per_can
  let adults_fed_by_bread := cc.bread_loaves * cc.adults_per_loaf
  adults_fed_by_cans + adults_fed_by_bread

/-- Theorem stating that 22 adults can be fed with remaining resources -/
theorem adults_fed_is_22 (cc : CommunityCenter) (h1 : cc.soup_cans = 8) (h2 : cc.bread_loaves = 2)
    (h3 : cc.adults_per_can = 4) (h4 : cc.children_per_can = 7) (h5 : cc.adults_per_loaf = 3)
    (h6 : cc.children_per_loaf = 4) :
    adults_fed_after_children cc 24 = 22 := by
  sorry

end adults_fed_is_22_l2700_270010


namespace number_of_pupils_theorem_l2700_270095

/-- The number of pupils sent up for examination -/
def N : ℕ := 28

/-- The average marks of all pupils -/
def overall_average : ℚ := 39

/-- The average marks if 7 specific pupils were not sent up -/
def new_average : ℚ := 45

/-- The marks of the 7 specific pupils -/
def specific_pupils_marks : List ℕ := [25, 12, 15, 19, 31, 18, 27]

/-- The sum of marks of the 7 specific pupils -/
def sum_specific_marks : ℕ := specific_pupils_marks.sum

theorem number_of_pupils_theorem :
  (N * overall_average - sum_specific_marks) / (N - 7) = new_average :=
sorry

end number_of_pupils_theorem_l2700_270095


namespace smallest_solution_abs_equation_l2700_270088

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 4 * x + 3 ∧
  (∀ (y : ℝ), y * |y| = 4 * y + 3 → x ≤ y) ∧
  x = -3 := by
  sorry

end smallest_solution_abs_equation_l2700_270088


namespace workshop_average_age_l2700_270067

theorem workshop_average_age (total_members : ℕ) (overall_avg : ℝ)
  (num_girls num_boys num_adults num_teens : ℕ)
  (avg_girls avg_boys avg_teens : ℝ) :
  total_members = 50 →
  overall_avg = 20 →
  num_girls = 25 →
  num_boys = 15 →
  num_adults = 5 →
  num_teens = 5 →
  avg_girls = 18 →
  avg_boys = 19 →
  avg_teens = 16 →
  (total_members : ℝ) * overall_avg =
    (num_girls : ℝ) * avg_girls + (num_boys : ℝ) * avg_boys +
    (num_adults : ℝ) * ((total_members : ℝ) * overall_avg - (num_girls : ℝ) * avg_girls -
    (num_boys : ℝ) * avg_boys - (num_teens : ℝ) * avg_teens) / num_adults +
    (num_teens : ℝ) * avg_teens →
  ((total_members : ℝ) * overall_avg - (num_girls : ℝ) * avg_girls -
   (num_boys : ℝ) * avg_boys - (num_teens : ℝ) * avg_teens) / num_adults = 37 :=
by sorry

end workshop_average_age_l2700_270067


namespace circle_line_theorem_l2700_270093

/-- Given two circles C₁ and C₂ passing through (2, -1), prove that the line
    through (D₁, E₁) and (D₂, E₂) has equation 2x - y + 2 = 0 -/
theorem circle_line_theorem (D₁ E₁ D₂ E₂ : ℝ) : 
  (2^2 + (-1)^2 + 2*D₁ - E₁ - 3 = 0) →
  (2^2 + (-1)^2 + 2*D₂ - E₂ - 3 = 0) →
  ∃ (k : ℝ), 2*D₁ - E₁ + 2 = k ∧ 2*D₂ - E₂ + 2 = k :=
by sorry

end circle_line_theorem_l2700_270093


namespace average_of_a_and_b_l2700_270043

theorem average_of_a_and_b (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50)
  (h2 : (b + c) / 2 = 70)
  (h3 : c - a = 40) :
  (a + b) / 2 = 50 := by
sorry

end average_of_a_and_b_l2700_270043


namespace max_linear_term_bound_l2700_270030

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem max_linear_term_bound {a b c : ℝ} :
  (∀ x : ℝ, |x| ≤ 1 → |quadratic_function a b c x| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |a * x + b| ≤ 2) ∧
  (∃ a b : ℝ, ∃ x : ℝ, |x| ≤ 1 ∧ |a * x + b| = 2) :=
sorry

end max_linear_term_bound_l2700_270030


namespace profit_per_meter_is_55_l2700_270019

/-- Profit per meter of cloth -/
def profit_per_meter (total_meters : ℕ) (total_profit : ℕ) : ℚ :=
  total_profit / total_meters

/-- Theorem: The profit per meter of cloth is 55 rupees -/
theorem profit_per_meter_is_55
  (total_meters : ℕ)
  (selling_price : ℕ)
  (total_profit : ℕ)
  (h1 : total_meters = 40)
  (h2 : selling_price = 8200)
  (h3 : total_profit = 2200) :
  profit_per_meter total_meters total_profit = 55 := by
  sorry

end profit_per_meter_is_55_l2700_270019


namespace all_girls_same_color_probability_l2700_270082

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 10

/-- Represents the total number of marbles -/
def total_marbles : ℕ := 30

/-- Represents the number of girls selecting marbles -/
def num_girls : ℕ := 15

/-- The probability that all girls select the same colored marble -/
def probability_same_color : ℚ := 0

theorem all_girls_same_color_probability :
  marbles_per_color = 10 →
  total_marbles = 30 →
  num_girls = 15 →
  probability_same_color = 0 := by
  sorry

end all_girls_same_color_probability_l2700_270082


namespace cylinder_radius_proof_l2700_270048

theorem cylinder_radius_proof (r : ℝ) : 
  let h : ℝ := 3
  let volume (r h : ℝ) := π * r^2 * h
  let volume_increase_height := volume r (h + 3) - volume r h
  let volume_increase_radius := volume (r + 3) h - volume r h
  volume_increase_height = volume_increase_radius →
  r = 3 + 3 * Real.sqrt 2 :=
by sorry

end cylinder_radius_proof_l2700_270048


namespace correct_cracker_distribution_l2700_270054

/-- Represents the distribution of crackers to friends -/
structure CrackerDistribution where
  initial : ℕ
  first_fraction : ℚ
  second_percentage : ℚ
  third_remaining : ℕ

/-- Calculates the number of crackers each friend receives -/
def distribute_crackers (d : CrackerDistribution) : ℕ × ℕ × ℕ := sorry

/-- Theorem stating the correct distribution of crackers -/
theorem correct_cracker_distribution :
  let d := CrackerDistribution.mk 100 (2/3) (37/200) 7
  distribute_crackers d = (66, 6, 7) := by sorry

end correct_cracker_distribution_l2700_270054


namespace circle_constant_value_l2700_270024

-- Define the circle equation
def circle_equation (x y c : ℝ) : Prop :=
  x^2 + 4*x + y^2 + 8*y + c = 0

-- Define the center of the circle
def circle_center (x y : ℝ) : Prop :=
  x = -2 ∧ y = -4

-- Define the radius of the circle
def circle_radius (r : ℝ) : Prop :=
  r = 5

-- Theorem statement
theorem circle_constant_value :
  ∀ (c : ℝ), 
  (∀ (x y : ℝ), circle_equation x y c → 
    ∃ (h k : ℝ), circle_center h k ∧ 
    ∃ (r : ℝ), circle_radius r ∧ 
    (x - h)^2 + (y - k)^2 = r^2) →
  c = -5 := by sorry

end circle_constant_value_l2700_270024


namespace greatest_integer_square_thrice_plus_81_l2700_270042

theorem greatest_integer_square_thrice_plus_81 :
  ∀ x : ℤ, x^2 = 3*x + 81 → x ≤ 9 :=
by sorry

end greatest_integer_square_thrice_plus_81_l2700_270042


namespace calculation_proof_l2700_270026

theorem calculation_proof : (180 : ℚ) / (15 + 12 * 3 - 9) = 30 / 7 := by
  sorry

end calculation_proof_l2700_270026


namespace max_sum_digits_divisible_by_13_l2700_270076

theorem max_sum_digits_divisible_by_13 :
  ∀ A B C : ℕ,
  A < 10 → B < 10 → C < 10 →
  (2000 + 100 * A + 10 * B + C) % 13 = 0 →
  A + B + C ≤ 26 :=
by sorry

end max_sum_digits_divisible_by_13_l2700_270076


namespace factorization_equality_l2700_270036

theorem factorization_equality (x y : ℝ) : 2 * x^2 * y - 8 * y = 2 * y * (x + 2) * (x - 2) := by
  sorry

end factorization_equality_l2700_270036


namespace pass_through_walls_l2700_270027

theorem pass_through_walls (k : ℕ) (n : ℕ) : 
  k = 8 → 
  (k * Real.sqrt (k / ((k - 1) * k + (k - 1))) = Real.sqrt (k * (k / n))) ↔ 
  n = 63 := by
sorry

end pass_through_walls_l2700_270027


namespace smallest_angle_measure_l2700_270089

/-- The measure of a right angle in degrees -/
def right_angle : ℝ := 90

/-- A triangle with one angle 80% larger than a right angle -/
structure SpecialIsoscelesTriangle where
  /-- The measure of the largest angle in degrees -/
  large_angle : ℝ
  /-- The fact that the largest angle is 80% larger than a right angle -/
  angle_condition : large_angle = 1.8 * right_angle

theorem smallest_angle_measure (t : SpecialIsoscelesTriangle) :
  (180 - t.large_angle) / 2 = 9 := by sorry

end smallest_angle_measure_l2700_270089


namespace right_triangle_area_l2700_270086

/-- The area of a right triangle with one leg of 30 inches and a hypotenuse of 34 inches is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 := by
  sorry

end right_triangle_area_l2700_270086


namespace eq1_represents_parallel_lines_eq2_represents_four_lines_eq3_represents_specific_lines_eq4_represents_half_circle_l2700_270091

-- Define the equations
def eq1 (x y : ℝ) : Prop := (2*x - y)^2 = 1
def eq2 (x y : ℝ) : Prop := 16*x^4 - 8*x^2*y^2 + y^4 - 8*x^2 - 2*y^2 + 1 = 0
def eq3 (x y : ℝ) : Prop := x^2*(1 - abs y / y) + y^2 + y*(abs y) = 8
def eq4 (x y : ℝ) : Prop := x^2 + x*(abs x) + y^2 + (abs x)*y^2/x = 8

-- Define geometric shapes
def ParallelLines (f : ℝ → ℝ → Prop) : Prop := 
  ∃ a b c d : ℝ, ∀ x y : ℝ, f x y ↔ (y = a*x + b ∨ y = c*x + d) ∧ a = c ∧ b ≠ d

def FourLines (f : ℝ → ℝ → Prop) : Prop := 
  ∃ a₁ b₁ a₂ b₂ a₃ b₃ a₄ b₄ : ℝ, ∀ x y : ℝ, 
    f x y ↔ (y = a₁*x + b₁ ∨ y = a₂*x + b₂ ∨ y = a₃*x + b₃ ∨ y = a₄*x + b₄)

def SpecificLines (f : ℝ → ℝ → Prop) : Prop := 
  ∃ a b c : ℝ, ∀ x y : ℝ, 
    f x y ↔ ((y > 0 ∧ y = a) ∨ (y < 0 ∧ (x = b ∨ x = c)))

def HalfCircle (f : ℝ → ℝ → Prop) : Prop := 
  ∃ r : ℝ, ∀ x y : ℝ, f x y ↔ x > 0 ∧ x^2 + y^2 = r^2

-- Theorem statements
theorem eq1_represents_parallel_lines : ParallelLines eq1 := sorry

theorem eq2_represents_four_lines : FourLines eq2 := sorry

theorem eq3_represents_specific_lines : SpecificLines eq3 := sorry

theorem eq4_represents_half_circle : HalfCircle eq4 := sorry

end eq1_represents_parallel_lines_eq2_represents_four_lines_eq3_represents_specific_lines_eq4_represents_half_circle_l2700_270091


namespace reciprocal_difference_sequence_l2700_270011

theorem reciprocal_difference_sequence (a : ℕ → ℚ) :
  a 1 = 1/3 ∧
  (∀ n : ℕ, n > 1 → a n = 1 / (1 - a (n-1))) →
  a 2023 = 1/3 := by
  sorry

end reciprocal_difference_sequence_l2700_270011


namespace min_value_of_expression_l2700_270022

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + 3 * c)) + (b / (8 * c + 4 * a)) + (9 * c / (3 * a + 2 * b)) ≥ 47 / 48 := by
  sorry

end min_value_of_expression_l2700_270022


namespace initial_kittens_l2700_270039

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica : ℕ := 3

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara : ℕ := 6

/-- The number of kittens Tim has left -/
def kittens_left : ℕ := 9

/-- Theorem: Tim's initial number of kittens was 18 -/
theorem initial_kittens : 
  kittens_to_jessica + kittens_to_sara + kittens_left = 18 := by
  sorry

end initial_kittens_l2700_270039


namespace inequality_proof_l2700_270007

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b^3 / (a^2 + 8*b*c)) + (c^3 / (b^2 + 8*c*a)) + (a^3 / (c^2 + 8*a*b)) ≥ (1/9) * (a + b + c) := by
  sorry

end inequality_proof_l2700_270007


namespace fiona_pages_587_equal_reading_time_l2700_270083

/-- Represents the book reading scenario -/
structure BookReading where
  totalPages : ℕ
  fionaSpeed : ℕ  -- seconds per page
  davidSpeed : ℕ  -- seconds per page

/-- Calculates the number of pages Fiona should read for equal reading time -/
def fionaPages (br : BookReading) : ℕ :=
  (br.totalPages * br.davidSpeed) / (br.fionaSpeed + br.davidSpeed)

/-- Theorem stating that Fiona should read 587 pages -/
theorem fiona_pages_587 (br : BookReading) 
  (h1 : br.totalPages = 900)
  (h2 : br.fionaSpeed = 40)
  (h3 : br.davidSpeed = 75) : 
  fionaPages br = 587 := by
  sorry

/-- Theorem stating that Fiona and David spend equal time reading -/
theorem equal_reading_time (br : BookReading) 
  (h1 : br.totalPages = 900)
  (h2 : br.fionaSpeed = 40)
  (h3 : br.davidSpeed = 75) : 
  br.fionaSpeed * (fionaPages br) = br.davidSpeed * (br.totalPages - fionaPages br) := by
  sorry

end fiona_pages_587_equal_reading_time_l2700_270083


namespace equality_division_property_l2700_270096

theorem equality_division_property (a b c : ℝ) (h1 : a = b) (h2 : c ≠ 0) :
  a / (c^2) = b / (c^2) := by sorry

end equality_division_property_l2700_270096


namespace original_average_l2700_270077

theorem original_average (n : ℕ) (a : ℝ) (h1 : n = 10) (h2 : (n * a + n * 4) / n = 27) : a = 23 := by
  sorry

end original_average_l2700_270077


namespace abs_value_of_complex_l2700_270045

theorem abs_value_of_complex (z : ℂ) : z = 1 - 2 * Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end abs_value_of_complex_l2700_270045


namespace fourth_term_constant_implies_n_equals_5_l2700_270063

theorem fourth_term_constant_implies_n_equals_5 (n : ℕ) (x : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (Nat.choose n 3) * (-1/2)^3 * x^((n-5)/2) = k) →
  n = 5 :=
sorry

end fourth_term_constant_implies_n_equals_5_l2700_270063


namespace decagon_adjacent_probability_l2700_270023

/-- A decagon is a polygon with 10 vertices -/
def Decagon := {n : ℕ // n = 10}

/-- The number of ways to choose 2 distinct vertices from a decagon -/
def totalChoices (d : Decagon) : ℕ := (d.val.choose 2)

/-- The number of ways to choose 2 adjacent vertices from a decagon -/
def adjacentChoices (d : Decagon) : ℕ := 2 * d.val

/-- The probability of choosing two adjacent vertices in a decagon -/
def adjacentProbability (d : Decagon) : ℚ :=
  (adjacentChoices d : ℚ) / (totalChoices d : ℚ)

theorem decagon_adjacent_probability (d : Decagon) :
  adjacentProbability d = 4/9 := by sorry

end decagon_adjacent_probability_l2700_270023


namespace f_min_value_g_leq_f_range_l2700_270062

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|
def g (x : ℝ) : ℝ := |x - 3| + |x - 2|

-- Theorem for the minimum value of f
theorem f_min_value : ∃ (m : ℝ), m = 3 ∧ ∀ x, f x ≥ m := by sorry

-- Theorem for the range of a where g(a) ≤ f(x) for all x
theorem g_leq_f_range : ∀ a : ℝ, (∀ x : ℝ, g a ≤ f x) ↔ (1 ≤ a ∧ a ≤ 4) := by sorry

end f_min_value_g_leq_f_range_l2700_270062


namespace range_of_a_l2700_270000

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (2*x - 1) / (x - 1) ≤ 0 → x^2 - (2*a + 1)*x + a*(a + 1) < 0) ∧ 
  (∃ x : ℝ, x^2 - (2*a + 1)*x + a*(a + 1) < 0 ∧ ¬((2*x - 1) / (x - 1) ≤ 0)) →
  0 ≤ a ∧ a < 1/2 :=
sorry

end range_of_a_l2700_270000


namespace circle_tangency_and_intersection_l2700_270085

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def circle_O₂ (x y r : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = r^2

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y + 1 - 2 * Real.sqrt 2 = 0

-- Define the theorem
theorem circle_tangency_and_intersection :
  (∀ x y : ℝ, circle_O₁ x y → ¬circle_O₂ x y (Real.sqrt (12 - 8 * Real.sqrt 2))) →
  (∀ x y : ℝ, circle_O₁ x y → circle_O₂ x y (Real.sqrt (12 - 8 * Real.sqrt 2)) → tangent_line x y) ∧
  (∃ A B : ℝ × ℝ, 
    circle_O₁ A.1 A.2 ∧ circle_O₁ B.1 B.2 ∧
    circle_O₂ A.1 A.2 2 ∧ circle_O₂ B.1 B.2 2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8) →
  (∀ x y : ℝ, circle_O₂ x y 2 ∨ circle_O₂ x y (Real.sqrt 20)) :=
sorry

end circle_tangency_and_intersection_l2700_270085


namespace relationship_between_exponents_l2700_270073

theorem relationship_between_exponents 
  (a b c d : ℝ) 
  (x y q z : ℝ) 
  (h1 : a^(2*x) = c^(3*q)) 
  (h2 : a^(2*x) = b^2) 
  (h3 : c^(2*y) = a^(3*z)) 
  (h4 : c^(2*y) = d^2) 
  (h5 : a ≠ 0) 
  (h6 : b ≠ 0) 
  (h7 : c ≠ 0) 
  (h8 : d ≠ 0) : 
  9*q*z = 4*x*y := by
sorry

end relationship_between_exponents_l2700_270073


namespace library_shelves_needed_l2700_270099

theorem library_shelves_needed 
  (total_books : ℕ) 
  (sorted_books : ℕ) 
  (books_per_shelf : ℕ) 
  (h1 : total_books = 1500) 
  (h2 : sorted_books = 375) 
  (h3 : books_per_shelf = 45) : 
  (total_books - sorted_books) / books_per_shelf = 25 := by
  sorry

end library_shelves_needed_l2700_270099


namespace geometric_sequence_common_ratio_l2700_270094

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_sum : a 2 + (a 1 + a 2 + a 3) = 0) : 
  q = -1 := by
sorry

end geometric_sequence_common_ratio_l2700_270094


namespace min_value_implies_a_values_l2700_270078

theorem min_value_implies_a_values (a : ℝ) : 
  (∃ (m : ℝ), ∀ (x : ℝ), |x + 1| + |x + a| ≥ m ∧ (∃ (y : ℝ), |y + 1| + |y + a| = m) ∧ m = 1) →
  a = 0 ∨ a = 2 := by
sorry

end min_value_implies_a_values_l2700_270078


namespace quadratic_one_solution_sum_l2700_270069

theorem quadratic_one_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x, 3 * x^2 + b₁ * x + 6 * x + 4 = 0 → (b₁ + 6)^2 = 48) ∧ 
  (∀ x, 3 * x^2 + b₂ * x + 6 * x + 4 = 0 → (b₂ + 6)^2 = 48) → 
  b₁ + b₂ = -12 := by
sorry

end quadratic_one_solution_sum_l2700_270069


namespace expression_equals_ten_l2700_270071

theorem expression_equals_ten :
  let a : ℚ := 3
  let b : ℚ := 2
  let c : ℚ := 2
  (c * a^3 + c * b^3) / (a^2 - a*b + b^2) = 10 := by
  sorry

end expression_equals_ten_l2700_270071


namespace janice_bottle_caps_l2700_270092

/-- The number of boxes available to store bottle caps -/
def num_boxes : ℕ := 79

/-- The number of bottle caps that must be in each box -/
def caps_per_box : ℕ := 4

/-- The total number of bottle caps Janice has -/
def total_caps : ℕ := num_boxes * caps_per_box

theorem janice_bottle_caps : total_caps = 316 := by
  sorry

end janice_bottle_caps_l2700_270092


namespace unoccupied_volume_correct_l2700_270005

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of an ice cube -/
structure IceCubeDimensions where
  side : ℕ

/-- Calculates the unoccupied volume in a tank given its dimensions, water depth, and ice cubes -/
def unoccupiedVolume (tank : TankDimensions) (waterDepth : ℕ) (iceCube : IceCubeDimensions) (numIceCubes : ℕ) : ℕ :=
  let tankVolume := tank.length * tank.width * tank.height
  let waterVolume := tank.length * tank.width * waterDepth
  let iceCubeVolume := iceCube.side * iceCube.side * iceCube.side
  let totalIceVolume := numIceCubes * iceCubeVolume
  tankVolume - (waterVolume + totalIceVolume)

/-- Theorem stating the unoccupied volume in the tank under given conditions -/
theorem unoccupied_volume_correct :
  let tank : TankDimensions := ⟨12, 12, 15⟩
  let waterDepth : ℕ := 7
  let iceCube : IceCubeDimensions := ⟨3⟩
  let numIceCubes : ℕ := 15
  unoccupiedVolume tank waterDepth iceCube numIceCubes = 747 := by
  sorry

end unoccupied_volume_correct_l2700_270005


namespace defective_units_shipped_l2700_270055

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ)
  (h1 : defective_rate = 0.06)
  (h2 : shipped_rate = 0.04) :
  (defective_rate * shipped_rate * total_units) / total_units = 0.0024 := by
  sorry

end defective_units_shipped_l2700_270055


namespace largest_number_with_given_hcf_and_lcm_factors_l2700_270068

theorem largest_number_with_given_hcf_and_lcm_factors 
  (a b c : ℕ+) 
  (hcf_eq : Nat.gcd a b = 42 ∧ Nat.gcd (Nat.gcd a b) c = 42)
  (lcm_factors : ∃ (m : ℕ+), Nat.lcm (Nat.lcm a b) c = 42 * 10 * 20 * 25 * 30 * m) :
  max a (max b c) = 1260 := by
sorry

end largest_number_with_given_hcf_and_lcm_factors_l2700_270068


namespace arithmetic_sequence_third_term_l2700_270004

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 17th term is 12 and the 18th term is 15, the 3rd term is -30. -/
theorem arithmetic_sequence_third_term 
  (a : ℕ → ℤ) 
  (h_arithmetic : isArithmeticSequence a) 
  (h_17th : a 17 = 12) 
  (h_18th : a 18 = 15) : 
  a 3 = -30 := by
sorry

end arithmetic_sequence_third_term_l2700_270004


namespace parabola_axis_of_symmetry_l2700_270017

/-- The axis of symmetry of the parabola y = x^2 + 4x - 5 is the line x = -2 -/
theorem parabola_axis_of_symmetry :
  let f : ℝ → ℝ := fun x ↦ x^2 + 4*x - 5
  ∃ (a : ℝ), a = -2 ∧ ∀ (x y : ℝ), f (a + x) = f (a - x) := by
  sorry

end parabola_axis_of_symmetry_l2700_270017


namespace dragon_eventual_defeat_l2700_270035

/-- Represents the probabilities of head growth after each cut -/
structure HeadGrowthProbabilities where
  two_heads : ℝ
  one_head : ℝ
  no_heads : ℝ

/-- The probability of eventually defeating the dragon -/
def defeat_probability (probs : HeadGrowthProbabilities) : ℝ :=
  sorry

/-- The theorem stating that the dragon will eventually be defeated -/
theorem dragon_eventual_defeat (probs : HeadGrowthProbabilities) 
  (h1 : probs.two_heads = 1/4)
  (h2 : probs.one_head = 1/3)
  (h3 : probs.no_heads = 5/12)
  (h4 : probs.two_heads + probs.one_head + probs.no_heads = 1) :
  defeat_probability probs = 1 := by
  sorry

end dragon_eventual_defeat_l2700_270035


namespace least_common_remainder_least_common_remainder_achieved_least_common_remainder_is_126_l2700_270031

theorem least_common_remainder (n : ℕ) : n > 1 ∧ n % 25 = 1 ∧ n % 7 = 1 → n ≥ 126 := by
  sorry

theorem least_common_remainder_achieved : 126 % 25 = 1 ∧ 126 % 7 = 1 := by
  sorry

theorem least_common_remainder_is_126 : ∃ (n : ℕ), n = 126 ∧ n > 1 ∧ n % 25 = 1 ∧ n % 7 = 1 ∧ 
  ∀ (m : ℕ), m > 1 ∧ m % 25 = 1 ∧ m % 7 = 1 → m ≥ n := by
  sorry

end least_common_remainder_least_common_remainder_achieved_least_common_remainder_is_126_l2700_270031


namespace greatest_common_length_of_cords_l2700_270057

theorem greatest_common_length_of_cords :
  let cord_lengths : List ℝ := [Real.sqrt 20, Real.pi, Real.exp 1, Real.sqrt 98]
  ∀ x : ℝ, (∀ l ∈ cord_lengths, ∃ n : ℕ, l = x * n) → x ≤ 1 :=
by sorry

end greatest_common_length_of_cords_l2700_270057


namespace personalized_pencil_cost_l2700_270028

/-- The cost of personalized pencils with a discount for large orders -/
theorem personalized_pencil_cost 
  (base_cost : ℝ)  -- Cost for 100 pencils
  (base_quantity : ℕ)  -- Base quantity (100 pencils)
  (discount_threshold : ℕ)  -- Threshold for discount (1000 pencils)
  (discount_rate : ℝ)  -- Discount rate (5%)
  (order_quantity : ℕ)  -- Quantity ordered (2500 pencils)
  (h1 : base_cost = 30)
  (h2 : base_quantity = 100)
  (h3 : discount_threshold = 1000)
  (h4 : discount_rate = 0.05)
  (h5 : order_quantity = 2500) :
  let cost_per_pencil := base_cost / base_quantity
  let full_cost := cost_per_pencil * order_quantity
  let discounted_cost := full_cost * (1 - discount_rate)
  (if order_quantity > discount_threshold then discounted_cost else full_cost) = 712.5 := by
  sorry


end personalized_pencil_cost_l2700_270028


namespace unique_number_l2700_270044

def is_valid_digit (d : Nat) : Bool :=
  d ∈ [0, 1, 6, 8, 9]

def rotate_digit (d : Nat) : Nat :=
  match d with
  | 6 => 9
  | 9 => 6
  | _ => d

def rotate_number (n : Nat) : Nat :=
  let tens := n / 10
  let ones := n % 10
  10 * (rotate_digit ones) + (rotate_digit tens)

def satisfies_condition (n : Nat) : Bool :=
  n >= 10 ∧ n < 100 ∧
  is_valid_digit (n / 10) ∧
  is_valid_digit (n % 10) ∧
  n - (rotate_number n) = 75

theorem unique_number : ∃! n, satisfies_condition n :=
  sorry

end unique_number_l2700_270044


namespace chess_game_draw_probability_l2700_270001

theorem chess_game_draw_probability
  (p_a_not_lose : ℝ)
  (p_b_not_lose : ℝ)
  (h_a : p_a_not_lose = 0.8)
  (h_b : p_b_not_lose = 0.7)
  (h_game : ∀ (p_a_win p_draw : ℝ),
    p_a_win + p_draw = p_a_not_lose ∧
    (1 - p_a_win) = p_b_not_lose) :
  ∃ (p_draw : ℝ), p_draw = 0.5 := by
sorry

end chess_game_draw_probability_l2700_270001


namespace quentavious_nickels_l2700_270049

/-- Proves the number of nickels Quentavious left with -/
theorem quentavious_nickels (initial_nickels : ℕ) (gum_per_nickel : ℕ) (gum_received : ℕ) :
  initial_nickels = 5 →
  gum_per_nickel = 2 →
  gum_received = 6 →
  initial_nickels - (gum_received / gum_per_nickel) = 2 :=
by
  sorry

end quentavious_nickels_l2700_270049


namespace cos_two_x_value_l2700_270084

theorem cos_two_x_value (x : ℝ) 
  (h1 : x ∈ Set.Ioo (-3 * π / 4) (π / 4))
  (h2 : Real.cos (π / 4 - x) = -3 / 5) : 
  Real.cos (2 * x) = -24 / 25 := by
sorry

end cos_two_x_value_l2700_270084


namespace sum_of_squares_of_roots_l2700_270033

theorem sum_of_squares_of_roots : ∃ (r₁ r₂ r₃ r₄ : ℝ),
  (∀ x : ℝ, (x^2 + 4*x)^2 - 2016*(x^2 + 4*x) + 2017 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
  r₁^2 + r₂^2 + r₃^2 + r₄^2 = 4048 := by
  sorry

end sum_of_squares_of_roots_l2700_270033


namespace saras_quarters_l2700_270061

theorem saras_quarters (initial final given : ℕ) 
  (h1 : initial = 21)
  (h2 : final = 70)
  (h3 : given = final - initial) : 
  given = 49 := by sorry

end saras_quarters_l2700_270061


namespace fraction_increase_l2700_270015

theorem fraction_increase (x y : ℝ) (square : ℝ) :
  (2 * x * y) / (x + square) = (1 / 5) * (2 * (5 * x) * (5 * y)) / (5 * x + 5 * square) →
  square = 3 * y :=
by sorry

end fraction_increase_l2700_270015


namespace geometric_sequence_condition_l2700_270050

def S (n : ℕ) (m : ℝ) : ℝ := 3^(n+1) + m

def a (n : ℕ) (m : ℝ) : ℝ :=
  if n = 1 then S 1 m
  else S n m - S (n-1) m

theorem geometric_sequence_condition (m : ℝ) :
  (∀ n : ℕ, n ≥ 2 → (a (n+1) m) * (a (n-1) m) = (a n m)^2) ↔ m = -3 :=
sorry

end geometric_sequence_condition_l2700_270050


namespace triangle_problem_l2700_270016

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_problem (t : Triangle) 
  (h1 : (t.a + t.b) * (Real.sin t.A - Real.sin t.B) = (t.c - t.b) * Real.sin t.C)
  (h2 : 2 * t.c = 3 * t.b)
  (h3 : 1/2 * t.b * t.c * Real.sin t.A = 6 * Real.sqrt 3) :
  t.A = π/3 ∧ t.a = 2 * Real.sqrt (21/3) := by
  sorry

end triangle_problem_l2700_270016


namespace largest_coin_through_hole_l2700_270003

-- Define the diameter of coins
def diameter (coin : String) : ℝ :=
  match coin with
  | "1 kopeck" => 1
  | "20 kopeck" => 2
  | _ => 0

-- Define a circular hole
structure CircularHole where
  diameter : ℝ

-- Define a function to check if a coin can pass through a hole when paper is folded
def canPassThroughWhenFolded (coin : String) (hole : CircularHole) : Prop :=
  diameter coin ≤ 2 * hole.diameter

theorem largest_coin_through_hole :
  let hole : CircularHole := ⟨diameter "1 kopeck"⟩
  canPassThroughWhenFolded "20 kopeck" hole := by
  sorry

end largest_coin_through_hole_l2700_270003


namespace P_on_x_axis_AP_parallel_y_axis_l2700_270047

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of point P with coordinates (m+1, 2m-4) -/
def P (m : ℝ) : Point :=
  { x := m + 1, y := 2 * m - 4 }

/-- Point A with coordinates (-5, 2) -/
def A : Point :=
  { x := -5, y := 2 }

/-- Theorem: If P lies on the x-axis, then its coordinates are (3,0) -/
theorem P_on_x_axis (m : ℝ) : P m = { x := 3, y := 0 } ↔ (P m).y = 0 := by
  sorry

/-- Theorem: If AP is parallel to y-axis, then P's coordinates are (-5,-16) -/
theorem AP_parallel_y_axis (m : ℝ) : P m = { x := -5, y := -16 } ↔ (P m).x = A.x := by
  sorry

end P_on_x_axis_AP_parallel_y_axis_l2700_270047


namespace min_value_xy_plus_four_over_xy_l2700_270037

theorem min_value_xy_plus_four_over_xy (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) : 
  ∀ z w : ℝ, z > 0 → w > 0 → z + w = 2 → x * y + 4 / (x * y) ≤ z * w + 4 / (z * w) ∧ 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ a * b + 4 / (a * b) = 5 :=
sorry

end min_value_xy_plus_four_over_xy_l2700_270037


namespace parallelogram_area_32_22_l2700_270070

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 32 and height 22 is 704 -/
theorem parallelogram_area_32_22 : parallelogram_area 32 22 = 704 := by
  sorry

end parallelogram_area_32_22_l2700_270070


namespace bus_stop_time_l2700_270058

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : 
  speed_without_stops = 40 →
  speed_with_stops = 30 →
  (1 - speed_with_stops / speed_without_stops) * 60 = 15 :=
by
  sorry

#check bus_stop_time

end bus_stop_time_l2700_270058


namespace equilateral_triangles_in_cube_l2700_270065

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- An equilateral triangle is a triangle in which all three sides have the same length -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The number of equilateral triangles that can be formed with vertices of a cube -/
def num_equilateral_triangles_in_cube (c : Cube) : ℕ :=
  8

/-- Theorem: The number of equilateral triangles that can be formed with vertices of a cube is 8 -/
theorem equilateral_triangles_in_cube (c : Cube) :
  num_equilateral_triangles_in_cube c = 8 := by
  sorry

end equilateral_triangles_in_cube_l2700_270065


namespace box_paperclips_relation_small_box_medium_box_large_box_l2700_270018

/-- Represents the number of paperclips a box can hold based on its volume -/
noncomputable def paperclips (volume : ℝ) : ℝ :=
  50 * (volume / 16)

theorem box_paperclips_relation (v : ℝ) :
  paperclips v = 50 * (v / 16) :=
by sorry

theorem small_box : paperclips 16 = 50 :=
by sorry

theorem medium_box : paperclips 32 = 100 :=
by sorry

theorem large_box : paperclips 64 = 200 :=
by sorry

end box_paperclips_relation_small_box_medium_box_large_box_l2700_270018


namespace money_distribution_sum_l2700_270064

/-- Represents the share of money for each person --/
structure Share where
  amount : ℝ

/-- Represents the distribution of money among A, B, and C --/
structure Distribution where
  a : Share
  b : Share
  c : Share

/-- The conditions of the problem --/
def satisfiesConditions (d : Distribution) : Prop :=
  d.b.amount = 0.65 * d.a.amount ∧
  d.c.amount = 0.40 * d.a.amount ∧
  d.c.amount = 32

/-- The total sum of money --/
def totalSum (d : Distribution) : ℝ :=
  d.a.amount + d.b.amount + d.c.amount

/-- The theorem to prove --/
theorem money_distribution_sum :
  ∀ d : Distribution, satisfiesConditions d → totalSum d = 164 := by
  sorry

end money_distribution_sum_l2700_270064


namespace solve_halloween_decorations_l2700_270038

/-- Represents the Halloween decoration problem --/
def halloween_decorations 
  (skulls : ℕ) 
  (broomsticks : ℕ) 
  (spiderwebs : ℕ) 
  (cauldrons : ℕ) 
  (total_planned : ℕ) : Prop :=
  let pumpkins := 2 * spiderwebs
  let total_put_up := skulls + broomsticks + spiderwebs + pumpkins + cauldrons
  let left_to_put_up := total_planned - total_put_up
  left_to_put_up = 30

/-- Theorem stating the solution to the Halloween decoration problem --/
theorem solve_halloween_decorations : 
  halloween_decorations 12 4 12 1 83 :=
by
  sorry

#check solve_halloween_decorations

end solve_halloween_decorations_l2700_270038


namespace machine_job_completion_time_l2700_270075

theorem machine_job_completion_time : ∃ (x : ℝ), 
  x > 0 ∧
  (1 / (x + 4) + 1 / (x + 2) + 1 / ((x + 4 + x + 2) / 2) = 1 / x) ∧
  x = 1 := by
  sorry

end machine_job_completion_time_l2700_270075


namespace triangle_side_constraints_l2700_270051

theorem triangle_side_constraints (n : ℕ+) : 
  (2 * n + 10 < 3 * n + 5 ∧ 3 * n + 5 < n + 15) ∧
  (2 * n + 10 + (n + 15) > 3 * n + 5) ∧
  (2 * n + 10 + (3 * n + 5) > n + 15) ∧
  (n + 15 + (3 * n + 5) > 2 * n + 10) ↔
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 :=
by sorry

end triangle_side_constraints_l2700_270051


namespace total_votes_l2700_270014

theorem total_votes (veggies_votes : ℕ) (meat_votes : ℕ) 
  (h1 : veggies_votes = 337) (h2 : meat_votes = 335) : 
  veggies_votes + meat_votes = 672 := by
  sorry

end total_votes_l2700_270014


namespace smallest_number_with_conditions_l2700_270009

def is_multiple_of_29 (n : ℕ) : Prop := ∃ k : ℕ, n = 29 * k

def last_two_digits_are_29 (n : ℕ) : Prop := n % 100 = 29

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem smallest_number_with_conditions : 
  (is_multiple_of_29 51729) ∧ 
  (last_two_digits_are_29 51729) ∧ 
  (sum_of_digits 51729 = 29) ∧
  (∀ m : ℕ, m < 51729 → 
    ¬(is_multiple_of_29 m ∧ last_two_digits_are_29 m ∧ sum_of_digits m = 29)) := by
  sorry

end smallest_number_with_conditions_l2700_270009


namespace beam_travel_time_l2700_270080

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length 4 -/
structure Square where
  A : Point := ⟨0, 0⟩
  B : Point := ⟨4, 0⟩
  C : Point := ⟨4, 4⟩
  D : Point := ⟨0, 4⟩

/-- The beam's path in the square -/
structure BeamPath (s : Square) where
  F : Point
  E : Point
  BE : ℝ
  EF : ℝ
  FC : ℝ
  speed : ℝ

/-- Theorem stating the time taken for the beam to travel from F to E -/
theorem beam_travel_time (s : Square) (path : BeamPath s) 
  (h1 : path.BE = 2)
  (h2 : path.EF = 2)
  (h3 : path.FC = 2)
  (h4 : path.speed = 1)
  (h5 : path.E = ⟨2, 0⟩) :
  ∃ t : ℝ, t = 2 * Real.sqrt 61 ∧ 
    t * path.speed = Real.sqrt ((10 - path.F.x)^2 + (6 - path.F.y)^2) := by
  sorry

end beam_travel_time_l2700_270080


namespace first_day_is_friday_l2700_270040

/-- Days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after n days -/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (dayAfter d m)

/-- Theorem: If the 25th day of a month is a Monday, then the 1st day of that month is a Friday -/
theorem first_day_is_friday (d : DayOfWeek) : 
  dayAfter d 24 = DayOfWeek.Monday → d = DayOfWeek.Friday :=
by
  sorry


end first_day_is_friday_l2700_270040


namespace rest_of_body_length_l2700_270097

theorem rest_of_body_length
  (total_height : ℝ)
  (leg_ratio : ℝ)
  (head_ratio : ℝ)
  (h1 : total_height = 60)
  (h2 : leg_ratio = 1 / 3)
  (h3 : head_ratio = 1 / 4)
  : total_height - (leg_ratio * total_height + head_ratio * total_height) = 25 := by
  sorry

end rest_of_body_length_l2700_270097


namespace log_inequality_implies_upper_bound_l2700_270056

theorem log_inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, a < Real.log (|x - 3| + |x + 7|)) → a < 1 := by
  sorry

end log_inequality_implies_upper_bound_l2700_270056


namespace shelves_needed_l2700_270006

/-- Given a total of 14 books, with 2 taken by a librarian and 3 books fitting on each shelf,
    prove that 4 shelves are needed to store the remaining books. -/
theorem shelves_needed (total_books : ℕ) (taken_books : ℕ) (books_per_shelf : ℕ) :
  total_books = 14 →
  taken_books = 2 →
  books_per_shelf = 3 →
  ((total_books - taken_books) / books_per_shelf : ℕ) = 4 := by
  sorry

#check shelves_needed

end shelves_needed_l2700_270006


namespace circle_area_through_points_l2700_270090

/-- The area of a circle with center P(-3, 4) passing through Q(9, -3) is 193π square units. -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (-3, 4)
  let Q : ℝ × ℝ := (9, -3)
  let r : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (π * r^2) = 193 * π := by sorry

end circle_area_through_points_l2700_270090


namespace merry_go_round_time_l2700_270074

theorem merry_go_round_time (dave_time chuck_time erica_time : ℝ) : 
  dave_time = 10 →
  chuck_time = 5 * dave_time →
  erica_time = chuck_time * 1.3 →
  erica_time = 65 := by
sorry

end merry_go_round_time_l2700_270074


namespace min_value_of_f_l2700_270012

/-- The quadratic function f(x) = 3x^2 + 8x + 15 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 8 * x + 15

/-- The minimum value of f(x) is 29/3 -/
theorem min_value_of_f : 
  ∀ x : ℝ, f x ≥ 29/3 ∧ ∃ x₀ : ℝ, f x₀ = 29/3 :=
sorry

end min_value_of_f_l2700_270012


namespace strawberry_weight_theorem_l2700_270060

/-- The total weight of Marco's and his dad's strawberries -/
def total_weight (marco_weight : ℕ) (weight_difference : ℕ) : ℕ :=
  marco_weight + (marco_weight - weight_difference)

/-- Theorem: The total weight of strawberries is 47 pounds -/
theorem strawberry_weight_theorem (marco_weight : ℕ) (weight_difference : ℕ) 
  (h1 : marco_weight = 30)
  (h2 : weight_difference = 13) :
  total_weight marco_weight weight_difference = 47 := by
  sorry

#eval total_weight 30 13

end strawberry_weight_theorem_l2700_270060


namespace parallel_planes_line_sufficient_not_necessary_l2700_270002

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relations
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (line_in_plane : Line → Plane → Prop)

theorem parallel_planes_line_sufficient_not_necessary 
  (α β : Plane) (l : Line) 
  (h_distinct : α ≠ β) 
  (h_l_in_α : line_in_plane l α) :
  (∀ α β l, plane_parallel α β → line_parallel_plane l β) ∧ 
  (∃ α β l, line_parallel_plane l β ∧ ¬plane_parallel α β) := by
  sorry


end parallel_planes_line_sufficient_not_necessary_l2700_270002


namespace common_tangent_lines_l2700_270053

-- Define the circles
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle_E (x y : ℝ) : Prop := x^2 + (y - Real.sqrt 3)^2 = 1

-- Define the potential tangent lines
def line1 (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 1 = 0
def line2 (x y : ℝ) : Prop := Real.sqrt 3 * x + y - Real.sqrt 3 - 2 = 0
def line3 (x y : ℝ) : Prop := Real.sqrt 3 * x + y - Real.sqrt 3 + 2 = 0

-- Define what it means for a line to be tangent to a circle
def is_tangent_to (line : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), line x y ∧ circle x y ∧
  ∀ (x' y' : ℝ), line x' y' → circle x' y' → (x' = x ∧ y' = y)

-- State the theorem
theorem common_tangent_lines :
  (is_tangent_to line1 circle_C ∧ is_tangent_to line1 circle_E) ∧
  (is_tangent_to line2 circle_C ∧ is_tangent_to line2 circle_E) ∧
  (is_tangent_to line3 circle_C ∧ is_tangent_to line3 circle_E) :=
sorry

end common_tangent_lines_l2700_270053


namespace area_perimeter_relation_l2700_270032

/-- A stepped shape with n rows -/
structure SteppedShape (n : ℕ) where
  (n_pos : n > 0)
  (bottom_row : ℕ)
  (bottom_odd : Odd bottom_row)
  (bottom_eq : bottom_row = 2 * n - 1)
  (top_row : ℕ)
  (top_eq : top_row = 1)

/-- The area of a stepped shape -/
def area (shape : SteppedShape n) : ℕ := n ^ 2

/-- The perimeter of a stepped shape -/
def perimeter (shape : SteppedShape n) : ℕ := 6 * n - 2

/-- The main theorem relating area and perimeter of a stepped shape -/
theorem area_perimeter_relation (shape : SteppedShape n) :
  36 * (area shape) = (perimeter shape + 2) ^ 2 := by sorry

end area_perimeter_relation_l2700_270032


namespace cheaper_fluid_cost_l2700_270008

/-- Represents the cost of cleaning fluids and drum quantities -/
structure CleaningSupplies where
  total_drums : ℕ
  expensive_drums : ℕ
  cheap_drums : ℕ
  expensive_cost : ℚ
  total_cost : ℚ

/-- Theorem stating that given the conditions, the cheaper fluid costs $20 per drum -/
theorem cheaper_fluid_cost (supplies : CleaningSupplies)
  (h1 : supplies.total_drums = 7)
  (h2 : supplies.expensive_drums + supplies.cheap_drums = supplies.total_drums)
  (h3 : supplies.expensive_cost = 30)
  (h4 : supplies.total_cost = 160)
  (h5 : supplies.cheap_drums = 5) :
  (supplies.total_cost - supplies.expensive_cost * supplies.expensive_drums) / supplies.cheap_drums = 20 :=
by sorry

end cheaper_fluid_cost_l2700_270008


namespace ratio_of_squares_to_difference_l2700_270013

theorem ratio_of_squares_to_difference (a b : ℝ) : 
  0 < b → 0 < a → a > b → (a^2 + b^2 = 7 * (a - b)) → (a / b = Real.sqrt 6) := by
  sorry

end ratio_of_squares_to_difference_l2700_270013
