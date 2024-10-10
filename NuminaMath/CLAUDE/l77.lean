import Mathlib

namespace curve_symmetry_l77_7710

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x + y = 0

-- Theorem: The curve is symmetrical about the line x + y = 0
theorem curve_symmetry :
  ∀ (x y : ℝ), curve x y → 
  ∃ (x' y' : ℝ), curve x' y' ∧ symmetry_line ((x + x')/2) ((y + y')/2) :=
sorry

end curve_symmetry_l77_7710


namespace village_burn_time_l77_7789

/-- Represents the number of cottages remaining after n intervals -/
def A : ℕ → ℕ
| 0 => 90
| n + 1 => 2 * A n - 96

/-- The time it takes Trodgor to burn down the village -/
def burnTime : ℕ := 1920

theorem village_burn_time : 
  ∀ n : ℕ, A n = 0 → n * 480 = burnTime := by
  sorry

#check village_burn_time

end village_burn_time_l77_7789


namespace quadratic_inequality_condition_l77_7726

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - x + 1 > 0) ↔ a > (1 / 4) :=
sorry

end quadratic_inequality_condition_l77_7726


namespace crayon_count_l77_7767

theorem crayon_count (small_left medium_left large_left : ℕ) 
  (h_small : small_left = 60)
  (h_medium : medium_left = 98)
  (h_large : large_left = 168) :
  ∃ (small_initial medium_initial large_initial : ℕ),
    small_initial = 100 ∧
    medium_initial = 392 ∧
    large_initial = 294 ∧
    small_left = (3 : ℚ) / 5 * small_initial ∧
    medium_left = (1 : ℚ) / 4 * medium_initial ∧
    large_left = (4 : ℚ) / 7 * large_initial ∧
    (2 : ℚ) / 5 * small_initial + 
    (3 : ℚ) / 4 * medium_initial + 
    (3 : ℚ) / 7 * large_initial = 460 := by
  sorry


end crayon_count_l77_7767


namespace student_group_allocation_schemes_l77_7723

theorem student_group_allocation_schemes (n : ℕ) (k : ℕ) (m : ℕ) 
  (h1 : n = 12) 
  (h2 : k = 4) 
  (h3 : m = 3) 
  (h4 : n = k * m) : 
  (Nat.choose n m * Nat.choose (n - m) m * Nat.choose (n - 2*m) m * m^k : ℕ) = 
  (Nat.choose 12 3 * Nat.choose 9 3 * Nat.choose 6 3 * 3^4 : ℕ) := by
  sorry

end student_group_allocation_schemes_l77_7723


namespace sunset_colors_l77_7739

/-- The duration of the sunset in hours -/
def sunset_duration : ℕ := 2

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The interval between color changes in minutes -/
def color_change_interval : ℕ := 10

/-- The number of colors the sky turns during the sunset -/
def number_of_colors : ℕ := sunset_duration * minutes_per_hour / color_change_interval

theorem sunset_colors :
  number_of_colors = 12 := by
  sorry

end sunset_colors_l77_7739


namespace average_speed_approx_202_l77_7796

/-- Calculates the average speed given initial and final odometer readings and total driving time -/
def average_speed (initial_reading final_reading : ℕ) (total_time : ℚ) : ℚ :=
  (final_reading - initial_reading : ℚ) / total_time

theorem average_speed_approx_202 (initial_reading final_reading : ℕ) (total_time : ℚ) :
  initial_reading = 12321 →
  final_reading = 14741 →
  total_time = 12 →
  ∃ ε > 0, |average_speed initial_reading final_reading total_time - 202| < ε :=
by
  sorry

#eval average_speed 12321 14741 12

end average_speed_approx_202_l77_7796


namespace translation_result_l77_7754

def translate_point (x y dx dy : Int) : (Int × Int) :=
  (x + dx, y - dy)

theorem translation_result :
  let initial_point := (-2, 3)
  let x_translation := 3
  let y_translation := 2
  translate_point initial_point.1 initial_point.2 x_translation y_translation = (1, 1) := by
  sorry

end translation_result_l77_7754


namespace exist_numbers_same_divisors_less_sum_l77_7730

/-- The number of natural divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The sum of all natural divisors of a natural number -/
def sum_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

/-- There exist two natural numbers with the same number of divisors,
    where one is greater than the other, but has a smaller sum of divisors -/
theorem exist_numbers_same_divisors_less_sum :
  ∃ x y : ℕ, x > y ∧ num_divisors x = num_divisors y ∧ sum_divisors x < sum_divisors y :=
sorry

end exist_numbers_same_divisors_less_sum_l77_7730


namespace maria_carrots_l77_7705

def carrot_problem (initial_carrots thrown_out_carrots picked_next_day : ℕ) : Prop :=
  initial_carrots - thrown_out_carrots + picked_next_day = 52

theorem maria_carrots : carrot_problem 48 11 15 := by
  sorry

end maria_carrots_l77_7705


namespace magnitude_of_AB_l77_7743

def vector_AB : ℝ × ℝ := (1, 1)

theorem magnitude_of_AB : Real.sqrt ((vector_AB.1)^2 + (vector_AB.2)^2) = Real.sqrt 2 := by
  sorry

end magnitude_of_AB_l77_7743


namespace square_of_T_number_is_T_number_l77_7775

/-- Definition of a T number -/
def is_T_number (x : ℤ) : Prop := ∃ (a b : ℤ), x = a^2 + a*b + b^2

/-- Theorem: The square of a T number is still a T number -/
theorem square_of_T_number_is_T_number (x : ℤ) (h : is_T_number x) : is_T_number (x^2) := by
  sorry

end square_of_T_number_is_T_number_l77_7775


namespace final_total_cost_is_correct_l77_7732

def spiral_notebook_price : ℝ := 15
def personal_planner_price : ℝ := 10
def spiral_notebook_discount_threshold : ℕ := 5
def personal_planner_discount_threshold : ℕ := 10
def spiral_notebook_discount_rate : ℝ := 0.2
def personal_planner_discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.07
def spiral_notebooks_bought : ℕ := 6
def personal_planners_bought : ℕ := 12

def calculate_discounted_price (price : ℝ) (quantity : ℕ) (discount_rate : ℝ) : ℝ :=
  price * quantity * (1 - discount_rate)

def calculate_total_cost : ℝ :=
  let spiral_notebook_cost := 
    calculate_discounted_price spiral_notebook_price spiral_notebooks_bought spiral_notebook_discount_rate
  let personal_planner_cost := 
    calculate_discounted_price personal_planner_price personal_planners_bought personal_planner_discount_rate
  let subtotal := spiral_notebook_cost + personal_planner_cost
  subtotal * (1 + sales_tax_rate)

theorem final_total_cost_is_correct : calculate_total_cost = 186.18 := by sorry

end final_total_cost_is_correct_l77_7732


namespace percentage_problem_l77_7737

theorem percentage_problem (p : ℝ) : p = 80 :=
  by
  -- Define the number as 15
  let number : ℝ := 15
  
  -- Define the condition: 40% of 15 is greater than p% of 5 by 2
  have h : 0.4 * number = p / 100 * 5 + 2 := by sorry
  
  -- Proof goes here
  sorry

end percentage_problem_l77_7737


namespace opposite_numbers_not_just_opposite_signs_l77_7750

theorem opposite_numbers_not_just_opposite_signs : ¬ (∀ a b : ℝ, (a > 0 ∧ b < 0) → (a = -b)) := by
  sorry

end opposite_numbers_not_just_opposite_signs_l77_7750


namespace centroid_property_l77_7787

/-- The centroid of a triangle divides each median in the ratio 2:1 and creates three equal-area subtriangles. -/
structure Centroid (xA yA xB yB xC yC : ℚ) where
  x : ℚ
  y : ℚ
  is_centroid : x = (xA + xB + xC) / 3 ∧ y = (yA + yB + yC) / 3

/-- Given a triangle ABC with vertices A(5,8), B(3,-2), and C(6,1),
    if D(m,n) is the centroid of the triangle, then 10m + n = 49. -/
theorem centroid_property :
  let d : Centroid 5 8 3 (-2) 6 1 := ⟨(14/3), (7/3), by sorry⟩
  10 * d.x + d.y = 49 := by sorry

end centroid_property_l77_7787


namespace nancy_count_l77_7716

theorem nancy_count (a b c d e f : ℕ) (h_mean : (a + b + c + d + e + f) / 6 = 7)
  (h_a : a = 6) (h_b : b = 12) (h_c : c = 1) (h_d : d = 12) (h_f : f = 8) :
  e = 3 := by
  sorry

end nancy_count_l77_7716


namespace inscribed_rectangle_circle_circumference_l77_7727

theorem inscribed_rectangle_circle_circumference 
  (width : Real) (height : Real) (circle : Real → Prop) 
  (rectangle : Real → Real → Prop) (circumference : Real) :
  width = 9 →
  height = 12 →
  rectangle width height →
  (∀ x y, rectangle x y → circle (Real.sqrt (x^2 + y^2))) →
  circumference = Real.pi * Real.sqrt (width^2 + height^2) →
  circumference = 15 * Real.pi := by
  sorry

end inscribed_rectangle_circle_circumference_l77_7727


namespace arithmetic_geometric_sequence_l77_7741

/-- An arithmetic sequence with first term 18 and non-zero common difference d -/
def arithmeticSequence (d : ℝ) (n : ℕ) : ℝ := 18 + (n - 1 : ℝ) * d

theorem arithmetic_geometric_sequence (d : ℝ) (h1 : d ≠ 0) :
  (arithmeticSequence d 4) ^ 2 = (arithmeticSequence d 1) * (arithmeticSequence d 8) →
  d = 2 := by
  sorry

end arithmetic_geometric_sequence_l77_7741


namespace system_of_equations_solution_l77_7721

theorem system_of_equations_solution (x y : ℚ) 
  (eq1 : 2 * x + y = 7) 
  (eq2 : x + 2 * y = 8) : 
  (x + y) / 3 = 5 / 3 := by
sorry

end system_of_equations_solution_l77_7721


namespace complex_power_eight_l77_7773

theorem complex_power_eight : (3 * Complex.cos (π / 4) - 3 * Complex.I * Complex.sin (π / 4)) ^ 8 = 6552 := by
  sorry

end complex_power_eight_l77_7773


namespace opposite_leg_length_l77_7742

/-- Represents a right triangle with a 30° angle -/
structure RightTriangle30 where
  /-- Length of the hypotenuse -/
  hypotenuse : ℝ
  /-- Length of the leg opposite to the 30° angle -/
  opposite_leg : ℝ
  /-- Constraint that the hypotenuse is twice the opposite leg -/
  hyp_constraint : hypotenuse = 2 * opposite_leg

/-- 
Theorem: In a right triangle with a 30° angle and hypotenuse of 18 inches, 
the leg opposite to the 30° angle is 9 inches long.
-/
theorem opposite_leg_length (triangle : RightTriangle30) 
  (h : triangle.hypotenuse = 18) : triangle.opposite_leg = 9 := by
  sorry

end opposite_leg_length_l77_7742


namespace picnic_adult_child_difference_l77_7740

/-- A picnic scenario with men, women, adults, and children -/
structure Picnic where
  total : ℕ
  men : ℕ
  women : ℕ
  adults : ℕ
  children : ℕ

/-- Conditions for the picnic scenario -/
def PicnicConditions (p : Picnic) : Prop :=
  p.total = 240 ∧
  p.men = p.women + 40 ∧
  p.men = 90 ∧
  p.adults > p.children ∧
  p.total = p.men + p.women + p.children ∧
  p.adults = p.men + p.women

/-- Theorem stating the difference between adults and children -/
theorem picnic_adult_child_difference (p : Picnic) 
  (h : PicnicConditions p) : p.adults - p.children = 40 := by
  sorry

#check picnic_adult_child_difference

end picnic_adult_child_difference_l77_7740


namespace unique_solution_of_equation_l77_7766

theorem unique_solution_of_equation :
  ∃! (x y z : ℝ), x^2 + 5*y^2 + 5*z^2 - 4*x*z - 2*y - 4*y*z + 1 = 0 ∧ x = 4 ∧ y = 1 ∧ z = 2 := by
  sorry

end unique_solution_of_equation_l77_7766


namespace angle_c_measure_l77_7788

theorem angle_c_measure (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  A + B = 80 →       -- Given condition
  C = 100            -- Conclusion to prove
:= by sorry

end angle_c_measure_l77_7788


namespace fence_birds_count_l77_7749

/-- The number of birds on a fence after new birds land -/
def total_birds (initial_pairs : ℕ) (birds_per_pair : ℕ) (new_birds : ℕ) : ℕ :=
  initial_pairs * birds_per_pair + new_birds

/-- Theorem stating the total number of birds on the fence -/
theorem fence_birds_count :
  let initial_pairs : ℕ := 12
  let birds_per_pair : ℕ := 2
  let new_birds : ℕ := 8
  total_birds initial_pairs birds_per_pair new_birds = 32 := by
  sorry

end fence_birds_count_l77_7749


namespace apollo_chariot_wheels_cost_l77_7751

/-- Represents the cost in golden apples for chariot wheels over a year -/
structure ChariotWheelsCost where
  total : ℕ  -- Total cost for the year
  second_half_multiplier : ℕ  -- Multiplier for the second half of the year

/-- 
Calculates the cost for the first half of the year given the total cost
and the multiplier for the second half of the year.
-/
def first_half_cost (c : ChariotWheelsCost) : ℕ :=
  c.total / (1 + c.second_half_multiplier)

/-- 
Theorem: If the total cost for a year is 54 golden apples, and the cost for the 
second half of the year is double the cost for the first half, then the cost 
for the first half of the year is 18 golden apples.
-/
theorem apollo_chariot_wheels_cost : 
  let c := ChariotWheelsCost.mk 54 2
  first_half_cost c = 18 := by
  sorry

end apollo_chariot_wheels_cost_l77_7751


namespace existence_of_irreducible_fractions_l77_7747

theorem existence_of_irreducible_fractions : ∃ (a b : ℕ), 
  (Nat.gcd a b = 1) ∧ 
  (Nat.gcd (a + 1) b = 1) ∧ 
  (Nat.gcd (a + 1) (b + 1) = 1) := by
  sorry

end existence_of_irreducible_fractions_l77_7747


namespace distinct_choices_eq_eight_l77_7755

/-- Represents the set of marbles Tom has -/
inductive Marble : Type
| Red : Marble
| Green : Marble
| Blue : Marble
| Yellow : Marble

/-- The number of each type of marble Tom has -/
def marbleCounts : Marble → ℕ
| Marble.Red => 1
| Marble.Green => 1
| Marble.Blue => 1
| Marble.Yellow => 4

/-- The total number of marbles Tom has -/
def totalMarbles : ℕ := (marbleCounts Marble.Red) + (marbleCounts Marble.Green) + 
                        (marbleCounts Marble.Blue) + (marbleCounts Marble.Yellow)

/-- A function to calculate the number of distinct ways to choose 3 marbles -/
def distinctChoices : ℕ := sorry

/-- Theorem stating that the number of distinct ways to choose 3 marbles is 8 -/
theorem distinct_choices_eq_eight : distinctChoices = 8 := by sorry

end distinct_choices_eq_eight_l77_7755


namespace union_A_B_when_a_neg_four_complement_A_intersect_B_eq_B_l77_7772

-- Define the sets A and B
def A : Set ℝ := {x | (1 - 2*x) / (x - 3) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a ≤ 0}

-- Theorem 1: Union of A and B when a = -4
theorem union_A_B_when_a_neg_four :
  A ∪ B (-4) = {x : ℝ | -2 ≤ x ∧ x < 3} := by sorry

-- Theorem 2: Condition for (CᵣA) ∩ B = B
theorem complement_A_intersect_B_eq_B (a : ℝ) :
  (Aᶜ ∩ B a = B a) ↔ a > -1/4 := by sorry

end union_A_B_when_a_neg_four_complement_A_intersect_B_eq_B_l77_7772


namespace max_c_value_l77_7719

theorem max_c_value (c d : ℝ) (h : 5 * c + (d - 12)^2 = 235) :
  c ≤ 47 ∧ ∃ d', 5 * 47 + (d' - 12)^2 = 235 := by
  sorry

end max_c_value_l77_7719


namespace same_remainder_divisor_l77_7780

theorem same_remainder_divisor : ∃ (N : ℕ), N > 1 ∧ 
  N = 23 ∧ 
  (∀ (k : ℕ), k > N → ¬(1743 % k = 2019 % k ∧ 2019 % k = 3008 % k)) ∧
  (1743 % N = 2019 % N ∧ 2019 % N = 3008 % N) :=
by sorry

end same_remainder_divisor_l77_7780


namespace average_speed_two_hours_car_average_speed_l77_7714

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 > 0 → speed2 > 0 → (speed1 + speed2) / 2 = (speed1 * 1 + speed2 * 1) / (1 + 1) := by
  sorry

/-- The average speed of a car traveling 90 km in the first hour and 60 km in the second hour is 75 km/h -/
theorem car_average_speed : 
  let speed1 := 90
  let speed2 := 60
  (speed1 + speed2) / 2 = 75 := by
  sorry

end average_speed_two_hours_car_average_speed_l77_7714


namespace tan_theta_minus_pi_fourth_l77_7713

theorem tan_theta_minus_pi_fourth (θ : Real) :
  (-π/2 < θ) → (θ < 0) → -- θ is in the fourth quadrant
  (Real.sin (θ + π/4) = 3/5) →
  (Real.tan (θ - π/4) = -4/3) := by
  sorry

end tan_theta_minus_pi_fourth_l77_7713


namespace inequality_proof_l77_7784

theorem inequality_proof (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) :
  (x^2 / (x - 1)^2) + (y^2 / (y - 1)^2) + (z^2 / (z - 1)^2) ≥ 1 := by
  sorry

end inequality_proof_l77_7784


namespace g_divisibility_l77_7725

def g : ℕ → ℕ
  | 0 => 1
  | n + 1 => g n ^ 2 + g n + 1

theorem g_divisibility (n : ℕ) : 
  (g n ^ 2 + 1) ∣ (g (n + 1) ^ 2 + 1) := by
  sorry

end g_divisibility_l77_7725


namespace remainder_of_1731_base12_div_9_l77_7731

/-- Converts a base-12 number to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base-12 representation of the number --/
def base12Number : List Nat := [1, 7, 3, 1]

theorem remainder_of_1731_base12_div_9 :
  (base12ToDecimal base12Number) % 9 = 1 := by
  sorry

end remainder_of_1731_base12_div_9_l77_7731


namespace calculation_proof_l77_7745

theorem calculation_proof : (4 + 6 + 10) / 3 - 2 / 3 = 6 := by
  sorry

end calculation_proof_l77_7745


namespace bobs_family_adults_l77_7700

theorem bobs_family_adults (total_apples : ℕ) (num_children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) 
  (h1 : total_apples = 450)
  (h2 : num_children = 33)
  (h3 : apples_per_child = 10)
  (h4 : apples_per_adult = 3) :
  (total_apples - num_children * apples_per_child) / apples_per_adult = 40 := by
  sorry

end bobs_family_adults_l77_7700


namespace train_crossing_time_l77_7795

/-- The time taken for a train to cross a telegraph post -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 320 ∧ train_speed_kmh = 72 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 16 := by
  sorry

#check train_crossing_time

end train_crossing_time_l77_7795


namespace rush_order_cost_rush_order_cost_is_five_l77_7753

/-- Calculate the extra amount paid for a rush order given the following conditions:
  * There are 4 people ordering dinner
  * Each main meal costs $12.0
  * 2 appetizers are ordered at $6.00 each
  * A 20% tip is included
  * The total amount spent is $77
-/
theorem rush_order_cost (num_people : ℕ) (main_meal_cost : ℚ) (num_appetizers : ℕ) 
  (appetizer_cost : ℚ) (tip_percentage : ℚ) (total_spent : ℚ) : ℚ :=
  let subtotal := num_people * main_meal_cost + num_appetizers * appetizer_cost
  let tip := subtotal * tip_percentage
  let total_before_rush := subtotal + tip
  total_spent - total_before_rush

/-- The extra amount paid for the rush order is $5.0 -/
theorem rush_order_cost_is_five : 
  rush_order_cost 4 12 2 6 (1/5) 77 = 5 := by
  sorry

end rush_order_cost_rush_order_cost_is_five_l77_7753


namespace range_of_a_l77_7765

theorem range_of_a (a : ℝ) : 
  (∀ x > a, 2 * x + 2 / (x - a) ≥ 5) → a ≥ (1 : ℝ) / 2 := by
  sorry

end range_of_a_l77_7765


namespace win_sector_area_l77_7799

/-- Given a circular spinner with radius 10 cm and a probability of winning 2/5,
    the area of the WIN sector is 40π square centimeters. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (A_win : ℝ) :
  r = 10 →
  p = 2 / 5 →
  A_win = p * π * r^2 →
  A_win = 40 * π :=
by sorry

end win_sector_area_l77_7799


namespace x_value_in_terms_of_acd_l77_7781

theorem x_value_in_terms_of_acd (x y z a b c d : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (ha : a ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : x * y / (x + y) = a)
  (h2 : x * z / (x + z) = b)
  (h3 : y * z / (y + z) = c)
  (h4 : y * z / (y - z) = d) :
  x = 2 * a * c / (a - c - d) := by
sorry

end x_value_in_terms_of_acd_l77_7781


namespace sin_2x_value_l77_7791

theorem sin_2x_value (x : Real) (h : (1 + Real.tan x) / (1 - Real.tan x) = 2) : 
  Real.sin (2 * x) = 3/5 := by
sorry

end sin_2x_value_l77_7791


namespace complex_number_simplification_l77_7761

/-- Given a complex number z = (-1-2i) / (1+i)^2, prove that z = -1 + (1/2)i -/
theorem complex_number_simplification :
  let z : ℂ := (-1 - 2*I) / (1 + I)^2
  z = -1 + (1/2)*I :=
by sorry

end complex_number_simplification_l77_7761


namespace present_expenditure_l77_7734

theorem present_expenditure (P : ℝ) : 
  P * (1 + 0.1)^2 = 24200.000000000004 → P = 20000 := by
  sorry

end present_expenditure_l77_7734


namespace boys_to_girls_ratio_l77_7774

theorem boys_to_girls_ratio (B G : ℕ) (h_positive : B > 0 ∧ G > 0) : 
  (1/3 : ℚ) * B + (2/3 : ℚ) * G = (192/360 : ℚ) * (B + G) → 
  (B : ℚ) / G = 2/3 := by
  sorry

end boys_to_girls_ratio_l77_7774


namespace calculate_running_speed_l77_7769

/-- Given a swimming speed and an average speed for swimming and running,
    calculate the running speed. -/
theorem calculate_running_speed
  (swimming_speed : ℝ)
  (average_speed : ℝ)
  (h1 : swimming_speed = 1)
  (h2 : average_speed = 4.5)
  : (2 * average_speed - swimming_speed) = 8 := by
  sorry

#check calculate_running_speed

end calculate_running_speed_l77_7769


namespace planes_through_three_points_l77_7764

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define collinearity for three points
def collinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ (t : ℝ), (p3.x - p1.x) = t * (p2.x - p1.x) ∧
             (p3.y - p1.y) = t * (p2.y - p1.y) ∧
             (p3.z - p1.z) = t * (p2.z - p1.z)

-- Define a function to count the number of planes through three points
def count_planes (p1 p2 p3 : Point3D) : Nat ⊕ Nat → Prop
  | Sum.inl 1 => ¬collinear p1 p2 p3
  | Sum.inr 0 => collinear p1 p2 p3
  | _ => False

-- Theorem statement
theorem planes_through_three_points (p1 p2 p3 : Point3D) :
  (count_planes p1 p2 p3 (Sum.inl 1)) ∨ (count_planes p1 p2 p3 (Sum.inr 0)) :=
sorry

end planes_through_three_points_l77_7764


namespace power_two_1000_mod_13_l77_7794

theorem power_two_1000_mod_13 : 2^1000 % 13 = 3 := by
  sorry

end power_two_1000_mod_13_l77_7794


namespace number_solution_l77_7729

theorem number_solution (x : ℝ) : 0.6 * x = 0.3 * 10 + 27 → x = 50 := by
  sorry

end number_solution_l77_7729


namespace exponential_equation_implication_l77_7786

theorem exponential_equation_implication (x : ℝ) : 
  4 * (3 : ℝ)^x = 2187 → (x + 2) * (x - 2) = 21 := by
  sorry

end exponential_equation_implication_l77_7786


namespace a_is_arithmetic_sequence_b_max_min_l77_7717

-- Define the sequence a_n implicitly through S_n
def S (n : ℕ+) : ℚ := (1 / 2) * n^2 - 2 * n

-- Define a_n as the difference of consecutive S_n terms
def a (n : ℕ+) : ℚ := S n - S (n - 1)

-- Define b_n
def b (n : ℕ+) : ℚ := (a n + 1) / (a n)

-- Theorem 1: a_n is an arithmetic sequence with common difference 1
theorem a_is_arithmetic_sequence : ∀ n : ℕ+, n > 1 → a (n + 1) - a n = 1 :=
sorry

-- Theorem 2: Maximum and minimum values of b_n
theorem b_max_min :
  (∀ n : ℕ+, b n ≤ b 3) ∧
  (∀ n : ℕ+, b n ≥ b 2) ∧
  (b 3 = 3) ∧
  (b 2 = -1) :=
sorry

end a_is_arithmetic_sequence_b_max_min_l77_7717


namespace harry_pumpkin_packets_l77_7706

/-- The number of pumpkin seed packets Harry bought -/
def pumpkin_packets : ℕ := 3

/-- The cost of one packet of pumpkin seeds in dollars -/
def pumpkin_cost : ℚ := 2.5

/-- The cost of one packet of tomato seeds in dollars -/
def tomato_cost : ℚ := 1.5

/-- The cost of one packet of chili pepper seeds in dollars -/
def chili_cost : ℚ := 0.9

/-- The number of tomato seed packets Harry bought -/
def tomato_packets : ℕ := 4

/-- The number of chili pepper seed packets Harry bought -/
def chili_packets : ℕ := 5

/-- The total amount Harry spent in dollars -/
def total_spent : ℚ := 18

theorem harry_pumpkin_packets :
  pumpkin_packets * pumpkin_cost + 
  tomato_packets * tomato_cost + 
  chili_packets * chili_cost = total_spent :=
by sorry

end harry_pumpkin_packets_l77_7706


namespace water_container_problem_l77_7763

theorem water_container_problem :
  ∀ (x : ℝ),
    x > 0 →
    x / 2 + (2 * x) / 3 + (4 * x) / 4 = 26 →
    x + 2 * x + 4 * x + 26 = 84 :=
by
  sorry

end water_container_problem_l77_7763


namespace circle_radius_decrease_l77_7709

theorem circle_radius_decrease (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := 0.25 * A
  let r' := Real.sqrt (A' / π)
  r' / r = 0.5 := by
sorry

end circle_radius_decrease_l77_7709


namespace factorization_proof_l77_7718

theorem factorization_proof (a : ℝ) : 180 * a^2 + 45 * a = 45 * a * (4 * a + 1) := by
  sorry

end factorization_proof_l77_7718


namespace intersection_implies_sum_l77_7777

-- Define the functions
def f (a b x : ℝ) : ℝ := -|x - a| + b
def g (c d x : ℝ) : ℝ := |x - c| + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) :
  (f a b 3 = 6 ∧ f a b 9 = 2) ∧
  (g c d 3 = 6 ∧ g c d 9 = 2) →
  a + c = 12 := by
sorry

end intersection_implies_sum_l77_7777


namespace chair_price_l77_7712

theorem chair_price (num_tables : ℕ) (num_chairs : ℕ) (total_cost : ℕ) 
  (h1 : num_tables = 2)
  (h2 : num_chairs = 3)
  (h3 : total_cost = 110)
  (h4 : ∀ (chair_price : ℕ), num_tables * (4 * chair_price) + num_chairs * chair_price = total_cost) :
  ∃ (chair_price : ℕ), chair_price = 10 ∧ 
    num_tables * (4 * chair_price) + num_chairs * chair_price = total_cost :=
by sorry

end chair_price_l77_7712


namespace lola_cupcakes_count_l77_7771

/-- The number of mini cupcakes Lola baked -/
def lola_cupcakes : ℕ := sorry

/-- The number of pop tarts Lola baked -/
def lola_poptarts : ℕ := 10

/-- The number of blueberry pies Lola baked -/
def lola_pies : ℕ := 8

/-- The number of mini cupcakes Lulu made -/
def lulu_cupcakes : ℕ := 16

/-- The number of pop tarts Lulu made -/
def lulu_poptarts : ℕ := 12

/-- The number of blueberry pies Lulu made -/
def lulu_pies : ℕ := 14

/-- The total number of pastries made by Lola and Lulu -/
def total_pastries : ℕ := 73

theorem lola_cupcakes_count : lola_cupcakes = 13 := by
  sorry

end lola_cupcakes_count_l77_7771


namespace inequality_proof_l77_7720

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (2*b + a + c)^2 / (2*b^2 + (c + a)^2) +
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 :=
by sorry

end inequality_proof_l77_7720


namespace divisibility_property_l77_7724

theorem divisibility_property (n a b c d : ℤ) 
  (hn : n > 0)
  (h1 : n ∣ (a + b + c + d))
  (h2 : n ∣ (a^2 + b^2 + c^2 + d^2)) :
  n ∣ (a^4 + b^4 + c^4 + d^4 + 4*a*b*c*d) := by
  sorry

end divisibility_property_l77_7724


namespace ag_replacement_terminates_l77_7715

/-- Represents a sequence of As and Gs -/
inductive AGSequence
| empty : AGSequence
| cons : Char → AGSequence → AGSequence

/-- Represents the operation of replacing "AG" with "GAAA" -/
def replaceAG (s : AGSequence) : AGSequence :=
  sorry

/-- Predicate to check if a sequence contains "AG" -/
def containsAG (s : AGSequence) : Prop :=
  sorry

/-- The main theorem stating that the process will eventually terminate -/
theorem ag_replacement_terminates (initial : AGSequence) :
  ∃ (n : ℕ) (final : AGSequence), (∀ k, k ≥ n → replaceAG^[k] initial = final) ∧ ¬containsAG final :=
  sorry

end ag_replacement_terminates_l77_7715


namespace orchestra_admission_l77_7782

theorem orchestra_admission (initial_ratio_violinists : ℝ) (initial_ratio_cellists : ℝ) (initial_ratio_trumpeters : ℝ)
  (violinist_increase : ℝ) (cellist_decrease : ℝ) (total_admitted : ℕ) :
  initial_ratio_violinists = 1.6 →
  initial_ratio_cellists = 1 →
  initial_ratio_trumpeters = 0.4 →
  violinist_increase = 0.25 →
  cellist_decrease = 0.2 →
  total_admitted = 32 →
  ∃ (violinists cellists trumpeters : ℕ),
    violinists = 20 ∧
    cellists = 8 ∧
    trumpeters = 4 ∧
    violinists + cellists + trumpeters = total_admitted :=
by sorry

end orchestra_admission_l77_7782


namespace fine_arts_packaging_volume_l77_7793

/-- The volume needed to package a fine arts collection given box dimensions, cost per box, and minimum total cost. -/
theorem fine_arts_packaging_volume 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (cost_per_box : ℝ) 
  (min_total_cost : ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 12)
  (h4 : cost_per_box = 0.5)
  (h5 : min_total_cost = 225) :
  (min_total_cost / cost_per_box) * (box_length * box_width * box_height) = 2160000 := by
  sorry

#check fine_arts_packaging_volume

end fine_arts_packaging_volume_l77_7793


namespace sum_of_coefficients_l77_7798

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ x : ℚ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ = 129 := by
sorry

end sum_of_coefficients_l77_7798


namespace horizontal_asymptote_of_f_l77_7722

noncomputable def f (x : ℝ) : ℝ := (7 * x^2 - 4) / (4 * x^2 + 7 * x + 3)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N : ℝ, ∀ x > N, |f x - 7/4| < ε :=
by sorry

end horizontal_asymptote_of_f_l77_7722


namespace sum_difference_implies_sum_l77_7758

theorem sum_difference_implies_sum (a b : ℕ+) : 
  (a.val * b.val * (a.val * b.val + 1)) / 2 - 
  (a.val * (a.val + 1) * b.val * (b.val + 1)) / 4 = 1200 →
  a.val + b.val = 21 := by sorry

end sum_difference_implies_sum_l77_7758


namespace salary_calculation_l77_7704

theorem salary_calculation (salary : ℝ) 
  (food_expense : salary * (1 / 5) = salary / 5)
  (rent_expense : salary * (1 / 10) = salary / 10)
  (clothes_expense : salary * (3 / 5) = 3 * salary / 5)
  (remaining : salary - (salary / 5 + salary / 10 + 3 * salary / 5) = 14000) :
  salary = 140000 := by
sorry

end salary_calculation_l77_7704


namespace minimum_m_value_l77_7759

theorem minimum_m_value (a b c m : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : m > 0) 
  (h4 : (1 / (a - b)) + (m / (b - c)) ≥ (9 / (a - c))) : 
  m ≥ 4 := by
  sorry

end minimum_m_value_l77_7759


namespace least_positive_integer_with_remainder_one_l77_7735

theorem least_positive_integer_with_remainder_one (n : ℕ) : n = 2311 ↔ 
  (n > 1) ∧ 
  (∀ d ∈ ({2, 3, 5, 7, 11} : Set ℕ), n % d = 1) ∧ 
  (∀ m : ℕ, m > 1 → (∀ d ∈ ({2, 3, 5, 7, 11} : Set ℕ), m % d = 1) → m ≥ n) := by
sorry

end least_positive_integer_with_remainder_one_l77_7735


namespace vasya_always_wins_l77_7728

/-- Represents a player in the game -/
inductive Player : Type
| Petya : Player
| Vasya : Player

/-- Represents a move in the game -/
inductive Move : Type
| Positive : Move
| Negative : Move

/-- Represents the game state -/
structure GameState :=
(moves : List Move)
(current_player : Player)

/-- The number of divisions on each side of the triangle -/
def n : Nat := 2008

/-- The total number of cells in the triangle -/
def total_cells : Nat := n * n

/-- Determines the winner based on the final game state -/
def winner (final_state : GameState) : Player :=
  sorry

/-- The main theorem stating that Vasya always wins -/
theorem vasya_always_wins :
  ∀ (game : GameState),
  game.moves.length = total_cells →
  game.current_player = Player.Vasya →
  winner game = Player.Vasya :=
sorry

end vasya_always_wins_l77_7728


namespace cow_value_increase_l77_7797

def starting_weight : ℝ := 732
def weight_increase_factor : ℝ := 1.35
def price_per_pound : ℝ := 2.75

theorem cow_value_increase : 
  let new_weight := starting_weight * weight_increase_factor
  let value_at_new_weight := new_weight * price_per_pound
  let value_at_starting_weight := starting_weight * price_per_pound
  value_at_new_weight - value_at_starting_weight = 704.55 := by sorry

end cow_value_increase_l77_7797


namespace solution_equation_one_solution_equation_two_l77_7783

-- First equation
theorem solution_equation_one : 
  ∃ x : ℝ, (2 - x) / (x - 3) = 3 / (3 - x) ↔ x = 5 := by sorry

-- Second equation
theorem solution_equation_two : 
  ∃ x : ℝ, 4 / (x^2 - 1) + 1 = (x - 1) / (x + 1) ↔ x = -1 := by sorry

end solution_equation_one_solution_equation_two_l77_7783


namespace valid_pictures_invalid_pictures_l77_7790

-- Define a 4x4 grid
def Grid := Fin 4 → Fin 4 → Option ℕ

-- Define adjacency in the grid
def adjacent (x₁ y₁ x₂ y₂ : Fin 4) : Prop :=
  (x₁ = x₂ ∧ y₁.val + 1 = y₂.val) ∨
  (x₁ = x₂ ∧ y₂.val + 1 = y₁.val) ∨
  (y₁ = y₂ ∧ x₁.val + 1 = x₂.val) ∨
  (y₁ = y₂ ∧ x₂.val + 1 = x₁.val)

-- Define a valid grid configuration
def valid_grid (g : Grid) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 15 →
    ∃ x₁ y₁ x₂ y₂ : Fin 4,
      g x₁ y₁ = some n ∧
      g x₂ y₂ = some (n + 1) ∧
      adjacent x₁ y₁ x₂ y₂

-- Define the specific configurations for Pictures 3 and 5
def picture3 : Grid := fun x y =>
  match x, y with
  | 0, 0 => some 1 | 0, 1 => some 2 | 0, 2 => some 7 | 0, 3 => some 8
  | 1, 0 => some 14 | 1, 1 => some 3 | 1, 2 => some 6 | 1, 3 => some 9
  | 2, 0 => some 15 | 2, 1 => some 4 | 2, 2 => some 5 | 2, 3 => some 10
  | 3, 0 => some 16 | 3, 1 => none | 3, 2 => none | 3, 3 => some 11
  
def picture5 : Grid := fun x y =>
  match x, y with
  | 0, 0 => none | 0, 1 => some 4 | 0, 2 => some 5 | 0, 3 => some 6
  | 1, 0 => none | 1, 1 => some 3 | 1, 2 => none | 1, 3 => some 7
  | 2, 0 => some 14 | 2, 1 => some 2 | 2, 2 => some 9 | 2, 3 => some 8
  | 3, 0 => some 15 | 3, 1 => some 1 | 3, 2 => some 10 | 3, 3 => none

-- Theorem stating that Pictures 3 and 5 are valid configurations
theorem valid_pictures :
  valid_grid picture3 ∧ valid_grid picture5 := by sorry

-- Theorem stating that Pictures 1, 2, 4, and 6 are not valid configurations
theorem invalid_pictures :
  ¬ (∃ g : Grid, valid_grid g ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃,
      g x₁ y₁ = some 3 ∧ g x₂ y₂ = some 2 ∧ g x₃ y₃ = some 1 ∧
      adjacent x₁ y₁ x₂ y₂ ∧ adjacent x₂ y₂ x₃ y₃ ∧
      (∃ x₄ y₄ x₅ y₅, g x₄ y₄ = some 11 ∧ g x₅ y₅ = some 10 ∧ ¬adjacent x₄ y₄ x₅ y₅))) ∧
  ¬ (∃ g : Grid, valid_grid g ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃,
      g x₁ y₁ = some 1 ∧ g x₂ y₂ = some 2 ∧ g x₃ y₃ = some 3 ∧
      adjacent x₁ y₁ x₂ y₂ ∧ ¬adjacent x₂ y₂ x₃ y₃)) ∧
  ¬ (∃ g : Grid, valid_grid g ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,
      g x₁ y₁ = some 1 ∧ g x₂ y₂ = some 2 ∧ g x₃ y₃ = some 3 ∧ g x₄ y₄ = some 4 ∧
      adjacent x₁ y₁ x₂ y₂ ∧ adjacent x₂ y₂ x₃ y₃ ∧ ¬adjacent x₃ y₃ x₄ y₄)) ∧
  ¬ (∃ g : Grid, valid_grid g ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,
      g x₁ y₁ = some 4 ∧ g x₂ y₂ = some 5 ∧ g x₃ y₃ = some 6 ∧ g x₄ y₄ = some 7 ∧
      adjacent x₁ y₁ x₂ y₂ ∧ adjacent x₂ y₂ x₃ y₃ ∧ ¬adjacent x₃ y₃ x₄ y₄)) := by sorry

end valid_pictures_invalid_pictures_l77_7790


namespace grapes_and_watermelon_cost_l77_7762

/-- The cost of a pack of peanuts -/
def peanuts_cost : ℝ := sorry

/-- The cost of a cluster of grapes -/
def grapes_cost : ℝ := sorry

/-- The cost of a watermelon -/
def watermelon_cost : ℝ := sorry

/-- The cost of a box of figs -/
def figs_cost : ℝ := sorry

/-- The total cost of all items -/
def total_cost : ℝ := 30

/-- The statement of the problem -/
theorem grapes_and_watermelon_cost :
  (peanuts_cost + grapes_cost + watermelon_cost + figs_cost = total_cost) →
  (figs_cost = 2 * peanuts_cost) →
  (watermelon_cost = peanuts_cost - grapes_cost) →
  (grapes_cost + watermelon_cost = 7.5) :=
by sorry

end grapes_and_watermelon_cost_l77_7762


namespace hotel_elevator_cubic_at_15_l77_7770

/-- The hotel elevator cubic polynomial -/
def hotel_elevator_cubic (P : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, P x = a*x^3 + b*x^2 + c*x + d) ∧
  P 11 = 11 ∧ P 12 = 12 ∧ P 13 = 14 ∧ P 14 = 15

theorem hotel_elevator_cubic_at_15 (P : ℝ → ℝ) (h : hotel_elevator_cubic P) : P 15 = 13 := by
  sorry

end hotel_elevator_cubic_at_15_l77_7770


namespace point_in_bottom_right_region_of_line_l77_7757

/-- A point (x, y) is in the bottom-right region of the line ax + by + c = 0 (including the boundary) if ax + by + c ≥ 0 -/
def in_bottom_right_region (a b c x y : ℝ) : Prop := a * x + b * y + c ≥ 0

theorem point_in_bottom_right_region_of_line (t : ℝ) :
  in_bottom_right_region 1 (-2) 4 2 t → t ≥ 3 := by
  sorry

end point_in_bottom_right_region_of_line_l77_7757


namespace polygon_sides_l77_7756

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 = 3 * 360) →
  n = 8 :=
by sorry

end polygon_sides_l77_7756


namespace cookies_difference_l77_7776

def cookies_bought : ℝ := 125.75
def cookies_eaten : ℝ := 8.5

theorem cookies_difference : cookies_bought - cookies_eaten = 117.25 := by
  sorry

end cookies_difference_l77_7776


namespace parabola_shift_l77_7711

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- The original parabola y = -3x^2 -/
def original_parabola : Parabola :=
  { f := fun x => -3 * x^2 }

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { f := fun x => p.f x + v }

/-- The final parabola after shifting -/
def final_parabola : Parabola :=
  shift_vertical (shift_horizontal original_parabola 5) 2

theorem parabola_shift :
  final_parabola.f = fun x => -3 * (x - 5)^2 + 2 := by
  sorry

end parabola_shift_l77_7711


namespace uncovered_area_of_overlapping_squares_l77_7768

theorem uncovered_area_of_overlapping_squares :
  ∀ (large_side small_side : ℝ),
    large_side = 10 →
    small_side = 4 →
    large_side > 0 →
    small_side > 0 →
    large_side ≥ small_side →
    (large_side ^ 2 - small_side ^ 2) = 84 := by
  sorry

end uncovered_area_of_overlapping_squares_l77_7768


namespace total_liquid_drunk_l77_7760

/-- Converts pints to cups -/
def pints_to_cups (pints : ℝ) : ℝ := 2 * pints

/-- The amount of coffee Elijah drank in pints -/
def elijah_coffee : ℝ := 8.5

/-- The amount of water Emilio drank in pints -/
def emilio_water : ℝ := 9.5

/-- Theorem: The total amount of liquid drunk by Elijah and Emilio is 36 cups -/
theorem total_liquid_drunk : 
  pints_to_cups elijah_coffee + pints_to_cups emilio_water = 36 := by
  sorry

end total_liquid_drunk_l77_7760


namespace negative_reciprocal_of_negative_three_l77_7738

theorem negative_reciprocal_of_negative_three :
  -(1 / -3) = 1 / 3 := by
  sorry

end negative_reciprocal_of_negative_three_l77_7738


namespace initial_oranges_l77_7736

/-- Theorem: Initial number of oranges in the bin -/
theorem initial_oranges (thrown_away removed : ℕ) (added new_count : ℕ) :
  removed = 25 →
  added = 21 →
  new_count = 36 →
  ∃ initial : ℕ, initial - removed + added = new_count ∧ initial = 40 :=
by
  sorry

end initial_oranges_l77_7736


namespace jills_age_l77_7752

/-- Given that the sum of Henry and Jill's present ages is 41, and 7 years ago Henry was twice the age of Jill, prove that Jill's present age is 16 years. -/
theorem jills_age (henry_age jill_age : ℕ) 
  (sum_of_ages : henry_age + jill_age = 41)
  (past_relation : henry_age - 7 = 2 * (jill_age - 7)) : 
  jill_age = 16 := by
  sorry

end jills_age_l77_7752


namespace half_abs_diff_squares_20_15_l77_7779

theorem half_abs_diff_squares_20_15 : (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end half_abs_diff_squares_20_15_l77_7779


namespace problem_solution_l77_7708

theorem problem_solution :
  (∀ x y : ℝ, 28 * x^4 * y^2 / (7 * x^3 * y) = 4 * x * y) ∧
  ((2 * (1/3 : ℝ) + 3 * (1/2 : ℝ))^2 - (2 * (1/3 : ℝ) + (1/2 : ℝ)) * (2 * (1/3 : ℝ) - (1/2 : ℝ)) = 4.5) :=
by sorry

end problem_solution_l77_7708


namespace terminal_side_symmetry_ratio_l77_7733

theorem terminal_side_symmetry_ratio (θ : Real) (x y : Real) :
  θ ∈ Set.Ioo 0 360 →
  -- Terminal side of θ is symmetric to terminal side of 660° w.r.t. x-axis
  (∃ k : ℤ, θ + 660 = 360 * (2 * k + 1)) →
  x ≠ 0 ∨ y ≠ 0 →  -- P(x, y) is not the origin
  y / x = Real.tan θ →  -- P(x, y) is on the terminal side of θ
  x * y / (x^2 + y^2) = Real.sqrt 3 / 4 := by
  sorry

end terminal_side_symmetry_ratio_l77_7733


namespace store_profits_l77_7792

theorem store_profits (profit_a profit_b : ℝ) 
  (h : profit_a * 1.2 = profit_b * 0.9) : 
  profit_a = 0.75 * profit_b := by
sorry

end store_profits_l77_7792


namespace complex_sum_value_l77_7703

theorem complex_sum_value : 
  ∀ (c d : ℂ), c = 3 + 2*I ∧ d = 1 - 2*I → 3*c + 4*d = 13 - 2*I :=
by
  sorry

end complex_sum_value_l77_7703


namespace domain_and_rule_determine_function_exists_non_increasing_power_function_exists_function_without_zero_l77_7744

-- Define a function type
def Function (α β : Type) := α → β

-- Statement 1
theorem domain_and_rule_determine_function (α β : Type) :
  ∀ (D : Set α) (f : Function α β), ∃! (F : Function α β), ∀ x ∈ D, F x = f x :=
sorry

-- Statement 2
theorem exists_non_increasing_power_function :
  ∃ (n : ℝ), ¬ (∀ x y : ℝ, 0 < x ∧ x < y → x^n < y^n) :=
sorry

-- Statement 3
theorem exists_function_without_zero :
  ∃ (f : ℝ → ℝ) (a b : ℝ), a ≠ b ∧ f a > 0 ∧ f b < 0 ∧ ¬ (∃ c ∈ Set.Ioo a b, f c = 0) :=
sorry

end domain_and_rule_determine_function_exists_non_increasing_power_function_exists_function_without_zero_l77_7744


namespace tangent_circle_slope_l77_7702

/-- Circle represented by its equation in the form x² + y² + ax + by + c = 0 -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line y = mx contains the center of a circle tangent to two given circles -/
def has_tangent_circle (w₁ w₂ : Circle) (m : ℝ) : Prop :=
  ∃ (x y r : ℝ),
    y = m * x ∧
    (x - 4)^2 + (y - 10)^2 = (r + 3)^2 ∧
    (x + 4)^2 + (y - 10)^2 = (11 - r)^2

/-- The main theorem -/
theorem tangent_circle_slope (w₁ w₂ : Circle) :
  w₁.a = 8 ∧ w₁.b = -20 ∧ w₁.c = -75 ∧
  w₂.a = -8 ∧ w₂.b = -20 ∧ w₂.c = 125 →
  ∃ (m : ℝ),
    m > 0 ∧
    has_tangent_circle w₁ w₂ m ∧
    (∀ m' : ℝ, 0 < m' ∧ m' < m → ¬ has_tangent_circle w₁ w₂ m') ∧
    m^2 = 5/4 :=
by sorry

end tangent_circle_slope_l77_7702


namespace square_root_equality_l77_7701

theorem square_root_equality (x a : ℝ) (hx : x > 0) : 
  Real.sqrt x = 2 * a - 3 ∧ Real.sqrt x = 5 - a → a = 8/3 ∧ x = 49/9 := by
  sorry

end square_root_equality_l77_7701


namespace trash_can_purchase_l77_7785

/-- Represents the unit price of trash can type A -/
def price_A : ℕ := 500

/-- Represents the unit price of trash can type B -/
def price_B : ℕ := 550

/-- Represents the total number of trash cans to be purchased -/
def total_cans : ℕ := 6

/-- Represents the maximum total cost allowed -/
def max_cost : ℕ := 3100

/-- Theorem stating the correct unit prices and purchase options -/
theorem trash_can_purchase :
  (price_B = price_A + 50) ∧
  (2000 / price_A = 2200 / price_B) ∧
  (∀ a b : ℕ, 
    a + b = total_cans ∧ 
    price_A * a + price_B * b ≤ max_cost ∧
    a ≥ 0 ∧ b ≥ 0 →
    (a = 4 ∧ b = 2) ∨ (a = 5 ∧ b = 1) ∨ (a = 6 ∧ b = 0)) := by
  sorry

end trash_can_purchase_l77_7785


namespace father_son_age_ratio_l77_7746

/-- Proves that given the conditions, the ratio of father's age to son's age is 19:7 -/
theorem father_son_age_ratio :
  ∀ (son_age father_age : ℕ),
    (father_age - 6 = 3 * (son_age - 6)) →
    (son_age + father_age = 156) →
    (father_age : ℚ) / son_age = 19 / 7 :=
by
  sorry

end father_son_age_ratio_l77_7746


namespace square_diagonal_point_l77_7778

-- Define the square EFGH
def Square (E F G H : ℝ × ℝ) : Prop :=
  let side := dist E F
  dist E F = side ∧ dist F G = side ∧ dist G H = side ∧ dist H E = side ∧
  (E.1 - G.1) * (F.1 - H.1) + (E.2 - G.2) * (F.2 - H.2) = 0

-- Define point Q on diagonal AC
def OnDiagonal (Q E G : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (t * E.1 + (1 - t) * G.1, t * E.2 + (1 - t) * G.2)

-- Define circumcenter
def Circumcenter (R E F Q : ℝ × ℝ) : Prop :=
  dist R E = dist R F ∧ dist R F = dist R Q

-- Main theorem
theorem square_diagonal_point (E F G H Q R₁ R₂ : ℝ × ℝ) :
  Square E F G H →
  dist E F = 8 →
  OnDiagonal Q E G →
  dist E Q > dist G Q →
  Circumcenter R₁ E F Q →
  Circumcenter R₂ G H Q →
  (R₁.1 - Q.1) * (R₂.1 - Q.1) + (R₁.2 - Q.2) * (R₂.2 - Q.2) = 0 →
  dist E Q = 8 * Real.sqrt 2 :=
by sorry

end square_diagonal_point_l77_7778


namespace shaded_area_of_circles_l77_7748

theorem shaded_area_of_circles (R : ℝ) (h : R = 8) : 
  let large_circle_area := π * R^2
  let small_circle_radius := R / 2
  let small_circle_area := π * small_circle_radius^2
  let shaded_area := large_circle_area - 2 * small_circle_area
  shaded_area = 32 * π := by sorry

end shaded_area_of_circles_l77_7748


namespace odd_number_power_divisibility_l77_7707

theorem odd_number_power_divisibility (a : ℕ) (h_odd : Odd a) :
  (∀ m : ℕ, ∃ (k : ℕ → ℕ), Function.Injective k ∧ ∀ n : ℕ, (a ^ (k n) - 1) % (2 ^ m) = 0) ∧
  (∃ (S : Finset ℕ), ∀ m : ℕ, (a ^ m - 1) % (2 ^ m) = 0 → m ∈ S) :=
by sorry

end odd_number_power_divisibility_l77_7707
