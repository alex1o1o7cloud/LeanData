import Mathlib

namespace NUMINAMATH_CALUDE_prob_green_ball_is_13_28_l1836_183696

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers A, B, C, and D -/
def containers : List Container := [
  ⟨4, 6⟩,  -- Container A
  ⟨8, 6⟩,  -- Container B
  ⟨8, 6⟩,  -- Container C
  ⟨3, 7⟩   -- Container D
]

/-- The number of containers -/
def numContainers : ℕ := containers.length

/-- The probability of selecting a green ball -/
def probGreenBall : ℚ := 
  (1 / numContainers) * (containers.map greenProbability).sum

theorem prob_green_ball_is_13_28 : probGreenBall = 13/28 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_ball_is_13_28_l1836_183696


namespace NUMINAMATH_CALUDE_lines_in_same_plane_l1836_183634

-- Define the necessary types
variable (Point Line Plane : Type)

-- Define the necessary relations
variable (lies_in : Point → Line → Prop)
variable (lies_in_plane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem lines_in_same_plane 
  (a b : Line) 
  (α : Plane) 
  (h1 : intersect a b) 
  (h2 : lies_in_plane a α) 
  (h3 : lies_in_plane b α) :
  ∀ (c : Line), parallel c b → (∃ (p : Point), lies_in p a ∧ lies_in p c) → 
  lies_in_plane c α :=
sorry

end NUMINAMATH_CALUDE_lines_in_same_plane_l1836_183634


namespace NUMINAMATH_CALUDE_stating_not_always_two_triangles_form_rectangle_l1836_183609

/-- Represents a non-isosceles right triangle -/
structure NonIsoscelesRightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  leg1_ne_leg2 : leg1 ≠ leg2
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

/-- Represents a rectangle constructed from non-isosceles right triangles -/
structure RectangleFromTriangles where
  width : ℝ
  height : ℝ
  triangle : NonIsoscelesRightTriangle
  num_triangles : ℕ
  area_equality : width * height = num_triangles * (triangle.leg1 * triangle.leg2 / 2)

/-- 
Theorem stating that it's not always necessary for any two identical non-isosceles 
right triangles to form a rectangle when a larger rectangle is constructed from 
these triangles without gaps or overlaps
-/
theorem not_always_two_triangles_form_rectangle 
  (r : RectangleFromTriangles) : 
  ¬ ∀ (t1 t2 : NonIsoscelesRightTriangle), 
    t1 = r.triangle → t2 = r.triangle → 
    ∃ (w h : ℝ), w * h = t1.leg1 * t1.leg2 + t2.leg1 * t2.leg2 := by
  sorry

end NUMINAMATH_CALUDE_stating_not_always_two_triangles_form_rectangle_l1836_183609


namespace NUMINAMATH_CALUDE_a_neg_three_sufficient_not_necessary_l1836_183681

/-- Two lines in the plane, parameterized by a real number a -/
def line1 (a : ℝ) := {(x, y) : ℝ × ℝ | x + a * y + 2 = 0}
def line2 (a : ℝ) := {(x, y) : ℝ × ℝ | a * x + (a + 2) * y + 1 = 0}

/-- The condition for two lines to be perpendicular -/
def are_perpendicular (a : ℝ) : Prop :=
  a * (a + 3) = 0

/-- The statement to be proved -/
theorem a_neg_three_sufficient_not_necessary :
  (∀ a : ℝ, a = -3 → are_perpendicular a) ∧
  ¬(∀ a : ℝ, are_perpendicular a → a = -3) :=
sorry

end NUMINAMATH_CALUDE_a_neg_three_sufficient_not_necessary_l1836_183681


namespace NUMINAMATH_CALUDE_raffle_prize_calculation_l1836_183650

theorem raffle_prize_calculation (kept_amount : ℝ) (kept_percentage : ℝ) (total_prize : ℝ) : 
  kept_amount = 80 → kept_percentage = 0.80 → kept_amount = kept_percentage * total_prize → 
  total_prize = 100 := by
sorry

end NUMINAMATH_CALUDE_raffle_prize_calculation_l1836_183650


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1836_183652

theorem inequality_equivalence (x : ℝ) : 
  ‖‖x - 2‖ - 1‖ ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1836_183652


namespace NUMINAMATH_CALUDE_ratio_expression_l1836_183602

theorem ratio_expression (a b : ℚ) (h : a / b = 4 / 1) :
  (a - 3 * b) / (2 * a - b) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_expression_l1836_183602


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1836_183695

theorem arithmetic_calculations :
  (456 - 9 * 8 = 384) ∧
  (387 + 126 - 212 = 301) ∧
  (533 - (108 + 209) = 216) ∧
  ((746 - 710) / 6 = 6) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1836_183695


namespace NUMINAMATH_CALUDE_xyz_congruence_l1836_183637

theorem xyz_congruence (x y z : ℕ) : 
  x < 8 → y < 8 → z < 8 →
  (x + 3*y + 2*z) % 8 = 1 →
  (2*x + y + 3*z) % 8 = 5 →
  (3*x + 2*y + z) % 8 = 3 →
  (x*y*z) % 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_xyz_congruence_l1836_183637


namespace NUMINAMATH_CALUDE_rocket_max_height_l1836_183603

/-- The height of a rocket as a function of time -/
def rocket_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 50

/-- Theorem stating that the maximum height of the rocket is 175 meters -/
theorem rocket_max_height :
  ∃ (max_height : ℝ), max_height = 175 ∧ ∀ (t : ℝ), rocket_height t ≤ max_height :=
by sorry

end NUMINAMATH_CALUDE_rocket_max_height_l1836_183603


namespace NUMINAMATH_CALUDE_dining_bill_share_l1836_183667

theorem dining_bill_share (people : ℕ) (bill tip_percent tax_percent : ℚ) 
  (h_people : people = 15)
  (h_bill : bill = 350)
  (h_tip_percent : tip_percent = 18 / 100)
  (h_tax_percent : tax_percent = 5 / 100) :
  (bill + bill * tip_percent + bill * tax_percent) / people = 287 / 10 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_share_l1836_183667


namespace NUMINAMATH_CALUDE_pizza_slice_volume_l1836_183613

/-- The volume of a pizza slice -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_slices : ℕ) :
  thickness = 1/2 →
  diameter = 16 →
  num_slices = 8 →
  (π * (diameter/2)^2 * thickness) / num_slices = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_pizza_slice_volume_l1836_183613


namespace NUMINAMATH_CALUDE_forty_percent_bought_something_l1836_183622

/-- Given advertising costs, number of customers, item price, and profit,
    calculates the percentage of customers who made a purchase. -/
def percentage_of_customers_who_bought (advertising_cost : ℕ) (num_customers : ℕ) 
  (item_price : ℕ) (profit : ℕ) : ℚ :=
  (profit / item_price : ℚ) / num_customers * 100

/-- Theorem stating that under the given conditions, 
    40% of customers made a purchase. -/
theorem forty_percent_bought_something :
  percentage_of_customers_who_bought 1000 100 25 1000 = 40 := by
  sorry

#eval percentage_of_customers_who_bought 1000 100 25 1000

end NUMINAMATH_CALUDE_forty_percent_bought_something_l1836_183622


namespace NUMINAMATH_CALUDE_sector_radius_and_angle_l1836_183679

/-- Given a sector with perimeter 4 and area 1, prove its radius is 1 and central angle is 2 -/
theorem sector_radius_and_angle (r θ : ℝ) 
  (h_perimeter : 2 * r + θ * r = 4)
  (h_area : 1/2 * θ * r^2 = 1) : 
  r = 1 ∧ θ = 2 := by sorry

end NUMINAMATH_CALUDE_sector_radius_and_angle_l1836_183679


namespace NUMINAMATH_CALUDE_newspaper_printing_time_l1836_183666

/-- Represents the time taken to print newspapers -/
def print_time (presses : ℕ) (newspapers : ℕ) : ℚ :=
  6 * (4 : ℚ) * newspapers / (8000 * presses)

theorem newspaper_printing_time :
  print_time 2 6000 = 9 :=
by sorry

end NUMINAMATH_CALUDE_newspaper_printing_time_l1836_183666


namespace NUMINAMATH_CALUDE_butter_profit_percentage_l1836_183655

/-- Calculates the profit percentage for a butter mixture sale --/
theorem butter_profit_percentage
  (butter1_weight : ℝ)
  (butter1_price : ℝ)
  (butter2_weight : ℝ)
  (butter2_price : ℝ)
  (selling_price : ℝ)
  (h1 : butter1_weight = 44)
  (h2 : butter1_price = 150)
  (h3 : butter2_weight = 36)
  (h4 : butter2_price = 125)
  (h5 : selling_price = 194.25) :
  let total_cost := butter1_weight * butter1_price + butter2_weight * butter2_price
  let total_weight := butter1_weight + butter2_weight
  let total_selling_price := total_weight * selling_price
  let profit := total_selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 40 := by
sorry

end NUMINAMATH_CALUDE_butter_profit_percentage_l1836_183655


namespace NUMINAMATH_CALUDE_intersection_point_of_AB_CD_l1836_183670

def A : ℝ × ℝ × ℝ := (3, -2, 5)
def B : ℝ × ℝ × ℝ := (13, -12, 10)
def C : ℝ × ℝ × ℝ := (-2, 5, -8)
def D : ℝ × ℝ × ℝ := (3, -1, 12)

def line_intersection (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

theorem intersection_point_of_AB_CD :
  line_intersection A B C D = (-1/11, 1/11, 15/11) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_AB_CD_l1836_183670


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1836_183614

theorem decimal_to_fraction : (2.375 : ℚ) = 19 / 8 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1836_183614


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_over_four_l1836_183692

theorem tan_theta_minus_pi_over_four (θ : Real) :
  let z : ℂ := Complex.mk (Real.cos θ - 4/5) (Real.sin θ - 3/5)
  z.re = 0 → Real.tan (θ - Real.pi/4) = -7 :=
by sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_over_four_l1836_183692


namespace NUMINAMATH_CALUDE_roger_step_goal_time_l1836_183678

/-- Represents the time it takes Roger to reach his step goal -/
def time_to_reach_goal (steps_per_interval : ℕ) (interval_duration : ℕ) (goal_steps : ℕ) : ℕ :=
  (goal_steps * interval_duration) / steps_per_interval

/-- Proves that Roger will take 150 minutes to reach his goal of 10,000 steps -/
theorem roger_step_goal_time :
  time_to_reach_goal 2000 30 10000 = 150 := by
  sorry

end NUMINAMATH_CALUDE_roger_step_goal_time_l1836_183678


namespace NUMINAMATH_CALUDE_f_properties_l1836_183629

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2

def is_smallest_positive_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, (0 < S ∧ S < T) → ∃ y, f (y + S) ≠ f y

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem f_properties :
  is_smallest_positive_period π f ∧
  is_monotonically_decreasing f (π / 3) (5 * π / 6) ∧
  ∀ α : ℝ, 
    (3 * π / 2 < α ∧ α < 2 * π) →  -- α in fourth quadrant
    Real.cos α = 3 / 5 → 
    f (α / 2 + 7 * π / 12) = 8 / 5 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1836_183629


namespace NUMINAMATH_CALUDE_comparison_theorem_l1836_183646

theorem comparison_theorem :
  (-4 / 7 : ℚ) > -2 / 3 ∧ -(-7 : ℤ) > -|(-7 : ℤ)| := by sorry

end NUMINAMATH_CALUDE_comparison_theorem_l1836_183646


namespace NUMINAMATH_CALUDE_jasons_quarters_l1836_183618

/-- Given that Jason had 49 quarters initially and his dad gave him 25 quarters,
    prove that Jason now has 74 quarters. -/
theorem jasons_quarters (initial : ℕ) (given : ℕ) (total : ℕ) 
    (h1 : initial = 49) 
    (h2 : given = 25) 
    (h3 : total = initial + given) : 
  total = 74 := by
  sorry

end NUMINAMATH_CALUDE_jasons_quarters_l1836_183618


namespace NUMINAMATH_CALUDE_least_cubes_for_6x9x12_block_l1836_183630

/-- Represents the dimensions of a cuboidal block in centimeters -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the least number of equal cubes that can be cut from a block -/
def leastNumberOfEqualCubes (d : BlockDimensions) : ℕ :=
  (d.length * d.width * d.height) / (Nat.gcd d.length (Nat.gcd d.width d.height))^3

/-- Theorem stating that for a 6x9x12 cm block, the least number of equal cubes is 24 -/
theorem least_cubes_for_6x9x12_block :
  leastNumberOfEqualCubes ⟨6, 9, 12⟩ = 24 := by
  sorry

#eval leastNumberOfEqualCubes ⟨6, 9, 12⟩

end NUMINAMATH_CALUDE_least_cubes_for_6x9x12_block_l1836_183630


namespace NUMINAMATH_CALUDE_congruent_side_length_for_specific_triangle_l1836_183648

/-- Represents an isosceles triangle with base length and area -/
structure IsoscelesTriangle where
  base : ℝ
  area : ℝ

/-- Calculates the length of a congruent side in an isosceles triangle -/
def congruentSideLength (triangle : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating that for an isosceles triangle with base 30 and area 72, 
    the length of a congruent side is 15.75 -/
theorem congruent_side_length_for_specific_triangle :
  let triangle : IsoscelesTriangle := { base := 30, area := 72 }
  congruentSideLength triangle = 15.75 := by sorry

end NUMINAMATH_CALUDE_congruent_side_length_for_specific_triangle_l1836_183648


namespace NUMINAMATH_CALUDE_sixteenth_root_of_unity_l1836_183657

theorem sixteenth_root_of_unity : 
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 15 ∧ 
  (Complex.tan (π / 8) + Complex.I) / (Complex.tan (π / 8) - Complex.I) = 
  Complex.exp (Complex.I * (2 * ↑n * π / 16)) :=
sorry

end NUMINAMATH_CALUDE_sixteenth_root_of_unity_l1836_183657


namespace NUMINAMATH_CALUDE_ratio_multiple_choice_to_free_response_l1836_183687

/-- Represents the number of problems of each type in Stacy's homework assignment --/
structure HomeworkAssignment where
  total : Nat
  truefalse : Nat
  freeresponse : Nat
  multiplechoice : Nat

/-- Conditions for Stacy's homework assignment --/
def stacysHomework : HomeworkAssignment where
  total := 45
  truefalse := 6
  freeresponse := 13  -- 6 + 7
  multiplechoice := 26 -- 45 - (13 + 6)

theorem ratio_multiple_choice_to_free_response :
  (stacysHomework.multiplechoice : ℚ) / stacysHomework.freeresponse = 2 / 1 := by
  sorry

#check ratio_multiple_choice_to_free_response

end NUMINAMATH_CALUDE_ratio_multiple_choice_to_free_response_l1836_183687


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l1836_183642

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℝ) + (⌈x⌉ : ℝ) = 7 ↔ 3 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l1836_183642


namespace NUMINAMATH_CALUDE_characterization_of_special_numbers_l1836_183669

/-- A natural number n > 1 satisfies the given condition if and only if it's prime or a square of a prime -/
theorem characterization_of_special_numbers (n : ℕ) (h : n > 1) :
  (∀ d : ℕ, d > 1 → d ∣ n → (d - 1) ∣ (n - 1)) ↔ 
  (Nat.Prime n ∨ ∃ p : ℕ, Nat.Prime p ∧ n = p^2) := by
  sorry

end NUMINAMATH_CALUDE_characterization_of_special_numbers_l1836_183669


namespace NUMINAMATH_CALUDE_negative_integers_satisfying_condition_l1836_183663

theorem negative_integers_satisfying_condition :
  ∀ a : ℤ, a < 0 →
    ((4 * a + 1) / 6 : ℚ) > -2 ↔ (a = -1 ∨ a = -2 ∨ a = -3) :=
by sorry

end NUMINAMATH_CALUDE_negative_integers_satisfying_condition_l1836_183663


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1836_183653

theorem necessary_not_sufficient_condition (a : ℝ) : 
  (∀ x, ax + 1 = 0 → x^2 + x - 6 = 0) ∧ 
  (∃ x, x^2 + x - 6 = 0 ∧ ax + 1 ≠ 0) →
  a = -1/2 ∨ a = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1836_183653


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l1836_183649

/-- The number of fruit options Joe has -/
def num_fruits : ℕ := 4

/-- The number of meals Joe has in a day -/
def num_meals : ℕ := 3

/-- The probability of choosing any specific fruit for a meal -/
def prob_single_fruit : ℚ := 1 / num_fruits

/-- The probability of eating the same fruit for all meals -/
def prob_same_fruit : ℚ := num_fruits * prob_single_fruit ^ num_meals

/-- The probability of eating at least two different kinds of fruit in a day -/
def prob_different_fruits : ℚ := 1 - prob_same_fruit

theorem joe_fruit_probability :
  prob_different_fruits = 15 / 16 :=
sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l1836_183649


namespace NUMINAMATH_CALUDE_journey_ratio_l1836_183628

/-- Proves the ratio of distance after storm to total journey distance -/
theorem journey_ratio (speed : ℝ) (time : ℝ) (storm_distance : ℝ) : 
  speed = 30 ∧ time = 20 ∧ storm_distance = 200 →
  (speed * time - storm_distance) / (2 * speed * time) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_journey_ratio_l1836_183628


namespace NUMINAMATH_CALUDE_fibonacci_mod_13_not_4_l1836_183664

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem fibonacci_mod_13_not_4 (n : ℕ) : fibonacci n % 13 ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_mod_13_not_4_l1836_183664


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1836_183643

-- Define set A
def A : Set ℝ := {x | x^2 - 5*x + 6 ≤ 0}

-- Define set B
def B : Set ℝ := {x | |2*x - 1| > 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1836_183643


namespace NUMINAMATH_CALUDE_jackson_collection_l1836_183626

/-- Calculates the total number of souvenirs collected by Jackson -/
def total_souvenirs (hermit_crabs : ℕ) (shells_per_crab : ℕ) (starfish_per_shell : ℕ) (dollars_per_starfish : ℕ) : ℕ :=
  let spiral_shells := hermit_crabs * shells_per_crab
  let starfish := spiral_shells * starfish_per_shell
  let sand_dollars := starfish * dollars_per_starfish
  hermit_crabs + spiral_shells + starfish + sand_dollars

/-- Theorem stating that Jackson's collection totals 3672 souvenirs -/
theorem jackson_collection :
  total_souvenirs 72 5 3 2 = 3672 := by
  sorry


end NUMINAMATH_CALUDE_jackson_collection_l1836_183626


namespace NUMINAMATH_CALUDE_concentric_circles_area_difference_l1836_183636

theorem concentric_circles_area_difference
  (r : ℝ) -- radius of smaller circle
  (R : ℝ) -- radius of larger circle
  (h1 : r > 0)
  (h2 : R > r)
  (h3 : π * R^2 = 4 * π * r^2) -- area ratio 1:4
  (h4 : π * r^2 = 4 * π) -- area of smaller circle is 4π
  : π * R^2 - π * r^2 = 12 * π := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_area_difference_l1836_183636


namespace NUMINAMATH_CALUDE_dress_price_inconsistency_l1836_183659

theorem dress_price_inconsistency :
  ¬∃ (D : ℝ), D > 0 ∧ 7 * D + 4 * 5 + 8 * 15 + 6 * 20 = 250 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_inconsistency_l1836_183659


namespace NUMINAMATH_CALUDE_xy_sum_when_equation_zero_l1836_183671

theorem xy_sum_when_equation_zero (x y : ℝ) :
  (3 * x - y + 5)^2 + |2 * x - y + 3| = 0 → x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_when_equation_zero_l1836_183671


namespace NUMINAMATH_CALUDE_no_integer_solution_l1836_183606

theorem no_integer_solution : ¬∃ (x y : ℤ), Real.sqrt ((x^2 : ℝ) + x + 1) + Real.sqrt ((y^2 : ℝ) - y + 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1836_183606


namespace NUMINAMATH_CALUDE_sum_M_N_equals_two_l1836_183639

/-- Definition of M -/
def M : ℚ := 1^5 + 2^4 * 3^3 - 4^2 / 5^1

/-- Definition of N -/
def N : ℚ := 1^5 - 2^4 * 3^3 + 4^2 / 5^1

/-- Theorem: The sum of M and N is equal to 2 -/
theorem sum_M_N_equals_two : M + N = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_M_N_equals_two_l1836_183639


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l1836_183608

/-- Taxi fare function -/
def fare (k : ℝ) (d : ℝ) : ℝ := 20 + k * d

/-- Theorem: If the fare for 60 miles is $140, then the fare for 85 miles is $190 -/
theorem taxi_fare_calculation (k : ℝ) :
  fare k 60 = 140 → fare k 85 = 190 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l1836_183608


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l1836_183615

theorem difference_of_squares_special_case : (4 + Real.sqrt 6) * (4 - Real.sqrt 6) = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l1836_183615


namespace NUMINAMATH_CALUDE_unique_number_digit_sum_l1836_183672

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem unique_number_digit_sum :
  ∃! N : ℕ, 400 < N ∧ N < 600 ∧ N % 2 = 1 ∧ N % 5 = 0 ∧ N % 11 = 0 ∧ sumOfDigits N = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_number_digit_sum_l1836_183672


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_12_l1836_183610

theorem x_plus_2y_equals_12 (x y : ℝ) (h1 : x = 6) (h2 : y = 3) : x + 2*y = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_12_l1836_183610


namespace NUMINAMATH_CALUDE_mean_value_theorem_application_l1836_183674

-- Define the function f(x) = x^2 + 3
def f (x : ℝ) : ℝ := x^2 + 3

-- State the theorem
theorem mean_value_theorem_application :
  ∃ c ∈ (Set.Ioo (-1) 2), 
    (deriv f c) = (f 2 - f (-1)) / (2 - (-1)) :=
by
  sorry

end NUMINAMATH_CALUDE_mean_value_theorem_application_l1836_183674


namespace NUMINAMATH_CALUDE_parabola_directrix_l1836_183682

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = 3 * x^2 - 6 * x + 1

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop := y = -25 / 12

/-- Theorem: The directrix of the given parabola has the equation y = -25/12 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_equation x y → ∃ y_directrix : ℝ, directrix_equation y_directrix :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1836_183682


namespace NUMINAMATH_CALUDE_vector_operations_l1836_183644

def a : ℝ × ℝ := (3, 3)
def b : ℝ × ℝ := (1, 4)

theorem vector_operations :
  (2 • a - b = (5, 2)) ∧
  (∃ m : ℝ, m = -2 ∧ ∃ k : ℝ, k • (m • a + b) = 2 • a - b) := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l1836_183644


namespace NUMINAMATH_CALUDE_p_greater_than_q_l1836_183676

theorem p_greater_than_q (x y : ℝ) (h1 : x < y) (h2 : y < 0) 
  (p : ℝ := (x^2 + y^2)*(x - y)) (q : ℝ := (x^2 - y^2)*(x + y)) : p > q := by
  sorry

end NUMINAMATH_CALUDE_p_greater_than_q_l1836_183676


namespace NUMINAMATH_CALUDE_intersection_point_of_f_and_inverse_l1836_183611

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 9*x^2 + 24*x + 36

-- State the theorem
theorem intersection_point_of_f_and_inverse :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-3, -3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_f_and_inverse_l1836_183611


namespace NUMINAMATH_CALUDE_shaded_triangle_probability_l1836_183625

/-- The total number of triangles in the diagram -/
def total_triangles : ℕ := 10

/-- The number of shaded or partially shaded triangles -/
def shaded_triangles : ℕ := 3

/-- Each triangle has an equal probability of being selected -/
axiom equal_probability : True

/-- The probability of selecting a shaded or partially shaded triangle -/
def shaded_probability : ℚ := shaded_triangles / total_triangles

theorem shaded_triangle_probability : 
  shaded_probability = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_shaded_triangle_probability_l1836_183625


namespace NUMINAMATH_CALUDE_exponent_operation_l1836_183690

theorem exponent_operation (a : ℝ) : -(-a)^2 * a^4 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_operation_l1836_183690


namespace NUMINAMATH_CALUDE_unique_prime_divides_sigma_pred_l1836_183647

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Theorem: The only prime p that divides σ(p-1) is 3 -/
theorem unique_prime_divides_sigma_pred :
  ∀ p : ℕ, Nat.Prime p → (p ∣ sigma (p - 1) ↔ p = 3) := by sorry

end NUMINAMATH_CALUDE_unique_prime_divides_sigma_pred_l1836_183647


namespace NUMINAMATH_CALUDE_total_worth_is_22800_l1836_183619

def engagement_ring_cost : ℝ := 4000
def car_cost : ℝ := 2000
def diamond_bracelet_cost : ℝ := 2 * engagement_ring_cost
def designer_gown_cost : ℝ := 0.5 * diamond_bracelet_cost
def jewelry_set_cost : ℝ := 1.2 * engagement_ring_cost

def total_worth : ℝ := engagement_ring_cost + car_cost + diamond_bracelet_cost + designer_gown_cost + jewelry_set_cost

theorem total_worth_is_22800 : total_worth = 22800 := by sorry

end NUMINAMATH_CALUDE_total_worth_is_22800_l1836_183619


namespace NUMINAMATH_CALUDE_raft_drift_time_l1836_183683

/-- The time for a raft to drift from B to A, given boat travel times -/
theorem raft_drift_time (boat_speed : ℝ) (current_speed : ℝ) (distance : ℝ) 
  (h1 : distance / (boat_speed + current_speed) = 7)
  (h2 : distance / (boat_speed - current_speed) = 5)
  (h3 : boat_speed > 0)
  (h4 : current_speed > 0) :
  distance / current_speed = 35 := by
sorry

end NUMINAMATH_CALUDE_raft_drift_time_l1836_183683


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_circle_l1836_183673

theorem square_area_from_rectangle_circle (rectangle_length : ℝ) (circle_radius : ℝ) (square_side : ℝ) : 
  rectangle_length = (2 / 5) * circle_radius →
  circle_radius = square_side →
  rectangle_length * 10 = 180 →
  square_side ^ 2 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_circle_l1836_183673


namespace NUMINAMATH_CALUDE_cyclic_system_solutions_l1836_183631

def cyclicSystem (x : Fin 5 → ℝ) (y : ℝ) : Prop :=
  ∀ i : Fin 5, x i + x ((i + 2) % 5) = y * x ((i + 1) % 5)

theorem cyclic_system_solutions :
  ∀ x : Fin 5 → ℝ, ∀ y : ℝ,
    cyclicSystem x y ↔
      ((∀ i : Fin 5, x i = 0) ∨
      (y = 2 ∧ ∃ s : ℝ, ∀ i : Fin 5, x i = s) ∨
      ((y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) ∧
        ∃ s t : ℝ, x 0 = s ∧ x 1 = t ∧ x 2 = -s + y*t ∧ x 3 = -y*s - t ∧ x 4 = y*s - t)) :=
by
  sorry


end NUMINAMATH_CALUDE_cyclic_system_solutions_l1836_183631


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1836_183612

-- Define the given speeds
def speed_with_current : ℝ := 15
def current_speed : ℝ := 2.8

-- Define the speed against the current
def speed_against_current : ℝ := speed_with_current - 2 * current_speed

-- Theorem statement
theorem mans_speed_against_current :
  speed_against_current = 9.4 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l1836_183612


namespace NUMINAMATH_CALUDE_brick_height_calculation_l1836_183645

/-- Calculates the height of a brick given wall dimensions and brick count -/
theorem brick_height_calculation (wall_length wall_width wall_height : ℝ) 
  (brick_length brick_width : ℝ) (brick_count : ℕ) :
  wall_length = 27 →
  wall_width = 2 →
  wall_height = 0.75 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_count = 27000 →
  ∃ (brick_height : ℝ), 
    brick_height = (wall_length * wall_width * wall_height) / (brick_length * brick_width * brick_count) ∧
    brick_height = 0.075 := by
  sorry

end NUMINAMATH_CALUDE_brick_height_calculation_l1836_183645


namespace NUMINAMATH_CALUDE_garden_area_l1836_183607

theorem garden_area (perimeter : ℝ) (length width : ℝ) : 
  perimeter = 2 * (length + width) →
  length = 3 * width →
  perimeter = 84 →
  length * width = 330.75 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l1836_183607


namespace NUMINAMATH_CALUDE_boys_percentage_of_school_l1836_183617

theorem boys_percentage_of_school (total_students : ℕ) (boys_representation : ℕ) 
  (h1 : total_students = 180)
  (h2 : boys_representation = 162)
  (h3 : boys_representation = (180 / 100) * (boys_percentage / 100 * total_students)) :
  boys_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_boys_percentage_of_school_l1836_183617


namespace NUMINAMATH_CALUDE_quadratic_roots_l1836_183677

theorem quadratic_roots (p q a b : ℤ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = a ∨ x = b) →  -- polynomial has roots a and b
  a ≠ b →                                    -- roots are distinct
  a ≠ 0 →                                    -- a is non-zero
  b ≠ 0 →                                    -- b is non-zero
  (a + p) % (q - 2*b) = 0 →                  -- a + p is divisible by q - 2b
  a = 1 ∨ a = 3 :=                           -- possible values for a
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1836_183677


namespace NUMINAMATH_CALUDE_annie_money_left_l1836_183632

/-- Calculates the amount of money Annie has left after buying hamburgers and milkshakes. -/
def money_left (initial_money hamburger_price milkshake_price hamburger_count milkshake_count : ℕ) : ℕ :=
  initial_money - (hamburger_price * hamburger_count + milkshake_price * milkshake_count)

/-- Proves that Annie has $70 left after her purchases. -/
theorem annie_money_left :
  money_left 132 4 5 8 6 = 70 := by
  sorry

end NUMINAMATH_CALUDE_annie_money_left_l1836_183632


namespace NUMINAMATH_CALUDE_sarah_tom_seating_probability_l1836_183662

/-- The number of chairs in the row -/
def n : ℕ := 10

/-- The probability that Sarah and Tom do not sit next to each other -/
def probability_not_adjacent : ℚ := 4 / 5

/-- Theorem stating that the probability of Sarah and Tom not sitting next to each other
    in a row of n chairs is equal to probability_not_adjacent -/
theorem sarah_tom_seating_probability :
  (1 : ℚ) - (n - 1 : ℚ) / (n.choose 2 : ℚ) = probability_not_adjacent :=
sorry

end NUMINAMATH_CALUDE_sarah_tom_seating_probability_l1836_183662


namespace NUMINAMATH_CALUDE_contrapositive_theorem_l1836_183660

theorem contrapositive_theorem (a b : ℝ) :
  (∀ a b, a > b → 2^a > 2^b - 1) ↔
  (∀ a b, 2^a ≤ 2^b - 1 → a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_theorem_l1836_183660


namespace NUMINAMATH_CALUDE_initial_speed_calculation_l1836_183661

/-- Represents a baseball player's training progress -/
structure BaseballTraining where
  initialSpeed : ℝ
  trainingWeeks : ℕ
  speedGainPerWeek : ℝ
  finalSpeedIncrease : ℝ

/-- Theorem stating the initial speed of a baseball player given their training progress -/
theorem initial_speed_calculation (training : BaseballTraining)
  (h1 : training.trainingWeeks = 16)
  (h2 : training.speedGainPerWeek = 1)
  (h3 : training.finalSpeedIncrease = 0.2)
  : training.initialSpeed = 80 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_calculation_l1836_183661


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1836_183600

def U : Set Nat := {0, 1, 2, 4, 8}
def A : Set Nat := {1, 2, 8}
def B : Set Nat := {2, 4, 8}

theorem complement_intersection_theorem : 
  (U \ (A ∩ B)) = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1836_183600


namespace NUMINAMATH_CALUDE_divisible_by_six_l1836_183685

theorem divisible_by_six (n : ℤ) : ∃ k : ℤ, n * (n + 1) * (n + 2) = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l1836_183685


namespace NUMINAMATH_CALUDE_infinite_series_solution_l1836_183635

theorem infinite_series_solution : ∃! x : ℝ, 0 < x ∧ x < 1 ∧ x^3 + 2*x - 1 = 0 ∧ 
  x = (1 - x) / (1 + x^2) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_solution_l1836_183635


namespace NUMINAMATH_CALUDE_davids_english_marks_l1836_183694

def davidsMaths : ℕ := 89
def davidsPhysics : ℕ := 82
def davidsChemistry : ℕ := 87
def davidsBiology : ℕ := 81
def averageMarks : ℕ := 85
def numberOfSubjects : ℕ := 5

theorem davids_english_marks :
  ∃ (englishMarks : ℕ),
    (englishMarks + davidsMaths + davidsPhysics + davidsChemistry + davidsBiology) / numberOfSubjects = averageMarks ∧
    englishMarks = 86 := by
  sorry

end NUMINAMATH_CALUDE_davids_english_marks_l1836_183694


namespace NUMINAMATH_CALUDE_f_properties_l1836_183604

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1) ∧
  (∀ (α : ℝ), α ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → f α = 6 / 5 → Real.cos (2 * α) = (3 - 4 * Real.sqrt 3) / 10) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l1836_183604


namespace NUMINAMATH_CALUDE_average_weight_b_c_l1836_183665

/-- Given three weights a, b, and c, prove that the average weight of b and c is 50 kg
    under the specified conditions. -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 60 →  -- The average weight of a, b, and c is 60 kg
  (a + b) / 2 = 70 →      -- The average weight of a and b is 70 kg
  b = 60 →                -- The weight of b is 60 kg
  (b + c) / 2 = 50 :=     -- The average weight of b and c is 50 kg
by sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l1836_183665


namespace NUMINAMATH_CALUDE_expression_evaluation_l1836_183638

theorem expression_evaluation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  ((x^2 - 1)^2 * (x^3 - x^2 + 1)^2 / (x^5 - 1)^2)^2 * 
  ((x^2 + 1)^2 * (x^3 + x^2 + 1)^2 / (x^5 + 1)^2)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1836_183638


namespace NUMINAMATH_CALUDE_people_to_left_of_kolya_l1836_183699

/-- Represents a person in the line -/
structure Person where
  name : String

/-- Represents the arrangement of people in a line -/
structure Arrangement where
  people : List Person
  kolya_index : Nat
  sasha_index : Nat

/-- The number of people to the right of a person at a given index -/
def peopleToRight (arr : Arrangement) (index : Nat) : Nat :=
  arr.people.length - index - 1

/-- The number of people to the left of a person at a given index -/
def peopleToLeft (arr : Arrangement) (index : Nat) : Nat :=
  index

theorem people_to_left_of_kolya (arr : Arrangement) 
  (h1 : peopleToRight arr arr.kolya_index = 12)
  (h2 : peopleToLeft arr arr.sasha_index = 20)
  (h3 : peopleToRight arr arr.sasha_index = 8) :
  peopleToLeft arr arr.kolya_index = 16 := by
  sorry

end NUMINAMATH_CALUDE_people_to_left_of_kolya_l1836_183699


namespace NUMINAMATH_CALUDE_min_distance_MN_l1836_183623

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  -- Equation of the hyperbola: x²/4 - y² = 1
  equation : ℝ → ℝ → Prop
  -- One asymptote has equation x - 2y = 0
  asymptote : ℝ → ℝ → Prop
  -- The hyperbola passes through (2√2, 1)
  passes_through : Prop

/-- Represents a point on the hyperbola -/
structure PointOnHyperbola (C : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : C.equation x y

/-- Represents the vertices of the hyperbola -/
structure HyperbolaVertices (C : Hyperbola) where
  A₁ : ℝ × ℝ  -- Left vertex
  A₂ : ℝ × ℝ  -- Right vertex

/-- Function to calculate |MN| given a point P on the hyperbola -/
def distance_MN (C : Hyperbola) (V : HyperbolaVertices C) (P : PointOnHyperbola C) : ℝ :=
  sorry  -- Definition of |MN| calculation

/-- The main theorem to prove -/
theorem min_distance_MN (C : Hyperbola) (V : HyperbolaVertices C) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 3 ∧
  ∀ (P : PointOnHyperbola C), distance_MN C V P ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_MN_l1836_183623


namespace NUMINAMATH_CALUDE_categorical_variables_are_correct_l1836_183658

-- Define the type for variables
inductive Variable
  | Smoking
  | Gender
  | Religious_Belief
  | Nationality

-- Define a function to check if a variable is categorical
def is_categorical (v : Variable) : Prop :=
  v = Variable.Gender ∨ v = Variable.Religious_Belief ∨ v = Variable.Nationality

-- Define the set of all variables
def all_variables : Set Variable :=
  {Variable.Smoking, Variable.Gender, Variable.Religious_Belief, Variable.Nationality}

-- Define the set of categorical variables
def categorical_variables : Set Variable :=
  {v ∈ all_variables | is_categorical v}

-- The theorem to prove
theorem categorical_variables_are_correct :
  categorical_variables = {Variable.Gender, Variable.Religious_Belief, Variable.Nationality} :=
by sorry

end NUMINAMATH_CALUDE_categorical_variables_are_correct_l1836_183658


namespace NUMINAMATH_CALUDE_smallest_among_four_l1836_183675

theorem smallest_among_four (a b c d : ℚ) (h1 : a = -1) (h2 : b = 0) (h3 : c = 1) (h4 : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
  sorry

end NUMINAMATH_CALUDE_smallest_among_four_l1836_183675


namespace NUMINAMATH_CALUDE_exist_special_integers_l1836_183641

theorem exist_special_integers : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  ((7^7 : ℕ) ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) := by
  sorry

end NUMINAMATH_CALUDE_exist_special_integers_l1836_183641


namespace NUMINAMATH_CALUDE_unique_function_determination_l1836_183601

theorem unique_function_determination (f : ℝ → ℝ) 
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y - 1)) :
  ∀ x : ℝ, f x = x + 1 := by
sorry

end NUMINAMATH_CALUDE_unique_function_determination_l1836_183601


namespace NUMINAMATH_CALUDE_arrangements_equal_24_l1836_183680

/-- Represents the number of traditional Chinese paintings -/
def traditional_paintings : Nat := 3

/-- Represents the number of oil paintings -/
def oil_paintings : Nat := 2

/-- Represents the number of ink paintings -/
def ink_paintings : Nat := 1

/-- Calculates the number of arrangements for the paintings -/
def calculate_arrangements : Nat :=
  -- The actual calculation is not provided here
  -- It should consider the constraints mentioned in the problem
  sorry

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_equal_24 : calculate_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_equal_24_l1836_183680


namespace NUMINAMATH_CALUDE_stock_price_change_l1836_183624

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let day1_price := initial_price * (1 - 0.25)
  let day2_price := day1_price * (1 + 0.35)
  (day2_price - initial_price) / initial_price = 0.0125 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l1836_183624


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l1836_183693

/-- An ellipse with center at the origin, foci at (±√2, 0), intersected by the line y = x + 1
    such that the x-coordinate of the midpoint of the chord is -2/3 -/
def special_ellipse (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (a^2 = b^2 + 2) ∧
  (∃ (x₁ x₂ y₁ y₂ : ℝ),
    (x₁^2 / a^2 + y₁^2 / b^2 = 1) ∧
    (x₂^2 / a^2 + y₂^2 / b^2 = 1) ∧
    (y₁ = x₁ + 1) ∧ (y₂ = x₂ + 1) ∧
    ((x₁ + x₂) / 2 = -2/3))

/-- The equation of the special ellipse is x²/4 + y²/2 = 1 -/
theorem special_ellipse_equation :
  ∀ x y : ℝ, special_ellipse x y ↔ x^2/4 + y^2/2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l1836_183693


namespace NUMINAMATH_CALUDE_brennan_pepper_proof_l1836_183656

/-- The amount of pepper Brennan used (in grams) -/
def pepper_used : ℝ := 0.16

/-- The amount of pepper Brennan has left (in grams) -/
def pepper_left : ℝ := 0.09

/-- The initial amount of pepper Brennan had (in grams) -/
def initial_pepper : ℝ := pepper_used + pepper_left

theorem brennan_pepper_proof : initial_pepper = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_brennan_pepper_proof_l1836_183656


namespace NUMINAMATH_CALUDE_ara_current_height_l1836_183651

/-- Represents a person's height and growth --/
structure Person where
  originalHeight : ℝ
  growthFactor : ℝ

/-- Calculates the current height of a person given their original height and growth factor --/
def currentHeight (p : Person) : ℝ := p.originalHeight * (1 + p.growthFactor)

/-- Theorem stating Ara's current height given the conditions --/
theorem ara_current_height (shea ara : Person) 
  (h1 : shea.growthFactor = 0.25)
  (h2 : currentHeight shea = 75)
  (h3 : ara.originalHeight = shea.originalHeight)
  (h4 : ara.growthFactor = shea.growthFactor / 3) :
  currentHeight ara = 65 := by
  sorry


end NUMINAMATH_CALUDE_ara_current_height_l1836_183651


namespace NUMINAMATH_CALUDE_triangle_inequality_generalization_l1836_183616

theorem triangle_inequality_generalization (x y z : ℝ) :
  (|x - y| + |y - z| + |z - x| ≤ 2 * Real.sqrt 2 * Real.sqrt (x^2 + y^2 + z^2)) ∧
  ((0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) → |x - y| + |y - z| + |z - x| ≤ 2 * Real.sqrt (x^2 + y^2 + z^2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_generalization_l1836_183616


namespace NUMINAMATH_CALUDE_orange_pricing_and_purchase_l1836_183627

-- Define variables
variable (x y m : ℝ)

-- Define the theorem
theorem orange_pricing_and_purchase :
  -- Conditions
  (3 * x + 2 * y = 78) →
  (2 * x + 3 * y = 72) →
  (18 * m + 12 * (100 - m) ≤ 1440) →
  (m ≤ 100) →
  -- Conclusions
  (x = 18 ∧ y = 12) ∧
  (∀ n, n ≤ 100 ∧ 18 * n + 12 * (100 - n) ≤ 1440 → n ≤ 40) :=
by sorry

end NUMINAMATH_CALUDE_orange_pricing_and_purchase_l1836_183627


namespace NUMINAMATH_CALUDE_exam_average_score_l1836_183668

theorem exam_average_score (max_score : ℕ) (amar_percent bhavan_percent chetan_percent deepak_percent : ℚ) :
  max_score = 1100 →
  amar_percent = 64 / 100 →
  bhavan_percent = 36 / 100 →
  chetan_percent = 44 / 100 →
  deepak_percent = 52 / 100 →
  let amar_score := (amar_percent * max_score : ℚ).floor
  let bhavan_score := (bhavan_percent * max_score : ℚ).floor
  let chetan_score := (chetan_percent * max_score : ℚ).floor
  let deepak_score := (deepak_percent * max_score : ℚ).floor
  let total_score := amar_score + bhavan_score + chetan_score + deepak_score
  (total_score / 4 : ℚ).floor = 539 := by
  sorry

#eval (64 / 100 : ℚ) * 1100  -- Expected output: 704
#eval (36 / 100 : ℚ) * 1100  -- Expected output: 396
#eval (44 / 100 : ℚ) * 1100  -- Expected output: 484
#eval (52 / 100 : ℚ) * 1100  -- Expected output: 572
#eval ((704 + 396 + 484 + 572) / 4 : ℚ)  -- Expected output: 539

end NUMINAMATH_CALUDE_exam_average_score_l1836_183668


namespace NUMINAMATH_CALUDE_no_palindromes_with_two_fives_l1836_183640

def isPalindrome (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 2000 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 ≠ (n / 10) % 10)

def hasTwoFives (n : ℕ) : Prop :=
  (n / 1000 = 5) ∨ ((n / 100) % 10 = 5) ∨ ((n / 10) % 10 = 5) ∨ (n % 10 = 5)

theorem no_palindromes_with_two_fives :
  ¬∃ n : ℕ, isPalindrome n ∧ hasTwoFives n :=
sorry

end NUMINAMATH_CALUDE_no_palindromes_with_two_fives_l1836_183640


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1836_183686

-- Define the ellipse C
def C (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the points
def O : ℝ × ℝ := (0, 0)
def F (c : ℝ) : ℝ × ℝ := (c, 0)
def A (a : ℝ) : ℝ × ℝ := (-a, 0)
def B (a : ℝ) : ℝ × ℝ := (a, 0)
def P (x y : ℝ) : ℝ × ℝ := (x, y)
def M (x y : ℝ) : ℝ × ℝ := (x, y)
def E (y : ℝ) : ℝ × ℝ := (0, y)

-- Define the lines
def line_l (a c : ℝ) (x y : ℝ) : Prop := 
  ∃ (y_M : ℝ), y - y_M = (y_M / (c + a)) * (x - c)

def line_BM (a c : ℝ) (x y y_E : ℝ) : Prop := 
  y - y_E/2 = -(y_E/2) / (a + c) * x

-- Main theorem
theorem ellipse_eccentricity 
  (a b c : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : c < a)
  (h3 : ∃ (x y : ℝ), C a b x y ∧ P x y = (x, y) ∧ x = c)
  (h4 : ∃ (x y y_E : ℝ), line_l a c x y ∧ line_BM a c x y y_E ∧ M x y = (x, y))
  : c/a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1836_183686


namespace NUMINAMATH_CALUDE_inequality_proof_l1836_183654

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ 
  4 * Real.sqrt ((a*b / (a+b))^2 + (a*b / (a+b)) * (a*c / (a+c)) + (a*c / (a+c))^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1836_183654


namespace NUMINAMATH_CALUDE_exceed_permutations_l1836_183689

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def word_length : ℕ := 6
def repeated_letter_count : ℕ := 3

theorem exceed_permutations :
  factorial word_length / factorial repeated_letter_count = 120 := by
  sorry

end NUMINAMATH_CALUDE_exceed_permutations_l1836_183689


namespace NUMINAMATH_CALUDE_complement_of_A_l1836_183697

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1, 2}

theorem complement_of_A : (U \ A) = {0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1836_183697


namespace NUMINAMATH_CALUDE_ellipse_constant_product_l1836_183688

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line passing through (1, 0) with slope k -/
def line (x y k : ℝ) : Prop := y = k * (x - 1)

/-- The dot product of vectors PE and QE -/
def dot_product (xP yP xQ yQ xE : ℝ) : ℝ :=
  (xE - xP) * (xE - xQ) + (-yP) * (-yQ)

theorem ellipse_constant_product :
  ∀ (xP yP xQ yQ k : ℝ),
    ellipse xP yP →
    ellipse xQ yQ →
    line xP yP k →
    line xQ yQ k →
    xP ≠ xQ →
    dot_product xP yP xQ yQ (17/8) = 33/64 := by sorry

end NUMINAMATH_CALUDE_ellipse_constant_product_l1836_183688


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1836_183605

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (m₁ n₁ m₂ n₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁

/-- Line l₁ with equation x + ay + 6 = 0 -/
def l₁ (a : ℝ) (x y : ℝ) : Prop :=
  x + a * y + 6 = 0

/-- Line l₂ with equation (a-2)x + 3ay + 18 = 0 -/
def l₂ (a : ℝ) (x y : ℝ) : Prop :=
  (a - 2) * x + 3 * a * y + 18 = 0

/-- The main theorem stating that when l₁ and l₂ are parallel, a = 0 -/
theorem parallel_lines_a_value :
  ∀ a : ℝ, parallel_lines 1 a (a - 2) (3 * a) → a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1836_183605


namespace NUMINAMATH_CALUDE_a_nine_equals_a_three_times_a_seven_l1836_183633

def exponential_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = q ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem a_nine_equals_a_three_times_a_seven
  (a : ℕ → ℝ) (q : ℝ) (h : exponential_sequence a q) :
  a 9 = a 3 * a 7 := by
  sorry

end NUMINAMATH_CALUDE_a_nine_equals_a_three_times_a_seven_l1836_183633


namespace NUMINAMATH_CALUDE_rational_number_problems_l1836_183620

theorem rational_number_problems :
  (∀ (a b : ℚ), a * b = -2 ∧ a = 1/7 → b = -14) ∧
  (∀ (x y z : ℚ), x + y + z = -5 ∧ x = 1 ∧ y = -4 → z = -2) := by sorry

end NUMINAMATH_CALUDE_rational_number_problems_l1836_183620


namespace NUMINAMATH_CALUDE_origin_outside_circle_l1836_183691

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) :
  let circle := fun (x y : ℝ) => x^2 + y^2 + 2*a*x + 2*y + (a-1)^2 = 0
  ¬ circle 0 0 := by
  sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l1836_183691


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l1836_183684

theorem complex_modulus_theorem (t : ℝ) (i : ℂ) (h_i : i^2 = -1) :
  let z : ℂ := (1 - t*i) / (1 + i)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs (Real.sqrt 3 + t*i) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l1836_183684


namespace NUMINAMATH_CALUDE_remainder_3572_div_49_l1836_183698

theorem remainder_3572_div_49 : 3572 % 49 = 44 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3572_div_49_l1836_183698


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l1836_183621

theorem square_perimeter_sum (a b : ℝ) (h1 : a + b = 85) (h2 : a - b = 41) :
  4 * (Real.sqrt a.toNNReal + Real.sqrt b.toNNReal) = 4 * (Real.sqrt 63 + Real.sqrt 22) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l1836_183621
