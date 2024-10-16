import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3385_338576

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 - 10 = 188 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3385_338576


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_l3385_338557

def Bag := Fin 4

def is_black : Bag → Prop :=
  fun b => b.val < 2

def Draw := Fin 2 → Bag

def exactly_one_black (draw : Draw) : Prop :=
  (is_black (draw 0) ∧ ¬is_black (draw 1)) ∨ (¬is_black (draw 0) ∧ is_black (draw 1))

def exactly_two_black (draw : Draw) : Prop :=
  is_black (draw 0) ∧ is_black (draw 1)

theorem mutually_exclusive_not_opposite :
  (∃ (draw : Draw), exactly_one_black draw) ∧
  (∃ (draw : Draw), exactly_two_black draw) ∧
  (¬∃ (draw : Draw), exactly_one_black draw ∧ exactly_two_black draw) ∧
  (∃ (draw : Draw), ¬exactly_one_black draw ∧ ¬exactly_two_black draw) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_l3385_338557


namespace NUMINAMATH_CALUDE_square_of_three_tenths_plus_one_tenth_l3385_338571

theorem square_of_three_tenths_plus_one_tenth (ε : Real) :
  (0.3 : Real)^2 + 0.1 = 0.19 := by
  sorry

end NUMINAMATH_CALUDE_square_of_three_tenths_plus_one_tenth_l3385_338571


namespace NUMINAMATH_CALUDE_david_age_is_seven_l3385_338597

/-- David's age in years -/
def david_age : ℕ := 7

/-- Yuan's age in years -/
def yuan_age : ℕ := 2 * david_age

theorem david_age_is_seven :
  (yuan_age = david_age + 7) ∧ (yuan_age = 2 * david_age) → david_age = 7 := by
  sorry

end NUMINAMATH_CALUDE_david_age_is_seven_l3385_338597


namespace NUMINAMATH_CALUDE_quadruple_equation_solutions_l3385_338584

theorem quadruple_equation_solutions :
  {q : ℕ × ℕ × ℕ × ℕ | let (x, y, z, n) := q; x^2 + y^2 + z^2 + 1 = 2^n} =
  {(0,0,0,0), (1,0,0,1), (0,1,0,1), (0,0,1,1), (1,1,1,2)} := by
  sorry

end NUMINAMATH_CALUDE_quadruple_equation_solutions_l3385_338584


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l3385_338548

theorem log_sum_equals_two : Real.log 0.01 / Real.log 10 + Real.log 16 / Real.log 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l3385_338548


namespace NUMINAMATH_CALUDE_salary_changes_l3385_338593

theorem salary_changes (S : ℝ) (S_pos : S > 0) :
  S * (1 + 0.3) * (1 - 0.2) * (1 + 0.1) * (1 - 0.25) = S * 1.04 := by
  sorry

end NUMINAMATH_CALUDE_salary_changes_l3385_338593


namespace NUMINAMATH_CALUDE_xyz_sum_of_squares_l3385_338522

theorem xyz_sum_of_squares (x y z : ℝ) 
  (h1 : (2*x + 2*y + 3*z) / 7 = 9)
  (h2 : (x^2 * y^2 * z^3)^(1/7) = 6)
  (h3 : 7 / ((2/x) + (2/y) + (3/z)) = 4) :
  x^2 + y^2 + z^2 = 351 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_of_squares_l3385_338522


namespace NUMINAMATH_CALUDE_sons_age_l3385_338509

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 27 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 25 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3385_338509


namespace NUMINAMATH_CALUDE_max_female_students_with_4_teachers_min_total_people_l3385_338533

/-- Represents a study group composition --/
structure StudyGroup where
  male_students : ℕ
  female_students : ℕ
  teachers : ℕ

/-- Checks if a study group satisfies the given conditions --/
def is_valid_group (g : StudyGroup) : Prop :=
  g.male_students > g.female_students ∧
  g.female_students > g.teachers ∧
  2 * g.teachers > g.male_students

/-- The maximum number of female students when there are 4 teachers is 6 --/
theorem max_female_students_with_4_teachers :
  ∀ g : StudyGroup, is_valid_group g → g.teachers = 4 → g.female_students ≤ 6 := by
  sorry

/-- The minimum number of people in a valid study group is 12 --/
theorem min_total_people :
  ∀ g : StudyGroup, is_valid_group g →
    g.male_students + g.female_students + g.teachers ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_max_female_students_with_4_teachers_min_total_people_l3385_338533


namespace NUMINAMATH_CALUDE_inscribed_circles_area_ratio_l3385_338583

theorem inscribed_circles_area_ratio : 
  ∀ (R r : ℝ), R > 0 → r > 0 →
  (∃ (s : ℝ), s > 0 ∧ R = (s * Real.sqrt 2) / 2 ∧ r = s / 2) →
  (π * r^2) / (π * R^2) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circles_area_ratio_l3385_338583


namespace NUMINAMATH_CALUDE_l_shape_area_is_52_l3385_338511

/-- The area of an 'L' shaped figure formed from a rectangle with given dimensions,
    after subtracting a corner rectangle and an inner rectangle. -/
def l_shape_area (large_length large_width corner_length corner_width inner_length inner_width : ℕ) : ℕ :=
  large_length * large_width - (corner_length * corner_width + inner_length * inner_width)

/-- Theorem stating that the area of the specific 'L' shaped figure is 52 square units. -/
theorem l_shape_area_is_52 :
  l_shape_area 10 6 3 2 2 1 = 52 := by
  sorry

end NUMINAMATH_CALUDE_l_shape_area_is_52_l3385_338511


namespace NUMINAMATH_CALUDE_square_difference_l3385_338547

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : 
  (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3385_338547


namespace NUMINAMATH_CALUDE_inequality_sequence_properties_l3385_338535

/-- Definition of the nth inequality in the sequence -/
def nth_inequality (n : ℕ+) (x : ℝ) : Prop :=
  x + (2*n*(2*n-1))/x < 4*n - 1

/-- Definition of the solution set for the nth inequality -/
def nth_solution_set (n : ℕ+) (x : ℝ) : Prop :=
  (2*n - 1 : ℝ) < x ∧ x < 2*n

/-- Definition of the special inequality with parameter a -/
def special_inequality (a : ℕ+) (x : ℝ) : Prop :=
  x + (12*a)/(x+1) < 4*a + 2

/-- Definition of the solution set for the special inequality -/
def special_solution_set (a : ℕ+) (x : ℝ) : Prop :=
  2 < x ∧ x < 4*a - 1

/-- Main theorem statement -/
theorem inequality_sequence_properties :
  ∀ (n : ℕ+),
    (∀ (x : ℝ), nth_inequality n x ↔ nth_solution_set n x) ∧
    (∀ (a : ℕ+) (x : ℝ), special_inequality a x ↔ special_solution_set a x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_sequence_properties_l3385_338535


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l3385_338568

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes (m₁ m₂ : ℝ) : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, (y = m₁ * x + b₁ ↔ y = m₂ * x + b₂)) ↔ m₁ = m₂

/-- The problem statement -/
theorem parallel_lines_k_value :
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, (y = 5 * x - 3 ↔ y = 3 * k * x + 7)) → k = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l3385_338568


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l3385_338545

theorem profit_percent_calculation (selling_price : ℝ) (cost_price : ℝ) 
  (h : cost_price = 0.9 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = (1 / 9) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l3385_338545


namespace NUMINAMATH_CALUDE_distance_between_points_l3385_338553

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-4, 2)
  let p2 : ℝ × ℝ := (3, 6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 65 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3385_338553


namespace NUMINAMATH_CALUDE_cylinder_dimensions_l3385_338549

/-- Proves that a cylinder with equal surface area to a sphere of radius 4 cm
    and with equal height and diameter has height and diameter of 8 cm. -/
theorem cylinder_dimensions (r : ℝ) (h : ℝ) :
  r = 4 →  -- radius of the sphere is 4 cm
  (4 * π * r^2 : ℝ) = 2 * π * r * h →  -- surface areas are equal
  h = 2 * r →  -- height equals diameter
  h = 8 ∧ (2 * r) = 8 :=  -- height and diameter are both 8 cm
by sorry

end NUMINAMATH_CALUDE_cylinder_dimensions_l3385_338549


namespace NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l3385_338592

/-- The number of rectangles on a 4x4 grid with sides parallel to axes -/
def num_rectangles : ℕ := 36

/-- The number of ways to choose 2 items from 4 -/
def choose_two_from_four : ℕ := 6

theorem rectangles_on_4x4_grid :
  num_rectangles = choose_two_from_four * choose_two_from_four :=
by sorry

end NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l3385_338592


namespace NUMINAMATH_CALUDE_apple_boxes_l3385_338566

theorem apple_boxes (apples_per_crate : ℕ) (crates : ℕ) (rotten_apples : ℕ) (apples_per_box : ℕ) :
  apples_per_crate = 42 →
  crates = 12 →
  rotten_apples = 4 →
  apples_per_box = 10 →
  (crates * apples_per_crate - rotten_apples) / apples_per_box = 50 := by
  sorry

end NUMINAMATH_CALUDE_apple_boxes_l3385_338566


namespace NUMINAMATH_CALUDE_divisor_with_remainder_54_l3385_338537

theorem divisor_with_remainder_54 :
  ∃ (n : ℕ), n > 0 ∧ (55^55 + 55) % n = 54 ∧ n = 56 := by sorry

end NUMINAMATH_CALUDE_divisor_with_remainder_54_l3385_338537


namespace NUMINAMATH_CALUDE_b_share_is_1500_l3385_338579

/-- Calculates the share of the second child (B) when distributing money among three children in a given ratio -/
def calculate_b_share (total_money : ℚ) (ratio_a ratio_b ratio_c : ℕ) : ℚ :=
  let total_parts := ratio_a + ratio_b + ratio_c
  let part_value := total_money / total_parts
  ratio_b * part_value

/-- Theorem stating that given $4500 distributed in the ratio 2:3:4, B's share is $1500 -/
theorem b_share_is_1500 :
  calculate_b_share 4500 2 3 4 = 1500 := by
  sorry

#eval calculate_b_share 4500 2 3 4

end NUMINAMATH_CALUDE_b_share_is_1500_l3385_338579


namespace NUMINAMATH_CALUDE_ring_stack_height_58cm_l3385_338582

def ring_stack_height (top_diameter : ℕ) (diameter_decrease : ℕ) (min_diameter : ℕ) (ring_thickness : ℕ) : ℕ :=
  let num_rings := (top_diameter - min_diameter) / diameter_decrease + 1
  let sum_inside_diameters := num_rings * (top_diameter - 1 + (top_diameter - 1 - (num_rings - 1) * diameter_decrease)) / 2
  sum_inside_diameters + 2 * ring_thickness

theorem ring_stack_height_58cm :
  ring_stack_height 15 2 3 1 = 58 := by
  sorry

end NUMINAMATH_CALUDE_ring_stack_height_58cm_l3385_338582


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3385_338565

/-- 
If the quadratic equation 2x^2 - 5x + m = 0 has two equal real roots,
then m = 25/8.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - 5 * x + m = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - 5 * y + m = 0 → y = x) → 
  m = 25/8 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3385_338565


namespace NUMINAMATH_CALUDE_orange_marbles_count_l3385_338516

def total_marbles : ℕ := 24
def blue_marbles : ℕ := total_marbles / 2
def red_marbles : ℕ := 6

theorem orange_marbles_count : total_marbles - blue_marbles - red_marbles = 6 := by
  sorry

end NUMINAMATH_CALUDE_orange_marbles_count_l3385_338516


namespace NUMINAMATH_CALUDE_problem_solution_l3385_338563

def f (x : ℝ) := |2 * x - 1|

def g (x : ℝ) := f x + f (x - 1)

theorem problem_solution :
  (∀ x : ℝ, f x < 4 ↔ -3/2 < x ∧ x < 5/2) ∧
  (∃ a : ℝ, (∀ x : ℝ, g x ≥ a) ∧
    ∀ m n : ℝ, m > 0 → n > 0 → m + n = a →
      2/m + 1/n ≥ 3/2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3385_338563


namespace NUMINAMATH_CALUDE_root_conditions_imply_relation_l3385_338518

/-- Given two equations with specific root conditions, prove a relation between constants c and d -/
theorem root_conditions_imply_relation (c d : ℝ) : 
  (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    ((r₁ + c) * (r₁ + d) * (r₁ - 7)) / ((r₁ + 4)^2) = 0 ∧
    ((r₂ + c) * (r₂ + d) * (r₂ - 7)) / ((r₂ + 4)^2) = 0 ∧
    ((r₃ + c) * (r₃ + d) * (r₃ - 7)) / ((r₃ + 4)^2) = 0) →
  (∃! (r : ℝ), ((r + 2*c) * (r + 5) * (r + 8)) / ((r + d) * (r - 7)) = 0) →
  100 * c + d = 408 := by
sorry

end NUMINAMATH_CALUDE_root_conditions_imply_relation_l3385_338518


namespace NUMINAMATH_CALUDE_clock_strikes_in_day_l3385_338512

def clock_strikes (hour : Nat) : Nat :=
  if hour ≤ 12 then hour else hour - 12

def total_strikes : Nat :=
  (List.range 24).map clock_strikes |> List.sum

theorem clock_strikes_in_day : total_strikes = 156 := by
  sorry

end NUMINAMATH_CALUDE_clock_strikes_in_day_l3385_338512


namespace NUMINAMATH_CALUDE_second_month_bill_l3385_338595

/-- Represents Elvin's telephone bill structure -/
structure TelephoneBill where
  internetCharge : ℝ
  callCharge : ℝ

/-- Calculates the total bill given internet and call charges -/
def totalBill (bill : TelephoneBill) : ℝ :=
  bill.internetCharge + bill.callCharge

theorem second_month_bill 
  (januaryBill : TelephoneBill) 
  (h1 : totalBill januaryBill = 40) 
  (secondMonthBill : TelephoneBill) 
  (h2 : secondMonthBill.internetCharge = januaryBill.internetCharge)
  (h3 : secondMonthBill.callCharge = 2 * januaryBill.callCharge) :
  totalBill secondMonthBill = 40 + januaryBill.callCharge := by
  sorry

#check second_month_bill

end NUMINAMATH_CALUDE_second_month_bill_l3385_338595


namespace NUMINAMATH_CALUDE_area_transformation_l3385_338524

-- Define a function representing the area between a curve and the x-axis
noncomputable def area_between_curve_and_x_axis (f : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem area_transformation (g : ℝ → ℝ) 
  (h : area_between_curve_and_x_axis g = 8) : 
  area_between_curve_and_x_axis (λ x => 4 * g (x + 3)) = 32 := by
  sorry

end NUMINAMATH_CALUDE_area_transformation_l3385_338524


namespace NUMINAMATH_CALUDE_smallest_dance_class_size_l3385_338543

theorem smallest_dance_class_size :
  ∀ n : ℕ,
  n > 40 →
  (∀ m : ℕ, m > 40 ∧ 5 * m + 2 < 5 * n + 2 → m = n) →
  5 * n + 2 = 207 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_dance_class_size_l3385_338543


namespace NUMINAMATH_CALUDE_distance_between_given_lines_l3385_338561

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ × ℝ  -- Point on first line
  b : ℝ × ℝ  -- Point on second line
  d : ℝ × ℝ  -- Direction vector (same for both lines)

/-- The distance between two parallel lines -/
def distance (lines : ParallelLines) : ℝ :=
  sorry

/-- Theorem stating that the distance between the given parallel lines is 0 -/
theorem distance_between_given_lines :
  let lines : ParallelLines := {
    a := (3, -4),
    b := (2, -1),
    d := (-1, 3)
  }
  distance lines = 0 := by sorry

end NUMINAMATH_CALUDE_distance_between_given_lines_l3385_338561


namespace NUMINAMATH_CALUDE_flagpole_height_l3385_338532

/-- Given a person and a flagpole under the same lighting conditions, 
    we can determine the height of the flagpole using the ratio of heights to shadow lengths. -/
theorem flagpole_height
  (person_height : ℝ)
  (person_shadow : ℝ)
  (flagpole_shadow : ℝ)
  (h_person_height : person_height = 1.6)
  (h_person_shadow : person_shadow = 0.4)
  (h_flagpole_shadow : flagpole_shadow = 5)
  (h_positive : person_height > 0 ∧ person_shadow > 0 ∧ flagpole_shadow > 0) :
  (person_height / person_shadow) * flagpole_shadow = 20 :=
sorry

#check flagpole_height

end NUMINAMATH_CALUDE_flagpole_height_l3385_338532


namespace NUMINAMATH_CALUDE_village_population_problem_l3385_338578

theorem village_population_problem (original : ℕ) : 
  (original : ℝ) * 0.9 * 0.75 = 5130 → original = 7600 := by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l3385_338578


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3385_338588

/-- The distance from the focus to the directrix of a parabola y^2 = 8x is 4 -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y^2 = 8*x → 
  ∃ (f d : ℝ × ℝ), 
    (f.1 - d.1)^2 + (f.2 - d.2)^2 = 4^2 ∧
    (∀ (p : ℝ × ℝ), p.2^2 = 8*p.1 → 
      (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.1 - d.1)^2 + (p.2 - d.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3385_338588


namespace NUMINAMATH_CALUDE_M_mod_49_l3385_338534

/-- M is the 92-digit number formed by concatenating integers from 1 to 50 -/
def M : ℕ := sorry

/-- The sum of digits from 1 to 50 -/
def sum_digits : ℕ := (50 * (1 + 50)) / 2

theorem M_mod_49 : M % 49 = 18 := by sorry

end NUMINAMATH_CALUDE_M_mod_49_l3385_338534


namespace NUMINAMATH_CALUDE_complex_real_condition_l3385_338570

theorem complex_real_condition (z : ℂ) : (z + 2*I).im = 0 ↔ z.im = -2 := by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3385_338570


namespace NUMINAMATH_CALUDE_expression_evaluation_l3385_338586

theorem expression_evaluation (x : ℝ) (h : x = 1.25) :
  (3 * x^2 - 8 * x + 2) * (4 * x - 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3385_338586


namespace NUMINAMATH_CALUDE_rocket_max_altitude_l3385_338517

/-- The altitude function of the rocket -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 25

/-- Theorem: The maximum altitude reached by the rocket is 45 meters -/
theorem rocket_max_altitude :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 45 := by
  sorry

end NUMINAMATH_CALUDE_rocket_max_altitude_l3385_338517


namespace NUMINAMATH_CALUDE_product_first_two_terms_l3385_338515

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℝ
  /-- The common difference between consecutive terms -/
  d : ℝ
  /-- The seventh term of the sequence is 20 -/
  seventh_term : a₁ + 6 * d = 20
  /-- The common difference is 2 -/
  common_diff : d = 2

/-- The product of the first two terms of the arithmetic sequence is 80 -/
theorem product_first_two_terms (seq : ArithmeticSequence) :
  seq.a₁ * (seq.a₁ + seq.d) = 80 := by
  sorry


end NUMINAMATH_CALUDE_product_first_two_terms_l3385_338515


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l3385_338500

-- Define the conditions p and q
def p (a : ℝ) : Prop := a > 4
def q (a : ℝ) : Prop := 5 < a ∧ a < 6

-- Theorem stating that p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient :
  (∀ a : ℝ, q a → p a) ∧ 
  (∃ a : ℝ, p a ∧ ¬q a) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l3385_338500


namespace NUMINAMATH_CALUDE_equation_solutions_l3385_338531

theorem equation_solutions (a : ℝ) (h : a < 0) :
  ∃! (s : Finset ℝ), s.card = 4 ∧
  (∀ x ∈ s, -π < x ∧ x < π) ∧
  (∀ x ∈ s, (a - 1) * (Real.sin (2 * x) + Real.cos x) + (a + 1) * (Real.sin x - Real.cos (2 * x)) = 0) ∧
  (∀ x, -π < x ∧ x < π →
    (a - 1) * (Real.sin (2 * x) + Real.cos x) + (a + 1) * (Real.sin x - Real.cos (2 * x)) = 0 →
    x ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3385_338531


namespace NUMINAMATH_CALUDE_ceiling_sum_evaluation_l3385_338514

theorem ceiling_sum_evaluation : 
  ⌈Real.sqrt (16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)^2⌉ + ⌈(16/9 : ℝ)^(1/3)⌉ = 10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_evaluation_l3385_338514


namespace NUMINAMATH_CALUDE_range_of_a_open_interval_l3385_338501

theorem range_of_a_open_interval :
  (∃ a : ℝ, ∀ x : ℝ, x^2 - a*x + 1 > 0) ↔ ∃ a : ℝ, -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_open_interval_l3385_338501


namespace NUMINAMATH_CALUDE_literary_society_book_exchange_l3385_338559

/-- The number of books exchanged in a Literary Society book sharing ceremony -/
def books_exchanged (x : ℕ) : ℕ := x * (x - 1)

/-- The theorem stating that for a group of x students where each student gives one book to every
    other student, and a total of 210 books are exchanged, the equation x(x-1) = 210 holds -/
theorem literary_society_book_exchange (x : ℕ) (h : books_exchanged x = 210) :
  x * (x - 1) = 210 := by sorry

end NUMINAMATH_CALUDE_literary_society_book_exchange_l3385_338559


namespace NUMINAMATH_CALUDE_sum_of_squares_difference_l3385_338572

theorem sum_of_squares_difference (a b : ℕ+) (h : a.val^2 - b.val^4 = 2009) : 
  a.val + b.val = 47 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_difference_l3385_338572


namespace NUMINAMATH_CALUDE_optimal_choice_is_96_l3385_338574

def count_rectangles (perimeter : ℕ) : ℕ :=
  if perimeter % 2 = 0 then
    (perimeter / 2 - 1) / 2
  else
    0

theorem optimal_choice_is_96 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 97 → count_rectangles n ≤ count_rectangles 96 :=
by sorry

end NUMINAMATH_CALUDE_optimal_choice_is_96_l3385_338574


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_180_4620_l3385_338594

theorem gcd_lcm_sum_180_4620 : 
  Nat.gcd 180 4620 + Nat.lcm 180 4620 = 13920 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_180_4620_l3385_338594


namespace NUMINAMATH_CALUDE_stationery_sales_equation_l3385_338585

/-- Represents the sales equation for a stationery store during a promotional event. -/
theorem stationery_sales_equation (x : ℝ) : 
  (1.2 * 0.8 * x + 2 * 0.9 * (60 - x) = 87) ↔ 
  (x ≥ 0 ∧ x ≤ 60 ∧ 
   1.2 * (1 - 0.2) * x + 2 * (1 - 0.1) * (60 - x) = 87) := by
  sorry

#check stationery_sales_equation

end NUMINAMATH_CALUDE_stationery_sales_equation_l3385_338585


namespace NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_three_l3385_338544

theorem simplify_fraction_with_sqrt_three : 
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_three_l3385_338544


namespace NUMINAMATH_CALUDE_sphere_volume_after_drilling_l3385_338513

/-- The remaining volume of a sphere after drilling two cylindrical holes -/
theorem sphere_volume_after_drilling (sphere_diameter : ℝ) (hole1_depth hole1_diameter hole2_depth hole2_diameter : ℝ) : 
  sphere_diameter = 12 ∧ 
  hole1_depth = 5 ∧ 
  hole1_diameter = 1 ∧ 
  hole2_depth = 5 ∧ 
  hole2_diameter = 1.5 → 
  (4 / 3 * π * (sphere_diameter / 2)^3) - (π * (hole1_diameter / 2)^2 * hole1_depth) - (π * (hole2_diameter / 2)^2 * hole2_depth) = 283.9375 * π := by
  sorry

#check sphere_volume_after_drilling

end NUMINAMATH_CALUDE_sphere_volume_after_drilling_l3385_338513


namespace NUMINAMATH_CALUDE_problem_l3385_338591

def l₁ (x y : ℝ) : Prop := x - 2*y + 3 = 0

def l₂ (x y : ℝ) : Prop := 2*x + y + 3 = 0

def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d : ℝ, (∀ x y, f x y ↔ a*x + b*y = c) ∧
                 (∀ x y, g x y ↔ d*x - a*y = 0)

def p : Prop := ¬(perpendicular l₁ l₂)

def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + 2 > Real.exp x₀

theorem problem : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_l3385_338591


namespace NUMINAMATH_CALUDE_toy_store_revenue_l3385_338503

theorem toy_store_revenue (december : ℝ) (november : ℝ) (january : ℝ) 
  (h1 : november = (3/5) * december) 
  (h2 : january = (1/6) * november) : 
  december = (20/7) * ((november + january) / 2) := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_l3385_338503


namespace NUMINAMATH_CALUDE_solution_inequality_l3385_338587

theorem solution_inequality (x : ℝ) : 
  x > 0 → x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) ≥ 15 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_inequality_l3385_338587


namespace NUMINAMATH_CALUDE_right_triangle_sets_l3385_338581

theorem right_triangle_sets (a b c : ℝ) : 
  (((a = 3 ∧ b = 4 ∧ c = 5) ∨ 
    (a = 5 ∧ b = 12 ∧ c = 13) ∨ 
    (a = 1 ∧ b = 2 ∧ c = Real.sqrt 3)) → 
   a^2 + b^2 = c^2) ∧
  (a = 3 ∧ b = 5 ∧ c = 7 → a^2 + b^2 ≠ c^2) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l3385_338581


namespace NUMINAMATH_CALUDE_maggies_tractor_rate_l3385_338569

/-- Maggie's weekly income calculation --/
theorem maggies_tractor_rate (office_rate : ℝ) (office_hours tractor_hours total_income : ℝ) :
  office_rate = 10 →
  office_hours = 2 * tractor_hours →
  tractor_hours = 13 →
  total_income = 416 →
  total_income = office_rate * office_hours + tractor_hours * (total_income - office_rate * office_hours) / tractor_hours →
  (total_income - office_rate * office_hours) / tractor_hours = 12 := by
sorry


end NUMINAMATH_CALUDE_maggies_tractor_rate_l3385_338569


namespace NUMINAMATH_CALUDE_no_zero_roots_l3385_338523

-- Define the equations
def equation1 (x : ℝ) : Prop := 5 * x^2 - 15 = 35
def equation2 (x : ℝ) : Prop := (3*x-2)^2 = (2*x)^2
def equation3 (x : ℝ) : Prop := x^2 + 3*x - 4 = 2*x + 3

-- Theorem statement
theorem no_zero_roots :
  (∀ x : ℝ, equation1 x → x ≠ 0) ∧
  (∀ x : ℝ, equation2 x → x ≠ 0) ∧
  (∀ x : ℝ, equation3 x → x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_no_zero_roots_l3385_338523


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3385_338598

theorem quadratic_equation_solution :
  let a : ℝ := 2
  let b : ℝ := -6
  let c : ℝ := 1
  let x₁ : ℝ := (3 + Real.sqrt 7) / 2
  let x₂ : ℝ := (3 - Real.sqrt 7) / 2
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3385_338598


namespace NUMINAMATH_CALUDE_gcd_g_x_is_20_l3385_338527

def g (x : ℤ) : ℤ := (3*x + 4)*(8*x + 5)*(15*x + 11)*(x + 17)

theorem gcd_g_x_is_20 (x : ℤ) (h : 34560 ∣ x) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 20 := by
sorry

end NUMINAMATH_CALUDE_gcd_g_x_is_20_l3385_338527


namespace NUMINAMATH_CALUDE_prob_gpa_at_least_3_75_l3385_338542

/-- Grade points for each letter grade -/
def gradePoints : Char → ℕ
| 'A' => 4
| 'B' => 3
| 'C' => 2
| 'D' => 1
| _ => 0

/-- Calculate GPA from total points -/
def calculateGPA (totalPoints : ℕ) : ℚ :=
  totalPoints / 4

/-- Probability of getting an A in English -/
def probAEnglish : ℚ := 1 / 3

/-- Probability of getting a B in English -/
def probBEnglish : ℚ := 1 / 2

/-- Probability of getting an A in History -/
def probAHistory : ℚ := 1 / 5

/-- Probability of getting a B in History -/
def probBHistory : ℚ := 1 / 2

theorem prob_gpa_at_least_3_75 :
  let mathPoints := gradePoints 'B'
  let sciencePoints := gradePoints 'B'
  let totalFixedPoints := mathPoints + sciencePoints
  let requiredPoints := 15
  let probBothA := probAEnglish * probAHistory
  probBothA = 1 / 15 ∧
  (∀ (englishGrade historyGrade : Char),
    calculateGPA (totalFixedPoints + gradePoints englishGrade + gradePoints historyGrade) ≥ 15 / 4 →
    (englishGrade = 'A' ∧ historyGrade = 'A')) →
  probBothA = 1 / 15 := by
sorry

end NUMINAMATH_CALUDE_prob_gpa_at_least_3_75_l3385_338542


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3385_338577

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 130 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 245 := by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l3385_338577


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3385_338556

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 - a 5 + a 9 - a 13 + a 17 = 117 →
  a 3 + a 15 = 234 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3385_338556


namespace NUMINAMATH_CALUDE_line_inclination_45_degrees_l3385_338536

theorem line_inclination_45_degrees (a : ℝ) : 
  (∃ (x y : ℝ), ax + (2*a - 3)*y = 0) →   -- Line equation
  (Real.arctan (-a / (2*a - 3)) = π/4) →  -- 45° inclination
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_line_inclination_45_degrees_l3385_338536


namespace NUMINAMATH_CALUDE_jill_second_bus_ride_time_l3385_338550

/-- The time Jill spends waiting for her first bus, in minutes -/
def first_bus_wait : ℕ := 12

/-- The time Jill spends riding on her first bus, in minutes -/
def first_bus_ride : ℕ := 30

/-- The total time Jill spends on her first bus (waiting and riding), in minutes -/
def first_bus_total : ℕ := first_bus_wait + first_bus_ride

/-- The time Jill spends on her second bus ride, in minutes -/
def second_bus_ride : ℕ := first_bus_total / 2

theorem jill_second_bus_ride_time :
  second_bus_ride = 21 := by sorry

end NUMINAMATH_CALUDE_jill_second_bus_ride_time_l3385_338550


namespace NUMINAMATH_CALUDE_line_through_point_l3385_338505

/-- The value of k for which the line -3/4 - 3kx = 7y passes through (1/3, -8) -/
theorem line_through_point (k : ℝ) : 
  (-3/4 : ℝ) - 3 * k * (1/3) = 7 * (-8) → k = 55.25 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l3385_338505


namespace NUMINAMATH_CALUDE_f_is_bitwise_or_l3385_338502

/-- Bitwise OR operation for positive integers -/
def bitwiseOr (a b : ℕ+) : ℕ+ := sorry

/-- The function f we want to prove is equal to bitwise OR -/
noncomputable def f : ℕ+ → ℕ+ → ℕ+ := sorry

/-- Condition (i): f(a,b) ≤ a + b for all a, b ∈ ℤ⁺ -/
axiom condition_i (a b : ℕ+) : f a b ≤ a + b

/-- Condition (ii): f(a,f(b,c)) = f(f(a,b),c) for all a, b, c ∈ ℤ⁺ -/
axiom condition_ii (a b c : ℕ+) : f a (f b c) = f (f a b) c

/-- Condition (iii): Both (f(a,b) choose a) and (f(a,b) choose b) are odd numbers for all a, b ∈ ℤ⁺ -/
axiom condition_iii (a b : ℕ+) : Odd (Nat.choose (f a b) a) ∧ Odd (Nat.choose (f a b) b)

/-- f is surjective -/
axiom f_surjective : Function.Surjective f

/-- The main theorem: f is equal to bitwise OR -/
theorem f_is_bitwise_or : ∀ (a b : ℕ+), f a b = bitwiseOr a b := by sorry

end NUMINAMATH_CALUDE_f_is_bitwise_or_l3385_338502


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l3385_338526

theorem geometric_sequence_formula (a : ℕ → ℝ) (q : ℝ) (h1 : q < 1) 
  (h2 : a 2 + a 4 = 5/8) (h3 : a 3 = 1/4) 
  (h4 : ∀ n : ℕ, a (n + 1) = q * a n) : 
  ∀ n : ℕ, a n = (1/2)^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l3385_338526


namespace NUMINAMATH_CALUDE_urgent_painting_time_l3385_338562

/-- Represents the time required to paint an office -/
structure PaintingTime where
  painters : ℕ
  days : ℚ

/-- Represents the total work required to paint an office -/
def totalWork (pt : PaintingTime) : ℚ := pt.painters * pt.days

theorem urgent_painting_time 
  (first_office : PaintingTime)
  (second_office_normal : PaintingTime)
  (second_office_urgent : PaintingTime)
  (h1 : first_office.painters = 3)
  (h2 : first_office.days = 2)
  (h3 : second_office_normal.painters = 2)
  (h4 : totalWork first_office = totalWork second_office_normal)
  (h5 : second_office_urgent.painters = second_office_normal.painters)
  (h6 : second_office_urgent.days = 3/4 * second_office_normal.days) :
  second_office_urgent.days = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_urgent_painting_time_l3385_338562


namespace NUMINAMATH_CALUDE_circle_chord_tangent_relation_l3385_338521

/-- Given a circle with diameter AB and radius r, chord BF extended to meet
    the tangent at A at point C, and point E on BC extended such that BE = DC,
    prove that h = √(r² - d²), where d is the distance from E to the tangent at B
    and h is the distance from E to the diameter AB. -/
theorem circle_chord_tangent_relation (r d h : ℝ) : h = Real.sqrt (r^2 - d^2) :=
sorry

end NUMINAMATH_CALUDE_circle_chord_tangent_relation_l3385_338521


namespace NUMINAMATH_CALUDE_fraction_simplification_l3385_338599

theorem fraction_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : y + 1/x ≠ 0) :
  (x + 1/y) / (y + 1/x) = x / y := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3385_338599


namespace NUMINAMATH_CALUDE_cades_remaining_marbles_l3385_338580

/-- Represents the number of marbles Cade has left after giving some away -/
def marblesLeft (initial : Nat) (givenAway : Nat) : Nat :=
  initial - givenAway

/-- Theorem stating that Cade has 79 marbles left -/
theorem cades_remaining_marbles :
  marblesLeft 87 8 = 79 := by
  sorry

end NUMINAMATH_CALUDE_cades_remaining_marbles_l3385_338580


namespace NUMINAMATH_CALUDE_half_abs_diff_cubes_20_15_l3385_338590

theorem half_abs_diff_cubes_20_15 : 
  (1/2 : ℝ) * |20^3 - 15^3| = 2312.5 := by sorry

end NUMINAMATH_CALUDE_half_abs_diff_cubes_20_15_l3385_338590


namespace NUMINAMATH_CALUDE_square_park_circumference_l3385_338567

/-- The circumference of a square park with side length 5 kilometers is 20 kilometers. -/
theorem square_park_circumference :
  ∀ (side_length circumference : ℝ),
  side_length = 5 →
  circumference = 4 * side_length →
  circumference = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_square_park_circumference_l3385_338567


namespace NUMINAMATH_CALUDE_baking_difference_l3385_338519

/-- Given a recipe and current state of baking, calculate the difference between
    remaining sugar and flour to be added. -/
def sugar_flour_difference (total_flour total_sugar added_flour : ℕ) : ℕ :=
  total_sugar - (total_flour - added_flour)

/-- Theorem stating the difference between remaining sugar and flour to be added
    for the given recipe and current state. -/
theorem baking_difference : sugar_flour_difference 9 11 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_baking_difference_l3385_338519


namespace NUMINAMATH_CALUDE_trapezoid_longer_side_length_l3385_338530

/-- Given a square of side length s divided into a pentagon and three congruent trapezoids,
    if all four shapes have equal area, then the length of the longer parallel side
    of each trapezoid is s/2 -/
theorem trapezoid_longer_side_length (s : ℝ) (s_pos : s > 0) :
  let square_area := s^2
  let shape_area := square_area / 4
  let trapezoid_height := s / 2
  ∃ x : ℝ,
    x > 0 ∧
    x < s ∧
    shape_area = (x + s/2) * trapezoid_height / 2 ∧
    x = s / 2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_longer_side_length_l3385_338530


namespace NUMINAMATH_CALUDE_mathopolis_intersections_l3385_338596

/-- A city with a grid-like street layout -/
structure City where
  ns_streets : ℕ  -- number of north-south streets
  ew_streets : ℕ  -- number of east-west streets

/-- The number of intersections in a city with a grid-like street layout -/
def num_intersections (c : City) : ℕ := c.ns_streets * c.ew_streets

/-- Mathopolis with its specific street layout -/
def mathopolis : City := { ns_streets := 10, ew_streets := 10 }

/-- Theorem: The number of intersections in Mathopolis is 100 -/
theorem mathopolis_intersections : num_intersections mathopolis = 100 := by
  sorry

end NUMINAMATH_CALUDE_mathopolis_intersections_l3385_338596


namespace NUMINAMATH_CALUDE_probability_of_two_in_pascal_triangle_l3385_338589

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) :=
  sorry

/-- Counts the occurrences of a specific number in Pascal's Triangle -/
def countOccurrences (triangle : List (List ℕ)) (target : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of elements in Pascal's Triangle -/
def totalElements (triangle : List (List ℕ)) : ℕ :=
  sorry

/-- The main theorem: probability of selecting 2 from first 20 rows of Pascal's Triangle -/
theorem probability_of_two_in_pascal_triangle :
  let triangle := PascalTriangle 20
  let occurrences := countOccurrences triangle 2
  let total := totalElements triangle
  (occurrences : ℚ) / total = 6 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_two_in_pascal_triangle_l3385_338589


namespace NUMINAMATH_CALUDE_prob_13_11_is_quarter_l3385_338551

/-- Represents a table tennis game with specific scoring probabilities -/
structure TableTennisGame where
  /-- Probability of player A scoring when A serves -/
  prob_a_scores_on_a_serve : ℝ
  /-- Probability of player A scoring when B serves -/
  prob_a_scores_on_b_serve : ℝ

/-- Calculates the probability of reaching a 13:11 score from a 10:10 tie -/
def prob_13_11 (game : TableTennisGame) : ℝ :=
  sorry

/-- The main theorem stating the probability of reaching 13:11 is 1/4 -/
theorem prob_13_11_is_quarter (game : TableTennisGame) 
  (h1 : game.prob_a_scores_on_a_serve = 2/3)
  (h2 : game.prob_a_scores_on_b_serve = 1/2) :
  prob_13_11 game = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_13_11_is_quarter_l3385_338551


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l3385_338539

/-- A quadratic function passing through (1,0) and (5,0) with minimum value 36 -/
def quadratic (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficient_sum (a b c : ℝ) :
  (quadratic a b c 1 = 0) →
  (quadratic a b c 5 = 0) →
  (∃ x, ∀ y, quadratic a b c y ≥ quadratic a b c x) →
  (∃ x, quadratic a b c x = 36) →
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l3385_338539


namespace NUMINAMATH_CALUDE_race_length_is_18_l3385_338546

/-- The length of a cross-country relay race with 5 members -/
def race_length : ℕ :=
  let other_members : ℕ := 4
  let other_distance : ℕ := 3
  let ralph_multiplier : ℕ := 2
  (other_members * other_distance) + (ralph_multiplier * other_distance)

/-- Theorem: The length of the race is 18 km -/
theorem race_length_is_18 : race_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_race_length_is_18_l3385_338546


namespace NUMINAMATH_CALUDE_dihedral_angle_relation_l3385_338525

/-- Regular quadrilateral prism -/
structure RegularQuadPrism where
  -- We don't need to define the specific geometry, just the existence of the prism
  prism : Unit

/-- Dihedral angle between lateral face and base -/
def lateral_base_angle (p : RegularQuadPrism) : ℝ :=
  sorry

/-- Dihedral angle between adjacent lateral faces -/
def adjacent_lateral_angle (p : RegularQuadPrism) : ℝ :=
  sorry

/-- Theorem stating the relationship between dihedral angles in a regular quadrilateral prism -/
theorem dihedral_angle_relation (p : RegularQuadPrism) :
  Real.cos (adjacent_lateral_angle p) = -(Real.cos (lateral_base_angle p))^2 := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_relation_l3385_338525


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3385_338560

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 3*x - 2*a^2 + 4 = 0}

-- State the theorem
theorem union_of_A_and_B (a : ℝ) :
  (A a ∩ B a = {1}) →
  ((a = 2 ∧ A a ∪ B a = {-4, 1}) ∨ (a = -2 ∧ A a ∪ B a = {-4, -3, 1})) :=
by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3385_338560


namespace NUMINAMATH_CALUDE_snow_leopard_lineup_l3385_338552

/-- The number of ways to arrange 9 distinct objects in a row, 
    where 3 specific objects must be placed at the ends and middle -/
def arrangement_count : ℕ := 4320

/-- The number of ways to arrange 3 objects in 3 specific positions -/
def short_leopard_arrangements : ℕ := 6

/-- The number of ways to arrange the remaining 6 objects -/
def remaining_leopard_arrangements : ℕ := 720

theorem snow_leopard_lineup : 
  arrangement_count = short_leopard_arrangements * remaining_leopard_arrangements :=
sorry

end NUMINAMATH_CALUDE_snow_leopard_lineup_l3385_338552


namespace NUMINAMATH_CALUDE_log_base_2_derivative_l3385_338508

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_log_base_2_derivative_l3385_338508


namespace NUMINAMATH_CALUDE_min_sum_of_squares_min_sum_of_squares_achievable_l3385_338575

theorem min_sum_of_squares (a b : ℝ) (h : a * b = -6) : a^2 + b^2 ≥ 12 := by
  sorry

theorem min_sum_of_squares_achievable : ∃ (a b : ℝ), a * b = -6 ∧ a^2 + b^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_min_sum_of_squares_achievable_l3385_338575


namespace NUMINAMATH_CALUDE_not_perfect_square_l3385_338554

theorem not_perfect_square : 
  (∃ x : ℝ, (6:ℝ)^210 = x^2) ∧
  (∀ x : ℝ, (7:ℝ)^301 ≠ x^2) ∧
  (∃ x : ℝ, (8:ℝ)^402 = x^2) ∧
  (∃ x : ℝ, (9:ℝ)^302 = x^2) ∧
  (∃ x : ℝ, (10:ℝ)^404 = x^2) :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3385_338554


namespace NUMINAMATH_CALUDE_equation_solution_l3385_338573

theorem equation_solution (x y r s : ℚ) : 
  (3 * x + 2 * y = 16) → 
  (5 * x + 3 * y = 26) → 
  (r = x) → 
  (s = y) → 
  (r - s = 2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3385_338573


namespace NUMINAMATH_CALUDE_part1_part2_l3385_338540

/-- The function f(x) = x³ - x² --/
def f (x : ℝ) : ℝ := x^3 - x^2

/-- Part 1: At least one of f(m) and f(n) is not less than zero --/
theorem part1 (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n > 1) :
  max (f m) (f n) ≥ 0 := by sorry

/-- Part 2: a + b < 4/3 --/
theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (heq : f a = f b) :
  a + b < 4/3 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3385_338540


namespace NUMINAMATH_CALUDE_time_to_drain_pool_l3385_338504

/-- The time it takes to drain a rectangular pool given its dimensions, capacity, and drainage rate. -/
theorem time_to_drain_pool 
  (length width depth : ℝ) 
  (capacity : ℝ) 
  (drainage_rate : ℝ) 
  (h1 : length = 150)
  (h2 : width = 50)
  (h3 : depth = 10)
  (h4 : capacity = 0.8)
  (h5 : drainage_rate = 60) :
  (length * width * depth * capacity) / drainage_rate = 1000 := by
  sorry


end NUMINAMATH_CALUDE_time_to_drain_pool_l3385_338504


namespace NUMINAMATH_CALUDE_x_plus_reciprocal_x_l3385_338558

theorem x_plus_reciprocal_x (x : ℝ) 
  (h1 : x^3 + 1/x^3 = 110) 
  (h2 : (x + 1/x)^2 - 2*x - 2/x = 38) : 
  x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_reciprocal_x_l3385_338558


namespace NUMINAMATH_CALUDE_blue_tile_fraction_is_three_fourths_l3385_338529

/-- Represents the tiling pattern of an 8x8 square -/
structure TilingPattern :=
  (size : Nat)
  (blue_tiles_per_corner : Nat)
  (total_corners : Nat)

/-- The fraction of blue tiles in the tiling pattern -/
def blue_tile_fraction (pattern : TilingPattern) : Rat :=
  let total_blue_tiles := pattern.blue_tiles_per_corner * pattern.total_corners
  let total_tiles := pattern.size * pattern.size
  total_blue_tiles / total_tiles

/-- Theorem stating that the fraction of blue tiles in the given pattern is 3/4 -/
theorem blue_tile_fraction_is_three_fourths (pattern : TilingPattern) 
  (h1 : pattern.size = 8)
  (h2 : pattern.blue_tiles_per_corner = 12)
  (h3 : pattern.total_corners = 4) : 
  blue_tile_fraction pattern = 3/4 := by
  sorry

#check blue_tile_fraction_is_three_fourths

end NUMINAMATH_CALUDE_blue_tile_fraction_is_three_fourths_l3385_338529


namespace NUMINAMATH_CALUDE_library_books_before_grant_l3385_338564

/-- The number of books purchased with the grant -/
def books_purchased : ℕ := 2647

/-- The total number of books after the purchase -/
def total_books : ℕ := 8582

/-- The number of books before the grant -/
def books_before : ℕ := total_books - books_purchased

theorem library_books_before_grant :
  books_before = 5935 :=
sorry

end NUMINAMATH_CALUDE_library_books_before_grant_l3385_338564


namespace NUMINAMATH_CALUDE_new_to_original_student_ratio_l3385_338520

theorem new_to_original_student_ratio 
  (original_avg : ℝ) 
  (new_student_avg : ℝ) 
  (avg_decrease : ℝ) 
  (h1 : original_avg = 40)
  (h2 : new_student_avg = 34)
  (h3 : avg_decrease = 4)
  (h4 : original_avg = (original_avg - avg_decrease) + 6) :
  ∃ (O N : ℕ), N = 2 * O ∧ N > 0 ∧ O > 0 := by
  sorry

end NUMINAMATH_CALUDE_new_to_original_student_ratio_l3385_338520


namespace NUMINAMATH_CALUDE_company_dividend_percentage_l3385_338506

/-- Calculates the dividend percentage paid by a company given the face value of a share,
    the investor's return on investment, and the investor's purchase price per share. -/
def dividend_percentage (face_value : ℚ) (roi : ℚ) (purchase_price : ℚ) : ℚ :=
  (roi * purchase_price / face_value) * 100

/-- Theorem stating that under the given conditions, the dividend percentage is 18.5% -/
theorem company_dividend_percentage :
  let face_value : ℚ := 50
  let roi : ℚ := 25 / 100
  let purchase_price : ℚ := 37
  dividend_percentage face_value roi purchase_price = 185 / 10 := by
  sorry

end NUMINAMATH_CALUDE_company_dividend_percentage_l3385_338506


namespace NUMINAMATH_CALUDE_sum_of_possible_y_values_l3385_338510

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- Two angles of the triangle
  angle1 : ℝ
  angle2 : ℝ
  -- The triangle is isosceles
  isIsosceles : angle1 = angle2 ∨ angle1 = 180 - angle1 - angle2 ∨ angle2 = 180 - angle1 - angle2
  -- The sum of angles in a triangle is 180°
  sumOfAngles : angle1 + angle2 + (180 - angle1 - angle2) = 180

-- Theorem statement
theorem sum_of_possible_y_values (t : IsoscelesTriangle) (h1 : t.angle1 = 40 ∨ t.angle2 = 40) :
  ∃ y1 y2 : ℝ, (y1 = t.angle1 ∨ y1 = t.angle2) ∧ 
             (y2 = t.angle1 ∨ y2 = t.angle2) ∧
             y1 ≠ y2 ∧
             y1 + y2 = 140 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_y_values_l3385_338510


namespace NUMINAMATH_CALUDE_daily_wage_c_value_l3385_338538

/-- The daily wage of worker c given the conditions of the problem -/
def daily_wage_c (days_a days_b days_c : ℕ) 
                 (wage_ratio_a wage_ratio_b wage_ratio_c : ℕ) 
                 (total_earning : ℚ) : ℚ :=
  let wage_a := total_earning * wage_ratio_a / 
    (days_a * wage_ratio_a + days_b * wage_ratio_b + days_c * wage_ratio_c)
  wage_a * wage_ratio_c / wage_ratio_a

theorem daily_wage_c_value : 
  daily_wage_c 6 9 4 3 4 5 1480 = 100 / 3 := by
  sorry

#eval daily_wage_c 6 9 4 3 4 5 1480

end NUMINAMATH_CALUDE_daily_wage_c_value_l3385_338538


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3385_338507

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3385_338507


namespace NUMINAMATH_CALUDE_triangle_medians_and_area_sum_l3385_338528

theorem triangle_medians_and_area_sum (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let medians_sum := 3 / 4 * (a^2 + b^2 + c^2)
  medians_sum + area^2 = 4033.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_medians_and_area_sum_l3385_338528


namespace NUMINAMATH_CALUDE_complement_of_A_l3385_338541

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | 1 / x ≥ 1}

theorem complement_of_A : 
  Set.compl A = {x : ℝ | x ≤ 0 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3385_338541


namespace NUMINAMATH_CALUDE_angle_bisector_c_value_l3385_338555

/-- Triangle with vertices A, B, C in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Angle bisector of a triangle -/
def angleBisector (t : Triangle) (v : ℝ × ℝ) (l : LineEquation) : Prop :=
  -- This is a placeholder for the actual definition of an angle bisector
  True

theorem angle_bisector_c_value (t : Triangle) (l : LineEquation) :
  t.A = (-2, 3) →
  t.B = (-6, -8) →
  t.C = (4, -1) →
  l.a = 5 →
  l.b = 4 →
  angleBisector t t.B l →
  l.c + 5 = -155/7 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_c_value_l3385_338555
