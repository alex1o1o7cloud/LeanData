import Mathlib

namespace imaginary_complex_implies_modulus_l885_88568

/-- Given a real number t, if the complex number z = (1-ti)/(1+i) is purely imaginary, 
    then |√3 + ti| = 2 -/
theorem imaginary_complex_implies_modulus (t : ℝ) : 
  let z : ℂ := (1 - t * Complex.I) / (1 + Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs (Real.sqrt 3 + t * Complex.I) = 2 := by
  sorry

end imaginary_complex_implies_modulus_l885_88568


namespace count_distinct_cube_colorings_l885_88558

/-- The number of distinct colorings of a cube with six colors -/
def distinct_cube_colorings : ℕ := 30

/-- The number of faces on a cube -/
def cube_faces : ℕ := 6

/-- The number of rotational symmetries of a cube -/
def cube_rotations : ℕ := 24

/-- Theorem stating the number of distinct colorings of a cube -/
theorem count_distinct_cube_colorings :
  distinct_cube_colorings = (cube_faces * (cube_faces - 1) * (cube_faces - 2) / 2) := by
  sorry

#check count_distinct_cube_colorings

end count_distinct_cube_colorings_l885_88558


namespace f_composition_of_five_l885_88583

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 1

theorem f_composition_of_five : f (f (f (f (f 5)))) = 166 := by
  sorry

end f_composition_of_five_l885_88583


namespace cubic_sum_inequality_l885_88557

theorem cubic_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end cubic_sum_inequality_l885_88557


namespace ladder_slide_l885_88509

theorem ladder_slide (L d s : ℝ) (h1 : L = 20) (h2 : d = 4) (h3 : s = 3) :
  ∃ y : ℝ, y = Real.sqrt (400 - (2 * Real.sqrt 96 - 3)^2) - 4 :=
sorry

end ladder_slide_l885_88509


namespace acute_triangle_perpendicular_pyramid_l885_88501

theorem acute_triangle_perpendicular_pyramid (a b c : ℝ) 
  (h_acute : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ (x y z : ℝ), 
    x^2 + y^2 = c^2 ∧
    y^2 + z^2 = a^2 ∧
    x^2 + z^2 = b^2 ∧
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 :=
by sorry

end acute_triangle_perpendicular_pyramid_l885_88501


namespace lizette_stamp_count_l885_88540

/-- The number of stamps Minerva has -/
def minerva_stamps : ℕ := 688

/-- The number of additional stamps Lizette has compared to Minerva -/
def additional_stamps : ℕ := 125

/-- The total number of stamps Lizette has -/
def lizette_stamps : ℕ := minerva_stamps + additional_stamps

theorem lizette_stamp_count : lizette_stamps = 813 := by
  sorry

end lizette_stamp_count_l885_88540


namespace radio_loss_percentage_l885_88546

theorem radio_loss_percentage (original_price sold_price : ℚ) :
  original_price = 490 →
  sold_price = 465.50 →
  (original_price - sold_price) / original_price * 100 = 5 := by
  sorry

end radio_loss_percentage_l885_88546


namespace discriminant_of_quadratic_poly_l885_88534

/-- The discriminant of a quadratic polynomial ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The quadratic polynomial 3x² + (3 + 1/3)x + 1/3 -/
def quadratic_poly (x : ℚ) : ℚ := 3*x^2 + (3 + 1/3)*x + 1/3

theorem discriminant_of_quadratic_poly :
  discriminant 3 (3 + 1/3) (1/3) = 64/9 := by
  sorry

end discriminant_of_quadratic_poly_l885_88534


namespace g_squared_difference_l885_88554

-- Define the function g
def g : ℝ → ℝ := λ x => 3

-- State the theorem
theorem g_squared_difference (x : ℝ) : g ((x - 1)^2) = 3 := by
  sorry

end g_squared_difference_l885_88554


namespace penalty_kicks_count_l885_88523

theorem penalty_kicks_count (total_players : ℕ) (goalies : ℕ) : 
  total_players = 20 ∧ goalies = 3 → 
  (total_players - goalies) * goalies + goalies * (goalies - 1) = 57 := by
  sorry

end penalty_kicks_count_l885_88523


namespace f_value_at_pi_over_4_f_monotone_increasing_l885_88512

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x) + 2 * (Real.cos x) ^ 2) / Real.cos x

def domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * Real.pi + Real.pi / 2

theorem f_value_at_pi_over_4 :
  f (Real.pi / 4) = 2 * Real.sqrt 2 :=
sorry

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Ioo 0 (Real.pi / 4)) :=
sorry

end f_value_at_pi_over_4_f_monotone_increasing_l885_88512


namespace price_increase_percentage_l885_88529

/-- Calculate the percentage increase given an initial and new price -/
def percentage_increase (initial_price new_price : ℚ) : ℚ :=
  ((new_price - initial_price) / initial_price) * 100

/-- Theorem: The percentage increase from R$ 5.00 to R$ 5.55 is 11% -/
theorem price_increase_percentage :
  let initial_price : ℚ := 5
  let new_price : ℚ := (111 : ℚ) / 20
  percentage_increase initial_price new_price = 11 := by
sorry

#eval percentage_increase 5 (111 / 20)

end price_increase_percentage_l885_88529


namespace original_price_calculation_l885_88553

/-- Given a sale price and a percent decrease, calculate the original price of an item. -/
theorem original_price_calculation (sale_price : ℝ) (percent_decrease : ℝ) 
  (h1 : sale_price = 75)
  (h2 : percent_decrease = 25) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - percent_decrease / 100) = sale_price ∧ 
    original_price = 100 := by
  sorry

end original_price_calculation_l885_88553


namespace max_value_tan_l885_88559

/-- Given a function f(x) = 3sin(x) + 2cos(x), when f(x) reaches its maximum value, tan(x) = 3/2 -/
theorem max_value_tan (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 3 * Real.sin x + 2 * Real.cos x
  ∃ (x_max : ℝ), (∀ y, f y ≤ f x_max) → Real.tan x_max = 3/2 := by
  sorry

end max_value_tan_l885_88559


namespace bicycles_in_garage_l885_88525

theorem bicycles_in_garage (cars : ℕ) (total_wheels : ℕ) (bicycle_wheels : ℕ) (car_wheels : ℕ) : 
  cars = 16 → 
  total_wheels = 82 → 
  bicycle_wheels = 2 → 
  car_wheels = 4 → 
  ∃ bicycles : ℕ, bicycles * bicycle_wheels + cars * car_wheels = total_wheels ∧ bicycles = 9 :=
by sorry

end bicycles_in_garage_l885_88525


namespace parabolas_intersection_l885_88595

/-- First parabola equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- Second parabola equation -/
def g (x : ℝ) : ℝ := 9 * x^2 + 6 * x + 2

/-- The set of intersection points of the two parabolas -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = g p.1 ∧ p.2 = f p.1}

theorem parabolas_intersection :
  intersection_points = {(0, 2), (-5/3, 17)} := by sorry

end parabolas_intersection_l885_88595


namespace golden_section_length_l885_88513

/-- Given a segment AB of length 2 with C as its golden section point (AC > BC),
    the length of AC is √5 - 1 -/
theorem golden_section_length (A B C : ℝ) : 
  (B - A = 2) →
  (C - A) / (B - C) = (1 + Real.sqrt 5) / 2 →
  C - A > B - C →
  C - A = Real.sqrt 5 - 1 := by
  sorry

end golden_section_length_l885_88513


namespace greatest_base7_digit_sum_l885_88555

/-- Represents a base-7 digit (0 to 6) -/
def Base7Digit := Fin 7

/-- Represents a base-7 number as a list of digits -/
def Base7Number := List Base7Digit

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : Base7Number :=
  sorry

/-- Calculates the sum of digits in a base-7 number -/
def digitSum (num : Base7Number) : ℕ :=
  sorry

/-- Checks if a base-7 number is less than 1729 in decimal -/
def isLessThan1729 (num : Base7Number) : Prop :=
  sorry

theorem greatest_base7_digit_sum :
  ∃ (n : Base7Number), isLessThan1729 n ∧
    digitSum n = 22 ∧
    ∀ (m : Base7Number), isLessThan1729 m → digitSum m ≤ 22 :=
  sorry

end greatest_base7_digit_sum_l885_88555


namespace mary_current_books_l885_88591

/-- Calculates the number of books Mary has checked out after a series of library transactions. -/
def marysBooks (initialBooks : ℕ) (firstReturn : ℕ) (firstCheckout : ℕ) (secondReturn : ℕ) (secondCheckout : ℕ) : ℕ :=
  (((initialBooks - firstReturn) + firstCheckout) - secondReturn) + secondCheckout

/-- Proves that Mary currently has 12 books checked out from the library. -/
theorem mary_current_books :
  marysBooks 5 3 5 2 7 = 12 := by
  sorry

end mary_current_books_l885_88591


namespace arithmetic_sequence_common_difference_l885_88572

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 4) :
  ∃ d : ℝ, d = -1 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l885_88572


namespace square_perimeter_unchanged_l885_88567

/-- The perimeter of a square with side length 5 remains unchanged after cutting out four small rectangles from its corners. -/
theorem square_perimeter_unchanged (side_length : ℝ) (h : side_length = 5) :
  let original_perimeter := 4 * side_length
  let modified_perimeter := original_perimeter
  modified_perimeter = 20 := by
  sorry

end square_perimeter_unchanged_l885_88567


namespace rectangle_ratio_squared_l885_88593

theorem rectangle_ratio_squared (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a ≤ b) : 
  (a / b + 1 / 2 = b / Real.sqrt (a^2 + b^2)) → (a / b)^2 = (3 - Real.sqrt 5) / 2 := by
  sorry

end rectangle_ratio_squared_l885_88593


namespace pure_imaginary_z_l885_88532

theorem pure_imaginary_z (z : ℂ) : 
  (∃ (a : ℝ), z = Complex.I * a) → 
  Complex.abs (z - 1) = Complex.abs (-1 + Complex.I) → 
  z = Complex.I ∨ z = -Complex.I := by
  sorry

end pure_imaginary_z_l885_88532


namespace square_area_from_diagonal_l885_88552

theorem square_area_from_diagonal (a b : ℝ) :
  let diagonal := a + b
  ∃ s : ℝ, s > 0 ∧ s * s = (1/2) * diagonal * diagonal :=
by
  sorry

end square_area_from_diagonal_l885_88552


namespace square_area_from_diagonal_l885_88586

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  let s := d / Real.sqrt 2
  s * s = 72 := by sorry

end square_area_from_diagonal_l885_88586


namespace prob_two_rolls_eq_one_sixty_fourth_l885_88504

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The desired sum on each roll -/
def target_sum : ℕ := 9

/-- The set of all possible outcomes when rolling two dice -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range num_sides) (Finset.range num_sides)

/-- The set of outcomes that sum to the target -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun (a, b) => a + b + 2 = target_sum)

/-- The probability of rolling the target sum once -/
def prob_single_roll : ℚ :=
  (favorable_outcomes.card : ℚ) / (all_outcomes.card : ℚ)

theorem prob_two_rolls_eq_one_sixty_fourth :
  prob_single_roll * prob_single_roll = 1 / 64 := by
  sorry

end prob_two_rolls_eq_one_sixty_fourth_l885_88504


namespace truck_travel_distance_l885_88562

-- Define the given conditions
def initial_distance : ℝ := 300
def initial_fuel : ℝ := 10
def new_fuel : ℝ := 15

-- Define the theorem
theorem truck_travel_distance :
  (initial_distance / initial_fuel) * new_fuel = 450 := by
  sorry

end truck_travel_distance_l885_88562


namespace registration_methods_l885_88524

/-- The number of students signing up for interest groups -/
def num_students : ℕ := 4

/-- The number of interest groups available -/
def num_groups : ℕ := 3

/-- Theorem stating the total number of registration methods -/
theorem registration_methods :
  (num_groups ^ num_students : ℕ) = 81 := by
  sorry

end registration_methods_l885_88524


namespace exists_divisible_by_two_not_four_l885_88592

theorem exists_divisible_by_two_not_four : ∃ m : ℕ, (2 ∣ m) ∧ ¬(4 ∣ m) := by
  sorry

end exists_divisible_by_two_not_four_l885_88592


namespace tangent_line_cubic_l885_88541

/-- Given a curve y = x^3 and a point (1, 1) on this curve, 
    the equation of the tangent line at this point is 3x - y - 2 = 0 -/
theorem tangent_line_cubic (x y : ℝ) : 
  (y = x^3) → -- The curve equation
  (1 = 1^3) → -- The point (1, 1) satisfies the curve equation
  (3*x - y - 2 = 0) -- The equation of the tangent line
  := by sorry

end tangent_line_cubic_l885_88541


namespace pizza_combinations_l885_88569

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  (n.choose 1) + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end pizza_combinations_l885_88569


namespace brenda_spay_cats_l885_88522

/-- Represents the number of cats Brenda needs to spay -/
def num_cats : ℕ := sorry

/-- Represents the number of dogs Brenda needs to spay -/
def num_dogs : ℕ := sorry

/-- The total number of animals Brenda needs to spay -/
def total_animals : ℕ := 21

theorem brenda_spay_cats :
  (num_cats + num_dogs = total_animals) →
  (num_dogs = 2 * num_cats) →
  num_cats = 7 := by
  sorry

end brenda_spay_cats_l885_88522


namespace ten_bulb_signals_l885_88543

/-- The number of different signals that can be transmitted using a given number of light bulbs -/
def signalCount (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of different signals that can be transmitted using 10 light bulbs, 
    each of which can be either on or off, is equal to 2^10 (1024) -/
theorem ten_bulb_signals : signalCount 10 = 1024 := by
  sorry

end ten_bulb_signals_l885_88543


namespace complex_magnitude_equality_l885_88530

theorem complex_magnitude_equality (t : ℝ) : 
  t > 0 → (Complex.abs (-4 + t * Complex.I) = 2 * Real.sqrt 13 ↔ t = 6) := by
  sorry

end complex_magnitude_equality_l885_88530


namespace xyz_less_than_one_l885_88597

theorem xyz_less_than_one (x y z : ℝ) 
  (h1 : 2 * x > y^2 + z^2)
  (h2 : 2 * y > x^2 + z^2)
  (h3 : 2 * z > y^2 + x^2) : 
  x * y * z < 1 := by
  sorry

end xyz_less_than_one_l885_88597


namespace reconstruct_triangle_l885_88507

-- Define the types for points and triangles
def Point : Type := ℝ × ℝ
def Triangle : Type := Point × Point × Point

-- Define the external angle bisector
def externalAngleBisector (A B C : Point) : Point → Prop := sorry

-- Define the perpendicular from a point to a line
def perpendicularFoot (P A B : Point) : Point := sorry

-- Define the statement
theorem reconstruct_triangle (A' B' C' : Point) :
  ∃ (A B C : Point),
    -- A'B'C' is formed by external angle bisectors of ABC
    externalAngleBisector B C A A' ∧
    externalAngleBisector A C B B' ∧
    externalAngleBisector A B C C' ∧
    -- A, B, C are feet of perpendiculars from A', B', C' to opposite sides of A'B'C'
    A = perpendicularFoot A' B' C' ∧
    B = perpendicularFoot B' A' C' ∧
    C = perpendicularFoot C' A' B' :=
by
  sorry

end reconstruct_triangle_l885_88507


namespace square_to_acute_triangle_with_different_sides_l885_88579

/-- A part of a square -/
structure SquarePart where
  -- Add necessary fields

/-- A triangle formed from parts of a square -/
structure TriangleFromSquare where
  parts : Finset SquarePart
  -- Add necessary fields for angles and sides

/-- Represents a square that can be cut into parts -/
structure CuttableSquare where
  side : ℝ
  -- Add other necessary fields

/-- Predicate to check if a triangle has acute angles -/
def has_acute_angles (t : TriangleFromSquare) : Prop :=
  sorry

/-- Predicate to check if a triangle has different sides -/
def has_different_sides (t : TriangleFromSquare) : Prop :=
  sorry

/-- Theorem stating that a square can be cut into 3 parts to form a specific triangle -/
theorem square_to_acute_triangle_with_different_sides :
  ∃ (s : CuttableSquare) (t : TriangleFromSquare),
    t.parts.card = 3 ∧
    has_acute_angles t ∧
    has_different_sides t :=
  sorry

end square_to_acute_triangle_with_different_sides_l885_88579


namespace inequality_solution_set_l885_88582

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |a * x + 2| < 6 ↔ -1 < x ∧ x < 2) → a = -4 := by
  sorry

end inequality_solution_set_l885_88582


namespace triangle_problem_l885_88542

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * (b^2 + c^2 - a^2) →
  (1/2) * b * c * Real.sin A = 3/2 →
  (A = π/3 ∧ ((b*c - 4*Real.sqrt 3) * Real.cos A + a*c * Real.cos B) / (a^2 - b^2) = 1) := by
  sorry

end triangle_problem_l885_88542


namespace cosine_difference_l885_88575

theorem cosine_difference (α β : ℝ) 
  (h1 : α + β = π / 3)
  (h2 : Real.tan α + Real.tan β = 2) :
  Real.cos (α - β) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end cosine_difference_l885_88575


namespace weight_probability_l885_88506

/-- The probability that the weight of five eggs is less than 30 grams -/
def prob_less_than_30 : ℝ := 0.3

/-- The probability that the weight of five eggs is between [30, 40] grams -/
def prob_between_30_and_40 : ℝ := 0.5

/-- The probability that the weight of five eggs does not exceed 40 grams -/
def prob_not_exceed_40 : ℝ := prob_less_than_30 + prob_between_30_and_40

theorem weight_probability : prob_not_exceed_40 = 0.8 := by
  sorry

end weight_probability_l885_88506


namespace inequality_proof_l885_88526

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  6 * a * b * c ≤ a * b * (a + b) + b * c * (b + c) + a * c * (a + c) ∧
  a * b * (a + b) + b * c * (b + c) + a * c * (a + c) ≤ 2 * (a^3 + b^3 + c^3) := by
  sorry

end inequality_proof_l885_88526


namespace expression_equality_l885_88531

theorem expression_equality : 
  |Real.sqrt 8 - 2| + (π - 2023)^(0 : ℝ) + (-1/2)^(-2 : ℝ) - 2 * Real.cos (60 * π / 180) = 2 * Real.sqrt 2 + 2 := by
  sorry

end expression_equality_l885_88531


namespace solution_exists_for_all_primes_l885_88514

theorem solution_exists_for_all_primes (p : ℕ) (hp : Nat.Prime p) :
  ∃ n : ℤ, (6 * n^2 + 5 * n + 1) % p = 0 := by
  sorry

end solution_exists_for_all_primes_l885_88514


namespace stephanie_orange_spending_l885_88590

def num_visits : Nat := 8
def oranges_per_visit : Nat := 2

def prices : List Float := [0.50, 0.60, 0.55, 0.65, 0.70, 0.55, 0.50, 0.60]

theorem stephanie_orange_spending :
  prices.length = num_visits →
  (prices.map (· * oranges_per_visit.toFloat)).sum = 9.30 := by
  sorry

end stephanie_orange_spending_l885_88590


namespace quadratic_distinct_roots_l885_88518

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 9 = 0 ∧ x₂^2 + m*x₂ + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
by sorry

end quadratic_distinct_roots_l885_88518


namespace max_profit_on_day_6_l885_88584

-- Define the sales price function
def p (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 6 then 44 + x
  else if 6 < x ∧ x ≤ 20 then 56 - x
  else 0

-- Define the sales volume function
def q (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 8 then 48 - x
  else if 8 < x ∧ x ≤ 20 then 32 + x
  else 0

-- Define the profit function
def profit (x : ℕ) : ℝ := (p x - 25) * q x

-- Theorem statement
theorem max_profit_on_day_6 :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 20 → profit x ≤ profit 6 ∧ profit 6 = 1050 :=
sorry

end max_profit_on_day_6_l885_88584


namespace custom_op_theorem_l885_88556

def customOp (M N : Set ℕ) : Set ℕ := {x | x ∈ M ∨ x ∈ N ∧ x ∉ M ∩ N}

def M : Set ℕ := {0, 2, 4, 6, 8, 10}
def N : Set ℕ := {0, 3, 6, 9, 12, 15}

theorem custom_op_theorem :
  (customOp (customOp M N) M) = {2, 4, 8, 10, 3, 9, 12, 15} := by sorry

end custom_op_theorem_l885_88556


namespace second_smallest_hotdog_pack_l885_88528

def is_valid_hotdog_pack (n : ℕ) : Prop :=
  ∃ b : ℕ, 12 * n - 10 * b = 6 ∧ n % 5 = 3

theorem second_smallest_hotdog_pack :
  ∃ n : ℕ, is_valid_hotdog_pack n ∧
  (∀ m : ℕ, m < n → ¬is_valid_hotdog_pack m ∨ 
   (∃ k : ℕ, k < m ∧ is_valid_hotdog_pack k)) ∧
  n = 8 :=
sorry

end second_smallest_hotdog_pack_l885_88528


namespace sam_distance_l885_88537

/-- Given Marguerite's cycling distance and time, and Sam's cycling time,
    prove that Sam's distance is equal to (Marguerite's distance / Marguerite's time) * Sam's time,
    assuming they cycle at the same average speed. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ)
    (h1 : marguerite_distance > 0)
    (h2 : marguerite_time > 0)
    (h3 : sam_time > 0) :
  let sam_distance := (marguerite_distance / marguerite_time) * sam_time
  sam_distance = (marguerite_distance / marguerite_time) * sam_time :=
by
  sorry

#check sam_distance

end sam_distance_l885_88537


namespace area_of_large_rectangle_l885_88511

/-- The width of each smaller rectangle in feet -/
def small_rectangle_width : ℝ := 8

/-- The number of identical rectangles stacked vertically -/
def num_rectangles : ℕ := 3

/-- The length of each smaller rectangle in feet -/
def small_rectangle_length : ℝ := 2 * small_rectangle_width

/-- The width of the larger rectangle ABCD in feet -/
def large_rectangle_width : ℝ := small_rectangle_width

/-- The length of the larger rectangle ABCD in feet -/
def large_rectangle_length : ℝ := num_rectangles * small_rectangle_length

/-- The area of the larger rectangle ABCD in square feet -/
def large_rectangle_area : ℝ := large_rectangle_width * large_rectangle_length

theorem area_of_large_rectangle : large_rectangle_area = 384 := by
  sorry

end area_of_large_rectangle_l885_88511


namespace candy_difference_l885_88503

theorem candy_difference (anna_per_house billy_per_house anna_houses billy_houses : ℕ) 
  (h1 : anna_per_house = 14)
  (h2 : billy_per_house = 11)
  (h3 : anna_houses = 60)
  (h4 : billy_houses = 75) :
  anna_per_house * anna_houses - billy_per_house * billy_houses = 15 := by
  sorry

end candy_difference_l885_88503


namespace only_one_true_iff_in_range_l885_88533

/-- The proposition p: no solution for the quadratic inequality -/
def p (a : ℝ) : Prop := a > 0 ∧ ∀ x, x^2 + (a-1)*x + a^2 > 0

/-- The proposition q: probability condition -/
def q (a : ℝ) : Prop := a > 0 ∧ (min a 4 + 2) / 6 ≥ 5/6

/-- The main theorem -/
theorem only_one_true_iff_in_range (a : ℝ) :
  (p a ∧ ¬q a) ∨ (¬p a ∧ q a) ↔ a > 1/3 ∧ a < 3 :=
sorry

end only_one_true_iff_in_range_l885_88533


namespace tribe_leadership_organization_l885_88566

def tribe_size : ℕ := 12
def num_chiefs : ℕ := 1
def num_supporting_chiefs : ℕ := 3
def inferior_officers_per_chief : ℕ := 2

theorem tribe_leadership_organization :
  (tribe_size.choose num_chiefs) *
  ((tribe_size - num_chiefs).choose 1) *
  ((tribe_size - num_chiefs - 1).choose 1) *
  ((tribe_size - num_chiefs - 2).choose 1) *
  ((tribe_size - num_chiefs - num_supporting_chiefs).choose inferior_officers_per_chief) *
  ((tribe_size - num_chiefs - num_supporting_chiefs - inferior_officers_per_chief).choose inferior_officers_per_chief) *
  ((tribe_size - num_chiefs - num_supporting_chiefs - 2 * inferior_officers_per_chief).choose inferior_officers_per_chief) = 1069200 := by
  sorry

end tribe_leadership_organization_l885_88566


namespace sequence_general_term_l885_88505

/-- Given a sequence {aₙ} where the sequence of differences forms an arithmetic
    sequence with first term 1 and common difference 1, prove that the general
    term formula for {aₙ} is n(n+1)/2. -/
theorem sequence_general_term (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) - a n = n) →
  a 1 = 1 →
  ∀ n : ℕ, a n = n * (n + 1) / 2 :=
by sorry

end sequence_general_term_l885_88505


namespace additional_group_average_weight_l885_88544

theorem additional_group_average_weight 
  (initial_count : ℕ) 
  (additional_count : ℕ) 
  (weight_increase : ℝ) 
  (final_average : ℝ) : 
  initial_count = 30 →
  additional_count = 30 →
  weight_increase = 10 →
  final_average = 40 →
  let total_count := initial_count + additional_count
  let initial_average := final_average - weight_increase
  let initial_total_weight := initial_count * initial_average
  let final_total_weight := total_count * final_average
  let additional_total_weight := final_total_weight - initial_total_weight
  additional_total_weight / additional_count = 50 := by
sorry

end additional_group_average_weight_l885_88544


namespace fibonacci_determinant_l885_88510

/-- An arbitrary Fibonacci sequence -/
def FibonacciSequence (u : ℕ → ℤ) : Prop :=
  ∀ n, u (n + 2) = u n + u (n + 1)

/-- The main theorem about the determinant of consecutive Fibonacci terms -/
theorem fibonacci_determinant (u : ℕ → ℤ) (h : FibonacciSequence u) :
  ∀ n : ℕ, u (n - 1) * u (n + 1) - u n ^ 2 = (-1) ^ n :=
by sorry

end fibonacci_determinant_l885_88510


namespace first_month_sale_l885_88573

/-- Given the sales data for 6 months and the average sale, prove the sale amount for the first month -/
theorem first_month_sale
  (sales_2 sales_3 sales_4 sales_5 sales_6 : ℕ)
  (average_sale : ℕ)
  (h1 : sales_2 = 6500)
  (h2 : sales_3 = 9855)
  (h3 : sales_4 = 7230)
  (h4 : sales_5 = 7000)
  (h5 : sales_6 = 11915)
  (h6 : average_sale = 7500)
  : ∃ (sales_1 : ℕ), sales_1 = 2500 ∧ 
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / 6 = average_sale :=
by sorry

end first_month_sale_l885_88573


namespace photo_difference_l885_88563

theorem photo_difference (initial_photos : ℕ) (final_photos : ℕ) : 
  initial_photos = 400 →
  final_photos = 920 →
  let first_day_photos := initial_photos / 2
  let total_new_photos := final_photos - initial_photos
  let second_day_photos := total_new_photos - first_day_photos
  second_day_photos - first_day_photos = 120 := by
sorry


end photo_difference_l885_88563


namespace polygon_sides_proof_l885_88538

theorem polygon_sides_proof (x y : ℕ) : 
  (x - 2) * 180 + (y - 2) * 180 = 21 * (x + y + x * (x - 3) / 2 + y * (y - 3) / 2) - 39 →
  x * (x - 3) / 2 + y * (y - 3) / 2 - (x + y) = 99 →
  ((x = 17 ∧ y = 3) ∨ (x = 3 ∧ y = 17)) :=
by sorry

end polygon_sides_proof_l885_88538


namespace sequence_a_bounds_l885_88549

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => sequence_a n + (1 : ℚ)/(n+1)^2 * (sequence_a n)^2

theorem sequence_a_bounds : ∀ n : ℕ, (n+1 : ℚ)/(n+2) < sequence_a n ∧ sequence_a n < n+1 := by
  sorry

end sequence_a_bounds_l885_88549


namespace quadratic_range_for_x_less_than_neg_two_l885_88519

/-- Represents a quadratic function y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-value of a quadratic function at a given x -/
def QuadraticFunction.yValue (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_range_for_x_less_than_neg_two
  (f : QuadraticFunction)
  (h_a_pos : f.a > 0)
  (h_vertex : f.yValue (-1) = -6)
  (h_y_at_neg_two : f.yValue (-2) = -5)
  (x : ℝ)
  (h_x : x < -2) :
  f.yValue x > -5 := by
  sorry

end quadratic_range_for_x_less_than_neg_two_l885_88519


namespace rotate_line_theorem_l885_88517

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- Rotates a line counterclockwise by π/2 around a given point -/
def rotateLine (l : Line) (px py : ℝ) : Line :=
  sorry

theorem rotate_line_theorem (l : Line) :
  l.a = 2 ∧ l.b = -1 ∧ l.c = -2 →
  let rotated := rotateLine l 0 (-2)
  rotated.a = 1 ∧ rotated.b = 2 ∧ rotated.c = 4 :=
sorry

end rotate_line_theorem_l885_88517


namespace triangle_transformation_result_l885_88580

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Rotates a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : Point) : Point :=
  ⟨p.y, -p.x⟩

/-- Reflects a point over the x-axis -/
def reflectOverX (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Translates a point vertically by a given amount -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  ⟨p.x, p.y + dy⟩

/-- Rotates a point 180 degrees around the origin -/
def rotate180 (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

/-- Applies all transformations to a single point -/
def applyAllTransformations (p : Point) : Point :=
  rotate180 (translateVertical (reflectOverX (rotate90Clockwise p)) 3)

/-- The main theorem stating the result of the transformations -/
theorem triangle_transformation_result :
  let initial := Triangle.mk ⟨1, 2⟩ ⟨4, 2⟩ ⟨1, 5⟩
  let final := Triangle.mk (applyAllTransformations initial.A)
                           (applyAllTransformations initial.B)
                           (applyAllTransformations initial.C)
  final = Triangle.mk ⟨-2, -4⟩ ⟨-2, -7⟩ ⟨-5, -4⟩ := by
  sorry

end triangle_transformation_result_l885_88580


namespace fourth_root_equation_solutions_l885_88571

theorem fourth_root_equation_solutions :
  {x : ℝ | x > 0 ∧ (x^(1/4) : ℝ) = 15 / (8 - (x^(1/4) : ℝ))} = {625, 81} := by
  sorry

end fourth_root_equation_solutions_l885_88571


namespace lew_gumballs_correct_l885_88561

/-- The number of gumballs Carolyn bought -/
def carolyn_gumballs : ℕ := 17

/-- The number of gumballs Lew bought -/
def lew_gumballs : ℕ := 21

/-- The minimum number of gumballs Carey could have bought -/
def carey_min_gumballs : ℕ := 19

/-- The maximum number of gumballs Carey could have bought -/
def carey_max_gumballs : ℕ := 37

/-- The difference between the maximum and minimum number of gumballs Carey could have bought -/
def carey_gumballs_diff : ℕ := 18

/-- The minimum average number of gumballs -/
def min_avg : ℕ := 19

/-- The maximum average number of gumballs -/
def max_avg : ℕ := 25

theorem lew_gumballs_correct :
  ∀ x : ℕ,
  carey_min_gumballs ≤ x ∧ x ≤ carey_max_gumballs →
  (carolyn_gumballs + lew_gumballs + x : ℚ) / 3 ≥ min_avg ∧
  (carolyn_gumballs + lew_gumballs + x : ℚ) / 3 ≤ max_avg ∧
  carey_max_gumballs - carey_min_gumballs = carey_gumballs_diff →
  lew_gumballs = 21 :=
by sorry

end lew_gumballs_correct_l885_88561


namespace sets_equality_l885_88598

def A : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 1}
def B : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4*b + 5}

theorem sets_equality : A = B := by sorry

end sets_equality_l885_88598


namespace fraction_meaningful_condition_l885_88515

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = (x + 2) / (x - 1)) ↔ x ≠ 1 := by sorry

end fraction_meaningful_condition_l885_88515


namespace count_factors_l885_88520

/-- The number of distinct, whole-number factors of 3^5 * 5^3 * 7^2 -/
def num_factors : ℕ := 72

/-- The prime factorization of the number -/
def prime_factorization : List (ℕ × ℕ) := [(3, 5), (5, 3), (7, 2)]

/-- Theorem stating that the number of distinct, whole-number factors of 3^5 * 5^3 * 7^2 is 72 -/
theorem count_factors : 
  (List.prod (prime_factorization.map (fun (p, e) => e + 1))) = num_factors := by
  sorry

end count_factors_l885_88520


namespace equation_solution_l885_88539

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by
  sorry

end equation_solution_l885_88539


namespace hyperbola_triangle_perimeter_l885_88588

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- State the theorem
theorem hyperbola_triangle_perimeter :
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  (A.1 < 0 ∧ B.1 < 0) →  -- A and B are on the left branch
  F₁ ∈ Set.Icc A B →     -- F₁ is on the line segment AB
  dist A B = 6 →
  dist A F₂ + dist B F₂ + dist A B = 28 :=
sorry

end hyperbola_triangle_perimeter_l885_88588


namespace tenth_term_of_sequence_l885_88599

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem tenth_term_of_sequence (a₁ r : ℚ) (h₁ : a₁ = 12) (h₂ : r = 1/2) :
  geometric_sequence a₁ r 10 = 3/128 := by
  sorry

end tenth_term_of_sequence_l885_88599


namespace probability_at_least_one_shot_l885_88548

/-- The probability of making at least one shot out of three, given a success rate of 3/5 for each shot. -/
theorem probability_at_least_one_shot (success_rate : ℝ) (num_shots : ℕ) : 
  success_rate = 3/5 → num_shots = 3 → 1 - (1 - success_rate)^num_shots = 0.936 := by
  sorry


end probability_at_least_one_shot_l885_88548


namespace hypotenuse_square_of_right_triangle_from_polynomial_roots_l885_88578

/-- Given complex numbers a, b, and c that are zeros of a polynomial P(z) = z³ + pz² + qz + r,
    if |a|² + |b|² + |c|² = 300 and they form a right triangle in the complex plane,
    then the square of the hypotenuse h² = 400. -/
theorem hypotenuse_square_of_right_triangle_from_polynomial_roots
  (a b c : ℂ) (p q r : ℂ) :
  (a^3 + p*a^2 + q*a + r = 0) →
  (b^3 + p*b^2 + q*b + r = 0) →
  (c^3 + p*c^2 + q*c + r = 0) →
  Complex.abs a ^ 2 + Complex.abs b ^ 2 + Complex.abs c ^ 2 = 300 →
  ∃ (h : ℝ), (Complex.abs (a - c))^2 + (Complex.abs (b - c))^2 = h^2 →
  h^2 = 400 :=
by sorry

end hypotenuse_square_of_right_triangle_from_polynomial_roots_l885_88578


namespace total_coins_count_l885_88502

theorem total_coins_count (dimes nickels quarters : ℕ) : 
  dimes = 2 → nickels = 2 → quarters = 7 → dimes + nickels + quarters = 11 := by
  sorry

end total_coins_count_l885_88502


namespace quadratic_sum_l885_88560

/-- Given a quadratic function g(x) = 2x^2 + Bx + C, 
    if g(1) = 3 and g(2) = 0, then 2 + B + C + 2C = 23 -/
theorem quadratic_sum (B C : ℝ) : 
  (2 * 1^2 + B * 1 + C = 3) → 
  (2 * 2^2 + B * 2 + C = 0) → 
  (2 + B + C + 2 * C = 23) := by
  sorry

end quadratic_sum_l885_88560


namespace product_trailing_zeroes_l885_88535

/-- The number of trailing zeroes in a positive integer -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- The product of 25^5, 150^4, and 2008^3 -/
def largeProduct : ℕ := 25^5 * 150^4 * 2008^3

theorem product_trailing_zeroes :
  trailingZeroes largeProduct = 13 := by sorry

end product_trailing_zeroes_l885_88535


namespace parallelogram_area_l885_88527

/-- The area of a parallelogram with sides a and b and angle γ between them is ab sin γ -/
theorem parallelogram_area (a b γ : ℝ) (ha : a > 0) (hb : b > 0) (hγ : 0 < γ ∧ γ < π) :
  ∃ S : ℝ, S = a * b * Real.sin γ ∧ S > 0 := by
  sorry

end parallelogram_area_l885_88527


namespace power_of_product_l885_88565

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end power_of_product_l885_88565


namespace max_value_on_ellipse_l885_88576

theorem max_value_on_ellipse :
  ∀ x y : ℝ, x^2/4 + y^2 = 1 → 2*x + y ≤ Real.sqrt 17 := by
  sorry

end max_value_on_ellipse_l885_88576


namespace rational_function_value_l885_88536

-- Define the polynomials p and q
def p (k m : ℝ) (x : ℝ) : ℝ := k * x + m
def q (x : ℝ) : ℝ := (x + 4) * (x - 1)

-- State the theorem
theorem rational_function_value (k m : ℝ) :
  (p k m 0) / (q 0) = 0 →
  (p k m 2) / (q 2) = -1 →
  (p k m (-1)) / (q (-1)) = -1/2 := by
  sorry

end rational_function_value_l885_88536


namespace tourist_groups_speed_l885_88500

theorem tourist_groups_speed : ∀ (x y : ℝ),
  (x > 0 ∧ y > 0) →  -- Speeds are positive
  (4.5 * x + 2.5 * y = 30) →  -- First scenario equation
  (3 * x + 5 * y = 30) →  -- Second scenario equation
  (x = 5 ∧ y = 3) :=  -- Speeds of the two groups
by sorry

end tourist_groups_speed_l885_88500


namespace pig_price_calculation_l885_88547

/-- Given a total of 3 pigs and 10 hens costing Rs. 1200 in total,
    with hens costing an average of Rs. 30 each,
    prove that the average price of a pig is Rs. 300. -/
theorem pig_price_calculation (total_cost : ℕ) (num_pigs num_hens : ℕ) (avg_hen_price : ℕ) :
  total_cost = 1200 →
  num_pigs = 3 →
  num_hens = 10 →
  avg_hen_price = 30 →
  (total_cost - num_hens * avg_hen_price) / num_pigs = 300 := by
  sorry

end pig_price_calculation_l885_88547


namespace geometric_progression_first_term_l885_88551

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (hS : S = 8) 
  (hsum : sum_first_two = 5) : 
  ∃ a : ℝ, (a = 8 * (1 - Real.sqrt (3/8)) ∨ a = 8 * (1 + Real.sqrt (3/8))) ∧ 
    (∃ r : ℝ, S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end geometric_progression_first_term_l885_88551


namespace train_speed_l885_88521

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 385 →
  bridge_length = 140 →
  time = 42 →
  (train_length + bridge_length) / time * 3.6 = 45 := by
  sorry

end train_speed_l885_88521


namespace point_in_fourth_quadrant_implies_a_equals_two_l885_88570

-- Define the point P
def P (a : ℤ) : ℝ × ℝ := (a - 1, a - 3)

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant_implies_a_equals_two (a : ℤ) :
  in_fourth_quadrant (P a) → a = 2 := by
  sorry

end point_in_fourth_quadrant_implies_a_equals_two_l885_88570


namespace supplementary_angle_measure_l885_88564

theorem supplementary_angle_measure (angle : ℝ) (supplementary : ℝ) (complementary : ℝ) : 
  angle = 45 →
  angle + supplementary = 180 →
  angle + complementary = 90 →
  supplementary = 3 * complementary →
  supplementary = 135 := by
sorry

end supplementary_angle_measure_l885_88564


namespace four_hearts_probability_l885_88516

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
| Hearts | Diamonds | Clubs | Spades

/-- Represents the rank of a card -/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A function that maps a card index to its suit -/
def card_to_suit : Fin 52 → Suit := sorry

/-- A function that maps a card index to its rank -/
def card_to_rank : Fin 52 → Rank := sorry

/-- The number of hearts in a standard deck -/
def hearts_count : Nat := 13

/-- Theorem: The probability of drawing four hearts as the top four cards from a standard 52-card deck is 286/108290 -/
theorem four_hearts_probability (d : Deck) : 
  (hearts_count * (hearts_count - 1) * (hearts_count - 2) * (hearts_count - 3)) / 
  (d.cards.card * (d.cards.card - 1) * (d.cards.card - 2) * (d.cards.card - 3)) = 286 / 108290 :=
sorry

end four_hearts_probability_l885_88516


namespace g_of_3_equals_209_l885_88545

-- Define the function g
def g (x : ℝ) : ℝ := 9 * x^3 - 4 * x^2 + 3 * x - 7

-- Theorem statement
theorem g_of_3_equals_209 : g 3 = 209 := by
  sorry

end g_of_3_equals_209_l885_88545


namespace T_equality_l885_88581

theorem T_equality (x : ℝ) : 
  (x - 2)^4 + 5*(x - 2)^3 + 10*(x - 2)^2 + 10*(x - 2) + 5 = (x - 1)^4 + 1 := by
sorry

end T_equality_l885_88581


namespace rectangle_shorter_side_length_l885_88574

/-- Given a rectangle made from a rope of length 100cm with longer sides of 28cm,
    the length of each shorter side is 22cm. -/
theorem rectangle_shorter_side_length
  (total_length : ℝ)
  (longer_side : ℝ)
  (h1 : total_length = 100)
  (h2 : longer_side = 28)
  : (total_length - 2 * longer_side) / 2 = 22 := by
  sorry

end rectangle_shorter_side_length_l885_88574


namespace bipyramid_volume_l885_88587

/-- A bipyramid with square bases -/
structure Bipyramid :=
  (side : ℝ)
  (apex_angle : ℝ)

/-- The volume of a bipyramid -/
noncomputable def volume (b : Bipyramid) : ℝ :=
  sorry

/-- Theorem stating the volume of a specific bipyramid -/
theorem bipyramid_volume (b : Bipyramid) (h1 : b.side = 2) (h2 : b.apex_angle = π / 3) :
  volume b = 16 * Real.sqrt 3 / 9 := by
  sorry

end bipyramid_volume_l885_88587


namespace angle_value_l885_88508

theorem angle_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan β = 1/2) (h4 : Real.tan (α - β) = 1/3) : α = π/4 := by
  sorry

end angle_value_l885_88508


namespace floor_sum_equality_l885_88596

theorem floor_sum_equality (n : ℕ+) : 
  ∑' k : ℕ, ⌊(n + 2^k : ℝ) / 2^(k+1)⌋ = n := by sorry

end floor_sum_equality_l885_88596


namespace t_shirt_problem_l885_88550

/-- Represents a t-shirt package with its size and price -/
structure Package where
  size : Nat
  price : Rat

/-- Calculates the total number of t-shirts and the discounted price -/
def calculate_total (small medium large : Package) 
                    (small_qty medium_qty large_qty : Nat) : Nat × Rat :=
  let total_shirts := small.size * small_qty + medium.size * medium_qty + large.size * large_qty
  let total_price := small.price * small_qty + medium.price * medium_qty + large.price * large_qty
  let total_packages := small_qty + medium_qty + large_qty
  let discounted_price := if total_packages > 25 
                          then total_price * (1 - 5 / 100) 
                          else total_price
  (total_shirts, discounted_price)

theorem t_shirt_problem :
  let small : Package := ⟨6, 12⟩
  let medium : Package := ⟨12, 20⟩
  let large : Package := ⟨20, 30⟩
  let (total_shirts, discounted_price) := calculate_total small medium large 15 10 4
  total_shirts = 290 ∧ discounted_price = 475 := by
  sorry

end t_shirt_problem_l885_88550


namespace d_value_approx_l885_88589

-- Define the equation
def equation (d : ℝ) : Prop :=
  4 * ((3.6 * 0.48 * 2.50) / (d * 0.09 * 0.5)) = 3200.0000000000005

-- Theorem statement
theorem d_value_approx :
  ∃ d : ℝ, equation d ∧ abs (d - 0.3) < 0.0000001 :=
sorry

end d_value_approx_l885_88589


namespace tan_alpha_plus_pi_fourth_l885_88594

theorem tan_alpha_plus_pi_fourth (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.cos α = -4/5) :
  Real.tan (α + Real.pi/4) = 1/7 := by
  sorry

end tan_alpha_plus_pi_fourth_l885_88594


namespace extra_discount_is_four_percent_l885_88577

/-- Calculates the percentage of extra discount given initial price, first discount, and final price -/
def extra_discount_percentage (initial_price first_discount final_price : ℚ) : ℚ :=
  let price_after_first_discount := initial_price - first_discount
  let extra_discount_amount := price_after_first_discount - final_price
  (extra_discount_amount / price_after_first_discount) * 100

/-- Theorem stating that the extra discount percentage is 4% given the problem conditions -/
theorem extra_discount_is_four_percent :
  extra_discount_percentage 50 2.08 46 = 4 := by
  sorry

end extra_discount_is_four_percent_l885_88577


namespace role_assignment_count_l885_88585

def number_of_role_assignments (men : ℕ) (women : ℕ) : ℕ :=
  let male_role_assignments := men
  let female_role_assignments := women
  let remaining_actors := men + women - 2
  let either_gender_role_assignments := Nat.choose remaining_actors 4 * Nat.factorial 4
  male_role_assignments * female_role_assignments * either_gender_role_assignments

theorem role_assignment_count :
  number_of_role_assignments 6 7 = 33120 :=
sorry

end role_assignment_count_l885_88585
