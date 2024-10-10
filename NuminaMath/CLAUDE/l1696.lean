import Mathlib

namespace complement_A_intersect_B_l1696_169690

def A : Set ℝ := {x : ℝ | (2*x - 5)*(x + 3) > 0}
def B : Set ℝ := {1, 2, 3, 4, 5}

theorem complement_A_intersect_B : 
  (Set.compl A) ∩ B = {1, 2} := by sorry

end complement_A_intersect_B_l1696_169690


namespace fraction_value_at_x_equals_one_l1696_169675

theorem fraction_value_at_x_equals_one :
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x^2 - 4)
  f 1 = 4 := by
  sorry

end fraction_value_at_x_equals_one_l1696_169675


namespace solution_set_for_a_2_a_value_for_even_function_l1696_169673

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem solution_set_for_a_2 :
  {x : ℝ | f 2 x ≥ 2} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 5/2} := by sorry

-- Part 2
theorem a_value_for_even_function :
  (∀ x, f a x = f a (-x)) → a = -1 := by sorry

end solution_set_for_a_2_a_value_for_even_function_l1696_169673


namespace tomorrow_is_saturday_l1696_169618

-- Define the days of the week
inductive Day :=
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

def add_days (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => next_day (add_days d m)

theorem tomorrow_is_saturday 
  (h : add_days (next_day (next_day Day.Wednesday)) 5 = Day.Monday) : 
  next_day Day.Friday = Day.Saturday :=
by
  sorry

#check tomorrow_is_saturday

end tomorrow_is_saturday_l1696_169618


namespace a_negative_sufficient_not_necessary_l1696_169660

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem a_negative_sufficient_not_necessary :
  (∀ a : ℝ, a < 0 → ∃ x : ℝ, x < 0 ∧ f a x = 0) ∧
  (∃ a : ℝ, a ≥ 0 ∧ ∃ x : ℝ, x < 0 ∧ f a x = 0) := by
  sorry

end a_negative_sufficient_not_necessary_l1696_169660


namespace smallest_n_congruence_l1696_169633

theorem smallest_n_congruence : 
  ∃ (n : ℕ), n > 0 ∧ (7^n ≡ n^5 [ZMOD 3]) ∧ 
  ∀ (m : ℕ), m > 0 ∧ m < n → ¬(7^m ≡ m^5 [ZMOD 3]) :=
by sorry

end smallest_n_congruence_l1696_169633


namespace parabola_directrix_l1696_169601

/-- The equation of a parabola -/
def parabola_eq (x y : ℝ) : Prop := y = 4 * x^2 + 4 * x + 1

/-- The equation of the directrix -/
def directrix_eq (y : ℝ) : Prop := y = 11/16

/-- Theorem stating that the given directrix is correct for the parabola -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_eq x y → ∃ d : ℝ, directrix_eq d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
   ∃ f : ℝ × ℝ, (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.2 - d)^2) :=
sorry

end parabola_directrix_l1696_169601


namespace intersection_of_A_and_B_l1696_169674

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end intersection_of_A_and_B_l1696_169674


namespace complex_calculation_theorem_logarithm_calculation_theorem_l1696_169656

theorem complex_calculation_theorem :
  (2 ^ (1/3) * 3 ^ (1/2)) ^ 6 + (2 * 2 ^ (1/2)) ^ (4/3) - 4 * (16/49) ^ (-1/2) - 2 ^ (1/4) * 8 ^ 0.25 - (-2005) ^ 0 = 100 :=
by sorry

theorem logarithm_calculation_theorem :
  ((1 - Real.log 3 / Real.log 6) ^ 2 + Real.log 2 / Real.log 6 * Real.log 18 / Real.log 6) / (Real.log 4 / Real.log 6) = 1 :=
by sorry

end complex_calculation_theorem_logarithm_calculation_theorem_l1696_169656


namespace volcano_theorem_l1696_169658

def volcano_problem (initial_volcanoes : ℕ) (first_explosion_rate : ℚ) 
  (mid_year_explosion_rate : ℚ) (end_year_explosion_rate : ℚ) (intact_volcanoes : ℕ) : Prop :=
  let remaining_after_first := initial_volcanoes - (initial_volcanoes * first_explosion_rate).floor
  let remaining_after_mid := remaining_after_first - (remaining_after_first * mid_year_explosion_rate).floor
  let final_exploded := (remaining_after_mid * end_year_explosion_rate).floor
  initial_volcanoes - intact_volcanoes = 
    (initial_volcanoes * first_explosion_rate).floor + 
    (remaining_after_first * mid_year_explosion_rate).floor + 
    final_exploded

theorem volcano_theorem : 
  volcano_problem 200 (20/100) (40/100) (50/100) 48 := by
  sorry

end volcano_theorem_l1696_169658


namespace yellow_balls_count_l1696_169663

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 100 ∧ 
  white = 20 ∧ 
  green = 30 ∧ 
  red = 37 ∧ 
  purple = 3 ∧ 
  prob = 6/10 ∧ 
  (white + green : ℚ) / total + (total - white - green - red - purple : ℚ) / total = prob →
  total - white - green - red - purple = 10 := by
  sorry

end yellow_balls_count_l1696_169663


namespace vector_equation_l1696_169635

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (4, 2)

theorem vector_equation : c = 3 • a - b := by sorry

end vector_equation_l1696_169635


namespace quadratic_inequality_range_l1696_169691

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (1/2) * a * x^2 - a * x + 2 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end quadratic_inequality_range_l1696_169691


namespace cubic_function_properties_l1696_169665

-- Define the function f(x) = ax^3
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3

-- Theorem statement
theorem cubic_function_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, x < y → a < 0 → f a x > f a y) ∧
  (∃ x : ℝ, x ≠ 1 ∧ f a x = 3 * a * x - 2 * a) :=
by sorry

end cubic_function_properties_l1696_169665


namespace parabola_hyperbola_intersection_l1696_169638

/-- The parabola equation -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2/3 - x^2 = 1

/-- The directrix of the parabola -/
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

/-- Point F is the focus of the parabola -/
def focus (p : ℝ) (F : ℝ × ℝ) : Prop := F.1 = p/2 ∧ F.2 = 0

/-- Points M and N are the intersections of the directrix and hyperbola -/
def intersection_points (p : ℝ) (M N : ℝ × ℝ) : Prop :=
  directrix p M.1 ∧ hyperbola M.1 M.2 ∧
  directrix p N.1 ∧ hyperbola N.1 N.2

/-- Triangle MNF is a right-angled triangle with F as the right angle vertex -/
def right_triangle (F M N : ℝ × ℝ) : Prop :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2 + (N.1 - F.1)^2 + (N.2 - F.2)^2 =
  (M.1 - N.1)^2 + (M.2 - N.2)^2

theorem parabola_hyperbola_intersection (p : ℝ) (F M N : ℝ × ℝ) :
  parabola p F.1 F.2 →
  focus p F →
  intersection_points p M N →
  right_triangle F M N →
  p = 2 * Real.sqrt 3 := by sorry

end parabola_hyperbola_intersection_l1696_169638


namespace min_value_theorem_l1696_169619

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : 2 * m - n * (-2) - 2 = 0) :
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 2 * m' - n' * (-2) - 2 = 0 → 
    1 / m + 2 / n ≤ 1 / m' + 2 / n') ∧ 
  (1 / m + 2 / n = 3 + 2 * Real.sqrt 2) := by
sorry

end min_value_theorem_l1696_169619


namespace bookcase_shelves_l1696_169670

theorem bookcase_shelves (initial_books : ℕ) (books_bought : ℕ) (books_per_shelf : ℕ) (books_left_over : ℕ) : 
  initial_books = 56 →
  books_bought = 26 →
  books_per_shelf = 20 →
  books_left_over = 2 →
  (initial_books + books_bought - books_left_over) / books_per_shelf = 4 := by
sorry

end bookcase_shelves_l1696_169670


namespace base3_to_base10_conversion_l1696_169693

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

/-- The base-3 representation of the number -/
def base3Digits : List Nat := [1, 2, 0, 1, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Digits = 196 := by
  sorry

end base3_to_base10_conversion_l1696_169693


namespace max_value_expression_l1696_169603

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (⨆ x : ℝ, 3 * (a - x) * (x + Real.sqrt (x^2 + b*x + c^2))) = 
    3/2 * (a^2 + a*b + b^2/4 + c^2) := by
  sorry

end max_value_expression_l1696_169603


namespace fourth_power_sum_l1696_169697

theorem fourth_power_sum (x y z : ℝ) 
  (h1 : x + y + z = 2) 
  (h2 : x^2 + y^2 + z^2 = 5) 
  (h3 : x^3 + y^3 + z^3 = 8) : 
  x^4 + y^4 + z^4 = 113/6 := by
sorry

end fourth_power_sum_l1696_169697


namespace root_sum_squares_l1696_169695

theorem root_sum_squares (p q r s : ℂ) : 
  (p^4 - 24*p^3 + 50*p^2 - 26*p + 7 = 0) →
  (q^4 - 24*q^3 + 50*q^2 - 26*q + 7 = 0) →
  (r^4 - 24*r^3 + 50*r^2 - 26*r + 7 = 0) →
  (s^4 - 24*s^3 + 50*s^2 - 26*s + 7 = 0) →
  (p+q)^2 + (q+r)^2 + (r+s)^2 + (s+p)^2 + (p+r)^2 + (q+s)^2 = 1052 := by
  sorry

end root_sum_squares_l1696_169695


namespace remainder_of_n_l1696_169681

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 11 = 9) (h2 : n^3 % 11 = 5) : n % 11 = 3 := by
  sorry

end remainder_of_n_l1696_169681


namespace salt_water_evaporation_l1696_169664

/-- Given a salt water solution with initial weight of 200 grams and 5% salt concentration,
    if the salt concentration becomes 8% after evaporation,
    then 75 grams of water has evaporated. -/
theorem salt_water_evaporation (initial_weight : ℝ) (initial_concentration : ℝ) 
    (final_concentration : ℝ) (evaporated_water : ℝ) : 
  initial_weight = 200 →
  initial_concentration = 0.05 →
  final_concentration = 0.08 →
  initial_weight * initial_concentration = 
    (initial_weight - evaporated_water) * final_concentration →
  evaporated_water = 75 := by
  sorry

#check salt_water_evaporation

end salt_water_evaporation_l1696_169664


namespace partnership_profit_share_l1696_169605

/-- 
Given a partnership where:
- A invests 3 times as much as B
- B invests two-thirds of what C invests
- The total profit is 6600
Prove that B's share of the profit is 1200
-/
theorem partnership_profit_share 
  (c_investment : ℝ) 
  (total_profit : ℝ) 
  (h1 : total_profit = 6600) 
  (h2 : c_investment > 0) : 
  let b_investment := (2/3) * c_investment
  let a_investment := 3 * b_investment
  let total_investment := a_investment + b_investment + c_investment
  b_investment / total_investment * total_profit = 1200 := by
sorry

end partnership_profit_share_l1696_169605


namespace triangle_trig_max_value_l1696_169687

theorem triangle_trig_max_value (A B C : ℝ) : 
  A = π / 4 → 
  A + B + C = π → 
  0 < B → 
  B < π → 
  0 < C → 
  C < π → 
  ∃ (x : ℝ), x = 2 * Real.cos B + Real.sin (2 * C) ∧ x ≤ 3 / 2 ∧ 
  ∀ (y : ℝ), y = 2 * Real.cos B + Real.sin (2 * C) → y ≤ x :=
by sorry

end triangle_trig_max_value_l1696_169687


namespace sausages_problem_l1696_169611

def sausages_left (initial : ℕ) : ℕ :=
  let after_monday := initial - (2 * initial / 5)
  let after_tuesday := after_monday - (after_monday / 2)
  let after_wednesday := after_tuesday - (after_tuesday / 4)
  let after_thursday := after_wednesday - (after_wednesday / 3)
  after_thursday - (3 * after_thursday / 5)

theorem sausages_problem : sausages_left 1200 = 72 := by
  sorry

end sausages_problem_l1696_169611


namespace unique_integer_product_l1696_169652

/-- A function that returns true if the given number uses each digit from the given list exactly once -/
def uses_digits_once (n : ℕ) (digits : List ℕ) : Prop :=
  sorry

/-- A function that combines two natural numbers into a single number -/
def combine_numbers (a b : ℕ) : ℕ :=
  sorry

theorem unique_integer_product : ∃! n : ℕ, 
  uses_digits_once (combine_numbers (4 * n) (5 * n)) [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧ 
  n = 2469 := by
  sorry

end unique_integer_product_l1696_169652


namespace horner_eval_23_l1696_169642

def horner_polynomial (a b c d x : ℤ) : ℤ := ((a * x + b) * x + c) * x + d

theorem horner_eval_23 :
  let f : ℤ → ℤ := λ x => 7 * x^3 + 3 * x^2 - 5 * x + 11
  let horner : ℤ → ℤ := horner_polynomial 7 3 (-5) 11
  (∀ step : ℤ, step ≠ 85169 → (step = 7 ∨ step = 164 ∨ step = 3762 ∨ step = 86537)) ∧
  f 23 = horner 23 ∧
  f 23 = 86537 := by
sorry

end horner_eval_23_l1696_169642


namespace sqrt_ab_eq_a_plus_b_iff_zero_l1696_169630

theorem sqrt_ab_eq_a_plus_b_iff_zero (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt (a * b) = a + b ↔ a = 0 ∧ b = 0 := by
  sorry

end sqrt_ab_eq_a_plus_b_iff_zero_l1696_169630


namespace tank_capacity_calculation_l1696_169650

/-- Represents a tank with a leak and an inlet pipe. -/
structure Tank where
  capacity : ℝ
  leak_empty_time : ℝ
  inlet_rate : ℝ
  combined_empty_time : ℝ

/-- Theorem stating the relationship between tank properties and its capacity. -/
theorem tank_capacity_calculation (t : Tank)
  (h1 : t.leak_empty_time = 6)
  (h2 : t.inlet_rate = 4.5)
  (h3 : t.combined_empty_time = 8) :
  t.capacity = 6480 / 7 := by
  sorry

end tank_capacity_calculation_l1696_169650


namespace count_arrangements_11250_l1696_169631

def digits : List Nat := [1, 1, 2, 5, 0]

def is_multiple_of_two (n : Nat) : Bool :=
  n % 2 = 0

def is_five_digit (n : Nat) : Bool :=
  n ≥ 10000 ∧ n < 100000

def count_valid_arrangements (ds : List Nat) : Nat :=
  sorry

theorem count_arrangements_11250 : 
  count_valid_arrangements digits = 24 := by sorry

end count_arrangements_11250_l1696_169631


namespace cone_volume_l1696_169617

theorem cone_volume (slant_height height : ℝ) (h1 : slant_height = 15) (h2 : height = 9) :
  let radius := Real.sqrt (slant_height^2 - height^2)
  (1 / 3 : ℝ) * π * radius^2 * height = 432 * π := by
  sorry

end cone_volume_l1696_169617


namespace isosceles_triangle_perimeter_l1696_169678

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 7*x + 10 = 0

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  is_isosceles : base ≠ leg

-- Define the triangle with sides from the quadratic equation
def triangle_from_equation : IsoscelesTriangle :=
  { base := 2,
    leg := 5,
    is_isosceles := by norm_num }

-- State the theorem
theorem isosceles_triangle_perimeter :
  quadratic_equation triangle_from_equation.base ∧
  quadratic_equation triangle_from_equation.leg →
  triangle_from_equation.base + 2 * triangle_from_equation.leg = 12 :=
by
  sorry


end isosceles_triangle_perimeter_l1696_169678


namespace triangle_angle_properties_l1696_169640

theorem triangle_angle_properties (α : Real) 
  (h1 : 0 < α ∧ α < π) -- α is an internal angle of a triangle
  (h2 : Real.sin α + Real.cos α = 1/5) :
  Real.tan α = -4/3 ∧ 
  1 / (Real.cos α ^ 2 - Real.sin α ^ 2) = -25/7 := by
  sorry


end triangle_angle_properties_l1696_169640


namespace max_ratio_squared_l1696_169696

theorem max_ratio_squared (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b)
  (hx : 0 ≤ x ∧ x < a) (hy : 0 ≤ y ∧ y < b) (hx2 : x ≤ 2*a/3)
  (heq : a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b - y)^2) :
  (∃ ρ : ℝ, ∀ a' b' : ℝ, a' / b' ≤ ρ ∧ ρ^2 = 9/5) :=
sorry

end max_ratio_squared_l1696_169696


namespace greater_solution_quadratic_l1696_169624

theorem greater_solution_quadratic (x : ℝ) : 
  x^2 + 20*x - 96 = 0 → x ≤ 4 :=
by
  sorry

end greater_solution_quadratic_l1696_169624


namespace fifteenth_term_of_modified_arithmetic_sequence_l1696_169659

/-- Given an arithmetic sequence with first term 3, second term 15, and third term 27,
    prove that the 15th term is 339 when the common difference is doubled. -/
theorem fifteenth_term_of_modified_arithmetic_sequence :
  ∀ (a : ℕ → ℝ),
    a 1 = 3 →
    a 2 = 15 →
    a 3 = 27 →
    (∀ n : ℕ, a (n + 1) - a n = 2 * (a 2 - a 1)) →
    a 15 = 339 :=
by sorry

end fifteenth_term_of_modified_arithmetic_sequence_l1696_169659


namespace four_students_line_arrangement_l1696_169607

/-- The number of ways to arrange 4 students in a line with restrictions -/
def restricted_arrangements : ℕ := 12

/-- The total number of unrestricted arrangements of 4 students -/
def total_arrangements : ℕ := 24

/-- The number of arrangements where the fourth student is next to at least one other -/
def invalid_arrangements : ℕ := 12

theorem four_students_line_arrangement :
  restricted_arrangements = total_arrangements - invalid_arrangements :=
by sorry

end four_students_line_arrangement_l1696_169607


namespace cheryl_pesto_production_l1696_169648

/-- Prove that Cheryl can make 32 cups of pesto given the harvesting conditions --/
theorem cheryl_pesto_production (basil_per_pesto : ℕ) (basil_per_week : ℕ) (weeks : ℕ)
  (h1 : basil_per_pesto = 4)
  (h2 : basil_per_week = 16)
  (h3 : weeks = 8) :
  (basil_per_week * weeks) / basil_per_pesto = 32 := by
  sorry


end cheryl_pesto_production_l1696_169648


namespace local_value_in_product_l1696_169686

/-- The face value of a digit is the digit itself. -/
def faceValue (digit : ℕ) : ℕ := digit

/-- The local value of a digit in a number is the digit multiplied by its place value. -/
def localValue (digit : ℕ) (placeValue : ℕ) : ℕ := digit * placeValue

/-- The product of two numbers. -/
def product (a b : ℕ) : ℕ := a * b

/-- The theorem stating that the local value of 6 in the product of the face value of 7
    and the local value of 8 in 7098060 is equal to 60. -/
theorem local_value_in_product :
  let number := 7098060
  let faceValue7 := faceValue 7
  let localValue8 := localValue 8 1000
  let prod := product faceValue7 localValue8
  localValue 6 10 = 60 :=
by sorry

end local_value_in_product_l1696_169686


namespace parallel_segments_between_parallel_planes_l1696_169668

/-- Two planes are parallel if they do not intersect -/
def ParallelPlanes (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry

/-- A line segment between two planes -/
def LineSegmentBetweenPlanes (p q : Set (ℝ × ℝ × ℝ)) (s : Set (ℝ × ℝ × ℝ)) : Prop := sorry

/-- Two line segments are parallel -/
def ParallelLineSegments (s t : Set (ℝ × ℝ × ℝ)) : Prop := sorry

/-- The length of a line segment -/
def LengthOfLineSegment (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

theorem parallel_segments_between_parallel_planes 
  (p q : Set (ℝ × ℝ × ℝ)) 
  (s t : Set (ℝ × ℝ × ℝ)) :
  ParallelPlanes p q →
  LineSegmentBetweenPlanes p q s →
  LineSegmentBetweenPlanes p q t →
  ParallelLineSegments s t →
  LengthOfLineSegment s = LengthOfLineSegment t := by
  sorry

end parallel_segments_between_parallel_planes_l1696_169668


namespace ratio_antecedent_proof_l1696_169600

theorem ratio_antecedent_proof (ratio_antecedent ratio_consequent consequent : ℚ) : 
  ratio_antecedent = 4 →
  ratio_consequent = 6 →
  consequent = 75 →
  (ratio_antecedent / ratio_consequent) * consequent = 50 := by
sorry

end ratio_antecedent_proof_l1696_169600


namespace circle_center_on_line_l1696_169657

/-- Given a circle with equation x² + y² - 2ax + 4y - 6 = 0,
    if its center (h, k) satisfies h + 2k + 1 = 0, then a = 3 -/
theorem circle_center_on_line (a : ℝ) :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 - 2*a*x + 4*y - 6 = 0
  let center := fun (h k : ℝ) => ∀ x y, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + (k+2)^2 + 10)
  let on_line := fun (h k : ℝ) => h + 2*k + 1 = 0
  (∃ h k, center h k ∧ on_line h k) → a = 3 :=
by sorry

end circle_center_on_line_l1696_169657


namespace battery_life_is_19_5_hours_l1696_169637

/-- Represents the tablet's battery and usage characteristics -/
structure TabletBattery where
  passive_life : ℝ  -- Battery life in hours when not actively used
  active_life : ℝ   -- Battery life in hours when actively used
  used_time : ℝ     -- Total time the tablet has been on since last charge
  gaming_time : ℝ   -- Time spent gaming since last charge
  charge_rate_passive : ℝ  -- Additional passive battery life gained per hour of charging
  charge_rate_active : ℝ   -- Additional active battery life gained per hour of charging
  charge_time : ℝ   -- Time spent charging the tablet

/-- Calculates the remaining battery life after usage and charging -/
def remaining_battery_life (tb : TabletBattery) : ℝ :=
  sorry

/-- Theorem stating that the remaining battery life is 19.5 hours -/
theorem battery_life_is_19_5_hours (tb : TabletBattery) 
  (h1 : tb.passive_life = 36)
  (h2 : tb.active_life = 6)
  (h3 : tb.used_time = 15)
  (h4 : tb.gaming_time = 1.5)
  (h5 : tb.charge_rate_passive = 2)
  (h6 : tb.charge_rate_active = 0.5)
  (h7 : tb.charge_time = 3) :
  remaining_battery_life tb = 19.5 :=
sorry

end battery_life_is_19_5_hours_l1696_169637


namespace expression_evaluation_l1696_169606

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 2) :
  3 * x^y + 4 * y^x - 2 * x * y = 47 := by
  sorry

end expression_evaluation_l1696_169606


namespace arithmetic_sequence_18th_term_l1696_169646

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_18th_term (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_sum : a 2 + a 5 + a 8 = 33)
  (h_geom_mean : (a 5 + 1)^2 = (a 2 + 1) * (a 8 + 7)) :
  a 18 = 37 := by
sorry

end arithmetic_sequence_18th_term_l1696_169646


namespace ellipse_hyperbola_tangent_l1696_169689

/-- The value of m for which the ellipse x^2 + 9y^2 = 9 is tangent to the hyperbola x^2 - m(y + 3)^2 = 1 -/
theorem ellipse_hyperbola_tangent : ∃ (m : ℝ), 
  (∀ (x y : ℝ), x^2 + 9*y^2 = 9 ∧ x^2 - m*(y + 3)^2 = 1) →
  (∃! (x y : ℝ), x^2 + 9*y^2 = 9 ∧ x^2 - m*(y + 3)^2 = 1) →
  m = 8/9 := by
  sorry

end ellipse_hyperbola_tangent_l1696_169689


namespace range_of_x_l1696_169688

theorem range_of_x (x y : ℝ) (h1 : 2*x - y = 4) (h2 : -2 < y) (h3 : y ≤ 3) :
  1 < x ∧ x ≤ 7/2 := by
  sorry

end range_of_x_l1696_169688


namespace gcd_8512_13832_l1696_169613

theorem gcd_8512_13832 : Nat.gcd 8512 13832 = 1064 := by
  sorry

end gcd_8512_13832_l1696_169613


namespace f_properties_l1696_169672

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 + 1/2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-1/2) (Real.sqrt 3 / 2) ↔
    ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/4) ∧ f x = y) :=
sorry

end f_properties_l1696_169672


namespace line_through_point_parallel_to_line_l1696_169680

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line
  (given_line : Line)
  (given_point : Point)
  (result_line : Line) :
  given_line = Line.mk 1 2 (-1) →
  given_point = Point.mk 1 2 →
  result_line = Line.mk 1 2 (-5) →
  pointOnLine given_point result_line ∧ parallel given_line result_line := by
  sorry

end line_through_point_parallel_to_line_l1696_169680


namespace shipment_weight_problem_l1696_169694

theorem shipment_weight_problem (x y : ℕ) : 
  x + (30 - x) = 30 →  -- Total number of boxes is 30
  10 * x + y * (30 - x) = 18 * 30 →  -- Initial average weight is 18 pounds
  10 * x + y * (15 - x) = 16 * 15 →  -- New average weight after removing 15 heavier boxes
  y = 20 := by sorry

end shipment_weight_problem_l1696_169694


namespace buses_passed_count_l1696_169684

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a bus schedule -/
structure BusSchedule where
  startTime : Time
  interval : Nat

/-- Calculates the number of buses passed during a journey -/
def busesPassed (departureTime : Time) (journeyDuration : Nat) (cityASchedule : BusSchedule) (cityBSchedule : BusSchedule) : Nat :=
  sorry

theorem buses_passed_count :
  let cityASchedule : BusSchedule := ⟨⟨6, 0, by sorry, by sorry⟩, 2⟩
  let cityBSchedule : BusSchedule := ⟨⟨6, 30, by sorry, by sorry⟩, 1⟩
  let departureTime : Time := ⟨14, 30, by sorry, by sorry⟩
  let journeyDuration : Nat := 8
  busesPassed departureTime journeyDuration cityASchedule cityBSchedule = 5 := by
  sorry

end buses_passed_count_l1696_169684


namespace gcd_1855_1120_l1696_169629

theorem gcd_1855_1120 : Nat.gcd 1855 1120 = 35 := by
  sorry

end gcd_1855_1120_l1696_169629


namespace repeating_decimal_sum_l1696_169653

/-- Expresses the sum of three repeating decimals as a rational number -/
theorem repeating_decimal_sum : 
  (2 : ℚ) / 9 + (3 : ℚ) / 99 + (4 : ℚ) / 9999 = 283 / 11111 := by
  sorry

end repeating_decimal_sum_l1696_169653


namespace inscribed_circle_square_side_length_l1696_169645

theorem inscribed_circle_square_side_length 
  (circle_area : ℝ) 
  (h_area : circle_area = 3848.4510006474966) : 
  let square_side := 70
  let π := Real.pi
  circle_area = π * (square_side / 2)^2 := by
  sorry

end inscribed_circle_square_side_length_l1696_169645


namespace product_remainder_seven_l1696_169639

theorem product_remainder_seven (a b : ℕ) (ha : a = 326) (hb : b = 57) :
  (a * b) % 7 = 4 := by
sorry

end product_remainder_seven_l1696_169639


namespace double_root_values_l1696_169651

/-- A polynomial with integer coefficients of the form x^4 + b₃x³ + b₂x² + b₁x + 50 -/
def IntPolynomial (b₃ b₂ b₁ : ℤ) (x : ℝ) : ℝ := x^4 + b₃*x^3 + b₂*x^2 + b₁*x + 50

/-- s is a double root of the polynomial if both the polynomial and its derivative evaluate to 0 at s -/
def IsDoubleRoot (p : ℝ → ℝ) (s : ℝ) : Prop :=
  p s = 0 ∧ (deriv p) s = 0

theorem double_root_values (b₃ b₂ b₁ : ℤ) (s : ℤ) :
  IsDoubleRoot (IntPolynomial b₃ b₂ b₁) s → s = -5 ∨ s = -1 ∨ s = 1 ∨ s = 5 := by
  sorry

end double_root_values_l1696_169651


namespace scientific_notation_proof_l1696_169654

theorem scientific_notation_proof : 
  284000000 = 2.84 * (10 ^ 8) := by
  sorry

end scientific_notation_proof_l1696_169654


namespace simplified_robot_ratio_l1696_169655

/-- The number of animal robots Michael has -/
def michaels_robots : ℕ := 8

/-- The number of animal robots Tom has -/
def toms_robots : ℕ := 16

/-- The ratio of Tom's robots to Michael's robots -/
def robot_ratio : Rat := toms_robots / michaels_robots

theorem simplified_robot_ratio : robot_ratio = 2 / 1 := by
  sorry

end simplified_robot_ratio_l1696_169655


namespace number_of_ways_is_132_l1696_169622

/-- Represents a girl --/
inductive Girl
| Amy
| Beth
| Jo

/-- Represents a song --/
inductive Song
| One
| Two
| Three
| Four

/-- Represents whether a girl likes a song --/
def Likes : Girl → Song → Prop := sorry

/-- No song is liked by all three girls --/
def NoSongLikedByAll : Prop :=
  ∀ s : Song, ¬(Likes Girl.Amy s ∧ Likes Girl.Beth s ∧ Likes Girl.Jo s)

/-- For each pair of girls, there is at least one song liked by those two but disliked by the third --/
def PairwiseLikedSong : Prop :=
  (∃ s : Song, Likes Girl.Amy s ∧ Likes Girl.Beth s ∧ ¬Likes Girl.Jo s) ∧
  (∃ s : Song, Likes Girl.Beth s ∧ Likes Girl.Jo s ∧ ¬Likes Girl.Amy s) ∧
  (∃ s : Song, Likes Girl.Jo s ∧ Likes Girl.Amy s ∧ ¬Likes Girl.Beth s)

/-- The number of ways the girls can like the songs satisfying the conditions --/
def NumberOfWays : ℕ := sorry

/-- The theorem to be proved --/
theorem number_of_ways_is_132 
  (h1 : NoSongLikedByAll) 
  (h2 : PairwiseLikedSong) : 
  NumberOfWays = 132 := by sorry

end number_of_ways_is_132_l1696_169622


namespace rectangle_area_is_eight_l1696_169627

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*y^2 + 8*x - 16*y + 32 = 0

/-- The rectangle's height is twice the diameter of the circle -/
def rectangle_height_condition (height diameter : ℝ) : Prop :=
  height = 2 * diameter

/-- One pair of sides of the rectangle is parallel to the x-axis -/
def rectangle_orientation : Prop :=
  True  -- This condition is implicitly assumed and doesn't affect the calculation

/-- The area of the rectangle given its height and width -/
def rectangle_area (height width : ℝ) : ℝ :=
  height * width

/-- The main theorem stating that the area of the rectangle is 8 square units -/
theorem rectangle_area_is_eight :
  ∃ (x y height width diameter : ℝ),
    circle_equation x y ∧
    rectangle_height_condition height diameter ∧
    rectangle_orientation ∧
    rectangle_area height width = 8 :=
  sorry

end rectangle_area_is_eight_l1696_169627


namespace lottery_probability_l1696_169699

def lottery_size : ℕ := 90
def draw_size : ℕ := 5

def valid_outcomes : ℕ := 3 * (Nat.choose 86 3)

def total_outcomes : ℕ := Nat.choose lottery_size draw_size

theorem lottery_probability : 
  (valid_outcomes : ℚ) / total_outcomes = 258192 / 43949268 := by sorry

end lottery_probability_l1696_169699


namespace games_per_box_l1696_169609

theorem games_per_box (initial_games : ℕ) (sold_games : ℕ) (num_boxes : ℕ) 
  (h1 : initial_games = 35)
  (h2 : sold_games = 19)
  (h3 : num_boxes = 2)
  (h4 : initial_games > sold_games) :
  (initial_games - sold_games) / num_boxes = 8 := by
sorry

end games_per_box_l1696_169609


namespace beatrice_tv_shopping_l1696_169671

theorem beatrice_tv_shopping (first_store : ℕ) (online_store : ℕ) (auction_site : ℕ) :
  first_store = 8 →
  online_store = 3 * first_store →
  first_store + online_store + auction_site = 42 →
  auction_site = 10 := by
sorry

end beatrice_tv_shopping_l1696_169671


namespace dodecagon_vertex_product_l1696_169636

/-- Regular dodecagon in the complex plane -/
structure RegularDodecagon where
  center : ℂ
  vertex : ℂ

/-- The product of the complex representations of all vertices of a regular dodecagon -/
def vertexProduct (d : RegularDodecagon) : ℂ :=
  (d.center + 1)^12 - 1

/-- Theorem: The product of vertices of a regular dodecagon with center (2,1) and a vertex at (3,1) -/
theorem dodecagon_vertex_product :
  let d : RegularDodecagon := { center := 2 + 1*I, vertex := 3 + 1*I }
  vertexProduct d = -2926 - 3452*I :=
by
  sorry

end dodecagon_vertex_product_l1696_169636


namespace saltwater_concentration_l1696_169625

/-- The final concentration of saltwater in a cup after partial overflow and refilling -/
theorem saltwater_concentration (initial_concentration : ℝ) 
  (overflow_ratio : ℝ) (h1 : initial_concentration = 0.16) 
  (h2 : overflow_ratio = 0.1) : 
  initial_concentration * (1 - overflow_ratio) = 8/75 := by
  sorry

end saltwater_concentration_l1696_169625


namespace complex_modulus_l1696_169621

theorem complex_modulus (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) : 
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by sorry

end complex_modulus_l1696_169621


namespace f_negative_one_value_l1696_169685

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_negative_one_value :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is an odd function
  (∀ x : ℝ, x > 0 → f x = 2*x - 1) →  -- Definition of f for positive x
  f (-1) = -1 := by
  sorry

end f_negative_one_value_l1696_169685


namespace sqrt_difference_equality_l1696_169641

theorem sqrt_difference_equality : Real.sqrt (49 + 49) - Real.sqrt (36 + 25) = 7 * Real.sqrt 2 - Real.sqrt 61 := by
  sorry

end sqrt_difference_equality_l1696_169641


namespace partnership_contribution_time_l1696_169682

/-- Proves that given the conditions of the partnership problem, A contributed for 8 months -/
theorem partnership_contribution_time (a_contribution b_contribution total_profit a_share : ℚ)
  (b_time : ℕ) :
  a_contribution = 5000 →
  b_contribution = 6000 →
  b_time = 5 →
  total_profit = 8400 →
  a_share = 4800 →
  ∃ (a_time : ℕ),
    a_time = 8 ∧
    a_share / total_profit = (a_contribution * a_time) / (a_contribution * a_time + b_contribution * b_time) :=
by sorry

end partnership_contribution_time_l1696_169682


namespace suv_city_mpg_l1696_169667

/-- The average miles per gallon (mpg) for an SUV in the city. -/
def city_mpg : ℝ := 12.2

/-- The maximum distance in miles that the SUV can travel on 20 gallons of gasoline. -/
def max_distance : ℝ := 244

/-- The amount of gasoline in gallons used for the maximum distance. -/
def gas_amount : ℝ := 20

/-- Theorem stating that the average mpg in the city for the SUV is 12.2,
    given the maximum distance on 20 gallons of gasoline is 244 miles. -/
theorem suv_city_mpg :
  city_mpg = max_distance / gas_amount :=
by sorry

end suv_city_mpg_l1696_169667


namespace batsman_average_l1696_169602

/-- Represents a batsman's performance --/
structure Batsman where
  innings : ℕ
  runsInLastInning : ℕ
  averageIncrease : ℕ

/-- Calculates the average runs per inning after the last inning --/
def finalAverage (b : Batsman) : ℕ :=
  let previousAverage := b.runsInLastInning - b.averageIncrease
  previousAverage + b.averageIncrease

/-- Theorem: The batsman's average after 17 innings is 40 runs --/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 17)
  (h2 : b.runsInLastInning = 200)
  (h3 : b.averageIncrease = 10) : 
  finalAverage b = 40 := by
  sorry

#eval finalAverage { innings := 17, runsInLastInning := 200, averageIncrease := 10 }

end batsman_average_l1696_169602


namespace count_hundredths_in_half_l1696_169676

theorem count_hundredths_in_half : (0.5 : ℚ) / (0.01 : ℚ) = 50 := by sorry

end count_hundredths_in_half_l1696_169676


namespace symmetry_about_x_equals_one_l1696_169698

/-- Given a real-valued function f, prove that the graphs of y = f(x-1) and y = f(1-x) 
    are symmetric about the line x = 1 -/
theorem symmetry_about_x_equals_one (f : ℝ → ℝ) :
  ∀ (x y : ℝ), y = f (x - 1) ∧ y = f (1 - x) →
  (∃ (x' y' : ℝ), y' = f (x' - 1) ∧ y' = f (1 - x') ∧ 
   x' = 2 - x ∧ y' = y) :=
by sorry

end symmetry_about_x_equals_one_l1696_169698


namespace square_root_three_expansion_l1696_169669

theorem square_root_three_expansion 
  (a b m n : ℕ+) 
  (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n := by
  sorry

end square_root_three_expansion_l1696_169669


namespace sum_b_m_is_neg_eleven_fifths_l1696_169634

/-- Two lines intersecting at a point -/
structure IntersectingLines where
  m : ℚ
  b : ℚ
  x : ℚ
  y : ℚ
  h1 : y = m * x + 3
  h2 : y = 2 * x + b
  h3 : x = 5
  h4 : y = 7

/-- The sum of b and m for the intersecting lines -/
def sum_b_m (l : IntersectingLines) : ℚ := l.b + l.m

/-- Theorem stating that the sum of b and m is -11/5 -/
theorem sum_b_m_is_neg_eleven_fifths (l : IntersectingLines) : 
  sum_b_m l = -11/5 := by
  sorry

end sum_b_m_is_neg_eleven_fifths_l1696_169634


namespace square_hole_reassembly_l1696_169679

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Represents a square with a square hole -/
structure SquareWithHole :=
  (outer_side : ℝ)
  (hole_side : ℝ)
  (hole_position : ℝ × ℝ)

/-- Function to divide a square with a hole into four quadrilaterals -/
def divide_square (s : SquareWithHole) : Fin 4 → Quadrilateral :=
  sorry

/-- Function to check if a set of quadrilaterals can form a square with a hole -/
def can_form_square_with_hole (quads : Fin 4 → Quadrilateral) : Prop :=
  sorry

/-- The main theorem to be proved -/
theorem square_hole_reassembly 
  (s : SquareWithHole) : 
  can_form_square_with_hole (divide_square s) :=
sorry

end square_hole_reassembly_l1696_169679


namespace function_derivative_l1696_169612

/-- Given a function f(x) = α² - cos(x), prove that its derivative f'(x) = sin(x) -/
theorem function_derivative (α : ℝ) : 
  let f : ℝ → ℝ := λ x => α^2 - Real.cos x
  deriv f = λ x => Real.sin x := by
  sorry

end function_derivative_l1696_169612


namespace vector_bc_coordinates_l1696_169643

/-- Given points A, B, and vector AC, prove that vector BC has specific coordinates -/
theorem vector_bc_coordinates (A B C : ℝ × ℝ) (h1 : A = (0, 1)) (h2 : B = (3, 2)) 
  (h3 : C.1 - A.1 = -4 ∧ C.2 - A.2 = -3) : 
  (C.1 - B.1, C.2 - B.2) = (-7, -4) := by
  sorry

#check vector_bc_coordinates

end vector_bc_coordinates_l1696_169643


namespace arc_MTN_range_l1696_169623

/-- Represents a circle rolling along the base of an isosceles triangle -/
structure RollingCircle where
  -- Radius of the circle (equal to the altitude of the triangle)
  radius : ℝ
  -- Base angle of the isosceles triangle
  base_angle : ℝ
  -- Position of the tangent point T along AB (0 ≤ t ≤ 1)
  t : ℝ
  -- Constraint: 0 ≤ t ≤ 1
  t_range : 0 ≤ t ∧ t ≤ 1

/-- Calculates the angle of arc MTN for a given position of the rolling circle -/
def arcMTN (circle : RollingCircle) : ℝ :=
  sorry

/-- Theorem stating that arc MTN varies from 0° to 80° -/
theorem arc_MTN_range (circle : RollingCircle) :
  0 ≤ arcMTN circle ∧ arcMTN circle ≤ 80 ∧
  (∃ c1 : RollingCircle, arcMTN c1 = 0) ∧
  (∃ c2 : RollingCircle, arcMTN c2 = 80) :=
sorry

end arc_MTN_range_l1696_169623


namespace sum_of_four_numbers_l1696_169626

theorem sum_of_four_numbers : 1.84 + 5.23 + 2.41 + 8.64 = 18.12 := by
  sorry

end sum_of_four_numbers_l1696_169626


namespace trigonometric_identity_l1696_169666

theorem trigonometric_identity :
  (Real.sin (160 * π / 180) + Real.sin (40 * π / 180)) *
  (Real.sin (140 * π / 180) + Real.sin (20 * π / 180)) +
  (Real.sin (50 * π / 180) - Real.sin (70 * π / 180)) *
  (Real.sin (130 * π / 180) - Real.sin (110 * π / 180)) = 1 := by
  sorry

end trigonometric_identity_l1696_169666


namespace fraction_independence_l1696_169614

theorem fraction_independence (a b c a₁ b₁ c₁ : ℝ) (h₁ : a₁ ≠ 0) :
  (∀ x, (a * x^2 + b * x + c) / (a₁ * x^2 + b₁ * x + c₁) = (a / a₁)) ↔ 
  (a / a₁ = b / b₁ ∧ b / b₁ = c / c₁) :=
sorry

end fraction_independence_l1696_169614


namespace total_pay_per_episode_l1696_169692

def tv_show_pay (main_characters minor_characters minor_pay major_pay_ratio : ℕ) : ℕ :=
  let minor_total := minor_characters * minor_pay
  let major_total := main_characters * (major_pay_ratio * minor_pay)
  minor_total + major_total

theorem total_pay_per_episode :
  tv_show_pay 5 4 15000 3 = 285000 :=
by
  sorry

end total_pay_per_episode_l1696_169692


namespace bianca_winning_strategy_l1696_169644

/-- Represents a game state with two piles of marbles. -/
structure GameState where
  a : ℕ
  b : ℕ
  sum_eq_100 : a + b = 100

/-- Predicate to check if a move is valid. -/
def valid_move (s : GameState) (pile : ℕ) (remove : ℕ) : Prop :=
  (pile = s.a ∨ pile = s.b) ∧ 0 < remove ∧ remove ≤ pile / 2

/-- Predicate to check if a game state is a winning position for Bianca. -/
def is_winning_for_bianca (s : GameState) : Prop :=
  (s.a = 50 ∧ s.b = 50) ∨
  (s.a = 67 ∧ s.b = 33) ∨
  (s.a = 33 ∧ s.b = 67) ∨
  (s.a = 95 ∧ s.b = 5) ∨
  (s.a = 5 ∧ s.b = 95)

/-- Theorem stating that Bianca has a winning strategy if and only if
    the game state is one of the specified winning positions. -/
theorem bianca_winning_strategy (s : GameState) :
  (∀ (pile remove : ℕ), valid_move s pile remove →
    ∃ (new_s : GameState), ¬is_winning_for_bianca new_s) ↔
  is_winning_for_bianca s :=
sorry

end bianca_winning_strategy_l1696_169644


namespace shirt_fixing_time_l1696_169649

/-- Proves that the time to fix a shirt is 1.5 hours given the problem conditions --/
theorem shirt_fixing_time (num_shirts : ℕ) (num_pants : ℕ) (hourly_rate : ℚ) (total_cost : ℚ) :
  num_shirts = 10 →
  num_pants = 12 →
  hourly_rate = 30 →
  total_cost = 1530 →
  ∃ (time_per_shirt : ℚ),
    time_per_shirt = 3/2 ∧
    total_cost = hourly_rate * (num_shirts * time_per_shirt + num_pants * (2 * time_per_shirt)) :=
by sorry

end shirt_fixing_time_l1696_169649


namespace problem_solution_l1696_169628

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 3*x + 3/x + 1/x^2 = 26)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by sorry

end problem_solution_l1696_169628


namespace smallest_integer_solution_inequality_l1696_169661

theorem smallest_integer_solution_inequality (x : ℤ) :
  (∀ y : ℤ, 3 * y ≥ y - 5 → y ≥ -2) ∧
  (3 * (-2) ≥ -2 - 5) :=
sorry

end smallest_integer_solution_inequality_l1696_169661


namespace rain_given_northeast_wind_l1696_169632

/-- Probability of northeast winds blowing -/
def P_A : ℝ := 0.7

/-- Probability of rain -/
def P_B : ℝ := 0.8

/-- Probability of both northeast winds blowing and rain -/
def P_AB : ℝ := 0.65

/-- Theorem: The conditional probability of rain given northeast winds is 13/14 -/
theorem rain_given_northeast_wind :
  P_AB / P_A = 13 / 14 := by sorry

end rain_given_northeast_wind_l1696_169632


namespace max_value_implies_a_l1696_169604

def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

theorem max_value_implies_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, f a x ≤ 3) ∧ 
  (∃ x ∈ Set.Icc 0 2, f a x = 3) → 
  a = 5 - Real.sqrt 10 ∨ a = 1 + Real.sqrt 2 := by
  sorry

end max_value_implies_a_l1696_169604


namespace exam_proctoring_arrangements_l1696_169616

def female_teachers : ℕ := 2
def male_teachers : ℕ := 5
def total_teachers : ℕ := female_teachers + male_teachers
def stationary_positions : ℕ := 2

theorem exam_proctoring_arrangements :
  (female_teachers * (total_teachers - 1).choose stationary_positions) = 42 := by
  sorry

end exam_proctoring_arrangements_l1696_169616


namespace bricks_required_l1696_169615

/-- The number of bricks required to pave a rectangular courtyard -/
theorem bricks_required (courtyard_length courtyard_width brick_length brick_width : ℝ) :
  courtyard_length = 28 →
  courtyard_width = 13 →
  brick_length = 0.22 →
  brick_width = 0.12 →
  ↑(⌈(courtyard_length * courtyard_width * 10000) / (brick_length * brick_width)⌉) = 13788 := by
  sorry

end bricks_required_l1696_169615


namespace triangle_theorem_l1696_169620

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (Real.cos t.A * Real.cos t.B - Real.sin t.A * Real.sin t.B) = Real.cos (2 * t.C))
  (h2 : 2 * t.c = t.a + t.b)
  (h3 : t.a * t.b * Real.cos t.C = 18) :
  t.C = Real.pi / 3 ∧ t.c = 6 := by
  sorry

end triangle_theorem_l1696_169620


namespace simplify_expression_l1696_169662

theorem simplify_expression : 
  (625 : ℝ) ^ (1/4 : ℝ) * (256 : ℝ) ^ (1/2 : ℝ) = 80 :=
by
  have h1 : (625 : ℝ) = 5^4 := by norm_num
  have h2 : (256 : ℝ) = 2^8 := by norm_num
  sorry

end simplify_expression_l1696_169662


namespace same_solution_implies_c_value_l1696_169647

theorem same_solution_implies_c_value (x c : ℚ) : 
  (3 * x + 9 = 0) ∧ (c * x^2 - 7 = 6) → c = 13/9 := by sorry

end same_solution_implies_c_value_l1696_169647


namespace like_terms_exponent_l1696_169610

theorem like_terms_exponent (m n : ℕ) : 
  (∃ (x y : ℝ), -7 * x^(m+2) * y^2 = -3 * x^3 * y^n) → m^n = 1 := by
sorry

end like_terms_exponent_l1696_169610


namespace cube_root_equation_l1696_169677

theorem cube_root_equation (a b : ℝ) :
  let z : ℝ := (a + (a^2 + b^3)^(1/2))^(1/3) - ((a^2 + b^3)^(1/2) - a)^(1/3)
  z^3 + 3*b*z - 2*a = 0 := by sorry

end cube_root_equation_l1696_169677


namespace sum_of_squares_lower_bound_l1696_169683

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end sum_of_squares_lower_bound_l1696_169683


namespace marble_probability_l1696_169608

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) :
  total = 100 →
  p_white = 1/4 →
  p_green = 1/5 →
  ∃ (p_red_blue : ℚ), p_red_blue = 11/20 ∧ 
    p_white + p_green + p_red_blue = 1 :=
sorry

end marble_probability_l1696_169608
