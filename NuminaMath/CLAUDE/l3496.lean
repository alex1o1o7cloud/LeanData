import Mathlib

namespace NUMINAMATH_CALUDE_tan_40_plus_6_sin_40_l3496_349693

theorem tan_40_plus_6_sin_40 : 
  Real.tan (40 * π / 180) + 6 * Real.sin (40 * π / 180) = 
    Real.sqrt 3 + Real.cos (10 * π / 180) / Real.cos (40 * π / 180) := by sorry

end NUMINAMATH_CALUDE_tan_40_plus_6_sin_40_l3496_349693


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l3496_349675

/-- Given an initial angle of 60 degrees that is rotated 520 degrees clockwise,
    the resulting acute angle is 100 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 60 →
  rotation = 520 →
  (initial_angle + rotation) % 360 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l3496_349675


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l3496_349674

/-- A parabola with equation y = 2x^2 + 8x + m -/
structure Parabola where
  m : ℝ

/-- Predicate to check if a parabola has only two common points with the coordinate axes -/
def has_two_axis_intersections (p : Parabola) : Prop :=
  -- This is a placeholder for the actual condition
  True

/-- Theorem stating that if a parabola y = 2x^2 + 8x + m has only two common points
    with the coordinate axes, then m = 0 or m = 8 -/
theorem parabola_intersection_theorem (p : Parabola) :
  has_two_axis_intersections p → p.m = 0 ∨ p.m = 8 := by
  sorry


end NUMINAMATH_CALUDE_parabola_intersection_theorem_l3496_349674


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l3496_349668

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), 
    (7 * a) % 77 = 1 ∧ 
    (13 * b) % 77 = 1 ∧ 
    ((3 * a + 9 * b) % 77) = 10 :=
by sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l3496_349668


namespace NUMINAMATH_CALUDE_unique_hair_color_assignment_l3496_349659

/-- Represents the three people in the problem -/
inductive Person : Type
  | Belokurov : Person
  | Chernov : Person
  | Ryzhov : Person

/-- Represents the three hair colors in the problem -/
inductive HairColor : Type
  | Blond : HairColor
  | Brunette : HairColor
  | RedHaired : HairColor

/-- Represents the assignment of hair colors to people -/
def hairColorAssignment : Person → HairColor
  | Person.Belokurov => HairColor.RedHaired
  | Person.Chernov => HairColor.Blond
  | Person.Ryzhov => HairColor.Brunette

/-- Condition: No person has a hair color matching their surname -/
def noMatchingSurname (assignment : Person → HairColor) : Prop :=
  assignment Person.Belokurov ≠ HairColor.Blond ∧
  assignment Person.Chernov ≠ HairColor.Brunette ∧
  assignment Person.Ryzhov ≠ HairColor.RedHaired

/-- Condition: The brunette is not Belokurov -/
def brunetteNotBelokurov (assignment : Person → HairColor) : Prop :=
  assignment Person.Belokurov ≠ HairColor.Brunette

/-- Condition: All three hair colors are represented -/
def allColorsRepresented (assignment : Person → HairColor) : Prop :=
  (∃ p, assignment p = HairColor.Blond) ∧
  (∃ p, assignment p = HairColor.Brunette) ∧
  (∃ p, assignment p = HairColor.RedHaired)

/-- Main theorem: The given hair color assignment is the only one satisfying all conditions -/
theorem unique_hair_color_assignment :
  ∀ (assignment : Person → HairColor),
    noMatchingSurname assignment ∧
    brunetteNotBelokurov assignment ∧
    allColorsRepresented assignment →
    assignment = hairColorAssignment :=
by sorry

end NUMINAMATH_CALUDE_unique_hair_color_assignment_l3496_349659


namespace NUMINAMATH_CALUDE_ratio_sum_to_y_l3496_349662

theorem ratio_sum_to_y (w x y : ℝ) (hw_x : w / x = 1 / 3) (hw_y : w / y = 3 / 4) :
  (x + y) / y = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_to_y_l3496_349662


namespace NUMINAMATH_CALUDE_initial_bananas_count_l3496_349606

/-- The number of bananas Elizabeth bought initially -/
def initial_bananas : ℕ := sorry

/-- The number of bananas Elizabeth ate -/
def eaten_bananas : ℕ := 4

/-- The number of bananas Elizabeth has left -/
def remaining_bananas : ℕ := 8

/-- Theorem stating that the initial number of bananas is 12 -/
theorem initial_bananas_count : initial_bananas = 12 := by sorry

end NUMINAMATH_CALUDE_initial_bananas_count_l3496_349606


namespace NUMINAMATH_CALUDE_smallest_valid_n_l3496_349669

def is_valid_n (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  2 * n = 100 * c + 10 * b + a + 5

theorem smallest_valid_n :
  ∃ (n : ℕ), is_valid_n n ∧ ∀ m, is_valid_n m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l3496_349669


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_value_l3496_349621

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - (k^2 - 1) * x^2 - k^2 + 2

-- Define the derivative of f(x)
def f_deriv (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * (k^2 - 1) * x

-- Theorem statement
theorem tangent_perpendicular_implies_a_value (k : ℝ) (a : ℝ) (b : ℝ) :
  f k 1 = a →                   -- Point (1, a) is on the graph of f
  f_deriv k 1 = -1 →            -- Tangent line is perpendicular to x - y + b = 0
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_value_l3496_349621


namespace NUMINAMATH_CALUDE_tree_planting_optimization_l3496_349680

/-- Tree planting activity optimization problem -/
theorem tree_planting_optimization (total_families : ℕ) 
  (silver_poplars : ℕ) (purple_plums : ℕ) 
  (silver_poplar_time : ℚ) (purple_plum_time : ℚ) :
  total_families = 65 →
  silver_poplars = 150 →
  purple_plums = 160 →
  silver_poplar_time = 2/5 →
  purple_plum_time = 3/5 →
  ∃ (group_a_families : ℕ) (duration : ℚ),
    group_a_families = 25 ∧
    duration = 12/5 ∧
    group_a_families ≤ total_families ∧
    (group_a_families : ℚ) * silver_poplars * silver_poplar_time = 
      (total_families - group_a_families : ℚ) * purple_plums * purple_plum_time ∧
    duration = (group_a_families : ℚ) * silver_poplars * silver_poplar_time / group_a_families ∧
    ∀ (other_group_a : ℕ) (other_duration : ℚ),
      other_group_a ≤ total_families →
      (other_group_a : ℚ) * silver_poplars * silver_poplar_time = 
        (total_families - other_group_a : ℚ) * purple_plums * purple_plum_time →
      other_duration = (other_group_a : ℚ) * silver_poplars * silver_poplar_time / other_group_a →
      duration ≤ other_duration :=
by
  sorry

end NUMINAMATH_CALUDE_tree_planting_optimization_l3496_349680


namespace NUMINAMATH_CALUDE_min_value_theorem_l3496_349623

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  2 * a + 3 * b + 6 * c ≥ 27 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 27 ∧ 2 * a₀ + 3 * b₀ + 6 * c₀ = 27 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3496_349623


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_square_sum_l3496_349630

theorem arithmetic_geometric_mean_square_sum (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 125) :
  x^2 + y^2 = 1350 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_square_sum_l3496_349630


namespace NUMINAMATH_CALUDE_horner_method_v1_l3496_349691

def f (x : ℝ) : ℝ := 12 + 3*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v1 (a₆ a₅ : ℝ) (x : ℝ) : ℝ := a₆ * x + a₅

theorem horner_method_v1 : 
  let a₆ := 3
  let a₅ := 5
  let x := -4
  horner_v1 a₆ a₅ x = -7 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v1_l3496_349691


namespace NUMINAMATH_CALUDE_correct_machines_in_first_scenario_l3496_349611

/-- The number of machines in the first scenario -/
def machines_in_first_scenario : ℕ := 5

/-- The number of units produced in the first scenario -/
def units_first_scenario : ℕ := 20

/-- The number of hours in the first scenario -/
def hours_first_scenario : ℕ := 10

/-- The number of machines in the second scenario -/
def machines_second_scenario : ℕ := 10

/-- The number of units produced in the second scenario -/
def units_second_scenario : ℕ := 100

/-- The number of hours in the second scenario -/
def hours_second_scenario : ℕ := 25

/-- The production rate per machine is constant across both scenarios -/
axiom production_rate_constant : 
  (units_first_scenario : ℚ) / (machines_in_first_scenario * hours_first_scenario) = 
  (units_second_scenario : ℚ) / (machines_second_scenario * hours_second_scenario)

theorem correct_machines_in_first_scenario : 
  machines_in_first_scenario = 5 := by sorry

end NUMINAMATH_CALUDE_correct_machines_in_first_scenario_l3496_349611


namespace NUMINAMATH_CALUDE_magnitude_of_3_minus_4i_l3496_349631

theorem magnitude_of_3_minus_4i : Complex.abs (3 - 4 * Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_3_minus_4i_l3496_349631


namespace NUMINAMATH_CALUDE_complex_number_problem_l3496_349650

theorem complex_number_problem :
  let z : ℂ := ((1 - I)^2 + 3*(1 + I)) / (2 - I)
  ∃ (a b : ℝ), z^2 + a*z + b = 1 - I ∧ z = 1 + I ∧ a = -3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3496_349650


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_18_30_45_l3496_349652

/-- The sum of the greatest common factor and least common multiple of 18, 30, and 45 is 93 -/
theorem gcd_lcm_sum_18_30_45 : 
  (Nat.gcd 18 (Nat.gcd 30 45) + Nat.lcm 18 (Nat.lcm 30 45)) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_18_30_45_l3496_349652


namespace NUMINAMATH_CALUDE_total_pastries_count_l3496_349698

/-- The number of mini cupcakes Lola baked -/
def lola_cupcakes : ℕ := 13

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
def total_pastries : ℕ := lola_cupcakes + lola_poptarts + lola_pies + lulu_cupcakes + lulu_poptarts + lulu_pies

theorem total_pastries_count : total_pastries = 73 := by
  sorry

end NUMINAMATH_CALUDE_total_pastries_count_l3496_349698


namespace NUMINAMATH_CALUDE_bisection_method_step_l3496_349656

def f (x : ℝ) := x^5 + 8*x^3 - 1

theorem bisection_method_step (h1 : f 0 < 0) (h2 : f 0.5 > 0) :
  ∃ x₀ ∈ Set.Ioo 0 0.5, f x₀ = 0 ∧ 0.25 = (0 + 0.5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_bisection_method_step_l3496_349656


namespace NUMINAMATH_CALUDE_peaches_picked_proof_l3496_349661

/-- The number of peaches Mike had initially -/
def initial_peaches : ℕ := 34

/-- The total number of peaches Mike has now -/
def total_peaches : ℕ := 86

/-- The number of peaches Mike picked -/
def picked_peaches : ℕ := total_peaches - initial_peaches

theorem peaches_picked_proof :
  picked_peaches = total_peaches - initial_peaches :=
by sorry

end NUMINAMATH_CALUDE_peaches_picked_proof_l3496_349661


namespace NUMINAMATH_CALUDE_carries_shopping_money_l3496_349696

theorem carries_shopping_money (initial_amount : ℕ) (sweater_cost : ℕ) (tshirt_cost : ℕ) (shoes_cost : ℕ) 
  (h1 : initial_amount = 91)
  (h2 : sweater_cost = 24)
  (h3 : tshirt_cost = 6)
  (h4 : shoes_cost = 11) :
  initial_amount - (sweater_cost + tshirt_cost + shoes_cost) = 50 := by
  sorry

end NUMINAMATH_CALUDE_carries_shopping_money_l3496_349696


namespace NUMINAMATH_CALUDE_det_4523_equals_2_l3496_349663

def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem det_4523_equals_2 : det2x2 4 5 2 3 = 2 := by sorry

end NUMINAMATH_CALUDE_det_4523_equals_2_l3496_349663


namespace NUMINAMATH_CALUDE_dog_max_distance_l3496_349694

/-- The maximum distance a dog can be from the origin when tied to a post -/
theorem dog_max_distance (post_x post_y rope_length : ℝ) :
  post_x = 6 ∧ post_y = 8 ∧ rope_length = 15 →
  ∃ (max_distance : ℝ),
    max_distance = 25 ∧
    ∀ (x y : ℝ),
      (x - post_x)^2 + (y - post_y)^2 ≤ rope_length^2 →
      x^2 + y^2 ≤ max_distance^2 :=
by sorry

end NUMINAMATH_CALUDE_dog_max_distance_l3496_349694


namespace NUMINAMATH_CALUDE_sum_of_cubes_difference_l3496_349603

theorem sum_of_cubes_difference (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 2700 → a + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_difference_l3496_349603


namespace NUMINAMATH_CALUDE_not_prime_sum_products_l3496_349681

theorem not_prime_sum_products (a b c d : ℤ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) 
  (h5 : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : 
  ¬ Prime (a * b + c * d) := by
sorry

end NUMINAMATH_CALUDE_not_prime_sum_products_l3496_349681


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l3496_349665

theorem simplify_algebraic_expression (a b : ℝ) (h : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b^2 - b^3) / (a * b - a^3) = 2 * a * (a - b) / b :=
sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l3496_349665


namespace NUMINAMATH_CALUDE_work_completion_time_l3496_349620

/-- Given a work project with the following conditions:
  1. The initial number of workers was 10.
  2. Half of the workers (5) were absent.
  3. The remaining workers completed the work in 10 days.
  Prove that the original planned completion time was 5 days. -/
theorem work_completion_time 
  (initial_workers : ℕ) 
  (absent_workers : ℕ) 
  (actual_completion_time : ℕ) : 
  initial_workers = 10 →
  absent_workers = 5 →
  actual_completion_time = 10 →
  (initial_workers - absent_workers) * actual_completion_time = initial_workers * 5 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3496_349620


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l3496_349645

-- Problem 1
theorem equation_one_solution (x : ℝ) : 
  (5 / (x - 1) = 1 / (2 * x + 1)) ↔ x = -2/3 :=
sorry

-- Problem 2
theorem equation_two_no_solution : 
  ¬∃ (x : ℝ), (1 / (x - 2) + 2 = (1 - x) / (2 - x)) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l3496_349645


namespace NUMINAMATH_CALUDE_bookshop_inventory_l3496_349690

-- Define the total number of books
def total_books : ℕ := 300

-- Define the percentage of books displayed
def display_percentage : ℚ := 30 / 100

-- Define the number of books in storage
def storage_books : ℕ := 210

-- Theorem statement
theorem bookshop_inventory :
  (1 - display_percentage) * total_books = storage_books := by
  sorry

end NUMINAMATH_CALUDE_bookshop_inventory_l3496_349690


namespace NUMINAMATH_CALUDE_lg_expression_equals_one_l3496_349637

-- Define the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_one :
  lg 2 ^ 2 * lg 250 + lg 5 ^ 2 * lg 40 = 1 := by sorry

end NUMINAMATH_CALUDE_lg_expression_equals_one_l3496_349637


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_is_56_l3496_349666

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSide := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSide) * (box.width / cubeSide) * (box.depth / cubeSide)

/-- Theorem stating that the smallest number of cubes to fill the given box is 56 -/
theorem smallest_number_of_cubes_is_56 :
  smallestNumberOfCubes ⟨35, 20, 10⟩ = 56 := by
  sorry

#eval smallestNumberOfCubes ⟨35, 20, 10⟩

end NUMINAMATH_CALUDE_smallest_number_of_cubes_is_56_l3496_349666


namespace NUMINAMATH_CALUDE_intersection_A_B_when_m_zero_range_of_m_for_necessary_condition_l3496_349644

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 < 0
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

-- Theorem for part I
theorem intersection_A_B_when_m_zero :
  A ∩ B 0 = {x | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem for part II
theorem range_of_m_for_necessary_condition :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) ↔ m ≥ 4 ∨ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_m_zero_range_of_m_for_necessary_condition_l3496_349644


namespace NUMINAMATH_CALUDE_binary_1101011_equals_base5_412_l3496_349605

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem binary_1101011_equals_base5_412 : 
  decimal_to_base5 (binary_to_decimal [true, true, false, true, false, true, true]) = [4, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_binary_1101011_equals_base5_412_l3496_349605


namespace NUMINAMATH_CALUDE_one_fourths_in_five_thirds_l3496_349627

theorem one_fourths_in_five_thirds : (5 : ℚ) / 3 / (1 / 4) = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_one_fourths_in_five_thirds_l3496_349627


namespace NUMINAMATH_CALUDE_stirling_duality_l3496_349616

/-- Stirling number of the second kind -/
def stirling2 (N n : ℕ) : ℕ := sorry

/-- Stirling number of the first kind -/
def stirling1 (n M : ℕ) : ℤ := sorry

/-- Kronecker delta -/
def kroneckerDelta (N M : ℕ) : ℕ :=
  if N = M then 1 else 0

/-- The duality property of Stirling numbers -/
theorem stirling_duality (N M : ℕ) :
  (∑' n, (stirling2 N n : ℤ) * stirling1 n M) = kroneckerDelta N M := by
  sorry

end NUMINAMATH_CALUDE_stirling_duality_l3496_349616


namespace NUMINAMATH_CALUDE_consecutive_color_draw_probability_l3496_349648

def orange_chips : Nat := 4
def green_chips : Nat := 3
def blue_chips : Nat := 5
def total_chips : Nat := orange_chips + green_chips + blue_chips

def satisfying_arrangements : Nat := orange_chips.factorial * green_chips.factorial * blue_chips.factorial

theorem consecutive_color_draw_probability : 
  (satisfying_arrangements : ℚ) / (total_chips.factorial : ℚ) = 1 / 665280 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_color_draw_probability_l3496_349648


namespace NUMINAMATH_CALUDE_bill_sunday_miles_bill_sunday_miles_proof_l3496_349613

theorem bill_sunday_miles : ℕ → ℕ → ℕ → Prop :=
  fun bill_saturday bill_sunday julia_sunday =>
    (bill_sunday = bill_saturday + 4) →
    (julia_sunday = 2 * bill_sunday) →
    (bill_saturday + bill_sunday + julia_sunday = 28) →
    bill_sunday = 8

-- The proof would go here, but we'll skip it as requested
theorem bill_sunday_miles_proof : ∃ (bill_saturday bill_sunday julia_sunday : ℕ),
  bill_sunday_miles bill_saturday bill_sunday julia_sunday :=
sorry

end NUMINAMATH_CALUDE_bill_sunday_miles_bill_sunday_miles_proof_l3496_349613


namespace NUMINAMATH_CALUDE_min_people_with_hat_and_glove_l3496_349651

theorem min_people_with_hat_and_glove (n : ℕ) (gloves hats both : ℕ) : 
  n > 0 → 
  gloves = (3 * n) / 8 → 
  hats = (5 * n) / 6 → 
  both ≥ gloves + hats - n → 
  both ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_people_with_hat_and_glove_l3496_349651


namespace NUMINAMATH_CALUDE_zachary_needs_additional_money_l3496_349619

def football_cost : ℚ := 3.75
def shorts_cost : ℚ := 2.40
def shoes_cost : ℚ := 11.85
def zachary_money : ℚ := 10.00

theorem zachary_needs_additional_money :
  football_cost + shorts_cost + shoes_cost - zachary_money = 7.00 := by
  sorry

end NUMINAMATH_CALUDE_zachary_needs_additional_money_l3496_349619


namespace NUMINAMATH_CALUDE_ellipse_equation_proof_l3496_349607

/-- Represents an ellipse -/
structure Ellipse where
  center : ℝ × ℝ
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  point : ℝ × ℝ

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / 81 + y^2 / 45 = 1

/-- Given ellipse satisfies the conditions -/
def given_ellipse : Ellipse :=
  { center := (0, 0)
  , foci := ((-3, 0), (3, 0))
  , point := (3, 8) }

/-- Theorem: The equation of the given ellipse is x²/81 + y²/45 = 1 -/
theorem ellipse_equation_proof :
  ellipse_equation given_ellipse (given_ellipse.point.1) (given_ellipse.point.2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_proof_l3496_349607


namespace NUMINAMATH_CALUDE_system_solution_range_l3496_349670

theorem system_solution_range (x y a : ℝ) :
  (2 * x + y = 3 - a) →
  (x + 2 * y = 4 + 2 * a) →
  (x + y < 1) →
  (a < -4) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_range_l3496_349670


namespace NUMINAMATH_CALUDE_triangle_area_in_rectangle_l3496_349640

/-- Given a rectangle with dimensions 30 cm by 28 cm containing four congruent right-angled triangles,
    where the hypotenuse of each triangle forms part of the rectangle's perimeter,
    the total area of the four triangles is 56 cm². -/
theorem triangle_area_in_rectangle :
  ∀ (a b : ℝ),
  a > 0 → b > 0 →
  a + 2 * b = 30 →
  2 * b = 28 →
  4 * (1/2 * a * b) = 56 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_in_rectangle_l3496_349640


namespace NUMINAMATH_CALUDE_n_value_proof_l3496_349654

theorem n_value_proof (n : ℕ) 
  (h1 : ∃ k : ℕ, 31 * 13 * n = k)
  (h2 : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 2*x + 2*y + z = n)
  (h3 : (Finset.filter (λ (t : ℕ × ℕ × ℕ) => 
         t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧ 2*t.1 + 2*t.2.1 + t.2.2 = n) 
         (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card = 28) :
  n = 17 ∨ n = 18 := by
sorry

end NUMINAMATH_CALUDE_n_value_proof_l3496_349654


namespace NUMINAMATH_CALUDE_secretary_work_hours_l3496_349636

theorem secretary_work_hours (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / b = 2 / 3 →
  b / c = 3 / 5 →
  c = 40 →
  a + b + c = 80 :=
by sorry

end NUMINAMATH_CALUDE_secretary_work_hours_l3496_349636


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l3496_349687

theorem factor_implies_m_value (x y m : ℝ) : 
  (∃ k : ℝ, (1 - 2*x + y) * k = 4*x*y - 4*x^2 - y^2 - m) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l3496_349687


namespace NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_in_triangle_l3496_349685

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A centrally symmetric polygon -/
structure CentrallySymmetricPolygon where
  vertices : List (ℝ × ℝ)
  center : ℝ × ℝ
  isSymmetric : ∀ v ∈ vertices, ∃ v' ∈ vertices, v' = (2 * center.1 - v.1, 2 * center.2 - v.2)

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isPointInTriangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Check if a polygon is inside a triangle -/
def isPolygonInTriangle (p : CentrallySymmetricPolygon) (t : Triangle) : Prop :=
  ∀ v ∈ p.vertices, isPointInTriangle v t

/-- The area of a centrally symmetric polygon -/
def polygonArea (p : CentrallySymmetricPolygon) : ℝ := sorry

/-- The theorem stating that the largest centrally symmetric polygon 
    inscribed in a triangle has 2/3 the area of the triangle -/
theorem largest_centrally_symmetric_polygon_in_triangle 
  (t : Triangle) : 
  (∃ p : CentrallySymmetricPolygon, 
    isPolygonInTriangle p t ∧ 
    (∀ q : CentrallySymmetricPolygon, 
      isPolygonInTriangle q t → polygonArea q ≤ polygonArea p)) → 
  (∃ p : CentrallySymmetricPolygon, 
    isPolygonInTriangle p t ∧ 
    polygonArea p = (2/3) * triangleArea t) := by
  sorry

end NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_in_triangle_l3496_349685


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l3496_349643

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x + 2*y, x - 2*y)

theorem preimage_of_3_1 (x y : ℝ) :
  f (x, y) = (3, 1) → (x, y) = (2, 1/2) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l3496_349643


namespace NUMINAMATH_CALUDE_f_increasing_decreasing_l3496_349639

-- Define the function f(x) = x^3 - x^2 - x
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

-- Theorem statement
theorem f_increasing_decreasing :
  (∀ x < -1/3, f' x > 0) ∧
  (∀ x ∈ Set.Ioo (-1/3) 1, f' x < 0) ∧
  (∀ x > 1, f' x > 0) ∧
  (f 1 = -1) ∧
  (f' 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_decreasing_l3496_349639


namespace NUMINAMATH_CALUDE_derivative_at_one_l3496_349683

/-- Given f(x) = 2x³ + x² - 5, prove that f'(1) = 8 -/
theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x^3 + x^2 - 5) : 
  deriv f 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3496_349683


namespace NUMINAMATH_CALUDE_max_x_minus_y_l3496_349646

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), w = x - y → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l3496_349646


namespace NUMINAMATH_CALUDE_last_four_digits_of_m_l3496_349655

theorem last_four_digits_of_m (M : ℕ) (h1 : M > 0) 
  (h2 : ∃ (a b c d e : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    M % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (M^2) % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e) :
  M % 10000 = 9687 := by
sorry

end NUMINAMATH_CALUDE_last_four_digits_of_m_l3496_349655


namespace NUMINAMATH_CALUDE_rainfall_water_level_rise_l3496_349679

/-- Given 15 liters of rainfall per square meter, the rise in water level in a pool is 1.5 cm. -/
theorem rainfall_water_level_rise :
  let rainfall_per_sqm : ℝ := 15  -- liters per square meter
  let liters_to_cubic_cm : ℝ := 1000  -- 1 liter = 1000 cm³
  let sqm_to_sqcm : ℝ := 10000  -- 1 m² = 10000 cm²
  rainfall_per_sqm * liters_to_cubic_cm / sqm_to_sqcm = 1.5  -- cm
  := by sorry

end NUMINAMATH_CALUDE_rainfall_water_level_rise_l3496_349679


namespace NUMINAMATH_CALUDE_three_roots_implies_a_plus_minus_four_l3496_349684

theorem three_roots_implies_a_plus_minus_four (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℝ, |x^2 + a*x| = 4 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) →
  a = 4 ∨ a = -4 :=
by sorry

end NUMINAMATH_CALUDE_three_roots_implies_a_plus_minus_four_l3496_349684


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l3496_349677

theorem binomial_expansion_problem (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (Real.sqrt 5 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -64 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l3496_349677


namespace NUMINAMATH_CALUDE_intersection_exists_l3496_349642

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2 / 9 - x^2 / 4 = 1

/-- The line equation -/
def line (k x y : ℝ) : Prop := y = k * x

/-- The theorem statement -/
theorem intersection_exists : ∃ k : ℝ, 0 < k ∧ k < 2 ∧ 
  ∃ x y : ℝ, hyperbola x y ∧ line k x y :=
sorry

end NUMINAMATH_CALUDE_intersection_exists_l3496_349642


namespace NUMINAMATH_CALUDE_sum_of_four_with_common_divisors_l3496_349657

/-- A function that checks if two positive integers have a common divisor greater than 1 -/
def has_common_divisor_gt_1 (a b : ℕ+) : Prop :=
  ∃ (d : ℕ), d > 1 ∧ d ∣ a.val ∧ d ∣ b.val

/-- A function that checks if a positive integer can be expressed as the sum of four positive integers
    where each pair has a common divisor greater than 1 -/
def is_sum_of_four_with_common_divisors (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ+), 
    n = a.val + b.val + c.val + d.val ∧
    has_common_divisor_gt_1 a b ∧
    has_common_divisor_gt_1 a c ∧
    has_common_divisor_gt_1 a d ∧
    has_common_divisor_gt_1 b c ∧
    has_common_divisor_gt_1 b d ∧
    has_common_divisor_gt_1 c d

/-- Theorem stating that all integers greater than 31 can be expressed as the sum of four positive integers
    where each pair has a common divisor greater than 1 -/
theorem sum_of_four_with_common_divisors (n : ℕ) (h : n > 31) : 
  is_sum_of_four_with_common_divisors n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_with_common_divisors_l3496_349657


namespace NUMINAMATH_CALUDE_symmetric_line_values_l3496_349686

/-- Two lines are symmetric with respect to the origin if for any point (x, y) on one line,
    the point (-x, -y) lies on the other line. -/
def symmetric_lines (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a * x + 3 * y - 9 = 0 ↔ x - 3 * y + b = 0

/-- If the line ax + 3y - 9 = 0 is symmetric to the line x - 3y + b = 0
    with respect to the origin, then a = -1 and b = -9. -/
theorem symmetric_line_values (a b : ℝ) (h : symmetric_lines a b) : a = -1 ∧ b = -9 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_values_l3496_349686


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l3496_349632

-- Define the arithmetic square root function
noncomputable def arithmeticSqrt (x : ℝ) : ℝ := 
  Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_of_nine : arithmeticSqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l3496_349632


namespace NUMINAMATH_CALUDE_not_sixth_power_l3496_349699

theorem not_sixth_power (n : ℕ) : ¬ ∃ (k : ℤ), 6 * (n : ℤ)^3 + 3 = k^6 := by
  sorry

end NUMINAMATH_CALUDE_not_sixth_power_l3496_349699


namespace NUMINAMATH_CALUDE_vector_combination_l3496_349653

/-- Given three points A, B, and C in a plane, prove that the coordinates of 1/2 * AC - 1/4 * BC are (-3, 6) -/
theorem vector_combination (A B C : ℝ × ℝ) (h1 : A = (2, -4)) (h2 : B = (0, 6)) (h3 : C = (-8, 10)) :
  (1 / 2 : ℝ) • (C - A) - (1 / 4 : ℝ) • (C - B) = (-3, 6) := by
  sorry

end NUMINAMATH_CALUDE_vector_combination_l3496_349653


namespace NUMINAMATH_CALUDE_smallest_k_for_720k_square_and_cube_l3496_349695

theorem smallest_k_for_720k_square_and_cube :
  (∀ n : ℕ+, n < 1012500 → ¬(∃ a b : ℕ+, 720 * n = a^2 ∧ 720 * n = b^3)) ∧
  (∃ a b : ℕ+, 720 * 1012500 = a^2 ∧ 720 * 1012500 = b^3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_720k_square_and_cube_l3496_349695


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3496_349628

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i * i = -1 → Complex.im (5 * i / (1 - 2 * i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3496_349628


namespace NUMINAMATH_CALUDE_slope_range_from_angle_of_inclination_l3496_349612

theorem slope_range_from_angle_of_inclination :
  ∀ a k : ℝ, 
    (π / 4 ≤ a ∧ a ≤ π / 2) →
    k = Real.tan a →
    (1 ≤ k ∧ ∀ y : ℝ, y ≥ 1 → ∃ x : ℝ, π / 4 ≤ x ∧ x ≤ π / 2 ∧ y = Real.tan x) :=
by sorry

end NUMINAMATH_CALUDE_slope_range_from_angle_of_inclination_l3496_349612


namespace NUMINAMATH_CALUDE_floor_length_percentage_l3496_349641

/-- Proves that for a rectangular floor with length 20 meters, if the total cost to paint the floor
    at 3 currency units per square meter is 400 currency units, then the length is 200% more than
    the breadth. -/
theorem floor_length_percentage (breadth : ℝ) (percentage : ℝ) : 
  breadth > 0 →
  percentage > 0 →
  20 = breadth * (1 + percentage / 100) →
  400 = 3 * (20 * breadth) →
  percentage = 200 := by
sorry

end NUMINAMATH_CALUDE_floor_length_percentage_l3496_349641


namespace NUMINAMATH_CALUDE_cyclists_speed_problem_l3496_349618

/-- Two cyclists problem -/
theorem cyclists_speed_problem (north_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  north_speed = 10 →
  time = 1.4285714285714286 →
  distance = 50 →
  ∃ (south_speed : ℝ), south_speed = 25 ∧ (north_speed + south_speed) * time = distance :=
by sorry

end NUMINAMATH_CALUDE_cyclists_speed_problem_l3496_349618


namespace NUMINAMATH_CALUDE_circle_transformation_l3496_349602

theorem circle_transformation (x₀ y₀ x y : ℝ) :
  x₀^2 + y₀^2 = 9 → x = x₀ → y = 4*y₀ → x^2/9 + y^2/144 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_transformation_l3496_349602


namespace NUMINAMATH_CALUDE_circle_radius_is_one_l3496_349600

/-- The radius of a circle defined by the equation x^2 + y^2 - 2y = 0 is 1 -/
theorem circle_radius_is_one :
  let circle_eq := (fun x y : ℝ => x^2 + y^2 - 2*y = 0)
  ∃ (h k r : ℝ), r = 1 ∧ ∀ x y : ℝ, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_one_l3496_349600


namespace NUMINAMATH_CALUDE_wheat_cost_per_acre_l3496_349604

theorem wheat_cost_per_acre 
  (total_land : ℕ)
  (wheat_land : ℕ)
  (corn_cost_per_acre : ℕ)
  (total_capital : ℕ)
  (h1 : total_land = 4500)
  (h2 : wheat_land = 3400)
  (h3 : corn_cost_per_acre = 42)
  (h4 : total_capital = 165200) :
  ∃ (wheat_cost_per_acre : ℕ),
    wheat_cost_per_acre * wheat_land + 
    corn_cost_per_acre * (total_land - wheat_land) = 
    total_capital ∧ 
    wheat_cost_per_acre = 35 := by
  sorry

end NUMINAMATH_CALUDE_wheat_cost_per_acre_l3496_349604


namespace NUMINAMATH_CALUDE_min_value_problem_l3496_349629

theorem min_value_problem (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 2) :
  (1/3 : ℝ) * x^3 + y^2 + z ≥ 13/12 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l3496_349629


namespace NUMINAMATH_CALUDE_interest_difference_l3496_349638

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Proves that the difference between the principal and the simple interest is 1260 -/
theorem interest_difference :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let time : ℝ := 4
  principal - simple_interest principal rate time = 1260 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l3496_349638


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l3496_349692

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites and their y-coordinates are equal -/
def symmetric_wrt_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_wrt_y_axis (a, -2) (-1, b) →
  (a + b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l3496_349692


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l3496_349614

theorem smallest_angle_measure (ABC ABD : ℝ) (h1 : ABC = 24) (h2 : ABD = 20) :
  ∃ CBD : ℝ, CBD = ABC - ABD ∧ CBD = 4 ∧ ∀ x : ℝ, x ≥ 0 → x ≥ CBD := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l3496_349614


namespace NUMINAMATH_CALUDE_negative_i_fourth_power_l3496_349601

theorem negative_i_fourth_power (i : ℂ) (h : i^2 = -1) : (-i)^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_i_fourth_power_l3496_349601


namespace NUMINAMATH_CALUDE_sam_seashells_l3496_349634

theorem sam_seashells (initial_seashells : ℕ) (given_away : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 35)
  (h2 : given_away = 18)
  (h3 : remaining_seashells = initial_seashells - given_away) :
  remaining_seashells = 17 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l3496_349634


namespace NUMINAMATH_CALUDE_consecutive_numbers_square_sum_l3496_349697

theorem consecutive_numbers_square_sum (a b c : ℕ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (a + b + c = 27) → a^2 + b^2 + c^2 = 245 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_square_sum_l3496_349697


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l3496_349617

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 3 = 0 ∧ x = 1) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l3496_349617


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l3496_349688

theorem consecutive_integers_product (a b c d e : ℤ) : 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (e = d + 1) →
  (a * b * c * d * e = 15120) →
  (e = 9) :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l3496_349688


namespace NUMINAMATH_CALUDE_jellybean_problem_minimum_jellybean_count_l3496_349676

theorem jellybean_problem (n : ℕ) : 
  (n ≥ 150) ∧ (n % 17 = 15) → n ≥ 151 :=
by sorry

theorem minimum_jellybean_count : 
  ∃ (n : ℕ), n ≥ 150 ∧ n % 17 = 15 ∧ ∀ (m : ℕ), m ≥ 150 ∧ m % 17 = 15 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_jellybean_problem_minimum_jellybean_count_l3496_349676


namespace NUMINAMATH_CALUDE_jerichos_money_l3496_349608

theorem jerichos_money (x : ℕ) : 
  x - (14 + 7) = 9 → 2 * x = 60 := by
  sorry

end NUMINAMATH_CALUDE_jerichos_money_l3496_349608


namespace NUMINAMATH_CALUDE_blue_area_after_transformations_l3496_349624

/-- Represents the fraction of blue area remaining after a single transformation -/
def blue_fraction_after_one_transform : ℚ := 3/4

/-- Represents the number of transformations -/
def num_transformations : ℕ := 3

/-- Represents the fraction of the original area that remains blue after all transformations -/
def final_blue_fraction : ℚ := (blue_fraction_after_one_transform) ^ num_transformations

theorem blue_area_after_transformations :
  final_blue_fraction = 27/64 := by sorry

end NUMINAMATH_CALUDE_blue_area_after_transformations_l3496_349624


namespace NUMINAMATH_CALUDE_crate_stacking_probability_l3496_349649

def crate_height : Fin 3 → ℕ
| 0 => 2
| 1 => 3
| 2 => 5

def total_combinations : ℕ := 3^10

def valid_combinations : ℕ := 2940

theorem crate_stacking_probability :
  (valid_combinations : ℚ) / total_combinations = 980 / 19683 :=
sorry

end NUMINAMATH_CALUDE_crate_stacking_probability_l3496_349649


namespace NUMINAMATH_CALUDE_distance_circle_center_to_point_l3496_349626

/-- The distance between the center of a circle with polar equation ρ = 4sin θ 
    and a point with polar coordinates (2√2, π/4) is 2 -/
theorem distance_circle_center_to_point : 
  let circle_equation : ℝ → ℝ := λ θ => 4 * Real.sin θ
  let point_A : ℝ × ℝ := (2 * Real.sqrt 2, Real.pi / 4)
  ∃ center : ℝ × ℝ, 
    Real.sqrt ((center.1 - (point_A.1 * Real.cos point_A.2))^2 + 
               (center.2 - (point_A.1 * Real.sin point_A.2))^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_point_l3496_349626


namespace NUMINAMATH_CALUDE_convex_polygon_symmetry_l3496_349615

-- Define a convex polygon
structure ConvexPolygon where
  -- (Add necessary fields for a convex polygon)

-- Define a point inside the polygon
structure InnerPoint (P : ConvexPolygon) where
  point : ℝ × ℝ
  isInside : Bool -- Predicate to check if the point is inside the polygon

-- Define a line passing through a point
structure Line (P : ℝ × ℝ) where
  slope : ℝ
  -- The line is represented by y = slope * (x - P.1) + P.2

-- Function to check if a line divides the polygon into equal areas
def dividesEqualAreas (P : ConvexPolygon) (O : InnerPoint P) (l : Line O.point) : Prop :=
  -- (Add logic to check if the line divides the polygon into equal areas)
  sorry

-- Function to check if a point is the center of symmetry
def isCenterOfSymmetry (P : ConvexPolygon) (O : InnerPoint P) : Prop :=
  -- (Add logic to check if O is the center of symmetry)
  sorry

-- The main theorem
theorem convex_polygon_symmetry (P : ConvexPolygon) (O : InnerPoint P) :
  (∀ l : Line O.point, dividesEqualAreas P O l) → isCenterOfSymmetry P O :=
by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_symmetry_l3496_349615


namespace NUMINAMATH_CALUDE_james_and_louise_ages_l3496_349660

theorem james_and_louise_ages :
  ∀ (james louise : ℕ),
  james = louise + 6 →
  james + 8 = 4 * (louise - 4) →
  james + louise = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_james_and_louise_ages_l3496_349660


namespace NUMINAMATH_CALUDE_last_three_digits_sum_l3496_349673

theorem last_three_digits_sum (n : ℕ) : 9^15 + 15^15 ≡ 24 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_sum_l3496_349673


namespace NUMINAMATH_CALUDE_subtraction_of_integers_l3496_349635

theorem subtraction_of_integers : 2 - 3 = -1 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_integers_l3496_349635


namespace NUMINAMATH_CALUDE_nina_shirt_price_l3496_349625

/-- Given Nina's shopping scenario, prove the price of each shirt. -/
theorem nina_shirt_price :
  -- Define the number and price of toys
  let num_toys : ℕ := 3
  let price_per_toy : ℕ := 10
  -- Define the number and price of card packs
  let num_card_packs : ℕ := 2
  let price_per_card_pack : ℕ := 5
  -- Define the number of shirts (equal to toys + card packs)
  let num_shirts : ℕ := num_toys + num_card_packs
  -- Define the total amount spent
  let total_spent : ℕ := 70
  -- Calculate the cost of toys and card packs
  let cost_toys_and_cards : ℕ := num_toys * price_per_toy + num_card_packs * price_per_card_pack
  -- Calculate the remaining amount spent on shirts
  let amount_spent_on_shirts : ℕ := total_spent - cost_toys_and_cards
  -- Calculate the price per shirt
  let price_per_shirt : ℕ := amount_spent_on_shirts / num_shirts
  -- Prove that the price per shirt is 6
  price_per_shirt = 6 := by sorry

end NUMINAMATH_CALUDE_nina_shirt_price_l3496_349625


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l3496_349678

theorem least_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (m + 17) ∧ k ∣ (7*m - 9))) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ (n + 17) ∧ k ∣ (7*n - 9)) ∧
  n = 1 :=
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l3496_349678


namespace NUMINAMATH_CALUDE_mushroom_count_l3496_349609

theorem mushroom_count :
  ∀ n m : ℕ,
  n ≤ 70 →
  m = (52 * n) / 100 →
  ∃ x : ℕ,
  x ≤ 3 ∧
  2 * (m - x) = n - 3 →
  n = 25 :=
by sorry

end NUMINAMATH_CALUDE_mushroom_count_l3496_349609


namespace NUMINAMATH_CALUDE_parabola_point_relation_l3496_349622

-- Define the parabola function
def parabola (x c : ℝ) : ℝ := -x^2 + 6*x + c

-- Define the theorem
theorem parabola_point_relation (c y₁ y₂ y₃ : ℝ) :
  parabola 1 c = y₁ →
  parabola 3 c = y₂ →
  parabola 4 c = y₃ →
  y₂ > y₃ ∧ y₃ > y₁ := by
  sorry


end NUMINAMATH_CALUDE_parabola_point_relation_l3496_349622


namespace NUMINAMATH_CALUDE_expansion_distinct_terms_l3496_349667

/-- The number of distinct terms in the expansion of a product of two sums -/
def distinctTermsInExpansion (n m : ℕ) : ℕ := n * m

/-- Theorem: The number of distinct terms in the expansion of (a+b+c)(d+e+f+g+h+i) is 18 -/
theorem expansion_distinct_terms :
  distinctTermsInExpansion 3 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_expansion_distinct_terms_l3496_349667


namespace NUMINAMATH_CALUDE_number_divided_by_three_l3496_349671

theorem number_divided_by_three : ∃ x : ℝ, (x / 3 = x - 48) ∧ (x = 72) := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_l3496_349671


namespace NUMINAMATH_CALUDE_power_of_point_theorem_l3496_349658

-- Define the circle and points
variable (Circle : Type) (Point : Type)
variable (A B M D C : Point)
variable (diameter : Point → Point → Circle → Prop)
variable (on_circle : Point → Circle → Prop)
variable (intersect : Point → Point → Point → Prop)

-- Define the distance function
variable (dist : Point → Point → ℝ)

-- State the theorem
theorem power_of_point_theorem 
  (circle : Circle)
  (h1 : diameter A B circle)
  (h2 : intersect A M B)
  (h3 : on_circle D circle ∧ intersect A M D)
  (h4 : on_circle C circle ∧ intersect B M C) :
  (dist A B) ^ 2 = (dist A M) * (dist A D) + (dist B M) * (dist B C) ∨
  (dist A B) ^ 2 = (dist A M) * (dist A D) - (dist B M) * (dist B C) :=
sorry

end NUMINAMATH_CALUDE_power_of_point_theorem_l3496_349658


namespace NUMINAMATH_CALUDE_sqrt_difference_product_l3496_349633

theorem sqrt_difference_product : (Real.sqrt 6 + Real.sqrt 11) * (Real.sqrt 6 - Real.sqrt 11) = -5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_product_l3496_349633


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3496_349647

theorem quadratic_factorization (x : ℝ) :
  ∃ (m n : ℤ), 6 * x^2 - 5 * x - 6 = (6 * x + m) * (x + n) ∧ m - n = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3496_349647


namespace NUMINAMATH_CALUDE_horner_v2_value_l3496_349689

def horner_polynomial (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def horner_step (v : ℝ) (x : ℝ) (a : ℝ) : ℝ :=
  v * x + a

theorem horner_v2_value :
  let f := fun x => 8 * x^4 + 5 * x^3 + 3 * x^2 + 2 * x + 1
  let coeffs := [8, 5, 3, 2, 1]
  let x := 2
  let v₀ := coeffs.head!
  let v₁ := horner_step v₀ x (coeffs.get! 1)
  let v₂ := horner_step v₁ x (coeffs.get! 2)
  v₂ = 45 := by sorry

end NUMINAMATH_CALUDE_horner_v2_value_l3496_349689


namespace NUMINAMATH_CALUDE_tony_age_in_six_years_l3496_349610

/-- Given Jacob's current age and Tony's age relative to Jacob's, 
    calculate Tony's age after a certain number of years. -/
def tony_future_age (jacob_age : ℕ) (years_passed : ℕ) : ℕ :=
  (jacob_age / 2) + years_passed

/-- Theorem: Tony will be 18 years old in 6 years -/
theorem tony_age_in_six_years :
  tony_future_age 24 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_tony_age_in_six_years_l3496_349610


namespace NUMINAMATH_CALUDE_max_songs_in_three_hours_l3496_349682

/-- Represents the maximum number of songs that can be played in a given time -/
def max_songs_played (short_songs : ℕ) (long_songs : ℕ) (short_duration : ℕ) (long_duration : ℕ) (total_time : ℕ) : ℕ :=
  let short_used := min short_songs (total_time / short_duration)
  let remaining_time := total_time - short_used * short_duration
  let long_used := min long_songs (remaining_time / long_duration)
  short_used + long_used

/-- Theorem stating the maximum number of songs that can be played in 3 hours -/
theorem max_songs_in_three_hours :
  max_songs_played 50 50 3 5 180 = 56 := by
  sorry

end NUMINAMATH_CALUDE_max_songs_in_three_hours_l3496_349682


namespace NUMINAMATH_CALUDE_max_draw_without_pair_is_four_l3496_349672

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (white : Nat)
  (blue : Nat)
  (red : Nat)

/-- Represents the maximum number of socks that can be drawn without guaranteeing a pair -/
def maxDrawWithoutPair (drawer : SockDrawer) : Nat :=
  4

/-- Theorem stating that for the given sock drawer, the maximum number of socks
    that can be drawn without guaranteeing a pair is 4 -/
theorem max_draw_without_pair_is_four (drawer : SockDrawer) 
  (h1 : drawer.white = 16) 
  (h2 : drawer.blue = 3) 
  (h3 : drawer.red = 6) : 
  maxDrawWithoutPair drawer = 4 := by
  sorry

#eval maxDrawWithoutPair { white := 16, blue := 3, red := 6 }

end NUMINAMATH_CALUDE_max_draw_without_pair_is_four_l3496_349672


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3496_349664

theorem cube_volume_from_surface_area :
  ∀ (side : ℝ), 
    side > 0 →
    6 * side^2 = 486 →
    side^3 = 729 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3496_349664
