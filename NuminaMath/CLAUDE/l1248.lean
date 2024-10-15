import Mathlib

namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_point_five_l1248_124899

theorem reciprocal_of_negative_one_point_five :
  ((-1.5)⁻¹ : ℝ) = -2/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_point_five_l1248_124899


namespace NUMINAMATH_CALUDE_circles_tangent_internally_l1248_124894

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 16 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 64

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (4, -3)
def radius1 : ℝ := 3
def center2 : ℝ × ℝ := (0, 0)
def radius2 : ℝ := 8

-- Theorem stating that the circles are tangent internally
theorem circles_tangent_internally :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = radius2 - radius1 := by sorry

end NUMINAMATH_CALUDE_circles_tangent_internally_l1248_124894


namespace NUMINAMATH_CALUDE_arc_length_example_l1248_124821

/-- The length of an arc in a circle, given the radius and central angle -/
def arcLength (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  radius * centralAngle

theorem arc_length_example :
  let radius : ℝ := 10
  let centralAngle : ℝ := 2 * Real.pi / 3
  arcLength radius centralAngle = 20 * Real.pi / 3 := by
sorry


end NUMINAMATH_CALUDE_arc_length_example_l1248_124821


namespace NUMINAMATH_CALUDE_negation_equivalence_l1248_124847

variable (a : ℝ)

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*a*x + a > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1248_124847


namespace NUMINAMATH_CALUDE_unique_perimeter_l1248_124888

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  AB : ℕ+
  BC : ℕ+
  CD : ℕ+
  AD : ℕ+
  right_angle_B : True
  right_angle_C : True
  AB_equals_3 : AB = 3
  CD_equals_AD : CD = AD

/-- The perimeter of a SpecialQuadrilateral -/
def perimeter (q : SpecialQuadrilateral) : ℕ :=
  q.AB + q.BC + q.CD + q.AD

/-- Theorem stating that there's exactly one valid perimeter less than 2015 -/
theorem unique_perimeter :
  ∃! p : ℕ, p < 2015 ∧ ∃ q : SpecialQuadrilateral, perimeter q = p :=
by sorry

end NUMINAMATH_CALUDE_unique_perimeter_l1248_124888


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1248_124892

/-- The perimeter of a rectangle with length 100 and breadth 500 is 1200. -/
theorem rectangle_perimeter : 
  ∀ (length breadth perimeter : ℕ), 
    length = 100 → 
    breadth = 500 → 
    perimeter = 2 * (length + breadth) → 
    perimeter = 1200 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1248_124892


namespace NUMINAMATH_CALUDE_line_intercept_form_l1248_124810

/-- A line passing through the point (2,3) with slope 2 has the equation x/(1/2) + y/(-1) = 1 in intercept form. -/
theorem line_intercept_form (l : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ y - 3 = 2 * (x - 2)) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x / (1/2) + y / (-1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_intercept_form_l1248_124810


namespace NUMINAMATH_CALUDE_catenary_properties_l1248_124895

noncomputable def f (a b x : ℝ) : ℝ := a * Real.exp x + b * Real.exp (-x)

theorem catenary_properties :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
  (∀ x, f 1 b x = f 1 b (-x) → b = 1) ∧
  (∃ a b, ∀ x y, x < y → f a b x < f a b y) ∧
  ((∃ a b, ∀ x, f a b x ≥ 2 ∧ (∃ x₀, f a b x₀ = 2)) →
   (∀ a b, (∀ x, f a b x ≥ 2 ∧ (∃ x₀, f a b x₀ = 2)) → a + b ≥ 2) ∧
   (∃ a b, (∀ x, f a b x ≥ 2 ∧ (∃ x₀, f a b x₀ = 2)) ∧ a + b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_catenary_properties_l1248_124895


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l1248_124806

theorem reciprocal_of_negative_2023 : ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l1248_124806


namespace NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l1248_124844

theorem smallest_absolute_value_of_z (z : ℂ) (h : Complex.abs (z - 10) + Complex.abs (z + 3*I) = 17) :
  ∃ (w : ℂ), Complex.abs (z - 10) + Complex.abs (z + 3*I) = 17 ∧ Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 30/17 :=
by sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l1248_124844


namespace NUMINAMATH_CALUDE_intersection_A_B_equals_open_interval_2_3_l1248_124893

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 5}

-- Define the open interval (2, 3)
def open_interval_2_3 : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem intersection_A_B_equals_open_interval_2_3 : A ∩ B = open_interval_2_3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_equals_open_interval_2_3_l1248_124893


namespace NUMINAMATH_CALUDE_max_cables_theorem_l1248_124801

/-- Represents the maximum number of cables that can be used to connect computers
    in an organization with specific constraints. -/
def max_cables (total_employees : ℕ) (brand_a_computers : ℕ) (brand_b_computers : ℕ) : ℕ :=
  30

/-- Theorem stating that the maximum number of cables is 30 under given conditions. -/
theorem max_cables_theorem (total_employees : ℕ) (brand_a_computers : ℕ) (brand_b_computers : ℕ) :
  total_employees = 40 →
  brand_a_computers = 25 →
  brand_b_computers = 15 →
  total_employees = brand_a_computers + brand_b_computers →
  max_cables total_employees brand_a_computers brand_b_computers = 30 :=
by
  sorry

#check max_cables_theorem

end NUMINAMATH_CALUDE_max_cables_theorem_l1248_124801


namespace NUMINAMATH_CALUDE_apple_sale_discrepancy_l1248_124833

/-- Represents the number of apples sold for one cent by the first vendor -/
def apples_per_cent_vendor1 : ℕ := 3

/-- Represents the number of apples sold for one cent by the second vendor -/
def apples_per_cent_vendor2 : ℕ := 2

/-- Represents the number of unsold apples each vendor had -/
def unsold_apples_per_vendor : ℕ := 30

/-- Represents the total number of apples to be sold -/
def total_apples : ℕ := 2 * unsold_apples_per_vendor

/-- Represents the number of apples sold for two cents by the friend -/
def apples_per_two_cents_friend : ℕ := 5

/-- Calculates the revenue when apples are sold individually by vendors -/
def revenue_individual : ℕ := 
  (unsold_apples_per_vendor / apples_per_cent_vendor1) + 
  (unsold_apples_per_vendor / apples_per_cent_vendor2)

/-- Calculates the revenue when apples are sold by the friend -/
def revenue_friend : ℕ := 
  2 * (total_apples / apples_per_two_cents_friend)

theorem apple_sale_discrepancy : 
  revenue_individual = revenue_friend + 1 := by
  sorry

end NUMINAMATH_CALUDE_apple_sale_discrepancy_l1248_124833


namespace NUMINAMATH_CALUDE_rectangle_to_square_l1248_124881

theorem rectangle_to_square (original_length : ℝ) (original_width : ℝ) (square_side : ℝ) : 
  original_width = 24 →
  square_side = 12 →
  original_length * original_width = square_side * square_side →
  original_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l1248_124881


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l1248_124884

theorem consecutive_integers_sum_of_cubes (n : ℕ) : 
  n > 0 ∧ (n - 1)^2 + n^2 + (n + 1)^2 = 8555 → 
  (n - 1)^3 + n^3 + (n + 1)^3 = 446949 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l1248_124884


namespace NUMINAMATH_CALUDE_star_two_four_star_neg_three_x_l1248_124861

-- Define the new operation ※
def star (a b : ℝ) : ℝ := a^2 + 2*a*b

-- Theorem 1
theorem star_two_four : star 2 4 = 20 := by sorry

-- Theorem 2
theorem star_neg_three_x (x : ℝ) : star (-3) x = -3 + x → x = 12/7 := by sorry

end NUMINAMATH_CALUDE_star_two_four_star_neg_three_x_l1248_124861


namespace NUMINAMATH_CALUDE_positive_real_inequality_l1248_124896

theorem positive_real_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + 2*y^2 + 3*z^2 > x*y + 3*y*z + z*x := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l1248_124896


namespace NUMINAMATH_CALUDE_complex_equation_real_solutions_l1248_124823

theorem complex_equation_real_solutions :
  ∃! (s : Finset ℝ), (∀ a ∈ s, ∃ z : ℂ, Complex.abs z = 1 ∧ z^2 + a*z + a^2 - 1 = 0) ∧
                     (∀ a : ℝ, (∃ z : ℂ, Complex.abs z = 1 ∧ z^2 + a*z + a^2 - 1 = 0) → a ∈ s) ∧
                     Finset.card s = 5 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_real_solutions_l1248_124823


namespace NUMINAMATH_CALUDE_specific_dumbbell_system_weight_l1248_124882

/-- The weight of a dumbbell system with three pairs of dumbbells -/
def dumbbellSystemWeight (weight1 weight2 weight3 : ℕ) : ℕ :=
  2 * weight1 + 2 * weight2 + 2 * weight3

/-- Theorem: The total weight of the specific dumbbell system is 32 lb -/
theorem specific_dumbbell_system_weight :
  dumbbellSystemWeight 3 5 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_specific_dumbbell_system_weight_l1248_124882


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_l1248_124830

theorem rectangular_prism_surface_area (r h : ℝ) : 
  r = (36 / Real.pi) ^ (1/3) → 
  (4/3) * Real.pi * r^3 = 6 * 4 * h → 
  2 * (4 * 6 + 2 * 4 + 2 * 6) = 88 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_l1248_124830


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1248_124836

theorem gcd_of_specific_numbers : 
  let m : ℕ := 3333333
  let n : ℕ := 66666666
  gcd m n = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1248_124836


namespace NUMINAMATH_CALUDE_problem_statement_l1248_124825

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -24)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 8) :
  b / (a + b) + c / (b + c) + a / (c + a) = 19 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1248_124825


namespace NUMINAMATH_CALUDE_faulty_meter_profit_percent_l1248_124850

/-- The profit percentage for a shopkeeper using a faulty meter -/
theorem faulty_meter_profit_percent (actual_weight : ℝ) (expected_weight : ℝ) :
  actual_weight = 960 →
  expected_weight = 1000 →
  (1 - actual_weight / expected_weight) * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_faulty_meter_profit_percent_l1248_124850


namespace NUMINAMATH_CALUDE_stating_count_paths_correct_l1248_124804

/-- 
Counts the number of paths from (0,0) to (m,n) where m < n and 
at every intermediate point (a,b), a < b.
-/
def count_paths (m n : ℕ) : ℕ :=
  if m < n then
    (Nat.factorial (m + n - 1) * (n - m)) / (Nat.factorial m * Nat.factorial n)
  else 0

/-- 
Theorem stating that count_paths gives the correct number of paths
from (0,0) to (m,n) satisfying the given conditions.
-/
theorem count_paths_correct (m n : ℕ) (h : m < n) :
  count_paths m n = ((Nat.factorial (m + n - 1) * (n - m)) / (Nat.factorial m * Nat.factorial n)) :=
by sorry

end NUMINAMATH_CALUDE_stating_count_paths_correct_l1248_124804


namespace NUMINAMATH_CALUDE_buratino_spent_10_dollars_l1248_124897

/-- Represents a transaction at the exchange point -/
inductive Transaction
  | type1 : Transaction  -- Give 2 euros, receive 3 dollars and a candy
  | type2 : Transaction  -- Give 5 dollars, receive 3 euros and a candy

/-- Represents Buratino's exchange operations -/
structure ExchangeOperations where
  transactions : List Transaction
  initial_dollars : ℕ
  final_dollars : ℕ
  final_euros : ℕ
  candies : ℕ

/-- The condition that Buratino's exchange operations are valid -/
def valid_exchange (ops : ExchangeOperations) : Prop :=
  ops.final_euros = 0 ∧
  ops.candies = ops.transactions.length ∧
  ops.candies = 50 ∧
  ops.final_dollars < ops.initial_dollars

/-- Calculate the net dollar change from a list of transactions -/
def net_dollar_change (transactions : List Transaction) : ℤ :=
  transactions.foldl (fun acc t => match t with
    | Transaction.type1 => acc + 3
    | Transaction.type2 => acc - 5
  ) 0

/-- The main theorem stating that Buratino spent 10 dollars -/
theorem buratino_spent_10_dollars (ops : ExchangeOperations) 
  (h : valid_exchange ops) : 
  ops.initial_dollars - ops.final_dollars = 10 := by
  sorry


end NUMINAMATH_CALUDE_buratino_spent_10_dollars_l1248_124897


namespace NUMINAMATH_CALUDE_correct_num_raised_beds_l1248_124866

/-- The number of raised beds Abby is building -/
def num_raised_beds : ℕ := 2

/-- The length of each raised bed in feet -/
def bed_length : ℕ := 8

/-- The width of each raised bed in feet -/
def bed_width : ℕ := 4

/-- The height of each raised bed in feet -/
def bed_height : ℕ := 1

/-- The volume of soil in each bag in cubic feet -/
def soil_per_bag : ℕ := 4

/-- The total number of soil bags needed -/
def total_soil_bags : ℕ := 16

/-- Theorem stating that the number of raised beds Abby is building is correct -/
theorem correct_num_raised_beds :
  num_raised_beds * (bed_length * bed_width * bed_height) = total_soil_bags * soil_per_bag :=
by sorry

end NUMINAMATH_CALUDE_correct_num_raised_beds_l1248_124866


namespace NUMINAMATH_CALUDE_tims_prank_combinations_l1248_124815

/-- Represents the number of choices for each day of the prank --/
structure PrankChoices where
  day1 : Nat
  day2 : Nat
  day3 : Nat
  day4 : Nat → Nat
  day5 : Nat

/-- Calculates the total number of combinations for the prank --/
def totalCombinations (choices : PrankChoices) : Nat :=
  choices.day1 * choices.day2 * choices.day3 * 
  (choices.day3 * choices.day4 1 + choices.day3 * choices.day4 2 + choices.day3 * choices.day4 3) *
  choices.day5

/-- The specific choices for Tim's prank --/
def timsPrankChoices : PrankChoices where
  day1 := 1
  day2 := 2
  day3 := 3
  day4 := fun n => match n with
    | 1 => 2
    | 2 => 3
    | _ => 1
  day5 := 1

theorem tims_prank_combinations :
  totalCombinations timsPrankChoices = 36 := by
  sorry

end NUMINAMATH_CALUDE_tims_prank_combinations_l1248_124815


namespace NUMINAMATH_CALUDE_parabola_one_x_intercept_parabola_x_intercepts_parabola_opens_upward_l1248_124846

-- Define the parabola
def parabola (a c x : ℝ) : ℝ := a * x^2 + 2 * a * x + c

-- Theorem 1: If a = c, the parabola has only one point in common with the x-axis
theorem parabola_one_x_intercept (a : ℝ) (h : a ≠ 0) :
  ∃! x, parabola a a x = 0 :=
sorry

-- Theorem 2: If the x-intercepts satisfy 1/x₁ + 1/x₂ = 1, then c = -2a
theorem parabola_x_intercepts (a c x₁ x₂ : ℝ) (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0)
  (h₃ : parabola a c x₁ = 0) (h₄ : parabola a c x₂ = 0) (h₅ : 1/x₁ + 1/x₂ = 1) :
  c = -2 * a :=
sorry

-- Theorem 3: If (m,p) lies on y = -ax + c - 2a, -2 < m < -1, and p > n where (m,n) is on the parabola, then a > 0
theorem parabola_opens_upward (a c m : ℝ) (h₁ : -2 < m) (h₂ : m < -1)
  (h₃ : -a * m + c - 2 * a > parabola a c m) :
  a > 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_one_x_intercept_parabola_x_intercepts_parabola_opens_upward_l1248_124846


namespace NUMINAMATH_CALUDE_work_time_B_l1248_124834

theorem work_time_B (time_A time_BC time_AC : ℝ) (h1 : time_A = 4) (h2 : time_BC = 3) (h3 : time_AC = 2) : 
  (1 / time_A + 1 / time_BC - 1 / time_AC)⁻¹ = 12 := by
sorry

end NUMINAMATH_CALUDE_work_time_B_l1248_124834


namespace NUMINAMATH_CALUDE_bart_tuesday_surveys_l1248_124800

/-- Represents the number of surveys Bart completed on Tuesday -/
def tuesday_surveys : ℕ := sorry

/-- The amount earned per question in dollars -/
def earnings_per_question : ℚ := 1/5

/-- The number of questions in each survey -/
def questions_per_survey : ℕ := 10

/-- The number of surveys completed on Monday -/
def monday_surveys : ℕ := 3

/-- The total amount earned over two days in dollars -/
def total_earnings : ℚ := 14

theorem bart_tuesday_surveys :
  tuesday_surveys = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_bart_tuesday_surveys_l1248_124800


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1248_124855

theorem hyperbola_asymptote_slope (x y m : ℝ) : 
  (x^2 / 144 - y^2 / 81 = 1) →  -- hyperbola equation
  (∃ (k : ℝ), y = k * m * x ∧ y = -k * m * x) →  -- asymptotes
  (m > 0) →  -- m is positive
  (m = 3/4) :=  -- conclusion
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1248_124855


namespace NUMINAMATH_CALUDE_square_diagonals_equal_l1248_124835

-- Define a structure for a parallelogram
structure Parallelogram :=
  (diagonals_equal : Bool)

-- Define a structure for a square that is a parallelogram
structure Square extends Parallelogram

-- State the theorem
theorem square_diagonals_equal (s : Square) : s.diagonals_equal = true := by
  sorry


end NUMINAMATH_CALUDE_square_diagonals_equal_l1248_124835


namespace NUMINAMATH_CALUDE_dalton_savings_l1248_124820

def jump_rope_cost : ℕ := 7
def board_game_cost : ℕ := 12
def playground_ball_cost : ℕ := 4
def uncle_money : ℕ := 13
def additional_money_needed : ℕ := 4

def total_cost : ℕ := jump_rope_cost + board_game_cost + playground_ball_cost

theorem dalton_savings (savings : ℕ) : 
  savings = total_cost - (uncle_money + additional_money_needed) := by
  sorry

end NUMINAMATH_CALUDE_dalton_savings_l1248_124820


namespace NUMINAMATH_CALUDE_distance_midpoint_problem_l1248_124845

theorem distance_midpoint_problem (t : ℝ) : 
  let A : ℝ × ℝ := (2*t - 3, 0)
  let B : ℝ × ℝ := (1, 2*t + 2)
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = 2*t^2 + 3*t
  → t = 10/7 := by
sorry

end NUMINAMATH_CALUDE_distance_midpoint_problem_l1248_124845


namespace NUMINAMATH_CALUDE_problem_statement_l1248_124852

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 4)
  (h2 : x * y = 64) :
  (x + y) / 2 = (64^(1/(3+Real.sqrt 3)) + 64^((2+Real.sqrt 3)/(3+Real.sqrt 3))) / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1248_124852


namespace NUMINAMATH_CALUDE_sine_sum_inequality_l1248_124870

theorem sine_sum_inequality (x y z : ℝ) (h : x + y + z = 0) :
  Real.sin x + Real.sin y + Real.sin z ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_sum_inequality_l1248_124870


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l1248_124885

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 5)^2 + (y - 6)^2 + (z - 7)^2 + 2 = 2 →
  x + y + z = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l1248_124885


namespace NUMINAMATH_CALUDE_f_min_max_l1248_124864

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem f_min_max :
  let I : Set ℝ := Set.Icc 0 (2 * Real.pi)
  ∃ (min_val max_val : ℝ),
    (∀ x ∈ I, f x ≥ min_val) ∧
    (∀ x ∈ I, f x ≤ max_val) ∧
    (∃ x₁ ∈ I, f x₁ = min_val) ∧
    (∃ x₂ ∈ I, f x₂ = max_val) ∧
    min_val = -3 * Real.pi / 2 ∧
    max_val = Real.pi / 2 + 2 :=
by sorry

end NUMINAMATH_CALUDE_f_min_max_l1248_124864


namespace NUMINAMATH_CALUDE_exponential_function_inequality_l1248_124828

theorem exponential_function_inequality (m n : ℝ) : 
  let a : ℝ := (Real.sqrt 5 - Real.sqrt 2) / 2
  let f : ℝ → ℝ := fun x ↦ a^x
  0 < a ∧ a < 1 → f m > f n → m < n := by sorry

end NUMINAMATH_CALUDE_exponential_function_inequality_l1248_124828


namespace NUMINAMATH_CALUDE_part_one_part_two_l1248_124854

-- Define the equation
def equation (x m : ℝ) : Prop :=
  x / (x - 3) + m / (3 - x) = 3

-- Part 1
theorem part_one (m : ℝ) :
  equation 2 m → m = 5 := by
  sorry

-- Part 2
theorem part_two (x m : ℝ) :
  equation x m → x > 0 → m < 9 ∧ m ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1248_124854


namespace NUMINAMATH_CALUDE_shortest_midpoint_to_midpoint_path_length_l1248_124802

-- Define a regular cube
structure RegularCube where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

-- Define a path on the surface of the cube
def SurfacePath (cube : RegularCube) := ℝ

-- Define the property of being a valid path from midpoint to midpoint of opposite edges
def IsValidMidpointToMidpointPath (cube : RegularCube) (path : SurfacePath cube) : Prop :=
  sorry

-- Define the length of a path
def PathLength (cube : RegularCube) (path : SurfacePath cube) : ℝ :=
  sorry

-- Theorem statement
theorem shortest_midpoint_to_midpoint_path_length 
  (cube : RegularCube) 
  (h : cube.edgeLength = 2) :
  ∃ (path : SurfacePath cube), 
    IsValidMidpointToMidpointPath cube path ∧ 
    PathLength cube path = 4 ∧
    ∀ (other_path : SurfacePath cube), 
      IsValidMidpointToMidpointPath cube other_path → 
      PathLength cube other_path ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_shortest_midpoint_to_midpoint_path_length_l1248_124802


namespace NUMINAMATH_CALUDE_set_equality_implies_m_zero_l1248_124869

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {3, m}
def B (m : ℝ) : Set ℝ := {3*m, 3}

-- State the theorem
theorem set_equality_implies_m_zero :
  ∀ m : ℝ, A m = B m → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_m_zero_l1248_124869


namespace NUMINAMATH_CALUDE_rectangular_solid_spheres_l1248_124822

/-- A rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0

/-- A sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Predicate for a sphere being circumscribed around a rectangular solid -/
def isCircumscribed (s : Sphere) (r : RectangularSolid) : Prop :=
  sorry

/-- Predicate for a sphere being inscribed in a rectangular solid -/
def isInscribed (s : Sphere) (r : RectangularSolid) : Prop :=
  sorry

/-- Theorem: A rectangular solid with a circumscribed sphere does not necessarily have an inscribed sphere -/
theorem rectangular_solid_spheres (r : RectangularSolid) (s : Sphere) :
  isCircumscribed s r → ¬∀ (s' : Sphere), isInscribed s' r :=
sorry

end NUMINAMATH_CALUDE_rectangular_solid_spheres_l1248_124822


namespace NUMINAMATH_CALUDE_problem_solution_l1248_124827

theorem problem_solution (x y : ℝ) (h1 : x = 3) (h2 : y = 3) : 
  x - y^((x - y) / 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1248_124827


namespace NUMINAMATH_CALUDE_range_of_m_l1248_124824

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, m * x₀^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- Theorem statement
theorem range_of_m (m : ℝ) (h : p m ∧ q m) : -2 < m ∧ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1248_124824


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_4_6_8_10_l1248_124814

def is_divisible_by_all (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (n % 6 = 0) ∧ (n % 8 = 0) ∧ (n % 10 = 0)

theorem smallest_number_divisible_by_4_6_8_10 :
  ∀ n : ℕ, n ≥ 136 → (is_divisible_by_all (n - 16) → n ≥ 136) ∧
  is_divisible_by_all (136 - 16) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_4_6_8_10_l1248_124814


namespace NUMINAMATH_CALUDE_unique_determination_by_gcds_l1248_124868

theorem unique_determination_by_gcds :
  ∀ X : ℕ, X ≤ 100 →
  ∃ (M N : Fin 7 → ℕ), (∀ i, M i < 100 ∧ N i < 100) ∧
    ∀ Y : ℕ, Y ≤ 100 →
      (∀ i : Fin 7, Nat.gcd (X + M i) (N i) = Nat.gcd (Y + M i) (N i)) →
      X = Y :=
by sorry

end NUMINAMATH_CALUDE_unique_determination_by_gcds_l1248_124868


namespace NUMINAMATH_CALUDE_combination_equation_solution_l1248_124838

theorem combination_equation_solution (x : ℕ) : 
  (Nat.choose 34 (2*x) = Nat.choose 34 (4*x - 8)) → (x = 4 ∨ x = 7) := by
  sorry

end NUMINAMATH_CALUDE_combination_equation_solution_l1248_124838


namespace NUMINAMATH_CALUDE_power_equality_l1248_124875

theorem power_equality (n : ℕ) : 3^n = 3^2 * 9^4 * 81^3 → n = 22 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1248_124875


namespace NUMINAMATH_CALUDE_senior_citizen_tickets_l1248_124890

theorem senior_citizen_tickets (total_tickets : ℕ) (adult_price senior_price : ℕ) (total_receipts : ℕ) 
  (h1 : total_tickets = 529)
  (h2 : adult_price = 25)
  (h3 : senior_price = 15)
  (h4 : total_receipts = 9745) :
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 348 := by
  sorry

end NUMINAMATH_CALUDE_senior_citizen_tickets_l1248_124890


namespace NUMINAMATH_CALUDE_max_a_value_exists_max_a_l1248_124872

theorem max_a_value (a b : ℕ) (h : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120) : a ≤ 20 := by
  sorry

theorem exists_max_a : ∃ a b : ℕ, 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120 ∧ a = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_a_value_exists_max_a_l1248_124872


namespace NUMINAMATH_CALUDE_unique_triple_lcm_l1248_124811

theorem unique_triple_lcm : 
  ∃! (a b c : ℕ+), 
    Nat.lcm a b = 1200 ∧ 
    Nat.lcm b c = 1800 ∧ 
    Nat.lcm c a = 2400 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_lcm_l1248_124811


namespace NUMINAMATH_CALUDE_binomial_7_2_l1248_124856

theorem binomial_7_2 : Nat.choose 7 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_2_l1248_124856


namespace NUMINAMATH_CALUDE_mixed_sample_more_suitable_l1248_124898

-- Define the probability of having the disease
def disease_probability : ℝ := 0.1

-- Define the number of animals in each group
def group_size : ℕ := 2

-- Define the total number of animals
def total_animals : ℕ := 2 * group_size

-- Define the expected number of tests for individual testing
def expected_tests_individual : ℝ := total_animals

-- Define the probability of a negative mixed sample
def prob_negative_mixed : ℝ := (1 - disease_probability) ^ total_animals

-- Define the expected number of tests for mixed sample testing
def expected_tests_mixed : ℝ :=
  1 * prob_negative_mixed + (1 + total_animals) * (1 - prob_negative_mixed)

-- Theorem statement
theorem mixed_sample_more_suitable :
  expected_tests_mixed < expected_tests_individual :=
sorry

end NUMINAMATH_CALUDE_mixed_sample_more_suitable_l1248_124898


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l1248_124877

def arithmetic_progression (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_progression_sum
  (a : ℕ → ℚ)
  (h_ap : arithmetic_progression a)
  (h_sum1 : a 1 + a 4 + a 7 = 45)
  (h_sum2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 27 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l1248_124877


namespace NUMINAMATH_CALUDE_radical_axis_through_intersection_points_l1248_124851

-- Define a circle with center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the power of a point with respect to a circle
def powerOfPoint (p : ℝ × ℝ) (c : Circle) : ℝ :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 - c.radius^2

-- Define the radical axis of two circles
def radicalAxis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | powerOfPoint p c1 = powerOfPoint p c2}

-- Define the intersection points of two circles
def intersectionPoints (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | powerOfPoint p c1 = 0 ∧ powerOfPoint p c2 = 0}

-- Theorem statement
theorem radical_axis_through_intersection_points (c1 c2 : Circle) :
  intersectionPoints c1 c2 ⊆ radicalAxis c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_radical_axis_through_intersection_points_l1248_124851


namespace NUMINAMATH_CALUDE_min_squares_128_343_l1248_124809

/-- Represents a rectangle with height and width -/
structure Rectangle where
  height : ℕ
  width : ℕ

/-- Represents a polyomino spanning a rectangle -/
def SpanningPolyomino (r : Rectangle) : Type := Unit

/-- The number of unit squares in a spanning polyomino -/
def num_squares (r : Rectangle) (p : SpanningPolyomino r) : ℕ := sorry

/-- The minimum number of unit squares in any spanning polyomino for a given rectangle -/
def min_spanning_squares (r : Rectangle) : ℕ := sorry

/-- Theorem: The minimum number of unit squares in a spanning polyomino for a 128-by-343 rectangle is 470 -/
theorem min_squares_128_343 :
  let r : Rectangle := { height := 128, width := 343 }
  min_spanning_squares r = 470 := by sorry

end NUMINAMATH_CALUDE_min_squares_128_343_l1248_124809


namespace NUMINAMATH_CALUDE_perpendicular_to_horizontal_is_vertical_l1248_124867

/-- The angle of inclination of a line -/
def angle_of_inclination (l : Line2D) : ℝ := sorry

/-- A line is horizontal if its angle of inclination is 0 -/
def is_horizontal (l : Line2D) : Prop := angle_of_inclination l = 0

/-- Two lines are perpendicular if their angles of inclination sum to 90° -/
def are_perpendicular (l1 l2 : Line2D) : Prop :=
  angle_of_inclination l1 + angle_of_inclination l2 = 90

theorem perpendicular_to_horizontal_is_vertical (l1 l2 : Line2D) :
  is_horizontal l1 → are_perpendicular l1 l2 → angle_of_inclination l2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_to_horizontal_is_vertical_l1248_124867


namespace NUMINAMATH_CALUDE_inequality_conditions_l1248_124819

theorem inequality_conditions (a b c : ℝ) 
  (h : ∀ (x y z : ℝ), a * (x - y) * (x - z) + b * (y - x) * (y - z) + c * (z - x) * (z - y) ≥ 0) : 
  (-a + 2*b + 2*c ≥ 0) ∧ (2*a - b + 2*c ≥ 0) ∧ (2*a + 2*b - c ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_conditions_l1248_124819


namespace NUMINAMATH_CALUDE_f_of_g_eight_l1248_124849

def g (x : ℝ) : ℝ := 4 * x + 5

def f (x : ℝ) : ℝ := 6 * x - 11

theorem f_of_g_eight : f (g 8) = 211 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_eight_l1248_124849


namespace NUMINAMATH_CALUDE_money_theorem_l1248_124837

/-- Given the conditions on c and d, prove that c > 12.4 and d < 24 -/
theorem money_theorem (c d : ℝ) 
  (h1 : 7 * c - d > 80)
  (h2 : 4 * c + d = 44)
  (h3 : d < 2 * c) :
  c > 12.4 ∧ d < 24 := by
  sorry

end NUMINAMATH_CALUDE_money_theorem_l1248_124837


namespace NUMINAMATH_CALUDE_min_discount_factor_l1248_124848

def cost_price : ℝ := 800
def marked_price : ℝ := 1200
def min_profit_margin : ℝ := 0.2

theorem min_discount_factor (x : ℝ) : 
  (cost_price * (1 + min_profit_margin) = marked_price * x) → x = 0.8 :=
by sorry

end NUMINAMATH_CALUDE_min_discount_factor_l1248_124848


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1248_124865

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1248_124865


namespace NUMINAMATH_CALUDE_hair_growth_proof_l1248_124826

/-- Calculates the additional hair growth needed for donation -/
def additional_growth_needed (current_length donation_length desired_length : ℝ) : ℝ :=
  (donation_length + desired_length) - current_length

/-- Proves that the additional hair growth needed is 21 inches -/
theorem hair_growth_proof (current_length donation_length desired_length : ℝ) 
  (h1 : current_length = 14)
  (h2 : donation_length = 23)
  (h3 : desired_length = 12) :
  additional_growth_needed current_length donation_length desired_length = 21 :=
by sorry

end NUMINAMATH_CALUDE_hair_growth_proof_l1248_124826


namespace NUMINAMATH_CALUDE_equidistant_complex_function_d_squared_l1248_124840

/-- A complex function g(z) = (c+di)z with the property that g(z) is equidistant from z and the origin -/
def equidistant_complex_function (c d : ℝ) : ℂ → ℂ := λ z ↦ (c + d * Complex.I) * z

/-- The property that g(z) is equidistant from z and the origin for all z -/
def is_equidistant (g : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs (g z - z) = Complex.abs (g z)

theorem equidistant_complex_function_d_squared 
  (c d : ℝ) 
  (h1 : is_equidistant (equidistant_complex_function c d))
  (h2 : Complex.abs (c + d * Complex.I) = 7) : 
  d^2 = 195/4 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_complex_function_d_squared_l1248_124840


namespace NUMINAMATH_CALUDE_minutes_to_skate_on_ninth_day_l1248_124816

/-- The number of minutes Gage skated each day for the first 6 days -/
def minutes_per_day_first_6 : ℕ := 60

/-- The number of days Gage skated for 60 minutes -/
def days_skating_60_min : ℕ := 6

/-- The number of minutes Gage skated each day for the next 2 days -/
def minutes_per_day_next_2 : ℕ := 120

/-- The number of days Gage skated for 120 minutes -/
def days_skating_120_min : ℕ := 2

/-- The target average number of minutes per day for all 9 days -/
def target_average_minutes : ℕ := 100

/-- The total number of days Gage skated -/
def total_days : ℕ := 9

/-- Theorem stating the number of minutes Gage needs to skate on the 9th day -/
theorem minutes_to_skate_on_ninth_day :
  target_average_minutes * total_days -
  (minutes_per_day_first_6 * days_skating_60_min +
   minutes_per_day_next_2 * days_skating_120_min) = 300 := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_skate_on_ninth_day_l1248_124816


namespace NUMINAMATH_CALUDE_brick_surface_area_l1248_124886

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 3 cm rectangular prism is 164 square centimeters -/
theorem brick_surface_area :
  surface_area 10 4 3 = 164 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l1248_124886


namespace NUMINAMATH_CALUDE_infinite_non_prime_polynomials_l1248_124862

theorem infinite_non_prime_polynomials :
  ∃ f : ℕ → ℕ, ∀ k n : ℕ, ¬ Prime (n^4 + f k * n) := by
  sorry

end NUMINAMATH_CALUDE_infinite_non_prime_polynomials_l1248_124862


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1248_124841

theorem algebraic_expression_equality (x y : ℝ) : 
  x - 2*y + 2 = 5 → 4*y - 2*x + 1 = -5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1248_124841


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1248_124878

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  ((a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4) ∨ (a = 3 ∧ c = 5) ∨ (b = 3 ∧ c = 5)) →
  c = Real.sqrt 41 ∨ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1248_124878


namespace NUMINAMATH_CALUDE_negation_equivalence_l1248_124812

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Doctor : U → Prop)
variable (GoodAtMath : U → Prop)

-- Define the statements
def AllDoctorsGoodAtMath : Prop := ∀ x, Doctor x → GoodAtMath x
def AtLeastOneDoctorBadAtMath : Prop := ∃ x, Doctor x ∧ ¬GoodAtMath x

-- Theorem to prove
theorem negation_equivalence :
  AtLeastOneDoctorBadAtMath U Doctor GoodAtMath ↔ ¬(AllDoctorsGoodAtMath U Doctor GoodAtMath) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1248_124812


namespace NUMINAMATH_CALUDE_amy_current_age_l1248_124891

/-- Given that Mark is 7 years older than Amy and Mark will be 27 years old in 5 years,
    prove that Amy's current age is 15 years old. -/
theorem amy_current_age :
  ∀ (mark_age amy_age : ℕ),
  mark_age = amy_age + 7 →
  mark_age + 5 = 27 →
  amy_age = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_current_age_l1248_124891


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_range_l1248_124832

/-- The range of k for which the line y = kx - 1 intersects the right branch of
    the hyperbola x^2 - y^2 = 1 at two different points -/
theorem line_hyperbola_intersection_range :
  ∀ k : ℝ, (∃ x₁ x₂ y₁ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    x₁ > 0 ∧ x₂ > 0 ∧
    x₁^2 - y₁^2 = 1 ∧
    x₂^2 - y₂^2 = 1 ∧
    y₁ = k * x₁ - 1 ∧
    y₂ = k * x₂ - 1) ↔
  (1 < k ∧ k < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_range_l1248_124832


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l1248_124860

theorem quadratic_roots_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁^2 + p*x₁ + q = 0) ∧ 
    (x₂^2 + p*x₂ + q = 0) ∧ 
    |x₁ - x₂| = 2) →
  p = Real.sqrt (4*q + 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l1248_124860


namespace NUMINAMATH_CALUDE_intersection_condition_l1248_124859

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line2D) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- The condition for two lines to intersect -/
def intersect (l₁ l₂ : Line2D) : Prop :=
  ¬ parallel l₁ l₂

/-- Definition of the two lines in the problem -/
def l₁ (a : ℝ) : Line2D := ⟨1, -a, 3⟩
def l₂ (a : ℝ) : Line2D := ⟨a, -4, 5⟩

/-- The main theorem to prove -/
theorem intersection_condition :
  (∀ a : ℝ, intersect (l₁ a) (l₂ a) → a ≠ 2) ∧
  ¬(∀ a : ℝ, a ≠ 2 → intersect (l₁ a) (l₂ a)) := by
  sorry


end NUMINAMATH_CALUDE_intersection_condition_l1248_124859


namespace NUMINAMATH_CALUDE_value_of_a_l1248_124879

def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem value_of_a (a : ℝ) : A a ∪ B a = {0, 1, 2, 4, 16} → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1248_124879


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_6sqrt5_l1248_124874

theorem sqrt_sum_equals_6sqrt5 : 
  Real.sqrt ((2 - 3 * Real.sqrt 5) ^ 2) + Real.sqrt ((2 + 3 * Real.sqrt 5) ^ 2) = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_6sqrt5_l1248_124874


namespace NUMINAMATH_CALUDE_measure_S_eq_one_l1248_124871

open MeasureTheory

/-- The set of times where car A has completed twice as many laps as car B -/
def S (α : ℝ) : Set ℝ :=
  {t : ℝ | t ≥ α ∧ ⌊t⌋ = 2 * ⌊t - α⌋}

/-- The theorem stating that the measure of S is 1 -/
theorem measure_S_eq_one (α : ℝ) (hα : α > 0) :
  volume (S α) = 1 := by sorry

end NUMINAMATH_CALUDE_measure_S_eq_one_l1248_124871


namespace NUMINAMATH_CALUDE_college_students_count_l1248_124813

/-- Calculates the total number of students in a college given the ratio of boys to girls and the number of girls. -/
def totalStudents (boyRatio girlRatio numGirls : ℕ) : ℕ :=
  let numBoys := boyRatio * numGirls / girlRatio
  numBoys + numGirls

/-- Theorem: In a college where the ratio of boys to girls is 8:5 and there are 190 girls, the total number of students is 494. -/
theorem college_students_count :
  totalStudents 8 5 190 = 494 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l1248_124813


namespace NUMINAMATH_CALUDE_mario_earnings_l1248_124857

/-- Mario's work hours and earnings over two weeks in July --/
theorem mario_earnings :
  ∀ (third_week_hours second_week_hours : ℕ) 
    (hourly_rate third_week_earnings second_week_earnings : ℚ),
  third_week_hours = 28 →
  third_week_hours = second_week_hours + 10 →
  third_week_earnings = second_week_earnings + 68 →
  hourly_rate * (third_week_hours : ℚ) = third_week_earnings →
  hourly_rate * (second_week_hours : ℚ) = second_week_earnings →
  hourly_rate * ((third_week_hours + second_week_hours) : ℚ) = 312.8 :=
by sorry

end NUMINAMATH_CALUDE_mario_earnings_l1248_124857


namespace NUMINAMATH_CALUDE_decimal_point_problem_l1248_124842

theorem decimal_point_problem :
  ∃! (x : ℝ), x > 0 ∧ 100 * x = 9 * (1 / x) := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l1248_124842


namespace NUMINAMATH_CALUDE_problem_1_l1248_124883

theorem problem_1 : (-1)^3 + (1/7) * (2 - (-3)^2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1248_124883


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1248_124818

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_shift_theorem (x y : ℝ) :
  let original := Parabola.mk 2 0 0
  let shifted := shift_parabola original 4 3
  y = 2 * x^2 → y = shifted.a * (x - 4)^2 + shifted.b * (x - 4) + shifted.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1248_124818


namespace NUMINAMATH_CALUDE_bowling_team_weight_l1248_124853

theorem bowling_team_weight (original_players : ℕ) (original_avg_weight : ℝ)
  (new_players : ℕ) (second_player_weight : ℝ) (new_avg_weight : ℝ)
  (h1 : original_players = 7)
  (h2 : original_avg_weight = 112)
  (h3 : new_players = 2)
  (h4 : second_player_weight = 60)
  (h5 : new_avg_weight = 106) :
  let total_players := original_players + new_players
  let original_total_weight := original_players * original_avg_weight
  let new_total_weight := total_players * new_avg_weight
  let first_player_weight := new_total_weight - original_total_weight - second_player_weight
  first_player_weight = 110 := by
sorry

end NUMINAMATH_CALUDE_bowling_team_weight_l1248_124853


namespace NUMINAMATH_CALUDE_clara_lego_count_l1248_124807

/-- The number of legos each person has --/
structure LegoCount where
  kent : ℕ
  bruce : ℕ
  simon : ℕ
  clara : ℕ

/-- Conditions of the lego distribution --/
def lego_distribution (l : LegoCount) : Prop :=
  l.kent = 80 ∧
  l.bruce = l.kent + 30 ∧
  l.simon = l.bruce + (l.bruce / 4) ∧
  l.clara = l.simon + l.kent - ((l.simon + l.kent) / 10)

/-- Theorem stating Clara's lego count --/
theorem clara_lego_count (l : LegoCount) (h : lego_distribution l) : l.clara = 197 := by
  sorry


end NUMINAMATH_CALUDE_clara_lego_count_l1248_124807


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l1248_124880

/-- Given that N(5,8) is the midpoint of line segment CD and C(7,4) is one endpoint,
    the product of coordinates of point D is 36. -/
theorem midpoint_coordinate_product (C D N : ℝ × ℝ) : 
  C = (7, 4) → N = (5, 8) → N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 * D.2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l1248_124880


namespace NUMINAMATH_CALUDE_rhombus_diagonals_l1248_124808

/-- Given a rhombus where a height from the obtuse angle vertex divides a side into
    segments of length a and b, this theorem proves the lengths of its diagonals. -/
theorem rhombus_diagonals (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (d1 d2 : ℝ),
    d1 = Real.sqrt (2 * b * (a + b)) ∧
    d2 = Real.sqrt (2 * (2 * a + b) * (a + b)) ∧
    d1 > 0 ∧ d2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_l1248_124808


namespace NUMINAMATH_CALUDE_combination_equality_l1248_124839

theorem combination_equality (x : ℕ) : 
  (Nat.choose 20 (2*x - 1) = Nat.choose 20 (x + 3)) ↔ (x = 4 ∨ x = 6) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l1248_124839


namespace NUMINAMATH_CALUDE_contractor_absent_days_l1248_124843

/-- Proves the number of absent days for a contractor under specific conditions -/
theorem contractor_absent_days 
  (total_days : ℕ) 
  (daily_pay : ℚ) 
  (daily_fine : ℚ) 
  (total_received : ℚ) 
  (h1 : total_days = 30)
  (h2 : daily_pay = 25)
  (h3 : daily_fine = 7.5)
  (h4 : total_received = 360) :
  ∃ (absent_days : ℕ), 
    (absent_days : ℚ) * daily_fine + (total_days - absent_days : ℚ) * daily_pay = total_received ∧ 
    absent_days = 12 := by
  sorry


end NUMINAMATH_CALUDE_contractor_absent_days_l1248_124843


namespace NUMINAMATH_CALUDE_power_product_equality_l1248_124805

theorem power_product_equality : (-0.125)^2022 * 8^2023 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1248_124805


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1248_124863

theorem complex_equation_sum (x y : ℝ) :
  (x - 3*y : ℂ) + (2*x + 3*y)*I = 5 + I →
  x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1248_124863


namespace NUMINAMATH_CALUDE_binomial_probability_l1248_124858

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial random variable -/
def expectedValue (X : BinomialRV) : ℝ := X.n * X.p

/-- The probability mass function of a binomial random variable -/
def pmf (X : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose X.n k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

/-- Theorem: For a binomial random variable X with p = 1/3 and E(X) = 2, P(X=2) = 80/243 -/
theorem binomial_probability (X : BinomialRV) 
  (h_p : X.p = 1/3) 
  (h_exp : expectedValue X = 2) : 
  pmf X 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_l1248_124858


namespace NUMINAMATH_CALUDE_cubic_function_property_l1248_124889

/-- Given a cubic function f(x) = ax^3 + bx + 1 where f(-2) = 2, prove that f(2) = 0 -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x + 1
  f (-2) = 2 → f 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1248_124889


namespace NUMINAMATH_CALUDE_negation_equivalence_l1248_124803

theorem negation_equivalence : 
  (¬(x = 3 → x^2 - 2*x - 3 = 0)) ↔ (x ≠ 3 → x^2 - 2*x - 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1248_124803


namespace NUMINAMATH_CALUDE_cylinder_cone_height_relation_l1248_124876

/-- Given a right cylinder and a cone with equal base radii, volumes, surface areas, 
    and heights in the ratio of 1:3, prove that the height of the cylinder 
    is 4/5 of the base radius. -/
theorem cylinder_cone_height_relation 
  (r : ℝ) -- base radius
  (h_cyl : ℝ) -- height of cylinder
  (h_cone : ℝ) -- height of cone
  (h_ratio : h_cone = 3 * h_cyl) -- height ratio condition
  (h_vol : π * r^2 * h_cyl = 1/3 * π * r^2 * h_cone) -- equal volumes
  (h_area : 2 * π * r^2 + 2 * π * r * h_cyl = 
            π * r^2 + π * r * Real.sqrt (r^2 + h_cone^2)) -- equal surface areas
  : h_cyl = 4/5 * r :=
by sorry

end NUMINAMATH_CALUDE_cylinder_cone_height_relation_l1248_124876


namespace NUMINAMATH_CALUDE_complex_power_2016_pi_half_l1248_124829

theorem complex_power_2016_pi_half :
  let z : ℂ := Complex.exp (Complex.I * (π / 2 : ℝ))
  (z ^ 2016 : ℂ) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_2016_pi_half_l1248_124829


namespace NUMINAMATH_CALUDE_function_characterization_l1248_124887

theorem function_characterization :
  ∀ f : ℕ → ℤ,
  (∀ k l : ℕ, (f k - f l) ∣ (k^2 - l^2)) →
  ∃ (c : ℤ) (g : ℕ → Fin 2),
    (∀ x : ℕ, f x = (-1)^(g x).val * x + c) ∨
    (∀ x : ℕ, f x = x^2 + c) ∨
    (∀ x : ℕ, f x = -x^2 + c) :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l1248_124887


namespace NUMINAMATH_CALUDE_xyz_product_l1248_124873

/-- Given real numbers x, y, z, a, b, and c satisfying certain conditions,
    prove that their product xyz equals (a³ - 3ab² + 2c³) / 6 -/
theorem xyz_product (x y z a b c : ℝ) 
  (sum_eq : x + y + z = a)
  (sum_squares_eq : x^2 + y^2 + z^2 = b^2)
  (sum_cubes_eq : x^3 + y^3 + z^3 = c^3) :
  x * y * z = (a^3 - 3*a*b^2 + 2*c^3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l1248_124873


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l1248_124817

theorem sum_of_squares_theorem (a b m : ℝ) 
  (h1 : a^2 + a*b = 16 + m) 
  (h2 : b^2 + a*b = 9 - m) : 
  (a + b = 5) ∨ (a + b = -5) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l1248_124817


namespace NUMINAMATH_CALUDE_odd_power_congruence_l1248_124831

theorem odd_power_congruence (x : ℤ) (n : ℕ) (h_odd : Odd x) (h_n : n ≥ 1) :
  ∃ k : ℤ, x^(2^n) = 1 + k * 2^(n+2) := by
  sorry

end NUMINAMATH_CALUDE_odd_power_congruence_l1248_124831
