import Mathlib

namespace NUMINAMATH_CALUDE_points_per_round_l2484_248436

/-- Given a card game where:
  * Jane ends up with 60 points
  * She lost 20 points
  * She played 8 rounds
  Prove that the number of points awarded for winning one round is 10. -/
theorem points_per_round (final_points : ℕ) (lost_points : ℕ) (rounds : ℕ) :
  final_points = 60 →
  lost_points = 20 →
  rounds = 8 →
  (final_points + lost_points) / rounds = 10 :=
by sorry

end NUMINAMATH_CALUDE_points_per_round_l2484_248436


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_subset_implies_m_range_l2484_248491

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 2)*(x - m - 2) ≤ 0}

-- Theorem 1
theorem intersection_implies_m_value :
  ∀ m : ℝ, A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3} → m = 2 :=
sorry

-- Theorem 2
theorem subset_implies_m_range :
  ∀ m : ℝ, A ⊆ (Set.univ : Set ℝ) \ B m → m < -3 ∨ m > 5 :=
sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_subset_implies_m_range_l2484_248491


namespace NUMINAMATH_CALUDE_disrespectful_quadratic_max_root_sum_l2484_248453

/-- A quadratic polynomial with real coefficients and leading coefficient 1 -/
def QuadraticPolynomial (b c : ℝ) := fun (x : ℝ) ↦ x^2 + b*x + c

/-- The condition for a polynomial to be "disrespectful" -/
def isDisrespectful (p : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x : ℝ, p (p x) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)

/-- The sum of roots of a quadratic polynomial -/
def sumOfRoots (b c : ℝ) : ℝ := -b

theorem disrespectful_quadratic_max_root_sum (b c : ℝ) :
  let p := QuadraticPolynomial b c
  isDisrespectful p ∧ 
  (∀ b' c' : ℝ, isDisrespectful (QuadraticPolynomial b' c') → sumOfRoots b' c' ≤ sumOfRoots b c) →
  p 1 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_disrespectful_quadratic_max_root_sum_l2484_248453


namespace NUMINAMATH_CALUDE_test_total_points_l2484_248439

-- Define the test structure
structure Test where
  total_questions : ℕ
  two_point_questions : ℕ
  four_point_questions : ℕ

-- Define the function to calculate total points
def calculateTotalPoints (test : Test) : ℕ :=
  2 * test.two_point_questions + 4 * test.four_point_questions

-- Theorem statement
theorem test_total_points (test : Test) 
  (h1 : test.total_questions = 40)
  (h2 : test.two_point_questions = 30)
  (h3 : test.four_point_questions = 10)
  (h4 : test.total_questions = test.two_point_questions + test.four_point_questions) :
  calculateTotalPoints test = 100 := by
  sorry

-- Example usage
def exampleTest : Test := {
  total_questions := 40,
  two_point_questions := 30,
  four_point_questions := 10
}

#eval calculateTotalPoints exampleTest

end NUMINAMATH_CALUDE_test_total_points_l2484_248439


namespace NUMINAMATH_CALUDE_snyder_cookies_l2484_248488

/-- Given that Mrs. Snyder made a total of 86 cookies, with only red and pink colors,
    and 50 of them are pink, prove that she made 36 red cookies. -/
theorem snyder_cookies (total : ℕ) (pink : ℕ) (red : ℕ) : 
  total = 86 → pink = 50 → total = pink + red → red = 36 := by
  sorry

end NUMINAMATH_CALUDE_snyder_cookies_l2484_248488


namespace NUMINAMATH_CALUDE_overall_gain_percentage_l2484_248460

/-- Calculate the overall gain percentage for three items --/
theorem overall_gain_percentage 
  (cost_A cost_B cost_C : ℚ)
  (gain_A gain_B gain_C : ℚ)
  (h1 : cost_A = 700)
  (h2 : cost_B = 500)
  (h3 : cost_C = 300)
  (h4 : gain_A = 70)
  (h5 : gain_B = 50)
  (h6 : gain_C = 30) :
  (gain_A + gain_B + gain_C) / (cost_A + cost_B + cost_C) = 1 / 10 := by
sorry

#eval (70 + 50 + 30) / (700 + 500 + 300) -- This should evaluate to 0.1

end NUMINAMATH_CALUDE_overall_gain_percentage_l2484_248460


namespace NUMINAMATH_CALUDE_tan_double_angle_l2484_248482

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l2484_248482


namespace NUMINAMATH_CALUDE_base5_divisible_by_13_l2484_248444

/-- Converts a base 5 number to decimal --/
def base5ToDecimal (a b c d : ℕ) : ℕ :=
  a * 5^3 + b * 5^2 + c * 5 + d

/-- Checks if a number is divisible by 13 --/
def isDivisibleBy13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

theorem base5_divisible_by_13 :
  let y := 2
  let base5Num := base5ToDecimal 2 3 y 2
  isDivisibleBy13 base5Num :=
by sorry

end NUMINAMATH_CALUDE_base5_divisible_by_13_l2484_248444


namespace NUMINAMATH_CALUDE_vidyas_age_multiple_l2484_248490

theorem vidyas_age_multiple (vidya_age mother_age : ℕ) (h1 : vidya_age = 13) (h2 : mother_age = 44) :
  ∃ m : ℕ, m * vidya_age + 5 = mother_age ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_vidyas_age_multiple_l2484_248490


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l2484_248410

theorem solve_fraction_equation (y : ℚ) (h : (1:ℚ)/3 - (1:ℚ)/4 = 1/y) : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l2484_248410


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2484_248477

-- Problem 1
theorem problem_1 : -7 + (-3) - 4 - |(-8)| = -22 := by sorry

-- Problem 2
theorem problem_2 : (1/2 - 5/9 + 7/12) * (-36) = -19 := by sorry

-- Problem 3
theorem problem_3 : -3^2 + 16 / (-2) * (1/2) - (-1)^2023 = -14 := by sorry

-- Problem 4
theorem problem_4 (a b : ℝ) : 3*a^2 - 2*a*b - a^2 + 5*a*b = 2*a^2 + 3*a*b := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2484_248477


namespace NUMINAMATH_CALUDE_expression_equality_l2484_248408

theorem expression_equality : (2^1006 + 5^1007)^2 - (2^1006 - 5^1007)^2 = 40 * 10^1006 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2484_248408


namespace NUMINAMATH_CALUDE_students_in_sports_l2484_248475

theorem students_in_sports (total : ℕ) (basketball soccer baseball cricket : ℕ)
  (basketball_soccer basketball_baseball basketball_cricket : ℕ)
  (soccer_baseball cricket_soccer cricket_baseball : ℕ)
  (basketball_cricket_soccer : ℕ) (no_sport : ℕ)
  (h1 : total = 200)
  (h2 : basketball = 50)
  (h3 : soccer = 60)
  (h4 : baseball = 35)
  (h5 : cricket = 80)
  (h6 : basketball_soccer = 10)
  (h7 : basketball_baseball = 15)
  (h8 : basketball_cricket = 20)
  (h9 : soccer_baseball = 25)
  (h10 : cricket_soccer = 30)
  (h11 : cricket_baseball = 5)
  (h12 : basketball_cricket_soccer = 10)
  (h13 : no_sport = 30) :
  basketball + soccer + baseball + cricket -
  basketball_soccer - basketball_baseball - basketball_cricket -
  soccer_baseball - cricket_soccer - cricket_baseball +
  basketball_cricket_soccer = 130 := by
  sorry

end NUMINAMATH_CALUDE_students_in_sports_l2484_248475


namespace NUMINAMATH_CALUDE_shipment_composition_l2484_248498

/-- Represents a shipment of boxes with two possible weights -/
structure Shipment where
  total_boxes : ℕ
  weight1 : ℕ
  weight2 : ℕ
  count1 : ℕ
  count2 : ℕ
  initial_avg : ℚ

/-- Theorem about the composition of a specific shipment -/
theorem shipment_composition (s : Shipment) 
  (h1 : s.total_boxes = 30)
  (h2 : s.weight1 = 10)
  (h3 : s.weight2 = 20)
  (h4 : s.initial_avg = 18)
  (h5 : s.count1 + s.count2 = s.total_boxes)
  (h6 : s.weight1 * s.count1 + s.weight2 * s.count2 = s.initial_avg * s.total_boxes) :
  s.count1 = 6 ∧ s.count2 = 24 := by
  sorry

/-- Function to calculate the number of heavy boxes to remove to reach a target average -/
def boxes_to_remove (s : Shipment) (target_avg : ℚ) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_shipment_composition_l2484_248498


namespace NUMINAMATH_CALUDE_intersection_angle_l2484_248406

-- Define the lines
def line1 (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0
def line2 (x : ℝ) : Prop := x + 5 = 0

-- Define the angle between the lines
def angle_between_lines : ℝ := 30

-- Theorem statement
theorem intersection_angle :
  ∃ (x y : ℝ), line1 x y ∧ line2 x → angle_between_lines = 30 := by sorry

end NUMINAMATH_CALUDE_intersection_angle_l2484_248406


namespace NUMINAMATH_CALUDE_lamp_price_after_discounts_l2484_248418

/-- Calculates the final price of a lamp after applying two discounts -/
theorem lamp_price_after_discounts (original_price : ℝ) 
  (first_discount_rate : ℝ) (second_discount_rate : ℝ) : 
  original_price = 120 → 
  first_discount_rate = 0.20 → 
  second_discount_rate = 0.15 → 
  original_price * (1 - first_discount_rate) * (1 - second_discount_rate) = 81.60 := by
sorry

end NUMINAMATH_CALUDE_lamp_price_after_discounts_l2484_248418


namespace NUMINAMATH_CALUDE_equal_share_ratio_l2484_248465

def total_amount : ℕ := 5400
def num_children : ℕ := 3
def b_share : ℕ := 1800

theorem equal_share_ratio :
  ∃ (a_share c_share : ℕ),
    a_share + b_share + c_share = total_amount ∧
    a_share = c_share ∧
    a_share = b_share :=
by sorry

end NUMINAMATH_CALUDE_equal_share_ratio_l2484_248465


namespace NUMINAMATH_CALUDE_unique_solution_geometric_series_l2484_248495

theorem unique_solution_geometric_series :
  ∃! x : ℝ, x = x^3 * (1 / (1 + x)) ∧ |x| < 1 :=
by
  -- The unique solution is (√5 - 1) / 2
  sorry

end NUMINAMATH_CALUDE_unique_solution_geometric_series_l2484_248495


namespace NUMINAMATH_CALUDE_g_solution_set_a_range_l2484_248400

-- Define the functions f and g
def f (a x : ℝ) := 3 * abs (x - a) + abs (3 * x + 1)
def g (x : ℝ) := abs (4 * x - 1) - abs (x + 2)

-- Theorem for the solution set of g(x) < 6
theorem g_solution_set :
  {x : ℝ | g x < 6} = {x : ℝ | -7/5 < x ∧ x < 3} := by sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, f a x₁ = -g x₂) → a ∈ Set.Icc (-13/12) (5/12) := by sorry

end NUMINAMATH_CALUDE_g_solution_set_a_range_l2484_248400


namespace NUMINAMATH_CALUDE_specific_plot_fencing_cost_l2484_248493

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ

/-- Calculates the total cost of fencing for a given rectangular plot. -/
def total_fencing_cost (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth) * plot.fencing_cost_per_meter

/-- Theorem stating the total fencing cost for a specific rectangular plot. -/
theorem specific_plot_fencing_cost :
  let plot : RectangularPlot := {
    length := 56,
    breadth := 56 - 12,
    fencing_cost_per_meter := 26.5
  }
  total_fencing_cost plot = 5300 := by
  sorry


end NUMINAMATH_CALUDE_specific_plot_fencing_cost_l2484_248493


namespace NUMINAMATH_CALUDE_sports_club_membership_l2484_248437

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ)
  (h1 : total = 30)
  (h2 : badminton = 16)
  (h3 : tennis = 19)
  (h4 : both = 7) :
  total - (badminton + tennis - both) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_membership_l2484_248437


namespace NUMINAMATH_CALUDE_count_grid_paths_l2484_248405

/-- The number of paths from (0,0) to (m,n) on a grid, moving only right or up -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- Theorem: The number of distinct paths from the bottom-left corner to the top-right corner
    of an m × n grid, moving only upward or to the right, is equal to (m+n choose m) -/
theorem count_grid_paths (m n : ℕ) : 
  grid_paths m n = Nat.choose (m + n) m := by sorry

end NUMINAMATH_CALUDE_count_grid_paths_l2484_248405


namespace NUMINAMATH_CALUDE_books_returned_thursday_l2484_248424

/-- The number of books returned on Thursday given the initial conditions and final count. -/
theorem books_returned_thursday 
  (initial_wednesday : ℕ) 
  (checkout_wednesday : ℕ) 
  (checkout_thursday : ℕ) 
  (returned_friday : ℕ) 
  (final_friday : ℕ) 
  (h1 : initial_wednesday = 98) 
  (h2 : checkout_wednesday = 43) 
  (h3 : checkout_thursday = 5) 
  (h4 : returned_friday = 7) 
  (h5 : final_friday = 80) : 
  final_friday = initial_wednesday - checkout_wednesday - checkout_thursday + returned_friday + 23 := by
  sorry

#check books_returned_thursday

end NUMINAMATH_CALUDE_books_returned_thursday_l2484_248424


namespace NUMINAMATH_CALUDE_product_magnitude_l2484_248499

theorem product_magnitude (z₁ z₂ : ℂ) (h1 : Complex.abs z₁ = 3) (h2 : z₂ = Complex.mk 2 1) :
  Complex.abs (z₁ * z₂) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_product_magnitude_l2484_248499


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2484_248494

theorem root_sum_theorem (a b c : ℝ) : 
  a^3 - 30*a^2 + 65*a - 42 = 0 → 
  b^3 - 30*b^2 + 65*b - 42 = 0 → 
  c^3 - 30*c^2 + 65*c - 42 = 0 → 
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b) = 770/43 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2484_248494


namespace NUMINAMATH_CALUDE_combinatorial_identities_l2484_248454

-- Define combinatorial choice function
def C (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define permutation function
def A (n m : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - m))

theorem combinatorial_identities :
  (3 * C 8 3 - 2 * C 5 2 = 148) ∧
  (∀ n m : ℕ, n ≥ m → m ≥ 2 → A n m = n * A (n-1) (m-1)) :=
by sorry

end NUMINAMATH_CALUDE_combinatorial_identities_l2484_248454


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2484_248440

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set Nat := {1, 3, 5, 7, 9}
def B : Set Nat := {1, 2, 5, 6, 8}

theorem intersection_complement_equality : A ∩ (U \ B) = {3, 7, 9} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2484_248440


namespace NUMINAMATH_CALUDE_sumAreaVolume_specific_l2484_248478

/-- Represents a point in 3D space with integer coordinates -/
structure Point3D where
  x : Int
  y : Int
  z : Int

/-- Represents a parallelepiped in 3D space -/
structure Parallelepiped where
  base1 : Point3D
  base2 : Point3D
  base3 : Point3D
  base4 : Point3D
  height : Int

/-- Calculates the sum of surface area and volume of a parallelepiped -/
def sumAreaVolume (p : Parallelepiped) : Int :=
  sorry -- Actual calculation would go here

/-- The specific parallelepiped from the problem -/
def specificParallelepiped : Parallelepiped :=
  { base1 := { x := 0, y := 0, z := 0 },
    base2 := { x := 3, y := 4, z := 0 },
    base3 := { x := 7, y := 0, z := 0 },
    base4 := { x := 10, y := 4, z := 0 },
    height := 5 }

theorem sumAreaVolume_specific : sumAreaVolume specificParallelepiped = 365 := by
  sorry

end NUMINAMATH_CALUDE_sumAreaVolume_specific_l2484_248478


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2484_248419

/-- An isosceles triangle with two sides of 6cm and 13cm has a perimeter of 32cm. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 6 ∧ b = 13 ∧ c = 13 →  -- Two sides are 13cm (base) and one side is 6cm
  a + b > c ∧ a + c > b ∧ b + c > a →  -- Triangle inequality
  a + b + c = 32 :=  -- Perimeter is 32cm
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2484_248419


namespace NUMINAMATH_CALUDE_suit_price_calculation_l2484_248415

theorem suit_price_calculation (original_price : ℝ) (increase_rate : ℝ) (discount_rate : ℝ) : 
  original_price = 150 →
  increase_rate = 0.2 →
  discount_rate = 0.2 →
  let increased_price := original_price * (1 + increase_rate)
  let final_price := increased_price * (1 - discount_rate)
  final_price = 144 := by
sorry

end NUMINAMATH_CALUDE_suit_price_calculation_l2484_248415


namespace NUMINAMATH_CALUDE_ceiling_evaluation_l2484_248433

theorem ceiling_evaluation : ⌈(4 * (8 - 1/3 : ℚ))⌉ = 31 := by sorry

end NUMINAMATH_CALUDE_ceiling_evaluation_l2484_248433


namespace NUMINAMATH_CALUDE_smallest_sum_consecutive_primes_div_by_5_l2484_248449

/-- Three consecutive primes with sum divisible by 5 -/
def ConsecutivePrimesWithSumDivBy5 (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  q = Nat.succ p ∧ r = Nat.succ q ∧
  (p + q + r) % 5 = 0

/-- The smallest sum of three consecutive primes divisible by 5 -/
theorem smallest_sum_consecutive_primes_div_by_5 :
  ∃ (p q r : ℕ), ConsecutivePrimesWithSumDivBy5 p q r ∧
    ∀ (a b c : ℕ), ConsecutivePrimesWithSumDivBy5 a b c → p + q + r ≤ a + b + c ∧
    p + q + r = 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_consecutive_primes_div_by_5_l2484_248449


namespace NUMINAMATH_CALUDE_correct_scientific_notation_l2484_248455

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Check if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (n : ℝ) : Prop :=
  sn.coefficient * (10 : ℝ) ^ sn.exponent = n

/-- The number we want to represent in scientific notation -/
def target_number : ℝ := 2034000

/-- The proposed scientific notation representation -/
def proposed_representation : ScientificNotation := {
  coefficient := 2.034
  exponent := 6
  coeff_range := by sorry
}

/-- Theorem stating that the proposed representation is correct -/
theorem correct_scientific_notation :
  represents proposed_representation target_number :=
by sorry

end NUMINAMATH_CALUDE_correct_scientific_notation_l2484_248455


namespace NUMINAMATH_CALUDE_B_power_66_l2484_248496

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; -1, 0, 0; 0, 0, 1]

theorem B_power_66 : B ^ 66 = !![(-1 : ℝ), 0, 0; 0, -1, 0; 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_B_power_66_l2484_248496


namespace NUMINAMATH_CALUDE_male_employees_count_l2484_248466

/-- Proves the number of male employees in a company given certain conditions --/
theorem male_employees_count :
  ∀ (m f : ℕ),
  (m : ℚ) / f = 7 / 8 →
  ((m + 3 : ℚ) / f = 8 / 9) →
  m = 189 := by
sorry

end NUMINAMATH_CALUDE_male_employees_count_l2484_248466


namespace NUMINAMATH_CALUDE_crayons_per_box_l2484_248426

/-- Given an industrial machine that makes 321 crayons a day and 45 full boxes a day,
    prove that there are 7 crayons in each box. -/
theorem crayons_per_box :
  ∀ (total_crayons : ℕ) (total_boxes : ℕ),
    total_crayons = 321 →
    total_boxes = 45 →
    ∃ (crayons_per_box : ℕ),
      crayons_per_box * total_boxes ≤ total_crayons ∧
      (crayons_per_box + 1) * total_boxes > total_crayons ∧
      crayons_per_box = 7 :=
by sorry

end NUMINAMATH_CALUDE_crayons_per_box_l2484_248426


namespace NUMINAMATH_CALUDE_expression_evaluation_l2484_248457

theorem expression_evaluation :
  let a : ℝ := 2 + Real.sqrt 3
  (a - 1 - (2 * a - 1) / (a + 1)) / ((a^2 - 4 * a + 4) / (a + 1)) = (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2484_248457


namespace NUMINAMATH_CALUDE_number_divisible_by_19_l2484_248469

theorem number_divisible_by_19 (n : ℕ) : 
  19 ∣ (12000 + 3 * 10^n + 8) := by
  sorry

end NUMINAMATH_CALUDE_number_divisible_by_19_l2484_248469


namespace NUMINAMATH_CALUDE_joseph_kyle_distance_difference_l2484_248481

theorem joseph_kyle_distance_difference : 
  let joseph_speed : ℝ := 50
  let joseph_time : ℝ := 2.5
  let kyle_speed : ℝ := 62
  let kyle_time : ℝ := 2
  let joseph_distance := joseph_speed * joseph_time
  let kyle_distance := kyle_speed * kyle_time
  joseph_distance - kyle_distance = 1 := by
sorry

end NUMINAMATH_CALUDE_joseph_kyle_distance_difference_l2484_248481


namespace NUMINAMATH_CALUDE_base_seven_digits_of_1234_l2484_248447

theorem base_seven_digits_of_1234 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 1234 ∧ 1234 < 7^n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_1234_l2484_248447


namespace NUMINAMATH_CALUDE_range_of_H_l2484_248462

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

-- State the theorem about the range of H
theorem range_of_H :
  Set.range H = Set.Icc 1 5 := by sorry

end NUMINAMATH_CALUDE_range_of_H_l2484_248462


namespace NUMINAMATH_CALUDE_julia_born_1979_l2484_248402

def wayne_age_2021 : ℕ := 37
def peter_age_diff : ℕ := 3
def julia_age_diff : ℕ := 2

def julia_birth_year : ℕ := 2021 - wayne_age_2021 - peter_age_diff - julia_age_diff

theorem julia_born_1979 : julia_birth_year = 1979 := by
  sorry

end NUMINAMATH_CALUDE_julia_born_1979_l2484_248402


namespace NUMINAMATH_CALUDE_nearest_integer_to_power_l2484_248486

theorem nearest_integer_to_power : ∃ n : ℤ, 
  n = 376 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5)^4 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^4 - (m : ℝ)| := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_power_l2484_248486


namespace NUMINAMATH_CALUDE_product_expansion_sum_l2484_248458

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x : ℝ, (4 * x^2 - 3 * x + 2) * (5 - x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + c + d = 26 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l2484_248458


namespace NUMINAMATH_CALUDE_distance_walked_l2484_248473

/-- Represents the walking pace in miles per hour -/
def pace : ℝ := 4

/-- Represents the time walked in hours -/
def time : ℝ := 2

/-- Theorem stating that the distance walked is the product of pace and time -/
theorem distance_walked : pace * time = 8 := by sorry

end NUMINAMATH_CALUDE_distance_walked_l2484_248473


namespace NUMINAMATH_CALUDE_circle_slope_range_l2484_248442

theorem circle_slope_range (x y : ℝ) (h : x^2 + (y - 3)^2 = 1) :
  ∃ (k : ℝ), k ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) ∧ y = k * x :=
sorry

end NUMINAMATH_CALUDE_circle_slope_range_l2484_248442


namespace NUMINAMATH_CALUDE_inequality_constraint_l2484_248474

theorem inequality_constraint (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_constraint_l2484_248474


namespace NUMINAMATH_CALUDE_at_least_one_sum_of_primes_l2484_248476

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a number is the sum of two primes
def isSumOfTwoPrimes (n : ℕ) : Prop :=
  ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ n = p + q

-- Theorem statement
theorem at_least_one_sum_of_primes (n : ℕ) (h : n > 1) :
  isSumOfTwoPrimes (2*n) ∨ isSumOfTwoPrimes (2*n + 2) ∨ isSumOfTwoPrimes (2*n + 4) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_sum_of_primes_l2484_248476


namespace NUMINAMATH_CALUDE_pencil_length_l2484_248416

/-- The length of one pencil when two equal-length pencils together measure 24 cubes -/
theorem pencil_length (total_length : ℕ) (pencil_length : ℕ) : 
  total_length = 24 → 2 * pencil_length = total_length → pencil_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l2484_248416


namespace NUMINAMATH_CALUDE_chess_tournament_players_l2484_248497

theorem chess_tournament_players (total_games : ℕ) (h1 : total_games = 30) : ∃ n : ℕ, n > 0 ∧ total_games = n * (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l2484_248497


namespace NUMINAMATH_CALUDE_tricolor_triangles_odd_l2484_248446

/-- Represents the color of a point -/
inductive Color
| Red
| Yellow
| Blue

/-- Represents a point in the triangle -/
structure Point where
  color : Color

/-- Represents a triangle ABC with m interior points -/
structure ColoredTriangle where
  m : ℕ
  A : Point
  B : Point
  C : Point
  interior_points : Fin m → Point

/-- A function that counts the number of triangles with vertices of all different colors -/
def count_tricolor_triangles (t : ColoredTriangle) : ℕ := sorry

/-- The main theorem stating that the number of triangles with vertices of all different colors is odd -/
theorem tricolor_triangles_odd (t : ColoredTriangle) 
  (h1 : t.A.color = Color.Red)
  (h2 : t.B.color = Color.Yellow)
  (h3 : t.C.color = Color.Blue) :
  Odd (count_tricolor_triangles t) := by sorry

end NUMINAMATH_CALUDE_tricolor_triangles_odd_l2484_248446


namespace NUMINAMATH_CALUDE_sequence_perfect_squares_l2484_248445

theorem sequence_perfect_squares (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, (3 * ((10^n - 1) / 9) + 4) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_perfect_squares_l2484_248445


namespace NUMINAMATH_CALUDE_antenna_spire_height_l2484_248413

/-- The height of the Empire State Building's antenna spire -/
theorem antenna_spire_height :
  let total_height : ℕ := 1454
  let top_floor_height : ℕ := 1250
  let antenna_height := total_height - top_floor_height
  antenna_height = 204 :=
by sorry

end NUMINAMATH_CALUDE_antenna_spire_height_l2484_248413


namespace NUMINAMATH_CALUDE_surface_area_change_after_cube_removal_l2484_248463

/-- Represents a rectangular solid with length, width, and height -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Calculates the change in surface area after removing a cube from the center -/
def surfaceAreaChange (solid : RectangularSolid) (cubeSize : ℝ) : ℝ :=
  6 * cubeSize^2

/-- The theorem to be proved -/
theorem surface_area_change_after_cube_removal :
  let original := RectangularSolid.mk 4 3 2
  let cubeSize := 1
  surfaceAreaChange original cubeSize = 6 := by sorry

end NUMINAMATH_CALUDE_surface_area_change_after_cube_removal_l2484_248463


namespace NUMINAMATH_CALUDE_bella_roses_count_l2484_248464

/-- The number of roses in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of roses Bella received from her parents -/
def roses_from_parents_dozens : ℕ := 2

/-- The number of Bella's dancer friends -/
def number_of_friends : ℕ := 10

/-- The number of roses Bella received from each friend -/
def roses_per_friend : ℕ := 2

/-- The total number of roses Bella received -/
def total_roses : ℕ := roses_from_parents_dozens * dozen + number_of_friends * roses_per_friend

theorem bella_roses_count : total_roses = 44 := by
  sorry

end NUMINAMATH_CALUDE_bella_roses_count_l2484_248464


namespace NUMINAMATH_CALUDE_point_on_line_l2484_248471

/-- Given three points A, B, and C in the xy-plane, where C lies on the line AB,
    prove that the x-coordinate of C is 7.5 -/
theorem point_on_line (A B C : ℝ × ℝ) : 
  A = (0, 10) → 
  B = (5, 0) → 
  C.2 = -5 → 
  (C.1 - A.1) / (B.1 - A.1) = (C.2 - A.2) / (B.2 - A.2) → 
  C.1 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2484_248471


namespace NUMINAMATH_CALUDE_probability_sum_three_two_dice_l2484_248485

theorem probability_sum_three_two_dice : 
  let total_outcomes : ℕ := 6 * 6
  let favorable_outcomes : ℕ := 2
  favorable_outcomes / total_outcomes = (1 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_three_two_dice_l2484_248485


namespace NUMINAMATH_CALUDE_bookstore_sales_l2484_248435

theorem bookstore_sales (tuesday : ℕ) (total : ℕ) : 
  total = tuesday + 3 * tuesday + 9 * tuesday → 
  total = 91 → 
  tuesday = 7 := by
sorry

end NUMINAMATH_CALUDE_bookstore_sales_l2484_248435


namespace NUMINAMATH_CALUDE_marcel_potatoes_l2484_248421

/-- Given the conditions of Marcel and Dale's grocery shopping, prove that Marcel bought 4 potatoes. -/
theorem marcel_potatoes :
  ∀ (marcel_corn dale_corn marcel_potatoes dale_potatoes total_vegetables : ℕ),
  marcel_corn = 10 →
  dale_corn = marcel_corn / 2 →
  dale_potatoes = 8 →
  total_vegetables = 27 →
  total_vegetables = marcel_corn + dale_corn + marcel_potatoes + dale_potatoes →
  marcel_potatoes = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_marcel_potatoes_l2484_248421


namespace NUMINAMATH_CALUDE_factor_w4_minus_16_l2484_248412

theorem factor_w4_minus_16 (w : ℝ) : w^4 - 16 = (w-2)*(w+2)*(w^2+4) := by sorry

end NUMINAMATH_CALUDE_factor_w4_minus_16_l2484_248412


namespace NUMINAMATH_CALUDE_lucy_snack_bar_total_cost_l2484_248487

/-- The cost of a single sandwich at Lucy's Snack Bar -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda at Lucy's Snack Bar -/
def soda_cost : ℕ := 3

/-- The number of sandwiches Lucy wants to buy -/
def num_sandwiches : ℕ := 7

/-- The number of sodas Lucy wants to buy -/
def num_sodas : ℕ := 8

/-- The theorem stating that the total cost of Lucy's purchase is $52 -/
theorem lucy_snack_bar_total_cost : 
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = 52 := by
  sorry

end NUMINAMATH_CALUDE_lucy_snack_bar_total_cost_l2484_248487


namespace NUMINAMATH_CALUDE_jellybean_count_l2484_248431

theorem jellybean_count (black green orange : ℕ) 
  (green_count : green = black + 2)
  (orange_count : orange = green - 1)
  (total_count : black + green + orange = 27) :
  black = 8 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l2484_248431


namespace NUMINAMATH_CALUDE_xyz_value_l2484_248451

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 21) :
  x * y * z = 28 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2484_248451


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l2484_248483

theorem inequality_and_minimum_value :
  (∃ m n : ℝ, (∀ x : ℝ, |x + 1| + |2*x - 1| ≤ 3 ↔ m ≤ x ∧ x ≤ n) ∧
   m = -1 ∧ n = 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 →
   (1/a + 1/b + 1/c ≥ 9/2 ∧ 
    ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 2 ∧ 1/a₀ + 1/b₀ + 1/c₀ = 9/2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l2484_248483


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2484_248450

/-- Represents a company with employees -/
structure Company where
  total_employees : ℕ
  male_employees : ℕ
  female_employees : ℕ

/-- Represents a sample drawn from the company -/
structure Sample where
  female_count : ℕ
  male_count : ℕ

/-- Calculates the sample size given a company and a sample -/
def sample_size (c : Company) (s : Sample) : ℕ :=
  s.female_count + s.male_count

/-- Theorem stating that for a company with 120 employees, of which 90 are male,
    if a stratified sample by gender contains 3 female employees,
    then the total sample size is 12 -/
theorem stratified_sample_size 
  (c : Company) 
  (s : Sample) 
  (h1 : c.total_employees = 120) 
  (h2 : c.male_employees = 90) 
  (h3 : c.female_employees = c.total_employees - c.male_employees) 
  (h4 : s.female_count = 3) :
  sample_size c s = 12 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l2484_248450


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2484_248468

theorem quadratic_equations_solutions :
  (∀ x : ℝ, 2 * x^2 + 4 * x + 1 = 0 ↔ x = -1 + Real.sqrt 2 / 2 ∨ x = -1 - Real.sqrt 2 / 2) ∧
  (∀ x : ℝ, x^2 + 6 * x = 5 ↔ x = -3 + Real.sqrt 14 ∨ x = -3 - Real.sqrt 14) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2484_248468


namespace NUMINAMATH_CALUDE_root_transformation_l2484_248434

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 5 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + 5 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + 5 = 0) → 
  ((3*r₁)^3 - 12*(3*r₁)^2 + 135 = 0) ∧ 
  ((3*r₂)^3 - 12*(3*r₂)^2 + 135 = 0) ∧ 
  ((3*r₃)^3 - 12*(3*r₃)^2 + 135 = 0) := by
sorry

end NUMINAMATH_CALUDE_root_transformation_l2484_248434


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_l2484_248427

theorem min_four_dollar_frisbees :
  ∀ (x y : ℕ),
  x + y = 64 →
  3 * x + 4 * y = 200 →
  y ≥ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_l2484_248427


namespace NUMINAMATH_CALUDE_sum_odd_numbers_l2484_248484

/-- Sum of first n natural numbers -/
def sum_n (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of first n odd numbers -/
def sum_odd (n : ℕ) : ℕ := n^2

/-- The 35th odd number -/
def last_odd : ℕ := 69

/-- Number of odd numbers up to 69 -/
def num_odds : ℕ := (last_odd + 1) / 2

theorem sum_odd_numbers :
  3 * (sum_odd num_odds) = 3675 :=
sorry

end NUMINAMATH_CALUDE_sum_odd_numbers_l2484_248484


namespace NUMINAMATH_CALUDE_gala_trees_count_l2484_248467

theorem gala_trees_count (total : ℕ) (fuji gala honeycrisp : ℕ) : 
  total = fuji + gala + honeycrisp →
  fuji = (2 * total) / 3 →
  honeycrisp = total / 6 →
  fuji + (125 * fuji) / 1000 + (75 * fuji) / 1000 = 315 →
  gala = 66 := by
  sorry

end NUMINAMATH_CALUDE_gala_trees_count_l2484_248467


namespace NUMINAMATH_CALUDE_total_blankets_is_243_l2484_248404

/-- Represents the number of blankets collected over three days --/
def total_blankets : ℕ := 
  let day1_team := 15 * 2
  let day1_online := 5 * 4
  let day1_total := day1_team + day1_online

  let day2_new_members := 5 * 4
  let day2_original_members := 15 * 2 * 3
  let day2_online := 3 * 5
  let day2_total := day2_new_members + day2_original_members + day2_online

  let day3_schools := 22
  let day3_online := 7 * 3
  let day3_business := day2_total / 5
  let day3_total := day3_schools + day3_online + day3_business

  day1_total + day2_total + day3_total

/-- Theorem stating that the total number of blankets collected is 243 --/
theorem total_blankets_is_243 : total_blankets = 243 := by
  sorry

end NUMINAMATH_CALUDE_total_blankets_is_243_l2484_248404


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l2484_248423

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l2484_248423


namespace NUMINAMATH_CALUDE_chili_paste_can_difference_l2484_248492

def large_can_size : ℕ := 25
def small_can_size : ℕ := 15
def large_cans_needed : ℕ := 45

theorem chili_paste_can_difference :
  (large_cans_needed * large_can_size) / small_can_size - large_cans_needed = 30 :=
by sorry

end NUMINAMATH_CALUDE_chili_paste_can_difference_l2484_248492


namespace NUMINAMATH_CALUDE_pool_capacity_theorem_l2484_248428

/-- Represents the dimensions and draining parameters of a pool -/
structure Pool :=
  (width : ℝ)
  (length : ℝ)
  (depth : ℝ)
  (drainRate : ℝ)
  (drainTime : ℝ)

/-- Calculates the volume of a pool -/
def poolVolume (p : Pool) : ℝ :=
  p.width * p.length * p.depth

/-- Calculates the amount of water drained from a pool -/
def waterDrained (p : Pool) : ℝ :=
  p.drainRate * p.drainTime

/-- Theorem stating that if the water drained equals the pool volume, 
    then the pool was at 100% capacity -/
theorem pool_capacity_theorem (p : Pool) 
  (h1 : p.width = 80)
  (h2 : p.length = 150)
  (h3 : p.depth = 10)
  (h4 : p.drainRate = 60)
  (h5 : p.drainTime = 2000)
  (h6 : waterDrained p = poolVolume p) :
  poolVolume p / poolVolume p = 1 := by
  sorry


end NUMINAMATH_CALUDE_pool_capacity_theorem_l2484_248428


namespace NUMINAMATH_CALUDE_min_value_expression_l2484_248422

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt 3 * 3^(a + b) * (1/a + 1/b) ≥ 12 ∧
  (Real.sqrt 3 * 3^(a + b) * (1/a + 1/b) = 12 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2484_248422


namespace NUMINAMATH_CALUDE_binomial_sum_l2484_248489

theorem binomial_sum : Nat.choose 18 4 + Nat.choose 5 2 = 3070 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l2484_248489


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l2484_248480

theorem triangle_angle_not_all_greater_than_60 :
  ¬ ∀ (a b c : ℝ), 
    (a > 0) → (b > 0) → (c > 0) → 
    (a + b + c = 180) → 
    (a > 60 ∧ b > 60 ∧ c > 60) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l2484_248480


namespace NUMINAMATH_CALUDE_virginia_adrienne_difference_l2484_248448

/-- Represents the teaching years of Virginia, Adrienne, and Dennis -/
structure TeachingYears where
  virginia : ℕ
  adrienne : ℕ
  dennis : ℕ

/-- The conditions of the teaching years problem -/
def TeachingProblem (t : TeachingYears) : Prop :=
  t.virginia + t.adrienne + t.dennis = 75 ∧
  t.dennis = 34 ∧
  ∃ (x : ℕ), t.virginia = t.adrienne + x ∧ t.virginia = t.dennis - x

/-- The theorem stating that Virginia has taught 9 more years than Adrienne -/
theorem virginia_adrienne_difference (t : TeachingYears) 
  (h : TeachingProblem t) : t.virginia - t.adrienne = 9 := by
  sorry

end NUMINAMATH_CALUDE_virginia_adrienne_difference_l2484_248448


namespace NUMINAMATH_CALUDE_count_four_digit_with_seven_l2484_248461

/-- A four-digit positive integer with 7 as the thousands digit -/
def FourDigitWithSeven : Type := { n : ℕ // 7000 ≤ n ∧ n ≤ 7999 }

/-- The count of four-digit positive integers with 7 as the thousands digit -/
def CountFourDigitWithSeven : ℕ := Finset.card (Finset.filter (λ n => 7000 ≤ n ∧ n ≤ 7999) (Finset.range 10000))

theorem count_four_digit_with_seven :
  CountFourDigitWithSeven = 1000 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_with_seven_l2484_248461


namespace NUMINAMATH_CALUDE_parametric_to_standard_equation_l2484_248417

theorem parametric_to_standard_equation (x y α : ℝ) :
  x = Real.sqrt 3 * Real.cos α + 2 ∧ 
  y = Real.sqrt 3 * Real.sin α - 3 →
  (x - 2)^2 + (y + 3)^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_parametric_to_standard_equation_l2484_248417


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l2484_248443

theorem smallest_n_for_sqrt_difference : ∃ n : ℕ+, (∀ m : ℕ+, m < n → Real.sqrt m.val - Real.sqrt (m.val - 1) ≥ 0.1) ∧ (Real.sqrt n.val - Real.sqrt (n.val - 1) < 0.1) ∧ n = 26 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l2484_248443


namespace NUMINAMATH_CALUDE_power_of_six_tens_digit_one_l2484_248407

theorem power_of_six_tens_digit_one : ∃ n : ℕ, (6^n) % 100 ≥ 10 ∧ (6^n) % 100 < 20 := by
  sorry

end NUMINAMATH_CALUDE_power_of_six_tens_digit_one_l2484_248407


namespace NUMINAMATH_CALUDE_unique_digit_factorial_sum_l2484_248401

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_factorial_sum (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  factorial d1 + factorial d2 + factorial d3

def has_zero_digit (n : ℕ) : Prop :=
  n % 10 = 0 ∨ (n / 10) % 10 = 0 ∨ n / 100 = 0

theorem unique_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = digit_factorial_sum n ∧ has_zero_digit n :=
sorry

end NUMINAMATH_CALUDE_unique_digit_factorial_sum_l2484_248401


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2484_248430

theorem gain_percent_calculation (MP : ℝ) (MP_pos : MP > 0) : 
  let CP := 0.64 * MP
  let SP := 0.86 * MP
  let gain := SP - CP
  let gain_percent := (gain / CP) * 100
  gain_percent = 34.375 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2484_248430


namespace NUMINAMATH_CALUDE_work_rate_problem_l2484_248425

theorem work_rate_problem (A B C : ℚ) 
  (h1 : A + B = 1/8)
  (h2 : B + C = 1/12)
  (h3 : A + B + C = 1/6) :
  A + C = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_work_rate_problem_l2484_248425


namespace NUMINAMATH_CALUDE_dinner_lunch_cake_difference_l2484_248452

theorem dinner_lunch_cake_difference : 
  let lunch_cakes : ℕ := 6
  let dinner_cakes : ℕ := 9
  dinner_cakes - lunch_cakes = 3 := by sorry

end NUMINAMATH_CALUDE_dinner_lunch_cake_difference_l2484_248452


namespace NUMINAMATH_CALUDE_det_A_l2484_248459

def A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 4; -2, 3]

theorem det_A : Matrix.det A = 32 := by sorry

end NUMINAMATH_CALUDE_det_A_l2484_248459


namespace NUMINAMATH_CALUDE_triangle_theorem_l2484_248479

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) (h : 4 * t.a^2 = t.b * t.c * Real.cos t.A + t.a * t.c * Real.cos t.B) :
  (t.a / t.c = 1 / 2) ∧
  (t.a = 1 → Real.cos t.B = 3 / 4 → ∃ D : ℝ × ℝ, 
    (D.1 = (t.a + t.c) / 2 ∧ D.2 = 0) → 
    Real.sqrt ((D.1 - t.a)^2 + D.2^2) = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2484_248479


namespace NUMINAMATH_CALUDE_sum_of_consecutive_terms_l2484_248403

theorem sum_of_consecutive_terms (n : ℝ) : n + (n + 1) + (n + 2) + (n + 3) = 20 → n = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_terms_l2484_248403


namespace NUMINAMATH_CALUDE_smallest_integer_sqrt_difference_l2484_248456

theorem smallest_integer_sqrt_difference (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < 250001 → Real.sqrt m - Real.sqrt (m - 1) ≥ (1 : ℝ) / 1000) ∧ 
  (Real.sqrt 250001 - Real.sqrt 250000 < (1 : ℝ) / 1000) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_sqrt_difference_l2484_248456


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l2484_248414

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangles (r1 r2 : Rectangle) 
  (h1 : r1.length = 4)
  (h2 : r1.width = 30)
  (h3 : r2.width = 15)
  (h4 : area r1 = area r2) :
  r2.length = 8 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l2484_248414


namespace NUMINAMATH_CALUDE_cubic_root_product_l2484_248420

theorem cubic_root_product (p q r : ℝ) : 
  (3 * p^3 - 9 * p^2 + 5 * p - 15 = 0) ∧ 
  (3 * q^3 - 9 * q^2 + 5 * q - 15 = 0) ∧ 
  (3 * r^3 - 9 * r^2 + 5 * r - 15 = 0) → 
  p * q * r = 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l2484_248420


namespace NUMINAMATH_CALUDE_quadratic_condition_l2484_248411

theorem quadratic_condition (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, m * x^2 - 4*x + 3 = a * x^2 + b * x + c) → m ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_condition_l2484_248411


namespace NUMINAMATH_CALUDE_largest_number_less_than_150_divisible_by_3_l2484_248441

theorem largest_number_less_than_150_divisible_by_3 :
  ∃ (x : ℕ), x = 12 ∧
  (∀ (y : ℕ), 11 * y < 150 ∧ 3 ∣ y → y ≤ x) ∧
  11 * x < 150 ∧ 3 ∣ x :=
by sorry

end NUMINAMATH_CALUDE_largest_number_less_than_150_divisible_by_3_l2484_248441


namespace NUMINAMATH_CALUDE_haley_trees_l2484_248472

theorem haley_trees (initial_trees : ℕ) : 
  (((initial_trees - 5) - 8) - 3 = 12) → initial_trees = 28 := by
  sorry

end NUMINAMATH_CALUDE_haley_trees_l2484_248472


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_l2484_248438

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials :
  unitsDigit (sumFactorials 100) = unitsDigit (sumFactorials 4) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_l2484_248438


namespace NUMINAMATH_CALUDE_area_of_four_isosceles_triangles_l2484_248432

/-- The area of a figure composed of four isosceles triangles -/
theorem area_of_four_isosceles_triangles :
  ∀ (s : ℝ) (θ : ℝ),
  s = 1 →
  θ = 75 * π / 180 →
  2 * s^2 * Real.sin θ = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_four_isosceles_triangles_l2484_248432


namespace NUMINAMATH_CALUDE_unique_fraction_representation_l2484_248470

theorem unique_fraction_representation (n : ℕ) (hn : n > 0) :
  ∃! (a b : ℝ), a > 0 ∧ b > 0 ∧ (2 * n + 1 : ℝ) / (n * (n + 1)) = a / n + b / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_representation_l2484_248470


namespace NUMINAMATH_CALUDE_circle_center_center_coordinates_l2484_248409

theorem circle_center (x y : ℝ) : 
  x^2 + y^2 - 4*x - 2*y - 5 = 0 → (x - 2)^2 + (y - 1)^2 = 10 := by
  sorry

theorem center_coordinates : 
  ∃ (h : x^2 + y^2 - 4*x - 2*y - 5 = 0), (x = 2 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_center_coordinates_l2484_248409


namespace NUMINAMATH_CALUDE_no_polynomial_satisfies_conditions_l2484_248429

/-- A polynomial function over real numbers. -/
def PolynomialFunction := ℝ → ℝ

/-- The degree of a polynomial function. -/
noncomputable def degree (f : PolynomialFunction) : ℕ := sorry

/-- Predicate for a function satisfying the given conditions. -/
def satisfiesConditions (f : PolynomialFunction) : Prop :=
  ∀ x : ℝ, f (x + 1) = (f x)^2 ∧ (f x)^2 = f (f x)

theorem no_polynomial_satisfies_conditions :
  ¬ ∃ f : PolynomialFunction, degree f ≥ 1 ∧ satisfiesConditions f := by sorry

end NUMINAMATH_CALUDE_no_polynomial_satisfies_conditions_l2484_248429
