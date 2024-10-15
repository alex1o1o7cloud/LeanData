import Mathlib

namespace NUMINAMATH_CALUDE_jenny_distance_difference_l741_74125

theorem jenny_distance_difference (run_distance walk_distance : ℝ) 
  (h1 : run_distance = 0.6)
  (h2 : walk_distance = 0.4) :
  run_distance - walk_distance = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_jenny_distance_difference_l741_74125


namespace NUMINAMATH_CALUDE_circle_equation_satisfies_conditions_l741_74153

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def PointOnLine (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

theorem circle_equation_satisfies_conditions :
  ∃ (h k r : ℝ),
    -- The circle's equation
    (∀ x y, CircleEquation h k r x y ↔ (x - 2)^2 + (y + 1)^2 = 5) ∧
    -- The center lies on the line 3x + y - 5 = 0
    PointOnLine 3 1 (-5) h k ∧
    -- The circle passes through (0, 0)
    CircleEquation h k r 0 0 ∧
    -- The circle passes through (4, 0)
    CircleEquation h k r 4 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_satisfies_conditions_l741_74153


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeros_l741_74111

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 500! has 124 trailing zeros -/
theorem factorial_500_trailing_zeros :
  trailingZeros 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeros_l741_74111


namespace NUMINAMATH_CALUDE_polygon_sides_diagonals_l741_74110

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon has 11 sides if the number of its diagonals is 33 more than the number of its sides -/
theorem polygon_sides_diagonals : 
  ∃ (n : ℕ), n > 3 ∧ num_diagonals n = n + 33 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_diagonals_l741_74110


namespace NUMINAMATH_CALUDE_pizza_calories_l741_74174

theorem pizza_calories (total_slices : ℕ) (eaten_slices_1 : ℕ) (calories_1 : ℕ) 
  (eaten_slices_2 : ℕ) (calories_2 : ℕ) : 
  total_slices = 12 → 
  eaten_slices_1 = 3 →
  calories_1 = 300 →
  eaten_slices_2 = 4 →
  calories_2 = 400 →
  eaten_slices_1 * calories_1 + eaten_slices_2 * calories_2 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_pizza_calories_l741_74174


namespace NUMINAMATH_CALUDE_sqrt_t4_4t2_4_l741_74146

theorem sqrt_t4_4t2_4 (t : ℝ) : Real.sqrt (t^4 + 4*t^2 + 4) = |t^2 + 2| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_t4_4t2_4_l741_74146


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l741_74163

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 80 → 
  b = 150 → 
  c^2 = a^2 + b^2 → 
  c = 170 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l741_74163


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l741_74120

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  (3 * x^2 - 2 * x) / ((x - 4) * (x - 2)^2) = 
  10 / (x - 4) + (-7) / (x - 2) + (-4) / (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l741_74120


namespace NUMINAMATH_CALUDE_range_of_a_l741_74106

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + a| + |x - 1| + a < 2011) ↔ a < 1005 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l741_74106


namespace NUMINAMATH_CALUDE_tan_15_30_product_equals_two_l741_74166

theorem tan_15_30_product_equals_two :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 :=
by
  have tan_45_eq_1 : Real.tan (45 * π / 180) = 1 := by sorry
  have tan_sum_15_30 : Real.tan ((15 + 30) * π / 180) = 
    (Real.tan (15 * π / 180) + Real.tan (30 * π / 180)) / 
    (1 - Real.tan (15 * π / 180) * Real.tan (30 * π / 180)) := by sorry
  sorry

end NUMINAMATH_CALUDE_tan_15_30_product_equals_two_l741_74166


namespace NUMINAMATH_CALUDE_festival_attendance_theorem_l741_74118

/-- Represents the attendance for each day of a four-day music festival --/
structure FestivalAttendance where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Calculates the total attendance for all four days --/
def totalAttendance (attendance : FestivalAttendance) : ℕ :=
  attendance.day1 + attendance.day2 + attendance.day3 + attendance.day4

/-- Theorem stating that the total attendance for the festival is 3600 --/
theorem festival_attendance_theorem (attendance : FestivalAttendance) :
  (attendance.day2 = attendance.day1 / 2) →
  (attendance.day3 = attendance.day1 * 3) →
  (attendance.day4 = attendance.day2 * 2) →
  (totalAttendance attendance = 3600) :=
by
  sorry

#check festival_attendance_theorem

end NUMINAMATH_CALUDE_festival_attendance_theorem_l741_74118


namespace NUMINAMATH_CALUDE_youth_gathering_count_l741_74148

/-- The number of youths at a gathering, given the conditions from the problem. -/
def total_youths (male_youths : ℕ) : ℕ := 2 * male_youths + 12

/-- The theorem stating the total number of youths at the gathering. -/
theorem youth_gathering_count : 
  ∃ (male_youths : ℕ), 
    (male_youths : ℚ) / (total_youths male_youths : ℚ) = 9 / 20 ∧ 
    total_youths male_youths = 120 := by
  sorry


end NUMINAMATH_CALUDE_youth_gathering_count_l741_74148


namespace NUMINAMATH_CALUDE_average_difference_l741_74168

theorem average_difference (x : ℝ) : (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 8 ↔ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l741_74168


namespace NUMINAMATH_CALUDE_computer_employee_savings_l741_74159

/-- Calculates the employee savings on a computer purchase given the initial cost,
    markup percentage, and employee discount percentage. -/
def employeeSavings (initialCost : ℝ) (markupPercentage : ℝ) (discountPercentage : ℝ) : ℝ :=
  let retailPrice := initialCost * (1 + markupPercentage)
  retailPrice * discountPercentage

/-- Theorem stating that an employee saves $86.25 when buying a computer
    with a 15% markup and 15% employee discount, given an initial cost of $500. -/
theorem computer_employee_savings :
  employeeSavings 500 0.15 0.15 = 86.25 := by
  sorry


end NUMINAMATH_CALUDE_computer_employee_savings_l741_74159


namespace NUMINAMATH_CALUDE_todd_ate_five_cupcakes_l741_74109

def cupcake_problem (initial_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) : ℕ :=
  initial_cupcakes - packages * cupcakes_per_package

theorem todd_ate_five_cupcakes :
  cupcake_problem 50 9 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_todd_ate_five_cupcakes_l741_74109


namespace NUMINAMATH_CALUDE_morning_pear_sales_l741_74177

/-- Represents the sale of pears by a salesman in a day. -/
structure PearSales where
  morning : ℝ
  afternoon : ℝ
  total : ℝ

/-- Theorem stating the number of kilograms of pears sold in the morning. -/
theorem morning_pear_sales (sales : PearSales) 
  (h1 : sales.afternoon = 2 * sales.morning)
  (h2 : sales.total = 360)
  (h3 : sales.total = sales.morning + sales.afternoon) :
  sales.morning = 120 := by
  sorry

end NUMINAMATH_CALUDE_morning_pear_sales_l741_74177


namespace NUMINAMATH_CALUDE_sequence_is_increasing_l741_74124

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem sequence_is_increasing (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) - a n - 2 = 0) : 
  is_increasing a :=
sorry

end NUMINAMATH_CALUDE_sequence_is_increasing_l741_74124


namespace NUMINAMATH_CALUDE_F_properties_l741_74197

noncomputable def f (x : ℝ) : ℝ := 3 - 2 * abs x

noncomputable def g (x : ℝ) : ℝ := x^2 - 2*x

noncomputable def F (x : ℝ) : ℝ :=
  if f x ≥ g x then g x else f x

theorem F_properties :
  (∃ (M : ℝ), M = 7 - 2 * Real.sqrt 7 ∧ ∀ (x : ℝ), F x ≤ M) ∧
  (¬ ∃ (m : ℝ), ∀ (x : ℝ), F x ≥ m) :=
sorry

end NUMINAMATH_CALUDE_F_properties_l741_74197


namespace NUMINAMATH_CALUDE_congruent_rectangle_perimeter_l741_74129

/-- Given a rectangle of width y and height 2y divided into a square of side x
    and four congruent rectangles, the perimeter of one of the congruent rectangles
    is 3y - 2x. -/
theorem congruent_rectangle_perimeter
  (y : ℝ) (x : ℝ)
  (h1 : y > 0)
  (h2 : x > 0)
  (h3 : x < y)
  (h4 : x < 2*y) :
  ∃ (l w : ℝ),
    l > 0 ∧ w > 0 ∧
    x + 2*l = y ∧
    x + 2*w = 2*y ∧
    2*l + 2*w = 3*y - 2*x :=
by sorry

end NUMINAMATH_CALUDE_congruent_rectangle_perimeter_l741_74129


namespace NUMINAMATH_CALUDE_speed_ratio_problem_l741_74127

/-- 
Given two people traveling in opposite directions for one hour, 
if one person takes 35 minutes longer to reach the other's destination when they swap,
then the ratio of their speeds is 3:4.
-/
theorem speed_ratio_problem (v₁ v₂ : ℝ) : 
  v₁ > 0 → v₂ > 0 → 
  (60 * v₂ = 60 * v₁ / v₂ + 35 * v₁) → 
  v₁ / v₂ = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_speed_ratio_problem_l741_74127


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l741_74140

theorem simplify_fraction_product : 8 * (15 / 4) * (-25 / 45) = -50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l741_74140


namespace NUMINAMATH_CALUDE_M_subset_N_l741_74107

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
def N : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- State the theorem
theorem M_subset_N : M ⊆ N := by sorry

end NUMINAMATH_CALUDE_M_subset_N_l741_74107


namespace NUMINAMATH_CALUDE_range_of_a_l741_74119

theorem range_of_a (x a : ℝ) : 
  (∀ x, (x - 1) / (x - 3) < 0 → |x - a| < 2) ∧ 
  (∃ x, |x - a| < 2 ∧ (x - 1) / (x - 3) ≥ 0) →
  a ∈ Set.Icc 1 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l741_74119


namespace NUMINAMATH_CALUDE_equation_solution_l741_74155

theorem equation_solution (y : ℚ) : 
  (y ≠ 5) → (y ≠ (3/2 : ℚ)) → 
  ((y^2 - 12*y + 35) / (y - 5) + (2*y^2 + 9*y - 18) / (2*y - 3) = 0) → 
  y = (1/2 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l741_74155


namespace NUMINAMATH_CALUDE_interesting_quartet_inequality_l741_74193

theorem interesting_quartet_inequality (p a b c : ℕ) : 
  Nat.Prime p → p % 2 = 1 →
  a ≠ b → b ≠ c → a ≠ c →
  (ab + 1) % p = 0 →
  (ac + 1) % p = 0 →
  (bc + 1) % p = 0 →
  (p : ℚ) + 2 ≤ (a + b + c : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_interesting_quartet_inequality_l741_74193


namespace NUMINAMATH_CALUDE_divisibility_prime_factorization_l741_74114

theorem divisibility_prime_factorization (a b : ℕ) : 
  (a ∣ b) ↔ (∀ p : ℕ, ∀ k : ℕ, Prime p → (p^k ∣ a) → (p^k ∣ b)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_prime_factorization_l741_74114


namespace NUMINAMATH_CALUDE_part1_part2_l741_74156

-- Part 1
def U1 : Set ℕ := {2, 3, 4}
def A1 : Set ℕ := {4, 3}
def B1 : Set ℕ := ∅

theorem part1 :
  (U1 \ A1 = {2}) ∧ (U1 \ B1 = U1) := by sorry

-- Part 2
def U2 : Set ℝ := {x | x ≤ 4}
def A2 : Set ℝ := {x | -2 < x ∧ x < 3}
def B2 : Set ℝ := {x | -3 < x ∧ x ≤ 3}

theorem part2 :
  (U2 \ A2 = {x | x ≤ -2 ∨ (3 ≤ x ∧ x ≤ 4)}) ∧
  (A2 ∩ B2 = {x | -2 < x ∧ x < 3}) ∧
  (U2 \ (A2 ∩ B2) = {x | x ≤ -2 ∨ (3 ≤ x ∧ x ≤ 4)}) ∧
  ((U2 \ A2) ∩ B2 = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3}) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l741_74156


namespace NUMINAMATH_CALUDE_problem_solution_l741_74167

theorem problem_solution (x : ℚ) : (1/2 * (12*x + 3) = 3*x + 2) → x = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l741_74167


namespace NUMINAMATH_CALUDE_min_digit_divisible_by_72_l741_74138

theorem min_digit_divisible_by_72 :
  ∃ (x : ℕ), x < 10 ∧ (983480 + x) % 72 = 0 ∧
  ∀ (y : ℕ), y < x → (983480 + y) % 72 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_digit_divisible_by_72_l741_74138


namespace NUMINAMATH_CALUDE_factor_x4_plus_64_l741_74143

theorem factor_x4_plus_64 (x : ℝ) : 
  x^4 + 64 = (x^2 + 4*x + 8) * (x^2 - 4*x + 8) := by
sorry

end NUMINAMATH_CALUDE_factor_x4_plus_64_l741_74143


namespace NUMINAMATH_CALUDE_unique_solution_l741_74187

theorem unique_solution : ∃! x : ℝ, x > 12 ∧ (x - 6) / 12 = 5 / (x - 12) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l741_74187


namespace NUMINAMATH_CALUDE_last_infected_on_fifth_exam_l741_74194

def total_mice : ℕ := 10
def infected_mice : ℕ := 3
def healthy_mice : ℕ := 7

-- The number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- The number of ways to arrange k items from n items
def arrange (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem last_infected_on_fifth_exam :
  choose infected_mice 2 * arrange 4 2 * choose healthy_mice 2 * arrange 2 2 = 1512 := by
  sorry

end NUMINAMATH_CALUDE_last_infected_on_fifth_exam_l741_74194


namespace NUMINAMATH_CALUDE_land_development_break_even_l741_74192

/-- Calculates the break-even price per lot given the total acreage, price per acre, and number of lots. -/
def breakEvenPricePerLot (totalAcres : ℕ) (pricePerAcre : ℕ) (numberOfLots : ℕ) : ℕ :=
  (totalAcres * pricePerAcre) / numberOfLots

/-- Proves that for 4 acres at $1,863 per acre split into 9 lots, the break-even price is $828 per lot. -/
theorem land_development_break_even :
  breakEvenPricePerLot 4 1863 9 = 828 := by
  sorry

#eval breakEvenPricePerLot 4 1863 9

end NUMINAMATH_CALUDE_land_development_break_even_l741_74192


namespace NUMINAMATH_CALUDE_sunglasses_and_caps_probability_l741_74136

theorem sunglasses_and_caps_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (prob_sunglasses_also_cap : ℚ) :
  total_sunglasses = 80 →
  total_caps = 45 →
  prob_sunglasses_also_cap = 3/8 →
  (total_sunglasses * prob_sunglasses_also_cap : ℚ) / total_caps = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_and_caps_probability_l741_74136


namespace NUMINAMATH_CALUDE_ten_factorial_minus_nine_factorial_l741_74188

theorem ten_factorial_minus_nine_factorial : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_ten_factorial_minus_nine_factorial_l741_74188


namespace NUMINAMATH_CALUDE_special_ellipse_properties_l741_74105

/-- An ellipse with one vertex at (0,1) and focus on the x-axis -/
structure SpecialEllipse where
  /-- The right focus of the ellipse -/
  focus : ℝ × ℝ
  /-- The distance from the right focus to the line x-y+2√2=0 is 3 -/
  focus_distance : (|focus.1 + 2 * Real.sqrt 2| : ℝ) / Real.sqrt 2 = 3

/-- The equation of the ellipse -/
def ellipse_equation (e : SpecialEllipse) (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 = 1

/-- A line passing through (0,1) -/
structure LineThroughA where
  /-- The slope of the line -/
  k : ℝ

/-- The equation of a line passing through (0,1) -/
def line_equation (l : LineThroughA) (x y : ℝ) : Prop :=
  y = l.k * x + 1

/-- The theorem to be proved -/
theorem special_ellipse_properties (e : SpecialEllipse) :
  (∀ x y, ellipse_equation e x y ↔ x^2 / 3 + y^2 = 1) ∧
  (∀ l : LineThroughA, line_equation l = line_equation ⟨1⟩ ∨ line_equation l = line_equation ⟨-1⟩ →
    ∀ l' : LineThroughA, ∃ x y, ellipse_equation e x y ∧ line_equation l x y ∧ line_equation l' x y →
      ∀ x' y', ellipse_equation e x' y' ∧ line_equation l' x' y' →
        (x - 0)^2 + (y - 1)^2 ≥ (x' - 0)^2 + (y' - 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_special_ellipse_properties_l741_74105


namespace NUMINAMATH_CALUDE_perception_permutations_l741_74182

def word_length : ℕ := 10
def repeated_letters : ℕ := 2

theorem perception_permutations :
  (word_length.factorial) / ((repeated_letters.factorial) * (repeated_letters.factorial)) = 907200 :=
by sorry

end NUMINAMATH_CALUDE_perception_permutations_l741_74182


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l741_74131

/-- If a, b, and c form a geometric sequence, then ax^2 + bx + c has no real roots -/
theorem quadratic_no_real_roots (a b c : ℝ) (h : b^2 = a*c) : 
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l741_74131


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l741_74181

theorem min_value_of_fraction_sum (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : (a - b) * (b - c) * (c - a) = -16) :
  ∃ (m : ℝ), m = 3/4 ∧ ∀ x y z : ℝ, 
    x > y → y > z → (x - y) * (y - z) * (z - x) = -16 → 
    1 / (x - y) + 1 / (y - z) - 1 / (z - x) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l741_74181


namespace NUMINAMATH_CALUDE_sequence_terms_l741_74123

def a (n : ℕ) : ℤ := (-1)^(n+1) * (3*n - 2)

theorem sequence_terms : 
  (a 1 = 1) ∧ (a 2 = -4) ∧ (a 3 = 7) ∧ (a 4 = -10) ∧ (a 5 = 13) := by
  sorry

end NUMINAMATH_CALUDE_sequence_terms_l741_74123


namespace NUMINAMATH_CALUDE_minimum_guests_l741_74147

theorem minimum_guests (total_food : ℝ) (max_per_guest : ℝ) (min_guests : ℕ) :
  total_food = 411 →
  max_per_guest = 2.5 →
  min_guests = ⌈total_food / max_per_guest⌉ →
  min_guests = 165 := by
  sorry

end NUMINAMATH_CALUDE_minimum_guests_l741_74147


namespace NUMINAMATH_CALUDE_least_cookies_l741_74165

theorem least_cookies (b : ℕ) : 
  b > 0 ∧ 
  b % 6 = 5 ∧ 
  b % 8 = 3 ∧ 
  b % 9 = 7 ∧
  (∀ c : ℕ, c > 0 ∧ c % 6 = 5 ∧ c % 8 = 3 ∧ c % 9 = 7 → b ≤ c) → 
  b = 179 := by
sorry

end NUMINAMATH_CALUDE_least_cookies_l741_74165


namespace NUMINAMATH_CALUDE_sequence_inequality_l741_74178

/-- S(n,m) is the number of sequences of length n consisting of 0 and 1 
    where there exists a 0 in any consecutive m digits -/
def S (n m : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem sequence_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  S (2015 * n) n * S (2015 * m) m ≥ S (2015 * n) m * S (2015 * m) n := by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l741_74178


namespace NUMINAMATH_CALUDE_max_hot_dogs_proof_l741_74190

-- Define pack sizes and prices
structure PackInfo where
  size : Nat
  price : Rat

-- Define the problem parameters
def budget : Rat := 300
def packInfos : List PackInfo := [
  ⟨8, 155/100⟩,
  ⟨20, 305/100⟩,
  ⟨50, 745/100⟩,
  ⟨100, 1410/100⟩,
  ⟨250, 2295/100⟩
]
def discountThreshold : Nat := 10
def discountRate : Rat := 5/100
def maxPacksPerSize : Nat := 30
def minTotalPacks : Nat := 15

-- Define a function to calculate the total number of hot dogs
def totalHotDogs (purchases : List (PackInfo × Nat)) : Nat :=
  purchases.foldl (fun acc (pack, quantity) => acc + pack.size * quantity) 0

-- Define a function to calculate the total cost
def totalCost (purchases : List (PackInfo × Nat)) : Rat :=
  purchases.foldl (fun acc (pack, quantity) =>
    let basePrice := pack.price * quantity
    let discountedPrice := if quantity > discountThreshold then basePrice * (1 - discountRate) else basePrice
    acc + discountedPrice
  ) 0

-- Theorem statement
theorem max_hot_dogs_proof :
  ∃ (purchases : List (PackInfo × Nat)),
    totalHotDogs purchases = 3250 ∧
    totalCost purchases ≤ budget ∧
    purchases.all (fun (_, quantity) => quantity ≤ maxPacksPerSize) ∧
    purchases.foldl (fun acc (_, quantity) => acc + quantity) 0 ≥ minTotalPacks ∧
    (∀ (otherPurchases : List (PackInfo × Nat)),
      totalCost otherPurchases ≤ budget →
      purchases.all (fun (_, quantity) => quantity ≤ maxPacksPerSize) →
      purchases.foldl (fun acc (_, quantity) => acc + quantity) 0 ≥ minTotalPacks →
      totalHotDogs otherPurchases ≤ totalHotDogs purchases) :=
by
  sorry

end NUMINAMATH_CALUDE_max_hot_dogs_proof_l741_74190


namespace NUMINAMATH_CALUDE_best_athlete_is_A_l741_74158

structure Athlete where
  name : String
  average : ℝ
  variance : ℝ

def betterPerformance (a b : Athlete) : Prop :=
  a.average > b.average ∨ (a.average = b.average ∧ a.variance < b.variance)

theorem best_athlete_is_A :
  let A := Athlete.mk "A" 185 3.6
  let B := Athlete.mk "B" 180 3.6
  let C := Athlete.mk "C" 185 7.4
  let D := Athlete.mk "D" 180 8.1
  ∀ x ∈ [B, C, D], betterPerformance A x := by
  sorry

end NUMINAMATH_CALUDE_best_athlete_is_A_l741_74158


namespace NUMINAMATH_CALUDE_tangent_ratio_given_sine_condition_l741_74135

theorem tangent_ratio_given_sine_condition (α : Real) 
  (h : 5 * Real.sin (2 * α) = Real.sin (2 * Real.pi / 180)) : 
  Real.tan (α + Real.pi / 180) / Real.tan (α - Real.pi / 180) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ratio_given_sine_condition_l741_74135


namespace NUMINAMATH_CALUDE_janet_return_time_l741_74162

/-- Represents the number of blocks Janet walks in each direction --/
structure WalkingDistance where
  north : ℕ
  west : ℕ
  south : ℕ
  east : ℕ

/-- Calculates the time taken to walk a given distance at a given speed --/
def timeToWalk (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

/-- Janet's walking pattern and speed --/
def janet : WalkingDistance × ℕ :=
  ({ north := 3
   , west := 3 * 7
   , south := 3
   , east := 3 * 2
   }, 2)

/-- Theorem: Janet takes 9 minutes to return home --/
theorem janet_return_time : 
  let (walk, speed) := janet
  timeToWalk (walk.south + (walk.west - walk.east)) speed = 9 := by
  sorry

end NUMINAMATH_CALUDE_janet_return_time_l741_74162


namespace NUMINAMATH_CALUDE_house_sale_percentage_l741_74130

theorem house_sale_percentage (market_value : ℝ) (num_people : ℕ) (after_tax_per_person : ℝ) (tax_rate : ℝ) :
  market_value = 500000 →
  num_people = 4 →
  after_tax_per_person = 135000 →
  tax_rate = 0.1 →
  ((num_people * after_tax_per_person / (1 - tax_rate) - market_value) / market_value) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_percentage_l741_74130


namespace NUMINAMATH_CALUDE_not_perfect_square_polynomial_l741_74180

theorem not_perfect_square_polynomial (n : ℕ) : ¬∃ (m : ℕ), n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_polynomial_l741_74180


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l741_74113

/-- The distance between the foci of an ellipse described by the equation
    √((x-4)² + (y+5)²) + √((x+6)² + (y-7)²) = 22 is equal to 2√2. -/
theorem ellipse_foci_distance : 
  let ellipse := {p : ℝ × ℝ | Real.sqrt ((p.1 - 4)^2 + (p.2 + 5)^2) + 
                               Real.sqrt ((p.1 + 6)^2 + (p.2 - 7)^2) = 22}
  let foci := ((4, -5), (-6, 7))
  ∃ (d : ℝ), d = Real.sqrt 8 ∧ 
    d = Real.sqrt ((foci.1.1 - foci.2.1)^2 + (foci.1.2 - foci.2.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l741_74113


namespace NUMINAMATH_CALUDE_franklins_gathering_theorem_l741_74173

/-- Represents the number of handshakes in Franklin's gathering --/
def franklins_gathering_handshakes (num_couples : ℕ) : ℕ :=
  let num_men := num_couples
  let num_women := num_couples
  let handshakes_among_men := num_men * (num_men - 1 + num_women - 1) / 2
  let franklins_handshakes := num_women
  handshakes_among_men + franklins_handshakes

/-- Theorem stating that the number of handshakes in Franklin's gathering with 15 couples is 225 --/
theorem franklins_gathering_theorem :
  franklins_gathering_handshakes 15 = 225 := by
  sorry

#eval franklins_gathering_handshakes 15

end NUMINAMATH_CALUDE_franklins_gathering_theorem_l741_74173


namespace NUMINAMATH_CALUDE_unique_modular_equivalence_l741_74161

theorem unique_modular_equivalence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 14525 [ZMOD 16] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_equivalence_l741_74161


namespace NUMINAMATH_CALUDE_carries_trip_l741_74133

theorem carries_trip (day1 : ℕ) (day3 : ℕ) (day4 : ℕ) (charge_distance : ℕ) (charge_count : ℕ) : 
  day1 = 135 → 
  day3 = 159 → 
  day4 = 189 → 
  charge_distance = 106 → 
  charge_count = 7 → 
  ∃ day2 : ℕ, day2 - day1 = 124 ∧ day1 + day2 + day3 + day4 = charge_distance * charge_count :=
by sorry

end NUMINAMATH_CALUDE_carries_trip_l741_74133


namespace NUMINAMATH_CALUDE_train_speed_l741_74121

/-- The speed of a train given its length, time to cross a walking man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) :
  train_length = 400 →
  crossing_time = 23.998 →
  man_speed_kmh = 3 →
  ∃ (train_speed_kmh : ℝ), 
    (train_speed_kmh ≥ 63.004 ∧ train_speed_kmh ≤ 63.006) ∧
    train_speed_kmh * 1000 / 3600 = 
      train_length / crossing_time + man_speed_kmh * 1000 / 3600 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l741_74121


namespace NUMINAMATH_CALUDE_range_of_linear_function_l741_74171

-- Define the function f on the closed interval [0, 1]
def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

-- State the theorem
theorem range_of_linear_function
  (a b : ℝ)
  (h_a_neg : a < 0)
  : Set.range (fun x => f x a b) = Set.Icc (a + b) b := by
  sorry

end NUMINAMATH_CALUDE_range_of_linear_function_l741_74171


namespace NUMINAMATH_CALUDE_problem_1_l741_74186

theorem problem_1 (α : Real) (h : Real.sin α - 2 * Real.cos α = 0) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l741_74186


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l741_74137

noncomputable def f (x : ℝ) : ℝ := x^3 * Real.sin x

theorem derivative_f_at_one :
  deriv f 1 = 3 * Real.sin 1 + Real.cos 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l741_74137


namespace NUMINAMATH_CALUDE_adams_goats_l741_74132

theorem adams_goats (adam andrew ahmed : ℕ) 
  (andrew_eq : andrew = 2 * adam + 5)
  (ahmed_eq : ahmed = andrew - 6)
  (ahmed_count : ahmed = 13) : 
  adam = 7 := by
  sorry

end NUMINAMATH_CALUDE_adams_goats_l741_74132


namespace NUMINAMATH_CALUDE_total_experienced_monthly_earnings_l741_74175

def total_sailors : ℕ := 30
def inexperienced_sailors : ℕ := 8
def group_a_sailors : ℕ := 12
def group_b_sailors : ℕ := 10
def inexperienced_hourly_wage : ℚ := 12
def group_a_wage_multiplier : ℚ := 4/3
def group_b_wage_multiplier : ℚ := 5/4
def group_a_weekly_hours : ℕ := 50
def group_b_weekly_hours : ℕ := 60
def weeks_per_month : ℕ := 4

def group_a_hourly_wage : ℚ := inexperienced_hourly_wage * group_a_wage_multiplier
def group_b_hourly_wage : ℚ := inexperienced_hourly_wage * group_b_wage_multiplier

def group_a_monthly_earnings : ℚ := group_a_hourly_wage * group_a_weekly_hours * weeks_per_month * group_a_sailors
def group_b_monthly_earnings : ℚ := group_b_hourly_wage * group_b_weekly_hours * weeks_per_month * group_b_sailors

theorem total_experienced_monthly_earnings :
  group_a_monthly_earnings + group_b_monthly_earnings = 74400 := by
  sorry

end NUMINAMATH_CALUDE_total_experienced_monthly_earnings_l741_74175


namespace NUMINAMATH_CALUDE_three_digit_one_more_than_multiple_l741_74139

/-- The least common multiple of 2, 3, 5, and 7 -/
def lcm_2357 : ℕ := 210

/-- Checks if a number is a three-digit positive integer -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Checks if a number is one more than a multiple of 2, 3, 5, and 7 -/
def is_one_more_than_multiple (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * lcm_2357 + 1

theorem three_digit_one_more_than_multiple :
  ∀ n : ℕ, is_three_digit n ∧ is_one_more_than_multiple n ↔ n = 211 ∨ n = 421 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_one_more_than_multiple_l741_74139


namespace NUMINAMATH_CALUDE_range_of_a_for_subset_l741_74170

/-- The set A is defined as a circle centered at (2, 1) with radius 1 -/
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 ≤ 1}

/-- The set B is defined as a diamond shape centered at (1, 1) -/
def B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2*|p.1 - 1| + |p.2 - 1| ≤ a}

/-- Theorem stating the range of a for which A is a subset of B -/
theorem range_of_a_for_subset (a : ℝ) : A ⊆ B a ↔ a ≥ 2 + Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_subset_l741_74170


namespace NUMINAMATH_CALUDE_set_membership_implies_m_values_l741_74160

theorem set_membership_implies_m_values (m : ℝ) :
  let A : Set ℝ := {1, m + 2, m^2 + 4}
  5 ∈ A → m = 3 ∨ m = 1 := by
sorry

end NUMINAMATH_CALUDE_set_membership_implies_m_values_l741_74160


namespace NUMINAMATH_CALUDE_exists_nonperiodic_sequence_satisfying_property_l741_74196

/-- A sequence of natural numbers. -/
def Sequence := ℕ → ℕ

/-- A sequence satisfies the given property if for any k, there exists a t such that
    the sequence remains constant when we add multiples of t to k. -/
def SatisfiesProperty (a : Sequence) : Prop :=
  ∀ k, ∃ t, ∀ m, a k = a (k + m * t)

/-- A sequence is periodic if there exists a period T such that
    for all k, a(k) = a(k + T). -/
def IsPeriodic (a : Sequence) : Prop :=
  ∃ T, ∀ k, a k = a (k + T)

/-- There exists a sequence that satisfies the property but is not periodic. -/
theorem exists_nonperiodic_sequence_satisfying_property :
  ∃ a : Sequence, SatisfiesProperty a ∧ ¬IsPeriodic a := by
  sorry

end NUMINAMATH_CALUDE_exists_nonperiodic_sequence_satisfying_property_l741_74196


namespace NUMINAMATH_CALUDE_triangle_inequality_l741_74199

-- Define the points
variable (A B C P A₁ B₁ C₁ : ℝ × ℝ)

-- Define the equilateral triangle ABC
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- Define that P is inside triangle ABC
def is_inside_triangle (P A B C : ℝ × ℝ) : Prop :=
  ∃ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧
  P = (α * A.1 + β * B.1 + γ * C.1, α * A.2 + β * B.2 + γ * C.2)

-- Define that A₁, B₁, C₁ are on the sides of triangle ABC
def on_side (X Y Z : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ Y = (t * X.1 + (1 - t) * Z.1, t * X.2 + (1 - t) * Z.2)

-- Define the theorem
theorem triangle_inequality (A B C P A₁ B₁ C₁ : ℝ × ℝ) 
  (h1 : is_equilateral A B C)
  (h2 : is_inside_triangle P A B C)
  (h3 : on_side A₁ B C)
  (h4 : on_side B₁ C A)
  (h5 : on_side C₁ A B) :
  dist A₁ B₁ * dist B₁ C₁ * dist C₁ A₁ ≥ dist A₁ B * dist B₁ C * dist C₁ A :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l741_74199


namespace NUMINAMATH_CALUDE_speed_ratio_with_head_start_l741_74195

/-- The ratio of speeds in a race where one runner has a head start -/
theorem speed_ratio_with_head_start (vA vB : ℝ) (h : vA > 0 ∧ vB > 0) : 
  (120 / vA = 60 / vB) → vA / vB = 2 := by
  sorry

#check speed_ratio_with_head_start

end NUMINAMATH_CALUDE_speed_ratio_with_head_start_l741_74195


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l741_74176

theorem cubic_root_ratio (a b c d : ℝ) (h : a ≠ 0) : 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3) → 
  c / d = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l741_74176


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l741_74128

/-- Proves that the speed of a boat in still water is 24 km/hr, given the speed of the stream and the boat's downstream travel information. -/
theorem boat_speed_in_still_water :
  let stream_speed : ℝ := 4
  let downstream_distance : ℝ := 140
  let downstream_time : ℝ := 5
  let downstream_speed : ℝ := downstream_distance / downstream_time
  let boat_speed_still_water : ℝ := downstream_speed - stream_speed
  boat_speed_still_water = 24 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l741_74128


namespace NUMINAMATH_CALUDE_equal_ratios_imply_p_equals_13_l741_74169

theorem equal_ratios_imply_p_equals_13 
  (a b c p : ℝ) 
  (h1 : (5 : ℝ) / (a + b) = p / (a + c)) 
  (h2 : p / (a + c) = (8 : ℝ) / (c - b)) : 
  p = 13 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_imply_p_equals_13_l741_74169


namespace NUMINAMATH_CALUDE_paint_usage_l741_74112

/-- Calculates the total amount of paint used by an artist for large and small canvases -/
theorem paint_usage (large_paint : ℕ) (small_paint : ℕ) (large_count : ℕ) (small_count : ℕ) :
  large_paint = 3 →
  small_paint = 2 →
  large_count = 3 →
  small_count = 4 →
  large_paint * large_count + small_paint * small_count = 17 := by
  sorry


end NUMINAMATH_CALUDE_paint_usage_l741_74112


namespace NUMINAMATH_CALUDE_johns_weight_l741_74191

theorem johns_weight (john mark : ℝ) 
  (h1 : john + mark = 240)
  (h2 : john - mark = john / 3) : 
  john = 144 := by
sorry

end NUMINAMATH_CALUDE_johns_weight_l741_74191


namespace NUMINAMATH_CALUDE_molecular_weight_AlCl3_l741_74172

/-- The molecular weight of 4 moles of AlCl3 -/
theorem molecular_weight_AlCl3 (atomic_weight_Al atomic_weight_Cl : ℝ) 
  (h1 : atomic_weight_Al = 26.98)
  (h2 : atomic_weight_Cl = 35.45) : ℝ := by
  sorry

#check molecular_weight_AlCl3

end NUMINAMATH_CALUDE_molecular_weight_AlCl3_l741_74172


namespace NUMINAMATH_CALUDE_expression_value_l741_74115

theorem expression_value (b : ℚ) (h : b = 1/3) : 
  (3 * b⁻¹ - b⁻¹ / 3) / b^2 = 72 := by sorry

end NUMINAMATH_CALUDE_expression_value_l741_74115


namespace NUMINAMATH_CALUDE_xyz_value_l741_74144

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 17)
  (h3 : x^3 + y^3 + z^3 = 27) :
  x * y * z = 32 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l741_74144


namespace NUMINAMATH_CALUDE_larger_number_problem_l741_74142

theorem larger_number_problem (x y : ℝ) 
  (sum : x + y = 40)
  (diff : x - y = 10)
  (prod : x * y = 375)
  (greater : x > y) : x = 25 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l741_74142


namespace NUMINAMATH_CALUDE_course_selection_ways_l741_74117

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem course_selection_ways (type_a type_b total : ℕ) : 
  type_a = 3 → type_b = 4 → total = 3 →
  (choose type_a 2 * choose type_b 1 + choose type_a 1 * choose type_b 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_ways_l741_74117


namespace NUMINAMATH_CALUDE_min_value_rational_function_l741_74108

theorem min_value_rational_function :
  ∀ x : ℝ, (3 * x^2 + 6 * x + 5) / (0.5 * x^2 + x + 1) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_rational_function_l741_74108


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l741_74157

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  abs (l.a * x₀ + l.b * y₀ + l.c) / Real.sqrt (l.a^2 + l.b^2) = c.radius

/-- The main theorem -/
theorem tangent_lines_to_circle 
  (c : Circle) 
  (p : ℝ × ℝ) 
  (h_circle : c.center = (1, 1) ∧ c.radius = 1) 
  (h_point : p = (2, 3)) :
  ∃ (l₁ l₂ : Line),
    isTangent l₁ c ∧ isTangent l₂ c ∧
    (l₁.a = 1 ∧ l₁.b = 0 ∧ l₁.c = -2) ∧
    (l₂.a = 3 ∧ l₂.b = -4 ∧ l₂.c = 6) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l741_74157


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l741_74149

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line l
def line_l (p m : ℝ) (x y : ℝ) : Prop := x = m*y + p/2

-- Define points A and B on the parabola and line
def point_on_parabola_and_line (p m : ℝ) (x y : ℝ) : Prop :=
  parabola p x y ∧ line_l p m x y

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁*x₂ + y₁*y₂ = -3

-- Define the distance between two points
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Define the minimization condition
def is_minimum (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∀ x₁' y₁' x₂' y₂', point_on_parabola_and_line p ((x₁' + x₂')/2) x₁' y₁' →
    point_on_parabola_and_line p ((x₁' + x₂')/2) x₂' y₂' →
    dot_product_condition x₁' y₁' x₂' y₂' →
    (|x₁' - p/2| + 4*|x₂' - p/2| ≥ |x₁ - p/2| + 4*|x₂ - p/2|)

-- The main theorem
theorem parabola_intersection_theorem (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  parabola p x₁ y₁ →
  parabola p x₂ y₂ →
  (∃ m : ℝ, line_l p m x₁ y₁ ∧ line_l p m x₂ y₂) →
  dot_product_condition x₁ y₁ x₂ y₂ →
  is_minimum p x₁ y₁ x₂ y₂ →
  distance x₁ y₁ x₂ y₂ = 9/2 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l741_74149


namespace NUMINAMATH_CALUDE_min_nSn_arithmetic_sequence_l741_74101

def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

theorem min_nSn_arithmetic_sequence (a₁ d : ℤ) :
  (arithmetic_sequence a₁ d 7 = 5) →
  (sum_arithmetic_sequence a₁ d 5 = -55) →
  (∀ n : ℕ, n > 0 → n * (sum_arithmetic_sequence a₁ d n) ≥ -343) ∧
  (∃ n : ℕ, n > 0 ∧ n * (sum_arithmetic_sequence a₁ d n) = -343) :=
sorry

end NUMINAMATH_CALUDE_min_nSn_arithmetic_sequence_l741_74101


namespace NUMINAMATH_CALUDE_distribute_five_among_three_l741_74184

/-- The number of ways to distribute n distinct objects among k distinct groups,
    with each group receiving at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects among 3 distinct groups,
    with each group receiving at least one object, is 150 -/
theorem distribute_five_among_three :
  distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_among_three_l741_74184


namespace NUMINAMATH_CALUDE_three_minus_a_equals_four_l741_74151

theorem three_minus_a_equals_four (a b : ℝ) 
  (eq1 : 3 + a = 4 - b) 
  (eq2 : 4 + b = 7 + a) : 
  3 - a = 4 := by
sorry

end NUMINAMATH_CALUDE_three_minus_a_equals_four_l741_74151


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l741_74102

theorem fractional_equation_solution : 
  ∃ x : ℝ, (3 / (x^2 - x) + 1 = x / (x - 1)) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l741_74102


namespace NUMINAMATH_CALUDE_lily_painting_time_l741_74104

/-- The time it takes to paint a lily -/
def time_for_lily : ℕ := sorry

/-- The time it takes to paint a rose -/
def time_for_rose : ℕ := 7

/-- The time it takes to paint an orchid -/
def time_for_orchid : ℕ := 3

/-- The time it takes to paint a vine -/
def time_for_vine : ℕ := 2

/-- The total time taken to paint all flowers and vines -/
def total_time : ℕ := 213

/-- The number of lilies painted -/
def num_lilies : ℕ := 17

/-- The number of roses painted -/
def num_roses : ℕ := 10

/-- The number of orchids painted -/
def num_orchids : ℕ := 6

/-- The number of vines painted -/
def num_vines : ℕ := 20

theorem lily_painting_time : time_for_lily = 5 := by
  sorry

end NUMINAMATH_CALUDE_lily_painting_time_l741_74104


namespace NUMINAMATH_CALUDE_reverse_digits_when_multiplied_by_nine_l741_74116

theorem reverse_digits_when_multiplied_by_nine : ∃ n : ℕ, 
  (100000 ≤ n ∧ n < 1000000) ∧  -- six-digit number
  (n * 9 = 
    ((n % 10) * 100000 + 
     ((n / 10) % 10) * 10000 + 
     ((n / 100) % 10) * 1000 + 
     ((n / 1000) % 10) * 100 + 
     ((n / 10000) % 10) * 10 + 
     (n / 100000))) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_reverse_digits_when_multiplied_by_nine_l741_74116


namespace NUMINAMATH_CALUDE_candy_bar_sales_l741_74145

theorem candy_bar_sales (members : ℕ) (price : ℚ) (total_earnings : ℚ) 
  (h1 : members = 20)
  (h2 : price = 1/2)
  (h3 : total_earnings = 80) :
  (total_earnings / price) / members = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_sales_l741_74145


namespace NUMINAMATH_CALUDE_total_distance_traveled_l741_74164

/-- Calculate the total distance traveled given cycling, walking, and jogging durations and speeds -/
theorem total_distance_traveled
  (cycling_time : ℚ) (cycling_speed : ℚ)
  (walking_time : ℚ) (walking_speed : ℚ)
  (jogging_time : ℚ) (jogging_speed : ℚ)
  (h1 : cycling_time = 20 / 60)
  (h2 : cycling_speed = 12)
  (h3 : walking_time = 40 / 60)
  (h4 : walking_speed = 3)
  (h5 : jogging_time = 50 / 60)
  (h6 : jogging_speed = 7) :
  let total_distance := cycling_time * cycling_speed + walking_time * walking_speed + jogging_time * jogging_speed
  ∃ ε > 0, |total_distance - 11.8333| < ε :=
sorry

#eval (20/60 : ℚ) * 12 + (40/60 : ℚ) * 3 + (50/60 : ℚ) * 7

end NUMINAMATH_CALUDE_total_distance_traveled_l741_74164


namespace NUMINAMATH_CALUDE_probability_two_even_balls_l741_74152

def total_balls : ℕ := 17
def even_balls : ℕ := 8

theorem probability_two_even_balls :
  (even_balls : ℚ) / total_balls * (even_balls - 1) / (total_balls - 1) = 7 / 34 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_even_balls_l741_74152


namespace NUMINAMATH_CALUDE_triangle_side_relation_l741_74150

/-- Given a triangle ABC with side lengths a, b, c satisfying the equation
    a² - 16b² - c² + 6ab + 10bc = 0, prove that a + c = 2b. -/
theorem triangle_side_relation (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : a^2 - 16*b^2 - c^2 + 6*a*b + 10*b*c = 0) :
  a + c = 2*b :=
sorry

end NUMINAMATH_CALUDE_triangle_side_relation_l741_74150


namespace NUMINAMATH_CALUDE_car_trading_problem_l741_74141

-- Define the profit per car for models A and B (in thousand yuan)
variable (profit_A profit_B : ℚ)

-- Define the number of cars of each model
variable (num_A num_B : ℕ)

-- Define the given conditions
axiom profit_condition_1 : 3 * profit_A + 2 * profit_B = 34
axiom profit_condition_2 : profit_A + 4 * profit_B = 28

-- Define the purchase prices (in thousand yuan)
def price_A : ℚ := 160
def price_B : ℚ := 140

-- Define the total number of cars and budget (in thousand yuan)
def total_cars : ℕ := 30
def max_budget : ℚ := 4400

-- Define the minimum profit (in thousand yuan)
def min_profit : ℚ := 177

-- Theorem statement
theorem car_trading_problem :
  (profit_A = 8 ∧ profit_B = 5) ∧
  ((num_A = 9 ∧ num_B = 21) ∨ (num_A = 10 ∧ num_B = 20)) ∧
  (num_A + num_B = total_cars) ∧
  (num_A * price_A + num_B * price_B ≤ max_budget) ∧
  (num_A * profit_A + num_B * profit_B ≥ min_profit) :=
sorry

end NUMINAMATH_CALUDE_car_trading_problem_l741_74141


namespace NUMINAMATH_CALUDE_jake_peaches_count_l741_74103

-- Define the number of apples and peaches for Steven and Jake
def steven_apples : ℕ := 52
def steven_peaches : ℕ := 13
def jake_apples : ℕ := steven_apples + 84
def jake_peaches : ℕ := steven_peaches - 10

-- Theorem to prove
theorem jake_peaches_count : jake_peaches = 3 := by
  sorry

end NUMINAMATH_CALUDE_jake_peaches_count_l741_74103


namespace NUMINAMATH_CALUDE_work_completion_time_l741_74100

theorem work_completion_time (y_completion_time x_remaining_time : ℕ) 
  (y_worked_days : ℕ) (h1 : y_completion_time = 16) (h2 : y_worked_days = 10) 
  (h3 : x_remaining_time = 9) : 
  (y_completion_time * x_remaining_time) / 
  (y_completion_time - y_worked_days) = 24 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l741_74100


namespace NUMINAMATH_CALUDE_q_range_l741_74183

def q (x : ℝ) : ℝ := (x^2 - 2)^2

theorem q_range : 
  (∀ y : ℝ, (∃ x : ℝ, q x = y) → y ≥ 0) ∧ 
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, q x = y) :=
sorry

end NUMINAMATH_CALUDE_q_range_l741_74183


namespace NUMINAMATH_CALUDE_multiplicative_inverse_480_mod_4799_l741_74198

theorem multiplicative_inverse_480_mod_4799 : 
  (∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a = 40 ∧ b = 399 ∧ c = 401) →
  (∃ (n : ℕ), n < 4799 ∧ (480 * n) % 4799 = 1) ∧
  (480 * 4789) % 4799 = 1 :=
by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_480_mod_4799_l741_74198


namespace NUMINAMATH_CALUDE_spiral_stripe_length_l741_74134

/-- The length of a spiral stripe on a right circular cylinder -/
theorem spiral_stripe_length (base_circumference height : ℝ) (h1 : base_circumference = 18) (h2 : height = 8) :
  let stripe_length := Real.sqrt (height^2 + (2 * base_circumference)^2)
  stripe_length = Real.sqrt 1360 := by
sorry

end NUMINAMATH_CALUDE_spiral_stripe_length_l741_74134


namespace NUMINAMATH_CALUDE_equal_numbers_l741_74154

theorem equal_numbers (x y z : ℕ) 
  (h1 : x ∣ Nat.gcd y z)
  (h2 : y ∣ Nat.gcd x z)
  (h3 : z ∣ Nat.gcd x y)
  (h4 : x ∣ Nat.lcm y z)
  (h5 : y ∣ Nat.lcm x z)
  (h6 : z ∣ Nat.lcm x y) :
  x = y ∧ y = z := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_l741_74154


namespace NUMINAMATH_CALUDE_valid_sequences_of_length_21_l741_74189

/-- Counts valid sequences of 0s and 1s of given length -/
def countValidSequences (n : ℕ) : ℕ :=
  if n ≤ 4 then 0
  else if n = 5 then 1
  else if n = 6 then 2
  else if n = 7 then 2
  else countValidSequences (n - 4) + 2 * countValidSequences (n - 5) + countValidSequences (n - 6)

/-- Theorem stating the number of valid sequences of length 21 -/
theorem valid_sequences_of_length_21 :
  countValidSequences 21 = 114 := by
  sorry

end NUMINAMATH_CALUDE_valid_sequences_of_length_21_l741_74189


namespace NUMINAMATH_CALUDE_white_balls_remain_odd_one_white_ball_left_l741_74126

/-- Represents the state of the bag with black and white balls -/
structure BagState where
  white : Nat
  black : Nat

/-- The process of drawing two balls and applying the rules -/
def drawBalls (state : BagState) : BagState :=
  sorry

/-- Predicate to check if the process has ended (0 or 1 ball left) -/
def processEnded (state : BagState) : Prop :=
  state.white + state.black ≤ 1

/-- Theorem stating that the number of white balls remains odd throughout the process -/
theorem white_balls_remain_odd (initial : BagState) (final : BagState) 
    (h_initial : initial.white = 2007 ∧ initial.black = 2007)
    (h_process : final = drawBalls initial ∨ (∃ intermediate, final = drawBalls intermediate ∧ ¬processEnded intermediate)) :
  Odd final.white :=
  sorry

/-- Main theorem proving that one white ball is left at the end -/
theorem one_white_ball_left (initial : BagState) (final : BagState)
    (h_initial : initial.white = 2007 ∧ initial.black = 2007)
    (h_process : final = drawBalls initial ∨ (∃ intermediate, final = drawBalls intermediate ∧ ¬processEnded intermediate))
    (h_ended : processEnded final) :
  final.white = 1 ∧ final.black = 0 :=
  sorry

end NUMINAMATH_CALUDE_white_balls_remain_odd_one_white_ball_left_l741_74126


namespace NUMINAMATH_CALUDE_denis_neighbors_l741_74185

-- Define the students
inductive Student : Type
| Anya : Student
| Borya : Student
| Vera : Student
| Gena : Student
| Denis : Student

-- Define the line as a function from position (1 to 5) to Student
def Line : Type := Fin 5 → Student

-- Define what it means for two students to be adjacent
def adjacent (s1 s2 : Student) (line : Line) : Prop :=
  ∃ i : Fin 4, (line i = s1 ∧ line (i.succ) = s2) ∨ (line i = s2 ∧ line (i.succ) = s1)

-- State the theorem
theorem denis_neighbors (line : Line) : 
  (line 0 = Student.Borya) →  -- Borya is at the beginning
  (adjacent Student.Vera Student.Anya line ∧ ¬adjacent Student.Vera Student.Gena line) →  -- Vera next to Anya but not Gena
  (¬adjacent Student.Anya Student.Borya line ∧ ¬adjacent Student.Anya Student.Gena line ∧ ¬adjacent Student.Borya Student.Gena line) →  -- Anya, Borya, Gena not adjacent
  (adjacent Student.Denis Student.Anya line ∧ adjacent Student.Denis Student.Gena line) :=  -- Denis is next to Anya and Gena
by sorry

end NUMINAMATH_CALUDE_denis_neighbors_l741_74185


namespace NUMINAMATH_CALUDE_solve_for_A_l741_74179

theorem solve_for_A : ∃ A : ℝ, (10 - A = 6) ∧ (A = 4) := by sorry

end NUMINAMATH_CALUDE_solve_for_A_l741_74179


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l741_74122

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes
    with at least one box remaining empty -/
def distributeWithEmptyBox (n k : ℕ) : ℕ :=
  if n < k then distribute n k else distribute n k

theorem distribute_five_balls_four_boxes :
  distributeWithEmptyBox 5 4 = 1024 := by sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l741_74122
