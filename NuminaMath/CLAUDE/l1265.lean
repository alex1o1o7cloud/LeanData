import Mathlib

namespace arithmetic_sequence_middle_term_l1265_126516

/-- Given an arithmetic sequence with first term 3^2 and third term 3^4, 
    the middle term y is equal to 45. -/
theorem arithmetic_sequence_middle_term : 
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
    a 0 = 3^2 →                                       -- first term
    a 2 = 3^4 →                                       -- third term
    a 1 = 45 :=                                       -- middle term (y)
by
  sorry

end arithmetic_sequence_middle_term_l1265_126516


namespace quadratic_inequality_solutions_l1265_126539

def has_three_integer_solutions (b : ℤ) : Prop :=
  ∃ x y z : ℤ, x < y ∧ y < z ∧
    (x^2 + b*x - 2 ≤ 0) ∧
    (y^2 + b*y - 2 ≤ 0) ∧
    (z^2 + b*z - 2 ≤ 0) ∧
    ∀ w : ℤ, (w^2 + b*w - 2 ≤ 0) → (w = x ∨ w = y ∨ w = z)

theorem quadratic_inequality_solutions :
  ∃! (s : Finset ℤ), s.card = 2 ∧ ∀ b : ℤ, b ∈ s ↔ has_three_integer_solutions b :=
sorry

end quadratic_inequality_solutions_l1265_126539


namespace inequality_range_l1265_126581

theorem inequality_range (a : ℝ) : 
  (∀ (x θ : ℝ), θ ∈ Set.Icc 0 (Real.pi / 2) → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≤ Real.sqrt 6 ∨ a ≥ 7/2) := by
  sorry

end inequality_range_l1265_126581


namespace quadratic_inequality_l1265_126589

/-- A quadratic function with a positive leading coefficient and symmetry about x = 2 -/
def symmetric_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ (∀ x, f x = a * x^2 + b * x + c) ∧ (∀ x, f x = f (4 - x))

theorem quadratic_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h1 : symmetric_quadratic f) 
  (h2 : f (2 - a^2) < f (1 + a - a^2)) : 
  a < 1 := by sorry

end quadratic_inequality_l1265_126589


namespace extreme_value_condition_l1265_126546

/-- A function f has an extreme value at x₀ -/
def has_extreme_value (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x₀ ≤ f x ∨ f x ≤ f x₀

theorem extreme_value_condition (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  ¬(∀ x₀ : ℝ, has_extreme_value f x₀ ↔ f x₀ = 0) :=
by sorry

end extreme_value_condition_l1265_126546


namespace simplify_fraction_product_l1265_126573

theorem simplify_fraction_product : 8 * (15 / 9) * (-45 / 40) = -1 := by
  sorry

end simplify_fraction_product_l1265_126573


namespace cream_needed_proof_l1265_126503

/-- The amount of additional cream needed when given a total required amount and an available amount -/
def additional_cream_needed (total_required : ℕ) (available : ℕ) : ℕ :=
  total_required - available

/-- Theorem stating that given 300 lbs total required and 149 lbs available, 151 lbs additional cream is needed -/
theorem cream_needed_proof :
  additional_cream_needed 300 149 = 151 := by
  sorry

end cream_needed_proof_l1265_126503


namespace audrey_twice_heracles_age_l1265_126542

def age_difference : ℕ := 7
def heracles_current_age : ℕ := 10

theorem audrey_twice_heracles_age (years : ℕ) : 
  (heracles_current_age + age_difference + years = 2 * heracles_current_age) → years = 3 := by
  sorry

end audrey_twice_heracles_age_l1265_126542


namespace polar_to_cartesian_circle_l1265_126523

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = 5 * Real.sin θ

-- Define the Cartesian equation of a circle
def circle_equation (x y : ℝ) (h k r : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_to_cartesian_circle :
  ∀ x y : ℝ, (∃ ρ θ : ℝ, polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ∃ h k r : ℝ, circle_equation x y h k r := by sorry

end polar_to_cartesian_circle_l1265_126523


namespace linear_system_ratio_l1265_126597

/-- Given a system of linear equations with a nontrivial solution, prove that xz/y^2 = 26/9 -/
theorem linear_system_ratio (x y z k : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + k * y + 4 * z = 0 →
  4 * x + k * y - 3 * z = 0 →
  x + 3 * y - 2 * z = 0 →
  x * z / (y ^ 2) = 26 / 9 := by
  sorry

end linear_system_ratio_l1265_126597


namespace jebbs_take_home_pay_l1265_126551

/-- Calculates the take-home pay given a gross salary and various tax rates and deductions. -/
def calculateTakeHomePay (grossSalary : ℚ) : ℚ :=
  let federalTaxRate1 := 0.10
  let federalTaxRate2 := 0.15
  let federalTaxRate3 := 0.25
  let federalTaxThreshold1 := 2500
  let federalTaxThreshold2 := 5000
  let stateTaxRate1 := 0.05
  let stateTaxRate2 := 0.07
  let stateTaxThreshold := 3000
  let socialSecurityTaxRate := 0.062
  let socialSecurityTaxCap := 4800
  let medicareTaxRate := 0.0145
  let healthInsurance := 300
  let retirementContributionRate := 0.07

  let federalTax := 
    federalTaxRate1 * federalTaxThreshold1 +
    federalTaxRate2 * (federalTaxThreshold2 - federalTaxThreshold1) +
    federalTaxRate3 * (grossSalary - federalTaxThreshold2)

  let stateTax := 
    stateTaxRate1 * stateTaxThreshold +
    stateTaxRate2 * (grossSalary - stateTaxThreshold)

  let socialSecurityTax := socialSecurityTaxRate * (min grossSalary socialSecurityTaxCap)

  let medicareTax := medicareTaxRate * grossSalary

  let retirementContribution := retirementContributionRate * grossSalary

  let totalDeductions := 
    federalTax + stateTax + socialSecurityTax + medicareTax + healthInsurance + retirementContribution

  grossSalary - totalDeductions

/-- Theorem stating that Jebb's take-home pay is $3,958.15 given his gross salary and deductions. -/
theorem jebbs_take_home_pay :
  calculateTakeHomePay 6500 = 3958.15 := by
  sorry


end jebbs_take_home_pay_l1265_126551


namespace new_class_mean_l1265_126588

theorem new_class_mean (total_students : ℕ) (first_group : ℕ) (second_group : ℕ) 
  (first_mean : ℚ) (second_mean : ℚ) :
  total_students = first_group + second_group →
  first_group = 45 →
  second_group = 5 →
  first_mean = 80 / 100 →
  second_mean = 90 / 100 →
  (first_group * first_mean + second_group * second_mean) / total_students = 81 / 100 := by
sorry

end new_class_mean_l1265_126588


namespace four_statements_incorrect_l1265_126558

/-- The alternating sum from 1 to 2002 -/
def alternating_sum : ℕ → ℤ
  | 0 => 0
  | n + 1 => if n % 2 = 0 then alternating_sum n + (n + 1) else alternating_sum n - (n + 1)

/-- The sum of n consecutive natural numbers starting from k -/
def consec_sum (n k : ℕ) : ℕ := n * (2 * k + n - 1) / 2

theorem four_statements_incorrect : 
  (¬ Even (alternating_sum 2002)) ∧ 
  (∃ (a b c : ℤ), Odd a ∧ Odd b ∧ Odd c ∧ (a * b) * (c - b) ≠ a) ∧
  (¬ Even (consec_sum 2002 1)) ∧
  (¬ ∃ (a b : ℤ), (a + b) * (a - b) = 2002) :=
by sorry

end four_statements_incorrect_l1265_126558


namespace unique_solution_is_four_l1265_126522

-- Define the equation
def equation (s x : ℝ) : Prop :=
  1 / (3 * x) = (s - x) / 9

-- State the theorem
theorem unique_solution_is_four :
  ∃! s : ℝ, (∃! x : ℝ, equation s x) ∧ s = 4 := by sorry

end unique_solution_is_four_l1265_126522


namespace stock_face_value_l1265_126504

/-- Calculates the face value of a stock given the discount rate, brokerage rate, and final cost price. -/
def calculate_face_value (discount_rate : ℚ) (brokerage_rate : ℚ) (final_cost : ℚ) : ℚ :=
  final_cost / ((1 - discount_rate) * (1 + brokerage_rate))

/-- Theorem stating that for a stock with 2% discount, 1/5% brokerage, and Rs 98.2 final cost, the face value is Rs 100. -/
theorem stock_face_value : 
  let discount_rate : ℚ := 2 / 100
  let brokerage_rate : ℚ := 1 / 500
  let final_cost : ℚ := 982 / 10
  calculate_face_value discount_rate brokerage_rate final_cost = 100 := by
  sorry

end stock_face_value_l1265_126504


namespace employee_count_l1265_126528

/-- Proves the number of employees given salary information -/
theorem employee_count 
  (avg_salary : ℝ) 
  (salary_increase : ℝ) 
  (manager_salary : ℝ) 
  (h1 : avg_salary = 1700)
  (h2 : salary_increase = 100)
  (h3 : manager_salary = 3800) :
  ∃ (E : ℕ), 
    (E : ℝ) * (avg_salary + salary_increase) = E * avg_salary + manager_salary ∧ 
    E = 20 :=
by sorry

end employee_count_l1265_126528


namespace symmetric_point_coordinates_l1265_126518

/-- A point in the second quadrant with given absolute values for its coordinates -/
structure SecondQuadrantPoint where
  x : ℝ
  y : ℝ
  second_quadrant : x < 0 ∧ y > 0
  abs_x : |x| = 2
  abs_y : |y| = 3

/-- The symmetric point with respect to the origin -/
def symmetric_point (p : SecondQuadrantPoint) : ℝ × ℝ := (-p.x, -p.y)

/-- Theorem stating that the symmetric point has coordinates (2, -3) -/
theorem symmetric_point_coordinates (p : SecondQuadrantPoint) : 
  symmetric_point p = (2, -3) := by sorry

end symmetric_point_coordinates_l1265_126518


namespace correct_expansion_l1265_126514

theorem correct_expansion (a b : ℝ) : (a - b) * (-a - b) = -a^2 + b^2 := by
  -- Definitions based on the given conditions (equations A, B, and C)
  have h1 : (a + b) * (a - b) = a^2 - b^2 := by sorry
  have h2 : (a + b) * (-a - b) = -(a + b)^2 := by sorry
  have h3 : (a - b) * (-a + b) = -(a - b)^2 := by sorry

  -- Proof of the correct expansion
  sorry

end correct_expansion_l1265_126514


namespace count_base7_with_456_l1265_126510

/-- Represents a positive integer in base 7 --/
def Base7Int : Type := ℕ+

/-- Checks if a Base7Int contains the digits 4, 5, or 6 --/
def containsDigit456 (n : Base7Int) : Prop := sorry

/-- The set of the smallest 2401 positive integers in base 7 --/
def smallestBase7Ints : Set Base7Int := {n | n.val ≤ 2401}

/-- The count of numbers in smallestBase7Ints that contain 4, 5, or 6 --/
def countWith456 : ℕ := sorry

theorem count_base7_with_456 : countWith456 = 2146 := by sorry

end count_base7_with_456_l1265_126510


namespace Z_in_third_quadrant_l1265_126580

-- Define the complex number Z
def Z : ℂ := -1 + (1 - Complex.I)^2

-- Theorem stating that Z is in the third quadrant
theorem Z_in_third_quadrant :
  Real.sign (Z.re) = -1 ∧ Real.sign (Z.im) = -1 :=
sorry


end Z_in_third_quadrant_l1265_126580


namespace smallest_three_digit_multiple_of_17_l1265_126500

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n → n ≥ 102 :=
by sorry

end smallest_three_digit_multiple_of_17_l1265_126500


namespace gcd_50400_37800_l1265_126583

theorem gcd_50400_37800 : Nat.gcd 50400 37800 = 12600 := by
  sorry

end gcd_50400_37800_l1265_126583


namespace isosceles_probability_2020gon_l1265_126533

/-- The number of vertices in the regular polygon -/
def n : ℕ := 2020

/-- The probability of forming an isosceles triangle by randomly selecting
    three distinct vertices from a regular n-gon -/
def isosceles_probability (n : ℕ) : ℚ :=
  (n * ((n - 2) / 2)) / Nat.choose n 3

/-- Theorem stating that the probability of forming an isosceles triangle
    by randomly selecting three distinct vertices from a regular 2020-gon
    is 1/673 -/
theorem isosceles_probability_2020gon :
  isosceles_probability n = 1 / 673 := by
  sorry

end isosceles_probability_2020gon_l1265_126533


namespace cans_display_rows_l1265_126584

def triangular_display (n : ℕ) : ℕ := (3 * n * (n + 1)) / 2

theorem cans_display_rows :
  ∃ (n : ℕ), triangular_display n = 225 ∧ n = 11 := by
sorry

end cans_display_rows_l1265_126584


namespace ground_mince_calculation_l1265_126502

/-- The total amount of ground mince used for lasagnas and cottage pies -/
def total_ground_mince (num_lasagnas : ℕ) (mince_per_lasagna : ℕ) 
                       (num_cottage_pies : ℕ) (mince_per_cottage_pie : ℕ) : ℕ :=
  num_lasagnas * mince_per_lasagna + num_cottage_pies * mince_per_cottage_pie

/-- Theorem stating the total amount of ground mince used -/
theorem ground_mince_calculation :
  total_ground_mince 100 2 100 3 = 500 := by
  sorry

end ground_mince_calculation_l1265_126502


namespace marbles_distribution_l1265_126521

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 35 →
  num_boys = 5 →
  marbles_per_boy = total_marbles / num_boys →
  marbles_per_boy = 7 := by
  sorry

end marbles_distribution_l1265_126521


namespace min_value_phi_l1265_126557

/-- Given real numbers a and b satisfying a^2 + b^2 - 4b + 3 = 0,
    and a function f(x) = a·sin(2x) + b·cos(2x) + 1 with maximum value φ(a,b),
    prove that the minimum value of φ(a,b) is 2. -/
theorem min_value_phi (a b : ℝ) (h : a^2 + b^2 - 4*b + 3 = 0) : 
  let f := fun (x : ℝ) ↦ a * Real.sin (2*x) + b * Real.cos (2*x) + 1
  let φ := fun (a b : ℝ) ↦ Real.sqrt (a^2 + b^2) + 1
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ φ a b ∧ 2 ≤ φ a b :=
by sorry

end min_value_phi_l1265_126557


namespace sum_of_squares_l1265_126565

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := by sorry

end sum_of_squares_l1265_126565


namespace sum_of_roots_even_l1265_126560

theorem sum_of_roots_even (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (h_distinct : ∃ (x y : ℤ), x ≠ y ∧ x^2 - 2*p*x + p*q = 0 ∧ y^2 - 2*p*y + p*q = 0) :
  ∃ (k : ℤ), 2 * p = 2 * k := by
  sorry

end sum_of_roots_even_l1265_126560


namespace alpha_sin_beta_lt_beta_sin_alpha_l1265_126513

theorem alpha_sin_beta_lt_beta_sin_alpha (α β : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) : 
  α * Real.sin β < β * Real.sin α := by
  sorry

end alpha_sin_beta_lt_beta_sin_alpha_l1265_126513


namespace alley_width_equals_ladder_height_l1265_126564

/-- Proof that the width of an alley equals the height of a ladder against one wall 
    when it forms specific angles with both walls. -/
theorem alley_width_equals_ladder_height 
  (l : ℝ) -- length of the ladder
  (x y : ℝ) -- heights on the walls
  (h_x_pos : x > 0)
  (h_y_pos : y > 0)
  (h_angle_Q : x / w = Real.sqrt 3) -- tan 60° = √3
  (h_angle_R : y / w = 1) -- tan 45° = 1
  : w = y :=
sorry

end alley_width_equals_ladder_height_l1265_126564


namespace construct_line_segment_l1265_126592

/-- A straight edge tool -/
structure StraightEdge where
  length : ℝ

/-- A right-angled triangle tool -/
structure RightTriangle where
  hypotenuse : ℝ

/-- A construction using given tools -/
structure Construction where
  straightEdge : StraightEdge
  rightTriangle : RightTriangle

/-- Theorem stating that a line segment of 37 cm can be constructed
    with a 20 cm straight edge and a right triangle with 15 cm hypotenuse -/
theorem construct_line_segment
  (c : Construction)
  (h1 : c.straightEdge.length = 20)
  (h2 : c.rightTriangle.hypotenuse = 15) :
  ∃ (segment_length : ℝ), segment_length = 37 ∧ 
  (∃ (constructed_segment : ℝ → ℝ → Prop), 
    constructed_segment 0 segment_length) :=
sorry

end construct_line_segment_l1265_126592


namespace ratio_problem_l1265_126563

theorem ratio_problem (first_part : ℝ) (ratio_percent : ℝ) (second_part : ℝ) :
  first_part = 5 →
  ratio_percent = 25 →
  first_part / (first_part + second_part) = ratio_percent / 100 →
  second_part = 15 := by
  sorry

end ratio_problem_l1265_126563


namespace right_triangle_area_leg_sum_l1265_126508

theorem right_triangle_area_leg_sum (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → 
  a^2 + b^2 = c^2 → 
  (a * b) / 2 + a = 75 ∨ (a * b) / 2 + b = 75 →
  ((a = 6 ∧ b = 23 ∧ c = 25) ∨ (a = 23 ∧ b = 6 ∧ c = 25) ∨
   (a = 15 ∧ b = 8 ∧ c = 17) ∨ (a = 8 ∧ b = 15 ∧ c = 17)) :=
by sorry

end right_triangle_area_leg_sum_l1265_126508


namespace relay_race_fifth_runner_l1265_126526

def relay_race (t1 t2 t3 t4 t5 : ℝ) : Prop :=
  t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧ t4 > 0 ∧ t5 > 0 ∧
  (t1/2 + t2 + t3 + t4 + t5) / (t1 + t2 + t3 + t4 + t5) = 0.95 ∧
  (t1 + t2/2 + t3 + t4 + t5) / (t1 + t2 + t3 + t4 + t5) = 0.90 ∧
  (t1 + t2 + t3/2 + t4 + t5) / (t1 + t2 + t3 + t4 + t5) = 0.88 ∧
  (t1 + t2 + t3 + t4/2 + t5) / (t1 + t2 + t3 + t4 + t5) = 0.85

theorem relay_race_fifth_runner (t1 t2 t3 t4 t5 : ℝ) :
  relay_race t1 t2 t3 t4 t5 →
  (t1 + t2 + t3 + t4 + t5/2) / (t1 + t2 + t3 + t4 + t5) = 0.92 :=
by sorry

end relay_race_fifth_runner_l1265_126526


namespace multiply_to_325027405_l1265_126586

theorem multiply_to_325027405 (m : ℕ) : m * 32519 = 325027405 → m = 9995 := by
  sorry

end multiply_to_325027405_l1265_126586


namespace jimmy_passing_points_l1265_126501

/-- The minimum number of points required to pass the class -/
def passingScore : ℕ := 50

/-- The number of exams Jimmy took -/
def numExams : ℕ := 3

/-- The number of points Jimmy earned per exam -/
def pointsPerExam : ℕ := 20

/-- The number of points Jimmy lost for bad behavior -/
def pointsLost : ℕ := 5

/-- The maximum number of additional points Jimmy can lose while still passing -/
def maxAdditionalPointsLost : ℕ := 5

theorem jimmy_passing_points : 
  numExams * pointsPerExam - pointsLost - maxAdditionalPointsLost ≥ passingScore := by
  sorry

end jimmy_passing_points_l1265_126501


namespace rectangular_field_fencing_costs_l1265_126540

/-- Given a rectangular field with sides in the ratio of 3:4 and an area of 8112 sq.m,
    prove the perimeter and fencing costs for different materials. -/
theorem rectangular_field_fencing_costs 
  (ratio : ℚ) 
  (area : ℝ) 
  (wrought_iron_cost : ℝ) 
  (wooden_cost : ℝ) 
  (chain_link_cost : ℝ) :
  ratio = 3 / 4 →
  area = 8112 →
  wrought_iron_cost = 45 →
  wooden_cost = 35 →
  chain_link_cost = 25 →
  ∃ (perimeter : ℝ) 
    (wrought_iron_total : ℝ) 
    (wooden_total : ℝ) 
    (chain_link_total : ℝ),
    perimeter = 364 ∧
    wrought_iron_total = 16380 ∧
    wooden_total = 12740 ∧
    chain_link_total = 9100 :=
by sorry

end rectangular_field_fencing_costs_l1265_126540


namespace multiple_properties_l1265_126571

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ (∃ p : ℤ, a - b = 4 * p) := by
  sorry

end multiple_properties_l1265_126571


namespace quadratic_real_roots_alpha_range_l1265_126567

theorem quadratic_real_roots_alpha_range :
  ∀ α : ℝ, 
  (∃ x : ℝ, x^2 - 2*x + α = 0) →
  α ≤ 1 :=
by sorry

end quadratic_real_roots_alpha_range_l1265_126567


namespace rays_dog_walks_63_blocks_l1265_126517

/-- Represents the distance of a single walk in blocks -/
structure Walk where
  to_destination : ℕ
  to_second_place : ℕ
  back_home : ℕ

/-- Calculates the total distance of a walk -/
def Walk.total (w : Walk) : ℕ := w.to_destination + w.to_second_place + w.back_home

/-- Represents Ray's daily dog walking routine -/
structure DailyWalk where
  morning : Walk
  afternoon : Walk
  evening : Walk

/-- Calculates the total distance of all walks in a day -/
def DailyWalk.total_distance (d : DailyWalk) : ℕ :=
  d.morning.total + d.afternoon.total + d.evening.total

/-- Ray's actual daily walk routine -/
def rays_routine : DailyWalk := {
  morning := { to_destination := 4, to_second_place := 7, back_home := 11 }
  afternoon := { to_destination := 3, to_second_place := 5, back_home := 8 }
  evening := { to_destination := 6, to_second_place := 9, back_home := 10 }
}

/-- Theorem stating that Ray's dog walks 63 blocks each day -/
theorem rays_dog_walks_63_blocks : DailyWalk.total_distance rays_routine = 63 := by
  sorry

end rays_dog_walks_63_blocks_l1265_126517


namespace solution_set_characterization_l1265_126512

/-- A function that is even and monotonically increasing on (0,+∞) -/
def f (a b : ℝ) (x : ℝ) : ℝ := (x - 2) * (a * x + b)

/-- The property of f being an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The property of f being monotonically increasing on (0,+∞) -/
def is_increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

/-- The solution set for f(2-x) > 0 -/
def solution_set (f : ℝ → ℝ) : Set ℝ := {x | f (2 - x) > 0}

/-- The theorem stating the solution set for f(2-x) > 0 -/
theorem solution_set_characterization {a b : ℝ} (h_even : is_even (f a b))
    (h_incr : is_increasing_on_positive (f a b)) :
    solution_set (f a b) = {x | x < 0 ∨ x > 4} := by
  sorry

end solution_set_characterization_l1265_126512


namespace rainfall_difference_l1265_126593

/-- Rainfall data for Thomas's science project in May --/
def rainfall_problem (day1 day2 day3 : ℝ) : Prop :=
  let normal_average := 140
  let this_year_total := normal_average - 58
  day1 = 26 ∧
  day2 = 34 ∧
  day3 < day2 ∧
  day1 + day2 + day3 = this_year_total

/-- The difference between the second and third day's rainfall is 12 cm --/
theorem rainfall_difference (day1 day2 day3 : ℝ) 
  (h : rainfall_problem day1 day2 day3) : day2 - day3 = 12 := by
  sorry


end rainfall_difference_l1265_126593


namespace island_puzzle_l1265_126595

-- Define the types of inhabitants
inductive Inhabitant
| Liar
| TruthTeller

-- Define the structure of an answer
structure Answer :=
  (liars : ℕ)
  (truthTellers : ℕ)

-- Define the function that represents how an inhabitant answers
def answer (t : Inhabitant) (actualLiars actualTruthTellers : ℕ) : Answer :=
  match t with
  | Inhabitant.Liar => 
      let liars := if actualLiars % 2 = 0 then actualLiars + 2 else actualLiars - 2
      let truthTellers := if actualTruthTellers % 2 = 0 then actualTruthTellers + 2 else actualTruthTellers - 2
      ⟨liars, truthTellers⟩
  | Inhabitant.TruthTeller => ⟨actualLiars, actualTruthTellers⟩

-- Define the theorem
theorem island_puzzle :
  ∃ (totalLiars totalTruthTellers : ℕ) 
    (first second : Inhabitant),
    totalLiars + totalTruthTellers > 0 ∧
    answer first (totalLiars - 1) (totalTruthTellers) = ⟨1001, 1002⟩ ∧
    answer second (totalLiars - 1) (totalTruthTellers) = ⟨1000, 999⟩ ∧
    totalLiars = 1000 ∧
    totalTruthTellers = 1000 ∧
    first = Inhabitant.Liar ∧
    second = Inhabitant.TruthTeller :=
  sorry


end island_puzzle_l1265_126595


namespace pizza_toppings_l1265_126596

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 24)
  (h2 : pepperoni_slices = 14)
  (h3 : mushroom_slices = 16)
  (h4 : ∀ s, s ≤ total_slices → (s ≤ pepperoni_slices ∨ s ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧ 
    both_toppings = 6 :=
by sorry

end pizza_toppings_l1265_126596


namespace length_of_ae_l1265_126545

/-- Given 5 consecutive points on a straight line, prove the length of ae -/
theorem length_of_ae (a b c d e : ℝ) : 
  (b - a) = 5 →
  (c - a) = 11 →
  (c - b) = 2 * (d - c) →
  (e - d) = 4 →
  (e - a) = 18 := by
  sorry

end length_of_ae_l1265_126545


namespace jason_retirement_age_l1265_126552

def military_career (join_age time_to_chief : ℕ) : Prop :=
  let time_to_senior_chief : ℕ := time_to_chief + (time_to_chief / 4)
  let time_to_master_chief : ℕ := time_to_senior_chief - (time_to_senior_chief / 10)
  let time_to_command_master_chief : ℕ := time_to_master_chief + (time_to_master_chief / 2)
  let additional_time : ℕ := 5
  let total_service_time : ℕ := time_to_chief + time_to_senior_chief + time_to_master_chief + 
                                 time_to_command_master_chief + additional_time
  join_age + total_service_time = 63

theorem jason_retirement_age : 
  military_career 18 8 := by sorry

end jason_retirement_age_l1265_126552


namespace no_quadratic_composition_with_given_zeros_l1265_126585

theorem no_quadratic_composition_with_given_zeros :
  ¬∃ (P Q : ℝ → ℝ),
    (∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) ∧
    (∃ d e f : ℝ, ∀ x, Q x = d * x^2 + e * x + f) ∧
    (∀ x, (P ∘ Q) x = 0 ↔ x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 7) :=
by sorry

end no_quadratic_composition_with_given_zeros_l1265_126585


namespace third_circle_radius_l1265_126594

theorem third_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 19) (h₂ : r₂ = 29) :
  (r₂^2 - r₁^2) * π = π * r₃^2 → r₃ = 4 * Real.sqrt 30 := by
  sorry

end third_circle_radius_l1265_126594


namespace wage_before_raise_l1265_126553

/-- Given a 33.33% increase from x results in $40, prove that x equals $30. -/
theorem wage_before_raise (x : ℝ) : x * (1 + 33.33 / 100) = 40 → x = 30 := by
  sorry

end wage_before_raise_l1265_126553


namespace proposition_d_true_others_false_l1265_126534

theorem proposition_d_true_others_false :
  (∃ x : ℝ, 3 * x^2 - 4 = 6 * x) ∧
  ¬(∀ x : ℝ, (x - Real.sqrt 2)^2 > 0) ∧
  ¬(∀ x : ℚ, x^2 > 0) ∧
  ¬(∃ x : ℤ, 3 * x = 128) :=
by sorry

end proposition_d_true_others_false_l1265_126534


namespace rectangle_area_modification_l1265_126505

/-- Given a rectangle with initial dimensions 5 × 7 inches, if shortening one side by 2 inches
    results in an area of 21 square inches, then doubling the length of the other side
    will result in an area of 70 square inches. -/
theorem rectangle_area_modification (length width : ℝ) : 
  length = 5 ∧ width = 7 ∧ 
  ((length - 2) * width = 21 ∨ length * (width - 2) = 21) →
  length * (2 * width) = 70 :=
sorry

end rectangle_area_modification_l1265_126505


namespace fourth_term_is_twenty_l1265_126530

def sequence_term (n : ℕ) : ℕ := n + 2^n

theorem fourth_term_is_twenty : sequence_term 4 = 20 := by
  sorry

end fourth_term_is_twenty_l1265_126530


namespace triangle_side_b_value_l1265_126574

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area is 2√3, B = π/3, and a² + c² = 3ac, then b = 4. -/
theorem triangle_side_b_value (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = 2 * Real.sqrt 3) →  -- Area condition
  (B = π/3) →                                     -- Angle B condition
  (a^2 + c^2 = 3*a*c) →                           -- Relation between a, c
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →          -- Law of cosines
  (b = 4) :=
by sorry

end triangle_side_b_value_l1265_126574


namespace min_value_of_z_l1265_126537

theorem min_value_of_z (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 12) :
  ∃ (z_min : ℝ), z_min = -5 ∧ ∀ (z : ℝ), z = 2 * x + Real.sqrt 3 * y → z ≥ z_min :=
by sorry

end min_value_of_z_l1265_126537


namespace work_completion_time_l1265_126525

/-- 
Given:
- a and b complete a work in 9 days
- a and b together can do the work in 6 days

Prove: a alone can complete the work in 18 days
-/
theorem work_completion_time (a b : ℝ) (h1 : a + b = 1 / 9) (h2 : a + b = 1 / 6) : a = 1 / 18 := by
  sorry

end work_completion_time_l1265_126525


namespace probability_not_green_l1265_126520

def total_balls : ℕ := 6 + 3 + 4 + 5
def non_green_balls : ℕ := 6 + 3 + 4

theorem probability_not_green (red_balls : ℕ) (yellow_balls : ℕ) (black_balls : ℕ) (green_balls : ℕ)
  (h_red : red_balls = 6)
  (h_yellow : yellow_balls = 3)
  (h_black : black_balls = 4)
  (h_green : green_balls = 5) :
  (red_balls + yellow_balls + black_balls : ℚ) / (red_balls + yellow_balls + black_balls + green_balls) = 13 / 18 :=
by sorry

end probability_not_green_l1265_126520


namespace differential_savings_proof_l1265_126524

def annual_income : ℕ := 45000
def retirement_contribution : ℕ := 4000
def mortgage_interest : ℕ := 5000
def charitable_donations : ℕ := 2000
def previous_tax_rate : ℚ := 40 / 100

def taxable_income : ℕ := annual_income - retirement_contribution - mortgage_interest - charitable_donations

def tax_bracket_1 : ℕ := 10000
def tax_bracket_2 : ℕ := 25000
def tax_bracket_3 : ℕ := 50000

def tax_rate_1 : ℚ := 0 / 100
def tax_rate_2 : ℚ := 10 / 100
def tax_rate_3 : ℚ := 25 / 100
def tax_rate_4 : ℚ := 35 / 100

def new_tax (income : ℕ) : ℚ :=
  if income ≤ tax_bracket_1 then
    income * tax_rate_1
  else if income ≤ tax_bracket_2 then
    tax_bracket_1 * tax_rate_1 + (income - tax_bracket_1) * tax_rate_2
  else if income ≤ tax_bracket_3 then
    tax_bracket_1 * tax_rate_1 + (tax_bracket_2 - tax_bracket_1) * tax_rate_2 + (income - tax_bracket_2) * tax_rate_3
  else
    tax_bracket_1 * tax_rate_1 + (tax_bracket_2 - tax_bracket_1) * tax_rate_2 + (tax_bracket_3 - tax_bracket_2) * tax_rate_3 + (income - tax_bracket_3) * tax_rate_4

theorem differential_savings_proof :
  (annual_income * previous_tax_rate - new_tax taxable_income) = 14250 := by
  sorry

end differential_savings_proof_l1265_126524


namespace cannonball_max_height_l1265_126591

/-- The height function of the cannonball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the cannonball -/
def max_height : ℝ := 161

/-- Theorem stating that the maximum height reached by the cannonball is 161 meters -/
theorem cannonball_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
sorry

end cannonball_max_height_l1265_126591


namespace probability_at_least_6_consecutive_heads_l1265_126538

def coin_flip_sequence := Fin 9 → Bool

def has_at_least_6_consecutive_heads (s : coin_flip_sequence) : Prop :=
  ∃ i, i + 5 < 9 ∧ (∀ j, i ≤ j ∧ j ≤ i + 5 → s j = true)

def total_sequences : ℕ := 2^9

def favorable_sequences : ℕ := 10

theorem probability_at_least_6_consecutive_heads :
  (favorable_sequences : ℚ) / total_sequences = 5 / 256 := by
  sorry

end probability_at_least_6_consecutive_heads_l1265_126538


namespace exchange_10_dollars_equals_1200_yen_l1265_126578

/-- The exchange rate from US dollars to Japanese yen -/
def exchange_rate : ℝ := 120

/-- The amount of US dollars to be exchanged -/
def dollars_to_exchange : ℝ := 10

/-- The function that calculates the amount of yen received for a given amount of dollars -/
def exchange (dollars : ℝ) : ℝ := dollars * exchange_rate

theorem exchange_10_dollars_equals_1200_yen :
  exchange dollars_to_exchange = 1200 := by
  sorry

end exchange_10_dollars_equals_1200_yen_l1265_126578


namespace polygon_diagonals_sides_l1265_126569

theorem polygon_diagonals_sides (n : ℕ) (h : n > 2) : 
  n * (n - 3) / 2 = 2 * n → n = 7 := by
  sorry

end polygon_diagonals_sides_l1265_126569


namespace neil_cookies_l1265_126575

theorem neil_cookies (total : ℕ) (first_fraction second_fraction third_fraction : ℚ) : 
  total = 60 ∧ 
  first_fraction = 1/3 ∧ 
  second_fraction = 1/4 ∧ 
  third_fraction = 2/5 →
  total - 
    (total * first_fraction).floor - 
    ((total - (total * first_fraction).floor) * second_fraction).floor - 
    ((total - (total * first_fraction).floor - ((total - (total * first_fraction).floor) * second_fraction).floor) * third_fraction).floor = 18 :=
by sorry

end neil_cookies_l1265_126575


namespace calculator_time_saved_l1265_126529

/-- The time saved by using a calculator for math homework -/
theorem calculator_time_saved 
  (time_with_calc : ℕ)      -- Time per problem with calculator
  (time_without_calc : ℕ)   -- Time per problem without calculator
  (num_problems : ℕ)        -- Number of problems in the assignment
  (h1 : time_with_calc = 2) -- It takes 2 minutes per problem with calculator
  (h2 : time_without_calc = 5) -- It takes 5 minutes per problem without calculator
  (h3 : num_problems = 20)  -- The assignment has 20 problems
  : (time_without_calc - time_with_calc) * num_problems = 60 := by
  sorry

end calculator_time_saved_l1265_126529


namespace min_value_expression_l1265_126548

theorem min_value_expression (x : ℝ) (h : x > 0) :
  4 * x + 9 / x^2 ≥ 3 * (36 : ℝ)^(1/3) ∧
  ∃ y > 0, 4 * y + 9 / y^2 = 3 * (36 : ℝ)^(1/3) :=
sorry

end min_value_expression_l1265_126548


namespace valid_sequences_count_l1265_126527

-- Define the square
def Square := {A : ℝ × ℝ | A = (1, 1) ∨ A = (-1, 1) ∨ A = (-1, -1) ∨ A = (1, -1)}

-- Define the transformations
inductive Transform
| L  -- 90° counterclockwise rotation
| R  -- 90° clockwise rotation
| H  -- reflection across x-axis
| V  -- reflection across y-axis

-- Define a sequence of transformations
def TransformSequence := List Transform

-- Function to check if a transformation is a reflection
def isReflection (t : Transform) : Bool :=
  match t with
  | Transform.H => true
  | Transform.V => true
  | _ => false

-- Function to count reflections in a sequence
def countReflections (seq : TransformSequence) : Nat :=
  seq.filter isReflection |>.length

-- Function to check if a sequence maps the square back to itself
def mapsToSelf (seq : TransformSequence) : Bool :=
  sorry  -- Implementation details omitted

-- Theorem statement
theorem valid_sequences_count (n : Nat) :
  (∃ (seqs : List TransformSequence),
    (∀ seq ∈ seqs,
      seq.length = 24 ∧
      mapsToSelf seq ∧
      Even (countReflections seq)) ∧
    seqs.length = n) :=
  sorry

#check valid_sequences_count

end valid_sequences_count_l1265_126527


namespace incorrect_statement_E_l1265_126519

theorem incorrect_statement_E (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  -- Statement A
  (∀ x y : ℝ, x > 0 → y > 0 → x > y → x^2 > y^2) ∧
  -- Statement B
  (2 * a * b / (a + b) < Real.sqrt (a * b)) ∧
  -- Statement C
  (∀ p : ℝ, p > 0 → ∀ x y : ℝ, x > 0 → y > 0 → x * y = p → 
    x + y ≥ 2 * Real.sqrt p ∧ (x + y = 2 * Real.sqrt p ↔ x = y)) ∧
  -- Statement D
  ((a + b)^3 > (a^3 + b^3) / 2) ∧
  -- Statement E (negation)
  ¬((a + b)^2 / 4 > (a^2 + b^2) / 2) := by
sorry

end incorrect_statement_E_l1265_126519


namespace total_cost_for_group_stay_l1265_126587

-- Define the rates and conditions
def weekdayRateFirstWeek : ℚ := 18
def weekendRateFirstWeek : ℚ := 20
def weekdayRateAdditionalWeeks : ℚ := 11
def weekendRateAdditionalWeeks : ℚ := 13
def securityDeposit : ℚ := 50
def groupDiscountRate : ℚ := 0.1
def groupSize : ℕ := 5
def stayDuration : ℕ := 23

-- Define the function to calculate the total cost
def calculateTotalCost : ℚ := sorry

-- Theorem statement
theorem total_cost_for_group_stay :
  calculateTotalCost = 327.6 := by sorry

end total_cost_for_group_stay_l1265_126587


namespace tangent_line_at_zero_range_of_a_inequality_proof_l1265_126554

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / Real.exp x - x + 1

theorem tangent_line_at_zero (h : ℝ) : 
  ∃ (m b : ℝ), m * h + b = f 1 h ∧ 
  ∀ x, m * x + b = 2 * x - 2 + f 1 0 := by sorry

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, f a x < 0) → a ≤ -1 := by sorry

theorem inequality_proof (x : ℝ) : 
  x > 0 → 2 / Real.exp x - 2 < (1/2) * x^2 - x := by sorry

end tangent_line_at_zero_range_of_a_inequality_proof_l1265_126554


namespace complex_number_in_third_quadrant_l1265_126535

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (2 - 3 * Complex.I) / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_third_quadrant_l1265_126535


namespace cube_paint_theorem_l1265_126570

/-- 
Given a cube with side length n, prove that if exactly one-third of the total number of faces 
of the n³ unit cubes (after cutting) are blue, then n = 3.
-/
theorem cube_paint_theorem (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

#check cube_paint_theorem

end cube_paint_theorem_l1265_126570


namespace v3_at_neg_one_l1265_126532

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 + 0.35*x + 1.8*x^2 - 3*x^3 + 6*x^4 - 5*x^5 + x^6

/-- v3 in Horner's method for f(x) -/
def v3 (x : ℝ) : ℝ := (((x - 5)*x + 6)*x - 3)

/-- Theorem: v3 equals -15 when x = -1 -/
theorem v3_at_neg_one : v3 (-1) = -15 := by sorry

end v3_at_neg_one_l1265_126532


namespace max_consecutive_integers_sum_45_l1265_126515

theorem max_consecutive_integers_sum_45 (n : ℕ) 
  (h : ∃ a : ℤ, (Finset.range n).sum (λ i => a + i) = 45) : n ≤ 90 := by
  sorry

end max_consecutive_integers_sum_45_l1265_126515


namespace abc_sum_product_zero_l1265_126507

theorem abc_sum_product_zero (a b c : ℝ) (h : 1/a + 1/b + 1/c = 1/(a+b+c)) :
  (a+b)*(b+c)*(a+c) = 0 := by sorry

end abc_sum_product_zero_l1265_126507


namespace square_difference_of_integers_l1265_126536

theorem square_difference_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 60) 
  (diff_eq : x - y = 16) : 
  x^2 - y^2 = 960 := by
sorry

end square_difference_of_integers_l1265_126536


namespace marbles_cost_marbles_cost_value_l1265_126566

def total_spent : ℚ := 20.52
def football_cost : ℚ := 4.95
def baseball_cost : ℚ := 6.52

theorem marbles_cost : ℚ :=
  total_spent - (football_cost + baseball_cost)

#check marbles_cost

theorem marbles_cost_value : marbles_cost = 9.05 := by sorry

end marbles_cost_marbles_cost_value_l1265_126566


namespace complex_sum_equal_negative_three_l1265_126599

theorem complex_sum_equal_negative_three (w : ℂ) 
  (h1 : w = Complex.exp (6 * Real.pi * Complex.I / 11))
  (h2 : w^11 = 1) :
  w / (1 + w^2) + w^2 / (1 + w^4) + w^3 / (1 + w^6) + w^4 / (1 + w^8) = -3 := by
  sorry

end complex_sum_equal_negative_three_l1265_126599


namespace least_n_satisfying_inequality_l1265_126544

theorem least_n_satisfying_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), k > 0 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) < (1 : ℚ) / 15 → k ≥ n) ∧ 
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ n = 4 :=
sorry

end least_n_satisfying_inequality_l1265_126544


namespace right_triangle_roots_l1265_126549

/-- Given complex numbers a and b, and complex roots z₁ and z₂ of z² + az + b = 0
    such that 0, z₁, and z₂ form a right triangle with z₂ opposite the right angle,
    prove that a²/b = 2 -/
theorem right_triangle_roots (a b z₁ z₂ : ℂ) 
    (h_root : z₁^2 + a*z₁ + b = 0 ∧ z₂^2 + a*z₂ + b = 0)
    (h_right_triangle : z₂ = z₁ * Complex.I) : 
    a^2 / b = 2 := by
  sorry

end right_triangle_roots_l1265_126549


namespace ice_cream_cost_l1265_126550

theorem ice_cream_cost (people : ℕ) (meal_cost : ℚ) (total_amount : ℚ) 
  (h1 : people = 3)
  (h2 : meal_cost = 10)
  (h3 : total_amount = 45)
  (h4 : total_amount ≥ people * meal_cost) :
  (total_amount - people * meal_cost) / people = 5 := by
  sorry

end ice_cream_cost_l1265_126550


namespace intersection_M_N_l1265_126590

-- Define the sets M and N
def M : Set ℝ := {x | x > -1}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end intersection_M_N_l1265_126590


namespace prob_two_tails_three_coins_l1265_126577

/-- A fair coin is a coin with equal probability of heads and tails. -/
def FairCoin : Type := Unit

/-- The outcome of tossing a coin. -/
inductive CoinOutcome
| Heads
| Tails

/-- The outcome of tossing multiple coins. -/
def MultiCoinOutcome (n : ℕ) := Fin n → CoinOutcome

/-- The number of coins being tossed. -/
def numCoins : ℕ := 3

/-- The total number of possible outcomes when tossing n fair coins. -/
def totalOutcomes (n : ℕ) : ℕ := 2^n

/-- The number of ways to get exactly k tails when tossing n coins. -/
def waysToGetKTails (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes. -/
def probability (favorableOutcomes totalOutcomes : ℕ) : ℚ := favorableOutcomes / totalOutcomes

/-- The main theorem: the probability of getting exactly 2 tails when tossing 3 fair coins is 3/8. -/
theorem prob_two_tails_three_coins : 
  probability (waysToGetKTails numCoins 2) (totalOutcomes numCoins) = 3/8 := by
  sorry

end prob_two_tails_three_coins_l1265_126577


namespace moving_circle_trajectory_l1265_126556

theorem moving_circle_trajectory (x y : ℝ) : 
  (∃ (t : ℝ), x^2 + y^2 = 4 + t^2 ∧ t = 1 ∨ t = -1) ↔ 
  (x^2 + y^2 = 9 ∨ x^2 + y^2 = 1) :=
by sorry

end moving_circle_trajectory_l1265_126556


namespace prime_factors_of_2008006_l1265_126541

theorem prime_factors_of_2008006 : 
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : Nat), 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ 
    Nat.Prime p₄ ∧ Nat.Prime p₅ ∧ Nat.Prime p₆ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₁ ≠ p₆ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₂ ≠ p₆ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₃ ≠ p₆ ∧
    p₄ ≠ p₅ ∧ p₄ ≠ p₆ ∧
    p₅ ≠ p₆ ∧
    2008006 = p₁ * p₂ * p₃ * p₄ * p₅ * p₆ ∧
    (∀ q : Nat, Nat.Prime q → q ∣ 2008006 → (q = p₁ ∨ q = p₂ ∨ q = p₃ ∨ q = p₄ ∨ q = p₅ ∨ q = p₆)) :=
by sorry


end prime_factors_of_2008006_l1265_126541


namespace coffee_consumption_theorem_l1265_126598

/-- Represents the relationship between sleep and coffee consumption -/
def coffee_sleep_relation (sleep : ℝ) (coffee : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ sleep * coffee = k

theorem coffee_consumption_theorem (sleep_monday sleep_tuesday coffee_monday : ℝ) 
  (h1 : sleep_monday > 0)
  (h2 : sleep_tuesday > 0)
  (h3 : coffee_monday > 0)
  (h4 : coffee_sleep_relation sleep_monday coffee_monday)
  (h5 : coffee_sleep_relation sleep_tuesday (sleep_monday * coffee_monday / sleep_tuesday))
  (h6 : sleep_monday = 9)
  (h7 : sleep_tuesday = 6)
  (h8 : coffee_monday = 2) :
  sleep_monday * coffee_monday / sleep_tuesday = 3 := by
  sorry

#check coffee_consumption_theorem

end coffee_consumption_theorem_l1265_126598


namespace quadratic_less_than_linear_l1265_126572

theorem quadratic_less_than_linear (x : ℝ) : -1/2 * x^2 + 2*x < -x + 5 := by
  sorry

end quadratic_less_than_linear_l1265_126572


namespace quadratic_two_zeros_l1265_126547

theorem quadratic_two_zeros (a b c : ℝ) (h : a * c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_two_zeros_l1265_126547


namespace digits_9998_to_10000_of_1_998_l1265_126509

/-- The decimal expansion of 1/998 -/
def decimal_expansion_1_998 : ℕ → ℕ := sorry

/-- The function that extracts a 3-digit number from the decimal expansion -/
def extract_three_digits (start : ℕ) : ℕ := 
  100 * (decimal_expansion_1_998 start) + 
  10 * (decimal_expansion_1_998 (start + 1)) + 
  decimal_expansion_1_998 (start + 2)

/-- The theorem stating that the 9998th through 10000th digits of 1/998 form 042 -/
theorem digits_9998_to_10000_of_1_998 : 
  extract_three_digits 9998 = 42 := by sorry

end digits_9998_to_10000_of_1_998_l1265_126509


namespace lawn_area_proof_l1265_126576

theorem lawn_area_proof (total_posts : ℕ) (post_spacing : ℕ) 
  (h_total_posts : total_posts = 24)
  (h_post_spacing : post_spacing = 5)
  (h_longer_side_posts : ∀ s l : ℕ, s + l = total_posts / 2 → l + 1 = 3 * (s + 1)) :
  ∃ short_side long_side : ℕ,
    short_side * long_side = 500 ∧
    short_side + 1 + long_side + 1 = total_posts ∧
    (long_side + 1) * post_spacing = (short_side + 1) * post_spacing * 3 :=
by sorry

end lawn_area_proof_l1265_126576


namespace tetrahedron_fits_in_box_l1265_126568

theorem tetrahedron_fits_in_box :
  ∀ (tetra_edge box_length box_width box_height : ℝ),
    tetra_edge = 12 →
    box_length = 9 ∧ box_width = 13 ∧ box_height = 15 →
    ∃ (cube_edge : ℝ),
      cube_edge = tetra_edge / Real.sqrt 2 ∧
      cube_edge ≤ box_length ∧
      cube_edge ≤ box_width ∧
      cube_edge ≤ box_height :=
by sorry

end tetrahedron_fits_in_box_l1265_126568


namespace club_member_age_difference_l1265_126562

/-- Given a club with 10 members, prove that replacing one member
    results in a 50-year difference between the old and new member's ages
    if the average age remains the same after 5 years. -/
theorem club_member_age_difference
  (n : ℕ) -- number of club members
  (a : ℝ) -- average age of members 5 years ago
  (o : ℝ) -- age of the old (replaced) member
  (n' : ℝ) -- age of the new member
  (h1 : n = 10) -- there are 10 members
  (h2 : n * a = (n - 1) * (a + 5) + n') -- average age remains the same after 5 years and replacement
  : |o - n'| = 50 := by
  sorry


end club_member_age_difference_l1265_126562


namespace kyle_car_payment_l1265_126579

def monthly_income : ℝ := 3200

def rent : ℝ := 1250
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries : ℝ := 300
def insurance : ℝ := 200
def miscellaneous : ℝ := 200
def gas_maintenance : ℝ := 350

def other_expenses : ℝ := rent + utilities + retirement_savings + groceries + insurance + miscellaneous + gas_maintenance

def car_payment : ℝ := monthly_income - other_expenses

theorem kyle_car_payment :
  car_payment = 350 := by sorry

end kyle_car_payment_l1265_126579


namespace matrix_equation_l1265_126582

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 12, 4]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -26/7, 34/7]

theorem matrix_equation : N * A = B := by sorry

end matrix_equation_l1265_126582


namespace max_chord_length_l1265_126506

-- Define the family of curves
def family_of_curves (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

-- Define the line y = 2x
def line (x y : ℝ) : Prop := y = 2 * x

-- Theorem statement
theorem max_chord_length :
  ∃ (max_length : ℝ),
    (∀ θ x₁ y₁ x₂ y₂ : ℝ,
      family_of_curves θ x₁ y₁ ∧
      family_of_curves θ x₂ y₂ ∧
      line x₁ y₁ ∧
      line x₂ y₂ →
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≤ max_length) ∧
    max_length = 8 * Real.sqrt 5 :=
by sorry

end max_chord_length_l1265_126506


namespace function_monotonicity_and_extrema_l1265_126555

noncomputable section

variable (a : ℝ)
variable (k : ℝ)

def f (x : ℝ) : ℝ := (x - a - 1) * Real.exp x - (1/2) * x^2 + a * x

theorem function_monotonicity_and_extrema (h : a > 0) :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 0 → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → f a x₁ > f a x₂) ∧
  (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, (∀ x, f a x₁ ≤ f a x) ∧ (∀ x, f a x₂ ≥ f a x) →
    (∀ a, a > 0 → f a x₁ - f a x₂ < k * a^3) ↔ k ≥ -1/6) :=
sorry

end

end function_monotonicity_and_extrema_l1265_126555


namespace total_distance_is_200_l1265_126559

/-- Represents the cycling journey of Jack and Peter -/
structure CyclingJourney where
  speed : ℝ
  timeHomeToStore : ℝ
  timeStoreToPeter : ℝ
  distanceStoreToPeter : ℝ

/-- Calculates the total distance cycled by Jack and Peter -/
def totalDistanceCycled (journey : CyclingJourney) : ℝ :=
  let distanceHomeToStore := journey.speed * journey.timeHomeToStore
  let distanceStoreToPeter := journey.distanceStoreToPeter
  distanceHomeToStore + 2 * distanceStoreToPeter

/-- Theorem stating the total distance cycled is 200 miles -/
theorem total_distance_is_200 (journey : CyclingJourney) 
  (h1 : journey.timeHomeToStore = 2 * journey.timeStoreToPeter)
  (h2 : journey.speed > 0)
  (h3 : journey.distanceStoreToPeter = 50) :
  totalDistanceCycled journey = 200 := by
  sorry


end total_distance_is_200_l1265_126559


namespace parallel_lines_m_value_l1265_126531

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℚ) : Prop := m₁ = m₂

/-- The slope of a line in the form ax + by + c = 0 is -a/b -/
def slope_general_form (a b : ℚ) : ℚ := -a / b

/-- The slope of a line in the form y = mx + b is m -/
def slope_slope_intercept_form (m : ℚ) : ℚ := m

theorem parallel_lines_m_value :
  ∀ m : ℚ, 
  parallel (slope_general_form 2 m) (slope_slope_intercept_form 3) →
  m = -2/3 := by
sorry


end parallel_lines_m_value_l1265_126531


namespace min_value_of_sum_l1265_126511

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b - a * b = 0) :
  a + b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4 * a₀ + b₀ - a₀ * b₀ = 0 ∧ a₀ + b₀ = 9 :=
by sorry

end min_value_of_sum_l1265_126511


namespace intersection_A_B_l1265_126543

def A : Set ℝ := {x | x^2 - 3*x ≤ 0}
def B : Set ℝ := {1, 2}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end intersection_A_B_l1265_126543


namespace seminar_fee_calculation_l1265_126561

/-- Proves that the regular seminar fee is $150 given the problem conditions --/
theorem seminar_fee_calculation (F : ℝ) : 
  (∃ (total_spent discounted_fee : ℝ),
    -- 5% discount applied
    discounted_fee = F * 0.95 ∧
    -- 10 teachers registered
    -- $10 food allowance per teacher
    total_spent = 10 * discounted_fee + 10 * 10 ∧
    -- Total spent is $1525
    total_spent = 1525) →
  F = 150 := by
  sorry

end seminar_fee_calculation_l1265_126561
