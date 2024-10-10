import Mathlib

namespace salary_change_percentage_l58_5895

theorem salary_change_percentage (original : ℝ) (h : original > 0) :
  let decreased := original * (1 - 0.5)
  let increased := decreased * (1 + 0.5)
  increased = original * 0.75 ∧ (original - increased) / original = 0.25 :=
by
  sorry

end salary_change_percentage_l58_5895


namespace min_groups_for_students_l58_5842

theorem min_groups_for_students (total_students : ℕ) (max_group_size : ℕ) (h1 : total_students = 30) (h2 : max_group_size = 12) :
  ∃ (num_groups : ℕ), 
    num_groups * (total_students / num_groups) = total_students ∧
    (total_students / num_groups) ≤ max_group_size ∧
    ∀ (k : ℕ), k * (total_students / k) = total_students ∧ (total_students / k) ≤ max_group_size → k ≥ num_groups :=
by
  sorry

end min_groups_for_students_l58_5842


namespace smallest_n_exceeding_15_l58_5896

-- Define a function to calculate the sum of digits of a natural number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the function g(n) as the sum of digits of 1/3^n to the right of the decimal point
def g (n : ℕ) : ℕ := sumOfDigits (10^n / 3^n)

-- Theorem statement
theorem smallest_n_exceeding_15 :
  (∀ k < 6, g k ≤ 15) ∧ g 6 > 15 := by sorry

end smallest_n_exceeding_15_l58_5896


namespace magnitude_of_z_l58_5864

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The given complex number -/
def z : ℂ := (1 - 2*i) * i

/-- Theorem stating that the magnitude of z is √5 -/
theorem magnitude_of_z : Complex.abs z = Real.sqrt 5 := by sorry

end magnitude_of_z_l58_5864


namespace deck_size_proof_l58_5888

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b : ℚ) = 1/3 → 
  (r : ℚ) / (r + b + 4 : ℚ) = 1/4 → 
  r + b = 12 := by
sorry

end deck_size_proof_l58_5888


namespace common_chord_and_length_l58_5820

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the equation of the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- State the theorem
theorem common_chord_and_length :
  -- The equation of the common chord
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord x y) ∧
  -- The length of the common chord
  (∃ a b c d : ℝ,
    circle1 a b ∧ circle2 a b ∧ circle1 c d ∧ circle2 c d ∧
    common_chord a b ∧ common_chord c d ∧
    ((a - c)^2 + (b - d)^2) = 20) :=
by sorry

end common_chord_and_length_l58_5820


namespace monochromatic_isosceles_independent_of_coloring_l58_5801

/-- A regular polygon with 6n+1 sides -/
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  is_regular : sides = 6 * n + 1

/-- A coloring of the vertices of a regular polygon -/
structure Coloring (n : ℕ) where
  polygon : RegularPolygon n
  red_vertices : ℕ
  valid_coloring : red_vertices ≤ polygon.sides

/-- An isosceles triangle in a regular polygon -/
structure IsoscelesTriangle (n : ℕ) where
  polygon : RegularPolygon n

/-- A monochromatic isosceles triangle (all vertices same color) -/
structure MonochromaticIsoscelesTriangle (n : ℕ) extends IsoscelesTriangle n where
  coloring : Coloring n

/-- The number of monochromatic isosceles triangles in a colored regular polygon -/
def num_monochromatic_isosceles_triangles (n : ℕ) (c : Coloring n) : ℕ := sorry

/-- The main theorem: the number of monochromatic isosceles triangles is independent of coloring -/
theorem monochromatic_isosceles_independent_of_coloring (n : ℕ) 
  (c1 c2 : Coloring n) (h : c1.red_vertices = c2.red_vertices) :
  num_monochromatic_isosceles_triangles n c1 = num_monochromatic_isosceles_triangles n c2 := by
  sorry

end monochromatic_isosceles_independent_of_coloring_l58_5801


namespace amp_calculation_l58_5826

-- Define the & operation
def amp (a b : ℤ) : ℤ := (a + b) * (a - b)

-- State the theorem
theorem amp_calculation : amp (amp 10 4) 2 = 7052 := by
  sorry

end amp_calculation_l58_5826


namespace average_tip_fraction_l58_5870

-- Define the weekly tip fractions
def week1_tip_fraction : ℚ := 2/4
def week2_tip_fraction : ℚ := 3/8
def week3_tip_fraction : ℚ := 5/16
def week4_tip_fraction : ℚ := 1/4

-- Define the number of weeks
def num_weeks : ℕ := 4

-- Theorem statement
theorem average_tip_fraction :
  (week1_tip_fraction + week2_tip_fraction + week3_tip_fraction + week4_tip_fraction) / num_weeks = 23/64 := by
  sorry

end average_tip_fraction_l58_5870


namespace lines_sum_l58_5804

-- Define the lines
def l₀ (x y : ℝ) : Prop := x - y + 1 = 0
def l₁ (a x y : ℝ) : Prop := a * x - 2 * y + 1 = 0
def l₂ (b x y : ℝ) : Prop := x + b * y + 3 = 0

-- Define perpendicularity and parallelism
def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, f x₁ y₁ → f x₂ y₂ → g x₁ y₁ → g x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (y₂ - y₁) - (y₂ - y₁) * (x₂ - x₁) = 0)

def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ, 
    f x₁ y₁ → f x₂ y₂ → g x₃ y₃ → g x₄ y₄ →
    (x₂ - x₁) * (y₄ - y₃) = (y₂ - y₁) * (x₄ - x₃)

-- Theorem statement
theorem lines_sum (a b : ℝ) : 
  perpendicular (l₀) (l₁ a) → parallel (l₀) (l₂ b) → a + b = -3 := by
  sorry

end lines_sum_l58_5804


namespace divisor_properties_l58_5848

def N (a b c : ℕ) (α β γ : ℕ) : ℕ := a^α * b^β * c^γ

variable (a b c α β γ : ℕ)
variable (ha : Nat.Prime a)
variable (hb : Nat.Prime b)
variable (hc : Nat.Prime c)

theorem divisor_properties :
  let n := N a b c α β γ
  -- Total number of divisors
  ∃ d : ℕ → ℕ, d n = (α + 1) * (β + 1) * (γ + 1) ∧
  -- Product of equidistant divisors
  ∀ x y : ℕ, x ∣ n → y ∣ n → x * y = n →
    ∃ z : ℕ, z ∣ n ∧ z * z = n ∧
  -- Product of all divisors
  ∃ P : ℕ, P = n ^ ((α + 1) * (β + 1) * (γ + 1) / 2) :=
by sorry

end divisor_properties_l58_5848


namespace price_decrease_percentage_l58_5833

theorem price_decrease_percentage (P : ℝ) (x : ℝ) (h₁ : P > 0) :
  (1.20 * P) * (1 - x / 100) = 0.75 * P → x = 37.5 := by
  sorry

end price_decrease_percentage_l58_5833


namespace inscribed_circle_rectangle_area_l58_5850

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (ratio : ℝ),
    r = 7 →
    ratio = 3 →
    let d := 2 * r
    let w := d
    let l := ratio * w
    l * w = 588 := by
  sorry

end inscribed_circle_rectangle_area_l58_5850


namespace wilsons_theorem_l58_5806

theorem wilsons_theorem (p : Nat) (hp : Nat.Prime p) : (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end wilsons_theorem_l58_5806


namespace ohara_triple_x_value_l58_5834

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℝ) : Prop :=
  Real.sqrt (abs a) + Real.sqrt (abs b) = x

/-- Theorem: If (-49, 64, x) is an O'Hara triple, then x = 15 -/
theorem ohara_triple_x_value :
  ∀ x : ℝ, is_ohara_triple (-49) 64 x → x = 15 := by
  sorry

end ohara_triple_x_value_l58_5834


namespace prob_at_least_one_8_l58_5871

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The probability of getting at least one 8 when rolling two fair 8-sided dice -/
def probAtLeastOne8 : ℚ := 15 / 64

/-- Theorem: The probability of getting at least one 8 when rolling two fair 8-sided dice is 15/64 -/
theorem prob_at_least_one_8 : 
  probAtLeastOne8 = (numSides^2 - (numSides - 1)^2) / numSides^2 := by
  sorry

end prob_at_least_one_8_l58_5871


namespace range_of_a_part1_range_of_a_part2_l58_5862

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

-- Theorem for the first part of the problem
theorem range_of_a_part1 (a : ℝ) :
  (∃ x ∈ Set.Icc 1 3, f a x > 0) → a < 4 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_a_part2 (a : ℝ) :
  (∀ x ∈ Set.Icc 1 4, f a x ≥ -a) → a ≤ 6 :=
sorry

end range_of_a_part1_range_of_a_part2_l58_5862


namespace sqrt_sum_inequality_l58_5811

theorem sqrt_sum_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt a + Real.sqrt b ≥ Real.sqrt (a + b) := by
  sorry

end sqrt_sum_inequality_l58_5811


namespace walnut_trees_before_planting_l58_5885

theorem walnut_trees_before_planting (trees_to_plant : ℕ) (final_trees : ℕ) 
  (h1 : trees_to_plant = 104)
  (h2 : final_trees = 211) :
  final_trees - trees_to_plant = 107 := by
  sorry

end walnut_trees_before_planting_l58_5885


namespace refrigerator_profit_percentage_l58_5897

/-- Calculates the percentage of profit for a refrigerator sale --/
theorem refrigerator_profit_percentage
  (discounted_price : ℝ)
  (discount_percentage : ℝ)
  (transport_cost : ℝ)
  (installation_cost : ℝ)
  (selling_price : ℝ)
  (h1 : discounted_price = 14500)
  (h2 : discount_percentage = 20)
  (h3 : transport_cost = 125)
  (h4 : installation_cost = 250)
  (h5 : selling_price = 20350) :
  ∃ (profit_percentage : ℝ),
    abs (profit_percentage - 36.81) < 0.01 ∧
    profit_percentage = (selling_price - (discounted_price + transport_cost + installation_cost)) /
                        (discounted_price + transport_cost + installation_cost) * 100 :=
by sorry


end refrigerator_profit_percentage_l58_5897


namespace quadrilateral_theorem_l58_5836

-- Define a quadrilateral
structure Quadrilateral :=
(A B C D : ℝ × ℝ)

-- Define the angle between two vectors
def angle (v w : ℝ × ℝ) : ℝ := sorry

-- Define the length of a vector
def length (v : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem quadrilateral_theorem (q : Quadrilateral) 
  (h : angle (q.C.1 - q.A.1, q.C.2 - q.A.2) (q.A.1 - q.C.1, q.A.2 - q.C.2) = 120) :
  (length (q.A.1 - q.C.1, q.A.2 - q.C.2) * length (q.B.1 - q.D.1, q.B.2 - q.D.2))^2 =
  (length (q.A.1 - q.B.1, q.A.2 - q.B.2) * length (q.C.1 - q.D.1, q.C.2 - q.D.2))^2 +
  (length (q.B.1 - q.C.1, q.B.2 - q.C.2) * length (q.A.1 - q.D.1, q.A.2 - q.D.2))^2 +
  length (q.A.1 - q.B.1, q.A.2 - q.B.2) * length (q.B.1 - q.C.1, q.B.2 - q.C.2) *
  length (q.C.1 - q.D.1, q.C.2 - q.D.2) * length (q.D.1 - q.A.1, q.D.2 - q.A.2) :=
by sorry

end quadrilateral_theorem_l58_5836


namespace pencil_cost_to_selling_ratio_l58_5890

/-- The ratio of the cost to the selling price for 70 pencils -/
theorem pencil_cost_to_selling_ratio 
  (C : ℝ) -- Cost price of one pencil
  (S : ℝ) -- Selling price of one pencil
  (h1 : C > 0) -- Assumption that cost is positive
  (h2 : S > 0) -- Assumption that selling price is positive
  (h3 : C > (2/7) * S) -- Assumption that cost is greater than 2/7 of selling price
  : (70 * C) / (70 * C - 20 * S) = C / (C - 2 * S / 7) :=
by sorry

end pencil_cost_to_selling_ratio_l58_5890


namespace power_inequality_l58_5846

theorem power_inequality (a b n : ℕ) (ha : a > b) (hb : b > 1) (hodd : Odd b) 
  (hn : n > 0) (hdiv : (b^n : ℕ) ∣ (a^n - 1)) : 
  (a : ℝ)^b > (3 : ℝ)^n / n := by
  sorry

end power_inequality_l58_5846


namespace percentage_passed_both_subjects_l58_5849

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 32)
  (h2 : failed_english = 56)
  (h3 : failed_both = 12) :
  100 - (failed_hindi + failed_english - failed_both) = 24 :=
by sorry

end percentage_passed_both_subjects_l58_5849


namespace polynomial_factorization_and_trig_inequality_l58_5889

theorem polynomial_factorization_and_trig_inequality :
  (∀ x : ℂ, x^12 + x^9 + x^6 + x^3 + 1 = (x^4 + x^3 + x^2 + 1) * (x^8 - x^7 + x^5 - x^4 + x^3 - x + 1)) ∧
  (∀ θ : ℝ, 5 + 8 * Real.cos θ + 4 * Real.cos (2 * θ) + Real.cos (3 * θ) ≥ 0) :=
by sorry

end polynomial_factorization_and_trig_inequality_l58_5889


namespace arithmetic_mean_problem_l58_5881

theorem arithmetic_mean_problem (y : ℚ) : 
  ((y + 10) + 20 + (3 * y) + 18 + (3 * y + 6) + 12) / 6 = 30 → y = 114 / 7 := by
  sorry

end arithmetic_mean_problem_l58_5881


namespace constant_expression_inequality_solution_l58_5835

-- Part 1: Prove that the expression simplifies to -9 for all real x
theorem constant_expression (x : ℝ) : x * (x - 6) - (3 - x)^2 = -9 := by
  sorry

-- Part 2: Prove that the solution to the inequality is x < 5
theorem inequality_solution : 
  {x : ℝ | x - 2*(x - 3) > 1} = {x : ℝ | x < 5} := by
  sorry

end constant_expression_inequality_solution_l58_5835


namespace remainder_of_large_number_l58_5829

theorem remainder_of_large_number (n : Nat) (d : Nat) (h : d = 180) :
  n = 1234567890123 → n % d = 123 := by
  sorry

end remainder_of_large_number_l58_5829


namespace jake_has_fewer_balloons_l58_5887

/-- The number of balloons each person has in the park scenario -/
structure BalloonCounts where
  allan : ℕ
  jake_initial : ℕ
  jake_bought : ℕ
  emily : ℕ

/-- The difference in balloon count between Jake and the combined total of Allan and Emily -/
def balloon_difference (counts : BalloonCounts) : ℤ :=
  (counts.jake_initial + counts.jake_bought : ℤ) - (counts.allan + counts.emily)

/-- Theorem stating that Jake has 4 fewer balloons than Allan and Emily combined -/
theorem jake_has_fewer_balloons (counts : BalloonCounts)
  (h1 : counts.allan = 6)
  (h2 : counts.jake_initial = 3)
  (h3 : counts.jake_bought = 4)
  (h4 : counts.emily = 5) :
  balloon_difference counts = -4 := by
  sorry

end jake_has_fewer_balloons_l58_5887


namespace sleep_deficit_l58_5824

def weeknights : ℕ := 5
def weekendNights : ℕ := 2
def actualWeekdaySleep : ℕ := 5
def actualWeekendSleep : ℕ := 6
def idealSleep : ℕ := 8

theorem sleep_deficit :
  (weeknights * idealSleep + weekendNights * idealSleep) -
  (weeknights * actualWeekdaySleep + weekendNights * actualWeekendSleep) = 19 := by
  sorry

end sleep_deficit_l58_5824


namespace function_value_at_symmetric_point_l58_5873

/-- Given a function f(x) = a * sin³(x) + b * tan(x) + 1 where f(2) = 3,
    prove that f(2π - 2) = -1 -/
theorem function_value_at_symmetric_point
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * (Real.sin x)^3 + b * Real.tan x + 1)
  (h2 : f 2 = 3) :
  f (2 * Real.pi - 2) = -1 := by
  sorry

end function_value_at_symmetric_point_l58_5873


namespace first_discount_percentage_l58_5894

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) : 
  original_price = 390 →
  final_price = 248.625 →
  ∃ (first_discount : ℝ),
    first_discount = 15 ∧
    final_price = original_price * (100 - first_discount) / 100 * 75 / 100 :=
by sorry

end first_discount_percentage_l58_5894


namespace degree_of_P_l58_5816

/-- The degree of a monomial in two variables --/
def monomialDegree (a b : ℕ) : ℕ := a + b

/-- The degree of a polynomial is the maximum degree of its monomials --/
def polynomialDegree (degrees : List ℕ) : ℕ := List.foldl max 0 degrees

/-- The polynomial -3a²b + 7a²b² - 2ab --/
def P (a b : ℝ) : ℝ := -3 * a^2 * b + 7 * a^2 * b^2 - 2 * a * b

theorem degree_of_P : 
  polynomialDegree [monomialDegree 2 1, monomialDegree 2 2, monomialDegree 1 1] = 4 := by
  sorry

end degree_of_P_l58_5816


namespace line_passes_through_circle_center_l58_5817

/-- The line 2x - y = 0 passes through the center of the circle (x-a)² + (y-2a)² = 1 for all real a -/
theorem line_passes_through_circle_center (a : ℝ) : 2 * a - 2 * a = 0 := by sorry

end line_passes_through_circle_center_l58_5817


namespace max_balloons_is_400_l58_5825

def small_bag_cost : ℕ := 4
def small_bag_balloons : ℕ := 50
def medium_bag_cost : ℕ := 6
def medium_bag_balloons : ℕ := 75
def large_bag_cost : ℕ := 12
def large_bag_balloons : ℕ := 200
def budget : ℕ := 24

def max_balloons (budget small_cost small_balloons medium_cost medium_balloons large_cost large_balloons : ℕ) : ℕ := 
  sorry

theorem max_balloons_is_400 : 
  max_balloons budget small_bag_cost small_bag_balloons medium_bag_cost medium_bag_balloons large_bag_cost large_bag_balloons = 400 :=
by sorry

end max_balloons_is_400_l58_5825


namespace equal_volume_implies_equal_breadth_l58_5856

/-- Represents the volume of earth dug in a project -/
structure EarthVolume where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of earth dug -/
def calculateVolume (v : EarthVolume) : ℝ :=
  v.depth * v.length * v.breadth

theorem equal_volume_implies_equal_breadth 
  (project1 : EarthVolume)
  (project2 : EarthVolume)
  (h1 : project1.depth = 100)
  (h2 : project1.length = 25)
  (h3 : project1.breadth = 30)
  (h4 : project2.depth = 75)
  (h5 : project2.length = 20)
  (h6 : calculateVolume project1 = calculateVolume project2) :
  project2.breadth = 50 := by
sorry

end equal_volume_implies_equal_breadth_l58_5856


namespace tims_medical_cost_tims_out_of_pocket_cost_l58_5832

/-- Calculates the out-of-pocket cost for Tim's medical visit --/
theorem tims_medical_cost (mri_cost : ℚ) (doctor_rate : ℚ) (exam_time : ℚ) 
  (visit_fee : ℚ) (insurance_coverage : ℚ) : ℚ :=
  let total_cost := mri_cost + doctor_rate * exam_time / 2 + visit_fee
  let insurance_payment := total_cost * insurance_coverage
  total_cost - insurance_payment

/-- Proves that Tim's out-of-pocket cost is $300 --/
theorem tims_out_of_pocket_cost : 
  tims_medical_cost 1200 300 (1/2) 150 (4/5) = 300 := by
  sorry

end tims_medical_cost_tims_out_of_pocket_cost_l58_5832


namespace fortieth_number_is_twelve_l58_5855

/-- Represents the value in a specific position of the arrangement --/
def arrangementValue (position : ℕ) : ℕ :=
  let rowNum : ℕ := (position - 1).sqrt + 1
  2 * rowNum

/-- The theorem stating that the 40th number in the arrangement is 12 --/
theorem fortieth_number_is_twelve : arrangementValue 40 = 12 := by
  sorry

end fortieth_number_is_twelve_l58_5855


namespace water_depth_calculation_l58_5807

def rons_height : ℝ := 13

def water_depth : ℝ := 16 * rons_height

theorem water_depth_calculation : water_depth = 208 := by
  sorry

end water_depth_calculation_l58_5807


namespace sum_of_cyclic_equations_l58_5899

theorem sum_of_cyclic_equations (p q r : ℝ) 
  (distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (eq1 : q = p * (4 - p))
  (eq2 : r = q * (4 - q))
  (eq3 : p = r * (4 - r)) :
  p + q + r = 6 ∨ p + q + r = 7 := by
sorry

end sum_of_cyclic_equations_l58_5899


namespace line_always_intersects_ellipse_l58_5827

/-- A line with equation y = kx + 2, where k is a real number. -/
structure Line where
  k : ℝ

/-- An ellipse with equation x² + y²/m = 1, where m is a positive real number. -/
structure Ellipse where
  m : ℝ
  h_pos : 0 < m

/-- 
Given a line y = kx + 2 and an ellipse x² + y²/m = 1,
if the line always intersects the ellipse for all real k,
then m is greater than or equal to 4.
-/
theorem line_always_intersects_ellipse (e : Ellipse) :
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 2 ∧ x^2 + y^2 / e.m = 1) →
  4 ≤ e.m :=
sorry

end line_always_intersects_ellipse_l58_5827


namespace det_B_equals_five_l58_5844

theorem det_B_equals_five (b c : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![b, 3; -1, c]
  B + 3 * B⁻¹ = 0 → Matrix.det B = 5 := by
sorry

end det_B_equals_five_l58_5844


namespace leading_coefficient_of_g_l58_5843

/-- A polynomial g satisfying g(x + 1) - g(x) = 6x + 6 for all x has leading coefficient 3 -/
theorem leading_coefficient_of_g (g : ℝ → ℝ) 
  (h : ∀ x, g (x + 1) - g x = 6 * x + 6) :
  ∃ a b c : ℝ, (∀ x, g x = 3 * x^2 + a * x + b) ∧ c = 3 ∧ c ≠ 0 ∧ 
  (∀ d, (∀ x, g x = d * x^2 + a * x + b) → d ≤ c) := by
  sorry

end leading_coefficient_of_g_l58_5843


namespace base7_146_equals_83_l58_5891

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base7_146_equals_83 :
  base7ToBase10 [6, 4, 1] = 83 := by sorry

end base7_146_equals_83_l58_5891


namespace students_not_playing_sports_l58_5861

theorem students_not_playing_sports (total : ℕ) (football : ℕ) (tennis : ℕ) (basketball : ℕ)
  (football_tennis : ℕ) (football_basketball : ℕ) (tennis_basketball : ℕ) (all_three : ℕ)
  (h_total : total = 50)
  (h_football : football = 26)
  (h_tennis : tennis = 20)
  (h_basketball : basketball = 15)
  (h_football_tennis : football_tennis = 9)
  (h_football_basketball : football_basketball = 7)
  (h_tennis_basketball : tennis_basketball = 6)
  (h_all_three : all_three = 4) :
  total - (football + tennis + basketball - football_tennis - football_basketball - tennis_basketball + all_three) = 7 := by
sorry

end students_not_playing_sports_l58_5861


namespace min_value_x_plus_2y_l58_5821

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = x*y) :
  x + 2*y ≥ 8 ∧ (x + 2*y = 8 ↔ x = 2*y) :=
by sorry

end min_value_x_plus_2y_l58_5821


namespace pottery_rim_diameter_l58_5838

theorem pottery_rim_diameter 
  (chord_length : ℝ) 
  (segment_height : ℝ) 
  (h1 : chord_length = 16) 
  (h2 : segment_height = 2) : 
  ∃ (diameter : ℝ), diameter = 34 ∧ 
  (∃ (radius : ℝ), 
    radius * 2 = diameter ∧
    radius^2 = (radius - segment_height)^2 + (chord_length / 2)^2) :=
by sorry

end pottery_rim_diameter_l58_5838


namespace line_perp_parallel_implies_planes_perp_l58_5831

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internals of the plane for this problem
  dummy : Unit

/-- A line in 3D space -/
structure Line3D where
  -- We don't need to define the internals of the line for this problem
  dummy : Unit

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def perpendicular_planes (p1 : Plane3D) (p2 : Plane3D) : Prop :=
  sorry

/-- The main theorem -/
theorem line_perp_parallel_implies_planes_perp 
  (a b g : Plane3D) (l : Line3D) 
  (h1 : a ≠ b) (h2 : a ≠ g) (h3 : b ≠ g)
  (h4 : perpendicular l a) (h5 : parallel l b) : 
  perpendicular_planes a b :=
sorry

end line_perp_parallel_implies_planes_perp_l58_5831


namespace odell_kershaw_passing_count_l58_5877

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- radius of the lane in meters
  direction : ℝ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def passingCount (runner1 runner2 : Runner) (duration : ℝ) : ℕ :=
  sorry

theorem odell_kershaw_passing_count :
  let odell : Runner := ⟨260, 55, 1⟩
  let kershaw : Runner := ⟨280, 65, -1⟩
  passingCount odell kershaw 45 = 64 :=
sorry

end odell_kershaw_passing_count_l58_5877


namespace square_difference_305_295_l58_5868

theorem square_difference_305_295 : (305 : ℤ)^2 - (295 : ℤ)^2 = 6000 := by
  sorry

end square_difference_305_295_l58_5868


namespace division_problem_l58_5841

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by sorry

end division_problem_l58_5841


namespace fraction_simplification_l58_5860

theorem fraction_simplification : (20 - 20) / (20 + 20) = 0 := by
  sorry

end fraction_simplification_l58_5860


namespace power_function_through_point_l58_5805

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x : ℝ, f x = x ^ α) →  -- f is a power function
  f 2 = Real.sqrt 2 →       -- f passes through (2, √2)
  ∀ x : ℝ, f x = x ^ (1/2)  -- f(x) = x^(1/2)
:= by sorry

end power_function_through_point_l58_5805


namespace red_surface_fraction_is_one_l58_5869

/-- Represents a cube with its edge length and number of smaller cubes -/
structure Cube where
  edge_length : ℕ
  num_small_cubes : ℕ

/-- Represents the composition of the cube in terms of colored smaller cubes -/
structure CubeComposition where
  total_cubes : Cube
  red_cubes : ℕ
  blue_cubes : ℕ

/-- The fraction of the surface area of the larger cube that is red -/
def red_surface_fraction (c : CubeComposition) : ℚ :=
  sorry

/-- The theorem stating the fraction of red surface area -/
theorem red_surface_fraction_is_one (c : CubeComposition) 
  (h1 : c.total_cubes.edge_length = 4)
  (h2 : c.total_cubes.num_small_cubes = 64)
  (h3 : c.red_cubes = 40)
  (h4 : c.blue_cubes = 24)
  (h5 : c.red_cubes + c.blue_cubes = c.total_cubes.num_small_cubes) :
  red_surface_fraction c = 1 := by
  sorry

end red_surface_fraction_is_one_l58_5869


namespace onion_transport_trips_l58_5812

theorem onion_transport_trips (bags_per_trip : ℕ) (weight_per_bag : ℕ) (total_weight : ℕ) : 
  bags_per_trip = 10 → weight_per_bag = 50 → total_weight = 10000 →
  (total_weight / (bags_per_trip * weight_per_bag) : ℕ) = 20 := by
sorry

end onion_transport_trips_l58_5812


namespace fraction_product_equals_fifteen_thirty_seconds_l58_5852

theorem fraction_product_equals_fifteen_thirty_seconds :
  (3 + 5 + 7) / (2 + 4 + 6) * (1 + 3 + 5) / (6 + 8 + 10) = 15 / 32 := by
  sorry

end fraction_product_equals_fifteen_thirty_seconds_l58_5852


namespace cobalt_percentage_is_15_percent_l58_5813

/-- Represents the composition of a mixture -/
structure Mixture where
  cobalt : ℝ
  lead : ℝ
  copper : ℝ

/-- The given mixture satisfies the problem conditions -/
def problem_mixture : Mixture where
  lead := 0.25
  copper := 0.60
  cobalt := 1 - (0.25 + 0.60)

/-- The total weight of the mixture in kg -/
def total_weight : ℝ := 5 + 12

theorem cobalt_percentage_is_15_percent (m : Mixture) 
  (h1 : m.lead = 0.25)
  (h2 : m.copper = 0.60)
  (h3 : m.lead + m.copper + m.cobalt = 1)
  (h4 : m.lead * total_weight = 5)
  (h5 : m.copper * total_weight = 12) :
  m.cobalt = 0.15 := by
  sorry

#check cobalt_percentage_is_15_percent

end cobalt_percentage_is_15_percent_l58_5813


namespace basketball_league_female_fraction_l58_5892

theorem basketball_league_female_fraction :
  -- Define variables
  let male_last_year : ℕ := 30
  let total_increase_rate : ℚ := 115 / 100
  let male_increase_rate : ℚ := 110 / 100
  let female_increase_rate : ℚ := 125 / 100

  -- Calculate values
  let male_this_year : ℚ := male_last_year * male_increase_rate
  let female_last_year : ℚ := (total_increase_rate * (male_last_year : ℚ) - male_this_year) / (female_increase_rate - total_increase_rate)
  let female_this_year : ℚ := female_last_year * female_increase_rate
  let total_this_year : ℚ := male_this_year + female_this_year

  -- Prove the fraction
  female_this_year / total_this_year = 25 / 69 := by sorry

end basketball_league_female_fraction_l58_5892


namespace roots_of_unity_cubic_equation_l58_5810

theorem roots_of_unity_cubic_equation :
  ∃ (c d : ℤ), ∃ (roots : Finset ℂ),
    (∀ z ∈ roots, z^3 = 1) ∧
    (∀ z ∈ roots, z^3 + c*z + d = 0) ∧
    (roots.card = 3) ∧
    (∀ z : ℂ, z^3 = 1 → z^3 + c*z + d = 0 → z ∈ roots) :=
by sorry

end roots_of_unity_cubic_equation_l58_5810


namespace complex_magnitude_equation_l58_5898

theorem complex_magnitude_equation (n : ℝ) :
  n > 0 → (Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 6 ↔ n = 5 * Real.sqrt 5) := by
  sorry

end complex_magnitude_equation_l58_5898


namespace min_coach_handshakes_l58_5819

/-- The total number of handshakes -/
def total_handshakes : ℕ := 325

/-- The number of gymnasts -/
def n : ℕ := 26

/-- The number of handshakes between gymnasts -/
def gymnast_handshakes : ℕ := n * (n - 1) / 2

/-- The number of handshakes by the first coach -/
def coach1_handshakes : ℕ := 0

/-- The number of handshakes by the second coach -/
def coach2_handshakes : ℕ := total_handshakes - gymnast_handshakes - coach1_handshakes

theorem min_coach_handshakes :
  gymnast_handshakes + coach1_handshakes + coach2_handshakes = total_handshakes ∧
  coach1_handshakes = 0 ∧
  coach2_handshakes ≥ 0 := by
  sorry

end min_coach_handshakes_l58_5819


namespace eva_second_semester_maths_score_l58_5830

/-- Represents Eva's scores in a semester -/
structure SemesterScores where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Calculates the total score for a semester -/
def totalScore (scores : SemesterScores) : ℕ :=
  scores.maths + scores.arts + scores.science

theorem eva_second_semester_maths_score :
  ∀ (first second : SemesterScores),
    first.maths = second.maths + 10 →
    first.arts + 15 = second.arts →
    first.science + (first.science / 3) = second.science →
    second.arts = 90 →
    second.science = 90 →
    totalScore first + totalScore second = 485 →
    second.maths = 80 := by
  sorry

end eva_second_semester_maths_score_l58_5830


namespace disk_color_difference_l58_5876

/-- Given a bag of disks with a specific color ratio and total count, 
    calculate the difference between green and blue disks. -/
theorem disk_color_difference 
  (total_disks : ℕ) 
  (blue_ratio yellow_ratio green_ratio red_ratio : ℕ) 
  (h_total : total_disks = 132)
  (h_ratio : blue_ratio + yellow_ratio + green_ratio + red_ratio = 22)
  (h_blue : blue_ratio = 3)
  (h_yellow : yellow_ratio = 7)
  (h_green : green_ratio = 8)
  (h_red : red_ratio = 4) :
  green_ratio * (total_disks / (blue_ratio + yellow_ratio + green_ratio + red_ratio)) -
  blue_ratio * (total_disks / (blue_ratio + yellow_ratio + green_ratio + red_ratio)) = 30 :=
by sorry

end disk_color_difference_l58_5876


namespace line_parallel_to_plane_l58_5893

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "contained in" relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (α β : Plane) (m : Line) 
  (h1 : parallel_planes α β) 
  (h2 : contained_in m β) : 
  parallel_line_plane m α :=
sorry

end line_parallel_to_plane_l58_5893


namespace function_period_l58_5840

/-- Given a constant a and a function f: ℝ → ℝ that satisfies
    f(x) = (f(x-a) - 1) / (f(x-a) + 1) for all x ∈ ℝ,
    prove that f has period 4a. -/
theorem function_period (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x, f x = (f (x - a) - 1) / (f (x - a) + 1)) :
  ∀ x, f (x + 4*a) = f x := by
  sorry

end function_period_l58_5840


namespace not_necessary_not_sufficient_l58_5872

-- Define the quadratic polynomials
def quadratic_poly (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution sets
def solution_set (a b c : ℝ) : Set ℝ := {x | quadratic_poly a b c x > 0}

-- Define the condition for equal ratios
def equal_ratios (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂

-- State the theorem
theorem not_necessary_not_sufficient
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  ¬(equal_ratios a₁ b₁ c₁ a₂ b₂ c₂ ↔ solution_set a₁ b₁ c₁ = solution_set a₂ b₂ c₂) :=
sorry

end not_necessary_not_sufficient_l58_5872


namespace no_solution_exists_l58_5828

theorem no_solution_exists : ¬∃ x : ℝ, (x / (-4) ≥ 3 + x) ∧ (|2 * x - 1| < 4 + 2 * x) := by
  sorry

end no_solution_exists_l58_5828


namespace trigonometric_simplification_l58_5808

theorem trigonometric_simplification (α : ℝ) : 
  Real.sin (π / 2 + α) * Real.cos (α - π / 3) + Real.sin (π - α) * Real.sin (α - π / 3) = 1 / 2 := by
  sorry

end trigonometric_simplification_l58_5808


namespace piglet_straws_l58_5886

theorem piglet_straws (total_straws : ℕ) (adult_pig_fraction : ℚ) (num_piglets : ℕ) :
  total_straws = 300 →
  adult_pig_fraction = 3/5 →
  num_piglets = 20 →
  (adult_pig_fraction * total_straws : ℚ) = (total_straws - adult_pig_fraction * total_straws : ℚ) →
  (total_straws - adult_pig_fraction * total_straws) / num_piglets = 9 :=
by
  sorry

end piglet_straws_l58_5886


namespace pyarelal_loss_is_900_l58_5854

/-- Calculates Pyarelal's share of the loss given the investment ratio and total loss -/
def pyarelal_loss (pyarelal_capital : ℚ) (total_loss : ℚ) : ℚ :=
  let ashok_capital := (1 : ℚ) / 9 * pyarelal_capital
  let total_capital := ashok_capital + pyarelal_capital
  let pyarelal_ratio := pyarelal_capital / total_capital
  pyarelal_ratio * total_loss

/-- Theorem stating that Pyarelal's loss is 900 given the conditions of the problem -/
theorem pyarelal_loss_is_900 (pyarelal_capital : ℚ) (h : pyarelal_capital > 0) :
  pyarelal_loss pyarelal_capital 1000 = 900 := by
  sorry

end pyarelal_loss_is_900_l58_5854


namespace favorite_numbers_sum_l58_5823

/-- Given the favorite numbers of Misty, Glory, and Dawn, prove their sum is 1500 -/
theorem favorite_numbers_sum (glory_fav : ℕ) (misty_fav : ℕ) (dawn_fav : ℕ) 
  (h1 : glory_fav = 450)
  (h2 : misty_fav * 3 = glory_fav)
  (h3 : dawn_fav = glory_fav * 2) :
  misty_fav + glory_fav + dawn_fav = 1500 := by
  sorry

end favorite_numbers_sum_l58_5823


namespace rahims_book_purchase_l58_5853

/-- Given Rahim's book purchases, prove the amount paid for books from the first shop -/
theorem rahims_book_purchase (first_shop_books : ℕ) (second_shop_books : ℕ) 
  (second_shop_cost : ℕ) (average_price : ℕ) (h1 : first_shop_books = 42) 
  (h2 : second_shop_books = 22) (h3 : second_shop_cost = 248) (h4 : average_price = 12) :
  (first_shop_books + second_shop_books) * average_price - second_shop_cost = 520 := by
  sorry

#check rahims_book_purchase

end rahims_book_purchase_l58_5853


namespace school_picnic_attendees_l58_5847

/-- The number of attendees at the school picnic. -/
def num_attendees : ℕ := 1006

/-- The total number of plates prepared by the school. -/
def total_plates : ℕ := 2015 - num_attendees

theorem school_picnic_attendees :
  (∀ n : ℕ, n ≤ num_attendees → total_plates - (n - 1) > 0) ∧
  (total_plates - (num_attendees - 1) = 4) ∧
  (num_attendees + total_plates = 2015) :=
sorry

end school_picnic_attendees_l58_5847


namespace three_random_events_l58_5839

/-- Represents an event that can occur in a probability space. -/
structure Event where
  description : String
  is_random : Bool

/-- The set of events we're considering. -/
def events : List Event := [
  ⟨"Selecting 3 out of 10 glass cups (8 good quality, 2 defective), all 3 selected are good quality", true⟩,
  ⟨"Randomly pressing a digit and it happens to be correct when forgetting the last digit of a phone number", true⟩,
  ⟨"Opposite electric charges attract each other", false⟩,
  ⟨"A person wins the first prize in a sports lottery", true⟩
]

/-- Counts the number of random events in a list of events. -/
def countRandomEvents (events : List Event) : Nat :=
  events.filter (·.is_random) |>.length

/-- The main theorem stating that exactly three of the given events are random. -/
theorem three_random_events : countRandomEvents events = 3 := by
  sorry

end three_random_events_l58_5839


namespace complex_number_properties_l58_5883

theorem complex_number_properties (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  ((a + b)^2 = a^2 + 2*a*b + b^2) ∧ 
  (a^2 = a*b → a = b) ∧ 
  (∃ c : ℂ, c ≠ 0 ∧ c + 1/c = 0) ∧
  (∃ x y : ℂ, Complex.abs x = Complex.abs y ∧ x ≠ y ∧ x ≠ -y) :=
by sorry

end complex_number_properties_l58_5883


namespace flare_problem_l58_5859

-- Define the height function
def h (v : ℝ) (t : ℝ) : ℝ := v * t - 4.9 * t^2

-- State the theorem
theorem flare_problem (v : ℝ) :
  h v 5 = 245 →
  v = 73.5 ∧
  ∃ t1 t2 : ℝ, t1 = 5 ∧ t2 = 10 ∧ ∀ t, t1 < t ∧ t < t2 → h v t > 245 :=
by sorry

end flare_problem_l58_5859


namespace maria_car_trip_l58_5878

/-- Proves that the fraction of remaining distance traveled between first and second stops is 1/4 -/
theorem maria_car_trip (total_distance : ℝ) (remaining_after_second : ℝ) 
  (h1 : total_distance = 280)
  (h2 : remaining_after_second = 105) :
  let first_stop := total_distance / 2
  let remaining_after_first := total_distance - first_stop
  let distance_between_stops := remaining_after_first - remaining_after_second
  distance_between_stops / remaining_after_first = 1 / 4 := by
  sorry

end maria_car_trip_l58_5878


namespace product_of_sum_and_cube_sum_l58_5879

theorem product_of_sum_and_cube_sum (x y : ℝ) 
  (h1 : x + y = 9) 
  (h2 : x^3 + y^3 = 351) : 
  x * y = 14 := by
sorry

end product_of_sum_and_cube_sum_l58_5879


namespace hex_tile_difference_l58_5866

/-- Represents the number of tiles in a hexagonal arrangement --/
structure HexTileArrangement where
  blue : ℕ
  green : ℕ

/-- Calculates the number of tiles needed for a border around a hexagonal arrangement --/
def border_tiles (side_length : ℕ) : ℕ := 6 * side_length

/-- Adds a border of green tiles to an existing arrangement --/
def add_border (arrangement : HexTileArrangement) (border_size : ℕ) : HexTileArrangement :=
  { blue := arrangement.blue,
    green := arrangement.green + border_tiles border_size }

/-- The main theorem to prove --/
theorem hex_tile_difference :
  let initial := HexTileArrangement.mk 12 8
  let first_border := add_border initial 3
  let second_border := add_border first_border 4
  second_border.green - second_border.blue = 38 := by sorry

end hex_tile_difference_l58_5866


namespace student_fails_by_10_marks_l58_5814

/-- Calculates the number of marks a student fails by in a test -/
def marksFailed (maxMarks : ℕ) (passingPercentage : ℚ) (studentScore : ℕ) : ℕ :=
  let passingMark := (maxMarks : ℚ) * passingPercentage
  (passingMark.ceil - studentScore).toNat

/-- Proves that a student who scores 80 marks in a 300-mark test with 30% passing requirement fails by 10 marks -/
theorem student_fails_by_10_marks :
  marksFailed 300 (30 / 100) 80 = 10 := by
  sorry

end student_fails_by_10_marks_l58_5814


namespace factor_difference_of_squares_l58_5865

theorem factor_difference_of_squares (x : ℝ) : 4 * x^2 - 144 = 4 * (x - 6) * (x + 6) := by
  sorry

end factor_difference_of_squares_l58_5865


namespace intersection_of_A_and_B_l58_5815

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {-1, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l58_5815


namespace quadratic_root_implies_a_l58_5851

theorem quadratic_root_implies_a (a : ℝ) : 
  let S := {x : ℝ | x^2 + 2*x + a = 0}
  (-1 : ℝ) ∈ S → a = 1 := by
sorry

end quadratic_root_implies_a_l58_5851


namespace total_trees_after_planting_l58_5837

def current_trees : ℕ := 33
def new_trees : ℕ := 44

theorem total_trees_after_planting :
  current_trees + new_trees = 77 := by sorry

end total_trees_after_planting_l58_5837


namespace equation_solution_l58_5874

theorem equation_solution : ∃! x : ℝ, (2 / (x - 3) = 3 / (x - 6)) ∧ x = -3 := by
  sorry

end equation_solution_l58_5874


namespace quadratic_equations_common_root_l58_5880

theorem quadratic_equations_common_root (k : ℝ) :
  (∃ x : ℝ, x^2 + k*x - 12 = 0 ∧ 3*x^2 - 8*x - 3*k = 0) →
  k = 1 :=
by sorry

end quadratic_equations_common_root_l58_5880


namespace range_of_m_l58_5803

-- Define the conditions
def condition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m*x + 3/2 > 0

def condition_q (m : ℝ) : Prop :=
  m > 1 ∧ m < 3 -- Simplified condition for foci on x-axis

-- Define the theorem
theorem range_of_m (m : ℝ) :
  condition_p m ∧ condition_q m → 2 < m ∧ m < Real.sqrt 6 :=
sorry

end range_of_m_l58_5803


namespace math_team_combinations_l58_5857

/-- The number of girls in the math club -/
def num_girls : ℕ := 4

/-- The number of boys in the math club -/
def num_boys : ℕ := 6

/-- The number of girls to be selected for the team -/
def girls_in_team : ℕ := 3

/-- The number of boys to be selected for the team -/
def boys_in_team : ℕ := 2

/-- The total number of possible team combinations -/
def total_combinations : ℕ := 60

theorem math_team_combinations :
  (Nat.choose num_girls girls_in_team) * (Nat.choose num_boys boys_in_team) = total_combinations :=
sorry

end math_team_combinations_l58_5857


namespace shop_markup_problem_l58_5858

/-- A shop owner purchases goods at a discount and wants to mark them up for profit. -/
theorem shop_markup_problem (L : ℝ) (C : ℝ) (M : ℝ) (S : ℝ) :
  C = 0.75 * L →           -- Cost price is 75% of list price
  S = 1.3 * C →            -- Selling price is 130% of cost price
  S = 0.75 * M →           -- Selling price is 75% of marked price
  M = 1.3 * L              -- Marked price is 130% of list price
:= by sorry

end shop_markup_problem_l58_5858


namespace second_wood_weight_l58_5818

/-- Represents a square piece of wood -/
structure Wood where
  side : ℝ
  weight : ℝ

/-- The weight of a square piece of wood is proportional to its area -/
axiom weight_prop_area {w1 w2 : Wood} :
  w1.weight / w2.weight = (w1.side ^ 2) / (w2.side ^ 2)

/-- Given two pieces of wood with specific properties, prove the weight of the second piece -/
theorem second_wood_weight (w1 w2 : Wood)
  (h1 : w1.side = 4)
  (h2 : w1.weight = 16)
  (h3 : w2.side = 6) :
  w2.weight = 36 := by
  sorry

#check second_wood_weight

end second_wood_weight_l58_5818


namespace not_p_necessary_not_sufficient_for_not_q_l58_5863

-- Define propositions p and q
def p (x : ℝ) : Prop := abs x < 1
def q (x : ℝ) : Prop := x^2 + x - 6 < 0

-- Theorem statement
theorem not_p_necessary_not_sufficient_for_not_q :
  (∀ x, ¬(q x) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ q x) :=
sorry

end not_p_necessary_not_sufficient_for_not_q_l58_5863


namespace token_game_1994_token_game_1991_l58_5809

/-- Represents the state of the token-passing game -/
structure GameState (N : ℕ) where
  tokens : Fin N → ℕ
  total_tokens : ℕ

/-- Defines a single move in the game -/
def move (state : GameState N) (i : Fin N) : GameState N :=
  sorry

/-- Determines if the game has terminated -/
def is_terminated (state : GameState N) : Prop :=
  ∀ i, state.tokens i ≤ 1

/-- Theorem for the token-passing game with 1994 girls -/
theorem token_game_1994 (n : ℕ) :
  (n < 1994 → ∃ (final_state : GameState 1994), is_terminated final_state) ∧
  (n = 1994 → ¬∃ (final_state : GameState 1994), is_terminated final_state) :=
  sorry

/-- Theorem for the token-passing game with 1991 girls -/
theorem token_game_1991 (n : ℕ) :
  n ≤ 1991 → ¬∃ (final_state : GameState 1991), is_terminated final_state :=
  sorry

end token_game_1994_token_game_1991_l58_5809


namespace apples_on_table_l58_5867

/-- The number of green apples on the table -/
def green_apples : ℕ := 2

/-- The number of red apples on the table -/
def red_apples : ℕ := 3

/-- The number of yellow apples on the table -/
def yellow_apples : ℕ := 14

/-- The total number of apples on the table -/
def total_apples : ℕ := green_apples + red_apples + yellow_apples

theorem apples_on_table : total_apples = 19 := by
  sorry

end apples_on_table_l58_5867


namespace q_value_proof_l58_5822

theorem q_value_proof (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1/p + 1/q = 3/2) 
  (h4 : p * q = 12) : 
  q = 9 + 3 * Real.sqrt 23 := by
sorry

end q_value_proof_l58_5822


namespace chromatic_number_lower_bound_l58_5875

/-- A simple graph -/
structure Graph (V : Type*) where
  adj : V → V → Prop

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- The maximum size of cliques in a graph -/
def omega (G : Graph V) : ℕ :=
  sorry

/-- The maximum size of independent sets in a graph -/
def omegaBar (G : Graph V) : ℕ :=
  sorry

/-- The chromatic number of a graph -/
def chromaticNumber (G : Graph V) : ℕ :=
  sorry

/-- The main theorem -/
theorem chromatic_number_lower_bound (G : Graph V) :
  chromaticNumber G ≥ max (omega G) (Fintype.card V / omegaBar G) :=
sorry

end chromatic_number_lower_bound_l58_5875


namespace percentage_of_seniors_with_cars_l58_5845

theorem percentage_of_seniors_with_cars :
  ∀ (total_students : ℕ) 
    (seniors : ℕ) 
    (lower_grades : ℕ) 
    (lower_grades_car_percentage : ℚ) 
    (total_car_percentage : ℚ),
  total_students = 1200 →
  seniors = 300 →
  lower_grades = 900 →
  lower_grades_car_percentage = 1/10 →
  total_car_percentage = 1/5 →
  (↑seniors * (seniors_car_percentage : ℚ) + ↑lower_grades * lower_grades_car_percentage) / ↑total_students = total_car_percentage →
  seniors_car_percentage = 1/2 :=
by
  sorry

#check percentage_of_seniors_with_cars

end percentage_of_seniors_with_cars_l58_5845


namespace prob_six_given_hugo_wins_l58_5882

/-- The number of players in the game -/
def num_players : ℕ := 5

/-- The number of sides on each die -/
def die_sides : ℕ := 6

/-- The probability of rolling a 6 on a single die -/
def prob_roll_six : ℚ := 1 / die_sides

/-- The probability of Hugo winning the game -/
def prob_hugo_wins : ℚ := 1 / num_players

/-- The probability that Hugo wins given his first roll was a 6 -/
noncomputable def prob_hugo_wins_given_six : ℚ := 875 / 1296

/-- Theorem: The probability that Hugo's first roll was 6, given that he won the game -/
theorem prob_six_given_hugo_wins :
  (prob_roll_six * prob_hugo_wins_given_six) / prob_hugo_wins = 4375 / 7776 := by sorry

end prob_six_given_hugo_wins_l58_5882


namespace distribute_four_balls_four_boxes_l58_5802

/-- The number of ways to distribute indistinguishable objects into distinguishable containers -/
def distribute_objects (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 32 ways to distribute 4 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_four_balls_four_boxes : distribute_objects 4 4 = 32 := by
  sorry

end distribute_four_balls_four_boxes_l58_5802


namespace area_circle_inscribed_square_l58_5800

/-- The area of a circle inscribed in a square with diagonal 10 meters is 12.5π square meters. -/
theorem area_circle_inscribed_square (d : ℝ) (A : ℝ) :
  d = 10 → A = π * (d / (2 * Real.sqrt 2))^2 → A = 12.5 * π := by
  sorry

end area_circle_inscribed_square_l58_5800


namespace soccer_game_analysis_l58_5884

-- Define the players
inductive Player : Type
| Amandine : Player
| Bobby : Player
| Charles : Player

-- Define the game structure
structure Game where
  total_phases : ℕ
  amandine_field : ℕ
  bobby_field : ℕ
  charles_goalkeeper : ℕ

-- Define the theorem
theorem soccer_game_analysis (g : Game) 
  (h1 : g.amandine_field = 12)
  (h2 : g.bobby_field = 21)
  (h3 : g.charles_goalkeeper = 8)
  (h4 : g.total_phases = g.amandine_field + (g.total_phases - g.amandine_field))
  (h5 : g.total_phases = g.bobby_field + (g.total_phases - g.bobby_field))
  (h6 : g.total_phases = (g.total_phases - g.charles_goalkeeper) + g.charles_goalkeeper) :
  g.total_phases = 25 ∧ (∃ n : ℕ, n = 6 ∧ n % 2 = 0 ∧ (n + 1) ≤ g.total_phases - g.amandine_field) := by
  sorry


end soccer_game_analysis_l58_5884
