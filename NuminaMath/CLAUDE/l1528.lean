import Mathlib

namespace binomial_expansion_coefficient_l1528_152875

theorem binomial_expansion_coefficient (p : ℝ) : 
  (∃ k : ℕ, Nat.choose 5 k * p^k = 80 ∧ 2*k = 6) → p = 2 := by
  sorry

end binomial_expansion_coefficient_l1528_152875


namespace lcm_gcf_ratio_280_450_l1528_152866

theorem lcm_gcf_ratio_280_450 : Nat.lcm 280 450 / Nat.gcd 280 450 = 1260 := by
  sorry

end lcm_gcf_ratio_280_450_l1528_152866


namespace dividend_calculation_l1528_152824

/-- Calculates the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) :
  investment = 14400 ∧ 
  face_value = 100 ∧ 
  premium_rate = 0.20 ∧ 
  dividend_rate = 0.05 →
  (investment / (face_value * (1 + premium_rate))) * (face_value * dividend_rate) = 600 := by
  sorry

end dividend_calculation_l1528_152824


namespace two_unusual_numbers_l1528_152804

/-- A number is unusual if it satisfies the given conditions --/
def IsUnusual (n : ℕ) : Prop :=
  10^99 ≤ n ∧ n < 10^100 ∧ 
  n^3 % 10^100 = n % 10^100 ∧ 
  n^2 % 10^100 ≠ n % 10^100

/-- There exist at least two distinct unusual numbers --/
theorem two_unusual_numbers : ∃ n₁ n₂ : ℕ, IsUnusual n₁ ∧ IsUnusual n₂ ∧ n₁ ≠ n₂ := by
  sorry

end two_unusual_numbers_l1528_152804


namespace candy_per_box_l1528_152801

/-- Given that Billy bought 7 boxes of candy and had a total of 21 pieces,
    prove that each box contained 3 pieces of candy. -/
theorem candy_per_box (num_boxes : ℕ) (total_pieces : ℕ) (h1 : num_boxes = 7) (h2 : total_pieces = 21) :
  total_pieces / num_boxes = 3 := by
sorry

end candy_per_box_l1528_152801


namespace intersection_complement_equality_l1528_152860

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {3, 4, 5}

theorem intersection_complement_equality : M ∩ (U \ N) = {1, 2} := by
  sorry

end intersection_complement_equality_l1528_152860


namespace function_properties_l1528_152893

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function is monotonically increasing on an interval [a, b] if
    for all x, y in [a, b], x ≤ y implies f(x) ≤ f(y) -/
def MonoIncOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

/-- Main theorem -/
theorem function_properties (f : ℝ → ℝ) 
    (heven : IsEven f)
    (hmono : MonoIncOn f (-1) 0)
    (hcond : ∀ x, f (1 - x) + f (1 + x) = 0) :
    (f (-3) = 0) ∧
    (MonoIncOn f 1 2) ∧
    (∀ x, f x = f (2 - x)) := by
  sorry


end function_properties_l1528_152893


namespace increase_in_average_marks_l1528_152805

/-- Proves that the increase in average marks is 0.5 when a mark is incorrectly entered as 67 instead of 45 in a class of 44 pupils. -/
theorem increase_in_average_marks 
  (num_pupils : ℕ) 
  (wrong_mark : ℕ) 
  (correct_mark : ℕ) 
  (h1 : num_pupils = 44) 
  (h2 : wrong_mark = 67) 
  (h3 : correct_mark = 45) : 
  (wrong_mark - correct_mark : ℚ) / num_pupils = 1/2 := by
  sorry

end increase_in_average_marks_l1528_152805


namespace count_special_primes_l1528_152881

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def swap_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

def is_special_prime (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ is_prime (swap_digits n)

theorem count_special_primes : 
  (∃ (s : Finset ℕ), (∀ n ∈ s, is_special_prime n) ∧ s.card = 9 ∧ 
   (∀ m : ℕ, is_special_prime m → m ∈ s)) :=
sorry

end count_special_primes_l1528_152881


namespace fifteen_balls_four_draws_l1528_152809

/-- The number of ways to draw n balls in order from a bin of m balls,
    where each ball remains outside the bin after it is drawn. -/
def orderedDraw (m n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * (m - i)) 1

/-- The problem statement -/
theorem fifteen_balls_four_draws :
  orderedDraw 15 4 = 32760 := by
  sorry

end fifteen_balls_four_draws_l1528_152809


namespace cone_cut_ratio_sum_l1528_152892

/-- Represents a right circular cone --/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the result of cutting a cone parallel to its base --/
structure CutCone where
  originalCone : Cone
  cutRadius : ℝ

def surfaceAreaRatio (cc : CutCone) : ℝ := sorry

def volumeRatio (cc : CutCone) : ℝ := sorry

def isCoprime (m n : ℕ) : Prop := sorry

theorem cone_cut_ratio_sum (m n : ℕ) :
  let originalCone : Cone := { height := 6, baseRadius := 5 }
  let cc : CutCone := { originalCone := originalCone, cutRadius := 25/8 }
  surfaceAreaRatio cc = m / n →
  volumeRatio cc = m / n →
  isCoprime m n →
  m + n = 20 := by sorry

end cone_cut_ratio_sum_l1528_152892


namespace square_rectangle_area_relation_l1528_152876

theorem square_rectangle_area_relation :
  ∀ x : ℝ,
  let square_side : ℝ := x - 3
  let rect_length : ℝ := x - 4
  let rect_width : ℝ := x + 5
  let square_area : ℝ := square_side ^ 2
  let rect_area : ℝ := rect_length * rect_width
  (rect_area = 3 * square_area) →
  (∃ y : ℝ, y ≠ x ∧ 
    let square_side' : ℝ := y - 3
    let rect_length' : ℝ := y - 4
    let rect_width' : ℝ := y + 5
    let square_area' : ℝ := square_side' ^ 2
    let rect_area' : ℝ := rect_length' * rect_width'
    (rect_area' = 3 * square_area')) →
  x + y = 7 :=
by sorry

end square_rectangle_area_relation_l1528_152876


namespace dogwood_tree_count_l1528_152819

theorem dogwood_tree_count (current_trees planted_trees : ℕ) : 
  current_trees = 34 → planted_trees = 49 → current_trees + planted_trees = 83 := by
  sorry

end dogwood_tree_count_l1528_152819


namespace john_bought_three_tshirts_l1528_152834

/-- The number of t-shirts John bought -/
def num_tshirts : ℕ := 3

/-- The cost of each t-shirt in dollars -/
def tshirt_cost : ℕ := 20

/-- The amount spent on pants in dollars -/
def pants_cost : ℕ := 50

/-- The total amount spent in dollars -/
def total_spent : ℕ := 110

theorem john_bought_three_tshirts :
  num_tshirts * tshirt_cost + pants_cost = total_spent :=
sorry

end john_bought_three_tshirts_l1528_152834


namespace dinner_payment_difference_l1528_152890

/-- Calculates the difference in payment between John and Jane for a dinner --/
theorem dinner_payment_difference (original_price : ℝ) (discount_percent : ℝ) 
  (tip_percent : ℝ) (h1 : original_price = 40) (h2 : discount_percent = 0.1) 
  (h3 : tip_percent = 0.15) : 
  let discounted_price := original_price * (1 - discount_percent)
  let john_payment := discounted_price + original_price * tip_percent
  let jane_payment := discounted_price + discounted_price * tip_percent
  john_payment - jane_payment = 0.6 := by sorry

end dinner_payment_difference_l1528_152890


namespace cuboid_volume_is_48_l1528_152826

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cuboid -/
def surfaceArea (d : CuboidDimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Calculates the volume of a cuboid -/
def volume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem stating the volume of a specific cuboid -/
theorem cuboid_volume_is_48 :
  ∃ (d : CuboidDimensions),
    d.length = 2 * d.width ∧
    d.height = 3 * d.width ∧
    surfaceArea d = 88 ∧
    volume d = 48 := by
  sorry

end cuboid_volume_is_48_l1528_152826


namespace nth_equation_proof_l1528_152853

theorem nth_equation_proof (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 := by
  sorry

end nth_equation_proof_l1528_152853


namespace committee_formation_count_l1528_152898

/-- Represents a department in the division of science -/
inductive Department
| physics
| chemistry
| biology
| mathematics

/-- The number of departments in the division -/
def num_departments : Nat := 4

/-- The number of male professors in each department -/
def male_professors_per_dept : Nat := 3

/-- The number of female professors in each department -/
def female_professors_per_dept : Nat := 3

/-- The total number of professors in the committee -/
def committee_size : Nat := 8

/-- The number of male professors in the committee -/
def male_committee_members : Nat := 4

/-- The number of female professors in the committee -/
def female_committee_members : Nat := 4

/-- The number of departments contributing exactly two professors -/
def depts_with_two_profs : Nat := 2

/-- The number of departments contributing one male and one female professor -/
def depts_with_one_each : Nat := 2

/-- The number of ways to form the committee under the given conditions -/
def committee_formation_ways : Nat := 48114

theorem committee_formation_count :
  (num_departments = 4) →
  (male_professors_per_dept = 3) →
  (female_professors_per_dept = 3) →
  (committee_size = 8) →
  (male_committee_members = 4) →
  (female_committee_members = 4) →
  (depts_with_two_profs = 2) →
  (depts_with_one_each = 2) →
  (committee_formation_ways = 48114) := by
  sorry


end committee_formation_count_l1528_152898


namespace equation_solution_l1528_152847

theorem equation_solution (x : ℤ) : 9*x + 2 ≡ 7 [ZMOD 15] ↔ x ≡ 10 [ZMOD 15] := by
  sorry

end equation_solution_l1528_152847


namespace trig_product_equals_one_sixteenth_l1528_152840

theorem trig_product_equals_one_sixteenth : 
  Real.cos (15 * π / 180) * Real.sin (30 * π / 180) * 
  Real.cos (75 * π / 180) * Real.sin (150 * π / 180) = 1 / 16 := by
  sorry

end trig_product_equals_one_sixteenth_l1528_152840


namespace shaded_area_of_square_with_removed_triangles_l1528_152842

/-- The area of a shape formed by removing four right triangles from a square -/
theorem shaded_area_of_square_with_removed_triangles 
  (square_side : ℝ) 
  (triangle_leg : ℝ) 
  (h1 : square_side = 6) 
  (h2 : triangle_leg = 2) : 
  square_side ^ 2 - 4 * (1 / 2 * triangle_leg ^ 2) = 28 := by
  sorry

end shaded_area_of_square_with_removed_triangles_l1528_152842


namespace group_size_problem_l1528_152879

/-- Given a group where each member contributes as many paise as there are members,
    and the total collection is 5929 paise, prove that the number of members is 77. -/
theorem group_size_problem (n : ℕ) (h1 : n * n = 5929) : n = 77 := by
  sorry

end group_size_problem_l1528_152879


namespace ann_boxes_sold_l1528_152839

theorem ann_boxes_sold (n : ℕ) (mark_sold ann_sold : ℕ) : 
  n = 12 →
  mark_sold = n - 11 →
  ann_sold < n →
  mark_sold ≥ 1 →
  ann_sold ≥ 1 →
  mark_sold + ann_sold < n →
  ann_sold = n - 2 :=
by sorry

end ann_boxes_sold_l1528_152839


namespace replaced_crew_weight_l1528_152856

/-- Proves that the replaced crew member weighs 40 kg given the conditions of the problem -/
theorem replaced_crew_weight (n : ℕ) (old_avg new_avg new_weight : ℝ) :
  n = 20 ∧
  new_avg = old_avg + 2 ∧
  new_weight = 80 →
  n * new_avg - (n - 1) * old_avg = 40 :=
by sorry

end replaced_crew_weight_l1528_152856


namespace last_three_digits_of_5_to_1999_l1528_152850

theorem last_three_digits_of_5_to_1999 : 5^1999 ≡ 125 [ZMOD 1000] := by
  sorry

end last_three_digits_of_5_to_1999_l1528_152850


namespace product_testing_theorem_l1528_152807

/-- The number of ways to choose k items from n items, where order matters and repetition is not allowed. -/
def A (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of products -/
def total_products : ℕ := 10

/-- The number of defective products -/
def defective_products : ℕ := 4

/-- The number of ways to find 4 defective products among 10 products, 
    where the first defective is found on the 2nd measurement and the last on the 8th -/
def ways_specific_case : ℕ := A 4 2 * A 5 2 * A 6 4

/-- The number of ways to find 4 defective products among 10 products in at most 6 measurements -/
def ways_at_most_6 : ℕ := A 4 4 + 4 * A 4 3 * A 6 1 + 4 * A 5 3 * A 6 2 + A 6 6

theorem product_testing_theorem :
  (ways_specific_case = A 4 2 * A 5 2 * A 6 4) ∧
  (ways_at_most_6 = A 4 4 + 4 * A 4 3 * A 6 1 + 4 * A 5 3 * A 6 2 + A 6 6) :=
sorry

end product_testing_theorem_l1528_152807


namespace set_intersection_equality_l1528_152846

def M : Set ℤ := {1, 2, 3}
def N : Set ℤ := {x : ℤ | 1 < x ∧ x < 4}

theorem set_intersection_equality : M ∩ N = {2, 3} := by sorry

end set_intersection_equality_l1528_152846


namespace parabola_fixed_point_l1528_152899

theorem parabola_fixed_point (u : ℝ) : 
  let f : ℝ → ℝ := λ x => 5 * x^2 + u * x + 3 * u
  f (-3) = 45 := by
sorry

end parabola_fixed_point_l1528_152899


namespace container_water_problem_l1528_152813

theorem container_water_problem (x y : ℝ) : 
  x > 0 ∧ y > 0 → -- Containers and total masses are positive
  (4 / 5 * y - x) + (y - x) = 8 * x → -- Pouring water from B to A
  y - x - (4 / 5 * y - x) = 50 → -- B has 50g more water than A
  x = 50 ∧ 4 / 5 * y - x = 150 ∧ y - x = 200 := by
sorry

end container_water_problem_l1528_152813


namespace probability_at_least_one_defective_l1528_152825

/-- The probability of selecting at least one defective bulb when choosing 2 bulbs at random from a box containing 20 bulbs, of which 4 are defective, is 7/19. -/
theorem probability_at_least_one_defective (total_bulbs : Nat) (defective_bulbs : Nat) 
    (h1 : total_bulbs = 20) 
    (h2 : defective_bulbs = 4) : 
  let p := 1 - (total_bulbs - defective_bulbs : ℚ) * (total_bulbs - defective_bulbs - 1) / 
           (total_bulbs * (total_bulbs - 1))
  p = 7 / 19 := by
  sorry

end probability_at_least_one_defective_l1528_152825


namespace congruence_system_solution_l1528_152851

theorem congruence_system_solution :
  ∃ x : ℤ, (x ≡ 1 [ZMOD 6] ∧ x ≡ 9 [ZMOD 14] ∧ x ≡ 7 [ZMOD 15]) ↔ x ≡ 37 [ZMOD 210] :=
by sorry

end congruence_system_solution_l1528_152851


namespace spinner_probability_l1528_152848

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_C = 1/6 → p_A + p_B + p_C + p_D = 1 → p_D = 1/4 := by
  sorry

end spinner_probability_l1528_152848


namespace specific_right_triangle_l1528_152833

/-- A right triangle with specific side lengths -/
structure RightTriangle where
  -- The length of the hypotenuse
  ab : ℝ
  -- The length of one of the other sides
  ac : ℝ
  -- The length of the remaining side
  bc : ℝ
  -- Constraint that this is a right triangle (Pythagorean theorem)
  pythagorean : ab ^ 2 = ac ^ 2 + bc ^ 2

/-- Theorem: In a right triangle with hypotenuse 5 and one side 4, the other side is 3 -/
theorem specific_right_triangle :
  ∃ (t : RightTriangle), t.ab = 5 ∧ t.ac = 4 ∧ t.bc = 3 := by
  sorry


end specific_right_triangle_l1528_152833


namespace sum_of_roots_product_polynomials_l1528_152897

theorem sum_of_roots_product_polynomials :
  let p₁ : Polynomial ℝ := 3 * X^3 + 3 * X^2 - 9 * X + 27
  let p₂ : Polynomial ℝ := 4 * X^3 - 16 * X^2 + 5
  (p₁ * p₂).roots.sum = 3 := by
  sorry

end sum_of_roots_product_polynomials_l1528_152897


namespace expression_simplification_l1528_152823

theorem expression_simplification :
  let a : ℚ := 3 / 2015
  let b : ℚ := 11 / 2016
  (6 + a) * (8 + b) - (11 - a) * (3 - b) - 12 * a = 11 / 112 := by sorry

end expression_simplification_l1528_152823


namespace wendy_sweaters_l1528_152889

/-- Represents the number of pieces of clothing a washing machine can wash in one load. -/
def machine_capacity : ℕ := 8

/-- Represents the number of shirts Wendy has to wash. -/
def num_shirts : ℕ := 39

/-- Represents the total number of loads Wendy has to do. -/
def total_loads : ℕ := 9

/-- Calculates the number of sweaters Wendy has to wash. -/
def num_sweaters : ℕ := (machine_capacity * total_loads) - num_shirts

theorem wendy_sweaters : num_sweaters = 33 := by
  sorry

end wendy_sweaters_l1528_152889


namespace time_subtraction_problem_l1528_152841

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Converts minutes to a Time structure -/
def minutesToTime (m : ℕ) : Time :=
  { hours := m / 60,
    minutes := m % 60,
    valid := by sorry }

/-- Subtracts two Time structures -/
def subtractTime (t1 t2 : Time) : Time :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  minutesToTime (totalMinutes1 - totalMinutes2)

theorem time_subtraction_problem :
  let currentTime : Time := { hours := 18, minutes := 27, valid := by sorry }
  let minutesToSubtract : ℕ := 2880717
  let resultTime : Time := subtractTime currentTime (minutesToTime minutesToSubtract)
  resultTime.hours = 6 ∧ resultTime.minutes = 30 := by sorry

end time_subtraction_problem_l1528_152841


namespace product_of_sums_l1528_152808

theorem product_of_sums : 
  (8 - Real.sqrt 500 + 8 + Real.sqrt 500) * (12 - Real.sqrt 72 + 12 + Real.sqrt 72) = 384 := by
  sorry

end product_of_sums_l1528_152808


namespace semicircle_circumference_from_rectangle_perimeter_l1528_152812

def rectangle_length : ℝ := 16
def rectangle_breadth : ℝ := 14

theorem semicircle_circumference_from_rectangle_perimeter :
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_breadth)
  let square_side := rectangle_perimeter / 4
  let semicircle_circumference := (π * square_side) / 2 + square_side
  ∃ ε > 0, |semicircle_circumference - 38.55| < ε := by
  sorry

end semicircle_circumference_from_rectangle_perimeter_l1528_152812


namespace equation_solution_l1528_152884

theorem equation_solution : ∃ x : ℝ, x = 37/10 ∧ Real.sqrt (3 * Real.sqrt (x - 3)) = (10 - x) ^ (1/4) := by
  sorry

end equation_solution_l1528_152884


namespace vector_proof_l1528_152836

theorem vector_proof (a b : ℝ × ℝ) : 
  b = (1, -2) → 
  (a.1 * b.1 + a.2 * b.2 = -Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) → 
  Real.sqrt (a.1^2 + a.2^2) = 3 * Real.sqrt 5 → 
  a = (-3, 6) := by sorry

end vector_proof_l1528_152836


namespace original_price_from_discounted_l1528_152803

/-- 
Given a product with an original price, this theorem proves that 
if the price after successive discounts of 15% and 25% is 306, 
then the original price was 480.
-/
theorem original_price_from_discounted (original_price : ℝ) : 
  (1 - 0.25) * (1 - 0.15) * original_price = 306 → original_price = 480 := by
  sorry

end original_price_from_discounted_l1528_152803


namespace fraction_equivalence_l1528_152896

theorem fraction_equivalence : (9 : ℚ) / (7 * 53) = 0.9 / (0.7 * 53) := by sorry

end fraction_equivalence_l1528_152896


namespace average_weight_increase_l1528_152883

theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total_weight := 7 * initial_average
  let new_total_weight := initial_total_weight - 75 + 99.5
  let new_average := new_total_weight / 7
  new_average - initial_average = 3.5 := by
sorry

end average_weight_increase_l1528_152883


namespace tangency_points_x_coordinates_l1528_152844

/-- Given a curve y = x^m and a point A(1,0), prove the x-coordinates of the first two tangency points -/
theorem tangency_points_x_coordinates (m : ℕ) (hm : m > 1) :
  let curve (x : ℝ) := x^m
  let tangent_line (a : ℝ) (x : ℝ) := m * a^(m-1) * (x - a) + a^m
  let a₁ := (tangent_line ⁻¹) 0 1  -- x-coordinate where tangent line passes through (1,0)
  let a₂ := (tangent_line ⁻¹) 0 a₁ -- x-coordinate where tangent line passes through (a₁,0)
  a₁ = m / (m - 1) ∧ a₂ = (m / (m - 1))^2 := by
sorry


end tangency_points_x_coordinates_l1528_152844


namespace units_digit_of_five_to_ten_l1528_152871

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 5^10 is 5 -/
theorem units_digit_of_five_to_ten : unitsDigit (5^10) = 5 := by sorry

end units_digit_of_five_to_ten_l1528_152871


namespace harvest_duration_l1528_152820

def harvest_problem (weekly_earning : ℕ) (total_earning : ℕ) : Prop :=
  weekly_earning * 89 = total_earning

theorem harvest_duration : harvest_problem 2 178 := by
  sorry

end harvest_duration_l1528_152820


namespace estate_area_calculation_l1528_152864

/-- Represents the scale of the map in miles per inch -/
def scale : ℝ := 350

/-- Represents the length of the rectangle on the map in inches -/
def map_length : ℝ := 9

/-- Represents the width of the rectangle on the map in inches -/
def map_width : ℝ := 6

/-- Calculates the actual length of the estate in miles -/
def actual_length : ℝ := scale * map_length

/-- Calculates the actual width of the estate in miles -/
def actual_width : ℝ := scale * map_width

/-- Calculates the actual area of the estate in square miles -/
def actual_area : ℝ := actual_length * actual_width

theorem estate_area_calculation :
  actual_area = 6615000 := by sorry

end estate_area_calculation_l1528_152864


namespace nth_roots_of_unity_real_roots_l1528_152857

theorem nth_roots_of_unity_real_roots (n : ℕ) (h : n > 0) :
  ¬ (∀ z : ℂ, z^n = 1 → (z.re = 1 ∧ z.im = 0)) :=
sorry

end nth_roots_of_unity_real_roots_l1528_152857


namespace least_three_digit_9_heavy_l1528_152802

def is_9_heavy (n : ℕ) : Prop := n % 9 > 5

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem least_three_digit_9_heavy : 
  (∀ n : ℕ, is_three_digit n → is_9_heavy n → 105 ≤ n) ∧ 
  is_three_digit 105 ∧ 
  is_9_heavy 105 :=
sorry

end least_three_digit_9_heavy_l1528_152802


namespace unique_natural_with_square_neighbors_l1528_152810

theorem unique_natural_with_square_neighbors :
  ∃! (n : ℕ), ∃ (k m : ℕ), n + 15 = k^2 ∧ n - 14 = m^2 :=
by
  -- The proof goes here
  sorry

end unique_natural_with_square_neighbors_l1528_152810


namespace inequality_solution_l1528_152831

theorem inequality_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 5) (h3 : x ≠ 7) :
  (x - 1) * (x - 4) * (x - 6) / ((x - 2) * (x - 5) * (x - 7)) > 0 ↔
  x < 1 ∨ (2 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ 7 < x :=
by sorry

end inequality_solution_l1528_152831


namespace buratino_coins_impossibility_l1528_152838

theorem buratino_coins_impossibility : ¬ ∃ (n : ℕ), 303 + 6 * n = 456 := by sorry

end buratino_coins_impossibility_l1528_152838


namespace rectangular_plot_length_l1528_152891

theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 60 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  cost_per_meter * perimeter = total_cost →
  length = 80 := by
sorry

end rectangular_plot_length_l1528_152891


namespace vitamin_d_pack_size_l1528_152827

/-- The number of Vitamin A supplements in each pack -/
def vitamin_a_pack_size : ℕ := 7

/-- The smallest number of each type of vitamin sold -/
def smallest_quantity_sold : ℕ := 119

/-- Theorem stating that the number of Vitamin D supplements in each pack is 17 -/
theorem vitamin_d_pack_size :
  ∃ (n m x : ℕ),
    n * vitamin_a_pack_size = m * x ∧
    n * vitamin_a_pack_size = smallest_quantity_sold ∧
    x > 1 ∧
    x < vitamin_a_pack_size ∧
    x = 17 := by
  sorry

end vitamin_d_pack_size_l1528_152827


namespace smallest_angle_theorem_l1528_152869

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.sin (3 * x) * Real.sin (4 * x) = Real.cos (3 * x) * Real.cos (4 * x)

-- Define the theorem
theorem smallest_angle_theorem :
  ∃ (x : ℝ), x > 0 ∧ x < π ∧ equation x ∧
  (∀ (y : ℝ), y > 0 ∧ y < x → ¬equation y) ∧
  x = 90 * (π / 180) / 7 :=
sorry

end smallest_angle_theorem_l1528_152869


namespace factorization_equality_l1528_152822

theorem factorization_equality (m n : ℝ) : m^2*n + 2*m*n^2 + n^3 = n*(m+n)^2 := by
  sorry

end factorization_equality_l1528_152822


namespace matching_probability_is_one_third_l1528_152835

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans a person has -/
def JellyBeans.total (jb : JellyBeans) : ℕ := jb.blue + jb.green + jb.yellow

/-- Abe's jelly beans -/
def abe : JellyBeans := { blue := 2, green := 2, yellow := 0 }

/-- Bob's jelly beans -/
def bob : JellyBeans := { blue := 3, green := 1, yellow := 2 }

/-- The probability of two people showing matching color jelly beans -/
def matchingProbability (person1 person2 : JellyBeans) : ℚ :=
  let totalProb : ℚ := 
    (person1.blue * person2.blue + person1.green * person2.green) / 
    (person1.total * person2.total)
  totalProb

theorem matching_probability_is_one_third : 
  matchingProbability abe bob = 1/3 := by
  sorry

end matching_probability_is_one_third_l1528_152835


namespace vowels_on_board_l1528_152867

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 2

/-- The total number of vowels written on the board -/
def total_vowels : ℕ := num_vowels * times_written

theorem vowels_on_board : total_vowels = 10 := by
  sorry

end vowels_on_board_l1528_152867


namespace price_increase_l1528_152821

theorem price_increase (P : ℝ) (x : ℝ) (h1 : P > 0) :
  1.25 * P * (1 + x / 100) = 1.625 * P → x = 30 := by
  sorry

end price_increase_l1528_152821


namespace system_solution_exists_l1528_152815

theorem system_solution_exists : ∃ (x y z : ℝ),
  (2 * x + 3 * y + z = 13) ∧
  (4 * x^2 + 9 * y^2 + z^2 - 2 * x + 15 * y + 3 * z = 82) := by
  sorry

end system_solution_exists_l1528_152815


namespace exists_bijection_Z_to_H_l1528_152817

-- Define the set ℍ
def ℍ : Set ℚ :=
  { x | ∀ S : Set ℚ, 
    (1/2 ∈ S) → 
    (∀ y ∈ S, 1/(1+y) ∈ S ∧ y/(1+y) ∈ S) → 
    x ∈ S }

-- State the theorem
theorem exists_bijection_Z_to_H : ∃ f : ℤ → ℍ, Function.Bijective f := by
  sorry

end exists_bijection_Z_to_H_l1528_152817


namespace marked_cells_bound_l1528_152885

/-- Represents a cell color on the board -/
inductive CellColor
| Black
| White

/-- Represents a (2n+1) × (2n+1) board -/
def Board (n : ℕ) := Fin (2*n+1) → Fin (2*n+1) → CellColor

/-- Counts the number of cells of a given color in a row -/
def countInRow (board : Board n) (row : Fin (2*n+1)) (color : CellColor) : ℕ := sorry

/-- Counts the number of cells of a given color in a column -/
def countInColumn (board : Board n) (col : Fin (2*n+1)) (color : CellColor) : ℕ := sorry

/-- Determines if a cell should be marked based on its row -/
def isMarkedInRow (board : Board n) (row col : Fin (2*n+1)) : Bool := sorry

/-- Determines if a cell should be marked based on its column -/
def isMarkedInColumn (board : Board n) (row col : Fin (2*n+1)) : Bool := sorry

/-- Counts the total number of marked cells on the board -/
def countMarkedCells (board : Board n) : ℕ := sorry

/-- Counts the total number of black cells on the board -/
def countBlackCells (board : Board n) : ℕ := sorry

/-- Counts the total number of white cells on the board -/
def countWhiteCells (board : Board n) : ℕ := sorry

/-- The main theorem: The number of marked cells is at least half the minimum of black and white cells -/
theorem marked_cells_bound (n : ℕ) (board : Board n) :
  2 * countMarkedCells board ≥ min (countBlackCells board) (countWhiteCells board) := by
  sorry

end marked_cells_bound_l1528_152885


namespace flag_movement_theorem_l1528_152872

/-- Calculates the total distance a flag moves on a flagpole given the pole height and a sequence of movements. -/
def totalFlagMovement (poleHeight : ℝ) (movements : List ℝ) : ℝ :=
  movements.map (abs) |>.sum

/-- Theorem stating the total distance a flag moves on a 60-foot flagpole when raised to the top, 
    lowered halfway, raised to the top again, and then lowered completely is 180 feet. -/
theorem flag_movement_theorem :
  let poleHeight : ℝ := 60
  let movements : List ℝ := [poleHeight, -poleHeight/2, poleHeight/2, -poleHeight]
  totalFlagMovement poleHeight movements = 180 := by
  sorry

#eval totalFlagMovement 60 [60, -30, 30, -60]

end flag_movement_theorem_l1528_152872


namespace value_of_a_l1528_152870

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2

-- Theorem statement
theorem value_of_a (a : ℝ) : f_derivative a (-1) = 3 → a = 1 := by
  sorry

end value_of_a_l1528_152870


namespace paths_to_n_2_l1528_152843

/-- The number of possible paths from (0,0) to (x, y) -/
def f (x y : ℕ) : ℕ := sorry

/-- The theorem stating that f(n, 2) = (1/2)(n^2 + 3n + 2) for all natural numbers n -/
theorem paths_to_n_2 (n : ℕ) : f n 2 = (n^2 + 3*n + 2) / 2 := by sorry

end paths_to_n_2_l1528_152843


namespace modulo_graph_intercepts_sum_l1528_152806

theorem modulo_graph_intercepts_sum (x₀ y₀ : ℕ) : 
  x₀ < 37 → y₀ < 37 →
  (2 * x₀) % 37 = 1 →
  (3 * y₀ + 1) % 37 = 0 →
  x₀ + y₀ = 31 := by
sorry

end modulo_graph_intercepts_sum_l1528_152806


namespace average_weight_problem_l1528_152837

theorem average_weight_problem (num_group1 : ℕ) (num_group2 : ℕ) (avg_weight_group2 : ℝ) (avg_weight_total : ℝ) :
  num_group1 = 24 →
  num_group2 = 8 →
  avg_weight_group2 = 45.15 →
  avg_weight_total = 48.975 →
  let total_num := num_group1 + num_group2
  let total_weight := total_num * avg_weight_total
  let weight_group2 := num_group2 * avg_weight_group2
  let weight_group1 := total_weight - weight_group2
  let avg_weight_group1 := weight_group1 / num_group1
  avg_weight_group1 = 50.25 := by
sorry

end average_weight_problem_l1528_152837


namespace sin_cos_shift_l1528_152873

open Real

theorem sin_cos_shift (x : ℝ) : 
  sin (2 * x) + Real.sqrt 3 * cos (2 * x) = 2 * sin (2 * (x + π / 6)) := by
  sorry

end sin_cos_shift_l1528_152873


namespace sufficient_not_necessary_l1528_152865

/-- An even function on ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem sufficient_not_necessary
  (f : ℝ → ℝ) (hf : EvenFunction f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ - f x₂ = 0) ∧
  (∃ x₁ x₂ : ℝ, f x₁ - f x₂ = 0 ∧ x₁ + x₂ ≠ 0) :=
by sorry

end sufficient_not_necessary_l1528_152865


namespace no_valid_a_l1528_152829

theorem no_valid_a : ¬∃ a : ℕ+, (a ≤ 100) ∧ 
  (∃ x y : ℤ, x ≠ y ∧ 
    2 * x^2 + (3 * a.val + 1) * x + a.val^2 = 0 ∧
    2 * y^2 + (3 * a.val + 1) * y + a.val^2 = 0) :=
by
  sorry

#check no_valid_a

end no_valid_a_l1528_152829


namespace class_funds_calculation_l1528_152868

/-- Proves that the class funds amount to $14 given the problem conditions -/
theorem class_funds_calculation (total_contribution student_count student_contribution : ℕ) 
  (h1 : total_contribution = 90)
  (h2 : student_count = 19)
  (h3 : student_contribution = 4) :
  total_contribution - (student_count * student_contribution) = 14 := by
  sorry

#check class_funds_calculation

end class_funds_calculation_l1528_152868


namespace calculate_dividend_l1528_152880

/-- Given a division with quotient, divisor, and remainder, calculate the dividend -/
theorem calculate_dividend (quotient divisor remainder : ℝ) :
  quotient = -415.2 →
  divisor = 2735 →
  remainder = 387.3 →
  (quotient * divisor) + remainder = -1135106.7 := by
  sorry

end calculate_dividend_l1528_152880


namespace petya_running_time_l1528_152886

theorem petya_running_time (V D : ℝ) (h1 : V > 0) (h2 : D > 0) : 
  (D / (1.25 * V) / 2) + (D / (0.8 * V) / 2) > D / V := by
  sorry

end petya_running_time_l1528_152886


namespace triangle_theorem_l1528_152862

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the main results -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 = t.b^2 + t.c^2 - t.b * t.c) 
  (h2 : t.a = Real.sqrt 3) : 
  (t.A = π/3) ∧ (Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3) := by
  sorry


end triangle_theorem_l1528_152862


namespace quadratic_two_real_roots_l1528_152845

theorem quadratic_two_real_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 4*x + 2 = 0 ∧ a * y^2 - 4*y + 2 = 0) ↔ 
  (a ≤ 2 ∧ a ≠ 0) :=
sorry

end quadratic_two_real_roots_l1528_152845


namespace first_hole_depth_l1528_152816

/-- Represents the depth of a hole dug by workers. -/
def hole_depth (workers : ℕ) (hours : ℕ) (rate : ℚ) : ℚ :=
  (workers : ℚ) * (hours : ℚ) * rate

/-- The work rate is constant for both holes. -/
def work_rate : ℚ := 1 / 12

theorem first_hole_depth :
  let first_hole := hole_depth 45 8 work_rate
  let second_hole := hole_depth 90 6 work_rate
  second_hole = 45 →
  first_hole = 30 := by sorry

end first_hole_depth_l1528_152816


namespace odometer_sum_of_squares_l1528_152818

/-- Represents the odometer reading as a three-digit number -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds + tens + ones ≤ 9

/-- Represents the car's journey -/
structure CarJourney where
  duration : Nat
  avg_speed : Nat
  start_reading : OdometerReading
  end_reading : OdometerReading
  journey_valid : 
    duration = 8 ∧ 
    avg_speed = 65 ∧
    end_reading.hundreds = start_reading.ones ∧
    end_reading.tens = start_reading.tens ∧
    end_reading.ones = start_reading.hundreds

theorem odometer_sum_of_squares (journey : CarJourney) : 
  journey.start_reading.hundreds^2 + 
  journey.start_reading.tens^2 + 
  journey.start_reading.ones^2 = 41 := by
  sorry

end odometer_sum_of_squares_l1528_152818


namespace two_primes_sum_and_product_l1528_152894

theorem two_primes_sum_and_product : ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p * q = 166 ∧ p + q = 85 := by
sorry

end two_primes_sum_and_product_l1528_152894


namespace bernoulli_zero_success_l1528_152882

/-- The number of trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The probability of failure in each trial -/
def q : ℚ := 1 - p

/-- The number of successes we're interested in -/
def k : ℕ := 0

/-- Theorem: The probability of 0 successes in 7 Bernoulli trials 
    with success probability 2/7 is (5/7)^7 -/
theorem bernoulli_zero_success : 
  (n.choose k) * p^k * q^(n-k) = (5/7)^7 := by sorry

end bernoulli_zero_success_l1528_152882


namespace product_w_z_is_24_l1528_152863

/-- Represents a parallelogram EFGH with given side lengths -/
structure Parallelogram where
  ef : ℝ
  fg : ℝ → ℝ
  gh : ℝ → ℝ
  he : ℝ
  is_parallelogram : ef = gh 0 ∧ fg 0 = he

/-- The product of w and z in the given parallelogram is 24 -/
theorem product_w_z_is_24 (p : Parallelogram)
    (h_ef : p.ef = 42)
    (h_fg : p.fg = fun z => 4 * z^3)
    (h_gh : p.gh = fun w => 3 * w + 6)
    (h_he : p.he = 32) :
    ∃ w z, p.gh w = p.ef ∧ p.fg z = p.he ∧ w * z = 24 := by
  sorry

end product_w_z_is_24_l1528_152863


namespace f_composition_negative_two_l1528_152877

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -2 then x + 2
  else if x < 3 then 2^x
  else Real.log x

theorem f_composition_negative_two : f (f (-2)) = 1 := by
  sorry

end f_composition_negative_two_l1528_152877


namespace associated_points_theorem_l1528_152858

/-- Definition of k times associated point -/
def k_times_associated_point (P M : ℝ × ℝ) (k : ℤ) : Prop :=
  let d_PM := Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)
  let d_PO := Real.sqrt (P.1^2 + P.2^2)
  d_PM = k * d_PO

/-- Main theorem -/
theorem associated_points_theorem :
  let P₁ : ℝ × ℝ := (-1.5, 0)
  let P₂ : ℝ × ℝ := (-1, 0)
  ∀ (b : ℝ),
  (∃ (M : ℝ × ℝ), k_times_associated_point P₁ M 2 ∧ M.2 = 0 →
    (M = (1.5, 0) ∨ M = (-4.5, 0))) ∧
  (∀ (M : ℝ × ℝ) (k : ℤ),
    k_times_associated_point P₁ M k ∧ M.1 = -1.5 ∧ -3 ≤ M.2 ∧ M.2 ≤ 5 →
    k ≤ 3) ∧
  (∃ (A B C : ℝ × ℝ),
    A = (b, 0) ∧ B = (b + 1, 0) ∧
    Real.sqrt ((C.1 - A.1)^2 + C.2^2) = Real.sqrt ((B.1 - A.1)^2 + (C.2 - B.2)^2) ∧
    C.2 / (C.1 - A.1) = Real.sqrt 3 / 3 →
    (∃ (Q : ℝ × ℝ), k_times_associated_point P₂ Q 2 ∧
      (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ Q = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))) →
      (-4 ≤ b ∧ b ≤ -3) ∨ (-1 ≤ b ∧ b ≤ 1))) :=
by
  sorry

end associated_points_theorem_l1528_152858


namespace simplify_expression_l1528_152861

theorem simplify_expression (z : ℝ) : z - 3 + 4*z + 5 - 6*z + 7 - 8*z + 9 = -9*z + 18 := by
  sorry

end simplify_expression_l1528_152861


namespace female_officers_count_l1528_152832

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_duty_percentage : ℚ) :
  total_on_duty = 170 →
  female_on_duty_ratio = 1/2 →
  female_duty_percentage = 17/100 →
  ∃ (total_female : ℕ), total_female = 500 ∧ 
    (↑total_on_duty * female_on_duty_ratio : ℚ) = (↑total_female * female_duty_percentage : ℚ) :=
by sorry

end female_officers_count_l1528_152832


namespace power_sum_divisibility_l1528_152854

theorem power_sum_divisibility (k : ℕ) :
  7 ∣ (2^k + 3^k) ↔ k % 6 = 3 := by sorry

end power_sum_divisibility_l1528_152854


namespace polynomial_expansion_properties_l1528_152855

theorem polynomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 243) ∧
  (a₁ + a₃ + a₅ = 122) := by
  sorry

end polynomial_expansion_properties_l1528_152855


namespace last_five_days_avg_l1528_152878

/-- Represents the TV production scenario in a factory --/
structure TVProduction where
  total_days : Nat
  first_period_days : Nat
  first_period_avg : Nat
  monthly_avg : Nat

/-- Calculates the average daily production for the last period --/
def last_period_avg (prod : TVProduction) : Rat :=
  let last_period_days := prod.total_days - prod.first_period_days
  let total_monthly_production := prod.monthly_avg * prod.total_days
  let first_period_production := prod.first_period_avg * prod.first_period_days
  let last_period_production := total_monthly_production - first_period_production
  last_period_production / last_period_days

/-- Theorem stating the average production for the last 5 days --/
theorem last_five_days_avg (prod : TVProduction) 
  (h1 : prod.total_days = 30)
  (h2 : prod.first_period_days = 25)
  (h3 : prod.first_period_avg = 50)
  (h4 : prod.monthly_avg = 45) :
  last_period_avg prod = 20 := by
  sorry

end last_five_days_avg_l1528_152878


namespace soccer_practice_probability_l1528_152814

theorem soccer_practice_probability (p : ℚ) (h : p = 5/8) :
  1 - p = 3/8 := by sorry

end soccer_practice_probability_l1528_152814


namespace intersection_equal_B_l1528_152852

def A : Set ℝ := {x | x^2 - 4*x - 21 = 0}

def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem intersection_equal_B (m : ℝ) : 
  (A ∩ B m) = B m ↔ m = 0 ∨ m = -1/7 ∨ m = 1/3 := by
  sorry

end intersection_equal_B_l1528_152852


namespace product_expansion_l1528_152849

theorem product_expansion (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * ((7 / y) + 14 * y^3) = 3 / y + 6 * y^3 := by
  sorry

end product_expansion_l1528_152849


namespace sin_330_degrees_l1528_152800

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l1528_152800


namespace geometric_sequence_ratio_l1528_152895

/-- A geometric sequence with its third term and sum of first three terms given -/
structure GeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  is_geometric : ∀ n, a (n + 1) = q * a n
  third_term : a 3 = 3/2
  third_sum : (a 1) + (a 2) + (a 3) = 9/2

/-- The common ratio of the geometric sequence is either 1 or -1/2 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : 
  seq.q = 1 ∨ seq.q = -1/2 := by
  sorry

end geometric_sequence_ratio_l1528_152895


namespace gcd_204_85_l1528_152859

theorem gcd_204_85 : Nat.gcd 204 85 = 57 := by
  sorry

end gcd_204_85_l1528_152859


namespace remaining_pool_area_l1528_152828

/-- The area of the remaining pool space given a circular pool with diameter 13 meters
    and a rectangular obstacle with dimensions 2.5 meters by 4 meters. -/
theorem remaining_pool_area :
  let pool_diameter : ℝ := 13
  let obstacle_length : ℝ := 2.5
  let obstacle_width : ℝ := 4
  let pool_area := π * (pool_diameter / 2) ^ 2
  let obstacle_area := obstacle_length * obstacle_width
  pool_area - obstacle_area = 132.7325 * π - 10 := by sorry

end remaining_pool_area_l1528_152828


namespace tv_show_episodes_l1528_152874

/-- Given a TV show with the following properties:
  * It ran for 10 seasons
  * The first half of seasons had 20 episodes per season
  * There were 225 total episodes
  This theorem proves that the number of episodes per season in the second half was 25. -/
theorem tv_show_episodes (total_seasons : ℕ) (first_half_episodes : ℕ) (total_episodes : ℕ) :
  total_seasons = 10 →
  first_half_episodes = 20 →
  total_episodes = 225 →
  (total_episodes - (total_seasons / 2 * first_half_episodes)) / (total_seasons / 2) = 25 :=
by sorry

end tv_show_episodes_l1528_152874


namespace total_bushels_is_65_l1528_152830

/-- The number of bushels needed for all animals for a day on Dany's farm -/
def total_bushels : ℕ :=
  let cow_count : ℕ := 5
  let cow_consumption : ℕ := 3
  let sheep_count : ℕ := 4
  let sheep_consumption : ℕ := 2
  let chicken_count : ℕ := 8
  let chicken_consumption : ℕ := 1
  let pig_count : ℕ := 6
  let pig_consumption : ℕ := 4
  let horse_count : ℕ := 2
  let horse_consumption : ℕ := 5
  cow_count * cow_consumption +
  sheep_count * sheep_consumption +
  chicken_count * chicken_consumption +
  pig_count * pig_consumption +
  horse_count * horse_consumption

theorem total_bushels_is_65 : total_bushels = 65 := by
  sorry

end total_bushels_is_65_l1528_152830


namespace function_has_max_and_min_l1528_152887

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*((a + 2)*x + 1)

-- State the theorem
theorem function_has_max_and_min (a : ℝ) :
  (∃ x₁ x₂ : ℝ, ∀ x : ℝ, f a x₁ ≤ f a x ∧ f a x ≤ f a x₂) ↔ (a > 2 ∨ a < -1) :=
sorry

end function_has_max_and_min_l1528_152887


namespace strongest_teams_in_tournament_l1528_152811

/-- Represents a volleyball team in the tournament -/
structure Team :=
  (name : String)
  (wins : Nat)
  (losses : Nat)

/-- Represents the tournament results -/
structure TournamentResults :=
  (teams : List Team)
  (numTeams : Nat)
  (roundRobin : Bool)
  (bestOfThree : Bool)

/-- Determines if a team is one of the two strongest teams -/
def isStrongestTeam (team : Team) (results : TournamentResults) : Prop :=
  ∃ (otherTeam : Team),
    otherTeam ∈ results.teams ∧
    team ∈ results.teams ∧
    team ≠ otherTeam ∧
    ∀ (t : Team), t ∈ results.teams →
      (t.wins < team.wins ∨ (t.wins = team.wins ∧ t.losses ≥ team.losses)) ∨
      (t.wins < otherTeam.wins ∨ (t.wins = otherTeam.wins ∧ t.losses ≥ otherTeam.losses))

theorem strongest_teams_in_tournament
  (results : TournamentResults)
  (h1 : results.numTeams = 6)
  (h2 : results.roundRobin = true)
  (h3 : results.bestOfThree = true)
  (first : Team)
  (second : Team)
  (fourth : Team)
  (fifth : Team)
  (sixth : Team)
  (h4 : first ∈ results.teams ∧ first.wins = 2 ∧ first.losses = 3)
  (h5 : second ∈ results.teams ∧ second.wins = 4 ∧ second.losses = 1)
  (h6 : fourth ∈ results.teams ∧ fourth.wins = 0 ∧ fourth.losses = 5)
  (h7 : fifth ∈ results.teams ∧ fifth.wins = 4 ∧ fifth.losses = 1)
  (h8 : sixth ∈ results.teams ∧ sixth.wins = 4 ∧ sixth.losses = 1)
  : isStrongestTeam fifth results ∧ isStrongestTeam sixth results :=
sorry

end strongest_teams_in_tournament_l1528_152811


namespace four_block_selection_count_l1528_152888

def grid_size : ℕ := 6
def blocks_to_select : ℕ := 4

theorem four_block_selection_count :
  (Nat.choose grid_size blocks_to_select) *
  (Nat.choose grid_size blocks_to_select) *
  (Nat.factorial blocks_to_select) = 5400 := by
  sorry

end four_block_selection_count_l1528_152888
