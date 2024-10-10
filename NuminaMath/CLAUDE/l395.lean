import Mathlib

namespace last_digit_of_sum_l395_39566

theorem last_digit_of_sum (n : ℕ) : 
  (54^2019 + 28^2021) % 10 = 2 := by
  sorry

end last_digit_of_sum_l395_39566


namespace election_winner_percentage_l395_39585

theorem election_winner_percentage (total_votes : ℕ) (majority : ℕ) : 
  total_votes = 470 → majority = 188 → 
  (70 : ℚ) * total_votes / 100 - ((100 : ℚ) - 70) * total_votes / 100 = majority := by
  sorry

end election_winner_percentage_l395_39585


namespace range_of_m_l395_39520

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x > 1 → 2*x + m + 2/(x-1) > 0) → m > -6 := by
  sorry

end range_of_m_l395_39520


namespace parallelogram_count_l395_39589

/-- 
Given an equilateral triangle ABC where each side is divided into n equal parts
and lines are drawn parallel to each side through these division points,
the total number of parallelograms formed is 3 * (n+1)^2 * n^2 / 4.
-/
theorem parallelogram_count (n : ℕ) : 
  (3 : ℚ) * (n + 1)^2 * n^2 / 4 = 3 * Nat.choose (n + 2) 4 := by
  sorry

end parallelogram_count_l395_39589


namespace last_two_digits_product_l395_39524

theorem last_two_digits_product (n : ℤ) : ∃ k : ℤ, 122 * 123 * 125 * 127 * n ≡ 50 [ZMOD 100] := by
  sorry

end last_two_digits_product_l395_39524


namespace parallelogram_division_slope_l395_39516

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Checks if a line with given slope passes through the origin and divides the parallelogram into two congruent polygons -/
def dividesParallelogramEqually (p : Parallelogram) (slope : ℚ) : Prop :=
  ∃ (a : ℝ),
    (p.v1.y + a) / p.v1.x = slope ∧
    (p.v3.y - a) / p.v3.x = slope ∧
    0 < a ∧ a < p.v2.y - p.v1.y

/-- The main theorem stating the slope of the line dividing the parallelogram equally -/
theorem parallelogram_division_slope :
  let p : Parallelogram := {
    v1 := { x := 12, y := 60 },
    v2 := { x := 12, y := 152 },
    v3 := { x := 32, y := 204 },
    v4 := { x := 32, y := 112 }
  }
  dividesParallelogramEqually p 16 := by sorry

end parallelogram_division_slope_l395_39516


namespace cuboids_painted_l395_39591

theorem cuboids_painted (faces_per_cuboid : ℕ) (total_faces : ℕ) (h1 : faces_per_cuboid = 6) (h2 : total_faces = 36) :
  total_faces / faces_per_cuboid = 6 :=
by sorry

end cuboids_painted_l395_39591


namespace triangle_side_length_l395_39571

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 2)
  (h2 : A = π / 6)  -- 30° in radians
  (h3 : C = 3 * π / 4)  -- 135° in radians
  (h4 : A + B + C = π)  -- sum of angles in a triangle
  (h5 : a / Real.sin A = b / Real.sin B)  -- Law of Sines
  : b = (Real.sqrt 2 - Real.sqrt 6) / 2 := by sorry

end triangle_side_length_l395_39571


namespace expansion_properties_l395_39532

/-- Given an expression (3x - 1/(2*3x))^n where the ratio of the binomial coefficient 
    of the fifth term to that of the third term is 14:3, this theorem proves various 
    properties about the expansion. -/
theorem expansion_properties (n : ℕ) 
  (h : (Nat.choose n 4 : ℚ) / (Nat.choose n 2 : ℚ) = 14 / 3) :
  n = 10 ∧ 
  (let coeff_x2 := (Nat.choose 10 2 : ℚ) * (-1/2)^2 * 3^2;
   coeff_x2 = 45/4) ∧
  (let rational_terms := [
     (Nat.choose 10 2 : ℚ) * (-1/2)^2,
     (Nat.choose 10 5 : ℚ) * (-1/2)^5,
     (Nat.choose 10 8 : ℚ) * (-1/2)^8
   ];
   rational_terms.length = 3) :=
by sorry


end expansion_properties_l395_39532


namespace divide_by_fraction_twelve_divided_by_one_sixth_l395_39533

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) :
  a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_sixth :
  12 / (1 / 6 : ℚ) = 72 := by sorry

end divide_by_fraction_twelve_divided_by_one_sixth_l395_39533


namespace clothing_sales_properties_l395_39537

/-- Represents the sales pattern of a new clothing item in July -/
structure ClothingSales where
  /-- The day number in July when maximum sales occurred -/
  max_day : ℕ
  /-- The maximum number of pieces sold in a day -/
  max_sales : ℕ
  /-- The number of days the clothing was popular -/
  popular_days : ℕ

/-- Calculates the sales for a given day in July -/
def daily_sales (day : ℕ) : ℕ :=
  if day ≤ 13 then 3 * day else 65 - 2 * day

/-- Calculates the cumulative sales up to a given day in July -/
def cumulative_sales (day : ℕ) : ℕ :=
  if day ≤ 13 
  then (3 + 3 * day) * day / 2
  else 273 + (51 - day) * (day - 13)

/-- Theorem stating the properties of the clothing sales in July -/
theorem clothing_sales_properties : ∃ (s : ClothingSales),
  s.max_day = 13 ∧ 
  s.max_sales = 39 ∧ 
  s.popular_days = 11 ∧
  daily_sales 1 = 3 ∧
  daily_sales 31 = 3 ∧
  (∀ d : ℕ, d < s.max_day → daily_sales (d + 1) = daily_sales d + 3) ∧
  (∀ d : ℕ, s.max_day < d ∧ d ≤ 31 → daily_sales d = daily_sales (d - 1) - 2) ∧
  (∃ d : ℕ, d ≥ 12 ∧ cumulative_sales d ≥ 200 ∧ cumulative_sales (d - 1) < 200) ∧
  (∃ d : ℕ, d ≤ 22 ∧ daily_sales d ≥ 20 ∧ daily_sales (d + 1) < 20) := by
  sorry

end clothing_sales_properties_l395_39537


namespace ten_digit_number_exists_l395_39554

def product_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * product_of_digits (n / 10)

theorem ten_digit_number_exists : ∃ n : ℕ, 
  1000000000 ≤ n ∧ n < 10000000000 ∧ 
  (∀ d, d ∣ n → d ≠ 0) ∧
  product_of_digits (n + product_of_digits n) = product_of_digits n :=
sorry

end ten_digit_number_exists_l395_39554


namespace treadmill_time_difference_l395_39551

theorem treadmill_time_difference : 
  let total_distance : ℝ := 8
  let constant_speed : ℝ := 3
  let day1_speed : ℝ := 6
  let day2_speed : ℝ := 3
  let day3_speed : ℝ := 4
  let day4_speed : ℝ := 3
  let daily_distance : ℝ := 2
  let constant_time := total_distance / constant_speed
  let varied_time := daily_distance / day1_speed + daily_distance / day2_speed + 
                     daily_distance / day3_speed + daily_distance / day4_speed
  (constant_time - varied_time) * 60 = 80 := by sorry

end treadmill_time_difference_l395_39551


namespace quadratic_roots_property_l395_39596

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 4 = 0 → x₂^2 - 3*x₂ - 4 = 0 → x₁^2 - 4*x₁ - x₂ + 2*x₁*x₂ = -7 :=
by sorry

end quadratic_roots_property_l395_39596


namespace strawberry_weight_calculation_l395_39568

/-- Calculates the weight of Marco's dad's strawberries after losing some. -/
def dads_strawberry_weight (total_initial : ℕ) (lost : ℕ) (marcos : ℕ) : ℕ :=
  total_initial - lost - marcos

/-- Theorem: Given the initial total weight of strawberries, the weight of strawberries lost,
    and Marco's current weight of strawberries, Marco's dad's current weight of strawberries
    is equal to the difference between the remaining total weight and Marco's current weight. -/
theorem strawberry_weight_calculation
  (total_initial : ℕ)
  (lost : ℕ)
  (marcos : ℕ)
  (h1 : total_initial = 36)
  (h2 : lost = 8)
  (h3 : marcos = 12) :
  dads_strawberry_weight total_initial lost marcos = 16 :=
by sorry

end strawberry_weight_calculation_l395_39568


namespace min_distance_curve_to_line_l395_39559

/-- The minimum distance from a point on the curve y = x^2 - ln x to the line y = x - 2 is √2 --/
theorem min_distance_curve_to_line :
  let f (x : ℝ) := x^2 - Real.log x
  let g (x : ℝ) := x - 2
  ∀ x > 0, ∃ y : ℝ, y = f x ∧
    (∀ x' > 0, ∃ y' : ℝ, y' = f x' →
      Real.sqrt 2 ≤ |y' - g x'|) ∧
    |y - g x| = Real.sqrt 2 :=
by sorry

end min_distance_curve_to_line_l395_39559


namespace expression_evaluation_l395_39573

theorem expression_evaluation :
  let a : ℝ := 1
  let b : ℝ := -1
  5 * (3 * a^2 * b - a * b^2) - (a * b^2 + 3 * a^2 * b) + 1 = -17 := by
sorry

end expression_evaluation_l395_39573


namespace peppers_weight_l395_39564

theorem peppers_weight (total_weight green_weight : Float) 
  (h1 : total_weight = 0.6666666666666666)
  (h2 : green_weight = 0.3333333333333333) :
  total_weight - green_weight = 0.3333333333333333 := by
  sorry

end peppers_weight_l395_39564


namespace circle_equation_for_given_points_l395_39511

/-- Given two points P and Q in a 2D plane, this function returns the standard equation
    of the circle with diameter PQ as a function from ℝ × ℝ → Prop -/
def circle_equation (P Q : ℝ × ℝ) : (ℝ × ℝ → Prop) :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  let h := (x₁ + x₂) / 2
  let k := (y₁ + y₂) / 2
  let r := Real.sqrt (((x₂ - x₁)^2 + (y₂ - y₁)^2) / 4)
  fun (x, y) ↦ (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the standard equation of the circle with diameter PQ,
    where P(3,4) and Q(-5,6), is (x + 1)^2 + (y - 5)^2 = 17 -/
theorem circle_equation_for_given_points :
  circle_equation (3, 4) (-5, 6) = fun (x, y) ↦ (x + 1)^2 + (y - 5)^2 = 17 := by
  sorry

end circle_equation_for_given_points_l395_39511


namespace arithmetic_sequence_fifth_term_l395_39536

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)  -- The sequence
  (S : ℕ → ℝ)  -- The sum sequence
  (h1 : ∀ n, S n = (n : ℝ) / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1)))  -- Definition of sum
  (h2 : ∀ n, a (n + 1) = a n + (a 2 - a 1))  -- Definition of arithmetic sequence

/-- The main theorem -/
theorem arithmetic_sequence_fifth_term 
  (seq : ArithmeticSequence) 
  (eq1 : seq.a 3 + seq.S 3 = 22) 
  (eq2 : seq.a 4 - seq.S 4 = -15) : 
  seq.a 5 = 11 := by sorry

end arithmetic_sequence_fifth_term_l395_39536


namespace even_increasing_function_inequality_l395_39538

-- Define an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define an increasing function on [0, +∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Main theorem
theorem even_increasing_function_inequality
  (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_incr : increasing_on_nonneg f) :
  ∀ k, f k > f 2 ↔ k > 2 ∨ k < -2 :=
sorry

end even_increasing_function_inequality_l395_39538


namespace triangle_area_is_40_5_l395_39535

/-- A line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- A right triangle formed by a line intersecting the x and y axes -/
structure RightTriangle where
  line : Line

/-- Calculate the area of the right triangle -/
def area (triangle : RightTriangle) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem triangle_area_is_40_5 :
  let l : Line := { point1 := (-3, 6), point2 := (-6, 3) }
  let t : RightTriangle := { line := l }
  area t = 40.5 := by
  sorry

end triangle_area_is_40_5_l395_39535


namespace four_integers_product_2002_sum_less_40_l395_39565

theorem four_integers_product_2002_sum_less_40 :
  ∀ a b c d : ℕ+,
  a * b * c * d = 2002 →
  (a : ℕ) + b + c + d < 40 →
  ((a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨
   (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
   (a = 1 ∧ b = 11 ∧ c = 14 ∧ d = 13) ∨
   (a = 1 ∧ b = 11 ∧ c = 13 ∧ d = 14) ∨
   (a = 2 ∧ b = 11 ∧ c = 7 ∧ d = 13) ∨
   (a = 2 ∧ b = 11 ∧ c = 13 ∧ d = 7) ∨
   (a = 2 ∧ b = 13 ∧ c = 7 ∧ d = 11) ∨
   (a = 2 ∧ b = 13 ∧ c = 11 ∧ d = 7) ∨
   (a = 7 ∧ b = 2 ∧ c = 11 ∧ d = 13) ∨
   (a = 7 ∧ b = 2 ∧ c = 13 ∧ d = 11) ∨
   (a = 7 ∧ b = 11 ∧ c = 2 ∧ d = 13) ∨
   (a = 7 ∧ b = 11 ∧ c = 13 ∧ d = 2) ∨
   (a = 7 ∧ b = 13 ∧ c = 2 ∧ d = 11) ∧
   (a = 7 ∧ b = 13 ∧ c = 11 ∧ d = 2)) :=
by sorry

end four_integers_product_2002_sum_less_40_l395_39565


namespace cranberry_calculation_l395_39555

/-- The initial number of cranberries in the bog -/
def initial_cranberries : ℕ := 60000

/-- The fraction of cranberries harvested by humans -/
def human_harvest_fraction : ℚ := 2/5

/-- The number of cranberries eaten by elk -/
def elk_eaten : ℕ := 20000

/-- The number of cranberries left after harvesting and elk eating -/
def remaining_cranberries : ℕ := 16000

/-- Theorem stating that the initial number of cranberries is correct given the conditions -/
theorem cranberry_calculation :
  (1 - human_harvest_fraction) * initial_cranberries - elk_eaten = remaining_cranberries :=
by sorry

end cranberry_calculation_l395_39555


namespace restaurant_glasses_count_l395_39531

/-- Represents the number of glasses in a restaurant with two box sizes --/
def total_glasses (small_box_count : ℕ) (large_box_count : ℕ) : ℕ :=
  small_box_count * 12 + large_box_count * 16

/-- Represents the average number of glasses per box --/
def average_glasses_per_box (small_box_count : ℕ) (large_box_count : ℕ) : ℚ :=
  (total_glasses small_box_count large_box_count : ℚ) / (small_box_count + large_box_count : ℚ)

theorem restaurant_glasses_count :
  ∃ (small_box_count : ℕ) (large_box_count : ℕ),
    small_box_count > 0 ∧
    large_box_count = small_box_count + 16 ∧
    average_glasses_per_box small_box_count large_box_count = 15 ∧
    total_glasses small_box_count large_box_count = 480 := by
  sorry


end restaurant_glasses_count_l395_39531


namespace hynek_problem_bounds_l395_39527

/-- Represents a digit assignment for Hynek's problem -/
structure DigitAssignment where
  a : Fin 5
  b : Fin 5
  c : Fin 5
  d : Fin 5
  e : Fin 5
  distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

/-- Calculates the sum for a given digit assignment -/
def calculateSum (assignment : DigitAssignment) : ℕ :=
  (assignment.a + 1) +
  11 * (assignment.b + 1) +
  111 * (assignment.c + 1) +
  1111 * (assignment.d + 1) +
  11111 * (assignment.e + 1)

/-- Checks if a number is divisible by 11 -/
def isDivisibleBy11 (n : ℕ) : Prop :=
  n % 11 = 0

/-- The main theorem stating the smallest and largest possible sums -/
theorem hynek_problem_bounds :
  (∃ (assignment : DigitAssignment),
    isDivisibleBy11 (calculateSum assignment) ∧
    (∀ (other : DigitAssignment),
      isDivisibleBy11 (calculateSum other) →
      calculateSum assignment ≤ calculateSum other)) ∧
  (∃ (assignment : DigitAssignment),
    isDivisibleBy11 (calculateSum assignment) ∧
    (∀ (other : DigitAssignment),
      isDivisibleBy11 (calculateSum other) →
      calculateSum other ≤ calculateSum assignment)) ∧
  (∀ (assignment : DigitAssignment),
    isDivisibleBy11 (calculateSum assignment) →
    23815 ≤ calculateSum assignment ∧ calculateSum assignment ≤ 60589) :=
sorry

end hynek_problem_bounds_l395_39527


namespace sum_of_e_and_f_l395_39525

theorem sum_of_e_and_f (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 5.2)
  (h2 : (c + d) / 2 = 5.8)
  (h3 : (a + b + c + d + e + f) / 6 = 5.4) :
  e + f = 10.4 := by
sorry

end sum_of_e_and_f_l395_39525


namespace point_M_coordinates_l395_39507

/-- Given a line MN with slope 2, point N at (1, -1), and point M on the line y = x + 1,
    prove that the coordinates of point M are (4, 5). -/
theorem point_M_coordinates :
  let slope_MN : ℝ := 2
  let N : ℝ × ℝ := (1, -1)
  let M : ℝ × ℝ := (x, y)
  (y = x + 1) →  -- M lies on y = x + 1
  ((y - N.2) / (x - N.1) = slope_MN) →  -- slope formula
  M = (4, 5) := by
sorry

end point_M_coordinates_l395_39507


namespace unique_solution_for_system_l395_39543

/-- The system of inequalities has a unique solution for specific values of a -/
theorem unique_solution_for_system (a : ℝ) :
  (∃! x y : ℝ, x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) ↔ 
  (a = 1 ∧ ∃! x y : ℝ, x = -1 ∧ y = 0 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) ∨
  (a = -3 ∧ ∃! x y : ℝ, x = 1 ∧ y = 2 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) :=
by sorry

end unique_solution_for_system_l395_39543


namespace specific_pyramid_sphere_radius_l395_39512

/-- Pyramid with equilateral triangular base -/
structure Pyramid :=
  (base_side : ℝ)
  (height : ℝ)

/-- The radius of the circumscribed sphere around the pyramid -/
def circumscribed_sphere_radius (p : Pyramid) : ℝ :=
  sorry

/-- Theorem: The radius of the circumscribed sphere for a specific pyramid -/
theorem specific_pyramid_sphere_radius :
  let p : Pyramid := { base_side := 6, height := 4 }
  circumscribed_sphere_radius p = 4 := by
  sorry

end specific_pyramid_sphere_radius_l395_39512


namespace rational_inequality_solution_l395_39541

theorem rational_inequality_solution (x : ℝ) : x / (x + 5) ≥ 0 ↔ x ∈ Set.Ici 0 ∪ Set.Iio (-5) := by
  sorry

end rational_inequality_solution_l395_39541


namespace ice_cube_volume_l395_39510

theorem ice_cube_volume (original_volume : ℝ) : 
  (original_volume > 0) →
  (original_volume * (1/4) * (1/4) = 0.25) →
  (original_volume = 4) := by
sorry

end ice_cube_volume_l395_39510


namespace max_sum_with_constraint_l395_39519

theorem max_sum_with_constraint (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 + y = 20) :
  x + y ≤ 81/4 :=
sorry

end max_sum_with_constraint_l395_39519


namespace molar_ratio_h2_ch4_l395_39557

/-- Represents the heat of reaction for H₂ combustion in kJ/mol -/
def heat_h2 : ℝ := -571.6

/-- Represents the heat of reaction for CH₄ combustion in kJ/mol -/
def heat_ch4 : ℝ := -890

/-- Represents the volume of the gas mixture in liters -/
def mixture_volume : ℝ := 112

/-- Represents the molar volume of gas under standard conditions in L/mol -/
def molar_volume : ℝ := 22.4

/-- Represents the total heat released in kJ -/
def total_heat_released : ℝ := 3695

/-- Theorem stating that the molar ratio of H₂ to CH₄ in the original mixture is 1:3 -/
theorem molar_ratio_h2_ch4 :
  ∃ (x y : ℝ),
    x + y = mixture_volume / molar_volume ∧
    (heat_h2 / 2) * x + heat_ch4 * y = total_heat_released ∧
    x / y = 1 / 3 :=
sorry

end molar_ratio_h2_ch4_l395_39557


namespace valid_triples_eq_solution_set_l395_39522

def is_valid_triple (a m n : ℕ) : Prop :=
  a ≥ 2 ∧ m ≥ 2 ∧ (a^n + 203) % (a^(m*n) + 1) = 0

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(a, m, n) | 
    (∃ k, (a = 2 ∧ m = 2 ∧ n = 4*k + 1) ∨
          (a = 2 ∧ m = 3 ∧ n = 6*k + 2) ∨
          (a = 2 ∧ m = 4 ∧ n = 8*k + 8) ∨
          (a = 2 ∧ m = 6 ∧ n = 12*k + 9) ∨
          (a = 3 ∧ m = 2 ∧ n = 4*k + 3) ∨
          (a = 4 ∧ m = 2 ∧ n = 4*k + 4) ∨
          (a = 5 ∧ m = 2 ∧ n = 4*k + 1) ∨
          (a = 8 ∧ m = 2 ∧ n = 4*k + 3) ∨
          (a = 10 ∧ m = 2 ∧ n = 4*k + 2)) ∨
    (a = 203 ∧ m ≥ 2 ∧ ∃ k, n = (2*k + 1)*m + 1)}

theorem valid_triples_eq_solution_set :
  {(a, m, n) : ℕ × ℕ × ℕ | is_valid_triple a m n} = solution_set :=
sorry

end valid_triples_eq_solution_set_l395_39522


namespace calculate_expression_l395_39514

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x

-- Define the # operation
def hash_op (x y : ℤ) : ℤ := x * y + y

-- Theorem statement
theorem calculate_expression : (at_op 8 5) - (at_op 5 8) + (hash_op 8 5) = 36 := by
  sorry

end calculate_expression_l395_39514


namespace tan_beta_value_l395_39550

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : Real.tan (α + β) = -1) : 
  Real.tan β = 3 := by
sorry

end tan_beta_value_l395_39550


namespace intersection_of_A_and_B_l395_39575

def A : Set ℝ := {x | ∃ y, y = Real.log x}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l395_39575


namespace complex_fraction_simplification_l395_39509

theorem complex_fraction_simplification :
  ∃ (i : ℂ), i * i = -1 → (5 * i) / (1 - 2 * i) = -2 + i :=
by sorry

end complex_fraction_simplification_l395_39509


namespace all_propositions_false_l395_39576

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ " => parallel_lines
local infix:50 " ∥ " => parallel_line_plane
local infix:50 " ⊂ " => line_in_plane

-- Theorem statement
theorem all_propositions_false :
  (∀ (a b : Line) (α : Plane), (a ∥ b) → (b ⊂ α) → (a ∥ α)) = False ∧
  (∀ (a b : Line) (α : Plane), (a ∥ α) → (b ∥ α) → (a ∥ b)) = False ∧
  (∀ (a b : Line) (α : Plane), (a ∥ b) → (b ∥ α) → (a ∥ α)) = False ∧
  (∀ (a b : Line) (α : Plane), (a ∥ α) → (b ⊂ α) → (a ∥ b)) = False :=
by sorry

end all_propositions_false_l395_39576


namespace probability_intersection_is_zero_l395_39583

def f (x : Nat) : Int :=
  6 * x - 4

def g (x : Nat) : Int :=
  2 * x - 1

def domain : Finset Nat :=
  {1, 2, 3, 4, 5, 6}

def A : Finset Int :=
  Finset.image f domain

def B : Finset Int :=
  Finset.image g domain

theorem probability_intersection_is_zero :
  (A ∩ B).card / (A ∪ B).card = 0 :=
sorry

end probability_intersection_is_zero_l395_39583


namespace circle_op_example_l395_39501

/-- Custom binary operation on real numbers -/
def circle_op (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- The main theorem to prove -/
theorem circle_op_example : circle_op 9 (circle_op 4 3) = 32 := by
  sorry

end circle_op_example_l395_39501


namespace car_speed_difference_l395_39502

theorem car_speed_difference (distance : ℝ) (speed_R : ℝ) : 
  distance = 750 ∧ 
  speed_R = 56.44102863722254 → 
  ∃ (speed_P : ℝ), 
    distance / speed_P = distance / speed_R - 2 ∧ 
    speed_P > speed_R ∧ 
    speed_P - speed_R = 10 := by
  sorry

end car_speed_difference_l395_39502


namespace opposite_values_l395_39546

theorem opposite_values (x y : ℝ) : 
  |x + y - 9| + (2*x - y + 3)^2 = 0 → x = 2 ∧ y = 7 := by
  sorry

end opposite_values_l395_39546


namespace group_average_calculation_l395_39598

theorem group_average_calculation (initial_group_size : ℕ) 
  (new_member_amount : ℚ) (new_average : ℚ) : 
  initial_group_size = 7 → 
  new_member_amount = 56 → 
  new_average = 20 → 
  (initial_group_size * new_average + new_member_amount) / (initial_group_size + 1) = new_average → 
  new_average = 20 := by
  sorry

end group_average_calculation_l395_39598


namespace no_three_naturals_with_pairwise_sums_as_power_of_three_l395_39544

theorem no_three_naturals_with_pairwise_sums_as_power_of_three :
  ¬ ∃ (a b c : ℕ), 
    (∃ m : ℕ, a + b = 3^m) ∧ 
    (∃ n : ℕ, b + c = 3^n) ∧ 
    (∃ p : ℕ, c + a = 3^p) :=
sorry

end no_three_naturals_with_pairwise_sums_as_power_of_three_l395_39544


namespace quadratic_equation_solution_l395_39548

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 1)^2 = 4 ∧ (x₂ - 1)^2 = 4 ∧ x₁ = 3 ∧ x₂ = -1 :=
by sorry

end quadratic_equation_solution_l395_39548


namespace mrs_hilt_chickens_l395_39588

theorem mrs_hilt_chickens (total_legs : ℕ) (num_dogs : ℕ) (dog_legs : ℕ) (chicken_legs : ℕ) 
  (h1 : total_legs = 12)
  (h2 : num_dogs = 2)
  (h3 : dog_legs = 4)
  (h4 : chicken_legs = 2) :
  (total_legs - num_dogs * dog_legs) / chicken_legs = 2 := by
sorry

end mrs_hilt_chickens_l395_39588


namespace five_digit_divisible_by_twelve_l395_39504

theorem five_digit_divisible_by_twelve : ∃! (n : Nat), n < 10 ∧ 51470 + n ≡ 0 [MOD 12] := by
  sorry

end five_digit_divisible_by_twelve_l395_39504


namespace second_supply_cost_l395_39579

def first_supply_cost : ℕ := 13
def total_budget : ℕ := 56
def remaining_budget : ℕ := 19

theorem second_supply_cost :
  total_budget - remaining_budget - first_supply_cost = 24 :=
by sorry

end second_supply_cost_l395_39579


namespace largest_non_sum_of_composites_l395_39526

def IsComposite (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ 1 ∧ m ≠ n ∧ n % m = 0

def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end largest_non_sum_of_composites_l395_39526


namespace largest_root_of_g_l395_39574

-- Define the function g(x)
def g (x : ℝ) : ℝ := 12 * x^4 - 17 * x^2 + 5

-- State the theorem
theorem largest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 5 / 2 ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end largest_root_of_g_l395_39574


namespace P_no_real_roots_l395_39572

/-- Recursive definition of the polynomial sequence P_n(x) -/
def P (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => λ _ => 1
  | n + 1 => λ x => x^(11*(n+1)) - P n x

/-- Theorem stating that P_n(x) has no real roots for all n ≥ 0 -/
theorem P_no_real_roots (n : ℕ) : ∀ x : ℝ, P n x ≠ 0 := by
  sorry

end P_no_real_roots_l395_39572


namespace polynomial_sum_of_coefficients_l395_39553

theorem polynomial_sum_of_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x : ℝ, x^5 + 2 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) : 
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
  sorry

end polynomial_sum_of_coefficients_l395_39553


namespace back_seat_holds_eight_l395_39587

/-- Represents the seating capacity of a bus with specific arrangements --/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  people_per_seat : Nat
  total_capacity : Nat

/-- Calculates the number of people that can be seated at the back of the bus --/
def back_seat_capacity (bus : BusSeating) : Nat :=
  bus.total_capacity - (bus.left_seats + bus.right_seats) * bus.people_per_seat

/-- Theorem stating that for the given bus configuration, the back seat can hold 8 people --/
theorem back_seat_holds_eight :
  let bus : BusSeating := {
    left_seats := 15,
    right_seats := 12,
    people_per_seat := 3,
    total_capacity := 89
  }
  back_seat_capacity bus = 8 := by
  sorry

#eval back_seat_capacity {
  left_seats := 15,
  right_seats := 12,
  people_per_seat := 3,
  total_capacity := 89
}

end back_seat_holds_eight_l395_39587


namespace parallel_transitivity_counterexample_l395_39561

-- Define the types for lines and planes in 3D space
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (subset : Line → Plane → Prop)

theorem parallel_transitivity_counterexample 
  (m n : Line) (α β : Plane) :
  ¬(∀ (m n : Line) (α β : Plane), 
    parallel m n → 
    parallel_line_plane n α → 
    parallel_plane α β → 
    parallel_line_plane m β) :=
sorry

end parallel_transitivity_counterexample_l395_39561


namespace intersection_equality_implies_a_geq_5_l395_39577

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | x ≤ a}
def B : Set ℝ := {x | x^2 - 5*x < 0}

-- State the theorem
theorem intersection_equality_implies_a_geq_5 (a : ℝ) :
  A a ∩ B = B → a ≥ 5 := by sorry

end intersection_equality_implies_a_geq_5_l395_39577


namespace monotonic_increase_interval_l395_39570

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem monotonic_increase_interval (x : ℝ) :
  StrictMonoOn f (Set.Ici 2) :=
sorry

end monotonic_increase_interval_l395_39570


namespace geometric_sequence_sum_l395_39530

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 :=
by
  sorry

end geometric_sequence_sum_l395_39530


namespace students_interested_in_both_l395_39593

theorem students_interested_in_both (total : ℕ) (music : ℕ) (sports : ℕ) (neither : ℕ) :
  total = 55 →
  music = 35 →
  sports = 45 →
  neither = 4 →
  ∃ both : ℕ, both = 29 ∧ total = music + sports - both + neither :=
by sorry

end students_interested_in_both_l395_39593


namespace vectors_collinear_l395_39594

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors a, b, m, and n
variable (a b m n : V)

-- State the theorem
theorem vectors_collinear (h1 : m = a + b) (h2 : n = 2 • a + 2 • b) (h3 : ¬ Collinear ℝ ({0, a, b} : Set V)) :
  Collinear ℝ ({0, m, n} : Set V) := by
  sorry

end vectors_collinear_l395_39594


namespace bug_on_square_probability_l395_39590

/-- Probability of returning to the starting vertex after n moves -/
def P (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => 2/3 - (1/3) * P n

/-- The problem statement -/
theorem bug_on_square_probability : P 8 = 3248/6561 := by
  sorry

end bug_on_square_probability_l395_39590


namespace sequence_sixth_term_l395_39500

theorem sequence_sixth_term (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : a 2 = 2)
  (h3 : ∀ n ≥ 2, 2 * (a n)^2 = (a (n+1))^2 + (a (n-1))^2)
  (h4 : ∀ n, a n > 0) :
  a 6 = 4 := by
sorry

end sequence_sixth_term_l395_39500


namespace commission_increase_l395_39518

theorem commission_increase (total_sales : ℕ) (big_sale_commission : ℝ) (new_average : ℝ) :
  total_sales = 6 ∧ big_sale_commission = 1000 ∧ new_average = 250 →
  (new_average * total_sales - big_sale_commission) / (total_sales - 1) = 100 ∧
  new_average - (new_average * total_sales - big_sale_commission) / (total_sales - 1) = 150 := by
sorry

end commission_increase_l395_39518


namespace lines_planes_perpendicular_l395_39582

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem lines_planes_perpendicular
  (m n : Line) (α β : Plane)
  (h1 : parallel_lines m n)
  (h2 : parallel_line_plane m α)
  (h3 : perpendicular_line_plane n β) :
  perpendicular_planes α β :=
sorry

end lines_planes_perpendicular_l395_39582


namespace square_difference_l395_39521

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 7/13) 
  (h2 : x - y = 1/91) : 
  x^2 - y^2 = 1/169 := by
sorry

end square_difference_l395_39521


namespace complex_fraction_equality_l395_39556

theorem complex_fraction_equality (a b : ℂ) 
  (h : (a + b) / (a - b) - (a - b) / (a + b) = 0) : 
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = 0 := by
  sorry

end complex_fraction_equality_l395_39556


namespace count_solutions_l395_39517

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem count_solutions : 
  (Finset.filter (fun n => n + S n + S (S n) = 2007) (Finset.range 2008)).card = 4 := by sorry

end count_solutions_l395_39517


namespace cube_volume_ratio_l395_39578

theorem cube_volume_ratio :
  let cube1_edge : ℚ := 8
  let cube2_edge : ℚ := 16
  let volume_ratio := (cube1_edge ^ 3) / (cube2_edge ^ 3)
  volume_ratio = 1 / 8 := by sorry

end cube_volume_ratio_l395_39578


namespace profit_growth_equation_l395_39563

theorem profit_growth_equation (x : ℝ) : 
  (250000 : ℝ) * (1 + x)^2 = 360000 → 25 * (1 + x)^2 = 36 := by
sorry

end profit_growth_equation_l395_39563


namespace arithmetic_sequence_squared_l395_39529

theorem arithmetic_sequence_squared (x y z : ℝ) (h : y - x = z - y) :
  (x^2 + x*z + z^2) - (x^2 + x*y + y^2) = (y^2 + y*z + z^2) - (x^2 + x*z + z^2) :=
by sorry

end arithmetic_sequence_squared_l395_39529


namespace max_m2_plus_n2_l395_39558

theorem max_m2_plus_n2 : ∃ (m n : ℕ),
  1 ≤ m ∧ m ≤ 2005 ∧
  1 ≤ n ∧ n ≤ 2005 ∧
  (n^2 + 2*m*n - 2*m^2)^2 = 1 ∧
  m^2 + n^2 = 702036 ∧
  ∀ (m' n' : ℕ),
    1 ≤ m' ∧ m' ≤ 2005 ∧
    1 ≤ n' ∧ n' ≤ 2005 ∧
    (n'^2 + 2*m'*n' - 2*m'^2)^2 = 1 →
    m'^2 + n'^2 ≤ 702036 :=
by sorry

end max_m2_plus_n2_l395_39558


namespace monotonically_decreasing_interval_of_f_l395_39539

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

theorem monotonically_decreasing_interval_of_f :
  ∀ x ∈ Set.Ioo (0 : ℝ) (Real.sqrt 2 / 2),
    ∀ y ∈ Set.Ioo (0 : ℝ) (Real.sqrt 2 / 2),
      x < y → f x > f y :=
by sorry

end monotonically_decreasing_interval_of_f_l395_39539


namespace sqrt_one_third_equality_l395_39505

theorem sqrt_one_third_equality : 3 * Real.sqrt (1/3) = Real.sqrt 3 := by sorry

end sqrt_one_third_equality_l395_39505


namespace vector_equality_l395_39586

/-- Given vectors a, b, and c in R², prove that c = a - 3b -/
theorem vector_equality (a b c : Fin 2 → ℝ) 
  (ha : a = ![1, 1]) 
  (hb : b = ![1, -1]) 
  (hc : c = ![-2, 4]) : 
  c = a - 3 • b := by sorry

end vector_equality_l395_39586


namespace sqrt_product_plus_one_equals_1720_l395_39503

theorem sqrt_product_plus_one_equals_1720 : 
  Real.sqrt ((43 : ℝ) * 42 * 41 * 40 + 1) = 1720 := by
  sorry

end sqrt_product_plus_one_equals_1720_l395_39503


namespace last_bead_is_white_l395_39584

/-- Represents the color of a bead -/
inductive BeadColor
| White
| Black
| Red

/-- Returns the color of the nth bead in the pattern -/
def nthBeadColor (n : ℕ) : BeadColor :=
  match n % 6 with
  | 1 => BeadColor.White
  | 2 | 3 => BeadColor.Black
  | _ => BeadColor.Red

/-- The total number of beads in the necklace -/
def totalBeads : ℕ := 85

theorem last_bead_is_white :
  nthBeadColor totalBeads = BeadColor.White := by
  sorry

end last_bead_is_white_l395_39584


namespace fourth_root_of_105413504_l395_39599

theorem fourth_root_of_105413504 : (105413504 : ℝ) ^ (1/4 : ℝ) = 101 := by
  sorry

end fourth_root_of_105413504_l395_39599


namespace sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l395_39567

theorem sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds :
  7 < Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) ∧
  Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) < 8 :=
by sorry

end sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l395_39567


namespace lg_equation_l395_39513

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem lg_equation : lg 2 * lg 50 + lg 25 - lg 5 * lg 20 = 1 := by sorry

end lg_equation_l395_39513


namespace exponent_base_problem_l395_39528

theorem exponent_base_problem (x : ℝ) (y : ℝ) :
  4^(2*x + 2) = y^(3*x - 1) → x = 1 → y = 16 := by
  sorry

end exponent_base_problem_l395_39528


namespace decimal_rep_1_13_150th_digit_l395_39580

/-- The decimal representation of 1/13 as a sequence of digits -/
def decimalRep : ℕ → Fin 10 := 
  fun n => match n % 6 with
  | 0 => 0
  | 1 => 7
  | 2 => 6
  | 3 => 9
  | 4 => 2
  | 5 => 3
  | _ => 0  -- This case is unreachable, but needed for exhaustiveness

/-- The 150th digit after the decimal point in the decimal representation of 1/13 is 3 -/
theorem decimal_rep_1_13_150th_digit : decimalRep 150 = 3 := by
  sorry

end decimal_rep_1_13_150th_digit_l395_39580


namespace function_upper_bound_condition_l395_39542

theorem function_upper_bound_condition (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → a * x - x^2 ≤ 1) ↔ a ≤ 2 :=
by sorry

end function_upper_bound_condition_l395_39542


namespace stock_profit_is_447_5_l395_39515

/-- Calculate the profit from a stock transaction with given parameters -/
def calculate_profit (num_shares : ℕ) (buy_price sell_price : ℚ) 
  (stamp_duty_rate transfer_fee_rate commission_rate : ℚ) 
  (min_commission : ℚ) : ℚ :=
  let total_cost := num_shares * buy_price
  let total_income := num_shares * sell_price
  let total_transaction := total_cost + total_income
  let stamp_duty := total_transaction * stamp_duty_rate
  let transfer_fee := total_transaction * transfer_fee_rate
  let commission := max (total_transaction * commission_rate) min_commission
  total_income - total_cost - stamp_duty - transfer_fee - commission

/-- The profit from the given stock transaction is 447.5 yuan -/
theorem stock_profit_is_447_5 : 
  calculate_profit 1000 5 (11/2) (1/1000) (1/1000) (3/1000) 5 = 447.5 := by
  sorry

end stock_profit_is_447_5_l395_39515


namespace reciprocal_sum_of_roots_l395_39549

theorem reciprocal_sum_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → 
  x₂^2 - 3*x₂ + 2 = 0 → 
  x₁ ≠ 0 → 
  x₂ ≠ 0 → 
  1/x₁ + 1/x₂ = 3/2 := by
sorry

end reciprocal_sum_of_roots_l395_39549


namespace unripe_oranges_per_day_is_24_l395_39560

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges_per_day : ℕ := 24

/-- The total number of sacks of unripe oranges after the harvest period -/
def total_unripe_oranges : ℕ := 1080

/-- The number of days in the harvest period -/
def harvest_days : ℕ := 45

/-- Theorem stating that the number of sacks of unripe oranges harvested per day is 24 -/
theorem unripe_oranges_per_day_is_24 : 
  unripe_oranges_per_day = total_unripe_oranges / harvest_days :=
by sorry

end unripe_oranges_per_day_is_24_l395_39560


namespace marbles_won_l395_39540

theorem marbles_won (initial : ℕ) (lost : ℕ) (final : ℕ) : 
  initial = 57 → lost = 18 → final = 64 → final - (initial - lost) = 25 := by
  sorry

end marbles_won_l395_39540


namespace solution_set_of_equations_l395_39506

theorem solution_set_of_equations (a b c d : ℝ) : 
  (a * b * c + d = 2 ∧
   b * c * d + a = 2 ∧
   c * d * a + b = 2 ∧
   d * a * b + c = 2) ↔ 
  ((a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
   (a = 3 ∧ b = -1 ∧ c = -1 ∧ d = -1) ∨
   (a = -1 ∧ b = 3 ∧ c = -1 ∧ d = -1)) :=
by sorry

#check solution_set_of_equations

end solution_set_of_equations_l395_39506


namespace journey_start_time_l395_39508

/-- Two people moving towards each other -/
structure Journey where
  start_time : ℝ
  meet_time : ℝ
  a_finish_time : ℝ
  b_finish_time : ℝ

/-- The journey satisfies the problem conditions -/
def satisfies_conditions (j : Journey) : Prop :=
  j.meet_time = 12 ∧ 
  j.a_finish_time = 16 ∧ 
  j.b_finish_time = 21 ∧ 
  0 < j.start_time ∧ j.start_time < j.meet_time

/-- The equation representing the journey -/
def journey_equation (j : Journey) : Prop :=
  1 / (j.meet_time - j.start_time) + 
  1 / (j.a_finish_time - j.meet_time) + 
  1 / (j.b_finish_time - j.meet_time) = 1

theorem journey_start_time (j : Journey) 
  (h1 : satisfies_conditions j) 
  (h2 : journey_equation j) : 
  j.start_time = 6 := by
  sorry


end journey_start_time_l395_39508


namespace rectangles_in_5x5_grid_l395_39523

/-- The number of rectangles in a n×n grid -/
def rectangles_in_grid (n : ℕ) : ℕ := (n.choose 2) * (n.choose 2)

/-- Theorem: The number of rectangles in a 5×5 grid is 100 -/
theorem rectangles_in_5x5_grid : rectangles_in_grid 5 = 100 := by
  sorry

end rectangles_in_5x5_grid_l395_39523


namespace chrysler_building_floors_l395_39562

theorem chrysler_building_floors :
  ∀ (chrysler leeward : ℕ),
    chrysler = leeward + 11 →
    chrysler + leeward = 35 →
    chrysler = 23 :=
by
  sorry

end chrysler_building_floors_l395_39562


namespace pure_imaginary_condition_l395_39592

theorem pure_imaginary_condition (x : ℝ) : 
  (x^2 - x : ℂ) + (x - 1 : ℂ) * Complex.I = Complex.I * (y : ℝ) → x = 0 :=
by sorry

end pure_imaginary_condition_l395_39592


namespace equal_balance_after_10_days_l395_39545

/-- Carol's initial borrowing in clams -/
def carol_initial : ℝ := 200

/-- Emily's initial borrowing in clams -/
def emily_initial : ℝ := 250

/-- Carol's daily interest rate -/
def carol_rate : ℝ := 0.15

/-- Emily's daily interest rate -/
def emily_rate : ℝ := 0.10

/-- Number of days after which Carol and Emily owe the same amount -/
def days_equal : ℕ := 10

/-- Carol's balance after t days -/
def carol_balance (t : ℝ) : ℝ := carol_initial * (1 + carol_rate * t)

/-- Emily's balance after t days -/
def emily_balance (t : ℝ) : ℝ := emily_initial * (1 + emily_rate * t)

theorem equal_balance_after_10_days :
  carol_balance days_equal = emily_balance days_equal :=
by sorry

end equal_balance_after_10_days_l395_39545


namespace whale_plankton_theorem_l395_39534

/-- Calculates the total amount of plankton consumed by a whale during a 5-hour feeding frenzy -/
def whale_plankton_consumption (x : ℕ) : ℕ :=
  let hour1 := x
  let hour2 := x + 3
  let hour3 := x + 6
  let hour4 := x + 9
  let hour5 := x + 12
  hour1 + hour2 + hour3 + hour4 + hour5

/-- Theorem stating the total plankton consumption given the problem conditions -/
theorem whale_plankton_theorem : 
  ∀ x : ℕ, (x + 6 = 93) → whale_plankton_consumption x = 465 :=
by
  sorry

#eval whale_plankton_consumption 87

end whale_plankton_theorem_l395_39534


namespace no_prime_square_diff_4048_l395_39595

theorem no_prime_square_diff_4048 : ¬ ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p^2 - q^2 = 4048 := by
  sorry

end no_prime_square_diff_4048_l395_39595


namespace secret_santa_five_friends_l395_39547

/-- The number of derangements for n elements -/
def derangement (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement (n - 1) + derangement (n - 2))

/-- The number of ways to distribute gifts in a Secret Santa game -/
def secretSantaDistributions (n : ℕ) : ℕ := derangement n

theorem secret_santa_five_friends :
  secretSantaDistributions 5 = 44 := by
  sorry

#eval secretSantaDistributions 5

end secret_santa_five_friends_l395_39547


namespace random_number_table_sampling_sequence_l395_39569

-- Define the steps as an enumeration
inductive SamplingStep
  | NumberIndividuals
  | ObtainSampleNumbers
  | SelectStartingNumber

-- Define the correct sequence
def correctSequence : List SamplingStep :=
  [SamplingStep.NumberIndividuals, SamplingStep.SelectStartingNumber, SamplingStep.ObtainSampleNumbers]

-- Theorem statement
theorem random_number_table_sampling_sequence :
  correctSequence = [SamplingStep.NumberIndividuals, SamplingStep.SelectStartingNumber, SamplingStep.ObtainSampleNumbers] :=
by sorry

end random_number_table_sampling_sequence_l395_39569


namespace greatest_multiple_of_eight_remainder_l395_39552

/-- A function that checks if a natural number uses only unique digits from 1 to 9 -/
def uniqueDigits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 8 using unique digits from 1 to 9 -/
noncomputable def M : ℕ := sorry

theorem greatest_multiple_of_eight_remainder :
  M % 1000 = 976 ∧ M % 8 = 0 ∧ uniqueDigits M ∧ ∀ k : ℕ, k > M → k % 8 = 0 → ¬(uniqueDigits k) := by
  sorry

end greatest_multiple_of_eight_remainder_l395_39552


namespace x_range_theorem_l395_39597

-- Define the propositions p and q
def p (x : ℝ) : Prop := Real.log (x^2 - 2*x - 2) ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

-- Define the range of x
def range_of_x (x : ℝ) : Prop := x ≤ -1 ∨ (0 < x ∧ x < 3) ∨ x ≥ 4

-- Theorem statement
theorem x_range_theorem (x : ℝ) : 
  (¬(p x) ∧ ¬(q x)) ∧ (p x ∨ q x) → range_of_x x :=
by sorry

end x_range_theorem_l395_39597


namespace least_positive_integer_divisible_by_four_primes_l395_39581

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ n : ℕ+, (n : ℕ) = 210 ∧ 
  (∀ m : ℕ+, m < n → ¬(∃ p q r s : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧ 
  (m : ℕ) % p = 0 ∧ (m : ℕ) % q = 0 ∧ (m : ℕ) % r = 0 ∧ (m : ℕ) % s = 0)) ∧
  (∃ p q r s : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧ 
  210 % p = 0 ∧ 210 % q = 0 ∧ 210 % r = 0 ∧ 210 % s = 0) :=
by sorry

end least_positive_integer_divisible_by_four_primes_l395_39581
