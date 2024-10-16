import Mathlib

namespace NUMINAMATH_CALUDE_pizza_order_proof_l1398_139887

theorem pizza_order_proof (num_people : ℕ) (slices_per_pizza : ℕ) (slices_per_person : ℕ) : 
  num_people = 6 → slices_per_pizza = 8 → slices_per_person = 4 →
  (num_people * slices_per_person) / slices_per_pizza = 3 := by
sorry

end NUMINAMATH_CALUDE_pizza_order_proof_l1398_139887


namespace NUMINAMATH_CALUDE_emily_square_subtraction_l1398_139801

theorem emily_square_subtraction : 49^2 = 50^2 - 99 := by
  sorry

end NUMINAMATH_CALUDE_emily_square_subtraction_l1398_139801


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1398_139824

/-- Proves that a rectangular field with sides in ratio 3:4 and fencing cost of 91 rupees at 0.25 rupees per meter has an area of 8112 square meters -/
theorem rectangular_field_area (x : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  x > 0 →
  cost_per_meter = 0.25 →
  total_cost = 91 →
  (14 * x * cost_per_meter = total_cost) →
  (3 * x) * (4 * x) = 8112 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1398_139824


namespace NUMINAMATH_CALUDE_toms_age_ratio_l1398_139892

theorem toms_age_ratio (T N : ℝ) : 
  (T - N = 3 * (T - 4 * N)) → T / N = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l1398_139892


namespace NUMINAMATH_CALUDE_soccer_league_female_fraction_l1398_139813

theorem soccer_league_female_fraction :
  ∀ (m f : ℝ),
  m > 0 → f > 0 →
  1.05 * m + 1.2 * f = 1.1 * (m + f) →
  (1.2 * f) / (1.05 * m + 1.2 * f) = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_soccer_league_female_fraction_l1398_139813


namespace NUMINAMATH_CALUDE_reflected_ray_slope_l1398_139844

/-- A light ray is emitted from a point, reflects off the y-axis, and is tangent to a circle. -/
theorem reflected_ray_slope (emissionPoint : ℝ × ℝ) (circleCenter : ℝ × ℝ) (circleRadius : ℝ) :
  emissionPoint = (-2, -3) →
  circleCenter = (-3, 2) →
  circleRadius = 1 →
  ∃ (k : ℝ), (k = -4/3 ∨ k = -3/4) ∧
    (∀ (x y : ℝ), (x + 3)^2 + (y - 2)^2 = 1 →
      (k * x - y - 2 * k - 3 = 0 →
        ((3 * k + 2 + 2 * k + 3)^2 / (k^2 + 1) = 1))) :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_slope_l1398_139844


namespace NUMINAMATH_CALUDE_corresponds_to_zero_one_l1398_139870

/-- A mapping f from A to B where (x, y) in A corresponds to (x-1, 3-y) in B -/
def f (x y : ℝ) : ℝ × ℝ := (x - 1, 3 - y)

/-- Theorem stating that (1, 2) in A corresponds to (0, 1) in B under mapping f -/
theorem corresponds_to_zero_one : f 1 2 = (0, 1) := by
  sorry

end NUMINAMATH_CALUDE_corresponds_to_zero_one_l1398_139870


namespace NUMINAMATH_CALUDE_committee_selection_l1398_139861

theorem committee_selection (n : ℕ) (h : Nat.choose n 3 = 20) : Nat.choose n 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l1398_139861


namespace NUMINAMATH_CALUDE_sally_bread_consumption_l1398_139831

/-- The number of sandwiches Sally eats on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The number of sandwiches Sally eats on Sunday -/
def sunday_sandwiches : ℕ := 1

/-- The number of bread pieces used in each sandwich -/
def bread_per_sandwich : ℕ := 2

/-- The total number of bread pieces Sally eats across Saturday and Sunday -/
def total_bread_pieces : ℕ := (saturday_sandwiches * bread_per_sandwich) + (sunday_sandwiches * bread_per_sandwich)

theorem sally_bread_consumption :
  total_bread_pieces = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_bread_consumption_l1398_139831


namespace NUMINAMATH_CALUDE_simplified_fraction_sum_l1398_139897

theorem simplified_fraction_sum (a b : ℕ) (h : a = 75 ∧ b = 180) :
  let g := Nat.gcd a b
  (a / g) + (b / g) = 17 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_sum_l1398_139897


namespace NUMINAMATH_CALUDE_construct_line_segment_l1398_139873

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

end NUMINAMATH_CALUDE_construct_line_segment_l1398_139873


namespace NUMINAMATH_CALUDE_complement_proof_l1398_139827

def U : Set ℕ := {1,2,3,4,5,6,7,8,9,10}
def A : Set ℕ := {1,2,3,4,5,6}
def D : Set ℕ := {1,2,3}

theorem complement_proof :
  (Set.compl A : Set ℕ) = {7,8,9,10} ∧
  (A \ D : Set ℕ) = {4,5,6} := by
  sorry

end NUMINAMATH_CALUDE_complement_proof_l1398_139827


namespace NUMINAMATH_CALUDE_interest_period_is_two_years_l1398_139825

/-- Given simple interest rate of 20% per annum and simple interest of $400,
    and compound interest of $440 for the same period and rate,
    prove that the time period is 2 years. -/
theorem interest_period_is_two_years 
  (simple_interest : ℝ) 
  (compound_interest : ℝ) 
  (rate : ℝ) :
  simple_interest = 400 →
  compound_interest = 440 →
  rate = 0.20 →
  ∃ t : ℝ, t = 2 ∧ (1 + rate)^t = (rate * simple_interest * t + simple_interest) / simple_interest :=
by sorry

end NUMINAMATH_CALUDE_interest_period_is_two_years_l1398_139825


namespace NUMINAMATH_CALUDE_sunday_to_friday_spending_ratio_l1398_139867

def friday_spending : ℝ := 20

theorem sunday_to_friday_spending_ratio :
  ∀ (sunday_multiple : ℝ),
  friday_spending + 2 * friday_spending + sunday_multiple * friday_spending = 120 →
  sunday_multiple * friday_spending / friday_spending = 3 := by
  sorry

end NUMINAMATH_CALUDE_sunday_to_friday_spending_ratio_l1398_139867


namespace NUMINAMATH_CALUDE_second_team_made_131_pieces_l1398_139821

/-- The number of fish fillet pieces made by the second team -/
def second_team_pieces (total : ℕ) (first_team : ℕ) (third_team : ℕ) : ℕ :=
  total - (first_team + third_team)

/-- Theorem stating that the second team made 131 pieces of fish fillets -/
theorem second_team_made_131_pieces : 
  second_team_pieces 500 189 180 = 131 := by
  sorry

end NUMINAMATH_CALUDE_second_team_made_131_pieces_l1398_139821


namespace NUMINAMATH_CALUDE_compute_expression_l1398_139896

theorem compute_expression : 10 + 4 * (5 + 3)^3 = 2058 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1398_139896


namespace NUMINAMATH_CALUDE_lisa_photos_last_weekend_l1398_139893

/-- Calculates the number of photos Lisa took last weekend given the conditions --/
def photos_last_weekend (animal_photos : ℕ) (flower_multiplier : ℕ) (scenery_difference : ℕ) (weekend_difference : ℕ) : ℕ :=
  let flower_photos := animal_photos * flower_multiplier
  let scenery_photos := flower_photos - scenery_difference
  let total_photos := animal_photos + flower_photos + scenery_photos
  total_photos - weekend_difference

/-- Theorem stating that Lisa took 45 photos last weekend --/
theorem lisa_photos_last_weekend :
  photos_last_weekend 10 3 10 15 = 45 := by
  sorry

#eval photos_last_weekend 10 3 10 15

end NUMINAMATH_CALUDE_lisa_photos_last_weekend_l1398_139893


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1398_139817

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℝ, 
  (∀ x : ℝ, (x + 2) * (3 * x^2 - x + 5) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 21 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1398_139817


namespace NUMINAMATH_CALUDE_intersection_set_complement_l1398_139830

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | x > 0}

theorem intersection_set_complement : A ∩ (𝒰 \ B) = {x | -1 ≤ x ∧ x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_set_complement_l1398_139830


namespace NUMINAMATH_CALUDE_prob_sum_five_or_nine_l1398_139815

def fair_dice_roll : Finset ℕ := Finset.range 6

def sum_outcomes (n : ℕ) : Finset (ℕ × ℕ) :=
  (fair_dice_roll.product fair_dice_roll).filter (fun p => p.1 + p.2 + 2 = n)

def prob_sum (n : ℕ) : ℚ :=
  (sum_outcomes n).card / (fair_dice_roll.card * fair_dice_roll.card : ℕ)

theorem prob_sum_five_or_nine :
  prob_sum 5 = 1/9 ∧ prob_sum 9 = 1/9 :=
sorry

end NUMINAMATH_CALUDE_prob_sum_five_or_nine_l1398_139815


namespace NUMINAMATH_CALUDE_vector_sum_l1398_139849

theorem vector_sum : 
  let v1 : Fin 3 → ℝ := ![4, -8, 10]
  let v2 : Fin 3 → ℝ := ![-7, 12, -15]
  v1 + v2 = ![-3, 4, -5] := by sorry

end NUMINAMATH_CALUDE_vector_sum_l1398_139849


namespace NUMINAMATH_CALUDE_not_divisible_by_11_l1398_139869

theorem not_divisible_by_11 : ¬(11 ∣ 98473092) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_11_l1398_139869


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1398_139804

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^4 + 8 * X^3 - 27 * X^2 - 32 * X + 52 = 
  (X^2 + 5 * X + 2) * q + (52 * X + 80) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1398_139804


namespace NUMINAMATH_CALUDE_blackboard_numbers_l1398_139832

theorem blackboard_numbers (a b : ℕ) : 
  (¬ ∃ a b : ℕ, 13 * a + 11 * b = 86) ∧ 
  (∃ a b : ℕ, 13 * a + 11 * b = 2015) := by
  sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l1398_139832


namespace NUMINAMATH_CALUDE_max_value_fraction_max_value_achievable_l1398_139881

theorem max_value_fraction (y : ℝ) :
  y^2 / (y^4 + 4*y^3 + y^2 + 8*y + 16) ≤ 1/25 :=
by sorry

theorem max_value_achievable :
  ∃ y : ℝ, y^2 / (y^4 + 4*y^3 + y^2 + 8*y + 16) = 1/25 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_max_value_achievable_l1398_139881


namespace NUMINAMATH_CALUDE_prob_at_least_one_of_two_is_eight_ninths_l1398_139872

/-- The number of candidates -/
def num_candidates : ℕ := 2

/-- The number of colleges -/
def num_colleges : ℕ := 3

/-- The probability of a candidate choosing any particular college -/
def prob_choose_college : ℚ := 1 / num_colleges

/-- The probability that both candidates choose the third college -/
def prob_both_choose_third : ℚ := prob_choose_college ^ num_candidates

/-- The probability that at least one of the first two colleges is selected -/
def prob_at_least_one_of_two : ℚ := 1 - prob_both_choose_third

theorem prob_at_least_one_of_two_is_eight_ninths :
  prob_at_least_one_of_two = 8 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_of_two_is_eight_ninths_l1398_139872


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1398_139820

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-2 : ℝ) 1 = {x | a * x^2 + b * x + c > 0}) :
  {x : ℝ | a * x^2 + (a + b) * x + c - a < 0} = 
    Set.Iic (-3 : ℝ) ∪ Set.Ioi (1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1398_139820


namespace NUMINAMATH_CALUDE_first_hundred_contains_all_naturals_l1398_139818

/-- A sequence of 200 numbers partitioned into blue and red -/
def Sequence := Fin 200 → ℕ

/-- The property that blue numbers are in ascending order from 1 to 100 -/
def BlueAscending (s : Sequence) : Prop :=
  ∃ (blue : Fin 200 → Bool),
    (∀ i : Fin 200, blue i → s i ∈ Finset.range 101) ∧
    (∀ i j : Fin 200, i < j → blue i → blue j → s i < s j)

/-- The property that red numbers are in descending order from 100 to 1 -/
def RedDescending (s : Sequence) : Prop :=
  ∃ (red : Fin 200 → Bool),
    (∀ i : Fin 200, red i → s i ∈ Finset.range 101) ∧
    (∀ i j : Fin 200, i < j → red i → red j → s i > s j)

/-- The main theorem -/
theorem first_hundred_contains_all_naturals (s : Sequence)
    (h1 : BlueAscending s) (h2 : RedDescending s) :
    ∀ n : ℕ, n ∈ Finset.range 101 → ∃ i : Fin 100, s i = n :=
  sorry

end NUMINAMATH_CALUDE_first_hundred_contains_all_naturals_l1398_139818


namespace NUMINAMATH_CALUDE_find_y_value_l1398_139865

theorem find_y_value (x y : ℝ) (h1 : x^2 - x + 3 = y + 3) (h2 : x = -5) (h3 : y > 0) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l1398_139865


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1398_139856

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 →
  Real.sqrt ((x - 2)^2 + (10 - 5)^2) = 13 →
  x = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1398_139856


namespace NUMINAMATH_CALUDE_line_graph_displays_trend_l1398_139876

/-- Enumeration of statistical graph types --/
inductive StatGraphType
  | BarGraph
  | LineGraph
  | PieChart
  | Histogram

/-- Function to determine if a graph type can display trends --/
def canDisplayTrend (t : StatGraphType) : Prop :=
  match t with
  | StatGraphType.LineGraph => True
  | _ => False

/-- Theorem stating that the line graph is the only type that can display trends --/
theorem line_graph_displays_trend :
  ∀ t : StatGraphType, canDisplayTrend t ↔ t = StatGraphType.LineGraph :=
sorry

end NUMINAMATH_CALUDE_line_graph_displays_trend_l1398_139876


namespace NUMINAMATH_CALUDE_simplify_irrational_denominator_l1398_139814

theorem simplify_irrational_denominator :
  (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 2) = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_irrational_denominator_l1398_139814


namespace NUMINAMATH_CALUDE_triangle_area_l1398_139884

/-- The area of a triangle with vertices at (5, -2), (5, 8), and (12, 8) is 35 square units. -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (5, -2)
  let v2 : ℝ × ℝ := (5, 8)
  let v3 : ℝ × ℝ := (12, 8)
  let area := (1/2) * abs ((v2.1 - v1.1) * (v3.2 - v1.2) - (v3.1 - v1.1) * (v2.2 - v1.2))
  area = 35 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1398_139884


namespace NUMINAMATH_CALUDE_volleyball_team_size_l1398_139898

theorem volleyball_team_size (managers : ℕ) (employees : ℕ) (teams : ℕ) :
  managers = 23 →
  employees = 7 →
  teams = 6 →
  (managers + employees) / teams = 5 := by
sorry

end NUMINAMATH_CALUDE_volleyball_team_size_l1398_139898


namespace NUMINAMATH_CALUDE_car_value_decrease_l1398_139850

theorem car_value_decrease (original_value current_value : ℝ) 
  (h1 : original_value = 4000)
  (h2 : current_value = 2800) :
  (original_value - current_value) / original_value * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_car_value_decrease_l1398_139850


namespace NUMINAMATH_CALUDE_f12_roots_l1398_139826

/-- Quadratic polynomial f_i(x) = x^2 + b_i*x + c_i -/
def f (i : ℕ) (x : ℝ) (b c : ℕ → ℝ) : ℝ := x^2 + b i * x + c i

/-- Relation between consecutive b_i coefficients -/
axiom b_relation (b : ℕ → ℝ) : ∀ i, b (i + 1) = 2 * b i

/-- Relation for c_i coefficients -/
axiom c_relation (b c : ℕ → ℝ) : ∀ i, c i = -32 * b i - 1024

/-- Roots of f_1(x) -/
axiom f1_roots (b c : ℕ → ℝ) : 
  ∃ x y, x = 32 ∧ y = -31 ∧ (f 1 x b c = 0 ∧ f 1 y b c = 0)

/-- Theorem: Roots of f_12(x) are 2016 and 32 -/
theorem f12_roots (b c : ℕ → ℝ) : 
  ∃ x y, x = 2016 ∧ y = 32 ∧ (f 12 x b c = 0 ∧ f 12 y b c = 0) := by
  sorry

end NUMINAMATH_CALUDE_f12_roots_l1398_139826


namespace NUMINAMATH_CALUDE_divisibility_condition_l1398_139883

def six_digit_number (n : ℕ) : ℕ := 850000 + n * 1000 + 475

theorem divisibility_condition (n : ℕ) : 
  n < 10 → (six_digit_number n % 45 = 0 ↔ n = 7) := by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1398_139883


namespace NUMINAMATH_CALUDE_third_circle_radius_l1398_139875

theorem third_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 19) (h₂ : r₂ = 29) :
  (r₂^2 - r₁^2) * π = π * r₃^2 → r₃ = 4 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l1398_139875


namespace NUMINAMATH_CALUDE_book_length_l1398_139877

theorem book_length (pages_read : ℚ) (pages_left : ℚ) : 
  pages_read = 2 / 3 * (pages_read + pages_left) →
  pages_left = 1 / 3 * (pages_read + pages_left) →
  pages_read = pages_left + 100 →
  pages_read + pages_left = 300 := by
  sorry

end NUMINAMATH_CALUDE_book_length_l1398_139877


namespace NUMINAMATH_CALUDE_prism_volume_l1398_139839

/-- The volume of a right rectangular prism with specific face areas and dimension ratio -/
theorem prism_volume (l w h : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  l * w = 10 → w * h = 18 → l * h = 36 →
  l = 2 * w →
  l * w * h = 36 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_l1398_139839


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1398_139822

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (5 * x + 1 / (3 * x)) ^ 8
  ∃ (p q : ℝ → ℝ), expansion = p x + (43750 / 81) + q x ∧ 
    (∀ y, y ≠ 0 → p y + q y = 0) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1398_139822


namespace NUMINAMATH_CALUDE_simplify_fraction_l1398_139833

theorem simplify_fraction (b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ -1/2) :
  1 - 1 / (1 + b / (1 + b)) = b / (1 + 2*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1398_139833


namespace NUMINAMATH_CALUDE_matrix_cube_l1398_139847

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 1, 1]

theorem matrix_cube : A^3 = !![3, -6; 6, -3] := by sorry

end NUMINAMATH_CALUDE_matrix_cube_l1398_139847


namespace NUMINAMATH_CALUDE_monkey_climb_height_l1398_139854

/-- The height of the tree that the monkey climbs -/
def tree_height : ℕ := 22

/-- The distance the monkey climbs up each hour -/
def climb_distance : ℕ := 3

/-- The distance the monkey slips back each hour -/
def slip_distance : ℕ := 2

/-- The total time it takes for the monkey to reach the top of the tree -/
def total_time : ℕ := 20

/-- Theorem stating that the height of the tree is 22 ft -/
theorem monkey_climb_height :
  tree_height = (total_time - 1) * (climb_distance - slip_distance) + climb_distance :=
by sorry

end NUMINAMATH_CALUDE_monkey_climb_height_l1398_139854


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1398_139882

theorem no_positive_integer_solutions :
  ¬ ∃ (x y z : ℕ+), x^4004 + y^4004 = z^2002 :=
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1398_139882


namespace NUMINAMATH_CALUDE_difference_in_base8_l1398_139808

/-- Converts a base 8 number represented as a list of digits to its decimal equivalent -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a decimal number to its base 8 representation as a list of digits -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec convert (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else convert (m / 8) ((m % 8) :: acc)
    convert n []

theorem difference_in_base8 :
  let a := base8ToDecimal [1, 2, 3, 4]
  let b := base8ToDecimal [7, 6, 5]
  decimalToBase8 (a - b) = [2, 2, 5] :=
by sorry

end NUMINAMATH_CALUDE_difference_in_base8_l1398_139808


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1398_139853

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

theorem circle_intersection_range (r : ℝ) (h1 : r > 0) (h2 : M ∩ N r = N r) :
  r ∈ Set.Ioo 0 (2 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1398_139853


namespace NUMINAMATH_CALUDE_checkerboard_triangle_area_theorem_l1398_139888

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A triangle on a 2D grid -/
structure GridTriangle where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint

/-- The area of a triangle -/
def triangleArea (t : GridTriangle) : ℝ :=
  sorry

/-- Whether two triangles are similar -/
def areSimilar (t1 t2 : GridTriangle) : Prop :=
  sorry

/-- The area of the white part of a triangle -/
def whiteArea (t : GridTriangle) : ℝ :=
  sorry

/-- The area of the black part of a triangle -/
def blackArea (t : GridTriangle) : ℝ :=
  sorry

/-- The main theorem -/
theorem checkerboard_triangle_area_theorem (X : GridTriangle) (S : ℝ) 
  (h : triangleArea X = S) : 
  ∃ Y : GridTriangle, areSimilar X Y ∧ whiteArea Y = S ∧ blackArea Y = S :=
sorry

end NUMINAMATH_CALUDE_checkerboard_triangle_area_theorem_l1398_139888


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1398_139862

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of a geometric sequence with first term 1/2 and common ratio 1/3 -/
theorem geometric_sequence_sum : 
  geometric_sum (1/2) (1/3) 8 = 4920/6561 := by
  sorry

#eval geometric_sum (1/2) (1/3) 8

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1398_139862


namespace NUMINAMATH_CALUDE_junipers_bones_l1398_139857

theorem junipers_bones (initial_bones : ℕ) : 
  (2 * initial_bones - 2 = 6) → initial_bones = 4 := by
  sorry

end NUMINAMATH_CALUDE_junipers_bones_l1398_139857


namespace NUMINAMATH_CALUDE_derivative_of_even_function_l1398_139886

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define the property that f is even
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem derivative_of_even_function (hf : IsEven f) :
  ∀ x, (deriv f) (-x) = -(deriv f x) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_even_function_l1398_139886


namespace NUMINAMATH_CALUDE_certain_number_proof_l1398_139809

theorem certain_number_proof (k : ℤ) (x : ℝ) 
  (h1 : x * (10 : ℝ)^(k : ℝ) > 100)
  (h2 : ∀ m : ℝ, m < 4.9956356288922485 → x * (10 : ℝ)^m ≤ 100) :
  x = 0.00101 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1398_139809


namespace NUMINAMATH_CALUDE_sector_central_angle_invariant_l1398_139836

/-- Theorem: If both the radius and arc length of a circular sector are doubled, then the central angle of the sector remains unchanged. -/
theorem sector_central_angle_invariant 
  (r₁ r₂ l₁ l₂ θ₁ θ₂ : Real) 
  (h1 : r₂ = 2 * r₁) 
  (h2 : l₂ = 2 * l₁) 
  (h3 : θ₁ = l₁ / r₁) 
  (h4 : θ₂ = l₂ / r₂) : 
  θ₁ = θ₂ := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_invariant_l1398_139836


namespace NUMINAMATH_CALUDE_cost_reduction_equation_l1398_139860

theorem cost_reduction_equation (x : ℝ) : 
  (∀ (total_reduction : ℝ), total_reduction = 0.36 → 
    ((1 - x) ^ 2 = 1 - total_reduction)) ↔ 
  ((1 - x) ^ 2 = 1 - 0.36) :=
sorry

end NUMINAMATH_CALUDE_cost_reduction_equation_l1398_139860


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_min_value_exists_l1398_139841

theorem min_value_of_function (x : ℝ) (h : x > 0) : 2 + 4*x + 1/x ≥ 6 :=
sorry

theorem equality_condition (x : ℝ) (h : x > 0) : 2 + 4*x + 1/x = 6 ↔ x = 1/2 :=
sorry

theorem min_value_exists : ∃ x : ℝ, x > 0 ∧ 2 + 4*x + 1/x = 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_min_value_exists_l1398_139841


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1398_139879

/-- The distance between vertices of a hyperbola -/
def distance_between_vertices (a b : ℝ) : ℝ := 2 * a

/-- The equation of the hyperbola -/
def is_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / (a^2) - y^2 / (b^2) = 1

theorem hyperbola_vertices_distance :
  ∀ x y : ℝ, is_hyperbola x y 4 2 → distance_between_vertices 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1398_139879


namespace NUMINAMATH_CALUDE_greatest_integer_jo_l1398_139852

theorem greatest_integer_jo (n : ℕ) : 
  n > 0 ∧ 
  n < 150 ∧ 
  ∃ k : ℕ, n + 2 = 9 * k ∧ 
  ∃ l : ℕ, n + 4 = 8 * l →
  n ≤ 146 ∧ 
  ∃ m : ℕ, 146 > 0 ∧ 
  146 < 150 ∧ 
  146 + 2 = 9 * m ∧ 
  ∃ p : ℕ, 146 + 4 = 8 * p :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_jo_l1398_139852


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_congruences_l1398_139863

theorem smallest_number_satisfying_congruences : ∃ n : ℕ, 
  n > 0 ∧
  n % 4 = 1 ∧
  n % 5 = 1 ∧
  n % 6 = 1 ∧
  n % 7 = 0 ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 1 ∧ m % 6 = 1 ∧ m % 7 = 0 → m ≥ n) ∧
  n = 301 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_congruences_l1398_139863


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1398_139838

-- Define the polynomial
def p (z : ℂ) : ℂ := (z - 2) * (z^2 + 4*z + 10) * (z^2 + 6*z + 13)

-- Define the set of solutions
def solutions : Set ℂ := {z : ℂ | p z = 0}

-- Define the ellipse passing through the solutions
def E : Set ℂ := sorry

-- Define eccentricity
def eccentricity (E : Set ℂ) : ℝ := sorry

-- Theorem statement
theorem ellipse_eccentricity : eccentricity E = Real.sqrt (4/25) := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1398_139838


namespace NUMINAMATH_CALUDE_applicants_with_experience_and_degree_l1398_139878

theorem applicants_with_experience_and_degree 
  (total : ℕ) 
  (experienced : ℕ) 
  (degreed : ℕ) 
  (inexperienced_no_degree : ℕ) 
  (h1 : total = 30)
  (h2 : experienced = 10)
  (h3 : degreed = 18)
  (h4 : inexperienced_no_degree = 3) :
  total - (experienced + degreed - (total - inexperienced_no_degree)) = 1 := by
sorry

end NUMINAMATH_CALUDE_applicants_with_experience_and_degree_l1398_139878


namespace NUMINAMATH_CALUDE_tulip_fraction_l1398_139800

-- Define the total number of flowers (arbitrary positive real number)
variable (total : ℝ) (total_pos : 0 < total)

-- Define the number of each type of flower
variable (pink_roses : ℝ) (red_roses : ℝ) (pink_tulips : ℝ) (red_tulips : ℝ)

-- All flowers are either roses or tulips, and either pink or red
axiom flower_sum : pink_roses + red_roses + pink_tulips + red_tulips = total

-- 1/4 of pink flowers are roses
axiom pink_rose_ratio : pink_roses = (1/4) * (pink_roses + pink_tulips)

-- 1/3 of red flowers are tulips
axiom red_tulip_ratio : red_tulips = (1/3) * (red_roses + red_tulips)

-- 7/10 of all flowers are red
axiom red_flower_ratio : red_roses + red_tulips = (7/10) * total

-- Theorem: The fraction of flowers that are tulips is 11/24
theorem tulip_fraction :
  (pink_tulips + red_tulips) / total = 11/24 := by sorry

end NUMINAMATH_CALUDE_tulip_fraction_l1398_139800


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_prism_inscribed_l1398_139811

/-- Given a rectangular prism with adjacent face areas of 2, 3, and 6,
    and all vertices lying on the same spherical surface,
    prove that the surface area of this sphere is 14π. -/
theorem sphere_surface_area_of_prism_inscribed (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b = 6 → b * c = 2 → a * c = 3 →
  (4 : ℝ) * Real.pi * ((a^2 + b^2 + c^2) / 4) = 14 * Real.pi := by
  sorry

#check sphere_surface_area_of_prism_inscribed

end NUMINAMATH_CALUDE_sphere_surface_area_of_prism_inscribed_l1398_139811


namespace NUMINAMATH_CALUDE_computer_profit_percentage_l1398_139895

theorem computer_profit_percentage (C : ℝ) (P : ℝ) : 
  2560 = C + 0.6 * C →
  2240 = C + P / 100 * C →
  P = 40 := by
sorry

end NUMINAMATH_CALUDE_computer_profit_percentage_l1398_139895


namespace NUMINAMATH_CALUDE_min_value_theorem_l1398_139802

theorem min_value_theorem (n : ℕ+) (a : ℝ) (x : ℝ) (ha : a > 0) (hx : x > 0) :
  (a^n.val + x^n.val) * (a + x)^n.val / x^n.val ≥ 2^(n.val + 1) * a^n.val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1398_139802


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1398_139806

/-- The number of players in the basketball team -/
def total_players : ℕ := 16

/-- The number of players to be chosen for a game -/
def team_size : ℕ := 7

/-- The number of players excluding the twins -/
def players_without_twins : ℕ := total_players - 2

/-- The number of ways to choose the team with the given conditions -/
def ways_to_choose_team : ℕ := Nat.choose players_without_twins team_size + Nat.choose players_without_twins (team_size - 2)

theorem basketball_team_selection :
  ways_to_choose_team = 5434 := by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1398_139806


namespace NUMINAMATH_CALUDE_wilson_cola_purchase_wilson_cola_purchase_correct_l1398_139835

theorem wilson_cola_purchase (hamburger_cost : ℕ) (total_cost : ℕ) (discount : ℕ) (cola_cost : ℕ) : ℕ :=
  let hamburgers := 2
  let hamburger_total := hamburgers * hamburger_cost
  let discounted_hamburger_cost := hamburger_total - discount
  let cola_total := total_cost - discounted_hamburger_cost
  cola_total / cola_cost

#check wilson_cola_purchase 5 12 4 2

theorem wilson_cola_purchase_correct : wilson_cola_purchase 5 12 4 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_wilson_cola_purchase_wilson_cola_purchase_correct_l1398_139835


namespace NUMINAMATH_CALUDE_cylinder_volume_with_square_section_l1398_139899

/-- Given a cylinder with a square axial section of area 4, its volume is 2π. -/
theorem cylinder_volume_with_square_section (r h : ℝ) : 
  r * h = 2 →  -- The axial section is a square
  r * r * h = 4 →  -- The area of the square is 4
  π * r * r * h = 2 * π :=  -- The volume of the cylinder is 2π
by
  sorry

#check cylinder_volume_with_square_section

end NUMINAMATH_CALUDE_cylinder_volume_with_square_section_l1398_139899


namespace NUMINAMATH_CALUDE_quadratic_other_intercept_l1398_139890

/-- Given a quadratic function f(x) = ax² + bx + c with vertex (5, -3) and 
    one x-intercept at (1, 0), the x-coordinate of the other x-intercept is 9. -/
theorem quadratic_other_intercept 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 1 = 0) 
  (h3 : (5 : ℝ) = -b / (2 * a)) 
  (h4 : f 5 = -3) : 
  ∃ x, x ≠ 1 ∧ f x = 0 ∧ x = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_other_intercept_l1398_139890


namespace NUMINAMATH_CALUDE_competition_end_time_l1398_139810

-- Define the start time of the competition
def start_time : Nat := 15 * 60  -- 3:00 p.m. in minutes since midnight

-- Define the duration of the competition
def duration : Nat := 875  -- in minutes

-- Define the end time of the competition
def end_time : Nat := (start_time + duration) % (24 * 60)

-- Theorem to prove
theorem competition_end_time :
  end_time = 5 * 60 + 35  -- 5:35 a.m. in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_competition_end_time_l1398_139810


namespace NUMINAMATH_CALUDE_jackie_apples_l1398_139894

-- Define the number of apples Adam has
def adam_apples : ℕ := 8

-- Define the difference between Jackie's and Adam's apples
def difference : ℕ := 2

-- Theorem: Jackie has 10 apples
theorem jackie_apples : adam_apples + difference = 10 := by
  sorry

end NUMINAMATH_CALUDE_jackie_apples_l1398_139894


namespace NUMINAMATH_CALUDE_rainfall_difference_l1398_139874

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


end NUMINAMATH_CALUDE_rainfall_difference_l1398_139874


namespace NUMINAMATH_CALUDE_min_value_a5_plus_a6_l1398_139845

/-- A positive arithmetic geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 1 ∧ ∀ n, a (n + 1) = q * a n ∧ a n > 0

/-- The theorem statement -/
theorem min_value_a5_plus_a6 (a : ℕ → ℝ) 
  (h_seq : ArithmeticGeometricSequence a) 
  (h_cond : a 4 + a 3 - a 2 - a 1 = 1) :
  ∃ (min_val : ℝ), min_val = 4 ∧ 
    (∀ a5a6, a5a6 = a 5 + a 6 → a5a6 ≥ min_val) ∧
    (∃ a5a6, a5a6 = a 5 + a 6 ∧ a5a6 = min_val) :=
sorry

end NUMINAMATH_CALUDE_min_value_a5_plus_a6_l1398_139845


namespace NUMINAMATH_CALUDE_length_BI_isosceles_triangle_l1398_139868

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Isosceles condition
  isIsosceles : ab > 0 ∧ bc > 0

/-- The incenter of a triangle -/
def incenter (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Length of BI in isosceles triangle ABC -/
theorem length_BI_isosceles_triangle (t : IsoscelesTriangle) 
  (h1 : t.ab = 6) 
  (h2 : t.bc = 8) : 
  ∃ (ε : ℝ), abs (distance (0, 0) (incenter t) - 4.4 * Real.sqrt 1.1) < ε :=
sorry

end NUMINAMATH_CALUDE_length_BI_isosceles_triangle_l1398_139868


namespace NUMINAMATH_CALUDE_cost_difference_white_brown_socks_l1398_139837

-- Define the cost of two white socks in cents
def cost_two_white_socks : ℕ := 45

-- Define the cost of 15 brown socks in cents
def cost_fifteen_brown_socks : ℕ := 300

-- Define the number of brown socks
def num_brown_socks : ℕ := 15

-- Theorem to prove
theorem cost_difference_white_brown_socks : 
  cost_two_white_socks - (cost_fifteen_brown_socks / num_brown_socks) = 25 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_white_brown_socks_l1398_139837


namespace NUMINAMATH_CALUDE_perpendicular_lines_and_intersection_l1398_139803

-- Define the slopes and y-intercept of the lines
def m_s : ℚ := 4/3
def b_s : ℚ := -100
def m_t : ℚ := -3/4

-- Define the lines
def line_s (x : ℚ) : ℚ := m_s * x + b_s
def line_t (x : ℚ) : ℚ := m_t * x

-- Define the intersection point
def intersection_x : ℚ := 48
def intersection_y : ℚ := -36

theorem perpendicular_lines_and_intersection :
  -- Line t is perpendicular to line s
  m_s * m_t = -1 ∧
  -- Line t passes through (0, 0)
  line_t 0 = 0 ∧
  -- The intersection point satisfies both line equations
  line_s intersection_x = intersection_y ∧
  line_t intersection_x = intersection_y :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_and_intersection_l1398_139803


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_2017_l1398_139829

theorem tens_digit_of_13_pow_2017 : ∃ n : ℕ, 13^2017 ≡ 30 + n [ZMOD 100] :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_2017_l1398_139829


namespace NUMINAMATH_CALUDE_isabel_paper_left_l1398_139866

/-- The number of pieces of paper Isabel bought in her first purchase -/
def first_purchase : ℕ := 900

/-- The number of pieces of paper Isabel bought in her second purchase -/
def second_purchase : ℕ := 300

/-- The number of pieces of paper Isabel used for a school project -/
def school_project : ℕ := 156

/-- The number of pieces of paper Isabel used for her artwork -/
def artwork : ℕ := 97

/-- The number of pieces of paper Isabel used for writing letters -/
def letters : ℕ := 45

/-- The theorem stating that Isabel has 902 pieces of paper left -/
theorem isabel_paper_left : 
  first_purchase + second_purchase - (school_project + artwork + letters) = 902 := by
  sorry

end NUMINAMATH_CALUDE_isabel_paper_left_l1398_139866


namespace NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l1398_139889

/-- A hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0 -/
structure Hyperbola where
  b : ℝ
  h_pos : b > 0

/-- The asymptote of a hyperbola -/
structure Asymptote where
  slope : ℝ
  y_intercept : ℝ

/-- The focus of a hyperbola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Distance between a point and a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : Asymptote) : ℝ :=
  sorry

/-- Theorem: If one asymptote of the hyperbola x^2 - y^2/b^2 = 1 (b > 0) is y = 2x, 
    then the distance from the focus to this asymptote is 2 -/
theorem hyperbola_focus_asymptote_distance 
  (h : Hyperbola) 
  (a : Asymptote) 
  (f : Focus) 
  (h_asymptote : a.slope = 2 ∧ a.y_intercept = 0) : 
  distance_point_to_line (f.x, f.y) a = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l1398_139889


namespace NUMINAMATH_CALUDE_island_puzzle_l1398_139864

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


end NUMINAMATH_CALUDE_island_puzzle_l1398_139864


namespace NUMINAMATH_CALUDE_oak_trees_remaining_l1398_139891

/-- The number of oak trees remaining in the park after cutting down damaged trees -/
def remaining_oak_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that the number of remaining oak trees is 7 -/
theorem oak_trees_remaining :
  remaining_oak_trees 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_remaining_l1398_139891


namespace NUMINAMATH_CALUDE_base_r_transaction_l1398_139858

def base_r_to_decimal (digits : List Nat) (r : Nat) : Nat :=
  digits.foldl (fun acc d => acc * r + d) 0

theorem base_r_transaction (r : Nat) : r = 8 :=
  by
  have h1 : base_r_to_decimal [4, 4, 0] r + base_r_to_decimal [3, 4, 0] r = base_r_to_decimal [1, 0, 0, 0] r :=
    sorry
  sorry

end NUMINAMATH_CALUDE_base_r_transaction_l1398_139858


namespace NUMINAMATH_CALUDE_major_axis_length_is_three_l1398_139842

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * (1 + major_minor_ratio)

/-- Theorem: The major axis length is 3 when the cylinder radius is 1 and the major axis is 50% longer than the minor axis -/
theorem major_axis_length_is_three :
  major_axis_length 1 0.5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_is_three_l1398_139842


namespace NUMINAMATH_CALUDE_max_value_2x_plus_y_l1398_139805

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2*y ≤ 3) (h2 : x ≥ 0) (h3 : y ≥ 0) :
  (∀ x' y' : ℝ, x' + 2*y' ≤ 3 → x' ≥ 0 → y' ≥ 0 → 2*x' + y' ≤ 2*x + y) →
  2*x + y = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_plus_y_l1398_139805


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l1398_139828

/-- Two lines in the form ax + by + c = 0 are parallel if and only if their slopes are equal -/
def are_parallel (a1 b1 a2 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

/-- The first line equation: x + ay + 3 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 3 = 0

/-- The second line equation: (a-2)x + 3y + a = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y + a = 0

theorem parallel_lines_condition (a : ℝ) : 
  are_parallel 1 a (a - 2) 3 ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l1398_139828


namespace NUMINAMATH_CALUDE_desired_weather_probability_l1398_139871

def days : ℕ := 5
def p_sun : ℚ := 1/4
def p_rain : ℚ := 3/4

def probability_k_sunny_days (k : ℕ) : ℚ :=
  (Nat.choose days k) * (p_sun ^ k) * (p_rain ^ (days - k))

theorem desired_weather_probability : 
  probability_k_sunny_days 1 + probability_k_sunny_days 2 = 135/2048 := by
  sorry

end NUMINAMATH_CALUDE_desired_weather_probability_l1398_139871


namespace NUMINAMATH_CALUDE_add_three_tenths_to_57_7_l1398_139843

theorem add_three_tenths_to_57_7 : (57.7 : ℝ) + (3 / 10 : ℝ) = 58 := by
  sorry

end NUMINAMATH_CALUDE_add_three_tenths_to_57_7_l1398_139843


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1398_139859

def vector_a (m : ℝ) : Fin 2 → ℝ := ![m, 1]
def vector_b : Fin 2 → ℝ := ![3, 3]

def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

theorem perpendicular_vectors (m : ℝ) :
  dot_product (λ i => vector_a m i - vector_b i) vector_b = 0 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1398_139859


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l1398_139807

theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 4*x + 1 = 0 ↔ (x - 2)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l1398_139807


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l1398_139812

/-- Proves that adding a specific amount of pure alcohol to a given solution results in the desired alcohol percentage -/
theorem alcohol_solution_proof (initial_volume : ℝ) (initial_percentage : ℝ) (added_alcohol : ℝ) (final_percentage : ℝ) :
  initial_volume = 100 →
  initial_percentage = 0.2 →
  added_alcohol = 14.285714285714286 →
  final_percentage = 0.3 →
  (initial_volume * initial_percentage + added_alcohol) / (initial_volume + added_alcohol) = final_percentage := by
  sorry

#check alcohol_solution_proof

end NUMINAMATH_CALUDE_alcohol_solution_proof_l1398_139812


namespace NUMINAMATH_CALUDE_constant_function_equals_derivative_l1398_139855

theorem constant_function_equals_derivative :
  ∀ (f : ℝ → ℝ), (∀ x, f x = 0) → ∀ x, f x = deriv f x := by sorry

end NUMINAMATH_CALUDE_constant_function_equals_derivative_l1398_139855


namespace NUMINAMATH_CALUDE_triangle_analogous_to_tetrahedron_l1398_139823

/-- Represents geometric objects -/
inductive GeometricObject
  | Quadrilateral
  | Pyramid
  | Triangle
  | Prism
  | Tetrahedron

/-- Defines the concept of analogous objects based on shared properties -/
def are_analogous (a b : GeometricObject) : Prop :=
  ∃ (property : GeometricObject → Prop), property a ∧ property b

/-- Theorem stating that a triangle is analogous to a tetrahedron -/
theorem triangle_analogous_to_tetrahedron :
  are_analogous GeometricObject.Triangle GeometricObject.Tetrahedron :=
sorry

end NUMINAMATH_CALUDE_triangle_analogous_to_tetrahedron_l1398_139823


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1398_139819

theorem quadratic_equal_roots (b : ℝ) :
  (∃ x : ℝ, b * x^2 + 2 * b * x + 4 = 0 ∧
   ∀ y : ℝ, b * y^2 + 2 * b * y + 4 = 0 → y = x) →
  b = 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1398_139819


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1398_139851

def k : ℕ := 2010^2 + 2^2010

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := 2010^2 + 2^2010) : 
  (k^2 + 2^k) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1398_139851


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1398_139885

/-- The polynomial P(x) = x + x^3 + x^9 + x^27 + x^81 + x^243 -/
def P (x : ℝ) : ℝ := x + x^3 + x^9 + x^27 + x^81 + x^243

theorem polynomial_division_remainder :
  (∃ Q₁ : ℝ → ℝ, P = fun x ↦ (x - 1) * Q₁ x + 6) ∧
  (∃ Q₂ : ℝ → ℝ, P = fun x ↦ (x^2 - 1) * Q₂ x + 6*x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1398_139885


namespace NUMINAMATH_CALUDE_field_fully_fenced_l1398_139848

/-- Proves that a square field can be completely fenced given the specified conditions -/
theorem field_fully_fenced (field_area : ℝ) (wire_cost : ℝ) (budget : ℝ) : 
  field_area = 5000 → 
  wire_cost = 30 → 
  budget = 120000 → 
  ∃ (wire_length : ℝ), wire_length = budget / wire_cost ∧ 
    wire_length ≥ 4 * Real.sqrt field_area := by
  sorry

end NUMINAMATH_CALUDE_field_fully_fenced_l1398_139848


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l1398_139840

/-- Calculates the total cost of fencing a rectangular plot. -/
def total_fencing_cost (length : ℝ) (breadth : ℝ) (cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Proves that the total cost of fencing the given rectangular plot is 5300 currency units. -/
theorem fencing_cost_calculation :
  let length : ℝ := 55
  let breadth : ℝ := 45
  let cost_per_meter : ℝ := 26.50
  total_fencing_cost length breadth cost_per_meter = 5300 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_calculation_l1398_139840


namespace NUMINAMATH_CALUDE_quadratic_root_implies_t_value_l1398_139816

theorem quadratic_root_implies_t_value (a t : ℝ) :
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (a + 3 * Complex.I : ℂ) ^ 2 - 4 * (a + 3 * Complex.I : ℂ) + t = 0 →
  t = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_t_value_l1398_139816


namespace NUMINAMATH_CALUDE_simplify_fraction_l1398_139846

theorem simplify_fraction : (150 : ℚ) / 450 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1398_139846


namespace NUMINAMATH_CALUDE_locus_general_case_locus_special_case_l1398_139834

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop := sorry

-- Define a point inside a triangle
def InsideTriangle (S : ℝ × ℝ) (P Q R : ℝ × ℝ) : Prop := sorry

-- Define a segment on a side of a triangle
def SegmentOnSide (A B : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop := sorry

-- Define the area of a triangle
def AreaTriangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define a line segment parallel to another line segment
def ParallelSegment (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the locus of points
def Locus (S : ℝ × ℝ) (P Q R : ℝ × ℝ) (A B C D E F : ℝ × ℝ) (S₀ : ℝ × ℝ) : Prop :=
  InsideTriangle S P Q R ∧
  AreaTriangle S A B + AreaTriangle S C D + AreaTriangle S E F =
  AreaTriangle S₀ A B + AreaTriangle S₀ C D + AreaTriangle S₀ E F

-- Theorem for the general case
theorem locus_general_case 
  (P Q R : ℝ × ℝ) 
  (A B C D E F : ℝ × ℝ) 
  (S₀ : ℝ × ℝ) 
  (h1 : Triangle P Q R)
  (h2 : SegmentOnSide A B P Q)
  (h3 : SegmentOnSide C D Q R)
  (h4 : SegmentOnSide E F R P)
  (h5 : InsideTriangle S₀ P Q R) :
  ∃ D' E' : ℝ × ℝ, 
    ParallelSegment D' E' C D ∧ 
    (∀ S : ℝ × ℝ, Locus S P Q R A B C D E F S₀ ↔ 
      (S = S₀ ∨ ParallelSegment S S₀ D' E')) :=
sorry

-- Theorem for the special case
theorem locus_special_case
  (P Q R : ℝ × ℝ) 
  (A B C D E F : ℝ × ℝ) 
  (S₀ : ℝ × ℝ) 
  (h1 : Triangle P Q R)
  (h2 : SegmentOnSide A B P Q)
  (h3 : SegmentOnSide C D Q R)
  (h4 : SegmentOnSide E F R P)
  (h5 : InsideTriangle S₀ P Q R)
  (h6 : ∃ k : ℝ, k > 0 ∧ 
    ‖A - B‖ / ‖P - Q‖ = k ∧ 
    ‖C - D‖ / ‖Q - R‖ = k ∧ 
    ‖E - F‖ / ‖R - P‖ = k) :
  ∀ S : ℝ × ℝ, InsideTriangle S P Q R → Locus S P Q R A B C D E F S₀ :=
sorry

end NUMINAMATH_CALUDE_locus_general_case_locus_special_case_l1398_139834


namespace NUMINAMATH_CALUDE_g_composition_of_3_l1398_139880

def g (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_composition_of_3 : g (g (g (g 3))) = 24 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_3_l1398_139880
