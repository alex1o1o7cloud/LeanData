import Mathlib

namespace NUMINAMATH_CALUDE_milk_mixture_theorem_l3865_386535

/-- Given two types of milk with different butterfat percentages, prove that mixing them in specific quantities results in a desired butterfat percentage. -/
theorem milk_mixture_theorem (x : ℝ) :
  -- Define the butterfat percentages
  let high_fat_percent : ℝ := 0.45
  let low_fat_percent : ℝ := 0.10
  let target_percent : ℝ := 0.20

  -- Define the quantities
  let low_fat_quantity : ℝ := 20
  
  -- Condition: The mixture's butterfat content equals the target percentage
  high_fat_percent * x + low_fat_percent * low_fat_quantity = target_percent * (x + low_fat_quantity) →
  -- Conclusion: The quantity of high-fat milk needed is 8 gallons
  x = 8 := by
  sorry

#check milk_mixture_theorem

end NUMINAMATH_CALUDE_milk_mixture_theorem_l3865_386535


namespace NUMINAMATH_CALUDE_pet_store_cages_l3865_386523

/-- Calculates the number of cages needed for a given number of animals and cage capacity -/
def cages_needed (animals : ℕ) (capacity : ℕ) : ℕ :=
  (animals + capacity - 1) / capacity

theorem pet_store_cages : 
  let initial_puppies : ℕ := 13
  let initial_kittens : ℕ := 10
  let initial_birds : ℕ := 15
  let sold_puppies : ℕ := 7
  let sold_kittens : ℕ := 4
  let sold_birds : ℕ := 5
  let puppy_capacity : ℕ := 2
  let kitten_capacity : ℕ := 3
  let bird_capacity : ℕ := 4
  let remaining_puppies := initial_puppies - sold_puppies
  let remaining_kittens := initial_kittens - sold_kittens
  let remaining_birds := initial_birds - sold_birds
  let total_cages := cages_needed remaining_puppies puppy_capacity + 
                     cages_needed remaining_kittens kitten_capacity + 
                     cages_needed remaining_birds bird_capacity
  total_cages = 8 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3865_386523


namespace NUMINAMATH_CALUDE_circular_field_area_l3865_386594

-- Define the constants
def fencing_cost_per_metre : ℝ := 4
def total_fencing_cost : ℝ := 5941.9251828093165

-- Define the theorem
theorem circular_field_area :
  ∃ (area : ℝ),
    (area ≥ 17.55 ∧ area ≤ 17.57) ∧
    (∃ (circumference radius : ℝ),
      circumference = total_fencing_cost / fencing_cost_per_metre ∧
      radius = circumference / (2 * Real.pi) ∧
      area = (Real.pi * radius ^ 2) / 10000) :=
by sorry

end NUMINAMATH_CALUDE_circular_field_area_l3865_386594


namespace NUMINAMATH_CALUDE_student_subject_assignment_l3865_386543

/-- The number of ways to assign students to subjects. -/
def num_assignments (n : ℕ) (k : ℕ) : ℕ := k^n

/-- The number of students. -/
def num_students : ℕ := 4

/-- The number of subjects. -/
def num_subjects : ℕ := 3

theorem student_subject_assignment :
  num_assignments num_students num_subjects = 81 := by
  sorry

end NUMINAMATH_CALUDE_student_subject_assignment_l3865_386543


namespace NUMINAMATH_CALUDE_simplified_expression_l3865_386589

theorem simplified_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 3*(a + b)) :
  a/b + b/a - 3/(a*b) = 1 := by
sorry

end NUMINAMATH_CALUDE_simplified_expression_l3865_386589


namespace NUMINAMATH_CALUDE_fraction_simplification_l3865_386571

theorem fraction_simplification :
  (12 : ℚ) / 11 * 15 / 28 * 44 / 45 = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3865_386571


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3865_386530

theorem solution_set_inequality (x : ℝ) :
  x^2 - |x| - 2 ≤ 0 ↔ x ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3865_386530


namespace NUMINAMATH_CALUDE_triangle_nth_root_l3865_386509

theorem triangle_nth_root (a b c : ℝ) (n : ℕ) (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) (h_n : n ≥ 2) :
  (a^(1/n) : ℝ) + (b^(1/n) : ℝ) > (c^(1/n) : ℝ) ∧
  (b^(1/n) : ℝ) + (c^(1/n) : ℝ) > (a^(1/n) : ℝ) ∧
  (a^(1/n) : ℝ) + (c^(1/n) : ℝ) > (b^(1/n) : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_triangle_nth_root_l3865_386509


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l3865_386548

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l3865_386548


namespace NUMINAMATH_CALUDE_no_valid_coloring_l3865_386558

/-- A coloring function that assigns one of three colors to each natural number -/
def Coloring := ℕ → Fin 3

/-- Predicate checking if a coloring satisfies the required property -/
def ValidColoring (c : Coloring) : Prop :=
  (∃ n : ℕ, c n = 0) ∧
  (∃ n : ℕ, c n = 1) ∧
  (∃ n : ℕ, c n = 2) ∧
  (∀ x y : ℕ, c x ≠ c y → c (x + y) ≠ c x ∧ c (x + y) ≠ c y)

theorem no_valid_coloring : ¬∃ c : Coloring, ValidColoring c := by
  sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l3865_386558


namespace NUMINAMATH_CALUDE_johns_apartment_rental_l3865_386534

/-- John's apartment rental problem -/
theorem johns_apartment_rental 
  (num_subletters : ℕ) 
  (subletter_payment : ℕ) 
  (annual_profit : ℕ) 
  (monthly_rent : ℕ) : 
  num_subletters = 3 → 
  subletter_payment = 400 → 
  annual_profit = 3600 → 
  monthly_rent = 900 → 
  (num_subletters * subletter_payment - monthly_rent) * 12 = annual_profit :=
by sorry

end NUMINAMATH_CALUDE_johns_apartment_rental_l3865_386534


namespace NUMINAMATH_CALUDE_triangle_angle_bound_triangle_side_ratio_l3865_386529

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a + t.c = 2 * t.b

theorem triangle_angle_bound (t : Triangle) (h : TriangleConditions t) : t.B ≤ Real.pi / 3 := by
  sorry

theorem triangle_side_ratio (t : Triangle) (h : TriangleConditions t) (h2 : t.C = 2 * t.A) :
  ∃ (k : ℝ), t.a = 4 * k ∧ t.b = 5 * k ∧ t.c = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_bound_triangle_side_ratio_l3865_386529


namespace NUMINAMATH_CALUDE_quadratic_three_times_point_range_l3865_386593

/-- A quadratic function y = -x^2 - x + c has at least one "three times point" (y = 3x) 
    in the range -3 < x < 1 if and only if -4 ≤ c < 5 -/
theorem quadratic_three_times_point_range (c : ℝ) : 
  (∃ x : ℝ, -3 < x ∧ x < 1 ∧ 3 * x = -x^2 - x + c) ↔ -4 ≤ c ∧ c < 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_three_times_point_range_l3865_386593


namespace NUMINAMATH_CALUDE_a_equals_one_l3865_386592

def star (x y : ℝ) : ℝ := x + y - x * y

theorem a_equals_one (a : ℝ) (h : a = star 1 (star 0 1)) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_one_l3865_386592


namespace NUMINAMATH_CALUDE_same_color_plate_probability_l3865_386519

/-- The probability of selecting three plates of the same color from 7 green and 5 yellow plates. -/
theorem same_color_plate_probability :
  let total_plates : ℕ := 7 + 5
  let green_plates : ℕ := 7
  let yellow_plates : ℕ := 5
  let total_combinations : ℕ := Nat.choose total_plates 3
  let green_combinations : ℕ := Nat.choose green_plates 3
  let yellow_combinations : ℕ := Nat.choose yellow_plates 3
  let same_color_combinations : ℕ := green_combinations + yellow_combinations
  (same_color_combinations : ℚ) / total_combinations = 9 / 44 := by
sorry


end NUMINAMATH_CALUDE_same_color_plate_probability_l3865_386519


namespace NUMINAMATH_CALUDE_circle_properties_l3865_386585

/-- A circle with diameter endpoints (2, -3) and (8, 9) has center (5, 3) and radius 3√5 -/
theorem circle_properties :
  let A : ℝ × ℝ := (2, -3)
  let B : ℝ × ℝ := (8, 9)
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let r : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  C = (5, 3) ∧ r = 3 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_circle_properties_l3865_386585


namespace NUMINAMATH_CALUDE_savings_ratio_l3865_386533

def january_amount : ℕ := 19
def march_amount : ℕ := 8
def total_amount : ℕ := 46

def february_amount : ℕ := total_amount - january_amount - march_amount

theorem savings_ratio : 
  (january_amount : ℚ) / (february_amount : ℚ) = 1 := by sorry

end NUMINAMATH_CALUDE_savings_ratio_l3865_386533


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l3865_386545

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  /-- The angle between faces ABC and BCD in radians -/
  angle : ℝ
  /-- The area of face ABC -/
  area_ABC : ℝ
  /-- The area of face BCD -/
  area_BCD : ℝ
  /-- The length of edge BC -/
  length_BC : ℝ

/-- Calculates the volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 320 -/
theorem volume_of_specific_tetrahedron :
  let t : Tetrahedron := {
    angle := 30 * π / 180,  -- 30 degrees in radians
    area_ABC := 120,
    area_BCD := 80,
    length_BC := 10
  }
  volume t = 320 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l3865_386545


namespace NUMINAMATH_CALUDE_triangle_side_length_l3865_386537

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if ∠B = 60°, ∠C = 75°, and a = 4, then b = 2√6. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  B = 60 * π / 180 →
  C = 75 * π / 180 →
  a = 4 →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  b = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3865_386537


namespace NUMINAMATH_CALUDE_compute_expression_l3865_386528

theorem compute_expression (x : ℝ) (h : x = 9) : 
  (x^9 - 27*x^6 + 729) / (x^6 - 27) = 730 + 1/26 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3865_386528


namespace NUMINAMATH_CALUDE_martha_cakes_l3865_386546

/-- The number of whole cakes Martha needs to buy -/
def cakes_needed (num_children : ℕ) (cakes_per_child : ℕ) (special_children : ℕ) 
  (parts_per_cake : ℕ) : ℕ :=
  let total_small_cakes := num_children * cakes_per_child
  let special_whole_cakes := (special_children * cakes_per_child + parts_per_cake - 1) / parts_per_cake
  let remaining_small_cakes := total_small_cakes - special_whole_cakes * parts_per_cake
  special_whole_cakes + (remaining_small_cakes + parts_per_cake - 1) / parts_per_cake

/-- The theorem stating the number of cakes Martha needs to buy -/
theorem martha_cakes : cakes_needed 5 25 2 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_martha_cakes_l3865_386546


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_144_l3865_386584

theorem percentage_of_360_equals_144 : ∃ (p : ℚ), p * 360 = 144 ∧ p = 40 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_144_l3865_386584


namespace NUMINAMATH_CALUDE_cashew_price_satisfies_conditions_l3865_386508

/-- The price per pound of cashews that satisfies the mixture conditions -/
def cashew_price : ℝ := 6.75

/-- The total weight of the mixture in pounds -/
def total_mixture : ℝ := 50

/-- The selling price of the mixture per pound -/
def mixture_price : ℝ := 5.70

/-- The weight of cashews used in the mixture in pounds -/
def cashew_weight : ℝ := 20

/-- The price of Brazil nuts per pound -/
def brazil_nut_price : ℝ := 5.00

/-- Theorem stating that the calculated cashew price satisfies the mixture conditions -/
theorem cashew_price_satisfies_conditions : 
  cashew_weight * cashew_price + (total_mixture - cashew_weight) * brazil_nut_price = 
  total_mixture * mixture_price :=
sorry

end NUMINAMATH_CALUDE_cashew_price_satisfies_conditions_l3865_386508


namespace NUMINAMATH_CALUDE_system_solution_exists_l3865_386518

theorem system_solution_exists (m : ℝ) : 
  m ≠ 3 → ∃ (x y : ℝ), y = m * x + 6 ∧ y = (2 * m - 3) * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_l3865_386518


namespace NUMINAMATH_CALUDE_savings_of_eight_hundred_bills_l3865_386564

/-- The total savings amount when exchanged into a given number of $100 bills -/
def savings_amount (num_bills : ℕ) : ℕ := 100 * num_bills

/-- Theorem: If a person has 8 $100 bills after exchanging all their savings, 
    their total savings amount to $800 -/
theorem savings_of_eight_hundred_bills : 
  savings_amount 8 = 800 := by sorry

end NUMINAMATH_CALUDE_savings_of_eight_hundred_bills_l3865_386564


namespace NUMINAMATH_CALUDE_power_ratio_approximation_l3865_386514

theorem power_ratio_approximation :
  let ratio := (10^2001 + 10^2003) / (10^2002 + 10^2002)
  ratio = 101 / 20 ∧ 
  ∀ n : ℤ, |ratio - 5| ≤ |ratio - n| := by
  sorry

end NUMINAMATH_CALUDE_power_ratio_approximation_l3865_386514


namespace NUMINAMATH_CALUDE_leahs_coins_value_l3865_386588

theorem leahs_coins_value (p n : ℕ) : 
  p + n = 15 ∧ 
  p = 2 * (n + 3) → 
  5 * n + p = 27 :=
by sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l3865_386588


namespace NUMINAMATH_CALUDE_parabola_max_value_l3865_386549

theorem parabola_max_value :
  ∃ (max : ℝ), max = 4 ∧ ∀ (x : ℝ), -x^2 + 2*x + 3 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_value_l3865_386549


namespace NUMINAMATH_CALUDE_pet_insurance_coverage_percentage_l3865_386532

theorem pet_insurance_coverage_percentage
  (insurance_duration : ℕ)
  (insurance_monthly_cost : ℚ)
  (procedure_cost : ℚ)
  (amount_saved : ℚ)
  (h1 : insurance_duration = 24)
  (h2 : insurance_monthly_cost = 20)
  (h3 : procedure_cost = 5000)
  (h4 : amount_saved = 3520)
  : (1 - (amount_saved / procedure_cost)) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_pet_insurance_coverage_percentage_l3865_386532


namespace NUMINAMATH_CALUDE_initial_investment_rate_l3865_386591

-- Define the initial investment
def initial_investment : ℝ := 1400

-- Define the additional investment
def additional_investment : ℝ := 700

-- Define the interest rate of the additional investment
def additional_rate : ℝ := 0.08

-- Define the total investment
def total_investment : ℝ := initial_investment + additional_investment

-- Define the desired total annual income rate
def total_income_rate : ℝ := 0.06

-- Define the function that calculates the total annual income
def total_annual_income (r : ℝ) : ℝ := 
  initial_investment * r + additional_investment * additional_rate

-- Theorem statement
theorem initial_investment_rate : 
  ∃ r : ℝ, total_annual_income r = total_income_rate * total_investment ∧ r = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_initial_investment_rate_l3865_386591


namespace NUMINAMATH_CALUDE_square_root_problem_l3865_386586

theorem square_root_problem (x a : ℝ) : 
  ((2 * a + 1) ^ 2 = x ∧ (4 - a) ^ 2 = x) → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l3865_386586


namespace NUMINAMATH_CALUDE_max_sides_cube_plane_intersection_l3865_386595

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- A plane is a flat, two-dimensional surface -/
structure Plane where
  -- We don't need to define the specifics of a plane for this problem

/-- A polygon is a plane figure with straight sides -/
structure Polygon where
  sides : ℕ

/-- The cross-section formed when a plane intersects a cube -/
def crossSection (c : Cube) (p : Plane) : Polygon :=
  sorry -- Implementation details not needed for the statement

/-- The maximum number of sides a polygon can have when it's formed by a plane intersecting a cube is 6 -/
theorem max_sides_cube_plane_intersection (c : Cube) (p : Plane) :
  (crossSection c p).sides ≤ 6 ∧ ∃ (c : Cube) (p : Plane), (crossSection c p).sides = 6 :=
sorry

end NUMINAMATH_CALUDE_max_sides_cube_plane_intersection_l3865_386595


namespace NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_simplify_expression3_l3865_386570

-- Expression 1
theorem simplify_expression1 (a b x : ℝ) (h : b ≠ 0) :
  (12 * a^3 * x^4 + 2 * a^2 * x^5) / (18 * a * b^2 * x + 3 * b^2 * x^2) = 
  (2 * a^2 * x^3) / (3 * b^2) :=
sorry

-- Expression 2
theorem simplify_expression2 (x : ℝ) (h : x ≠ -2) :
  (4 - 2*x + x^2) / (x + 2) - x - 2 = -6*x / (x + 2) :=
sorry

-- Expression 3
theorem simplify_expression3 (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  1 / ((a-b)*(a-c)) + 1 / ((b-a)*(b-c)) + 1 / ((c-a)*(c-b)) = 0 :=
sorry

end NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_simplify_expression3_l3865_386570


namespace NUMINAMATH_CALUDE_min_workers_for_painting_job_l3865_386581

/-- Represents the painting job scenario -/
structure PaintingJob where
  totalDays : ℕ
  workedDays : ℕ
  initialWorkers : ℕ
  completedFraction : ℚ
  
/-- Calculates the minimum number of workers needed to complete the job on time -/
def minWorkersNeeded (job : PaintingJob) : ℕ :=
  sorry

/-- The theorem stating the minimum number of workers needed for the specific scenario -/
theorem min_workers_for_painting_job :
  let job := PaintingJob.mk 40 8 10 (2/5)
  minWorkersNeeded job = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_workers_for_painting_job_l3865_386581


namespace NUMINAMATH_CALUDE_function_property_l3865_386563

def f (a : ℝ) (x : ℝ) : ℝ := sorry

theorem function_property (a : ℝ) :
  (∀ x, f a (x + 3) = 3 * f a x) →
  (∀ x ∈ Set.Ioo 0 3, f a x = Real.log x - a * x) →
  a > 1/3 →
  (∃ x ∈ Set.Ioo (-6) (-3), f a x = -1/9 ∧ ∀ y ∈ Set.Ioo (-6) (-3), f a y ≤ f a x) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_function_property_l3865_386563


namespace NUMINAMATH_CALUDE_function_domain_implies_a_range_l3865_386517

/-- If the function f(x) = √(2^(x^2 + 2ax - a) - 1) is defined for all real x, then -1 ≤ a ≤ 0 -/
theorem function_domain_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (2^(x^2 + 2*a*x - a) - 1)) → 
  -1 ≤ a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_function_domain_implies_a_range_l3865_386517


namespace NUMINAMATH_CALUDE_angle_expression_value_l3865_386555

theorem angle_expression_value (α : Real) 
  (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.sin α = Real.sqrt 15 / 4) :  -- sin α = √15/4
  Real.sin (α + π/4) / (Real.sin (2*α) + Real.cos (2*α) + 1) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_value_l3865_386555


namespace NUMINAMATH_CALUDE_tree_planting_theorem_l3865_386505

/-- The number of trees planted by Class 2-5 -/
def trees_2_5 : ℕ := 142

/-- The difference in trees planted between Class 2-5 and Class 2-3 -/
def difference : ℕ := 18

/-- The number of trees planted by Class 2-3 -/
def trees_2_3 : ℕ := trees_2_5 - difference

/-- The total number of trees planted by both classes -/
def total_trees : ℕ := trees_2_5 + trees_2_3

theorem tree_planting_theorem :
  trees_2_3 = 124 ∧ total_trees = 266 :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_theorem_l3865_386505


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3865_386547

theorem complex_equation_solution (a : ℝ) :
  (Complex.mk 2 a) * (Complex.mk a (-2)) = Complex.I * (-4) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3865_386547


namespace NUMINAMATH_CALUDE_shift_proof_l3865_386582

def original_function (x : ℝ) : ℝ := -3 * x + 2

def vertical_shift : ℝ := 3

def shifted_function (x : ℝ) : ℝ := original_function x + vertical_shift

theorem shift_proof : 
  ∀ x : ℝ, shifted_function x = -3 * x + 5 := by
sorry

end NUMINAMATH_CALUDE_shift_proof_l3865_386582


namespace NUMINAMATH_CALUDE_unicorn_tether_problem_l3865_386501

theorem unicorn_tether_problem (rope_length : ℝ) (tower_radius : ℝ) (unicorn_height : ℝ) 
  (rope_end_distance : ℝ) (p q r : ℕ) (h_rope_length : rope_length = 25)
  (h_tower_radius : tower_radius = 10) (h_unicorn_height : unicorn_height = 5)
  (h_rope_end_distance : rope_end_distance = 5) (h_r_prime : Nat.Prime r)
  (h_rope_tower_length : (p - Real.sqrt q) / r = 
    rope_length - Real.sqrt ((rope_end_distance + tower_radius)^2 + unicorn_height^2)) :
  p + q + r = 1128 := by
  sorry

end NUMINAMATH_CALUDE_unicorn_tether_problem_l3865_386501


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l3865_386573

theorem jacket_price_reduction (P : ℝ) (x : ℝ) : 
  P > 0 → 
  0 ≤ x → 
  x ≤ 100 → 
  P * (1 - x / 100) * (1 - 20 / 100) * (1 + 2 / 3) = P → 
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l3865_386573


namespace NUMINAMATH_CALUDE_some_number_value_l3865_386524

theorem some_number_value (x : ℚ) :
  (3 / 5 : ℚ) * ((2 / 3 + 3 / 8) / x) - 1 / 16 = 0.24999999999999994 →
  x = 48 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3865_386524


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l3865_386583

-- Define the center of the circle
def center : ℝ × ℝ := (2, 1)

-- Define the tangent line
def tangent_line (y : ℝ) : Prop := y + 1 = 0

-- Define the distance from a point to the tangent line
def distance_to_line (x y : ℝ) : ℝ := |y + 1|

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Theorem statement
theorem circle_tangent_to_line :
  ∀ x y : ℝ, circle_equation x y →
  distance_to_line x y = (4 : ℝ).sqrt ∧
  ∃ p : ℝ × ℝ, p.1 = x ∧ p.2 = y ∧ tangent_line p.2 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l3865_386583


namespace NUMINAMATH_CALUDE_f_g_f_3_equals_108_l3865_386542

def f (x : ℝ) : ℝ := 2 * x + 4

def g (x : ℝ) : ℝ := 5 * x + 2

theorem f_g_f_3_equals_108 : f (g (f 3)) = 108 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_3_equals_108_l3865_386542


namespace NUMINAMATH_CALUDE_ellipse_equation_l3865_386580

/-- Given a hyperbola and an ellipse with specific properties, prove the equation of the ellipse -/
theorem ellipse_equation (h : ℝ → ℝ → Prop) (e : ℝ → ℝ → Prop) :
  (∀ x y, h x y ↔ y^2/12 - x^2/4 = 1) →  -- Definition of the hyperbola
  (∃ a b, ∀ x y, e x y ↔ x^2/a + y^2/b = 1) →  -- General form of the ellipse
  (∃ v₁ v₂, v₁ ≠ v₂ ∧ h 0 v₁ ∧ h 0 (-v₁) ∧ 
    ∀ x y, e x y → (x - 0)^2 + (y - v₁)^2 + (x - 0)^2 + (y + v₁)^2 = 16) →  -- Vertices of hyperbola as foci of ellipse
  (∀ x y, e x y ↔ x^2/4 + y^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3865_386580


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3865_386598

theorem unique_solution_condition (a : ℝ) :
  (∃! x : ℝ, x ≠ 2 ∧ x ≠ -1 ∧ |x + 2| = |x| + a) ↔ a ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3865_386598


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l3865_386590

theorem tangent_line_intersection (a : ℝ) : 
  (∃ x : ℝ, x + Real.log x = a * x^2 + (a + 2) * x + 1 ∧ 
   2 * x - 1 = a * x^2 + (a + 2) * x + 1) ↔ 
  a = 8 := by sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l3865_386590


namespace NUMINAMATH_CALUDE_combined_girls_average_is_85_l3865_386522

/-- Represents the average scores and student counts for two high schools -/
structure SchoolData where
  adams_boys_avg : ℝ
  adams_girls_avg : ℝ
  adams_combined_avg : ℝ
  baker_boys_avg : ℝ
  baker_girls_avg : ℝ
  baker_combined_avg : ℝ
  combined_boys_avg : ℝ
  adams_boys_count : ℝ
  adams_girls_count : ℝ
  baker_boys_count : ℝ
  baker_girls_count : ℝ

/-- Theorem stating that the combined girls' average score for both schools is 85 -/
theorem combined_girls_average_is_85 (data : SchoolData)
  (h1 : data.adams_boys_avg = 72)
  (h2 : data.adams_girls_avg = 78)
  (h3 : data.adams_combined_avg = 75)
  (h4 : data.baker_boys_avg = 84)
  (h5 : data.baker_girls_avg = 91)
  (h6 : data.baker_combined_avg = 85)
  (h7 : data.combined_boys_avg = 80)
  (h8 : data.adams_boys_count = data.adams_girls_count)
  (h9 : data.baker_boys_count = 6 * data.baker_girls_count / 7)
  (h10 : data.adams_boys_count = data.baker_boys_count) :
  (data.adams_girls_avg * data.adams_girls_count + data.baker_girls_avg * data.baker_girls_count) /
  (data.adams_girls_count + data.baker_girls_count) = 85 := by
  sorry


end NUMINAMATH_CALUDE_combined_girls_average_is_85_l3865_386522


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3865_386516

theorem quadratic_inequality_solution (z : ℝ) :
  z^2 - 40*z + 350 ≤ 6 ↔ 20 - 2*Real.sqrt 14 ≤ z ∧ z ≤ 20 + 2*Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3865_386516


namespace NUMINAMATH_CALUDE_quadratic_properties_l3865_386502

def f (x : ℝ) := x^2 - 2*x - 1

theorem quadratic_properties :
  (∃ (x y : ℝ), (x, y) = (1, -2) ∧ ∀ t, f t ≥ f x) ∧
  (∃ (x₁ x₂ : ℝ), x₁ = 1 + Real.sqrt 2 ∧ 
                  x₂ = 1 - Real.sqrt 2 ∧ 
                  f x₁ = 0 ∧ 
                  f x₂ = 0 ∧
                  ∀ x, f x = 0 → x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3865_386502


namespace NUMINAMATH_CALUDE_bedroom_curtain_width_l3865_386565

theorem bedroom_curtain_width :
  let initial_width : ℝ := 16
  let initial_height : ℝ := 12
  let living_room_width : ℝ := 4
  let living_room_height : ℝ := 6
  let bedroom_height : ℝ := 4
  let remaining_area : ℝ := 160
  let total_area := initial_width * initial_height
  let living_room_area := living_room_width * living_room_height
  let bedroom_width := (total_area - living_room_area - remaining_area) / bedroom_height
  bedroom_width = 2 := by sorry

end NUMINAMATH_CALUDE_bedroom_curtain_width_l3865_386565


namespace NUMINAMATH_CALUDE_postage_calculation_l3865_386544

/-- Calculates the postage cost for a letter based on its weight and given rates. -/
def calculatePostage (weight : ℚ) (baseRate : ℚ) (additionalRate : ℚ) : ℚ :=
  let additionalWeight := max (weight - 1) 0
  let additionalCharges := ⌈additionalWeight⌉
  baseRate + additionalCharges * additionalRate

/-- Theorem stating that the postage for a 4.5-ounce letter is 1.18 dollars 
    given the specified rates. -/
theorem postage_calculation :
  let weight : ℚ := 4.5
  let baseRate : ℚ := 0.30
  let additionalRate : ℚ := 0.22
  calculatePostage weight baseRate additionalRate = 1.18 := by
  sorry


end NUMINAMATH_CALUDE_postage_calculation_l3865_386544


namespace NUMINAMATH_CALUDE_gems_calculation_l3865_386560

/-- Calculates the total number of gems received given an initial spend, gem rate, and bonus percentage. -/
def total_gems (spend : ℕ) (rate : ℕ) (bonus_percent : ℕ) : ℕ :=
  let initial_gems := spend * rate
  let bonus_gems := initial_gems * bonus_percent / 100
  initial_gems + bonus_gems

/-- Proves that given the specified conditions, the total number of gems received is 30000. -/
theorem gems_calculation :
  let spend := 250
  let rate := 100
  let bonus_percent := 20
  total_gems spend rate bonus_percent = 30000 := by
  sorry

end NUMINAMATH_CALUDE_gems_calculation_l3865_386560


namespace NUMINAMATH_CALUDE_A_3_2_equals_19_l3865_386503

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_19 : A 3 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_A_3_2_equals_19_l3865_386503


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3865_386551

/-- Given two planar vectors m and n, where m is parallel to n, 
    prove that the magnitude of n is 2√5. -/
theorem parallel_vectors_magnitude (m n : ℝ × ℝ) : 
  m = (-1, 2) → 
  n.1 = 2 → 
  (∃ k : ℝ, n = k • m) → 
  ‖n‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3865_386551


namespace NUMINAMATH_CALUDE_school_sample_size_l3865_386539

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  stratum_size : ℕ
  stratum_sample : ℕ
  total_sample : ℕ

/-- Checks if the given stratified sample is proportional -/
def is_proportional_sample (s : StratifiedSample) : Prop :=
  s.stratum_sample * s.total_population = s.total_sample * s.stratum_size

/-- Theorem stating that for the given population and sample sizes, 
    the total sample size is 45 -/
theorem school_sample_size :
  ∀ (s : StratifiedSample), 
    s.total_population = 1500 →
    s.stratum_size = 400 →
    s.stratum_sample = 12 →
    is_proportional_sample s →
    s.total_sample = 45 := by
  sorry

end NUMINAMATH_CALUDE_school_sample_size_l3865_386539


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3865_386531

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 20 + (3 * x) + 15 + (3 * x + 6)) / 5 = 30 → x = 99 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3865_386531


namespace NUMINAMATH_CALUDE_matrix_operation_result_l3865_386527

theorem matrix_operation_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 4; 0, -3]
  let C : Matrix (Fin 2) (Fin 2) ℤ := !![6, 0; -1, 8]
  A - B + C = !![12, -7; 1, 16] := by
sorry

end NUMINAMATH_CALUDE_matrix_operation_result_l3865_386527


namespace NUMINAMATH_CALUDE_dinosaur_model_price_reduction_l3865_386574

/-- The percentage reduction in dinosaur model prices for a school purchase --/
theorem dinosaur_model_price_reduction :
  -- Original price per model
  ∀ (original_price : ℕ),
  -- Number of models for kindergarten
  ∀ (k : ℕ),
  -- Number of models for elementary
  ∀ (e : ℕ),
  -- Total number of models
  ∀ (total : ℕ),
  -- Total amount paid
  ∀ (total_paid : ℕ),
  -- Conditions
  original_price = 100 →
  k = 2 →
  e = 2 * k →
  total = k + e →
  total > 5 →
  total_paid = 570 →
  -- Conclusion
  (1 - total_paid / (total * original_price : ℚ)) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_dinosaur_model_price_reduction_l3865_386574


namespace NUMINAMATH_CALUDE_triangle_ratio_l3865_386507

theorem triangle_ratio (a b c : ℝ) (A : ℝ) (S : ℝ) : 
  A = π/3 →  -- 60° in radians
  b = 1 → 
  S = Real.sqrt 3 → 
  S = (1/2) * b * c * Real.sin A → 
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A → 
  a / Real.sin A = 2 * Real.sqrt 39 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3865_386507


namespace NUMINAMATH_CALUDE_janet_stickers_l3865_386550

theorem janet_stickers (initial_stickers received_stickers : ℕ) : 
  initial_stickers = 3 → received_stickers = 53 → initial_stickers + received_stickers = 56 := by
  sorry

end NUMINAMATH_CALUDE_janet_stickers_l3865_386550


namespace NUMINAMATH_CALUDE_binary_38_correct_l3865_386575

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 38 -/
def binary_38 : List Bool := [false, true, true, false, false, true]

/-- Theorem stating that the binary representation of 38 is correct -/
theorem binary_38_correct : binary_to_decimal binary_38 = 38 := by
  sorry

#eval binary_to_decimal binary_38

end NUMINAMATH_CALUDE_binary_38_correct_l3865_386575


namespace NUMINAMATH_CALUDE_unknown_number_solution_l3865_386512

theorem unknown_number_solution :
  ∃! y : ℝ, (0.47 * 1442 - 0.36 * y) + 65 = 5 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_solution_l3865_386512


namespace NUMINAMATH_CALUDE_min_sum_product_l3865_386578

theorem min_sum_product (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 9/n = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 1 → m + n ≤ a + b) →
  m * n = 48 := by
sorry

end NUMINAMATH_CALUDE_min_sum_product_l3865_386578


namespace NUMINAMATH_CALUDE_initial_subscribers_count_l3865_386596

/-- Represents the monthly income of a streamer based on their number of subscribers -/
def streamer_income (initial_subscribers : ℕ) (gift_subscribers : ℕ) (income_per_subscriber : ℕ) : ℕ :=
  (initial_subscribers + gift_subscribers) * income_per_subscriber

/-- Proves that the initial number of subscribers is 150 given the problem conditions -/
theorem initial_subscribers_count :
  ∃ (x : ℕ), streamer_income x 50 9 = 1800 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_initial_subscribers_count_l3865_386596


namespace NUMINAMATH_CALUDE_intersecting_lines_sum_l3865_386513

/-- Two lines intersect at a point -/
structure IntersectingLines where
  m : ℝ
  b : ℝ
  intersect_x : ℝ
  intersect_y : ℝ
  eq1 : intersect_y = m * intersect_x + 2
  eq2 : intersect_y = -2 * intersect_x + b

/-- Theorem: For two lines y = mx + 2 and y = -2x + b intersecting at (4, 12), b + m = 22.5 -/
theorem intersecting_lines_sum (lines : IntersectingLines)
    (h1 : lines.intersect_x = 4)
    (h2 : lines.intersect_y = 12) :
    lines.b + lines.m = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_sum_l3865_386513


namespace NUMINAMATH_CALUDE_inequality_proof_l3865_386579

theorem inequality_proof (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > a*x ∧ a*x > a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3865_386579


namespace NUMINAMATH_CALUDE_average_age_is_35_l3865_386554

/-- The average age of Omi, Kimiko, and Arlette is 35 years old. -/
theorem average_age_is_35 (kimiko_age omi_age arlette_age : ℕ) : 
  kimiko_age = 28 →
  omi_age = 2 * kimiko_age →
  arlette_age = 3 * kimiko_age / 4 →
  (omi_age + kimiko_age + arlette_age) / 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_age_is_35_l3865_386554


namespace NUMINAMATH_CALUDE_range_x_when_m_is_one_range_m_for_not_p_sufficient_not_necessary_l3865_386521

-- Define propositions p and q
def p (x m : ℝ) : Prop := |2*x - m| ≥ 1
def q (x : ℝ) : Prop := (1 - 3*x) / (x + 2) > 0

-- Theorem for part (I)
theorem range_x_when_m_is_one :
  ∃ a b : ℝ, a = -2 ∧ b = 0 ∧
  ∀ x : ℝ, a < x ∧ x ≤ b ↔ p x 1 ∧ q x :=
sorry

-- Theorem for part (II)
theorem range_m_for_not_p_sufficient_not_necessary :
  ∃ a b : ℝ, a = -3 ∧ b = -1/3 ∧
  ∀ m : ℝ, a ≤ m ∧ m ≤ b ↔
    (∀ x : ℝ, ¬(p x m) → q x) ∧
    ¬(∀ x : ℝ, q x → ¬(p x m)) :=
sorry

end NUMINAMATH_CALUDE_range_x_when_m_is_one_range_m_for_not_p_sufficient_not_necessary_l3865_386521


namespace NUMINAMATH_CALUDE_fraction_equality_l3865_386525

theorem fraction_equality (P Q : ℤ) :
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ 4 →
    (P / (x + 3) + Q / (x^2 - 10*x + 16) = (x^2 - 6*x + 18) / (x^3 - 7*x^2 + 14*x - 48))) →
  (Q : ℚ) / P = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3865_386525


namespace NUMINAMATH_CALUDE_range_of_a_l3865_386511

def p (a : ℝ) : Prop := 2 * a + 1 > 5
def q (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 3

theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (-1 ≤ a ∧ a ≤ 2) ∨ a > 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3865_386511


namespace NUMINAMATH_CALUDE_u_value_when_m_is_3_l3865_386541

-- Define the functions u and t
def t (m : ℕ) : ℕ := 3^m + m
def u (m : ℕ) : ℕ := 4^(t m) - 3*(t m)

-- State the theorem
theorem u_value_when_m_is_3 : u 3 = 4^30 - 90 := by
  sorry

end NUMINAMATH_CALUDE_u_value_when_m_is_3_l3865_386541


namespace NUMINAMATH_CALUDE_contact_lenses_sales_l3865_386526

/-- Proves that the total number of pairs of contact lenses sold is 11 given the problem conditions --/
theorem contact_lenses_sales (soft_price hard_price : ℕ) (soft_hard_diff total_sales : ℕ) :
  soft_price = 150 →
  hard_price = 85 →
  soft_hard_diff = 5 →
  total_sales = 1455 →
  ∃ (soft hard : ℕ),
    soft = hard + soft_hard_diff ∧
    soft_price * soft + hard_price * hard = total_sales ∧
    soft + hard = 11 := by
  sorry

end NUMINAMATH_CALUDE_contact_lenses_sales_l3865_386526


namespace NUMINAMATH_CALUDE_equation_solutions_l3865_386538

theorem equation_solutions : 
  ∀ x : ℝ, x^4 + (4 - x)^4 = 272 ↔ x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3865_386538


namespace NUMINAMATH_CALUDE_count_prime_base_n_l3865_386506

/-- Represents the number 10001 in base n -/
def base_n (n : ℕ) : ℕ := n^4 + 1

/-- Counts the number of positive integers n ≥ 2 for which 10001_n is prime -/
theorem count_prime_base_n : ∃! (n : ℕ), n ≥ 2 ∧ Nat.Prime (base_n n) := by
  sorry

end NUMINAMATH_CALUDE_count_prime_base_n_l3865_386506


namespace NUMINAMATH_CALUDE_cubic_polynomial_problem_l3865_386561

variable (a b c : ℝ)
variable (P : ℝ → ℝ)

theorem cubic_polynomial_problem :
  (∀ x, x^3 - 2*x^2 - 4*x - 1 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∃ p q r s : ℝ, ∀ x, P x = p*x^3 + q*x^2 + r*x + s) →
  P a = b + 2*c →
  P b = 2*a + c →
  P c = a + 2*b →
  P (a + b + c) = -20 →
  ∀ x, P x = 4*x^3 - 6*x^2 - 12*x := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_problem_l3865_386561


namespace NUMINAMATH_CALUDE_age_difference_l3865_386568

/-- Given three people x, y, and z, where z is 10 decades younger than x,
    prove that the combined age of x and y is 100 years greater than
    the combined age of y and z. -/
theorem age_difference (x y z : ℕ) (h : z = x - 100) :
  (x + y) - (y + z) = 100 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3865_386568


namespace NUMINAMATH_CALUDE_cone_surface_area_l3865_386540

theorem cone_surface_area (r h : ℝ) (hr : r = 4) (hh : h = 2 * Real.sqrt 5) :
  let slant_height := Real.sqrt (r^2 + h^2)
  let base_area := π * r^2
  let lateral_area := π * r * slant_height
  base_area + lateral_area = 40 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3865_386540


namespace NUMINAMATH_CALUDE_matthew_crackers_l3865_386510

theorem matthew_crackers (initial_crackers : ℕ) 
  (friends : ℕ) 
  (crackers_eaten_per_friend : ℕ) 
  (crackers_left : ℕ) : 
  friends = 2 ∧ 
  crackers_eaten_per_friend = 6 ∧ 
  crackers_left = 11 ∧ 
  initial_crackers = friends * (crackers_eaten_per_friend * 2) + crackers_left → 
  initial_crackers = 35 := by
sorry

end NUMINAMATH_CALUDE_matthew_crackers_l3865_386510


namespace NUMINAMATH_CALUDE_tails_appearance_l3865_386559

/-- The number of coin flips -/
def total_flips : ℕ := 20

/-- The frequency of getting "heads" -/
def heads_frequency : ℚ := 45/100

/-- The number of times "tails" appears -/
def tails_count : ℕ := 11

/-- Theorem: Given a coin flipped 20 times with a frequency of getting "heads" of 0.45,
    the number of times "tails" appears is 11. -/
theorem tails_appearance :
  (total_flips : ℚ) * (1 - heads_frequency) = tails_count := by sorry

end NUMINAMATH_CALUDE_tails_appearance_l3865_386559


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3865_386556

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) : A = 64 * Real.pi → d = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3865_386556


namespace NUMINAMATH_CALUDE_multiple_calculation_l3865_386553

theorem multiple_calculation (a b m : ℤ) : 
  b = 8 → 
  b - a = 3 → 
  a * b = m * (a + b) + 14 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_multiple_calculation_l3865_386553


namespace NUMINAMATH_CALUDE_survey_preference_theorem_l3865_386504

theorem survey_preference_theorem (total_students : ℕ) 
                                  (mac_preference : ℕ) 
                                  (no_preference : ℕ) 
                                  (h1 : total_students = 350)
                                  (h2 : mac_preference = 100)
                                  (h3 : no_preference = 140) : 
  total_students - mac_preference - (mac_preference / 5) - no_preference = 90 := by
  sorry

#check survey_preference_theorem

end NUMINAMATH_CALUDE_survey_preference_theorem_l3865_386504


namespace NUMINAMATH_CALUDE_cruise_ship_cabins_l3865_386597

/-- Represents the total number of cabins on a cruise ship -/
def total_cabins : ℕ := 600

/-- Represents the number of Deluxe cabins -/
def deluxe_cabins : ℕ := 30

/-- Theorem stating that the total number of cabins on the cruise ship is 600 -/
theorem cruise_ship_cabins :
  (deluxe_cabins : ℝ) + 0.2 * total_cabins + 3/4 * total_cabins = total_cabins :=
by sorry

end NUMINAMATH_CALUDE_cruise_ship_cabins_l3865_386597


namespace NUMINAMATH_CALUDE_not_always_zero_l3865_386536

def heartsuit (x y : ℝ) : ℝ := |x - 2*y|

theorem not_always_zero : ¬ ∀ x : ℝ, heartsuit x x = 0 := by
  sorry

end NUMINAMATH_CALUDE_not_always_zero_l3865_386536


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l3865_386599

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (3 ∣ n) ∧ (4 ∣ n) ∧ (7 ∣ n) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (3 ∣ m) ∧ (4 ∣ m) ∧ (7 ∣ m) → n ≤ m) ∧
  n = 168 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l3865_386599


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3865_386569

/-- An arithmetic sequence with given terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_a2 : a 2 = 12)
  (h_a6 : a 6 = 4) :
  ∃ d : ℝ, d = -2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3865_386569


namespace NUMINAMATH_CALUDE_triangle_segment_calculation_l3865_386572

/-- Given a triangle ABC with point D on AB and point E on AD, prove that FC has a specific value. -/
theorem triangle_segment_calculation (DC CB : ℝ) (h1 : DC = 10) (h2 : CB = 12)
  (AB AD ED : ℝ) (h3 : AB = (1/5) * AD) (h4 : ED = (2/3) * AD) : 
  ∃ (FC : ℝ), FC = 35/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_calculation_l3865_386572


namespace NUMINAMATH_CALUDE_gcd_63_84_l3865_386577

theorem gcd_63_84 : Nat.gcd 63 84 = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_63_84_l3865_386577


namespace NUMINAMATH_CALUDE_problem_solution_l3865_386576

theorem problem_solution : Real.sqrt 9 + 2⁻¹ + (-1)^2023 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3865_386576


namespace NUMINAMATH_CALUDE_age_sum_proof_l3865_386587

theorem age_sum_proof (a b c : ℕ+) : 
  a * b * c = 72 → 
  a ≤ b ∧ a ≤ c → 
  a + b + c = 15 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l3865_386587


namespace NUMINAMATH_CALUDE_pencil_notebook_cost_l3865_386552

/-- Given the cost of pencils and notebooks, calculate the cost of a different quantity -/
theorem pencil_notebook_cost 
  (pencil_price notebook_price : ℕ) 
  (h1 : 4 * pencil_price + 3 * notebook_price = 9600)
  (h2 : 2 * pencil_price + 2 * notebook_price = 5400) :
  8 * pencil_price + 7 * notebook_price = 20400 :=
by sorry

end NUMINAMATH_CALUDE_pencil_notebook_cost_l3865_386552


namespace NUMINAMATH_CALUDE_equation_solution_l3865_386520

theorem equation_solution (x : ℝ) (h : x ≠ 2/3) :
  (7*x + 2) / (3*x^2 + 7*x - 6) = 3*x / (3*x - 2) ↔ 
  x = (-1 + Real.sqrt 7) / 3 ∨ x = (-1 - Real.sqrt 7) / 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3865_386520


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3865_386557

theorem contrapositive_equivalence (a b : ℝ) :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3865_386557


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l3865_386566

/-- The ratio of a man's age to his son's age in two years -/
def age_ratio (man_age son_age : ℕ) : ℚ :=
  (man_age + 2) / (son_age + 2)

/-- Theorem stating the age ratio of a man to his son in two years -/
theorem man_son_age_ratio (son_age : ℕ) (h1 : son_age = 22) :
  age_ratio (son_age + 24) son_age = 2 := by
  sorry

#check man_son_age_ratio

end NUMINAMATH_CALUDE_man_son_age_ratio_l3865_386566


namespace NUMINAMATH_CALUDE_store_a_cheaper_l3865_386562

/-- Represents the cost function for Store A -/
def cost_store_a (x : ℕ) : ℝ :=
  if x ≤ 10 then x
  else 10 + 0.7 * (x - 10)

/-- Represents the cost function for Store B -/
def cost_store_b (x : ℕ) : ℝ := 0.85 * x

/-- The number of exercise books Xiao Ming wants to buy -/
def num_books : ℕ := 22

theorem store_a_cheaper :
  cost_store_a num_books < cost_store_b num_books :=
sorry

end NUMINAMATH_CALUDE_store_a_cheaper_l3865_386562


namespace NUMINAMATH_CALUDE_target_distribution_l3865_386567

def target_parts : Nat := 10

def arrange_decreasing (n : Nat) (center : Nat) (middle : Nat) (outer : Nat) : Nat :=
  sorry

def arrange_equal_sum (n : Nat) (center : Nat) (middle : Nat) (outer : Nat) : Nat :=
  sorry

theorem target_distribution :
  (Nat.factorial target_parts = 3628800) ∧
  (arrange_decreasing target_parts 1 3 6 = 4320) ∧
  (arrange_equal_sum target_parts 1 3 6 = 34560) := by
  sorry

end NUMINAMATH_CALUDE_target_distribution_l3865_386567


namespace NUMINAMATH_CALUDE_brooke_jacks_eight_days_l3865_386500

/-- Represents the number of jumping jacks Sidney does on a given day -/
def sidney_jacks : Nat → Nat
  | 0 => 20  -- Monday
  | 1 => 36  -- Tuesday
  | n + 2 => sidney_jacks (n + 1) + (16 + 2 * n)  -- Following days

/-- The total number of jumping jacks Sidney does over 8 days -/
def sidney_total : Nat := (List.range 8).map sidney_jacks |>.sum

/-- The number of jumping jacks Brooke does is four times Sidney's -/
def brooke_total : Nat := 4 * sidney_total

theorem brooke_jacks_eight_days : brooke_total = 2880 := by
  sorry

end NUMINAMATH_CALUDE_brooke_jacks_eight_days_l3865_386500


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3865_386515

theorem constant_term_expansion (a : ℝ) : 
  (∃ (f : ℝ → ℝ), ∀ x, f x = (x + 1) * (x / 2 - a / Real.sqrt x)^6) →
  (∃ (g : ℝ → ℝ), ∀ x, g x = (x + 1) * (x / 2 - a / Real.sqrt x)^6 ∧ 
    (∃ c, c = 60 ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |g x - c| < ε))) →
  a = 2 ∨ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3865_386515
