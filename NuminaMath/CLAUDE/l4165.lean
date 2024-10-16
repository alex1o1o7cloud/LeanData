import Mathlib

namespace NUMINAMATH_CALUDE_log_sum_fifty_twenty_l4165_416503

theorem log_sum_fifty_twenty : Real.log 50 / Real.log 10 + Real.log 20 / Real.log 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_fifty_twenty_l4165_416503


namespace NUMINAMATH_CALUDE_omega_range_l4165_416521

theorem omega_range (ω : ℝ) (h1 : ω > 1/4) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x) - Real.cos (ω * x)
  let symmetry_axis (k : ℤ) : ℝ := (3*π/4 + k*π) / ω
  (∀ k : ℤ, symmetry_axis k ∉ Set.Ioo (2*π) (3*π)) →
  ω ∈ Set.Icc (3/8) (7/12) ∪ Set.Icc (7/8) (11/12) :=
by sorry

end NUMINAMATH_CALUDE_omega_range_l4165_416521


namespace NUMINAMATH_CALUDE_base_conversion_725_9_l4165_416519

def base_9_to_base_3 (n : Nat) : Nat :=
  -- Definition of conversion from base 9 to base 3
  sorry

theorem base_conversion_725_9 :
  base_9_to_base_3 725 = 210212 :=
sorry

end NUMINAMATH_CALUDE_base_conversion_725_9_l4165_416519


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4165_416554

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value :
  let m : ℝ × ℝ := (-2, 4)
  let n : ℝ × ℝ := (x, -1)
  parallel m n → x = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4165_416554


namespace NUMINAMATH_CALUDE_karen_donald_children_count_l4165_416507

/-- Represents the number of children Karen and Donald have -/
def karen_donald_children : ℕ := sorry

/-- Represents the number of children Tom and Eva have -/
def tom_eva_children : ℕ := 4

/-- Represents the total number of legs in the pool -/
def legs_in_pool : ℕ := 16

/-- Represents the number of people not in the pool -/
def people_not_in_pool : ℕ := 6

/-- Proves that Karen and Donald have 6 children given the conditions -/
theorem karen_donald_children_count :
  karen_donald_children = 6 := by sorry

end NUMINAMATH_CALUDE_karen_donald_children_count_l4165_416507


namespace NUMINAMATH_CALUDE_decreased_amount_l4165_416571

theorem decreased_amount (N : ℝ) (A : ℝ) (h1 : N = 50) (h2 : 0.20 * N - A = 6) : A = 4 := by
  sorry

end NUMINAMATH_CALUDE_decreased_amount_l4165_416571


namespace NUMINAMATH_CALUDE_average_salary_is_8800_l4165_416505

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 14000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

theorem average_salary_is_8800 : 
  (total_salary : ℚ) / num_people = 8800 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_is_8800_l4165_416505


namespace NUMINAMATH_CALUDE_arrangement_count_l4165_416578

def number_of_arrangements (n m k : ℕ) : ℕ :=
  2 * (m * (m - 1) * (m - 2) * (m - 3))

theorem arrangement_count :
  number_of_arrangements 8 6 2 = 720 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l4165_416578


namespace NUMINAMATH_CALUDE_base_salary_per_week_l4165_416573

def past_week_incomes : List ℝ := [406, 413, 420, 436, 395]
def num_past_weeks : ℕ := 5
def num_future_weeks : ℕ := 2
def total_weeks : ℕ := num_past_weeks + num_future_weeks
def average_commission_future : ℝ := 345
def average_weekly_income : ℝ := 500

def total_past_income : ℝ := past_week_incomes.sum
def total_income : ℝ := average_weekly_income * total_weeks
def total_future_income : ℝ := total_income - total_past_income
def total_future_commission : ℝ := average_commission_future * num_future_weeks
def total_future_base_salary : ℝ := total_future_income - total_future_commission

theorem base_salary_per_week : 
  total_future_base_salary / num_future_weeks = 370 := by sorry

end NUMINAMATH_CALUDE_base_salary_per_week_l4165_416573


namespace NUMINAMATH_CALUDE_total_milks_taken_l4165_416580

/-- The total number of milks taken is the sum of all individual milk selections. -/
theorem total_milks_taken (chocolate : ℕ) (strawberry : ℕ) (regular : ℕ) (almond : ℕ) (soy : ℕ)
  (h1 : chocolate = 120)
  (h2 : strawberry = 315)
  (h3 : regular = 230)
  (h4 : almond = 145)
  (h5 : soy = 97) :
  chocolate + strawberry + regular + almond + soy = 907 := by
  sorry

end NUMINAMATH_CALUDE_total_milks_taken_l4165_416580


namespace NUMINAMATH_CALUDE_original_number_l4165_416551

theorem original_number : ∃ x : ℚ, 213 * x = 3408 ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l4165_416551


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l4165_416526

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) :
  total_students = 150 →
  red_students = 60 →
  green_students = 90 →
  total_pairs = 75 →
  red_red_pairs = 28 →
  total_students = red_students + green_students →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 43 ∧ 
    green_green_pairs + red_red_pairs + (total_students - 2 * red_red_pairs - 2 * green_green_pairs) / 2 = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l4165_416526


namespace NUMINAMATH_CALUDE_parentheses_placement_l4165_416557

theorem parentheses_placement :
  let original := 0.5 + 0.5 / 0.5 + 0.5 / 0.5
  let with_parentheses := ((0.5 + 0.5) / 0.5 + 0.5) / 0.5
  with_parentheses = 5 ∧ with_parentheses ≠ original :=
by sorry

end NUMINAMATH_CALUDE_parentheses_placement_l4165_416557


namespace NUMINAMATH_CALUDE_parking_lot_spaces_l4165_416506

/-- Represents a parking lot with full-sized and compact car spaces. -/
structure ParkingLot where
  full_sized : ℕ
  compact : ℕ

/-- Calculates the total number of spaces in a parking lot. -/
def total_spaces (lot : ParkingLot) : ℕ :=
  lot.full_sized + lot.compact

/-- Represents the ratio of full-sized to compact car spaces. -/
structure SpaceRatio where
  full_sized : ℕ
  compact : ℕ

/-- Theorem: Given a parking lot with 330 full-sized car spaces and a ratio of 11:4
    for full-sized to compact car spaces, the total number of spaces is 450. -/
theorem parking_lot_spaces (ratio : SpaceRatio) 
    (h1 : ratio.full_sized = 11)
    (h2 : ratio.compact = 4)
    (lot : ParkingLot)
    (h3 : lot.full_sized = 330)
    (h4 : lot.full_sized * ratio.compact = lot.compact * ratio.full_sized) :
    total_spaces lot = 450 := by
  sorry

#check parking_lot_spaces

end NUMINAMATH_CALUDE_parking_lot_spaces_l4165_416506


namespace NUMINAMATH_CALUDE_workshop_workers_l4165_416591

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := sorry

/-- Represents the average salary of all workers -/
def average_salary : ℚ := 9500

/-- Represents the number of technicians -/
def num_technicians : ℕ := 7

/-- Represents the average salary of technicians -/
def technician_salary : ℚ := 12000

/-- Represents the average salary of non-technicians -/
def non_technician_salary : ℚ := 6000

/-- Theorem stating that the total number of workers is 12 -/
theorem workshop_workers : total_workers = 12 := by sorry

end NUMINAMATH_CALUDE_workshop_workers_l4165_416591


namespace NUMINAMATH_CALUDE_franks_age_to_tys_age_ratio_l4165_416536

/-- Proves that the ratio of Frank's age in 5 years to Ty's current age is 3:1 -/
theorem franks_age_to_tys_age_ratio : 
  let karen_age : ℕ := 2
  let carla_age : ℕ := karen_age + 2
  let ty_age : ℕ := 2 * carla_age + 4
  let frank_future_age : ℕ := 36
  (frank_future_age : ℚ) / ty_age = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_franks_age_to_tys_age_ratio_l4165_416536


namespace NUMINAMATH_CALUDE_teddy_pillows_l4165_416532

/-- The number of pounds in a ton -/
def pounds_per_ton : ℕ := 2000

/-- The amount of fluffy foam material Teddy has, in tons -/
def teddy_material : ℕ := 3

/-- The amount of fluffy foam material used for each pillow, in pounds -/
def material_per_pillow : ℕ := 5 - 3

/-- The number of pillows Teddy can make -/
def pillows_made : ℕ := (teddy_material * pounds_per_ton) / material_per_pillow

theorem teddy_pillows :
  pillows_made = 3000 := by sorry

end NUMINAMATH_CALUDE_teddy_pillows_l4165_416532


namespace NUMINAMATH_CALUDE_largest_n_with_conditions_l4165_416568

theorem largest_n_with_conditions : 
  ∃ (n : ℕ), n = 25 ∧ 
  (∃ (k : ℕ), n^2 = (k+1)^4 - k^4) ∧ 
  (∃ (b : ℕ), 3*n + 100 = b^2) ∧
  (∀ (m : ℕ), m > n → 
    (∀ (j : ℕ), m^2 ≠ (j+1)^4 - j^4) ∨ 
    (∀ (c : ℕ), 3*m + 100 ≠ c^2)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_conditions_l4165_416568


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_nineteen_fifths_l4165_416556

theorem greatest_integer_less_than_negative_nineteen_fifths :
  Int.floor (-19 / 5 : ℚ) = -4 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_nineteen_fifths_l4165_416556


namespace NUMINAMATH_CALUDE_correct_atomic_symbol_proof_l4165_416547

/-- Represents an element X in an ionic compound XCl_n -/
structure ElementX where
  m : ℕ  -- number of neutrons
  y : ℕ  -- number of electrons outside the nucleus
  n : ℕ  -- number of chlorine atoms in the compound

/-- Represents the atomic symbol of an isotope -/
structure AtomicSymbol where
  subscript : ℕ
  superscript : ℕ

/-- Returns the correct atomic symbol for an element X -/
def correct_atomic_symbol (x : ElementX) : AtomicSymbol :=
  { subscript := x.y + x.n
  , superscript := x.m + x.y + x.n }

/-- Theorem stating that the correct atomic symbol for element X is _{y+n}^{m+y+n}X -/
theorem correct_atomic_symbol_proof (x : ElementX) :
  correct_atomic_symbol x = { subscript := x.y + x.n, superscript := x.m + x.y + x.n } :=
by sorry

end NUMINAMATH_CALUDE_correct_atomic_symbol_proof_l4165_416547


namespace NUMINAMATH_CALUDE_cos_sin_identity_l4165_416584

theorem cos_sin_identity (α : Real) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = -(2 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l4165_416584


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l4165_416541

theorem quadratic_equation_result (x : ℝ) (h : 6 * x^2 + 9 = 4 * x + 16) : (12 * x - 4)^2 = 188 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l4165_416541


namespace NUMINAMATH_CALUDE_cube_solid_surface_area_l4165_416569

/-- A solid composed of 7 identical cubes -/
structure CubeSolid where
  -- The volume of each individual cube
  cube_volume : ℝ
  -- The side length of each cube
  cube_side : ℝ
  -- The total volume of the solid
  total_volume : ℝ
  -- The surface area of the solid
  surface_area : ℝ
  -- Conditions
  cube_volume_def : cube_volume = total_volume / 7
  cube_side_def : cube_side ^ 3 = cube_volume
  surface_area_def : surface_area = 30 * (cube_side ^ 2)

/-- Theorem: If the total volume of the CubeSolid is 875 cm³, then its surface area is 750 cm² -/
theorem cube_solid_surface_area (s : CubeSolid) (h : s.total_volume = 875) : 
  s.surface_area = 750 := by
  sorry

#check cube_solid_surface_area

end NUMINAMATH_CALUDE_cube_solid_surface_area_l4165_416569


namespace NUMINAMATH_CALUDE_ellipse_right_triangle_x_coordinate_l4165_416537

/-- The x-coordinate of a point on an ellipse forming a right triangle with the foci -/
theorem ellipse_right_triangle_x_coordinate 
  (x y : ℝ) 
  (h_ellipse : x^2/16 + y^2/25 = 1) 
  (h_on_ellipse : ∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y)
  (h_foci : ∃ (F₁ F₂ : ℝ × ℝ), F₁.1 = 0 ∧ F₂.1 = 0)
  (h_right_triangle : ∃ (F₁ F₂ : ℝ × ℝ), F₁.1 = 0 ∧ F₂.1 = 0 ∧ 
    (F₁.2 - y)^2 + x^2 + (F₂.2 - y)^2 + x^2 = (F₂.2 - F₁.2)^2) :
  x = 16/5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_right_triangle_x_coordinate_l4165_416537


namespace NUMINAMATH_CALUDE_original_denominator_proof_l4165_416566

theorem original_denominator_proof (d : ℤ) : 
  (2 : ℚ) / d ≠ 0 →
  (5 : ℚ) / (d + 3) = 1 / 3 →
  d = 12 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l4165_416566


namespace NUMINAMATH_CALUDE_factor_of_x4_plus_12_l4165_416582

theorem factor_of_x4_plus_12 (x : ℝ) : ∃ (y : ℝ), x^4 + 12 = (x^2 - 3*x + 3) * y := by
  sorry

end NUMINAMATH_CALUDE_factor_of_x4_plus_12_l4165_416582


namespace NUMINAMATH_CALUDE_union_of_sets_l4165_416524

theorem union_of_sets (a b : ℕ) (M N : Set ℕ) : 
  M = {3, 2^a} → N = {a, b} → M ∩ N = {2} → M ∪ N = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l4165_416524


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l4165_416538

theorem triangle_angle_sum (A B C : ℝ) (h1 : A = 40) (h2 : B = 80) : C = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l4165_416538


namespace NUMINAMATH_CALUDE_range_of_m_l4165_416576

-- Define the sets M and N
def M (m : ℝ) : Set ℝ := {x : ℝ | x + m ≥ 0}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 8 < 0}

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Theorem statement
theorem range_of_m (m : ℝ) : (Set.compl (M m) ∩ N = ∅) → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l4165_416576


namespace NUMINAMATH_CALUDE_selection_theorem_l4165_416592

def num_boys : ℕ := 5
def num_girls : ℕ := 4
def total_people : ℕ := num_boys + num_girls
def num_selected : ℕ := 4

/-- The number of ways to select 4 people from 5 boys and 4 girls, 
    ensuring at least one of boy A and girl B participates, 
    and both boys and girls are present -/
def selection_ways : ℕ := sorry

theorem selection_theorem : 
  selection_ways = (total_people.choose num_selected) - 
                   (num_boys.choose num_selected) - 
                   (num_girls.choose num_selected) := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l4165_416592


namespace NUMINAMATH_CALUDE_percentage_difference_l4165_416596

theorem percentage_difference (A B C : ℝ) 
  (hB_C : B = 0.63 * C) 
  (hB_A : B = 0.90 * A) : 
  A = 0.70 * C := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l4165_416596


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l4165_416559

theorem inequality_solution_sets (a : ℝ) :
  let f := fun x => a * x^2 - (a + 2) * x + 2
  (a = -1 → {x : ℝ | f x < 0} = {x : ℝ | x < -2 ∨ x > 1}) ∧
  (a = 0 → {x : ℝ | f x < 0} = {x : ℝ | x > 1}) ∧
  (a < 0 → {x : ℝ | f x < 0} = {x : ℝ | x < 2/a ∨ x > 1}) ∧
  (0 < a ∧ a < 2 → {x : ℝ | f x < 0} = {x : ℝ | 1 < x ∧ x < 2/a}) ∧
  (a = 2 → {x : ℝ | f x < 0} = ∅) ∧
  (a > 2 → {x : ℝ | f x < 0} = {x : ℝ | 2/a < x ∧ x < 1}) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l4165_416559


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l4165_416500

/-- Given a circle described by the equation x^2 + y^2 - 8 = 2x - 4y,
    prove that its center is at (1, -2) and its radius is √13. -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧
    radius = Real.sqrt 13 ∧
    ∀ (x y : ℝ), x^2 + y^2 - 8 = 2*x - 4*y ↔
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l4165_416500


namespace NUMINAMATH_CALUDE_go_complexity_vs_universe_atoms_l4165_416594

/-- Approximation of the upper limit of state space complexity of Go -/
def M : ℝ := 3^361

/-- Approximation of the total number of atoms in the observable universe -/
def N : ℝ := 10^80

/-- Approximation of log base 10 of 3 -/
def log10_3 : ℝ := 0.48

/-- The closest value to M/N among the given options -/
def closest_value : ℝ := 10^93

theorem go_complexity_vs_universe_atoms :
  abs (M / N - closest_value) = 
    min (abs (M / N - 10^33)) 
        (min (abs (M / N - 10^53)) 
             (min (abs (M / N - 10^73)) 
                  (abs (M / N - 10^93)))) := by
  sorry

end NUMINAMATH_CALUDE_go_complexity_vs_universe_atoms_l4165_416594


namespace NUMINAMATH_CALUDE_circle_radius_given_area_circumference_ratio_l4165_416598

theorem circle_radius_given_area_circumference_ratio 
  (A C : ℝ) (h1 : A > 0) (h2 : C > 0) (h3 : A / C = 10) : 
  ∃ r : ℝ, r > 0 ∧ A = π * r^2 ∧ C = 2 * π * r ∧ r = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_given_area_circumference_ratio_l4165_416598


namespace NUMINAMATH_CALUDE_coefficient_of_x_is_50_l4165_416570

def expression (x : ℝ) : ℝ := 5 * (x - 6) + 6 * (8 - 3 * x^2 + 3 * x) - 9 * (3 * x - 2)

theorem coefficient_of_x_is_50 : 
  ∃ (a b c : ℝ), ∀ x, expression x = a * x^2 + 50 * x + c :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_is_50_l4165_416570


namespace NUMINAMATH_CALUDE_winnie_balloon_distribution_l4165_416508

theorem winnie_balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 220) (h2 : num_friends = 9) :
  total_balloons % num_friends = 4 := by sorry

end NUMINAMATH_CALUDE_winnie_balloon_distribution_l4165_416508


namespace NUMINAMATH_CALUDE_lakeisha_lawn_mowing_l4165_416583

/-- The amount LaKeisha charges per square foot of lawn -/
def charge_per_sqft : ℚ := 1/10

/-- The cost of the book set -/
def book_cost : ℚ := 150

/-- The length of each lawn -/
def lawn_length : ℕ := 20

/-- The width of each lawn -/
def lawn_width : ℕ := 15

/-- The number of lawns already mowed -/
def lawns_mowed : ℕ := 3

/-- The additional square feet LaKeisha needs to mow -/
def additional_sqft : ℕ := 600

theorem lakeisha_lawn_mowing :
  (lawn_length * lawn_width * lawns_mowed * charge_per_sqft) + 
  (additional_sqft * charge_per_sqft) = book_cost :=
sorry

end NUMINAMATH_CALUDE_lakeisha_lawn_mowing_l4165_416583


namespace NUMINAMATH_CALUDE_nearest_integer_to_a_fifth_l4165_416522

theorem nearest_integer_to_a_fifth (a b c : ℝ) 
  (h_order : a ≥ b ∧ b ≥ c)
  (h_eq1 : a^2 * b * c + a * b^2 * c + a * b * c^2 + 8 = a + b + c)
  (h_eq2 : a^2 * b + a^2 * c + b^2 * c + b^2 * a + c^2 * a + c^2 * b + 3 * a * b * c = -4)
  (h_eq3 : a^2 * b^2 * c + a * b^2 * c^2 + a^2 * b * c^2 = 2 + a * b + b * c + c * a)
  (h_sum_pos : a + b + c > 0) :
  ∃ (n : ℤ), |n - a^5| < 1/2 ∧ n = 1279 := by
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_a_fifth_l4165_416522


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l4165_416517

/-- A quadratic function is a function of the form f(x) = ax^2 + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique
  (f : ℝ → ℝ)
  (h1 : IsQuadratic f)
  (h2 : ∀ x, f x + f (x + 1) = 2 * x^2 - 6 * x + 5) :
  ∀ x, f x = x^2 - 4 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l4165_416517


namespace NUMINAMATH_CALUDE_recipe_total_cups_l4165_416530

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the amount of butter -/
def totalIngredients (ratio : RecipeRatio) (butterCups : ℕ) : ℕ :=
  butterCups * (ratio.butter + ratio.flour + ratio.sugar) / ratio.butter

theorem recipe_total_cups (ratio : RecipeRatio) (butterCups : ℕ) 
    (h1 : ratio.butter = 1) 
    (h2 : ratio.flour = 5) 
    (h3 : ratio.sugar = 3) 
    (h4 : butterCups = 9) : 
  totalIngredients ratio butterCups = 81 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l4165_416530


namespace NUMINAMATH_CALUDE_odd_heads_probability_not_simple_closed_form_l4165_416533

/-- Represents the probability of getting heads on the i-th flip -/
def p (i : ℕ) : ℚ := 3/4 - i/200

/-- Represents the probability of having an odd number of heads after n flips -/
noncomputable def P : ℕ → ℚ
  | 0 => 0
  | n + 1 => (1 - 2 * p n) * P n + p n

/-- The statement that the probability of odd number of heads after 100 flips
    cannot be expressed in a simple closed form -/
theorem odd_heads_probability_not_simple_closed_form :
  ∃ (f : ℚ → Prop), f (P 100) ∧ ∀ (x : ℚ), f x → x = P 100 :=
sorry

end NUMINAMATH_CALUDE_odd_heads_probability_not_simple_closed_form_l4165_416533


namespace NUMINAMATH_CALUDE_a_5_value_l4165_416511

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem a_5_value (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) 
  (h_3 : a 3 = 7) (h_9 : a 9 = 19) : a 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_a_5_value_l4165_416511


namespace NUMINAMATH_CALUDE_total_players_on_ground_l4165_416560

/-- The number of cricket players -/
def cricket_players : ℕ := 16

/-- The number of hockey players -/
def hockey_players : ℕ := 12

/-- The number of football players -/
def football_players : ℕ := 18

/-- The number of softball players -/
def softball_players : ℕ := 13

/-- Theorem: The total number of players on the ground is 59 -/
theorem total_players_on_ground : 
  cricket_players + hockey_players + football_players + softball_players = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l4165_416560


namespace NUMINAMATH_CALUDE_frog_mouse_jump_difference_l4165_416529

/-- Represents the jumping contest between a grasshopper, a frog, and a mouse -/
def jumping_contest (grasshopper_jump mouse_jump frog_jump : ℕ) : Prop :=
  grasshopper_jump = 14 ∧
  frog_jump = grasshopper_jump + 37 ∧
  mouse_jump = grasshopper_jump + 21

/-- Theorem stating the difference between the frog's and mouse's jump distances -/
theorem frog_mouse_jump_difference 
  (grasshopper_jump mouse_jump frog_jump : ℕ)
  (h : jumping_contest grasshopper_jump mouse_jump frog_jump) :
  frog_jump - mouse_jump = 16 := by
  sorry

#check frog_mouse_jump_difference

end NUMINAMATH_CALUDE_frog_mouse_jump_difference_l4165_416529


namespace NUMINAMATH_CALUDE_f_not_prime_l4165_416565

def f (n : ℕ+) : ℕ := n^4 + 100 * n^2 + 169

theorem f_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (f n) := by
  sorry

end NUMINAMATH_CALUDE_f_not_prime_l4165_416565


namespace NUMINAMATH_CALUDE_initial_blocks_l4165_416531

theorem initial_blocks (initial final added : ℕ) : 
  final = initial + added → 
  final = 65 → 
  added = 30 → 
  initial = 35 := by sorry

end NUMINAMATH_CALUDE_initial_blocks_l4165_416531


namespace NUMINAMATH_CALUDE_min_value_expression_l4165_416577

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l4165_416577


namespace NUMINAMATH_CALUDE_gcd_18_30_l4165_416516

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l4165_416516


namespace NUMINAMATH_CALUDE_ceiling_minus_y_l4165_416510

theorem ceiling_minus_y (y : ℝ) (h : ⌈y⌉ - ⌊y⌋ = 1) : ⌈y⌉ - y = 1 - (y - ⌊y⌋) := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_y_l4165_416510


namespace NUMINAMATH_CALUDE_oblique_square_area_l4165_416520

/-- The area of an oblique two-dimensional drawing of a unit square -/
theorem oblique_square_area :
  ∀ (S_oblique : ℝ),
  (1 : ℝ) ^ 2 = 1 →  -- Side length of original square is 1
  S_oblique / 1 = Real.sqrt 2 / 4 →  -- Ratio of areas
  S_oblique = Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_oblique_square_area_l4165_416520


namespace NUMINAMATH_CALUDE_clips_for_huahuas_handkerchiefs_l4165_416514

/-- The number of clips needed to hang handkerchiefs on clotheslines -/
def clips_needed (handkerchiefs : ℕ) (clotheslines : ℕ) : ℕ :=
  -- We define this function without implementation, as the problem doesn't provide the exact formula
  sorry

/-- Theorem stating the number of clips needed for the given scenario -/
theorem clips_for_huahuas_handkerchiefs :
  clips_needed 40 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_clips_for_huahuas_handkerchiefs_l4165_416514


namespace NUMINAMATH_CALUDE_powers_of_four_unit_digits_l4165_416581

theorem powers_of_four_unit_digits (a b c : ℕ+) : 
  ¬(∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x = (4^a.val) % 10 ∧ 
    y = (4^b.val) % 10 ∧ 
    z = (4^c.val) % 10) := by
  sorry

end NUMINAMATH_CALUDE_powers_of_four_unit_digits_l4165_416581


namespace NUMINAMATH_CALUDE_power_of_product_l4165_416552

theorem power_of_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l4165_416552


namespace NUMINAMATH_CALUDE_b_payment_is_360_l4165_416549

/-- Represents the payment for a group of horses in a pasture -/
structure Payment where
  horses : ℕ
  months : ℕ
  amount : ℚ

/-- Calculates the total horse-months for a payment -/
def horse_months (p : Payment) : ℕ := p.horses * p.months

/-- Theorem: Given the conditions of the pasture rental, B's payment is Rs. 360 -/
theorem b_payment_is_360 
  (total_rent : ℚ)
  (a_payment : Payment)
  (b_payment : Payment)
  (c_payment : Payment)
  (h1 : total_rent = 870)
  (h2 : a_payment.horses = 12 ∧ a_payment.months = 8)
  (h3 : b_payment.horses = 16 ∧ b_payment.months = 9)
  (h4 : c_payment.horses = 18 ∧ c_payment.months = 6) :
  b_payment.amount = 360 := by
  sorry

end NUMINAMATH_CALUDE_b_payment_is_360_l4165_416549


namespace NUMINAMATH_CALUDE_constant_c_value_l4165_416567

theorem constant_c_value (b c : ℝ) :
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) →
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_constant_c_value_l4165_416567


namespace NUMINAMATH_CALUDE_line_x_coordinate_indeterminate_l4165_416579

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Given a line passing through (4, 0), (x₁, 3), and (-12, y₂), 
    prove that x₁ cannot be uniquely determined --/
theorem line_x_coordinate_indeterminate 
  (line : Line)
  (h1 : line.slope * 4 + line.y_intercept = 0)
  (h2 : ∃ x₁, line.slope * x₁ + line.y_intercept = 3)
  (h3 : ∃ y₂, line.slope * (-12) + line.y_intercept = y₂) :
  ¬(∃! x₁, line.slope * x₁ + line.y_intercept = 3) :=
sorry

end NUMINAMATH_CALUDE_line_x_coordinate_indeterminate_l4165_416579


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4165_416544

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 3 → x^2 - 2*x > 0) ∧ 
  (∃ x, x^2 - 2*x > 0 ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4165_416544


namespace NUMINAMATH_CALUDE_ferry_journey_difference_l4165_416527

/-- Represents a ferry with its speed and travel time -/
structure Ferry where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a ferry -/
def distance (f : Ferry) : ℝ := f.speed * f.time

theorem ferry_journey_difference :
  let ferry_p : Ferry := { speed := 8, time := 3 }
  let ferry_q : Ferry := { speed := ferry_p.speed + 1, time := (3 * distance ferry_p) / (ferry_p.speed + 1) }
  ferry_q.time - ferry_p.time = 5 := by
  sorry

end NUMINAMATH_CALUDE_ferry_journey_difference_l4165_416527


namespace NUMINAMATH_CALUDE_jeff_travel_distance_l4165_416504

/-- Calculates the total distance traveled given a list of journey segments --/
def totalDistance (segments : List (Real × Real)) : Real :=
  segments.map (λ (speed, time) => speed * time) |>.sum

/-- Represents Jeff's journey to the capital city --/
def jeffJourney : List (Real × Real) := [
  (80, 3),   -- 80 miles/hour for 3 hours
  (50, 2),   -- 50 miles/hour for 2 hours
  (70, 1),   -- 70 miles/hour for 1 hour
  (60, 2),   -- 60 miles/hour for 2 hours
  (45, 3),   -- 45 miles/hour for 3 hours
  (40, 2),   -- 40 miles/hour for 2 hours
  (30, 2.5)  -- 30 miles/hour for 2.5 hours
]

/-- Theorem: Jeff's total travel distance is 820 miles --/
theorem jeff_travel_distance :
  totalDistance jeffJourney = 820 := by
  sorry


end NUMINAMATH_CALUDE_jeff_travel_distance_l4165_416504


namespace NUMINAMATH_CALUDE_parabola_vertex_l4165_416535

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := (x - 2)^2 - 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, -3)

/-- Theorem: The vertex of the parabola y = (x-2)^2 - 3 is (2, -3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l4165_416535


namespace NUMINAMATH_CALUDE_chess_tournament_games_l4165_416542

/-- The number of games in a chess tournament -/
def num_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Proof that a chess tournament with 25 players, where each player plays 
    four times against every opponent, results in 1200 games total -/
theorem chess_tournament_games :
  num_games 25 4 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l4165_416542


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4165_416558

theorem polynomial_factorization :
  (∀ x : ℝ, 3 * x^2 - 7 * x - 6 = (x - 3) * (3 * x + 2)) ∧
  (∀ x : ℝ, 6 * x^2 - 7 * x - 5 = (2 * x + 1) * (3 * x - 5)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4165_416558


namespace NUMINAMATH_CALUDE_misread_weight_calculation_l4165_416564

/-- Given a class of boys with the following properties:
  * The class has 20 boys
  * The initially calculated average weight was 58.4 kg
  * The correct average weight is 58.7 kg
  * One weight was misread instead of 62 kg
  Prove that the misread weight was 56 kg -/
theorem misread_weight_calculation (class_size : ℕ) (initial_avg : ℝ) (correct_avg : ℝ) (correct_weight : ℝ) :
  class_size = 20 →
  initial_avg = 58.4 →
  correct_avg = 58.7 →
  correct_weight = 62 →
  ∃ (misread_weight : ℝ),
    class_size * correct_avg - class_size * initial_avg = correct_weight - misread_weight ∧
    misread_weight = 56 := by
  sorry

end NUMINAMATH_CALUDE_misread_weight_calculation_l4165_416564


namespace NUMINAMATH_CALUDE_art_gallery_total_pieces_l4165_416553

theorem art_gallery_total_pieces : 
  ∀ (D S : ℕ),
  (2 : ℚ) / 5 * D + (3 : ℚ) / 7 * S = (D + S) * (2 : ℚ) / 5 →
  (1 : ℚ) / 5 * D + (2 : ℚ) / 7 * S = 1500 →
  (2 : ℚ) / 5 * D = 600 →
  D + S = 5700 :=
by
  sorry

end NUMINAMATH_CALUDE_art_gallery_total_pieces_l4165_416553


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l4165_416546

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l4165_416546


namespace NUMINAMATH_CALUDE_f_negative_implies_a_range_l4165_416589

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- State the theorem
theorem f_negative_implies_a_range (a : ℝ) :
  (∃ x : ℝ, f a x < 0) → (a > 2 ∨ a < -2) := by
  sorry

end NUMINAMATH_CALUDE_f_negative_implies_a_range_l4165_416589


namespace NUMINAMATH_CALUDE_A_value_l4165_416562

noncomputable def A (x : ℝ) : ℝ :=
  (Real.sqrt 3 * x^(3/2) - 5 * x^(1/3) + 5 * x^(4/3) - Real.sqrt (3*x)) /
  (Real.sqrt (3*x + 10 * Real.sqrt 3 * x^(5/6) + 25 * x^(2/3)) *
   Real.sqrt (1 - 2/x + 1/x^2))

theorem A_value (x : ℝ) (hx : x > 0) :
  (0 < x ∧ x < 1 → A x = -x) ∧
  (x > 1 → A x = x) := by
  sorry

end NUMINAMATH_CALUDE_A_value_l4165_416562


namespace NUMINAMATH_CALUDE_functional_equation_solution_l4165_416588

/-- The functional equation satisfied by f --/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) - f (x - y) = 2 * y * (3 * x^2 + y^2)

/-- The theorem statement --/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f →
  ∃ a : ℝ, ∀ x : ℝ, f x = x^3 + a :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l4165_416588


namespace NUMINAMATH_CALUDE_ellipse_x_intersection_l4165_416539

/-- Definition of the ellipse based on the given conditions -/
def ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (0, 1)
  let F₂ : ℝ × ℝ := (4, 0)
  let d : ℝ := Real.sqrt 2 + 3
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = d

/-- The theorem stating the other intersection point of the ellipse with the x-axis -/
theorem ellipse_x_intersection :
  ellipse (1, 0) →
  ∃ x : ℝ, x ≠ 1 ∧ ellipse (x, 0) ∧ x = 3 * Real.sqrt 2 / 4 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_x_intersection_l4165_416539


namespace NUMINAMATH_CALUDE_total_highlighters_l4165_416513

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h1 : pink = 9)
  (h2 : yellow = 8)
  (h3 : blue = 5) :
  pink + yellow + blue = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l4165_416513


namespace NUMINAMATH_CALUDE_sum_in_interval_l4165_416561

theorem sum_in_interval :
  let a := 2 + 3 / 9
  let b := 3 + 3 / 4
  let c := 5 + 3 / 25
  let sum := a + b + c
  8 < sum ∧ sum < 9 := by
sorry

end NUMINAMATH_CALUDE_sum_in_interval_l4165_416561


namespace NUMINAMATH_CALUDE_number_added_to_multiples_of_three_l4165_416597

theorem number_added_to_multiples_of_three : ∃ x : ℕ, 
  x + (3 * 14 + 3 * 15 + 3 * 18) = 152 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_added_to_multiples_of_three_l4165_416597


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l4165_416555

theorem quadratic_root_difference : ∀ (x₁ x₂ : ℝ),
  (7 + 4 * Real.sqrt 3) * x₁^2 + (2 + Real.sqrt 3) * x₁ - 2 = 0 →
  (7 + 4 * Real.sqrt 3) * x₂^2 + (2 + Real.sqrt 3) * x₂ - 2 = 0 →
  x₁ ≠ x₂ →
  max x₁ x₂ - min x₁ x₂ = 6 - 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l4165_416555


namespace NUMINAMATH_CALUDE_remainder_problem_l4165_416501

theorem remainder_problem (N : ℕ) : N % 751 = 53 → N % 29 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l4165_416501


namespace NUMINAMATH_CALUDE_albums_needed_for_xiao_hong_l4165_416586

/-- Calculates the minimum number of complete photo albums needed to store a given number of photos. -/
def minimum_albums_needed (pages_per_album : ℕ) (photos_per_page : ℕ) (total_photos : ℕ) : ℕ :=
  (total_photos + pages_per_album * photos_per_page - 1) / (pages_per_album * photos_per_page)

/-- Proves that 6 albums are needed for the given conditions. -/
theorem albums_needed_for_xiao_hong : minimum_albums_needed 32 5 900 = 6 := by
  sorry

#eval minimum_albums_needed 32 5 900

end NUMINAMATH_CALUDE_albums_needed_for_xiao_hong_l4165_416586


namespace NUMINAMATH_CALUDE_original_average_age_l4165_416518

/-- Proves that the original average age of a class was 40 years, given the conditions of the problem -/
theorem original_average_age (original_count : ℕ) (new_count : ℕ) (new_avg_age : ℕ) (avg_decrease : ℕ) :
  original_count = 10 →
  new_count = 10 →
  new_avg_age = 32 →
  avg_decrease = 4 →
  ∃ (original_avg_age : ℕ),
    (original_avg_age * original_count + new_avg_age * new_count) / (original_count + new_count) 
    = original_avg_age - avg_decrease ∧
    original_avg_age = 40 :=
by sorry

end NUMINAMATH_CALUDE_original_average_age_l4165_416518


namespace NUMINAMATH_CALUDE_arcsin_arctan_equation_solution_l4165_416599

theorem arcsin_arctan_equation_solution :
  ∃ x : ℝ, x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ) ∧ Real.arcsin x + Real.arcsin (2*x) = Real.arctan x :=
by
  sorry

end NUMINAMATH_CALUDE_arcsin_arctan_equation_solution_l4165_416599


namespace NUMINAMATH_CALUDE_sum_equality_implies_k_value_l4165_416545

/-- Given a real number k > 1 satisfying the infinite sum equation, prove k equals the specified value. -/
theorem sum_equality_implies_k_value (k : ℝ) 
  (h1 : k > 1) 
  (h2 : ∑' n, (7 * n - 3) / k^n = 2) : 
  k = 2 + 1.5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sum_equality_implies_k_value_l4165_416545


namespace NUMINAMATH_CALUDE_free_all_friends_time_l4165_416543

/-- Time to pick a cheap handcuff lock in minutes -/
def cheap_time : ℕ := 10

/-- Time to pick an expensive handcuff lock in minutes -/
def expensive_time : ℕ := 15

/-- Represents the handcuffs on a person -/
structure Handcuffs :=
  (left_hand right_hand left_ankle right_ankle : Bool)

/-- True if the handcuff is expensive, False if it's cheap -/
def friend1 : Handcuffs := ⟨true, true, false, false⟩
def friend2 : Handcuffs := ⟨false, false, true, true⟩
def friend3 : Handcuffs := ⟨true, false, false, false⟩
def friend4 : Handcuffs := ⟨true, true, true, true⟩
def friend5 : Handcuffs := ⟨false, true, true, true⟩
def friend6 : Handcuffs := ⟨false, false, false, false⟩

/-- Calculate the time needed to free a friend -/
def free_time (h : Handcuffs) : ℕ :=
  (if h.left_hand then expensive_time else cheap_time) +
  (if h.right_hand then expensive_time else cheap_time) +
  (if h.left_ankle then expensive_time else cheap_time) +
  (if h.right_ankle then expensive_time else cheap_time)

/-- The total time to free all friends -/
def total_time : ℕ :=
  free_time friend1 + free_time friend2 + free_time friend3 +
  free_time friend4 + free_time friend5 + free_time friend6

theorem free_all_friends_time :
  total_time = 300 := by sorry

end NUMINAMATH_CALUDE_free_all_friends_time_l4165_416543


namespace NUMINAMATH_CALUDE_largest_amount_l4165_416525

theorem largest_amount (milk : Rat) (cider : Rat) (orange_juice : Rat)
  (h_milk : milk = 3/8)
  (h_cider : cider = 7/10)
  (h_orange_juice : orange_juice = 11/15) :
  max milk (max cider orange_juice) = orange_juice :=
by sorry

end NUMINAMATH_CALUDE_largest_amount_l4165_416525


namespace NUMINAMATH_CALUDE_magic_king_episodes_l4165_416528

/-- The total number of episodes in the Magic King show -/
def total_episodes : ℕ :=
  let first_three_seasons := 3 * 20
  let seasons_four_to_eight := 5 * 25
  let seasons_nine_to_eleven := 3 * 30
  let last_three_seasons := 3 * 15
  let holiday_specials := 5
  first_three_seasons + seasons_four_to_eight + seasons_nine_to_eleven + last_three_seasons + holiday_specials

/-- Theorem stating that the total number of episodes is 325 -/
theorem magic_king_episodes : total_episodes = 325 := by
  sorry

end NUMINAMATH_CALUDE_magic_king_episodes_l4165_416528


namespace NUMINAMATH_CALUDE_prime_divisibility_l4165_416534

theorem prime_divisibility (p m n : ℕ) : 
  Prime p → 
  p > 2 → 
  m > 1 → 
  n > 0 → 
  Prime ((m^(p*n) - 1) / (m^n - 1)) → 
  (p * n) ∣ ((p - 1)^n + 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_divisibility_l4165_416534


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l4165_416540

/-- Represents an ellipse -/
structure Ellipse where
  center : ℝ × ℝ
  passes_through : ℝ × ℝ
  a_b_ratio : ℝ

/-- Checks if the given equation represents the standard form of the ellipse -/
def is_standard_equation (e : Ellipse) (eq : ℝ → ℝ → Bool) : Prop :=
  (eq 3 0 = true) ∧ 
  (∀ x y, eq x y ↔ (x^2 / 9 + y^2 = 1 ∨ y^2 / 81 + x^2 / 9 = 1))

/-- Theorem: Given the conditions, the ellipse has one of the two standard equations -/
theorem ellipse_standard_equation (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.passes_through = (3, 0))
  (h3 : e.a_b_ratio = 3) :
  ∃ eq : ℝ → ℝ → Bool, is_standard_equation e eq := by
  sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l4165_416540


namespace NUMINAMATH_CALUDE_sequence_constant_iff_perfect_square_l4165_416593

/-- S(n) is defined as n minus the largest perfect square not exceeding n -/
def S (n : ℕ) : ℕ :=
  n - (Nat.sqrt n) ^ 2

/-- The sequence a_k is defined recursively -/
def a (A : ℕ) : ℕ → ℕ
  | 0 => A
  | k + 1 => a A k + S (a A k)

/-- A positive integer A makes the sequence eventually constant
    if and only if A is a perfect square -/
theorem sequence_constant_iff_perfect_square (A : ℕ) (h : A > 0) :
  (∃ N : ℕ, ∀ k ≥ N, a A k = a A N) ↔ ∃ m : ℕ, A = m^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_constant_iff_perfect_square_l4165_416593


namespace NUMINAMATH_CALUDE_ernesto_extra_distance_l4165_416575

/-- Given that Renaldo drove 15 kilometers, Ernesto drove some kilometers more than one-third of Renaldo's distance, and the total distance driven by both men is 27 kilometers, prove that Ernesto drove 7 kilometers more than one-third of Renaldo's distance. -/
theorem ernesto_extra_distance (renaldo_distance : ℝ) (ernesto_distance : ℝ) (total_distance : ℝ)
  (h1 : renaldo_distance = 15)
  (h2 : ernesto_distance > (1/3) * renaldo_distance)
  (h3 : total_distance = renaldo_distance + ernesto_distance)
  (h4 : total_distance = 27) :
  ernesto_distance - (1/3) * renaldo_distance = 7 := by
  sorry

end NUMINAMATH_CALUDE_ernesto_extra_distance_l4165_416575


namespace NUMINAMATH_CALUDE_gold_bars_worth_l4165_416515

/-- Calculate the total worth of gold bars in a safe -/
theorem gold_bars_worth (rows : ℕ) (bars_per_row : ℕ) (worth_per_bar : ℕ) :
  rows = 4 →
  bars_per_row = 20 →
  worth_per_bar = 20000 →
  rows * bars_per_row * worth_per_bar = 1600000 := by
  sorry

#check gold_bars_worth

end NUMINAMATH_CALUDE_gold_bars_worth_l4165_416515


namespace NUMINAMATH_CALUDE_scientific_notation_of_1_3_million_l4165_416509

theorem scientific_notation_of_1_3_million :
  1300000 = 1.3 * (10 : ℝ)^6 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1_3_million_l4165_416509


namespace NUMINAMATH_CALUDE_polynomial_expansion_l4165_416512

theorem polynomial_expansion :
  (fun z : ℝ => 3 * z^3 + 4 * z^2 - 8 * z - 5) *
  (fun z : ℝ => 2 * z^4 - 3 * z^2 + 1) =
  (fun z : ℝ => 6 * z^7 + 12 * z^6 - 25 * z^5 - 20 * z^4 + 34 * z^2 - 8 * z - 5) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l4165_416512


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_set_l4165_416563

theorem quadratic_equation_solution_set :
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x + 2
  {x : ℝ | f x = 0} = {1, 2} := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_set_l4165_416563


namespace NUMINAMATH_CALUDE_convex_polygon_in_rectangle_l4165_416523

/-- A convex polygon in the plane -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  convex : Convex ℝ (convexHull ℝ vertices)
  finite : Finite vertices

/-- The area of a polygon -/
def area (p : ConvexPolygon) : ℝ := sorry

/-- A rectangle in the plane -/
structure Rectangle where
  lower_left : ℝ × ℝ
  upper_right : ℝ × ℝ
  valid : lower_left.1 < upper_right.1 ∧ lower_left.2 < upper_right.2

/-- The area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  (r.upper_right.1 - r.lower_left.1) * (r.upper_right.2 - r.lower_left.2)

/-- A polygon is contained in a rectangle -/
def contained (p : ConvexPolygon) (r : Rectangle) : Prop := sorry

theorem convex_polygon_in_rectangle :
  ∀ (p : ConvexPolygon), area p = 1 →
  ∃ (r : Rectangle), contained p r ∧ rectangleArea r ≤ 2 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_in_rectangle_l4165_416523


namespace NUMINAMATH_CALUDE_construct_square_l4165_416587

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Four points lying on the sides of a quadrilateral -/
structure SidePoints where
  K : Point  -- on side AB
  P : Point  -- on side BC
  R : Point  -- on side CD
  Q : Point  -- on side AD

/-- Predicate to check if a point lies on a line segment -/
def liesBetween (P Q R : Point) : Prop := sorry

/-- Predicate to check if two line segments are perpendicular -/
def perpendicular (P Q R S : Point) : Prop := sorry

/-- Predicate to check if two line segments have equal length -/
def equalLength (P Q R S : Point) : Prop := sorry

/-- Main theorem: Given four points on the sides of a quadrilateral, 
    if certain conditions are met, then the quadrilateral is a square -/
theorem construct_square (ABCD : Quadrilateral) (sides : SidePoints) : 
  liesBetween ABCD.A sides.K ABCD.B ∧
  liesBetween ABCD.B sides.P ABCD.C ∧
  liesBetween ABCD.C sides.R ABCD.D ∧
  liesBetween ABCD.D sides.Q ABCD.A ∧
  perpendicular ABCD.A ABCD.B ABCD.B ABCD.C ∧
  perpendicular ABCD.B ABCD.C ABCD.C ABCD.D ∧
  perpendicular ABCD.C ABCD.D ABCD.D ABCD.A ∧
  perpendicular ABCD.D ABCD.A ABCD.A ABCD.B ∧
  equalLength ABCD.A ABCD.B ABCD.B ABCD.C ∧
  equalLength ABCD.B ABCD.C ABCD.C ABCD.D ∧
  equalLength ABCD.C ABCD.D ABCD.D ABCD.A →
  -- Conclusion: ABCD is a square
  -- (We don't provide a formal definition of a square here, 
  -- as it would typically be defined elsewhere in a real geometry library)
  True := by
  sorry

end NUMINAMATH_CALUDE_construct_square_l4165_416587


namespace NUMINAMATH_CALUDE_smallest_total_blocks_smallest_total_blocks_exist_l4165_416548

/-- Represents the dimensions of a cubic block -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cubic pedestal -/
structure Pedestal where
  sideLength : ℕ

/-- Represents a square foundation -/
structure Foundation where
  sideLength : ℕ
  thickness : ℕ

/-- Calculates the volume of a pedestal in terms of blocks -/
def pedestalVolume (p : Pedestal) : ℕ :=
  p.sideLength ^ 3

/-- Calculates the volume of a foundation in terms of blocks -/
def foundationVolume (f : Foundation) : ℕ :=
  f.sideLength ^ 2 * f.thickness

theorem smallest_total_blocks : ℕ × ℕ → Prop
  | (pedestal_side, foundation_side) =>
    let block : Block := ⟨1, 1, 1⟩
    let pedestal : Pedestal := ⟨pedestal_side⟩
    let foundation : Foundation := ⟨foundation_side, 1⟩
    (pedestalVolume pedestal = foundationVolume foundation) ∧
    (foundation_side = pedestal_side ^ (3/2)) ∧
    (pedestalVolume pedestal + foundationVolume foundation = 128) ∧
    ∀ (p : Pedestal) (f : Foundation),
      (pedestalVolume p = foundationVolume f) →
      (f.sideLength = p.sideLength ^ (3/2)) →
      (pedestalVolume p + foundationVolume f ≥ 128)

theorem smallest_total_blocks_exist :
  ∃ (pedestal_side foundation_side : ℕ),
    smallest_total_blocks (pedestal_side, foundation_side) :=
  sorry

end NUMINAMATH_CALUDE_smallest_total_blocks_smallest_total_blocks_exist_l4165_416548


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4165_416595

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) (x₀ y₁ y₂ y₃ : ℝ) :
  a ≠ 0 →
  f a b c (-2) = 0 →
  x₀ > 1 →
  f a b c x₀ = 0 →
  (a + b + c) * (4 * a + 2 * b + c) < 0 →
  ∃ y, y < 0 ∧ f a b c 0 = y →
  f a b c (-1) = y₁ →
  f a b c (-Real.sqrt 2 / 2) = y₂ →
  f a b c 1 = y₃ →
  y₃ > y₁ ∧ y₁ > y₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4165_416595


namespace NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_l4165_416574

theorem quadratic_equation_rational_solutions : 
  ∃ (c₁ c₂ : ℕ+), 
    (c₁ ≠ c₂) ∧
    (∀ c : ℕ+, (∃ x : ℚ, 3 * x^2 + 7 * x + c.val = 0) ↔ (c = c₁ ∨ c = c₂)) ∧
    (c₁.val * c₂.val = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_l4165_416574


namespace NUMINAMATH_CALUDE_probability_of_prime_on_die_l4165_416590

/-- A standard die with six faces -/
def StandardDie := Finset.range 6

/-- The set of prime numbers on a standard die -/
def PrimeNumbersOnDie : Finset Nat := {2, 3, 5}

/-- The probability of rolling a prime number on a standard die -/
def ProbabilityOfPrime : ℚ := (PrimeNumbersOnDie.card : ℚ) / (StandardDie.card : ℚ)

/-- The given probability expression -/
def GivenProbability (a : ℕ) : ℚ := (a : ℚ) / 72

theorem probability_of_prime_on_die (a : ℕ) :
  GivenProbability a = ProbabilityOfPrime → a = 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_prime_on_die_l4165_416590


namespace NUMINAMATH_CALUDE_nineteenth_term_is_zero_l4165_416585

/-- A sequence with specific properties -/
def special_sequence (a : ℕ → ℝ) : Prop :=
  a 3 = 2 ∧ 
  a 7 = 1 ∧ 
  ∃ d : ℝ, ∀ n : ℕ, 1 / (a (n + 1) + 1) - 1 / (a n + 1) = d

/-- Theorem stating that for a special sequence, the 19th term is 0 -/
theorem nineteenth_term_is_zero (a : ℕ → ℝ) (h : special_sequence a) : 
  a 19 = 0 := by sorry

end NUMINAMATH_CALUDE_nineteenth_term_is_zero_l4165_416585


namespace NUMINAMATH_CALUDE_y1_value_l4165_416502

theorem y1_value (y1 y2 y3 : ℝ) 
  (h_order : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h_eq : (1 - y1)^2 + 2*(y1 - y2)^2 + 3*(y2 - y3)^2 + 4*y3^2 = 1/2) :
  y1 = (3 * Real.sqrt 6 - 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_y1_value_l4165_416502


namespace NUMINAMATH_CALUDE_casas_alvero_prime_l4165_416572

/-- A polynomial with rational coefficients -/
def RationalPolynomial : Type := ℚ → ℚ

/-- The degree of a polynomial -/
def degree (p : RationalPolynomial) : ℕ := sorry

/-- The kth derivative of a polynomial -/
def derivative (p : RationalPolynomial) (k : ℕ) : RationalPolynomial := sorry

/-- Checks if a rational number is a root of a polynomial -/
def is_root (p : RationalPolynomial) (r : ℚ) : Prop := p r = 0

theorem casas_alvero_prime (p : RationalPolynomial) (d : ℕ) :
  degree p = d →
  Nat.Prime d →
  (∀ k : ℕ, 1 ≤ k → k ≤ d - 1 →
    ∃ r : ℚ, is_root p r ∧ is_root (derivative p k) r) →
  ∃ a b c : ℚ, ∀ x : ℚ, p x = c * (a * x + b) ^ d :=
sorry

end NUMINAMATH_CALUDE_casas_alvero_prime_l4165_416572


namespace NUMINAMATH_CALUDE_value_subtracted_l4165_416550

theorem value_subtracted (n : ℝ) (x : ℝ) : 
  (2 * n + 20 = 8 * n - x) → 
  (n = 4) → 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_value_subtracted_l4165_416550
