import Mathlib

namespace NUMINAMATH_CALUDE_total_messages_equals_680_l2369_236974

/-- The total number of messages sent by Alina and Lucia over three days -/
def total_messages (lucia_day1 : ℕ) (alina_difference : ℕ) : ℕ :=
  let alina_day1 := lucia_day1 - alina_difference
  let lucia_day2 := lucia_day1 / 3
  let alina_day2 := 2 * alina_day1
  let lucia_day3 := lucia_day1
  let alina_day3 := alina_day1
  lucia_day1 + lucia_day2 + lucia_day3 + alina_day1 + alina_day2 + alina_day3

/-- Theorem stating that the total number of messages sent over three days is 680 -/
theorem total_messages_equals_680 :
  total_messages 120 20 = 680 := by
  sorry

end NUMINAMATH_CALUDE_total_messages_equals_680_l2369_236974


namespace NUMINAMATH_CALUDE_integral_tan_sin_l2369_236948

open Real MeasureTheory

theorem integral_tan_sin : ∫ (x : ℝ) in Real.arcsin (2 / Real.sqrt 5)..Real.arcsin (3 / Real.sqrt 10), 
  (2 * Real.tan x + 5) / ((5 - Real.tan x) * Real.sin (2 * x)) = 2 * Real.log (3 / 2) := by sorry

end NUMINAMATH_CALUDE_integral_tan_sin_l2369_236948


namespace NUMINAMATH_CALUDE_sum_product_equality_l2369_236991

theorem sum_product_equality : 3 * 12 + 3 * 13 + 3 * 16 + 11 = 134 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l2369_236991


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2369_236968

theorem inequality_solution_range (k : ℝ) : 
  (k ≠ 0 ∧ k^2 * 1^2 - 6*k*1 + 8 ≥ 0) →
  k ∈ (Set.Ioi 4 : Set ℝ) ∪ (Set.Icc 0 2 : Set ℝ) ∪ (Set.Iio 0 : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2369_236968


namespace NUMINAMATH_CALUDE_derivative_of_x_exp_x_l2369_236945

theorem derivative_of_x_exp_x :
  let f : ℝ → ℝ := λ x ↦ x * Real.exp x
  deriv f = λ x ↦ (1 + x) * Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_x_exp_x_l2369_236945


namespace NUMINAMATH_CALUDE_modular_inverse_57_mod_59_l2369_236999

theorem modular_inverse_57_mod_59 : ∃ x : ℕ, x < 59 ∧ (57 * x) % 59 = 1 :=
by
  use 29
  sorry

end NUMINAMATH_CALUDE_modular_inverse_57_mod_59_l2369_236999


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_90_l2369_236902

theorem distinct_prime_factors_of_90 : Nat.card (Nat.factors 90).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_90_l2369_236902


namespace NUMINAMATH_CALUDE_initial_men_count_l2369_236988

/-- The initial number of men in a group where:
  1. The average age increases by 3 years when two women replace two men.
  2. The two men being replaced are 18 and 22 years old.
  3. The average age of the women is 30.5 years. -/
def initial_number_of_men : ℕ := 7

/-- The average age increase when women replace men -/
def age_increase : ℝ := 3

/-- The age of the first man being replaced -/
def first_man_age : ℕ := 18

/-- The age of the second man being replaced -/
def second_man_age : ℕ := 22

/-- The average age of the women -/
def women_average_age : ℝ := 30.5

theorem initial_men_count : 
  ∃ (A : ℝ), 
    (initial_number_of_men : ℝ) * (A + age_increase) = 
    initial_number_of_men * A - (first_man_age + second_man_age : ℝ) + 2 * women_average_age :=
by sorry

end NUMINAMATH_CALUDE_initial_men_count_l2369_236988


namespace NUMINAMATH_CALUDE_function_minimum_l2369_236973

def f (x : ℝ) : ℝ := x^2 - 8*x + 5

theorem function_minimum :
  ∃ (x_min : ℝ), 
    (∀ x, f x ≥ f x_min) ∧ 
    x_min = 4 ∧ 
    f x_min = -11 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_l2369_236973


namespace NUMINAMATH_CALUDE_total_gray_trees_l2369_236937

/-- Represents a rectangle with trees -/
structure TreeRectangle where
  totalTrees : ℕ
  whiteTrees : ℕ
  grayTrees : ℕ
  sum_eq : totalTrees = whiteTrees + grayTrees

/-- The problem setup -/
def dronePhotos (rect1 rect2 rect3 : TreeRectangle) : Prop :=
  rect1.totalTrees = rect2.totalTrees ∧
  rect1.totalTrees = rect3.totalTrees ∧
  rect1.totalTrees = 100 ∧
  rect1.whiteTrees = 82 ∧
  rect2.whiteTrees = 82

/-- The theorem to prove -/
theorem total_gray_trees (rect1 rect2 rect3 : TreeRectangle) 
  (h : dronePhotos rect1 rect2 rect3) : 
  rect1.grayTrees + rect2.grayTrees = 26 :=
by sorry

end NUMINAMATH_CALUDE_total_gray_trees_l2369_236937


namespace NUMINAMATH_CALUDE_complex_number_location_l2369_236970

theorem complex_number_location :
  let z : ℂ := 1 / (3 + Complex.I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l2369_236970


namespace NUMINAMATH_CALUDE_double_inequality_solution_l2369_236941

theorem double_inequality_solution (x : ℝ) : 
  -2 < (x^2 - 16*x + 11) / (x^2 - 3*x + 4) ∧ 
  (x^2 - 16*x + 11) / (x^2 - 3*x + 4) < 2 ↔ 
  1 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l2369_236941


namespace NUMINAMATH_CALUDE_sophie_widget_production_l2369_236922

/-- Sophie's widget production problem -/
theorem sophie_widget_production 
  (w t : ℕ) -- w: widgets per hour, t: hours worked on Wednesday
  (h1 : w = 3 * t) -- condition that w = 3t
  : w * t - (w + 5) * (t - 3) = 4 * t + 15 := by
  sorry

end NUMINAMATH_CALUDE_sophie_widget_production_l2369_236922


namespace NUMINAMATH_CALUDE_shelf_capacity_l2369_236997

/-- The number of CDs that a single rack can hold -/
def cds_per_rack : ℕ := 8

/-- The number of racks that can fit on a shelf -/
def racks_per_shelf : ℕ := 4

/-- The total number of CDs that can fit on a shelf -/
def total_cds : ℕ := cds_per_rack * racks_per_shelf

theorem shelf_capacity : total_cds = 32 := by
  sorry

end NUMINAMATH_CALUDE_shelf_capacity_l2369_236997


namespace NUMINAMATH_CALUDE_smallest_surface_area_is_cube_l2369_236983

-- Define a rectangular parallelepiped
structure Parallelepiped where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z

-- Define the volume of a parallelepiped
def volume (p : Parallelepiped) : ℝ := p.x * p.y * p.z

-- Define the surface area of a parallelepiped
def surfaceArea (p : Parallelepiped) : ℝ := 2 * (p.x * p.y + p.x * p.z + p.y * p.z)

-- State the theorem
theorem smallest_surface_area_is_cube (V : ℝ) (hV : 0 < V) :
  ∃ (p : Parallelepiped), volume p = V ∧
    ∀ (q : Parallelepiped), volume q = V → surfaceArea p ≤ surfaceArea q ∧
      (surfaceArea p = surfaceArea q → p.x = p.y ∧ p.y = p.z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_surface_area_is_cube_l2369_236983


namespace NUMINAMATH_CALUDE_vector_expression_equality_l2369_236939

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

theorem vector_expression_equality (a b : V) :
  (1 / 2 : ℝ) • ((2 : ℝ) • a - (4 : ℝ) • b) + (2 : ℝ) • b = a := by sorry

end NUMINAMATH_CALUDE_vector_expression_equality_l2369_236939


namespace NUMINAMATH_CALUDE_max_cars_theorem_l2369_236972

/-- Represents the maximum number of cars that can pass a point on a highway in 30 minutes -/
def M : ℕ := 6000

/-- Theorem stating the maximum number of cars and its relation to M/10 -/
theorem max_cars_theorem :
  (∀ (car_length : ℝ) (time : ℝ),
    car_length = 5 ∧ 
    time = 30 ∧ 
    (∀ (speed : ℝ) (distance : ℝ),
      distance = car_length * (speed / 10))) →
  M = 6000 ∧ M / 10 = 600 := by
  sorry

#check max_cars_theorem

end NUMINAMATH_CALUDE_max_cars_theorem_l2369_236972


namespace NUMINAMATH_CALUDE_fruit_basket_count_l2369_236917

/-- Represents the number of apples available -/
def num_apples : ℕ := 6

/-- Represents the number of oranges available -/
def num_oranges : ℕ := 8

/-- Represents the minimum number of apples required in each basket -/
def min_apples : ℕ := 2

/-- Calculates the number of possible fruit baskets -/
def num_fruit_baskets : ℕ :=
  (num_apples - min_apples + 1) * (num_oranges + 1)

/-- Theorem stating the number of possible fruit baskets -/
theorem fruit_basket_count :
  num_fruit_baskets = 45 := by sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l2369_236917


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l2369_236936

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 90 ∧ x - y = 10 → x * y = 2000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l2369_236936


namespace NUMINAMATH_CALUDE_next_term_correct_l2369_236933

/-- Represents a digit (0-9) -/
inductive Digit : Type
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Represents a sequence of digits -/
def Sequence := List Digit

/-- Generates the next term in the sequence based on the current term -/
def nextTerm (current : Sequence) : Sequence :=
  sorry

/-- The starting term of the sequence -/
def startTerm : Sequence :=
  [Digit.one]

/-- Generates the nth term of the sequence -/
def nthTerm (n : Nat) : Sequence :=
  sorry

/-- Converts a Sequence to a list of natural numbers -/
def sequenceToNatList (s : Sequence) : List Nat :=
  sorry

theorem next_term_correct :
  sequenceToNatList (nextTerm [Digit.one, Digit.one, Digit.four, Digit.two, Digit.one, Digit.three]) =
  [3, 1, 1, 2, 1, 3, 1, 4] :=
sorry

end NUMINAMATH_CALUDE_next_term_correct_l2369_236933


namespace NUMINAMATH_CALUDE_beijing_shanghai_train_time_l2369_236912

/-- The function relationship between total travel time and average speed for a train on the Beijing-Shanghai railway line -/
theorem beijing_shanghai_train_time (t : ℝ) (v : ℝ) (h : v ≠ 0) : 
  (t = 1463 / v) ↔ (1463 = t * v) :=
by sorry

end NUMINAMATH_CALUDE_beijing_shanghai_train_time_l2369_236912


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2369_236969

theorem complex_equation_sum (a b : ℝ) :
  (a - 2 * Complex.I) * Complex.I = b - Complex.I →
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2369_236969


namespace NUMINAMATH_CALUDE_function_composition_equality_l2369_236925

theorem function_composition_equality (a b c d : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x + b
  let g : ℝ → ℝ := λ x ↦ c * x^2 + d
  (∃ x : ℝ, f (g x) = g (f x)) ↔ (c = 0 ∨ a * b = 0) ∧ a * d = c * b^2 + d - b :=
by sorry

end NUMINAMATH_CALUDE_function_composition_equality_l2369_236925


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2369_236977

/-- A complex-valued function satisfying the given functional equation is constant and equal to 1. -/
theorem functional_equation_solution (f : ℂ → ℂ) : 
  (∀ z : ℂ, f z + z * f (1 - z) = 1 + z) → 
  (∀ z : ℂ, f z = 1) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2369_236977


namespace NUMINAMATH_CALUDE_dilution_proof_l2369_236953

/-- Proves that adding 7 ounces of water to 12 ounces of a 40% alcohol solution results in a 25% alcohol solution -/
theorem dilution_proof (original_volume : ℝ) (original_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : 
  original_volume = 12 →
  original_concentration = 0.4 →
  target_concentration = 0.25 →
  water_added = 7 →
  (original_volume * original_concentration) / (original_volume + water_added) = target_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_dilution_proof_l2369_236953


namespace NUMINAMATH_CALUDE_money_left_l2369_236924

def initial_amount : ℕ := 43
def total_spent : ℕ := 38

theorem money_left : initial_amount - total_spent = 5 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l2369_236924


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_opposite_of_negative_one_over_2023_l2369_236960

theorem opposite_of_negative_fraction (n : ℕ) (n_pos : n > 0) :
  ((-1 : ℚ) / n) + (1 : ℚ) / n = 0 :=
by sorry

theorem opposite_of_negative_one_over_2023 :
  ((-1 : ℚ) / 2023) + (1 : ℚ) / 2023 = 0 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_opposite_of_negative_one_over_2023_l2369_236960


namespace NUMINAMATH_CALUDE_both_correct_calculation_l2369_236905

/-- Represents a class test scenario -/
structure ClassTest where
  total : ℕ
  correct1 : ℕ
  correct2 : ℕ
  absent : ℕ

/-- Calculates the number of students who answered both questions correctly -/
def bothCorrect (test : ClassTest) : ℕ :=
  test.correct1 + test.correct2 - (test.total - test.absent)

/-- Theorem stating the number of students who answered both questions correctly -/
theorem both_correct_calculation (test : ClassTest) 
  (h1 : test.total = 25)
  (h2 : test.correct1 = 22)
  (h3 : test.correct2 = 20)
  (h4 : test.absent = 3) :
  bothCorrect test = 17 := by
  sorry

end NUMINAMATH_CALUDE_both_correct_calculation_l2369_236905


namespace NUMINAMATH_CALUDE_original_number_problem_l2369_236904

theorem original_number_problem (x : ℝ) : 3 * (2 * x + 5) = 111 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l2369_236904


namespace NUMINAMATH_CALUDE_technician_permanent_percentage_l2369_236978

def factory_workforce (total_workers : ℝ) : Prop :=
  let technicians := 0.8 * total_workers
  let non_technicians := 0.2 * total_workers
  let permanent_non_technicians := 0.2 * non_technicians
  let temporary_workers := 0.68 * total_workers
  ∃ (permanent_technicians : ℝ),
    permanent_technicians + permanent_non_technicians = total_workers - temporary_workers ∧
    permanent_technicians / technicians = 0.35

theorem technician_permanent_percentage :
  ∀ (total_workers : ℝ), total_workers > 0 → factory_workforce total_workers :=
by sorry

end NUMINAMATH_CALUDE_technician_permanent_percentage_l2369_236978


namespace NUMINAMATH_CALUDE_isosceles_triangle_third_side_l2369_236935

theorem isosceles_triangle_third_side 
  (a b c : ℝ) 
  (h_isosceles : (a = b ∧ c = 5) ∨ (a = c ∧ b = 5) ∨ (b = c ∧ a = 5)) 
  (h_side : a = 2 ∨ b = 2 ∨ c = 2) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a = 5 ∨ b = 5 ∨ c = 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_third_side_l2369_236935


namespace NUMINAMATH_CALUDE_veronica_extra_stairs_l2369_236934

/-- Given that Samir climbed 318 stairs and together with Veronica they climbed 495 stairs,
    prove that Veronica climbed 18 stairs more than half of Samir's amount. -/
theorem veronica_extra_stairs (samir_stairs : ℕ) (total_stairs : ℕ) 
    (h1 : samir_stairs = 318)
    (h2 : total_stairs = 495)
    (h3 : ∃ (veronica_stairs : ℕ), veronica_stairs > samir_stairs / 2 ∧ 
                                    veronica_stairs + samir_stairs = total_stairs) : 
  ∃ (veronica_stairs : ℕ), veronica_stairs = samir_stairs / 2 + 18 := by
  sorry

end NUMINAMATH_CALUDE_veronica_extra_stairs_l2369_236934


namespace NUMINAMATH_CALUDE_tangent_line_minimum_value_l2369_236990

theorem tangent_line_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : ∃ x y : ℝ, y = x - 2*a ∧ y = Real.log (x + b) ∧ 
    (Real.exp y) * (1 / (x + b)) = 1) :
  (1/a + 2/b) ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_value_l2369_236990


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2369_236946

-- Define the floor function
def floor (x : ℚ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_equation_solution :
  ∀ x : ℚ, floor (5 * x - 2) = 3 * x.num + x.den → x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2369_236946


namespace NUMINAMATH_CALUDE_min_toothpicks_for_five_squares_l2369_236928

/-- A square formed by toothpicks -/
structure ToothpickSquare where
  side_length : ℝ
  toothpicks_per_square : ℕ

/-- The arrangement of multiple toothpick squares -/
structure SquareArrangement where
  square : ToothpickSquare
  num_squares : ℕ

/-- The number of toothpicks needed for an arrangement of squares -/
def toothpicks_needed (arrangement : SquareArrangement) : ℕ :=
  sorry

/-- The theorem stating the minimum number of toothpicks needed -/
theorem min_toothpicks_for_five_squares
  (square : ToothpickSquare)
  (arrangement : SquareArrangement)
  (h1 : square.side_length = 6)
  (h2 : square.toothpicks_per_square = 4)
  (h3 : arrangement.square = square)
  (h4 : arrangement.num_squares = 5) :
  toothpicks_needed arrangement = 15 :=
sorry

end NUMINAMATH_CALUDE_min_toothpicks_for_five_squares_l2369_236928


namespace NUMINAMATH_CALUDE_line_equation_proof_l2369_236932

/-- Given a line defined by (-1, 4) · ((x, y) - (3, -5)) = 0, 
    prove that its equation in the form y = mx + b has m = 1/4 and b = -23/4 -/
theorem line_equation_proof (x y : ℝ) : 
  (-1 : ℝ) * (x - 3) + 4 * (y + 5) = 0 → 
  ∃ (m b : ℝ), y = m * x + b ∧ m = (1 : ℝ) / 4 ∧ b = -(23 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2369_236932


namespace NUMINAMATH_CALUDE_expression_evaluation_l2369_236958

theorem expression_evaluation (x y : ℝ) (hx : x = -2) (hy : y = 1/3) :
  (2*y + 3*x^2) - (x^2 - y) - x^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2369_236958


namespace NUMINAMATH_CALUDE_special_quadratic_roots_nonnegative_l2369_236952

/-- A quadratic polynomial with two distinct roots satisfying f(x^2 + y^2) ≥ f(2xy) for all x and y -/
structure SpecialQuadratic where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  has_two_distinct_roots : ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ f r₁ = 0 ∧ f r₂ = 0
  special_property : ∀ x y : ℝ, f (x^2 + y^2) ≥ f (2*x*y)

/-- The roots of a SpecialQuadratic are non-negative -/
theorem special_quadratic_roots_nonnegative (sq : SpecialQuadratic) :
  ∃ r₁ r₂ : ℝ, r₁ ≥ 0 ∧ r₂ ≥ 0 ∧ sq.f r₁ = 0 ∧ sq.f r₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_quadratic_roots_nonnegative_l2369_236952


namespace NUMINAMATH_CALUDE_inequality_for_increasing_function_l2369_236931

/-- An increasing function on the real line. -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

/-- Theorem: Given an increasing function f on ℝ and real numbers a and b
    such that a + b ≤ 0, the inequality f(a) + f(b) ≤ f(-a) + f(-b) holds. -/
theorem inequality_for_increasing_function
  (f : ℝ → ℝ) (hf : IncreasingFunction f) (a b : ℝ) (hab : a + b ≤ 0) :
  f a + f b ≤ f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_increasing_function_l2369_236931


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2369_236976

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 + 3 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 3 < 2*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2369_236976


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2369_236980

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {4, 5}

theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2369_236980


namespace NUMINAMATH_CALUDE_power_product_evaluation_l2369_236942

theorem power_product_evaluation :
  let a : ℕ := 3
  a^2 * a^5 = 2187 :=
by sorry

end NUMINAMATH_CALUDE_power_product_evaluation_l2369_236942


namespace NUMINAMATH_CALUDE_hexagon_perimeter_is_42_l2369_236985

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The length of ribbon required for each side of the display board -/
def ribbon_length_per_side : ℝ := 7

/-- The perimeter of a hexagonal display board -/
def hexagon_perimeter : ℝ := hexagon_sides * ribbon_length_per_side

/-- Theorem: The perimeter of the hexagonal display board is 42 cm -/
theorem hexagon_perimeter_is_42 : hexagon_perimeter = 42 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_is_42_l2369_236985


namespace NUMINAMATH_CALUDE_range_of_a_l2369_236950

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ |x^2 - 2*x|}

-- Define set B
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + a ≤ 0}

-- Theorem statement
theorem range_of_a (a : ℝ) : A ∩ B a = B a → a ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2369_236950


namespace NUMINAMATH_CALUDE_bc_values_l2369_236955

theorem bc_values (a b c : ℝ) 
  (sum_eq : a + b + c = 100)
  (prod_sum_eq : a * b + b * c + c * a = 20)
  (mixed_prod_eq : (a + b) * (a + c) = 24) :
  b * c = -176 ∨ b * c = 224 :=
by sorry

end NUMINAMATH_CALUDE_bc_values_l2369_236955


namespace NUMINAMATH_CALUDE_tan_sixty_degrees_l2369_236994

theorem tan_sixty_degrees : Real.tan (60 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sixty_degrees_l2369_236994


namespace NUMINAMATH_CALUDE_max_a_for_three_integer_solutions_l2369_236918

theorem max_a_for_three_integer_solutions : 
  ∃ (a : ℝ), 
    (∀ x : ℤ, (-1/3 : ℝ) * (x : ℝ) > 2/3 - (x : ℝ) ∧ 
               (1/2 : ℝ) * (x : ℝ) - 1 < (1/2 : ℝ) * (a - 2)) →
    (∃! (x₁ x₂ x₃ : ℤ), 
      ((-1/3 : ℝ) * (x₁ : ℝ) > 2/3 - (x₁ : ℝ) ∧ 
       (1/2 : ℝ) * (x₁ : ℝ) - 1 < (1/2 : ℝ) * (a - 2)) ∧
      ((-1/3 : ℝ) * (x₂ : ℝ) > 2/3 - (x₂ : ℝ) ∧ 
       (1/2 : ℝ) * (x₂ : ℝ) - 1 < (1/2 : ℝ) * (a - 2)) ∧
      ((-1/3 : ℝ) * (x₃ : ℝ) > 2/3 - (x₃ : ℝ) ∧ 
       (1/2 : ℝ) * (x₃ : ℝ) - 1 < (1/2 : ℝ) * (a - 2)) ∧
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
    a = 5 ∧ ∀ b > a, 
      ¬(∃! (x₁ x₂ x₃ : ℤ), 
        ((-1/3 : ℝ) * (x₁ : ℝ) > 2/3 - (x₁ : ℝ) ∧ 
         (1/2 : ℝ) * (x₁ : ℝ) - 1 < (1/2 : ℝ) * (b - 2)) ∧
        ((-1/3 : ℝ) * (x₂ : ℝ) > 2/3 - (x₂ : ℝ) ∧ 
         (1/2 : ℝ) * (x₂ : ℝ) - 1 < (1/2 : ℝ) * (b - 2)) ∧
        ((-1/3 : ℝ) * (x₃ : ℝ) > 2/3 - (x₃ : ℝ) ∧ 
         (1/2 : ℝ) * (x₃ : ℝ) - 1 < (1/2 : ℝ) * (b - 2)) ∧
        x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_three_integer_solutions_l2369_236918


namespace NUMINAMATH_CALUDE_age_sum_problem_l2369_236981

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 128 → a + b + c = 18 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_problem_l2369_236981


namespace NUMINAMATH_CALUDE_min_marks_group_a_l2369_236926

/-- Represents the number of marks for each question in a group -/
structure GroupMarks where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the number of questions in each group -/
structure GroupQuestions where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The examination setup -/
structure Examination where
  marks : GroupMarks
  questions : GroupQuestions
  total_questions : ℕ
  total_marks : ℕ

/-- Conditions for the examination -/
def valid_examination (e : Examination) : Prop :=
  e.total_questions = 100 ∧
  e.questions.a + e.questions.b + e.questions.c = e.total_questions ∧
  e.questions.b = 23 ∧
  e.questions.c = 1 ∧
  e.marks.b = 2 ∧
  e.marks.c = 3 ∧
  e.total_marks = e.questions.a * e.marks.a + e.questions.b * e.marks.b + e.questions.c * e.marks.c ∧
  e.questions.a * e.marks.a ≥ (60 * e.total_marks) / 100

theorem min_marks_group_a (e : Examination) (h : valid_examination e) :
  e.marks.a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_marks_group_a_l2369_236926


namespace NUMINAMATH_CALUDE_expression_simplification_l2369_236930

theorem expression_simplification (x y : ℚ) (hx : x = 3) (hy : y = -1/2) :
  x * (x - 4 * y) + (2 * x + y) * (2 * x - y) - (2 * x - y)^2 = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2369_236930


namespace NUMINAMATH_CALUDE_diet_soda_count_l2369_236943

/-- The number of diet soda bottles in a grocery store -/
def diet_soda : ℕ := sorry

/-- The number of regular soda bottles in the grocery store -/
def regular_soda : ℕ := 60

/-- The difference between regular and diet soda bottles -/
def difference : ℕ := 41

theorem diet_soda_count : diet_soda = 19 :=
  by
  have h1 : regular_soda = diet_soda + difference := sorry
  sorry

end NUMINAMATH_CALUDE_diet_soda_count_l2369_236943


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l2369_236986

theorem unique_solution_floor_equation :
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 4⌋ - ⌊(n : ℚ) / 2⌋^2 = 5 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l2369_236986


namespace NUMINAMATH_CALUDE_min_ratio_four_digit_number_l2369_236965

/-- A structure representing a four-digit number with distinct digits -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_nonzero : a ≠ 0
  distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  digits_range : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

/-- The value of a four-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- The sum of digits of a four-digit number -/
def digit_sum (n : FourDigitNumber) : Nat :=
  n.a + n.b + n.c + n.d

/-- The ratio of a four-digit number to the sum of its digits -/
def ratio (n : FourDigitNumber) : Rat :=
  (value n : Rat) / (digit_sum n : Rat)

theorem min_ratio_four_digit_number :
  ∃ (n : FourDigitNumber), 
    (∀ (m : FourDigitNumber), ratio n ≤ ratio m) ∧ 
    (ratio n = 60.5) ∧
    (value n = 1089) := by
  sorry

end NUMINAMATH_CALUDE_min_ratio_four_digit_number_l2369_236965


namespace NUMINAMATH_CALUDE_exam_mean_score_l2369_236961

theorem exam_mean_score (q σ : ℝ) 
  (h1 : 58 = q - 2 * σ) 
  (h2 : 98 = q + 3 * σ) : 
  q = 74 := by sorry

end NUMINAMATH_CALUDE_exam_mean_score_l2369_236961


namespace NUMINAMATH_CALUDE_existence_of_infinite_set_with_gcd_property_l2369_236982

theorem existence_of_infinite_set_with_gcd_property :
  ∃ (S : Set ℕ), Set.Infinite S ∧
  (∀ (x y z w : ℕ), x ∈ S → y ∈ S → z ∈ S → w ∈ S →
    x < y → z < w → (x, y) ≠ (z, w) →
    Nat.gcd (x * y + 2022) (z * w + 2022) = 1) :=
sorry

end NUMINAMATH_CALUDE_existence_of_infinite_set_with_gcd_property_l2369_236982


namespace NUMINAMATH_CALUDE_nail_polish_count_l2369_236962

theorem nail_polish_count (num_girls : ℕ) (nails_per_girl : ℕ) : 
  num_girls = 5 → nails_per_girl = 20 → num_girls * nails_per_girl = 100 := by
  sorry

end NUMINAMATH_CALUDE_nail_polish_count_l2369_236962


namespace NUMINAMATH_CALUDE_a_values_l2369_236919

def P : Set ℝ := {x | x^2 = 1}
def Q (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem a_values (a : ℝ) : Q a ⊆ P → a = 0 ∨ a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_values_l2369_236919


namespace NUMINAMATH_CALUDE_yellow_yellow_pairs_count_l2369_236900

/-- Represents the student pairing scenario in a math contest --/
structure ContestPairing where
  total_students : ℕ
  blue_students : ℕ
  yellow_students : ℕ
  total_pairs : ℕ
  blue_blue_pairs : ℕ

/-- The specific contest pairing scenario from the problem --/
def mathContest : ContestPairing := {
  total_students := 144
  blue_students := 63
  yellow_students := 81
  total_pairs := 72
  blue_blue_pairs := 27
}

/-- Theorem stating that the number of yellow-yellow pairs is 36 --/
theorem yellow_yellow_pairs_count (contest : ContestPairing) 
  (h1 : contest.total_students = contest.blue_students + contest.yellow_students)
  (h2 : contest.total_pairs * 2 = contest.total_students)
  (h3 : contest = mathContest) : 
  contest.yellow_students - (contest.total_pairs - contest.blue_blue_pairs - 
  (contest.blue_students - 2 * contest.blue_blue_pairs)) = 36 := by
  sorry

#check yellow_yellow_pairs_count

end NUMINAMATH_CALUDE_yellow_yellow_pairs_count_l2369_236900


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l2369_236996

-- Define the function f
def f (x a : ℝ) : ℝ := 5 - |x + a| - |x - 2|

-- Part 1
theorem solution_set_when_a_is_one :
  let a := 1
  {x : ℝ | f x a ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} :=
sorry

-- Part 2
theorem range_of_a_when_f_leq_one :
  {a : ℝ | ∀ x, f x a ≤ 1} = {a : ℝ | -6 ≤ a ∧ a ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l2369_236996


namespace NUMINAMATH_CALUDE_equation_solution_l2369_236957

theorem equation_solution : 
  ∀ x : ℝ, (Real.sqrt (x + 16) - 8 / Real.sqrt (x + 16) = 4) ↔ 
  (x = 20 + 8 * Real.sqrt 3 ∨ x = 20 - 8 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2369_236957


namespace NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l2369_236923

theorem square_sum_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 12) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l2369_236923


namespace NUMINAMATH_CALUDE_arithmetic_sequence_iff_c_eq_neg_one_l2369_236914

/-- Definition of the sum of the first n terms of the sequence -/
def S (n : ℕ) (c : ℝ) : ℝ := (n + 1)^2 + c

/-- Definition of the nth term of the sequence -/
def a (n : ℕ) (c : ℝ) : ℝ := S n c - S (n - 1) c

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: The sequence is arithmetic if and only if c = -1 -/
theorem arithmetic_sequence_iff_c_eq_neg_one (c : ℝ) :
  is_arithmetic_sequence (a · c) ↔ c = -1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_iff_c_eq_neg_one_l2369_236914


namespace NUMINAMATH_CALUDE_robin_gum_pieces_l2369_236911

/-- The number of gum packages Robin has -/
def num_packages : ℕ := 135

/-- The number of pieces in each package of gum -/
def pieces_per_package : ℕ := 46

/-- The total number of gum pieces Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_pieces : total_pieces = 6210 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_pieces_l2369_236911


namespace NUMINAMATH_CALUDE_expression_simplification_l2369_236984

theorem expression_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (x + a)^3 / ((a - b) * (a - c)) + (x + b)^3 / ((b - a) * (b - c)) + (x + c)^3 / ((c - a) * (c - b)) = a + b + c - 3*x :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2369_236984


namespace NUMINAMATH_CALUDE_third_set_candy_count_l2369_236920

/-- Represents the number of candies of a specific type in a set -/
structure CandyCount where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- Represents the total candy distribution across three sets -/
structure CandyDistribution where
  set1 : CandyCount
  set2 : CandyCount
  set3 : CandyCount

/-- The conditions of the candy distribution problem -/
def validDistribution (d : CandyDistribution) : Prop :=
  -- Total number of each type is equal across all sets
  d.set1.hard + d.set2.hard + d.set3.hard = 
  d.set1.chocolate + d.set2.chocolate + d.set3.chocolate ∧
  d.set1.hard + d.set2.hard + d.set3.hard = 
  d.set1.gummy + d.set2.gummy + d.set3.gummy ∧
  -- First set conditions
  d.set1.chocolate = d.set1.gummy ∧
  d.set1.hard = d.set1.chocolate + 7 ∧
  -- Second set conditions
  d.set2.hard = d.set2.chocolate ∧
  d.set2.gummy = d.set2.hard - 15 ∧
  -- Third set condition
  d.set3.hard = 0

/-- The main theorem stating that any valid distribution has 29 candies in the third set -/
theorem third_set_candy_count (d : CandyDistribution) : 
  validDistribution d → d.set3.chocolate + d.set3.gummy = 29 := by
  sorry


end NUMINAMATH_CALUDE_third_set_candy_count_l2369_236920


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2369_236964

/-- The number of sides of a regular polygon where the difference between 
    the number of diagonals and the number of sides is 7. -/
def polygon_sides : ℕ := 7

/-- The number of diagonals in a polygon with n sides. -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_polygon_sides :
  ∃ (n : ℕ), n > 0 ∧ num_diagonals n - n = 7 → n = polygon_sides :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2369_236964


namespace NUMINAMATH_CALUDE_square_flag_side_length_l2369_236992

theorem square_flag_side_length (total_fabric : ℝ) (square_flags : ℕ) (wide_flags : ℕ) (tall_flags : ℕ) (remaining_fabric : ℝ) :
  total_fabric = 1000 ∧ 
  square_flags = 16 ∧ 
  wide_flags = 20 ∧ 
  tall_flags = 10 ∧ 
  remaining_fabric = 294 →
  ∃ (side_length : ℝ),
    side_length = 4 ∧
    side_length^2 * square_flags + 15 * (wide_flags + tall_flags) = total_fabric - remaining_fabric :=
by sorry

end NUMINAMATH_CALUDE_square_flag_side_length_l2369_236992


namespace NUMINAMATH_CALUDE_sum_of_roots_l2369_236910

theorem sum_of_roots (x : ℝ) : 
  (∃ y z : ℝ, (3*x + 4)*(x - 5) + (3*x + 4)*(x - 7) = 0 ∧ x = y ∨ x = z) → 
  (∃ y z : ℝ, (3*x + 4)*(x - 5) + (3*x + 4)*(x - 7) = 0 ∧ x = y ∨ x = z ∧ y + z = 14/3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2369_236910


namespace NUMINAMATH_CALUDE_min_correct_answers_for_environmental_quiz_l2369_236927

/-- Represents a quiz with scoring rules -/
structure Quiz where
  totalQuestions : ℕ
  correctScore : ℕ
  incorrectDeduction : ℕ

/-- Calculates the score for a given number of correct answers -/
def calculateScore (quiz : Quiz) (correctAnswers : ℕ) : ℤ :=
  (quiz.correctScore * correctAnswers : ℤ) - 
  (quiz.incorrectDeduction * (quiz.totalQuestions - correctAnswers) : ℤ)

/-- The minimum number of correct answers needed to exceed the target score -/
def minCorrectAnswers (quiz : Quiz) (targetScore : ℤ) : ℕ :=
  quiz.totalQuestions.succ

theorem min_correct_answers_for_environmental_quiz :
  let quiz : Quiz := ⟨30, 10, 5⟩
  let targetScore : ℤ := 90
  minCorrectAnswers quiz targetScore = 17 ∧
  ∀ (x : ℕ), x ≥ minCorrectAnswers quiz targetScore → calculateScore quiz x > targetScore :=
by sorry

end NUMINAMATH_CALUDE_min_correct_answers_for_environmental_quiz_l2369_236927


namespace NUMINAMATH_CALUDE_circle_trajectory_and_line_l2369_236908

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 2
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 2

-- Define the trajectory of M
def M_trajectory (x y : ℝ) : Prop := x^2 / 2 - y^2 / 14 = 1 ∧ x ≥ Real.sqrt 2

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 14 * x - 27

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem circle_trajectory_and_line :
  ∃ (M : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ M → ∃ (r : ℝ), r > 0 ∧
      (∀ (x₁ y₁ : ℝ), C₁ x₁ y₁ → ((x - x₁)^2 + (y - y₁)^2 = (r + Real.sqrt 2)^2)) ∧
      (∀ (x₂ y₂ : ℝ), C₂ x₂ y₂ → ((x - x₂)^2 + (y - y₂)^2 = (r - Real.sqrt 2)^2))) ∧
    (∀ (x y : ℝ), (x, y) ∈ M ↔ M_trajectory x y) ∧
    (∃ (A B : ℝ × ℝ), A ∈ M ∧ B ∈ M ∧ A ≠ B ∧
      ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = P ∧
      (∀ (x y : ℝ), line_l x y ↔ (y - A.2) / (x - A.1) = (B.2 - A.2) / (B.1 - A.1) ∧ x ≠ A.1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_trajectory_and_line_l2369_236908


namespace NUMINAMATH_CALUDE_hide_and_seek_players_l2369_236903

-- Define variables for each person
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
axiom condition1 : Andrew → (Boris ∧ ¬Vasya)
axiom condition2 : Boris → (Gena ∨ Denis)
axiom condition3 : ¬Vasya → (¬Boris ∧ ¬Denis)
axiom condition4 : ¬Andrew → (Boris ∧ ¬Gena)

-- Theorem to prove
theorem hide_and_seek_players :
  (Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena) :=
sorry

end NUMINAMATH_CALUDE_hide_and_seek_players_l2369_236903


namespace NUMINAMATH_CALUDE_max_xy_given_constraint_l2369_236967

theorem max_xy_given_constraint (x y : ℝ) (h : 2 * x + y = 1) : 
  ∃ (max : ℝ), max = (1/8 : ℝ) ∧ ∀ (x' y' : ℝ), 2 * x' + y' = 1 → x' * y' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_xy_given_constraint_l2369_236967


namespace NUMINAMATH_CALUDE_factorization_and_simplification_l2369_236940

theorem factorization_and_simplification (x : ℝ) (h : x^2 ≠ 3 ∧ x^2 ≠ -1) :
  (12 * x^6 + 36 * x^4 - 9) / (3 * x^4 - 9 * x^2 - 9) =
  (4 * x^4 * (x^2 + 3) - 3) / ((x^2 - 3) * (x^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_and_simplification_l2369_236940


namespace NUMINAMATH_CALUDE_scientific_notation_6500_l2369_236906

theorem scientific_notation_6500 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 6500 = a * (10 : ℝ) ^ n ∧ a = 6.5 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_6500_l2369_236906


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2369_236959

theorem quadratic_equation_solution :
  ∃ (a b : ℕ+),
    (∃ (x : ℝ), x^2 + 8*x = 48 ∧ x > 0 ∧ x = Real.sqrt a - b) ∧
    a + b = 68 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2369_236959


namespace NUMINAMATH_CALUDE_isosceles_triangle_ef_length_l2369_236947

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  /-- The length of two equal sides -/
  side_length : ℝ
  /-- The ratio of EG to GF -/
  eg_gf_ratio : ℝ
  /-- The side length is positive -/
  side_length_pos : 0 < side_length
  /-- The ratio of EG to GF is 2 -/
  eg_gf_ratio_is_two : eg_gf_ratio = 2

/-- The theorem stating the length of EF in the isosceles triangle -/
theorem isosceles_triangle_ef_length (t : IsoscelesTriangle) (h : t.side_length = 10) :
  ∃ (ef : ℝ), ef = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_ef_length_l2369_236947


namespace NUMINAMATH_CALUDE_z_in_terms_of_x_l2369_236975

theorem z_in_terms_of_x (p : ℝ) (x z : ℝ) 
  (hx : x = 2 + 3^p) 
  (hz : z = 2 + 3^(-p)) : 
  z = (2*x - 3) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_z_in_terms_of_x_l2369_236975


namespace NUMINAMATH_CALUDE_min_square_value_l2369_236909

theorem min_square_value (a b : ℕ+) 
  (h1 : ∃ m : ℕ+, (15 * a + 16 * b : ℕ) = m ^ 2)
  (h2 : ∃ n : ℕ+, (16 * a - 15 * b : ℕ) = n ^ 2) :
  min (15 * a + 16 * b) (16 * a - 15 * b) ≥ 481 ^ 2 := by
sorry

end NUMINAMATH_CALUDE_min_square_value_l2369_236909


namespace NUMINAMATH_CALUDE_least_positive_integer_for_zero_sums_l2369_236913

theorem least_positive_integer_for_zero_sums (x₁ x₂ x₃ x₄ x₅ : ℝ) : 
  (∃ (S : Finset (Fin 5 × Fin 5 × Fin 5)), 
    S.card = 7 ∧ 
    (∀ (p q r : Fin 5), (p, q, r) ∈ S → p < q ∧ q < r) ∧
    (∀ (p q r : Fin 5), (p, q, r) ∈ S → x₁ * p.val + x₂ * q.val + x₃ * r.val = 0) →
    x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∧
  (∀ n : ℕ, n < 7 → 
    ∃ (x₁' x₂' x₃' x₄' x₅' : ℝ), 
      ∃ (S : Finset (Fin 5 × Fin 5 × Fin 5)),
        S.card = n ∧
        (∀ (p q r : Fin 5), (p, q, r) ∈ S → p < q ∧ q < r) ∧
        (∀ (p q r : Fin 5), (p, q, r) ∈ S → 
          x₁' * p.val + x₂' * q.val + x₃' * r.val = 0) ∧
        ¬(x₁' = 0 ∧ x₂' = 0 ∧ x₃' = 0 ∧ x₄' = 0 ∧ x₅' = 0)) := by
  sorry


end NUMINAMATH_CALUDE_least_positive_integer_for_zero_sums_l2369_236913


namespace NUMINAMATH_CALUDE_first_day_over_200_l2369_236907

def paperclips (n : ℕ) : ℕ := 5 * 3^(n - 1)

theorem first_day_over_200 :
  ∀ k : ℕ, k < 5 → paperclips k ≤ 200 ∧ paperclips 5 > 200 :=
by sorry

end NUMINAMATH_CALUDE_first_day_over_200_l2369_236907


namespace NUMINAMATH_CALUDE_tan_theta_value_l2369_236915

theorem tan_theta_value (θ : Real) 
  (h : (Real.sin (π - θ) + Real.cos (θ - 2*π)) / (Real.sin θ + Real.cos (π + θ)) = 1/2) : 
  Real.tan θ = -3 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l2369_236915


namespace NUMINAMATH_CALUDE_jose_share_of_profit_l2369_236993

/-- Calculates the share of profit for an investor given the total profit and investment ratios -/
def calculate_share_of_profit (total_profit : ℚ) (investment_ratio : ℚ) (total_investment_ratio : ℚ) : ℚ :=
  (investment_ratio / total_investment_ratio) * total_profit

theorem jose_share_of_profit (tom_investment : ℚ) (jose_investment : ℚ) 
  (tom_duration : ℚ) (jose_duration : ℚ) (total_profit : ℚ) :
  tom_investment = 3000 →
  jose_investment = 4500 →
  tom_duration = 12 →
  jose_duration = 10 →
  total_profit = 5400 →
  let tom_investment_ratio := tom_investment * tom_duration
  let jose_investment_ratio := jose_investment * jose_duration
  let total_investment_ratio := tom_investment_ratio + jose_investment_ratio
  calculate_share_of_profit total_profit jose_investment_ratio total_investment_ratio = 3000 := by
sorry

end NUMINAMATH_CALUDE_jose_share_of_profit_l2369_236993


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2369_236949

theorem simplify_fraction_product : 5 * (12 / 7) * (49 / -60) = -7 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2369_236949


namespace NUMINAMATH_CALUDE_value_of_expression_l2369_236979

theorem value_of_expression (a : ℝ) (h : a^2 - 2*a - 2 = 3) : 3*a*(a-2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2369_236979


namespace NUMINAMATH_CALUDE_pond_problem_l2369_236929

theorem pond_problem (initial_fish : ℕ) (fish_caught : ℕ) : 
  initial_fish = 50 →
  fish_caught = 7 →
  (initial_fish * 3 / 2) - (initial_fish - fish_caught) = 32 := by
  sorry

end NUMINAMATH_CALUDE_pond_problem_l2369_236929


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2369_236971

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 2) : 
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2369_236971


namespace NUMINAMATH_CALUDE_quadratic_prime_roots_unique_l2369_236995

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem quadratic_prime_roots_unique :
  ∃! k : ℤ, ∃ p q : ℕ,
    is_prime p ∧ is_prime q ∧
    p ≠ q ∧
    ∀ x : ℤ, x^2 - 74*x + k = 0 ↔ x = p ∨ x = q :=
sorry

end NUMINAMATH_CALUDE_quadratic_prime_roots_unique_l2369_236995


namespace NUMINAMATH_CALUDE_negative_number_with_abs_two_l2369_236963

theorem negative_number_with_abs_two (a : ℝ) (h1 : a < 0) (h2 : |a| = 2) : a = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_number_with_abs_two_l2369_236963


namespace NUMINAMATH_CALUDE_alternate_tree_planting_l2369_236901

/-- The number of ways to arrange n items from a set of m items, where order matters -/
def arrangements (m n : ℕ) : ℕ := sorry

/-- The number of ways to plant w willow trees and p poplar trees alternately in a row -/
def alternate_tree_arrangements (w p : ℕ) : ℕ :=
  2 * arrangements w w * arrangements p p

theorem alternate_tree_planting :
  alternate_tree_arrangements 4 4 = 1152 := by sorry

end NUMINAMATH_CALUDE_alternate_tree_planting_l2369_236901


namespace NUMINAMATH_CALUDE_log_problem_l2369_236989

-- Define the logarithm function for base 3
noncomputable def log3 (y : ℝ) : ℝ := Real.log y / Real.log 3

-- Define the logarithm function for base 9
noncomputable def log9 (y : ℝ) : ℝ := Real.log y / Real.log 9

theorem log_problem (x : ℝ) (h : log3 (x + 1) = 4) : log9 x = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2369_236989


namespace NUMINAMATH_CALUDE_sum_of_constants_l2369_236954

theorem sum_of_constants (a b : ℝ) : 
  (∀ x y : ℝ, y = a + b / x) →
  (2 = a + b / (-2)) →
  (7 = a + b / (-4)) →
  a + b = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_constants_l2369_236954


namespace NUMINAMATH_CALUDE_line_through_point_representation_l2369_236916

/-- A line in a 2D plane --/
structure Line where
  slope : Option ℝ
  yIntercept : ℝ

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point --/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  match l.slope with
  | some k => p.y = k * p.x + l.yIntercept
  | none => p.x = l.yIntercept

/-- The statement to be proven false --/
theorem line_through_point_representation (b : ℝ) :
  ∃ (k : ℝ), ∀ (l : Line), l.passesThrough ⟨0, b⟩ → 
  ∃ (k' : ℝ), l.slope = some k' ∧ l.yIntercept = b :=
sorry

end NUMINAMATH_CALUDE_line_through_point_representation_l2369_236916


namespace NUMINAMATH_CALUDE_option_b_more_cost_effective_l2369_236998

/-- Cost function for Option A -/
def cost_a (x : ℝ) : ℝ := 60 + 18 * x

/-- Cost function for Option B -/
def cost_b (x : ℝ) : ℝ := 150 + 15 * x

/-- Theorem stating that Option B is more cost-effective for 40 kg of blueberries -/
theorem option_b_more_cost_effective :
  cost_b 40 < cost_a 40 := by sorry

end NUMINAMATH_CALUDE_option_b_more_cost_effective_l2369_236998


namespace NUMINAMATH_CALUDE_smallest_third_term_is_negative_one_l2369_236987

/-- Given an arithmetic progression with first term 7, adding 3 to the second term
    and 15 to the third term results in a geometric progression. This function
    represents the smallest possible value for the third term of the resulting
    geometric progression. -/
def smallest_third_term_geometric : ℝ := sorry

/-- Theorem stating that the smallest possible value for the third term
    of the resulting geometric progression is -1. -/
theorem smallest_third_term_is_negative_one :
  smallest_third_term_geometric = -1 := by sorry

end NUMINAMATH_CALUDE_smallest_third_term_is_negative_one_l2369_236987


namespace NUMINAMATH_CALUDE_linear_coefficient_of_example_quadratic_l2369_236951

/-- Given a quadratic equation ax² + bx + c = 0, returns the coefficient of the linear term (b) -/
def linearCoefficient (a b c : ℚ) : ℚ := b

theorem linear_coefficient_of_example_quadratic :
  linearCoefficient 2 3 (-4) = 3 := by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_example_quadratic_l2369_236951


namespace NUMINAMATH_CALUDE_circumscribed_circle_radius_right_triangle_l2369_236921

/-- The radius of the circumscribed circle of a right triangle with sides 10, 8, and 6 is 5 -/
theorem circumscribed_circle_radius_right_triangle : 
  ∀ (a b c r : ℝ), 
  a = 10 → b = 8 → c = 6 → 
  a^2 = b^2 + c^2 → 
  r = a / 2 → 
  r = 5 := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_radius_right_triangle_l2369_236921


namespace NUMINAMATH_CALUDE_newton_interpolation_polynomial_l2369_236944

/-- The interpolation polynomial -/
def P (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

/-- The given points -/
def x₀ : ℝ := 2
def x₁ : ℝ := 4
def x₂ : ℝ := 5

/-- The given function values -/
def y₀ : ℝ := 1
def y₁ : ℝ := 15
def y₂ : ℝ := 28

theorem newton_interpolation_polynomial :
  P x₀ = y₀ ∧ P x₁ = y₁ ∧ P x₂ = y₂ ∧
  ∀ Q : ℝ → ℝ, (Q x₀ = y₀ ∧ Q x₁ = y₁ ∧ Q x₂ = y₂) →
  (∃ a b c : ℝ, ∀ x, Q x = a * x^2 + b * x + c) →
  (∀ x, Q x = P x) :=
sorry

end NUMINAMATH_CALUDE_newton_interpolation_polynomial_l2369_236944


namespace NUMINAMATH_CALUDE_student_b_score_l2369_236966

-- Define the scoring function
def calculateScore (totalQuestions : ℕ) (correctResponses : ℕ) : ℕ :=
  let incorrectResponses := totalQuestions - correctResponses
  correctResponses - 2 * incorrectResponses

-- Theorem statement
theorem student_b_score :
  calculateScore 100 91 = 73 := by
  sorry

end NUMINAMATH_CALUDE_student_b_score_l2369_236966


namespace NUMINAMATH_CALUDE_inequality_proof_l2369_236956

theorem inequality_proof (x y z : ℝ) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2369_236956


namespace NUMINAMATH_CALUDE_odd_coefficient_probability_l2369_236938

/-- The number of terms in the expansion of (1+x)^11 -/
def n : ℕ := 12

/-- The number of terms with odd coefficients in the expansion of (1+x)^11 -/
def k : ℕ := 8

/-- The probability of selecting a term with an odd coefficient from the expansion of (1+x)^11 -/
def p : ℚ := k / n

theorem odd_coefficient_probability : p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_odd_coefficient_probability_l2369_236938
