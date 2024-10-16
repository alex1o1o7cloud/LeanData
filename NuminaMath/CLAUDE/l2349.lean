import Mathlib

namespace NUMINAMATH_CALUDE_total_participants_is_260_l2349_234986

/-- Represents the voting scenario for a school dance -/
structure VotingScenario where
  initial_oct22_percent : ℝ
  initial_oct29_percent : ℝ
  additional_votes : ℕ
  final_oct29_percent : ℝ

/-- Calculates the total number of participants in the voting -/
def total_participants (scenario : VotingScenario) : ℕ :=
  sorry

/-- Theorem stating that the total number of participants is 260 -/
theorem total_participants_is_260 (scenario : VotingScenario) 
  (h1 : scenario.initial_oct22_percent = 0.35)
  (h2 : scenario.initial_oct29_percent = 0.65)
  (h3 : scenario.additional_votes = 80)
  (h4 : scenario.final_oct29_percent = 0.45) :
  total_participants scenario = 260 := by
  sorry

end NUMINAMATH_CALUDE_total_participants_is_260_l2349_234986


namespace NUMINAMATH_CALUDE_sarah_DC_probability_l2349_234974

def probability_DC : ℚ := 2/5

theorem sarah_DC_probability :
  let p : ℚ → ℚ := λ x => 1/3 + 1/6 * x
  ∃! x : ℚ, x = p x ∧ x = probability_DC :=
by sorry

end NUMINAMATH_CALUDE_sarah_DC_probability_l2349_234974


namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l2349_234944

theorem quadratic_square_of_binomial (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 116*x + c = (x + a)^2) → c = 3364 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l2349_234944


namespace NUMINAMATH_CALUDE_range_of_f_l2349_234920

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2349_234920


namespace NUMINAMATH_CALUDE_robot_walk_distance_l2349_234981

/-- Represents a rectangular field with robot paths -/
structure RobotField where
  length : ℕ
  width : ℕ
  path_width : ℕ
  b_distance : ℕ

/-- Calculates the total distance walked by the robot -/
def total_distance (field : RobotField) : ℕ :=
  let outer_loop := 2 * (field.length + field.width) - 1
  let second_loop := 2 * (field.length - 2 + field.width - 2) - 1
  let third_loop := 2 * (field.length - 4 + field.width - 4) - 1
  let fourth_loop := 2 * (field.length - 6 + field.width - 6) - 1
  let final_segment := field.length - field.path_width - field.b_distance
  outer_loop + second_loop + third_loop + fourth_loop + final_segment

/-- Theorem stating the total distance walked by the robot -/
theorem robot_walk_distance (field : RobotField) 
    (h1 : field.length = 16)
    (h2 : field.width = 8)
    (h3 : field.path_width = 1)
    (h4 : field.b_distance = 1) : 
  total_distance field = 154 := by
  sorry

end NUMINAMATH_CALUDE_robot_walk_distance_l2349_234981


namespace NUMINAMATH_CALUDE_problem_solution_l2349_234982

/-- The base-74 representation of the number in the problem -/
def base_74_num : ℕ := 235935623

/-- Converts the base-74 number to its decimal equivalent modulo 15 -/
def decimal_mod_15 : ℕ := base_74_num % 15

theorem problem_solution (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 14) 
  (h3 : (decimal_mod_15 - a) % 15 = 0) : a = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2349_234982


namespace NUMINAMATH_CALUDE_monotonic_subsequence_exists_l2349_234907

theorem monotonic_subsequence_exists (a : Fin 10 → ℝ) (h : Function.Injective a) :
  ∃ (i j k l : Fin 10), i < j ∧ j < k ∧ k < l ∧
    ((a i ≤ a j ∧ a j ≤ a k ∧ a k ≤ a l) ∨
     (a i ≥ a j ∧ a j ≥ a k ∧ a k ≥ a l)) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_subsequence_exists_l2349_234907


namespace NUMINAMATH_CALUDE_second_week_collection_l2349_234993

def total_goal : ℕ := 500
def first_week : ℕ := 158
def cans_needed : ℕ := 83

theorem second_week_collection : 
  total_goal - first_week - cans_needed = 259 := by
  sorry

end NUMINAMATH_CALUDE_second_week_collection_l2349_234993


namespace NUMINAMATH_CALUDE_remainder_N_mod_45_l2349_234931

def N : ℕ := sorry

theorem remainder_N_mod_45 : N % 45 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_N_mod_45_l2349_234931


namespace NUMINAMATH_CALUDE_minimum_values_l2349_234973

theorem minimum_values :
  (∀ x > 1, (x + 4 / (x - 1) ≥ 5) ∧ (x + 4 / (x - 1) = 5 ↔ x = 3)) ∧
  (∀ x, 0 < x → x < 1 → (4 / x + 1 / (1 - x) ≥ 9) ∧ (4 / x + 1 / (1 - x) = 9 ↔ x = 2/3)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_values_l2349_234973


namespace NUMINAMATH_CALUDE_volume_of_parallelepiped_l2349_234926

/-- A rectangular parallelepiped with given diagonal and side face diagonals -/
structure RectParallelepiped where
  diag : ℝ
  side_diag1 : ℝ
  side_diag2 : ℝ
  volume : ℝ

/-- The volume of a rectangular parallelepiped with the given dimensions -/
def volume_calc (p : RectParallelepiped) : Prop :=
  p.diag = 13 ∧ p.side_diag1 = 4 * Real.sqrt 10 ∧ p.side_diag2 = 3 * Real.sqrt 17 → p.volume = 144

theorem volume_of_parallelepiped :
  ∀ p : RectParallelepiped, volume_calc p :=
sorry

end NUMINAMATH_CALUDE_volume_of_parallelepiped_l2349_234926


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l2349_234965

theorem line_equation_equivalence (x y : ℝ) (h : 2*x - 5*y - 3 = 0) : 
  -4*x + 10*y + 3 = 0 := by sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l2349_234965


namespace NUMINAMATH_CALUDE_trajectory_of_M_is_ellipse_l2349_234978

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 32 = 0

-- Define point A
def point_A : ℝ × ℝ := (2, 0)

-- Define a moving point P on circle C
def point_P (x y : ℝ) : Prop := circle_C x y

-- Define point M as the intersection of perpendicular bisector of AP and line PC
def point_M (x y : ℝ) : Prop :=
  ∃ (px py : ℝ), point_P px py ∧
  ((x - 2)^2 + y^2 = (x - px)^2 + (y - py)^2) ∧
  ((x - 2) * (px - 2) + y * py = 0)

-- Theorem: The trajectory of point M is an ellipse
theorem trajectory_of_M_is_ellipse :
  ∀ (x y : ℝ), point_M x y ↔ x^2/9 + y^2/5 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_M_is_ellipse_l2349_234978


namespace NUMINAMATH_CALUDE_sculptures_not_on_display_sculptures_not_on_display_proof_l2349_234952

/-- Represents the total number of art pieces in the gallery -/
def total_art_pieces : ℕ := 3150

/-- Represents the fraction of art pieces on display -/
def fraction_on_display : ℚ := 1/3

/-- Represents the fraction of sculptures among displayed pieces -/
def fraction_sculptures_displayed : ℚ := 1/6

/-- Represents the fraction of paintings among pieces not on display -/
def fraction_paintings_not_displayed : ℚ := 1/3

/-- Represents that some sculptures are not on display -/
axiom some_sculptures_not_displayed : ∃ (n : ℕ), n > 0 ∧ n ≤ total_art_pieces

theorem sculptures_not_on_display : ℕ :=
  1400

theorem sculptures_not_on_display_proof : sculptures_not_on_display = 1400 := by
  sorry

end NUMINAMATH_CALUDE_sculptures_not_on_display_sculptures_not_on_display_proof_l2349_234952


namespace NUMINAMATH_CALUDE_technician_salary_l2349_234988

theorem technician_salary (total_workers : ℕ) (total_avg_salary : ℝ) 
  (num_technicians : ℕ) (non_tech_avg_salary : ℝ) :
  total_workers = 18 →
  total_avg_salary = 8000 →
  num_technicians = 6 →
  non_tech_avg_salary = 6000 →
  (total_workers * total_avg_salary - (total_workers - num_technicians) * non_tech_avg_salary) / num_technicians = 12000 := by
  sorry

end NUMINAMATH_CALUDE_technician_salary_l2349_234988


namespace NUMINAMATH_CALUDE_angle_bisection_quadrant_l2349_234937

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : Int, k * 2 * Real.pi + Real.pi < α ∧ α < k * 2 * Real.pi + 3 * Real.pi / 2

def is_in_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, (k * Real.pi + Real.pi / 2 < α ∧ α < k * Real.pi + Real.pi) ∨
             (k * Real.pi + 3 * Real.pi / 2 < α ∧ α < (k + 1) * Real.pi)

theorem angle_bisection_quadrant (α : Real) :
  is_in_third_quadrant α → is_in_second_or_fourth_quadrant (α / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_angle_bisection_quadrant_l2349_234937


namespace NUMINAMATH_CALUDE_surface_area_of_slice_theorem_l2349_234933

/-- Represents a right prism with isosceles triangular bases -/
structure IsoscelesPrism where
  height : ℝ
  base_length : ℝ
  side_length : ℝ

/-- Calculates the surface area of the sliced off portion of the prism -/
def surface_area_of_slice (prism : IsoscelesPrism) : ℝ :=
  sorry

/-- Theorem stating the surface area of the sliced portion -/
theorem surface_area_of_slice_theorem (prism : IsoscelesPrism) 
  (h1 : prism.height = 10)
  (h2 : prism.base_length = 10)
  (h3 : prism.side_length = 12) :
  surface_area_of_slice prism = 52.25 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_slice_theorem_l2349_234933


namespace NUMINAMATH_CALUDE_max_value_expression_l2349_234932

theorem max_value_expression (x k : ℕ) (hx : x > 0) (hk : k > 0) : 
  let y := k * x
  ∃ (max : ℚ), max = 2 ∧ ∀ (x' k' : ℕ), x' > 0 → k' > 0 → 
    let y' := k' * x'
    (x' + y')^2 / (x'^2 + y'^2 : ℚ) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2349_234932


namespace NUMINAMATH_CALUDE_book_purchase_change_l2349_234930

/-- Calculates the change received when buying two items with given prices and paying with a given amount. -/
def calculate_change (price1 : ℝ) (price2 : ℝ) (payment : ℝ) : ℝ :=
  payment - (price1 + price2)

/-- Theorem stating that buying two books priced at 5.5£ and 6.5£ with a 20£ bill results in 8£ change. -/
theorem book_purchase_change : calculate_change 5.5 6.5 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_change_l2349_234930


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2349_234908

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2349_234908


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2349_234963

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (9 ∣ (n + 6)) ∧ 
  (4 ∣ (n - 7)) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (9 ∣ (m + 6)) ∧ (4 ∣ (m - 7))) → false) ∧
  n = 111 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2349_234963


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2349_234915

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1 → x^2 > 1) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2349_234915


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2349_234962

theorem arithmetic_equality : 245 - 57 + 136 + 14 - 38 = 300 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2349_234962


namespace NUMINAMATH_CALUDE_smallest_n_with_divisible_sum_or_diff_l2349_234954

theorem smallest_n_with_divisible_sum_or_diff (n : ℕ) : n = 1006 ↔ 
  (∀ (S : Finset ℤ), S.card = n → 
    ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (2009 ∣ (a + b) ∨ 2009 ∣ (a - b))) ∧
  (∀ (m : ℕ), m < n → 
    ∃ (T : Finset ℤ), T.card = m ∧
      ∀ (a b : ℤ), a ∈ T → b ∈ T → a ≠ b → ¬(2009 ∣ (a + b) ∨ 2009 ∣ (a - b))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_divisible_sum_or_diff_l2349_234954


namespace NUMINAMATH_CALUDE_circle_area_theorem_l2349_234966

def A : ℝ × ℝ := (8, 15)
def B : ℝ × ℝ := (14, 9)

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def tangent_line (c : Circle) (p : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

def on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

def intersect_x_axis (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

theorem circle_area_theorem (ω : Circle) :
  on_circle A ω →
  on_circle B ω →
  (∃ (p : ℝ × ℝ), p.2 = 0 ∧ p ∈ tangent_line ω A ∧ p ∈ tangent_line ω B) →
  ω.radius^2 * Real.pi = 306 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l2349_234966


namespace NUMINAMATH_CALUDE_sin_cos_sum_negative_sqrt_two_l2349_234902

theorem sin_cos_sum_negative_sqrt_two (x : Real) : 
  0 ≤ x → x < 2 * Real.pi → Real.sin x + Real.cos x = -Real.sqrt 2 → x = 5 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_negative_sqrt_two_l2349_234902


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l2349_234906

theorem smallest_sum_of_squares (x y : ℝ) :
  (∀ a b : ℝ, x^2 + y^2 ≤ a^2 + b^2) → x^2 + y^2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l2349_234906


namespace NUMINAMATH_CALUDE_second_quadrant_necessary_not_sufficient_for_obtuse_l2349_234995

/-- An angle is in the second quadrant if it's between 90° and 180° exclusive. -/
def is_in_second_quadrant (α : ℝ) : Prop :=
  90 < α ∧ α < 180

/-- An angle is obtuse if it's between 90° and 180° exclusive. -/
def is_obtuse (α : ℝ) : Prop :=
  90 < α ∧ α < 180

theorem second_quadrant_necessary_not_sufficient_for_obtuse :
  (∀ α, is_obtuse α → is_in_second_quadrant α) ∧
  (∃ α, is_in_second_quadrant α ∧ ¬is_obtuse α) :=
by sorry

end NUMINAMATH_CALUDE_second_quadrant_necessary_not_sufficient_for_obtuse_l2349_234995


namespace NUMINAMATH_CALUDE_value_of_M_l2349_234975

theorem value_of_M : ∃ M : ℝ, (0.12 * M = 0.60 * 1500) ∧ (M = 7500) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l2349_234975


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2349_234994

/-- An isosceles triangle with a median to the leg dividing the perimeter -/
structure IsoscelesTriangleWithMedian where
  /-- Length of the leg of the isosceles triangle -/
  leg : ℝ
  /-- Length of the base of the isosceles triangle -/
  base : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : leg > 0
  /-- The median to the leg divides the perimeter into two parts -/
  medianDivides : leg + leg + base = 27
  /-- One part of the divided perimeter is 15 -/
  part1 : leg + leg / 2 = 15 ∨ leg / 2 + base = 15

/-- The theorem stating the possible base lengths of the isosceles triangle -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangleWithMedian) :
  t.base = 7 ∨ t.base = 11 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2349_234994


namespace NUMINAMATH_CALUDE_folded_rectangle_area_l2349_234903

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

theorem folded_rectangle_area :
  ∀ (r1 r2 r3 : Rectangle),
    perimeter r1 = perimeter r2 + 20 →
    perimeter r2 = perimeter r3 + 16 →
    r1.length = r2.length →
    r2.length = r3.length →
    r1.width = r2.width + 10 →
    r2.width = r3.width + 8 →
    area r1 = 504 :=
by sorry

end NUMINAMATH_CALUDE_folded_rectangle_area_l2349_234903


namespace NUMINAMATH_CALUDE_negative_difference_equality_l2349_234914

theorem negative_difference_equality (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_equality_l2349_234914


namespace NUMINAMATH_CALUDE_original_expression_proof_l2349_234940

theorem original_expression_proof (X a b c : ℤ) : 
  X + (a*b - 2*b*c + 3*a*c) = 2*b*c - 3*a*c + 2*a*b → 
  X = 4*b*c - 6*a*c + a*b := by
sorry

end NUMINAMATH_CALUDE_original_expression_proof_l2349_234940


namespace NUMINAMATH_CALUDE_nearest_integer_to_x_plus_2y_l2349_234956

theorem nearest_integer_to_x_plus_2y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : |x| + 2 * y = 6) (h2 : |x| * y + x^3 = 2) :
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |x + 2 * y - ↑n| ≤ |x + 2 * y - ↑m| :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_x_plus_2y_l2349_234956


namespace NUMINAMATH_CALUDE_megan_folders_l2349_234997

theorem megan_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) :
  initial_files = 93 →
  deleted_files = 21 →
  files_per_folder = 8 →
  (initial_files - deleted_files) / files_per_folder = 9 :=
by sorry

end NUMINAMATH_CALUDE_megan_folders_l2349_234997


namespace NUMINAMATH_CALUDE_take_home_pay_calculation_l2349_234943

/-- Calculate take-home pay after deductions -/
theorem take_home_pay_calculation (total_pay : ℝ) 
  (tax_rate insurance_rate pension_rate union_rate : ℝ) :
  total_pay = 500 →
  tax_rate = 0.10 →
  insurance_rate = 0.05 →
  pension_rate = 0.03 →
  union_rate = 0.02 →
  total_pay * (1 - (tax_rate + insurance_rate + pension_rate + union_rate)) = 400 := by
  sorry

end NUMINAMATH_CALUDE_take_home_pay_calculation_l2349_234943


namespace NUMINAMATH_CALUDE_canoe_rental_cost_l2349_234919

/-- Represents the daily rental cost and count of canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℝ
  kayak_cost : ℝ
  canoe_count : ℕ
  kayak_count : ℕ

/-- Calculates the total revenue from canoe and kayak rentals --/
def total_revenue (info : RentalInfo) : ℝ :=
  info.canoe_cost * info.canoe_count + info.kayak_cost * info.kayak_count

/-- Theorem stating that the daily rental cost of a canoe is $15 --/
theorem canoe_rental_cost (info : RentalInfo) :
  info.kayak_cost = 18 ∧
  info.canoe_count = (3 * info.kayak_count) / 2 ∧
  total_revenue info = 405 ∧
  info.canoe_count = info.kayak_count + 5 →
  info.canoe_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_canoe_rental_cost_l2349_234919


namespace NUMINAMATH_CALUDE_drawing_red_is_certain_l2349_234950

/-- Represents a ball in the box -/
inductive Ball
  | Red

/-- Represents the box containing balls -/
def Box := List Ball

/-- Defines a certain event -/
def CertainEvent (event : Prop) : Prop :=
  ∀ (outcome : Prop), event = outcome

/-- The box contains exactly two red balls -/
def TwoRedBalls (box : Box) : Prop :=
  box = [Ball.Red, Ball.Red]

/-- Drawing a ball from the box -/
def DrawBall (box : Box) : Ball :=
  match box with
  | [] => Ball.Red  -- Default case, should not occur
  | (b :: _) => b

/-- The main theorem: Drawing a red ball from a box with two red balls is a certain event -/
theorem drawing_red_is_certain (box : Box) (h : TwoRedBalls box) :
  CertainEvent (DrawBall box = Ball.Red) := by
  sorry

end NUMINAMATH_CALUDE_drawing_red_is_certain_l2349_234950


namespace NUMINAMATH_CALUDE_line_through_point_and_trisection_l2349_234990

/-- The line passing through (2,3) and one of the trisection points of the line segment
    joining (1,2) and (7,-4) has the equation 4x - 9y + 15 = 0 -/
theorem line_through_point_and_trisection :
  ∃ (t : ℝ) (x y : ℝ),
    -- Define the trisection point
    x = 1 + 2 * t * (7 - 1) ∧
    y = 2 + 2 * t * (-4 - 2) ∧
    0 ≤ t ∧ t ≤ 1 ∧
    -- The trisection point is on the line
    4 * x - 9 * y + 15 = 0 ∧
    -- The point (2,3) is on the line
    4 * 2 - 9 * 3 + 15 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_through_point_and_trisection_l2349_234990


namespace NUMINAMATH_CALUDE_complex_root_and_imaginary_value_l2349_234959

theorem complex_root_and_imaginary_value (m n a : ℝ) : 
  (Complex.I * Complex.I = -1) →
  ((2 : ℂ) - Complex.I) ^ 2 - m * ((2 : ℂ) - Complex.I) + n = 0 →
  (a ^ 2 - n * a + m : ℂ) + (a - m) * Complex.I = Complex.I * (b : ℝ) →
  m = 4 ∧ n = 5 ∧ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_root_and_imaginary_value_l2349_234959


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2349_234968

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 3 ∧ 
  k = -4 ∧ 
  c = Real.sqrt 53 ∧ 
  a = 4 ∧ 
  c^2 = a^2 + b^2 → 
  h + k + a + b = 3 + Real.sqrt 37 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2349_234968


namespace NUMINAMATH_CALUDE_apples_per_basket_l2349_234917

theorem apples_per_basket (total_baskets : ℕ) (total_apples : ℕ) 
  (h1 : total_baskets = 37) 
  (h2 : total_apples = 629) : 
  total_apples / total_baskets = 17 := by
sorry

end NUMINAMATH_CALUDE_apples_per_basket_l2349_234917


namespace NUMINAMATH_CALUDE_greatest_real_part_of_cube_l2349_234951

theorem greatest_real_part_of_cube (z₁ z₂ z₃ z₄ z₅ : ℂ) : 
  z₁ = -1 ∧ 
  z₂ = -Real.sqrt 2 + I ∧ 
  z₃ = -1 + Real.sqrt 3 * I ∧ 
  z₄ = 2 * I ∧ 
  z₅ = -1 - Real.sqrt 3 * I → 
  (z₄^3).re ≥ (z₁^3).re ∧ 
  (z₄^3).re ≥ (z₂^3).re ∧ 
  (z₄^3).re ≥ (z₃^3).re ∧ 
  (z₄^3).re ≥ (z₅^3).re :=
by sorry

end NUMINAMATH_CALUDE_greatest_real_part_of_cube_l2349_234951


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2349_234967

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (b < -1 → |a| + |b| > 1) ∧ 
  ∃ a b : ℝ, |a| + |b| > 1 ∧ b ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2349_234967


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2349_234946

theorem no_positive_integer_solutions : ∀ A : ℕ, 
  1 ≤ A → A ≤ 9 → 
  ¬∃ p q : ℕ, p > 0 ∧ q > 0 ∧ p * q = Nat.factorial A ∧ p + q = 10 * A + A := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2349_234946


namespace NUMINAMATH_CALUDE_range_of_a_l2349_234980

-- Define the statements p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x - 1 < 0

def q (a : ℝ) : Prop := (3 / (a - 1)) + 1 < 0

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(p a ∨ q a)) → (a ≤ -4 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2349_234980


namespace NUMINAMATH_CALUDE_triangle_problem_l2349_234905

theorem triangle_problem (a b c A B C : Real) (R : Real) :
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- Angles are positive
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Sides are positive
  a + b + c = π ∧  -- Sum of angles in a triangle
  2 * B * Real.cos A = C * Real.cos a + A * Real.cos c ∧  -- Given equation
  A + B + C = 8 ∧  -- Perimeter is 8
  R = Real.sqrt 3 ∧  -- Radius of circumscribed circle is √3
  2 * R * Real.sin (a / 2) = A ∧  -- Relation between side and circumradius
  2 * R * Real.sin (b / 2) = B ∧
  2 * R * Real.sin (c / 2) = C →
  a = π / 3 ∧  -- Angle A is 60°
  A * B * Real.sin c / 2 = 4 * Real.sqrt 3 / 3  -- Area of triangle
  :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2349_234905


namespace NUMINAMATH_CALUDE_smallest_bdf_l2349_234927

theorem smallest_bdf (a b c d e f : ℕ+) : 
  (∃ A : ℚ, A = (a / b) * (c / d) * (e / f) ∧ 
   ((a + 1) / b) * (c / d) * (e / f) = A + 3 ∧
   (a / b) * ((c + 1) / d) * (e / f) = A + 4 ∧
   (a / b) * (c / d) * ((e + 1) / f) = A + 5) →
  (∃ m : ℕ+, b * d * f = m ∧ ∀ n : ℕ+, b * d * f ≤ n) →
  b * d * f = 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_bdf_l2349_234927


namespace NUMINAMATH_CALUDE_janet_paperclips_used_l2349_234996

/-- The number of paper clips Janet used during the day -/
def paperclips_used (initial : ℕ) (found : ℕ) (given_per_friend : ℕ) (num_friends : ℕ) (final : ℕ) : ℕ :=
  initial + found - given_per_friend * num_friends - final

/-- Theorem stating that Janet used 64 paper clips during the day -/
theorem janet_paperclips_used :
  paperclips_used 85 20 5 3 26 = 64 := by
  sorry

end NUMINAMATH_CALUDE_janet_paperclips_used_l2349_234996


namespace NUMINAMATH_CALUDE_binomial_prob_l2349_234900

/-- A random variable following a binomial distribution B(2,p) -/
def X (p : ℝ) : Type := Unit

/-- The probability that X is greater than or equal to 1 -/
def prob_X_geq_1 (p : ℝ) : ℝ := 1 - (1 - p)^2

/-- The theorem stating that if P(X ≥ 1) = 5/9 for X ~ B(2,p), then p = 1/3 -/
theorem binomial_prob (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  prob_X_geq_1 p = 5/9 → p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_prob_l2349_234900


namespace NUMINAMATH_CALUDE_statement_1_false_statement_2_false_statement_3_false_statement_4_true_l2349_234983

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

variable (m n : Line)
variable (α β : Plane)

-- Statement ①
theorem statement_1_false :
  ¬(parallelLP m α → parallelLP n α → parallel m n) :=
sorry

-- Statement ②
theorem statement_2_false :
  ¬(subset m α → subset n α → parallelLP m β → parallelLP n β → parallelPP α β) :=
sorry

-- Statement ③
theorem statement_3_false :
  ¬(perpendicular α β → subset m α → perpendicularLP m β) :=
sorry

-- Statement ④
theorem statement_4_true :
  perpendicular α β → perpendicularLP m β → ¬(subset m α) → parallelLP m α :=
sorry

end NUMINAMATH_CALUDE_statement_1_false_statement_2_false_statement_3_false_statement_4_true_l2349_234983


namespace NUMINAMATH_CALUDE_xyz_value_l2349_234934

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 8 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l2349_234934


namespace NUMINAMATH_CALUDE_inequality_requires_conditional_structure_l2349_234945

-- Define the types of algorithms
inductive Algorithm
  | SolveInequality
  | CalculateAverage
  | CalculateCircleArea
  | FindRoots

-- Define a function to check if an algorithm requires a conditional structure
def requiresConditionalStructure (alg : Algorithm) : Prop :=
  match alg with
  | Algorithm.SolveInequality => true
  | _ => false

-- Theorem statement
theorem inequality_requires_conditional_structure :
  requiresConditionalStructure Algorithm.SolveInequality ∧
  ¬requiresConditionalStructure Algorithm.CalculateAverage ∧
  ¬requiresConditionalStructure Algorithm.CalculateCircleArea ∧
  ¬requiresConditionalStructure Algorithm.FindRoots :=
sorry

end NUMINAMATH_CALUDE_inequality_requires_conditional_structure_l2349_234945


namespace NUMINAMATH_CALUDE_exponential_inequality_l2349_234935

theorem exponential_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (1 / 3 : ℝ) ^ x < (1 / 3 : ℝ) ^ y := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2349_234935


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l2349_234925

/-- The area of a circle with radius r -/
def circle_area (r : ℝ) : ℝ := 3.14 * r ^ 2

/-- The theorem stating the area of the shaded region -/
theorem shaded_area_theorem :
  let large_radius : ℝ := 20
  let small_radius : ℝ := 10
  let num_small_circles : ℕ := 7
  let large_circle_area := circle_area large_radius
  let small_circle_area := circle_area small_radius
  large_circle_area - (num_small_circles : ℝ) * small_circle_area = 942 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l2349_234925


namespace NUMINAMATH_CALUDE_square_perimeter_l2349_234984

/-- The sum of the lengths of all sides of a square with side length 5 cm is 20 cm. -/
theorem square_perimeter (side_length : ℝ) (h : side_length = 5) : 
  4 * side_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2349_234984


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_one_range_of_m_when_a_gt_one_l2349_234929

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * |x + 1| - |x - 1|

-- Part 1
theorem solution_set_when_a_eq_one :
  {x : ℝ | f 1 x < 3/2} = {x : ℝ | x < 3/4} := by sorry

-- Part 2
theorem range_of_m_when_a_gt_one (a : ℝ) (m : ℝ) :
  a > 1 →
  (∃ x : ℝ, f a x ≤ -|2*m + 1|) →
  m ∈ Set.Icc (-3/2) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_one_range_of_m_when_a_gt_one_l2349_234929


namespace NUMINAMATH_CALUDE_password_config_exists_l2349_234936

/-- A password configuration is represented by a list of integers, 
    where each integer represents the count of a distinct character. -/
def PasswordConfig := List Nat

/-- The number of combinations for a given password configuration -/
def numCombinations (config : PasswordConfig) : Nat :=
  Nat.factorial 5 / (config.map Nat.factorial).prod

/-- Theorem: There exists a 5-character password configuration 
    that results in exactly 20 different combinations -/
theorem password_config_exists : ∃ (config : PasswordConfig), 
  config.sum = 5 ∧ numCombinations config = 20 := by
  sorry

end NUMINAMATH_CALUDE_password_config_exists_l2349_234936


namespace NUMINAMATH_CALUDE_loss_percentage_is_five_percent_l2349_234964

def original_price : ℚ := 490
def sale_price : ℚ := 465.5

def loss_amount : ℚ := original_price - sale_price

def loss_percentage : ℚ := (loss_amount / original_price) * 100

theorem loss_percentage_is_five_percent :
  loss_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_loss_percentage_is_five_percent_l2349_234964


namespace NUMINAMATH_CALUDE_a_gt_2_sufficient_not_necessary_l2349_234969

theorem a_gt_2_sufficient_not_necessary :
  (∀ a : ℝ, a > 2 → 2^a - a - 1 > 0) ∧
  (∃ a : ℝ, a ≤ 2 ∧ 2^a - a - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_2_sufficient_not_necessary_l2349_234969


namespace NUMINAMATH_CALUDE_warehouse_analysis_l2349_234972

/-- Represents the daily record of material movement --/
structure MaterialRecord where
  quantity : Int
  times : Nat

/-- Calculates the net change in material quantity --/
def netChange (records : List MaterialRecord) : Int :=
  records.foldl (fun acc r => acc + r.quantity * r.times) 0

/-- Calculates the transportation cost for Option 1 --/
def costOption1 (records : List MaterialRecord) : Int :=
  records.foldl (fun acc r =>
    acc + (if r.quantity > 0 then 5 else 8) * r.quantity.natAbs * r.times
  ) 0

/-- Calculates the transportation cost for Option 2 --/
def costOption2 (records : List MaterialRecord) : Int :=
  records.foldl (fun acc r => acc + 6 * r.quantity.natAbs * r.times) 0

theorem warehouse_analysis (records : List MaterialRecord) :
  records = [
    { quantity := -3, times := 2 },
    { quantity := 4, times := 1 },
    { quantity := -1, times := 3 },
    { quantity := 2, times := 3 },
    { quantity := -5, times := 2 }
  ] →
  netChange records = -9 ∧ costOption2 records < costOption1 records := by
  sorry

end NUMINAMATH_CALUDE_warehouse_analysis_l2349_234972


namespace NUMINAMATH_CALUDE_joneal_stops_in_quarter_A_l2349_234998

/-- Represents the quarters of the circular track -/
inductive Quarter : Type
| A : Quarter
| B : Quarter
| C : Quarter
| D : Quarter

/-- Calculates the quarter in which a runner stops after running a given distance -/
def stopQuarter (trackCircumference : ℕ) (runDistance : ℕ) : Quarter :=
  match (runDistance % trackCircumference) / (trackCircumference / 4) with
  | 0 => Quarter.A
  | 1 => Quarter.B
  | 2 => Quarter.C
  | _ => Quarter.D

theorem joneal_stops_in_quarter_A :
  let trackCircumference : ℕ := 100
  let runDistance : ℕ := 10000
  stopQuarter trackCircumference runDistance = Quarter.A := by
  sorry

end NUMINAMATH_CALUDE_joneal_stops_in_quarter_A_l2349_234998


namespace NUMINAMATH_CALUDE_gold_found_per_hour_l2349_234955

def diving_time : ℕ := 8
def chest_gold : ℕ := 100
def num_small_bags : ℕ := 2

def gold_per_hour : ℚ :=
  let small_bag_gold := chest_gold / 2
  let total_gold := chest_gold + num_small_bags * small_bag_gold
  total_gold / diving_time

theorem gold_found_per_hour :
  gold_per_hour = 25 := by sorry

end NUMINAMATH_CALUDE_gold_found_per_hour_l2349_234955


namespace NUMINAMATH_CALUDE_equilateral_triangle_revolution_surface_area_l2349_234924

/-- The surface area of a solid of revolution formed by rotating an equilateral triangle -/
theorem equilateral_triangle_revolution_surface_area 
  (side_length : ℝ) 
  (h_side : side_length = 2) : 
  let solid_surface_area := 2 * Real.pi * (side_length * Real.sqrt 3 / 2) * side_length
  solid_surface_area = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_revolution_surface_area_l2349_234924


namespace NUMINAMATH_CALUDE_twentieth_term_is_96_l2349_234913

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 20th term of the specific arithmetic sequence -/
theorem twentieth_term_is_96 :
  arithmeticSequenceTerm 1 5 20 = 96 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_is_96_l2349_234913


namespace NUMINAMATH_CALUDE_water_evaporation_per_day_l2349_234921

def initial_water : ℝ := 10
def evaporation_period : ℕ := 20
def evaporation_percentage : ℝ := 0.12

theorem water_evaporation_per_day :
  let total_evaporated := initial_water * evaporation_percentage
  let daily_evaporation := total_evaporated / evaporation_period
  daily_evaporation = 0.06 := by sorry

end NUMINAMATH_CALUDE_water_evaporation_per_day_l2349_234921


namespace NUMINAMATH_CALUDE_min_value_expression_l2349_234999

/-- The minimum value of (s+5-3|cos t|)^2 + (s-2|sin t|)^2 is 2, where s and t are real numbers. -/
theorem min_value_expression : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (s t : ℝ), (s + 5 - 3 * |Real.cos t|)^2 + (s - 2 * |Real.sin t|)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2349_234999


namespace NUMINAMATH_CALUDE_max_third_term_is_16_l2349_234970

/-- An arithmetic sequence of four positive integers -/
structure ArithmeticSequence :=
  (a : ℕ+) -- First term
  (d : ℕ+) -- Common difference
  (sum_eq_50 : a + (a + d) + (a + 2*d) + (a + 3*d) = 50)
  (third_term_even : Even (a + 2*d))

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℕ := seq.a + 2*seq.d

/-- Theorem: The maximum possible value for the third term is 16 -/
theorem max_third_term_is_16 :
  ∀ seq : ArithmeticSequence, third_term seq ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_third_term_is_16_l2349_234970


namespace NUMINAMATH_CALUDE_function_properties_l2349_234942

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_deriv : ∀ x ∈ Set.Ioo 0 (π/2), f' x * Real.sin x - f x * Real.cos x > 0) :
  f (π/4) > -Real.sqrt 2 * f (-π/6) ∧ f (π/3) > Real.sqrt 3 * f (π/6) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2349_234942


namespace NUMINAMATH_CALUDE_train_length_l2349_234928

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 72 * (1000 / 3600) → 
  crossing_time = 30 → 
  bridge_length = 350 → 
  train_speed * crossing_time - bridge_length = 250 := by sorry

end NUMINAMATH_CALUDE_train_length_l2349_234928


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2349_234938

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : 
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2349_234938


namespace NUMINAMATH_CALUDE_dishes_to_equalize_is_six_l2349_234958

/-- Represents the time in minutes for various chores -/
structure ChoreTime where
  sweepingPerRoom : ℕ
  dishWashingPerDish : ℕ
  laundryPerLoad : ℕ

/-- Represents the chores assigned to each child -/
structure ChoreAssignment where
  annaSweepingRooms : ℕ
  billyLaundryLoads : ℕ

/-- Calculates the number of dishes Billy needs to wash to equalize work time -/
def dishesToEqualize (times : ChoreTime) (assignment : ChoreAssignment) : ℕ :=
  let annaTime := times.sweepingPerRoom * assignment.annaSweepingRooms
  let billyLaundryTime := times.laundryPerLoad * assignment.billyLaundryLoads
  let timeDifference := annaTime - billyLaundryTime
  timeDifference / times.dishWashingPerDish

/-- The main theorem stating that Billy needs to wash 6 dishes to equalize work time -/
theorem dishes_to_equalize_is_six :
  let times : ChoreTime := ⟨3, 2, 9⟩
  let assignment : ChoreAssignment := ⟨10, 2⟩
  dishesToEqualize times assignment = 6 := by
  sorry


end NUMINAMATH_CALUDE_dishes_to_equalize_is_six_l2349_234958


namespace NUMINAMATH_CALUDE_power_sum_problem_l2349_234922

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : 2*a*x + 3*b*y = 6)
  (h2 : 2*a*x^2 + 3*b*y^2 = 14)
  (h3 : 2*a*x^3 + 3*b*y^3 = 33)
  (h4 : 2*a*x^4 + 3*b*y^4 = 87) :
  2*a*x^5 + 3*b*y^5 = 528 := by
sorry

end NUMINAMATH_CALUDE_power_sum_problem_l2349_234922


namespace NUMINAMATH_CALUDE_quadrilateral_I_greater_than_II_l2349_234992

/-- Quadrilateral I with vertices at (0,0), (2,0), (2,1), and (0,1) -/
def quadrilateral_I : List (ℝ × ℝ) := [(0,0), (2,0), (2,1), (0,1)]

/-- Quadrilateral II with vertices at (0,0), (1,0), (1,1), (0,2) -/
def quadrilateral_II : List (ℝ × ℝ) := [(0,0), (1,0), (1,1), (0,2)]

/-- Calculate the area of a quadrilateral given its vertices -/
def area (vertices : List (ℝ × ℝ)) : ℝ := sorry

/-- Calculate the perimeter of a quadrilateral given its vertices -/
def perimeter (vertices : List (ℝ × ℝ)) : ℝ := sorry

theorem quadrilateral_I_greater_than_II :
  area quadrilateral_I > area quadrilateral_II ∧
  perimeter quadrilateral_I > perimeter quadrilateral_II :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_I_greater_than_II_l2349_234992


namespace NUMINAMATH_CALUDE_tetrahedron_inequality_l2349_234971

/-- Represents a tetrahedron with base edge lengths a, b, c, 
    lateral edge lengths x, y, z, and d being the distance from 
    the top vertex to the centroid of the base. -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  d : ℝ

/-- Theorem stating that for any tetrahedron, the sum of lateral edge lengths
    is less than or equal to the sum of base edge lengths plus three times
    the distance from the top vertex to the centroid of the base. -/
theorem tetrahedron_inequality (t : Tetrahedron) : 
  t.x + t.y + t.z ≤ t.a + t.b + t.c + 3 * t.d := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_inequality_l2349_234971


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2349_234979

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := fun x ↦ 2 * x^2 - x - 1
  {x : ℝ | f x > 0} = {x : ℝ | x < -1/2 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2349_234979


namespace NUMINAMATH_CALUDE_room_painting_time_l2349_234909

theorem room_painting_time 
  (alice_rate : ℝ) 
  (bob_rate : ℝ) 
  (carla_rate : ℝ) 
  (t : ℝ) 
  (h_alice : alice_rate = 1 / 6) 
  (h_bob : bob_rate = 1 / 8) 
  (h_carla : carla_rate = 1 / 12) 
  (h_combined_work : (alice_rate + bob_rate + carla_rate) * t = 1) : 
  (1 / 6 + 1 / 8 + 1 / 12) * t = 1 := by
  sorry

end NUMINAMATH_CALUDE_room_painting_time_l2349_234909


namespace NUMINAMATH_CALUDE_perpendicular_to_plane_implies_parallel_l2349_234976

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_plane_implies_parallel
  (m n : Line) (α : Plane) 
  (hm : m ≠ n)
  (hα : perpendicular m α)
  (hβ : perpendicular n α) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_plane_implies_parallel_l2349_234976


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l2349_234918

theorem cookie_boxes_problem (n : ℕ) : 
  (∃ (mark_sold ann_sold : ℕ), 
    mark_sold = n - 9 ∧ 
    ann_sold = n - 2 ∧ 
    mark_sold ≥ 1 ∧ 
    ann_sold ≥ 1 ∧ 
    mark_sold + ann_sold < n) ↔ 
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_cookie_boxes_problem_l2349_234918


namespace NUMINAMATH_CALUDE_exists_expression_equal_100_l2349_234939

/-- Represents a sequence of digits with operators between them -/
inductive DigitExpression
  | single : Nat → DigitExpression
  | add : DigitExpression → DigitExpression → DigitExpression
  | sub : DigitExpression → DigitExpression → DigitExpression

/-- Evaluates a DigitExpression to its integer value -/
def evaluate : DigitExpression → Int
  | DigitExpression.single n => n
  | DigitExpression.add a b => evaluate a + evaluate b
  | DigitExpression.sub a b => evaluate a - evaluate b

/-- Checks if a DigitExpression uses the digits 1 to 9 in order -/
def usesDigitsInOrder : DigitExpression → Bool := sorry

/-- The main theorem stating that there exists a valid expression equaling 100 -/
theorem exists_expression_equal_100 : 
  ∃ (expr : DigitExpression), usesDigitsInOrder expr ∧ evaluate expr = 100 := by
  sorry

end NUMINAMATH_CALUDE_exists_expression_equal_100_l2349_234939


namespace NUMINAMATH_CALUDE_cubic_function_constant_term_l2349_234989

/-- Given a cubic function with specific properties, prove that the constant term is 16 -/
theorem cubic_function_constant_term (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f : ℤ → ℤ := λ x => x^3 + a*x^2 + b*x + c
  (f a = a^3) ∧ (f b = b^3) → c = 16 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_constant_term_l2349_234989


namespace NUMINAMATH_CALUDE_road_length_l2349_234947

/-- Proves that a road with streetlights installed every 10 meters on both sides, 
    with a total of 120 streetlights, is 590 meters long. -/
theorem road_length (streetlight_interval : Nat) (total_streetlights : Nat) (road_length : Nat) : 
  streetlight_interval = 10 → 
  total_streetlights = 120 → 
  road_length = (total_streetlights / 2 - 1) * streetlight_interval → 
  road_length = 590 :=
by sorry

end NUMINAMATH_CALUDE_road_length_l2349_234947


namespace NUMINAMATH_CALUDE_min_toggles_for_uniform_l2349_234916

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)
  (initialPattern : Bool) -- true for alternating, false for uniform

/-- Represents a rectangular region on the chessboard -/
structure Rectangle :=
  (top : ℕ)
  (left : ℕ)
  (bottom : ℕ)
  (right : ℕ)

/-- Function to toggle colors in a rectangle -/
def toggleColors (board : Chessboard) (rect : Rectangle) : Chessboard :=
  sorry

/-- Function to check if the board is uniform in color -/
def isUniform (board : Chessboard) : Bool :=
  sorry

/-- Theorem stating the minimum number of toggles required -/
theorem min_toggles_for_uniform (board : Chessboard) :
  board.size = 98 →
  board.initialPattern = true →
  ∃ (toggles : List Rectangle),
    (toggles.length = 98) ∧
    (isUniform (toggles.foldl toggleColors board)) ∧
    (∀ (otherToggles : List Rectangle),
      (otherToggles.length < 98) →
      ¬(isUniform (otherToggles.foldl toggleColors board))) :=
  sorry

end NUMINAMATH_CALUDE_min_toggles_for_uniform_l2349_234916


namespace NUMINAMATH_CALUDE_quadratic_roots_greater_than_half_l2349_234985

theorem quadratic_roots_greater_than_half (a : ℝ) :
  (∀ x : ℝ, (2 - a) * x^2 - 3 * a * x + 2 * a = 0 → x > (1/2 : ℝ)) ↔ 16/17 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_greater_than_half_l2349_234985


namespace NUMINAMATH_CALUDE_power_equation_solver_l2349_234953

theorem power_equation_solver (m : ℕ) : 5^m = 5 * 25^3 * 125^2 → m = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solver_l2349_234953


namespace NUMINAMATH_CALUDE_project_completion_time_l2349_234910

theorem project_completion_time 
  (initial_team : ℕ) 
  (initial_work : ℚ) 
  (initial_time : ℕ) 
  (additional_members : ℕ) 
  (total_team : ℕ) :
  initial_team = 8 →
  initial_work = 1/3 →
  initial_time = 30 →
  additional_members = 4 →
  total_team = initial_team + additional_members →
  let work_efficiency := initial_work / (initial_team * initial_time)
  let remaining_work := 1 - initial_work
  let remaining_time := remaining_work / (total_team * work_efficiency)
  initial_time + remaining_time = 70 := by
sorry

end NUMINAMATH_CALUDE_project_completion_time_l2349_234910


namespace NUMINAMATH_CALUDE_bobs_weight_l2349_234977

theorem bobs_weight (j b : ℝ) : 
  j + b = 200 → 
  b - 3 * j = b / 4 → 
  b = 2400 / 14 := by
sorry

end NUMINAMATH_CALUDE_bobs_weight_l2349_234977


namespace NUMINAMATH_CALUDE_junior_has_sixteen_rabbits_l2349_234912

/-- The number of toys bought on Monday -/
def monday_toys : ℕ := 6

/-- The number of toys bought on Wednesday -/
def wednesday_toys : ℕ := 2 * monday_toys

/-- The number of toys bought on Friday -/
def friday_toys : ℕ := 4 * monday_toys

/-- The number of toys bought on Saturday -/
def saturday_toys : ℕ := wednesday_toys / 2

/-- The total number of toys -/
def total_toys : ℕ := monday_toys + wednesday_toys + friday_toys + saturday_toys

/-- The number of toys each rabbit receives -/
def toys_per_rabbit : ℕ := 3

/-- The number of rabbits Junior has -/
def num_rabbits : ℕ := total_toys / toys_per_rabbit

theorem junior_has_sixteen_rabbits : num_rabbits = 16 := by
  sorry

end NUMINAMATH_CALUDE_junior_has_sixteen_rabbits_l2349_234912


namespace NUMINAMATH_CALUDE_f_of_g_3_l2349_234960

def g (x : ℝ) : ℝ := 4 * x + 5

def f (x : ℝ) : ℝ := 6 * x - 11

theorem f_of_g_3 : f (g 3) = 91 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_3_l2349_234960


namespace NUMINAMATH_CALUDE_min_values_problem_l2349_234923

theorem min_values_problem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a + b = 1) :
  (∀ x y : ℝ, x > y ∧ y > 0 ∧ x + y = 1 → a^2 + 2*b^2 ≤ x^2 + 2*y^2) ∧
  (∀ x y : ℝ, x > y ∧ y > 0 ∧ x + y = 1 → 4 / (a - b) + 1 / (2*b) ≤ 4 / (x - y) + 1 / (2*y)) ∧
  a^2 + 2*b^2 = 2/3 ∧
  4 / (a - b) + 1 / (2*b) = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_values_problem_l2349_234923


namespace NUMINAMATH_CALUDE_cd_length_ratio_l2349_234904

/-- Given three CDs, where two have the same length and the total length of all CDs is known,
    this theorem proves the ratio of the length of the third CD to one of the first two. -/
theorem cd_length_ratio (length_first_two : ℝ) (total_length : ℝ) : 
  length_first_two > 0 →
  total_length > 2 * length_first_two →
  (total_length - 2 * length_first_two) / length_first_two = 2 := by
  sorry

#check cd_length_ratio

end NUMINAMATH_CALUDE_cd_length_ratio_l2349_234904


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l2349_234901

theorem green_shirt_pairs (red_students green_students total_students total_pairs red_red_pairs : ℕ) 
  (h1 : red_students = 63)
  (h2 : green_students = 69)
  (h3 : total_students = red_students + green_students)
  (h4 : total_pairs = 66)
  (h5 : red_red_pairs = 26)
  (h6 : total_students = 2 * total_pairs) :
  green_students - (total_pairs - red_red_pairs - (red_students - 2 * red_red_pairs)) = 29 := by
  sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l2349_234901


namespace NUMINAMATH_CALUDE_platform_length_l2349_234948

/-- Calculates the length of a platform given the speed of a train, time to cross the platform,
    and the length of the train. -/
theorem platform_length (train_speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) :
  train_speed = 72 * (5 / 18) →  -- Convert km/hr to m/s
  crossing_time = 26 →
  train_length = 440 →
  train_speed * crossing_time - train_length = 80 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l2349_234948


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2349_234949

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-2, x)
  parallel a b → x = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2349_234949


namespace NUMINAMATH_CALUDE_loan_duration_b_l2349_234957

/-- Proves that the loan duration for B is 2 years given the problem conditions -/
theorem loan_duration_b (principal_b : ℕ) (principal_c : ℕ) (rate : ℚ) 
  (duration_c : ℕ) (total_interest : ℕ) :
  principal_b = 5000 →
  principal_c = 3000 →
  rate = 8/100 →
  duration_c = 4 →
  total_interest = 1760 →
  ∃ (n : ℕ), n = 2 ∧ 
    (principal_b * rate * n + principal_c * rate * duration_c = total_interest) :=
by sorry

end NUMINAMATH_CALUDE_loan_duration_b_l2349_234957


namespace NUMINAMATH_CALUDE_john_uber_profit_l2349_234991

def uber_earnings : ℕ := 30000
def car_purchase_price : ℕ := 18000
def car_trade_in_value : ℕ := 6000

theorem john_uber_profit :
  uber_earnings - (car_purchase_price - car_trade_in_value) = 18000 :=
by sorry

end NUMINAMATH_CALUDE_john_uber_profit_l2349_234991


namespace NUMINAMATH_CALUDE_identity_function_divisibility_l2349_234911

theorem identity_function_divisibility (f : ℕ+ → ℕ+) :
  (∀ (a b : ℕ+), (a.val ^ 2 + (f a).val * (f b).val) % ((f a).val + b.val) = 0) →
  (∀ (n : ℕ+), f n = n) :=
by sorry

end NUMINAMATH_CALUDE_identity_function_divisibility_l2349_234911


namespace NUMINAMATH_CALUDE_pasha_wins_l2349_234987

/-- Represents the game state -/
structure GameState where
  n : ℕ  -- Number of tokens
  k : ℕ  -- Game parameter

/-- Represents a move in the game -/
inductive Move
  | pasha : Move
  | roma : Move

/-- Represents the result of the game -/
inductive GameResult
  | pashaWins : GameResult
  | romaWins : GameResult

/-- The game progression function -/
def playGame (state : GameState) (strategy : GameState → Move) : GameResult :=
  sorry

/-- Pasha's winning strategy -/
def pashaStrategy (state : GameState) : Move :=
  sorry

/-- Theorem stating that Pasha can ensure at least one token reaches the end -/
theorem pasha_wins (n k : ℕ) (h : n > k * 2^k) :
  ∃ (strategy : GameState → Move),
    playGame ⟨n, k⟩ strategy = GameResult.pashaWins :=
  sorry

end NUMINAMATH_CALUDE_pasha_wins_l2349_234987


namespace NUMINAMATH_CALUDE_bad_games_count_l2349_234961

theorem bad_games_count (total_games working_games : ℕ) 
  (h1 : total_games = 11)
  (h2 : working_games = 6) :
  total_games - working_games = 5 := by
sorry

end NUMINAMATH_CALUDE_bad_games_count_l2349_234961


namespace NUMINAMATH_CALUDE_river_depth_l2349_234941

/-- The depth of a river given its width, flow rate, and volume of water per minute -/
theorem river_depth (width : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) : 
  width = 65 →
  flow_rate = 6 →
  volume_per_minute = 26000 →
  (width * (flow_rate * 1000 / 60) * 4 = volume_per_minute) := by sorry

end NUMINAMATH_CALUDE_river_depth_l2349_234941
