import Mathlib

namespace NUMINAMATH_CALUDE_trapezoid_area_is_12_sqrt_5_l3234_323461

/-- Represents a trapezoid with given measurements -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Calculates the area of a trapezoid given its measurements -/
def area (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that a trapezoid with the given measurements has an area of 12√5 -/
theorem trapezoid_area_is_12_sqrt_5 :
  ∀ t : Trapezoid,
    t.base1 = 3 ∧
    t.base2 = 6 ∧
    t.diagonal1 = 7 ∧
    t.diagonal2 = 8 →
    area t = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_12_sqrt_5_l3234_323461


namespace NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l3234_323405

theorem least_n_with_gcd_conditions (n : ℕ) : 
  n > 1500 ∧ 
  Nat.gcd 40 (n + 105) = 10 ∧ 
  Nat.gcd (n + 40) 105 = 35 ∧
  (∀ m : ℕ, m > 1500 → Nat.gcd 40 (m + 105) = 10 → Nat.gcd (m + 40) 105 = 35 → m ≥ n) →
  n = 1511 := by
sorry

end NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l3234_323405


namespace NUMINAMATH_CALUDE_cube_sum_eq_neg_26_l3234_323402

/-- ω is a nonreal complex number that is a cube root of unity -/
def ω : ℂ :=
  sorry

/-- ω is a nonreal cube root of unity -/
axiom ω_cube_root : ω ^ 3 = 1 ∧ ω ≠ 1

/-- The main theorem to prove -/
theorem cube_sum_eq_neg_26 :
  (1 + ω + 2 * ω^2)^3 + (1 - 2*ω + ω^2)^3 = -26 :=
sorry

end NUMINAMATH_CALUDE_cube_sum_eq_neg_26_l3234_323402


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3234_323469

/-- The standard equation of a circle with center (h, k) and radius r is (x-h)^2 + (y-k)^2 = r^2 -/
def StandardCircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Prove that for a circle with center (2, -1) and radius 3, the standard equation is (x-2)^2 + (y+1)^2 = 9 -/
theorem circle_equation_proof :
  ∀ (x y : ℝ), StandardCircleEquation 2 (-1) 3 x y ↔ (x - 2)^2 + (y + 1)^2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_circle_equation_proof_l3234_323469


namespace NUMINAMATH_CALUDE_nancy_chips_to_brother_l3234_323428

def tortilla_chips_problem (total_chips : ℕ) (kept_chips : ℕ) (sister_chips : ℕ) : ℕ :=
  total_chips - kept_chips - sister_chips

theorem nancy_chips_to_brother :
  tortilla_chips_problem 22 10 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_chips_to_brother_l3234_323428


namespace NUMINAMATH_CALUDE_no_natural_solution_l3234_323493

theorem no_natural_solution :
  ¬∃ (n m : ℕ), (n + 1) * (2 * n + 1) = 2 * m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l3234_323493


namespace NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l3234_323447

/-- The total surface area of a hemisphere (excluding its base) and cylinder side surface -/
theorem hemisphere_cylinder_surface_area (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  2 * π * r * h + 2 * π * r^2 = 150 * π := by sorry

end NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l3234_323447


namespace NUMINAMATH_CALUDE_complex_exp_seven_pi_over_two_eq_i_l3234_323456

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State the theorem
theorem complex_exp_seven_pi_over_two_eq_i :
  cexp (Complex.I * (7 * Real.pi / 2)) = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exp_seven_pi_over_two_eq_i_l3234_323456


namespace NUMINAMATH_CALUDE_id_tag_problem_l3234_323481

/-- The set of characters available for creating ID tags -/
def tag_chars : Finset Char := {'M', 'A', 'T', 'H', '2', '0', '3'}

/-- The number of times '2' can appear in a tag -/
def max_twos : Nat := 2

/-- The length of each ID tag -/
def tag_length : Nat := 5

/-- The total number of unique ID tags -/
def total_tags : Nat := 3720

/-- Theorem stating the result of the ID tag problem -/
theorem id_tag_problem :
  (total_tags : ℚ) / 10 = 372 := by sorry

end NUMINAMATH_CALUDE_id_tag_problem_l3234_323481


namespace NUMINAMATH_CALUDE_students_play_both_calculation_l3234_323419

/-- Represents the number of students who play both football and cricket -/
def students_play_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

theorem students_play_both_calculation :
  students_play_both 450 325 175 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_students_play_both_calculation_l3234_323419


namespace NUMINAMATH_CALUDE_min_sum_squares_l3234_323497

/-- A line intercepted by a circle with a given chord length -/
structure LineCircleIntersection where
  /-- Coefficient of x in the line equation -/
  a : ℝ
  /-- Coefficient of y in the line equation -/
  b : ℝ
  /-- The line equation: ax + 2by - 4 = 0 -/
  line_eq : ∀ (x y : ℝ), a * x + 2 * b * y - 4 = 0
  /-- The circle equation: x^2 + y^2 + 4x - 2y + 1 = 0 -/
  circle_eq : ∀ (x y : ℝ), x^2 + y^2 + 4*x - 2*y + 1 = 0
  /-- The chord length of the intersection is 4 -/
  chord_length : ℝ
  chord_length_eq : chord_length = 4

/-- The minimum value of a^2 + b^2 for a LineCircleIntersection is 2 -/
theorem min_sum_squares (lci : LineCircleIntersection) : 
  ∃ (m : ℝ), (∀ (a b : ℝ), a^2 + b^2 ≥ m) ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3234_323497


namespace NUMINAMATH_CALUDE_house_wall_nails_l3234_323434

/-- The number of nails needed for large planks -/
def large_planks_nails : ℕ := 15

/-- The number of nails needed for small planks -/
def small_planks_nails : ℕ := 5

/-- The total number of nails needed for the house wall -/
def total_nails : ℕ := large_planks_nails + small_planks_nails

theorem house_wall_nails : total_nails = 20 := by
  sorry

end NUMINAMATH_CALUDE_house_wall_nails_l3234_323434


namespace NUMINAMATH_CALUDE_total_shot_cost_l3234_323420

-- Define the given conditions
def pregnant_dogs : ℕ := 3
def puppies_per_dog : ℕ := 4
def shots_per_puppy : ℕ := 2
def cost_per_shot : ℕ := 5

-- Define the theorem
theorem total_shot_cost : 
  pregnant_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_shot_cost_l3234_323420


namespace NUMINAMATH_CALUDE_valid_param_iff_l3234_323495

/-- A parameterization of a line in 2D space -/
structure LineParam where
  x₀ : ℝ
  y₀ : ℝ
  dx : ℝ
  dy : ℝ

/-- The line equation y = 2x - 4 -/
def line_eq (x y : ℝ) : Prop := y = 2 * x - 4

/-- A parameterization is valid for the line y = 2x - 4 -/
def is_valid_param (p : LineParam) : Prop :=
  line_eq p.x₀ p.y₀ ∧ p.dy = 2 * p.dx

theorem valid_param_iff (p : LineParam) :
  is_valid_param p ↔ 
  (∀ t : ℝ, line_eq (p.x₀ + t * p.dx) (p.y₀ + t * p.dy)) :=
sorry

end NUMINAMATH_CALUDE_valid_param_iff_l3234_323495


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l3234_323455

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem complement_intersection_problem : (Mᶜ ∩ N) = {-3, -4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l3234_323455


namespace NUMINAMATH_CALUDE_davids_math_marks_l3234_323437

/-- Represents the marks obtained in each subject -/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average marks -/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.chemistry + m.biology) / 5

/-- Theorem: Given David's marks in other subjects and his average, his Mathematics marks must be 65 -/
theorem davids_math_marks (m : Marks) : 
  m.english = 51 → 
  m.physics = 82 → 
  m.chemistry = 67 → 
  m.biology = 85 → 
  average m = 70 → 
  m.mathematics = 65 := by
  sorry

#eval average { english := 51, mathematics := 65, physics := 82, chemistry := 67, biology := 85 }

end NUMINAMATH_CALUDE_davids_math_marks_l3234_323437


namespace NUMINAMATH_CALUDE_stating_spheres_fit_funnel_iff_l3234_323413

/-- Represents a conical funnel with two spheres inside it -/
structure ConicalFunnelWithSpheres where
  α : ℝ  -- Half of the axial section angle
  R : ℝ  -- Radius of the larger sphere
  r : ℝ  -- Radius of the smaller sphere
  h_angle_positive : 0 < α
  h_angle_less_than_pi_half : α < π / 2
  h_R_positive : 0 < R
  h_r_positive : 0 < r
  h_R_greater_r : r < R

/-- 
The necessary and sufficient condition for two spheres to be placed in a conical funnel 
such that they both touch its lateral surface
-/
def spheres_fit_condition (funnel : ConicalFunnelWithSpheres) : Prop :=
  Real.sin funnel.α ≤ (funnel.R - funnel.r) / funnel.R

/-- 
Theorem stating the necessary and sufficient condition for two spheres 
to fit in a conical funnel touching its lateral surface
-/
theorem spheres_fit_funnel_iff (funnel : ConicalFunnelWithSpheres) :
  (∃ (pos_R pos_r : ℝ), 
    pos_R > 0 ∧ pos_r > 0 ∧ pos_R = funnel.R ∧ pos_r = funnel.r ∧
    (∃ (config : ℝ × ℝ), 
      (config.1 > 0 ∧ config.2 > 0) ∧
      (config.1 + pos_R) * Real.sin funnel.α = pos_R ∧
      (config.2 + pos_r) * Real.sin funnel.α = pos_r ∧
      config.1 + pos_R + pos_r = config.2)) ↔
  spheres_fit_condition funnel :=
sorry

end NUMINAMATH_CALUDE_stating_spheres_fit_funnel_iff_l3234_323413


namespace NUMINAMATH_CALUDE_beta_still_water_speed_l3234_323453

/-- Represents a boat with its speed in still water -/
structure Boat where
  speed : ℝ

/-- Represents the river with its current speed -/
structure River where
  currentSpeed : ℝ

/-- Represents a journey on the river -/
inductive Direction
  | Upstream
  | Downstream

def effectiveSpeed (b : Boat) (r : River) (d : Direction) : ℝ :=
  match d with
  | Direction.Upstream => b.speed + r.currentSpeed
  | Direction.Downstream => b.speed - r.currentSpeed

theorem beta_still_water_speed 
  (alpha : Boat)
  (beta : Boat)
  (river : River)
  (h1 : alpha.speed = 56)
  (h2 : beta.speed = 52)
  (h3 : river.currentSpeed = 4)
  (h4 : effectiveSpeed alpha river Direction.Upstream / effectiveSpeed beta river Direction.Downstream = 5 / 4)
  (h5 : effectiveSpeed alpha river Direction.Downstream / effectiveSpeed beta river Direction.Upstream = 4 / 5) :
  beta.speed = 61 := by
  sorry

end NUMINAMATH_CALUDE_beta_still_water_speed_l3234_323453


namespace NUMINAMATH_CALUDE_min_a_value_l3234_323454

-- Define the function representing the left side of the inequality
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|

-- State the theorem
theorem min_a_value (a : ℝ) : 
  (∃ x : ℝ, f x a ≤ 8) → 
  (∀ b : ℝ, (∃ x : ℝ, f x b ≤ 8) → a ≤ b) → 
  a = -9 := by
sorry

end NUMINAMATH_CALUDE_min_a_value_l3234_323454


namespace NUMINAMATH_CALUDE_m_values_l3234_323425

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem m_values : 
  {m : ℝ | B m ⊆ A} = {1/3, -1/2} := by sorry

end NUMINAMATH_CALUDE_m_values_l3234_323425


namespace NUMINAMATH_CALUDE_water_amount_is_150_l3234_323438

/-- Represents the ratios of bleach, detergent, and water in a solution --/
structure SolutionRatio where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- The original ratio of the solution --/
def original_ratio : SolutionRatio := ⟨4, 40, 100⟩

/-- The altered ratio after tripling bleach to detergent and halving detergent to water --/
def altered_ratio : SolutionRatio :=
  let b := original_ratio.bleach * 3
  let d := original_ratio.detergent
  let w := original_ratio.water / 2
  ⟨b, d, w⟩

/-- The amount of detergent in the altered solution --/
def altered_detergent_amount : ℚ := 60

/-- Calculates the amount of water in the altered solution --/
def water_amount : ℚ :=
  altered_detergent_amount * (altered_ratio.water / altered_ratio.detergent)

/-- Theorem stating that the amount of water in the altered solution is 150 liters --/
theorem water_amount_is_150 : water_amount = 150 := by sorry

end NUMINAMATH_CALUDE_water_amount_is_150_l3234_323438


namespace NUMINAMATH_CALUDE_subset_sum_divisible_by_p_l3234_323468

theorem subset_sum_divisible_by_p (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) :
  let S := Finset.range (2 * p)
  (S.powerset.filter (fun A => A.card = p ∧ (A.sum id) % p = 0)).card =
    (Nat.choose (2 * p) p - 2) / p + 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_sum_divisible_by_p_l3234_323468


namespace NUMINAMATH_CALUDE_trig_equation_solution_range_l3234_323448

theorem trig_equation_solution_range :
  ∀ m : ℝ, 
  (∀ x : ℝ, ∃ y : ℝ, 4 * Real.cos y + Real.sin y ^ 2 + m - 4 = 0) ↔ 
  (0 ≤ m ∧ m ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_range_l3234_323448


namespace NUMINAMATH_CALUDE_employee_age_when_hired_l3234_323490

/-- Rule of 70 provision for retirement eligibility -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year the employee was hired -/
def hire_year : ℕ := 1986

/-- The year the employee became eligible for retirement -/
def retirement_year : ℕ := 2006

/-- The age of the employee when hired -/
def age_when_hired : ℕ := 50

theorem employee_age_when_hired :
  rule_of_70 age_when_hired (retirement_year - hire_year) ∧
  age_when_hired = 70 - (retirement_year - hire_year) := by
  sorry

end NUMINAMATH_CALUDE_employee_age_when_hired_l3234_323490


namespace NUMINAMATH_CALUDE_bookseller_sales_l3234_323452

/-- Bookseller's monthly sales problem -/
theorem bookseller_sales 
  (b1 b2 b3 b4 : ℕ) 
  (h1 : b1 + b2 + b3 = 45)
  (h2 : b4 = (3 * (b1 + b2)) / 4)
  (h3 : (b1 + b2 + b3 + b4) / 4 = 18) :
  b3 = 9 ∧ b1 + b2 = 36 ∧ b4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_bookseller_sales_l3234_323452


namespace NUMINAMATH_CALUDE_minimal_sequence_property_l3234_323488

def F_p (p : ℕ) : Set (ℕ → ℕ) :=
  {a | ∀ n > 0, a (n + 1) = (p + 1) * a n - p * a (n - 1) ∧ ∀ n, a n ≥ 0}

def minimal_sequence (p : ℕ) (n : ℕ) : ℕ :=
  (p^n - 1) / (p - 1)

theorem minimal_sequence_property (p : ℕ) (hp : p > 1) :
  minimal_sequence p ∈ F_p p ∧
  ∀ b ∈ F_p p, ∀ n, minimal_sequence p n ≤ b n :=
by sorry

end NUMINAMATH_CALUDE_minimal_sequence_property_l3234_323488


namespace NUMINAMATH_CALUDE_beef_price_per_pound_l3234_323464

/-- The price of beef per pound given the total cost, number of packs, and weight per pack -/
def price_per_pound (total_cost : ℚ) (num_packs : ℕ) (weight_per_pack : ℚ) : ℚ :=
  total_cost / (num_packs * weight_per_pack)

/-- Theorem: The price of beef per pound is $5.50 -/
theorem beef_price_per_pound :
  price_per_pound 110 5 4 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_beef_price_per_pound_l3234_323464


namespace NUMINAMATH_CALUDE_room_volume_l3234_323479

/-- Given a room with length three times its breadth, height twice its breadth,
    and floor area of 12 sq.m, prove that its volume is 48 cubic meters. -/
theorem room_volume (b l h : ℝ) (h1 : l = 3 * b) (h2 : h = 2 * b) (h3 : l * b = 12) :
  l * b * h = 48 := by
  sorry

end NUMINAMATH_CALUDE_room_volume_l3234_323479


namespace NUMINAMATH_CALUDE_books_left_l3234_323478

theorem books_left (initial_books given_away : ℝ) 
  (h1 : initial_books = 54.0)
  (h2 : given_away = 23.0) : 
  initial_books - given_away = 31.0 := by
sorry

end NUMINAMATH_CALUDE_books_left_l3234_323478


namespace NUMINAMATH_CALUDE_binomial_12_9_l3234_323458

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_9_l3234_323458


namespace NUMINAMATH_CALUDE_panda_weekly_consumption_l3234_323430

/-- The total bamboo consumption for a group of pandas in a week -/
def panda_bamboo_consumption 
  (small_pandas : ℕ) 
  (big_pandas : ℕ) 
  (small_daily_consumption : ℕ) 
  (big_daily_consumption : ℕ) : ℕ :=
  ((small_pandas * small_daily_consumption + big_pandas * big_daily_consumption) * 7)

/-- Theorem: The total bamboo consumption for 4 small pandas and 5 big pandas in a week is 2100 pounds -/
theorem panda_weekly_consumption : 
  panda_bamboo_consumption 4 5 25 40 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_panda_weekly_consumption_l3234_323430


namespace NUMINAMATH_CALUDE_exterior_angle_sum_l3234_323476

/-- In a triangle ABC, the exterior angle α at vertex A is equal to the sum of the two non-adjacent interior angles B and C. -/
theorem exterior_angle_sum (A B C : Real) (α : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → α = B + C :=
by sorry

end NUMINAMATH_CALUDE_exterior_angle_sum_l3234_323476


namespace NUMINAMATH_CALUDE_inverse_variation_sqrt_l3234_323429

/-- Given that z varies inversely as √w, prove that w = 64 when z = 2, 
    given that z = 8 when w = 4. -/
theorem inverse_variation_sqrt (z w : ℝ) (h : ∃ k : ℝ, ∀ w z, z * Real.sqrt w = k) :
  (∃ z₀ w₀, z₀ = 8 ∧ w₀ = 4 ∧ z₀ * Real.sqrt w₀ = 8 * Real.sqrt 4) →
  (∃ w₁, 2 * Real.sqrt w₁ = 8 * Real.sqrt 4 ∧ w₁ = 64) :=
by sorry


end NUMINAMATH_CALUDE_inverse_variation_sqrt_l3234_323429


namespace NUMINAMATH_CALUDE_susan_walk_distance_l3234_323445

/-- Given two people walking together for a total of 15 miles, where one person walks 3 miles less
    than the other, prove that the person who walked more covered 9 miles. -/
theorem susan_walk_distance (susan_distance erin_distance : ℝ) :
  susan_distance + erin_distance = 15 →
  erin_distance = susan_distance - 3 →
  susan_distance = 9 := by
sorry

end NUMINAMATH_CALUDE_susan_walk_distance_l3234_323445


namespace NUMINAMATH_CALUDE_fourteen_binary_l3234_323422

/-- The binary representation of a natural number -/
def binary_repr (n : ℕ) : List Bool :=
  sorry

theorem fourteen_binary : binary_repr 14 = [true, true, true, false] := by
  sorry

end NUMINAMATH_CALUDE_fourteen_binary_l3234_323422


namespace NUMINAMATH_CALUDE_determinant_solution_l3234_323470

/-- Given a ≠ 0 and b ≠ 0, the solution to the determinant equation is (3b^2 + ab) / (a + b) -/
theorem determinant_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let x := (3 * b^2 + a * b) / (a + b)
  (x + a) * ((x + b) * (2 * x) - x * (2 * x)) -
  x * (x * (2 * x) - x * (2 * x + a + b)) +
  x * (x * (2 * x) - (x + b) * (2 * x + a + b)) = 0 := by
  sorry

#check determinant_solution

end NUMINAMATH_CALUDE_determinant_solution_l3234_323470


namespace NUMINAMATH_CALUDE_certain_chinese_book_l3234_323491

def total_books : ℕ := 12
def chinese_books : ℕ := 10
def math_books : ℕ := 2
def drawn_books : ℕ := 3

theorem certain_chinese_book :
  ∀ (drawn : Finset ℕ),
    drawn.card = drawn_books →
    drawn ⊆ Finset.range total_books →
    ∃ (book : ℕ), book ∈ drawn ∧ book < chinese_books :=
sorry

end NUMINAMATH_CALUDE_certain_chinese_book_l3234_323491


namespace NUMINAMATH_CALUDE_bicycle_price_increase_l3234_323418

theorem bicycle_price_increase (initial_price : ℝ) (first_increase : ℝ) (second_increase : ℝ) :
  initial_price = 220 →
  first_increase = 0.08 →
  second_increase = 0.10 →
  let price_after_first := initial_price * (1 + first_increase)
  let final_price := price_after_first * (1 + second_increase)
  final_price = 261.36 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_increase_l3234_323418


namespace NUMINAMATH_CALUDE_remainder_x2023_plus_1_l3234_323457

theorem remainder_x2023_plus_1 (x : ℂ) : 
  (x^2023 + 1) % (x^6 - x^4 + x^2 - 1) = -x^3 + 1 := by sorry

end NUMINAMATH_CALUDE_remainder_x2023_plus_1_l3234_323457


namespace NUMINAMATH_CALUDE_little_john_money_distribution_l3234_323435

/-- Calculates the amount Little John gave to each of his two friends -/
def money_given_to_each_friend (initial_amount : ℚ) (spent_on_sweets : ℚ) (amount_left : ℚ) : ℚ :=
  (initial_amount - spent_on_sweets - amount_left) / 2

/-- Proves that Little John gave $2.20 to each of his two friends -/
theorem little_john_money_distribution :
  money_given_to_each_friend 10.50 2.25 3.85 = 2.20 := by
  sorry

#eval money_given_to_each_friend 10.50 2.25 3.85

end NUMINAMATH_CALUDE_little_john_money_distribution_l3234_323435


namespace NUMINAMATH_CALUDE_rent_share_ratio_l3234_323480

theorem rent_share_ratio (purity_share : ℚ) (rose_share : ℚ) (total_rent : ℚ) :
  rose_share = 1800 →
  total_rent = 5400 →
  total_rent = 5 * purity_share + purity_share + rose_share →
  rose_share / purity_share = 3 := by
  sorry

end NUMINAMATH_CALUDE_rent_share_ratio_l3234_323480


namespace NUMINAMATH_CALUDE_chinese_riddle_championship_arrangement_l3234_323450

theorem chinese_riddle_championship_arrangement (n : ℕ) (students : ℕ) (teacher : ℕ) (parents : ℕ) :
  n = 6 →
  students = 3 →
  teacher = 1 →
  parents = 2 →
  (students.factorial * 2 * (n - students - 1).factorial) = 72 :=
by sorry

end NUMINAMATH_CALUDE_chinese_riddle_championship_arrangement_l3234_323450


namespace NUMINAMATH_CALUDE_order_of_ab_squared_a_ab_l3234_323498

theorem order_of_ab_squared_a_ab (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) : 
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end NUMINAMATH_CALUDE_order_of_ab_squared_a_ab_l3234_323498


namespace NUMINAMATH_CALUDE_probability_specific_arrangement_l3234_323412

def num_letters : ℕ := 8

theorem probability_specific_arrangement (n : ℕ) (h : n = num_letters) :
  (1 : ℚ) / (n.factorial : ℚ) = 1 / 40320 :=
sorry

end NUMINAMATH_CALUDE_probability_specific_arrangement_l3234_323412


namespace NUMINAMATH_CALUDE_cherry_pie_degrees_l3234_323499

/-- Calculates the number of degrees for cherry pie in a pie chart given the class preferences. -/
theorem cherry_pie_degrees (total_students : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ)
  (h1 : total_students = 48)
  (h2 : chocolate = 15)
  (h3 : apple = 10)
  (h4 : blueberry = 9)
  (h5 : (total_students - (chocolate + apple + blueberry)) % 2 = 0) :
  (((total_students - (chocolate + apple + blueberry)) / 2 : ℚ) / total_students) * 360 = 52.5 := by
  sorry

#eval ((7 : ℚ) / 48) * 360  -- Should output 52.5

end NUMINAMATH_CALUDE_cherry_pie_degrees_l3234_323499


namespace NUMINAMATH_CALUDE_total_value_is_correct_l3234_323408

/-- The number of £5 notes issued by the Bank of England -/
def num_notes : ℕ := 440000000

/-- The face value of each note in pounds -/
def face_value : ℕ := 5

/-- The total face value of all notes in pounds -/
def total_value : ℕ := num_notes * face_value

/-- Theorem: The total face value of all notes is £2,200,000,000 -/
theorem total_value_is_correct : total_value = 2200000000 := by
  sorry

end NUMINAMATH_CALUDE_total_value_is_correct_l3234_323408


namespace NUMINAMATH_CALUDE_exists_m_for_second_quadrant_l3234_323466

/-- A point is in the second quadrant if its x-coordinate is negative and y-coordinate is positive -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point P -/
def x_coord (m : ℝ) : ℝ := m - 1

/-- The y-coordinate of point P -/
def y_coord : ℝ := 3

theorem exists_m_for_second_quadrant : 
  ∃ m : ℝ, m < 1 ∧ is_in_second_quadrant (x_coord m) y_coord :=
sorry

end NUMINAMATH_CALUDE_exists_m_for_second_quadrant_l3234_323466


namespace NUMINAMATH_CALUDE_line_equation_with_x_intercept_and_slope_angle_l3234_323432

theorem line_equation_with_x_intercept_and_slope_angle 
  (x_intercept : ℝ) 
  (slope_angle : ℝ) 
  (h1 : x_intercept = 2) 
  (h2 : slope_angle = 135) :
  ∃ (m b : ℝ), ∀ x y : ℝ, y = m * x + b ∧ 
    (x = x_intercept ∧ y = 0) ∧ 
    m = Real.tan (π - slope_angle * π / 180) ∧
    y = -x + 2 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_with_x_intercept_and_slope_angle_l3234_323432


namespace NUMINAMATH_CALUDE_shreehari_pencils_l3234_323427

/-- Calculates the minimum number of pencils initially possessed given the number of students and pencils per student. -/
def min_initial_pencils (num_students : ℕ) (pencils_per_student : ℕ) : ℕ :=
  num_students * pencils_per_student

/-- Proves that given 25 students and 5 pencils per student, the minimum number of pencils initially possessed is 125. -/
theorem shreehari_pencils : min_initial_pencils 25 5 = 125 := by
  sorry

end NUMINAMATH_CALUDE_shreehari_pencils_l3234_323427


namespace NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l3234_323485

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem sum_of_fourth_and_fifth_terms :
  let a₁ : ℝ := 4096
  let r : ℝ := 1/4
  (geometric_sequence a₁ r 4) + (geometric_sequence a₁ r 5) = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l3234_323485


namespace NUMINAMATH_CALUDE_jimmys_cabin_friends_l3234_323496

def hostel_stay_days : ℕ := 3
def hostel_cost_per_night : ℕ := 15
def cabin_stay_days : ℕ := 2
def cabin_cost_per_night : ℕ := 45
def total_lodging_cost : ℕ := 75

theorem jimmys_cabin_friends :
  ∃ (n : ℕ), 
    hostel_stay_days * hostel_cost_per_night + 
    cabin_stay_days * (cabin_cost_per_night / (n + 1)) = total_lodging_cost ∧
    n = 2 := by
  sorry

end NUMINAMATH_CALUDE_jimmys_cabin_friends_l3234_323496


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3234_323414

theorem inequality_equivalence (b c : ℝ) : 
  (∀ x : ℝ, |2*x - 3| < 5 ↔ -x^2 + b*x + c > 0) → b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3234_323414


namespace NUMINAMATH_CALUDE_largest_m_bound_l3234_323473

theorem largest_m_bound (x y z t : ℕ+) (h1 : x + y = z + t) (h2 : 2 * x * y = z * t) (h3 : x ≥ y) :
  ∃ (m : ℝ), m = 3 + 2 * Real.sqrt 2 ∧ 
  (∀ (m' : ℝ), (∀ (x' y' z' t' : ℕ+), 
    x' + y' = z' + t' → 2 * x' * y' = z' * t' → x' ≥ y' → 
    (x' : ℝ) / (y' : ℝ) ≥ m') → m' ≤ m) := by
  sorry

end NUMINAMATH_CALUDE_largest_m_bound_l3234_323473


namespace NUMINAMATH_CALUDE_shaded_probability_is_half_l3234_323417

/-- Represents a game board with an equilateral triangle -/
structure GameBoard where
  /-- The number of regions the triangle is divided into -/
  total_regions : ℕ
  /-- The number of shaded regions -/
  shaded_regions : ℕ
  /-- Proof that the total number of regions is 6 -/
  total_is_six : total_regions = 6
  /-- Proof that the number of shaded regions is 3 -/
  shaded_is_three : shaded_regions = 3

/-- The probability of the spinner landing in a shaded region -/
def shaded_probability (board : GameBoard) : ℚ :=
  board.shaded_regions / board.total_regions

/-- Theorem stating that the probability of landing in a shaded region is 1/2 -/
theorem shaded_probability_is_half (board : GameBoard) :
  shaded_probability board = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_probability_is_half_l3234_323417


namespace NUMINAMATH_CALUDE_expression_evaluation_l3234_323433

theorem expression_evaluation (a b c : ℚ) (ha : a = 14) (hb : b = 19) (hc : c = 23) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

#eval (14 : ℚ) + 19 + 23

end NUMINAMATH_CALUDE_expression_evaluation_l3234_323433


namespace NUMINAMATH_CALUDE_no_real_roots_l3234_323489

theorem no_real_roots : ¬ ∃ x : ℝ, x^2 - x + 2 = 0 := by sorry

end NUMINAMATH_CALUDE_no_real_roots_l3234_323489


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3234_323460

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 → (x + y : ℕ) ≤ (a + b : ℕ)) → 
  (x + y : ℕ) = 49 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3234_323460


namespace NUMINAMATH_CALUDE_range_of_a_l3234_323421

theorem range_of_a (a : ℝ) (h_a_pos : a > 0) : 
  (∀ m : ℝ, (3 * a < m ∧ m < 4 * a) → (1 < m ∧ m < 3/2)) →
  (1/3 ≤ a ∧ a ≤ 3/8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3234_323421


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3234_323426

theorem negation_of_universal_proposition (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x₀ : ℝ, f x₀ ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3234_323426


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l3234_323472

theorem sin_cos_sixth_power_sum (α : Real) (h : Real.cos (2 * α) = 1 / 5) :
  Real.sin α ^ 6 + Real.cos α ^ 6 = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l3234_323472


namespace NUMINAMATH_CALUDE_initial_manager_percentage_l3234_323446

/-- The initial percentage of managers in a room with 200 employees,
    given that 99.99999999999991 managers leave and the resulting
    percentage is 98%, is approximately 99%. -/
theorem initial_manager_percentage :
  let total_employees : ℕ := 200
  let managers_who_left : ℝ := 99.99999999999991
  let final_percentage : ℝ := 98
  let initial_percentage : ℝ := 
    ((managers_who_left + (final_percentage / 100) * (total_employees - managers_who_left)) / total_employees) * 100
  ∀ ε > 0, |initial_percentage - 99| < ε :=
by sorry


end NUMINAMATH_CALUDE_initial_manager_percentage_l3234_323446


namespace NUMINAMATH_CALUDE_stratified_sampling_young_employees_l3234_323424

theorem stratified_sampling_young_employees 
  (total_employees : ℕ) 
  (young_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 200) 
  (h2 : young_employees = 120) 
  (h3 : sample_size = 25) :
  ↑sample_size * (↑young_employees / ↑total_employees) = 15 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_young_employees_l3234_323424


namespace NUMINAMATH_CALUDE_solution_set_l3234_323477

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (3 - m, 1)

-- Define the condition for P being in the second quadrant
def is_in_second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0

-- Define the inequality
def inequality (m x : ℝ) : Prop := (2 - m) * x + m > 2

theorem solution_set (m : ℝ) : 
  is_in_second_quadrant (P m) → 
  (∀ x : ℝ, inequality m x ↔ x < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_l3234_323477


namespace NUMINAMATH_CALUDE_equation_solutions_l3234_323482

theorem equation_solutions :
  (∃ x : ℚ, 5 * x - 9 = 3 * x - 16 ∧ x = -7/2) ∧
  (∃ x : ℚ, (3 * x - 1) / 3 = 1 - (x + 2) / 4 ∧ x = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3234_323482


namespace NUMINAMATH_CALUDE_club_officer_selection_count_l3234_323416

/-- Represents a club with members of two genders -/
structure Club where
  total_members : Nat
  boys : Nat
  girls : Nat

/-- Calculates the number of ways to select a president, vice-president, and secretary -/
def selectOfficers (club : Club) : Nat :=
  club.total_members * club.boys * (club.total_members - 2)

/-- The theorem to prove -/
theorem club_officer_selection_count (club : Club) 
  (h1 : club.total_members = 30)
  (h2 : club.boys = 15)
  (h3 : club.girls = 15)
  (h4 : club.total_members = club.boys + club.girls) :
  selectOfficers club = 12600 := by
  sorry

#eval selectOfficers { total_members := 30, boys := 15, girls := 15 }

end NUMINAMATH_CALUDE_club_officer_selection_count_l3234_323416


namespace NUMINAMATH_CALUDE_surface_area_of_problem_lshape_l3234_323467

/-- Represents the L-shaped structure formed by unit cubes -/
structure LShape where
  base_length : Nat
  top_length : Nat
  top_start_position : Nat

/-- Calculates the surface area of the L-shaped structure -/
def surface_area (l : LShape) : Nat :=
  let base_visible_top := l.base_length - l.top_length
  let base_visible_sides := 2 * l.base_length
  let base_visible_ends := 2
  let top_visible_top := l.top_length
  let top_visible_sides := 2 * l.top_length
  let top_visible_ends := 2
  base_visible_top + base_visible_sides + base_visible_ends +
  top_visible_top + top_visible_sides + top_visible_ends

/-- The specific L-shaped structure described in the problem -/
def problem_lshape : LShape :=
  { base_length := 10
    top_length := 5
    top_start_position := 5 }

theorem surface_area_of_problem_lshape :
  surface_area problem_lshape = 45 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_problem_lshape_l3234_323467


namespace NUMINAMATH_CALUDE_share_calculation_l3234_323436

theorem share_calculation (total : ℚ) (a b c : ℚ) 
  (h1 : total = 510)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) : b = 90 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l3234_323436


namespace NUMINAMATH_CALUDE_rationalize_denominator_sum_l3234_323400

theorem rationalize_denominator_sum :
  ∃ (A B C D E F : ℤ),
    (F > 0) ∧
    (∀ (x : ℝ), x > 0 → (1 / (Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 11) = 
      (A * Real.sqrt 2 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F)) ∧
    (A + B + C + D + E + F = 136) :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sum_l3234_323400


namespace NUMINAMATH_CALUDE_couscous_dishes_l3234_323475

/-- Calculates the number of dishes a restaurant can make from couscous shipments -/
theorem couscous_dishes (shipment1 shipment2 shipment3 pounds_per_dish : ℕ) :
  shipment1 = 7 →
  shipment2 = 13 →
  shipment3 = 45 →
  pounds_per_dish = 5 →
  (shipment1 + shipment2 + shipment3) / pounds_per_dish = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_couscous_dishes_l3234_323475


namespace NUMINAMATH_CALUDE_steves_coins_l3234_323487

theorem steves_coins (total_coins : ℕ) (nickel_value dime_value : ℕ) (swap_increase : ℕ) :
  total_coins = 30 →
  nickel_value = 5 →
  dime_value = 10 →
  swap_increase = 120 →
  ∃ (nickels dimes : ℕ),
    nickels + dimes = total_coins ∧
    dimes * dime_value + nickels * nickel_value + swap_increase = nickels * dime_value + dimes * nickel_value →
    dimes * dime_value + nickels * nickel_value = 165 :=
by sorry

end NUMINAMATH_CALUDE_steves_coins_l3234_323487


namespace NUMINAMATH_CALUDE_range_of_a_l3234_323404

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - a| > 5) ↔ (a > 8 ∨ a < -2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3234_323404


namespace NUMINAMATH_CALUDE_sallys_car_fuel_efficiency_l3234_323409

/-- Calculates the fuel efficiency of Sally's car given her trip expenses and savings --/
theorem sallys_car_fuel_efficiency :
  ∀ (savings : ℝ) (parking : ℝ) (entry : ℝ) (meal : ℝ) (distance : ℝ) (gas_price : ℝ) (additional_savings : ℝ),
    savings = 28 →
    parking = 10 →
    entry = 55 →
    meal = 25 →
    distance = 165 →
    gas_price = 3 →
    additional_savings = 95 →
    (2 * distance) / ((savings + additional_savings - (parking + entry + meal)) / gas_price) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_sallys_car_fuel_efficiency_l3234_323409


namespace NUMINAMATH_CALUDE_salt_solution_dilution_l3234_323459

def initial_solution_volume : ℝ := 40
def initial_salt_concentration : ℝ := 0.15
def target_salt_concentration : ℝ := 0.10
def water_added : ℝ := 20

theorem salt_solution_dilution :
  let initial_salt_amount : ℝ := initial_solution_volume * initial_salt_concentration
  let final_solution_volume : ℝ := initial_solution_volume + water_added
  let final_salt_concentration : ℝ := initial_salt_amount / final_solution_volume
  final_salt_concentration = target_salt_concentration := by sorry

end NUMINAMATH_CALUDE_salt_solution_dilution_l3234_323459


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l3234_323410

theorem triangle_sine_inequality (a b c : Real) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h3 : a + b + c ≤ 2 * Real.pi) : 
  Real.sin a + Real.sin b > Real.sin c ∧
  Real.sin b + Real.sin c > Real.sin a ∧
  Real.sin c + Real.sin a > Real.sin b :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l3234_323410


namespace NUMINAMATH_CALUDE_unique_positive_number_l3234_323401

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x^2 = (Real.sqrt 16)^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l3234_323401


namespace NUMINAMATH_CALUDE_tangent_points_line_passes_through_fixed_point_l3234_323406

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Tangent line from a point to a parabola -/
def tangent_line (p : Parabola) (m : Point) : Line :=
  sorry

/-- Tangent point of a line to a parabola -/
def tangent_point (p : Parabola) (l : Line) : Point :=
  sorry

/-- Given a parabola C, a line l, and a point M on l, prove that the line AB 
    formed by the tangent points A and B of the tangent lines from M to C 
    always passes through a fixed point -/
theorem tangent_points_line_passes_through_fixed_point 
  (C : Parabola) 
  (l : Line) 
  (M : Point) 
  (m : ℝ) 
  (h1 : C.equation = fun x y => x^2 = 4*y) 
  (h2 : l.equation = fun x y => y = -m) 
  (h3 : m > 0) 
  (h4 : l.equation M.x M.y) :
  let t1 := tangent_line C M
  let t2 := tangent_line C M
  let A := tangent_point C t1
  let B := tangent_point C t2
  let AB : Line := sorry
  AB.equation 0 m := by sorry

end NUMINAMATH_CALUDE_tangent_points_line_passes_through_fixed_point_l3234_323406


namespace NUMINAMATH_CALUDE_total_passengers_per_hour_l3234_323403

/-- Calculates the total number of different passengers stepping on and off trains at a station within an hour -/
theorem total_passengers_per_hour 
  (train_interval : ℕ) 
  (passengers_leaving : ℕ) 
  (passengers_boarding : ℕ) 
  (hour_in_minutes : ℕ) :
  train_interval = 5 →
  passengers_leaving = 200 →
  passengers_boarding = 320 →
  hour_in_minutes = 60 →
  (hour_in_minutes / train_interval) * (passengers_leaving + passengers_boarding) = 6240 := by
  sorry

end NUMINAMATH_CALUDE_total_passengers_per_hour_l3234_323403


namespace NUMINAMATH_CALUDE_unique_solution_l3234_323415

theorem unique_solution (m n : ℕ+) 
  (eq : 2 * m.val + 3 = 5 * n.val - 2)
  (ineq : 5 * n.val - 2 < 15) :
  m.val = 5 ∧ n.val = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l3234_323415


namespace NUMINAMATH_CALUDE_cos_value_third_quadrant_l3234_323462

theorem cos_value_third_quadrant (θ : Real) :
  tanθ = Real.sqrt 2 / 4 →
  θ > π ∧ θ < 3 * π / 2 →
  cosθ = -2 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_value_third_quadrant_l3234_323462


namespace NUMINAMATH_CALUDE_masha_number_l3234_323411

theorem masha_number (x y : ℕ) : 
  (x + y = 2002 ∨ x * y = 2002) →
  (∀ a : ℕ, (a + y = 2002 ∨ a * y = 2002) → ∃ b ≠ y, (a + b = 2002 ∨ a * b = 2002)) →
  (∀ a : ℕ, (x + a = 2002 ∨ x * a = 2002) → ∃ b ≠ x, (b + a = 2002 ∨ b * a = 2002)) →
  max x y = 1001 :=
by sorry

end NUMINAMATH_CALUDE_masha_number_l3234_323411


namespace NUMINAMATH_CALUDE_dragons_games_count_l3234_323484

theorem dragons_games_count :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (60 * initial_games) / 100 →
    ∃ (total_games : ℕ),
      total_games = initial_games + 11 ∧
      (initial_wins + 8 : ℚ) / total_games = 55 / 100 ∧
      total_games = 50 :=
by sorry

end NUMINAMATH_CALUDE_dragons_games_count_l3234_323484


namespace NUMINAMATH_CALUDE_target_breaking_permutations_l3234_323442

theorem target_breaking_permutations :
  let total_targets : ℕ := 10
  let column_a_targets : ℕ := 4
  let column_b_targets : ℕ := 4
  let column_c_targets : ℕ := 2
  (column_a_targets + column_b_targets + column_c_targets = total_targets) →
  (Nat.factorial total_targets) / 
  (Nat.factorial column_a_targets * Nat.factorial column_b_targets * Nat.factorial column_c_targets) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_target_breaking_permutations_l3234_323442


namespace NUMINAMATH_CALUDE_happy_snakes_not_purple_l3234_323492

structure Snake where
  purple : Bool
  happy : Bool
  can_add : Bool
  can_subtract : Bool

def Tom's_collection : Set Snake := sorry

theorem happy_snakes_not_purple :
  ∀ (s : Snake),
  s ∈ Tom's_collection →
  (s.happy → s.can_add) ∧
  (s.purple → ¬s.can_subtract) ∧
  (¬s.can_subtract → ¬s.can_add) →
  (s.happy → ¬s.purple) := by
  sorry

#check happy_snakes_not_purple

end NUMINAMATH_CALUDE_happy_snakes_not_purple_l3234_323492


namespace NUMINAMATH_CALUDE_rational_coefficient_terms_count_rational_coefficient_terms_count_is_126_l3234_323441

theorem rational_coefficient_terms_count : ℕ :=
  let expansion := (λ x y : ℝ => (x * (2 ^ (1/4 : ℝ)) + y * (5 ^ (1/2 : ℝ))) ^ 500)
  let total_terms := 501
  let is_rational_coeff := λ k : ℕ => (k % 4 = 0) ∧ ((500 - k) % 2 = 0)
  (Finset.range total_terms).filter is_rational_coeff |>.card

/-- The number of terms with rational coefficients in the expansion of (x∗∜2+y∗√5)^500 is 126 -/
theorem rational_coefficient_terms_count_is_126 : 
  rational_coefficient_terms_count = 126 := by sorry

end NUMINAMATH_CALUDE_rational_coefficient_terms_count_rational_coefficient_terms_count_is_126_l3234_323441


namespace NUMINAMATH_CALUDE_victors_marks_percentage_l3234_323431

/-- Given that Victor scored 240 marks out of a maximum of 300 marks,
    prove that the percentage of marks he got is 80%. -/
theorem victors_marks_percentage (marks_obtained : ℝ) (maximum_marks : ℝ) 
    (h1 : marks_obtained = 240)
    (h2 : maximum_marks = 300) :
    (marks_obtained / maximum_marks) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_victors_marks_percentage_l3234_323431


namespace NUMINAMATH_CALUDE_expression_evaluation_l3234_323463

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (3 + Real.sqrt 3)⁻¹ + (Real.sqrt 3 - 3)⁻¹ = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3234_323463


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3234_323443

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (2 / (x + 3))
  else if x < -3 then Int.floor (2 / (x + 3))
  else 0  -- Arbitrary value for x = -3, as g is not defined there

theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3234_323443


namespace NUMINAMATH_CALUDE_constant_function_m_values_l3234_323486

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x - |x + 1|

-- State the theorem
theorem constant_function_m_values
  (m : ℝ)
  (h_exists : ∃ (a b : ℝ), -2 ≤ a ∧ a < b ∧
    ∀ x, x ∈ Set.Icc a b → ∃ c, f m x = c) :
  m = 1 ∨ m = -1 := by
sorry

end NUMINAMATH_CALUDE_constant_function_m_values_l3234_323486


namespace NUMINAMATH_CALUDE_max_sum_at_five_l3234_323444

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The theorem stating when S_n reaches its maximum value -/
theorem max_sum_at_five (seq : ArithmeticSequence)
    (h1 : seq.a 1 + seq.a 5 + seq.a 9 = 6)
    (h2 : seq.S 11 = -11) :
    ∃ n_max : ℕ, n_max = 5 ∧ ∀ n : ℕ, seq.S n ≤ seq.S n_max := by
  sorry

end NUMINAMATH_CALUDE_max_sum_at_five_l3234_323444


namespace NUMINAMATH_CALUDE_merchant_profit_l3234_323440

theorem merchant_profit (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) :
  markup_percentage = 0.40 →
  discount_percentage = 0.10 →
  let marked_price := cost_price * (1 + markup_percentage)
  let selling_price := marked_price * (1 - discount_percentage)
  let profit := selling_price - cost_price
  let profit_percentage := profit / cost_price
  profit_percentage = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_l3234_323440


namespace NUMINAMATH_CALUDE_money_calculation_l3234_323407

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def totalMoney (n50 : ℕ) (n500 : ℕ) : ℕ :=
  50 * n50 + 500 * n500

/-- Proves that the total amount of money is 10350 rupees given the specified conditions -/
theorem money_calculation :
  ∀ (n50 n500 : ℕ),
    n50 = 37 →
    n50 + n500 = 54 →
    totalMoney n50 n500 = 10350 := by
  sorry

#eval totalMoney 37 17  -- Should output 10350

end NUMINAMATH_CALUDE_money_calculation_l3234_323407


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l3234_323451

theorem rectangle_side_ratio (a b c d : ℝ) (h : a / c = b / d ∧ a / c = 4 / 5) : b / d = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l3234_323451


namespace NUMINAMATH_CALUDE_oscar_review_questions_l3234_323483

/-- The total number of questions Professor Oscar must review -/
def total_questions (questions_per_exam : ℕ) (num_classes : ℕ) (students_per_class : ℕ) : ℕ :=
  questions_per_exam * num_classes * students_per_class

/-- Proof that Professor Oscar must review 1750 questions -/
theorem oscar_review_questions :
  total_questions 10 5 35 = 1750 := by
  sorry

end NUMINAMATH_CALUDE_oscar_review_questions_l3234_323483


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3234_323494

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 1) :
  Real.sqrt (((a.1 + 2 * b.1) ^ 2) + ((a.2 + 2 * b.2) ^ 2)) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3234_323494


namespace NUMINAMATH_CALUDE_place_value_ratio_l3234_323474

theorem place_value_ratio : 
  let number : ℝ := 37492.1053
  let ten_thousands_place_value : ℝ := 10000
  let ten_thousandths_place_value : ℝ := 0.0001
  ten_thousands_place_value / ten_thousandths_place_value = 100000000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l3234_323474


namespace NUMINAMATH_CALUDE_sine_function_period_l3234_323471

/-- 
Given a function y = A sin(Bx + C) + D, where A, B, C, and D are constants,
if the graph covers two periods in an interval of 4π, then B = 1.
-/
theorem sine_function_period (A B C D : ℝ) : 
  (∃ (a b : ℝ), b - a = 4 * π ∧ 
    (∀ x ∈ Set.Icc a b, ∃ k : ℤ, A * Real.sin (B * x + C) + D = A * Real.sin (B * (x + 2 * k * π / B) + C) + D)) →
  B = 1 := by
sorry

end NUMINAMATH_CALUDE_sine_function_period_l3234_323471


namespace NUMINAMATH_CALUDE_problem_statement_l3234_323465

theorem problem_statement (a : ℝ) (h : a^2 + a - 1 = 0) :
  2 * a^2 + 2 * a + 2008 = 2010 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3234_323465


namespace NUMINAMATH_CALUDE_pencil_rows_l3234_323423

/-- Given a total number of pencils and the number of pencils per row,
    calculate the number of complete rows that can be formed. -/
def calculate_rows (total_pencils : ℕ) (pencils_per_row : ℕ) : ℕ :=
  total_pencils / pencils_per_row

/-- Theorem stating that 30 pencils arranged in rows of 5 will form 6 complete rows -/
theorem pencil_rows : calculate_rows 30 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencil_rows_l3234_323423


namespace NUMINAMATH_CALUDE_no_valid_schedule_l3234_323439

theorem no_valid_schedule : ¬∃ (a b : ℕ+), (29 ∣ a) ∧ (32 ∣ b) ∧ (a + b = 29 * 32) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_schedule_l3234_323439


namespace NUMINAMATH_CALUDE_cut_rectangle_properties_l3234_323449

/-- Represents a rectangle cut into four pieces by two equal diagonals intersecting at right angles -/
structure CutRectangle where
  width : ℝ
  height : ℝ
  diag_intersect_center : Bool
  diag_right_angle : Bool
  diag_equal_length : Bool

/-- Theorem about properties of a specific cut rectangle -/
theorem cut_rectangle_properties (rect : CutRectangle) 
  (h_width : rect.width = 20)
  (h_height : rect.height = 30)
  (h_center : rect.diag_intersect_center = true)
  (h_right : rect.diag_right_angle = true)
  (h_equal : rect.diag_equal_length = true) :
  ∃ (square_side triangle_area pentagon_area hole_area : ℝ),
    square_side = 20 ∧
    triangle_area = 100 ∧
    pentagon_area = 200 ∧
    hole_area = 200 := by
  sorry


end NUMINAMATH_CALUDE_cut_rectangle_properties_l3234_323449
