import Mathlib

namespace NUMINAMATH_CALUDE_faulty_meter_profit_percent_l2294_229425

/-- The profit percentage for a shopkeeper using a faulty meter -/
theorem faulty_meter_profit_percent (actual_weight : ℝ) (expected_weight : ℝ) :
  actual_weight = 960 →
  expected_weight = 1000 →
  (1 - actual_weight / expected_weight) * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_faulty_meter_profit_percent_l2294_229425


namespace NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l2294_229477

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l2294_229477


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2294_229455

theorem gcd_of_specific_numbers : 
  let m : ℕ := 3333333
  let n : ℕ := 66666666
  gcd m n = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2294_229455


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_nine_l2294_229410

theorem greatest_integer_with_gcf_nine (n : ℕ) : 
  n < 200 ∧ 
  Nat.gcd n 72 = 9 ∧ 
  (∀ m : ℕ, m < 200 ∧ Nat.gcd m 72 = 9 → m ≤ n) → 
  n = 189 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_nine_l2294_229410


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_coefficient_x_squared_in_expansion_proof_l2294_229433

/-- The coefficient of x^2 in the expansion of (x + 2/x^2)^5 is 10 -/
theorem coefficient_x_squared_in_expansion : ℕ :=
  let expansion := (fun x => (x + 2 / x^2)^5)
  let coefficient_x_squared := 10
  coefficient_x_squared

/-- Proof of the theorem -/
theorem coefficient_x_squared_in_expansion_proof :
  coefficient_x_squared_in_expansion = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_coefficient_x_squared_in_expansion_proof_l2294_229433


namespace NUMINAMATH_CALUDE_percentage_calculation_l2294_229431

theorem percentage_calculation (whole : ℝ) (part : ℝ) (h1 : whole = 200) (h2 : part = 50) :
  (part / whole) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2294_229431


namespace NUMINAMATH_CALUDE_zach_current_tickets_l2294_229434

def ferris_wheel_cost : ℕ := 2
def roller_coaster_cost : ℕ := 7
def log_ride_cost : ℕ := 1
def additional_tickets_needed : ℕ := 9

def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + log_ride_cost

theorem zach_current_tickets : total_cost - additional_tickets_needed = 1 := by
  sorry

end NUMINAMATH_CALUDE_zach_current_tickets_l2294_229434


namespace NUMINAMATH_CALUDE_complex_exp_angle_l2294_229405

theorem complex_exp_angle (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → ∃ r θ : ℝ, z = r * Complex.exp (Complex.I * θ) ∧ θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_angle_l2294_229405


namespace NUMINAMATH_CALUDE_triangle_inequality_bounds_l2294_229416

theorem triangle_inequality_bounds (a b c : ℝ) 
  (triangle_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (sum_two : a + b + c = 2) :
  1 ≤ a * b + b * c + c * a - a * b * c ∧ 
  a * b + b * c + c * a - a * b * c ≤ 1 + 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_bounds_l2294_229416


namespace NUMINAMATH_CALUDE_operation_result_l2294_229400

theorem operation_result (x : ℕ) (h : x = 40) : (((x / 4) * 5) + 10) - 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l2294_229400


namespace NUMINAMATH_CALUDE_fraction_value_l2294_229453

theorem fraction_value (x y : ℝ) (h1 : 2 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 5) (h3 : ∃ (n : ℤ), x / y = n) : 
  x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2294_229453


namespace NUMINAMATH_CALUDE_no_triple_exists_l2294_229450

theorem no_triple_exists : ¬∃ (a b c : ℕ+), 
  let p := (a.val - 2) * (b.val - 2) * (c.val - 2) + 12
  Nat.Prime p ∧ 
  (∃ k : ℕ+, k * p = a.val^2 + b.val^2 + c.val^2 + a.val * b.val * c.val - 2017) ∧
  p < a.val^2 + b.val^2 + c.val^2 + a.val * b.val * c.val - 2017 :=
by sorry

end NUMINAMATH_CALUDE_no_triple_exists_l2294_229450


namespace NUMINAMATH_CALUDE_tan_inequality_l2294_229469

theorem tan_inequality (h1 : 130 * π / 180 > π / 2) (h2 : 130 * π / 180 < π)
                       (h3 : 140 * π / 180 > π / 2) (h4 : 140 * π / 180 < π) :
  Real.tan (130 * π / 180) < Real.tan (140 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tan_inequality_l2294_229469


namespace NUMINAMATH_CALUDE_equidistant_complex_function_d_squared_l2294_229480

/-- A complex function g(z) = (c+di)z with the property that g(z) is equidistant from z and the origin -/
def equidistant_complex_function (c d : ℝ) : ℂ → ℂ := λ z ↦ (c + d * Complex.I) * z

/-- The property that g(z) is equidistant from z and the origin for all z -/
def is_equidistant (g : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs (g z - z) = Complex.abs (g z)

theorem equidistant_complex_function_d_squared 
  (c d : ℝ) 
  (h1 : is_equidistant (equidistant_complex_function c d))
  (h2 : Complex.abs (c + d * Complex.I) = 7) : 
  d^2 = 195/4 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_complex_function_d_squared_l2294_229480


namespace NUMINAMATH_CALUDE_chord_bisection_l2294_229489

/-- The ellipse defined by x²/16 + y²/8 = 1 -/
def Ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 8 = 1

/-- A point (x, y) lies on the line x + y - 3 = 0 -/
def Line (x y : ℝ) : Prop := x + y - 3 = 0

/-- The midpoint of two points -/
def Midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

theorem chord_bisection (x₁ y₁ x₂ y₂ : ℝ) :
  Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧ Midpoint x₁ y₁ x₂ y₂ 2 1 →
  Line x₁ y₁ ∧ Line x₂ y₂ := by
  sorry

end NUMINAMATH_CALUDE_chord_bisection_l2294_229489


namespace NUMINAMATH_CALUDE_max_product_fg_l2294_229452

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the conditions on the ranges of f and g
axiom f_range : ∀ x, -3 ≤ f x ∧ f x ≤ 4
axiom g_range : ∀ x, -3 ≤ g x ∧ g x ≤ 2

-- Theorem stating the maximum value of f(x) · g(x)
theorem max_product_fg : 
  ∃ x : ℝ, ∀ y : ℝ, f y * g y ≤ f x * g x ∧ f x * g x = 12 :=
sorry

end NUMINAMATH_CALUDE_max_product_fg_l2294_229452


namespace NUMINAMATH_CALUDE_copper_ion_beakers_l2294_229401

theorem copper_ion_beakers (total_beakers : ℕ) (drops_per_test : ℕ) (total_drops_used : ℕ) (negative_beakers : ℕ) : 
  total_beakers = 22 → 
  drops_per_test = 3 → 
  total_drops_used = 45 → 
  negative_beakers = 7 → 
  total_beakers - negative_beakers = 15 := by
sorry

end NUMINAMATH_CALUDE_copper_ion_beakers_l2294_229401


namespace NUMINAMATH_CALUDE_train_length_l2294_229444

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 144 → time_s = 2.7497800175985923 → 
  speed_kmh * (5/18) * time_s = 110.9912007039437 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2294_229444


namespace NUMINAMATH_CALUDE_parabola_one_x_intercept_parabola_x_intercepts_parabola_opens_upward_l2294_229459

-- Define the parabola
def parabola (a c x : ℝ) : ℝ := a * x^2 + 2 * a * x + c

-- Theorem 1: If a = c, the parabola has only one point in common with the x-axis
theorem parabola_one_x_intercept (a : ℝ) (h : a ≠ 0) :
  ∃! x, parabola a a x = 0 :=
sorry

-- Theorem 2: If the x-intercepts satisfy 1/x₁ + 1/x₂ = 1, then c = -2a
theorem parabola_x_intercepts (a c x₁ x₂ : ℝ) (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0)
  (h₃ : parabola a c x₁ = 0) (h₄ : parabola a c x₂ = 0) (h₅ : 1/x₁ + 1/x₂ = 1) :
  c = -2 * a :=
sorry

-- Theorem 3: If (m,p) lies on y = -ax + c - 2a, -2 < m < -1, and p > n where (m,n) is on the parabola, then a > 0
theorem parabola_opens_upward (a c m : ℝ) (h₁ : -2 < m) (h₂ : m < -1)
  (h₃ : -a * m + c - 2 * a > parabola a c m) :
  a > 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_one_x_intercept_parabola_x_intercepts_parabola_opens_upward_l2294_229459


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l2294_229491

/-- Given two points A and B in the plane, this theorem states that
    the equation 4x - 2y - 5 = 0 represents the perpendicular bisector
    of the line segment connecting A and B. -/
theorem perpendicular_bisector_equation (A B : ℝ × ℝ) :
  A = (1, 2) →
  B = (3, 1) →
  ∀ (x y : ℝ), (4 * x - 2 * y - 5 = 0) ↔
    (((x - 1)^2 + (y - 2)^2 = (x - 3)^2 + (y - 1)^2) ∧
     ((y - 2) * (3 - 1) = -(x - 1) * (1 - 2))) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l2294_229491


namespace NUMINAMATH_CALUDE_inverse_power_function_at_4_l2294_229494

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- State the theorem
theorem inverse_power_function_at_4 (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) :
  Function.invFun f 4 = 16 := by
sorry

end NUMINAMATH_CALUDE_inverse_power_function_at_4_l2294_229494


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2294_229472

theorem mean_equality_implies_z_value :
  let x₁ : ℚ := 7
  let x₂ : ℚ := 15
  let x₃ : ℚ := 21
  let y₁ : ℚ := 18
  (x₁ + x₂ + x₃) / 3 = (y₁ + z) / 2 →
  z = 32 / 3 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2294_229472


namespace NUMINAMATH_CALUDE_radical_axis_through_intersection_points_l2294_229426

-- Define a circle with center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the power of a point with respect to a circle
def powerOfPoint (p : ℝ × ℝ) (c : Circle) : ℝ :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 - c.radius^2

-- Define the radical axis of two circles
def radicalAxis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | powerOfPoint p c1 = powerOfPoint p c2}

-- Define the intersection points of two circles
def intersectionPoints (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | powerOfPoint p c1 = 0 ∧ powerOfPoint p c2 = 0}

-- Theorem statement
theorem radical_axis_through_intersection_points (c1 c2 : Circle) :
  intersectionPoints c1 c2 ⊆ radicalAxis c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_radical_axis_through_intersection_points_l2294_229426


namespace NUMINAMATH_CALUDE_expression_evaluation_l2294_229482

theorem expression_evaluation : 
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2294_229482


namespace NUMINAMATH_CALUDE_correct_calculation_l2294_229457

theorem correct_calculation (x : ℤ) (h : x + 54 = 78) : x + 45 = 69 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2294_229457


namespace NUMINAMATH_CALUDE_anthony_pizza_fraction_l2294_229451

theorem anthony_pizza_fraction (total_slices : ℕ) (whole_slice : ℚ) (shared_slice : ℚ) : 
  total_slices = 16 → 
  whole_slice = 1 / total_slices → 
  shared_slice = 1 / (2 * total_slices) → 
  whole_slice + 2 * shared_slice = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_anthony_pizza_fraction_l2294_229451


namespace NUMINAMATH_CALUDE_social_practice_choices_l2294_229481

/-- The number of classes in the first year of high school -/
def first_year_classes : Nat := 14

/-- The number of classes in the second year of high school -/
def second_year_classes : Nat := 14

/-- The number of classes in the third year of high school -/
def third_year_classes : Nat := 15

/-- The number of ways to choose students from 1 class to participate in social practice activities -/
def choose_one_class : Nat := first_year_classes + second_year_classes + third_year_classes

/-- The number of ways to choose students from one class in each grade to participate in social practice activities -/
def choose_one_from_each : Nat := first_year_classes * second_year_classes * third_year_classes

/-- The number of ways to choose students from 2 classes to participate in social practice activities, with the requirement that these 2 classes are from different grades -/
def choose_two_different_grades : Nat := 
  first_year_classes * second_year_classes + 
  first_year_classes * third_year_classes + 
  second_year_classes * third_year_classes

theorem social_practice_choices : 
  choose_one_class = 43 ∧ 
  choose_one_from_each = 2940 ∧ 
  choose_two_different_grades = 616 := by sorry

end NUMINAMATH_CALUDE_social_practice_choices_l2294_229481


namespace NUMINAMATH_CALUDE_overall_loss_percentage_l2294_229495

/-- Calculate the overall loss percentage for three items given their cost prices, discounts, tax, and sale prices. -/
theorem overall_loss_percentage
  (radio_cost speaker_cost headphones_cost : ℚ)
  (radio_discount speaker_discount headphones_discount : ℚ)
  (tax : ℚ)
  (radio_sale speaker_sale headphones_sale : ℚ)
  (h1 : radio_cost = 1500)
  (h2 : speaker_cost = 2500)
  (h3 : headphones_cost = 800)
  (h4 : radio_discount = 10 / 100)
  (h5 : speaker_discount = 5 / 100)
  (h6 : headphones_discount = 12 / 100)
  (h7 : tax = 15 / 100)
  (h8 : radio_sale = 1275)
  (h9 : speaker_sale = 2300)
  (h10 : headphones_sale = 700)
  : ∃ (loss_percentage : ℚ), abs (loss_percentage - 1697 / 10000) < 1 / 10000 :=
sorry

end NUMINAMATH_CALUDE_overall_loss_percentage_l2294_229495


namespace NUMINAMATH_CALUDE_sequence_property_l2294_229407

def sequence_sum (a : ℕ+ → ℚ) (n : ℕ+) : ℚ :=
  (Finset.range n).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_property (a : ℕ+ → ℚ) 
    (h : ∀ n : ℕ+, a n + sequence_sum a n = 2 * n.val + 1) :
  (∀ n : ℕ+, a n = 2 - 1 / (2 ^ n.val)) ∧
  (∀ n : ℕ+, (Finset.range n).sum (fun i => 1 / (2 ^ (i + 1) * a ⟨i + 1, Nat.succ_pos i⟩ * a ⟨i + 2, Nat.succ_pos (i + 1)⟩)) < 1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l2294_229407


namespace NUMINAMATH_CALUDE_badminton_match_duration_l2294_229456

theorem badminton_match_duration :
  ∀ (hours minutes : ℕ),
    hours = 12 ∧ minutes = 25 →
    hours * 60 + minutes = 745 := by sorry

end NUMINAMATH_CALUDE_badminton_match_duration_l2294_229456


namespace NUMINAMATH_CALUDE_combination_equality_l2294_229479

theorem combination_equality (x : ℕ) : 
  (Nat.choose 20 (2*x - 1) = Nat.choose 20 (x + 3)) ↔ (x = 4 ∨ x = 6) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l2294_229479


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_l2294_229498

theorem rectangular_prism_surface_area (r h : ℝ) : 
  r = (36 / Real.pi) ^ (1/3) → 
  (4/3) * Real.pi * r^3 = 6 * 4 * h → 
  2 * (4 * 6 + 2 * 4 + 2 * 6) = 88 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_l2294_229498


namespace NUMINAMATH_CALUDE_lcm_1008_672_l2294_229445

theorem lcm_1008_672 : Nat.lcm 1008 672 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1008_672_l2294_229445


namespace NUMINAMATH_CALUDE_exactly_one_sick_probability_l2294_229427

/-- The probability of an employee being sick on any given day -/
def prob_sick : ℚ := 1 / 40

/-- The probability of an employee not being sick on any given day -/
def prob_not_sick : ℚ := 1 - prob_sick

/-- The probability of exactly one out of three employees being sick -/
def prob_one_sick_out_of_three : ℚ :=
  3 * prob_sick * prob_not_sick * prob_not_sick

theorem exactly_one_sick_probability :
  prob_one_sick_out_of_three = 4563 / 64000 := by sorry

end NUMINAMATH_CALUDE_exactly_one_sick_probability_l2294_229427


namespace NUMINAMATH_CALUDE_problem_solution_l2294_229474

theorem problem_solution (x : ℝ) : 
  x + Real.sqrt (x^2 + 1) + 1 / (x + Real.sqrt (x^2 + 1)) = 22 →
  x^2 - Real.sqrt (x^4 + 1) + 1 / (x^2 - Real.sqrt (x^4 + 1)) = 242 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2294_229474


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2294_229423

theorem fraction_sum_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y - 1) :
  x / y + y / x = (x^2 * y^2 - 4 * x * y + 1) / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2294_229423


namespace NUMINAMATH_CALUDE_equilateral_cone_lateral_surface_angle_l2294_229418

/-- Represents a cone with an equilateral triangle as its front view -/
structure EquilateralCone where
  side_length : ℝ
  generatrix_length : ℝ
  base_radius : ℝ
  lateral_surface_angle : ℝ

/-- The properties of an EquilateralCone -/
def is_valid_equilateral_cone (c : EquilateralCone) : Prop :=
  c.generatrix_length = c.side_length ∧
  c.base_radius = c.side_length / 2 ∧
  2 * Real.pi * c.base_radius = (c.lateral_surface_angle * Real.pi * c.generatrix_length) / 180

/-- Theorem: The lateral surface angle of an EquilateralCone is 180° -/
theorem equilateral_cone_lateral_surface_angle (c : EquilateralCone) 
  (h : is_valid_equilateral_cone c) : c.lateral_surface_angle = 180 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_cone_lateral_surface_angle_l2294_229418


namespace NUMINAMATH_CALUDE_lee_surpasses_hernandez_in_may_l2294_229449

def months : List String := ["March", "April", "May", "June", "July", "August"]

def hernandez_hrs : List Nat := [4, 8, 9, 5, 7, 6]
def lee_hrs : List Nat := [3, 9, 10, 6, 8, 8]

def cumulative_sum (list : List Nat) : List Nat :=
  list.scanl (· + ·) 0

def first_surpass (list1 list2 : List Nat) : Option Nat :=
  (list1.zip list2).findIdx? (fun (a, b) => b > a)

theorem lee_surpasses_hernandez_in_may :
  first_surpass (cumulative_sum hernandez_hrs) (cumulative_sum lee_hrs) = some 2 :=
sorry

end NUMINAMATH_CALUDE_lee_surpasses_hernandez_in_may_l2294_229449


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2294_229496

theorem min_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → a - b = 1 → (∀ x y : ℝ, x > 0 → y > 0 → x - y = 1 → 1/a + 1/b ≤ 1/x + 1/y) → 
  1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2294_229496


namespace NUMINAMATH_CALUDE_factorization_equality_l2294_229468

theorem factorization_equality (a b : ℝ) : 3 * a^2 * b - 3 * a * b + 6 * b = 3 * b * (a^2 - a + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2294_229468


namespace NUMINAMATH_CALUDE_sandy_siblings_l2294_229485

def total_tokens : ℕ := 1000000
def sandy_share : ℕ := total_tokens / 2
def extra_tokens : ℕ := 375000

theorem sandy_siblings :
  ∃ (num_siblings : ℕ),
    num_siblings > 0 ∧
    sandy_share = (total_tokens - sandy_share) / num_siblings + extra_tokens ∧
    num_siblings = 4 :=
by sorry

end NUMINAMATH_CALUDE_sandy_siblings_l2294_229485


namespace NUMINAMATH_CALUDE_infinitely_many_squares_l2294_229448

theorem infinitely_many_squares (k : ℕ) (hk : k ≥ 2) :
  ∃ (f : ℕ → ℕ), Function.Injective f ∧
  ∀ (i : ℕ), ∃ (u v : ℕ), k * (f i) + 1 = u^2 ∧ (k + 1) * (f i) + 1 = v^2 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_squares_l2294_229448


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l2294_229429

theorem sum_of_roots_equation (x : ℝ) : 
  (7 = (x^3 - 2*x^2 - 8*x) / (x + 2)) → 
  (∃ y z : ℝ, x = y ∨ x = z ∧ y + z = 4) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l2294_229429


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2294_229466

/-- The asymptote equations of the hyperbola x^2 - y^2/4 = 1 are y = 2x and y = -2x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 - y^2/4 = 1) → (∃ (k : ℝ), k = 2 ∨ k = -2) ∧ (y = k*x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2294_229466


namespace NUMINAMATH_CALUDE_ghee_mixture_theorem_l2294_229463

/-- Represents the composition of a ghee mixture -/
structure GheeMixture where
  total : ℝ
  pure_ghee : ℝ
  vanaspati : ℝ

/-- The original ghee mixture before addition -/
def original_mixture : GheeMixture :=
  { total := 30
  , pure_ghee := 15
  , vanaspati := 15 }

/-- The amount of pure ghee added to the mixture -/
def added_pure_ghee : ℝ := 20

/-- The final mixture after addition of pure ghee -/
def final_mixture : GheeMixture :=
  { total := original_mixture.total + added_pure_ghee
  , pure_ghee := original_mixture.pure_ghee + added_pure_ghee
  , vanaspati := original_mixture.vanaspati }

theorem ghee_mixture_theorem :
  (original_mixture.pure_ghee = original_mixture.vanaspati) ∧
  (original_mixture.pure_ghee + original_mixture.vanaspati = original_mixture.total) ∧
  (final_mixture.vanaspati / final_mixture.total = 0.3) →
  original_mixture.total = 30 := by
  sorry

end NUMINAMATH_CALUDE_ghee_mixture_theorem_l2294_229463


namespace NUMINAMATH_CALUDE_point_on_transformed_graph_l2294_229483

/-- Given a function f where f(3) = -2, prove that (1, -5/2) lies on the graph of 4y = 2f(3x) - 6 
    and that the sum of its coordinates is -3/2 -/
theorem point_on_transformed_graph (f : ℝ → ℝ) (h : f 3 = -2) :
  let g : ℝ → ℝ := λ x => (2 * f (3 * x) - 6) / 4
  g 1 = -5/2 ∧ 1 + (-5/2) = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_point_on_transformed_graph_l2294_229483


namespace NUMINAMATH_CALUDE_set_size_from_average_change_l2294_229403

theorem set_size_from_average_change (S : Finset ℝ) (initial_avg final_avg : ℝ) :
  initial_avg = (S.sum id) / S.card →
  final_avg = ((S.sum id) + 6) / S.card →
  initial_avg = 6.2 →
  final_avg = 6.8 →
  S.card = 10 := by
  sorry

end NUMINAMATH_CALUDE_set_size_from_average_change_l2294_229403


namespace NUMINAMATH_CALUDE_distance_midpoint_problem_l2294_229492

theorem distance_midpoint_problem (t : ℝ) : 
  let A : ℝ × ℝ := (2*t - 3, 0)
  let B : ℝ × ℝ := (1, 2*t + 2)
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = 2*t^2 + 3*t
  → t = 10/7 := by
sorry

end NUMINAMATH_CALUDE_distance_midpoint_problem_l2294_229492


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2294_229430

/-- A function from nonzero reals to nonzero reals -/
def NonzeroRealFunction := ℝ* → ℝ*

/-- The property that f(x+y)(f(x) + f(y)) = f(x)f(y) for all nonzero real x and y -/
def SatisfiesProperty (f : NonzeroRealFunction) : Prop :=
  ∀ x y : ℝ*, f (x + y) * (f x + f y) = f x * f y

/-- The property that a function is increasing -/
def IsIncreasing (f : NonzeroRealFunction) : Prop :=
  ∀ x y : ℝ*, x < y → f x < f y

theorem functional_equation_solution :
  ∀ f : NonzeroRealFunction, IsIncreasing f → SatisfiesProperty f →
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ*, f x = 1 / (a * x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2294_229430


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l2294_229486

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the side lengths and angles
def sideLength (p q : ℝ × ℝ) : ℝ := sorry
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_side_lengths (t : Triangle) :
  sideLength t.A t.C = 5 →
  sideLength t.B t.C - sideLength t.A t.B = 2 →
  angle t.C t.A t.B = 2 * angle t.A t.C t.B →
  sideLength t.A t.B = 4 ∧ sideLength t.B t.C = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l2294_229486


namespace NUMINAMATH_CALUDE_sides_in_nth_figure_formula_l2294_229446

/-- The number of sides in the n-th figure of a sequence starting with a hexagon
    and increasing by 5 sides for each subsequent figure. -/
def sides_in_nth_figure (n : ℕ) : ℕ := 5 * n + 1

/-- Theorem stating that the number of sides in the n-th figure is 5n + 1 -/
theorem sides_in_nth_figure_formula (n : ℕ) :
  sides_in_nth_figure n = 5 * n + 1 := by sorry

end NUMINAMATH_CALUDE_sides_in_nth_figure_formula_l2294_229446


namespace NUMINAMATH_CALUDE_cylinder_cone_height_relation_l2294_229420

/-- Given a right cylinder and a cone with equal base radii, volumes, surface areas, 
    and heights in the ratio of 1:3, prove that the height of the cylinder 
    is 4/5 of the base radius. -/
theorem cylinder_cone_height_relation 
  (r : ℝ) -- base radius
  (h_cyl : ℝ) -- height of cylinder
  (h_cone : ℝ) -- height of cone
  (h_ratio : h_cone = 3 * h_cyl) -- height ratio condition
  (h_vol : π * r^2 * h_cyl = 1/3 * π * r^2 * h_cone) -- equal volumes
  (h_area : 2 * π * r^2 + 2 * π * r * h_cyl = 
            π * r^2 + π * r * Real.sqrt (r^2 + h_cone^2)) -- equal surface areas
  : h_cyl = 4/5 * r :=
by sorry

end NUMINAMATH_CALUDE_cylinder_cone_height_relation_l2294_229420


namespace NUMINAMATH_CALUDE_equation_solution_l2294_229476

theorem equation_solution (x y : ℕ) : 
  (x^2 + 1)^y - (x^2 - 1)^y = 2*x^y ↔ 
  (x = 1 ∧ y = 1) ∨ (x = 0 ∧ ∃ k : ℕ, y = 2*k + 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2294_229476


namespace NUMINAMATH_CALUDE_mount_tai_temp_difference_l2294_229447

/-- The temperature difference between two points is the absolute value of their difference. -/
def temperature_difference (t1 t2 : ℝ) : ℝ := |t1 - t2|

/-- The average temperature at the top of Mount Tai in January (in °C). -/
def temp_top : ℝ := -9

/-- The average temperature at the foot of Mount Tai in January (in °C). -/
def temp_foot : ℝ := -1

/-- The temperature difference between the foot and top of Mount Tai is 8°C. -/
theorem mount_tai_temp_difference : temperature_difference temp_foot temp_top = 8 := by
  sorry

end NUMINAMATH_CALUDE_mount_tai_temp_difference_l2294_229447


namespace NUMINAMATH_CALUDE_sequence_range_l2294_229461

/-- Given an infinite sequence {a_n} satisfying the recurrence relation
    a_{n+1} = p * a_n + 1 / a_n for n ∈ ℕ*, where p is a positive real number,
    a_1 = 2, and {a_n} is monotonically decreasing, prove that p ∈ (1/2, 3/4). -/
theorem sequence_range (p : ℝ) (a : ℕ+ → ℝ) 
  (h_pos : p > 0)
  (h_rec : ∀ n : ℕ+, a (n + 1) = p * a n + 1 / a n)
  (h_init : a 1 = 2)
  (h_decr : ∀ n : ℕ+, a (n + 1) ≤ a n) :
  p > 1/2 ∧ p < 3/4 := by
sorry


end NUMINAMATH_CALUDE_sequence_range_l2294_229461


namespace NUMINAMATH_CALUDE_factorization_ax2_minus_a_l2294_229422

theorem factorization_ax2_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ax2_minus_a_l2294_229422


namespace NUMINAMATH_CALUDE_soccer_ball_inflation_time_l2294_229436

/-- The time in minutes Alexia takes to inflate one soccer ball -/
def alexia_time : ℕ := 18

/-- The time in minutes Ermias takes to inflate one soccer ball -/
def ermias_time : ℕ := 25

/-- The number of soccer balls Alexia inflates -/
def alexia_balls : ℕ := 36

/-- The number of additional balls Ermias inflates compared to Alexia -/
def additional_balls : ℕ := 8

/-- The total time in minutes taken by Alexia and Ermias to inflate all soccer balls -/
def total_time : ℕ := alexia_time * alexia_balls + ermias_time * (alexia_balls + additional_balls)

theorem soccer_ball_inflation_time : total_time = 1748 := by sorry

end NUMINAMATH_CALUDE_soccer_ball_inflation_time_l2294_229436


namespace NUMINAMATH_CALUDE_point_outside_circle_l2294_229473

theorem point_outside_circle :
  let P : ℝ × ℝ := (-2, -2)
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4}
  P ∉ circle ∧ (P.1^2 + P.2^2 > 4) := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2294_229473


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2294_229458

theorem chess_tournament_games (n : ℕ) (h : n = 20) : 
  (n * (n - 1)) / 2 = 190 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2294_229458


namespace NUMINAMATH_CALUDE_machine_worked_twelve_minutes_l2294_229490

/-- An industrial machine that makes shirts -/
structure ShirtMachine where
  shirts_per_minute : ℕ
  shirts_made_today : ℕ

/-- Calculate the number of minutes the machine worked today -/
def minutes_worked_today (machine : ShirtMachine) : ℕ :=
  machine.shirts_made_today / machine.shirts_per_minute

/-- Theorem: The machine worked for 12 minutes today -/
theorem machine_worked_twelve_minutes 
  (machine : ShirtMachine) 
  (h1 : machine.shirts_per_minute = 6)
  (h2 : machine.shirts_made_today = 72) : 
  minutes_worked_today machine = 12 := by
  sorry

#eval minutes_worked_today ⟨6, 72⟩

end NUMINAMATH_CALUDE_machine_worked_twelve_minutes_l2294_229490


namespace NUMINAMATH_CALUDE_p_geq_q_l2294_229412

theorem p_geq_q (a b : ℝ) (h : a > 2) : a + 1 / (a - 2) ≥ -b^2 - 2*b + 3 := by
  sorry

end NUMINAMATH_CALUDE_p_geq_q_l2294_229412


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l2294_229478

theorem pizza_toppings_combinations (n : ℕ) (k : ℕ) : n = 5 ∧ k = 2 → Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l2294_229478


namespace NUMINAMATH_CALUDE_max_product_sum_1998_l2294_229417

theorem max_product_sum_1998 :
  ∀ x y : ℤ, x + y = 1998 → x * y ≤ 998001 :=
by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_1998_l2294_229417


namespace NUMINAMATH_CALUDE_arithmetic_sequence_61st_term_l2294_229493

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_61st_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_5 : a 5 = 33)
  (h_45 : a 45 = 153) :
  a 61 = 201 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_61st_term_l2294_229493


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_area_sum_l2294_229440

/-- A point in 2D space with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- A parallelogram defined by four points -/
structure Parallelogram where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : Int :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).natAbs

/-- Calculate the perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : Int :=
  distance p.a p.b + distance p.b p.c + distance p.c p.d + distance p.d p.a

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : Int :=
  (distance p.a p.b * distance p.b p.c).natAbs

/-- The theorem to be proved -/
theorem parallelogram_perimeter_area_sum :
  ∀ (p : Parallelogram),
    p.a = Point.mk 2 7 →
    p.b = Point.mk 7 7 →
    p.c = Point.mk 7 2 →
    perimeter p + area p = 45 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_area_sum_l2294_229440


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l2294_229424

theorem solve_fraction_equation :
  ∀ x : ℚ, (1 / 4 : ℚ) - (1 / 6 : ℚ) = 1 / x → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l2294_229424


namespace NUMINAMATH_CALUDE_x_range_for_positive_f_l2294_229439

def f (a x : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

theorem x_range_for_positive_f :
  (∀ a ∈ Set.Icc (-1) 1, ∀ x, f a x > 0) →
  {x : ℝ | x < 1 ∨ x > 3} = {x : ℝ | ∃ a ∈ Set.Icc (-1) 1, f a x > 0} :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_positive_f_l2294_229439


namespace NUMINAMATH_CALUDE_group_size_proof_l2294_229414

theorem group_size_proof (total_paise : ℕ) (contribution : ℕ → ℕ) : 
  (total_paise = 1369) →
  (∀ n : ℕ, contribution n = n) →
  (∃ n : ℕ, n * contribution n = total_paise) →
  (∃ n : ℕ, n * n = total_paise) →
  (∃ n : ℕ, n = 37) :=
by sorry

end NUMINAMATH_CALUDE_group_size_proof_l2294_229414


namespace NUMINAMATH_CALUDE_least_divisible_by_first_ten_l2294_229409

theorem least_divisible_by_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∧
  n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_first_ten_l2294_229409


namespace NUMINAMATH_CALUDE_cone_base_area_l2294_229470

/-- Represents a cone with given properties -/
structure Cone where
  slant_height : ℝ
  unfolds_to_semicircle : Prop

/-- Theorem: Area of the base of a cone with given properties -/
theorem cone_base_area (c : Cone) 
    (h1 : c.slant_height = 10)
    (h2 : c.unfolds_to_semicircle) : 
    ∃ (r : ℝ), r * r * Real.pi = 25 * Real.pi := by
  sorry

#check cone_base_area

end NUMINAMATH_CALUDE_cone_base_area_l2294_229470


namespace NUMINAMATH_CALUDE_colors_in_box_l2294_229432

/-- The number of color boxes -/
def num_boxes : ℕ := 3

/-- The total number of pencils -/
def total_pencils : ℕ := 21

/-- The number of colors in each box -/
def colors_per_box : ℕ := total_pencils / num_boxes

/-- Theorem stating that the number of colors in each box is 7 -/
theorem colors_in_box : colors_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_colors_in_box_l2294_229432


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2294_229442

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 6th term of the arithmetic sequence is 11 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_a2 : a 2 = 3)
    (h_sum : a 3 + a 5 = 14) :
  a 6 = 11 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2294_229442


namespace NUMINAMATH_CALUDE_digit_150_of_1_13_l2294_229421

def decimal_representation_1_13 : List ℕ := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_13 : 
  (decimal_representation_1_13[(150 - 1) % decimal_representation_1_13.length] = 3) := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_1_13_l2294_229421


namespace NUMINAMATH_CALUDE_bacteria_growth_30_min_l2294_229443

/-- Calculates the bacterial population after a given number of 5-minute intervals -/
def bacterial_population (initial_population : ℕ) (num_intervals : ℕ) : ℕ :=
  initial_population * (3 ^ num_intervals)

/-- Theorem stating the bacterial population after 30 minutes -/
theorem bacteria_growth_30_min (initial_population : ℕ) 
  (h1 : initial_population = 50) : 
  bacterial_population initial_population 6 = 36450 := by
  sorry

#eval bacterial_population 50 6

end NUMINAMATH_CALUDE_bacteria_growth_30_min_l2294_229443


namespace NUMINAMATH_CALUDE_remainder_theorem_l2294_229464

def polynomial (x : ℝ) : ℝ := 5*x^6 - 3*x^5 + 6*x^4 - 7*x^3 + 3*x^2 + 5*x - 14

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + 272 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2294_229464


namespace NUMINAMATH_CALUDE_binomial_10_4_l2294_229475

theorem binomial_10_4 : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_4_l2294_229475


namespace NUMINAMATH_CALUDE_sphere_only_all_circular_views_l2294_229428

/-- Enumeration of common geometric bodies -/
inductive GeometricBody
  | Cone
  | Cylinder
  | Sphere
  | HollowCylinder

/-- Definition for a view of a geometric body being circular -/
def isCircularView (body : GeometricBody) (view : String) : Prop := sorry

/-- Theorem stating that only a sphere has all circular views -/
theorem sphere_only_all_circular_views (body : GeometricBody) :
  (isCircularView body "front" ∧ 
   isCircularView body "side" ∧ 
   isCircularView body "top") ↔ 
  body = GeometricBody.Sphere :=
sorry

end NUMINAMATH_CALUDE_sphere_only_all_circular_views_l2294_229428


namespace NUMINAMATH_CALUDE_max_books_borrowed_is_ten_l2294_229411

/-- Represents the distribution of books borrowed by students in a class --/
structure BookDistribution where
  totalStudents : Nat
  zeroBooksStudents : Nat
  oneBooksStudents : Nat
  twoBooksStudents : Nat
  averageBooksPerStudent : Nat

/-- Calculates the maximum number of books a single student could have borrowed --/
def maxBooksBorrowed (d : BookDistribution) : Nat :=
  let remainingStudents := d.totalStudents - (d.zeroBooksStudents + d.oneBooksStudents + d.twoBooksStudents)
  let totalBooks := d.totalStudents * d.averageBooksPerStudent
  let accountedBooks := d.oneBooksStudents + 2 * d.twoBooksStudents
  let remainingBooks := totalBooks - accountedBooks
  let booksForOthers := (remainingStudents - 1) * 3
  3 + (remainingBooks - booksForOthers)

/-- The maximum number of books any single student could have borrowed is 10 --/
theorem max_books_borrowed_is_ten (d : BookDistribution) 
  (h1 : d.totalStudents = 40)
  (h2 : d.zeroBooksStudents = 2)
  (h3 : d.oneBooksStudents = 12)
  (h4 : d.twoBooksStudents = 14)
  (h5 : d.averageBooksPerStudent = 2) :
  maxBooksBorrowed d = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_is_ten_l2294_229411


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2294_229465

-- Define the edge lengths
def edge_length_cube1 : ℚ := 4
def edge_length_cube2 : ℚ := 24  -- 2 feet = 24 inches

-- Define the volume ratio
def volume_ratio : ℚ := (edge_length_cube1 / edge_length_cube2) ^ 3

-- Theorem statement
theorem cube_volume_ratio :
  volume_ratio = 1 / 216 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2294_229465


namespace NUMINAMATH_CALUDE_max_distance_theorem_l2294_229467

/-- Represents the characteristics of a motor boat on a river -/
structure RiverBoat where
  upstream_distance : ℝ  -- Distance the boat can travel upstream on a full tank
  downstream_distance : ℝ -- Distance the boat can travel downstream on a full tank

/-- Calculates the maximum round trip distance for a given boat -/
def max_round_trip_distance (boat : RiverBoat) : ℝ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating the maximum round trip distance for the given boat -/
theorem max_distance_theorem (boat : RiverBoat) 
  (h1 : boat.upstream_distance = 40)
  (h2 : boat.downstream_distance = 60) :
  max_round_trip_distance boat = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_theorem_l2294_229467


namespace NUMINAMATH_CALUDE_compound_weight_proof_l2294_229499

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of iodine atoms in the compound -/
def num_I_atoms : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 408

/-- Theorem stating that the molecular weight of the compound with 1 Al atom and 3 I atoms 
    is approximately equal to 408 g/mol -/
theorem compound_weight_proof : 
  ∃ ε > 0, |atomic_weight_Al + num_I_atoms * atomic_weight_I - molecular_weight| < ε :=
sorry

end NUMINAMATH_CALUDE_compound_weight_proof_l2294_229499


namespace NUMINAMATH_CALUDE_total_notebooks_is_303_l2294_229413

/-- The total number of notebooks in a classroom with specific distribution of notebooks among students. -/
def total_notebooks : ℕ :=
  let total_students : ℕ := 60
  let students_with_5 : ℕ := total_students / 4
  let students_with_3 : ℕ := total_students / 5
  let students_with_7 : ℕ := total_students / 3
  let students_with_4 : ℕ := total_students - (students_with_5 + students_with_3 + students_with_7)
  (students_with_5 * 5) + (students_with_3 * 3) + (students_with_7 * 7) + (students_with_4 * 4)

theorem total_notebooks_is_303 : total_notebooks = 303 := by
  sorry

end NUMINAMATH_CALUDE_total_notebooks_is_303_l2294_229413


namespace NUMINAMATH_CALUDE_unique_k_for_equation_l2294_229487

theorem unique_k_for_equation : ∃! k : ℕ+, 
  (∃ a b : ℕ+, a^2 + b^2 = k * a * b) ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_for_equation_l2294_229487


namespace NUMINAMATH_CALUDE_mean_of_smallest_elements_l2294_229435

/-- F(n, r) represents the arithmetic mean of the smallest elements
    in all r-element subsets of {1, 2, ..., n} -/
def F (n r : ℕ) : ℚ :=
  sorry

/-- Theorem stating that F(n, r) = (n+1)/(r+1) for 1 ≤ r ≤ n -/
theorem mean_of_smallest_elements (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) :
  F n r = (n + 1 : ℚ) / (r + 1) :=
by sorry

end NUMINAMATH_CALUDE_mean_of_smallest_elements_l2294_229435


namespace NUMINAMATH_CALUDE_possible_values_of_k_l2294_229438

def A : Set ℝ := {-1, 1}

def B (k : ℝ) : Set ℝ := {x : ℝ | k * x = 1}

theorem possible_values_of_k :
  ∀ k : ℝ, B k ⊆ A ↔ k = -1 ∨ k = 0 ∨ k = 1 :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_k_l2294_229438


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l2294_229471

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

-- Theorem statement
theorem f_composition_negative_two (f : ℝ → ℝ) :
  (∀ x ≥ 0, f x = 1 - Real.sqrt x) →
  (∀ x < 0, f x = 2^x) →
  f (f (-2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l2294_229471


namespace NUMINAMATH_CALUDE_complex_power_2016_pi_half_l2294_229497

theorem complex_power_2016_pi_half :
  let z : ℂ := Complex.exp (Complex.I * (π / 2 : ℝ))
  (z ^ 2016 : ℂ) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_2016_pi_half_l2294_229497


namespace NUMINAMATH_CALUDE_sequence_formula_l2294_229484

theorem sequence_formula (n : ℕ) : 
  Real.cos ((n + 2 : ℝ) * π / 2) = 
    if n % 4 = 0 then 1
    else if n % 4 = 1 then 0
    else if n % 4 = 2 then -1
    else 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l2294_229484


namespace NUMINAMATH_CALUDE_negation_equivalence_l2294_229460

variable (a : ℝ)

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*a*x + a > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2294_229460


namespace NUMINAMATH_CALUDE_roots_of_equation_l2294_229437

theorem roots_of_equation : 
  {x : ℝ | (18 / (x^2 - 9) - 3 / (x - 3) = 2)} = {3, -6} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2294_229437


namespace NUMINAMATH_CALUDE_complement_of_M_wrt_U_l2294_229402

open Set

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}

theorem complement_of_M_wrt_U : (U \ M) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_wrt_U_l2294_229402


namespace NUMINAMATH_CALUDE_no_integer_solution_l2294_229404

theorem no_integer_solution : 
  ¬∃ (x y : ℤ), (18 * x + 27 * y = 21) ∧ (27 * x + 18 * y = 69) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2294_229404


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l2294_229462

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x
  {x : ℝ | f x = 0} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l2294_229462


namespace NUMINAMATH_CALUDE_power_equality_l2294_229419

theorem power_equality (n : ℕ) : 3^n = 3^2 * 9^4 * 81^3 → n = 22 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2294_229419


namespace NUMINAMATH_CALUDE_square_diagonals_equal_l2294_229454

-- Define a structure for a parallelogram
structure Parallelogram :=
  (diagonals_equal : Bool)

-- Define a structure for a square that is a parallelogram
structure Square extends Parallelogram

-- State the theorem
theorem square_diagonals_equal (s : Square) : s.diagonals_equal = true := by
  sorry


end NUMINAMATH_CALUDE_square_diagonals_equal_l2294_229454


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l2294_229408

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : 4 * a^2 + b^2 + 16 * c^2 = 1) : 
  (0 < a * b ∧ a * b < 1/4) ∧ 
  (1/a^2 + 1/b^2 + 1/(4*a*b*c^2) > 49) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l2294_229408


namespace NUMINAMATH_CALUDE_xy_difference_l2294_229441

theorem xy_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_difference_l2294_229441


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2294_229406

/-- A cubic function with specific properties -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x

/-- The derivative of the cubic function -/
def f_deriv (p q : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*p*x + q

theorem cubic_function_properties (p q : ℝ) :
  (∃ a : ℝ, a ≠ 0 ∧ f p q a = 0) →  -- Intersects x-axis at non-origin point
  (∃ x_min : ℝ, f p q x_min = -4 ∧ ∀ x : ℝ, f p q x ≥ -4) →  -- Minimum y-value is -4
  p = 6 ∧ q = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2294_229406


namespace NUMINAMATH_CALUDE_lisa_interest_earned_l2294_229488

/-- UltraSavingsAccount represents the parameters of the savings account -/
structure UltraSavingsAccount where
  principal : ℝ
  rate : ℝ
  years : ℕ

/-- calculate_interest computes the interest earned for a given UltraSavingsAccount -/
def calculate_interest (account : UltraSavingsAccount) : ℝ :=
  account.principal * ((1 + account.rate) ^ account.years - 1)

/-- Theorem stating that Lisa's interest earned is $821 -/
theorem lisa_interest_earned (account : UltraSavingsAccount) 
  (h1 : account.principal = 2000)
  (h2 : account.rate = 0.035)
  (h3 : account.years = 10) :
  ⌊calculate_interest account⌋ = 821 := by
  sorry

end NUMINAMATH_CALUDE_lisa_interest_earned_l2294_229488


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l2294_229415

theorem smallest_b_for_factorization : ∃ (b : ℕ), 
  (∀ (x p q : ℤ), (x^2 + b*x + 1800 = (x + p) * (x + q)) → (p > 0 ∧ q > 0)) ∧
  (∀ (b' : ℕ), b' < b → ¬∃ (p q : ℤ), (p > 0 ∧ q > 0 ∧ x^2 + b'*x + 1800 = (x + p) * (x + q))) ∧
  b = 85 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l2294_229415
