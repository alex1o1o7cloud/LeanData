import Mathlib

namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2423_242329

theorem root_sum_reciprocal (α β γ : ℝ) : 
  (60 * α^3 - 80 * α^2 + 24 * α - 2 = 0) →
  (60 * β^3 - 80 * β^2 + 24 * β - 2 = 0) →
  (60 * γ^3 - 80 * γ^2 + 24 * γ - 2 = 0) →
  (α ≠ β) → (β ≠ γ) → (α ≠ γ) →
  (0 < α) → (α < 1) →
  (0 < β) → (β < 1) →
  (0 < γ) → (γ < 1) →
  (1 / (1 - α) + 1 / (1 - β) + 1 / (1 - γ) = 22) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2423_242329


namespace NUMINAMATH_CALUDE_parabola_inequality_l2423_242398

/-- A parabola with x = 1 as its axis of symmetry -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  axis_symmetry : -b / (2 * a) = 1

theorem parabola_inequality (p : Parabola) : 2 * p.c < 3 * p.b := by
  sorry

end NUMINAMATH_CALUDE_parabola_inequality_l2423_242398


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l2423_242309

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 2023) : (2023 - x)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l2423_242309


namespace NUMINAMATH_CALUDE_remaining_child_meal_capacity_l2423_242339

/-- Represents the meal capacity and consumption for a trekking group -/
structure TrekkingMeal where
  total_adults : ℕ
  adults_fed : ℕ
  adult_meal_capacity : ℕ
  child_meal_capacity : ℕ
  remaining_child_capacity : ℕ

/-- Theorem stating that given the conditions of the trekking meal,
    the number of children that can be catered with the remaining food is 36 -/
theorem remaining_child_meal_capacity
  (meal : TrekkingMeal)
  (h1 : meal.total_adults = 55)
  (h2 : meal.adult_meal_capacity = 70)
  (h3 : meal.child_meal_capacity = 90)
  (h4 : meal.adults_fed = 42)
  (h5 : meal.remaining_child_capacity = 36) :
  meal.remaining_child_capacity = 36 := by
  sorry


end NUMINAMATH_CALUDE_remaining_child_meal_capacity_l2423_242339


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2423_242334

/-- Given an arithmetic sequence where:
    - S_n is the sum of the first n terms
    - S_{2n} is the sum of the first 2n terms
    - S_{3n} is the sum of the first 3n terms
    This theorem proves that if S_n = 45 and S_{2n} = 60, then S_{3n} = 65. -/
theorem arithmetic_sequence_sum (n : ℕ) (S_n S_2n S_3n : ℝ) 
  (h1 : S_n = 45)
  (h2 : S_2n = 60) :
  S_3n = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2423_242334


namespace NUMINAMATH_CALUDE_planted_fraction_is_thirteen_fifteenths_l2423_242347

/-- Represents a right triangle field with an unplanted rectangle at the right angle -/
structure FieldWithUnplantedRectangle where
  /-- Length of the first leg of the right triangle -/
  leg1 : ℝ
  /-- Length of the second leg of the right triangle -/
  leg2 : ℝ
  /-- Width of the unplanted rectangle -/
  rect_width : ℝ
  /-- Height of the unplanted rectangle -/
  rect_height : ℝ
  /-- Shortest distance from the unplanted rectangle to the hypotenuse -/
  dist_to_hypotenuse : ℝ

/-- Calculates the fraction of the field that is planted -/
def planted_fraction (field : FieldWithUnplantedRectangle) : ℝ :=
  sorry

/-- Theorem stating the planted fraction for the given field configuration -/
theorem planted_fraction_is_thirteen_fifteenths :
  let field := FieldWithUnplantedRectangle.mk 5 12 1 4 3
  planted_fraction field = 13 / 15 := by
  sorry

end NUMINAMATH_CALUDE_planted_fraction_is_thirteen_fifteenths_l2423_242347


namespace NUMINAMATH_CALUDE_whispered_numbers_l2423_242375

/-- Represents a digit sum calculation step -/
def DigitSumStep (n : ℕ) : ℕ := sorry

/-- The maximum possible digit sum for a 2022-digit number -/
def MaxInitialSum : ℕ := 2022 * 9

theorem whispered_numbers (initial_number : ℕ) 
  (h1 : initial_number ≤ MaxInitialSum) 
  (whisper1 : ℕ) 
  (h2 : whisper1 = DigitSumStep initial_number)
  (whisper2 : ℕ) 
  (h3 : whisper2 = DigitSumStep whisper1)
  (h4 : 10 ≤ whisper2 ∧ whisper2 ≤ 99)
  (h5 : DigitSumStep whisper2 = 1) :
  whisper1 = 19 ∨ whisper1 = 28 := by sorry

end NUMINAMATH_CALUDE_whispered_numbers_l2423_242375


namespace NUMINAMATH_CALUDE_circle_area_through_point_l2423_242349

/-- The area of a circle with center R(5, -2) passing through the point S(-4, 7) is 162π. -/
theorem circle_area_through_point : 
  let R : ℝ × ℝ := (5, -2)
  let S : ℝ × ℝ := (-4, 7)
  let radius := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
  π * radius^2 = 162 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_through_point_l2423_242349


namespace NUMINAMATH_CALUDE_multiply_twelve_problem_l2423_242312

theorem multiply_twelve_problem (x : ℚ) : 
  (12 * x * 2 = 7899665 - 7899593) → x = 3 := by
sorry

end NUMINAMATH_CALUDE_multiply_twelve_problem_l2423_242312


namespace NUMINAMATH_CALUDE_sum_first_15_odd_integers_l2423_242304

/-- The sum of the first n odd positive integers -/
def sumFirstNOddIntegers (n : ℕ) : ℕ :=
  n * n

theorem sum_first_15_odd_integers : sumFirstNOddIntegers 15 = 225 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_integers_l2423_242304


namespace NUMINAMATH_CALUDE_original_cost_of_mixed_nuts_l2423_242321

/-- Calculates the original cost of a bag of mixed nuts -/
theorem original_cost_of_mixed_nuts
  (bag_size : ℕ)
  (serving_size : ℕ)
  (cost_per_serving_after_coupon : ℚ)
  (coupon_value : ℚ)
  (h1 : bag_size = 40)
  (h2 : serving_size = 1)
  (h3 : cost_per_serving_after_coupon = 1/2)
  (h4 : coupon_value = 5) :
  bag_size * cost_per_serving_after_coupon + coupon_value = 25 :=
sorry

end NUMINAMATH_CALUDE_original_cost_of_mixed_nuts_l2423_242321


namespace NUMINAMATH_CALUDE_max_triangle_area_l2423_242379

noncomputable section

def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def focal_distance : ℝ := 2

def eccentricity : ℝ := Real.sqrt 2 / 2

def right_focus : ℝ × ℝ := (1, 0)

def point_k : ℝ × ℝ := (2, 0)

def line_intersects_ellipse (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 2) ∧ ellipse x y

def triangle_area (F P Q : ℝ × ℝ) : ℝ :=
  abs ((P.1 - F.1) * (Q.2 - F.2) - (Q.1 - F.1) * (P.2 - F.2)) / 2

theorem max_triangle_area :
  ∃ (max_area : ℝ),
    max_area = Real.sqrt 2 / 4 ∧
    ∀ (k : ℝ) (P Q : ℝ × ℝ),
      k ≠ 0 →
      line_intersects_ellipse k P.1 P.2 →
      line_intersects_ellipse k Q.1 Q.2 →
      P ≠ Q →
      triangle_area right_focus P Q ≤ max_area :=
sorry

end

end NUMINAMATH_CALUDE_max_triangle_area_l2423_242379


namespace NUMINAMATH_CALUDE_game_score_total_l2423_242348

theorem game_score_total (dad_score : ℕ) (olaf_score : ℕ) : 
  dad_score = 7 → 
  olaf_score = 3 * dad_score → 
  olaf_score + dad_score = 28 := by
sorry

end NUMINAMATH_CALUDE_game_score_total_l2423_242348


namespace NUMINAMATH_CALUDE_operations_equality_l2423_242399

theorem operations_equality : 3 * 5 + 7 * 9 = 78 := by
  sorry

end NUMINAMATH_CALUDE_operations_equality_l2423_242399


namespace NUMINAMATH_CALUDE_dee_has_least_money_l2423_242328

-- Define the people
inductive Person : Type
  | Ada : Person
  | Ben : Person
  | Cal : Person
  | Dee : Person
  | Eve : Person

-- Define a function to represent the amount of money each person has
variable (money : Person → ℕ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q
axiom cal_more_than_ada_ben : money Person.Cal > money Person.Ada ∧ money Person.Cal > money Person.Ben
axiom ada_eve_more_than_dee : money Person.Ada > money Person.Dee ∧ money Person.Eve > money Person.Dee
axiom ben_between_ada_dee : money Person.Ben > money Person.Dee ∧ money Person.Ben < money Person.Ada

-- Theorem to prove
theorem dee_has_least_money :
  ∀ (p : Person), p ≠ Person.Dee → money Person.Dee < money p :=
sorry

end NUMINAMATH_CALUDE_dee_has_least_money_l2423_242328


namespace NUMINAMATH_CALUDE_problem_1_l2423_242352

theorem problem_1 : (-1)^2020 * (2020 - Real.pi)^0 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2423_242352


namespace NUMINAMATH_CALUDE_smallest_q_value_l2423_242343

def sum_of_range (n : ℕ) : ℕ := n * (n + 1) / 2

theorem smallest_q_value (p : ℕ) : 
  let initial_sum := sum_of_range 6
  let total_count := 6 + p + q
  let total_sum := initial_sum + 5 * p + 7 * q
  let mean := 5.3
  ∃ q : ℕ, q ≥ 0 ∧ (total_sum : ℝ) / total_count = mean ∧ 
    ∀ q' : ℕ, q' ≥ 0 → (initial_sum + 5 * p + 7 * q' : ℝ) / (6 + p + q') = mean → q ≤ q'
  := by sorry

end NUMINAMATH_CALUDE_smallest_q_value_l2423_242343


namespace NUMINAMATH_CALUDE_light_bulbs_theorem_l2423_242302

/-- The number of light bulbs in the kitchen -/
def kitchen_bulbs : ℕ := 35

/-- The fraction of broken light bulbs in the kitchen -/
def kitchen_broken_fraction : ℚ := 3/5

/-- The number of broken light bulbs in the foyer -/
def foyer_broken : ℕ := 10

/-- The fraction of broken light bulbs in the foyer -/
def foyer_broken_fraction : ℚ := 1/3

/-- The total number of unbroken light bulbs in both the foyer and kitchen -/
def total_unbroken : ℕ := 34

theorem light_bulbs_theorem : 
  kitchen_bulbs * (1 - kitchen_broken_fraction) + 
  (foyer_broken / foyer_broken_fraction) * (1 - foyer_broken_fraction) = total_unbroken := by
sorry

end NUMINAMATH_CALUDE_light_bulbs_theorem_l2423_242302


namespace NUMINAMATH_CALUDE_polygon_sides_l2423_242314

theorem polygon_sides (n : ℕ) (sum_angles : ℝ) : sum_angles = 1800 → (n - 2) * 180 = sum_angles → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2423_242314


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2423_242395

theorem complex_modulus_problem (z : ℂ) : (Complex.I^3 * z = 1 + Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2423_242395


namespace NUMINAMATH_CALUDE_max_value_of_shui_l2423_242340

/-- Represents the digits assigned to each Chinese character -/
structure ChineseDigits where
  jin : Fin 8
  xin : Fin 8
  li : Fin 8
  ke : Fin 8
  ba : Fin 8
  shan : Fin 8
  qiong : Fin 8
  shui : Fin 8

/-- All digits are unique -/
def all_unique (d : ChineseDigits) : Prop :=
  d.jin ≠ d.xin ∧ d.jin ≠ d.li ∧ d.jin ≠ d.ke ∧ d.jin ≠ d.ba ∧ d.jin ≠ d.shan ∧ d.jin ≠ d.qiong ∧ d.jin ≠ d.shui ∧
  d.xin ≠ d.li ∧ d.xin ≠ d.ke ∧ d.xin ≠ d.ba ∧ d.xin ≠ d.shan ∧ d.xin ≠ d.qiong ∧ d.xin ≠ d.shui ∧
  d.li ≠ d.ke ∧ d.li ≠ d.ba ∧ d.li ≠ d.shan ∧ d.li ≠ d.qiong ∧ d.li ≠ d.shui ∧
  d.ke ≠ d.ba ∧ d.ke ≠ d.shan ∧ d.ke ≠ d.qiong ∧ d.ke ≠ d.shui ∧
  d.ba ≠ d.shan ∧ d.ba ≠ d.qiong ∧ d.ba ≠ d.shui ∧
  d.shan ≠ d.qiong ∧ d.shan ≠ d.shui ∧
  d.qiong ≠ d.shui

/-- The sum of digits in each phrase is 19 -/
def sum_is_19 (d : ChineseDigits) : Prop :=
  d.jin.val + d.jin.val + d.xin.val + d.li.val = 19 ∧
  d.ke.val + d.ba.val + d.shan.val = 19 ∧
  d.shan.val + d.qiong.val + d.shui.val + d.jin.val = 19

/-- The ordering constraint: 尽 > 山 > 力 -/
def ordering_constraint (d : ChineseDigits) : Prop :=
  d.jin > d.shan ∧ d.shan > d.li

theorem max_value_of_shui (d : ChineseDigits) 
  (h1 : all_unique d) 
  (h2 : sum_is_19 d) 
  (h3 : ordering_constraint d) : 
  d.shui.val ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_shui_l2423_242340


namespace NUMINAMATH_CALUDE_stamps_leftover_l2423_242369

theorem stamps_leftover (olivia parker quinn : ℕ) (album_capacity : ℕ) 
  (h1 : olivia = 52) 
  (h2 : parker = 66) 
  (h3 : quinn = 23) 
  (h4 : album_capacity = 15) : 
  (olivia + parker + quinn) % album_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_stamps_leftover_l2423_242369


namespace NUMINAMATH_CALUDE_ellipse_max_sum_l2423_242360

/-- The maximum value of x + y for points on the ellipse x^2/16 + y^2/9 = 1 is 5 -/
theorem ellipse_max_sum (x y : ℝ) : 
  x^2/16 + y^2/9 = 1 → x + y ≤ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀^2/16 + y₀^2/9 = 1 ∧ x₀ + y₀ = 5 := by
  sorry

#check ellipse_max_sum

end NUMINAMATH_CALUDE_ellipse_max_sum_l2423_242360


namespace NUMINAMATH_CALUDE_sqrt_5_times_sqrt_6_minus_1_over_sqrt_5_range_l2423_242390

theorem sqrt_5_times_sqrt_6_minus_1_over_sqrt_5_range : 
  4 < Real.sqrt 5 * (Real.sqrt 6 - 1 / Real.sqrt 5) ∧ 
  Real.sqrt 5 * (Real.sqrt 6 - 1 / Real.sqrt 5) < 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_times_sqrt_6_minus_1_over_sqrt_5_range_l2423_242390


namespace NUMINAMATH_CALUDE_quadratic_value_theorem_l2423_242377

theorem quadratic_value_theorem (x : ℝ) (h : x^2 + 4*x - 2 = 0) :
  3*x^2 + 12*x - 23 = -17 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_theorem_l2423_242377


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2423_242305

theorem quadratic_root_implies_m_value :
  ∀ m : ℝ, (2^2 + m*2 + 2 = 0) → m = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2423_242305


namespace NUMINAMATH_CALUDE_amanda_borrowed_amount_l2423_242374

/-- Calculates the earnings for a given number of hours based on the specified payment cycle -/
def calculateEarnings (hours : Nat) : Nat :=
  let cycleEarnings := [2, 4, 6, 8, 10, 12]
  let fullCycles := hours / 6
  let remainingHours := hours % 6
  fullCycles * (cycleEarnings.sum) + (cycleEarnings.take remainingHours).sum

/-- The amount Amanda borrowed is equal to her earnings from 45 hours of mowing -/
theorem amanda_borrowed_amount : calculateEarnings 45 = 306 := by
  sorry

#eval calculateEarnings 45

end NUMINAMATH_CALUDE_amanda_borrowed_amount_l2423_242374


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2423_242391

theorem sum_of_reciprocals (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -2) : 
  x + y = 4/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2423_242391


namespace NUMINAMATH_CALUDE_tangent_lines_count_l2423_242370

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  -- Add appropriate fields for a line

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Counts the number of lines tangent to both circles -/
def countTangentLines (c1 c2 : Circle) : ℕ := sorry

/-- The main theorem -/
theorem tangent_lines_count 
  (c1 c2 : Circle) 
  (h1 : c1.radius = 5) 
  (h2 : c2.radius = 8) 
  (h3 : Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = 13) :
  countTangentLines c1 c2 = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_lines_count_l2423_242370


namespace NUMINAMATH_CALUDE_white_washing_cost_l2423_242315

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the area of two opposite walls of a room -/
def wall_area (d : Dimensions) : ℝ := 2 * d.length * d.height

/-- Calculates the total area of four walls of a room -/
def total_wall_area (d : Dimensions) : ℝ := wall_area d + wall_area { d with length := d.width }

/-- Calculates the area of a rectangular object -/
def area (d : Dimensions) : ℝ := d.length * d.width

theorem white_washing_cost 
  (room : Dimensions) 
  (door : Dimensions)
  (window : Dimensions)
  (num_windows : ℕ)
  (cost_per_sqft : ℝ)
  (h_room : room = { length := 25, width := 15, height := 12 })
  (h_door : door = { length := 6, width := 3, height := 0 })
  (h_window : window = { length := 4, width := 3, height := 0 })
  (h_num_windows : num_windows = 3)
  (h_cost : cost_per_sqft = 8) :
  (total_wall_area room - (area door + num_windows * area window)) * cost_per_sqft = 7248 := by
  sorry

end NUMINAMATH_CALUDE_white_washing_cost_l2423_242315


namespace NUMINAMATH_CALUDE_negation_of_existence_l2423_242301

theorem negation_of_existence (Z : Type) [Ring Z] : 
  (¬ ∃ x : Z, x^2 = 2*x) ↔ (∀ x : Z, x^2 ≠ 2*x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2423_242301


namespace NUMINAMATH_CALUDE_distance_borya_vasya_l2423_242356

/-- Represents the positions of houses along a road -/
structure HousePositions where
  andrey : ℝ
  borya : ℝ
  vasya : ℝ
  gena : ℝ

/-- The race setup along the road -/
def RaceSetup (h : HousePositions) : Prop :=
  h.gena - h.andrey = 2450 ∧
  h.vasya - h.andrey = h.gena - h.borya ∧
  (h.borya + h.gena) / 2 - (h.andrey + h.vasya) / 2 = 1000

theorem distance_borya_vasya (h : HousePositions) (race : RaceSetup h) :
  h.vasya - h.borya = 450 := by
  sorry

end NUMINAMATH_CALUDE_distance_borya_vasya_l2423_242356


namespace NUMINAMATH_CALUDE_pythagorean_theorem_construct_incommensurable_segments_l2423_242397

-- Define a type for geometric constructions
def GeometricConstruction : Type := Unit

-- Define a function to represent the construction of a segment
def constructSegment (length : ℝ) : GeometricConstruction := sorry

-- Define the Pythagorean theorem
theorem pythagorean_theorem (a b c : ℝ) : 
  a^2 + b^2 = c^2 ↔ ∃ (triangle : GeometricConstruction), true := sorry

-- Theorem stating that √2, √3, and √5 can be geometrically constructed
theorem construct_incommensurable_segments : 
  ∃ (construct_sqrt2 construct_sqrt3 construct_sqrt5 : GeometricConstruction),
    (∃ (a : ℝ), a^2 = 2 ∧ constructSegment a = construct_sqrt2) ∧
    (∃ (b : ℝ), b^2 = 3 ∧ constructSegment b = construct_sqrt3) ∧
    (∃ (c : ℝ), c^2 = 5 ∧ constructSegment c = construct_sqrt5) :=
sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_construct_incommensurable_segments_l2423_242397


namespace NUMINAMATH_CALUDE_sin_arccos_twelve_thirteenths_l2423_242346

theorem sin_arccos_twelve_thirteenths : Real.sin (Real.arccos (12/13)) = 5/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_twelve_thirteenths_l2423_242346


namespace NUMINAMATH_CALUDE_students_at_higher_fee_l2423_242319

/-- Represents the inverse proportionality between number of students and tuition fee -/
def inverse_proportional (s f : ℝ) : Prop := ∃ k : ℝ, s * f = k

/-- Theorem: Given inverse proportionality and initial conditions, prove the number of students at $2500 -/
theorem students_at_higher_fee 
  (s₁ s₂ f₁ f₂ : ℝ) 
  (h_inverse : inverse_proportional s₁ f₁ ∧ inverse_proportional s₂ f₂)
  (h_initial : s₁ = 40 ∧ f₁ = 2000)
  (h_new_fee : f₂ = 2500) :
  s₂ = 32 := by
  sorry

end NUMINAMATH_CALUDE_students_at_higher_fee_l2423_242319


namespace NUMINAMATH_CALUDE_range_of_g_range_of_g_complete_l2423_242324

def f (x : ℝ) : ℝ := 5 * x + 3

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ y ∈ Set.range g, -157 ≤ y ∧ y ≤ 1093 :=
sorry

theorem range_of_g_complete :
  ∀ y, -157 ≤ y ∧ y ≤ 1093 → ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_g_range_of_g_complete_l2423_242324


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2423_242331

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}

-- Define set B
def B : Set ℝ := {y : ℝ | y ≥ 1/2}

-- State the theorem
theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {x : ℝ | -2 ≤ x ∧ x < 1/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2423_242331


namespace NUMINAMATH_CALUDE_south_movement_representation_l2423_242323

/-- Represents the direction of movement -/
inductive Direction
  | North
  | South

/-- Represents a movement with a distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Converts a movement to its signed representation -/
def Movement.toSigned (m : Movement) : ℝ :=
  match m.direction with
  | Direction.North => m.distance
  | Direction.South => -m.distance

/-- The problem statement -/
theorem south_movement_representation :
  let north20 : Movement := ⟨20, Direction.North⟩
  let south120 : Movement := ⟨120, Direction.South⟩
  north20.toSigned = 20 →
  south120.toSigned = -120 := by
  sorry

end NUMINAMATH_CALUDE_south_movement_representation_l2423_242323


namespace NUMINAMATH_CALUDE_sum_odd_divisors_300_eq_124_l2423_242350

/-- The sum of all odd divisors of 300 -/
def sum_odd_divisors_300 : ℕ := 124

/-- Theorem: The sum of all odd divisors of 300 is 124 -/
theorem sum_odd_divisors_300_eq_124 : sum_odd_divisors_300 = 124 := by sorry

end NUMINAMATH_CALUDE_sum_odd_divisors_300_eq_124_l2423_242350


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l2423_242381

/-- Represents a square quilt block -/
structure QuiltBlock where
  total_squares : ℕ
  divided_squares : ℕ
  divided_rectangles : ℕ
  shaded_column : ℕ

/-- The fraction of the quilt block that is shaded -/
def shaded_fraction (q : QuiltBlock) : ℚ :=
  q.shaded_column / q.total_squares

/-- Theorem stating that the shaded fraction is 1/3 for the given quilt block configuration -/
theorem quilt_shaded_fraction :
  ∀ q : QuiltBlock,
    q.total_squares = 9 ∧
    q.divided_squares = 3 ∧
    q.divided_rectangles = 3 ∧
    q.shaded_column = 1 →
    shaded_fraction q = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_quilt_shaded_fraction_l2423_242381


namespace NUMINAMATH_CALUDE_chips_ounces_amber_chips_problem_l2423_242342

/-- Represents the problem of determining the number of ounces in a bag of chips. -/
theorem chips_ounces (total_money : ℚ) (candy_price : ℚ) (candy_ounces : ℚ) 
  (chips_price : ℚ) (max_ounces : ℚ) : ℚ :=
  let candy_bags := total_money / candy_price
  let candy_total_ounces := candy_bags * candy_ounces
  let chips_bags := total_money / chips_price
  let chips_ounces_per_bag := max_ounces / chips_bags
  chips_ounces_per_bag

/-- Proves that given the conditions in the problem, a bag of chips contains 17 ounces. -/
theorem amber_chips_problem : 
  chips_ounces 7 1 12 (14/10) 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_chips_ounces_amber_chips_problem_l2423_242342


namespace NUMINAMATH_CALUDE_triangle_area_sum_l2423_242300

-- Define points on a line
variable (A B C D E : ℝ)

-- Define lengths
variable (AB BC CD : ℝ)

-- Define areas
variable (S_MAC S_NBC S_MCD S_NCE : ℝ)

-- State the theorem
theorem triangle_area_sum :
  A < B ∧ B < C ∧ C < D ∧ D < E →  -- Points are on the same line in order
  AB = 4 →
  BC = 3 →
  CD = 2 →
  S_MAC + S_NBC = 51 →
  S_MCD + S_NCE = 32 →
  S_MCD + S_NBC = 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_sum_l2423_242300


namespace NUMINAMATH_CALUDE_intersection_A_M_range_of_b_l2423_242341

-- Define the sets A, B, M, and U
def A : Set ℝ := {x | -3 < x ∧ x ≤ 6}
def B (b : ℝ) : Set ℝ := {x | b - 3 < x ∧ x < b + 7}
def M : Set ℝ := {x | -4 ≤ x ∧ x < 5}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∩ M = {x | -3 < x < 5}
theorem intersection_A_M : A ∩ M = {x : ℝ | -3 < x ∧ x < 5} := by sorry

-- Theorem 2: If B ∪ (¬UM) = R, then -2 ≤ b < -1
theorem range_of_b (b : ℝ) (h : B b ∪ (Mᶜ) = Set.univ) : -2 ≤ b ∧ b < -1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_M_range_of_b_l2423_242341


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2423_242358

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Opposite sides of angles A, B, C are a, b, c respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  Real.cos C * Real.sin (A + π/6) - Real.sin C * Real.sin (A - π/3) = 1/2 →
  -- Perimeter condition
  a + b + c = 4 →
  -- Area condition
  1/2 * a * c * Real.sin B = Real.sqrt 3 / 3 →
  -- Conclusion: B = π/3 and b = 3/2
  B = π/3 ∧ b = 3/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2423_242358


namespace NUMINAMATH_CALUDE_distance_major_minor_endpoints_l2423_242357

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * (x + 2)^2 + 16 * y^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (-2, 0)

-- Define the semi-major axis length
def semi_major : ℝ := 4

-- Define the semi-minor axis length
def semi_minor : ℝ := 1

-- Define an endpoint of the major axis
def major_endpoint : ℝ × ℝ := (center.1 + semi_major, center.2)

-- Define an endpoint of the minor axis
def minor_endpoint : ℝ × ℝ := (center.1, center.2 + semi_minor)

-- Theorem statement
theorem distance_major_minor_endpoints : 
  Real.sqrt ((major_endpoint.1 - minor_endpoint.1)^2 + (major_endpoint.2 - minor_endpoint.2)^2) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_major_minor_endpoints_l2423_242357


namespace NUMINAMATH_CALUDE_only_postcard_win_is_systematic_l2423_242396

-- Define the type for sampling methods
inductive SamplingMethod
| EmployeeRep
| MarketResearch
| LotteryDraw
| PostcardWin
| ExamAnalysis

-- Define what constitutes systematic sampling
def is_systematic_sampling (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.PostcardWin => True
  | _ => False

-- Theorem stating that only PostcardWin is systematic sampling
theorem only_postcard_win_is_systematic :
  ∀ (method : SamplingMethod),
    is_systematic_sampling method ↔ method = SamplingMethod.PostcardWin :=
by sorry

#check only_postcard_win_is_systematic

end NUMINAMATH_CALUDE_only_postcard_win_is_systematic_l2423_242396


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2423_242326

/-- Given a distance of 8640 meters and a time of 36 minutes, 
    the average speed is 4 meters per second. -/
theorem average_speed_calculation (distance : ℝ) (time_minutes : ℝ) :
  distance = 8640 ∧ time_minutes = 36 →
  (distance / (time_minutes * 60)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2423_242326


namespace NUMINAMATH_CALUDE_count_multiples_of_24_l2423_242386

def smallest_square_multiple_of_24 : ℕ := 144
def smallest_fourth_power_multiple_of_24 : ℕ := 1296

theorem count_multiples_of_24 :
  (Finset.range (smallest_fourth_power_multiple_of_24 / 24 + 1) ∩ 
   Finset.filter (λ n => n ≥ smallest_square_multiple_of_24 / 24) 
                 (Finset.range (smallest_fourth_power_multiple_of_24 / 24 + 1))).card = 49 :=
by sorry

end NUMINAMATH_CALUDE_count_multiples_of_24_l2423_242386


namespace NUMINAMATH_CALUDE_second_watermelon_weight_l2423_242365

theorem second_watermelon_weight (total_weight first_weight : ℝ) 
  (h1 : total_weight = 14.02)
  (h2 : first_weight = 9.91) : 
  total_weight - first_weight = 4.11 := by
sorry

end NUMINAMATH_CALUDE_second_watermelon_weight_l2423_242365


namespace NUMINAMATH_CALUDE_school_travel_time_difference_l2423_242330

/-- The problem of calculating the late arrival time of a boy traveling to school. -/
theorem school_travel_time_difference (distance : ℝ) (speed1 speed2 : ℝ) (early_time : ℝ) : 
  distance = 2.5 →
  speed1 = 5 →
  speed2 = 10 →
  early_time = 8 / 60 →
  (distance / speed1 - (distance / speed2 + early_time)) * 60 = 7 := by
  sorry

end NUMINAMATH_CALUDE_school_travel_time_difference_l2423_242330


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2423_242371

def repeating_decimal_246 : ℚ := 246 / 999
def repeating_decimal_135 : ℚ := 135 / 999
def repeating_decimal_579 : ℚ := 579 / 999

theorem repeating_decimal_subtraction :
  repeating_decimal_246 - repeating_decimal_135 - repeating_decimal_579 = -24 / 51 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2423_242371


namespace NUMINAMATH_CALUDE_f_2018_equals_neg_8_l2423_242335

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2018_equals_neg_8 (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 3) = -1 / f x)
  (h3 : ∀ x ∈ Set.Icc (-3) (-2), f x = 4 * x) :
  f 2018 = -8 := by
  sorry

end NUMINAMATH_CALUDE_f_2018_equals_neg_8_l2423_242335


namespace NUMINAMATH_CALUDE_gcd_459_357_l2423_242332

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2423_242332


namespace NUMINAMATH_CALUDE_b_income_percentage_over_c_l2423_242325

/-- Given the monthly incomes of A, B, and C, prove that B's monthly income is 12% more than C's. -/
theorem b_income_percentage_over_c (a_annual : ℕ) (c_monthly : ℕ) (h1 : a_annual = 571200) (h2 : c_monthly = 17000) :
  let a_monthly : ℕ := a_annual / 12
  let b_monthly : ℕ := (2 * a_monthly) / 5
  (b_monthly : ℚ) / c_monthly - 1 = 12 / 100 := by sorry

end NUMINAMATH_CALUDE_b_income_percentage_over_c_l2423_242325


namespace NUMINAMATH_CALUDE_correct_expansion_of_expression_l2423_242322

theorem correct_expansion_of_expression (a : ℝ) : 
  5 + a - 2 * (3 * a - 5) = 5 + a - 6 * a + 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_expansion_of_expression_l2423_242322


namespace NUMINAMATH_CALUDE_first_three_digit_in_square_sum_row_l2423_242345

/-- Represents a position in Pascal's triangle -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Returns the value at a given position in Pascal's triangle -/
def pascalValue (pos : Position) : Nat :=
  sorry

/-- Returns the sum of a row in Pascal's triangle -/
def rowSum (n : Nat) : Nat :=
  2^n

/-- Checks if a number is a three-digit number -/
def isThreeDigit (n : Nat) : Bool :=
  100 ≤ n ∧ n ≤ 999

/-- The theorem to be proved -/
theorem first_three_digit_in_square_sum_row :
  let pos := Position.mk 16 1
  (isThreeDigit (pascalValue pos)) ∧
  (∃ k : Nat, rowSum 16 = k * k) ∧
  (∀ n < 16, ∀ i ≤ n, ¬(isThreeDigit (pascalValue (Position.mk n i)) ∧ ∃ k : Nat, rowSum n = k * k)) :=
by sorry

end NUMINAMATH_CALUDE_first_three_digit_in_square_sum_row_l2423_242345


namespace NUMINAMATH_CALUDE_figure_colorings_l2423_242378

/-- Represents the number of ways to color a single equilateral triangle --/
def triangle_colorings : ℕ := 6

/-- Represents the number of ways to color each subsequent triangle --/
def subsequent_triangle_colorings : ℕ := 3

/-- Represents the number of ways to color the additional dot --/
def additional_dot_colorings : ℕ := 2

/-- The total number of dots in the figure --/
def total_dots : ℕ := 10

/-- The number of triangles in the figure --/
def num_triangles : ℕ := 3

theorem figure_colorings :
  triangle_colorings * subsequent_triangle_colorings ^ (num_triangles - 1) * additional_dot_colorings = 108 := by
  sorry

end NUMINAMATH_CALUDE_figure_colorings_l2423_242378


namespace NUMINAMATH_CALUDE_function_positive_l2423_242306

theorem function_positive (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, (x + 1) * f x + x * (deriv^[2] f x) > 0) : 
  ∀ x : ℝ, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_positive_l2423_242306


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2423_242367

theorem sufficient_not_necessary (x : ℝ) : 
  (x = 2 → (x - 2) * (x - 1) = 0) ∧ 
  ¬((x - 2) * (x - 1) = 0 → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2423_242367


namespace NUMINAMATH_CALUDE_smallest_n_terminating_with_3_l2423_242311

def is_terminating_decimal (n : ℕ+) : Prop :=
  ∃ (a b : ℕ), n = 2^a * 5^b

def contains_digit_3 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 3

theorem smallest_n_terminating_with_3 :
  ∀ n : ℕ+, n < 32 →
    ¬(is_terminating_decimal n ∧ contains_digit_3 n) ∧
    (is_terminating_decimal 32 ∧ contains_digit_3 32) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_terminating_with_3_l2423_242311


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2423_242318

theorem inequality_equivalence (x y : ℝ) :
  (2 * y - 3 * x > Real.sqrt (9 * x^2)) ↔ ((y > 3 * x ∧ x ≥ 0) ∨ (y > 0 ∧ x < 0)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2423_242318


namespace NUMINAMATH_CALUDE_sin_ten_pi_thirds_l2423_242361

theorem sin_ten_pi_thirds : Real.sin (10 * π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_ten_pi_thirds_l2423_242361


namespace NUMINAMATH_CALUDE_coefficient_a2_value_l2423_242351

/-- Given a complex number z and a polynomial expansion of (x-z)^4,
    prove that the coefficient of x^2 is -3 + 3√3i. -/
theorem coefficient_a2_value (z : ℂ) (a₀ a₁ a₂ a₃ a₄ : ℂ) :
  z = (1/2 : ℂ) + (Complex.I * Real.sqrt 3) / 2 →
  (fun x : ℂ ↦ (x - z)^4) = (fun x : ℂ ↦ a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) →
  a₂ = -3 + Complex.I * (3 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_coefficient_a2_value_l2423_242351


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_power_of_three_l2423_242393

theorem infinitely_many_divisible_by_power_of_three (k : ℕ) (hk : k > 0) :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, (3^k : ℕ) ∣ (f n)^3 + 10 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_power_of_three_l2423_242393


namespace NUMINAMATH_CALUDE_t_has_six_values_l2423_242383

/-- A type representing single-digit positive integers -/
def SingleDigit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The theorem stating that t can have 6 distinct values -/
theorem t_has_six_values 
  (p q r s t : SingleDigit) 
  (h1 : p.val - q.val = r.val)
  (h2 : r.val - s.val = t.val)
  (h3 : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
        q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
        r ≠ s ∧ r ≠ t ∧ 
        s ≠ t) :
  ∃ (values : Finset ℕ), values.card = 6 ∧ t.val ∈ values ∧ 
  ∀ x, x ∈ values → ∃ (p' q' r' s' t' : SingleDigit), 
    p'.val - q'.val = r'.val ∧ 
    r'.val - s'.val = t'.val ∧ 
    t'.val = x ∧
    p' ≠ q' ∧ p' ≠ r' ∧ p' ≠ s' ∧ p' ≠ t' ∧ 
    q' ≠ r' ∧ q' ≠ s' ∧ q' ≠ t' ∧ 
    r' ≠ s' ∧ r' ≠ t' ∧ 
    s' ≠ t' := by
  sorry

end NUMINAMATH_CALUDE_t_has_six_values_l2423_242383


namespace NUMINAMATH_CALUDE_land_sections_area_l2423_242336

theorem land_sections_area (x y z : ℝ) 
  (h1 : x = (2/5) * (x + y + z))
  (h2 : y / z = (3/2) / (4/3))
  (h3 : z = x - 16) :
  x + y + z = 136 := by
  sorry

end NUMINAMATH_CALUDE_land_sections_area_l2423_242336


namespace NUMINAMATH_CALUDE_water_consumption_proof_l2423_242387

/-- Calculates the total water consumption for horses over a given period. -/
def total_water_consumption (initial_horses : ℕ) (added_horses : ℕ) (drinking_water : ℕ) (bathing_water : ℕ) (days : ℕ) : ℕ :=
  let total_horses := initial_horses + added_horses
  let daily_consumption_per_horse := drinking_water + bathing_water
  let daily_consumption := total_horses * daily_consumption_per_horse
  daily_consumption * days

/-- Proves that the total water consumption for the given conditions is 1568 liters. -/
theorem water_consumption_proof :
  total_water_consumption 3 5 5 2 28 = 1568 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_proof_l2423_242387


namespace NUMINAMATH_CALUDE_sqrt_three_sum_product_l2423_242373

theorem sqrt_three_sum_product : Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt 27) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_sum_product_l2423_242373


namespace NUMINAMATH_CALUDE_yadav_clothes_transport_expenditure_l2423_242355

/-- Represents Mr. Yadav's financial situation --/
structure YadavFinances where
  monthlySalary : ℝ
  consumablePercentage : ℝ
  rentPercentage : ℝ
  utilitiesPercentage : ℝ
  entertainmentPercentage : ℝ
  clothesTransportPercentage : ℝ
  annualSavings : ℝ

/-- Calculates Mr. Yadav's monthly expenditure on clothes and transport --/
def monthlyClothesTransportExpenditure (y : YadavFinances) : ℝ :=
  let totalSpentPercentage := y.consumablePercentage + y.rentPercentage + y.utilitiesPercentage + y.entertainmentPercentage
  let remainingPercentage := 1 - totalSpentPercentage
  let monthlyRemainder := y.monthlySalary * remainingPercentage
  monthlyRemainder * y.clothesTransportPercentage

/-- Theorem stating that Mr. Yadav's monthly expenditure on clothes and transport is 2052 --/
theorem yadav_clothes_transport_expenditure (y : YadavFinances) 
  (h1 : y.consumablePercentage = 0.6)
  (h2 : y.rentPercentage = 0.2)
  (h3 : y.utilitiesPercentage = 0.1)
  (h4 : y.entertainmentPercentage = 0.05)
  (h5 : y.clothesTransportPercentage = 0.5)
  (h6 : y.annualSavings = 24624) :
  monthlyClothesTransportExpenditure y = 2052 := by
  sorry

#check yadav_clothes_transport_expenditure

end NUMINAMATH_CALUDE_yadav_clothes_transport_expenditure_l2423_242355


namespace NUMINAMATH_CALUDE_expression_evaluation_l2423_242338

theorem expression_evaluation : 
  2 * (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2423_242338


namespace NUMINAMATH_CALUDE_homework_time_reduction_l2423_242310

theorem homework_time_reduction (x : ℝ) : 
  (∀ t₀ t₂ : ℝ, t₀ > 0 ∧ t₂ > 0 ∧ t₀ > t₂ →
    (∃ t₁ : ℝ, t₁ = t₀ * (1 - x) ∧ t₂ = t₁ * (1 - x)) ↔
    t₀ * (1 - x)^2 = t₂) →
  100 * (1 - x)^2 = 70 :=
by sorry

end NUMINAMATH_CALUDE_homework_time_reduction_l2423_242310


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2423_242394

theorem solution_set_inequality (x : ℝ) : x / (x + 1) ≤ 0 ↔ x ∈ Set.Ioc (-1) 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2423_242394


namespace NUMINAMATH_CALUDE_shared_friends_l2423_242382

theorem shared_friends (james_friends : ℕ) (john_friends : ℕ) (combined_list : ℕ) :
  james_friends = 75 →
  john_friends = 3 * james_friends →
  combined_list = 275 →
  james_friends + john_friends - combined_list = 25 := by
sorry

end NUMINAMATH_CALUDE_shared_friends_l2423_242382


namespace NUMINAMATH_CALUDE_tim_water_consumption_l2423_242320

/-- The number of ounces in a quart -/
def ounces_per_quart : ℕ := 32

/-- The number of quarts in each bottle Tim drinks -/
def quarts_per_bottle : ℚ := 3/2

/-- The number of bottles Tim drinks per day -/
def bottles_per_day : ℕ := 2

/-- The additional ounces Tim drinks per day -/
def additional_ounces_per_day : ℕ := 20

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The total amount of water Tim drinks in a week, in ounces -/
def water_per_week : ℕ := 812

theorem tim_water_consumption :
  (bottles_per_day * (quarts_per_bottle * ounces_per_quart).floor + additional_ounces_per_day) * days_per_week = water_per_week :=
sorry

end NUMINAMATH_CALUDE_tim_water_consumption_l2423_242320


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l2423_242376

/-- A quadratic function f(x) = ax^2 + bx + c satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_positive : a > 0
  condition_1 : |a + b + c| = 3
  condition_2 : |4*a + 2*b + c| = 3
  condition_3 : |9*a + 3*b + c| = 3

/-- The theorem stating the possible coefficients of the quadratic function -/
theorem quadratic_coefficients (f : QuadraticFunction) :
  (f.a = 6 ∧ f.b = -24 ∧ f.c = 21) ∨
  (f.a = 3 ∧ f.b = -15 ∧ f.c = 15) ∨
  (f.a = 3 ∧ f.b = -9 ∧ f.c = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l2423_242376


namespace NUMINAMATH_CALUDE_parabola_equation_l2423_242307

/-- A parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  vertex_origin : equation 0 0
  focus_x_axis : ∃ (f : ℝ), equation f 0 ∧ f ≠ 0

/-- The line y = 2x + 1 -/
def line (x y : ℝ) : Prop := y = 2 * x + 1

/-- The chord created by intersecting the parabola with the line -/
def chord (p : Parabola) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  p.equation x₁ y₁ ∧ p.equation x₂ y₂ ∧ line x₁ y₁ ∧ line x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

theorem parabola_equation (p : Parabola) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), chord p x₁ y₁ x₂ y₂ ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 15) →
  p.equation = λ x y => y^2 = 12 * x :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2423_242307


namespace NUMINAMATH_CALUDE_alpha_beta_composition_l2423_242380

theorem alpha_beta_composition (α β : ℝ → ℝ) (h_α : ∀ x, α x = 4 * x + 9) (h_β : ∀ x, β x = 7 * x + 6) :
  (∃ x, (α ∘ β) x = 4) ↔ (∃ x, x = -29/28) :=
by sorry

end NUMINAMATH_CALUDE_alpha_beta_composition_l2423_242380


namespace NUMINAMATH_CALUDE_find_other_number_l2423_242344

theorem find_other_number (A B : ℕ+) (hA : A = 24) (hHCF : Nat.gcd A B = 16) (hLCM : Nat.lcm A B = 312) : B = 208 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l2423_242344


namespace NUMINAMATH_CALUDE_nancy_crayon_packs_l2423_242354

theorem nancy_crayon_packs (total_crayons : ℕ) (crayons_per_pack : ℕ) (h1 : total_crayons = 615) (h2 : crayons_per_pack = 15) :
  total_crayons / crayons_per_pack = 41 := by
  sorry

end NUMINAMATH_CALUDE_nancy_crayon_packs_l2423_242354


namespace NUMINAMATH_CALUDE_max_diagonal_of_rectangle_l2423_242364

/-- The maximum diagonal of a rectangle with perimeter 40 --/
theorem max_diagonal_of_rectangle (l w : ℝ) : 
  l > 0 → w > 0 → l + w = 20 → 
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 20 → 
  Real.sqrt (l^2 + w^2) ≤ 20 :=
by sorry

end NUMINAMATH_CALUDE_max_diagonal_of_rectangle_l2423_242364


namespace NUMINAMATH_CALUDE_triangle_area_with_median_l2423_242363

/-- Given a triangle with two sides of length 1 and √15, and a median of length 2 to the third side,
    the area of the triangle is √15/2. -/
theorem triangle_area_with_median (a b c : ℝ) (m : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 15) (h3 : m = 2)
    (hm : m^2 = (2*a^2 + 2*b^2 - c^2) / 4) : (a * b) / 2 = Real.sqrt 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_median_l2423_242363


namespace NUMINAMATH_CALUDE_competition_matches_l2423_242372

theorem competition_matches (n : ℕ) (h : n = 6) : n * (n - 1) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_competition_matches_l2423_242372


namespace NUMINAMATH_CALUDE_minsu_marbles_left_l2423_242384

/-- Calculates the number of marbles left after distribution -/
def marblesLeft (totalMarbles : ℕ) (largeBulk smallBulk : ℕ) (largeBoxes smallBoxes : ℕ) : ℕ :=
  totalMarbles - (largeBulk * largeBoxes + smallBulk * smallBoxes)

/-- Theorem stating the number of marbles left after Minsu's distribution -/
theorem minsu_marbles_left :
  marblesLeft 240 35 6 4 3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_minsu_marbles_left_l2423_242384


namespace NUMINAMATH_CALUDE_measure_one_kg_grain_l2423_242353

/-- Represents a balance scale --/
structure BalanceScale where
  isInaccurate : Bool

/-- Represents a weight --/
structure Weight where
  mass : ℝ
  isAccurate : Bool

/-- Represents a bag of grain --/
structure GrainBag where
  mass : ℝ

/-- Function to measure a specific mass of grain --/
def measureGrain (scale : BalanceScale) (reference : Weight) (bag : GrainBag) (targetMass : ℝ) : Prop :=
  scale.isInaccurate ∧ reference.isAccurate ∧ reference.mass = targetMass

/-- Theorem stating that it's possible to measure 1 kg of grain using inaccurate scales and an accurate 1 kg weight --/
theorem measure_one_kg_grain 
  (scale : BalanceScale) 
  (reference : Weight) 
  (bag : GrainBag) : 
  measureGrain scale reference bag 1 → 
  ∃ (measuredGrain : GrainBag), measuredGrain.mass = 1 :=
sorry

end NUMINAMATH_CALUDE_measure_one_kg_grain_l2423_242353


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l2423_242366

theorem no_linear_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, (x^2 - x + m) * (x - 8) = x^3 - 9*x^2 + 0*x + (-8*m)) → m = -8 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l2423_242366


namespace NUMINAMATH_CALUDE_students_in_grade_6_l2423_242359

theorem students_in_grade_6 (total : ℕ) (grade_4 : ℕ) (grade_5 : ℕ) (grade_6 : ℕ) :
  total = 100 → grade_4 = 30 → grade_5 = 35 → total = grade_4 + grade_5 + grade_6 → grade_6 = 35 := by
  sorry

end NUMINAMATH_CALUDE_students_in_grade_6_l2423_242359


namespace NUMINAMATH_CALUDE_marias_average_balance_l2423_242392

/-- Given Maria's savings account balances for four months, prove that the average monthly balance is $300. -/
theorem marias_average_balance (jan feb mar apr : ℕ) 
  (h_jan : jan = 150)
  (h_feb : feb = 300)
  (h_mar : mar = 450)
  (h_apr : apr = 300) :
  (jan + feb + mar + apr) / 4 = 300 := by
  sorry

end NUMINAMATH_CALUDE_marias_average_balance_l2423_242392


namespace NUMINAMATH_CALUDE_least_divisor_for_perfect_square_twenty_one_gives_perfect_square_twenty_one_is_least_l2423_242362

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

theorem least_divisor_for_perfect_square : 
  ∀ n : ℕ, n > 0 → is_perfect_square (16800 / n) → n ≥ 21 :=
by sorry

theorem twenty_one_gives_perfect_square : 
  is_perfect_square (16800 / 21) :=
by sorry

theorem twenty_one_is_least :
  ∀ n : ℕ, n > 0 → is_perfect_square (16800 / n) → n = 21 :=
by sorry

end NUMINAMATH_CALUDE_least_divisor_for_perfect_square_twenty_one_gives_perfect_square_twenty_one_is_least_l2423_242362


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2423_242385

theorem repeating_decimal_sum : 
  (2 : ℚ) / 9 + 5 / 99 + 3 / 9999 = 910 / 3333 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2423_242385


namespace NUMINAMATH_CALUDE_impossibility_l2423_242388

/-- The number of piles -/
def n : ℕ := 2018

/-- The i-th prime number -/
def p (i : ℕ) : ℕ := sorry

/-- The initial configuration of piles -/
def initial_config : Fin n → ℕ := λ i => p i.val

/-- The desired final configuration of piles -/
def final_config : Fin n → ℕ := λ _ => n

/-- Split operation: split a pile and add a chip to one of the new piles -/
def split (config : Fin n → ℕ) (i : Fin n) (k : ℕ) (add_to_first : Bool) : Fin n → ℕ := sorry

/-- Merge operation: merge two piles and add a chip to the merged pile -/
def merge (config : Fin n → ℕ) (i j : Fin n) : Fin n → ℕ := sorry

/-- Predicate to check if a configuration is reachable from the initial configuration -/
def is_reachable (config : Fin n → ℕ) : Prop := sorry

theorem impossibility : ¬ is_reachable final_config := by sorry

end NUMINAMATH_CALUDE_impossibility_l2423_242388


namespace NUMINAMATH_CALUDE_same_gender_probability_l2423_242316

/-- The probability of selecting 2 students of the same gender from a group of 5 students with 3 boys and 2 girls -/
theorem same_gender_probability (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_students = 5)
  (h2 : boys = 3)
  (h3 : girls = 2)
  (h4 : total_students = boys + girls) :
  (Nat.choose boys 2 + Nat.choose girls 2) / Nat.choose total_students 2 = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_same_gender_probability_l2423_242316


namespace NUMINAMATH_CALUDE_chinese_chess_probability_l2423_242389

theorem chinese_chess_probability (p_win p_draw : ℝ) 
  (h_win : p_win = 0.5) 
  (h_draw : p_draw = 0.2) : 
  p_win + p_draw = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_chinese_chess_probability_l2423_242389


namespace NUMINAMATH_CALUDE_cobbler_efficiency_l2423_242337

/-- Represents the cobbler's work schedule and output --/
structure CobblerSchedule where
  hours_per_day : ℕ -- Hours worked per day from Monday to Thursday
  friday_hours : ℕ  -- Hours worked on Friday
  shoes_per_week : ℕ -- Number of shoes mended in a week

/-- Calculates the number of shoes mended per hour --/
def shoes_per_hour (schedule : CobblerSchedule) : ℚ :=
  schedule.shoes_per_week / (4 * schedule.hours_per_day + schedule.friday_hours)

/-- Theorem stating that the cobbler mends 3 shoes per hour --/
theorem cobbler_efficiency (schedule : CobblerSchedule) 
  (h1 : schedule.hours_per_day = 8)
  (h2 : schedule.friday_hours = 3)
  (h3 : schedule.shoes_per_week = 105) :
  shoes_per_hour schedule = 3 := by
  sorry

#eval shoes_per_hour ⟨8, 3, 105⟩

end NUMINAMATH_CALUDE_cobbler_efficiency_l2423_242337


namespace NUMINAMATH_CALUDE_class_gender_composition_l2423_242317

theorem class_gender_composition (num_boys num_girls : ℕ) :
  num_boys = 2 * num_girls →
  num_boys = num_girls + 7 →
  num_girls - 1 = 6 := by
sorry

end NUMINAMATH_CALUDE_class_gender_composition_l2423_242317


namespace NUMINAMATH_CALUDE_vector_subtraction_l2423_242333

/-- Given two vectors AB and AC in a plane, prove that vector BC is their difference. -/
theorem vector_subtraction (AB AC : ℝ × ℝ) (h1 : AB = (3, 4)) (h2 : AC = (1, 3)) :
  AC - AB = (-2, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2423_242333


namespace NUMINAMATH_CALUDE_angel_letters_count_l2423_242308

theorem angel_letters_count :
  ∀ (large_envelopes small_letters letters_per_large : ℕ),
    large_envelopes = 30 →
    letters_per_large = 2 →
    small_letters = 20 →
    large_envelopes * letters_per_large + small_letters = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_angel_letters_count_l2423_242308


namespace NUMINAMATH_CALUDE_remaining_black_cards_l2423_242368

theorem remaining_black_cards (total_cards : Nat) (black_cards : Nat) (removed_cards : Nat) :
  total_cards = 52 →
  black_cards = 26 →
  removed_cards = 4 →
  black_cards - removed_cards = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_black_cards_l2423_242368


namespace NUMINAMATH_CALUDE_division_problem_l2423_242327

theorem division_problem (n : ℕ) : 
  n / 22 = 12 ∧ n % 22 = 1 → n = 265 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2423_242327


namespace NUMINAMATH_CALUDE_angle_inequality_equivalence_l2423_242313

theorem angle_inequality_equivalence (θ : Real) : 
  (0 < θ ∧ θ < Real.pi / 2) ↔ 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → 
    x^2 * Real.cos θ - x * (1 - x) + 2 * (1 - x)^2 * Real.sin θ > 0) :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_equivalence_l2423_242313


namespace NUMINAMATH_CALUDE_evaluate_expression_l2423_242303

theorem evaluate_expression : 2 + 3 * 4 - 5 + 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2423_242303
