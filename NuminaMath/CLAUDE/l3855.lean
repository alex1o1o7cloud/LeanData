import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3855_385542

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 2 → 1/x < 1/2) ∧
  (∃ x, 1/x < 1/2 ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3855_385542


namespace NUMINAMATH_CALUDE_quiz_probability_l3855_385567

theorem quiz_probability : 
  let n_questions : ℕ := 5
  let n_choices : ℕ := 6
  let p_correct : ℚ := 1 / n_choices
  let p_incorrect : ℚ := 1 - p_correct
  1 - p_incorrect ^ n_questions = 4651 / 7776 :=
by sorry

end NUMINAMATH_CALUDE_quiz_probability_l3855_385567


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3855_385524

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0) ↔ 
  (a < -2 ∨ a ≥ 6/5) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3855_385524


namespace NUMINAMATH_CALUDE_problem_1_l3855_385532

theorem problem_1 : (1) - 3 + 8 - 15 - 6 = -16 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3855_385532


namespace NUMINAMATH_CALUDE_part_1_part_2_l3855_385553

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Define the function g
def g (x : ℝ) : ℝ := f 2 x - |x + 1|

-- Theorem for part 1
theorem part_1 : ∃ (a : ℝ), (∀ x, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) ∧ a = 2 := by sorry

-- Theorem for part 2
theorem part_2 : ∃ (min_value : ℝ), (∀ x, g x ≥ min_value) ∧ min_value = -1/2 := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_l3855_385553


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l3855_385566

theorem smallest_n_satisfying_inequality : 
  ∀ n : ℕ, n > 0 → (1 / n - 1 / (n + 1) < 1 / 15) ↔ n ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l3855_385566


namespace NUMINAMATH_CALUDE_choose_two_from_five_l3855_385589

theorem choose_two_from_five (n : ℕ) (k : ℕ) : n = 5 ∧ k = 2 → Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_five_l3855_385589


namespace NUMINAMATH_CALUDE_f_is_odd_iff_l3855_385538

-- Define the function f(x) = x|x + a| + b
def f (a b x : ℝ) : ℝ := x * abs (x + a) + b

-- State the theorem
theorem f_is_odd_iff (a b : ℝ) :
  (∀ x, f a b (-x) = -f a b x) ↔ 
  (∀ x, x * abs (-x + a) + b = -(x * abs (x + a) + b)) :=
by sorry

end NUMINAMATH_CALUDE_f_is_odd_iff_l3855_385538


namespace NUMINAMATH_CALUDE_flour_to_add_l3855_385573

/-- Given a recipe requiring a total amount of flour and an amount already added,
    calculate the remaining amount of flour to be added. -/
def remaining_flour (total : ℕ) (added : ℕ) : ℕ :=
  total - added

theorem flour_to_add : remaining_flour 10 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_flour_to_add_l3855_385573


namespace NUMINAMATH_CALUDE_xyz_sum_square_l3855_385509

theorem xyz_sum_square (x y z : ℕ+) 
  (h_gcd : Nat.gcd x.val (Nat.gcd y.val z.val) = 1)
  (h_x_div : x.val ∣ y.val * z.val * (x.val + y.val + z.val))
  (h_y_div : y.val ∣ x.val * z.val * (x.val + y.val + z.val))
  (h_z_div : z.val ∣ x.val * y.val * (x.val + y.val + z.val))
  (h_sum_div : (x.val + y.val + z.val) ∣ (x.val * y.val * z.val)) :
  ∃ (k : ℕ), x.val * y.val * z.val * (x.val + y.val + z.val) = k * k := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_square_l3855_385509


namespace NUMINAMATH_CALUDE_jace_neighbor_payment_l3855_385518

/-- Proves that Jace gave 0 cents to his neighbor -/
theorem jace_neighbor_payment (earned : ℕ) (debt : ℕ) (remaining : ℕ) : 
  earned = 1000 → debt = 358 → remaining = 642 → (earned - debt - remaining) * 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_jace_neighbor_payment_l3855_385518


namespace NUMINAMATH_CALUDE_diamond_example_l3855_385529

/-- Definition of the diamond operation for real numbers -/
def diamond (x y : ℝ) : ℝ := (x + y)^2 * (x - y)^2

/-- Theorem stating that 2 ◇ (3 ◇ 4) = 5745329 -/
theorem diamond_example : diamond 2 (diamond 3 4) = 5745329 := by
  sorry

end NUMINAMATH_CALUDE_diamond_example_l3855_385529


namespace NUMINAMATH_CALUDE_xy_sum_eleven_l3855_385523

-- Define the logarithm base 2 function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem xy_sum_eleven (x y : ℝ) 
  (hx : x ≥ 1) 
  (hy : y ≥ 1) 
  (hxy : x * y = 10) 
  (hineq : x^(log2 x) * y^(log2 y) ≥ 10) : 
  x + y = 11 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_eleven_l3855_385523


namespace NUMINAMATH_CALUDE_michaels_fruit_cost_l3855_385516

/-- Calculates the total cost of fruit for pies -/
def total_fruit_cost (peach_pies apple_pies blueberry_pies : ℕ) 
                     (fruit_per_pie : ℕ) 
                     (apple_blueberry_price peach_price : ℚ) : ℚ :=
  let peach_pounds := peach_pies * fruit_per_pie
  let apple_pounds := apple_pies * fruit_per_pie
  let blueberry_pounds := blueberry_pies * fruit_per_pie
  let apple_blueberry_cost := (apple_pounds + blueberry_pounds) * apple_blueberry_price
  let peach_cost := peach_pounds * peach_price
  apple_blueberry_cost + peach_cost

/-- Theorem: The total cost of fruit for Michael's pie order is $51.00 -/
theorem michaels_fruit_cost :
  total_fruit_cost 5 4 3 3 1 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_michaels_fruit_cost_l3855_385516


namespace NUMINAMATH_CALUDE_right_triangle_trig_l3855_385598

/-- Given a right-angled triangle ABC where ∠A = 90° and tan C = 2,
    prove that cos C = √5/5 and sin C = 2√5/5 -/
theorem right_triangle_trig (A B C : ℝ) (h1 : A = Real.pi / 2) (h2 : Real.tan C = 2) :
  Real.cos C = Real.sqrt 5 / 5 ∧ Real.sin C = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_l3855_385598


namespace NUMINAMATH_CALUDE_lindseys_remaining_money_l3855_385581

/-- Calculates Lindsey's remaining money after saving and spending --/
theorem lindseys_remaining_money (september : ℝ) (october : ℝ) (november : ℝ) 
  (h_sep : september = 50)
  (h_oct : october = 37)
  (h_nov : november = 11)
  (h_dec : december = november * 1.1)
  (h_mom_bonus : total_savings > 75 → mom_bonus = total_savings * 0.2)
  (h_spending : spending = (total_savings + mom_bonus) * 0.75)
  : remaining_money = 33.03 :=
by
  sorry

where
  december : ℝ := november * 1.1
  total_savings : ℝ := september + october + november + december
  mom_bonus : ℝ := if total_savings > 75 then total_savings * 0.2 else 0
  spending : ℝ := (total_savings + mom_bonus) * 0.75
  remaining_money : ℝ := total_savings + mom_bonus - spending

end NUMINAMATH_CALUDE_lindseys_remaining_money_l3855_385581


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3855_385533

theorem trig_expression_equality : 
  (Real.sin (38 * π / 180) * Real.sin (38 * π / 180) + 
   Real.cos (38 * π / 180) * Real.sin (52 * π / 180) - 
   Real.tan (15 * π / 180) ^ 2) / 
  (3 * Real.tan (15 * π / 180)) = 
  (2 + Real.sqrt 3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3855_385533


namespace NUMINAMATH_CALUDE_triangle_inequality_l3855_385500

theorem triangle_inequality (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h1 : a * y + b * x = c)
  (h2 : c * x + a * z = b)
  (h3 : b * z + c * y = a) :
  (x / (1 - y * z)) + (y / (1 - z * x)) + (z / (1 - x * y)) ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3855_385500


namespace NUMINAMATH_CALUDE_inscribed_circle_sector_ratio_l3855_385592

theorem inscribed_circle_sector_ratio :
  ∀ (R r : ℝ),
  R > 0 → r > 0 →
  R = (2 * Real.sqrt 3 + 3) * r / 3 →
  (π * r^2) / ((π * R^2) / 6) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_sector_ratio_l3855_385592


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l3855_385586

theorem cube_volume_ratio : 
  let cube_volume (edge : ℚ) : ℚ := edge ^ 3
  let cube1_edge : ℚ := 4
  let cube2_edge : ℚ := 10
  (cube_volume cube1_edge) / (cube_volume cube2_edge) = 8 / 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l3855_385586


namespace NUMINAMATH_CALUDE_special_triangle_bisecting_lines_angle_l3855_385564

/-- Triangle with specific side lengths -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_eq : a = 13
  b_eq : b = 14
  c_eq : c = 15

/-- A line that bisects both perimeter and area of the triangle -/
structure BisectingLine (t : SpecialTriangle) where
  bisects_perimeter : Bool
  bisects_area : Bool

/-- The acute angle between two bisecting lines -/
def acute_angle (t : SpecialTriangle) (l1 l2 : BisectingLine t) : ℝ := sorry

theorem special_triangle_bisecting_lines_angle 
  (t : SpecialTriangle) 
  (l1 l2 : BisectingLine t) 
  (h_unique : ∀ (l : BisectingLine t), l = l1 ∨ l = l2) :
  Real.tan (acute_angle t l1 l2) = Real.sqrt 6 / 12 := by sorry

end NUMINAMATH_CALUDE_special_triangle_bisecting_lines_angle_l3855_385564


namespace NUMINAMATH_CALUDE_unique_sequence_sum_property_l3855_385568

-- Define the sequence type
def UniqueIntegerSequence := ℕ+ → ℕ+

-- Define the property that every positive integer occurs exactly once
def IsUniqueSequence (a : UniqueIntegerSequence) : Prop :=
  ∀ n : ℕ+, ∃! k : ℕ+, a k = n

-- State the theorem
theorem unique_sequence_sum_property (a : UniqueIntegerSequence) 
    (h : IsUniqueSequence a) : 
    ∃ ℓ m : ℕ+, 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_sum_property_l3855_385568


namespace NUMINAMATH_CALUDE_minimum_handshakes_l3855_385537

theorem minimum_handshakes (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  (n * k) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_minimum_handshakes_l3855_385537


namespace NUMINAMATH_CALUDE_smallest_stairs_l3855_385511

theorem smallest_stairs (n : ℕ) : 
  n > 15 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 → 
  n ≥ 52 :=
by sorry

end NUMINAMATH_CALUDE_smallest_stairs_l3855_385511


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3855_385562

theorem absolute_value_inequality (x : ℝ) : 
  ‖‖x - 2‖ - 1‖ ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3855_385562


namespace NUMINAMATH_CALUDE_thirtieth_term_is_61_l3855_385515

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem thirtieth_term_is_61 :
  arithmetic_sequence 3 2 30 = 61 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_is_61_l3855_385515


namespace NUMINAMATH_CALUDE_A_inter_B_eq_open_interval_l3855_385551

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | -Real.sqrt 3 < x ∧ x < Real.sqrt 3}

-- State the theorem
theorem A_inter_B_eq_open_interval : A ∩ B = {x | 0 < x ∧ x < Real.sqrt 3} := by sorry

end NUMINAMATH_CALUDE_A_inter_B_eq_open_interval_l3855_385551


namespace NUMINAMATH_CALUDE_democrat_ratio_l3855_385544

theorem democrat_ratio (total_participants : ℕ) 
  (female_democrats : ℕ) (total_democrats : ℕ) :
  total_participants = 750 →
  female_democrats = 125 →
  total_democrats = total_participants / 3 →
  female_democrats * 2 ≤ total_participants →
  (total_democrats - female_democrats) * 4 = 
    total_participants - female_democrats * 2 := by
  sorry

end NUMINAMATH_CALUDE_democrat_ratio_l3855_385544


namespace NUMINAMATH_CALUDE_modulus_of_pure_imaginary_z_l3855_385563

/-- If z = (x^2 - 1) + (x - 1)i where x is a real number and z is a pure imaginary number, then |z| = 2 -/
theorem modulus_of_pure_imaginary_z (x : ℝ) (z : ℂ) 
  (h1 : z = Complex.mk (x^2 - 1) (x - 1))
  (h2 : z.re = 0) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_pure_imaginary_z_l3855_385563


namespace NUMINAMATH_CALUDE_shop_monthly_rent_l3855_385587

/-- The monthly rent of a rectangular shop given its dimensions and annual rent per square foot -/
def monthly_rent (length width annual_rent_per_sqft : ℕ) : ℕ :=
  length * width * annual_rent_per_sqft / 12

/-- Proof that the monthly rent of a shop with given dimensions is 3600 -/
theorem shop_monthly_rent :
  monthly_rent 18 20 120 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_shop_monthly_rent_l3855_385587


namespace NUMINAMATH_CALUDE_common_chord_length_O1_O2_l3855_385546

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The length of the common chord between two circles -/
def common_chord_length (c1 c2 : Circle) : ℝ := sorry

/-- Circle O₁ with equation (x+1)²+(y-3)²=9 -/
def O1 : Circle :=
  { equation := λ x y ↦ (x + 1)^2 + (y - 3)^2 = 9 }

/-- Circle O₂ with equation x²+y²-4x+2y-11=0 -/
def O2 : Circle :=
  { equation := λ x y ↦ x^2 + y^2 - 4*x + 2*y - 11 = 0 }

/-- Theorem stating that the length of the common chord between O₁ and O₂ is 24/5 -/
theorem common_chord_length_O1_O2 :
  common_chord_length O1 O2 = 24/5 := by sorry

end NUMINAMATH_CALUDE_common_chord_length_O1_O2_l3855_385546


namespace NUMINAMATH_CALUDE_target_hitting_probability_l3855_385534

theorem target_hitting_probability (miss_prob : ℝ) (hit_prob : ℝ) :
  miss_prob = 0.20 →
  hit_prob = 1 - miss_prob →
  hit_prob = 0.80 :=
by
  sorry

end NUMINAMATH_CALUDE_target_hitting_probability_l3855_385534


namespace NUMINAMATH_CALUDE_next_free_haircut_in_ten_l3855_385539

-- Define the constants from the problem
def haircuts_per_free : ℕ := 14
def free_haircuts_received : ℕ := 5
def total_haircuts : ℕ := 79

-- Define a function to calculate the number of haircuts until the next free one
def haircuts_until_next_free (total : ℕ) (free : ℕ) (per_free : ℕ) : ℕ :=
  per_free - (total - free) % per_free

-- Theorem statement
theorem next_free_haircut_in_ten :
  haircuts_until_next_free total_haircuts free_haircuts_received haircuts_per_free = 10 := by
  sorry


end NUMINAMATH_CALUDE_next_free_haircut_in_ten_l3855_385539


namespace NUMINAMATH_CALUDE_haleigh_dogs_count_l3855_385576

/-- The number of cats Haleigh has -/
def num_cats : ℕ := 3

/-- The total number of leggings needed -/
def total_leggings : ℕ := 14

/-- The number of leggings each animal needs -/
def leggings_per_animal : ℕ := 1

/-- The number of dogs Haleigh has -/
def num_dogs : ℕ := total_leggings - (num_cats * leggings_per_animal)

theorem haleigh_dogs_count : num_dogs = 11 := by sorry

end NUMINAMATH_CALUDE_haleigh_dogs_count_l3855_385576


namespace NUMINAMATH_CALUDE_sequence_length_6_to_202_l3855_385535

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Proof that the arithmetic sequence from 6 to 202 with step 2 has 99 terms -/
theorem sequence_length_6_to_202 : 
  arithmeticSequenceLength 6 202 2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_6_to_202_l3855_385535


namespace NUMINAMATH_CALUDE_triangle_area_is_168_l3855_385548

-- Define the function representing the curve
def f (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define x-intercepts
def x_intercept1 : ℝ := 4
def x_intercept2 : ℝ := -3

-- Define y-intercept
def y_intercept : ℝ := f 0

-- Theorem statement
theorem triangle_area_is_168 :
  let base : ℝ := x_intercept1 - x_intercept2
  let height : ℝ := y_intercept
  (1/2 : ℝ) * base * height = 168 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_168_l3855_385548


namespace NUMINAMATH_CALUDE_farmer_euclid_field_l3855_385531

theorem farmer_euclid_field (a b c x : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c^2 = a^2 + b^2)
  (h4 : (b / c) * x + (a / c) * x = 3) :
  (a * b / 2 - x^2) / (a * b / 2) = 2393 / 2890 := by sorry

end NUMINAMATH_CALUDE_farmer_euclid_field_l3855_385531


namespace NUMINAMATH_CALUDE_problem_solution_l3855_385588

theorem problem_solution (x y : Real) 
  (h1 : x + Real.cos y = 3009)
  (h2 : x + 3009 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 3009 + Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3855_385588


namespace NUMINAMATH_CALUDE_partner_a_profit_share_l3855_385540

/-- Calculates the share of profit for partner A given the initial investments,
    changes after 8 months, and total profit at the end of the year. -/
theorem partner_a_profit_share
  (a_initial : ℕ)
  (b_initial : ℕ)
  (a_change : ℤ)
  (b_change : ℕ)
  (total_profit : ℕ)
  (h1 : a_initial = 6000)
  (h2 : b_initial = 4000)
  (h3 : a_change = -1000)
  (h4 : b_change = 1000)
  (h5 : total_profit = 630) :
  ((a_initial * 8 + (a_initial + a_change) * 4) * total_profit) /
  ((a_initial * 8 + (a_initial + a_change) * 4) + (b_initial * 8 + (b_initial + b_change) * 4)) = 357 :=
by sorry

end NUMINAMATH_CALUDE_partner_a_profit_share_l3855_385540


namespace NUMINAMATH_CALUDE_remainder_property_l3855_385597

theorem remainder_property (x : ℕ) (h : x > 0) :
  (100 % x = 4) → ((100 + x) % x = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_property_l3855_385597


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3855_385549

/-- Given a line segment with midpoint (-3, 4) and one endpoint (0, 2),
    prove that the other endpoint is (-6, 6) -/
theorem line_segment_endpoint (midpoint endpoint1 endpoint2 : ℝ × ℝ) :
  midpoint = (-3, 4) →
  endpoint1 = (0, 2) →
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (-6, 6) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3855_385549


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3855_385584

theorem infinitely_many_solutions (b : ℝ) : 
  (∀ x : ℝ, 3 * (5 + b * x) = 18 * x + 15) → b = 6 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l3855_385584


namespace NUMINAMATH_CALUDE_common_point_theorem_l3855_385503

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Constructs a line based on the given conditions -/
def construct_line (a d r : ℝ) : Line :=
  { a := a
  , b := a + d
  , c := a * r + 2 * d }

theorem common_point_theorem (a d r : ℝ) :
  (construct_line a d r).contains (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_common_point_theorem_l3855_385503


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3855_385519

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx - 2 > 0 ↔ -4 < x ∧ x < 1) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3855_385519


namespace NUMINAMATH_CALUDE_fifty_cent_items_count_l3855_385545

/-- Represents the number of items at each price point -/
structure ItemCounts where
  fiftyc : ℕ
  twofifty : ℕ
  four : ℕ

/-- Calculates the total number of items -/
def total_items (c : ItemCounts) : ℕ :=
  c.fiftyc + c.twofifty + c.four

/-- Calculates the total cost in cents -/
def total_cost (c : ItemCounts) : ℕ :=
  50 * c.fiftyc + 250 * c.twofifty + 400 * c.four

/-- The main theorem to prove -/
theorem fifty_cent_items_count (c : ItemCounts) :
  total_items c = 50 ∧ total_cost c = 5000 → c.fiftyc = 40 := by
  sorry

#check fifty_cent_items_count

end NUMINAMATH_CALUDE_fifty_cent_items_count_l3855_385545


namespace NUMINAMATH_CALUDE_cube_equals_self_mod_thousand_l3855_385558

theorem cube_equals_self_mod_thousand (n : ℕ) :
  100 ≤ n ∧ n ≤ 999 →
  (n^3 % 1000 = n) ↔ (n = 376 ∨ n = 625) := by
  sorry

end NUMINAMATH_CALUDE_cube_equals_self_mod_thousand_l3855_385558


namespace NUMINAMATH_CALUDE_domain_of_g_l3855_385557

-- Define the function f with domain [0,2]
def f : Set ℝ := Set.Icc 0 2

-- Define the function g(x) = f(x²)
def g (x : ℝ) : Prop := x^2 ∈ f

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x} = Set.Icc (-Real.sqrt 2) (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l3855_385557


namespace NUMINAMATH_CALUDE_excircle_incircle_relation_l3855_385560

/-- Given a triangle ABC with inscribed circle radius r, excircle radii r_a, r_b, r_c,
    and semiperimeter p, prove that (r_a * r_b * r_c) / r = p^2 -/
theorem excircle_incircle_relation (r r_a r_b r_c p : ℝ) : r > 0 → r_a > 0 → r_b > 0 → r_c > 0 → p > 0 →
  (r_a * r_b * r_c) / r = p^2 := by sorry

end NUMINAMATH_CALUDE_excircle_incircle_relation_l3855_385560


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3855_385574

def complex_number : ℂ := Complex.I * (1 + Complex.I)

theorem complex_number_in_third_quadrant :
  Real.sign (complex_number.re) = -1 ∧ Real.sign (complex_number.im) = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3855_385574


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l3855_385555

theorem cosine_sine_identity : 
  Real.cos (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.sin (160 * π / 180) * Real.sin (10 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l3855_385555


namespace NUMINAMATH_CALUDE_valid_scheduling_orders_l3855_385507

def number_of_lecturers : ℕ := 7

def number_of_dependencies : ℕ := 2

theorem valid_scheduling_orders :
  (number_of_lecturers.factorial / 2^number_of_dependencies : ℕ) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_valid_scheduling_orders_l3855_385507


namespace NUMINAMATH_CALUDE_factorial_ratio_l3855_385522

theorem factorial_ratio : (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3855_385522


namespace NUMINAMATH_CALUDE_log_inequality_l3855_385517

theorem log_inequality (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  Real.log ((4 * x - 5) / |x - 2|) / Real.log (x^2) ≥ 1/2 ↔ -1 + Real.sqrt 6 ≤ x ∧ x ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l3855_385517


namespace NUMINAMATH_CALUDE_problem_solution_l3855_385594

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the problem conditions
def problem_conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  ∃ (p q r : ℕ), 
    Real.sqrt (log10 a) = p ∧
    Real.sqrt (log10 b) = q ∧
    log10 (Real.sqrt (a * b^2)) = r ∧
    p + q + r = 150

-- State the theorem
theorem problem_solution (a b : ℝ) : 
  problem_conditions a b → a^2 * b^3 = 10^443 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3855_385594


namespace NUMINAMATH_CALUDE_range_of_m_l3855_385561

/-- The function f(x) = -x^2 + 2x + 5 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 5

/-- Theorem stating the range of m given the conditions on f -/
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 6) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 6) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 5) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 5) →
  m ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3855_385561


namespace NUMINAMATH_CALUDE_square_operation_l3855_385572

theorem square_operation (x y : ℝ) (h1 : y = 68.70953354520753) (h2 : y^2 - x^2 = 4321) :
  ∃ (z : ℝ), z^2 = x^2 ∧ z = x :=
sorry

end NUMINAMATH_CALUDE_square_operation_l3855_385572


namespace NUMINAMATH_CALUDE_duck_ratio_is_two_to_one_l3855_385505

/-- The ratio of ducks at North Pond to Lake Michigan -/
def duck_ratio (north_pond : ℕ) (lake_michigan : ℕ) : ℚ :=
  (north_pond : ℚ) / (lake_michigan : ℚ)

theorem duck_ratio_is_two_to_one :
  let lake_michigan : ℕ := 100
  let north_pond : ℕ := 206
  ∀ R : ℚ, north_pond = lake_michigan * R + 6 →
    duck_ratio north_pond lake_michigan = 2 := by
  sorry

end NUMINAMATH_CALUDE_duck_ratio_is_two_to_one_l3855_385505


namespace NUMINAMATH_CALUDE_remainder_of_2685976_div_8_l3855_385582

theorem remainder_of_2685976_div_8 : 2685976 % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2685976_div_8_l3855_385582


namespace NUMINAMATH_CALUDE_maximal_k_inequality_l3855_385530

theorem maximal_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ k : ℝ, (k + a/b) * (k + b/c) * (k + c/a) ≤ (a/b + b/c + c/a) * (b/a + c/b + a/c) ↔ k ≤ Real.rpow 9 (1/3) - 1 :=
by sorry

end NUMINAMATH_CALUDE_maximal_k_inequality_l3855_385530


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3855_385590

/-- The quadratic equation x^2 - 2x - 6 = 0 has two distinct real roots -/
theorem quadratic_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*x₁ - 6 = 0) ∧ 
  (x₂^2 - 2*x₂ - 6 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3855_385590


namespace NUMINAMATH_CALUDE_cricket_average_score_l3855_385510

theorem cricket_average_score 
  (avg_2_matches : ℝ) 
  (avg_5_matches : ℝ) 
  (num_matches : ℕ) 
  (h1 : avg_2_matches = 20) 
  (h2 : avg_5_matches = 26) 
  (h3 : num_matches = 5) :
  let remaining_matches := num_matches - 2
  let total_score_5 := avg_5_matches * num_matches
  let total_score_2 := avg_2_matches * 2
  let remaining_score := total_score_5 - total_score_2
  remaining_score / remaining_matches = 30 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_score_l3855_385510


namespace NUMINAMATH_CALUDE_triangle_properties_l3855_385512

-- Define the triangle ABC
def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  a = 2 * Real.sqrt 2 ∧ b = 5 ∧ c = Real.sqrt 13

-- Theorem to prove the three parts of the problem
theorem triangle_properties {A B C a b c : Real} 
  (h : triangle_ABC A B C a b c) : 
  C = π / 4 ∧ 
  Real.sin A = 2 * Real.sqrt 13 / 13 ∧ 
  Real.sin (2 * A + π / 4) = 17 * Real.sqrt 2 / 26 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3855_385512


namespace NUMINAMATH_CALUDE_ratio_DO_OP_l3855_385580

/-- Parallelogram ABCD with points P on AB and Q on BC -/
structure Parallelogram (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D P Q O : V)
  (is_parallelogram : (C - B) = (D - A))
  (P_on_AB : ∃ t : ℝ, P = A + t • (B - A) ∧ 0 ≤ t ∧ t ≤ 1)
  (Q_on_BC : ∃ s : ℝ, Q = B + s • (C - B) ∧ 0 ≤ s ∧ s ≤ 1)
  (prop_AB_BP : 3 • (B - A) = 7 • (P - B))
  (prop_BC_BQ : 3 • (C - B) = 4 • (Q - B))
  (O_intersect : ∃ r t : ℝ, O = A + r • (Q - A) ∧ O = D + t • (P - D))

/-- The ratio DO : OP is 7 : 3 -/
theorem ratio_DO_OP (V : Type*) [AddCommGroup V] [Module ℝ V] (para : Parallelogram V) :
  ∃ k : ℝ, para.D - para.O = (7 * k) • (para.O - para.P) ∧ k ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ratio_DO_OP_l3855_385580


namespace NUMINAMATH_CALUDE_borrowed_sheets_theorem_l3855_385565

/-- Represents a set of algebra notes -/
structure AlgebraNotes where
  total_pages : ℕ
  total_sheets : ℕ
  borrowed_sheets : ℕ

/-- Calculates the average page number of remaining sheets -/
def average_page_number (notes : AlgebraNotes) : ℚ :=
  let remaining_sheets := notes.total_sheets - notes.borrowed_sheets
  let sum_of_remaining_pages := (notes.total_pages * (notes.total_pages + 1)) / 2 -
    (notes.borrowed_sheets * 2 * (notes.borrowed_sheets * 2 + 1)) / 2
  sum_of_remaining_pages / (2 * remaining_sheets)

/-- Main theorem: The average page number of remaining sheets is 31 when 20 sheets are borrowed -/
theorem borrowed_sheets_theorem (notes : AlgebraNotes)
  (h1 : notes.total_pages = 80)
  (h2 : notes.total_sheets = 40)
  (h3 : notes.borrowed_sheets = 20) :
  average_page_number notes = 31 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sheets_theorem_l3855_385565


namespace NUMINAMATH_CALUDE_min_occupied_seats_for_150_l3855_385569

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure that the next person must sit next to someone -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  (total_seats - 2) / 4 + 1

theorem min_occupied_seats_for_150 :
  min_occupied_seats 150 = 37 := by
  sorry

end NUMINAMATH_CALUDE_min_occupied_seats_for_150_l3855_385569


namespace NUMINAMATH_CALUDE_james_local_taxes_l3855_385502

/-- Calculates the amount of local taxes paid in cents per hour -/
def local_taxes_cents (hourly_wage : ℚ) (tax_rate : ℚ) : ℚ :=
  hourly_wage * 100 * tax_rate

theorem james_local_taxes :
  local_taxes_cents 25 (24/1000) = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_local_taxes_l3855_385502


namespace NUMINAMATH_CALUDE_monthly_rate_is_42_l3855_385508

/-- The monthly parking rate that satisfies the given conditions -/
def monthly_rate : ℚ :=
  let weekly_rate : ℚ := 10
  let weeks_per_year : ℕ := 52
  let months_per_year : ℕ := 12
  let yearly_savings : ℚ := 16
  (weekly_rate * weeks_per_year - yearly_savings) / months_per_year

/-- Proof that the monthly parking rate is $42 -/
theorem monthly_rate_is_42 : monthly_rate = 42 := by
  sorry

end NUMINAMATH_CALUDE_monthly_rate_is_42_l3855_385508


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l3855_385550

open Real

/-- A function f(x) = kx - ln(x) is monotonically increasing on (1, +∞) if and only if k ≥ 1 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > 1, StrictMono (fun x => k * x - log x)) ↔ k ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l3855_385550


namespace NUMINAMATH_CALUDE_intersection_line_slope_l3855_385501

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 2*y + 40 = 0

-- Define the intersection points
def intersection (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧
  circle2 C.1 C.2 ∧ circle2 D.1 D.2 ∧
  C ≠ D

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) (h : intersection C D) :
  (D.2 - C.2) / (D.1 - C.1) = -2/3 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l3855_385501


namespace NUMINAMATH_CALUDE_eulers_formula_l3855_385585

/-- A convex polyhedron with vertices, edges, and faces. -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces

/-- Euler's formula for convex polyhedra -/
theorem eulers_formula (P : ConvexPolyhedron) : P.V - P.E + P.F = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l3855_385585


namespace NUMINAMATH_CALUDE_equal_selection_probability_l3855_385596

/-- Represents a sampling method -/
structure SamplingMethod where
  -- Add necessary fields here
  reasonable : Bool

/-- Represents a sample from a population -/
structure Sample where
  size : ℕ
  method : SamplingMethod

/-- Represents the probability of an individual being selected in a sample -/
def selectionProbability (s : Sample) (individual : ℕ) : ℝ :=
  -- Definition would go here
  sorry

theorem equal_selection_probability 
  (s1 s2 : Sample) 
  (h1 : s1.size = s2.size) 
  (h2 : s1.method.reasonable) 
  (h3 : s2.method.reasonable) 
  (individual : ℕ) : 
  selectionProbability s1 individual = selectionProbability s2 individual :=
sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l3855_385596


namespace NUMINAMATH_CALUDE_product_103_97_l3855_385570

theorem product_103_97 : 103 * 97 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_103_97_l3855_385570


namespace NUMINAMATH_CALUDE_total_price_is_1797_80_l3855_385552

/-- The price of Marion's bike in dollars -/
def marion_bike_price : ℝ := 356

/-- The price of Stephanie's bike before discount in dollars -/
def stephanie_bike_price_before_discount : ℝ := 2 * marion_bike_price

/-- The discount percentage Stephanie received -/
def stephanie_discount_percent : ℝ := 0.1

/-- The price of Patrick's bike before promotion in dollars -/
def patrick_bike_price_before_promotion : ℝ := 3 * marion_bike_price

/-- The percentage of the original price Patrick pays -/
def patrick_payment_percent : ℝ := 0.75

/-- The total price paid for all three bikes -/
def total_price : ℝ := 
  marion_bike_price + 
  stephanie_bike_price_before_discount * (1 - stephanie_discount_percent) + 
  patrick_bike_price_before_promotion * patrick_payment_percent

/-- Theorem stating that the total price paid for the three bikes is $1797.80 -/
theorem total_price_is_1797_80 : total_price = 1797.80 := by
  sorry

end NUMINAMATH_CALUDE_total_price_is_1797_80_l3855_385552


namespace NUMINAMATH_CALUDE_linear_function_proof_l3855_385578

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

theorem linear_function_proof (f : ℝ → ℝ) 
  (h_linear : is_linear f) 
  (h_composite : ∀ x, f (f x) = 4 * x - 1) 
  (h_specific : f 3 = -5) : 
  ∀ x, f x = -2 * x + 1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_proof_l3855_385578


namespace NUMINAMATH_CALUDE_pens_in_drawer_l3855_385528

/-- The number of pens in Maria's desk drawer -/
def total_pens (red_pens black_pens blue_pens : ℕ) : ℕ :=
  red_pens + black_pens + blue_pens

/-- Theorem stating the total number of pens in Maria's desk drawer -/
theorem pens_in_drawer : 
  let red_pens : ℕ := 8
  let black_pens : ℕ := red_pens + 10
  let blue_pens : ℕ := red_pens + 7
  total_pens red_pens black_pens blue_pens = 41 := by
  sorry

end NUMINAMATH_CALUDE_pens_in_drawer_l3855_385528


namespace NUMINAMATH_CALUDE_symmetric_point_l3855_385583

/-- Given two points A and B in a plane, find the symmetric point A' of A with respect to B. -/
theorem symmetric_point (A B : ℝ × ℝ) (A' : ℝ × ℝ) : 
  A = (2, 1) → B = (-3, 7) → 
  (B.1 = (A.1 + A'.1) / 2 ∧ B.2 = (A.2 + A'.2) / 2) →
  A' = (-8, 13) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_l3855_385583


namespace NUMINAMATH_CALUDE_tangent_line_to_polar_curve_l3855_385521

/-- Given a line in polar coordinates ρcos(θ + π/3) = 1 tangent to a curve ρ = r (r > 0),
    prove that r = 1 -/
theorem tangent_line_to_polar_curve (r : ℝ) (h1 : r > 0) : 
  (∃ θ : ℝ, r * Real.cos (θ + π/3) = 1) → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_polar_curve_l3855_385521


namespace NUMINAMATH_CALUDE_jake_brought_one_balloon_l3855_385541

/-- The number of balloons Allan and Jake brought to the park in total -/
def total_balloons : ℕ := 3

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := total_balloons - allan_balloons

/-- Theorem stating that Jake brought 1 balloon to the park -/
theorem jake_brought_one_balloon : jake_balloons = 1 := by
  sorry

end NUMINAMATH_CALUDE_jake_brought_one_balloon_l3855_385541


namespace NUMINAMATH_CALUDE_max_sum_of_three_naturals_l3855_385506

theorem max_sum_of_three_naturals (a b c : ℕ) (h1 : a + b = 1014) (h2 : c - b = 497) (h3 : a > b) :
  (∀ a' b' c' : ℕ, a' + b' = 1014 → c' - b' = 497 → a' > b' → a' + b' + c' ≤ a + b + c) ∧
  a + b + c = 2017 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_three_naturals_l3855_385506


namespace NUMINAMATH_CALUDE_cafeteria_extra_apples_l3855_385504

/-- The number of extra apples in the cafeteria -/
def extra_apples (red_apples green_apples students_wanting_fruit : ℕ) : ℕ :=
  red_apples + green_apples - students_wanting_fruit

/-- Theorem: The cafeteria ends up with 40 extra apples -/
theorem cafeteria_extra_apples :
  extra_apples 42 7 9 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_extra_apples_l3855_385504


namespace NUMINAMATH_CALUDE_complement_A_union_B_equals_target_l3855_385575

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℝ := {y : ℝ | 1 ≤ y ∧ y ≤ 3}

-- State the theorem
theorem complement_A_union_B_equals_target :
  (Set.compl A) ∪ B = {x : ℝ | x < 0 ∨ x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_equals_target_l3855_385575


namespace NUMINAMATH_CALUDE_binomial_26_6_l3855_385556

theorem binomial_26_6 (h1 : Nat.choose 24 5 = 42504) 
                      (h2 : Nat.choose 24 6 = 134596) 
                      (h3 : Nat.choose 23 5 = 33649) : 
  Nat.choose 26 6 = 230230 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l3855_385556


namespace NUMINAMATH_CALUDE_binary_1101_equals_13_l3855_385527

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- The binary representation of 1101₂ -/
def binary_1101 : List Bool := [true, true, false, true]

theorem binary_1101_equals_13 : binary_to_decimal binary_1101 = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_equals_13_l3855_385527


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_max_value_g_l3855_385543

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

-- Define the function g
def g (x : ℝ) : ℝ := f x - x^2 + x

-- Theorem for the maximum value of g(x)
theorem max_value_g :
  ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 5/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_max_value_g_l3855_385543


namespace NUMINAMATH_CALUDE_largest_n_with_perfect_squares_l3855_385536

theorem largest_n_with_perfect_squares : ∃ (N : ℤ),
  (∃ (a : ℤ), N + 496 = a^2) ∧
  (∃ (b : ℤ), N + 224 = b^2) ∧
  (∀ (M : ℤ), M > N →
    ¬(∃ (x : ℤ), M + 496 = x^2) ∨
    ¬(∃ (y : ℤ), M + 224 = y^2)) ∧
  N = 4265 :=
sorry

end NUMINAMATH_CALUDE_largest_n_with_perfect_squares_l3855_385536


namespace NUMINAMATH_CALUDE_max_identical_end_digits_of_square_l3855_385520

theorem max_identical_end_digits_of_square (n : ℕ) (h : n % 10 ≠ 0) :
  ∀ k : ℕ, k > 4 → ∃ d : ℕ, d < 10 ∧ (n^2) % (10^k) ≠ d * ((10^k - 1) / 9) :=
sorry

end NUMINAMATH_CALUDE_max_identical_end_digits_of_square_l3855_385520


namespace NUMINAMATH_CALUDE_last_four_digits_5_pow_2011_l3855_385599

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem last_four_digits_5_pow_2011 :
  last_four_digits (5^2011) = 8125 :=
by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_5_pow_2011_l3855_385599


namespace NUMINAMATH_CALUDE_total_shaded_area_is_one_third_l3855_385593

/-- Represents the fractional area shaded at each step of the square division process -/
def shadedAreaSequence : ℕ → ℚ
  | 0 => 5 / 16
  | n + 1 => shadedAreaSequence n + 5 / 16^(n + 2)

/-- The theorem stating that the total shaded area is 1/3 of the square -/
theorem total_shaded_area_is_one_third :
  (∑' n, shadedAreaSequence n) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_total_shaded_area_is_one_third_l3855_385593


namespace NUMINAMATH_CALUDE_min_sum_tangents_l3855_385514

/-- In an acute triangle ABC, given that a = 2b * sin(C), 
    the minimum value of tan(A) + tan(B) + tan(C) is 3√3 -/
theorem min_sum_tangents (A B C : Real) (a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a = 2 * b * Real.sin C ∧  -- Given condition
  a = c * Real.sin A ∧  -- Law of sines
  b = c * Real.sin B ∧  -- Law of sines
  c = c * Real.sin C  -- Law of sines
  →
  (Real.tan A + Real.tan B + Real.tan C ≥ 3 * (3 : Real).sqrt) ∧
  ∃ (A' B' C' : Real), 
    0 < A' ∧ 0 < B' ∧ 0 < C' ∧
    A' + B' + C' = π ∧
    Real.tan A' + Real.tan B' + Real.tan C' = 3 * (3 : Real).sqrt :=
by sorry

end NUMINAMATH_CALUDE_min_sum_tangents_l3855_385514


namespace NUMINAMATH_CALUDE_expand_binomials_l3855_385577

theorem expand_binomials (x y : ℝ) : (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l3855_385577


namespace NUMINAMATH_CALUDE_tangent_line_max_a_l3855_385579

/-- Given a real number a, if there exists a common tangent line to the curves y = x^2 and y = a ln x for x > 0, then a ≤ 2e -/
theorem tangent_line_max_a (a : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ 
    (∃ (m : ℝ), (2 * x = a / x) ∧ 
      (x^2 = a * Real.log x + m))) → 
  a ≤ 2 * Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_max_a_l3855_385579


namespace NUMINAMATH_CALUDE_sum_and_ratio_problem_l3855_385571

theorem sum_and_ratio_problem (x y : ℝ) 
  (sum_eq : x + y = 500)
  (ratio_eq : x / y = 4 / 5) :
  y - x = 500 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_ratio_problem_l3855_385571


namespace NUMINAMATH_CALUDE_circle_op_equation_solution_l3855_385547

-- Define the € operation
def circle_op (x y : ℝ) : ℝ := 3 * x * y

-- State the theorem
theorem circle_op_equation_solution :
  ∀ z : ℝ, circle_op (circle_op 4 5) z = 540 → z = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_op_equation_solution_l3855_385547


namespace NUMINAMATH_CALUDE_speedster_convertibles_l3855_385559

theorem speedster_convertibles (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) :
  speedsters = (2 : ℚ) / 3 * total →
  total - speedsters = 50 →
  convertibles = (4 : ℚ) / 5 * speedsters →
  convertibles = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_speedster_convertibles_l3855_385559


namespace NUMINAMATH_CALUDE_polyhedron_vertices_l3855_385513

/-- A polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices --/
structure Polyhedron where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- Euler's formula for polyhedra states that for a polyhedron with F faces, E edges, and V vertices, F - E + V = 2 --/
axiom euler_formula (p : Polyhedron) : p.faces - p.edges + p.vertices = 2

/-- Theorem: A polyhedron with 6 faces and 12 edges has 8 vertices --/
theorem polyhedron_vertices (p : Polyhedron) (h1 : p.faces = 6) (h2 : p.edges = 12) : p.vertices = 8 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_vertices_l3855_385513


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l3855_385591

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) :
  ∃ k : ℤ, (n - 1) + n + (n + 1) + (n + 2) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l3855_385591


namespace NUMINAMATH_CALUDE_optimal_carriages_and_passengers_l3855_385525

/-- The daily round-trip frequency as a function of the number of carriages -/
def y (x : ℕ) : ℝ := -2 * x + 24

/-- The total number of carriages operated daily -/
def S (x : ℕ) : ℝ := x * y x

/-- The daily number of passengers transported -/
def W (x : ℕ) : ℝ := 110 * S x

/-- The constraint on the number of carriages -/
def valid_x (x : ℕ) : Prop := 0 ≤ x ∧ x ≤ 12

theorem optimal_carriages_and_passengers :
  ∃ (x : ℕ), valid_x x ∧
    (∀ (x' : ℕ), valid_x x' → W x' ≤ W x) ∧
    x = 6 ∧
    W x = 7920 :=
sorry

end NUMINAMATH_CALUDE_optimal_carriages_and_passengers_l3855_385525


namespace NUMINAMATH_CALUDE_system_solution_l3855_385526

theorem system_solution : 
  ∀ x y : ℝ, 
    (3 * x + Real.sqrt (3 * x - y) + y = 6 ∧ 
     9 * x^2 + 3 * x - y - y^2 = 36) ↔ 
    ((x = 2 ∧ y = -3) ∨ (x = 6 ∧ y = -18)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3855_385526


namespace NUMINAMATH_CALUDE_weekly_average_expenditure_l3855_385554

/-- The average expenditure for a week given the average expenditures for two parts of the week -/
theorem weekly_average_expenditure 
  (first_three_days_avg : ℝ) 
  (next_four_days_avg : ℝ) 
  (h1 : first_three_days_avg = 350)
  (h2 : next_four_days_avg = 420) :
  (3 * first_three_days_avg + 4 * next_four_days_avg) / 7 = 390 := by
sorry

end NUMINAMATH_CALUDE_weekly_average_expenditure_l3855_385554


namespace NUMINAMATH_CALUDE_find_m_value_l3855_385595

theorem find_m_value (m : ℚ) : 
  (∃ (x y : ℚ), m * x - y = 4 ∧ x = 4 ∧ y = 3) → m = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_find_m_value_l3855_385595
