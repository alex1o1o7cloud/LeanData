import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_seating_theorem_l952_95267

/-- Represents the number of people at the table -/
def total_people : ℕ := 12

/-- Represents the number of math majors -/
def math_majors : ℕ := 5

/-- Represents the number of physics majors -/
def physics_majors : ℕ := 4

/-- Represents the number of biology majors -/
def biology_majors : ℕ := 3

/-- The probability of math and physics majors sitting consecutively -/
def consecutive_seating_probability : ℚ := 7 / 240

theorem consecutive_seating_theorem :
  let total := total_people
  let math := math_majors
  let physics := physics_majors
  let bio := biology_majors
  total = math + physics + bio →
  consecutive_seating_probability = 7 / 240 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_seating_theorem_l952_95267


namespace NUMINAMATH_CALUDE_smallest_additional_airplanes_lucas_airplanes_arrangement_l952_95285

theorem smallest_additional_airplanes (current_airplanes : ℕ) (row_size : ℕ) : ℕ :=
  let next_multiple := (current_airplanes + row_size - 1) / row_size * row_size
  next_multiple - current_airplanes

theorem lucas_airplanes_arrangement :
  smallest_additional_airplanes 37 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_additional_airplanes_lucas_airplanes_arrangement_l952_95285


namespace NUMINAMATH_CALUDE_combined_selling_price_l952_95287

/-- Calculate the combined selling price of three items with given costs, exchange rate, profits, discount, and tax. -/
theorem combined_selling_price (exchange_rate : ℝ) (cost_a cost_b cost_c : ℝ)
  (profit_a profit_b profit_c : ℝ) (discount_b tax : ℝ) :
  exchange_rate = 70 ∧
  cost_a = 10 ∧
  cost_b = 15 ∧
  cost_c = 20 ∧
  profit_a = 0.25 ∧
  profit_b = 0.30 ∧
  profit_c = 0.20 ∧
  discount_b = 0.10 ∧
  tax = 0.08 →
  let cost_rs_a := cost_a * exchange_rate
  let cost_rs_b := cost_b * exchange_rate * (1 - discount_b)
  let cost_rs_c := cost_c * exchange_rate
  let selling_price_a := cost_rs_a * (1 + profit_a) * (1 + tax)
  let selling_price_b := cost_rs_b * (1 + profit_b) * (1 + tax)
  let selling_price_c := cost_rs_c * (1 + profit_c) * (1 + tax)
  selling_price_a + selling_price_b + selling_price_c = 4086.18 := by
sorry


end NUMINAMATH_CALUDE_combined_selling_price_l952_95287


namespace NUMINAMATH_CALUDE_solution_to_system_l952_95250

theorem solution_to_system : 
  ∀ x y : ℝ, 
  3 * x^2 - 9 * y^2 = 0 → 
  x + y = 5 → 
  ((x = (15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 * Real.sqrt 3 - 5) / 2) ∨
   (x = (-15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 + 5 * Real.sqrt 3) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_to_system_l952_95250


namespace NUMINAMATH_CALUDE_inequality_solution_l952_95235

def solution_set (m : ℝ) : Set ℝ :=
  if m = -3 then { x | x > 1 }
  else if -3 < m ∧ m < -1 then { x | x < m / (m + 3) ∨ x > 1 }
  else if m < -3 then { x | 1 < x ∧ x < m / (m + 3) }
  else ∅

theorem inequality_solution (m : ℝ) (h : m < -1) :
  { x : ℝ | (m + 3) * x^2 - (2 * m + 3) * x + m > 0 } = solution_set m :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l952_95235


namespace NUMINAMATH_CALUDE_margie_change_l952_95273

/-- Calculates the change received after a purchase -/
def change_received (banana_price : ℚ) (orange_price : ℚ) (num_bananas : ℕ) (num_oranges : ℕ) (paid_amount : ℚ) : ℚ :=
  let total_cost := banana_price * num_bananas + orange_price * num_oranges
  paid_amount - total_cost

/-- Proves that Margie received $7.60 in change -/
theorem margie_change : 
  let banana_price : ℚ := 30/100
  let orange_price : ℚ := 60/100
  let num_bananas : ℕ := 4
  let num_oranges : ℕ := 2
  let paid_amount : ℚ := 10
  change_received banana_price orange_price num_bananas num_oranges paid_amount = 76/10 := by
  sorry

#eval change_received (30/100) (60/100) 4 2 10

end NUMINAMATH_CALUDE_margie_change_l952_95273


namespace NUMINAMATH_CALUDE_sequence_properties_l952_95227

def a (i : ℕ+) : ℕ := (7^(2^i.val) - 1) / 6

theorem sequence_properties :
  ∀ i : ℕ+,
    (∀ j : ℕ+, (a (j + 1)) % (a j) = 0) ∧
    (a i) % 3 ≠ 0 ∧
    (a i) % (2^(i.val + 2)) = 0 ∧
    (a i) % (2^(i.val + 3)) ≠ 0 ∧
    ∃ p : ℕ, ∃ n : ℕ, Prime p ∧ 6 * (a i) + 1 = p^n ∧
    ∃ x y : ℕ, a i = x^2 + y^2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l952_95227


namespace NUMINAMATH_CALUDE_expression_value_l952_95238

theorem expression_value (a b : ℝ) (h : 2 * a - 3 * b = 5) :
  10 - 4 * a + 6 * b = 0 := by sorry

end NUMINAMATH_CALUDE_expression_value_l952_95238


namespace NUMINAMATH_CALUDE_rug_overlap_problem_l952_95210

theorem rug_overlap_problem (total_rug_area single_coverage double_coverage triple_coverage : ℝ) :
  total_rug_area = 200 →
  single_coverage + double_coverage + triple_coverage = 140 →
  double_coverage = 24 →
  triple_coverage = 18 :=
by sorry

end NUMINAMATH_CALUDE_rug_overlap_problem_l952_95210


namespace NUMINAMATH_CALUDE_special_rectangle_side_lengths_l952_95283

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  AB : ℝ  -- Length of side AB
  BC : ℝ  -- Length of side BC
  ratio_condition : AB / BC = 7 / 5
  square_area : ℝ  -- Area of the common square
  square_area_value : square_area = 72

/-- Theorem stating the side lengths of the special rectangle -/
theorem special_rectangle_side_lengths (rect : SpecialRectangle) : 
  rect.AB = 42 ∧ rect.BC = 30 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_side_lengths_l952_95283


namespace NUMINAMATH_CALUDE_outfit_combinations_l952_95290

/-- The number of red shirts -/
def red_shirts : ℕ := 8

/-- The number of green shirts -/
def green_shirts : ℕ := 7

/-- The number of blue pants -/
def blue_pants : ℕ := 8

/-- The number of red hats -/
def red_hats : ℕ := 10

/-- The number of green hats -/
def green_hats : ℕ := 9

/-- The number of black belts -/
def black_belts : ℕ := 5

/-- The number of brown belts -/
def brown_belts : ℕ := 4

/-- The total number of possible outfits -/
def total_outfits : ℕ := red_shirts * blue_pants * green_hats * brown_belts + 
                         green_shirts * blue_pants * red_hats * black_belts

theorem outfit_combinations : total_outfits = 5104 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l952_95290


namespace NUMINAMATH_CALUDE_paint_calculation_l952_95258

theorem paint_calculation (initial_paint : ℚ) : 
  (1 / 4 * initial_paint + 1 / 6 * (3 / 4 * initial_paint) = 135) → 
  ⌈initial_paint⌉ = 463 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l952_95258


namespace NUMINAMATH_CALUDE_area_of_CALI_l952_95253

/-- Square BERK with side length 10 -/
def BERK : Set (ℝ × ℝ) := sorry

/-- Points T, O, W, N as midpoints of BE, ER, RK, KB respectively -/
def T : ℝ × ℝ := sorry
def O : ℝ × ℝ := sorry
def W : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

/-- Square CALI whose edges contain vertices of BERK -/
def CALI : Set (ℝ × ℝ) := sorry

/-- CA is parallel to BO -/
def CA_parallel_BO : Prop := sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem area_of_CALI : area CALI = 180 :=
sorry

end NUMINAMATH_CALUDE_area_of_CALI_l952_95253


namespace NUMINAMATH_CALUDE_cosine_squared_sum_equality_l952_95225

theorem cosine_squared_sum_equality (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2 * x))^2 + (Real.cos (3 * x))^2 = 1 ↔ 
  (∃ k : ℤ, x = (k * Real.pi / 2 + Real.pi / 4) ∨ x = (k * Real.pi / 3 + Real.pi / 6)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_squared_sum_equality_l952_95225


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_1080_l952_95219

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n items --/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of ways to divide 6 families into 4 groups and allocate to 4 villages --/
def allocation_schemes : ℕ :=
  let group_formations := choose 6 2 * choose 4 2 * choose 2 1 * choose 1 1 / (arrange 2 2 * arrange 2 2)
  let village_allocations := arrange 4 4
  group_formations * village_allocations

theorem allocation_schemes_eq_1080 : allocation_schemes = 1080 := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_eq_1080_l952_95219


namespace NUMINAMATH_CALUDE_max_value_ratio_l952_95252

theorem max_value_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_ratio_l952_95252


namespace NUMINAMATH_CALUDE_pizza_order_proof_l952_95277

theorem pizza_order_proof (num_people : ℕ) (slices_per_pizza : ℕ) (slices_per_person : ℕ) : 
  num_people = 6 → slices_per_pizza = 8 → slices_per_person = 4 →
  (num_people * slices_per_person) / slices_per_pizza = 3 := by
sorry

end NUMINAMATH_CALUDE_pizza_order_proof_l952_95277


namespace NUMINAMATH_CALUDE_expression_value_l952_95245

theorem expression_value (a : ℝ) (h : a = -3) : 
  (3 * a⁻¹ + a⁻¹ / 3) / a = 10 / 27 := by sorry

end NUMINAMATH_CALUDE_expression_value_l952_95245


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l952_95268

/-- The distance between the foci of a hyperbola with equation x^2/32 - y^2/8 = 1 is 4√10 -/
theorem hyperbola_foci_distance :
  ∀ (x y : ℝ),
  x^2 / 32 - y^2 / 8 = 1 →
  ∃ (f₁ f₂ : ℝ × ℝ),
  (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (4 * Real.sqrt 10)^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l952_95268


namespace NUMINAMATH_CALUDE_pipe_A_fill_time_l952_95298

/-- The time (in minutes) it takes for pipe B to fill the tanker alone -/
def time_B : ℝ := 40

/-- The total time (in minutes) it takes to fill the tanker when B is used for half the time
    and A and B fill it together for the other half -/
def total_time : ℝ := 29.999999999999993

/-- The time (in minutes) it takes for pipe A to fill the tanker alone -/
def time_A : ℝ := 60

/-- Proves that the time taken by pipe A to fill the tanker alone is 60 minutes -/
theorem pipe_A_fill_time :
  (time_B / 2) / time_B + (total_time / 2) * (1 / time_A + 1 / time_B) = 1 :=
sorry

end NUMINAMATH_CALUDE_pipe_A_fill_time_l952_95298


namespace NUMINAMATH_CALUDE_A_empty_iff_A_singleton_iff_A_singleton_elements_l952_95289

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

theorem A_empty_iff (a : ℝ) : A a = ∅ ↔ a > 9/8 := by sorry

theorem A_singleton_iff (a : ℝ) : (∃! x, x ∈ A a) ↔ a = 0 ∨ a = 9/8 := by sorry

theorem A_singleton_elements (a : ℝ) :
  (a = 0 → A a = {2/3}) ∧ (a = 9/8 → A a = {4/3}) := by sorry

end NUMINAMATH_CALUDE_A_empty_iff_A_singleton_iff_A_singleton_elements_l952_95289


namespace NUMINAMATH_CALUDE_problem_solution_l952_95200

theorem problem_solution (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 4 = d + 2 * Real.sqrt (a + b + c - d)) : 
  d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l952_95200


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l952_95208

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l952_95208


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l952_95294

theorem least_addition_for_divisibility :
  ∃! x : ℕ, x < 23 ∧ (1053 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1053 + y) % 23 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l952_95294


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l952_95293

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation z(1+i) = 2i
def equation (z : ℂ) : Prop := z * (1 + i) = 2 * i

-- Define the fourth quadrant
def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant :
  ∃ z : ℂ, equation z ∧ fourth_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l952_95293


namespace NUMINAMATH_CALUDE_problem_solution_l952_95274

theorem problem_solution (x y : ℝ) : 
  ((x + 2)^3 < x^3 + 8*x^2 + 42*x + 27) ∧
  ((x^3 + 8*x^2 + 42*x + 27) < (x + 4)^3) ∧
  (y = x + 3) ∧
  ((x + 3)^3 = x^3 + 9*x^2 + 27*x + 27) ∧
  ((x + 3)^3 = x^3 + 8*x^2 + 42*x + 27) ∧
  (x^2 = 15*x) →
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l952_95274


namespace NUMINAMATH_CALUDE_rectangle_area_change_l952_95291

theorem rectangle_area_change 
  (l w : ℝ) 
  (h_pos_l : l > 0) 
  (h_pos_w : w > 0) : 
  let new_length := 1.1 * l
  let new_width := 0.9 * w
  let new_area := new_length * new_width
  let original_area := l * w
  new_area / original_area = 0.99 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l952_95291


namespace NUMINAMATH_CALUDE_test_questions_count_l952_95232

theorem test_questions_count : ∀ (total_questions : ℕ),
  (total_questions % 4 = 0) →  -- Test consists of 4 equal sections
  (20 : ℝ) / total_questions > 0.60 →  -- Percentage correct > 60%
  (20 : ℝ) / total_questions < 0.70 →  -- Percentage correct < 70%
  total_questions = 32 := by
sorry

end NUMINAMATH_CALUDE_test_questions_count_l952_95232


namespace NUMINAMATH_CALUDE_point_inside_iff_odd_intersections_l952_95236

/-- A closed, non-self-intersecting path in a plane. -/
structure ClosedPath :=
  (path : Set (ℝ × ℝ))
  (closed : IsClosed path)
  (non_self_intersecting : ∀ x y : ℝ × ℝ, x ∈ path → y ∈ path → x ≠ y → (∃ t : ℝ, 0 < t ∧ t < 1 ∧ (1 - t) • x + t • y ∉ path))

/-- A point in the plane. -/
def Point := ℝ × ℝ

/-- The number of intersections between a line segment and a path. -/
def intersectionCount (p q : Point) (path : ClosedPath) : ℕ :=
  sorry

/-- A point is known to be outside the region bounded by the path. -/
def isOutside (p : Point) (path : ClosedPath) : Prop :=
  sorry

/-- A point is inside the region bounded by the path. -/
def isInside (p : Point) (path : ClosedPath) : Prop :=
  ∀ q : Point, isOutside q path → Odd (intersectionCount p q path)

theorem point_inside_iff_odd_intersections (p : Point) (path : ClosedPath) :
  isInside p path ↔ ∀ q : Point, isOutside q path → Odd (intersectionCount p q path) :=
sorry

end NUMINAMATH_CALUDE_point_inside_iff_odd_intersections_l952_95236


namespace NUMINAMATH_CALUDE_hyperbola_implies_constant_difference_constant_difference_not_sufficient_l952_95297

-- Define a point in a plane
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a trajectory as a function from time to point
def Trajectory := ℝ → Point

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define a hyperbola
def isHyperbola (t : Trajectory) : Prop := sorry

-- Define the constant difference property
def hasConstantDifference (t : Trajectory) : Prop :=
  ∃ (F₁ F₂ : Point) (k : ℝ), ∀ (time : ℝ),
    |distance (t time) F₁ - distance (t time) F₂| = k

theorem hyperbola_implies_constant_difference (t : Trajectory) :
  isHyperbola t → hasConstantDifference t := by sorry

theorem constant_difference_not_sufficient (t : Trajectory) :
  ∃ t, hasConstantDifference t ∧ ¬isHyperbola t := by sorry

end NUMINAMATH_CALUDE_hyperbola_implies_constant_difference_constant_difference_not_sufficient_l952_95297


namespace NUMINAMATH_CALUDE_domain_shift_l952_95228

-- Define a function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_shifted : Set ℝ := Set.Icc (-2) 3

-- Theorem statement
theorem domain_shift (h : ∀ x, f (x + 1) ∈ domain_f_shifted ↔ x ∈ domain_f_shifted) :
  (∀ x, f x ∈ Set.Icc (-1) 4 ↔ x ∈ Set.Icc (-1) 4) :=
sorry

end NUMINAMATH_CALUDE_domain_shift_l952_95228


namespace NUMINAMATH_CALUDE_mean_daily_profit_l952_95211

theorem mean_daily_profit (days_in_month : ℕ) (first_half_mean : ℚ) (second_half_mean : ℚ) :
  days_in_month = 30 →
  first_half_mean = 225 →
  second_half_mean = 475 →
  (first_half_mean * 15 + second_half_mean * 15) / days_in_month = 350 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_daily_profit_l952_95211


namespace NUMINAMATH_CALUDE_book_arrangement_l952_95204

theorem book_arrangement (n : ℕ) (a b c : ℕ) (h1 : n = a + b + c) (h2 : a = 3) (h3 : b = 2) (h4 : c = 2) :
  (n.factorial) / (a.factorial * b.factorial) = 420 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_l952_95204


namespace NUMINAMATH_CALUDE_kittens_given_to_friends_l952_95224

/-- The number of kittens Alyssa initially had -/
def initial_kittens : ℕ := 8

/-- The number of kittens Alyssa has left -/
def remaining_kittens : ℕ := 4

/-- The number of kittens Alyssa gave to her friends -/
def kittens_given_away : ℕ := initial_kittens - remaining_kittens

theorem kittens_given_to_friends :
  kittens_given_away = 4 := by sorry

end NUMINAMATH_CALUDE_kittens_given_to_friends_l952_95224


namespace NUMINAMATH_CALUDE_equation_root_l952_95226

theorem equation_root : ∃ x : ℝ, (2 / x = 3 / (x + 1)) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_l952_95226


namespace NUMINAMATH_CALUDE_g_five_equals_248_l952_95248

theorem g_five_equals_248 (g : ℤ → ℤ) 
  (h1 : g 1 > 1)
  (h2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y)
  (h3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1) :
  g 5 = 248 := by sorry

end NUMINAMATH_CALUDE_g_five_equals_248_l952_95248


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l952_95209

theorem painted_cube_theorem (n : ℕ) (h : n > 0) :
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l952_95209


namespace NUMINAMATH_CALUDE_jonah_running_time_l952_95288

/-- Represents the problem of determining Jonah's running time. -/
theorem jonah_running_time (calories_per_hour : ℕ) (extra_time : ℕ) (extra_calories : ℕ) : 
  calories_per_hour = 30 →
  extra_time = 5 →
  extra_calories = 90 →
  ∃ (actual_time : ℕ), 
    actual_time * calories_per_hour = (actual_time + extra_time) * calories_per_hour - extra_calories ∧
    actual_time = 2 :=
by sorry

end NUMINAMATH_CALUDE_jonah_running_time_l952_95288


namespace NUMINAMATH_CALUDE_M_inter_complement_N_eq_l952_95292

/-- The universal set U (real numbers) -/
def U : Set ℝ := Set.univ

/-- Set M defined as {x | -2 ≤ x ≤ 2} -/
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

/-- Set N defined as the domain of y = ln(x-1), which is {x | x > 1} -/
def N : Set ℝ := {x | x > 1}

/-- Theorem stating that the intersection of M and the complement of N in U
    is equal to the set {x | -2 ≤ x ≤ 1} -/
theorem M_inter_complement_N_eq :
  M ∩ (U \ N) = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_M_inter_complement_N_eq_l952_95292


namespace NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l952_95216

theorem neg_p_sufficient_not_necessary_for_neg_q :
  let p := {x : ℝ | x < -1}
  let q := {x : ℝ | x < -4}
  (∀ x, x ∉ p → x ∉ q) ∧ (∃ x, x ∉ q ∧ x ∈ p) := by
  sorry

end NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l952_95216


namespace NUMINAMATH_CALUDE_first_number_10th_group_l952_95282

/-- Sequence term definition -/
def a (n : ℕ) : ℤ := 2 * n - 3

/-- Sum of first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The first number in the kth group -/
def first_in_group (k : ℕ) : ℕ := sum_first_n (k - 1) + 1

theorem first_number_10th_group :
  a (first_in_group 10) = 89 :=
sorry

end NUMINAMATH_CALUDE_first_number_10th_group_l952_95282


namespace NUMINAMATH_CALUDE_unique_solution_abs_equation_l952_95264

theorem unique_solution_abs_equation :
  ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_abs_equation_l952_95264


namespace NUMINAMATH_CALUDE_fred_stickers_l952_95221

theorem fred_stickers (jerry george fred : ℕ) 
  (h1 : jerry = 3 * george)
  (h2 : george = fred - 6)
  (h3 : jerry = 36) : 
  fred = 18 := by
sorry

end NUMINAMATH_CALUDE_fred_stickers_l952_95221


namespace NUMINAMATH_CALUDE_additional_candles_l952_95259

/-- 
Given:
- initial_candles: The initial number of candles on Molly's birthday cake
- current_age: Molly's current age
Prove that the number of additional candles is equal to current_age - initial_candles
-/
theorem additional_candles (initial_candles current_age : ℕ) :
  initial_candles = 14 →
  current_age = 20 →
  current_age - initial_candles = 6 := by
  sorry

end NUMINAMATH_CALUDE_additional_candles_l952_95259


namespace NUMINAMATH_CALUDE_prob_sum_five_is_one_ninth_l952_95241

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := faces * faces

/-- The number of favorable outcomes (sum of 5) when rolling two dice -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a sum of 5 with two dice -/
def prob_sum_five : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_five_is_one_ninth :
  prob_sum_five = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_sum_five_is_one_ninth_l952_95241


namespace NUMINAMATH_CALUDE_decagon_perimeter_decagon_perimeter_30_l952_95278

/-- The perimeter of a regular decagon with side length 3 units is 30 units. -/
theorem decagon_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun (num_sides : ℝ) (side_length : ℝ) (perimeter : ℝ) =>
    num_sides = 10 ∧ side_length = 3 → perimeter = num_sides * side_length

/-- The theorem applied to our specific case. -/
theorem decagon_perimeter_30 : decagon_perimeter 10 3 30 := by
  sorry

end NUMINAMATH_CALUDE_decagon_perimeter_decagon_perimeter_30_l952_95278


namespace NUMINAMATH_CALUDE_unique_prime_pair_sum_73_l952_95207

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem unique_prime_pair_sum_73 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p ≠ q ∧ p + q = 73 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_pair_sum_73_l952_95207


namespace NUMINAMATH_CALUDE_prism_volume_l952_95295

/-- The volume of a right rectangular prism with specific face areas and dimension ratio -/
theorem prism_volume (l w h : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  l * w = 10 → w * h = 18 → l * h = 36 →
  l = 2 * w →
  l * w * h = 36 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_l952_95295


namespace NUMINAMATH_CALUDE_problem_solution_l952_95242

theorem problem_solution (a b : ℕ) 
  (sum_eq : a + b = 31462)
  (b_div_20 : b % 20 = 0)
  (a_eq_b_div_10 : a = b / 10) : 
  b - a = 28462 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l952_95242


namespace NUMINAMATH_CALUDE_polygon_area_is_six_l952_95284

/-- The vertices of the polygon -/
def vertices : List (ℤ × ℤ) := [
  (0, 0), (0, 2), (1, 2), (2, 3), (2, 2), (3, 2), (3, 0), (2, 0), (2, 1), (1, 0)
]

/-- Calculate the area of a polygon given its vertices using the Shoelace formula -/
def polygonArea (vs : List (ℤ × ℤ)) : ℚ :=
  let pairs := vs.zip (vs.rotate 1)
  let sum := pairs.foldl (fun acc (p, q) => acc + (p.1 * q.2 - p.2 * q.1)) 0
  (sum.natAbs : ℚ) / 2

/-- The theorem stating that the area of the given polygon is 6 square units -/
theorem polygon_area_is_six :
  polygonArea vertices = 6 := by sorry

end NUMINAMATH_CALUDE_polygon_area_is_six_l952_95284


namespace NUMINAMATH_CALUDE_base_r_transaction_l952_95262

def base_r_to_decimal (digits : List Nat) (r : Nat) : Nat :=
  digits.foldl (fun acc d => acc * r + d) 0

theorem base_r_transaction (r : Nat) : r = 8 :=
  by
  have h1 : base_r_to_decimal [4, 4, 0] r + base_r_to_decimal [3, 4, 0] r = base_r_to_decimal [1, 0, 0, 0] r :=
    sorry
  sorry

end NUMINAMATH_CALUDE_base_r_transaction_l952_95262


namespace NUMINAMATH_CALUDE_projectile_max_height_l952_95214

/-- Represents the elevation of a projectile at time t -/
def elevation (t : ℝ) : ℝ := 200 * t - 10 * t^2

/-- The time at which the projectile reaches its maximum height -/
def max_height_time : ℝ := 10

theorem projectile_max_height :
  ∀ t : ℝ, elevation t ≤ elevation max_height_time ∧
  elevation max_height_time = 1000 := by
  sorry

#check projectile_max_height

end NUMINAMATH_CALUDE_projectile_max_height_l952_95214


namespace NUMINAMATH_CALUDE_limit_proof_l952_95272

/-- The limit of (2 - e^(arcsin^2(√x)))^(3/x) as x approaches 0 is e^(-3) -/
theorem limit_proof : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ →
  |(2 - Real.exp (Real.arcsin (Real.sqrt x))^2)^(3/x) - Real.exp (-3)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_proof_l952_95272


namespace NUMINAMATH_CALUDE_charlie_calculation_l952_95299

theorem charlie_calculation (x : ℝ) : 
  (x / 7 + 20 = 21) → (x * 7 - 20 = 29) := by
sorry

end NUMINAMATH_CALUDE_charlie_calculation_l952_95299


namespace NUMINAMATH_CALUDE_min_sum_of_product_3960_l952_95203

theorem min_sum_of_product_3960 (a b c : ℕ+) (h : a * b * c = 3960) :
  (∀ x y z : ℕ+, x * y * z = 3960 → a + b + c ≤ x + y + z) ∧ a + b + c = 72 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_3960_l952_95203


namespace NUMINAMATH_CALUDE_annulus_equal_area_division_l952_95223

theorem annulus_equal_area_division (r : ℝ) : 
  r > 0 ∧ r < 14 ∧ 
  (π * (14^2 - r^2) = π * (r^2 - 2^2)) → 
  r = 10 := by sorry

end NUMINAMATH_CALUDE_annulus_equal_area_division_l952_95223


namespace NUMINAMATH_CALUDE_special_function_at_one_seventh_l952_95239

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 0 1 → f x ∈ Set.Icc 0 1) ∧
  f 0 = 0 ∧ f 1 = 1 ∧
  ∃ a : ℝ, 0 ≤ a ∧ a ≤ 1 ∧
    ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x ≤ y →
      f ((x + y) / 2) = (1 - a) * f x + a * f y

theorem special_function_at_one_seventh (f : ℝ → ℝ) (h : special_function f) :
  f (1/7) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_one_seventh_l952_95239


namespace NUMINAMATH_CALUDE_soap_weight_calculation_l952_95220

/-- Calculates the total weight of soap bars given the weights of other items and suitcase weights. -/
theorem soap_weight_calculation (initial_weight final_weight perfume_weight chocolate_weight jam_weight : ℝ) : 
  initial_weight = 5 →
  perfume_weight = 5 * 1.2 / 16 →
  chocolate_weight = 4 →
  jam_weight = 2 * 8 / 16 →
  final_weight = 11 →
  final_weight - initial_weight - (perfume_weight + chocolate_weight + jam_weight) = 0.625 := by
  sorry

#check soap_weight_calculation

end NUMINAMATH_CALUDE_soap_weight_calculation_l952_95220


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_range_l952_95254

/-- The range of k for which the line y = kx intersects the hyperbola x^2/9 - y^2/4 = 1 -/
theorem line_hyperbola_intersection_range :
  ∀ k : ℝ, 
  (∃ x y : ℝ, y = k * x ∧ x^2 / 9 - y^2 / 4 = 1) ↔ 
  -2/3 < k ∧ k < 2/3 :=
by sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_range_l952_95254


namespace NUMINAMATH_CALUDE_johnson_volunteers_count_l952_95256

def total_volunteers (math_classes : ℕ) (students_per_class : ℕ) (teacher_volunteers : ℕ) (additional_needed : ℕ) : ℕ :=
  math_classes * students_per_class + teacher_volunteers + additional_needed

theorem johnson_volunteers_count :
  total_volunteers 6 5 13 7 = 50 := by
  sorry

end NUMINAMATH_CALUDE_johnson_volunteers_count_l952_95256


namespace NUMINAMATH_CALUDE_unique_polynomial_with_given_value_l952_95251

/-- A polynomial with natural number coefficients less than 10 -/
def PolynomialWithSmallCoeffs (p : Polynomial ℕ) : Prop :=
  ∀ i, (p.coeff i) < 10

theorem unique_polynomial_with_given_value :
  ∀ p : Polynomial ℕ,
  PolynomialWithSmallCoeffs p →
  p.eval 10 = 1248 →
  p = Polynomial.monomial 3 1 + Polynomial.monomial 2 2 + Polynomial.monomial 1 4 + Polynomial.monomial 0 8 :=
by sorry

end NUMINAMATH_CALUDE_unique_polynomial_with_given_value_l952_95251


namespace NUMINAMATH_CALUDE_factorization_x4_plus_81_l952_95205

theorem factorization_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6*x + 9) * (x^2 - 6*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_plus_81_l952_95205


namespace NUMINAMATH_CALUDE_potion_combinations_eq_thirteen_l952_95206

/-- The number of ways to combine roots and minerals for a potion. -/
def potionCombinations : ℕ :=
  let totalRoots : ℕ := 3
  let totalMinerals : ℕ := 5
  let incompatibleCombinations : ℕ := 2
  totalRoots * totalMinerals - incompatibleCombinations

/-- Theorem stating that the number of potion combinations is 13. -/
theorem potion_combinations_eq_thirteen : potionCombinations = 13 := by
  sorry

end NUMINAMATH_CALUDE_potion_combinations_eq_thirteen_l952_95206


namespace NUMINAMATH_CALUDE_sarah_marriage_age_l952_95265

/-- The game that predicts marriage age based on name, age, birth month, and siblings' ages -/
def marriage_age_prediction 
  (name_length : ℕ) 
  (age : ℕ) 
  (birth_month : ℕ) 
  (sibling_ages : List ℕ) : ℕ :=
  let step1 := name_length + 2 * age
  let step2 := step1 * (sibling_ages.sum)
  let step3 := step2 / (sibling_ages.length)
  step3 * birth_month

/-- Theorem stating that Sarah's predicted marriage age is 966 -/
theorem sarah_marriage_age : 
  marriage_age_prediction 5 9 7 [5, 7] = 966 := by
  sorry

#eval marriage_age_prediction 5 9 7 [5, 7]

end NUMINAMATH_CALUDE_sarah_marriage_age_l952_95265


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l952_95266

def initial_amount : ℕ := 120
def hamburger_cost : ℕ := 4
def milkshake_cost : ℕ := 3
def hamburgers_bought : ℕ := 8
def milkshakes_bought : ℕ := 6

theorem money_left_after_purchase : 
  initial_amount - (hamburger_cost * hamburgers_bought + milkshake_cost * milkshakes_bought) = 70 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l952_95266


namespace NUMINAMATH_CALUDE_exactly_seven_numbers_l952_95257

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def swap_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_cube (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m^3

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ is_perfect_cube (n - swap_digits n)

theorem exactly_seven_numbers :
  ∃! (s : Finset ℕ), s.card = 7 ∧ ∀ n, n ∈ s ↔ satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_exactly_seven_numbers_l952_95257


namespace NUMINAMATH_CALUDE_negation_equivalence_l952_95275

-- Define the original statement
def P : Prop := ∀ n : ℤ, 3 ∣ n → Odd n

-- Define the correct negation
def not_P : Prop := ∃ n : ℤ, 3 ∣ n ∧ ¬(Odd n)

-- Theorem stating that not_P is indeed the negation of P
theorem negation_equivalence : ¬P ↔ not_P := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l952_95275


namespace NUMINAMATH_CALUDE_car_distance_traveled_l952_95281

theorem car_distance_traveled (time : ℝ) (speed : ℝ) (distance : ℝ) :
  time = 11 →
  speed = 65 →
  distance = speed * time →
  distance = 715 := by sorry

end NUMINAMATH_CALUDE_car_distance_traveled_l952_95281


namespace NUMINAMATH_CALUDE_expression_quadrupled_l952_95280

variables (x y : ℝ) (h : x ≠ y)

theorem expression_quadrupled :
  (2*x)^2 * (2*y) / (2*x - 2*y) = 4 * (x^2 * y / (x - y)) :=
sorry

end NUMINAMATH_CALUDE_expression_quadrupled_l952_95280


namespace NUMINAMATH_CALUDE_bill_apples_left_l952_95222

/-- The number of apples Bill has left after distributing to teachers and baking pies -/
def apples_left (initial_apples : ℕ) (num_children : ℕ) (apples_per_teacher : ℕ) 
  (num_teachers_per_child : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - (num_children * apples_per_teacher * num_teachers_per_child) - (num_pies * apples_per_pie)

/-- Theorem stating that Bill has 18 apples left -/
theorem bill_apples_left : 
  apples_left 50 2 3 2 2 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_bill_apples_left_l952_95222


namespace NUMINAMATH_CALUDE_max_ate_second_most_l952_95218

-- Define the children as a finite type
inductive Child : Type
  | Chris : Child
  | Max : Child
  | Brandon : Child
  | Kayla : Child
  | Tanya : Child

-- Define the eating relation
def ate_more_than (a b : Child) : Prop := sorry

-- Define the conditions
axiom chris_ate_more_than_max : ate_more_than Child.Chris Child.Max
axiom brandon_ate_less_than_kayla : ate_more_than Child.Kayla Child.Brandon
axiom kayla_ate_less_than_max : ate_more_than Child.Max Child.Kayla
axiom kayla_ate_more_than_tanya : ate_more_than Child.Kayla Child.Tanya

-- Define what it means to be the second most
def is_second_most (c : Child) : Prop :=
  ∃ (first : Child), (first ≠ c) ∧
    (∀ (other : Child), other ≠ first → other ≠ c → ate_more_than c other)

-- The theorem to prove
theorem max_ate_second_most : is_second_most Child.Max := by sorry

end NUMINAMATH_CALUDE_max_ate_second_most_l952_95218


namespace NUMINAMATH_CALUDE_both_in_picture_probability_l952_95234

/-- Alice's lap time in seconds -/
def alice_lap_time : ℕ := 120

/-- Bob's lap time in seconds -/
def bob_lap_time : ℕ := 75

/-- Bob's start delay in seconds -/
def bob_start_delay : ℕ := 15

/-- Duration of one-third of the track for Alice in seconds -/
def alice_third_track : ℕ := alice_lap_time / 3

/-- Duration of one-third of the track for Bob in seconds -/
def bob_third_track : ℕ := bob_lap_time / 3

/-- Least common multiple of Alice and Bob's lap times -/
def lcm_lap_times : ℕ := lcm alice_lap_time bob_lap_time

/-- Time window for taking the picture in seconds -/
def picture_window : ℕ := 60

/-- Probability of both Alice and Bob being in the picture -/
def probability_both_in_picture : ℚ := 11 / 1200

theorem both_in_picture_probability :
  probability_both_in_picture = 11 / 1200 := by
  sorry

end NUMINAMATH_CALUDE_both_in_picture_probability_l952_95234


namespace NUMINAMATH_CALUDE_day_200_N_minus_1_is_wednesday_l952_95255

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  day : Nat

/-- Function to determine the day of the week for a given YearDay -/
def dayOfWeek (yd : YearDay) : DayOfWeek := sorry

theorem day_200_N_minus_1_is_wednesday 
  (N : Int)
  (h1 : dayOfWeek ⟨N, 400⟩ = DayOfWeek.Wednesday)
  (h2 : dayOfWeek ⟨N + 2, 300⟩ = DayOfWeek.Wednesday) :
  dayOfWeek ⟨N - 1, 200⟩ = DayOfWeek.Wednesday := by
  sorry

end NUMINAMATH_CALUDE_day_200_N_minus_1_is_wednesday_l952_95255


namespace NUMINAMATH_CALUDE_good_games_count_l952_95237

def games_from_friend : ℕ := 50
def games_from_garage_sale : ℕ := 27
def non_working_games : ℕ := 74

def total_games : ℕ := games_from_friend + games_from_garage_sale

theorem good_games_count : total_games - non_working_games = 3 := by
  sorry

end NUMINAMATH_CALUDE_good_games_count_l952_95237


namespace NUMINAMATH_CALUDE_proportions_sum_l952_95279

theorem proportions_sum (x y : ℚ) :
  (4 : ℚ) / 7 = x / 63 ∧ (4 : ℚ) / 7 = 84 / y → x + y = 183 := by
  sorry

end NUMINAMATH_CALUDE_proportions_sum_l952_95279


namespace NUMINAMATH_CALUDE_tonys_walking_speed_l952_95244

/-- Proves that Tony's walking speed is 2 MPH given the problem conditions -/
theorem tonys_walking_speed :
  let store_distance : ℝ := 4
  let running_speed : ℝ := 10
  let average_time_minutes : ℝ := 56
  let walking_speed : ℝ := 2

  (walking_speed * store_distance + 2 * (store_distance / running_speed) * 60) / 3 = average_time_minutes
  ∧ walking_speed > 0 := by sorry

end NUMINAMATH_CALUDE_tonys_walking_speed_l952_95244


namespace NUMINAMATH_CALUDE_rome_trip_notes_l952_95247

/-- Represents the number of notes carried by a person -/
structure Notes where
  euros : ℕ
  dollars : ℕ

/-- The total number of notes carried by both people -/
def total_notes (donald : Notes) (mona : Notes) : ℕ :=
  donald.euros + donald.dollars + mona.euros + mona.dollars

theorem rome_trip_notes :
  ∀ (donald : Notes),
    donald.euros + donald.dollars = 39 →
    donald.euros = donald.dollars →
    ∃ (mona : Notes),
      mona.euros = 3 * donald.euros ∧
      mona.dollars = donald.dollars ∧
      donald.euros + mona.euros = 2 * (donald.dollars + mona.dollars) ∧
      total_notes donald mona = 118 := by
  sorry


end NUMINAMATH_CALUDE_rome_trip_notes_l952_95247


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_l952_95215

theorem infinite_geometric_series_sum : 
  let a : ℚ := 2/5
  let r : ℚ := 1/2
  let series_sum := a / (1 - r)
  series_sum = 4/5 := by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_l952_95215


namespace NUMINAMATH_CALUDE_factor_expression_l952_95230

theorem factor_expression (x : ℝ) : 
  (20 * x^4 + 100 * x^2 - 10) - (-5 * x^4 + 15 * x^2 - 10) = 5 * x^2 * (5 * x^2 + 17) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l952_95230


namespace NUMINAMATH_CALUDE_complement_of_union_l952_95213

-- Define the universal set U
def U : Set ℕ := {x | x ≤ 9 ∧ x > 0}

-- Define sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

-- State the theorem
theorem complement_of_union :
  (U \ (A ∪ B)) = {7, 8, 9} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l952_95213


namespace NUMINAMATH_CALUDE_angle_inequality_l952_95217

open Real

theorem angle_inequality (a b c : ℝ) 
  (ha : a = sin (33 * π / 180))
  (hb : b = cos (55 * π / 180))
  (hc : c = tan (55 * π / 180)) :
  c > b ∧ b > a :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_l952_95217


namespace NUMINAMATH_CALUDE_f_monotone_increasing_on_interval_l952_95229

/-- The function f(x) = (1/2)^(x^2 - x - 1) is monotonically increasing on (-∞, 1/2) -/
theorem f_monotone_increasing_on_interval :
  ∀ x y : ℝ, x < y → x < (1/2 : ℝ) → y < (1/2 : ℝ) →
  ((1/2 : ℝ) ^ (x^2 - x - 1)) < ((1/2 : ℝ) ^ (y^2 - y - 1)) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_on_interval_l952_95229


namespace NUMINAMATH_CALUDE_product_of_roots_l952_95286

theorem product_of_roots (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂)
  (h₁ : x₁^2 - 2006*x₁ = 1)
  (h₂ : x₂^2 - 2006*x₂ = 1) : 
  x₁ * x₂ = -1 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_l952_95286


namespace NUMINAMATH_CALUDE_inequality_of_exponential_l952_95201

theorem inequality_of_exponential (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (1/3 : ℝ)^a < (1/3 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_exponential_l952_95201


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l952_95270

/-- The speed of cyclist C in mph -/
def speed_C : ℝ := 9

/-- The speed of cyclist D in mph -/
def speed_D : ℝ := speed_C + 6

/-- The distance between Newport and Kingston in miles -/
def distance : ℝ := 80

/-- The distance from Kingston where cyclists meet on D's return journey in miles -/
def meeting_distance : ℝ := 20

theorem cyclist_speed_problem :
  speed_C = 9 ∧
  speed_D = speed_C + 6 ∧
  distance / speed_C = (distance + meeting_distance) / speed_D :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l952_95270


namespace NUMINAMATH_CALUDE_expansion_coefficient_l952_95271

theorem expansion_coefficient (a : ℝ) (h1 : a > 0) 
  (h2 : (1 + 1) * (a + 1)^6 = 1458) : 
  (1 + 6 * 4) = 61 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l952_95271


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l952_95212

/-- The quadratic function f(x) = ax^2 + bx -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

/-- The equation f(x) = 6 -/
def equation (a b : ℝ) (x : ℝ) : Prop := f a b x = 6

theorem roots_of_quadratic (a b : ℝ) :
  equation a b (-2) ∧ equation a b 3 →
  (equation a b (-2) ∧ equation a b 3 ∧
   ∀ x : ℝ, equation a b x → x = -2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l952_95212


namespace NUMINAMATH_CALUDE_cafeteria_choices_theorem_l952_95202

/-- Represents the number of ways to choose foods in a cafeteria -/
def cafeteriaChoices (n : ℕ) : ℕ :=
  n + 1

/-- Theorem stating that the number of ways to choose n foods in the cafeteria is n + 1 -/
theorem cafeteria_choices_theorem (n : ℕ) : 
  cafeteriaChoices n = n + 1 := by
  sorry

/-- Apples are taken in groups of 3 -/
def appleGroup : ℕ := 3

/-- Yogurts are taken in pairs -/
def yogurtPair : ℕ := 2

/-- Maximum number of bread pieces allowed -/
def maxBread : ℕ := 2

/-- Maximum number of cereal bowls allowed -/
def maxCereal : ℕ := 1

end NUMINAMATH_CALUDE_cafeteria_choices_theorem_l952_95202


namespace NUMINAMATH_CALUDE_quotient_with_negative_remainder_l952_95260

theorem quotient_with_negative_remainder
  (dividend : ℤ)
  (divisor : ℤ)
  (remainder : ℤ)
  (h1 : dividend = 474232)
  (h2 : divisor = 800)
  (h3 : remainder = -968)
  (h4 : dividend = divisor * (dividend / divisor) + remainder) :
  dividend / divisor = 594 := by
  sorry

end NUMINAMATH_CALUDE_quotient_with_negative_remainder_l952_95260


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_condition_l952_95276

/-- The range of m for which the line y = kx + 1 always intersects 
    with the ellipse x²/5 + y²/m = 1 for any real k -/
theorem line_ellipse_intersection_condition (k : ℝ) :
  (∀ x y : ℝ, y = k * x + 1 → x^2 / 5 + y^2 / m = 1) ↔ (m ≥ 1 ∧ m ≠ 5) :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_condition_l952_95276


namespace NUMINAMATH_CALUDE_fraction_reduction_l952_95249

theorem fraction_reduction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^3 - b^3) / (a*b) - (a*b^2 - b^3) / (a*b - a^3) = (a^2 + a*b + b^2) / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l952_95249


namespace NUMINAMATH_CALUDE_arrangements_with_pair_together_eq_48_l952_95240

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange five people in a row, with two specific people always standing together -/
def arrangements_with_pair_together : ℕ :=
  factorial 4 * factorial 2

theorem arrangements_with_pair_together_eq_48 :
  arrangements_with_pair_together = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_pair_together_eq_48_l952_95240


namespace NUMINAMATH_CALUDE_arithmetic_subsequence_multiples_of_3_l952_95233

/-- An arithmetic sequence with common difference d -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- The subsequence of an arithmetic sequence with indices that are multiples of 3 -/
def SubsequenceMultiplesOf3 (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = a (3 * n)

theorem arithmetic_subsequence_multiples_of_3 (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) :
  ArithmeticSequence a d → SubsequenceMultiplesOf3 a b →
  ArithmeticSequence b (3 * d) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_subsequence_multiples_of_3_l952_95233


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l952_95246

theorem partial_fraction_decomposition (x : ℝ) (h : x ≠ 0) :
  (-2 * x^2 + 5 * x - 6) / (x^3 + 2 * x) = 
  (-3 : ℝ) / x + (x + 5) / (x^2 + 2) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l952_95246


namespace NUMINAMATH_CALUDE_junipers_bones_l952_95261

theorem junipers_bones (initial_bones : ℕ) : 
  (2 * initial_bones - 2 = 6) → initial_bones = 4 := by
  sorry

end NUMINAMATH_CALUDE_junipers_bones_l952_95261


namespace NUMINAMATH_CALUDE_correct_divisor_l952_95269

theorem correct_divisor (D : ℕ) (mistaken_divisor correct_quotient : ℕ) 
  (h1 : mistaken_divisor = 12)
  (h2 : D = mistaken_divisor * 35)
  (h3 : correct_quotient = 20)
  (h4 : D % (D / correct_quotient) = 0) :
  D / correct_quotient = 21 := by
sorry

end NUMINAMATH_CALUDE_correct_divisor_l952_95269


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l952_95263

def vector_a (m : ℝ) : Fin 2 → ℝ := ![m, 1]
def vector_b : Fin 2 → ℝ := ![3, 3]

def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

theorem perpendicular_vectors (m : ℝ) :
  dot_product (λ i => vector_a m i - vector_b i) vector_b = 0 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l952_95263


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l952_95296

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

end NUMINAMATH_CALUDE_fencing_cost_calculation_l952_95296


namespace NUMINAMATH_CALUDE_weekly_pay_solution_l952_95231

def weekly_pay_problem (y_pay : ℝ) (x_percentage : ℝ) : Prop :=
  let x_pay := x_percentage * y_pay
  x_pay + y_pay = 638

theorem weekly_pay_solution :
  weekly_pay_problem 290 1.2 :=
by sorry

end NUMINAMATH_CALUDE_weekly_pay_solution_l952_95231


namespace NUMINAMATH_CALUDE_difference_of_squares_305_295_l952_95243

theorem difference_of_squares_305_295 : 305^2 - 295^2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_305_295_l952_95243
