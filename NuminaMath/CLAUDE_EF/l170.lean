import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_composition_equality_l170_17072

-- Define the piecewise function h
noncomputable def h (x : ℝ) : ℝ :=
  if x < 0 then x else 3 * x - 17

-- Theorem statement
theorem h_composition_equality {b : ℝ} (hb : b < 0) :
  h (h (h 7)) = h (h (h b)) ↔ b = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_composition_equality_l170_17072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l170_17009

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x - 3) / (x + 1)

theorem f_minimum_value :
  ∀ x ∈ Set.Icc 0 1, f x ≥ -4 ∧ ∃ y ∈ Set.Icc 0 1, f y = -4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l170_17009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l170_17080

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 7771) :
  Int.gcd (8 * b^2 + 55 * b + 144) (4 * b + 19) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l170_17080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_inequality_l170_17084

theorem sin_cos_inequality (α β : Real) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2) :
  (Real.sin α)^3 / Real.sin β + (Real.cos α)^3 / Real.cos β ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_inequality_l170_17084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_given_field_l170_17085

/-- Represents a rectangular field with one uncovered side -/
structure RectangularField where
  area : ℝ
  uncoveredSide : ℝ

/-- Calculates the fencing required for a rectangular field -/
noncomputable def fencingRequired (field : RectangularField) : ℝ :=
  let width := field.area / field.uncoveredSide
  2 * width + field.uncoveredSide

/-- Theorem: The fencing required for the given field is 74 feet -/
theorem fencing_for_given_field :
  let field : RectangularField := { area := 680, uncoveredSide := 40 }
  fencingRequired field = 74 := by
  -- Unfold the definition of fencingRequired
  unfold fencingRequired
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_given_field_l170_17085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_3_nonnegative_range_l170_17023

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log_3_nonnegative_range :
  {x : ℝ | f x ≥ 0} = Set.Icc 3 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_3_nonnegative_range_l170_17023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_reduction_percentage_l170_17046

/-- Given a tax reduction and consumption increase, calculate the percentage of tax reduction -/
theorem tax_reduction_percentage 
  (original_tax : ℝ) 
  (original_consumption : ℝ) 
  (tax_reduction_percent : ℝ) 
  (consumption_increase_percent : ℝ) 
  (revenue_decrease_percent : ℝ)
  (original_tax_positive : 0 < original_tax)
  (original_consumption_positive : 0 < original_consumption)
  (consumption_increase : consumption_increase_percent = 9)
  (revenue_decrease : revenue_decrease_percent = 14.98)
  (h : (1 - tax_reduction_percent / 100) * (1 + consumption_increase_percent / 100) = 
       (1 - revenue_decrease_percent / 100)) :
  ∃ (ε : ℝ), abs (tax_reduction_percent - 22.01) < ε ∧ ε > 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_reduction_percentage_l170_17046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_loss_l170_17056

/-- Calculates the overall profit or loss percentage when selling two cars at the same price,
    with a gain on one and an equal percentage loss on the other. -/
theorem car_sale_profit_loss (price : ℝ) (percentage : ℝ) : 
  price > 0 → 
  percentage > 0 → 
  percentage < 100 →
  let cp1 := price / (1 + percentage / 100)
  let cp2 := price / (1 - percentage / 100)
  let total_cost := cp1 + cp2
  let total_sale := 2 * price
  let profit_loss := (total_sale - total_cost) / total_cost * 100
  profit_loss < 0 ∧ profit_loss > -2.25 :=
by
  intros h_price h_percentage h_percentage_lt_100
  -- The proof goes here
  sorry

-- Remove the #eval line as it's not necessary for the theorem and might cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_loss_l170_17056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_athlete_speed_theorem_l170_17021

/-- The speed of the faster athlete on a circular track -/
noncomputable def faster_athlete_speed (C : ℝ) (t1 : ℝ) (t2 : ℝ) : ℝ :=
  (C / t1 + C / t2) / 2

/-- Theorem: The speed of the faster athlete on a circular track -/
theorem faster_athlete_speed_theorem (C t1 t2 : ℝ) 
  (h1 : C > 0) (h2 : t1 > 0) (h3 : t2 > 0) :
  faster_athlete_speed C t1 t2 = (C / t1 + C / t2) / 2 :=
by
  -- Unfold the definition of faster_athlete_speed
  unfold faster_athlete_speed
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_athlete_speed_theorem_l170_17021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_g_l170_17039

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := x^2 + 4 * Real.sqrt x

-- State the theorem
theorem evaluate_g : 3 * g 3 - g 9 = -66 + 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_g_l170_17039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l170_17069

noncomputable def f (x : ℝ) : ℝ := 2016^x + Real.log (Real.sqrt (x^2 + 1) + x) / Real.log 2016 - 2016^(-x)

theorem f_solution_set : 
  {x : ℝ | f (3*x + 1) + f x > 0} = {x : ℝ | x > -1/4} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l170_17069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l170_17010

noncomputable def f (a b c x : ℝ) : ℝ := a + b * Real.cos x + c * Real.sin x

theorem range_of_a (a b c : ℝ) :
  (f a b c 0 = 1) →
  (f a b c (-π/4) = a) →
  (∀ x ∈ Set.Icc 0 (π/2), |f a b c x| ≤ Real.sqrt 2) →
  a ∈ Set.Icc 0 (4 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l170_17010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_to_origin_l170_17090

/-- The function representing the graph y = x + 1/x for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := x + 1/x

/-- The squared distance from a point (x, f(x)) to the origin -/
noncomputable def distance_squared (x : ℝ) : ℝ := x^2 + (f x)^2

/-- The x-coordinate of the point closest to the origin -/
noncomputable def closest_point_x : ℝ := 1 / Real.sqrt (Real.sqrt 2)

/-- The y-coordinate of the point closest to the origin -/
noncomputable def closest_point_y : ℝ := (1 + Real.sqrt 2) / Real.sqrt (Real.sqrt 2)

/-- Theorem stating that the chosen point is indeed the closest to the origin -/
theorem closest_point_to_origin :
  ∀ x > 0, distance_squared x ≥ distance_squared closest_point_x := by
  sorry

#check closest_point_to_origin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_to_origin_l170_17090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_of_f_alt_l170_17013

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := lg (1 - x)

-- Theorem stating the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iio 1 :=
by
  sorry -- The proof is omitted for now

-- Alternative formulation using set builder notation
theorem domain_of_f_alt :
  {x : ℝ | x < 1} = Set.Iio 1 :=
by
  sorry -- The proof is omitted for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_of_f_alt_l170_17013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_missing_angles_sum_l170_17064

/-- The sum of interior angles of a convex polygon with n sides is (n-2) * 180°. -/
axiom interior_angles_sum (n : ℕ) : n ≥ 3 → (n - 2) * 180 = (n - 2) * 180

/-- Given a convex polygon where the sum of all but two interior angles is 3240°,
    prove that the sum of the two remaining angles is 180°. -/
theorem two_missing_angles_sum (n : ℕ) (h : n ≥ 3) :
  (n - 2) * 180 = 3240 + 180 := by
  sorry

#check two_missing_angles_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_missing_angles_sum_l170_17064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_integer_values_part2_l170_17032

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (a^2 - a) / x^2 - 3 / x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 2 x > -1} = {x : ℝ | x < 0 ∨ (0 < x ∧ x < 1) ∨ x > 2} :=
by sorry

-- Part 2
theorem integer_values_part2 :
  {a : ℤ | ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f (a : ℝ) x < 6 / x^3} = {-2, -1, 0, 1, 2, 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_integer_values_part2_l170_17032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l170_17017

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4)

theorem monotone_decreasing_interval :
  ∀ x₁ x₂ : ℝ, 3 * Real.pi / 8 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 7 * Real.pi / 8 → f x₁ ≥ f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l170_17017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l170_17062

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 4

-- Define proposition p
def p (m : ℝ) : Prop := ∀ x ≥ 2, Monotone (fun x => f m x)

-- Define proposition q
def q (m : ℝ) : Prop := ∀ x, m*x^2 + 4*(m-2)*x + 4 > 0

-- Define the set of valid m values
def valid_m : Set ℝ := {m | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- Theorem statement
theorem m_range : valid_m = Set.Iic 1 ∪ Set.Ioo 2 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l170_17062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_less_than_two_l170_17092

theorem sin_plus_cos_less_than_two :
  ∀ x : ℝ, x ∈ Set.Icc 0 (π / 2) → Real.sin x + Real.cos x < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_less_than_two_l170_17092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l170_17078

/-- Given r > 0, the line cos θ + y sin θ = r is tangent to the circle x² + y² = r² -/
theorem line_tangent_to_circle (r : ℝ) (hr : r > 0) :
  ∀ θ : ℝ, ∃! (x y : ℝ), (Real.cos θ + y * Real.sin θ = r) ∧ (x^2 + y^2 = r^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l170_17078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_cai_running_stats_l170_17022

def running_distances : List Int := [460, 220, -250, -10, -330, 50, 560]
def standard_distance : Nat := 3000
def reward_threshold : Nat := 10000

theorem teacher_cai_running_stats :
  let max_distance := running_distances.maximum?
  let min_distance := running_distances.minimum?
  let total_distance := (running_distances.sum + 7 * standard_distance)
  (∀ m n, max_distance = some m → min_distance = some n → m - n = 890) ∧
  (total_distance = 21700) ∧
  (total_distance > reward_threshold) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_cai_running_stats_l170_17022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equality_solution_l170_17091

theorem sin_cos_equality_solution :
  ∃ x : ℝ, x = π / 12 ∧ Real.sin (2 * x) * Real.sin (4 * x) = Real.cos (2 * x) * Real.cos (4 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equality_solution_l170_17091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_solution_l170_17098

noncomputable def x₀ : ℝ := Real.exp 4 -- This is an approximation, not the exact solution

theorem max_integer_solution (h : ∀ x : ℝ, x₀ = x ↔ Real.log x = 8 - 2 * x) :
  (∃ n : ℤ, (n : ℝ) ≤ x₀ ∧ ∀ m : ℤ, (m : ℝ) ≤ x₀ → m ≤ n) ∧
  (∀ n : ℤ, (n : ℝ) ≤ x₀ → n ≤ 3) ∧
  (3 : ℝ) ≤ x₀ :=
by
  sorry

#check max_integer_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_solution_l170_17098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l170_17030

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + (1/2) * t, (Real.sqrt 3 / 2) * t)

def curve_C (x y : ℝ) : Prop := y^2 = 8 * x

def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    A = line_l t₁ ∧ 
    B = line_l t₂ ∧ 
    curve_C A.1 A.2 ∧ 
    curve_C B.1 B.2 ∧
    t₁ ≠ t₂

theorem chord_length (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l170_17030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buy_seven_items_l170_17088

/-- Calculates the total number of items purchased given the conditions --/
def totalItems (totalMoney hotChocolateCost sandwichCost : ℚ) : ℕ :=
  let sandwiches := (totalMoney / sandwichCost).floor.toNat
  let remainingMoney := totalMoney - sandwiches * sandwichCost
  let hotChocolates := (remainingMoney / hotChocolateCost).floor.toNat
  sandwiches + hotChocolates

/-- Theorem stating that under the given conditions, 7 items will be purchased --/
theorem buy_seven_items :
  totalItems 35 (3/2) 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_buy_seven_items_l170_17088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_prime_greater_than_3n_l170_17033

/-- p_n denotes the n-th prime number -/
def nth_prime : ℕ → ℕ := sorry

/-- Theorem: For all n ≥ 12, the n-th prime number is greater than 3n -/
theorem nth_prime_greater_than_3n (n : ℕ) (h : n ≥ 12) : nth_prime n > 3 * n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_prime_greater_than_3n_l170_17033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_theorem_l170_17067

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then x^2 - 4
  else if x > 2 then 2*x
  else 0  -- This else case is added to make the function total

theorem function_value_theorem (x₀ : ℝ) :
  f x₀ = 8 → x₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_theorem_l170_17067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expensive_candy_price_l170_17027

/-- Proves that given a mixture of candy with specific weights and prices, 
    the price of the more expensive candy is $3.00 per pound. -/
theorem expensive_candy_price 
  (total_weight : ℝ) 
  (cheap_weight : ℝ) 
  (cheap_price : ℝ) 
  (mixture_price : ℝ) 
  (h1 : total_weight = 80) 
  (h2 : cheap_weight = 64) 
  (h3 : cheap_price = 2) 
  (h4 : mixture_price = 2.2) : 
  (total_weight * mixture_price - cheap_weight * cheap_price) / (total_weight - cheap_weight) = 3 := by
  -- Placeholder for the proof
  sorry

#check expensive_candy_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expensive_candy_price_l170_17027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_l170_17047

theorem unique_number : ∃! n : ℕ, n > 0 ∧ (n : ℝ) + (n : ℝ)^2 = 252 ∧ 180 % n = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_l170_17047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l170_17061

-- Define the functions f, g, and h
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - m * Real.log x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x^2 - (m + 1) * x
noncomputable def h (m : ℝ) (x : ℝ) : ℝ := f m x - g m x

-- State the theorem
theorem unique_intersection (m : ℝ) (hm : m ≥ 1) :
  ∃! x : ℝ, x > 0 ∧ h m x = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l170_17061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkered_triangle_division_l170_17005

/-- Represents a part of the triangle -/
structure TrianglePart where
  numbers : List Nat
  area : Nat
deriving Inhabited

/-- Represents the checkered triangle -/
structure CheckeredTriangle where
  parts : List TrianglePart
  total_sum : Nat
deriving Inhabited

/-- Checks if all parts have the same sum -/
def all_parts_equal_sum (triangle : CheckeredTriangle) (target_sum : Nat) : Prop :=
  ∀ part ∈ triangle.parts, (part.numbers.sum = target_sum)

/-- Checks if all parts have different areas -/
def all_parts_different_areas (triangle : CheckeredTriangle) : Prop :=
  ∀ i j, i ≠ j → i < triangle.parts.length → j < triangle.parts.length →
    (triangle.parts[i]!).area ≠ (triangle.parts[j]!).area

/-- The main theorem -/
theorem checkered_triangle_division (triangle : CheckeredTriangle) 
    (h1 : triangle.total_sum = 63)
    (h2 : triangle.parts.length = 3) :
    ∃ (target_sum : Nat), 
      all_parts_equal_sum triangle target_sum ∧ 
      all_parts_different_areas triangle ∧
      target_sum = 21 := by
  sorry

#check checkered_triangle_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkered_triangle_division_l170_17005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_between_11_and_21_divisible_by_2_l170_17070

def is_between (a b x : ℕ) : Bool := a ≤ x ∧ x ≤ b

def divisible_by_two (n : ℕ) : Bool := n % 2 = 0

def numbers_in_range : List ℕ :=
  (List.range 11).map (λ i => i + 11) |>.filter (λ n => is_between 11 21 n && divisible_by_two n)

theorem average_of_numbers_between_11_and_21_divisible_by_2 :
  numbers_in_range.sum / numbers_in_range.length = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_between_11_and_21_divisible_by_2_l170_17070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l170_17053

def complex_i : ℂ := Complex.I

def z : ℂ := complex_i * (1 - complex_i)

theorem z_in_first_quadrant : z.re ≥ 0 ∧ z.im ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l170_17053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fourth_vertex_l170_17014

theorem square_fourth_vertex (z1 z2 z3 z4 : ℂ) : 
  z1 = (3 + I) / (1 - I) → 
  z2 = -2 + I → 
  z3 = 0 → 
  (Set.ncard {z1, z2, z3, z4} = 4) →
  (Complex.abs (z1 - z2) = Complex.abs (z2 - z3)) →
  (Complex.abs (z1 - z2) = Complex.abs (z3 - z4)) →
  (Complex.abs (z1 - z4) = Complex.abs (z2 - z3)) →
  z4 = -1 + 3*I := by
  sorry

#check square_fourth_vertex

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fourth_vertex_l170_17014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_triangle_side_b_triangle_angle_A_l170_17055

-- Part 1
theorem triangle_angle_B (a b c : ℝ) (h1 : a = 20) (h2 : b = 29) (h3 : c = 21) :
  let B := Real.arccos ((a^2 + c^2 - b^2) / (2*a*c))
  B = π/2 := by sorry

-- Part 2
theorem triangle_side_b (a b c : ℝ) (B : ℝ) 
  (h1 : a = 3 * Real.sqrt 3) (h2 : c = 2) (h3 : B = 5*π/6) :
  let b := Real.sqrt (a^2 + c^2 - 2*a*c*(Real.cos B))
  b = 7 := by sorry

-- Part 3
theorem triangle_angle_A (a b c : ℝ) 
  (h1 : a = 2) (h2 : b = Real.sqrt 2) (h3 : c = Real.sqrt 3 + 1) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))
  A = π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_triangle_side_b_triangle_angle_A_l170_17055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_danielles_apartment_rooms_l170_17083

-- Define the number of rooms in each apartment
def danielles_rooms : ℕ := sorry
def heidis_rooms : ℕ := sorry
def grants_rooms : ℕ := sorry

-- State the conditions
axiom heidi_danielle : heidis_rooms = 3 * danielles_rooms
axiom grant_heidi : grants_rooms * 9 = heidis_rooms
axiom grants_rooms_value : grants_rooms = 2

-- Theorem to prove
theorem danielles_apartment_rooms : danielles_rooms = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_danielles_apartment_rooms_l170_17083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_distance_calculation_l170_17015

/-- The actual distance calculation problem --/
theorem actual_distance_calculation
  (t : ℝ)  -- Temperature in °C
  (a : ℝ)  -- Original length of rod at 0°C in meters
  (n : ℝ)  -- Measured distance in meters
  (lambda : ℝ)  -- Linear expansion coefficient of copper
  (h1 : t = 26)
  (h2 : a = 3)
  (h3 : n = 7856.728)
  (h4 : lambda = 0.000017)
  : ∃ L : ℝ, abs (L - 7860.201) < 0.001 ∧ L = n * (1 + lambda * t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_distance_calculation_l170_17015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_equation_sum_l170_17066

noncomputable def floor (x : ℝ) := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - floor x

def is_not_square_divisible (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

theorem fractional_equation_sum :
  ∃ (a b c d : ℕ) (α₁ α₂ : ℝ),
    (∀ x : ℝ, x^2 + frac x = 21 ↔ x = α₁ ∨ x = α₂) ∧
    (α₁ + α₂ = (Real.sqrt a - Real.sqrt b) / c - d) ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    is_not_square_divisible a ∧
    is_not_square_divisible b ∧
    a + b + c + d = 169 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_equation_sum_l170_17066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l170_17094

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

theorem odd_function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1) 1 → f a b (-x) = -(f a b x)) →
  f a b (1/2) = 2/5 →
  (∃ g : ℝ → ℝ, (∀ x, x ∈ Set.Ioo (-1) 1 → g x = x / (1 + x^2)) ∧
    (∀ x, x ∈ Set.Ioo (-1) 1 → f a b x = g x) ∧
    (∀ x y, x ∈ Set.Ioo (-1) 1 → y ∈ Set.Ioo (-1) 1 → x < y → g x < g y) ∧
    (Set.Ioo 0 (1/2) = {t : ℝ | f a b (t-1) + f a b t < 0})) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l170_17094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l170_17018

theorem coefficient_x_cubed_in_expansion : 
  (Polynomial.coeff ((2 * Polynomial.X^2 + Polynomial.X - 1)^5) 3) = -30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l170_17018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_r_value_l170_17097

/-- The set A representing a circle centered at (0,0) with radius 2 -/
def A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- The set B representing a circle centered at (3,4) with radius r -/
def B (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

/-- Theorem stating that if A and B intersect at exactly one point, then r is either 3 or 7 -/
theorem intersection_implies_r_value (r : ℝ) (h1 : r > 0) 
  (h2 : ∃! p, p ∈ A ∩ B r) : r = 3 ∨ r = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_r_value_l170_17097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_expensive_journey_cost_l170_17019

/-- Represents a city in the travel problem -/
structure City where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two cities -/
noncomputable def distance (a b : City) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Calculates the cost of a bus trip -/
def busCost (dist : ℝ) : ℝ :=
  0.20 * dist

/-- Calculates the cost of an airplane trip -/
def planeCost (dist : ℝ) : ℝ :=
  120 + 0.12 * dist

/-- Calculates the minimum cost between bus and plane for a given distance -/
noncomputable def minCost (dist : ℝ) : ℝ :=
  min (busCost dist) (planeCost dist)

/-- The main theorem stating the least expensive total cost for the journey -/
theorem least_expensive_journey_cost
  (X Y Z : City)
  (right_triangle : distance X Y = Real.sqrt ((distance X Z)^2 + (distance Y Z)^2))
  (dist_XZ : distance X Z = 3600)
  (dist_XY : distance X Y = 3900) :
  minCost (distance X Y) + minCost (distance Y Z) + minCost (distance Z X) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_expensive_journey_cost_l170_17019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_length_l170_17041

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the line parametrically
def line_eq (t x y : ℝ) : Prop := x = 2*t - 1 ∧ y = t + 1

-- Define the length of the chord
noncomputable def chord_length : ℝ := 12 * Real.sqrt 5 / 5

-- Theorem statement
theorem chord_intercept_length :
  ∀ (x y t : ℝ), circle_eq x y → line_eq t x y → 
  ∃ (x1 y1 x2 y2 : ℝ), 
    circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ 
    line_eq t x1 y1 ∧ line_eq t x2 y2 ∧
    Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = chord_length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_length_l170_17041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_side_range_l170_17011

/-- In an obtuse triangle ABC with sides a, b, c where a < b < c, a = 2, and b = 3, 
    the range of possible values for c is (√13, 5). -/
theorem obtuse_triangle_side_range :
  ∀ (a b c : ℝ),
  a = 2 →
  b = 3 →
  a < b →
  b < c →
  (∃ (A B C : ℝ), 
    A + B + C = π ∧ 
    c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) ∧
    π/2 < C ∧ C < π) →
  Real.sqrt 13 < c ∧ c < 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_side_range_l170_17011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_reduction_temperature_drop_l170_17040

-- Define the relationship between distance and fuel consumption
noncomputable def fuel_consumption_rate : ℚ := 2 / 4

-- Define the relationship between distance and temperature change
noncomputable def temperature_change_rate : ℚ := 1 / 8

-- Define the distance traveled
def distance_traveled : ℚ := 16

-- Theorem for fuel reduction
theorem fuel_reduction :
  distance_traveled * fuel_consumption_rate = 8 := by
  -- Proof steps would go here
  sorry

-- Theorem for temperature drop
theorem temperature_drop :
  distance_traveled * temperature_change_rate = 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_reduction_temperature_drop_l170_17040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l170_17003

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r, ∀ m, a (m + 1) = r * a m

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (Real.log (a 3 * a 8 * a 13) / Real.log 10 = 6) →
  a 1 * a 15 = 10^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l170_17003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_l170_17054

noncomputable section

def circle_radius : ℝ := 3
def triangle_base : ℝ := 8
def triangle_side : ℝ := 5

def circle_area : ℝ := Real.pi * circle_radius^2
def triangle_height : ℝ := Real.sqrt (triangle_side^2 - (triangle_base/2)^2)
def triangle_area : ℝ := (1/2) * triangle_base * triangle_height

theorem area_difference : 
  circle_area - triangle_area = 9 * Real.pi - 12 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_l170_17054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l170_17049

/-- The minimum distance between a point and a line in 2D space. -/
noncomputable def min_distance_point_line (px py x1 y1 x2 y2 : ℝ) : ℝ :=
  let a := y2 - y1
  let b := x1 - x2
  let c := x2 * y1 - x1 * y2
  abs (a * px + b * py + c) / Real.sqrt (a^2 + b^2)

/-- Theorem stating the minimum distance between point M and line AB. -/
theorem min_distance_theorem :
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (1, 0)
  let M : ℝ × ℝ := (0, -3)
  let line_eq (x : ℝ) : ℝ := -x + 1
  min_distance_point_line M.1 M.2 A.1 A.2 B.1 B.2 = 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l170_17049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l170_17050

noncomputable def f (x a : ℝ) : ℝ := Real.sin x - 2 * x - a

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 Real.pi, f x a ≤ -1) ∧
  (∃ x ∈ Set.Icc 0 Real.pi, f x a = -1) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l170_17050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_denominator_of_sum_l170_17001

noncomputable def smallest_denominator_sum (x y : ℚ) : ℕ :=
  (x + y).den

theorem smallest_denominator_of_sum
  (x y : ℚ)
  (hx : x.den = 60)
  (hy : y.den = 70)
  (hx_reduced : Int.gcd x.num x.den = 1)
  (hy_reduced : Int.gcd y.num y.den = 1) :
  ∃ (a b : ℤ), x = a / 60 ∧ y = b / 70 ∧ smallest_denominator_sum x y = 84 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_denominator_of_sum_l170_17001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l170_17057

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x + x^2 else Real.exp (-x) + x^2

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (-a) + f a ≤ 2 * f 1) → (-1 ≤ a ∧ a ≤ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l170_17057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellen_lego_count_l170_17075

/-- Calculates the final number of Legos after a competition --/
def finalLegos (initial : ℕ) (rankBonus : ℕ) (percentageBonus : ℚ) : ℕ :=
  initial + rankBonus + (percentageBonus * initial).floor.toNat

/-- Theorem stating that given the initial conditions, Ellen ends up with 2190 Legos --/
theorem ellen_lego_count :
  finalLegos 2080 17 (4.5 / 100) = 2190 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellen_lego_count_l170_17075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_equal_diagonals_rhombus_not_necessarily_equal_diagonals_l170_17073

-- Define a quadrilateral
structure Quadrilateral where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ

-- Define a rectangle
structure Rectangle extends Quadrilateral where
  right_angles : ∀ i, angles i = 90
  opposite_sides_equal : sides 0 = sides 2 ∧ sides 1 = sides 3

-- Define a rhombus
structure Rhombus extends Quadrilateral where
  all_sides_equal : ∀ i j, sides i = sides j

-- Define a function to calculate the length of a diagonal
noncomputable def diagonal_length (q : Quadrilateral) : ℝ := sorry

-- Theorem: Rectangles have equal diagonals
theorem rectangle_equal_diagonals (r : Rectangle) :
  diagonal_length r.toQuadrilateral = diagonal_length r.toQuadrilateral := by sorry

-- Theorem: Rhombuses do not necessarily have equal diagonals
theorem rhombus_not_necessarily_equal_diagonals :
  ∃ (rh : Rhombus), diagonal_length rh.toQuadrilateral ≠ diagonal_length rh.toQuadrilateral := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_equal_diagonals_rhombus_not_necessarily_equal_diagonals_l170_17073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_t_range_sufficient_condition_a_range_l170_17006

-- Define the propositions
def P (t : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (4 - t) + y^2 / (t - 1) = 1 ∧ 
  (4 - t > 0 ∧ t - 1 > 0)

def q (t a : ℝ) : Prop := t^2 - (a + 3) * t + (a + 2) < 0

-- Theorem 1
theorem ellipse_t_range (t : ℝ) : P t → 1 < t ∧ t < 5/2 := by
  sorry

-- Theorem 2
theorem sufficient_condition_a_range (a : ℝ) : 
  (∀ t : ℝ, P t → q t a) ∧ (∃ t : ℝ, q t a ∧ ¬P t) → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_t_range_sufficient_condition_a_range_l170_17006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l170_17029

noncomputable section

-- Define the curve C
def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define points A and B in polar coordinates
def point_A : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)
def point_B : ℝ × ℝ := (3, Real.pi / 2)

-- Define the line l passing through A and B
def line_l (t : ℝ) : ℝ × ℝ := sorry

-- Define the intersection points P and Q
def point_P : ℝ × ℝ := sorry
def point_Q : ℝ × ℝ := sorry

-- Statement to prove
theorem intersection_distance_product :
  let BP := Real.sqrt ((point_P.1 - point_B.1)^2 + (point_P.2 - point_B.2)^2)
  let BQ := Real.sqrt ((point_Q.1 - point_B.1)^2 + (point_Q.2 - point_B.2)^2)
  BP * BQ = 120 / 19 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l170_17029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l170_17004

-- Define an even function f on the real numbers
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 + 2*x else x^2 - 2*x

-- State the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_def_neg : ∀ x : ℝ, x ≤ 0 → f x = x^2 + 2*x

-- Define the solution set
def solution_set : Set ℝ := {x | -5 < x ∧ x < 1}

-- State the theorem
theorem f_inequality_solution : 
  {x : ℝ | f (x + 2) < 3} = solution_set :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l170_17004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_x_coordinate_of_parabola_points_l170_17059

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem midpoint_x_coordinate_of_parabola_points 
  (M N : ℝ × ℝ) 
  (hM : parabola M) 
  (hN : parabola N) 
  (h_dist : distance M focus + distance N focus = 6) : 
  (M.1 + N.1) / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_x_coordinate_of_parabola_points_l170_17059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_minus_cot_period_l170_17045

open Real

/-- The function f(x) = tan x - cot x has a period of π. -/
theorem tan_minus_cot_period : ∀ x : ℝ, tan x - (1 / tan x) = tan (x + π) - (1 / tan (x + π)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_minus_cot_period_l170_17045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_l170_17095

/-- The degree of a polynomial x^m + y^n + 4^(m+n) where m and n are natural numbers -/
def polyDegree (m n : ℕ) : ℕ :=
  max m n

theorem degree_of_polynomial (m n : ℕ) :
  polyDegree m n = max m n := by
  rfl  -- reflexivity is enough here as the definition matches exactly


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_l170_17095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_20_deg_less_than_7_over_20_l170_17099

theorem sin_20_deg_less_than_7_over_20 :
  Real.sin (π / 9) < 7 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_20_deg_less_than_7_over_20_l170_17099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l170_17048

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem angle_between_vectors (a b : V) 
  (ha : ‖a‖ = 5)
  (hb : ‖b‖ = 7)
  (hab : ‖a + b‖ = 10) :
  inner a b / (‖a‖ * ‖b‖) = 13 / 35 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l170_17048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_5_value_l170_17000

def S (n : ℕ) : ℕ := n^2 + 1

def a : ℕ → ℕ
  | 0 => S 0 -- Add case for 0
  | 1 => S 1
  | n + 1 => S (n + 1) - S n

def b (n : ℕ) : ℚ := 2 / (a n + 1)

theorem b_5_value : b 5 = 1/5 := by
  -- Expand the definitions
  unfold b
  unfold a
  -- Simplify the expression
  simp [S]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_5_value_l170_17000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_m_zero_domain_range_m_two_answer_is_BC_l170_17031

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 - 2*x + m)

-- Define the domain P and range Q as sets
def P (m : ℝ) : Set ℝ := {x | ∃ y, f m x = y}
def Q (m : ℝ) : Set ℝ := {y | ∃ x, f m x = y}

-- Theorem for m = 0
theorem domain_range_m_zero :
  P 0 = {x | x ≥ 2 ∨ x ≤ 0} ∧ Q 0 = Set.Ici 0 := by
  sorry

-- Theorem for m = 2
theorem domain_range_m_two :
  P 2 = Set.univ ∧ Q 2 = Set.Ici 1 := by
  sorry

-- Theorem stating that the answer is BC
theorem answer_is_BC :
  (P 0 ≠ Set.univ) ∧
  (P 2 = Set.univ) ∧
  (Q 0 = Set.Ici 0) ∧
  (Q 2 ≠ Set.Ici 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_m_zero_domain_range_m_two_answer_is_BC_l170_17031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l170_17025

/-- Given vectors a, b, and c in ℝ³, prove that a is parallel to c and perpendicular to b -/
theorem vector_relationships (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (-2, -3, 1)) 
  (hb : b = (2, 0, 4)) 
  (hc : c = (-4, -6, 2)) : 
  (∃ (k : ℝ), c = k • a) ∧ (a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0) := by
  sorry

#check vector_relationships

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l170_17025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_46382_between_consecutive_integers_sum_c_d_equals_nine_l170_17020

theorem log_46382_between_consecutive_integers :
  ∃ (c d : ℕ), c + 1 = d ∧ 
  (c : ℝ) < Real.log 46382 / Real.log 10 ∧ Real.log 46382 / Real.log 10 < (d : ℝ) ∧
  c = 4 ∧ d = 5 := by
  sorry

-- Additional definitions to match the problem conditions
def n : ℕ := 46382

axiom n_bounds : 10000 < n ∧ n < 100000

axiom log_10000 : Real.log 10000 / Real.log 10 = 4

axiom log_100000 : Real.log 100000 / Real.log 10 = 5

-- Theorem to prove c + d = 9
theorem sum_c_d_equals_nine (c d : ℕ) 
  (h : c + 1 = d ∧ (c : ℝ) < Real.log n / Real.log 10 ∧ Real.log n / Real.log 10 < (d : ℝ) ∧ c = 4 ∧ d = 5) :
  c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_46382_between_consecutive_integers_sum_c_d_equals_nine_l170_17020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l170_17079

-- Define the function f(x) = √(x^2 + 2x - 3)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 2*x - 3)

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | x^2 + 2*x - 3 ≥ 0}

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x y, x ∈ domain → y ∈ domain → (f x ≥ f y ↔ x ≤ y) ↔ x ∈ Set.Iic (-3 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l170_17079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_minimum_l170_17071

theorem quadratic_inequality_minimum (a b c : ℝ) (h1 : a < b) 
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  ∃ (m : ℝ), (∀ a' b' c' : ℝ, a' < b' → 
    (∀ x : ℝ, a' * x^2 + b' * x + c' ≥ 0) → 
    (a' + 2*b' + 4*c') / (b' - a') ≥ m) ∧ 
  (∃ a' b' c' : ℝ, a' < b' ∧ 
    (∀ x : ℝ, a' * x^2 + b' * x + c' ≥ 0) ∧ 
    (a' + 2*b' + 4*c') / (b' - a') = m) ∧ 
  m = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_minimum_l170_17071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ray_l170_17037

-- Define the points F₁ and F₂ in the 2D plane
def F₁ : ℝ × ℝ := (-8, 3)
def F₂ : ℝ × ℝ := (2, 3)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points P satisfying the condition
def locus : Set (ℝ × ℝ) :=
  {P | distance P F₁ - distance P F₂ = 10}

-- Theorem statement
theorem locus_is_ray : 
  ∃ (A : ℝ × ℝ) (v : ℝ × ℝ), locus = {P | ∃ t : ℝ, t ≥ 0 ∧ P = A + t • v} := by
  sorry

#check locus_is_ray

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ray_l170_17037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_approx_62_percent_l170_17081

/-- Represents the population of a town over three decades -/
structure TownPopulation where
  pop1991 : ℕ
  pop2001 : ℕ
  pop2011 : ℕ

/-- The conditions of the problem -/
def SatisfiesConditions (t : TownPopulation) : Prop :=
  ∃ (p q r : ℕ),
    t.pop1991 = p * p ∧
    t.pop2001 = t.pop1991 + 150 ∧
    t.pop2001 = q * q + 9 ∧
    t.pop2011 = t.pop2001 + 150 ∧
    t.pop2011 = r * r

/-- The percent growth over 20 years -/
noncomputable def PercentGrowth (t : TownPopulation) : ℝ :=
  (t.pop2011 - t.pop1991 : ℝ) / t.pop1991 * 100

/-- The main theorem -/
theorem population_growth_approx_62_percent (t : TownPopulation) 
  (h : SatisfiesConditions t) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |PercentGrowth t - 62| < ε := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_approx_62_percent_l170_17081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_relation_l170_17074

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Point on parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

/-- Line through two points -/
noncomputable def line_slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

/-- Theorem statement -/
theorem slope_relation (P M N : PointOnParabola) 
  (hM : M ≠ P) (hN : N ≠ P) (hMN : M ≠ N) :
  let k1 := line_slope P.x P.y M.x M.y
  let k2 := line_slope P.x P.y N.x N.y
  let k3 := line_slope M.x M.y N.x N.y
  1 / k1 + 1 / k2 - 1 / k3 = P.y / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_relation_l170_17074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l170_17076

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ := a / x + log x

def F (x : ℝ) : ℝ := 2 / x + log x + 1 + log x / x

theorem min_value_and_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (exp 1), f a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 1 (exp 1), f a x = 3/2) →
  a = sqrt (exp 1) ∧
  ∀ x > 1, F x / (2 * exp (x - 1)) > (exp 1 + 1) / (x * exp x + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l170_17076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calories_in_250g_of_beverage_l170_17012

/-- Represents the beverage mixture --/
structure Beverage where
  orange_juice : ℝ
  honey : ℝ
  water : ℝ

/-- Calculates the total weight of the beverage --/
noncomputable def total_weight (b : Beverage) : ℝ :=
  b.orange_juice + b.honey + b.water

/-- Calculates the total calories in the beverage --/
noncomputable def total_calories (b : Beverage) : ℝ :=
  (b.orange_juice * 45 / 100) + (b.honey * 304 / 100)

/-- Calculates the calories in a given weight of the beverage --/
noncomputable def calories_in_weight (b : Beverage) (weight : ℝ) : ℝ :=
  (total_calories b / total_weight b) * weight

/-- The main theorem to prove --/
theorem calories_in_250g_of_beverage :
  let b : Beverage := { orange_juice := 150, honey := 50, water := 300 }
  ∃ ε > 0, |calories_in_weight b 250 - 110| < ε := by
  sorry

#eval "Proof completed."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calories_in_250g_of_beverage_l170_17012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l170_17026

/-- Given three points D, E, and F in a plane, prove that a unique triangle ABC exists
    where D is the midpoint of AB, E is 1/4 of the way from B to C,
    and F is 1/8 of the way from C to A, unless these points are collinear
    or satisfy specific exceptional conditions. -/
theorem triangle_existence
  (D E F : EuclideanSpace ℝ (Fin 2))
  (h_not_collinear : ¬ Collinear ℝ ({D, E, F} : Set (EuclideanSpace ℝ (Fin 2))))
  : ∃! (A B C : EuclideanSpace ℝ (Fin 2)),
    D = (1 : ℝ) • A + (1 : ℝ) • B ∧
    E = (3 : ℝ) • B + (1 : ℝ) • C ∧
    F = (7 : ℝ) • C + (1 : ℝ) • A :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l170_17026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_number_condition_l170_17060

/-- A function that checks if a number satisfies the given condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  let x := n / 100
  let y := (n / 10) % 10
  let z := n % 10
  (y * z = 10 * x + y) ∧ (100 ≤ n) ∧ (n < 1000)

/-- The theorem stating the equivalence between satisfying the condition and being one of the specified numbers -/
theorem three_digit_number_condition (n : ℕ) :
  satisfiesCondition n ↔ n ∈ ({153, 255, 357, 459, 126, 246, 366, 486} : Finset ℕ) := by
  sorry

#check three_digit_number_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_number_condition_l170_17060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_revolution_volume_maximized_at_60_degrees_l170_17058

noncomputable section

open Real

/-- The volume of a solid of revolution formed by rotating an equilateral triangle -/
def volume_of_revolution (a : ℝ) (α : ℝ) : ℝ :=
  (π / 3) * a^3 * (sin α + (Real.sqrt 3 / 2) * cos α + (1 / 2) * sin α) *
  (cos α * ((Real.sqrt 3 / 2) * cos α + (1 / 2) * sin α) + ((1 / 2) * cos α - (Real.sqrt 3 / 2) * sin α) * sin α)

/-- The theorem stating the maximum volume and the angle at which it occurs -/
theorem max_volume_revolution (a : ℝ) (h : a > 0) :
  (∃ (α : ℝ), ∀ (β : ℝ), volume_of_revolution a α ≥ volume_of_revolution a β) ∧
  (volume_of_revolution a (π / 3) = π * a^3 / 2) := by
  sorry

/-- The theorem stating that the volume is maximized at α = 60° -/
theorem volume_maximized_at_60_degrees (a : ℝ) (h : a > 0) :
  ∀ (α : ℝ), volume_of_revolution a (π / 3) ≥ volume_of_revolution a α := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_revolution_volume_maximized_at_60_degrees_l170_17058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_b_l170_17065

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a^2 + |c-10| + √(b-8) = 12a - 36, then sin B = 4/5 -/
theorem triangle_sin_b (a b c : ℝ) (h : a^2 + |c - 10| + Real.sqrt (b - 8) = 12 * a - 36) :
  Real.sin (Real.arcsin (b / c)) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_b_l170_17065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l170_17024

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 1 2 then x + 1
  else if x ∈ Set.Ioc 2 4 then 2 * (x - 3)^2 + 1
  else 0  -- This case should never occur given the domain

-- State the theorem
theorem f_properties :
  (∀ x ∈ Set.Icc 1 4, f x = f x) ∧  -- f is well-defined on [1,4]
  (f 1 = 2) ∧
  (f 2 = 3) ∧
  (∀ x ∈ Set.Icc 1 2, f x = x + 1) ∧  -- Linear on [1,2]
  (∀ x ∈ Set.Ioc 2 4, ∃ a b c : ℝ, f x = a * x^2 + b * x + c) ∧  -- Quadratic on (2,4]
  (f 3 = 1) ∧  -- Vertex at (3,1)
  Continuous f ∧  -- f is continuous
  Set.range f = Set.Icc 1 3  -- Range is [1,3]
  := by
    sorry  -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l170_17024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_dot_product_l170_17007

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation x²=8y -/
def Parabola : Set Point := {p : Point | p.x^2 = 8 * p.y}

/-- Represents a line passing through a fixed point T(0, t) -/
def Line (t : ℝ) : Set Point := {p : Point | ∃ (k : ℝ), p.y = k * p.x + t}

/-- The dot product of two vectors represented by points -/
def dotProduct (p q : Point) : ℝ := p.x * q.x + p.y * q.y

theorem parabola_intersection_dot_product (t : ℝ) (h : t > 0) :
  ∀ A B : Point,
    A ∈ Parabola → B ∈ Parabola → A ∈ Line t → B ∈ Line t →
    dotProduct A B = t^2 - 8*t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_dot_product_l170_17007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_scalene_triangle_similar_to_IHO_l170_17002

-- Define a structure for a triangle with special points
structure TriangleWithSpecialPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  I : ℝ × ℝ
  H : ℝ × ℝ
  O : ℝ × ℝ

-- Define what it means for a triangle to be scalene
def isScalene (t : TriangleWithSpecialPoints) : Prop :=
  let ab := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let bc := Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)
  let ca := Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)
  ab ≠ bc ∧ bc ≠ ca ∧ ca ≠ ab

-- Define similarity between two triangles
def areSimilar (t1 t2 : TriangleWithSpecialPoints) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    (((t1.B.1 - t1.A.1)^2 + (t1.B.2 - t1.A.2)^2) / ((t2.B.1 - t2.A.1)^2 + (t2.B.2 - t2.A.2)^2) = k^2) ∧
    (((t1.C.1 - t1.B.1)^2 + (t1.C.2 - t1.B.2)^2) / ((t2.C.1 - t2.B.1)^2 + (t2.C.2 - t2.B.2)^2) = k^2) ∧
    (((t1.A.1 - t1.C.1)^2 + (t1.A.2 - t1.C.2)^2) / ((t2.A.1 - t2.C.1)^2 + (t2.A.2 - t2.C.2)^2) = k^2)

-- The main theorem
theorem no_scalene_triangle_similar_to_IHO (t : TriangleWithSpecialPoints) :
  isScalene t → ¬(areSimilar t { A := t.A, B := t.B, C := t.C, I := t.I, H := t.H, O := t.O }) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_scalene_triangle_similar_to_IHO_l170_17002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l170_17051

theorem complex_power_sum (θ : Real) (a_2016 b_2016 : Real) :
  θ = π / 2 →
  (Complex.exp (θ * Complex.I)) ^ 2016 = Complex.ofReal a_2016 + Complex.I * Complex.ofReal b_2016 →
  a_2016 + b_2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l170_17051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_seventh_pi_l170_17068

theorem cosine_sum_seventh_pi : 
  Real.cos (π / 7) - Real.cos (2 * π / 7) + Real.cos (3 * π / 7) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_seventh_pi_l170_17068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_greater_than_four_l170_17042

-- Define a standard die
def standard_die : Finset ℕ := Finset.range 6

-- Define the event of rolling a number greater than 4
def greater_than_four (n : ℕ) : Prop := n > 4 ∧ n ≤ 6

-- Provide an instance for DecidablePred
instance : DecidablePred greater_than_four :=
  fun n => by
    simp [greater_than_four]
    exact And.decidable

-- Theorem statement
theorem probability_greater_than_four :
  (Finset.filter greater_than_four standard_die).card / standard_die.card = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_greater_than_four_l170_17042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_condition_l170_17034

theorem inequality_and_condition (k l m : ℝ) (x y z : ℝ) (a b : ℝ) :
  (k + l + m)^2 ≥ 3 * (k * l + l * m + m * k) ∧
  (a * (x + y + z) = b * (x * y + y * z + z * x) ∧ 
   b * (x * y + y * z + z * x) = x * y * z) ∧
  x > 0 ∧ y > 0 ∧ z > 0 →
  a ≥ 3 * b^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_condition_l170_17034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l170_17043

-- Define the ellipse
noncomputable def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/16 = 1

-- Define the focus G
noncomputable def focus : ℝ × ℝ := (2 * Real.sqrt 5, 0)

-- Define a point on the ellipse
def point_on_ellipse (p : ℝ × ℝ) : Prop := 
  is_on_ellipse p.1 p.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem ellipse_chord_theorem (C D : ℝ × ℝ) : 
  point_on_ellipse C → point_on_ellipse D → 
  distance C focus = 2 → 
  distance D focus = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l170_17043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_l170_17063

/-- The cosine of the dihedral angle formed by two planes containing two inscribed spheres -/
theorem dihedral_angle_cosine (R : ℝ) (h : R > 0) : 
  let r : ℝ := R / 2
  let centers_distance : ℝ := R + r
  let cos_30 : ℝ := 2 / 3
  cos_30 * centers_distance / R = 1
  →
  (1 : ℝ) / 9 = Real.cos (Real.arccos (1 / (centers_distance / R))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_l170_17063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l170_17028

/-- Calculates the length of a train given its speed and time to cross a pole. -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time * (1000 / 3600)

/-- Theorem: A train with a speed of 60 km/hr that crosses a pole in 15 seconds has a length of approximately 250 meters. -/
theorem train_length_approx :
  ∃ (length : ℝ), abs (trainLength 60 15 - length) < 1 ∧ abs (length - 250) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l170_17028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l170_17077

theorem circular_table_seating (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 8) :
  (n * Nat.factorial (k - 1)) = 45360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l170_17077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_intercept_l170_17052

/-- Given a regression line with slope 1.23 passing through (4, 5), 
    prove that its y-intercept is 0.08 -/
theorem regression_line_intercept 
  (slope : ℝ) 
  (center_x center_y : ℝ) 
  (h1 : slope = 1.23)
  (h2 : center_x = 4)
  (h3 : center_y = 5) : 
  center_y = slope * center_x + 0.08 := by
  sorry

#check regression_line_intercept

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_intercept_l170_17052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_applicants_count_l170_17016

def LawSchoolApplicants : Type := Unit

-- Define predicates as boolean functions
def political_science (a : LawSchoolApplicants) : Bool := sorry
def high_gpa (a : LawSchoolApplicants) : Bool := sorry

axiom total_applicants : Finset LawSchoolApplicants

axiom ps_count : (total_applicants.filter (fun a => political_science a = true)).card = 15
axiom high_gpa_count : (total_applicants.filter (fun a => high_gpa a = true)).card = 20
axiom not_ps_low_gpa_count : (total_applicants.filter (fun a => political_science a = false ∧ high_gpa a = false)).card = 10
axiom ps_and_high_gpa_count : (total_applicants.filter (fun a => political_science a = true ∧ high_gpa a = true)).card = 5

theorem total_applicants_count : total_applicants.card = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_applicants_count_l170_17016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_floor_sum_l170_17035

open Real

theorem min_value_trig_floor_sum :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (π / 2) ∧
  (⌊(sin x + cos x) / tan x⌋ : ℤ) + 
  (⌊(cos x + tan x) / sin x⌋ : ℤ) + 
  (⌊(tan x + sin x) / cos x⌋ : ℤ) = 3 ∧
  ∀ (y : ℝ), y ∈ Set.Ioo 0 (π / 2) →
  (⌊(sin y + cos y) / tan y⌋ : ℤ) + 
  (⌊(cos y + tan y) / sin y⌋ : ℤ) + 
  (⌊(tan y + sin y) / cos y⌋ : ℤ) ≥ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_floor_sum_l170_17035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l170_17086

noncomputable def f (x : ℝ) : ℝ := Real.sin x

noncomputable def point : ℝ × ℝ := (Real.pi / 3, Real.sqrt 3 / 2)

noncomputable def tangent_line (x : ℝ) : ℝ := (1 / 2) * x - Real.pi / 6 + Real.sqrt 3 / 2

theorem tangent_line_is_correct :
  -- The proposed line passes through the point of tangency
  tangent_line (point.1) = point.2 ∧
  -- The slope of the proposed line equals the derivative of f at the point
  (1 / 2) = Real.cos (point.1) ∧
  -- The proposed line is indeed tangent to the curve
  ∀ x : ℝ, x ≠ point.1 → |f x - tangent_line x| ≤ |f x - (f point.1 + Real.cos point.1 * (x - point.1))| :=
by
  sorry

#check tangent_line_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l170_17086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_max_value_and_occurrence_a_is_arithmetic_sequence_a_common_difference_l170_17093

-- Define the sequence S_n
def S (n : ℕ) : ℤ := 3 * n - n^2

-- Define the sequence a_n
def a : ℕ → ℤ
| 0 => S 1  -- We start from 0 to align with Lean's ℕ type
| n + 1 => S (n + 2) - S (n + 1)

theorem S_max_value_and_occurrence :
  (∀ n : ℕ, S (n + 1) ≤ 2) ∧
  (S 1 = 2 ∧ S 2 = 2) ∧
  (∀ n : ℕ, S (n + 1) = 2 → (n = 0 ∨ n = 1)) :=
sorry

theorem a_is_arithmetic_sequence :
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d :=
sorry

theorem a_common_difference :
  ∀ n : ℕ, a (n + 1) - a n = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_max_value_and_occurrence_a_is_arithmetic_sequence_a_common_difference_l170_17093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_distance_range_l170_17082

noncomputable section

def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

def inscribed_triangle (A B C : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ellipse C.1 C.2

def on_positive_x_axis (A : ℝ × ℝ) : Prop := A.1 > 0 ∧ A.2 = 0

def slope_product (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) / (B.1 - A.1) * (C.2 - A.2) / (C.1 - A.1) = -1/4

def centroid (G A B C : ℝ × ℝ) : Prop :=
  G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3

def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem triangle_centroid_distance_range (A B C G : ℝ × ℝ) :
  inscribed_triangle A B C →
  on_positive_x_axis A →
  slope_product A B C →
  centroid G A B C →
  (2 * Real.sqrt 13 + 4) / 3 ≤ distance G A + distance G B + distance G C ∧
  distance G A + distance G B + distance G C < 16 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_distance_range_l170_17082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l170_17096

theorem min_k_value (m n a k : ℕ) (h1 : k > 1) (h2 : 5^m + 63*n + 49 = a^k) :
  k ≥ 5 ∧ ∃ (m' n' a' : ℕ), m' > 0 ∧ n' > 0 ∧ a' > 0 ∧ 5^m' + 63*n' + 49 = a'^5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l170_17096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_circumference_l170_17087

/-- Given two circles, prove that the circumference of the smaller circle is approximately 263.8934 meters -/
theorem smaller_circle_circumference 
  (c₁ c₂ : ℝ) -- circumferences of the two circles
  (a₁ a₂ : ℝ) -- areas of the two circles
  (h₁ : c₂ = 352) -- circumference of the larger circle is 352 meters
  (h₂ : a₂ - a₁ = 4313.735577562732) -- difference between areas
  (h₃ : c₁ ≤ c₂) -- c₁ is the circumference of the smaller circle
  : ∃ ε > 0, |c₁ - 263.8934| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_circumference_l170_17087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_albert_family_ages_l170_17008

/-- Albert's family ages problem -/
theorem albert_family_ages 
  (father_age_at_albert_birth : ℕ)
  (mother_age_at_brother_birth : ℕ)
  (brother_age_diff : ℕ)
  (mother_age_at_sister_birth : ℕ)
  (eldest_age_diff : ℕ)
  (father_age_at_eldest_birth : ℕ)
  (albert_current_age : ℕ)
  (h1 : father_age_at_albert_birth = 48)
  (h2 : mother_age_at_brother_birth = 46)
  (h3 : brother_age_diff = 2)
  (h4 : mother_age_at_sister_birth = 50)
  (h5 : eldest_age_diff = 4)
  (h6 : father_age_at_eldest_birth = 40)
  (h7 : albert_current_age = 18) :
  (father_age_at_albert_birth - (mother_age_at_brother_birth - brother_age_diff) = 4) ∧ 
  (mother_age_at_sister_birth - mother_age_at_brother_birth = 4) ∧
  (father_age_at_albert_birth + albert_current_age = 66) ∧
  ((mother_age_at_brother_birth - brother_age_diff) + albert_current_age = 62) ∧
  (albert_current_age + eldest_age_diff = 22) ∧
  (albert_current_age - brother_age_diff = 16) ∧
  (albert_current_age - (mother_age_at_sister_birth - (mother_age_at_brother_birth - brother_age_diff)) = 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_albert_family_ages_l170_17008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_zero_implies_cos_4_sum_zero_l170_17038

theorem cos_sin_sum_zero_implies_cos_4_sum_zero
  (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 0)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) :
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_zero_implies_cos_4_sum_zero_l170_17038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_415_is_1_l170_17036

/-- The decimal expansion of 18/47 -/
def decimal_expansion : ℚ → List ℕ := sorry

/-- The repeating sequence in the decimal expansion of 18/47 -/
def repeating_sequence : List ℕ := sorry

/-- The length of the repeating sequence -/
def sequence_length : ℕ := 94

/-- The 415th digit in the decimal expansion of 18/47 -/
def digit_415 : ℕ := sorry

theorem digit_415_is_1 : 
  decimal_expansion (18/47) ≠ [] → 
  sequence_length = 94 → 
  digit_415 = (repeating_sequence.get? (415 % sequence_length)).getD 0 → 
  digit_415 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_415_is_1_l170_17036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_validity_l170_17044

/-- The slope of the line -/
def m : ℚ := 7/4

/-- The y-intercept of the line -/
def b : ℚ := -19/4

/-- The line equation -/
def line_eq (x y : ℚ) : Prop := y = m * x + b

/-- A point lies on the line -/
def point_on_line (p : ℚ × ℚ) : Prop := line_eq p.1 p.2

/-- A vector is parallel to the line -/
def vector_parallel (v : ℚ × ℚ) : Prop := ∃ (k : ℚ), v = (k, k * m)

/-- A parameterization is valid -/
def valid_param (p₀ v : ℚ × ℚ) : Prop := point_on_line p₀ ∧ vector_parallel v

/-- Parameterization A -/
def param_A : (ℚ × ℚ) × (ℚ × ℚ) := ((5, 2), (-4, -7))

/-- Parameterization B -/
def param_B : (ℚ × ℚ) × (ℚ × ℚ) := ((19, 7), (8, 14))

/-- Parameterization C -/
def param_C : (ℚ × ℚ) × (ℚ × ℚ) := ((3, -1/4), (4/7, 1))

/-- Parameterization D -/
def param_D : (ℚ × ℚ) × (ℚ × ℚ) := ((19/7, 0), (1, 7/4))

/-- Parameterization E -/
def param_E : (ℚ × ℚ) × (ℚ × ℚ) := ((0, -19/4), (28, -49))

theorem parameterization_validity :
  valid_param param_A.1 param_A.2 ∧
  valid_param param_B.1 param_B.2 ∧
  ¬valid_param param_C.1 param_C.2 ∧
  ¬valid_param param_D.1 param_D.2 ∧
  ¬valid_param param_E.1 param_E.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_validity_l170_17044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_trajectory_max_area_l170_17089

-- Define the circle C₁
noncomputable def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the line l₁
noncomputable def l₁ (x y : ℝ) : Prop := x - 2*y + 3*Real.sqrt 5 = 0

-- Define the curve C (trajectory of point N)
noncomputable def C (x y : ℝ) : Prop := x^2/9 + y^2/3 = 1

-- Define the line l
noncomputable def l (x y m : ℝ) : Prop := 2*x + y + m = 0

-- Define the area of triangle OBD
noncomputable def area_OBD (m : ℝ) : ℝ := (Real.sqrt (3*m^2*(39 - m^2)))/13

theorem circle_tangent_line_trajectory_max_area :
  -- C₁ is tangent to l₁
  (∃ x y : ℝ, C₁ x y ∧ l₁ x y) →
  -- The equation of curve C is correct
  (∀ x y : ℝ, (∃ x₀ y₀ : ℝ, C₁ x₀ y₀ ∧ x = x₀ ∧ y = y₀/Real.sqrt 3) ↔ C x y) →
  -- The maximum area of triangle OBD
  (∃ m : ℝ, ∀ m' : ℝ, area_OBD m' ≤ area_OBD m ∧ area_OBD m = 3*Real.sqrt 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_trajectory_max_area_l170_17089
