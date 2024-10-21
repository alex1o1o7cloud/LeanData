import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_pond_problem_l151_15137

theorem duck_pond_problem (small_pond : ℕ) (large_pond : ℕ) 
  (green_small : ℚ) (green_large : ℚ) (total_green : ℚ) :
  small_pond = 20 ∧ 
  green_small = 1/5 ∧ 
  green_large = 3/20 ∧ 
  total_green = 4/25 →
  (green_small * small_pond + green_large * large_pond) / (small_pond + large_pond) = total_green →
  large_pond = 80 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_duck_pond_problem_l151_15137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_natural_and_increasing_l151_15121

/-- A sequence where odd terms (except the first) are arithmetic means and even terms are geometric means of their neighbors -/
def special_sequence (a : ℕ) (q : ℚ) : ℕ → ℚ
  | 0 => a  -- Add this case to handle zero
  | 1 => a
  | 2 => a * q
  | 3 => a * q^2
  | n + 4 =>  -- Change n + 3 to n + 4 to avoid overlap with previous cases
    if (n + 4) % 2 = 0 then
      2 * special_sequence a q (n + 3) - special_sequence a q (n + 2)
    else
      (special_sequence a q (n + 3))^2 / (special_sequence a q (n + 2))

/-- Theorem stating that the special sequence consists of natural numbers and is increasing -/
theorem special_sequence_natural_and_increasing (a : ℕ) (q : ℚ) (h : q > 1) :
  (∀ n : ℕ, ∃ m : ℕ, special_sequence a q n = m) ∧
  (∀ n : ℕ, special_sequence a q n < special_sequence a q (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_natural_and_increasing_l151_15121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l151_15165

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (3/2) 3 ∧ a ≠ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l151_15165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antifreeze_concentration_problem_l151_15139

/-- Calculates the final antifreeze concentration after replacing part of a solution with pure antifreeze -/
noncomputable def final_antifreeze_concentration (initial_volume : ℝ) (initial_concentration : ℝ) 
  (replaced_volume : ℝ) : ℝ :=
  let remaining_antifreeze := initial_volume * initial_concentration - replaced_volume * initial_concentration
  let total_antifreeze := remaining_antifreeze + replaced_volume
  total_antifreeze / initial_volume

/-- The final antifreeze concentration is 50% given the specified initial conditions -/
theorem antifreeze_concentration_problem :
  final_antifreeze_concentration 10 0.3 2.85714285714 = 0.5 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval final_antifreeze_concentration 10 0.3 2.85714285714

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antifreeze_concentration_problem_l151_15139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x0_value_l151_15198

theorem max_x0_value (x : Fin 1996 → ℝ) 
  (h_positive : ∀ i, x i > 0)
  (h_equal : x 0 = x 1995)
  (h_relation : ∀ i : Fin 1995, x i.val + 2 / x i.val = 2 * x (i.val + 1) + 1 / x (i.val + 1)) :
  x 0 ≤ 2^997 ∧ ∃ x' : Fin 1996 → ℝ, x' 0 = 2^997 ∧
    (∀ i, x' i > 0) ∧
    x' 0 = x' 1995 ∧
    (∀ i : Fin 1995, x' i.val + 2 / x' i.val = 2 * x' (i.val + 1) + 1 / x' (i.val + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x0_value_l151_15198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_disc_rotation_theorem_l151_15183

/-- Represents a circular disc with 2n sectors, n white and n black -/
structure ColoredDisc (n : ℕ) where
  sectors : Fin (2 * n) → Bool

/-- Calculates the total arc length where colors differ for a given rotation -/
def arc_length_diff_color (n : ℕ) (large_disc small_disc : ColoredDisc n) (rotation : ℝ) : ℝ :=
  sorry

/-- The theorem statement -/
theorem colored_disc_rotation_theorem (n : ℕ) (large_disc small_disc : ColoredDisc n) :
  ∃ (rotation : ℝ), 
    rotation ≥ 0 ∧ rotation < 2 * Real.pi ∧
    (arc_length_diff_color n large_disc small_disc rotation ≥ Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_disc_rotation_theorem_l151_15183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryegrass_percentage_in_mixture_l151_15142

theorem ryegrass_percentage_in_mixture (x_ryegrass : Real) (x_bluegrass : Real) 
  (y_ryegrass : Real) (y_fescue : Real) (x_proportion : Real) :
  x_ryegrass = 0.40 →
  x_bluegrass = 0.60 →
  y_ryegrass = 0.25 →
  y_fescue = 0.75 →
  x_proportion = 0.13333333333333332 →
  x_ryegrass + x_bluegrass = 1 →
  y_ryegrass + y_fescue = 1 →
  (x_ryegrass * x_proportion + y_ryegrass * (1 - x_proportion)) = 0.27 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryegrass_percentage_in_mixture_l151_15142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_tax_rate_is_four_percent_l151_15127

/-- Given an item price, original tax rate, and tax difference after reduction,
    calculate the new tax rate. -/
noncomputable def new_tax_rate (price : ℝ) (original_rate : ℝ) (tax_difference : ℝ) : ℝ :=
  original_rate - (tax_difference / price * 100)

/-- Theorem stating that the new tax rate is 4% given the problem conditions. -/
theorem new_tax_rate_is_four_percent :
  let price : ℝ := 1000
  let original_rate : ℝ := 5
  let tax_difference : ℝ := 10
  new_tax_rate price original_rate tax_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_tax_rate_is_four_percent_l151_15127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_probability_at_C_l151_15178

/-- Represents a point on the lattice -/
structure LatticePoint where
  x : Int
  y : Int

/-- Represents the lattice -/
def lattice : Set LatticePoint := {p | p.x ≥ -2 ∧ p.x ≤ 3 ∧ p.y ≥ -2 ∧ p.y ≤ 3}

/-- The starting point A -/
def A : LatticePoint := ⟨0, 0⟩

/-- The target point C -/
def C : LatticePoint := ⟨0, 2⟩

/-- Function to determine if a point is red (including A and C) -/
def isRed (p : LatticePoint) : Prop := (p.x + p.y) % 2 = 0

/-- The set of all red points on the lattice -/
def RedPoints : Set LatticePoint := {p ∈ lattice | isRed p}

/-- The number of steps the ant takes -/
def steps : Nat := 7

/-- Represents the position of the ant after a certain number of steps -/
def antPosition (steps : Nat) : LatticePoint := sorry

/-- Represents the probability measure -/
noncomputable def Prob {α : Type*} (event : Set α) : ℝ := sorry

/-- Theorem: The probability of the ant being at C after 7 steps is 1/9 -/
theorem ant_probability_at_C : 
  Prob {p : LatticePoint | p = antPosition steps ∧ p = C} = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_probability_at_C_l151_15178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_seven_halves_l151_15101

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -1 / x else x^2

-- State the theorem
theorem f_sum_equals_seven_halves : f 2 + f (-2) = 7/2 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if-then-else expressions
  simp [if_pos (show 2 > 0 from by norm_num), if_neg (show ¬(-2 > 0) from by norm_num)]
  -- Perform arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_seven_halves_l151_15101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_inequality_system_l151_15185

theorem range_of_a_for_inequality_system (a : ℝ) : 
  (∃ (S : Finset ℤ), S.card = 5 ∧ 
    (∀ x ∈ S, (5 - 2*x ≥ -1 ∧ x - a > 0))) → 
  (-2 ≤ a ∧ a < -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_inequality_system_l151_15185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_mappings_count_l151_15105

def count_special_mappings (n : ℕ) : ℕ :=
  if n < 3 then 0 else (2 * n - 5) / 2 * n.factorial

theorem special_mappings_count {n : ℕ} (h : n ≥ 3) :
  let A := Fin n
  (Fintype.card { f : A → A // 
    (∃ c : A, ∀ x : A, (f^[n-2]) x = c) ∧
    (¬ ∃ c : A, ∀ x : A, (f^[n]) x = c) }) =
  count_special_mappings n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_mappings_count_l151_15105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_solution_of_nested_cube_root_equation_l151_15175

-- Define the function g(y)
noncomputable def g (y : ℝ) : ℝ := (15 * y + (15 * y + 8) ^ (1/3)) ^ (1/3)

-- State the theorem
theorem approximate_solution_of_nested_cube_root_equation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |g (168/5) - 8| < ε := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_solution_of_nested_cube_root_equation_l151_15175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_n4_minus_n2_l151_15140

theorem largest_divisor_of_n4_minus_n2 :
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (n : ℤ), (k : ℤ) ∣ (n^4 - n^2)) ∧ 
  (∀ (m : ℕ), m > k → ∃ (n : ℤ), ¬((m : ℤ) ∣ (n^4 - n^2))) ∧ 
  k = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_n4_minus_n2_l151_15140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_per_camper_approx_l151_15194

/-- Represents the fishing scenario with trout and salmon --/
structure FishingScenario where
  troutWeight : ℚ
  salmonWeight : ℚ
  salmonCount : ℕ
  camperCount : ℕ

/-- Calculates the total weight of trout and salmon --/
def totalWeight (scenario : FishingScenario) : ℚ :=
  scenario.troutWeight + scenario.salmonWeight * scenario.salmonCount

/-- Calculates the amount of fish per camper --/
noncomputable def fishPerCamper (scenario : FishingScenario) : ℚ :=
  totalWeight scenario / scenario.camperCount

/-- Theorem stating that in the given scenario, each camper eats approximately 1.45 pounds of fish --/
theorem fish_per_camper_approx (scenario : FishingScenario) 
  (h1 : scenario.troutWeight = 8)
  (h2 : scenario.salmonWeight = 12)
  (h3 : scenario.salmonCount = 2)
  (h4 : scenario.camperCount = 22) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |fishPerCamper scenario - 145/100| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_per_camper_approx_l151_15194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_concentric_circles_l151_15189

/-- Given two concentric circles where the outer radius is three times the inner radius,
    and the width between the circles is 3 feet, the area of the region between
    the circles is 18π square feet. -/
theorem area_between_concentric_circles :
  ∀ (r : ℝ), r > 0 →
  (3 * r - r = 3) →
  (π * (3 * r)^2 - π * r^2 : ℝ) = 18 * π :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_concentric_circles_l151_15189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_31_and_32_l151_15196

-- Define the floor function
noncomputable def floor (z : ℝ) : ℤ := Int.floor z

-- Define the problem conditions
def problem_conditions (x y : ℝ) : Prop :=
  y = 4 * (floor x) + 1 ∧
  y = 2 * (floor (x + 3)) + 7 ∧
  ¬(∃ n : ℤ, x = n)

-- Theorem statement
theorem x_plus_y_between_31_and_32 {x y : ℝ} 
  (h : problem_conditions x y) : 
  31 < x + y ∧ x + y < 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_31_and_32_l151_15196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_max_value_l151_15144

theorem triangle_angle_max_value (A B C : ℝ) :
  (A + B + C = π) →
  ((Real.sin A + Real.sqrt 3 * Real.cos A) / (Real.cos A - Real.sqrt 3 * Real.sin A) = Real.tan (7 * π / 12)) →
  (∀ B' C', A + B' + C' = π → Real.sin (2 * B') + 2 * Real.cos C' ≤ 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_max_value_l151_15144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l151_15191

/-- The function f(x) = a^(x-1) - 2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) - 2

/-- The theorem states that (1, -1) is the only point that always lies on the graph of f -/
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x y : ℝ, (∀ b : ℝ, b > 0 → b ≠ 1 → f b x = y) → x = 1 ∧ y = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l151_15191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stuffing_time_is_one_minute_l151_15199

/-- The time it takes Earl and Ellen to stuff 60 envelopes together -/
noncomputable def stuffing_time (earl_rate : ℝ) (ellen_rate : ℝ) (total_envelopes : ℝ) : ℝ :=
  total_envelopes / (earl_rate + ellen_rate / 1.5)

/-- Theorem stating that Earl and Ellen can stuff 60 envelopes in 1 minute -/
theorem stuffing_time_is_one_minute :
  stuffing_time 36 36 60 = 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stuffing_time_is_one_minute_l151_15199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_minus_one_squared_l151_15187

open MeasureTheory

theorem integral_sqrt_one_minus_x_minus_one_squared : 
  ∫ x in Set.Icc 0 1, Real.sqrt (1 - (x - 1)^2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_minus_one_squared_l151_15187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l151_15126

-- Define the functions f and g
def f (x m : ℝ) : ℝ := -(x - 1)^2 + m
noncomputable def g (x : ℝ) : ℝ := x * Real.exp x

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∃ x₁ x₂ : ℝ, f x₁ m ≥ g x₂) → m ≥ -1 / Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l151_15126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l151_15102

/-- The area of a trapezoid with given bases and height -/
noncomputable def trapezoid_area (base1 : ℝ) (base2 : ℝ) (height : ℝ) : ℝ :=
  (base1 + base2) * height / 2

/-- Theorem: The area of a trapezoid with bases 200 and 140, and height 80, is 13600 -/
theorem park_area :
  trapezoid_area 200 140 80 = 13600 := by
  -- Unfold the definition of trapezoid_area
  unfold trapezoid_area
  -- Simplify the arithmetic expression
  simp [mul_add, mul_div_assoc]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l151_15102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_largest_angle_l151_15176

-- Define a hexagon type
structure Hexagon where
  angles : Fin 6 → ℕ
  is_convex : ∀ i, angles i < 180
  is_arithmetic_seq : ∃ d : ℕ, ∀ i : Fin 5, angles i.succ = angles i + d
  sum_720 : (angles 0) + (angles 1) + (angles 2) + (angles 3) + (angles 4) + (angles 5) = 720

-- Define the largest angle function
def largest_angle (h : Hexagon) : ℕ := 
  Finset.max' (Finset.univ.image h.angles) (by simp [Finset.univ_nonempty])

-- State the theorem
theorem max_largest_angle :
  ∀ h : Hexagon, largest_angle h ≤ 175 ∧ ∃ h : Hexagon, largest_angle h = 175 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_largest_angle_l151_15176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_is_one_l151_15192

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = x^2

/-- A right triangle on a parabola -/
structure RightTriangleOnParabola where
  A : ParabolaPoint
  B : ParabolaPoint
  C : ParabolaPoint
  hypotenuse_parallel : A.y = B.y
  is_right_triangle : (C.x - A.x) * (C.x - B.x) = (C.y - A.y) * (B.x - A.x)

/-- The height of the triangle from C to AB -/
def triangle_height (t : RightTriangleOnParabola) : ℝ :=
  t.C.y - t.A.y

/-- Theorem: The height of the right triangle on parabola is 1 -/
theorem height_is_one (t : RightTriangleOnParabola) : triangle_height t = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_is_one_l151_15192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_bushes_l151_15107

/-- Represents the number of trees in the meadow -/
def T : ℕ := sorry

/-- Represents the number of bushes in the meadow -/
def B : ℕ := sorry

/-- Represents the number of birds on each tree -/
def x : ℕ := sorry

/-- Represents the number of birds on each bush -/
def y : ℕ := sorry

/-- There are 6 fewer bushes than trees -/
axiom bush_tree_relation : B = T - 6

/-- Each tree has at least 10 more birds than each bush -/
axiom bird_distribution : x ≥ y + 10

/-- Total number of birds on the trees is 128 -/
axiom total_birds_on_trees : T * x = 128

/-- The number of bushes is 2 -/
theorem number_of_bushes : B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_bushes_l151_15107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_w_l151_15123

theorem smallest_positive_w (y w : Real) : 
  Real.sin y = 1 → 
  Real.sin (y + w) = Real.sqrt 3 / 2 → 
  w > 0 → 
  (∀ w' : Real, w' > 0 ∧ Real.sin y = 1 ∧ Real.sin (y + w') = Real.sqrt 3 / 2 → w ≤ w') → 
  w = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_w_l151_15123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_AB_AC_l151_15182

def A : Fin 3 → ℝ := ![2, -8, -1]
def B : Fin 3 → ℝ := ![4, -6, 0]
def C : Fin 3 → ℝ := ![-2, -5, -1]

def vector_AB : Fin 3 → ℝ := λ i => B i - A i
def vector_AC : Fin 3 → ℝ := λ i => C i - A i

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

theorem cosine_of_angle_AB_AC :
  dot_product vector_AB vector_AC / (magnitude vector_AB * magnitude vector_AC) = -2/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_AB_AC_l151_15182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_A_l151_15108

-- Define the right triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0
  ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 1
  bc_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 3

-- Define the theorem
theorem sine_of_angle_A (t : RightTriangle) : 
  Real.sin (Real.arctan ((t.C.2 - t.A.2) / (t.C.1 - t.A.1))) = 3 * Real.sqrt 10 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_A_l151_15108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_seven_point_five_l151_15143

/-- The distance to the place given the man's rowing speed, current velocity, and round trip time --/
noncomputable def distance_to_place (rowing_speed : ℝ) (current_velocity : ℝ) (total_time : ℝ) : ℝ :=
  (rowing_speed * total_time) / 2

/-- Theorem stating that given the specific conditions, the distance to the place is 7.5 km --/
theorem distance_is_seven_point_five :
  let rowing_speed : ℝ := 8
  let current_velocity : ℝ := 2
  let total_time : ℝ := 2
  distance_to_place rowing_speed current_velocity total_time = 7.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval distance_to_place 8 2 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_seven_point_five_l151_15143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_digit_is_six_l151_15180

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℕ
  repeatingPart : List ℕ
  repeatingPartNonempty : repeatingPart.length > 0

/-- The repeating decimal representation of 2.1756756... -/
def number : RepeatingDecimal :=
  { integerPart := 2,
    repeatingPart := [7, 5, 6],
    repeatingPartNonempty := by simp }

/-- Returns the digit at the nth position after the decimal point in a repeating decimal -/
def digitAtPosition (d : RepeatingDecimal) (n : ℕ) : ℕ :=
  d.repeatingPart[((n - 1) % d.repeatingPart.length) % d.repeatingPart.length]'
    (by
      apply Nat.mod_lt
      exact d.repeatingPartNonempty)

/-- Theorem: The 100th digit after the decimal point in the repeating decimal 2.1̇75̇6̇ is 6 -/
theorem hundredth_digit_is_six :
  digitAtPosition number 100 = 6 := by
  rw [digitAtPosition, number]
  simp
  sorry

#eval digitAtPosition number 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_digit_is_six_l151_15180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_power_mn_equals_p_2n_q_m_l151_15156

theorem eighteen_power_mn_equals_p_2n_q_m (m n : ℤ) (P Q : ℕ) 
  (h1 : P = (3 : ℕ)^(m.natAbs)) (h2 : Q = (2 : ℕ)^(n.natAbs)) : 
  (18 : ℕ)^(m.natAbs * n.natAbs) = P^(2*n.natAbs) * Q^(m.natAbs) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_power_mn_equals_p_2n_q_m_l151_15156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l151_15113

/-- Converts speed from km/h to m/s -/
noncomputable def kmh_to_ms (v : ℝ) : ℝ := v * (1000 / 3600)

/-- Calculates the effective speed of the train considering headwind -/
noncomputable def effective_speed (train_speed headwind : ℝ) : ℝ :=
  kmh_to_ms train_speed - kmh_to_ms headwind

/-- Calculates the stopping distance of the train -/
noncomputable def stopping_distance (initial_speed deceleration : ℝ) : ℝ :=
  (initial_speed ^ 2) / (2 * deceleration)

/-- Calculates the length of the bridge -/
noncomputable def bridge_length (train_length stopping_dist : ℝ) : ℝ :=
  train_length + stopping_dist

theorem bridge_length_calculation (train_length : ℝ) (train_speed : ℝ) (deceleration : ℝ) (headwind : ℝ)
  (h1 : train_length = 200)
  (h2 : train_speed = 60)
  (h3 : deceleration = 2)
  (h4 : headwind = 10) :
  ∃ (ε : ℝ), abs (bridge_length train_length (stopping_distance (effective_speed train_speed headwind) deceleration) - 248.30) < ε ∧ ε > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l151_15113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expr_l151_15186

-- Define the vectors
noncomputable def a : ℝ × ℝ := (3, -2)
noncomputable def b (x y : ℝ) : ℝ × ℝ := (x, y - 1)

-- Define the parallelism condition
def parallel (x y : ℝ) : Prop := 3 * (y - 1) = -2 * x

-- Define the expression to be minimized
noncomputable def expr (x y : ℝ) : ℝ := 3 / x + 2 / y

-- Theorem statement
theorem min_value_of_expr (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_parallel : parallel x y) :
  expr x y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ parallel x₀ y₀ ∧ expr x₀ y₀ = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expr_l151_15186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l151_15104

/-- The original function f(x) = (1-x)/(1+x) -/
noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

/-- The function g(x) = f(x-1) + 1 -/
noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

/-- Theorem: g is an odd function -/
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l151_15104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_series_l151_15119

/-- The function f(x) = x^x for x ∈ [0, 1] -/
noncomputable def f (x : ℝ) : ℝ := x^x

/-- The infinite alternating series 1 - 1/2² + 1/3³ - 1/4⁴ + ... -/
noncomputable def alternating_series : ℝ := ∑' n, (-1)^n / (n + 1)^(n + 1)

/-- Theorem stating that the integral of x^x from 0 to 1 equals the alternating series -/
theorem integral_equals_series : ∫ x in Set.Icc 0 1, f x = alternating_series := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_series_l151_15119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_value_l151_15193

/-- A cubic function passing through specific points -/
def cubic_function (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

theorem cubic_function_value (a b c d : ℝ) :
  (cubic_function a b c d 1 = 4) →
  (cubic_function a b c d 0 = -2) →
  (cubic_function a b c d (-1) = -8) →
  4 * a - 2 * b + c - 3 * d = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_value_l151_15193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_chord_length_l151_15157

noncomputable def circle_C (x y : ℝ) (D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

noncomputable def center_C : ℝ × ℝ := (-Real.sqrt 2, 2 * Real.sqrt 2)
noncomputable def radius_C : ℝ := Real.sqrt 2

def tangent_line (x y : ℝ) : Prop :=
  3*x + 4*y = 0

def y_axis (x : ℝ) : Prop :=
  x = 0

def intersecting_line (x y : ℝ) : Prop :=
  x - y + 2 * Real.sqrt 2 = 0

theorem circle_properties :
  ∃ (D E F : ℝ),
    (∀ x y, circle_C x y D E F ↔ (x - center_C.1)^2 + (y - center_C.2)^2 = radius_C^2) ∧
    (∃ x y, circle_C x y D E F ∧ tangent_line x y) ∧
    (∃ y, circle_C 0 y D E F) ∧
    D = 2 * Real.sqrt 2 ∧
    E = -4 * Real.sqrt 2 ∧
    F = 8 := by
  sorry

theorem chord_length :
  ∃ A B : ℝ × ℝ,
    (circle_C A.1 A.2 (2 * Real.sqrt 2) (-4 * Real.sqrt 2) 8) ∧
    (circle_C B.1 B.2 (2 * Real.sqrt 2) (-4 * Real.sqrt 2) 8) ∧
    (intersecting_line A.1 A.2) ∧
    (intersecting_line B.1 B.2) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_chord_length_l151_15157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_male_count_l151_15148

theorem stratified_sample_male_count :
  ∀ (total_male total_female sample_size : ℕ),
    total_male = 56 →
    total_female = 42 →
    sample_size = 28 →
    let total := total_male + total_female
    let sample_ratio := (sample_size : ℚ) / total
    let male_in_sample := (total_male : ℚ) * sample_ratio
    male_in_sample = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_male_count_l151_15148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_targets_lower_bound_l151_15195

/-- The expected number of hit targets when n boys randomly choose from n targets -/
noncomputable def E (n : ℕ) : ℝ := n * (1 - (1 - 1 / n)^n)

/-- Theorem stating that the expected number of hit targets is always at least n/2 -/
theorem expected_targets_lower_bound (n : ℕ) (hn : n > 0) : E n ≥ n / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_targets_lower_bound_l151_15195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l151_15118

noncomputable section

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
def C₂ (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

-- Define the common chord
def common_chord (x y : ℝ) : Prop := 2*x + y = 0

-- Define the center of C₁
def center_C₁ : ℝ × ℝ := (2, 1)

-- Define the radius of C₁
noncomputable def radius_C₁ : ℝ := Real.sqrt 10

-- Theorem statement
theorem common_chord_length :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧
    common_chord A.1 A.2 ∧ common_chord B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l151_15118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x5_expansion_l151_15152

def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

def coefficient_x5 (a b c : ℤ) : ℤ :=
  let f1 := λ (r : ℕ) => (binomial_coefficient 6 r : ℤ) * (-1 : ℤ)^r
  let f2 := λ (s : ℕ) => (binomial_coefficient 6 s : ℤ) * 3^(6-s)
  (f1 1 * f2 6) + (3 * f1 2 * f2 5) + (3^2 * f1 3 * f2 4) +
  (3^3 * f1 4 * f2 3) + (3^4 * f1 5 * f2 2) + (3^5 * f1 6 * f2 1)

theorem coefficient_x5_expansion :
  coefficient_x5 1 2 (-3) = -168 := by
  sorry

#eval coefficient_x5 1 2 (-3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x5_expansion_l151_15152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_with_tangent_circles_l151_15145

/-- A circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- An equilateral triangle with side length -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Configuration of three circles tangent to each other and to a side of an equilateral triangle -/
structure CircleConfiguration where
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle
  triangle : EquilateralTriangle

/-- Predicate to check if the configuration is valid -/
def isValidConfiguration (config : CircleConfiguration) : Prop :=
  config.circle1.radius = 2 ∧
  config.circle2.radius = 2 ∧
  config.circle3.radius = 2 ∧
  -- Add conditions for tangency and alignment here
  True  -- Placeholder for additional conditions

theorem perimeter_of_triangle_with_tangent_circles 
  (config : CircleConfiguration) 
  (h : isValidConfiguration config) : 
  3 * config.triangle.sideLength = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_with_tangent_circles_l151_15145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_integer_part_l151_15128

/-- The integer part of a real number -/
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

/-- The theorem to be proved -/
theorem remainder_of_integer_part (n : ℕ) (h : n ≥ 2009) :
  (integerPart ((3 + Real.sqrt 8) ^ (2 * n))) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_integer_part_l151_15128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_right_triangles_construction_l151_15161

/-- Given a hypotenuse length, an acute angle, and a leg length, 
    two right-angled triangles can be constructed. -/
theorem two_right_triangles_construction 
  (h : ℝ) (α : ℝ) (k : ℝ) 
  (h_pos : h > 0) 
  (α_pos : α > 0) 
  (α_acute : α < Real.pi / 2) 
  (k_pos : k > 0) 
  (k_less_h : k < h) : 
  ∃ (a b c d : ℝ), 
    a^2 + b^2 = h^2 ∧ 
    Real.sin α = b / h ∧
    c^2 + k^2 = h^2 ∧
    a > 0 ∧ b > 0 ∧ c > 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_right_triangles_construction_l151_15161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_correct_total_sector_area_enclosed_area_proof_l151_15134

/-- The area enclosed by a shape composed of 12 congruent circular arcs, 
    each with length π/2, centered at the vertices of a regular octagon 
    with side length 3 -/
noncomputable def enclosed_area : ℝ := 18 * (1 + Real.sqrt 2) + 3 * Real.pi / 2

/-- The length of each circular arc -/
noncomputable def arc_length : ℝ := Real.pi / 2

/-- The number of circular arcs -/
def num_arcs : ℕ := 12

/-- The side length of the regular octagon -/
def octagon_side : ℝ := 3

/-- Theorem stating that the enclosed area is correct given the conditions -/
theorem enclosed_area_correct : 
  enclosed_area = 18 * (1 + Real.sqrt 2) + 3 * Real.pi / 2 :=
by
  -- Unfold the definition of enclosed_area
  unfold enclosed_area
  -- The equality holds by definition
  rfl

/-- Lemma: The radius of each circular arc is 1/2 -/
lemma arc_radius : ℝ := 1 / 2

/-- Lemma: The area of the regular octagon -/
noncomputable def octagon_area : ℝ := 18 * (1 + Real.sqrt 2)

/-- Lemma: The area of one sector -/
noncomputable def sector_area : ℝ := Real.pi / 8

/-- Theorem: The total area of all sectors -/
theorem total_sector_area : num_arcs * sector_area = 3 * Real.pi / 2 :=
by sorry

/-- Main theorem proving that the enclosed area is correct -/
theorem enclosed_area_proof :
  enclosed_area = octagon_area + num_arcs * sector_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_correct_total_sector_area_enclosed_area_proof_l151_15134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proof_methods_characterization_l151_15167

/- Define the proof methods -/
inductive ProofMethod
| Synthetic
| Analytic
| Contradiction

/- Define the characteristics of proof methods -/
axiom isCauseToEffect : ProofMethod → Prop
axiom isEffectToCause : ProofMethod → Prop
axiom isDirectMethod : ProofMethod → Prop

/- Define the statements -/
def statement1 : Prop := isCauseToEffect ProofMethod.Synthetic
def statement2 : Prop := ¬isDirectMethod ProofMethod.Analytic
def statement3 : Prop := isEffectToCause ProofMethod.Analytic
def statement4 : Prop := isDirectMethod ProofMethod.Contradiction

/- Define the set of correct statements -/
def correctStatements : Set Prop := {statement1, statement3}

/- Theorem to prove -/
theorem proof_methods_characterization :
  (∀ m : ProofMethod, isDirectMethod m ↔ m = ProofMethod.Synthetic ∨ m = ProofMethod.Analytic) →
  (¬isDirectMethod ProofMethod.Contradiction) →
  isCauseToEffect ProofMethod.Synthetic →
  isEffectToCause ProofMethod.Analytic →
  correctStatements = {statement1, statement3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proof_methods_characterization_l151_15167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l151_15117

noncomputable def f (x : ℝ) := Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

noncomputable def g (x : ℝ) := f (x + Real.pi/4)

theorem function_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
   ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧ 
  (∀ (k : ℤ), ∃ (c : ℝ), ∀ (x : ℝ), f (2 * (Real.pi/6 + k * Real.pi/2) - x) = f (x + c)) ∧
  (∃ (M : ℝ), M = 1/2 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/6) (Real.pi/3) → g x ≤ M) ∧
  (∃ (m : ℝ), m = -1/4 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/6) (Real.pi/3) → m ≤ g x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l151_15117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l151_15181

noncomputable section

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the point
def point : ℝ × ℝ := (3, 0)

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

-- Theorem statement
theorem distance_to_asymptote :
  ∃ (A B C : ℝ), 
    (∀ x y, hyperbola x y → (A * x + B * y + C = 0 ∨ A * x - B * y + C = 0)) ∧
    distance_point_to_line point.1 point.2 A B C = 9/5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l151_15181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_characterization_valid_arrangement_4l_valid_arrangement_4l_minus_1_l151_15147

/-- A permutation of the multiset [1, 1, 2, 2, ..., n, n] -/
def ValidArrangement (n : ℕ) (p : List ℕ) : Prop :=
  (∀ k ∈ Finset.range n, p.count k.succ = 2) ∧
  (∀ k ∈ Finset.range n, ∃ i j, i < j ∧ p.get? i = some k.succ ∧ p.get? j = some k.succ ∧ j - i - 1 = k.succ)

/-- The main theorem -/
theorem valid_arrangement_characterization (n : ℕ) :
  (∃ p : List ℕ, ValidArrangement n p) ↔ (∃ l : ℕ, n = 4 * l ∨ n = 4 * l - 1) := by
  sorry

/-- Helper function to construct a valid arrangement for n = 4l -/
def constructArrangement4l (l : ℕ) : List ℕ := sorry

/-- Helper function to construct a valid arrangement for n = 4l-1 -/
def constructArrangement4lMinus1 (l : ℕ) : List ℕ := sorry

/-- Proof that the construction for n = 4l is valid -/
theorem valid_arrangement_4l (l : ℕ) : ValidArrangement (4 * l) (constructArrangement4l l) := by
  sorry

/-- Proof that the construction for n = 4l-1 is valid -/
theorem valid_arrangement_4l_minus_1 (l : ℕ) : ValidArrangement (4 * l - 1) (constructArrangement4lMinus1 l) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_characterization_valid_arrangement_4l_valid_arrangement_4l_minus_1_l151_15147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_line_equation_l151_15111

open Matrix

-- Define the matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, 2]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 0, 1]

-- Define the equation of line l'
def l' (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem original_line_equation :
  ∀ (x y : ℝ),
  let AB_inv := A * B⁻¹
  (∃ (x₀ y₀ : ℝ), l' (AB_inv.vecMul ![x₀, y₀] 0) (AB_inv.vecMul ![x₀, y₀] 1)) →
  x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_line_equation_l151_15111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_electricity_consumption_l151_15122

/-- Represents the average monthly electricity consumption in kWh for the first half of the year -/
def x : ℝ := sorry

/-- Represents the decrease in average monthly electricity consumption in kWh for the second half of the year -/
def decrease : ℝ := 2000

/-- Represents the total annual electricity consumption in kWh -/
def total_consumption : ℝ := 150000

/-- Theorem stating that the equation representing the factory's electricity consumption is correct -/
theorem factory_electricity_consumption : 6 * x + 6 * (x - decrease) = total_consumption := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_electricity_consumption_l151_15122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equal_g_inv_iff_x_eq_neg_two_l151_15135

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * x + 4

-- Define the inverse function of g
noncomputable def g_inv (x : ℝ) : ℝ := (x - 4) / 3

-- Theorem statement
theorem g_equal_g_inv_iff_x_eq_neg_two :
  ∀ x : ℝ, g x = g_inv x ↔ x = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equal_g_inv_iff_x_eq_neg_two_l151_15135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_identity_l151_15168

theorem cosine_sum_identity (α : ℝ) :
  Real.cos α + Real.cos (2 * α) + Real.cos (6 * α) + Real.cos (7 * α) = 
  4 * Real.cos (α / 2) * Real.cos (5 * α / 2) * Real.cos (4 * α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_identity_l151_15168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imo_is_perfect_square_l151_15172

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundred_nonzero : hundreds ≠ 0
  hundred_less_ten : hundreds < 10
  tens_less_ten : tens < 10
  ones_less_ten : ones < 10

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Converts the first two digits of a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.firstTwoDigits (n : ThreeDigitNumber) : Nat :=
  10 * n.hundreds + n.tens

/-- Inserts 2021 zeros between each digit of a ThreeDigitNumber -/
def ThreeDigitNumber.insertZeros (n : ThreeDigitNumber) : Nat :=
  n.hundreds * (10^4042) + n.tens * (10^2021) + n.ones

/-- The main theorem -/
theorem imo_is_perfect_square (n : ThreeDigitNumber) 
  (h : (n.toNat : ℝ).sqrt = (n.firstTwoDigits : ℝ) - (n.ones : ℝ).sqrt) :
  ∃ m : Nat, n.insertZeros = m^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imo_is_perfect_square_l151_15172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_condition_l151_15190

/-- Given two vectors a and b in a real vector space, 
    if they are not parallel and a + (1/4)λb is parallel to -a + b, 
    then λ = -4 -/
theorem vector_parallel_condition (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V] 
  (a b : V) (lambda : ℝ) 
  (h1 : ¬ ∃ (k : ℝ), a = k • b) 
  (h2 : ∃ (μ : ℝ), a + (1/4 * lambda) • b = μ • (-a + b)) : 
  lambda = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_condition_l151_15190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l151_15106

theorem trigonometric_identities (α : ℝ) 
  (h1 : Real.cos (π + α) = -3/5)
  (h2 : α ∈ Set.Ioo (3*π/2) (2*π)) :
  Real.sin (π/2 + α) = 3/5 ∧ 
  Real.cos (2*α) = -7/25 ∧ 
  Real.sin (α - π/4) = -7*Real.sqrt 2/10 ∧ 
  Real.tan (α/2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l151_15106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sneakers_final_price_l151_15125

/-- Calculate the final price of sneakers after applying a coupon and membership discount -/
theorem sneakers_final_price (original_price coupon_discount membership_discount_percent : ℚ) :
  original_price = 120 →
  coupon_discount = 10 →
  membership_discount_percent = 10 →
  (original_price - coupon_discount) * (1 - membership_discount_percent / 100) = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sneakers_final_price_l151_15125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_of_intersection_l151_15138

/-- A line parameterized by lambda -/
def line (lambda : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | lambda * p.1 - p.2 - lambda + 1 = 0}

/-- The circle C -/
def circleC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.2 = 0}

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_of_intersection (lambda : ℝ) :
  ∃ (A B : ℝ × ℝ), A ∈ line lambda ∧ A ∈ circleC ∧ B ∈ line lambda ∧ B ∈ circleC ∧
  ∀ (X Y : ℝ × ℝ), X ∈ line lambda ∧ X ∈ circleC ∧ Y ∈ line lambda ∧ Y ∈ circleC →
  distance A B ≤ distance X Y ∧ distance A B = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_of_intersection_l151_15138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_capacity_l151_15173

theorem water_tank_capacity : ∃ c : ℝ, c = 75 ∧ c / 3 + 5 = 2 * c / 5 := by
  -- Define the initial water level and tank capacity
  let initial_level (c : ℝ) := c / 3
  let after_adding (c : ℝ) := (initial_level c + 5) / c

  -- Prove the tank capacity is 75 liters
  use 75
  constructor
  · rfl
  · norm_num

  -- Note: The full proof is omitted for brevity, but can be completed later

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_capacity_l151_15173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_cos_2x0_value_l151_15169

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x) * (Real.cos x - Real.sqrt 3 * Real.sin x)

def monotonic_increasing_intervals : Set ℝ :=
  {x | ∃ n : ℕ, x ∈ Set.Icc (Real.pi/4 + n*Real.pi) (3*Real.pi/4 + n*Real.pi)}

theorem f_monotonic_increasing :
  {x | ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y > f x} = monotonic_increasing_intervals :=
sorry

theorem cos_2x0_value (x0 : ℝ) (h1 : f x0 = 6/5) (h2 : x0 ∈ Set.Icc 0 (Real.pi/2)) :
  Real.cos (2*x0) = (4 + 3*Real.sqrt 3)/10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_cos_2x0_value_l151_15169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_satisfying_inequality_l151_15162

open BigOperators

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 9  -- Adding the base case for 0
  | 1 => 9
  | (n + 2) => (4 - sequence_a (n + 1)) / 3

noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, sequence_a (i + 1)

theorem smallest_n_satisfying_inequality :
  (∀ n < 7, |S n - n - 6| ≥ 1/125) ∧ |S 7 - 7 - 6| < 1/125 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_satisfying_inequality_l151_15162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_time_l151_15151

/-- Given two people who can complete a task in time1 and time2 minutes respectively,
    this function calculates the time it takes them to complete the task together. -/
noncomputable def combined_time (time1 time2 : ℝ) : ℝ :=
  1 / (1 / time1 + 1 / time2)

/-- Theorem stating that for two people who can complete a task in 30 and 45 minutes respectively,
    it takes them 18 minutes to complete the task together. -/
theorem task_completion_time : combined_time 30 45 = 18 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval combined_time 30 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_time_l151_15151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_range_l151_15146

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

-- State the theorem
theorem f_min_value_range (a : ℝ) :
  (∀ x, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_range_l151_15146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alices_and_charlies_average_score_l151_15174

theorem alices_and_charlies_average_score (total_students : ℕ) 
  (absent_students : ℕ) (initial_average : ℚ) (final_average : ℚ) :
  total_students = 20 →
  absent_students = 3 →
  initial_average = 78 →
  final_average = 80 →
  let graded_initially : ℕ := total_students - absent_students
  let graded_finally : ℕ := total_students - 1
  let alice_and_charlie_average : ℚ := 
    (final_average * graded_finally - initial_average * graded_initially) / 2
  alice_and_charlie_average = 97 := by
  sorry

-- Remove the #eval line as it's not necessary and was causing an error

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alices_and_charlies_average_score_l151_15174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l151_15115

theorem equation_solution :
  ∃ x : ℝ, 5 * (5 : ℝ)^x + Real.sqrt (25 * (25 : ℝ)^x) = 50 ∧ x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l151_15115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sticks_for_rectangle_l151_15120

/-- Represents a stick with an integer length in centimeters -/
structure Stick where
  length : ℕ

/-- Represents a collection of sticks cut from a 2-meter rod -/
structure RodCut where
  sticks : List Stick
  total_length : ℕ
  h_total_length : total_length = 200  -- 2 meters = 200 cm
  h_integer_lengths : ∀ s ∈ sticks, s.length > 0

/-- Predicate to check if a rectangle can be formed from given sticks -/
def can_form_rectangle (cut : RodCut) : Prop :=
  ∃ (w h : ℕ), w > 0 ∧ h > 0 ∧ w + h = cut.sticks.length ∧
  (∃ (w_sticks h_sticks : List Stick),
    w_sticks.length = w ∧
    h_sticks.length = h ∧
    w_sticks ++ h_sticks = cut.sticks ∧
    (w_sticks.map (λ s => s.length)).sum = (h_sticks.map (λ s => s.length)).sum)

/-- The main theorem stating the minimum number of sticks required -/
theorem min_sticks_for_rectangle :
  ∀ n : ℕ, n ≥ 102 →
    (∀ cut : RodCut, cut.sticks.length = n → can_form_rectangle cut) ∧
    (∃ cut : RodCut, cut.sticks.length = 101 ∧ ¬can_form_rectangle cut) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sticks_for_rectangle_l151_15120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pencil_lifts_for_specific_graph_l151_15112

/-- A graph with vertices and edges -/
structure Graph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)

/-- The degree of a vertex in a graph -/
def degree (G : Graph) (v : Nat) : Nat :=
  (G.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- The set of vertices with odd degree in a graph -/
def oddDegreeVertices (G : Graph) : Finset Nat :=
  G.vertices.filter (λ v => degree G v % 2 = 1)

/-- The minimum number of pencil lifts required to draw a graph -/
def minPencilLifts (G : Graph) : Nat :=
  (oddDegreeVertices G).card / 2 + 1

theorem min_pencil_lifts_for_specific_graph (G : Graph) 
  (h1 : (oddDegreeVertices G).card = 10) : 
  minPencilLifts G = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pencil_lifts_for_specific_graph_l151_15112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_range_l151_15188

/-- The function f(x) = |e^x - a/e^x| is monotonically decreasing on [1, 2] -/
def is_monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f y ≤ f x

/-- The function f(x) = |e^x - a/e^x| -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |Real.exp x - a / Real.exp x|

/-- The theorem stating the range of a for which f is monotonically decreasing on [1, 2] -/
theorem f_monotone_decreasing_range (a : ℝ) :
  is_monotone_decreasing (f a) → a ≤ -Real.exp 4 ∨ a ≥ Real.exp 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_range_l151_15188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l151_15141

/-- The area of a square inscribed in the ellipse x^2/4 + y^2/8 = 1, 
    with sides parallel to the coordinate axes -/
theorem inscribed_square_area : 
  ∃ (s : ℝ), s > 0 ∧ 
  (∀ (x y : ℝ), x^2/4 + y^2/8 = 1 → |x| ≤ s ∧ |y| ≤ s) ∧
  (∃ (x y : ℝ), x^2/4 + y^2/8 = 1 ∧ |x| = s ∧ |y| = s) ∧
  s^2 = 32/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l151_15141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systems_equivalence_l151_15184

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- Define the two systems of equations
def system_I (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + y^2 = a ∧ x + y = 0

def system_II (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + y^2 = a ∧ frac (x + y) = 0

-- Theorem statement
theorem systems_equivalence (a : ℝ) :
  (system_I a ↔ system_II a) ↔ a < (1/2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_systems_equivalence_l151_15184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_range_l151_15150

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 ∧ x ≤ 4 then Real.log x
  else if x > 0 ∧ x < 1 then 3 * Real.log (1/x)
  else 0  -- Default value for other cases

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x - a * x

theorem three_zeros_range (a : ℝ) :
  (∃ x y z, x ∈ Set.Icc (1/4 : ℝ) 4 ∧ 
            y ∈ Set.Icc (1/4 : ℝ) 4 ∧ 
            z ∈ Set.Icc (1/4 : ℝ) 4 ∧ 
            x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
            g a x = 0 ∧ g a y = 0 ∧ g a z = 0) ↔
  (a ≥ Real.log 4 / 4 ∧ a < 1 / Real.exp 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_range_l151_15150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_gcd_lcm_pair_l151_15131

theorem unique_gcd_lcm_pair : ∃! (a b : ℕ), 
  (Nat.gcd a b) * (Nat.lcm a b) = 720 ∧ 
  a + b = 50 ∧
  Nat.gcd a b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_gcd_lcm_pair_l151_15131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_selection_theorem_l151_15130

-- Define the chessboard as a type
def Chessboard := Fin 8 → Fin 8 → Bool

-- Define a valid placement of pieces
def ValidPlacement (board : Chessboard) : Prop :=
  (∀ row, (Finset.filter (λ col ↦ board row col) Finset.univ).card = 4) ∧
  (∀ col, (Finset.filter (λ row ↦ board row col) Finset.univ).card = 4)

-- Define a selection of pieces
def Selection := Fin 8 → Fin 8

-- Define a valid selection
def ValidSelection (sel : Selection) : Prop :=
  Function.Injective sel

-- The main theorem
theorem chessboard_selection_theorem (board : Chessboard) (h : ValidPlacement board) :
  ∃ (sel : Selection), ValidSelection sel ∧ ∀ i, board i (sel i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_selection_theorem_l151_15130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l151_15164

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x) + 1

-- State the theorem
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x y : ℝ), π / 6 < x ∧ x < y ∧ y < π / 2 → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l151_15164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_net_square_area_l151_15154

/-- Represents the side length of a square in the unfolded net of a cube -/
def side_length (x : ℝ) : ℝ := x

/-- The area of a square given its side length -/
def square_area (s : ℝ) : ℝ := s^2

/-- The condition that two squares form part of the largest possible unfolded net of a cube from identical circular sheets -/
def is_largest_net (s1 s2 : ℝ) : Prop :=
  (4 * s1)^2 + s1^2 = (3 * s2)^2 + (4 * s2)^2

theorem cube_net_square_area :
  ∀ s1 s2 : ℝ,
  s1 = 10 →
  is_largest_net s1 s2 →
  square_area s2 = 68 := by
  sorry

#check cube_net_square_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_net_square_area_l151_15154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l151_15133

-- Define the function f(x) as noncomputable due to dependency on Real.pi
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi) * Real.cos (Real.pi - x)

-- State the theorem
theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T')) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/2), f x ≥ -Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/2), f x = -Real.sqrt 3 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l151_15133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_three_fruit_selection_l151_15103

theorem max_three_fruit_selection 
  (N : ℕ) (apple banana pear : Finset (Fin N)) 
  (h1 : (apple.card : ℝ) / N = 0.7)
  (h2 : (banana.card : ℝ) / N = 0.4)
  (h3 : (pear.card : ℝ) / N = 0.3)
  (h4 : ∀ s : Fin N, s ∈ apple ∪ banana ∪ pear) :
  ((apple ∩ banana ∩ pear).card : ℝ) / N ≤ 0.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_three_fruit_selection_l151_15103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x₂_value_l151_15163

open Real

-- Define the angle α
noncomputable def α : ℝ := 2 * π / 3

-- Define the rotation angle
noncomputable def rotation_angle : ℝ := π / 6

-- Define point A on the unit circle
noncomputable def A : ℝ × ℝ := (-1/2, Real.sqrt 3/2)

-- Define point B after rotation
noncomputable def B : ℝ × ℝ := (cos (α + rotation_angle), sin (α + rotation_angle))

-- Theorem statement
theorem x₂_value : B.1 = -Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x₂_value_l151_15163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_lollipops_per_friend_l151_15197

def total_lollipops : ℕ := 60
def num_friends : ℕ := 6

def cherry_percent : ℚ := 30 / 100
def watermelon_percent : ℚ := 20 / 100
def sour_apple_percent : ℚ := 15 / 100

def remaining_percent : ℚ := 1 - (cherry_percent + watermelon_percent + sour_apple_percent)
def grape_percent : ℚ := remaining_percent / 2

def grape_lollipops : ℕ := (grape_percent * total_lollipops).floor.toNat

theorem grape_lollipops_per_friend :
  grape_lollipops / num_friends = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_lollipops_per_friend_l151_15197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l151_15136

theorem angle_B_measure (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π/2 →  -- A is acute
  0 < B ∧ B < π/2 →  -- B is acute
  0 < C ∧ C < π/2 →  -- C is acute
  a = 2 * Real.sqrt 2 →
  b = 3 →
  Real.cos A = Real.sqrt 3 / 3 →
  Real.sin A = b * Real.sin B / a →  -- Law of sines
  B = π/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l151_15136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_geometric_a_geometric_implies_lambda_mu_a_is_arithmetic_l151_15170

/-- Definition of the sequence a_n and its sum S_n -/
def sequence_a (a : ℕ → ℝ) (S : ℕ → ℝ) (lambda mu : ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n ≥ 2 → S n = lambda * n * a n + mu * a (n - 1)

/-- Definition of the sequence b_n -/
def sequence_b (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = a (n + 1) - 2 * a n

/-- Theorem 1: b_n is geometric when lambda = 0 and mu = 4 -/
theorem b_is_geometric (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) :
  sequence_a a S 0 4 → sequence_b a b → ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n :=
by sorry

/-- Theorem 2: If a_n is geometric, then lambda = 1 and mu = 0 -/
theorem a_geometric_implies_lambda_mu (a : ℕ → ℝ) (S : ℕ → ℝ) (lambda mu : ℝ) :
  sequence_a a S lambda mu → (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) → lambda = 1 ∧ mu = 0 :=
by sorry

/-- Theorem 3: If a_2 = 3 and lambda + mu = 3/2, then a_n is arithmetic -/
theorem a_is_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) (lambda mu : ℝ) :
  sequence_a a S lambda mu → a 2 = 3 → lambda + mu = 3/2 → ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_geometric_a_geometric_implies_lambda_mu_a_is_arithmetic_l151_15170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_proof_l151_15124

noncomputable def triangle_area (base altitude : ℝ) : ℝ := (1 / 2) * base * altitude

theorem triangle_altitude_proof (area base : ℝ) (h1 : area = 1250) (h2 : base = 50) :
  ∃ altitude : ℝ, triangle_area base altitude = area ∧ altitude = 50 := by
  use 50
  constructor
  · rw [triangle_area, h1, h2]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_proof_l151_15124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_bounds_l151_15129

-- Define the complex number t
variable (t : ℂ)

-- Define z in terms of t
noncomputable def z (t : ℂ) : ℂ := t + 3 + 3 * Complex.I * Real.sqrt 3

-- Define the condition that (t + 3) / (t - 3) is a pure imaginary number
def is_pure_imaginary (w : ℂ) : Prop := w.re = 0 ∧ w.im ≠ 0

-- State the theorem
theorem trajectory_and_bounds :
  (is_pure_imaginary ((t + 3) / (t - 3))) →
  (∃ (x y : ℝ), t = Complex.mk x y ∧ x^2 + y^2 = 9 ∧ y ≠ 0) ∧
  (∀ t, Complex.abs (z t) ≤ 9) ∧
  (∀ t, Complex.abs (z t) ≥ 3) ∧
  (∃ (t₁ t₂ : ℂ), Complex.abs (z t₁) = 9 ∧ Complex.abs (z t₂) = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_bounds_l151_15129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_l151_15166

/-- Represents a digit in the string -/
def Digit := Fin 10

/-- Represents the string of 2023 digits -/
def DigitString := Fin 2023 → Digit

/-- Checks if a two-digit number is divisible by 17 or 29 -/
def isDivisibleBy17Or29 (n : ℕ) : Prop :=
  n % 17 = 0 ∨ n % 29 = 0

/-- The main theorem -/
theorem largest_last_digit
  (s : DigitString)
  (first_digit : s 0 = ⟨2, by norm_num⟩)
  (divisibility : ∀ i : Fin 2022, isDivisibleBy17Or29 ((s i).val * 10 + (s (i + 1)).val)) :
  (∃ (t : DigitString), 
    t 0 = ⟨2, by norm_num⟩ ∧ 
    (∀ i : Fin 2022, isDivisibleBy17Or29 ((t i).val * 10 + (t (i + 1)).val)) ∧
    (t 2022).val > (s 2022).val) → 
  (s 2022).val ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_l151_15166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithm_solution_l151_15177

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Cryptarithm solution type -/
structure CryptarithmSolution where
  T : Digit
  O : Digit
  K : Digit
  different : T ≠ O ∧ T ≠ K ∧ O ≠ K
  not_zero : T.val ≠ 0

/-- Converts a three-digit number represented by digits to a natural number -/
def toNat' (a b c : Digit) : ℕ := 100 * a.val + 10 * b.val + c.val

theorem cryptarithm_solution :
  ∃! (sol : CryptarithmSolution),
    toNat' sol.T sol.O sol.K = toNat' sol.K sol.O sol.T + toNat' sol.K sol.T sol.O ∧
    sol.T = ⟨9, by norm_num⟩ ∧ sol.O = ⟨5, by norm_num⟩ ∧ sol.K = ⟨4, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithm_solution_l151_15177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_meeting_theorem_l151_15149

/-- Represents the scenario of two cars traveling towards each other -/
structure CarScenario where
  distance : ℝ  -- Total distance between locations A and B
  speed_a : ℝ   -- Speed of car A
  speed_b : ℝ   -- Speed of car B

/-- Calculates the time for two cars to meet -/
noncomputable def time_to_meet (scenario : CarScenario) : ℝ :=
  scenario.distance / (scenario.speed_a + scenario.speed_b)

/-- Calculates the time for two cars to be a certain distance apart -/
noncomputable def time_to_distance (scenario : CarScenario) (target_distance : ℝ) : Set ℝ :=
  {t | t = (scenario.distance - target_distance) / (scenario.speed_a + scenario.speed_b) ∨
       t = (scenario.distance + target_distance) / (scenario.speed_a + scenario.speed_b)}

/-- The main theorem stating the results for the given scenario -/
theorem car_meeting_theorem (scenario : CarScenario) 
    (h1 : scenario.distance = 450)
    (h2 : scenario.speed_a = 115)
    (h3 : scenario.speed_b = 85) : 
    time_to_meet scenario = 2.25 ∧ 
    time_to_distance scenario 50 = {2, 2.5} := by
  sorry

-- Remove the #eval statements as they are causing issues with non-computable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_meeting_theorem_l151_15149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l151_15158

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + x - 1

-- State the theorem
theorem tangent_line_at_zero :
  let f' := fun x => Real.cos x + 1
  let tangent_slope := f' 0
  let tangent_point := (0, f 0)
  (fun x y => y = tangent_slope * x + (f 0 - tangent_slope * 0)) =
  (fun x y => y = 2 * x - 1) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l151_15158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combinations_with_repetition_at_least_once_l151_15155

def number_of_combinations_with_repetition_at_least_once (n r : ℕ) : ℕ :=
  Nat.choose (r - 1) (n - 1)

theorem combinations_with_repetition_at_least_once 
  (n r : ℕ) (h : r ≥ n) : 
  number_of_combinations_with_repetition_at_least_once n r = Nat.choose (r - 1) (n - 1) :=
by
  -- Unfold the definition of number_of_combinations_with_repetition_at_least_once
  unfold number_of_combinations_with_repetition_at_least_once
  -- The equality now holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combinations_with_repetition_at_least_once_l151_15155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_passes_through_one_one_l151_15160

-- Define a power function
noncomputable def power_function (n : ℝ) : ℝ → ℝ := fun x ↦ x ^ n

-- Theorem statement
theorem power_function_passes_through_one_one (n : ℝ) :
  power_function n 1 = 1 := by
  -- Unfold the definition of power_function
  unfold power_function
  -- Simplify the expression 1^n
  simp [Real.rpow_one]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_passes_through_one_one_l151_15160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_specific_value_l151_15153

noncomputable section

variable (f : ℚ → ℝ)

axiom functional_equation : ∀ x y : ℚ, f (x + y) = f x * f y - f (x * y) + 1

axiom distinct_values : f 1988 ≠ f 1987

theorem f_specific_value : f (-1987/1988) = 1/1988 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_specific_value_l151_15153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_l151_15110

theorem triangle_angle_sum (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c = 8) :
  ∃ (A B C : ℝ), 
    0 < A ∧ A < π ∧
    0 < B ∧ B < π ∧
    0 < C ∧ C < π ∧
    A + B + C = π ∧
    Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) ∧
    Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) ∧
    Real.cos C = (a^2 + b^2 - c^2) / (2*a*b) ∧
    max A (max B C) + min A (min B C) = 2*π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_l151_15110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_element_l151_15109

universe u

def K : Finset (Fin 5) := Finset.univ

def F : Finset (Finset (Fin 5)) :=
  sorry

axiom F_card : F.card = 16

axiom F_subset_powerset : ∀ S, S ∈ F → S ⊆ K

axiom F_distinct : ∀ S T, S ∈ F → T ∈ F → S = T → S = T

axiom F_intersection_nonempty :
  ∀ S T U, S ∈ F → T ∈ F → U ∈ F → (S ∩ T ∩ U).Nonempty

theorem common_element :
  ∃ x ∈ K, ∀ S ∈ F, x ∈ S :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_element_l151_15109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l151_15116

/-- The sum of the infinite series from n = 2 to infinity of (3n^3 - 2n^2 + 2n - 1) / (n^6 - n^4 + n^3 - n + 1) is equal to 3. -/
theorem infinite_series_sum : 
  (∑' n, (3 * n^3 - 2 * n^2 + 2 * n - 1) / (n^6 - n^4 + n^3 - n + 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l151_15116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_half_of_2017_l151_15100

noncomputable def f (x : ℝ) : ℝ := 2^x / (2^x + Real.sqrt 2)

theorem sum_of_f_equals_half_of_2017 :
  (Finset.range 2017).sum (fun i => f ((i + 1 : ℕ) / 2018)) = 2017 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_half_of_2017_l151_15100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_weight_lower_bound_l151_15132

/-- The minimum weight of a single crate in kg -/
def min_crate_weight : ℝ := 120

/-- The possible number of crates the trailer can carry on a trip -/
def possible_crate_counts : List ℕ := [3, 4, 5]

/-- The maximum number of crates the trailer can carry on a single trip -/
noncomputable def max_crate_count : ℕ := 
  (List.maximum possible_crate_counts).getD 0

theorem max_weight_lower_bound :
  ↑max_crate_count * min_crate_weight ≥ 600 := by
  -- Unfold the definition of max_crate_count
  unfold max_crate_count
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_weight_lower_bound_l151_15132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l151_15159

theorem trigonometric_identity (Y γ : Real) 
  (h : 4 * Real.tan Y ^ 2 + 4 * (1 / Real.tan Y) ^ 2 - 1 / Real.sin γ ^ 2 - 1 / Real.cos γ ^ 2 = 17) :
  Real.cos Y ^ 2 - Real.cos γ ^ 4 = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l151_15159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exotic_animal_park_l151_15179

theorem exotic_animal_park (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 200) (h2 : total_legs = 558) : ℕ := by
  let x := 94 -- number of two-legged birds
  let y := 54 -- number of three-legged creatures
  let z := 52 -- number of four-legged mammals

  have head_eq : x + y + z = total_heads := by
    rw [h1]
    norm_num

  have leg_eq : 2*x + 3*y + 4*z = total_legs := by
    rw [h2]
    norm_num

  exact x

#check exotic_animal_park

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exotic_animal_park_l151_15179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_same_combination_l151_15114

/-- Represents the number of candies of each color in the jar -/
def initial_candies : ℕ := 8

/-- Represents the total number of candies picked by each person -/
def candies_picked : ℕ := 3

/-- Represents a combination of candies picked -/
structure CandyCombination where
  red : ℕ
  blue : ℕ
  green : ℕ
  sum_eq_total : red + blue + green = candies_picked

/-- Calculates the probability of picking a specific combination of candies -/
noncomputable def probability_of_combination (jar : ℕ × ℕ × ℕ) (combo : CandyCombination) : ℚ :=
  sorry

/-- Calculates the probability of Sarah picking the same combination after Jared -/
noncomputable def probability_of_same_after (jar : ℕ × ℕ × ℕ) (combo : CandyCombination) : ℚ :=
  sorry

/-- List of all possible combinations of 3 candies -/
def all_combinations : List CandyCombination :=
  sorry

/-- The main theorem stating the probability of Jared and Sarah picking the same combination -/
theorem probability_same_combination :
  ∃ (p : ℚ),
    p = (all_combinations.map (λ combo =>
      probability_of_combination (initial_candies, initial_candies, initial_candies) combo *
      probability_of_same_after (initial_candies - combo.red, initial_candies - combo.blue, initial_candies - combo.green) combo
    )).sum :=
by
  sorry

#eval "This is a placeholder for the actual computation"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_same_combination_l151_15114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_complement_implies_m_range_l151_15171

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ m + 1}

-- Theorem 1
theorem intersection_implies_m_value (m : ℝ) : A ∩ B m = Set.Icc 1 3 → m = 2 := by sorry

-- Theorem 2
theorem subset_complement_implies_m_range (m : ℝ) : A ⊆ (B m)ᶜ → m > 4 ∨ m < -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_complement_implies_m_range_l151_15171
