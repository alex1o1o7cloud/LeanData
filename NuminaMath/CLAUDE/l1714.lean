import Mathlib

namespace NUMINAMATH_CALUDE_function_through_points_l1714_171406

/-- Given a function f(x) = a^x - k that passes through (1, 3) and (0, 2), 
    prove that f(x) = 2^x + 1 -/
theorem function_through_points 
  (f : ℝ → ℝ) 
  (a k : ℝ) 
  (h1 : ∀ x, f x = a^x - k) 
  (h2 : f 1 = 3) 
  (h3 : f 0 = 2) : 
  ∀ x, f x = 2^x + 1 := by
sorry

end NUMINAMATH_CALUDE_function_through_points_l1714_171406


namespace NUMINAMATH_CALUDE_tan_cos_eq_sin_minus_m_sin_l1714_171482

theorem tan_cos_eq_sin_minus_m_sin (m : ℝ) : 
  Real.tan (π / 12) * Real.cos (5 * π / 12) = Real.sin (5 * π / 12) - m * Real.sin (π / 12) → 
  m = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_cos_eq_sin_minus_m_sin_l1714_171482


namespace NUMINAMATH_CALUDE_sequence_problem_l1714_171467

-- Define arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom : is_geometric_sequence b)
  (h_eq : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
  (h_b7 : b 7 = a 7)
  (h_nonzero : ∀ n : ℕ, a n ≠ 0) :
  b 6 * b 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1714_171467


namespace NUMINAMATH_CALUDE_muffin_mix_buyers_l1714_171466

theorem muffin_mix_buyers (total_buyers : ℕ) (cake_mix_buyers : ℕ) (both_mix_buyers : ℕ) 
  (neither_mix_prob : ℚ) (h1 : total_buyers = 100) (h2 : cake_mix_buyers = 50) 
  (h3 : both_mix_buyers = 15) (h4 : neither_mix_prob = 1/4) : ℕ :=
by
  -- Proof goes here
  sorry

#check muffin_mix_buyers

end NUMINAMATH_CALUDE_muffin_mix_buyers_l1714_171466


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l1714_171435

def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ StrictMono f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l1714_171435


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1714_171420

theorem ratio_of_sum_to_difference (a b : ℝ) : 
  0 < b → b < a → a + b = 7 * (a - b) → a / b = 2 := by sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1714_171420


namespace NUMINAMATH_CALUDE_rectangle_C_in_position_I_l1714_171459

-- Define the Rectangle type
structure Rectangle where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the five rectangles
def A : Rectangle := ⟨1, 4, 3, 1⟩
def B : Rectangle := ⟨4, 3, 1, 2⟩
def C : Rectangle := ⟨1, 1, 2, 4⟩
def D : Rectangle := ⟨3, 2, 2, 3⟩
def E : Rectangle := ⟨4, 4, 1, 1⟩

-- Define a function to check if two rectangles can be placed side by side
def canPlaceSideBySide (r1 r2 : Rectangle) : Bool :=
  r1.right = r2.left

-- Define a function to check if two rectangles can be placed top to bottom
def canPlaceTopToBottom (r1 r2 : Rectangle) : Bool :=
  r1.bottom = r2.top

-- Theorem: Rectangle C is the only one that can be placed in position I
theorem rectangle_C_in_position_I :
  ∃! r : Rectangle, r = C ∧ 
  (∃ r2 r3 : Rectangle, r2 ≠ r ∧ r3 ≠ r ∧ r2 ≠ r3 ∧
   canPlaceSideBySide r r2 ∧ canPlaceSideBySide r2 r3 ∧
   (∃ r4 : Rectangle, r4 ≠ r ∧ r4 ≠ r2 ∧ r4 ≠ r3 ∧
    canPlaceTopToBottom r r4 ∧
    (∃ r5 : Rectangle, r5 ≠ r ∧ r5 ≠ r2 ∧ r5 ≠ r3 ∧ r5 ≠ r4 ∧
     canPlaceTopToBottom r2 r5 ∧ canPlaceSideBySide r4 r5))) :=
sorry

end NUMINAMATH_CALUDE_rectangle_C_in_position_I_l1714_171459


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1714_171432

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1714_171432


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_geq_five_l1714_171448

theorem inequality_holds_iff_p_geq_five (p : ℝ) :
  (∀ x : ℝ, x > 0 → Real.log (x + p) - (1/2 : ℝ) ≥ Real.log (Real.sqrt (2*x))) ↔ p ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_geq_five_l1714_171448


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l1714_171444

theorem min_value_quadratic_form (x y : ℝ) :
  x^2 + 2*x*y + 2*y^2 ≥ 0 ∧ (x^2 + 2*x*y + 2*y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l1714_171444


namespace NUMINAMATH_CALUDE_time_to_reach_destination_l1714_171402

/-- Calculates the time needed to reach a destination given initial movement and remaining distance -/
theorem time_to_reach_destination (initial_distance : ℝ) (initial_time : ℝ) (remaining_distance_yards : ℝ) : 
  initial_distance > 0 ∧ initial_time > 0 ∧ remaining_distance_yards > 0 →
  (remaining_distance_yards * 3) / (initial_distance / initial_time) = 75 :=
by
  sorry

#check time_to_reach_destination 80 20 100

end NUMINAMATH_CALUDE_time_to_reach_destination_l1714_171402


namespace NUMINAMATH_CALUDE_extreme_values_when_a_is_two_unique_zero_range_of_a_l1714_171461

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Theorem for part 1
theorem extreme_values_when_a_is_two :
  let f := f 2
  ∃ (x_max x_min : ℝ), 
    (∀ x, f x ≤ f x_max) ∧
    (∀ x, f x ≥ f x_min) ∧
    f x_max = 1 ∧
    f x_min = 0 :=
sorry

-- Theorem for part 2
theorem unique_zero_range_of_a :
  ∀ a : ℝ, 
    (∃! x : ℝ, x > 0 ∧ f a x = 0) ↔ 
    (a = 2 ∨ a ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_when_a_is_two_unique_zero_range_of_a_l1714_171461


namespace NUMINAMATH_CALUDE_problem_statement_l1714_171403

theorem problem_statement (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : (a * Real.sin (π/5) + b * Real.cos (π/5)) / (a * Real.cos (π/5) - b * Real.sin (π/5)) = Real.tan (8*π/15)) : 
  b / a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1714_171403


namespace NUMINAMATH_CALUDE_max_value_of_sum_l1714_171415

theorem max_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  (1 / (a + 9 * b) + 1 / (9 * a + b)) ≤ 5 / 24 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = 1 ∧
    1 / (a₀ + 9 * b₀) + 1 / (9 * a₀ + b₀) = 5 / 24 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l1714_171415


namespace NUMINAMATH_CALUDE_symmetry_y_axis_symmetry_x_axis_symmetry_origin_area_greater_than_pi_l1714_171438

-- Define the curve C
def C (x y : ℝ) : Prop := x^4 + y^2 = 1

-- Symmetry about y=0
theorem symmetry_y_axis (x y : ℝ) : C x y ↔ C x (-y) := by sorry

-- Symmetry about x=0
theorem symmetry_x_axis (x y : ℝ) : C x y ↔ C (-x) y := by sorry

-- Symmetry about (0,0)
theorem symmetry_origin (x y : ℝ) : C x y ↔ C (-x) (-y) := by sorry

-- Define the area of C
noncomputable def area_C : ℝ := sorry

-- Area of C is greater than π
theorem area_greater_than_pi : area_C > π := by sorry

end NUMINAMATH_CALUDE_symmetry_y_axis_symmetry_x_axis_symmetry_origin_area_greater_than_pi_l1714_171438


namespace NUMINAMATH_CALUDE_average_first_15_even_numbers_l1714_171457

theorem average_first_15_even_numbers : 
  let first_15_even : List ℕ := List.range 15 |>.map (fun n => 2 * (n + 1))
  (first_15_even.sum / first_15_even.length : ℚ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_first_15_even_numbers_l1714_171457


namespace NUMINAMATH_CALUDE_x_eq_3_sufficient_not_necessary_l1714_171409

-- Define vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2, x - 1)
def b (x : ℝ) : ℝ × ℝ := (x + 1, 4)

-- Define parallel condition for 2D vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem x_eq_3_sufficient_not_necessary :
  (∀ x : ℝ, x = 3 → parallel (a x) (b x)) ∧
  ¬(∀ x : ℝ, parallel (a x) (b x) → x = 3) :=
sorry

end NUMINAMATH_CALUDE_x_eq_3_sufficient_not_necessary_l1714_171409


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1714_171428

theorem linear_equation_solution :
  let x : ℝ := -4
  let y : ℝ := 2
  x + 3 * y = 2 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1714_171428


namespace NUMINAMATH_CALUDE_integral_roots_problem_l1714_171447

theorem integral_roots_problem (x y z : ℕ) : 
  z^x = y^(2*x) ∧ 
  2^z = 2*(8^x) ∧ 
  x + y + z = 20 →
  x = 5 ∧ y = 4 ∧ z = 16 := by
sorry

end NUMINAMATH_CALUDE_integral_roots_problem_l1714_171447


namespace NUMINAMATH_CALUDE_product_upper_bound_l1714_171469

theorem product_upper_bound (x : ℝ) (h : x ∈ Set.Icc 0 1) : x * (1 - x) ≤ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_upper_bound_l1714_171469


namespace NUMINAMATH_CALUDE_inverse_proportion_solution_l1714_171416

/-- Given two inversely proportional quantities, this is their constant product. -/
def InverseProportionConstant (x y : ℝ) : ℝ := x * y

theorem inverse_proportion_solution (a b c : ℝ → ℝ) :
  (∀ x y, InverseProportionConstant (a x) (b x) = InverseProportionConstant (a y) (b y)) →
  (∀ x y, InverseProportionConstant (b x) (c x) = InverseProportionConstant (b y) (c y)) →
  a 1 = 40 →
  b 1 = 5 →
  c 2 = 10 →
  b 2 = 7 →
  c 3 = 5.6 →
  a 3 = 16 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_solution_l1714_171416


namespace NUMINAMATH_CALUDE_no_real_solution_range_l1714_171473

theorem no_real_solution_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + x - a = 0) ↔ a < -1/4 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_range_l1714_171473


namespace NUMINAMATH_CALUDE_xyz_sum_l1714_171424

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x.val * y.val + z.val = y.val * z.val + x.val)
  (h2 : y.val * z.val + x.val = z.val * x.val + y.val)
  (h3 : z.val * x.val + y.val = x.val * y.val + z.val)
  (h4 : x.val * y.val + z.val = 56) : 
  x.val + y.val + z.val = 21 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l1714_171424


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1714_171493

open Real

theorem trigonometric_equation_solution :
  ∀ x : ℝ, 
    4 * sin x * cos (π/2 - x) + 4 * sin (π + x) * cos x + 2 * sin (3*π/2 - x) * cos (π + x) = 1 ↔ 
    (∃ k : ℤ, x = arctan (1/3) + π * k) ∨ (∃ n : ℤ, x = π/4 + π * n) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1714_171493


namespace NUMINAMATH_CALUDE_central_cell_is_two_l1714_171446

/-- Represents a 3x3 grid with numbers from 0 to 8 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two cells are neighbors -/
def is_neighbor (a b : Fin 3 × Fin 3) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

/-- Checks if the grid satisfies the consecutive number condition -/
def consecutive_condition (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, is_neighbor (i, j) (k, l) →
    (g i j = g k l + 1 ∨ g i j + 1 = g k l)

/-- Calculates the sum of corner cells -/
def corner_sum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- Theorem: In a valid 3x3 grid where the sum of corner cells is 18,
    the number in the central cell must be 2 -/
theorem central_cell_is_two (g : Grid) 
  (h1 : consecutive_condition g) 
  (h2 : corner_sum g = 18) : 
  g 1 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_is_two_l1714_171446


namespace NUMINAMATH_CALUDE_cubic_equation_mean_solution_l1714_171476

theorem cubic_equation_mean_solution : 
  let f : ℝ → ℝ := λ x ↦ x^3 + 4*x^2 - 5*x - 14
  let solutions := {x : ℝ | f x = 0}
  ∃ (s₁ s₂ s₃ : ℝ), solutions = {s₁, s₂, s₃} ∧ (s₁ + s₂ + s₃) / 3 = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_mean_solution_l1714_171476


namespace NUMINAMATH_CALUDE_orange_calories_distribution_l1714_171455

theorem orange_calories_distribution :
  let num_oranges : ℕ := 5
  let pieces_per_orange : ℕ := 8
  let num_people : ℕ := 4
  let calories_per_orange : ℕ := 80
  let total_pieces : ℕ := num_oranges * pieces_per_orange
  let pieces_per_person : ℕ := total_pieces / num_people
  let oranges_per_person : ℚ := pieces_per_person / pieces_per_orange
  let calories_per_person : ℚ := oranges_per_person * calories_per_orange
  calories_per_person = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_calories_distribution_l1714_171455


namespace NUMINAMATH_CALUDE_plan_comparison_l1714_171427

def suit_price : ℝ := 500
def tie_price : ℝ := 80
def num_suits : ℕ := 20

def plan1_cost (x : ℝ) : ℝ := 8400 + 80 * x
def plan2_cost (x : ℝ) : ℝ := 9000 + 72 * x

theorem plan_comparison (x : ℝ) (h : x > 20) :
  plan1_cost x ≤ plan2_cost x ↔ x ≤ 75 := by sorry

end NUMINAMATH_CALUDE_plan_comparison_l1714_171427


namespace NUMINAMATH_CALUDE_employee_reduction_l1714_171487

theorem employee_reduction (original : ℕ) : 
  let after_first := (9 : ℚ) / 10 * original
  let after_second := (19 : ℚ) / 20 * after_first
  let after_third := (22 : ℚ) / 25 * after_second
  after_third = 195 → original = 259 := by
sorry

end NUMINAMATH_CALUDE_employee_reduction_l1714_171487


namespace NUMINAMATH_CALUDE_ellie_and_hank_weight_l1714_171442

/-- The weights of Ellie, Frank, Gina, and Hank satisfy the given conditions
    and Ellie and Hank weigh 355 pounds together. -/
theorem ellie_and_hank_weight (e f g h : ℝ) 
    (ef_sum : e + f = 310)
    (fg_sum : f + g = 280)
    (gh_sum : g + h = 325)
    (g_minus_h : g = h + 10) :
  e + h = 355 := by
  sorry

end NUMINAMATH_CALUDE_ellie_and_hank_weight_l1714_171442


namespace NUMINAMATH_CALUDE_parabola_vertex_l1714_171492

/-- The vertex of the parabola y = -2x^2 + 3 is (0, 3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -2 * x^2 + 3 → (0, 3) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1714_171492


namespace NUMINAMATH_CALUDE_base_8_246_to_base_10_l1714_171419

def base_8_to_10 (a b c : ℕ) : ℕ := a * 8^2 + b * 8^1 + c * 8^0

theorem base_8_246_to_base_10 : base_8_to_10 2 4 6 = 166 := by
  sorry

end NUMINAMATH_CALUDE_base_8_246_to_base_10_l1714_171419


namespace NUMINAMATH_CALUDE_cricket_target_run_l1714_171489

/-- Given a cricket game with specific run rates, calculate the target run. -/
theorem cricket_target_run (total_overs : ℕ) (first_overs : ℕ) (remaining_overs : ℕ)
  (first_rate : ℝ) (remaining_rate : ℝ) :
  total_overs = first_overs + remaining_overs →
  first_overs = 10 →
  remaining_overs = 22 →
  first_rate = 3.2 →
  remaining_rate = 11.363636363636363 →
  ↑⌊(first_overs : ℝ) * first_rate + (remaining_overs : ℝ) * remaining_rate⌋ = 282 := by
  sorry

end NUMINAMATH_CALUDE_cricket_target_run_l1714_171489


namespace NUMINAMATH_CALUDE_fourth_guard_distance_l1714_171400

theorem fourth_guard_distance (l w : ℝ) (h1 : l = 300) (h2 : w = 200) : 
  let P := 2 * (l + w)
  let three_guards_distance := 850
  let fourth_guard_distance := P - three_guards_distance
  fourth_guard_distance = 150 := by
sorry

end NUMINAMATH_CALUDE_fourth_guard_distance_l1714_171400


namespace NUMINAMATH_CALUDE_bruce_shopping_result_l1714_171460

def bruce_shopping (initial_amount : ℚ) (shirt_price : ℚ) (shirt_count : ℕ) 
  (pants_price : ℚ) (sock_price : ℚ) (sock_count : ℕ) (belt_price : ℚ) 
  (belt_discount : ℚ) (total_discount : ℚ) : ℚ :=
  let shirt_total := shirt_price * shirt_count
  let sock_total := sock_price * sock_count
  let discounted_belt_price := belt_price * (1 - belt_discount)
  let subtotal := shirt_total + pants_price + sock_total + discounted_belt_price
  let final_total := subtotal * (1 - total_discount)
  initial_amount - final_total

theorem bruce_shopping_result : 
  bruce_shopping 71 5 5 26 3 2 12 0.25 0.1 = 11.6 := by
  sorry

end NUMINAMATH_CALUDE_bruce_shopping_result_l1714_171460


namespace NUMINAMATH_CALUDE_parabola_equation_l1714_171463

-- Define a parabola
def Parabola (a b c : ℝ) := {(x, y) : ℝ × ℝ | y = a * x^2 + b * x + c}

-- Define the properties of our specific parabola
def ParabolaProperties (p : Set (ℝ × ℝ)) :=
  ∃ a : ℝ, a ≠ 0 ∧ 
  p = Parabola 0 0 0 ∧  -- vertex at origin
  (∀ x y : ℝ, (x, y) ∈ p → (x, y) ∈ p) ∧  -- y-axis symmetry
  (-4, -2) ∈ p  -- passes through (-4, -2)

-- Theorem statement
theorem parabola_equation :
  ∃ p : Set (ℝ × ℝ), ParabolaProperties p ∧ p = {(x, y) : ℝ × ℝ | x^2 = -8*y} :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1714_171463


namespace NUMINAMATH_CALUDE_expression_value_l1714_171429

theorem expression_value (a b : ℝ) (h : 2 * a - 3 * b = 2) : 
  5 - 6 * a + 9 * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1714_171429


namespace NUMINAMATH_CALUDE_gumball_difference_l1714_171495

/-- The number of gumballs Carl bought -/
def carl_gumballs : ℕ := 16

/-- The number of gumballs Lewis bought -/
def lewis_gumballs : ℕ := 12

/-- The minimum average number of gumballs -/
def min_average : ℚ := 19

/-- The maximum average number of gumballs -/
def max_average : ℚ := 25

/-- The number of people who bought gumballs -/
def num_people : ℕ := 3

theorem gumball_difference :
  ∃ (min_x max_x : ℕ),
    (∀ x : ℕ, 
      (min_average ≤ (carl_gumballs + lewis_gumballs + x : ℚ) / num_people ∧
       (carl_gumballs + lewis_gumballs + x : ℚ) / num_people ≤ max_average) →
      min_x ≤ x ∧ x ≤ max_x) ∧
    max_x - min_x = 18 := by
  sorry

end NUMINAMATH_CALUDE_gumball_difference_l1714_171495


namespace NUMINAMATH_CALUDE_dance_attendance_l1714_171483

theorem dance_attendance (girls : ℕ) (boys : ℕ) : 
  boys = 2 * girls ∧ 
  boys = (girls - 1) + 8 → 
  boys = 14 := by
sorry

end NUMINAMATH_CALUDE_dance_attendance_l1714_171483


namespace NUMINAMATH_CALUDE_circle_area_l1714_171408

theorem circle_area (r : ℝ) (h : 3 / (2 * Real.pi * r) = r) : r ^ 2 * Real.pi = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l1714_171408


namespace NUMINAMATH_CALUDE_solution_system_l1714_171426

theorem solution_system (x y : ℝ) 
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := by
sorry

end NUMINAMATH_CALUDE_solution_system_l1714_171426


namespace NUMINAMATH_CALUDE_max_k_for_inequality_l1714_171425

theorem max_k_for_inequality : 
  (∃ k : ℤ, ∀ x y : ℝ, x > 0 → y > 0 → 4 * x^2 + 9 * y^2 ≥ 2^k * x * y) ∧ 
  (∀ k : ℤ, k > 3 → ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x^2 + 9 * y^2 < 2^k * x * y) :=
sorry

end NUMINAMATH_CALUDE_max_k_for_inequality_l1714_171425


namespace NUMINAMATH_CALUDE_y_increase_proof_l1714_171439

/-- Represents a line in the Cartesian plane -/
structure Line where
  slope : ℝ

/-- Calculates the change in y given a change in x for a line -/
def Line.deltaY (l : Line) (deltaX : ℝ) : ℝ :=
  l.slope * deltaX

theorem y_increase_proof (l : Line) (h : l.deltaY 4 = 6) :
  l.deltaY 12 = 18 := by
  sorry

end NUMINAMATH_CALUDE_y_increase_proof_l1714_171439


namespace NUMINAMATH_CALUDE_unique_solution_system_l1714_171474

/-- The system of equations has only one real solution (0, 0, 0, 0) -/
theorem unique_solution_system :
  ∃! (x y z w : ℝ),
    x = z + w + Real.sqrt (z * w * x) ∧
    y = w + x + Real.sqrt (w * x * y) ∧
    z = x + y + Real.sqrt (x * y * z) ∧
    w = y + z + Real.sqrt (y * z * w) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1714_171474


namespace NUMINAMATH_CALUDE_abc_xyz_inequality_l1714_171481

theorem abc_xyz_inequality (a b c x y z : ℝ) 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) :
  a*x + b*y + c*z ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_xyz_inequality_l1714_171481


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1714_171498

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}

-- Define set B
def B : Set ℝ := {y : ℝ | y ≥ 1/2}

-- State the theorem
theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {x : ℝ | -2 ≤ x ∧ x < 1/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1714_171498


namespace NUMINAMATH_CALUDE_homework_time_reduction_l1714_171458

theorem homework_time_reduction (x : ℝ) : 
  (∀ t₀ t₂ : ℝ, t₀ > 0 ∧ t₂ > 0 ∧ t₀ > t₂ →
    (∃ t₁ : ℝ, t₁ = t₀ * (1 - x) ∧ t₂ = t₁ * (1 - x)) ↔
    t₀ * (1 - x)^2 = t₂) →
  100 * (1 - x)^2 = 70 :=
by sorry

end NUMINAMATH_CALUDE_homework_time_reduction_l1714_171458


namespace NUMINAMATH_CALUDE_polygon_sides_l1714_171450

theorem polygon_sides (n : ℕ) (sum_angles : ℝ) : sum_angles = 1800 → (n - 2) * 180 = sum_angles → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1714_171450


namespace NUMINAMATH_CALUDE_smallest_n_terminating_with_3_l1714_171494

def is_terminating_decimal (n : ℕ+) : Prop :=
  ∃ (a b : ℕ), n = 2^a * 5^b

def contains_digit_3 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 3

theorem smallest_n_terminating_with_3 :
  ∀ n : ℕ+, n < 32 →
    ¬(is_terminating_decimal n ∧ contains_digit_3 n) ∧
    (is_terminating_decimal 32 ∧ contains_digit_3 32) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_terminating_with_3_l1714_171494


namespace NUMINAMATH_CALUDE_number_of_trays_l1714_171475

def cookies_per_tray : ℕ := 24
def number_of_packs : ℕ := 8
def cookies_per_pack : ℕ := 12

theorem number_of_trays : ℕ := by
  -- Prove that the number of trays is 4
  sorry

end NUMINAMATH_CALUDE_number_of_trays_l1714_171475


namespace NUMINAMATH_CALUDE_floor_tiles_theorem_l1714_171423

/-- Represents a square floor divided into four congruent sections -/
structure SquareFloor :=
  (section_side : ℕ)

/-- The number of tiles on the main diagonal of the entire floor -/
def main_diagonal_tiles (floor : SquareFloor) : ℕ :=
  4 * floor.section_side - 3

/-- The total number of tiles covering the entire floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  (4 * floor.section_side) ^ 2

/-- Theorem stating the relationship between the number of tiles on the main diagonal
    and the total number of tiles on the floor -/
theorem floor_tiles_theorem (floor : SquareFloor) 
  (h : main_diagonal_tiles floor = 75) : total_tiles floor = 25600 := by
  sorry

#check floor_tiles_theorem

end NUMINAMATH_CALUDE_floor_tiles_theorem_l1714_171423


namespace NUMINAMATH_CALUDE_bullet_speed_difference_l1714_171478

/-- The speed of the horse in feet per second -/
def horse_speed : ℝ := 20

/-- The speed of the bullet in feet per second -/
def bullet_speed : ℝ := 400

/-- The difference in bullet speed when fired in the same direction as the horse's movement
    versus the opposite direction -/
def speed_difference : ℝ := (bullet_speed + horse_speed) - (bullet_speed - horse_speed)

theorem bullet_speed_difference :
  speed_difference = 40 := by sorry

end NUMINAMATH_CALUDE_bullet_speed_difference_l1714_171478


namespace NUMINAMATH_CALUDE_total_spots_l1714_171441

/-- The number of spots on each dog -/
structure DogSpots where
  rover : ℕ
  cisco : ℕ
  granger : ℕ
  sparky : ℕ
  bella : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (d : DogSpots) : Prop :=
  d.rover = 46 ∧
  d.cisco = d.rover / 2 - 5 ∧
  d.granger = 5 * d.cisco ∧
  d.sparky = 3 * d.rover ∧
  d.bella = 2 * (d.granger + d.sparky)

/-- The theorem to be proven -/
theorem total_spots (d : DogSpots) (h : satisfiesConditions d) : 
  d.granger + d.cisco + d.sparky + d.bella = 702 := by
  sorry

end NUMINAMATH_CALUDE_total_spots_l1714_171441


namespace NUMINAMATH_CALUDE_ratio_problem_l1714_171480

theorem ratio_problem (a b x m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 4 / 5) 
  (h4 : x = a + 0.25 * a) (h5 : m = b - 0.2 * b) : m / x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1714_171480


namespace NUMINAMATH_CALUDE_circle_C_equation_line_l_equation_l1714_171470

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the line x - y + 1 = 0
def line_1 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the x-axis
def x_axis (y : ℝ) : Prop := y = 0

-- Define the line x + y + 3 = 0
def line_2 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the circle x^2 + (y - 3)^2 = 4
def circle_C2 (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x = -1 ∨ 4*x - 3*y + 4 = 0

-- Theorem 1
theorem circle_C_equation : 
  ∀ x y : ℝ, 
  (∃ x₀, line_1 x₀ 0 ∧ x_axis 0) → 
  (∀ x₁ y₁, line_2 x₁ y₁ → ∃ t, circle_C (x₁ + t) (y₁ + t) ∧ ¬(∃ s ≠ t, circle_C (x₁ + s) (y₁ + s))) →
  circle_C x y :=
sorry

-- Theorem 2
theorem line_l_equation :
  ∀ x y : ℝ,
  circle_C2 x y →
  (∃ x₀ y₀, x₀ = -1 ∧ y₀ = 0 ∧ line_l x₀ y₀) →
  (∃ p q : ℝ × ℝ, circle_C2 p.1 p.2 ∧ circle_C2 q.1 q.2 ∧ line_l p.1 p.2 ∧ line_l q.1 q.2 ∧ (p.1 - q.1)^2 + (p.2 - q.2)^2 = 12) →
  line_l x y :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_line_l_equation_l1714_171470


namespace NUMINAMATH_CALUDE_f_formula_g_minimum_l1714_171453

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 7*x + 13

-- Define the function g
def g (a x : ℝ) : ℝ := f (x + a) - 7*x

-- Theorem for part (I)
theorem f_formula (x : ℝ) : f (2*x - 3) = 4*x^2 + 2*x + 1 := by sorry

-- Theorem for part (II)
theorem g_minimum (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, g a x ≥ 
    if a ≤ -3 then a^2 + 13*a + 22
    else if a < -1 then 7*a + 13
    else a^2 + 9*a + 14) ∧
  (∃ x ∈ Set.Icc 1 3, g a x = 
    if a ≤ -3 then a^2 + 13*a + 22
    else if a < -1 then 7*a + 13
    else a^2 + 9*a + 14) := by sorry

end NUMINAMATH_CALUDE_f_formula_g_minimum_l1714_171453


namespace NUMINAMATH_CALUDE_square_sum_problem_l1714_171404

theorem square_sum_problem (square triangle : ℝ) 
  (h1 : 2 * square + 2 * triangle = 16)
  (h2 : 2 * square + 3 * triangle = 19) :
  4 * square = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_problem_l1714_171404


namespace NUMINAMATH_CALUDE_vector_inequalities_l1714_171414

theorem vector_inequalities (a b c m n p : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 1) 
  (h2 : m^2 + n^2 + p^2 = 1) : 
  (|a*m + b*n + c*p| ≤ 1) ∧ 
  (a*b*c ≠ 0 → m^4/a^2 + n^4/b^2 + p^4/c^2 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_inequalities_l1714_171414


namespace NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_75_sin_15_l1714_171488

theorem cos_75_cos_15_minus_sin_75_sin_15 :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) -
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_75_sin_15_l1714_171488


namespace NUMINAMATH_CALUDE_sum_first_15_odd_integers_l1714_171462

/-- The sum of the first n odd positive integers -/
def sumFirstNOddIntegers (n : ℕ) : ℕ :=
  n * n

theorem sum_first_15_odd_integers : sumFirstNOddIntegers 15 = 225 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_integers_l1714_171462


namespace NUMINAMATH_CALUDE_count_primes_with_digit_sum_10_l1714_171465

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 10

theorem count_primes_with_digit_sum_10 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_condition n) ∧ S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_primes_with_digit_sum_10_l1714_171465


namespace NUMINAMATH_CALUDE_plate_729_circulation_plate_363_circulation_plate_255_circulation_l1714_171436

def is_valid_plate (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 999

def monday_rule (n : ℕ) : Prop := n % 2 = 1

def tuesday_rule (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  (List.sum digits) ≥ 11

def wednesday_rule (n : ℕ) : Prop := n % 3 = 0

def thursday_rule (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  (List.sum digits) ≤ 14

def friday_rule (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  ∃ (i j : Fin 3), i ≠ j ∧ digits[i] = digits[j]

def saturday_rule (n : ℕ) : Prop := n < 500

def sunday_rule (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  ∀ d ∈ digits, d ≤ 5

theorem plate_729_circulation :
  is_valid_plate 729 ∧
  monday_rule 729 ∧
  tuesday_rule 729 ∧
  wednesday_rule 729 ∧
  ¬thursday_rule 729 ∧
  ¬friday_rule 729 ∧
  ¬saturday_rule 729 ∧
  ¬sunday_rule 729 := by sorry

theorem plate_363_circulation :
  is_valid_plate 363 ∧
  monday_rule 363 ∧
  tuesday_rule 363 ∧
  wednesday_rule 363 ∧
  thursday_rule 363 ∧
  friday_rule 363 ∧
  saturday_rule 363 ∧
  ¬sunday_rule 363 := by sorry

theorem plate_255_circulation :
  is_valid_plate 255 ∧
  monday_rule 255 ∧
  tuesday_rule 255 ∧
  wednesday_rule 255 ∧
  thursday_rule 255 ∧
  friday_rule 255 ∧
  saturday_rule 255 ∧
  sunday_rule 255 := by sorry

end NUMINAMATH_CALUDE_plate_729_circulation_plate_363_circulation_plate_255_circulation_l1714_171436


namespace NUMINAMATH_CALUDE_min_value_of_f_l1714_171434

theorem min_value_of_f (x : ℝ) (h : x ≥ 5/2) : (x^2 - 4*x + 5) / (2*x - 4) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1714_171434


namespace NUMINAMATH_CALUDE_unique_perpendicular_plane_l1714_171407

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Perpendicularity of a plane to a line -/
def isPerpendicular (p : Plane3D) (l : Line3D) : Prop :=
  -- Definition of perpendicularity
  sorry

/-- A plane contains a point -/
def planeContainsPoint (p : Plane3D) (pt : Point3D) : Prop :=
  -- Definition of a plane containing a point
  sorry

theorem unique_perpendicular_plane 
  (M : Point3D) (h : Line3D) : 
  ∃! (p : Plane3D), planeContainsPoint p M ∧ isPerpendicular p h :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_plane_l1714_171407


namespace NUMINAMATH_CALUDE_function_inequality_l1714_171433

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x ∈ Set.Ioo 0 (π/2), tan x * deriv f x < f x) : 
  f (π/6) * sin 1 > (1/2) * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1714_171433


namespace NUMINAMATH_CALUDE_door_can_be_opened_l1714_171430

/-- Represents a device with toggle switches and a display -/
structure Device where
  combinations : Fin 32 → ℕ

/-- Represents the notebook used for communication -/
structure Notebook where
  pages : Fin 1001 → Option (Fin 32)

/-- Represents the state of the operation -/
structure OperationState where
  deviceA : Device
  deviceB : Device
  notebook : Notebook
  time : ℕ

/-- Checks if a matching combination is found -/
def isMatchingCombinationFound (state : OperationState) : Prop :=
  ∃ (i : Fin 32), state.deviceA.combinations i = state.deviceB.combinations i

/-- Defines the time constraints of the operation -/
def isWithinTimeConstraint (state : OperationState) : Prop :=
  state.time ≤ 75

/-- Theorem stating that a matching combination can be found within the time constraint -/
theorem door_can_be_opened (initialState : OperationState) :
  ∃ (finalState : OperationState),
    isMatchingCombinationFound finalState ∧
    isWithinTimeConstraint finalState :=
  sorry


end NUMINAMATH_CALUDE_door_can_be_opened_l1714_171430


namespace NUMINAMATH_CALUDE_only_molality_can_be_calculated_l1714_171491

-- Define the given quantities
variable (mass_solute : ℝ)
variable (mass_solvent : ℝ)
variable (molar_mass_solute : ℝ)
variable (molar_mass_solvent : ℝ)

-- Define the quantitative descriptions
def can_calculate_molarity (mass_solute molar_mass_solute mass_solvent : ℝ) : Prop :=
  ∃ (volume_solution : ℝ), volume_solution > 0

def can_calculate_molality (mass_solute molar_mass_solute mass_solvent : ℝ) : Prop :=
  mass_solvent > 0 ∧ molar_mass_solute > 0

def can_calculate_density (mass_solute mass_solvent : ℝ) : Prop :=
  ∃ (volume_solution : ℝ), volume_solution > 0

-- Theorem statement
theorem only_molality_can_be_calculated
  (mass_solute mass_solvent molar_mass_solute molar_mass_solvent : ℝ) :
  can_calculate_molality mass_solute molar_mass_solute mass_solvent ∧
  ¬can_calculate_molarity mass_solute molar_mass_solute mass_solvent ∧
  ¬can_calculate_density mass_solute mass_solvent :=
sorry

end NUMINAMATH_CALUDE_only_molality_can_be_calculated_l1714_171491


namespace NUMINAMATH_CALUDE_eleventh_term_value_l1714_171477

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_term (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The 11th term of a geometric sequence with first term 5 and common ratio 2/3 -/
def eleventh_term : ℚ := geometric_term 5 (2/3) 11

theorem eleventh_term_value : eleventh_term = 5120/59049 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_value_l1714_171477


namespace NUMINAMATH_CALUDE_expression_evaluation_l1714_171464

theorem expression_evaluation :
  let x : ℝ := 2
  (x + 1) * (x - 1) + x * (3 - x) = 5 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1714_171464


namespace NUMINAMATH_CALUDE_closure_property_implies_divisibility_characterization_l1714_171440

theorem closure_property_implies_divisibility_characterization 
  (S : Set ℤ) 
  (closure : ∀ a b : ℤ, a ∈ S → b ∈ S → (a + b) ∈ S) 
  (has_negative : ∃ n : ℤ, n < 0 ∧ n ∈ S) 
  (has_positive : ∃ p : ℤ, p > 0 ∧ p ∈ S) : 
  ∃ d : ℤ, ∀ x : ℤ, x ∈ S ↔ d ∣ x := by
sorry

end NUMINAMATH_CALUDE_closure_property_implies_divisibility_characterization_l1714_171440


namespace NUMINAMATH_CALUDE_range_of_a_l1714_171410

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x, x^2 + (a-1)*x + 1 > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a-1)^x < (a-1)^y

-- Define the theorem
theorem range_of_a : 
  (∃ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a))) → 
  (∃ a : ℝ, (-1 < a ∧ a ≤ 2) ∨ a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1714_171410


namespace NUMINAMATH_CALUDE_binomial_505_505_equals_1_l1714_171479

theorem binomial_505_505_equals_1 : Nat.choose 505 505 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_505_505_equals_1_l1714_171479


namespace NUMINAMATH_CALUDE_problem_solution_l1714_171471

theorem problem_solution : 
  (Real.sqrt 6 + Real.sqrt 8 * Real.sqrt 12 = 5 * Real.sqrt 6) ∧ 
  (Real.sqrt 4 - Real.sqrt 2 / (Real.sqrt 2 + 1) = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1714_171471


namespace NUMINAMATH_CALUDE_david_average_marks_l1714_171468

def david_marks : List Nat := [96, 95, 82, 97, 95]

theorem david_average_marks :
  (david_marks.sum / david_marks.length : ℚ) = 93 := by sorry

end NUMINAMATH_CALUDE_david_average_marks_l1714_171468


namespace NUMINAMATH_CALUDE_hill_climbing_speed_l1714_171417

/-- Proves that given a round trip journey with specified conditions, 
    the average speed while going up is 2.25 km/h -/
theorem hill_climbing_speed 
  (time_up : ℝ) 
  (time_down : ℝ) 
  (avg_speed_total : ℝ) 
  (h1 : time_up = 4) 
  (h2 : time_down = 2) 
  (h3 : avg_speed_total = 3) : 
  (avg_speed_total * (time_up + time_down) / 2) / time_up = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_hill_climbing_speed_l1714_171417


namespace NUMINAMATH_CALUDE_nine_point_chords_l1714_171445

/-- The number of chords that can be drawn from n points on a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords from 9 points on a circle is 36 -/
theorem nine_point_chords : num_chords 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nine_point_chords_l1714_171445


namespace NUMINAMATH_CALUDE_equation_solution_l1714_171454

theorem equation_solution : ∃ x : ℚ, (x / (x + 1) = 1 + 1 / x) ∧ (x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1714_171454


namespace NUMINAMATH_CALUDE_constant_function_from_zero_derivative_l1714_171496

theorem constant_function_from_zero_derivative (f : ℝ → ℝ) (h : ∀ x, HasDerivAt f 0 x) :
  ∃ c, ∀ x, f x = c := by sorry

end NUMINAMATH_CALUDE_constant_function_from_zero_derivative_l1714_171496


namespace NUMINAMATH_CALUDE_binomial_plus_three_l1714_171472

theorem binomial_plus_three : (Nat.choose 13 11) + 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_binomial_plus_three_l1714_171472


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l1714_171413

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 2023) : (2023 - x)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l1714_171413


namespace NUMINAMATH_CALUDE_part1_part2_l1714_171405

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the set A
def A (a b c : ℝ) : Set ℝ := {x | f a b c x = x}

-- Part 1
theorem part1 (a b c : ℝ) :
  A a b c = {1, 2} → f a b c 0 = 2 →
  ∃ (M m : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x ≤ M ∧ m ≤ f a b c x) ∧ M = 10 ∧ m = 1 :=
sorry

-- Part 2
theorem part2 (a b c : ℝ) :
  A a b c = {2} → a ≥ 1 →
  ∃ (g : ℝ → ℝ), (∀ a' ≥ 1, g a' ≥ 63/4) ∧
    (∃ (M m : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x ≤ M ∧ m ≤ f a b c x) ∧ g a = M + m) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l1714_171405


namespace NUMINAMATH_CALUDE_white_washing_cost_l1714_171451

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

end NUMINAMATH_CALUDE_white_washing_cost_l1714_171451


namespace NUMINAMATH_CALUDE_not_less_than_negative_double_l1714_171418

theorem not_less_than_negative_double {x y : ℝ} (h : x < y) : ¬(-2 * x < -2 * y) := by
  sorry

end NUMINAMATH_CALUDE_not_less_than_negative_double_l1714_171418


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1714_171490

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 / (1 + i)) + i^3) = -3/2 :=
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1714_171490


namespace NUMINAMATH_CALUDE_grocery_store_inventory_l1714_171484

theorem grocery_store_inventory (regular_soda diet_soda apples : ℕ) : 
  regular_soda = 79 → 
  diet_soda = 53 → 
  regular_soda - diet_soda = 26 → 
  ¬∃ f : ℕ → ℕ → ℕ, f regular_soda diet_soda = apples :=
by sorry

end NUMINAMATH_CALUDE_grocery_store_inventory_l1714_171484


namespace NUMINAMATH_CALUDE_beach_problem_l1714_171486

/-- The number of people originally in the first row at the beach -/
def first_row : ℕ := by sorry

/-- The number of people originally in the second row at the beach -/
def second_row : ℕ := 20

/-- The number of people in the third row at the beach -/
def third_row : ℕ := 18

/-- The number of people who left the first row -/
def left_first : ℕ := 3

/-- The number of people who left the second row -/
def left_second : ℕ := 5

/-- The total number of people left relaxing on the beach -/
def total_left : ℕ := 54

theorem beach_problem : 
  first_row = 24 :=
by
  have h1 : first_row - left_first + (second_row - left_second) + third_row = total_left := by sorry
  sorry

end NUMINAMATH_CALUDE_beach_problem_l1714_171486


namespace NUMINAMATH_CALUDE_redo_profit_is_5000_l1714_171449

/-- Calculates the profit for Redo's horseshoe manufacturing --/
def calculate_profit (initial_outlay : ℕ) (cost_per_set : ℕ) (selling_price : ℕ) (num_sets : ℕ) : ℤ :=
  let revenue := num_sets * selling_price
  let manufacturing_costs := initial_outlay + (cost_per_set * num_sets)
  (revenue : ℤ) - manufacturing_costs

/-- Proves that the profit for Redo's horseshoe manufacturing is $5,000 --/
theorem redo_profit_is_5000 :
  calculate_profit 10000 20 50 500 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_redo_profit_is_5000_l1714_171449


namespace NUMINAMATH_CALUDE_orange_water_concentration_decrease_l1714_171485

theorem orange_water_concentration_decrease 
  (initial_weight : ℝ) 
  (initial_water_percentage : ℝ) 
  (final_weight : ℝ) : 
  initial_weight = 5 →
  initial_water_percentage = 0.95 →
  final_weight = 25 →
  let initial_water := initial_weight * initial_water_percentage
  let initial_dry := initial_weight * (1 - initial_water_percentage)
  let final_water_percentage := (final_weight - initial_dry) / final_weight
  final_water_percentage - initial_water_percentage = -0.04 :=
by sorry

end NUMINAMATH_CALUDE_orange_water_concentration_decrease_l1714_171485


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1714_171437

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  property : a 2 + 4 * a 7 + a 12 = 96

/-- Theorem stating the relationship in the arithmetic sequence -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) : 
  2 * seq.a 3 + seq.a 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1714_171437


namespace NUMINAMATH_CALUDE_gcd_459_357_l1714_171499

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l1714_171499


namespace NUMINAMATH_CALUDE_at_least_one_positive_l1714_171421

theorem at_least_one_positive (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (x : ℝ) (hx : x = a^2 - b*c)
  (y : ℝ) (hy : y = b^2 - c*a)
  (z : ℝ) (hz : z = c^2 - a*b) :
  x > 0 ∨ y > 0 ∨ z > 0 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_positive_l1714_171421


namespace NUMINAMATH_CALUDE_nine_nine_nine_squared_plus_nine_nine_nine_l1714_171431

theorem nine_nine_nine_squared_plus_nine_nine_nine (n : ℕ) : 999 * 999 + 999 = 999000 := by
  sorry

end NUMINAMATH_CALUDE_nine_nine_nine_squared_plus_nine_nine_nine_l1714_171431


namespace NUMINAMATH_CALUDE_lisa_income_percentage_l1714_171443

-- Define variables for incomes
variable (T : ℝ) -- Tim's income
variable (M : ℝ) -- Mary's income
variable (J : ℝ) -- Juan's income
variable (L : ℝ) -- Lisa's income

-- Define the conditions
variable (h1 : M = 1.60 * T) -- Mary's income is 60% more than Tim's
variable (h2 : T = 0.50 * J) -- Tim's income is 50% less than Juan's
variable (h3 : L = 1.30 * M) -- Lisa's income is 30% more than Mary's
variable (h4 : L = 0.75 * J) -- Lisa's income is 25% less than Juan's

-- Define the theorem
theorem lisa_income_percentage :
  (L / (M + J)) * 100 = 41.67 :=
sorry

end NUMINAMATH_CALUDE_lisa_income_percentage_l1714_171443


namespace NUMINAMATH_CALUDE_die_throw_probability_l1714_171452

/-- Represents a fair six-sided die throw -/
def DieFace := Fin 6

/-- The probability of a specific outcome when throwing a fair die three times -/
def prob_single_outcome : ℚ := 1 / 216

/-- Checks if a + bi is a root of x^2 - 2x + c = 0 -/
def is_root (a b c : ℕ) : Prop :=
  a = 1 ∧ c = b^2 + 1

/-- The number of favorable outcomes -/
def favorable_outcomes : ℕ := 2

theorem die_throw_probability :
  (favorable_outcomes : ℚ) * prob_single_outcome = 1 / 108 := by
  sorry


end NUMINAMATH_CALUDE_die_throw_probability_l1714_171452


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l1714_171422

/-- Represents a repeating decimal with a repeating part of length 2 -/
def RepeatingDecimal2 (a b : ℕ) : ℚ :=
  (a * 10 + b : ℚ) / 99

/-- Represents a repeating decimal with a repeating part of length 1 -/
def RepeatingDecimal1 (a : ℕ) : ℚ :=
  (a : ℚ) / 9

/-- The product of 0.overline{03} and 0.overline{3} is equal to 1/99 -/
theorem product_of_repeating_decimals :
  (RepeatingDecimal2 0 3) * (RepeatingDecimal1 3) = 1 / 99 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l1714_171422


namespace NUMINAMATH_CALUDE_jungkook_money_l1714_171412

def initial_amount (notebook_cost pencil_cost remaining : ℕ) : Prop :=
  ∃ (total : ℕ),
    notebook_cost = total / 2 ∧
    pencil_cost = (total - notebook_cost) / 2 ∧
    remaining = total - notebook_cost - pencil_cost ∧
    remaining = 750

theorem jungkook_money : 
  ∀ (notebook_cost pencil_cost remaining : ℕ),
    initial_amount notebook_cost pencil_cost remaining →
    ∃ (total : ℕ), total = 3000 :=
by
  sorry

end NUMINAMATH_CALUDE_jungkook_money_l1714_171412


namespace NUMINAMATH_CALUDE_rectangle_y_value_l1714_171401

/-- Given a rectangle with vertices (-3, y), (1, y), (1, -2), and (-3, -2),
    if the area of the rectangle is 12, then y = 1. -/
theorem rectangle_y_value (y : ℝ) : 
  let vertices := [(-3, y), (1, y), (1, -2), (-3, -2)]
  let length := 1 - (-3)
  let height := y - (-2)
  let area := length * height
  area = 12 → y = 1 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l1714_171401


namespace NUMINAMATH_CALUDE_school_travel_time_difference_l1714_171497

/-- The problem of calculating the late arrival time of a boy traveling to school. -/
theorem school_travel_time_difference (distance : ℝ) (speed1 speed2 : ℝ) (early_time : ℝ) : 
  distance = 2.5 →
  speed1 = 5 →
  speed2 = 10 →
  early_time = 8 / 60 →
  (distance / speed1 - (distance / speed2 + early_time)) * 60 = 7 := by
  sorry

end NUMINAMATH_CALUDE_school_travel_time_difference_l1714_171497


namespace NUMINAMATH_CALUDE_milk_mixture_problem_l1714_171456

/-- Proves that the butterfat percentage of the added milk is 10% given the conditions of the problem -/
theorem milk_mixture_problem (final_percentage : ℝ) (initial_volume : ℝ) (initial_percentage : ℝ) (added_volume : ℝ) :
  final_percentage = 20 →
  initial_volume = 8 →
  initial_percentage = 30 →
  added_volume = 8 →
  (initial_volume * initial_percentage + added_volume * (100 * final_percentage - initial_volume * initial_percentage) / added_volume) / (initial_volume + added_volume) = 10 := by
sorry

end NUMINAMATH_CALUDE_milk_mixture_problem_l1714_171456


namespace NUMINAMATH_CALUDE_min_value_problem_l1714_171411

/-- The problem statement -/
theorem min_value_problem (m n : ℝ) : 
  (∃ (x y : ℝ), x + y - 1 = 0 ∧ 3 * x - y - 7 = 0 ∧ m * x + y + n = 0) →
  (m * n > 0) →
  (∀ k : ℝ, (1 / m + 2 / n ≥ k) → k ≤ 8) ∧ 
  (∃ m₀ n₀ : ℝ, (∃ (x y : ℝ), x + y - 1 = 0 ∧ 3 * x - y - 7 = 0 ∧ m₀ * x + y + n₀ = 0) ∧ 
                (m₀ * n₀ > 0) ∧ 
                (1 / m₀ + 2 / n₀ = 8)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1714_171411
