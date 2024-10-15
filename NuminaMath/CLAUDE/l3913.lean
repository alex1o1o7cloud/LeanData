import Mathlib

namespace NUMINAMATH_CALUDE_eulersRelationHoldsForNewPolyhedron_l3913_391369

/-- Represents a polyhedron formed from a cube by marking midpoints of edges,
    connecting them on each face, and cutting off 8 pyramids around each vertex. -/
structure NewPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's Relation for convex polyhedra -/
def eulersRelation (p : NewPolyhedron) : Prop :=
  p.vertices + p.faces = p.edges + 2

/-- The specific new polyhedron formed from the cube -/
def cubeTransformedPolyhedron : NewPolyhedron :=
  { vertices := 12
  , edges := 24
  , faces := 14 }

/-- Theorem stating that Euler's Relation holds for the new polyhedron -/
theorem eulersRelationHoldsForNewPolyhedron :
  eulersRelation cubeTransformedPolyhedron := by
  sorry

end NUMINAMATH_CALUDE_eulersRelationHoldsForNewPolyhedron_l3913_391369


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_is_negative_two_l3913_391303

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Theorem statement
theorem cube_root_of_negative_eight_is_negative_two :
  cubeRoot (-8) = -2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_is_negative_two_l3913_391303


namespace NUMINAMATH_CALUDE_remaining_distance_l3913_391306

theorem remaining_distance (total : ℕ) (monday : ℕ) (tuesday : ℕ) 
  (h1 : total = 8205)
  (h2 : monday = 907)
  (h3 : tuesday = 582) :
  total - (monday + tuesday) = 6716 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l3913_391306


namespace NUMINAMATH_CALUDE_marble_bag_problem_l3913_391304

theorem marble_bag_problem :
  ∀ (r b : ℕ),
  r + b > 0 →
  r = (r + b) / 3 →
  (r - 3) = (r + b - 3) / 4 →
  r = (r + b - 2) / 3 →
  r + b = 19 := by
sorry

end NUMINAMATH_CALUDE_marble_bag_problem_l3913_391304


namespace NUMINAMATH_CALUDE_haley_magazine_boxes_l3913_391332

theorem haley_magazine_boxes (total_magazines : ℕ) (magazines_per_box : ℕ) (h1 : total_magazines = 63) (h2 : magazines_per_box = 9) :
  total_magazines / magazines_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_haley_magazine_boxes_l3913_391332


namespace NUMINAMATH_CALUDE_largest_tray_size_l3913_391394

theorem largest_tray_size (tim_sweets peter_sweets : ℕ) 
  (h1 : tim_sweets = 36) 
  (h2 : peter_sweets = 44) : 
  Nat.gcd tim_sweets peter_sweets = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_tray_size_l3913_391394


namespace NUMINAMATH_CALUDE_average_income_is_400_l3913_391359

def daily_incomes : List ℝ := [300, 150, 750, 200, 600]

theorem average_income_is_400 : 
  (daily_incomes.sum / daily_incomes.length : ℝ) = 400 := by
  sorry

end NUMINAMATH_CALUDE_average_income_is_400_l3913_391359


namespace NUMINAMATH_CALUDE_same_color_left_neighbor_l3913_391338

/-- The number of children in the circle. -/
def total_children : ℕ := 150

/-- The number of children in blue jackets who have a left neighbor in a red jacket. -/
def blue_with_red_left : ℕ := 12

/-- Theorem stating the number of children with a left neighbor wearing a jacket of the same color. -/
theorem same_color_left_neighbor :
  total_children - 2 * blue_with_red_left = 126 := by
  sorry

end NUMINAMATH_CALUDE_same_color_left_neighbor_l3913_391338


namespace NUMINAMATH_CALUDE_crystal_sales_revenue_l3913_391318

def original_cupcake_price : ℚ := 3
def original_cookie_price : ℚ := 2
def discount_factor : ℚ := 1/2
def cupcakes_sold : ℕ := 16
def cookies_sold : ℕ := 8

theorem crystal_sales_revenue : 
  (original_cupcake_price * discount_factor * cupcakes_sold) + 
  (original_cookie_price * discount_factor * cookies_sold) = 32 := by
sorry

end NUMINAMATH_CALUDE_crystal_sales_revenue_l3913_391318


namespace NUMINAMATH_CALUDE_max_abs_ab_l3913_391317

/-- The quadratic function f(x) -/
def f (a b c x : ℝ) : ℝ := a * (3 * a + 2 * c) * x^2 - 2 * b * (2 * a + c) * x + b^2 + (c + a)^2

/-- Theorem stating the maximum value of |ab| given the conditions -/
theorem max_abs_ab (a b c : ℝ) (h : ∀ x : ℝ, f a b c x ≤ 1) : 
  |a * b| ≤ 3 * Real.sqrt 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_ab_l3913_391317


namespace NUMINAMATH_CALUDE_find_B_l3913_391308

theorem find_B (x y A : ℕ) (hx : x > 1) (hy : y > 1) (hxy : x > y) 
  (heq : x * y = x + y + A) : x / y = 12 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l3913_391308


namespace NUMINAMATH_CALUDE_candy_mixture_cost_per_pound_l3913_391326

theorem candy_mixture_cost_per_pound 
  (weight_expensive : ℝ) 
  (price_expensive : ℝ) 
  (weight_cheap : ℝ) 
  (price_cheap : ℝ) 
  (h1 : weight_expensive = 30)
  (h2 : price_expensive = 8)
  (h3 : weight_cheap = 60)
  (h4 : price_cheap = 5) : 
  (weight_expensive * price_expensive + weight_cheap * price_cheap) / (weight_expensive + weight_cheap) = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_per_pound_l3913_391326


namespace NUMINAMATH_CALUDE_inequality_proof_l3913_391320

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c ≥ 1) :
  1 / (2 + a) + 1 / (2 + b) + 1 / (2 + c) ≤ 1 ∧
  (1 / (2 + a) + 1 / (2 + b) + 1 / (2 + c) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3913_391320


namespace NUMINAMATH_CALUDE_product_of_numbers_l3913_391336

theorem product_of_numbers (x y : ℝ) 
  (sum_of_squares : x^2 + y^2 = 289) 
  (sum_of_numbers : x + y = 23) : 
  x * y = 120 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3913_391336


namespace NUMINAMATH_CALUDE_smallest_s_for_F_l3913_391388

def F (a b c d : ℕ) : ℕ := a * b^(c^d)

theorem smallest_s_for_F : 
  (∀ s : ℕ, s > 0 ∧ s < 9 → F s s 2 2 < 65536) ∧ 
  F 9 9 2 2 = 65536 := by
  sorry

end NUMINAMATH_CALUDE_smallest_s_for_F_l3913_391388


namespace NUMINAMATH_CALUDE_ratio_squares_l3913_391396

theorem ratio_squares (a b c d : ℤ) 
  (h1 : b * c + a * d = 1)
  (h2 : a * c + 2 * b * d = 1) :
  (a^2 + c^2 : ℚ) / (b^2 + d^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_ratio_squares_l3913_391396


namespace NUMINAMATH_CALUDE_power_function_domain_and_odd_l3913_391386

def A : Set ℝ := {-1, 1, 1/2, 3}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem power_function_domain_and_odd (a : ℝ) :
  a ∈ A →
  (Set.univ = {x : ℝ | ∃ y, y = x^a} ∧ is_odd_function (λ x => x^a)) ↔
  (a = 1 ∨ a = 3) :=
sorry

end NUMINAMATH_CALUDE_power_function_domain_and_odd_l3913_391386


namespace NUMINAMATH_CALUDE_distribute_six_among_three_l3913_391375

/-- The number of ways to distribute n positions among k schools -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n positions among k schools,
    where each school receives at least one position -/
def distributeAtLeastOne (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n positions among k schools,
    where each school receives at least one position and
    the number of positions for each school is distinct -/
def distributeDistinct (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to distribute 6 positions among 3 schools,
    where each school receives at least one position and
    the number of positions for each school is distinct -/
theorem distribute_six_among_three : distributeDistinct 6 3 = 6 := by sorry

end NUMINAMATH_CALUDE_distribute_six_among_three_l3913_391375


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3913_391313

theorem imaginary_part_of_z (z : ℂ) (m : ℝ) : 
  z = 1 - m * I → z^2 = -2 * I → z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3913_391313


namespace NUMINAMATH_CALUDE_lcm_gcd_difference_even_nonnegative_l3913_391323

theorem lcm_gcd_difference_even_nonnegative (a b : ℕ+) :
  let n := Nat.lcm a b + Nat.gcd a b - a - b
  n % 2 = 0 ∧ n ≥ 0 := by sorry

end NUMINAMATH_CALUDE_lcm_gcd_difference_even_nonnegative_l3913_391323


namespace NUMINAMATH_CALUDE_no_two_cubes_between_squares_l3913_391367

theorem no_two_cubes_between_squares : ¬∃ (a b n : ℕ+), n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_two_cubes_between_squares_l3913_391367


namespace NUMINAMATH_CALUDE_f_fixed_points_l3913_391387

def f (x : ℝ) : ℝ := x^2 - 3*x

theorem f_fixed_points : 
  ∀ x : ℝ, f (f x) = f x ↔ x = 0 ∨ x = 3 ∨ x = -1 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_fixed_points_l3913_391387


namespace NUMINAMATH_CALUDE_solve_inequality_range_of_a_l3913_391352

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2 * x - a| + |2 * x + 3|
def g (x : ℝ) : ℝ := |2 * x - 3| + 2

-- Statement for part (i)
theorem solve_inequality (x : ℝ) : |g x| < 5 ↔ 0 < x ∧ x < 3 := by sorry

-- Statement for part (ii)
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) ↔ a ≤ -5 ∨ a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_range_of_a_l3913_391352


namespace NUMINAMATH_CALUDE_segment_lengths_l3913_391370

-- Define the points on the number line
def A : ℝ := 2
def B : ℝ := -5
def C : ℝ := -2
def D : ℝ := 4

-- Define the length of a segment
def segmentLength (x y : ℝ) : ℝ := |y - x|

-- Theorem statement
theorem segment_lengths :
  segmentLength A B = 7 ∧ segmentLength C D = 6 := by
  sorry

end NUMINAMATH_CALUDE_segment_lengths_l3913_391370


namespace NUMINAMATH_CALUDE_range_of_sum_l3913_391360

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 3 then |Real.log x / Real.log 3|
  else if x ≥ 3 then 1/3 * x^2 - 10/3 * x + 8
  else 0  -- Define for all reals, though we only care about positive x

theorem range_of_sum (a b c d : ℝ) :
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧
  f a = f b ∧ f b = f c ∧ f c = f d →
  ∃ (x : ℝ), x ∈ Set.Icc (10 + 2 * Real.sqrt 2) (41/3) ∧
             x = 2*a + b + c + d :=
sorry

end NUMINAMATH_CALUDE_range_of_sum_l3913_391360


namespace NUMINAMATH_CALUDE_total_cost_of_purchase_leas_purchase_l3913_391365

/-- The total cost of Léa's purchases is $28 given the prices and quantities of items she bought. -/
theorem total_cost_of_purchase : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun book_price binder_price notebook_price book_quantity binder_quantity notebook_quantity =>
    book_price * book_quantity +
    binder_price * binder_quantity +
    notebook_price * notebook_quantity = 28

/-- Léa's actual purchase -/
theorem leas_purchase : total_cost_of_purchase 16 2 1 1 3 6 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_purchase_leas_purchase_l3913_391365


namespace NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_34_seconds_l3913_391363

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger (jogger_speed : Real) (train_speed : Real) 
  (initial_distance : Real) (train_length : Real) : Real :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- Proof that the train passes the jogger in 34 seconds under given conditions -/
theorem train_passes_jogger_in_34_seconds :
  train_passing_jogger 9 45 240 100 = 34 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_34_seconds_l3913_391363


namespace NUMINAMATH_CALUDE_darks_wash_time_l3913_391378

/-- Represents the time for washing and drying clothes -/
structure LaundryTime where
  whites_wash : ℕ
  whites_dry : ℕ
  darks_dry : ℕ
  colors_wash : ℕ
  colors_dry : ℕ
  total_time : ℕ

/-- Theorem stating the time for washing darks -/
theorem darks_wash_time (lt : LaundryTime) 
  (h1 : lt.whites_wash = 72)
  (h2 : lt.whites_dry = 50)
  (h3 : lt.darks_dry = 65)
  (h4 : lt.colors_wash = 45)
  (h5 : lt.colors_dry = 54)
  (h6 : lt.total_time = 344) :
  ∃ (darks_wash : ℕ), 
    lt.whites_wash + darks_wash + lt.colors_wash + 
    lt.whites_dry + lt.darks_dry + lt.colors_dry = lt.total_time ∧ 
    darks_wash = 58 := by
  sorry

end NUMINAMATH_CALUDE_darks_wash_time_l3913_391378


namespace NUMINAMATH_CALUDE_partnership_profit_l3913_391322

/-- 
Given a business partnership between Mary and Mike:
- Mary invests $700
- Mike invests $300
- 1/3 of profit is divided equally for efforts
- 2/3 of profit is divided in ratio of investments (7:3)
- Mary received $800 more than Mike

This theorem proves that the total profit P satisfies the equation:
[(P/6) + (7/10) * (2P/3)] - [(P/6) + (3/10) * (2P/3)] = 800
-/
theorem partnership_profit (P : ℝ) : 
  (P / 6 + 7 / 10 * (2 * P / 3)) - (P / 6 + 3 / 10 * (2 * P / 3)) = 800 → 
  P = 3000 := by
sorry


end NUMINAMATH_CALUDE_partnership_profit_l3913_391322


namespace NUMINAMATH_CALUDE_only_fourth_statement_correct_l3913_391382

/-- Represents a programming statement --/
inductive Statement
| Input (s : String)
| Output (s : String)

/-- Checks if an input statement is syntactically correct --/
def isValidInputStatement (s : Statement) : Prop :=
  match s with
  | Statement.Input str => ∃ (vars : List String), str = s!"INPUT {String.intercalate ", " vars}"
  | _ => False

/-- Checks if an output statement is syntactically correct --/
def isValidOutputStatement (s : Statement) : Prop :=
  match s with
  | Statement.Output str => ∃ (expr : String), str = s!"PRINT {expr}"
  | _ => False

/-- The given statements --/
def statements : List Statement :=
  [Statement.Input "a; b; c",
   Statement.Input "x=3",
   Statement.Output "\"A=4\"",
   Statement.Output "3*2"]

/-- Theorem stating that only the fourth statement is correct --/
theorem only_fourth_statement_correct :
  ∃! (i : Fin 4), isValidInputStatement (statements[i.val]) ∨ 
                  isValidOutputStatement (statements[i.val]) :=
by sorry

end NUMINAMATH_CALUDE_only_fourth_statement_correct_l3913_391382


namespace NUMINAMATH_CALUDE_union_A_complement_B_l3913_391351

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1, 2}
def B : Set Int := {-1, 0, 3}

theorem union_A_complement_B : A ∪ (U \ B) = {-2, -1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_l3913_391351


namespace NUMINAMATH_CALUDE_closest_point_parabola_to_line_l3913_391334

/-- The point on the parabola y = x^2 that is closest to the line 2x - y = 4 is (1, 1) -/
theorem closest_point_parabola_to_line :
  let parabola := λ x : ℝ => (x, x^2)
  let line := {p : ℝ × ℝ | 2 * p.1 - p.2 = 4}
  let distance := λ p : ℝ × ℝ => |2 * p.1 - p.2 - 4| / Real.sqrt 5
  ∀ x : ℝ, distance (parabola x) ≥ distance (parabola 1) :=
by sorry

end NUMINAMATH_CALUDE_closest_point_parabola_to_line_l3913_391334


namespace NUMINAMATH_CALUDE_valid_array_iff_even_l3913_391329

/-- Represents an n × n array with entries -1, 0, or 1 -/
def ValidArray (n : ℕ) := Matrix (Fin n) (Fin n) (Fin 3)

/-- Checks if all 2n sums of rows and columns are different -/
def HasAllDifferentSums (A : ValidArray n) : Prop := sorry

/-- The main theorem: such an array exists if and only if n is even -/
theorem valid_array_iff_even (n : ℕ) :
  (∃ A : ValidArray n, HasAllDifferentSums A) ↔ Even n := by sorry

end NUMINAMATH_CALUDE_valid_array_iff_even_l3913_391329


namespace NUMINAMATH_CALUDE_problem_solution_l3913_391397

theorem problem_solution :
  -- 1. Contrapositive statement
  (¬ (∀ a b : ℝ, (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ 
    (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0))) ∧
  
  -- 2. Sufficient but not necessary condition
  ((∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧ 
   (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1)) ∧
  
  -- 3. False conjunction does not imply both propositions are false
  (¬ (∀ P Q : Prop, ¬(P ∧ Q) → (¬P ∧ ¬Q))) ∧
  
  -- 4. Correct negation of existential statement
  (¬ (∀ x : ℝ, ¬(x^2 + x + 1 < 0)) ↔ 
    (∃ x : ℝ, x^2 + x + 1 ≥ 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3913_391397


namespace NUMINAMATH_CALUDE_weight_of_new_person_l3913_391328

theorem weight_of_new_person (n : ℕ) (initial_weight replaced_weight new_average_increase : ℝ) :
  n = 8 →
  replaced_weight = 65 →
  new_average_increase = 5 →
  (n * new_average_increase + replaced_weight) = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l3913_391328


namespace NUMINAMATH_CALUDE_polynomial_expansion_sum_l3913_391377

theorem polynomial_expansion_sum (m : ℝ) (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + m * x)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 63 →
  m = 1 ∨ m = -3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_sum_l3913_391377


namespace NUMINAMATH_CALUDE_solution_set_l3913_391389

theorem solution_set (x y z : Real) : 
  x + y + z = Real.pi ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
  ((x = Real.pi/6 ∧ y = Real.pi/3 ∧ z = Real.pi/2) ∨
   (x = Real.pi ∧ y = 0 ∧ z = 0) ∨
   (x = 0 ∧ y = Real.pi ∧ z = 0) ∨
   (x = 0 ∧ y = 0 ∧ z = Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l3913_391389


namespace NUMINAMATH_CALUDE_sin_product_equals_two_fifths_l3913_391391

theorem sin_product_equals_two_fifths (α : Real) (h : Real.tan α = 2) :
  Real.sin α * Real.sin (π / 2 - α) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_two_fifths_l3913_391391


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l3913_391390

-- Define the parabola and hyperbola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def hyperbola (x y : ℝ) : Prop := x^2/7 - y^2/9 = 1

-- Define the focus of the hyperbola
def hyperbola_focus : ℝ × ℝ := (4, 0)

-- Define the parameter p of the parabola
def p : ℝ := 8

-- Define point K
def K : ℝ × ℝ := (-4, 0)

-- Define the relationship between |AK| and |AF|
def AK_AF_relation (A F K : ℝ × ℝ) : Prop :=
  (A.1 - K.1)^2 + (A.2 - K.2)^2 = 2 * ((A.1 - F.1)^2 + (A.2 - F.2)^2)

theorem parabola_triangle_area 
  (A : ℝ × ℝ)
  (h1 : parabola p A.1 A.2)
  (h2 : AK_AF_relation A hyperbola_focus K) :
  (1/2) * ((K.1 - hyperbola_focus.1)^2 + (K.2 - hyperbola_focus.2)^2) = 32 :=
sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l3913_391390


namespace NUMINAMATH_CALUDE_tangent_angle_sum_l3913_391346

-- Define the basic structures
structure Point :=
  (x y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define the problem setup
def circumcircle (t : Triangle) : Circle := sorry

def is_acute_angled (t : Triangle) : Prop := sorry

def is_tangent_to_side (c : Circle) (p1 p2 : Point) : Prop := sorry

def angle_between_tangents (c : Circle) (p : Point) : ℝ := sorry

-- Theorem statement
theorem tangent_angle_sum 
  (t : Triangle)
  (O : Point)
  (S_A S_B S_C : Circle) :
  is_acute_angled t →
  O = (circumcircle t).center →
  S_A.center = O ∧ S_B.center = O ∧ S_C.center = O →
  is_tangent_to_side S_A t.B t.C →
  is_tangent_to_side S_B t.C t.A →
  is_tangent_to_side S_C t.A t.B →
  angle_between_tangents S_A t.A + 
  angle_between_tangents S_B t.B + 
  angle_between_tangents S_C t.C = 180 :=
by sorry

end NUMINAMATH_CALUDE_tangent_angle_sum_l3913_391346


namespace NUMINAMATH_CALUDE_absolute_value_sum_equality_l3913_391335

theorem absolute_value_sum_equality (a b c d : ℝ) 
  (h1 : |a - b| + |c - d| = 99)
  (h2 : |a - c| + |b - d| = 1) :
  |a - d| + |b - c| = 99 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_sum_equality_l3913_391335


namespace NUMINAMATH_CALUDE_factorization_sum_l3913_391300

theorem factorization_sum (a b c d e f g h j k : ℤ) :
  (∀ x y : ℝ, 8 * x^4 - 125 * y^4 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + j*x*y + k*y^2)) →
  a + b + c + d + e + f + g + h + j + k = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l3913_391300


namespace NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l3913_391373

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop := n ≥ 10000000 ∧ n < 100000000

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k : Nat, n / (10^k) % 10 = d

theorem largest_eight_digit_with_even_digits :
  (∀ m : Nat, is_eight_digit m ∧ contains_all_even_digits m → m ≤ 99986420) ∧
  is_eight_digit 99986420 ∧
  contains_all_even_digits 99986420 :=
sorry

end NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l3913_391373


namespace NUMINAMATH_CALUDE_fraction_equality_l3913_391384

theorem fraction_equality : (45 : ℚ) / (7 - 3 / 4) = 36 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3913_391384


namespace NUMINAMATH_CALUDE_third_roots_unity_quadratic_roots_l3913_391341

theorem third_roots_unity_quadratic_roots :
  ∃ (S : Finset ℂ), 
    (∀ z ∈ S, z^3 = 1) ∧ 
    (∀ z ∈ S, ∃ a : ℤ, z^2 + a*z - 1 = 0) ∧
    S.card = 3 ∧
    (∀ z : ℂ, z^3 = 1 → (∃ a : ℤ, z^2 + a*z - 1 = 0) → z ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_third_roots_unity_quadratic_roots_l3913_391341


namespace NUMINAMATH_CALUDE_distance_to_concert_l3913_391337

/-- The distance to a concert given the distance driven before stopping for gas and the remaining distance after getting gas -/
theorem distance_to_concert 
  (distance_before_gas : ℕ) 
  (distance_after_gas : ℕ) 
  (h1 : distance_before_gas = 32)
  (h2 : distance_after_gas = 46) :
  distance_before_gas + distance_after_gas = 78 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_concert_l3913_391337


namespace NUMINAMATH_CALUDE_percentage_problem_l3913_391301

theorem percentage_problem : 
  ∃ x : ℝ, (120 / 100) * x = 1800 → (20 / 100) * x = 300 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3913_391301


namespace NUMINAMATH_CALUDE_innovative_numbers_l3913_391344

def is_innovative (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = a^2 - b^2

theorem innovative_numbers :
  is_innovative 31 ∧ is_innovative 41 ∧ is_innovative 16 ∧ ¬is_innovative 54 :=
by sorry

end NUMINAMATH_CALUDE_innovative_numbers_l3913_391344


namespace NUMINAMATH_CALUDE_zebra_crossing_distance_l3913_391309

/-- Given a boulevard with zebra crossing, calculate the distance between stripes --/
theorem zebra_crossing_distance (boulevard_width : ℝ) (stripe_length : ℝ) (gate_distance : ℝ)
  (h1 : boulevard_width = 60)
  (h2 : stripe_length = 65)
  (h3 : gate_distance = 22) :
  (boulevard_width * gate_distance) / stripe_length = 20.31 := by
  sorry

end NUMINAMATH_CALUDE_zebra_crossing_distance_l3913_391309


namespace NUMINAMATH_CALUDE_system_solution_l3913_391340

theorem system_solution (a b : ℚ) 
  (eq1 : 2 * a + 3 * b = 18)
  (eq2 : 4 * a + 5 * b = 31) :
  2 * a + b = 8 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3913_391340


namespace NUMINAMATH_CALUDE_circleplus_composition_l3913_391357

-- Define the ⊕ operation
def circleplus (y : Int) : Int := 9 - y

-- Define the prefix ⊕ operation
def prefix_circleplus (y : Int) : Int := y - 9

-- Theorem to prove
theorem circleplus_composition : prefix_circleplus (circleplus 18) = -18 := by
  sorry

end NUMINAMATH_CALUDE_circleplus_composition_l3913_391357


namespace NUMINAMATH_CALUDE_power_division_19_l3913_391392

theorem power_division_19 : (19 : ℕ)^12 / (19 : ℕ)^5 = 893871739 := by sorry

end NUMINAMATH_CALUDE_power_division_19_l3913_391392


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3913_391380

theorem possible_values_of_a (a : ℝ) : 
  2 ∈ ({1, a^2 - 3*a + 2, a + 1} : Set ℝ) → a = 3 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3913_391380


namespace NUMINAMATH_CALUDE_cubic_equation_transformation_and_solutions_l3913_391316

theorem cubic_equation_transformation_and_solutions 
  (p q : ℝ) 
  (hp : p < 0) 
  (hpq : 4 * p^3 + 27 * q^2 ≤ 0) : 
  ∃ (k r : ℝ), 
    k = 2 * Real.sqrt (-p/3) ∧ 
    r = q / (2 * Real.sqrt (-p^3/27)) ∧
    (∀ x, x^3 + p*x + q = 0 ↔ (∃ t, x = k*t ∧ 4*t^3 - 3*t - r = 0)) ∧
    (∃ φ, r = Real.cos φ ∧
      (∀ t, 4*t^3 - 3*t - r = 0 ↔ 
        t = Real.cos (φ/3) ∨ 
        t = Real.cos ((φ + 2*Real.pi)/3) ∨ 
        t = Real.cos ((φ + 4*Real.pi)/3))) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_transformation_and_solutions_l3913_391316


namespace NUMINAMATH_CALUDE_calculation_proof_l3913_391366

theorem calculation_proof : 6 * (-1/2) + Real.sqrt 3 * Real.sqrt 8 + (-15)^0 = 2 * Real.sqrt 6 - 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3913_391366


namespace NUMINAMATH_CALUDE_metal_sheet_weight_l3913_391361

/-- Represents a square piece of metal sheet -/
structure MetalSquare where
  side : ℝ
  weight : ℝ

/-- Given conditions of the problem -/
def problem_conditions (s1 s2 : MetalSquare) : Prop :=
  s1.side = 4 ∧ s1.weight = 16 ∧ s2.side = 6

/-- Theorem statement -/
theorem metal_sheet_weight (s1 s2 : MetalSquare) :
  problem_conditions s1 s2 → s2.weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_metal_sheet_weight_l3913_391361


namespace NUMINAMATH_CALUDE_smallest_n_for_solutions_greater_than_negative_one_l3913_391399

theorem smallest_n_for_solutions_greater_than_negative_one :
  ∀ (n : ℤ), (∀ (x : ℝ), 
    x^3 - (5*n - 9)*x^2 + (6*n^2 - 31*n - 106)*x - 6*(n - 8)*(n + 2) = 0 
    → x > -1) 
  ↔ n ≥ 8 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_solutions_greater_than_negative_one_l3913_391399


namespace NUMINAMATH_CALUDE_tracy_initial_balloons_l3913_391381

theorem tracy_initial_balloons : 
  ∀ T : ℕ,
  (T + 24) / 2 + 20 = 35 →
  T = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_tracy_initial_balloons_l3913_391381


namespace NUMINAMATH_CALUDE_function_inequality_l3913_391355

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - a * x + (a + 3) / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2 * Real.exp x - 4 * x + 2 * a

theorem function_inequality (a : ℝ) (h₁ : a ≥ 0) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (1/2 : ℝ) 2 ∧ x₂ ∈ Set.Icc (1/2 : ℝ) 2 ∧ f a x₁ > g a x₂) →
  a ∈ Set.Icc 1 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3913_391355


namespace NUMINAMATH_CALUDE_percentage_error_calculation_l3913_391349

theorem percentage_error_calculation (x y : ℝ) (hx : x ≠ 0) (hy : y + 15 ≠ 0) :
  let error_x := ((x * 10 - x / 10) / (x * 10)) * 100
  let error_y := (30 / (y + 15)) * 100
  let total_error := ((10 * x - x / 10 + 30) / (10 * x + y + 15)) * 100
  (error_x = 99) ∧
  (error_y = (30 / (y + 15)) * 100) ∧
  (total_error = ((10 * x - x / 10 + 30) / (10 * x + y + 15)) * 100) := by
sorry

end NUMINAMATH_CALUDE_percentage_error_calculation_l3913_391349


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l3913_391319

theorem least_number_divisible_by_five_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → 
    ¬(∃ (q₁ q₂ q₃ q₄ q₅ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0)) ∧
  n = 2310 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l3913_391319


namespace NUMINAMATH_CALUDE_product_of_roots_rational_l3913_391321

-- Define the polynomial
def polynomial (a b c d e : ℤ) (z : ℂ) : ℂ :=
  a * z^4 + b * z^3 + c * z^2 + d * z + e

-- Define the theorem
theorem product_of_roots_rational
  (a b c d e : ℤ)
  (r₁ r₂ r₃ r₄ : ℂ)
  (h₁ : a ≠ 0)
  (h₂ : polynomial a b c d e r₁ = 0)
  (h₃ : polynomial a b c d e r₂ = 0)
  (h₄ : polynomial a b c d e r₃ = 0)
  (h₅ : polynomial a b c d e r₄ = 0)
  (h₆ : ∃ q : ℚ, r₁ + r₂ = q)
  (h₇ : r₃ + r₄ ≠ r₁ + r₂)
  : ∃ q : ℚ, r₁ * r₂ = q :=
sorry

end NUMINAMATH_CALUDE_product_of_roots_rational_l3913_391321


namespace NUMINAMATH_CALUDE_intersection_count_is_two_l3913_391307

/-- The number of intersection points between two circles -/
def intersection_count (c1 c2 : ℝ × ℝ → Prop) : ℕ :=
  sorry

/-- First circle: (x - 2.5)² + y² = 6.25 -/
def circle1 (p : ℝ × ℝ) : Prop :=
  (p.1 - 2.5)^2 + p.2^2 = 6.25

/-- Second circle: x² + (y - 5)² = 25 -/
def circle2 (p : ℝ × ℝ) : Prop :=
  p.1^2 + (p.2 - 5)^2 = 25

/-- Theorem stating that the number of intersection points between the two circles is 2 -/
theorem intersection_count_is_two :
  intersection_count circle1 circle2 = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_count_is_two_l3913_391307


namespace NUMINAMATH_CALUDE_volume_cube_with_pyramids_l3913_391327

/-- The volume of a solid formed by a cube with pyramids on each face --/
theorem volume_cube_with_pyramids (a : ℝ) (a_pos : 0 < a) :
  let cube_volume := a^3
  let sphere_radius := a / 2 * Real.sqrt 3
  let pyramid_height := a / 2 * (Real.sqrt 3 - 1)
  let pyramid_volume := 1 / 3 * a^2 * pyramid_height
  let total_pyramid_volume := 6 * pyramid_volume
  cube_volume + total_pyramid_volume = a^3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_volume_cube_with_pyramids_l3913_391327


namespace NUMINAMATH_CALUDE_florist_roses_l3913_391345

theorem florist_roses (initial : ℕ) (sold : ℕ) (picked : ℕ) : 
  initial = 37 → sold = 16 → picked = 19 → initial - sold + picked = 40 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l3913_391345


namespace NUMINAMATH_CALUDE_conditional_probability_l3913_391314

open Set
open Finset

def Ω : Finset ℕ := {1,2,3,4,5,6}
def A : Finset ℕ := {2,3,5}
def B : Finset ℕ := {1,2,4,5,6}

def P (X : Finset ℕ) : ℚ := (X.card : ℚ) / (Ω.card : ℚ)

theorem conditional_probability : 
  P (A ∩ B) / P B = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_l3913_391314


namespace NUMINAMATH_CALUDE_ninth_minus_eighth_difference_l3913_391343

/-- The side length of the nth square in the sequence -/
def side_length (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- The number of tiles in the nth square -/
def tile_count (n : ℕ) : ℕ := (side_length n) ^ 2

theorem ninth_minus_eighth_difference : tile_count 9 - tile_count 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ninth_minus_eighth_difference_l3913_391343


namespace NUMINAMATH_CALUDE_triangular_sum_iff_squares_sum_l3913_391312

/-- A triangular number is a positive integer of the form n * (n + 1) / 2 -/
def IsTriangular (k : ℕ) : Prop :=
  ∃ n : ℕ, k = n * (n + 1) / 2

/-- m is a sum of two triangular numbers -/
def IsSumOfTwoTriangular (m : ℕ) : Prop :=
  ∃ a b : ℕ, IsTriangular a ∧ IsTriangular b ∧ m = a + b

/-- n is a sum of two squares -/
def IsSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ x y : ℤ, n = x^2 + y^2

/-- Main theorem: m is a sum of two triangular numbers if and only if 4m + 1 is a sum of two squares -/
theorem triangular_sum_iff_squares_sum (m : ℕ) :
  IsSumOfTwoTriangular m ↔ IsSumOfTwoSquares (4 * m + 1) :=
sorry

end NUMINAMATH_CALUDE_triangular_sum_iff_squares_sum_l3913_391312


namespace NUMINAMATH_CALUDE_perimeter_is_18_l3913_391383

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the arrangement of rectangles -/
structure Arrangement where
  base : Rectangle
  middle : Rectangle
  top : Rectangle

/-- Calculates the perimeter of the arrangement -/
def perimeter (arr : Arrangement) : ℕ :=
  2 * (arr.base.width + arr.base.height + arr.middle.height + arr.top.height)

/-- The theorem stating that the perimeter of the specific arrangement is 18 -/
theorem perimeter_is_18 : 
  let r := Rectangle.mk 2 1
  let arr := Arrangement.mk (Rectangle.mk 4 2) (Rectangle.mk 4 2) r
  perimeter arr = 18 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_18_l3913_391383


namespace NUMINAMATH_CALUDE_at_most_one_integer_root_l3913_391398

theorem at_most_one_integer_root (n : ℤ) :
  ∃! (k : ℤ), k^4 - 1993*k^3 + (1993 + n)*k^2 - 11*k + n = 0 :=
by sorry

end NUMINAMATH_CALUDE_at_most_one_integer_root_l3913_391398


namespace NUMINAMATH_CALUDE_total_money_l3913_391376

/-- Given three people A, B, and C with some money between them, 
    prove that their total amount is 450 under certain conditions. -/
theorem total_money (a b c : ℕ) : 
  a + c = 200 → b + c = 350 → c = 100 → a + b + c = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l3913_391376


namespace NUMINAMATH_CALUDE_power_of_two_divisibility_l3913_391325

theorem power_of_two_divisibility (n : ℕ) : n ≥ 1 → (
  (∃ m : ℕ, m ≥ 1 ∧ (2^n - 1) ∣ (m^2 + 9)) ↔ 
  ∃ k : ℕ, n = 2^k
) := by sorry

end NUMINAMATH_CALUDE_power_of_two_divisibility_l3913_391325


namespace NUMINAMATH_CALUDE_circle_area_reduction_l3913_391385

theorem circle_area_reduction (r : ℝ) (h : r > 0) :
  let new_r := 0.9 * r
  let original_area := π * r^2
  let new_area := π * new_r^2
  new_area = 0.81 * original_area := by
  sorry

end NUMINAMATH_CALUDE_circle_area_reduction_l3913_391385


namespace NUMINAMATH_CALUDE_tangency_point_unique_l3913_391315

/-- The point of tangency between two parabolas -/
def point_of_tangency : ℝ × ℝ := (-9.5, -31.5)

/-- First parabola equation -/
def parabola1 (x y : ℝ) : Prop := y = x^2 + 20*x + 72

/-- Second parabola equation -/
def parabola2 (x y : ℝ) : Prop := x = y^2 + 64*y + 992

theorem tangency_point_unique :
  ∀ x y : ℝ, parabola1 x y ∧ parabola2 x y → (x, y) = point_of_tangency :=
sorry

end NUMINAMATH_CALUDE_tangency_point_unique_l3913_391315


namespace NUMINAMATH_CALUDE_negation_equivalence_quadratic_no_solution_sufficient_not_necessary_l3913_391364

-- Statement 1
theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 2 = 0) ↔ (∀ x : ℝ, x^2 - x - 2 ≠ 0) := by sorry

-- Statement 2
theorem quadratic_no_solution (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 4*x + m = 0) → m > 4 := by sorry

-- Statement 3
theorem sufficient_not_necessary :
  (∀ a : ℝ, a > 1 → 1/a < 1) ∧ (∃ a : ℝ, 1/a < 1 ∧ ¬(a > 1)) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_quadratic_no_solution_sufficient_not_necessary_l3913_391364


namespace NUMINAMATH_CALUDE_unique_number_with_remainders_l3913_391348

theorem unique_number_with_remainders : ∃! n : ℕ,
  35 < n ∧ n < 70 ∧
  n % 6 = 3 ∧
  n % 7 = 1 ∧
  n % 8 = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_number_with_remainders_l3913_391348


namespace NUMINAMATH_CALUDE_book_store_inventory_l3913_391330

theorem book_store_inventory (initial : ℝ) (first_addition : ℝ) (second_addition : ℝ) 
  (h1 : initial = 41.0)
  (h2 : first_addition = 33.0)
  (h3 : second_addition = 2.0) :
  initial + first_addition + second_addition = 76.0 := by
  sorry

end NUMINAMATH_CALUDE_book_store_inventory_l3913_391330


namespace NUMINAMATH_CALUDE_find_x_l3913_391362

theorem find_x : ∃ (x : ℕ+), 
  let n : ℤ := (x : ℤ)^2 + 3*(x : ℤ) + 20
  let d : ℤ := 3*(x : ℤ) + 4
  n = d * (x : ℤ) + 8 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_find_x_l3913_391362


namespace NUMINAMATH_CALUDE_one_quadrilateral_is_rhombus_l3913_391305

-- Define the properties of quadrilaterals
structure QuadrilateralProperties where
  opposite_sides_equal : Bool
  opposite_sides_parallel : Bool
  adjacent_sides_equal : Bool
  diagonals_perpendicular_and_bisected : Bool

-- Define a function to check if a quadrilateral with given properties is a rhombus
def is_rhombus (props : QuadrilateralProperties) : Bool :=
  (props.opposite_sides_equal && props.adjacent_sides_equal) ||
  (props.adjacent_sides_equal && props.diagonals_perpendicular_and_bisected) ||
  (props.opposite_sides_equal && props.diagonals_perpendicular_and_bisected)

-- Theorem statement
theorem one_quadrilateral_is_rhombus 
  (quad1 props1 quad2 props2 : QuadrilateralProperties) 
  (h1 : props1.opposite_sides_equal + props1.opposite_sides_parallel + 
        props1.adjacent_sides_equal + props1.diagonals_perpendicular_and_bisected = 2)
  (h2 : props2.opposite_sides_equal + props2.opposite_sides_parallel + 
        props2.adjacent_sides_equal + props2.diagonals_perpendicular_and_bisected = 2)
  (h3 : props1.opposite_sides_equal + props2.opposite_sides_equal = 1)
  (h4 : props1.opposite_sides_parallel + props2.opposite_sides_parallel = 1)
  (h5 : props1.adjacent_sides_equal + props2.adjacent_sides_equal = 1)
  (h6 : props1.diagonals_perpendicular_and_bisected + props2.diagonals_perpendicular_and_bisected = 1) :
  is_rhombus props1 ∨ is_rhombus props2 :=
sorry

end NUMINAMATH_CALUDE_one_quadrilateral_is_rhombus_l3913_391305


namespace NUMINAMATH_CALUDE_coupon_usage_day_l3913_391379

theorem coupon_usage_day (coupon_count : Nat) (interval : Nat) : 
  coupon_count = 6 →
  interval = 10 →
  (∀ i : Fin coupon_count, ((i.val * interval) % 7 ≠ 0)) →
  (0 * interval) % 7 = 3 :=
by sorry

end NUMINAMATH_CALUDE_coupon_usage_day_l3913_391379


namespace NUMINAMATH_CALUDE_min_sum_given_product_l3913_391354

theorem min_sum_given_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b = a*b) :
  a + b ≥ 5 + 2*Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l3913_391354


namespace NUMINAMATH_CALUDE_box_volume_example_l3913_391350

/-- The volume of a rectangular box -/
def box_volume (width length height : ℝ) : ℝ := width * length * height

/-- Theorem: The volume of a box with width 9 cm, length 4 cm, and height 7 cm is 252 cm³ -/
theorem box_volume_example : box_volume 9 4 7 = 252 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_example_l3913_391350


namespace NUMINAMATH_CALUDE_loan_interest_equality_l3913_391333

theorem loan_interest_equality (total : ℝ) (second_part : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time1 : ℝ) (n : ℝ) :
  total = 2665 →
  second_part = 1332.5 →
  rate1 = 0.03 →
  rate2 = 0.05 →
  time1 = 5 →
  (total - second_part) * rate1 * time1 = second_part * rate2 * n →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_loan_interest_equality_l3913_391333


namespace NUMINAMATH_CALUDE_student_calculation_difference_l3913_391353

theorem student_calculation_difference : 
  let number : ℝ := 60.00000000000002
  let correct_answer := (4/5) * number
  let student_answer := number / (4/5)
  student_answer - correct_answer = 27.000000000000014 := by
sorry

end NUMINAMATH_CALUDE_student_calculation_difference_l3913_391353


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3913_391310

theorem min_value_quadratic (x y : ℝ) : 
  y = x^2 + 16*x + 10 → (∀ z, y ≤ z → z = x^2 + 16*x + 10) → y = -54 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3913_391310


namespace NUMINAMATH_CALUDE_skating_time_for_seventh_day_l3913_391374

def skating_minutes_first_four_days : ℕ := 80
def skating_minutes_next_two_days : ℕ := 100
def total_days : ℕ := 7
def target_average : ℕ := 100

theorem skating_time_for_seventh_day :
  let total_minutes_six_days := 4 * skating_minutes_first_four_days + 2 * skating_minutes_next_two_days
  let required_total_minutes := total_days * target_average
  required_total_minutes - total_minutes_six_days = 180 := by
  sorry

end NUMINAMATH_CALUDE_skating_time_for_seventh_day_l3913_391374


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3913_391356

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  (m - 1) * x^2 + 2 * x + 1 = 0

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

-- State the theorem
theorem quadratic_roots_condition (m : ℝ) :
  has_two_distinct_real_roots m ↔ m < 2 ∧ m ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3913_391356


namespace NUMINAMATH_CALUDE_polygon_sides_l3913_391393

theorem polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  (exterior_angle = 36) → (n * exterior_angle = 360) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3913_391393


namespace NUMINAMATH_CALUDE_no_positive_abc_equality_l3913_391324

theorem no_positive_abc_equality : ¬∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = a*b + a*c + b*c ∧ a*b + a*c + b*c = a*b*c := by
  sorry

end NUMINAMATH_CALUDE_no_positive_abc_equality_l3913_391324


namespace NUMINAMATH_CALUDE_total_boys_in_circle_l3913_391347

/-- Given a circular arrangement of boys, this function checks if two positions are opposite --/
def areOpposite (n : ℕ) (pos1 pos2 : ℕ) : Prop :=
  pos2 - pos1 = n / 2 ∨ pos1 - pos2 = n / 2

/-- Theorem stating the total number of boys in the circular arrangement --/
theorem total_boys_in_circle (n : ℕ) : 
  areOpposite n 7 27 ∧ 
  areOpposite n 11 36 ∧ 
  areOpposite n 15 42 → 
  n = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_boys_in_circle_l3913_391347


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3913_391371

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, y * f (2 * x) - x * f (2 * y) = 8 * x * y * (x^2 - y^2)

/-- The theorem stating that any function satisfying the functional equation
    has the form f(x) = x³ + cx for some constant c -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x^3 + c * x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3913_391371


namespace NUMINAMATH_CALUDE_collinear_vectors_n_value_l3913_391339

/-- Two vectors in ℝ² are collinear if their cross product is zero -/
def collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem collinear_vectors_n_value :
  ∀ n : ℝ, collinear (n, 1) (4, n) → n = 2 ∨ n = -2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_n_value_l3913_391339


namespace NUMINAMATH_CALUDE_optimal_selling_price_l3913_391342

/-- Represents the problem of finding the optimal selling price -/
def OptimalSellingPrice (purchase_price initial_price initial_volume : ℝ)
                        (volume_decrease_rate : ℝ) (target_profit : ℝ) : Prop :=
  let price_increase (x : ℝ) := initial_price + x
  let volume (x : ℝ) := initial_volume - volume_decrease_rate * x
  let profit (x : ℝ) := (price_increase x - purchase_price) * volume x
  ∃ x : ℝ, profit x = target_profit ∧ (price_increase x = 60 ∨ price_increase x = 80)

/-- The main theorem stating the optimal selling price -/
theorem optimal_selling_price :
  OptimalSellingPrice 40 50 500 10 8000 := by
  sorry

#check optimal_selling_price

end NUMINAMATH_CALUDE_optimal_selling_price_l3913_391342


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3913_391372

theorem isosceles_triangle_base_length 
  (perimeter : ℝ) 
  (one_side : ℝ) 
  (h_perimeter : perimeter = 29) 
  (h_one_side : one_side = 7) 
  (h_isosceles : ∃ (base equal_side : ℝ), 
    base > 0 ∧ equal_side > 0 ∧ 
    base + 2 * equal_side = perimeter ∧ 
    (base = one_side ∨ equal_side = one_side)) :
  ∃ (base : ℝ), base = 7 ∧ 
    ∃ (equal_side : ℝ), 
      base > 0 ∧ equal_side > 0 ∧
      base + 2 * equal_side = perimeter ∧
      (base = one_side ∨ equal_side = one_side) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3913_391372


namespace NUMINAMATH_CALUDE_motorcycle_material_cost_is_250_l3913_391358

/-- Represents the factory's production and sales data -/
structure FactoryData where
  car_material_cost : ℕ
  cars_produced : ℕ
  car_price : ℕ
  motorcycles_sold : ℕ
  motorcycle_price : ℕ
  profit_increase : ℕ

/-- Calculates the cost of materials for motorcycle production -/
def motorcycle_material_cost (data : FactoryData) : ℕ :=
  data.motorcycles_sold * data.motorcycle_price -
  (data.cars_produced * data.car_price - data.car_material_cost + data.profit_increase)

/-- Theorem stating the cost of materials for motorcycle production -/
theorem motorcycle_material_cost_is_250 (data : FactoryData)
  (h1 : data.car_material_cost = 100)
  (h2 : data.cars_produced = 4)
  (h3 : data.car_price = 50)
  (h4 : data.motorcycles_sold = 8)
  (h5 : data.motorcycle_price = 50)
  (h6 : data.profit_increase = 50) :
  motorcycle_material_cost data = 250 := by
  sorry

end NUMINAMATH_CALUDE_motorcycle_material_cost_is_250_l3913_391358


namespace NUMINAMATH_CALUDE_factor_sum_l3913_391395

theorem factor_sum (P Q : ℤ) : 
  (∃ b c : ℤ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l3913_391395


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3913_391302

theorem complex_equation_solution (z : ℂ) :
  (2 - 5 * Complex.I) * z = 29 → z = 2 + 5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3913_391302


namespace NUMINAMATH_CALUDE_new_players_joined_new_players_joined_game_l3913_391311

theorem new_players_joined (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  let new_players := (total_lives - initial_players * lives_per_player) / lives_per_player
  new_players

theorem new_players_joined_game : new_players_joined 8 6 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_new_players_joined_new_players_joined_game_l3913_391311


namespace NUMINAMATH_CALUDE_abs_neg_three_equals_three_l3913_391368

theorem abs_neg_three_equals_three :
  abs (-3 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_equals_three_l3913_391368


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1384_l3913_391331

theorem rightmost_three_digits_of_7_to_1384 :
  7^1384 ≡ 401 [ZMOD 1000] :=
by sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1384_l3913_391331
