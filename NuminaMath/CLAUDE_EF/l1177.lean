import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1177_117753

noncomputable def f (x : ℝ) : ℝ := 4 * Real.tan x * Real.sin (Real.pi / 2 - x) * Real.cos (x - Real.pi / 3) - Real.sqrt 3

theorem f_properties :
  -- Part 1: The smallest positive period of f is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
    -- The smallest positive period is π
    (let p := Real.pi; p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- Part 2: The monotonically increasing intervals
  (∀ (k : ℤ), ∀ (x y : ℝ),
    x ∈ Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12) ∧
    y ∈ Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12) ∧
    x < y →
    f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1177_117753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_10_is_21_216_l1177_117754

/-- A die is represented as a natural number from 1 to 6 -/
def Die : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 6 }

/-- A roll of three dice is represented as a triple of Die -/
def ThreeDiceRoll : Type := Die × Die × Die

/-- The sum of a three dice roll -/
def roll_sum (r : ThreeDiceRoll) : ℕ :=
  r.1.val + r.2.1.val + r.2.2.val

/-- The set of all possible three dice rolls -/
def all_rolls : Set ThreeDiceRoll :=
  Set.univ

/-- The set of three dice rolls that sum to 10 -/
def rolls_sum_10 : Set ThreeDiceRoll :=
  { r | r ∈ all_rolls ∧ roll_sum r = 10 }

/-- The total number of possible outcomes when rolling three dice -/
def total_outcomes : ℕ := 216

/-- The number of outcomes where the sum of the dice is 10 -/
def favorable_outcomes : ℕ := 21

/-- The probability of rolling a sum of 10 with three dice -/
noncomputable def prob_sum_10 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_10_is_21_216 :
  prob_sum_10 = 21 / 216 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_10_is_21_216_l1177_117754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_partition_l1177_117776

theorem impossibility_of_partition : ¬ ∃ (partition : List (List ℕ)),
  (partition.length = 11) ∧
  (∀ group, group ∈ partition → group.length = 3) ∧
  (∀ n, n ∈ Finset.range 33 → n + 1 ∈ partition.join) ∧
  (∀ group, group ∈ partition → ∃ a b c, a ∈ group ∧ b ∈ group ∧ c ∈ group ∧ c = a + b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_partition_l1177_117776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_calculation_l1177_117781

/-- Calculates the unoccupied volume in a cylindrical tank with water and spherical balls -/
theorem unoccupied_volume_calculation 
  (tank_radius : ℝ) 
  (tank_height : ℝ) 
  (water_percentage : ℝ) 
  (num_balls : ℕ) 
  (ball_radius : ℝ) 
  (h₁ : tank_radius = 5)
  (h₂ : tank_height = 10)
  (h₃ : water_percentage = 0.4)
  (h₄ : num_balls = 15)
  (h₅ : ball_radius = 1) : 
  (π * tank_radius^2 * tank_height) - 
  (water_percentage * (π * tank_radius^2 * tank_height) + 
   ↑num_balls * ((4/3) * π * ball_radius^3)) = 130 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_calculation_l1177_117781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_price_calculation_l1177_117738

-- Define the given quantities
noncomputable def total_flour : ℚ := 6
noncomputable def flour_for_cakes : ℚ := 4
noncomputable def flour_per_cake : ℚ := 1/2
noncomputable def flour_for_cupcakes : ℚ := 2
noncomputable def flour_per_cupcake : ℚ := 1/5
noncomputable def price_per_cupcake : ℚ := 1
noncomputable def total_earnings : ℚ := 30

-- Define the theorem
theorem cake_price_calculation :
  let num_cakes : ℚ := flour_for_cakes / flour_per_cake
  let num_cupcakes : ℚ := flour_for_cupcakes / flour_per_cupcake
  let earnings_from_cupcakes : ℚ := num_cupcakes * price_per_cupcake
  let earnings_from_cakes : ℚ := total_earnings - earnings_from_cupcakes
  let price_per_cake : ℚ := earnings_from_cakes / num_cakes
  price_per_cake = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_price_calculation_l1177_117738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_intersection_implies_a_range_l1177_117741

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 1
def g (a x : ℝ) : ℝ := 2 * x + a

-- Define the interval [1/2, 1]
def I : Set ℝ := Set.Icc (1/2) 1

-- State the theorem
theorem function_intersection_implies_a_range :
  ∀ a : ℝ, (∃ x₁ x₂, x₁ ∈ I ∧ x₂ ∈ I ∧ f x₁ = g a x₂) → a ∈ Set.Icc (-3) (-3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_intersection_implies_a_range_l1177_117741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_range_of_a_full_range_of_a_l1177_117751

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (6 - 2*x) + Real.log (x + 2) / Real.log 10

-- Define set A (domain of f)
def A : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x > 3 ∨ x < 2}

-- Define set C
def C (a : ℝ) : Set ℝ := {x : ℝ | x < 2*a + 1}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | -2 < x ∧ x < 2} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) : B ∩ C a = C a → a ≤ 1/2 := by sorry

-- Theorem for the full range of a
theorem full_range_of_a : {a : ℝ | ∃ (x : ℝ), B ∩ C a = C a} = Set.Iic (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_range_of_a_full_range_of_a_l1177_117751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1177_117714

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x else Real.sqrt (-x)

-- State the theorem
theorem solution_exists (a : ℝ) (h : f a + f (-1) = 2) : a = 1 ∨ a = -1 := by
  -- The proof is skipped using 'sorry'
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1177_117714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_equivalence_l1177_117774

variable (robot_can_fly : Prop)
variable (tree_produces_fruit : Prop)

theorem statement_equivalence :
  (robot_can_fly → ¬tree_produces_fruit) ↔
  ((¬tree_produces_fruit → ¬robot_can_fly) ∧
   (¬robot_can_fly ∨ ¬tree_produces_fruit)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_equivalence_l1177_117774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_paper_distribution_l1177_117727

theorem colored_paper_distribution (groups : ℕ) (students_per_group : ℕ) 
  (bundles : ℕ) (sheets_per_bundle : ℕ) (leftover_sheets : ℕ) : 
  groups = 7 → 
  students_per_group = 6 → 
  bundles = 14 → 
  sheets_per_bundle = 105 → 
  leftover_sheets = 84 → 
  (bundles * sheets_per_bundle - leftover_sheets) / (groups * students_per_group) = 33 := by
  intro hg hs hb hsh hl
  sorry

#check colored_paper_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_paper_distribution_l1177_117727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_70_minutes_l1177_117707

/-- Represents a cyclist with their speeds on different terrains -/
structure Cyclist where
  flatSpeed : ℚ
  downhillSpeed : ℚ
  uphillSpeed : ℚ

/-- Represents a route segment with distance and terrain type -/
inductive Terrain
  | Flat
  | Downhill
  | Uphill

structure RouteSegment where
  distance : ℚ
  terrain : Terrain

/-- Calculates the time taken for a cyclist to complete a route segment -/
def timeTaken (c : Cyclist) (s : RouteSegment) : ℚ :=
  match s.terrain with
  | Terrain.Flat => s.distance / c.flatSpeed
  | Terrain.Downhill => s.distance / c.downhillSpeed
  | Terrain.Uphill => s.distance / c.uphillSpeed

/-- Calculates the total time taken for a cyclist to complete a route -/
def totalTime (c : Cyclist) (route : List RouteSegment) : ℚ :=
  (route.map (timeTaken c)).sum

theorem time_difference_is_70_minutes 
  (minnie : Cyclist)
  (penny : Cyclist)
  (minnieRoute : List RouteSegment)
  (pennyRoute : List RouteSegment)
  (h1 : minnie.flatSpeed = 30)
  (h2 : minnie.downhillSpeed = 45)
  (h3 : minnie.uphillSpeed = 15/2)
  (h4 : penny.flatSpeed = 45)
  (h5 : penny.downhillSpeed = 60)
  (h6 : penny.uphillSpeed = 15)
  (h7 : minnieRoute = [
    ⟨15, Terrain.Uphill⟩, 
    ⟨20, Terrain.Downhill⟩, 
    ⟨30, Terrain.Flat⟩
  ])
  (h8 : pennyRoute = [
    ⟨30, Terrain.Flat⟩, 
    ⟨20, Terrain.Uphill⟩, 
    ⟨15, Terrain.Downhill⟩
  ]) :
  (totalTime minnie minnieRoute - totalTime penny pennyRoute) * 60 = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_70_minutes_l1177_117707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_mod_55_l1177_117715

/-- M is the number formed by concatenating the integers from 1 to 55 -/
def M : ℕ := sorry

/-- The remainder when M is divided by 55 is 45 -/
theorem M_mod_55 : M % 55 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_mod_55_l1177_117715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equivalence_l1177_117798

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Converts a polar point to standard form (r > 0, 0 ≤ θ < 2π) -/
noncomputable def toStandardPolar (p : PolarPoint) : PolarPoint :=
  if p.r < 0 then
    { r := -p.r, θ := (p.θ + Real.pi) % (2 * Real.pi) }
  else
    { r := p.r, θ := p.θ % (2 * Real.pi) }

theorem polar_equivalence :
  let p := PolarPoint.mk (-3) (Real.pi / 6)
  let standard_p := toStandardPolar p
  standard_p.r = 3 ∧ standard_p.θ = 7 * Real.pi / 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equivalence_l1177_117798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_4_l1177_117719

-- Define the sequence T
def T : ℕ → ℕ
  | 0 => 3  -- Add a case for 0
  | 1 => 3
  | n + 2 => 3^(T (n + 1))

-- State the theorem
theorem t_100_mod_4 : T 100 % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_4_l1177_117719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_green_ball_selected_l1177_117718

def containerA : ℕ × ℕ := (10, 5)  -- (red balls, green balls)
def containerB : ℕ × ℕ := (3, 6)
def containerC : ℕ × ℕ := (3, 6)

def totalContainers : ℕ := 3

def probabilityGreenBall (container : ℕ × ℕ) : ℚ :=
  ↑container.2 / ↑(container.1 + container.2)

theorem probability_green_ball_selected :
  (1 : ℚ) / totalContainers * (probabilityGreenBall containerA +
                               probabilityGreenBall containerB +
                               probabilityGreenBall containerC) = 5 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_green_ball_selected_l1177_117718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_cut_length_is_two_l1177_117792

/-- Calculates the length of hair cut off in the second cut -/
noncomputable def second_cut_length (initial_length : ℝ) (growth : ℝ) (final_length : ℝ) : ℝ :=
  initial_length / 2 + growth - final_length

theorem second_cut_length_is_two :
  second_cut_length 24 4 14 = 2 := by
  -- Unfold the definition of second_cut_length
  unfold second_cut_length
  -- Simplify the arithmetic
  simp [add_sub, div_eq_mul_inv]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_cut_length_is_two_l1177_117792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_area_theorem_l1177_117706

/-- The area of a circle with radius r -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/-- The area of a ring-shaped region between two concentric circles -/
noncomputable def ring_area (inner_radius outer_radius : ℝ) : ℝ :=
  circle_area outer_radius - circle_area inner_radius

/-- Theorem: The area of the ring-shaped region between concentric circles
    with radii 4 and 15 is 209π -/
theorem ring_area_theorem :
  ring_area 4 15 = 209 * Real.pi := by
  -- Expand the definition of ring_area
  unfold ring_area
  -- Expand the definition of circle_area
  unfold circle_area
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_area_theorem_l1177_117706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1177_117766

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

-- State the theorem
theorem inequality_solution_set :
  ∀ x : ℝ, 0 < x → 
  ((f (Real.log x) - f (Real.log (1/x))) / 2 < f 1 ↔ x < Real.exp 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1177_117766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_value_of_log2_l1177_117745

-- Define the interval
def I : Set ℝ := Set.Icc 1 (2^2018)

-- Define the function f(x) = log₂(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the "average value" property
def is_average_value (M : ℝ) (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x₁, x₁ ∈ I → ∃! x₂, x₂ ∈ I ∧ (f x₁ + f x₂) / 2 = M

-- Theorem statement
theorem average_value_of_log2 :
  is_average_value 1009 f I := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_value_of_log2_l1177_117745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l1177_117755

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define the focal length
noncomputable def focal_length (e : (ℝ → ℝ → Prop)) : ℝ :=
  2 * Real.sqrt 3

-- Theorem statement
theorem ellipse_focal_length :
  focal_length ellipse = 2 * Real.sqrt 3 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l1177_117755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_pi_half_plus_theta_l1177_117721

theorem sin_three_pi_half_plus_theta (θ : Real) (h : Real.tan θ = 1/3) :
  Real.sin (3*Real.pi/2 + θ) = -3*Real.sqrt 10/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_pi_half_plus_theta_l1177_117721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_rate_equation_l1177_117720

/-- Represents the annual average growth rate of vegetable yield -/
def x : ℝ := sorry

/-- Initial vegetable yield in tons -/
def initial_yield : ℝ := 80

/-- Final vegetable yield in tons -/
def final_yield : ℝ := 100

/-- Number of years between initial and final yield -/
def years : ℕ := 2

/-- Theorem stating that the equation correctly represents the growth rate -/
theorem growth_rate_equation :
  initial_yield * (1 + x)^years = final_yield := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_rate_equation_l1177_117720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dist_circle_to_line_l1177_117709

-- Define the circle
noncomputable def circle_A : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ c : ℝ × ℝ, (p.1 - c.1)^2 + (p.2 - c.2)^2 = 1 ∧ (c.1 - 2)^2 + (c.2 - 3)^2 = 1}

-- Define the line
def line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 - 4 = 0}

-- Define the distance function from a point to the line
noncomputable def dist_to_line (p : ℝ × ℝ) : ℝ :=
  |3 * p.1 - 4 * p.2 - 4| / Real.sqrt 25

-- Theorem statement
theorem max_dist_circle_to_line :
  ∃ c : ℝ × ℝ, c ∈ circle_A ∧ 
    ∀ p : ℝ × ℝ, p ∈ circle_A → dist_to_line c ≥ dist_to_line p ∧
    dist_to_line c = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dist_circle_to_line_l1177_117709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_order_is_sixty_l1177_117790

/-- Represents an assembly line production scenario -/
structure AssemblyLine where
  r1 : ℝ  -- Initial production rate (cogs per hour)
  r2 : ℝ  -- Secondary production rate (cogs per hour)
  n : ℝ   -- Number of additional cogs produced at rate r2
  a : ℝ   -- Overall average output (cogs per hour)

/-- Calculates the initial order size for a given assembly line scenario -/
noncomputable def initialOrderSize (al : AssemblyLine) : ℝ :=
  (al.a * (al.n / al.r2 + al.n / al.a) - al.n) / (1 - al.a / al.r1)

/-- Theorem stating that for the given parameters, the initial order size is 60 -/
theorem initial_order_is_sixty :
  let al : AssemblyLine := {
    r1 := 90
    r2 := 60
    n := 60
    a := 72
  }
  initialOrderSize al = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_order_is_sixty_l1177_117790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_side_ratio_l1177_117743

/-- 
Given an obtuse triangle with interior angles forming an arithmetic sequence,
prove that the ratio of the longest side to the shortest side is greater than 2.
-/
theorem obtuse_triangle_side_ratio (α : ℝ) (m : ℝ) : 
  (π/3 < α) ∧ (α < π/2) ∧  -- Ensures the triangle is obtuse
  (m = (Real.sin (π/3 + α)) / (Real.sin (π/3 - α))) →  -- Ratio of sides using law of sines
  m > 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_side_ratio_l1177_117743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_pocket_problem_l1177_117772

/-- Represents the pocket operations --/
inductive PocketOp
| Left  : PocketOp  -- x → x + 1
| Right : PocketOp  -- x → x^(-1)

/-- Applies the pocket operation to a number --/
noncomputable def applyOp (x : ℝ) (op : PocketOp) : ℝ :=
  match op with
  | PocketOp.Left  => x + 1
  | PocketOp.Right => x⁻¹

/-- The expected value after n steps --/
noncomputable def expectedValue (n : ℕ) (initial : ℝ) : ℝ :=
  sorry  -- Definition omitted for brevity

/-- The main theorem --/
theorem rachel_pocket_problem (initial : ℝ) (n : ℕ) 
    (h_initial : initial = 1000) (h_steps : n = 8) :
    ⌊expectedValue n initial / 10⌋ = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_pocket_problem_l1177_117772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reese_practice_time_l1177_117786

/-- The number of weeks in one month (on average) -/
noncomputable def weeksPerMonth : ℝ := 4.345

/-- The number of months Reese has been practicing -/
def monthsPracticing : ℕ := 5

/-- The total number of hours Reese practices in five months -/
noncomputable def totalHours : ℝ := 80

/-- The number of hours Reese practices per week -/
noncomputable def hoursPerWeek : ℝ := totalHours / (monthsPracticing * weeksPerMonth)

theorem reese_practice_time : 
  hoursPerWeek = totalHours / (monthsPracticing * weeksPerMonth) := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reese_practice_time_l1177_117786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_transformation_l1177_117749

/-- Given a function f: ℝ → ℝ, area_under_curve f computes the area between
    the graph of f and the x-axis. -/
noncomputable def area_under_curve (f : ℝ → ℝ) : ℝ := sorry

/-- Given a function f: ℝ → ℝ and a real number k,
    horizontal_shift f k returns the function f(x - k). -/
def horizontal_shift (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x ↦ f (x - k)

/-- Given a function f: ℝ → ℝ and a real number c,
    vertical_scale f c returns the function c * f(x). -/
def vertical_scale (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := λ x ↦ c * f x

theorem area_transformation (f : ℝ → ℝ) :
  area_under_curve f = 15 →
  area_under_curve (vertical_scale (horizontal_shift f 4) 5) = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_transformation_l1177_117749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1177_117789

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ 2 * x - y + 1 = 0) ∧
    (m = (deriv f) point.1) ∧
    (point.2 = m * point.1 + b) ∧
    (point.2 = f point.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1177_117789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_second_hour_l1177_117771

/-- Proves that given a car's speed of 140 km/h in the first hour and an average speed of 90 km/h over two hours, the speed in the second hour is 40 km/h. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_first_hour = 140) 
  (h2 : average_speed = 90) 
  (h3 : total_time = 2) : 
  (average_speed * total_time - speed_first_hour) / (total_time / 2) = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_second_hour_l1177_117771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1177_117783

/-- The circle equation: x^2 + y^2 - 4x - 4y - 10 = 0 -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 10 = 0

/-- The line equation: x + y - 14 = 0 -/
def lineEq (x y : ℝ) : Prop := x + y - 14 = 0

/-- The distance from a point (x, y) to the line x + y - 14 = 0 -/
noncomputable def distanceToLine (x y : ℝ) : ℝ := |x + y - 14| / Real.sqrt 2

/-- Theorem: The maximum distance from any point on the circle to the line is 8√2 -/
theorem max_distance_circle_to_line :
  ∃ (x y : ℝ), circleEq x y ∧ 
    (∀ (x' y' : ℝ), circleEq x' y' → distanceToLine x y ≥ distanceToLine x' y') ∧
    distanceToLine x y = 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1177_117783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_is_union_l1177_117717

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) / (x - 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ 1 ∧ x ≠ 2}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry

-- Additional lemma to show the domain is equivalent to the union of intervals
theorem domain_is_union : 
  domain_f = Set.Icc 1 2 ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_is_union_l1177_117717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l1177_117702

/-- The function f(x) = a*sin(x) + b*cos(x) -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x

/-- The function g(x) = f(x + π/3) / x -/
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a b (x + Real.pi/3) / x

/-- Theorem stating the properties of f and g based on the given conditions -/
theorem f_and_g_properties :
  ∃ (a b : ℝ),
    (f a b (Real.pi/3) = 0) ∧
    ((deriv (f a b)) (Real.pi/3) = 1) ∧
    (a = 1/2 ∧ b = -Real.sqrt 3 / 2) ∧
    (∀ x : ℝ, 0 < x → x ≤ Real.pi/2 → g a b x ≥ 2/Real.pi) ∧
    (g a b (Real.pi/2) = 2/Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l1177_117702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1177_117747

-- Define the function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (3 - a) ^ x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) →
  a ∈ Set.Ioo 2 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1177_117747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_higher_punctuality_in_large_group_l1177_117725

/-- Represents a tour group -/
structure TourGroup where
  size : ℕ
  costPerPerson : ℚ
  perceivedImportance : ℚ

/-- The probability of a tourist being punctual -/
noncomputable def punctualityProbability (group : TourGroup) : ℚ :=
  1 - (group.perceivedImportance / group.costPerPerson)

/-- Large tour group -/
def largeGroup : TourGroup :=
  { size := 50
  , costPerPerson := 100
  , perceivedImportance := 2 }

/-- Small tour group -/
def smallGroup : TourGroup :=
  { size := 10
  , costPerPerson := 200
  , perceivedImportance := 10 }

theorem higher_punctuality_in_large_group :
  punctualityProbability largeGroup > punctualityProbability smallGroup := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_higher_punctuality_in_large_group_l1177_117725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_solution_l1177_117773

-- Define the base a
variable (a : ℝ)

-- State the conditions
axiom a_pos : a > 0
axiom a_neq_one : a ≠ 1
axiom exp_property : ∀ x : ℝ, x < 0 → a^x > 1

-- Define the solution set
def solution_set : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem log_inequality_solution :
  {x : ℝ | Real.log x > 0} = solution_set :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_solution_l1177_117773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_day_of_week_proof_l1177_117746

/-- Represents days of the week -/
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

/-- Function to calculate the day of the week for a given year and day -/
def DayOfWeek.ofYearDay : YearDay → DayOfWeek :=
  sorry

/-- Given that the 250th day of year M is a Thursday and
    the 150th day of year M+1 is a Thursday,
    prove that the 50th day of year M-1 is a Friday -/
theorem day_of_week_proof (M : Int)
  (h1 : DayOfWeek.Thursday = DayOfWeek.ofYearDay ⟨M, 250⟩)
  (h2 : DayOfWeek.Thursday = DayOfWeek.ofYearDay ⟨M+1, 150⟩) :
  DayOfWeek.Friday = DayOfWeek.ofYearDay ⟨M-1, 50⟩ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_day_of_week_proof_l1177_117746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_laps_proof_l1177_117760

/-- Calculates the number of laps run by the boys given the conditions of the track problem -/
def track_problem (lap_length : ℚ) (girls_distance : ℚ) (lap_difference : ℕ) : ℕ :=
  let girls_laps : ℕ := (girls_distance / lap_length).ceil.toNat
  let boys_laps : ℕ := girls_laps - lap_difference
  boys_laps

/-- Proves that the boys ran 27 laps given the conditions of the track problem -/
theorem boys_laps_proof (h1 : lap_length = 3/4)
                        (h2 : girls_distance = 27)
                        (h3 : lap_difference = 9) :
  track_problem lap_length girls_distance lap_difference = 27 := by
  sorry

#eval track_problem (3/4) 27 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_laps_proof_l1177_117760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l1177_117780

/-- A circle passes through (0,1) and is tangent to y = x^3 at (1,1). Its center is (1/2, 7/6). -/
theorem circle_center (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (x - center.1)^2 + (y - center.2)^2 = (center.1 - 1)^2 + (center.2 - 1)^2) →
  (0, 1) ∈ C →
  (1, 1) ∈ C →
  (∀ (x : ℝ), x ≠ 1 → ((x, x^3) ∈ C → (x - 1)^2 + (x^3 - 1)^2 > (center.1 - 1)^2 + (center.2 - 1)^2)) →
  center = (1/2, 7/6) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l1177_117780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_table_distinct_rows_l1177_117700

/-- Represents a node in the triangle table -/
structure Node where
  value : ℕ
  row : ℕ
  column : ℕ

/-- The triangle table -/
inductive TriangleTable (a : ℕ) : Node → Prop
  | root : TriangleTable a ⟨a, 0, 0⟩
  | left (parent : Node) (h : TriangleTable a parent) :
      TriangleTable a ⟨parent.value ^ 2, parent.row + 1, parent.column * 2⟩
  | right (parent : Node) (h : TriangleTable a parent) :
      TriangleTable a ⟨parent.value + 1, parent.row + 1, parent.column * 2 + 1⟩

/-- The theorem stating that all numbers in each row are distinct -/
theorem triangle_table_distinct_rows (a : ℕ) (h : a > 1) :
  ∀ r : ℕ, ∀ n1 n2 : Node, 
    TriangleTable a n1 → 
    TriangleTable a n2 → 
    n1.row = r → 
    n2.row = r → 
    n1.column ≠ n2.column → 
    n1.value ≠ n2.value :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_table_distinct_rows_l1177_117700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_exponential_logs_l1177_117763

-- Define the function f(x) = ln x * ln(11 - x)
noncomputable def f (x : ℝ) : ℝ := Real.log x * Real.log (11 - x)

-- State the theorem
theorem compare_exponential_logs :
  (6 : ℝ)^(Real.log 5) > (7 : ℝ)^(Real.log 4) ∧ (7 : ℝ)^(Real.log 4) > (8 : ℝ)^(Real.log 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_exponential_logs_l1177_117763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_some_equal_area_rectangles_l1177_117752

/-- A rectangle in a grid --/
structure GridRectangle where
  width : Nat
  height : Nat

/-- The problem setup --/
structure SquareGrid where
  side : Nat
  rectangles : List GridRectangle
  rectangles_count : (rectangles.length = 8)
  total_area : (List.sum (rectangles.map (λ r => r.width * r.height))) = side * side
  valid_dimensions : ∀ r ∈ rectangles, r.width ≤ side ∧ r.height ≤ side

/-- The main theorem --/
theorem some_equal_area_rectangles (grid : SquareGrid) (h : grid.side = 6) :
  ∃ (r1 r2 : GridRectangle), r1 ∈ grid.rectangles ∧ r2 ∈ grid.rectangles ∧ r1 ≠ r2 ∧ r1.width * r1.height = r2.width * r2.height :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_some_equal_area_rectangles_l1177_117752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1177_117735

noncomputable def w : ℝ × ℝ := (3/5, -4/5)

noncomputable def proj (v : ℝ × ℝ) (w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let magnitude_squared := w.1 * w.1 + w.2 * w.2
  (dot_product / magnitude_squared * w.1, dot_product / magnitude_squared * w.2)

theorem projection_theorem :
  proj (2, -4) w = (3/5, -4/5) →
  proj (-1, 5) w = (-69/25, 92/25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1177_117735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l1177_117764

-- Define the constants
noncomputable def a : ℝ := Real.log (1/3)
noncomputable def b : ℝ := 3^(1/10)
noncomputable def c : ℝ := Real.sin 3

-- State the theorem
theorem order_of_constants : b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l1177_117764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_bounds_l1177_117758

noncomputable def f (x : ℝ) := 4 / x

theorem inverse_proportion_bounds (m n : ℝ) :
  (∀ x, -4 ≤ x ∧ x ≤ m → n ≤ f x ∧ f x ≤ n + 3) →
  m < 0 →
  m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_bounds_l1177_117758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_difference_l1177_117713

theorem average_difference : ∃ (avg1 avg2 : ℝ), 
  let set1 : List ℝ := [24, 35, 58]
  let set2 : List ℝ := [19, 51, 29]
  avg1 = (set1.sum) / (set1.length : ℝ) ∧
  avg2 = (set2.sum) / (set2.length : ℝ) ∧
  avg1 - avg2 = 6 := by
  let set1 : List ℝ := [24, 35, 58]
  let set2 : List ℝ := [19, 51, 29]
  let avg1 := (set1.sum) / (set1.length : ℝ)
  let avg2 := (set2.sum) / (set2.length : ℝ)
  use avg1, avg2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_difference_l1177_117713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_diversity_l1177_117769

theorem grid_diversity (grid : Fin 101 → Fin 101 → Fin 101) 
  (h_count : ∀ k : Fin 101, (Finset.sum Finset.univ fun i => Finset.sum Finset.univ fun j => if grid i j = k then 1 else 0) = 101) :
  ∃ i : Fin 101, (11 ≤ (Finset.univ.filter (fun j => grid i j ∈ Finset.univ)).card) ∨
                 (11 ≤ (Finset.univ.filter (fun j => grid j i ∈ Finset.univ)).card) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_diversity_l1177_117769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_r_l1177_117729

-- Define the type for composite positive integers
def CompositePositiveInteger : Type := {n : ℕ // n > 1 ∧ ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b}

-- Define the function r
def r : CompositePositiveInteger → ℕ := sorry

-- State the theorem
theorem range_of_r :
  (∀ m : ℕ, m > 3 → ∃ n : CompositePositiveInteger, r n = m) ∧
  (∀ n : CompositePositiveInteger, r n > 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_r_l1177_117729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_10_in_expansion_l1177_117703

theorem coefficient_x_10_in_expansion : ∃ (c : ℤ), c = 19 ∧
  (∀ (x : ℝ), (2 + x)^10 * (x - 1) = c * x^10 + (λ y => y^11) x + (λ y => y^9) x + 
  (λ y => y^8) x + (λ y => y^7) x + (λ y => y^6) x + (λ y => y^5) x + 
  (λ y => y^4) x + (λ y => y^3) x + (λ y => y^2) x + (λ y => y) x + (λ y => 1) x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_10_in_expansion_l1177_117703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1177_117726

/-- The time it takes for person a to complete the work alone -/
noncomputable def time_a : ℝ := 10

/-- The time it takes for person b to complete the work alone -/
noncomputable def time_b : ℝ := 9

/-- The time it takes for persons a and b to complete the work together -/
noncomputable def time_ab : ℝ := 4.7368421052631575

/-- The rate at which a person completes work is the reciprocal of their time -/
noncomputable def rate (time : ℝ) : ℝ := 1 / time

theorem work_completion_time :
  rate time_a + rate time_b = rate time_ab := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1177_117726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dinner_tip_percentage_l1177_117723

/-- Calculates the tip percentage given the following conditions:
  * 4 people ordering dinner
  * Each main meal costs $12.00
  * 2 appetizers at $6.00 each
  * An extra $5.00 is added for rush order
  * Total spent is $77.00
-/
theorem dinner_tip_percentage 
  (num_people : ℝ)
  (main_meal_cost : ℝ)
  (num_appetizers : ℝ)
  (appetizer_cost : ℝ)
  (rush_order_fee : ℝ)
  (total_spent : ℝ) :
  num_people = 4 →
  main_meal_cost = 12 →
  num_appetizers = 2 →
  appetizer_cost = 6 →
  rush_order_fee = 5 →
  total_spent = 77 →
  let subtotal := num_people * main_meal_cost + num_appetizers * appetizer_cost
  let tip := total_spent - (subtotal + rush_order_fee)
  tip / subtotal * 100 = 20 := by
  sorry

#check dinner_tip_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dinner_tip_percentage_l1177_117723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_a_length_l1177_117711

-- Define the triangle ABC
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Define the angle A in radians (60 degrees = π/3 radians)
noncomputable def angle_A : ℝ := Real.pi / 3

-- Define the area of the triangle
noncomputable def triangle_area (b c : ℝ) : ℝ := (b * c * Real.sin angle_A) / 2

-- State the theorem
theorem side_a_length (a b c : ℝ) :
  triangle a b c →
  triangle_area b c = (3 * Real.sqrt 3) / 2 →
  b + c = 3 * Real.sqrt 3 →
  a = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_a_length_l1177_117711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_a18_equals_13_correct_l1177_117799

def sequence_prob (a : Fin 18 → ℝ) : Prop :=
  a 1 = 0 ∧ ∀ k : Fin 17, |a (k + 1) - a k| = 1

noncomputable def prob_a18_equals_13 (a : Fin 18 → ℝ) (h : sequence_prob a) : ℝ :=
  17 / 16384

theorem prob_a18_equals_13_correct (a : Fin 18 → ℝ) (h : sequence_prob a) :
  prob_a18_equals_13 a h = 17 / 16384 := by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_a18_equals_13_correct_l1177_117799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1177_117739

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

def monotonous_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

theorem omega_range (ω : ℝ) : 
  (monotonous_in_interval (f ω) (-Real.pi/6) (Real.pi/4)) ↔ 
  (ω ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1177_117739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1177_117728

noncomputable def h (x : ℝ) : ℝ := (2 * x - 3) / (x - 5)

theorem domain_of_h :
  {x : ℝ | ∃ y, h x = y} = {x : ℝ | x ≠ 5} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1177_117728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1177_117795

/-- Given a hyperbola and a line intersecting its asymptotes, if a point on the x-axis is equidistant from the intersection points, then the hyperbola has a specific eccentricity. -/
theorem hyperbola_eccentricity (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m ≠ 0) :
  let hyperbola := λ x y : ℝ ↦ x^2 / a^2 - y^2 / b^2 = 1
  let line := λ x y : ℝ ↦ x - 3*y + m = 0
  let asymptote1 := λ x : ℝ ↦ (b/a) * x
  let asymptote2 := λ x : ℝ ↦ -(b/a) * x
  let A := (m*a/(3*b-a), m*b/(3*b-a))
  let B := (-m*a/(3*b+a), m*b/(3*b+a))
  let P := (m, 0)
  let dist := λ p q : ℝ × ℝ ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (∀ x y : ℝ, hyperbola x y → (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) →
  (∀ x : ℝ, line x (asymptote1 x) ∨ line x (asymptote2 x)) →
  (dist P A = dist P B) →
  let e := Real.sqrt (a^2 + b^2) / a
  e = Real.sqrt 5 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1177_117795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_affine_coordinate_system_properties_l1177_117796

noncomputable section

/-- α-affine coordinate system -/
structure AffineCoordSystem where
  α : Real
  α_pos : α > 0
  α_lt_pi : α < π
  α_ne_pi_div_2 : α ≠ π / 2

/-- Vector in α-affine coordinate system -/
structure AffineVector (A : AffineCoordSystem) where
  x : Real
  y : Real

/-- Magnitude of a vector in α-affine coordinate system -/
noncomputable def magnitude (A : AffineCoordSystem) (v : AffineVector A) : Real :=
  Real.sqrt (v.x^2 + v.y^2 + 2 * v.x * v.y * Real.cos A.α)

/-- Parallel vectors in α-affine coordinate system -/
def isParallel (A : AffineCoordSystem) (v w : AffineVector A) : Prop :=
  v.x * w.y = v.y * w.x

/-- Angle between vectors in α-affine coordinate system -/
noncomputable def angleBetween (A : AffineCoordSystem) (v w : AffineVector A) : Real :=
  Real.arccos ((v.x * w.x + v.y * w.y + (v.x * w.y + v.y * w.x) * Real.cos A.α) /
    (magnitude A v * magnitude A w))

theorem affine_coordinate_system_properties (A : AffineCoordSystem) :
  (∀ v : AffineVector A, magnitude A v = Real.sqrt (v.x^2 + v.y^2 + 2 * v.x * v.y * Real.cos A.α)) ∧
  (∀ v w : AffineVector A, isParallel A v w ↔ v.x * w.y = v.y * w.x) ∧
  (let v := { x := -1, y := 2 : AffineVector A }
   let w := { x := -2, y := 1 : AffineVector A }
   angleBetween A v w = π / 3 → A.α = π / 3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_affine_coordinate_system_properties_l1177_117796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1177_117724

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  h : ∀ n, a (n + 1) = a n * q

/-- Sum of first n terms of a geometric sequence -/
noncomputable def sumFirstN (seq : GeometricSequence) (n : ℕ) : ℝ :=
  if seq.q = 1 then
    n * seq.a 0
  else
    seq.a 0 * (1 - seq.q^n) / (1 - seq.q)

/-- Theorem stating that if S_3 = 3a_3 for a geometric sequence, then q = -1 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  sumFirstN seq 3 = 3 * seq.a 3 → seq.q = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1177_117724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l1177_117734

/-- The differential equation y'' - 4y' + 3y = x - 1 --/
def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (deriv^[2] y) x - 4 * (deriv y) x + 3 * y x = x - 1

/-- The general solution of the differential equation --/
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  C₁ * Real.exp (3 * x) + C₂ * Real.exp x + (1/3) * x + 1/9

/-- Theorem stating that the general_solution satisfies the differential equation --/
theorem general_solution_satisfies_diff_eq (C₁ C₂ : ℝ) :
  ∀ x, diff_eq (general_solution C₁ C₂) x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l1177_117734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_finite_harmonious_set_sqrt3_set_is_harmonious_harmonious_sets_intersect_not_all_harmonious_sets_union_is_real_l1177_117759

-- Definition of a harmonious set
def HarmoniousSet (S : Set ℝ) : Prop :=
  S.Nonempty ∧ ∀ a b, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a - b) ∈ S

-- Statement 1
theorem exists_finite_harmonious_set :
  ∃ S : Set ℝ, HarmoniousSet S ∧ S.Finite := by
  sorry

-- Statement 2
def sqrt3_set : Set ℝ := {x | ∃ k : ℤ, x = Real.sqrt 3 * k}

theorem sqrt3_set_is_harmonious : HarmoniousSet sqrt3_set := by
  sorry

-- Statement 3
theorem harmonious_sets_intersect (S₁ S₂ : Set ℝ) :
  HarmoniousSet S₁ → HarmoniousSet S₂ → (S₁ ∩ S₂).Nonempty := by
  sorry

-- Statement 4
theorem not_all_harmonious_sets_union_is_real :
  ∃ S₁ S₂ : Set ℝ, HarmoniousSet S₁ ∧ HarmoniousSet S₂ ∧ S₁ ≠ S₂ ∧ (S₁ ∪ S₂ ≠ Set.univ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_finite_harmonious_set_sqrt3_set_is_harmonious_harmonious_sets_intersect_not_all_harmonious_sets_union_is_real_l1177_117759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_gravity_correct_center_on_symmetry_axis_l1177_117744

/-- Represents a T-shaped plate composed of two rectangles -/
structure TPlate where
  h₁ : ℝ  -- height of the stem
  w : ℝ   -- width of the stem
  h₂ : ℝ  -- height of the top bar
  W : ℝ   -- width of the top bar
  stem_positive : h₁ > 0 ∧ w > 0
  top_bar_positive : h₂ > 0 ∧ W > 0

/-- The center of gravity of a T-shaped plate -/
noncomputable def center_of_gravity (plate : TPlate) : ℝ × ℝ :=
  let A₁ := plate.h₁ * plate.w
  let A₂ := plate.h₂ * plate.W
  let x_cm := (plate.w / 2 * A₁ + plate.W / 2 * A₂) / (A₁ + A₂)
  let y_cm := (plate.h₁ / 2 * A₁ + (plate.h₁ + plate.h₂ / 2) * A₂) / (A₁ + A₂)
  (x_cm, y_cm)

theorem center_of_gravity_correct (plate : TPlate) :
  center_of_gravity plate = 
    let A₁ := plate.h₁ * plate.w
    let A₂ := plate.h₂ * plate.W
    let x_cm := (plate.w / 2 * A₁ + plate.W / 2 * A₂) / (A₁ + A₂)
    let y_cm := (plate.h₁ / 2 * A₁ + (plate.h₁ + plate.h₂ / 2) * A₂) / (A₁ + A₂)
    (x_cm, y_cm) := by
  sorry

/-- The T-shaped plate has an axis of symmetry -/
def has_symmetry_axis (plate : TPlate) : Prop := 
  plate.W > plate.w

/-- The center of gravity lies on the symmetry axis -/
theorem center_on_symmetry_axis (plate : TPlate) :
  has_symmetry_axis plate → 
  (center_of_gravity plate).1 = plate.W / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_gravity_correct_center_on_symmetry_axis_l1177_117744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1177_117705

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 1 + Real.log (x - 1)

-- Define the domain of f
def dom_f : Set ℝ := {x | x > 1}

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 1) + 1

-- Define the domain of g
def dom_g : Set ℝ := {x | x > 0}

-- State the theorem
theorem inverse_function_theorem :
  ∀ x ∈ dom_f, ∀ y ∈ dom_g,
    (f (g y) = y ∧ g (f x) = x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1177_117705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_shift_size_l1177_117750

-- Define the company and its properties
structure Company where
  total_employees : ℕ
  first_shift : ℕ
  second_shift : ℕ
  third_shift : ℕ
  first_shift_participation : ℚ
  second_shift_participation : ℚ
  third_shift_participation : ℚ
  total_participation : ℚ

-- Define the specific company x
def company_x : Company where
  total_employees := 150  -- 60 + 50 + 40
  first_shift := 60
  second_shift := 50  -- This is what we want to prove
  third_shift := 40
  first_shift_participation := 1/5
  second_shift_participation := 2/5
  third_shift_participation := 1/10
  total_participation := 6/25

-- Theorem statement
theorem second_shift_size (c : Company) (h1 : c = company_x) :
  c.second_shift = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_shift_size_l1177_117750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1177_117785

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)

-- Define the function g
def g (x : ℝ) : ℝ := x^2 + 2

-- Theorem statement
theorem problem_solution :
  (f 2 = 1/3) ∧ (g 2 = 6) ∧ (f (g 3) = 1/12) := by
  constructor
  · -- Prove f 2 = 1/3
    simp [f]
    norm_num
  · constructor
    · -- Prove g 2 = 6
      simp [g]
      norm_num
    · -- Prove f (g 3) = 1/12
      simp [f, g]
      norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1177_117785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_intersection_length_l1177_117768

/-- Represents a trapezoid ABCD with bases a and b -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h_a_gt_b : a > b

/-- Represents the points M and N on the diagonals of the trapezoid -/
def intersectionPoints (t : Trapezoid) : ℝ → ℝ → Prop :=
  λ m n ↦ ∃ (k : ℝ), k = t.a / 2 ∧ 
    ∃ (x y : ℝ), x ∈ Set.Icc 0 t.a ∧ y ∈ Set.Icc 0 t.b ∧
    m = x * k / t.a ∧ n = y * k / t.b

/-- The main theorem: length of MN in the trapezoid -/
theorem trapezoid_intersection_length (t : Trapezoid) (m n : ℝ) 
  (h : intersectionPoints t m n) : 
  ∃ (mn : ℝ), mn = t.a * t.b / (t.a + 2 * t.b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_intersection_length_l1177_117768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_20_plus_cot_10_equals_csc_20_l1177_117787

theorem tan_20_plus_cot_10_equals_csc_20 :
  Real.tan (20 * π / 180) + Real.tan (π / 2 - 10 * π / 180) = 1 / Real.sin (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_20_plus_cot_10_equals_csc_20_l1177_117787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_33_l1177_117775

-- Define the expression
noncomputable def expression : ℝ := (1/2)^(-5 : ℤ) + Real.log 2 + Real.log 5

-- State the theorem
theorem expression_equals_33 : expression = 33 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_33_l1177_117775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_special_divisors_l1177_117736

def isDivisor (d n : ℕ) : Bool := n % d = 0

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d => isDivisor d n)

def differences (l : List ℕ) : List ℕ :=
  List.zipWith (·-·) (l.tail) l

def isArithmeticSequence (l : List ℕ) : Prop :=
  let diffs := differences l
  ∀ i j, i < diffs.length → j < diffs.length → diffs.get! i = diffs.get! j

theorem unique_number_with_special_divisors :
  ∃! n : ℕ, n > 3 ∧
    (divisors n).length > 3 ∧
    isArithmeticSequence (divisors n) ∧
    n = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_special_divisors_l1177_117736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l1177_117704

theorem angle_problem (α β : Real) 
  (h1 : 0 < α)
  (h2 : α < π/2)
  (h3 : π/2 < β)
  (h4 : β < π)
  (h5 : Real.tan (α/2) = 1/2)
  (h6 : Real.cos (β - α) = Real.sqrt 2/10) :
  Real.sin α = 4/5 ∧ β = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l1177_117704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_division_exists_l1177_117761

/-- Represents a cell in the figure --/
structure Cell where

/-- Represents a corner in the figure --/
structure Corner where
  cells : List Cell

/-- Represents the entire figure --/
structure Figure where
  cells : List Cell

/-- Predicate to check if a corner is a three-cell corner --/
def isThreeCellCorner (c : Corner) : Bool :=
  c.cells.length = 3

/-- Predicate to check if a corner is a four-cell corner --/
def isFourCellCorner (c : Corner) : Bool :=
  c.cells.length = 4

/-- Predicate to check if a list of corners is a valid division of the figure --/
def isValidDivision (f : Figure) (corners : List Corner) : Prop :=
  (corners.filter isThreeCellCorner).length = 2 ∧
  (∀ c ∈ corners, isThreeCellCorner c ∨ isFourCellCorner c) ∧
  (corners.map Corner.cells).join.length = f.cells.length

/-- Theorem stating that there exists a valid division of the figure --/
theorem figure_division_exists (f : Figure) : ∃ (corners : List Corner), isValidDivision f corners := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_division_exists_l1177_117761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_equals_two_thirds_l1177_117793

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (3 : ℝ)^x else 1 - Real.sqrt x

-- Theorem statement
theorem f_composition_negative_two_equals_two_thirds :
  f (f (-2)) = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_equals_two_thirds_l1177_117793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l1177_117737

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem sinusoidal_function_properties
  (A ω φ : ℝ)
  (h1 : A > 0)
  (h2 : ω > 0)
  (h3 : 0 < φ ∧ φ < Real.pi / 2)
  (h4 : ∃ x : ℝ, f A ω φ x = 0)
  (h5 : ∀ x y : ℝ, f A ω φ x = 0 → f A ω φ y = 0 → x ≠ y → |x - y| = Real.pi / 2)
  (h6 : f A ω φ (2 * Real.pi / 3) = -2) :
  (A = 2 ∧ ω = 2 ∧ φ = Real.pi / 6) ∧
  (∀ k : ℤ, ∃ c : ℝ, c = Real.pi * k / 2 - Real.pi / 12 ∧ 
    (∀ x : ℝ, f A ω φ (2 * x + c) = -f A ω φ (-2 * x + c))) ∧
  (∀ k : ℤ, ∀ x y : ℝ, 
    -Real.pi / 3 + Real.pi * k ≤ x ∧ x < y ∧ y ≤ Real.pi / 6 + Real.pi * k → 
    f A ω φ x < f A ω φ y) ∧
  (∀ x : ℝ, Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 2 → 
    -1 ≤ f A ω φ x ∧ f A ω φ x ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l1177_117737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_l1177_117730

def satisfies_conditions (n : ℕ) : Prop :=
  ¬(2 ∣ n) ∧ ¬(3 ∣ n) ∧
  ∀ a b : ℕ, (2^a : ℤ) - (3^b : ℤ) ≠ n ∧ (3^b : ℤ) - (2^a : ℤ) ≠ n

theorem smallest_satisfying_number :
  satisfies_conditions 35 ∧
  ∀ m : ℕ, m < 35 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_l1177_117730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_factor_of_polynomials_l1177_117748

-- Define the polynomials
def p₁ (x : ℝ) : ℝ := x^3 + x^2
def p₂ (x : ℝ) : ℝ := x^2 + 2*x + 1
def p₃ (x : ℝ) : ℝ := x^2 - 1

-- Define the common factor
def common_factor (x : ℝ) : ℝ := x + 1

-- Helper function to check if a polynomial is divisible by (x + 1)
def is_divisible_by_x_plus_one (p : ℝ → ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, p x = (x + 1) * q x

-- Theorem statement
theorem common_factor_of_polynomials :
  is_divisible_by_x_plus_one p₁ ∧
  is_divisible_by_x_plus_one p₂ ∧
  is_divisible_by_x_plus_one p₃ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_factor_of_polynomials_l1177_117748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_mp_nq_is_60_degrees_l1177_117794

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop :=
  let d_AB := ((t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2)
  let d_BC := ((t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2)
  let d_CA := ((t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2)
  d_AB = d_BC ∧ d_BC = d_CA

/-- Checks if a point is on a line segment -/
def is_on_segment (P : Point) (A : Point) (B : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P.x = A.x + t * (B.x - A.x) ∧ P.y = A.y + t * (B.y - A.y)

/-- Calculates the distance between two points -/
noncomputable def distance (P : Point) (Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

/-- Calculates the angle between two lines -/
noncomputable def angle_between_lines (P1 P2 Q1 Q2 : Point) : ℝ := 
  sorry

/-- Main theorem -/
theorem angle_between_mp_nq_is_60_degrees 
  (ABC : Triangle) (M N P Q : Point) 
  (h_equilateral : is_equilateral ABC)
  (h_M_on_AB : is_on_segment M ABC.A ABC.B)
  (h_N_on_AB : is_on_segment N ABC.A ABC.B)
  (h_P_on_BC : is_on_segment P ABC.B ABC.C)
  (h_Q_on_CA : is_on_segment Q ABC.C ABC.A)
  (h_lengths : distance M ABC.A + distance ABC.A Q = 
               distance N ABC.B + distance ABC.B P ∧
               distance N ABC.B + distance ABC.B P = 
               distance ABC.A ABC.B) :
  angle_between_lines M P N Q = 60 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_mp_nq_is_60_degrees_l1177_117794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_condition_l1177_117788

theorem acute_triangle_condition (x : ℝ) : 
  (x > 0) →
  (x^2 + 6 > x^2 + 4) →
  (x^2 + 4 > 4*x) →
  (((x^2 + 4)^2 + (4*x)^2 - (x^2 + 6)^2) / (2 * 4*x * (x^2 + 4)) > 0) →
  (x > Real.sqrt (5/3)) := by
  sorry

#check acute_triangle_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_condition_l1177_117788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_tangent_l1177_117708

/-- The value of m for which the parabola y = x^2 + 4 and the hyperbola y^2 - mx^2 = 4 are tangent -/
noncomputable def tangent_value : ℝ := 8 + 4 * Real.sqrt 3

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 4

/-- Hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m * x^2 = 4

/-- Theorem stating that if the parabola and hyperbola are tangent, then m equals the tangent_value -/
theorem parabola_hyperbola_tangent :
  ∀ m : ℝ, (∃ x y : ℝ, parabola x y ∧ hyperbola m x y ∧
    (∀ x' y' : ℝ, parabola x' y' ∧ hyperbola m x' y' → (x = x' ∧ y = y'))) →
  m = tangent_value := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_tangent_l1177_117708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1177_117791

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1177_117791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1177_117732

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^3 else -x^3

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (f (3*a - 1) ≥ 8 * f a) ↔ (a ≤ 1/5 ∨ a ≥ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1177_117732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stall_problem_l1177_117784

theorem stall_problem (area_diff : ℝ) (cost_A cost_B : ℝ) (ratio : ℚ) (total_stalls : ℕ) (min_ratio : ℕ) :
  area_diff = 2 →
  cost_A = 40 →
  cost_B = 30 →
  ratio = 3/5 →
  total_stalls = 90 →
  min_ratio = 3 →
  ∃ (area_A area_B : ℝ) (max_cost : ℝ),
    area_A = area_B + area_diff ∧
    (60 / area_A) = (60 / area_B) * (ratio : ℝ) ∧
    area_A = 5 ∧
    area_B = 3 ∧
    max_cost = 10520 ∧
    ∀ (num_A : ℕ),
      num_A ≤ total_stalls ∧
      (total_stalls - num_A) ≥ min_ratio * num_A →
      num_A * area_A * cost_A + (total_stalls - num_A) * area_B * cost_B ≤ max_cost :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stall_problem_l1177_117784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_range_l1177_117782

noncomputable section

-- Define the ellipse and line
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def line (x y : ℝ) : Prop := x + y = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_major_axis_range (a b : ℝ) (P Q : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  ellipse a b P.1 P.2 ∧
  ellipse a b Q.1 Q.2 ∧
  line P.1 P.2 ∧
  line Q.1 Q.2 ∧
  (P.1 * Q.1 + P.2 * Q.2 = 0) ∧
  (Real.sqrt 3 / 3 ≤ eccentricity a b ∧ eccentricity a b ≤ Real.sqrt 2 / 2) →
  Real.sqrt 5 ≤ 2 * a ∧ 2 * a ≤ Real.sqrt 6 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_range_l1177_117782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_making_cost_is_56_l1177_117770

/-- Represents the cost calculation for making ice cubes --/
noncomputable def ice_making_cost 
  (total_weight : ℝ)
  (cube_weight : ℝ)
  (water_per_cube : ℝ)
  (cubes_per_hour : ℝ)
  (maker_cost_per_hour : ℝ)
  (water_cost_per_ounce : ℝ) : ℝ :=
  let num_cubes := total_weight / cube_weight
  let hours_required := num_cubes / cubes_per_hour
  let maker_cost := hours_required * maker_cost_per_hour
  let water_needed := num_cubes * water_per_cube
  let water_cost := water_needed * water_cost_per_ounce
  maker_cost + water_cost

/-- Theorem stating that the cost of making 10 pounds of ice cubes is $56 --/
theorem ice_making_cost_is_56 :
  ice_making_cost 10 (1/16) 2 10 1.5 0.1 = 56 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_making_cost_is_56_l1177_117770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_even_and_decreasing_l1177_117757

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 2)

-- State the theorem
theorem cos_half_even_and_decreasing :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_even_and_decreasing_l1177_117757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l1177_117778

theorem binomial_expansion_coefficient :
  let a : ℤ := 3
  let b : ℤ := 2
  let n : ℕ := 6
  let r : ℕ := 3
  Nat.choose n r * a^(n - r) * (-b)^r = -4320 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l1177_117778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1177_117731

noncomputable section

open Real

def g (x : ℝ) : ℝ := x / log x

def f (a : ℝ) (x : ℝ) : ℝ := g x - a * x

theorem problem_statement :
  (∀ x > e, MonotoneOn g (Set.Ioi e)) ∧
  (∀ x ∈ Set.Ioo 0 1 ∪ Set.Ioo 1 e, StrictAntiOn g (Set.Ioo 0 1 ∪ Set.Ioo 1 e)) ∧
  (∃ a_min : ℝ, a_min = 1/4 ∧ ∀ a ≥ a_min, StrictAntiOn (f a) (Set.Ioi 1)) ∧
  (∃ a_range : Set ℝ, a_range = Set.Ici (e^2 / 2 - 1/4) ∧
    ∀ a ∈ a_range, ∀ x₁ ∈ Set.Icc e (e^2),
      ∃ x₂ ∈ Set.Icc e (e^2), g x₁ ≤ deriv (f a) x₂ + 2*a) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1177_117731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_democrat_ratio_l1177_117710

theorem female_democrat_ratio (total_participants : ℕ) 
  (male_democrat_ratio : ℚ) (total_democrat_ratio : ℚ) 
  (female_democrats : ℕ) 
  (h1 : total_participants = 990)
  (h2 : male_democrat_ratio = 1/4)
  (h3 : total_democrat_ratio = 1/3)
  (h4 : female_democrats = 165)
  : (female_democrats : ℚ) / ((total_participants : ℚ) - 
    (male_democrat_ratio * total_democrat_ratio * total_participants)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_democrat_ratio_l1177_117710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equivalence_l1177_117701

theorem cos_sin_equivalence : ∀ x : ℝ, 
  Real.cos (2 * (x - π / 12)) = Real.sin (2 * x + 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equivalence_l1177_117701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_distance_l1177_117765

/-- Calculates the one-way distance of a round trip given total time, outbound speed, and return speed -/
noncomputable def calculate_distance (total_time : ℝ) (outbound_speed : ℝ) (return_speed : ℝ) : ℝ :=
  (total_time * outbound_speed * return_speed) / (outbound_speed + return_speed)

/-- Theorem: Given a round trip with total time of 3 hours, outbound speed of 16 mph,
    and return speed of 24 mph, the one-way distance is 28.8 miles -/
theorem round_trip_distance :
  calculate_distance 3 16 24 = 28.8 := by
  -- Unfold the definition of calculate_distance
  unfold calculate_distance
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_distance_l1177_117765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_satisfying_inequality_l1177_117740

-- Define the nested radical function
noncomputable def nestedRadical (a : ℝ) : ℝ := (1 + Real.sqrt (1 + 4*a)) / 2

-- Define the nested fraction function
noncomputable def nestedFraction (a : ℝ) : ℝ := (a + Real.sqrt (a^2 + 4)) / 2

-- Define the inequality function
def inequalityFunction (a : ℝ) : Prop :=
  nestedRadical a - (2 / nestedFraction a) > 7

-- State the theorem
theorem smallest_integer_satisfying_inequality :
  ∀ n : ℕ, n > 0 ∧ n < 43 → ¬(inequalityFunction (n : ℝ)) ∧ inequalityFunction 43 := by
  sorry

#check smallest_integer_satisfying_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_satisfying_inequality_l1177_117740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_count_l1177_117756

theorem congruence_solutions_count : 
  let solutions := {x : ℕ | x < 100 ∧ (x + 17) % 46 = 73 % 46}
  Finset.card (Finset.filter (fun x => x < 100 ∧ (x + 17) % 46 = 73 % 46) (Finset.range 100)) = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_count_l1177_117756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_purchase_l1177_117716

/-- Represents the price of an item in kopecks -/
def ItemPrice (p : ℕ) : Prop := ∃ (a : ℕ), p = 100 * a + 99

/-- The total cost of Kolya's purchase in kopecks -/
def TotalCost : ℕ := 20083

/-- Represents the number of items Kolya could have bought -/
def ValidPurchase (n : ℕ) : Prop := 
  ∃ (p : ℕ), ItemPrice p ∧ n * p = TotalCost

theorem kolya_purchase : 
  {n : ℕ | ValidPurchase n} = {17, 117} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_purchase_l1177_117716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1177_117733

theorem remainder_theorem (n : ℕ) (h : n % 10 = 7) : n % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1177_117733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unfair_die_expected_value_l1177_117722

/-- An unfair six-sided die with specific probabilities -/
structure UnfairDie where
  sides : Nat
  prob_six : ℚ
  prob_other : ℚ

/-- The expected value of rolling the unfair die -/
noncomputable def expected_value (d : UnfairDie) : ℚ :=
  d.prob_six * 6 + (1 - d.prob_six) * (1 + 2 + 3 + 4 + 5) / 5

/-- Theorem: The expected value of rolling the specific unfair die is 4.5 -/
theorem unfair_die_expected_value :
  let d : UnfairDie := ⟨6, 1/2, 1/10⟩
  expected_value d = 9/2 := by
  sorry

#eval (9 : ℚ) / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unfair_die_expected_value_l1177_117722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_a_beats_b_l1177_117762

-- Define the runners and their speeds
noncomputable def runner_a_speed : ℝ := 224 / 28
noncomputable def runner_b_speed : ℝ := 224 / 32

-- Define the time b takes to run 224 meters
def time_b : ℝ := 32

-- Define the distance a runs in the same time
noncomputable def distance_a : ℝ := runner_a_speed * time_b

-- Define the distance b runs
def distance_b : ℝ := 224

-- Theorem statement
theorem distance_a_beats_b : distance_a - distance_b = 32 := by
  -- Expand definitions
  unfold distance_a
  unfold runner_a_speed
  -- Perform algebraic manipulations
  simp [mul_div_assoc, mul_comm]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_a_beats_b_l1177_117762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_lower_bound_existence_l1177_117767

-- Define the concept of a lower bound
def is_lower_bound (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x, f x ≥ M

-- Define the concept of a greatest lower bound
def has_greatest_lower_bound (f : ℝ → ℝ) : Prop :=
  ∃ M, is_lower_bound f M ∧ ∀ N, is_lower_bound f N → N ≤ M

-- Define the piecewise function
noncomputable def piecewise (x : ℝ) : ℝ :=
  if x > 0 then 1 else if x = 0 then 0 else -1

theorem greatest_lower_bound_existence :
  (has_greatest_lower_bound Real.sin) ∧
  (¬ has_greatest_lower_bound Real.log) ∧
  (has_greatest_lower_bound Real.exp) ∧
  (has_greatest_lower_bound piecewise) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_lower_bound_existence_l1177_117767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_of_lines_l1177_117777

/-- The slope of the angle bisector of two lines with slopes m₁ and m₂ -/
noncomputable def angle_bisector_slope (m₁ m₂ : ℝ) : ℝ :=
  (m₁ + m₂ + Real.sqrt (m₁^2 + m₂^2 + 2*m₁*m₂)) / (1 - m₁*m₂)

/-- The acute angle bisector slope is the positive root -/
noncomputable def acute_angle_bisector_slope (m₁ m₂ : ℝ) : ℝ :=
  if angle_bisector_slope m₁ m₂ > 0 then angle_bisector_slope m₁ m₂
  else -angle_bisector_slope m₁ m₂

theorem angle_bisector_slope_of_lines (m₁ m₂ : ℝ) 
  (h₁ : m₁ = 2) (h₂ : m₂ = 4) :
  acute_angle_bisector_slope m₁ m₂ = -12/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_of_lines_l1177_117777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_hexagon_intersection_area_l1177_117712

/-- The area of a regular hexagon with side length s -/
noncomputable def area_of_regular_hexagon (s : ℝ) : ℝ := 
  (3 * Real.sqrt 3 / 2) * s^2

/-- The area of the regular hexagon formed by intersecting an octahedron of edge length a with a plane -/
noncomputable def area_of_regular_hexagon_from_octahedron_intersection (a : ℝ) : ℝ :=
  area_of_regular_hexagon (a / 2)

/-- The area of a regular hexagon formed by intersecting an octahedron with a plane -/
theorem octahedron_hexagon_intersection_area (a : ℝ) (h : a > 0) :
  ∃ (A : ℝ), A = (3 * Real.sqrt 3 / 8) * a^2 ∧ 
  A = area_of_regular_hexagon_from_octahedron_intersection a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_hexagon_intersection_area_l1177_117712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_single_point_l1177_117779

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => λ x => Real.sqrt (2 - x)
  | (n + 1) => λ x => f n (Real.sqrt ((n + 1)^2 + 1 - x))

-- Define the domain of a function
def domain (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | ∃ y, f x = y}

-- State the theorem
theorem f_domain_single_point :
  (∃ N : ℕ, 
    (∀ n > N, domain (f n) = ∅) ∧
    (domain (f N) ≠ ∅) ∧
    (∃ c : ℝ, domain (f N) = {c})) ∧
  (let N := 5
   let c := 2
   (∀ n > N, domain (f n) = ∅) ∧
   domain (f N) = {c}) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_single_point_l1177_117779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1177_117742

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- State the theorem
theorem function_symmetry 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_bound : |φ| < π/2) 
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) 
  (h_odd : ∀ x, f ω φ (x + π/12) = -f ω φ (-x + π/12)) :
  ∀ x, f ω φ (5*π/6 + x) = f ω φ (5*π/6 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1177_117742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planeAngle_correct_l1177_117797

/-- Represents a pyramid with an isosceles triangular base -/
structure IsoscelesPyramid where
  /-- The acute angle between the equal sides of the base triangle -/
  α : ℝ
  /-- The angle between lateral edges and the base plane -/
  β : ℝ
  /-- Assumption that α is an acute angle -/
  h_α_acute : 0 < α ∧ α < Real.pi / 2
  /-- Assumption that β is a positive angle less than π/2 -/
  h_β_pos : 0 < β ∧ β < Real.pi / 2

/-- 
The angle between a plane drawn through the side opposite to α 
and the midpoint of the pyramid's height, and the base plane 
-/
noncomputable def planeAngle (p : IsoscelesPyramid) : ℝ :=
  Real.arctan ((Real.tan p.β) / (2 * Real.cos p.α))

theorem planeAngle_correct (p : IsoscelesPyramid) : 
  planeAngle p = Real.arctan ((Real.tan p.β) / (2 * Real.cos p.α)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planeAngle_correct_l1177_117797
