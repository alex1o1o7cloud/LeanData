import Mathlib

namespace NUMINAMATH_CALUDE_non_vegan_gluten_cupcakes_l3435_343513

/-- Given a set of cupcakes with specific properties, prove that the number of non-vegan cupcakes containing gluten is 28. -/
theorem non_vegan_gluten_cupcakes
  (total : ℕ)
  (gluten_free : ℕ)
  (vegan : ℕ)
  (vegan_gluten_free : ℕ)
  (h1 : total = 80)
  (h2 : gluten_free = total / 2)
  (h3 : vegan = 24)
  (h4 : vegan_gluten_free = vegan / 2)
  : total - gluten_free - (vegan - vegan_gluten_free) = 28 := by
  sorry

#check non_vegan_gluten_cupcakes

end NUMINAMATH_CALUDE_non_vegan_gluten_cupcakes_l3435_343513


namespace NUMINAMATH_CALUDE_elberta_amount_l3435_343599

/-- The amount of money Granny Smith has -/
def granny_smith : ℕ := 120

/-- The amount of money Anjou has -/
def anjou : ℕ := granny_smith / 2

/-- The amount of money Elberta has -/
def elberta : ℕ := anjou + 5

/-- Theorem stating that Elberta has $65 -/
theorem elberta_amount : elberta = 65 := by
  sorry

end NUMINAMATH_CALUDE_elberta_amount_l3435_343599


namespace NUMINAMATH_CALUDE_angle_expression_value_l3435_343575

/-- Given that point P(1,2) is on the terminal side of angle α, 
    prove that (6sinα + 8cosα) / (3sinα - 2cosα) = 5 -/
theorem angle_expression_value (α : Real) (h : Complex.exp (α * Complex.I) = ⟨1, 2⟩) :
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_value_l3435_343575


namespace NUMINAMATH_CALUDE_quartic_roots_polynomial_problem_l3435_343518

theorem quartic_roots_polynomial_problem (a b c d : ℂ) (P : ℂ → ℂ) :
  (a^4 + 4*a^3 + 6*a^2 + 8*a + 10 = 0) →
  (b^4 + 4*b^3 + 6*b^2 + 8*b + 10 = 0) →
  (c^4 + 4*c^3 + 6*c^2 + 8*c + 10 = 0) →
  (d^4 + 4*d^3 + 6*d^2 + 8*d + 10 = 0) →
  (P a = b + c + d) →
  (P b = a + c + d) →
  (P c = a + b + d) →
  (P d = a + b + c) →
  (P (a + b + c + d) = -20) →
  (∀ x, P x = -10/37*x^4 - 30/37*x^3 - 56/37*x^2 - 118/37*x - 148/37) :=
by sorry

end NUMINAMATH_CALUDE_quartic_roots_polynomial_problem_l3435_343518


namespace NUMINAMATH_CALUDE_coinciding_rest_days_count_l3435_343594

/-- Chris's schedule cycle length -/
def chris_cycle : ℕ := 6

/-- Dana's schedule cycle length -/
def dana_cycle : ℕ := 6

/-- Number of days Chris works in his cycle -/
def chris_work_days : ℕ := 4

/-- Number of days Dana works in her cycle -/
def dana_work_days : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 1200

/-- The number of times Chris and Dana have rest-days on the same day -/
def coinciding_rest_days : ℕ := total_days / chris_cycle

theorem coinciding_rest_days_count :
  coinciding_rest_days = 200 :=
sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_count_l3435_343594


namespace NUMINAMATH_CALUDE_marcy_water_amount_l3435_343528

/-- The amount of water Marcy keeps by her desk -/
def water_amount (sip_interval : ℕ) (sip_volume : ℕ) (total_time : ℕ) : ℚ :=
  (total_time / sip_interval * sip_volume : ℚ) / 1000

/-- Theorem stating that Marcy keeps 2 liters of water by her desk -/
theorem marcy_water_amount :
  water_amount 5 40 250 = 2 := by
  sorry

#eval water_amount 5 40 250

end NUMINAMATH_CALUDE_marcy_water_amount_l3435_343528


namespace NUMINAMATH_CALUDE_not_A_implies_not_all_mc_or_not_three_math_l3435_343529

-- Define the predicates
def got_all_mc_right (student : String) : Prop := sorry
def solved_at_least_three_math (student : String) : Prop := sorry
def received_A (student : String) : Prop := sorry

-- Ms. Carroll's rule
axiom ms_carroll_rule (student : String) :
  got_all_mc_right student ∧ solved_at_least_three_math student → received_A student

-- Theorem to prove
theorem not_A_implies_not_all_mc_or_not_three_math (student : String) :
  ¬(received_A student) → ¬(got_all_mc_right student) ∨ ¬(solved_at_least_three_math student) :=
by sorry

end NUMINAMATH_CALUDE_not_A_implies_not_all_mc_or_not_three_math_l3435_343529


namespace NUMINAMATH_CALUDE_square_greater_than_abs_l3435_343514

theorem square_greater_than_abs (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_abs_l3435_343514


namespace NUMINAMATH_CALUDE_incircle_radius_given_tangent_circles_l3435_343543

-- Define the triangle and circles
structure Triangle :=
  (A B C : Point)

structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the property of being tangent
def is_tangent (c1 c2 : Circle) : Prop := sorry

-- Define the property of being inside a triangle
def is_inside (c : Circle) (t : Triangle) : Prop := sorry

-- Define the incircle of a triangle
def incircle (t : Triangle) : Circle := sorry

-- Main theorem
theorem incircle_radius_given_tangent_circles 
  (t : Triangle) (k : Circle) (k1 k2 k3 : Circle) :
  k = incircle t →
  is_inside k1 t ∧ is_inside k2 t ∧ is_inside k3 t →
  is_tangent k k1 ∧ is_tangent k k2 ∧ is_tangent k k3 →
  k1.radius = 1 ∧ k2.radius = 4 ∧ k3.radius = 9 →
  k.radius = 11 := by
  sorry

end NUMINAMATH_CALUDE_incircle_radius_given_tangent_circles_l3435_343543


namespace NUMINAMATH_CALUDE_circle_equation_l3435_343578

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := y = x - 1

-- Define the line l₂
def l₂ (x : ℝ) : Prop := x = -1

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 6*x + 1 = 0

-- Define the circle equation
def circle_eq (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

theorem circle_equation 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : l₁ x₁ y₁) 
  (h₂ : l₁ x₂ y₂) 
  (h₃ : quadratic_eq x₁) 
  (h₄ : quadratic_eq x₂) :
  ∃ (a b r : ℝ), 
    (circle_eq x₁ y₁ a b r ∧ 
     circle_eq x₂ y₂ a b r ∧ 
     (a = 3 ∧ b = 2 ∧ r = 4) ∨ 
     (a = 11 ∧ b = -6 ∧ r = 12)) ∧
    ∀ (x : ℝ), l₂ x → (x - a)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3435_343578


namespace NUMINAMATH_CALUDE_cow_milk_production_l3435_343559

/-- Given two groups of cows with different efficiencies, calculate the milk production of the second group based on the first group's rate. -/
theorem cow_milk_production
  (a b c d e f g : ℝ)
  (h₁ : a > 0)
  (h₂ : c > 0)
  (h₃ : f > 0) :
  let rate := b / (a * c * f)
  let second_group_production := d * rate * g * e
  second_group_production = b * d * e * g / (a * c * f) :=
by sorry

end NUMINAMATH_CALUDE_cow_milk_production_l3435_343559


namespace NUMINAMATH_CALUDE_number_representation_l3435_343539

/-- Represents a number in terms of millions, ten thousands, and thousands -/
structure NumberComposition :=
  (millions : ℕ)
  (ten_thousands : ℕ)
  (thousands : ℕ)

/-- Converts a NumberComposition to its standard integer representation -/
def to_standard (n : NumberComposition) : ℕ :=
  n.millions * 1000000 + n.ten_thousands * 10000 + n.thousands * 1000

/-- Converts a natural number to its representation in ten thousands -/
def to_ten_thousands (n : ℕ) : ℚ :=
  (n : ℚ) / 10000

theorem number_representation (n : NumberComposition) 
  (h : n = ⟨6, 3, 4⟩) : 
  to_standard n = 6034000 ∧ to_ten_thousands (to_standard n) = 603.4 := by
  sorry

end NUMINAMATH_CALUDE_number_representation_l3435_343539


namespace NUMINAMATH_CALUDE_circles_intersect_distance_between_centers_l3435_343562

/-- Given two circles M and N, prove that they intersect --/
theorem circles_intersect : ∀ (a : ℝ),
  a > 0 →
  (∃ (x y : ℝ), x^2 + y^2 - 2*a*y = 0 ∧ x + y = 0 ∧ (x - (-x))^2 = 4) →
  a = Real.sqrt 2 →
  ∃ (x y : ℝ), 
    x^2 + (y - a)^2 = a^2 ∧
    (x - 1)^2 + (y - 1)^2 = 1 :=
by
  sorry

/-- The distance between the centers of the circles is between |R-r| and R+r --/
theorem distance_between_centers (a : ℝ) (h : a = Real.sqrt 2) :
  Real.sqrt 2 - 1 < Real.sqrt (1 + (Real.sqrt 2 - 1)^2) ∧
  Real.sqrt (1 + (Real.sqrt 2 - 1)^2) < Real.sqrt 2 + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_distance_between_centers_l3435_343562


namespace NUMINAMATH_CALUDE_inequalities_proof_l3435_343590

theorem inequalities_proof (a b : ℝ) (h1 : b > a) (h2 : a * b > 0) :
  (1 / a > 1 / b) ∧ (a + b < 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3435_343590


namespace NUMINAMATH_CALUDE_extreme_points_inequality_l3435_343532

/-- The function f(x) = x - a/x - 2ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - 2 * Real.log x

/-- Predicate to check if x is an extreme point of f -/
def is_extreme_point (a : ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → f a y ≠ f a x

theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  is_extreme_point a x₁ →
  is_extreme_point a x₂ →
  f a x₂ < x₂ - 1 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_inequality_l3435_343532


namespace NUMINAMATH_CALUDE_triangle_area_l3435_343551

/-- The area of a triangle with base 4 and height 5 is 10 -/
theorem triangle_area : 
  ∀ (base height area : ℝ), 
  base = 4 → 
  height = 5 → 
  area = (base * height) / 2 → 
  area = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3435_343551


namespace NUMINAMATH_CALUDE_set_inclusion_iff_a_range_l3435_343555

/-- The set A -/
def A : Set ℝ := {x | -2 < x ∧ x < 3}

/-- The set B -/
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}

/-- The set C parameterized by a -/
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

/-- The main theorem stating the equivalence between C being a subset of (A ∩ ℝ\B) and the range of a -/
theorem set_inclusion_iff_a_range :
  ∀ a : ℝ, (C a ⊆ (A ∩ (Set.univ \ B))) ↔ (0 < a ∧ a ≤ 2/3) :=
sorry

end NUMINAMATH_CALUDE_set_inclusion_iff_a_range_l3435_343555


namespace NUMINAMATH_CALUDE_quadratic_equation_k_l3435_343561

/-- The equation (k-1)x^(|k|+1)-x+5=0 is quadratic in x -/
def is_quadratic (k : ℝ) : Prop :=
  (k - 1 ≠ 0) ∧ (|k| + 1 = 2)

theorem quadratic_equation_k (k : ℝ) :
  is_quadratic k → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_l3435_343561


namespace NUMINAMATH_CALUDE_zero_subset_M_l3435_343501

-- Define the set M
def M : Set ℝ := {x | x > -2}

-- State the theorem
theorem zero_subset_M : {0} ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_zero_subset_M_l3435_343501


namespace NUMINAMATH_CALUDE_concentric_circles_angle_l3435_343587

theorem concentric_circles_angle (r₁ r₂ r₃ : ℝ) (shaded_area unshaded_area : ℝ) (θ : ℝ) : 
  r₁ = 4 →
  r₂ = 3 →
  r₃ = 2 →
  shaded_area = (3/4) * unshaded_area →
  shaded_area + unshaded_area = 29 * π →
  shaded_area = 11 * θ + 9 * π →
  θ = 6 * π / 77 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_angle_l3435_343587


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l3435_343506

def f (x : ℝ) := x^3 - 9

theorem root_exists_in_interval :
  ∃ (x : ℝ), x ∈ Set.Ioo 2 3 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l3435_343506


namespace NUMINAMATH_CALUDE_checkerboard_square_selection_l3435_343581

theorem checkerboard_square_selection (b : ℕ) : 
  let n := 2 * b + 1
  (n^2 * (n - 1)) / 2 = n * (n - 1) * n / 2 - n * (n - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_checkerboard_square_selection_l3435_343581


namespace NUMINAMATH_CALUDE_cross_in_square_side_length_l3435_343538

/-- Represents a cross shape inside a square -/
structure CrossInSquare where
  a : ℝ  -- Side length of the largest square
  area_cross : ℝ -- Area of the cross

/-- The area of the cross is equal to the sum of areas of its component squares -/
def cross_area_equation (c : CrossInSquare) : Prop :=
  c.area_cross = 2 * (c.a / 2)^2 + 2 * (c.a / 4)^2

/-- Theorem stating that if the area of the cross is 810 cm², then the side length of the largest square is 36 cm -/
theorem cross_in_square_side_length 
  (c : CrossInSquare) 
  (h1 : c.area_cross = 810) 
  (h2 : cross_area_equation c) : 
  c.a = 36 := by
sorry

end NUMINAMATH_CALUDE_cross_in_square_side_length_l3435_343538


namespace NUMINAMATH_CALUDE_g_domain_l3435_343512

def f_domain : Set ℝ := Set.Icc 0 2

def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 1)

theorem g_domain (f : ℝ → ℝ) (h : ∀ x, x ∈ f_domain ↔ f x ≠ 0) :
  ∀ x, x ∈ Set.Icc (-1) 1 ↔ g f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_g_domain_l3435_343512


namespace NUMINAMATH_CALUDE_oprah_car_collection_reduction_l3435_343571

/-- The number of years required to reduce a car collection -/
def years_to_reduce (initial_cars : ℕ) (target_cars : ℕ) (cars_per_year : ℕ) : ℕ :=
  (initial_cars - target_cars) / cars_per_year

/-- Theorem: It takes 60 years to reduce Oprah's car collection from 3500 to 500 cars -/
theorem oprah_car_collection_reduction :
  years_to_reduce 3500 500 50 = 60 := by
  sorry

end NUMINAMATH_CALUDE_oprah_car_collection_reduction_l3435_343571


namespace NUMINAMATH_CALUDE_box_requires_130_cubes_l3435_343503

/-- Calculates the minimum number of cubes required to build a box -/
def min_cubes_for_box (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height + cube_volume - 1) / cube_volume

/-- Proves that a box with dimensions 10 cm × 13 cm × 5 cm requires 130 cubes of 5 cubic cm each -/
theorem box_requires_130_cubes :
  min_cubes_for_box 10 13 5 5 = 130 := by
  sorry

#eval min_cubes_for_box 10 13 5 5

end NUMINAMATH_CALUDE_box_requires_130_cubes_l3435_343503


namespace NUMINAMATH_CALUDE_gym_attendance_l3435_343519

theorem gym_attendance (initial : ℕ) 
  (h1 : initial + 5 - 2 = 19) : initial = 16 := by
  sorry

end NUMINAMATH_CALUDE_gym_attendance_l3435_343519


namespace NUMINAMATH_CALUDE_point_symmetry_false_l3435_343547

/-- Two points in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry about x-axis -/
def symmetricAboutXAxis (p1 p2 : Point2D) : Prop :=
  p1.x = p2.x ∧ p1.y = -p2.y

/-- The main theorem -/
theorem point_symmetry_false : 
  ¬ symmetricAboutXAxis ⟨-3, -4⟩ ⟨3, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_false_l3435_343547


namespace NUMINAMATH_CALUDE_binary_digit_difference_l3435_343524

/-- Returns the number of digits in the base-2 representation of a natural number -/
def binaryDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

/-- The difference in the number of digits between the base-2 representations of 1200 and 200 is 3 -/
theorem binary_digit_difference : binaryDigits 1200 - binaryDigits 200 = 3 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l3435_343524


namespace NUMINAMATH_CALUDE_harrietts_pennies_l3435_343563

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | "penny" => 1
  | _ => 0

/-- The problem statement -/
theorem harrietts_pennies :
  let quarters := 10
  let dimes := 3
  let nickels := 3
  let total_cents := 300  -- $3 in cents
  let other_coins_value := 
    quarters * coin_value "quarter" + 
    dimes * coin_value "dime" + 
    nickels * coin_value "nickel"
  let pennies := total_cents - other_coins_value
  pennies = 5 := by sorry

end NUMINAMATH_CALUDE_harrietts_pennies_l3435_343563


namespace NUMINAMATH_CALUDE_walk_time_to_school_l3435_343549

/-- Represents Maria's travel to school -/
structure SchoolTravel where
  walkSpeed : ℝ
  skateSpeed : ℝ
  distance : ℝ

/-- The conditions of Maria's travel -/
def travelConditions (t : SchoolTravel) : Prop :=
  t.distance = 25 * t.walkSpeed + 13 * t.skateSpeed ∧
  t.distance = 11 * t.walkSpeed + 20 * t.skateSpeed

/-- The theorem to prove -/
theorem walk_time_to_school (t : SchoolTravel) 
  (h : travelConditions t) : t.distance / t.walkSpeed = 51 := by
  sorry

end NUMINAMATH_CALUDE_walk_time_to_school_l3435_343549


namespace NUMINAMATH_CALUDE_wheel_marking_theorem_l3435_343569

theorem wheel_marking_theorem :
  ∃ (R : ℝ), R > 0 ∧
    ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 360 →
      ∃ (n : ℕ), ∃ (k : ℤ),
        n / (2 * π * R) = θ / 360 + k ∧
        0 ≤ n / (2 * π * R) - k ∧
        n / (2 * π * R) - k < 1 / 360 :=
by sorry

end NUMINAMATH_CALUDE_wheel_marking_theorem_l3435_343569


namespace NUMINAMATH_CALUDE_total_days_2010_to_2015_l3435_343533

def is_leap_year (year : ℕ) : Bool :=
  year = 2012

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def years_in_period : List ℕ := [2010, 2011, 2012, 2013, 2014, 2015]

theorem total_days_2010_to_2015 :
  (years_in_period.map days_in_year).sum = 2191 := by
  sorry

end NUMINAMATH_CALUDE_total_days_2010_to_2015_l3435_343533


namespace NUMINAMATH_CALUDE_ribbon_length_difference_l3435_343516

/-- Proves that the difference in ribbon length between two wrapping methods
    for a box matches one side of the box. -/
theorem ribbon_length_difference (l w h bow : ℕ) 
  (hl : l = 22) (hw : w = 22) (hh : h = 11) (hbow : bow = 24) :
  (2 * l + 4 * w + 2 * h + bow) - (2 * l + 2 * w + 4 * h + bow) = l := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_difference_l3435_343516


namespace NUMINAMATH_CALUDE_stone_game_loser_l3435_343510

/-- Represents a pile of stones -/
structure Pile :=
  (count : Nat)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)
  (currentPlayer : Nat)

/-- Defines a valid move in the game -/
def validMove (state : GameState) : Prop :=
  ∃ (p : Pile) (n : Nat), p ∈ state.piles ∧ 1 ≤ n ∧ n < p.count

/-- The initial game state -/
def initialState : GameState :=
  { piles := [⟨6⟩, ⟨8⟩, ⟨8⟩, ⟨9⟩], currentPlayer := 1 }

/-- The number of players -/
def numPlayers : Nat := 5

/-- The losing player -/
def losingPlayer : Nat := 3

theorem stone_game_loser :
  ¬∃ (moves : Nat), 
    (moves + initialState.piles.length = (initialState.piles.map Pile.count).sum) ∧
    (moves % numPlayers + 1 = losingPlayer) ∧
    (∀ (state : GameState), state.piles.length ≤ moves + initialState.piles.length → validMove state) :=
sorry

end NUMINAMATH_CALUDE_stone_game_loser_l3435_343510


namespace NUMINAMATH_CALUDE_relationship_a_ab_ab_squared_l3435_343583

theorem relationship_a_ab_ab_squared (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end NUMINAMATH_CALUDE_relationship_a_ab_ab_squared_l3435_343583


namespace NUMINAMATH_CALUDE_indeterminate_equation_solution_l3435_343534

theorem indeterminate_equation_solution (a b : ℤ) :
  ∃ (x y z u v w t : ℤ),
    x^4 + y^4 + z^4 = u^2 + v^2 + w^2 + t^2 ∧
    x = a ∧
    y = b ∧
    z = a + b ∧
    u = a^2 + a*b + b^2 ∧
    v = a*b ∧
    w = a*b*(a + b) ∧
    t = b*(a + b) := by
  sorry

end NUMINAMATH_CALUDE_indeterminate_equation_solution_l3435_343534


namespace NUMINAMATH_CALUDE_contest_scores_mode_and_median_l3435_343507

def scores : List ℕ := [91, 95, 89, 93, 88, 94, 95]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem contest_scores_mode_and_median :
  mode scores = 95 ∧ median scores = 93 := by sorry

end NUMINAMATH_CALUDE_contest_scores_mode_and_median_l3435_343507


namespace NUMINAMATH_CALUDE_absolute_difference_of_solution_l3435_343523

theorem absolute_difference_of_solution (x y : ℝ) : 
  (Int.floor x + (y - Int.floor y) = 3.7) →
  ((x - Int.floor x) + Int.floor y = 6.2) →
  |x - y| = 3.5 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_of_solution_l3435_343523


namespace NUMINAMATH_CALUDE_unique_solution_l3435_343517

theorem unique_solution : ∃! (x y : ℕ+), 
  (x : ℝ)^(y : ℝ) + 3 = (y : ℝ)^(x : ℝ) ∧ 
  3 * (x : ℝ)^(y : ℝ) = (y : ℝ)^(x : ℝ) + 13 ∧
  x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3435_343517


namespace NUMINAMATH_CALUDE_treewidth_bound_for_grid_free_graphs_l3435_343589

/-- A k-grid of order h in a graph -/
def kGridOfOrderH (G : Graph) (k h : ℕ) : Prop := sorry

/-- The treewidth of a graph -/
def treewidth (G : Graph) : ℕ := sorry

/-- Theorem: If a graph G does not contain a k-grid of order h, then its treewidth is less than h + k - 1 -/
theorem treewidth_bound_for_grid_free_graphs
  (G : Graph) (h k : ℕ) (h_ge_k : h ≥ k) (k_ge_1 : k ≥ 1)
  (no_grid : ¬ kGridOfOrderH G k h) :
  treewidth G < h + k - 1 := by
  sorry

end NUMINAMATH_CALUDE_treewidth_bound_for_grid_free_graphs_l3435_343589


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3435_343553

theorem min_value_sum_squares (x y z a : ℝ) (h : x + 2*y + 3*z = a) :
  x^2 + y^2 + z^2 ≥ a^2 / 14 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3435_343553


namespace NUMINAMATH_CALUDE_truck_capacity_problem_l3435_343531

/-- The capacity of a large truck in tons -/
def large_truck_capacity : ℝ := sorry

/-- The capacity of a small truck in tons -/
def small_truck_capacity : ℝ := sorry

/-- The total capacity of a given number of large and small trucks -/
def total_capacity (large_trucks small_trucks : ℕ) : ℝ :=
  (large_trucks : ℝ) * large_truck_capacity + (small_trucks : ℝ) * small_truck_capacity

theorem truck_capacity_problem :
  total_capacity 3 4 = 22 ∧ total_capacity 5 2 = 25 →
  total_capacity 4 3 = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_truck_capacity_problem_l3435_343531


namespace NUMINAMATH_CALUDE_total_sides_is_118_l3435_343564

/-- The number of sides for each shape --/
def sides_of_shape (shape : String) : ℕ :=
  match shape with
  | "triangle" => 3
  | "square" => 4
  | "pentagon" => 5
  | "hexagon" => 6
  | "heptagon" => 7
  | "octagon" => 8
  | "nonagon" => 9
  | "hendecagon" => 11
  | "circle" => 0
  | _ => 0

/-- The count of each shape in the top layer --/
def top_layer : List (String × ℕ) :=
  [("triangle", 6), ("nonagon", 1), ("heptagon", 2)]

/-- The count of each shape in the middle layer --/
def middle_layer : List (String × ℕ) :=
  [("square", 4), ("hexagon", 2), ("hendecagon", 1)]

/-- The count of each shape in the bottom layer --/
def bottom_layer : List (String × ℕ) :=
  [("octagon", 3), ("circle", 5), ("pentagon", 1), ("nonagon", 1)]

/-- Calculate the total number of sides for a given layer --/
def total_sides_in_layer (layer : List (String × ℕ)) : ℕ :=
  layer.foldl (fun acc (shape, count) => acc + count * sides_of_shape shape) 0

/-- The main theorem stating that the total number of sides is 118 --/
theorem total_sides_is_118 :
  total_sides_in_layer top_layer +
  total_sides_in_layer middle_layer +
  total_sides_in_layer bottom_layer = 118 := by
  sorry

end NUMINAMATH_CALUDE_total_sides_is_118_l3435_343564


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l3435_343582

theorem angle_sum_in_circle (x : ℝ) : 
  4 * x + 5 * x + x + 2 * x = 360 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l3435_343582


namespace NUMINAMATH_CALUDE_ratio_problem_l3435_343545

theorem ratio_problem (a b c x y : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : b / c = 3 / x)
  (h3 : a / c = y / 4)
  (h4 : x = 1) : y = 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3435_343545


namespace NUMINAMATH_CALUDE_equal_distances_l3435_343542

/-- Represents a right triangle with squares on its sides -/
structure RightTriangleWithSquares where
  -- The lengths of the sides of the right triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The acute angle α
  α : ℝ
  -- Conditions
  right_triangle : c^2 = a^2 + b^2
  acute_angle : 0 < α ∧ α < π / 2
  angle_sum : α + (π / 2 - α) = π / 2

/-- The theorem stating that the distances O₁O₂ and CO₃ are equal -/
theorem equal_distances (t : RightTriangleWithSquares) : 
  (t.a + t.b) / Real.sqrt 2 = t.c / Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_equal_distances_l3435_343542


namespace NUMINAMATH_CALUDE_cookie_ratio_proof_l3435_343530

theorem cookie_ratio_proof (raisin_cookies oatmeal_cookies : ℕ) : 
  raisin_cookies = 42 → 
  raisin_cookies + oatmeal_cookies = 49 → 
  raisin_cookies / oatmeal_cookies = 6 := by
sorry

end NUMINAMATH_CALUDE_cookie_ratio_proof_l3435_343530


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3435_343593

/-- The eccentricity of a hyperbola given its equation and a point it passes through -/
theorem hyperbola_eccentricity (m : ℝ) (h : 2 - 4 / m = 1) : 
  Real.sqrt (1 + 4 / 2) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3435_343593


namespace NUMINAMATH_CALUDE_octahedron_non_prime_sum_pairs_l3435_343588

-- Define the type for die faces
def DieFace := Fin 8

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := sorry

-- Define a function to get the value on a die face
def faceValue (face : DieFace) : ℕ := face.val + 1

-- Define a function to check if the sum of two face values is not prime
def sumNotPrime (face1 face2 : DieFace) : Prop :=
  ¬(isPrime (faceValue face1 + faceValue face2))

-- The main theorem
theorem octahedron_non_prime_sum_pairs :
  ∃ (pairs : Finset (DieFace × DieFace)),
    pairs.card = 8 ∧
    (∀ (pair : DieFace × DieFace), pair ∈ pairs → sumNotPrime pair.1 pair.2) ∧
    (∀ (face1 face2 : DieFace), 
      face1 ≠ face2 → sumNotPrime face1 face2 → 
      (face1, face2) ∈ pairs ∨ (face2, face1) ∈ pairs) :=
sorry

end NUMINAMATH_CALUDE_octahedron_non_prime_sum_pairs_l3435_343588


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3435_343592

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  17 * x^2 - 16 * x * y + 4 * y^2 - 34 * x + 16 * y + 13 = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (1, 1)
def point_B : ℝ × ℝ := (1, -1)

-- Define the center
def center : ℝ × ℝ := (1, 0)

-- Define the conjugate axis equations
def conjugate_axis_eq (x y : ℝ) : Prop :=
  y = (13 + 5 * Real.sqrt 17) / 16 * (x - 1) ∨
  y = (13 - 5 * Real.sqrt 17) / 16 * (x - 1)

theorem hyperbola_properties :
  (hyperbola_eq point_A.1 point_A.2) ∧
  (hyperbola_eq point_B.1 point_B.2) →
  (∃ (x y : ℝ), hyperbola_eq x y ∧ conjugate_axis_eq x y) ∧
  (center.1 = 1 ∧ center.2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3435_343592


namespace NUMINAMATH_CALUDE_vacation_cost_l3435_343546

/-- 
If dividing a total cost among 5 people results in a per-person cost that is $120 more than 
dividing the same total cost among 8 people, then the total cost is $1600.
-/
theorem vacation_cost (total_cost : ℝ) : 
  (total_cost / 5 - total_cost / 8 = 120) → total_cost = 1600 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l3435_343546


namespace NUMINAMATH_CALUDE_seating_arrangement_count_l3435_343509

/-- The number of seats at the bus station -/
def total_seats : ℕ := 10

/-- The number of passengers -/
def num_passengers : ℕ := 4

/-- The number of consecutive empty seats required -/
def consecutive_empty_seats : ℕ := 5

/-- Calculate the number of ways to arrange seating -/
def seating_arrangements (total : ℕ) (passengers : ℕ) (empty_block : ℕ) : ℕ :=
  (Nat.factorial passengers) * (Nat.factorial (total - passengers - empty_block + 1) / Nat.factorial (total - passengers - empty_block - 1))

theorem seating_arrangement_count : 
  seating_arrangements total_seats num_passengers consecutive_empty_seats = 480 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_count_l3435_343509


namespace NUMINAMATH_CALUDE_first_valid_row_count_l3435_343511

def is_valid_arrangement (total_trees : ℕ) (num_rows : ℕ) : Prop :=
  num_rows > 0 ∧ total_trees % num_rows = 0

theorem first_valid_row_count : 
  let total_trees := 84
  ∀ (n : ℕ), n > 0 → is_valid_arrangement total_trees n →
    (is_valid_arrangement total_trees 6 ∧
     is_valid_arrangement total_trees 4) →
    2 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_first_valid_row_count_l3435_343511


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3435_343515

/-- Given a triangle ABC with angle B = 45°, side c = 2√2, and side b = 4√3/3,
    prove that angle A is either 7π/12 or π/12 -/
theorem triangle_angle_A (A B C : ℝ) (a b c : ℝ) : 
  B = π/4 → c = 2 * Real.sqrt 2 → b = 4 * Real.sqrt 3 / 3 →
  A = 7*π/12 ∨ A = π/12 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3435_343515


namespace NUMINAMATH_CALUDE_triangle_properties_l3435_343552

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Add conditions here
  c = Real.sqrt 2 ∧
  Real.cos C = 3/4 ∧
  2 * c * Real.sin A = b * Real.sin C

-- State the theorem
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : 
  b = 2 ∧ 
  Real.sin A = Real.sqrt 14 / 8 ∧ 
  Real.sin (2 * A + π/3) = (5 * Real.sqrt 7 + 9 * Real.sqrt 3) / 32 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3435_343552


namespace NUMINAMATH_CALUDE_largest_gold_coins_distribution_l3435_343598

theorem largest_gold_coins_distribution (n : ℕ) : 
  n > 50 ∧ n < 150 ∧ 
  ∃ (k : ℕ), n = 7 * k + 2 ∧
  (∀ m : ℕ, m > n → ¬(∃ j : ℕ, m = 7 * j + 2)) →
  n = 149 := by
sorry

end NUMINAMATH_CALUDE_largest_gold_coins_distribution_l3435_343598


namespace NUMINAMATH_CALUDE_triangle_area_determines_p_l3435_343537

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, p),
    if the area of triangle ABC is 36, then p = 12.75 -/
theorem triangle_area_determines_p :
  ∀ p : ℝ,
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  triangle_area = 36 → p = 12.75 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_determines_p_l3435_343537


namespace NUMINAMATH_CALUDE_puppy_weight_l3435_343572

theorem puppy_weight (puppy smaller_kitten larger_kitten : ℝ)
  (total_weight : puppy + smaller_kitten + larger_kitten = 30)
  (weight_comparison1 : puppy + larger_kitten = 3 * smaller_kitten)
  (weight_comparison2 : puppy + smaller_kitten = larger_kitten) :
  puppy = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_l3435_343572


namespace NUMINAMATH_CALUDE_consecutive_sum_property_l3435_343527

theorem consecutive_sum_property : ∃ (a : Fin 10 → ℝ),
  (∀ i : Fin 6, (a i) + (a (i+1)) + (a (i+2)) + (a (i+3)) + (a (i+4)) > 0) ∧
  (∀ j : Fin 4, (a j) + (a (j+1)) + (a (j+2)) + (a (j+3)) + (a (j+4)) + (a (j+5)) + (a (j+6)) < 0) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_property_l3435_343527


namespace NUMINAMATH_CALUDE_max_value_quadratic_function_l3435_343540

theorem max_value_quadratic_function (f : ℝ → ℝ) (h : ∀ x ∈ (Set.Ioo 0 1), f x = x * (1 - x)) :
  ∃ x ∈ (Set.Ioo 0 1), ∀ y ∈ (Set.Ioo 0 1), f x ≥ f y ∧ f x = 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_function_l3435_343540


namespace NUMINAMATH_CALUDE_amys_chicken_soup_cans_l3435_343526

/-- Amy's soup purchase problem -/
theorem amys_chicken_soup_cans (total_soups : ℕ) (tomato_soup_cans : ℕ) (chicken_soup_cans : ℕ) :
  total_soups = 9 →
  tomato_soup_cans = 3 →
  total_soups = tomato_soup_cans + chicken_soup_cans →
  chicken_soup_cans = 6 := by
  sorry

end NUMINAMATH_CALUDE_amys_chicken_soup_cans_l3435_343526


namespace NUMINAMATH_CALUDE_cubic_identity_l3435_343596

theorem cubic_identity (a b c : ℝ) : 
  (b + c - 2 * a)^3 + (c + a - 2 * b)^3 + (a + b - 2 * c)^3 = 
  (b + c - 2 * a) * (c + a - 2 * b) * (a + b - 2 * c) := by sorry

end NUMINAMATH_CALUDE_cubic_identity_l3435_343596


namespace NUMINAMATH_CALUDE_acute_angles_equal_positive_angles_less_than_90_l3435_343536

-- Define the sets A and D
def A : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}
def D : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}

-- Theorem statement
theorem acute_angles_equal_positive_angles_less_than_90 : A = D := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_equal_positive_angles_less_than_90_l3435_343536


namespace NUMINAMATH_CALUDE_x_range_equivalence_l3435_343525

theorem x_range_equivalence (x : ℝ) : 
  (∀ a b : ℝ, a > 0 → b > 0 → x^2 + 2*x < a/b + 16*b/a) ↔ x > -4 ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_x_range_equivalence_l3435_343525


namespace NUMINAMATH_CALUDE_max_sum_constrained_l3435_343586

theorem max_sum_constrained (x y : ℝ) : 
  x^2 + y^2 = 100 → xy = 40 → x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_constrained_l3435_343586


namespace NUMINAMATH_CALUDE_college_students_count_l3435_343584

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 200) :
  boys + girls = 520 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l3435_343584


namespace NUMINAMATH_CALUDE_petes_walking_distance_l3435_343500

/-- Represents a pedometer with a maximum count --/
structure Pedometer where
  max_count : ℕ
  reset_count : ℕ
  final_reading : ℕ

/-- Calculates the total steps based on pedometer data --/
def total_steps (p : Pedometer) : ℕ :=
  p.reset_count * (p.max_count + 1) + p.final_reading

/-- Converts steps to miles --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

theorem petes_walking_distance (p : Pedometer) (steps_per_mile : ℕ) :
  p.max_count = 99999 ∧
  p.reset_count = 38 ∧
  p.final_reading = 75000 ∧
  steps_per_mile = 1800 →
  steps_to_miles (total_steps p) steps_per_mile = 2150 := by
  sorry

#eval steps_to_miles (total_steps { max_count := 99999, reset_count := 38, final_reading := 75000 }) 1800

end NUMINAMATH_CALUDE_petes_walking_distance_l3435_343500


namespace NUMINAMATH_CALUDE_M_equals_N_l3435_343591

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt x}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Theorem statement
theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l3435_343591


namespace NUMINAMATH_CALUDE_max_table_height_value_l3435_343576

/-- Triangle DEF with sides a, b, c -/
structure Triangle (a b c : ℝ) : Type :=
  (side_a : a > 0)
  (side_b : b > 0)
  (side_c : c > 0)

/-- The maximum table height function -/
def maxTableHeight (t : Triangle 25 29 32) : ℝ := sorry

/-- Theorem stating the maximum table height -/
theorem max_table_height_value (t : Triangle 25 29 32) : 
  maxTableHeight t = 64 * Real.sqrt 29106 / 1425 := by sorry

end NUMINAMATH_CALUDE_max_table_height_value_l3435_343576


namespace NUMINAMATH_CALUDE_min_x_squared_isosceles_trapezoid_l3435_343566

/-- Represents a trapezoid ABCD with specific properties -/
structure IsoscelesTrapezoid where
  -- Length of base AB
  ab : ℝ
  -- Length of base CD
  cd : ℝ
  -- Length of side AD (equal to BC)
  x : ℝ
  -- Ensures the trapezoid is isosceles
  isIsosceles : ad = bc
  -- Ensures a circle with center on AB is tangent to AD and BC
  hasTangentCircle : ∃ (center : ℝ), 0 ≤ center ∧ center ≤ ab ∧
    ∃ (radius : ℝ), radius > 0 ∧
    (center - radius)^2 + x^2 = (ab/2)^2 ∧
    (center + radius)^2 + x^2 = (ab/2)^2

/-- The theorem stating the minimum value of x^2 for the given trapezoid -/
theorem min_x_squared_isosceles_trapezoid (t : IsoscelesTrapezoid)
  (h1 : t.ab = 50)
  (h2 : t.cd = 14) :
  ∃ (m : ℝ), m^2 = 800 ∧ ∀ (y : ℝ), t.x = y → y^2 ≥ m^2 := by
  sorry

end NUMINAMATH_CALUDE_min_x_squared_isosceles_trapezoid_l3435_343566


namespace NUMINAMATH_CALUDE_total_odd_green_and_red_marbles_l3435_343568

/-- Represents a person's marble collection --/
structure MarbleCollection where
  green : Nat
  red : Nat
  blue : Nat

/-- Counts odd numbers of green and red marbles --/
def countOddGreenAndRed (mc : MarbleCollection) : Nat :=
  (if mc.green % 2 = 1 then mc.green else 0) +
  (if mc.red % 2 = 1 then mc.red else 0)

theorem total_odd_green_and_red_marbles :
  let sara := MarbleCollection.mk 3 5 6
  let tom := MarbleCollection.mk 4 7 2
  let lisa := MarbleCollection.mk 5 3 7
  countOddGreenAndRed sara + countOddGreenAndRed tom + countOddGreenAndRed lisa = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_odd_green_and_red_marbles_l3435_343568


namespace NUMINAMATH_CALUDE_h_increasing_implies_k_range_l3435_343585

def h (k : ℝ) (x : ℝ) : ℝ := 2 * x - k

theorem h_increasing_implies_k_range (k : ℝ) :
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → h k x₁ < h k x₂) →
  k ∈ Set.Ici (-2) :=
sorry

end NUMINAMATH_CALUDE_h_increasing_implies_k_range_l3435_343585


namespace NUMINAMATH_CALUDE_greatest_integer_gcd_eighteen_l3435_343544

theorem greatest_integer_gcd_eighteen : ∃ n : ℕ, n < 200 ∧ n.gcd 18 = 6 ∧ ∀ m : ℕ, m < 200 → m.gcd 18 = 6 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_gcd_eighteen_l3435_343544


namespace NUMINAMATH_CALUDE_function_inequality_l3435_343554

/-- Given a > 0 and a continuous function f: (0, +∞) → ℝ satisfying certain conditions,
    prove that f(x)f(y) ≤ f(xy) for all x, y > 0 -/
theorem function_inequality (a : ℝ) (f : ℝ → ℝ) (h_a : a > 0) 
    (h_cont : Continuous f) (h_dom : ∀ x, x > 0 → f x ≠ 0)
    (h_fa : f a = 1)
    (h_ineq : ∀ x y, x > 0 → y > 0 → f x * f y + f (a / x) * f (a / y) ≤ 2 * f (x * y)) :
  ∀ x y, x > 0 → y > 0 → f x * f y ≤ f (x * y) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3435_343554


namespace NUMINAMATH_CALUDE_number_of_b_objects_l3435_343548

theorem number_of_b_objects (total : ℕ) (a : ℕ) (b : ℕ) : 
  total = 35 →
  total = a + b →
  a = 17 →
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_number_of_b_objects_l3435_343548


namespace NUMINAMATH_CALUDE_tangent_and_perpendicular_l3435_343535

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

-- Define the given line
def given_line (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3*x + y + 2 = 0

theorem tangent_and_perpendicular :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve f
    f x₀ = y₀ ∧
    -- The tangent line passes through (x₀, y₀)
    tangent_line x₀ y₀ ∧
    -- The slope of the tangent line at (x₀, y₀) is f'(x₀)
    (3 : ℝ) = -f' x₀ ∧
    -- The given line and tangent line are perpendicular
    (2 : ℝ) * (3 : ℝ) = -(6 : ℝ) * (1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_tangent_and_perpendicular_l3435_343535


namespace NUMINAMATH_CALUDE_derivative_zero_implies_x_equals_plus_minus_a_l3435_343557

theorem derivative_zero_implies_x_equals_plus_minus_a (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := fun x ↦ (x^2 + a^2) / x
  let f' : ℝ → ℝ := fun x ↦ (x^2 - a^2) / x^2
  ∀ x₀ : ℝ, x₀ ≠ 0 → f' x₀ = 0 → x₀ = a ∨ x₀ = -a := by
  sorry

end NUMINAMATH_CALUDE_derivative_zero_implies_x_equals_plus_minus_a_l3435_343557


namespace NUMINAMATH_CALUDE_negation_equivalence_l3435_343570

theorem negation_equivalence :
  (¬ ∀ (a b : ℝ), a + b = 0 → a^2 + b^2 = 0) ↔
  (∃ (a b : ℝ), a + b = 0 ∧ a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3435_343570


namespace NUMINAMATH_CALUDE_quadratic_properties_l3435_343556

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the roots of the equation
def are_roots (a b c x₁ x₂ : ℝ) : Prop :=
  quadratic_equation a b c x₁ ∧ quadratic_equation a b c x₂

theorem quadratic_properties
  (a b c x₁ x₂ : ℝ) (ha : a ≠ 0) (h_roots : are_roots a b c x₁ x₂) :
  (¬ (∃ z : ℂ, x₁ = z ∧ x₂ = z ∧ z.im ≠ 0)) ∧
  (∀ x, a * x^2 + b * x + c = a * (x - x₁) * (x - x₂)) ∧
  (x₁^2 * x₂ + x₁ * x₂^2 = -b * c / a^2) ∧
  (b^2 - 4*a*c < 0 → ∃ y : ℝ, x₁ - x₂ = Complex.I * y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3435_343556


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l3435_343550

theorem green_shirt_pairs
  (blue_count : ℕ)
  (yellow_count : ℕ)
  (green_count : ℕ)
  (total_students : ℕ)
  (total_pairs : ℕ)
  (blue_blue_pairs : ℕ)
  (h1 : blue_count = 70)
  (h2 : yellow_count = 80)
  (h3 : green_count = 50)
  (h4 : total_students = blue_count + yellow_count + green_count)
  (h5 : total_pairs = 100)
  (h6 : blue_blue_pairs = 30)
  (h7 : total_students = total_pairs * 2) :
  (green_count / 2 : ℕ) = 25 := by
sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l3435_343550


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l3435_343520

/-- Represents a triangular grid figure made of toothpicks -/
structure TriangularGrid where
  total_toothpicks : ℕ
  total_triangles : ℕ
  horizontal_toothpicks : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (grid : TriangularGrid) : ℕ :=
  grid.horizontal_toothpicks

/-- Theorem: For a specific triangular grid, the minimum number of toothpicks 
    to remove to eliminate all triangles is 15 -/
theorem min_toothpicks_removal (grid : TriangularGrid) 
    (h1 : grid.total_toothpicks = 40)
    (h2 : grid.total_triangles > 35)
    (h3 : grid.horizontal_toothpicks = 15) : 
  min_toothpicks_to_remove grid = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_toothpicks_removal_l3435_343520


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l3435_343558

theorem no_function_satisfies_conditions : ¬∃ f : ℝ → ℝ, 
  (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧ 
  (∀ x : ℝ, f (f x) = (x - 1) * f x + 2) := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l3435_343558


namespace NUMINAMATH_CALUDE_train_length_calculation_l3435_343579

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (time_to_pass : ℝ) : 
  train_speed = 21 → 
  bridge_length = 130 → 
  time_to_pass = 142.2857142857143 → 
  ∃ (train_length : ℝ), (abs (train_length - 700) < 0.1) ∧ 
    (train_length + bridge_length = train_speed * (1000 / 3600) * time_to_pass) := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3435_343579


namespace NUMINAMATH_CALUDE_multiple_y_solutions_l3435_343505

theorem multiple_y_solutions : ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧
  (∃ (x₁ : ℝ), x₁^2 + y₁^2 - 10 = 0 ∧ x₁^2 - x₁*y₁ - 3*y₁ + 12 = 0) ∧
  (∃ (x₂ : ℝ), x₂^2 + y₂^2 - 10 = 0 ∧ x₂^2 - x₂*y₂ - 3*y₂ + 12 = 0) :=
by sorry

end NUMINAMATH_CALUDE_multiple_y_solutions_l3435_343505


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3435_343567

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (4 - x^2)}
def B : Set ℝ := {y | y > 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3435_343567


namespace NUMINAMATH_CALUDE_directrix_of_symmetrical_parabola_l3435_343577

-- Define the original parabola
def original_parabola (x y : ℝ) : Prop := y = 2 * x^2

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := y = x

-- Define the symmetrical parabola
def symmetrical_parabola (x y : ℝ) : Prop := y^2 = (1/2) * x

-- Theorem statement
theorem directrix_of_symmetrical_parabola :
  ∀ (x : ℝ), (∃ (y : ℝ), symmetrical_parabola x y) → (x = -1/8) = 
  (∀ (p : ℝ), p ≠ 0 → (∃ (h k : ℝ), ∀ (x y : ℝ), 
    symmetrical_parabola x y ↔ (y - k)^2 = 4 * p * (x - h) ∧ x = h - p)) :=
by sorry

end NUMINAMATH_CALUDE_directrix_of_symmetrical_parabola_l3435_343577


namespace NUMINAMATH_CALUDE_circle_equation_l3435_343502

/-- A circle with center on y = -2x, passing through (2, -1), and tangent to x + y = 1 -/
theorem circle_equation : ∀ a : ℝ,
  (∀ x y : ℝ, y = -2 * x → (x - a)^2 + (y + 2*a)^2 = (2 - a)^2 + (-1 + 2*a)^2) →
  ((a - 1)^2 + (2*a + 1)^2 = 2) →
  (∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 2 ↔ 
    (x - a)^2 + (y + 2*a)^2 = (2 - a)^2 + (-1 + 2*a)^2 ∧ 
    ((x + y - 1) / Real.sqrt 2)^2 = (2 - a)^2 + (-1 + 2*a)^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3435_343502


namespace NUMINAMATH_CALUDE_only_cat_owners_count_l3435_343595

/-- The number of people owning only cats in a pet ownership scenario. -/
def num_only_cat_owners : ℕ := 
  let total_pet_owners : ℕ := 59
  let only_dog_owners : ℕ := 15
  let cat_and_dog_owners : ℕ := 5
  let cat_dog_snake_owners : ℕ := 3
  total_pet_owners - (only_dog_owners + cat_and_dog_owners + cat_dog_snake_owners)

/-- Theorem stating that the number of people owning only cats is 36. -/
theorem only_cat_owners_count : num_only_cat_owners = 36 := by
  sorry

end NUMINAMATH_CALUDE_only_cat_owners_count_l3435_343595


namespace NUMINAMATH_CALUDE_play_role_assignment_l3435_343565

def number_of_assignments (men women : ℕ) (male_roles female_roles either_roles : ℕ) : ℕ :=
  men * women * (Nat.choose (men + women - male_roles - female_roles) either_roles)

theorem play_role_assignment :
  number_of_assignments 4 7 1 1 4 = 3528 := by
  sorry

end NUMINAMATH_CALUDE_play_role_assignment_l3435_343565


namespace NUMINAMATH_CALUDE_total_interest_received_l3435_343508

/-- Calculate simple interest -/
def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

/-- Calculate total interest from two loans -/
def total_interest (principal1 principal2 rate time1 time2 : ℕ) : ℕ :=
  simple_interest principal1 rate time1 + simple_interest principal2 rate time2

/-- Theorem stating the total interest received by A -/
theorem total_interest_received : 
  total_interest 5000 3000 12 2 4 = 2440 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_received_l3435_343508


namespace NUMINAMATH_CALUDE_cube_roots_of_primes_not_in_arithmetic_progression_l3435_343521

theorem cube_roots_of_primes_not_in_arithmetic_progression 
  (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ¬∃ (a d : ℝ), {(p : ℝ)^(1/3), (q : ℝ)^(1/3), (r : ℝ)^(1/3)} ⊆ {a + n * d | n : ℤ} :=
by sorry

end NUMINAMATH_CALUDE_cube_roots_of_primes_not_in_arithmetic_progression_l3435_343521


namespace NUMINAMATH_CALUDE_smallest_addend_to_palindrome_l3435_343541

/-- A function that checks if a positive integer is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The smallest positive integer that can be added to 2002 to produce a larger palindrome -/
def smallestAddend : ℕ := 110

theorem smallest_addend_to_palindrome : 
  (isPalindrome 2002) ∧ 
  (isPalindrome (2002 + smallestAddend)) ∧ 
  (∀ k : ℕ, k < smallestAddend → ¬ isPalindrome (2002 + k)) := by sorry

end NUMINAMATH_CALUDE_smallest_addend_to_palindrome_l3435_343541


namespace NUMINAMATH_CALUDE_no_integer_solution_x4_plus_6_eq_y3_l3435_343560

theorem no_integer_solution_x4_plus_6_eq_y3 :
  ∀ (x y : ℤ), (x^4 + 6) % 13 ≠ y^3 % 13 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_x4_plus_6_eq_y3_l3435_343560


namespace NUMINAMATH_CALUDE_geometric_series_problem_l3435_343573

theorem geometric_series_problem (a₁ : ℝ) (r₁ r₂ : ℝ) (m : ℝ) :
  a₁ = 15 →
  a₁ * r₁ = 9 →
  a₁ * r₂ = 9 + m →
  a₁ / (1 - r₂) = 3 * (a₁ / (1 - r₁)) →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l3435_343573


namespace NUMINAMATH_CALUDE_expression_simplification_l3435_343580

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 - 3) : 
  (a^2 - 4*a + 4) / (a^2 - 4) / ((a - 2) / (a^2 + 2*a)) + 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3435_343580


namespace NUMINAMATH_CALUDE_music_books_cost_l3435_343597

/-- Calculates the amount spent on music books including tax --/
def amount_spent_on_music_books (total_budget : ℝ) (math_book_price : ℝ) (math_book_count : ℕ) 
  (math_book_discount : ℝ) (science_book_price : ℝ) (art_book_price : ℝ) (art_book_tax : ℝ) 
  (music_book_tax : ℝ) : ℝ :=
  let math_books_cost := math_book_count * math_book_price * (1 - math_book_discount)
  let science_books_cost := (math_book_count + 6) * science_book_price
  let art_books_cost := 2 * math_book_count * art_book_price * (1 + art_book_tax)
  let remaining_budget := total_budget - (math_books_cost + science_books_cost + art_books_cost)
  remaining_budget

/-- Theorem stating that the amount spent on music books including tax is $160 --/
theorem music_books_cost (total_budget : ℝ) (math_book_price : ℝ) (math_book_count : ℕ) 
  (math_book_discount : ℝ) (science_book_price : ℝ) (art_book_price : ℝ) (art_book_tax : ℝ) 
  (music_book_tax : ℝ) :
  total_budget = 500 ∧ 
  math_book_price = 20 ∧ 
  math_book_count = 4 ∧ 
  math_book_discount = 0.1 ∧ 
  science_book_price = 10 ∧ 
  art_book_price = 20 ∧ 
  art_book_tax = 0.05 ∧ 
  music_book_tax = 0.07 → 
  amount_spent_on_music_books total_budget math_book_price math_book_count math_book_discount 
    science_book_price art_book_price art_book_tax music_book_tax = 160 := by
  sorry


end NUMINAMATH_CALUDE_music_books_cost_l3435_343597


namespace NUMINAMATH_CALUDE_percentage_of_non_science_majors_l3435_343522

theorem percentage_of_non_science_majors
  (women_science_percentage : Real)
  (men_class_percentage : Real)
  (men_science_percentage : Real)
  (h1 : women_science_percentage = 0.1)
  (h2 : men_class_percentage = 0.4)
  (h3 : men_science_percentage = 0.8500000000000001) :
  1 - (women_science_percentage * (1 - men_class_percentage) +
       men_science_percentage * men_class_percentage) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_non_science_majors_l3435_343522


namespace NUMINAMATH_CALUDE_negative_two_times_inequality_l3435_343504

theorem negative_two_times_inequality {a b : ℝ} (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_two_times_inequality_l3435_343504


namespace NUMINAMATH_CALUDE_max_tulips_is_15_l3435_343574

/-- Represents the cost of a yellow tulip in rubles -/
def yellow_cost : ℕ := 50

/-- Represents the cost of a red tulip in rubles -/
def red_cost : ℕ := 31

/-- Represents the maximum budget in rubles -/
def max_budget : ℕ := 600

/-- Represents a valid bouquet of tulips -/
structure Bouquet where
  yellow : ℕ
  red : ℕ
  odd_total : (yellow + red) % 2 = 1
  color_diff : (yellow = red + 1) ∨ (red = yellow + 1)
  within_budget : yellow * yellow_cost + red * red_cost ≤ max_budget

/-- The maximum number of tulips in a bouquet -/
def max_tulips : ℕ := 15

/-- Theorem stating that the maximum number of tulips in a valid bouquet is 15 -/
theorem max_tulips_is_15 : 
  ∀ b : Bouquet, b.yellow + b.red ≤ max_tulips ∧ 
  ∃ b' : Bouquet, b'.yellow + b'.red = max_tulips :=
sorry

end NUMINAMATH_CALUDE_max_tulips_is_15_l3435_343574
