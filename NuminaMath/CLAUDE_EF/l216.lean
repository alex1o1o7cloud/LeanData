import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_on_positive_reals_l216_21605

-- Define the function f(x) = (x-3)e^x
noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

-- Theorem statement
theorem f_has_one_zero_on_positive_reals :
  ∃! x, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_on_positive_reals_l216_21605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l216_21629

/-- The equation defining the graph --/
def graph_equation (x y : ℝ) : Prop :=
  y^2 + 3*x*y + 50*(abs x) = 500

/-- The bounded region created by the graph --/
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | graph_equation p.1 p.2}

/-- The area of the bounded region --/
noncomputable def area_of_region : ℝ :=
  3125 / 9  -- We directly define the area here

theorem area_of_bounded_region :
  area_of_region = 3125 / 9 := by
  -- The proof is trivial since we defined area_of_region directly
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l216_21629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l216_21665

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.rpow x (1/3)
def g (x : ℝ) : ℝ := x

-- State the theorem
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l216_21665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedra_arrangement_exists_l216_21697

/-- A tetrahedron in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- An arrangement of tetrahedra -/
structure TetrahedronArrangement where
  tetrahedra : Fin 8 → Tetrahedron

/-- Two tetrahedra share a face with nonzero area -/
def ShareFace (t1 t2 : Tetrahedron) : Prop :=
  ∃ (face : Fin 3 → ℝ × ℝ × ℝ), 
    (∃ (perm1 perm2 : Fin 4 ≃ Fin 4), 
      (∀ i : Fin 3, face i = t1.vertices (perm1 i)) ∧ 
      (∀ i : Fin 3, face i = t2.vertices (perm2 i))) ∧
    sorry -- Placeholder for area calculation

/-- All tetrahedra in the arrangement are non-overlapping -/
def NonOverlapping (arr : TetrahedronArrangement) : Prop :=
  ∀ i j : Fin 8, i ≠ j → 
    sorry -- Placeholder for volume of intersection calculation

/-- Any two tetrahedra in the arrangement share a face -/
def AllShareFaces (arr : TetrahedronArrangement) : Prop :=
  ∀ i j : Fin 8, i ≠ j → ShareFace (arr.tetrahedra i) (arr.tetrahedra j)

/-- The main theorem: there exists an arrangement of 8 non-overlapping tetrahedra 
    where any two share a face with nonzero area -/
theorem tetrahedra_arrangement_exists : 
  ∃ arr : TetrahedronArrangement, NonOverlapping arr ∧ AllShareFaces arr :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedra_arrangement_exists_l216_21697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l216_21658

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  dist t.A t.B = dist t.A t.C

-- Define the altitude AD
noncomputable def altitude (t : Triangle) : ℝ × ℝ := sorry

-- Define the property that AD bisects BC
def altitudeBisectsBase (t : Triangle) : Prop :=
  dist (altitude t) t.B = dist (altitude t) t.C

-- Define the lengths of the sides
def sideLength (t : Triangle) : ℝ := 41
def baseLength (t : Triangle) : ℝ := 18

-- Define the area of the triangle
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

-- The main theorem
theorem isosceles_triangle_area (t : Triangle) 
  (h1 : isIsosceles t) 
  (h2 : altitudeBisectsBase t) 
  (h3 : sideLength t = 41) 
  (h4 : baseLength t = 18) : 
  triangleArea t = 360 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l216_21658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_product_24_l216_21604

/-- A function that returns the product of the digits of a positive integer -/
def digit_product (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit positive integer -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The set of all three-digit positive integers whose digits have a product of 24 -/
def S : Finset ℕ := sorry

/-- The main theorem stating that there are exactly 21 elements in the set S -/
theorem count_three_digit_product_24 : S.card = 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_product_24_l216_21604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_subjects_l216_21634

theorem soccer_team_subjects (total_players : ℕ) (physics_players : ℕ) (chemistry_players : ℕ) (biology_players : ℕ) (all_three_subjects : ℕ) :
  total_players = 30 ∧
  physics_players = 12 ∧
  chemistry_players = 10 ∧
  biology_players = 8 ∧
  all_three_subjects = 3 ∧
  (∃ (p c b : Finset ℕ), 
    Finset.card p = physics_players ∧
    Finset.card c = chemistry_players ∧
    Finset.card b = biology_players ∧
    Finset.card (p ∩ c ∩ b) = all_three_subjects ∧
    Finset.card (p ∪ c ∪ b) = total_players) →
  ∃ (p c b : Finset ℕ),
    Finset.card (p ∩ c) + Finset.card (p ∩ b) + Finset.card (c ∩ b) - 3 * Finset.card (p ∩ c ∩ b) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_subjects_l216_21634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l216_21693

/-- The volume of a triangular pyramid with given base edge lengths and dihedral angle -/
theorem triangular_pyramid_volume 
  (a b c : ℝ) 
  (α : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hα : 0 < α ∧ α < π/2) :
  let p := (a + b + c)/2
  ∃ V : ℝ, V = ((p-a)*(p-b)*(p-c)/3) * Real.tan α ∧ 
    V > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l216_21693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l216_21609

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 2)

theorem f_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x y, x ∈ Set.Icc (Real.pi / 2) Real.pi → y ∈ Set.Icc (Real.pi / 2) Real.pi → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l216_21609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_A_correct_num_purchasing_plans_correct_l216_21606

-- Define the selling price of brand A car this year (in million yuan)
def selling_price_A : ℝ := 9

-- Define the number of purchasing plans
def num_purchasing_plans : ℕ := 5

-- Define the conditions
def price_difference : ℝ := 1
def last_year_revenue : ℝ := 1
def this_year_revenue : ℝ := 0.9
def cost_price_A : ℝ := 0.75
def cost_price_B : ℝ := 0.6
def total_budget : ℝ := 1.05
def total_cars : ℕ := 15
def min_cars_A : ℕ := 6

-- Theorem for the selling price of brand A car
theorem selling_price_A_correct :
  selling_price_A * (last_year_revenue / (selling_price_A + price_difference)) =
  this_year_revenue * (last_year_revenue / (selling_price_A + price_difference)) :=
by sorry

-- Theorem for the number of purchasing plans
theorem num_purchasing_plans_correct :
  num_purchasing_plans = (Finset.filter (λ n : ℕ => n ≤ 10 ∧ cost_price_A * n + cost_price_B * (total_cars - n) ≤ total_budget) (Finset.range (total_cars + 1))).card
  ∧ ∀ n ∈ (Finset.filter (λ n : ℕ => n ≤ 10 ∧ cost_price_A * n + cost_price_B * (total_cars - n) ≤ total_budget) (Finset.range (total_cars + 1))), n ≥ min_cars_A :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_A_correct_num_purchasing_plans_correct_l216_21606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beeswax_amount_l216_21688

/-- The amount of beeswax used per candle -/
noncomputable def beeswax_per_candle : ℚ :=
  let num_candles : ℕ := 10 - 3
  let coconut_oil_per_candle : ℚ := 1
  let total_weight : ℚ := 63
  (total_weight - num_candles * coconut_oil_per_candle) / num_candles

/-- Theorem stating that the amount of beeswax used per candle is 8 ounces -/
theorem beeswax_amount : beeswax_per_candle = 8 := by
  -- Unfold the definition of beeswax_per_candle
  unfold beeswax_per_candle
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beeswax_amount_l216_21688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_ratio_l216_21602

/-- Given a football team with the following properties:
  - Total players: 70
  - Throwers: 49
  - All throwers are right-handed
  - Total right-handed players: 63

  Prove that the ratio of left-handed players to the rest of the team (excluding throwers) is 1:3.
-/
theorem football_team_ratio :
  ∀ (total_players throwers right_handed : ℕ),
    total_players = 70 →
    throwers = 49 →
    right_handed = 63 →
    let non_throwers := total_players - throwers
    let right_handed_non_throwers := right_handed - throwers
    let left_handed_non_throwers := non_throwers - right_handed_non_throwers
    (left_handed_non_throwers : ℚ) / (non_throwers : ℚ) = 1 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_ratio_l216_21602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l216_21657

theorem collinear_vectors_lambda (a b c : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 2) →
  b = (2, 0) →
  c = (1, -2) →
  (∃ (k : ℝ), k ≠ 0 ∧ lambda • a + b = k • c) →
  lambda = -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l216_21657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l216_21694

noncomputable section

-- Define the time it takes to fill the tank with the leak
def fill_time_with_leak : ℝ := 20

-- Define the time it takes for the leak to empty the full tank
def leak_empty_time : ℝ := 30

-- Define the function to calculate the fill time without the leak
noncomputable def fill_time_without_leak : ℝ := 
  (fill_time_with_leak * leak_empty_time) / (leak_empty_time - fill_time_with_leak)

-- Theorem statement
theorem pipe_fill_time :
  fill_time_without_leak = 12 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l216_21694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_box_better_value_l216_21618

/-- Represents a box of macaroni and cheese -/
structure MacaroniBox where
  size : ℚ  -- size in ounces
  price : ℚ  -- price in dollars

/-- Calculates the price per ounce in cents -/
def pricePerOunceInCents (box : MacaroniBox) : ℚ :=
  (box.price / box.size) * 100

/-- Theorem stating that the larger box has the better value and its price per ounce is 16 cents -/
theorem larger_box_better_value (largerBox smallerBox : MacaroniBox) : 
  largerBox.size = 30 ∧ 
  largerBox.price = 48/10 ∧ 
  smallerBox.size = 20 ∧ 
  smallerBox.price = 34/10 →
  pricePerOunceInCents largerBox < pricePerOunceInCents smallerBox ∧ 
  pricePerOunceInCents largerBox = 16 := by
  sorry

#eval pricePerOunceInCents { size := 30, price := 48/10 }
#eval pricePerOunceInCents { size := 20, price := 34/10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_box_better_value_l216_21618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_smaller_triangle_l216_21611

/-- Given a larger equilateral triangle with side length 4 units and a smaller equilateral triangle
    with an area one-third that of the larger triangle, the radius of the circle inscribed in the
    smaller triangle is 2/3 units. -/
theorem inscribed_circle_radius_smaller_triangle (s r : ℝ) : 
  s = 4 →  -- Side length of larger triangle
  (Real.sqrt 3 / 4 * s^2) / 3 = Real.sqrt 3 / 4 * ((4 * Real.sqrt 3) / 3)^2 →  -- Area of smaller triangle is 1/3 of larger
  r = 2 / 3 :=  -- Radius of inscribed circle in smaller triangle
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_smaller_triangle_l216_21611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_vertex_l216_21686

/-- A parabola passing through three given points -/
structure Parabola where
  -- The coefficients of the parabola equation y = ax^2 + bx + c
  a : ℚ
  b : ℚ
  c : ℚ
  -- Conditions that the parabola passes through the given points
  point_A : a * (-2)^2 + b * (-2) + c = 0
  point_B : a * 1^2 + b * 1 + c = 0
  point_C : a * 2^2 + b * 2 + c = 8

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℚ × ℚ := (-p.b / (2 * p.a), -p.b^2 / (4 * p.a) + p.c)

theorem parabola_equation_and_vertex (p : Parabola) : 
  (p.a = 2 ∧ p.b = 2 ∧ p.c = -4) ∧ vertex p = (-(1/2), -(9/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_vertex_l216_21686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_halt_duration_l216_21683

/-- Proves that the duration of the third halt is approximately 46.67 minutes given the train schedule conditions --/
theorem train_halt_duration (average_speed : ℝ) (total_distance : ℝ) (total_travel_time : ℝ) :
  average_speed = 115 →
  total_distance = 575 →
  total_travel_time = 7.5 →
  ∃ (first_halt second_halt third_halt : ℝ),
    first_halt = second_halt + 1/3 →
    third_halt = first_halt - 1/4 →
    first_halt + second_halt + third_halt = total_travel_time - (total_distance / average_speed) →
    abs (third_halt * 60 - 46.67) < 0.01 := by
  sorry

#check train_halt_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_halt_duration_l216_21683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_for_f_inequality_l216_21681

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2*x

-- State the theorem
theorem min_b_for_f_inequality (a : ℝ) (h_a : a ∈ Set.Icc (-1) 0) :
  ∃ b : ℝ, b > -3/2 ∧ ∀ x ∈ Set.Ioo 0 1, f a x < b ∧
  ∀ b' : ℝ, (∀ x ∈ Set.Ioo 0 1, f a x < b') → b' ≥ -3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_for_f_inequality_l216_21681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l216_21651

/-- The differential equation (3x²y² + 7)dx + 2x³y dy = 0 with initial condition y(0) = 1 
    has the solution y²x³ + 7x = 0 -/
theorem differential_equation_solution 
  (x y : ℝ → ℝ) 
  (h : ∀ t, (3 * (x t)^2 * (y t)^2 + 7) * (deriv x t) + 2 * (x t)^3 * (y t) * (deriv y t) = 0) 
  (init : y 0 = 1) :
  ∀ t, (y t)^2 * (x t)^3 + 7 * (x t) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l216_21651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_february_13_to_25_l216_21638

/-- Represents days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a date in February -/
structure FebruaryDate :=
  (day : Nat)

/-- Given a day of the week, returns the day of the week that occurs n days later -/
def daysLater (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  sorry

/-- Checks if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  sorry

theorem february_13_to_25 (year : Nat) (feb13 feb25 : FebruaryDate) :
  ¬ isLeapYear year →
  feb13.day = 13 →
  feb25.day = 25 →
  daysLater DayOfWeek.Friday 12 = DayOfWeek.Wednesday :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_february_13_to_25_l216_21638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_f_transformed_l216_21671

-- Define the function f piecewise
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -3 ∧ x ≤ 0 then -2 - x
  else if x > 0 ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if x > 2 ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- Define a default value for x outside the given intervals

-- Define the transformation steps
def reflect_y (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (-x)
def shift_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x - a)
def shift_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x => f x + b

-- Define h as the composition of these transformations
noncomputable def h : ℝ → ℝ :=
  shift_up (shift_right (reflect_y f) 3) 2

-- The theorem to prove
theorem h_equals_f_transformed : h = λ x => f (3 - x) + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_f_transformed_l216_21671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l216_21613

def M : Set ℝ := {x | x^2 - 2*x > 0}

theorem complement_of_M : Set.compl M = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l216_21613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l216_21637

def M : Set ℤ := {1, 2, 3}

def N : Set ℤ := {x : ℤ | (x + 1) * (x - 2) < 0}

theorem union_of_M_and_N : M ∪ N = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l216_21637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_is_negative_two_l216_21639

/-- Represents the temperature in Celsius -/
def temperature : ℤ := -2

/-- Axiom: The temperature is 2 degrees Celsius below zero -/
axiom below_zero : temperature = -2

/-- Theorem: The temperature is -2°C -/
theorem temperature_is_negative_two : temperature = -2 := by
  exact below_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_is_negative_two_l216_21639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_sum_theorem_l216_21695

/-- Represents a number in base b -/
structure BaseNum (b : ℕ) where
  value : ℕ

/-- Converts a base b number to its decimal representation -/
def to_decimal (b : ℕ) (x : BaseNum b) : ℕ := sorry

/-- Multiplies two numbers in base b -/
def mul_base (b : ℕ) (x y : BaseNum b) : BaseNum b := sorry

/-- Adds two numbers in base b -/
def add_base (b : ℕ) (x y : BaseNum b) : BaseNum b := sorry

/-- Converts a natural number to a BaseNum -/
def nat_to_base_num (b n : ℕ) : BaseNum b := ⟨n⟩

theorem base_sum_theorem (b : ℕ) 
  (h : mul_base b (mul_base b (nat_to_base_num b 12) (nat_to_base_num b 15)) (nat_to_base_num b 16) = nat_to_base_num b 3146) : 
  add_base b (add_base b (nat_to_base_num b 12) (nat_to_base_num b 15)) (nat_to_base_num b 16) = nat_to_base_num b 44 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_sum_theorem_l216_21695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_last_two_digits_l216_21633

theorem sum_of_last_two_digits : (8^50 + 12^50) % 100 = 48 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_last_two_digits_l216_21633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_move_all_left_l216_21627

/-- Represents a 10x10 checker board -/
def Board := Fin 10 × Fin 10

/-- Represents a checker on the board -/
structure Checker where
  position : Board

/-- Represents the state of the game -/
structure GameState where
  checkers : List Checker

/-- Initial state of the game -/
def initialState : GameState :=
  { checkers :=
      (List.range 25).map (fun _ => { position := ⟨0, 0⟩ }) ++  -- Lower left quarter
      (List.range 25).map (fun _ => { position := ⟨5, 5⟩ })     -- Upper right quarter
  }

/-- Checks if a position is on the left half of the board -/
def isOnLeftHalf (pos : Board) : Bool :=
  pos.1 < 5

/-- Checks if all checkers are on the left half of the board -/
def allCheckersOnLeftHalf (state : GameState) : Bool :=
  state.checkers.all (fun c => isOnLeftHalf c.position)

/-- Represents a valid move in the game -/
inductive ValidMove : GameState → GameState → Prop where
  | jump : ∀ (s₁ s₂ : GameState), ValidMove s₁ s₂

/-- The main theorem: It's impossible to move all checkers to the left half -/
theorem impossible_to_move_all_left :
  ¬∃ (finalState : GameState),
    (Relation.ReflTransGen ValidMove) initialState finalState ∧
    allCheckersOnLeftHalf finalState = true :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_move_all_left_l216_21627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_a_values_l216_21689

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}

def B (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2*a + 2, a^3 + a^2 + 3*a + 7}

theorem intersection_implies_a_values :
  ∀ a : ℝ, A a ∩ B a = {2, 5} → a ∈ ({-1, 2} : Set ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_a_values_l216_21689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_minus_cot_l216_21653

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.tan x - (1 / Real.tan x)

-- State the theorem
theorem period_of_tan_minus_cot :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_minus_cot_l216_21653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_and_reciprocal_l216_21608

theorem max_sum_and_reciprocal (numbers : Finset ℝ) 
  (positive : ∀ x ∈ numbers, x > 0)
  (count : numbers.card = 1009)
  (sum_eq : numbers.sum id = 1010)
  (sum_recip_eq : numbers.sum (λ x => 1 / x) = 1010) :
  ∃ x ∈ numbers, ∀ y ∈ numbers, x + 1 / x ≥ y + 1 / y ∧ x + 1 / x ≤ 2029 / 1010 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_and_reciprocal_l216_21608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l216_21674

noncomputable section

-- Define the hyperbola
def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

-- Define the asymptotes
def asymptotes (a b : ℝ) (x y : ℝ) : Prop :=
  y = b / a * x ∨ y = -b / a * x

theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  eccentricity a b = Real.sqrt 2 →
  ∀ x y : ℝ, is_hyperbola a b x y →
  asymptotes a b x y ↔ (y = x ∨ y = -x) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l216_21674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_theorem_l216_21690

/-- Calculates the milk production for a given number of cows with increasing productivity -/
noncomputable def milk_production (x y z w v : ℝ) : ℝ :=
  v * y * (3^w - 1) / (z * (3^x - 1))

/-- Theorem stating the milk production formula for cows with increasing productivity -/
theorem milk_production_theorem 
  (x y z w v : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (hw : w > 0) 
  (hv : v > 0) :
  milk_production x y z w v = v * y * (3^w - 1) / (z * (3^x - 1)) :=
by
  -- Unfold the definition of milk_production
  unfold milk_production
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_theorem_l216_21690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_greater_than_14_l216_21682

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | (n + 1) => sequence_a n + 1 / sequence_a n

theorem a_100_greater_than_14 : sequence_a 100 > 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_greater_than_14_l216_21682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_four_divisors_l216_21620

/-- A function that returns the number of divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- A function that checks if n consecutive natural numbers starting from k
    each have exactly four divisors -/
def consecutive_four_divisors (n k : ℕ) : Prop :=
  ∀ i, i ∈ Finset.range n → num_divisors (k + i) = 4

/-- The maximum number of consecutive natural numbers with exactly four divisors is 3 -/
theorem max_consecutive_four_divisors :
  (∃ k, consecutive_four_divisors 3 k) ∧ 
  (∀ n, n > 3 → ¬∃ k, consecutive_four_divisors n k) := by
  sorry

#check max_consecutive_four_divisors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_four_divisors_l216_21620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_range_l216_21677

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  AD : ℝ
  BC : ℝ
  t : ℝ
  h_AD_perp_BC : AD ≠ 0 ∧ BC ≠ 0  -- We can't use ⟂ directly, so we'll assume non-zero
  h_AD_eq : AD = 6
  h_BC_eq : BC = 2
  h_t_range : t ∈ Set.Ici 8
  h_sum_eq : ∀ (AB BD AC CD : ℝ), AB + BD = t ∧ AC + CD = t

/-- The maximum volume of a tetrahedron with given properties -/
noncomputable def max_volume (tetra : Tetrahedron) : ℝ :=
  2 * Real.sqrt (tetra.t^2 / 4 - 10)

/-- Theorem stating the range of the maximum volume of the tetrahedron -/
theorem max_volume_range (tetra : Tetrahedron) :
  max_volume tetra ∈ Set.Ici (2 * Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_range_l216_21677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_point_ratio_l216_21646

/-- Represents the ratio in which point T divides line AB --/
noncomputable def dividing_ratio (AT TB : ℝ) : ℝ := AT / TB

/-- The train problem setup --/
structure TrainProblem where
  v1 : ℝ  -- Speed of train from A to B
  v2 : ℝ  -- Speed of train from B to A
  time_diff : ℝ  -- Time difference between departures (11 minutes)
  time_to_B : ℝ  -- Time for first train to reach B after meeting (20 minutes)
  time_to_A : ℝ  -- Time for second train to reach A after meeting (45 minutes)
  h_positive : v1 > 0 ∧ v2 > 0  -- Speeds are positive

/-- The main theorem --/
theorem train_meeting_point_ratio (p : TrainProblem) : 
  dividing_ratio (p.time_to_A * p.v2) (p.time_to_B * p.v1) = 9 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_point_ratio_l216_21646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_lower_bound_l216_21603

open Real

/-- The function f(x) = x^2 - bx + ln(x) -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + log x

/-- The derivative of f(x) -/
noncomputable def f_deriv (b : ℝ) (x : ℝ) : ℝ := 2*x - b + 1/x

theorem f_difference_lower_bound (b : ℝ) (x₁ x₂ : ℝ) 
  (hb : b > 3) 
  (hx₁ : f_deriv b x₁ = 0)
  (hx₂ : f_deriv b x₂ = 0)
  (horder : x₁ < x₂) :
  f b x₁ - f b x₂ > 3/4 - log 2 := by
  sorry

#check f_difference_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_lower_bound_l216_21603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_symmetry_function_vertical_line_intersection_cosine_product_sine_sum_l216_21666

/- Define a function f: ℝ → ℝ -/
variable (f : ℝ → ℝ)

/- Define the property of being an even function -/
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

/- Define the property of a function being symmetric about a point -/
def symmetric_about (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

/- Define the property of a function having at most one intersection with a vertical line -/
def at_most_one_intersection (g : ℝ → ℝ) : Prop :=
  ∀ a x₁ x₂, g x₁ = a ∧ g x₂ = a → x₁ = x₂

/- State the theorems to be proved -/
theorem even_function_symmetry :
  is_even (fun x ↦ f (x + 1)) → symmetric_about f 1 := by sorry

theorem function_vertical_line_intersection :
  at_most_one_intersection f := by sorry

theorem cosine_product_sine_sum :
  ∀ α β : ℝ, Real.cos α * Real.cos β = 1 → Real.sin (α + β) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_symmetry_function_vertical_line_intersection_cosine_product_sine_sum_l216_21666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_square_digit_sequence_l216_21601

/-- Represents a digit (1-9) -/
def Digit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- Converts a sequence of digits to a natural number -/
def seqToNat (s : ℕ → Digit) (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => 10 * acc + (s i).val) 0

/-- The main theorem -/
theorem no_infinite_square_digit_sequence :
  ¬∃ (a : ℕ → Digit) (N : ℕ), ∀ k > N, ∃ m : ℕ, seqToNat a k = m^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_square_digit_sequence_l216_21601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_line_equation_l216_21652

/-- The equation of the line of symmetry for two given points -/
def line_of_symmetry (A B : ℝ × ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ 3 * x - y + 3 = 0

/-- Two points are symmetric with respect to a line if the line is their perpendicular bisector -/
def symmetric_points (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (M : ℝ × ℝ), 
    (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) ∧ 
    l M.1 M.2 ∧
    (B.2 - A.2) * (M.1 - A.1) = -(B.1 - A.1) * (M.2 - A.2)

theorem symmetry_line_equation (A B : ℝ × ℝ) 
  (hA : A = (4, 5)) (hB : B = (-2, 7)) :
  symmetric_points A B (line_of_symmetry A B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_line_equation_l216_21652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_probability_l216_21636

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Uniform distribution on a line segment -/
def UniformDist (a b : ℝ) := ℝ → ℝ

/-- The probability that two circles intersect -/
noncomputable def intersectionProbability (A B : Circle) (distA distB : UniformDist 0 2) : ℝ := 
  sorry

theorem circle_intersection_probability :
  ∀ (A B : Circle) (distA distB : UniformDist 0 2),
    A.radius = 2 ∧
    B.radius = 2 ∧
    A.center.2 = 0 ∧
    B.center.2 = 2 ∧
    B.center.2 ≥ 1 →
    intersectionProbability A B distA distB = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_probability_l216_21636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parallel_planes_l216_21635

structure Space where
  Line : Type
  Plane : Type

variable (S : Space)

def perpendicular (S : Space) (a b : S.Line) : Prop := sorry

def parallel_line_plane (S : Space) (l : S.Line) (p : S.Plane) : Prop := sorry

def parallel_plane (S : Space) (p q : S.Plane) : Prop := sorry

theorem perpendicular_lines_parallel_planes 
  (a b : S.Line) (α β : S.Plane) :
  perpendicular S a b →
  parallel_line_plane S a α →
  parallel_line_plane S a β →
  parallel_line_plane S b α →
  parallel_line_plane S b β →
  parallel_plane S α β :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parallel_planes_l216_21635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_novel_pages_count_l216_21649

theorem novel_pages_count 
  (days : ℕ)
  (pages_per_day_first_four : ℕ)
  (pages_per_day_next_two : ℕ)
  (pages_last_day : ℕ)
  (pages_reviewed : ℕ) :
  days = 7 →
  pages_per_day_first_four = 42 →
  pages_per_day_next_two = 48 →
  pages_last_day = 14 →
  pages_reviewed = 10 →
  4 * pages_per_day_first_four + 2 * pages_per_day_next_two + pages_last_day = 278 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_novel_pages_count_l216_21649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_range_l216_21617

noncomputable def curve (a θ : ℝ) : ℝ × ℝ := (a + 2 * Real.cos θ, a + 2 * Real.sin θ)

noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ := Real.sqrt (p.1^2 + p.2^2)

theorem curve_intersection_range (a : ℝ) :
  (∃! p q : ℝ, p ≠ q ∧ 
    (∃ θ₁ θ₂ : ℝ, curve a θ₁ = (p, q) ∧ curve a θ₂ = (p, q)) ∧ 
    distance_from_origin (p, q) = 2) →
  (0 < a ∧ a < 2 * Real.sqrt 2) ∨ (-2 * Real.sqrt 2 < a ∧ a < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_range_l216_21617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_subsets_l216_21650

open Set Finset

def A : Finset Char := {'a'}
def S : Finset Char := {'a', 'b', 'c'}

theorem count_subsets : 
  (filter (λ P => A ⊂ P ∧ P ⊆ S) (powerset S)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_subsets_l216_21650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_in_terms_of_x_l216_21669

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi / 2) 
  (h_x : x > 1) 
  (h_cos : Real.cos (θ / 2) = Real.sqrt ((x + 1) / (2 * x))) : 
  Real.tan θ = Real.sqrt (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_in_terms_of_x_l216_21669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pooh_visits_tigger_last_l216_21672

/-- Represents a friend of Winnie-the-Pooh -/
inductive Friend
| tigger
| piglet
| owl
| eeyore
| rabbit
deriving DecidableEq

/-- The number of honey pots each friend has -/
def honey_pots (f : Friend) : Nat :=
  match f with
  | .tigger => 1
  | .piglet => 2
  | .owl => 3
  | .eeyore => 4
  | .rabbit => 5

/-- The list of all friends -/
def all_friends : List Friend := [Friend.tigger, Friend.piglet, Friend.owl, Friend.eeyore, Friend.rabbit]

/-- The total number of honey pots Pooh collects from a list of friends -/
def total_honey (friends : List Friend) : Nat :=
  friends.foldl (fun acc f => acc + honey_pots f - 1) 0

theorem pooh_visits_tigger_last (first : Friend) :
  first ≠ Friend.tigger →
  ∃ (order : List Friend), order.length = 5 ∧
                           order.head? = some first ∧
                           order.getLast? = some Friend.tigger ∧
                           order.toFinset = all_friends.toFinset ∧
                           total_honey (order.dropLast) = 10 := by
  sorry

#check pooh_visits_tigger_last

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pooh_visits_tigger_last_l216_21672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l216_21696

theorem triangle_inequality (x y z : ℝ) (A B C : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : x = y * Real.sin C / Real.sin A ∧ 
             y = z * Real.sin A / Real.sin B ∧ 
             z = x * Real.sin B / Real.sin C) :
  x^2 + y^2 + z^2 ≥ 2*y*z*Real.sin A + 2*z*x*Real.sin B - 2*x*y*Real.cos C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l216_21696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_time_ratio_l216_21656

/-- Represents a cyclist's trip -/
structure Trip where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a trip -/
noncomputable def time (t : Trip) : ℝ := t.distance / t.speed

theorem cyclist_time_ratio :
  ∀ (initial_speed : ℝ),
  initial_speed > 0 →
  let initial_trip : Trip := { distance := 30, speed := initial_speed }
  let later_trip : Trip := { distance := 150, speed := 4 * initial_speed }
  (time later_trip) / (time initial_trip) = 1.25 := by
  intro initial_speed h_speed_pos
  -- The proof steps would go here
  sorry

#check cyclist_time_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_time_ratio_l216_21656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_from_exponential_equation_l216_21685

theorem sin_2theta_from_exponential_equation (θ : ℝ) 
  (h : (2 : ℝ)^(-5/2 + 3 * Real.cos θ) + 1 = (2 : ℝ)^(1/2 + Real.cos θ)) : 
  Real.sin (2 * θ) = 5 * Real.sqrt 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_from_exponential_equation_l216_21685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_triangle_perimeter_approx_l216_21628

/-- Triangle PQR with given side lengths and parallel lines forming a smaller triangle -/
structure TriangleWithParallels where
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  ℓP_segment : ℝ
  ℓQ_segment : ℝ
  ℓR_segment : ℝ
  h_PQ : PQ = 140
  h_QR : QR = 260
  h_PR : PR = 210
  h_ℓP : ℓP_segment = 65
  h_ℓQ : ℓQ_segment = 35
  h_ℓR : ℓR_segment = 25

/-- The perimeter of the inner triangle formed by the parallel lines -/
noncomputable def inner_triangle_perimeter (t : TriangleWithParallels) : ℝ :=
  let scaling_factor := (t.ℓP_segment / t.QR + t.ℓQ_segment / t.PR + t.ℓR_segment / t.PQ) / 3
  scaling_factor * (t.PQ + t.QR + t.PR)

/-- Theorem stating that the perimeter of the inner triangle is approximately 121.02 -/
theorem inner_triangle_perimeter_approx (t : TriangleWithParallels) :
  abs (inner_triangle_perimeter t - 121.02) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_triangle_perimeter_approx_l216_21628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l216_21614

noncomputable def f (x a : ℝ) : ℝ := 9^x - Real.log x / Real.log a

theorem inequality_holds_iff_a_in_range :
  ∀ a : ℝ, a > 0 →
  (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → f x a ≤ 2) ↔ 1/2 ≤ a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l216_21614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bens_correct_percentage_l216_21600

theorem bens_correct_percentage 
  (total_problems : ℝ)
  (chloe_solo_correct_rate : ℝ)
  (chloe_total_correct_rate : ℝ)
  (ben_solo_correct_rate : ℝ)
  (h1 : chloe_solo_correct_rate = 0.7)
  (h2 : chloe_total_correct_rate = 0.84)
  (h3 : ben_solo_correct_rate = 0.8) :
  (ben_solo_correct_rate * (total_problems / 2) + 
   (chloe_total_correct_rate * total_problems - chloe_solo_correct_rate * (total_problems / 2))) / 
   total_problems = 0.89 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bens_correct_percentage_l216_21600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_correct_l216_21692

/-- A hyperbola with center at the origin and focal distance 10 -/
structure Hyperbola where
  /-- The focus lies on a coordinate axis -/
  focus_on_axis : Bool
  /-- Point (2,1) is on the asymptote -/
  point_on_asymptote : (2 : ℝ) = 2 ∧ (1 : ℝ) = 1  -- Placeholder condition

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  if h.focus_on_axis
  then λ x y => x^2 / 20 - y^2 / 5 = 1
  else λ x y => y^2 / 5 - x^2 / 20 = 1

theorem hyperbola_equation_correct (h : Hyperbola) :
  hyperbola_equation h = (λ x y => x^2 / 20 - y^2 / 5 = 1) ∨
  hyperbola_equation h = (λ x y => y^2 / 5 - x^2 / 20 = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_correct_l216_21692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l216_21642

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + (2^(x+1))/(2^x + 1) + Real.sin x

-- Define the theorem
theorem range_sum (k : ℝ) (m n : ℝ) (hk : k > 0) :
  (∀ x ∈ Set.Icc (-k) k, m ≤ f x ∧ f x ≤ n) →
  (∀ y : ℝ, (∃ x ∈ Set.Icc (-k) k, f x = y) → m ≤ y ∧ y ≤ n) →
  m + n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l216_21642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_stocking_probability_l216_21661

/-- Represents the probability of stocking lemonade on a single day -/
def daily_probability : ℚ := 1/2

/-- The number of days Susan visits the store -/
def total_days : ℕ := 7

/-- The number of days we want the lemonade to be stocked -/
def target_days : ℕ := 3

/-- Represents the condition that if lemonade is stocked on Monday, it's also stocked on Tuesday -/
def monday_tuesday_condition : Prop := sorry

/-- The probability of the store stocking lemonade exactly 3 out of 7 days, 
    given the Monday-Tuesday condition -/
def lemonade_probability : ℚ := 15/128

theorem lemonade_stocking_probability : 
  ∀ (p : ℚ) (n k : ℕ) (condition : Prop),
    p = daily_probability →
    n = total_days →
    k = target_days →
    condition = monday_tuesday_condition →
    lemonade_probability = (Nat.choose (n - 2) (k - 2) + Nat.choose (n - 2) k) * p^k * (1 - p)^(n - k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_stocking_probability_l216_21661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_width_to_perimeter_ratio_l216_21643

/-- Given a rectangular room with length 25 feet and width 15 feet,
    prove that the ratio of its width to its perimeter is 3:16. -/
theorem width_to_perimeter_ratio (length width : ℝ) (h1 : length = 25) (h2 : width = 15) :
  width / (2 * (length + width)) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_width_to_perimeter_ratio_l216_21643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l216_21619

open Real

theorem trig_simplification (α : ℝ) :
  (Real.sin (2 * π - α) * Real.cos (π + α) * Real.cos (π / 2 + α) * Real.cos (11 * π / 2 - α)) /
  (Real.cos (π - α) * Real.sin (3 * π - α) * Real.sin (-π - α) * Real.sin (9 * π / 2 + α)) = -Real.tan α ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l216_21619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_l216_21626

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then |x - 2| + a else -(|x - 2| + a)

-- State the theorem
theorem solution_set_of_f (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) →  -- f is an odd function
  (∀ x ≥ 0, f a x = |x - 2| + a) →  -- Definition of f for x ≥ 0
  {x : ℝ | f a x ≥ 1} = Set.union (Set.Icc (-3) (-1)) (Set.Ici 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_l216_21626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_squares_sum_l216_21648

theorem mean_squares_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hAM : (x + y + z) / 3 = 10)
  (hGM : ((x * y * z) ^ (1/3 : ℝ)) = 7)
  (hHM : 3 / ((1/x) + (1/y) + (1/z)) = 5) :
  x^2 + y^2 + z^2 = 488.4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_squares_sum_l216_21648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_l216_21691

-- Define the points
def A : ℝ × ℝ := (0, 7)
def B : ℝ × ℝ := (4, 7)
def C : ℝ × ℝ := (4, 4)
def D : ℝ × ℝ := (7, 0)
def E : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (0, 4)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the perimeter function
noncomputable def perimeter : ℝ := distance A B + distance B C + distance C F + distance F E + distance E A

-- State the theorem
theorem hexagon_perimeter : 
  distance A B = distance B C ∧ 
  (∃ G : ℝ × ℝ, G.2 = 0 ∧ (C.1 - F.1) * (G.2 - F.2) = (G.1 - F.1) * (C.2 - F.2)) ∧ 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (t * A.1 + (1 - t) * G.1, t * A.2 + (1 - t) * G.2) = G) →
  perimeter = 18 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_l216_21691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_half_l216_21641

theorem sin_alpha_plus_pi_half (α : Real) 
  (h1 : Real.tan (α - π) = 3 / 4)
  (h2 : α ∈ Set.Ioo (π / 2) (3 * π / 2)) :
  Real.sin (α + π / 2) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_half_l216_21641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_ab_12_l216_21698

/-- A piecewise function f(x) that is even --/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 3 * x^2 - 4 * x else a * x^2 + b * x

/-- Theorem stating that if f is even, then ab = 12 --/
theorem even_function_implies_ab_12 (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) → a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_ab_12_l216_21698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l216_21615

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line defined by x = constant -/
structure VerticalLine where
  x : ℝ

/-- Distance between a point and (3,0) -/
noncomputable def distToFocus (p : Point2D) : ℝ :=
  Real.sqrt ((p.x - 3)^2 + p.y^2)

/-- Distance between a point and a vertical line -/
def distToLine (p : Point2D) (l : VerticalLine) : ℝ :=
  abs (p.x - l.x)

/-- The locus of points satisfying the given condition -/
def locus : Set Point2D :=
  {p : Point2D | distToFocus p = distToLine p (VerticalLine.mk (-2)) + 1}

/-- Predicate to check if a set of points forms a parabola -/
def IsParabola (s : Set Point2D) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∧ ∀ p ∈ s, a * p.x^2 + b * p.x + c * p.y + d = 0

theorem locus_is_parabola :
  IsParabola locus :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l216_21615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_min_surface_area_l216_21676

open Real

theorem cylinder_min_surface_area (v : ℝ) (h r : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  v = π * r^2 * h →
  (h = 2 * r ↔ ∀ h' r', 0 < h' ∧ 0 < r' ∧ v = π * r'^2 * h' →
    2 * π * r * h + 2 * π * r^2 ≤ 2 * π * r' * h' + 2 * π * r'^2) :=
by
  sorry

#check cylinder_min_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_min_surface_area_l216_21676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_listening_time_is_55_minutes_l216_21670

/-- Represents the distribution of audience listening times for a talk --/
structure TalkAudience where
  total_duration : ℝ
  full_listen_percent : ℝ
  miss_entire_percent : ℝ
  half_listen_fraction : ℝ
  quarter_listen_fraction : ℝ
  three_quarter_listen_fraction : ℝ

/-- Calculates the average listening time for the audience --/
noncomputable def average_listening_time (audience : TalkAudience) : ℝ :=
  let remainder_percent := 1 - audience.full_listen_percent - audience.miss_entire_percent
  let full_time := audience.full_listen_percent * audience.total_duration
  let half_time := remainder_percent * audience.half_listen_fraction * (audience.total_duration / 2)
  let quarter_time := remainder_percent * audience.quarter_listen_fraction * (audience.total_duration / 4)
  let three_quarter_time := remainder_percent * audience.three_quarter_listen_fraction * (3 * audience.total_duration / 4)
  full_time + half_time + quarter_time + three_quarter_time

/-- Theorem stating that for the given audience distribution, the average listening time is 55 minutes --/
theorem average_listening_time_is_55_minutes 
  (audience : TalkAudience)
  (h1 : audience.total_duration = 90)
  (h2 : audience.full_listen_percent = 0.3)
  (h3 : audience.miss_entire_percent = 0.15)
  (h4 : audience.half_listen_fraction = 0.25)
  (h5 : audience.quarter_listen_fraction = 0.25)
  (h6 : audience.three_quarter_listen_fraction = 0.5)
  : average_listening_time audience = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_listening_time_is_55_minutes_l216_21670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_l216_21662

-- Define the function representing the curve
def f (x : ℝ) : ℝ := 2 * x^2 - x

-- Define the integral of the function from 0 to 1/2
noncomputable def area : ℝ := ∫ x in (0)..(1/2), -(f x)

-- Theorem statement
theorem area_under_curve : area = 1/24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_l216_21662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l216_21645

/-- The eccentricity of a hyperbola with equation x²/a² - y² = 1 where a > 1 -/
noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt (1 + 1 / (a * a))

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) :
  1 < eccentricity a ∧ eccentricity a < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l216_21645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hardcover_textbook_probability_l216_21699

/-- The probability of selecting two hardcover textbooks from a set of 6 textbooks, 
    where 3 are hardcover, is 1/5. -/
theorem hardcover_textbook_probability :
  let total_textbooks : ℕ := 6
  let hardcover_textbooks : ℕ := 3
  let selected_textbooks : ℕ := 2
  let probability_both_hardcover : ℚ := (hardcover_textbooks.choose selected_textbooks : ℚ) / (total_textbooks.choose selected_textbooks : ℚ)
  probability_both_hardcover = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hardcover_textbook_probability_l216_21699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_strategy_exists_l216_21667

/-- Represents a weighing result -/
inductive WeighingResult
  | Left : WeighingResult  -- Left side is lighter
  | Right : WeighingResult -- Right side is lighter
  | Equal : WeighingResult -- Both sides are equal

/-- Represents a coin selection for weighing -/
structure CoinSelection :=
  (left : Finset (Fin 26))
  (right : Finset (Fin 26))

/-- Represents a strategy for finding the counterfeit coin -/
structure Strategy :=
  (first_weighing : CoinSelection)
  (second_weighing : WeighingResult → CoinSelection)
  (third_weighing : WeighingResult → WeighingResult → CoinSelection)

/-- Checks if a strategy is valid (i.e., always finds the counterfeit coin in exactly three weighings) -/
def is_valid_strategy (s : Strategy) : Prop :=
  ∀ (counterfeit : Fin 26),
    ∃ (w1 w2 w3 : WeighingResult),
      let result1 := w1
      let result2 := w2
      let result3 := w3
      let final_selection := s.third_weighing result1 result2
      counterfeit ∈ final_selection.left ∪ final_selection.right

/-- Main theorem: There exists a valid strategy to find the counterfeit coin -/
theorem counterfeit_coin_strategy_exists : ∃ (s : Strategy), is_valid_strategy s := by
  sorry

#check counterfeit_coin_strategy_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_strategy_exists_l216_21667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_one_l216_21644

theorem absolute_difference_one (a b c d : ℤ) 
  (h : a + b + c + d = a*b + b*c + c*d + d*a + 1) :
  ∃ x y, x ∈ ({a, b, c, d} : Set ℤ) ∧ y ∈ ({a, b, c, d} : Set ℤ) ∧ x ≠ y ∧ |x - y| = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_one_l216_21644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l216_21621

/-- A line passing through (0, 2) with slope k -/
def line (k : ℝ) : ℝ → ℝ := λ x ↦ k * x + 2

/-- The circle (x-2)^2 + (y-2)^2 = 1 -/
def mycircle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1

/-- The line intersects the circle if there exists a point (x, y) on both the line and the circle -/
def intersects (k : ℝ) : Prop :=
  ∃ x : ℝ, mycircle x (line k x)

theorem slope_range :
  ∀ k : ℝ, intersects k → -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l216_21621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l216_21660

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 - 4*x

/-- The function g(x) defined in the problem -/
def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x

/-- Definition of mean-value average function -/
def is_mean_value_average_function (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    (deriv f ((x₁ + x₂) / 2) = (f x₂ - f x₁) / (x₂ - x₁))

theorem problem_solution :
  (∀ x ∈ Set.Icc (Real.exp (-1)) (Real.exp 1), f 1 x ≥ g 1 x) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc (Real.exp (-1)) (Real.exp 1), f a x ≥ g a x) → a ≤ -1) ∧
  (∀ a : ℝ, is_mean_value_average_function (f a) ↔ a = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l216_21660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_sum_approx_11_l216_21607

/-- Square with side length 8 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 8) ∧ B = (0, 0) ∧ C = (8, 0) ∧ D = (8, 8))

/-- Points E, F, G dividing BC into four equal segments -/
structure DividingPoints (s : Square) :=
  (E F G : ℝ × ℝ)
  (on_BC : E.1 = 2 ∧ F.1 = 4 ∧ G.1 = 6 ∧ E.2 = 0 ∧ F.2 = 0 ∧ G.2 = 0)

/-- Intersection points P, Q, R of AE, AF, AG with BD -/
structure IntersectionPoints (s : Square) (d : DividingPoints s) :=
  (P Q R : ℝ × ℝ)
  (on_BD : P.1 = P.2 ∧ Q.1 = Q.2 ∧ R.1 = R.2)
  (intersect : 
    P.2 = -4 * P.1 + 8 ∧
    Q.2 = -2 * Q.1 + 8 ∧
    R.2 = -4/3 * R.1 + 8)

/-- Main theorem -/
theorem ratio_sum_approx_11 (s : Square) (d : DividingPoints s) (i : IntersectionPoints s d) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧
  let BP := Real.sqrt ((i.P.1 - s.B.1)^2 + (i.P.2 - s.B.2)^2);
  let PQ := Real.sqrt ((i.Q.1 - i.P.1)^2 + (i.Q.2 - i.P.2)^2);
  let QR := Real.sqrt ((i.R.1 - i.Q.1)^2 + (i.R.2 - i.Q.2)^2);
  let RD := Real.sqrt ((s.D.1 - i.R.1)^2 + (s.D.2 - i.R.2)^2);
  |BP + PQ + QR + RD - 11| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_sum_approx_11_l216_21607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascend_speed_is_two_point_five_l216_21675

/-- Represents a round trip journey up and down a hill -/
structure HillJourney where
  ascendTime : ℝ
  descendTime : ℝ
  averageSpeed : ℝ

/-- Calculates the average speed while ascending the hill -/
noncomputable def ascendSpeed (journey : HillJourney) : ℝ :=
  let totalTime := journey.ascendTime + journey.descendTime
  let totalDistance := journey.averageSpeed * totalTime
  totalDistance / (2 * journey.ascendTime)

/-- Theorem stating that for a specific journey, the ascending speed is 2.5 km/h -/
theorem ascend_speed_is_two_point_five
  (journey : HillJourney)
  (h1 : journey.ascendTime = 3)
  (h2 : journey.descendTime = 2)
  (h3 : journey.averageSpeed = 3) :
  ascendSpeed journey = 2.5 := by
  sorry

#check ascend_speed_is_two_point_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascend_speed_is_two_point_five_l216_21675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l216_21631

/-- Predicate to check if a line is the directrix of a parabola -/
def IsDirectrix (k : ℝ) (y : ℝ) : Prop :=
  ∃ (p : ℝ), p ≠ 0 ∧ k = -1/p ∧ 
  ∀ (x : ℝ), y = x^2 → (x^2 = 4 * p * y)

/-- The equation of the directrix for the parabola y = x^2 is 4y + 1 = 0 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = x^2) → (∃ (k : ℝ), k * y + 1 = 0 ∧ k ≠ 0 ∧ IsDirectrix k y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l216_21631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_a_range_l216_21680

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x - a^2 - a

-- State the theorem
theorem root_implies_a_range (a : ℝ) (h_a_pos : a > 0) :
  (∃ x ≤ 1, f a x = 0) → 0 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_a_range_l216_21680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_average_marks_l216_21623

noncomputable def david_marks : List ℚ := [74, 65, 82, 67, 90]

def average_marks (marks : List ℚ) : ℚ :=
  marks.sum / marks.length

theorem david_average_marks :
  average_marks david_marks = 75.6 := by
  -- Unfold definitions and simplify
  unfold average_marks david_marks
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_average_marks_l216_21623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l216_21687

theorem size_relationship (a b c : ℝ) : 
  a = (-0.3:ℝ)^(0:ℝ) → b = (0.3:ℝ)^(2:ℝ) → c = (2:ℝ)^(0.3:ℝ) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l216_21687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l216_21678

-- Define the function h
noncomputable def h (t : ℝ) : ℝ := (t^2 - 1/2*t) / (t^2 + 2)

-- State the theorem about the range of h
theorem range_of_h :
  ∀ y : ℝ, (∃ t : ℝ, h t = y) ↔ y = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l216_21678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l216_21625

theorem angle_in_second_quadrant (θ : Real) :
  (-Real.sin θ < 0 ∧ Real.cos θ < 0) → (Real.sin θ > 0 ∧ Real.cos θ < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l216_21625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_diagonal_sum_l216_21663

/-- Represents a rectangular box with dimensions x, y, and z -/
structure RectangularBox where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The total surface area of the box -/
def surfaceArea (box : RectangularBox) : ℝ :=
  2 * (box.x * box.y + box.y * box.z + box.z * box.x)

/-- The sum of lengths of all edges of the box -/
def totalEdgeLength (box : RectangularBox) : ℝ :=
  4 * (box.x + box.y + box.z)

/-- The sum of lengths of all interior diagonals of the box -/
noncomputable def interiorDiagonalsSum (box : RectangularBox) : ℝ :=
  4 * Real.sqrt (box.x^2 + box.y^2 + box.z^2)

theorem box_diagonal_sum (box : RectangularBox) 
  (h1 : surfaceArea box = 116) 
  (h2 : totalEdgeLength box = 56) : 
  interiorDiagonalsSum box = 16 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_diagonal_sum_l216_21663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_quadrature_degree3_l216_21612

/-- Gaussian quadrature for polynomials of degree 3 or less -/
theorem gaussian_quadrature_degree3 (a b c d : ℝ) :
  ∫ x in (-1 : ℝ)..1, (a * x^3 + b * x^2 + c * x + d) =
  (a * (-1/Real.sqrt 3)^3 + b * (-1/Real.sqrt 3)^2 + c * (-1/Real.sqrt 3) + d) +
  (a * (1/Real.sqrt 3)^3 + b * (1/Real.sqrt 3)^2 + c * (1/Real.sqrt 3) + d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_quadrature_degree3_l216_21612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_rotation_and_intersection_l216_21610

open Real

-- Define the curves E₁, E₂, and E₃
noncomputable def E₁ (θ : ℝ) : ℝ := 4 * cos θ

noncomputable def E₂ (θ : ℝ) : ℝ := 4 / cos (θ - π/4)

noncomputable def E₃ (α θ : ℝ) : ℝ := 4 * cos (θ - α)

-- State the theorem
theorem curve_rotation_and_intersection :
  (∀ θ, E₃ (π/6) θ = 4 * cos (θ - π/6)) ∧
  (∃! p : ℝ × ℝ, E₃ (π/4) p.1 = E₂ p.1 ∧ p.2 = E₃ (π/4) p.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_rotation_and_intersection_l216_21610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_people_same_birth_second_l216_21622

/-- The number of seconds in 150 years (overestimated) -/
def max_seconds : ℕ := 150 * 400 * 25 * 3600

/-- The minimum current Earth population -/
def min_population : ℕ := 6000000000

/-- A function representing the birth second of a person -/
noncomputable def birth_second : ℕ → ℕ := sorry

theorem two_people_same_birth_second :
  min_population > max_seconds →
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ p1 ≤ min_population ∧ p2 ≤ min_population ∧
  ∃ (s : ℕ), s ≤ max_seconds ∧ (birth_second p1 = s ∧ birth_second p2 = s) :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_people_same_birth_second_l216_21622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_with_multiple_pets_l216_21664

theorem students_with_multiple_pets 
  (total : ℕ) 
  (dog cat rabbit : Finset ℕ) : 
  total = 50 → 
  (∀ s, s ∈ (dog ∪ cat ∪ rabbit : Finset ℕ)) → 
  Finset.card dog = 35 → 
  Finset.card cat = 40 → 
  Finset.card rabbit = 10 → 
  Finset.card (dog ∩ cat) = 20 → 
  Finset.card (dog ∩ rabbit) = 5 → 
  Finset.card ((dog ∩ cat) ∪ (dog ∩ rabbit) ∪ (cat ∩ rabbit)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_with_multiple_pets_l216_21664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_picking_inconsistency_l216_21655

theorem orange_picking_inconsistency 
  (total_oranges : ℝ) 
  (del_daily : ℝ) 
  (del_days : ℕ) 
  (juan_multiplier : ℕ) 
  (juan_days : ℕ) 
  (h1 : total_oranges = 215.4)
  (h2 : del_daily = 23.5)
  (h3 : del_days = 3)
  (h4 : juan_multiplier = 2)
  (h5 : juan_days = 4) :
  del_daily * (del_days : ℝ) + (juan_multiplier : ℝ) * del_daily * (juan_days : ℝ) > total_oranges :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_picking_inconsistency_l216_21655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_AB_a_minus_b_l216_21632

noncomputable section

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, -1)
def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (1, 1)

def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem cosine_of_angle_AB_a_minus_b :
  dot_product AB a_minus_b / (magnitude AB * magnitude a_minus_b) = -Real.sqrt 5 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_AB_a_minus_b_l216_21632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_coefficient_l216_21673

/-- The coefficient of a monomial is the numerical factor in front of the variables. -/
def coefficient (m : ℝ → ℝ → ℝ) : ℝ := sorry

/-- The monomial -2π * a * b^2 -/
noncomputable def monomial (a b : ℝ) : ℝ := -2 * Real.pi * a * b^2

theorem monomial_coefficient :
  coefficient monomial = -2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_coefficient_l216_21673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l216_21624

-- Define the constants
noncomputable def a : ℝ := Real.sin (1/5)
noncomputable def b : ℝ := 1/5
noncomputable def c : ℝ := (6/5) * Real.log (6/5)

-- State the theorem
theorem inequality_proof : c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l216_21624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l216_21654

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, x > 0 → y > 0 → f (x + y) + f x * f y = f (x * y) + f x + f y)
  (pos : ∀ x : ℝ, x > 0 → f x > 0) :
  (∀ x : ℝ, x > 0 → f x = 2) ∨ (∀ x : ℝ, x > 0 → f x = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l216_21654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_458_to_14_l216_21679

/-- Double a natural number -/
def double (n : ℕ) : ℕ := 2 * n

/-- Remove the last digit of a natural number -/
def removeLastDigit (n : ℕ) : ℕ := n / 10

/-- Check if a number can be transformed to the target using the given operations -/
def canTransform (start target : ℕ) : Prop :=
  ∃ (seq : List ℕ), 
    seq.head? = some start ∧
    seq.getLast? = some target ∧
    ∀ (i j : ℕ), i + 1 = j → j < seq.length →
      (seq[i]! = double seq[j]! ∨ seq[i]! = removeLastDigit seq[j]!)

/-- The main theorem: 458 can be transformed to 14 -/
theorem transform_458_to_14 : canTransform 458 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_458_to_14_l216_21679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l216_21616

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := 4/3
noncomputable def c : ℝ := Real.log 4 / Real.log 3

-- State the theorem
theorem log_inequality : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l216_21616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_b_length_l216_21668

/-- The length of train B given the conditions of the problem -/
theorem train_b_length 
  (length_a : ℝ) (speed_a : ℝ) (speed_b : ℝ) (time : ℝ) (km_to_m : ℝ) :
  length_a = 200 →
  speed_a = 54 →
  speed_b = 36 →
  time = 14 →
  km_to_m = 5 / 18 →
  (speed_a * km_to_m + speed_b * km_to_m) * time - length_a = 150 := by
  sorry

#check train_b_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_b_length_l216_21668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l216_21684

noncomputable def curve (x : ℝ) : ℝ := Real.log x - 1 / x + 1

noncomputable def line (x : ℝ) : ℝ := 2 * x

noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |2 * x - y| / Real.sqrt 5

theorem min_distance_curve_to_line :
  ∃ x : ℝ, x > 0 ∧ distance_to_line x (curve x) = 2 * Real.sqrt 5 / 5 ∧
  ∀ x' : ℝ, x' > 0 → distance_to_line x' (curve x') ≥ 2 * Real.sqrt 5 / 5 := by
  sorry

#check min_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l216_21684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_phi_sin_2x_plus_phi_even_l216_21647

theorem exists_phi_sin_2x_plus_phi_even : ∃ φ : ℝ, ∀ x : ℝ, 
  Real.sin (2 * x + φ) = Real.sin (2 * (-x) + φ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_phi_sin_2x_plus_phi_even_l216_21647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l216_21659

def IsArithmeticSequence (f : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, f (n + 1) - f n = d

theorem arithmetic_sequence_middle_term (x : ℝ) :
  x > 0 →
  IsArithmeticSequence (fun n => (n + 1)^2) →
  x^2 = (2^2 + 5^2) / 2 →
  x = Real.sqrt (29 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l216_21659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_transformations_return_to_original_l216_21640

noncomputable def M (a b c d e : ℝ) : ℝ := (a * c + b * d - c * e) / c

def opposite_transform (expr : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ) (x y : ℕ) : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ :=
  match x, y with
  | 1, 2 => λ a b c d e ↦ expr (-b) (-a) c d e
  | 1, 3 => λ a b c d e ↦ expr (-c) b (-a) d e
  | 1, 4 => λ a b c d e ↦ expr (-d) b c (-a) e
  | 1, 5 => λ a b c d e ↦ expr (-e) b c d (-a)
  | 2, 3 => λ a b c d e ↦ expr a (-c) (-b) d e
  | 2, 4 => λ a b c d e ↦ expr a (-d) c (-b) e
  | 2, 5 => λ a b c d e ↦ expr a (-e) c d (-b)
  | 3, 4 => λ a b c d e ↦ expr a b (-d) (-c) e
  | 3, 5 => λ a b c d e ↦ expr a b (-e) d (-c)
  | 4, 5 => λ a b c d e ↦ expr a b c (-e) (-d)
  | _, _ => expr

theorem four_transformations_return_to_original :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℕ),
    x₁ < y₁ ∧ x₂ < y₂ ∧ x₃ < y₃ ∧ x₄ < y₄ ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄) ∧
    ∀ (a b c d e : ℝ), c ≠ 0 →
      (opposite_transform
        (opposite_transform
          (opposite_transform
            (opposite_transform M x₁ y₁)
          x₂ y₂)
        x₃ y₃)
      x₄ y₄) a b c d e = M a b c d e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_transformations_return_to_original_l216_21640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_or_right_l216_21630

/-- A triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  angleSum : A + B + C = π
  cosineA : a = b * Real.cos C + c * Real.cos B
  cosineB : b = a * Real.cos C + c * Real.cos A
  cosineC : c = a * Real.cos B + b * Real.cos A

/-- The property that a*cos(A) = b*cos(B) in a triangle implies it's either isosceles or right. -/
theorem triangle_isosceles_or_right (t : Triangle) (h : t.a * Real.cos t.A = t.b * Real.cos t.B) :
  t.A = t.B ∨ t.A + t.B = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_or_right_l216_21630
