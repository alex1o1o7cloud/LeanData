import Mathlib

namespace smallest_first_term_arithmetic_progression_l3146_314636

theorem smallest_first_term_arithmetic_progression 
  (S₃ S₆ : ℕ) (d₁ : ℚ) 
  (h₁ : d₁ ≥ 1/2) 
  (h₂ : S₃ = 3 * d₁ + 3 * (S₆ - 2 * S₃) / 3) 
  (h₃ : S₆ = 6 * d₁ + 15 * (S₆ - 2 * S₃) / 3) :
  d₁ ≥ 5/9 :=
sorry

end smallest_first_term_arithmetic_progression_l3146_314636


namespace treewidth_iff_bramble_order_l3146_314670

/-- A graph represented by its vertex set and edge relation -/
structure Graph (V : Type) :=
  (edge : V → V → Prop)

/-- The treewidth of a graph -/
def treewidth {V : Type} (G : Graph V) : ℕ := sorry

/-- A bramble in a graph -/
def Bramble {V : Type} (G : Graph V) := Set (Set V)

/-- The order of a bramble -/
def brambleOrder {V : Type} (G : Graph V) (B : Bramble G) : ℕ := sorry

/-- Main theorem: A graph has treewidth ≥ k iff it contains a bramble of order > k -/
theorem treewidth_iff_bramble_order {V : Type} (G : Graph V) (k : ℕ) :
  treewidth G ≥ k ↔ ∃ (B : Bramble G), brambleOrder G B > k := by
  sorry

end treewidth_iff_bramble_order_l3146_314670


namespace bounded_sequence_periodic_l3146_314682

/-- A bounded sequence of integers satisfying the given recurrence relation -/
def BoundedSequence (a : ℕ → ℤ) : Prop :=
  ∃ M : ℕ, ∀ n : ℕ, |a n| ≤ M ∧
  ∀ n ≥ 5, a n = (a (n-1) + a (n-2) + a (n-3) * a (n-4)) / (a (n-1) * a (n-2) + a (n-3) + a (n-4))

/-- Definition of a periodic sequence -/
def IsPeriodic (a : ℕ → ℤ) (l : ℕ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ ∀ k ≥ l, a (k + T) = a k

/-- The main theorem -/
theorem bounded_sequence_periodic (a : ℕ → ℤ) (h : BoundedSequence a) :
  ∃ l : ℕ, IsPeriodic a l := by sorry

end bounded_sequence_periodic_l3146_314682


namespace max_sum_on_circle_l3146_314618

def is_on_circle (x y : ℤ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 16

theorem max_sum_on_circle :
  ∃ (a b : ℤ), is_on_circle a b ∧
  ∀ (x y : ℤ), is_on_circle x y → x + y ≤ a + b ∧
  a + b = 3 :=
sorry

end max_sum_on_circle_l3146_314618


namespace south_side_maximum_l3146_314639

/-- Represents the number of paths for each side of the mountain -/
structure MountainPaths where
  east : Nat
  west : Nat
  south : Nat
  north : Nat

/-- Calculates the number of ways to ascend and descend for a given side -/
def waysForSide (paths : MountainPaths) (side : Nat) : Nat :=
  side * (paths.east + paths.west + paths.south + paths.north - side)

/-- Theorem stating that the south side provides the maximum number of ways -/
theorem south_side_maximum (paths : MountainPaths) 
    (h1 : paths.east = 2)
    (h2 : paths.west = 3)
    (h3 : paths.south = 4)
    (h4 : paths.north = 1) :
  ∀ side, waysForSide paths paths.south ≥ waysForSide paths side :=
by
  sorry

#eval waysForSide { east := 2, west := 3, south := 4, north := 1 } 4

end south_side_maximum_l3146_314639


namespace quadratic_minimum_value_l3146_314605

-- Define the quadratic function
def quadratic_function (m x : ℝ) : ℝ := m * x^2 - 4 * x + 1

-- State the theorem
theorem quadratic_minimum_value (m : ℝ) :
  (∃ x_min : ℝ, ∀ x : ℝ, quadratic_function m x ≥ quadratic_function m x_min) ∧
  (∃ x_min : ℝ, quadratic_function m x_min = -3) →
  m = 1 := by
  sorry

end quadratic_minimum_value_l3146_314605


namespace sin_alpha_value_l3146_314604

theorem sin_alpha_value (α : Real) (h : 3 * Real.sin (2 * α) = Real.cos α) : 
  Real.sin α = 1/6 := by
  sorry

end sin_alpha_value_l3146_314604


namespace point_distance_from_origin_l3146_314649

theorem point_distance_from_origin (x : ℚ) : 
  |x| = (5 : ℚ) / 2 → x = (5 : ℚ) / 2 ∨ x = -(5 : ℚ) / 2 := by
  sorry

end point_distance_from_origin_l3146_314649


namespace solve_for_a_l3146_314638

theorem solve_for_a (x y a : ℚ) 
  (hx : x = 1)
  (hy : y = -2)
  (heq : 2 * x - a * y = 3) :
  a = 1 / 2 := by
  sorry

end solve_for_a_l3146_314638


namespace polynomial_factorization_l3146_314694

theorem polynomial_factorization (x : ℤ) : 
  x^4 + 3*x^3 - 15*x^2 - 19*x + 30 = (x+2)*(x+5)*(x-1)*(x-3) :=
by
  sorry

#check polynomial_factorization

end polynomial_factorization_l3146_314694


namespace at_least_one_perpendicular_l3146_314652

structure GeometricSpace where
  Plane : Type
  Line : Type
  Point : Type

variable {G : GeometricSpace}

-- Define the necessary relations
def perpendicular (α β : G.Plane) : Prop := sorry
def contains (α : G.Plane) (l : G.Line) : Prop := sorry
def perpendicular_lines (l₁ l₂ : G.Line) : Prop := sorry
def perpendicular_line_plane (l : G.Line) (α : G.Plane) : Prop := sorry

-- State the theorem
theorem at_least_one_perpendicular
  (α β : G.Plane) (n m : G.Line)
  (h1 : perpendicular α β)
  (h2 : contains α n)
  (h3 : contains β m)
  (h4 : perpendicular_lines m n) :
  perpendicular_line_plane n β ∨ perpendicular_line_plane m α :=
sorry

end at_least_one_perpendicular_l3146_314652


namespace dress_price_calculation_l3146_314668

-- Define the original price
def original_price : ℝ := 120

-- Define the discount rate
def discount_rate : ℝ := 0.30

-- Define the tax rate
def tax_rate : ℝ := 0.15

-- Define the total selling price
def total_selling_price : ℝ := original_price * (1 - discount_rate) * (1 + tax_rate)

-- Theorem to prove
theorem dress_price_calculation :
  total_selling_price = 96.6 := by sorry

end dress_price_calculation_l3146_314668


namespace complex_number_solution_l3146_314641

theorem complex_number_solution (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 1 - I) 
  (h₂ : z₁ * z₂ = 1 + I) : 
  z₂ = I := by sorry

end complex_number_solution_l3146_314641


namespace perfect_square_condition_l3146_314673

theorem perfect_square_condition (k : ℕ) : 
  (∃ m : ℤ, k^2 - 101*k = m^2) ↔ (k = 101 ∨ k = 2601) := by
  sorry

end perfect_square_condition_l3146_314673


namespace find_x_l3146_314651

theorem find_x : ∃ x : ℚ, (3 * x + 5) / 7 = 17 ∧ x = 38 := by
  sorry

end find_x_l3146_314651


namespace rectangle_area_reduction_l3146_314699

theorem rectangle_area_reduction (original_area : ℝ) 
  (h1 : original_area = 432) 
  (length_reduction : ℝ) (width_reduction : ℝ)
  (h2 : length_reduction = 0.15)
  (h3 : width_reduction = 0.20) : 
  original_area * (1 - length_reduction) * (1 - width_reduction) = 293.76 := by
  sorry

end rectangle_area_reduction_l3146_314699


namespace only_possible_knight_count_l3146_314617

/-- Represents a person on the island -/
inductive Person
| Knight
| Liar

/-- The total number of people on the island -/
def total_people : Nat := 2021

/-- A function that determines if a person's claim is true given their position and type -/
def claim_is_true (position : Nat) (person_type : Person) (num_knights : Nat) : Prop :=
  match person_type with
  | Person.Knight => total_people - position - (total_people - num_knights) > position - num_knights
  | Person.Liar => total_people - position - (total_people - num_knights) ≤ position - num_knights

/-- The main theorem stating that the only possible number of knights is 1010 -/
theorem only_possible_knight_count :
  ∃! num_knights : Nat,
    num_knights ≤ total_people ∧
    ∀ position : Nat, position < total_people →
      (claim_is_true position Person.Knight num_knights ∧ position < num_knights) ∨
      (claim_is_true position Person.Liar num_knights ∧ position ≥ num_knights) :=
by
  -- The proof goes here
  sorry

end only_possible_knight_count_l3146_314617


namespace remainder_of_B_l3146_314620

theorem remainder_of_B (A : ℕ) : (9 * A + 13) % 9 = 4 := by
  sorry

end remainder_of_B_l3146_314620


namespace girls_to_boys_ratio_l3146_314646

theorem girls_to_boys_ratio (girls boys : ℕ) : 
  girls = boys + 5 →
  girls + boys = 35 →
  girls * 3 = boys * 4 := by
sorry

end girls_to_boys_ratio_l3146_314646


namespace correct_substitution_l3146_314607

theorem correct_substitution (x y : ℝ) :
  (5 * x + 3 * y = 22) ∧ (y = x - 2) →
  5 * x + 3 * (x - 2) = 22 :=
by sorry

end correct_substitution_l3146_314607


namespace analytical_method_is_effect_to_cause_l3146_314609

/-- Represents the possible descriptions of the analytical method -/
inductive AnalyticalMethodDescription
  | causeToEffect
  | effectToCause
  | mutualInference
  | converseProof

/-- Definition of the analytical method -/
structure AnalyticalMethod :=
  (description : AnalyticalMethodDescription)
  (isReasoningMethod : Bool)

/-- Theorem stating that the analytical method is correctly described as reasoning from effect to cause -/
theorem analytical_method_is_effect_to_cause :
  ∀ (am : AnalyticalMethod), 
    am.isReasoningMethod = true → 
    am.description = AnalyticalMethodDescription.effectToCause :=
by
  sorry

end analytical_method_is_effect_to_cause_l3146_314609


namespace geometric_figures_sequence_l3146_314658

/-- The number of nonoverlapping unit squares in the nth figure -/
def f (n : ℕ) : ℕ := 2 * n^2 + 4 * n + 4

theorem geometric_figures_sequence :
  f 0 = 4 ∧ f 1 = 10 ∧ f 2 = 20 ∧ f 3 = 34 → f 150 = 45604 :=
by
  sorry

end geometric_figures_sequence_l3146_314658


namespace a_range_l3146_314678

/-- The function f as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x - 3*a else a^x - 2

/-- The theorem stating the range of a -/
theorem a_range (a : ℝ) :
  (a > 0 ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → f a x > f a y) ↔ (a > 0 ∧ a ≤ 1/3) :=
sorry

end a_range_l3146_314678


namespace investment_difference_l3146_314685

def initial_investment : ℝ := 500

def jackson_multiplier : ℝ := 4

def brandon_percentage : ℝ := 0.2

def jackson_final (initial : ℝ) (multiplier : ℝ) : ℝ := initial * multiplier

def brandon_final (initial : ℝ) (percentage : ℝ) : ℝ := initial * percentage

theorem investment_difference :
  jackson_final initial_investment jackson_multiplier - brandon_final initial_investment brandon_percentage = 1900 := by
  sorry

end investment_difference_l3146_314685


namespace calculation_proof_l3146_314696

theorem calculation_proof : (-49 : ℚ) * (4/7) - (4/7) / (-8/7) = -55/2 := by
  sorry

end calculation_proof_l3146_314696


namespace no_real_roots_l3146_314681

theorem no_real_roots (a b : ℝ) : ¬ ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end no_real_roots_l3146_314681


namespace constant_term_expansion_l3146_314627

/-- The constant term in the expansion of (x - 1/x)^6 -/
def constantTerm : ℤ := -20

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem constant_term_expansion :
  constantTerm = -binomial 6 3 := by sorry

end constant_term_expansion_l3146_314627


namespace line_parametrization_l3146_314689

/-- The slope of the line -/
def m : ℚ := 3/4

/-- The y-intercept of the line -/
def b : ℚ := -5

/-- The x-coordinate of the point on the line -/
def x₀ : ℚ := -8

/-- The y-component of the direction vector -/
def v : ℚ := 7

/-- The equation of the line -/
def line_eq (x y : ℚ) : Prop := y = m * x + b

/-- The parametric form of the line -/
def parametric_eq (s l t x y : ℚ) : Prop :=
  x = x₀ + t * l ∧ y = s + t * v

theorem line_parametrization (s l : ℚ) :
  (∀ t x y, parametric_eq s l t x y → line_eq x y) →
  s = -11 ∧ l = 28/3 := by sorry

end line_parametrization_l3146_314689


namespace trigonometric_equation_solution_l3146_314687

theorem trigonometric_equation_solution (x : ℝ) : 
  (∃ k : ℤ, x = 2 * π / 9 + 2 * π / 3 * k ∨ x = -2 * π / 9 + 2 * π / 3 * k) ↔ 
  Real.cos (3 * x - π / 6) - Real.sin (3 * x - π / 6) * Real.tan (π / 6) = 1 / (2 * Real.cos (7 * π / 6)) :=
by sorry

end trigonometric_equation_solution_l3146_314687


namespace johns_house_paintable_area_l3146_314656

/-- Calculates the total paintable wall area in John's house -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (non_paintable_area : ℝ) : ℝ :=
  num_bedrooms * (2 * (length * height + width * height) - non_paintable_area)

/-- Proves that the total paintable wall area in John's house is 1820 square feet -/
theorem johns_house_paintable_area :
  total_paintable_area 4 15 12 10 85 = 1820 := by
  sorry

#eval total_paintable_area 4 15 12 10 85

end johns_house_paintable_area_l3146_314656


namespace simplify_expression_1_simplify_expression_2_l3146_314606

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : (2*x + 1) * (2*x - 1) = 4*x^2 - 1 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x y : ℝ) : (x - 2*y)^2 - x*y = x^2 - 5*x*y + 4*y^2 := by
  sorry

end simplify_expression_1_simplify_expression_2_l3146_314606


namespace no_triangle_pairs_l3146_314688

/-- Given a set of n different elements, prove that if 4m ≤ n², 
    then there exists a set of m non-ordered pairs that do not form any triangles. -/
theorem no_triangle_pairs (n m : ℕ) (h : 4 * m ≤ n ^ 2) :
  ∃ (S : Finset (Fin n)) (P : Finset (Fin n × Fin n)),
    S.card = n ∧ 
    P.card = m ∧
    (∀ (p : Fin n × Fin n), p ∈ P → p.1 ≠ p.2) ∧
    (∀ (a b c : Fin n × Fin n), a ∈ P → b ∈ P → c ∈ P → 
      ¬(a.1 = b.1 ∧ b.2 = c.1 ∧ c.2 = a.2)) :=
by sorry

end no_triangle_pairs_l3146_314688


namespace pond_freezes_on_seventh_day_l3146_314615

/-- Represents a rectangular pond with given dimensions and freezing properties -/
structure Pond where
  length : ℝ
  width : ℝ
  daily_freeze_distance : ℝ
  first_day_freeze_percent : ℝ
  second_day_freeze_percent : ℝ

/-- Calculates the day when the pond is completely frozen -/
def freezing_day (p : Pond) : ℕ :=
  sorry

/-- Theorem stating that the pond will be completely frozen on the 7th day -/
theorem pond_freezes_on_seventh_day (p : Pond) 
  (h1 : p.length * p.width = 5000)
  (h2 : p.length + p.width = 70.5)
  (h3 : p.daily_freeze_distance = 10)
  (h4 : p.first_day_freeze_percent = 0.202)
  (h5 : p.second_day_freeze_percent = 0.186) : 
  freezing_day p = 7 :=
sorry

end pond_freezes_on_seventh_day_l3146_314615


namespace brenda_erasers_count_l3146_314647

/-- The number of groups Brenda creates -/
def num_groups : ℕ := 3

/-- The number of erasers in each group -/
def erasers_per_group : ℕ := 90

/-- The total number of erasers Brenda has -/
def total_erasers : ℕ := num_groups * erasers_per_group

theorem brenda_erasers_count : total_erasers = 270 := by
  sorry

end brenda_erasers_count_l3146_314647


namespace fangfang_floor_climb_l3146_314603

def time_between_floors (start_floor end_floor : ℕ) (time : ℝ) : Prop :=
  time = (end_floor - start_floor) * 15

theorem fangfang_floor_climb : 
  time_between_floors 1 3 30 → time_between_floors 2 6 60 :=
by
  sorry

end fangfang_floor_climb_l3146_314603


namespace tangent_circle_equation_l3146_314692

/-- A circle C tangent to the y-axis at point (0,2) and tangent to the line 4x-3y+9=0 -/
structure TangentCircle where
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The radius of the circle -/
  radius : ℝ
  /-- The circle is tangent to the y-axis at (0,2) -/
  tangent_y_axis : center.1 = radius
  /-- The circle's center is on the line y=2 -/
  center_on_line : center.2 = 2
  /-- The circle is tangent to the line 4x-3y+9=0 -/
  tangent_line : |4 * center.1 - 3 * center.2 + 9| / Real.sqrt 25 = radius

/-- The standard equation of the circle is either (x-3)^2+(y-2)^2=9 or (x+1/3)^2+(y-2)^2=1/9 -/
theorem tangent_circle_equation (c : TangentCircle) :
  (c.center = (3, 2) ∧ c.radius = 3) ∨ (c.center = (-1/3, 2) ∧ c.radius = 1/3) := by
  sorry

end tangent_circle_equation_l3146_314692


namespace men_in_room_l3146_314621

/-- Represents the number of people in a room -/
structure RoomPopulation where
  men : ℕ
  women : ℕ

/-- Calculates the final number of men in the room -/
def finalMenCount (initial : RoomPopulation) : ℕ :=
  initial.men + 2

/-- Theorem: Given the initial conditions and final number of women,
    prove that there are 14 men in the room -/
theorem men_in_room (initial : RoomPopulation) 
    (h1 : initial.men = 4 * initial.women / 5)  -- Initial ratio 4:5
    (h2 : 2 * (initial.women - 3) = 24)         -- Final women count after changes
    : finalMenCount initial = 14 := by
  sorry


end men_in_room_l3146_314621


namespace milk_production_increase_l3146_314663

/-- Given the initial milk production rate and an increase in production rate,
    calculate the new amount of milk produced by double the cows in triple the time. -/
theorem milk_production_increase (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let initial_rate := y / (x * z)
  let increased_rate := initial_rate * 1.1
  let new_production := 2 * x * increased_rate * 3 * z
  new_production = 6.6 * y := by sorry

end milk_production_increase_l3146_314663


namespace max_sum_of_squares_l3146_314654

/-- Given a system of equations, prove that the maximum value of a^2 + b^2 + c^2 + d^2 is 258 -/
theorem max_sum_of_squares (a b c d : ℝ) 
  (eq1 : a + b = 17)
  (eq2 : a * b + c + d = 86)
  (eq3 : a * d + b * c = 180)
  (eq4 : c * d = 110) :
  a^2 + b^2 + c^2 + d^2 ≤ 258 ∧ ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 258 ∧ 
    a + b = 17 ∧ a * b + c + d = 86 ∧ a * d + b * c = 180 ∧ c * d = 110 := by
  sorry


end max_sum_of_squares_l3146_314654


namespace proportion_equation_l3146_314631

theorem proportion_equation (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 2 * x = 3 * y) :
  x / 3 = y / 2 := by
  sorry

end proportion_equation_l3146_314631


namespace girls_in_class_l3146_314661

theorem girls_in_class (total_students : ℕ) (girl_ratio boy_ratio : ℕ) : 
  total_students = 36 → 
  girl_ratio = 4 → 
  boy_ratio = 5 → 
  (girl_ratio + boy_ratio : ℚ) * (total_students / (girl_ratio + boy_ratio : ℕ)) = girl_ratio * (total_students / (girl_ratio + boy_ratio : ℕ)) →
  girl_ratio * (total_students / (girl_ratio + boy_ratio : ℕ)) = 16 := by
sorry

end girls_in_class_l3146_314661


namespace net_calorie_intake_l3146_314610

/-- Calculate net calorie intake after jogging -/
theorem net_calorie_intake
  (breakfast_calories : ℕ)
  (jogging_time : ℕ)
  (calorie_burn_rate : ℕ)
  (h1 : breakfast_calories = 900)
  (h2 : jogging_time = 30)
  (h3 : calorie_burn_rate = 10) :
  breakfast_calories - jogging_time * calorie_burn_rate = 600 :=
by sorry

end net_calorie_intake_l3146_314610


namespace empty_box_weight_l3146_314684

-- Define the number of balls
def num_balls : ℕ := 30

-- Define the weight of each ball in kg
def ball_weight : ℝ := 0.36

-- Define the total weight of the box with balls in kg
def total_weight : ℝ := 11.26

-- Theorem to prove
theorem empty_box_weight :
  total_weight - (num_balls : ℝ) * ball_weight = 0.46 := by
  sorry

end empty_box_weight_l3146_314684


namespace simplify_fraction_l3146_314637

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -3) :
  (x + 4) / (x^2 + 3*x) - 1 / (3*x + x^2) = 1 / x := by
  sorry

end simplify_fraction_l3146_314637


namespace series_sum_equals_ln2_minus_half_l3146_314632

open Real

/-- The sum of the series Σ(1/((2n-1) * 2n * (2n+1))) for n from 1 to infinity -/
noncomputable def seriesSum : ℝ := ∑' n, 1 / ((2*n - 1) * (2*n) * (2*n + 1))

/-- Theorem stating that the sum of the series equals ln 2 - 1/2 -/
theorem series_sum_equals_ln2_minus_half : seriesSum = log 2 - 1/2 := by
  sorry

end series_sum_equals_ln2_minus_half_l3146_314632


namespace area_enclosed_by_functions_l3146_314693

/-- The area enclosed by y = x and f(x) = 2 - x^2 -/
theorem area_enclosed_by_functions : ∃ (a : ℝ), a = (9 : ℝ) / 2 ∧ 
  a = ∫ x in (-2 : ℝ)..1, (2 - x^2 - x) := by sorry

end area_enclosed_by_functions_l3146_314693


namespace circle_center_and_radius_l3146_314669

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the circle --/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x - 8*y + 9

theorem circle_center_and_radius :
  ∃ (c : Circle), (∀ (x y : ℝ), circle_equation x y ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
                   c.center = (3, -4) ∧
                   c.radius = Real.sqrt 34 := by
  sorry

end circle_center_and_radius_l3146_314669


namespace negation_of_positive_sum_l3146_314642

theorem negation_of_positive_sum (x y : ℝ) :
  (¬(x > 0 ∧ y > 0 → x + y > 0)) ↔ (x ≤ 0 ∨ y ≤ 0 → x + y ≤ 0) :=
by sorry

end negation_of_positive_sum_l3146_314642


namespace fraction_value_at_three_l3146_314675

theorem fraction_value_at_three : 
  let x : ℝ := 3
  (x^12 + 18*x^6 + 81) / (x^6 + 9) = 738 := by
sorry

end fraction_value_at_three_l3146_314675


namespace kevin_cards_l3146_314602

/-- The number of cards Kevin ends up with given his initial cards and found cards -/
def total_cards (initial : ℕ) (found : ℕ) : ℕ :=
  initial + found

/-- Theorem stating that Kevin ends up with 54 cards -/
theorem kevin_cards : total_cards 7 47 = 54 := by
  sorry

end kevin_cards_l3146_314602


namespace triangle_angle_measure_l3146_314644

theorem triangle_angle_measure (a b c : ℝ) : 
  a + b + c = 180 →  -- sum of angles in a triangle is 180°
  a = 40 →           -- one angle is 40°
  b = 2 * c →        -- one angle is twice the other
  c = 140 / 3 :=     -- prove that the third angle is 140/3°
by sorry

end triangle_angle_measure_l3146_314644


namespace sams_remaining_pennies_l3146_314665

/-- Given an initial number of pennies and a number of spent pennies,
    calculate the remaining number of pennies. -/
def remaining_pennies (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem stating that Sam's remaining pennies are 5 given the initial and spent amounts. -/
theorem sams_remaining_pennies :
  remaining_pennies 98 93 = 5 := by
  sorry

end sams_remaining_pennies_l3146_314665


namespace right_triangle_sets_l3146_314679

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  ¬ is_right_triangle 1.1 1.5 1.9 ∧
  ¬ is_right_triangle 5 11 12 ∧
  is_right_triangle 1.2 1.6 2.0 ∧
  ¬ is_right_triangle 3 4 8 :=
by sorry

end right_triangle_sets_l3146_314679


namespace april_coffee_cost_l3146_314674

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the cost of coffee for a given day -/
def coffeeCost (day: DayOfWeek) (isEarthDay: Bool) : ℚ :=
  match day with
  | DayOfWeek.Monday => 3.5
  | DayOfWeek.Friday => 3
  | _ => if isEarthDay then 3 else 4

/-- Calculates the total cost of coffee for April -/
def aprilCoffeeCost (startDay: DayOfWeek) : ℚ :=
  sorry

/-- Theorem stating that Jon's total spending on coffee in April is $112 -/
theorem april_coffee_cost :
  aprilCoffeeCost DayOfWeek.Thursday = 112 := by
  sorry

end april_coffee_cost_l3146_314674


namespace min_t_value_l3146_314628

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-3, 2]
def interval : Set ℝ := Set.Icc (-3) 2

-- Statement: The minimum value of t that satisfies |f(x₁) - f(x₂)| ≤ t for all x₁, x₂ in the interval is 20
theorem min_t_value : 
  (∃ t : ℝ, ∀ x₁ x₂ : ℝ, x₁ ∈ interval → x₂ ∈ interval → |f x₁ - f x₂| ≤ t) ∧ 
  (∀ t : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ interval → x₂ ∈ interval → |f x₁ - f x₂| ≤ t) → t ≥ 20) :=
by sorry

end min_t_value_l3146_314628


namespace f_is_even_l3146_314664

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_is_even (g : ℝ → ℝ) (h : isEven g) :
  isEven (fun x ↦ |g (x^3)|) := by sorry

end f_is_even_l3146_314664


namespace triangle_pqr_rotation_l3146_314657

/-- Triangle PQR with given properties and rotation of PQ --/
theorem triangle_pqr_rotation (P Q R : ℝ × ℝ) (h1 : P = (0, 0)) (h2 : R = (8, 0))
  (h3 : Q.1 ≥ 0 ∧ Q.2 ≥ 0) -- Q in first quadrant
  (h4 : (Q.1 - R.1) * (Q.2 - R.2) = 0) -- ∠QRP = 90°
  (h5 : (Q.2 - P.2) = (Q.1 - P.1)) -- ∠QPR = 45°
  : (- Q.2, Q.1) = (-8, 8) := by
  sorry

end triangle_pqr_rotation_l3146_314657


namespace club_officer_selection_l3146_314680

/-- The number of ways to select officers in a club with special conditions -/
def select_officers (total_members : ℕ) (officers_needed : ℕ) (special_members : ℕ) : ℕ :=
  let remaining_members := total_members - special_members
  let case1 := remaining_members * (remaining_members - 1) * (remaining_members - 2) * (remaining_members - 3)
  let case2 := officers_needed * (officers_needed - 1) * (officers_needed - 2) * remaining_members
  case1 + case2

/-- Theorem stating the number of ways to select officers under given conditions -/
theorem club_officer_selection :
  select_officers 25 4 3 = 176088 :=
sorry

end club_officer_selection_l3146_314680


namespace f_max_value_l3146_314630

/-- A function f(x) with specific properties --/
def f (a b : ℝ) (x : ℝ) : ℝ := (4 - x^2) * (a * x^2 + b * x + 5)

/-- The theorem stating the maximum value of f(x) --/
theorem f_max_value (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (-3 - x)) →  -- Symmetry condition
  (∃ M : ℝ, ∀ x : ℝ, f a b x ≤ M ∧ ∃ x₀ : ℝ, f a b x₀ = M) →  -- Maximum exists
  (∃ M : ℝ, M = 36 ∧ ∀ x : ℝ, f a b x ≤ M ∧ ∃ x₀ : ℝ, f a b x₀ = M) :=
by
  sorry


end f_max_value_l3146_314630


namespace inequality_solution_set_l3146_314659

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + 2| < a) → a > 3 :=
by sorry

end inequality_solution_set_l3146_314659


namespace not_domain_zero_to_three_l3146_314667

/-- The function f(x) = x^2 - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The theorem stating that [0, 3] cannot be the domain of f(x) given its value range is [1, 2] -/
theorem not_domain_zero_to_three :
  (∀ y ∈ Set.Icc 1 2, ∃ x, f x = y) →
  ¬(∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc 1 2) :=
by sorry

end not_domain_zero_to_three_l3146_314667


namespace necessary_not_sufficient_condition_l3146_314648

-- Define the propositions p and q
def p (x : ℝ) : Prop := x / (x - 2) < 0
def q (x m : ℝ) : Prop := 0 < x ∧ x < m

-- Define the set of x that satisfy p
def set_p : Set ℝ := {x | p x}

-- Define the set of x that satisfy q
def set_q (m : ℝ) : Set ℝ := {x | q x m}

-- State the theorem
theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ x, q x m → p x) ∧ (∃ x, p x ∧ ¬q x m) → m > 2 :=
by sorry

end necessary_not_sufficient_condition_l3146_314648


namespace quadratic_inequality_range_l3146_314616

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end quadratic_inequality_range_l3146_314616


namespace min_fraction_sum_l3146_314613

def is_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

def are_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem min_fraction_sum (A B C D : ℕ) 
  (h1 : is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D) 
  (h2 : are_distinct A B C D) :
  (A : ℚ) / B + (C : ℚ) / D ≥ 31 / 56 := by
  sorry

end min_fraction_sum_l3146_314613


namespace bhupathi_amount_l3146_314619

theorem bhupathi_amount (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : A + B = 1210) (h4 : (4/15) * A = (2/5) * B) : B = 484 :=
by
  sorry

end bhupathi_amount_l3146_314619


namespace consecutive_integers_median_l3146_314660

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (median : ℕ) : 
  n = 81 → sum = 9^5 → sum = n * median → median = 729 := by
  sorry

end consecutive_integers_median_l3146_314660


namespace volleyball_scoring_l3146_314601

/-- Volleyball scoring problem -/
theorem volleyball_scoring (L : ℕ) : 
  (∃ (N A : ℕ),
    N = L + 3 ∧ 
    A = 2 * (L + N) ∧ 
    L + N + A + 17 = 50) → 
  L = 6 := by
  sorry

end volleyball_scoring_l3146_314601


namespace triangle_right_angled_l3146_314633

theorem triangle_right_angled (α β γ : Real) : 
  0 < α ∧ 0 < β ∧ 0 < γ →  -- Angles are positive
  α + β + γ = Real.pi →    -- Sum of angles in a triangle
  (Real.sin α + Real.sin β) / (Real.cos α + Real.cos β) = Real.sin γ →
  γ = Real.pi / 2 := by
sorry

end triangle_right_angled_l3146_314633


namespace soccer_handshakes_l3146_314650

/-- Calculates the total number of handshakes in a soccer match -/
theorem soccer_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  team_size = 11 → num_teams = 2 → num_referees = 3 →
  (team_size * team_size * (num_teams - 1) / 2) + (team_size * num_teams * num_referees) = 187 := by
  sorry

end soccer_handshakes_l3146_314650


namespace probability_of_three_in_eight_elevenths_l3146_314645

def decimal_representation (n d : ℕ) : List ℕ :=
  sorry -- Implementation of decimal representation calculation

theorem probability_of_three_in_eight_elevenths (n d : ℕ) (h : n = 8 ∧ d = 11) :
  let rep := decimal_representation n d
  (rep.count 3) / rep.length = 0 :=
sorry

end probability_of_three_in_eight_elevenths_l3146_314645


namespace camp_total_is_250_l3146_314622

/-- Represents the distribution of students in a boys camp --/
structure CampDistribution where
  total : ℕ
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ
  schoolAScience : ℕ
  schoolAMath : ℕ
  schoolALiterature : ℕ
  schoolBScience : ℕ
  schoolBMath : ℕ
  schoolBLiterature : ℕ
  schoolCScience : ℕ
  schoolCMath : ℕ
  schoolCLiterature : ℕ

/-- The camp distribution satisfies the given conditions --/
def isValidDistribution (d : CampDistribution) : Prop :=
  d.schoolA = d.total / 5 ∧
  d.schoolB = 3 * d.total / 10 ∧
  d.schoolC = d.total / 2 ∧
  d.schoolAScience = 3 * d.schoolA / 10 ∧
  d.schoolAMath = 2 * d.schoolA / 5 ∧
  d.schoolALiterature = 3 * d.schoolA / 10 ∧
  d.schoolBScience = d.schoolB / 4 ∧
  d.schoolBMath = 7 * d.schoolB / 20 ∧
  d.schoolBLiterature = 2 * d.schoolB / 5 ∧
  d.schoolCScience = 3 * d.schoolC / 20 ∧
  d.schoolCMath = d.schoolC / 2 ∧
  d.schoolCLiterature = 7 * d.schoolC / 20 ∧
  d.schoolA - d.schoolAScience = 35 ∧
  d.schoolBMath = 20

/-- Theorem: Given the conditions, the total number of boys in the camp is 250 --/
theorem camp_total_is_250 (d : CampDistribution) (h : isValidDistribution d) : d.total = 250 := by
  sorry


end camp_total_is_250_l3146_314622


namespace pizza_order_cost_is_185_l3146_314634

/-- Represents the cost calculation for a pizza order with special offers --/
def pizza_order_cost (
  large_pizza_price : ℚ)
  (medium_pizza_price : ℚ)
  (small_pizza_price : ℚ)
  (topping_price : ℚ)
  (drink_price : ℚ)
  (garlic_bread_price : ℚ)
  (triple_cheese_count : ℕ)
  (triple_cheese_toppings : ℕ)
  (meat_lovers_count : ℕ)
  (meat_lovers_toppings : ℕ)
  (veggie_delight_count : ℕ)
  (veggie_delight_toppings : ℕ)
  (drink_count : ℕ)
  (garlic_bread_count : ℕ) : ℚ :=
  let triple_cheese_cost := (triple_cheese_count / 2) * large_pizza_price + triple_cheese_count * triple_cheese_toppings * topping_price
  let meat_lovers_cost := ((meat_lovers_count + 2) / 3) * medium_pizza_price + meat_lovers_count * meat_lovers_toppings * topping_price
  let veggie_delight_cost := ((veggie_delight_count * 3) / 5) * small_pizza_price + veggie_delight_count * veggie_delight_toppings * topping_price
  let drink_and_bread_cost := drink_count * drink_price + max 0 (garlic_bread_count - drink_count) * garlic_bread_price
  triple_cheese_cost + meat_lovers_cost + veggie_delight_cost + drink_and_bread_cost

/-- Theorem stating that the given order costs $185 --/
theorem pizza_order_cost_is_185 :
  pizza_order_cost 10 8 5 (5/2) 2 4 6 2 4 3 10 1 8 5 = 185 := by
  sorry

end pizza_order_cost_is_185_l3146_314634


namespace cylinder_height_ratio_l3146_314677

/-- Given a cylinder whose radius is tripled and whose new volume is 18 times the original,
    prove that the ratio of the new height to the original height is 2:1. -/
theorem cylinder_height_ratio 
  (r : ℝ) -- original radius
  (h : ℝ) -- original height
  (h' : ℝ) -- new height
  (volume_ratio : ℝ) -- ratio of new volume to old volume
  (h_pos : 0 < h) -- ensure original height is positive
  (r_pos : 0 < r) -- ensure original radius is positive
  (volume_eq : π * (3 * r)^2 * h' = volume_ratio * (π * r^2 * h)) -- volume equation
  (volume_ratio_eq : volume_ratio = 18) -- new volume is 18 times the old one
  : h' / h = 2 := by
sorry

end cylinder_height_ratio_l3146_314677


namespace sum_inequality_l3146_314662

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - 2*a) / (a^2 + b*c) + 
  (c + a - 2*b) / (b^2 + c*a) + 
  (a + b - 2*c) / (c^2 + a*b) ≥ 0 := by
  sorry

end sum_inequality_l3146_314662


namespace power_of_three_mod_eleven_l3146_314653

theorem power_of_three_mod_eleven : 3^1320 % 11 = 1 := by
  sorry

end power_of_three_mod_eleven_l3146_314653


namespace yoongis_calculation_l3146_314640

theorem yoongis_calculation (x : ℝ) : x / 9 = 30 → x - 37 = 233 := by
  sorry

end yoongis_calculation_l3146_314640


namespace milburg_population_l3146_314611

/-- The total population of Milburg -/
def total_population (adults children teenagers seniors : ℕ) : ℕ :=
  adults + children + teenagers + seniors

/-- Theorem: The total population of Milburg is 12,292 -/
theorem milburg_population : total_population 5256 2987 1709 2340 = 12292 := by
  sorry

end milburg_population_l3146_314611


namespace smallest_x_for_cube_1680x_l3146_314608

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_x_for_cube_1680x : 
  (∀ x : ℕ, x > 0 ∧ x < 44100 → ¬ is_perfect_cube (1680 * x)) ∧ 
  is_perfect_cube (1680 * 44100) := by
sorry

end smallest_x_for_cube_1680x_l3146_314608


namespace least_k_for_inequality_l3146_314683

theorem least_k_for_inequality : ∃ k : ℤ, (∀ j : ℤ, 0.00010101 * (10 : ℝ)^j > 10 → k ≤ j) ∧ 0.00010101 * (10 : ℝ)^k > 10 ∧ k = 6 :=
by sorry

end least_k_for_inequality_l3146_314683


namespace constant_function_proof_l3146_314600

theorem constant_function_proof (f : ℚ → ℚ) 
  (h1 : ∀ x y : ℚ, f (x + y) = f x + f y + 2547)
  (h2 : f 2004 = 2547) : 
  f 2547 = 2547 := by sorry

end constant_function_proof_l3146_314600


namespace intersection_condition_l3146_314614

-- Define the parabola C: y^2 = x
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the line l: y = kx + 1
def line (k x y : ℝ) : Prop := y = k * x + 1

-- Define the condition for two distinct intersection points
def has_two_distinct_intersections (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    parabola x₁ y₁ ∧ line k x₁ y₁ ∧
    parabola x₂ y₂ ∧ line k x₂ y₂

-- Theorem statement
theorem intersection_condition :
  (∀ k : ℝ, has_two_distinct_intersections k → k ≠ 0) ∧
  (∃ k : ℝ, k ≠ 0 ∧ ¬has_two_distinct_intersections k) :=
sorry

end intersection_condition_l3146_314614


namespace four_digit_count_is_900_l3146_314676

/-- The count of four-digit positive integers with thousands digit 3 and non-zero hundreds digit -/
def four_digit_count : ℕ :=
  let thousands_digit := 3
  let hundreds_choices := 9  -- 1 to 9
  let tens_choices := 10     -- 0 to 9
  let ones_choices := 10     -- 0 to 9
  hundreds_choices * tens_choices * ones_choices

theorem four_digit_count_is_900 : four_digit_count = 900 := by
  sorry

end four_digit_count_is_900_l3146_314676


namespace claire_cleaning_hours_l3146_314612

/-- Calculates the hours spent cleaning given Claire's daily schedule. -/
def hours_cleaning (total_day_hours sleep_hours cooking_hours crafting_hours : ℕ) : ℕ :=
  let working_hours := total_day_hours - sleep_hours
  let cleaning_hours := working_hours - cooking_hours - 2 * crafting_hours
  cleaning_hours

/-- Theorem stating that Claire spends 4 hours cleaning given her schedule. -/
theorem claire_cleaning_hours :
  hours_cleaning 24 8 2 5 = 4 := by
  sorry

#eval hours_cleaning 24 8 2 5

end claire_cleaning_hours_l3146_314612


namespace max_vector_sum_diff_l3146_314629

/-- Given plane vectors a, b, and c satisfying the specified conditions,
    the maximum value of |a + b - c| is 3√2. -/
theorem max_vector_sum_diff (a b c : ℝ × ℝ) 
  (h1 : ‖a‖ = ‖b‖ ∧ ‖a‖ ≠ 0)
  (h2 : a.1 * b.1 + a.2 * b.2 = 0)  -- dot product = 0 means perpendicular
  (h3 : ‖c‖ = 2 * Real.sqrt 2)
  (h4 : ‖c - a‖ = 1) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧ 
  ∀ (x : ℝ × ℝ), x = a + b - c → ‖x‖ ≤ max :=
sorry

end max_vector_sum_diff_l3146_314629


namespace prism_volume_is_400_l3146_314626

/-- The volume of a right rectangular prism with face areas 40, 50, and 80 square centimeters -/
def prism_volume : ℝ := 400

/-- The areas of the three faces of the prism -/
def face_area_1 : ℝ := 40
def face_area_2 : ℝ := 50
def face_area_3 : ℝ := 80

/-- Theorem: The volume of the prism is 400 cubic centimeters -/
theorem prism_volume_is_400 :
  ∃ (a b c : ℝ),
    a * b = face_area_1 ∧
    a * c = face_area_2 ∧
    b * c = face_area_3 ∧
    a * b * c = prism_volume :=
by sorry

end prism_volume_is_400_l3146_314626


namespace tangent_line_correct_l3146_314643

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 3)

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 2*x - y + 1 = 0

/-- Theorem stating that the tangent line equation is correct -/
theorem tangent_line_correct :
  let (a, b) := point
  tangent_line a b ∧
  ∀ x, tangent_line x (f x) → x = a := by sorry

end tangent_line_correct_l3146_314643


namespace number_line_points_l3146_314671

/-- Represents a point on a number line -/
structure Point where
  value : ℚ

/-- The distance between two points on a number line -/
def distance (p q : Point) : ℚ := q.value - p.value

theorem number_line_points (A B C : Point)
  (hA : A.value = 2)
  (hAB : distance A B = -7)
  (hBC : distance B C = 1 + 2/3) :
  B.value = -5 ∧ C.value = -10/3 := by
  sorry

end number_line_points_l3146_314671


namespace greatest_distance_between_circle_centers_l3146_314624

/-- The greatest distance between centers of two circles in a rectangle --/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 18)
  (h_height : rectangle_height = 15)
  (h_diameter : circle_diameter = 7)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = Real.sqrt 185 ∧
    ∀ (d' : ℝ), d' ≤ d :=
by sorry

end greatest_distance_between_circle_centers_l3146_314624


namespace water_bottle_volume_l3146_314686

theorem water_bottle_volume (total_cost : ℝ) (num_bottles : ℕ) (price_per_liter : ℝ) 
  (h1 : total_cost = 12)
  (h2 : num_bottles = 6)
  (h3 : price_per_liter = 1) :
  (total_cost / (num_bottles : ℝ)) / price_per_liter = 2 := by
  sorry

end water_bottle_volume_l3146_314686


namespace max_k_value_l3146_314690

open Real

noncomputable def f (x : ℝ) : ℝ := exp x - x - 2

theorem max_k_value (k : ℤ) :
  (∀ x : ℝ, x > 0 → (k - x) / (x + 1) * (exp x - 1) < 1) →
  k ≤ 2 :=
sorry

end max_k_value_l3146_314690


namespace multiple_indecomposable_factorizations_l3146_314655

/-- The set V_n for a given n -/
def V_n (n : ℕ) : Set ℕ := {m : ℕ | ∃ k : ℕ+, m = 1 + k * n}

/-- A number is indecomposable in V_n if it cannot be expressed as the product of two members of V_n -/
def Indecomposable (m : ℕ) (n : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ a b : ℕ, a ∈ V_n n → b ∈ V_n n → a * b ≠ m

/-- There exists a number in V_n with multiple indecomposable factorizations -/
theorem multiple_indecomposable_factorizations (n : ℕ) (h : n > 2) :
  ∃ m : ℕ, m ∈ V_n n ∧
    ∃ (a b c d : ℕ),
      Indecomposable a n ∧ Indecomposable b n ∧ Indecomposable c n ∧ Indecomposable d n ∧
      a * b = m ∧ c * d = m ∧ (a ≠ c ∨ b ≠ d) :=
  sorry

end multiple_indecomposable_factorizations_l3146_314655


namespace distribute_toys_count_l3146_314672

/-- The number of ways to distribute 4 toys out of 6 distinct toys to 4 distinct people -/
def distribute_toys : ℕ :=
  Nat.factorial 6 / Nat.factorial 2

/-- Theorem stating that distributing 4 toys out of 6 distinct toys to 4 distinct people results in 360 different arrangements -/
theorem distribute_toys_count : distribute_toys = 360 := by
  sorry

end distribute_toys_count_l3146_314672


namespace bicycle_problem_l3146_314623

/-- Prove that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) 
  (h1 : distance = 12)
  (h2 : speed_ratio = 1.2)
  (h3 : time_difference = 1/6) :
  ∃ (speed_B : ℝ), 
    distance / speed_B - distance / (speed_ratio * speed_B) = time_difference ∧ 
    speed_B = 12 := by
sorry

end bicycle_problem_l3146_314623


namespace max_profit_l3146_314698

noncomputable section

-- Define the cost function G(x)
def G (x : ℝ) : ℝ := 2.8 + x

-- Define the revenue function R(x)
def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x
  else 11

-- Define the profit function f(x)
def f (x : ℝ) : ℝ := R x - G x

-- Theorem stating the maximum profit and the corresponding production quantity
theorem max_profit :
  ∃ (x_max : ℝ), x_max = 4 ∧
  ∀ (x : ℝ), 0 ≤ x → f x ≤ f x_max ∧
  f x_max = 3.6 :=
sorry

end

end max_profit_l3146_314698


namespace unique_solution_iff_b_less_than_two_l3146_314695

/-- The equation has exactly one real solution -/
def has_unique_real_solution (b : ℝ) : Prop :=
  ∃! x : ℝ, x^3 - b*x^2 - 3*b*x + b^2 - 4 = 0

/-- The main theorem -/
theorem unique_solution_iff_b_less_than_two :
  ∀ b : ℝ, has_unique_real_solution b ↔ b < 2 := by sorry

end unique_solution_iff_b_less_than_two_l3146_314695


namespace factor_81_minus_27x_cubed_l3146_314666

theorem factor_81_minus_27x_cubed (x : ℝ) :
  81 - 27 * x^3 = 3 * (3 - x) * (81 + 27 * x + 9 * x^2) := by
  sorry

end factor_81_minus_27x_cubed_l3146_314666


namespace unique_solution_equation_l3146_314697

theorem unique_solution_equation (x : ℝ) (h1 : x ≠ 0) :
  (9 * x) ^ 18 = (27 * x) ^ 9 ↔ x = 1 / 3 := by
  sorry

end unique_solution_equation_l3146_314697


namespace average_marks_math_chem_l3146_314691

theorem average_marks_math_chem (M P C B : ℕ) : 
  M + P = 80 →
  C + B = 120 →
  C = P + 20 →
  B = M - 15 →
  (M + C) / 2 = 50 := by
sorry

end average_marks_math_chem_l3146_314691


namespace small_circle_radius_l3146_314625

/-- Given a large circle of radius 10 meters containing seven congruent smaller circles
    arranged with six forming a hexagon around one central circle, prove that the radius
    of each smaller circle is 5 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) :
  R = 10 ∧  -- Radius of the large circle
  (2 * r + 2 * r = 2 * R) →  -- Diameter of large circle equals two radii plus one diameter of small circles
  r = 5 :=
by sorry

end small_circle_radius_l3146_314625


namespace parabola_chords_fixed_point_and_isosceles_triangle_l3146_314635

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point A
def point_A : ℝ × ℝ := (1, 2)

-- Define a chord on the parabola passing through A
def chord_through_A (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2 ∧ parabola point_A.1 point_A.2

-- Define perpendicularity of two chords
def perpendicular_chords (P Q : ℝ × ℝ) : Prop :=
  (P.1 - point_A.1) * (Q.1 - point_A.1) + (P.2 - point_A.2) * (Q.2 - point_A.2) = 0

-- Define the point T
def point_T : ℝ × ℝ := (5, -2)

-- Define a line passing through a point
def line_through_point (P Q : ℝ × ℝ) (T : ℝ × ℝ) : Prop :=
  (T.2 - P.2) * (Q.1 - P.1) = (Q.2 - P.2) * (T.1 - P.1)

-- Define an isosceles triangle
def isosceles_triangle (P Q : ℝ × ℝ) : Prop :=
  (P.1 - point_A.1)^2 + (P.2 - point_A.2)^2 = (Q.1 - point_A.1)^2 + (Q.2 - point_A.2)^2

-- Theorem statement
theorem parabola_chords_fixed_point_and_isosceles_triangle
  (P Q : ℝ × ℝ)
  (h1 : chord_through_A P)
  (h2 : chord_through_A Q)
  (h3 : perpendicular_chords P Q)
  (h4 : line_through_point P Q point_T) :
  (∀ R : ℝ × ℝ, chord_through_A R → perpendicular_chords P R → line_through_point P R point_T) ∧
  (∃! R : ℝ × ℝ, chord_through_A R ∧ perpendicular_chords P R ∧ isosceles_triangle P R) :=
sorry

end parabola_chords_fixed_point_and_isosceles_triangle_l3146_314635
