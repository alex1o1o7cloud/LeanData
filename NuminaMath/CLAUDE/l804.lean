import Mathlib

namespace class_size_is_30_l804_80480

/-- Represents the capacity of a hotel room -/
structure RoomCapacity where
  queen_bed_capacity : Nat
  queen_beds : Nat
  couch_capacity : Nat

/-- Calculates the total number of students in a class given room requirements -/
def calculate_class_size (room_capacity : RoomCapacity) (rooms_booked : Nat) : Nat :=
  (room_capacity.queen_bed_capacity * room_capacity.queen_beds + room_capacity.couch_capacity) * rooms_booked

/-- Theorem stating that the class size is 30 given the specific room configuration and booking requirements -/
theorem class_size_is_30 :
  let room_capacity : RoomCapacity := { queen_bed_capacity := 2, queen_beds := 2, couch_capacity := 1 }
  let rooms_booked := 6
  calculate_class_size room_capacity rooms_booked = 30 := by
  sorry


end class_size_is_30_l804_80480


namespace PQ_length_l804_80497

-- Define the triangles and their properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle : ℝ

-- Define the given triangles
def triangle_PQR : Triangle := {
  a := 8,   -- PR
  b := 10,  -- QR
  c := 5,   -- PQ (to be proved)
  angle := 60
}

def triangle_STU : Triangle := {
  a := 3,   -- SU
  b := 4,   -- TU (derived from similarity)
  c := 2,   -- ST
  angle := 60
}

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop :=
  t1.angle = t2.angle ∧ t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

-- Theorem statement
theorem PQ_length :
  similar triangle_PQR triangle_STU →
  triangle_PQR.c = 5 := by sorry

end PQ_length_l804_80497


namespace abc_inequality_l804_80446

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c) (sum_sq : a^2 + b^2 + c^2 = 9) :
  a * b * c + 1 > 3 * a :=
by sorry

end abc_inequality_l804_80446


namespace train_length_l804_80406

/-- Given a train that crosses a 300-meter platform in 36 seconds and a signal pole in 18 seconds, 
    prove that the length of the train is 300 meters. -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
    (h1 : platform_length = 300)
    (h2 : platform_time = 36)
    (h3 : pole_time = 18) : 
  let train_length := (platform_length * pole_time) / (platform_time - pole_time)
  train_length = 300 := by
sorry

end train_length_l804_80406


namespace geometric_sequence_sum_l804_80498

/-- Given a geometric sequence with common ratio 2 and sum of first four terms equal to 1,
    the sum of the first eight terms is 17. -/
theorem geometric_sequence_sum (a : ℝ) : 
  (∃ (S₄ S₈ : ℝ), 
    S₄ = a * (1 + 2 + 2^2 + 2^3) ∧
    S₄ = 1 ∧
    S₈ = a * (1 + 2 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7)) →
  (∃ S₈ : ℝ, S₈ = a * (1 + 2 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7) ∧ S₈ = 17) :=
by sorry

end geometric_sequence_sum_l804_80498


namespace parabola_tangent_line_l804_80472

/-- A parabola is tangent to a line if they intersect at exactly one point. -/
def is_tangent (a : ℝ) : Prop :=
  ∃! x : ℝ, a * x^2 + 10 = 2 * x

/-- The value of a for which the parabola y = ax^2 + 10 is tangent to the line y = 2x -/
theorem parabola_tangent_line : 
  ∃ a : ℝ, is_tangent a ∧ a = (1 : ℝ) / 10 :=
sorry

end parabola_tangent_line_l804_80472


namespace smallest_angle_BFE_l804_80443

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the incenter of a triangle
def incenter (t : Triangle) : Point := sorry

-- Define the measure of an angle
def angle_measure (p q r : Point) : ℝ := sorry

-- State the theorem
theorem smallest_angle_BFE (ABC : Triangle) :
  let D := incenter ABC
  let ABD := Triangle.mk ABC.A ABC.B D
  let E := incenter ABD
  let BDE := Triangle.mk ABC.B D E
  let F := incenter BDE
  ∃ (n : ℕ), 
    (∀ m : ℕ, m < n → ¬(∃ ABC : Triangle, angle_measure ABC.B F E = m)) ∧
    (∃ ABC : Triangle, angle_measure ABC.B F E = n) ∧
    n = 113 :=
sorry

end smallest_angle_BFE_l804_80443


namespace B_power_101_l804_80475

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  !![1, 0, 0;
     0, 0, 1;
     0, 1, 0]

theorem B_power_101 : B^101 = B := by sorry

end B_power_101_l804_80475


namespace polynomial_factor_implies_coefficients_l804_80499

/-- The polynomial p(x) = ax^4 + bx^3 + 20x^2 - 12x + 10 -/
def p (a b : ℚ) (x : ℚ) : ℚ := a * x^4 + b * x^3 + 20 * x^2 - 12 * x + 10

/-- The factor q(x) = 2x^2 + 3x - 4 -/
def q (x : ℚ) : ℚ := 2 * x^2 + 3 * x - 4

/-- Theorem stating that if q(x) is a factor of p(x), then a = 2 and b = 27 -/
theorem polynomial_factor_implies_coefficients (a b : ℚ) :
  (∃ r : ℚ → ℚ, ∀ x, p a b x = q x * r x) → a = 2 ∧ b = 27 := by
  sorry

end polynomial_factor_implies_coefficients_l804_80499


namespace perpendicular_lines_a_values_l804_80469

/-- Given two lines l₁ and l₂, prove that if they are perpendicular, then a = 0 or a = 5/3 -/
theorem perpendicular_lines_a_values (a : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | a * x + 3 * y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | 2 * x + (a^2 - a) * y + 3 = 0}
  let perpendicular := ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁, y₁) ∈ l₁ → (x₂, y₂) ∈ l₂ → 
    (x₂ - x₁) * (a * (x₂ - x₁) + 3 * (y₂ - y₁)) + 
    (y₂ - y₁) * (2 * (x₂ - x₁) + (a^2 - a) * (y₂ - y₁)) = 0
  perpendicular → a = 0 ∨ a = 5/3 :=
by sorry


end perpendicular_lines_a_values_l804_80469


namespace gcd_property_l804_80428

theorem gcd_property (n : ℕ) :
  (∃ d : ℕ, d = Nat.gcd (7 * n + 5) (5 * n + 4) ∧ (d = 1 ∨ d = 3)) ∧
  (Nat.gcd (7 * n + 5) (5 * n + 4) = 3 ↔ ∃ k : ℕ, n = 3 * k + 1) :=
by sorry

end gcd_property_l804_80428


namespace simple_interest_principal_calculation_l804_80457

/-- Proves that given a simple interest of 4016.25, an interest rate of 10% per annum,
    and a time period of 5 years, the principal sum is 8032.5. -/
theorem simple_interest_principal_calculation :
  let simple_interest : ℝ := 4016.25
  let rate : ℝ := 10  -- 10% per annum
  let time : ℝ := 5   -- 5 years
  let principal : ℝ := simple_interest * 100 / (rate * time)
  principal = 8032.5 := by
  sorry

end simple_interest_principal_calculation_l804_80457


namespace multiply_negative_four_with_three_halves_l804_80494

theorem multiply_negative_four_with_three_halves : (-4 : ℚ) * (3/2) = -6 := by
  sorry

end multiply_negative_four_with_three_halves_l804_80494


namespace modulus_of_z_l804_80453

theorem modulus_of_z (z : ℂ) (h : z^2 = 48 - 14*I) : Complex.abs z = 5 * Real.sqrt 2 := by
  sorry

end modulus_of_z_l804_80453


namespace time_to_see_again_l804_80440

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a person walking -/
structure Walker where
  initialPosition : Point
  speed : ℝ

/-- The setup of the problem -/
def problemSetup : Prop := ∃ (sam kim : Walker) (tower : Point) (r : ℝ),
  -- Sam's initial position and speed
  sam.initialPosition = ⟨-100, -150⟩ ∧ sam.speed = 4 ∧
  -- Kim's initial position and speed
  kim.initialPosition = ⟨-100, 150⟩ ∧ kim.speed = 2 ∧
  -- Tower's position and radius
  tower = ⟨0, 0⟩ ∧ r = 100 ∧
  -- Initial distance between Sam and Kim
  (sam.initialPosition.x - kim.initialPosition.x)^2 + (sam.initialPosition.y - kim.initialPosition.y)^2 = 300^2

/-- The theorem to be proved -/
theorem time_to_see_again (setup : problemSetup) : 
  ∃ (t : ℝ), t = 240 ∧ 
  (∀ (t' : ℝ), t' < t → ∃ (x y : ℝ), 
    x^2 + y^2 = 100^2 ∧ 
    (y - (-150)) / (x - (-100 + 4 * t')) = (150 - (-150)) / (((-100 + 2 * t') - (-100 + 4 * t'))) ∧
    x * (150 - (-150)) = y * (((-100 + 2 * t') - (-100 + 4 * t'))))
  ∧ 
  (∃ (x y : ℝ), 
    x^2 + y^2 = 100^2 ∧ 
    (y - (-150)) / (x - (-100 + 4 * t)) = (150 - (-150)) / (((-100 + 2 * t) - (-100 + 4 * t))) ∧
    x * (150 - (-150)) = y * (((-100 + 2 * t) - (-100 + 4 * t)))) :=
by
  sorry


end time_to_see_again_l804_80440


namespace mixture_replacement_l804_80434

/-- Represents the mixture replacement problem -/
theorem mixture_replacement (initial_a initial_b replaced_amount : ℝ) : 
  initial_a = 64 →
  initial_b = initial_a / 4 →
  (initial_a - (4/5) * replaced_amount) / (initial_b - (1/5) * replaced_amount + replaced_amount) = 2/3 →
  replaced_amount = 40 :=
by
  sorry

#check mixture_replacement

end mixture_replacement_l804_80434


namespace sum_of_integers_l804_80435

theorem sum_of_integers (a b : ℕ+) (h1 : a - b = 4) (h2 : a * b = 63) : a + b = 18 := by
  sorry

end sum_of_integers_l804_80435


namespace jason_newspaper_earnings_l804_80460

-- Define the initial and final amounts for Jason
def jason_initial : ℕ := 3
def jason_final : ℕ := 63

-- Define Jason's earnings
def jason_earnings : ℕ := jason_final - jason_initial

-- Theorem to prove
theorem jason_newspaper_earnings :
  jason_earnings = 60 := by
  sorry

end jason_newspaper_earnings_l804_80460


namespace min_value_rational_function_l804_80444

theorem min_value_rational_function :
  (∀ x : ℝ, x > -1 → ((x^2 + 7*x + 10) / (x + 1)) ≥ 9) ∧
  (∃ x : ℝ, x > -1 ∧ ((x^2 + 7*x + 10) / (x + 1)) = 9) := by
  sorry

end min_value_rational_function_l804_80444


namespace min_bailing_rate_solution_bailing_rate_is_eight_gallons_per_minute_l804_80455

/-- Represents the problem of finding the minimum bailing rate -/
def MinBailingRateProblem (distance : Real) (rowingSpeed : Real) (waterIntakeRate : Real) (maxWaterCapacity : Real) : Prop :=
  ∃ (bailingRate : Real),
    bailingRate ≥ 0 ∧
    (distance / rowingSpeed) * 60 * (waterIntakeRate - bailingRate) ≤ maxWaterCapacity ∧
    ∀ (r : Real), r ≥ 0 ∧ (distance / rowingSpeed) * 60 * (waterIntakeRate - r) ≤ maxWaterCapacity → r ≥ bailingRate

/-- The solution to the minimum bailing rate problem -/
theorem min_bailing_rate_solution :
  MinBailingRateProblem 1 4 10 30 → (∃ (minRate : Real), minRate = 8) :=
by
  sorry

/-- Proof that 8 gallons per minute is the minimum bailing rate required -/
theorem bailing_rate_is_eight_gallons_per_minute :
  MinBailingRateProblem 1 4 10 30 :=
by
  sorry

end min_bailing_rate_solution_bailing_rate_is_eight_gallons_per_minute_l804_80455


namespace circle_center_range_l804_80433

theorem circle_center_range (a : ℝ) : 
  let C : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + (p.2 - (a-2))^2 = 9}
  let M : ℝ × ℝ := (0, 3)
  (3, -2) ∈ C ∧ (0, -5) ∈ C ∧ 
  (∃ N ∈ C, (N.1 - M.1)^2 + (N.2 - M.2)^2 = 4 * ((N.1 - a)^2 + (N.2 - (a-2))^2)) →
  (-3 ≤ a ∧ a ≤ 0) ∨ (1 ≤ a ∧ a ≤ 4) :=
sorry

end circle_center_range_l804_80433


namespace points_on_line_relationship_l804_80478

/-- Given a line y = -3x + b and three points on this line, prove that y₁ > y₂ > y₃ -/
theorem points_on_line_relationship (b : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = -3 * (-2) + b)
  (h2 : y₂ = -3 * (-1) + b)
  (h3 : y₃ = -3 * 1 + b) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end points_on_line_relationship_l804_80478


namespace total_votes_l804_80485

/-- Proves that the total number of votes is 290 given the specified conditions -/
theorem total_votes (votes_against : ℕ) (votes_in_favor : ℕ) (total_votes : ℕ) : 
  votes_in_favor = votes_against + 58 →
  votes_against = (40 * total_votes) / 100 →
  total_votes = votes_in_favor + votes_against →
  total_votes = 290 := by
sorry

end total_votes_l804_80485


namespace largest_non_representable_largest_non_representable_proof_l804_80441

def is_representable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 5 * a + 8 * b + 12 * c

theorem largest_non_representable : ℕ :=
  19

theorem largest_non_representable_proof :
  (¬ is_representable largest_non_representable) ∧
  (∀ m : ℕ, m > largest_non_representable → is_representable m) :=
sorry

end largest_non_representable_largest_non_representable_proof_l804_80441


namespace only_C_in_position_I_l804_80490

-- Define a structure for a rectangle with labeled sides
structure LabeledRectangle where
  top : ℕ
  bottom : ℕ
  left : ℕ
  right : ℕ

-- Define the five rectangles
def rectangle_A : LabeledRectangle := ⟨1, 9, 4, 6⟩
def rectangle_B : LabeledRectangle := ⟨0, 6, 1, 3⟩
def rectangle_C : LabeledRectangle := ⟨8, 2, 3, 5⟩
def rectangle_D : LabeledRectangle := ⟨5, 8, 7, 4⟩
def rectangle_E : LabeledRectangle := ⟨2, 0, 9, 7⟩

-- Define a function to check if a rectangle can be placed in position I
def can_be_placed_in_position_I (r : LabeledRectangle) : Prop :=
  ∃ (r2 r4 : LabeledRectangle), 
    r.right = r2.left ∧ r.bottom = r4.top

-- Theorem stating that only rectangle C can be placed in position I
theorem only_C_in_position_I : 
  can_be_placed_in_position_I rectangle_C ∧
  ¬can_be_placed_in_position_I rectangle_A ∧
  ¬can_be_placed_in_position_I rectangle_B ∧
  ¬can_be_placed_in_position_I rectangle_D ∧
  ¬can_be_placed_in_position_I rectangle_E :=
sorry

end only_C_in_position_I_l804_80490


namespace f_properties_l804_80407

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -4 * x^2 else x^2 - x

theorem f_properties :
  (∃ a : ℝ, f a = -1/4 ∧ (a = -1/4 ∨ a = 1/2)) ∧
  (∃ b : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f x - b = 0 ∧ f y - b = 0 ∧ f z - b = 0) →
    -1/4 < b ∧ b < 0) :=
by sorry

end f_properties_l804_80407


namespace probability_two_forks_one_spoon_one_knife_l804_80417

/-- The number of forks in the drawer -/
def num_forks : ℕ := 8

/-- The number of spoons in the drawer -/
def num_spoons : ℕ := 5

/-- The number of knives in the drawer -/
def num_knives : ℕ := 7

/-- The total number of pieces of silverware -/
def total_silverware : ℕ := num_forks + num_spoons + num_knives

/-- The number of pieces to be drawn -/
def num_drawn : ℕ := 4

/-- The probability of drawing 2 forks, 1 spoon, and 1 knife -/
theorem probability_two_forks_one_spoon_one_knife :
  (Nat.choose num_forks 2 * Nat.choose num_spoons 1 * Nat.choose num_knives 1 : ℚ) /
  (Nat.choose total_silverware num_drawn : ℚ) = 196 / 969 := by
  sorry

end probability_two_forks_one_spoon_one_knife_l804_80417


namespace point_B_coordinates_l804_80486

/-- Given point A (2, 4), vector a⃗ = (3, 4), and AB⃗ = 2a⃗, prove that the coordinates of point B are (8, 12). -/
theorem point_B_coordinates (A B : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (2, 4) →
  a = (3, 4) →
  B.1 - A.1 = 2 * a.1 →
  B.2 - A.2 = 2 * a.2 →
  B = (8, 12) := by
sorry

end point_B_coordinates_l804_80486


namespace f_neg_two_eq_twelve_l804_80452

/-- The polynomial function f(x) = x^5 + 4x^4 + x^2 + 20x + 16 -/
def f (x : ℝ) : ℝ := x^5 + 4*x^4 + x^2 + 20*x + 16

/-- Theorem: The value of f(-2) is 12 -/
theorem f_neg_two_eq_twelve : f (-2) = 12 := by
  sorry

end f_neg_two_eq_twelve_l804_80452


namespace not_all_projections_same_l804_80429

/-- Represents a 3D shape -/
inductive Shape
  | Cube
  | Sphere
  | Cone

/-- Represents a type of orthographic projection -/
inductive Projection
  | FrontView
  | SideView
  | TopView

/-- Represents the result of an orthographic projection -/
inductive ProjectionResult
  | Square
  | Circle
  | IsoscelesTriangle

/-- Returns the projection result for a given shape and projection type -/
def projectShape (s : Shape) (p : Projection) : ProjectionResult :=
  match s, p with
  | Shape.Cube, _ => ProjectionResult.Square
  | Shape.Sphere, _ => ProjectionResult.Circle
  | Shape.Cone, Projection.TopView => ProjectionResult.Circle
  | Shape.Cone, _ => ProjectionResult.IsoscelesTriangle

/-- Theorem stating that it's not true that all projections are the same for all shapes -/
theorem not_all_projections_same : ¬ (∀ (s1 s2 : Shape) (p1 p2 : Projection), 
  projectShape s1 p1 = projectShape s2 p2) := by
  sorry


end not_all_projections_same_l804_80429


namespace successive_discounts_equivalence_l804_80491

theorem successive_discounts_equivalence :
  let original_price : ℝ := 50
  let first_discount : ℝ := 0.30
  let second_discount : ℝ := 0.15
  let equivalent_discount : ℝ := 0.405

  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)

  final_price = original_price * (1 - equivalent_discount) :=
by sorry

end successive_discounts_equivalence_l804_80491


namespace modulo_residue_sum_of_cubes_l804_80401

theorem modulo_residue_sum_of_cubes (m : ℕ) (h : m = 17) :
  (512^3 + (6*104)^3 + (8*289)^3 + (5*68)^3) % m = 9 := by
  sorry

end modulo_residue_sum_of_cubes_l804_80401


namespace functional_equation_solution_l804_80492

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := ℝ → ℝ

/-- The functional equation that f must satisfy -/
def SatisfiesEquation (f : PositiveRealFunction) (α : ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (f x + y) = α * x + 1 / f (1 / y)

theorem functional_equation_solution :
  ∀ α : ℝ, α ≠ 0 →
    (∃ f : PositiveRealFunction, SatisfiesEquation f α) ↔
    (α = 1 ∧ ∃ f : PositiveRealFunction, SatisfiesEquation f 1 ∧ ∀ x, x > 0 → f x = x) :=
by sorry

end functional_equation_solution_l804_80492


namespace sum_of_xy_l804_80496

theorem sum_of_xy (x y : ℕ+) (h : x + y + x * y = 54) : x + y = 14 := by
  sorry

end sum_of_xy_l804_80496


namespace bisection_termination_condition_l804_80404

/-- Bisection method termination condition -/
theorem bisection_termination_condition 
  (f : ℝ → ℝ) (x₁ x₂ e : ℝ) (h_e : e > 0) :
  |x₁ - x₂| < e → ∃ x, x ∈ [x₁, x₂] ∧ |f x| < e :=
sorry

end bisection_termination_condition_l804_80404


namespace sine_cosine_cube_difference_l804_80424

theorem sine_cosine_cube_difference (α : ℝ) (n : ℝ) 
  (h : Real.sin α - Real.cos α = n) : 
  Real.sin α ^ 3 - Real.cos α ^ 3 = (3 * n - n^3) / 2 := by
  sorry

end sine_cosine_cube_difference_l804_80424


namespace problem_solution_l804_80420

theorem problem_solution (x : ℝ) (h : 3 * x^2 - x = 1) :
  6 * x^3 + 7 * x^2 - 5 * x + 2010 = 2013 := by
  sorry

end problem_solution_l804_80420


namespace vlad_sister_height_difference_l804_80488

/-- The height difference between two people given their heights in centimeters -/
def height_difference (height1 : ℝ) (height2 : ℝ) : ℝ :=
  height1 - height2

/-- Vlad's height in centimeters -/
def vlad_height : ℝ := 190.5

/-- Vlad's sister's height in centimeters -/
def sister_height : ℝ := 86.36

/-- Theorem: The height difference between Vlad and his sister is 104.14 centimeters -/
theorem vlad_sister_height_difference :
  height_difference vlad_height sister_height = 104.14 := by
  sorry


end vlad_sister_height_difference_l804_80488


namespace money_left_l804_80431

def salary_distribution (S : ℝ) : Prop :=
  let house_rent := (2/5) * S
  let food := (3/10) * S
  let conveyance := (1/8) * S
  let food_and_conveyance := food + conveyance
  food_and_conveyance = 3399.999999999999

theorem money_left (S : ℝ) (h : salary_distribution S) : 
  S - ((2/5 + 3/10 + 1/8) * S) = 1400 := by
  sorry

end money_left_l804_80431


namespace complex_modulus_l804_80445

theorem complex_modulus (x y : ℝ) (h : (1 + Complex.I) * x = 1 - Complex.I * y) :
  Complex.abs (x - Complex.I * y) = Real.sqrt 2 := by
  sorry

end complex_modulus_l804_80445


namespace sock_pair_difference_sock_conditions_l804_80412

/-- Represents the number of socks of a specific color -/
structure SockCount where
  red : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- Represents the sock collection of Joseph -/
def josephSocks : SockCount where
  red := 6
  blue := 12
  white := 8
  black := 2

theorem sock_pair_difference : 
  let blue_pairs := josephSocks.blue / 2
  let black_pairs := josephSocks.black / 2
  blue_pairs - black_pairs = 5 := by
  sorry

theorem sock_conditions : 
  -- Joseph has more pairs of blue socks than black socks
  josephSocks.blue > josephSocks.black ∧
  -- He has one less pair of red socks than white socks
  josephSocks.red / 2 + 1 = josephSocks.white / 2 ∧
  -- He has twice as many blue socks as red socks
  josephSocks.blue = 2 * josephSocks.red ∧
  -- He has 28 socks in total
  josephSocks.red + josephSocks.blue + josephSocks.white + josephSocks.black = 28 := by
  sorry

end sock_pair_difference_sock_conditions_l804_80412


namespace modulus_of_z_l804_80436

theorem modulus_of_z (z : ℂ) (h : (z + Complex.I) * Complex.I = -3 + 4 * Complex.I) : 
  Complex.abs z = 2 * Real.sqrt 5 := by
  sorry

end modulus_of_z_l804_80436


namespace quadratic_two_distinct_roots_exist_quadratic_with_two_roots_l804_80413

/-- A quadratic equation x^2 + bx + c = 0 has two distinct real roots if and only if its discriminant is positive -/
theorem quadratic_two_distinct_roots (b c : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + b*x₁ + c = 0 ∧ x₂^2 + b*x₂ + c = 0 ↔ b^2 - 4*c > 0 := by
  sorry

/-- There exist real values b and c such that the quadratic equation x^2 + bx + c = 0 has two distinct real roots -/
theorem exist_quadratic_with_two_roots : ∃ (b c : ℝ), ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + b*x₁ + c = 0 ∧ x₂^2 + b*x₂ + c = 0 := by
  sorry

end quadratic_two_distinct_roots_exist_quadratic_with_two_roots_l804_80413


namespace largest_even_digit_multiple_of_9_under_1000_l804_80426

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_even_digit_multiple_of_9_under_1000 :
  ∃ (n : ℕ), n = 888 ∧
    has_only_even_digits n ∧
    n < 1000 ∧
    n % 9 = 0 ∧
    ∀ m : ℕ, has_only_even_digits m ∧ m < 1000 ∧ m % 9 = 0 → m ≤ n :=
by sorry

end largest_even_digit_multiple_of_9_under_1000_l804_80426


namespace min_value_of_function_l804_80419

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (x^2 + x + 25) / x ≥ 11 ∧ ∃ y > 0, (y^2 + y + 25) / y = 11 := by
  sorry

end min_value_of_function_l804_80419


namespace problem_one_problem_two_l804_80462

-- Part 1
theorem problem_one : (2 * Real.sqrt 3 - 1)^2 + (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = 12 - 4 * Real.sqrt 3 := by
  sorry

-- Part 2
theorem problem_two : (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - 6 * Real.sqrt (1/2) = -6 * Real.sqrt 5 := by
  sorry

end problem_one_problem_two_l804_80462


namespace card_drawing_probability_ratio_l804_80418

theorem card_drawing_probability_ratio :
  let total_cards : ℕ := 60
  let num_range : ℕ := 15
  let cards_per_num : ℕ := 4
  let draw_count : ℕ := 4

  let p : ℚ := (num_range : ℚ) / (Nat.choose total_cards draw_count)
  let q : ℚ := (num_range * (num_range - 1) * Nat.choose cards_per_num 3 * Nat.choose cards_per_num 1 : ℚ) / 
                (Nat.choose total_cards draw_count)

  q / p = 224 := by sorry

end card_drawing_probability_ratio_l804_80418


namespace total_cost_is_840_l804_80422

/-- The cost of a single movie ticket in dollars -/
def movie_ticket_cost : ℚ := 30

/-- The number of movie tickets -/
def num_movie_tickets : ℕ := 8

/-- The number of football game tickets -/
def num_football_tickets : ℕ := 5

/-- The ratio of the cost of 8 movie tickets to 1 football game ticket -/
def cost_ratio : ℚ := 2

/-- The total cost of buying movie tickets and football game tickets -/
def total_cost : ℚ :=
  (num_movie_tickets : ℚ) * movie_ticket_cost +
  (num_football_tickets : ℚ) * ((num_movie_tickets : ℚ) * movie_ticket_cost / cost_ratio)

theorem total_cost_is_840 : total_cost = 840 := by
  sorry

end total_cost_is_840_l804_80422


namespace quadratic_inequality_solution_set_l804_80495

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo (-3) 2 = {x : ℝ | a * x^2 - 5*x + b > 0}) : 
  {x : ℝ | b * x^2 - 5*x + a > 0} = Set.Iic (-1/3) ∪ Set.Ici (1/2) := by
  sorry

end quadratic_inequality_solution_set_l804_80495


namespace sues_necklace_beads_l804_80403

theorem sues_necklace_beads (purple : ℕ) (blue : ℕ) (green : ℕ) 
  (h1 : purple = 7)
  (h2 : blue = 2 * purple)
  (h3 : green = blue + 11) :
  purple + blue + green = 46 := by
  sorry

end sues_necklace_beads_l804_80403


namespace unique_fibonacci_partition_l804_80456

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def is_fibonacci (n : ℕ) : Prop := ∃ k, fibonacci k = n

def is_partition (A B : Set ℕ) : Prop :=
  (A ∩ B = ∅) ∧ (A ∪ B = Set.univ)

def is_prohibited (S A B : Set ℕ) : Prop :=
  ∀ k l s, (k ∈ A ∧ l ∈ A ∧ s ∈ S) ∨ (k ∈ B ∧ l ∈ B ∧ s ∈ S) → k + l ≠ s

theorem unique_fibonacci_partition :
  ∃! (A B : Set ℕ), is_partition A B ∧ is_prohibited {n | is_fibonacci n} A B :=
sorry

end unique_fibonacci_partition_l804_80456


namespace modulus_of_3_minus_4i_l804_80447

theorem modulus_of_3_minus_4i : Complex.abs (3 - 4*I) = 5 := by sorry

end modulus_of_3_minus_4i_l804_80447


namespace quadratic_inequality_product_l804_80464

/-- Given a quadratic inequality x^2 + bx + c < 0 with solution set {x | 2 < x < 4}, 
    prove that bc = -48 -/
theorem quadratic_inequality_product (b c : ℝ) 
  (h : ∀ x, x^2 + b*x + c < 0 ↔ 2 < x ∧ x < 4) : b*c = -48 := by
  sorry

end quadratic_inequality_product_l804_80464


namespace fraction_meaningful_l804_80414

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x + 1)) ↔ x ≠ -1 :=
sorry

end fraction_meaningful_l804_80414


namespace divisibility_of_P_and_Q_l804_80468

/-- Given that there exists a natural number n such that 1997 divides 111...1 (n ones),
    prove that 1997 divides both P and Q. -/
theorem divisibility_of_P_and_Q (n : ℕ) (h : ∃ k : ℕ, (10^n - 1) / 9 = 1997 * k) :
  ∃ (p q : ℕ), P = 1997 * p ∧ Q = 1997 * q :=
sorry

end divisibility_of_P_and_Q_l804_80468


namespace characterize_g_l804_80458

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the properties of g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 6 * x + 1

-- State the theorem
theorem characterize_g :
  ∀ g : ℝ → ℝ, is_valid_g g ↔ (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
sorry

end characterize_g_l804_80458


namespace negation_of_universal_proposition_l804_80465

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) := by
  sorry

end negation_of_universal_proposition_l804_80465


namespace line_passes_through_fixed_point_l804_80481

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (3 + m) * (-3) + 4 * 3 - 3 + 3 * m = 0 :=
by sorry

end line_passes_through_fixed_point_l804_80481


namespace soccer_league_games_l804_80483

/-- The number of games played in a soccer league where each team plays every other team once -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a soccer league with 10 teams, where each team plays every other team once, 
    the total number of games played is 45 -/
theorem soccer_league_games : num_games 10 = 45 := by
  sorry

end soccer_league_games_l804_80483


namespace paper_strips_length_l804_80421

/-- The total length of overlapping paper strips -/
def total_length (n : ℕ) (sheet_length : ℝ) (overlap : ℝ) : ℝ :=
  sheet_length + (n - 1) * (sheet_length - overlap)

/-- Theorem: The total length of 30 sheets of 25 cm paper strips overlapped by 6 cm is 576 cm -/
theorem paper_strips_length :
  total_length 30 25 6 = 576 := by
  sorry

end paper_strips_length_l804_80421


namespace round_trip_speed_l804_80405

/-- Proves that given a person's average speed for a round trip is 75 km/hr,
    and the return speed is 50% faster than the initial speed,
    the initial speed is 62.5 km/hr. -/
theorem round_trip_speed (v : ℝ) : 
  (v > 0) →                           -- Initial speed is positive
  (2 / (1 / v + 1 / (1.5 * v)) = 75)  -- Average speed is 75 km/hr
  → v = 62.5 := by sorry

end round_trip_speed_l804_80405


namespace complex_coordinates_of_product_l804_80425

theorem complex_coordinates_of_product : 
  let z : ℂ := (2 - Complex.I) * (1 + Complex.I)
  Complex.re z = 3 ∧ Complex.im z = 1 := by sorry

end complex_coordinates_of_product_l804_80425


namespace egg_purchase_cost_l804_80437

def dozen : ℕ := 12
def egg_price : ℚ := 0.50

theorem egg_purchase_cost (num_dozens : ℕ) : 
  (num_dozens * dozen * egg_price : ℚ) = 18 :=
by
  sorry

end egg_purchase_cost_l804_80437


namespace polynomial_expansion_l804_80439

theorem polynomial_expansion (x : ℝ) : 
  (5 * x + 3) * (7 * x^2 + 2 * x + 4) = 35 * x^3 + 31 * x^2 + 26 * x + 12 := by
  sorry

end polynomial_expansion_l804_80439


namespace danny_chemistry_marks_l804_80438

theorem danny_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (h1 : english = 76) 
  (h2 : mathematics = 65) 
  (h3 : physics = 82) 
  (h4 : biology = 75) 
  (h5 : average = 73) 
  (h6 : (english + mathematics + physics + biology + chemistry) / 5 = average) :
  chemistry = 67 :=
by
  sorry

end danny_chemistry_marks_l804_80438


namespace picture_placement_l804_80408

/-- Given a wall of width 30 feet, with two pictures each 4 feet wide and spaced 1 foot apart
    hung in the center, the distance from the end of the wall to the nearest edge of the first
    picture is 10.5 feet. -/
theorem picture_placement (wall_width : ℝ) (picture_width : ℝ) (picture_space : ℝ)
  (h_wall : wall_width = 30)
  (h_picture : picture_width = 4)
  (h_space : picture_space = 1) :
  let total_picture_space := 2 * picture_width + picture_space
  (wall_width - total_picture_space) / 2 = 10.5 := by
  sorry

end picture_placement_l804_80408


namespace cashew_price_l804_80427

/-- Proves the price of cashew nuts given the conditions of the problem -/
theorem cashew_price (peanut_price : ℕ) (cashew_amount peanut_amount total_amount : ℕ) (total_price : ℕ) :
  peanut_price = 130 →
  cashew_amount = 3 →
  peanut_amount = 2 →
  total_amount = 5 →
  total_price = 178 →
  cashew_amount * (total_price * total_amount - peanut_price * peanut_amount) / cashew_amount = 210 :=
by
  sorry

end cashew_price_l804_80427


namespace solution_sets_intersection_and_union_l804_80476

def equation1 (p : ℝ) (x : ℝ) : Prop := x^2 - p*x + 6 = 0

def equation2 (q : ℝ) (x : ℝ) : Prop := x^2 + 6*x - q = 0

def solution_set (equation : ℝ → Prop) : Set ℝ :=
  {x | equation x}

theorem solution_sets_intersection_and_union
  (p q : ℝ)
  (M : Set ℝ)
  (N : Set ℝ)
  (h1 : M = solution_set (equation1 p))
  (h2 : N = solution_set (equation2 q))
  (h3 : M ∩ N = {2}) :
  p = 5 ∧ q = 16 ∧ M ∪ N = {2, 3, -8} := by
  sorry

end solution_sets_intersection_and_union_l804_80476


namespace work_completion_time_l804_80459

theorem work_completion_time (x : ℝ) : 
  x > 0 →  -- p's completion time is positive
  (2 / x + 3 * (1 / x + 1 / 6) = 1) →  -- work equation
  x = 10 :=
by sorry

end work_completion_time_l804_80459


namespace sqrt_seven_inequality_l804_80449

theorem sqrt_seven_inequality (m n : ℕ+) (h : (m : ℝ) / n < Real.sqrt 7) :
  7 - (m : ℝ)^2 / n^2 ≥ 3 / n^2 := by
  sorry

end sqrt_seven_inequality_l804_80449


namespace inequality_solution_l804_80477

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, x^2 - (a + 3) * x + 2 * (a + 1) ≥ 0 ↔ 
    (a ≥ 1 ∧ (x ≥ a + 1 ∨ x ≤ 2)) ∨ 
    (a < 1 ∧ (x ≥ 2 ∨ x ≤ a + 1))) := by
  sorry

end inequality_solution_l804_80477


namespace toys_sold_is_eighteen_l804_80415

/-- Given a selling price, gain equal to the cost of 3 toys, and the cost of one toy,
    calculate the number of toys sold. -/
def number_of_toys_sold (selling_price gain cost_per_toy : ℕ) : ℕ :=
  (selling_price - gain) / cost_per_toy

/-- Theorem stating that given the conditions in the problem, 
    the number of toys sold is 18. -/
theorem toys_sold_is_eighteen :
  let selling_price := 21000
  let gain := 3 * 1000
  let cost_per_toy := 1000
  number_of_toys_sold selling_price gain cost_per_toy = 18 := by
  sorry

#eval number_of_toys_sold 21000 (3 * 1000) 1000

end toys_sold_is_eighteen_l804_80415


namespace triangle_equality_l804_80474

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  h : A + B + C = 180  -- Sum of angles in a triangle is 180°

-- Define the theorem
theorem triangle_equality (t : Triangle) 
  (h₁ : t.A > t.B)  -- A > B
  (h₂ : ∃ (C₁ C₂ : ℝ), C₁ + C₂ = t.C ∧ C₁ = 2 * C₂)  -- C₁ + C₂ = C and C₁ = 2C₂
  : t.A = t.B := by
  sorry

end triangle_equality_l804_80474


namespace regression_equation_proof_l804_80400

/-- Given an exponential model and a regression line equation, 
    prove the resulting regression equation. -/
theorem regression_equation_proof 
  (y : ℝ → ℝ) 
  (k a : ℝ) 
  (h1 : ∀ x, y x = Real.exp (k * x + a)) 
  (h2 : ∀ x, 0.25 * x - 2.58 = Real.log (y x)) : 
  ∀ x, y x = Real.exp (0.25 * x - 2.58) := by
sorry

end regression_equation_proof_l804_80400


namespace even_increasing_function_property_l804_80487

/-- A function that is even on ℝ and increasing on (-∞, 0] -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

/-- Theorem stating that for an even function increasing on (-∞, 0],
    if f(a) ≤ f(2-a), then a ≥ 1 -/
theorem even_increasing_function_property (f : ℝ → ℝ) (a : ℝ) 
    (h1 : EvenIncreasingFunction f) (h2 : f a ≤ f (2 - a)) : 
    a ≥ 1 := by
  sorry

end even_increasing_function_property_l804_80487


namespace real_roots_sum_product_l804_80482

theorem real_roots_sum_product (c d : ℝ) : 
  (c^4 - 6*c + 3 = 0) → 
  (d^4 - 6*d + 3 = 0) → 
  (c ≠ d) →
  (c*d + c + d = Real.sqrt 3) := by
sorry

end real_roots_sum_product_l804_80482


namespace fraction_subtraction_l804_80411

theorem fraction_subtraction : 
  (3 + 7 + 11) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 7 + 11) = 33 / 28 := by
  sorry

end fraction_subtraction_l804_80411


namespace circle_tangent_to_directrix_l804_80442

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the right focus of the hyperbola
def right_focus : ℝ × ℝ := (2, 0)

-- Define the right directrix of the hyperbola
def right_directrix (x : ℝ) : Prop := x = 1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Theorem statement
theorem circle_tangent_to_directrix :
  ∀ x y : ℝ,
  hyperbola x y →
  circle_equation x y ↔
    (∃ (cx cy : ℝ), (cx, cy) = right_focus ∧
      ∀ (dx : ℝ), right_directrix dx →
        (x - cx)^2 + (y - cy)^2 = (x - dx)^2) :=
by sorry

end circle_tangent_to_directrix_l804_80442


namespace intersection_of_A_and_B_l804_80450

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l804_80450


namespace distinct_collections_count_l804_80479

/-- Represents the count of each letter in "MATHEMATICAL" --/
structure LetterCount where
  a : Nat
  e : Nat
  i : Nat
  t : Nat
  m : Nat
  h : Nat
  l : Nat
  c : Nat

/-- The initial count of letters in "MATHEMATICAL" --/
def initialCount : LetterCount := {
  a := 3, e := 1, i := 1,
  t := 2, m := 2, h := 1, l := 1, c := 2
}

/-- A collection of letters that fell off --/
structure FallenLetters where
  vowels : Finset Char
  consonants : Finset Char

/-- Checks if a collection of fallen letters is valid --/
def isValidCollection (letters : FallenLetters) : Prop :=
  letters.vowels.card = 3 ∧ letters.consonants.card = 3

/-- Counts distinct collections considering indistinguishable letters --/
def countDistinctCollections (count : LetterCount) : Nat :=
  sorry

theorem distinct_collections_count :
  countDistinctCollections initialCount = 80 :=
sorry

end distinct_collections_count_l804_80479


namespace negative_expressions_l804_80489

/-- Represents a number with an approximate value -/
structure ApproxNumber where
  value : ℝ

/-- Given approximate values for P, Q, R, S, and T -/
def P : ApproxNumber := ⟨-4.2⟩
def Q : ApproxNumber := ⟨-2.3⟩
def R : ApproxNumber := ⟨0⟩
def S : ApproxNumber := ⟨1.1⟩
def T : ApproxNumber := ⟨2.7⟩

/-- Helper function to extract the value from ApproxNumber -/
def getValue (x : ApproxNumber) : ℝ := x.value

/-- Theorem stating which expressions are negative -/
theorem negative_expressions :
  (getValue P - getValue Q < 0) ∧
  (getValue P + getValue T < 0) ∧
  (getValue P * getValue Q ≥ 0) ∧
  ((getValue S / getValue Q) * getValue P ≥ 0) ∧
  (getValue R / (getValue P * getValue Q) ≥ 0) ∧
  ((getValue S + getValue T) / getValue R ≥ 0) :=
sorry

end negative_expressions_l804_80489


namespace string_cutting_l804_80402

/-- Proves that cutting off 1/4 of a 2/3 meter long string leaves 50 cm remaining. -/
theorem string_cutting (string_length : ℚ) (h1 : string_length = 2/3) :
  (string_length * 100 - (1/4 * string_length * 100)) = 50 := by
  sorry

end string_cutting_l804_80402


namespace equal_segments_exist_l804_80430

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) :=
  (vertices : Fin (2*n) → ℝ × ℝ)
  (is_regular : sorry)

/-- A pairing of vertices in a regular polygon -/
def VertexPairing (n : ℕ) := Fin n → Fin (2*n) × Fin (2*n)

/-- The distance between two vertices in a regular polygon -/
def distance (p : RegularPolygon n) (i j : Fin (2*n)) : ℝ := sorry

theorem equal_segments_exist (m : ℕ) (n : ℕ) (h : n = 4*m + 2 ∨ n = 4*m + 3) 
  (p : RegularPolygon n) (pairing : VertexPairing n) : 
  ∃ (i j k l : Fin n), i ≠ j ∧ k ≠ l ∧ i ≠ k ∧ j ≠ l ∧
    distance p (pairing i).1 (pairing i).2 = distance p (pairing k).1 (pairing k).2 :=
sorry

end equal_segments_exist_l804_80430


namespace cambridge_population_l804_80451

theorem cambridge_population : ∃ (p : ℕ), p > 0 ∧ (
  ∀ (w a : ℚ),
  w > 0 ∧ a > 0 ∧
  w + a = 12 * p ∧
  w / 6 + a / 8 = 12 →
  p = 7
) := by
  sorry

end cambridge_population_l804_80451


namespace partnership_investment_l804_80448

/-- A partnership problem where three partners invest different amounts and receive different shares. -/
theorem partnership_investment (a_investment b_investment c_investment : ℝ)
  (b_share a_share : ℝ) :
  b_investment = 11000 →
  c_investment = 18000 →
  b_share = 2200 →
  a_share = 1400 →
  (b_share / b_investment = a_share / a_investment) →
  a_investment = 7000 := by
  sorry

end partnership_investment_l804_80448


namespace quadratic_solution_set_theorem_l804_80484

/-- Given a quadratic function f(x) = ax² + bx + c, 
    this is the type of its solution set when f(x) > 0 -/
def QuadraticSolutionSet (a b c : ℝ) := Set ℝ

/-- The condition that the solution set of ax² + bx + c > 0 
    is the open interval (3, 6) -/
def SolutionSetCondition (a b c : ℝ) : Prop :=
  QuadraticSolutionSet a b c = {x : ℝ | 3 < x ∧ x < 6}

theorem quadratic_solution_set_theorem 
  (a b c : ℝ) (h : SolutionSetCondition a b c) :
  QuadraticSolutionSet c b a = {x : ℝ | x < 1/6 ∨ x > 1/3} := by
  sorry

end quadratic_solution_set_theorem_l804_80484


namespace watchman_max_demand_l804_80461

/-- The amount of the bet made by the trespasser with his friends -/
def bet_amount : ℕ := 100

/-- The trespasser's net loss if he pays the watchman -/
def net_loss_if_pay (amount : ℕ) : ℤ := amount - bet_amount

/-- The trespasser's net loss if he doesn't pay the watchman -/
def net_loss_if_not_pay : ℕ := bet_amount

/-- Predicate to determine if the trespasser will pay for a given amount -/
def will_pay (amount : ℕ) : Prop :=
  net_loss_if_pay amount < net_loss_if_not_pay

/-- The maximum amount the watchman can demand -/
def max_demand : ℕ := 199

theorem watchman_max_demand :
  (∀ n : ℕ, n ≤ max_demand → will_pay n) ∧
  (∀ n : ℕ, n > max_demand → ¬will_pay n) :=
sorry

end watchman_max_demand_l804_80461


namespace sum_cube_inequality_l804_80432

theorem sum_cube_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end sum_cube_inequality_l804_80432


namespace min_value_of_function_l804_80471

theorem min_value_of_function (x : ℝ) : |3 - x| + |x - 7| ≥ 4 := by
  sorry

end min_value_of_function_l804_80471


namespace reciprocal_of_fraction_difference_l804_80409

theorem reciprocal_of_fraction_difference : (((2 : ℚ) / 3 - (3 : ℚ) / 4)⁻¹ : ℚ) = -12 := by
  sorry

end reciprocal_of_fraction_difference_l804_80409


namespace journey_fraction_by_foot_l804_80467

/-- Given a journey with specific conditions, proves the fraction traveled by foot -/
theorem journey_fraction_by_foot :
  let total_distance : ℝ := 30.000000000000007
  let bus_fraction : ℝ := 3/5
  let car_distance : ℝ := 2
  let foot_distance : ℝ := total_distance - bus_fraction * total_distance - car_distance
  foot_distance / total_distance = 1/3 := by
sorry

end journey_fraction_by_foot_l804_80467


namespace fraction_problem_l804_80423

/-- The fraction of p's amount that q and r each have -/
def fraction_of_p (p q r : ℚ) : ℚ :=
  q / p

/-- The problem statement -/
theorem fraction_problem (p q r : ℚ) : 
  p = 56 → 
  p = 2 * (fraction_of_p p q r) * p + 42 → 
  q = r → 
  fraction_of_p p q r = 1/8 := by
sorry

end fraction_problem_l804_80423


namespace shaded_area_calculation_l804_80416

/-- Calculates the area of the shaded region in a grid with two unshaded triangles. -/
theorem shaded_area_calculation (grid_width grid_height : ℝ)
  (large_triangle_base large_triangle_height : ℝ)
  (small_triangle_base small_triangle_height : ℝ)
  (h1 : grid_width = 15)
  (h2 : grid_height = 5)
  (h3 : large_triangle_base = grid_width)
  (h4 : large_triangle_height = grid_height)
  (h5 : small_triangle_base = 3)
  (h6 : small_triangle_height = 2) :
  grid_width * grid_height - (1/2 * large_triangle_base * large_triangle_height) -
  (1/2 * small_triangle_base * small_triangle_height) = 34.5 := by
  sorry

end shaded_area_calculation_l804_80416


namespace monthly_payment_difference_l804_80466

/-- The cost of the house in dollars -/
def house_cost : ℕ := 480000

/-- The cost of the trailer in dollars -/
def trailer_cost : ℕ := 120000

/-- The loan term in years -/
def loan_term : ℕ := 20

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- Calculates the monthly payment for a given cost over the loan term -/
def monthly_payment (cost : ℕ) : ℚ :=
  cost / (loan_term * months_per_year)

/-- The statement to be proved -/
theorem monthly_payment_difference :
  monthly_payment house_cost - monthly_payment trailer_cost = 1500 := by
  sorry

end monthly_payment_difference_l804_80466


namespace square_side_length_l804_80463

theorem square_side_length (s : ℝ) : s > 0 → s^2 = 3 * (4 * s) → s = 12 := by
  sorry

end square_side_length_l804_80463


namespace y_range_l804_80493

theorem y_range (y : ℝ) (h1 : y > 0) (h2 : Real.log y / Real.log 3 ≤ 3 - Real.log (9 * y) / Real.log 3) : 
  0 < y ∧ y ≤ Real.sqrt 3 := by
  sorry

end y_range_l804_80493


namespace line_equation_and_sum_l804_80470

/-- Given a line with slope 5 passing through (-2, 4), prove its equation and m + b value -/
theorem line_equation_and_sum (m b : ℝ) : 
  m = 5 → -- Given slope
  4 = m * (-2) + b → -- Point (-2, 4) lies on the line
  (∀ x y, y = m * x + b ↔ y - 4 = m * (x + 2)) → -- Line equation
  m + b = 19 := by
sorry

end line_equation_and_sum_l804_80470


namespace geometric_sequence_problem_l804_80410

/-- Geometric sequence with first term a and common ratio r -/
def geometric_sequence (a r : ℝ) : ℕ → ℝ := fun n => a * r^(n - 1)

/-- Sequence b_n as defined in the problem -/
def b_sequence (a : ℕ → ℝ) : ℕ → ℝ := 
  fun n => (Finset.range n).sum (fun k => (n - k) * a (k + 1))

/-- Sum of first n terms of a sequence -/
def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := (Finset.range n).sum (fun k => a (k + 1))

theorem geometric_sequence_problem (m : ℝ) (h_m : m ≠ 0) :
  ∃ (a : ℕ → ℝ), 
    (∃ r, a = geometric_sequence m r) ∧ 
    b_sequence a 1 = m ∧
    b_sequence a 2 = 3/2 * m ∧
    (∀ n : ℕ, n > 0 → 1 ≤ sequence_sum a n ∧ sequence_sum a n ≤ 3) →
    (∀ n, a n = m * (-1/2)^(n-1)) ∧
    (2 ≤ m ∧ m ≤ 3) := by
  sorry

end geometric_sequence_problem_l804_80410


namespace acrobats_count_correct_l804_80454

/-- The number of acrobats at the farm. -/
def num_acrobats : ℕ := 13

/-- The number of elephants at the farm. -/
def num_elephants : ℕ := sorry

/-- The number of horses at the farm. -/
def num_horses : ℕ := sorry

/-- The total number of legs at the farm. -/
def total_legs : ℕ := 54

/-- The total number of heads at the farm. -/
def total_heads : ℕ := 20

/-- Theorem stating that the number of acrobats is correct given the conditions. -/
theorem acrobats_count_correct :
  2 * num_acrobats + 4 * num_elephants + 4 * num_horses = total_legs ∧
  num_acrobats + num_elephants + num_horses = total_heads ∧
  num_acrobats = 13 := by
  sorry


end acrobats_count_correct_l804_80454


namespace boat_upstream_speed_l804_80473

/-- Proves that the upstream speed is approximately 29.82 miles per hour given the conditions of the boat problem -/
theorem boat_upstream_speed (distance : ℝ) (downstream_time : ℝ) (time_difference : ℝ) :
  distance = 90 ∧ 
  downstream_time = 2.5191640969412834 ∧ 
  time_difference = 0.5 →
  ∃ upstream_speed : ℝ, 
    distance = upstream_speed * (downstream_time + time_difference) ∧
    abs (upstream_speed - 29.82) < 0.01 := by
  sorry

end boat_upstream_speed_l804_80473
