import Mathlib

namespace right_triangle_leg_l233_23373

theorem right_triangle_leg (a c : ℝ) (h1 : a = 12) (h2 : c = 13) :
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ b = 5 := by
  sorry

end right_triangle_leg_l233_23373


namespace factorization_identities_l233_23368

theorem factorization_identities :
  (∀ x y : ℝ, x^4 - 16*y^4 = (x^2 + 4*y^2)*(x + 2*y)*(x - 2*y)) ∧
  (∀ a : ℝ, -2*a^3 + 12*a^2 - 16*a = -2*a*(a - 2)*(a - 4)) := by
sorry

end factorization_identities_l233_23368


namespace current_waiting_room_count_l233_23370

/-- The number of people in the interview room -/
def interview_room_count : ℕ := 5

/-- The number of people currently in the waiting room -/
def waiting_room_count : ℕ := 22

/-- The condition that if three more people arrive in the waiting room,
    the number becomes five times the number of people in the interview room -/
axiom waiting_room_condition :
  waiting_room_count + 3 = 5 * interview_room_count

theorem current_waiting_room_count :
  waiting_room_count = 22 :=
sorry

end current_waiting_room_count_l233_23370


namespace min_omega_value_l233_23385

open Real

theorem min_omega_value (f : ℝ → ℝ) (ω φ : ℝ) (x₀ : ℝ) :
  (ω > 0) →
  (∀ x, f x = sin (ω * x + φ)) →
  (∀ x, f x₀ ≤ f x ∧ f x ≤ f (x₀ + 2016 * π)) →
  (∀ ω' > 0, (∀ x, f x₀ ≤ sin (ω' * x + φ) ∧ sin (ω' * x + φ) ≤ f (x₀ + 2016 * π)) → ω ≤ ω') →
  ω = 1 / 2016 :=
sorry

end min_omega_value_l233_23385


namespace product_of_powers_equals_product_of_consecutive_integers_l233_23375

theorem product_of_powers_equals_product_of_consecutive_integers (k : ℕ) :
  (∃ (a b : ℕ), 2^a * 3^b = k * (k + 1)) ↔ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 8 := by
  sorry

end product_of_powers_equals_product_of_consecutive_integers_l233_23375


namespace truncated_pyramid_volume_l233_23306

/-- Given a truncated pyramid with base areas S₁ and S₂ (where S₁ < S₂) and volume V,
    the volume of the complete pyramid is (V * S₂ * √S₂) / (S₂ * √S₂ - S₁ * √S₁) -/
theorem truncated_pyramid_volume (S₁ S₂ V : ℝ) (h₁ : 0 < S₁) (h₂ : S₁ < S₂) (h₃ : 0 < V) :
  let complete_volume := (V * S₂ * Real.sqrt S₂) / (S₂ * Real.sqrt S₂ - S₁ * Real.sqrt S₁)
  ∃ (h : ℝ), h > 0 ∧ complete_volume = (1 / 3) * S₂ * h :=
by sorry

end truncated_pyramid_volume_l233_23306


namespace min_surface_area_height_l233_23352

/-- Represents a square-bottomed, lidless rectangular tank -/
structure Tank where
  side : ℝ
  height : ℝ

/-- The volume of the tank -/
def volume (t : Tank) : ℝ := t.side^2 * t.height

/-- The surface area of the tank -/
def surfaceArea (t : Tank) : ℝ := t.side^2 + 4 * t.side * t.height

/-- Theorem: For a tank with volume 4, the height that minimizes surface area is 1 -/
theorem min_surface_area_height :
  ∃ (t : Tank), volume t = 4 ∧ 
    (∀ (t' : Tank), volume t' = 4 → surfaceArea t ≤ surfaceArea t') ∧
    t.height = 1 := by
  sorry

end min_surface_area_height_l233_23352


namespace four_true_propositions_l233_23353

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define angle and side length for a triangle
def angle (t : Triangle) (v : Fin 3) : ℝ := sorry
def side_length (t : Triangle) (v : Fin 3) : ℝ := sorry

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define properties for a quadrilateral
def is_parallelogram (q : Quadrilateral) : Prop := sorry
def diagonals_bisect (q : Quadrilateral) : Prop := sorry

-- The four propositions
theorem four_true_propositions :
  (∀ t : Triangle, angle t 2 > angle t 1 → side_length t 0 > side_length t 1) ∧
  (∀ a b : ℝ, a * b ≠ 0 → a = 0 ∨ b ≠ 0) ∧
  (∀ a b : ℝ, a * b = 0 → a = 0 ∨ b = 0) ∧
  (∀ q : Quadrilateral, diagonals_bisect q → is_parallelogram q) :=
sorry

end four_true_propositions_l233_23353


namespace monotonic_function_a_range_l233_23349

/-- A function f is monotonic on an interval [a,b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The main theorem stating the range of 'a' for which the given function
    is monotonic on the interval [1,3]. -/
theorem monotonic_function_a_range :
  ∀ a : ℝ,
  (IsMonotonic (fun x => (1/3) * x^3 + a * x^2 + 5 * x + 6) 1 3) →
  (a ≤ -3 ∨ a ≥ -Real.sqrt 5) :=
by sorry

end monotonic_function_a_range_l233_23349


namespace arithmetic_sequence_property_l233_23335

/-- Arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (d : ℝ) (m : ℕ)
  (h_arith : arithmetic_sequence a d)
  (h_d_neq_0 : d ≠ 0)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_am : a m = 8) :
  m = 8 := by
sorry

end arithmetic_sequence_property_l233_23335


namespace fish_bucket_problem_l233_23381

/-- Proves that the number of buckets is 9 given the conditions of the fish problem -/
theorem fish_bucket_problem (total_fish_per_bucket : ℕ) (mackerels_per_bucket : ℕ) (total_mackerels : ℕ)
  (h1 : total_fish_per_bucket = 9)
  (h2 : mackerels_per_bucket = 3)
  (h3 : total_mackerels = 27) :
  total_mackerels / mackerels_per_bucket = 9 := by
  sorry

end fish_bucket_problem_l233_23381


namespace product_of_numbers_with_given_sum_and_difference_l233_23333

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 24 ∧ x - y = 8 → x * y = 128 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l233_23333


namespace bill_donut_order_combinations_l233_23341

/-- The number of ways to distribute remaining donuts after ensuring at least one of each kind -/
def donut_combinations (total_donuts : ℕ) (donut_kinds : ℕ) (remaining_donuts : ℕ) : ℕ :=
  Nat.choose (remaining_donuts + donut_kinds - 1) (donut_kinds - 1)

/-- Theorem stating the number of combinations for Bill's donut order -/
theorem bill_donut_order_combinations :
  donut_combinations 8 5 3 = 35 := by
  sorry

end bill_donut_order_combinations_l233_23341


namespace nonAthleticParentsCount_l233_23356

/-- Represents the number of students with various athletic parent combinations -/
structure AthleticParents where
  total : Nat
  athleticDad : Nat
  athleticMom : Nat
  bothAthletic : Nat

/-- Calculates the number of students with both non-athletic parents -/
def nonAthleticParents (ap : AthleticParents) : Nat :=
  ap.total - (ap.athleticDad + ap.athleticMom - ap.bothAthletic)

/-- Theorem stating that given the specific numbers in the problem, 
    the number of students with both non-athletic parents is 19 -/
theorem nonAthleticParentsCount : 
  let ap : AthleticParents := {
    total := 45,
    athleticDad := 17,
    athleticMom := 20,
    bothAthletic := 11
  }
  nonAthleticParents ap = 19 := by
  sorry

end nonAthleticParentsCount_l233_23356


namespace third_quadrant_angle_property_l233_23318

theorem third_quadrant_angle_property (α : Real) : 
  (3 * π / 2 < α) ∧ (α < 2 * π) →
  |Real.sin (α / 2)| / Real.sin (α / 2) + |Real.cos (α / 2)| / Real.cos (α / 2) + 3 = 3 :=
by sorry

end third_quadrant_angle_property_l233_23318


namespace youngest_child_age_problem_l233_23389

/-- The age of the youngest child given the conditions of the problem -/
def youngest_child_age (n : ℕ) (interval : ℕ) (sum : ℕ) : ℕ :=
  (sum - (n - 1) * n * interval / 2) / n

/-- Theorem stating the age of the youngest child under the given conditions -/
theorem youngest_child_age_problem :
  youngest_child_age 5 2 55 = 7 := by
  sorry

end youngest_child_age_problem_l233_23389


namespace product_mod_seven_l233_23350

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end product_mod_seven_l233_23350


namespace cost_of_thousand_gum_l233_23330

/-- The cost of a single piece of gum in cents -/
def cost_of_one_gum : ℕ := 1

/-- The number of pieces of gum -/
def num_gum : ℕ := 1000

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The cost of multiple pieces of gum in dollars -/
def cost_in_dollars (n : ℕ) : ℚ :=
  (n * cost_of_one_gum : ℚ) / cents_per_dollar

theorem cost_of_thousand_gum :
  cost_in_dollars num_gum = 10 := by
  sorry

end cost_of_thousand_gum_l233_23330


namespace sheela_income_proof_l233_23361

/-- Sheela's monthly income in rupees -/
def monthly_income : ℝ := 17272.73

/-- The amount Sheela deposits in the bank in rupees -/
def deposit : ℝ := 3800

/-- The percentage of Sheela's monthly income that she deposits -/
def deposit_percentage : ℝ := 22

theorem sheela_income_proof :
  deposit = (deposit_percentage / 100) * monthly_income :=
by sorry

end sheela_income_proof_l233_23361


namespace water_displaced_squared_l233_23393

-- Define the dimensions of the barrel and cube
def barrel_radius : ℝ := 4
def barrel_height : ℝ := 10
def cube_side : ℝ := 8

-- Define the volume of water displaced
def water_displaced : ℝ := cube_side ^ 3

-- Theorem statement
theorem water_displaced_squared :
  water_displaced ^ 2 = 262144 := by sorry

end water_displaced_squared_l233_23393


namespace max_rooks_on_chessboard_sixteen_rooks_achievable_l233_23398

/-- Represents a chessboard with rooks --/
structure Chessboard :=
  (size : ℕ)
  (white_rooks : ℕ)
  (black_rooks : ℕ)

/-- Predicate to check if the rook placement is valid --/
def valid_placement (board : Chessboard) : Prop :=
  board.white_rooks ≤ board.size * 2 ∧ 
  board.black_rooks ≤ board.size * 2 ∧
  board.white_rooks = board.black_rooks

/-- Theorem stating the maximum number of rooks of each color --/
theorem max_rooks_on_chessboard :
  ∀ (board : Chessboard),
    board.size = 8 →
    valid_placement board →
    board.white_rooks ≤ 16 ∧
    board.black_rooks ≤ 16 :=
by sorry

/-- Theorem stating that 16 rooks of each color is achievable --/
theorem sixteen_rooks_achievable :
  ∃ (board : Chessboard),
    board.size = 8 ∧
    valid_placement board ∧
    board.white_rooks = 16 ∧
    board.black_rooks = 16 :=
by sorry

end max_rooks_on_chessboard_sixteen_rooks_achievable_l233_23398


namespace brothers_age_fraction_l233_23310

/-- Given three brothers with ages M, O, and Y, prove that Y/O = 1/3 -/
theorem brothers_age_fraction (M O Y : ℕ) : 
  Y = 5 → 
  M + O + Y = 28 → 
  O = 2 * (M - 1) + 1 → 
  Y / O = 1 / 3 := by
  sorry

end brothers_age_fraction_l233_23310


namespace halfway_point_sixth_twelfth_l233_23337

theorem halfway_point_sixth_twelfth (x y : ℚ) (hx : x = 1/6) (hy : y = 1/12) :
  (x + y) / 2 = 1/8 := by
  sorry

end halfway_point_sixth_twelfth_l233_23337


namespace davids_remaining_money_l233_23399

theorem davids_remaining_money (initial_amount spent_amount remaining_amount : ℕ) :
  initial_amount = 1800 →
  remaining_amount = spent_amount - 800 →
  initial_amount - spent_amount = remaining_amount →
  remaining_amount = 500 := by
sorry

end davids_remaining_money_l233_23399


namespace extreme_value_at_negative_three_l233_23383

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extreme_value_at_negative_three (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3)) →
  a = 5 := by sorry

end extreme_value_at_negative_three_l233_23383


namespace positive_sum_of_squares_implies_inequality_l233_23358

theorem positive_sum_of_squares_implies_inequality 
  (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x^2 + y^2 + z^2 = 8) : 
  x^8 + y^8 + z^8 > 16 * Real.sqrt (2/3) := by
  sorry

end positive_sum_of_squares_implies_inequality_l233_23358


namespace quadratic_inequality_solution_l233_23363

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c > 0 ↔ x < -2 ∨ x > 4) →
  f a b c 2 < f a b c (-1) ∧ f a b c (-1) < f a b c 5 := by
  sorry

end quadratic_inequality_solution_l233_23363


namespace eight_couples_handshakes_l233_23339

/-- The number of handshakes in a gathering of couples -/
def count_handshakes (n : ℕ) : ℕ :=
  let total_people := 2 * n
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 8 couples, where each person shakes hands with
    everyone except their spouse and one other person, the total number of
    handshakes is 104. -/
theorem eight_couples_handshakes :
  count_handshakes 8 = 104 := by
  sorry

#eval count_handshakes 8  -- Should output 104

end eight_couples_handshakes_l233_23339


namespace max_tickets_after_scarf_l233_23328

def ticket_cost : ℕ := 15
def initial_money : ℕ := 160
def scarf_cost : ℕ := 25

theorem max_tickets_after_scarf : 
  ∀ n : ℕ, n ≤ (initial_money - scarf_cost) / ticket_cost → n ≤ 9 :=
by sorry

end max_tickets_after_scarf_l233_23328


namespace water_bottles_per_day_l233_23359

theorem water_bottles_per_day (total_bottles : ℕ) (total_days : ℕ) (bottles_per_day : ℕ) : 
  total_bottles = 153 → 
  total_days = 17 → 
  total_bottles = bottles_per_day * total_days → 
  bottles_per_day = 9 := by
sorry

end water_bottles_per_day_l233_23359


namespace quadratic_always_nonnegative_implies_a_range_l233_23317

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → a ∈ Set.Icc (-1 : ℝ) 3 :=
by sorry

end quadratic_always_nonnegative_implies_a_range_l233_23317


namespace intersection_of_M_and_N_l233_23348

def M : Set ℤ := {0, 1, 2, 3, 4}
def N : Set ℤ := {-2, 0, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 2} := by
  sorry

end intersection_of_M_and_N_l233_23348


namespace molecular_weight_proof_l233_23388

/-- Given a compound where 7 moles have a total molecular weight of 854 grams,
    prove that the molecular weight of 1 mole is 122 grams/mole. -/
theorem molecular_weight_proof (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 854)
  (h2 : num_moles = 7) :
  total_weight / num_moles = 122 := by
  sorry

end molecular_weight_proof_l233_23388


namespace ab_greater_than_a_plus_b_l233_23331

theorem ab_greater_than_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a - b = a / b) : a * b > a + b := by
  sorry

end ab_greater_than_a_plus_b_l233_23331


namespace avocado_cost_l233_23332

theorem avocado_cost (initial_amount : ℕ) (num_avocados : ℕ) (change : ℕ) : 
  initial_amount = 20 → num_avocados = 3 → change = 14 → 
  (initial_amount - change) / num_avocados = 2 := by
  sorry

end avocado_cost_l233_23332


namespace volume_of_rotated_composite_shape_l233_23395

/-- The volume of a solid formed by rotating a composite shape about the x-axis -/
theorem volume_of_rotated_composite_shape (π : ℝ) :
  let rectangle1_height : ℝ := 6
  let rectangle1_width : ℝ := 1
  let rectangle2_height : ℝ := 3
  let rectangle2_width : ℝ := 3
  let volume1 : ℝ := π * rectangle1_height^2 * rectangle1_width
  let volume2 : ℝ := π * rectangle2_height^2 * rectangle2_width
  let total_volume : ℝ := volume1 + volume2
  total_volume = 63 * π :=
by sorry

end volume_of_rotated_composite_shape_l233_23395


namespace field_division_theorem_l233_23351

/-- Represents a rectangular field of squares -/
structure RectangularField where
  width : ℕ
  height : ℕ
  total_squares : ℕ
  h_total : total_squares = width * height

/-- Represents a line dividing the field -/
structure DividingLine where
  x : ℕ
  y : ℕ

/-- Calculates the area of a triangle formed by a dividing line -/
def triangle_area (field : RectangularField) (line : DividingLine) : ℕ :=
  line.x * line.y / 2

theorem field_division_theorem (field : RectangularField) 
  (h_total : field.total_squares = 18) 
  (line1 : DividingLine) 
  (line2 : DividingLine) 
  (h_line1 : line1 = ⟨4, field.height⟩) 
  (h_line2 : line2 = ⟨field.width, 2⟩) :
  triangle_area field line1 = 6 ∧ 
  triangle_area field line2 = 6 ∧ 
  field.total_squares - triangle_area field line1 - triangle_area field line2 = 6 := by
  sorry

#check field_division_theorem

end field_division_theorem_l233_23351


namespace regression_and_range_correct_l233_23344

/-- Represents a data point with protein content and production cost -/
structure DataPoint where
  x : Float  -- protein content
  y : Float  -- production cost

/-- The set of given data points -/
def dataPoints : List DataPoint := [
  ⟨0, 19⟩, ⟨0.69, 32⟩, ⟨1.39, 40⟩, ⟨1.79, 44⟩, ⟨2.40, 52⟩, ⟨2.56, 53⟩, ⟨2.94, 54⟩
]

/-- The mean of x values -/
def xMean : Float := 1.68

/-- The mean of y values -/
def yMean : Float := 42

/-- The sum of squared differences between x values and their mean -/
def sumSquaredDiffX : Float := 6.79

/-- The sum of the product of differences between x values and their mean,
    and y values and their mean -/
def sumProductDiff : Float := 81.41

/-- Calculates the slope of the regression line -/
def calculateSlope (sumProductDiff sumSquaredDiffX : Float) : Float :=
  sumProductDiff / sumSquaredDiffX

/-- Calculates the y-intercept of the regression line -/
def calculateIntercept (slope xMean yMean : Float) : Float :=
  yMean - slope * xMean

/-- The regression equation -/
def regressionEquation (x : Float) (slope intercept : Float) : Float :=
  slope * x + intercept

/-- Theorem stating the correctness of the regression equation and protein content range -/
theorem regression_and_range_correct :
  let slope := calculateSlope sumProductDiff sumSquaredDiffX
  let intercept := calculateIntercept slope xMean yMean
  (∀ x, regressionEquation x slope intercept = 11.99 * x + 21.86) ∧
  (∀ y, 60 ≤ y ∧ y ≤ 70 → 
    3.18 ≤ (y - intercept) / slope ∧ (y - intercept) / slope ≤ 4.02) := by
  sorry

end regression_and_range_correct_l233_23344


namespace monotone_increasing_f_implies_a_range_l233_23357

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + 2*x - 2 * Real.log x

theorem monotone_increasing_f_implies_a_range (h : Monotone f) :
  (∀ x > 0, 2 * a ≤ x^2 + 2*x) → a ≤ 0 := by sorry

end monotone_increasing_f_implies_a_range_l233_23357


namespace area_F1AB_when_slope_is_one_line_equation_when_y_intercept_smallest_l233_23302

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define a point on the ellipse
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : (x^2 / e.a^2) + (y^2 / e.b^2) = 1

-- Define the focal distance
def focalDistance (e : Ellipse) : ℝ := 3

-- Define the ratio of major axis to focal distance
axiom majorAxisFocalRatio (e : Ellipse) : 2 * e.a / (2 * focalDistance e) = Real.sqrt 2

-- Define the right focus
def rightFocus : ℝ × ℝ := (3, 0)

-- Define a line passing through the right focus
structure LineThroughRightFocus where
  slope : ℝ

-- Define the area of triangle F1AB
def areaF1AB (e : Ellipse) (l : LineThroughRightFocus) : ℝ := sorry

-- Define the y-intercept of the perpendicular bisector of AB
def yInterceptPerpBisector (e : Ellipse) (l : LineThroughRightFocus) : ℝ := sorry

-- Theorem 1
theorem area_F1AB_when_slope_is_one (e : Ellipse) :
  areaF1AB e { slope := 1 } = 12 := sorry

-- Theorem 2
theorem line_equation_when_y_intercept_smallest (e : Ellipse) :
  ∃ (l : LineThroughRightFocus),
    (∀ (l' : LineThroughRightFocus), yInterceptPerpBisector e l ≤ yInterceptPerpBisector e l') ∧
    l.slope = -Real.sqrt 2 / 2 := sorry

end area_F1AB_when_slope_is_one_line_equation_when_y_intercept_smallest_l233_23302


namespace orange_pyramid_sum_l233_23329

/-- Calculates the number of oranges in a single layer of the pyramid -/
def oranges_in_layer (n : ℕ) : ℕ := n * n / 2

/-- Calculates the total number of oranges in the pyramid stack -/
def total_oranges (base_size : ℕ) : ℕ :=
  (List.range base_size).map oranges_in_layer |>.sum

/-- The theorem stating that a pyramid with base size 6 contains 44 oranges -/
theorem orange_pyramid_sum : total_oranges 6 = 44 := by
  sorry

#eval total_oranges 6

end orange_pyramid_sum_l233_23329


namespace gcd_of_2_powers_l233_23313

theorem gcd_of_2_powers : Nat.gcd (2^1040 - 1) (2^1030 - 1) = 1023 := by sorry

end gcd_of_2_powers_l233_23313


namespace nested_square_root_simplification_l233_23360

theorem nested_square_root_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (y * Real.sqrt (y * Real.sqrt (y * Real.sqrt y))) = (y ^ 9) ^ (1/4) := by
  sorry

end nested_square_root_simplification_l233_23360


namespace mouse_meiosis_observation_l233_23372

/-- Available materials for mouse cell meiosis observation --/
inductive Material
  | MouseKidney
  | MouseTestis
  | MouseLiver
  | SudanIIIStain
  | GentianVioletSolution
  | JanusGreenBStain
  | DissociationFixativeSolution

/-- Types of cells produced during meiosis --/
inductive DaughterCell
  | Spermatogonial
  | SecondarySpermatocyte
  | Spermatid

/-- Theorem for correct mouse cell meiosis observation procedure --/
theorem mouse_meiosis_observation 
  (available_materials : List Material)
  (meiosis_occurs_in_gonads : Bool)
  (spermatogonial_cells_undergo_mitosis_and_meiosis : Bool) :
  (MouseTestis ∈ available_materials) →
  (DissociationFixativeSolution ∈ available_materials) →
  (GentianVioletSolution ∈ available_materials) →
  meiosis_occurs_in_gonads →
  spermatogonial_cells_undergo_mitosis_and_meiosis →
  (correct_tissue = MouseTestis) ∧
  (post_hypotonic_solution = DissociationFixativeSolution) ∧
  (staining_solution = GentianVioletSolution) ∧
  (daughter_cells = [DaughterCell.Spermatogonial, DaughterCell.SecondarySpermatocyte, DaughterCell.Spermatid]) := by
  sorry

end mouse_meiosis_observation_l233_23372


namespace percentage_left_approx_20_l233_23380

-- Define the initial population
def initial_population : ℕ := 4675

-- Define the percentage of people who died
def death_percentage : ℚ := 5 / 100

-- Define the final population
def final_population : ℕ := 3553

-- Define the function to calculate the percentage who left
def percentage_left (init : ℕ) (death_perc : ℚ) (final : ℕ) : ℚ :=
  let remaining := init - (init * death_perc).floor
  ((remaining - final : ℚ) / remaining) * 100

-- Theorem statement
theorem percentage_left_approx_20 :
  ∃ ε > 0, abs (percentage_left initial_population death_percentage final_population - 20) < ε :=
sorry

end percentage_left_approx_20_l233_23380


namespace min_value_implies_a_l233_23366

def f (x a : ℝ) : ℝ := |x + 1| + 2 * |x - a|

theorem min_value_implies_a (a : ℝ) : 
  (∀ x, f x a ≥ 5) ∧ (∃ x, f x a = 5) → a = -6 ∨ a = 4 := by
  sorry

end min_value_implies_a_l233_23366


namespace rectangle_y_value_l233_23340

/-- Rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of rectangle EFGH is 40 square units -/
def area (r : Rectangle) : ℝ := 5 * r.y

theorem rectangle_y_value (r : Rectangle) (h_area : area r = 40) : r.y = 8 := by
  sorry

end rectangle_y_value_l233_23340


namespace painter_inventory_theorem_l233_23326

/-- Represents the painter's paint inventory and room painting capacity -/
structure PainterInventory where
  initialRooms : ℕ
  remainingRooms : ℕ
  lostCans : ℕ

/-- Calculates the number of cans needed to paint a given number of rooms -/
def cansNeeded (inventory : PainterInventory) (rooms : ℕ) : ℕ :=
  (rooms * inventory.lostCans) / (inventory.initialRooms - inventory.remainingRooms)

/-- Theorem stating that under the given conditions, 15 cans are needed to paint 25 rooms -/
theorem painter_inventory_theorem (inventory : PainterInventory) 
  (h1 : inventory.initialRooms = 30)
  (h2 : inventory.remainingRooms = 25)
  (h3 : inventory.lostCans = 3) :
  cansNeeded inventory 25 = 15 := by
  sorry

end painter_inventory_theorem_l233_23326


namespace peter_speed_proof_l233_23322

/-- Peter's speed in miles per hour -/
def peter_speed : ℝ := 5

/-- Juan's speed in miles per hour -/
def juan_speed : ℝ := peter_speed + 3

/-- Time traveled in hours -/
def time : ℝ := 1.5

/-- Total distance between Juan and Peter after traveling -/
def total_distance : ℝ := 19.5

theorem peter_speed_proof :
  peter_speed * time + juan_speed * time = total_distance :=
by sorry

end peter_speed_proof_l233_23322


namespace bike_ride_time_l233_23336

theorem bike_ride_time (distance1 distance2 time1 : ℝ) (distance1_pos : 0 < distance1) (time1_pos : 0 < time1) :
  distance1 = 2 ∧ time1 = 6 ∧ distance2 = 5 →
  distance2 / (distance1 / time1) = 15 := by sorry

end bike_ride_time_l233_23336


namespace divisible_by_2520_l233_23316

theorem divisible_by_2520 (n : ℕ) : ∃ k : ℤ, (n^7 : ℤ) - 14*(n^5 : ℤ) + 49*(n^3 : ℤ) - 36*(n : ℤ) = 2520 * k := by
  sorry

end divisible_by_2520_l233_23316


namespace largest_prime_factor_of_4519_l233_23347

theorem largest_prime_factor_of_4519 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4519 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4519 → q ≤ p :=
by sorry

end largest_prime_factor_of_4519_l233_23347


namespace g_of_3_eq_neg_1_l233_23314

def g (x : ℝ) : ℝ := 2 * (x - 2)^2 - 3 * (x - 2)

theorem g_of_3_eq_neg_1 : g 3 = -1 := by
  sorry

end g_of_3_eq_neg_1_l233_23314


namespace cubic_equation_solution_l233_23342

theorem cubic_equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃! (a b c : ℝ), ∀ x : ℝ, 
    (x + m)^3 - (x + n)^3 = (m + n)^3 ∧ x = a * m + b * n + c :=
by
  sorry

end cubic_equation_solution_l233_23342


namespace regular_polygon_sides_l233_23325

theorem regular_polygon_sides (n : ℕ) : 
  n > 3 → 
  (n : ℚ) / (n * (n - 3) / 2 : ℚ) = 1/4 → 
  n = 11 := by
sorry

end regular_polygon_sides_l233_23325


namespace three_possible_values_l233_23387

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n ≤ 99999

def construct_number (a b : ℕ) : ℕ := a * 10000 + 3750 + b

theorem three_possible_values :
  ∃ (s : Finset ℕ), s.card = 3 ∧
    (∀ a ∈ s, is_single_digit a ∧
      (∃ b, is_single_digit b ∧
        is_five_digit (construct_number a b) ∧
        (construct_number a b) % 24 = 0)) ∧
    (∀ a, is_single_digit a →
      (∃ b, is_single_digit b ∧
        is_five_digit (construct_number a b) ∧
        (construct_number a b) % 24 = 0) →
      a ∈ s) :=
sorry

end three_possible_values_l233_23387


namespace existence_of_divisible_n_l233_23382

theorem existence_of_divisible_n : ∃ (n : ℕ), n > 0 ∧ (2009 * 2010 * 2011) ∣ ((n^2 - 5) * (n^2 + 6) * (n^2 + 30)) := by
  sorry

end existence_of_divisible_n_l233_23382


namespace percentage_of_students_with_glasses_l233_23305

def total_students : ℕ := 325
def students_without_glasses : ℕ := 195

theorem percentage_of_students_with_glasses :
  (((total_students - students_without_glasses) : ℚ) / total_students) * 100 = 40 := by
  sorry

end percentage_of_students_with_glasses_l233_23305


namespace complex_modulus_l233_23386

theorem complex_modulus (z : ℂ) (h : (z - 2) * (1 - Complex.I) = 1 + Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l233_23386


namespace octal_subtraction_l233_23323

def octal_to_decimal (n : ℕ) : ℕ := sorry

def decimal_to_octal (n : ℕ) : ℕ := sorry

theorem octal_subtraction :
  decimal_to_octal (octal_to_decimal 5374 - octal_to_decimal 2645) = 1527 := by sorry

end octal_subtraction_l233_23323


namespace quadratic_factorization_l233_23303

theorem quadratic_factorization (a : ℝ) : a^2 + 4*a + 4 = (a + 2)^2 := by
  sorry

end quadratic_factorization_l233_23303


namespace minimize_sum_of_distances_l233_23319

/-- The point that minimizes the sum of distances to two fixed points lies on the line connecting the first point and the reflection of the second point across the y-axis. -/
theorem minimize_sum_of_distances (A B C : ℝ × ℝ) (h1 : A = (3, 3)) (h2 : B = (-1, -1)) (h3 : C.1 = -3) :
  (∃ k : ℝ, C = (-3, k) ∧ 
    (∀ k' : ℝ, dist A C + dist B C ≤ dist A (C.1, k') + dist B (C.1, k'))) →
  C.2 = -9 :=
by sorry


end minimize_sum_of_distances_l233_23319


namespace grain_warehouse_analysis_l233_23320

def grain_records : List Int := [26, -32, -25, 34, -38, 10]
def current_stock : Int := 480

theorem grain_warehouse_analysis :
  (List.sum grain_records < 0) ∧
  (current_stock - List.sum grain_records = 505) := by
  sorry

end grain_warehouse_analysis_l233_23320


namespace complex_modulus_equality_l233_23301

theorem complex_modulus_equality (t : ℝ) : 
  t > 0 → Complex.abs (-3 + t * Complex.I) = 5 → t = 4 := by sorry

end complex_modulus_equality_l233_23301


namespace not_coplanar_implies_not_collinear_three_collinear_implies_coplanar_l233_23367

-- Define a type for points in space
variable (Point : Type)

-- Define the property of being coplanar for four points
variable (coplanar : Point → Point → Point → Point → Prop)

-- Define the property of being collinear for three points
variable (collinear : Point → Point → Point → Prop)

-- Theorem 1: If four points are not coplanar, then any three of them are not collinear
theorem not_coplanar_implies_not_collinear 
  (p q r s : Point) : 
  ¬(coplanar p q r s) → 
  (¬(collinear p q r) ∧ ¬(collinear p q s) ∧ ¬(collinear p r s) ∧ ¬(collinear q r s)) :=
sorry

-- Theorem 2: If there exist three collinear points among four points, then these four points are coplanar
theorem three_collinear_implies_coplanar 
  (p q r s : Point) : 
  (collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) → 
  coplanar p q r s :=
sorry

end not_coplanar_implies_not_collinear_three_collinear_implies_coplanar_l233_23367


namespace new_circle_equation_l233_23397

/-- Given a circle with equation x^2 + 2x + y^2 = 0, prove that a new circle 
    with the same center and radius 2 has the equation (x+1)^2 + y^2 = 4 -/
theorem new_circle_equation (x y : ℝ) : 
  (∀ x y, x^2 + 2*x + y^2 = 0 → ∃ h k, (x - h)^2 + (y - k)^2 = 1) →
  (∀ x y, (x + 1)^2 + y^2 = 4 ↔ (x - (-1))^2 + (y - 0)^2 = 2^2) :=
by sorry

end new_circle_equation_l233_23397


namespace marcella_shoes_theorem_l233_23346

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_remaining_pairs (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - min initial_pairs shoes_lost

/-- Theorem stating that with 27 initial pairs and 9 individual shoes lost,
    the maximum number of complete pairs remaining is 18. -/
theorem marcella_shoes_theorem :
  max_remaining_pairs 27 9 = 18 := by
  sorry

end marcella_shoes_theorem_l233_23346


namespace arithmetic_geometric_sequence_equality_l233_23365

/-- Arithmetic sequence {a_n} -/
def a (n : ℕ) : ℝ := 2*n + 2

/-- Geometric sequence {b_n} -/
def b (n : ℕ) : ℝ := 8 * 2^(n-2)

theorem arithmetic_geometric_sequence_equality :
  (a 1 + a 2 = 10) →
  (a 4 - a 3 = 2) →
  (b 2 = a 3) →
  (b 3 = a 7) →
  (a 15 = b 4) := by
  sorry

end arithmetic_geometric_sequence_equality_l233_23365


namespace exist_abcd_equation_l233_23390

theorem exist_abcd_equation : ∃ (a b c d : ℕ), 
  (a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1) ∧
  (1 / a + 1 / (a * b) + 1 / (a * b * c) + 1 / (a * b * c * d) : ℚ) = 37 / 48 ∧
  b = 4 := by sorry

end exist_abcd_equation_l233_23390


namespace triangle_property_l233_23394

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + c^2 = b^2 + Real.sqrt 2 * a * c →
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 1 ∧ 
    a = b * y ∧ b = c * z ∧ c = a * x) →
  B = π / 4 ∧ 
  (∀ A' C', A' + C' = 3 * π / 4 → 
    Real.sqrt 2 * Real.cos A' + Real.cos C' ≤ 1) :=
by sorry

end triangle_property_l233_23394


namespace area_difference_l233_23391

/-- The difference in area between a square and a rectangle -/
theorem area_difference (square_side : ℝ) (rect_length rect_width : ℝ) : 
  square_side = 5 → rect_length = 3 → rect_width = 6 → 
  square_side * square_side - rect_length * rect_width = 7 := by
  sorry

#check area_difference

end area_difference_l233_23391


namespace sinusoidal_oscillations_l233_23369

/-- A sinusoidal function that completes 5 oscillations from 0 to 2π has b = 5 -/
theorem sinusoidal_oscillations (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, (a * Real.sin (b * x + c) + d = a * Real.sin (b * (x + 2 * Real.pi) + c) + d)) →
  (∃ n : ℕ, n = 5 ∧ ∀ x : ℝ, a * Real.sin (b * x + c) + d = a * Real.sin (b * (x + 2 * Real.pi / n) + c) + d) →
  b = 5 := by
  sorry

end sinusoidal_oscillations_l233_23369


namespace least_number_with_remainder_l233_23309

theorem least_number_with_remainder (n : ℕ) : n = 282 ↔ 
  (n > 0 ∧ 
   n % 31 = 3 ∧ 
   n % 9 = 3 ∧ 
   ∀ m : ℕ, m > 0 → m % 31 = 3 → m % 9 = 3 → m ≥ n) := by
sorry

end least_number_with_remainder_l233_23309


namespace store_a_cheapest_l233_23343

/-- Represents the cost calculation for purchasing soccer balls from different stores -/
def soccer_ball_cost (num_balls : ℕ) : ℕ → ℕ
| 0 => num_balls * 25 - (num_balls / 10) * 3 * 25  -- Store A
| 1 => num_balls * (25 - 5)                        -- Store B
| 2 => num_balls * 25 - ((num_balls * 25) / 200) * 40  -- Store C
| _ => 0  -- Invalid store

theorem store_a_cheapest :
  let num_balls : ℕ := 58
  soccer_ball_cost num_balls 0 < soccer_ball_cost num_balls 1 ∧
  soccer_ball_cost num_balls 0 < soccer_ball_cost num_balls 2 :=
by sorry

end store_a_cheapest_l233_23343


namespace square_binomial_constant_l233_23321

theorem square_binomial_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 100*x + c = (x + a)^2) → c = 2500 :=
by sorry

end square_binomial_constant_l233_23321


namespace min_abs_z_on_line_segment_l233_23334

theorem min_abs_z_on_line_segment (z : ℂ) (h : Complex.abs (z - 2*Complex.I) + Complex.abs (z - 5) = 7) :
  ∃ (min_abs_z : ℝ), min_abs_z = Real.sqrt (100 / 29) ∧
  ∀ (w : ℂ), Complex.abs (w - 2*Complex.I) + Complex.abs (w - 5) = 7 →
  Complex.abs w ≥ min_abs_z :=
sorry

end min_abs_z_on_line_segment_l233_23334


namespace cookies_left_for_neil_l233_23354

def total_cookies : ℕ := 20
def fraction_given : ℚ := 2/5

theorem cookies_left_for_neil :
  total_cookies - (total_cookies * fraction_given).floor = 12 :=
by sorry

end cookies_left_for_neil_l233_23354


namespace coefficient_equals_49_l233_23345

/-- The coefficient of x^3y^5 in the expansion of (x+2y)(x-y)^7 -/
def coefficient : ℤ :=
  2 * (Nat.choose 7 4) - (Nat.choose 7 5)

/-- Theorem stating that the coefficient of x^3y^5 in the expansion of (x+2y)(x-y)^7 is 49 -/
theorem coefficient_equals_49 : coefficient = 49 := by
  sorry

end coefficient_equals_49_l233_23345


namespace waiter_customers_l233_23364

theorem waiter_customers (non_tipping : ℕ) (tip_amount : ℕ) (total_tips : ℕ) : 
  non_tipping = 5 → tip_amount = 3 → total_tips = 15 → 
  non_tipping + (total_tips / tip_amount) = 10 :=
by sorry

end waiter_customers_l233_23364


namespace weight_difference_theorem_l233_23308

-- Define the given conditions
def joe_weight : ℝ := 44
def initial_average : ℝ := 30
def new_average : ℝ := 31
def final_average : ℝ := 30

-- Define the number of students in the initial group
def initial_students : ℕ := 13

-- Define the theorem
theorem weight_difference_theorem :
  let total_weight_with_joe := initial_average * initial_students + joe_weight
  let remaining_students := initial_students + 1 - 2
  let final_total_weight := final_average * remaining_students
  let leaving_students_total_weight := total_weight_with_joe - final_total_weight
  let leaving_students_average_weight := leaving_students_total_weight / 2
  leaving_students_average_weight - joe_weight = -7 := by sorry

end weight_difference_theorem_l233_23308


namespace distinct_solutions_count_l233_23362

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 6*x + 5

-- State the theorem
theorem distinct_solutions_count :
  ∃! (s : Finset ℝ), (∀ d ∈ s, g (g (g (g d))) = 5) ∧ s.card = 6 := by
  sorry

end distinct_solutions_count_l233_23362


namespace factor_proof_l233_23324

theorem factor_proof :
  (∃ n : ℤ, 28 = 4 * n) ∧ (∃ m : ℤ, 162 = 9 * m) := by sorry

end factor_proof_l233_23324


namespace socks_combination_l233_23377

/-- The number of ways to choose k items from a set of n items, where order doesn't matter -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- There are 6 socks in the drawer -/
def total_socks : ℕ := 6

/-- We need to choose 4 socks -/
def socks_to_choose : ℕ := 4

/-- The number of ways to choose 4 socks from 6 socks is 15 -/
theorem socks_combination : choose total_socks socks_to_choose = 15 := by
  sorry

end socks_combination_l233_23377


namespace tangent_equality_implies_angle_l233_23312

theorem tangent_equality_implies_angle (x : Real) : 
  0 < x → x < 180 →
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 110 := by
sorry

end tangent_equality_implies_angle_l233_23312


namespace min_value_expression_min_value_achievable_l233_23374

theorem min_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 + (c - 4)^2 ≥ 10.1 :=
by sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 5 ∧
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 + (c - 4)^2 = 10.1 :=
by sorry

end min_value_expression_min_value_achievable_l233_23374


namespace millet_majority_day_l233_23384

def seed_mix : ℝ := 0.25
def millet_eaten_daily : ℝ := 0.25
def total_seeds_daily : ℝ := 1

def millet_proportion (n : ℕ) : ℝ := 1 - (1 - seed_mix)^n

theorem millet_majority_day :
  ∀ k : ℕ, k < 5 → millet_proportion k ≤ 0.5 ∧
  millet_proportion 5 > 0.5 := by sorry

end millet_majority_day_l233_23384


namespace consecutive_page_numbers_l233_23315

theorem consecutive_page_numbers : ∃ (n : ℕ), 
  n > 0 ∧ 
  n * (n + 1) = 20412 ∧ 
  n + (n + 1) = 283 := by
  sorry

end consecutive_page_numbers_l233_23315


namespace charlie_score_l233_23378

theorem charlie_score (team_total : ℕ) (num_players : ℕ) (others_average : ℕ) (h1 : team_total = 60) (h2 : num_players = 8) (h3 : others_average = 5) :
  team_total - (num_players - 1) * others_average = 25 := by
  sorry

end charlie_score_l233_23378


namespace james_fleet_capacity_l233_23311

/-- Represents the fleet of gas transportation vans --/
structure Fleet :=
  (total_vans : ℕ)
  (large_vans : ℕ)
  (medium_vans : ℕ)
  (small_van : ℕ)
  (medium_capacity : ℕ)
  (small_capacity : ℕ)
  (large_capacity : ℕ)

/-- Calculates the total capacity of the fleet --/
def total_capacity (f : Fleet) : ℕ :=
  f.large_vans * f.medium_capacity +
  f.medium_vans * f.medium_capacity +
  f.small_van * f.small_capacity +
  (f.total_vans - f.large_vans - f.medium_vans - f.small_van) * f.large_capacity

/-- Theorem stating the total capacity of James' fleet --/
theorem james_fleet_capacity :
  ∃ (f : Fleet),
    f.total_vans = 6 ∧
    f.medium_vans = 2 ∧
    f.small_van = 1 ∧
    f.medium_capacity = 8000 ∧
    f.small_capacity = (7 * f.medium_capacity) / 10 ∧
    f.large_capacity = (3 * f.medium_capacity) / 2 ∧
    total_capacity f = 57600 :=
  sorry

end james_fleet_capacity_l233_23311


namespace new_tram_properties_l233_23379

/-- Represents the properties of a tram journey -/
structure TramJourney where
  distance : ℝ
  old_time : ℝ
  new_time : ℝ
  old_speed : ℝ
  new_speed : ℝ

/-- Theorem stating the properties of the new tram journey -/
theorem new_tram_properties (j : TramJourney) 
  (h1 : j.distance = 20)
  (h2 : j.new_time = j.old_time - 1/5)
  (h3 : j.new_speed = j.old_speed + 5)
  (h4 : j.distance = j.old_speed * j.old_time)
  (h5 : j.distance = j.new_speed * j.new_time) :
  j.new_time = 4/5 ∧ j.new_speed = 25 := by
  sorry

end new_tram_properties_l233_23379


namespace product_47_33_l233_23307

theorem product_47_33 : 47 * 33 = 1551 := by
  sorry

end product_47_33_l233_23307


namespace ribbon_per_box_l233_23376

theorem ribbon_per_box
  (total_ribbon : ℝ)
  (num_boxes : ℕ)
  (remaining_ribbon : ℝ)
  (h1 : total_ribbon = 4.5)
  (h2 : num_boxes = 5)
  (h3 : remaining_ribbon = 1)
  : (total_ribbon - remaining_ribbon) / num_boxes = 0.7 := by
  sorry

end ribbon_per_box_l233_23376


namespace room_length_calculation_l233_23327

/-- Given a rectangular room with known width, paving cost per square meter, and total paving cost,
    calculate the length of the room. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ)
    (h1 : width = 4)
    (h2 : cost_per_sqm = 800)
    (h3 : total_cost = 17600) :
    total_cost / (width * cost_per_sqm) = 5.5 := by
  sorry

end room_length_calculation_l233_23327


namespace evaluate_fraction_l233_23371

theorem evaluate_fraction : (18 : ℝ) / (14 * 5.3) = 1.8 / 7.42 := by sorry

end evaluate_fraction_l233_23371


namespace parabola_vertex_l233_23300

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * (x + 9)^2 - 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-9, -3)

/-- Theorem: The vertex of the parabola y = 2(x+9)^2 - 3 is at the point (-9, -3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 := by
sorry

end parabola_vertex_l233_23300


namespace museum_visitors_l233_23304

theorem museum_visitors (V : ℕ) : 
  (130 : ℕ) + (3 * V / 4 : ℕ) = V → V = 520 :=
by sorry

end museum_visitors_l233_23304


namespace sin_2x_value_l233_23396

theorem sin_2x_value (x : ℝ) (h : Real.cos (π / 4 + x) = 3 / 5) : 
  Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_2x_value_l233_23396


namespace unique_solution_sqrt_equation_l233_23392

theorem unique_solution_sqrt_equation :
  ∃! x : ℝ, (x - 1 ≥ 0) ∧ (x + 1 ≥ 0) ∧ (x^2 - 1 ≥ 0) ∧
  (Real.sqrt (x - 1) * Real.sqrt (x + 1) = -Real.sqrt (x^2 - 1)) :=
by sorry

end unique_solution_sqrt_equation_l233_23392


namespace rectangular_field_area_l233_23338

theorem rectangular_field_area (w l : ℝ) : 
  l = 2 * w + 35 →
  2 * (w + l) = 700 →
  w * l = 25725 := by
sorry

end rectangular_field_area_l233_23338


namespace min_sticks_arrangement_l233_23355

theorem min_sticks_arrangement (n : ℕ) : n = 1012 ↔ 
  (n > 1000) ∧
  (∃ k : ℕ, n = 3 * k + 1) ∧
  (∃ m : ℕ, n = 5 * m + 2) ∧
  (∃ p : ℕ, n = 2 * p * (p + 1)) ∧
  (∀ x : ℕ, x > 1000 → 
    ((∃ k : ℕ, x = 3 * k + 1) ∧
     (∃ m : ℕ, x = 5 * m + 2) ∧
     (∃ p : ℕ, x = 2 * p * (p + 1))) → x ≥ n) :=
by sorry

end min_sticks_arrangement_l233_23355
