import Mathlib

namespace NUMINAMATH_CALUDE_line_plane_zero_angle_l2422_242220

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- The angle between a line and a plane -/
def angle_line_plane (l : Line3D) (p : Plane3D) : ℝ :=
  sorry

/-- A line is parallel to a plane -/
def is_parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line lies within a plane -/
def lies_within (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- If the angle between a line and a plane is 0°, then the line is either parallel to the plane or lies within it -/
theorem line_plane_zero_angle (l : Line3D) (p : Plane3D) :
  angle_line_plane l p = 0 → is_parallel l p ∨ lies_within l p :=
sorry

end NUMINAMATH_CALUDE_line_plane_zero_angle_l2422_242220


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_geq_five_l2422_242252

theorem inequality_holds_iff_p_geq_five (p : ℝ) :
  (∀ x : ℝ, x > 0 → Real.log (x + p) - (1/2 : ℝ) ≥ Real.log (Real.sqrt (2*x))) ↔ p ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_geq_five_l2422_242252


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2422_242259

/-- A function that checks if three numbers can form a right-angled triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only {2, 2, 3} cannot form a right-angled triangle -/
theorem right_triangle_sets :
  is_right_triangle 1 (Real.sqrt 2) (Real.sqrt 3) ∧
  is_right_triangle 3 4 5 ∧
  is_right_triangle 5 12 13 ∧
  ¬is_right_triangle 2 2 3 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_sets_l2422_242259


namespace NUMINAMATH_CALUDE_average_marks_of_all_students_l2422_242267

theorem average_marks_of_all_students
  (batch1_size : ℕ) (batch2_size : ℕ) (batch3_size : ℕ)
  (batch1_avg : ℝ) (batch2_avg : ℝ) (batch3_avg : ℝ)
  (h1 : batch1_size = 40)
  (h2 : batch2_size = 50)
  (h3 : batch3_size = 60)
  (h4 : batch1_avg = 45)
  (h5 : batch2_avg = 55)
  (h6 : batch3_avg = 65) :
  let total_students := batch1_size + batch2_size + batch3_size
  let total_marks := batch1_size * batch1_avg + batch2_size * batch2_avg + batch3_size * batch3_avg
  total_marks / total_students = 56.33 := by
sorry

end NUMINAMATH_CALUDE_average_marks_of_all_students_l2422_242267


namespace NUMINAMATH_CALUDE_max_sum_of_solutions_l2422_242213

def is_solution (x y : ℤ) : Prop := x^2 + y^2 = 100

theorem max_sum_of_solutions :
  ∃ (a b : ℤ), is_solution a b ∧ 
  (∀ (x y : ℤ), is_solution x y → x + y ≤ a + b) ∧
  a + b = 14 := by sorry

end NUMINAMATH_CALUDE_max_sum_of_solutions_l2422_242213


namespace NUMINAMATH_CALUDE_f_extremum_range_l2422_242230

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - a * x^2 + x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * a * x + 1

-- Define the condition for exactly one extremum point in (-1, 0)
def has_one_extremum (a : ℝ) : Prop :=
  ∃! x, x ∈ Set.Ioo (-1) 0 ∧ f' a x = 0

-- State the theorem
theorem f_extremum_range :
  ∀ a : ℝ, has_one_extremum a ↔ a ∈ Set.Ioi (-1/5) ∪ {-1} :=
sorry

end NUMINAMATH_CALUDE_f_extremum_range_l2422_242230


namespace NUMINAMATH_CALUDE_fourth_guard_distance_l2422_242253

theorem fourth_guard_distance (l w : ℝ) (h1 : l = 300) (h2 : w = 200) : 
  let P := 2 * (l + w)
  let three_guards_distance := 850
  let fourth_guard_distance := P - three_guards_distance
  fourth_guard_distance = 150 := by
sorry

end NUMINAMATH_CALUDE_fourth_guard_distance_l2422_242253


namespace NUMINAMATH_CALUDE_half_product_uniqueness_l2422_242255

theorem half_product_uniqueness (x : ℕ) :
  (∃ n : ℕ, x = n * (n + 1) / 2) →
  (∀ n k : ℕ, x = n * (n + 1) / 2 ∧ x = k * (k + 1) / 2 → n = k) :=
by sorry

end NUMINAMATH_CALUDE_half_product_uniqueness_l2422_242255


namespace NUMINAMATH_CALUDE_number_thought_of_l2422_242203

theorem number_thought_of (x : ℝ) : (x / 5 + 10 = 21) → x = 55 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l2422_242203


namespace NUMINAMATH_CALUDE_total_marbles_l2422_242288

/-- Given a bag of marbles with red, blue, green, and yellow marbles in the ratio 3:4:2:5,
    and 30 yellow marbles, prove that the total number of marbles is 84. -/
theorem total_marbles (red blue green yellow total : ℕ) 
  (h_ratio : red + blue + green + yellow = total)
  (h_proportion : 3 * yellow = 5 * red ∧ 4 * yellow = 5 * blue ∧ 2 * yellow = 5 * green)
  (h_yellow : yellow = 30) : total = 84 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l2422_242288


namespace NUMINAMATH_CALUDE_charlie_same_color_probability_l2422_242211

def total_marbles : ℕ := 10
def red_marbles : ℕ := 3
def green_marbles : ℕ := 3
def blue_marbles : ℕ := 4

def alice_draw : ℕ := 3
def bob_draw : ℕ := 3
def charlie_draw : ℕ := 4

theorem charlie_same_color_probability :
  let total_outcomes := (total_marbles.choose alice_draw) * ((total_marbles - alice_draw).choose bob_draw) * ((total_marbles - alice_draw - bob_draw).choose charlie_draw)
  let favorable_outcomes := 
    2 * (red_marbles.min green_marbles).choose 3 * (total_marbles - red_marbles - green_marbles).choose 1 +
    (blue_marbles.choose 3) * (total_marbles - blue_marbles).choose 1 +
    blue_marbles.choose 4
  (favorable_outcomes : ℚ) / total_outcomes = 13 / 1400 := by
  sorry

end NUMINAMATH_CALUDE_charlie_same_color_probability_l2422_242211


namespace NUMINAMATH_CALUDE_square_sum_problem_l2422_242277

theorem square_sum_problem (square triangle : ℝ) 
  (h1 : 2 * square + 2 * triangle = 16)
  (h2 : 2 * square + 3 * triangle = 19) :
  4 * square = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_problem_l2422_242277


namespace NUMINAMATH_CALUDE_sin_300_degrees_l2422_242245

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l2422_242245


namespace NUMINAMATH_CALUDE_valid_k_characterization_l2422_242204

/-- A function f: ℤ → ℤ is nonlinear if there exist x, y ∈ ℤ such that 
    f(x + y) ≠ f(x) + f(y) or f(ax) ≠ af(x) for some a ∈ ℤ -/
def Nonlinear (f : ℤ → ℤ) : Prop :=
  ∃ x y : ℤ, f (x + y) ≠ f x + f y ∨ ∃ a : ℤ, f (a * x) ≠ a * f x

/-- The set of non-negative integer values of k for which there exists a nonlinear function
    f: ℤ → ℤ satisfying the given equation for all integers a, b, c with a + b + c = 0 -/
def ValidK : Set ℕ :=
  {k : ℕ | ∃ f : ℤ → ℤ, Nonlinear f ∧
    ∀ a b c : ℤ, a + b + c = 0 →
      f a + f b + f c = (f (a - b) + f (b - c) + f (c - a)) / k}

theorem valid_k_characterization : ValidK = {0, 1, 3, 9} := by
  sorry

end NUMINAMATH_CALUDE_valid_k_characterization_l2422_242204


namespace NUMINAMATH_CALUDE_no_extreme_points_implies_m_range_l2422_242286

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + m*x + 1

-- Define what it means for f to have no extreme points
def has_no_extreme_points (m : ℝ) : Prop :=
  ∀ x : ℝ, ∃ ε > 0, ∀ y : ℝ, |y - x| < ε → f m y ≠ f m x ∨ (f m y < f m x ↔ y < x)

-- State the theorem
theorem no_extreme_points_implies_m_range (m : ℝ) :
  has_no_extreme_points m → m ≥ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_no_extreme_points_implies_m_range_l2422_242286


namespace NUMINAMATH_CALUDE_total_selling_price_l2422_242239

/-- Calculate the total selling price of three items given their cost prices and profit/loss percentages -/
theorem total_selling_price
  (cost_A cost_B cost_C : ℝ)
  (loss_A gain_B loss_C : ℝ)
  (h_cost_A : cost_A = 1400)
  (h_cost_B : cost_B = 2500)
  (h_cost_C : cost_C = 3200)
  (h_loss_A : loss_A = 0.15)
  (h_gain_B : gain_B = 0.10)
  (h_loss_C : loss_C = 0.05) :
  cost_A * (1 - loss_A) + cost_B * (1 + gain_B) + cost_C * (1 - loss_C) = 6980 :=
by sorry

end NUMINAMATH_CALUDE_total_selling_price_l2422_242239


namespace NUMINAMATH_CALUDE_cubic_function_root_product_l2422_242233

/-- A cubic function with specific properties -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  x₁ : ℝ
  x₂ : ℝ
  root_zero : d = 0
  root_x₁ : a * x₁^3 + b * x₁^2 + c * x₁ + d = 0
  root_x₂ : a * x₂^3 + b * x₂^2 + c * x₂ + d = 0
  extreme_value_1 : (3 * a * ((3 - Real.sqrt 3) / 3)^2 + 2 * b * ((3 - Real.sqrt 3) / 3) + c) = 0
  extreme_value_2 : (3 * a * ((3 + Real.sqrt 3) / 3)^2 + 2 * b * ((3 + Real.sqrt 3) / 3) + c) = 0
  a_nonzero : a ≠ 0

/-- The product of non-zero roots of the cubic function is 2 -/
theorem cubic_function_root_product (f : CubicFunction) : f.x₁ * f.x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_root_product_l2422_242233


namespace NUMINAMATH_CALUDE_pasture_rent_problem_l2422_242224

/-- Represents the rent share of a person -/
structure RentShare where
  oxen : ℕ
  months : ℕ

/-- Calculates the total ox-months for a given rent share -/
def oxMonths (share : RentShare) : ℕ := share.oxen * share.months

/-- The problem statement -/
theorem pasture_rent_problem (a b c : RentShare) (c_rent : ℚ) 
  (h1 : a.oxen = 10 ∧ a.months = 7)
  (h2 : b.oxen = 12 ∧ b.months = 5)
  (h3 : c.oxen = 15 ∧ c.months = 3)
  (h4 : c_rent = 53.99999999999999)
  : ∃ (total_rent : ℚ), total_rent = 210 := by
  sorry

#check pasture_rent_problem

end NUMINAMATH_CALUDE_pasture_rent_problem_l2422_242224


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2422_242227

theorem fraction_equals_zero (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2422_242227


namespace NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_y_is_six_l2422_242272

theorem smallest_integer_y (y : ℤ) : (10 - 5*y < -15) ↔ (y ≥ 6) := by
  sorry

theorem smallest_integer_y_is_six : ∃ (y : ℤ), (10 - 5*y < -15) ∧ (∀ (z : ℤ), (10 - 5*z < -15) → z ≥ y) ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_y_is_six_l2422_242272


namespace NUMINAMATH_CALUDE_problem_statement_l2422_242276

theorem problem_statement (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : (a * Real.sin (π/5) + b * Real.cos (π/5)) / (a * Real.cos (π/5) - b * Real.sin (π/5)) = Real.tan (8*π/15)) : 
  b / a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2422_242276


namespace NUMINAMATH_CALUDE_candy_distribution_bijective_l2422_242265

/-- The candy distribution function -/
def f (n : ℕ) (x : ℕ) : ℕ := (x * (x + 1) / 2) % n

/-- Proposition: The candy distribution function is bijective iff n is a power of 2 -/
theorem candy_distribution_bijective (n : ℕ) (h : n > 0) :
  Function.Bijective (f n) ↔ ∃ k : ℕ, n = 2^k := by sorry

end NUMINAMATH_CALUDE_candy_distribution_bijective_l2422_242265


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l2422_242206

theorem product_of_sums_and_differences : (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * (2 * Real.sqrt 5 - 5 * Real.sqrt 2) = -30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l2422_242206


namespace NUMINAMATH_CALUDE_mixture_capacity_l2422_242280

/-- Represents a vessel containing a mixture of alcohol and water -/
structure Vessel where
  capacity : ℝ
  alcohol_percentage : ℝ

/-- Represents the final mixture -/
structure FinalMixture where
  total_volume : ℝ
  vessel_capacity : ℝ

def mixture_problem (vessel1 vessel2 : Vessel) (final : FinalMixture) : Prop :=
  vessel1.capacity = 2 ∧
  vessel1.alcohol_percentage = 0.35 ∧
  vessel2.capacity = 6 ∧
  vessel2.alcohol_percentage = 0.50 ∧
  final.total_volume = 8 ∧
  final.vessel_capacity = 10 ∧
  vessel1.capacity + vessel2.capacity = final.total_volume

theorem mixture_capacity (vessel1 vessel2 : Vessel) (final : FinalMixture) 
  (h : mixture_problem vessel1 vessel2 final) : 
  final.vessel_capacity = 10 := by
  sorry

#check mixture_capacity

end NUMINAMATH_CALUDE_mixture_capacity_l2422_242280


namespace NUMINAMATH_CALUDE_vector_dot_product_l2422_242246

theorem vector_dot_product (α : ℝ) (b : Fin 2 → ℝ) :
  let a : Fin 2 → ℝ := ![Real.cos α, Real.sin α]
  (a • b = -1) →
  (a • (2 • a - b) = 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l2422_242246


namespace NUMINAMATH_CALUDE_tax_discount_commute_l2422_242263

theorem tax_discount_commute (p t d : ℝ) (h1 : 0 ≤ p) (h2 : 0 ≤ t) (h3 : 0 ≤ d) (h4 : d ≤ 1) :
  p * (1 + t) * (1 - d) = p * (1 - d) * (1 + t) :=
by sorry

#check tax_discount_commute

end NUMINAMATH_CALUDE_tax_discount_commute_l2422_242263


namespace NUMINAMATH_CALUDE_cory_candy_purchase_l2422_242244

/-- The amount of money Cory has initially -/
def cory_money : ℝ := 20

/-- The cost of one pack of candies -/
def candy_pack_cost : ℝ := 49

/-- The number of candy packs Cory wants to buy -/
def num_packs : ℕ := 2

/-- The additional amount of money Cory needs -/
def additional_money_needed : ℝ := num_packs * candy_pack_cost - cory_money

theorem cory_candy_purchase :
  additional_money_needed = 78 := by
  sorry

end NUMINAMATH_CALUDE_cory_candy_purchase_l2422_242244


namespace NUMINAMATH_CALUDE_castle_tour_limit_l2422_242291

structure Castle where
  side_length : ℝ
  num_halls : ℕ
  hall_side_length : ℝ
  has_doors : Bool

def max_visitable_halls (c : Castle) : ℕ :=
  sorry

theorem castle_tour_limit (c : Castle) 
  (h1 : c.side_length = 100)
  (h2 : c.num_halls = 100)
  (h3 : c.hall_side_length = 10)
  (h4 : c.has_doors = true) :
  max_visitable_halls c ≤ 91 :=
sorry

end NUMINAMATH_CALUDE_castle_tour_limit_l2422_242291


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l2422_242208

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the line passing through the focus
def line_through_focus (p : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), p = (focus.1 + t, focus.2 + t)

-- Theorem statement
theorem parabola_intersection_length 
  (A B : PointOnParabola) 
  (h_line_A : line_through_focus (A.x, A.y))
  (h_line_B : line_through_focus (B.x, B.y))
  (h_sum : A.x + B.x = 6) :
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l2422_242208


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2422_242223

/-- Given that (10, -6) is the midpoint of a line segment with one endpoint at (12, 4),
    prove that the sum of coordinates of the other endpoint is -8. -/
theorem midpoint_coordinate_sum :
  ∀ (x y : ℝ),
  (10 : ℝ) = (x + 12) / 2 →
  (-6 : ℝ) = (y + 4) / 2 →
  x + y = -8 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2422_242223


namespace NUMINAMATH_CALUDE_jake_debt_l2422_242273

/-- The amount Jake originally owed given his payment and work details --/
def original_debt (prepaid_amount : ℕ) (hourly_rate : ℕ) (hours_worked : ℕ) : ℕ :=
  prepaid_amount + hourly_rate * hours_worked

/-- Theorem stating that Jake's original debt was $100 --/
theorem jake_debt : original_debt 40 15 4 = 100 := by
  sorry

end NUMINAMATH_CALUDE_jake_debt_l2422_242273


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2422_242287

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = Set.Ioo (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2422_242287


namespace NUMINAMATH_CALUDE_experiment_comparison_l2422_242279

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  total : Nat
  red : Nat
  black : Nat

/-- Represents the result of an experiment -/
structure ExperimentResult where
  expectation : ℚ
  variance : ℚ

/-- Calculates the result of drawing with replacement -/
def drawWithReplacement (bag : BagContents) (draws : Nat) : ExperimentResult :=
  sorry

/-- Calculates the result of drawing without replacement -/
def drawWithoutReplacement (bag : BagContents) (draws : Nat) : ExperimentResult :=
  sorry

theorem experiment_comparison (bag : BagContents) (draws : Nat) :
  let withReplacement := drawWithReplacement bag draws
  let withoutReplacement := drawWithoutReplacement bag draws
  (bag.total = 5 ∧ bag.red = 2 ∧ bag.black = 3 ∧ draws = 2) →
  (withReplacement.expectation = withoutReplacement.expectation ∧
   withReplacement.variance > withoutReplacement.variance) :=
by sorry

end NUMINAMATH_CALUDE_experiment_comparison_l2422_242279


namespace NUMINAMATH_CALUDE_third_triangular_square_l2422_242256

/-- A number that is both triangular and square --/
def TriangularSquare (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * (a + 1) / 2 ∧ n = b * b

/-- The first two triangular square numbers --/
def FirstTwoTriangularSquares : Prop :=
  TriangularSquare 1 ∧ TriangularSquare 36

/-- Checks if a number is the third triangular square number --/
def IsThirdTriangularSquare (n : ℕ) : Prop :=
  TriangularSquare n ∧
  FirstTwoTriangularSquares ∧
  ∀ m : ℕ, m < n → TriangularSquare m → (m = 1 ∨ m = 36)

/-- 1225 is the third triangular square number --/
theorem third_triangular_square :
  IsThirdTriangularSquare 1225 :=
sorry

end NUMINAMATH_CALUDE_third_triangular_square_l2422_242256


namespace NUMINAMATH_CALUDE_millet_exceeds_half_on_wednesday_l2422_242209

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  millet : Real
  otherSeeds : Real

/-- Calculates the next day's feeder state based on the current state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := 0.7 * state.millet + 0.2,
    otherSeeds := 0.1 * state.otherSeeds + 0.3 }

/-- Initial state of the feeder on Monday -/
def initialState : FeederState :=
  { day := 1, millet := 0.2, otherSeeds := 0.3 }

/-- Theorem stating that on Wednesday, the proportion of millet exceeds half of the total seeds -/
theorem millet_exceeds_half_on_wednesday :
  let wednesdayState := nextDay (nextDay initialState)
  wednesdayState.millet > (wednesdayState.millet + wednesdayState.otherSeeds) / 2 :=
by sorry


end NUMINAMATH_CALUDE_millet_exceeds_half_on_wednesday_l2422_242209


namespace NUMINAMATH_CALUDE_abs_neg_six_equals_six_l2422_242254

theorem abs_neg_six_equals_six : abs (-6 : ℤ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_six_equals_six_l2422_242254


namespace NUMINAMATH_CALUDE_integral_roots_problem_l2422_242251

theorem integral_roots_problem (x y z : ℕ) : 
  z^x = y^(2*x) ∧ 
  2^z = 2*(8^x) ∧ 
  x + y + z = 20 →
  x = 5 ∧ y = 4 ∧ z = 16 := by
sorry

end NUMINAMATH_CALUDE_integral_roots_problem_l2422_242251


namespace NUMINAMATH_CALUDE_product_one_to_six_l2422_242249

theorem product_one_to_six : (List.range 6).foldl (· * ·) 1 = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_one_to_six_l2422_242249


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2422_242261

/-- Given two points C(3, 7) and D(8, 10), prove that the sum of the slope and y-intercept
    of the line passing through these points is 29/5 -/
theorem line_slope_intercept_sum (C D : ℝ × ℝ) : 
  C = (3, 7) → D = (8, 10) → 
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := C.2 - m * C.1
  m + b = 29/5 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2422_242261


namespace NUMINAMATH_CALUDE_second_grade_survey_count_l2422_242241

/-- Calculates the number of students to be surveyed from the second grade in a stratified sampling method. -/
theorem second_grade_survey_count
  (total_students : ℕ)
  (grade_ratio : Fin 3 → ℕ)
  (total_surveyed : ℕ)
  (h1 : total_students = 1500)
  (h2 : grade_ratio 0 = 4 ∧ grade_ratio 1 = 5 ∧ grade_ratio 2 = 6)
  (h3 : total_surveyed = 150) :
  (total_surveyed * grade_ratio 1) / (grade_ratio 0 + grade_ratio 1 + grade_ratio 2) = 50 :=
by sorry

end NUMINAMATH_CALUDE_second_grade_survey_count_l2422_242241


namespace NUMINAMATH_CALUDE_x_value_l2422_242264

theorem x_value (x : ℝ) (h : (1 / 4 : ℝ) - (1 / 5 : ℝ) = 5 / x) : x = 100 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2422_242264


namespace NUMINAMATH_CALUDE_point_movement_theorem_point_M_movement_l2422_242242

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Moves a point on the number line -/
def move (p : Point) (distance : ℝ) : Point :=
  ⟨p.value + distance⟩

theorem point_movement_theorem (M N : Point) :
  (M.value = 9) →
  (move (move N (-4)) 6 = M) →
  N.value = 7 :=
sorry

theorem point_M_movement (M : Point) :
  (M.value = 9) →
  (∃ (new_M : Point), (move M 4 = new_M ∨ move M (-4) = new_M) ∧ (new_M.value = 5 ∨ new_M.value = 13)) :=
sorry

end NUMINAMATH_CALUDE_point_movement_theorem_point_M_movement_l2422_242242


namespace NUMINAMATH_CALUDE_all_positive_integers_l2422_242225

def is_valid_set (A : Set ℕ) : Prop :=
  1 ∈ A ∧
  ∃ k : ℕ, k ≠ 1 ∧ k ∈ A ∧
  ∀ m n : ℕ, m ∈ A → n ∈ A → m ≠ n →
    ((m + 1) / (Nat.gcd (m + 1) (n + 1))) ∈ A

theorem all_positive_integers (A : Set ℕ) :
  is_valid_set A → A = {n : ℕ | n > 0} :=
by sorry

end NUMINAMATH_CALUDE_all_positive_integers_l2422_242225


namespace NUMINAMATH_CALUDE_root_difference_squared_l2422_242270

theorem root_difference_squared (a : ℝ) (r s : ℝ) : 
  r^2 - (a+1)*r + a = 0 → 
  s^2 - (a+1)*s + a = 0 → 
  (r-s)^2 = a^2 - 2*a + 1 := by
sorry

end NUMINAMATH_CALUDE_root_difference_squared_l2422_242270


namespace NUMINAMATH_CALUDE_total_heads_l2422_242271

theorem total_heads (hens cows : ℕ) : 
  hens = 28 →
  2 * hens + 4 * cows = 144 →
  hens + cows = 50 := by
sorry

end NUMINAMATH_CALUDE_total_heads_l2422_242271


namespace NUMINAMATH_CALUDE_equation_one_solutions_l2422_242232

theorem equation_one_solutions (x : ℝ) :
  (x - 1)^2 - 5 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l2422_242232


namespace NUMINAMATH_CALUDE_S_is_circle_l2422_242237

-- Define the set of complex numbers satisfying the condition
def S : Set ℂ := {z : ℂ | Complex.abs (z - Complex.I) = Complex.abs (3 + 4 * Complex.I)}

-- Theorem stating that S is a circle
theorem S_is_circle : ∃ (c : ℂ) (r : ℝ), S = {z : ℂ | Complex.abs (z - c) = r} :=
sorry

end NUMINAMATH_CALUDE_S_is_circle_l2422_242237


namespace NUMINAMATH_CALUDE_roots_equal_magnitude_implies_real_ratio_l2422_242284

theorem roots_equal_magnitude_implies_real_ratio 
  (p q : ℂ) 
  (h_q_nonzero : q ≠ 0) 
  (h_roots_equal_magnitude : ∀ z₁ z₂ : ℂ, z₁^2 + p*z₁ + q^2 = 0 → z₂^2 + p*z₂ + q^2 = 0 → Complex.abs z₁ = Complex.abs z₂) :
  ∃ r : ℝ, p / q = r := by sorry

end NUMINAMATH_CALUDE_roots_equal_magnitude_implies_real_ratio_l2422_242284


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2422_242214

theorem complex_number_in_third_quadrant : 
  let i : ℂ := Complex.I
  let z : ℂ := i + 2 * i^2 + 3 * i^3
  (z.re < 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2422_242214


namespace NUMINAMATH_CALUDE_sandy_comic_books_l2422_242222

theorem sandy_comic_books (x : ℕ) : 
  (x / 2 : ℚ) - 3 + 6 = 13 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_sandy_comic_books_l2422_242222


namespace NUMINAMATH_CALUDE_pythagorean_triplets_l2422_242266

theorem pythagorean_triplets :
  ∀ (a b c : ℤ), a^2 + b^2 = c^2 ↔ 
    ∃ (d p q : ℤ), a = 2*d*p*q ∧ b = d*(q^2 - p^2) ∧ c = d*(p^2 + q^2) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triplets_l2422_242266


namespace NUMINAMATH_CALUDE_brick_height_l2422_242236

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: The height of a rectangular prism with given dimensions -/
theorem brick_height (l w sa : ℝ) (hl : l = 8) (hw : w = 4) (hsa : sa = 112) :
  ∃ h : ℝ, surface_area l w h = sa ∧ h = 2 := by
sorry

end NUMINAMATH_CALUDE_brick_height_l2422_242236


namespace NUMINAMATH_CALUDE_square_six_z_minus_five_l2422_242238

theorem square_six_z_minus_five (z : ℝ) (hz : 3 * z^2 + 2 * z = 5 * z + 11) : 
  (6 * z - 5)^2 = 141 := by
  sorry

end NUMINAMATH_CALUDE_square_six_z_minus_five_l2422_242238


namespace NUMINAMATH_CALUDE_luke_laundry_loads_l2422_242296

def total_clothing : ℕ := 47
def first_load : ℕ := 17
def pieces_per_load : ℕ := 6

theorem luke_laundry_loads : 
  (total_clothing - first_load) / pieces_per_load = 5 :=
by sorry

end NUMINAMATH_CALUDE_luke_laundry_loads_l2422_242296


namespace NUMINAMATH_CALUDE_cyclic_iff_arithmetic_progression_l2422_242260

/-- A quadrilateral with sides a, b, d, c (in that order) -/
structure Quadrilateral :=
  (a b d c : ℝ)

/-- The property of sides forming an arithmetic progression -/
def is_arithmetic_progression (q : Quadrilateral) : Prop :=
  ∃ k : ℝ, q.b = q.a + k ∧ q.d = q.a + 2*k ∧ q.c = q.a + 3*k

/-- The property of a quadrilateral being cyclic (inscribable in a circle) -/
def is_cyclic (q : Quadrilateral) : Prop :=
  q.a + q.c = q.b + q.d

/-- Theorem: A quadrilateral is cyclic if and only if its sides form an arithmetic progression -/
theorem cyclic_iff_arithmetic_progression (q : Quadrilateral) :
  is_cyclic q ↔ is_arithmetic_progression q :=
sorry

end NUMINAMATH_CALUDE_cyclic_iff_arithmetic_progression_l2422_242260


namespace NUMINAMATH_CALUDE_central_cell_is_two_l2422_242283

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

end NUMINAMATH_CALUDE_central_cell_is_two_l2422_242283


namespace NUMINAMATH_CALUDE_prime_sum_inequality_l2422_242234

theorem prime_sum_inequality (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r →
  (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r ≥ 1 →
  ({p, q, r} : Set ℕ) = {2, 3, 5} :=
sorry

end NUMINAMATH_CALUDE_prime_sum_inequality_l2422_242234


namespace NUMINAMATH_CALUDE_exists_divisible_by_five_l2422_242257

def T : Set ℤ := {s | ∃ a : ℤ, s = a^2 + (a+1)^2 + (a+2)^2 + (a+3)^2}

theorem exists_divisible_by_five : ∃ s ∈ T, 5 ∣ s := by sorry

end NUMINAMATH_CALUDE_exists_divisible_by_five_l2422_242257


namespace NUMINAMATH_CALUDE_angle_GDA_measure_l2422_242235

-- Define the points
variable (A B C D E F G : Point)

-- Define the shapes
def is_regular_pentagon (C D E : Point) : Prop := sorry

def is_square (A B C D : Point) : Prop := sorry

-- Define the angle measure
def angle_measure (G D A : Point) : ℝ := sorry

-- State the theorem
theorem angle_GDA_measure 
  (h1 : is_regular_pentagon C D E)
  (h2 : is_square A B C D)
  (h3 : is_square D E F G) :
  angle_measure G D A = 72 := by sorry

end NUMINAMATH_CALUDE_angle_GDA_measure_l2422_242235


namespace NUMINAMATH_CALUDE_max_distance_to_line_l2422_242269

noncomputable section

-- Define the curve C
def C : Set (ℝ × ℝ) := {(x, y) | x^2 / 3 + y^2 = 1}

-- Define the line l
def l : Set (ℝ × ℝ) := {(x, y) | x - y - 4 = 0}

-- Define point M
def M : ℝ × ℝ := (-2, 2)

-- Define the midpoint P of MN
def P (N : ℝ × ℝ) : ℝ × ℝ := ((N.1 + M.1) / 2, (N.2 + M.2) / 2)

-- Define the distance function from a point to a line
def dist_to_line (P : ℝ × ℝ) : ℝ :=
  |P.1 - P.2 - 4| / Real.sqrt 2

-- Theorem statement
theorem max_distance_to_line :
  ∃ (max_dist : ℝ), max_dist = 7 * Real.sqrt 2 / 2 ∧
  ∀ (N : ℝ × ℝ), N ∈ C → dist_to_line (P N) ≤ max_dist :=
sorry

end NUMINAMATH_CALUDE_max_distance_to_line_l2422_242269


namespace NUMINAMATH_CALUDE_ordering_of_powers_l2422_242217

theorem ordering_of_powers : 6^8 < 3^15 ∧ 3^15 < 8^10 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_powers_l2422_242217


namespace NUMINAMATH_CALUDE_floor_sqrt_10_l2422_242247

theorem floor_sqrt_10 : ⌊Real.sqrt 10⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_10_l2422_242247


namespace NUMINAMATH_CALUDE_modulo_five_power_difference_l2422_242218

theorem modulo_five_power_difference : (27^1235 - 19^1235) % 5 = 2 := by sorry

end NUMINAMATH_CALUDE_modulo_five_power_difference_l2422_242218


namespace NUMINAMATH_CALUDE_total_books_l2422_242248

theorem total_books (tim_books sam_books : ℕ) 
  (h1 : tim_books = 44) 
  (h2 : sam_books = 52) : 
  tim_books + sam_books = 96 := by
sorry

end NUMINAMATH_CALUDE_total_books_l2422_242248


namespace NUMINAMATH_CALUDE_dance_attendance_l2422_242298

theorem dance_attendance (girls : ℕ) (boys : ℕ) : 
  boys = 2 * girls ∧ 
  boys = (girls - 1) + 8 → 
  boys = 14 := by
sorry

end NUMINAMATH_CALUDE_dance_attendance_l2422_242298


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2422_242294

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (Complex.I + 1)) :
  z.im = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2422_242294


namespace NUMINAMATH_CALUDE_grocery_store_inventory_l2422_242299

theorem grocery_store_inventory (regular_soda diet_soda apples : ℕ) : 
  regular_soda = 79 → 
  diet_soda = 53 → 
  regular_soda - diet_soda = 26 → 
  ¬∃ f : ℕ → ℕ → ℕ, f regular_soda diet_soda = apples :=
by sorry

end NUMINAMATH_CALUDE_grocery_store_inventory_l2422_242299


namespace NUMINAMATH_CALUDE_nine_point_chords_l2422_242282

/-- The number of chords that can be drawn from n points on a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords from 9 points on a circle is 36 -/
theorem nine_point_chords : num_chords 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nine_point_chords_l2422_242282


namespace NUMINAMATH_CALUDE_harmonic_quadrilateral_properties_l2422_242275

-- Define a structure for a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a quadrilateral as a collection of four points
structure Quadrilateral :=
  (A B C D : Point)

-- Define the property of a harmonic quadrilateral
def is_harmonic (q : Quadrilateral) : Prop :=
  ∃ (AB CD AC BD AD BC : ℝ),
    AB * CD = AC * BD ∧ AB * CD = AD * BC

-- Define the concyclic property for four points
def are_concyclic (A B C D : Point) : Prop :=
  ∃ (center : Point) (radius : ℝ),
    (A.x - center.x)^2 + (A.y - center.y)^2 = radius^2 ∧
    (B.x - center.x)^2 + (B.y - center.y)^2 = radius^2 ∧
    (C.x - center.x)^2 + (C.y - center.y)^2 = radius^2 ∧
    (D.x - center.x)^2 + (D.y - center.y)^2 = radius^2

-- State the theorem
theorem harmonic_quadrilateral_properties
  (ABCD : Quadrilateral)
  (A1 B1 C1 D1 : Point)
  (h1 : is_harmonic ABCD)
  (h2 : is_harmonic ⟨A1, ABCD.B, ABCD.C, ABCD.D⟩)
  (h3 : is_harmonic ⟨ABCD.A, B1, ABCD.C, ABCD.D⟩)
  (h4 : is_harmonic ⟨ABCD.A, ABCD.B, C1, ABCD.D⟩)
  (h5 : is_harmonic ⟨ABCD.A, ABCD.B, ABCD.C, D1⟩) :
  are_concyclic ABCD.A ABCD.B C1 D1 ∧ is_harmonic ⟨A1, B1, C1, D1⟩ :=
sorry

end NUMINAMATH_CALUDE_harmonic_quadrilateral_properties_l2422_242275


namespace NUMINAMATH_CALUDE_matrix_power_2023_l2422_242268

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l2422_242268


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2422_242210

/-- The function f(x) = 3x^2 - 15x + 7 attains its minimum value when x = 5/2. -/
theorem min_value_quadratic (x : ℝ) :
  ∀ y : ℝ, 3 * x^2 - 15 * x + 7 ≤ 3 * y^2 - 15 * y + 7 ↔ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2422_242210


namespace NUMINAMATH_CALUDE_quadratic_equation_root_value_l2422_242231

theorem quadratic_equation_root_value (a b : ℝ) : 
  (∀ x, a * x^2 + b * x = 6) → -- The quadratic equation
  (a * 2^2 + b * 2 = 6) →     -- x = 2 is a root
  4 * a + 2 * b = 6 :=        -- The value of 4a + 2b
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_value_l2422_242231


namespace NUMINAMATH_CALUDE_walking_time_proof_l2422_242200

/-- Proves that walking 1.5 km at 5 km/h takes 18 minutes -/
theorem walking_time_proof (speed : ℝ) (distance : ℝ) (time_minutes : ℝ) : 
  speed = 5 → distance = 1.5 → time_minutes = (distance / speed) * 60 → time_minutes = 18 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_proof_l2422_242200


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_sine_l2422_242278

/-- For a hyperbola with eccentricity √10 and transverse axis along the y-axis,
    the sine of the slope angle of its asymptote is √10/10. -/
theorem hyperbola_asymptote_slope_sine (e : ℝ) (h : e = Real.sqrt 10) :
  ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 ∧ Real.sin θ = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_sine_l2422_242278


namespace NUMINAMATH_CALUDE_heart_then_king_probability_l2422_242221

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of hearts in a standard deck -/
def num_hearts : ℕ := 13

/-- The number of kings in a standard deck -/
def num_kings : ℕ := 4

/-- The probability of drawing a heart first and a king second from a standard deck -/
theorem heart_then_king_probability :
  (num_hearts / deck_size) * ((num_kings - 1) / (deck_size - 1)) +
  ((num_hearts - 1) / deck_size) * (num_kings / (deck_size - 1)) =
  1 / deck_size :=
sorry

end NUMINAMATH_CALUDE_heart_then_king_probability_l2422_242221


namespace NUMINAMATH_CALUDE_function_and_extrema_l2422_242205

noncomputable def f (a b c x : ℝ) : ℝ := a * x - b / x + c

theorem function_and_extrema :
  ∀ a b c : ℝ,
  (f a b c 1 = 0) →
  (∀ x : ℝ, x ≠ 0 → HasDerivAt (f a b c) (-x + 3) 2) →
  (∀ x : ℝ, x ≠ 0 → f a b c x = -3 * x - 8 / x + 11) ∧
  (∃ x : ℝ, f a b c x = 11 + 4 * Real.sqrt 6 ∧ IsLocalMin (f a b c) x) ∧
  (∃ x : ℝ, f a b c x = 11 - 4 * Real.sqrt 6 ∧ IsLocalMax (f a b c) x) :=
by sorry

end NUMINAMATH_CALUDE_function_and_extrema_l2422_242205


namespace NUMINAMATH_CALUDE_matrix_multiplication_l2422_242202

theorem matrix_multiplication (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec ![3, -2] = ![4, 1])
  (h2 : N.mulVec ![-4, 6] = ![-2, 0]) :
  N.mulVec ![7, 0] = ![14, 4.2] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_l2422_242202


namespace NUMINAMATH_CALUDE_parallelogram_angle_c_l2422_242285

-- Define a parallelogram structure
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

-- Define angle measure in degrees
def angle_measure (p : Parallelogram) (vertex : Char) : ℝ := sorry

-- State the theorem
theorem parallelogram_angle_c (p : Parallelogram) :
  angle_measure p 'A' + 40 = angle_measure p 'B' →
  angle_measure p 'C' = 70 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_c_l2422_242285


namespace NUMINAMATH_CALUDE_simplify_fraction_l2422_242219

theorem simplify_fraction : (270 / 5400) * 30 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2422_242219


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l2422_242229

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_k_value :
  ∀ k : ℝ,
  let A : Point := ⟨3, 1⟩
  let B : Point := ⟨-2, k⟩
  let C : Point := ⟨8, 11⟩
  collinear A B C → k = -9 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l2422_242229


namespace NUMINAMATH_CALUDE_box_volume_l2422_242297

/-- The volume of a rectangular box formed from a cardboard sheet -/
theorem box_volume (initial_length initial_width corner_side : ℝ) 
  (h1 : initial_length = 13)
  (h2 : initial_width = 9)
  (h3 : corner_side = 2) : 
  (initial_length - 2 * corner_side) * (initial_width - 2 * corner_side) * corner_side = 90 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l2422_242297


namespace NUMINAMATH_CALUDE_nell_card_count_l2422_242295

/-- The number of cards Nell has after receiving cards from Jeff -/
def total_cards (initial_cards given_cards : ℝ) : ℝ :=
  initial_cards + given_cards

/-- Theorem stating that Nell's total cards equal the sum of her initial cards and those given by Jeff -/
theorem nell_card_count (initial_cards given_cards : ℝ) :
  total_cards initial_cards given_cards = initial_cards + given_cards :=
by sorry

end NUMINAMATH_CALUDE_nell_card_count_l2422_242295


namespace NUMINAMATH_CALUDE_wall_thickness_l2422_242293

/-- Calculates the thickness of a wall given its dimensions and the number of bricks used. -/
theorem wall_thickness
  (wall_length : Real)
  (wall_height : Real)
  (brick_length : Real)
  (brick_width : Real)
  (brick_height : Real)
  (num_bricks : Nat)
  (h_wall_length : wall_length = 9)
  (h_wall_height : wall_height = 6)
  (h_brick_length : brick_length = 0.25)
  (h_brick_width : brick_width = 0.1125)
  (h_brick_height : brick_height = 0.06)
  (h_num_bricks : num_bricks = 7200) :
  ∃ (wall_thickness : Real),
    wall_thickness = 0.225 ∧
    wall_length * wall_height * wall_thickness =
      num_bricks * brick_length * brick_width * brick_height :=
by sorry

end NUMINAMATH_CALUDE_wall_thickness_l2422_242293


namespace NUMINAMATH_CALUDE_probability_theorem_l2422_242281

/- Define the number of white and black balls -/
def white_balls : ℕ := 2
def black_balls : ℕ := 3
def total_balls : ℕ := white_balls + black_balls

/- Define the probability of drawing a white ball and a black ball -/
def prob_white : ℚ := white_balls / total_balls
def prob_black : ℚ := black_balls / total_balls

/- Part I: Sampling with replacement -/
def prob_different_colors : ℚ := prob_white * prob_black * 2

/- Part II: Sampling without replacement -/
def prob_zero_white : ℚ := (black_balls / total_balls) * ((black_balls - 1) / (total_balls - 1))
def prob_one_white : ℚ := (black_balls / total_balls) * (white_balls / (total_balls - 1)) + 
                          (white_balls / total_balls) * (black_balls / (total_balls - 1))
def prob_two_white : ℚ := (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))

def expectation : ℚ := 0 * prob_zero_white + 1 * prob_one_white + 2 * prob_two_white
def variance : ℚ := (0 - expectation)^2 * prob_zero_white + 
                    (1 - expectation)^2 * prob_one_white + 
                    (2 - expectation)^2 * prob_two_white

theorem probability_theorem :
  prob_different_colors = 12/25 ∧
  prob_zero_white = 3/10 ∧
  prob_one_white = 3/5 ∧
  prob_two_white = 1/10 ∧
  expectation = 4/5 ∧
  variance = 9/25 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l2422_242281


namespace NUMINAMATH_CALUDE_total_bad_vegetables_l2422_242228

/-- Calculate the total number of bad vegetables picked by Carol and her mom -/
theorem total_bad_vegetables (carol_carrots carol_cucumbers carol_tomatoes : ℕ)
  (mom_carrots mom_cucumbers mom_tomatoes : ℕ)
  (carol_good_carrot_percent carol_good_cucumber_percent carol_good_tomato_percent : ℚ)
  (mom_good_carrot_percent mom_good_cucumber_percent mom_good_tomato_percent : ℚ)
  (h1 : carol_carrots = 29)
  (h2 : carol_cucumbers = 15)
  (h3 : carol_tomatoes = 10)
  (h4 : mom_carrots = 16)
  (h5 : mom_cucumbers = 12)
  (h6 : mom_tomatoes = 14)
  (h7 : carol_good_carrot_percent = 80/100)
  (h8 : carol_good_cucumber_percent = 95/100)
  (h9 : carol_good_tomato_percent = 90/100)
  (h10 : mom_good_carrot_percent = 85/100)
  (h11 : mom_good_cucumber_percent = 70/100)
  (h12 : mom_good_tomato_percent = 75/100) :
  (carol_carrots - ⌊carol_carrots * carol_good_carrot_percent⌋) +
  (carol_cucumbers - ⌊carol_cucumbers * carol_good_cucumber_percent⌋) +
  (carol_tomatoes - ⌊carol_tomatoes * carol_good_tomato_percent⌋) +
  (mom_carrots - ⌊mom_carrots * mom_good_carrot_percent⌋) +
  (mom_cucumbers - ⌊mom_cucumbers * mom_good_cucumber_percent⌋) +
  (mom_tomatoes - ⌊mom_tomatoes * mom_good_tomato_percent⌋) = 19 := by
  sorry


end NUMINAMATH_CALUDE_total_bad_vegetables_l2422_242228


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2422_242226

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 = 20) →
  (a 3 + a 4 = 40) →
  (a 5 + a 6 = 80) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2422_242226


namespace NUMINAMATH_CALUDE_independent_of_b_implies_k_equals_two_l2422_242292

/-- If the algebraic expression ab(5ka-3b)-(ka-b)(3ab-4a²) is independent of b, then k = 2 -/
theorem independent_of_b_implies_k_equals_two (a b k : ℝ) :
  (∀ b, ∃ C, a * b * (5 * k * a - 3 * b) - (k * a - b) * (3 * a * b - 4 * a^2) = C) →
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_independent_of_b_implies_k_equals_two_l2422_242292


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2422_242212

theorem divisibility_implies_equality (a b : ℕ) :
  (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) →
  a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2422_242212


namespace NUMINAMATH_CALUDE_volleyball_club_boys_count_l2422_242216

theorem volleyball_club_boys_count :
  ∀ (total_members boys girls present : ℕ),
  total_members = 30 →
  present = 18 →
  boys + girls = total_members →
  present = boys + girls / 3 →
  boys = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_club_boys_count_l2422_242216


namespace NUMINAMATH_CALUDE_kabadi_player_count_l2422_242240

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 15

/-- The total number of players -/
def total_players : ℕ := 50

/-- The number of players who play kho kho only -/
def kho_kho_only : ℕ := 40

/-- The number of players who play both games -/
def both_games : ℕ := 5

theorem kabadi_player_count :
  kabadi_players = total_players - kho_kho_only + both_games :=
by sorry

end NUMINAMATH_CALUDE_kabadi_player_count_l2422_242240


namespace NUMINAMATH_CALUDE_triangle_with_given_altitudes_is_obtuse_l2422_242250

/-- A triangle with given altitudes --/
structure Triangle where
  alt1 : ℝ
  alt2 : ℝ
  alt3 : ℝ

/-- Definition of an obtuse triangle --/
def isObtuse (t : Triangle) : Prop :=
  ∃ θ : ℝ, θ > Real.pi / 2 ∧ θ < Real.pi ∧
    (Real.cos θ = -(5 : ℝ) / 16)

/-- Theorem: A triangle with altitudes 1/2, 1, and 2/5 is obtuse --/
theorem triangle_with_given_altitudes_is_obtuse :
  let t : Triangle := { alt1 := 1/2, alt2 := 1, alt3 := 2/5 }
  isObtuse t := by
  sorry


end NUMINAMATH_CALUDE_triangle_with_given_altitudes_is_obtuse_l2422_242250


namespace NUMINAMATH_CALUDE_sophomore_latin_probability_l2422_242201

/-- Represents the percentage of students in each class -/
structure ClassDistribution :=
  (freshmen : ℚ)
  (sophomores : ℚ)
  (juniors : ℚ)
  (seniors : ℚ)

/-- Represents the percentage of students taking Latin in each class -/
structure LatinRates :=
  (freshmen : ℚ)
  (sophomores : ℚ)
  (juniors : ℚ)
  (seniors : ℚ)

/-- The probability that a randomly chosen Latin student is a sophomore -/
def sophomoreProbability (dist : ClassDistribution) (rates : LatinRates) : ℚ :=
  (dist.sophomores * rates.sophomores) /
  (dist.freshmen * rates.freshmen + dist.sophomores * rates.sophomores +
   dist.juniors * rates.juniors + dist.seniors * rates.seniors)

theorem sophomore_latin_probability :
  let dist : ClassDistribution := {
    freshmen := 2/5, sophomores := 3/10, juniors := 1/5, seniors := 1/10
  }
  let rates : LatinRates := {
    freshmen := 1, sophomores := 4/5, juniors := 1/2, seniors := 1/5
  }
  sophomoreProbability dist rates = 6/19 := by sorry

end NUMINAMATH_CALUDE_sophomore_latin_probability_l2422_242201


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l2422_242290

theorem cylinder_volume_equality (x : ℝ) (hx : x > 0) : 
  (π * (7 + x)^2 * 5 = π * 7^2 * (5 + 2*x)) → x = 28/5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l2422_242290


namespace NUMINAMATH_CALUDE_penny_splitting_game_result_l2422_242243

/-- Represents the result of the penny splitting game. -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- The penny splitting game. -/
def pennySplittingGame (n : ℕ) : GameResult :=
  sorry

/-- Theorem stating the conditions for each player's victory. -/
theorem penny_splitting_game_result (n : ℕ) (h : n ≥ 3) :
  pennySplittingGame n = 
    if n = 3 ∨ n % 2 = 0 then
      GameResult.FirstPlayerWins
    else
      GameResult.SecondPlayerWins :=
  sorry

end NUMINAMATH_CALUDE_penny_splitting_game_result_l2422_242243


namespace NUMINAMATH_CALUDE_men_joined_correct_l2422_242207

/-- The number of men who joined the camp -/
def men_joined : ℕ := 30

/-- The initial number of men in the camp -/
def initial_men : ℕ := 10

/-- The initial number of days the food would last -/
def initial_days : ℕ := 20

/-- The number of days the food lasts after more men join -/
def final_days : ℕ := 5

/-- The total amount of food in man-days -/
def total_food : ℕ := initial_men * initial_days

theorem men_joined_correct :
  (initial_men + men_joined) * final_days = total_food :=
by sorry

end NUMINAMATH_CALUDE_men_joined_correct_l2422_242207


namespace NUMINAMATH_CALUDE_det_max_value_l2422_242274

open Real Matrix

theorem det_max_value (θ : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 1, 1; 1 + tan θ, 1, 1; 1, 1, 1 + cos θ]
  ∀ φ : ℝ, det A ≤ det (!![1, 1, 1; 1 + tan φ, 1, 1; 1, 1, 1 + cos φ]) :=
by sorry

end NUMINAMATH_CALUDE_det_max_value_l2422_242274


namespace NUMINAMATH_CALUDE_david_average_marks_l2422_242262

def david_marks : List Nat := [96, 95, 82, 97, 95]

theorem david_average_marks :
  (david_marks.sum / david_marks.length : ℚ) = 93 := by sorry

end NUMINAMATH_CALUDE_david_average_marks_l2422_242262


namespace NUMINAMATH_CALUDE_roy_julia_multiple_l2422_242215

-- Define variables for current ages
variable (R J K : ℕ)

-- Define the multiple
variable (M : ℕ)

-- Roy is 6 years older than Julia
def roy_julia_diff : Prop := R = J + 6

-- Roy is half of 6 years older than Kelly
def roy_kelly_diff : Prop := R = K + 3

-- In 4 years, Roy will be some multiple of Julia's age
def future_age_multiple : Prop := R + 4 = M * (J + 4)

-- In 4 years, Roy's age multiplied by Kelly's age would be 108
def future_age_product : Prop := (R + 4) * (K + 4) = 108

theorem roy_julia_multiple
  (h1 : roy_julia_diff R J)
  (h2 : roy_kelly_diff R K)
  (h3 : future_age_multiple R J M)
  (h4 : future_age_product R K) :
  M = 2 := by sorry

end NUMINAMATH_CALUDE_roy_julia_multiple_l2422_242215


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l2422_242289

theorem cos_pi_third_minus_alpha (α : ℝ) (h : Real.sin (π / 6 + α) = 1 / 3) :
  Real.cos (π / 3 - α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l2422_242289


namespace NUMINAMATH_CALUDE_solution_set_for_a_2_find_a_value_l2422_242258

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_for_a_2 :
  {x : ℝ | |x - 2| ≥ 4 - |x - 4|} = {x : ℝ | x ≥ 5 ∨ x ≤ 1} :=
sorry

-- Part 2
theorem find_a_value (a : ℝ) (h : a > 1) :
  ({x : ℝ | |f a (2*x + a) - 2*(f a x)| ≤ 2} = {x : ℝ | 1 ≤ x ∧ x ≤ 2}) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_2_find_a_value_l2422_242258
