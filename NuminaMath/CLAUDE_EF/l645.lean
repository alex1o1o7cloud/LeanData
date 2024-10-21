import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_shared_l645_64567

/-- Proves that the total amount shared among A, B, C, and D is 263.49 dollars -/
theorem total_amount_shared (euro_to_dollar yen_to_dollar b_share : ℝ) 
  (h_euro : euro_to_dollar = 1.18)
  (h_yen : yen_to_dollar = 1.10 / 120)
  (h_b : b_share = 58) : ℝ := by
  have h1 : ∀ x : ℝ, x > 0 → 
    x + x * euro_to_dollar + 0.75 * x + x * 120 * yen_to_dollar = 263.49 := by sorry
  exact 263.49

#check total_amount_shared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_shared_l645_64567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_selling_price_l645_64531

/-- Calculates the selling price of a machine given its purchase details, repair costs, depreciation, and profit margin. -/
def calculate_selling_price (purchase_price : ℝ) (sales_tax_rate : ℝ) (repair_cost_part : ℝ) 
  (repair_cost_software : ℝ) (service_tax_rate : ℝ) (transportation_cost : ℝ) 
  (depreciation_rate : ℝ) (years_owned : ℕ) (profit_rate : ℝ) : ℝ :=
  let initial_cost := purchase_price * (1 + sales_tax_rate)
  let total_repair_cost := (repair_cost_part + repair_cost_software) * (1 + service_tax_rate)
  let total_cost := initial_cost + total_repair_cost + transportation_cost
  let depreciated_value := total_cost * (1 - depreciation_rate) ^ years_owned
  depreciated_value * (1 + profit_rate)

/-- The selling price of the machine is approximately 31049.44 given the specified conditions. -/
theorem machine_selling_price :
  abs (calculate_selling_price 18000 0.10 3000 4000 0.05 1500 0.15 2 0.50 - 31049.44) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_selling_price_l645_64531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l645_64520

/-- A function g(x) with vertical asymptotes at x = 2 and x = -1 -/
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ := (x - 3) / (x^2 + c*x + d)

/-- The sum of c and d is -3 given the conditions on g(x) -/
theorem asymptote_sum (c d : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → g c d x ≠ 0⁻¹) ∧ 
  (∀ x : ℝ, x ≠ -1 → g c d x ≠ 0⁻¹) → 
  c + d = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l645_64520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l645_64503

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C
  sum_angles : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  -- Proposition 1
  (t.A > t.B ∧ t.B > t.C → Real.sin t.A > Real.sin t.B ∧ Real.sin t.B > Real.sin t.C) ∧
  -- Proposition 2
  (Real.cos t.A / t.a = Real.cos t.B / t.b ∧ Real.cos t.B / t.b = Real.cos t.C / t.c → 
    t.A = t.B ∧ t.B = t.C) ∧
  -- Proposition 4
  ((1 + Real.tan t.A) * (1 + Real.tan t.B) = 2 → t.C > π/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l645_64503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l645_64506

/-- Shaded area of Figure A -/
noncomputable def shaded_area_A : ℝ := 9 - Real.pi

/-- Shaded area of Figure B -/
noncomputable def shaded_area_B : ℝ := 9 - Real.pi

/-- Shaded area of Figure C -/
def shaded_area_C : ℝ := 4

/-- Theorem stating that Figure A and B have equal and largest shaded areas -/
theorem largest_shaded_area :
  shaded_area_A = shaded_area_B ∧
  shaded_area_A > shaded_area_C ∧
  shaded_area_B > shaded_area_C :=
by
  constructor
  · -- Prove shaded_area_A = shaded_area_B
    rfl
  constructor
  · -- Prove shaded_area_A > shaded_area_C
    sorry
  · -- Prove shaded_area_B > shaded_area_C
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l645_64506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roberts_bike_ride_time_l645_64583

/-- The time taken for Robert to cover a one-mile stretch of highway on a semicircular path -/
theorem roberts_bike_ride_time : ∃ (time_hours : ℝ), time_hours = π / 10 := by
  -- Define constants
  let highway_length_feet : ℝ := 5280
  let highway_width : ℝ := 40
  let speed_mph : ℝ := 5
  
  -- Calculate intermediate values
  let radius : ℝ := highway_width / 2
  let num_semicircles : ℝ := highway_length_feet / highway_width
  let total_distance_feet : ℝ := num_semicircles * (π * radius)
  let total_distance_miles : ℝ := total_distance_feet / highway_length_feet
  
  -- Calculate time taken
  let time_hours : ℝ := total_distance_miles / speed_mph
  
  -- Prove that the time taken is π/10 hours
  use time_hours
  sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roberts_bike_ride_time_l645_64583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l645_64572

theorem binomial_expansion_constant_term :
  ∀ x : ℝ, x ≠ 0 →
  let expansion := (x^(1/3) - 2/x)^8
  let fifth_term_max := ∀ k : ℕ, k ≠ 5 → (Nat.choose 8 4) ≥ (Nat.choose 8 (k-1))
  let constant_term := (Nat.choose 8 2) * ((-2 : ℝ)^2)
  fifth_term_max → constant_term = 112 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l645_64572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_division_simplification_l645_64563

theorem root_division_simplification :
  ((7 : ℝ) ^ (1/4) / (3 : ℝ) ^ (1/3)) / ((7 : ℝ) ^ (1/2) / (3 : ℝ) ^ (1/6)) = 
  (1 / 7 : ℝ) ^ (1/4) * (1 / 3 : ℝ) ^ (1/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_division_simplification_l645_64563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_share_analysis_l645_64571

/-- Represents the data for market share analysis -/
structure MarketShareData where
  n : ℕ  -- number of months
  sum_x_diff_sq : ℝ  -- ∑(xi - x̄)²
  sum_y_diff_sq : ℝ  -- ∑(yi - ȳ)²
  sum_xy_diff : ℝ  -- ∑(xi - x̄)(yi - ȳ)
  mean_x : ℝ  -- x̄
  mean_y : ℝ  -- ȳ

/-- The main theorem about market share analysis -/
theorem market_share_analysis (data : MarketShareData) 
  (h_n : data.n = 7)
  (h_sum_x_diff_sq : data.sum_x_diff_sq = 28)
  (h_sum_y_diff_sq : data.sum_y_diff_sq = 118)
  (h_sum_xy_diff : data.sum_xy_diff = 56)
  (h_mean_x : data.mean_x = 4)
  (h_mean_y : data.mean_y = 17) :
  let r := data.sum_xy_diff / Real.sqrt (data.sum_x_diff_sq * data.sum_y_diff_sq)
  let b := data.sum_xy_diff / data.sum_x_diff_sq
  let a := data.mean_y - b * data.mean_x
  let y_pred := b * 10 + a
  (abs (r - 0.98) < 0.01) ∧ 
  (b = 2) ∧ 
  (a = 9) ∧ 
  (y_pred = 29) := by
  sorry

#check market_share_analysis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_share_analysis_l645_64571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_even_fifth_power_end_l645_64524

theorem two_digit_even_fifth_power_end (n : ℕ) : 
  (10 ≤ n ∧ n < 100 ∧ n % 2 = 0 ∧ n % 100 = (n^5) % 100) ↔ 
  n ∈ ({32, 24, 76, 68} : Set ℕ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_even_fifth_power_end_l645_64524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_BC_equals_four_cube_root_two_l645_64544

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions of the problem
class ProblemConditions (t : Triangle) where
  A_origin : t.A = (0, 0)
  on_parabola : 
    t.A.2 = parabola t.A.1 ∧
    t.B.2 = parabola t.B.1 ∧
    t.C.2 = parabola t.C.1
  BC_parallel_x : t.B.2 = t.C.2
  area : (1/2) * abs ((t.B.1 - t.A.1) * (t.C.2 - t.A.2) - (t.C.1 - t.A.1) * (t.B.2 - t.A.2)) = 128

-- State the theorem
theorem length_BC_equals_four_cube_root_two 
  (t : Triangle) 
  [pc : ProblemConditions t] : 
  Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) = 4 * (2 : ℝ)^(1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_BC_equals_four_cube_root_two_l645_64544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_processing_rates_l645_64530

/-- Represents the number of parts processed per hour by Xiao Liang -/
def xiao_liang_rate : ℝ := sorry

/-- Represents the number of parts processed per hour by Xiao Ming -/
def xiao_ming_rate : ℝ := sorry

/-- Xiao Ming processes 10 more parts per hour than Xiao Liang -/
axiom rate_difference : xiao_ming_rate = xiao_liang_rate + 10

/-- The time for Xiao Ming to process 150 parts equals the time for Xiao Liang to process 120 parts -/
axiom processing_time_equality : 150 / xiao_ming_rate = 120 / xiao_liang_rate

theorem processing_rates :
  xiao_liang_rate = 40 ∧ xiao_ming_rate = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_processing_rates_l645_64530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_time_theorem_l645_64537

/-- Represents the highway construction scenario -/
structure HighwayConstruction where
  width : ℝ
  speed : ℝ
  totalDistance : ℝ

/-- Calculates the time taken to cover the distance in the given construction scenario -/
noncomputable def timeTaken (h : HighwayConstruction) : ℝ :=
  h.totalDistance / h.speed

/-- Theorem stating the time taken for the specific scenario -/
theorem construction_time_theorem (h : HighwayConstruction) 
  (h_width : h.width = 60)
  (h_speed : h.speed = 4)
  (h_distance : h.totalDistance = 2) :
  timeTaken h = 0.5 := by
  -- Unfold the definition of timeTaken
  unfold timeTaken
  -- Substitute the known values
  rw [h_distance, h_speed]
  -- Perform the division
  norm_num

#check construction_time_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_time_theorem_l645_64537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pairwise_non_overlapping_circles_countable_l645_64517

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define when two circles are non-overlapping
def non_overlapping (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 > (c1.radius + c2.radius)^2

-- Define a set of pairwise non-overlapping circles
def pairwise_non_overlapping (S : Set Circle) : Prop :=
  ∀ c1 c2, c1 ∈ S → c2 ∈ S → c1 ≠ c2 → non_overlapping c1 c2

-- The main theorem
theorem pairwise_non_overlapping_circles_countable (S : Set Circle) 
  (h : pairwise_non_overlapping S) : 
  (Set.Finite S) ∨ (Set.Countable S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pairwise_non_overlapping_circles_countable_l645_64517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_distance_l645_64581

/-- Represents the distance between two points in meters. -/
def Distance := ℝ

/-- Represents the speed of a person in meters per unit time. -/
def Speed := ℝ

/-- Represents a point on a line. -/
structure Point where
  x : ℝ

/-- The setup of the problem with two people moving towards each other. -/
structure TwoPersonMeeting where
  A : Point
  B : Point
  C : Point
  D : Point
  speed_A : Speed
  speed_B : Speed
  distance_BC_when_A_at_C : Distance
  distance_AC_when_B_at_C : Distance

/-- The main theorem to be proved. -/
theorem meeting_point_distance (setup : TwoPersonMeeting) : 
  setup.distance_BC_when_A_at_C = (240 : ℝ) →
  setup.distance_AC_when_B_at_C = (360 : ℝ) →
  setup.C.x - setup.A.x = setup.B.x - setup.C.x →
  setup.D.x - setup.C.x = (144 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_distance_l645_64581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l645_64598

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (2^x + 3)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) → -1 < y ∧ y < 1/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l645_64598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lineup_count_l645_64579

/-- The number of ways to arrange 5 people in a line with restrictions -/
def lineupWays : ℕ := 72

/-- The number of people in the lineup -/
def totalPeople : ℕ := 5

/-- The number of possible positions for the youngest person -/
def youngestPositions : ℕ := totalPeople - 2

/-- The number of ways to arrange the remaining people -/
def remainingArrangements : ℕ := Nat.factorial (totalPeople - 1)

theorem lineup_count :
  lineupWays = youngestPositions * remainingArrangements :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lineup_count_l645_64579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l645_64505

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*(2*a - 1)*x + 8) / Real.log (1/2)

-- Define the property of f being decreasing on [a, +∞)
def is_decreasing_from (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → f y < f x

-- State the theorem
theorem f_decreasing_implies_a_range (a : ℝ) :
  is_decreasing_from (f a) a → a ∈ Set.Ioo (-4/3) 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l645_64505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_angles_polygon_l645_64552

/-- Sum of interior angles of an n-sided polygon -/
def sum_interior_angles (n : ℕ) : ℝ := sorry

/-- The sum of interior angles of an n-sided polygon is (n-2) * 180° -/
theorem sum_interior_angles_polygon (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2) * 180 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_angles_polygon_l645_64552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_angle_relation_l645_64501

/-- A regular triangular pyramid inscribed in a sphere -/
structure RegularPyramid where
  base : Set (Fin 3 → ℝ)  -- Representing the base as a set of 3D points
  apex : Fin 3 → ℝ        -- Representing the apex as a 3D point
  inscribed_in_sphere : Prop

/-- The angle between the lateral face and the base of a pyramid -/
noncomputable def lateral_base_angle (p : RegularPyramid) : ℝ := sorry

/-- Theorem stating the relationship between the lateral-base angles of two pyramids -/
theorem pyramid_angle_relation 
  (P Q : RegularPyramid)
  (h1 : P.base = Q.base)
  (h2 : P.inscribed_in_sphere)
  (h3 : Q.inscribed_in_sphere)
  (h4 : lateral_base_angle P = Real.pi / 4) :
  Real.tan (lateral_base_angle Q) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_angle_relation_l645_64501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_3_l645_64510

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then 2 * x
  else x^2 / 2

-- Theorem statement
theorem f_inverse_of_3 (a : ℝ) (h : f a = 3) : a = 3/2 ∨ a = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_3_l645_64510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_equations_l645_64502

/-- Given a triangle ABC with vertex A at (0,1), altitude on AB with equation x+2y-4=0,
    and median on AC with equation 2x+y-3=0, prove that the equations of the lines
    containing sides AB, BC, and AC are as stated. -/
theorem triangle_side_equations (B C : ℝ × ℝ) : 
  let A : ℝ × ℝ := (0, 1)
  let altitude_AB : ℝ → ℝ → Prop := λ x y => x + 2*y - 4 = 0
  let median_AC : ℝ → ℝ → Prop := λ x y => 2*x + y - 3 = 0
  let line_AB : ℝ → ℝ → Prop := λ x y => 2*x - y + 1 = 0
  let line_BC : ℝ → ℝ → Prop := λ x y => 2*x + 3*y - 7 = 0
  let line_AC : ℝ → ℝ → Prop := λ x y => y = 1
  (∀ x y, altitude_AB x y ↔ (x + 2*y - 4 = 0)) →
  (∀ x y, median_AC x y ↔ (2*x + y - 3 = 0)) →
  (line_AB A.1 A.2) ∧
  (line_AB B.1 B.2) ∧
  (line_BC B.1 B.2) ∧
  (line_BC C.1 C.2) ∧
  (line_AC A.1 A.2) ∧
  (line_AC C.1 C.2)
:= by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_equations_l645_64502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l645_64592

theorem binomial_expansion_coefficient (n : ℕ) : 
  (∃ k : ℕ, (n.choose 2 = n.choose 6) ∧ 
   (∀ r : ℕ, (n.choose r) = (n.choose (n - r)))) →
  (n.choose 5) = 56 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l645_64592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_average_median_l645_64549

theorem absolute_difference_average_median (a b : ℝ) (h : 1 < a ∧ a < b) :
  let numbers := [1, a + 1, 2 * a + b, a + b + 1]
  let average := (numbers.sum) / 4
  let median := (numbers.get! 1 + numbers.get! 2) / 2
  |average - median| = 1/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_average_median_l645_64549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oddSumRepresentations_eq_fib_l645_64569

/-- The number of ways to represent a positive integer as the sum of positive odd integers -/
def oddSumRepresentations : ℕ+ → ℕ := sorry

/-- The n-th Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: The number of ways to represent n as the sum of positive odd integers
    (considering different orders as distinct) is equal to the n-th Fibonacci number -/
theorem oddSumRepresentations_eq_fib (n : ℕ+) :
  oddSumRepresentations n = fib n.val := by
  sorry

#check oddSumRepresentations_eq_fib

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oddSumRepresentations_eq_fib_l645_64569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_n_on_C_l645_64582

/-- Sequence of points A_n(a_n, b_n) -/
def A : ℕ+ → ℝ × ℝ := sorry

/-- Initial condition for A_1 -/
axiom A_1 : A 1 = (0, 1)

/-- Recurrence relation for a_n -/
axiom a_recurrence (n : ℕ+) : (A (n + 1)).1 = 1 + (A n).1 / ((A n).1^2 + (A n).2^2)

/-- Recurrence relation for b_n -/
axiom b_recurrence (n : ℕ+) : (A (n + 1)).2 = -(A n).2 / ((A n).1^2 + (A n).2^2)

/-- Circle C with center (1/2, 0) and radius √5/2 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1/2)^2 + p.2^2 = 5/4}

/-- Theorem: For all n ≥ 4, A_n lies on circle C -/
theorem A_n_on_C (n : ℕ+) (h : n ≥ 4) : A n ∈ C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_n_on_C_l645_64582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sine_cosine_l645_64529

theorem max_value_sine_cosine :
  ∃ (θ : ℝ), 0 < θ ∧ θ < 3 * Real.pi / 2 ∧
  (fun θ => Real.sin (θ / 3) * (1 + Real.cos (2 * θ / 3))) θ = 8 * Real.sqrt 5 / 25 ∧
  ∀ (φ : ℝ), 0 < φ ∧ φ < 3 * Real.pi / 2 →
    (fun θ => Real.sin (θ / 3) * (1 + Real.cos (2 * θ / 3))) φ ≤ 8 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sine_cosine_l645_64529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_curve_min_distance_to_line_max_distance_to_line_l645_64584

-- Define the curve C
noncomputable def C (x y : ℝ) : Prop := x^2 + y^2/3 = 1

-- Define the line l
def l (x y : ℝ) : Prop := x + y - 8 = 0

-- Define the distance function from a point to the line l
noncomputable def dist_to_l (x y : ℝ) : ℝ := |x + y - 8| / Real.sqrt 2

theorem point_outside_curve : C 4 4 → False := by sorry

theorem min_distance_to_line : 
  ∃ (x y : ℝ), C x y ∧ ∀ (x' y' : ℝ), C x' y' → dist_to_l x y ≤ dist_to_l x' y' ∧ dist_to_l x y = 3 * Real.sqrt 2 := by sorry

theorem max_distance_to_line : 
  ∃ (x y : ℝ), C x y ∧ ∀ (x' y' : ℝ), C x' y' → dist_to_l x' y' ≤ dist_to_l x y ∧ dist_to_l x y = 5 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_curve_min_distance_to_line_max_distance_to_line_l645_64584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_triangle_problem_l645_64538

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi/6) * Real.cos x

-- Theorem for the range of f
theorem range_of_f :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi/2 →
  ∃ y : ℝ, f x = y ∧ 0 ≤ y ∧ y ≤ 3/2 :=
by
  sorry

-- Theorem for the triangle problem
theorem triangle_problem (A B : ℝ) (b c : ℝ) :
  0 < A ∧ A < Real.pi/2 →  -- A is acute
  f A = 1 →               -- f(A) = 1
  b = 2 →                 -- side b = 2
  c = 3 →                 -- side c = 3
  Real.cos (A - B) = 5 * Real.sqrt 7 / 14 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_triangle_problem_l645_64538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_assignment_l645_64594

-- Define the girls and colors
inductive Girl : Type
  | Katya | Olya | Liza | Rita

inductive Color : Type
  | Pink | Green | Yellow | Blue

-- Define the dress assignment function
def dress : Girl → Color := sorry

-- Define the standing order
def next_to : Girl → Girl → Prop := sorry

-- Conditions
axiom not_pink_or_blue_katya : dress Girl.Katya ≠ Color.Pink ∧ dress Girl.Katya ≠ Color.Blue

axiom green_between_liza_and_yellow : 
  ∃ (g : Girl), dress g = Color.Green ∧ 
    ((next_to g Girl.Liza ∧ ∃ (y : Girl), next_to g y ∧ dress y = Color.Yellow) ∨
     (next_to Girl.Liza g ∧ ∃ (y : Girl), next_to y g ∧ dress y = Color.Yellow))

axiom not_green_or_blue_rita : dress Girl.Rita ≠ Color.Green ∧ dress Girl.Rita ≠ Color.Blue

axiom olya_between_rita_and_pink : 
  ∃ (p : Girl), dress p = Color.Pink ∧
    ((next_to Girl.Olya Girl.Rita ∧ next_to Girl.Olya p) ∨
     (next_to Girl.Rita Girl.Olya ∧ next_to p Girl.Olya))

-- Theorem statement
theorem dress_assignment : 
  dress Girl.Katya = Color.Green ∧
  dress Girl.Olya = Color.Blue ∧
  dress Girl.Liza = Color.Pink ∧
  dress Girl.Rita = Color.Yellow := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_assignment_l645_64594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_profit_percentage_approx_l645_64504

/-- Represents the financial structure of a restaurant --/
structure Restaurant where
  initial_cost : ℝ
  initial_profit_percentage : ℝ
  food_cost_percentage : ℝ
  labor_cost_percentage : ℝ
  overhead_cost_percentage : ℝ
  food_cost_increase : ℝ
  labor_cost_increase : ℝ
  overhead_cost_decrease : ℝ
  selling_price_increase : ℝ

/-- Calculates the new profit percentage after changes --/
noncomputable def new_profit_percentage (r : Restaurant) : ℝ :=
  let new_food_cost := r.food_cost_percentage * r.initial_cost * (1 + r.food_cost_increase)
  let new_labor_cost := r.labor_cost_percentage * r.initial_cost * (1 + r.labor_cost_increase)
  let new_overhead_cost := r.overhead_cost_percentage * r.initial_cost * (1 - r.overhead_cost_decrease)
  let new_total_cost := new_food_cost + new_labor_cost + new_overhead_cost
  let initial_selling_price := r.initial_cost * (1 + r.initial_profit_percentage / 100)
  let new_selling_price := initial_selling_price * (1 + r.selling_price_increase)
  let new_profit := new_selling_price - new_total_cost
  (new_profit / new_selling_price) * 100

/-- Theorem stating that the new profit percentage is approximately 62.07% --/
theorem new_profit_percentage_approx (r : Restaurant) 
  (h1 : r.initial_cost = 100)
  (h2 : r.initial_profit_percentage = 170)
  (h3 : r.food_cost_percentage = 0.65)
  (h4 : r.labor_cost_percentage = 0.25)
  (h5 : r.overhead_cost_percentage = 0.1)
  (h6 : r.food_cost_increase = 0.14)
  (h7 : r.labor_cost_increase = 0.05)
  (h8 : r.overhead_cost_decrease = 0.08)
  (h9 : r.selling_price_increase = 0.07) :
  ∃ ε > 0, |new_profit_percentage r - 62.07| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_profit_percentage_approx_l645_64504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_triangle_area_l645_64534

-- Define the rhombus
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  angle : ℝ

-- Define the function to calculate the area of a triangle in the rhombus
noncomputable def triangle_area (r : Rhombus) : ℝ :=
  (1/4) * r.diagonal1 * r.diagonal2 * Real.sin r.angle

-- Theorem statement
theorem rhombus_triangle_area (r : Rhombus) 
  (h1 : r.diagonal1 = 15) 
  (h2 : r.diagonal2 = 20) : 
  triangle_area r = 37.5 * Real.sin r.angle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_triangle_area_l645_64534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_x_plus_200_integer_count_l645_64539

theorem sqrt_x_plus_200_integer_count : 
  (Finset.range 14).card = 14 ∧
  ∀ n : ℕ, n ∈ Finset.range 14 → 
    ∃ x : ℤ, x < 0 ∧ (x + 200 : ℤ) = (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_x_plus_200_integer_count_l645_64539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_box_is_twenty_cents_l645_64526

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the storage problem -/
structure StorageProblem where
  boxDim : BoxDimensions
  totalSpace : ℝ
  totalMonthlyCost : ℝ

/-- Calculates the cost per box per month -/
noncomputable def costPerBoxPerMonth (p : StorageProblem) : ℝ :=
  p.totalMonthlyCost / (p.totalSpace / boxVolume p.boxDim)

/-- Theorem: The cost per box per month is $0.20 -/
theorem cost_per_box_is_twenty_cents (p : StorageProblem) 
  (h1 : p.boxDim = ⟨15, 12, 10⟩) 
  (h2 : p.totalSpace = 1080000)
  (h3 : p.totalMonthlyCost = 120) : 
  costPerBoxPerMonth p = 0.2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_box_is_twenty_cents_l645_64526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_power_product_l645_64596

theorem no_prime_power_product (p q : ℕ) (m n : ℕ) : 
  Nat.Prime p → Nat.Prime q → 
  (2^p - p^2) * (2^q - q^2) ≠ p^m * q^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_power_product_l645_64596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l645_64593

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the conditions
variable (a b : ℝ)

axiom a_gt_b : a > b
axiom b_gt_zero : b > 0
axiom minor_axis_length : 2*b = 2
axiom max_triangle_area : ∃ (x y : ℝ), ellipse_C x y a b ∧ Real.sqrt 3 = (a^2 - b^2) * b

-- Define the theorem
theorem ellipse_properties :
  (∀ x y, ellipse_C x y a b ↔ x^2/4 + y^2 = 1) ∧
  (∃ k m : ℝ, ∀ x y, 
    ellipse_C x y a b → 
    (x = k*y + m ∧ x ≠ a) → 
    (∃ x1 y1 x2 y2 : ℝ, 
      ellipse_C x1 y1 a b ∧ 
      ellipse_C x2 y2 a b ∧ 
      x1 = k*y1 + m ∧ 
      x2 = k*y2 + m ∧ 
      (x1 - a)*(x2 - a) + y1*y2 = 0) → 
    m = 6/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l645_64593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l645_64574

theorem intersection_count (m n : ℕ) :
  let total_intersections := m * n * (m - 1) * (n - 1) / 4
  total_intersections = m * n * (m - 1) * (n - 1) / 4 :=
by
  -- Define total_intersections
  let total_intersections := m * n * (m - 1) * (n - 1) / 4
  -- The proof would go here
  sorry

#check intersection_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l645_64574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l645_64516

/-- The function f(x) = x + 9/(2x-2) for x > 1 -/
noncomputable def f (x : ℝ) : ℝ := x + 9 / (2 * x - 2)

/-- The domain condition x > 1 -/
def domain (x : ℝ) : Prop := x > 1

theorem f_minimum :
  (∀ x, domain x → f x ≥ 3 * Real.sqrt 2 + 1) ∧
  (∃ x, domain x ∧ f x = 3 * Real.sqrt 2 + 1) := by
  sorry

#check f_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l645_64516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_theorem_l645_64514

noncomputable section

/-- The function f(x) = (2x+3)/(2x-4) -/
def f (x : ℝ) : ℝ := (2*x + 3) / (2*x - 4)

/-- Point P -/
def P : ℝ × ℝ := (2, 1)

/-- Origin O -/
def O : ℝ × ℝ := (0, 0)

/-- Vector OP -/
def OP : ℝ × ℝ := P

theorem dot_product_theorem (A B : ℝ × ℝ) 
  (h1 : ∃ (l : Set (ℝ × ℝ)), P ∈ l ∧ A ∈ l ∧ B ∈ l)  -- Line l passes through P, A, and B
  (h2 : f A.1 = A.2 ∧ f B.1 = B.2) :  -- A and B are on the graph of f
  (A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2) • OP = 10 := by 
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_theorem_l645_64514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_max_value_and_t_l645_64588

-- Define the solution set
def solution_set (x : ℝ) : Prop := 0 < x ∧ x < 8

-- Define the inequality
def inequality (x a b : ℝ) : Prop := |x + a| < b

theorem inequality_solution :
  (∃ a b : ℝ, ∀ x : ℝ, inequality x a b ↔ solution_set x) →
  (∃ a b : ℝ, a = -4 ∧ b = 4 ∧
    ∀ x : ℝ, inequality x a b ↔ solution_set x) :=
by sorry

noncomputable def expression (a b t : ℝ) : ℝ := Real.sqrt (a * t + 16) + Real.sqrt (b * t)

theorem max_value_and_t (a b : ℝ) (h : a = -4 ∧ b = 4) :
  ∃ max_val t : ℝ, max_val = 8 ∧ t = 2 ∧
    ∀ t' : ℝ, expression a b t' ≤ max_val ∧
    (expression a b t' = max_val ↔ t' = t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_max_value_and_t_l645_64588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_volume_after_two_hours_original_balloon_volume_l645_64513

/-- A balloon that increases its volume by two-fifths every hour underwater -/
structure Balloon where
  volume : ℝ

/-- The volume of the balloon after one hour underwater -/
noncomputable def volume_after_one_hour (b : Balloon) : ℝ :=
  b.volume * (1 + 2/5)

/-- The volume of the balloon after two hours underwater -/
noncomputable def volume_after_two_hours (b : Balloon) : ℝ :=
  volume_after_one_hour b * (1 + 2/5)

/-- Theorem stating that a balloon with volume 500 cm³ will have volume 980 cm³ after 2 hours underwater -/
theorem balloon_volume_after_two_hours (b : Balloon) :
  b.volume = 500 → volume_after_two_hours b = 980 := by
  sorry

/-- Theorem stating that if a balloon has volume 980 cm³ after 2 hours underwater, its original volume was 500 cm³ -/
theorem original_balloon_volume (b : Balloon) :
  volume_after_two_hours b = 980 → b.volume = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_volume_after_two_hours_original_balloon_volume_l645_64513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l645_64540

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x^2 + 1)

-- State the theorem
theorem f_monotone_increasing (a : ℝ) (h : a > 0) :
  StrictMonoOn (f a) (Set.Ioo (-1) 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l645_64540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_time_approx_15_minutes_l645_64560

-- Define the given constants
noncomputable def boat_speed : ℝ := 16
noncomputable def upstream_time_minutes : ℝ := 20
noncomputable def current_speed : ℝ := 2.28571428571

-- Define the function to calculate downstream time
noncomputable def downstream_time (boat_speed current_speed upstream_time_minutes : ℝ) : ℝ :=
  let upstream_time_hours := upstream_time_minutes / 60
  let upstream_speed := boat_speed - current_speed
  let downstream_speed := boat_speed + current_speed
  let distance := upstream_speed * upstream_time_hours
  (distance / downstream_speed) * 60

-- Theorem statement
theorem downstream_time_approx_15_minutes :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |downstream_time boat_speed current_speed upstream_time_minutes - 15| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_time_approx_15_minutes_l645_64560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paddle_pricing_and_optimal_purchase_l645_64580

-- Define the prices of paddles and balls
variable (x y : ℝ)  -- Price of straight and horizontal paddles
def ball_price : ℝ := 2

-- Define the conditions
def condition1 (x y : ℝ) : Prop := 20 * (x + 10 * ball_price) + 15 * (y + 10 * ball_price) = 9000
def condition2 (x y : ℝ) : Prop := 10 * (y + 10 * ball_price) = 5 * (x + 10 * ball_price) + 1600

-- Define the cost function
def cost (x y m : ℝ) : ℝ := m * (x + 10 * ball_price) + (40 - m) * (y + 10 * ball_price)

-- Define the constraint on the number of straight paddles
def constraint (m : ℝ) : Prop := m ≤ 3 * (40 - m)

-- Theorem statement
theorem paddle_pricing_and_optimal_purchase :
  ∃ x y : ℝ, condition1 x y ∧ condition2 x y ∧
  (x = 220 ∧ y = 260) ∧
  (∃ m : ℝ, m = 30 ∧ constraint m ∧ 
   (∀ n : ℝ, constraint n → cost x y m ≤ cost x y n) ∧
   cost x y m = 10000) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paddle_pricing_and_optimal_purchase_l645_64580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l645_64547

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem min_dot_product (a b : E) 
  (h1 : ‖a‖ = 3)
  (h2 : ‖a - 2 • b‖ ≤ 1) :
  ∃ (m : ℝ), m = -1/8 ∧ ∀ (x : ℝ), x = inner a b → m ≤ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l645_64547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_theorem_l645_64528

/-- The perpendicular bisector of a line segment -/
def perpendicular_bisector (C D : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

/-- The half-plane bounded by a line that contains a point -/
def half_plane (l : Set (EuclideanSpace ℝ (Fin 2))) (P : EuclideanSpace ℝ (Fin 2)) : 
  Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

/-- The set of points on a line segment between two points -/
def line_segment (A B : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

theorem perpendicular_bisector_theorem 
  (C D A B : EuclideanSpace ℝ (Fin 2)) :
  let l := perpendicular_bisector C D
  let h := half_plane l C
  A ∈ h → B ∈ h → 
  ∀ M, M ∈ line_segment A B → dist M C < dist M D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_theorem_l645_64528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_days_l645_64562

/-- Represents the job completion scenario -/
structure JobCompletion where
  totalDays : ℕ
  initialWorkers : ℕ
  initialWorkDays : ℕ
  initialWorkDone : ℚ
  firedWorkers : ℕ

/-- Calculates the total days to complete the job -/
def calculateTotalDays (job : JobCompletion) : ℕ :=
  job.initialWorkDays + 
    ((1 - job.initialWorkDone) / 
      (job.initialWorkDone / job.initialWorkDays * 
        (job.initialWorkers - job.firedWorkers : ℚ))).ceil.toNat

/-- Theorem stating that the job will be completed in 28 days -/
theorem job_completion_days (job : JobCompletion) 
  (h1 : job.totalDays = 100)
  (h2 : job.initialWorkers = 10)
  (h3 : job.initialWorkDays = 20)
  (h4 : job.initialWorkDone = 1/4)
  (h5 : job.firedWorkers = 2) :
  calculateTotalDays job = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_days_l645_64562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_of_18_with_9_and_0_l645_64522

/-- A function that checks if a positive integer consists only of digits 9 and 0 -/
def has_only_nine_and_zero (n : ℕ) : Prop := sorry

/-- The largest positive multiple of 18 consisting only of digits 9 and 0 -/
def m : ℕ := sorry

theorem largest_multiple_of_18_with_9_and_0 :
  (∀ k : ℕ, k > m → ¬(has_only_nine_and_zero k ∧ 18 ∣ k)) ∧
  has_only_nine_and_zero m ∧
  18 ∣ m ∧
  m / 18 = 500 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_of_18_with_9_and_0_l645_64522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_roots_for_specific_a_l645_64533

-- Define the polynomial function
def f (a : ℤ) (x : ℤ) : ℤ :=
  x^4 - 16*x^3 + (81-2*a)*x^2 + (16*a-142)*x + a^2 - 21*a + 68

-- Theorem statement
theorem integer_roots_for_specific_a :
  ∃ (a : ℤ), a = -4 ∧ 
  (∀ x : ℤ, x ∈ ({2, 3, 4, 7} : Set ℤ) → f a x = 0) ∧
  (∀ x : ℤ, f a x = 0 → x ∈ ({2, 3, 4, 7} : Set ℤ)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_roots_for_specific_a_l645_64533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_PQR_ABC_l645_64586

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points D, E, F on the sides of the triangle
noncomputable def D : ℝ × ℝ := (3/4 * B.1 + 1/4 * C.1, 3/4 * B.2 + 1/4 * C.2)
noncomputable def E : ℝ × ℝ := (3/4 * A.1 + 1/4 * C.1, 3/4 * A.2 + 1/4 * C.2)
noncomputable def F : ℝ × ℝ := (1/3 * A.1 + 2/3 * B.1, 1/3 * A.2 + 2/3 * B.2)

-- Define the intersection points P, Q, R
noncomputable def P : ℝ × ℝ := sorry
noncomputable def Q : ℝ × ℝ := sorry
noncomputable def R : ℝ × ℝ := sorry

-- Define the area function
noncomputable def area (p q r : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_PQR_ABC : area P Q R / area A B C = 1/27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_PQR_ABC_l645_64586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_condition_l645_64519

theorem quadratic_roots_condition (x₁ x₂ p q : ℝ) : 
  x₁^2 + x₂^2 = 1 → 
  p = -(x₁ + x₂) → 
  q = x₁ * x₂ → 
  p^2 - 2*q = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_condition_l645_64519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_difference_square_product_l645_64546

theorem prime_difference_square_product (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) :
  let a := ((p + 1) / 2) ^ 2
  let b := ((p - 1) / 2) ^ 2
  0 < a ∧ 0 < b ∧ (a - b = p) ∧ ∃ k, a * b = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_difference_square_product_l645_64546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_interval_division_l645_64578

theorem coprime_interval_division (p q : ℕ) (h_coprime : Nat.Coprime p q) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ p + q - 2 →
  ∃! x : ℚ, (x = (↑i : ℚ) / p ∨ x = (↑j : ℚ) / q) ∧
             (↑k : ℚ) / (↑(p + q) : ℚ) < x ∧ x < (↑(k + 1) : ℚ) / (↑(p + q) : ℚ) ∧
             ((∃ i : ℕ, i < p ∧ x = (↑i : ℚ) / p) ∨
              (∃ j : ℕ, j < q ∧ x = (↑j : ℚ) / q)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_interval_division_l645_64578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l645_64536

theorem trig_identity (α : ℝ) (h1 : Real.cos α = 3/5) (h2 : α ∈ Set.Ioo (-π/2) 0) :
  Real.sin (2 * α) / (1 - Real.cos (2 * α)) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l645_64536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_and_sum_product_l645_64545

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x^2)

theorem f_inequality_and_sum_product (a b c : ℝ) : 
  (∀ a b : ℝ, a ≠ b → |f a - f b| < |a - b|) ∧
  (a + b + c = f (2 * Real.sqrt 2) → a + b + c ≥ a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_and_sum_product_l645_64545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_midpoint_to_directrix_l645_64597

/-- The distance between the midpoint of AB and the directrix of parabola E -/
theorem distance_midpoint_to_directrix :
  ∀ (M : ℝ × ℝ) (l : Set (ℝ × ℝ)) (E : Set (ℝ × ℝ)) (p : ℝ) (A B : ℝ × ℝ),
  M = (Real.sqrt 2/2, -Real.sqrt 2/2) →
  (M.1^2 + M.2^2 = 1) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ y + Real.sqrt 2/2 = x - Real.sqrt 2/2) →
  (∃ (f : ℝ), f = Real.sqrt 2 ∧ (∀ (x y : ℝ), (x, y) ∈ E ↔ y^2 = 2*f*x)) →
  p > 0 →
  A ∈ l ∧ A ∈ E →
  B ∈ l ∧ B ∈ E →
  A ≠ B →
  let midpoint := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  abs (midpoint.2 + p/2) = 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_midpoint_to_directrix_l645_64597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l645_64587

-- Define the curve C in polar coordinates
noncomputable def curve_C (ρ θ : ℝ) : Prop :=
  ρ - 4 * Real.cos θ + 3 * ρ * Real.sin θ ^ 2 = 0

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + Real.sqrt 3 / 2 * t, 1 / 2 * t)

-- Define the curve C' after transformation
def curve_C' (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + y ^ 2 = 4

-- Define the point M
def point_M : ℝ × ℝ := (1, 0)

-- State the theorem
theorem intersection_distance_sum :
  ∃ (t₁ t₂ : ℝ),
    let (x₁, y₁) := line_l t₁
    let (x₂, y₂) := line_l t₂
    curve_C' x₁ y₁ ∧ 
    curve_C' x₂ y₂ ∧
    (t₁ ≠ t₂) ∧
    Real.sqrt ((x₁ - 1) ^ 2 + y₁ ^ 2) + Real.sqrt ((x₂ - 1) ^ 2 + y₂ ^ 2) = Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l645_64587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandths_place_of_three_sixteenths_l645_64573

theorem thousandths_place_of_three_sixteenths (f : ℚ) (d : ℕ) : 
  f = 3 / 16 →
  d = (((f.num * 10^3) / f.den).toNat % 10) →
  d = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandths_place_of_three_sixteenths_l645_64573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l645_64512

-- Define the curve C
noncomputable def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + y - 3| / Real.sqrt 2

-- Statement of the theorem
theorem max_distance_curve_to_line :
  ∃ (x₀ y₀ : ℝ),
    curve_C x₀ y₀ ∧
    (∀ (x y : ℝ), curve_C x y → distance_to_line x y ≤ distance_to_line x₀ y₀) ∧
    x₀ = -4 * Real.sqrt 5 / 5 ∧
    y₀ = -Real.sqrt 5 / 5 ∧
    distance_to_line x₀ y₀ = (Real.sqrt 5 + 3) / Real.sqrt 2 := by
  sorry

#check max_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l645_64512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_store_problem_l645_64532

/-- The number of ways to choose pets for four people from a store with given pet counts. -/
def pet_store_combinations (puppies kittens hamsters birds : ℕ) : ℕ :=
  puppies * kittens * hamsters * birds * 24

/-- Theorem stating the number of ways to choose pets for four people from a specific store. -/
theorem pet_store_problem :
  pet_store_combinations 20 10 12 5 = 288000 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_store_problem_l645_64532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_theorem_l645_64521

/-- Triangle ABC with centroid G -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ
  is_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Sum of distances from centroid to vertices -/
noncomputable def d₁ (t : Triangle) : ℝ :=
  distance t.G t.A + distance t.G t.B + distance t.G t.C

/-- Sum of side lengths of the triangle -/
noncomputable def d₂ (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- The main theorem: d₁ = (1/3) * d₂ for any triangle -/
theorem centroid_distance_theorem (t : Triangle) : d₁ t = (1/3) * d₂ t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_theorem_l645_64521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l645_64555

theorem cosine_identity (a : ℝ) :
  Real.cos (75 * π / 180 - a) = 1/3 →
  Real.cos (30 * π / 180 + 2*a) = 7/9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l645_64555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_oxygen_time_to_normal_l645_64543

/-- The blood oxygen saturation model -/
noncomputable def blood_oxygen_saturation (S₀ K t : ℝ) : ℝ := S₀ * Real.exp (K * t)

/-- The theorem statement -/
theorem additional_oxygen_time_to_normal (S₀ K : ℝ) 
  (h₁ : S₀ = 60)
  (h₂ : blood_oxygen_saturation S₀ K 1 = 70)
  (h₃ : K = Real.log (70 / 60)) :
  ∃ t : ℝ, t ≥ 0 ∧ blood_oxygen_saturation S₀ K (1 + t) ≥ 95 ∧ 
  ∀ t' : ℝ, t' < t → blood_oxygen_saturation S₀ K (1 + t') < 95 ∧
  t = 1.875 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_oxygen_time_to_normal_l645_64543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_l645_64509

/-- A random variable following a binomial distribution B(10, p) -/
noncomputable def ξ : ℕ → ℝ := sorry

/-- The parameter p of the binomial distribution -/
noncomputable def p : ℝ := sorry

/-- The expected value of ξ is 4 -/
axiom expected_value : ∑' k, k * ξ k = 4

/-- The probability mass function of ξ -/
noncomputable def prob_mass (k : ℕ) : ℝ := sorry

theorem binomial_probability :
  prob_mass 2 = (Nat.choose 10 2 : ℝ) * (2/5)^2 * (3/5)^8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_l645_64509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l645_64566

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log (1/4))^2 - (Real.log x / Real.log (1/4)) + 5

-- Define the domain
def domain : Set ℝ := Set.Icc 2 4

-- State the theorem
theorem f_extrema :
  ∃ (min max : ℝ),
    (∀ x ∈ domain, f x ≥ min) ∧
    (∃ x ∈ domain, f x = min) ∧
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    min = 23/4 ∧ max = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l645_64566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l645_64525

def is_valid (n : ℕ) : Prop :=
  n > 0 ∧ ¬(2 ∣ n) ∧ ¬(3 ∣ n) ∧
  ∀ a b : ℕ, (2^a : ℤ) - (3^b : ℤ) ≠ n ∧ (3^b : ℤ) - (2^a : ℤ) ≠ n

theorem smallest_valid_n : 
  (∀ k < 35, ¬(is_valid k)) ∧ is_valid 35 := by
  sorry

#check smallest_valid_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l645_64525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_condition_l645_64599

-- Define the function f as noncomputable
noncomputable def f (m : ℝ) : ℝ := (-1 + Real.sqrt (4 * m - 3)) / 2

-- State the theorem
theorem f_satisfies_condition : ∀ t : ℝ, t ≥ 0 → f (t^2 + t + 1) = t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_condition_l645_64599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_a_l645_64500

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*x else 1/x

theorem possible_values_of_a :
  ∀ a : ℝ, f 1 + f a = -2 ↔ a = -1 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_a_l645_64500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_l645_64541

-- Define a set A with only two subsets
def A : Set ℝ := sorry

-- Define the property that if x is in A, then (x+3)/(x-1) is also in A
def A_closed (x : ℝ) : x ∈ A → (x + 3) / (x - 1) ∈ A := sorry

-- Define that A has only two subsets
def A_two_subsets : ∀ (B : Set ℝ), B ⊆ A → (B = ∅ ∨ B = A) := sorry

-- Theorem statement
theorem x_value : ∀ x : ℝ, x ∈ A → (x = 3 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_l645_64541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_dice_sums_l645_64527

def dice_faces : Set ℤ := {-5, -3, -1, 2, 4, 6}

theorem modified_dice_sums :
  (∀ x ∈ dice_faces, ∀ y ∈ dice_faces, x + y ≠ 7) ∧
  (∃ x ∈ dice_faces, ∃ y ∈ dice_faces, x + y = 3) ∧
  (∃ x ∈ dice_faces, ∃ y ∈ dice_faces, x + y = 4) ∧
  (∃ x ∈ dice_faces, ∃ y ∈ dice_faces, x + y = 5) ∧
  (∃ x ∈ dice_faces, ∃ y ∈ dice_faces, x + y = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_dice_sums_l645_64527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l645_64507

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | -1 < x ∧ x < 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l645_64507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_count_l645_64558

theorem distinct_prime_factors_count : ∃ (S : Finset Nat), 
  (∀ p ∈ S, Nat.Prime p) ∧ 
  (S.prod id = 101 * 103 * 105 * 107) ∧ 
  S.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_count_l645_64558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_covered_area_of_overlapping_squares_l645_64559

/-- Two congruent squares with side length 12, where one square's corner
    is positioned at the center of the other square, cover an area of 252 square units. -/
theorem covered_area_of_overlapping_squares : ℝ := by
  let square_side_length : ℝ := 12
  let single_square_area : ℝ := square_side_length ^ 2
  let total_area_without_overlap : ℝ := 2 * single_square_area
  let overlap_side_length : ℝ := square_side_length / 2
  let overlap_area : ℝ := overlap_side_length ^ 2
  let covered_area : ℝ := total_area_without_overlap - overlap_area
  have h : covered_area = 252 := by sorry
  exact covered_area


end NUMINAMATH_CALUDE_ERRORFEEDBACK_covered_area_of_overlapping_squares_l645_64559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_odd_after_removal_l645_64575

/-- Represents a standard 6-sided die --/
structure StandardDie where
  faces : Fin 6 → ℕ
  total_dots : ℕ
  valid_faces : ∀ i, faces i ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)
  sum_dots : (Finset.univ.sum faces) = total_dots

/-- The probability of selecting a dot from a face with n dots --/
def prob_select_dot (d : StandardDie) (n : ℕ) : ℚ :=
  n / d.total_dots

/-- The probability of removing exactly one dot from a face with n dots --/
def prob_remove_one_dot (d : StandardDie) (n : ℕ) : ℚ :=
  if n ≥ 2 then
    2 * (prob_select_dot d n) * ((d.total_dots - 1) - n) / (d.total_dots - 1)
  else
    0

/-- The probability of removing zero or two dots from a face with n dots --/
def prob_remove_zero_or_two_dots (d : StandardDie) (n : ℕ) : ℚ :=
  if n ≥ 2 then
    (1 - prob_select_dot d n) * (1 - n / (d.total_dots - 1)) + 
    (prob_select_dot d n) * ((n - 1) / (d.total_dots - 1))
  else
    1 - prob_select_dot d n

/-- The theorem to be proved --/
theorem prob_odd_after_removal (d : StandardDie) :
  d.total_dots = 21 →
  (1/6 : ℚ) * (
    prob_remove_zero_or_two_dots d 1 +
    prob_remove_one_dot d 2 +
    prob_remove_zero_or_two_dots d 3 +
    prob_remove_one_dot d 4 +
    prob_remove_zero_or_two_dots d 5 +
    prob_remove_one_dot d 6
  ) = 137/252 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_odd_after_removal_l645_64575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ozone_effect_significant_ozone_experiment_results_l645_64564

-- Define the total number of mice and group sizes
def total_mice : Nat := 40
def group_size : Nat := 20

-- Define the median
noncomputable def median : ℝ := 23.4

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) Nat :=
  ![![6, 14],
    ![14, 6]]

-- Define the K^2 statistic formula
noncomputable def k_squared (a b c d : Nat) : ℝ :=
  (total_mice * (a * d - b * c)^2 : ℝ) / 
  ((a + b) * (c + d) * (a + c) * (b + d) : ℝ)

-- Define the critical value for 95% confidence
noncomputable def critical_value : ℝ := 3.841

-- Theorem statement
theorem ozone_effect_significant : 
  let a := contingency_table 0 0
  let b := contingency_table 0 1
  let c := contingency_table 1 0
  let d := contingency_table 1 1
  k_squared a b c d > critical_value := by
  sorry

-- Main theorem combining all parts
theorem ozone_experiment_results :
  ∃ (m : ℝ), m = median ∧
  contingency_table = ![![6, 14], ![14, 6]] ∧
  k_squared (contingency_table 0 0) (contingency_table 0 1) 
            (contingency_table 1 0) (contingency_table 1 1) > critical_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ozone_effect_significant_ozone_experiment_results_l645_64564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_tulips_percentage_l645_64561

theorem garden_tulips_percentage (F : ℚ) (h : F > 0) : 
  let white_flowers := (4/5 : ℚ) * F
  let yellow_flowers := F - white_flowers
  let white_tulips := (1/2 : ℚ) * white_flowers
  let yellow_daisies := (2/3 : ℚ) * yellow_flowers
  let yellow_tulips := yellow_flowers - yellow_daisies
  let total_tulips := white_tulips + yellow_tulips
  (total_tulips / F) = (7/15 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_tulips_percentage_l645_64561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l645_64511

open Real

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * x + 2 * Real.cos x

-- State the theorem
theorem f_monotonic_decreasing :
  ∀ x ∈ Set.Ioo (π / 3) (2 * π / 3),
    x ∈ Set.Ioo 0 π →
    ∀ y ∈ Set.Ioo (π / 3) (2 * π / 3),
      x < y → f y < f x :=
by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l645_64511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_points_are_perpendicular_and_reflection_l645_64590

/-- A line segment in a plane --/
structure LineSegment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- The set of points that form an annulus of area t when rotating AB around them --/
noncomputable def AnnulusPoints (AB : LineSegment) (t : ℝ) : Set (ℝ × ℝ) :=
  {X | ∃ (r₁ r₂ : ℝ), r₁ < r₂ ∧ Real.pi * (r₂^2 - r₁^2) = t ∧ 
       (∃ (θ : ℝ), X.1 + r₁ * Real.cos θ = AB.start.1 ∧ X.2 + r₁ * Real.sin θ = AB.start.2) ∧
       (∃ (θ : ℝ), X.1 + r₂ * Real.cos θ = AB.endpoint.1 ∧ X.2 + r₂ * Real.sin θ = AB.endpoint.2)}

/-- The perpendicular and its reflection across the bisector of AB --/
def PerpendicularAndReflection (AB : LineSegment) : Set (ℝ × ℝ) :=
  {X | ∃ (d : ℝ), 
       ((X.1 - AB.start.1) * (AB.endpoint.1 - AB.start.1) + (X.2 - AB.start.2) * (AB.endpoint.2 - AB.start.2) = 0 ∧
        (X.1 - AB.start.1)^2 + (X.2 - AB.start.2)^2 = d^2) ∨
       ((X.1 - ((AB.start.1 + AB.endpoint.1) / 2))^2 + (X.2 - ((AB.start.2 + AB.endpoint.2) / 2))^2 = 
        ((AB.start.1 + AB.endpoint.1) / 2 - AB.start.1)^2 + ((AB.start.2 + AB.endpoint.2) / 2 - AB.start.2)^2)}

theorem annulus_points_are_perpendicular_and_reflection 
  (AB : LineSegment) 
  (h_unit_length : (AB.endpoint.1 - AB.start.1)^2 + (AB.endpoint.2 - AB.start.2)^2 = 1) 
  (t : ℝ) :
  AnnulusPoints AB t = PerpendicularAndReflection AB :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_points_are_perpendicular_and_reflection_l645_64590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l645_64565

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 169

-- Define the line
def my_line (k x y : ℝ) : Prop := k * x - y - 4 * k + 5 = 0

-- Theorem statement
theorem circle_line_intersection :
  ∀ k : ℝ,
  (∃ x y : ℝ, my_line k x y ∧ x = 4 ∧ y = 5) ∧
  (∃ k₀ : ℝ, k₀ = -3/4 ∧
    ∀ x y : ℝ, my_circle x y ∧ my_line k₀ x y →
      ∀ x' y' : ℝ, my_circle x' y' ∧ my_line k₀ x' y' →
        (x - 1) * (x' - x) + (y - 1) * (y' - y) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l645_64565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l645_64550

theorem vector_subtraction (p q : Fin 3 → ℝ) (hp : p = ![(-2 : ℝ), 3, 4]) (hq : q = ![(1 : ℝ), -2, 5]) :
  p - 5 • q = ![(-7 : ℝ), 13, -21] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l645_64550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fixed_point_l645_64576

-- Define the parabola C
noncomputable def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point A on the parabola
def point_A : ℝ × ℝ := (1, 2)

-- Define a general point on the parabola
noncomputable def point_on_parabola (y : ℝ) : ℝ × ℝ := (y^2/4, y)

-- Define the perpendicularity condition
def perpendicular (A P Q : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (Q.1 - A.1) + (P.2 - A.2) * (Q.2 - A.2) = 0

-- The main theorem
theorem parabola_fixed_point :
  ∀ (y₁ y₂ : ℝ),
  y₁ ≠ y₂ →
  parabola_C (point_on_parabola y₁).1 (point_on_parabola y₁).2 →
  parabola_C (point_on_parabola y₂).1 (point_on_parabola y₂).2 →
  perpendicular point_A (point_on_parabola y₁) (point_on_parabola y₂) →
  ∃ (t : ℝ), (5 : ℝ) = t * (point_on_parabola y₁).2 + (1 - t) * (point_on_parabola y₂).2 ∧
             (-2 : ℝ) = t * (point_on_parabola y₁).1 + (1 - t) * (point_on_parabola y₂).1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fixed_point_l645_64576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_with_rightmost_digit_one_l645_64542

def a (n : ℕ) : ℕ := Nat.factorial (n + 6) / Nat.factorial (n - 1)

def rightmostNonZeroDigit (m : ℕ) : ℕ :=
  m % 10

theorem smallest_k_with_rightmost_digit_one :
  ∃ k : ℕ, k = 1 ∧ 
    (∀ n : ℕ, 0 < n → n < k → rightmostNonZeroDigit (a n) ≠ 1) ∧
    rightmostNonZeroDigit (a k) = 1 := by
  sorry

#eval rightmostNonZeroDigit (a 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_with_rightmost_digit_one_l645_64542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_produce_450_gizmos_l645_64589

/-- Represents the production rate of gadgets per worker per hour -/
noncomputable def gadget_rate : ℝ := 240 / (80 * 1)

/-- Represents the production rate of gizmos per worker per hour -/
noncomputable def gizmo_rate : ℝ := 160 / (80 * 1)

/-- Represents the number of gizmos produced by 40 workers in 3 hours -/
def gizmos_40_3 : ℝ := 270

/-- Represents the number of gadgets produced by 100 workers in 2 hours -/
def gadgets_100_2 : ℝ := 500

/-- The theorem states that 100 workers produce 450 gizmos in 2 hours -/
theorem workers_produce_450_gizmos : 
  100 * gizmo_rate * 2 = 450 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_produce_450_gizmos_l645_64589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l645_64515

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the vectors m and n
noncomputable def m (t : Triangle) : Fin 2 → Real
  | 0 => Real.cos (t.A - t.B)
  | 1 => Real.sin (t.A - t.B)

noncomputable def n (t : Triangle) : Fin 2 → Real
  | 0 => Real.cos t.B
  | 1 => -Real.sin t.B

-- Define the dot product
def dot_product (v w : Fin 2 → Real) : Real :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 4 * Real.sqrt 2)
  (h2 : t.b = 5)
  (h3 : dot_product (m t) (n t) = -3/5) : 
  Real.sin t.A = 4/5 ∧ 
  t.B = Real.pi/4 ∧ 
  t.b * Real.cos t.B = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l645_64515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simulation_probability_l645_64556

/-- Represents a group of three shots -/
def ShotGroup := List Nat

/-- Checks if a digit represents a score -/
def isScore (n : Nat) : Bool :=
  n ∈ [1, 2, 3, 4]

/-- Counts the number of scores in a shot group -/
def countScores (group : ShotGroup) : Nat :=
  group.filter isScore |>.length

/-- The list of shot groups from the simulation -/
def simulationData : List ShotGroup := [
  [9, 0, 7], [9, 6, 6], [1, 9, 1], [9, 2, 5], [2, 7, 1],
  [9, 3, 2], [8, 1, 2], [4, 5, 8], [5, 6, 9], [6, 8, 3],
  [4, 3, 1], [2, 5, 7], [3, 9, 3], [0, 2, 7], [5, 5, 6],
  [4, 8, 8], [7, 3, 0], [1, 1, 3], [5, 3, 7], [9, 8, 9]
]

theorem simulation_probability : 
  (simulationData.filter (λ g => countScores g = 1)).length / simulationData.length = 2 / 5 := by
  sorry

#eval (simulationData.filter (λ g => countScores g = 1)).length
#eval simulationData.length
#eval (simulationData.filter (λ g => countScores g = 1)).length / simulationData.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simulation_probability_l645_64556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_of_extrema_l645_64523

noncomputable def f (x : ℝ) := 2 * Real.sin (Real.pi / 2 * x + Real.pi / 5)

theorem min_difference_of_extrema (x₁ x₂ : ℝ) : 
  (∀ x : ℝ, f x₁ ≤ f x ∧ f x ≤ f x₂) → 
  ∃ d : ℝ, d = 2 ∧ ∀ y₁ y₂ : ℝ, (∀ y : ℝ, f y₁ ≤ f y ∧ f y ≤ f y₂) → |y₁ - y₂| ≥ d :=
by
  sorry

#check min_difference_of_extrema

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_of_extrema_l645_64523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_values_max_a_value_l645_64551

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x^2 - 1

-- Theorem 1: f has no extreme values in ℝ
theorem no_extreme_values : ∀ x : ℝ, ∃ y : ℝ, f y > f x :=
sorry

-- Theorem 2: Maximum value of a for which f(x) ≥ ax holds for all x ≥ 0
theorem max_a_value : (∀ x : ℝ, x ≥ 0 → f x ≥ (Real.exp 1 - 2) * x) ∧ 
  ∀ ε > 0, ∃ x : ℝ, x ≥ 0 ∧ f x < (Real.exp 1 - 2 + ε) * x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_values_max_a_value_l645_64551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kyler_wins_three_l645_64570

/-- Represents a player in the chess tournament -/
inductive Player
  | Peter
  | Emma
  | Kyler

/-- Represents the game results for a player -/
structure GameResults where
  wins : Nat
  losses : Nat
  total_games : Nat

/-- The chess tournament with given conditions -/
def chess_tournament : Player → GameResults
  | Player.Peter => ⟨5, 3, 8⟩
  | Player.Emma => ⟨4, 4, 8⟩
  | Player.Kyler => ⟨3, 2, 5⟩  -- We now explicitly state Kyler's wins

theorem kyler_wins_three :
  (chess_tournament Player.Kyler).wins = 3 := by
  -- The proof is now trivial since we explicitly defined Kyler's wins
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kyler_wins_three_l645_64570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l645_64508

-- Define the system of inequalities
def system (x y : ℝ) : Prop :=
  x + y ≤ 3 ∧ 2 * x + y ≥ 2 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the quadrilateral region
def quadrilateral_region : Set (ℝ × ℝ) :=
  {p | system p.1 p.2}

-- Define the longest side of the quadrilateral
noncomputable def longest_side (region : Set (ℝ × ℝ)) : ℝ :=
  Real.sqrt 18

-- Theorem statement
theorem longest_side_length :
  longest_side quadrilateral_region = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l645_64508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_value_l645_64518

theorem max_n_value (n : ℕ) (a : ℕ → ℕ) : 
  (∀ i j, i ≠ j → a i ≠ a j) →
  (a 1 = 1) →
  (a n = 2000) →
  (∀ i : ℕ, 2 ≤ i → i ≤ n → (a i - a (i-1) = 3 ∨ a i - a (i-1) = 5)) →
  n ≤ 1996 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_value_l645_64518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_both_curves_l645_64557

-- Define the two curves
noncomputable def curve1 (x : ℝ) : ℝ := Real.exp x
noncomputable def curve2 (x : ℝ) : ℝ := -x^2 / 4

-- Define the proposed tangent line
def tangent_line (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem tangent_to_both_curves :
  ∃ (t : ℝ),
    (curve1 t = tangent_line t ∧ 
     curve2 t = tangent_line t) ∧
    (∀ x : ℝ, x ≠ t → 
      curve1 x ≠ tangent_line x ∧ 
      curve2 x ≠ tangent_line x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_both_curves_l645_64557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equivalence_l645_64548

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x - Real.pi / 6)

theorem cos_shift_equivalence (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ (axis zero_point : ℝ), |axis - zero_point| = Real.pi / 4 ∧ 
    (∀ x, f ω x = f ω (2 * axis - x)) ∧ 
    (f ω zero_point = 0 ∨ f ω (zero_point + Real.pi / (2 * ω)) = 0)) :
  ∀ x, f ω x = Real.cos (2 * (x - Real.pi / 12)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equivalence_l645_64548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l645_64591

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- First line equation: 3x + 4y - 6 = 0 -/
def line1 (x y : ℝ) : Prop := 3*x + 4*y - 6 = 0

/-- Second line equation: 6x + 8y + 3 = 0 -/
def line2 (x y : ℝ) : Prop := 6*x + 8*y + 3 = 0

theorem distance_between_given_lines :
  distance_between_parallel_lines 6 8 (-12) 3 = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l645_64591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_chalk_probability_l645_64554

/-- Represents a box of chalk -/
structure ChalkBox where
  white : ℕ
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- The left box of chalk -/
def leftBox : ChalkBox :=
  { white := 20, yellow := 5, red := 0, blue := 0 }

/-- The right box of chalk -/
def rightBox : ChalkBox :=
  { white := 0, yellow := 6, red := 5, blue := 4 }

/-- The probability of picking from the left box -/
def leftProb : ℝ := 0.4

/-- The probability of picking from the right box -/
def rightProb : ℝ := 0.6

/-- The probability of picking a yellow chalk from a given box -/
noncomputable def yellowProbFromBox (box : ChalkBox) : ℝ :=
  (box.yellow : ℝ) / (box.white + box.yellow + box.red + box.blue : ℝ)

/-- Theorem: The probability of picking a yellow chalk is 0.32 -/
theorem yellow_chalk_probability :
  leftProb * yellowProbFromBox leftBox + rightProb * yellowProbFromBox rightBox = 0.32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_chalk_probability_l645_64554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_with_y_axis_foci_l645_64595

theorem ellipse_with_y_axis_foci (x y : ℝ) :
  let a := Real.sin (Real.sqrt 2) - Real.sin (Real.sqrt 3)
  let b := Real.cos (Real.sqrt 2) - Real.cos (Real.sqrt 3)
  (Real.sqrt 2 + Real.sqrt 3 > π) →
  (x^2 / a + y^2 / b = 1) →
  ∃ (c : ℝ), c > 0 ∧ 
    ∀ (u v : ℝ), x^2 + y^2 = u^2 + v^2 ∧ u^2 / a + v^2 / b = 1 →
      (y - c)^2 + x^2 + (y + c)^2 + x^2 = 4 * Real.sqrt (u^2 + v^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_with_y_axis_foci_l645_64595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l645_64568

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x + 1/(2^x - 1)

-- State the theorem
theorem f_minimum_value :
  (∀ x : ℝ, x > 0 → f x ≥ 3) ∧ (∃ x : ℝ, x > 0 ∧ f x = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l645_64568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l645_64577

/-- Given a hyperbola C: y²/a² - x²/b² = 1 (a > b > 0), if one of its asymptotes
    is tangent to y = 1 + ln x + ln 2, then its eccentricity is √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∃ m n : ℝ, n = (a / b) * m ∧ n = 1 + Real.log m + Real.log 2 ∧
   (∀ x : ℝ, x > 0 → (a / b) ≥ 1 / x)) →
  Real.sqrt (1 + (a / b)^2) = Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l645_64577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_line_through_origin_l645_64535

-- Define Line2 as a structure since it's not part of the standard library
structure Line2 where
  slope : Real
  yIntercept : Real

-- Define a function to check if a line passes through a point
def passesThrough (l : Line2) (x y : Real) : Prop :=
  y = l.slope * x + l.yIntercept

theorem slope_range_for_line_through_origin (θ : Real) (l : Line2) :
  (60 * π / 180 ≤ θ ∧ θ ≤ 135 * π / 180) →
  passesThrough l 0 0 →
  l.slope = Real.tan θ →
  l.slope ∈ Set.Iic (-1) ∪ Set.Ici (Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_line_through_origin_l645_64535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_squared_distances_l645_64553

/-- A square with center at the origin and diagonal length 2 -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_square : A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧ D.1^2 + D.2^2 = 1

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  abs (l.a * p.1 + l.b * p.2 + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Sum of squared distances from square vertices to their projections on a line -/
noncomputable def sumSquaredDistances (s : Square) (l : Line) : ℝ :=
  (distanceToLine s.A l)^2 + (distanceToLine s.B l)^2 +
  (distanceToLine s.C l)^2 + (distanceToLine s.D l)^2

/-- The main theorem -/
theorem constant_sum_squared_distances (s : Square) (l : Line) (k : ℝ)
    (h1 : distanceToLine (0, 0) l = k)
    (h2 : k > 1) :
    sumSquaredDistances s l = 4 * k^2 + 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_squared_distances_l645_64553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_equals_one_l645_64585

/-- The maximum distance from any point on a circle to a line --/
noncomputable def max_distance (a : ℝ) : ℝ := a + 2

/-- The line l in parametric form --/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t, Real.sqrt 3 * t + 2)

/-- The circle C in parametric form --/
noncomputable def circle_C (a θ : ℝ) : ℝ × ℝ := (a * Real.cos θ, a * Real.sin θ)

theorem circle_radius_equals_one (a : ℝ) :
  (a > 0) → (max_distance a = 3) → (a = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_equals_one_l645_64585
