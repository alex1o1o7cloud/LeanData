import Mathlib

namespace brothers_additional_lambs_l3757_375729

/-- The number of lambs Merry takes care of -/
def merrys_lambs : ℕ := 10

/-- The total number of lambs -/
def total_lambs : ℕ := 23

/-- The number of lambs Merry's brother takes care of -/
def brothers_lambs : ℕ := total_lambs - merrys_lambs

/-- The additional number of lambs Merry's brother takes care of compared to Merry -/
def additional_lambs : ℕ := brothers_lambs - merrys_lambs

theorem brothers_additional_lambs :
  additional_lambs = 3 ∧ brothers_lambs > merrys_lambs := by
  sorry

end brothers_additional_lambs_l3757_375729


namespace inequality_properties_l3757_375779

theorem inequality_properties (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) : 
  (a + b < a * b) ∧ (a * b < b^2) := by
  sorry

end inequality_properties_l3757_375779


namespace sqrt_sum_equals_six_l3757_375792

theorem sqrt_sum_equals_six : 
  Real.sqrt (11 + 6 * Real.sqrt 2) + Real.sqrt (11 - 6 * Real.sqrt 2) = 6 := by
  sorry

end sqrt_sum_equals_six_l3757_375792


namespace regular_polygon_60_properties_l3757_375707

/-- A regular polygon with 60 sides -/
structure RegularPolygon60 where
  -- No additional fields needed as the number of sides is fixed

/-- The number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The measure of an exterior angle in a regular polygon -/
def exterior_angle (n : ℕ) : ℚ := 360 / n

theorem regular_polygon_60_properties (p : RegularPolygon60) :
  (num_diagonals 60 = 1710) ∧ (exterior_angle 60 = 6) := by
  sorry

end regular_polygon_60_properties_l3757_375707


namespace power_sum_difference_l3757_375730

theorem power_sum_difference : 4^1 + 3^2 - 2^3 + 1^4 = 6 := by
  sorry

end power_sum_difference_l3757_375730


namespace mans_rowing_speed_in_still_water_l3757_375705

/-- Proves that a man's rowing speed in still water is 25 km/hr given the conditions of downstream speed and time. -/
theorem mans_rowing_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 3) -- Current speed in km/hr
  (h2 : distance = 90) -- Distance in meters
  (h3 : time = 17.998560115190784) -- Time in seconds
  : ∃ (still_water_speed : ℝ), still_water_speed = 25 := by
  sorry


end mans_rowing_speed_in_still_water_l3757_375705


namespace equation_solution_l3757_375740

theorem equation_solution : ∃ (x y : ℝ), x + y + x*y = 4 ∧ 3*x*y = 4 ∧ x = 2/3 := by
  sorry

end equation_solution_l3757_375740


namespace quadratic_equation_roots_l3757_375727

theorem quadratic_equation_roots (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁^2 + 4*r₁ = 0 ∧ r₂^2 + 4*r₂ = 0) :=
by
  sorry


end quadratic_equation_roots_l3757_375727


namespace exists_polygon_with_equal_area_division_l3757_375709

/-- A polygon in the plane --/
structure Polygon where
  vertices : Set (ℝ × ℝ)
  is_closed : ∀ (p : ℝ × ℝ), p ∈ vertices → ∃ (q : ℝ × ℝ), q ∈ vertices ∧ q ≠ p

/-- A point is on the boundary of a polygon --/
def OnBoundary (p : ℝ × ℝ) (poly : Polygon) : Prop :=
  p ∈ poly.vertices

/-- A line divides a polygon into two parts --/
def DividesPolygon (l : Set (ℝ × ℝ)) (poly : Polygon) : Prop :=
  ∃ (A B : Set (ℝ × ℝ)), A ∪ B = poly.vertices ∧ A ∩ B ⊆ l

/-- The area of a set of points in the plane --/
def Area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- A line divides a polygon into two equal parts --/
def DividesEquallyByArea (l : Set (ℝ × ℝ)) (poly : Polygon) : Prop :=
  ∃ (A B : Set (ℝ × ℝ)), 
    DividesPolygon l poly ∧
    Area A = Area B

/-- Main theorem: There exists a polygon and a point on its boundary such that 
    any line passing through this point divides the area of the polygon into two equal parts --/
theorem exists_polygon_with_equal_area_division :
  ∃ (poly : Polygon) (p : ℝ × ℝ), 
    OnBoundary p poly ∧
    ∀ (l : Set (ℝ × ℝ)), p ∈ l → DividesEquallyByArea l poly := by
  sorry

end exists_polygon_with_equal_area_division_l3757_375709


namespace original_cost_was_75_l3757_375768

/-- Represents the selling price of a key chain -/
def selling_price : ℝ := 100

/-- Represents the original profit percentage -/
def original_profit_percentage : ℝ := 0.25

/-- Represents the new profit percentage -/
def new_profit_percentage : ℝ := 0.50

/-- Represents the new manufacturing cost -/
def new_manufacturing_cost : ℝ := 50

/-- Calculates the original manufacturing cost based on the given conditions -/
def original_manufacturing_cost : ℝ := selling_price * (1 - original_profit_percentage)

/-- Theorem stating that the original manufacturing cost was $75 -/
theorem original_cost_was_75 : 
  original_manufacturing_cost = 75 := by sorry

end original_cost_was_75_l3757_375768


namespace power_of_two_equality_l3757_375763

theorem power_of_two_equality (n : ℕ) : 2^n = 2 * 16^2 * 4^3 → n = 15 := by
  sorry

end power_of_two_equality_l3757_375763


namespace share_ratio_l3757_375739

theorem share_ratio (total money : ℕ) (a_share : ℕ) (x : ℚ) :
  total = 600 →
  a_share = 240 →
  a_share = x * (total - a_share) →
  (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share)))))))) = total - a_share - (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share)))))) →
  (a_share : ℚ) / (total - a_share) = 2/3 := by
  sorry

end share_ratio_l3757_375739


namespace correct_calculation_l3757_375787

theorem correct_calculation (x y : ℝ) : -2 * x^2 * y - 3 * y * x^2 = -5 * x^2 * y := by
  sorry

end correct_calculation_l3757_375787


namespace profit_ratio_calculation_l3757_375759

theorem profit_ratio_calculation (p q : ℕ) (investment_ratio_p investment_ratio_q : ℕ) 
  (investment_duration_p investment_duration_q : ℕ) :
  investment_ratio_p = 7 →
  investment_ratio_q = 5 →
  investment_duration_p = 2 →
  investment_duration_q = 4 →
  (investment_ratio_p * investment_duration_p) / (investment_ratio_q * investment_duration_q) = 7 / 10 := by
  sorry

end profit_ratio_calculation_l3757_375759


namespace min_value_reciprocal_sum_l3757_375716

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1 / x + 3 / y) ≥ 2 + Real.sqrt 3 := by
  sorry

end min_value_reciprocal_sum_l3757_375716


namespace terrell_weight_lifting_l3757_375775

/-- The number of times Terrell lifts the original weights -/
def original_lifts : ℕ := 10

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 25

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 3

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 10

/-- The number of times Terrell must lift the new weights to match the total weight -/
def new_lifts : ℕ := 25

theorem terrell_weight_lifting :
  num_weights * original_weight * original_lifts =
  num_weights * new_weight * new_lifts := by
  sorry

end terrell_weight_lifting_l3757_375775


namespace smallest_perfect_square_divisible_by_2_3_5_l3757_375737

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, n^2 = 900 ∧ 
  (∀ m : ℕ, m > 0 → m^2 < 900 → ¬(2 ∣ m^2 ∧ 3 ∣ m^2 ∧ 5 ∣ m^2)) ∧
  2 ∣ 900 ∧ 3 ∣ 900 ∧ 5 ∣ 900 :=
by sorry

end smallest_perfect_square_divisible_by_2_3_5_l3757_375737


namespace vector_sum_zero_implies_parallel_l3757_375799

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_sum_zero_implies_parallel (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a + b = 0 → parallel a b) ∧ ¬(parallel a b → a + b = 0) := by sorry

end vector_sum_zero_implies_parallel_l3757_375799


namespace f_has_max_and_min_l3757_375745

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m+6)*x + 1

/-- The derivative of f with respect to x -/
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + (m+6)

/-- Theorem stating the condition for f to have both a maximum and a minimum -/
theorem f_has_max_and_min (m : ℝ) : 
  (∃ (a b : ℝ), ∀ x, f m x ≤ f m a ∧ f m x ≥ f m b) ↔ m < -3 ∨ m > 6 := by
  sorry

end f_has_max_and_min_l3757_375745


namespace youngest_age_proof_l3757_375746

theorem youngest_age_proof (n : ℕ) (current_avg : ℝ) (past_avg : ℝ) :
  n = 7 ∧ current_avg = 30 ∧ past_avg = 27 →
  (n : ℝ) * current_avg - (n - 1 : ℝ) * past_avg = 48 := by
  sorry

end youngest_age_proof_l3757_375746


namespace words_with_consonant_l3757_375726

def letter_set : Finset Char := {'A', 'B', 'C', 'D', 'E'}
def vowel_set : Finset Char := {'A', 'E'}
def word_length : Nat := 5

def total_words : Nat := letter_set.card ^ word_length
def all_vowel_words : Nat := vowel_set.card ^ word_length

theorem words_with_consonant :
  total_words - all_vowel_words = 3093 :=
sorry

end words_with_consonant_l3757_375726


namespace complex_exp_13pi_over_2_l3757_375783

theorem complex_exp_13pi_over_2 : Complex.exp ((13 * Real.pi / 2) * Complex.I) = Complex.I := by
  sorry

end complex_exp_13pi_over_2_l3757_375783


namespace forces_equilibrium_l3757_375797

/-- A 2D vector representing a force -/
structure Force where
  x : ℝ
  y : ℝ

/-- Add two forces -/
def Force.add (f g : Force) : Force :=
  ⟨f.x + g.x, f.y + g.y⟩

instance : Add Force :=
  ⟨Force.add⟩

/-- The zero force -/
def Force.zero : Force :=
  ⟨0, 0⟩

instance : Zero Force :=
  ⟨Force.zero⟩

theorem forces_equilibrium (f₁ f₂ f₃ f₄ : Force) 
    (h₁ : f₁ = ⟨-2, -1⟩)
    (h₂ : f₂ = ⟨-3, 2⟩)
    (h₃ : f₃ = ⟨4, -3⟩)
    (h₄ : f₄ = ⟨1, 2⟩) :
    f₁ + f₂ + f₃ + f₄ = 0 := by
  sorry

end forces_equilibrium_l3757_375797


namespace oliver_puzzle_cost_l3757_375762

/-- The amount of money Oliver spent on the puzzle -/
def puzzle_cost (initial_amount savings frisbee_cost birthday_gift final_amount : ℕ) : ℕ :=
  initial_amount + savings - frisbee_cost + birthday_gift - final_amount

theorem oliver_puzzle_cost : 
  puzzle_cost 9 5 4 8 15 = 3 := by sorry

end oliver_puzzle_cost_l3757_375762


namespace intersection_of_M_and_N_l3757_375781

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end intersection_of_M_and_N_l3757_375781


namespace minimal_circle_and_intersecting_line_l3757_375794

-- Define the right-angled triangle
def triangle : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 / 4 + p.2 / 2 ≤ 1}

-- Define the circle equation
def circle_equation (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Define the line equation
def line_equation (slope : ℝ) (intercept : ℝ) (point : ℝ × ℝ) : Prop :=
  point.2 = slope * point.1 + intercept

theorem minimal_circle_and_intersecting_line :
  ∃ (center : ℝ × ℝ) (radius : ℝ) (intercept : ℝ),
    (∀ p ∈ triangle, circle_equation center radius p) ∧
    (∀ c, ∀ r, (∀ p ∈ triangle, circle_equation c r p) → r ≥ radius) ∧
    center = (2, 1) ∧
    radius^2 = 5 ∧
    (intercept = -1 - Real.sqrt 5 ∨ intercept = -1 + Real.sqrt 5) ∧
    (∃ A B : ℝ × ℝ,
      A ≠ B ∧
      circle_equation center radius A ∧
      circle_equation center radius B ∧
      line_equation 1 intercept A ∧
      line_equation 1 intercept B ∧
      ((A.1 - center.1) * (B.1 - center.1) + (A.2 - center.2) * (B.2 - center.2) = 0)) :=
by sorry

end minimal_circle_and_intersecting_line_l3757_375794


namespace symmetric_points_nm_value_l3757_375713

/-- Given two points P and Q symmetric with respect to the y-axis, prove that n^m = 1/2 -/
theorem symmetric_points_nm_value (m n : ℝ) : 
  (m - 1 = -2) → (4 = n + 2) → n^m = 1/2 := by sorry

end symmetric_points_nm_value_l3757_375713


namespace bonus_calculation_l3757_375732

/-- A quadratic function f(x) = kx^2 + b that satisfies certain conditions -/
def f (k b x : ℝ) : ℝ := k * x^2 + b

theorem bonus_calculation (k b : ℝ) (h1 : k > 0) 
  (h2 : f k b 10 = 0) (h3 : f k b 20 = 2) : f k b 200 = 266 := by
  sorry

end bonus_calculation_l3757_375732


namespace fencing_cost_per_meter_l3757_375752

/-- Given a rectangular plot with the following properties:
  - The length is 60 meters
  - The length is 20 meters more than the breadth
  - The total cost of fencing is Rs. 5300
  Prove that the cost of fencing per meter is Rs. 26.50 -/
theorem fencing_cost_per_meter
  (length : ℝ)
  (breadth : ℝ)
  (total_cost : ℝ)
  (h1 : length = 60)
  (h2 : length = breadth + 20)
  (h3 : total_cost = 5300) :
  total_cost / (2 * length + 2 * breadth) = 26.5 := by
  sorry

end fencing_cost_per_meter_l3757_375752


namespace sandy_spending_percentage_l3757_375798

/-- Given that Sandy took $310 for shopping and had $217 left after spending,
    prove that she spent 30% of her money. -/
theorem sandy_spending_percentage (money_taken : ℝ) (money_left : ℝ) : 
  money_taken = 310 → money_left = 217 → 
  (money_taken - money_left) / money_taken * 100 = 30 := by
  sorry

end sandy_spending_percentage_l3757_375798


namespace investment_inconsistency_l3757_375738

theorem investment_inconsistency :
  ¬ ∃ (r x y : ℝ), 
    x + y = 10000 ∧ 
    x > y ∧ 
    y > 0 ∧ 
    0.05 * y = 6000 ∧ 
    r * x = 0.05 * y + 160 ∧ 
    r > 0 := by
  sorry

end investment_inconsistency_l3757_375738


namespace overweight_condition_equiv_l3757_375776

/-- Ideal weight formula -/
def ideal_weight (h : ℝ) : ℝ := 22 * h^2

/-- Overweight threshold -/
def overweight_threshold (h : ℝ) : ℝ := 1.1 * ideal_weight h

/-- Overweight condition -/
def is_overweight (W h : ℝ) : Prop := W > overweight_threshold h

/-- Quadratic overweight condition -/
def quadratic_overweight (c d e : ℝ) (W h : ℝ) : Prop := W > c * h^2 + d * h + e

theorem overweight_condition_equiv :
  ∃ c d e : ℝ, ∀ W h : ℝ, is_overweight W h ↔ quadratic_overweight c d e W h :=
sorry

end overweight_condition_equiv_l3757_375776


namespace ellipse_m_value_l3757_375774

/-- An ellipse with equation x^2 + my^2 = 1, where m is a positive real number -/
structure Ellipse (m : ℝ) where
  equation : ∀ x y : ℝ, x^2 + m*y^2 = 1

/-- The foci of the ellipse are on the x-axis -/
def foci_on_x_axis (e : Ellipse m) : Prop :=
  ∃ c : ℝ, c^2 = 1 - 1/m

/-- The length of the major axis is twice the length of the minor axis -/
def major_axis_twice_minor (e : Ellipse m) : Prop :=
  1 = 2 * (1/m).sqrt

/-- Theorem: For an ellipse with equation x^2 + my^2 = 1, where the foci are on the x-axis
    and the length of the major axis is twice the length of the minor axis, m = 4 -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m) 
    (h1 : foci_on_x_axis e) (h2 : major_axis_twice_minor e) : m = 4 := by
  sorry

end ellipse_m_value_l3757_375774


namespace cube_parabola_locus_l3757_375706

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  origin : Point3D
  sideLength : ℝ

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ :=
  sorry

/-- Calculate the distance from a point to a plane -/
def distanceToPlane (p : Point3D) (plane : Plane3D) : ℝ :=
  sorry

/-- Check if a point is on a face of the cube -/
def isOnFace (p : Point3D) (cube : Cube) (face : Plane3D) : Prop :=
  sorry

/-- Define a parabola as a set of points -/
def isParabola (points : Set Point3D) : Prop :=
  sorry

theorem cube_parabola_locus (cube : Cube) (B : Point3D) (faceBCC₁B₁ planeCDD₁C₁ : Plane3D) :
  let locus := {M : Point3D | isOnFace M cube faceBCC₁B₁ ∧ 
                               distance M B = distanceToPlane M planeCDD₁C₁}
  isParabola locus := by
  sorry

end cube_parabola_locus_l3757_375706


namespace waiter_tip_problem_l3757_375766

theorem waiter_tip_problem (total_customers : ℕ) (tip_amount : ℕ) (total_tips : ℕ) :
  total_customers = 7 →
  tip_amount = 9 →
  total_tips = 27 →
  total_customers - (total_tips / tip_amount) = 4 :=
by
  sorry

end waiter_tip_problem_l3757_375766


namespace exists_infinite_periodic_sequence_l3757_375704

/-- A sequence of natural numbers -/
def InfiniteSequence := ℕ → ℕ

/-- Property: every natural number appears infinitely many times in the sequence -/
def AppearsInfinitelyOften (s : InfiniteSequence) : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, ∃ i ≥ k, s i = n

/-- Property: the sequence is periodic modulo m for every positive integer m -/
def PeriodicModulo (s : InfiniteSequence) : Prop :=
  ∀ m : ℕ+, ∃ p : ℕ+, ∀ i : ℕ, s (i + p) ≡ s i [MOD m]

/-- Theorem: There exists a sequence of natural numbers that appears infinitely often
    and is periodic modulo every positive integer -/
theorem exists_infinite_periodic_sequence :
  ∃ s : InfiniteSequence, AppearsInfinitelyOften s ∧ PeriodicModulo s := by
  sorry

end exists_infinite_periodic_sequence_l3757_375704


namespace max_sum_of_unknown_pairs_l3757_375736

def pairwise_sums (a b c d : ℕ) : Finset ℕ :=
  {a + b, a + c, a + d, b + c, b + d, c + d}

theorem max_sum_of_unknown_pairs (a b c d : ℕ) :
  let sums := pairwise_sums a b c d
  ∀ x y, x ∈ sums → y ∈ sums →
    {210, 335, 296, 245, x, y} = sums →
    x + y ≤ 717 :=
sorry

end max_sum_of_unknown_pairs_l3757_375736


namespace population_growth_l3757_375708

/-- The initial population of the town -/
def initial_population : ℝ := 1000

/-- The growth rate in the first year -/
def first_year_growth : ℝ := 0.10

/-- The growth rate in the second year -/
def second_year_growth : ℝ := 0.20

/-- The final population after two years -/
def final_population : ℝ := 1320

/-- Theorem stating the relationship between initial and final population -/
theorem population_growth :
  initial_population * (1 + first_year_growth) * (1 + second_year_growth) = final_population := by
  sorry

#check population_growth

end population_growth_l3757_375708


namespace midpoint_specific_segment_l3757_375751

/-- Given two points in polar coordinates, returns their midpoint. -/
def polar_midpoint (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ :=
  sorry

theorem midpoint_specific_segment :
  let A : ℝ × ℝ := (5, π/4)
  let B : ℝ × ℝ := (5, 3*π/4)
  let M : ℝ × ℝ := polar_midpoint A.1 A.2 B.1 B.2
  M.1 = 5*Real.sqrt 2/2 ∧ M.2 = 3*π/8 :=
sorry

end midpoint_specific_segment_l3757_375751


namespace min_value_squared_sum_l3757_375703

theorem min_value_squared_sum (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) 
  (h2 : t * u * v * w = 25) : 
  ∃ (min : ℝ), min = 80 ∧ 
  ∀ (x : ℝ), x = (p*t)^2 + (q*u)^2 + (r*v)^2 + (s*w)^2 → x ≥ min :=
sorry

end min_value_squared_sum_l3757_375703


namespace min_value_xy_plus_two_over_xy_l3757_375710

theorem min_value_xy_plus_two_over_xy (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ z w : ℝ, z > 0 → w > 0 → z + w = 1 → x * y + 2 / (x * y) ≤ z * w + 2 / (z * w)) ∧ 
  x * y + 2 / (x * y) = 33 / 4 :=
sorry

end min_value_xy_plus_two_over_xy_l3757_375710


namespace exactly_one_B_divisible_by_7_l3757_375719

def is_multiple_of_7 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * k

def number_47B (B : ℕ) : ℕ :=
  400 + 70 + B

theorem exactly_one_B_divisible_by_7 :
  ∃! B : ℕ, B ≤ 9 ∧ is_multiple_of_7 (number_47B B) :=
sorry

end exactly_one_B_divisible_by_7_l3757_375719


namespace product_division_theorem_l3757_375773

theorem product_division_theorem (x y : ℝ) (hx : x = 1.6666666666666667) (hx_nonzero : x ≠ 0) :
  Real.sqrt ((5 * x) / y) = x → y = 3 := by
  sorry

end product_division_theorem_l3757_375773


namespace adjacent_rectangles_area_l3757_375770

/-- The total area of two adjacent rectangles -/
theorem adjacent_rectangles_area 
  (u v w z : Real) 
  (hu : u > 0) 
  (hv : v > 0) 
  (hw : w > 0) 
  (hz : z > w) : 
  let first_rectangle := (u + v) * w
  let second_rectangle := (u + v) * (z - w)
  first_rectangle + second_rectangle = (u + v) * z :=
by sorry

end adjacent_rectangles_area_l3757_375770


namespace min_dot_product_l3757_375700

-- Define the rectangle ABCD
def Rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ A.2 = B.2 ∧ B.1 = C.1 ∧ C.2 = D.2

-- Define the points P and Q
def P (x : ℝ) : ℝ × ℝ := (2 - x, 0)
def Q (x : ℝ) : ℝ × ℝ := (2, 1 + x)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem min_dot_product :
  ∀ (A B C D : ℝ × ℝ) (x : ℝ),
    Rectangle A B C D →
    A = (0, 1) →
    B = (2, 1) →
    C = (2, 0) →
    D = (0, 0) →
    0 ≤ x →
    x ≤ 2 →
    (∀ y : ℝ, 0 ≤ y → y ≤ 2 →
      dot_product ((-2 + y, 1)) (y, 1 + y) ≥ 3/4) :=
by sorry

end min_dot_product_l3757_375700


namespace smallest_prime_12_less_than_perfect_square_l3757_375790

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_prime_12_less_than_perfect_square :
  ∃ (n : ℕ), is_perfect_square n ∧ 
             is_prime (n - 12) ∧ 
             (n - 12 = 13) ∧
             (∀ m : ℕ, is_perfect_square m → is_prime (m - 12) → m - 12 ≥ 13) :=
sorry

end smallest_prime_12_less_than_perfect_square_l3757_375790


namespace circle_area_equal_perimeter_l3757_375725

theorem circle_area_equal_perimeter (s : ℝ) (r : ℝ) : 
  s > 0 → 
  r > 0 → 
  s^2 = 16 → 
  4 * s = 2 * Real.pi * r → 
  Real.pi * r^2 = 64 / Real.pi := by
  sorry

end circle_area_equal_perimeter_l3757_375725


namespace unique_solution_system_l3757_375756

theorem unique_solution_system (x y z : ℝ) :
  (x + y = 2 ∧ x * y - z^2 = 1) ↔ (x = 1 ∧ y = 1 ∧ z = 0) := by
  sorry

end unique_solution_system_l3757_375756


namespace range_of_m_l3757_375750

-- Define P and q as functions of x and m
def P (x : ℝ) : Prop := |4 - x| / 3 ≤ 2

def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(P x) → ¬(q x m)) →
  (∃ x, P x ∧ ¬(q x m)) →
  m ≥ 9 :=
sorry

end range_of_m_l3757_375750


namespace divisibility_count_l3757_375702

theorem divisibility_count : ∃ (S : Finset Nat), 
  (∀ n ∈ S, n ≤ 30 ∧ n > 0 ∧ (n! % (n * (n + 2) / 3) = 0)) ∧ 
  Finset.card S = 30 := by
  sorry

end divisibility_count_l3757_375702


namespace fred_current_money_l3757_375765

/-- Fred's money situation --/
def fred_money_problem (initial_amount earned_amount : ℕ) : Prop :=
  initial_amount + earned_amount = 40

/-- Theorem: Fred now has 40 dollars --/
theorem fred_current_money :
  fred_money_problem 19 21 :=
by
  sorry

end fred_current_money_l3757_375765


namespace probability_different_tens_proof_l3757_375715

/-- The number of integers in the range 10 to 79, inclusive. -/
def total_numbers : ℕ := 70

/-- The number of different tens digits available in the range 10 to 79. -/
def available_tens_digits : ℕ := 7

/-- The number of integers for each tens digit. -/
def numbers_per_tens : ℕ := 10

/-- The number of integers to be chosen. -/
def chosen_count : ℕ := 7

/-- The probability of selecting 7 different integers from the range 10 to 79 (inclusive)
    such that each has a different tens digit. -/
def probability_different_tens : ℚ := 10000000 / 93947434

theorem probability_different_tens_proof :
  (numbers_per_tens ^ chosen_count : ℚ) / (total_numbers.choose chosen_count) = probability_different_tens :=
sorry

end probability_different_tens_proof_l3757_375715


namespace prime_product_theorem_l3757_375712

def largest_one_digit_prime : ℕ := 7
def second_largest_one_digit_prime : ℕ := 5
def second_largest_two_digit_prime : ℕ := 89

theorem prime_product_theorem :
  largest_one_digit_prime * second_largest_one_digit_prime * second_largest_two_digit_prime = 3115 := by
  sorry

end prime_product_theorem_l3757_375712


namespace classroom_capacity_l3757_375764

/-- The number of rows of desks in the classroom -/
def num_rows : ℕ := 8

/-- The number of desks in the first row -/
def first_row_desks : ℕ := 10

/-- The increase in number of desks for each subsequent row -/
def desk_increase : ℕ := 2

/-- The total number of desks in the classroom -/
def total_desks : ℕ := (num_rows * (2 * first_row_desks + (num_rows - 1) * desk_increase)) / 2

theorem classroom_capacity :
  total_desks = 136 := by sorry

end classroom_capacity_l3757_375764


namespace largest_four_digit_divisible_by_98_l3757_375723

theorem largest_four_digit_divisible_by_98 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 98 = 0 → n ≤ 9998 :=
by sorry

end largest_four_digit_divisible_by_98_l3757_375723


namespace a_5_value_l3757_375741

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_5_value (a : ℕ → ℝ) (h_geo : geometric_sequence a) 
  (h_3 : a 3 = -1) (h_7 : a 7 = -9) : a 5 = -3 := by
  sorry

end a_5_value_l3757_375741


namespace melted_to_spending_value_ratio_l3757_375717

-- Define the weight of a quarter in ounces
def quarter_weight : ℚ := 1/5

-- Define the value of melted gold per ounce in dollars
def melted_gold_value_per_ounce : ℚ := 100

-- Define the spending value of a quarter in dollars
def quarter_spending_value : ℚ := 1/4

-- Theorem statement
theorem melted_to_spending_value_ratio : 
  (melted_gold_value_per_ounce / quarter_weight) / (1 / quarter_spending_value) = 80 := by
  sorry

end melted_to_spending_value_ratio_l3757_375717


namespace equation_solution_l3757_375784

theorem equation_solution : 
  let S : Set ℝ := {x | (x^4 + 4*x^3*Real.sqrt 3 + 12*x^2 + 8*x*Real.sqrt 3 + 4) + (x^2 + 2*x*Real.sqrt 3 + 3) = 0}
  S = {-Real.sqrt 3, -Real.sqrt 3 + 1, -Real.sqrt 3 - 1} := by
  sorry

end equation_solution_l3757_375784


namespace solve_linear_equation_l3757_375778

theorem solve_linear_equation :
  ∀ x : ℚ, -4 * x - 15 = 12 * x + 5 → x = -5/4 := by
  sorry

end solve_linear_equation_l3757_375778


namespace set_inclusion_equivalence_l3757_375755

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 ≤ 5/4}

def B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1 - 1| + 2*|p.2 - 2| ≤ a}

-- State the theorem
theorem set_inclusion_equivalence (a : ℝ) : A ⊆ B a ↔ a ≥ 5/2 := by
  sorry

end set_inclusion_equivalence_l3757_375755


namespace roller_coaster_tickets_l3757_375796

/-- The number of friends going on the roller coaster ride -/
def num_friends : ℕ := 8

/-- The total number of tickets needed for all friends -/
def total_tickets : ℕ := 48

/-- The number of tickets required per ride -/
def tickets_per_ride : ℕ := total_tickets / num_friends

theorem roller_coaster_tickets : tickets_per_ride = 6 := by
  sorry

end roller_coaster_tickets_l3757_375796


namespace parabola_x_axis_intersections_l3757_375701

/-- The number of intersection points between y = 3x^2 + 2x + 1 and the x-axis is 0 -/
theorem parabola_x_axis_intersections :
  let f (x : ℝ) := 3 * x^2 + 2 * x + 1
  (∃ x : ℝ, f x = 0) = False :=
by sorry

end parabola_x_axis_intersections_l3757_375701


namespace roys_pen_ratio_l3757_375724

/-- Proves that the ratio of black pens to blue pens is 2:1 given the conditions of Roy's pen collection --/
theorem roys_pen_ratio :
  ∀ (blue black red : ℕ),
    blue = 2 →
    red = 2 * black - 2 →
    blue + black + red = 12 →
    black / blue = 2 / 1 :=
by
  sorry

end roys_pen_ratio_l3757_375724


namespace quadratic_roots_property_l3757_375789

theorem quadratic_roots_property (m n : ℝ) : 
  m^2 - 1840*m + 2009 = 0 → 
  n^2 - 1840*n + 2009 = 0 → 
  (m^2 - 1841*m + 2009) * (n^2 - 1841*n + 2009) = 2009 := by
sorry

end quadratic_roots_property_l3757_375789


namespace arithmetic_sequence_problem_l3757_375721

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 8)
  (h_fourth : a 4 = 7) : 
  a 5 = 10 := by
sorry

end arithmetic_sequence_problem_l3757_375721


namespace solution_set_inequality_l3757_375718

theorem solution_set_inequality (x : ℝ) : 
  (x - 1)^2 > 4 ↔ x < -1 ∨ x > 3 := by sorry

end solution_set_inequality_l3757_375718


namespace max_insurmountable_questions_max_insurmountable_questions_is_10_l3757_375777

theorem max_insurmountable_questions :
  ∀ (x₀ x₁ x₂ x₃ : ℕ),
    x₀ + x₁ + x₂ + x₃ = 40 →
    3 * x₃ + 2 * x₂ + x₁ = 64 →
    x₂ = 2 * x₀ →
    x₀ ≤ 10 :=
by
  sorry

theorem max_insurmountable_questions_is_10 :
  ∃ (x₀ x₁ x₂ x₃ : ℕ),
    x₀ + x₁ + x₂ + x₃ = 40 ∧
    3 * x₃ + 2 * x₂ + x₁ = 64 ∧
    x₂ = 2 * x₀ ∧
    x₀ = 10 :=
by
  sorry

end max_insurmountable_questions_max_insurmountable_questions_is_10_l3757_375777


namespace shifted_sine_equivalence_shift_amount_l3757_375748

/-- Proves that the given function is equivalent to a shifted sine function -/
theorem shifted_sine_equivalence (x : ℝ) : 
  (1/2 : ℝ) * Real.sin (4*x) - (Real.sqrt 3 / 2) * Real.cos (4*x) = Real.sin (4*x - π/3) :=
by sorry

/-- Proves that the shift is π/12 units to the right -/
theorem shift_amount : 
  ∃ (k : ℝ), ∀ (x : ℝ), Real.sin (4*x - π/3) = Real.sin (4*(x - k)) ∧ k = π/12 :=
by sorry

end shifted_sine_equivalence_shift_amount_l3757_375748


namespace pufferfish_count_swordfish_pufferfish_relation_total_fish_count_l3757_375771

/-- The number of pufferfish in an aquarium exhibit -/
def num_pufferfish : ℕ := 15

/-- The number of swordfish in an aquarium exhibit -/
def num_swordfish : ℕ := 5 * num_pufferfish

/-- The total number of fish in the aquarium exhibit -/
def total_fish : ℕ := 90

/-- Theorem stating that the number of pufferfish is 15 -/
theorem pufferfish_count : num_pufferfish = 15 := by sorry

/-- Theorem verifying the relationship between swordfish and pufferfish -/
theorem swordfish_pufferfish_relation : num_swordfish = 5 * num_pufferfish := by sorry

/-- Theorem verifying the total number of fish -/
theorem total_fish_count : num_swordfish + num_pufferfish = total_fish := by sorry

end pufferfish_count_swordfish_pufferfish_relation_total_fish_count_l3757_375771


namespace strawberry_sales_formula_l3757_375743

/-- The relationship between strawberry sales volume and total sales price -/
theorem strawberry_sales_formula (n : ℕ+) :
  let price_increase : ℝ := 40.5
  let total_price : ℕ+ → ℝ := λ k => k.val * price_increase
  total_price n = n.val * price_increase :=
by sorry

end strawberry_sales_formula_l3757_375743


namespace sum_of_specific_T_values_l3757_375734

def T (n : ℕ) : ℤ :=
  (-1 : ℤ) + 4 - 3 + 8 - 5 + ((-1)^n * (2*n : ℤ)) + ((-1)^(n+1) * (n : ℤ))

theorem sum_of_specific_T_values :
  T 27 + T 43 + T 60 = -84 ∨
  T 27 + T 43 + T 60 = -42 ∨
  T 27 + T 43 + T 60 = 0 ∨
  T 27 + T 43 + T 60 = 42 ∨
  T 27 + T 43 + T 60 = 84 := by
  sorry

end sum_of_specific_T_values_l3757_375734


namespace impossible_score_53_l3757_375728

/-- Represents the score of a quiz -/
structure QuizScore where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  total_questions : ℕ
  score : ℤ

/-- Calculates the score based on the number of correct, incorrect, and unanswered questions -/
def calculate_score (c i u : ℕ) : ℤ :=
  (4 : ℤ) * c - i

/-- Checks if a QuizScore is valid according to the quiz rules -/
def is_valid_score (qs : QuizScore) : Prop :=
  qs.correct + qs.incorrect + qs.unanswered = qs.total_questions ∧
  qs.score = calculate_score qs.correct qs.incorrect qs.unanswered

/-- Theorem: It's impossible to achieve a score of 53 in the given quiz -/
theorem impossible_score_53 :
  ¬ ∃ (qs : QuizScore), qs.total_questions = 15 ∧ is_valid_score qs ∧ qs.score = 53 :=
by sorry

end impossible_score_53_l3757_375728


namespace power_equation_solution_l3757_375760

theorem power_equation_solution : 2^4 - 7 = 3^3 + (-18) := by
  sorry

end power_equation_solution_l3757_375760


namespace real_part_reciprocal_l3757_375753

/-- For a nonreal complex number z with |z| = 2, 
    the real part of 1/(2-z) is (2-x)/(8-4x+x^2), where x is the real part of z -/
theorem real_part_reciprocal (z : ℂ) (x : ℝ) (h1 : z.im ≠ 0) (h2 : Complex.abs z = 2) 
  (h3 : z.re = x) : 
  Complex.re (1 / (2 - z)) = (2 - x) / (8 - 4*x + x^2) := by
  sorry

end real_part_reciprocal_l3757_375753


namespace marks_pond_depth_l3757_375758

/-- Given that Peter's pond is 5 feet deep and Mark's pond is 4 feet deeper than 3 times Peter's pond,
    prove that the depth of Mark's pond is 19 feet. -/
theorem marks_pond_depth (peters_depth : ℕ) (marks_depth : ℕ) 
  (h1 : peters_depth = 5)
  (h2 : marks_depth = 3 * peters_depth + 4) :
  marks_depth = 19 := by
  sorry

end marks_pond_depth_l3757_375758


namespace fence_cost_for_square_plot_l3757_375757

theorem fence_cost_for_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 81) (h2 : price_per_foot = 58) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 2088 := by
sorry

end fence_cost_for_square_plot_l3757_375757


namespace defective_units_shipped_l3757_375733

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ) :
  defective_rate = 0.1 →
  shipped_rate = 0.05 →
  (defective_rate * shipped_rate * 100) = 0.5 := by
  sorry

end defective_units_shipped_l3757_375733


namespace room_occupancy_l3757_375767

theorem room_occupancy (x : ℕ) : 
  (3 * x / 8 : ℚ) - 6 = 18 → x = 64 := by
  sorry

end room_occupancy_l3757_375767


namespace rachel_painting_time_l3757_375731

def minutes_per_day_first_6 : ℕ := 100
def days_first_period : ℕ := 6
def minutes_per_day_next_2 : ℕ := 120
def days_second_period : ℕ := 2
def target_average : ℕ := 110
def total_days : ℕ := 10

theorem rachel_painting_time :
  (minutes_per_day_first_6 * days_first_period +
   minutes_per_day_next_2 * days_second_period +
   (target_average * total_days - 
    (minutes_per_day_first_6 * days_first_period +
     minutes_per_day_next_2 * days_second_period))) / total_days = target_average :=
by sorry

end rachel_painting_time_l3757_375731


namespace sum_of_remainders_mod_500_l3757_375735

def remainders : Finset ℕ := Finset.image (fun n => (3^n) % 500) (Finset.range 101)

def T : ℕ := Finset.sum remainders id

theorem sum_of_remainders_mod_500 : T % 500 = (Finset.sum (Finset.range 101) (fun n => (3^n) % 500)) % 500 := by
  sorry

end sum_of_remainders_mod_500_l3757_375735


namespace min_sum_perpendicular_sides_right_triangle_l3757_375761

theorem min_sum_perpendicular_sides_right_triangle (a b : ℝ) (h_positive_a : a > 0) (h_positive_b : b > 0) (h_area : a * b / 2 = 50) :
  a + b ≥ 20 ∧ (a + b = 20 ↔ a = 10 ∧ b = 10) :=
by sorry

end min_sum_perpendicular_sides_right_triangle_l3757_375761


namespace imaginary_part_of_complex_fraction_l3757_375780

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (1 - 2*I) / (1 + I^3)
  Complex.im z = -1/2 := by sorry

end imaginary_part_of_complex_fraction_l3757_375780


namespace inequalities_solution_range_l3757_375714

theorem inequalities_solution_range (m : ℝ) : 
  (∃ (x₁ x₂ x₃ x₄ : ℤ), 
    (∀ x : ℤ, (3 * ↑x - m < 0 ∧ 7 - 2 * ↑x < 5) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) ∧
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄) →
  (15 < m ∧ m ≤ 18) :=
by sorry

end inequalities_solution_range_l3757_375714


namespace percent_of_y_l3757_375749

theorem percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 10 + (3 * y) / 10) / y = 1 / 2 := by
  sorry

end percent_of_y_l3757_375749


namespace alice_bob_meet_l3757_375772

/-- The number of points on the circle -/
def n : ℕ := 24

/-- Alice's starting position -/
def alice_start : ℕ := 1

/-- Bob's starting position -/
def bob_start : ℕ := 12

/-- Alice's movement per turn (clockwise) -/
def alice_move : ℕ := 7

/-- Bob's movement per turn (counterclockwise) -/
def bob_move : ℕ := 17

/-- The number of turns it takes for Alice and Bob to meet -/
def meeting_turns : ℕ := 5

/-- Theorem stating that Alice and Bob meet after the specified number of turns -/
theorem alice_bob_meet :
  (alice_start + meeting_turns * alice_move) % n = 
  (bob_start - meeting_turns * bob_move + n * meeting_turns) % n :=
sorry

end alice_bob_meet_l3757_375772


namespace square_independence_of_p_l3757_375786

theorem square_independence_of_p (m n p k : ℕ) : 
  m > 0 → n > 0 → p.Prime → p > m → 
  m^2 + n^2 + p^2 - 2*m*n - 2*m*p - 2*n*p = k^2 → 
  ∃ f : ℕ → ℕ, ∀ q : ℕ, q.Prime → q > m → 
    m^2 + n^2 + q^2 - 2*m*n - 2*m*q - 2*n*q = (f q)^2 := by
  sorry

end square_independence_of_p_l3757_375786


namespace gcd_lcm_product_75_90_l3757_375742

theorem gcd_lcm_product_75_90 : Nat.gcd 75 90 * Nat.lcm 75 90 = 6750 := by
  sorry

end gcd_lcm_product_75_90_l3757_375742


namespace brownie_problem_l3757_375793

def initial_brownies : ℕ := 16

theorem brownie_problem (B : ℕ) (h1 : B = initial_brownies) :
  let remaining_after_children : ℚ := 3/4 * B
  let remaining_after_family : ℚ := 1/2 * remaining_after_children
  let final_remaining : ℚ := remaining_after_family - 1
  final_remaining = 5 := by sorry

end brownie_problem_l3757_375793


namespace g_of_eight_l3757_375791

/-- Given a function g : ℝ → ℝ satisfying the equation
    g(x) + g(3x+y) + 7xy = g(4x - y) + 3x^2 + 2 for all real x and y,
    prove that g(8) = -30. -/
theorem g_of_eight (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, g x + g (3*x + y) + 7*x*y = g (4*x - y) + 3*x^2 + 2) : 
  g 8 = -30 := by
  sorry

end g_of_eight_l3757_375791


namespace smallest_factor_correct_l3757_375788

/-- The smallest positive integer a such that both 112 and 33 are factors of a * 43 * 62 * 1311 -/
def smallest_factor : ℕ := 1848

/-- Theorem stating that the smallest_factor is correct -/
theorem smallest_factor_correct :
  (∀ k : ℕ, k > 0 → 112 ∣ (k * 43 * 62 * 1311) → 33 ∣ (k * 43 * 62 * 1311) → k ≥ smallest_factor) ∧
  (112 ∣ (smallest_factor * 43 * 62 * 1311)) ∧
  (33 ∣ (smallest_factor * 43 * 62 * 1311)) :=
by sorry

end smallest_factor_correct_l3757_375788


namespace complex_sum_powers_l3757_375754

theorem complex_sum_powers (z : ℂ) (h : z^2 - z + 1 = 0) :
  z^99 + z^100 + z^101 + z^102 + z^103 = 2 + Complex.I * Real.sqrt 3 ∨
  z^99 + z^100 + z^101 + z^102 + z^103 = 2 - Complex.I * Real.sqrt 3 :=
by sorry

end complex_sum_powers_l3757_375754


namespace green_rotten_no_smell_count_l3757_375722

/-- Represents the types of fruits in the orchard -/
inductive Fruit
| Apple
| Orange
| Pear

/-- Represents the colors of fruits -/
inductive Color
| Red
| Green
| Orange
| Yellow
| Brown

structure OrchardData where
  total_fruits : Fruit → ℕ
  color_distribution : Fruit → Color → ℚ
  rotten_percentage : Fruit → ℚ
  strong_smell_percentage : Fruit → ℚ

def orchard_data : OrchardData := {
  total_fruits := λ f => match f with
    | Fruit.Apple => 200
    | Fruit.Orange => 150
    | Fruit.Pear => 100,
  color_distribution := λ f c => match f, c with
    | Fruit.Apple, Color.Red => 1/2
    | Fruit.Apple, Color.Green => 1/2
    | Fruit.Orange, Color.Orange => 2/5
    | Fruit.Orange, Color.Yellow => 3/5
    | Fruit.Pear, Color.Green => 3/10
    | Fruit.Pear, Color.Brown => 7/10
    | _, _ => 0,
  rotten_percentage := λ f => match f with
    | Fruit.Apple => 2/5
    | Fruit.Orange => 1/4
    | Fruit.Pear => 7/20,
  strong_smell_percentage := λ f => match f with
    | Fruit.Apple => 7/10
    | Fruit.Orange => 1/2
    | Fruit.Pear => 4/5
}

/-- Calculates the number of green rotten fruits without a strong smell in the orchard -/
def green_rotten_no_smell (data : OrchardData) : ℕ :=
  sorry

theorem green_rotten_no_smell_count :
  green_rotten_no_smell orchard_data = 14 :=
sorry

end green_rotten_no_smell_count_l3757_375722


namespace perimeter_of_picture_area_l3757_375744

/-- Given a sheet of paper and a margin, calculate the perimeter of the remaining area --/
def perimeter_of_remaining_area (paper_width paper_length margin : ℕ) : ℕ :=
  2 * ((paper_width - 2 * margin) + (paper_length - 2 * margin))

/-- Theorem: The perimeter of the remaining area for a 12x16 inch paper with 2-inch margins is 40 inches --/
theorem perimeter_of_picture_area : perimeter_of_remaining_area 12 16 2 = 40 := by
  sorry

#eval perimeter_of_remaining_area 12 16 2

end perimeter_of_picture_area_l3757_375744


namespace probability_theorem_l3757_375711

/-- Represents the number of guests -/
def num_guests : ℕ := 4

/-- Represents the number of roll types -/
def num_roll_types : ℕ := 4

/-- Represents the number of each roll type prepared -/
def rolls_per_type : ℕ := 3

/-- Represents the total number of rolls -/
def total_rolls : ℕ := num_roll_types * rolls_per_type

/-- Represents the number of rolls each guest receives -/
def rolls_per_guest : ℕ := num_roll_types

/-- Calculates the probability of each guest receiving one roll of each type -/
def probability_one_of_each : ℚ :=
  (rolls_per_type ^ num_roll_types * (rolls_per_type - 1) ^ num_roll_types * (rolls_per_type - 2) ^ num_roll_types) /
  (Nat.choose total_rolls rolls_per_guest * Nat.choose (total_rolls - rolls_per_guest) rolls_per_guest * Nat.choose (total_rolls - 2*rolls_per_guest) rolls_per_guest)

theorem probability_theorem :
  probability_one_of_each = 12 / 321 :=
by sorry

end probability_theorem_l3757_375711


namespace oil_tank_capacity_l3757_375795

theorem oil_tank_capacity (C : ℝ) (h1 : C > 0) :
  (C / 6 : ℝ) / C = 1 / 6 ∧ (C / 6 + 4) / C = 1 / 3 → C = 24 := by
  sorry

end oil_tank_capacity_l3757_375795


namespace trig_sum_zero_l3757_375785

theorem trig_sum_zero : Real.sin (0 * π / 180) + Real.cos (90 * π / 180) + Real.tan (180 * π / 180) = 0 := by
  sorry

end trig_sum_zero_l3757_375785


namespace range_of_m_l3757_375782

def A (m : ℝ) : Set ℝ := {x : ℝ | x^2 - m*x + 3 = 0}

theorem range_of_m :
  ∀ m : ℝ, (A m ∩ {1, 3} = A m) ↔ ((-2 * Real.sqrt 3 < m ∧ m < 2 * Real.sqrt 3) ∨ m = 4) :=
by sorry

end range_of_m_l3757_375782


namespace find_T_l3757_375747

theorem find_T : ∃ T : ℚ, (1/3 : ℚ) * (1/8 : ℚ) * T = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ T = 120 := by
  sorry

end find_T_l3757_375747


namespace tangential_quadrilateral_theorem_l3757_375769

/-- A circle in a 2D plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Check if four points are concyclic -/
def are_concyclic (A B C D : Point) : Prop := sorry

/-- Check if a point lies on a line segment -/
def point_on_segment (P A B : Point) : Prop := sorry

/-- Check if a circle is tangent to a line segment -/
def circle_tangent_to_segment (circle : Circle) (A B : Point) : Prop := sorry

/-- Distance between two points -/
def distance (A B : Point) : ℝ := sorry

theorem tangential_quadrilateral_theorem 
  (A B C D : Point) 
  (circle1 circle2 : Circle) :
  are_concyclic A B C D →
  point_on_segment circle2.center A B →
  circle_tangent_to_segment circle2 B C →
  circle_tangent_to_segment circle2 C D →
  circle_tangent_to_segment circle2 D A →
  distance A D + distance B C = distance A B := by
  sorry

end tangential_quadrilateral_theorem_l3757_375769


namespace root_transformation_l3757_375720

theorem root_transformation (α β : ℝ) : 
  (2 * α^2 - 5 * α + 3 = 0) → 
  (2 * β^2 - 5 * β + 3 = 0) → 
  ((2 * α - 7)^2 + 9 * (2 * α - 7) + 20 = 0) ∧
  ((2 * β - 7)^2 + 9 * (2 * β - 7) + 20 = 0) :=
by sorry

end root_transformation_l3757_375720
