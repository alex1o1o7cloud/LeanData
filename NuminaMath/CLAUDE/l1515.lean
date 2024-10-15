import Mathlib

namespace NUMINAMATH_CALUDE_unique_intersection_point_l1515_151562

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

-- Theorem statement
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-3, -3) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l1515_151562


namespace NUMINAMATH_CALUDE_alice_gadget_sales_l1515_151555

/-- The worth of gadgets Alice sold -/
def gadget_worth : ℝ := 2500

/-- Alice's monthly basic salary -/
def basic_salary : ℝ := 240

/-- Alice's commission rate -/
def commission_rate : ℝ := 0.02

/-- Amount Alice saves -/
def savings : ℝ := 29

/-- Percentage of total earnings Alice saves -/
def savings_rate : ℝ := 0.10

/-- Alice's total earnings -/
def total_earnings : ℝ := basic_salary + commission_rate * gadget_worth

theorem alice_gadget_sales :
  gadget_worth = 2500 ∧
  basic_salary = 240 ∧
  commission_rate = 0.02 ∧
  savings = 29 ∧
  savings_rate = 0.10 ∧
  savings = savings_rate * total_earnings :=
by sorry

end NUMINAMATH_CALUDE_alice_gadget_sales_l1515_151555


namespace NUMINAMATH_CALUDE_f_greater_than_one_factorial_inequality_l1515_151563

noncomputable def f (x : ℝ) : ℝ := (1/x + 1/2) * Real.log (x + 1)

theorem f_greater_than_one (x : ℝ) (hx : x > 0) : f x > 1 := by sorry

theorem factorial_inequality (n : ℕ) :
  5/6 < Real.log (n.factorial : ℝ) - (n + 1/2) * Real.log n + n ∧
  Real.log (n.factorial : ℝ) - (n + 1/2) * Real.log n + n ≤ 1 := by sorry

end NUMINAMATH_CALUDE_f_greater_than_one_factorial_inequality_l1515_151563


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l1515_151573

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 26 equally spaced trees, 
    with trees at each end and 12 meters between consecutive trees, is 300 meters -/
theorem yard_length_26_trees : 
  yard_length 26 12 = 300 := by sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l1515_151573


namespace NUMINAMATH_CALUDE_steiner_inellipse_center_distance_l1515_151505

/-- Triangle with vertices (0, 0), (3, 0), and (0, 3/2) -/
def T : Set (ℝ × ℝ) := {(0, 0), (3, 0), (0, 3/2)}

/-- The Steiner inellipse of triangle T -/
def E : Set (ℝ × ℝ) := sorry

/-- The center of the Steiner inellipse E -/
def center_E : ℝ × ℝ := sorry

/-- The distance from the center of E to (0, 0) -/
def distance_to_origin : ℝ := sorry

theorem steiner_inellipse_center_distance :
  distance_to_origin = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_steiner_inellipse_center_distance_l1515_151505


namespace NUMINAMATH_CALUDE_max_real_part_of_roots_l1515_151571

open Complex

-- Define the polynomial
def p (z : ℂ) : ℂ := z^6 - z^4 + z^2 - 1

-- Theorem statement
theorem max_real_part_of_roots :
  ∃ (z : ℂ), p z = 0 ∧ 
  ∀ (w : ℂ), p w = 0 → z.re ≥ w.re ∧
  z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_real_part_of_roots_l1515_151571


namespace NUMINAMATH_CALUDE_intersection_y_intercept_sum_l1515_151551

/-- Given two lines that intersect at (3,3), prove that the sum of their y-intercepts is 4 -/
theorem intersection_y_intercept_sum (c d : ℝ) : 
  (3 = (1/3)*3 + c) → (3 = (1/3)*3 + d) → c + d = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_sum_l1515_151551


namespace NUMINAMATH_CALUDE_collinear_vectors_k_l1515_151528

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-2, 1]
def c : Fin 2 → ℝ := ![3, 2]

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, ∀ i : Fin 2, v i = t * w i

theorem collinear_vectors_k (k : ℝ) :
  collinear c (fun i ↦ k * a i + b i) → k = -1 := by
  sorry

#check collinear_vectors_k

end NUMINAMATH_CALUDE_collinear_vectors_k_l1515_151528


namespace NUMINAMATH_CALUDE_problem_statement_l1515_151557

def p : Prop := ∀ x : ℝ, x^2 - 1 ≥ -1

def q : Prop := 4 + 2 = 7

theorem problem_statement : 
  p ∧ ¬q ∧ ¬(p ∧ q) ∧ (p ∨ q) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1515_151557


namespace NUMINAMATH_CALUDE_percentage_of_non_roses_l1515_151585

theorem percentage_of_non_roses (roses tulips daisies : ℕ) 
  (h_roses : roses = 25)
  (h_tulips : tulips = 40)
  (h_daisies : daisies = 35) :
  (100 : ℚ) * (tulips + daisies : ℚ) / (roses + tulips + daisies : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_non_roses_l1515_151585


namespace NUMINAMATH_CALUDE_only_valid_solutions_l1515_151507

/-- A structure representing a solution to the equation AB = B^V --/
structure Solution :=
  (a : Nat) (b : Nat) (v : Nat)
  (h1 : a ≠ b) -- Different letters correspond to different digits
  (h2 : a * 10 + b ≥ 10 ∧ a * 10 + b < 100) -- AB is a two-digit number
  (h3 : a * 10 + b = b ^ v) -- AB = B^V

/-- The set of all valid solutions --/
def validSolutions : Set Solution :=
  { s : Solution | s.a = 3 ∧ s.b = 2 ∧ s.v = 5 ∨
                   s.a = 3 ∧ s.b = 6 ∧ s.v = 2 ∨
                   s.a = 6 ∧ s.b = 4 ∧ s.v = 3 }

/-- Theorem stating that the only solutions are 32 = 2^5, 36 = 6^2, and 64 = 4^3 --/
theorem only_valid_solutions (s : Solution) : s ∈ validSolutions := by
  sorry

end NUMINAMATH_CALUDE_only_valid_solutions_l1515_151507


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l1515_151598

theorem function_satisfies_equation :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 2*x) = |x + 1| :=
by
  -- Define f(x) = √(x + 1)
  let f := λ x : ℝ ↦ Real.sqrt (x + 1)
  
  -- Prove that this f satisfies the equation
  -- for all x ∈ ℝ
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l1515_151598


namespace NUMINAMATH_CALUDE_special_line_equation_l1515_151577

/-- A line passing through point (1,4) with the sum of its x and y intercepts equal to zero -/
structure SpecialLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (1,4) -/
  passes_through_point : slope * 1 + y_intercept = 4
  /-- The sum of x and y intercepts is zero -/
  sum_of_intercepts_zero : (-y_intercept / slope) + y_intercept = 0

/-- The equation of a SpecialLine is either 4x - y = 0 or x - y + 3 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.slope = 4 ∧ l.y_intercept = 0) ∨ (l.slope = 1 ∧ l.y_intercept = 3) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l1515_151577


namespace NUMINAMATH_CALUDE_watermelon_weights_sum_l1515_151544

/-- Watermelon weights problem -/
theorem watermelon_weights_sum : 
  -- Given conditions
  let michael_largest : ℝ := 12
  let clay_first : ℝ := 1.5 * michael_largest
  let john_first : ℝ := clay_first / 2
  let emily : ℝ := 0.75 * john_first
  let sophie_first : ℝ := emily + 3
  let michael_smallest : ℝ := michael_largest * 0.7
  let clay_second : ℝ := clay_first * 1.2
  let john_second : ℝ := (john_first + emily) / 2
  let sophie_second : ℝ := 3 * (clay_second - clay_first)
  -- Theorem statement
  michael_largest + michael_smallest + clay_first + clay_second + 
  john_first + john_second + emily + sophie_first + sophie_second = 104.175 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_weights_sum_l1515_151544


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l1515_151542

/-- Represents a systematic sample of students -/
structure SystematicSample where
  totalStudents : Nat
  sampleSize : Nat
  sampleNumbers : List Nat

/-- Checks if a given sample is a valid systematic sample -/
def isValidSystematicSample (sample : SystematicSample) : Prop :=
  sample.totalStudents = 20 ∧
  sample.sampleSize = 4 ∧
  sample.sampleNumbers = [5, 10, 15, 20]

/-- Theorem stating that the given sample is the correct systematic sample -/
theorem correct_systematic_sample :
  ∃ (sample : SystematicSample), isValidSystematicSample sample :=
sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l1515_151542


namespace NUMINAMATH_CALUDE_max_value_2x_minus_y_l1515_151524

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0) 
  (h2 : y + 1 ≥ 0) 
  (h3 : x + y + 1 ≤ 0) : 
  ∃ (max : ℝ), max = 1 ∧ ∀ z, 2*x - y ≤ z → z ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_minus_y_l1515_151524


namespace NUMINAMATH_CALUDE_probability_four_white_balls_l1515_151526

/-- The probability of drawing 4 white balls from a box containing 7 white balls and 5 black balls -/
theorem probability_four_white_balls (white_balls black_balls drawn : ℕ) : 
  white_balls = 7 →
  black_balls = 5 →
  drawn = 4 →
  (Nat.choose white_balls drawn : ℚ) / (Nat.choose (white_balls + black_balls) drawn) = 7 / 99 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_white_balls_l1515_151526


namespace NUMINAMATH_CALUDE_sum_of_powers_divisible_by_ten_l1515_151594

theorem sum_of_powers_divisible_by_ten (n : ℕ) (h : ¬ (4 ∣ n)) :
  10 ∣ (1^n + 2^n + 3^n + 4^n) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_divisible_by_ten_l1515_151594


namespace NUMINAMATH_CALUDE_spider_has_eight_legs_l1515_151501

/-- The number of legs a human has -/
def human_legs : ℕ := 2

/-- The number of legs a spider has -/
def spider_legs : ℕ := 2 * (2 * human_legs)

/-- Theorem stating that a spider has 8 legs -/
theorem spider_has_eight_legs : spider_legs = 8 := by
  sorry

end NUMINAMATH_CALUDE_spider_has_eight_legs_l1515_151501


namespace NUMINAMATH_CALUDE_geometric_relations_l1515_151518

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (intersect : Plane → Plane → Line)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem geometric_relations
  (l m : Line) (α β γ : Plane)
  (h1 : intersect β γ = l)
  (h2 : parallel l α)
  (h3 : contains α m)
  (h4 : perpendicular m γ) :
  perpendicularPlanes α γ ∧ perpendicularLines l m :=
by sorry

end NUMINAMATH_CALUDE_geometric_relations_l1515_151518


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l1515_151576

theorem number_puzzle_solution : 
  ∃ x : ℚ, x - (3/5) * x = 50 ∧ x = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l1515_151576


namespace NUMINAMATH_CALUDE_garden_fencing_theorem_l1515_151559

/-- Calculates the perimeter of a rectangular garden with given length and width. -/
def garden_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: A rectangular garden with length 60 yards and width equal to half its length
    requires 180 yards of fencing to enclose it. -/
theorem garden_fencing_theorem :
  let length : ℝ := 60
  let width : ℝ := length / 2
  garden_perimeter length width = 180 := by
sorry

end NUMINAMATH_CALUDE_garden_fencing_theorem_l1515_151559


namespace NUMINAMATH_CALUDE_parabola_c_value_l1515_151595

/-- A parabola passing through two given points has a specific c-value -/
theorem parabola_c_value (b c : ℝ) : 
  (∀ x, 2 = x^2 + b*x + c → x = 1 ∨ x = 5) →
  c = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1515_151595


namespace NUMINAMATH_CALUDE_kg_to_ton_conversion_min_to_hour_conversion_kg_to_g_conversion_l1515_151586

-- Define conversion rates
def kg_to_ton : ℝ := 1000
def min_to_hour : ℝ := 60
def kg_to_g : ℝ := 1000

-- Theorem statements
theorem kg_to_ton_conversion : 56 / kg_to_ton = 0.056 := by sorry

theorem min_to_hour_conversion : 45 / min_to_hour = 0.75 := by sorry

theorem kg_to_g_conversion : 0.3 * kg_to_g = 300 := by sorry

end NUMINAMATH_CALUDE_kg_to_ton_conversion_min_to_hour_conversion_kg_to_g_conversion_l1515_151586


namespace NUMINAMATH_CALUDE_even_and_mono_decreasing_implies_ordering_l1515_151554

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define a monotonically decreasing function on an interval
def MonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- Main theorem
theorem even_and_mono_decreasing_implies_ordering (f : ℝ → ℝ)
  (h1 : EvenFunction f)
  (h2 : MonoDecreasing (fun x => f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 := by
  sorry

end NUMINAMATH_CALUDE_even_and_mono_decreasing_implies_ordering_l1515_151554


namespace NUMINAMATH_CALUDE_ellipse_to_circle_transformation_l1515_151504

/-- Proves that the given scaling transformation transforms the ellipse into the circle -/
theorem ellipse_to_circle_transformation (x y x' y' : ℝ) :
  (x'^2 / 10 + y'^2 / 8 = 1) →
  (x' = (Real.sqrt 10 / 5) * x ∧ y' = (Real.sqrt 2 / 2) * y) →
  (x^2 + y^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_to_circle_transformation_l1515_151504


namespace NUMINAMATH_CALUDE_suzy_jump_ropes_l1515_151514

theorem suzy_jump_ropes (yesterday : ℕ) (additional : ℕ) : 
  yesterday = 247 → additional = 131 → yesterday + (yesterday + additional) = 625 := by
  sorry

end NUMINAMATH_CALUDE_suzy_jump_ropes_l1515_151514


namespace NUMINAMATH_CALUDE_rug_inner_length_is_three_l1515_151503

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : RectDimensions) : ℝ := d.length * d.width

/-- Represents the rug with its three regions -/
structure Rug where
  inner : RectDimensions
  middle : RectDimensions
  outer : RectDimensions

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem rug_inner_length_is_three (r : Rug) :
  r.inner.width = 2 →
  r.middle.length = r.inner.length + 3 →
  r.middle.width = r.inner.width + 3 →
  r.outer.length = r.middle.length + 3 →
  r.outer.width = r.middle.width + 3 →
  isArithmeticProgression (area r.inner) (area r.middle) (area r.outer) →
  r.inner.length = 3 := by
  sorry

end NUMINAMATH_CALUDE_rug_inner_length_is_three_l1515_151503


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l1515_151553

theorem bernoulli_inequality (h : ℝ) (hgt : h > -1) :
  (∀ x > 1, (1 + h)^x > 1 + h*x) ∧
  (∀ x < 0, (1 + h)^x > 1 + h*x) ∧
  (∀ x, 0 < x → x < 1 → (1 + h)^x < 1 + h*x) := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l1515_151553


namespace NUMINAMATH_CALUDE_cube_sum_equals_94_l1515_151582

theorem cube_sum_equals_94 (a b c : ℝ) 
  (h1 : a + b + c = 7) 
  (h2 : a * b + a * c + b * c = 11) 
  (h3 : a * b * c = -6) : 
  a^3 + b^3 + c^3 = 94 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_94_l1515_151582


namespace NUMINAMATH_CALUDE_factor_tree_proof_l1515_151591

theorem factor_tree_proof (A B C D E : ℕ) 
  (hB : B = 4 * D)
  (hC : C = 7 * E)
  (hA : A = B * C)
  (hD : D = 4 * 3)
  (hE : E = 7 * 3) :
  A = 7056 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_proof_l1515_151591


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l1515_151537

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_transitivity
  (α β γ : Plane) (m n : Line)
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h_diff_lines : m ≠ n)
  (h1 : perp n α)
  (h2 : perp n β)
  (h3 : perp m α)
  : perp m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l1515_151537


namespace NUMINAMATH_CALUDE_bugs_meeting_time_l1515_151532

/-- Two circles tangent at point O with radii 7 and 3 inches, and bugs moving at 4π and 3π inches per minute respectively -/
structure CircleSetup where
  r1 : ℝ
  r2 : ℝ
  v1 : ℝ
  v2 : ℝ
  h_r1 : r1 = 7
  h_r2 : r2 = 3
  h_v1 : v1 = 4 * Real.pi
  h_v2 : v2 = 3 * Real.pi

/-- Time taken for bugs to meet again at point O -/
def meetingTime (setup : CircleSetup) : ℝ :=
  7

/-- Theorem stating that the meeting time is 7 minutes -/
theorem bugs_meeting_time (setup : CircleSetup) :
  meetingTime setup = 7 := by
  sorry

end NUMINAMATH_CALUDE_bugs_meeting_time_l1515_151532


namespace NUMINAMATH_CALUDE_probability_three_heads_is_one_eighth_l1515_151508

/-- Represents the possible outcomes of a coin flip -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the set of five coins -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (halfDollar : CoinOutcome)

/-- The probability of the penny, dime, and quarter all coming up heads -/
def probabilityThreeHeads : ℚ := 1 / 8

/-- Theorem stating that the probability of the penny, dime, and quarter
    all coming up heads when flipping five coins is 1/8 -/
theorem probability_three_heads_is_one_eighth :
  probabilityThreeHeads = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_probability_three_heads_is_one_eighth_l1515_151508


namespace NUMINAMATH_CALUDE_unique_polynomial_with_integer_root_l1515_151536

theorem unique_polynomial_with_integer_root :
  ∃! (a : ℕ+), ∃ (x : ℤ), x^2 - (a : ℤ) * x + (a : ℤ) = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_polynomial_with_integer_root_l1515_151536


namespace NUMINAMATH_CALUDE_integral_sin_over_square_l1515_151550

open Real MeasureTheory

/-- The definite integral of sin(x) / (1 + cos(x) + sin(x))^2 from 0 to π/2 equals ln(2) - 1/2 -/
theorem integral_sin_over_square : ∫ x in (0)..(π/2), sin x / (1 + cos x + sin x)^2 = log 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin_over_square_l1515_151550


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1515_151561

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1515_151561


namespace NUMINAMATH_CALUDE_european_scientist_ratio_l1515_151522

theorem european_scientist_ratio (total : ℕ) (usa : ℕ) (canada : ℚ) : 
  total = 70 →
  usa = 21 →
  canada = 1/5 →
  (total - (canada * total).num - usa) / total = 1/2 := by
sorry

end NUMINAMATH_CALUDE_european_scientist_ratio_l1515_151522


namespace NUMINAMATH_CALUDE_ac_length_l1515_151599

/-- A quadrilateral with diagonals intersecting at O --/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)
  (OA : ℝ)
  (OC : ℝ)
  (OD : ℝ)
  (OB : ℝ)
  (BD : ℝ)
  (hOA : dist O A = OA)
  (hOC : dist O C = OC)
  (hOD : dist O D = OD)
  (hOB : dist O B = OB)
  (hBD : dist B D = BD)

/-- The theorem stating the length of AC in the given quadrilateral --/
theorem ac_length (q : Quadrilateral) 
  (h1 : q.OA = 6)
  (h2 : q.OC = 9)
  (h3 : q.OD = 6)
  (h4 : q.OB = 7)
  (h5 : q.BD = 10) :
  dist q.A q.C = 11.5 := by sorry

end NUMINAMATH_CALUDE_ac_length_l1515_151599


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l1515_151596

/-- The perimeter of a region formed by four 90° arcs of circles with circumference 48 -/
theorem shaded_region_perimeter (c : ℝ) (h : c = 48) : 
  4 * (90 / 360 * c) = 48 := by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l1515_151596


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l1515_151515

theorem sqrt_difference_equality : Real.sqrt (64 + 36) - Real.sqrt (81 - 64) = 10 - Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l1515_151515


namespace NUMINAMATH_CALUDE_cube_volume_l1515_151581

theorem cube_volume (n : ℝ) : 
  (∃ (s : ℝ), s * Real.sqrt 2 = 4 ∧ s^3 = n * Real.sqrt 2) → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l1515_151581


namespace NUMINAMATH_CALUDE_fraction_unchanged_l1515_151569

theorem fraction_unchanged (x y : ℝ) (h : x + y ≠ 0) :
  (3 * (2 * y)) / (2 * x + 2 * y) = (3 * y) / (x + y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l1515_151569


namespace NUMINAMATH_CALUDE_parabola_intersection_l1515_151593

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 9 * x - 8
  let g (x : ℝ) := x^2 - 3 * x + 4
  (f 3 = g 3 ∧ f 3 = -8) ∧ (f (-2) = g (-2) ∧ f (-2) = 22) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1515_151593


namespace NUMINAMATH_CALUDE_product_prs_l1515_151509

theorem product_prs (p r s : ℕ) : 
  4^p + 4^3 = 320 → 
  3^r + 27 = 108 → 
  2^s + 7^4 = 2617 → 
  p * r * s = 112 := by
sorry

end NUMINAMATH_CALUDE_product_prs_l1515_151509


namespace NUMINAMATH_CALUDE_dog_group_arrangements_count_l1515_151538

/-- The number of ways to divide 12 dogs into three groups -/
def dog_group_arrangements : ℕ :=
  let total_dogs : ℕ := 12
  let group_1_size : ℕ := 4
  let group_2_size : ℕ := 6
  let group_3_size : ℕ := 2
  let dogs_to_distribute : ℕ := total_dogs - 2  -- Fluffy and Nipper are pre-assigned
  let remaining_group_1_size : ℕ := group_1_size - 1  -- Fluffy is already in group 1
  let remaining_group_2_size : ℕ := group_2_size - 1  -- Nipper is already in group 2
  (Nat.choose dogs_to_distribute remaining_group_1_size) * 
  (Nat.choose (dogs_to_distribute - remaining_group_1_size) remaining_group_2_size)

/-- Theorem stating the number of ways to arrange the dogs -/
theorem dog_group_arrangements_count : dog_group_arrangements = 2520 := by
  sorry

end NUMINAMATH_CALUDE_dog_group_arrangements_count_l1515_151538


namespace NUMINAMATH_CALUDE_epsilon_delta_condition_l1515_151527

def f (x : ℝ) := x^2 + 1

theorem epsilon_delta_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x - 1| < b → |f x - 3| < a) ↔ b ≤ a / 2 := by
  sorry

end NUMINAMATH_CALUDE_epsilon_delta_condition_l1515_151527


namespace NUMINAMATH_CALUDE_final_game_score_l1515_151579

/-- Represents the points scored by each player in the basketball game -/
structure PlayerPoints where
  bailey : ℕ
  michiko : ℕ
  akiko : ℕ
  chandra : ℕ

/-- Calculates the total points scored by the team -/
def total_points (p : PlayerPoints) : ℕ :=
  p.bailey + p.michiko + p.akiko + p.chandra

/-- Proves that the team scored 54 points in the final game -/
theorem final_game_score :
  ∃ (p : PlayerPoints),
    p.bailey = 14 ∧
    p.michiko = p.bailey / 2 ∧
    p.akiko = p.michiko + 4 ∧
    p.chandra = 2 * p.akiko ∧
    total_points p = 54 := by
  sorry

end NUMINAMATH_CALUDE_final_game_score_l1515_151579


namespace NUMINAMATH_CALUDE_unique_prime_twice_square_l1515_151516

theorem unique_prime_twice_square : 
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (x y : ℕ), p + 1 = 2 * x^2 ∧ p^2 + 1 = 2 * y^2) ∧ 
    p = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_twice_square_l1515_151516


namespace NUMINAMATH_CALUDE_factorial_ratio_l1515_151575

theorem factorial_ratio : Nat.factorial 45 / Nat.factorial 42 = 85140 := by sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1515_151575


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_sixth_l1515_151567

theorem reciprocal_of_negative_one_sixth : 
  ((-1 / 6 : ℚ)⁻¹ : ℚ) = -6 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_sixth_l1515_151567


namespace NUMINAMATH_CALUDE_function_satisfying_condition_l1515_151587

theorem function_satisfying_condition (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, |f x - f y| = 2 * |x - y|) :
  ∃ c : ℝ, (∀ x : ℝ, f x = 2 * x + c) ∨ (∀ x : ℝ, f x = -2 * x + c) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_condition_l1515_151587


namespace NUMINAMATH_CALUDE_min_value_expression_l1515_151530

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1515_151530


namespace NUMINAMATH_CALUDE_range_of_m_l1515_151525

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - |x| else x^2 - 4*x + 3

theorem range_of_m (m : ℝ) : f (f m) ≥ 0 → m ∈ Set.Icc (-2) (2 + Real.sqrt 2) ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1515_151525


namespace NUMINAMATH_CALUDE_larger_number_proof_l1515_151513

theorem larger_number_proof (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1515_151513


namespace NUMINAMATH_CALUDE_checkerboard_sum_l1515_151510

/-- The number of rectangles in a 7x7 checkerboard -/
def r' : ℕ := 784

/-- The number of squares in a 7x7 checkerboard -/
def s' : ℕ := 140

/-- m' and n' are relatively prime positive integers such that s'/r' = m'/n' -/
def m' : ℕ := 5
def n' : ℕ := 28

theorem checkerboard_sum : m' + n' = 33 := by sorry

end NUMINAMATH_CALUDE_checkerboard_sum_l1515_151510


namespace NUMINAMATH_CALUDE_alvin_wood_gathering_l1515_151558

theorem alvin_wood_gathering (total_needed wood_from_friend wood_from_brother : ℕ) 
  (h1 : total_needed = 376)
  (h2 : wood_from_friend = 123)
  (h3 : wood_from_brother = 136) :
  total_needed - (wood_from_friend + wood_from_brother) = 117 := by
  sorry

end NUMINAMATH_CALUDE_alvin_wood_gathering_l1515_151558


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l1515_151517

theorem quadratic_two_roots (a b c : ℝ) (h1 : 2016 + a^2 + a*c < a*b) (h2 : a ≠ 0) :
  ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l1515_151517


namespace NUMINAMATH_CALUDE_remuneration_problem_l1515_151529

/-- Represents the remuneration problem -/
theorem remuneration_problem (annual_clothing : ℕ) (annual_coins : ℕ) 
  (months_worked : ℕ) (received_clothing : ℕ) (received_coins : ℕ) :
  annual_clothing = 1 →
  annual_coins = 10 →
  months_worked = 7 →
  received_clothing = 1 →
  received_coins = 2 →
  ∃ (clothing_value : ℚ),
    clothing_value = 46 / 5 ∧
    (clothing_value + annual_coins : ℚ) / 12 = (clothing_value + received_coins) / months_worked :=
by sorry

end NUMINAMATH_CALUDE_remuneration_problem_l1515_151529


namespace NUMINAMATH_CALUDE_middle_school_students_l1515_151534

theorem middle_school_students (band_percentage : ℝ) (band_students : ℕ) (total_students : ℕ) : 
  band_percentage = 0.20 →
  band_students = 168 →
  (band_percentage * total_students : ℝ) = band_students →
  total_students = 840 := by
sorry

end NUMINAMATH_CALUDE_middle_school_students_l1515_151534


namespace NUMINAMATH_CALUDE_determine_y_from_one_point_determine_y_from_k_one_additional_data_necessary_and_sufficient_l1515_151592

/-- A structure representing a proportional relationship between x and y --/
structure ProportionalRelationship where
  k : ℝ  -- Constant of proportionality
  proportional : ∀ (x y : ℝ), y = k * x

/-- Given a proportional relationship and one point, we can determine y for any x --/
theorem determine_y_from_one_point 
  (rel : ProportionalRelationship) (x₀ y₀ : ℝ) (h : y₀ = rel.k * x₀) :
  ∀ (x : ℝ), ∃! (y : ℝ), y = rel.k * x :=
sorry

/-- Given a proportional relationship and k, we can determine y for any x --/
theorem determine_y_from_k (rel : ProportionalRelationship) :
  ∀ (x : ℝ), ∃! (y : ℝ), y = rel.k * x :=
sorry

/-- One additional piece of data (either k or a point) is necessary and sufficient --/
theorem one_additional_data_necessary_and_sufficient :
  ∀ (x y : ℝ → ℝ), (∃ (k : ℝ), ∀ (t : ℝ), y t = k * x t) →
  ((∃ (k : ℝ), ∀ (t : ℝ), y t = k * x t) ∨ 
   (∃ (x₀ y₀ : ℝ), y x₀ = y₀ ∧ ∀ (t : ℝ), y t = (y₀ / x₀) * x t)) ∧
  (∀ (t : ℝ), ∃! (yt : ℝ), y t = yt) :=
sorry

end NUMINAMATH_CALUDE_determine_y_from_one_point_determine_y_from_k_one_additional_data_necessary_and_sufficient_l1515_151592


namespace NUMINAMATH_CALUDE_current_speed_l1515_151556

/-- The speed of the current given a motorboat's constant speed and trip times -/
theorem current_speed (boat_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) 
  (h1 : boat_speed = 30)
  (h2 : upstream_time = 40 / 60)
  (h3 : downstream_time = 25 / 60) :
  ∃ c : ℝ, c = 90 / 13 ∧ 
  (boat_speed - c) * upstream_time = (boat_speed + c) * downstream_time :=
sorry

end NUMINAMATH_CALUDE_current_speed_l1515_151556


namespace NUMINAMATH_CALUDE_farm_animal_ratio_l1515_151597

theorem farm_animal_ratio : 
  let goats : ℕ := 66
  let chickens : ℕ := 2 * goats
  let total_goats_chickens : ℕ := goats + chickens
  let ducks : ℕ := 99  -- We define this to match the problem constraints
  let pigs : ℕ := ducks / 3
  goats = pigs + 33 →
  (ducks : ℚ) / total_goats_chickens = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_farm_animal_ratio_l1515_151597


namespace NUMINAMATH_CALUDE_absolute_difference_x_y_l1515_151540

theorem absolute_difference_x_y (x y : ℝ) 
  (h1 : ⌊x⌋ + (y - ⌊y⌋) = 2.4)
  (h2 : (x - ⌊x⌋) + ⌊y⌋ = 5.1) : 
  |x - y| = 3.3 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_x_y_l1515_151540


namespace NUMINAMATH_CALUDE_outside_county_attendance_l1515_151580

/-- The number of kids from Lawrence county who went to camp -/
def lawrence_camp : ℕ := 34044

/-- The total number of kids who attended the camp -/
def total_camp : ℕ := 458988

/-- The number of kids from outside the county who attended the camp -/
def outside_county : ℕ := total_camp - lawrence_camp

theorem outside_county_attendance : outside_county = 424944 := by
  sorry

end NUMINAMATH_CALUDE_outside_county_attendance_l1515_151580


namespace NUMINAMATH_CALUDE_caiden_roofing_problem_l1515_151564

theorem caiden_roofing_problem (cost_per_foot : ℝ) (free_feet : ℝ) (remaining_cost : ℝ) :
  cost_per_foot = 8 →
  free_feet = 250 →
  remaining_cost = 400 →
  ∃ (total_feet : ℝ), total_feet = 300 ∧ (total_feet - free_feet) * cost_per_foot = remaining_cost :=
by sorry

end NUMINAMATH_CALUDE_caiden_roofing_problem_l1515_151564


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1515_151570

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∃ a b : ℝ, a = -1 ∧ b = 2 ∧ a * b = -2) ∧
  (∃ a b : ℝ, a * b = -2 ∧ (a ≠ -1 ∨ b ≠ 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1515_151570


namespace NUMINAMATH_CALUDE_function_property_l1515_151545

theorem function_property (f : ℝ → ℝ) (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x + f y) 
  (h2 : f 8 = -3) : ∃ a : ℝ, a > 0 ∧ f a = 1/2 ∧ a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1515_151545


namespace NUMINAMATH_CALUDE_similar_quadrilaterals_rectangle_areas_l1515_151578

/-- Given two similar quadrilaterals with sides (a, b, c, d) and (a', b', c', d') respectively,
    prove that the areas of rectangles formed by pairs of corresponding sides
    are in proportion to the squares of the sides of the original quadrilaterals. -/
theorem similar_quadrilaterals_rectangle_areas
  (a b c d a' b' c' d' : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0)
  (h_similar : ∃ (k : ℝ), k > 0 ∧ a' = k * a ∧ b' = k * b ∧ c' = k * c ∧ d' = k * d) :
  ∃ (m : ℝ), m > 0 ∧
    a * a' / (b * b') = a^2 / b^2 ∧
    b * b' / (c * c') = b^2 / c^2 ∧
    c * c' / (d * d') = c^2 / d^2 ∧
    d * d' / (a * a') = d^2 / a^2 :=
by sorry

end NUMINAMATH_CALUDE_similar_quadrilaterals_rectangle_areas_l1515_151578


namespace NUMINAMATH_CALUDE_function_properties_l1515_151531

/-- The function f(x) = x³ + 2ax² + bx + a -/
def f (a b x : ℝ) : ℝ := x^3 + 2*a*x^2 + b*x + a

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 4*a*x + b

theorem function_properties (a b : ℝ) :
  f a b (-1) = 1 ∧ f_derivative a b (-1) = 0 →
  a = 1 ∧ b = 1 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 1 x ≤ 5 ∧ f 1 1 1 = 5 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1515_151531


namespace NUMINAMATH_CALUDE_observation_probability_l1515_151560

theorem observation_probability 
  (total_students : Nat) 
  (total_periods : Nat) 
  (zi_shi_duration : Nat) 
  (total_duration : Nat) :
  total_students = 4 →
  total_periods = 4 →
  zi_shi_duration = 2 →
  total_duration = 8 →
  (zi_shi_duration : ℚ) / total_duration = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_observation_probability_l1515_151560


namespace NUMINAMATH_CALUDE_surface_area_cube_with_holes_l1515_151549

/-- The surface area of a cube with smaller cubes dug out from each face -/
theorem surface_area_cube_with_holes (edge_length : ℝ) (hole_length : ℝ) : 
  edge_length = 10 →
  hole_length = 2 →
  (6 * edge_length^2) - (6 * hole_length^2) + (6 * 5 * hole_length^2) = 696 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_cube_with_holes_l1515_151549


namespace NUMINAMATH_CALUDE_total_profit_is_135000_l1515_151584

/-- Represents an investor in the partnership business -/
structure Investor where
  name : String
  investment : ℕ
  months : ℕ

/-- Calculates the total profit given the investors and C's profit share -/
def calculateTotalProfit (investors : List Investor) (cProfit : ℕ) : ℕ :=
  let totalInvestmentMonths := investors.map (λ i => i.investment * i.months) |>.sum
  let cInvestmentMonths := (investors.find? (λ i => i.name = "C")).map (λ i => i.investment * i.months)
  match cInvestmentMonths with
  | some im => cProfit * totalInvestmentMonths / im
  | none => 0

/-- Theorem stating that the total profit is 135000 given the specified conditions -/
theorem total_profit_is_135000 (investors : List Investor) (h1 : investors.length = 5)
    (h2 : investors.any (λ i => i.name = "A" ∧ i.investment = 12000 ∧ i.months = 6))
    (h3 : investors.any (λ i => i.name = "B" ∧ i.investment = 16000 ∧ i.months = 12))
    (h4 : investors.any (λ i => i.name = "C" ∧ i.investment = 20000 ∧ i.months = 12))
    (h5 : investors.any (λ i => i.name = "D" ∧ i.investment = 24000 ∧ i.months = 12))
    (h6 : investors.any (λ i => i.name = "E" ∧ i.investment = 18000 ∧ i.months = 6))
    (h7 : calculateTotalProfit investors 36000 = 135000) : 
  calculateTotalProfit investors 36000 = 135000 := by
  sorry


end NUMINAMATH_CALUDE_total_profit_is_135000_l1515_151584


namespace NUMINAMATH_CALUDE_gasoline_spending_increase_l1515_151590

theorem gasoline_spending_increase (P Q : ℝ) (P_positive : P > 0) (Q_positive : Q > 0) :
  let new_price := 1.25 * P
  let new_quantity := 0.88 * Q
  let original_spending := P * Q
  let new_spending := new_price * new_quantity
  (new_spending - original_spending) / original_spending = 0.1 := by
sorry

end NUMINAMATH_CALUDE_gasoline_spending_increase_l1515_151590


namespace NUMINAMATH_CALUDE_subset_implies_a_value_l1515_151539

theorem subset_implies_a_value (A B : Set ℤ) (a : ℤ) :
  A = {0, 1} →
  B = {-1, 0, a + 3} →
  A ⊆ B →
  a = -2 := by sorry

end NUMINAMATH_CALUDE_subset_implies_a_value_l1515_151539


namespace NUMINAMATH_CALUDE_ada_was_in_seat_two_l1515_151520

/-- Represents the seats in the row --/
inductive Seat
  | one
  | two
  | three
  | four
  | five

/-- Represents the friends --/
inductive Friend
  | ada
  | bea
  | ceci
  | dee
  | edie

/-- Represents the seating arrangement --/
def Arrangement := Friend → Seat

/-- The initial seating arrangement --/
def initial_arrangement : Arrangement := sorry

/-- The final seating arrangement after all movements --/
def final_arrangement : Arrangement := sorry

/-- Bea moves one seat to the right --/
def bea_moves (arr : Arrangement) : Arrangement := sorry

/-- Ceci moves left and then back --/
def ceci_moves (arr : Arrangement) : Arrangement := sorry

/-- Dee and Edie switch seats, then Edie moves right --/
def dee_edie_move (arr : Arrangement) : Arrangement := sorry

/-- Ada's original seat --/
def ada_original_seat : Seat := sorry

theorem ada_was_in_seat_two :
  ada_original_seat = Seat.two ∧
  final_arrangement = dee_edie_move (ceci_moves (bea_moves initial_arrangement)) ∧
  (final_arrangement Friend.ada = Seat.one ∨ final_arrangement Friend.ada = Seat.five) :=
sorry

end NUMINAMATH_CALUDE_ada_was_in_seat_two_l1515_151520


namespace NUMINAMATH_CALUDE_tom_nail_purchase_l1515_151533

/-- The number of additional nails Tom needs to buy for his project -/
def additional_nails_needed (initial : ℝ) (toolshed : ℝ) (drawer : ℝ) (neighbor : ℝ) (thank_you : ℝ) (required : ℝ) : ℝ :=
  required - (initial + toolshed + drawer + neighbor + thank_you)

/-- Theorem stating the number of additional nails Tom needs to buy -/
theorem tom_nail_purchase (initial : ℝ) (toolshed : ℝ) (drawer : ℝ) (neighbor : ℝ) (thank_you : ℝ) (required : ℝ)
    (h1 : initial = 247.5)
    (h2 : toolshed = 144.25)
    (h3 : drawer = 0.75)
    (h4 : neighbor = 58.75)
    (h5 : thank_you = 37.25)
    (h6 : required = 761.58) :
    additional_nails_needed initial toolshed drawer neighbor thank_you required = 273.08 := by
  sorry

end NUMINAMATH_CALUDE_tom_nail_purchase_l1515_151533


namespace NUMINAMATH_CALUDE_bridge_length_l1515_151521

/-- Given a train crossing a bridge, this theorem calculates the length of the bridge. -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 170)
  (h2 : train_speed = 45 * 1000 / 3600)  -- Convert km/hr to m/s
  (h3 : crossing_time = 30) :
  train_speed * crossing_time - train_length = 205 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1515_151521


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_one_l1515_151589

theorem sum_of_roots_eq_one : 
  let f : ℝ → ℝ := λ x => (x + 3) * (x - 4) - 20
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_one_l1515_151589


namespace NUMINAMATH_CALUDE_alcohol_concentration_proof_l1515_151572

theorem alcohol_concentration_proof :
  ∀ (vessel1_capacity vessel2_capacity total_liquid final_capacity : ℝ)
    (vessel2_concentration final_concentration : ℝ),
  vessel1_capacity = 2 →
  vessel2_capacity = 6 →
  vessel2_concentration = 0.4 →
  total_liquid = 8 →
  final_capacity = 10 →
  final_concentration = 0.29000000000000004 →
  ∃ (vessel1_concentration : ℝ),
    vessel1_concentration = 0.25 ∧
    vessel1_concentration * vessel1_capacity + vessel2_concentration * vessel2_capacity =
      final_concentration * final_capacity :=
by
  sorry

#check alcohol_concentration_proof

end NUMINAMATH_CALUDE_alcohol_concentration_proof_l1515_151572


namespace NUMINAMATH_CALUDE_problem_statement_problem_statement_2_l1515_151512

/-- The quadratic function used in the problem -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 8

/-- Proposition p: x^2 - 2x - 8 ≤ 0 -/
def p (x : ℝ) : Prop := f x ≤ 0

/-- Proposition q: 2 - m ≤ x ≤ 2 + m -/
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

theorem problem_statement (m : ℝ) (h : m > 0) :
  (∀ x, p x → q m x) ∧ (∃ x, q m x ∧ ¬p x) → m ≥ 4 :=
sorry

theorem problem_statement_2 (x : ℝ) :
  let m := 5
  (p x ∨ q m x) ∧ ¬(p x ∧ q m x) →
  (-3 ≤ x ∧ x < -2) ∨ (4 < x ∧ x ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_problem_statement_2_l1515_151512


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l1515_151552

theorem quadratic_root_m_value : ∀ m : ℝ,
  ((-1 : ℝ)^2 + m * (-1) - 1 = 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l1515_151552


namespace NUMINAMATH_CALUDE_borrowing_methods_count_l1515_151566

/-- Represents the number of books of each type -/
structure BookCounts where
  physics : Nat
  history : Nat
  mathematics : Nat

/-- Represents the number of students of each type -/
structure StudentCounts where
  science : Nat
  liberal_arts : Nat

/-- Calculates the number of ways to distribute books to students -/
def calculate_borrowing_methods (books : BookCounts) (students : StudentCounts) : Nat :=
  sorry

/-- Theorem stating the correct number of borrowing methods -/
theorem borrowing_methods_count :
  let books := BookCounts.mk 3 2 4
  let students := StudentCounts.mk 4 3
  calculate_borrowing_methods books students = 76 := by
  sorry

end NUMINAMATH_CALUDE_borrowing_methods_count_l1515_151566


namespace NUMINAMATH_CALUDE_kelly_initial_games_l1515_151547

/-- The number of games Kelly gave away -/
def games_given_away : ℕ := 91

/-- The number of games Kelly has left -/
def games_left : ℕ := 92

/-- The initial number of games Kelly had -/
def initial_games : ℕ := games_given_away + games_left

theorem kelly_initial_games : initial_games = 183 := by
  sorry

end NUMINAMATH_CALUDE_kelly_initial_games_l1515_151547


namespace NUMINAMATH_CALUDE_polygon_sides_l1515_151568

theorem polygon_sides (n : ℕ) (sum_angles : ℝ) : 
  n > 2 ∧ sum_angles = 2190 ∧ sum_angles = (n - 3) * 180 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1515_151568


namespace NUMINAMATH_CALUDE_tatiana_age_l1515_151511

/-- Calculates the total full years given an age in years, months, weeks, days, and hours -/
def calculate_full_years (years months weeks days hours : ℕ) : ℕ :=
  let months_to_years := months / 12
  let weeks_to_years := weeks / 52
  let days_to_years := days / 365
  let hours_to_years := hours / (24 * 365)
  years + months_to_years + weeks_to_years + days_to_years + hours_to_years

/-- Theorem stating that the age of 72 years, 72 months, 72 weeks, 72 days, and 72 hours is equivalent to 79 full years -/
theorem tatiana_age : calculate_full_years 72 72 72 72 72 = 79 := by
  sorry

end NUMINAMATH_CALUDE_tatiana_age_l1515_151511


namespace NUMINAMATH_CALUDE_chord_length_squared_l1515_151506

/-- The square of the length of a chord that is a common external tangent to two circles -/
theorem chord_length_squared (r₁ r₂ R : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : R > 0)
  (h₄ : r₁ + r₂ < R) : 
  let d := R - (r₁ + r₂) + Real.sqrt (r₁ * r₂)
  4 * (R^2 - d^2) = 516 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_squared_l1515_151506


namespace NUMINAMATH_CALUDE_divide_into_triominoes_l1515_151548

/-- An L-shaped triomino is a shape consisting of three connected cells in an L shape -/
def LShapedTriomino : Type := Unit

/-- A grid is represented by its size, which is always of the form 6n+1 for some natural number n -/
structure Grid :=
  (n : ℕ)

/-- A cell in the grid, represented by its row and column coordinates -/
structure Cell :=
  (row : ℕ)
  (col : ℕ)

/-- A configuration is a grid with one cell removed -/
structure Configuration :=
  (grid : Grid)
  (removed_cell : Cell)

/-- A division of a configuration into L-shaped triominoes -/
def Division (config : Configuration) : Type := Unit

/-- The main theorem: any configuration can be divided into L-shaped triominoes -/
theorem divide_into_triominoes (config : Configuration) : 
  ∃ (d : Division config), True :=
sorry

end NUMINAMATH_CALUDE_divide_into_triominoes_l1515_151548


namespace NUMINAMATH_CALUDE_graces_age_l1515_151546

/-- Grace's age problem -/
theorem graces_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ) : 
  mother_age = 80 →
  grandmother_age = 2 * mother_age →
  grace_age = (3 * grandmother_age) / 8 →
  grace_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_graces_age_l1515_151546


namespace NUMINAMATH_CALUDE_octahedron_ant_path_probability_l1515_151565

/-- Represents a vertex in the octahedron --/
inductive Vertex
| Top
| Bottom
| Middle1
| Middle2
| Middle3
| Middle4

/-- Represents an octahedron --/
structure Octahedron where
  vertices : List Vertex
  edges : List (Vertex × Vertex)
  is_regular : Bool

/-- Represents the ant's path --/
structure AntPath where
  start : Vertex
  a : Vertex
  b : Vertex
  c : Vertex

/-- Function to check if a vertex is in the middle ring --/
def is_middle_ring (v : Vertex) : Bool :=
  match v with
  | Vertex.Middle1 | Vertex.Middle2 | Vertex.Middle3 | Vertex.Middle4 => true
  | _ => false

/-- Function to get adjacent vertices --/
def get_adjacent_vertices (o : Octahedron) (v : Vertex) : List Vertex :=
  sorry

/-- Function to calculate the probability of returning to the start --/
def return_probability (o : Octahedron) (path : AntPath) : Rat :=
  sorry

theorem octahedron_ant_path_probability (o : Octahedron) (path : AntPath) :
  o.is_regular = true →
  is_middle_ring path.start = true →
  path.a ∈ get_adjacent_vertices o path.start →
  path.b ∈ get_adjacent_vertices o path.a →
  path.c ∈ get_adjacent_vertices o path.b →
  return_probability o path = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_octahedron_ant_path_probability_l1515_151565


namespace NUMINAMATH_CALUDE_cube_adjacent_diagonals_perpendicular_l1515_151588

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the internal structure of a cube for this problem

/-- A face diagonal is a line segment that connects opposite corners of a face -/
structure FaceDiagonal where
  cube : Cube
  face : Nat  -- We can use natural numbers to identify faces (1 to 6)

/-- The angle between two face diagonals -/
def angle_between_diagonals (d1 d2 : FaceDiagonal) : ℝ := sorry

/-- Two faces of a cube are adjacent if they share an edge -/
def adjacent_faces (f1 f2 : Nat) : Prop := sorry

/-- Theorem: The angle between the diagonals of any two adjacent faces of a cube is 90 degrees -/
theorem cube_adjacent_diagonals_perpendicular (c : Cube) (f1 f2 : Nat) (d1 d2 : FaceDiagonal)
  (h1 : d1.cube = c) (h2 : d2.cube = c) (h3 : d1.face = f1) (h4 : d2.face = f2)
  (h5 : adjacent_faces f1 f2) :
  angle_between_diagonals d1 d2 = 90 := by sorry

end NUMINAMATH_CALUDE_cube_adjacent_diagonals_perpendicular_l1515_151588


namespace NUMINAMATH_CALUDE_balance_scale_theorem_l1515_151541

/-- Represents a weight on the balance scale -/
structure Weight where
  pan : Bool  -- true for left pan, false for right pan
  value : ℝ
  number : ℕ

/-- Represents the state of the balance scale -/
structure BalanceScale where
  k : ℕ  -- number of weights on each pan
  weights : List Weight

/-- Checks if the left pan is heavier -/
def leftPanHeavier (scale : BalanceScale) : Prop :=
  let leftSum := (scale.weights.filter (fun w => w.pan)).map (fun w => w.value) |>.sum
  let rightSum := (scale.weights.filter (fun w => !w.pan)).map (fun w => w.value) |>.sum
  leftSum > rightSum

/-- Checks if swapping weights with the same number makes the right pan heavier or balances the pans -/
def swapMakesRightHeavierOrBalance (scale : BalanceScale) : Prop :=
  ∀ i, i ≤ scale.k →
    let swappedWeights := scale.weights.map (fun w => if w.number = i then { w with pan := !w.pan } else w)
    let swappedLeftSum := (swappedWeights.filter (fun w => w.pan)).map (fun w => w.value) |>.sum
    let swappedRightSum := (swappedWeights.filter (fun w => !w.pan)).map (fun w => w.value) |>.sum
    swappedRightSum ≥ swappedLeftSum

/-- The main theorem stating that k can only be 1 or 2 -/
theorem balance_scale_theorem (scale : BalanceScale) :
  leftPanHeavier scale →
  swapMakesRightHeavierOrBalance scale →
  scale.k = 1 ∨ scale.k = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_balance_scale_theorem_l1515_151541


namespace NUMINAMATH_CALUDE_spring_equation_l1515_151535

theorem spring_equation (RI G SP T M N : ℤ) (L : ℚ) : 
  RI + G + SP = 50 ∧
  RI + T + M = 63 ∧
  G + T + SP = 25 ∧
  SP + M = 13 ∧
  M + RI = 48 ∧
  N = 1 →
  L * M * T + SP * RI * N * G = 2023 →
  L = 341 / 40 := by
sorry

end NUMINAMATH_CALUDE_spring_equation_l1515_151535


namespace NUMINAMATH_CALUDE_min_value_sum_of_fractions_l1515_151543

theorem min_value_sum_of_fractions (x y a b : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : a > 0) (h4 : b > 0) (h5 : x + y = 1) :
  a / x + b / y ≥ (Real.sqrt a + Real.sqrt b)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_fractions_l1515_151543


namespace NUMINAMATH_CALUDE_unique_odd_number_with_congruences_l1515_151523

theorem unique_odd_number_with_congruences : ∃! x : ℕ,
  500 < x ∧ x < 1000 ∧
  x % 25 = 6 ∧
  x % 9 = 7 ∧
  Odd x ∧
  x = 781 := by
  sorry

end NUMINAMATH_CALUDE_unique_odd_number_with_congruences_l1515_151523


namespace NUMINAMATH_CALUDE_max_value_part_i_one_root_condition_part_ii_inequality_condition_part_iii_l1515_151583

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1
def g (x : ℝ) : ℝ := Real.exp x

-- Part I
theorem max_value_part_i :
  ∃ (M : ℝ), M = 1 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 0, (f 1 x) * (g x) ≤ M :=
sorry

-- Part II
theorem one_root_condition_part_ii :
  ∀ k : ℝ, (∃! x : ℝ, f (-1) x = k * g x) ↔ 
  (k > 0 ∧ k < Real.exp (-1)) ∨ (k > 3 * Real.exp (-2)) :=
sorry

-- Part III
theorem inequality_condition_part_iii :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → x₁ ≠ x₂ → 
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) ↔ 
  (a ≥ -1 ∧ a ≤ 2 - 2 * Real.log 2) :=
sorry

end

end NUMINAMATH_CALUDE_max_value_part_i_one_root_condition_part_ii_inequality_condition_part_iii_l1515_151583


namespace NUMINAMATH_CALUDE_library_books_count_l1515_151519

/-- Given a library with identical bookcases, prove the total number of books -/
theorem library_books_count (num_bookcases : ℕ) (shelves_per_bookcase : ℕ) (books_per_shelf : ℕ) :
  num_bookcases = 28 →
  shelves_per_bookcase = 6 →
  books_per_shelf = 19 →
  num_bookcases * shelves_per_bookcase * books_per_shelf = 3192 :=
by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l1515_151519


namespace NUMINAMATH_CALUDE_bisecting_line_exists_unique_l1515_151502

/-- A triangle with sides of length 6, 8, and 10 units. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10

/-- A line that intersects two sides of the triangle. -/
structure BisectingLine (T : Triangle) where
  x : ℝ  -- Intersection point on side b
  y : ℝ  -- Intersection point on side c
  hx : 0 < x ∧ x < T.b
  hy : 0 < y ∧ y < T.c

/-- The bisecting line divides the perimeter in half. -/
def bisects_perimeter (T : Triangle) (L : BisectingLine T) : Prop :=
  L.x + L.y = (T.a + T.b + T.c) / 2

/-- The bisecting line divides the area in half. -/
def bisects_area (T : Triangle) (L : BisectingLine T) : Prop :=
  L.x * L.y = (T.a * T.b) / 4

/-- The main theorem: existence and uniqueness of the bisecting line. -/
theorem bisecting_line_exists_unique (T : Triangle) :
  ∃! L : BisectingLine T, bisects_perimeter T L ∧ bisects_area T L :=
sorry

end NUMINAMATH_CALUDE_bisecting_line_exists_unique_l1515_151502


namespace NUMINAMATH_CALUDE_contradiction_proof_l1515_151500

theorem contradiction_proof (a b c : ℝ) 
  (h1 : a + b + c > 0) 
  (h2 : a * b + b * c + a * c > 0) 
  (h3 : a * b * c > 0) : 
  ¬(a > 0 ∧ b > 0 ∧ c > 0) ↔ (a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_contradiction_proof_l1515_151500


namespace NUMINAMATH_CALUDE_last_segment_speed_l1515_151574

theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : total_distance = 96)
  (h2 : total_time = 90 / 60)
  (h3 : speed1 = 60)
  (h4 : speed2 = 65)
  (h5 : (speed1 + speed2 + (3 * total_distance / total_time - speed1 - speed2)) / 3 = total_distance / total_time) :
  3 * total_distance / total_time - speed1 - speed2 = 67 := by
  sorry

end NUMINAMATH_CALUDE_last_segment_speed_l1515_151574
