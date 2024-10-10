import Mathlib

namespace ratio_d_b_is_negative_four_l84_8451

def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem ratio_d_b_is_negative_four
  (a b c d : ℝ)
  (h_even : ∀ x, f a b c d x = f a b c d (-x))
  (h_solution : ∀ x, f a b c d x < 0 ↔ -2 < x ∧ x < 2) :
  d / b = -4 := by
  sorry

end ratio_d_b_is_negative_four_l84_8451


namespace parallelogram_area_example_l84_8431

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 26 cm and height 14 cm is 364 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 26 14 = 364 := by
  sorry

end parallelogram_area_example_l84_8431


namespace gcd_factorial_seven_eight_l84_8484

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcd_factorial_seven_eight_l84_8484


namespace min_max_problem_l84_8426

theorem min_max_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (8 / x + 2 / y ≥ 18) ∧ (Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) ≤ 2 * Real.sqrt 2) := by
  sorry

end min_max_problem_l84_8426


namespace students_not_enrolled_l84_8495

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ)
  (h1 : total = 120)
  (h2 : french = 65)
  (h3 : german = 50)
  (h4 : both = 25) :
  total - (french + german - both) = 30 := by
  sorry

end students_not_enrolled_l84_8495


namespace inequality_equality_l84_8470

theorem inequality_equality (x : ℝ) : 
  x > 0 → (x * Real.sqrt (16 - x^2) + Real.sqrt (16*x - x^4) ≥ 16 ↔ x = 2 * Real.sqrt 2) := by
  sorry

end inequality_equality_l84_8470


namespace solution_set_of_inequality_l84_8400

theorem solution_set_of_inequality (x : ℝ) :
  (x - 2) / (x + 3) > 0 ↔ x ∈ Set.Ioi (-3) ∪ Set.Ioi 2 := by
  sorry

end solution_set_of_inequality_l84_8400


namespace minimize_sum_distances_l84_8493

/-- A type representing points on a line -/
structure Point where
  x : ℝ

/-- The distance between two points on a line -/
def distance (p q : Point) : ℝ := |p.x - q.x|

/-- The sum of distances from a point to a list of points -/
def sum_distances (q : Point) (points : List Point) : ℝ :=
  points.foldl (fun sum p => sum + distance p q) 0

theorem minimize_sum_distances 
  (p₁ p₂ p₃ p₄ p₅ p₆ p₇ p₈ : Point)
  (h : p₁.x < p₂.x ∧ p₂.x < p₃.x ∧ p₃.x < p₄.x ∧ p₄.x < p₅.x ∧ p₅.x < p₆.x ∧ p₆.x < p₇.x ∧ p₇.x < p₈.x) :
  ∃ (q : Point), 
    (∀ (r : Point), sum_distances q [p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈] ≤ sum_distances r [p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈]) ∧
    q.x = (p₄.x + p₅.x) / 2 := by
  sorry


end minimize_sum_distances_l84_8493


namespace fraction_repetend_l84_8479

/-- The repetend of the decimal representation of 7/29 -/
def repetend : Nat := 241379

/-- The length of the repetend -/
def repetend_length : Nat := 6

/-- The fraction we're considering -/
def fraction : Rat := 7 / 29

theorem fraction_repetend :
  ∃ (k : ℕ), (fraction * 10^repetend_length - fraction) * 10^k = repetend / (10^repetend_length - 1) :=
sorry

end fraction_repetend_l84_8479


namespace tan_difference_pi_12_pi_6_l84_8417

theorem tan_difference_pi_12_pi_6 : 
  Real.tan (π / 12) - Real.tan (π / 6) = 7 - 4 * Real.sqrt 3 := by
  sorry

end tan_difference_pi_12_pi_6_l84_8417


namespace paving_company_calculation_l84_8475

/-- Represents the properties of a street paved with cement -/
structure Street where
  length : Real
  width : Real
  thickness : Real
  cement_used : Real

/-- Calculates the volume of cement used for a street -/
def cement_volume (s : Street) : Real :=
  s.length * s.width * s.thickness

/-- Cement density in tons per cubic meter -/
def cement_density : Real := 1

theorem paving_company_calculation (lexi_street tess_street : Street) 
  (h1 : lexi_street.length = 200)
  (h2 : lexi_street.width = 10)
  (h3 : lexi_street.thickness = 0.1)
  (h4 : lexi_street.cement_used = 10)
  (h5 : tess_street.length = 100)
  (h6 : tess_street.thickness = 0.1)
  (h7 : tess_street.cement_used = 5.1) :
  tess_street.width = 0.51 ∧ lexi_street.cement_used + tess_street.cement_used = 15.1 := by
  sorry


end paving_company_calculation_l84_8475


namespace perpendicular_slope_l84_8427

/-- Given two points (3, -4) and (-2, 5) on a line, the slope of a line perpendicular to this line is 5/9. -/
theorem perpendicular_slope : 
  let p1 : ℝ × ℝ := (3, -4)
  let p2 : ℝ × ℝ := (-2, 5)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  (- (1 / m)) = 5 / 9 := by sorry

end perpendicular_slope_l84_8427


namespace triangle_with_specific_properties_l84_8440

/-- Represents a triangle with side lengths and circumradius -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  r : ℕ

/-- Represents the distances from circumcenter to sides -/
structure CircumcenterDistances where
  d : ℕ
  e : ℕ

/-- The theorem statement -/
theorem triangle_with_specific_properties 
  (t : Triangle) 
  (dist : CircumcenterDistances) 
  (h1 : t.r = 25)
  (h2 : t.a > t.b)
  (h3 : t.a^2 + 4 * dist.d^2 = 2500)
  (h4 : t.b^2 + 4 * dist.e^2 = 2500) :
  t.a = 15 ∧ t.b = 7 ∧ t.c = 20 :=
by sorry

end triangle_with_specific_properties_l84_8440


namespace max_b_in_box_l84_8425

theorem max_b_in_box (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) → (c < b) → (b < a) →
  b ≤ 12 :=
by sorry

end max_b_in_box_l84_8425


namespace x_squared_vs_two_to_x_l84_8439

theorem x_squared_vs_two_to_x (x : ℝ) :
  ¬(∀ x, x^2 < 1 → 2^x < 1) ∧ ¬(∀ x, 2^x < 1 → x^2 < 1) :=
sorry

end x_squared_vs_two_to_x_l84_8439


namespace other_root_of_quadratic_l84_8436

theorem other_root_of_quadratic (m : ℝ) : 
  ((-4 : ℝ)^2 + m * (-4) - 20 = 0) → 
  ((5 : ℝ)^2 + m * 5 - 20 = 0) :=
by sorry

end other_root_of_quadratic_l84_8436


namespace sum_25_terms_equals_625_l84_8449

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

def sum_arithmetic_sequence (n : ℕ) : ℕ := n * (arithmetic_sequence 1 + arithmetic_sequence n) / 2

theorem sum_25_terms_equals_625 : sum_arithmetic_sequence 25 = 625 := by sorry

end sum_25_terms_equals_625_l84_8449


namespace median_of_special_list_l84_8462

/-- Represents the sum of integers from 1 to n -/
def triangularSum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents our special list where each number n appears n times, up to 250 -/
def specialList : List ℕ := sorry

/-- The length of our special list -/
def listLength : ℕ := triangularSum 250

/-- The index of the median element in our list -/
def medianIndex : ℕ := (listLength + 1) / 2

/-- Function to find the smallest n such that triangularSum n ≥ target -/
def findSmallestN (target : ℕ) : ℕ := sorry

theorem median_of_special_list :
  let n := findSmallestN medianIndex
  n = 177 := by sorry

end median_of_special_list_l84_8462


namespace complex_magnitude_problem_l84_8408

theorem complex_magnitude_problem (z w : ℂ) : 
  Complex.abs (3 * z - w) = 30 →
  Complex.abs (z + 3 * w) = 6 →
  Complex.abs (z + w) = 3 →
  ∃! (abs_z : ℝ), abs_z > 0 ∧ Complex.abs z = abs_z :=
by sorry

end complex_magnitude_problem_l84_8408


namespace simplify_powers_l84_8485

theorem simplify_powers (a : ℝ) : (a^5 * a^3) * (a^2)^4 = a^16 := by
  sorry

end simplify_powers_l84_8485


namespace calculate_swimming_speed_triathlete_swimming_speed_l84_8422

/-- Calculates the swimming speed given the total distance, running speed, and average speed -/
theorem calculate_swimming_speed 
  (total_distance : ℝ) 
  (running_distance : ℝ) 
  (running_speed : ℝ) 
  (average_speed : ℝ) : ℝ :=
  let swimming_distance := total_distance - running_distance
  let total_time := total_distance / average_speed
  let running_time := running_distance / running_speed
  let swimming_time := total_time - running_time
  let swimming_speed := swimming_distance / swimming_time
  swimming_speed

/-- Proves that the swimming speed is 6 miles per hour given the problem conditions -/
theorem triathlete_swimming_speed :
  calculate_swimming_speed 8 4 10 7.5 = 6 := by
  sorry

end calculate_swimming_speed_triathlete_swimming_speed_l84_8422


namespace hyperbola_condition_l84_8419

/-- Represents a hyperbola equation with parameter m -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / a - y^2 / b = 1 ↔ x^2 / (m - 10) - y^2 / (m - 8) = 1

/-- m > 10 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem hyperbola_condition (m : ℝ) :
  (m > 10 → is_hyperbola m) ∧ (∃ m₀ : ℝ, m₀ ≤ 10 ∧ is_hyperbola m₀) :=
sorry

end hyperbola_condition_l84_8419


namespace workshop_workers_count_workshop_workers_count_proof_l84_8455

/-- Given a workshop with workers, prove that the total number of workers is 28 -/
theorem workshop_workers_count : ℕ :=
  let total_average : ℚ := 8000
  let technician_count : ℕ := 7
  let technician_average : ℚ := 14000
  let non_technician_average : ℚ := 6000
  28

/-- Proof of the theorem -/
theorem workshop_workers_count_proof :
  let total_average : ℚ := 8000
  let technician_count : ℕ := 7
  let technician_average : ℚ := 14000
  let non_technician_average : ℚ := 6000
  workshop_workers_count = 28 := by
  sorry

end workshop_workers_count_workshop_workers_count_proof_l84_8455


namespace program_output_l84_8473

-- Define the program steps as a function
def program (a b : Int) : (Int × Int × Int) :=
  let a' := if a < 0 then -a else a
  let b' := b * b
  let a'' := a' + b'
  let c := a'' - 2 * b'
  let a''' := a'' / c
  let b'' := b' * c + 1
  (a''', b'', c)

-- State the theorem
theorem program_output : program (-6) 2 = (5, 9, 2) := by
  sorry

end program_output_l84_8473


namespace sqrt_sum_given_diff_l84_8402

theorem sqrt_sum_given_diff (x : ℝ) :
  Real.sqrt (100 - x^2) - Real.sqrt (36 - x^2) = 5 →
  Real.sqrt (100 - x^2) + Real.sqrt (36 - x^2) = 12.8 := by
sorry

end sqrt_sum_given_diff_l84_8402


namespace player_a_not_losing_probability_l84_8464

theorem player_a_not_losing_probability 
  (p_win : ℝ) 
  (p_draw : ℝ) 
  (h1 : p_win = 0.3) 
  (h2 : p_draw = 0.4) : 
  p_win + p_draw = 0.7 := by
  sorry

end player_a_not_losing_probability_l84_8464


namespace total_ice_cubes_l84_8441

/-- The number of ice cubes Dave originally had -/
def original_cubes : ℕ := 2

/-- The number of new ice cubes Dave made -/
def new_cubes : ℕ := 7

/-- Theorem: The total number of ice cubes Dave had is 9 -/
theorem total_ice_cubes : original_cubes + new_cubes = 9 := by
  sorry

end total_ice_cubes_l84_8441


namespace solution_set_of_inequality_l84_8434

open Set
open Function
open Real

noncomputable def f (x : ℝ) : ℝ := x * (x^2 - Real.cos (x/3) + 2)

theorem solution_set_of_inequality :
  let S := {x : ℝ | x ∈ Ioo (-3) 3 ∧ f (1 + x) + f 2 < f (1 - x)}
  S = Ioo (-2) (-1) :=
by sorry

end solution_set_of_inequality_l84_8434


namespace outfit_combinations_l84_8407

/-- Calculates the number of outfits given the number of clothing items --/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (jackets : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (jackets + 1)

/-- Theorem stating the number of outfits given specific quantities of clothing items --/
theorem outfit_combinations :
  number_of_outfits 8 5 4 2 = 600 := by
  sorry

end outfit_combinations_l84_8407


namespace cone_base_circumference_l84_8447

/-- 
Given a right circular cone with volume 27π cubic centimeters and height 9 cm,
prove that the circumference of the base is 6π cm.
-/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 27 * Real.pi ∧ h = 9 ∧ V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 6 * Real.pi :=
by sorry

end cone_base_circumference_l84_8447


namespace simplify_fraction_l84_8448

theorem simplify_fraction (x : ℝ) (h : x ≠ -1) :
  (x^2 - 1) / (x + 1) = x - 1 := by
sorry

end simplify_fraction_l84_8448


namespace johns_earnings_ratio_l84_8404

def saturday_earnings : ℤ := 18
def previous_weekend_earnings : ℤ := 20
def pogo_stick_cost : ℤ := 60
def additional_needed : ℤ := 13

def total_earnings : ℤ := pogo_stick_cost - additional_needed

theorem johns_earnings_ratio :
  let sunday_earnings := total_earnings - saturday_earnings - previous_weekend_earnings
  saturday_earnings / sunday_earnings = 2 := by
  sorry

end johns_earnings_ratio_l84_8404


namespace fundraising_excess_l84_8430

/-- Proves that Scott, Mary, and Ken exceeded their fundraising goal by $600 --/
theorem fundraising_excess (ken : ℕ) (mary scott : ℕ) (goal : ℕ) : 
  ken = 600 →
  mary = 5 * ken →
  mary = 3 * scott →
  goal = 4000 →
  mary + scott + ken - goal = 600 := by
sorry

end fundraising_excess_l84_8430


namespace daps_equivalent_to_dips_l84_8443

/-- Represents the conversion rate between daps and dops -/
def daps_to_dops : ℚ := 5 / 4

/-- Represents the conversion rate between dops and dips -/
def dops_to_dips : ℚ := 3 / 8

/-- The number of dips we want to convert to daps -/
def target_dips : ℚ := 40

/-- Theorem stating the equivalence between daps and dips -/
theorem daps_equivalent_to_dips : 
  (target_dips * daps_to_dops * dops_to_dips)⁻¹ * target_dips = 18.75 := by
  sorry

end daps_equivalent_to_dips_l84_8443


namespace functional_equation_proof_l84_8458

/-- Given a function f: ℝ → ℝ satisfying the functional equation
    f(x + y) = f(x) * f(y) for all real x and y, and f(3) = 4,
    prove that f(9) = 64. -/
theorem functional_equation_proof (f : ℝ → ℝ) 
    (h1 : ∀ x y : ℝ, f (x + y) = f x * f y) 
    (h2 : f 3 = 4) : 
  f 9 = 64 := by
  sorry

end functional_equation_proof_l84_8458


namespace quadratic_equation_roots_ratio_l84_8459

theorem quadratic_equation_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 10*x + k = 0 ∧ y^2 + 10*y + k = 0) → 
  k = 18.75 := by
sorry

end quadratic_equation_roots_ratio_l84_8459


namespace least_bananas_total_l84_8415

/-- Represents the number of bananas taken by each monkey -/
structure BananaCounts where
  b₁ : ℕ
  b₂ : ℕ
  b₃ : ℕ

/-- Represents the final distribution of bananas for each monkey -/
structure FinalDistribution where
  m₁ : ℕ
  m₂ : ℕ
  m₃ : ℕ

/-- Calculates the final distribution based on the initial banana counts -/
def calculateFinalDistribution (counts : BananaCounts) : FinalDistribution :=
  { m₁ := counts.b₁ / 2 + counts.b₂ / 12 + counts.b₃ * 3 / 32
  , m₂ := counts.b₁ / 6 + counts.b₂ * 2 / 3 + counts.b₃ * 3 / 32
  , m₃ := counts.b₁ / 6 + counts.b₂ / 12 + counts.b₃ * 3 / 4 }

/-- Checks if the final distribution satisfies the 4:3:2 ratio -/
def satisfiesRatio (dist : FinalDistribution) : Prop :=
  3 * dist.m₁ = 4 * dist.m₂ ∧ 2 * dist.m₁ = 3 * dist.m₃

/-- The main theorem stating the least possible total number of bananas -/
theorem least_bananas_total (counts : BananaCounts) :
  (∀ (dist : FinalDistribution), dist = calculateFinalDistribution counts → satisfiesRatio dist) →
  counts.b₁ + counts.b₂ + counts.b₃ ≥ 148 :=
by sorry

end least_bananas_total_l84_8415


namespace sin_square_sum_range_l84_8406

open Real

theorem sin_square_sum_range (α β : ℝ) (h : 3 * (sin α)^2 - 2 * sin α + 2 * (sin β)^2 = 0) :
  ∃ (x : ℝ), x = (sin α)^2 + (sin β)^2 ∧ 0 ≤ x ∧ x ≤ 4/9 ∧
  ∀ (y : ℝ), y = (sin α)^2 + (sin β)^2 → 0 ≤ y ∧ y ≤ 4/9 :=
by sorry

end sin_square_sum_range_l84_8406


namespace minimum_guests_l84_8496

theorem minimum_guests (total_food : ℕ) (max_per_guest : ℕ) (h1 : total_food = 323) (h2 : max_per_guest = 2) :
  ∃ (min_guests : ℕ), min_guests = 162 ∧ min_guests * max_per_guest ≥ total_food ∧
  ∀ (n : ℕ), n * max_per_guest ≥ total_food → n ≥ min_guests :=
by sorry

end minimum_guests_l84_8496


namespace same_result_different_parentheses_l84_8461

-- Define the exponentiation operation
def power (a b : ℕ) : ℕ := a ^ b

-- Define the two different parenthesization methods
def method1 (n : ℕ) : ℕ := power (power n 7) (power 7 7)
def method2 (n : ℕ) : ℕ := power (power n (power 7 7)) 7

-- Theorem statement
theorem same_result_different_parentheses :
  ∃ (n : ℕ), method1 n = method2 n :=
sorry

end same_result_different_parentheses_l84_8461


namespace pure_imaginary_condition_l84_8453

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ (b : ℝ), (Complex.I : ℂ) * b = (1 + Complex.I) / (1 + a * Complex.I)) → a = -1 :=
by sorry

end pure_imaginary_condition_l84_8453


namespace parabola_zeros_difference_l84_8478

/-- A parabola with vertex (3, -2) passing through (5, 14) has zeros with difference √2 -/
theorem parabola_zeros_difference (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (x - 3)^2 - 2) →  -- Vertex form
  (4 * 4 + b * 5 + c = 14) →  -- Point (5, 14) satisfies the equation
  (∃ m n : ℝ, m > n ∧ 
    a * m^2 + b * m + c = 0 ∧ 
    a * n^2 + b * n + c = 0 ∧ 
    m - n = Real.sqrt 2) :=
by sorry

end parabola_zeros_difference_l84_8478


namespace other_root_of_complex_equation_l84_8411

theorem other_root_of_complex_equation (z : ℂ) :
  z^2 = -99 + 64*I ∧ (5 + 8*I)^2 = -99 + 64*I → z = 5 + 8*I ∨ z = -5 - 8*I :=
by sorry

end other_root_of_complex_equation_l84_8411


namespace AAA_not_sufficient_for_congruence_l84_8405

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  sum_angles : α + β + γ = π

-- Define triangle congruence
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define AAA criterion
def AAA_equal (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem: AAA criterion is not sufficient for triangle congruence
theorem AAA_not_sufficient_for_congruence :
  ¬(∀ (t1 t2 : Triangle), AAA_equal t1 t2 → congruent t1 t2) :=
sorry

end AAA_not_sufficient_for_congruence_l84_8405


namespace cannot_fit_rectangles_l84_8454

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

/-- The large rectangle -/
def largeRectangle : Rectangle := { width := 13, height := 7 }

/-- The small rectangle -/
def smallRectangle : Rectangle := { width := 2, height := 3 }

/-- The number of small rectangles -/
def numSmallRectangles : ℕ := 15

/-- Theorem stating that it's not possible to fit 15 small rectangles into the large rectangle -/
theorem cannot_fit_rectangles : 
  area largeRectangle > numSmallRectangles * area smallRectangle :=
sorry

end cannot_fit_rectangles_l84_8454


namespace earliest_meeting_time_l84_8432

def anna_lap_time : ℕ := 5
def stephanie_lap_time : ℕ := 8
def james_lap_time : ℕ := 9
def tom_lap_time : ℕ := 10

theorem earliest_meeting_time :
  let lap_times := [anna_lap_time, stephanie_lap_time, james_lap_time, tom_lap_time]
  Nat.lcm (Nat.lcm (Nat.lcm anna_lap_time stephanie_lap_time) james_lap_time) tom_lap_time = 360 :=
by sorry

end earliest_meeting_time_l84_8432


namespace remaining_digits_average_l84_8490

theorem remaining_digits_average (digits : Finset ℕ) (subset : Finset ℕ) :
  Finset.card digits = 9 →
  (Finset.sum digits id) / 9 = 18 →
  Finset.card subset = 4 →
  subset ⊆ digits →
  (Finset.sum subset id) / 4 = 8 →
  let remaining := digits \ subset
  ((Finset.sum remaining id) / (Finset.card remaining) : ℚ) = 26 := by
sorry

end remaining_digits_average_l84_8490


namespace amaya_movie_watching_time_l84_8471

/-- Calculates the total time spent watching a movie with interruptions and rewinds -/
def total_watching_time (segment1 segment2 segment3 rewind1 rewind2 : ℕ) : ℕ :=
  segment1 + segment2 + segment3 + rewind1 + rewind2

/-- Theorem stating that the total watching time for Amaya's movie is 120 minutes -/
theorem amaya_movie_watching_time :
  total_watching_time 35 45 20 5 15 = 120 := by
  sorry

#eval total_watching_time 35 45 20 5 15

end amaya_movie_watching_time_l84_8471


namespace nina_age_l84_8487

/-- Given the ages of Max, Leah, Alex, and Nina, prove Nina's age --/
theorem nina_age (max_age leah_age alex_age nina_age : ℕ) 
  (h1 : max_age = leah_age - 5)
  (h2 : leah_age = alex_age + 6)
  (h3 : nina_age = alex_age + 2)
  (h4 : max_age = 16) : 
  nina_age = 17 := by
  sorry

#check nina_age

end nina_age_l84_8487


namespace quadratic_equation_1_l84_8463

theorem quadratic_equation_1 : 
  ∃ x₁ x₂ : ℝ, (x₁ + 1)^2 - 144 = 0 ∧ (x₂ + 1)^2 - 144 = 0 ∧ x₁ ≠ x₂ :=
by sorry

end quadratic_equation_1_l84_8463


namespace sin_cos_tan_l84_8476

theorem sin_cos_tan (α : Real) (h : Real.tan α = 4) : 
  Real.sin α * Real.cos α = 4 / 17 := by
sorry

end sin_cos_tan_l84_8476


namespace percent_problem_l84_8457

theorem percent_problem (x : ℝ) : 
  (30 / 100 * 100 = 50 / 100 * x + 10) → x = 40 := by
  sorry

end percent_problem_l84_8457


namespace packaging_methods_different_boxes_l84_8499

theorem packaging_methods_different_boxes (n : ℕ) (m : ℕ) :
  n > 0 → m > 0 → (number_of_packaging_methods : ℕ) = m^n :=
by sorry

end packaging_methods_different_boxes_l84_8499


namespace zinc_copper_mixture_weight_l84_8450

/-- Proves that the total weight of a zinc-copper mixture is 70 kg,
    given a 9:11 ratio and 31.5 kg of zinc used. -/
theorem zinc_copper_mixture_weight
  (zinc_ratio : ℝ)
  (copper_ratio : ℝ)
  (zinc_weight : ℝ)
  (h_ratio : zinc_ratio / copper_ratio = 9 / 11)
  (h_zinc : zinc_weight = 31.5) :
  zinc_weight + (copper_ratio / zinc_ratio) * zinc_weight = 70 :=
by sorry

end zinc_copper_mixture_weight_l84_8450


namespace two_digit_three_digit_percentage_equality_l84_8435

theorem two_digit_three_digit_percentage_equality :
  ∃! (A B : ℕ),
    (A ≥ 10 ∧ A ≤ 99) ∧
    (B ≥ 100 ∧ B ≤ 999) ∧
    (A * (1 + B / 100 : ℚ) = B * (1 - A / 100 : ℚ)) ∧
    A = 40 ∧
    B = 200 := by
  sorry

end two_digit_three_digit_percentage_equality_l84_8435


namespace seventh_term_is_25_over_3_l84_8421

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℚ
  -- Common difference
  d : ℚ
  -- Sum of first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- Sixth term is 7
  sixth_term : a + 5*d = 7

/-- The seventh term of the arithmetic sequence is 25/3 -/
theorem seventh_term_is_25_over_3 (seq : ArithmeticSequence) :
  seq.a + 6*seq.d = 25/3 := by
  sorry

end seventh_term_is_25_over_3_l84_8421


namespace identity_is_unique_satisfying_function_l84_8437

/-- A function satisfying the given property -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x^3 + f y * x + f z = 0 → f x^3 + y * f x + z = 0

/-- The main theorem stating that the identity function is the only function satisfying the property -/
theorem identity_is_unique_satisfying_function :
  ∀ f : ℝ → ℝ, SatisfyingFunction f → f = id := by sorry

end identity_is_unique_satisfying_function_l84_8437


namespace gauss_polynomial_reciprocal_l84_8492

/-- Definition of a Gauss polynomial -/
def is_gauss_polynomial (g : ℤ → ℤ → (ℝ → ℝ)) : Prop :=
  ∀ (k l : ℤ) (x : ℝ), x ≠ 0 → x^(k*l) * g k l (1/x) = g k l x

/-- Theorem: Gauss polynomials are reciprocal -/
theorem gauss_polynomial_reciprocal (g : ℤ → ℤ → (ℝ → ℝ)) (h : is_gauss_polynomial g) :
  ∀ (k l : ℤ) (x : ℝ), x ≠ 0 → x^(k*l) * g k l (1/x) = g k l x :=
sorry

end gauss_polynomial_reciprocal_l84_8492


namespace valid_probability_is_one_fourteenth_l84_8486

/-- Represents a bead color -/
inductive BeadColor
| Red
| White
| Blue

/-- Represents a configuration of beads -/
def BeadConfiguration := List BeadColor

/-- Checks if a configuration has no adjacent beads of the same color -/
def noAdjacentSameColor (config : BeadConfiguration) : Bool :=
  sorry

/-- Generates all possible bead configurations -/
def allConfigurations : List BeadConfiguration :=
  sorry

/-- Counts the number of valid configurations -/
def countValidConfigurations : Nat :=
  sorry

/-- The total number of possible configurations -/
def totalConfigurations : Nat := 420

/-- The probability of a valid configuration -/
def validProbability : ℚ :=
  sorry

theorem valid_probability_is_one_fourteenth :
  validProbability = 1 / 14 := by
  sorry

end valid_probability_is_one_fourteenth_l84_8486


namespace hanson_employees_count_l84_8401

theorem hanson_employees_count :
  ∃ (E : ℕ) (M B B' : ℤ), 
    M = E * B + 2 ∧ 
    3 * M = E * B' + 1 → 
    E = 5 := by
  sorry

end hanson_employees_count_l84_8401


namespace max_sum_for_product_1386_l84_8412

theorem max_sum_for_product_1386 :
  ∃ (A B C : ℕ+),
    A * B * C = 1386 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    ∀ (X Y Z : ℕ+),
      X * Y * Z = 1386 →
      X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z →
      X + Y + Z ≤ A + B + C ∧
      A + B + C = 88 :=
by sorry

end max_sum_for_product_1386_l84_8412


namespace sixth_salary_proof_l84_8489

theorem sixth_salary_proof (known_salaries : List ℝ) 
  (h1 : known_salaries = [1000, 2500, 3100, 3650, 2000])
  (h2 : (known_salaries.sum + x) / 6 = 2291.67) : x = 1500 := by
  sorry

end sixth_salary_proof_l84_8489


namespace roots_product_equals_squared_difference_l84_8442

theorem roots_product_equals_squared_difference (m n : ℝ) 
  (α β γ δ : ℝ) : 
  (α^2 + m*α - 1 = 0) → 
  (β^2 + m*β - 1 = 0) → 
  (γ^2 + n*γ - 1 = 0) → 
  (δ^2 + n*δ - 1 = 0) → 
  (α - γ)*(β - γ)*(α - δ)*(β - δ) = (m - n)^2 := by
  sorry

end roots_product_equals_squared_difference_l84_8442


namespace sqrt_inequality_l84_8403

theorem sqrt_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

end sqrt_inequality_l84_8403


namespace even_function_interval_l84_8494

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the interval
def interval (m : ℝ) : Set ℝ := Set.Icc (2*m) (m+6)

-- State the theorem
theorem even_function_interval (m : ℝ) :
  (∀ x ∈ interval m, f x = f (-x)) → m = -2 := by
  sorry

end even_function_interval_l84_8494


namespace different_color_probability_l84_8433

def blue_chips : ℕ := 4
def yellow_chips : ℕ := 5
def green_chips : ℕ := 3

def total_chips : ℕ := blue_chips + yellow_chips + green_chips

theorem different_color_probability :
  let p_blue_first := blue_chips / total_chips
  let p_yellow_first := yellow_chips / total_chips
  let p_green_first := green_chips / total_chips
  let p_not_blue_second := (yellow_chips + green_chips) / (total_chips - 1)
  let p_not_yellow_second := (blue_chips + green_chips) / (total_chips - 1)
  let p_not_green_second := (blue_chips + yellow_chips) / (total_chips - 1)
  (p_blue_first * p_not_blue_second + 
   p_yellow_first * p_not_yellow_second + 
   p_green_first * p_not_green_second) = 47 / 66 :=
by
  sorry

end different_color_probability_l84_8433


namespace ship_elevation_change_l84_8472

/-- The average change in elevation per hour for a ship traveling between Lake Ontario and Lake Erie -/
theorem ship_elevation_change (lake_ontario_elevation lake_erie_elevation : ℝ) (travel_time : ℝ) :
  lake_ontario_elevation = 75 ∧ 
  lake_erie_elevation = 174.28 ∧ 
  travel_time = 8 →
  (lake_erie_elevation - lake_ontario_elevation) / travel_time = 12.41 :=
by sorry

end ship_elevation_change_l84_8472


namespace parallelogram_side_sum_l84_8418

/-- A parallelogram with sides measuring 10, 12, 5y-2, and 3x+6 units consecutively has x + y = 22/5 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  3 * x + 6 = 12 → 5 * y - 2 = 10 → x + y = 22 / 5 := by
  sorry

end parallelogram_side_sum_l84_8418


namespace special_function_is_identity_l84_8491

/-- A function satisfying certain conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≤ x) ∧ (∀ x y, f (x + y) ≤ f x + f y)

/-- Theorem: If f is a SpecialFunction, then f(x) = x for all x in ℝ -/
theorem special_function_is_identity (f : ℝ → ℝ) (hf : SpecialFunction f) :
  ∀ x, f x = x := by
  sorry

end special_function_is_identity_l84_8491


namespace relay_race_time_difference_l84_8488

def apple_distance : ℝ := 24
def apple_speed : ℝ := 3
def mac_distance : ℝ := 28
def mac_speed : ℝ := 4
def orange_distance : ℝ := 32
def orange_speed : ℝ := 5

def minutes_per_hour : ℝ := 60

theorem relay_race_time_difference :
  (apple_distance / apple_speed + mac_distance / mac_speed) * minutes_per_hour -
  (orange_distance / orange_speed * minutes_per_hour) = 516 := by
sorry

end relay_race_time_difference_l84_8488


namespace add_9999_seconds_to_10_15_30_l84_8482

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time -/
def initialTime : Time :=
  { hours := 10, minutes := 15, seconds := 30 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 9999

/-- The expected final time -/
def expectedFinalTime : Time :=
  { hours := 13, minutes := 2, seconds := 9 }

theorem add_9999_seconds_to_10_15_30 :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end add_9999_seconds_to_10_15_30_l84_8482


namespace min_sum_first_two_terms_l84_8469

/-- A sequence of positive integers satisfying the given recurrence relation -/
def ValidSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = (a n + 3024) / (1 + a (n + 1))

/-- The theorem stating the minimum possible value of a₁ + a₂ -/
theorem min_sum_first_two_terms (a : ℕ → ℕ) (h : ValidSequence a) :
    ∀ b : ℕ → ℕ, ValidSequence b → a 1 + a 2 ≤ b 1 + b 2 :=
  sorry

end min_sum_first_two_terms_l84_8469


namespace inequality_proof_l84_8428

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b ≤ 1) :
  9 * a^2 * b + 9 * a * b^2 - a^2 - 10 * a * b - b^2 + a + b ≥ 0 := by
  sorry

end inequality_proof_l84_8428


namespace chocolate_bars_count_l84_8466

/-- Given the total number of treats and the counts of chewing gums and candies,
    calculate the number of chocolate bars. -/
theorem chocolate_bars_count 
  (total_treats : ℕ) 
  (chewing_gums : ℕ) 
  (candies : ℕ) 
  (h1 : total_treats = 155) 
  (h2 : chewing_gums = 60) 
  (h3 : candies = 40) : 
  total_treats - (chewing_gums + candies) = 55 := by
  sorry

#eval 155 - (60 + 40)  -- Should output 55

end chocolate_bars_count_l84_8466


namespace factorial_sum_equality_l84_8410

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 3 * Nat.factorial 4 + Nat.factorial 4 = 1416 := by
  sorry

end factorial_sum_equality_l84_8410


namespace bryans_books_l84_8414

/-- Calculates the total number of books given the number of bookshelves and books per shelf. -/
def total_books (num_shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  num_shelves * books_per_shelf

/-- Theorem stating that Bryan's total number of books is 504. -/
theorem bryans_books : 
  total_books 9 56 = 504 := by
  sorry

end bryans_books_l84_8414


namespace f_monotone_iff_f_greater_than_2x_l84_8498

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 + Real.log ((x / a) + 1)

-- Theorem 1: Monotonicity condition
theorem f_monotone_iff (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1) 0, Monotone (f a)) ↔ a ∈ Set.Iic (1 - Real.exp 1) ∪ Set.Ici 1 :=
sorry

-- Theorem 2: Inequality for specific a and x
theorem f_greater_than_2x (a : ℝ) (x : ℝ) (ha : a ∈ Set.Ioo 0 1) (hx : x > 0) :
  f a x > 2 * x :=
sorry

end f_monotone_iff_f_greater_than_2x_l84_8498


namespace box_interior_area_l84_8497

/-- Calculates the surface area of the interior of a box formed from a rectangular sheet of cardboard
    with square corners cut out and edges folded upwards. -/
def interior_surface_area (sheet_length : ℕ) (sheet_width : ℕ) (corner_size : ℕ) : ℕ :=
  (sheet_length - 2 * corner_size) * (sheet_width - 2 * corner_size)

/-- Theorem stating that the interior surface area of the box formed from a 35x50 sheet
    with 7-unit corners cut out is 756 square units. -/
theorem box_interior_area :
  interior_surface_area 35 50 7 = 756 := by
  sorry

end box_interior_area_l84_8497


namespace no_solution_for_prime_factor_conditions_l84_8456

/-- P(n) denotes the greatest prime factor of n -/
def greatest_prime_factor (n : ℕ) : ℕ :=
  sorry

theorem no_solution_for_prime_factor_conditions : 
  ∀ n : ℕ, n > 1 → 
  ¬(greatest_prime_factor n = Real.sqrt n ∧ 
    greatest_prime_factor (n + 54) = Real.sqrt (n + 54)) :=
by sorry

end no_solution_for_prime_factor_conditions_l84_8456


namespace factory_sampling_is_systematic_l84_8481

/-- Represents a sampling method --/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a sampling process --/
structure SamplingProcess where
  interval : ℕ  -- Time interval between samples
  continuous : Bool  -- Whether the process is continuous

/-- Determines if a sampling process is systematic --/
def is_systematic (process : SamplingProcess) : Prop :=
  process.interval > 0 ∧ process.continuous

/-- Theorem: A sampling process with a fixed positive time interval 
    from a continuous process is systematic sampling --/
theorem factory_sampling_is_systematic 
  (process : SamplingProcess) 
  (h1 : process.interval = 10)  -- 10-minute interval
  (h2 : process.continuous = true)  -- Conveyor belt implies continuous process
  : is_systematic process ∧ 
    (λ method : SamplingMethod => 
      is_systematic process → method = SamplingMethod.Systematic) SamplingMethod.Systematic := by
  sorry


end factory_sampling_is_systematic_l84_8481


namespace fifth_term_of_geometric_sequence_l84_8420

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) :
  IsGeometricSequence a → a 3 = 18 → a 4 = 24 → a 5 = 32 := by
  sorry


end fifth_term_of_geometric_sequence_l84_8420


namespace square_division_l84_8444

theorem square_division (a : ℕ) (h1 : a > 0) :
  (a * a = 25) ∧
  (∃ b : ℕ, b > 0 ∧ a * a = 24 * 1 * 1 + b * b) ∧
  (a = 5) :=
by sorry

end square_division_l84_8444


namespace range_of_a_l84_8445

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ((a - 5) * x > a - 5) ↔ x < 1) → a < 5 := by
  sorry

end range_of_a_l84_8445


namespace zero_subset_X_l84_8452

-- Define the set X
def X : Set ℝ := {x | x > -4}

-- State the theorem
theorem zero_subset_X : {0} ⊆ X := by sorry

end zero_subset_X_l84_8452


namespace charles_and_jen_whistles_l84_8483

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The difference in whistles between Sean and Charles -/
def sean_charles_diff : ℕ := 32

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := sean_whistles - sean_charles_diff

/-- The difference in whistles between Jen and Charles -/
def jen_charles_diff : ℕ := 15

/-- The number of whistles Jen has -/
def jen_whistles : ℕ := charles_whistles + jen_charles_diff

/-- The total number of whistles Charles and Jen have -/
def total_whistles : ℕ := charles_whistles + jen_whistles

theorem charles_and_jen_whistles : total_whistles = 41 := by
  sorry

end charles_and_jen_whistles_l84_8483


namespace candy_bar_cost_l84_8465

/-- The cost of a candy bar, given that it costs $1 less than a chocolate that costs $3. -/
theorem candy_bar_cost : ℝ := by
  -- Define the cost of the candy bar
  let candy_cost : ℝ := 2

  -- Define the cost of the chocolate
  let chocolate_cost : ℝ := 3

  -- Assert that the chocolate costs $1 more than the candy bar
  have h1 : chocolate_cost = candy_cost + 1 := by sorry

  -- Prove that the candy bar costs $2
  have h2 : candy_cost = 2 := by sorry

  -- Return the cost of the candy bar
  exact candy_cost


end candy_bar_cost_l84_8465


namespace guppy_ratio_l84_8423

/-- Represents the number of guppies each person has -/
structure Guppies where
  haylee : ℕ
  jose : ℕ
  charliz : ℕ
  nicolai : ℕ

/-- The conditions of the guppy problem -/
def guppy_conditions (g : Guppies) : Prop :=
  g.haylee = 36 ∧
  g.charliz = g.jose / 3 ∧
  g.nicolai = 4 * g.charliz ∧
  g.haylee + g.jose + g.charliz + g.nicolai = 84

/-- The theorem stating the ratio of Jose's guppies to Haylee's guppies -/
theorem guppy_ratio (g : Guppies) (h : guppy_conditions g) : 
  g.jose * 2 = g.haylee :=
sorry

end guppy_ratio_l84_8423


namespace three_digit_divisible_by_11_and_5_l84_8446

theorem three_digit_divisible_by_11_and_5 : 
  (Finset.filter (fun n => n % 55 = 0) (Finset.range 900 ⊔ Finset.range 100)).card = 17 := by
  sorry

end three_digit_divisible_by_11_and_5_l84_8446


namespace arithmetic_sequence_sums_l84_8409

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sums (a : ℕ → ℝ) :
  isArithmeticSequence a →
  isArithmeticSequence (λ n : ℕ => a (3*n + 1) + a (3*n + 2) + a (3*n + 3)) :=
by sorry

end arithmetic_sequence_sums_l84_8409


namespace last_eight_digits_of_product_l84_8438

def product : ℕ := 11 * 101 * 1001 * 10001 * 1000001 * 111

theorem last_eight_digits_of_product : product % 100000000 = 87654321 := by
  sorry

end last_eight_digits_of_product_l84_8438


namespace journey_distance_correct_total_distance_l84_8416

-- Define the journey parameters
def total_time : ℝ := 30
def speed_first_half : ℝ := 20
def speed_second_half : ℝ := 10

-- Define the total distance
def total_distance : ℝ := 400

-- Theorem statement
theorem journey_distance :
  (total_distance / 2 / speed_first_half) + (total_distance / 2 / speed_second_half) = total_time :=
by sorry

-- Proof that the total distance is correct
theorem correct_total_distance : total_distance = 400 :=
by sorry

end journey_distance_correct_total_distance_l84_8416


namespace largest_negative_integer_congruence_l84_8480

theorem largest_negative_integer_congruence :
  ∃ (x : ℤ), x = -14 ∧ 
  (∀ (y : ℤ), y < 0 → 26 * y + 8 ≡ 4 [ZMOD 18] → y ≤ x) ∧
  (26 * x + 8 ≡ 4 [ZMOD 18]) := by
  sorry

end largest_negative_integer_congruence_l84_8480


namespace number_problem_l84_8429

theorem number_problem (a b c : ℕ) :
  Nat.gcd a b = 15 →
  Nat.gcd b c = 6 →
  b * c = 1800 →
  Nat.lcm a b = 3150 →
  a = 315 ∧ b = 150 ∧ c = 12 := by
  sorry

end number_problem_l84_8429


namespace b_completion_time_l84_8468

/-- Given two workers a and b, this theorem proves how long it takes b to complete a job alone. -/
theorem b_completion_time (work : ℝ) (a b : ℝ → ℝ) :
  (∀ t, a t + b t = work / 16) →  -- a and b together complete the work in 16 days
  (∀ t, a t = work / 20) →        -- a alone completes the work in 20 days
  (∀ t, b t = work / 80) :=       -- b alone completes the work in 80 days
by sorry

end b_completion_time_l84_8468


namespace speed_of_X_is_60_l84_8424

-- Define the speed of person Y
def speed_Y : ℝ := 60

-- Define the time difference between X and Y's start
def time_difference : ℝ := 3

-- Define the distance ahead
def distance_ahead : ℝ := 30

-- Define the time difference between Y catching up to X and X catching up to Y
def catch_up_time_difference : ℝ := 3

-- Define the speed of person X
def speed_X : ℝ := 60

-- Theorem statement
theorem speed_of_X_is_60 :
  ∀ (t₁ t₂ : ℝ),
  t₂ - t₁ = catch_up_time_difference →
  speed_X * (time_difference + t₁) = speed_Y * t₁ + distance_ahead →
  speed_X * (time_difference + t₂) + distance_ahead = speed_Y * t₂ →
  speed_X = speed_Y :=
by sorry

end speed_of_X_is_60_l84_8424


namespace constant_function_l84_8477

theorem constant_function (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2011 * x) = 2011) :
  ∀ x : ℝ, f (3 * x) = 2011 := by
sorry

end constant_function_l84_8477


namespace certain_number_proof_l84_8460

theorem certain_number_proof : 
  ∃ x : ℕ, (7899665 : ℕ) - (12 * 3 * x) = 7899593 ∧ x = 2 := by
  sorry

end certain_number_proof_l84_8460


namespace earliest_saturday_after_second_monday_after_second_thursday_l84_8474

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Returns the next date -/
def nextDate (d : Date) : Date :=
  { day := d.day + 1, dayOfWeek := nextDay d.dayOfWeek }

/-- Finds the nth occurrence of a specific day of the week, starting from a given date -/
def findNthDay (start : Date) (target : DayOfWeek) (n : Nat) : Date :=
  sorry

/-- Finds the first occurrence of a specific day of the week, starting from a given date -/
def findNextDay (start : Date) (target : DayOfWeek) : Date :=
  sorry

/-- Main theorem: The earliest possible date for the first Saturday after the second Monday 
    following the second Thursday of any month is the 17th -/
theorem earliest_saturday_after_second_monday_after_second_thursday (startDate : Date) : 
  (findNextDay 
    (findNthDay 
      (findNthDay startDate DayOfWeek.Thursday 2) 
      DayOfWeek.Monday 
      2) 
    DayOfWeek.Saturday).day ≥ 17 :=
  sorry

end earliest_saturday_after_second_monday_after_second_thursday_l84_8474


namespace tan_30_squared_plus_sin_45_squared_l84_8467

theorem tan_30_squared_plus_sin_45_squared : 
  (Real.tan (30 * π / 180))^2 + (Real.sin (45 * π / 180))^2 = 5/6 := by
  sorry

end tan_30_squared_plus_sin_45_squared_l84_8467


namespace ferry_travel_time_l84_8413

/-- Represents the travel time of Ferry P in hours -/
def t : ℝ := 3

/-- Speed of Ferry P in km/h -/
def speed_p : ℝ := 6

/-- Speed of Ferry Q in km/h -/
def speed_q : ℝ := speed_p + 3

/-- Distance traveled by Ferry P in km -/
def distance_p : ℝ := speed_p * t

/-- Distance traveled by Ferry Q in km -/
def distance_q : ℝ := 2 * distance_p

/-- Travel time of Ferry Q in hours -/
def time_q : ℝ := t + 1

theorem ferry_travel_time :
  speed_q * time_q = distance_q ∧ t = 3 := by sorry

end ferry_travel_time_l84_8413
