import Mathlib

namespace cost_of_beads_per_bracelet_l1477_147715

/-- Proves the cost of beads per bracelet given the selling price, string cost, number of bracelets sold, and total profit -/
theorem cost_of_beads_per_bracelet 
  (selling_price : ℝ)
  (string_cost : ℝ)
  (bracelets_sold : ℕ)
  (total_profit : ℝ)
  (h1 : selling_price = 6)
  (h2 : string_cost = 1)
  (h3 : bracelets_sold = 25)
  (h4 : total_profit = 50) :
  let bead_cost := (bracelets_sold : ℝ) * selling_price - total_profit - bracelets_sold * string_cost
  bead_cost / (bracelets_sold : ℝ) = 3 := by
sorry

end cost_of_beads_per_bracelet_l1477_147715


namespace vegetable_growth_rate_equation_l1477_147722

theorem vegetable_growth_rate_equation 
  (initial_production final_production : ℝ) 
  (growth_years : ℕ) 
  (x : ℝ) 
  (h1 : initial_production = 800)
  (h2 : final_production = 968)
  (h3 : growth_years = 2)
  (h4 : final_production = initial_production * (1 + x) ^ growth_years) :
  800 * (1 + x)^2 = 968 := by
sorry

end vegetable_growth_rate_equation_l1477_147722


namespace painting_width_l1477_147731

/-- Given a wall and a painting with specific dimensions, prove the width of the painting -/
theorem painting_width
  (wall_height : ℝ)
  (wall_width : ℝ)
  (painting_height : ℝ)
  (painting_area_percentage : ℝ)
  (h1 : wall_height = 5)
  (h2 : wall_width = 10)
  (h3 : painting_height = 2)
  (h4 : painting_area_percentage = 0.16)
  : (wall_height * wall_width * painting_area_percentage) / painting_height = 4 := by
  sorry

end painting_width_l1477_147731


namespace count_multiples_l1477_147761

theorem count_multiples (n : ℕ) : 
  (Finset.filter (fun x => x % 7 = 0 ∧ x % 14 ≠ 0) (Finset.range 350)).card = 25 := by
  sorry

end count_multiples_l1477_147761


namespace resulting_solution_percentage_l1477_147716

/-- Calculates the percentage of chemicals in the resulting solution when a portion of a 90% solution is replaced with an equal amount of 20% solution. -/
theorem resulting_solution_percentage 
  (original_concentration : Real) 
  (replacement_concentration : Real)
  (replaced_portion : Real) :
  original_concentration = 0.9 →
  replacement_concentration = 0.2 →
  replaced_portion = 0.7142857142857143 →
  let remaining_portion := 1 - replaced_portion
  let chemicals_in_remaining := remaining_portion * original_concentration
  let chemicals_in_added := replaced_portion * replacement_concentration
  let total_chemicals := chemicals_in_remaining + chemicals_in_added
  let resulting_concentration := total_chemicals / 1
  resulting_concentration = 0.4 := by
  sorry

end resulting_solution_percentage_l1477_147716


namespace girls_from_clay_middle_school_l1477_147721

theorem girls_from_clay_middle_school
  (total_students : ℕ)
  (total_boys : ℕ)
  (total_girls : ℕ)
  (jonas_students : ℕ)
  (clay_students : ℕ)
  (hart_students : ℕ)
  (jonas_boys : ℕ)
  (h1 : total_students = 150)
  (h2 : total_boys = 90)
  (h3 : total_girls = 60)
  (h4 : jonas_students = 50)
  (h5 : clay_students = 70)
  (h6 : hart_students = 30)
  (h7 : jonas_boys = 25)
  (h8 : total_students = total_boys + total_girls)
  (h9 : total_students = jonas_students + clay_students + hart_students)
  : ∃ clay_girls : ℕ, clay_girls = 30 ∧ clay_girls ≤ clay_students :=
by sorry

end girls_from_clay_middle_school_l1477_147721


namespace appended_number_theorem_l1477_147753

theorem appended_number_theorem (a x : ℕ) (ha : 0 < a) (hx : x ≤ 9) :
  (10 * a + x - a^2 = (11 - x) * a) ↔ (x = a) := by
sorry

end appended_number_theorem_l1477_147753


namespace cos_105_degrees_l1477_147752

theorem cos_105_degrees : 
  Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_degrees_l1477_147752


namespace intersection_points_distance_l1477_147767

noncomputable section

-- Define the curve C in polar coordinates
def curve_C (a : ℝ) (θ : ℝ) : ℝ := 2 * Real.sin θ + 2 * a * Real.cos θ

-- Define the line l in parametric form
def line_l (t : ℝ) : ℝ × ℝ := (-2 + Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t)

-- Define point P in polar coordinates
def point_P : ℝ × ℝ := (2, Real.pi)

-- Theorem statement
theorem intersection_points_distance (a : ℝ) :
  a > 0 →
  ∃ (M N : ℝ × ℝ),
    (∃ (t₁ t₂ : ℝ), M = line_l t₁ ∧ N = line_l t₂) ∧
    (∃ (θ₁ θ₂ : ℝ), curve_C a θ₁ = Real.sqrt ((M.1)^2 + (M.2)^2) ∧
                    curve_C a θ₂ = Real.sqrt ((N.1)^2 + (N.2)^2)) ∧
    Real.sqrt ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2) +
    Real.sqrt ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2) = 5 * Real.sqrt 2 →
  a = 2 := by
  sorry

end

end intersection_points_distance_l1477_147767


namespace max_xyz_value_l1477_147734

theorem max_xyz_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x * y + 3 * z = (x + 3 * z) * (y + 3 * z)) :
  x * y * z ≤ 1 / 81 :=
sorry

end max_xyz_value_l1477_147734


namespace blocks_added_to_tower_l1477_147730

/-- The number of blocks added to a tower -/
def blocks_added (initial final : ℝ) : ℝ := final - initial

/-- Proof that 65.0 blocks were added to the tower -/
theorem blocks_added_to_tower : blocks_added 35.0 100 = 65.0 := by
  sorry

end blocks_added_to_tower_l1477_147730


namespace sqrt_three_squared_l1477_147778

theorem sqrt_three_squared : Real.sqrt 3 * Real.sqrt 3 = 3 := by
  sorry

end sqrt_three_squared_l1477_147778


namespace rose_cost_l1477_147709

/-- The cost of each red rose, given the conditions of Jezebel's flower purchase. -/
theorem rose_cost (num_roses : ℕ) (num_sunflowers : ℕ) (sunflower_cost : ℚ) (total_cost : ℚ) :
  num_roses = 24 →
  num_sunflowers = 3 →
  sunflower_cost = 3 →
  total_cost = 45 →
  (total_cost - num_sunflowers * sunflower_cost) / num_roses = 3/2 := by
  sorry

end rose_cost_l1477_147709


namespace experiment_duration_in_seconds_l1477_147700

/-- Converts hours to seconds -/
def hoursToSeconds (hours : ℕ) : ℕ := hours * 3600

/-- Converts minutes to seconds -/
def minutesToSeconds (minutes : ℕ) : ℕ := minutes * 60

/-- Represents the duration of an experiment -/
structure ExperimentDuration where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the total seconds of an experiment duration -/
def totalSeconds (duration : ExperimentDuration) : ℕ :=
  hoursToSeconds duration.hours + minutesToSeconds duration.minutes + duration.seconds

/-- Theorem stating that the experiment lasting 2 hours, 45 minutes, and 30 seconds is equivalent to 9930 seconds -/
theorem experiment_duration_in_seconds :
  totalSeconds { hours := 2, minutes := 45, seconds := 30 } = 9930 := by
  sorry


end experiment_duration_in_seconds_l1477_147700


namespace smallest_n_congruence_l1477_147720

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(725 * m ≡ 1275 * m [ZMOD 35])) ∧ 
  (725 * n ≡ 1275 * n [ZMOD 35]) :=
by sorry

end smallest_n_congruence_l1477_147720


namespace other_endpoint_coordinates_l1477_147772

/-- Given a line segment with midpoint (3, 1) and one endpoint at (7, -3),
    prove that the other endpoint is at (-1, 5). -/
theorem other_endpoint_coordinates :
  ∀ (x y : ℝ),
  (3 = (7 + x) / 2) →
  (1 = (-3 + y) / 2) →
  x = -1 ∧ y = 5 := by
sorry

end other_endpoint_coordinates_l1477_147772


namespace angle_B_in_arithmetic_sequence_triangle_l1477_147749

/-- In a triangle ABC where the interior angles A, B, and C form an arithmetic sequence, 
    the measure of angle B is 60°. -/
theorem angle_B_in_arithmetic_sequence_triangle : 
  ∀ (A B C : ℝ),
  (0 < A) ∧ (A < 180) ∧
  (0 < B) ∧ (B < 180) ∧
  (0 < C) ∧ (C < 180) ∧
  (A + B + C = 180) ∧
  (2 * B = A + C) →
  B = 60 := by
sorry

end angle_B_in_arithmetic_sequence_triangle_l1477_147749


namespace triangle_properties_l1477_147738

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a - 2 * t.c * Real.cos t.B = t.c) 
  (h2 : Real.cos t.B = 1/3) 
  (h3 : t.c = 3) 
  (h4 : 0 < t.A ∧ t.A < Real.pi/2) 
  (h5 : 0 < t.B ∧ t.B < Real.pi/2) 
  (h6 : 0 < t.C ∧ t.C < Real.pi/2) :
  t.b = 2 * Real.sqrt 6 ∧ 
  1/2 < Real.sin t.C ∧ Real.sin t.C < Real.sqrt 2 / 2 := by
  sorry

end triangle_properties_l1477_147738


namespace nate_running_distance_l1477_147776

/-- The total distance Nate ran given the length of a football field and additional distance -/
def total_distance (field_length : ℝ) (additional_distance : ℝ) : ℝ :=
  4 * field_length + additional_distance

/-- Theorem stating that Nate's total running distance is 1172 meters -/
theorem nate_running_distance :
  total_distance 168 500 = 1172 := by
  sorry

end nate_running_distance_l1477_147776


namespace peter_class_size_l1477_147719

/-- The number of hands in Peter's class, excluding Peter's hands -/
def hands_excluding_peter : ℕ := 20

/-- The number of hands each student has -/
def hands_per_student : ℕ := 2

/-- The total number of hands in the class, including Peter's -/
def total_hands : ℕ := hands_excluding_peter + hands_per_student

/-- The number of students in Peter's class, including Peter -/
def students_in_class : ℕ := total_hands / hands_per_student

theorem peter_class_size :
  students_in_class = 11 :=
sorry

end peter_class_size_l1477_147719


namespace polynomial_with_positive_integer_roots_l1477_147740

theorem polynomial_with_positive_integer_roots :
  ∀ (a b c : ℝ),
  (∃ (p q r s : ℕ+),
    (∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + b = (x - p)*(x - q)*(x - r)*(x - s)) ∧
    p + q + r + s = -a ∧
    p*q + p*r + p*s + q*r + q*s + r*s = b ∧
    p*q*r + p*q*s + p*r*s + q*r*s = -c ∧
    p*q*r*s = b) →
  ((a = -21 ∧ b = 112 ∧ c = -204) ∨ (a = -12 ∧ b = 48 ∧ c = -80)) := by
sorry

end polynomial_with_positive_integer_roots_l1477_147740


namespace stratified_sampling_male_athletes_l1477_147712

/-- Represents the number of male athletes to be drawn in a stratified sampling -/
def male_athletes_drawn (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ) : ℚ :=
  (male_athletes : ℚ) * (sample_size : ℚ) / (total_athletes : ℚ)

/-- Theorem stating that in the given scenario, 4 male athletes should be drawn -/
theorem stratified_sampling_male_athletes :
  male_athletes_drawn 30 20 6 = 4 := by
  sorry

end stratified_sampling_male_athletes_l1477_147712


namespace inequality_system_solution_set_l1477_147725

theorem inequality_system_solution_set :
  let S := {x : ℝ | x - 3 < 2 ∧ 3 * x + 1 ≥ 2 * x}
  S = {x : ℝ | -1 ≤ x ∧ x < 5} := by
  sorry

end inequality_system_solution_set_l1477_147725


namespace volunteer_arrangement_count_volunteer_arrangement_problem_l1477_147758

theorem volunteer_arrangement_count : Nat → Nat → Nat
  | n, k => if k ≤ n then n.factorial / (n - k).factorial else 0

theorem volunteer_arrangement_problem :
  volunteer_arrangement_count 6 4 = 360 := by
  sorry

end volunteer_arrangement_count_volunteer_arrangement_problem_l1477_147758


namespace total_amount_after_three_years_l1477_147702

/-- Calculates the compound interest for a given principal, rate, and time --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The original bill amount --/
def initial_amount : ℝ := 350

/-- The interest rate for the first year --/
def first_year_rate : ℝ := 0.03

/-- The interest rate for the second and third years --/
def later_years_rate : ℝ := 0.05

/-- The total time period in years --/
def total_years : ℕ := 3

theorem total_amount_after_three_years :
  let amount_after_first_year := compound_interest initial_amount first_year_rate 1
  let final_amount := compound_interest amount_after_first_year later_years_rate 2
  ∃ ε > 0, |final_amount - 397.45| < ε :=
sorry

end total_amount_after_three_years_l1477_147702


namespace mr_green_potato_yield_l1477_147724

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (garden : GardenDimensions) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (garden.length : ℝ) * step_length * (garden.width : ℝ) * step_length * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden -/
theorem mr_green_potato_yield :
  let garden := GardenDimensions.mk 18 25
  let step_length := 2.5
  let yield_per_sqft := 0.75
  expected_potato_yield garden step_length yield_per_sqft = 2109.375 := by
  sorry

end mr_green_potato_yield_l1477_147724


namespace axis_of_symmetry_compare_points_range_of_t_max_t_value_l1477_147781

-- Define the parabola
def parabola (t x y : ℝ) : Prop := y = x^2 - 2*t*x + 1

-- Theorem 1: Axis of symmetry
theorem axis_of_symmetry (t : ℝ) :
  ∀ x y : ℝ, parabola t x y → (∀ ε > 0, ∃ y₁ y₂ : ℝ, 
    parabola t (t - ε) y₁ ∧ parabola t (t + ε) y₂ ∧ y₁ = y₂) :=
sorry

-- Theorem 2: Comparing points
theorem compare_points (t m n : ℝ) :
  parabola t (t-2) m → parabola t (t+3) n → n > m :=
sorry

-- Theorem 3: Range of t
theorem range_of_t (t : ℝ) :
  (∀ x₁ y₁ x₂ y₂ : ℝ, -1 ≤ x₁ → x₁ < 3 → x₂ = 3 →
    parabola t x₁ y₁ → parabola t x₂ y₂ → y₁ ≤ y₂) → t ≤ 1 :=
sorry

-- Theorem 4: Maximum value of t
theorem max_t_value :
  ∃ t_max : ℝ, t_max = 5 ∧
  ∀ t y₁ y₂ : ℝ, parabola t (t+1) y₁ → parabola t (2*t-4) y₂ → y₁ ≥ y₂ → t ≤ t_max :=
sorry

end axis_of_symmetry_compare_points_range_of_t_max_t_value_l1477_147781


namespace absolute_value_equation_unique_solution_l1477_147750

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x + 3| := by
  sorry

end absolute_value_equation_unique_solution_l1477_147750


namespace domain_of_linear_function_domain_of_rational_function_domain_of_square_root_function_domain_of_reciprocal_square_root_function_domain_of_rational_function_with_linear_denominator_domain_of_arcsin_function_l1477_147704

-- Function 1: z = 4 - x - 2y
theorem domain_of_linear_function (x y : ℝ) :
  ∃ z : ℝ, z = 4 - x - 2*y :=
sorry

-- Function 2: p = 3 / (x^2 + y^2)
theorem domain_of_rational_function (x y : ℝ) :
  (x ≠ 0 ∨ y ≠ 0) → ∃ p : ℝ, p = 3 / (x^2 + y^2) :=
sorry

-- Function 3: z = √(1 - x^2 - y^2)
theorem domain_of_square_root_function (x y : ℝ) :
  x^2 + y^2 ≤ 1 → ∃ z : ℝ, z = Real.sqrt (1 - x^2 - y^2) :=
sorry

-- Function 4: q = 1 / √(xy)
theorem domain_of_reciprocal_square_root_function (x y : ℝ) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) → ∃ q : ℝ, q = 1 / Real.sqrt (x*y) :=
sorry

-- Function 5: u = (x^2 * y) / (2x + 1 - y)
theorem domain_of_rational_function_with_linear_denominator (x y : ℝ) :
  2*x + 1 - y ≠ 0 → ∃ u : ℝ, u = (x^2 * y) / (2*x + 1 - y) :=
sorry

-- Function 6: v = arcsin(x + y)
theorem domain_of_arcsin_function (x y : ℝ) :
  -1 ≤ x + y ∧ x + y ≤ 1 → ∃ v : ℝ, v = Real.arcsin (x + y) :=
sorry

end domain_of_linear_function_domain_of_rational_function_domain_of_square_root_function_domain_of_reciprocal_square_root_function_domain_of_rational_function_with_linear_denominator_domain_of_arcsin_function_l1477_147704


namespace park_visitors_l1477_147799

theorem park_visitors (saturday_visitors : ℕ) (sunday_extra : ℕ) : 
  saturday_visitors = 200 → sunday_extra = 40 → 
  saturday_visitors + (saturday_visitors + sunday_extra) = 440 := by
sorry

end park_visitors_l1477_147799


namespace ed_doug_marble_difference_l1477_147710

theorem ed_doug_marble_difference :
  ∀ (ed_initial : ℕ) (ed_lost : ℕ) (ed_current : ℕ) (doug : ℕ),
    ed_initial > doug →
    ed_lost = 20 →
    ed_current = 17 →
    doug = 5 →
    ed_initial = ed_current + ed_lost →
    ed_initial - doug = 32 := by
  sorry

end ed_doug_marble_difference_l1477_147710


namespace ratio_limit_is_27_l1477_147744

/-- The ratio of the largest element to the sum of other elements in the geometric series -/
def ratio (n : ℕ) : ℚ :=
  let a := 3
  let r := 10
  (a * r^n) / (a * (r^n - 1) / (r - 1))

/-- The limit of the ratio as n approaches infinity is 27 -/
theorem ratio_limit_is_27 : ∀ ε > 0, ∃ N, ∀ n ≥ N, |ratio n - 27| < ε :=
sorry

end ratio_limit_is_27_l1477_147744


namespace unique_interior_point_is_median_intersection_l1477_147791

/-- A point with integer coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle with vertices on lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Predicate to check if a point is inside a triangle -/
def IsInside (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Predicate to check if a point is on the boundary of a triangle -/
def IsOnBoundary (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- The intersection point of the medians of a triangle -/
def MedianIntersection (t : LatticeTriangle) : LatticePoint := sorry

/-- Main theorem -/
theorem unique_interior_point_is_median_intersection (t : LatticeTriangle) 
  (h1 : ∀ p : LatticePoint, IsOnBoundary p t → (p = t.A ∨ p = t.B ∨ p = t.C))
  (h2 : ∃! O : LatticePoint, IsInside O t) :
  ∃ O : LatticePoint, IsInside O t ∧ O = MedianIntersection t := by
  sorry

end unique_interior_point_is_median_intersection_l1477_147791


namespace quadratic_roots_l1477_147782

theorem quadratic_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  (2*(a + b))^2 - 4*3*a*(b + c) > 0 := by sorry

end quadratic_roots_l1477_147782


namespace pqr_value_exists_l1477_147777

theorem pqr_value_exists :
  ∃ (p q r : ℝ), (p * q) * (q * r) * (r * p) = 16 ∧ p * q * r = 4 :=
by sorry

end pqr_value_exists_l1477_147777


namespace compute_expression_l1477_147711

theorem compute_expression : 18 * (200 / 3 + 50 / 6 + 16 / 18 + 2) = 1402 := by
  sorry

end compute_expression_l1477_147711


namespace binomial_problem_l1477_147717

def binomial_expansion (m n : ℕ) (x : ℝ) : ℝ :=
  (1 + m * x) ^ n

theorem binomial_problem (m n : ℕ) (h1 : m ≠ 0) (h2 : n ≥ 2) :
  (∃ k, k = 5 ∧ ∀ j, j ≠ k → Nat.choose n j ≤ Nat.choose n k) →
  (Nat.choose n 2 * m^2 = 9 * Nat.choose n 1 * m) →
  (m = 2 ∧ n = 10 ∧ (binomial_expansion m n (-9)) % 6 = 1) :=
sorry

end binomial_problem_l1477_147717


namespace quadratic_roots_theorem_l1477_147770

/-- Represents a continued fraction with repeating terms a and b -/
def RepeatingContinuedFraction (a b : ℤ) : ℝ :=
  sorry

/-- The other root of a quadratic equation with integer coefficients -/
def OtherRoot (a b : ℤ) : ℝ :=
  sorry

theorem quadratic_roots_theorem (a b : ℤ) :
  ∃ (p q r : ℤ), 
    (p * (RepeatingContinuedFraction a b)^2 + q * (RepeatingContinuedFraction a b) + r = 0) →
    (OtherRoot a b = -1 / (RepeatingContinuedFraction b a)) :=
  sorry

end quadratic_roots_theorem_l1477_147770


namespace absent_percentage_l1477_147732

def total_students : ℕ := 100
def present_students : ℕ := 86

theorem absent_percentage : 
  (total_students - present_students) * 100 / total_students = 14 := by
  sorry

end absent_percentage_l1477_147732


namespace incorrect_calculation_D_l1477_147728

theorem incorrect_calculation_D :
  (∀ x : ℝ, x * 0 = 0) ∧
  (∀ x y : ℝ, y ≠ 0 → x / y = x * (1 / y)) ∧
  (∀ x y : ℝ, x * (-y) = -(x * y)) →
  ¬(1 / 3 / (-1) = 3 * (-1)) :=
by sorry

end incorrect_calculation_D_l1477_147728


namespace sum_three_digit_even_numbers_l1477_147765

/-- The sum of all even natural numbers between 100 and 998 (inclusive) is 247050. -/
theorem sum_three_digit_even_numbers : 
  (Finset.range 450).sum (fun i => 100 + 2 * i) = 247050 := by
  sorry

end sum_three_digit_even_numbers_l1477_147765


namespace probability_more_ones_than_sixes_l1477_147788

/-- Represents the outcome of rolling a single die -/
inductive DieOutcome
  | One
  | Two
  | Three
  | Four
  | Five
  | Six

/-- Represents the outcome of rolling five dice -/
def FiveDiceRoll := Vector DieOutcome 5

/-- The total number of possible outcomes when rolling five fair six-sided dice -/
def totalOutcomes : Nat := 7776

/-- The number of outcomes where there are more 1's than 6's -/
def favorableOutcomes : Nat := 2676

/-- The probability of rolling more 1's than 6's when rolling five fair six-sided dice -/
def probabilityMoreOnesThanSixes : Rat := favorableOutcomes / totalOutcomes

theorem probability_more_ones_than_sixes :
  probabilityMoreOnesThanSixes = 2676 / 7776 := by
  sorry

end probability_more_ones_than_sixes_l1477_147788


namespace count_triples_eq_two_l1477_147751

/-- The number of positive integer triples (x, y, z) satisfying x · y = 6 and y · z = 15 -/
def count_triples : Nat :=
  (Finset.filter (fun t : Nat × Nat × Nat =>
    t.1 * t.2.1 = 6 ∧ t.2.1 * t.2.2 = 15 ∧
    t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0)
    (Finset.product (Finset.range 7) (Finset.product (Finset.range 4) (Finset.range 16)))).card

theorem count_triples_eq_two : count_triples = 2 := by
  sorry

end count_triples_eq_two_l1477_147751


namespace complex_distance_bounds_l1477_147786

theorem complex_distance_bounds (z : ℂ) (h : Complex.abs (z + 2 - 2*Complex.I) = 1) :
  (∃ w : ℂ, Complex.abs (w + 2 - 2*Complex.I) = 1 ∧ Complex.abs (w - 3 - 2*Complex.I) = 6) ∧
  (∃ v : ℂ, Complex.abs (v + 2 - 2*Complex.I) = 1 ∧ Complex.abs (v - 3 - 2*Complex.I) = 4) ∧
  (∀ u : ℂ, Complex.abs (u + 2 - 2*Complex.I) = 1 → 
    Complex.abs (u - 3 - 2*Complex.I) ≤ 6 ∧ Complex.abs (u - 3 - 2*Complex.I) ≥ 4) := by
  sorry

end complex_distance_bounds_l1477_147786


namespace polynomial_factorization_exists_l1477_147763

theorem polynomial_factorization_exists :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (p q r s : ℤ),
    ∀ (x : ℤ), x * (x - a) * (x - b) * (x - c) + 1 = (x^2 + p*x + q) * (x^2 + r*x + s) :=
by sorry

end polynomial_factorization_exists_l1477_147763


namespace even_function_implies_a_equals_one_l1477_147706

/-- Given that f(x) = x³(a·2ˣ - 2⁻ˣ) is an even function, prove that a = 1 -/
theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 := by
sorry

end even_function_implies_a_equals_one_l1477_147706


namespace volleyball_team_combinations_l1477_147742

theorem volleyball_team_combinations : Nat.choose 16 7 = 11440 := by
  sorry

end volleyball_team_combinations_l1477_147742


namespace exponent_properties_l1477_147769

theorem exponent_properties (a x y : ℝ) (h1 : a^x = 3) (h2 : a^y = 2) :
  a^(x - y) = 3/2 ∧ a^(2*x + y) = 18 := by
  sorry

end exponent_properties_l1477_147769


namespace combination_18_choose_4_l1477_147723

theorem combination_18_choose_4 : Nat.choose 18 4 = 3060 := by
  sorry

end combination_18_choose_4_l1477_147723


namespace vector_values_l1477_147708

-- Define the vectors
def OA (m : ℝ) : Fin 2 → ℝ := ![(-2 : ℝ), m]
def OB (n : ℝ) : Fin 2 → ℝ := ![n, (1 : ℝ)]
def OC : Fin 2 → ℝ := ![(5 : ℝ), (-1 : ℝ)]

-- Define collinearity
def collinear (A B C : Fin 2 → ℝ) : Prop :=
  ∃ (t : ℝ), B - A = t • (C - A)

-- Define perpendicularity
def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

theorem vector_values (m n : ℝ) :
  collinear (OA m) (OB n) OC ∧
  perpendicular (OA m) (OB n) →
  m = 3 ∧ n = 3/2 := by sorry

end vector_values_l1477_147708


namespace cakes_served_total_l1477_147798

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := 5

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := 6

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served over two days -/
def total_cakes : ℕ := lunch_cakes + dinner_cakes + yesterday_cakes

theorem cakes_served_total :
  total_cakes = 14 := by sorry

end cakes_served_total_l1477_147798


namespace jimmy_cards_theorem_l1477_147729

def jimmy_cards_problem (initial_cards : ℕ) (cards_to_bob : ℕ) : Prop :=
  let cards_after_bob := initial_cards - cards_to_bob
  let cards_to_mary := 2 * cards_to_bob
  let final_cards := cards_after_bob - cards_to_mary
  initial_cards = 18 ∧ cards_to_bob = 3 → final_cards = 9

theorem jimmy_cards_theorem : jimmy_cards_problem 18 3 := by
  sorry

end jimmy_cards_theorem_l1477_147729


namespace problem_solution_l1477_147737

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/5 = 5*y) : x = 625/3 := by
  sorry

end problem_solution_l1477_147737


namespace baron_munchausen_claim_false_l1477_147795

theorem baron_munchausen_claim_false : 
  ¬ (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → ∃ m : ℕ, 0 ≤ m ∧ m ≤ 99 ∧ ∃ k : ℕ, (n * 100 + m) = k^2) :=
by sorry

end baron_munchausen_claim_false_l1477_147795


namespace sector_area_120_deg_sqrt3_radius_l1477_147748

theorem sector_area_120_deg_sqrt3_radius (π : ℝ) (h_pi : π = Real.pi) : 
  let angle : ℝ := 120 * π / 180
  let radius : ℝ := Real.sqrt 3
  let area : ℝ := 1/2 * angle * radius^2
  area = π := by
sorry

end sector_area_120_deg_sqrt3_radius_l1477_147748


namespace solve_for_q_l1477_147760

theorem solve_for_q (n m q : ℚ) 
  (eq1 : (3 : ℚ) / 4 = n / 88)
  (eq2 : (3 : ℚ) / 4 = (m + n) / 100)
  (eq3 : (3 : ℚ) / 4 = (q - m) / 150) :
  q = 121.5 := by
sorry

end solve_for_q_l1477_147760


namespace only_2020_is_very_good_l1477_147793

/-- Represents a four-digit number YEAR --/
structure Year where
  Y : Fin 10
  E : Fin 10
  A : Fin 10
  R : Fin 10

/-- Checks if a Year is in the 21st century --/
def is_21st_century (year : Year) : Prop :=
  2001 ≤ year.Y * 1000 + year.E * 100 + year.A * 10 + year.R ∧ 
  year.Y * 1000 + year.E * 100 + year.A * 10 + year.R ≤ 2100

/-- The system of linear equations for a given Year --/
def system_has_multiple_solutions (year : Year) : Prop :=
  ∃ (x y z w : ℝ) (x' y' z' w' : ℝ),
    (x ≠ x' ∨ y ≠ y' ∨ z ≠ z' ∨ w ≠ w') ∧
    (year.Y * x + year.E * y + year.A * z + year.R * w = year.Y) ∧
    (year.R * x + year.Y * y + year.E * z + year.A * w = year.E) ∧
    (year.A * x + year.R * y + year.Y * z + year.E * w = year.A) ∧
    (year.E * x + year.A * y + year.R * z + year.Y * w = year.R) ∧
    (year.Y * x' + year.E * y' + year.A * z' + year.R * w' = year.Y) ∧
    (year.R * x' + year.Y * y' + year.E * z' + year.A * w' = year.E) ∧
    (year.A * x' + year.R * y' + year.Y * z' + year.E * w' = year.A) ∧
    (year.E * x' + year.A * y' + year.R * z' + year.Y * w' = year.R)

/-- The main theorem stating that 2020 is the only "very good" year in the 21st century --/
theorem only_2020_is_very_good :
  ∀ (year : Year),
    is_21st_century year ∧ system_has_multiple_solutions year ↔
    year.Y = 2 ∧ year.E = 0 ∧ year.A = 2 ∧ year.R = 0 :=
sorry

end only_2020_is_very_good_l1477_147793


namespace tangent_line_sum_l1477_147718

/-- Given a function f: ℝ → ℝ with a tangent line y=-x+8 at x=5, prove f(5) + f'(5) = 2 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : ∀ x, f 5 + (deriv f 5) * (x - 5) = -x + 8) : 
  f 5 + deriv f 5 = 2 := by
  sorry

end tangent_line_sum_l1477_147718


namespace existence_of_small_power_l1477_147754

theorem existence_of_small_power (p e : ℝ) (h1 : 0 < p) (h2 : p < 1) (h3 : e > 0) :
  ∃ n : ℕ, (1 - p) ^ n < e := by
sorry

end existence_of_small_power_l1477_147754


namespace cube_surface_area_equals_prism_volume_l1477_147743

/-- The surface area of a cube with volume equal to a rectangular prism of dimensions 12 × 3 × 18 is equal to the volume of the prism. -/
theorem cube_surface_area_equals_prism_volume :
  let prism_length : ℝ := 12
  let prism_width : ℝ := 3
  let prism_height : ℝ := 18
  let prism_volume := prism_length * prism_width * prism_height
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = prism_volume := by
  sorry

end cube_surface_area_equals_prism_volume_l1477_147743


namespace max_value_inequality_l1477_147701

theorem max_value_inequality (x y : ℝ) (hx : |x - 1| ≤ 1) (hy : |y - 2| ≤ 1) :
  |x - y + 1| ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), |x₀ - 1| ≤ 1 ∧ |y₀ - 2| ≤ 1 ∧ |x₀ - y₀ + 1| = 2 :=
by sorry

end max_value_inequality_l1477_147701


namespace dogwood_trees_to_cut_l1477_147747

/-- The number of dogwood trees in the first part of the park -/
def trees_part1 : ℝ := 5.0

/-- The number of dogwood trees in the second part of the park -/
def trees_part2 : ℝ := 4.0

/-- The number of dogwood trees that will be left after the work is done -/
def trees_left : ℝ := 2.0

/-- The number of dogwood trees to be cut down -/
def trees_to_cut : ℝ := trees_part1 + trees_part2 - trees_left

theorem dogwood_trees_to_cut :
  trees_to_cut = 7.0 := by sorry

end dogwood_trees_to_cut_l1477_147747


namespace integer_sum_and_square_is_twelve_l1477_147787

theorem integer_sum_and_square_is_twelve : ∃ N : ℕ+, (N : ℤ)^2 + (N : ℤ) = 12 := by
  sorry

end integer_sum_and_square_is_twelve_l1477_147787


namespace greatest_integer_less_than_M_over_100_l1477_147768

def M : ℚ :=
  (1 / (3 * 4 * 5 * 6 * 16 * 17 * 18) +
   1 / (4 * 5 * 6 * 7 * 15 * 16 * 17 * 18) +
   1 / (5 * 6 * 7 * 8 * 14 * 15 * 16 * 17 * 18) +
   1 / (6 * 7 * 8 * 9 * 13 * 14 * 15 * 16 * 17 * 18) +
   1 / (7 * 8 * 9 * 10 * 12 * 13 * 14 * 15 * 16 * 17 * 18) +
   1 / (8 * 9 * 10 * 11 * 11 * 12 * 13 * 14 * 15 * 16 * 17 * 18) +
   1 / (9 * 10 * 11 * 12 * 10 * 11 * 12 * 13 * 14 * 15 * 16 * 17 * 18)) * (2 * 17 * 18)

theorem greatest_integer_less_than_M_over_100 : 
  ∀ n : ℤ, n ≤ ⌊M / 100⌋ ↔ n ≤ 145 :=
by sorry

end greatest_integer_less_than_M_over_100_l1477_147768


namespace monthly_income_calculation_l1477_147797

/-- Proves that if 32% of a person's monthly income is Rs. 3800, then their monthly income is Rs. 11875. -/
theorem monthly_income_calculation (deposit : ℝ) (percentage : ℝ) (monthly_income : ℝ) 
  (h1 : deposit = 3800)
  (h2 : percentage = 32)
  (h3 : deposit = (percentage / 100) * monthly_income) :
  monthly_income = 11875 := by
  sorry

end monthly_income_calculation_l1477_147797


namespace intersection_value_l1477_147790

theorem intersection_value (m : ℝ) (B : Set ℝ) : 
  ({1, m - 2} : Set ℝ) ∩ B = {2} → m = 4 := by
sorry

end intersection_value_l1477_147790


namespace playground_area_l1477_147792

theorem playground_area (perimeter : ℝ) (length width : ℝ) : 
  perimeter = 100 → 
  length = 3 * width → 
  2 * length + 2 * width = perimeter → 
  length * width = 468.75 := by
sorry

end playground_area_l1477_147792


namespace quadratic_root_condition_l1477_147707

theorem quadratic_root_condition (a b : ℝ) : 
  a > 0 → 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x - 1 = 0 ∧ a * y^2 + b * y - 1 = 0) →
  (∃ r : ℝ, 1 < r ∧ r < 2 ∧ a * r^2 + b * r - 1 = 0) →
  ∀ z : ℝ, z > -1 → ∃ a' b' : ℝ, a' - b' = z ∧ 
    a' > 0 ∧
    (∃ x y : ℝ, x ≠ y ∧ a' * x^2 + b' * x - 1 = 0 ∧ a' * y^2 + b' * y - 1 = 0) ∧
    (∃ r : ℝ, 1 < r ∧ r < 2 ∧ a' * r^2 + b' * r - 1 = 0) :=
by sorry

end quadratic_root_condition_l1477_147707


namespace inequality_proof_l1477_147757

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end inequality_proof_l1477_147757


namespace university_volunteer_selection_l1477_147714

theorem university_volunteer_selection (undergrad : ℕ) (masters : ℕ) (doctoral : ℕ) 
  (selected_doctoral : ℕ) (h1 : undergrad = 4400) (h2 : masters = 400) (h3 : doctoral = 200) 
  (h4 : selected_doctoral = 10) :
  (undergrad + masters + doctoral) * selected_doctoral / doctoral = 250 := by
  sorry

end university_volunteer_selection_l1477_147714


namespace december_24_is_sunday_l1477_147727

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in November or December -/
structure Date where
  month : Nat
  day : Nat

/-- Function to determine the day of the week for a given date -/
def dayOfWeek (date : Date) (thanksgivingDOW : DayOfWeek) : DayOfWeek :=
  sorry

theorem december_24_is_sunday 
  (thanksgiving : Date)
  (h1 : thanksgiving.month = 11)
  (h2 : thanksgiving.day = 24)
  (h3 : dayOfWeek thanksgiving DayOfWeek.Friday = DayOfWeek.Friday) :
  dayOfWeek ⟨12, 24⟩ DayOfWeek.Friday = DayOfWeek.Sunday :=
sorry

end december_24_is_sunday_l1477_147727


namespace journey_remaining_distance_l1477_147746

/-- Represents a journey with two stopovers and a final destination -/
structure Journey where
  total_distance : ℕ
  first_stopover : ℕ
  second_stopover : ℕ

/-- Calculates the remaining distance to the destination after the second stopover -/
def remaining_distance (j : Journey) : ℕ :=
  j.total_distance - (j.first_stopover + j.second_stopover)

/-- Theorem: For the given journey, the remaining distance is 68 miles -/
theorem journey_remaining_distance :
  let j : Journey := {
    total_distance := 436,
    first_stopover := 132,
    second_stopover := 236
  }
  remaining_distance j = 68 := by
  sorry

end journey_remaining_distance_l1477_147746


namespace smallest_m_value_l1477_147726

def count_quadruplets (m : ℕ) : ℕ :=
  sorry

theorem smallest_m_value :
  ∃ (m : ℕ),
    (count_quadruplets m = 125000) ∧
    (∀ (a b c d : ℕ), (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 125 ∧
                       Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = m) →
                      (count_quadruplets m = 125000)) ∧
    (∀ (m' : ℕ), m' < m →
      (count_quadruplets m' ≠ 125000 ∨
       ∃ (a b c d : ℕ), Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 125 ∧
                         Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = m' ∧
                         count_quadruplets m' ≠ 125000)) ∧
    m = 9450000 :=
by sorry

end smallest_m_value_l1477_147726


namespace deck_size_l1477_147713

theorem deck_size (r b u : ℕ) : 
  r + b + u > 0 →
  r / (r + b + u : ℚ) = 1 / 5 →
  r / ((r + b + u + 3) : ℚ) = 1 / 6 →
  r + b + u = 15 := by
sorry

end deck_size_l1477_147713


namespace total_money_l1477_147755

/-- Given three people A, B, and C with the following conditions:
  1. A and C together have 200 rupees
  2. B and C together have 360 rupees
  3. C has 60 rupees
  Prove that the total amount they have is 500 rupees -/
theorem total_money (A B C : ℕ) 
  (h1 : A + C = 200) 
  (h2 : B + C = 360) 
  (h3 : C = 60) : 
  A + B + C = 500 := by
  sorry

end total_money_l1477_147755


namespace f_has_two_zeros_l1477_147736

def f (x : ℝ) := x^2 - x - 1

theorem f_has_two_zeros : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end f_has_two_zeros_l1477_147736


namespace person_a_speed_l1477_147703

theorem person_a_speed (v_a v_b : ℝ) : 
  v_a > v_b →
  8 * (v_a + v_b) = 6 * (v_a + v_b + 4) →
  6 * ((v_a + 2) - (v_b + 2)) = 6 →
  v_a = 6.5 := by
sorry

end person_a_speed_l1477_147703


namespace divisible_by_six_l1477_147785

theorem divisible_by_six (n : ℤ) : ∃ k : ℤ, n * (n^2 + 5) = 6 * k := by
  sorry

end divisible_by_six_l1477_147785


namespace edward_good_games_l1477_147773

def games_from_friend : ℕ := 41
def games_from_garage_sale : ℕ := 14
def non_working_games : ℕ := 31

theorem edward_good_games :
  games_from_friend + games_from_garage_sale - non_working_games = 24 := by
  sorry

end edward_good_games_l1477_147773


namespace cookie_production_cost_l1477_147739

/-- The cost to produce one cookie -/
def production_cost : ℝ := sorry

/-- The selling price of one cookie -/
def selling_price : ℝ := 1.2 * production_cost

/-- The number of cookies sold -/
def cookies_sold : ℕ := 50

/-- The total revenue from selling the cookies -/
def total_revenue : ℝ := 60

theorem cookie_production_cost :
  production_cost = 1 :=
by sorry

end cookie_production_cost_l1477_147739


namespace wheel_probability_l1477_147745

theorem wheel_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 := by
  sorry

end wheel_probability_l1477_147745


namespace polynomial_remainder_l1477_147735

theorem polynomial_remainder (x : ℝ) : 
  (4*x^3 - 9*x^2 + 12*x - 14) % (2*x - 4) = 6 := by sorry

end polynomial_remainder_l1477_147735


namespace probability_of_desired_event_l1477_147775

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of coins being flipped -/
structure CoinSet :=
(penny : CoinFlip)
(nickel : CoinFlip)
(dime : CoinFlip)
(quarter : CoinFlip)
(fifty_cent : CoinFlip)

/-- The total number of possible outcomes when flipping 5 coins -/
def total_outcomes : ℕ := 32

/-- Predicate for the desired event: at least penny, dime, and 50-cent coin are heads -/
def desired_event (cs : CoinSet) : Prop :=
  cs.penny = CoinFlip.Heads ∧ cs.dime = CoinFlip.Heads ∧ cs.fifty_cent = CoinFlip.Heads

/-- The number of outcomes satisfying the desired event -/
def successful_outcomes : ℕ := 4

/-- Theorem stating the probability of the desired event -/
theorem probability_of_desired_event :
  (successful_outcomes : ℚ) / total_outcomes = 1 / 8 := by sorry

end probability_of_desired_event_l1477_147775


namespace pure_imaginary_fraction_l1477_147784

/-- If (a + i) / (1 - i) is a pure imaginary number, then a = 1 -/
theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + I) / (1 - I) = I * b) → a = 1 := by
  sorry

end pure_imaginary_fraction_l1477_147784


namespace equation_solution_l1477_147783

theorem equation_solution : 
  {x : ℝ | (12 - 3*x)^2 = x^2} = {3, 6} := by sorry

end equation_solution_l1477_147783


namespace modulo_congruence_unique_solution_l1477_147794

theorem modulo_congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 45689 ≡ n [ZMOD 23] ∧ n = 11 := by
  sorry

end modulo_congruence_unique_solution_l1477_147794


namespace problem_statement_l1477_147764

theorem problem_statement (x y : ℝ) (h : |x - 3| + Real.sqrt (y - 2) = 0) : 
  (y - x)^2023 = -1 := by
  sorry

end problem_statement_l1477_147764


namespace multiply_subtract_distribute_compute_expression_l1477_147779

theorem multiply_subtract_distribute (a b c : ℕ) : a * c - b * c = (a - b) * c := by sorry

theorem compute_expression : 65 * 1313 - 25 * 1313 = 52520 := by sorry

end multiply_subtract_distribute_compute_expression_l1477_147779


namespace no_conclusive_deduction_l1477_147733

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for Bars, Fins, and Grips
variable (Bar Fin Grip : U → Prop)

-- Define the given conditions
variable (some_bars_not_fins : ∃ x, Bar x ∧ ¬Fin x)
variable (no_fins_are_grips : ∀ x, Fin x → ¬Grip x)

-- Define the statements to be proved
def some_bars_not_grips := ∃ x, Bar x ∧ ¬Grip x
def some_grips_not_bars := ∃ x, Grip x ∧ ¬Bar x
def no_bar_is_grip := ∀ x, Bar x → ¬Grip x
def some_bars_are_grips := ∃ x, Bar x ∧ Grip x

-- Theorem stating that none of the above statements can be conclusively deduced
theorem no_conclusive_deduction :
  ¬(some_bars_not_grips U Bar Grip ∨
     some_grips_not_bars U Grip Bar ∨
     no_bar_is_grip U Bar Grip ∨
     some_bars_are_grips U Bar Grip) :=
sorry

end no_conclusive_deduction_l1477_147733


namespace shirts_sold_l1477_147762

theorem shirts_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) : 
  initial = 49 → remaining = 28 → sold = initial - remaining → sold = 21 := by
sorry

end shirts_sold_l1477_147762


namespace log_identity_l1477_147796

/-- Given real numbers a and b greater than 1 satisfying lg(a + b) = lg(a) + lg(b),
    prove that lg(a - 1) + lg(b - 1) = 0 and lg(1/a + 1/b) = 0 -/
theorem log_identity (a b : ℝ) (ha : a > 1) (hb : b > 1) 
    (h : Real.log (a + b) = Real.log a + Real.log b) :
  Real.log (a - 1) + Real.log (b - 1) = 0 ∧ Real.log (1/a + 1/b) = 0 := by
  sorry

end log_identity_l1477_147796


namespace modulus_of_complex_square_root_l1477_147789

theorem modulus_of_complex_square_root (w : ℂ) (h : w^2 = -48 + 36*I) : 
  Complex.abs w = 2 * Real.sqrt 15 := by
  sorry

end modulus_of_complex_square_root_l1477_147789


namespace wire_length_around_square_field_l1477_147774

theorem wire_length_around_square_field (area : ℝ) (n : ℕ) (wire_length : ℝ) : 
  area = 69696 → n = 15 → wire_length = 15840 → 
  wire_length = n * 4 * Real.sqrt area := by
  sorry

end wire_length_around_square_field_l1477_147774


namespace line_plane_parallel_l1477_147756

-- Define the types for lines and planes
variable (L : Type) [LinearOrder L]
variable (P : Type)

-- Define the relations
variable (subset : L → P → Prop)  -- line is contained in plane
variable (parallel : L → P → Prop)  -- line is parallel to plane
variable (coplanar : L → L → Prop)  -- two lines are coplanar
variable (parallel_lines : L → L → Prop)  -- two lines are parallel

-- State the theorem
theorem line_plane_parallel (m n : L) (α : P) :
  subset m α → parallel n α → coplanar m n → parallel_lines m n := by sorry

end line_plane_parallel_l1477_147756


namespace matrix_equality_l1477_147780

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = ![![10, 6], ![-4, 2]]) : 
  B * A = ![![10, 6], ![-4, 2]] := by
  sorry

end matrix_equality_l1477_147780


namespace product_of_repeating_decimal_and_five_l1477_147705

-- Define the repeating decimal 0.456̄
def repeating_decimal : ℚ := 456 / 999

-- State the theorem
theorem product_of_repeating_decimal_and_five :
  repeating_decimal * 5 = 760 / 333 := by
  sorry

end product_of_repeating_decimal_and_five_l1477_147705


namespace prism_min_faces_and_pyramid_min_vertices_l1477_147766

/-- A prism is a three-dimensional geometric shape with two parallel polygonal bases and rectangular faces connecting corresponding edges of the bases. -/
structure Prism where
  bases : ℕ -- number of sides in each base
  height : ℝ
  mk_pos : height > 0

/-- A pyramid is a three-dimensional geometric shape with a polygonal base and triangular faces meeting at a point (apex). -/
structure Pyramid where
  base_sides : ℕ -- number of sides in the base
  height : ℝ
  mk_pos : height > 0

/-- The number of faces in a prism. -/
def Prism.num_faces (p : Prism) : ℕ := p.bases + 2

/-- The number of vertices in a pyramid. -/
def Pyramid.num_vertices (p : Pyramid) : ℕ := p.base_sides + 1

theorem prism_min_faces_and_pyramid_min_vertices :
  (∀ p : Prism, p.num_faces ≥ 5) ∧
  (∀ p : Pyramid, p.num_vertices ≥ 4) := by
  sorry

end prism_min_faces_and_pyramid_min_vertices_l1477_147766


namespace digits_until_2014_l1477_147741

def odd_sequence (n : ℕ) : ℕ := 2 * n - 1

def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

def total_digits (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => digit_count (odd_sequence (i + 1)))

theorem digits_until_2014 :
  ∃ n : ℕ, odd_sequence n > 2014 ∧ total_digits (n - 1) = 7850 := by sorry

end digits_until_2014_l1477_147741


namespace sequence_general_term_l1477_147759

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 3^n) : 
  ∀ n : ℕ, n ≥ 1 → a n = (3^n - 1) / 2 := by
sorry

end sequence_general_term_l1477_147759


namespace one_student_passes_probability_l1477_147771

/-- The probability that exactly one out of three students passes, given their individual passing probabilities -/
theorem one_student_passes_probability
  (p_jia p_yi p_bing : ℚ)
  (h_jia : p_jia = 4 / 5)
  (h_yi : p_yi = 3 / 5)
  (h_bing : p_bing = 7 / 10) :
  (p_jia * (1 - p_yi) * (1 - p_bing)) +
  ((1 - p_jia) * p_yi * (1 - p_bing)) +
  ((1 - p_jia) * (1 - p_yi) * p_bing) =
  47 / 250 := by
  sorry

end one_student_passes_probability_l1477_147771
