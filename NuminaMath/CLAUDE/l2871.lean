import Mathlib

namespace article_price_l2871_287121

theorem article_price (decreased_price : ℚ) (decrease_percentage : ℚ) (original_price : ℚ) : 
  decreased_price = 1050 ∧ 
  decrease_percentage = 40 ∧ 
  decreased_price = original_price * (1 - decrease_percentage / 100) → 
  original_price = 1750 := by
sorry

end article_price_l2871_287121


namespace quartic_polynomial_extrema_bounds_l2871_287134

/-- A polynomial of degree 4 with real coefficients -/
def QuarticPolynomial (a₀ a₁ a₂ : ℝ) : ℝ → ℝ := λ x ↦ x^4 + a₁*x^3 + a₂*x^2 + a₁*x + a₀

/-- The local maximum of a function -/
noncomputable def LocalMax (f : ℝ → ℝ) : ℝ := sorry

/-- The local minimum of a function -/
noncomputable def LocalMin (f : ℝ → ℝ) : ℝ := sorry

/-- Theorem: Bounds for the difference between local maximum and minimum of a quartic polynomial -/
theorem quartic_polynomial_extrema_bounds (a₀ a₁ a₂ : ℝ) :
  let f := QuarticPolynomial a₀ a₁ a₂
  let M := LocalMax f
  let m := LocalMin f
  3/10 * (a₁^2/4 - 2*a₂/9)^2 < M - m ∧ M - m < 3 * (a₁^2/4 - 2*a₂/9)^2 := by
  sorry

end quartic_polynomial_extrema_bounds_l2871_287134


namespace ellipse_standard_equation_l2871_287100

/-- Given an ellipse with eccentricity e and focal length 2c, 
    prove that its standard equation is of the form x²/a² + y²/b² = 1 
    where a and b are the semi-major and semi-minor axes respectively. -/
theorem ellipse_standard_equation 
  (e : ℝ) 
  (c : ℝ) 
  (h_e : e = 2/3) 
  (h_c : 2*c = 16) : 
  ∃ (a b : ℝ), 
    (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ↔ 
      (x^2/144 + y^2/80 = 1 ∨ x^2/80 + y^2/144 = 1)) := by
sorry

end ellipse_standard_equation_l2871_287100


namespace right_angle_triangle_identification_l2871_287101

/-- Checks if three lengths can form a right-angled triangle -/
def isRightAngleTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

theorem right_angle_triangle_identification :
  (¬ isRightAngleTriangle 2 3 4) ∧
  (¬ isRightAngleTriangle 3 3 4) ∧
  (isRightAngleTriangle 9 12 15) ∧
  (¬ isRightAngleTriangle 4 5 6) := by
  sorry

#check right_angle_triangle_identification

end right_angle_triangle_identification_l2871_287101


namespace chord_length_l2871_287146

/-- The length of the chord formed by the intersection of the line x = 1 and the circle (x-2)² + y² = 4 is 2√3 -/
theorem chord_length : ∃ (A B : ℝ × ℝ), 
  (A.1 = 1 ∧ (A.1 - 2)^2 + A.2^2 = 4) ∧ 
  (B.1 = 1 ∧ (B.1 - 2)^2 + B.2^2 = 4) ∧ 
  A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
by sorry

end chord_length_l2871_287146


namespace planes_perpendicular_l2871_287132

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel, perpendicular, and subset relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane)
  (h1 : parallel m n)
  (h2 : perpendicular n β)
  (h3 : subset m α) :
  perp_planes α β := by sorry

end planes_perpendicular_l2871_287132


namespace inscribed_circle_diameter_l2871_287137

/-- Given a right triangle with legs of length 8 and 15, the diameter of its inscribed circle is 6 -/
theorem inscribed_circle_diameter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_a : a = 8) (h_b : b = 15) : 
  2 * (a * b) / (a + b + c) = 6 := by
  sorry

end inscribed_circle_diameter_l2871_287137


namespace cubic_difference_l2871_287131

theorem cubic_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) :
  x^3 - y^3 = 176 * Real.sqrt 13 := by
  sorry

end cubic_difference_l2871_287131


namespace prob_different_grades_is_two_thirds_l2871_287189

/-- Represents the number of students in each grade -/
def students_per_grade : ℕ := 2

/-- Represents the total number of students -/
def total_students : ℕ := 2 * students_per_grade

/-- Represents the number of students to be selected -/
def selected_students : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Represents the probability of selecting students from different grades -/
def prob_different_grades : ℚ := 
  (choose students_per_grade 1 * choose students_per_grade 1 : ℚ) / 
  (choose total_students selected_students : ℚ)

theorem prob_different_grades_is_two_thirds : 
  prob_different_grades = 2 / 3 := by sorry

end prob_different_grades_is_two_thirds_l2871_287189


namespace brady_earnings_brady_earnings_200_l2871_287124

/-- Brady's earnings for transcribing recipe cards -/
theorem brady_earnings : ℕ → ℚ
  | cards => 
    let base_pay := (70 : ℚ) / 100 * cards
    let bonus := 10 * (cards / 100 : ℕ)
    base_pay + bonus

/-- Proof of Brady's earnings for 200 cards -/
theorem brady_earnings_200 : brady_earnings 200 = 160 := by
  sorry

end brady_earnings_brady_earnings_200_l2871_287124


namespace no_prime_of_form_3811_l2871_287114

def a (n : ℕ) : ℕ := 38 * 10^n + (10^n - 1)

theorem no_prime_of_form_3811 : ∀ n : ℕ, ¬ Nat.Prime (a n) := by
  sorry

end no_prime_of_form_3811_l2871_287114


namespace opposite_not_positive_implies_non_negative_l2871_287180

theorem opposite_not_positive_implies_non_negative (a : ℝ) :
  (-a ≤ 0) → (a ≥ 0) := by sorry

end opposite_not_positive_implies_non_negative_l2871_287180


namespace commission_rate_is_four_percent_l2871_287181

/-- Calculates the commission rate given base pay, goal earnings, and required sales. -/
def calculate_commission_rate (base_pay : ℚ) (goal_earnings : ℚ) (required_sales : ℚ) : ℚ :=
  ((goal_earnings - base_pay) / required_sales) * 100

/-- Proves that the commission rate is 4% given the problem conditions. -/
theorem commission_rate_is_four_percent
  (base_pay : ℚ)
  (goal_earnings : ℚ)
  (required_sales : ℚ)
  (h1 : base_pay = 190)
  (h2 : goal_earnings = 500)
  (h3 : required_sales = 7750) :
  calculate_commission_rate base_pay goal_earnings required_sales = 4 := by
  sorry

#eval calculate_commission_rate 190 500 7750

end commission_rate_is_four_percent_l2871_287181


namespace smallest_number_theorem_l2871_287122

def is_multiple_of_36 (n : ℕ) : Prop := ∃ k : ℕ, n = 36 * k

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of_36 n ∧ (digit_product n % 9 = 0)

theorem smallest_number_theorem :
  satisfies_conditions 936 ∧ ∀ m : ℕ, m < 936 → ¬(satisfies_conditions m) :=
sorry

end smallest_number_theorem_l2871_287122


namespace condo_units_calculation_l2871_287173

/-- Calculates the total number of units in a condo development -/
theorem condo_units_calculation (total_floors : ℕ) (regular_units_per_floor : ℕ) 
  (penthouse_units_per_floor : ℕ) (penthouse_floors : ℕ) : 
  total_floors = 23 → 
  regular_units_per_floor = 12 → 
  penthouse_units_per_floor = 2 → 
  penthouse_floors = 2 → 
  (total_floors - penthouse_floors) * regular_units_per_floor + 
    penthouse_floors * penthouse_units_per_floor = 256 := by
  sorry

#check condo_units_calculation

end condo_units_calculation_l2871_287173


namespace arithmetic_sequence_sum_l2871_287144

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  roots_property : a 1 + a 2011 = 10 ∧ a 1 * a 2011 = 16

/-- The sum of specific terms in the arithmetic sequence is 15 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) : 
  seq.a 2 + seq.a 1006 + seq.a 2010 = 15 := by
  sorry

end arithmetic_sequence_sum_l2871_287144


namespace urn_probability_l2871_287149

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents a single operation -/
def perform_operation (state : UrnState) (drawn : BallColor) : UrnState :=
  match drawn with
  | BallColor.Red => UrnState.mk (state.red + 3) state.blue
  | BallColor.Blue => UrnState.mk state.red (state.blue + 3)

/-- Represents the sequence of operations -/
def operation_sequence := List BallColor

/-- Calculates the probability of a specific operation sequence -/
def sequence_probability (seq : operation_sequence) : ℚ :=
  sorry

/-- Counts the number of valid operation sequences -/
def count_valid_sequences : ℕ :=
  sorry

theorem urn_probability :
  let initial_state : UrnState := UrnState.mk 2 1
  let final_state : UrnState := UrnState.mk 10 6
  let num_operations : ℕ := 5
  (count_valid_sequences * sequence_probability (List.replicate num_operations BallColor.Red)) = 16/115 :=
sorry

end urn_probability_l2871_287149


namespace average_marks_l2871_287179

def english_marks : ℕ := 96
def math_marks : ℕ := 95
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 97
def biology_marks : ℕ := 95

def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

theorem average_marks :
  (total_marks : ℚ) / num_subjects = 93 := by sorry

end average_marks_l2871_287179


namespace moon_weight_calculation_l2871_287170

/-- The weight of the moon in tons -/
def moon_weight : ℝ := 250

/-- The weight of Mars in tons -/
def mars_weight : ℝ := 2 * moon_weight

/-- The percentage of iron in the moon's composition -/
def iron_percentage : ℝ := 0.5

/-- The percentage of carbon in the moon's composition -/
def carbon_percentage : ℝ := 0.2

/-- The percentage of other elements in the moon's composition -/
def other_percentage : ℝ := 1 - iron_percentage - carbon_percentage

/-- The weight of other elements on Mars in tons -/
def mars_other_elements : ℝ := 150

theorem moon_weight_calculation :
  moon_weight = mars_other_elements / other_percentage / 2 := by sorry

end moon_weight_calculation_l2871_287170


namespace domain_of_composite_function_l2871_287128

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Ioo (-1) 1

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | ∃ y ∈ domain_f, y = 2*x + 1} = Set.Ioo (-1) 0 := by sorry

end domain_of_composite_function_l2871_287128


namespace max_planes_from_four_lines_l2871_287117

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of lines -/
def num_lines : ℕ := 4

/-- The number of lines needed to define a plane -/
def lines_per_plane : ℕ := 2

/-- The maximum number of planes that can be defined by four lines starting from the same point -/
def max_planes : ℕ := choose num_lines lines_per_plane

theorem max_planes_from_four_lines : 
  max_planes = 6 := by sorry

end max_planes_from_four_lines_l2871_287117


namespace product_difference_theorem_l2871_287164

theorem product_difference_theorem (N : ℕ) : ∃ (a b c d : ℕ),
  a + b = c + d ∧ c * d = N * (a * b) + a * b := by
  use 4*N - 2, 1, 2*N, 2*N - 1
  sorry

end product_difference_theorem_l2871_287164


namespace c_oxen_count_l2871_287116

/-- Represents the number of oxen-months for a person's grazing arrangement -/
def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

/-- Calculates the share of rent based on oxen-months and total rent -/
def rent_share (own_oxen_months : ℕ) (total_oxen_months : ℕ) (total_rent : ℕ) : ℕ :=
  (own_oxen_months * total_rent) / total_oxen_months

theorem c_oxen_count (x : ℕ) : 
  oxen_months 10 7 + oxen_months 12 5 + oxen_months x 3 = 130 + 3 * x →
  rent_share (oxen_months x 3) (130 + 3 * x) 280 = 72 →
  x = 15 := by
  sorry

end c_oxen_count_l2871_287116


namespace square_root_of_one_incorrect_l2871_287123

theorem square_root_of_one_incorrect : ¬(∀ x : ℝ, x^2 = 1 → x = 1) := by
  sorry

#check square_root_of_one_incorrect

end square_root_of_one_incorrect_l2871_287123


namespace cube_volume_ratio_and_surface_area_l2871_287169

/-- Edge length of the smaller cube in inches -/
def small_cube_edge : ℝ := 4

/-- Edge length of the larger cube in feet -/
def large_cube_edge : ℝ := 2

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Volume of a cube given its edge length -/
def cube_volume (edge : ℝ) : ℝ := edge ^ 3

/-- Surface area of a cube given its edge length -/
def cube_surface_area (edge : ℝ) : ℝ := 6 * edge ^ 2

theorem cube_volume_ratio_and_surface_area :
  (cube_volume small_cube_edge) / (cube_volume (large_cube_edge * feet_to_inches)) = 1 / 216 ∧
  cube_surface_area small_cube_edge = 96 := by
  sorry

end cube_volume_ratio_and_surface_area_l2871_287169


namespace hyperbola_focus_to_asymptote_distance_l2871_287150

-- Define the hyperbola and its properties
theorem hyperbola_focus_to_asymptote_distance :
  ∀ (M : ℝ × ℝ),
  let F₁ : ℝ × ℝ := (-Real.sqrt 10, 0)
  let F₂ : ℝ × ℝ := (Real.sqrt 10, 0)
  let MF₁ : ℝ × ℝ := (M.1 - F₁.1, M.2 - F₁.2)
  let MF₂ : ℝ × ℝ := (M.2 - F₂.1, M.2 - F₂.2)
  -- M is on the hyperbola
  -- MF₁ · MF₂ = 0
  (MF₁.1 * MF₂.1 + MF₁.2 * MF₂.2 = 0) →
  -- |MF₁| · |MF₂| = 2
  (Real.sqrt (MF₁.1^2 + MF₁.2^2) * Real.sqrt (MF₂.1^2 + MF₂.2^2) = 2) →
  -- The distance from a focus to one of its asymptotes is 1
  (1 : ℝ) = 
    (Real.sqrt 10) / Real.sqrt (1 + (1/3)^2) :=
by sorry


end hyperbola_focus_to_asymptote_distance_l2871_287150


namespace difference_of_squares_fraction_l2871_287115

theorem difference_of_squares_fraction : (235^2 - 221^2) / 14 = 456 := by
  sorry

end difference_of_squares_fraction_l2871_287115


namespace complete_square_quadratic_l2871_287196

theorem complete_square_quadratic (x : ℝ) : 
  x^2 + 6*x + 5 = 0 ↔ (x + 3)^2 = 4 := by
  sorry

end complete_square_quadratic_l2871_287196


namespace circles_and_tangent_line_l2871_287186

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def circle_O2_center : ℝ × ℝ := (3, 3)

-- Define the external tangency condition
def externally_tangent (O1 O2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (O1.1 - O2.1)^2 + (O1.2 - O2.2)^2 = (r1 + r2)^2

-- Theorem statement
theorem circles_and_tangent_line :
  ∃ (r2 : ℝ),
    -- Circle O₂ equation
    (∀ x y : ℝ, (x - 3)^2 + (y - 3)^2 = r2^2) ∧
    -- External tangency condition
    externally_tangent (0, -1) circle_O2_center 2 r2 ∧
    -- Common internal tangent line equation
    (∀ x y : ℝ, circle_O1 x y ∧ (x - 3)^2 + (y - 3)^2 = r2^2 →
      3*x + 4*y = 6) :=
by
  sorry

end circles_and_tangent_line_l2871_287186


namespace fred_balloon_count_l2871_287143

theorem fred_balloon_count (total sam dan : ℕ) (h1 : total = 72) (h2 : sam = 46) (h3 : dan = 16) :
  total - (sam + dan) = 10 := by
  sorry

end fred_balloon_count_l2871_287143


namespace students_playing_sports_l2871_287139

theorem students_playing_sports (basketball cricket both : ℕ) 
  (h1 : basketball = 12) 
  (h2 : cricket = 8) 
  (h3 : both = 3) : 
  basketball + cricket - both = 17 := by
sorry

end students_playing_sports_l2871_287139


namespace expanded_ohara_triple_64_49_l2871_287130

/-- Definition of an Expanded O'Hara triple -/
def is_expanded_ohara_triple (a b x : ℕ) : Prop :=
  2 * (Real.sqrt a + Real.sqrt b) = x

/-- Theorem: If (64, 49, x) is an Expanded O'Hara triple, then x = 30 -/
theorem expanded_ohara_triple_64_49 (x : ℕ) :
  is_expanded_ohara_triple 64 49 x → x = 30 := by
  sorry

end expanded_ohara_triple_64_49_l2871_287130


namespace complex_equation_real_solution_l2871_287177

theorem complex_equation_real_solution (a : ℝ) : 
  (((a : ℂ) / (1 + Complex.I) + (1 + Complex.I) / 2).im = 0) → a = 1 := by
  sorry

end complex_equation_real_solution_l2871_287177


namespace empty_subset_of_set_l2871_287188

theorem empty_subset_of_set : ∅ ⊆ ({2, 0, 1} : Set ℕ) := by sorry

end empty_subset_of_set_l2871_287188


namespace faye_flowers_proof_l2871_287185

theorem faye_flowers_proof (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) (remaining_bouquets : ℕ) 
  (h1 : flowers_per_bouquet = 5)
  (h2 : wilted_flowers = 48)
  (h3 : remaining_bouquets = 8) :
  flowers_per_bouquet * remaining_bouquets + wilted_flowers = 88 :=
by sorry

end faye_flowers_proof_l2871_287185


namespace interior_nodes_line_property_l2871_287125

/-- A point with integer coordinates -/
structure Node where
  x : ℤ
  y : ℤ

/-- A triangle with vertices at nodes -/
structure Triangle where
  a : Node
  b : Node
  c : Node

/-- Checks if a node is inside a triangle -/
def Node.isInside (n : Node) (t : Triangle) : Prop :=
  sorry

/-- Checks if a line through two nodes passes through a vertex of a triangle -/
def Line.passesThroughVertex (p q : Node) (t : Triangle) : Prop :=
  sorry

/-- Checks if a line through two nodes is parallel to a side of a triangle -/
def Line.isParallelToSide (p q : Node) (t : Triangle) : Prop :=
  sorry

/-- Main theorem -/
theorem interior_nodes_line_property (t : Triangle) (p q : Node) :
  p.isInside t ∧ q.isInside t →
  (∀ r : Node, r.isInside t → r = p ∨ r = q) →
  Line.passesThroughVertex p q t ∨ Line.isParallelToSide p q t :=
sorry

end interior_nodes_line_property_l2871_287125


namespace perpendicular_lines_l2871_287159

-- Define the types for lines and planes in 3D space
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_lines 
  (a b c d : Line) (α β : Plane)
  (h1 : perp a b)
  (h2 : perp_line_plane a α)
  (h3 : perp_line_plane c α) :
  perp c b :=
sorry

end perpendicular_lines_l2871_287159


namespace polynomial_expansion_l2871_287106

theorem polynomial_expansion (x : ℝ) :
  (5 * x - 3) * (2 * x^2 + 4 * x + 1) = 10 * x^3 + 14 * x^2 - 7 * x - 3 := by
  sorry

end polynomial_expansion_l2871_287106


namespace digit_sum_problem_l2871_287140

theorem digit_sum_problem (a p v e s r : ℕ) 
  (h1 : a + p = v)
  (h2 : v + e = s)
  (h3 : s + a = r)
  (h4 : p + e + r = 14)
  (h5 : a ≠ 0 ∧ p ≠ 0 ∧ v ≠ 0 ∧ e ≠ 0 ∧ s ≠ 0 ∧ r ≠ 0) :
  s = 7 := by
  sorry

end digit_sum_problem_l2871_287140


namespace patio_rows_l2871_287142

theorem patio_rows (r c : ℕ) : 
  r * c = 30 →
  (r + 4) * (c - 2) = 30 →
  r = 3 :=
by sorry

end patio_rows_l2871_287142


namespace cosine_sum_difference_l2871_287195

theorem cosine_sum_difference : 
  Real.cos (π / 15) - Real.cos (2 * π / 15) - Real.cos (4 * π / 15) + Real.cos (7 * π / 15) = -1/2 := by
  sorry

end cosine_sum_difference_l2871_287195


namespace white_pairs_coincide_l2871_287104

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : Nat
  blue : Nat
  white : Nat

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : Nat
  blue_blue : Nat
  red_white : Nat
  white_white : Nat

/-- The main theorem to prove -/
theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.red = 4 ∧ 
  counts.blue = 7 ∧ 
  counts.white = 10 ∧
  pairs.red_red = 3 ∧
  pairs.blue_blue = 4 ∧
  pairs.red_white = 3 →
  pairs.white_white = 4 := by
  sorry

end white_pairs_coincide_l2871_287104


namespace remainder_of_7n_mod_4_l2871_287113

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_of_7n_mod_4_l2871_287113


namespace M_equals_N_l2871_287120

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | ∃ a b, a ∈ M ∧ b ∈ M ∧ x = a * b}

theorem M_equals_N : M = N := by
  sorry

end M_equals_N_l2871_287120


namespace paint_per_statue_l2871_287108

theorem paint_per_statue (total_paint : ℚ) (num_statues : ℕ) 
  (h1 : total_paint = 7/8)
  (h2 : num_statues = 7) : 
  total_paint / num_statues = 1/8 := by
  sorry

end paint_per_statue_l2871_287108


namespace weight_of_replaced_person_l2871_287153

/-- Theorem: Weight of replaced person in a group
Given a group of 9 persons, if replacing one person with a new person weighing 87.5 kg
increases the average weight by 2.5 kg, then the weight of the replaced person was 65 kg. -/
theorem weight_of_replaced_person
  (n : ℕ) -- number of persons in the group
  (w : ℝ) -- total weight of the original group
  (new_weight : ℝ) -- weight of the new person
  (avg_increase : ℝ) -- increase in average weight
  (h1 : n = 9)
  (h2 : new_weight = 87.5)
  (h3 : avg_increase = 2.5)
  (h4 : (w - (w / n) + new_weight) / n = (w / n) + avg_increase) :
  w / n = 65 :=
sorry

end weight_of_replaced_person_l2871_287153


namespace r_squared_ssr_inverse_relation_l2871_287166

/-- Represents a regression model -/
structure RegressionModel where
  R_squared : ℝ  -- Coefficient of determination
  SSR : ℝ        -- Sum of squares of residuals

/-- States that as R² increases, SSR decreases in a regression model -/
theorem r_squared_ssr_inverse_relation (model1 model2 : RegressionModel) :
  model1.R_squared > model2.R_squared → model1.SSR < model2.SSR := by
  sorry

end r_squared_ssr_inverse_relation_l2871_287166


namespace smallest_positive_b_existence_l2871_287118

theorem smallest_positive_b_existence :
  ∃ (b y : ℝ), b > 0 ∧ y > 0 ∧
  ((9 * Real.sqrt ((3*b)^2 + 2^2) + 5*b^2 - 2) / (Real.sqrt (2 + 5*b^2) - 5) = -4) ∧
  (y^4 + 105*y^2 + 562 = 0) ∧
  (y^2 > 2) ∧
  (b = Real.sqrt (y^2 - 2) / Real.sqrt 5) ∧
  (∀ (b' : ℝ), b' > 0 → 
    ((9 * Real.sqrt ((3*b')^2 + 2^2) + 5*b'^2 - 2) / (Real.sqrt (2 + 5*b'^2) - 5) = -4) →
    b ≤ b') :=
by sorry

end smallest_positive_b_existence_l2871_287118


namespace binary_to_quaternary_conversion_l2871_287110

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ := sorry

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (d : ℕ) : List (Fin 4) := sorry

theorem binary_to_quaternary_conversion :
  let binary : List Bool := [true, true, false, true, true, false, true, true, false, true]
  let quaternary : List (Fin 4) := [3, 1, 1, 3, 1]
  binary_to_decimal binary = (quaternary.map (λ x => x.val)).foldl (λ acc x => acc * 4 + x) 0 :=
by sorry

end binary_to_quaternary_conversion_l2871_287110


namespace volleyball_match_probability_l2871_287191

-- Define the probability of team A winning a set in the first four sets
def p_win_first_four : ℚ := 2 / 3

-- Define the probability of team A winning the fifth set
def p_win_fifth : ℚ := 1 / 2

-- Define the number of ways to choose 2 wins out of 4 sets
def ways_to_win_two_of_four : ℕ := 6

-- State the theorem
theorem volleyball_match_probability :
  let p_three_two := ways_to_win_two_of_four * p_win_first_four^2 * (1 - p_win_first_four)^2 * p_win_fifth
  p_three_two = 4 / 27 := by
  sorry

end volleyball_match_probability_l2871_287191


namespace no_equal_group_division_l2871_287171

theorem no_equal_group_division (k : ℕ) : 
  ¬ ∃ (g1 g2 : List ℕ), 
    (∀ n, n ∈ g1 ∪ g2 ↔ 1 ≤ n ∧ n ≤ k) ∧ 
    (∀ n, n ∈ g1 → n ∉ g2) ∧
    (∀ n, n ∈ g2 → n ∉ g1) ∧
    (g1.foldl (λ acc x => acc * 10 + x) 0 = g2.foldl (λ acc x => acc * 10 + x) 0) :=
by sorry

end no_equal_group_division_l2871_287171


namespace remaining_macaroons_weight_is_103_l2871_287127

/-- Calculates the total weight of remaining macaroons after Steve's snack --/
def remaining_macaroons_weight (
  coconut_count : ℕ)
  (coconut_weight : ℕ)
  (coconut_bags : ℕ)
  (almond_count : ℕ)
  (almond_weight : ℕ)
  (almond_bags : ℕ)
  (white_count : ℕ)
  (white_weight : ℕ) : ℕ :=
  let remaining_coconut := (coconut_count / coconut_bags) * (coconut_bags - 1) * coconut_weight
  let remaining_almond := (almond_count - almond_count / almond_bags / 2) * almond_weight
  let remaining_white := (white_count - 1) * white_weight
  remaining_coconut + remaining_almond + remaining_white

theorem remaining_macaroons_weight_is_103 :
  remaining_macaroons_weight 12 5 4 8 8 2 2 10 = 103 := by
  sorry

#eval remaining_macaroons_weight 12 5 4 8 8 2 2 10

end remaining_macaroons_weight_is_103_l2871_287127


namespace odd_factors_360_l2871_287197

/-- The number of odd factors of 360 -/
def num_odd_factors_360 : ℕ := sorry

/-- Theorem: The number of odd factors of 360 is 6 -/
theorem odd_factors_360 : num_odd_factors_360 = 6 := by sorry

end odd_factors_360_l2871_287197


namespace abcd_16_bits_l2871_287156

def base_16_to_decimal (a b c d : ℕ) : ℕ :=
  a * 16^3 + b * 16^2 + c * 16 + d

def bits_required (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem abcd_16_bits :
  bits_required (base_16_to_decimal 10 11 12 13) = 16 := by
  sorry

end abcd_16_bits_l2871_287156


namespace prob_different_suits_expanded_deck_l2871_287102

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- Calculates the probability of drawing two cards of different suits -/
def prob_different_suits (d : Deck) : ℚ :=
  let remaining_cards := d.total_cards - 1
  let different_suit_cards := d.total_cards - d.cards_per_suit
  different_suit_cards / remaining_cards

/-- Theorem: The probability of drawing two cards of different suits
    from a 78-card deck with 6 suits of 13 cards each is 65/77 -/
theorem prob_different_suits_expanded_deck :
  let d : Deck := ⟨78, 6, 13, rfl⟩
  prob_different_suits d = 65 / 77 := by sorry

end prob_different_suits_expanded_deck_l2871_287102


namespace marys_nickels_l2871_287151

/-- The number of nickels Mary has after receiving some from her dad -/
def total_nickels (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: Mary's total nickels is the sum of her initial nickels and received nickels -/
theorem marys_nickels (initial : ℕ) (received : ℕ) :
  total_nickels initial received = initial + received := by
  sorry

end marys_nickels_l2871_287151


namespace cleaning_earnings_l2871_287190

/-- Calculates the total earnings for cleaning a building -/
def total_earnings (floors : ℕ) (rooms_per_floor : ℕ) (hours_per_room : ℕ) (hourly_rate : ℕ) : ℕ :=
  floors * rooms_per_floor * hours_per_room * hourly_rate

/-- Proves that cleaning a 4-floor building with 10 rooms per floor,
    taking 6 hours per room at $15 per hour, results in $3600 earnings -/
theorem cleaning_earnings :
  total_earnings 4 10 6 15 = 3600 := by
  sorry

end cleaning_earnings_l2871_287190


namespace triangle_uniqueness_l2871_287182

/-- Triangle defined by its three side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

/-- Two triangles are congruent if their corresponding sides are equal -/
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

theorem triangle_uniqueness (t1 t2 : Triangle) : 
  t1.a = t2.a → t1.b = t2.b → t1.c = t2.c → congruent t1 t2 := by
  sorry

#check triangle_uniqueness

end triangle_uniqueness_l2871_287182


namespace symmetric_line_y_axis_l2871_287172

/-- Given a line ax + by + c = 0, returns the line symmetric to it with respect to the y-axis -/
def symmetricLineY (a b c : ℝ) : ℝ × ℝ × ℝ := (-a, b, c)

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def lineThroughPoints (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ × ℝ :=
  let a := y₂ - y₁
  let b := x₁ - x₂
  let c := x₂ * y₁ - x₁ * y₂
  (a, b, c)

theorem symmetric_line_y_axis :
  let original_line := (3, -4, 5)
  let symmetric_line := symmetricLineY 3 (-4) 5
  let y_intercept := (0, 5/4)
  let x_intercept_symmetric := (5/3, 0)
  let line_through_points := lineThroughPoints 0 (5/4) (5/3) 0
  symmetric_line = line_through_points := by sorry

end symmetric_line_y_axis_l2871_287172


namespace work_completion_time_l2871_287147

/-- 
Given:
- A's work rate is half of B's
- A and B together finish a job in 32 days
Prove that B alone will finish the job in 48 days
-/
theorem work_completion_time (a b : ℝ) (h1 : a = (1/2) * b) (h2 : (a + b) * 32 = 1) :
  (1 / b) = 48 := by sorry

end work_completion_time_l2871_287147


namespace raft_sticks_total_l2871_287168

theorem raft_sticks_total (simon_sticks : ℕ) (gerry_sticks : ℕ) (micky_sticks : ℕ) : 
  simon_sticks = 36 →
  gerry_sticks = 2 * (simon_sticks / 3) →
  micky_sticks = simon_sticks + gerry_sticks + 9 →
  simon_sticks + gerry_sticks + micky_sticks = 129 :=
by
  sorry

end raft_sticks_total_l2871_287168


namespace room_length_calculation_l2871_287161

/-- Given a rectangular room with width 12 m, surrounded by a 2 m wide veranda on all sides,
    if the area of the veranda is 132 m², then the length of the room is 17 m. -/
theorem room_length_calculation (room_width : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_width = 12 →
  veranda_width = 2 →
  veranda_area = 132 →
  ∃ (room_length : ℝ),
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) -
    room_length * room_width = veranda_area ∧
    room_length = 17 :=
by sorry

end room_length_calculation_l2871_287161


namespace stating_three_card_draw_probability_value_l2871_287133

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck := Fin 52

/-- The probability of drawing a specific sequence of three cards from a standard deck -/
def three_card_draw_probability : ℚ :=
  -- Probability of first card being a non-heart King
  (3 : ℚ) / 52 *
  -- Probability of second card being a heart (not King of hearts)
  12 / 51 *
  -- Probability of third card being a spade or diamond
  26 / 50

/-- 
Theorem stating that the probability of drawing a non-heart King, 
then a heart (not King of hearts), then a spade or diamond 
from a standard 52-card deck is 26/3675
-/
theorem three_card_draw_probability_value : 
  three_card_draw_probability = 26 / 3675 := by
  sorry

end stating_three_card_draw_probability_value_l2871_287133


namespace points_in_quadrants_I_and_II_l2871_287109

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of Quadrant I -/
def inQuadrantI (p : Point2D) : Prop := p.x > 0 ∧ p.y > 0

/-- Definition of Quadrant II -/
def inQuadrantII (p : Point2D) : Prop := p.x < 0 ∧ p.y > 0

/-- The set of points satisfying the given inequalities -/
def satisfiesInequalities (p : Point2D) : Prop :=
  p.y > 3 * p.x ∧ p.y > 6 - 2 * p.x

theorem points_in_quadrants_I_and_II :
  ∀ p : Point2D, satisfiesInequalities p → inQuadrantI p ∨ inQuadrantII p :=
by sorry

end points_in_quadrants_I_and_II_l2871_287109


namespace percentage_increase_l2871_287107

theorem percentage_increase (x : ℝ) (h1 : x = 62.4) (h2 : x > 52) :
  (x - 52) / 52 * 100 = 20 := by
  sorry

end percentage_increase_l2871_287107


namespace ohara_quadruple_example_l2871_287112

theorem ohara_quadruple_example :
  ∀ (x : ℤ), (Real.sqrt 9 + Real.sqrt 16 + 3^2 : ℝ) = x → x = 16 := by
sorry

end ohara_quadruple_example_l2871_287112


namespace robin_gum_count_l2871_287158

theorem robin_gum_count (initial_gum : ℝ) (additional_gum : ℝ) (total_gum : ℝ) : 
  initial_gum = 18.0 → additional_gum = 44.0 → total_gum = initial_gum + additional_gum → total_gum = 62.0 :=
by sorry

end robin_gum_count_l2871_287158


namespace division_problem_l2871_287175

theorem division_problem (total : ℕ) (a b c : ℕ) : 
  total = 770 →
  a = b + 40 →
  c = a + 30 →
  total = a + b + c →
  b = 220 := by
sorry

end division_problem_l2871_287175


namespace smallest_next_divisor_after_391_l2871_287141

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_next_divisor_after_391 (m : ℕ) 
  (h1 : is_even m) 
  (h2 : is_four_digit m) 
  (h3 : m % 391 = 0) : 
  ∃ d : ℕ, d > 391 ∧ m % d = 0 ∧ (∀ k : ℕ, 391 < k ∧ k < d → m % k ≠ 0) → d = 782 :=
sorry

end smallest_next_divisor_after_391_l2871_287141


namespace smallest_n_divisible_by_13_l2871_287136

theorem smallest_n_divisible_by_13 : 
  ∃ (n : ℕ), (13 ∣ (5^n + n^5)) ∧ (∀ m : ℕ, m < n → ¬(13 ∣ (5^m + m^5))) ∧ n = 12 := by
  sorry

end smallest_n_divisible_by_13_l2871_287136


namespace badminton_players_l2871_287163

theorem badminton_players (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 30)
  (h2 : tennis = 19)
  (h3 : both = 7)
  (h4 : neither = 2) :
  total - tennis - neither + both = 16 :=
by sorry

end badminton_players_l2871_287163


namespace odd_prime_square_root_l2871_287184

theorem odd_prime_square_root (p : ℕ) (hp : Prime p) (hp_odd : Odd p) :
  ∀ k : ℕ, (∃ m : ℕ, m > 0 ∧ m * m = k * k - p * k) ↔ k = ((p + 1) / 2) ^ 2 := by
  sorry

end odd_prime_square_root_l2871_287184


namespace max_cards_in_original_position_l2871_287105

/-- Represents a two-digit number card -/
structure Card :=
  (tens : Nat)
  (ones : Nat)
  (h1 : tens < 10)
  (h2 : ones < 10)

/-- The list of all cards from 00 to 99 in ascending order -/
def initial_arrangement : List Card := sorry

/-- Checks if two cards are adjacent according to the rearrangement rule -/
def are_adjacent (c1 c2 : Card) : Prop := sorry

/-- A valid rearrangement of cards -/
def valid_rearrangement (arrangement : List Card) : Prop :=
  arrangement.length = 100 ∧
  ∀ i, i < 99 → are_adjacent (arrangement.get ⟨i, sorry⟩) (arrangement.get ⟨i+1, sorry⟩)

/-- The number of cards in their original positions after rearrangement -/
def cards_in_original_position (arrangement : List Card) : Nat := sorry

/-- Theorem stating the maximum number of cards that can remain in their original positions -/
theorem max_cards_in_original_position :
  ∀ arrangement : List Card,
    valid_rearrangement arrangement →
    cards_in_original_position arrangement ≤ 50 :=
sorry

end max_cards_in_original_position_l2871_287105


namespace cow_milk_production_l2871_287119

/-- Given a number of cows and total weekly milk production, 
    calculate the daily milk production per cow. -/
def daily_milk_per_cow (num_cows : ℕ) (weekly_milk : ℕ) : ℚ :=
  (weekly_milk : ℚ) / 7 / num_cows

theorem cow_milk_production : daily_milk_per_cow 52 1820 = 5 := by
  sorry

end cow_milk_production_l2871_287119


namespace running_increase_calculation_l2871_287129

theorem running_increase_calculation 
  (initial_miles : ℕ) 
  (increase_percentage : ℚ) 
  (total_days : ℕ) 
  (days_per_week : ℕ) : 
  initial_miles = 100 →
  increase_percentage = 1/5 →
  total_days = 280 →
  days_per_week = 7 →
  (initial_miles * (1 + increase_percentage) - initial_miles) / (total_days / days_per_week) = 3 :=
by sorry

end running_increase_calculation_l2871_287129


namespace complement_of_A_in_U_l2871_287194

def U : Set ℕ := {1, 2, 3, 4}

def A : Set ℕ := {x : ℕ | x ^ 2 - 5 * x + 4 < 0}

theorem complement_of_A_in_U :
  U \ A = {1, 4} := by sorry

end complement_of_A_in_U_l2871_287194


namespace brothers_combined_age_l2871_287111

/-- Given the ages of Michael and his three brothers, prove their combined age is 53 years. -/
theorem brothers_combined_age :
  ∀ (michael oldest older younger : ℕ),
  -- The oldest brother is 1 year older than twice Michael's age when Michael was a year younger
  oldest = 2 * (michael - 1) + 1 →
  -- The younger brother is 5 years old
  younger = 5 →
  -- The younger brother's age is a third of the older brother's age
  older = 3 * younger →
  -- The other brother is half the age of the oldest brother
  older = oldest / 2 →
  -- The other brother is three years younger than Michael
  older = michael - 3 →
  -- The other brother is twice as old as their youngest brother
  older = 2 * younger →
  -- The combined age of all four brothers is 53
  michael + oldest + older + younger = 53 := by
sorry


end brothers_combined_age_l2871_287111


namespace right_triangle_moment_of_inertia_l2871_287199

/-- Moment of inertia of a right triangle relative to its hypotenuse -/
theorem right_triangle_moment_of_inertia (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let c := Real.sqrt (a^2 + b^2)
  let moment_of_inertia := (a^2 + b^2) / 18
  moment_of_inertia = (a^2 + b^2) / 18 :=
by sorry

end right_triangle_moment_of_inertia_l2871_287199


namespace intersection_right_angle_coordinates_l2871_287155

-- Define the line and parabola
def line (x y : ℝ) : Prop := x - 2*y - 1 = 0
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define points A and B as intersections
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ parabola A.1 A.2 ∧
  line B.1 B.2 ∧ parabola B.1 B.2 ∧
  A ≠ B

-- Define point C on the parabola
def point_on_parabola (C : ℝ × ℝ) : Prop := parabola C.1 C.2

-- Define right angle ACB
def right_angle (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem intersection_right_angle_coordinates :
  ∀ A B C : ℝ × ℝ,
  intersection_points A B →
  point_on_parabola C →
  right_angle A B C →
  (C = (1, -2) ∨ C = (9, -6)) :=
sorry

end intersection_right_angle_coordinates_l2871_287155


namespace product_equals_two_l2871_287192

theorem product_equals_two : 
  (∀ (a b c : ℝ), a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) →
  6 * 15 * 5 = 2 :=
by sorry

end product_equals_two_l2871_287192


namespace alison_small_tub_cost_l2871_287162

/-- The cost of small tubs given the number of large and small tubs, their total cost, and the cost of large tubs. -/
def small_tub_cost (num_large : ℕ) (num_small : ℕ) (total_cost : ℕ) (large_cost : ℕ) : ℕ :=
  (total_cost - num_large * large_cost) / num_small

/-- Theorem stating that the cost of each small tub is 5 dollars. -/
theorem alison_small_tub_cost :
  small_tub_cost 3 6 48 6 = 5 := by
sorry

#eval small_tub_cost 3 6 48 6

end alison_small_tub_cost_l2871_287162


namespace determine_set_B_l2871_287152

def U : Set Nat := {2, 4, 6, 8, 10}

theorem determine_set_B (A B : Set Nat) 
  (h1 : (A ∪ B)ᶜ = {8, 10})
  (h2 : A ∩ (U \ B) = {2}) :
  B = {4, 6} := by
  sorry

end determine_set_B_l2871_287152


namespace constant_expression_implies_a_equals_three_l2871_287183

theorem constant_expression_implies_a_equals_three (a : ℝ) :
  (∀ x : ℝ, x < 0 → ∃ c : ℝ, ∀ y : ℝ, y < 0 → 
    |y| + 2 * (y^2022)^(1/2022) + a * (y^2023)^(1/2023) = c) → 
  a = 3 := by
sorry

end constant_expression_implies_a_equals_three_l2871_287183


namespace arbitrary_triangle_angle_ratio_not_arbitrary_quadrilateral_angle_ratio_not_arbitrary_pentagon_angle_ratio_l2871_287174

-- Triangle
theorem arbitrary_triangle_angle_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ A / a = B / b ∧ B / b = C / c :=
sorry

-- Convex Quadrilateral
theorem not_arbitrary_quadrilateral_angle_ratio :
  ∃ (p q r s : ℝ), p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧
    ¬∃ (A B C D : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧
      A + B + C + D = 360 ∧
      A < B + C + D ∧ B < A + C + D ∧ C < A + B + D ∧ D < A + B + C ∧
      A / p = B / q ∧ B / q = C / r ∧ C / r = D / s :=
sorry

-- Convex Pentagon
theorem not_arbitrary_pentagon_angle_ratio :
  ∃ (u v w x y : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ x > 0 ∧ y > 0 ∧
    ¬∃ (A B C D E : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ E > 0 ∧
      A + B + C + D + E = 540 ∧
      2 * A < B + C + D + E ∧ 2 * B < A + C + D + E ∧ 2 * C < A + B + D + E ∧
      2 * D < A + B + C + E ∧ 2 * E < A + B + C + D ∧
      A / u = B / v ∧ B / v = C / w ∧ C / w = D / x ∧ D / x = E / y :=
sorry

end arbitrary_triangle_angle_ratio_not_arbitrary_quadrilateral_angle_ratio_not_arbitrary_pentagon_angle_ratio_l2871_287174


namespace distance_to_outside_point_gt_three_l2871_287103

/-- A circle with center O and radius 3 -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 3)

/-- A point outside the circle -/
structure OutsidePoint (c : Circle) :=
  (point : ℝ × ℝ)
  (h_outside : dist point c.center > c.radius)

/-- The theorem stating that the distance from the center to an outside point is greater than 3 -/
theorem distance_to_outside_point_gt_three (c : Circle) (p : OutsidePoint c) :
  dist p.point c.center > 3 := by
  sorry

end distance_to_outside_point_gt_three_l2871_287103


namespace propositions_proof_l2871_287176

theorem propositions_proof :
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ a + 1/a ≤ b + 1/b) ∧
  (∀ m n : ℝ, m > n ∧ n > 0 → (m + 1) / (n + 1) < m / n) ∧
  (∀ c a b : ℝ, c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) ∧
  (∀ a b : ℝ, a ≥ b ∧ b > -1 → a / (a + 1) ≥ b / (b + 1)) :=
by sorry

end propositions_proof_l2871_287176


namespace complex_equation_solution_l2871_287135

theorem complex_equation_solution (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  z = Complex.I + 1 := by
sorry

end complex_equation_solution_l2871_287135


namespace janet_piano_hours_l2871_287138

/-- Represents the number of hours per week Janet takes piano lessons -/
def piano_hours : ℕ := sorry

/-- The cost per hour of clarinet lessons -/
def clarinet_cost_per_hour : ℕ := 40

/-- The number of hours per week of clarinet lessons -/
def clarinet_hours_per_week : ℕ := 3

/-- The cost per hour of piano lessons -/
def piano_cost_per_hour : ℕ := 28

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The additional amount spent on piano lessons compared to clarinet lessons in a year -/
def additional_piano_cost : ℕ := 1040

theorem janet_piano_hours :
  piano_hours = 5 ∧
  clarinet_cost_per_hour * clarinet_hours_per_week * weeks_per_year + additional_piano_cost =
  piano_cost_per_hour * piano_hours * weeks_per_year :=
by sorry

end janet_piano_hours_l2871_287138


namespace x_minus_y_value_l2871_287126

theorem x_minus_y_value (x y : ℤ) 
  (sum_eq : x + y = 290) 
  (y_eq : y = 245) : 
  x - y = -200 := by
sorry

end x_minus_y_value_l2871_287126


namespace sqrt_equation_solutions_sqrt_equation_unique_solution_sqrt_equation_boundary_solution_sqrt_equation_no_solution_l2871_287160

/-- The equation √(x+1) - √(2x+1) = m has solutions as described -/
theorem sqrt_equation_solutions (m : ℝ) :
  (∃ x : ℝ, Real.sqrt (x + 1) - Real.sqrt (2 * x + 1) = m) ↔ 
  m ≤ Real.sqrt 2 / 2 :=
sorry

/-- The equation √(x+1) - √(2x+1) = m has exactly one solution when m < √2/2 -/
theorem sqrt_equation_unique_solution (m : ℝ) (h : m < Real.sqrt 2 / 2) :
  ∃! x : ℝ, Real.sqrt (x + 1) - Real.sqrt (2 * x + 1) = m :=
sorry

/-- The equation √(x+1) - √(2x+1) = m has exactly one solution when m = √2/2 -/
theorem sqrt_equation_boundary_solution :
  ∃! x : ℝ, Real.sqrt (x + 1) - Real.sqrt (2 * x + 1) = Real.sqrt 2 / 2 :=
sorry

/-- The equation √(x+1) - √(2x+1) = m has no solutions when m > √2/2 -/
theorem sqrt_equation_no_solution (m : ℝ) (h : m > Real.sqrt 2 / 2) :
  ¬∃ x : ℝ, Real.sqrt (x + 1) - Real.sqrt (2 * x + 1) = m :=
sorry

end sqrt_equation_solutions_sqrt_equation_unique_solution_sqrt_equation_boundary_solution_sqrt_equation_no_solution_l2871_287160


namespace math_test_problem_l2871_287165

theorem math_test_problem (total : ℕ) (word_problems : ℕ) (answered : ℕ) (blank : ℕ) :
  total = 45 →
  word_problems = 17 →
  answered = 38 →
  blank = 7 →
  total = answered + blank →
  total - word_problems - blank = 21 := by
  sorry

end math_test_problem_l2871_287165


namespace peaches_at_stand_l2871_287178

/-- The total number of peaches at Sally's stand after picking more -/
def total_peaches (initial : ℕ) (picked : ℕ) : ℕ :=
  initial + picked

/-- Theorem stating that the total number of peaches is 55 given the initial and picked amounts -/
theorem peaches_at_stand (initial : ℕ) (picked : ℕ) 
  (h1 : initial = 13) (h2 : picked = 42) : 
  total_peaches initial picked = 55 := by
  sorry

end peaches_at_stand_l2871_287178


namespace symmetric_point_wrt_origin_l2871_287187

/-- Given a point M with coordinates (3, -5), prove that its symmetric point
    with respect to the origin has coordinates (-3, 5). -/
theorem symmetric_point_wrt_origin :
  let M : ℝ × ℝ := (3, -5)
  let symmetric_point : ℝ × ℝ → ℝ × ℝ := λ (x, y) => (-x, -y)
  symmetric_point M = (-3, 5) := by
  sorry

end symmetric_point_wrt_origin_l2871_287187


namespace world_cup_stats_l2871_287145

def world_cup_data : List ℕ := [32, 31, 16, 16, 14, 12]

def median (l : List ℕ) : ℚ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem world_cup_stats :
  median world_cup_data = 16 ∧ mode world_cup_data = 16 := by sorry

end world_cup_stats_l2871_287145


namespace min_cards_for_two_of_each_suit_l2871_287198

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (jokers : ℕ)

/-- Defines the minimum number of cards to draw to ensure at least n cards of each suit -/
def min_cards_to_draw (d : Deck) (n : ℕ) : ℕ :=
  d.suits * (d.cards_per_suit - n + 1) + d.jokers + n - 1

/-- Theorem: The minimum number of cards to draw to ensure at least 2 cards of each suit is 43 -/
theorem min_cards_for_two_of_each_suit (d : Deck) 
  (h1 : d.total_cards = 54)
  (h2 : d.suits = 4)
  (h3 : d.cards_per_suit = 13)
  (h4 : d.jokers = 2) :
  min_cards_to_draw d 2 = 43 := by
  sorry

end min_cards_for_two_of_each_suit_l2871_287198


namespace paper_towel_case_rolls_l2871_287193

theorem paper_towel_case_rolls : ∀ (case_price individual_price : ℚ) (savings_percentage : ℚ),
  case_price = 9 →
  individual_price = 1 →
  savings_percentage = 25 →
  ∃ (n : ℕ), n = 12 ∧ case_price = (1 - savings_percentage / 100) * (n * individual_price) :=
by
  sorry

end paper_towel_case_rolls_l2871_287193


namespace penny_revenue_l2871_287148

/-- Calculates the total money earned from selling cheesecake pies -/
def cheesecake_revenue (price_per_slice : ℕ) (slices_per_pie : ℕ) (pies_sold : ℕ) : ℕ :=
  price_per_slice * slices_per_pie * pies_sold

/-- Proves that Penny makes $294 from selling 7 cheesecake pies -/
theorem penny_revenue : cheesecake_revenue 7 6 7 = 294 := by
  sorry

end penny_revenue_l2871_287148


namespace leet_puzzle_solution_l2871_287157

theorem leet_puzzle_solution :
  ∀ (L E T M : ℕ),
    L ≠ 0 →
    L < 10 ∧ E < 10 ∧ T < 10 ∧ M < 10 →
    1000 * L + 110 * E + T + 100 * L + 10 * M + T = 1000 * T + L →
    T = L + 1 →
    1000 * E + 100 * L + 10 * M + 0 = 1880 :=
by
  sorry

end leet_puzzle_solution_l2871_287157


namespace equal_fractions_imply_x_equals_four_l2871_287167

theorem equal_fractions_imply_x_equals_four (x : ℝ) :
  (x ≠ 0) → (x ≠ -2) → (6 / (x + 2) = 4 / x) → x = 4 := by
  sorry

end equal_fractions_imply_x_equals_four_l2871_287167


namespace value_of_x_l2871_287154

theorem value_of_x : ∃ x : ℝ, 3 * x + 15 = (1 / 3) * (7 * x + 45) → x = 0 := by
  sorry

end value_of_x_l2871_287154
