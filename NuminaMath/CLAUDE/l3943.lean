import Mathlib

namespace NUMINAMATH_CALUDE_fish_tagging_problem_l3943_394360

/-- The number of fish initially tagged in a pond -/
def initially_tagged (total_fish : ℕ) (catch_size : ℕ) (tagged_in_catch : ℕ) : ℕ :=
  (tagged_in_catch * total_fish) / catch_size

theorem fish_tagging_problem (total_fish : ℕ) (catch_size : ℕ) (tagged_in_catch : ℕ)
  (h1 : total_fish = 1500)
  (h2 : catch_size = 50)
  (h3 : tagged_in_catch = 2) :
  initially_tagged total_fish catch_size tagged_in_catch = 60 := by
  sorry

end NUMINAMATH_CALUDE_fish_tagging_problem_l3943_394360


namespace NUMINAMATH_CALUDE_combined_work_theorem_l3943_394377

/-- The time taken for three workers to complete a task together, given their individual completion times -/
def combined_completion_time (time_A time_B time_C : ℚ) : ℚ :=
  1 / (1 / time_A + 1 / time_B + 1 / time_C)

/-- Theorem: Given the individual completion times, the combined completion time is 72/13 days -/
theorem combined_work_theorem :
  combined_completion_time 12 18 24 = 72 / 13 := by
  sorry

end NUMINAMATH_CALUDE_combined_work_theorem_l3943_394377


namespace NUMINAMATH_CALUDE_tan_150_degrees_l3943_394397

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l3943_394397


namespace NUMINAMATH_CALUDE_stadium_entrance_exit_plans_stadium_plans_eq_35_l3943_394348

/-- The number of possible entrance and exit plans for a student at a school stadium. -/
theorem stadium_entrance_exit_plans : ℕ :=
  let south_gates : ℕ := 4
  let north_gates : ℕ := 3
  let west_gates : ℕ := 2
  let entrance_options : ℕ := south_gates + north_gates
  let exit_options : ℕ := west_gates + north_gates
  entrance_options * exit_options

/-- Proof that the number of possible entrance and exit plans is 35. -/
theorem stadium_plans_eq_35 : stadium_entrance_exit_plans = 35 := by
  sorry

end NUMINAMATH_CALUDE_stadium_entrance_exit_plans_stadium_plans_eq_35_l3943_394348


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3943_394324

theorem fraction_to_decimal (n : ℕ) (d : ℕ) (h : d = 2^3 * 5^7) :
  (n : ℚ) / d = 0.0006625 ↔ n = 53 :=
sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3943_394324


namespace NUMINAMATH_CALUDE_percentage_difference_l3943_394391

theorem percentage_difference (x y : ℝ) (h : y = x + 0.6667 * x) :
  x = y * (1 - 0.6667) := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3943_394391


namespace NUMINAMATH_CALUDE_drawings_on_last_page_l3943_394318

theorem drawings_on_last_page 
  (initial_notebooks : Nat) 
  (pages_per_notebook : Nat) 
  (initial_drawings_per_page : Nat) 
  (reorganized_drawings_per_page : Nat)
  (filled_notebooks : Nat)
  (filled_pages_last_notebook : Nat) :
  initial_notebooks = 12 →
  pages_per_notebook = 35 →
  initial_drawings_per_page = 4 →
  reorganized_drawings_per_page = 7 →
  filled_notebooks = 6 →
  filled_pages_last_notebook = 25 →
  (initial_notebooks * pages_per_notebook * initial_drawings_per_page) -
  (filled_notebooks * pages_per_notebook * reorganized_drawings_per_page) -
  (filled_pages_last_notebook * reorganized_drawings_per_page) = 5 := by
  sorry

end NUMINAMATH_CALUDE_drawings_on_last_page_l3943_394318


namespace NUMINAMATH_CALUDE_complex_quotient_real_l3943_394327

theorem complex_quotient_real (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (∃ (r : ℝ), z₁ / z₂ = r) → a = -3/2 := by sorry

end NUMINAMATH_CALUDE_complex_quotient_real_l3943_394327


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l3943_394326

/-- Proves that the initial number of bananas per child is 2 --/
theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) : 
  total_children = 780 →
  absent_children = 390 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ), 
    (total_children - absent_children) * (initial_bananas + extra_bananas) = total_children * initial_bananas ∧
    initial_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_per_child_l3943_394326


namespace NUMINAMATH_CALUDE_locus_equals_thales_circles_l3943_394334

/-- A triangle in a plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A point in the plane -/
def Point : Type := ℝ × ℝ

/-- The angle subtended by a side of the triangle from a point -/
noncomputable def subtended_angle (t : Triangle) (p : Point) (side : Fin 3) : ℝ :=
  sorry

/-- The sum of angles subtended by the three sides of the triangle from a point -/
noncomputable def sum_of_subtended_angles (t : Triangle) (p : Point) : ℝ :=
  (subtended_angle t p 0) + (subtended_angle t p 1) + (subtended_angle t p 2)

/-- The Thales' circle for a side of the triangle -/
def thales_circle (t : Triangle) (side : Fin 3) : Set Point :=
  sorry

/-- The set of points on all Thales' circles, excluding the triangle's vertices -/
def thales_circles_points (t : Triangle) : Set Point :=
  (thales_circle t 0 ∪ thales_circle t 1 ∪ thales_circle t 2) \ {t.A, t.B, t.C}

/-- The theorem stating the equivalence of the locus and the Thales' circles points -/
theorem locus_equals_thales_circles (t : Triangle) :
  {p : Point | sum_of_subtended_angles t p = π} = thales_circles_points t :=
  sorry

end NUMINAMATH_CALUDE_locus_equals_thales_circles_l3943_394334


namespace NUMINAMATH_CALUDE_unique_solution_linear_equation_l3943_394382

theorem unique_solution_linear_equation (a b : ℝ) (ha : a ≠ 0) :
  ∃! x : ℝ, a * x = b ∧ x = b / a := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_linear_equation_l3943_394382


namespace NUMINAMATH_CALUDE_solve_mushroom_problem_l3943_394381

def mushroom_pieces_problem (total_mushrooms : ℕ) 
                            (kenny_pieces : ℕ) 
                            (karla_pieces : ℕ) 
                            (remaining_pieces : ℕ) : Prop :=
  let total_pieces := kenny_pieces + karla_pieces + remaining_pieces
  total_pieces / total_mushrooms = 4

theorem solve_mushroom_problem : 
  mushroom_pieces_problem 22 38 42 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_mushroom_problem_l3943_394381


namespace NUMINAMATH_CALUDE_prob_AC_less_than_8_l3943_394305

/-- The probability that AC < 8 cm given the conditions of the problem -/
def probability_AC_less_than_8 : ℝ := 0.46

/-- The length of AB in cm -/
def AB : ℝ := 10

/-- The length of BC in cm -/
def BC : ℝ := 6

/-- The angle ABC in radians -/
def angle_ABC : Set ℝ := Set.Ioo 0 (Real.pi / 2)

/-- The theorem stating the probability of AC < 8 cm -/
theorem prob_AC_less_than_8 :
  ∃ (p : ℝ → Bool), p = λ β => ‖(0, -AB) - (BC * Real.cos β, BC * Real.sin β)‖ < 8 ∧
  ∫ β in angle_ABC, (if p β then 1 else 0) / Real.pi * 2 = probability_AC_less_than_8 :=
sorry

end NUMINAMATH_CALUDE_prob_AC_less_than_8_l3943_394305


namespace NUMINAMATH_CALUDE_equality_of_variables_l3943_394395

theorem equality_of_variables (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h₁ : (x₁^2 - x₃*x₅)*(x₂^2 - x₃*x₅) ≤ 0)
  (h₂ : (x₂^2 - x₄*x₁)*(x₃^2 - x₄*x₁) ≤ 0)
  (h₃ : (x₃^2 - x₅*x₂)*(x₄^2 - x₅*x₂) ≤ 0)
  (h₄ : (x₄^2 - x₁*x₃)*(x₅^2 - x₁*x₃) ≤ 0)
  (h₅ : (x₅^2 - x₂*x₄)*(x₁^2 - x₂*x₄) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ :=
by sorry

end NUMINAMATH_CALUDE_equality_of_variables_l3943_394395


namespace NUMINAMATH_CALUDE_base_number_proof_l3943_394339

theorem base_number_proof (x : ℝ) (n : ℕ) 
  (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^28) 
  (h2 : n = 27) : 
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l3943_394339


namespace NUMINAMATH_CALUDE_similar_triangle_coordinates_l3943_394343

-- Define the vertices of triangle ABC
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)

-- Define the similarity ratio
def ratio : ℝ := 2

-- Define the possible coordinates of C'
def C'_pos : ℝ × ℝ := (6, 4)
def C'_neg : ℝ × ℝ := (-6, -4)

-- Theorem statement
theorem similar_triangle_coordinates :
  ∀ (C' : ℝ × ℝ), 
    (∃ (k : ℝ), k = ratio ∧ C' = (k * C.1, k * C.2)) ∨
    (∃ (k : ℝ), k = -ratio ∧ C' = (k * C.1, k * C.2)) →
    C' = C'_pos ∨ C' = C'_neg :=
by sorry

end NUMINAMATH_CALUDE_similar_triangle_coordinates_l3943_394343


namespace NUMINAMATH_CALUDE_min_value_fraction_l3943_394353

theorem min_value_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (a + b) / (a * b * c) ≥ 16 / 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3943_394353


namespace NUMINAMATH_CALUDE_toucan_female_fraction_l3943_394376

theorem toucan_female_fraction (total_birds : ℝ) (h1 : total_birds > 0) :
  let parrot_fraction : ℝ := 3/5
  let toucan_fraction : ℝ := 1 - parrot_fraction
  let female_parrot_fraction : ℝ := 1/3
  let male_bird_fraction : ℝ := 1/2
  let female_toucan_count : ℝ := toucan_fraction * total_birds * female_toucan_fraction
  let female_parrot_count : ℝ := parrot_fraction * total_birds * female_parrot_fraction
  let total_female_count : ℝ := female_toucan_count + female_parrot_count
  female_toucan_count + female_parrot_count = male_bird_fraction * total_birds →
  female_toucan_fraction = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_toucan_female_fraction_l3943_394376


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l3943_394309

def adult_ticket_price : ℕ := 5
def child_ticket_price : ℕ := 2
def total_tickets : ℕ := 85
def total_amount : ℕ := 275

theorem adult_tickets_sold (a c : ℕ) : 
  a + c = total_tickets → 
  a * adult_ticket_price + c * child_ticket_price = total_amount → 
  a = 35 := by
sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l3943_394309


namespace NUMINAMATH_CALUDE_oldest_harper_child_age_l3943_394365

/-- The age of the oldest Harper child given the ages of the other three and the average age of all four. -/
theorem oldest_harper_child_age 
  (average_age : ℝ) 
  (younger_child1 : ℕ) 
  (younger_child2 : ℕ) 
  (younger_child3 : ℕ) 
  (h1 : average_age = 9) 
  (h2 : younger_child1 = 6) 
  (h3 : younger_child2 = 8) 
  (h4 : younger_child3 = 10) : 
  ∃ (oldest_child : ℕ), 
    (younger_child1 + younger_child2 + younger_child3 + oldest_child) / 4 = average_age ∧ 
    oldest_child = 12 := by
  sorry

end NUMINAMATH_CALUDE_oldest_harper_child_age_l3943_394365


namespace NUMINAMATH_CALUDE_expression_zero_iff_x_eq_three_l3943_394356

theorem expression_zero_iff_x_eq_three (x : ℝ) :
  (4 * x - 8 ≠ 0) →
  ((x^2 - 6*x + 9) / (4*x - 8) = 0 ↔ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_expression_zero_iff_x_eq_three_l3943_394356


namespace NUMINAMATH_CALUDE_similar_triangles_height_cycle_height_problem_l3943_394338

theorem similar_triangles_height (h₁ : ℝ) (b₁ : ℝ) (b₂ : ℝ) (h₁_pos : h₁ > 0) (b₁_pos : b₁ > 0) (b₂_pos : b₂ > 0) :
  h₁ / b₁ = (h₁ * b₂ / b₁) / b₂ :=
by sorry

theorem cycle_height_problem (h₁ : ℝ) (b₁ : ℝ) (b₂ : ℝ) 
  (h₁_val : h₁ = 2.5) (b₁_val : b₁ = 5) (b₂_val : b₂ = 4) :
  h₁ * b₂ / b₁ = 2 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_cycle_height_problem_l3943_394338


namespace NUMINAMATH_CALUDE_nth_equation_proof_l3943_394398

theorem nth_equation_proof (n : ℕ) : (n + 1) * (n^2 - n + 1) - 1 = n^3 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_proof_l3943_394398


namespace NUMINAMATH_CALUDE_chord_line_equation_l3943_394385

/-- Given an ellipse and a point as the midpoint of a chord, find the equation of the line containing the chord -/
theorem chord_line_equation (x y : ℝ) :
  (x^2 / 36 + y^2 / 9 = 1) →  -- Ellipse equation
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 / 36 + y₁^2 / 9 = 1) ∧  -- Point (x₁, y₁) is on the ellipse
    (x₂^2 / 36 + y₂^2 / 9 = 1) ∧  -- Point (x₂, y₂) is on the ellipse
    ((x₁ + x₂) / 2 = 1) ∧  -- Midpoint x-coordinate is 1
    ((y₁ + y₂) / 2 = 1) →  -- Midpoint y-coordinate is 1
  ∃ (m : ℝ), m = -1/4 ∧ y - 1 = m * (x - 1) :=  -- Line equation
by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l3943_394385


namespace NUMINAMATH_CALUDE_solution_set_implies_m_range_l3943_394342

theorem solution_set_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - m| > 4) → (m > 3 ∨ m < -5) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_range_l3943_394342


namespace NUMINAMATH_CALUDE_radish_patch_size_proof_l3943_394373

/-- The size of a pea patch in square feet -/
def pea_patch_size : ℝ := 30

/-- The size of a radish patch in square feet -/
def radish_patch_size : ℝ := 15

theorem radish_patch_size_proof :
  (pea_patch_size = 2 * radish_patch_size) ∧
  (pea_patch_size / 6 = 5) →
  radish_patch_size = 15 := by
  sorry

end NUMINAMATH_CALUDE_radish_patch_size_proof_l3943_394373


namespace NUMINAMATH_CALUDE_vector_minimization_and_angle_l3943_394329

/-- Given vectors OP, OA, OB, and a point C on line OP, prove that OC minimizes CA · CB and calculate cos ∠ACB -/
theorem vector_minimization_and_angle (O P A B C : ℝ × ℝ) : 
  O = (0, 0) →
  P = (2, 1) →
  A = (1, 7) →
  B = (5, 1) →
  (∃ t : ℝ, C = (t * 2, t * 1)) →
  (∀ D : ℝ × ℝ, (∃ s : ℝ, D = (s * 2, s * 1)) → 
    (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) ≤ 
    (A.1 - D.1) * (B.1 - D.1) + (A.2 - D.2) * (B.2 - D.2)) →
  C = (4, 2) ∧ 
  (((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2)) / 
   (((A.1 - C.1)^2 + (A.2 - C.2)^2) * ((B.1 - C.1)^2 + (B.2 - C.2)^2))^(1/2) = -4 * 17^(1/2) / 17) :=
by sorry


end NUMINAMATH_CALUDE_vector_minimization_and_angle_l3943_394329


namespace NUMINAMATH_CALUDE_john_sleep_week_total_l3943_394346

/-- The amount of sleep John got during a week with varying sleep patterns. -/
def johnSleepWeek (recommendedSleep : ℝ) : ℝ :=
  let mondayTuesday := 2 * 3
  let wednesday := 0.8 * recommendedSleep
  let thursdayFriday := 2 * (0.5 * recommendedSleep)
  let saturday := 0.7 * recommendedSleep + 2
  let sunday := 0.4 * recommendedSleep
  mondayTuesday + wednesday + thursdayFriday + saturday + sunday

/-- Theorem stating that John's total sleep for the week is 31.2 hours. -/
theorem john_sleep_week_total : johnSleepWeek 8 = 31.2 := by
  sorry

#eval johnSleepWeek 8

end NUMINAMATH_CALUDE_john_sleep_week_total_l3943_394346


namespace NUMINAMATH_CALUDE_eccentricity_of_parametric_ellipse_l3943_394399

/-- Eccentricity of an ellipse defined by parametric equations -/
theorem eccentricity_of_parametric_ellipse :
  let x : ℝ → ℝ := λ φ ↦ 3 * Real.cos φ
  let y : ℝ → ℝ := λ φ ↦ Real.sqrt 5 * Real.sin φ
  let a : ℝ := 3
  let b : ℝ := Real.sqrt 5
  let c : ℝ := Real.sqrt (a^2 - b^2)
  c / a = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_eccentricity_of_parametric_ellipse_l3943_394399


namespace NUMINAMATH_CALUDE_anna_scores_proof_l3943_394375

def anna_scores : List ℕ := [94, 87, 86, 78, 71, 58]

theorem anna_scores_proof :
  -- The list has 6 elements
  anna_scores.length = 6 ∧
  -- All elements are less than 95
  (∀ x ∈ anna_scores, x < 95) ∧
  -- All elements are different
  anna_scores.Nodup ∧
  -- The list is sorted in descending order
  anna_scores.Sorted (· ≥ ·) ∧
  -- The first three scores are 86, 78, and 71
  [86, 78, 71].Sublist anna_scores ∧
  -- The mean of all scores is 79
  anna_scores.sum / anna_scores.length = 79 := by
sorry

end NUMINAMATH_CALUDE_anna_scores_proof_l3943_394375


namespace NUMINAMATH_CALUDE_johns_income_l3943_394306

theorem johns_income (john_tax_rate ingrid_tax_rate combined_tax_rate : ℚ)
  (ingrid_income : ℕ) :
  john_tax_rate = 30 / 100 →
  ingrid_tax_rate = 40 / 100 →
  combined_tax_rate = 35625 / 100000 →
  ingrid_income = 72000 →
  ∃ john_income : ℕ,
    john_income = 56000 ∧
    (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) /
      (john_income + ingrid_income) = combined_tax_rate :=
by sorry

end NUMINAMATH_CALUDE_johns_income_l3943_394306


namespace NUMINAMATH_CALUDE_triangle_existence_implies_m_greater_than_six_l3943_394386

/-- The function f(x) = x^3 - 3x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

/-- Theorem: If there exists a triangle with side lengths f(a), f(b), and f(c) for a, b, c in [0,2], then m > 6 -/
theorem triangle_existence_implies_m_greater_than_six (m : ℝ) :
  (∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 
    f m a + f m b > f m c ∧ 
    f m b + f m c > f m a ∧ 
    f m c + f m a > f m b) →
  m > 6 := by
sorry


end NUMINAMATH_CALUDE_triangle_existence_implies_m_greater_than_six_l3943_394386


namespace NUMINAMATH_CALUDE_incircle_radius_altitude_ratio_l3943_394380

/-- An isosceles right triangle with inscribed circle -/
structure IsoscelesRightTriangle where
  -- Side length of the equal sides
  side : ℝ
  -- Radius of the inscribed circle
  incircle_radius : ℝ
  -- Altitude to the hypotenuse
  altitude : ℝ
  -- The triangle is isosceles and right-angled
  is_isosceles : side = altitude * Real.sqrt 2
  -- Relationship between incircle radius and altitude
  radius_altitude_relation : incircle_radius = altitude * (Real.sqrt 2 - 1)

/-- The ratio of the inscribed circle radius to the altitude in an isosceles right triangle is √2 - 1 -/
theorem incircle_radius_altitude_ratio (t : IsoscelesRightTriangle) :
  t.incircle_radius / t.altitude = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_incircle_radius_altitude_ratio_l3943_394380


namespace NUMINAMATH_CALUDE_initial_speed_calculation_l3943_394369

/-- Proves that the initial speed satisfies the given equation under the problem conditions -/
theorem initial_speed_calculation (D T : ℝ) (hD : D > 0) (hT : T > 0) 
  (h_time_constraint : T/3 + (D/3) / 25 = T) : ∃ S : ℝ, S = 2*D/T ∧ S = 100 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_calculation_l3943_394369


namespace NUMINAMATH_CALUDE_two_distinct_roots_condition_l3943_394378

theorem two_distinct_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + k = 0 ∧ x₂^2 - 2*x₂ + k = 0) → k < 1 :=
by sorry

end NUMINAMATH_CALUDE_two_distinct_roots_condition_l3943_394378


namespace NUMINAMATH_CALUDE_distance_XY_is_1000_l3943_394364

/-- The distance between two points X and Y --/
def distance : ℝ := sorry

/-- The time taken to travel from X to Y --/
def time_XY : ℝ := 10

/-- The time taken to travel from Y to X --/
def time_YX : ℝ := 4

/-- The average speed for the entire journey --/
def avg_speed : ℝ := 142.85714285714286

/-- Theorem stating that the distance between X and Y is 1000 miles --/
theorem distance_XY_is_1000 : distance = 1000 := by sorry

end NUMINAMATH_CALUDE_distance_XY_is_1000_l3943_394364


namespace NUMINAMATH_CALUDE_determine_phi_l3943_394347

-- Define the functions and constants
noncomputable def ω : ℝ := 2
noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (ω * x + φ)
noncomputable def g (x : ℝ) := Real.cos (ω * x)

-- State the theorem
theorem determine_phi :
  (ω > 0) →
  (∀ φ, |φ| < π / 2 →
    (∀ x, f x φ = f (x + π) φ) →
    (∀ x, f (x - 2*π/3) φ = g x)) →
  ∃ φ, φ = -π / 6 :=
by sorry

end NUMINAMATH_CALUDE_determine_phi_l3943_394347


namespace NUMINAMATH_CALUDE_prob_same_color_is_zero_l3943_394396

/-- Represents the number of balls of each color in the bag -/
structure BallCounts where
  green : Nat
  white : Nat
  blue : Nat
  red : Nat

/-- Calculates the total number of balls in the bag -/
def totalBalls (counts : BallCounts) : Nat :=
  counts.green + counts.white + counts.blue + counts.red

/-- Represents the number of balls to be drawn -/
def ballsToDraw : Nat := 5

/-- Calculates the probability of drawing all balls of the same color -/
def probSameColor (counts : BallCounts) : ℚ :=
  if counts.green ≥ ballsToDraw ∨ counts.white ≥ ballsToDraw ∨ 
     counts.blue ≥ ballsToDraw ∨ counts.red ≥ ballsToDraw
  then 1 / (totalBalls counts).choose ballsToDraw
  else 0

/-- Theorem: The probability of drawing 5 balls of the same color is 0 -/
theorem prob_same_color_is_zero : 
  probSameColor { green := 10, white := 9, blue := 7, red := 4 } = 0 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_zero_l3943_394396


namespace NUMINAMATH_CALUDE_suna_travel_distance_l3943_394337

theorem suna_travel_distance (D : ℝ) 
  (h1 : (1 - 7/15) * (1 - 5/8) * (1 - 2/3) * D = 2.6) : D = 39 := by
  sorry

end NUMINAMATH_CALUDE_suna_travel_distance_l3943_394337


namespace NUMINAMATH_CALUDE_two_distinct_prime_factors_iff_n_zero_l3943_394319

def base_6_to_decimal (base_6_num : List Nat) : Nat :=
  base_6_num.enum.foldr (λ (i, digit) acc => acc + digit * (6 ^ i)) 0

def append_fives (n : Nat) : List Nat :=
  [1, 2, 0, 0] ++ List.replicate (10 * n + 2) 5

def result_number (n : Nat) : Nat :=
  base_6_to_decimal (append_fives n)

def has_exactly_two_distinct_prime_factors (x : Nat) : Prop :=
  ∃ p q : Nat, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  ∃ a b : Nat, x = p^a * q^b ∧ 
  ∀ r : Nat, Nat.Prime r → r ∣ x → (r = p ∨ r = q)

theorem two_distinct_prime_factors_iff_n_zero (n : Nat) :
  has_exactly_two_distinct_prime_factors (result_number n) ↔ n = 0 := by
  sorry

#check two_distinct_prime_factors_iff_n_zero

end NUMINAMATH_CALUDE_two_distinct_prime_factors_iff_n_zero_l3943_394319


namespace NUMINAMATH_CALUDE_max_triangle_area_l3943_394384

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (1, 0)

-- Define a chord passing through the right focus
def chord_through_right_focus (m : ℝ) (y : ℝ) : ℝ := m * y + 1

-- Define the area of triangle PF₁Q
def triangle_area (y₁ y₂ : ℝ) : ℝ := |y₁ - y₂|

-- Theorem statement
theorem max_triangle_area :
  ∃ (max_area : ℝ), max_area = 3 ∧
  ∀ (m : ℝ) (y₁ y₂ : ℝ),
    ellipse (chord_through_right_focus m y₁) y₁ →
    ellipse (chord_through_right_focus m y₂) y₂ →
    triangle_area y₁ y₂ ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l3943_394384


namespace NUMINAMATH_CALUDE_lcm_48_180_l3943_394379

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by sorry

end NUMINAMATH_CALUDE_lcm_48_180_l3943_394379


namespace NUMINAMATH_CALUDE_prime_cube_sum_of_squares_l3943_394330

theorem prime_cube_sum_of_squares (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p^3 = p^2 + q^2 + r^2 → 
  p = 3 ∧ q = 3 ∧ r = 3 := by
sorry

end NUMINAMATH_CALUDE_prime_cube_sum_of_squares_l3943_394330


namespace NUMINAMATH_CALUDE_book_to_bookmark_ratio_l3943_394325

def books : ℕ := 72
def bookmarks : ℕ := 16

theorem book_to_bookmark_ratio : 
  (books / (Nat.gcd books bookmarks)) / (bookmarks / (Nat.gcd books bookmarks)) = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_book_to_bookmark_ratio_l3943_394325


namespace NUMINAMATH_CALUDE_water_filling_canal_is_certain_l3943_394393

-- Define the type for events
inductive Event : Type
  | WaitingForRabbit : Event
  | ScoopingMoon : Event
  | WaterFillingCanal : Event
  | SeekingFishByTree : Event

-- Define what it means for an event to be certain
def isCertainEvent (e : Event) : Prop :=
  match e with
  | Event.WaterFillingCanal => true
  | _ => false

-- State the theorem
theorem water_filling_canal_is_certain :
  isCertainEvent Event.WaterFillingCanal :=
sorry

end NUMINAMATH_CALUDE_water_filling_canal_is_certain_l3943_394393


namespace NUMINAMATH_CALUDE_triangle_problem_l3943_394340

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : Real.tan (π/4 - abc.C) = Real.sqrt 3 - 2)
  (h2 : abc.c = Real.sqrt 7)
  (h3 : abc.a + abc.b = 5) :
  abc.C = π/3 ∧ 
  (1/2 * abc.a * abc.b * Real.sin abc.C = 3 * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3943_394340


namespace NUMINAMATH_CALUDE_value_of_a_l3943_394332

theorem value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 16 - 6 * a) : a = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3943_394332


namespace NUMINAMATH_CALUDE_max_k_value_l3943_394317

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 4 = k^2 * (x^2 / y^2 + y^2 / x^2 + 2) + 2 * k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_k_value_l3943_394317


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3943_394336

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3943_394336


namespace NUMINAMATH_CALUDE_factorial_800_trailing_zeros_l3943_394372

/-- Counts the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- The number of trailing zeros in 800! is 199 -/
theorem factorial_800_trailing_zeros : trailingZeros 800 = 199 := by
  sorry

end NUMINAMATH_CALUDE_factorial_800_trailing_zeros_l3943_394372


namespace NUMINAMATH_CALUDE_difference_of_integers_l3943_394349

/-- Given positive integers a and b satisfying 2a - 9b + 18ab = 2018, prove that b - a = 223 -/
theorem difference_of_integers (a b : ℕ+) (h : 2 * a - 9 * b + 18 * a * b = 2018) : 
  b - a = 223 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_integers_l3943_394349


namespace NUMINAMATH_CALUDE_total_eggs_january_l3943_394303

/-- Represents a hen with a specific egg-laying frequency -/
structure Hen where
  frequency : ℕ  -- Number of days between each egg

/-- Calculates the number of eggs laid by a hen in a given number of days -/
def eggsLaid (h : Hen) (days : ℕ) : ℕ :=
  (days + h.frequency - 1) / h.frequency

/-- The three hens owned by Xiao Ming's family -/
def hens : List Hen := [
  { frequency := 1 },  -- First hen lays an egg every day
  { frequency := 2 },  -- Second hen lays an egg every two days
  { frequency := 3 }   -- Third hen lays an egg every three days
]

/-- The total number of eggs laid by all hens in January -/
def totalEggsInJanuary : ℕ :=
  (hens.map (eggsLaid · 31)).sum

theorem total_eggs_january : totalEggsInJanuary = 56 := by
  sorry

#eval totalEggsInJanuary  -- This should output 56

end NUMINAMATH_CALUDE_total_eggs_january_l3943_394303


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_8_pow_351_l3943_394367

theorem sum_of_last_two_digits_of_8_pow_351 :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ (8^351) % 100 = 10 * a + b ∧ a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_8_pow_351_l3943_394367


namespace NUMINAMATH_CALUDE_train_length_calculation_l3943_394345

-- Define the walking speed in meters per second
def walking_speed : ℝ := 1

-- Define the time taken for the train to pass Xiao Ming
def time_ming : ℝ := 22

-- Define the time taken for the train to pass Xiao Hong
def time_hong : ℝ := 24

-- Define the train's speed (to be solved)
def train_speed : ℝ := 23

-- Define the train's length (to be proved)
def train_length : ℝ := 528

-- Theorem statement
theorem train_length_calculation :
  train_length = time_ming * (train_speed + walking_speed) ∧
  train_length = time_hong * (train_speed - walking_speed) := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l3943_394345


namespace NUMINAMATH_CALUDE_not_prime_m_plus_n_minus_one_l3943_394368

theorem not_prime_m_plus_n_minus_one (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2)
  (h3 : (m + n - 1) ∣ (m^2 + n^2 - 1)) : ¬ Nat.Prime (m + n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_m_plus_n_minus_one_l3943_394368


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3943_394361

theorem sufficient_but_not_necessary (x : ℝ) :
  (((1 : ℝ) / x < 1) → (x > 1)) ∧ ¬((x > 1) → ((1 : ℝ) / x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3943_394361


namespace NUMINAMATH_CALUDE_line_equation_problem_l3943_394358

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 3)^2 + (y - 4)^2 = 16

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the line of symmetry
def symmetry_line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define a line
def line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Define the tangency condition
def is_tangent (m b : ℝ) : Prop := ∃ x y : ℝ, unit_circle x y ∧ line m b x y

-- State the theorem
theorem line_equation_problem (M N : ℝ × ℝ) (k : ℝ) :
  (∃ k, symmetry_line k M.1 M.2 ∧ symmetry_line k N.1 N.2) →
  circle_C M.1 M.2 →
  circle_C N.1 N.2 →
  (∃ m b, is_tangent m b ∧ line m b M.1 M.2 ∧ line m b N.1 N.2) →
  ∃ m b, m = 1 ∧ b = 2 ∧ ∀ x y, line m b x y ↔ y = x + 2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_problem_l3943_394358


namespace NUMINAMATH_CALUDE_small_s_conference_teams_l3943_394351

-- Define the number of games in the tournament
def num_games : ℕ := 36

-- Define the function to calculate the number of games for n teams
def games_for_teams (n : ℕ) : ℕ := n * (n - 1) / 2

-- Theorem statement
theorem small_s_conference_teams :
  ∃ (n : ℕ), n > 0 ∧ games_for_teams n = num_games ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_small_s_conference_teams_l3943_394351


namespace NUMINAMATH_CALUDE_square_area_l3943_394370

-- Define the square WXYZ
structure Square (W X Y Z : ℝ × ℝ) : Prop where
  is_square : true  -- We assume WXYZ is a square

-- Define the points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define the properties of the square and points
def square_properties (W X Y Z : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  Square W X Y Z ∧
  (∃ (t : ℝ), t > 0 ∧ t < 1 ∧ P = (1 - t) • X + t • Y) ∧  -- P is on XY
  (∃ (s : ℝ), s > 0 ∧ s < 1 ∧ Q = (1 - s) • W + s • Z) ∧  -- Q is on WZ
  (Y.1 - P.1)^2 + (Y.2 - P.2)^2 = 16 ∧  -- YP = 4
  (Q.1 - Z.1)^2 + (Q.2 - Z.2)^2 = 9    -- QZ = 3

-- Define the angle trisection property
def angle_trisected (W P Q : ℝ × ℝ) : Prop :=
  ∃ (θ : ℝ), θ > 0 ∧ 
    (P.2 - W.2) / (P.1 - W.1) = Real.tan θ ∧
    (Q.2 - W.2) / (Q.1 - W.1) = Real.tan (2 * θ)

-- Theorem statement
theorem square_area (W X Y Z : ℝ × ℝ) (P Q : ℝ × ℝ) :
  square_properties W X Y Z P Q →
  angle_trisected W P Q →
  (Y.1 - W.1)^2 + (Y.2 - W.2)^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l3943_394370


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l3943_394307

-- Define the polynomials
def f (x : ℝ) : ℝ := -2 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 4 * x^2 + 6 * x + 3
def j (x : ℝ) : ℝ := 3 * x^2 - x + 2

-- State the theorem
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x + j x = -x^2 + 11 * x - 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l3943_394307


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3943_394310

theorem quadratic_root_in_unit_interval 
  (a b c : ℝ) 
  (h : 2*a + 3*b + 6*c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3943_394310


namespace NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l3943_394312

theorem no_integer_satisfies_conditions : ¬∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), n = 16 * k) ∧ 
  (23 < Real.sqrt n) ∧ 
  (Real.sqrt n < 23.2) := by
sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l3943_394312


namespace NUMINAMATH_CALUDE_max_candy_types_l3943_394341

/-- A type representing a student --/
def Student : Type := ℕ

/-- A type representing a candy type --/
def CandyType : Type := ℕ

/-- The total number of students --/
def total_students : ℕ := 1000

/-- A function representing whether a student received a certain candy type --/
def received (s : Student) (c : CandyType) : Prop := sorry

/-- The condition that for any 11 types of candy, each student received at least one of those types --/
def condition_eleven (N : ℕ) : Prop :=
  ∀ (s : Student) (cs : Finset CandyType),
    cs.card = 11 → (∃ c ∈ cs, received s c)

/-- The condition that for any two types of candy, there exists a student who received exactly one of those types --/
def condition_two (N : ℕ) : Prop :=
  ∀ (c1 c2 : CandyType),
    c1 ≠ c2 → (∃ s : Student, (received s c1 ∧ ¬received s c2) ∨ (¬received s c1 ∧ received s c2))

/-- The main theorem stating that the maximum possible value of N is 5501 --/
theorem max_candy_types :
  ∃ N : ℕ,
    (∀ N' : ℕ, condition_eleven N' ∧ condition_two N' → N' ≤ N) ∧
    condition_eleven N ∧ condition_two N ∧
    N = 5501 := by sorry

end NUMINAMATH_CALUDE_max_candy_types_l3943_394341


namespace NUMINAMATH_CALUDE_milena_grandfather_age_difference_l3943_394355

/-- Calculates the age difference between a child and their grandfather given the child's age,
    the ratio of grandmother's age to child's age, and the age difference between grandparents. -/
def age_difference_child_grandfather (child_age : ℕ) (grandmother_ratio : ℕ) (grandparents_diff : ℕ) : ℕ :=
  (child_age * grandmother_ratio + grandparents_diff) - child_age

/-- The age difference between Milena and her grandfather is 58 years. -/
theorem milena_grandfather_age_difference :
  age_difference_child_grandfather 7 9 2 = 58 := by
  sorry

end NUMINAMATH_CALUDE_milena_grandfather_age_difference_l3943_394355


namespace NUMINAMATH_CALUDE_marbles_problem_l3943_394344

/-- Represents the number of marbles left in a box after removing some marbles. -/
def marblesLeft (total white : ℕ) : ℕ :=
  let red := (total - white) / 2
  let blue := (total - white) / 2
  let removed := 2 * (white - blue)
  total - removed

/-- Theorem stating that given the conditions of the problem, 40 marbles are left. -/
theorem marbles_problem : marblesLeft 50 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_marbles_problem_l3943_394344


namespace NUMINAMATH_CALUDE_gina_netflix_minutes_l3943_394352

/-- The number of shows Gina's sister watches per week -/
def sister_shows : ℕ := 24

/-- The length of each show in minutes -/
def show_length : ℕ := 50

/-- The ratio of shows Gina chooses compared to her sister -/
def gina_ratio : ℕ := 3

/-- The total number of shows watched by both Gina and her sister -/
def total_shows : ℕ := sister_shows * (gina_ratio + 1)

/-- The number of shows Gina chooses -/
def gina_shows : ℕ := total_shows * gina_ratio / (gina_ratio + 1)

theorem gina_netflix_minutes : gina_shows * show_length = 900 :=
by sorry

end NUMINAMATH_CALUDE_gina_netflix_minutes_l3943_394352


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3943_394388

def a : Fin 2 → ℝ := ![1, 2]
def b (y : ℝ) : Fin 2 → ℝ := ![-2, y]

theorem parallel_vectors_magnitude (y : ℝ) 
  (h : a 0 * (b y 1) = a 1 * (b y 0)) : 
  Real.sqrt ((3 * a 0 + b y 0)^2 + (3 * a 1 + b y 1)^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3943_394388


namespace NUMINAMATH_CALUDE_largest_square_divisor_l3943_394323

theorem largest_square_divisor : 
  ∃ (x : ℕ), x = 12 ∧ 
  x^2 ∣ (24 * 35 * 46 * 57) ∧ 
  ∀ (y : ℕ), y > x → ¬(y^2 ∣ (24 * 35 * 46 * 57)) := by
  sorry

end NUMINAMATH_CALUDE_largest_square_divisor_l3943_394323


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l3943_394328

theorem quadratic_inequality_roots (c : ℝ) : 
  (∀ x, -x^2 + c*x + 3 < 0 ↔ x < -3 ∨ x > 2) → c = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l3943_394328


namespace NUMINAMATH_CALUDE_smallest_binary_multiple_of_48_squared_l3943_394374

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

def target_number : ℕ := 11111111100000000

theorem smallest_binary_multiple_of_48_squared :
  (target_number % (48^2) = 0) ∧
  is_binary_number target_number ∧
  ∀ m : ℕ, m < target_number →
    ¬(m % (48^2) = 0 ∧ is_binary_number m) :=
by sorry

#eval target_number % (48^2)  -- Should output 0
#eval target_number.digits 10  -- Should output [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

end NUMINAMATH_CALUDE_smallest_binary_multiple_of_48_squared_l3943_394374


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l3943_394362

theorem arithmetic_sequence_sum_divisibility :
  ∀ (x c : ℕ+),
  ∃ (d : ℕ+),
  (d = 15) ∧
  (d ∣ (15 * x + 105 * c)) ∧
  (∀ (k : ℕ+), k > d → ¬(∀ (y z : ℕ+), k ∣ (15 * y + 105 * z))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l3943_394362


namespace NUMINAMATH_CALUDE_right_triangle_side_ratio_range_l3943_394316

theorem right_triangle_side_ratio_range (a b c x : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Positive side lengths
  a^2 + b^2 = c^2 →        -- Pythagorean theorem
  a + b = c * x →          -- Given condition
  x ∈ Set.Ioo 1 (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_ratio_range_l3943_394316


namespace NUMINAMATH_CALUDE_bird_count_l3943_394394

theorem bird_count (cardinals bluebirds swallows : ℕ) : 
  cardinals = 3 * bluebirds ∧ 
  swallows = bluebirds / 2 ∧ 
  swallows = 2 → 
  cardinals + bluebirds + swallows = 18 := by
sorry

end NUMINAMATH_CALUDE_bird_count_l3943_394394


namespace NUMINAMATH_CALUDE_mushroom_collection_l3943_394331

theorem mushroom_collection (N : ℕ) : 
  (100 ≤ N ∧ N < 1000) →  -- N is a three-digit number
  (N / 100 + (N / 10) % 10 + N % 10 = 14) →  -- sum of digits is 14
  (N % 50 = 0) →  -- divisible by 50
  (N % 25 = 0) →  -- 8% of N is an integer (since 8% = 2/25)
  (N % 50 = 0) →  -- 14% of N is an integer (since 14% = 7/50)
  N = 950 := by
sorry

end NUMINAMATH_CALUDE_mushroom_collection_l3943_394331


namespace NUMINAMATH_CALUDE_station_length_l3943_394363

/-- The length of a station given a train passing through it -/
theorem station_length (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 250 →
  train_speed_kmh = 36 →
  passing_time = 45 →
  (train_speed_kmh * 1000 / 3600) * passing_time - train_length = 200 :=
by sorry

end NUMINAMATH_CALUDE_station_length_l3943_394363


namespace NUMINAMATH_CALUDE_mr_martin_purchase_cost_l3943_394313

/-- The cost of Mrs. Martin's purchase -/
def mrs_martin_cost : ℝ := 12.75

/-- The number of coffee cups Mrs. Martin bought -/
def mrs_martin_coffee : ℕ := 3

/-- The number of bagels Mrs. Martin bought -/
def mrs_martin_bagels : ℕ := 2

/-- The cost of one bagel -/
def bagel_cost : ℝ := 1.5

/-- The number of coffee cups Mr. Martin bought -/
def mr_martin_coffee : ℕ := 2

/-- The number of bagels Mr. Martin bought -/
def mr_martin_bagels : ℕ := 5

/-- Theorem stating that Mr. Martin's purchase costs $14.00 -/
theorem mr_martin_purchase_cost : 
  ∃ (coffee_cost : ℝ), 
    mrs_martin_cost = mrs_martin_coffee * coffee_cost + mrs_martin_bagels * bagel_cost ∧
    mr_martin_coffee * coffee_cost + mr_martin_bagels * bagel_cost = 14 :=
by sorry

end NUMINAMATH_CALUDE_mr_martin_purchase_cost_l3943_394313


namespace NUMINAMATH_CALUDE_quadratic_real_equal_roots_l3943_394320

theorem quadratic_real_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - (m - 1) * x + 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - (m - 1) * y + 2 * y + 12 = 0 → y = x) ↔ 
  (m = -10 ∨ m = 14) := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_equal_roots_l3943_394320


namespace NUMINAMATH_CALUDE_count_bijections_on_three_element_set_l3943_394359

def S : Finset ℕ := {1, 2, 3}

theorem count_bijections_on_three_element_set :
  Fintype.card { f : S → S | Function.Bijective f } = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_bijections_on_three_element_set_l3943_394359


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l3943_394308

-- Define the polynomial
def p (x : ℂ) : ℂ := x^4 + 2*x^3 + 6*x^2 + 34*x + 49

-- State the theorem
theorem pure_imaginary_solutions :
  p (Complex.I * Real.sqrt 17) = 0 ∧ p (-Complex.I * Real.sqrt 17) = 0 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l3943_394308


namespace NUMINAMATH_CALUDE_num_expressions_correct_l3943_394322

/-- The number of algebraically different expressions obtained by placing parentheses in a₁ / a₂ / ... / aₙ -/
def num_expressions (n : ℕ) : ℕ :=
  if n ≥ 2 then 2^(n-2) else 0

/-- Theorem stating that for n ≥ 2, the number of algebraically different expressions
    obtained by placing parentheses in a₁ / a₂ / ... / aₙ is equal to 2^(n-2) -/
theorem num_expressions_correct (n : ℕ) (h : n ≥ 2) :
  num_expressions n = 2^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_num_expressions_correct_l3943_394322


namespace NUMINAMATH_CALUDE_percentage_of_b_grades_l3943_394300

def grading_scale : List (String × Nat × Nat) :=
  [("A", 93, 100), ("B", 87, 92), ("C", 78, 86), ("D", 70, 77), ("F", 0, 69)]

def grades : List Nat :=
  [88, 66, 92, 83, 90, 99, 74, 78, 85, 72, 95, 86, 79, 68, 81, 64, 87, 91, 76, 89]

def is_grade_b (grade : Nat) : Bool :=
  87 ≤ grade ∧ grade ≤ 92

def count_grade_b (grades : List Nat) : Nat :=
  grades.filter is_grade_b |>.length

theorem percentage_of_b_grades :
  (count_grade_b grades : Rat) / grades.length * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_b_grades_l3943_394300


namespace NUMINAMATH_CALUDE_cube_side_length_is_one_l3943_394302

/-- The surface area of a cuboid formed by joining two cubes with side length s is 10 -/
def cuboid_surface_area (s : ℝ) : ℝ := 10 * s^2

/-- Theorem: If two cubes with side length s are joined to form a cuboid with surface area 10, then s = 1 -/
theorem cube_side_length_is_one :
  ∃ (s : ℝ), s > 0 ∧ cuboid_surface_area s = 10 → s = 1 :=
by sorry

end NUMINAMATH_CALUDE_cube_side_length_is_one_l3943_394302


namespace NUMINAMATH_CALUDE_system_solution_l3943_394314

theorem system_solution :
  let solutions : List (ℤ × ℤ) := [(-5, -3), (-3, -5), (3, 5), (5, 3)]
  ∀ x y : ℤ, (x^2 - x*y + y^2 = 19 ∧ x^4 + x^2*y^2 + y^4 = 931) ↔ (x, y) ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3943_394314


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l3943_394357

theorem quadratic_roots_sum_and_product (m n : ℝ) : 
  (m^2 - 4*m = 12) → (n^2 - 4*n = 12) → m + n + m*n = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l3943_394357


namespace NUMINAMATH_CALUDE_sequence_property_l3943_394392

def is_increasing_positive_integer_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem sequence_property (a : ℕ → ℕ) 
  (h1 : is_increasing_positive_integer_sequence a) 
  (h2 : ∀ n : ℕ, a (n + 2) = a (n + 1) + 2 * a n) 
  (h3 : a 5 = 52) : 
  a 7 = 212 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l3943_394392


namespace NUMINAMATH_CALUDE_runner_time_difference_l3943_394389

theorem runner_time_difference (total_distance : ℝ) (half_distance : ℝ) (second_half_time : ℝ) :
  total_distance = 40 →
  half_distance = total_distance / 2 →
  second_half_time = 10 →
  ∃ (initial_speed : ℝ),
    initial_speed > 0 ∧
    half_distance / initial_speed + half_distance / (initial_speed / 2) = second_half_time + half_distance / initial_speed ∧
    second_half_time - half_distance / initial_speed = 5 :=
by sorry

end NUMINAMATH_CALUDE_runner_time_difference_l3943_394389


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l3943_394390

/-- The coordinates of the foci of the ellipse 25x^2 + 16y^2 = 1 are (0, 3/20) and (0, -3/20) -/
theorem ellipse_foci_coordinates :
  let ellipse := {(x, y) : ℝ × ℝ | 25 * x^2 + 16 * y^2 = 1}
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁ ∈ ellipse ∧ f₂ ∈ ellipse) ∧ 
    (∀ p ∈ ellipse, (dist p f₁) + (dist p f₂) = (dist (1/5, 0) (-1/5, 0))) ∧
    f₁ = (0, 3/20) ∧ f₂ = (0, -3/20) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l3943_394390


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3943_394333

theorem max_value_on_circle (x y : ℝ) : 
  x^2 + y^2 = 20*x + 24*y + 26 → (5*x + 3*y ≤ 73) ∧ ∃ x y, x^2 + y^2 = 20*x + 24*y + 26 ∧ 5*x + 3*y = 73 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3943_394333


namespace NUMINAMATH_CALUDE_smallest_multiples_product_l3943_394371

theorem smallest_multiples_product (c d : ℕ) : 
  (c ≥ 10 ∧ c < 100 ∧ c % 7 = 0 ∧ ∀ x, x ≥ 10 ∧ x < 100 ∧ x % 7 = 0 → c ≤ x) →
  (d ≥ 100 ∧ d < 1000 ∧ d % 5 = 0 ∧ ∀ y, y ≥ 100 ∧ y < 1000 ∧ y % 5 = 0 → d ≤ y) →
  (c * d) - 100 = 1300 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiples_product_l3943_394371


namespace NUMINAMATH_CALUDE_salary_percent_increase_l3943_394366

theorem salary_percent_increase 
  (original_salary new_salary increase : ℝ) 
  (h1 : new_salary = 90000)
  (h2 : increase = 25000)
  (h3 : original_salary = new_salary - increase) :
  (increase / original_salary) * 100 = (25000 / (90000 - 25000)) * 100 := by
sorry

end NUMINAMATH_CALUDE_salary_percent_increase_l3943_394366


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l3943_394383

/-- Given a constant distance and two different walking rates, where one rate
    results in a 14-minute journey and the other in a 12-minute journey,
    prove that the ratio of the faster rate to the slower rate is 7/6. -/
theorem walking_rate_ratio (distance : ℝ) (usual_rate new_rate : ℝ) :
  distance > 0 →
  usual_rate > 0 →
  new_rate > 0 →
  distance = usual_rate * 14 →
  distance = new_rate * 12 →
  new_rate / usual_rate = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l3943_394383


namespace NUMINAMATH_CALUDE_wrapping_and_ribbons_fractions_l3943_394315

/-- Given a roll of wrapping paper, prove the fractions used for wrapping and ribbons on each present -/
theorem wrapping_and_ribbons_fractions
  (total_wrap : ℚ) -- Total fraction of roll used for wrapping
  (total_ribbon : ℚ) -- Total fraction of roll used for ribbons
  (num_presents : ℕ) -- Number of presents
  (h1 : total_wrap = 2/5) -- Condition: 2/5 of roll used for wrapping
  (h2 : total_ribbon = 1/5) -- Condition: 1/5 of roll used for ribbons
  (h3 : num_presents = 5) -- Condition: 5 presents
  : (total_wrap / num_presents = 2/25) ∧ (total_ribbon / num_presents = 1/25) := by
  sorry


end NUMINAMATH_CALUDE_wrapping_and_ribbons_fractions_l3943_394315


namespace NUMINAMATH_CALUDE_gcd_problem_l3943_394304

theorem gcd_problem : ∃ b : ℕ+, Nat.gcd (20 * b) (18 * 24) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3943_394304


namespace NUMINAMATH_CALUDE_two_card_picks_from_two_decks_l3943_394301

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)

/-- Represents the total collection of cards from two shuffled decks -/
def ShuffledDecks (d : Deck) : Nat :=
  2 * d.cards

/-- The number of ways to pick two different cards from shuffled decks -/
def PickTwoCards (total : Nat) : Nat :=
  total * (total - 1)

theorem two_card_picks_from_two_decks :
  let standard_deck : Deck := { cards := 52, suits := 4, cards_per_suit := 13 }
  let shuffled_total := ShuffledDecks standard_deck
  PickTwoCards shuffled_total = 10692 := by
  sorry

end NUMINAMATH_CALUDE_two_card_picks_from_two_decks_l3943_394301


namespace NUMINAMATH_CALUDE_insufficient_info_for_unique_height_l3943_394335

/-- Represents the relationship between height and shadow length -/
noncomputable def height_shadow_relation (a b : ℝ) (s : ℝ) : ℝ :=
  a * Real.sqrt s + b * s

theorem insufficient_info_for_unique_height :
  ∀ (a₁ b₁ a₂ b₂ : ℝ),
  (height_shadow_relation a₁ b₁ 40.25 = 17.5) →
  (height_shadow_relation a₂ b₂ 40.25 = 17.5) →
  (a₁ ≠ a₂ ∨ b₁ ≠ b₂) →
  ∃ (h₁ h₂ : ℝ), 
    h₁ ≠ h₂ ∧ 
    height_shadow_relation a₁ b₁ 28.75 = h₁ ∧
    height_shadow_relation a₂ b₂ 28.75 = h₂ :=
by sorry

end NUMINAMATH_CALUDE_insufficient_info_for_unique_height_l3943_394335


namespace NUMINAMATH_CALUDE_harry_hours_formula_l3943_394350

/-- Represents the payment structure and hours worked for Harry and James -/
structure PaymentSystem where
  x : ℝ  -- Base hourly rate
  S : ℝ  -- Number of hours James is paid at regular rate
  H : ℝ  -- Number of hours Harry worked

/-- Calculates Harry's pay for the week -/
def harry_pay (p : PaymentSystem) : ℝ :=
  18 * p.x + 1.5 * p.x * (p.H - 18)

/-- Calculates James' pay for the week -/
def james_pay (p : PaymentSystem) : ℝ :=
  p.S * p.x + 2 * p.x * (41 - p.S)

/-- Theorem stating the relationship between Harry's hours worked and James' regular hours -/
theorem harry_hours_formula (p : PaymentSystem) :
  harry_pay p = james_pay p →
  p.H = (91 - 3 * p.S) / 1.5 := by
  sorry

end NUMINAMATH_CALUDE_harry_hours_formula_l3943_394350


namespace NUMINAMATH_CALUDE_maxwell_twice_sister_age_sister_current_age_l3943_394387

/-- Maxwell's current age -/
def maxwell_age : ℕ := 6

/-- Maxwell's sister's current age -/
def sister_age : ℕ := 2

/-- In 2 years, Maxwell will be twice his sister's age -/
theorem maxwell_twice_sister_age : 
  maxwell_age + 2 = 2 * (sister_age + 2) := by sorry

/-- Proof that Maxwell's sister is currently 2 years old -/
theorem sister_current_age : sister_age = 2 := by sorry

end NUMINAMATH_CALUDE_maxwell_twice_sister_age_sister_current_age_l3943_394387


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l3943_394321

theorem complex_expression_simplification (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l3943_394321


namespace NUMINAMATH_CALUDE_ball_attendees_l3943_394311

theorem ball_attendees :
  ∀ n m : ℕ,
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_attendees_l3943_394311


namespace NUMINAMATH_CALUDE_f_properties_l3943_394354

noncomputable def f (x : ℝ) : ℝ := (1 / (2^x - 1) + 1/2) * x^3

theorem f_properties :
  (∀ x, x ≠ 0 → f x ≠ 0) ∧
  (∀ x, x ≠ 0 → f (-x) = f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3943_394354
